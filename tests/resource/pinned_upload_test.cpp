#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "Cuda/JuicerCudaHostViews.h"
#include "Cuda/JuicerCudaResourceOperations.h"
#include "juicer_dir_boundary_api.h"

#include "observed_operations.h"

using JuicerTest::calls;

namespace {
    using Json = nlohmann::json;
    using JuicerCuda::PinnedUploadPurgeDisposition;
    constexpr std::size_t kCap = std::size_t{128} * 1024u * 1024u;

    void require_cuda(cudaError_t error) {
        if (error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }
    }

    class UploadGate final {
    public:
        explicit UploadGate(cudaStream_t stream) : _stream(stream) {
            require_cuda(cudaLaunchHostFunc(stream, &UploadGate::wait, this));
        }
        ~UploadGate() {
            release();
            // The fixture checks CUDA completion; teardown must not throw while
            // ensuring that the callback no longer refers to this gate.
            (void)cudaStreamSynchronize(_stream);
        }
        void release() noexcept {
            try {
                {
                    std::lock_guard lock(_mutex);
                    _released = true;
                }
                _condition.notify_all();
            } catch (...) {
                _timedOut.store(true);
            }
        }
        bool timed_out() const {
            return _timedOut.load();
        }

    private:
        static void CUDART_CB wait(void* pointer) {
            auto& gate = *static_cast<UploadGate*>(pointer);
            try {
                std::unique_lock lock(gate._mutex);
                if (!gate._condition.wait_for(lock, std::chrono::seconds(10), [&] {
                        return gate._released;
                    })) {
                    gate._timedOut.store(true);
                }
            } catch (...) {
                gate._timedOut.store(true);
            }
        }
        std::mutex _mutex;
        std::condition_variable _condition;
        bool _released = false;
        cudaStream_t _stream = nullptr;
        std::atomic<bool> _timedOut{false};
    };


    class PinnedUpload : public testing::Test {
    protected:
        void SetUp() override {
            calls = {};
            require_cuda(cudaSetDevice(0));
            require_cuda(cudaFree(nullptr));
            CUcontext context = nullptr;
            ASSERT_EQ(cuCtxGetCurrent(&context), CUDA_SUCCESS);
            ASSERT_NE(context, nullptr);
            key = {0, context};
            std::string diagnostic;
            ledger = JuicerCuda::DeviceAllocationLedger::create({0, 8192}, diagnostic);
            ASSERT_TRUE(ledger && ledger->bind_or_validate_cap(8192, diagnostic)) << diagnostic;
            resources = std::make_unique<JuicerCuda::Resources>(key, 1, ledger);
            require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            require_cuda(cudaMalloc(&destination, kCap));
        }

        void TearDown() override {
            if (stream) {
                EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
            }
            resources.reset();
            if (ledger) {
                EXPECT_EQ(ledger->snapshot().chargedBytes, 0);
            }
            JuicerCuda::purge_pinned_upload_staging_for_context(key.deviceId, key.contextOpaque, PinnedUploadPurgeDisposition::NormalRetire);
            if (destination) {
                EXPECT_EQ(cudaFree(destination), cudaSuccess);
            }
            if (stream) {
                EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
            }
        }

        bool upload(const void* source, std::size_t bytes) {
            return JuicerCuda::enqueue_host_to_device_copy(*resources,
                                                           {"resource_test", "test source"},
                                                           destination,
                                                           source,
                                                           bytes,
                                                           stream,
                                                           failure);
        }

        template <typename T>
        std::vector<T> read(const void* pointer, std::size_t count) {
            std::vector<T> output(count);
            require_cuda(cudaMemcpy(output.data(), pointer, output.size() * sizeof(T), cudaMemcpyDeviceToHost));
            return output;
        }

        JuicerCuda::ResourceManager::DeviceContextKey key{};
        std::shared_ptr<JuicerCuda::DeviceAllocationLedger> ledger;
        std::unique_ptr<JuicerCuda::Resources> resources;
        cudaStream_t stream = nullptr;
        void* destination = nullptr;
        JuicerCuda::Failure failure;
    };

    struct PrintUploadInput {
        std::array<float, 2> axis{-2.0f, 0.0f};
        std::array<std::array<float, 3>, 2> density{{{0.25f, 0.5f, 0.75f}, {1.0f, 1.25f, 1.5f}}};
        std::array<std::array<float, 3>, Spectral::kNumSamples> spectra{};
        std::array<std::array<float, 3>, Spectral::kNumSamples> sensitivities{};
        std::array<float, Spectral::kNumSamples> base{};
        JuicerCuda::PrintResourceInput input;

        PrintUploadInput() {
            base.fill(1.0f);
            input.descriptors.profileTablesHash = 1;
            input.descriptors.filmDensityTablesHash = 1;
            input.descriptors.mainIlluminantHash = 1;
            input.filmChannelDensityCmy = JuicerCuda::ThreeChannelSamplesView(spectra);
            input.filmBaseDensity = base;
            input.printLogExposure = axis;
            input.printDensityCurvesCmy = JuicerCuda::ThreeChannelSamplesView(density);
            input.printSensitivityCmy = JuicerCuda::ThreeChannelSamplesView(sensitivities);
        }

        void change_curve(std::uint64_t hash) {
            axis = {-1.5f, 0.5f};
            density[0][0] = 0.375f;
            density[1][0] = 1.125f;
            input.descriptors.profileTablesHash = hash;
        }

        std::array<float, 2> cyan_samples() const {
            return {density[0][0], density[1][0]};
        }
    };


    std::array<float, 81> preflash_light(const Json& row) {
        const auto words = row.at("filtered_illuminant").get<std::array<std::uint32_t, 81>>();
        std::array<float, 81> light{};
        std::ranges::transform(words, light.begin(), [](std::uint32_t word) {
            return std::bit_cast<float>(word);
        });
        return light;
    }

    TEST_F(PinnedUpload, PrintResidencyAndFailedPreflashOverwrite) {
        const std::filesystem::path directory(JUICER_PREFLASH_FIXTURES);
        const auto manifest = Json::parse(std::ifstream(directory / "manifest.json"));
        const auto fixture = Json::parse(std::ifstream(directory /
                                                       manifest.at("presets").at(JUICER_TEST_PRESET_NAME).at("native").get<std::string>()));
        const auto& cases = fixture.at("cases");
        ASSERT_EQ(cases.at(0).at("id"), "preflash-unit");
        ASSERT_EQ(cases.at(2).at("id"), "preflash-pattern");
        const auto lightA = preflash_light(cases.at(0));
        const auto lightB = preflash_light(cases.at(2));
        ASSERT_NE(lightA, lightB);
        PrintUploadInput print;
        print.input.mainIlluminant = lightA;
        print.input.descriptors.preflashActive = true;
        print.input.descriptors.preflashIlluminantHash = 1;
        print.input.preflashIlluminant = lightA;
        const auto prepare = [&] {
            return JuicerCuda::prepare_print_resources(*resources, print.input, stream, failure);
        };
        ASSERT_TRUE(prepare()) << failure.diagnostic;
        require_cuda(cudaStreamSynchronize(stream));
        const auto* mainSlot = resources->printIllumFiltered;
        const auto filmHash = resources->printFilmDensityTablesDescriptorHash;
        const auto* preflashSlot = resources->printPreflashIllumFiltered;
        const auto* paperSlot = resources->printDcC.x;
        const auto charge = ledger->snapshot().chargedBytes;
        ASSERT_GT(charge, 0U);
        EXPECT_EQ(resources->printMainIlluminantDescriptorHash, 1U);
        EXPECT_EQ(resources->printPreflashIlluminantDescriptorHash, 1U);
        ASSERT_EQ(read<float>(preflashSlot, lightA.size()), std::vector<float>(lightA.begin(), lightA.end()));

        calls.copies = 0;
        calls.copyBytes = 0;
        ASSERT_TRUE(prepare()) << failure.diagnostic;
        EXPECT_EQ(calls.copies, 0U);
        EXPECT_EQ(calls.copyBytes, 0U);

        print.change_curve(2);
        ASSERT_TRUE(prepare()) << failure.diagnostic;
        require_cuda(cudaStreamSynchronize(stream));
        EXPECT_EQ(calls.copies, 7U);
        EXPECT_EQ(calls.copyBytes, sizeof(float) * (4 * print.axis.size() + 3 * print.sensitivities.size()));
        EXPECT_EQ(resources->printProfileTablesDescriptorHash, 2U);
        EXPECT_EQ(resources->printIllumFiltered, mainSlot);
        EXPECT_EQ(resources->printFilmDensityTablesDescriptorHash, filmHash);
        EXPECT_EQ(resources->printPreflashIllumFiltered, preflashSlot);
        EXPECT_EQ(ledger->snapshot().chargedBytes, charge);
        ASSERT_EQ(resources->printDcC.x, paperSlot);
        EXPECT_EQ(read<float>(paperSlot, print.axis.size()), std::vector<float>(print.axis.begin(), print.axis.end()));
        const auto cyan = print.cyan_samples();
        EXPECT_EQ(read<float>(resources->printDcC.y, cyan.size()), std::vector<float>(cyan.begin(), cyan.end()));

        for (const auto key : {2U, 1U}) {
            const auto& light = key == 2U ? lightB : lightA;
            print.input.descriptors.mainIlluminantHash = key;
            print.input.mainIlluminant = light;
            calls.copies = 0;
            calls.copyBytes = 0;
            ASSERT_TRUE(prepare()) << failure.diagnostic;
            require_cuda(cudaStreamSynchronize(stream));
            EXPECT_EQ(calls.copies, 1U);
            EXPECT_EQ(calls.copyBytes, sizeof(light));
            EXPECT_EQ(resources->printMainIlluminantDescriptorHash, key);
            ASSERT_EQ(resources->printIllumFiltered, mainSlot);
            EXPECT_EQ(read<float>(mainSlot, light.size()), std::vector<float>(light.begin(), light.end()));
            EXPECT_EQ(resources->printPreflashIllumFiltered, preflashSlot);
            EXPECT_EQ(resources->printPreflashIlluminantDescriptorHash, 1U);
            EXPECT_EQ(resources->printFilmDensityTablesDescriptorHash, filmHash);
            EXPECT_EQ(resources->printProfileTablesDescriptorHash, 2U);
            EXPECT_EQ(ledger->snapshot().chargedBytes, charge);
            ASSERT_TRUE(prepare()) << failure.diagnostic;
            EXPECT_EQ(calls.copies, 1U);
        }

        print.input.descriptors.preflashIlluminantHash = 2;
        print.input.preflashIlluminant = lightB;
        calls.copies = 0;
        calls.copyBytes = 0;
        calls.nextCopy = cudaErrorInvalidValue;
        EXPECT_FALSE(prepare());
        EXPECT_EQ(failure.status.category, FJ_STATUS_CUDA_FAILURE);
        EXPECT_EQ(failure.status.api, FJ_API_CUDA_RUNTIME);
        EXPECT_EQ(failure.status.native_code, cudaErrorInvalidValue);
        EXPECT_EQ(resources->printPreflashIlluminantDescriptorHash, 0U);
        EXPECT_EQ(calls.copies, 0U);
        EXPECT_EQ(calls.copyBytes, 0U);
        ASSERT_TRUE(prepare()) << failure.diagnostic;
        require_cuda(cudaStreamSynchronize(stream));
        EXPECT_EQ(calls.copies, 1U);
        EXPECT_EQ(calls.copyBytes, sizeof(lightB));
        EXPECT_EQ(resources->printPreflashIlluminantDescriptorHash, 2U);
        EXPECT_EQ(resources->printPreflashIllumFiltered, preflashSlot);
        EXPECT_EQ(read<float>(preflashSlot, lightB.size()), std::vector<float>(lightB.begin(), lightB.end()));
        EXPECT_EQ(ledger->snapshot().chargedBytes, charge);
        ASSERT_TRUE(prepare()) << failure.diagnostic;
        EXPECT_EQ(calls.copies, 1U);
        EXPECT_EQ(calls.copyBytes, sizeof(lightB));

        print.input.descriptors.preflashActive = false;
        print.input.descriptors.preflashIlluminantHash = 0;
        print.input.preflashIlluminant = {};
        calls.copies = 0;
        calls.copyBytes = 0;
        ASSERT_TRUE(prepare()) << failure.diagnostic;
        EXPECT_EQ(calls.copies, 0U);
        EXPECT_EQ(calls.copyBytes, 0U);
        EXPECT_EQ(resources->printPreflashIllumFiltered, nullptr);
        EXPECT_EQ(resources->printPreflashIlluminantDescriptorHash, 0U);
        EXPECT_EQ(resources->printIllumFiltered, mainSlot);
        print.input.descriptors.preflashActive = true;
        print.input.descriptors.preflashIlluminantHash = 2;
        print.input.preflashIlluminant = lightB;
        ASSERT_TRUE(prepare()) << failure.diagnostic;
        require_cuda(cudaStreamSynchronize(stream));
        ASSERT_NE(resources->printPreflashIllumFiltered, nullptr);
        EXPECT_EQ(calls.copies, 1U);
        EXPECT_EQ(calls.copyBytes, sizeof(lightB));
        EXPECT_EQ(read<float>(resources->printPreflashIllumFiltered, lightB.size()), std::vector<float>(lightB.begin(), lightB.end()));
        ASSERT_TRUE(prepare()) << failure.diagnostic;
        EXPECT_EQ(calls.copies, 1U);
        EXPECT_EQ(resources->printIllumFiltered, mainSlot);
    }

    TEST_F(PinnedUpload, ExpiredSourceAndBusyPool) {
        const std::uint8_t byte = 0;
        ASSERT_FALSE(upload(&byte, kCap + 1));
        EXPECT_EQ(failure.status.category, FJ_STATUS_PREPARATION_FAILURE);
        EXPECT_FALSE(JuicerCuda::allocation_capacity_exhausted(failure));
        EXPECT_EQ(calls.copies, 0U);
        std::vector<std::uint8_t> source(kCap, 0x31);
        ASSERT_TRUE(upload(source.data(), source.size())) << failure.diagnostic;
        require_cuda(cudaStreamSynchronize(stream));
        const auto* warmed = calls.lastStaging;
        std::fill(source.begin(), source.end(), 0x69);
        UploadGate gate(stream);
        ASSERT_TRUE(upload(source.data(), source.size())) << failure.diagnostic;
        ASSERT_EQ(calls.lastStaging, warmed);
        auto* const event = calls.lastEvent;
        ASSERT_FALSE(upload(&byte, 1));
        EXPECT_EQ(failure.status.category, FJ_STATUS_PREPARATION_FAILURE);
        EXPECT_FALSE(JuicerCuda::allocation_capacity_exhausted(failure));
        EXPECT_EQ(calls.copies, 2U);
        std::vector<std::uint8_t>().swap(source);
        EXPECT_EQ(cudaEventQuery(event), cudaErrorNotReady);
        gate.release();
        require_cuda(cudaEventSynchronize(event));
        const auto output = read<std::uint8_t>(destination, kCap);
        EXPECT_TRUE(std::ranges::all_of(output, [](std::uint8_t value) {
            return value == 0x69;
        }));
        EXPECT_FALSE(gate.timed_out());
    }

    class PinnedUploadRecordFailure : public PinnedUpload, public testing::WithParamInterface<cudaError_t> {};

    TEST_P(PinnedUploadRecordFailure, QuarantinedUploadCannotReuseAnOlderSuccessfulEvent) {
        constexpr std::size_t bytes = 4096;
        std::vector<std::uint8_t> source(bytes, 0x73);
        ASSERT_TRUE(upload(source.data(), bytes)) << failure.diagnostic;
        require_cuda(cudaStreamSynchronize(stream));
        auto* const event = calls.lastEvent;
        void* const quarantined = calls.lastStaging;
        UploadGate gate(stream);
        calls.nextRecord = GetParam();
        calls.nextSynchronization = cudaErrorInvalidValue;
        EXPECT_FALSE(upload(source.data(), bytes));
        EXPECT_EQ(failure.status.category, GetParam() == cudaErrorContextIsDestroyed ? FJ_STATUS_CONTEXT_LOSS : FJ_STATUS_CUDA_FAILURE);
        EXPECT_EQ(failure.status.api, FJ_API_CUDA_RUNTIME);
        EXPECT_EQ(failure.status.native_code, GetParam());
        EXPECT_EQ(calls.nextSynchronization, cudaSuccess);
        EXPECT_NE(failure.diagnostic.find("cudaStreamSynchronize"), std::string::npos);
        EXPECT_EQ(cudaEventQuery(event), cudaSuccess);
        ASSERT_TRUE(upload(source.data(), bytes)) << failure.diagnostic;
        EXPECT_NE(calls.lastStaging, quarantined);
        gate.release();
        require_cuda(cudaStreamSynchronize(stream));
        EXPECT_EQ(read<std::uint8_t>(destination, bytes), source);
        EXPECT_FALSE(gate.timed_out());
        JuicerCuda::purge_pinned_upload_staging_for_context(key.deviceId, key.contextOpaque, PinnedUploadPurgeDisposition::NormalRetire);
        require_cuda(cudaFreeHost(quarantined));
    }

    INSTANTIATE_TEST_SUITE_P(RecordFailure, PinnedUploadRecordFailure, testing::Values(cudaErrorInvalidResourceHandle, cudaErrorContextIsDestroyed));

    TEST_F(PinnedUpload, DirOwnerExpiresWhileWeightsArePending) {
        const auto fixture = Json::parse(std::ifstream(JUICER_DIR_BOUNDARY_FIXTURE));
        ASSERT_EQ(fixture.at("preset"), JUICER_TEST_PRESET_NAME);
        const auto& rows = fixture.at("cases");
        const auto row = std::ranges::find_if(rows, [](const Json& item) {
            return item.at("input").at("name") == "literal-9x6";
        });
        ASSERT_NE(row, rows.end());
        ASSERT_TRUE(row->at("success").get<bool>());
        const auto& horizontal = row->at("horizontal");
        const auto& vertical = row->at("vertical");
        auto expected = horizontal.at("weights").get<std::vector<std::uint64_t>>();
        const auto verticalWeights = vertical.at("weights").get<std::vector<std::uint64_t>>();
        expected.insert(expected.end(), verticalWeights.begin(), verticalWeights.end());

        const FjDirBoundaryInput input{
            .component = {
                .sigma_px = 4.0F,
                .weight = 1.0F,
                .radius = 12,
                .reference_operator = FJ_DIR_OPERATOR_YVV_REFLECT,
                .feedforward = 0.5,
                .feedback = {0.5, 0.0, 0.0},
                .normalization_denominator = 2.0,
                .feedforward_numerator = 1.0,
                .feedback_numerators = {1.0, 0.0, 0.0},
                .boundary_truncation_accuracy = 1e-11,
                .boundary_certification_tolerance = 1e-6,
                .boundary_derivation_version = 1,
            },
            .width = 9,
            .height = 6,
        };
        const auto release = [](FjDirBoundaryOwner* owner) {
            (void)fj_dir_boundary_release(owner, nullptr);
        };
        FjDirBoundaryOwner* raw = nullptr;
        FjDirBoundaryFailure detail{};
        ASSERT_EQ(fj_dir_boundary_acquire(&input, &raw, &detail, nullptr).category, FJ_STATUS_SUCCESS);
        std::unique_ptr<FjDirBoundaryOwner, decltype(release)> owner(raw, release);
        FjDirBoundaryView view{};
        ASSERT_EQ(fj_dir_boundary_view(owner.get(), &view, &detail, nullptr).category, FJ_STATUS_SUCCESS);
        ASSERT_EQ(view.count, expected.size());
        EXPECT_EQ(view.identity, row->at("key").get<std::uint64_t>());
        for (std::size_t i = 0; i < 9; ++i) {
            EXPECT_EQ(std::bit_cast<std::uint64_t>(view.horizontal_terminal[i]), horizontal.at("terminal").at(i).get<std::uint64_t>());
            EXPECT_EQ(std::bit_cast<std::uint64_t>(view.vertical_terminal[i]), vertical.at("terminal").at(i).get<std::uint64_t>());
        }

        const std::vector<double> warm(view.count, 0.0);
        ASSERT_TRUE(upload(warm.data(), warm.size() * sizeof(double))) << failure.diagnostic;
        require_cuda(cudaStreamSynchronize(stream));
        UploadGate gate(stream);
        ASSERT_TRUE(upload(view.weights, view.count * sizeof(double))) << failure.diagnostic;
        auto* const pending = calls.lastEvent;
        ASSERT_EQ(cudaEventQuery(pending), cudaErrorNotReady);
        ASSERT_EQ(fj_dir_boundary_release(owner.release(), nullptr).category, FJ_STATUS_SUCCESS);
        ASSERT_EQ(cudaEventQuery(pending), cudaErrorNotReady);
        gate.release();
        require_cuda(cudaEventSynchronize(pending));
        EXPECT_EQ(read<std::uint64_t>(destination, expected.size()), expected);
        EXPECT_FALSE(gate.timed_out());
    }
} // namespace
