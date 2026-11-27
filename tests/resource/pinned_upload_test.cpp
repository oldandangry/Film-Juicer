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
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#include "juicer_cuda_api.h"
#include "RenderValues.h"

#include "allocation_fault.h"
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
            EXPECT_TRUE(JuicerCuda::purge_pinned_upload_staging_for_context(key.deviceId, key.contextOpaque, 1, PinnedUploadPurgeDisposition::NormalRetire, failure)) << failure.diagnostic;
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

    TEST_F(PinnedUpload, MetadataAllocationFailurePrecedesAnyCudaHandleAcquisition) {
        const std::array source{0.25F, 0.5F};
        // Initialize the policy before fault injection and force a new block for this epoch.
        ASSERT_TRUE(JuicerCuda::purge_pinned_upload_staging_for_context(
            key.deviceId, key.contextOpaque, 1, PinnedUploadPurgeDisposition::NormalRetire, failure))
            << failure.diagnostic;
        const auto initialCharge = JuicerCuda::ResourceManager::global_state().pinnedStagingBytes.load();
        const auto before = calls.uploadAllocations;
        JuicerTest::fail_allocation_after(0);
        const bool uploaded = upload(source.data(), sizeof(source));
        EXPECT_TRUE(JuicerTest::take_allocation_fault());
        EXPECT_FALSE(uploaded);
        EXPECT_EQ(failure.diagnostic, "test source: pinned staging pool_insert_failed");
        EXPECT_EQ(calls.uploadAllocations, before);
        EXPECT_EQ(JuicerCuda::ResourceManager::global_state().pinnedStagingBytes.load(), initialCharge);
        EXPECT_EQ(calls.copies, 0U);
        EXPECT_TRUE(upload(source.data(), sizeof(source))) << failure.diagnostic;
        EXPECT_EQ(calls.uploadAllocations, before + 1);
    }

    TEST_F(PinnedUpload, UnpublishedBlockRetainsBothHandlesAndCapacityUntilRetry) {
        const std::array source{0.25F, 0.5F};
        const auto charged = [] {
            return JuicerCuda::ResourceManager::global_state().pinnedStagingBytes.load();
        };
        const auto initialCharge = charged();
        const auto eventDestroys = calls.successfulEventDestroys;
        const auto hostFrees = calls.successfulHostFrees;
        // Return a real event along with a synthetic creation failure so both
        // of the unpublished handles are present on the rollback path.
        calls.nextCreateUploadEvent = cudaErrorInvalidResourceHandle;
        calls.nextDestroyEvent = cudaErrorContextIsDestroyed;
        calls.nextFreeHost = cudaErrorInvalidValue;
        EXPECT_FALSE(upload(source.data(), sizeof(source)));
        EXPECT_EQ(failure.status.native_code, cudaErrorInvalidResourceHandle);
        EXPECT_TRUE(JuicerCuda::context_loss(failure));
        EXPECT_EQ(charged(), initialCharge + sizeof(source));
        EXPECT_EQ(calls.successfulEventDestroys, eventDestroys);
        EXPECT_EQ(calls.successfulHostFrees, hostFrees);
        EXPECT_EQ(calls.nextDestroyEvent, cudaSuccess);
        EXPECT_EQ(calls.nextFreeHost, cudaSuccess);
        EXPECT_EQ(calls.copies, 0U);

        ASSERT_TRUE(JuicerCuda::purge_pinned_upload_staging_for_context(
            key.deviceId, key.contextOpaque, 1, PinnedUploadPurgeDisposition::NormalRetire, failure))
            << failure.diagnostic;
        EXPECT_EQ(charged(), initialCharge);
        EXPECT_EQ(calls.successfulEventDestroys, eventDestroys + 1);
        EXPECT_EQ(calls.successfulHostFrees, hostFrees + 1);
        ASSERT_TRUE(JuicerCuda::purge_pinned_upload_staging_for_context(
            key.deviceId, key.contextOpaque, 1, PinnedUploadPurgeDisposition::NormalRetire, failure));
        EXPECT_EQ(calls.successfulEventDestroys, eventDestroys + 1);
        EXPECT_EQ(calls.successfulHostFrees, hostFrees + 1);
    }

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
                                                       manifest.at("presets").at(JUICER_TEST_PRESET_NAME).at("cases").get<std::string>()));
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

    TEST_F(PinnedUpload, ForeignContextCannotDisposeOfTheProducingEpoch) {
        constexpr std::array<std::uint8_t, 4> source{0x31, 0x52, 0x73, 0x94};
        ASSERT_TRUE(upload(source.data(), source.size())) << failure.diagnostic;
        require_cuda(cudaStreamSynchronize(stream));
        auto* const staging = calls.lastStaging;
        auto* const completion = calls.lastEvent;
        CUcontext foreign = nullptr;
        ASSERT_EQ(cuCtxCreate(&foreign, nullptr, 0, key.deviceId), CUDA_SUCCESS);
        const bool retired = JuicerCuda::purge_pinned_upload_staging_for_context(
            key.deviceId, key.contextOpaque, 1, PinnedUploadPurgeDisposition::NormalRetire, failure);
        EXPECT_FALSE(retired);
        EXPECT_EQ(failure.status.category, FJ_STATUS_PREPARATION_FAILURE);
        EXPECT_EQ(failure.status.api, FJ_API_NONE);
        EXPECT_EQ(cuCtxDestroy(foreign), CUDA_SUCCESS);
        ASSERT_EQ(cuCtxSetCurrent(reinterpret_cast<CUcontext>(key.contextOpaque)), CUDA_SUCCESS);
        ASSERT_TRUE(upload(source.data(), source.size())) << failure.diagnostic;
        EXPECT_EQ(calls.lastStaging, staging);
        EXPECT_EQ(calls.lastEvent, completion);
        require_cuda(cudaStreamSynchronize(stream));
        EXPECT_EQ(read<std::uint8_t>(destination, source.size()), (std::vector<std::uint8_t>{source.begin(), source.end()}));
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
        EXPECT_TRUE(JuicerCuda::purge_pinned_upload_staging_for_context(key.deviceId, key.contextOpaque, 1, PinnedUploadPurgeDisposition::NormalRetire, failure)) << failure.diagnostic;
    }

    INSTANTIATE_TEST_SUITE_P(RecordFailure, PinnedUploadRecordFailure, testing::Values(cudaErrorInvalidResourceHandle, cudaErrorContextIsDestroyed));

    TEST_F(PinnedUpload, DirColdCallbackInstallsBothMatricesAndStagesWeights) {
        const std::filesystem::path path(JUICER_DIR_BOUNDARY_FIXTURE);
        const auto context = Json::parse(std::ifstream(path));
        ASSERT_EQ(context.at("preset"), JUICER_TEST_PRESET_NAME);
        const auto fixture = Json::parse(std::ifstream(path.parent_path() / context.at("payload").get<std::string>()));
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

        Spektrafilm::DirGaussianComponentPlan component{};
        component.sigmaPixels = 4.0F;
        component.weight = 1.F;
        component.radius = 12;
        component.referenceOperator = Spektrafilm::DirReferenceOperator::SpektrafilmLargeYvvReflect;
        component.iir.feedforward = 0.5;
        component.iir.feedback = {0.5, 0.0, 0.0};
        component.iir.normalizationDenominator = 2.0;
        component.iir.feedforwardNumerator = 1.0;
        component.iir.feedbackNumerators = {1.0, 0.0, 0.0};
        component.boundaryTruncationAccuracy = 1e-11;
        component.boundaryCertificationTolerance = 1e-6;
        component.boundaryDerivationVersion = 1;
        component.boundaryIdentity = row->at("key").get<std::uint64_t>();

        using Boundary = JuicerCuda::Resources::DeviceSpatialDirBoundary;
        Boundary& boundary = resources->spatialDirBoundaries[0];
        struct Preparation {
            FjDirBoundaryView view{};
            bool fail = true;
            unsigned calls = 0;
        } preparation{};
        auto candidate = std::make_unique<double[]>(expected.size());
        std::ranges::transform(expected, candidate.get(), [](std::uint64_t word) {
            return std::bit_cast<double>(word);
        });
        preparation.view = {
            candidate.get(), expected.size(), component.boundaryIdentity, 9, 6, static_cast<int>(horizontal.at("weights").size() / 3), static_cast<int>(vertical.at("weights").size() / 3), horizontal.at("terminal_size").get<int>(), vertical.at("terminal_size").get<int>(), {}, {}};
        for (std::size_t i = 0; i < 9; ++i) {
            preparation.view.horizontal_terminal[i] = std::bit_cast<double>(horizontal.at("terminal").at(i).get<std::uint64_t>());
            preparation.view.vertical_terminal[i] = std::bit_cast<double>(vertical.at("terminal").at(i).get<std::uint64_t>());
        }
        const FjHostPreparation host{&preparation, nullptr, [](void* user, const FjDirBoundaryInput* input, FjDirBoundaryView* out, FjDirBoundaryFailure* detail, FjErrorBuffer*) -> FjStatus {
                                         auto& data = *static_cast<Preparation*>(user);
                                         ++data.calls;
                                         EXPECT_EQ(input->component.boundary_identity, data.view.identity);
                                         EXPECT_EQ(input->width, data.view.width);
                                         EXPECT_EQ(input->height, data.view.height);
                                         if (data.fail) {
                                             detail->reason = FJ_DIR_BOUNDARY_ALLOCATION;
                                             return {FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0};
                                         }
                                         *out = data.view;
                                         return {FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
                                     }};
        const auto ensure = [&] {
            return JuicerCuda::ensure_spatial_dir_boundary(*resources, boundary, component, 9, 6, stream, failure, &host);
        };
        EXPECT_FALSE(ensure());
        EXPECT_EQ(failure.status.category, FJ_STATUS_ALLOCATION_FAILURE);
        EXPECT_EQ(boundary.weights, nullptr);
        preparation.fail = false;

        std::optional<UploadGate> gate;
        calls.beforeCopyUser = &gate;
        calls.beforeCopy = [](cudaStream_t stream, void* target) {
            static_cast<std::optional<UploadGate>*>(target)->emplace(stream);
        };
        ASSERT_TRUE(ensure()) << failure.diagnostic;
        ASSERT_TRUE(gate);
        EXPECT_EQ(boundary.horizontalTerminalSize, preparation.view.horizontal_terminal_size);
        EXPECT_EQ(boundary.verticalTerminalSize, preparation.view.vertical_terminal_size);
        for (std::size_t i = 0; i < 9; ++i) {
            EXPECT_EQ(std::bit_cast<std::uint64_t>(boundary.horizontalTerminalMatrix[i]), horizontal.at("terminal").at(i).get<std::uint64_t>());
            EXPECT_EQ(std::bit_cast<std::uint64_t>(boundary.verticalTerminalMatrix[i]), vertical.at("terminal").at(i).get<std::uint64_t>());
        }
        auto* const first = boundary.weights;
        auto* const pending = calls.lastEvent;
        ASSERT_EQ(cudaEventQuery(pending), cudaErrorNotReady);
        EXPECT_TRUE(ensure()) << failure.diagnostic;
        EXPECT_EQ(preparation.calls, 2U);
        EXPECT_EQ(boundary.weights, first);
        candidate.reset();
        ASSERT_EQ(cudaEventQuery(pending), cudaErrorNotReady);
        gate->release();
        require_cuda(cudaEventSynchronize(pending));
        EXPECT_EQ(read<std::uint64_t>(boundary.weights, expected.size()), expected);
        EXPECT_FALSE(gate->timed_out());

        // Begin the old boundary's last read after the replacement allocates.
        // This keeps it pending through installation even with a synchronous
        // device allocator. The retirement event follows that queued read.
        gate.reset();
        candidate = std::make_unique<double[]>(expected.size());
        std::ranges::transform(expected, candidate.get(), [](std::uint64_t word) {
            return std::bit_cast<double>(word);
        });
        component.boundaryIdentity ^= 1U;
        preparation.view.identity = component.boundaryIdentity;
        preparation.view.weights = candidate.get();
        struct LastRead {
            std::optional<UploadGate>& gate;
            const double* oldWeights;
            void* destination;
            std::size_t bytes;
        } lastRead{gate, first, destination, expected.size() * sizeof(double)};
        calls.beforeCopyUser = &lastRead;
        calls.beforeCopy = [](cudaStream_t stream, void* user) {
            auto& read = *static_cast<LastRead*>(user);
            read.gate.emplace(stream);
            EXPECT_EQ(cudaMemcpyAsync(read.destination, read.oldWeights, read.bytes, cudaMemcpyDeviceToDevice, stream), cudaSuccess);
        };
        ASSERT_TRUE(ensure()) << failure.diagnostic;
        ASSERT_TRUE(gate);
        EXPECT_NE(boundary.weights, first);
        EXPECT_EQ(boundary.identity, component.boundaryIdentity);
        EXPECT_EQ(preparation.calls, 3U);
        EXPECT_EQ(cudaStreamQuery(stream), cudaErrorNotReady);
        candidate.reset();
        gate->release();
        require_cuda(cudaStreamSynchronize(stream));
        EXPECT_FALSE(gate->timed_out());
        EXPECT_EQ(read<std::uint64_t>(destination, expected.size()), expected);
        EXPECT_EQ(read<std::uint64_t>(boundary.weights, expected.size()), expected);
    }
} // namespace
