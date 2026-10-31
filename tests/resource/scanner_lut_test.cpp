#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "Cuda/JuicerCudaHostViews.h"
#include "Cuda/JuicerCudaResources.h"
#include "GamutCompression.h"
#include "../ffi/exposure_fixture.h"
#include "../ffi/host_preparation_fixture.h"
#include "../ffi/juicer_test_api.h"

namespace JuicerCuda::ScannerLutTest {
    enum class Phase : std::uint8_t {
        None,
        AfterPrepare,
        AfterUpload
    };

    class UploadGate final {
    public:
        explicit UploadGate(cudaStream_t stream) noexcept : _stream(stream) {}
        UploadGate(const UploadGate&) = delete;
        UploadGate& operator=(const UploadGate&) = delete;
        ~UploadGate() {
            release();
            if (_started.load())
                (void)cudaStreamSynchronize(_stream);
        }

        bool start() noexcept {
            if (_startCalls.fetch_add(1) != 0) {
                _failed.store(true);
                return false;
            }
            if (cudaLaunchHostFunc(_stream, &UploadGate::block, this) != cudaSuccess) {
                _failed.store(true);
                return false;
            }
            _started.store(true);
            return true;
        }

        void release() noexcept {
            try {
                {
                    std::lock_guard lock(_mutex);
                    _released = true;
                }
                _condition.notify_all();
            } catch (...) {
                _failed.store(true);
            }
        }

        bool started() const noexcept {
            return _started.load();
        }
        bool passed() const noexcept {
            return _passed.load();
        }
        bool failed() const noexcept {
            return _failed.load();
        }
        unsigned start_calls() const noexcept {
            return _startCalls.load();
        }

    private:
        static void CUDART_CB block(void* opaque) noexcept {
            auto& gate = *static_cast<UploadGate*>(opaque);
            try {
                std::unique_lock lock(gate._mutex);
                const bool released = gate._condition.wait_for(
                    lock,
                    std::chrono::seconds(10),
                    [&] {
                        return gate._released;
                    });
                gate._passed.store(released);
            } catch (...) {
                gate._failed.store(true);
            }
        }

        cudaStream_t _stream;
        std::mutex _mutex;
        std::condition_variable _condition;
        bool _released = false;
        std::atomic<bool> _started{false};
        std::atomic<bool> _passed{false};
        std::atomic<bool> _failed{false};
        std::atomic<unsigned> _startCalls{0};
    };

    struct Controls {
        int uploadSlot = -1;
        int retireSlot = -1;
        Phase quarantine = Phase::None;
        UploadGate* uploadGate = nullptr;
        std::size_t uploadVisits = 0;
        std::size_t retireVisits = 0;
        std::size_t gamutVisits = 0;
        bool uploadConsumed = false;
        bool retireConsumed = false;
        bool copyGateArmed = false;
        bool copyGateStarted = false;
        bool hookFailed = false;
        std::array<int, 6> retiredSlots{};
    };

    thread_local Controls* controls = nullptr;

    void after_prepare(Resources& resources) noexcept {
        if (controls && controls->quarantine == Phase::AfterPrepare) {
            try {
                std::lock_guard lock(resources.m);
                resources.frameUseFenceQuarantined = true;
            } catch (...) {
                controls->hookFailed = true;
            }
        }
    }

    void after_upload(Resources& resources) noexcept {
        if (controls && controls->quarantine == Phase::AfterUpload) {
            try {
                std::lock_guard lock(resources.m);
                resources.frameUseFenceQuarantined = true;
            } catch (...) {
                controls->hookFailed = true;
            }
        }
    }

    bool before_upload(Resources&, const char* label, Failure& failure) noexcept {
        if (!controls || !label)
            return true;
        if (std::string_view(label) == "output gamut Cmax") {
            ++controls->gamutVisits;
            if (controls->uploadSlot == 6) {
                controls->uploadConsumed = true;
                failure.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
                try {
                    failure.diagnostic = "ScannerLutTest output gamut Cmax upload fault";
                } catch (...) {
                    controls->hookFailed = true;
                }
                return false;
            }
            return true;
        }
        static constexpr std::array<std::string_view, 6> labels{
            "focused scan PCHIP log2 XYZ",
            "focused scan PCHIP C slopes",
            "focused scan PCHIP M slopes",
            "focused scan PCHIP Y slopes",
            "focused scan PCHIP cell minima",
            "focused scan PCHIP cell maxima"};
        const std::array<std::string_view, 6>::const_iterator found = std::find(labels.begin(), labels.end(), label);
        if (found == labels.end())
            return true;
        const auto slot = static_cast<int>(found - labels.begin());
        ++controls->uploadVisits;
        if (slot == 5 && controls->uploadGate)
            controls->copyGateArmed = true;
        if (controls->uploadSlot == slot) {
            controls->uploadConsumed = true;
            failure.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
            try {
                failure.diagnostic = label;
            } catch (...) {
                controls->hookFailed = true;
            }
            return false;
        }
        return true;
    }

    std::int32_t retire_fence_error(Resources&, const char* label) noexcept {
        if (!controls || !label)
            return static_cast<std::int32_t>(cudaSuccess);
        static constexpr std::array<std::string_view, 6> labels{
            "focused scan PCHIP log2 XYZ",
            "focused scan PCHIP C slopes",
            "focused scan PCHIP M slopes",
            "focused scan PCHIP Y slopes",
            "focused scan PCHIP cell minima",
            "focused scan PCHIP cell maxima"};
        const std::array<std::string_view, 6>::const_iterator found = std::find(labels.begin(), labels.end(), label);
        if (found == labels.end())
            return static_cast<std::int32_t>(cudaSuccess);
        const auto slot = static_cast<std::size_t>(found - labels.begin());
        if (controls->retireVisits < controls->retiredSlots.size())
            controls->retiredSlots[controls->retireVisits] = static_cast<int>(slot);
        ++controls->retireVisits;
        if (controls->retireSlot == static_cast<int>(slot)) {
            controls->retireConsumed = true;
            return static_cast<std::int32_t>(cudaErrorInvalidValue);
        }
        return static_cast<std::int32_t>(cudaSuccess);
    }
} // namespace JuicerCuda::ScannerLutTest

namespace JuicerCuda::PinnedUploadTest {
    std::int32_t injected_error(Operation operation) noexcept {
        auto* const controls = ScannerLutTest::controls;
        if (operation == Operation::Copy && controls && controls->copyGateArmed && controls->uploadGate) {
            controls->copyGateArmed = false;
            controls->copyGateStarted = controls->uploadGate->start();
        }
        return 0;
    }
    void before_diagnostic() {}
} // namespace JuicerCuda::PinnedUploadTest

namespace {
    using namespace JuicerCuda;
    using Json = nlohmann::json;
    using ScannerLutTest::Controls;
    using ScannerLutTest::Phase;
    using ScannerLutTest::UploadGate;

    Json g_fixture;
    std::filesystem::path g_logical_fixture;

    void require(bool condition, const std::string& message) {
        if (!condition)
            throw std::runtime_error(message);
    }

    float float_word(const Json& word) {
        return std::bit_cast<float>(word.get<std::uint32_t>());
    }

    std::array<float, 3> float_triplet(const Json& words) {
        require(words.size() == 3, "scanner fixture triplet extent");
        return {float_word(words[0]), float_word(words[1]), float_word(words[2])};
    }

    struct PreparedInput {
        FocusedRouteResourceInput input;
        std::array<std::array<float, 3>, 81> sensitivity{};
        std::array<std::array<float, 3>, 2> density{};
        std::array<float, 2> exposureAxis{-1.0f, 1.0f};
        std::array<float, 81> exposureIlluminant{};
        std::array<float, 243> mallettBasis{};
        std::array<std::vector<float>, 3> eps;
        std::array<std::vector<float>, 3> weighted;
        std::vector<float> baseline;

        explicit PreparedInput(const Json& row) {
            const auto& source = row.at("input");
            const int routeValue = row.at("route").get<int>();
            input.route = static_cast<Spektrafilm::ScanRoute>(routeValue);
            const bool printRoute = routeValue == static_cast<int>(Spektrafilm::ScanRoute::NegativePrintScan) ||
                                    routeValue == static_cast<int>(Spektrafilm::ScanRoute::PositivePrintScan);
            require(source.at("medium").get<int>() == (printRoute ? 1 : 0), "scanner fixture medium agrees with route");

            input.densityBounds.route = input.route;
            input.densityBounds.medium = printRoute ? Spektrafilm::DensityMedium::Print : Spektrafilm::DensityMedium::Film;
            input.densityBounds.polarity = (routeValue >= 2) ? Spektrafilm::ProfilePolarity::Positive : Spektrafilm::ProfilePolarity::Negative;
            input.densityBounds.hash = row.at("descriptor_hash").get<std::uint64_t>();
            const auto rangeMin = float_triplet(source.at("range_min_cmy"));
            input.densityBounds.invSpanCmy = float_triplet(source.at("inverse_span_cmy"));
            for (std::size_t channel = 0; channel < 3; ++channel) {
                input.densityBounds.dataMinCmy[channel] = printRoute ? rangeMin[channel] : -rangeMin[channel];
                input.densityBounds.dataMaxCmy[channel] = input.densityBounds.dataMinCmy[channel] +
                                                          1.0f / input.densityBounds.invSpanCmy[channel];
            }

            input.scannerLutDescriptor.route = input.route;
            input.scannerLutDescriptor.medium = printRoute ? Scanner::ScannedMediumKind::Print : Scanner::ScannedMediumKind::Film;
            input.scannerLutDescriptor.polarity = input.densityBounds.polarity;
            input.scannerLutDescriptor.densityBoundsHash = input.densityBounds.hash;
            input.scannerLutDescriptor.lutResolution = source.at("resolution").get<std::uint32_t>();
            input.scannerLutDescriptor.hash = row.at("descriptor_hash").get<std::uint64_t>();

            for (std::size_t i = 0; i < 3; ++i) {
                const char* names[] = {"eps_c", "eps_m", "eps_y"};
                eps[i].reserve(source.at(names[i]).size());
                for (const auto& word : source.at(names[i]))
                    eps[i].push_back(float_word(word));
                const char* weightedNames[] = {"weighted_x", "weighted_y", "weighted_z"};
                weighted[i].reserve(source.at(weightedNames[i]).size());
                for (const auto& word : source.at(weightedNames[i]))
                    weighted[i].push_back(float_word(word));
            }
            baseline.reserve(source.at("baseline").size());
            for (const auto& word : source.at("baseline"))
                baseline.push_back(float_word(word));
            input.scannerTables = {std::span<const float>(eps[0]), std::span<const float>(eps[1]), std::span<const float>(eps[2]), std::span<const float>(weighted[0]), std::span<const float>(weighted[1]), std::span<const float>(weighted[2]), std::span<const float>(baseline), source.at("sample_count").get<int>(), float_word(source.at("inv_yn")), source.at("has_baseline").get<bool>()};

            for (auto& rowSensitivity : sensitivity)
                rowSensitivity = {1.0f, 1.0f, 1.0f};
            for (auto& rowDensity : density)
                rowDensity = {0.1f, 0.2f, 0.3f};
            input.filmRaw.rgbToRawMethod = Spektrafilm::RgbToRawMethod::Mallett2019;
            input.filmRaw.finalSensitivityRgb = ThreeChannelSamplesView(std::span<const std::array<float, 3>>(sensitivity));
            input.filmRaw.finalSensitivityHash = 0x53434e4c55540001ull + static_cast<std::uint64_t>(routeValue);
            input.filmDevelop.logExposure = std::span<const float>(exposureAxis);
            input.filmDevelop.normalizedDensityCurvesRgb = ThreeChannelSamplesView(std::span<const std::array<float, 3>>(density));
            input.filmDevelop.normalizedDensityCurvesHash = 0x53434e4c55540011ull + static_cast<std::uint64_t>(routeValue);
            exposureIlluminant.fill(1.0f);
            mallettBasis.fill(1.0f);
            input.exposureIlluminant = std::span<const float>(exposureIlluminant);
            input.mallettBasis = std::span<const float>(mallettBasis);
            input.exposureSampleCount = 81;
            input.mallettRows = 81;
            input.mallettColumns = 3;
            input.mallettAvailable = true;
        }
    };

    struct ContextFixture {
        CUcontext previous = nullptr;
        CUcontext context = nullptr;
        cudaStream_t stream = nullptr;
        std::shared_ptr<DeviceAllocationLedger> ledger;
        Resources* resources = nullptr;
        ResourceManager::DeviceContextKey key{};
        bool previousCaptured = false;

        ContextFixture() {
            try {
                require(cuInit(0) == CUDA_SUCCESS, "CUDA driver initialization");
                require(cuCtxGetCurrent(&previous) == CUDA_SUCCESS, "read prior CUDA context");
                previousCaptured = true;
                CUdevice device{};
                require(cuDeviceGet(&device, 0) == CUDA_SUCCESS, "select CUDA device");
                require(cuCtxCreate(&context, nullptr, 0, device) == CUDA_SUCCESS, "create scanner fixture context");
                require(cuCtxSetCurrent(context) == CUDA_SUCCESS, "make scanner fixture context current");
                key = {0, context};
                std::string diagnostic;
                ledger = DeviceAllocationLedger::create({0, std::size_t{1024} * 1024u * 1024u}, diagnostic);
                require(ledger && ledger->bind_or_validate_cap(std::size_t{1024} * 1024u * 1024u, diagnostic), "bind scanner fixture device ledger: " + diagnostic);
                Failure failure;
                resources = create(key, 1, ledger, failure);
                if (!resources)
                    throw std::runtime_error("create exact-context scanner resource owner: " + failure.diagnostic);
                require(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess, "create scanner fixture stream");
            } catch (...) {
                cleanup();
                throw;
            }
        }

        ~ContextFixture() {
            cleanup();
        }

        void cleanup() noexcept {
            ScannerLutTest::controls = nullptr;
            if (context) {
                (void)cuCtxSetCurrent(context);
                if (stream)
                    (void)cudaStreamSynchronize(stream);
                if (resources) {
                    Failure ignored;
                    (void)drain_for_context_retire(*resources, ignored);
                    destroy(resources);
                    resources = nullptr;
                }
                purge_pinned_upload_staging_for_context(key.deviceId, key.contextOpaque, PinnedUploadPurgeDisposition::NormalRetire);
                if (stream)
                    (void)cudaStreamDestroy(stream);
                (void)cuCtxDestroy(context);
                context = nullptr;
            }
            if (previousCaptured)
                (void)cuCtxSetCurrent(previous);
        }

        void close() {
            require(cudaStreamSynchronize(stream) == cudaSuccess, "complete scanner stream before owner drain");
            PinnedUploadTest::poll(key);
            Failure failure;
            const bool drained = drain_for_context_retire(*resources, failure);
            if (!drained)
                throw std::runtime_error("drain exact scanner resource owner: " + failure.diagnostic);
            {
                std::lock_guard lock(resources->m);
                require(resources->retireQueue.empty() && resources->pendingFrameUseEvents.empty(),
                        "scanner resource owner has no pending retire or frame-use events before context destruction");
            }
            const auto state = ledger->snapshot();
            require(state.chargedBytes == 0 && state.reservedBytes == 0 && state.recordCount == 0,
                    "scanner owner retirement drains all device accounting");
            purge_pinned_upload_staging_for_context(key.deviceId, key.contextOpaque, PinnedUploadPurgeDisposition::NormalRetire);
            const auto staging = PinnedUploadTest::snapshot(key);
            require(staging.totalBytes == 0 && staging.inFlight == 0 && staging.reserved == 0,
                    "scanner staging drained before fixture context teardown");
        }
    };

    struct LutPointers {
        std::array<float*, 6> values{};
        std::uint32_t resolution = 0;
        std::uint64_t hash = 0;
    };

    LutPointers pointers(Resources& resources, int route) {
        std::lock_guard lock(resources.m);
        const auto& lut = route == 1 || route == 3 ? resources.scanPrintLut : resources.scanNegativeLut;
        return {{lut.log2PchipXYZ, lut.slopeC, lut.slopeM, lut.slopeY, lut.cellMin, lut.cellMax}, lut.res, lut.hash};
    }

    void expect_words(const LutPointers& lut, const Json& native, std::uint64_t expectedHash, cudaStream_t stream) {
        require(cudaStreamSynchronize(stream) == cudaSuccess, "complete scanner upload stream before exact device readback");
        require(lut.resolution == native.at("resolution").get<std::uint32_t>(), "scanner LUT resolution published from fixture row");
        require(lut.hash == expectedHash, "scanner LUT descriptor identity published from fixture input");
        const std::array<const char*, 6> names{"log2_xyz", "slope_c", "slope_m", "slope_y", "cell_min", "cell_max"};
        const auto resolution = static_cast<std::size_t>(lut.resolution);
        const std::array<std::size_t, 6> extents{resolution * resolution * resolution * 3,
                                                 resolution * resolution * resolution * 3,
                                                 resolution * resolution * resolution * 3,
                                                 resolution * resolution * resolution * 3,
                                                 (resolution - 1) * (resolution - 1) * (resolution - 1) * 3,
                                                 (resolution - 1) * (resolution - 1) * (resolution - 1) * 3};
        for (std::size_t lane = 0; lane < names.size(); ++lane) {
            require(lut.values[lane] != nullptr, std::string("published scanner pointer: ") + names[lane]);
            const auto& expected = native.at(names[lane]);
            require(expected.size() == extents[lane], std::string("fixture scanner extent: ") + names[lane]);
            std::vector<std::uint32_t> words(extents[lane]);
            require(cudaMemcpy(words.data(), lut.values[lane], words.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost) == cudaSuccess,
                    std::string("download scanner LUT ") + names[lane]);
            for (std::size_t i = 0; i < words.size(); ++i)
                require(words[i] == expected[i].get<std::uint32_t>(), std::string("scanner LUT exact frozen word mismatch: ") + names[lane] + "[" + std::to_string(i) + "]");
        }
    }

    bool prepare(ContextFixture& fixture, const Json& row, PreparedInput& input, Failure& failure, const FjHostPreparation* preparation = nullptr) {
        (void)row;
        return prepare_focused_route_resources(*fixture.resources, input.input, fixture.stream, failure, preparation);
    }

    bool prepare_with_owner(ContextFixture& fixture, const Json& row, PreparedInput& input, Failure& failure, Controls* controls = nullptr) {
        ScannerLutTest::controls = controls;
        bool result = false;
        try {
            result = JuicerTestSupport::with_host_preparation([&](const FjHostPreparation* preparation) {
                return prepare(fixture, row, input, failure, preparation);
            });
        } catch (...) {
            ScannerLutTest::controls = nullptr;
            if (controls && controls->quarantine != Phase::None) {
                std::lock_guard lock(fixture.resources->m);
                fixture.resources->frameUseFenceQuarantined = false;
            }
            throw;
        }
        ScannerLutTest::controls = nullptr;
        if (controls && controls->quarantine != Phase::None) {
            std::lock_guard lock(fixture.resources->m);
            fixture.resources->frameUseFenceQuarantined = false;
        }
        require(!controls || !controls->hookFailed, "scanner fixture hook failed to record its operation");
        return result;
    }

    using AllocationBytes = std::map<void*, std::uint64_t>;

    AllocationBytes allocation_bytes(Resources& resources) {
        std::lock_guard lock(resources.deviceAllocationRecordsMutex);
        AllocationBytes allocations;
        for (const auto& [pointer, reservation] : resources.deviceAllocationRecords) {
            require(reservation.active() && reservation.state() == DeviceReservationState::Committed,
                    "serving allocation has one active committed reservation");
            allocations.emplace(pointer, reservation.bytes());
        }
        return allocations;
    }

    void check_retirement_ownership(Resources& resources, const LutPointers& old, int failedSlot, const AllocationBytes& baseline, const DeviceLedgerSnapshot& before) {
        const auto prefixCount = static_cast<std::size_t>(failedSlot);
        auto expected = baseline;
        std::uint64_t prefixBytes = 0;
        const auto side = static_cast<std::uint64_t>(old.resolution);
        const auto voxelBytes = side * side * side * 3u * sizeof(float);
        const auto cellBytes = (side - 1u) * (side - 1u) * (side - 1u) * 3u * sizeof(float);
        for (std::size_t lane = 0; lane < prefixCount; ++lane) {
            require(baseline.at(old.values[lane]) == (lane < 4 ? voxelBytes : cellBytes), "exact old scanner allocation byte extent");
            prefixBytes += baseline.at(old.values[lane]);
            require(expected.erase(old.values[lane]) == 1, "retired prefix leaves the serving allocation map");
        }
        require(allocation_bytes(resources) == expected,
                "failed retirement frees every new candidate and preserves all other serving allocations");
        std::array<bool, 6> queued{};
        std::uint64_t queuedBytes = 0;
        std::uint64_t queuedCount = 0;
        {
            std::lock_guard lock(resources.m);
            for (const auto& entry : resources.retireQueue) {
                const std::array<float*, 6>::const_iterator found = std::find(old.values.begin(), old.values.end(), entry.ptr);
                const auto lane = static_cast<std::size_t>(found - old.values.begin());
                require(lane < prefixCount && !queued[lane] && entry.kind == Resources::RetireKind::DeviceFree &&
                            entry.doneEventOpaque && entry.bytes == baseline.at(entry.ptr) && entry.deviceReservation.active() &&
                            entry.deviceReservation.state() == DeviceReservationState::Retiring &&
                            entry.deviceReservation.bytes() == entry.bytes,
                        "queued scanner retirement uniquely owns its exact fenced prefix allocation");
                queued[lane] = true;
                queuedBytes += entry.bytes;
                ++queuedCount;
            }
        }
        // Each preceding retire operation may reap a completed prefix fence.
        // Account for those physical frees instead of assuming all fences wait.
        const auto reapedBytes = prefixBytes - queuedBytes;
        const auto reapedCount = static_cast<std::uint64_t>(prefixCount) - queuedCount;
        const auto after = resources.deviceLedger->snapshot();
        require(after.reservedBytes == 0 && after.committedBytes == before.committedBytes - prefixBytes &&
                    after.retiringBytes == queuedBytes && after.chargedBytes == before.chargedBytes - reapedBytes &&
                    after.recordCount == before.recordCount - reapedCount &&
                    after.chargedBytes == after.committedBytes + after.retiringBytes,
                "scanner retirement ledger exactly matches committed, queued and physically reaped ownership");
    }

    const Json& product(int route) {
        for (const auto& row : g_fixture.at("products"))
            if (row.at("route").get<int>() == route)
                return row;
        throw std::runtime_error("missing frozen scanner product route");
    }

    void prepare_route(ContextFixture& fixture, int route, Failure& failure) {
        const auto& row = product(route);
        PreparedInput input(row);
        const bool prepared = prepare_with_owner(fixture, row, input, failure);
        if (!prepared)
            throw std::runtime_error("prepare scanner route " + std::to_string(route) + ": " + failure.diagnostic);
        expect_words(pointers(*fixture.resources, route), row.at("native"), row.at("descriptor_hash").get<std::uint64_t>(), fixture.stream);
    }

    void check_failure(const Failure& failure, const std::string& fragment) {
        require(failure.status.category != FJ_STATUS_SUCCESS, "failure category is non-success");
        require(failure.diagnostic.find(fragment) != std::string::npos, "scanner failure diagnostic: expected " + fragment + ", got " + failure.diagnostic);
    }

    void require_route_metadata_cleared(Resources& resources) {
        std::lock_guard lock(resources.m);
        require(resources.routeDensityBoundsHash == 0 && resources.routeScannerDescriptorHash == 0,
                "earlier route metadata writes are cleared when scanner publication has not completed");
    }

    class ScannerLut : public ::testing::Test {
    protected:
        void SetUp() override {
            ScannerLutTest::controls = nullptr;
            require(g_fixture.contains("products") && g_fixture.at("products").size() == 4, "four frozen scanner LUT product rows");
        }
    };

    TEST_F(ScannerLut, ColdWarmAndCallbackFault) {
        for (int route = 0; route < 4; ++route) {
            ContextFixture routeFixture;
            Failure routeFailure;
            const auto& routeRow = product(route);
            prepare_route(routeFixture, route, routeFailure);
            Controls armed;
            armed.uploadSlot = 0;
            if (route == 0)
                require(fj_test_scanner_lut_fault(0, 1).category == FJ_STATUS_SUCCESS, "arm scanner acquire panic before warm hit");
            PreparedInput warmInput(routeRow);
            require(prepare_with_owner(routeFixture, routeRow, warmInput, routeFailure, &armed), "scanner hit stays warm while producer fault is armed");
            require(!armed.uploadConsumed && armed.uploadVisits == 0, "warm hit does not consume callback-upload fault");
            const auto previousRoute = pointers(*routeFixture.resources, route);
            expect_words(pointers(*routeFixture.resources, route), routeRow.at("native"), routeRow.at("descriptor_hash").get<std::uint64_t>(), routeFixture.stream);
            if (route == 0) {
                auto changedRoute = PreparedInput(routeRow);
                changedRoute.input.scannerLutDescriptor.hash ^= 0x0100000000000000ull;
                changedRoute.input.densityBounds.hash ^= 0x0100000000000000ull;
                changedRoute.input.scannerLutDescriptor.densityBoundsHash = changedRoute.input.densityBounds.hash;
                require(!prepare_with_owner(routeFixture, routeRow, changedRoute, routeFailure),
                        "changed scanner descriptor proves warm hit left the armed acquire panic pending");
                require(routeFailure.status.category == FJ_STATUS_INTERNAL_FAILURE,
                        "actual scanner miss consumes the acquire panic left by the warm hit");
                const auto afterPanic = pointers(*routeFixture.resources, route);
                require(afterPanic.values == previousRoute.values && afterPanic.hash == previousRoute.hash &&
                            afterPanic.resolution == previousRoute.resolution,
                        "acquire panic retains old scanner LUT pointers");
                require_route_metadata_cleared(*routeFixture.resources);
                require(fj_test_scanner_lut_live_owners() == 0, "panic callback leaves no Rust scanner LUT owner");
                require(prepare_with_owner(routeFixture, routeRow, warmInput, routeFailure), "scanner route retries after acquire panic");
                expect_words(pointers(*routeFixture.resources, route), routeRow.at("native"), routeRow.at("descriptor_hash").get<std::uint64_t>(), routeFixture.stream);
            }
            routeFixture.close();
        }

        ContextFixture fixture;
        Failure failure;
        const auto& row = product(0);
        PreparedInput seed(row);
        require(prepare_with_owner(fixture, row, seed, failure), "seed route zero before callback fault transitions");
        const auto previous = pointers(*fixture.resources, 0);
        for (unsigned fault = 2; fault <= 11; ++fault) {
            auto faultInput = PreparedInput(row);
            faultInput.input.scannerLutDescriptor.hash ^= 0x0200000000000000ull + fault;
            faultInput.input.densityBounds.hash = faultInput.input.scannerLutDescriptor.hash;
            faultInput.input.scannerLutDescriptor.densityBoundsHash = faultInput.input.densityBounds.hash;
            require(fj_test_scanner_lut_fault(0, fault).category == FJ_STATUS_SUCCESS, "arm actual scanner callback reservation fault");
            require(!prepare_with_owner(fixture, row, faultInput, failure), "scanner reservation fault reaches actual native miss");
            require(failure.status.category == FJ_STATUS_ALLOCATION_FAILURE, "scanner reservation fault keeps allocation-failure category");
            const auto afterFault = pointers(*fixture.resources, 0);
            require(afterFault.values == previous.values && afterFault.hash == previous.hash && afterFault.resolution == previous.resolution,
                    "scanner reservation fault retains old serving LUT");
            require_route_metadata_cleared(*fixture.resources);
            require(fj_test_scanner_lut_live_owners() == 0, "scanner reservation fault drops candidate owner");
        }
        auto viewPanic = PreparedInput(row);
        viewPanic.input.scannerLutDescriptor.hash ^= 0x0200000000000010ull;
        viewPanic.input.densityBounds.hash = viewPanic.input.scannerLutDescriptor.hash;
        viewPanic.input.scannerLutDescriptor.densityBoundsHash = viewPanic.input.densityBounds.hash;
        require(fj_test_scanner_lut_fault(1, 1).category == FJ_STATUS_SUCCESS, "arm scanner view panic");
        require(!prepare_with_owner(fixture, row, viewPanic, failure), "scanner view panic reaches actual native miss");
        require(failure.status.category == FJ_STATUS_INTERNAL_FAILURE, "scanner view panic retains internal-failure category");
        const auto afterViewPanic = pointers(*fixture.resources, 0);
        require(afterViewPanic.values == previous.values && afterViewPanic.hash == previous.hash && afterViewPanic.resolution == previous.resolution,
                "scanner view panic retains old serving LUT");
        require_route_metadata_cleared(*fixture.resources);
        require(fj_test_scanner_lut_live_owners() == 0, "view panic clears the callback view and drops candidate owner");
        require(prepare_with_owner(fixture, row, seed, failure), "retry after scanner callback faults");
        expect_words(pointers(*fixture.resources, 0), row.at("native"), row.at("descriptor_hash").get<std::uint64_t>(), fixture.stream);
        fixture.close();
    }

    TEST_F(ScannerLut, BothOwnerRechecks) {
        const auto& row = product(0);
        for (const auto phase : {Phase::AfterPrepare, Phase::AfterUpload}) {
            ContextFixture fixture;
            Failure failure;
            PreparedInput first(row);
            require(prepare_with_owner(fixture, row, first, failure), "seed exact old scanner LUT");
            const auto old = pointers(*fixture.resources, 0);
            const auto beforeLedger = fixture.ledger->snapshot();
            auto changed = PreparedInput(row);
            changed.input.scannerLutDescriptor.hash ^= phase == Phase::AfterPrepare ? 0x100ull : 0x200ull;
            changed.input.densityBounds.hash = changed.input.scannerLutDescriptor.hash;
            changed.input.scannerLutDescriptor.densityBoundsHash = changed.input.densityBounds.hash;
            Controls controls;
            controls.quarantine = phase;
            require(!prepare_with_owner(fixture, row, changed, failure, &controls), "quarantined owner recheck rejects in-flight candidate");
            const auto afterRecheck = pointers(*fixture.resources, 0);
            require(afterRecheck.values == old.values && afterRecheck.hash == old.hash && afterRecheck.resolution == old.resolution,
                    "owner recheck retains previous published LUT");
            require_route_metadata_cleared(*fixture.resources);
            require(fj_test_scanner_lut_live_owners() == 0, "owner recheck drops Rust candidate after native return");
            require(failure.diagnostic.find("quarantined") != std::string::npos, "owner recheck reports exact quarantine failure");
            const auto afterFailure = fixture.ledger->snapshot();
            require(afterFailure.reservedBytes == 0 && afterFailure.recordCount == beforeLedger.recordCount &&
                        afterFailure.chargedBytes == beforeLedger.chargedBytes,
                    "owner recheck frees candidate allocations and reservations");
            require(prepare_with_owner(fixture, row, changed, failure), "retry after owner recheck restore");
            fixture.close();
        }
    }

    TEST_F(ScannerLut, SixUploadFailures) {
        for (int slot = 0; slot < 6; ++slot) {
            ContextFixture fixture;
            Failure failure;
            const auto& row = product(0);
            PreparedInput original(row);
            require(prepare_with_owner(fixture, row, original, failure), "seed old route LUT before upload fault");
            const auto old = pointers(*fixture.resources, 0);
            const auto beforeLedger = fixture.ledger->snapshot();
            auto changed = PreparedInput(row);
            changed.input.scannerLutDescriptor.hash ^= 0x1000ull + static_cast<std::uint64_t>(slot);
            changed.input.densityBounds.hash = changed.input.scannerLutDescriptor.hash;
            changed.input.scannerLutDescriptor.densityBoundsHash = changed.input.densityBounds.hash;
            Controls controls;
            controls.uploadSlot = slot;
            require(!prepare_with_owner(fixture, row, changed, failure, &controls), "selected scanner LUT allocation failure");
            require(controls.uploadConsumed && controls.uploadVisits == static_cast<std::size_t>(slot) + 1u, "upload chain short circuits at selected slot");
            const auto afterUpload = pointers(*fixture.resources, 0);
            require(afterUpload.values == old.values && afterUpload.hash == old.hash && afterUpload.resolution == old.resolution,
                    "failed upload keeps old serving LUT");
            require_route_metadata_cleared(*fixture.resources);
            check_failure(failure, "focused scan PCHIP");
            const auto afterLedger = fixture.ledger->snapshot();
            require(afterLedger.reservedBytes == 0 && afterLedger.chargedBytes == beforeLedger.chargedBytes &&
                        afterLedger.recordCount == beforeLedger.recordCount,
                    "upload failure frees every allocated candidate and leaves no reservation");
            require(fj_test_scanner_lut_live_owners() == 0, "failed upload releases candidate callback owner");
            require(prepare_with_owner(fixture, row, changed, failure), "upload recovery after one-shot failure");
            expect_words(pointers(*fixture.resources, 0), row.at("native"), changed.input.scannerLutDescriptor.hash, fixture.stream);
            fixture.close();
        }
    }

    TEST_F(ScannerLut, OrderedRetirementFailures) {
        const auto& row = product(0);
        for (int slot = 0; slot < 6; ++slot) {
            ContextFixture fixture;
            Failure failure;
            PreparedInput first(row);
            require(prepare_with_owner(fixture, row, first, failure), "seed scanner LUT before retire fault");
            const auto old = pointers(*fixture.resources, 0);
            const auto beforeLedger = fixture.ledger->snapshot();
            const auto baselineAllocations = allocation_bytes(*fixture.resources);
            require(beforeLedger.reservedBytes == 0 && beforeLedger.retiringBytes == 0 && fixture.resources->retireQueue.empty(),
                    "fresh scanner baseline has only committed device allocations");
            auto changed = PreparedInput(row);
            changed.input.scannerLutDescriptor.hash ^= 0x10000ull + static_cast<std::uint64_t>(slot);
            changed.input.densityBounds.hash = changed.input.scannerLutDescriptor.hash;
            changed.input.scannerLutDescriptor.densityBoundsHash = changed.input.densityBounds.hash;
            Controls controls;
            controls.retireSlot = slot;
            require(!prepare_with_owner(fixture, row, changed, failure, &controls), "selected retirement fence failure");
            require(controls.uploadVisits == 6 && controls.retireConsumed && controls.retireVisits == static_cast<std::size_t>(slot) + 1u, "six real uploads precede ordered old-pointer retirement");
            const auto current = pointers(*fixture.resources, 0);
            require(current.hash == 0 && current.resolution == old.resolution, "retirement failure clears descriptor hash");
            require_route_metadata_cleared(*fixture.resources);
            check_retirement_ownership(*fixture.resources, old, slot, baselineAllocations, beforeLedger);
            for (int visited = 0; visited <= slot; ++visited)
                require(controls.retiredSlots[static_cast<std::size_t>(visited)] == visited,
                        "retire fence visits are in fixed scanner array order");
            for (int lane = 0; lane < 6; ++lane)
                require(current.values[static_cast<std::size_t>(lane)] == (lane < slot ? nullptr : old.values[static_cast<std::size_t>(lane)]),
                        "retirement failure has actual retired-prefix null slots");
            check_failure(failure, "retire failed");
            require(failure.status.category == FJ_STATUS_CUDA_FAILURE && failure.status.api == FJ_API_CUDA_RUNTIME &&
                        failure.status.native_code == cudaErrorInvalidValue,
                    "retire-fence failure preserves exact CUDA runtime status");
            require(prepare_with_owner(fixture, row, changed, failure), "scanner LUT retry after clearing retirement fault");
            expect_words(pointers(*fixture.resources, 0), row.at("native"), changed.input.scannerLutDescriptor.hash, fixture.stream);
            fixture.close();
        }
    }

    TEST_F(ScannerLut, LaterGamutFailure) {
        const auto& row = product(0);
        ContextFixture fixture;
        Failure failure;
        PreparedInput first(row);
        require(prepare_with_owner(fixture, row, first, failure), "seed scanner LUT before gamut test");
        auto changed = PreparedInput(row);
        changed.input.scannerLutDescriptor.hash ^= 0x100000ull;
        changed.input.densityBounds.hash = changed.input.scannerLutDescriptor.hash;
        changed.input.scannerLutDescriptor.densityBoundsHash = changed.input.densityBounds.hash;
        std::vector<float> cmax(Gamut::kOutputBoundaryValueCount, 1.0f);
        changed.input.outputGamut = {std::span<const float>(cmax), 0x1001ull, 0x2001ull, true};
        Controls controls;
        controls.uploadSlot = 6;
        require(!prepare_with_owner(fixture, row, changed, failure, &controls), "later gamut upload fails after scanner publication");
        require(controls.uploadConsumed && controls.gamutVisits == 1, "actual later Cmax allocation fault reached");
        const auto published = pointers(*fixture.resources, 0);
        require(published.hash == changed.input.scannerLutDescriptor.hash && published.values[0] != nullptr,
                "scanner LUT remains published before later gamut failure");
        require(fixture.resources->routeScannerDescriptorHash == changed.input.scannerLutDescriptor.hash,
                "route scanner descriptor metadata advanced before gamut failure");
        require(fixture.resources->routeDensityBoundsHash == changed.input.densityBounds.hash,
                "route density-bounds metadata advanced before gamut failure");
        check_failure(failure, "output gamut Cmax");
        expect_words(published, row.at("native"), changed.input.scannerLutDescriptor.hash, fixture.stream);
        require(prepare_with_owner(fixture, row, changed, failure), "gamut and scanner retry after one-shot failure");
        expect_words(pointers(*fixture.resources, 0), row.at("native"), changed.input.scannerLutDescriptor.hash, fixture.stream);
        fixture.close();
    }

    TEST_F(ScannerLut, AsyncBackingAndTeardown) {
        ContextFixture fixture;
        Failure failure;
        const int route = 1;
        const auto& row = product(route);
        prepare_route(fixture, route, failure);
        const auto before = PinnedUploadTest::snapshot(fixture.key);
        require(before.available > 0 && before.totalBytes > 0, "warm upload staging and device allocations");
        const auto deviceBefore = fixture.ledger->snapshot();
        require(deviceBefore.chargedBytes > 0 && deviceBefore.recordCount > 0, "warm exact-context scanner device allocations");

        UploadGate gate(fixture.stream);
        Controls controls;
        controls.uploadGate = &gate;

        PreparedInput changed(row);
        changed.input.scannerLutDescriptor.hash ^= 0x200000ull;
        changed.input.densityBounds.hash = changed.input.scannerLutDescriptor.hash;
        changed.input.scannerLutDescriptor.densityBoundsHash = changed.input.densityBounds.hash;
        require(prepare_with_owner(fixture, row, changed, failure, &controls), "warmed scanner callback enqueues upload behind explicit-stream gate");
        require(controls.uploadVisits == 6 && !controls.copyGateArmed && controls.copyGateStarted &&
                    gate.started() && gate.start_calls() == 1,
                "sixth scanner upload armed and started the gate only at pinned-copy injection");
        require(cudaStreamQuery(fixture.stream) == cudaErrorNotReady, "scanner fixture stream remains blocked after callback backing expired");
        const auto inflight = PinnedUploadTest::snapshot(fixture.key);
        require(inflight.inFlight > 0 && inflight.totalBytes >= before.totalBytes, "pinned staging owns pending scanner upload bytes");
        gate.release();
        require(cudaStreamSynchronize(fixture.stream) == cudaSuccess && gate.passed() && !gate.failed(), "release bounded upload gate and complete stream");
        expect_words(pointers(*fixture.resources, route), row.at("native"), changed.input.scannerLutDescriptor.hash, fixture.stream);
        PinnedUploadTest::poll(fixture.key);
        const auto completed = PinnedUploadTest::snapshot(fixture.key);
        require(completed.inFlight == 0 && completed.reserved == 0, "completed staging is reusable before owner teardown");
        fixture.close();
    }
} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "FAIL Scanner LUT: expected one logical fixture path before GoogleTest options\n");
        return 2;
    }
    try {
        g_logical_fixture = argv[1];
        g_fixture = load_shared_fixture(g_logical_fixture, "manifest.json");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL Scanner LUT fixture: %s\n", error.what());
        return 2;
    }
    for (int index = 1; index + 1 < argc; ++index)
        argv[index] = argv[index + 1];
    --argc;
    argv[argc] = nullptr;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
