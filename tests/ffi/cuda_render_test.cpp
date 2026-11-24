#include "prepared_boundary.h"
#include "prepared_raw_render_fixture.h"
#include "spatial_optics_fixture.h"
#include "Cuda/Scan/JuicerCudaScanDensityCaptureTest.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "prepared_descriptors.h"
#include "case_results.h"
#include "juicer_cuda_owner.h"
#include "juicer_cuda_prepared.h"
#include "juicer_test_api.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#include "Cuda/JuicerCudaFilmGrain.h"
#include "Hash.h"
#include "ScanRoute.h"

namespace JuicerProcess::TestSupport {
    class RootLifetimeObserver final {
    public:
        static std::shared_ptr<JuicerCuda::Resources> resources(Root& root, const JuicerCuda::ResourceManager::DeviceContextKey& key) {
            std::lock_guard<std::mutex> lock(root._cudaResourcesMutex);
            std::shared_ptr<JuicerCuda::Resources> owner;
            std::uint64_t epoch = 0;
            for (const auto& [identity, entry] : root._cudaContextResources) {
                if (identity.deviceContextKey == key && identity.contextEpoch > epoch) {
                    owner = entry.frameOwner;
                    epoch = identity.contextEpoch;
                }
            }
            return owner;
        }

        static std::size_t fences(Root& root) {
            std::shared_ptr<JuicerCuda::Resources> owner;
            {
                std::lock_guard<std::mutex> lock(root._cudaResourcesMutex);
                if (!root._cudaContextResources.empty()) {
                    owner = root._cudaContextResources.begin()->second.frameOwner;
                }
            }
            if (!owner) {
                return 0;
            }
            std::lock_guard<std::mutex> lock(owner->m);
            return owner->pendingFrameUseEvents.size();
        }
    };
} // namespace JuicerProcess::TestSupport

namespace JuicerCuda::PinnedUploadTest {
    thread_local std::array<std::int32_t, 5> errors{};
    thread_local std::array<unsigned, 5> consumed{};
    thread_local UploadObservation uploadObservation{};
    thread_local bool observingUploads = false;
    std::int32_t injected_error(Operation operation) noexcept {
        const auto index = static_cast<std::size_t>(operation);
        const auto error = std::exchange(errors[index], 0);
        consumed[index] += error != 0 ? 1u : 0u;
        return error;
    }
    void before_diagnostic() {}
    void after_overwrite_staging(Resources&) noexcept {}
    void reset_upload_observation() noexcept {
        uploadObservation = {};
        observingUploads = true;
    }
    UploadObservation finish_upload_observation() noexcept {
        observingUploads = false;
        return uploadObservation;
    }
    void observe_staged_upload(const char* label, std::size_t bytes) noexcept {
        if (!observingUploads || !label)
            return;
        const std::string_view name(label);
        if (!name.starts_with("phase4B "))
            return;
        const auto add = [bytes](TransferCount& transfer) {
            ++transfer.calls;
            transfer.bytes += bytes;
        };
        add(uploadObservation.print);
        constexpr std::array<std::string_view, 4> filmLanes{
            "phase4B print film density epsC", "phase4B print film density epsM", "phase4B print film density epsY", "phase4B print film density base"};
        for (std::size_t lane = 0; lane < filmLanes.size(); ++lane)
            if (name == filmLanes[lane])
                add(uploadObservation.filmDensity[lane]);
        if (name == "phase4B filtered main print illuminant")
            add(uploadObservation.mainIlluminant);
        if (name == "phase4B filtered preflash print illuminant")
            add(uploadObservation.preflashIlluminant);
    }
} // namespace JuicerCuda::PinnedUploadTest

namespace JuicerCuda::RenderTest {
    enum class Injection : std::uint8_t {
        None,
        Allocation,
        Standard,
        Unknown,
        Typed
    };
    thread_local Injection injection = Injection::None;
    thread_local FjStatus injectedStatus{};
    thread_local bool injectedDeferredDirError = false;
    thread_local int completionOverride = 0;
    thread_local unsigned completionCalls = 0;
    thread_local int lastCompletion = 0;
    thread_local bool missingFilmBank = false;
    thread_local bool nullGrainPlane = false;
    thread_local unsigned filmProxyReservations = 0;
    thread_local PrintLaunchObservation* printLaunchObserver = nullptr;

    void set_print_launch_observer(PrintLaunchObservation* observation) noexcept {
        printLaunchObserver = observation;
    }

    void observe_print_launch(const PrintPreparedView& prepared, const PrintExposePayload& expose) {
        if (!printLaunchObserver)
            return;
        if (prepared.spectralSampleCount != 81)
            throw std::runtime_error("Print launch observation requires 81 spectral samples");
        auto& out = *printLaunchObserver;
        out = {};
        out.filmDensityC = reinterpret_cast<std::uintptr_t>(prepared.filmDensityTables.epsC);
        out.filmDensityM = reinterpret_cast<std::uintptr_t>(prepared.filmDensityTables.epsM);
        out.filmDensityY = reinterpret_cast<std::uintptr_t>(prepared.filmDensityTables.epsY);
        out.filmBase = reinterpret_cast<std::uintptr_t>(prepared.filmDensityTables.baseDensityMin);
        out.mainIlluminant = reinterpret_cast<std::uintptr_t>(prepared.mainIlluminant);
        out.preflashIlluminant = reinterpret_cast<std::uintptr_t>(prepared.preflashIlluminant);
        out.filmDensityHash = prepared.filmDensityTablesHash;
        out.profileTablesHash = prepared.profileTablesHash;
        out.mainIlluminantHash = prepared.mainIlluminantHash;
        out.preflashIlluminantHash = prepared.preflashIlluminantHash;
        out.spectralSampleCount = prepared.spectralSampleCount;
        out.printExposure = expose.printExposure;
        out.printPreflashExposure = expose.printPreflashExposure;
        out.printMidgrayFactor = expose.printMidgrayFactor;
        std::copy(std::begin(expose.printPreflashRaw), std::end(expose.printPreflashRaw), std::begin(out.printPreflashRaw));
        const auto& cyan = prepared.printDcC;
        if (prepared.printDcM.x != cyan.x || prepared.printDcY.x != cyan.x)
            throw std::runtime_error("prepared Print curves did not share the cyan-owned axis");
        const auto bind = [](PrintLaunchObservation::PaperLane& lane, const float* device, int count) {
            if (!device || count <= 0)
                throw std::runtime_error("prepared Print table lane is absent");
            lane.address = reinterpret_cast<std::uintptr_t>(device);
            lane.words.resize(static_cast<std::size_t>(count));
        };
        bind(out.paperTables[0], cyan.x, cyan.n);
        const std::array channels{cyan, prepared.printDcM, prepared.printDcY, prepared.printSensC, prepared.printSensM, prepared.printSensY};
        for (std::size_t lane = 0; lane < channels.size(); ++lane)
            bind(out.paperTables[lane + 1], channels[lane].y, channels[lane].n);
        out.valid = true;
    }
    void complete_print_launch_observation(PrintLaunchObservation& out, std::uintptr_t stream) {
        if (!out.valid)
            return;
        auto* const streamHandle = reinterpret_cast<cudaStream_t>(stream);
        if (cudaStreamSynchronize(streamHandle) != cudaSuccess)
            throw std::runtime_error("Print launch observation stream synchronization failed");
        const auto copy = [](float* destination, std::uintptr_t device, std::size_t count) {
            if (!device) {
                std::fill_n(destination, count, 0.0f);
                return;
            }
            if (cudaMemcpy(destination, reinterpret_cast<const void*>(device), count * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
                throw std::runtime_error("Print launch observation device readback failed");
        };
        copy(out.mainLight.data(), out.mainIlluminant, out.mainLight.size());
        copy(out.preflashLight.data(), out.preflashIlluminant, out.preflashLight.size());
        copy(out.filmBaseWords.data(), out.filmBase, out.filmBaseWords.size());
        copy(out.filmDyes.data(), out.filmDensityC, 81);
        copy(out.filmDyes.data() + 81, out.filmDensityM, 81);
        copy(out.filmDyes.data() + 162, out.filmDensityY, 81);
        for (auto& lane : out.paperTables) {
            if (cudaMemcpy(lane.words.data(), reinterpret_cast<const void*>(lane.address), lane.words.size() * sizeof(lane.words[0]), cudaMemcpyDeviceToHost) != cudaSuccess)
                throw std::runtime_error("Print paper table readback failed");
        }
    }
    void observe_film_proxy_reservation() noexcept {
        ++filmProxyReservations;
    }
    bool omit_film_bank() noexcept {
        return std::exchange(missingFilmBank, false);
    }
    bool omit_grain_plane() noexcept {
        return std::exchange(nullGrainPlane, false);
    }
    int completion_status(int actual) noexcept {
        ++completionCalls;
        lastCompletion = actual;
        return completionOverride ? std::exchange(completionOverride, 0) : actual;
    }


    struct PendingWork {
        cudaStream_t stream = nullptr;
        cudaEvent_t event = nullptr;
        unsigned calls = 0;
        unsigned checkpoint = 0;
        std::atomic<bool> released{false};
        bool observedPending = false;
        std::chrono::milliseconds maximumHold;

        explicit PendingWork(cudaStream_t inputStream, unsigned inputCheckpoint, std::chrono::milliseconds hold = std::chrono::milliseconds(250))
            : stream(inputStream), checkpoint(inputCheckpoint), maximumHold(hold) {
            if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
                throw std::runtime_error("create pending-work event");
            }
        }
        ~PendingWork() {
            released.store(true);
            (void)cudaStreamSynchronize(stream);
            (void)cudaEventDestroy(event);
        }
        static void CUDART_CB wait(void* user) {
            auto& work = *static_cast<PendingWork*>(user);
            const auto deadline = std::chrono::steady_clock::now() + work.maximumHold;
            while (!work.released.load()) {
                if (std::chrono::steady_clock::now() >= deadline) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    };
    thread_local PendingWork* pendingWork = nullptr;
    thread_local PendingWork* scanConsumerHold = nullptr;

    void hold_scan_consumer(void* stream) {
        auto* const hold = std::exchange(scanConsumerHold, nullptr);
        if (!hold)
            return;
        if (stream != hold->stream || cudaLaunchHostFunc(hold->stream, PendingWork::wait, hold) != cudaSuccess ||
            cudaEventRecord(hold->event, hold->stream) != cudaSuccess)
            throw std::runtime_error("hold the actual scanner launch on the fixture stream");
    }

    // Finite device copies keep the upload event outstanding without a host
    // gate or GPU wait. Enqueue only after the renderer's synchronous paths.
    struct ExpiringUpload {
        static constexpr std::size_t kBytes = std::size_t{64} * 1024u * 1024u;
        std::shared_ptr<Resources> resources;
        void* first = nullptr;
        void* second = nullptr;
        cudaStream_t stream = nullptr;
        FjFloatSpan source{};
        PinnedUploadTest::Snapshot staging;
        JuicerCudaTest::StateLifetimeCase lifetimeCase = JuicerCudaTest::StateLifetimeCase::SuppliedStream;
        unsigned calls = 0;

        ~ExpiringUpload() {
            (void)cudaStreamSynchronize(stream);
            (void)cudaFree(first);
            (void)cudaFree(second);
        }
        void enqueue() {
            ++calls;
            for (unsigned i = 0; i < 128; ++i) {
                if (cudaMemcpyAsync(second, first, kBytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
                    cudaMemcpyAsync(first, second, kBytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
                    throw std::runtime_error("queue finite expiry work");
                }
            }
            Failure failure;
            using Case = JuicerCudaTest::StateLifetimeCase;
            using Operation = PinnedUploadTest::Operation;
            if (lifetimeCase == Case::PinnedAllocation || lifetimeCase == Case::PinnedEventCreation) {
                // Occupy each existing reusable block before requesting a real cold
                // allocation. These uploads use the held state's actual first word.
                const auto available = PinnedUploadTest::snapshot(resources->ownerContextKey).available;
                for (std::size_t i = 0; i < available; ++i) {
                    if (!PinnedUploadTest::upload(*resources, second, source.data, sizeof(float), stream, failure))
                        throw ExecutionFailure{failure};
                }
                const auto operation = lifetimeCase == Case::PinnedAllocation ? Operation::Allocate : Operation::CreateEvent;
                PinnedUploadTest::errors[static_cast<std::size_t>(operation)] = cudaErrorInvalidValue;
            } else if (lifetimeCase == Case::PinnedCopy) {
                PinnedUploadTest::errors[static_cast<std::size_t>(Operation::Copy)] = cudaErrorInvalidValue;
            } else if (lifetimeCase == Case::PinnedEventRecord || lifetimeCase == Case::PinnedSynchronization) {
                PinnedUploadTest::errors[static_cast<std::size_t>(Operation::RecordEvent)] = cudaErrorInvalidValue;
                if (lifetimeCase == Case::PinnedSynchronization)
                    PinnedUploadTest::errors[static_cast<std::size_t>(Operation::Synchronize)] = cudaErrorInvalidValue;
            }
            const bool uploaded = PinnedUploadTest::upload(*resources, second, source.data, source.count * sizeof(float), stream, failure);
            staging = PinnedUploadTest::snapshot(resources->ownerContextKey);
            if (!uploaded)
                throw ExecutionFailure{failure};
        }
    };
    thread_local ExpiringUpload* expiringUpload = nullptr;
    thread_local ExpiringUpload* cancellingUpload = nullptr;
    thread_local unsigned cancellationCalls = 0;
    thread_local unsigned uploadAtCancellation = 0;

    thread_local PendingWork* completionWork = nullptr;

    void after_execute() {
        if (expiringUpload) {
            expiringUpload->enqueue();
        }
        if (completionWork) {
            if (cudaLaunchHostFunc(completionWork->stream, PendingWork::wait, completionWork) != cudaSuccess ||
                cudaEventRecord(completionWork->event, completionWork->stream) != cudaSuccess) {
                throw std::runtime_error("enqueue stream completion probe");
            }
            completionWork->observedPending = cudaEventQuery(completionWork->event) == cudaErrorNotReady;
        }
    }

    void before_abort_query() {
        if (cancellingUpload && ++cancellationCalls == uploadAtCancellation)
            cancellingUpload->enqueue();
        if (pendingWork && ++pendingWork->calls == pendingWork->checkpoint) {
            if (cudaLaunchHostFunc(pendingWork->stream, PendingWork::wait, pendingWork) != cudaSuccess ||
                cudaEventRecord(pendingWork->event, pendingWork->stream) != cudaSuccess) {
                throw std::runtime_error("enqueue pending cancellation work");
            }
            pendingWork->observedPending = cudaEventQuery(pendingWork->event) == cudaErrorNotReady;
        }
    }

    void before_execute() {
        switch (injection) {
            case Injection::None:
                return;
            case Injection::Allocation:
                throw std::bad_alloc();
            case Injection::Standard:
                throw std::runtime_error("render injection 100% diagnostic");
            case Injection::Unknown:
                throw 17;
            case Injection::Typed:
                throw ExecutionFailure{{injectedStatus, "render injection 100% diagnostic"}, injectedDeferredDirError};
        }
    }
} // namespace JuicerCuda::RenderTest

namespace {
    void require(bool condition, const char* detail) {
        if (!condition) {
            throw std::runtime_error(detail);
        }
    }

    void require_status(FjStatus actual, FjStatus expected, const char* detail) {
        if (actual.category != expected.category || actual.api != expected.api || actual.native_code != expected.native_code) {
            throw std::runtime_error(std::string(detail) + ": category=" + std::to_string(actual.category) +
                                     " api=" + std::to_string(actual.api) + " code=" + std::to_string(actual.native_code));
        }
    }

    void require_status(FjRenderOutcome actual, FjStatus expected, const char* detail) {
        require_status(actual.status, expected, detail);
        require(actual.flags == 0, "ordinary render returned a deferred-DIR flag");
    }

    struct AbortQuery {
        std::thread::id caller = std::this_thread::get_id();
        unsigned calls = 0;
        unsigned cancelAt = 0;
        bool sameThread = true;
        bool reentry = false;
        bool reentryRejected = true;
        std::uint32_t result = FJ_ABORT_REQUESTED;
        FjCuda* cuda = nullptr;
        const FjFrame* frame = nullptr;

        static std::uint32_t query(void* user) noexcept {
            auto& self = *static_cast<AbortQuery*>(user);
            self.sameThread = self.sameThread && self.caller == std::this_thread::get_id();
            ++self.calls;
            if (self.reentry) {
                FjCudaContext context{};
                const auto result = fj_cuda_inspect(self.cuda, self.frame, &context, nullptr);
                self.reentryRejected = self.reentryRejected && result.category == FJ_STATUS_UNSUPPORTED_INPUT;
            }
            return self.calls == self.cancelAt ? self.result : FJ_ABORT_CONTINUE;
        }
    };
} // namespace

namespace JuicerCudaTest {
    void check_prepared_state_lifetime(std::optional<JuicerTestSupport::PreparedStateHold>& owner, const FjAssets* assets, FjCuda* cuda, const FjCudaContext& context, const FjFrame& inputFrame, const FjSubmission& submission, FjPreparedHostData& prepared, FjErrorBuffer* error, StateLifetimeCase lifetimeCase) {
        JuicerProcess::Root& root = JuicerCuda::borrow_root(cuda);
        require(prepared.spatial_dir.hash != 0 && prepared.optics.flags == 0 && prepared.grain.hash == 0 && prepared.effects.hash == 0, "held-state lifetime retains active spatial DIR");
        const JuicerCuda::ResourceManager::DeviceContextKey key{context.device_id, reinterpret_cast<void*>(context.context)};
        const auto route = prepared.route;
        require(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(inputFrame.stream)) == cudaSuccess, "complete prior held-state warm render");
        JuicerCuda::purge_pinned_upload_staging_for_context(key.deviceId, key.contextOpaque, JuicerCuda::PinnedUploadPurgeDisposition::NormalRetire);
        auto frame = inputFrame;
        if (lifetimeCase == StateLifetimeCase::PresentNullStream || lifetimeCase == StateLifetimeCase::AbsentStream)
            frame.stream = 0;
        if (lifetimeCase == StateLifetimeCase::AbsentStream)
            frame.flags &= ~FJ_FRAME_STREAM_PRESENT;
        JuicerCuda::RenderTest::ExpiringUpload upload;
        upload.lifetimeCase = lifetimeCase;
        upload.resources = JuicerProcess::TestSupport::RootLifetimeObserver::resources(root, key);
        require(upload.resources != nullptr, "held-state expiry has native resources");
        std::shared_ptr<JuicerCuda::DeviceAllocationLedger> deferredLedger;
        if (lifetimeCase == StateLifetimeCase::DeferredDestruction) {
            std::string diagnostic;
            deferredLedger = JuicerCuda::DeviceAllocationLedger::create({context.device_id, 2 * upload.kBytes}, diagnostic);
            require(deferredLedger && deferredLedger->bind_or_validate_cap(2 * upload.kBytes, diagnostic), "bind deferred held-state upload ledger");
            upload.resources = std::shared_ptr<JuicerCuda::Resources>(new JuicerCuda::Resources(key, 1, deferredLedger), JuicerCuda::destroy);
        }
        upload.stream = reinterpret_cast<cudaStream_t>(frame.stream);
        require(cudaMalloc(&upload.first, upload.kBytes) == cudaSuccess && cudaMalloc(&upload.second, upload.kBytes) == cudaSuccess, "allocate held-state expiry workspace");
        require(cudaMemsetAsync(upload.first, 0x28, upload.kBytes, upload.stream) == cudaSuccess, "initialize held-state expiry work");
        upload.source = prepared.film_exposure.tc_lut_rgba.count != 0 ? prepared.film_exposure.tc_lut_rgba : prepared.film_development.density_rgb;
        require(upload.source.count != 0, "held-state source is substantial storage");
        const float* address = upload.source.data;
        const std::vector<float> expected(address, address + upload.source.count);
        JuicerCuda::Failure failure;
        require(JuicerCuda::PinnedUploadTest::upload(*upload.resources, upload.second, address, expected.size() * sizeof(float), upload.stream, failure), "warm held-state native staging");
        require(cudaStreamSynchronize(upload.stream) == cudaSuccess, "complete held-state staging warmup");
        JuicerCuda::PinnedUploadTest::poll(key);
        JuicerCuda::PinnedUploadTest::consumed = {};
        AbortQuery cancel;
        const bool cancellation = lifetimeCase == StateLifetimeCase::CancelAfterPrepare || lifetimeCase == StateLifetimeCase::CancelAfterExecute;
        cancel.cancelAt = lifetimeCase == StateLifetimeCase::CancelAfterPrepare ? 2u : 3u;
        JuicerCuda::RenderTest::cancellationCalls = 0;
        JuicerCuda::RenderTest::uploadAtCancellation = cancel.cancelAt;
        JuicerCuda::RenderTest::cancellingUpload = cancellation ? &upload : nullptr;
        JuicerCuda::RenderTest::expiringUpload = cancellation ? nullptr : &upload;
        const auto result = render_raw_prepared(cuda, frame, submission, prepared, cancellation ? FjAbortCallback{AbortQuery::query, &cancel} : FjAbortCallback{}, error);
        JuicerCuda::RenderTest::expiringUpload = nullptr;
        JuicerCuda::RenderTest::cancellingUpload = nullptr;
        const bool failed = lifetimeCase == StateLifetimeCase::PinnedAllocation || lifetimeCase == StateLifetimeCase::PinnedEventCreation || lifetimeCase == StateLifetimeCase::PinnedCopy || lifetimeCase == StateLifetimeCase::PinnedSynchronization;
        const FjStatus expectedStatus = failed ? FjStatus{FJ_STATUS_CUDA_FAILURE, FJ_API_CUDA_RUNTIME, cudaErrorInvalidValue} : FjStatus{cancellation ? FJ_STATUS_CANCELLED : FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
        require_status(result, expectedStatus, "held-state lifetime C return");
        require(upload.calls == 1, "held-state upload executed exactly once at actual native boundary");
        require(upload.staging.reserved == 0, "held-state upload has no escaped reservation");
        if (cancellation)
            require(cancel.calls == cancel.cancelAt && cancel.sameThread, "held-state cancellation checkpoint and thread");
        using Operation = JuicerCuda::PinnedUploadTest::Operation;
        const auto consumed = [&](Operation operation) {
            return JuicerCuda::PinnedUploadTest::consumed[static_cast<std::size_t>(operation)];
        };
        if (lifetimeCase == StateLifetimeCase::PinnedAllocation)
            require(consumed(Operation::Allocate) == 1 && consumed(Operation::CreateEvent) == 0, "actual pinned allocation failure consumed");
        if (lifetimeCase == StateLifetimeCase::PinnedEventCreation)
            require(consumed(Operation::CreateEvent) == 1, "actual pinned event creation failure consumed");
        if (lifetimeCase == StateLifetimeCase::PinnedCopy)
            require(consumed(Operation::Copy) == 1 && consumed(Operation::RecordEvent) == 0, "actual pinned copy failure consumed");
        if (lifetimeCase == StateLifetimeCase::PinnedEventRecord || lifetimeCase == StateLifetimeCase::PinnedSynchronization)
            require(consumed(Operation::RecordEvent) == 1, "actual pinned event record failure consumed");
        if (lifetimeCase == StateLifetimeCase::PinnedSynchronization)
            require(consumed(Operation::Synchronize) == 1 && upload.staging.quarantined == 1, "actual pinned synchronization failure quarantines backing");
        cudaEvent_t event = nullptr;
        for (const auto& block : upload.staging.blocks) {
            if (block.capacity >= expected.size() * sizeof(float) && std::memcmp(block.pointer, expected.data(), expected.size() * sizeof(float)) == 0) {
                require(block.pointer != address, "held-state source copied to native staging");
                event = static_cast<cudaEvent_t>(block.event);
                break;
            }
        }
        const bool asynchronous = lifetimeCase == StateLifetimeCase::SuppliedStream || lifetimeCase == StateLifetimeCase::PresentNullStream || lifetimeCase == StateLifetimeCase::IdleRetirement || lifetimeCase == StateLifetimeCase::DeferredDestruction;
        if (asynchronous)
            require(event && cudaEventQuery(event) == cudaErrorNotReady, "held-state native staging is pending at C return");
        if (lifetimeCase == StateLifetimeCase::AbsentStream || lifetimeCase == StateLifetimeCase::PinnedEventRecord)
            require(cudaStreamQuery(upload.stream) == cudaSuccess, "held-state required return-time completion");
        require_status(fj_test_assets_release_cached_payloads(assets, error), {FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, "release held-state asset cache");
        const auto live = fj_test_prepared_live_states();
        owner.reset();
        require(fj_test_prepared_live_states() + 1 == live, "consume held state exactly once after C return");
        prepared = {};
        upload.source = {};
        if (asynchronous)
            require(cudaEventQuery(event) == cudaErrorNotReady, "state and invocation expire while native staging remains pending");
        void* const destination = upload.second;
        if (lifetimeCase == StateLifetimeCase::DeferredDestruction) {
            JuicerCuda::DeviceByteReservation reservation;
            require(deferredLedger->reserve({key, 1, upload.kBytes}, reservation, failure), "reserve deferred upload destination");
            std::string diagnostic;
            require(reservation.commit(upload.kBytes, diagnostic), "commit deferred upload destination");
            upload.resources->deviceAllocationRecords.emplace(upload.second, std::move(reservation));
            upload.second = nullptr; // The exact-context native queue now owns it.
            CUdevice device = 0;
            CUcontext foreign = nullptr;
            require(cuDeviceGet(&device, context.device_id) == CUDA_SUCCESS && cuCtxCreate(&foreign, nullptr, 0, device) == CUDA_SUCCESS, "create fixture-owned foreign context");
            upload.resources.reset();
            require(deferredLedger->snapshot().chargedBytes == upload.kBytes && deferredLedger->snapshot().recordCount == 1, "foreign-context destruction retains actual native upload allocation");
            require(cuCtxSetCurrent(reinterpret_cast<CUcontext>(context.context)) == CUDA_SUCCESS, "restore exact held-state upload context");
            require(cuCtxDestroy(foreign) == CUDA_SUCCESS, "destroy only fixture-owned foreign context");
            require(cudaEventSynchronize(event) == cudaSuccess, "complete native deferred upload after state and resource-handle expiry");
        } else if (lifetimeCase == StateLifetimeCase::IdleRetirement) {
            const std::weak_ptr<JuicerCuda::Resources> nativeOwner = upload.resources;
            std::string diagnostic;
            require(root.retire_idle_context(context.device_id, reinterpret_cast<void*>(context.context), diagnostic), "retire native owner after Rust state expiry");
            require(!nativeOwner.expired(), "test retains native retirement owner independently of Rust state");
            upload.resources.reset();
            require(nativeOwner.expired(), "final native retirement owner released");
            require(JuicerCuda::PinnedUploadTest::snapshot(key).totalBytes == 0, "native retirement drains pinned backing");
        } else {
            require(cudaStreamSynchronize(upload.stream) == cudaSuccess, "complete native staging after state expiry");
        }
        if (!failed || lifetimeCase == StateLifetimeCase::PinnedSynchronization) {
            std::vector<float> actual(expected.size());
            require(cudaMemcpy(actual.data(), destination, actual.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess, "read retained native upload");
            require(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0, "all held-state words survive native staging");
        }
        if (lifetimeCase == StateLifetimeCase::DeferredDestruction) {
            require(JuicerCuda::drain_deferred_resources(key, failure), "drain actual held-state native deferred queue on exact owner context");
            require(deferredLedger->snapshot().chargedBytes == 0 && deferredLedger->snapshot().recordCount == 0, "native deferred upload ownership drains exactly once to zero");
            require(JuicerCuda::drain_deferred_resources(key, failure), "empty deferred held-state drain is inert");
        }
        if (lifetimeCase == StateLifetimeCase::PinnedSynchronization) {
            // A failed record leaves no completion proof in the pool. After an
            // actual stream completion, the fixture reclaims that abandoned host
            // allocation explicitly, as the owning pinned-upload test does.
            require(upload.staging.blocks.size() == 1, "quarantined held-state staging has one actual allocation");
            void* pointer = upload.staging.blocks.front().pointer;
            JuicerCuda::purge_pinned_upload_staging_for_context(key.deviceId, key.contextOpaque, JuicerCuda::PinnedUploadPurgeDisposition::NormalRetire);
            require(cudaFreeHost(pointer) == cudaSuccess, "reclaim completed quarantined held-state staging");
        }
        require(std::all_of(JuicerCuda::PinnedUploadTest::errors.begin(), JuicerCuda::PinnedUploadTest::errors.end(), [](auto code) {
                    return code == 0;
                }),
                "all held-state operation faults consumed");
        std::cout << "held-state lifetime route=" << route << " case=" << static_cast<unsigned>(lifetimeCase) << " consumed=1 native-reservation=0\n";
    }

    void check_frame_bindings(FjCuda* cuda, const FjCudaContext& context, const FjFrame& frame, const FjSubmission& submission, const FjPreparedHostData& prepared, const FjPreparedStateView* state, const FjHostPreparation* hostPreparation, const std::string& invocationContext) {
        auto& root = JuicerCuda::borrow_root(cuda);
        const JuicerCuda::ResourceManager::DeviceContextKey contextKey{context.device_id, reinterpret_cast<void*>(context.context)};
        const auto check = [&](const FjFrame& candidateFrame, const FjSubmission& candidateSubmission, const FjPreparedHostData& candidate, const char* rejectedField) {
            std::array<char, 256> message{};
            FjErrorBuffer error{message.data(), message.size(), 0};
            AbortQuery callback;
            callback.cancelAt = 1;
            const auto result = rejectedField
                                    ? fj_cuda_render(cuda, &context, &candidateFrame, &candidateSubmission, &candidate, {AbortQuery::query, &callback}, &error)
                                    : render_raw_prepared(cuda, candidateFrame, candidateSubmission, candidate, {AbortQuery::query, &callback}, &error, hostPreparation);
            require_status(result, {rejectedField ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_CANCELLED, FJ_API_NONE, 0}, "prepared frame binding");
            require(callback.calls == (rejectedField ? 0u : 1u), "frame binding admission precedes preparation");
            if (rejectedField) {
                require(std::string(message.data(), error.length) == std::string("MalformedPreparedHostData field=") + rejectedField, "frame binding rejection origin");
            }
        };
        // A valid producer projection reaches the first cancellation checkpoint;
        // malformed bindings must stop before that checkpoint or any GPU work.
        check(frame, submission, prepared, nullptr);
        const auto rejectBeforeExecution = [&](const FjPreparedHostData& changed, const char* field, std::string_view name) {
            const auto owner = JuicerProcess::TestSupport::RootLifetimeObserver::resources(root, contextKey);
            const auto fences = JuicerProcess::TestSupport::RootLifetimeObserver::fences(root);
            check(frame, submission, changed, field);
            require(JuicerProcess::TestSupport::RootLifetimeObserver::resources(root, contextKey) == owner, "invalid input reached context resource publication");
            require(JuicerProcess::TestSupport::RootLifetimeObserver::fences(root) == fences, "invalid input retained new CUDA work");
            std::cout << "frame binding route=" << prepared.route << " case=" << name << " field=" << field << " rejected before execution\n";
        };
        auto scannerInput = prepared;
        --scannerInput.scanner_spectra.dye_c.count;
        rejectBeforeExecution(scannerInput, "scanner_spectra", "short-dye");
        scannerInput = prepared;
        scannerInput.scanner_spectra.weighted_y.data = nullptr;
        rejectBeforeExecution(scannerInput, "scanner_spectra", "missing-weight-data");
        scannerInput = prepared;
        scannerInput.scanner_spectra.base_density = {prepared.scanner_spectra.dye_c.data, 80};
        rejectBeforeExecution(scannerInput, "scanner_spectra.base_density", "short-base");
        scannerInput.scanner_spectra.base_density.count = 0;
        rejectBeforeExecution(scannerInput, "scanner_spectra.base_density", "unexpected-base-data");
        scannerInput.scanner_spectra.base_density = {nullptr, 81};
        rejectBeforeExecution(scannerInput, "scanner_spectra.base_density", "missing-base-data");
        scannerInput = prepared;
        std::swap(scannerInput.scanner_spectra.dye_c, scannerInput.scanner_spectra.dye_m);
        rejectBeforeExecution(scannerInput, "submission_identity", "swapped-dyes");
        scannerInput = prepared;
        scannerInput.scanner_spectra.inverse_y_normalization *= 2.F;
        rejectBeforeExecution(scannerInput, "submission_identity", "changed-inverse-y");
        scannerInput = prepared;
        auto& cat02First = scannerInput.output_color.cat02[0];
        cat02First = std::nextafter(cat02First, std::numeric_limits<float>::infinity());
        rejectBeforeExecution(scannerInput, "submission_identity", "changed-cat02");
        check(frame, submission, prepared, nullptr);
        // These literal corruptions replace the retired post-copy recipe
        // mutations; a complete held state cannot contain either mismatch.
        if (prepared.film_exposure.method != FJ_RAW_MALLETT_2019) {
            auto changed = prepared;
            changed.film_exposure.method = FJ_RAW_MALLETT_2019;
            check(frame, submission, changed, "film_exposure.tc_lut_rgba");
        }
        if (prepared.route == FJ_ROUTE_NEGATIVE_PRINT || prepared.route == FJ_ROUTE_POSITIVE_PRINT) {
            auto changed = prepared;
            changed.print.density_cmy = {};
            check(frame, submission, changed, "print.density_cmy");
            require(prepared.print.film_base_density.data && prepared.print.film_base_density.count == 81,
                    "Print film base fixture requires the complete enlarger span");
            changed = prepared;
            changed.print.film_base_density = {};
            rejectBeforeExecution(changed, "print.film_base_density", "print-film-base-null-zero");
            changed = prepared;
            --changed.print.film_base_density.count;
            rejectBeforeExecution(changed, "print.film_base_density", "print-film-base-short-positive");
            check(frame, submission, prepared, nullptr);
        }
        if (prepared.spatial_dir.hash != 0) {
            const bool extentCaseAvailable = prepared.optics.flags == 0 && prepared.grain.hash == 0 && prepared.effects.hash == 0;
            std::vector<std::string> ids{"filter-domain-x", "filter-domain-y", "filter-domain-width", "filter-domain-height"};
            if (extentCaseAvailable)
                ids.insert(ids.begin(), "render-extent-width");
            JuicerTests::Cohort cohort(invocationContext + "/frame-spatial-dir", ids);
            for (int fault = 0; fault < 5; ++fault) {
                auto changedFrame = frame;
                auto changed = prepared;
                if (fault == 0) {
                    // Optics and full-frame grain have their own earlier rejection.
                    if (prepared.optics.flags != 0 || prepared.grain.hash != 0 || prepared.effects.hash != 0) {
                        continue;
                    }
                    --changedFrame.render_window.x2;
                    --changed.spatial_dir.render_extent.width;
                } else if (fault == 1) {
                    ++changed.spatial_dir.filter_domain_extent.x;
                } else if (fault == 2) {
                    ++changed.spatial_dir.filter_domain_extent.y;
                } else if (fault == 3) {
                    --changed.spatial_dir.filter_domain_extent.width;
                } else {
                    --changed.spatial_dir.filter_domain_extent.height;
                }
                constexpr std::array<const char*, 5> faultIds{"render-extent-width", "filter-domain-x", "filter-domain-y", "filter-domain-width", "filter-domain-height"};
                auto row = cohort.case_event(faultIds[fault]);
                check(changedFrame, submission, changed, "spatial_dir.frame");
                row.passed();
            }
        }
        if (prepared.grain.hash != 0) {
            require(state != nullptr, "grain frame probes require their held complete state");
            constexpr std::array<std::string_view, 5> faultIds{"session-seed", "clip-token", "pixel-size", "integral-time-frame-token", "fractional-time"};
            JuicerTests::Cohort faultCohort(invocationContext + "/frame-grain-bindings", std::vector<std::string>(faultIds.begin(), faultIds.end()));
            for (int fault = 0; fault < 5; ++fault) {
                auto changed = frame;
                auto identity = submission;
                switch (fault) {
                    case 0:
                        ++changed.session_seed;
                        break;
                    case 1:
                        ++changed.clip_token;
                        break;
                    case 2:
                        changed.pixel_size_um *= 2.0f;
                        break;
                    case 3:
                        changed.time_frames += 1.0;
                        ++identity.frame_token;
                        break;
                    case 4:
                        changed.time_frames += 0.25;
                        break;
                }
                auto row = faultCohort.case_event(faultIds[fault]);
                check(changed, identity, prepared, "grain.frame");
                row.passed();
            }
            // Rebuild only test inputs through the authoritative producer. Signed
            // and fractional times exercise both the frame token and grain phase.
            constexpr std::array<double, 5> times{-37.75, -1.0, -0.25, 0.0, 37.25};
            constexpr std::array<std::string_view, 5> timeIds{"negative-fractional", "negative-integral", "negative-quarter", "zero", "positive-fractional"};
            JuicerTests::Cohort timeCohort(invocationContext + "/frame-grain-time", std::vector<std::string>(timeIds.begin(), timeIds.end()));
            for (std::size_t index = 0; index < times.size(); ++index) {
                const double time = times[index];
                auto row = timeCohort.case_event(timeIds[index]);
                auto changedFrame = frame;
                changedFrame.time_frames = time;
                auto identity = submission;
                identity.frame_token = static_cast<std::uint64_t>(static_cast<std::int64_t>(std::floor(time)));
                const auto& render = prepared.grain.render_extent;
                const auto& full = prepared.grain.full_frame_extent;
                Spektrafilm::VisualGrainFrameDescriptor grain;
                std::string diagnostic;
                FjGrainFrameInput facts{};
                facts.render_extent = render;
                facts.full_frame_extent = full;
                facts.pixel_size_um = frame.pixel_size_um;
                facts.frame_time = time;
                facts.frame_rate = frame.frame_rate;
                facts.session_seed = frame.session_seed;
                facts.clip_token = frame.clip_token;
                const auto raw = JuicerTestSupport::grain_frame_input(*state, facts);
                const auto coefficients = JuicerTestSupport::grain_coefficients(state->grain);
                FjGrainFrameMath math{};
                FjGrainFrameFailure failure{};
                require(fj_test_grain_frame_prepare(&raw, &math, &failure, nullptr).category == FJ_STATUS_SUCCESS,
                        "produce temporal grain math");
                require(Spektrafilm::compose_visual_grain_frame_descriptor(
                            coefficients, static_cast<Spektrafilm::ProfilePolarity>(state->polarity), math, grain, diagnostic),
                        "produce temporal grain descriptor");
                JuicerCuda::PreparedDescriptors descriptors;
                descriptors.grainPayload = {coefficients, state->grain.hash, state->grain.layers_hash};
                descriptors.grain = grain;
                FjPreparedHostData encoded{};
                encode_prepared_descriptors(descriptors, encoded);
                auto changed = prepared;
                changed.grain = encoded.grain;
                check(changedFrame, identity, changed, nullptr);
                ++changed.grain.frame0;
                check(changedFrame, identity, changed, "grain.frame");
                changed.grain = encoded.grain;
                changed.grain.frame_alpha += 0.125f;
                check(changedFrame, identity, changed, "grain.frame");
                row.passed();
            }
        }
        std::cout << "frame bindings route=" << prepared.route << " grain=" << (prepared.grain.hash != 0) << " passed\n";
    }

    void check_grain_dispatch(FjCuda* cuda, const FjFrame& frame, const FjSubmission& submission, const FjPreparedHostData& current, const FjHostPreparation* hostPreparation) {
        require((current.grain.flags & FJ_GRAIN_ACTIVE) != 0 && current.film_grain == nullptr, "dispatch controls require a resolved Current frame");
        const std::string route = Spektrafilm::scan_route_is_print(static_cast<Spektrafilm::ScanRoute>(current.route)) ? "print" : "direct";
        JuicerTests::Cohort cases(kGrainDispatchCaller, std::vector<std::string>(kGrainDispatchCases.begin(), kGrainDispatchCases.end()));
        const std::vector<float> sums(JuicerCuda::FilmGrain::kBankEntries, 1.F);
        FjFilmGrain film{};
        film.bank_hash = 1;
        film.sums = {sums.data(), sums.size()};
        film.pitch = 1.F;
        film.gain = 1.F;

        std::array<char, 256> message{};
        FjErrorBuffer error{message.data(), message.size(), 0};
        const auto render = [&](const FjPreparedHostData& candidate, const FjSubmission& identity, FjAbortCallback abort = {}) {
            return render_raw_prepared(cuda, frame, identity, candidate, abort, &error, hostPreparation);
        };
        const auto diagnostic = [&] {
            return std::string_view(message.data(), error.length);
        };
        {
            auto row = cases.case_event("conflicting-models");
            auto conflict = current;
            conflict.film_grain = &film;
            AbortQuery abort;
            abort.cancelAt = 1;
            require_status(render(conflict, submission, {AbortQuery::query, &abort}), {FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0}, "conflicting models rejected by raw descriptor admission");
            require(abort.calls == 0 && diagnostic() == "conflicting grain models", "grain conflict must precede executor preparation");
            row.passed();
        }

        auto selectedFilm = current;
        selectedFilm.grain = {};
        selectedFilm.noise = {};
        selectedFilm.film_development.density_layers_hash = 0;
        for (auto& layer : selectedFilm.film_development.density_layers)
            std::fill(std::begin(layer), std::end(layer), FjFloatSpan{});
        selectedFilm.film_grain = &film;

        // The two models are separate instances, not conflicting memberships
        // for the same submission snapshot.
        auto filmSubmission = submission;
        filmSubmission.instance_token ^= std::uint64_t{1} << 63;
        const auto check_runtime_failure = [&](const FjPreparedHostData& candidate, const FjSubmission& identity, const std::string& prefix) {
            const auto outcome = render(candidate, identity);
            const std::string detail = prefix + " (got '" + std::string(diagnostic()) + "')";
            require_status(outcome, {FJ_STATUS_CUDA_FAILURE, FJ_API_CUDA_RUNTIME, cudaErrorInvalidValue}, detail.c_str());
            const std::string expected = prefix + ": " + cudaGetErrorString(cudaErrorInvalidValue);
            if (diagnostic() != expected)
                throw std::runtime_error("grain failure attribution: expected '" + expected + "', got '" + std::string(diagnostic()) + "'");
        };
        {
            auto row = cases.case_event("malformed-film-frame");
            film.pitch = std::numeric_limits<float>::quiet_NaN();
            check_runtime_failure(selectedFilm, filmSubmission, route + " Film grain binding failed: invalid frame");
            film.pitch = 1.F;
            struct GridAxis {
                double extent = 4.0;
                double step = 1.0;
                double origin = 0.0;
            };
            const auto& window = frame.render_window;
            const double width = static_cast<double>(window.x2) - window.x1;
            const double height = static_cast<double>(window.y2) - window.y1;
            const auto reject_grid = [&](GridAxis axis) {
                auto malformed = film;
                malformed.sample_pitch = 1.F;
                malformed.sample_extent[0] = axis.extent;
                malformed.sample_extent[1] = height;
                malformed.sample_step[0] = axis.step;
                malformed.sample_step[1] = 1.0;
                malformed.sample_origin[0] = axis.origin;
                auto candidate = selectedFilm;
                candidate.film_grain = &malformed;
                check_runtime_failure(candidate, filmSubmission, route + " Film grain binding failed: invalid frame");
            };
            reject_grid({.extent = std::numeric_limits<double>::infinity()});
            reject_grid({.step = std::numeric_limits<double>::quiet_NaN()});
            reject_grid({.origin = std::numeric_limits<double>::infinity()});
            reject_grid({.step = 0.0});
            reject_grid({.step = -1.0});
            reject_grid({.origin = 4.0});
            reject_grid({.extent = 262144.0});
            reject_grid({.step = std::numeric_limits<double>::max()});
            require(width > 2.0, "raw border controls need positive coverage beside two empty cells");
            reject_grid({.extent = width, .origin = -2.25});
            std::cout << route << " raw_film_grid=second-leading-empty rejected\n";
            reject_grid({.extent = width - 2.25});
            std::cout << route << " raw_film_grid=second-trailing-empty rejected\n";
            auto batched = film;
            batched.sample_pitch = 1.F;
            batched.sample_extent[0] = 273.0;
            batched.sample_extent[1] = 96.0;
            batched.sample_step[0] = batched.sample_step[1] = 91.0;
            batched.sample_origin[0] = 273.0;
            require(JuicerCuda::FilmGrain::proxy_sample_bound(batched.sample_extent, batched.sample_step) == 8649,
                    "malformed grid must be large enough to request batched workspace");
            auto candidate = selectedFilm;
            candidate.film_grain = &batched;
            const auto before = JuicerCuda::RenderTest::filmProxyReservations;
            check_runtime_failure(candidate, filmSubmission, route + " Film grain binding failed: invalid frame");
            const auto attempts = JuicerCuda::RenderTest::filmProxyReservations - before;
            std::cout << route << " malformed-batched-film-grid workspace_reservation_attempts=" << attempts << '\n';
            require(attempts == 0, "invalid batched grid attempted to reserve Film proxy workspace");
            auto malformedBorder = film;
            malformedBorder.calibrated_border[0] = 2;
            candidate.film_grain = &malformedBorder;
            check_runtime_failure(candidate, filmSubmission, route + " Film grain binding failed: invalid frame");
            auto mixed = batched;
            mixed.sample_origin[0] = 0.0;
            mixed.sample_extent[0] = width * mixed.sample_step[0];
            mixed.sample_extent[1] = height * mixed.sample_step[1];
            mixed.calibrated_border[0] = 1;
            candidate.film_grain = &mixed;
            check_runtime_failure(candidate, filmSubmission, route + " Film grain binding failed: invalid frame");
            require(JuicerCuda::RenderTest::filmProxyReservations == before,
                    "invalid calibrated border attempted to reserve Film proxy workspace");
            std::cout << route << " raw_film_grid=invalid-calibrated-border rejected mixed-grid-and-border rejected\n";
            auto full = film;
            full.gain = 0.F;
            candidate.film_grain = &full;
            require_status(render(candidate, filmSubmission), {FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, "zero-step Film frame must recover after rejecting the batched grid");
            require(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(frame.stream)) == cudaSuccess,
                    "complete Film recovery on the caller stream");
            std::cout << route << " raw_film_grid=zero-step-recovery passed\n";
            row.passed();
        }
        {
            auto row = cases.case_event("invalid-film-fraction");
            auto malformed = film;
            malformed.temporal_fraction = 1.F;
            auto candidate = selectedFilm;
            candidate.film_grain = &malformed;
            check_runtime_failure(candidate, filmSubmission, route + " Film grain binding failed: invalid frame");
            row.passed();
        }
        {
            auto row = cases.case_event("missing-film-bank");
            JuicerCuda::RenderTest::missingFilmBank = true;
            check_runtime_failure(selectedFilm, filmSubmission, route + " Film grain binding failed: missing frame bank");
            require(!JuicerCuda::RenderTest::missingFilmBank, "missing Film bank fault reached binding");
            row.passed();
        }
        {
            auto row = cases.case_event("film-launch");
            JuicerCuda::RenderTest::nullGrainPlane = true;
            // The real launch rejects the null C plane before enqueueing Film work.
            check_runtime_failure(selectedFilm, filmSubmission, route + " Film grain launch failed");
            require(!JuicerCuda::RenderTest::nullGrainPlane, "Film launch fault reached developed density dispatch");
            row.passed();
        }
        require_status(fj_cuda_retire_instance(cuda, filmSubmission.instance_token, &error), {FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, "retire raw Film instance");
        {
            auto row = cases.case_event("current-launch");
            JuicerCuda::RenderTest::nullGrainPlane = true;
            check_runtime_failure(current, submission, route + " visual grain launch failed");
            require(!JuicerCuda::RenderTest::nullGrainPlane, "Current launch fault reached developed density dispatch");
            row.passed();
        }
    }

    void check_spectral_packet_overlap(FjCuda* cuda, const FjCudaContext& context, const FjFrame& frame, const std::array<BoundScannerPacket, 2>& packets) {
        constexpr FjStatus success{FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
        constexpr FjStatus cancelled{FJ_STATUS_CANCELLED, FJ_API_NONE, 0};
        constexpr auto maximumHold = std::chrono::seconds(5);
        auto& root = JuicerCuda::borrow_root(cuda);
        const JuicerCuda::ResourceManager::DeviceContextKey key{context.device_id, reinterpret_cast<void*>(context.context)};
        const auto instance = packets[0].submission.instance_token ^ UINT64_C(0x2900000000000000);
        require(instance && frame.stream && frame.destination.row_bytes > 0 &&
                    frame.destination.bounds.y2 > frame.destination.bounds.y1 &&
                    packets[0].submission.scanner_hash != packets[1].submission.scanner_hash &&
                    packets[0].prepared.print_profile_asset_version != packets[1].prepared.print_profile_asset_version,
                "packet overlap needs distinct paper bindings and pitched destinations");
        const auto bytes = static_cast<std::size_t>(frame.destination.row_bytes) *
                           static_cast<std::size_t>(frame.destination.bounds.y2 - frame.destination.bounds.y1);
        struct OtherFrameStorage {
            cudaStream_t stream = nullptr;
            void* destination = nullptr;
            ~OtherFrameStorage() {
                if (stream)
                    (void)cudaStreamSynchronize(stream);
                (void)cudaFree(destination);
                if (stream)
                    (void)cudaStreamDestroy(stream);
            }
        } other;
        require(cudaStreamCreateWithFlags(&other.stream, cudaStreamNonBlocking) == cudaSuccess &&
                    cudaMalloc(&other.destination, bytes) == cudaSuccess,
                "allocate the second guarded destination on the same CUDA context");
        std::array frames{frame, frame};
        frames[1].stream = reinterpret_cast<std::uintptr_t>(other.stream);
        frames[1].destination.address = reinterpret_cast<std::uintptr_t>(other.destination);
        frames[1].time_frames += 1.0;
        const std::array streams{reinterpret_cast<cudaStream_t>(frame.stream), other.stream};
        std::vector<std::byte> guards(bytes);
        const auto download = [&](std::size_t lane, std::vector<std::byte>& raster) {
            require(cudaStreamSynchronize(streams[lane]) == cudaSuccess &&
                        cudaMemcpy(raster.data(), reinterpret_cast<const void*>(frames[lane].destination.address), bytes, cudaMemcpyDeviceToHost) == cudaSuccess,
                    "read the completed guarded scanner packet destination");
        };
        download(0, guards);
        const auto reset = [&](std::size_t lane) {
            require(cudaMemcpyAsync(reinterpret_cast<void*>(frames[lane].destination.address), guards.data(), bytes, cudaMemcpyHostToDevice, streams[lane]) == cudaSuccess &&
                        cudaStreamSynchronize(streams[lane]) == cudaSuccess,
                    "initialize scanner packet canaries before submission");
        };

        std::uint64_t next = 1;
        std::array<char, 1024> message{};
        FjErrorBuffer error{message.data(), message.size(), 0};
        const auto render = [&](std::size_t lane, FjStatus expected, JuicerCuda::RenderTest::PendingWork* hold = nullptr, FjAbortCallback abort = FjAbortCallback{}) {
            auto submitted = packets[lane].submission;
            submitted.instance_token = instance;
            submitted.submission_id = next++;
            JuicerCuda::RenderTest::scanConsumerHold = hold;
            const auto outcome = render_raw_prepared(cuda, frames[lane], submitted, packets[lane].prepared, abort, &error);
            const auto* const unused = std::exchange(JuicerCuda::RenderTest::scanConsumerHold, nullptr);
            const std::string detail = "spectral packet lifetime: " + std::string(message.data(), error.length);
            require_status(outcome, expected, detail.c_str());
            require(!unused, "the hold reached the actual final scanner launcher");
            return submitted.submission_id;
        };
        std::array reference{std::vector<std::byte>(bytes), std::vector<std::byte>(bytes)};
        for (std::size_t lane = 0; lane < frames.size(); ++lane) {
            reset(lane);
            (void)render(lane, success);
            download(lane, reference[lane]);
        }
        require(reference[0] != reference[1], "the independent paper requests have distinct delivered outputs");

        auto owner = JuicerProcess::TestSupport::RootLifetimeObserver::resources(root, key);
        require(static_cast<bool>(owner), "the isolated packet requests share a live frame owner");
        const auto epoch = owner->contextEpoch;
        const auto use_event = [&] {
            const std::lock_guard<std::mutex> lock(owner->m);
            require(!owner->pendingFrameUseEvents.empty(), "scanner packet has its original last-use fence");
            return static_cast<cudaEvent_t>(owner->pendingFrameUseEvents.back().eventOpaque);
        };
        struct PacketReadback {
            cudaEvent_t event = nullptr;
            const int* flag = nullptr;
        };
        const auto readback = [&](std::uint64_t id, const BoundScannerPacket& packet) {
            const std::lock_guard<std::mutex> lock(owner->m);
            const auto entry = std::find_if(owner->pendingScanErrorReadbacks.begin(), owner->pendingScanErrorReadbacks.end(), [id](const auto& holding) {
                return holding.identity.submissionId == id;
            });
            require(entry != owner->pendingScanErrorReadbacks.end() && entry->host && entry->eventOpaque &&
                        entry->identity.contextEpoch == epoch && entry->identity.scannerHash == packet.submission.scanner_hash &&
                        entry->identity.scannedProfileAssetVersion == packet.prepared.print_profile_asset_version,
                    "scanner packet retained the exact producing identity and flag");
            return PacketReadback{static_cast<cudaEvent_t>(entry->eventOpaque), entry->host};
        };
        reset(0);
        reset(1);
        std::vector<std::byte> actual(bytes);
        {
            JuicerCuda::RenderTest::PendingWork gate(streams[0], 0, maximumHold);
            const auto a = render(0, success, &gate);
            auto* const aUse = use_event();
            const auto aReadback = readback(a, packets[0]);
            require(cudaEventQuery(gate.event) == cudaErrorNotReady && cudaEventQuery(aUse) == cudaErrorNotReady &&
                        cudaEventQuery(aReadback.event) == cudaErrorNotReady,
                    "A scanner consumer and its retained events are still pending before B");
            const auto b = render(1, success);
            auto* const bUse = use_event();
            const auto bReadback = readback(b, packets[1]);
            const bool sameOwner = JuicerProcess::TestSupport::RootLifetimeObserver::resources(root, key) == owner;
            {
                const std::lock_guard<std::mutex> lock(owner->m);
                const bool retainedA = std::any_of(owner->pendingScanErrorReadbacks.begin(), owner->pendingScanErrorReadbacks.end(), [a](const auto& entry) {
                    return entry.identity.submissionId == a;
                });
                const bool retainedUse = aUse != bUse && std::any_of(owner->pendingFrameUseEvents.begin(), owner->pendingFrameUseEvents.end(), [aUse](const auto& entry) {
                                             return entry.eventOpaque == aUse;
                                         });
                std::cerr << "spectral packet B queued: same owner=" << sameOwner << " A use retained=" << retainedUse
                          << " A readback retained=" << retainedA << '\n';
                require(sameOwner && retainedUse && retainedA && aReadback.event != bReadback.event && aReadback.flag != bReadback.flag &&
                            cudaEventQuery(aUse) == cudaErrorNotReady && cudaEventQuery(aReadback.event) == cudaErrorNotReady,
                        "B queued its distinct paper packet before A's original use completed");
            }
            gate.released.store(true);
            for (std::size_t lane = 0; lane < frames.size(); ++lane) {
                download(lane, actual);
                require(actual == reference[lane], "the overlapped packet exactly matches its isolated raster and guards");
            }
            require(cudaEventQuery(aUse) == cudaSuccess && cudaEventQuery(bUse) == cudaSuccess &&
                        cudaEventQuery(aReadback.event) == cudaSuccess && cudaEventQuery(bReadback.event) == cudaSuccess &&
                        *aReadback.flag == 0 && *bReadback.flag == 0,
                    "both frame-use and producing scan events completed with zero flags");
            std::cout << "spectral packet overlap epoch=" << epoch << " A=" << a << " event=" << aReadback.event
                      << " B=" << b << " event=" << bReadback.event << " flags=0,0\n";
        }
        const auto reporter = render(0, success);
        download(0, actual);
        require(actual == reference[0], "same-owner reporter preserves A's completed output");
        {
            const std::lock_guard<std::mutex> lock(owner->m);
            require(owner->pendingScanErrorReadbacks.size() == 1 &&
                        owner->pendingScanErrorReadbacks.front().identity.submissionId == reporter,
                    "the reporter consumed both completed predecessors");
        }
        {
            JuicerCuda::RenderTest::PendingWork gate(streams[0], 0, maximumHold);
            AbortQuery abort;
            abort.cancelAt = 3;
            (void)render(0, cancelled, &gate, {AbortQuery::query, &abort});
            auto* const fence = use_event();
            require(abort.calls == 3 && abort.sameThread && cudaEventQuery(fence) == cudaErrorNotReady,
                    "post-enqueue abort retained the original stream's pending last-use event");
            gate.released.store(true);
            require(cudaStreamSynchronize(streams[0]) == cudaSuccess && cudaEventQuery(fence) == cudaSuccess,
                    "drain the original abort stream before retirement");
        }
        std::string retirement;
        require(root.retire_idle_context(context.device_id, reinterpret_cast<void*>(context.context), retirement), "retire the drained packet owner");
        owner.reset();
        (void)render(0, success);
        download(0, actual);
        const auto replacement = JuicerProcess::TestSupport::RootLifetimeObserver::resources(root, key);
        require(replacement && replacement->contextEpoch > epoch && actual == reference[0],
                "the newer exact-context epoch delivers the isolated reference again");
        require_status(fj_cuda_retire_instance(cuda, instance, &error), success, "retire scanner packet overlap fixture instance");
    }

    void check_scan_readback(FjCuda* cuda, const FjCudaContext& context, const FjFrame& frame, const FjSubmission& submission, const FjPreparedHostData& prepared, ReadbackProbe probe, const BoundScannerPacket* switchedReporter) {
        const auto route = static_cast<Spektrafilm::ScanRoute>(prepared.route);
        const bool receiver = probe == ReadbackProbe::FusedDir;
        JuicerTests::Cohort rows(receiver ? kDirScanReadbackCaller : probe == ReadbackProbe::Plane ? kPlaneScanReadbackCaller
                                                                                                   : kScanReadbackCaller,
                                 std::vector<std::string>(kScanReadbackCases.begin(), kScanReadbackCases.end()));
        constexpr FjStatus success{FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
        constexpr FjStatus preparationFailure{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        auto& root = JuicerCuda::borrow_root(cuda);
        const JuicerCuda::ResourceManager::DeviceContextKey key{context.device_id, reinterpret_cast<void*>(context.context)};
        auto* const stream = reinterpret_cast<cudaStream_t>(frame.stream);
        const auto* const destination = reinterpret_cast<const void*>(frame.destination.address);
        const auto& bounds = frame.destination.bounds;
        require(frame.destination.row_bytes > 0 && bounds.y2 > bounds.y1, "scan readback fixture uses a positive pitch and extent");
        const auto bytes = static_cast<std::size_t>(frame.destination.row_bytes) * static_cast<std::size_t>(bounds.y2 - bounds.y1);
        const auto read_image = [&](std::vector<std::byte>& image) {
            require(cudaStreamSynchronize(stream) == cudaSuccess &&
                        cudaMemcpy(image.data(), destination, image.size(), cudaMemcpyDeviceToHost) == cudaSuccess,
                    "scan readback fixture completed pitched output");
        };
        std::array<char, 1024> message{};
        FjErrorBuffer error{message.data(), message.size(), 0};
        auto baseline = submission;
        baseline.instance_token ^= UINT64_C(0x4000000000000000);
        require(baseline.instance_token != 0, "scan readback fixture instance token");
        require_status(render_raw_prepared(cuda, frame, baseline, prepared, {}, &error), success, "healthy scan before nonfinite injection");
        std::vector<std::byte> healthy(bytes), produced(bytes), compared(bytes);
        read_image(healthy);

        auto nonfinite = prepared;
        auto producer = baseline;
        ++producer.submission_id;
        if (receiver) {
            require(static_cast<DirNonlinearMode>(nonfinite.dir_couplers.mode) == DirNonlinearMode::PositiveReceiverLangmuir &&
                        nonfinite.dir_couplers.receiver_c_ref_rgb[1] != 0.0f && nonfinite.dir_couplers.receiver_kr_rgb[1] > 0.0f,
                    "positive DIR fixture binds an active green receiver");
            nonfinite.dir_couplers.receiver_kr_rgb[1] = 0.0f;
        } else {
            nonfinite.scanner_spectra.inverse_y_normalization = std::numeric_limits<float>::infinity();
            const auto& color = nonfinite.output_color;
            const auto gamut = (color.flags & FJ_COLOR_GAMUT_COMPRESSION) != 0 ? color.gamut_recipe_hash : 0;
            const auto binding = Hash::hash_uint64_values({JuicerCuda::spectral_binding_digest(nonfinite), gamut});
            producer.scanner_hash = JuicerCuda::ResourceManager::make_key_digests(nonfinite.recipe_hash, nonfinite.dir_couplers.hash, binding, nonfinite.auto_exposure.hash).scannerHash;
            require(producer.scanner_hash != submission.scanner_hash, "nonfinite scan changes the admitted binding");
        }
        const bool paper = Spektrafilm::scan_route_is_print(route);
        const bool fusedInline = probe == ReadbackProbe::Inline && paper && (prepared.scanner_post.flags & FJ_POST_GLARE) != 0u;
        const auto profile = paper ? prepared.print_profile_key : prepared.film_profile_key;
        const auto version = paper ? prepared.print_profile_asset_version : prepared.film_profile_asset_version;
        std::shared_ptr<JuicerCuda::Resources> owner;
        {
            auto row = rows.case_event("producer");
            const auto planes = JuicerCuda::ScanDensityCaptureTest::plane_scan_launches();
            const auto fused = JuicerCuda::ScanDensityCaptureTest::fused_dir_scan_launches();
            require_status(render_raw_prepared(cuda, frame, producer, nonfinite, {}, &error), success, "nonfinite producer finishes asynchronous submission");
            require(JuicerCuda::ScanDensityCaptureTest::plane_scan_launches() - planes == (probe == ReadbackProbe::Plane ? 1u : 0u) &&
                        JuicerCuda::ScanDensityCaptureTest::fused_dir_scan_launches() - fused == (receiver || fusedInline ? 1u : 0u),
                    "nonfinite producer used the expected scanner kernel entry");
            read_image(produced);
            require(produced != healthy, "nonfinite producer reaches the optical output");
            owner = JuicerProcess::TestSupport::RootLifetimeObserver::resources(root, key);
            require(static_cast<bool>(owner), "nonfinite producer retains its exact context owner");
            const std::lock_guard<std::mutex> lock(owner->m);
            require(owner->pendingScanErrorReadbacks.size() == 1, "one completed predecessor readback");
            const auto& held = owner->pendingScanErrorReadbacks.front();
            constexpr int greenReceiverFailure = (1 << 8) | (2 << 9);
            require(held.host && held.eventOpaque && cudaEventQuery(static_cast<cudaEvent_t>(held.eventOpaque)) == cudaSuccess &&
                        (receiver ? (*held.host & ~1) == greenReceiverFailure : *held.host == 1) &&
                        held.identity.submissionId == producer.submission_id &&
                        held.identity.frameToken == producer.frame_token && held.identity.scannerHash == producer.scanner_hash &&
                        held.identity.scannedProfileAssetVersion == version && held.identity.contextEpoch == owner->contextEpoch,
                    "actual arithmetic set the expected flags on the producing submission");
            std::cout << "scanner failure route=" << prepared.route << " plane=" << (probe == ReadbackProbe::Plane)
                      << " fused=" << (receiver || fusedInline) << " flags=" << *held.host
                      << " submission=" << held.identity.submissionId << '\n';
            row.passed();
        }
        auto reportingFrame = frame;
        reportingFrame.time_frames += 1.0;
        const auto& reporterPrepared = switchedReporter ? switchedReporter->prepared : prepared;
        auto reporter = switchedReporter ? switchedReporter->submission : baseline;
        reporter.instance_token = baseline.instance_token;
        reporter.submission_id = producer.submission_id + 1;
        reporter.frame_token = producer.frame_token + 1;
        if (switchedReporter) {
            require(reporter.scanner_hash != baseline.scanner_hash && reporterPrepared.film_profile_asset_version != prepared.film_profile_asset_version &&
                        (!paper || reporterPrepared.print_profile_asset_version != prepared.print_profile_asset_version),
                    "reporter holds a different bound capture film and scanned medium");
        }
        {
            auto row = rows.case_event("reporter");
            const auto outcome = render_raw_prepared(cuda, reportingFrame, reporter, reporterPrepared, {}, &error);
            require_status(outcome.status, preparationFailure, "distinct raw reporter consumes the earlier scan failure");
            const std::string expected = receiver ? "earlier CUDA DIR submission failed component=dir route=" +
                                                        std::string(Spektrafilm::scan_route_key(route)) +
                                                        " profile=" + std::string(prepared.film_profile_key.data, prepared.film_profile_key.count) +
                                                        " polarity=" + std::to_string(prepared.capture_polarity) +
                                                        " recipe_hash=" + std::to_string(prepared.dir_couplers.hash) +
                                                        " descriptor_hash=" + std::to_string(prepared.spatial_dir.hash) +
                                                        " context_epoch=" + std::to_string(owner->contextEpoch) +
                                                        " channel_mask=2 requirement=finite_defined_receiver_arithmetic"
                                                  : "earlier CUDA scan produced non-finite RGB route=" +
                                                        std::string(Spektrafilm::scan_route_key(route)) +
                                                        (paper ? " medium=paper profile=" : " medium=film profile=") +
                                                        std::string(profile.data, profile.count) +
                                                        " profile_asset_version=" + std::to_string(version) +
                                                        " submission_id=" + std::to_string(producer.submission_id) +
                                                        " frame_token=" + std::to_string(producer.frame_token) +
                                                        " scanner_hash=" + std::to_string(producer.scanner_hash) +
                                                        " context_epoch=" + std::to_string(owner->contextEpoch);
            require(outcome.flags == (receiver ? FJ_RENDER_DEFERRED_DIR_ERROR : 0u) &&
                        std::string_view(message.data(), error.length) == expected,
                    "raw reporter returns producing route and medium provenance with the expected DIR outcome");
            read_image(compared);
            require(compared == produced, "failed reporter preserves the entire pitched destination");
            const std::lock_guard<std::mutex> lock(owner->m);
            require(owner->pendingScanErrorReadbacks.empty(), "failed reporter retires the sole readback");
            row.passed();
        }
        {
            auto row = rows.case_event("retry");
            auto retry = baseline;
            retry.submission_id = reporter.submission_id + 1;
            const auto outcome = render_raw_prepared(cuda, frame, retry, prepared, {}, &error);
            const auto retryDetail = "scan retry: " + std::string(message.data(), error.length);
            require_status(outcome, success, retryDetail.c_str());
            read_image(compared);
            require(compared == healthy && JuicerProcess::TestSupport::RootLifetimeObserver::resources(root, key) == owner,
                    "retry restores the guarded pitched image on the same context owner");
            require_status(fj_cuda_retire_instance(cuda, baseline.instance_token, &error), success, "retire scanner readback fixture instance");
            row.passed();
        }
    }

    FjStatus check_render_contract(FjCuda* cuda, const FjCudaContext& context, const FjFrame& frame, const FjSubmission& submission, const FjPreparedHostData& prepared, FjErrorBuffer* error, const std::string& invocationContext) {
        JuicerProcess::Root& root = JuicerCuda::borrow_root(cuda);
        constexpr FjStatus unsupported{FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0};
        constexpr FjStatus preparationFailure{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        constexpr FjStatus success{FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
        constexpr FjStatus cancelled{FJ_STATUS_CANCELLED, FJ_API_NONE, 0};
        require(prepared.spatial_dir.hash != 0 && prepared.optics.flags == 0 && prepared.grain.hash == 0 && prepared.effects.hash == 0, "render contract exercises partial spatial DIR admission");
        const auto render = [&](FjAbortCallback callback = {}) {
            return render_raw_prepared(cuda, frame, submission, prepared, callback, error);
        };
        AbortQuery beforeEnqueue;
        beforeEnqueue.cancelAt = 1;
        require(JuicerProcess::TestSupport::RootLifetimeObserver::fences(root) == 0, "cold cancellation starts without resources");
        require_status(render({AbortQuery::query, &beforeEnqueue}), cancelled, "cancel before first preparation");
        require(beforeEnqueue.calls == 1 && beforeEnqueue.sameThread, "pre-enqueue callback");
        require(JuicerProcess::TestSupport::RootLifetimeObserver::fences(root) == 0, "pre-enqueue cancellation created no frame fence");
        constexpr std::array<const char*, 7> argumentFaultIds{"null-owner", "null-context", "null-frame", "null-submission", "null-prepared", "malformed-error-buffer", "null-error-output"};
        JuicerTests::Cohort argumentFaults(invocationContext + "/render-null-arguments", std::vector<std::string>(argumentFaultIds.begin(), argumentFaultIds.end()));
        for (int fault = 0; fault < 7; ++fault) {
            auto row = argumentFaults.case_event(argumentFaultIds[fault]);
            FjErrorBuffer malformedError{nullptr, 4, 92};
            const auto result = fault == 6
                                    ? render_raw_prepared(cuda, frame, submission, prepared, {}, nullptr)
                                    : fj_cuda_render(fault == 0 ? nullptr : cuda, fault == 1 ? nullptr : &context, fault == 2 ? nullptr : &frame, fault == 3 ? nullptr : &submission, fault == 4 ? nullptr : &prepared, {}, fault == 5 ? &malformedError : error);
            require_status(result, fault == 6 ? success : unsupported, "null argument");
            row.passed();
        }
        require_status(fj_cuda_render(reinterpret_cast<FjCuda*>(std::uintptr_t{1}), &context, &frame, &submission, &prepared, {}, error), preparationFailure, "unregistered owner");
        constexpr std::array<const char*, 4> contextFaultIds{"invalid-device", "null-context-handle", "different-device", "different-context"};
        JuicerTests::Cohort contextFaults(invocationContext + "/render-context-identity", std::vector<std::string>(contextFaultIds.begin(), contextFaultIds.end()));
        for (int fault = 0; fault < 4; ++fault) {
            auto row = contextFaults.case_event(contextFaultIds[fault]);
            auto changed = context;
            if (fault == 0)
                changed.device_id = -1;
            if (fault == 1)
                changed.context = 0;
            if (fault == 2)
                ++changed.device_id;
            if (fault == 3)
                ++changed.context;
            require_status(fj_cuda_render(cuda, &changed, &frame, &submission, &prepared, {}, error), unsupported, "context identity");
            row.passed();
        }
        constexpr std::array<const char*, 9> submissionFaultIds{"zero-instance-token", "zero-submission-id", "different-frame-token", "different-upload-hash", "different-dir-hash", "zero-scanner-hash", "different-auto-exposure-hash", "zero-upload-hash", "zero-auto-exposure-hash"};
        JuicerTests::Cohort submissionFaults(invocationContext + "/render-submission-identity", std::vector<std::string>(submissionFaultIds.begin(), submissionFaultIds.end()));
        for (int fault = 0; fault < 9; ++fault) {
            auto row = submissionFaults.case_event(submissionFaultIds[fault]);
            auto changed = submission;
            switch (fault) {
                case 0:
                    changed.instance_token = 0;
                    break;
                case 1:
                    changed.submission_id = 0;
                    break;
                case 2:
                    ++changed.frame_token;
                    break;
                case 3:
                    ++changed.upload_core_hash;
                    break;
                case 4:
                    ++changed.dir_hash;
                    break;
                case 5:
                    changed.scanner_hash = 0;
                    break;
                case 6:
                    ++changed.auto_exposure_hash;
                    break;
                case 7:
                    changed.upload_core_hash = 0;
                    break;
                case 8:
                    changed.auto_exposure_hash = 0;
                    break;
            }
            require_status(fj_cuda_render(cuda, &context, &frame, &changed, &prepared, {}, error), unsupported, "submission identity");
            row.passed();
        }
        constexpr std::array<const char*, 13> frameFaultIds{"null-source-address", "negative-row-bytes", "reserved-flags", "missing-stream-flag", "nan-time", "overflow-time", "negative-pixel-size", "negative-frame-rate", "empty-render-window", "empty-full-frame", "unsupported-components", "unsupported-depth", "aliased-source-destination"};
        JuicerTests::Cohort frameFaults(invocationContext + "/render-frame-input", std::vector<std::string>(frameFaultIds.begin(), frameFaultIds.end()));
        for (int fault = 0; fault < 13; ++fault) {
            auto row = frameFaults.case_event(frameFaultIds[fault]);
            auto changed = frame;
            switch (fault) {
                case 0:
                    changed.source.address = 0;
                    break;
                case 1:
                    changed.destination.row_bytes = -1;
                    break;
                case 2:
                    changed.flags |= 128;
                    break;
                case 3:
                    changed.flags &= ~FJ_FRAME_STREAM_PRESENT;
                    break;
                case 4:
                    changed.time_frames = std::numeric_limits<double>::quiet_NaN();
                    break;
                case 5:
                    changed.time_frames = 0x1p63;
                    break;
                case 6:
                    changed.pixel_size_um = -1;
                    break;
                case 7:
                    changed.frame_rate = -1;
                    break;
                case 8:
                    changed.render_window.x2 = changed.render_window.x1;
                    break;
                case 9:
                    changed.full_frame_extent = {};
                    break;
                case 10:
                    changed.destination.components = 2;
                    break;
                case 11:
                    changed.source.depth = 2;
                    break;
                case 12:
                    changed.source.address = changed.destination.address;
                    break;
            }
            require_status(fj_cuda_render(cuda, &context, &changed, &submission, &prepared, {}, error), unsupported, "frame input");
            row.passed();
        }
        constexpr std::array<const char*, 10> preparedFaultIds{"unknown-route", "different-recipe-hash", "null-sensitivity", "short-density-count", "different-preview-width", "different-source-bounds", "different-scanner-normalization", "invalid-profile-key", "reserved-scanner-post-flags", "reserved-output-color-flags"};
        JuicerTests::Cohort preparedFaults(invocationContext + "/render-prepared-input", std::vector<std::string>(preparedFaultIds.begin(), preparedFaultIds.end()));
        for (int fault = 0; fault < 10; ++fault) {
            auto row = preparedFaults.case_event(preparedFaultIds[fault]);
            auto changed = prepared;
            switch (fault) {
                case 0:
                    changed.route = 99;
                    break;
                case 1:
                    ++changed.recipe_hash;
                    break;
                case 2:
                    changed.film_exposure.sensitivity_rgb.data = nullptr;
                    break;
                case 3:
                    --changed.film_development.density_rgb.count;
                    break;
                case 4:
                    ++changed.auto_exposure.preview_width;
                    break;
                case 5:
                    ++changed.auto_exposure.source_bounds.x1;
                    break;
                case 6:
                    changed.scanner_spectra.inverse_y_normalization = std::bit_cast<float>(
                        std::bit_cast<std::uint32_t>(changed.scanner_spectra.inverse_y_normalization) ^ 1U);
                    break;
                case 7:
                    changed.film_profile_key = {nullptr, 1};
                    break;
                case 8:
                    changed.scanner_post.flags |= 128;
                    break;
                case 9:
                    changed.output_color.flags |= 128;
                    break;
            }
            require_status(fj_cuda_render(cuda, &context, &frame, &submission, &changed, {}, error), unsupported, "prepared input");
            row.passed();
        }
        // The reinspection must reject a different current context, even if the
        // caller supplies the identity returned by an earlier successful inspect.
        CUcontext original = nullptr;
        CUcontext other = nullptr;
        require(cuCtxGetCurrent(&original) == CUDA_SUCCESS, "get original context");
        require(cuCtxCreate(&other, nullptr, 0, context.device_id) == CUDA_SUCCESS, "create fixture context");
        const auto wrongCurrent = render();
        CUstream foreignStream = nullptr;
        require(cuStreamCreate(&foreignStream, CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS, "create foreign stream");
        require(cuCtxSetCurrent(original) == CUDA_SUCCESS, "restore original context");
        auto wrongStreamFrame = frame;
        wrongStreamFrame.stream = reinterpret_cast<std::uintptr_t>(foreignStream);
        const auto wrongStream = fj_cuda_render(cuda, &context, &wrongStreamFrame, &submission, &prepared, {}, error);
        require(cuCtxSetCurrent(other) == CUDA_SUCCESS, "select fixture context for cleanup");
        require(cuStreamDestroy(foreignStream) == CUDA_SUCCESS, "destroy fixture stream");
        require(cuCtxDestroy(other) == CUDA_SUCCESS, "destroy fixture context");
        require(cuCtxSetCurrent(original) == CUDA_SUCCESS, "restore current context after cleanup");
        require_status(wrongCurrent, unsupported, "actual current context");
        require_status(wrongStream, unsupported, "foreign stream context");

        using JuicerCuda::RenderTest::Injection;
        constexpr std::array<Injection, 4> injections{Injection::Allocation, Injection::Standard, Injection::Unknown, Injection::Typed};
        constexpr std::array<const char*, 4> injectionIds{"allocation", "standard", "unknown", "typed"};
        JuicerTests::Cohort injectionCases(invocationContext + "/render-exception-injection", std::vector<std::string>(injectionIds.begin(), injectionIds.end()));
        for (std::size_t index = 0; index < injections.size(); ++index) {
            const auto injection = injections[index];
            auto row = injectionCases.case_event(injectionIds[index]);
            JuicerCuda::RenderTest::injection = injection;
            JuicerCuda::RenderTest::injectedStatus = JuicerCuda::driver_failure_status(CUDA_ERROR_CONTEXT_IS_DESTROYED);
            const auto result = render();
            JuicerCuda::RenderTest::injection = Injection::None;
            const auto expected = injection == Injection::Typed ? JuicerCuda::RenderTest::injectedStatus
                                                                : FjStatus{injection == Injection::Allocation ? FJ_STATUS_ALLOCATION_FAILURE : FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
            require_status(result, expected, "C exception containment");
            require(error->length > 0, "owned exception diagnostic");
            row.passed();
        }
        const std::array<FjStatus, 5> typedStatuses{preparationFailure, JuicerCuda::runtime_failure_status(cudaErrorMemoryAllocation), JuicerCuda::runtime_failure_status(cudaErrorInvalidValue), JuicerCuda::cufft_failure_status(6), JuicerCuda::driver_failure_status(CUDA_ERROR_CONTEXT_IS_DESTROYED)};
        constexpr std::array<const char*, 5> typedStatusIds{"preparation-failure", "runtime-allocation", "runtime-invalid-value", "cufft", "driver-context-lost"};
        constexpr std::array<std::size_t, 4> capacities{0, 1, 8, 64};
        constexpr std::array<const char*, 4> capacityIds{"zero", "one", "eight", "sixty-four"};
        std::vector<std::string> typedCapacityIds;
        for (const char* statusId : typedStatusIds)
            for (const char* capacityId : capacityIds)
                typedCapacityIds.emplace_back(std::string(statusId) + "/" + capacityId);
        JuicerTests::Cohort typedCapacities(invocationContext + "/render-typed-status-capacity", typedCapacityIds);
        for (std::size_t statusIndex = 0; statusIndex < typedStatuses.size(); ++statusIndex) {
            const auto typed = typedStatuses[statusIndex];
            for (std::size_t capacityIndex = 0; capacityIndex < capacities.size(); ++capacityIndex) {
                const auto capacity = capacities[capacityIndex];
                auto row = typedCapacities.case_event(std::string(typedStatusIds[statusIndex]) + "/" + capacityIds[capacityIndex]);
                std::array<char, 64> message{};
                FjErrorBuffer output{capacity == 0 ? nullptr : message.data(), capacity, 92};
                JuicerCuda::RenderTest::injection = Injection::Typed;
                JuicerCuda::RenderTest::injectedStatus = typed;
                const auto result = render_raw_prepared(cuda, frame, submission, prepared, {}, &output);
                JuicerCuda::RenderTest::injection = Injection::None;
                require_status(result, typed, "status independent of diagnostic capacity");
                const std::string full = "render injection 100% diagnostic";
                const auto length = capacity == 0 ? 0 : std::min(capacity - 1, full.size());
                require(output.length == length, "diagnostic length");
                require(capacity == 0 || (message[length] == '\0' && full.compare(0, length, message.data()) == 0), "diagnostic contents");
                row.passed();
            }
        }
        constexpr std::array<bool, 2> deferredCases{false, true};
        constexpr std::array<const char*, 2> deferredIds{"not-deferred", "deferred"};
        std::vector<std::string> deferredCapacityIds;
        for (const char* deferredId : deferredIds)
            for (const char* capacityId : capacityIds)
                deferredCapacityIds.emplace_back(std::string(deferredId) + "/" + capacityId);
        JuicerTests::Cohort deferredCapacities(invocationContext + "/render-deferred-capacity", deferredCapacityIds);
        for (std::size_t deferredIndex = 0; deferredIndex < deferredCases.size(); ++deferredIndex) {
            const bool deferred = deferredCases[deferredIndex];
            for (std::size_t capacityIndex = 0; capacityIndex < capacities.size(); ++capacityIndex) {
                const auto capacity = capacities[capacityIndex];
                auto row = deferredCapacities.case_event(std::string(deferredIds[deferredIndex]) + "/" + capacityIds[capacityIndex]);
                std::array<char, 64> text{};
                FjErrorBuffer output{text.data(), capacity, 0};
                JuicerCuda::RenderTest::injection = Injection::Typed;
                JuicerCuda::RenderTest::injectedStatus = preparationFailure;
                JuicerCuda::RenderTest::injectedDeferredDirError = deferred;
                const auto outcome = render_raw_prepared(cuda, frame, submission, prepared, {}, &output);
                JuicerCuda::RenderTest::injection = Injection::None;
                JuicerCuda::RenderTest::injectedDeferredDirError = false;
                require_status(outcome.status, preparationFailure, "typed deferred DIR status");
                require(outcome.flags == (deferred ? FJ_RENDER_DEFERRED_DIR_ERROR : 0U), "deferred provenance independent of diagnostic capacity and text");
                row.passed();
            }
        }
        // EQUAL is prepared before publication. Its former late exception
        // transport is covered here without adding preparation work to render.
        constexpr std::array<std::uint32_t, 3> categories{FJ_STATUS_UNSUPPORTED_INPUT, FJ_STATUS_INTERNAL_FAILURE, FJ_STATUS_ALLOCATION_FAILURE};
        constexpr std::array<const char*, 3> categoryIds{"unsupported-input", "internal-failure", "allocation-failure"};
        constexpr std::array<bool, 2> completionCases{false, true};
        constexpr std::array<const char*, 2> completionIds{"completion-success", "completion-failure"};
        std::vector<std::string> categoryCompletionIds;
        for (const char* categoryId : categoryIds)
            for (const char* completionId : completionIds)
                categoryCompletionIds.emplace_back(std::string(categoryId) + "/" + completionId);
        JuicerTests::Cohort categoryCompletionCases(invocationContext + "/render-absent-stream-completion", categoryCompletionIds);
        for (std::size_t categoryIndex = 0; categoryIndex < categories.size(); ++categoryIndex) {
            const auto category = categories[categoryIndex];
            for (std::size_t completionIndex = 0; completionIndex < completionCases.size(); ++completionIndex) {
                const bool completionFailure = completionCases[completionIndex];
                auto row = categoryCompletionCases.case_event(std::string(categoryIds[categoryIndex]) + "/" + completionIds[completionIndex]);
                auto absentStream = frame;
                absentStream.flags &= ~FJ_FRAME_STREAM_PRESENT;
                absentStream.stream = 0;
                JuicerCuda::RenderTest::completionCalls = 0;
                JuicerCuda::RenderTest::injection = Injection::Typed;
                JuicerCuda::RenderTest::injectedStatus = {category, FJ_API_NONE, 0};
                JuicerCuda::RenderTest::completionOverride = completionFailure ? static_cast<int>(cudaErrorInvalidValue) : 0;
                const auto outcome = render_raw_prepared(cuda, absentStream, submission, prepared, {}, error);
                JuicerCuda::RenderTest::injection = Injection::None;
                const FjStatus expected = completionFailure
                                              ? JuicerCuda::runtime_failure_status(cudaErrorInvalidValue)
                                              : FjStatus{category, FJ_API_NONE, 0};
                require_status(outcome, expected, "native exception/absent-stream completion precedence");
                require(JuicerCuda::RenderTest::completionCalls == 1 && JuicerCuda::RenderTest::lastCompletion == cudaSuccess,
                        "actual absent-stream completion precedes its test-only status override");
                require(cudaStreamQuery(nullptr) == cudaSuccess, "native exception unwind completes absent-stream work");
                row.passed();
            }
        }
        // The production adapter uses this same admitted body after inspection
        // and its short latch; an exported call under the admission must still fail.
        {
            JuicerCuda::NativeCall admitted(cuda);
            FjCudaContext inspected{};
            require_status(admitted.inspect(&frame, &inspected, error), success, "admitted inspection");
            require_status(render(), unsupported, "export cannot bypass an existing admission");
            JuicerCuda::PendingContextLossRecovery recovery;
            std::string diagnostic;
            require_status(admitted.render(&inspected, &frame, &submission, &prepared, {}, recovery, diagnostic), success, "one inspection and render admission");
            require(!recovery.pending, "ordinary render schedules no host recovery");
        }
        constexpr std::array<unsigned, 3> checkpoints{1, 2, 3};
        constexpr std::array<const char*, 3> checkpointIds{"before-enqueue", "after-enqueue", "after-execute"};
        JuicerTests::Cohort cancellationCheckpoints(invocationContext + "/render-cancellation-checkpoints", std::vector<std::string>(checkpointIds.begin(), checkpointIds.end()));
        for (std::size_t checkpointIndex = 0; checkpointIndex < checkpoints.size(); ++checkpointIndex) {
            const unsigned checkpoint = checkpoints[checkpointIndex];
            auto row = cancellationCheckpoints.case_event(checkpointIds[checkpointIndex]);
            AbortQuery callback;
            callback.cancelAt = checkpoint;
            {
                JuicerCuda::RenderTest::PendingWork pending(reinterpret_cast<cudaStream_t>(frame.stream), checkpoint);
                // Only post-preparation checkpoints are delayed. The abort query
                // itself never enqueues work or changes native ownership.
                JuicerCuda::RenderTest::pendingWork = checkpoint > 1 ? &pending : nullptr;
                const auto result = render({AbortQuery::query, &callback});
                JuicerCuda::RenderTest::pendingWork = nullptr;
                require_status(result, cancelled, "bounded cancellation");
                require(callback.calls == checkpoint && callback.sameThread, "callback count and calling thread");
                if (checkpoint > 1) {
                    require(pending.observedPending, "GPU work was pending at cancellation");
                    const auto completion = cudaEventQuery(pending.event);
                    require(completion == cudaSuccess || completion == cudaErrorNotReady, "cancelled work completion query");
                    // The accepted scratch-shed path may synchronize on abort.
                    // Otherwise every outstanding frame must retain its fence.
                    if (completion == cudaErrorNotReady) {
                        require(JuicerProcess::TestSupport::RootLifetimeObserver::fences(root) != 0, "aborted frame retains its use fence");
                    }
                    std::cout << "cancel checkpoint=" << checkpoint << " completed=" << (completion == cudaSuccess) << '\n';
                }
            }
            const auto calls = callback.calls;
            require_status(render(), success, "successful render after cancellation");
            require(callback.calls == calls, "callback retained after return");
            row.passed();
        }
        AbortQuery invalid;
        invalid.cancelAt = 1;
        invalid.result = 7;
        require_status(render({AbortQuery::query, &invalid}), unsupported, "unknown abort return is not cancellation");
        AbortQuery callback;
        callback.reentry = true;
        callback.cuda = cuda;
        callback.frame = &frame;
        require_status(render({AbortQuery::query, &callback}), success, "continuing query");
        require(callback.calls == 3 && callback.sameThread && callback.reentryRejected, "bounded same-thread query without reentry");
        require_status(render({nullptr, &callback}), success, "null callback with unused user pointer");
        require(callback.calls == 3, "null query invoked or callback retained");

        // Isolate the C entry's return-time contract from schedule-internal
        // synchronization by queuing finite work after the executor completes.
        constexpr std::array<const char*, 3> streamIds{"present-supplied-stream", "present-default-stream", "absent-default-stream"};
        JuicerTests::Cohort streamCases(invocationContext + "/render-stream-presence", std::vector<std::string>(streamIds.begin(), streamIds.end()));
        for (int streamCase = 0; streamCase < 3; ++streamCase) {
            auto row = streamCases.case_event(streamIds[streamCase]);
            auto streamFrame = frame;
            if (streamCase != 0)
                streamFrame.stream = 0;
            if (streamCase == 2)
                streamFrame.flags &= ~FJ_FRAME_STREAM_PRESENT;
            JuicerCuda::RenderTest::PendingWork pending(reinterpret_cast<cudaStream_t>(streamFrame.stream), 0);
            JuicerCuda::RenderTest::completionWork = &pending;
            const auto result = render_raw_prepared(cuda, streamFrame, submission, prepared, {}, error);
            JuicerCuda::RenderTest::completionWork = nullptr;
            require_status(result, success, "stream-presence return contract");
            require(pending.observedPending, "stream completion probe actually queued pending work");
            require(cudaEventQuery(pending.event) == (streamCase == 2 ? cudaSuccess : cudaErrorNotReady), "supplied streams return asynchronously, absence completes");
            row.passed();
        }
        // Explicit default-stream and absent-stream calls both execute; absence
        // additionally guarantees completion before caller storage/image release.
        constexpr std::array<bool, 2> suppliedCases{true, false};
        constexpr std::array<const char*, 2> suppliedIds{"stream-supplied", "stream-absent"};
        JuicerTests::Cohort suppliedStreamCases(invocationContext + "/render-default-stream", std::vector<std::string>(suppliedIds.begin(), suppliedIds.end()));
        for (std::size_t suppliedIndex = 0; suppliedIndex < suppliedCases.size(); ++suppliedIndex) {
            const bool supplied = suppliedCases[suppliedIndex];
            auto row = suppliedStreamCases.case_event(suppliedIds[suppliedIndex]);
            auto defaultFrame = frame;
            defaultFrame.stream = 0;
            if (!supplied)
                defaultFrame.flags &= ~FJ_FRAME_STREAM_PRESENT;
            require_status(render_raw_prepared(cuda, defaultFrame, submission, prepared, {}, error), success, "default stream render");
            if (!supplied)
                require(cudaStreamQuery(nullptr) == cudaSuccess, "absent stream must complete before return");
            AbortQuery cancel;
            cancel.cancelAt = 2;
            require_status(render_raw_prepared(cuda, defaultFrame, submission, prepared, {AbortQuery::query, &cancel}, error), cancelled, "default stream cancellation");
            if (!supplied)
                require(cudaStreamQuery(nullptr) == cudaSuccess, "absent stream abort completion");
            row.passed();
        }
        constexpr std::array<bool, 2> uploadStorageCases{false, true};
        constexpr std::array<const char*, 2> uploadStorageIds{"borrowed-prepared-storage", "rust-tc-allocation"};
        JuicerTests::Cohort uploadStorage(invocationContext + "/render-expiring-upload-storage", std::vector<std::string>(uploadStorageIds.begin(), uploadStorageIds.end()));
        for (std::size_t storageIndex = 0; storageIndex < uploadStorageCases.size(); ++storageIndex) {
            const bool rustTc = uploadStorageCases[storageIndex];
            auto row = uploadStorage.case_event(uploadStorageIds[storageIndex]);
            const JuicerCuda::ResourceManager::DeviceContextKey uploadKey{context.device_id, reinterpret_cast<void*>(context.context)};
            JuicerCuda::RenderTest::ExpiringUpload upload;
            upload.resources = JuicerProcess::TestSupport::RootLifetimeObserver::resources(root, uploadKey);
            require(upload.resources != nullptr, "expiry has a real native resource owner");
            upload.stream = reinterpret_cast<cudaStream_t>(frame.stream);
            require(cudaMalloc(&upload.first, upload.kBytes) == cudaSuccess &&
                        cudaMalloc(&upload.second, upload.kBytes) == cudaSuccess,
                    "allocate expiry copy workspace");
            require(cudaMemsetAsync(upload.first, 0x26, upload.kBytes, upload.stream) == cudaSuccess, "initialize finite expiry work");
            const auto original = prepared.film_development.density_rgb;
            // The original span can be native triplet rows. Copy its object
            // representation into actual contiguous C scalars without indexing it.
            std::vector<float> borrowed(original.count);
            std::memcpy(borrowed.data(), original.data, original.count * sizeof(float));
            struct TcOwner {
                FjOwnedFilmTcLut record{};
                ~TcOwner() noexcept {
                    (void)fj_test_reconstruction_release_tc_lut(&record);
                }
            } tcOwner;
            if (rustTc) {
                std::vector<float> spectra(std::size_t{192} * 192u * 81u, 1.0f);
                std::array<float, 243> sensitivity{};
                sensitivity.fill(1.0f);
                std::array<float, 81> spd{};
                spd.fill(1.0f);
                const FjFilmTcLutInput input{{spectra.data(), spectra.size()}, {sensitivity.data(), sensitivity.size()}, {spd.data(), spd.size()}, {1, 1, 1}, 0, 2, 0, {nullptr, 0}, 0, 0, {0, 0}, {nullptr, 0}};
                require(fj_test_reconstruction_tc_lut(&input, &tcOwner.record, error).category == FJ_STATUS_SUCCESS, "real Rust TC allocation for native staging expiry");
                upload.source = tcOwner.record.samples;
            } else
                upload.source = {borrowed.data(), borrowed.size()};
            const float* sourceAddress = upload.source.data;
            const std::vector<float> expected(upload.source.data, upload.source.data + upload.source.count);
            JuicerCuda::Failure warmFailure;
            require(JuicerCuda::PinnedUploadTest::upload(*upload.resources, upload.second, upload.source.data, upload.source.count * sizeof(float), upload.stream, warmFailure), "warm native expiry staging");
            require(cudaStreamSynchronize(upload.stream) == cudaSuccess, "complete expiry warmup before measured call");
            JuicerCuda::PinnedUploadTest::poll(uploadKey);
            auto expired = prepared;
            if (!rustTc)
                expired.film_development.density_rgb = upload.source;
            JuicerCuda::RenderTest::expiringUpload = &upload;
            const auto outcome = render_raw_prepared(cuda, frame, submission, expired, {}, error);
            JuicerCuda::RenderTest::expiringUpload = nullptr;
            require_status(outcome, success, "render with expiring borrowed storage");
            require(upload.staging.inFlight != 0, "native upload remains outstanding at C return");
            cudaEvent_t event = nullptr;
            for (const auto& block : upload.staging.blocks) {
                if (block.capacity >= expected.size() * sizeof(float) &&
                    cudaEventQuery(static_cast<cudaEvent_t>(block.event)) == cudaErrorNotReady &&
                    std::memcmp(block.pointer, expected.data(), expected.size() * sizeof(float)) == 0) {
                    require(block.pointer != sourceAddress, "native staging does not borrow the caller allocation");
                    event = static_cast<cudaEvent_t>(block.event);
                    break;
                }
            }
            require(event != nullptr, "actual native staging event is pending");
            if (rustTc) {
                require(fj_test_reconstruction_release_tc_lut(&tcOwner.record).category == FJ_STATUS_SUCCESS && !tcOwner.record.samples.data && tcOwner.record.capacity == 0, "Rust TC allocation released before native upload completes");
            } else {
                std::fill(borrowed.begin(), borrowed.end(), -8192.0f);
                std::vector<float>().swap(borrowed);
            }
            expired = {};
            upload.source = {};
            require(cudaEventQuery(event) == cudaErrorNotReady, "caller storage expires before upload completion");
            require(cudaEventSynchronize(event) == cudaSuccess, "complete native expiry upload after caller release");
            std::vector<float> actual(expected.size());
            require(cudaMemcpy(actual.data(), upload.second, actual.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess, "read native staged values");
            require(std::memcmp(actual.data(), expected.data(), expected.size() * sizeof(float)) == 0, "native staging retained exact borrowed values");
            std::cout << (rustTc ? "Rust TC allocation" : "caller prepared storage") << " expired after C return with native upload event outstanding\n";
            row.passed();
        }
        const JuicerCuda::ResourceManager::DeviceContextKey key{context.device_id, reinterpret_cast<void*>(context.context)};
        JuicerCuda::ResourceManager::RegistryContextSnapshot oldEpoch;
        require(JuicerCuda::ResourceManager::registry_begin_submission(key, oldEpoch), "admit epoch observation");
        require(JuicerCuda::ResourceManager::registry_note_submission_end(key), "end epoch observation");
        require(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(frame.stream)) == cudaSuccess, "complete before idle retirement");
        std::string retirement;
        require(root.retire_idle_context(context.device_id, reinterpret_cast<void*>(context.context), retirement), "retire idle resources without resetting host context");
        require_status(render(), success, "render after native epoch transition");
        JuicerCuda::ResourceManager::RegistryContextSnapshot newEpoch;
        require(JuicerCuda::ResourceManager::registry_begin_submission(key, newEpoch), "observe replacement epoch");
        require(JuicerCuda::ResourceManager::registry_note_submission_end(key), "end replacement epoch observation");
        require(newEpoch.contextEpoch > oldEpoch.contextEpoch, "native epoch advances without caller epoch");
        require_status(render({[](void*) -> std::uint32_t {
                                   return FJ_ABORT_REQUESTED;
                               },
                               nullptr}),
                       cancelled,
                       "query with null user");

        std::atomic<bool> started{false};
        std::atomic<bool> acquired{false};
        std::thread blocked;
        bool serialized = false;
        {
            JuicerCuda::NativeCall gate(cuda);
            blocked = std::thread([&] {
                started.store(true);
                JuicerCuda::NativeCall next(cuda);
                acquired.store(true);
            });
            while (!started.load())
                std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            serialized = !acquired.load();
        }
        blocked.join();
        require(serialized && acquired.load(), "one owner gate serializes native calls");
        // A sequential callback may run on another admitted host thread, with
        // its exact host-owned context current. No callback or inspection moves.
        FjStatus threaded{FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
        std::thread worker([&] {
            if (cuCtxSetCurrent(original) != CUDA_SUCCESS)
                return;
            threaded = render().status;
            (void)cuCtxSetCurrent(nullptr);
        });
        worker.join();
        require_status(threaded, success, "sequential callback thread");
        require(cuCtxSetCurrent(original) == CUDA_SUCCESS, "main callback context remains current");
        std::cout << "render contract route=" << prepared.route << " passed\n";
        return render().status;
    }
} // namespace JuicerCudaTest
