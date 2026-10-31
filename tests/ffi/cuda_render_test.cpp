#include "prepared_boundary.h"
#include "prepared_raw_render_fixture.h"
#include "spatial_optics_fixture.h"

#include <algorithm>
#include <array>
#include <atomic>
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
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "prepared_descriptors.h"
#include "juicer_cuda_owner.h"
#include "juicer_test_api.h"
#include "JuicerState.h"
#include "ColorTransforms.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"

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
    std::int32_t injected_error(Operation operation) noexcept {
        const auto index = static_cast<std::size_t>(operation);
        const auto error = std::exchange(errors[index], 0);
        consumed[index] += error != 0 ? 1u : 0u;
        return error;
    }
    void before_diagnostic() {}
} // namespace JuicerCuda::PinnedUploadTest

namespace JuicerAssets::IlluminantTest {
    thread_local unsigned aborts = 0;
    void frame_abort() noexcept {
        ++aborts;
    }
} // namespace JuicerAssets::IlluminantTest

namespace JuicerCuda::RenderTest {
    enum class Injection : std::uint8_t {
        None,
        Allocation,
        Standard,
        Unknown,
        Typed,
        Preflash,
        ScannerCorrection
    };
    thread_local Injection injection = Injection::None;
    thread_local FjStatus injectedStatus{};
    thread_local bool injectedDeferredDirError = false;
    thread_local int completionOverride = 0;
    thread_local unsigned completionCalls = 0;
    thread_local int lastCompletion = 0;
    thread_local const JuicerProcess::Root::CudaFramePreparationRequest* framePreparationRequest = nullptr;
    thread_local JuicerCuda::ResourceManager::DeviceContextKey framePreparationKey{};
    thread_local JuicerCuda::ResourceManager::SubmissionSnapshot framePreparationSubmission{};
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

        explicit PendingWork(cudaStream_t inputStream, unsigned inputCheckpoint)
            : stream(inputStream), checkpoint(inputCheckpoint) {
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
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
            while (!work.released.load()) {
                if (std::chrono::steady_clock::now() >= deadline) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    };
    thread_local PendingWork* pendingWork = nullptr;

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
            case Injection::Preflash: {
                JuicerCuda::Failure failure;
                auto frame = JuicerProcess::root().prepare_cuda_frame(framePreparationKey, framePreparationSubmission, *framePreparationRequest, {}, nullptr, failure);
                if (!frame.active()) {
                    throw std::runtime_error("native preparation failed before the EQUAL fault boundary");
                }
                throw std::runtime_error("EQUAL preflash fault did not escape native preparation");
            }
            case Injection::ScannerCorrection: {
                JuicerCuda::Failure failure;
                auto frame = JuicerProcess::root().prepare_cuda_frame(framePreparationKey, framePreparationSubmission, *framePreparationRequest, {}, nullptr, failure);
                if (!frame.active()) {
                    throw JuicerCuda::ExecutionFailure{failure};
                }

                Scanner::ScannerColorCorrectionDescriptor correction{};
                std::string diagnostic;
                bool valid = false;
                if (Spektrafilm::scan_route_is_print(framePreparationRequest->recipe->profileRoute.scanRoute)) {
                    const auto print = frame.print_resources();
                    if (!print.active || !print.mainIlluminantHost) {
                        frame.abort();
                        throw std::runtime_error("native scanner comparison has no prepared print correction source");
                    }
                    const Scanner::PrintCorrectionDerivationInput input{
                        framePreparationRequest->recipe,
                        framePreparationRequest->scannerTables,
                        print.mainIlluminantHost,
                        print.spectralSampleCount,
                        print.preflashRawCmy,
                        print.normalizer};
                    valid = Scanner::build_print_scanner_color_correction_descriptor(input, correction, diagnostic);
                } else {
                    valid = Scanner::build_direct_scanner_color_correction_descriptor(
                        *framePreparationRequest->recipe,
                        *framePreparationRequest->scannerTables,
                        correction,
                        diagnostic);
                }
                if (!valid) {
                    frame.abort();
                    throw JuicerCuda::ExecutionFailure{{{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, diagnostic}};
                }
                if (!frame.finish(nullptr, failure)) {
                    throw JuicerCuda::ExecutionFailure{failure};
                }
                return;
            }
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
    void check_prepared_state_lifetime(std::optional<PreparedState>& owner, FjAssets* assets, FjCuda* cuda, const FjCudaContext& context, const FjFrame& inputFrame, const FjSubmission& submission, FjPreparedHostData& prepared, FjErrorBuffer* error, StateLifetimeCase lifetimeCase) {
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
        upload.resources = JuicerProcess::TestSupport::RootLifetimeObserver::resources(JuicerProcess::root(), key);
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
        require_status(fj_legacy_assets_release_cached_payloads(assets, error), {FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, "release held-state asset cache");
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
            require(JuicerProcess::root().retire_idle_context(context.device_id, reinterpret_cast<void*>(context.context), diagnostic), "retire native owner after Rust state expiry");
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

    void check_frame_bindings(FjCuda* cuda, const FjCudaContext& context, const FjFrame& frame, const FjSubmission& submission, const FjPreparedHostData& prepared, const RenderRecipe* recipe) {
        const auto check = [&](const FjFrame& candidateFrame, const FjSubmission& candidateSubmission, const FjPreparedHostData& candidate, const char* rejectedField) {
            std::array<char, 256> message{};
            FjErrorBuffer error{message.data(), message.size(), 0};
            AbortQuery callback;
            callback.cancelAt = 1;
            const auto result = rejectedField
                                    ? fj_cuda_render(cuda, &context, &candidateFrame, &candidateSubmission, &candidate, {AbortQuery::query, &callback}, &error)
                                    : render_raw_prepared(cuda, candidateFrame, candidateSubmission, candidate, {AbortQuery::query, &callback}, &error);
            require_status(result, {rejectedField ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_CANCELLED, FJ_API_NONE, 0}, "prepared frame binding");
            require(callback.calls == (rejectedField ? 0u : 1u), "frame binding admission precedes preparation");
            if (rejectedField) {
                require(std::string(message.data(), error.length) == std::string("MalformedPreparedHostData field=") + rejectedField, "frame binding rejection origin");
            }
        };
        // A valid producer projection reaches the first cancellation checkpoint;
        // malformed bindings must stop before that checkpoint or any GPU work.
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
        }
        if (prepared.spatial_dir.hash != 0) {
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
                check(changedFrame, submission, changed, "spatial_dir.frame");
            }
        }
        if (prepared.grain.hash != 0) {
            require(recipe != nullptr, "retained grain frame probes require their authoritative recipe");
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
                check(changed, identity, prepared, "grain.frame");
            }
            // Rebuild only test inputs through the authoritative producer. Signed
            // and fractional times exercise both the frame token and grain phase.
            for (const double time : {-37.75, -1.0, -0.25, 0.0, 37.25}) {
                auto changedFrame = frame;
                changedFrame.time_frames = time;
                auto identity = submission;
                identity.frame_token = static_cast<std::uint64_t>(static_cast<std::int64_t>(std::floor(time)));
                const auto& render = prepared.grain.render_extent;
                const auto& full = prepared.grain.full_frame_extent;
                Spektrafilm::VisualGrainFrameDescriptor grain;
                std::string diagnostic;
                require(JuicerTestSupport::prepare_grain_frame(
                            {&recipe->visualGrain, &recipe->filmDevelop, recipe->profileRoute.capturePolarity, {render.x, render.y, render.width, render.height}, {full.x, full.y, full.width, full.height}, frame.pixel_size_um, time, frame.frame_rate, frame.session_seed, frame.clip_token},
                            grain,
                            diagnostic),
                        "produce temporal grain descriptor");
                JuicerCuda::PreparedDescriptors descriptors;
                descriptors.grainRecipe = recipe->visualGrain;
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
            }
        }
        std::cout << "frame bindings route=" << prepared.route << " grain=" << (prepared.grain.hash != 0) << " passed\n";
    }

    void check_SF_TEMP_BRIDGE_native_scanner_white_zero(
        FjCuda* cuda,
        const FjCudaContext& context,
        const FjFrame& frame,
        const FjSubmission& submission,
        const FjPreparedHostData& prepared) {
        struct RouteCase {
            Spektrafilm::ScanRoute route;
            const char* filmKey;
        };
        struct Levels {
            double black;
            double white;
        };
        constexpr std::array routes{
            RouteCase{Spektrafilm::ScanRoute::NegativePrintScan, "kodak_portra_400"},
            RouteCase{Spektrafilm::ScanRoute::PositiveDirectScan, "fujifilm_provia_100f"},
            RouteCase{Spektrafilm::ScanRoute::PositivePrintScan, "fujifilm_provia_100f"}};

        const auto bytes = [&] {
            require(frame.destination.address != 0 && frame.destination.row_bytes > 0 &&
                        frame.destination.bounds.y2 > frame.destination.bounds.y1,
                    "native scanner comparison has a readable destination");
            require(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(frame.stream)) == cudaSuccess,
                    "drain native scanner comparison frame before observing pixels");
            const std::size_t count = static_cast<std::size_t>(frame.destination.row_bytes) *
                                      static_cast<std::size_t>(frame.destination.bounds.y2 - frame.destination.bounds.y1);
            std::vector<unsigned char> result(count);
            require(cudaMemcpy(result.data(), reinterpret_cast<const void*>(frame.destination.address), count, cudaMemcpyDeviceToHost) == cudaSuccess,
                    "read native scanner comparison destination");
            return result;
        };
        auto failureFrame = frame;
        failureFrame.flags &= ~FJ_FRAME_STREAM_PRESENT;
        failureFrame.stream = 0;

        for (const RouteCase& routeCase : routes) {
            const bool print = Spektrafilm::scan_route_is_print(routeCase.route);
            auto build_product = [&](Levels levels, FocusedRenderStateBuildProduct& product, std::string& diagnostic) {
                ParamSnapshot controls;
                controls.filmProfileKey = routeCase.filmKey;
                controls.printProfileKey = "kodak_portra_endura";
                controls.scanRoute = routeCase.route;
                controls.spectralUpsamplingMode = 1;
                controls.enlIll = 7;
                controls.scannerBlackCorrection = 1;
                controls.scannerWhiteCorrection = 1;
                controls.scannerBlackLevel = levels.black;
                controls.scannerWhiteLevel = levels.white;
                controls.grainControls.active = false;
                controls.dirCouplers.active = false;
                return print ? build_print_render_state_product(controls, product, diagnostic)
                             : build_direct_render_state_product(controls, product, diagnostic);
            };
            auto run_native = [&](FocusedRenderStateBuildProduct& product, JuicerCuda::RenderTest::Injection injection) {
                Scanner::ScannerSpectralLutDescriptor scannerLut{};
                std::string diagnostic;
                const bool lutBuilt = print
                                          ? Scanner::build_print_scanner_spectral_lut_descriptor(
                                                {&product.recipe.profileRoute, &product.recipe.densityBounds, &product.recipe.scannerOutput},
                                                scannerLut,
                                                diagnostic)
                                          : Scanner::build_direct_scanner_spectral_lut_descriptor(
                                                {&product.recipe.profileRoute, &product.recipe.densityBounds, &product.recipe.scannerOutput},
                                                scannerLut,
                                                diagnostic);
                require(lutBuilt, diagnostic.c_str());

                JuicerProcess::Root::CudaFramePreparationRequest request;
                request.recipe = &product.recipe;
                request.exposureTables = &product.payload.exposureTables;
                request.filmRawConfig = &product.payload.filmRawConfig;
                request.filmTcLut = product.payload.filmTcLut ? &*product.payload.filmTcLut : nullptr;
                request.printMainIlluminant = product.payload.printMainIlluminant ? &*product.payload.printMainIlluminant : nullptr;
                request.scannerTables = &product.payload.scannerTables;
                request.scannerColor = &product.payload.scannerColor;
                request.scannerLutDescriptor = &scannerLut;
                request.outputBoundaryTable = product.payload.outputBoundaryTable.get();
                request.requestedWidth = frame.render_window.x2 - frame.render_window.x1;
                request.requestedHeight = frame.render_window.y2 - frame.render_window.y1;

                auto& key = JuicerCuda::RenderTest::framePreparationKey;
                key = {context.device_id, reinterpret_cast<void*>(context.context)};
                auto& token = JuicerCuda::RenderTest::framePreparationSubmission;
                token.instanceToken.value = 0x5343414e0000ULL + static_cast<std::uint32_t>(routeCase.route);
                static std::uint64_t nextFrameToken = 940;
                token.frameToken.value = nextFrameToken++;
                token.snapshotId = token.frameToken.value;
                token.deviceContextKey = key;
                token.keyDigests = JuicerCuda::ResourceManager::make_key_digests(
                    product.payload.uploadCoreHash,
                    product.recipe.dirCouplers.hash,
                    product.payload.scannerHash,
                    0);
                JuicerCuda::RenderTest::framePreparationRequest = &request;
                JuicerCuda::RenderTest::completionCalls = 0;
                JuicerCuda::RenderTest::completionOverride = 0;
                JuicerCuda::RenderTest::injection = injection;
                JuicerAssets::IlluminantTest::aborts = 0;
                std::array<char, 512> errorText{};
                FjErrorBuffer error{errorText.data(), errorText.size(), 0};
                const auto result = render_raw_prepared(cuda, failureFrame, submission, prepared, {}, &error);
                JuicerCuda::RenderTest::injection = JuicerCuda::RenderTest::Injection::None;
                JuicerCuda::RenderTest::framePreparationRequest = nullptr;
                return std::pair{result, std::string(errorText.data(), error.length)};
            };

            FocusedRenderStateBuildProduct failedProduct;
            std::string diagnostic;
            const bool failedProductBuilt = build_product({0.0, 0.0}, failedProduct, diagnostic);
            require(failedProductBuilt, diagnostic.c_str());
            require(failedProduct.recipe.scannerOutput.blackCorrection &&
                        failedProduct.recipe.scannerOutput.whiteCorrection &&
                        failedProduct.recipe.scannerOutput.blackLevel == 0.0f &&
                        failedProduct.recipe.scannerOutput.whiteLevel == 0.0f,
                    "native scanner comparison retains correction flags 3 and encoded levels zero");
            const auto beforeFailure = bytes();
            const auto [failed, failureDiagnostic] = run_native(failedProduct, JuicerCuda::RenderTest::Injection::ScannerCorrection);
            const std::string expectedDiagnostic = print
                                                       ? "MalformedRequiredProfileData phase=8B print corrected midgray"
                                                       : "MalformedRequiredProfileData phase=8B direct corrected midgray";
            require_status(failed, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "native scanner white-zero correction failure");
            require(failureDiagnostic == expectedDiagnostic, "native scanner white-zero diagnostic preserves the corrected-midgray rejection");
            require(JuicerCuda::RenderTest::completionCalls == 1 && JuicerCuda::RenderTest::lastCompletion == cudaSuccess,
                    "native scanner correction failure completes the absent stream exactly once");
            require(JuicerAssets::IlluminantTest::aborts == 1, "native scanner correction failure aborts its prepared Root frame");
            require(cudaStreamQuery(nullptr) == cudaSuccess, "native scanner correction failure drains absent-stream work");
            require(bytes() == beforeFailure, "native scanner correction failure does not render pixels");

            FocusedRenderStateBuildProduct recoveredProduct;
            const bool recoveredProductBuilt = build_product({0.01234567890123, 0.98765432109876}, recoveredProduct, diagnostic);
            require(recoveredProductBuilt, diagnostic.c_str());
            require(recoveredProduct.recipe.scannerOutput.blackCorrection &&
                        recoveredProduct.recipe.scannerOutput.whiteCorrection &&
                        recoveredProduct.recipe.scannerOutput.blackLevel == static_cast<float>(0.01234567890123) &&
                        recoveredProduct.recipe.scannerOutput.whiteLevel == static_cast<float>(0.98765432109876),
                    "native scanner recovery retains the original valid encoded levels");
            const auto beforeRecovery = bytes();
            const auto [recovered, recoveryDiagnostic] = run_native(recoveredProduct, JuicerCuda::RenderTest::Injection::ScannerCorrection);
            require_status(recovered, {FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, "native scanner correction recovery");
            require(recoveryDiagnostic.empty(), "native scanner correction recovery has no failure diagnostic");
            require(JuicerCuda::RenderTest::completionCalls == 1 && JuicerCuda::RenderTest::lastCompletion == cudaSuccess,
                    "native scanner correction recovery completes the absent stream exactly once");
            require(JuicerAssets::IlluminantTest::aborts == 0, "successful native scanner correction does not abort its Root frame");
            require(bytes() == beforeRecovery, "valid native scanner correction recovery preserves output identity");
        }
        std::cout << "PASS SF_TEMP_BRIDGE_NATIVE_SCANNER_WHITE_ZERO: 3 native Root/correction failures before pixels and valid correction/render recoveries\n";
    }

    FjStatus check_render_contract(FjCuda* cuda, const FjCudaContext& context, const FjFrame& frame, const FjSubmission& submission, const FjPreparedHostData& prepared, FjErrorBuffer* error) {
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
        require(JuicerProcess::TestSupport::RootLifetimeObserver::fences(JuicerProcess::root()) == 0, "cold cancellation starts without resources");
        require_status(render({AbortQuery::query, &beforeEnqueue}), cancelled, "cancel before first preparation");
        require(beforeEnqueue.calls == 1 && beforeEnqueue.sameThread, "pre-enqueue callback");
        require(JuicerProcess::TestSupport::RootLifetimeObserver::fences(JuicerProcess::root()) == 0, "pre-enqueue cancellation created no frame fence");
        for (int fault = 0; fault < 7; ++fault) {
            FjErrorBuffer malformedError{nullptr, 4, 92};
            const auto result = fj_cuda_render(fault == 0 ? nullptr : cuda, fault == 1 ? nullptr : &context, fault == 2 ? nullptr : &frame, fault == 3 ? nullptr : &submission, fault == 4 ? nullptr : &prepared, {}, fault == 5 ? &malformedError : fault == 6 ? nullptr
                                                                                                                                                                                                                                                                : error);
            require_status(result, fault == 6 ? success : unsupported, "null argument");
        }
        require_status(fj_cuda_render(reinterpret_cast<FjCuda*>(std::uintptr_t{1}), &context, &frame, &submission, &prepared, {}, error), preparationFailure, "unregistered owner");
        for (int fault = 0; fault < 4; ++fault) {
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
        }
        for (int fault = 0; fault < 9; ++fault) {
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
        }
        for (int fault = 0; fault < 13; ++fault) {
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
        }
        for (int fault = 0; fault < 10; ++fault) {
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
                    ++changed.scanner_lut.density_bounds_hash;
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
        for (const auto injection : {Injection::Allocation, Injection::Standard, Injection::Unknown, Injection::Typed}) {
            JuicerCuda::RenderTest::injection = injection;
            JuicerCuda::RenderTest::injectedStatus = JuicerCuda::driver_failure_status(CUDA_ERROR_CONTEXT_IS_DESTROYED);
            const auto result = render();
            JuicerCuda::RenderTest::injection = Injection::None;
            const auto expected = injection == Injection::Typed ? JuicerCuda::RenderTest::injectedStatus
                                                                : FjStatus{injection == Injection::Allocation ? FJ_STATUS_ALLOCATION_FAILURE : FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
            require_status(result, expected, "C exception containment");
            require(error->length > 0, "owned exception diagnostic");
        }
        for (const auto typed : {preparationFailure, JuicerCuda::runtime_failure_status(cudaErrorMemoryAllocation), JuicerCuda::runtime_failure_status(cudaErrorInvalidValue), JuicerCuda::cufft_failure_status(6), JuicerCuda::driver_failure_status(CUDA_ERROR_CONTEXT_IS_DESTROYED)}) {
            for (const std::size_t capacity : {std::size_t{0}, std::size_t{1}, std::size_t{8}, std::size_t{64}}) {
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
            }
        }
        for (const bool deferred : {false, true}) {
            for (const std::size_t capacity : {std::size_t{0}, std::size_t{1}, std::size_t{8}, std::size_t{64}}) {
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
            }
        }
        // EQUAL is prepared before publication. Its former late exception
        // transport is covered here without adding preparation work to render.
        for (const auto category : {FJ_STATUS_UNSUPPORTED_INPUT, FJ_STATUS_INTERNAL_FAILURE, FJ_STATUS_ALLOCATION_FAILURE}) {
            for (const bool completionFailure : {false, true}) {
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
        for (unsigned checkpoint = 1; checkpoint <= 3; ++checkpoint) {
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
                        require(JuicerProcess::TestSupport::RootLifetimeObserver::fences(JuicerProcess::root()) != 0, "aborted frame retains its use fence");
                    }
                    std::cout << "cancel checkpoint=" << checkpoint << " completed=" << (completion == cudaSuccess) << '\n';
                }
            }
            const auto calls = callback.calls;
            require_status(render(), success, "successful render after cancellation");
            require(callback.calls == calls, "callback retained after return");
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
        for (int streamCase = 0; streamCase < 3; ++streamCase) {
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
        }
        // Explicit default-stream and absent-stream calls both execute; absence
        // additionally guarantees completion before caller storage/image release.
        for (const bool supplied : {true, false}) {
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
        }
        for (bool rustTc : {false, true}) {
            const JuicerCuda::ResourceManager::DeviceContextKey uploadKey{context.device_id, reinterpret_cast<void*>(context.context)};
            JuicerCuda::RenderTest::ExpiringUpload upload;
            upload.resources = JuicerProcess::TestSupport::RootLifetimeObserver::resources(JuicerProcess::root(), uploadKey);
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
        }
        const JuicerCuda::ResourceManager::DeviceContextKey key{context.device_id, reinterpret_cast<void*>(context.context)};
        JuicerCuda::ResourceManager::RegistryContextSnapshot oldEpoch;
        require(JuicerCuda::ResourceManager::registry_begin_submission(key, oldEpoch), "admit epoch observation");
        require(JuicerCuda::ResourceManager::registry_note_submission_end(key), "end epoch observation");
        require(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(frame.stream)) == cudaSuccess, "complete before idle retirement");
        std::string retirement;
        require(JuicerProcess::root().retire_idle_context(context.device_id, reinterpret_cast<void*>(context.context), retirement), "retire idle resources without resetting host context");
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

        // SF_TEMP_BRIDGE_NATIVE_EQUAL_COMPLETION: retain the original six rows
        // until I7 after the constructor and typed-transport G2 witnesses pass.
        if (prepared.route == 1) {
            ParamSnapshot controls;
            controls.filmProfileKey = "kodak_portra_400";
            controls.printProfileKey = "kodak_portra_endura";
            controls.scanRoute = Spektrafilm::ScanRoute::NegativePrintScan;
            controls.spectralUpsamplingMode = 1;
            controls.enlIll = 7;
            controls.printPreflashExposure = .1;
            controls.grainControls.active = false;
            controls.dirCouplers.active = false;
            FocusedRenderStateBuildProduct product;
            std::string diagnostic;
            require(build_print_render_state_product(controls, product, diagnostic), diagnostic.c_str());
            Scanner::ScannerSpectralLutDescriptor descriptor;
            require(Scanner::build_print_scanner_spectral_lut_descriptor({&product.recipe.profileRoute, &product.recipe.densityBounds, &product.recipe.scannerOutput}, descriptor, diagnostic), diagnostic.c_str());
            JuicerProcess::Root::CudaFramePreparationRequest request;
            request.recipe = &product.recipe;
            request.exposureTables = &product.payload.exposureTables;
            request.filmRawConfig = &product.payload.filmRawConfig;
            request.printMainIlluminant = &*product.payload.printMainIlluminant;
            request.scannerTables = &product.payload.scannerTables;
            request.scannerColor = &product.payload.scannerColor;
            request.scannerLutDescriptor = &descriptor;
            request.outputBoundaryTable = product.payload.outputBoundaryTable.get();
            request.requestedWidth = 16;
            request.requestedHeight = 16;
            JuicerCuda::RenderTest::framePreparationRequest = &request;
            JuicerCuda::RenderTest::framePreparationKey = {context.device_id, reinterpret_cast<void*>(context.context)};
            auto& token = JuicerCuda::RenderTest::framePreparationSubmission;
            token.instanceToken.value = 0x494c4c554d;
            token.frameToken.value = 900;
            token.snapshotId = 900;
            token.deviceContextKey = JuicerCuda::RenderTest::framePreparationKey;
            token.keyDigests = JuicerCuda::ResourceManager::make_key_digests(product.payload.uploadCoreHash, product.recipe.dirCouplers.hash, product.payload.scannerHash, 0);
            auto failureFrame = frame;
            failureFrame.flags &= ~FJ_FRAME_STREAM_PRESENT;
            failureFrame.stream = 0;
            for (std::uint32_t fault : {1u, 2u, 3u})
                for (bool completionFailure : {false, true}) {
                    // Each attempt uses a cold print descriptor; no successful commit
                    // precedes the EQUAL binding. The real absent-stream completion
                    // runs before its test-only status override.
                    require(JuicerProcess::root().assets().release_cached_payloads().category == FJ_STATUS_SUCCESS, "preflash source release");
                    JuicerAssets::IlluminantTest::aborts = 0;
                    JuicerCuda::RenderTest::completionCalls = 0;
                    require(fj_test_illuminant_arm_fault(3, 1, fault).category == FJ_STATUS_SUCCESS, "arm native preflash export");
                    JuicerCuda::RenderTest::injection = JuicerCuda::RenderTest::Injection::Preflash;
                    JuicerCuda::RenderTest::completionOverride = completionFailure ? static_cast<int>(cudaErrorInvalidValue) : 0;
                    const auto failed = render_raw_prepared(cuda, failureFrame, submission, prepared, {}, error);
                    JuicerCuda::RenderTest::injection = JuicerCuda::RenderTest::Injection::None;
                    const FjStatus expected = completionFailure ? JuicerCuda::runtime_failure_status(cudaErrorInvalidValue) : FjStatus{fault == 1 ? FJ_STATUS_UNSUPPORTED_INPUT : fault == 2 ? FJ_STATUS_INTERNAL_FAILURE
                                                                                                                                                                                             : FJ_STATUS_ALLOCATION_FAILURE,
                                                                                                                                       FJ_API_NONE,
                                                                                                                                       0};
                    require_status(failed, expected, "native EQUAL preflash/completion precedence");
                    require(JuicerCuda::RenderTest::completionCalls == 1 && JuicerCuda::RenderTest::lastCompletion == cudaSuccess, "actual absent-stream completion precedes status override");
                    require(JuicerAssets::IlluminantTest::aborts == 1, "native preparation exception aborted one admitted frame");
                    require(cudaStreamQuery(nullptr) == cudaSuccess, "native preflash unwind completed absent stream work");
                    require(fj_test_illuminant_live_lenses() == 0 && fj_test_csv_live_owners() == 0, "native unwind reclaimed Rust sources/lenses");
                    std::array<char, 128> bytes{};
                    FjErrorBuffer errorBuffer{bytes.data(), bytes.size(), 0};
                    FjIlluminant equal{};
                    require(fj_legacy_illuminant_equal_energy(&equal, &errorBuffer).category == FJ_STATUS_SUCCESS, "intended native preflash raw export consumed fault");
                }
            JuicerCuda::RenderTest::framePreparationRequest = nullptr;
            JuicerCuda::Failure failure;
            auto recovered = JuicerProcess::root().prepare_cuda_frame(JuicerCuda::RenderTest::framePreparationKey, token, request, {}, nullptr, failure);
            require(recovered.active(), "native preflash resources reusable after abort");
            require(recovered.finish(nullptr, failure), "recovered preflash finish");
            require_status(render(), success, "ordinary render reuses resources after native preflash abort");
        }
        if (prepared.route == static_cast<std::uint32_t>(Spektrafilm::ScanRoute::NegativePrintScan)) {
            check_SF_TEMP_BRIDGE_native_scanner_white_zero(cuda, context, frame, submission, prepared);
        }
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
