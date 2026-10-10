#include "prepared_boundary.h"

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
#include "SpectralProcessing.h"
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
    std::int32_t injected_error(Operation) noexcept {
        return 0;
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
        Preflash
    };
    thread_local Injection injection = Injection::None;
    thread_local FjStatus injectedStatus{};
    thread_local bool injectedDeferredDirError = false;
    thread_local int completionOverride = 0;
    thread_local unsigned completionCalls = 0;
    thread_local int lastCompletion = 0;
    thread_local const JuicerProcess::Root::CudaFramePreparationRequest* preflashRequest = nullptr;
    thread_local JuicerCuda::ResourceManager::DeviceContextKey preflashKey{};
    thread_local JuicerCuda::ResourceManager::SubmissionSnapshot preflashSubmission{};
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

        ~ExpiringUpload() {
            (void)cudaStreamSynchronize(stream);
            (void)cudaFree(first);
            (void)cudaFree(second);
        }
        void enqueue() {
            for (unsigned i = 0; i < 128; ++i) {
                if (cudaMemcpyAsync(second, first, kBytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
                    cudaMemcpyAsync(first, second, kBytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
                    throw std::runtime_error("queue finite expiry work");
                }
            }
            Failure failure;
            if (!PinnedUploadTest::upload(*resources, second, source.data, source.count * sizeof(float), stream, failure)) {
                throw std::runtime_error(failure.diagnostic);
            }
            staging = PinnedUploadTest::snapshot(resources->ownerContextKey);
        }
    };
    thread_local ExpiringUpload* expiringUpload = nullptr;

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
                auto frame = JuicerProcess::root().prepare_cuda_frame(preflashKey, preflashSubmission, *preflashRequest, {}, nullptr, failure);
                if (!frame.active()) {
                    throw std::runtime_error("native preparation failed before the EQUAL fault boundary");
                }
                throw std::runtime_error("EQUAL preflash fault did not escape native preparation");
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
    void check_frame_bindings(FjCuda* cuda, const FjCudaContext& context, const FjFrame& frame, const FjSubmission& submission, const FjPreparedHostData& prepared, const RenderRecipe& recipe) {
        const auto check = [&](const FjFrame& candidateFrame, const FjSubmission& candidateSubmission, const FjPreparedHostData& candidate, const char* rejectedField) {
            std::array<char, 256> message{};
            FjErrorBuffer error{message.data(), message.size(), 0};
            AbortQuery callback;
            callback.cancelAt = 1;
            const auto result = fj_cuda_render(cuda, &context, &candidateFrame, &candidateSubmission, &candidate, {AbortQuery::query, &callback}, &error);
            require_status(result, {rejectedField ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_CANCELLED, FJ_API_NONE, 0}, "prepared frame binding");
            require(callback.calls == (rejectedField ? 0u : 1u), "frame binding admission precedes preparation");
            if (rejectedField) {
                require(std::string(message.data(), error.length) == std::string("MalformedPreparedHostData field=") + rejectedField, "frame binding rejection origin");
            }
        };
        // A valid producer projection reaches the first cancellation checkpoint;
        // malformed bindings must stop before that checkpoint or any GPU work.
        check(frame, submission, prepared, nullptr);
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
                require(Spektrafilm::build_visual_grain_frame_descriptor(
                            {&recipe.visualGrain, &recipe.filmDevelop, recipe.profileRoute.capturePolarity, {render.x, render.y, render.width, render.height}, {full.x, full.y, full.width, full.height}, frame.pixel_size_um, time, frame.frame_rate, frame.session_seed, frame.clip_token},
                            grain,
                            diagnostic),
                        "produce temporal grain descriptor");
                JuicerCuda::PreparedDescriptors descriptors;
                descriptors.grainRecipe = recipe.visualGrain;
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

    FjStatus check_render_contract(FjCuda* cuda, const FjCudaContext& context, const FjFrame& frame, const FjSubmission& submission, const FjPreparedHostData& prepared, FjErrorBuffer* error) {
        constexpr FjStatus unsupported{FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0};
        constexpr FjStatus preparationFailure{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        constexpr FjStatus success{FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
        constexpr FjStatus cancelled{FJ_STATUS_CANCELLED, FJ_API_NONE, 0};
        require(prepared.spatial_dir.hash != 0 && prepared.optics.flags == 0 && prepared.grain.hash == 0 && prepared.effects.hash == 0, "render contract exercises partial spatial DIR admission");
        const auto render = [&](FjAbortCallback callback = {}) {
            return fj_cuda_render(cuda, &context, &frame, &submission, &prepared, callback, error);
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
                const auto result = fj_cuda_render(cuda, &context, &frame, &submission, &prepared, {}, &output);
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
                const auto outcome = fj_cuda_render(cuda, &context, &frame, &submission, &prepared, {}, &output);
                JuicerCuda::RenderTest::injection = Injection::None;
                JuicerCuda::RenderTest::injectedDeferredDirError = false;
                require_status(outcome.status, preparationFailure, "typed deferred DIR status");
                require(outcome.flags == (deferred ? FJ_RENDER_DEFERRED_DIR_ERROR : 0U), "deferred provenance independent of diagnostic capacity and text");
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
            const auto result = fj_cuda_render(cuda, &context, &streamFrame, &submission, &prepared, {}, error);
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
            require_status(fj_cuda_render(cuda, &context, &defaultFrame, &submission, &prepared, {}, error), success, "default stream render");
            if (!supplied)
                require(cudaStreamQuery(nullptr) == cudaSuccess, "absent stream must complete before return");
            AbortQuery cancel;
            cancel.cancelAt = 2;
            require_status(fj_cuda_render(cuda, &context, &defaultFrame, &submission, &prepared, {AbortQuery::query, &cancel}, error), cancelled, "default stream cancellation");
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
            const auto outcome = fj_cuda_render(cuda, &context, &frame, &submission, &expired, {}, error);
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
            JuicerCuda::RenderTest::preflashRequest = &request;
            JuicerCuda::RenderTest::preflashKey = {context.device_id, reinterpret_cast<void*>(context.context)};
            auto& token = JuicerCuda::RenderTest::preflashSubmission;
            token.instanceToken.value = 0x494c4c554d;
            token.frameToken.value = 900;
            token.snapshotId = 900;
            token.deviceContextKey = JuicerCuda::RenderTest::preflashKey;
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
                    const auto failed = fj_cuda_render(cuda, &context, &failureFrame, &submission, &prepared, {}, error);
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
            JuicerCuda::RenderTest::preflashRequest = nullptr;
            JuicerCuda::Failure failure;
            auto recovered = JuicerProcess::root().prepare_cuda_frame(JuicerCuda::RenderTest::preflashKey, token, request, {}, nullptr, failure);
            require(recovered.active(), "native preflash resources reusable after abort");
            require(recovered.finish(nullptr, failure), "recovered preflash finish");
            require_status(render(), success, "ordinary render reuses resources after native preflash abort");
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
