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

#include <cuda.h>
#include <cuda_runtime.h>

#include "prepared_descriptors.h"
#include "juicer_cuda_owner.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"

namespace JuicerProcess::TestSupport {
    class RootLifetimeObserver final {
    public:
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

    thread_local PendingWork* completionWork = nullptr;

    void after_execute() {
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
            case Injection::Typed:
                throw ExecutionFailure{{injectedStatus, "render injection 100% diagnostic"}};
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
            threaded = render();
            (void)cuCtxSetCurrent(nullptr);
        });
        worker.join();
        require_status(threaded, success, "sequential callback thread");
        require(cuCtxSetCurrent(original) == CUDA_SUCCESS, "main callback context remains current");
        std::cout << "render contract route=" << prepared.route << " passed\n";
        return render();
    }
} // namespace JuicerCudaTest
