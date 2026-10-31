#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include <cuda_runtime.h>

#include "JuicerState.h"
#include "Logging.h"
#include "PreparedState.h"
#include "ProcessRoot.h"
#include "SpectralData.h"
#include "ColorTransforms.h"
#include "mainProcessing.h"
#include "juicer_cuda_call_scope.h"
#include "juicer_cuda_owner.h"

namespace {
    inline bool is_finite(float value);
    inline bool is_finite(double value);

#if JUICER_DIAGNOSTICS_COMPILED
    inline const char* nonempty_cstr_or(const char* value, const char* fallback) {
        return (value && value[0] != '\0') ? value : fallback;
    }

    inline int bool_to_i32(bool value) {
        return value ? 1 : 0;
    }


#endif

    inline double finite_or(double value, double fallback) {
        return is_finite(value) ? value : fallback;
    }

    inline bool is_positive_finite(double value) {
        return is_finite(value) && value > 0.0;
    }

    inline double positive_finite_or(double value, double fallback) {
        return is_positive_finite(value) ? value : fallback;
    }

    inline bool is_finite(float value) {
        return std::isfinite(value);
    }

    inline bool is_finite(double value) {
        return std::isfinite(value);
    }

    void recover_context_loss_state(
        InstanceState* instanceState,
        const JuicerCuda::ResourceManager::DeviceContextKey& key,
        const char* stage,
        const JuicerCuda::Failure& failure) {
        if (!instanceState) {
            return;
        }
        if (!JuicerCuda::context_loss(failure)) {
            return;
        }

        std::string retireError;
#if JUICER_DIAGNOSTICS_COMPILED
        const char* stageName = nonempty_cstr_or(stage, "unknown_stage");
        const bool traceInfo = JTRACE_ENABLED(1);
        const bool retireAccepted = JuicerProcess::root().retire_reset_context(
            key.deviceId,
            key.contextOpaque,
            retireError);
#else
        (void)stage;
        JuicerProcess::root().retire_reset_context(
            key.deviceId,
            key.contextOpaque,
            retireError);
#endif

#if JUICER_DIAGNOSTICS_COMPILED
        bool latchCleared = false;
#endif
        {
            std::lock_guard<std::mutex> lock(instanceState->submissionSnapshotLatchMutex);
            if (instanceState->submissionSnapshotLatchValid &&
                instanceState->submissionSnapshotLatch.deviceContextKey == key) {
                instanceState->submissionSnapshotLatch = JuicerCuda::ResourceManager::SubmissionSnapshot{};
                instanceState->submissionSnapshotLatchValid = false;
#if JUICER_DIAGNOSTICS_COMPILED
                latchCleared = true;
#endif
            }
        }

#if JUICER_DIAGNOSTICS_COMPILED
        if (traceInfo) {
            const std::uintptr_t contextBits = reinterpret_cast<std::uintptr_t>(key.contextOpaque);
            std::string msg;
            msg.reserve(192);
            msg = "stage=";
            msg += stageName;
            msg += " device_id=";
            msg += std::to_string(key.deviceId);
            msg += " context=";
            msg += std::to_string(contextBits);
            msg += " error_code=";
            msg += std::to_string(failure.status.native_code);
            msg += " retire_accepted=";
            msg += std::to_string(bool_to_i32(retireAccepted));
            msg += " latch_cleared=";
            msg += std::to_string(bool_to_i32(latchCleared));
            if (!retireError.empty()) {
                msg += " retire_error=";
                msg += retireError;
            }
            JTRACE("MSLCY", msg);
        }
#endif
    }

    std::int64_t frame_index_from_time(double time) {
        return static_cast<std::int64_t>(std::floor(finite_or(time, 0.0)));
    }

#if JUICER_DIAGNOSTICS_COMPILED
    const char* submission_snapshot_action_label(bool reusingSnapshotLatch) {
        return reusingSnapshotLatch ? "reuse" : "new";
    }
#endif

} // namespace

namespace {
    struct SubmissionSelection final {
        InstanceState* instanceState = nullptr;
        bool traceVerbose = false;
    };

    FjStatus select_submission(
        void* user,
        const FjCudaContext* context,
        const FjSubmission* requested,
        FjSubmission* selected,
        FjErrorBuffer* error) noexcept {
        auto* access = static_cast<SubmissionSelection*>(user);
        if (!access || !access->instanceState || !context || !requested || !selected) {
            return JuicerCuda::write_status(
                {FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0},
                "invalid submission selection callback arguments",
                error);
        }
        *selected = {};

        auto& state = *access->instanceState;
        JuicerCuda::ResourceManager::SubmissionSnapshot snapshot{};
        snapshot.instanceToken.value = requested->instance_token;
        snapshot.frameToken.value = requested->frame_token;
        snapshot.deviceContextKey = {
            context->device_id,
            reinterpret_cast<void*>(context->context)};
        snapshot.keyDigests = JuicerCuda::ResourceManager::make_key_digests(
            requested->upload_core_hash,
            requested->dir_hash,
            requested->scanner_hash,
            requested->auto_exposure_hash);

        bool reusingSnapshotLatch = false;
        try {
            {
                std::lock_guard<std::mutex> latchLock(state.submissionSnapshotLatchMutex);
                const auto& latched = state.submissionSnapshotLatch;
                const bool digestsMatch =
                    latched.keyDigests.uploadCoreHash == snapshot.keyDigests.uploadCoreHash &&
                    latched.keyDigests.dirHash == snapshot.keyDigests.dirHash &&
                    latched.keyDigests.scannerHash == snapshot.keyDigests.scannerHash &&
                    latched.keyDigests.autoExposureHash == snapshot.keyDigests.autoExposureHash;
                if (state.submissionSnapshotLatchValid &&
                    latched.instanceToken.value == snapshot.instanceToken.value &&
                    latched.frameToken.value == snapshot.frameToken.value &&
                    latched.deviceContextKey == snapshot.deviceContextKey &&
                    digestsMatch && latched.snapshotId != 0) {
                    snapshot = latched;
                    reusingSnapshotLatch = true;
                } else {
                    std::uint64_t nextSnapshotId =
                        state.submissionSnapshotIdNext.fetch_add(1, std::memory_order_relaxed);
                    if (nextSnapshotId == 0) {
                        nextSnapshotId =
                            state.submissionSnapshotIdNext.fetch_add(1, std::memory_order_relaxed);
                    }
                    snapshot.snapshotId = nextSnapshotId;
                    state.submissionSnapshotLatch = snapshot;
                    state.submissionSnapshotLatchValid = true;
                }
            }
        } catch (const std::bad_alloc&) {
            return JuicerCuda::write_status(
                {FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0},
                "submission snapshot selection allocation failed",
                error);
        } catch (...) {
            return JuicerCuda::write_status(
                {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0},
                "submission snapshot selection failed",
                error);
        }

#if JUICER_DIAGNOSTICS_COMPILED
        if (access->traceVerbose) {
            try {
                std::string message = "path=cuda action=";
                message += submission_snapshot_action_label(reusingSnapshotLatch);
                message += " frame_token=";
                message += std::to_string(snapshot.frameToken.value);
                message += " snapshot_id=";
                message += std::to_string(snapshot.snapshotId);
                message += " instance_token=";
                message += std::to_string(snapshot.instanceToken.value);
                JTRACE_VERBOSE("MSSNP", message);
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
        }
#else
        (void)reusingSnapshotLatch;
#endif

        *selected = {
            snapshot.instanceToken.value,
            snapshot.frameToken.value,
            snapshot.snapshotId,
            snapshot.keyDigests.uploadCoreHash,
            snapshot.keyDigests.dirHash,
            snapshot.keyDigests.scannerHash,
            snapshot.keyDigests.autoExposureHash};
        return JuicerCuda::write_status({FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, {}, error);
    }

    struct AbortAccess final {
        bool (*query)(void*) = nullptr;
        void* user = nullptr;
        bool failed = false;
    };

    struct RenderCompletion final {
        OFX::ImageEffect* effect = nullptr;
        InstanceState* instanceState = nullptr;
        JuicerCuda::CallScope* callScope = nullptr;
        const AbortAccess* abortAccess = nullptr;
        bool completed = false;
    };

    void deliver_deferred_dir_failure(
        OFX::ImageEffect* effect,
        std::string_view diagnostic) noexcept {
        if (!effect) {
            return;
        }
        try {
            std::string deliveryText(diagnostic);
            for (std::size_t position = 0;
                 (position = deliveryText.find('%', position)) != std::string::npos;
                 position += 2u) {
                deliveryText.insert(position, 1u, '%');
            }
            effect->sendMessage(
                OFX::Message::eMessageError,
                "FilmJuicerDeferredCudaFailure",
                deliveryText);
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }

    FjCallCompletionResult complete_render_call(
        void* user,
        const FjCallCompletionInput* input) noexcept {
        auto* completion = static_cast<RenderCompletion*>(user);
        if (!completion || !completion->callScope || !input) {
            return {{{FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, 0}, 1};
        }
        completion->completed = true;

        std::string_view diagnostic;
        if (input->diagnostic.data && input->diagnostic.count != 0) {
            diagnostic = {input->diagnostic.data, input->diagnostic.count};
        }
        const FjRenderOutcome outcome = input->outcome;
#if defined(JUICER_NOISE_TEST_HOOK)
        if (input->render_outcome_valid != 0) {
            JuicerAssets::NoiseTest::outcome(outcome.status);
        }
#endif
        auto& recovery = completion->callScope->recovery();
        if (!recovery.pending && outcome.status.category == FJ_STATUS_CONTEXT_LOSS &&
            input->has_inspected_context != 0) {
            try {
                recovery = {
                    true,
                    {outcome.status, std::string(diagnostic)},
                    "native_render"};
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
        }
        if ((outcome.flags & FJ_RENDER_DEFERRED_DIR_ERROR) != 0) {
            deliver_deferred_dir_failure(completion->effect, diagnostic);
        }
        const bool fatal = input->force_fatal != 0 || input->abort_failed != 0 ||
                           (completion->abortAccess && completion->abortAccess->failed) ||
                           (input->render_outcome_valid != 0 &&
                            outcome.status.category != FJ_STATUS_SUCCESS &&
                            outcome.status.category != FJ_STATUS_CANCELLED);
        if (fatal) {
            try {
                JTRACE("CUDA", std::string(diagnostic));
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
        }
        if (input->recovery_scope_entered != 0) {
            try {
#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
                JuicerCuda::ExecutorTest::observe_recovery_start(recovery.pending);
#endif
                if (recovery.pending) {
                    JuicerCuda::NativeCall recoveryCall(JuicerCuda::borrowed_owner());
                    recover_context_loss_state(
                        completion->instanceState,
                        {input->inspected_context.device_id,
                         reinterpret_cast<void*>(input->inspected_context.context)},
                        recovery.stage,
                        recovery.failure);
                    recovery = JuicerCuda::PendingContextLossRecovery{};
                }
#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
                JuicerCuda::ExecutorTest::observe_recovery_end();
#endif
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
        }
        return {outcome, fatal ? 1u : 0u};
    }

    std::uint32_t query_abort(void* user) noexcept {
        auto* access = static_cast<AbortAccess*>(user);
        if (!access || !access->query) {
            return FJ_ABORT_REQUESTED;
        }
        try {
            return access->query(access->user) ? FJ_ABORT_REQUESTED : FJ_ABORT_CONTINUE;
        } catch (...) {
            access->failed = true;
            return FJ_ABORT_REQUESTED;
        }
    }
} // namespace

// --- Spatial DIR: defensive curve utilities (monotonic + robust interpolation) ---

bool curve_ok(const Spectral::Curve& c) {
    const size_t N = c.lambda_nm.size();
    if (N < 2 || c.linear.size() != N)
        return false;
    const float* lambdaData = c.lambda_nm.data();
    float prev = lambdaData[0];
    if (!is_finite(prev))
        return false;
    for (size_t i = 1; i < N; ++i) {
        const float xi = lambdaData[i];
        if (!is_finite(xi))
            return false;
        if (xi < prev)
            return false; // allow duplicates (xi == prev), but never decreasing
        prev = xi;
    }
    return true;
}


// JuicerProcessor method definitions matching JuicerProcessing.h

JuicerProcessor::JuicerProcessor(OFX::ImageEffect& effect)
    : OFX::ImageProcessor(effect) {
}

void JuicerProcessor::setSrcDst(const SourceDestinationImages& images) {
    _srcImg = images.src;
    setDstImg(images.dst);
}

void JuicerProcessor::setDirectFrameRequest(const DirectFrameRequest& request) {
    setRenderWindow(request.renderWindow);
    _effectsGeometry = request.effectsGeometry;
    _fullFrameExtent = request.fullFrameExtent;
    _framePolicy = request.framePolicy;
    _directStateHold = request.state;
    _printStateHold.reset();
    _sessionSeed = request.sessionSeed;
    _instanceToken = request.instanceToken;
    _clipToken = request.clipToken;
    _timeFrames = finite_or(request.frameTime, 0.0);
    _frameIndex = frame_index_from_time(_timeFrames);
    _frameRate = positive_finite_or(request.frameRate, 0.0);
}

void JuicerProcessor::setPrintFrameRequest(const PrintFrameRequest& request) {
    setRenderWindow(request.renderWindow);
    _effectsGeometry = request.effectsGeometry;
    _fullFrameExtent = request.fullFrameExtent;
    _framePolicy = request.framePolicy;
    _printStateHold = request.state;
    _directStateHold.reset();
    _sessionSeed = request.sessionSeed;
    _instanceToken = request.instanceToken;
    _clipToken = request.clipToken;
    _timeFrames = finite_or(request.frameTime, 0.0);
    _frameIndex = frame_index_from_time(_timeFrames);
    _frameRate = positive_finite_or(request.frameRate, 0.0);
}

void JuicerProcessor::setInstanceState(InstanceState* s) {
    _instanceState = s;
}

void JuicerProcessor::process() {
    if (_isEnabledCudaRender) {
        processImagesCUDA();
        return;
    }
    JTRACE("SPEKTRAFILM", "FATAL: SpektrafilmCpuPixelPipelineNotImplementedForPhase3C at JuicerProcessor::process");
    OFX::throwSuiteStatusException(kOfxStatErrFatal);
}

// OFX callback adapter for the prepared-state render boundary.
void JuicerProcessor::processImagesCUDA() {
    if (!_srcImg || !_dstImg) {
        return;
    }

    const OfxRectI renderWindow = _renderWindow;
    if (renderWindow.x2 <= renderWindow.x1 || renderWindow.y2 <= renderWindow.y1) {
        return;
    }

    const bool traceInfo = JTRACE_ENABLED(1);
    const bool traceVerbose = JTRACE_ENABLED(3);
    JTRACE_VERBOSE("CUDA", "processImagesCUDA");
    const auto abi_rect = [](const OfxRectI& rect) -> FjRect {
        return {rect.x1, rect.y1, rect.x2, rect.y2};
    };
    const auto image_record = [&](const OFX::Image& image) -> FjImage {
        return {
            reinterpret_cast<std::uintptr_t>(image.getPixelData()),
            abi_rect(image.getBounds()),
            image.getRowBytes(),
            image.getPixelComponents() == OFX::ePixelComponentRGB
                ? FJ_COMPONENTS_RGB
            : image.getPixelComponents() == OFX::ePixelComponentRGBA
                ? FJ_COMPONENTS_RGBA
                : FJ_COMPONENTS_UNKNOWN,
            image.getPixelDepth() == OFX::eBitDepthFloat
                ? FJ_DEPTH_FLOAT32
                : FJ_DEPTH_UNKNOWN};
    };
    const auto& geometry = _effectsGeometry;
    const FjFrame rawFrame{
        image_record(*_srcImg),
        image_record(*_dstImg),
        abi_rect(renderWindow),
        abi_rect(_fullFrameExtent),
        reinterpret_cast<std::uintptr_t>(_pCudaStream),
        (_cudaStreamPropertyPresent ? FJ_FRAME_STREAM_PRESENT : 0U) |
            (traceInfo ? FJ_FRAME_TRACE_INFO : 0U) |
            (traceVerbose ? FJ_FRAME_TRACE_VERBOSE : 0U),
        _framePolicy.geometry.pixel_size_um,
        _timeFrames,
        _frameRate,
        _sessionSeed,
        _clipToken,
        {{geometry.pixelDefinition.x,
          geometry.pixelDefinition.y,
          geometry.pixelDefinition.width,
          geometry.pixelDefinition.height},
         geometry.canonicalX,
         geometry.canonicalY,
         geometry.canonicalWidth,
         geometry.canonicalHeight,
         geometry.scaleX,
         geometry.scaleY,
         geometry.pixelAspectRatio}};

    const bool printRoute = static_cast<bool>(_printStateHold);
    const std::shared_ptr<const DirectRenderState> directState = _directStateHold;
    const std::shared_ptr<const PrintRenderState> printState = _printStateHold;
    const PreparedState* prepared = printRoute
                                        ? (printState ? &printState->prepared : nullptr)
                                        : (directState ? &directState->prepared : nullptr);
    FjPreparedStateView stateView{};
    std::optional<std::uint32_t> route;
    if (prepared) {
        std::array<char, 2048> viewText{};
        FjErrorBuffer viewError{viewText.data(), viewText.size(), 0};
        const FjStatus viewStatus = prepared->view(stateView, viewError);
        if (viewStatus.category != FJ_STATUS_SUCCESS) {
            JTRACE("CUDA", std::string(viewText.data(), viewError.length));
            OFX::throwSuiteStatusException(kOfxStatErrFatal);
        }
        route = stateView.route;
    }

    RenderCompletion completion{&_effect, _instanceState, nullptr, nullptr};
    JuicerCuda::CallScope callScope(
        JuicerCuda::borrowed_owner(), route, &complete_render_call, &completion);
    completion.callScope = &callScope;

    auto should_abort_effect = [this]() -> bool {
        return _effect.abort();
    };
    AbortAccess abortAccess{
        [](void* user) -> bool {
            return (*static_cast<decltype(should_abort_effect)*>(user))();
        },
        &should_abort_effect,
        false};
    completion.abortAccess = &abortAccess;
    const FjAbortCallback abortCallback{&query_abort, &abortAccess};

    std::array<char, 2048> errorText{};
    FjErrorBuffer error{errorText.data(), errorText.size(), 0};
    auto& assets = JuicerProcess::root().assets();
    FjCallCompletionResult result{};
    if (!prepared) {
        result = reject_missing_render_state(
            assets.borrow_rust_assets(), callScope.access(), rawFrame, abortCallback, error);
    } else if (!_instanceState) {
        result = prepared->reject_missing_instance(
            assets.borrow_rust_assets(), callScope.access(), rawFrame, abortCallback, error);
    } else {
        const FjSubmissionAccess submission{
            _instanceToken,
            static_cast<std::uint64_t>(_frameIndex),
            nullptr,
            &select_submission};
        SubmissionSelection selection{_instanceState, traceVerbose};
        FjSubmissionAccess selectedSubmission = submission;
        selectedSubmission.user = &selection;
        result = prepared->render(
            assets.borrow_rust_assets(),
            callScope.access(),
            rawFrame,
            _framePolicy,
            selectedSubmission,
            abortCallback,
            error);
    }

    if (result.fatal != 0) {
        if (!completion.completed) {
            if (!callScope.diagnostic().empty()) {
                JTRACE("CUDA", callScope.diagnostic());
            } else if (error.length != 0) {
                JTRACE("CUDA", std::string(errorText.data(), error.length));
            } else {
                JTRACE("CUDA", "prepared render failed without a diagnostic");
            }
        }
        OFX::throwSuiteStatusException(kOfxStatErrFatal);
    }
}
