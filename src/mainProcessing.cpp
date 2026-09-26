#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <string>

#include <cuda_runtime.h>

#include "Cuda/JuicerCudaExecutor.h"
#include "JuicerState.h"
#include "Logging.h"
#include "ProcessRoot.h"
#include "SpectralData.h"
#include "SpectralProcessing.h"
#include "mainProcessing.h"

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
        cudaError_t error,
        const std::string& detail) {
        if (!instanceState) {
            return;
        }
        if (!JuicerCuda::is_cuda_context_loss_signal(error, detail)) {
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
            msg += std::to_string(static_cast<int>(error));
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
    JuicerCuda::FrameRect native_rect(const OfxRectI& rect) {
        return {rect.x1, rect.y1, rect.x2, rect.y2};
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
    _diffusionFrameSetDescriptor = request.diffusionFrameSet;
    _scatterHalationDescriptor = request.scatterHalation;
    _nComponents = request.components;
    _directStateHold = request.state;
    _printStateHold.reset();
    _sessionSeed = request.sessionSeed;
    _instanceToken = request.instanceToken;
    _clipToken = request.clipToken;
    _timeFrames = finite_or(request.frameTime, 0.0);
    _frameIndex = frame_index_from_time(_timeFrames);
    _frameRate = positive_finite_or(request.frameRate, 0.0);
    _pixelSizeUm =
        is_finite(request.pixelSizeUm)
            ? std::max(0.0f, request.pixelSizeUm)
            : 0.0f;
}

void JuicerProcessor::setPrintFrameRequest(const PrintFrameRequest& request) {
    setRenderWindow(request.renderWindow);
    _effectsGeometry = request.effectsGeometry;
    _fullFrameExtent = request.fullFrameExtent;
    _diffusionFrameSetDescriptor = request.diffusionFrameSet;
    _scatterHalationDescriptor = request.scatterHalation;
    _nComponents = request.components;
    _printStateHold = request.state;
    _directStateHold.reset();
    _sessionSeed = request.sessionSeed;
    _instanceToken = request.instanceToken;
    _clipToken = request.clipToken;
    _timeFrames = finite_or(request.frameTime, 0.0);
    _frameIndex = frame_index_from_time(_timeFrames);
    _frameRate = positive_finite_or(request.frameRate, 0.0);
    _pixelSizeUm =
        is_finite(request.pixelSizeUm)
            ? std::max(0.0f, request.pixelSizeUm)
            : 0.0f;
}

void JuicerProcessor::setInstanceState(InstanceState* s) {
    _instanceState = s;
}

void JuicerProcessor::process() {
    if (_isEnabledCudaRender) {
        OFX::ImageProcessor::process();
        return;
    }
    JTRACE("SPEKTRAFILM", "FATAL: SpektrafilmCpuPixelPipelineNotImplementedForPhase3C at JuicerProcessor::process");
    OFX::throwSuiteStatusException(kOfxStatErrFatal);
}

// FJ_TEMP_BRIDGE: C++ OFX executor adapter; remove S6.D.
void JuicerProcessor::processImagesCUDA() {
    if (!_srcImg || !_dstImg) {
        return;
    }

    if (!(_nComponents == 1 || _nComponents == 3 || _nComponents == 4)) {
        std::string msg = "FATAL: CUDA render requested with unsupported component count=";
        msg += std::to_string(_nComponents);
        JTRACE("CUDA", msg);
        OFX::throwSuiteStatusException(kOfxStatErrFatal);
    }

    const OfxRectI srcBounds = _srcImg->getBounds();
    const OfxRectI dstBounds = _dstImg->getBounds();
    const OfxRectI win = _renderWindow;
    const int width = win.x2 - win.x1;
    const int height = win.y2 - win.y1;
    if (width <= 0 || height <= 0) {
        return;
    }
    const bool traceInfo = JTRACE_ENABLED(1);
    const bool traceVerbose = JTRACE_ENABLED(3);
    const auto should_abort_effect = [this]() -> bool {
        return _effect.abort();
    };

    const int bytesPerPixel = _nComponents * static_cast<int>(sizeof(float));
    const std::ptrdiff_t srcRowBytes = _srcImg->getRowBytes();
    const std::ptrdiff_t dstRowBytes = _dstImg->getRowBytes();
    if (srcRowBytes <= 0 || dstRowBytes <= 0) {
        JTRACE("CUDA", "FATAL: invalid row bytes for CUDA copy");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    const std::ptrdiff_t xSrc = static_cast<std::ptrdiff_t>(win.x1 - srcBounds.x1);
    const std::ptrdiff_t ySrc = static_cast<std::ptrdiff_t>(win.y1 - srcBounds.y1);
    const std::ptrdiff_t xDst = static_cast<std::ptrdiff_t>(win.x1 - dstBounds.x1);
    const std::ptrdiff_t yDst = static_cast<std::ptrdiff_t>(win.y1 - dstBounds.y1);

    const std::ptrdiff_t widthBytes = static_cast<std::ptrdiff_t>(width) * static_cast<std::ptrdiff_t>(bytesPerPixel);
    if (xSrc < 0 || ySrc < 0 || xDst < 0 || yDst < 0 || widthBytes <= 0) {
        JTRACE("CUDA", "FATAL: CUDA render window out of bounds");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    const unsigned char* srcBase = static_cast<const unsigned char*>(_srcImg->getPixelData());
    unsigned char* dstBase = static_cast<unsigned char*>(_dstImg->getPixelData());
    if (!srcBase || !dstBase) {
        JTRACE("CUDA", "FATAL: missing device pointers for CUDA render");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    JuicerCuda::ResourceManager::DeviceContextKey deviceContextKey{};
    try {
        deviceContextKey = JuicerCuda::inspect_frame(srcBase, dstBase, traceInfo);
    } catch (const JuicerCuda::ExecutionFailure&) {
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    if (should_abort_effect()) {
        return;
    }

    const unsigned char* srcPtr = srcBase + ySrc * srcRowBytes + xSrc * bytesPerPixel;
    unsigned char* dstPtr = dstBase + yDst * dstRowBytes + xDst * bytesPerPixel;

    JTRACE_VERBOSE("CUDA", "processImagesCUDA");

    const RenderRecipe* directRecipe =
        (_directStateHold &&
         !Spektrafilm::scan_route_is_print(_directStateHold->recipe.profileRoute.scanRoute))
            ? &_directStateHold->recipe
            : nullptr;
    const FocusedRenderPayload* directPayload =
        directRecipe ? &_directStateHold->payload : nullptr;
    const RenderRecipe* printRecipe =
        (_printStateHold &&
         Spektrafilm::scan_route_is_print(_printStateHold->recipe.profileRoute.scanRoute))
            ? &_printStateHold->recipe
            : nullptr;
    const FocusedRenderPayload* printPayload =
        printRecipe ? &_printStateHold->payload : nullptr;
    const RenderRecipe* focusedRecipe = directRecipe ? directRecipe : printRecipe;
    if (!focusedRecipe) {
        JTRACE("CUDA", "FATAL: render state unavailable; cannot serve CUDA render");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
    if (!_instanceState) {
        JTRACE("CUDA", "FATAL: instance state missing; cannot serve CUDA render");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    const auto frame_domain = [](const OfxRectI& bounds) {
        return Spektrafilm::DiffusionFrameDomain{
            bounds.x1,
            bounds.y1,
            bounds.x2 - bounds.x1,
            bounds.y2 - bounds.y1};
    };
    if (_diffusionFrameSetDescriptor || _scatterHalationDescriptor) {
        const Spektrafilm::DiffusionFrameDomain fullFrameDomain =
            frame_domain(_fullFrameExtent);
        if (fullFrameDomain.width <= 0 || fullFrameDomain.height <= 0 ||
            frame_domain(win) != fullFrameDomain ||
            frame_domain(srcBounds) != fullFrameDomain ||
            frame_domain(dstBounds) != fullFrameDomain) {
            JTRACE(
                "SPEKTRAFILM",
                "ResourceDescriptorMismatch component=scatter_halation field=full_frame_domain");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
    }
    if (_diffusionFrameSetDescriptor) {
        const auto& diffusionFullFrame =
            _diffusionFrameSetDescriptor->fullFrame;
        if (frame_domain(win) != diffusionFullFrame ||
            frame_domain(srcBounds) != diffusionFullFrame ||
            frame_domain(_fullFrameExtent) != diffusionFullFrame) {
            JTRACE(
                "SPEKTRAFILM",
                "ResourceDescriptorMismatch component=diffusion field=full_frame_domain");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
    }

    std::string cameraLensDiagnostic;
    if (!Spektrafilm::preflight_camera_lens_blur(
            focusedRecipe->spatialOptics.cameraLensBlur,
            cameraLensDiagnostic)) {
        JTRACE("SPEKTRAFILM", cameraLensDiagnostic);
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    JuicerCuda::PendingContextLossRecovery pendingContextLossRecovery{};

    auto run_pending_context_loss_recovery = [&]() {
        if (!pendingContextLossRecovery.pending) {
            return;
        }
        recover_context_loss_state(
            _instanceState,
            deviceContextKey,
            pendingContextLossRecovery.stage,
            pendingContextLossRecovery.error,
            pendingContextLossRecovery.detail);
        pendingContextLossRecovery = JuicerCuda::PendingContextLossRecovery{};
    };

    auto run_pending_context_loss_recovery_noexcept = [&]() noexcept {
        try {
#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
            JuicerCuda::ExecutorTest::observe_recovery_start(pendingContextLossRecovery.pending);
#endif
            run_pending_context_loss_recovery();
#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
            JuicerCuda::ExecutorTest::observe_recovery_end();
#endif
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    };

    struct ContextLossRecoveryScope {
        decltype(run_pending_context_loss_recovery_noexcept)* onExit = nullptr;
        ~ContextLossRecoveryScope() noexcept {
            if (onExit) {
                (*onExit)();
            }
        }
    } contextLossRecoveryScope{&run_pending_context_loss_recovery_noexcept};

    const JuicerCuda::DirFailureMessage dirFailureMessage{
        [](void* user, const std::string& diagnostic) noexcept {
            if (diagnostic.find("component=dir") == std::string::npos) {
                return;
            }
            try {
                std::string deliveryText = diagnostic;
                for (std::size_t position = 0;
                     (position = deliveryText.find('%', position)) !=
                     std::string::npos;
                     position += 2u) {
                    deliveryText.insert(position, 1u, '%');
                }
                static_cast<JuicerProcessor*>(user)->_effect.sendMessage(
                    OFX::Message::eMessageError,
                    "FilmJuicerDeferredCudaFailure",
                    deliveryText);
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
        },
        this};

    if (should_abort_effect()) {
        return;
    }

    const FilmRawRecipe* focusedFilmRaw = &focusedRecipe->filmRaw;
    const Spektrafilm::AutoExposureMethod cameraMeteringMethod =
        focusedFilmRaw->autoExposureMethod;
    const JuicerCuda::AutoExposurePreviewDescriptor autoExposureDescriptor =
        JuicerCuda::make_auto_exposure_preview_descriptor(
            native_rect(srcBounds), native_rect(srcBounds), cameraMeteringMethod);
    JuicerCuda::ResourceManager::SubmissionSnapshot snapshot{};
    {
        snapshot.instanceToken.value = _instanceToken;
        snapshot.frameToken.value = static_cast<std::uint64_t>(_frameIndex);
        snapshot.deviceContextKey = deviceContextKey;
        const std::uint64_t uploadCoreHash =
            directPayload ? directPayload->uploadCoreHash : printPayload->uploadCoreHash;
        const std::uint64_t scannerRuntimeHash =
            directPayload ? directPayload->scannerHash : printPayload->scannerHash;
        snapshot.keyDigests =
            JuicerCuda::ResourceManager::make_key_digests(
                uploadCoreHash,
                focusedRecipe->dirCouplers.hash,
                scannerRuntimeHash,
                autoExposureDescriptor.hash);
#if JUICER_DIAGNOSTICS_COMPILED
        bool reusingSnapshotLatch = false;
#endif
        {
            std::lock_guard<std::mutex> latchLock(_instanceState->submissionSnapshotLatchMutex);
            const auto& latched = _instanceState->submissionSnapshotLatch;
            const bool digestsMatch =
                latched.keyDigests.uploadCoreHash == snapshot.keyDigests.uploadCoreHash &&
                latched.keyDigests.dirHash == snapshot.keyDigests.dirHash &&
                latched.keyDigests.scannerHash == snapshot.keyDigests.scannerHash &&
                latched.keyDigests.autoExposureHash == snapshot.keyDigests.autoExposureHash;
            if (_instanceState->submissionSnapshotLatchValid &&
                latched.instanceToken.value == snapshot.instanceToken.value &&
                latched.frameToken.value == snapshot.frameToken.value &&
                latched.deviceContextKey == snapshot.deviceContextKey &&
                digestsMatch &&
                latched.snapshotId != 0) {
                snapshot = latched;
#if JUICER_DIAGNOSTICS_COMPILED
                reusingSnapshotLatch = true;
#endif
            } else {
                std::uint64_t nextSnapshotId =
                    _instanceState->submissionSnapshotIdNext.fetch_add(1, std::memory_order_relaxed);
                if (nextSnapshotId == 0) {
                    nextSnapshotId = _instanceState->submissionSnapshotIdNext.fetch_add(1, std::memory_order_relaxed);
                }
                snapshot.snapshotId = nextSnapshotId;
                _instanceState->submissionSnapshotLatch = snapshot;
                _instanceState->submissionSnapshotLatchValid = true;
            }
        }
#if JUICER_DIAGNOSTICS_COMPILED
        if (traceVerbose) {
            std::string msg;
            msg.reserve(128);
            msg = "path=cuda action=";
            msg += submission_snapshot_action_label(reusingSnapshotLatch);
            msg += " frame_token=";
            msg += std::to_string(snapshot.frameToken.value);
            msg += " snapshot_id=";
            msg += std::to_string(snapshot.snapshotId);
            msg += " instance_token=";
            msg += std::to_string(snapshot.instanceToken.value);
            JTRACE_VERBOSE("MSSNP", msg);
        }
#endif
    }

    const JuicerCuda::ExecutionFrame frame{
        native_rect(srcBounds),
        native_rect(win),
        native_rect(_fullFrameExtent),
        srcBase,
        srcPtr,
        dstPtr,
        srcRowBytes,
        dstRowBytes,
        _nComponents,
        _pCudaStream,
        _diffusionFrameSetDescriptor,
        _scatterHalationDescriptor,
        _effectsGeometry,
        _pixelSizeUm,
        _timeFrames,
        _frameRate,
        _sessionSeed,
        _clipToken,
        autoExposureDescriptor,
        traceInfo,
        traceVerbose};
    try {
        if (directRecipe) {
            JuicerCuda::execute_direct(
                {*directRecipe, *directPayload, frame, snapshot},
                pendingContextLossRecovery,
                dirFailureMessage);
        } else {
            JuicerCuda::execute_print(
                {*printRecipe, *printPayload, frame, snapshot},
                pendingContextLossRecovery,
                dirFailureMessage);
        }
    } catch (const JuicerCuda::ExecutionFailure&) {
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
}
