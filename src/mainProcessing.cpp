// JuicerProcessing.cpp

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <cctype>
#include <string>
#include <atomic>
#include <sstream>
#include <mutex>
#include <limits>
#include <chrono>
#include <thread>

#include "GaussianSciPy.h"

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
#include <cuda_runtime.h>
#include <cuda.h>
#if defined(_WIN32)
#include <windows.h>
#endif
#endif

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__) && defined(JUICER_CUDA_SELF_CHECK) && (JUICER_CUDA_SELF_CHECK != 0)
#include "Cuda/JuicerCudaSelfCheck.h"
#endif

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
#include "Cuda/JuicerCudaResources.h"
#include "Cuda/JuicerCudaPayloads.h"
#include "Cuda/JuicerCudaAutoExposure.h"
#include "Cuda/ResourceManager/JuicerCudaResourceKeys.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#include "Cuda/ResourceManager/JuicerCudaResourceTelemetry.h"
#include "GeneratedColorSpaces.h"
#endif

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
extern "C" cudaError_t juicer_cuda_negative_pipeline(
    const JuicerCuda::PipelineRunParams* hParams,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_build_gate_defect_mask(
    const JuicerCuda::PipelineRunParams* hParams,
    float* dGateMask,
    int gateWidth,
    int gateHeight,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_build_spatial_dir(
    const JuicerCuda::PipelineRunParams* hParams,
    float* dCorrY,
    float* dCorrM,
    float* dCorrC,
    float* dTmp,
    const float* dKernel,
    int kernelRadius,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_negative_pipeline_optics(
    const JuicerCuda::PipelineRunParams* hParams,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    float* dScratchBlurred,
    float* dAux,
    float* dGrainTmp,
    float* dGrainTmpShared,
    float* dGrainTmpMid,
    float* dGrainTmpCoarse,
    const float* dLensBlurKernel,
    int lensBlurRadius,
    const float* dUnsharpKernel,
    int unsharpRadius,
    float unsharpAmount,
    int glareOriginX,
    int glareOriginY,
    std::uint64_t glareSeed,
    float glarePercent,
    float glareRoughness,
    const float* dGlareKernel,
    int glareRadius,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_print_pipeline(
    const JuicerCuda::PipelineRunParams* hParams,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_print_pipeline_optics(
    const JuicerCuda::PipelineRunParams* hParams,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    float* dScratchBlurred,
    float* dAux,
    float* dGrainTmp,
    float* dGrainTmpShared,
    float* dGrainTmpMid,
    float* dGrainTmpCoarse,
    const float* dLensBlurKernel,
    int lensBlurRadius,
    const float* dUnsharpKernel,
    int unsharpRadius,
    float unsharpAmount,
    int glareOriginX,
    int glareOriginY,
    std::uint64_t glareSeed,
    float glarePercent,
    float glareRoughness,
    const float* dGlareKernel,
    int glareRadius,
    void* cudaStreamOpaque);
#endif

// Resolve OFX support library C++ wrappers — suppress MSVC C5040 for dynamic exception specs
#pragma warning(push)
#pragma warning(disable: 5040)
#include "ofxsProcessing.h"
#include "ofxsImageEffect.h"
#pragma warning(pop)
#include "Logging.h"
#include "Hash.h"
#include "SpectralData.h"
#include "SpectralProcessing.h"
#include "FilmProcessing.h"
#include "ColorTransforms.h"
#include "WorkingState.h"
#include "Print.h"
#include "JuicerState.h"
#include "Scanner.h"
#include "OutputEncoding.h"
#include "ScannerOptics.h"
#include "Couplers.h"
#include "mainProcessing.h"
#include "PipelineRunner.h"

namespace {
    constexpr std::uint64_t kSeedPassGrain = 1;
    constexpr std::uint64_t kSeedPassGlare = 2;

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    using CuCtxGetCurrentFn = CUresult(CUDAAPI*)(CUcontext*);

    struct CudaDriverDispatch {
        CuCtxGetCurrentFn cuCtxGetCurrent = nullptr;
        const char* loadError = nullptr;
    };

    const CudaDriverDispatch& cuda_driver_dispatch() {
        static CudaDriverDispatch dispatch{};
        static std::once_flag once;
        std::call_once(once, []() {
#if defined(_WIN32)
            HMODULE module = GetModuleHandleA("nvcuda.dll");
            if (!module) {
                module = LoadLibraryA("nvcuda.dll");
            }
            if (!module) {
                dispatch.loadError = "nvcuda.dll not available";
                return;
            }
            dispatch.cuCtxGetCurrent =
                reinterpret_cast<CuCtxGetCurrentFn>(GetProcAddress(module, "cuCtxGetCurrent"));
            if (!dispatch.cuCtxGetCurrent) {
                dispatch.loadError = "cuCtxGetCurrent symbol not found";
                return;
            }
#else
            dispatch.loadError = "dynamic cuCtxGetCurrent loader unsupported on this platform";
            return;
#endif
        });
        return dispatch;
    }

    bool query_current_cuda_context(void*& outContextOpaque, std::string& outError) {
        outContextOpaque = nullptr;
        outError.clear();

        const CudaDriverDispatch& dispatch = cuda_driver_dispatch();
        if (!dispatch.cuCtxGetCurrent) {
            outError = dispatch.loadError ? dispatch.loadError : "driver dispatch unavailable";
            return false;
        }

        CUcontext currentContext = nullptr;
        const CUresult ctxResult = dispatch.cuCtxGetCurrent(&currentContext);
        if (ctxResult != CUDA_SUCCESS) {
            outError = std::string("cuCtxGetCurrent failed (code=")
                + std::to_string(static_cast<int>(ctxResult)) + ")";
            return false;
        }
        if (!currentContext) {
            outError = "current CUDA context is null";
            return false;
        }

        outContextOpaque = reinterpret_cast<void*>(currentContext);
        return true;
    }

    std::string ascii_lower_copy(const std::string& value) {
        std::string out = value;
        std::transform(out.begin(), out.end(), out.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        return out;
    }

    bool text_has_context_loss_marker(const std::string& text) {
        if (text.empty()) {
            return false;
        }
        const std::string lower = ascii_lower_copy(text);
        return lower.find("context is destroyed") != std::string::npos ||
            lower.find("context destroyed") != std::string::npos ||
            lower.find("cudaerrorcontextisdestroyed") != std::string::npos ||
            lower.find("device lost") != std::string::npos ||
            lower.find("driver shutting down") != std::string::npos ||
            lower.find("context reset") != std::string::npos ||
            lower.find("device unavailable") != std::string::npos ||
            lower.find("cudaerrordeviceuninitialized") != std::string::npos;
    }

    bool is_cuda_context_loss_signal(cudaError_t error, const std::string& detail) {
        if (text_has_context_loss_marker(detail)) {
            return true;
        }
        if (error == cudaSuccess || error == cudaErrorNotReady) {
            return false;
        }
        const char* errorName = cudaGetErrorName(error);
        if (errorName && text_has_context_loss_marker(errorName)) {
            return true;
        }
        const char* errorText = cudaGetErrorString(error);
        if (errorText && text_has_context_loss_marker(errorText)) {
            return true;
        }
        return false;
    }

    void recover_context_loss_slot(
        InstanceState* instanceState,
        const JuicerCuda::ResourceManager::DeviceContextKey& key,
        const char* stage,
        cudaError_t error,
        const std::string& detail) {
        if (!instanceState) {
            return;
        }
        if (!is_cuda_context_loss_signal(error, detail)) {
            return;
        }

        const char* stageName = (stage && *stage) ? stage : "unknown_stage";
        std::string retireError;
        const bool retireAccepted = JuicerCuda::ResourceManager::command_retire_context_reset(
            key,
            retireError);

        bool slotErased = false;
        {
            std::lock_guard<std::mutex> lock(instanceState->cudaMutex);
            const auto it = instanceState->cudaByDevice.find(key);
            if (it != instanceState->cudaByDevice.end()) {
                instanceState->cudaByDevice.erase(it);
                slotErased = true;
            }
        }

        bool latchCleared = false;
        {
            std::lock_guard<std::mutex> lock(instanceState->submissionSnapshotLatchMutex);
            if (instanceState->submissionSnapshotLatchValid &&
                instanceState->submissionSnapshotLatch.deviceContextKey == key) {
                instanceState->submissionSnapshotLatch = JuicerCuda::ResourceManager::SubmissionSnapshot{};
                instanceState->submissionSnapshotLatchValid = false;
                latchCleared = true;
            }
        }

        const std::uintptr_t contextBits = reinterpret_cast<std::uintptr_t>(key.contextOpaque);
        std::string msg = std::string("stage=") + stageName
            + " device_id=" + std::to_string(key.deviceId)
            + " context=" + std::to_string(contextBits)
            + " error_code=" + std::to_string(static_cast<int>(error))
            + " retire_accepted=" + std::to_string(retireAccepted ? 1 : 0)
            + " slot_erased=" + std::to_string(slotErased ? 1 : 0)
            + " latch_cleared=" + std::to_string(latchCleared ? 1 : 0);
        if (!retireError.empty()) {
            msg += " retire_error=" + retireError;
        }
        JTRACE("MSLCY", msg);
    }
#endif

    std::int64_t frame_index_from_time(double time) {
        if (!std::isfinite(time)) {
            return 0;
        }
        return static_cast<std::int64_t>(std::floor(time));
    }

    std::uint64_t safe_session_seed(const InstanceState* state) {
        if (state && state->sessionSeed != 0) {
            return state->sessionSeed;
        }
        return 1;
    }

    std::uint64_t make_seed_base(std::uintptr_t clipToken,
                                 std::int64_t frameIndex,
                                 std::uint64_t sessionSeed,
                                 std::uint64_t passId) {
        const std::uint64_t fields[4] = {
            static_cast<std::uint64_t>(clipToken),
            static_cast<std::uint64_t>(frameIndex),
            sessionSeed,
            passId
        };
        std::uint64_t h = Hash::hash_bytes(fields, sizeof(fields));
        if (h == 0) {
            h = 1;
        }
        return h;
    }

    std::uint64_t make_auto_exposure_reusable_key_hash(
        const OfxRectI& meterBounds,
        const OfxRectI& srcBounds,
        std::ptrdiff_t srcRowBytes,
        int nComponents,
        const Spectral::FilmRawConfig& filmRaw,
        int meteringMethod) {
        std::uint64_t h = Hash::kFnvOffset;
        Hash::hash_bytes_update(h, &meterBounds, sizeof(meterBounds));
        Hash::hash_bytes_update(h, &srcBounds, sizeof(srcBounds));
        Hash::hash_bytes_update(h, &srcRowBytes, sizeof(srcRowBytes));
        Hash::hash_bytes_update(h, &nComponents, sizeof(nComponents));
        Hash::hash_bytes_update(h, &filmRaw.inputColorSpace, sizeof(filmRaw.inputColorSpace));
        Hash::hash_bytes_update(h, &filmRaw.applyCctfDecoding, sizeof(filmRaw.applyCctfDecoding));
        Hash::hash_bytes_update(h, &filmRaw.inputRGBToXYZ, sizeof(filmRaw.inputRGBToXYZ));
        Hash::hash_bytes_update(h, &meteringMethod, sizeof(meteringMethod));
        if (h == 0) {
            h = 1;
        }
        return h;
    }

    float compute_print_midgray_factor_cached(
        InstanceState* instanceState,
        const WorkingState& ws,
        const Print::Runtime& prt,
        const Print::Params& printParams,
        const Couplers::Runtime& dirRT)
    {
        const float yKey = std::isfinite(printParams.yFilter) ? printParams.yFilter : 0.0f;
        const float mKey = std::isfinite(printParams.mFilter) ? printParams.mFilter : 0.0f;
        const float cKey = std::isfinite(printParams.cFilter) ? printParams.cFilter : 0.0f;
        const std::uint64_t neutralFilterHash =
            (prt.neutralFilterHash != 0) ? prt.neutralFilterHash : Print::kDefaultNeutralFilterHash;
        const float exposureCompScale = printParams.exposureCompensationEnabled
            ? printParams.exposureCompensationScale
            : 1.0f;

        if (instanceState) {
            std::lock_guard<std::mutex> lock(instanceState->printMidgrayMutex);
            if (instanceState->printMidgrayValid &&
                instanceState->printMidgrayBuildCounter == ws.buildCounter &&
                instanceState->printMidgrayYShiftSteps == yKey &&
                instanceState->printMidgrayMShiftSteps == mKey &&
                instanceState->printMidgrayCShiftSteps == cKey &&
                instanceState->printMidgrayNeutralFilterHash == neutralFilterHash &&
                instanceState->printMidgrayExposureCompScale == exposureCompScale) {
                return instanceState->printMidgrayFactor;
            }
        }

        float kMid = Pipeline::PipelineRunner::compute_midgray_factor(
            ws,
            prt,
            printParams,
            dirRT,
            exposureCompScale);
        if (!std::isfinite(kMid) || !(kMid > 0.0f)) {
            kMid = 1.0f;
        }

        if (instanceState) {
            std::lock_guard<std::mutex> lock(instanceState->printMidgrayMutex);
            instanceState->printMidgrayValid = true;
            instanceState->printMidgrayBuildCounter = ws.buildCounter;
            instanceState->printMidgrayYShiftSteps = yKey;
            instanceState->printMidgrayMShiftSteps = mKey;
            instanceState->printMidgrayCShiftSteps = cKey;
            instanceState->printMidgrayNeutralFilterHash = neutralFilterHash;
            instanceState->printMidgrayExposureCompScale = exposureCompScale;
            instanceState->printMidgrayFactor = kMid;
        }

        return kMid;
    }

    int stbn_frame_index(std::int64_t frameIndex, int frames, std::uint64_t sessionSeed) {
        if (frames <= 0) {
            return 0;
        }
        const std::int64_t phase = static_cast<std::int64_t>(sessionSeed % static_cast<std::uint64_t>(frames));
        std::int64_t t = frameIndex + phase;
        int f = static_cast<int>(t % frames);
        if (f < 0) {
            f += frames;
        }
        return f;
    }

    int stbn_offset(std::uint64_t sessionSeed, int dim, std::uint64_t salt) {
        if (dim <= 0) {
            return 0;
        }
        const std::uint64_t fields[2] = { sessionSeed, salt };
        const std::uint64_t h = Hash::hash_bytes(fields, sizeof(fields));
        return static_cast<int>(h % static_cast<std::uint64_t>(dim));
    }

    constexpr std::uint64_t kSeedPassWeave = 3;

    double hash_to_unit(std::uint64_t h) {
        // Convert to [0,1) using the top 53 bits (double mantissa).
        constexpr double kInv = 1.0 / 9007199254740992.0; // 2^53
        return static_cast<double>(h >> 11) * kInv;
    }

    double phase_from_seed(std::uint64_t sessionSeed, std::uint64_t passId, int axis, int component) {
        const std::uint64_t fields[4] = {
            sessionSeed,
            passId,
            static_cast<std::uint64_t>(axis),
            static_cast<std::uint64_t>(component)
        };
        std::uint64_t h = Hash::hash_bytes(fields, sizeof(fields));
        if (h == 0) {
            h = 1;
        }
        constexpr double kTwoPi = 6.28318530717958647692;
        return hash_to_unit(h) * kTwoPi;
    }

    double sin_sum(const double* freqs, int count, std::uint64_t sessionSeed, std::uint64_t passId, int axis, int componentOffset, double timeSeconds) {
        constexpr double kTwoPi = 6.28318530717958647692;
        double sum = 0.0;
        for (int i = 0; i < count; ++i) {
            const double phase = phase_from_seed(sessionSeed, passId, axis, componentOffset + i);
            sum += std::sin(kTwoPi * freqs[i] * timeSeconds + phase);
        }
        return sum;
    }

    struct GateWeaveSignal {
        float dxPx = 0.0f;
        float dyPx = 0.0f;
        float cosRot = 1.0f;
        float sinRot = 0.0f;
    };

    GateWeaveSignal compute_gate_weave(
        std::uint64_t sessionSeed,
        double timeSeconds,
        double translateRmsUm,
        double rotateRmsDeg,
        double pixelSizeUm,
        double amount)
    {
        GateWeaveSignal out{};
        if (!(amount > 0.0) || !(pixelSizeUm > 0.0) || !std::isfinite(pixelSizeUm)) {
            return out;
        }

        const double translateRms = translateRmsUm * amount;
        const double rotateRms = rotateRmsDeg * amount;
        if (!(translateRms > 0.0 || rotateRms > 0.0)) {
            return out;
        }

        constexpr double driftFreqs[] = { 0.15, 0.35, 0.80 };
        constexpr double jitterFreqs[] = { 6.0, 12.0 };
        constexpr int driftCount = static_cast<int>(sizeof(driftFreqs) / sizeof(driftFreqs[0]));
        constexpr int jitterCount = static_cast<int>(sizeof(jitterFreqs) / sizeof(jitterFreqs[0]));
        const double driftNorm = 1.0 / std::sqrt(0.5 * static_cast<double>(driftCount));
        const double jitterNorm = 1.0 / std::sqrt(0.5 * static_cast<double>(jitterCount));
        const double driftWeight = 0.85;
        const double jitterWeight = 0.15;
        const double weightNorm = 1.0 / std::sqrt(driftWeight * driftWeight + jitterWeight * jitterWeight);

        for (int axis = 0; axis < 2; ++axis) {
            const double drift = sin_sum(driftFreqs, driftCount, sessionSeed, kSeedPassWeave, axis, 0, timeSeconds) * driftNorm;
            const double jitter = sin_sum(jitterFreqs, jitterCount, sessionSeed, kSeedPassWeave, axis, 10, timeSeconds) * jitterNorm;
            const double composite = (driftWeight * drift + jitterWeight * jitter) * weightNorm;
            const double deltaUm = composite * translateRms;
            const double deltaPx = deltaUm / pixelSizeUm;
            if (axis == 0) {
                out.dxPx = static_cast<float>(deltaPx);
            }
            else {
                out.dyPx = static_cast<float>(deltaPx);
            }
        }

        if (rotateRms > 0.0) {
            const double rotSignal = sin_sum(driftFreqs, driftCount, sessionSeed, kSeedPassWeave, 2, 0, timeSeconds) * driftNorm;
            const double rotDeg = rotSignal * rotateRms;
            const double rotRad = rotDeg * (3.14159265358979323846 / 180.0);
            out.cosRot = static_cast<float>(std::cos(rotRad));
            out.sinRot = static_cast<float>(std::sin(rotRad));
        }
        return out;
    }
}

namespace JuicerProc {

    // Copied from main.cpp helper, unchanged behavior.
    void copyNonFloatRect(OFX::Image* src, OFX::Image* dst) {
        const OfxRectI bounds = src->getBounds();
        const OFX::PixelComponentEnum comps = src->getPixelComponents();
        const OFX::BitDepthEnum depth = src->getPixelDepth();

        int nComponents = 0;
        switch (comps) {
        case OFX::ePixelComponentRGBA: nComponents = 4; break;
        case OFX::ePixelComponentRGB:  nComponents = 3; break;
        case OFX::ePixelComponentAlpha:nComponents = 1; break;
        default: return;
        }
        int bytesPerComp = 0;
        switch (depth) {
        case OFX::eBitDepthUByte:  bytesPerComp = 1; break;
        case OFX::eBitDepthUShort: bytesPerComp = 2; break;
        case OFX::eBitDepthFloat:  bytesPerComp = 4; break;
        default: return;
        }
        const size_t bytesPerPixel = size_t(nComponents * bytesPerComp);
        for (int y = bounds.y1; y < bounds.y2; ++y) {
            for (int x = bounds.x1; x < bounds.x2; ++x) {
                const void* s = src->getPixelAddress(x, y);
                void* d = dst->getPixelAddress(x, y);
                if (!s || !d) continue;
                std::memcpy(d, s, bytesPerPixel);
            }
        }
    }

}

// --- Spatial DIR: defensive curve utilities (monotonic + robust interpolation) ---

static unsigned int compute_thread_count(int width, int height) {
    if (width <= 0 || height <= 0) {
        return 1u;
    }
    const unsigned int w = static_cast<unsigned int>(width);
    const unsigned int h = static_cast<unsigned int>(height);
    unsigned int nCPUs = (std::min(w, 4096u) * h) / 4096u;
    if (nCPUs == 0) {
        nCPUs = 1;
    }
    const unsigned int maxThreads = OFX::MultiThread::getNumCPUs();
    if (maxThreads > 0) {
        nCPUs = std::min(nCPUs, maxThreads);
    }
    return std::max(1u, nCPUs);
}

static std::uint64_t hash_scanner_settings(const Scanner::Settings& settings, const Scanner::Options& options) {
    const std::uint64_t lutHash = Hash::hash_bytes(&settings.useLut, sizeof(settings.useLut));
    const float fields[3] = {
        options.lensBlurSigmaPx,
        options.unsharpSigmaPx,
        options.unsharpAmount
    };
    const std::uint64_t optHash = Hash::hash_float_span(fields, 3);
    if (lutHash == 0 || optHash == 0) {
        JTRACE("HASH", "FATAL: invalid scanner settings for hashing");
        return 0;
    }
    const std::uint64_t combined[2] = { lutHash, optHash };
    return Hash::hash_bytes(combined, sizeof(combined));
}

static std::uint64_t hash_scanner_runtime_lane(
    const WorkingState* ws,
    const Scanner::Settings& settings,
    const Scanner::Options& options,
    std::uint32_t frameBoundsVersion) {
    if (!ws) {
        return 0;
    }
    const std::uint64_t settingsHash = hash_scanner_settings(settings, options);
    if (settingsHash == 0) {
        return 0;
    }

    const std::uint64_t negColorHash = ws->negativeStaticKey.colorRuntimeHash;
    const std::uint64_t negLutRes = static_cast<std::uint64_t>(
        std::clamp(ws->negativeStaticKey.lutResolution, 17u, 128u));
    const bool printValid = ws->printScannerValid;
    const std::uint64_t printColorHash = printValid ? ws->printStaticKey.colorRuntimeHash : 0;
    const std::uint64_t printLutRes = printValid
        ? static_cast<std::uint64_t>(std::clamp(ws->printStaticKey.lutResolution, 17u, 128u))
        : 0;

    const std::uint64_t fields[] = {
        settingsHash,
        static_cast<std::uint64_t>(frameBoundsVersion),
        static_cast<std::uint64_t>(ws->negativeScannerValid ? 1 : 0),
        negColorHash,
        negLutRes,
        static_cast<std::uint64_t>(printValid ? 1 : 0),
        printColorHash,
        printLutRes
    };
    return Hash::hash_bytes(fields, sizeof(fields));
}

struct ScannerPreflightResult {
    const Scanner::ScannerMediumRuntime* mediumRuntime = nullptr;
    const Scanner::ColorRuntime* colorRuntime = nullptr;
    Scanner::ScannerStaticKey staticKey{};
};

static bool validate_scanner_preflight_runtime(
    bool runtimeValid,
    const char* mediumLabel,
    const Scanner::ScannerMediumRuntime* mediumRuntime,
    ScannerPreflightResult& out,
    std::string& outError) {
    out = ScannerPreflightResult{};
    outError.clear();

    const std::string label = (mediumLabel && *mediumLabel) ? mediumLabel : "scanner";
    if (!runtimeValid) {
        outError = label + " scanner runtime invalid";
        return false;
    }
    if (!mediumRuntime) {
        outError = label + " scanner medium runtime missing";
        return false;
    }

    const Spectral::SpectralTables* tables = mediumRuntime->tables;
    if (!tables || tables->K <= 0) {
        outError = label + " scanner spectral tables unavailable";
        return false;
    }

    Scanner::ScannerStaticKey staticKey = mediumRuntime->staticKey;
    if (tables->tablesHash != staticKey.tablesHash) {
        outError = label + " scanner tables hash mismatch for medium";
        return false;
    }
    if (mediumRuntime->range.digest == 0) {
        outError = label + " scanner density range missing or invalid";
        return false;
    }

    const std::uint64_t illumHash = tables->illuminantHash;
    if (illumHash != 0 && mediumRuntime->illuminant.hash != 0 && illumHash != mediumRuntime->illuminant.hash) {
        outError = label + " scanner illuminant hash mismatch for medium";
        return false;
    }

    const Scanner::ColorRuntime* colorPtr = mediumRuntime->color;
    if (!colorPtr || colorPtr->hash == 0) {
        outError = label + " scanner color runtime missing or invalid";
        return false;
    }
    if (staticKey.colorRuntimeHash != colorPtr->hash) {
        outError = label + " scanner static key color hash mismatch";
        return false;
    }

    Scanner::finalize_static_key(staticKey);
    if (staticKey.hash == 0) {
        outError = label + " scanner static key missing or invalid";
        return false;
    }

    out.mediumRuntime = mediumRuntime;
    out.colorRuntime = colorPtr;
    out.staticKey = staticKey;
    return true;
}

static inline bool curve_ok(const Spectral::Curve& c) {
    const size_t N = c.lambda_nm.size();
    if (N < 2 || c.linear.size() != N) return false;
    float prev = c.lambda_nm[0];
    if (!std::isfinite(prev)) return false;
    for (size_t i = 1; i < N; ++i) {
        float xi = c.lambda_nm[i];
        if (!std::isfinite(xi)) return false;
        if (xi < prev) return false; // allow duplicates (xi == prev), but never decreasing
        prev = xi;
    }
    return true;
}


// JuicerProcessor method definitions matching JuicerProcessing.h

JuicerProcessor::JuicerProcessor(OFX::ImageEffect& effect)
    : OFX::ImageProcessor(effect)
    , _srcImg(nullptr)
    , _nComponents(0)
    , _scannerOptions{}
    , _scannerSettings{}
    , _printParams{}
    , _halationOverride{}
    , _hasHalationOverride(false)
    , _grainOverride{}
    , _hasGrainOverride(false)
    , _printGlareOverride{}
    , _hasPrintGlareOverride(false)
    , _dirRT{}
    , _prt(nullptr)
    , _ws(nullptr)
    , _wsReady(false)
    , _printReady(false)
    , _exposureScale(1.0f)
    , _outputEncoding{}
    , _scratch{}
    , _density{}
    , _frameBoundsVersion(0)
    , _pixelSizeUm(0.0f)
{
}

void JuicerProcessor::setSrcDst(OFX::Image* src, OFX::Image* dst) {
    _srcImg = src;
    setDstImg(dst);
}

void JuicerProcessor::setRenderWindowRect(const OfxRectI& rect) { setRenderWindow(rect); }
void JuicerProcessor::setComponents(int n) { _nComponents = n; }
void JuicerProcessor::setScannerOptions(const Scanner::Options& o) { _scannerOptions = o; }
void JuicerProcessor::setScannerSettings(const Scanner::Settings& s) { _scannerSettings = s; }
void JuicerProcessor::setPrintParams(const Print::Params& p) { _printParams = p; }
void JuicerProcessor::setHalationOverride(const Profiles::HalationMetadata& halation) {
    _halationOverride = halation;
    _hasHalationOverride = true;
}
void JuicerProcessor::setGrainOverride(const Profiles::GrainMetadata& grain) {
    _grainOverride = grain;
    _hasGrainOverride = true;
}
void JuicerProcessor::setPrintGlareOverride(const Profiles::ProfileGlare& glare) {
    _printGlareOverride = glare;
    _printGlareOverride.compensationRemovalFactor = 0.0f;
    _printGlareOverride.compensationRemovalDensity = 0.0f;
    _printGlareOverride.compensationRemovalTransition = 0.0f;
    _hasPrintGlareOverride = true;
}
void JuicerProcessor::setDirRuntime(const Couplers::Runtime& rt) { _dirRT = rt; }
void JuicerProcessor::setWorkingState(const WorkingState* ws, bool wsReady) {
    _ws = ws;
    _wsReady = wsReady;
    // Align DIR normalization constants to per-instance maxima if available
    if (_wsReady && _ws) {
        _dirRT.dMax[0] = (std::isfinite(_ws->dMax[0]) && _ws->dMax[0] > 1e-4f) ? _ws->dMax[0] : 1.0f;
        _dirRT.dMax[1] = (std::isfinite(_ws->dMax[1]) && _ws->dMax[1] > 1e-4f) ? _ws->dMax[1] : 1.0f;
        _dirRT.dMax[2] = (std::isfinite(_ws->dMax[2]) && _ws->dMax[2] > 1e-4f) ? _ws->dMax[2] : 1.0f;
    }
}
void JuicerProcessor::setPrintRuntime(const Print::Runtime* prt, bool printReady) { _prt = prt; _printReady = printReady; }
void JuicerProcessor::setExposure(float exposureScale) {
    _exposureScale = exposureScale;
}

void JuicerProcessor::setCameraAutoExposure(bool enabled, int meteringMethod, double sliderEV) {
    _cameraAutoEnabled = enabled;
    _cameraMeteringMethod = meteringMethod;
    _cameraSliderEV = sliderEV;
}

void JuicerProcessor::setOutputEncoding(const OutputEncoding::Params& p) {
    _outputEncoding = p;
}

void JuicerProcessor::setInstanceState(InstanceState* s) {
    _instanceState = s;
}

void JuicerProcessor::setClipToken(std::uintptr_t token) {
    _clipToken = token;
}

void JuicerProcessor::setGateWeaveAmount(double amount) {
    if (std::isfinite(amount)) {
        _gateWeaveAmount = amount;
    }
}

void JuicerProcessor::setFrameTime(double time) {
    _timeFrames = time;
    _frameIndex = frame_index_from_time(time);
    _frameTimeHash = Hash::hash_bytes(&_frameIndex, sizeof(_frameIndex));
    if (_frameTimeHash == 0) {
        _frameTimeHash = 1;
    }
}

void JuicerProcessor::setFrameRate(double frameRate) {
    if (std::isfinite(frameRate) && frameRate > 0.0) {
        _frameRate = frameRate;
    }
    else {
        _frameRate = 0.0;
    }
}

void JuicerProcessor::setFrameBoundsVersion(std::uint32_t v) {
    _frameBoundsVersion = v;
}

void JuicerProcessor::setPixelSizeUm(float pixelSizeUm) {
    _pixelSizeUm = pixelSizeUm;
}

void JuicerProcessor::setRenderHints(bool interactiveRenderStatus, bool renderQualityDraft, bool sequentialRenderStatus) {
    _renderInteractiveStatus = interactiveRenderStatus;
    _renderQualityDraft = renderQualityDraft;
    _renderSequentialStatus = sequentialRenderStatus;
}

JuicerProcessor::RenderContext JuicerProcessor::prepareRenderContext() const {
    RenderContext ctx{};
    ctx.window = _renderWindow;
    ctx.width = _renderWindow.x2 - _renderWindow.x1;
    ctx.height = _renderWindow.y2 - _renderWindow.y1;
    ctx.exposureScaleSafe = (std::isfinite(_exposureScale) && _exposureScale > 0.0f)
        ? _exposureScale
        : 1.0f;
    ctx.useSpatialDIR = (_dirRT.active && std::isfinite(_dirRT.spatialSigmaPixels) &&
        _dirRT.spatialSigmaPixels > 0.0f && _nComponents >= 3 && _wsReady && _ws);
    ctx.printActive = (_wsReady && _ws && _printReady && _prt && !_printParams.bypass);

    ctx.kMidSpectral = 1.0f;
    if (ctx.printActive) {
        ctx.kMidSpectral = compute_print_midgray_factor_cached(
            _instanceState,
            *_ws,
            *_prt,
            _printParams,
            _dirRT);
    }

    ctx.pixelSizeUm = (std::isfinite(_pixelSizeUm) && _pixelSizeUm > 0.0f) ? _pixelSizeUm : 0.0f;
    return ctx;
}

bool JuicerProcessor::ensureDensityCapacity(int width, int height) {
    if (width <= 0 || height <= 0) {
        return false;
    }
    const size_t planeSize = size_t(width) * size_t(height);
    try {
        _density.c.resize(planeSize);
        _density.m.resize(planeSize);
        _density.y.resize(planeSize);
    }
    catch (...) {
        JTRACE("SCAN", "FATAL: failed to allocate density slab");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
    _density.width = width;
    _density.height = height;
    _density.originX = _renderWindow.x1;
    _density.originY = _renderWindow.y1;
    _density.stride = width;
    return true;
}

void JuicerProcessor::writeMediumDensities(const RenderContext& ctx, unsigned int threadCount) {
    if (!_ws || !_wsReady || ctx.width <= 0 || ctx.height <= 0) {
        return;
    }

    _density.medium = ctx.printActive
        ? Scanner::ScannerMedium::Print
        : Scanner::ScannerMedium::Negative;

    if (ctx.useSpatialDIR) {
        struct SpatialDIRUser {
            OFX::ImageEffect* effect = nullptr;
            OFX::Image* srcImg = nullptr;
            OfxRectI window{};
        };
        SpatialDIRUser user{};
        user.effect = &_effect;
        user.srcImg = _srcImg;
        user.window = ctx.window;

        SpatialDIR::Callbacks callbacks{};
        callbacks.user = &user;
        callbacks.fetchRGB = [](void* u, int xx, int yy, float rgb[3]) -> bool {
            auto* self = static_cast<SpatialDIRUser*>(u);
            const int x = self->window.x1 + xx;
            const int y = self->window.y1 + yy;
            const float* srcPix = reinterpret_cast<const float*>(self->srcImg->getPixelAddress(x, y));
            if (!srcPix) return false;
            rgb[0] = srcPix[0];
            rgb[1] = srcPix[1];
            rgb[2] = srcPix[2];
            return true;
            };
        callbacks.abortCheck = [](void* u) -> bool {
            auto* self = static_cast<SpatialDIRUser*>(u);
            return self->effect->abort();
            };

        SpatialDIR::buildSpatialDIRCorrections(
            ctx.width,
            ctx.height,
            *_ws,
            _dirRT,
            ctx.exposureScaleSafe,
            callbacks,
            _scratch.dirWorkspace,
            _scratch.gaussianKernel);
    }

    std::atomic<bool> abortFlag{ false };
    std::atomic<bool> failure{ false };
    const int width = ctx.width;
    const int height = ctx.height;
    const int originX = ctx.window.x1;
    const int originY = ctx.window.y1;

    const unsigned int nThreads = std::max(1u, threadCount);
    if (ctx.printActive) {
        _scratch.printScratchPerWorker.resize(nThreads);
    }

    Pipeline::PipelineRunnerConfig runnerCfg{};
    runnerCfg.enablePrint = ctx.printActive;
    const Pipeline::PipelineRunner runner(runnerCfg);

    struct DensityProcessor final : OFX::MultiThread::Processor {
        JuicerProcessor& self;
        const RenderContext& ctx;
        const Pipeline::PipelineRunner& runner;
        std::atomic<bool>& abortFlag;
        std::atomic<bool>& failure;
        const int width;
        const int height;
        const int originX;
        const int originY;

        DensityProcessor(
            JuicerProcessor& self_,
            const RenderContext& ctx_,
            const Pipeline::PipelineRunner& runner_,
            std::atomic<bool>& abortFlag_,
            std::atomic<bool>& failure_,
            int width_,
            int height_,
            int originX_,
            int originY_)
            : self(self_)
            , ctx(ctx_)
            , runner(runner_)
            , abortFlag(abortFlag_)
            , failure(failure_)
            , width(width_)
            , height(height_)
            , originX(originX_)
            , originY(originY_)
        {
        }

        void multiThreadFunction(unsigned int threadId, unsigned int nThreads) override {
            const int rowsPerThread = (height + int(nThreads) - 1) / int(nThreads);
            const int yStart = rowsPerThread * int(threadId);
            if (yStart >= height) {
                return;
            }
            const int yEnd = std::min(height, rowsPerThread * int(threadId + 1));

            JuicerProc::PrintPipelineScratch* printScratch = nullptr;
            if (ctx.printActive) {
                if (threadId < self._scratch.printScratchPerWorker.size()) {
                    printScratch = &self._scratch.printScratchPerWorker[threadId];
                }
            }

            for (int yOff = yStart; yOff < yEnd && !abortFlag.load(std::memory_order_relaxed); ++yOff) {
                if (self._effect.abort()) {
                    abortFlag.store(true, std::memory_order_relaxed);
                    break;
                }
                const int y = originY + yOff;
                const size_t rowOffset = size_t(yOff) * size_t(width);
                for (int xOff = 0; xOff < width; ++xOff) {
                    if (abortFlag.load(std::memory_order_relaxed)) {
                        break;
                    }
                    const int x = originX + xOff;
                    const size_t idx = rowOffset + size_t(xOff);

                    Pipeline::DensityPixelInputs pxIn{};
                    pxIn.exposureScale = ctx.exposureScaleSafe;
                    pxIn.dirRuntime = &self._dirRT;
                    pxIn.applyDirRuntime = true;

                    if (ctx.useSpatialDIR) {
                        pxIn.useFilmRawOverride = true;
                        pxIn.filmRawOverride.v[0] = self._scratch.dirWorkspace.filmRaw_B[idx];
                        pxIn.filmRawOverride.v[1] = self._scratch.dirWorkspace.filmRaw_G[idx];
                        pxIn.filmRawOverride.v[2] = self._scratch.dirWorkspace.filmRaw_R[idx];
                        pxIn.useSpatialDIR = true;
                        pxIn.spatialLogECorrectionsYMC[0] = self._scratch.dirWorkspace.corrYBlur[idx];
                        pxIn.spatialLogECorrectionsYMC[1] = self._scratch.dirWorkspace.corrMBlur[idx];
                        pxIn.spatialLogECorrectionsYMC[2] = self._scratch.dirWorkspace.corrCBlur[idx];
                    }
                    else {
                        const float* srcPix = reinterpret_cast<const float*>(self._srcImg->getPixelAddress(x, y));
                        if (!srcPix) {
                            self._density.c[idx] = 0.0f;
                            self._density.m[idx] = 0.0f;
                            self._density.y[idx] = 0.0f;
                            continue;
                        }
                        pxIn.rgb.v[0] = srcPix[0];
                        pxIn.rgb.v[1] = srcPix[1];
                        pxIn.rgb.v[2] = srcPix[2];
                    }

                    if (ctx.printActive) {
                        pxIn.printRuntime = self._prt;
                        pxIn.printParams = &self._printParams;
                        pxIn.midgrayFactor = ctx.kMidSpectral;
                        pxIn.printScratch = printScratch;
                    }

                    Pipeline::DensityPixelOutputs pxOut{};
                    if (!runner.run_density_pixel(*self._ws, pxIn, pxOut)) {
                        if (ctx.printActive) {
                            failure.store(true, std::memory_order_relaxed);
                            abortFlag.store(true, std::memory_order_relaxed);
                            break;
                        }
                        self._density.c[idx] = 0.0f;
                        self._density.m[idx] = 0.0f;
                        self._density.y[idx] = 0.0f;
                        continue;
                    }

                    if (ctx.printActive) {
                        if (pxOut.medium != Pipeline::DensityMedium::Print) {
                            failure.store(true, std::memory_order_relaxed);
                            abortFlag.store(true, std::memory_order_relaxed);
                            break;
                        }
                        self._density.c[idx] = pxOut.printDensity.v[0];
                        self._density.m[idx] = pxOut.printDensity.v[1];
                        self._density.y[idx] = pxOut.printDensity.v[2];
                    }
                    else {
                        self._density.c[idx] = pxOut.negativeDensity.v[0];
                        self._density.m[idx] = pxOut.negativeDensity.v[1];
                        self._density.y[idx] = pxOut.negativeDensity.v[2];
                    }
                }
            }
        }
    };

    DensityProcessor densityProcessor(
        *this,
        ctx,
        runner,
        abortFlag,
        failure,
        width,
        height,
        originX,
        originY);
    densityProcessor.multiThread(nThreads);

    if (failure.load(std::memory_order_relaxed)) {
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
    if (abortFlag.load(std::memory_order_relaxed)) {
        return;
    }
}


void JuicerProcessor::renderScannerFromDensity(const RenderContext& ctx, unsigned int threadCount) {
    if (!_ws || !_wsReady) {
        return;
    }

    // The scanner now consumes only the staged CMY density slab; legacy RGB entry points are removed.
    Scanner::ScannerMediumRuntime printMediumOverride{};
    const Scanner::ScannerMediumRuntime* mediumRuntime = ctx.printActive
        ? &_ws->printMediumRuntime
        : &_ws->negativeMediumRuntime;
    const bool scannerRuntimeValid = ctx.printActive ? _ws->printScannerValid : _ws->negativeScannerValid;
    if (ctx.printActive) {
        printMediumOverride = _ws->printMediumRuntime;
        if (_hasPrintGlareOverride) {
            printMediumOverride.glare.active = _printGlareOverride.active;
            printMediumOverride.glare.percent = _printGlareOverride.percent;
            printMediumOverride.glare.roughness = _printGlareOverride.roughness;
            printMediumOverride.glare.blur = _printGlareOverride.blur;
            printMediumOverride.glare.compensationRemovalFactor = 0.0f;
            printMediumOverride.glare.compensationRemovalDensity = 0.0f;
            printMediumOverride.glare.compensationRemovalTransition = 0.0f;
        }
        const std::uint64_t glareHash = Scanner::hash_glare(printMediumOverride.glare);
        if (glareHash == 0) {
            JTRACE("HASH", "FATAL: failed to hash print glare override parameters");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        printMediumOverride.staticKey.glareHash = glareHash;
        mediumRuntime = &printMediumOverride;
    }

    ScannerPreflightResult scannerPreflight{};
    std::string scannerPreflightError;
    if (!validate_scanner_preflight_runtime(
            scannerRuntimeValid,
            ctx.printActive ? "print" : "negative",
            mediumRuntime,
            scannerPreflight,
            scannerPreflightError)) {
        JTRACE("MSSKV", std::string("path=cpu result=fail medium=") + (ctx.printActive ? "print" : "negative")
            + " reason=" + scannerPreflightError);
        JTRACE("SCAN", std::string("FATAL: ") + scannerPreflightError);
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
    JTRACE_VERBOSE("MSSKV", std::string("path=cpu result=ok medium=") + (ctx.printActive ? "print" : "negative")
        + " static_key_hash=" + std::to_string(scannerPreflight.staticKey.hash));

    mediumRuntime = scannerPreflight.mediumRuntime;
    const Scanner::ColorRuntime* colorPtr = scannerPreflight.colorRuntime;
    Scanner::ScannerStaticKey staticKey = scannerPreflight.staticKey;

    if (_density.medium != scannerPreflight.mediumRuntime->medium) {
        JTRACE("SCAN", "FATAL: density slab medium does not match selected scanner medium");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    Scanner::ScannerRuntimeKey runtimeKey{};
    runtimeKey.settingsHash = hash_scanner_settings(_scannerSettings, _scannerOptions);
    runtimeKey.frameBoundsVersion = _frameBoundsVersion;
    if (runtimeKey.settingsHash == 0) {
        JTRACE("HASH", "FATAL: scanner runtime settings hash invalid");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
    Scanner::finalize_runtime_key(runtimeKey);

    Scanner::ScannerKey scannerKey{};
    scannerKey.staticKey = staticKey;
    scannerKey.runtimeKey = runtimeKey;
    Scanner::finalize_scanner_key(scannerKey);
    if (scannerKey.hash == 0) {
        JTRACE("HASH", "FATAL: scanner combined key invalid");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    struct ScannerRuntimeLease {
        InstanceState* state = nullptr;
        ScannerOptics::Runtime* runtime = nullptr;
        std::atomic<bool>* inUse = nullptr;
        const char* slotName = "none";

        explicit ScannerRuntimeLease(InstanceState* s) : state(s) {}

        ScannerOptics::Runtime* acquire() {
            if (!state) {
                slotName = "none";
                return nullptr;
            }
            bool expected = false;
            if (state->scannerRuntimeAInUse.compare_exchange_strong(
                    expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
                inUse = &state->scannerRuntimeAInUse;
                runtime = &state->scannerRuntimeA;
                slotName = "A";
                return runtime;
            }
            expected = false;
            if (state->scannerRuntimeBInUse.compare_exchange_strong(
                    expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
                inUse = &state->scannerRuntimeBInUse;
                runtime = &state->scannerRuntimeB;
                slotName = "B";
                return runtime;
            }
            slotName = "none";
            return nullptr;
        }

        ~ScannerRuntimeLease() {
            if (inUse) {
                inUse->store(false, std::memory_order_release);
            }
        }
    };

    ScannerRuntimeLease runtimeLease(_instanceState);
    ScannerOptics::Runtime* opticsRuntime = runtimeLease.acquire();
    constexpr std::uint32_t kScannerRuntimeLeaseMaxWaitUs = 16000u;
    constexpr std::uint32_t kScannerRuntimeLeasePollSleepUs = 50u;
    bool waitedForLease = false;
    const auto leaseWaitStart = std::chrono::steady_clock::now();
    while (!opticsRuntime) {
        waitedForLease = true;
        if (_effect.abort()) {
            JTRACE_VERBOSE("MSSRL", "event=runtime_lease outcome=abort");
            return;
        }
        const auto elapsedUs = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - leaseWaitStart).count());
        if (elapsedUs >= kScannerRuntimeLeaseMaxWaitUs) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(kScannerRuntimeLeasePollSleepUs));
        opticsRuntime = runtimeLease.acquire();
    }
    const std::uint64_t leaseWaitUs = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - leaseWaitStart).count());
    if (!opticsRuntime) {
        const std::string msg = std::string("event=runtime_lease outcome=timeout")
            + " wait_us=" + std::to_string(leaseWaitUs)
            + " wait_budget_us=" + std::to_string(static_cast<unsigned long long>(kScannerRuntimeLeaseMaxWaitUs));
        JTRACE("MSSRL", msg);
        JTRACE("SCAN", "FATAL: scanner runtime lease unavailable after bounded wait");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
    JTRACE_VERBOSE("MSSRL", std::string("event=runtime_lease outcome=")
        + (waitedForLease ? "wait_acquired" : "acquired")
        + " wait_us=" + std::to_string(leaseWaitUs)
        + " slot=" + runtimeLease.slotName);

    const std::uint64_t sessionSeed = safe_session_seed(_instanceState);
    const std::uint64_t seedBase = make_seed_base(_clipToken, _frameIndex, sessionSeed, kSeedPassGlare);

    ScannerOptics::RenderContext optCtx{};
    optCtx.medium = mediumRuntime;
    optCtx.density = &_density;
    optCtx.runtime = opticsRuntime;
    optCtx.srcImage = _srcImg;
    optCtx.color = colorPtr;
    optCtx.nComponents = _nComponents;
    optCtx.bounds = ctx.window;
    optCtx.copyAlpha = (_nComponents == 4);
    optCtx.dstView.originX = ctx.window.x1;
    optCtx.dstView.originY = ctx.window.y1;
    optCtx.dstView.width = ctx.width;
    optCtx.dstView.height = ctx.height;
    optCtx.dstView.strideBytes = _dstImg ? _dstImg->getRowBytes() : 0;
    optCtx.dstView.r = (_dstImg)
        ? reinterpret_cast<float*>(_dstImg->getPixelAddress(ctx.window.x1, ctx.window.y1))
        : nullptr;
    optCtx.dstView.g = optCtx.dstView.r;
    optCtx.dstView.b = optCtx.dstView.r;
    optCtx.dstView.a = (_dstImg && _nComponents == 4) ? optCtx.dstView.r + 3 : nullptr;
    optCtx.options = _scannerOptions;
    optCtx.settings = _scannerSettings;
    optCtx.runtimeKey = runtimeKey;
    optCtx.scannerKey = scannerKey;
    optCtx.seedBase = seedBase;
    optCtx.hasBaseline = (_ws ? _ws->hasBaseline : false);
    optCtx.threadCount = std::max(1u, threadCount);
    optCtx.abort.shouldAbort = [this]() -> bool { return _effect.abort(); };

    ScannerOptics::render_density_to_rgb(optCtx);
}

void JuicerProcessor::processImpl() {
    if (!_srcImg || !_dstImg) return;

    if (_isEnabledOpenCLRender || _isEnabledCudaRender || _isEnabledMetalRender) {
        // CPU-only staging layer: GPU/device paths are intentionally disabled until parity lands.
        JTRACE("SCAN", "FATAL: GPU paths are unsupported in scanner staging");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    if (_nComponents < 1) {
        return;
    }

    const bool wsReady = _wsReady && _ws;
    if (!wsReady) {
        JTRACE("BUILD", "FATAL: working state unavailable; cannot render");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
    if (_nComponents < 3) {
        for (int y = _renderWindow.y1; y < _renderWindow.y2; ++y) {
            for (int x = _renderWindow.x1; x < _renderWindow.x2; ++x) {
                float* dstPix = reinterpret_cast<float*>(_dstImg->getPixelAddress(x, y));
                const float* srcPix = reinterpret_cast<const float*>(_srcImg->getPixelAddress(x, y));
                if (!dstPix || !srcPix) {
                    continue;
                }
                if (_nComponents == 1) {
                    dstPix[0] = srcPix[0];
                }
            }
        }
        return;
    }

#if defined(JUICER_SPD_DEBUG)
    Spectral::spd_probe_reset();
#endif

    RenderContext ctx = prepareRenderContext();
    if (ctx.width <= 0 || ctx.height <= 0) {
        return;
    }

    const unsigned int threadCount = compute_thread_count(ctx.width, ctx.height);
    ensureDensityCapacity(ctx.width, ctx.height);
    writeMediumDensities(ctx, threadCount);
    if (_effect.abort()) {
        return;
    }
    renderScannerFromDensity(ctx, threadCount);
}

void JuicerProcessor::process() {
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
    // CUDA-only mode: refuse CPU/OpenCL/Metal entry points.
    if (!_isEnabledCudaRender) {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
#endif
    if (_isEnabledOpenCLRender || _isEnabledCudaRender || _isEnabledMetalRender) {
        OFX::ImageProcessor::process();
        return;
    }
    processImpl();
}

void JuicerProcessor::multiThreadProcessImages(OfxRectI) {
    processImpl();
}

void JuicerProcessor::processImagesCUDA() {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
    OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
#else
    enum class RenderMode {
        NegativeOnly,
        Print
    };

    if (!_srcImg || !_dstImg) {
        return;
    }

    if (!(_nComponents == 1 || _nComponents == 3 || _nComponents == 4)) {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }

    const OfxRectI srcBounds = _srcImg->getBounds();
    const OfxRectI dstBounds = _dstImg->getBounds();
    const OfxRectI win = _renderWindow;
    const int width = win.x2 - win.x1;
    const int height = win.y2 - win.y1;
    if (width <= 0 || height <= 0) {
        return;
    }

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

    int deviceId = -1;
    void* contextOpaque = nullptr;
    {
        cudaPointerAttributes srcAttr{};
        cudaError_t attrErr = cudaPointerGetAttributes(&srcAttr, srcBase);
#if CUDART_VERSION >= 10000
        if (attrErr == cudaSuccess) {
            deviceId = srcAttr.device;
        }
#else
        if (attrErr == cudaSuccess) {
            deviceId = srcAttr.device;
        }
#endif

        cudaPointerAttributes dstAttr{};
        cudaError_t dstAttrErr = cudaPointerGetAttributes(&dstAttr, dstBase);
        if (dstAttrErr == cudaSuccess && deviceId >= 0 && dstAttr.device != deviceId) {
            JTRACE("CUDA", "FATAL: source/destination device mismatch");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }

        if (deviceId < 0) {
            int cur = -1;
            cudaError_t devErr = cudaGetDevice(&cur);
            if (devErr != cudaSuccess || cur < 0) {
                JTRACE("CUDA", "FATAL: failed to determine CUDA device for OFX pointers");
                throw OFX::Exception::Suite(kOfxStatErrFatal);
            }
            deviceId = cur;
        }

        cudaError_t setErr = cudaSetDevice(deviceId);
        if (setErr != cudaSuccess) {
            const char* msg = cudaGetErrorString(setErr);
            JTRACE("CUDA", std::string("FATAL: cudaSetDevice failed: ") + (msg ? msg : "(unknown)"));
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }

        std::string contextError;
        if (!query_current_cuda_context(contextOpaque, contextError)) {
            JTRACE("CUDA", std::string("FATAL: failed to capture CUDA context identity: ") + contextError);
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
    }

    if (_effect.abort()) {
        return;
    }

    const unsigned char* srcPtr = srcBase + ySrc * srcRowBytes + xSrc * bytesPerPixel;
    unsigned char* dstPtr = dstBase + yDst * dstRowBytes + xDst * bytesPerPixel;

    JTRACE_VERBOSE("CUDA", "processImagesCUDA");

    const bool wsReady = _wsReady && _ws;
    if (!wsReady) {
        JTRACE("CUDA", "FATAL: working state unavailable; cannot serve CUDA render");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
    if (!_instanceState) {
        JTRACE("CUDA", "FATAL: instance state missing; cannot serve CUDA render");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    JuicerCuda::ResourceManager::DeviceContextKey deviceContextKey{};
    deviceContextKey.deviceId = deviceId;
    deviceContextKey.contextOpaque = contextOpaque;

    struct PendingContextLossRecovery {
        bool pending = false;
        cudaError_t error = cudaSuccess;
        const char* stage = nullptr;
        std::string detail;
    } pendingContextLossRecovery{};

    auto mark_context_loss_recovery = [&](const char* stage, cudaError_t error, const std::string& detail) {
        if (pendingContextLossRecovery.pending) {
            return;
        }
        if (!is_cuda_context_loss_signal(error, detail)) {
            return;
        }
        pendingContextLossRecovery.pending = true;
        pendingContextLossRecovery.error = error;
        pendingContextLossRecovery.stage = stage;
        pendingContextLossRecovery.detail = detail;
    };

    auto run_pending_context_loss_recovery = [&]() {
        if (!pendingContextLossRecovery.pending) {
            return;
        }
        recover_context_loss_slot(
            _instanceState,
            deviceContextKey,
            pendingContextLossRecovery.stage,
            pendingContextLossRecovery.error,
            pendingContextLossRecovery.detail);
        pendingContextLossRecovery = PendingContextLossRecovery{};
    };

    auto run_pending_context_loss_recovery_noexcept = [&]() noexcept {
        try {
            run_pending_context_loss_recovery();
        }
        catch (...) {
        }
    };

    struct ContextLossRecoveryScope {
        decltype(run_pending_context_loss_recovery_noexcept)* onExit = nullptr;
        ~ContextLossRecoveryScope() noexcept {
            if (onExit) {
                (*onExit)();
            }
        }
    } contextLossRecoveryScope{ &run_pending_context_loss_recovery_noexcept };

    JuicerCuda::Resources* cudaResources = nullptr;
    {
        std::lock_guard<std::mutex> lock(_instanceState->cudaMutex);
        auto& slot = _instanceState->cudaByDevice[deviceContextKey];
        if (!slot) {
            slot.reset(JuicerCuda::create());
            if (!slot) {
                JTRACE("CUDA", "FATAL: failed to allocate CUDA resources");
                throw OFX::Exception::Suite(kOfxStatErrFatal);
            }
        }
        cudaResources = slot.get();
    }

    if (_effect.abort()) {
        return;
    }

    const std::uint64_t autoExposureReusableKeyHash = make_auto_exposure_reusable_key_hash(
        srcBounds,
        srcBounds,
        srcRowBytes,
        _nComponents,
        _ws->filmRaw,
        _cameraMeteringMethod);

    JuicerCuda::ResourceManager::SubmissionTransaction submissionTxn{};
    struct SubmissionTxnScope {
        JuicerCuda::ResourceManager::SubmissionTransaction* transaction = nullptr;
        bool committed = false;

        ~SubmissionTxnScope() {
            if (transaction && !committed) {
                JuicerCuda::ResourceManager::rollback_submission(*transaction, "scope_exit");
            }
        }
    } submissionTxnScope{ &submissionTxn, false };
    {
        JuicerCuda::ResourceManager::SubmissionSnapshot snapshot{};
        snapshot.instanceToken.value =
            (_instanceState->instanceToken != 0) ? _instanceState->instanceToken : safe_session_seed(_instanceState);
        snapshot.frameToken.value = static_cast<std::uint64_t>(_frameIndex);
        snapshot.deviceContextKey = deviceContextKey;
        const std::uint64_t uploadCoreHash =
            (_ws->uploadCoreHash != 0) ? _ws->uploadCoreHash : _ws->coreHash;
        const std::uint64_t scannerRuntimeHash = hash_scanner_runtime_lane(
            _ws,
            _scannerSettings,
            _scannerOptions,
            _frameBoundsVersion);
        snapshot.keyDigests =
            JuicerCuda::ResourceManager::make_key_digests(
                uploadCoreHash,
                _ws->dirHash,
                scannerRuntimeHash,
                autoExposureReusableKeyHash);
        snapshot.keySchemaVersion = 1;
        snapshot.traceSchemaVersion = JuicerCuda::ResourceManager::kTraceSchemaVersion;
        bool reusingSnapshotLatch = false;
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
                latched.keySchemaVersion == snapshot.keySchemaVersion &&
                latched.traceSchemaVersion == snapshot.traceSchemaVersion &&
                digestsMatch &&
                latched.snapshotId != 0) {
                snapshot = latched;
                reusingSnapshotLatch = true;
            }
            else {
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
        if (JTRACE_ENABLED(3)) {
            const std::string msg = std::string("path=cuda action=")
                + (reusingSnapshotLatch ? "reuse" : "new")
                + " frame_token=" + std::to_string(snapshot.frameToken.value)
                + " snapshot_id=" + std::to_string(snapshot.snapshotId)
                + " instance_token=" + std::to_string(snapshot.instanceToken.value);
            JTRACE_VERBOSE("MSSNP", msg);
        }

        std::string submissionError;
        if (!JuicerCuda::ResourceManager::begin_submission(submissionTxn, snapshot, submissionError)) {
            mark_context_loss_recovery("begin_submission", cudaErrorUnknown, submissionError);
            JTRACE("CUDA", std::string("FATAL: begin_submission failed: ") + submissionError);
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        if (!JuicerCuda::ResourceManager::acquire_plan(submissionTxn, submissionError)) {
            mark_context_loss_recovery("acquire_plan", cudaErrorUnknown, submissionError);
            JTRACE("CUDA", std::string("FATAL: acquire_plan failed: ") + submissionError);
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
    }

    std::string uploadError;
    if (!cudaResources) {
        JTRACE("CUDA", "FATAL: CUDA resources missing after allocation");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
    if (!JuicerCuda::ResourceManager::command_ensure_uploaded(
            submissionTxn,
            *cudaResources,
            *_ws,
            _pCudaStream,
            uploadError)) {
        mark_context_loss_recovery("command_ensure_uploaded", cudaErrorUnknown, uploadError);
        JTRACE("CUDA", std::string("CUDA WorkingState upload failed: ") + uploadError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
        throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
        throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
    }
    if (JTRACE_ENABLED(3)) {
        std::lock_guard<std::mutex> resLock(cudaResources->m);
        const std::uint64_t build = _ws ? _ws->buildCounter : 0;
        std::string msg = std::string("cuda upload build=") + std::to_string(build)
            + " uploaded=" + std::to_string(cudaResources->uploadedBuildCounter)
            + " printIllumBuild=" + std::to_string(cudaResources->printIllumBuildCounter)
            + " printPreflashBuild=" + std::to_string(cudaResources->printPreflashBuildCounter);
        JTRACE_VERBOSE("PRINTDBG", msg);
    }

    if (_effect.abort()) {
        if (cudaResources) {
            JuicerCuda::record_use(*cudaResources, _pCudaStream);
        }
        return;
    }

#if defined(JUICER_CUDA_VALIDATE_PRIMITIVES) && (JUICER_CUDA_VALIDATE_PRIMITIVES != 0)
    if (JTRACE_ENABLED(3)) {
        std::string validateError;
        if (!cudaResources) {
            JTRACE("CUDA", "FATAL: CUDA resources missing for validation");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        if (!JuicerCuda::validate_density_primitives(*cudaResources, *_ws, _pCudaStream, validateError)) {
            JTRACE("CUDA", std::string("FATAL: CUDA primitive validation failed: ") + validateError);
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }

        if (_ws && _printReady && _prt && !_printParams.bypass) {
            const float kMidSpectral = compute_print_midgray_factor_cached(
                _instanceState,
                *_ws,
                *_prt,
                _printParams,
                _dirRT);
            if (!JuicerCuda::validate_print_primitives(*cudaResources, *_ws, *_prt, _printParams, kMidSpectral, _pCudaStream, validateError)) {
                JTRACE("CUDA", std::string("FATAL: CUDA print validation failed: ") + validateError);
                throw OFX::Exception::Suite(kOfxStatErrFatal);
            }
        }

        JuicerCuda::record_use(*cudaResources, _pCudaStream);
    }
#endif

#if defined(JUICER_CUDA_SELF_CHECK) && (JUICER_CUDA_SELF_CHECK != 0)
    // Runtime CUDA self-check.
    // This is intentionally a host-runtime probe (not JUICER_TESTS), and is designed to be easy
    // to remove later: disable JUICER_CUDA_SELF_CHECK or delete Cuda/JuicerCudaSelfCheck.*.
    static std::once_flag sSelfCheckOnce;
    static bool sSelfCheckOk = true;
    static const char* sSelfCheckErr = nullptr;
    std::call_once(sSelfCheckOnce, [&]() {
        const bool ok = juicer_cuda_runtime_self_check(_pCudaStream, &sSelfCheckErr);
        sSelfCheckOk = ok;
        if (!ok) {
            JTRACE("CUDA", std::string("CUDA self-check failed; forcing CPU fallback. Error: ") + (sSelfCheckErr ? sSelfCheckErr : "(unknown)"));
        } else {
            JTRACE("CUDA", "CUDA self-check passed");
        }
    });
    if (!sSelfCheckOk) {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
#endif

    if (_nComponents == 1) {
        if (srcPtr == dstPtr && srcRowBytes == dstRowBytes) {
            return;
        }
        const cudaStream_t stream = reinterpret_cast<cudaStream_t>(_pCudaStream);
        const cudaError_t err = cudaMemcpy2DAsync(
            dstPtr,
            static_cast<size_t>(dstRowBytes),
            srcPtr,
            static_cast<size_t>(srcRowBytes),
            static_cast<size_t>(widthBytes),
            static_cast<size_t>(height),
            cudaMemcpyDeviceToDevice,
            stream);
        if (err != cudaSuccess) {
            const char* msg = cudaGetErrorString(err);
            mark_context_loss_recovery("copy_alpha_memcpy2d", err, msg ? msg : "");
            JTRACE("CUDA", std::string("FATAL: cudaMemcpy2DAsync failed: ") + (msg ? msg : "(unknown)"));
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        JuicerCuda::record_use(*cudaResources, _pCudaStream);
        return;
    }

    const RenderMode renderMode = _printParams.bypass ? RenderMode::NegativeOnly : RenderMode::Print;

    OfxRectI meterBounds = srcBounds;
    if (_cameraAutoEnabled && _instanceState) {
        std::lock_guard<std::mutex> lock(_instanceState->autoExposureMutex);
        if (_instanceState->autoExposureCanonicalValid) {
            meterBounds = _instanceState->autoExposureCanonicalBounds;
        }
    }
    auto clamp_rect = [](OfxRectI r, const OfxRectI& bounds) {
        r.x1 = std::clamp(r.x1, bounds.x1, bounds.x2);
        r.x2 = std::clamp(r.x2, bounds.x1, bounds.x2);
        r.y1 = std::clamp(r.y1, bounds.y1, bounds.y2);
        r.y2 = std::clamp(r.y2, bounds.y1, bounds.y2);
        if (r.x2 < r.x1) {
            const int tmp = r.x1;
            r.x1 = r.x2;
            r.x2 = tmp;
        }
        if (r.y2 < r.y1) {
            const int tmp = r.y1;
            r.y1 = r.y2;
            r.y2 = tmp;
        }
        return r;
    };
    meterBounds = clamp_rect(meterBounds, srcBounds);
    if ((meterBounds.x2 - meterBounds.x1) <= 0 || (meterBounds.y2 - meterBounds.y1) <= 0) {
        meterBounds = srcBounds;
    }

    auto setup_camera_auto_exposure = [&](
        JuicerCuda::PipelineRunParams& run,
        JuicerCuda::Resources* cudaResources) {
        if (!_cameraAutoEnabled || !cudaResources) {
            return;
        }
        if (_effect.abort()) {
            return;
        }
        if (!(run.nComponents == 3 || run.nComponents == 4)) {
            return;
        }

        const int meterWidth = meterBounds.x2 - meterBounds.x1;
        const int meterHeight = meterBounds.y2 - meterBounds.y1;
        if (meterWidth <= 0 || meterHeight <= 0) {
            return;
        }

        std::string aeError;
        if (!JuicerCuda::ResourceManager::command_ensure_auto_exposure_buffers(
                submissionTxn,
                *cudaResources,
                meterWidth,
                meterHeight,
                autoExposureReusableKeyHash,
                _pCudaStream,
                aeError)) {
            JTRACE("CUDA", std::string("CUDA auto-exposure buffer allocation failed: ") + aeError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
            throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
            throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
        }

        JuicerCudaAutoExposureScratch scratch{};
        scratch.partialsA = cudaResources->autoExposureScratch.partialsA;
        scratch.partialsB = cudaResources->autoExposureScratch.partialsB;
        scratch.partialCapacity = cudaResources->autoExposureScratch.partialCapacity;
        scratch.maxYBits = cudaResources->autoExposureScratch.maxYBits;
        scratch.histogram = cudaResources->autoExposureScratch.histogram;
        scratch.weightsX = cudaResources->autoExposureScratch.weightsX;
        scratch.weightsY = cudaResources->autoExposureScratch.weightsY;

        JuicerCudaAutoExposureDeviceState state{};
        state.exposureScale = cudaResources->autoExposureExposureScale;
        state.autoEV = cudaResources->autoExposureAutoEV;
        state.valid = cudaResources->autoExposureValid;

        std::uint64_t meterStateKey = Hash::kFnvOffset;
        const double timeFrames = std::isfinite(_timeFrames) ? _timeFrames : 0.0;
        Hash::hash_bytes_update(meterStateKey, &timeFrames, sizeof(timeFrames));
        Hash::hash_bytes_update(meterStateKey, &_clipToken, sizeof(_clipToken));
        Hash::hash_bytes_update(meterStateKey, &meterBounds, sizeof(meterBounds));
        Hash::hash_bytes_update(meterStateKey, &srcBounds, sizeof(srcBounds));
        Hash::hash_bytes_update(meterStateKey, &srcRowBytes, sizeof(srcRowBytes));
        Hash::hash_bytes_update(meterStateKey, &run.nComponents, sizeof(run.nComponents));
        Hash::hash_bytes_update(meterStateKey, &run.filmRaw.inputColorSpaceIndex, sizeof(run.filmRaw.inputColorSpaceIndex));
        Hash::hash_bytes_update(meterStateKey, &run.filmRaw.applyCctfDecoding, sizeof(run.filmRaw.applyCctfDecoding));
        Hash::hash_bytes_update(meterStateKey, &run.filmRaw.inputRGBToXYZ, sizeof(run.filmRaw.inputRGBToXYZ));
        Hash::hash_bytes_update(meterStateKey, &_cameraMeteringMethod, sizeof(_cameraMeteringMethod));
        if (meterStateKey == 0) {
            meterStateKey = 1;
        }

        auto slider_equal = [](double a, double b) -> bool {
            if (!(std::isfinite(a) && std::isfinite(b))) {
                return false;
            }
            return std::abs(a - b) <= 1e-12;
        };

        const bool needMeter = (cudaResources->autoExposureKeyHash != meterStateKey);
        const bool needSliderUpdate = !slider_equal(cudaResources->autoExposureSliderEV, _cameraSliderEV);
        const char* errMsg = nullptr;
        if (needMeter) {
            if (_cameraMeteringMethod == 0) {
                if (!scratch.weightsX || !scratch.weightsY ||
                    cudaResources->autoExposureScratch.weightsWidth != meterWidth ||
                    cudaResources->autoExposureScratch.weightsHeight != meterHeight) {
                    const int rcW = juicer_cuda_auto_exposure_build_center_weight_tables(
                        meterWidth,
                        meterHeight,
                        scratch.weightsX,
                        scratch.weightsY,
                        _pCudaStream,
                        &errMsg);
                    if (rcW != 0) {
                        JTRACE("CUDA", std::string("CUDA auto-exposure weight build failed: ") + (errMsg ? errMsg : "(unknown)"));
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                        throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                        throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                    }
                    cudaResources->autoExposureScratch.weightsWidth = meterWidth;
                    cudaResources->autoExposureScratch.weightsHeight = meterHeight;
                }
            }

            const int rc = juicer_cuda_auto_exposure_meter_to_device(
                srcBase,
                static_cast<std::size_t>(srcRowBytes),
                srcBounds.x1,
                srcBounds.y1,
                srcBounds.x2,
                srcBounds.y2,
                meterBounds.x1,
                meterBounds.y1,
                meterBounds.x2,
                meterBounds.y2,
                run.nComponents,
                run.filmRaw.inputColorSpaceIndex,
                run.filmRaw.applyCctfDecoding,
                run.filmRaw.inputRGBToXYZ,
                _cameraMeteringMethod,
                _cameraSliderEV,
                scratch,
                state,
                _pCudaStream,
                &errMsg);
            if (rc != 0) {
                JTRACE("CUDA", std::string("CUDA auto-exposure metering failed: ") + (errMsg ? errMsg : "(unknown)"));
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
            }
            cudaResources->autoExposureKeyHash = meterStateKey;
            cudaResources->autoExposureSliderEV = _cameraSliderEV;
        }
        else if (needSliderUpdate) {
            const int rc = juicer_cuda_auto_exposure_update_scale_to_device(
                _cameraSliderEV,
                state,
                _pCudaStream,
                &errMsg);
            if (rc != 0) {
                JTRACE("CUDA", std::string("CUDA auto-exposure slider update failed: ") + (errMsg ? errMsg : "(unknown)"));
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
            }
            cudaResources->autoExposureSliderEV = _cameraSliderEV;
        }

        run.filmExpose.exposureScaleDevice = cudaResources->autoExposureExposureScale;
        run.filmExpose.exposureScale = 1.0f;
    };

    struct GrainSetupResult {
        bool wantGrain = false;
        bool wantGrainSublayers = false;
        bool wantGrainBlur = false;
        bool wantGrainMix = false;
        float grainBlurSigmaPx = 0.0f;
        float grainBlurSigmaMidPx = 0.0f;
        float grainBlurSigmaCoarsePx = 0.0f;
        float grainDyeSigmaPx[3][3] = { {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f} };
    };

    auto setup_grain_payload = [&](JuicerCuda::PipelineRunParams& run,
                                   const Profiles::GrainMetadata& grainUi,
                                   bool includeDefects) -> GrainSetupResult {
        GrainSetupResult result{};

        auto nanmax_vector = [](const std::vector<float>& values, float& outMax) -> bool {
            double m = -std::numeric_limits<double>::infinity();
            bool found = false;
            for (float v : values) {
                if (std::isfinite(v)) {
                    m = std::max(m, static_cast<double>(v));
                    found = true;
                }
            }
            if (!found || !std::isfinite(m)) {
                return false;
            }
            outMax = static_cast<float>(m);
            return std::isfinite(outMax);
        };
        auto nanmax_curve = [&](const Spectral::Curve& curve, float& outMax) -> bool {
            return nanmax_vector(curve.linear, outMax);
        };

        bool wantGrain = grainUi.active && std::isfinite(_pixelSizeUm) && _pixelSizeUm > 0.0f;
        bool wantGrainSublayers = false;
        bool wantGrainBlur = false;
        bool wantGrainMix = false;
        float grainBlurSigmaPx = 0.0f;
        float grainBlurSigmaMidPx = 0.0f;
        float grainBlurSigmaCoarsePx = 0.0f;
        float grainDyeSigmaPx[3][3] = { {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f} };

        run.grain = JuicerCuda::GrainPayload{};
        run.grainKernels = JuicerCuda::GrainKernelPayload{};
        {
            const std::uint64_t sessionSeed = safe_session_seed(_instanceState);
            const double fps = (std::isfinite(_frameRate) && _frameRate > 0.0) ? _frameRate : 24.0;
            const double timeFrames = std::isfinite(_timeFrames) ? _timeFrames : static_cast<double>(_frameIndex);
            const double alphaFrames = std::isfinite(timeFrames)
                ? (timeFrames - static_cast<double>(_frameIndex))
                : 0.0;
            const float timeAlpha = static_cast<float>(std::clamp(alphaFrames, 0.0, 1.0));
            const double timeSeconds = (fps > 0.0) ? (timeFrames / fps) : 0.0;
            const double weaveAmount = std::isfinite(_gateWeaveAmount)
                ? std::clamp(_gateWeaveAmount, 0.0, 10.0)
                : 0.0;
            const GateWeaveSignal weave = compute_gate_weave(
                sessionSeed,
                timeSeconds,
                6.0,
                0.005,
                static_cast<double>(_pixelSizeUm),
                weaveAmount);
            const double debugScalePx = (std::isfinite(_pixelSizeUm) && _pixelSizeUm > 0.0)
                ? (4.0 * 6.0 * weaveAmount / static_cast<double>(_pixelSizeUm))
                : 1.0;
            const int breathingPeriodFrames = std::max(1, static_cast<int>(std::llround(fps * 2.5)));
            const double clumpPeriodSec = std::isfinite(grainUi.clumpMorphPeriodSec)
                ? std::clamp(static_cast<double>(grainUi.clumpMorphPeriodSec), 5.0, 60.0)
                : 25.0;
            const double clumpFps = (fps > 0.0) ? fps : 24.0;
            const int clumpMorphPeriodFrames = std::max(1, static_cast<int>(std::llround(clumpFps * clumpPeriodSec)));
            const double longEdgePx = static_cast<double>(std::max(width, height));
            const double filmFormatMm = (std::isfinite(_pixelSizeUm) && _pixelSizeUm > 0.0f && longEdgePx > 0.0)
                ? (static_cast<double>(_pixelSizeUm) * longEdgePx / 1000.0)
                : 0.0;
            const double pitchMm = (std::isfinite(filmFormatMm) && filmFormatMm > 0.0)
                ? (filmFormatMm * static_cast<double>(height) / longEdgePx)
                : 0.0;
            const int pitchPx = (std::isfinite(_pixelSizeUm) && _pixelSizeUm > 0.0f && pitchMm > 0.0)
                ? static_cast<int>(std::llround(pitchMm * 1000.0 / static_cast<double>(_pixelSizeUm)))
                : height;
            const double filmScale = (std::isfinite(filmFormatMm) && filmFormatMm > 0.0) ? (filmFormatMm / 10.0) : 1.0;
            run.grain.seedBase = make_seed_base(_clipToken, _frameIndex, sessionSeed, kSeedPassGrain);
            run.grain.seedBaseNext = make_seed_base(_clipToken, _frameIndex + 1, sessionSeed, kSeedPassGrain);
            run.grain.frameIndex = _frameIndex;
            run.grain.stbnSessionSeed = sessionSeed;
            run.grain.clipToken = static_cast<std::uint64_t>(_clipToken);
            run.grain.timeAlpha = timeAlpha;
            run.gateWeave.active = (weaveAmount > 0.0 && std::isfinite(_pixelSizeUm) && _pixelSizeUm > 0.0f) ? 1 : 0;
            run.gateWeave.dxPx = weave.dxPx;
            run.gateWeave.dyPx = weave.dyPx;
            run.gateWeave.cosRot = weave.cosRot;
            run.gateWeave.sinRot = weave.sinRot;
            run.gateWeave.debugScalePx = static_cast<float>((debugScalePx > 1e-6) ? debugScalePx : 1.0);
            run.grain.pitchPx = pitchPx;
            run.grain.breathingPeriodFrames = breathingPeriodFrames;
            run.grain.breathingAmplitude = 0.01902219f;
            run.grain.breathingCellUmSmall = static_cast<float>(2500.0 * filmScale);
            run.grain.breathingCellUmLarge = static_cast<float>(5000.0 * filmScale);
            run.grain.breathingMix = 0.30f;
            run.grain.breathingDriftUmPerFrame = 1.0f;
            run.grain.sizeMixWeight = 0.30f;
            run.grain.sizeMixScale = 3.0f;
            run.grain.clumpTemporalMix = std::clamp(grainUi.clumpTemporalMix, 0.0f, 0.30f);
            run.grain.clumpMorphPeriodFrames = clumpMorphPeriodFrames;
            run.grain.wangCellMm = 2.0f;
            if (cudaResources && cudaResources->stbnData &&
                cudaResources->stbnWidth > 0 && cudaResources->stbnHeight > 0 && cudaResources->stbnFrames > 0) {
                run.grain.stbn = cudaResources->stbnData;
                run.grain.stbnWidth = cudaResources->stbnWidth;
                run.grain.stbnHeight = cudaResources->stbnHeight;
                run.grain.stbnFrames = cudaResources->stbnFrames;
                run.grain.stbnOffsetX = stbn_offset(sessionSeed, run.grain.stbnWidth, 0xA5u);
                run.grain.stbnOffsetY = stbn_offset(sessionSeed, run.grain.stbnHeight, 0x5Au);
                run.grain.stbnFrame = stbn_frame_index(_frameIndex, run.grain.stbnFrames, sessionSeed);
            }
            if (cudaResources && cudaResources->wangTilesData && cudaResources->wangLutData &&
                cudaResources->wangWidth > 0 && cudaResources->wangHeight > 0 &&
                cudaResources->wangCount > 0 && cudaResources->wangColors > 0) {
                run.grain.wangTiles = cudaResources->wangTilesData;
                run.grain.wangLut = cudaResources->wangLutData;
                run.grain.wangWidth = cudaResources->wangWidth;
                run.grain.wangHeight = cudaResources->wangHeight;
                run.grain.wangCount = cudaResources->wangCount;
                run.grain.wangColors = cudaResources->wangColors;
            }
        }
        if (wantGrain) {
            float densityMin[3] = {
                grainUi.densityMin[0],
                grainUi.densityMin[1],
                grainUi.densityMin[2]
            };
            float uniformity[3] = {
                grainUi.uniformity[0],
                grainUi.uniformity[1],
                grainUi.uniformity[2]
            };
            for (int i = 0; i < 3; ++i) {
                if (!std::isfinite(densityMin[i]) || densityMin[i] < 0.0f) {
                    densityMin[i] = 0.0f;
                }
                if (!std::isfinite(uniformity[i])) {
                    uniformity[i] = 0.0f;
                }
                uniformity[i] = std::clamp(uniformity[i], 0.0f, 1.0f);
            }

            float maxC = 0.0f;
            float maxM = 0.0f;
            float maxY = 0.0f;
            const bool maxOk =
                nanmax_curve(_ws->densR, maxC) &&
                nanmax_curve(_ws->densG, maxM) &&
                nanmax_curve(_ws->densB, maxY);
            if (!maxOk) {
                wantGrain = false;
            }

            const float pixelAreaUm2 = _pixelSizeUm * _pixelSizeUm;
            if (!std::isfinite(pixelAreaUm2) || !(pixelAreaUm2 > 0.0f)) {
                wantGrain = false;
            }

            const int nSubLayers = (grainUi.nSubLayers > 0) ? grainUi.nSubLayers : 1;
            run.grain.nSubLayers = nSubLayers;
            run.grain.originX = win.x1;
            run.grain.originY = win.y1;
            run.grain.pixelSizeUm = static_cast<float>(_pixelSizeUm);
            run.grain.blurSigmaPx = std::isfinite(grainUi.blur) ? std::max(0.0f, grainUi.blur) : 0.0f;
            run.grain.blurDyeCloudsUm = std::isfinite(grainUi.blurDyeCloudsUm) ? std::max(0.0f, grainUi.blurDyeCloudsUm) : 0.0f;
            run.grain.sizeMixWeight = (std::isfinite(grainUi.sizeMixWeight)) ? std::clamp(grainUi.sizeMixWeight, 0.0f, 1.0f) : 0.0f;
            run.grain.sizeMixWeightMid = (std::isfinite(grainUi.sizeMixWeightMid)) ? std::clamp(grainUi.sizeMixWeightMid, 0.0f, 1.0f) : 0.0f;
            run.grain.sizeMixScale = (std::isfinite(grainUi.sizeMixScale)) ? std::max(1.0f, grainUi.sizeMixScale) : 1.0f;
            run.grain.breathingDebug = grainUi.breathingDebug ? 1 : 0;
            run.grain.debugView = std::clamp(grainUi.debugView, 0, 6);
            run.grain.amplitude = std::isfinite(grainUi.amplitude) ? std::max(0.0f, grainUi.amplitude) : 1.0f;
            const float chromaMix = (std::isfinite(grainUi.chroma))
                ? std::clamp(grainUi.chroma, 0.0f, 1.0f)
                : 1.0f;
            run.grain.chromaMix = chromaMix;
            run.grain.chromaSharedWeight = std::sqrt(std::max(0.0f, 1.0f - chromaMix));
            run.grain.chromaIndWeight = std::sqrt(std::max(0.0f, chromaMix));
            run.grain.microStructure[0] = grainUi.microStructure[0];
            run.grain.microStructure[1] = grainUi.microStructure[1];
            if (includeDefects) {
                run.grain.filmDustAmount = std::isfinite(grainUi.filmDustAmount)
                    ? std::clamp(grainUi.filmDustAmount, 0.0f, 10.0f)
                    : 0.0f;
                run.grain.gateDustAmount = std::isfinite(grainUi.gateDustAmount)
                    ? std::clamp(grainUi.gateDustAmount, 0.0f, 10.0f)
                    : 0.0f;
                run.grain.filmScratchAmount = std::isfinite(grainUi.filmScratchAmount)
                    ? std::clamp(grainUi.filmScratchAmount, 0.0f, 10.0f)
                    : 0.0f;
                run.grain.gateScratchAmount = std::isfinite(grainUi.gateScratchAmount)
                    ? std::clamp(grainUi.gateScratchAmount, 0.0f, 10.0f)
                    : 0.0f;
            }
            for (int i = 0; i < 3; ++i) {
                run.grain.densityMin[i] = densityMin[i];
                run.grain.uniformity[i] = uniformity[i];
            }

            bool paramsOk = wantGrain;
            float blurAreaRatio = 1.0f;
            float blurRatioSum = 0.0f;
            int blurRatioCount = 0;
            if (paramsOk) {
                constexpr float kDefaultParticleAreaUm2 = 0.335f;
                constexpr float kDefaultParticleScale[3] = { 1.10f, 1.27f, 2.08f };
                const float densityMaxCurves[3] = { maxC, maxM, maxY };
                for (int i = 0; i < 3; ++i) {
                    const float densityMax = densityMaxCurves[i] + densityMin[i];
                    const float particleArea = grainUi.agxParticleAreaUm2 * grainUi.agxParticleScale[i];
                    if (!std::isfinite(particleArea) || !(particleArea > 0.0f)) {
                        paramsOk = false;
                        break;
                    }
                    const float particleAreaRef = kDefaultParticleAreaUm2 * kDefaultParticleScale[i];
                    if (std::isfinite(particleAreaRef) && particleAreaRef > 0.0f) {
                        blurRatioSum += particleArea / particleAreaRef;
                        blurRatioCount += 1;
                    }
                    float nParticles = pixelAreaUm2 / particleArea;
                    if (nSubLayers > 1) {
                        nParticles /= static_cast<float>(nSubLayers);
                    }
                    if (!std::isfinite(nParticles) || !(nParticles > 0.0f)) {
                        paramsOk = false;
                        break;
                    }
                    const float odParticle = densityMax / nParticles;
                    run.grain.densityMax[i] = densityMax;
                    run.grain.nParticles[i] = nParticles;
                    run.grain.odParticle[i] = std::isfinite(odParticle) ? odParticle : 0.0f;
                }
            }
            if (!paramsOk) {
                wantGrain = false;
            }

            if (wantGrain) {
                if (blurRatioCount > 0) {
                    blurAreaRatio = blurRatioSum / static_cast<float>(blurRatioCount);
                }
                grainBlurSigmaPx = run.grain.blurSigmaPx;
                if (std::isfinite(grainBlurSigmaPx)) {
                    grainBlurSigmaPx *= std::sqrt(std::max(blurAreaRatio, 0.0f));
                }

                if (!std::isfinite(run.grain.microStructure[1]) || !(run.grain.microStructure[1] > 0.0f)) {
                    run.grain.microStructure[0] = 0.0f;
                    run.grain.microStructure[1] = 0.0f;
                }

                if (grainUi.sublayersActive && _ws->hasDensityCurvesLayers && cudaResources->hasDensityCurvesLayers) {
                    float densityMaxLayers[3][3] = { {0.0f, 0.0f, 0.0f},
                                                     {0.0f, 0.0f, 0.0f},
                                                     {0.0f, 0.0f, 0.0f} };
                    bool layersOk = true;
                    for (int layer = 0; layer < 3; ++layer) {
                        for (int ch = 0; ch < 3; ++ch) {
                            if (!nanmax_vector(_ws->densityCurvesLayers[layer][ch], densityMaxLayers[layer][ch])) {
                                layersOk = false;
                            }
                        }
                    }

                    if (layersOk) {
                        for (int ch = 0; ch < 3; ++ch) {
                            float total = 0.0f;
                            for (int layer = 0; layer < 3; ++layer) {
                                total += densityMaxLayers[layer][ch];
                            }
                            if (!(std::isfinite(total) && total > 0.0f)) {
                                layersOk = false;
                                break;
                            }

                            for (int layer = 0; layer < 3; ++layer) {
                                const float fraction = densityMaxLayers[layer][ch] / total;
                                const float minLayer = fraction * densityMin[ch];
                                const float maxLayer = densityMaxLayers[layer][ch] + minLayer;
                                const float particleAreaLayer = grainUi.agxParticleAreaUm2 * grainUi.agxParticleScale[ch] * grainUi.agxParticleScaleLayers[layer];
                                if (!std::isfinite(particleAreaLayer) || !(particleAreaLayer > 0.0f)) {
                                    layersOk = false;
                                    break;
                                }
                                const float nParticlesLayer = pixelAreaUm2 * fraction / particleAreaLayer;
                                const float odParticle = (nParticlesLayer > 0.0f) ? (maxLayer / nParticlesLayer) : 0.0f;
                                run.grain.densityMinLayers[layer][ch] = minLayer;
                                run.grain.densityMaxLayers[layer][ch] = maxLayer;
                                run.grain.nParticlesLayers[layer][ch] = std::isfinite(nParticlesLayer) ? nParticlesLayer : 0.0f;
                                run.grain.odParticleLayers[layer][ch] = std::isfinite(odParticle) ? odParticle : 0.0f;
                                run.grain.densityCurvesLayers[layer][ch] = cudaResources->densityCurvesLayers[layer][ch];
                                const float dyeSigma = run.grain.blurDyeCloudsUm * std::sqrt(std::max(0.0f, run.grain.odParticleLayers[layer][ch]));
                                grainDyeSigmaPx[layer][ch] = std::isfinite(dyeSigma) ? dyeSigma : 0.0f;
                            }
                            if (!layersOk) {
                                break;
                            }
                        }
                    }
                    wantGrainSublayers = layersOk;
                }
            }
            if (std::isfinite(grainBlurSigmaPx)) {
                wantGrainBlur = wantGrainSublayers ? (grainBlurSigmaPx > 0.0f) : (grainBlurSigmaPx > 0.4f);
            }
        }
        run.grain.active = wantGrain ? 1 : 0;
        run.grain.sublayersActive = wantGrainSublayers ? 1 : 0;

        // Debug view scaling: stable linear mapping for signed delta fields.
        {
            float densityMaxAvg = (run.grain.densityMax[0] + run.grain.densityMax[1] + run.grain.densityMax[2]) * (1.0f / 3.0f);
            if (!std::isfinite(densityMaxAvg) || densityMaxAvg <= 0.0f) {
                densityMaxAvg = 1.0f;
            }
            run.grain.debugScale = 0.25f / std::max(1e-6f, densityMaxAvg);
        }

        // Phase 3: three-scale mix configuration (fine + mid + coarse).
        {
            float wC = std::isfinite(run.grain.sizeMixWeight) ? std::clamp(run.grain.sizeMixWeight, 0.0f, 1.0f) : 0.0f;
            float wM = std::isfinite(run.grain.sizeMixWeightMid) ? std::clamp(run.grain.sizeMixWeightMid, 0.0f, 1.0f) : 0.0f;
            float wF = 1.0f - wM - wC;
            if (wF < 0.0f) {
                wF = 0.0f;
            }
            float wSum = wF + wM + wC;
            if (wSum > 0.0f) {
                const float invSum = 1.0f / wSum;
                wF *= invSum;
                wM *= invSum;
                wC *= invSum;
            }
            else {
                wF = 1.0f;
                wM = 0.0f;
                wC = 0.0f;
            }
            run.grain.sizeMixWeight = wC;
            run.grain.sizeMixWeightMid = wM;

            const float scale = std::isfinite(run.grain.sizeMixScale) ? std::max(1.0f, run.grain.sizeMixScale) : 1.0f;
            const bool canMix = wantGrain && wantGrainBlur && (grainBlurSigmaPx > 0.0f) && (scale > 1.0f);
            wantGrainMix = canMix && ((wM > 0.0f) || (wC > 0.0f));

            if (wantGrainMix) {
                const float sigmaF = grainBlurSigmaPx;
                const float sigmaCRaw = sigmaF * std::sqrt(scale);
                grainBlurSigmaCoarsePx = std::max(sigmaF, std::min(sigmaCRaw, sigmaF * 4.0f));
                if (!(std::isfinite(grainBlurSigmaCoarsePx) && grainBlurSigmaCoarsePx > 0.0f)) {
                    wantGrainMix = false;
                    grainBlurSigmaCoarsePx = 0.0f;
                }
                if (wantGrainMix) {
                    grainBlurSigmaMidPx = std::sqrt(std::max(0.0f, sigmaF * grainBlurSigmaCoarsePx));
                    if (!(std::isfinite(grainBlurSigmaMidPx) && grainBlurSigmaMidPx > 0.0f)) {
                        wantGrainMix = false;
                        grainBlurSigmaMidPx = 0.0f;
                    }
                }
            }
            if (!wantGrainMix) {
                run.grain.sizeMixGain = 1.0f;
                run.grain.sizeMixWeight = 0.0f;
                run.grain.sizeMixWeightMid = 0.0f;
                grainBlurSigmaMidPx = 0.0f;
                grainBlurSigmaCoarsePx = 0.0f;
            }
            else {
                auto kernel_energy_2d = [](float sigma) -> float {
                    if (!(std::isfinite(sigma) && sigma > 0.0f)) {
                        return 1.0f;
                    }
                    const int radiusRaw = JuicerGaussian::scipy_gaussian_radius(sigma, 4.0f);
                    const int radius = std::min(radiusRaw, 75);
                    if (radius <= 0) {
                        return 1.0f;
                    }
                    const double s2 = static_cast<double>(sigma) * static_cast<double>(sigma) * 2.0;
                    double wsum = 0.0;
                    std::vector<double> w;
                    w.resize(static_cast<size_t>(2 * radius + 1));
                    for (int i = -radius; i <= radius; ++i) {
                        const double wi = std::exp(-(static_cast<double>(i * i)) / s2);
                        w[static_cast<size_t>(i + radius)] = wi;
                        wsum += wi;
                    }
                    const double invW = (wsum != 0.0) ? (1.0 / wsum) : 0.0;
                    double sumSq = 0.0;
                    for (double wi : w) {
                        const double wn = wi * invW;
                        sumSq += wn * wn;
                    }
                    const double e1 = std::max(0.0, sumSq);
                    const double e2 = e1 * e1;
                    return static_cast<float>(std::max(1e-12, e2));
                };

                const float sigmaF = grainBlurSigmaPx;
                const float sigmaM = grainBlurSigmaMidPx;
                const float sigmaC = grainBlurSigmaCoarsePx;
                const float eF = kernel_energy_2d(sigmaF);
                const float eM = kernel_energy_2d(sigmaM);
                const float eC = kernel_energy_2d(sigmaC);
                const float midScale = std::sqrt(scale);
                const float rM = midScale * (eM / std::max(1e-12f, eF));
                const float rC = scale * (eC / std::max(1e-12f, eF));
                const float denom = wF * wF + wM * wM * rM + wC * wC * rC;
                run.grain.sizeMixGain = (denom > 1e-12f) ? (1.0f / std::sqrt(denom)) : 1.0f;
            }
        }

        result.wantGrain = wantGrain;
        result.wantGrainSublayers = wantGrainSublayers;
        result.wantGrainBlur = wantGrainBlur;
        result.wantGrainMix = wantGrainMix;
        result.grainBlurSigmaPx = grainBlurSigmaPx;
        result.grainBlurSigmaMidPx = grainBlurSigmaMidPx;
        result.grainBlurSigmaCoarsePx = grainBlurSigmaCoarsePx;
        for (int layer = 0; layer < 3; ++layer) {
            for (int ch = 0; ch < 3; ++ch) {
                result.grainDyeSigmaPx[layer][ch] = grainDyeSigmaPx[layer][ch];
            }
        }

        return result;
    };

    auto launch_base_pipeline_graph = [&](int renderModeKey,
                                          JuicerCuda::PipelineRunParams& run) -> cudaError_t {
        std::string graphError;
        int graphErrCode = static_cast<int>(cudaErrorUnknown);
        if (!JuicerCuda::ResourceManager::command_launch_base_pipeline_graph(
                submissionTxn,
                run,
                renderModeKey,
                _pCudaStream,
                graphErrCode,
                graphError)) {
            mark_context_loss_recovery("command_launch_base_pipeline_graph", cudaErrorUnknown, graphError);
            JTRACE("CUDA", std::string("CUDA base graph launch command failed: ") + graphError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
            throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
            throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
        }
        return static_cast<cudaError_t>(graphErrCode);
    };

    auto prepare_scan_error_stage = [&](JuicerCuda::Resources* resources,
                                        JuicerCuda::PipelineRunParams& run,
                                        cudaStream_t stream) -> cudaEvent_t {
        std::string scanFlagError;
        if (!JuicerCuda::ResourceManager::command_ensure_scan_error_flag(
                submissionTxn,
                *resources,
                _pCudaStream,
                scanFlagError)) {
            mark_context_loss_recovery("command_ensure_scan_error_flag", cudaErrorUnknown, scanFlagError);
            JTRACE("CUDA", std::string("CUDA scan error flag allocation failed: ") + scanFlagError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
            throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
            throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
        }
        run.scanStage.scanErrorFlag = resources->scanErrorFlag;
        if (!run.scanStage.scanErrorFlag) {
            JTRACE("CUDA", "FATAL: scan error flag missing after allocation");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }

        cudaEvent_t scanEvent = resources->scanErrorEventOpaque
            ? reinterpret_cast<cudaEvent_t>(resources->scanErrorEventOpaque)
            : nullptr;
        if (resources->scanErrorPending && scanEvent && resources->scanErrorHost) {
            cudaError_t pollErr = cudaEventQuery(scanEvent);
            if (pollErr == cudaSuccess) {
                resources->scanErrorPending = 0;
                if (*resources->scanErrorHost != 0) {
                    JTRACE("CUDA", "FATAL: previous scan produced non-finite RGB");
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
                }
            }
            else if (pollErr == cudaErrorNotReady) {
                // Do not block the CPU in steady-state: order this stream after the pending readback
                // and reuse the staging/event on this submission.
                const cudaError_t waitErr = cudaStreamWaitEvent(stream, scanEvent, 0);
                if (waitErr != cudaSuccess) {
                    const char* msg = cudaGetErrorString(waitErr);
                    mark_context_loss_recovery("scan_error_stream_wait", waitErr, msg ? msg : "");
                    JTRACE("CUDA", std::string("CUDA scan error stream wait failed: ") + (msg ? msg : "(unknown)"));
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
                }
                resources->scanErrorPending = 0;
            }
            else {
                const char* msg = cudaGetErrorString(pollErr);
                mark_context_loss_recovery("scan_error_event_query", pollErr, msg ? msg : "");
                JTRACE("CUDA", std::string("CUDA scan error event query failed: ") + (msg ? msg : "(unknown)"));
                throw OFX::Exception::Suite(kOfxStatErrFatal);
            }
        }

        cudaError_t flagErr = cudaMemsetAsync(run.scanStage.scanErrorFlag, 0, sizeof(int), stream);
        if (flagErr != cudaSuccess) {
            const char* msg = cudaGetErrorString(flagErr);
            mark_context_loss_recovery("scan_error_flag_memset", flagErr, msg ? msg : "");
            JTRACE("CUDA", std::string("CUDA scan error flag memset failed: ") + (msg ? msg : "(unknown)"));
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
            throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
            throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
        }

        return scanEvent;
    };

    auto finalize_scan_error_stage = [&](JuicerCuda::Resources* resources,
                                         const JuicerCuda::PipelineRunParams& run,
                                         cudaStream_t stream,
                                         cudaEvent_t scanEvent,
                                         const char* stageLabel) {
        const std::string stage = (stageLabel && *stageLabel) ? stageLabel : "pipeline";
        cudaError_t flagErr = cudaSuccess;
        if (resources->scanErrorHost && scanEvent) {
            flagErr = cudaMemcpyAsync(resources->scanErrorHost, run.scanStage.scanErrorFlag, sizeof(int), cudaMemcpyDeviceToHost, stream);
            if (flagErr != cudaSuccess) {
                const char* msg = cudaGetErrorString(flagErr);
                mark_context_loss_recovery("scan_error_flag_readback", flagErr, msg ? msg : "");
                JTRACE("CUDA", std::string("CUDA scan error flag readback failed: ") + (msg ? msg : "(unknown)"));
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
            }
            cudaError_t evErr = cudaEventRecord(scanEvent, stream);
            if (evErr != cudaSuccess) {
                const char* msg = cudaGetErrorString(evErr);
                mark_context_loss_recovery("scan_error_event_record", evErr, msg ? msg : "");
                JTRACE("CUDA", std::string("CUDA scan error event record failed: ") + (msg ? msg : "(unknown)"));
                throw OFX::Exception::Suite(kOfxStatErrFatal);
            }
            resources->scanErrorPending = 1;
            cudaError_t pollErr = cudaEventQuery(scanEvent);
            if (pollErr == cudaSuccess) {
                resources->scanErrorPending = 0;
                if (*resources->scanErrorHost != 0) {
                    JTRACE("CUDA", std::string("FATAL: ") + stage + " pipeline scan produced non-finite RGB");
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
                }
            }
            else if (pollErr != cudaErrorNotReady) {
                const char* msg = cudaGetErrorString(pollErr);
                mark_context_loss_recovery("scan_error_event_query", pollErr, msg ? msg : "");
                JTRACE("CUDA", std::string("CUDA scan error event query failed: ") + (msg ? msg : "(unknown)"));
                throw OFX::Exception::Suite(kOfxStatErrFatal);
            }
        }
        else {
            static std::atomic<bool> sScanErrorReadbackUnavailableWarned{ false };
            if (!sScanErrorReadbackUnavailableWarned.exchange(true)) {
                JTRACE("CUDA", "scan error host/event staging unavailable; skipping asynchronous scan-error readback validation");
            }
        }
    };

    auto commit_submission_or_throw = [&]() {
        std::string commitError;
        if (!JuicerCuda::ResourceManager::commit_submission(submissionTxn, _pCudaStream, commitError)) {
            mark_context_loss_recovery("commit_submission", cudaErrorUnknown, commitError);
            JTRACE("CUDA", std::string("FATAL: commit_submission failed: ") + commitError);
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        submissionTxnScope.committed = true;
    };

    auto is_scratch_contention_exhausted = [](const std::string& error) -> bool {
        return JuicerCuda::ResourceManager::error_is_scratch_exhausted(error);
    };

    auto setup_scan_stage_resources = [&](JuicerCuda::Resources* resources,
                                          JuicerCuda::PipelineRunParams& run,
                                          cudaStream_t stream,
                                          bool negativeMedium) -> cudaEvent_t {
        run.scanStage.scannerUseLut = _scannerSettings.useLut ? 1 : 0;
        run.scanStage.scanLutLog2XYZ = nullptr;
        run.scanStage.scanLutRes = 0;
        if (run.scanStage.scannerUseLut) {
            std::string lutError;
            if (!JuicerCuda::ResourceManager::command_ensure_scan_lut(
                    submissionTxn,
                    *resources,
                    *_ws,
                    negativeMedium,
                    _pCudaStream,
                    lutError)) {
                mark_context_loss_recovery(
                    negativeMedium ? "command_ensure_scan_lut_negative" : "command_ensure_scan_lut_print",
                    cudaErrorUnknown,
                    lutError);
                JTRACE("CUDA", std::string("CUDA ") + (negativeMedium ? "scan" : "print scan")
                    + " LUT upload failed: " + lutError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
            }
            const JuicerCuda::Resources::DeviceSpectralLut& scanLut =
                negativeMedium ? resources->scanNegativeLut : resources->scanPrintLut;
            run.scanStage.scanLutLog2XYZ = scanLut.log2XYZ;
            run.scanStage.scanLutRes = static_cast<int>(scanLut.res);
            if (!run.scanStage.scanLutLog2XYZ || run.scanStage.scanLutRes <= 0) {
                JTRACE("CUDA", std::string("FATAL: ") + (negativeMedium ? "scan" : "print scan")
                    + " LUT missing after successful upload");
                throw OFX::Exception::Suite(kOfxStatErrFatal);
            }
        }

        const JuicerCuda::Resources::DeviceScanMedium& scanMedium =
            negativeMedium ? resources->scanNegative : resources->scanPrint;
        run.scanStage.scanTables.epsC = scanMedium.tables.epsC;
        run.scanStage.scanTables.epsM = scanMedium.tables.epsM;
        run.scanStage.scanTables.epsY = scanMedium.tables.epsY;
        run.scanStage.scanTables.Ax = scanMedium.tables.Ax;
        run.scanStage.scanTables.Ay = scanMedium.tables.Ay;
        run.scanStage.scanTables.Az = scanMedium.tables.Az;
        run.scanStage.scanTables.baseMin = scanMedium.tables.baseMin;
        run.scanStage.scanTables.K = scanMedium.tables.K;
        run.scanStage.scanTables.hasBaseline = scanMedium.tables.hasBaseline;
        run.scanStage.scanTables.invYn = scanMedium.tables.invYn;
        run.scanStage.scanTables.mediumIsNegative = scanMedium.mediumIsNegative;
        for (int i = 0; i < 3; ++i) {
            run.scanStage.scanTables.min_cmy[i] = scanMedium.min_cmy[i];
            run.scanStage.scanTables.inv_max_cmy[i] = scanMedium.inv_max_cmy[i];
        }

        return prepare_scan_error_stage(resources, run, stream);
    };

    auto setup_spatial_dir_stage = [&](JuicerCuda::Resources* resources,
                                       JuicerCuda::PipelineRunParams& run,
                                       int frameWidth,
                                       int frameHeight,
                                       bool useSpatialDir) -> bool {
        run.filmDevelop.spatialDir.active = useSpatialDir ? 1 : 0;
        run.filmDevelop.spatialDir.corrY = nullptr;
        run.filmDevelop.spatialDir.corrM = nullptr;
        run.filmDevelop.spatialDir.corrC = nullptr;
        if (!useSpatialDir) {
            return true;
        }
        if (_effect.abort()) {
            JuicerCuda::record_use(*resources, _pCudaStream);
            return false;
        }

        std::string dirError;
        if (!JuicerCuda::ResourceManager::command_ensure_spatial_dir_scratch(
                submissionTxn,
                *resources,
                frameWidth,
                frameHeight,
                _pCudaStream,
                dirError)) {
            if (is_scratch_contention_exhausted(dirError)) {
                JTRACE("CUDA", std::string("CUDA spatial DIR scratch deferred by contention policy: ") + dirError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
            }
            mark_context_loss_recovery("command_ensure_spatial_dir_scratch", cudaErrorUnknown, dirError);
            JTRACE("CUDA", std::string("CUDA spatial DIR scratch allocation failed: ") + dirError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
            throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
            throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
        }
        if (!JuicerCuda::ResourceManager::command_ensure_spatial_dir_kernel(
                submissionTxn,
                *resources,
                resources->spatialDirKernel,
                _dirRT.spatialSigmaPixels,
                _pCudaStream,
                dirError)) {
            mark_context_loss_recovery("command_ensure_spatial_dir_kernel", cudaErrorUnknown, dirError);
            JTRACE("CUDA", std::string("CUDA spatial DIR kernel upload failed: ") + dirError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
            throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
            throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
        }

        run.filmDevelop.spatialDir.corrY = resources->spatialDirScratch.corrY;
        run.filmDevelop.spatialDir.corrM = resources->spatialDirScratch.corrM;
        run.filmDevelop.spatialDir.corrC = resources->spatialDirScratch.corrC;

        const cudaError_t dirErr = juicer_cuda_build_spatial_dir(
            &run,
            resources->spatialDirScratch.corrY,
            resources->spatialDirScratch.corrM,
            resources->spatialDirScratch.corrC,
            resources->spatialDirScratch.tmp,
            resources->spatialDirKernel.weights,
            resources->spatialDirKernel.radius,
            _pCudaStream);
        if (dirErr != cudaSuccess) {
            const char* msg = cudaGetErrorString(dirErr);
            mark_context_loss_recovery("build_spatial_dir", dirErr, msg ? msg : "");
            JTRACE("CUDA", std::string("FATAL: spatial DIR build failed: ") + (msg ? msg : "(unknown)"));
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        return true;
    };

    // RenderMode::NegativeOnly (PrintBypass=true).
    if (renderMode == RenderMode::NegativeOnly) {
        ScannerPreflightResult scannerPreflight{};
        std::string scannerPreflightError;
        if (!validate_scanner_preflight_runtime(
                _ws->negativeScannerValid,
                "negative",
                &_ws->negativeMediumRuntime,
                scannerPreflight,
                scannerPreflightError)) {
            JTRACE("MSSKV", std::string("path=cuda result=fail medium=negative reason=") + scannerPreflightError);
            JTRACE("CUDA", std::string("FATAL: ") + scannerPreflightError);
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        JTRACE_VERBOSE("MSSKV", std::string("path=cuda result=ok medium=negative static_key_hash=")
            + std::to_string(scannerPreflight.staticKey.hash));
        const Scanner::ScannerMediumRuntime& negativeMediumRuntime = *scannerPreflight.mediumRuntime;

        const float lensBlurSigmaPx = _scannerOptions.lensBlurSigmaPx;
        const float unsharpSigmaPx = _scannerOptions.unsharpSigmaPx;
        const float unsharpAmount = _scannerOptions.unsharpAmount;
        const bool glareActive = negativeMediumRuntime.glare.active && (negativeMediumRuntime.glare.percent > 0.0f);
        const bool useSpatialDIR = (_dirRT.active && std::isfinite(_dirRT.spatialSigmaPixels) && _dirRT.spatialSigmaPixels > 0.0f);

        JuicerCuda::PipelineRunParams run{};
        run.src = srcPtr;
        run.srcRowBytes = static_cast<std::size_t>(srcRowBytes);
        run.dst = dstPtr;
        run.dstRowBytes = static_cast<std::size_t>(dstRowBytes);
        run.width = width;
        run.height = height;
        run.nComponents = _nComponents;

        // Film raw conversion payload
        run.filmRaw.inputColorSpaceIndex = Spectral::inputColorSpaceToIndex(_ws->filmRaw.inputColorSpace);
        run.filmRaw.applyCctfDecoding = _ws->filmRaw.applyCctfDecoding ? 1 : 0;
        run.filmRaw.applyInputChromaticAdapt = _ws->filmRaw.applyInputChromaticAdapt ? 1 : 0;
        run.filmRaw.spectralUpsamplingMode = static_cast<int>(_ws->filmRaw.spectralUpsamplingMode);
        for (int i = 0; i < 9; ++i) {
            run.filmRaw.inputRGBToXYZ[i] = _ws->filmRaw.inputRGBToXYZ.m[i];
            run.filmRaw.inputXYZAdapt[i] = _ws->filmRaw.inputXYZAdapt.m[i];
        }
        run.filmRaw.midgrayScale = _ws->filmRaw.midgrayScale;
        for (int i = 0; i < 3; ++i) {
            run.filmRaw.refIllumWhiteXYZ[i] = _ws->filmRaw.refIllumWhiteXYZ[i];
        }

        run.filmExpose.exposureScale = _exposureScale;
        run.filmDevelop.gammaFactorB = _ws->gammaFactorB;
        run.filmDevelop.gammaFactorG = _ws->gammaFactorG;
        run.filmDevelop.gammaFactorR = _ws->gammaFactorR;
        run.filmDevelop.dirPrecorrected = _ws->dirPrecorrected ? 1 : 0;

        // DIR runtime payload.
        run.filmDevelop.dir.active = _dirRT.active ? 1 : 0;
        run.filmDevelop.dir.highShift = _dirRT.highShift;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                run.filmDevelop.dir.M[r * 3 + c] = _dirRT.M[r][c];
            }
        }
        for (int i = 0; i < 3; ++i) {
            run.filmDevelop.dir.dMax[i] = _dirRT.dMax[i];
        }

        // Scan color payload + output encoding
        {
            const Scanner::ColorRuntime& color = *scannerPreflight.colorRuntime;
            for (int i = 0; i < 9; ++i) {
                run.scanStage.scanColor.cat02[i] = color.cat02[i];
                run.scanStage.scanColor.xyzToRgb[i] = color.xyzToRgb[i];
            }
            for (int i = 0; i < 3; ++i) {
                run.scanStage.scanColor.illuminantXYZ[i] = color.illuminantXYZ[i];
            }

            run.scanStage.scanColor.encoding.outputColorSpaceIndex = OutputEncoding::toIndex(color.encoding.colorSpace);
            run.scanStage.scanColor.encoding.applyCctfEncoding = color.encoding.applyCctfEncoding ? 1 : 0;
            run.scanStage.scanColor.encoding.preserveLinearRange = color.encoding.preserveLinearRange ? 1 : 0;
            run.scanStage.scanColor.encoding.inputIsOutputSpace = color.encoding.inputIsOutputSpace ? 1 : 0;

            const auto& outSpace = GeneratedColorSpaces::get(color.encoding.colorSpace);
            run.scanStage.scanColor.encoding.cctf.kind = static_cast<int>(outSpace.cctf.kind);
            run.scanStage.scanColor.encoding.cctf.gamma = outSpace.cctf.gamma;
            run.scanStage.scanColor.encoding.cctf.a = outSpace.cctf.a;
            run.scanStage.scanColor.encoding.cctf.b = outSpace.cctf.b;
            run.scanStage.scanColor.encoding.cctf.c = outSpace.cctf.c;
            run.scanStage.scanColor.encoding.cctf.d = outSpace.cctf.d;
            run.scanStage.scanColor.encoding.cctf.linearCutoff = outSpace.cctf.linearCutoff;

            const OutputEncoding::Matrix3x3 dwgToOutput = OutputEncoding::dwg_to_output_matrix(color.encoding.colorSpace);
            for (int i = 0; i < 9; ++i) {
                run.scanStage.scanColor.encoding.dwgToOutput[i] = dwgToOutput.m[i];
            }
        }

        const cudaStream_t stream = _pCudaStream ? reinterpret_cast<cudaStream_t>(_pCudaStream) : nullptr;

        {
            if (!cudaResources) {
                JTRACE("CUDA", "FATAL: CUDA resources missing for negative pipeline");
                throw OFX::Exception::Suite(kOfxStatErrFatal);
            }

            if (_effect.abort()) {
                JuicerCuda::record_use(*cudaResources, _pCudaStream);
                return;
            }

            setup_camera_auto_exposure(run, cudaResources);

            run.filmDevelop.densB = { cudaResources->densB.x, cudaResources->densB.y, cudaResources->densB.n, cudaResources->densB.domainBegin, cudaResources->densB.domainEnd };
            run.filmDevelop.densG = { cudaResources->densG.x, cudaResources->densG.y, cudaResources->densG.n, cudaResources->densG.domainBegin, cudaResources->densG.domainEnd };
            run.filmDevelop.densR = { cudaResources->densR.x, cudaResources->densR.y, cudaResources->densR.n, cudaResources->densR.domainBegin, cudaResources->densR.domainEnd };
            run.filmDevelop.dirDensB = { cudaResources->dirDensB.x, cudaResources->dirDensB.y, cudaResources->dirDensB.n, cudaResources->dirDensB.domainBegin, cudaResources->dirDensB.domainEnd };
            run.filmDevelop.dirDensG = { cudaResources->dirDensG.x, cudaResources->dirDensG.y, cudaResources->dirDensG.n, cudaResources->dirDensG.domainBegin, cudaResources->dirDensG.domainEnd };
            run.filmDevelop.dirDensR = { cudaResources->dirDensR.x, cudaResources->dirDensR.y, cudaResources->dirDensR.n, cudaResources->dirDensR.domainBegin, cudaResources->dirDensR.domainEnd };
            run.filmExpose.sensB = { cudaResources->sensB.x, cudaResources->sensB.y, cudaResources->sensB.n, cudaResources->sensB.domainBegin, cudaResources->sensB.domainEnd };
            run.filmExpose.sensG = { cudaResources->sensG.x, cudaResources->sensG.y, cudaResources->sensG.n, cudaResources->sensG.domainBegin, cudaResources->sensG.domainEnd };
            run.filmExpose.sensR = { cudaResources->sensR.x, cudaResources->sensR.y, cudaResources->sensR.n, cudaResources->sensR.domainBegin, cudaResources->sensR.domainEnd };

            run.filmExpose.tablesAx = cudaResources->tablesAx;
            run.filmExpose.tablesAy = cudaResources->tablesAy;
            run.filmExpose.tablesAz = cudaResources->tablesAz;
            run.filmExpose.tablesIllum = cudaResources->tablesIllum;
            run.filmExpose.tablesK = cudaResources->tablesK;
            for (int i = 0; i < 9; ++i) {
                run.filmExpose.spdSInv[i] = cudaResources->spdSInv[i];
            }

            run.filmExpose.hanatosLut = cudaResources->hanatosLut;
            run.filmExpose.hanatosN = cudaResources->hanatosN;
            run.filmExpose.hanatosLutIntegrated = cudaResources->hanatosLutIntegrated;
            run.filmExpose.hanatosNIntegrated = cudaResources->hanatosNIntegrated;
            run.filmExpose.mallettBasis = cudaResources->mallettBasis;
            run.filmExpose.mallettBasisK = cudaResources->mallettBasisK;

            cudaEvent_t scanEvent = setup_scan_stage_resources(cudaResources, run, stream, true);
            if (!setup_spatial_dir_stage(cudaResources, run, width, height, useSpatialDIR)) {
                return;
            }

            const bool wantGlare = glareActive;
            float glarePercent = 0.0f;
            float glareRoughness = 0.0f;
            float glareBlurSigmaPx = 0.0f;
            std::uint64_t glareSeed = 0;
            if (wantGlare && _ws) {
                glarePercent = _ws->negativeMediumRuntime.glare.percent;
                glareRoughness = _ws->negativeMediumRuntime.glare.roughness;
                glareBlurSigmaPx = _ws->negativeMediumRuntime.glare.blur;

                const std::uint64_t sessionSeed = safe_session_seed(_instanceState);
                const std::uint64_t seedBase = make_seed_base(_clipToken, _frameIndex, sessionSeed, kSeedPassGlare);

                const std::uint64_t glareFields[4] = {
                    seedBase,
                    static_cast<std::uint64_t>(_frameBoundsVersion),
                    _ws->negativeMediumRuntime.staticKey.glareHash,
                    static_cast<std::uint64_t>(Scanner::ScannerMedium::Negative)
                };
                glareSeed = Hash::hash_bytes(glareFields, sizeof(glareFields));
            }

            const Profiles::HalationMetadata halationUi = _hasHalationOverride ? _halationOverride : Profiles::HalationMetadata{};
            float halationStrengthBGR[3] = {
                halationUi.strength[2],
                halationUi.strength[1],
                halationUi.strength[0]
            };
            float halationScatterStrengthBGR[3] = {
                halationUi.scatteringStrength[2],
                halationUi.scatteringStrength[1],
                halationUi.scatteringStrength[0]
            };
            float halationSizeBGR[3] = {
                halationUi.sizeUm[2],
                halationUi.sizeUm[1],
                halationUi.sizeUm[0]
            };
            float halationScatterSizeBGR[3] = {
                halationUi.scatteringSizeUm[2],
                halationUi.scatteringSizeUm[1],
                halationUi.scatteringSizeUm[0]
            };
            for (int i = 0; i < 3; ++i) {
                if (!std::isfinite(halationStrengthBGR[i]) || halationStrengthBGR[i] <= 0.0f) {
                    halationStrengthBGR[i] = 0.0f;
                }
                if (!std::isfinite(halationScatterStrengthBGR[i]) || halationScatterStrengthBGR[i] <= 0.0f) {
                    halationScatterStrengthBGR[i] = 0.0f;
                }
                if (!std::isfinite(halationSizeBGR[i]) || halationSizeBGR[i] <= 0.0f) {
                    halationSizeBGR[i] = 0.0f;
                }
                if (!std::isfinite(halationScatterSizeBGR[i]) || halationScatterSizeBGR[i] <= 0.0f) {
                    halationScatterSizeBGR[i] = 0.0f;
                }
            }
            float halationSigmaPx[3] = { 0.0f, 0.0f, 0.0f };
            float halationScatterSigmaPx[3] = { 0.0f, 0.0f, 0.0f };
            if (std::isfinite(_pixelSizeUm) && _pixelSizeUm > 0.0f) {
                for (int i = 0; i < 3; ++i) {
                    halationSigmaPx[i] = halationSizeBGR[i] / _pixelSizeUm;
                    halationScatterSigmaPx[i] = halationScatterSizeBGR[i] / _pixelSizeUm;
                }
            }
            const bool wantHalation = halationUi.active &&
                (halationStrengthBGR[0] > 0.0f || halationStrengthBGR[1] > 0.0f || halationStrengthBGR[2] > 0.0f ||
                 halationScatterStrengthBGR[0] > 0.0f || halationScatterStrengthBGR[1] > 0.0f || halationScatterStrengthBGR[2] > 0.0f) &&
                (std::isfinite(_pixelSizeUm) && _pixelSizeUm > 0.0f);

            const Profiles::GrainMetadata grainUi = _hasGrainOverride ? _grainOverride : _ws->grain;
            const GrainSetupResult grainSetup = setup_grain_payload(run, grainUi, true);
            const bool wantGrain = grainSetup.wantGrain;
            const bool wantGrainSublayers = grainSetup.wantGrainSublayers;
            const bool wantGrainBlur = grainSetup.wantGrainBlur;
            const bool wantGrainMix = grainSetup.wantGrainMix;
            const float grainBlurSigmaPx = grainSetup.grainBlurSigmaPx;
            const float grainBlurSigmaMidPx = grainSetup.grainBlurSigmaMidPx;
            const bool needGrainShared = wantGrain &&
                (run.grain.debugView == 0 || run.grain.debugView == 1) &&
                (std::isfinite(run.grain.chromaMix) && run.grain.chromaMix < 0.999f);

            const bool wantLensBlur = std::isfinite(lensBlurSigmaPx) && lensBlurSigmaPx > 0.0f;
            const bool wantUnsharp = std::isfinite(unsharpSigmaPx) && unsharpSigmaPx > 0.0f &&
                std::isfinite(unsharpAmount) && unsharpAmount != 0.0f;
            const bool wantGlareBlur = wantGlare && std::isfinite(glareBlurSigmaPx) && glareBlurSigmaPx > 0.0f;
            const bool wantWeave = (run.gateWeave.active != 0);
            const bool wantDefects = (run.grain.filmDustAmount > 0.0f) || (run.grain.gateDustAmount > 0.0f) ||
                (run.grain.filmScratchAmount > 0.0f) || (run.grain.gateScratchAmount > 0.0f);
            const bool needGateMask = (run.grain.gateDustAmount > 0.0f) || (run.grain.gateScratchAmount > 0.0f);
            const bool wantOptics = wantLensBlur || wantUnsharp || wantGlare || wantHalation || wantGrain || wantWeave || wantDefects;

            if (_effect.abort()) {
                JuicerCuda::record_use(*cudaResources, _pCudaStream);
                return;
            }

            cudaError_t err = cudaSuccess;
            if (!wantOptics) {
                err = launch_base_pipeline_graph(
                    static_cast<int>(renderMode),
                    run);
            }
            else {
                std::string opticsError;
                const bool needBlurredScratch = wantGlareBlur || wantGrainBlur || wantGrainSublayers;
                const bool needAuxScratch = wantGrainSublayers;
                const bool needGrainScratch = wantGrainMix;
                if (!JuicerCuda::ResourceManager::command_ensure_optics_scratch(
                        submissionTxn,
                        *cudaResources,
                        width,
                        height,
                        needBlurredScratch,
                        needAuxScratch,
                        needGrainScratch,
                        needGrainShared,
                        needGateMask,
                        _pCudaStream,
                        opticsError)) {
                    if (is_scratch_contention_exhausted(opticsError)) {
                        JTRACE("CUDA", std::string("CUDA optics scratch deferred by contention policy: ") + opticsError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                        throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                        throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                    }
                    mark_context_loss_recovery("command_ensure_optics_scratch", cudaErrorUnknown, opticsError);
                    JTRACE("CUDA", std::string("CUDA optics scratch allocation failed: ") + opticsError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                }
                if (needGateMask && cudaResources->scannerScratch.gateMask) {
                    struct GateMaskHashFields {
                        std::uint64_t sessionSeed = 0;
                        std::uint64_t originX = 0;
                        std::uint64_t originY = 0;
                        std::uint64_t width = 0;
                        std::uint64_t height = 0;
                        float pixelSizeUm = 0.0f;
                        float gateDustAmount = 0.0f;
                        float gateScratchAmount = 0.0f;
                    };
                    GateMaskHashFields fields{};
                    fields.sessionSeed = run.grain.stbnSessionSeed;
                    fields.originX = static_cast<std::uint64_t>(run.grain.originX);
                    fields.originY = static_cast<std::uint64_t>(run.grain.originY);
                    fields.width = static_cast<std::uint64_t>(width);
                    fields.height = static_cast<std::uint64_t>(height);
                    fields.pixelSizeUm = run.grain.pixelSizeUm;
                    fields.gateDustAmount = run.grain.gateDustAmount;
                    fields.gateScratchAmount = run.grain.gateScratchAmount;
                    std::uint64_t gateHash = Hash::hash_bytes(&fields, sizeof(fields));
                    if (gateHash == 0) {
                        gateHash = 1;
                    }
                    if (gateHash != cudaResources->scannerScratch.gateMaskHash) {
                        if (_effect.abort()) {
                            JuicerCuda::record_use(*cudaResources, _pCudaStream);
                            return;
                        }
                        cudaError_t gateErr = juicer_cuda_build_gate_defect_mask(
                            &run,
                            cudaResources->scannerScratch.gateMask,
                            cudaResources->scannerScratch.gateWidth,
                            cudaResources->scannerScratch.gateHeight,
                            _pCudaStream);
                        if (gateErr != cudaSuccess) {
                            const char* msg = cudaGetErrorString(gateErr);
                            mark_context_loss_recovery("build_gate_mask_negative", gateErr, msg ? msg : "");
                            JTRACE("CUDA", std::string("FATAL: gate defect mask build failed: ") + (msg ? msg : "(unknown)"));
                            throw OFX::Exception::Suite(kOfxStatErrFatal);
                        }
                        cudaResources->scannerScratch.gateMaskHash = gateHash;
                    }
                    run.grain.gateMask = cudaResources->scannerScratch.gateMask;
                    run.grain.gateMaskWidth = cudaResources->scannerScratch.gateWidth;
                    run.grain.gateMaskHeight = cudaResources->scannerScratch.gateHeight;
                }
                if (!JuicerCuda::ResourceManager::command_ensure_gaussian_kernel(
                        submissionTxn,
                        *cudaResources,
                        cudaResources->scannerLensBlurKernel,
                        lensBlurSigmaPx,
                        _pCudaStream,
                        opticsError)) {
                    JTRACE("CUDA", std::string("CUDA lens blur kernel upload failed: ") + opticsError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                }
                if (!JuicerCuda::ResourceManager::command_ensure_gaussian_kernel(
                        submissionTxn,
                        *cudaResources,
                        cudaResources->scannerUnsharpKernel,
                        unsharpSigmaPx,
                        _pCudaStream,
                        opticsError)) {
                    JTRACE("CUDA", std::string("CUDA unsharp kernel upload failed: ") + opticsError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                }
                if (!JuicerCuda::ResourceManager::command_ensure_gaussian_kernel(
                        submissionTxn,
                        *cudaResources,
                        cudaResources->scannerGlareKernel,
                        wantGlare ? glareBlurSigmaPx : 0.0f,
                        _pCudaStream,
                        opticsError)) {
                    JTRACE("CUDA", std::string("CUDA glare kernel upload failed: ") + opticsError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                }

                run.grainKernels.blurKernel = nullptr;
                run.grainKernels.blurRadius = 0;
                run.grainKernels.blurKernelMid = nullptr;
                run.grainKernels.blurRadiusMid = 0;
                run.grainKernels.blurKernelCoarse = nullptr;
                run.grainKernels.blurRadiusCoarse = 0;
                for (int layer = 0; layer < 3; ++layer) {
                    for (int ch = 0; ch < 3; ++ch) {
                        run.grainKernels.dyeKernel[layer][ch] = nullptr;
                        run.grainKernels.dyeRadius[layer][ch] = 0;
                    }
                }
                if (wantGrain) {
                    if (!JuicerCuda::ResourceManager::command_ensure_gaussian_kernel(
                            submissionTxn,
                            *cudaResources,
                            cudaResources->grainBlurKernel,
                            wantGrainBlur ? grainBlurSigmaPx : 0.0f,
                            _pCudaStream,
                            opticsError)) {
                        JTRACE("CUDA", std::string("CUDA grain blur kernel upload failed: ") + opticsError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                        throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                        throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                    }
                    if (wantGrainBlur) {
                        run.grainKernels.blurKernel = cudaResources->grainBlurKernel.weights;
                        run.grainKernels.blurRadius = cudaResources->grainBlurKernel.radius;
                    }
                    if (wantGrainMix) {
                        if (!JuicerCuda::ResourceManager::command_ensure_gaussian_kernel(
                                submissionTxn,
                                *cudaResources,
                                cudaResources->grainBlurKernelMid,
                                grainBlurSigmaMidPx,
                                _pCudaStream,
                                opticsError)) {
                            JTRACE("CUDA", std::string("CUDA grain mid blur kernel upload failed: ") + opticsError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                            throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                            throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                        }
                        if (!JuicerCuda::ResourceManager::command_ensure_gaussian_kernel(
                                submissionTxn,
                                *cudaResources,
                                cudaResources->grainBlurKernelCoarse,
                                grainSetup.grainBlurSigmaCoarsePx,
                                _pCudaStream,
                                opticsError)) {
                            JTRACE("CUDA", std::string("CUDA grain coarse blur kernel upload failed: ") + opticsError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                            throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                            throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                        }
                        run.grainKernels.blurKernelMid = cudaResources->grainBlurKernelMid.weights;
                        run.grainKernels.blurRadiusMid = cudaResources->grainBlurKernelMid.radius;
                        run.grainKernels.blurKernelCoarse = cudaResources->grainBlurKernelCoarse.weights;
                        run.grainKernels.blurRadiusCoarse = cudaResources->grainBlurKernelCoarse.radius;
                    }

                    if (wantGrainSublayers) {
                        for (int layer = 0; layer < 3; ++layer) {
                            for (int ch = 0; ch < 3; ++ch) {
                                const float sigma = grainSetup.grainDyeSigmaPx[layer][ch];
                                if (!JuicerCuda::ResourceManager::command_ensure_gaussian_kernel(
                                        submissionTxn,
                                        *cudaResources,
                                        cudaResources->grainDyeKernel[layer][ch],
                                        sigma,
                                        _pCudaStream,
                                        opticsError)) {
                                    JTRACE("CUDA", std::string("CUDA grain dye-cloud kernel upload failed: ") + opticsError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                                    throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                                }
                                if (sigma > 0.0f) {
                                    run.grainKernels.dyeKernel[layer][ch] = cudaResources->grainDyeKernel[layer][ch].weights;
                                    run.grainKernels.dyeRadius[layer][ch] = cudaResources->grainDyeKernel[layer][ch].radius;
                                }
                            }
                        }
                    }
                }

                run.halation.active = wantHalation ? 1 : 0;
                for (int i = 0; i < 3; ++i) {
                    run.halation.strength[i] = wantHalation ? halationStrengthBGR[i] : 0.0f;
                    run.halation.scatteringStrength[i] = wantHalation ? halationScatterStrengthBGR[i] : 0.0f;
                    run.halationKernels.halationKernel[i] = nullptr;
                    run.halationKernels.halationRadius[i] = 0;
                    run.halationKernels.scatteringKernel[i] = nullptr;
                    run.halationKernels.scatteringRadius[i] = 0;
                }
                if (wantHalation) {
                    for (int i = 0; i < 3; ++i) {
                        if (halationStrengthBGR[i] > 0.0f && halationSigmaPx[i] > 0.0f) {
                            if (!JuicerCuda::ResourceManager::command_ensure_halation_kernel(
                                    submissionTxn,
                                    *cudaResources,
                                    cudaResources->halationKernel[i],
                                    halationSigmaPx[i],
                                    _pCudaStream,
                                    opticsError)) {
                                JTRACE("CUDA", std::string("CUDA halation kernel upload failed: ") + opticsError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                                throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                                throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                            }
                            run.halationKernels.halationKernel[i] = cudaResources->halationKernel[i].weights;
                            run.halationKernels.halationRadius[i] = cudaResources->halationKernel[i].radius;
                        }
                        if (halationScatterStrengthBGR[i] > 0.0f && halationScatterSigmaPx[i] > 0.0f) {
                            if (!JuicerCuda::ResourceManager::command_ensure_halation_kernel(
                                    submissionTxn,
                                    *cudaResources,
                                    cudaResources->halationScatterKernel[i],
                                    halationScatterSigmaPx[i],
                                    _pCudaStream,
                                    opticsError)) {
                                JTRACE("CUDA", std::string("CUDA halation scatter kernel upload failed: ") + opticsError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                                throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                                throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                            }
                            run.halationKernels.scatteringKernel[i] = cudaResources->halationScatterKernel[i].weights;
                            run.halationKernels.scatteringRadius[i] = cudaResources->halationScatterKernel[i].radius;
                        }
                    }
                }

                err = juicer_cuda_negative_pipeline_optics(
                    &run,
                    cudaResources->scannerScratch.rgbR,
                    cudaResources->scannerScratch.rgbG,
                    cudaResources->scannerScratch.rgbB,
                    cudaResources->scannerScratch.tmp,
                    cudaResources->scannerScratch.blurred,
                    cudaResources->scannerScratch.aux,
                    cudaResources->scannerScratch.grainTmp,
                    cudaResources->scannerScratch.grainTmpShared,
                    cudaResources->scannerScratch.grainTmpMid,
                    cudaResources->scannerScratch.grainTmpCoarse,
                    cudaResources->scannerLensBlurKernel.weights,
                    cudaResources->scannerLensBlurKernel.radius,
                    cudaResources->scannerUnsharpKernel.weights,
                    cudaResources->scannerUnsharpKernel.radius,
                    unsharpAmount,
                    win.x1,
                    win.y1,
                    glareSeed,
                    glarePercent,
                    glareRoughness,
                    cudaResources->scannerGlareKernel.weights,
                    cudaResources->scannerGlareKernel.radius,
                    _pCudaStream);
            }
            if (err != cudaSuccess) {
                const char* msg = cudaGetErrorString(err);
                mark_context_loss_recovery("negative_pipeline_kernel_launch", err, msg ? msg : "");
                JTRACE("CUDA", std::string("FATAL: negative pipeline kernel launch failed: ") + (msg ? msg : "(unknown)"));
                throw OFX::Exception::Suite(kOfxStatErrFatal);
            }

            if (_effect.abort()) {
                JuicerCuda::record_use(*cudaResources, _pCudaStream);
                return;
            }

            finalize_scan_error_stage(cudaResources, run, stream, scanEvent, "negative");

            JuicerCuda::record_use(*cudaResources, _pCudaStream);
        }
        commit_submission_or_throw();
        return;
    }

    // RenderMode::Print (PrintBypass=false).
    {
        if (!_printReady || !_prt) {
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
            throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
            throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
        }

        ScannerPreflightResult scannerPreflight{};
        std::string scannerPreflightError;
        if (!validate_scanner_preflight_runtime(
                _ws->printScannerValid,
                "print",
                &_ws->printMediumRuntime,
                scannerPreflight,
                scannerPreflightError)) {
            JTRACE("MSSKV", std::string("path=cuda result=fail medium=print reason=") + scannerPreflightError);
            JTRACE("CUDA", std::string("FATAL: ") + scannerPreflightError);
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        JTRACE_VERBOSE("MSSKV", std::string("path=cuda result=ok medium=print static_key_hash=")
            + std::to_string(scannerPreflight.staticKey.hash));

        const bool useSpatialDIR = (_dirRT.active && std::isfinite(_dirRT.spatialSigmaPixels) && _dirRT.spatialSigmaPixels > 0.0f);

        // Print exposure compensation factor is computed on CPU (no image reads; safe for CUDA renders).
        const float kMidSpectral = compute_print_midgray_factor_cached(
            _instanceState,
            *_ws,
            *_prt,
            _printParams,
            _dirRT);

        JuicerCuda::PipelineRunParams run{};
        run.src = srcPtr;
        run.srcRowBytes = static_cast<std::size_t>(srcRowBytes);
        run.dst = dstPtr;
        run.dstRowBytes = static_cast<std::size_t>(dstRowBytes);
        run.width = width;
        run.height = height;
        run.nComponents = _nComponents;

        // Film raw conversion payload
        run.filmRaw.inputColorSpaceIndex = Spectral::inputColorSpaceToIndex(_ws->filmRaw.inputColorSpace);
        run.filmRaw.applyCctfDecoding = _ws->filmRaw.applyCctfDecoding ? 1 : 0;
        run.filmRaw.applyInputChromaticAdapt = _ws->filmRaw.applyInputChromaticAdapt ? 1 : 0;
        run.filmRaw.spectralUpsamplingMode = static_cast<int>(_ws->filmRaw.spectralUpsamplingMode);
        for (int i = 0; i < 9; ++i) {
            run.filmRaw.inputRGBToXYZ[i] = _ws->filmRaw.inputRGBToXYZ.m[i];
            run.filmRaw.inputXYZAdapt[i] = _ws->filmRaw.inputXYZAdapt.m[i];
        }
        run.filmRaw.midgrayScale = _ws->filmRaw.midgrayScale;
        for (int i = 0; i < 3; ++i) {
            run.filmRaw.refIllumWhiteXYZ[i] = _ws->filmRaw.refIllumWhiteXYZ[i];
        }

        run.filmExpose.exposureScale = _exposureScale;
        run.filmDevelop.gammaFactorB = _ws->gammaFactorB;
        run.filmDevelop.gammaFactorG = _ws->gammaFactorG;
        run.filmDevelop.gammaFactorR = _ws->gammaFactorR;
        run.filmDevelop.dirPrecorrected = _ws->dirPrecorrected ? 1 : 0;

        // DIR runtime payload.
        run.filmDevelop.dir.active = _dirRT.active ? 1 : 0;
        run.filmDevelop.dir.highShift = _dirRT.highShift;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                run.filmDevelop.dir.M[r * 3 + c] = _dirRT.M[r][c];
            }
        }
        for (int i = 0; i < 3; ++i) {
            run.filmDevelop.dir.dMax[i] = _dirRT.dMax[i];
        }

        // Print scan color payload + output encoding (print medium).
        {
            const Scanner::ColorRuntime& color = *scannerPreflight.colorRuntime;
            for (int i = 0; i < 9; ++i) {
                run.scanStage.scanColor.cat02[i] = color.cat02[i];
                run.scanStage.scanColor.xyzToRgb[i] = color.xyzToRgb[i];
            }
            for (int i = 0; i < 3; ++i) {
                run.scanStage.scanColor.illuminantXYZ[i] = color.illuminantXYZ[i];
            }

            run.scanStage.scanColor.encoding.outputColorSpaceIndex = OutputEncoding::toIndex(color.encoding.colorSpace);
            run.scanStage.scanColor.encoding.applyCctfEncoding = color.encoding.applyCctfEncoding ? 1 : 0;
            run.scanStage.scanColor.encoding.preserveLinearRange = color.encoding.preserveLinearRange ? 1 : 0;
            run.scanStage.scanColor.encoding.inputIsOutputSpace = color.encoding.inputIsOutputSpace ? 1 : 0;

            const auto& outSpace = GeneratedColorSpaces::get(color.encoding.colorSpace);
            run.scanStage.scanColor.encoding.cctf.kind = static_cast<int>(outSpace.cctf.kind);
            run.scanStage.scanColor.encoding.cctf.gamma = outSpace.cctf.gamma;
            run.scanStage.scanColor.encoding.cctf.a = outSpace.cctf.a;
            run.scanStage.scanColor.encoding.cctf.b = outSpace.cctf.b;
            run.scanStage.scanColor.encoding.cctf.c = outSpace.cctf.c;
            run.scanStage.scanColor.encoding.cctf.d = outSpace.cctf.d;
            run.scanStage.scanColor.encoding.cctf.linearCutoff = outSpace.cctf.linearCutoff;

            const OutputEncoding::Matrix3x3 dwgToOutput = OutputEncoding::dwg_to_output_matrix(color.encoding.colorSpace);
            for (int i = 0; i < 9; ++i) {
                run.scanStage.scanColor.encoding.dwgToOutput[i] = dwgToOutput.m[i];
            }
        }

        const cudaStream_t stream = _pCudaStream ? reinterpret_cast<cudaStream_t>(_pCudaStream) : nullptr;

        {
            if (!cudaResources) {
                JTRACE("CUDA", "FATAL: CUDA resources missing for print pipeline");
                throw OFX::Exception::Suite(kOfxStatErrFatal);
            }

            if (_effect.abort()) {
                JuicerCuda::record_use(*cudaResources, _pCudaStream);
                return;
            }

            setup_camera_auto_exposure(run, cudaResources);

            // Ensure the print illuminant filtered is available for current print params.
            std::string illumError;
            if (!JuicerCuda::ResourceManager::command_ensure_print_illuminant_filtered(
                    submissionTxn,
                    *cudaResources,
                    *_ws,
                    *_prt,
                    _printParams,
                    _pCudaStream,
                    illumError)) {
                mark_context_loss_recovery("command_ensure_print_illuminant_filtered", cudaErrorUnknown, illumError);
                JTRACE("CUDA", std::string("CUDA print illuminant upload failed: ") + illumError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
            }
            if (JTRACE_ENABLED(3)) {
                std::lock_guard<std::mutex> resLock(cudaResources->m);
                std::string msg = std::string("cuda print payload build=") + std::to_string(_ws ? _ws->buildCounter : 0)
                    + " uploadCoreHash=" + std::to_string(_ws ? ((_ws->uploadCoreHash != 0) ? _ws->uploadCoreHash : _ws->coreHash) : 0)
                    + " neutralY/M/C=" + std::to_string(_prt ? _prt->neutralY : 0.0f)
                    + "/" + std::to_string(_prt ? _prt->neutralM : 0.0f)
                    + "/" + std::to_string(_prt ? _prt->neutralC : 0.0f)
                    + " yFilter=" + std::to_string(_printParams.yFilter)
                    + " mFilter=" + std::to_string(_printParams.mFilter)
                    + " cFilter=" + std::to_string(_printParams.cFilter)
                    + " illumBuild=" + std::to_string(cudaResources->printIllumBuildCounter)
                    + " illumCoreHash=" + std::to_string(cudaResources->printIllumCoreHash)
                    + " illumNeutralHash=" + std::to_string(cudaResources->printIllumNeutralFilterHash)
                    + " illumY/M/Csteps=" + std::to_string(cudaResources->printIllumYShiftSteps)
                    + "/" + std::to_string(cudaResources->printIllumMShiftSteps)
                    + "/" + std::to_string(cudaResources->printIllumCShiftSteps)
                    + " preflashValid=" + std::to_string(cudaResources->printPreflashValid ? 1 : 0)
                    + " preflashBuild=" + std::to_string(cudaResources->printPreflashBuildCounter);
                JTRACE_VERBOSE("PRINTDBG", msg);
            }

            // Film density curves + sensitivities + SPD reconstruction tables.
            run.filmDevelop.densB = { cudaResources->densB.x, cudaResources->densB.y, cudaResources->densB.n, cudaResources->densB.domainBegin, cudaResources->densB.domainEnd };
            run.filmDevelop.densG = { cudaResources->densG.x, cudaResources->densG.y, cudaResources->densG.n, cudaResources->densG.domainBegin, cudaResources->densG.domainEnd };
            run.filmDevelop.densR = { cudaResources->densR.x, cudaResources->densR.y, cudaResources->densR.n, cudaResources->densR.domainBegin, cudaResources->densR.domainEnd };
            run.filmDevelop.dirDensB = { cudaResources->dirDensB.x, cudaResources->dirDensB.y, cudaResources->dirDensB.n, cudaResources->dirDensB.domainBegin, cudaResources->dirDensB.domainEnd };
            run.filmDevelop.dirDensG = { cudaResources->dirDensG.x, cudaResources->dirDensG.y, cudaResources->dirDensG.n, cudaResources->dirDensG.domainBegin, cudaResources->dirDensG.domainEnd };
            run.filmDevelop.dirDensR = { cudaResources->dirDensR.x, cudaResources->dirDensR.y, cudaResources->dirDensR.n, cudaResources->dirDensR.domainBegin, cudaResources->dirDensR.domainEnd };
            run.filmExpose.sensB = { cudaResources->sensB.x, cudaResources->sensB.y, cudaResources->sensB.n, cudaResources->sensB.domainBegin, cudaResources->sensB.domainEnd };
            run.filmExpose.sensG = { cudaResources->sensG.x, cudaResources->sensG.y, cudaResources->sensG.n, cudaResources->sensG.domainBegin, cudaResources->sensG.domainEnd };
            run.filmExpose.sensR = { cudaResources->sensR.x, cudaResources->sensR.y, cudaResources->sensR.n, cudaResources->sensR.domainBegin, cudaResources->sensR.domainEnd };

            run.filmExpose.tablesAx = cudaResources->tablesAx;
            run.filmExpose.tablesAy = cudaResources->tablesAy;
            run.filmExpose.tablesAz = cudaResources->tablesAz;
            run.filmExpose.tablesIllum = cudaResources->tablesIllum;
            run.filmExpose.tablesK = cudaResources->tablesK;
            for (int i = 0; i < 9; ++i) {
                run.filmExpose.spdSInv[i] = cudaResources->spdSInv[i];
            }

            run.filmExpose.hanatosLut = cudaResources->hanatosLut;
            run.filmExpose.hanatosN = cudaResources->hanatosN;
            run.filmExpose.hanatosLutIntegrated = cudaResources->hanatosLutIntegrated;
            run.filmExpose.hanatosNIntegrated = cudaResources->hanatosNIntegrated;
            run.filmExpose.mallettBasis = cudaResources->mallettBasis;
            run.filmExpose.mallettBasisK = cudaResources->mallettBasisK;

            cudaEvent_t scanEvent = setup_scan_stage_resources(cudaResources, run, stream, false);
            if (!setup_spatial_dir_stage(cudaResources, run, width, height, useSpatialDIR)) {
                return;
            }

            // Print pipeline payloads.
            run.printExpose.active = 1;
            run.printExpose.negTables.epsC = cudaResources->scanNegative.tables.epsC;
            run.printExpose.negTables.epsM = cudaResources->scanNegative.tables.epsM;
            run.printExpose.negTables.epsY = cudaResources->scanNegative.tables.epsY;
            run.printExpose.negTables.Ax = cudaResources->scanNegative.tables.Ax;
            run.printExpose.negTables.Ay = cudaResources->scanNegative.tables.Ay;
            run.printExpose.negTables.Az = cudaResources->scanNegative.tables.Az;
            run.printExpose.negTables.baseMin = cudaResources->scanNegative.tables.baseMin;
            run.printExpose.negTables.K = cudaResources->scanNegative.tables.K;
            run.printExpose.negTables.hasBaseline = cudaResources->scanNegative.tables.hasBaseline;
            run.printExpose.negTables.invYn = cudaResources->scanNegative.tables.invYn;
            run.printExpose.negTables.mediumIsNegative = cudaResources->scanNegative.mediumIsNegative;
            for (int i = 0; i < 3; ++i) {
                run.printExpose.negTables.min_cmy[i] = cudaResources->scanNegative.min_cmy[i];
                run.printExpose.negTables.inv_max_cmy[i] = cudaResources->scanNegative.inv_max_cmy[i];
            }

            run.printExpose.printIllumFiltered = cudaResources->printIllumFiltered;
            run.printExpose.printIllumK = cudaResources->printIllumK;
            run.printExpose.printSensC = { cudaResources->printSensC.x, cudaResources->printSensC.y, cudaResources->printSensC.n, cudaResources->printSensC.domainBegin, cudaResources->printSensC.domainEnd };
            run.printExpose.printSensM = { cudaResources->printSensM.x, cudaResources->printSensM.y, cudaResources->printSensM.n, cudaResources->printSensM.domainBegin, cudaResources->printSensM.domainEnd };
            run.printExpose.printSensY = { cudaResources->printSensY.x, cudaResources->printSensY.y, cudaResources->printSensY.n, cudaResources->printSensY.domainBegin, cudaResources->printSensY.domainEnd };
            run.printDevelop.printDcC = { cudaResources->printDcC.x, cudaResources->printDcC.y, cudaResources->printDcC.n, cudaResources->printDcC.domainBegin, cudaResources->printDcC.domainEnd };
            run.printDevelop.printDcM = { cudaResources->printDcM.x, cudaResources->printDcM.y, cudaResources->printDcM.n, cudaResources->printDcM.domainBegin, cudaResources->printDcM.domainEnd };
            run.printDevelop.printDcY = { cudaResources->printDcY.x, cudaResources->printDcY.y, cudaResources->printDcY.n, cudaResources->printDcY.domainBegin, cudaResources->printDcY.domainEnd };
            run.printDevelop.printGammaC = cudaResources->printGammaC;
            run.printDevelop.printGammaM = cudaResources->printGammaM;
            run.printDevelop.printGammaY = cudaResources->printGammaY;
            run.printExpose.printExposure = _printParams.exposure;
            run.printExpose.printPreflashExposure = _printParams.preflashExposure;
            run.printExpose.printMidgrayFactor = kMidSpectral;
            for (int i = 0; i < 3; ++i) {
                run.printExpose.printPreflashRaw[i] = cudaResources->printPreflashRaw[i];
            }

            if (!run.printExpose.printIllumFiltered || run.printExpose.printIllumK <= 0 ||
                !run.printExpose.printSensC.y || !run.printExpose.printSensM.y || !run.printExpose.printSensY.y ||
                !run.printDevelop.printDcC.y || !run.printDevelop.printDcM.y || !run.printDevelop.printDcY.y) {
                JTRACE("CUDA", "CUDA print payloads missing; cannot render print pipeline");
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
            }

            // Scanner optics/glare for the print medium.
            Scanner::ScannerMediumRuntime printMedium = _ws->printMediumRuntime;
            if (_hasPrintGlareOverride) {
                printMedium.glare.active = _printGlareOverride.active;
                printMedium.glare.percent = _printGlareOverride.percent;
                printMedium.glare.roughness = _printGlareOverride.roughness;
                printMedium.glare.blur = _printGlareOverride.blur;
                printMedium.glare.compensationRemovalFactor = 0.0f;
                printMedium.glare.compensationRemovalDensity = 0.0f;
                printMedium.glare.compensationRemovalTransition = 0.0f;
                const std::uint64_t glareHash = Scanner::hash_glare(printMedium.glare);
                if (glareHash == 0) {
                    JTRACE("HASH", "FATAL: failed to hash print glare override parameters");
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
                }
                printMedium.staticKey.glareHash = glareHash;
            }

            const bool wantGlare = printMedium.glare.active && (printMedium.glare.percent > 0.0f);
            float glarePercent = 0.0f;
            float glareRoughness = 0.0f;
            float glareBlurSigmaPx = 0.0f;
            std::uint64_t glareSeed = 0;
            if (wantGlare) {
                glarePercent = printMedium.glare.percent;
                glareRoughness = printMedium.glare.roughness;
                glareBlurSigmaPx = printMedium.glare.blur;

                const std::uint64_t sessionSeed = safe_session_seed(_instanceState);
                const std::uint64_t seedBase = make_seed_base(_clipToken, _frameIndex, sessionSeed, kSeedPassGlare);

                const std::uint64_t glareFields[4] = {
                    seedBase,
                    static_cast<std::uint64_t>(_frameBoundsVersion),
                    printMedium.staticKey.glareHash,
                    static_cast<std::uint64_t>(Scanner::ScannerMedium::Print)
                };
                glareSeed = Hash::hash_bytes(glareFields, sizeof(glareFields));
            }

            const Profiles::HalationMetadata halationUi = _hasHalationOverride ? _halationOverride : Profiles::HalationMetadata{};
            float halationStrengthBGR[3] = {
                halationUi.strength[2],
                halationUi.strength[1],
                halationUi.strength[0]
            };
            float halationScatterStrengthBGR[3] = {
                halationUi.scatteringStrength[2],
                halationUi.scatteringStrength[1],
                halationUi.scatteringStrength[0]
            };
            float halationSizeBGR[3] = {
                halationUi.sizeUm[2],
                halationUi.sizeUm[1],
                halationUi.sizeUm[0]
            };
            float halationScatterSizeBGR[3] = {
                halationUi.scatteringSizeUm[2],
                halationUi.scatteringSizeUm[1],
                halationUi.scatteringSizeUm[0]
            };
            for (int i = 0; i < 3; ++i) {
                if (!std::isfinite(halationStrengthBGR[i]) || halationStrengthBGR[i] <= 0.0f) {
                    halationStrengthBGR[i] = 0.0f;
                }
                if (!std::isfinite(halationScatterStrengthBGR[i]) || halationScatterStrengthBGR[i] <= 0.0f) {
                    halationScatterStrengthBGR[i] = 0.0f;
                }
                if (!std::isfinite(halationSizeBGR[i]) || halationSizeBGR[i] <= 0.0f) {
                    halationSizeBGR[i] = 0.0f;
                }
                if (!std::isfinite(halationScatterSizeBGR[i]) || halationScatterSizeBGR[i] <= 0.0f) {
                    halationScatterSizeBGR[i] = 0.0f;
                }
            }
            float halationSigmaPx[3] = { 0.0f, 0.0f, 0.0f };
            float halationScatterSigmaPx[3] = { 0.0f, 0.0f, 0.0f };
            if (std::isfinite(_pixelSizeUm) && _pixelSizeUm > 0.0f) {
                for (int i = 0; i < 3; ++i) {
                    halationSigmaPx[i] = halationSizeBGR[i] / _pixelSizeUm;
                    halationScatterSigmaPx[i] = halationScatterSizeBGR[i] / _pixelSizeUm;
                }
            }
            const bool wantHalation = halationUi.active &&
                (halationStrengthBGR[0] > 0.0f || halationStrengthBGR[1] > 0.0f || halationStrengthBGR[2] > 0.0f ||
                 halationScatterStrengthBGR[0] > 0.0f || halationScatterStrengthBGR[1] > 0.0f || halationScatterStrengthBGR[2] > 0.0f) &&
                (std::isfinite(_pixelSizeUm) && _pixelSizeUm > 0.0f);

            const Profiles::GrainMetadata grainUi = _hasGrainOverride ? _grainOverride : _ws->grain;
            const GrainSetupResult grainSetup = setup_grain_payload(run, grainUi, false);
            const bool wantGrain = grainSetup.wantGrain;
            const bool wantGrainSublayers = grainSetup.wantGrainSublayers;
            const bool wantGrainBlur = grainSetup.wantGrainBlur;
            const bool wantGrainMix = grainSetup.wantGrainMix;
            const float grainBlurSigmaPx = grainSetup.grainBlurSigmaPx;
            const float grainBlurSigmaMidPx = grainSetup.grainBlurSigmaMidPx;
            const bool needGrainShared = wantGrain &&
                (run.grain.debugView == 0 || run.grain.debugView == 1) &&
                (std::isfinite(run.grain.chromaMix) && run.grain.chromaMix < 0.999f);

            const float lensBlurSigmaPx = _scannerOptions.lensBlurSigmaPx;
            const float unsharpSigmaPx = _scannerOptions.unsharpSigmaPx;
            const float unsharpAmount = _scannerOptions.unsharpAmount;

            const bool wantLensBlur = std::isfinite(lensBlurSigmaPx) && lensBlurSigmaPx > 0.0f;
            const bool wantUnsharp = std::isfinite(unsharpSigmaPx) && unsharpSigmaPx > 0.0f &&
                std::isfinite(unsharpAmount) && unsharpAmount != 0.0f;
            const bool wantGlareBlur = wantGlare && std::isfinite(glareBlurSigmaPx) && glareBlurSigmaPx > 0.0f;
            const bool wantWeave = (run.gateWeave.active != 0);
            const bool wantDefects = (run.grain.filmDustAmount > 0.0f) || (run.grain.gateDustAmount > 0.0f) ||
                (run.grain.filmScratchAmount > 0.0f) || (run.grain.gateScratchAmount > 0.0f);
            const bool needGateMask = (run.grain.gateDustAmount > 0.0f) || (run.grain.gateScratchAmount > 0.0f);
            const bool wantOptics = wantLensBlur || wantUnsharp || wantGlare || wantHalation || wantGrain || wantWeave || wantDefects;

            if (_effect.abort()) {
                JuicerCuda::record_use(*cudaResources, _pCudaStream);
                return;
            }

            cudaError_t err = cudaSuccess;
            if (!wantOptics) {
                err = launch_base_pipeline_graph(
                    static_cast<int>(renderMode),
                    run);
            }
            else {
                std::string opticsError;
                const bool needBlurredScratch = wantGlareBlur || wantGrainBlur || wantGrainSublayers;
                const bool needAuxScratch = wantGrainSublayers;
                const bool needGrainScratch = wantGrainMix;
                if (!JuicerCuda::ResourceManager::command_ensure_optics_scratch(
                        submissionTxn,
                        *cudaResources,
                        width,
                        height,
                        needBlurredScratch,
                        needAuxScratch,
                        needGrainScratch,
                        needGrainShared,
                        needGateMask,
                        _pCudaStream,
                        opticsError)) {
                    if (is_scratch_contention_exhausted(opticsError)) {
                        JTRACE("CUDA", std::string("CUDA optics scratch deferred by contention policy: ") + opticsError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                        throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                        throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                    }
                    mark_context_loss_recovery("command_ensure_optics_scratch", cudaErrorUnknown, opticsError);
                    JTRACE("CUDA", std::string("CUDA optics scratch allocation failed: ") + opticsError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                }
                if (needGateMask && cudaResources->scannerScratch.gateMask) {
                    struct GateMaskHashFields {
                        std::uint64_t sessionSeed = 0;
                        std::uint64_t originX = 0;
                        std::uint64_t originY = 0;
                        std::uint64_t width = 0;
                        std::uint64_t height = 0;
                        float pixelSizeUm = 0.0f;
                        float gateDustAmount = 0.0f;
                        float gateScratchAmount = 0.0f;
                    };
                    GateMaskHashFields fields{};
                    fields.sessionSeed = run.grain.stbnSessionSeed;
                    fields.originX = static_cast<std::uint64_t>(run.grain.originX);
                    fields.originY = static_cast<std::uint64_t>(run.grain.originY);
                    fields.width = static_cast<std::uint64_t>(width);
                    fields.height = static_cast<std::uint64_t>(height);
                    fields.pixelSizeUm = run.grain.pixelSizeUm;
                    fields.gateDustAmount = run.grain.gateDustAmount;
                    fields.gateScratchAmount = run.grain.gateScratchAmount;
                    std::uint64_t gateHash = Hash::hash_bytes(&fields, sizeof(fields));
                    if (gateHash == 0) {
                        gateHash = 1;
                    }
                    if (gateHash != cudaResources->scannerScratch.gateMaskHash) {
                        if (_effect.abort()) {
                            JuicerCuda::record_use(*cudaResources, _pCudaStream);
                            return;
                        }
                        cudaError_t gateErr = juicer_cuda_build_gate_defect_mask(
                            &run,
                            cudaResources->scannerScratch.gateMask,
                            cudaResources->scannerScratch.gateWidth,
                            cudaResources->scannerScratch.gateHeight,
                            _pCudaStream);
                        if (gateErr != cudaSuccess) {
                            const char* msg = cudaGetErrorString(gateErr);
                            mark_context_loss_recovery("build_gate_mask_print", gateErr, msg ? msg : "");
                            JTRACE("CUDA", std::string("FATAL: gate defect mask build failed: ") + (msg ? msg : "(unknown)"));
                            throw OFX::Exception::Suite(kOfxStatErrFatal);
                        }
                        cudaResources->scannerScratch.gateMaskHash = gateHash;
                    }
                    run.grain.gateMask = cudaResources->scannerScratch.gateMask;
                    run.grain.gateMaskWidth = cudaResources->scannerScratch.gateWidth;
                    run.grain.gateMaskHeight = cudaResources->scannerScratch.gateHeight;
                }
                if (!JuicerCuda::ResourceManager::command_ensure_gaussian_kernel(
                        submissionTxn,
                        *cudaResources,
                        cudaResources->scannerLensBlurKernel,
                        lensBlurSigmaPx,
                        _pCudaStream,
                        opticsError)) {
                    JTRACE("CUDA", std::string("CUDA lens blur kernel upload failed: ") + opticsError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                }
                if (!JuicerCuda::ResourceManager::command_ensure_gaussian_kernel(
                        submissionTxn,
                        *cudaResources,
                        cudaResources->scannerUnsharpKernel,
                        unsharpSigmaPx,
                        _pCudaStream,
                        opticsError)) {
                    JTRACE("CUDA", std::string("CUDA unsharp kernel upload failed: ") + opticsError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                }
                if (!JuicerCuda::ResourceManager::command_ensure_gaussian_kernel(
                        submissionTxn,
                        *cudaResources,
                        cudaResources->scannerGlareKernel,
                        wantGlare ? glareBlurSigmaPx : 0.0f,
                        _pCudaStream,
                        opticsError)) {
                    JTRACE("CUDA", std::string("CUDA glare kernel upload failed: ") + opticsError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                }

                run.grainKernels.blurKernel = nullptr;
                run.grainKernels.blurRadius = 0;
                run.grainKernels.blurKernelMid = nullptr;
                run.grainKernels.blurRadiusMid = 0;
                run.grainKernels.blurKernelCoarse = nullptr;
                run.grainKernels.blurRadiusCoarse = 0;
                for (int layer = 0; layer < 3; ++layer) {
                    for (int ch = 0; ch < 3; ++ch) {
                        run.grainKernels.dyeKernel[layer][ch] = nullptr;
                        run.grainKernels.dyeRadius[layer][ch] = 0;
                    }
                }
                if (wantGrain) {
                    if (!JuicerCuda::ResourceManager::command_ensure_gaussian_kernel(
                            submissionTxn,
                            *cudaResources,
                            cudaResources->grainBlurKernel,
                            wantGrainBlur ? grainBlurSigmaPx : 0.0f,
                            _pCudaStream,
                            opticsError)) {
                        JTRACE("CUDA", std::string("CUDA grain blur kernel upload failed: ") + opticsError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                        throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                        throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                    }
                    if (wantGrainBlur) {
                        run.grainKernels.blurKernel = cudaResources->grainBlurKernel.weights;
                        run.grainKernels.blurRadius = cudaResources->grainBlurKernel.radius;
                    }
                    if (wantGrainMix) {
                        if (!JuicerCuda::ResourceManager::command_ensure_gaussian_kernel(
                                submissionTxn,
                                *cudaResources,
                                cudaResources->grainBlurKernelMid,
                                grainBlurSigmaMidPx,
                                _pCudaStream,
                                opticsError)) {
                            JTRACE("CUDA", std::string("CUDA grain mid blur kernel upload failed: ") + opticsError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                            throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                            throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                        }
                        if (!JuicerCuda::ResourceManager::command_ensure_gaussian_kernel(
                                submissionTxn,
                                *cudaResources,
                                cudaResources->grainBlurKernelCoarse,
                                grainSetup.grainBlurSigmaCoarsePx,
                                _pCudaStream,
                                opticsError)) {
                            JTRACE("CUDA", std::string("CUDA grain coarse blur kernel upload failed: ") + opticsError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                            throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                            throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                        }
                        run.grainKernels.blurKernelMid = cudaResources->grainBlurKernelMid.weights;
                        run.grainKernels.blurRadiusMid = cudaResources->grainBlurKernelMid.radius;
                        run.grainKernels.blurKernelCoarse = cudaResources->grainBlurKernelCoarse.weights;
                        run.grainKernels.blurRadiusCoarse = cudaResources->grainBlurKernelCoarse.radius;
                    }

                    if (wantGrainSublayers) {
                        for (int layer = 0; layer < 3; ++layer) {
                            for (int ch = 0; ch < 3; ++ch) {
                                const float sigma = grainSetup.grainDyeSigmaPx[layer][ch];
                                if (!JuicerCuda::ResourceManager::command_ensure_gaussian_kernel(
                                        submissionTxn,
                                        *cudaResources,
                                        cudaResources->grainDyeKernel[layer][ch],
                                        sigma,
                                        _pCudaStream,
                                        opticsError)) {
                                    JTRACE("CUDA", std::string("CUDA grain dye-cloud kernel upload failed: ") + opticsError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                                    throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                                }
                                if (sigma > 0.0f) {
                                    run.grainKernels.dyeKernel[layer][ch] = cudaResources->grainDyeKernel[layer][ch].weights;
                                    run.grainKernels.dyeRadius[layer][ch] = cudaResources->grainDyeKernel[layer][ch].radius;
                                }
                            }
                        }
                    }
                }

                run.halation.active = wantHalation ? 1 : 0;
                for (int i = 0; i < 3; ++i) {
                    run.halation.strength[i] = wantHalation ? halationStrengthBGR[i] : 0.0f;
                    run.halation.scatteringStrength[i] = wantHalation ? halationScatterStrengthBGR[i] : 0.0f;
                    run.halationKernels.halationKernel[i] = nullptr;
                    run.halationKernels.halationRadius[i] = 0;
                    run.halationKernels.scatteringKernel[i] = nullptr;
                    run.halationKernels.scatteringRadius[i] = 0;
                }
                if (wantHalation) {
                    for (int i = 0; i < 3; ++i) {
                        if (halationStrengthBGR[i] > 0.0f && halationSigmaPx[i] > 0.0f) {
                            if (!JuicerCuda::ResourceManager::command_ensure_halation_kernel(
                                    submissionTxn,
                                    *cudaResources,
                                    cudaResources->halationKernel[i],
                                    halationSigmaPx[i],
                                    _pCudaStream,
                                    opticsError)) {
                                JTRACE("CUDA", std::string("CUDA halation kernel upload failed: ") + opticsError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                                throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                                throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                            }
                            run.halationKernels.halationKernel[i] = cudaResources->halationKernel[i].weights;
                            run.halationKernels.halationRadius[i] = cudaResources->halationKernel[i].radius;
                        }
                        if (halationScatterStrengthBGR[i] > 0.0f && halationScatterSigmaPx[i] > 0.0f) {
                            if (!JuicerCuda::ResourceManager::command_ensure_halation_kernel(
                                    submissionTxn,
                                    *cudaResources,
                                    cudaResources->halationScatterKernel[i],
                                    halationScatterSigmaPx[i],
                                    _pCudaStream,
                                    opticsError)) {
                                JTRACE("CUDA", std::string("CUDA halation scatter kernel upload failed: ") + opticsError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                                throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                                throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                            }
                            run.halationKernels.scatteringKernel[i] = cudaResources->halationScatterKernel[i].weights;
                            run.halationKernels.scatteringRadius[i] = cudaResources->halationScatterKernel[i].radius;
                        }
                    }
                }

                err = juicer_cuda_print_pipeline_optics(
                    &run,
                    cudaResources->scannerScratch.rgbR,
                    cudaResources->scannerScratch.rgbG,
                    cudaResources->scannerScratch.rgbB,
                    cudaResources->scannerScratch.tmp,
                    cudaResources->scannerScratch.blurred,
                    cudaResources->scannerScratch.aux,
                    cudaResources->scannerScratch.grainTmp,
                    cudaResources->scannerScratch.grainTmpShared,
                    cudaResources->scannerScratch.grainTmpMid,
                    cudaResources->scannerScratch.grainTmpCoarse,
                    cudaResources->scannerLensBlurKernel.weights,
                    cudaResources->scannerLensBlurKernel.radius,
                    cudaResources->scannerUnsharpKernel.weights,
                    cudaResources->scannerUnsharpKernel.radius,
                    unsharpAmount,
                    win.x1,
                    win.y1,
                    glareSeed,
                    glarePercent,
                    glareRoughness,
                    cudaResources->scannerGlareKernel.weights,
                    cudaResources->scannerGlareKernel.radius,
                    _pCudaStream);
            }
            if (err != cudaSuccess) {
                const char* msg = cudaGetErrorString(err);
                mark_context_loss_recovery("print_pipeline_kernel_launch", err, msg ? msg : "");
                JTRACE("CUDA", std::string("FATAL: print pipeline kernel launch failed: ") + (msg ? msg : "(unknown)"));
                throw OFX::Exception::Suite(kOfxStatErrFatal);
            }

            if (_effect.abort()) {
                JuicerCuda::record_use(*cudaResources, _pCudaStream);
                return;
            }

            finalize_scan_error_stage(cudaResources, run, stream, scanEvent, "print");

            JuicerCuda::record_use(*cudaResources, _pCudaStream);
        }
        commit_submission_or_throw();
        return;
    }
#endif
}
