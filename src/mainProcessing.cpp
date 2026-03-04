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
    inline float sanitize_nonnegative_or(float value, float fallback);
    inline float sanitize_unit_or(float value, float fallback);
    inline bool is_finite(float value);
    inline bool is_finite(double value);

    inline int pixel_component_count(OFX::PixelComponentEnum comps) {
        switch (comps) {
        case OFX::ePixelComponentRGBA: return 4;
        case OFX::ePixelComponentRGB: return 3;
        case OFX::ePixelComponentAlpha: return 1;
        default: return 0;
        }
    }

    inline int bytes_per_component(OFX::BitDepthEnum depth) {
        switch (depth) {
        case OFX::eBitDepthUByte: return 1;
        case OFX::eBitDepthUShort: return 2;
        case OFX::eBitDepthFloat: return 4;
        default: return 0;
        }
    }

    inline void copy_float3(float dst[3], const float src[3]) {
        std::memcpy(dst, src, 3u * sizeof(float));
    }

    inline void load_float3_from_planar(
        float dst[3],
        const float*& c0,
        const float*& c1,
        const float*& c2) {
        dst[0] = *c0++;
        dst[1] = *c1++;
        dst[2] = *c2++;
    }

    inline void copy_float9(float dst[9], const float src[9]) {
        std::memcpy(dst, src, 9u * sizeof(float));
    }

    inline void copy_float3x3(float dst[3][3], const float src[3][3]) {
        std::memcpy(dst, src, 9u * sizeof(float));
    }

    inline void sanitize_nonnegative_triplet(float values[3]) {
        for (float* valueIt = values; valueIt != values + 3; ++valueIt) {
            *valueIt = sanitize_nonnegative_or(*valueIt, 0.0f);
        }
    }

    inline void sanitize_unit_triplet(float values[3]) {
        for (float* valueIt = values; valueIt != values + 3; ++valueIt) {
            *valueIt = sanitize_unit_or(*valueIt, 0.0f);
        }
    }

    inline void zero_float2(float values[2]) {
        float* valueIt = values;
        for (int i = 0; i < 2; ++i, ++valueIt) {
            *valueIt = 0.0f;
        }
    }

    inline void copy_float2(float dst[2], const float src[2]) {
        std::memcpy(dst, src, 2u * sizeof(float));
    }

    inline float sanitize_amount_0_10(float value) {
        return is_finite(value) ? std::clamp(value, 0.0f, 10.0f) : 0.0f;
    }

    inline float sanitize_nonnegative_or(float value, float fallback) {
        if (!is_finite(value)) {
            return fallback;
        }
        return std::max(0.0f, value);
    }

    inline float sanitize_unit_or(float value, float fallback) {
        if (!is_finite(value)) {
            return fallback;
        }
        return std::clamp(value, 0.0f, 1.0f);
    }

    inline float finite_or_zero(float value) {
        return is_finite(value) ? value : 0.0f;
    }

    inline double finite_or(double value, double fallback) {
        return is_finite(value) ? value : fallback;
    }

    inline double sanitize_finite_clamped_or(double value, double fallback, double lo, double hi) {
        return is_finite(value) ? std::clamp(value, lo, hi) : fallback;
    }

    inline void divide_triplet(float dst[3], const float src[3], float denominator) {
        float* dstIt = dst;
        const float* srcIt = src;
        for (int i = 0; i < 3; ++i, ++dstIt, ++srcIt) {
            *dstIt = *srcIt / denominator;
        }
    }

    inline bool any_positive_triplet(const float values[3]) {
        const float* valueIt = values;
        const float* const valueEnd = values + 3;
        for (; valueIt < valueEnd; ++valueIt) {
            if (*valueIt > 0.0f) {
                return true;
            }
        }
        return false;
    }

    inline bool is_gpu_render_requested(bool openclEnabled, bool cudaEnabled, bool metalEnabled) {
        return openclEnabled || cudaEnabled || metalEnabled;
    }

    inline bool has_grain_defects(
        float filmDustAmount,
        float gateDustAmount,
        float filmScratchAmount,
        float gateScratchAmount) {
        return (filmDustAmount > 0.0f) || (gateDustAmount > 0.0f) ||
            (filmScratchAmount > 0.0f) || (gateScratchAmount > 0.0f);
    }

    inline bool needs_gate_mask_for_defects(float gateDustAmount, float gateScratchAmount) {
        return (gateDustAmount > 0.0f) || (gateScratchAmount > 0.0f);
    }

    inline bool needs_blurred_optics_scratch(bool wantGlareBlur, bool wantGrainBlur, bool wantGrainSublayers) {
        return wantGlareBlur || wantGrainBlur || wantGrainSublayers;
    }

    inline bool needs_independent_grain_path(int debugView, float chromaMix) {
        return (debugView == 0 || debugView == 1) &&
            (is_finite(chromaMix) && chromaMix < 0.999f);
    }

    inline bool is_positive_finite(float value) {
        return is_finite(value) && value > 0.0f;
    }

    inline bool is_positive_finite(double value) {
        return is_finite(value) && value > 0.0;
    }

    inline bool is_nonzero_finite(float value) {
        return is_finite(value) && value != 0.0f;
    }

    inline bool is_finite(float value) {
        return std::isfinite(value);
    }

    inline bool is_finite(double value) {
        return std::isfinite(value);
    }

    inline bool wants_unsharp(float sigmaPx, float amount) {
        return is_positive_finite(sigmaPx) && is_nonzero_finite(amount);
    }

    inline bool needs_grain_shared(bool wantGrain, int debugView, float chromaMix) {
        return wantGrain && needs_independent_grain_path(debugView, chromaMix);
    }

    inline bool wants_optics_stage(
        bool wantLensBlur,
        bool wantUnsharp,
        bool wantGlare,
        bool wantHalation,
        bool wantGrain,
        bool wantWeave,
        bool wantDefects) {
        return wantLensBlur || wantUnsharp || wantGlare || wantHalation || wantGrain || wantWeave || wantDefects;
    }

    struct OpticsScratchNeeds {
        bool blurred = false;
        bool aux = false;
        bool grain = false;
    };

    inline OpticsScratchNeeds build_optics_scratch_needs(
        bool wantGlareBlur,
        bool wantGrainBlur,
        bool wantGrainSublayers,
        bool wantGrainMix) {
        OpticsScratchNeeds needs{};
        needs.blurred = needs_blurred_optics_scratch(wantGlareBlur, wantGrainBlur, wantGrainSublayers);
        needs.aux = wantGrainSublayers;
        needs.grain = wantGrainMix;
        return needs;
    }

    inline bool spatial_dir_enabled(const Couplers::Runtime& dirRT) {
        return dirRT.active && is_positive_finite(dirRT.spatialSigmaPixels);
    }

    inline float sanitize_dir_dmax_value(float value) {
        if (!is_finite(value) || value <= 1e-4f) {
            return 1.0f;
        }
        return value;
    }

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
            outError = "cuCtxGetCurrent failed (code=";
            outError += std::to_string(static_cast<int>(ctxResult));
            outError += ")";
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
        const bool traceInfo = JTRACE_ENABLED(1);
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
            msg += std::to_string(retireAccepted ? 1 : 0);
            msg += " slot_erased=";
            msg += std::to_string(slotErased ? 1 : 0);
            msg += " latch_cleared=";
            msg += std::to_string(latchCleared ? 1 : 0);
            if (!retireError.empty()) {
                msg += " retire_error=";
                msg += retireError;
            }
            JTRACE("MSLCY", msg);
        }
    }
#endif

    std::int64_t frame_index_from_time(double time) {
        return static_cast<std::int64_t>(std::floor(finite_or(time, 0.0)));
    }

    std::uint64_t safe_session_seed(const InstanceState* state) {
        if (state && state->sessionSeed != 0) {
            return state->sessionSeed;
        }
        return 1;
    }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    struct DiagnosticsHookPolicy {
        bool diagnosticsMode = false;
        bool validatePrimitives = false;
        bool runtimeSelfCheck = false;
    };

    bool parse_env_toggle(const char* name, bool fallback) {
        return JuicerLogging::parse_env_int(name, fallback ? 1 : 0) != 0;
    }

    const DiagnosticsHookPolicy& diagnostics_hook_policy() {
        static const DiagnosticsHookPolicy policy = []() {
            DiagnosticsHookPolicy out{};
            out.diagnosticsMode = parse_env_toggle("JUICER_DIAGNOSTICS_MODE", false);
            out.validatePrimitives = parse_env_toggle("JUICER_DIAGNOSTICS_VALIDATE", true);
            out.runtimeSelfCheck = parse_env_toggle("JUICER_DIAGNOSTICS_SELF_CHECK", true);
            return out;
        }();
        return policy;
    }

    void trace_validation_hook_state_once(
        bool compiled,
        bool modeEnabled,
        bool toggleEnabled,
        bool verboseEnabled,
        bool active) {
        static std::once_flag once;
        std::call_once(once, [&]() {
            if (!JTRACE_ENABLED(2)) {
                return;
            }
            const char* reason = "active";
            if (!compiled) {
                reason = "compile_disabled";
            }
            else if (!modeEnabled) {
                reason = "diagnostics_mode_disabled";
            }
            else if (!toggleEnabled) {
                reason = "validation_toggle_disabled";
            }
            else if (!verboseEnabled) {
                reason = "diagnostics_level_below_verbose";
            }
            std::string msg;
            msg.reserve(160);
            msg = "event=diagnostics_hook";
            msg += " hook=validation";
            msg += " mode=";
            msg += (modeEnabled ? "diagnostics" : "serving");
            msg += " compiled=";
            msg += std::to_string(compiled ? 1 : 0);
            msg += " toggle_enabled=";
            msg += std::to_string(toggleEnabled ? 1 : 0);
            msg += " verbose_enabled=";
            msg += std::to_string(verboseEnabled ? 1 : 0);
            msg += " active=";
            msg += std::to_string(active ? 1 : 0);
            msg += " reason=";
            msg += reason;
            JTRACE("MSDBG", msg);
        });
    }

    void trace_self_check_hook_state_once(
        bool compiled,
        bool modeEnabled,
        bool toggleEnabled,
        bool active) {
        static std::once_flag once;
        std::call_once(once, [&]() {
            if (!JTRACE_ENABLED(2)) {
                return;
            }
            const char* reason = "active";
            if (!compiled) {
                reason = "compile_disabled";
            }
            else if (!modeEnabled) {
                reason = "diagnostics_mode_disabled";
            }
            else if (!toggleEnabled) {
                reason = "self_check_toggle_disabled";
            }
            std::string msg;
            msg.reserve(144);
            msg = "event=diagnostics_hook";
            msg += " hook=self_check";
            msg += " mode=";
            msg += (modeEnabled ? "diagnostics" : "serving");
            msg += " compiled=";
            msg += std::to_string(compiled ? 1 : 0);
            msg += " toggle_enabled=";
            msg += std::to_string(toggleEnabled ? 1 : 0);
            msg += " active=";
            msg += std::to_string(active ? 1 : 0);
            msg += " reason=";
            msg += reason;
            JTRACE("MSDBG", msg);
        });
    }
#endif

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
        const float yKey = finite_or_zero(printParams.yFilter);
        const float mKey = finite_or_zero(printParams.mFilter);
        const float cKey = finite_or_zero(printParams.cFilter);
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
        if (!is_positive_finite(kMid)) {
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
        const double* freqIt = freqs;
        for (int i = 0; i < count; ++i, ++freqIt) {
            const double phase = phase_from_seed(sessionSeed, passId, axis, componentOffset + i);
            sum += std::sin(kTwoPi * (*freqIt) * timeSeconds + phase);
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
        if (!(amount > 0.0) || !is_positive_finite(pixelSizeUm)) {
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
        if (!src || !dst) {
            return;
        }
        const OfxRectI bounds = src->getBounds();
        const int xStart = bounds.x1;
        const int xEnd = bounds.x2;
        const int yStart = bounds.y1;
        const int yEnd = bounds.y2;
        const OFX::PixelComponentEnum comps = src->getPixelComponents();
        const OFX::BitDepthEnum depth = src->getPixelDepth();

        const int nComponents = pixel_component_count(comps);
        if (nComponents <= 0) {
            return;
        }
        const int bytesPerComp = bytes_per_component(depth);
        if (bytesPerComp <= 0) {
            return;
        }
        const size_t bytesPerPixel = size_t(nComponents * bytesPerComp);
        const int width = xEnd - xStart;
        if (width <= 0) {
            return;
        }
        if (yStart >= yEnd) {
            return;
        }
        const size_t rowBytes = static_cast<size_t>(width) * bytesPerPixel;
        for (int y = yStart; y < yEnd; ++y) {
            const std::uint8_t* sRow = reinterpret_cast<const std::uint8_t*>(src->getPixelAddress(xStart, y));
            std::uint8_t* dRow = reinterpret_cast<std::uint8_t*>(dst->getPixelAddress(xStart, y));
            if (sRow && dRow) {
                std::memcpy(dRow, sRow, rowBytes);
                continue;
            }
            if (sRow && !dRow) {
                const std::uint8_t* sPix = sRow;
                int x = xStart;
                for (int xOff = 0; xOff < width; ++xOff, ++x, sPix += bytesPerPixel) {
                    void* d = dst->getPixelAddress(x, y);
                    if (!d) continue;
                    std::memcpy(d, sPix, bytesPerPixel);
                }
                continue;
            }
            if (!sRow && dRow) {
                std::uint8_t* dPix = dRow;
                int x = xStart;
                for (int xOff = 0; xOff < width; ++xOff, ++x, dPix += bytesPerPixel) {
                    const void* s = src->getPixelAddress(x, y);
                    if (!s) continue;
                    std::memcpy(dPix, s, bytesPerPixel);
                }
                continue;
            }
            for (int x = xStart; x < xEnd; ++x) {
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
    const std::uint64_t scaledPixels =
        static_cast<std::uint64_t>(std::min(w, 4096u)) * static_cast<std::uint64_t>(h);
    unsigned int nCPUs = static_cast<unsigned int>(scaledPixels / 4096u);
    if (nCPUs == 0) {
        nCPUs = 1;
    }
    static const unsigned int maxThreads = []() -> unsigned int {
        const unsigned int value = OFX::MultiThread::getNumCPUs();
        return (value > 0) ? value : 1u;
    }();
    nCPUs = std::min(nCPUs, maxThreads);
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

static std::uint64_t make_gate_mask_hash(
    std::uint64_t sessionSeed,
    int originX,
    int originY,
    int width,
    int height,
    float pixelSizeUm,
    float gateDustAmount,
    float gateScratchAmount) {
    const std::uint64_t originXBits = static_cast<std::uint64_t>(originX);
    const std::uint64_t originYBits = static_cast<std::uint64_t>(originY);
    const std::uint64_t widthBits = static_cast<std::uint64_t>(width);
    const std::uint64_t heightBits = static_cast<std::uint64_t>(height);
    std::uint64_t h = Hash::kFnvOffset;
    Hash::hash_bytes_update(h, &sessionSeed, sizeof(sessionSeed));
    Hash::hash_bytes_update(h, &originXBits, sizeof(originXBits));
    Hash::hash_bytes_update(h, &originYBits, sizeof(originYBits));
    Hash::hash_bytes_update(h, &widthBits, sizeof(widthBits));
    Hash::hash_bytes_update(h, &heightBits, sizeof(heightBits));
    Hash::hash_bytes_update(h, &pixelSizeUm, sizeof(pixelSizeUm));
    Hash::hash_bytes_update(h, &gateDustAmount, sizeof(gateDustAmount));
    Hash::hash_bytes_update(h, &gateScratchAmount, sizeof(gateScratchAmount));
    if (h == 0) {
        h = 1;
    }
    return h;
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

    const char* label = (mediumLabel && *mediumLabel) ? mediumLabel : "scanner";
    auto set_error = [&](const char* suffix) {
        outError = label;
        outError += suffix;
    };
    if (!runtimeValid) {
        set_error(" scanner runtime invalid");
        return false;
    }
    if (!mediumRuntime) {
        set_error(" scanner medium runtime missing");
        return false;
    }

    const Spectral::SpectralTables* tables = mediumRuntime->tables;
    if (!tables || tables->K <= 0) {
        set_error(" scanner spectral tables unavailable");
        return false;
    }

    Scanner::ScannerStaticKey staticKey = mediumRuntime->staticKey;
    if (tables->tablesHash != staticKey.tablesHash) {
        set_error(" scanner tables hash mismatch for medium");
        return false;
    }
    if (mediumRuntime->range.digest == 0) {
        set_error(" scanner density range missing or invalid");
        return false;
    }

    const std::uint64_t illumHash = tables->illuminantHash;
    if (illumHash != 0 && mediumRuntime->illuminant.hash != 0 && illumHash != mediumRuntime->illuminant.hash) {
        set_error(" scanner illuminant hash mismatch for medium");
        return false;
    }

    const Scanner::ColorRuntime* colorPtr = mediumRuntime->color;
    if (!colorPtr || colorPtr->hash == 0) {
        set_error(" scanner color runtime missing or invalid");
        return false;
    }
    if (staticKey.colorRuntimeHash != colorPtr->hash) {
        set_error(" scanner static key color hash mismatch");
        return false;
    }

    Scanner::finalize_static_key(staticKey);
    if (staticKey.hash == 0) {
        set_error(" scanner static key missing or invalid");
        return false;
    }

    out.mediumRuntime = mediumRuntime;
    out.colorRuntime = colorPtr;
    out.staticKey = staticKey;
    return true;
}

bool curve_ok(const Spectral::Curve& c) {
    const size_t N = c.lambda_nm.size();
    if (N < 2 || c.linear.size() != N) return false;
    const float* lambdaData = c.lambda_nm.data();
    float prev = lambdaData[0];
    if (!is_finite(prev)) return false;
    for (size_t i = 1; i < N; ++i) {
        const float xi = lambdaData[i];
        if (!is_finite(xi)) return false;
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
        const float* srcDMax = _ws->dMax;
        float* dstDMax = _dirRT.dMax;
        for (int i = 0; i < 3; ++i, ++srcDMax, ++dstDMax) {
            *dstDMax = sanitize_dir_dmax_value(*srcDMax);
        }
    }
}
void JuicerProcessor::setPrintRuntime(const Print::Runtime* prt, bool printReady) { _prt = prt; _printReady = printReady; }
void JuicerProcessor::setExposure(float exposureScale) {
    _exposureScale = is_positive_finite(exposureScale) ? exposureScale : 1.0f;
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
    _gateWeaveAmount = finite_or(amount, _gateWeaveAmount);
}

void JuicerProcessor::setFrameTime(double time) {
    _timeFrames = finite_or(time, 0.0);
    _frameIndex = frame_index_from_time(_timeFrames);
    _frameTimeHash = Hash::hash_bytes(&_frameIndex, sizeof(_frameIndex));
    if (_frameTimeHash == 0) {
        _frameTimeHash = 1;
    }
}

void JuicerProcessor::setFrameRate(double frameRate) {
    if (is_positive_finite(frameRate)) {
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
    _pixelSizeUm = sanitize_nonnegative_or(pixelSizeUm, 0.0f);
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
    ctx.exposureScaleSafe = is_positive_finite(_exposureScale) ? _exposureScale : 1.0f;
    ctx.useSpatialDIR = (spatial_dir_enabled(_dirRT) &&
        _nComponents >= 3 && _wsReady && _ws);
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

    ctx.pixelSizeUm = is_positive_finite(_pixelSizeUm) ? _pixelSizeUm : 0.0f;
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
            int srcComponents = 0;
            size_t srcStride = 0;
            bool canReadRowRgb = false;
            int cachedY = std::numeric_limits<int>::min();
            const float* cachedRow = nullptr;
        };
        SpatialDIRUser user{};
        user.effect = &_effect;
        user.srcImg = _srcImg;
        user.window = ctx.window;
        user.srcComponents = _nComponents;
        user.srcStride = static_cast<size_t>(std::max(_nComponents, 0));
        user.canReadRowRgb = (_nComponents >= 3);

        SpatialDIR::Callbacks callbacks{};
        callbacks.user = &user;
        callbacks.fetchRGB = [](void* u, int xx, int yy, float rgb[3]) -> bool {
            auto* self = static_cast<SpatialDIRUser*>(u);
            const int y = self->window.y1 + yy;
            if (self->cachedY != y) {
                self->cachedY = y;
                self->cachedRow = reinterpret_cast<const float*>(self->srcImg->getPixelAddress(self->window.x1, y));
            }
            const float* srcPix = nullptr;
            if (self->cachedRow && self->canReadRowRgb) {
                const size_t xOffset = static_cast<size_t>(xx);
                srcPix = self->cachedRow + xOffset * self->srcStride;
            }
            else {
                const int x = self->window.x1 + xx;
                srcPix = reinterpret_cast<const float*>(self->srcImg->getPixelAddress(x, y));
            }
            if (!srcPix) return false;
            copy_float3(rgb, srcPix);
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
    if (ctx.printActive && _scratch.printScratchPerWorker.size() != nThreads) {
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
            const int yEnd = std::min(height, yStart + rowsPerThread);
            const bool printActive = ctx.printActive;
            const bool useSpatialDIR = ctx.useSpatialDIR;

            JuicerProc::PrintPipelineScratch* printScratch = nullptr;
            if (printActive && threadId < self._scratch.printScratchPerWorker.size()) {
                printScratch = &self._scratch.printScratchPerWorker[threadId];
            }

            const auto& dirWorkspace = self._scratch.dirWorkspace;
            const float* filmRawB = useSpatialDIR ? dirWorkspace.filmRaw_B.data() : nullptr;
            const float* filmRawG = useSpatialDIR ? dirWorkspace.filmRaw_G.data() : nullptr;
            const float* filmRawR = useSpatialDIR ? dirWorkspace.filmRaw_R.data() : nullptr;
            const float* corrY = useSpatialDIR ? dirWorkspace.corrYBlur.data() : nullptr;
            const float* corrM = useSpatialDIR ? dirWorkspace.corrMBlur.data() : nullptr;
            const float* corrC = useSpatialDIR ? dirWorkspace.corrCBlur.data() : nullptr;

            Pipeline::DensityPixelInputs pxIn{};
            pxIn.exposureScale = ctx.exposureScaleSafe;
            pxIn.dirRuntime = &self._dirRT;
            pxIn.applyDirRuntime = true;
            if (printActive) {
                pxIn.printRuntime = self._prt;
                pxIn.printParams = &self._printParams;
                pxIn.midgrayFactor = ctx.kMidSpectral;
                pxIn.printScratch = printScratch;
            }
            if (useSpatialDIR) {
                pxIn.useFilmRawOverride = true;
                pxIn.useSpatialDIR = true;
            }
            Pipeline::DensityPixelOutputs pxOut{};
            const WorkingState& wsRef = *self._ws;
            float* densityC = self._density.c.data();
            float* densityM = self._density.m.data();
            float* densityY = self._density.y.data();
            const size_t srcStride = static_cast<size_t>(self._nComponents);
            auto run_and_store_density = [&](size_t idx) -> bool {
                if (!runner.run_density_pixel(wsRef, pxIn, pxOut)) {
                    if (printActive) {
                        failure.store(true, std::memory_order_relaxed);
                        abortFlag.store(true, std::memory_order_relaxed);
                        return false;
                    }
                    densityC[idx] = 0.0f;
                    densityM[idx] = 0.0f;
                    densityY[idx] = 0.0f;
                    return true;
                }

                if (printActive) {
                    if (pxOut.medium != Pipeline::DensityMedium::Print) {
                        failure.store(true, std::memory_order_relaxed);
                        abortFlag.store(true, std::memory_order_relaxed);
                        return false;
                    }
                    densityC[idx] = pxOut.printDensity.v[0];
                    densityM[idx] = pxOut.printDensity.v[1];
                    densityY[idx] = pxOut.printDensity.v[2];
                    return true;
                }

                densityC[idx] = pxOut.negativeDensity.v[0];
                densityM[idx] = pxOut.negativeDensity.v[1];
                densityY[idx] = pxOut.negativeDensity.v[2];
                return true;
            };

            for (int yOff = yStart; yOff < yEnd && !abortFlag.load(std::memory_order_relaxed); ++yOff) {
                if (self._effect.abort()) {
                    abortFlag.store(true, std::memory_order_relaxed);
                    break;
                }
                const int y = originY + yOff;
                const size_t rowOffset = size_t(yOff) * size_t(width);
                if (useSpatialDIR) {
                    size_t idx = rowOffset;
                    const float* filmRawBIt = filmRawB + rowOffset;
                    const float* filmRawGIt = filmRawG + rowOffset;
                    const float* filmRawRIt = filmRawR + rowOffset;
                    const float* corrYIt = corrY + rowOffset;
                    const float* corrMIt = corrM + rowOffset;
                    const float* corrCIt = corrC + rowOffset;
                    for (int xOff = 0; xOff < width; ++xOff, ++idx) {
                        if (abortFlag.load(std::memory_order_relaxed)) {
                            break;
                        }
                        load_float3_from_planar(pxIn.filmRawOverride.v, filmRawBIt, filmRawGIt, filmRawRIt);
                        load_float3_from_planar(pxIn.spatialLogECorrectionsYMC, corrYIt, corrMIt, corrCIt);
                        if (!run_and_store_density(idx)) {
                            break;
                        }
                    }
                    continue;
                }

                const float* srcRow = reinterpret_cast<const float*>(self._srcImg->getPixelAddress(originX, y));
                if (srcRow) {
                    const float* srcPixIt = srcRow;
                    size_t idx = rowOffset;
                    for (int xOff = 0; xOff < width; ++xOff, ++idx) {
                        if (abortFlag.load(std::memory_order_relaxed)) {
                            break;
                        }
                        const float* srcPix = srcPixIt;
                        srcPixIt += srcStride;
                        copy_float3(pxIn.rgb.v, srcPix);
                        if (!run_and_store_density(idx)) {
                            break;
                        }
                    }
                    continue;
                }

                size_t idx = rowOffset;
                int x = originX;
                for (int xOff = 0; xOff < width; ++xOff, ++idx, ++x) {
                    if (abortFlag.load(std::memory_order_relaxed)) {
                        break;
                    }
                    const float* srcPix =
                        reinterpret_cast<const float*>(self._srcImg->getPixelAddress(x, y));
                    if (!srcPix) {
                        densityC[idx] = 0.0f;
                        densityM[idx] = 0.0f;
                        densityY[idx] = 0.0f;
                        continue;
                    }
                    copy_float3(pxIn.rgb.v, srcPix);
                    if (!run_and_store_density(idx)) {
                        break;
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
    const bool traceInfo = JTRACE_ENABLED(1);
    const bool traceVerbose = JTRACE_ENABLED(3);

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
    const char* cpuMediumLabel = ctx.printActive ? "print" : "negative";
    if (!validate_scanner_preflight_runtime(
            scannerRuntimeValid,
            cpuMediumLabel,
            mediumRuntime,
            scannerPreflight,
            scannerPreflightError)) {
        if (traceInfo) {
            std::string preflightMsg;
            preflightMsg.reserve(96 + scannerPreflightError.size());
            preflightMsg = "path=cpu result=fail medium=";
            preflightMsg += cpuMediumLabel;
            preflightMsg += " reason=";
            preflightMsg += scannerPreflightError;
            JTRACE("MSSKV", preflightMsg);
            std::string fatalMsg;
            fatalMsg.reserve(8 + scannerPreflightError.size());
            fatalMsg = "FATAL: ";
            fatalMsg += scannerPreflightError;
            JTRACE("SCAN", fatalMsg);
        }
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
    if (traceVerbose) {
        std::string preflightMsg;
        preflightMsg.reserve(96);
        preflightMsg = "path=cpu result=ok medium=";
        preflightMsg += cpuMediumLabel;
        preflightMsg += " static_key_hash=";
        preflightMsg += std::to_string(scannerPreflight.staticKey.hash);
        JTRACE_VERBOSE("MSSKV", preflightMsg);
    }

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
    const auto leaseDeadline = leaseWaitStart + std::chrono::microseconds(kScannerRuntimeLeaseMaxWaitUs);
    while (!opticsRuntime) {
        waitedForLease = true;
        if (_effect.abort()) {
            JTRACE_VERBOSE("MSSRL", "event=runtime_lease outcome=abort");
            return;
        }
        if (std::chrono::steady_clock::now() >= leaseDeadline) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(kScannerRuntimeLeasePollSleepUs));
        opticsRuntime = runtimeLease.acquire();
    }
    const std::uint64_t leaseWaitUs = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - leaseWaitStart).count());
    if (!opticsRuntime) {
        if (traceInfo) {
            std::string msg;
            msg.reserve(128);
            msg = "event=runtime_lease outcome=timeout";
            msg += " wait_us=";
            msg += std::to_string(leaseWaitUs);
            msg += " wait_budget_us=";
            msg += std::to_string(static_cast<unsigned long long>(kScannerRuntimeLeaseMaxWaitUs));
            JTRACE("MSSRL", msg);
        }
        JTRACE("SCAN", "FATAL: scanner runtime lease unavailable after bounded wait");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
    if (traceVerbose) {
        std::string msg;
        msg.reserve(96);
        msg = "event=runtime_lease outcome=";
        msg += (waitedForLease ? "wait_acquired" : "acquired");
        msg += " wait_us=";
        msg += std::to_string(leaseWaitUs);
        msg += " slot=";
        msg += runtimeLease.slotName;
        JTRACE_VERBOSE("MSSRL", msg);
    }

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
    const unsigned int scannerThreadCount = std::max(1u, threadCount);
    optCtx.threadCount = scannerThreadCount;
    optCtx.abort.shouldAbort = [this]() -> bool { return _effect.abort(); };

    ScannerOptics::render_density_to_rgb(optCtx);
}

void JuicerProcessor::processImpl() {
    if (!_srcImg || !_dstImg) return;

    if (is_gpu_render_requested(_isEnabledOpenCLRender, _isEnabledCudaRender, _isEnabledMetalRender)) {
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
    const int xStart = _renderWindow.x1;
    const int xEnd = _renderWindow.x2;
    const int yStart = _renderWindow.y1;
    const int yEnd = _renderWindow.y2;

    if (_nComponents < 3) {
        if (_nComponents != 1) {
            return;
        }
        const int width = xEnd - xStart;
        if (width <= 0) {
            return;
        }
        const size_t rowBytes = static_cast<size_t>(width) * sizeof(float);
        for (int y = yStart; y < yEnd; ++y) {
            float* dstRow = reinterpret_cast<float*>(_dstImg->getPixelAddress(xStart, y));
            const float* srcRow = reinterpret_cast<const float*>(_srcImg->getPixelAddress(xStart, y));
            if (dstRow && srcRow) {
                std::memcpy(dstRow, srcRow, rowBytes);
                continue;
            }
            if (dstRow && !srcRow) {
                float* dstPixIt = dstRow;
                int x = xStart;
                for (int xOff = 0; xOff < width; ++xOff, ++x, ++dstPixIt) {
                    const float* srcPix = reinterpret_cast<const float*>(_srcImg->getPixelAddress(x, y));
                    if (!srcPix) {
                        continue;
                    }
                    *dstPixIt = *srcPix;
                }
                continue;
            }
            if (!dstRow && srcRow) {
                const float* srcPixIt = srcRow;
                int x = xStart;
                for (int xOff = 0; xOff < width; ++xOff, ++x, ++srcPixIt) {
                    float* dstPix = reinterpret_cast<float*>(_dstImg->getPixelAddress(x, y));
                    if (!dstPix) {
                        continue;
                    }
                    *dstPix = *srcPixIt;
                }
                continue;
            }
            for (int x = xStart; x < xEnd; ++x) {
                float* dstPix = reinterpret_cast<float*>(_dstImg->getPixelAddress(x, y));
                const float* srcPix = reinterpret_cast<const float*>(_srcImg->getPixelAddress(x, y));
                if (!dstPix || !srcPix) {
                    continue;
                }
                *dstPix = *srcPix;
            }
        }
        return;
    }

#if defined(JUICER_SPD_DEBUG) && (JUICER_SPD_DEBUG != 0)
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
    if (is_gpu_render_requested(_isEnabledOpenCLRender, _isEnabledCudaRender, _isEnabledMetalRender)) {
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
    const bool traceInfo = JTRACE_ENABLED(1);
    const bool traceVerbose = JTRACE_ENABLED(3);

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
            if (traceInfo) {
                const char* msg = cudaGetErrorString(setErr);
                std::string traceMsg;
                traceMsg.reserve(64);
                traceMsg = "FATAL: cudaSetDevice failed: ";
                traceMsg += (msg ? msg : "(unknown)");
                JTRACE("CUDA", traceMsg);
            }
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }

        std::string contextError;
        if (!query_current_cuda_context(contextOpaque, contextError)) {
            if (traceInfo) {
                std::string traceMsg;
                traceMsg.reserve(72 + contextError.size());
                traceMsg = "FATAL: failed to capture CUDA context identity: ";
                traceMsg += contextError;
                JTRACE("CUDA", traceMsg);
            }
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
        if (slot->deviceId < 0) {
            slot->deviceId = deviceContextKey.deviceId;
        }
        if (!slot->ownerContextOpaque) {
            slot->ownerContextOpaque = deviceContextKey.contextOpaque;
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
        if (traceVerbose) {
            std::string msg;
            msg.reserve(128);
            msg = "path=cuda action=";
            msg += (reusingSnapshotLatch ? "reuse" : "new");
            msg += " frame_token=";
            msg += std::to_string(snapshot.frameToken.value);
            msg += " snapshot_id=";
            msg += std::to_string(snapshot.snapshotId);
            msg += " instance_token=";
            msg += std::to_string(snapshot.instanceToken.value);
            JTRACE_VERBOSE("MSSNP", msg);
        }

        std::string submissionError;
        if (!JuicerCuda::ResourceManager::begin_submission(submissionTxn, snapshot, submissionError)) {
            mark_context_loss_recovery("begin_submission", cudaErrorUnknown, submissionError);
            if (traceInfo) {
                std::string msg;
                msg.reserve(40 + submissionError.size());
                msg = "FATAL: begin_submission failed: ";
                msg += submissionError;
                JTRACE("CUDA", msg);
            }
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        if (!JuicerCuda::ResourceManager::acquire_plan(submissionTxn, submissionError)) {
            mark_context_loss_recovery("acquire_plan", cudaErrorUnknown, submissionError);
            if (traceInfo) {
                std::string msg;
                msg.reserve(34 + submissionError.size());
                msg = "FATAL: acquire_plan failed: ";
                msg += submissionError;
                JTRACE("CUDA", msg);
            }
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
            _scannerSettings.useLut,
            _pCudaStream,
            uploadError)) {
        mark_context_loss_recovery("command_ensure_uploaded", cudaErrorUnknown, uploadError);
        if (traceInfo) {
            std::string msg;
            msg.reserve(40 + uploadError.size());
            msg = "CUDA WorkingState upload failed: ";
            msg += uploadError;
            JTRACE("CUDA", msg);
        }
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
        throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
        throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
    }
    if (traceVerbose) {
        std::lock_guard<std::mutex> resLock(cudaResources->m);
        std::string msg;
        msg.reserve(192);
        msg = "cuda upload build=";
        msg += std::to_string(_ws->buildCounter);
        msg += " uploaded=";
        msg += std::to_string(cudaResources->uploadedBuildCounter);
        msg += " printIllumBuild=";
        msg += std::to_string(cudaResources->printIllumBuildCounter);
        msg += " printPreflashBuild=";
        msg += std::to_string(cudaResources->printPreflashBuildCounter);
        JTRACE_VERBOSE("PRINTDBG", msg);
    }

    if (_effect.abort()) {
        if (cudaResources) {
            JuicerCuda::record_use(*cudaResources, _pCudaStream);
        }
        return;
    }

#if defined(JUICER_CUDA_VALIDATE_PRIMITIVES) && (JUICER_CUDA_VALIDATE_PRIMITIVES != 0)
    {
        const DiagnosticsHookPolicy& diagnosticsPolicy = diagnostics_hook_policy();
        const bool validationHookActive =
            diagnosticsPolicy.diagnosticsMode &&
            diagnosticsPolicy.validatePrimitives &&
            traceVerbose;
        trace_validation_hook_state_once(
            true,
            diagnosticsPolicy.diagnosticsMode,
            diagnosticsPolicy.validatePrimitives,
            traceVerbose,
            validationHookActive);
        if (validationHookActive) {
            std::string validateError;
            if (!cudaResources) {
                JTRACE("CUDA", "FATAL: CUDA resources missing for validation");
                throw OFX::Exception::Suite(kOfxStatErrFatal);
            }
            if (!JuicerCuda::validate_density_primitives(*cudaResources, *_ws, _pCudaStream, validateError)) {
                std::string msg;
                msg.reserve(48 + validateError.size());
                msg = "FATAL: CUDA primitive validation failed: ";
                msg += validateError;
                JTRACE("CUDA", msg);
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
                    std::string msg;
                    msg.reserve(44 + validateError.size());
                    msg = "FATAL: CUDA print validation failed: ";
                    msg += validateError;
                    JTRACE("CUDA", msg);
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
                }
            }

            JuicerCuda::record_use(*cudaResources, _pCudaStream);
        }
    }
#endif

#if defined(JUICER_CUDA_SELF_CHECK) && (JUICER_CUDA_SELF_CHECK != 0)
    {
        const DiagnosticsHookPolicy& diagnosticsPolicy = diagnostics_hook_policy();
        const bool selfCheckHookActive =
            diagnosticsPolicy.diagnosticsMode &&
            diagnosticsPolicy.runtimeSelfCheck;
        trace_self_check_hook_state_once(
            true,
            diagnosticsPolicy.diagnosticsMode,
            diagnosticsPolicy.runtimeSelfCheck,
            selfCheckHookActive);
        if (selfCheckHookActive) {
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
                    if (traceInfo) {
                        std::string msg;
                        msg.reserve(96);
                        msg = "CUDA self-check failed; forcing CPU fallback. Error: ";
                        msg += (sSelfCheckErr ? sSelfCheckErr : "(unknown)");
                        JTRACE("CUDA", msg);
                    }
                }
                else {
                    JTRACE("CUDA", "CUDA self-check passed");
                }
            });
            if (!sSelfCheckOk) {
                OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
            }
        }
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
            if (traceInfo) {
                std::string traceMsg;
                traceMsg.reserve(64);
                traceMsg = "FATAL: cudaMemcpy2DAsync failed: ";
                traceMsg += (msg ? msg : "(unknown)");
                JTRACE("CUDA", traceMsg);
            }
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
            if (traceInfo) {
                std::string msg;
                msg.reserve(56 + aeError.size());
                msg = "CUDA auto-exposure buffer allocation failed: ";
                msg += aeError;
                JTRACE("CUDA", msg);
            }
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
        const double timeFrames = finite_or(_timeFrames, 0.0);
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
            if (!(is_finite(a) && is_finite(b))) {
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
                        if (traceInfo) {
                            std::string msg;
                            msg.reserve(128);
                            msg = "CUDA auto-exposure weight build failed: ";
                            msg += (errMsg ? errMsg : "(unknown)");
                            JTRACE("CUDA", msg);
                        }
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
                if (traceInfo) {
                    std::string msg;
                    msg.reserve(128);
                    msg = "CUDA auto-exposure metering failed: ";
                    msg += (errMsg ? errMsg : "(unknown)");
                    JTRACE("CUDA", msg);
                }
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
                if (traceInfo) {
                    std::string msg;
                    msg.reserve(128);
                    msg = "CUDA auto-exposure slider update failed: ";
                    msg += (errMsg ? errMsg : "(unknown)");
                    JTRACE("CUDA", msg);
                }
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

    struct HalationSetupResult {
        float strengthBGR[3] = { 0.0f, 0.0f, 0.0f };
        float scatterStrengthBGR[3] = { 0.0f, 0.0f, 0.0f };
        float sigmaPx[3] = { 0.0f, 0.0f, 0.0f };
        float scatterSigmaPx[3] = { 0.0f, 0.0f, 0.0f };
        bool wantHalation = false;
    };

    auto setup_halation_payload = [&](const Profiles::HalationMetadata& halationUi) -> HalationSetupResult {
        HalationSetupResult result{};

        const float strengthBGR[3] = {
            halationUi.strength[2],
            halationUi.strength[1],
            halationUi.strength[0]
        };
        const float scatterStrengthBGR[3] = {
            halationUi.scatteringStrength[2],
            halationUi.scatteringStrength[1],
            halationUi.scatteringStrength[0]
        };
        const float sizeBGR[3] = {
            halationUi.sizeUm[2],
            halationUi.sizeUm[1],
            halationUi.sizeUm[0]
        };
        const float scatterSizeBGR[3] = {
            halationUi.scatteringSizeUm[2],
            halationUi.scatteringSizeUm[1],
            halationUi.scatteringSizeUm[0]
        };

        copy_float3(result.strengthBGR, strengthBGR);
        copy_float3(result.scatterStrengthBGR, scatterStrengthBGR);
        copy_float3(result.sigmaPx, sizeBGR);
        copy_float3(result.scatterSigmaPx, scatterSizeBGR);
        sanitize_nonnegative_triplet(result.strengthBGR);
        sanitize_nonnegative_triplet(result.scatterStrengthBGR);
        sanitize_nonnegative_triplet(result.sigmaPx);
        sanitize_nonnegative_triplet(result.scatterSigmaPx);
        const bool hasPixelSize = is_positive_finite(_pixelSizeUm);

        if (hasPixelSize) {
            divide_triplet(result.sigmaPx, result.sigmaPx, _pixelSizeUm);
            divide_triplet(result.scatterSigmaPx, result.scatterSigmaPx, _pixelSizeUm);
        }

        result.wantHalation = halationUi.active &&
            (any_positive_triplet(result.strengthBGR) || any_positive_triplet(result.scatterStrengthBGR)) &&
            hasPixelSize;
        return result;
    };

    auto setup_grain_payload = [&](JuicerCuda::PipelineRunParams& run,
                                   const Profiles::GrainMetadata& grainUi,
                                   bool includeDefects) -> GrainSetupResult {
        GrainSetupResult result{};

        auto nanmax_vector = [](const std::vector<float>& values, float& outMax) -> bool {
            double m = -std::numeric_limits<double>::infinity();
            bool found = false;
            const float* data = values.data();
            const float* const dataEnd = data + values.size();
            for (; data < dataEnd; ++data) {
                const float v = *data;
                if (is_finite(v)) {
                    m = std::max(m, static_cast<double>(v));
                    found = true;
                }
            }
            if (!found || !is_finite(m)) {
                return false;
            }
            outMax = static_cast<float>(m);
            return is_finite(outMax);
        };
        auto nanmax_curve = [&](const Spectral::Curve& curve, float& outMax) -> bool {
            return nanmax_vector(curve.linear, outMax);
        };

        bool wantGrain = grainUi.active && is_positive_finite(_pixelSizeUm);
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
            const double fps = is_positive_finite(_frameRate) ? _frameRate : 24.0;
            const double timeFrames = finite_or(_timeFrames, static_cast<double>(_frameIndex));
            const double alphaFrames = timeFrames - static_cast<double>(_frameIndex);
            const float timeAlpha = static_cast<float>(std::clamp(alphaFrames, 0.0, 1.0));
            const double timeSeconds = timeFrames / fps;
            const double weaveAmount = sanitize_finite_clamped_or(_gateWeaveAmount, 0.0, 0.0, 10.0);
            const bool hasPixelSize = is_positive_finite(_pixelSizeUm);
            const GateWeaveSignal weave = compute_gate_weave(
                sessionSeed,
                timeSeconds,
                6.0,
                0.005,
                static_cast<double>(_pixelSizeUm),
                weaveAmount);
            const double debugScalePx = hasPixelSize
                ? (4.0 * 6.0 * weaveAmount / static_cast<double>(_pixelSizeUm))
                : 1.0;
            const int breathingPeriodFrames = std::max(1, static_cast<int>(std::llround(fps * 2.5)));
            const double clumpPeriodSec = sanitize_finite_clamped_or(
                static_cast<double>(grainUi.clumpMorphPeriodSec),
                25.0,
                5.0,
                60.0);
            const int clumpMorphPeriodFrames = std::max(1, static_cast<int>(std::llround(fps * clumpPeriodSec)));
            const double longEdgePx = static_cast<double>(std::max(width, height));
            const double filmFormatMm = (hasPixelSize && longEdgePx > 0.0)
                ? (static_cast<double>(_pixelSizeUm) * longEdgePx / 1000.0)
                : 0.0;
            const double pitchMm = is_positive_finite(filmFormatMm)
                ? (filmFormatMm * static_cast<double>(height) / longEdgePx)
                : 0.0;
            const int pitchPx = (hasPixelSize && is_positive_finite(pitchMm))
                ? static_cast<int>(std::llround(pitchMm * 1000.0 / static_cast<double>(_pixelSizeUm)))
                : height;
            const double filmScale = is_positive_finite(filmFormatMm) ? (filmFormatMm / 10.0) : 1.0;
            run.grain.seedBase = make_seed_base(_clipToken, _frameIndex, sessionSeed, kSeedPassGrain);
            run.grain.seedBaseNext = make_seed_base(_clipToken, _frameIndex + 1, sessionSeed, kSeedPassGrain);
            run.grain.frameIndex = _frameIndex;
            run.grain.stbnSessionSeed = sessionSeed;
            run.grain.clipToken = static_cast<std::uint64_t>(_clipToken);
            run.grain.timeAlpha = timeAlpha;
            run.gateWeave.active = (weaveAmount > 0.0 && hasPixelSize) ? 1 : 0;
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
            run.grain.clumpTemporalMix = static_cast<float>(
                sanitize_finite_clamped_or(static_cast<double>(grainUi.clumpTemporalMix), 0.0, 0.0, 0.30));
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
            float densityMin[3];
            float uniformity[3];
            copy_float3(densityMin, grainUi.densityMin.data());
            copy_float3(uniformity, grainUi.uniformity.data());
            sanitize_nonnegative_triplet(densityMin);
            sanitize_unit_triplet(uniformity);

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
            if (!is_positive_finite(pixelAreaUm2)) {
                wantGrain = false;
            }

            const int nSubLayers = (grainUi.nSubLayers > 0) ? grainUi.nSubLayers : 1;
            run.grain.nSubLayers = nSubLayers;
            run.grain.originX = win.x1;
            run.grain.originY = win.y1;
            run.grain.pixelSizeUm = static_cast<float>(_pixelSizeUm);
            run.grain.blurSigmaPx = sanitize_nonnegative_or(grainUi.blur, 0.0f);
            run.grain.blurDyeCloudsUm = sanitize_nonnegative_or(grainUi.blurDyeCloudsUm, 0.0f);
            run.grain.sizeMixWeight = sanitize_unit_or(grainUi.sizeMixWeight, 0.0f);
            run.grain.sizeMixWeightMid = sanitize_unit_or(grainUi.sizeMixWeightMid, 0.0f);
            run.grain.sizeMixScale = std::max(1.0f, sanitize_nonnegative_or(grainUi.sizeMixScale, 1.0f));
            run.grain.breathingDebug = grainUi.breathingDebug ? 1 : 0;
            run.grain.debugView = std::clamp(grainUi.debugView, 0, 6);
            run.grain.amplitude = sanitize_nonnegative_or(grainUi.amplitude, 1.0f);
            const float chromaMix = sanitize_unit_or(grainUi.chroma, 1.0f);
            run.grain.chromaMix = chromaMix;
            run.grain.chromaSharedWeight = std::sqrt(std::max(0.0f, 1.0f - chromaMix));
            run.grain.chromaIndWeight = std::sqrt(std::max(0.0f, chromaMix));
            copy_float2(run.grain.microStructure, grainUi.microStructure.data());
            if (includeDefects) {
                run.grain.filmDustAmount = sanitize_amount_0_10(grainUi.filmDustAmount);
                run.grain.gateDustAmount = sanitize_amount_0_10(grainUi.gateDustAmount);
                run.grain.filmScratchAmount = sanitize_amount_0_10(grainUi.filmScratchAmount);
                run.grain.gateScratchAmount = sanitize_amount_0_10(grainUi.gateScratchAmount);
            }
            copy_float3(run.grain.densityMin, densityMin);
            copy_float3(run.grain.uniformity, uniformity);

            bool paramsOk = wantGrain;
            float blurAreaRatio = 1.0f;
            float blurRatioSum = 0.0f;
            int blurRatioCount = 0;
            if (paramsOk) {
                constexpr float kDefaultParticleAreaUm2 = 0.335f;
                constexpr float kDefaultParticleScale[3] = { 1.10f, 1.27f, 2.08f };
                const float densityMaxCurves[3] = { maxC, maxM, maxY };
                const float* densityMaxCurveIt = densityMaxCurves;
                const float* densityMinIt = densityMin;
                const float* grainScaleIt = grainUi.agxParticleScale.data();
                const float* defaultScaleIt = kDefaultParticleScale;
                float* densityMaxOutIt = run.grain.densityMax;
                float* nParticlesOutIt = run.grain.nParticles;
                float* odParticleOutIt = run.grain.odParticle;
                for (int i = 0; i < 3; ++i,
                     ++densityMaxCurveIt, ++densityMinIt, ++grainScaleIt, ++defaultScaleIt,
                     ++densityMaxOutIt, ++nParticlesOutIt, ++odParticleOutIt) {
                    const float densityMax = *densityMaxCurveIt + *densityMinIt;
                    const float particleArea = grainUi.agxParticleAreaUm2 * (*grainScaleIt);
                    if (!is_positive_finite(particleArea)) {
                        paramsOk = false;
                        break;
                    }
                    const float particleAreaRef = kDefaultParticleAreaUm2 * (*defaultScaleIt);
                    if (is_positive_finite(particleAreaRef)) {
                        blurRatioSum += particleArea / particleAreaRef;
                        blurRatioCount += 1;
                    }
                    float nParticles = pixelAreaUm2 / particleArea;
                    if (nSubLayers > 1) {
                        nParticles /= static_cast<float>(nSubLayers);
                    }
                    if (!is_positive_finite(nParticles)) {
                        paramsOk = false;
                        break;
                    }
                    const float odParticle = densityMax / nParticles;
                    *densityMaxOutIt = densityMax;
                    *nParticlesOutIt = nParticles;
                    *odParticleOutIt = finite_or_zero(odParticle);
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
                if (is_positive_finite(grainBlurSigmaPx)) {
                    grainBlurSigmaPx *= std::sqrt(std::max(blurAreaRatio, 0.0f));
                }

                if (!is_positive_finite(run.grain.microStructure[1])) {
                    zero_float2(run.grain.microStructure);
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
                            if (!is_positive_finite(total)) {
                                layersOk = false;
                                break;
                            }

                            for (int layer = 0; layer < 3; ++layer) {
                                const float fraction = densityMaxLayers[layer][ch] / total;
                                const float minLayer = fraction * densityMin[ch];
                                const float maxLayer = densityMaxLayers[layer][ch] + minLayer;
                                const float particleAreaLayer = grainUi.agxParticleAreaUm2 * grainUi.agxParticleScale[ch] * grainUi.agxParticleScaleLayers[layer];
                                if (!is_positive_finite(particleAreaLayer)) {
                                    layersOk = false;
                                    break;
                                }
                                const float nParticlesLayer = pixelAreaUm2 * fraction / particleAreaLayer;
                                const float odParticle = (nParticlesLayer > 0.0f) ? (maxLayer / nParticlesLayer) : 0.0f;
                                run.grain.densityMinLayers[layer][ch] = minLayer;
                                run.grain.densityMaxLayers[layer][ch] = maxLayer;
                                run.grain.nParticlesLayers[layer][ch] = finite_or_zero(nParticlesLayer);
                                run.grain.odParticleLayers[layer][ch] = finite_or_zero(odParticle);
                                run.grain.densityCurvesLayers[layer][ch] = cudaResources->densityCurvesLayers[layer][ch];
                                const float dyeSigma = run.grain.blurDyeCloudsUm * std::sqrt(std::max(0.0f, run.grain.odParticleLayers[layer][ch]));
                                grainDyeSigmaPx[layer][ch] = finite_or_zero(dyeSigma);
                            }
                            if (!layersOk) {
                                break;
                            }
                        }
                    }
                    wantGrainSublayers = layersOk;
                }
            }
            if (is_positive_finite(grainBlurSigmaPx)) {
                wantGrainBlur = wantGrainSublayers ? (grainBlurSigmaPx > 0.0f) : (grainBlurSigmaPx > 0.4f);
            }
        }
        run.grain.active = wantGrain ? 1 : 0;
        run.grain.sublayersActive = wantGrainSublayers ? 1 : 0;

        // Debug view scaling: stable linear mapping for signed delta fields.
        {
            float densityMaxAvg = (run.grain.densityMax[0] + run.grain.densityMax[1] + run.grain.densityMax[2]) * (1.0f / 3.0f);
            if (!is_positive_finite(densityMaxAvg)) {
                densityMaxAvg = 1.0f;
            }
            run.grain.debugScale = 0.25f / std::max(1e-6f, densityMaxAvg);
        }

        // Phase 3: three-scale mix configuration (fine + mid + coarse).
        {
            float wC = sanitize_unit_or(run.grain.sizeMixWeight, 0.0f);
            float wM = sanitize_unit_or(run.grain.sizeMixWeightMid, 0.0f);
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

            const float scale = std::max(1.0f, sanitize_nonnegative_or(run.grain.sizeMixScale, 1.0f));
            const bool canMix = wantGrain && wantGrainBlur && is_positive_finite(grainBlurSigmaPx) && (scale > 1.0f);
            wantGrainMix = canMix && ((wM > 0.0f) || (wC > 0.0f));

            if (wantGrainMix) {
                const float sigmaF = grainBlurSigmaPx;
                const float sigmaCRaw = sigmaF * std::sqrt(scale);
                grainBlurSigmaCoarsePx = std::max(sigmaF, std::min(sigmaCRaw, sigmaF * 4.0f));
                if (!is_positive_finite(grainBlurSigmaCoarsePx)) {
                    wantGrainMix = false;
                    grainBlurSigmaCoarsePx = 0.0f;
                }
                if (wantGrainMix) {
                    grainBlurSigmaMidPx = std::sqrt(std::max(0.0f, sigmaF * grainBlurSigmaCoarsePx));
                    if (!is_positive_finite(grainBlurSigmaMidPx)) {
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
                    if (!is_positive_finite(sigma)) {
                        return 1.0f;
                    }
                    const int radiusRaw = JuicerGaussian::scipy_gaussian_radius(sigma, 4.0f);
                    const int radius = std::min(radiusRaw, 75);
                    if (radius <= 0) {
                        return 1.0f;
                    }
                    constexpr int kMaxRadius = 75;
                    const double s2 = static_cast<double>(sigma) * static_cast<double>(sigma) * 2.0;
                    double wsum = 0.0;
                    double w[(kMaxRadius * 2) + 1] = {};
                    for (int i = -radius; i <= radius; ++i) {
                        const double wi = std::exp(-(static_cast<double>(i * i)) / s2);
                        w[i + radius] = wi;
                        wsum += wi;
                    }
                    const double invW = (wsum != 0.0) ? (1.0 / wsum) : 0.0;
                    double sumSq = 0.0;
                    for (int i = -radius; i <= radius; ++i) {
                        const double wn = w[i + radius] * invW;
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
        copy_float3x3(result.grainDyeSigmaPx, grainDyeSigmaPx);

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
            if (traceInfo) {
                std::string msg;
                msg.reserve(56 + graphError.size());
                msg = "CUDA base graph launch command failed: ";
                msg += graphError;
                JTRACE("CUDA", msg);
            }
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
            if (traceInfo) {
                std::string msg;
                msg.reserve(56 + scanFlagError.size());
                msg = "CUDA scan error flag allocation failed: ";
                msg += scanFlagError;
                JTRACE("CUDA", msg);
            }
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
                    if (traceInfo) {
                        std::string traceMsg;
                        traceMsg.reserve(64);
                        traceMsg = "CUDA scan error stream wait failed: ";
                        traceMsg += (msg ? msg : "(unknown)");
                        JTRACE("CUDA", traceMsg);
                    }
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
                }
                resources->scanErrorPending = 0;
            }
            else {
                const char* msg = cudaGetErrorString(pollErr);
                mark_context_loss_recovery("scan_error_event_query", pollErr, msg ? msg : "");
                if (traceInfo) {
                    std::string traceMsg;
                    traceMsg.reserve(64);
                    traceMsg = "CUDA scan error event query failed: ";
                    traceMsg += (msg ? msg : "(unknown)");
                    JTRACE("CUDA", traceMsg);
                }
                throw OFX::Exception::Suite(kOfxStatErrFatal);
            }
        }

        cudaError_t flagErr = cudaMemsetAsync(run.scanStage.scanErrorFlag, 0, sizeof(int), stream);
        if (flagErr != cudaSuccess) {
            const char* msg = cudaGetErrorString(flagErr);
            mark_context_loss_recovery("scan_error_flag_memset", flagErr, msg ? msg : "");
            if (traceInfo) {
                std::string traceMsg;
                traceMsg.reserve(64);
                traceMsg = "CUDA scan error flag memset failed: ";
                traceMsg += (msg ? msg : "(unknown)");
                JTRACE("CUDA", traceMsg);
            }
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
        const char* stage = (stageLabel && *stageLabel) ? stageLabel : "pipeline";
        cudaError_t flagErr = cudaSuccess;
        if (resources->scanErrorHost && scanEvent) {
            flagErr = cudaMemcpyAsync(resources->scanErrorHost, run.scanStage.scanErrorFlag, sizeof(int), cudaMemcpyDeviceToHost, stream);
            if (flagErr != cudaSuccess) {
                const char* msg = cudaGetErrorString(flagErr);
                mark_context_loss_recovery("scan_error_flag_readback", flagErr, msg ? msg : "");
                if (traceInfo) {
                    std::string traceMsg;
                    traceMsg.reserve(64);
                    traceMsg = "CUDA scan error flag readback failed: ";
                    traceMsg += (msg ? msg : "(unknown)");
                    JTRACE("CUDA", traceMsg);
                }
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
                if (traceInfo) {
                    std::string traceMsg;
                    traceMsg.reserve(64);
                    traceMsg = "CUDA scan error event record failed: ";
                    traceMsg += (msg ? msg : "(unknown)");
                    JTRACE("CUDA", traceMsg);
                }
                throw OFX::Exception::Suite(kOfxStatErrFatal);
            }
            resources->scanErrorPending = 1;
            cudaError_t pollErr = cudaEventQuery(scanEvent);
            if (pollErr == cudaSuccess) {
                resources->scanErrorPending = 0;
                if (*resources->scanErrorHost != 0) {
                    if (traceInfo) {
                        std::string traceMsg;
                        traceMsg.reserve(64);
                        traceMsg = "FATAL: ";
                        traceMsg += stage;
                        traceMsg += " pipeline scan produced non-finite RGB";
                        JTRACE("CUDA", traceMsg);
                    }
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
                }
            }
            else if (pollErr != cudaErrorNotReady) {
                const char* msg = cudaGetErrorString(pollErr);
                mark_context_loss_recovery("scan_error_event_query", pollErr, msg ? msg : "");
                if (traceInfo) {
                    std::string traceMsg;
                    traceMsg.reserve(64);
                    traceMsg = "CUDA scan error event query failed: ";
                    traceMsg += (msg ? msg : "(unknown)");
                    JTRACE("CUDA", traceMsg);
                }
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
            if (traceInfo) {
                std::string msg;
                msg.reserve(40 + commitError.size());
                msg = "FATAL: commit_submission failed: ";
                msg += commitError;
                JTRACE("CUDA", msg);
            }
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
        const char* scanLabel = negativeMedium ? "scan" : "print scan";
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
                if (traceInfo) {
                    std::string msg;
                    msg.reserve(48 + lutError.size());
                    msg = "CUDA ";
                    msg += scanLabel;
                    msg += " LUT upload failed: ";
                    msg += lutError;
                    JTRACE("CUDA", msg);
                }
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
                if (traceInfo) {
                    std::string msg;
                    msg.reserve(56);
                    msg = "FATAL: ";
                    msg += scanLabel;
                    msg += " LUT missing after successful upload";
                    JTRACE("CUDA", msg);
                }
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
        copy_float3(run.scanStage.scanTables.min_cmy, scanMedium.min_cmy);
        copy_float3(run.scanStage.scanTables.inv_max_cmy, scanMedium.inv_max_cmy);

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
                if (traceInfo) {
                    std::string msg;
                    msg.reserve(72 + dirError.size());
                    msg = "CUDA spatial DIR scratch deferred by contention policy: ";
                    msg += dirError;
                    JTRACE("CUDA", msg);
                }
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
            }
            mark_context_loss_recovery("command_ensure_spatial_dir_scratch", cudaErrorUnknown, dirError);
            if (traceInfo) {
                std::string msg;
                msg.reserve(56 + dirError.size());
                msg = "CUDA spatial DIR scratch allocation failed: ";
                msg += dirError;
                JTRACE("CUDA", msg);
            }
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
            if (traceInfo) {
                std::string msg;
                msg.reserve(56 + dirError.size());
                msg = "CUDA spatial DIR kernel upload failed: ";
                msg += dirError;
                JTRACE("CUDA", msg);
            }
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
            if (traceInfo) {
                std::string traceMsg;
                traceMsg.reserve(64);
                traceMsg = "FATAL: spatial DIR build failed: ";
                traceMsg += (msg ? msg : "(unknown)");
                JTRACE("CUDA", traceMsg);
            }
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        return true;
    };

    auto trace_cuda_scanner_preflight_fail = [&](const char* mediumLabel, const std::string& error) {
        if (!traceInfo) {
            return;
        }
        std::string msg;
        msg.reserve(96 + error.size());
        msg = "path=cuda result=fail medium=";
        msg += mediumLabel;
        msg += " reason=";
        msg += error;
        JTRACE("MSSKV", msg);
        std::string fatalMsg;
        fatalMsg.reserve(8 + error.size());
        fatalMsg = "FATAL: ";
        fatalMsg += error;
        JTRACE("CUDA", fatalMsg);
    };

    auto trace_cuda_scanner_preflight_ok = [&](const char* mediumLabel, std::uint64_t staticKeyHash) {
        if (!traceVerbose) {
            return;
        }
        std::string msg;
        msg.reserve(96);
        msg = "path=cuda result=ok medium=";
        msg += (mediumLabel ? mediumLabel : "unspecified");
        msg += " static_key_hash=";
        msg += std::to_string(staticKeyHash);
        JTRACE_VERBOSE("MSSKV", msg);
    };

    auto throw_cuda_optics_scratch_failure = [&](const std::string& opticsError) {
        if (is_scratch_contention_exhausted(opticsError)) {
            if (traceInfo) {
                std::string msg;
                msg.reserve(64 + opticsError.size());
                msg = "CUDA optics scratch deferred by contention policy: ";
                msg += opticsError;
                JTRACE("CUDA", msg);
            }
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
            throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
            throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
        }
        mark_context_loss_recovery("command_ensure_optics_scratch", cudaErrorUnknown, opticsError);
        if (traceInfo) {
            std::string msg;
            msg.reserve(56 + opticsError.size());
            msg = "CUDA optics scratch allocation failed: ";
            msg += opticsError;
            JTRACE("CUDA", msg);
        }
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
        throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
        throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
    };

    auto throw_cuda_gate_mask_build_failure = [&](const char* stageTag, cudaError_t gateErr) {
        const char* msg = cudaGetErrorString(gateErr);
        mark_context_loss_recovery(stageTag ? stageTag : "build_gate_mask", gateErr, msg ? msg : "");
        if (traceInfo) {
            std::string traceMsg;
            traceMsg.reserve(64);
            traceMsg = "FATAL: gate defect mask build failed: ";
            traceMsg += (msg ? msg : "(unknown)");
            JTRACE("CUDA", traceMsg);
        }
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    };

    auto ensure_gaussian_kernel_or_throw = [&](auto& kernel,
                                               float sigma,
                                               const char* kernelLabel,
                                               std::string& opticsError) {
        if (JuicerCuda::ResourceManager::command_ensure_gaussian_kernel(
                submissionTxn,
                *cudaResources,
                kernel,
                sigma,
                _pCudaStream,
                opticsError)) {
            return;
        }
        if (traceInfo) {
            std::string msg;
            msg.reserve(48 + opticsError.size());
            msg = "CUDA ";
            msg += (kernelLabel ? kernelLabel : "gaussian");
            msg += " kernel upload failed: ";
            msg += opticsError;
            JTRACE("CUDA", msg);
        }
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
        throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
        throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
    };

    auto ensure_halation_kernel_or_throw = [&](auto& kernel,
                                               float sigma,
                                               const char* kernelLabel,
                                               std::string& opticsError) {
        if (JuicerCuda::ResourceManager::command_ensure_halation_kernel(
                submissionTxn,
                *cudaResources,
                kernel,
                sigma,
                _pCudaStream,
                opticsError)) {
            return;
        }
        if (traceInfo) {
            std::string msg;
            msg.reserve(48 + opticsError.size());
            msg = "CUDA ";
            msg += (kernelLabel ? kernelLabel : "halation");
            msg += " kernel upload failed: ";
            msg += opticsError;
            JTRACE("CUDA", msg);
        }
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
        throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
        throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
    };

    auto ensure_grain_dye_kernel_or_throw = [&](auto& kernel,
                                                float sigma,
                                                std::string& opticsError) {
        ensure_gaussian_kernel_or_throw(kernel, sigma, "grain dye-cloud", opticsError);
    };

    auto throw_pipeline_launch_failure = [&](const char* stageTag,
                                             const char* failurePrefix,
                                             cudaError_t pipelineErr) {
        const char* msg = cudaGetErrorString(pipelineErr);
        mark_context_loss_recovery(stageTag ? stageTag : "pipeline_kernel_launch", pipelineErr, msg ? msg : "");
        if (traceInfo) {
            std::string traceMsg;
            traceMsg.reserve(96);
            traceMsg = "FATAL: ";
            traceMsg += (failurePrefix ? failurePrefix : "pipeline kernel launch failed");
            traceMsg += ": ";
            traceMsg += (msg ? msg : "(unknown)");
            JTRACE("CUDA", traceMsg);
        }
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    };

    auto reset_grain_kernel_slots = [&](JuicerCuda::PipelineRunParams& run) {
        run.grainKernels.blurKernel = nullptr;
        run.grainKernels.blurRadius = 0;
        run.grainKernels.blurKernelMid = nullptr;
        run.grainKernels.blurRadiusMid = 0;
        run.grainKernels.blurKernelCoarse = nullptr;
        run.grainKernels.blurRadiusCoarse = 0;
        std::fill_n(&run.grainKernels.dyeKernel[0][0], 9, nullptr);
        std::fill_n(&run.grainKernels.dyeRadius[0][0], 9, 0);
    };

    auto reset_halation_kernel_slots = [&](JuicerCuda::PipelineRunParams& run,
                                           bool wantHalation,
                                           const float* halationStrengthBGR,
                                           const float* halationScatterStrengthBGR) {
        run.halation.active = wantHalation ? 1 : 0;
        float* strengthIt = run.halation.strength;
        float* scatterStrengthIt = run.halation.scatteringStrength;
        const float* srcStrengthIt = halationStrengthBGR;
        const float* srcScatterStrengthIt = halationScatterStrengthBGR;
        const float* const srcStrengthEnd = srcStrengthIt + 3;
        const float fillValue = 0.0f;
        for (; srcStrengthIt != srcStrengthEnd;
             ++strengthIt, ++scatterStrengthIt, ++srcStrengthIt, ++srcScatterStrengthIt) {
            *strengthIt = wantHalation ? *srcStrengthIt : fillValue;
            *scatterStrengthIt = wantHalation ? *srcScatterStrengthIt : fillValue;
        }
        std::fill_n(run.halationKernels.halationKernel, 3, nullptr);
        std::fill_n(run.halationKernels.halationRadius, 3, 0);
        std::fill_n(run.halationKernels.scatteringKernel, 3, nullptr);
        std::fill_n(run.halationKernels.scatteringRadius, 3, 0);
    };

    auto bind_grain_dye_kernels_or_throw = [&](JuicerCuda::PipelineRunParams& run,
                                               const GrainSetupResult& grainSetup,
                                               std::string& opticsError) {
        for (int layer = 0; layer < 3; ++layer) {
            for (int ch = 0; ch < 3; ++ch) {
                const float sigma = grainSetup.grainDyeSigmaPx[layer][ch];
                ensure_grain_dye_kernel_or_throw(
                    cudaResources->grainDyeKernel[layer][ch],
                    sigma,
                    opticsError);
                if (sigma > 0.0f) {
                    run.grainKernels.dyeKernel[layer][ch] = cudaResources->grainDyeKernel[layer][ch].weights;
                    run.grainKernels.dyeRadius[layer][ch] = cudaResources->grainDyeKernel[layer][ch].radius;
                }
            }
        }
    };

    auto bind_halation_kernels_or_throw = [&](JuicerCuda::PipelineRunParams& run,
                                              const float* halationStrengthBGR,
                                              const float* halationSigmaPx,
                                              const float* halationScatterStrengthBGR,
                                              const float* halationScatterSigmaPx,
                                              std::string& opticsError) {
        const float* strengthIt = halationStrengthBGR;
        const float* sigmaIt = halationSigmaPx;
        const float* scatterStrengthIt = halationScatterStrengthBGR;
        const float* scatterSigmaIt = halationScatterSigmaPx;
        const float* const strengthEnd = strengthIt + 3;
        int i = 0;
        for (; strengthIt != strengthEnd; ++strengthIt, ++sigmaIt, ++scatterStrengthIt, ++scatterSigmaIt, ++i) {
            if (*strengthIt > 0.0f && *sigmaIt > 0.0f) {
                ensure_halation_kernel_or_throw(
                    cudaResources->halationKernel[i],
                    *sigmaIt,
                    "halation",
                    opticsError);
                run.halationKernels.halationKernel[i] = cudaResources->halationKernel[i].weights;
                run.halationKernels.halationRadius[i] = cudaResources->halationKernel[i].radius;
            }
            if (*scatterStrengthIt > 0.0f && *scatterSigmaIt > 0.0f) {
                ensure_halation_kernel_or_throw(
                    cudaResources->halationScatterKernel[i],
                    *scatterSigmaIt,
                    "halation scatter",
                    opticsError);
                run.halationKernels.scatteringKernel[i] = cudaResources->halationScatterKernel[i].weights;
                run.halationKernels.scatteringRadius[i] = cudaResources->halationScatterKernel[i].radius;
            }
        }
    };

    auto initialize_pipeline_run = [&](JuicerCuda::PipelineRunParams& run) {
        run.src = srcPtr;
        run.srcRowBytes = static_cast<std::size_t>(srcRowBytes);
        run.dst = dstPtr;
        run.dstRowBytes = static_cast<std::size_t>(dstRowBytes);
        run.width = width;
        run.height = height;
        run.nComponents = _nComponents;
    };

    auto populate_scan_color_payload = [&](JuicerCuda::PipelineRunParams& run,
                                           const Scanner::ColorRuntime& color) {
        copy_float9(run.scanStage.scanColor.cat02, color.cat02);
        copy_float9(run.scanStage.scanColor.xyzToRgb, color.xyzToRgb);
        copy_float3(run.scanStage.scanColor.illuminantXYZ, color.illuminantXYZ);

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
        copy_float9(run.scanStage.scanColor.encoding.dwgToOutput, dwgToOutput.m);
    };

    auto populate_common_pipeline_payload = [&](JuicerCuda::PipelineRunParams& run,
                                                const ScannerPreflightResult& scannerPreflight) {
        run.filmRaw.inputColorSpaceIndex = Spectral::inputColorSpaceToIndex(_ws->filmRaw.inputColorSpace);
        run.filmRaw.applyCctfDecoding = _ws->filmRaw.applyCctfDecoding ? 1 : 0;
        run.filmRaw.applyInputChromaticAdapt = _ws->filmRaw.applyInputChromaticAdapt ? 1 : 0;
        run.filmRaw.spectralUpsamplingMode = static_cast<int>(_ws->filmRaw.spectralUpsamplingMode);
        copy_float9(run.filmRaw.inputRGBToXYZ, _ws->filmRaw.inputRGBToXYZ.m);
        copy_float9(run.filmRaw.inputXYZAdapt, _ws->filmRaw.inputXYZAdapt.m);
        run.filmRaw.midgrayScale = _ws->filmRaw.midgrayScale;
        copy_float3(run.filmRaw.refIllumWhiteXYZ, _ws->filmRaw.refIllumWhiteXYZ);

        run.filmExpose.exposureScale = _exposureScale;
        run.filmDevelop.gammaFactorB = _ws->gammaFactorB;
        run.filmDevelop.gammaFactorG = _ws->gammaFactorG;
        run.filmDevelop.gammaFactorR = _ws->gammaFactorR;
        run.filmDevelop.dirPrecorrected = _ws->dirPrecorrected ? 1 : 0;

        run.filmDevelop.dir.active = _dirRT.active ? 1 : 0;
        run.filmDevelop.dir.highShift = _dirRT.highShift;
        copy_float9(run.filmDevelop.dir.M, &_dirRT.M[0][0]);
        copy_float3(run.filmDevelop.dir.dMax, _dirRT.dMax);

        populate_scan_color_payload(run, *scannerPreflight.colorRuntime);
    };

    auto populate_film_runtime_payload = [&](JuicerCuda::PipelineRunParams& run) {
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
        copy_float9(run.filmExpose.spdSInv, cudaResources->spdSInv);

        run.filmExpose.hanatosLut = cudaResources->hanatosLut;
        run.filmExpose.hanatosN = cudaResources->hanatosN;
        run.filmExpose.hanatosLutIntegrated = cudaResources->hanatosLutIntegrated;
        run.filmExpose.hanatosNIntegrated = cudaResources->hanatosNIntegrated;
        run.filmExpose.mallettBasis = cudaResources->mallettBasis;
        run.filmExpose.mallettBasisK = cudaResources->mallettBasisK;
    };

    // RenderMode::NegativeOnly (PrintBypass=true).
    if (renderMode == RenderMode::NegativeOnly) {
        constexpr const char* kCudaNegativeMediumLabel = "negative";
        ScannerPreflightResult scannerPreflight{};
        std::string scannerPreflightError;
        if (!validate_scanner_preflight_runtime(
                _ws->negativeScannerValid,
                kCudaNegativeMediumLabel,
                &_ws->negativeMediumRuntime,
                scannerPreflight,
                scannerPreflightError)) {
            trace_cuda_scanner_preflight_fail(kCudaNegativeMediumLabel, scannerPreflightError);
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        trace_cuda_scanner_preflight_ok(kCudaNegativeMediumLabel, scannerPreflight.staticKey.hash);
        const Scanner::ScannerMediumRuntime& negativeMediumRuntime = *scannerPreflight.mediumRuntime;

        const float lensBlurSigmaPx = _scannerOptions.lensBlurSigmaPx;
        const float unsharpSigmaPx = _scannerOptions.unsharpSigmaPx;
        const float unsharpAmount = _scannerOptions.unsharpAmount;
        const bool glareActive = negativeMediumRuntime.glare.active && (negativeMediumRuntime.glare.percent > 0.0f);
        const bool useSpatialDIR = spatial_dir_enabled(_dirRT);

        JuicerCuda::PipelineRunParams run{};
        initialize_pipeline_run(run);
        populate_common_pipeline_payload(run, scannerPreflight);

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

            populate_film_runtime_payload(run);

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
            const HalationSetupResult halationSetup = setup_halation_payload(halationUi);
            const float* halationStrengthBGR = halationSetup.strengthBGR;
            const float* halationScatterStrengthBGR = halationSetup.scatterStrengthBGR;
            const float* halationSigmaPx = halationSetup.sigmaPx;
            const float* halationScatterSigmaPx = halationSetup.scatterSigmaPx;
            const bool wantHalation = halationSetup.wantHalation;

            const Profiles::GrainMetadata grainUi = _hasGrainOverride ? _grainOverride : _ws->grain;
            const GrainSetupResult grainSetup = setup_grain_payload(run, grainUi, true);
            const bool wantGrain = grainSetup.wantGrain;
            const bool wantGrainSublayers = grainSetup.wantGrainSublayers;
            const bool wantGrainBlur = grainSetup.wantGrainBlur;
            const bool wantGrainMix = grainSetup.wantGrainMix;
            const float grainBlurSigmaPx = grainSetup.grainBlurSigmaPx;
            const float grainBlurSigmaMidPx = grainSetup.grainBlurSigmaMidPx;
            const bool needGrainShared = needs_grain_shared(
                wantGrain, run.grain.debugView, run.grain.chromaMix);

            const bool wantLensBlur = is_positive_finite(lensBlurSigmaPx);
            const bool wantUnsharp = wants_unsharp(unsharpSigmaPx, unsharpAmount);
            const bool wantGlareBlur = wantGlare && is_positive_finite(glareBlurSigmaPx);
            const bool wantWeave = (run.gateWeave.active != 0);
            const bool wantDefects = has_grain_defects(
                run.grain.filmDustAmount,
                run.grain.gateDustAmount,
                run.grain.filmScratchAmount,
                run.grain.gateScratchAmount);
            const bool needGateMask = needs_gate_mask_for_defects(
                run.grain.gateDustAmount,
                run.grain.gateScratchAmount);
            const bool wantOptics = wants_optics_stage(
                wantLensBlur, wantUnsharp, wantGlare, wantHalation, wantGrain, wantWeave, wantDefects);

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
                const OpticsScratchNeeds scratchNeeds = build_optics_scratch_needs(
                    wantGlareBlur,
                    wantGrainBlur,
                    wantGrainSublayers,
                    wantGrainMix);
                if (!JuicerCuda::ResourceManager::command_ensure_optics_scratch(
                        submissionTxn,
                        *cudaResources,
                        width,
                        height,
                        scratchNeeds.blurred,
                        scratchNeeds.aux,
                        scratchNeeds.grain,
                        needGrainShared,
                        needGateMask,
                        _pCudaStream,
                        opticsError)) {
                    throw_cuda_optics_scratch_failure(opticsError);
                }
                if (needGateMask && cudaResources->scannerScratch.gateMask) {
                    const std::uint64_t gateHash = make_gate_mask_hash(
                        run.grain.stbnSessionSeed,
                        run.grain.originX,
                        run.grain.originY,
                        width,
                        height,
                        run.grain.pixelSizeUm,
                        run.grain.gateDustAmount,
                        run.grain.gateScratchAmount);
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
                            throw_cuda_gate_mask_build_failure("build_gate_mask_negative", gateErr);
                        }
                        cudaResources->scannerScratch.gateMaskHash = gateHash;
                    }
                    run.grain.gateMask = cudaResources->scannerScratch.gateMask;
                    run.grain.gateMaskWidth = cudaResources->scannerScratch.gateWidth;
                    run.grain.gateMaskHeight = cudaResources->scannerScratch.gateHeight;
                }
                ensure_gaussian_kernel_or_throw(cudaResources->scannerLensBlurKernel, lensBlurSigmaPx, "lens blur", opticsError);
                ensure_gaussian_kernel_or_throw(cudaResources->scannerUnsharpKernel, unsharpSigmaPx, "unsharp", opticsError);
                ensure_gaussian_kernel_or_throw(
                    cudaResources->scannerGlareKernel,
                    wantGlare ? glareBlurSigmaPx : 0.0f,
                    "glare",
                    opticsError);

                reset_grain_kernel_slots(run);
                if (wantGrain) {
                    ensure_gaussian_kernel_or_throw(
                        cudaResources->grainBlurKernel,
                        wantGrainBlur ? grainBlurSigmaPx : 0.0f,
                        "grain blur",
                        opticsError);
                    if (wantGrainBlur) {
                        run.grainKernels.blurKernel = cudaResources->grainBlurKernel.weights;
                        run.grainKernels.blurRadius = cudaResources->grainBlurKernel.radius;
                    }
                    if (wantGrainMix) {
                        ensure_gaussian_kernel_or_throw(
                            cudaResources->grainBlurKernelMid,
                            grainBlurSigmaMidPx,
                            "grain mid blur",
                            opticsError);
                        ensure_gaussian_kernel_or_throw(
                            cudaResources->grainBlurKernelCoarse,
                            grainSetup.grainBlurSigmaCoarsePx,
                            "grain coarse blur",
                            opticsError);
                        run.grainKernels.blurKernelMid = cudaResources->grainBlurKernelMid.weights;
                        run.grainKernels.blurRadiusMid = cudaResources->grainBlurKernelMid.radius;
                        run.grainKernels.blurKernelCoarse = cudaResources->grainBlurKernelCoarse.weights;
                        run.grainKernels.blurRadiusCoarse = cudaResources->grainBlurKernelCoarse.radius;
                    }

                    if (wantGrainSublayers) {
                        bind_grain_dye_kernels_or_throw(run, grainSetup, opticsError);
                    }
                }

                reset_halation_kernel_slots(run, wantHalation, halationStrengthBGR, halationScatterStrengthBGR);
                if (wantHalation) {
                    bind_halation_kernels_or_throw(
                        run,
                        halationStrengthBGR,
                        halationSigmaPx,
                        halationScatterStrengthBGR,
                        halationScatterSigmaPx,
                        opticsError);
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
                throw_pipeline_launch_failure(
                    "negative_pipeline_kernel_launch",
                    "negative pipeline kernel launch failed",
                    err);
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
        constexpr const char* kCudaPrintMediumLabel = "print";
        if (!validate_scanner_preflight_runtime(
                _ws->printScannerValid,
                kCudaPrintMediumLabel,
                &_ws->printMediumRuntime,
                scannerPreflight,
                scannerPreflightError)) {
            trace_cuda_scanner_preflight_fail(kCudaPrintMediumLabel, scannerPreflightError);
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        trace_cuda_scanner_preflight_ok(kCudaPrintMediumLabel, scannerPreflight.staticKey.hash);

        const bool useSpatialDIR = spatial_dir_enabled(_dirRT);

        // Print exposure compensation factor is computed on CPU (no image reads; safe for CUDA renders).
        const float kMidSpectral = compute_print_midgray_factor_cached(
            _instanceState,
            *_ws,
            *_prt,
            _printParams,
            _dirRT);

        JuicerCuda::PipelineRunParams run{};
        initialize_pipeline_run(run);
        populate_common_pipeline_payload(run, scannerPreflight);

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
                if (traceInfo) {
                    std::string msg;
                    msg.reserve(48 + illumError.size());
                    msg = "CUDA print illuminant upload failed: ";
                    msg += illumError;
                    JTRACE("CUDA", msg);
                }
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
            }
            if (traceVerbose) {
                std::lock_guard<std::mutex> resLock(cudaResources->m);
                const std::uint64_t uploadCoreHash =
                    (_ws->uploadCoreHash != 0) ? _ws->uploadCoreHash : _ws->coreHash;
                std::string msg;
                msg.reserve(384);
                msg = "cuda print payload build=";
                msg += std::to_string(_ws->buildCounter);
                msg += " uploadCoreHash=";
                msg += std::to_string(uploadCoreHash);
                msg += " neutralY/M/C=";
                msg += std::to_string(_prt->neutralY);
                msg += "/";
                msg += std::to_string(_prt->neutralM);
                msg += "/";
                msg += std::to_string(_prt->neutralC);
                msg += " yFilter=";
                msg += std::to_string(_printParams.yFilter);
                msg += " mFilter=";
                msg += std::to_string(_printParams.mFilter);
                msg += " cFilter=";
                msg += std::to_string(_printParams.cFilter);
                msg += " illumBuild=";
                msg += std::to_string(cudaResources->printIllumBuildCounter);
                msg += " illumCoreHash=";
                msg += std::to_string(cudaResources->printIllumCoreHash);
                msg += " illumNeutralHash=";
                msg += std::to_string(cudaResources->printIllumNeutralFilterHash);
                msg += " illumY/M/Csteps=";
                msg += std::to_string(cudaResources->printIllumYShiftSteps);
                msg += "/";
                msg += std::to_string(cudaResources->printIllumMShiftSteps);
                msg += "/";
                msg += std::to_string(cudaResources->printIllumCShiftSteps);
                msg += " preflashValid=";
                msg += std::to_string(cudaResources->printPreflashValid ? 1 : 0);
                msg += " preflashBuild=";
                msg += std::to_string(cudaResources->printPreflashBuildCounter);
                JTRACE_VERBOSE("PRINTDBG", msg);
            }

            // Film density curves + sensitivities + SPD reconstruction tables.
            populate_film_runtime_payload(run);

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
            copy_float3(run.printExpose.negTables.min_cmy, cudaResources->scanNegative.min_cmy);
            copy_float3(run.printExpose.negTables.inv_max_cmy, cudaResources->scanNegative.inv_max_cmy);

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
            copy_float3(run.printExpose.printPreflashRaw, cudaResources->printPreflashRaw);

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
            const HalationSetupResult halationSetup = setup_halation_payload(halationUi);
            const float* halationStrengthBGR = halationSetup.strengthBGR;
            const float* halationScatterStrengthBGR = halationSetup.scatterStrengthBGR;
            const float* halationSigmaPx = halationSetup.sigmaPx;
            const float* halationScatterSigmaPx = halationSetup.scatterSigmaPx;
            const bool wantHalation = halationSetup.wantHalation;

            const Profiles::GrainMetadata grainUi = _hasGrainOverride ? _grainOverride : _ws->grain;
            const GrainSetupResult grainSetup = setup_grain_payload(run, grainUi, false);
            const bool wantGrain = grainSetup.wantGrain;
            const bool wantGrainSublayers = grainSetup.wantGrainSublayers;
            const bool wantGrainBlur = grainSetup.wantGrainBlur;
            const bool wantGrainMix = grainSetup.wantGrainMix;
            const float grainBlurSigmaPx = grainSetup.grainBlurSigmaPx;
            const float grainBlurSigmaMidPx = grainSetup.grainBlurSigmaMidPx;
            const bool needGrainShared = needs_grain_shared(
                wantGrain, run.grain.debugView, run.grain.chromaMix);

            const float lensBlurSigmaPx = _scannerOptions.lensBlurSigmaPx;
            const float unsharpSigmaPx = _scannerOptions.unsharpSigmaPx;
            const float unsharpAmount = _scannerOptions.unsharpAmount;

            const bool wantLensBlur = is_positive_finite(lensBlurSigmaPx);
            const bool wantUnsharp = wants_unsharp(unsharpSigmaPx, unsharpAmount);
            const bool wantGlareBlur = wantGlare && is_positive_finite(glareBlurSigmaPx);
            const bool wantWeave = (run.gateWeave.active != 0);
            const bool wantDefects = has_grain_defects(
                run.grain.filmDustAmount,
                run.grain.gateDustAmount,
                run.grain.filmScratchAmount,
                run.grain.gateScratchAmount);
            const bool needGateMask = needs_gate_mask_for_defects(
                run.grain.gateDustAmount,
                run.grain.gateScratchAmount);
            const bool wantOptics = wants_optics_stage(
                wantLensBlur, wantUnsharp, wantGlare, wantHalation, wantGrain, wantWeave, wantDefects);

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
                const OpticsScratchNeeds scratchNeeds = build_optics_scratch_needs(
                    wantGlareBlur,
                    wantGrainBlur,
                    wantGrainSublayers,
                    wantGrainMix);
                if (!JuicerCuda::ResourceManager::command_ensure_optics_scratch(
                        submissionTxn,
                        *cudaResources,
                        width,
                        height,
                        scratchNeeds.blurred,
                        scratchNeeds.aux,
                        scratchNeeds.grain,
                        needGrainShared,
                        needGateMask,
                        _pCudaStream,
                        opticsError)) {
                    throw_cuda_optics_scratch_failure(opticsError);
                }
                if (needGateMask && cudaResources->scannerScratch.gateMask) {
                    const std::uint64_t gateHash = make_gate_mask_hash(
                        run.grain.stbnSessionSeed,
                        run.grain.originX,
                        run.grain.originY,
                        width,
                        height,
                        run.grain.pixelSizeUm,
                        run.grain.gateDustAmount,
                        run.grain.gateScratchAmount);
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
                            throw_cuda_gate_mask_build_failure("build_gate_mask_print", gateErr);
                        }
                        cudaResources->scannerScratch.gateMaskHash = gateHash;
                    }
                    run.grain.gateMask = cudaResources->scannerScratch.gateMask;
                    run.grain.gateMaskWidth = cudaResources->scannerScratch.gateWidth;
                    run.grain.gateMaskHeight = cudaResources->scannerScratch.gateHeight;
                }
                ensure_gaussian_kernel_or_throw(cudaResources->scannerLensBlurKernel, lensBlurSigmaPx, "lens blur", opticsError);
                ensure_gaussian_kernel_or_throw(cudaResources->scannerUnsharpKernel, unsharpSigmaPx, "unsharp", opticsError);
                ensure_gaussian_kernel_or_throw(
                    cudaResources->scannerGlareKernel,
                    wantGlare ? glareBlurSigmaPx : 0.0f,
                    "glare",
                    opticsError);

                reset_grain_kernel_slots(run);
                if (wantGrain) {
                    ensure_gaussian_kernel_or_throw(
                        cudaResources->grainBlurKernel,
                        wantGrainBlur ? grainBlurSigmaPx : 0.0f,
                        "grain blur",
                        opticsError);
                    if (wantGrainBlur) {
                        run.grainKernels.blurKernel = cudaResources->grainBlurKernel.weights;
                        run.grainKernels.blurRadius = cudaResources->grainBlurKernel.radius;
                    }
                    if (wantGrainMix) {
                        ensure_gaussian_kernel_or_throw(
                            cudaResources->grainBlurKernelMid,
                            grainBlurSigmaMidPx,
                            "grain mid blur",
                            opticsError);
                        ensure_gaussian_kernel_or_throw(
                            cudaResources->grainBlurKernelCoarse,
                            grainSetup.grainBlurSigmaCoarsePx,
                            "grain coarse blur",
                            opticsError);
                        run.grainKernels.blurKernelMid = cudaResources->grainBlurKernelMid.weights;
                        run.grainKernels.blurRadiusMid = cudaResources->grainBlurKernelMid.radius;
                        run.grainKernels.blurKernelCoarse = cudaResources->grainBlurKernelCoarse.weights;
                        run.grainKernels.blurRadiusCoarse = cudaResources->grainBlurKernelCoarse.radius;
                    }

                    if (wantGrainSublayers) {
                        bind_grain_dye_kernels_or_throw(run, grainSetup, opticsError);
                    }
                }

                reset_halation_kernel_slots(run, wantHalation, halationStrengthBGR, halationScatterStrengthBGR);
                if (wantHalation) {
                    bind_halation_kernels_or_throw(
                        run,
                        halationStrengthBGR,
                        halationSigmaPx,
                        halationScatterStrengthBGR,
                        halationScatterSigmaPx,
                        opticsError);
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
                throw_pipeline_launch_failure(
                    "print_pipeline_kernel_launch",
                    "print pipeline kernel launch failed",
                    err);
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
