// JuicerProcessing.cpp

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <sstream>
#include <mutex>
#include <limits>

#include "GaussianSciPy.h"

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
#include <cuda_runtime.h>
#include <cuda.h>
#if defined(_WIN32)
#include <windows.h>
#endif
#endif

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
#include "Cuda/JuicerCudaDirProfile.h"
#include "Cuda/JuicerCudaDirectFilmPayloads.h"
#include "Cuda/JuicerCudaResources.h"
#include "Cuda/JuicerCudaPayloads.h"
#include "Cuda/ResourceManager/JuicerCudaResourceCore.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#endif

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
extern "C" cudaError_t juicer_cuda_negative_pipeline(
    const JuicerCuda::PipelineRunParams* hParams,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_negative_direct_pipeline(
    const JuicerCuda::DirectPipelineRunParams* hParams,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_print_focused_pipeline(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_direct_focused_scanner_post_effects(
    const JuicerCuda::DirectPipelineRunParams* hParams,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    float* dScratchBlurred,
    const float* dLensBlurKernel,
    int lensBlurRadius,
    const float* dUnsharpKernel,
    int unsharpRadius,
    float unsharpAmount,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_print_focused_scanner_post_effects(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    float* dScratchBlurred,
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

extern "C" cudaError_t juicer_cuda_build_gate_defect_mask(
    const JuicerCuda::PipelineRunParams* hParams,
    float* dGateMask,
    int gateWidth,
    int gateHeight,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_build_spatial_dir(
    const JuicerCuda::PipelineRunParams* hParams,
    float* rawCorrectionY,
    float* rawCorrectionM,
    float* rawCorrectionC,
    float* filteredCorrectionY,
    float* filteredCorrectionM,
    float* filteredCorrectionC,
    float* filterTemp,
    float* iirForwardTemp,
    const float* dGaussianKernel,
    int gaussianRadius,
    float gaussianSigma,
    float gaussianWeight,
    const float* dTailKernel0,
    int tailRadius0,
    float tailSigma0,
    float tailWeight0,
    const float* dTailKernel1,
    int tailRadius1,
    float tailSigma1,
    float tailWeight1,
    const float* dTailKernel2,
    int tailRadius2,
    float tailSigma2,
    float tailWeight2,
    void* cudaStreamOpaque,
    JuicerCuda::SpatialDirBuildProfile* profile);

extern "C" cudaError_t juicer_cuda_build_direct_spatial_dir(
    const JuicerCuda::DirectPipelineRunParams* hParams,
    float* rawCorrectionY,
    float* rawCorrectionM,
    float* rawCorrectionC,
    float* filteredCorrectionY,
    float* filteredCorrectionM,
    float* filteredCorrectionC,
    float* filterTemp,
    float* iirForwardTemp,
    const float* dGaussianKernel,
    int gaussianRadius,
    float gaussianSigma,
    float gaussianWeight,
    const float* dTailKernel0,
    int tailRadius0,
    float tailSigma0,
    float tailWeight0,
    const float* dTailKernel1,
    int tailRadius1,
    float tailSigma1,
    float tailWeight1,
    const float* dTailKernel2,
    int tailRadius2,
    float tailSigma2,
    float tailWeight2,
    void* cudaStreamOpaque,
    JuicerCuda::SpatialDirBuildProfile* profile);

extern "C" cudaError_t juicer_cuda_build_print_spatial_dir(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    float* rawCorrectionY,
    float* rawCorrectionM,
    float* rawCorrectionC,
    float* filteredCorrectionY,
    float* filteredCorrectionM,
    float* filteredCorrectionC,
    float* filterTemp,
    float* iirForwardTemp,
    const float* dGaussianKernel,
    int gaussianRadius,
    float gaussianSigma,
    float gaussianWeight,
    const float* dTailKernel0,
    int tailRadius0,
    float tailSigma0,
    float tailWeight0,
    const float* dTailKernel1,
    int tailRadius1,
    float tailSigma1,
    float tailWeight1,
    const float* dTailKernel2,
    int tailRadius2,
    float tailSigma2,
    float tailWeight2,
    void* cudaStreamOpaque,
    JuicerCuda::SpatialDirBuildProfile* profile);

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
#pragma warning(disable : 5040)
#include "ofxsProcessing.h"
#include "ofxsImageEffect.h"
#pragma warning(pop)
#include "Logging.h"
#include "Hash.h"
#include "SpectralData.h"
#include "ColorTransforms.h"
#include "Print.h"
#include "ProcessRoot.h"
#include "JuicerState.h"
#include "Scanner.h"
#include "OutputColor.h"
#include "Couplers.h"
#include "mainProcessing.h"

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__) && defined(JUICER_CUDA_SELF_CHECK) && (JUICER_CUDA_SELF_CHECK != 0)
bool juicer_cuda_runtime_self_check(void* cudaStreamOpaque, const char** outError);
#endif

namespace JuicerProcScanner {

    struct ScannerPreflightResult {
        const Scanner::ScannerMediumRuntime* mediumRuntime = nullptr;
        const Scanner::ColorRuntime* colorRuntime = nullptr;
        Scanner::ScannerStaticKey staticKey{};
        Scanner::ScannerRuntimeEffectsKey effectsKey{};
    };

    struct ScannerKeyBundle {
        Scanner::ScannerRuntimeKey runtimeKey{};
        Scanner::ScannerKey scannerKey{};
    };

    struct ScannerMediumRuntimeBinding {
        const Scanner::ScannerMediumRuntime* selectedRuntime = nullptr;
        Scanner::ScannerMediumRuntime printOverride{};
        bool usesPrintOverride = false;
        bool valid = false;
        const char* label = "scanner";

        const Scanner::ScannerMediumRuntime* runtime() const {
            return usesPrintOverride ? &printOverride : selectedRuntime;
        }
    };

    std::uint64_t hash_scanner_runtime_lane(
        const WorkingState* ws,
        const Scanner::Settings& settings,
        const Scanner::Options& options,
        std::uint32_t frameBoundsVersion);

    ScannerMediumRuntimeBinding bind_scanner_medium_runtime(
        const WorkingState& ws,
        bool printActive,
        bool hasPrintGlareOverride,
        const Profiles::ProfileGlare* printGlareOverride,
        bool forcePrintGlareHash);

    ScannerPreflightResult validate_scanner_preflight_or_throw(
        bool runtimeValid,
        const char* mediumLabel,
        const Scanner::ScannerMediumRuntime* mediumRuntime,
        bool traceInfoEnabled,
        bool traceVerboseEnabled,
        const char* path,
        const char* fatalTag);

    ScannerKeyBundle make_scanner_key_bundle_or_throw(
        const Scanner::ScannerStaticKey& staticKey,
        const Scanner::ScannerRuntimeEffectsKey& effectsKey,
        const Scanner::Settings& settings,
        const Scanner::Options& options,
        std::uint32_t frameBoundsVersion);

} // namespace JuicerProcScanner

namespace {
    inline float sanitize_nonnegative_or(float value, float fallback);
    inline bool is_finite(float value);
    inline bool is_finite(double value);

    inline void copy_float3(float dst[3], const float src[3]) {
        std::memcpy(dst, src, 3u * sizeof(float));
    }

    inline const char* nonempty_cstr_or(const char* value, const char* fallback) {
        return (value && value[0] != '\0') ? value : fallback;
    }

    inline const char* detail_or_unknown(const char* detail) {
        return nonempty_cstr_or(detail, "(unknown)");
    }

    inline const char* cstr_or_null_if_empty(const std::string& value) {
        return value.empty() ? nullptr : value.c_str();
    }

    inline int bool_to_i32(bool value) {
        return value ? 1 : 0;
    }

    inline void clear_glare_compensation_fields(Profiles::ProfileGlare& glare) {
        glare.printShadowCompensationFactor = 0.0f;
        glare.printShadowCompensationDensity = 0.0f;
        glare.printShadowCompensationTransition = 0.0f;
    }

    inline const char* scanner_medium_label_from_print_active(bool printActive) {
        return printActive ? "print" : "negative";
    }

    inline std::uint64_t bool_to_u64(bool value) {
        return value ? 1ull : 0ull;
    }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    bool spatial_dir_profile_enabled() {
        static const bool enabled = []() {
            const char* value = std::getenv("JUICER_DIR_PROFILE");
            if (!value || value[0] == '\0') {
                return false;
            }
            return std::strcmp(value, "0") != 0 &&
                   std::strcmp(value, "false") != 0 &&
                   std::strcmp(value, "FALSE") != 0 &&
                   std::strcmp(value, "off") != 0 &&
                   std::strcmp(value, "OFF") != 0;
        }();
        return enabled;
    }

    std::filesystem::path spatial_dir_profile_path() {
        const char* explicitPath = std::getenv("JUICER_DIR_PROFILE_PATH");
        if (explicitPath && explicitPath[0] != '\0') {
            try {
                return std::filesystem::path(explicitPath);
            } catch (...) {
                return {};
            }
        }
        try {
            std::filesystem::path path = std::filesystem::temp_directory_path();
            path /= "juicer_dir_profile.txt";
            return path;
        } catch (...) {
            return {};
        }
    }

    void write_spatial_dir_profile_line(const std::string& line) {
        static std::mutex profileMutex;
        static bool headerWritten = false;
        const std::filesystem::path path = spatial_dir_profile_path();
        if (path.empty()) {
            return;
        }

        std::lock_guard<std::mutex> lock(profileMutex);
        try {
            const std::filesystem::path parent = path.parent_path();
            if (!parent.empty()) {
                std::error_code ec;
                std::filesystem::create_directories(parent, ec);
            }

            std::ofstream out(path, std::ios::out | std::ios::app | std::ios::binary);
            if (!out.is_open()) {
                return;
            }
            if (!headerWritten) {
                const auto now = std::chrono::system_clock::now();
                const auto secs =
                    std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
                out << "INIT | dir_profile enabled time_s=" << secs << '\n';
                headerWritten = true;
            }
            out << "DIR_PROFILE | " << line << '\n';
            out.flush();
        } catch (...) {
        }
    }

    double elapsed_ms_since(std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - start)
            .count();
    }

    struct CudaEventElapsedTimer {
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        bool active = false;

        ~CudaEventElapsedTimer() {
            destroy();
        }

        bool begin(void* streamOpaque) {
            const cudaStream_t stream =
                streamOpaque ? reinterpret_cast<cudaStream_t>(streamOpaque) : nullptr;
            cudaError_t err = cudaEventCreateWithFlags(&start, cudaEventDefault);
            if (err != cudaSuccess) {
                destroy();
                return false;
            }
            err = cudaEventCreateWithFlags(&stop, cudaEventDefault);
            if (err != cudaSuccess) {
                destroy();
                return false;
            }
            err = cudaEventRecord(start, stream);
            if (err != cudaSuccess) {
                destroy();
                return false;
            }
            active = true;
            return true;
        }

        float finish(void* streamOpaque) {
            if (!active) {
                return -1.0f;
            }
            const cudaStream_t stream =
                streamOpaque ? reinterpret_cast<cudaStream_t>(streamOpaque) : nullptr;
            cudaError_t err = cudaEventRecord(stop, stream);
            if (err != cudaSuccess) {
                destroy();
                return -1.0f;
            }
            err = cudaEventSynchronize(stop);
            if (err != cudaSuccess) {
                destroy();
                return -1.0f;
            }
            float elapsedMs = -1.0f;
            err = cudaEventElapsedTime(&elapsedMs, start, stop);
            destroy();
            return err == cudaSuccess ? elapsedMs : -1.0f;
        }

        void destroy() {
            if (start) {
                cudaEventDestroy(start);
                start = nullptr;
            }
            if (stop) {
                cudaEventDestroy(stop);
                stop = nullptr;
            }
            active = false;
        }
    };

    int active_tail_component_count(const JuicerCuda::SpatialDirBuildProfile& profile) {
        int count = 0;
        for (float weight : profile.tailWeight) {
            if (weight > 0.0f) {
                ++count;
            }
        }
        return count;
    }

    const char* dir_filter_operator_label(float sigma, float weight) {
        if (!(weight > 0.0f) || !(sigma > 0.0f)) {
            return "none";
        }
        return sigma >= 3.0f ? "iir_yvv" : "fir_reflect";
    }

    Spektrafilm::DirFrameExtent spatial_dir_extent_from_rect(const OfxRectI& rect) {
        Spektrafilm::DirFrameExtent extent{};
        extent.x = rect.x1;
        extent.y = rect.y1;
        extent.width = rect.x2 - rect.x1;
        extent.height = rect.y2 - rect.y1;
        return extent;
    }

    const Spektrafilm::DirGaussianComponentPlan& spatial_dir_component_or_empty(
        const Spektrafilm::SpatialDirDescriptor& descriptor,
        int index) {
        static const Spektrafilm::DirGaussianComponentPlan kEmpty{};
        if (index < 0 || index >= descriptor.filterPlan.componentCount) {
            return kEmpty;
        }
        return descriptor.filterPlan.components[static_cast<std::size_t>(index)];
    }

    void bind_spatial_dir_workspace_metadata(
        const Spektrafilm::SpatialDirDescriptor& descriptor,
        JuicerProcess::Root::PreparedCudaFrame::WorkspaceRequest& request) {
        if (descriptor.hash == 0) {
            return;
        }
        request.needSpatialDir = true;
        request.spatialDirDescriptorHash = descriptor.hash;
        request.spatialDirScratchTier = descriptor.scratchTier;
        request.spatialDirPlaneRoles = descriptor.planeRoles;
        request.spatialDirTargetScratchTier = descriptor.targetScratchTier;
        request.spatialDirTargetPlaneRoles = descriptor.targetPlaneRoles;
    }

    // SF_TEMP_BRIDGE_bind_filtered_correction_to_payload: Phase 5 removes this
    // once FilmDevelopPayload stops exposing corr* aliases for spatial DIR.
    void SF_TEMP_BRIDGE_bind_filtered_correction_to_payload(
        JuicerCuda::FilmDevelopPayload& payload,
        const JuicerProcess::Root::PreparedCudaFrame::SpatialDirScratchView& scratch) {
        payload.spatialDir.active = 1;
        payload.spatialDir.corrY = scratch.filteredCorrectionY;
        payload.spatialDir.corrM = scratch.filteredCorrectionM;
        payload.spatialDir.corrC = scratch.filteredCorrectionC;
    }

    void trace_spatial_dir_descriptor_build(
        const char* route,
        const Spektrafilm::SpatialDirDescriptor& descriptor) {
#if JUICER_DIAGNOSTICS_COMPILED
        if (!JTRACE_ENABLED(1)) {
            return;
        }
        const Spektrafilm::DirScratchPlaneRoles& roles = descriptor.planeRoles;
        const Spektrafilm::DirScratchPlaneRoles& targetRoles = descriptor.targetPlaneRoles;
        std::string msg = "event=spatial_dir_descriptor_build route=";
        msg += nonempty_cstr_or(route, "unknown");
        msg += " descriptor_hash=";
        msg += std::to_string(static_cast<unsigned long long>(descriptor.hash));
        msg += " legacy_compatibility_hash=";
        msg += std::to_string(static_cast<unsigned long long>(descriptor.legacyCompatibilityHash));
        msg += " dir_recipe_hash=";
        msg += std::to_string(static_cast<unsigned long long>(descriptor.dirRecipeHash));
        msg += " support=";
        msg += Spektrafilm::to_cstr(descriptor.support);
        msg += " source_contract=";
        msg += Spektrafilm::to_cstr(descriptor.sourceContract);
        msg += " boundary_mode=";
        msg += Spektrafilm::to_cstr(descriptor.boundaryMode);
        msg += " scratch_tier=";
        msg += Spektrafilm::to_cstr(descriptor.scratchTier);
        msg += " target_scratch_tier=";
        msg += Spektrafilm::to_cstr(descriptor.targetScratchTier);
        msg += " approximation=";
        msg += Spektrafilm::to_cstr(descriptor.approximation);
        msg += " component_count=";
        msg += std::to_string(descriptor.filterPlan.componentCount);
        msg += " render_extent=";
        msg += std::to_string(descriptor.renderExtent.x) + "," +
               std::to_string(descriptor.renderExtent.y) + "," +
               std::to_string(descriptor.renderExtent.width) + "x" +
               std::to_string(descriptor.renderExtent.height);
        msg += " full_frame_extent=";
        msg += std::to_string(descriptor.fullFrameExtent.x) + "," +
               std::to_string(descriptor.fullFrameExtent.y) + "," +
               std::to_string(descriptor.fullFrameExtent.width) + "x" +
               std::to_string(descriptor.fullFrameExtent.height);
        msg += " filter_domain_extent=";
        msg += std::to_string(descriptor.filterDomainExtent.x) + "," +
               std::to_string(descriptor.filterDomainExtent.y) + "," +
               std::to_string(descriptor.filterDomainExtent.width) + "x" +
               std::to_string(descriptor.filterDomainExtent.height);
        msg += " raw_correction_planes=";
        msg += std::to_string(roles.rawCorrectionPlanes);
        msg += " filtered_correction_planes=";
        msg += std::to_string(roles.filteredCorrectionPlanes);
        msg += " filter_temp_planes=";
        msg += std::to_string(roles.filterTempPlanes);
        msg += " iir_forward_temp_planes=";
        msg += std::to_string(roles.iirForwardTempPlanes);
        msg += " cached_log_raw_planes=";
        msg += std::to_string(roles.cachedLogRawPlanes);
        msg += " SF_TEMP_BRIDGE_corr_planes=";
        msg += std::to_string(roles.SF_TEMP_BRIDGE_corrPlanes);
        msg += " SF_TEMP_BRIDGE_mix_planes=";
        msg += std::to_string(roles.SF_TEMP_BRIDGE_mixPlanes);
        msg += " SF_TEMP_BRIDGE_tmp_planes=";
        msg += std::to_string(roles.SF_TEMP_BRIDGE_tmpPlanes);
        msg += " target_raw_correction_planes=";
        msg += std::to_string(targetRoles.rawCorrectionPlanes);
        msg += " target_filtered_correction_planes=";
        msg += std::to_string(targetRoles.filteredCorrectionPlanes);
        msg += " target_filter_temp_planes=";
        msg += std::to_string(targetRoles.filterTempPlanes);
        msg += " target_iir_forward_temp_planes=";
        msg += std::to_string(targetRoles.iirForwardTempPlanes);
        for (int component = 0; component < Spektrafilm::DirFilterPlan::kMaxComponents; ++component) {
            const Spektrafilm::DirGaussianComponentPlan& plan =
                spatial_dir_component_or_empty(descriptor, component);
            msg += " component";
            msg += std::to_string(component);
            msg += "_sigma_px=";
            msg += std::to_string(plan.sigmaPixels);
            msg += " component";
            msg += std::to_string(component);
            msg += "_weight=";
            msg += std::to_string(plan.weight);
            msg += " component";
            msg += std::to_string(component);
            msg += "_reference_operator=";
            msg += Spektrafilm::to_cstr(plan.referenceOperator);
            msg += " component";
            msg += std::to_string(component);
            msg += "_backend=";
            msg += Spektrafilm::to_cstr(plan.backend);
            msg += " component";
            msg += std::to_string(component);
            msg += "_target_backend=";
            msg += Spektrafilm::to_cstr(plan.targetBackend);
            msg += " component";
            msg += std::to_string(component);
            msg += "_target_scratch_tier=";
            msg += Spektrafilm::to_cstr(plan.targetScratchTier);
        }
        JTRACE("DIR_DESCRIPTOR", msg);
#else
        (void)route;
        (void)descriptor;
#endif
    }

    void trace_spatial_dir_bridge_use(
        const char* route,
        const char* bridgeName,
        const Spektrafilm::SpatialDirDescriptor& descriptor) {
#if JUICER_DIAGNOSTICS_COMPILED
        if (!JTRACE_ENABLED(1)) {
            return;
        }
        std::string msg = "event=spatial_dir_bridge_use route=";
        msg += nonempty_cstr_or(route, "unknown");
        msg += " bridge=";
        msg += nonempty_cstr_or(bridgeName, "none");
        msg += " descriptor_hash=";
        msg += std::to_string(static_cast<unsigned long long>(descriptor.hash));
        msg += " dir_recipe_hash=";
        msg += std::to_string(static_cast<unsigned long long>(descriptor.dirRecipeHash));
        msg += " support=";
        msg += Spektrafilm::to_cstr(descriptor.support);
        msg += " source_contract=";
        msg += Spektrafilm::to_cstr(descriptor.sourceContract);
        msg += " scratch_tier=";
        msg += Spektrafilm::to_cstr(descriptor.scratchTier);
        msg += " component_count=";
        msg += std::to_string(descriptor.filterPlan.componentCount);
        JTRACE("DIR_BRIDGE", msg);
#else
        (void)route;
        (void)bridgeName;
        (void)descriptor;
#endif
    }

    void trace_spatial_dir_profile(
        const char* route,
        int width,
        int height,
        const Spektrafilm::SpatialDirDescriptor& descriptor,
        const JuicerCuda::SpatialDirBuildProfile& profile,
        bool dirActive,
        bool scratchOverflow,
        double descriptorMs,
        double prepareMs,
        double buildHostMs,
        float pipelineCudaMs,
        double pipelineLaunchHostMs) {
        const std::uint64_t pixels =
            static_cast<std::uint64_t>(std::max(0, width)) *
            static_cast<std::uint64_t>(std::max(0, height));
        const int admittedPlaneCount = dirActive ? descriptor.planeRoles.total_float_planes() : 0;
        const std::uint64_t scratchBytesApprox =
            dirActive ? pixels * static_cast<std::uint64_t>(std::max(0, admittedPlaneCount)) * sizeof(float) : 0ull;
        const int activeTails = active_tail_component_count(profile);
        const Spektrafilm::DirScratchPlaneRoles& roles = descriptor.planeRoles;
        const Spektrafilm::DirScratchPlaneRoles& targetRoles = descriptor.targetPlaneRoles;
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3);
        oss << "route=" << nonempty_cstr_or(route, "unknown")
            << " dir_active=" << bool_to_i32(dirActive)
            << " spatial_dir_bridge="
            << (dirActive ? nonempty_cstr_or(profile.SF_TEMP_BRIDGE_name, "unreported") : "none")
            << " width=" << width
            << " height=" << height
            << " descriptor_hash=" << descriptor.hash
            << " legacy_compatibility_hash=" << descriptor.legacyCompatibilityHash
            << " dir_recipe_hash=" << descriptor.dirRecipeHash
            << " descriptor_support=" << Spektrafilm::to_cstr(descriptor.support)
            << " source_contract=" << Spektrafilm::to_cstr(descriptor.sourceContract)
            << " boundary_mode=" << Spektrafilm::to_cstr(descriptor.boundaryMode)
            << " scratch_tier=" << Spektrafilm::to_cstr(descriptor.scratchTier)
            << " target_scratch_tier=" << Spektrafilm::to_cstr(descriptor.targetScratchTier)
            << " approximation_marker=" << Spektrafilm::to_cstr(descriptor.approximation)
            << " component_count=" << descriptor.filterPlan.componentCount
            << " render_origin=" << descriptor.renderExtent.x << "," << descriptor.renderExtent.y
            << " render_extent=" << descriptor.renderExtent.width << "x" << descriptor.renderExtent.height
            << " full_frame_origin=" << descriptor.fullFrameExtent.x << "," << descriptor.fullFrameExtent.y
            << " full_frame_extent=" << descriptor.fullFrameExtent.width << "x" << descriptor.fullFrameExtent.height
            << " filter_domain_origin=" << descriptor.filterDomainExtent.x << "," << descriptor.filterDomainExtent.y
            << " filter_domain_extent=" << descriptor.filterDomainExtent.width << "x" << descriptor.filterDomainExtent.height
            << " gaussian_sigma_px=" << descriptor.gaussianSigmaPixels
            << " gaussian_radius=" << profile.gaussianRadius
            << " gaussian_operator=" << Spektrafilm::to_cstr(spatial_dir_component_or_empty(descriptor, 0).referenceOperator)
            << " gaussian_backend=" << Spektrafilm::to_cstr(spatial_dir_component_or_empty(descriptor, 0).backend)
            << " gaussian_target_backend=" << Spektrafilm::to_cstr(spatial_dir_component_or_empty(descriptor, 0).targetBackend)
            << " gaussian_target_scratch_tier=" << Spektrafilm::to_cstr(spatial_dir_component_or_empty(descriptor, 0).targetScratchTier)
            << " gaussian_weight=" << descriptor.gaussianWeight
            << " tail0_sigma_px=" << profile.tailSigma[0]
            << " tail0_radius=" << profile.tailRadius[0]
            << " tail0_operator=" << dir_filter_operator_label(profile.tailSigma[0], profile.tailWeight[0])
            << " tail0_weight=" << profile.tailWeight[0]
            << " tail1_sigma_px=" << profile.tailSigma[1]
            << " tail1_radius=" << profile.tailRadius[1]
            << " tail1_operator=" << dir_filter_operator_label(profile.tailSigma[1], profile.tailWeight[1])
            << " tail1_weight=" << profile.tailWeight[1]
            << " tail2_sigma_px=" << profile.tailSigma[2]
            << " tail2_radius=" << profile.tailRadius[2]
            << " tail2_operator=" << dir_filter_operator_label(profile.tailSigma[2], profile.tailWeight[2])
            << " tail2_weight=" << profile.tailWeight[2]
            << " active_tail_components=" << activeTails
            << " expected_correction_launches=" << (dirActive ? 1 : 0)
            << " descriptor_ms=" << descriptorMs
            << " prepare_ms=" << prepareMs
            << " build_host_ms=" << buildHostMs
            << " build_cuda_ms=" << profile.total.elapsedMs
            << " dir_source_launches=" << profile.correctionLaunches
            << " dir_source_ms=" << profile.correction.elapsedMs
            << " dir_filter_bank_launches="
            << (profile.baseFilterLaunches + profile.tailFilterLaunches[0] +
                profile.tailFilterLaunches[1] + profile.tailFilterLaunches[2])
            << " dir_filter_bank_ms="
            << (profile.baseFilter.elapsedMs + profile.tailFilter[0].elapsedMs +
                profile.tailFilter[1].elapsedMs + profile.tailFilter[2].elapsedMs)
            << " pipeline_cuda_ms=" << pipelineCudaMs
            << " pipeline_launch_host_ms=" << pipelineLaunchHostMs
            << " total_launches=" << profile.totalLaunches
            << " correction_launches=" << profile.correctionLaunches
            << " correction_ms=" << profile.correction.elapsedMs
            << " correction_clamp_hits=" << profile.correctionClampHits
            << " base_filter_launches=" << profile.baseFilterLaunches
            << " base_filter_ms=" << profile.baseFilter.elapsedMs
            << " tail0_filter_launches=" << profile.tailFilterLaunches[0]
            << " tail0_filter_ms=" << profile.tailFilter[0].elapsedMs
            << " tail1_filter_launches=" << profile.tailFilterLaunches[1]
            << " tail1_filter_ms=" << profile.tailFilter[1].elapsedMs
            << " tail2_filter_launches=" << profile.tailFilterLaunches[2]
            << " tail2_filter_ms=" << profile.tailFilter[2].elapsedMs
            << " scale_copy_launches=" << profile.scaleCopyLaunches
            << " scale_copy_ms=" << profile.scaleCopy.elapsedMs
            << " add_scaled_launches=" << profile.addScaledLaunches
            << " add_scaled_ms=" << profile.addScaled.elapsedMs
            << " scratch_source=" << (dirActive ? (scratchOverflow ? "overflow" : "retained") : "none")
            << " raw_correction_planes=" << (dirActive ? roles.rawCorrectionPlanes : 0)
            << " filtered_correction_planes=" << (dirActive ? roles.filteredCorrectionPlanes : 0)
            << " filter_temp_planes=" << (dirActive ? roles.filterTempPlanes : 0)
            << " iir_forward_temp_planes=" << (dirActive ? roles.iirForwardTempPlanes : 0)
            << " cached_log_raw_planes=" << (dirActive ? roles.cachedLogRawPlanes : 0)
            << " SF_TEMP_BRIDGE_corr_planes=" << (dirActive ? roles.SF_TEMP_BRIDGE_corrPlanes : 0)
            << " SF_TEMP_BRIDGE_mix_planes=" << (dirActive ? roles.SF_TEMP_BRIDGE_mixPlanes : 0)
            << " SF_TEMP_BRIDGE_tmp_planes=" << (dirActive ? roles.SF_TEMP_BRIDGE_tmpPlanes : 0)
            << " spatial_dir_planes=" << (dirActive ? roles.total_float_planes() : 0)
            << " shared_tmp_planes=" << (dirActive ? roles.filterTempPlanes : 0)
            << " target_raw_correction_planes=" << (dirActive ? targetRoles.rawCorrectionPlanes : 0)
            << " target_filtered_correction_planes=" << (dirActive ? targetRoles.filteredCorrectionPlanes : 0)
            << " target_filter_temp_planes=" << (dirActive ? targetRoles.filterTempPlanes : 0)
            << " target_iir_forward_temp_planes=" << (dirActive ? targetRoles.iirForwardTempPlanes : 0)
            << " scratch_bytes_approx=" << scratchBytesApprox;
        for (int component = 0; component < Spektrafilm::DirFilterPlan::kMaxComponents; ++component) {
            const Spektrafilm::DirGaussianComponentPlan& plan =
                spatial_dir_component_or_empty(descriptor, component);
            oss << " component" << component << "_sigma_px=" << plan.sigmaPixels
                << " component" << component << "_weight=" << plan.weight
                << " component" << component << "_reference_operator=" << Spektrafilm::to_cstr(plan.referenceOperator)
                << " component" << component << "_backend=" << Spektrafilm::to_cstr(plan.backend)
                << " component" << component << "_target_backend=" << Spektrafilm::to_cstr(plan.targetBackend)
                << " component" << component << "_target_scratch_tier=" << Spektrafilm::to_cstr(plan.targetScratchTier);
        }
        const std::string line = oss.str();
        write_spatial_dir_profile_line(line);
        JTRACE("DIR_PROFILE", line);
    }
#endif

    inline void apply_glare_override_fields(
        Profiles::ProfileGlare& dstGlare,
        const Profiles::ProfileGlare& srcGlare) {
        dstGlare.active = srcGlare.active;
        dstGlare.percent = srcGlare.percent;
        dstGlare.roughness = srcGlare.roughness;
        dstGlare.blur = srcGlare.blur;
        clear_glare_compensation_fields(dstGlare);
    }

    inline bool assign_glare_hash(Scanner::ScannerMediumRuntime& mediumRuntime) {
        const std::uint64_t glareRuntimeHash = Scanner::hash_glare(mediumRuntime.glare);
        if (glareRuntimeHash == 0) {
            return false;
        }
        mediumRuntime.effectsKey.glareRuntimeHash = glareRuntimeHash;
        return true;
    }

    std::uint64_t hash_scanner_settings(
        const Scanner::Settings& settings,
        const Scanner::Options& options) {
        const std::uint64_t lutHash = Hash::hash_bytes(&settings.useLut, sizeof(settings.useLut));
        const float fields[3] = {
            options.lensBlurSigmaPx,
            options.unsharpSigmaPx,
            options.unsharpAmount};
        const std::uint64_t optHash = Hash::hash_float_span(fields, 3);
        if (lutHash == 0 || optHash == 0) {
            JTRACE("HASH", "FATAL: invalid scanner settings for hashing");
            return 0;
        }
        const std::uint64_t combined[2] = {lutHash, optHash};
        return Hash::hash_bytes(combined, sizeof(combined));
    }

    const char* scanner_medium_label_or_default(const char* mediumLabel) {
        return nonempty_cstr_or(mediumLabel, "scanner");
    }

    struct ScannerMediumSelection {
        const Scanner::ScannerMediumRuntime* runtime = nullptr;
        bool valid = false;
        const char* label = "scanner";
    };

    ScannerMediumSelection select_scanner_medium(
        const WorkingState& ws,
        bool printActive) {
        ScannerMediumSelection selection{};
        selection.label = scanner_medium_label_from_print_active(printActive);
        if (printActive) {
            selection.runtime = &ws.printMediumRuntime;
            selection.valid = ws.printScannerValid;
            return selection;
        }
        selection.runtime = &ws.negativeMediumRuntime;
        selection.valid = ws.negativeScannerValid;
        return selection;
    }

    void trace_scanner_preflight_fail_if_enabled(
        bool traceInfoEnabled,
        const char* path,
        const char* mediumLabel,
        const std::string& error,
        const char* fatalTag) {
        if (!traceInfoEnabled) {
            return;
        }
        const char* label = scanner_medium_label_or_default(mediumLabel);
        std::string msg;
        msg.reserve(96 + error.size());
        msg = "path=";
        msg += nonempty_cstr_or(path, "unspecified");
        msg += " result=fail medium=";
        msg += label;
        msg += " reason=";
        msg += error;
        JTRACE("MSSKV", msg);
        std::string fatalMsg;
        fatalMsg.reserve(8 + error.size());
        fatalMsg = "FATAL: ";
        fatalMsg += error;
        JTRACE(nonempty_cstr_or(fatalTag, "SCAN"), fatalMsg);
    }

    void trace_scanner_preflight_ok_if_enabled(
        bool traceVerboseEnabled,
        const char* path,
        const char* mediumLabel,
        std::uint64_t staticKeyHash) {
        if (!traceVerboseEnabled) {
            return;
        }
        const char* label = scanner_medium_label_or_default(mediumLabel);
        std::string msg;
        msg.reserve(96);
        msg = "path=";
        msg += nonempty_cstr_or(path, "unspecified");
        msg += " result=ok medium=";
        msg += label;
        msg += " static_key_hash=";
        msg += std::to_string(staticKeyHash);
        JTRACE_VERBOSE("MSSKV", msg);
    }

    void assign_glare_hash_or_throw(Scanner::ScannerMediumRuntime& mediumRuntime) {
        if (assign_glare_hash(mediumRuntime)) {
            Scanner::finalize_runtime_effects_key(mediumRuntime.effectsKey);
            return;
        }
        JTRACE("HASH", "FATAL: failed to hash print glare override parameters");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    bool validate_scanner_preflight_runtime(
        bool runtimeValid,
        const char* mediumLabel,
        const Scanner::ScannerMediumRuntime* mediumRuntime,
        JuicerProcScanner::ScannerPreflightResult& out,
        std::string& outError) {
        out = JuicerProcScanner::ScannerPreflightResult{};
        outError.clear();

        const char* label = scanner_medium_label_or_default(mediumLabel);
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
        Scanner::ScannerRuntimeEffectsKey effectsKey = mediumRuntime->effectsKey;
        if (tables->tablesHash != staticKey.tablesHash) {
            set_error(" scanner tables hash mismatch for medium");
            return false;
        }
        if (mediumRuntime->range.digest == 0) {
            set_error(" scanner density range missing or invalid");
            return false;
        }
        if (mediumRuntime->range.digest != staticKey.densityRangeHash) {
            set_error(" scanner density range hash mismatch for medium");
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
        if (effectsKey.colorRuntimeHash != colorPtr->hash) {
            set_error(" scanner runtime-effects key color hash mismatch");
            return false;
        }

        const std::uint64_t expectedGlareHash = Scanner::hash_glare(mediumRuntime->glare);
        if (expectedGlareHash == 0) {
            set_error(" scanner glare hash missing or invalid");
            return false;
        }
        if (effectsKey.glareRuntimeHash != expectedGlareHash) {
            set_error(" scanner runtime-effects key glare hash mismatch");
            return false;
        }

        const std::uint64_t storedStaticHash = staticKey.hash;
        Scanner::finalize_static_key(staticKey);
        if (staticKey.hash == 0) {
            set_error(" scanner static key missing or invalid");
            return false;
        }
        if (storedStaticHash != staticKey.hash) {
            set_error(" scanner static key hash mismatch");
            return false;
        }
        const std::uint64_t storedEffectsHash = effectsKey.hash;
        Scanner::finalize_runtime_effects_key(effectsKey);
        if (effectsKey.hash == 0 || storedEffectsHash != effectsKey.hash) {
            set_error(" scanner runtime-effects key hash mismatch");
            return false;
        }

        out.mediumRuntime = mediumRuntime;
        out.colorRuntime = colorPtr;
        out.staticKey = staticKey;
        out.effectsKey = effectsKey;
        return true;
    }

    inline void trace_cuda_fatal_prefixed_if(
        bool traceEnabled,
        const char* prefix,
        const char* detail = nullptr) {
        if (!traceEnabled) {
            return;
        }
        const char* safePrefix = nonempty_cstr_or(prefix, "CUDA operation failed");
        const bool hasDetail = detail && detail[0] != '\0';
        std::string traceMsg;
        traceMsg.reserve(
            8 + std::strlen(safePrefix) + (hasDetail ? (2 + std::strlen(detail)) : 0));
        traceMsg = "FATAL: ";
        traceMsg += safePrefix;
        if (hasDetail) {
            traceMsg += ": ";
            traceMsg += detail;
        }
        JTRACE("CUDA", traceMsg);
    }

    inline void copy_float9(float dst[9], const float src[9]) {
        std::memcpy(dst, src, 9u * sizeof(float));
    }

    inline float sanitize_nonnegative_or(float value, float fallback) {
        if (!is_finite(value)) {
            return fallback;
        }
        return std::max(0.0f, value);
    }

    inline double finite_or(double value, double fallback) {
        return is_finite(value) ? value : fallback;
    }

    inline bool is_gpu_render_requested(bool openclEnabled, bool cudaEnabled, bool metalEnabled) {
        return openclEnabled || cudaEnabled || metalEnabled;
    }

    inline bool is_positive_finite(float value) {
        return is_finite(value) && value > 0.0f;
    }

    inline bool is_positive_finite(double value) {
        return is_finite(value) && value > 0.0;
    }

    inline float positive_finite_or(float value, float fallback) {
        return is_positive_finite(value) ? value : fallback;
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
            outError = nonempty_cstr_or(dispatch.loadError, "driver dispatch unavailable");
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

    void recover_context_loss_state(
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

        std::string retireError;
#if JUICER_DIAGNOSTICS_COMPILED
        const char* stageName = nonempty_cstr_or(stage, "unknown_stage");
        const bool traceInfo = JTRACE_ENABLED(1);
        const bool retireAccepted = JuicerProcess::root().retire_reset_context(
            key.deviceId,
            key.contextOpaque,
            retireError);
#else
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
#endif

    std::int64_t frame_index_from_time(double time) {
        return static_cast<std::int64_t>(std::floor(finite_or(time, 0.0)));
    }

    std::uint64_t session_seed_or_default(std::uint64_t sessionSeed) {
        return (sessionSeed != 0) ? sessionSeed : 1;
    }

    std::uint64_t instance_token_or_session_seed(std::uint64_t instanceToken, std::uint64_t sessionSeed) {
        return (instanceToken != 0) ? instanceToken : session_seed_or_default(sessionSeed);
    }

#if JUICER_DIAGNOSTICS_COMPILED
    const char* runtime_lease_outcome_label(bool waitedForLease) {
        return waitedForLease ? "wait_acquired" : "acquired";
    }

    const char* submission_snapshot_action_label(bool reusingSnapshotLatch) {
        return reusingSnapshotLatch ? "reuse" : "new";
    }
#endif

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__) &&                                    \
    ((defined(JUICER_CUDA_VALIDATE_PRIMITIVES) && (JUICER_CUDA_VALIDATE_PRIMITIVES != 0)) || \
     (defined(JUICER_CUDA_SELF_CHECK) && (JUICER_CUDA_SELF_CHECK != 0)))
    struct DiagnosticsHookPolicy {
        bool diagnosticsMode = false;
        bool validatePrimitives = false;
        bool runtimeSelfCheck = false;
    };

    inline const char* diagnostics_mode_label(bool modeEnabled) {
        return modeEnabled ? "diagnostics" : "serving";
    }

    bool parse_env_toggle(const char* name, bool fallback) {
        return JuicerLogging::parse_env_int(name, bool_to_i32(fallback)) != 0;
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

#if defined(JUICER_CUDA_VALIDATE_PRIMITIVES) && (JUICER_CUDA_VALIDATE_PRIMITIVES != 0)
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
            } else if (!modeEnabled) {
                reason = "diagnostics_mode_disabled";
            } else if (!toggleEnabled) {
                reason = "validation_toggle_disabled";
            } else if (!verboseEnabled) {
                reason = "diagnostics_level_below_verbose";
            }
            std::string msg;
            msg.reserve(160);
            msg = "event=diagnostics_hook";
            msg += " hook=validation";
            msg += " mode=";
            msg += diagnostics_mode_label(modeEnabled);
            msg += " compiled=";
            msg += std::to_string(bool_to_i32(compiled));
            msg += " toggle_enabled=";
            msg += std::to_string(bool_to_i32(toggleEnabled));
            msg += " verbose_enabled=";
            msg += std::to_string(bool_to_i32(verboseEnabled));
            msg += " active=";
            msg += std::to_string(bool_to_i32(active));
            msg += " reason=";
            msg += reason;
            JTRACE("MSDBG", msg);
        });
    }
#endif

#if defined(JUICER_CUDA_SELF_CHECK) && (JUICER_CUDA_SELF_CHECK != 0)
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
            } else if (!modeEnabled) {
                reason = "diagnostics_mode_disabled";
            } else if (!toggleEnabled) {
                reason = "self_check_toggle_disabled";
            }
            std::string msg;
            msg.reserve(144);
            msg = "event=diagnostics_hook";
            msg += " hook=self_check";
            msg += " mode=";
            msg += diagnostics_mode_label(modeEnabled);
            msg += " compiled=";
            msg += std::to_string(bool_to_i32(compiled));
            msg += " toggle_enabled=";
            msg += std::to_string(bool_to_i32(toggleEnabled));
            msg += " active=";
            msg += std::to_string(bool_to_i32(active));
            msg += " reason=";
            msg += reason;
            JTRACE("MSDBG", msg);
        });
    }
#endif
#endif

} // namespace

namespace JuicerProcScanner {

    std::uint64_t hash_scanner_runtime_lane(
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

        std::uint64_t negStaticHash = 0;
        std::uint64_t negEffectsHash = 0;
        if (ws->negativeScannerValid) {
            Scanner::ScannerStaticKey negStaticKey = ws->negativeStaticKey;
            Scanner::finalize_static_key(negStaticKey);
            negStaticHash = negStaticKey.hash;
            Scanner::ScannerRuntimeEffectsKey negEffectsKey = ws->negativeEffectsKey;
            Scanner::finalize_runtime_effects_key(negEffectsKey);
            negEffectsHash = negEffectsKey.hash;
        }

        const bool printValid = ws->printScannerValid;
        std::uint64_t printStaticHash = 0;
        std::uint64_t printEffectsHash = 0;
        if (printValid) {
            Scanner::ScannerStaticKey printStaticKey = ws->printStaticKey;
            Scanner::finalize_static_key(printStaticKey);
            printStaticHash = printStaticKey.hash;
            Scanner::ScannerRuntimeEffectsKey printEffectsKey = ws->printEffectsKey;
            Scanner::finalize_runtime_effects_key(printEffectsKey);
            printEffectsHash = printEffectsKey.hash;
        }

        const std::uint64_t fields[] = {
            settingsHash,
            static_cast<std::uint64_t>(frameBoundsVersion),
            bool_to_u64(ws->negativeScannerValid),
            negStaticHash,
            negEffectsHash,
            bool_to_u64(printValid),
            printStaticHash,
            printEffectsHash};
        return Hash::hash_bytes(fields, sizeof(fields));
    }

    ScannerMediumRuntimeBinding bind_scanner_medium_runtime(
        const WorkingState& ws,
        bool printActive,
        bool hasPrintGlareOverride,
        const Profiles::ProfileGlare* printGlareOverride,
        bool forcePrintGlareHash) {
        ScannerMediumRuntimeBinding binding{};
        const ScannerMediumSelection selection = select_scanner_medium(ws, printActive);
        binding.selectedRuntime = selection.runtime;
        binding.valid = selection.valid;
        binding.label = selection.label;
        binding.usesPrintOverride = false;
        if (!printActive) {
            return binding;
        }

        binding.printOverride = ws.printMediumRuntime;
        if (hasPrintGlareOverride && printGlareOverride) {
            apply_glare_override_fields(binding.printOverride.glare, *printGlareOverride);
        }
        if (forcePrintGlareHash || hasPrintGlareOverride) {
            assign_glare_hash_or_throw(binding.printOverride);
        }
        binding.usesPrintOverride = true;
        return binding;
    }

    ScannerPreflightResult validate_scanner_preflight_or_throw(
        bool runtimeValid,
        const char* mediumLabel,
        const Scanner::ScannerMediumRuntime* mediumRuntime,
        bool traceInfoEnabled,
        bool traceVerboseEnabled,
        const char* path,
        const char* fatalTag) {
        ScannerPreflightResult scannerPreflight{};
        std::string scannerPreflightError;
        if (!validate_scanner_preflight_runtime(
                runtimeValid,
                mediumLabel,
                mediumRuntime,
                scannerPreflight,
                scannerPreflightError)) {
            trace_scanner_preflight_fail_if_enabled(
                traceInfoEnabled,
                path,
                mediumLabel,
                scannerPreflightError,
                fatalTag);
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        trace_scanner_preflight_ok_if_enabled(
            traceVerboseEnabled,
            path,
            mediumLabel,
            scannerPreflight.staticKey.hash);
        return scannerPreflight;
    }

    ScannerKeyBundle make_scanner_key_bundle_or_throw(
        const Scanner::ScannerStaticKey& staticKey,
        const Scanner::ScannerRuntimeEffectsKey& effectsKey,
        const Scanner::Settings& settings,
        const Scanner::Options& options,
        std::uint32_t frameBoundsVersion) {
        ScannerKeyBundle bundle{};
        bundle.runtimeKey.settingsHash = hash_scanner_settings(settings, options);
        bundle.runtimeKey.frameBoundsVersion = frameBoundsVersion;
        if (bundle.runtimeKey.settingsHash == 0) {
            JTRACE("HASH", "FATAL: scanner runtime settings hash invalid");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        Scanner::finalize_runtime_key(bundle.runtimeKey);

        bundle.scannerKey.staticKey = staticKey;
        bundle.scannerKey.effectsKey = effectsKey;
        bundle.scannerKey.runtimeKey = bundle.runtimeKey;
        Scanner::finalize_scanner_key(bundle.scannerKey);
        if (bundle.scannerKey.hash == 0) {
            JTRACE("HASH", "FATAL: scanner combined key invalid");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        return bundle;
    }

} // namespace JuicerProcScanner

namespace {

    JuicerCuda::AutoExposurePreviewDescriptor make_auto_exposure_preview_descriptor(
        const OfxRectI& sourceBounds,
        const OfxRectI& meterBounds,
        Spektrafilm::AutoExposureMethod method) {
        JuicerCuda::AutoExposurePreviewDescriptor descriptor{};
        descriptor.sourceX1 = sourceBounds.x1;
        descriptor.sourceY1 = sourceBounds.y1;
        descriptor.sourceX2 = sourceBounds.x2;
        descriptor.sourceY2 = sourceBounds.y2;
        descriptor.meterX1 = meterBounds.x1;
        descriptor.meterY1 = meterBounds.y1;
        descriptor.meterX2 = meterBounds.x2;
        descriptor.meterY2 = meterBounds.y2;
        descriptor.method = method;
        descriptor.sampling = JuicerCuda::AutoExposurePreviewDescriptor::Sampling::NearestNeighbor;

        const int width = std::max(0, meterBounds.x2 - meterBounds.x1);
        const int height = std::max(0, meterBounds.y2 - meterBounds.y1);
        const int longEdge = std::max(width, height);
        descriptor.previewWidth = width;
        descriptor.previewHeight = height;
        if (longEdge > JuicerCuda::AutoExposurePreviewDescriptor::kMaxLongEdge) {
            if (width >= height) {
                descriptor.previewWidth = JuicerCuda::AutoExposurePreviewDescriptor::kMaxLongEdge;
                descriptor.previewHeight = std::max(
                    1,
                    (height * JuicerCuda::AutoExposurePreviewDescriptor::kMaxLongEdge + width / 2) /
                        width);
            } else {
                descriptor.previewHeight = JuicerCuda::AutoExposurePreviewDescriptor::kMaxLongEdge;
                descriptor.previewWidth = std::max(
                    1,
                    (width * JuicerCuda::AutoExposurePreviewDescriptor::kMaxLongEdge + height / 2) /
                        height);
            }
        }
        descriptor.hash = Hash::kFnvOffset;
        Hash::hash_bytes_update(descriptor.hash, &descriptor.sourceX1, sizeof(descriptor.sourceX1));
        Hash::hash_bytes_update(descriptor.hash, &descriptor.sourceY1, sizeof(descriptor.sourceY1));
        Hash::hash_bytes_update(descriptor.hash, &descriptor.sourceX2, sizeof(descriptor.sourceX2));
        Hash::hash_bytes_update(descriptor.hash, &descriptor.sourceY2, sizeof(descriptor.sourceY2));
        Hash::hash_bytes_update(descriptor.hash, &descriptor.meterX1, sizeof(descriptor.meterX1));
        Hash::hash_bytes_update(descriptor.hash, &descriptor.meterY1, sizeof(descriptor.meterY1));
        Hash::hash_bytes_update(descriptor.hash, &descriptor.meterX2, sizeof(descriptor.meterX2));
        Hash::hash_bytes_update(descriptor.hash, &descriptor.meterY2, sizeof(descriptor.meterY2));
        Hash::hash_bytes_update(descriptor.hash, &descriptor.previewWidth, sizeof(descriptor.previewWidth));
        Hash::hash_bytes_update(descriptor.hash, &descriptor.previewHeight, sizeof(descriptor.previewHeight));
        Hash::hash_bytes_update(descriptor.hash, &descriptor.method, sizeof(descriptor.method));
        Hash::hash_bytes_update(descriptor.hash, &descriptor.sampling, sizeof(descriptor.sampling));
        if (descriptor.hash == 0) {
            descriptor.hash = 1;
        }
        return descriptor;
    }

} // namespace

// --- Spatial DIR: defensive curve utilities (monotonic + robust interpolation) ---

inline std::uint64_t upload_core_hash_or_core_hash(const WorkingState& ws) {
    return (ws.uploadCoreHash != 0) ? ws.uploadCoreHash : ws.coreHash;
}

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
    : OFX::ImageProcessor(effect), _srcImg(nullptr), _nComponents(0), _scannerOptions{}, _scannerSettings{}, _printParams{}, _halationOverride{}, _hasHalationOverride(false), _grainOverride{}, _hasGrainOverride(false), _printGlareOverride{}, _hasPrintGlareOverride(false), _dirRT{}, _prt(nullptr), _ws(nullptr), _wsReady(false), _printReady(false), _exposureScale(1.0f), _outputEncoding{}, _frameBoundsVersion(0), _pixelSizeUm(0.0f) {
}

void JuicerProcessor::setSrcDst(const SourceDestinationImages& images) {
    _srcImg = images.src;
    setDstImg(images.dst);
}

void JuicerProcessor::setDirectFrameRequest(const DirectFrameRequest& request) {
    setRenderWindowRect(request.renderWindow);
    _fullFrameExtent = request.fullFrameExtent;
    setComponents(request.components);
    _directStateHold = request.state;
    _recipeHold = _directStateHold
                      ? std::shared_ptr<const RenderRecipe>(_directStateHold, &_directStateHold->recipe)
                      : nullptr;
    _wsHold.reset();
    _ws = nullptr;
    _wsReady = false;
    _prt = nullptr;
    _printReady = false;
    setSessionTokens(SessionTokens{request.sessionSeed, request.instanceToken});
    setClipToken(request.clipToken);
    setFrameTime(request.frameTime);
    setFrameRate(request.frameRate);
    setPixelSizeUm(request.pixelSizeUm);
}

void JuicerProcessor::setPrintFrameRequest(const PrintFrameRequest& request) {
    setRenderWindowRect(request.renderWindow);
    _fullFrameExtent = request.fullFrameExtent;
    setComponents(request.components);
    _printStateHold = request.state;
    _recipeHold = _printStateHold
                      ? std::shared_ptr<const RenderRecipe>(_printStateHold, &_printStateHold->recipe)
                      : nullptr;
    _directStateHold.reset();
    _wsHold.reset();
    _ws = nullptr;
    _wsReady = false;
    _prt = nullptr;
    _printReady = false;
    setSessionTokens(SessionTokens{request.sessionSeed, request.instanceToken});
    setClipToken(request.clipToken);
    setFrameTime(request.frameTime);
    setFrameRate(request.frameRate);
    setPixelSizeUm(request.pixelSizeUm);
}

void JuicerProcessor::setFrameRequest(const FrameRequest& request) {
    // Broad request adapter retained for host plumbing; focused CUDA routes use typed requests.
    setRenderWindowRect(request.renderWindow);
    setComponents(request.components);
    _scannerOptions = request.scannerOptions;
    _scannerSettings = request.scannerSettings;
    _printParams = request.printParams;
    _halationOverride = request.halationOverride;
    _hasHalationOverride = request.hasHalationOverride;
    _grainOverride = request.grainOverride;
    _hasGrainOverride = request.hasGrainOverride;
    _printGlareOverride = request.printGlareOverride;
    if (request.hasPrintGlareOverride) {
        clear_glare_compensation_fields(_printGlareOverride);
    }
    _hasPrintGlareOverride = request.hasPrintGlareOverride;
    _recipeHold = request.recipe;
    _wsHold = request.workingState;
    setWorkingState(_wsHold.get(), request.workingStateReady);
    setPrintRuntime(
        request.printRt ? request.printRt : ((_ws && _ws->printRT) ? _ws->printRT.get() : nullptr),
        request.printRtReady);
    setExposure(request.exposureScale);
    CameraAutoExposureSettings autoExposureSettings{};
    autoExposureSettings.enabled = request.cameraAutoEnabled;
    autoExposureSettings.meteringMethod = request.cameraMeteringMethod;
    autoExposureSettings.sliderEV = request.cameraSliderEV;
    setCameraAutoExposure(autoExposureSettings);
    setAutoExposureMeterBounds(
        request.autoExposureMeterBounds,
        request.autoExposureMeterBoundsValid);
    _outputEncoding = request.outputEncoding;
    setSessionTokens(SessionTokens{request.sessionSeed, request.instanceToken});
    setClipToken(request.clipToken);
    setGateWeaveAmount(request.gateWeaveAmount);
    setFrameTime(request.frameTime);
    setFrameRate(request.frameRate);
    setFrameBoundsVersion(request.frameBoundsVersion);
    setPixelSizeUm(request.pixelSizeUm);
    setRenderHints(
        request.interactiveRenderStatus,
        request.renderQualityDraft,
        request.sequentialRenderStatus);
}

void JuicerProcessor::setRenderWindowRect(const OfxRectI& rect) {
    setRenderWindow(rect);
}
void JuicerProcessor::setComponents(int n) {
    _nComponents = n;
}
void JuicerProcessor::setScannerOptions(const Scanner::Options& o) {
    _scannerOptions = o;
}
void JuicerProcessor::setScannerSettings(const Scanner::Settings& s) {
    _scannerSettings = s;
}
void JuicerProcessor::setPrintParams(const Print::Params& p) {
    _printParams = p;
}
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
    clear_glare_compensation_fields(_printGlareOverride);
    _hasPrintGlareOverride = true;
}
void JuicerProcessor::setWorkingState(const WorkingState* ws, bool wsReady) {
    if (_wsHold.get() != ws) {
        _wsHold.reset();
    }
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
void JuicerProcessor::setPrintRuntime(const Print::Runtime* prt, bool printReady) {
    _prt = prt;
    _printReady = printReady;
}
void JuicerProcessor::setExposure(float exposureScale) {
    _exposureScale = positive_finite_or(exposureScale, 1.0f);
}

void JuicerProcessor::setCameraAutoExposure(const CameraAutoExposureSettings& settings) {
    _cameraAutoEnabled = settings.enabled;
    _cameraMeteringMethod = settings.meteringMethod;
    _cameraSliderEV = settings.sliderEV;
}

void JuicerProcessor::setAutoExposureMeterBounds(const OfxRectI& bounds, bool valid) {
    _autoExposureMeterBounds = bounds;
    _autoExposureMeterBoundsValid = valid;
}

void JuicerProcessor::setOutputEncoding(const OutputEncoding::Params& p) {
    _outputEncoding = p;
}

void JuicerProcessor::setInstanceState(InstanceState* s) {
    _instanceState = s;
}

void JuicerProcessor::setSessionTokens(const SessionTokens& tokens) {
    _sessionSeed = session_seed_or_default(tokens.sessionSeed);
    _instanceToken = instance_token_or_session_seed(tokens.instanceToken, _sessionSeed);
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
    _frameRate = positive_finite_or(frameRate, 0.0);
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

void JuicerProcessor::process() {
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
    // CUDA-only mode: refuse CPU/OpenCL/Metal entry points.
    if (!_isEnabledCudaRender) {
        JTRACE("CUDA", "FATAL: JUICER_CUDA_ONLY rejected non-CUDA render request");
        OFX::throwSuiteStatusException(kOfxStatErrFatal);
    }
#endif
    if (is_gpu_render_requested(_isEnabledOpenCLRender, _isEnabledCudaRender, _isEnabledMetalRender)) {
        OFX::ImageProcessor::process();
        return;
    }
    JTRACE("SPEKTRAFILM", "FATAL: SpektrafilmCpuPixelPipelineNotImplementedForPhase3C at JuicerProcessor::process");
    OFX::throwSuiteStatusException(kOfxStatErrFatal);
}

void JuicerProcessor::processImagesCUDA() {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
    JTRACE("CUDA", "FATAL: CUDA render requested but the CUDA backend is unavailable in this build");
    OFX::throwSuiteStatusException(kOfxStatErrFatal);
#else
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
            const char* msg = detail_or_unknown(cudaGetErrorString(setErr));
            trace_cuda_fatal_prefixed_if(
                traceInfo,
                "cudaSetDevice failed",
                msg);
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }

        std::string contextError;
        if (!query_current_cuda_context(contextOpaque, contextError)) {
            trace_cuda_fatal_prefixed_if(
                traceInfo,
                "failed to capture CUDA context identity",
                cstr_or_null_if_empty(contextError));
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
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
    const DirectRenderPayload* directPayload =
        directRecipe ? &_directStateHold->payload : nullptr;
    const RenderRecipe* printRecipe =
        (_printStateHold &&
         Spektrafilm::scan_route_is_print(_printStateHold->recipe.profileRoute.scanRoute))
            ? &_printStateHold->recipe
            : nullptr;
    const PrintRenderPayload* printPayload =
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

    // Highlight boost remains film-exposure owned. Any adjacent Exact optics request is
    // descriptor-resolved and blocked before CUDA context/resource preparation.
    Spektrafilm::ExactOpticsExecutionPlan exactOpticsPlan{};
    const Spektrafilm::ExactOpticsFrameExtent exactFullFrameExtent{
        _fullFrameExtent.x2 - _fullFrameExtent.x1,
        _fullFrameExtent.y2 - _fullFrameExtent.y1};
    if (!Spektrafilm::build_exact_optics_execution_plan(
            focusedRecipe->spatialOptics,
            focusedRecipe->profileRoute.scanRoute,
            _pixelSizeUm,
            exactFullFrameExtent,
            exactOpticsPlan)) {
        JTRACE("SPEKTRAFILM", "ResourceDescriptorMismatch phase=6A field=exact_optics_execution_plan");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
    if (!exactOpticsPlan.blockingDiagnostic.empty()) {
        JTRACE("SPEKTRAFILM", exactOpticsPlan.blockingDiagnostic);
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
        recover_context_loss_state(
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

    auto record_cuda_use = [&](JuicerProcess::Root::PreparedCudaFrame& frame) {
        frame.record_use(_pCudaStream);
    };

    auto throw_cuda_policy_fatal = [&](const char* failurePrefix = nullptr,
                                       const char* detail = nullptr) {
        trace_cuda_fatal_prefixed_if(
            traceInfo,
            nonempty_cstr_or(failurePrefix, "CUDA render cannot continue"),
            detail);
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    };

    auto throw_cuda_stage_fatal = [&](const char* stageTag,
                                      const char* failurePrefix,
                                      cudaError_t errorCode) {
        const char* errorMsg = cudaGetErrorString(errorCode);
        const char* detail = detail_or_unknown(errorMsg);
        mark_context_loss_recovery(nonempty_cstr_or(stageTag, "cuda_stage"), errorCode, detail);
        throw_cuda_policy_fatal(nonempty_cstr_or(failurePrefix, "CUDA stage failed"), detail);
    };

    auto trace_and_throw_cuda_policy_fatal = [&](const char* prefix, const char* detail) {
        throw_cuda_policy_fatal(prefix, detail);
    };

    auto throw_submission_fatal = [&](const char* stageTag,
                                      const char* failurePrefix,
                                      const std::string& error) {
        mark_context_loss_recovery(
            nonempty_cstr_or(stageTag, "submission_stage"),
            cudaErrorUnknown,
            error);
        trace_cuda_fatal_prefixed_if(
            traceInfo,
            nonempty_cstr_or(failurePrefix, "submission stage failed"),
            cstr_or_null_if_empty(error));
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    };

    auto copy_scan_tables_payload = [&](auto& dstTables,
                                        auto& dstMediumIsNegative,
                                        float* dstMinCmy,
                                        float* dstInvMaxCmy,
                                        const JuicerCuda::Resources::DeviceScanMedium& scanMedium) {
        dstTables.epsC = scanMedium.tables.epsC;
        dstTables.epsM = scanMedium.tables.epsM;
        dstTables.epsY = scanMedium.tables.epsY;
        dstTables.Ax = scanMedium.tables.Ax;
        dstTables.Ay = scanMedium.tables.Ay;
        dstTables.Az = scanMedium.tables.Az;
        dstTables.baseDensityMin = scanMedium.tables.baseDensityMin;
        dstTables.K = scanMedium.tables.K;
        dstTables.hasBaseline = scanMedium.tables.hasBaseline;
        dstTables.invYn = scanMedium.tables.invYn;
        dstMediumIsNegative = scanMedium.mediumIsNegative;
        copy_float3(dstMinCmy, scanMedium.min_cmy);
        copy_float3(dstInvMaxCmy, scanMedium.inv_max_cmy);
    };

    if (should_abort_effect()) {
        return;
    }

    const FilmRawRecipe* focusedFilmRaw = &focusedRecipe->filmRaw;
    const bool cameraAutoEnabled =
        focusedFilmRaw->autoExposureEnabled;
    const Spektrafilm::AutoExposureMethod cameraMeteringMethod =
        focusedFilmRaw->autoExposureMethod;
    OfxRectI meterBounds = srcBounds;
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
    const JuicerCuda::AutoExposurePreviewDescriptor autoExposureDescriptor =
        make_auto_exposure_preview_descriptor(srcBounds, meterBounds, cameraMeteringMethod);
    JuicerProcess::Root::AutoExposureBufferRequest autoExposureBufferRequest{};
    autoExposureBufferRequest.enabled =
        cameraAutoEnabled &&
        (_nComponents == 3 || _nComponents == 4) &&
        autoExposureDescriptor.previewWidth > 0 &&
        autoExposureDescriptor.previewHeight > 0;
    autoExposureBufferRequest.descriptor = autoExposureDescriptor;
    autoExposureBufferRequest.reusableKeyHash = autoExposureDescriptor.hash;

    JuicerCuda::ResourceManager::SubmissionSnapshot snapshot{};
    {
        snapshot.instanceToken.value = instance_token_or_session_seed(_instanceToken, _sessionSeed);
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
        snapshot.keySchemaVersion = JuicerCuda::ResourceManager::kSubmissionKeySchemaVersion;
        snapshot.traceSchemaVersion = JuicerCuda::ResourceManager::kTraceSchemaVersion;
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
                latched.keySchemaVersion == snapshot.keySchemaVersion &&
                latched.traceSchemaVersion == snapshot.traceSchemaVersion &&
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

    // Phase 3C validation-only direct launch. All inactive print/DIR/optics/grain payloads stay
    // value-initialized, and direct preparation never resolves the static-noise resource family.
    if (directRecipe) {
        auto throw_direct_restriction = [&](const char* diagnostic) {
            trace_and_throw_cuda_policy_fatal("CUDA direct route blocked", diagnostic);
        };
        if (!directRecipe->directStructuralReady) {
            throw_direct_restriction("DirectStructuralRecipeNotReadyForPhase3C");
        }
        if (cameraMeteringMethod == Spektrafilm::AutoExposureMethod::Median) {
            throw_direct_restriction(Spektrafilm::kQuantizedMedianNotAcceptedForPhase3);
        }
        if (!(_nComponents == 3 || _nComponents == 4)) {
            throw_direct_restriction("UnsupportedDirectComponentCountForPhase3C");
        }
        JTRACE("PHASE3D", "direct_negative branch accepted; restrictions=clear");
        const bool dirProfileEnabled = spatial_dir_profile_enabled();

        Scanner::ScannerSpectralLutDescriptor scannerDescriptor{};
        std::string scannerDescriptorDiagnostic;
        Scanner::DirectScannerSpectralLutDescriptorInput scannerDescriptorInput{};
        scannerDescriptorInput.profileRoute = &directRecipe->profileRoute;
        scannerDescriptorInput.densityBounds = &directRecipe->densityBounds;
        scannerDescriptorInput.scannerOutput = &directRecipe->scannerOutput;
        if (!Scanner::build_direct_scanner_spectral_lut_descriptor(
                scannerDescriptorInput,
                scannerDescriptor,
                scannerDescriptorDiagnostic)) {
            throw_direct_restriction(scannerDescriptorDiagnostic.c_str());
        }
        Scanner::ScannerColorCorrectionDescriptor scannerCorrection{};
        if (!Scanner::build_direct_scanner_color_correction_descriptor(
                *directRecipe,
                directPayload->scannerTables,
                scannerCorrection,
                scannerDescriptorDiagnostic)) {
            throw_direct_restriction(scannerDescriptorDiagnostic.c_str());
        }
        Scanner::ScannerPostEffectsDescriptor scannerPostEffects{};
        if (!Scanner::build_scanner_post_effects_descriptor(
                directRecipe->scannerOutput,
                scannerPostEffects,
                scannerDescriptorDiagnostic)) {
            throw_direct_restriction(scannerDescriptorDiagnostic.c_str());
        }
        Spektrafilm::SpatialDirDescriptor directSpatialDir{};
        const auto directSpatialDirDescriptorStart = std::chrono::steady_clock::now();
        const OfxRectI directFullFrameRect =
            (_fullFrameExtent.x2 > _fullFrameExtent.x1 && _fullFrameExtent.y2 > _fullFrameExtent.y1)
                ? _fullFrameExtent
                : srcBounds;
        if (!Spektrafilm::build_spatial_dir_descriptor(
                directRecipe->dirCouplers,
                _pixelSizeUm,
                spatial_dir_extent_from_rect(win),
                spatial_dir_extent_from_rect(directFullFrameRect),
                "direct",
                directSpatialDir)) {
            trace_spatial_dir_descriptor_build("direct", directSpatialDir);
            throw_direct_restriction(
                "ResourceDescriptorMismatch phase=3D-3 field=spatial_dir_descriptor");
        }
        trace_spatial_dir_descriptor_build("direct", directSpatialDir);
        const double directSpatialDirDescriptorMs =
            dirProfileEnabled ? elapsed_ms_since(directSpatialDirDescriptorStart) : 0.0;

        JuicerProcess::Root::DirectCudaPreparationRequest directPreparation{};
        directPreparation.recipe = directRecipe;
        directPreparation.exposureTables = &directPayload->exposureTables;
        directPreparation.spdSInv = directPayload->spdSInv.data();
        directPreparation.filmRawConfig = &directPayload->filmRawConfig;
        directPreparation.scannerTables = &directPayload->scannerTables;
        directPreparation.scannerColor = &directPayload->scannerColor;
        directPreparation.scannerLutDescriptor = &scannerDescriptor;
        directPreparation.scannerPostEffects = &scannerPostEffects;
        directPreparation.spatialDirDescriptor = &directSpatialDir;
        directPreparation.scannerWorkspaceNeedsSpatialDir = directSpatialDir.hash != 0;
        directPreparation.frameWidth = width;
        directPreparation.frameHeight = height;

        std::string directPrepareError;
        JuicerProcess::Root::PreparedCudaFrame preparedFrame =
            JuicerProcess::root().prepare_cuda_frame(
                deviceContextKey,
                snapshot,
                directPreparation,
                autoExposureBufferRequest,
                _pCudaStream,
                directPrepareError);
        if (!preparedFrame.active()) {
            throw_submission_fatal(
                preparedFrame.failure_stage_tag(),
                preparedFrame.failure_prefix(),
                directPrepareError);
        }

        const JuicerProcess::Root::PreparedCudaFrame::DirectPreparedView prepared =
            preparedFrame.direct_resources();
        if (!prepared.active ||
            prepared.densityBoundsHash != directRecipe->densityBounds.hash ||
            prepared.scannerDescriptorHash != scannerDescriptor.hash ||
            prepared.selectedMethod != directRecipe->filmRaw.rgbToRawMethod) {
            preparedFrame.abort("direct_prepared_view_descriptor_mismatch");
            throw_direct_restriction("ResourceDescriptorMismatch phase=3C field=direct_prepared_view");
        }

        JuicerCuda::DirectPipelineRunParams run{};
        run.src = srcPtr;
        run.srcRowBytes = static_cast<std::size_t>(srcRowBytes);
        run.dst = dstPtr;
        run.dstRowBytes = static_cast<std::size_t>(dstRowBytes);
        run.width = width;
        run.height = height;
        run.nComponents = _nComponents;

        const int fullSourceWidth = srcBounds.x2 - srcBounds.x1;
        const int fullSourceHeight = srcBounds.y2 - srcBounds.y1;
        (void)fullSourceWidth;
        (void)fullSourceHeight;
        const float* autoExposureScaleDevice = nullptr;
        if (cameraAutoEnabled) {
            const auto buffers = preparedFrame.auto_exposure_buffers();
            if (!buffers.active) {
                preparedFrame.abort("direct_auto_exposure_buffers_missing");
                throw_direct_restriction("MissingRequiredResource phase=3C field=auto_exposure_buffers");
            }
            const char* meterError = nullptr;
            if (cameraMeteringMethod == Spektrafilm::AutoExposureMethod::CenterWeighted &&
                (buffers.weightsWidth != autoExposureDescriptor.previewWidth ||
                 buffers.weightsHeight != autoExposureDescriptor.previewHeight)) {
                const int weightsRc = juicer_cuda_auto_exposure_build_center_weight_tables(
                    autoExposureDescriptor.previewWidth,
                    autoExposureDescriptor.previewHeight,
                    buffers.scratch.weightsX,
                    buffers.scratch.weightsY,
                    _pCudaStream,
                    &meterError);
                if (weightsRc != 0) {
                    preparedFrame.abort("direct_auto_exposure_weights_failed");
                    throw_direct_restriction(detail_or_unknown(meterError));
                }
                preparedFrame.mark_auto_exposure_weights_built(
                    JuicerProcess::Root::PreparedCudaFrame::AutoExposureWeightsExtent{
                        autoExposureDescriptor.previewWidth,
                        autoExposureDescriptor.previewHeight});
            }
            const int meterRc = juicer_cuda_auto_exposure_meter_to_device(
                srcBase,
                static_cast<std::size_t>(srcRowBytes),
                autoExposureDescriptor,
                run.nComponents,
                directRecipe->filmRaw.inputColorSpace,
                bool_to_i32(directRecipe->filmRaw.inputCctfDecoding),
                prepared.film.inputRGBToXYZ,
                buffers.scratch,
                buffers.deviceState,
                _pCudaStream,
                &meterError);
            if (meterRc != 0) {
                preparedFrame.abort("direct_auto_exposure_meter_failed");
                throw_direct_restriction(detail_or_unknown(meterError));
            }
            preparedFrame.mark_auto_exposure_metered(
                JuicerProcess::Root::PreparedCudaFrame::AutoExposureMeteredResult{
                    autoExposureDescriptor.hash});
            autoExposureScaleDevice = buffers.deviceState.exposureScale;
        }

        JuicerCuda::DirectFilmPayloadPack directFilmPayloads{};
        std::string packDiagnostic;
        if (!JuicerCuda::pack_direct_film_payloads(
                directRecipe->filmRaw,
                directRecipe->filmDevelop,
                directRecipe->dirCouplers,
                directRecipe->grainContract,
                directRecipe->densityBounds,
                prepared.film,
                autoExposureScaleDevice,
                directFilmPayloads,
                packDiagnostic)) {
            preparedFrame.abort("direct_film_payload_pack_failed");
            throw_direct_restriction(packDiagnostic.c_str());
        }
        run.filmRaw = directFilmPayloads.filmRaw;
        run.filmExpose = directFilmPayloads.filmExposure;
        run.filmExpose.routeCorrectionScale = scannerCorrection.exposureScale;
        run.filmDevelop = directFilmPayloads.filmDevelop;

        JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker focusedWorkspace{};
        if (directSpatialDir.hash != 0 || scannerPostEffects.active()) {
            JuicerProcess::Root::PreparedCudaFrame::WorkspaceRequest request{};
            bind_spatial_dir_workspace_metadata(directSpatialDir, request);
            request.needOptics = scannerPostEffects.active();
            request.requestedWidth = width;
            request.requestedHeight = height;
            request.needBlurred =
                scannerPostEffects.glareActive && scannerPostEffects.glareBlurSigmaPx > 0.0f;
            focusedWorkspace = preparedFrame.bind_workspace_request(request);
        }
        JuicerCuda::SpatialDirBuildProfile directDirProfile{};
        bool directDirProfileCaptured = false;
        bool directDirScratchOverflow = false;
        double directDirPrepareMs = 0.0;
        double directDirBuildHostMs = 0.0;
        double directPipelineLaunchHostMs = 0.0;
        float directPipelineCudaMs = -1.0f;
        if (directSpatialDir.hash != 0) {
            std::string spatialError;
            const auto directDirPrepareStart = std::chrono::steady_clock::now();
            if (!preparedFrame.prepare_spatial_dir_resources(
                    directSpatialDir,
                    focusedWorkspace,
                    _pCudaStream,
                    spatialError)) {
                preparedFrame.abort("direct_spatial_dir_prepare_failed");
                throw_submission_fatal(
                    "direct_spatial_dir_prepare",
                    "direct spatial DIR preparation failed",
                    spatialError);
            }
            if (dirProfileEnabled) {
                directDirPrepareMs = elapsed_ms_since(directDirPrepareStart);
            }
            const auto scratch = preparedFrame.spatial_dir_scratch(focusedWorkspace);
            const auto resources =
                preparedFrame.spatial_dir_resources(focusedWorkspace, directSpatialDir.hash);
            if (!scratch.active || !resources.active) {
                preparedFrame.abort("direct_spatial_dir_binding_failed");
                throw_direct_restriction(
                    "MissingRequiredResource phase=3D-3 field=prepared_spatial_dir");
            }
            SF_TEMP_BRIDGE_bind_filtered_correction_to_payload(run.filmDevelop, scratch);
            // SF_TEMP_BRIDGE_map_legacy_corr_planes_to_rawCorrection: Phase 5 removes
            // these old wrapper arguments after source/filter/final-develop split lands.
            float* SF_TEMP_BRIDGE_map_legacy_corr_planes_to_rawCorrectionY = scratch.rawCorrectionY;
            float* SF_TEMP_BRIDGE_map_legacy_corr_planes_to_rawCorrectionM = scratch.rawCorrectionM;
            float* SF_TEMP_BRIDGE_map_legacy_corr_planes_to_rawCorrectionC = scratch.rawCorrectionC;
            // SF_TEMP_BRIDGE_map_legacy_mix_planes_to_filteredCorrection: Phase 5 removes this alias.
            float* SF_TEMP_BRIDGE_map_legacy_mix_planes_to_filteredCorrectionY = scratch.filteredCorrectionY;
            float* SF_TEMP_BRIDGE_map_legacy_mix_planes_to_filteredCorrectionM = scratch.filteredCorrectionM;
            float* SF_TEMP_BRIDGE_map_legacy_mix_planes_to_filteredCorrectionC = scratch.filteredCorrectionC;
            float* SF_TEMP_BRIDGE_map_legacy_tmp_plane_to_filterTemp = scratch.filterTemp;
            float* SF_TEMP_BRIDGE_map_legacy_iir_forward_plane_to_iirForwardTemp =
                scratch.iirForwardTemp;
            directDirScratchOverflow = scratch.overflow;
            const auto directDirBuildStart = std::chrono::steady_clock::now();
            trace_spatial_dir_bridge_use(
                "direct",
                "SF_TEMP_BRIDGE_build_direct_spatial_dir",
                directSpatialDir);
            const cudaError_t dirError = juicer_cuda_build_direct_spatial_dir(
                &run,
                SF_TEMP_BRIDGE_map_legacy_corr_planes_to_rawCorrectionY,
                SF_TEMP_BRIDGE_map_legacy_corr_planes_to_rawCorrectionM,
                SF_TEMP_BRIDGE_map_legacy_corr_planes_to_rawCorrectionC,
                SF_TEMP_BRIDGE_map_legacy_mix_planes_to_filteredCorrectionY,
                SF_TEMP_BRIDGE_map_legacy_mix_planes_to_filteredCorrectionM,
                SF_TEMP_BRIDGE_map_legacy_mix_planes_to_filteredCorrectionC,
                SF_TEMP_BRIDGE_map_legacy_tmp_plane_to_filterTemp,
                SF_TEMP_BRIDGE_map_legacy_iir_forward_plane_to_iirForwardTemp,
                resources.gaussian.weights,
                resources.gaussian.radius,
                resources.gaussian.sigma,
                directSpatialDir.gaussianWeight,
                resources.exponential[0].weights,
                resources.exponential[0].radius,
                resources.exponential[0].sigma,
                directSpatialDir.exponentialWeights[0],
                resources.exponential[1].weights,
                resources.exponential[1].radius,
                resources.exponential[1].sigma,
                directSpatialDir.exponentialWeights[1],
                resources.exponential[2].weights,
                resources.exponential[2].radius,
                resources.exponential[2].sigma,
                directSpatialDir.exponentialWeights[2],
                _pCudaStream,
                dirProfileEnabled ? &directDirProfile : nullptr);
            if (dirProfileEnabled) {
                directDirBuildHostMs = elapsed_ms_since(directDirBuildStart);
            }
            if (dirError != cudaSuccess) {
                preparedFrame.abort("direct_spatial_dir_launch_failed");
                throw_cuda_stage_fatal(
                    "direct_spatial_dir_launch",
                    "direct spatial DIR build failed",
                    dirError);
            }
            directDirProfileCaptured = dirProfileEnabled;
        }

        const JuicerCuda::Resources::DeviceScanMedium& scanMedium = *prepared.scanMedium;
        copy_scan_tables_payload(
            run.scanStage.scanTables,
            run.scanStage.scanTables.mediumIsNegative,
            run.scanStage.scanTables.min_cmy,
            run.scanStage.scanTables.inv_max_cmy,
            scanMedium);
        run.scanStage.scannerUseLut = 1;
        run.scanStage.scanLutLog10XYZ = prepared.scanLut->log10XYZ;
        run.scanStage.scanLutSlopeC = prepared.scanLut->slopeC;
        run.scanStage.scanLutSlopeM = prepared.scanLut->slopeM;
        run.scanStage.scanLutSlopeY = prepared.scanLut->slopeY;
        run.scanStage.scanLutCellMin = prepared.scanLut->cellMin;
        run.scanStage.scanLutCellMax = prepared.scanLut->cellMax;
        run.scanStage.scanLutRes = static_cast<int>(prepared.scanLut->res);
        const Scanner::ColorRuntime& color = *prepared.scannerColor;
        copy_float9(run.scanStage.scanColor.cat02, color.cat02);
        copy_float9(run.scanStage.scanColor.xyzToRgb, color.xyzToRgb);
        copy_float3(run.scanStage.scanColor.illuminantXYZ, color.illuminantXYZ);
        run.scanStage.scanColor.encoding.outputColorSpaceIndex = OutputEncoding::toIndex(color.encoding.colorSpace);
        run.scanStage.scanColor.encoding.applyCctfEncoding = bool_to_i32(color.encoding.applyCctfEncoding);
        run.scanStage.scanColor.encoding.preserveLinearRange = bool_to_i32(color.encoding.preserveLinearRange);
        run.scanStage.scanColor.encoding.inputIsOutputSpace = bool_to_i32(color.encoding.inputIsOutputSpace);
        const auto& outputSpace = GeneratedColorSpaces::get(color.encoding.colorSpace);
        run.scanStage.scanColor.encoding.cctf.kind = static_cast<int>(outputSpace.cctf.kind);
        run.scanStage.scanColor.encoding.cctf.gamma = outputSpace.cctf.gamma;
        run.scanStage.scanColor.encoding.cctf.a = outputSpace.cctf.a;
        run.scanStage.scanColor.encoding.cctf.b = outputSpace.cctf.b;
        run.scanStage.scanColor.encoding.cctf.c = outputSpace.cctf.c;
        run.scanStage.scanColor.encoding.cctf.d = outputSpace.cctf.d;
        run.scanStage.scanColor.encoding.cctf.linearCutoff = outputSpace.cctf.linearCutoff;
        const OutputEncoding::Matrix3x3 dwgToOutput =
            OutputEncoding::dwg_to_output_matrix(color.encoding.colorSpace);
        copy_float9(run.scanStage.scanColor.encoding.dwgToOutput, dwgToOutput.m);
        run.scanStage.correctionActive = scannerCorrection.active ? 1 : 0;
        run.scanStage.correctionSlope = scannerCorrection.xyzSlope;
        run.scanStage.correctionOffset = scannerCorrection.xyzOffset;

        std::string scanError;
        if (!preparedFrame.prepare_scan_error_stage(run.scanStage.scanErrorFlag, _pCudaStream, scanError)) {
            preparedFrame.abort("direct_scan_error_stage_failed");
            throw_submission_fatal("direct_scan_error_stage", "direct scan error stage failed", scanError);
        }
        cudaError_t launchError = cudaSuccess;
        const auto directPipelineLaunchStart = std::chrono::steady_clock::now();
        CudaEventElapsedTimer directPipelineCudaTimer;
        if (dirProfileEnabled) {
            directPipelineCudaTimer.begin(_pCudaStream);
        }
        if (scannerPostEffects.active()) {
            const auto post = preparedFrame.scanner_post_effects_resources(
                focusedWorkspace,
                scannerPostEffects.hash);
            if (!post.active) {
                preparedFrame.abort("direct_scanner_post_effects_view_failed");
                throw_direct_restriction(
                    "MissingRequiredResource phase=8C field=scanner_post_effects");
            }
            launchError = juicer_cuda_direct_focused_scanner_post_effects(
                &run,
                post.scratch.rgbR,
                post.scratch.rgbG,
                post.scratch.rgbB,
                post.scratch.tmp,
                post.scratch.blurred,
                post.lensBlur.weights,
                post.lensBlur.radius,
                post.unsharp.weights,
                post.unsharp.radius,
                scannerPostEffects.unsharpAmount,
                _pCudaStream);
        } else {
            launchError = juicer_cuda_negative_direct_pipeline(&run, _pCudaStream);
        }
        if (dirProfileEnabled) {
            directPipelineLaunchHostMs = elapsed_ms_since(directPipelineLaunchStart);
            if (launchError == cudaSuccess) {
                directPipelineCudaMs = directPipelineCudaTimer.finish(_pCudaStream);
            }
        }
        if (launchError != cudaSuccess) {
            preparedFrame.abort("direct_negative_pipeline_launch_failed");
            throw_cuda_stage_fatal("direct_negative_pipeline_launch", "direct negative pipeline launch failed", launchError);
        }
        if (dirProfileEnabled) {
            trace_spatial_dir_profile(
                "direct",
                width,
                height,
                directSpatialDir,
                directDirProfile,
                directDirProfileCaptured,
                directDirScratchOverflow,
                directSpatialDirDescriptorMs,
                directDirPrepareMs,
                directDirBuildHostMs,
                directPipelineCudaMs,
                directPipelineLaunchHostMs);
        }
        JTRACE("PHASE3D", "direct_negative kernel launch accepted");
        if (!preparedFrame.finalize_scan_error_stage(run.scanStage.scanErrorFlag, _pCudaStream, scanError)) {
            preparedFrame.abort("direct_scan_error_finalize_failed");
            throw_submission_fatal("direct_scan_error_finalize", "direct scan error finalize failed", scanError);
        }
        record_cuda_use(preparedFrame);
        std::string finishError;
        if (!preparedFrame.finish(_pCudaStream, finishError)) {
            throw_submission_fatal("direct_prepared_frame_finish", "direct prepared frame finish failed", finishError);
        }
        JTRACE("PHASE3D", "direct_negative destination submission completed");
        JuicerCuda::LaunchGraphCounters::record_frame_completed();
        return;
    }

    if (printRecipe) {
        auto throw_print_restriction = [&](const char* diagnostic) {
            trace_and_throw_cuda_policy_fatal("CUDA print route blocked", diagnostic);
        };
        if (!printRecipe->printStructuralReady) {
            throw_print_restriction("PrintStructuralRecipeNotReadyForPhase4C");
        }
        if (!(_nComponents == 3 || _nComponents == 4)) {
            throw_print_restriction("UnsupportedPrintComponentCountForPhase4C");
        }

        Scanner::ScannerSpectralLutDescriptor scannerDescriptor{};
        std::string scannerDescriptorDiagnostic;
        if (!Scanner::build_print_scanner_spectral_lut_descriptor(
                Scanner::PrintScannerSpectralLutDescriptorInput{
                    &printRecipe->profileRoute,
                    &printRecipe->densityBounds,
                    &printRecipe->scannerOutput,
                    &printRecipe->print.mediumHandoff},
                scannerDescriptor,
                scannerDescriptorDiagnostic)) {
            throw_print_restriction(scannerDescriptorDiagnostic.c_str());
        }
        Scanner::ScannerPostEffectsDescriptor scannerPostEffects{};
        if (!Scanner::build_scanner_post_effects_descriptor(
                printRecipe->scannerOutput,
                scannerPostEffects,
                scannerDescriptorDiagnostic)) {
            throw_print_restriction(scannerDescriptorDiagnostic.c_str());
        }
        const bool dirProfileEnabled = spatial_dir_profile_enabled();
        Spektrafilm::SpatialDirDescriptor spatialDir{};
        const auto spatialDirDescriptorStart = std::chrono::steady_clock::now();
        const OfxRectI printFullFrameRect =
            (_fullFrameExtent.x2 > _fullFrameExtent.x1 && _fullFrameExtent.y2 > _fullFrameExtent.y1)
                ? _fullFrameExtent
                : srcBounds;
        if (!Spektrafilm::build_spatial_dir_descriptor(
                printRecipe->dirCouplers,
                _pixelSizeUm,
                spatial_dir_extent_from_rect(win),
                spatial_dir_extent_from_rect(printFullFrameRect),
                "print",
                spatialDir)) {
            trace_spatial_dir_descriptor_build("print", spatialDir);
            throw_print_restriction(
                "ResourceDescriptorMismatch phase=4C field=spatial_dir_descriptor");
        }
        trace_spatial_dir_descriptor_build("print", spatialDir);
        const double spatialDirDescriptorMs =
            dirProfileEnabled ? elapsed_ms_since(spatialDirDescriptorStart) : 0.0;

        JuicerProcess::Root::PrintCudaPreparationRequest preparation{};
        preparation.recipe = printRecipe;
        preparation.exposureTables = &printPayload->exposureTables;
        preparation.spdSInv = printPayload->spdSInv.data();
        preparation.filmRawConfig = &printPayload->filmRawConfig;
        preparation.scannerTables = &printPayload->scannerTables;
        preparation.scannerColor = &printPayload->scannerColor;
        preparation.scannerLutDescriptor = &scannerDescriptor;
        preparation.scannerPostEffects = &scannerPostEffects;
        preparation.spatialDirDescriptor = &spatialDir;
        preparation.scannerWorkspaceNeedsSpatialDir = spatialDir.hash != 0;
        preparation.frameWidth = width;
        preparation.frameHeight = height;

        std::string prepareError;
        JuicerProcess::Root::PreparedCudaFrame preparedFrame =
            JuicerProcess::root().prepare_cuda_frame(
                deviceContextKey,
                snapshot,
                preparation,
                autoExposureBufferRequest,
                _pCudaStream,
                prepareError);
        if (!preparedFrame.active()) {
            throw_submission_fatal(
                preparedFrame.failure_stage_tag(),
                preparedFrame.failure_prefix(),
                prepareError);
        }

        const JuicerProcess::Root::PreparedCudaFrame::PrintRoutePreparedView prepared =
            preparedFrame.print_route_resources();
        const JuicerProcess::Root::PreparedCudaFrame::PrintPreparedView preparedPrint =
            preparedFrame.print_resources();
        if (!prepared.active || !preparedPrint.active ||
            prepared.densityBoundsHash != printRecipe->densityBounds.hash ||
            prepared.scannerDescriptorHash != scannerDescriptor.hash ||
            prepared.selectedMethod != printRecipe->filmRaw.rgbToRawMethod) {
            preparedFrame.abort("print_prepared_view_descriptor_mismatch");
            throw_print_restriction(
                "ResourceDescriptorMismatch phase=4C field=print_prepared_view");
        }
        Scanner::ScannerColorCorrectionDescriptor scannerCorrection{};
        Scanner::PrintCorrectionDerivationInput correctionInput{};
        correctionInput.recipe = printRecipe;
        correctionInput.scannerTables = &printPayload->scannerTables;
        correctionInput.mainIlluminant = preparedPrint.mainIlluminantHost;
        correctionInput.spectralSampleCount = preparedPrint.spectralSampleCount;
        correctionInput.preflashRawCmy = preparedPrint.preflashRawCmy;
        correctionInput.normalizer = preparedPrint.normalizer;
        if (!Scanner::build_print_scanner_color_correction_descriptor(
                correctionInput,
                scannerCorrection,
                scannerDescriptorDiagnostic)) {
            preparedFrame.abort("print_scanner_correction_descriptor_failed");
            throw_print_restriction(scannerDescriptorDiagnostic.c_str());
        }
        if (traceVerbose) {
            const PrintFilterRecipe& filters = printRecipe->print.filters;
            std::string msg;
            msg.reserve(512);
            msg = "event=focused_route_prepared recipeHash=";
            msg += std::to_string(printRecipe->hash);
            msg += " uiYmcCc(Y/M/C)=";
            msg += std::to_string(filters.userCmyCc.y);
            msg += "/";
            msg += std::to_string(filters.userCmyCc.m);
            msg += "/";
            msg += std::to_string(filters.userCmyCc.c);
            msg += " neutralCmyCc(C/M/Y)=";
            msg += std::to_string(filters.neutralCmyCc.c);
            msg += "/";
            msg += std::to_string(filters.neutralCmyCc.m);
            msg += "/";
            msg += std::to_string(filters.neutralCmyCc.y);
            msg += " userCmyCc(C/M/Y)=";
            msg += std::to_string(filters.userCmyCc.c);
            msg += "/";
            msg += std::to_string(filters.userCmyCc.m);
            msg += "/";
            msg += std::to_string(filters.userCmyCc.y);
            msg += " mainCmyCc(C/M/Y)=";
            msg += std::to_string(filters.mainCmyCc.c);
            msg += "/";
            msg += std::to_string(filters.mainCmyCc.m);
            msg += "/";
            msg += std::to_string(filters.mainCmyCc.y);
            msg += " filterRecipeHash=";
            msg += std::to_string(filters.hash);
            msg += " filteredMainIlluminantDescriptorHash=";
            msg += std::to_string(preparedPrint.mainIlluminantHash);
            msg += " balanceHash=";
            msg += std::to_string(preparedPrint.balanceHash);
            msg += " preparationHash=";
            msg += std::to_string(preparedPrint.preparationHash);
            JTRACE_VERBOSE("PHASE4C_BALANCE", msg);
        }

        JuicerCuda::PrintPipelineRunParams run{};
        run.src = srcPtr;
        run.srcRowBytes = static_cast<std::size_t>(srcRowBytes);
        run.dst = dstPtr;
        run.dstRowBytes = static_cast<std::size_t>(dstRowBytes);
        run.width = width;
        run.height = height;
        run.nComponents = _nComponents;

        const float* autoExposureScaleDevice = nullptr;
        if (cameraAutoEnabled) {
            const auto buffers = preparedFrame.auto_exposure_buffers();
            if (!buffers.active) {
                preparedFrame.abort("print_auto_exposure_buffers_missing");
                throw_print_restriction(
                    "MissingRequiredResource phase=4C field=auto_exposure_buffers");
            }
            const char* meterError = nullptr;
            if (cameraMeteringMethod == Spektrafilm::AutoExposureMethod::CenterWeighted &&
                (buffers.weightsWidth != autoExposureDescriptor.previewWidth ||
                 buffers.weightsHeight != autoExposureDescriptor.previewHeight)) {
                const int weightsRc = juicer_cuda_auto_exposure_build_center_weight_tables(
                    autoExposureDescriptor.previewWidth,
                    autoExposureDescriptor.previewHeight,
                    buffers.scratch.weightsX,
                    buffers.scratch.weightsY,
                    _pCudaStream,
                    &meterError);
                if (weightsRc != 0) {
                    preparedFrame.abort("print_auto_exposure_weights_failed");
                    throw_print_restriction(detail_or_unknown(meterError));
                }
                preparedFrame.mark_auto_exposure_weights_built(
                    JuicerProcess::Root::PreparedCudaFrame::AutoExposureWeightsExtent{
                        autoExposureDescriptor.previewWidth,
                        autoExposureDescriptor.previewHeight});
            }
            const int meterRc = juicer_cuda_auto_exposure_meter_to_device(
                srcBase,
                static_cast<std::size_t>(srcRowBytes),
                autoExposureDescriptor,
                run.nComponents,
                printRecipe->filmRaw.inputColorSpace,
                bool_to_i32(printRecipe->filmRaw.inputCctfDecoding),
                prepared.film.inputRGBToXYZ,
                buffers.scratch,
                buffers.deviceState,
                _pCudaStream,
                &meterError);
            if (meterRc != 0) {
                preparedFrame.abort("print_auto_exposure_meter_failed");
                throw_print_restriction(detail_or_unknown(meterError));
            }
            preparedFrame.mark_auto_exposure_metered(
                JuicerProcess::Root::PreparedCudaFrame::AutoExposureMeteredResult{
                    autoExposureDescriptor.hash});
            autoExposureScaleDevice = buffers.deviceState.exposureScale;
        }

        JuicerCuda::DirectFilmPayloadPack filmPayloads{};
        std::string payloadDiagnostic;
        if (!JuicerCuda::pack_direct_film_payloads(
                printRecipe->filmRaw,
                printRecipe->filmDevelop,
                printRecipe->dirCouplers,
                printRecipe->grainContract,
                printRecipe->enlargerFilmBounds,
                prepared.film,
                autoExposureScaleDevice,
                filmPayloads,
                payloadDiagnostic)) {
            preparedFrame.abort("print_film_payload_pack_failed");
            throw_print_restriction(payloadDiagnostic.c_str());
        }
        JuicerCuda::PrintCudaPayloadPack printPayloads{};
        if (!JuicerCuda::pack_print_cuda_payloads(
                printRecipe->print,
                preparedPrint,
                printPayloads,
                payloadDiagnostic)) {
            preparedFrame.abort("print_payload_pack_failed");
            throw_print_restriction(payloadDiagnostic.c_str());
        }
        run.filmRaw = filmPayloads.filmRaw;
        run.filmExpose = filmPayloads.filmExposure;
        run.filmDevelop = filmPayloads.filmDevelop;
        run.printExpose = printPayloads.expose;
        run.printExpose.routeCorrectionScale = scannerCorrection.exposureScale;
        run.printDevelop = printPayloads.develop;

        JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker focusedWorkspace{};
        if (spatialDir.hash != 0 || scannerPostEffects.active()) {
            JuicerProcess::Root::PreparedCudaFrame::WorkspaceRequest request{};
            bind_spatial_dir_workspace_metadata(spatialDir, request);
            request.needOptics = scannerPostEffects.active();
            request.requestedWidth = width;
            request.requestedHeight = height;
            request.needBlurred =
                scannerPostEffects.glareActive && scannerPostEffects.glareBlurSigmaPx > 0.0f;
            focusedWorkspace = preparedFrame.bind_workspace_request(request);
        }
        JuicerCuda::SpatialDirBuildProfile printDirProfile{};
        bool printDirProfileCaptured = false;
        bool printDirScratchOverflow = false;
        double printDirPrepareMs = 0.0;
        double printDirBuildHostMs = 0.0;
        double printPipelineLaunchHostMs = 0.0;
        float printPipelineCudaMs = -1.0f;
        if (spatialDir.hash != 0) {
            std::string spatialError;
            const auto printDirPrepareStart = std::chrono::steady_clock::now();
            if (!preparedFrame.prepare_spatial_dir_resources(
                    spatialDir,
                    focusedWorkspace,
                    _pCudaStream,
                    spatialError)) {
                preparedFrame.abort("print_spatial_dir_prepare_failed");
                throw_submission_fatal(
                    "print_spatial_dir_prepare",
                    "print spatial DIR preparation failed",
                    spatialError);
            }
            if (dirProfileEnabled) {
                printDirPrepareMs = elapsed_ms_since(printDirPrepareStart);
            }
            const auto scratch = preparedFrame.spatial_dir_scratch(focusedWorkspace);
            const auto resources =
                preparedFrame.spatial_dir_resources(focusedWorkspace, spatialDir.hash);
            if (!scratch.active || !resources.active) {
                preparedFrame.abort("print_spatial_dir_binding_failed");
                throw_print_restriction(
                    "MissingRequiredResource phase=4C field=prepared_spatial_dir");
            }
            SF_TEMP_BRIDGE_bind_filtered_correction_to_payload(run.filmDevelop, scratch);
            // SF_TEMP_BRIDGE_map_legacy_corr_planes_to_rawCorrection: Phase 5 removes
            // these old wrapper arguments after source/filter/final-develop split lands.
            float* SF_TEMP_BRIDGE_map_legacy_corr_planes_to_rawCorrectionY = scratch.rawCorrectionY;
            float* SF_TEMP_BRIDGE_map_legacy_corr_planes_to_rawCorrectionM = scratch.rawCorrectionM;
            float* SF_TEMP_BRIDGE_map_legacy_corr_planes_to_rawCorrectionC = scratch.rawCorrectionC;
            // SF_TEMP_BRIDGE_map_legacy_mix_planes_to_filteredCorrection: Phase 5 removes this alias.
            float* SF_TEMP_BRIDGE_map_legacy_mix_planes_to_filteredCorrectionY = scratch.filteredCorrectionY;
            float* SF_TEMP_BRIDGE_map_legacy_mix_planes_to_filteredCorrectionM = scratch.filteredCorrectionM;
            float* SF_TEMP_BRIDGE_map_legacy_mix_planes_to_filteredCorrectionC = scratch.filteredCorrectionC;
            float* SF_TEMP_BRIDGE_map_legacy_tmp_plane_to_filterTemp = scratch.filterTemp;
            float* SF_TEMP_BRIDGE_map_legacy_iir_forward_plane_to_iirForwardTemp =
                scratch.iirForwardTemp;
            printDirScratchOverflow = scratch.overflow;
            const auto printDirBuildStart = std::chrono::steady_clock::now();
            trace_spatial_dir_bridge_use(
                "print",
                "SF_TEMP_BRIDGE_build_print_spatial_dir",
                spatialDir);
            const cudaError_t dirError = juicer_cuda_build_print_spatial_dir(
                &run,
                SF_TEMP_BRIDGE_map_legacy_corr_planes_to_rawCorrectionY,
                SF_TEMP_BRIDGE_map_legacy_corr_planes_to_rawCorrectionM,
                SF_TEMP_BRIDGE_map_legacy_corr_planes_to_rawCorrectionC,
                SF_TEMP_BRIDGE_map_legacy_mix_planes_to_filteredCorrectionY,
                SF_TEMP_BRIDGE_map_legacy_mix_planes_to_filteredCorrectionM,
                SF_TEMP_BRIDGE_map_legacy_mix_planes_to_filteredCorrectionC,
                SF_TEMP_BRIDGE_map_legacy_tmp_plane_to_filterTemp,
                SF_TEMP_BRIDGE_map_legacy_iir_forward_plane_to_iirForwardTemp,
                resources.gaussian.weights,
                resources.gaussian.radius,
                resources.gaussian.sigma,
                spatialDir.gaussianWeight,
                resources.exponential[0].weights,
                resources.exponential[0].radius,
                resources.exponential[0].sigma,
                spatialDir.exponentialWeights[0],
                resources.exponential[1].weights,
                resources.exponential[1].radius,
                resources.exponential[1].sigma,
                spatialDir.exponentialWeights[1],
                resources.exponential[2].weights,
                resources.exponential[2].radius,
                resources.exponential[2].sigma,
                spatialDir.exponentialWeights[2],
                _pCudaStream,
                dirProfileEnabled ? &printDirProfile : nullptr);
            if (dirProfileEnabled) {
                printDirBuildHostMs = elapsed_ms_since(printDirBuildStart);
            }
            if (dirError != cudaSuccess) {
                preparedFrame.abort("print_spatial_dir_launch_failed");
                throw_cuda_stage_fatal(
                    "print_spatial_dir_launch",
                    "print spatial DIR build failed",
                    dirError);
            }
            printDirProfileCaptured = dirProfileEnabled;
        }

        const JuicerCuda::Resources::DeviceScanMedium& scanMedium = *prepared.scanMedium;
        copy_scan_tables_payload(
            run.scanStage.scanTables,
            run.scanStage.scanTables.mediumIsNegative,
            run.scanStage.scanTables.min_cmy,
            run.scanStage.scanTables.inv_max_cmy,
            scanMedium);
        run.scanStage.scannerUseLut = 1;
        run.scanStage.scanLutLog10XYZ = prepared.scanLut->log10XYZ;
        run.scanStage.scanLutSlopeC = prepared.scanLut->slopeC;
        run.scanStage.scanLutSlopeM = prepared.scanLut->slopeM;
        run.scanStage.scanLutSlopeY = prepared.scanLut->slopeY;
        run.scanStage.scanLutCellMin = prepared.scanLut->cellMin;
        run.scanStage.scanLutCellMax = prepared.scanLut->cellMax;
        run.scanStage.scanLutRes = static_cast<int>(prepared.scanLut->res);
        const Scanner::ColorRuntime& color = *prepared.scannerColor;
        copy_float9(run.scanStage.scanColor.cat02, color.cat02);
        copy_float9(run.scanStage.scanColor.xyzToRgb, color.xyzToRgb);
        copy_float3(run.scanStage.scanColor.illuminantXYZ, color.illuminantXYZ);
        run.scanStage.scanColor.encoding.outputColorSpaceIndex =
            OutputEncoding::toIndex(color.encoding.colorSpace);
        run.scanStage.scanColor.encoding.applyCctfEncoding =
            bool_to_i32(color.encoding.applyCctfEncoding);
        run.scanStage.scanColor.encoding.preserveLinearRange =
            bool_to_i32(color.encoding.preserveLinearRange);
        run.scanStage.scanColor.encoding.inputIsOutputSpace =
            bool_to_i32(color.encoding.inputIsOutputSpace);
        const auto& outputSpace = GeneratedColorSpaces::get(color.encoding.colorSpace);
        run.scanStage.scanColor.encoding.cctf.kind = static_cast<int>(outputSpace.cctf.kind);
        run.scanStage.scanColor.encoding.cctf.gamma = outputSpace.cctf.gamma;
        run.scanStage.scanColor.encoding.cctf.a = outputSpace.cctf.a;
        run.scanStage.scanColor.encoding.cctf.b = outputSpace.cctf.b;
        run.scanStage.scanColor.encoding.cctf.c = outputSpace.cctf.c;
        run.scanStage.scanColor.encoding.cctf.d = outputSpace.cctf.d;
        run.scanStage.scanColor.encoding.cctf.linearCutoff = outputSpace.cctf.linearCutoff;
        const OutputEncoding::Matrix3x3 dwgToOutput =
            OutputEncoding::dwg_to_output_matrix(color.encoding.colorSpace);
        copy_float9(run.scanStage.scanColor.encoding.dwgToOutput, dwgToOutput.m);
        run.scanStage.correctionActive = scannerCorrection.active ? 1 : 0;
        run.scanStage.correctionSlope = scannerCorrection.xyzSlope;
        run.scanStage.correctionOffset = scannerCorrection.xyzOffset;

        std::string scanError;
        if (!preparedFrame.prepare_scan_error_stage(
                run.scanStage.scanErrorFlag,
                _pCudaStream,
                scanError)) {
            preparedFrame.abort("print_scan_error_stage_failed");
            throw_submission_fatal(
                "print_scan_error_stage",
                "print scan error stage failed",
                scanError);
        }
        cudaError_t launchError = cudaSuccess;
        const auto printPipelineLaunchStart = std::chrono::steady_clock::now();
        CudaEventElapsedTimer printPipelineCudaTimer;
        if (dirProfileEnabled) {
            printPipelineCudaTimer.begin(_pCudaStream);
        }
        if (scannerPostEffects.active()) {
            const auto post = preparedFrame.scanner_post_effects_resources(
                focusedWorkspace,
                scannerPostEffects.hash);
            if (!post.active) {
                preparedFrame.abort("print_scanner_post_effects_view_failed");
                throw_print_restriction(
                    "MissingRequiredResource phase=8C field=scanner_post_effects");
            }
            const std::uint64_t glareSeed = Hash::hash_uint64_values(
                {printRecipe->hash,
                 snapshot.frameToken.value,
                 static_cast<std::uint64_t>(srcBounds.x1),
                 static_cast<std::uint64_t>(srcBounds.y1)});
            launchError = juicer_cuda_print_focused_scanner_post_effects(
                &run,
                post.scratch.rgbR,
                post.scratch.rgbG,
                post.scratch.rgbB,
                post.scratch.tmp,
                post.scratch.blurred,
                post.lensBlur.weights,
                post.lensBlur.radius,
                post.unsharp.weights,
                post.unsharp.radius,
                scannerPostEffects.unsharpAmount,
                srcBounds.x1,
                srcBounds.y1,
                glareSeed,
                scannerPostEffects.glarePercent,
                scannerPostEffects.glareRoughness,
                post.glare.weights,
                post.glare.radius,
                _pCudaStream);
        } else {
            launchError = juicer_cuda_print_focused_pipeline(&run, _pCudaStream);
        }
        if (dirProfileEnabled) {
            printPipelineLaunchHostMs = elapsed_ms_since(printPipelineLaunchStart);
            if (launchError == cudaSuccess) {
                printPipelineCudaMs = printPipelineCudaTimer.finish(_pCudaStream);
            }
        }
        if (launchError != cudaSuccess) {
            preparedFrame.abort("print_pipeline_launch_failed");
            throw_cuda_stage_fatal(
                "print_pipeline_launch",
                "focused print pipeline launch failed",
                launchError);
        }
        if (dirProfileEnabled) {
            trace_spatial_dir_profile(
                "print",
                width,
                height,
                spatialDir,
                printDirProfile,
                printDirProfileCaptured,
                printDirScratchOverflow,
                spatialDirDescriptorMs,
                printDirPrepareMs,
                printDirBuildHostMs,
                printPipelineCudaMs,
                printPipelineLaunchHostMs);
        }
        if (!preparedFrame.finalize_scan_error_stage(
                run.scanStage.scanErrorFlag,
                _pCudaStream,
                scanError)) {
            preparedFrame.abort("print_scan_error_finalize_failed");
            throw_submission_fatal(
                "print_scan_error_finalize",
                "print scan error finalize failed",
                scanError);
        }
        record_cuda_use(preparedFrame);
        std::string finishError;
        if (!preparedFrame.finish(_pCudaStream, finishError)) {
            preparedFrame.abort("print_prepared_frame_finish_failed");
            throw_submission_fatal(
                "print_prepared_frame_finish",
                "print prepared frame finish failed",
                finishError);
        }
        JTRACE("PHASE4C", "focused print destination submission completed");
        JuicerCuda::LaunchGraphCounters::record_frame_completed();
        return;
    }

    trace_and_throw_cuda_policy_fatal(
        "CUDA unfocused launch blocked",
        "FocusedRenderStateRequiredAfterPhase4C");
#endif
}
