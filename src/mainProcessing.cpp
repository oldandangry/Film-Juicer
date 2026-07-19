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
#include <optional>

#include "GaussianSciPy.h"
#include "VisualGrainFrameDescriptor.h"

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
#include <cuda_runtime.h>
#include <cuda.h>
#if defined(_WIN32)
#include <windows.h>
#endif
#endif

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
#include "Cuda/Diffusion/JuicerCudaDiffusion.h"
#include "Cuda/JuicerCudaDirProfile.h"
#include "Cuda/JuicerCudaDirectFilmPayloads.h"
#include "Cuda/JuicerCudaFilmFoundationPayloads.h"
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

extern "C" cudaError_t juicer_cuda_profile_direct_focused_pipeline_stages(
    const JuicerCuda::DirectPipelineRunParams* hParams,
    float* dPlane0,
    float* dPlane1,
    float* dPlane2,
    JuicerCuda::CompositePipelineProfile* profile,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_profile_print_focused_pipeline_stages(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    float* dPlane0,
    float* dPlane1,
    float* dPlane2,
    JuicerCuda::CompositePipelineProfile* profile,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_direct_focused_scan_linear_rgb(
    const JuicerCuda::DirectPipelineRunParams* hParams,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    JuicerCuda::CompositePipelineProfile* aliasProfile,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_print_focused_scan_linear_rgb(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    float* dScratchBlurred,
    int glareOriginX,
    int glareOriginY,
    std::uint64_t glareSeed,
    float glarePercent,
    float glareRoughness,
    const float* dGlareKernel,
    int glareRadius,
    JuicerCuda::CompositePipelineProfile* aliasProfile,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_direct_focused_capture_density(
    const JuicerCuda::DirectPipelineRunParams* hParams,
    float* dDensityC,
    float* dDensityM,
    float* dDensityY,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_print_focused_capture_density(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    float* dDensityC,
    float* dDensityM,
    float* dDensityY,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_direct_camera_film_linear_exposure(
    const JuicerCuda::DirectPipelineRunParams* hParams,
    JuicerCuda::CameraFilmLinearExposurePlanes planes,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_print_camera_film_linear_exposure(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    JuicerCuda::CameraFilmLinearExposurePlanes planes,
    void* cudaStreamOpaque);

extern "C" cudaError_t
juicer_cuda_direct_focused_capture_density_from_camera_film_linear(
    const JuicerCuda::DirectPipelineRunParams* hParams,
    JuicerCuda::CameraFilmLinearExposurePlanes planes,
    float* dDensityC,
    float* dDensityM,
    float* dDensityY,
    void* cudaStreamOpaque);

extern "C" cudaError_t
juicer_cuda_print_focused_capture_density_from_camera_film_linear(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    JuicerCuda::CameraFilmLinearExposurePlanes planes,
    float* dDensityC,
    float* dDensityM,
    float* dDensityY,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_print_focused_continue_from_capture_density(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    float* dDensityC,
    float* dDensityM,
    float* dDensityY,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_print_focused_enlarger_linear_exposure(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    const float* dDensityC,
    const float* dDensityM,
    const float* dDensityY,
    JuicerCuda::EnlargerPrintLinearExposurePlanes planes,
    void* cudaStreamOpaque);

extern "C" cudaError_t
juicer_cuda_print_focused_develop_from_enlarger_linear(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    JuicerCuda::EnlargerPrintLinearExposurePlanes planes,
    float* dDensityC,
    float* dDensityM,
    float* dDensityY,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_direct_focused_scan_linear_density_rgb(
    const JuicerCuda::DirectPipelineRunParams* hParams,
    const float* dDensityC,
    const float* dDensityM,
    const float* dDensityY,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_print_focused_scan_linear_density_rgb(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    const float* dDensityC,
    const float* dDensityM,
    const float* dDensityY,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    float* dScratchBlurred,
    int glareOriginX,
    int glareOriginY,
    std::uint64_t glareSeed,
    float glarePercent,
    float glareRoughness,
    const float* dGlareKernel,
    int glareRadius,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_direct_focused_scanner_post_output(
    const JuicerCuda::DirectPipelineRunParams* hParams,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    const float* dLensBlurKernel,
    int lensBlurRadius,
    const float* dUnsharpKernel,
    int unsharpRadius,
    float unsharpAmount,
    const JuicerCuda::GrainPayload* gateDefects,
    const JuicerCuda::GateWeavePayload* weave,
    const float* gateMask,
    int gateMaskWidth,
    int gateMaskHeight,
    JuicerCuda::CompositePipelineProfile* aliasProfile,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_print_focused_scanner_post_output(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    const float* dLensBlurKernel,
    int lensBlurRadius,
    const float* dUnsharpKernel,
    int unsharpRadius,
    float unsharpAmount,
    const JuicerCuda::GrainPayload* gateDefects,
    const JuicerCuda::GateWeavePayload* weave,
    const float* gateMask,
    int gateMaskWidth,
    int gateMaskHeight,
    JuicerCuda::CompositePipelineProfile* aliasProfile,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_apply_visual_grain(
    const JuicerCuda::GrainPayload* grain,
    const JuicerCuda::GrainKernelPayload* kernels,
    int width,
    int height,
    float* densityC,
    float* densityM,
    float* densityY,
    float* filterTemp,
    float* scaleWork,
    float* deltaAccum,
    float* layerWork,
    float* sharedDelta,
    JuicerCuda::VisualGrainRuntimeProfile* profile,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_apply_film_defects(
    const JuicerCuda::GrainPayload* defects,
    int width,
    int height,
    float* densityC,
    float* densityM,
    float* densityY,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_build_gate_defect_mask_focused(
    const JuicerCuda::GrainPayload* defects,
    float* gateMask,
    int gateMaskWidth,
    int gateMaskHeight,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_build_direct_spatial_dir(
    const JuicerCuda::DirectPipelineRunParams* hParams,
    JuicerCuda::SpatialDirBuildRequest request);

extern "C" cudaError_t juicer_cuda_build_print_spatial_dir(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    JuicerCuda::SpatialDirBuildRequest request);

extern "C" cudaError_t juicer_cuda_build_direct_spatial_dir_cached_log_raw(
    const JuicerCuda::DirectPipelineRunParams* hParams,
    float* logRawB,
    float* logRawG,
    float* logRawR,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_build_print_spatial_dir_cached_log_raw(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    float* logRawB,
    float* logRawG,
    float* logRawR,
    void* cudaStreamOpaque);

extern "C" cudaError_t
juicer_cuda_build_direct_spatial_dir_cached_log_raw_from_camera_film_linear(
    const JuicerCuda::DirectPipelineRunParams* hParams,
    JuicerCuda::CameraFilmLinearExposurePlanes cameraFilmLinear,
    float* logRawB,
    float* logRawG,
    float* logRawR,
    void* cudaStreamOpaque);

extern "C" cudaError_t
juicer_cuda_build_print_spatial_dir_cached_log_raw_from_camera_film_linear(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    JuicerCuda::CameraFilmLinearExposurePlanes cameraFilmLinear,
    float* logRawB,
    float* logRawG,
    float* logRawR,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_print_pipeline(
    const JuicerCuda::PipelineRunParams* hParams,
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
#include "SpectralProcessing.h"
#include "Print.h"
#include "ProcessRoot.h"
#include "JuicerState.h"
#include "Scanner.h"
#include "OutputColor.h"
#include "Couplers.h"
#include "mainProcessing.h"

namespace {
    inline bool is_finite(float value);
    inline bool is_finite(double value);

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
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

    struct DiffusionStageBinding {
        Spektrafilm::PlanLayout layout{};
        Spektrafilm::DiffusionStageTileGeometry geometry{};
        Spektrafilm::DiffusionFrameDomain fullFrame{};
        JuicerCuda::Diffusion::SpectrumPackageView spectra{};
        JuicerCuda::Diffusion::ExecutionWorkspaceView execution{};
        JuicerCuda::Diffusion::StagePlaneSet stagePlanes{};
        bool active = false;
    };

    bool same_diffusion_spectrum_key(
        const Spektrafilm::DiffusionSpectrumKey& left,
        const Spektrafilm::DiffusionSpectrumKey& right) noexcept {
        return left.sampleHash == right.sampleHash &&
               left.extent == right.extent &&
               left.layoutSchema == right.layoutSchema &&
               left.hash == right.hash;
    }

    bool same_diffusion_plan_key(
        const Spektrafilm::DiffusionPlanKey& left,
        const Spektrafilm::DiffusionPlanKey& right) noexcept {
        return left.profileDigest == right.profileDigest &&
               left.extent == right.extent &&
               left.layoutSchema == right.layoutSchema &&
               left.hash == right.hash;
    }

    bool bind_diffusion_stage(
        const Spektrafilm::DiffusionFrameSetDescriptor& frameSet,
        const Spektrafilm::DiffusionStageFrameDescriptor& stageDescriptor,
        Spektrafilm::DiffusionLinearStage expectedStage,
        const JuicerCuda::Diffusion::DiffusionPreparedView& prepared,
        DiffusionStageBinding& out,
        std::string& diagnostic) {
        out = {};
        diagnostic.clear();
        const char* stageLabel =
            expectedStage ==
                    Spektrafilm::DiffusionLinearStage::CameraFilmLinear
                ? "camera"
                : "enlarger";
        if (stageDescriptor.stage != expectedStage ||
            stageDescriptor.route != frameSet.route || !prepared.active ||
            prepared.executionDescriptor.hash == 0 ||
            prepared.executionDescriptor.frameSetHash != frameSet.hash ||
            prepared.executionDescriptor.stageCount == 0 ||
            prepared.executionDescriptor.stageCount >
                prepared.executionDescriptor.stages.size() ||
            prepared.executionDescriptor.uniqueSpectrumCount == 0 ||
            prepared.executionDescriptor.uniqueSpectrumCount >
                prepared.executionDescriptor.spectrumKeys.size() ||
            prepared.spectrumCount !=
                prepared.executionDescriptor.uniqueSpectrumCount) {
            diagnostic = "MissingRequiredResource component=diffusion stage=";
            diagnostic += stageLabel;
            diagnostic += " field=execution_descriptor";
            return false;
        }

        const Spektrafilm::DiffusionStageTileGeometry* stageGeometry =
            nullptr;
        for (std::size_t stageIndex = 0;
             stageIndex < prepared.executionDescriptor.stageCount;
             ++stageIndex) {
            const auto& candidate =
                prepared.executionDescriptor.stages[stageIndex];
            if (candidate.stage != expectedStage) {
                continue;
            }
            if (stageGeometry) {
                diagnostic =
                    "ResourceDescriptorMismatch component=diffusion stage=";
                diagnostic += stageLabel;
                diagnostic += " field=duplicate_stage";
                return false;
            }
            stageGeometry = &candidate;
        }
        if (!stageGeometry ||
            stageGeometry->stageDescriptorHash != stageDescriptor.hash ||
            prepared.executionDescriptor.fullFrame != frameSet.fullFrame ||
            stageDescriptor.fullFrame != frameSet.fullFrame ||
            stageGeometry->spectrumKeyIndex >=
                prepared.executionDescriptor.uniqueSpectrumCount ||
            stageGeometry->spectrumKeyIndex >= prepared.spectrumCount) {
            diagnostic =
                "ResourceDescriptorMismatch component=diffusion stage=";
            diagnostic += stageLabel;
            diagnostic += " field=stage_geometry";
            return false;
        }

        const std::size_t spectrumIndex =
            stageGeometry->spectrumKeyIndex;
        if (!same_diffusion_spectrum_key(
                prepared.executionDescriptor.spectrumKeys[spectrumIndex],
                prepared.spectra[spectrumIndex].key) ||
            !same_diffusion_plan_key(
                prepared.executionDescriptor.planKey,
                prepared.execution.planKey)) {
            diagnostic =
                "ResourceDescriptorMismatch component=diffusion stage=";
            diagnostic += stageLabel;
            diagnostic += " field=resource_key";
            return false;
        }

        const auto& spectra = prepared.spectra[spectrumIndex].spectra;
        const auto& execution = prepared.execution.execution;
        const auto& planes = prepared.execution.stagePlanes;
        if (!spectra.red || !spectra.green || !spectra.blue ||
            !execution.transformBuffer || execution.r2cPlan == 0 ||
            execution.c2rPlan == 0 || !planes.redSensitive ||
            !planes.greenSensitive || !planes.blueSensitive ||
            !planes.auxiliary ||
            planes.redSensitive == planes.greenSensitive ||
            planes.redSensitive == planes.blueSensitive ||
            planes.redSensitive == planes.auxiliary ||
            planes.greenSensitive == planes.blueSensitive ||
            planes.greenSensitive == planes.auxiliary ||
            planes.blueSensitive == planes.auxiliary ||
            planes.rowStrideFloats <
                static_cast<std::size_t>(frameSet.fullFrame.width)) {
            diagnostic = "MissingRequiredResource component=diffusion stage=";
            diagnostic += stageLabel;
            diagnostic += " field=execution_workspace";
            return false;
        }

        out.layout = prepared.executionDescriptor.layout;
        out.geometry = *stageGeometry;
        out.fullFrame = prepared.executionDescriptor.fullFrame;
        out.spectra = spectra;
        out.execution = execution;
        out.stagePlanes = planes;
        out.active = true;
        return true;
    }

    JuicerCuda::CameraFilmLinearExposurePlanes camera_film_linear_planes(
        const JuicerCuda::Diffusion::StagePlaneSet& planes) noexcept {
        return {
            planes.redSensitive,
            planes.greenSensitive,
            planes.blueSensitive,
            planes.rowStrideFloats};
    }

    JuicerCuda::EnlargerPrintLinearExposurePlanes
    enlarger_print_linear_planes(
        const JuicerCuda::Diffusion::StagePlaneSet& planes) noexcept {
        return {
            planes.redSensitive,
            planes.greenSensitive,
            planes.blueSensitive,
            planes.rowStrideFloats};
    }

    const char* diffusion_failure_api_label(
        JuicerCuda::Diffusion::FailureApi api) noexcept {
        switch (api) {
            case JuicerCuda::Diffusion::FailureApi::Validation:
                return "validation";
            case JuicerCuda::Diffusion::FailureApi::Cuda:
                return "cuda";
            case JuicerCuda::Diffusion::FailureApi::Cufft:
                return "cufft";
            case JuicerCuda::Diffusion::FailureApi::None:
                break;
        }
        return "none";
    }

    std::string diffusion_launch_failure_diagnostic(
        const JuicerCuda::Diffusion::LaunchResult& result) {
        std::string diagnostic = "DiffusionLaunchFailure api=";
        diagnostic += diffusion_failure_api_label(result.api);
        diagnostic += " code=";
        diagnostic += std::to_string(result.code);
        diagnostic += " stage=";
        diagnostic += nonempty_cstr_or(result.stage, "unknown");
        return diagnostic;
    }
#endif

    inline void clear_glare_compensation_fields(Profiles::ProfileGlare& glare) {
        glare.printShadowCompensationFactor = 0.0f;
        glare.printShadowCompensationDensity = 0.0f;
        glare.printShadowCompensationTransition = 0.0f;
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
            return;
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

    struct AliasRouteProfileExtent {
        int width = 0;
        int height = 0;
    };

    void initialize_spatial_dir_rgb_alias_profile(
        JuicerCuda::CompositePipelineProfile& profile,
        AliasRouteProfileExtent extent) {
        profile = JuicerCuda::CompositePipelineProfile{};
        profile.width = extent.width;
        profile.height = extent.height;
        profile.captured = 1;
        profile.aliasRouteCaptured = 1;
        profile.profileKind = "focused_alias_real_stage_attribution";
        profile.profileNote = "spatial_dir_rgb_alias_attributed";
    }

    void record_alias_route_stage(
        JuicerCuda::CompositePipelineProfile& profile,
        JuicerCuda::SpatialDirStageProfile& stage,
        CudaEventElapsedTimer& timer,
        void* streamOpaque) {
        if (!profile.aliasRouteCaptured || !timer.active) {
            return;
        }
        const float elapsedMs = timer.finish(streamOpaque);
        if (elapsedMs < 0.0f) {
            return;
        }
        stage.elapsedMs = elapsedMs;
        profile.total.elapsedMs += elapsedMs;
    }

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

    const char* visual_grain_scratch_shape_label(
        Spektrafilm::VisualGrainScratchShape shape) {
        switch (shape) {
            case Spektrafilm::VisualGrainScratchShape::Streamed:
                return "streamed";
            case Spektrafilm::VisualGrainScratchShape::StreamedShared:
                return "streamed_shared";
            case Spektrafilm::VisualGrainScratchShape::StreamedLayers:
                return "streamed_layers";
            case Spektrafilm::VisualGrainScratchShape::StreamedLayersShared:
                return "streamed_layers_shared";
            case Spektrafilm::VisualGrainScratchShape::None:
            default:
                return "none";
        }
    }

    int visual_grain_scratch_plane_count(
        Spektrafilm::VisualGrainScratchShape shape) {
        switch (shape) {
            case Spektrafilm::VisualGrainScratchShape::Streamed:
                return 3;
            case Spektrafilm::VisualGrainScratchShape::StreamedShared:
            case Spektrafilm::VisualGrainScratchShape::StreamedLayers:
                return 4;
            case Spektrafilm::VisualGrainScratchShape::StreamedLayersShared:
                return 5;
            case Spektrafilm::VisualGrainScratchShape::None:
            default:
                return 0;
        }
    }

    const char* profile_polarity_label(
        Spektrafilm::ProfilePolarity polarity) {
        switch (polarity) {
            case Spektrafilm::ProfilePolarity::Negative:
                return "negative";
            case Spektrafilm::ProfilePolarity::Positive:
                return "positive";
            case Spektrafilm::ProfilePolarity::Unsupported:
            default:
                return "unsupported";
        }
    }

    const char* strict_yvv_shape_label(
        const Spektrafilm::SpatialDirDescriptor& descriptor,
        const Spektrafilm::DirScratchPlaneRoles& roles) {
        if (descriptor.approximation != Spektrafilm::DirApproximationMarker::SpektrafilmStrict ||
            (descriptor.scratchTier != Spektrafilm::DirScratchTier::Tier1IChannels &&
             descriptor.scratchTier != Spektrafilm::DirScratchTier::Tier2)) {
            return "none";
        }
        if (roles.filterTempPlanes == 3 &&
            roles.cachedLogRawPlanes == 3) {
            return "strict_yvv_channels_aliased_forward_cached_lograw";
        }
        if (roles.filterTempPlanes == 3 &&
            roles.cachedLogRawPlanes == 2) {
            return "strict_yvv_channels_aliased_forward_cached_lograw_bg";
        }
        if (roles.filterTempPlanes == 3) {
            return "strict_yvv_channels_aliased_forward";
        }
        if (roles.filterTempPlanes == 2) {
            return "strict_yvv_low_scratch_pair";
        }
        if (roles.rawCorrectionPlanes == 1 &&
            roles.filterTempPlanes == 1) {
            return "strict_yvv_component_streamed";
        }
        if (roles.filterTempPlanes == 1) {
            return "strict_yvv_single_temp_sequential";
        }
        return "unknown";
    }

    const char* final_develop_lograw_source_label(
        bool dirActive,
        const Spektrafilm::DirScratchPlaneRoles& roles,
        const Spektrafilm::DirScratchPlaneRoles& targetRoles,
        bool fusedScannerPostSpatialDirHandoff) {
        if (!dirActive) {
            return "none";
        }
        if (fusedScannerPostSpatialDirHandoff) {
            if (roles.cachedLogRawPlanes == 3 && targetRoles.cachedLogRawPlanes == 3) {
                return "source_build_cached_fused_scan";
            }
            if (roles.cachedLogRawPlanes == 2 && targetRoles.cachedLogRawPlanes == 2) {
                return "source_build_cached_bg_fused_scan";
            }
            return "recompute_source_rgb_fused_scan";
        }
        if (targetRoles.cachedLogRawPlanes == 0) {
            return "recompute_source_rgb";
        }
        if (targetRoles.cachedLogRawPlanes == 3) {
            if (roles.cachedLogRawPlanes == 3) {
                return "retained_cached";
            }
            if (roles.cachedLogRawPlanes == 0) {
                return "staged_cached";
            }
        }
        return "unsupported";
    }

    const char* final_develop_lograw_compute_label(
        bool dirActive,
        const Spektrafilm::DirScratchPlaneRoles& targetRoles,
        bool fusedScannerPostSpatialDirHandoff) {
        if (!dirActive) {
            return "none";
        }
        if (fusedScannerPostSpatialDirHandoff && targetRoles.cachedLogRawPlanes == 3) {
            return "cached_source_build";
        }
        if (fusedScannerPostSpatialDirHandoff && targetRoles.cachedLogRawPlanes == 2) {
            return "cached_source_build_bg";
        }
        if (fusedScannerPostSpatialDirHandoff || targetRoles.cachedLogRawPlanes == 0) {
            return "raw_only";
        }
        if (targetRoles.cachedLogRawPlanes == 3) {
            return "cached_or_staged";
        }
        return "unsupported";
    }

    int final_develop_cached_lograw_planes(
        bool dirActive,
        const Spektrafilm::DirScratchPlaneRoles& targetRoles,
        bool /*fusedScannerPostSpatialDirHandoff*/) {
        return dirActive && targetRoles.cachedLogRawPlanes > 0
                   ? targetRoles.cachedLogRawPlanes
                   : 0;
    }

    int final_develop_staged_cached_lograw_planes(
        bool dirActive,
        const Spektrafilm::DirScratchPlaneRoles& roles,
        const Spektrafilm::DirScratchPlaneRoles& targetRoles,
        bool fusedScannerPostSpatialDirHandoff) {
        if (fusedScannerPostSpatialDirHandoff) {
            return 0;
        }
        return dirActive && roles.cachedLogRawPlanes == 0 &&
                       targetRoles.cachedLogRawPlanes == 3
                   ? 3
                   : 0;
    }

    const char* final_develop_cached_lograw_release_label(
        bool dirActive,
        const Spektrafilm::DirScratchPlaneRoles& targetRoles,
        bool fusedScannerPostSpatialDirHandoff) {
        if (fusedScannerPostSpatialDirHandoff) {
            return dirActive && targetRoles.cachedLogRawPlanes > 0 ? "after_fused_scan_linear" : "none";
        }
        return dirActive && targetRoles.cachedLogRawPlanes == 3 ? "after_final_develop" : "none";
    }

    bool fused_alias_uses_source_build_cached_log_raw(
        bool fusedScannerPostSpatialDirHandoff,
        const Spektrafilm::DirScratchPlaneRoles& roles,
        const Spektrafilm::DirScratchPlaneRoles& targetRoles) noexcept {
        return fusedScannerPostSpatialDirHandoff &&
               roles.cachedLogRawPlanes > 0 &&
               roles.cachedLogRawPlanes == targetRoles.cachedLogRawPlanes;
    }

    bool spatial_dir_required_cached_log_raw_present(
        const JuicerProcess::Root::PreparedCudaFrame::SpatialDirScratchView& scratch,
        int cachedLogRawPlanes) noexcept {
        if (cachedLogRawPlanes <= 0) {
            return true;
        }
        if (!scratch.logRawB) {
            return false;
        }
        if (cachedLogRawPlanes >= 2 && !scratch.logRawG) {
            return false;
        }
        if (cachedLogRawPlanes >= 3 && !scratch.logRawR) {
            return false;
        }
        return true;
    }

    struct SpatialDirProfileAdmittedRoles {
        Spektrafilm::DirScratchPlaneRoles roles;
        Spektrafilm::DirScratchPlaneRoles targetRoles;
    };

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

    void bind_spatial_dir_final_develop_to_payload(
        JuicerCuda::FilmDevelopPayload& payload,
        const JuicerProcess::Root::PreparedCudaFrame::SpatialDirScratchView& scratch) {
        payload.spatialDir.active = 1;
        payload.spatialDir.filteredCorrectionY = scratch.filteredCorrectionY;
        payload.spatialDir.filteredCorrectionM = scratch.filteredCorrectionM;
        payload.spatialDir.filteredCorrectionC = scratch.filteredCorrectionC;
        payload.spatialDir.logRawB = scratch.logRawB;
        payload.spatialDir.logRawG = scratch.logRawG;
        payload.spatialDir.logRawR = scratch.logRawR;
    }

    void clear_spatial_dir_final_develop_payload_bindings(
        JuicerCuda::FilmDevelopPayload& payload) {
        payload.spatialDir = JuicerCuda::SpatialDirPayload{};
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
        msg += " cached_log_raw_planes=";
        msg += std::to_string(roles.cachedLogRawPlanes);
        msg += " target_raw_correction_planes=";
        msg += std::to_string(targetRoles.rawCorrectionPlanes);
        msg += " target_filtered_correction_planes=";
        msg += std::to_string(targetRoles.filteredCorrectionPlanes);
        msg += " target_filter_temp_planes=";
        msg += std::to_string(targetRoles.filterTempPlanes);
        msg += " target_cached_log_raw_planes=";
        msg += std::to_string(targetRoles.cachedLogRawPlanes);
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

    void trace_spatial_dir_profile(
        const char* route,
        int width,
        int height,
        const Spektrafilm::SpatialDirDescriptor& descriptor,
        const SpatialDirProfileAdmittedRoles& admittedRoles,
        const JuicerCuda::SpatialDirBuildProfile& profile,
        bool dirActive,
        bool scratchOverflow,
        double descriptorMs,
        double prepareMs,
        double buildHostMs,
        float pipelineCudaMs,
        double pipelineLaunchHostMs,
        const JuicerCuda::CompositePipelineProfile& compositeProfile,
        const ProfileRoute& profileRoute,
        std::uint64_t routeRecipeHash,
        const VisualGrainRecipe& visualGrainRecipe,
        const FilmJuicerEffectsRecipe& effectsRecipe,
        const Spektrafilm::VisualGrainFrameDescriptor* visualGrainDescriptor,
        Spektrafilm::VisualGrainScratchShape visualGrainScratchShape,
        bool visualGrainScratchOverflow,
        std::uint64_t visualGrainStaticVersion,
        const JuicerProcess::Root::PreparedCudaFrame::UploadTraceView& uploadTrace,
        const JuicerCuda::VisualGrainRuntimeProfile& visualGrainProfile,
        double visualGrainDescriptorMs,
        double preparedFramePrepareMs,
        double preparedFrameFinishMs,
        double visualGrainSubmitHostMs,
        bool focusedSplitActive,
        bool filmEffectsActive,
        bool scannerPostEffectsActive) {
        const std::uint64_t pixels =
            static_cast<std::uint64_t>(std::max(0, width)) *
            static_cast<std::uint64_t>(std::max(0, height));
        const int admittedPlaneCount = dirActive ? admittedRoles.roles.total_float_planes() : 0;
        const std::uint64_t scratchBytesApprox =
            dirActive ? pixels * static_cast<std::uint64_t>(std::max(0, admittedPlaneCount)) * sizeof(float) : 0ull;
        const bool visualGrainActive = visualGrainDescriptor != nullptr;
        const int visualGrainScratchPlanes =
            visualGrainActive
                ? visual_grain_scratch_plane_count(visualGrainScratchShape)
                : 0;
        const std::uint64_t visualGrainScratchBytes =
            visualGrainActive
                ? pixels *
                          static_cast<std::uint64_t>(visualGrainScratchPlanes) *
                          sizeof(float) +
                      sizeof(JuicerCuda::GrainFrameUniforms)
                : 0ull;
        const std::uint64_t visualGrainStageLiveBytes =
            visualGrainActive
                ? pixels *
                          static_cast<std::uint64_t>(visualGrainScratchPlanes + 3) *
                          sizeof(float) +
                      sizeof(JuicerCuda::GrainFrameUniforms)
                : 0ull;
        const int activeTails = active_tail_component_count(profile);
        const Spektrafilm::DirScratchPlaneRoles& roles = admittedRoles.roles;
        const Spektrafilm::DirScratchPlaneRoles& targetRoles = admittedRoles.targetRoles;
        const int expectedCorrectionLaunches =
            dirActive ? (roles.rawCorrectionPlanes == 1 ? 3 : 1) : 0;
        const bool fusedScannerPostSpatialDirHandoff =
            scannerPostEffectsActive && dirActive &&
            descriptor.approximation == Spektrafilm::DirApproximationMarker::SpektrafilmStrict;
        const bool materializedScannerPostSpatialDirHandoff =
            scannerPostEffectsActive && dirActive && !fusedScannerPostSpatialDirHandoff;
        const int finalDevelopCachedLogRawPlanes =
            final_develop_cached_lograw_planes(
                dirActive,
                targetRoles,
                fusedScannerPostSpatialDirHandoff);
        const int finalDevelopStagedCachedLogRawPlanes =
            final_develop_staged_cached_lograw_planes(
                dirActive,
                roles,
                targetRoles,
                fusedScannerPostSpatialDirHandoff);
        const std::uint64_t finalDevelopCachedLogRawBytesApprox =
            pixels * static_cast<std::uint64_t>(finalDevelopCachedLogRawPlanes) * sizeof(float);
        const std::uint64_t finalDevelopStagedCachedLogRawBytesApprox =
            pixels * static_cast<std::uint64_t>(finalDevelopStagedCachedLogRawPlanes) * sizeof(float);
        const Spektrafilm::DirScratchTier admittedTargetScratchTier =
            targetRoles.cachedLogRawPlanes > 0 ? descriptor.targetScratchTier : descriptor.scratchTier;
        const JuicerCuda::PrintDevelopBreakdownProfile& printBreakdown =
            compositeProfile.printDevelopBreakdown;
        const int scannerPostDensityIntermediatePlanes =
            materializedScannerPostSpatialDirHandoff ? 3 : 0;
        const std::uint64_t scannerPostDensityIntermediateBytes =
            pixels *
            static_cast<std::uint64_t>(scannerPostDensityIntermediatePlanes) *
            sizeof(float);
        const int scannerPostRgbAliasPlanes =
            fusedScannerPostSpatialDirHandoff ? 3 : 0;
        const std::uint64_t scannerPostRgbAliasSavedBytes =
            pixels *
            static_cast<std::uint64_t>(scannerPostRgbAliasPlanes) *
            sizeof(float);
        const char* scannerPostRgbSource =
            scannerPostEffectsActive
                ? (fusedScannerPostSpatialDirHandoff
                       ? "spatial_dir_filtered_alias"
                       : "optics_rgb_scratch")
                : "none";
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3);
        oss << "route=" << nonempty_cstr_or(route, "unknown")
            << " dir_active=" << bool_to_i32(dirActive)
            << " width=" << width
            << " height=" << height
            << " route_recipe_hash=" << routeRecipeHash
            << " film_profile_key=" << profileRoute.filmProfileKey
            << " print_profile_key="
            << (profileRoute.printProfileKey.empty() ? "none" : profileRoute.printProfileKey)
            << " capture_polarity="
            << profile_polarity_label(profileRoute.capturePolarity)
            << " focused_split_active=" << bool_to_i32(focusedSplitActive)
            << " film_effects_active=" << bool_to_i32(filmEffectsActive)
            << " visual_grain_active=" << bool_to_i32(visualGrainActive)
            << " visual_grain_profile_captured=" << visualGrainProfile.captured
            << " visual_grain_descriptor_hash="
            << (visualGrainDescriptor ? visualGrainDescriptor->hash : 0ull)
            << " visual_grain_recipe_hash="
            << (visualGrainDescriptor ? visualGrainDescriptor->recipeHash : 0ull)
            << " visual_grain_descriptor_ms=" << visualGrainDescriptorMs
            << " prepared_frame_prepare_ms=" << preparedFramePrepareMs
            << " prepared_frame_finish_ms=" << preparedFrameFinishMs
            << " visual_grain_cuda_ms=" << visualGrainProfile.total.elapsedMs
            << " visual_grain_submit_host_ms=" << visualGrainSubmitHostMs
            << " visual_grain_total_launches=" << visualGrainProfile.totalLaunches
            << " visual_grain_mix_evaluations=" << visualGrainProfile.mixEvaluations
            << " visual_grain_scale_evaluations=" << visualGrainProfile.scaleEvaluations
            << " visual_grain_frame_uniform_preparation_launches="
            << visualGrainProfile.frameUniformPreparationLaunches
            << " visual_grain_clear_launches=" << visualGrainProfile.clearLaunches
            << " visual_grain_layer_particle_launches="
            << visualGrainProfile.layerParticleLaunches
            << " visual_grain_simple_particle_launches="
            << visualGrainProfile.simpleParticleLaunches
            << " visual_grain_dye_blur_pass_launches="
            << visualGrainProfile.dyeBlurPassLaunches
            << " visual_grain_correlation_blur_pass_launches="
            << visualGrainProfile.correlationBlurPassLaunches
            << " visual_grain_form_delta_launches=" << visualGrainProfile.formDeltaLaunches
            << " visual_grain_subtract_launches=" << visualGrainProfile.subtractLaunches
            << " visual_grain_layer_accumulate_launches="
            << visualGrainProfile.layerAccumulateLaunches
            << " visual_grain_weighted_accumulate_launches="
            << visualGrainProfile.weightedAccumulateLaunches
            << " visual_grain_scale_launches=" << visualGrainProfile.scaleLaunches
            << " visual_grain_shared_mix_launches="
            << visualGrainProfile.sharedMixLaunches
            << " visual_grain_reconstruct_launches="
            << visualGrainProfile.reconstructLaunches
            << " visual_grain_debug_launches=" << visualGrainProfile.debugLaunches
            << " visual_grain_copy_operations=" << visualGrainProfile.copyOperations
            << " visual_grain_copy_bytes=" << visualGrainProfile.copyBytes
            << " visual_grain_sublayers_active="
            << bool_to_i32(visualGrainActive && visualGrainRecipe.sublayersActive)
            << " visual_grain_sublayer_count="
            << (visualGrainActive ? visualGrainRecipe.nSubLayers : 0)
            << " visual_grain_particle_area_um2="
            << (visualGrainActive ? visualGrainRecipe.particleAreaUm2 : 0.0f)
            << " visual_grain_particle_scale_cmy="
            << (visualGrainActive ? visualGrainRecipe.particleScaleCmy[0] : 0.0f)
            << ","
            << (visualGrainActive ? visualGrainRecipe.particleScaleCmy[1] : 0.0f)
            << ","
            << (visualGrainActive ? visualGrainRecipe.particleScaleCmy[2] : 0.0f)
            << " visual_grain_particle_scale_layers="
            << (visualGrainActive ? visualGrainRecipe.particleScaleLayers[0] : 0.0f)
            << ","
            << (visualGrainActive ? visualGrainRecipe.particleScaleLayers[1] : 0.0f)
            << ","
            << (visualGrainActive ? visualGrainRecipe.particleScaleLayers[2] : 0.0f)
            << " visual_grain_uniformity_cmy="
            << (visualGrainActive ? visualGrainRecipe.uniformityCmy[0] : 0.0f)
            << ","
            << (visualGrainActive ? visualGrainRecipe.uniformityCmy[1] : 0.0f)
            << ","
            << (visualGrainActive ? visualGrainRecipe.uniformityCmy[2] : 0.0f)
            << " visual_grain_amplitude="
            << (visualGrainActive ? visualGrainRecipe.amplitude : 0.0f)
            << " visual_grain_chroma_mix="
            << (visualGrainActive ? visualGrainRecipe.chromaMix : 0.0f)
            << " visual_grain_chroma_shared_weight="
            << (visualGrainActive ? visualGrainRecipe.chromaSharedWeight : 0.0f)
            << " visual_grain_chroma_independent_weight="
            << (visualGrainActive ? visualGrainRecipe.chromaIndependentWeight : 0.0f)
            << " visual_grain_debug_view="
            << (visualGrainActive ? visualGrainRecipe.debugView : 0)
            << " visual_grain_correlation_sigma_px="
            << (visualGrainActive ? visualGrainRecipe.correlationSigmaPx : 0.0f)
            << " visual_grain_dye_cloud_blur_um="
            << (visualGrainActive ? visualGrainRecipe.dyeCloudBlurUm : 0.0f)
            << " visual_grain_size_mix_scale="
            << (visualGrainActive ? visualGrainRecipe.sizeMixScale : 0.0f)
            << " visual_grain_clump_temporal_mix="
            << (visualGrainActive ? visualGrainRecipe.clumpTemporalMix : 0.0f)
            << " visual_grain_clump_morph_period_sec="
            << (visualGrainActive ? visualGrainRecipe.clumpMorphPeriodSec : 0.0f)
            << " visual_grain_frame0="
            << (visualGrainDescriptor ? visualGrainDescriptor->frame0 : 0)
            << " visual_grain_frame_alpha="
            << (visualGrainDescriptor ? visualGrainDescriptor->frameAlpha : 0.0f)
            << " visual_grain_fine_weight="
            << (visualGrainDescriptor ? visualGrainDescriptor->effectiveFineWeight : 0.0f)
            << " visual_grain_mid_weight="
            << (visualGrainDescriptor ? visualGrainDescriptor->effectiveMidWeight : 0.0f)
            << " visual_grain_coarse_weight="
            << (visualGrainDescriptor ? visualGrainDescriptor->effectiveCoarseWeight : 0.0f)
            << " visual_grain_correlation_radii="
            << (visualGrainDescriptor ? visualGrainDescriptor->correlation[0].radius : 0)
            << ","
            << (visualGrainDescriptor ? visualGrainDescriptor->correlation[1].radius : 0)
            << ","
            << (visualGrainDescriptor ? visualGrainDescriptor->correlation[2].radius : 0)
            << " visual_grain_dye_radii_l0="
            << (visualGrainDescriptor ? visualGrainDescriptor->dyeCloud[0][0].radius : 0)
            << ","
            << (visualGrainDescriptor ? visualGrainDescriptor->dyeCloud[0][1].radius : 0)
            << ","
            << (visualGrainDescriptor ? visualGrainDescriptor->dyeCloud[0][2].radius : 0)
            << " visual_grain_dye_radii_l1="
            << (visualGrainDescriptor ? visualGrainDescriptor->dyeCloud[1][0].radius : 0)
            << ","
            << (visualGrainDescriptor ? visualGrainDescriptor->dyeCloud[1][1].radius : 0)
            << ","
            << (visualGrainDescriptor ? visualGrainDescriptor->dyeCloud[1][2].radius : 0)
            << " visual_grain_dye_radii_l2="
            << (visualGrainDescriptor ? visualGrainDescriptor->dyeCloud[2][0].radius : 0)
            << ","
            << (visualGrainDescriptor ? visualGrainDescriptor->dyeCloud[2][1].radius : 0)
            << ","
            << (visualGrainDescriptor ? visualGrainDescriptor->dyeCloud[2][2].radius : 0)
            << " visual_grain_scratch_shape="
            << visual_grain_scratch_shape_label(
                   visualGrainActive
                       ? visualGrainScratchShape
                       : Spektrafilm::VisualGrainScratchShape::None)
            << " visual_grain_scratch_source="
            << (visualGrainActive
                    ? (visualGrainScratchOverflow ? "overflow" : "retained")
                    : "none")
            << " visual_grain_scratch_planes=" << visualGrainScratchPlanes
            << " visual_grain_scratch_bytes=" << visualGrainScratchBytes
            << " visual_grain_stage_live_bytes=" << visualGrainStageLiveBytes
            << " visual_grain_static_version=" << visualGrainStaticVersion
            << " resource_uploaded_build_counter=" << uploadTrace.uploadedBuildCounter
            << " resource_direct_upload_counter=" << uploadTrace.directUploadCounter
            << " resource_print_preparation_counter="
            << uploadTrace.printPreparationCounter
            << " effect_film_dust_amount=" << effectsRecipe.filmDustAmount
            << " effect_film_scratch_amount=" << effectsRecipe.filmScratchAmount
            << " effect_gate_dust_amount=" << effectsRecipe.gateDustAmount
            << " effect_gate_scratch_amount=" << effectsRecipe.gateScratchAmount
            << " effect_gate_weave_amount=" << effectsRecipe.gateWeaveAmount
            << " descriptor_hash=" << descriptor.hash
            << " dir_recipe_hash=" << descriptor.dirRecipeHash
            << " descriptor_support=" << Spektrafilm::to_cstr(descriptor.support)
            << " source_contract=" << Spektrafilm::to_cstr(descriptor.sourceContract)
            << " boundary_mode=" << Spektrafilm::to_cstr(descriptor.boundaryMode)
            << " scratch_tier=" << Spektrafilm::to_cstr(descriptor.scratchTier)
            << " target_scratch_tier=" << Spektrafilm::to_cstr(admittedTargetScratchTier)
            << " approximation_marker=" << Spektrafilm::to_cstr(descriptor.approximation)
            << " strict_yvv_shape=" << strict_yvv_shape_label(descriptor, roles)
            << " final_develop_lograw_source="
            << final_develop_lograw_source_label(
                   dirActive,
                   roles,
                   targetRoles,
                   fusedScannerPostSpatialDirHandoff)
            << " final_develop_lograw_compute="
            << final_develop_lograw_compute_label(
                   dirActive,
                   targetRoles,
                   fusedScannerPostSpatialDirHandoff)
            << " final_develop_cached_log_raw_planes=" << finalDevelopCachedLogRawPlanes
            << " final_develop_cached_log_raw_main_planes="
            << (dirActive ? roles.cachedLogRawPlanes : 0)
            << " final_develop_cached_log_raw_stage_planes="
            << finalDevelopStagedCachedLogRawPlanes
            << " final_develop_cached_log_raw_bytes_approx="
            << finalDevelopCachedLogRawBytesApprox
            << " final_develop_cached_log_raw_stage_bytes_approx="
            << finalDevelopStagedCachedLogRawBytesApprox
            << " final_develop_cached_log_raw_release="
            << final_develop_cached_lograw_release_label(
                   dirActive,
                   targetRoles,
                   fusedScannerPostSpatialDirHandoff)
            << " scanner_post_effects_active=" << bool_to_i32(scannerPostEffectsActive)
            << " scanner_post_dir_handoff="
            << (fusedScannerPostSpatialDirHandoff
                    ? "fused_final_develop_scan_linear"
                    : (materializedScannerPostSpatialDirHandoff ? "materialized_density_scan_linear" : "none"))
            << " scanner_post_rgb_source="
            << scannerPostRgbSource
            << " scanner_post_rgb_alias_planes="
            << scannerPostRgbAliasPlanes
            << " scanner_post_rgb_alias_saved_bytes_approx="
            << scannerPostRgbAliasSavedBytes
            << " scanner_post_density_intermediate_planes="
            << scannerPostDensityIntermediatePlanes
            << " scanner_post_density_intermediate_bytes_approx="
            << scannerPostDensityIntermediateBytes
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
            << " expected_correction_launches=" << expectedCorrectionLaunches
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
            << " composite_profile_captured=" << compositeProfile.captured
            << " composite_profile_kind=" << nonempty_cstr_or(compositeProfile.profileKind, "none")
            << " composite_profile_note=" << nonempty_cstr_or(compositeProfile.profileNote, "none")
            << " alias_route_profile_captured=" << compositeProfile.aliasRouteCaptured
            << " alias_fused_scan_linear_ms=" << compositeProfile.aliasFusedScanLinear.elapsedMs
            << " alias_scanner_post_output_ms=" << compositeProfile.aliasScannerPostOutput.elapsedMs
            << " alias_glare_launches=" << compositeProfile.aliasGlare.launches
            << " alias_glare_ms=" << compositeProfile.aliasGlare.elapsedMs
            << " alias_final_develop_scan_linear_launches="
            << compositeProfile.aliasFinalDevelopScanLinear.launches
            << " alias_final_develop_scan_linear_ms="
            << compositeProfile.aliasFinalDevelopScanLinear.elapsedMs
            << " alias_lens_blur_launches=" << compositeProfile.aliasLensBlur.launches
            << " alias_lens_blur_ms=" << compositeProfile.aliasLensBlur.elapsedMs
            << " alias_unsharp_launches=" << compositeProfile.aliasUnsharp.launches
            << " alias_unsharp_ms=" << compositeProfile.aliasUnsharp.elapsedMs
            << " alias_output_encode_launches=" << compositeProfile.aliasOutputEncode.launches
            << " alias_output_encode_ms=" << compositeProfile.aliasOutputEncode.elapsedMs
            << " composite_profile_width=" << compositeProfile.width
            << " composite_profile_height=" << compositeProfile.height
            << " composite_profile_total_launches=" << compositeProfile.totalLaunches
            << " composite_profile_total_ms=" << compositeProfile.total.elapsedMs
            << " composite_film_raw_launches=" << compositeProfile.filmRaw.launches
            << " composite_film_raw_ms=" << compositeProfile.filmRaw.elapsedMs
            << " composite_film_develop_launches=" << compositeProfile.filmDevelop.launches
            << " composite_film_develop_ms=" << compositeProfile.filmDevelop.elapsedMs
            << " composite_print_develop_launches=" << compositeProfile.printDevelop.launches
            << " composite_print_develop_ms=" << compositeProfile.printDevelop.elapsedMs
            << " print_develop_profile_captured=" << printBreakdown.captured
            << " print_develop_profile_kind=" << nonempty_cstr_or(printBreakdown.profileKind, "none")
            << " print_develop_profile_note=" << nonempty_cstr_or(printBreakdown.profileNote, "none")
            << " print_develop_total_launches=" << printBreakdown.total.launches
            << " print_develop_total_ms=" << printBreakdown.total.elapsedMs
            << " print_develop_spectral_integrate_launches="
            << printBreakdown.spectralIntegrate.launches
            << " print_develop_spectral_integrate_ms="
            << printBreakdown.spectralIntegrate.elapsedMs
            << " print_develop_exposure_scale_launches="
            << printBreakdown.exposureScale.launches
            << " print_develop_exposure_scale_ms="
            << printBreakdown.exposureScale.elapsedMs
            << " print_develop_log_encode_launches=" << printBreakdown.logEncode.launches
            << " print_develop_log_encode_ms=" << printBreakdown.logEncode.elapsedMs
            << " print_develop_density_curve_launches="
            << printBreakdown.densityCurve.launches
            << " print_develop_density_curve_ms="
            << printBreakdown.densityCurve.elapsedMs
            << " composite_scanner_linear_launches=" << compositeProfile.scannerLinear.launches
            << " composite_scanner_linear_ms=" << compositeProfile.scannerLinear.elapsedMs
            << " composite_output_encode_launches=" << compositeProfile.outputEncode.launches
            << " composite_output_encode_ms=" << compositeProfile.outputEncode.elapsedMs
            << " composite_glare_launches=" << compositeProfile.glare.launches
            << " composite_glare_ms=" << compositeProfile.glare.elapsedMs
            << " composite_lens_blur_launches=" << compositeProfile.lensBlur.launches
            << " composite_lens_blur_ms=" << compositeProfile.lensBlur.elapsedMs
            << " composite_unsharp_launches=" << compositeProfile.unsharp.launches
            << " composite_unsharp_ms=" << compositeProfile.unsharp.elapsedMs
            << " composite_grain_profile_kind=not_captured_by_scratch_replay"
            << " composite_grain_launches=0"
            << " composite_grain_ms=0.000"
            << " total_launches=" << profile.totalLaunches
            << " correction_launches=" << profile.correctionLaunches
            << " correction_ms=" << profile.correction.elapsedMs
            << " base_filter_launches=" << profile.baseFilterLaunches
            << " base_filter_ms=" << profile.baseFilter.elapsedMs
            << " tail0_filter_launches=" << profile.tailFilterLaunches[0]
            << " tail0_filter_ms=" << profile.tailFilter[0].elapsedMs
            << " tail1_filter_launches=" << profile.tailFilterLaunches[1]
            << " tail1_filter_ms=" << profile.tailFilter[1].elapsedMs
            << " tail2_filter_launches=" << profile.tailFilterLaunches[2]
            << " tail2_filter_ms=" << profile.tailFilter[2].elapsedMs
            << " scratch_source=" << (dirActive ? (scratchOverflow ? "overflow" : "retained") : "none")
            << " raw_correction_planes=" << (dirActive ? roles.rawCorrectionPlanes : 0)
            << " filtered_correction_planes=" << (dirActive ? roles.filteredCorrectionPlanes : 0)
            << " filter_temp_planes=" << (dirActive ? roles.filterTempPlanes : 0)
            << " cached_log_raw_planes=" << (dirActive ? roles.cachedLogRawPlanes : 0)
            << " spatial_dir_planes=" << (dirActive ? roles.total_float_planes() : 0)
            << " shared_tmp_planes=" << (dirActive && roles.filterTempPlanes > 0 ? 1 : 0)
            << " target_raw_correction_planes=" << (dirActive ? targetRoles.rawCorrectionPlanes : 0)
            << " target_filtered_correction_planes=" << (dirActive ? targetRoles.filteredCorrectionPlanes : 0)
            << " target_filter_temp_planes=" << (dirActive ? targetRoles.filterTempPlanes : 0)
            << " target_cached_log_raw_planes=" << (dirActive ? targetRoles.cachedLogRawPlanes : 0)
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

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    struct CudaFailureTrace {
        const char* prefix = "CUDA operation failed";
        const char* detail = nullptr;
    };

    inline void trace_cuda_fatal_prefixed_if(
        bool traceEnabled,
        const CudaFailureTrace& failure) {
        if (!traceEnabled) {
            return;
        }
        const char* safePrefix = nonempty_cstr_or(failure.prefix, "CUDA operation failed");
        const bool hasDetail = failure.detail && failure.detail[0] != '\0';
        std::string traceMsg;
        traceMsg.reserve(
            8 + std::strlen(safePrefix) + (hasDetail ? (2 + std::strlen(failure.detail)) : 0));
        traceMsg = "FATAL: ";
        traceMsg += safePrefix;
        if (hasDetail) {
            traceMsg += ": ";
            traceMsg += failure.detail;
        }
        JTRACE("CUDA", traceMsg);
    }

    inline void copy_float9(float dst[9], const float src[9]) {
        std::memcpy(dst, src, 9u * sizeof(float));
    }
#endif

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

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    // NOLINTBEGIN(bugprone-easily-swappable-parameters) Both route call sites share this fixed descriptor context order.
    bool build_visual_grain_descriptor_for_frame(
        const Spektrafilm::VisualGrainRecipe& recipe,
        const Spektrafilm::FilmDevelopRecipe& filmDevelop,
        Spektrafilm::ProfilePolarity capturePolarity,
        const Spektrafilm::VisualGrainFrameExtent& renderExtent,
        const Spektrafilm::VisualGrainFrameExtent& fullFrameExtent,
        float pixelSizeUm,
        double frameTime,
        double frameRate,
        std::uint64_t sessionSeed,
        std::uint64_t clipToken,
        std::optional<Spektrafilm::VisualGrainFrameDescriptor>& out,
        std::string& diagnostic) {
        out.reset();
        diagnostic.clear();
        if (!recipe.active) {
            return true;
        }
        Spektrafilm::VisualGrainFrameDescriptorInput input{};
        input.recipe = &recipe;
        input.filmDevelop = &filmDevelop;
        input.capturePolarity = capturePolarity;
        input.renderExtent = renderExtent;
        input.fullFrameExtent = fullFrameExtent;
        input.pixelSizeUm = pixelSizeUm;
        input.frameTime = frameTime;
        input.frameRate = frameRate;
        input.sessionSeed = sessionSeed;
        input.clipToken = clipToken;
        input.staticNoiseVersion =
            JuicerAssets::Library::kProcessAssetVersion;
        Spektrafilm::VisualGrainFrameDescriptor descriptor{};
        if (!Spektrafilm::build_visual_grain_frame_descriptor(
                input,
                descriptor,
                diagnostic)) {
            return false;
        }
        out = descriptor;
        return true;
    }

    bool build_film_juicer_effects_descriptor_for_frame(
        const Spektrafilm::FilmJuicerEffectsRecipe& recipe,
        const Spektrafilm::FilmJuicerEffectsFrameExtent& renderExtent,
        const Spektrafilm::FilmJuicerEffectsFrameExtent& fullFrameExtent,
        float pixelSizeUm,
        double frameTime,
        double frameRate,
        std::uint64_t sessionSeed,
        std::uint64_t clipToken,
        std::optional<Spektrafilm::FilmJuicerEffectsFrameDescriptor>& out,
        std::string& diagnostic) {
        out.reset();
        diagnostic.clear();
        if (!recipe.active) {
            return true;
        }
        Spektrafilm::FilmJuicerEffectsFrameDescriptor descriptor{};
        if (!Spektrafilm::build_film_juicer_effects_frame_descriptor(
                {&recipe,
                 renderExtent,
                 fullFrameExtent,
                 pixelSizeUm,
                 frameTime,
                 frameRate,
                 sessionSeed,
                 clipToken},
                descriptor,
                diagnostic)) {
            return false;
        }
        out = descriptor;
        return true;
    }

    // NOLINTEND(bugprone-easily-swappable-parameters)
    bool visual_grain_full_frame_preflight(
        const std::optional<Spektrafilm::VisualGrainFrameDescriptor>& descriptor,
        const OfxRectI& renderWindow,
        const OfxRectI& sourceBounds,
        const OfxRectI& fullFrameBounds) {
        if (!descriptor.has_value() || !descriptor->requiresFullFrame) {
            return true;
        }
        const Spektrafilm::VisualGrainFrameExtent& full =
            descriptor->fullFrameExtent;
        const auto matches = [&full](const OfxRectI& rect) {
            return rect.x1 == full.x && rect.y1 == full.y &&
                   rect.x2 - rect.x1 == full.width &&
                   rect.y2 - rect.y1 == full.height;
        };
        return descriptor->renderExtent.x == full.x &&
               descriptor->renderExtent.y == full.y &&
               descriptor->renderExtent.width == full.width &&
               descriptor->renderExtent.height == full.height &&
               matches(renderWindow) && matches(sourceBounds) &&
               matches(fullFrameBounds);
    }

    bool film_juicer_effects_full_frame_preflight(
        const std::optional<Spektrafilm::FilmJuicerEffectsFrameDescriptor>& descriptor,
        const OfxRectI& renderWindow,
        const OfxRectI& sourceBounds,
        const OfxRectI& fullFrameBounds) {
        if (!descriptor.has_value() || !descriptor->requiresFullFrame) {
            return true;
        }
        const Spektrafilm::FilmJuicerEffectsFrameExtent& full =
            descriptor->fullFrameExtent;
        const auto matches = [&full](const OfxRectI& rect) {
            return rect.x1 == full.x && rect.y1 == full.y &&
                   rect.x2 - rect.x1 == full.width &&
                   rect.y2 - rect.y1 == full.height;
        };
        return matches(renderWindow) && matches(sourceBounds) &&
               matches(fullFrameBounds);
    }
#endif

#if JUICER_DIAGNOSTICS_COMPILED
    const char* submission_snapshot_action_label(bool reusingSnapshotLatch) {
        return reusingSnapshotLatch ? "reuse" : "new";
    }
#endif

} // namespace

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
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
#endif

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
    : OFX::ImageProcessor(effect), _srcImg(nullptr), _nComponents(0), _scannerOptions{}, _scannerSettings{}, _printParams{}, _halationOverride{}, _hasHalationOverride(false), _printGlareOverride{}, _hasPrintGlareOverride(false), _dirRT{}, _prt(nullptr), _ws(nullptr), _wsReady(false), _printReady(false), _exposureScale(1.0f), _outputEncoding{}, _frameBoundsVersion(0), _pixelSizeUm(0.0f) {
}

void JuicerProcessor::setSrcDst(const SourceDestinationImages& images) {
    _srcImg = images.src;
    setDstImg(images.dst);
}

void JuicerProcessor::setDirectFrameRequest(const DirectFrameRequest& request) {
    setRenderWindowRect(request.renderWindow);
    _fullFrameExtent = request.fullFrameExtent;
    _diffusionFrameSetDescriptor = request.diffusionFrameSet;
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
    _diffusionFrameSetDescriptor = request.diffusionFrameSet;
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
    _diffusionFrameSetDescriptor.reset();
    setComponents(request.components);
    _scannerOptions = request.scannerOptions;
    _scannerSettings = request.scannerSettings;
    _printParams = request.printParams;
    _halationOverride = request.halationOverride;
    _hasHalationOverride = request.hasHalationOverride;
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
    _pixelSizeUm = is_finite(pixelSizeUm) ? std::max(0.0f, pixelSizeUm) : 0.0f;
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
                CudaFailureTrace{"cudaSetDevice failed", msg});
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }

        std::string contextError;
        if (!query_current_cuda_context(contextOpaque, contextError)) {
            trace_cuda_fatal_prefixed_if(
                traceInfo,
                CudaFailureTrace{
                    "failed to capture CUDA context identity",
                    cstr_or_null_if_empty(contextError)});
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

    if (_diffusionFrameSetDescriptor) {
        const auto frame_domain = [](const OfxRectI& bounds) {
            return Spektrafilm::DiffusionFrameDomain{
                bounds.x1,
                bounds.y1,
                bounds.x2 - bounds.x1,
                bounds.y2 - bounds.y1};
        };
        if (!Spektrafilm::diffusion_full_frame_matches(
                *_diffusionFrameSetDescriptor,
                frame_domain(win),
                frame_domain(srcBounds),
                frame_domain(_fullFrameExtent))) {
            JTRACE(
                "SPEKTRAFILM",
                "ResourceDescriptorMismatch component=diffusion field=full_frame_domain");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
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

    auto throw_cuda_policy_fatal = [&](const char* failurePrefix = nullptr,
                                       const char* detail = nullptr) {
        trace_cuda_fatal_prefixed_if(
            traceInfo,
            CudaFailureTrace{
                nonempty_cstr_or(failurePrefix, "CUDA render cannot continue"),
                detail});
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
            CudaFailureTrace{
                nonempty_cstr_or(failurePrefix, "submission stage failed"),
                cstr_or_null_if_empty(error)});
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    };

    auto record_cuda_use = [&](JuicerProcess::Root::PreparedCudaFrame& frame) {
        std::string useError;
        if (!frame.record_use(_pCudaStream, useError)) {
            frame.abort("prepared_frame_use_fence_failed");
            throw_submission_fatal(
                "prepared_frame_use_fence",
                "CUDA prepared-frame use fencing failed",
                useError);
        }
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

        std::optional<Spektrafilm::VisualGrainFrameDescriptor>
            directVisualGrainDescriptor;
        std::string directGrainDescriptorDiagnostic;
        std::chrono::steady_clock::time_point directGrainDescriptorStart{};
        if (dirProfileEnabled) {
            directGrainDescriptorStart = std::chrono::steady_clock::now();
        }
        if (!build_visual_grain_descriptor_for_frame(
                directRecipe->visualGrain,
                directRecipe->filmDevelop,
                directRecipe->profileRoute.capturePolarity,
                {win.x1, win.y1, width, height},
                {directFullFrameRect.x1,
                 directFullFrameRect.y1,
                 directFullFrameRect.x2 - directFullFrameRect.x1,
                 directFullFrameRect.y2 - directFullFrameRect.y1},
                _pixelSizeUm,
                _timeFrames,
                _frameRate,
                _sessionSeed,
                static_cast<std::uint64_t>(_clipToken),
                directVisualGrainDescriptor,
                directGrainDescriptorDiagnostic)) {
            throw_direct_restriction(
                directGrainDescriptorDiagnostic.c_str());
        }
        const double directGrainDescriptorMs =
            dirProfileEnabled
                ? elapsed_ms_since(directGrainDescriptorStart)
                : 0.0;
        if (!visual_grain_full_frame_preflight(
                directVisualGrainDescriptor,
                win,
                srcBounds,
                directFullFrameRect)) {
            throw_direct_restriction(
                "ResourceDescriptorMismatch phase=grain_preflight field=full_frame_extent");
        }
        std::optional<Spektrafilm::FilmJuicerEffectsFrameDescriptor>
            directEffectsDescriptor;
        std::string directEffectsDescriptorDiagnostic;
        if (!build_film_juicer_effects_descriptor_for_frame(
                directRecipe->filmJuicerEffects,
                {win.x1, win.y1, width, height},
                {directFullFrameRect.x1,
                 directFullFrameRect.y1,
                 directFullFrameRect.x2 - directFullFrameRect.x1,
                 directFullFrameRect.y2 - directFullFrameRect.y1},
                _pixelSizeUm,
                _timeFrames,
                _frameRate,
                _sessionSeed,
                static_cast<std::uint64_t>(_clipToken),
                directEffectsDescriptor,
                directEffectsDescriptorDiagnostic)) {
            throw_direct_restriction(
                directEffectsDescriptorDiagnostic.c_str());
        }
        if (!film_juicer_effects_full_frame_preflight(
                directEffectsDescriptor,
                win,
                srcBounds,
                directFullFrameRect)) {
            throw_direct_restriction(
                "ResourceDescriptorMismatch phase=effects_preflight field=full_frame_extent");
        }
        const bool grainStageActive =
            directVisualGrainDescriptor.has_value();
        const bool grainDebugActive =
            grainStageActive && directRecipe->visualGrain.debugView != 0;
        const bool filmEffectsActive =
            directEffectsDescriptor.has_value() &&
            directEffectsDescriptor->filmActive;
        const bool gateOutputActive =
            directEffectsDescriptor.has_value() &&
            directEffectsDescriptor->gateOutputActive;
        const bool cameraDiffusionActive =
            _diffusionFrameSetDescriptor.has_value() &&
            _diffusionFrameSetDescriptor->camera.has_value();
        const bool useFocusedSplit =
            cameraDiffusionActive || scannerPostEffects.active() || grainStageActive ||
            filmEffectsActive || gateOutputActive;
        const bool captureDensityConsumerActive =
            grainStageActive || filmEffectsActive;

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
        directPreparation.diffusionFrameSetDescriptor =
            _diffusionFrameSetDescriptor
                ? &*_diffusionFrameSetDescriptor
                : nullptr;
        directPreparation.visualGrainDescriptor =
            directVisualGrainDescriptor;
        directPreparation.effectsDescriptor = directEffectsDescriptor;
        directPreparation.requestedWidth = width;
        directPreparation.requestedHeight = height;
        directPreparation.needCompositeProfileWorkspace =
            dirProfileEnabled && !cameraDiffusionActive;

        std::string directPrepareError;
        std::chrono::steady_clock::time_point directPreparedFrameStart{};
        if (dirProfileEnabled) {
            directPreparedFrameStart = std::chrono::steady_clock::now();
        }
        JuicerProcess::Root::PreparedCudaFrame preparedFrame =
            JuicerProcess::root().prepare_cuda_frame(
                deviceContextKey,
                snapshot,
                directPreparation,
                autoExposureBufferRequest,
                _pCudaStream,
                directPrepareError);
        const double directPreparedFramePrepareMs =
            dirProfileEnabled
                ? elapsed_ms_since(directPreparedFrameStart)
                : 0.0;
        if (!preparedFrame.active()) {
            throw_submission_fatal(
                preparedFrame.failure_stage_tag(),
                preparedFrame.failure_prefix(),
                directPrepareError);
        }

        const auto diffusionPrepared = preparedFrame.diffusion_resources();
        DiffusionStageBinding directCameraDiffusion{};
        if (_diffusionFrameSetDescriptor) {
            if (!diffusionPrepared.active ||
                diffusionPrepared.executionDescriptor.hash == 0 ||
                diffusionPrepared.executionDescriptor.frameSetHash !=
                    _diffusionFrameSetDescriptor->hash ||
                diffusionPrepared.executionDescriptor.contextEpoch !=
                    snapshot.contextEpoch ||
                diffusionPrepared.spectrumCount == 0 ||
                diffusionPrepared.spectrumCount !=
                    diffusionPrepared.executionDescriptor.uniqueSpectrumCount) {
                preparedFrame.abort("direct_diffusion_prepared_view_mismatch");
                throw_direct_restriction(
                    "MissingRequiredResource component=diffusion field=prepared_view");
            }
            if (!_diffusionFrameSetDescriptor->camera ||
                _diffusionFrameSetDescriptor->enlarger) {
                const char* seam = _diffusionFrameSetDescriptor->camera
                                       ? "unexpected_enlarger_print_linear_exposure"
                                       : "missing_camera_film_linear_exposure";
                std::string diagnostic =
                    "ExactDiffusionStageBoundaryUnavailable route=";
                diagnostic += Spektrafilm::scan_route_label(
                    _diffusionFrameSetDescriptor->route);
                diagnostic += " stage=camera seam=";
                diagnostic += seam;
                diagnostic += " frame_set_hash=";
                diagnostic +=
                    std::to_string(_diffusionFrameSetDescriptor->hash);
                preparedFrame.abort(
                    "direct_diffusion_stage_boundary_unavailable");
                throw_direct_restriction(diagnostic.c_str());
            }
            std::string cameraBindingDiagnostic;
            if (!bind_diffusion_stage(
                    *_diffusionFrameSetDescriptor,
                    *_diffusionFrameSetDescriptor->camera,
                    Spektrafilm::DiffusionLinearStage::CameraFilmLinear,
                    diffusionPrepared,
                    directCameraDiffusion,
                    cameraBindingDiagnostic)) {
                preparedFrame.abort("direct_diffusion_camera_binding_failed");
                throw_direct_restriction(cameraBindingDiagnostic.c_str());
            }
        } else if (diffusionPrepared.active) {
            preparedFrame.abort("direct_disabled_diffusion_view_active");
            throw_direct_restriction(
                "ResourceDescriptorMismatch component=diffusion field=disabled_prepared_view");
        }

        const JuicerProcess::Root::PreparedCudaFrame::DirectPreparedView prepared =
            preparedFrame.direct_resources();
        JuicerProcess::Root::PreparedCudaFrame::UploadTraceView directUploadTrace{};
        if (dirProfileEnabled) {
            directUploadTrace = preparedFrame.upload_trace_view();
        }
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
            JuicerCuda::AutoExposureSourceFormat autoExposureSourceFormat{};
            autoExposureSourceFormat.componentCount = run.nComponents;
            autoExposureSourceFormat.inputColorSpaceIndex = directRecipe->filmRaw.inputColorSpace;
            autoExposureSourceFormat.applyCctfDecoding = bool_to_i32(directRecipe->filmRaw.inputCctfDecoding);
            const int meterRc = juicer_cuda_auto_exposure_meter_to_device(
                srcBase,
                static_cast<std::size_t>(srcRowBytes),
                autoExposureDescriptor,
                autoExposureSourceFormat,
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
                directRecipe->densityBounds,
                prepared.film,
                autoExposureScaleDevice,
                scannerCorrection.exposureScale,
                directFilmPayloads,
                packDiagnostic)) {
            preparedFrame.abort("direct_film_payload_pack_failed");
            throw_direct_restriction(packDiagnostic.c_str());
        }
        run.filmRaw = directFilmPayloads.filmRaw;
        run.filmExpose = directFilmPayloads.filmExposure;
        run.filmDevelop = directFilmPayloads.filmDevelop;

        JuicerCuda::CameraFilmLinearExposurePlanes directCameraFilmLinear{};
        if (cameraDiffusionActive) {
            if (!directCameraDiffusion.active) {
                preparedFrame.abort("direct_diffusion_camera_binding_inactive");
                throw_direct_restriction(
                    "MissingRequiredResource component=diffusion field=camera_stage_binding");
            }
            preparedFrame.mark_diffusion_work_enqueued();
            directCameraFilmLinear = camera_film_linear_planes(
                directCameraDiffusion.stagePlanes);
            const cudaError_t exposureError =
                juicer_cuda_direct_camera_film_linear_exposure(
                    &run,
                    directCameraFilmLinear,
                    _pCudaStream);
            if (exposureError != cudaSuccess) {
                preparedFrame.abort(
                    "direct_camera_film_linear_exposure_launch_failed");
                throw_cuda_stage_fatal(
                    "direct_camera_film_linear_exposure_launch",
                    "direct camera film-linear exposure launch failed",
                    exposureError);
            }
            JuicerCuda::Diffusion::StageLaunchRequest diffusionLaunch{};
            diffusionLaunch.layout = directCameraDiffusion.layout;
            diffusionLaunch.geometry = directCameraDiffusion.geometry;
            diffusionLaunch.fullFrame = directCameraDiffusion.fullFrame;
            diffusionLaunch.spectra = directCameraDiffusion.spectra;
            diffusionLaunch.execution = directCameraDiffusion.execution;
            diffusionLaunch.planes = &directCameraDiffusion.stagePlanes;
            diffusionLaunch.stream = reinterpret_cast<cudaStream_t>(
                _pCudaStream);
            const JuicerCuda::Diffusion::LaunchResult diffusionResult =
                JuicerCuda::Diffusion::launch_stage(diffusionLaunch);
            if (!diffusionResult.ok()) {
                preparedFrame.abort("direct_camera_diffusion_launch_failed");
                if (diffusionResult.api ==
                    JuicerCuda::Diffusion::FailureApi::Cuda) {
                    throw_cuda_stage_fatal(
                        diffusionResult.stage,
                        "direct camera diffusion launch failed",
                        static_cast<cudaError_t>(diffusionResult.code));
                }
                const std::string diagnostic =
                    diffusion_launch_failure_diagnostic(diffusionResult);
                throw_direct_restriction(diagnostic.c_str());
            }
            // launch_stage rotates semantic plane pointers after each channel.
            directCameraFilmLinear = camera_film_linear_planes(
                directCameraDiffusion.stagePlanes);
        }

        const bool directCompositeProfileRequested =
            dirProfileEnabled && !cameraDiffusionActive;
        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceRequest workspaceRequest =
            preparedFrame.workspace_request();
        const bool directUseFusedScannerPostSpatialDirHandoff =
            !cameraDiffusionActive && directSpatialDir.hash != 0 &&
            scannerPostEffects.active() &&
            directSpatialDir.approximation ==
                Spektrafilm::DirApproximationMarker::SpektrafilmStrict &&
            !captureDensityConsumerActive;
        JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker focusedWorkspace{};
        if (workspaceRequest.has_any_family()) {
            focusedWorkspace = preparedFrame.bind_workspace_request(workspaceRequest);
            std::string transitionError;
            if (!preparedFrame.checkpoint_large_scratch_transition(
                    focusedWorkspace,
                    _pCudaStream,
                    "direct_large_scratch_transition",
                    transitionError)) {
                preparedFrame.abort("direct_large_scratch_transition_failed");
                throw_submission_fatal(
                    "direct_large_scratch_transition",
                    "direct large scratch transition checkpoint failed",
                    transitionError);
            }
        }
        JuicerCuda::SpatialDirBuildProfile directDirProfile{};
        bool directDirProfileCaptured = false;
        bool directDirScratchOverflow = false;
        Spektrafilm::DirScratchPlaneRoles directDirAdmittedRoles = directSpatialDir.planeRoles;
        Spektrafilm::DirScratchPlaneRoles directDirAdmittedTargetRoles = directSpatialDir.targetPlaneRoles;
        bool directDirUsesSourceBuildCachedLogRaw = false;
        double directDirPrepareMs = 0.0;
        double directDirBuildHostMs = 0.0;
        double directPreparedFrameFinishMs = 0.0;
        double directPipelineLaunchHostMs = 0.0;
        float directPipelineCudaMs = -1.0f;
        double directGrainSubmitHostMs = 0.0;
        JuicerCuda::VisualGrainRuntimeProfile directGrainProfile{};
        JuicerCuda::CompositePipelineProfile directCompositeProfile{};
        bool directCaptureDensityReady = false;
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
        }
        JuicerProcess::Root::PreparedCudaFrame::CaptureFilmDensityWorkspaceView
            directCaptureDensity{};
        JuicerProcess::Root::PreparedCudaFrame::FocusedRgbWorkspaceView
            directFocusedRgb{};
        JuicerProcess::Root::PreparedCudaFrame::VisualGrainWorkspaceView
            directGrainWorkspace{};
        JuicerProcess::Root::PreparedCudaFrame::PreparedVisualGrainView
            directGrainResources{};
        JuicerProcess::Root::PreparedCudaFrame::ScannerWorkspaceView
            directEffectsWorkspace{};
        JuicerCuda::GrainPayload directGrainPayload{};
        JuicerCuda::GrainKernelPayload directGrainKernels{};
        JuicerCuda::GrainPayload directEffectsPayload{};
        JuicerCuda::GateWeavePayload directWeavePayload{};
        const Spektrafilm::FilmJuicerEffectsFrameDescriptor*
            preparedDirectEffectsDescriptor =
                preparedFrame.film_juicer_effects_descriptor();
        if (directEffectsDescriptor.has_value()) {
            std::string effectsPayloadDiagnostic;
            if (!preparedDirectEffectsDescriptor ||
                !JuicerCuda::pack_film_juicer_effects_payload(
                    *preparedDirectEffectsDescriptor,
                    directEffectsPayload,
                    directWeavePayload,
                    effectsPayloadDiagnostic)) {
                preparedFrame.abort("direct_effects_payload_pack_failed");
                throw_direct_restriction(
                    effectsPayloadDiagnostic.empty()
                        ? "MissingRequiredResource phase=effects_route field=effects_binding"
                        : effectsPayloadDiagnostic.c_str());
            }
        } else if (preparedDirectEffectsDescriptor) {
            preparedFrame.abort("direct_inactive_effects_descriptor_present");
            throw_direct_restriction(
                "ResourceDescriptorMismatch phase=effects_route field=inactive_descriptor");
        }
        if (useFocusedSplit) {
            std::string focusedWorkspaceError;
            if (!preparedFrame.stage_optical_workspace(
                    focusedWorkspace,
                    _pCudaStream,
                    focusedWorkspaceError)) {
                preparedFrame.abort("direct_focused_workspace_stage_failed");
                throw_submission_fatal(
                    "direct_focused_workspace_stage",
                    "direct focused workspace staging failed",
                    focusedWorkspaceError);
            }
            directCaptureDensity =
                preparedFrame.capture_film_density_workspace(
                    focusedWorkspace);
            directFocusedRgb = preparedFrame.focused_rgb_workspace(
                focusedWorkspace);
            if (!directCaptureDensity.active || !directFocusedRgb.active ||
                directCaptureDensity.c != directFocusedRgb.r ||
                directCaptureDensity.m != directFocusedRgb.g ||
                directCaptureDensity.y != directFocusedRgb.b) {
                preparedFrame.abort("direct_focused_triplet_binding_failed");
                throw_direct_restriction(
                    "MissingRequiredResource phase=grain_route field=focused_triplet");
            }
            directEffectsWorkspace =
                preparedFrame.scanner_workspace(focusedWorkspace);
            if (!directEffectsWorkspace.active ||
                (directEffectsDescriptor.has_value() &&
                 directEffectsDescriptor->gateMaskActive &&
                 !directEffectsWorkspace.hasGateMask)) {
                preparedFrame.abort("direct_effects_workspace_binding_failed");
                throw_direct_restriction(
                    "MissingRequiredResource phase=effects_route field=effects_workspace");
            }
        }
        if (grainStageActive) {
            directGrainResources = preparedFrame.visual_grain_resources();
            directGrainWorkspace = preparedFrame.visual_grain_workspace(
                focusedWorkspace);
            std::string grainPayloadDiagnostic;
            if (!directGrainResources.active ||
                !directGrainWorkspace.active ||
                !JuicerCuda::pack_visual_grain_payload(
                    directRecipe->visualGrain,
                    directGrainResources,
                    directGrainPayload,
                    directGrainKernels,
                    grainPayloadDiagnostic)) {
                preparedFrame.abort("direct_visual_grain_payload_pack_failed");
                throw_direct_restriction(
                    grainPayloadDiagnostic.empty()
                        ? "MissingRequiredResource phase=grain_route field=grain_binding"
                        : grainPayloadDiagnostic.c_str());
            }
            directGrainPayload.frameUniforms =
                directGrainWorkspace.frameUniforms;
        }
        if (directSpatialDir.hash != 0) {
            const auto scratch = preparedFrame.spatial_dir_scratch(focusedWorkspace);
            const auto resources =
                preparedFrame.spatial_dir_resources(focusedWorkspace, directSpatialDir.hash);
            if (!scratch.active || !resources.active) {
                preparedFrame.abort("direct_spatial_dir_binding_failed");
                throw_direct_restriction(
                    "MissingRequiredResource phase=3D-3 field=prepared_spatial_dir");
            }
            directDirScratchOverflow = scratch.overflow;
            directDirAdmittedRoles = scratch.planeRoles;
            directDirAdmittedTargetRoles = scratch.targetPlaneRoles;
            directDirUsesSourceBuildCachedLogRaw =
                fused_alias_uses_source_build_cached_log_raw(
                    directUseFusedScannerPostSpatialDirHandoff,
                    directDirAdmittedRoles,
                    directDirAdmittedTargetRoles);
            if (directDirUsesSourceBuildCachedLogRaw &&
                !spatial_dir_required_cached_log_raw_present(
                    scratch,
                    directDirAdmittedRoles.cachedLogRawPlanes)) {
                preparedFrame.abort("direct_spatial_dir_source_build_cached_log_raw_missing");
                throw_direct_restriction(
                    "MissingRequiredResource phase=3D-3 field=spatial_dir_source_build_cached_log_raw");
            }
            float* cachedLogRawB =
                directDirUsesSourceBuildCachedLogRaw ? scratch.logRawB : nullptr;
            float* cachedLogRawG =
                directDirUsesSourceBuildCachedLogRaw ? scratch.logRawG : nullptr;
            float* cachedLogRawR =
                directDirUsesSourceBuildCachedLogRaw ? scratch.logRawR : nullptr;
            JuicerCuda::SpatialDirBuildRequest dirBuildRequest{};
            dirBuildRequest.planes.rawCorrectionY = scratch.rawCorrectionY;
            dirBuildRequest.planes.rawCorrectionM = scratch.rawCorrectionM;
            dirBuildRequest.planes.rawCorrectionC = scratch.rawCorrectionC;
            dirBuildRequest.planes.filteredCorrectionY = scratch.filteredCorrectionY;
            dirBuildRequest.planes.filteredCorrectionM = scratch.filteredCorrectionM;
            dirBuildRequest.planes.filteredCorrectionC = scratch.filteredCorrectionC;
            dirBuildRequest.planes.filterTemp = scratch.filterTemp;
            dirBuildRequest.planes.filterTempM = scratch.filterTempM;
            dirBuildRequest.planes.filterTempC = scratch.filterTempC;
            dirBuildRequest.planes.logRawB = cachedLogRawB;
            dirBuildRequest.planes.logRawG = cachedLogRawG;
            dirBuildRequest.planes.logRawR = cachedLogRawR;
            dirBuildRequest.cameraFilmLinear = directCameraFilmLinear;
            dirBuildRequest.gaussian.kernel = resources.gaussian.weights;
            dirBuildRequest.gaussian.radius = resources.gaussian.radius;
            dirBuildRequest.gaussian.sigma = resources.gaussian.sigma;
            dirBuildRequest.gaussian.weight = directSpatialDir.gaussianWeight;
            for (int tailIndex = 0; tailIndex < 3; ++tailIndex) {
                dirBuildRequest.tails[tailIndex].kernel = resources.exponential[tailIndex].weights;
                dirBuildRequest.tails[tailIndex].radius = resources.exponential[tailIndex].radius;
                dirBuildRequest.tails[tailIndex].sigma = resources.exponential[tailIndex].sigma;
                dirBuildRequest.tails[tailIndex].weight = directSpatialDir.exponentialWeights[tailIndex];
            }
            dirBuildRequest.streamOpaque = _pCudaStream;
            dirBuildRequest.profile = dirProfileEnabled ? &directDirProfile : nullptr;
            const auto directDirBuildStart = std::chrono::steady_clock::now();
            const cudaError_t dirError = juicer_cuda_build_direct_spatial_dir(
                &run,
                dirBuildRequest);
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
            std::string releaseError;
            if (!preparedFrame.release_spatial_dir_build_scratch_after_build(
                    focusedWorkspace,
                    _pCudaStream,
                    releaseError)) {
                preparedFrame.abort("direct_spatial_dir_build_scratch_release_failed");
                throw_submission_fatal(
                    "direct_spatial_dir_build_scratch_release",
                    "direct spatial DIR build scratch release failed",
                    releaseError);
            }
            if (!directUseFusedScannerPostSpatialDirHandoff) {
                if (!preparedFrame.stage_spatial_dir_cached_log_raw_for_final_develop(
                        focusedWorkspace,
                        _pCudaStream,
                        releaseError)) {
                    preparedFrame.abort("direct_spatial_dir_cached_log_raw_stage_failed");
                    throw_submission_fatal(
                        "direct_spatial_dir_cached_log_raw_stage",
                        "direct spatial DIR cached log raw staging failed",
                        releaseError);
                }
            }
            const auto finalScratch = preparedFrame.spatial_dir_scratch(focusedWorkspace);
            const bool directDirRequiresCachedLogRaw =
                directDirUsesSourceBuildCachedLogRaw ||
                (!directUseFusedScannerPostSpatialDirHandoff &&
                 directDirAdmittedTargetRoles.cachedLogRawPlanes == 3);
            if (!finalScratch.filteredCorrectionY || !finalScratch.filteredCorrectionM ||
                !finalScratch.filteredCorrectionC) {
                preparedFrame.abort("direct_spatial_dir_filtered_correction_missing");
                throw_direct_restriction(
                    "MissingRequiredResource phase=3D-3 field=spatial_dir_filtered_correction");
            }
            if (directDirRequiresCachedLogRaw &&
                !spatial_dir_required_cached_log_raw_present(
                    finalScratch,
                    directDirAdmittedTargetRoles.cachedLogRawPlanes)) {
                preparedFrame.abort("direct_spatial_dir_cached_log_raw_missing");
                throw_direct_restriction(
                    "MissingRequiredResource phase=3D-3 field=spatial_dir_cached_log_raw");
            }
            if (directDirRequiresCachedLogRaw && !directDirUsesSourceBuildCachedLogRaw) {
                const cudaError_t logRawError = cameraDiffusionActive
                                                    ? juicer_cuda_build_direct_spatial_dir_cached_log_raw_from_camera_film_linear(
                                                          &run,
                                                          directCameraFilmLinear,
                                                          finalScratch.logRawB,
                                                          finalScratch.logRawG,
                                                          finalScratch.logRawR,
                                                          _pCudaStream)
                                                    : juicer_cuda_build_direct_spatial_dir_cached_log_raw(
                                                          &run,
                                                          finalScratch.logRawB,
                                                          finalScratch.logRawG,
                                                          finalScratch.logRawR,
                                                          _pCudaStream);
                if (logRawError != cudaSuccess) {
                    preparedFrame.abort("direct_spatial_dir_cached_log_raw_launch_failed");
                    throw_cuda_stage_fatal(
                        "direct_spatial_dir_cached_log_raw_launch",
                        "direct spatial DIR cached log raw build failed",
                        logRawError);
                }
            }
            bind_spatial_dir_final_develop_to_payload(run.filmDevelop, finalScratch);
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
        run.scanStage.scanLutLog2PchipXYZ = prepared.scanLut->log2PchipXYZ;
        run.scanStage.scanLutPchipSlopeC = prepared.scanLut->slopeC;
        run.scanStage.scanLutPchipSlopeM = prepared.scanLut->slopeM;
        run.scanStage.scanLutPchipSlopeY = prepared.scanLut->slopeY;
        run.scanStage.scanLutPchipCellMin = prepared.scanLut->cellMin;
        run.scanStage.scanLutPchipCellMax = prepared.scanLut->cellMax;
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
        if (directCompositeProfileRequested &&
            directUseFusedScannerPostSpatialDirHandoff &&
            workspaceRequest.aliasScannerRgbFromSpatialDirFiltered) {
            initialize_spatial_dir_rgb_alias_profile(
                directCompositeProfile,
                AliasRouteProfileExtent{width, height});
        } else if (directCompositeProfileRequested &&
                   !workspaceRequest.aliasScannerRgbFromSpatialDirFiltered) {
            std::string profileWorkspaceError;
            if (preparedFrame.try_stage_profile_optical_workspace(
                    focusedWorkspace,
                    _pCudaStream,
                    profileWorkspaceError)) {
                const auto profileScratch = preparedFrame.scanner_workspace(focusedWorkspace);
                if (profileScratch.active) {
                    const cudaError_t profileError = juicer_cuda_profile_direct_focused_pipeline_stages(
                        &run,
                        profileScratch.rgbR,
                        profileScratch.rgbG,
                        profileScratch.rgbB,
                        &directCompositeProfile,
                        _pCudaStream);
                    if (profileError != cudaSuccess) {
                        directCompositeProfile = JuicerCuda::CompositePipelineProfile{};
                        directCompositeProfile.profileKind = "focused_split_attribution";
                        directCompositeProfile.profileNote = "profile_failed";
                    }
                } else {
                    directCompositeProfile.profileKind = "focused_split_attribution";
                    directCompositeProfile.profileNote = "missing_profile_scratch";
                }
            } else {
                directCompositeProfile.profileKind = "focused_split_attribution";
                directCompositeProfile.profileNote = "profile_workspace_failed";
            }
        } else if (directCompositeProfileRequested) {
            directCompositeProfile.profileKind = "focused_split_attribution";
            directCompositeProfile.profileNote =
                "scratch_replay_skipped_live_spatial_dir_alias";
        }
        cudaError_t launchError = cudaSuccess;
        const auto directPipelineLaunchStart = std::chrono::steady_clock::now();
        CudaEventElapsedTimer directPipelineCudaTimer;
        if (dirProfileEnabled) {
            directPipelineCudaTimer.begin(_pCudaStream);
        }
        if (useFocusedSplit &&
            !directUseFusedScannerPostSpatialDirHandoff) {
            launchError = cameraDiffusionActive
                              ? juicer_cuda_direct_focused_capture_density_from_camera_film_linear(
                                    &run,
                                    directCameraFilmLinear,
                                    directCaptureDensity.c,
                                    directCaptureDensity.m,
                                    directCaptureDensity.y,
                                    _pCudaStream)
                              : juicer_cuda_direct_focused_capture_density(
                                    &run,
                                    directCaptureDensity.c,
                                    directCaptureDensity.m,
                                    directCaptureDensity.y,
                                    _pCudaStream);
            if (launchError != cudaSuccess) {
                preparedFrame.abort("direct_capture_density_launch_failed");
                throw_cuda_stage_fatal(
                    "direct_capture_density_launch",
                    "direct capture-film density launch failed",
                    launchError);
            }
            directCaptureDensityReady = true;
            if (directSpatialDir.hash != 0) {
                std::string releaseError;
                if (!preparedFrame.release_spatial_dir_cached_log_raw_after_final_develop(
                        focusedWorkspace,
                        _pCudaStream,
                        releaseError)) {
                    preparedFrame.abort("direct_spatial_dir_cached_log_raw_release_failed");
                    throw_submission_fatal(
                        "direct_spatial_dir_cached_log_raw_release",
                        "direct spatial DIR cached log raw release failed",
                        releaseError);
                }
                run.filmDevelop.spatialDir.logRawB = nullptr;
                run.filmDevelop.spatialDir.logRawG = nullptr;
                run.filmDevelop.spatialDir.logRawR = nullptr;
            }
            if (grainStageActive) {
                CudaEventElapsedTimer directGrainCudaTimer;
                std::chrono::steady_clock::time_point directGrainSubmitStart{};
                if (dirProfileEnabled) {
                    directGrainCudaTimer.begin(_pCudaStream);
                    directGrainSubmitStart = std::chrono::steady_clock::now();
                }
                launchError = juicer_cuda_apply_visual_grain(
                    &directGrainPayload,
                    &directGrainKernels,
                    width,
                    height,
                    directCaptureDensity.c,
                    directCaptureDensity.m,
                    directCaptureDensity.y,
                    directGrainWorkspace.filterTemp,
                    directGrainWorkspace.scaleWork,
                    directGrainWorkspace.deltaAccum,
                    directGrainWorkspace.layerWork,
                    directGrainWorkspace.sharedDelta,
                    dirProfileEnabled ? &directGrainProfile : nullptr,
                    _pCudaStream);
                if (dirProfileEnabled) {
                    directGrainSubmitHostMs =
                        elapsed_ms_since(directGrainSubmitStart);
                    directGrainProfile.total.elapsedMs =
                        directGrainCudaTimer.finish(_pCudaStream);
                    directGrainProfile.total.launches =
                        directGrainProfile.totalLaunches;
                }
                if (launchError != cudaSuccess) {
                    preparedFrame.abort("direct_visual_grain_launch_failed");
                    throw_cuda_stage_fatal(
                        "direct_visual_grain_launch",
                        "direct visual grain launch failed",
                        launchError);
                }
            }
            if (filmEffectsActive && !grainDebugActive) {
                launchError = juicer_cuda_apply_film_defects(
                    &directEffectsPayload,
                    width,
                    height,
                    directCaptureDensity.c,
                    directCaptureDensity.m,
                    directCaptureDensity.y,
                    _pCudaStream);
                if (launchError != cudaSuccess) {
                    preparedFrame.abort("direct_film_effects_launch_failed");
                    throw_cuda_stage_fatal(
                        "direct_film_effects_launch",
                        "direct film defects launch failed",
                        launchError);
                }
            }
        }

        if (cameraDiffusionActive) {
            std::string diffusionReleaseError;
            if (!preparedFrame.release_diffusion_resources_after_use(
                    _pCudaStream,
                    diffusionReleaseError)) {
                preparedFrame.abort("direct_diffusion_release_after_use_failed");
                throw_submission_fatal(
                    "direct_diffusion_release_after_use",
                    "direct diffusion resource release failed",
                    diffusionReleaseError);
            }
            directCameraDiffusion = {};
            directCameraFilmLinear = {};
        }

        if (gateOutputActive && !grainDebugActive &&
            directEffectsDescriptor->gateMaskActive) {
            launchError = juicer_cuda_build_gate_defect_mask_focused(
                &directEffectsPayload,
                directEffectsWorkspace.gateMask,
                directEffectsWorkspace.gateMaskWidth,
                directEffectsWorkspace.gateMaskHeight,
                _pCudaStream);
            if (launchError != cudaSuccess) {
                preparedFrame.abort("direct_gate_mask_launch_failed");
                throw_cuda_stage_fatal(
                    "direct_gate_mask_launch",
                    "direct gate defect mask launch failed",
                    launchError);
            }
            preparedFrame.mark_gate_mask_built(
                directEffectsDescriptor->hash);
        }

        if (useFocusedSplit) {
            JuicerProcess::Root::PreparedCudaFrame::ScannerPostEffectsPreparedView
                post{};
            if (scannerPostEffects.active() && !grainDebugActive) {
                post = preparedFrame.scanner_post_effects_resources(
                    focusedWorkspace,
                    scannerPostEffects.hash);
                if (!post.active ||
                    post.scratch.rgbR != directFocusedRgb.r ||
                    post.scratch.rgbG != directFocusedRgb.g ||
                    post.scratch.rgbB != directFocusedRgb.b) {
                    preparedFrame.abort("direct_scanner_post_effects_view_failed");
                    throw_direct_restriction(
                        "MissingRequiredResource phase=8C field=scanner_post_effects");
                }
            }
            const bool directAliasProfileActive =
                dirProfileEnabled &&
                directUseFusedScannerPostSpatialDirHandoff &&
                workspaceRequest.aliasScannerRgbFromSpatialDirFiltered &&
                !grainDebugActive;
            if (!grainDebugActive) {
                if (directUseFusedScannerPostSpatialDirHandoff) {
                    CudaEventElapsedTimer directAliasScanLinearTimer;
                    if (directAliasProfileActive) {
                        directAliasScanLinearTimer.begin(_pCudaStream);
                    }
                    launchError = juicer_cuda_direct_focused_scan_linear_rgb(
                        &run,
                        directFocusedRgb.r,
                        directFocusedRgb.g,
                        directFocusedRgb.b,
                        directAliasProfileActive ? &directCompositeProfile : nullptr,
                        _pCudaStream);
                    if (launchError == cudaSuccess && directAliasProfileActive) {
                        record_alias_route_stage(
                            directCompositeProfile,
                            directCompositeProfile.aliasFusedScanLinear,
                            directAliasScanLinearTimer,
                            _pCudaStream);
                    }
                } else {
                    if (!directCaptureDensityReady) {
                        preparedFrame.abort("direct_capture_density_not_ready");
                        throw_direct_restriction(
                            "MissingRequiredResource phase=grain_route field=capture_density");
                    }
                    launchError = juicer_cuda_direct_focused_scan_linear_density_rgb(
                        &run,
                        directCaptureDensity.c,
                        directCaptureDensity.m,
                        directCaptureDensity.y,
                        directFocusedRgb.r,
                        directFocusedRgb.g,
                        directFocusedRgb.b,
                        _pCudaStream);
                }
                if (launchError != cudaSuccess) {
                    preparedFrame.abort("direct_scanner_linear_launch_failed");
                    throw_cuda_stage_fatal(
                        "direct_scanner_linear_launch",
                        "direct scanner-linear launch failed",
                        launchError);
                }
            }
            if (directDirUsesSourceBuildCachedLogRaw) {
                std::string releaseError;
                if (!preparedFrame.release_spatial_dir_cached_log_raw_after_final_develop(
                        focusedWorkspace,
                        _pCudaStream,
                        releaseError)) {
                    preparedFrame.abort("direct_spatial_dir_cached_log_raw_release_failed");
                    throw_submission_fatal(
                        "direct_spatial_dir_cached_log_raw_release",
                        "direct spatial DIR cached log raw release failed",
                        releaseError);
                }
                run.filmDevelop.spatialDir.logRawB = nullptr;
                run.filmDevelop.spatialDir.logRawG = nullptr;
                run.filmDevelop.spatialDir.logRawR = nullptr;
            }
            CudaEventElapsedTimer directAliasPostOutputTimer;
            if (directAliasProfileActive) {
                directAliasPostOutputTimer.begin(_pCudaStream);
            }
            launchError = juicer_cuda_direct_focused_scanner_post_output(
                &run,
                directFocusedRgb.r,
                directFocusedRgb.g,
                directFocusedRgb.b,
                post.active ? post.scratch.tmp : nullptr,
                post.active ? post.lensBlur.weights : nullptr,
                post.active ? post.lensBlur.radius : 0,
                post.active ? post.unsharp.weights : nullptr,
                post.active ? post.unsharp.radius : 0,
                post.active ? scannerPostEffects.unsharpAmount : 0.0f,
                gateOutputActive && !grainDebugActive
                    ? &directEffectsPayload
                    : nullptr,
                gateOutputActive && !grainDebugActive
                    ? &directWeavePayload
                    : nullptr,
                gateOutputActive && !grainDebugActive &&
                        directEffectsDescriptor->gateMaskActive
                    ? directEffectsWorkspace.gateMask
                    : nullptr,
                gateOutputActive && !grainDebugActive &&
                        directEffectsDescriptor->gateMaskActive
                    ? directEffectsWorkspace.gateMaskWidth
                    : 0,
                gateOutputActive && !grainDebugActive &&
                        directEffectsDescriptor->gateMaskActive
                    ? directEffectsWorkspace.gateMaskHeight
                    : 0,
                directAliasProfileActive ? &directCompositeProfile : nullptr,
                _pCudaStream);
            if (launchError == cudaSuccess && directAliasProfileActive) {
                record_alias_route_stage(
                    directCompositeProfile,
                    directCompositeProfile.aliasScannerPostOutput,
                    directAliasPostOutputTimer,
                    _pCudaStream);
            }
            // Focused RGB can alias the spatial-DIR filtered planes, so scanner output is their final consumer.
            if (launchError == cudaSuccess && directSpatialDir.hash != 0) {
                std::string releaseError;
                if (!preparedFrame.release_spatial_dir_stage_after_scanner_output(
                        focusedWorkspace,
                        _pCudaStream,
                        releaseError)) {
                    preparedFrame.abort("direct_spatial_dir_stage_release_after_scanner_output_failed");
                    throw_submission_fatal(
                        "direct_spatial_dir_stage_release_after_scanner_output",
                        "direct spatial DIR stage release after scanner output failed",
                        releaseError);
                }
                clear_spatial_dir_final_develop_payload_bindings(run.filmDevelop);
            }
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
        if (directSpatialDir.hash != 0 && !useFocusedSplit) {
            std::string releaseError;
            if (!preparedFrame.release_spatial_dir_cached_log_raw_after_final_develop(
                    focusedWorkspace,
                    _pCudaStream,
                    releaseError)) {
                preparedFrame.abort("direct_spatial_dir_cached_log_raw_release_failed");
                throw_submission_fatal(
                    "direct_spatial_dir_cached_log_raw_release",
                    "direct spatial DIR cached log raw release failed",
                    releaseError);
            }
            run.filmDevelop.spatialDir.logRawB = nullptr;
            run.filmDevelop.spatialDir.logRawG = nullptr;
            run.filmDevelop.spatialDir.logRawR = nullptr;
        }
        std::chrono::steady_clock::time_point directPreparedFrameFinishStart{};
        if (dirProfileEnabled) {
            directPreparedFrameFinishStart = std::chrono::steady_clock::now();
        }
        if (!preparedFrame.finalize_scan_error_stage(run.scanStage.scanErrorFlag, _pCudaStream, scanError)) {
            preparedFrame.abort("direct_scan_error_finalize_failed");
            throw_submission_fatal("direct_scan_error_finalize", "direct scan error finalize failed", scanError);
        }
        record_cuda_use(preparedFrame);
        std::string finishError;
        if (!preparedFrame.finish(_pCudaStream, finishError)) {
            throw_submission_fatal("direct_prepared_frame_finish", "direct prepared frame finish failed", finishError);
        }
        if (dirProfileEnabled) {
            directPreparedFrameFinishMs =
                elapsed_ms_since(directPreparedFrameFinishStart);
            trace_spatial_dir_profile(
                "direct",
                width,
                height,
                directSpatialDir,
                SpatialDirProfileAdmittedRoles{directDirAdmittedRoles, directDirAdmittedTargetRoles},
                directDirProfile,
                directDirProfileCaptured,
                directDirScratchOverflow,
                directSpatialDirDescriptorMs,
                directDirPrepareMs,
                directDirBuildHostMs,
                directPipelineCudaMs,
                directPipelineLaunchHostMs,
                directCompositeProfile,
                directRecipe->profileRoute,
                directRecipe->hash,
                directRecipe->visualGrain,
                directRecipe->filmJuicerEffects,
                directVisualGrainDescriptor
                    ? &*directVisualGrainDescriptor
                    : nullptr,
                directGrainWorkspace.scratchShape,
                directGrainWorkspace.overflow,
                directGrainResources.staticNoise.version,
                directUploadTrace,
                directGrainProfile,
                directGrainDescriptorMs,
                directPreparedFramePrepareMs,
                directPreparedFrameFinishMs,
                directGrainSubmitHostMs,
                useFocusedSplit,
                filmEffectsActive,
                scannerPostEffects.active());
        }
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

        std::optional<Spektrafilm::VisualGrainFrameDescriptor>
            printVisualGrainDescriptor;
        std::string printGrainDescriptorDiagnostic;
        std::chrono::steady_clock::time_point printGrainDescriptorStart{};
        if (dirProfileEnabled) {
            printGrainDescriptorStart = std::chrono::steady_clock::now();
        }
        if (!build_visual_grain_descriptor_for_frame(
                printRecipe->visualGrain,
                printRecipe->filmDevelop,
                printRecipe->profileRoute.capturePolarity,
                {win.x1, win.y1, width, height},
                {printFullFrameRect.x1,
                 printFullFrameRect.y1,
                 printFullFrameRect.x2 - printFullFrameRect.x1,
                 printFullFrameRect.y2 - printFullFrameRect.y1},
                _pixelSizeUm,
                _timeFrames,
                _frameRate,
                _sessionSeed,
                static_cast<std::uint64_t>(_clipToken),
                printVisualGrainDescriptor,
                printGrainDescriptorDiagnostic)) {
            throw_print_restriction(
                printGrainDescriptorDiagnostic.c_str());
        }
        const double printGrainDescriptorMs =
            dirProfileEnabled
                ? elapsed_ms_since(printGrainDescriptorStart)
                : 0.0;
        if (!visual_grain_full_frame_preflight(
                printVisualGrainDescriptor,
                win,
                srcBounds,
                printFullFrameRect)) {
            throw_print_restriction(
                "ResourceDescriptorMismatch phase=grain_preflight field=full_frame_extent");
        }
        std::optional<Spektrafilm::FilmJuicerEffectsFrameDescriptor>
            printEffectsDescriptor;
        std::string printEffectsDescriptorDiagnostic;
        if (!build_film_juicer_effects_descriptor_for_frame(
                printRecipe->filmJuicerEffects,
                {win.x1, win.y1, width, height},
                {printFullFrameRect.x1,
                 printFullFrameRect.y1,
                 printFullFrameRect.x2 - printFullFrameRect.x1,
                 printFullFrameRect.y2 - printFullFrameRect.y1},
                _pixelSizeUm,
                _timeFrames,
                _frameRate,
                _sessionSeed,
                static_cast<std::uint64_t>(_clipToken),
                printEffectsDescriptor,
                printEffectsDescriptorDiagnostic)) {
            throw_print_restriction(
                printEffectsDescriptorDiagnostic.c_str());
        }
        if (!film_juicer_effects_full_frame_preflight(
                printEffectsDescriptor,
                win,
                srcBounds,
                printFullFrameRect)) {
            throw_print_restriction(
                "ResourceDescriptorMismatch phase=effects_preflight field=full_frame_extent");
        }
        const bool grainStageActive =
            printVisualGrainDescriptor.has_value();
        const bool grainDebugActive =
            grainStageActive && printRecipe->visualGrain.debugView != 0;
        const bool filmEffectsActive =
            printEffectsDescriptor.has_value() &&
            printEffectsDescriptor->filmActive;
        const bool gateOutputActive =
            printEffectsDescriptor.has_value() &&
            printEffectsDescriptor->gateOutputActive;
        const bool cameraDiffusionActive =
            _diffusionFrameSetDescriptor.has_value() &&
            _diffusionFrameSetDescriptor->camera.has_value();
        const bool enlargerDiffusionActive =
            _diffusionFrameSetDescriptor.has_value() &&
            _diffusionFrameSetDescriptor->enlarger.has_value();
        const bool routeDiffusionActive =
            cameraDiffusionActive || enlargerDiffusionActive;
        const bool useFocusedSplit =
            routeDiffusionActive || scannerPostEffects.active() || grainStageActive ||
            filmEffectsActive || gateOutputActive;
        const bool captureDensityConsumerActive =
            grainStageActive || filmEffectsActive;

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
        preparation.diffusionFrameSetDescriptor =
            _diffusionFrameSetDescriptor
                ? &*_diffusionFrameSetDescriptor
                : nullptr;
        preparation.visualGrainDescriptor =
            printVisualGrainDescriptor;
        preparation.effectsDescriptor = printEffectsDescriptor;
        preparation.requestedWidth = width;
        preparation.requestedHeight = height;
        preparation.needCompositeProfileWorkspace =
            dirProfileEnabled && !routeDiffusionActive;

        std::string prepareError;
        std::chrono::steady_clock::time_point printPreparedFrameStart{};
        if (dirProfileEnabled) {
            printPreparedFrameStart = std::chrono::steady_clock::now();
        }
        JuicerProcess::Root::PreparedCudaFrame preparedFrame =
            JuicerProcess::root().prepare_cuda_frame(
                deviceContextKey,
                snapshot,
                preparation,
                autoExposureBufferRequest,
                _pCudaStream,
                prepareError);
        const double printPreparedFramePrepareMs =
            dirProfileEnabled
                ? elapsed_ms_since(printPreparedFrameStart)
                : 0.0;
        if (!preparedFrame.active()) {
            throw_submission_fatal(
                preparedFrame.failure_stage_tag(),
                preparedFrame.failure_prefix(),
                prepareError);
        }

        const auto diffusionPrepared = preparedFrame.diffusion_resources();
        DiffusionStageBinding printCameraDiffusion{};
        DiffusionStageBinding printEnlargerDiffusion{};
        if (_diffusionFrameSetDescriptor) {
            if (!diffusionPrepared.active ||
                diffusionPrepared.executionDescriptor.hash == 0 ||
                diffusionPrepared.executionDescriptor.frameSetHash !=
                    _diffusionFrameSetDescriptor->hash ||
                diffusionPrepared.executionDescriptor.contextEpoch !=
                    snapshot.contextEpoch ||
                diffusionPrepared.spectrumCount == 0 ||
                diffusionPrepared.spectrumCount !=
                    diffusionPrepared.executionDescriptor.uniqueSpectrumCount) {
                preparedFrame.abort("print_diffusion_prepared_view_mismatch");
                throw_print_restriction(
                    "MissingRequiredResource component=diffusion field=prepared_view");
            }
            if (_diffusionFrameSetDescriptor->camera) {
                std::string cameraBindingDiagnostic;
                if (!bind_diffusion_stage(
                        *_diffusionFrameSetDescriptor,
                        *_diffusionFrameSetDescriptor->camera,
                        Spektrafilm::DiffusionLinearStage::CameraFilmLinear,
                        diffusionPrepared,
                        printCameraDiffusion,
                        cameraBindingDiagnostic)) {
                    preparedFrame.abort(
                        "print_diffusion_camera_binding_failed");
                    throw_print_restriction(
                        cameraBindingDiagnostic.c_str());
                }
            }
            if (_diffusionFrameSetDescriptor->enlarger) {
                std::string enlargerBindingDiagnostic;
                if (!bind_diffusion_stage(
                        *_diffusionFrameSetDescriptor,
                        *_diffusionFrameSetDescriptor->enlarger,
                        Spektrafilm::DiffusionLinearStage::EnlargerPrintLinear,
                        diffusionPrepared,
                        printEnlargerDiffusion,
                        enlargerBindingDiagnostic)) {
                    preparedFrame.abort(
                        "print_diffusion_enlarger_binding_failed");
                    throw_print_restriction(
                        enlargerBindingDiagnostic.c_str());
                }
            }
        } else if (diffusionPrepared.active) {
            preparedFrame.abort("print_disabled_diffusion_view_active");
            throw_print_restriction(
                "ResourceDescriptorMismatch component=diffusion field=disabled_prepared_view");
        }

        const JuicerProcess::Root::PreparedCudaFrame::PrintRoutePreparedView prepared =
            preparedFrame.print_route_resources();
        const JuicerProcess::Root::PreparedCudaFrame::PrintPreparedView preparedPrint =
            preparedFrame.print_resources();
        JuicerProcess::Root::PreparedCudaFrame::UploadTraceView printUploadTrace{};
        if (dirProfileEnabled) {
            printUploadTrace = preparedFrame.upload_trace_view();
        }
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
            JuicerCuda::AutoExposureSourceFormat autoExposureSourceFormat{};
            autoExposureSourceFormat.componentCount = run.nComponents;
            autoExposureSourceFormat.inputColorSpaceIndex = printRecipe->filmRaw.inputColorSpace;
            autoExposureSourceFormat.applyCctfDecoding = bool_to_i32(printRecipe->filmRaw.inputCctfDecoding);
            const int meterRc = juicer_cuda_auto_exposure_meter_to_device(
                srcBase,
                static_cast<std::size_t>(srcRowBytes),
                autoExposureDescriptor,
                autoExposureSourceFormat,
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
                printRecipe->enlargerFilmBounds,
                prepared.film,
                autoExposureScaleDevice,
                1.0f,
                filmPayloads,
                payloadDiagnostic)) {
            preparedFrame.abort("print_film_payload_pack_failed");
            throw_print_restriction(payloadDiagnostic.c_str());
        }
        JuicerCuda::PrintCudaPayloadPack printPayloads{};
        if (!JuicerCuda::pack_print_cuda_payloads(
                printRecipe->print,
                preparedPrint,
                scannerCorrection.exposureScale,
                printPayloads,
                payloadDiagnostic)) {
            preparedFrame.abort("print_payload_pack_failed");
            throw_print_restriction(payloadDiagnostic.c_str());
        }
        run.filmRaw = filmPayloads.filmRaw;
        run.filmExpose = filmPayloads.filmExposure;
        run.filmDevelop = filmPayloads.filmDevelop;
        run.printExpose = printPayloads.expose;
        run.printDevelop = printPayloads.develop;

        JuicerCuda::CameraFilmLinearExposurePlanes printCameraFilmLinear{};
        if (cameraDiffusionActive) {
            if (!printCameraDiffusion.active) {
                preparedFrame.abort("print_diffusion_camera_binding_inactive");
                throw_print_restriction(
                    "MissingRequiredResource component=diffusion field=camera_stage_binding");
            }
            preparedFrame.mark_diffusion_work_enqueued();
            printCameraFilmLinear = camera_film_linear_planes(
                printCameraDiffusion.stagePlanes);
            const cudaError_t exposureError =
                juicer_cuda_print_camera_film_linear_exposure(
                    &run,
                    printCameraFilmLinear,
                    _pCudaStream);
            if (exposureError != cudaSuccess) {
                preparedFrame.abort(
                    "print_camera_film_linear_exposure_launch_failed");
                throw_cuda_stage_fatal(
                    "print_camera_film_linear_exposure_launch",
                    "print camera film-linear exposure launch failed",
                    exposureError);
            }
            JuicerCuda::Diffusion::StageLaunchRequest diffusionLaunch{};
            diffusionLaunch.layout = printCameraDiffusion.layout;
            diffusionLaunch.geometry = printCameraDiffusion.geometry;
            diffusionLaunch.fullFrame = printCameraDiffusion.fullFrame;
            diffusionLaunch.spectra = printCameraDiffusion.spectra;
            diffusionLaunch.execution = printCameraDiffusion.execution;
            diffusionLaunch.planes = &printCameraDiffusion.stagePlanes;
            diffusionLaunch.stream = reinterpret_cast<cudaStream_t>(
                _pCudaStream);
            const JuicerCuda::Diffusion::LaunchResult diffusionResult =
                JuicerCuda::Diffusion::launch_stage(diffusionLaunch);
            if (!diffusionResult.ok()) {
                preparedFrame.abort("print_camera_diffusion_launch_failed");
                if (diffusionResult.api ==
                    JuicerCuda::Diffusion::FailureApi::Cuda) {
                    throw_cuda_stage_fatal(
                        diffusionResult.stage,
                        "print camera diffusion launch failed",
                        static_cast<cudaError_t>(diffusionResult.code));
                }
                const std::string diagnostic =
                    diffusion_launch_failure_diagnostic(diffusionResult);
                throw_print_restriction(diagnostic.c_str());
            }
            // launch_stage rotates semantic plane pointers after each channel.
            printCameraFilmLinear = camera_film_linear_planes(
                printCameraDiffusion.stagePlanes);
        }

        const bool printCompositeProfileRequested =
            dirProfileEnabled && !routeDiffusionActive;
        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceRequest workspaceRequest =
            preparedFrame.workspace_request();
        const bool printUseFusedScannerPostSpatialDirHandoff =
            !routeDiffusionActive && spatialDir.hash != 0 &&
            scannerPostEffects.active() &&
            spatialDir.approximation ==
                Spektrafilm::DirApproximationMarker::SpektrafilmStrict &&
            !captureDensityConsumerActive;
        JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker focusedWorkspace{};
        if (workspaceRequest.has_any_family()) {
            focusedWorkspace = preparedFrame.bind_workspace_request(workspaceRequest);
            std::string transitionError;
            if (!preparedFrame.checkpoint_large_scratch_transition(
                    focusedWorkspace,
                    _pCudaStream,
                    "print_large_scratch_transition",
                    transitionError)) {
                preparedFrame.abort("print_large_scratch_transition_failed");
                throw_submission_fatal(
                    "print_large_scratch_transition",
                    "print large scratch transition checkpoint failed",
                    transitionError);
            }
        }
        JuicerCuda::SpatialDirBuildProfile printDirProfile{};
        bool printDirProfileCaptured = false;
        bool printDirScratchOverflow = false;
        Spektrafilm::DirScratchPlaneRoles printDirAdmittedRoles = spatialDir.planeRoles;
        Spektrafilm::DirScratchPlaneRoles printDirAdmittedTargetRoles = spatialDir.targetPlaneRoles;
        bool printDirUsesSourceBuildCachedLogRaw = false;
        double printDirPrepareMs = 0.0;
        double printDirBuildHostMs = 0.0;
        double printPreparedFrameFinishMs = 0.0;
        double printPipelineLaunchHostMs = 0.0;
        float printPipelineCudaMs = -1.0f;
        double printGrainSubmitHostMs = 0.0;
        JuicerCuda::VisualGrainRuntimeProfile printGrainProfile{};
        JuicerCuda::CompositePipelineProfile printCompositeProfile{};
        bool printCaptureDensityReady = false;
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
        }
        JuicerProcess::Root::PreparedCudaFrame::CaptureFilmDensityWorkspaceView
            printCaptureDensity{};
        JuicerProcess::Root::PreparedCudaFrame::FocusedRgbWorkspaceView
            printFocusedRgb{};
        JuicerProcess::Root::PreparedCudaFrame::VisualGrainWorkspaceView
            printGrainWorkspace{};
        JuicerProcess::Root::PreparedCudaFrame::PreparedVisualGrainView
            printGrainResources{};
        JuicerProcess::Root::PreparedCudaFrame::ScannerWorkspaceView
            printEffectsWorkspace{};
        JuicerCuda::GrainPayload printGrainPayload{};
        JuicerCuda::GrainKernelPayload printGrainKernels{};
        JuicerCuda::GrainPayload printEffectsPayload{};
        JuicerCuda::GateWeavePayload printWeavePayload{};
        const Spektrafilm::FilmJuicerEffectsFrameDescriptor*
            preparedPrintEffectsDescriptor =
                preparedFrame.film_juicer_effects_descriptor();
        if (printEffectsDescriptor.has_value()) {
            std::string effectsPayloadDiagnostic;
            if (!preparedPrintEffectsDescriptor ||
                !JuicerCuda::pack_film_juicer_effects_payload(
                    *preparedPrintEffectsDescriptor,
                    printEffectsPayload,
                    printWeavePayload,
                    effectsPayloadDiagnostic)) {
                preparedFrame.abort("print_effects_payload_pack_failed");
                throw_print_restriction(
                    effectsPayloadDiagnostic.empty()
                        ? "MissingRequiredResource phase=effects_route field=effects_binding"
                        : effectsPayloadDiagnostic.c_str());
            }
        } else if (preparedPrintEffectsDescriptor) {
            preparedFrame.abort("print_inactive_effects_descriptor_present");
            throw_print_restriction(
                "ResourceDescriptorMismatch phase=effects_route field=inactive_descriptor");
        }
        if (useFocusedSplit) {
            std::string focusedWorkspaceError;
            if (!preparedFrame.stage_optical_workspace(
                    focusedWorkspace,
                    _pCudaStream,
                    focusedWorkspaceError)) {
                preparedFrame.abort("print_focused_workspace_stage_failed");
                throw_submission_fatal(
                    "print_focused_workspace_stage",
                    "print focused workspace staging failed",
                    focusedWorkspaceError);
            }
            printCaptureDensity =
                preparedFrame.capture_film_density_workspace(
                    focusedWorkspace);
            printFocusedRgb = preparedFrame.focused_rgb_workspace(
                focusedWorkspace);
            if (!printCaptureDensity.active || !printFocusedRgb.active ||
                printCaptureDensity.c != printFocusedRgb.r ||
                printCaptureDensity.m != printFocusedRgb.g ||
                printCaptureDensity.y != printFocusedRgb.b) {
                preparedFrame.abort("print_focused_triplet_binding_failed");
                throw_print_restriction(
                    "MissingRequiredResource phase=grain_route field=focused_triplet");
            }
            printEffectsWorkspace =
                preparedFrame.scanner_workspace(focusedWorkspace);
            if (!printEffectsWorkspace.active ||
                (printEffectsDescriptor.has_value() &&
                 printEffectsDescriptor->gateMaskActive &&
                 !printEffectsWorkspace.hasGateMask)) {
                preparedFrame.abort("print_effects_workspace_binding_failed");
                throw_print_restriction(
                    "MissingRequiredResource phase=effects_route field=effects_workspace");
            }
        }
        if (grainStageActive) {
            printGrainResources = preparedFrame.visual_grain_resources();
            printGrainWorkspace = preparedFrame.visual_grain_workspace(
                focusedWorkspace);
            std::string grainPayloadDiagnostic;
            if (!printGrainResources.active ||
                !printGrainWorkspace.active ||
                !JuicerCuda::pack_visual_grain_payload(
                    printRecipe->visualGrain,
                    printGrainResources,
                    printGrainPayload,
                    printGrainKernels,
                    grainPayloadDiagnostic)) {
                preparedFrame.abort("print_visual_grain_payload_pack_failed");
                throw_print_restriction(
                    grainPayloadDiagnostic.empty()
                        ? "MissingRequiredResource phase=grain_route field=grain_binding"
                        : grainPayloadDiagnostic.c_str());
            }
            printGrainPayload.frameUniforms =
                printGrainWorkspace.frameUniforms;
        }
        if (spatialDir.hash != 0) {
            const auto scratch = preparedFrame.spatial_dir_scratch(focusedWorkspace);
            const auto resources =
                preparedFrame.spatial_dir_resources(focusedWorkspace, spatialDir.hash);
            if (!scratch.active || !resources.active) {
                preparedFrame.abort("print_spatial_dir_binding_failed");
                throw_print_restriction(
                    "MissingRequiredResource phase=4C field=prepared_spatial_dir");
            }
            printDirScratchOverflow = scratch.overflow;
            printDirAdmittedRoles = scratch.planeRoles;
            printDirAdmittedTargetRoles = scratch.targetPlaneRoles;
            printDirUsesSourceBuildCachedLogRaw =
                fused_alias_uses_source_build_cached_log_raw(
                    printUseFusedScannerPostSpatialDirHandoff,
                    printDirAdmittedRoles,
                    printDirAdmittedTargetRoles);
            if (printDirUsesSourceBuildCachedLogRaw &&
                !spatial_dir_required_cached_log_raw_present(
                    scratch,
                    printDirAdmittedRoles.cachedLogRawPlanes)) {
                preparedFrame.abort("print_spatial_dir_source_build_cached_log_raw_missing");
                throw_print_restriction(
                    "MissingRequiredResource phase=4C field=spatial_dir_source_build_cached_log_raw");
            }
            float* cachedLogRawB =
                printDirUsesSourceBuildCachedLogRaw ? scratch.logRawB : nullptr;
            float* cachedLogRawG =
                printDirUsesSourceBuildCachedLogRaw ? scratch.logRawG : nullptr;
            float* cachedLogRawR =
                printDirUsesSourceBuildCachedLogRaw ? scratch.logRawR : nullptr;
            JuicerCuda::SpatialDirBuildRequest dirBuildRequest{};
            dirBuildRequest.planes.rawCorrectionY = scratch.rawCorrectionY;
            dirBuildRequest.planes.rawCorrectionM = scratch.rawCorrectionM;
            dirBuildRequest.planes.rawCorrectionC = scratch.rawCorrectionC;
            dirBuildRequest.planes.filteredCorrectionY = scratch.filteredCorrectionY;
            dirBuildRequest.planes.filteredCorrectionM = scratch.filteredCorrectionM;
            dirBuildRequest.planes.filteredCorrectionC = scratch.filteredCorrectionC;
            dirBuildRequest.planes.filterTemp = scratch.filterTemp;
            dirBuildRequest.planes.filterTempM = scratch.filterTempM;
            dirBuildRequest.planes.filterTempC = scratch.filterTempC;
            dirBuildRequest.planes.logRawB = cachedLogRawB;
            dirBuildRequest.planes.logRawG = cachedLogRawG;
            dirBuildRequest.planes.logRawR = cachedLogRawR;
            dirBuildRequest.cameraFilmLinear = printCameraFilmLinear;
            dirBuildRequest.gaussian.kernel = resources.gaussian.weights;
            dirBuildRequest.gaussian.radius = resources.gaussian.radius;
            dirBuildRequest.gaussian.sigma = resources.gaussian.sigma;
            dirBuildRequest.gaussian.weight = spatialDir.gaussianWeight;
            for (int tailIndex = 0; tailIndex < 3; ++tailIndex) {
                dirBuildRequest.tails[tailIndex].kernel = resources.exponential[tailIndex].weights;
                dirBuildRequest.tails[tailIndex].radius = resources.exponential[tailIndex].radius;
                dirBuildRequest.tails[tailIndex].sigma = resources.exponential[tailIndex].sigma;
                dirBuildRequest.tails[tailIndex].weight = spatialDir.exponentialWeights[tailIndex];
            }
            dirBuildRequest.streamOpaque = _pCudaStream;
            dirBuildRequest.profile = dirProfileEnabled ? &printDirProfile : nullptr;
            const auto printDirBuildStart = std::chrono::steady_clock::now();
            const cudaError_t dirError = juicer_cuda_build_print_spatial_dir(
                &run,
                dirBuildRequest);
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
            std::string releaseError;
            if (!preparedFrame.release_spatial_dir_build_scratch_after_build(
                    focusedWorkspace,
                    _pCudaStream,
                    releaseError)) {
                preparedFrame.abort("print_spatial_dir_build_scratch_release_failed");
                throw_submission_fatal(
                    "print_spatial_dir_build_scratch_release",
                    "print spatial DIR build scratch release failed",
                    releaseError);
            }
            if (!printUseFusedScannerPostSpatialDirHandoff) {
                if (!preparedFrame.stage_spatial_dir_cached_log_raw_for_final_develop(
                        focusedWorkspace,
                        _pCudaStream,
                        releaseError)) {
                    preparedFrame.abort("print_spatial_dir_cached_log_raw_stage_failed");
                    throw_submission_fatal(
                        "print_spatial_dir_cached_log_raw_stage",
                        "print spatial DIR cached log raw staging failed",
                        releaseError);
                }
            }
            const auto finalScratch = preparedFrame.spatial_dir_scratch(focusedWorkspace);
            const bool printDirRequiresCachedLogRaw =
                printDirUsesSourceBuildCachedLogRaw ||
                (!printUseFusedScannerPostSpatialDirHandoff &&
                 printDirAdmittedTargetRoles.cachedLogRawPlanes == 3);
            if (!finalScratch.filteredCorrectionY || !finalScratch.filteredCorrectionM ||
                !finalScratch.filteredCorrectionC) {
                preparedFrame.abort("print_spatial_dir_filtered_correction_missing");
                throw_print_restriction(
                    "MissingRequiredResource phase=4C field=spatial_dir_filtered_correction");
            }
            if (printDirRequiresCachedLogRaw &&
                !spatial_dir_required_cached_log_raw_present(
                    finalScratch,
                    printDirAdmittedTargetRoles.cachedLogRawPlanes)) {
                preparedFrame.abort("print_spatial_dir_cached_log_raw_missing");
                throw_print_restriction(
                    "MissingRequiredResource phase=4C field=spatial_dir_cached_log_raw");
            }
            if (printDirRequiresCachedLogRaw && !printDirUsesSourceBuildCachedLogRaw) {
                const cudaError_t logRawError = cameraDiffusionActive
                                                    ? juicer_cuda_build_print_spatial_dir_cached_log_raw_from_camera_film_linear(
                                                          &run,
                                                          printCameraFilmLinear,
                                                          finalScratch.logRawB,
                                                          finalScratch.logRawG,
                                                          finalScratch.logRawR,
                                                          _pCudaStream)
                                                    : juicer_cuda_build_print_spatial_dir_cached_log_raw(
                                                          &run,
                                                          finalScratch.logRawB,
                                                          finalScratch.logRawG,
                                                          finalScratch.logRawR,
                                                          _pCudaStream);
                if (logRawError != cudaSuccess) {
                    preparedFrame.abort("print_spatial_dir_cached_log_raw_launch_failed");
                    throw_cuda_stage_fatal(
                        "print_spatial_dir_cached_log_raw_launch",
                        "print spatial DIR cached log raw build failed",
                        logRawError);
                }
            }
            bind_spatial_dir_final_develop_to_payload(run.filmDevelop, finalScratch);
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
        run.scanStage.scanLutLog2PchipXYZ = prepared.scanLut->log2PchipXYZ;
        run.scanStage.scanLutPchipSlopeC = prepared.scanLut->slopeC;
        run.scanStage.scanLutPchipSlopeM = prepared.scanLut->slopeM;
        run.scanStage.scanLutPchipSlopeY = prepared.scanLut->slopeY;
        run.scanStage.scanLutPchipCellMin = prepared.scanLut->cellMin;
        run.scanStage.scanLutPchipCellMax = prepared.scanLut->cellMax;
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
        if (printCompositeProfileRequested &&
            printUseFusedScannerPostSpatialDirHandoff &&
            workspaceRequest.aliasScannerRgbFromSpatialDirFiltered) {
            initialize_spatial_dir_rgb_alias_profile(
                printCompositeProfile,
                AliasRouteProfileExtent{width, height});
        } else if (printCompositeProfileRequested &&
                   !workspaceRequest.aliasScannerRgbFromSpatialDirFiltered) {
            std::string profileWorkspaceError;
            if (preparedFrame.try_stage_profile_optical_workspace(
                    focusedWorkspace,
                    _pCudaStream,
                    profileWorkspaceError)) {
                const auto profileScratch = preparedFrame.scanner_workspace(focusedWorkspace);
                if (profileScratch.active) {
                    const cudaError_t profileError = juicer_cuda_profile_print_focused_pipeline_stages(
                        &run,
                        profileScratch.rgbR,
                        profileScratch.rgbG,
                        profileScratch.rgbB,
                        &printCompositeProfile,
                        _pCudaStream);
                    if (profileError != cudaSuccess) {
                        printCompositeProfile = JuicerCuda::CompositePipelineProfile{};
                        printCompositeProfile.profileKind = "focused_split_attribution";
                        printCompositeProfile.profileNote = "profile_failed";
                    }
                } else {
                    printCompositeProfile.profileKind = "focused_split_attribution";
                    printCompositeProfile.profileNote = "missing_profile_scratch";
                }
            } else {
                printCompositeProfile.profileKind = "focused_split_attribution";
                printCompositeProfile.profileNote = "profile_workspace_failed";
            }
        } else if (printCompositeProfileRequested) {
            printCompositeProfile.profileKind = "focused_split_attribution";
            printCompositeProfile.profileNote =
                "scratch_replay_skipped_live_spatial_dir_alias";
        }
        cudaError_t launchError = cudaSuccess;
        const auto printPipelineLaunchStart = std::chrono::steady_clock::now();
        CudaEventElapsedTimer printPipelineCudaTimer;
        if (dirProfileEnabled) {
            printPipelineCudaTimer.begin(_pCudaStream);
        }
        if (useFocusedSplit &&
            !printUseFusedScannerPostSpatialDirHandoff) {
            launchError = cameraDiffusionActive
                              ? juicer_cuda_print_focused_capture_density_from_camera_film_linear(
                                    &run,
                                    printCameraFilmLinear,
                                    printCaptureDensity.c,
                                    printCaptureDensity.m,
                                    printCaptureDensity.y,
                                    _pCudaStream)
                              : juicer_cuda_print_focused_capture_density(
                                    &run,
                                    printCaptureDensity.c,
                                    printCaptureDensity.m,
                                    printCaptureDensity.y,
                                    _pCudaStream);
            if (launchError != cudaSuccess) {
                preparedFrame.abort("print_capture_density_launch_failed");
                throw_cuda_stage_fatal(
                    "print_capture_density_launch",
                    "print capture-film density launch failed",
                    launchError);
            }
            printCaptureDensityReady = true;
            if (spatialDir.hash != 0) {
                std::string releaseError;
                if (!preparedFrame.release_spatial_dir_cached_log_raw_after_final_develop(
                        focusedWorkspace,
                        _pCudaStream,
                        releaseError)) {
                    preparedFrame.abort("print_spatial_dir_cached_log_raw_release_failed");
                    throw_submission_fatal(
                        "print_spatial_dir_cached_log_raw_release",
                        "print spatial DIR cached log raw release failed",
                        releaseError);
                }
                run.filmDevelop.spatialDir.logRawB = nullptr;
                run.filmDevelop.spatialDir.logRawG = nullptr;
                run.filmDevelop.spatialDir.logRawR = nullptr;
            }
            if (grainStageActive) {
                CudaEventElapsedTimer printGrainCudaTimer;
                std::chrono::steady_clock::time_point printGrainSubmitStart{};
                if (dirProfileEnabled) {
                    printGrainCudaTimer.begin(_pCudaStream);
                    printGrainSubmitStart = std::chrono::steady_clock::now();
                }
                launchError = juicer_cuda_apply_visual_grain(
                    &printGrainPayload,
                    &printGrainKernels,
                    width,
                    height,
                    printCaptureDensity.c,
                    printCaptureDensity.m,
                    printCaptureDensity.y,
                    printGrainWorkspace.filterTemp,
                    printGrainWorkspace.scaleWork,
                    printGrainWorkspace.deltaAccum,
                    printGrainWorkspace.layerWork,
                    printGrainWorkspace.sharedDelta,
                    dirProfileEnabled ? &printGrainProfile : nullptr,
                    _pCudaStream);
                if (dirProfileEnabled) {
                    printGrainSubmitHostMs =
                        elapsed_ms_since(printGrainSubmitStart);
                    printGrainProfile.total.elapsedMs =
                        printGrainCudaTimer.finish(_pCudaStream);
                    printGrainProfile.total.launches =
                        printGrainProfile.totalLaunches;
                }
                if (launchError != cudaSuccess) {
                    preparedFrame.abort("print_visual_grain_launch_failed");
                    throw_cuda_stage_fatal(
                        "print_visual_grain_launch",
                        "print visual grain launch failed",
                        launchError);
                }
            }
            if (filmEffectsActive && !grainDebugActive) {
                launchError = juicer_cuda_apply_film_defects(
                    &printEffectsPayload,
                    width,
                    height,
                    printCaptureDensity.c,
                    printCaptureDensity.m,
                    printCaptureDensity.y,
                    _pCudaStream);
                if (launchError != cudaSuccess) {
                    preparedFrame.abort("print_film_effects_launch_failed");
                    throw_cuda_stage_fatal(
                        "print_film_effects_launch",
                        "print film defects launch failed",
                        launchError);
                }
            }
            if (!grainDebugActive) {
                if (enlargerDiffusionActive) {
                    if (!printEnlargerDiffusion.active) {
                        preparedFrame.abort(
                            "print_diffusion_enlarger_binding_inactive");
                        throw_print_restriction(
                            "MissingRequiredResource component=diffusion field=enlarger_stage_binding");
                    }
                    preparedFrame.mark_diffusion_work_enqueued();
                    JuicerCuda::EnlargerPrintLinearExposurePlanes
                        printEnlargerLinear = enlarger_print_linear_planes(
                            printEnlargerDiffusion.stagePlanes);
                    launchError =
                        juicer_cuda_print_focused_enlarger_linear_exposure(
                            &run,
                            printCaptureDensity.c,
                            printCaptureDensity.m,
                            printCaptureDensity.y,
                            printEnlargerLinear,
                            _pCudaStream);
                    if (launchError != cudaSuccess) {
                        preparedFrame.abort(
                            "print_enlarger_linear_exposure_launch_failed");
                        throw_cuda_stage_fatal(
                            "print_enlarger_linear_exposure_launch",
                            "print enlarger-linear exposure launch failed",
                            launchError);
                    }

                    JuicerCuda::Diffusion::StageLaunchRequest
                        diffusionLaunch{};
                    diffusionLaunch.layout = printEnlargerDiffusion.layout;
                    diffusionLaunch.geometry =
                        printEnlargerDiffusion.geometry;
                    diffusionLaunch.fullFrame =
                        printEnlargerDiffusion.fullFrame;
                    diffusionLaunch.spectra = printEnlargerDiffusion.spectra;
                    diffusionLaunch.execution =
                        printEnlargerDiffusion.execution;
                    diffusionLaunch.planes =
                        &printEnlargerDiffusion.stagePlanes;
                    diffusionLaunch.stream =
                        reinterpret_cast<cudaStream_t>(_pCudaStream);
                    const JuicerCuda::Diffusion::LaunchResult
                        diffusionResult =
                            JuicerCuda::Diffusion::launch_stage(
                                diffusionLaunch);
                    if (!diffusionResult.ok()) {
                        preparedFrame.abort(
                            "print_enlarger_diffusion_launch_failed");
                        if (diffusionResult.api ==
                            JuicerCuda::Diffusion::FailureApi::Cuda) {
                            throw_cuda_stage_fatal(
                                diffusionResult.stage,
                                "print enlarger diffusion launch failed",
                                static_cast<cudaError_t>(
                                    diffusionResult.code));
                        }
                        const std::string diagnostic =
                            diffusion_launch_failure_diagnostic(
                                diffusionResult);
                        throw_print_restriction(diagnostic.c_str());
                    }
                    // launch_stage rotates semantic plane pointers after each channel.
                    printEnlargerLinear = enlarger_print_linear_planes(
                        printEnlargerDiffusion.stagePlanes);
                    launchError =
                        juicer_cuda_print_focused_develop_from_enlarger_linear(
                            &run,
                            printEnlargerLinear,
                            printCaptureDensity.c,
                            printCaptureDensity.m,
                            printCaptureDensity.y,
                            _pCudaStream);
                    if (launchError != cudaSuccess) {
                        preparedFrame.abort(
                            "print_develop_from_enlarger_linear_launch_failed");
                        throw_cuda_stage_fatal(
                            "print_develop_from_enlarger_linear_launch",
                            "print development from enlarger-linear exposure failed",
                            launchError);
                    }
                } else {
                    launchError =
                        juicer_cuda_print_focused_continue_from_capture_density(
                            &run,
                            printCaptureDensity.c,
                            printCaptureDensity.m,
                            printCaptureDensity.y,
                            _pCudaStream);
                    if (launchError != cudaSuccess) {
                        preparedFrame.abort(
                            "print_continuation_from_capture_density_launch_failed");
                        throw_cuda_stage_fatal(
                            "print_continuation_from_capture_density_launch",
                            "print continuation from capture density launch failed",
                            launchError);
                    }
                }
            }
        }

        if (routeDiffusionActive) {
            std::string diffusionReleaseError;
            if (!preparedFrame.release_diffusion_resources_after_use(
                    _pCudaStream,
                    diffusionReleaseError)) {
                preparedFrame.abort("print_diffusion_release_after_use_failed");
                throw_submission_fatal(
                    "print_diffusion_release_after_use",
                    "print diffusion resource release failed",
                    diffusionReleaseError);
            }
            printCameraDiffusion = {};
            printEnlargerDiffusion = {};
            printCameraFilmLinear = {};
        }

        if (gateOutputActive && !grainDebugActive &&
            printEffectsDescriptor->gateMaskActive) {
            launchError = juicer_cuda_build_gate_defect_mask_focused(
                &printEffectsPayload,
                printEffectsWorkspace.gateMask,
                printEffectsWorkspace.gateMaskWidth,
                printEffectsWorkspace.gateMaskHeight,
                _pCudaStream);
            if (launchError != cudaSuccess) {
                preparedFrame.abort("print_gate_mask_launch_failed");
                throw_cuda_stage_fatal(
                    "print_gate_mask_launch",
                    "print gate defect mask launch failed",
                    launchError);
            }
            preparedFrame.mark_gate_mask_built(
                printEffectsDescriptor->hash);
        }

        if (useFocusedSplit) {
            JuicerProcess::Root::PreparedCudaFrame::ScannerPostEffectsPreparedView
                post{};
            if (scannerPostEffects.active() && !grainDebugActive) {
                post = preparedFrame.scanner_post_effects_resources(
                    focusedWorkspace,
                    scannerPostEffects.hash);
                if (!post.active ||
                    post.scratch.rgbR != printFocusedRgb.r ||
                    post.scratch.rgbG != printFocusedRgb.g ||
                    post.scratch.rgbB != printFocusedRgb.b) {
                    preparedFrame.abort("print_scanner_post_effects_view_failed");
                    throw_print_restriction(
                        "MissingRequiredResource phase=8C field=scanner_post_effects");
                }
            }
            const std::uint64_t glareSeed = Hash::hash_uint64_values(
                {printRecipe->hash,
                 snapshot.frameToken.value,
                 static_cast<std::uint64_t>(srcBounds.x1),
                 static_cast<std::uint64_t>(srcBounds.y1)});
            const bool printAliasProfileActive =
                dirProfileEnabled &&
                printUseFusedScannerPostSpatialDirHandoff &&
                workspaceRequest.aliasScannerRgbFromSpatialDirFiltered &&
                !grainDebugActive;
            if (!grainDebugActive) {
                if (printUseFusedScannerPostSpatialDirHandoff) {
                    CudaEventElapsedTimer printAliasScanLinearTimer;
                    if (printAliasProfileActive) {
                        printAliasScanLinearTimer.begin(_pCudaStream);
                    }
                    launchError = juicer_cuda_print_focused_scan_linear_rgb(
                        &run,
                        printFocusedRgb.r,
                        printFocusedRgb.g,
                        printFocusedRgb.b,
                        post.scratch.tmp,
                        post.scratch.blurred,
                        srcBounds.x1,
                        srcBounds.y1,
                        glareSeed,
                        scannerPostEffects.glarePercent,
                        scannerPostEffects.glareRoughness,
                        post.glare.weights,
                        post.glare.radius,
                        printAliasProfileActive ? &printCompositeProfile : nullptr,
                        _pCudaStream);
                    if (launchError == cudaSuccess && printAliasProfileActive) {
                        record_alias_route_stage(
                            printCompositeProfile,
                            printCompositeProfile.aliasFusedScanLinear,
                            printAliasScanLinearTimer,
                            _pCudaStream);
                    }
                } else {
                    if (!printCaptureDensityReady) {
                        preparedFrame.abort("print_capture_density_not_ready");
                        throw_print_restriction(
                            "MissingRequiredResource phase=grain_route field=capture_density");
                    }
                    launchError = juicer_cuda_print_focused_scan_linear_density_rgb(
                        &run,
                        printCaptureDensity.c,
                        printCaptureDensity.m,
                        printCaptureDensity.y,
                        printFocusedRgb.r,
                        printFocusedRgb.g,
                        printFocusedRgb.b,
                        post.active ? post.scratch.tmp : nullptr,
                        post.active ? post.scratch.blurred : nullptr,
                        srcBounds.x1,
                        srcBounds.y1,
                        glareSeed,
                        post.active ? scannerPostEffects.glarePercent : 0.0f,
                        post.active ? scannerPostEffects.glareRoughness : 0.0f,
                        post.active ? post.glare.weights : nullptr,
                        post.active ? post.glare.radius : 0,
                        _pCudaStream);
                }
                if (launchError != cudaSuccess) {
                    preparedFrame.abort("print_scanner_linear_launch_failed");
                    throw_cuda_stage_fatal(
                        "print_scanner_linear_launch",
                        "print scanner-linear launch failed",
                        launchError);
                }
            }
            if (printDirUsesSourceBuildCachedLogRaw) {
                std::string releaseError;
                if (!preparedFrame.release_spatial_dir_cached_log_raw_after_final_develop(
                        focusedWorkspace,
                        _pCudaStream,
                        releaseError)) {
                    preparedFrame.abort("print_spatial_dir_cached_log_raw_release_failed");
                    throw_submission_fatal(
                        "print_spatial_dir_cached_log_raw_release",
                        "print spatial DIR cached log raw release failed",
                        releaseError);
                }
                run.filmDevelop.spatialDir.logRawB = nullptr;
                run.filmDevelop.spatialDir.logRawG = nullptr;
                run.filmDevelop.spatialDir.logRawR = nullptr;
            }
            CudaEventElapsedTimer printAliasPostOutputTimer;
            if (printAliasProfileActive) {
                printAliasPostOutputTimer.begin(_pCudaStream);
            }
            launchError = juicer_cuda_print_focused_scanner_post_output(
                &run,
                printFocusedRgb.r,
                printFocusedRgb.g,
                printFocusedRgb.b,
                post.active ? post.scratch.tmp : nullptr,
                post.active ? post.lensBlur.weights : nullptr,
                post.active ? post.lensBlur.radius : 0,
                post.active ? post.unsharp.weights : nullptr,
                post.active ? post.unsharp.radius : 0,
                post.active ? scannerPostEffects.unsharpAmount : 0.0f,
                gateOutputActive && !grainDebugActive
                    ? &printEffectsPayload
                    : nullptr,
                gateOutputActive && !grainDebugActive
                    ? &printWeavePayload
                    : nullptr,
                gateOutputActive && !grainDebugActive &&
                        printEffectsDescriptor->gateMaskActive
                    ? printEffectsWorkspace.gateMask
                    : nullptr,
                gateOutputActive && !grainDebugActive &&
                        printEffectsDescriptor->gateMaskActive
                    ? printEffectsWorkspace.gateMaskWidth
                    : 0,
                gateOutputActive && !grainDebugActive &&
                        printEffectsDescriptor->gateMaskActive
                    ? printEffectsWorkspace.gateMaskHeight
                    : 0,
                printAliasProfileActive ? &printCompositeProfile : nullptr,
                _pCudaStream);
            if (launchError == cudaSuccess && printAliasProfileActive) {
                record_alias_route_stage(
                    printCompositeProfile,
                    printCompositeProfile.aliasScannerPostOutput,
                    printAliasPostOutputTimer,
                    _pCudaStream);
            }
            // Focused RGB can alias the spatial-DIR filtered planes, so scanner output is their final consumer.
            if (launchError == cudaSuccess && spatialDir.hash != 0) {
                std::string releaseError;
                if (!preparedFrame.release_spatial_dir_stage_after_scanner_output(
                        focusedWorkspace,
                        _pCudaStream,
                        releaseError)) {
                    preparedFrame.abort("print_spatial_dir_stage_release_after_scanner_output_failed");
                    throw_submission_fatal(
                        "print_spatial_dir_stage_release_after_scanner_output",
                        "print spatial DIR stage release after scanner output failed",
                        releaseError);
                }
                clear_spatial_dir_final_develop_payload_bindings(run.filmDevelop);
            }
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
        if (spatialDir.hash != 0 && !useFocusedSplit) {
            std::string releaseError;
            if (!preparedFrame.release_spatial_dir_cached_log_raw_after_final_develop(
                    focusedWorkspace,
                    _pCudaStream,
                    releaseError)) {
                preparedFrame.abort("print_spatial_dir_cached_log_raw_release_failed");
                throw_submission_fatal(
                    "print_spatial_dir_cached_log_raw_release",
                    "print spatial DIR cached log raw release failed",
                    releaseError);
            }
            run.filmDevelop.spatialDir.logRawB = nullptr;
            run.filmDevelop.spatialDir.logRawG = nullptr;
            run.filmDevelop.spatialDir.logRawR = nullptr;
        }
        std::chrono::steady_clock::time_point printPreparedFrameFinishStart{};
        if (dirProfileEnabled) {
            printPreparedFrameFinishStart = std::chrono::steady_clock::now();
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
        if (dirProfileEnabled) {
            printPreparedFrameFinishMs =
                elapsed_ms_since(printPreparedFrameFinishStart);
            trace_spatial_dir_profile(
                "print",
                width,
                height,
                spatialDir,
                SpatialDirProfileAdmittedRoles{printDirAdmittedRoles, printDirAdmittedTargetRoles},
                printDirProfile,
                printDirProfileCaptured,
                printDirScratchOverflow,
                spatialDirDescriptorMs,
                printDirPrepareMs,
                printDirBuildHostMs,
                printPipelineCudaMs,
                printPipelineLaunchHostMs,
                printCompositeProfile,
                printRecipe->profileRoute,
                printRecipe->hash,
                printRecipe->visualGrain,
                printRecipe->filmJuicerEffects,
                printVisualGrainDescriptor
                    ? &*printVisualGrainDescriptor
                    : nullptr,
                printGrainWorkspace.scratchShape,
                printGrainWorkspace.overflow,
                printGrainResources.staticNoise.version,
                printUploadTrace,
                printGrainProfile,
                printGrainDescriptorMs,
                printPreparedFramePrepareMs,
                printPreparedFrameFinishMs,
                printGrainSubmitHostMs,
                useFocusedSplit,
                filmEffectsActive,
                scannerPostEffects.active());
        }
        return;
    }

    trace_and_throw_cuda_policy_fatal(
        "CUDA unfocused launch blocked",
        "FocusedRenderStateRequiredAfterPhase4C");
#endif
}
