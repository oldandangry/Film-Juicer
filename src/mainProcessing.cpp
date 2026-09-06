// JuicerProcessing.cpp

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <cctype>
#include <string>
#include <mutex>
#include <limits>
#include <optional>

#include "FilmEffectsFrameDescriptors.h"
#include "GaussianSciPy.h"

#include <cuda_runtime.h>
#include <cuda.h>
#if defined(_WIN32)
#include <windows.h>
#endif

#include "Cuda/Diffusion/JuicerCudaDiffusion.h"
#include "Cuda/JuicerCudaFilmPayloads.h"
#include "Cuda/JuicerCudaResources.h"
#include "Cuda/JuicerCudaPayloads.h"
#include "Cuda/ResourceManager/JuicerCudaResourceCore.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"

extern "C" cudaError_t juicer_cuda_negative_direct_pipeline(
    const JuicerCuda::DirectPipelineRunParams* hParams,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_print_focused_pipeline(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_direct_focused_scan_linear_rgb(
    const JuicerCuda::DirectPipelineRunParams* hParams,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
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
    const float* filmDustTransmittance,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_print_focused_enlarger_linear_exposure(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    const float* dDensityC,
    const float* dDensityM,
    const float* dDensityY,
    JuicerCuda::EnlargerPrintLinearExposurePlanes planes,
    const float* filmDustTransmittance,
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
    const float* filmDustTransmittance,
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
    const JuicerCuda::FilmDefectsPayload* gateDefects,
    const JuicerCuda::GateWeavePayload* weave,
    const float* gateTransmittance,
    int gateTransmittanceWidth,
    int gateTransmittanceHeight,
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
    const JuicerCuda::FilmDefectsPayload* gateDefects,
    const JuicerCuda::GateWeavePayload* weave,
    const float* gateTransmittance,
    int gateTransmittanceWidth,
    int gateTransmittanceHeight,
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
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_apply_film_defects(
    const JuicerCuda::FilmDefectsPayload* defects,
    int width,
    int height,
    float* densityC,
    float* densityM,
    float* densityY,
    float* filmDustTransmittance,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_build_gate_defect_transmittance_focused(
    const JuicerCuda::FilmDefectsPayload* defects,
    float* gateTransmittance,
    int gateTransmittanceWidth,
    int gateTransmittanceHeight,
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
#include "ProcessRoot.h"
#include "JuicerState.h"
#include "Scanner.h"
#include "OutputColor.h"
#include "mainProcessing.h"

namespace {
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
               left.hash == right.hash;
    }

    bool same_diffusion_plan_key(
        const Spektrafilm::DiffusionPlanKey& left,
        const Spektrafilm::DiffusionPlanKey& right) noexcept {
        return left.extent == right.extent &&
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
        if (stageDescriptor.stage != expectedStage || !prepared.active ||
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
        out.fullFrame = frameSet.fullFrame;
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

    bool launch_scatter_halation_for_route(
        JuicerCuda::ScatterHalationPreparedView view,
        JuicerCuda::CameraFilmLinearExposurePlanes cameraDiffusionCarrier,
        bool cameraDiffusionLeaseActive,
        const char* route,
        const std::string& filmProfileKey,
        std::uint64_t filmProfileAssetVersionToken,
        const JuicerCuda::ResourceManager::DeviceContextKey& contextKey,
        std::uint64_t contextEpoch,
        cudaStream_t stream,
        std::string& outDiagnostic) {
        const auto stage_label = [](JuicerCuda::ScatterHalationLaunchStage stage) {
            switch (stage) {
                case JuicerCuda::ScatterHalationLaunchStage::Binding:
                    return "Binding";
                case JuicerCuda::ScatterHalationLaunchStage::ScatterAccumulatorClear:
                    return "ScatterAccumulatorClear";
                case JuicerCuda::ScatterHalationLaunchStage::ScatterTail:
                    return "ScatterTail";
                case JuicerCuda::ScatterHalationLaunchStage::ScatterCore:
                    return "ScatterCore";
                case JuicerCuda::ScatterHalationLaunchStage::BackReflectionAccumulatorClear:
                    return "BackReflectionAccumulatorClear";
                case JuicerCuda::ScatterHalationLaunchStage::BackReflectionBounce:
                    return "BackReflectionBounce";
                case JuicerCuda::ScatterHalationLaunchStage::BackReflectionFinalize:
                    return "BackReflectionFinalize";
                case JuicerCuda::ScatterHalationLaunchStage::None:
                    break;
            }
            return static_cast<const char*>(nullptr);
        };
        const auto channel_label = [](JuicerCuda::ScatterHalationLaunchChannel channel) {
            switch (channel) {
                case JuicerCuda::ScatterHalationLaunchChannel::Red:
                    return "Red";
                case JuicerCuda::ScatterHalationLaunchChannel::Green:
                    return "Green";
                case JuicerCuda::ScatterHalationLaunchChannel::Blue:
                    return "Blue";
                case JuicerCuda::ScatterHalationLaunchChannel::None:
                    break;
            }
            return static_cast<const char*>(nullptr);
        };
        const auto valid_distinct_planes = [](const auto& planes, int width) {
            return planes.redSensitive && planes.greenSensitive &&
                   planes.blueSensitive &&
                   planes.rowStrideFloats >= static_cast<std::size_t>(width) &&
                   planes.redSensitive != planes.greenSensitive &&
                   planes.redSensitive != planes.blueSensitive &&
                   planes.greenSensitive != planes.blueSensitive;
        };
        const auto append_identity = [&](const char* failedRequirement,
                                         cudaError_t status) {
            outDiagnostic = "ScatterHalationLaunchFailure route=";
            outDiagnostic += nonempty_cstr_or(route, "unknown");
            outDiagnostic += " domain=FilmLinearExposure film_profile_key=";
            outDiagnostic += filmProfileKey;
            outDiagnostic += " film_profile_asset_version_token=";
            outDiagnostic += std::to_string(filmProfileAssetVersionToken);
            outDiagnostic += " backend=Exact descriptor_recipe_hash=";
            outDiagnostic += std::to_string(
                view.descriptor ? view.descriptor->recipeHash : 0);
            outDiagnostic += " device_id=" +
                             std::to_string(contextKey.deviceId);
            outDiagnostic += " context=" +
                             std::to_string(static_cast<unsigned long long>(
                                 reinterpret_cast<std::uintptr_t>(
                                     contextKey.contextOpaque)));
            outDiagnostic += " context_epoch=" +
                             std::to_string(contextEpoch);
            outDiagnostic += " cuda_status=" +
                             std::to_string(static_cast<int>(status));
            outDiagnostic += " failed_requirement=";
            outDiagnostic += failedRequirement;
        };

        if (!view.descriptor || view.descriptor->recipeHash == 0 ||
            view.fullFrameWidth <= 0 || view.fullFrameHeight <= 0 ||
            !view.filterTemp || !view.weightedAccumulation ||
            view.filterTemp == view.weightedAccumulation) {
            append_identity("prepared_view_binding", cudaErrorInvalidValue);
            return false;
        }
        if (view.carrierSource ==
            JuicerCuda::ScatterHalationCarrierSource::CameraDiffusionStagePlanes) {
            const bool preparedCarrierEmpty =
                !view.currentCarrier.redSensitive &&
                !view.currentCarrier.greenSensitive &&
                !view.currentCarrier.blueSensitive &&
                view.currentCarrier.rowStrideFloats == 0;
            if (!cameraDiffusionLeaseActive || !preparedCarrierEmpty ||
                !valid_distinct_planes(
                    cameraDiffusionCarrier,
                    view.fullFrameWidth) ||
                cameraDiffusionCarrier.redSensitive == view.filterTemp ||
                cameraDiffusionCarrier.redSensitive ==
                    view.weightedAccumulation ||
                cameraDiffusionCarrier.greenSensitive == view.filterTemp ||
                cameraDiffusionCarrier.greenSensitive ==
                    view.weightedAccumulation ||
                cameraDiffusionCarrier.blueSensitive == view.filterTemp ||
                cameraDiffusionCarrier.blueSensitive ==
                    view.weightedAccumulation) {
                append_identity(
                    "camera_diffusion_carrier_binding",
                    cudaErrorInvalidValue);
                return false;
            }
            view.currentCarrier = cameraDiffusionCarrier;
        } else if (cameraDiffusionLeaseActive ||
                   cameraDiffusionCarrier.redSensitive ||
                   cameraDiffusionCarrier.greenSensitive ||
                   cameraDiffusionCarrier.blueSensitive ||
                   cameraDiffusionCarrier.rowStrideFloats != 0 ||
                   !valid_distinct_planes(
                       view.currentCarrier,
                       view.fullFrameWidth)) {
            append_identity("dedicated_carrier_binding", cudaErrorInvalidValue);
            return false;
        }

        const JuicerCuda::ScatterHalationLaunchResult result =
            JuicerCuda::launch_scatter_halation(view, stream);
        if (result.status == cudaSuccess) {
            outDiagnostic.clear();
            return true;
        }
        append_identity(
            result.stage == JuicerCuda::ScatterHalationLaunchStage::Binding
                ? "launcher_binding"
                : "launcher_submission",
            result.status);
        if (const char* stage = stage_label(result.stage)) {
            outDiagnostic += " observing_stage=";
            outDiagnostic += stage;
        }
        if (const char* channel = channel_label(result.channel)) {
            outDiagnostic += " observing_channel=";
            outDiagnostic += channel;
        }
        if (result.gaussianIndex >= 0) {
            outDiagnostic += " observing_gaussian_index=" +
                             std::to_string(result.gaussianIndex);
        }
        return false;
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

    Spektrafilm::DirFrameExtent spatial_dir_extent_from_rect(const OfxRectI& rect) {
        Spektrafilm::DirFrameExtent extent{};
        extent.x = rect.x1;
        extent.y = rect.y1;
        extent.width = rect.x2 - rect.x1;
        extent.height = rect.y2 - rect.y1;
        return extent;
    }

#if JUICER_DIAGNOSTICS_COMPILED
    const Spektrafilm::DirGaussianComponentPlan& spatial_dir_component_or_empty(
        const Spektrafilm::SpatialDirDescriptor& descriptor,
        int index) {
        static const Spektrafilm::DirGaussianComponentPlan kEmpty{};
        if (index < 0 || index >= descriptor.filterPlan.componentCount) {
            return kEmpty;
        }
        return descriptor.filterPlan.components[static_cast<std::size_t>(index)];
    }
#endif

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

    std::int64_t frame_index_from_time(double time) {
        return static_cast<std::int64_t>(std::floor(finite_or(time, 0.0)));
    }

    std::uint64_t session_seed_or_default(std::uint64_t sessionSeed) {
        return (sessionSeed != 0) ? sessionSeed : 1;
    }

    std::uint64_t instance_token_or_session_seed(std::uint64_t instanceToken, std::uint64_t sessionSeed) {
        return (instanceToken != 0) ? instanceToken : session_seed_or_default(sessionSeed);
    }

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
        const Spektrafilm::FilmJuicerEffectsGeometry& geometry,
        float filmFormatLongEdgeMm,
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
                 clipToken,
                 geometry,
                 filmFormatLongEdgeMm},
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

#if JUICER_DIAGNOSTICS_COMPILED
    const char* submission_snapshot_action_label(bool reusingSnapshotLatch) {
        return reusingSnapshotLatch ? "reuse" : "new";
    }
#endif

} // namespace

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
        if (descriptor.hash == 0) {
            descriptor.hash = 1;
        }
        return descriptor;
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
    _sessionSeed = session_seed_or_default(request.sessionSeed);
    _instanceToken =
        instance_token_or_session_seed(request.instanceToken, _sessionSeed);
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
    _sessionSeed = session_seed_or_default(request.sessionSeed);
    _instanceToken =
        instance_token_or_session_seed(request.instanceToken, _sessionSeed);
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
        std::optional<Spektrafilm::VisualGrainFrameDescriptor>
            directVisualGrainDescriptor;
        std::string directGrainDescriptorDiagnostic;
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
                _effectsGeometry,
                directRecipe->filmRaw.filmFormatLongEdgeMm,
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
        const bool captureDensityConsumerActive =
            grainStageActive || filmEffectsActive;
        const ScatterHalationFrameDescriptor* halationRequestDescriptor =
            _scatterHalationDescriptor ? &*_scatterHalationDescriptor : nullptr;

        JuicerProcess::Root::CudaFramePreparationRequest directPreparation{};
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
        directPreparation.scatterHalationDescriptor = halationRequestDescriptor;
        directPreparation.visualGrainDescriptor =
            directVisualGrainDescriptor;
        directPreparation.effectsDescriptor = directEffectsDescriptor;
        directPreparation.requestedWidth = width;
        directPreparation.requestedHeight = height;
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

        JuicerCuda::ScatterHalationPreparedView directHalation =
            preparedFrame.scatter_halation_resources();
        const bool halationExecutable = directHalation.descriptor != nullptr;
        const bool cameraFilmLinearActive =
            cameraDiffusionActive || halationExecutable;
        const bool directUseFocusedSplit =
            cameraFilmLinearActive || scannerPostEffects.active() ||
            grainStageActive || filmEffectsActive || gateOutputActive;
        const auto directHalationCompletionDiagnostic =
            [&](const char* boundary, const std::string& cudaStatus) {
                std::string diagnostic =
                    "ScatterHalationCompletionObservation route=";
                diagnostic += Spektrafilm::scan_route_label(
                    directRecipe->profileRoute.scanRoute);
                diagnostic +=
                    " domain=FilmLinearExposure component=pipeline film_profile_key=";
                diagnostic += directRecipe->profileRoute.filmProfileKey;
                diagnostic += " film_profile_asset_version_token=";
                diagnostic += std::to_string(
                    directRecipe->profileRoute.filmProfileAssetVersionToken);
                diagnostic +=
                    " backend=Exact in_flight_descriptor_recipe_hash=";
                diagnostic += std::to_string(
                    directHalation.descriptor
                        ? directHalation.descriptor->recipeHash
                        : 0);
                diagnostic += " device_id=" +
                              std::to_string(deviceContextKey.deviceId);
                diagnostic += " context=" +
                              std::to_string(static_cast<unsigned long long>(
                                  reinterpret_cast<std::uintptr_t>(
                                      deviceContextKey.contextOpaque)));
                diagnostic += " context_epoch=" +
                              std::to_string(snapshot.contextEpoch);
                diagnostic += " observing_completion_boundary=";
                diagnostic += boundary;
                diagnostic += " cuda_status=";
                diagnostic += cudaStatus;
                return diagnostic;
            };

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
                throw_direct_restriction(cameraBindingDiagnostic.c_str());
            }
        } else if (diffusionPrepared.active) {
            throw_direct_restriction(
                "ResourceDescriptorMismatch component=diffusion field=disabled_prepared_view");
        }

        const JuicerProcess::Root::PreparedCudaFrame::FocusedPreparedView prepared =
            preparedFrame.focused_resources();
        if (!prepared.active ||
            prepared.densityBoundsHash != directRecipe->densityBounds.hash ||
            prepared.scannerDescriptorHash != scannerDescriptor.hash ||
            prepared.selectedMethod != directRecipe->filmRaw.rgbToRawMethod) {
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
                throw_direct_restriction(detail_or_unknown(meterError));
            }
            preparedFrame.mark_auto_exposure_metered(
                JuicerProcess::Root::PreparedCudaFrame::AutoExposureMeteredResult{
                    autoExposureDescriptor.hash});
            autoExposureScaleDevice = buffers.deviceState.exposureScale;
        }

        JuicerCuda::FilmPayloadPack directFilmPayloads{};
        std::string packDiagnostic;
        if (!JuicerCuda::pack_film_payloads(
                directRecipe->filmRaw,
                directRecipe->filmDevelop,
                directRecipe->dirCouplers,
                directRecipe->densityBounds,
                prepared.film,
                autoExposureScaleDevice,
                scannerCorrection.exposureScale,
                directFilmPayloads,
                packDiagnostic)) {
            throw_direct_restriction(packDiagnostic.c_str());
        }
        run.filmRaw = directFilmPayloads.filmRaw;
        run.filmExpose = directFilmPayloads.filmExposure;
        run.filmDevelop = directFilmPayloads.filmDevelop;

        JuicerCuda::CameraFilmLinearExposurePlanes directCameraFilmLinear{};
        if (cameraDiffusionActive) {
            if (!directCameraDiffusion.active) {
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
        } else if (halationExecutable) {
            directCameraFilmLinear = directHalation.currentCarrier;
            const cudaError_t exposureError =
                juicer_cuda_direct_camera_film_linear_exposure(
                    &run,
                    directCameraFilmLinear,
                    _pCudaStream);
            if (exposureError != cudaSuccess) {
                throw_cuda_stage_fatal(
                    "direct_camera_film_linear_exposure_launch",
                    "direct camera film-linear exposure launch failed",
                    exposureError);
            }
        }
        if (halationExecutable) {
            std::string halationDiagnostic;
            // bind_camera_diffusion_carrier_immediately
            if (!launch_scatter_halation_for_route(
                    directHalation,
                    cameraDiffusionActive
                        ? directCameraFilmLinear
                        : JuicerCuda::CameraFilmLinearExposurePlanes{},
                    cameraDiffusionActive && directCameraDiffusion.active,
                    Spektrafilm::scan_route_label(
                        directRecipe->profileRoute.scanRoute),
                    directRecipe->profileRoute.filmProfileKey,
                    directRecipe->profileRoute.filmProfileAssetVersionToken,
                    deviceContextKey,
                    snapshot.contextEpoch,
                    reinterpret_cast<cudaStream_t>(_pCudaStream),
                    halationDiagnostic)) {
                throw_submission_fatal(
                    "direct_scatter_halation_launch",
                    "direct scatter-halation launch failed",
                    halationDiagnostic);
            }
            // post_halation_camera_film_linear
        }

        const bool directUseFusedScannerPostSpatialDirHandoff =
            !cameraFilmLinearActive && directSpatialDir.hash != 0 &&
            scannerPostEffects.active() &&
            directSpatialDir.approximation ==
                Spektrafilm::DirApproximationMarker::SpektrafilmStrict &&
            !captureDensityConsumerActive;
        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker focusedWorkspace =
            preparedFrame.workspace_lease();
        if (focusedWorkspace.active()) {
            std::string transitionError;
            if (!preparedFrame.checkpoint_large_scratch_transition(
                    focusedWorkspace,
                    _pCudaStream,
                    "direct_large_scratch_transition",
                    transitionError)) {
                throw_submission_fatal(
                    "direct_large_scratch_transition",
                    "direct large scratch transition checkpoint failed",
                    transitionError);
            }
        }
        bool directDirUsesSourceBuildCachedLogRaw = false;
        bool directCaptureDensityReady = false;
        if (directSpatialDir.hash != 0) {
            std::string spatialError;
            if (!preparedFrame.prepare_spatial_dir_resources(
                    directSpatialDir,
                    focusedWorkspace,
                    _pCudaStream,
                    spatialError)) {
                throw_submission_fatal(
                    "direct_spatial_dir_prepare",
                    "direct spatial DIR preparation failed",
                    spatialError);
            }
        }
        JuicerProcess::Root::PreparedCudaFrame::CaptureFilmDensityWorkspaceView
            directCaptureDensity{};
        JuicerProcess::Root::PreparedCudaFrame::FocusedRgbWorkspaceView
            directFocusedRgb{};
        JuicerProcess::Root::PreparedCudaFrame::VisualGrainWorkspaceView
            directGrainWorkspace{};
        JuicerCuda::PreparedVisualGrainView directGrainResources{};
        JuicerProcess::Root::PreparedCudaFrame::ScannerWorkspaceView
            directEffectsWorkspace{};
        JuicerCuda::GrainPayload directGrainPayload{};
        JuicerCuda::GrainKernelPayload directGrainKernels{};
        JuicerCuda::FilmDefectsPayload directEffectsPayload{};
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
                throw_direct_restriction(
                    effectsPayloadDiagnostic.empty()
                        ? "MissingRequiredResource phase=effects_route field=effects_binding"
                        : effectsPayloadDiagnostic.c_str());
            }
        } else if (preparedDirectEffectsDescriptor) {
            throw_direct_restriction(
                "ResourceDescriptorMismatch phase=effects_route field=inactive_descriptor");
        }
        if (directUseFocusedSplit) {
            std::string focusedWorkspaceError;
            if (!preparedFrame.stage_optical_workspace(
                    focusedWorkspace,
                    _pCudaStream,
                    focusedWorkspaceError)) {
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
                throw_direct_restriction(
                    "MissingRequiredResource phase=grain_route field=focused_triplet");
            }
            directEffectsWorkspace =
                preparedFrame.scanner_workspace(focusedWorkspace);
            if (!directEffectsWorkspace.active ||
                (directEffectsDescriptor.has_value() && directEffectsDescriptor->filmDust.slotProbability > 0.0f &&
                 !directEffectsWorkspace.filmDustTransmittance) ||
                (directEffectsDescriptor.has_value() &&
                 directEffectsDescriptor->gateTransmittanceActive &&
                 !directEffectsWorkspace.hasGateTransmittance)) {
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
                throw_direct_restriction(
                    "MissingRequiredResource phase=3D-3 field=prepared_spatial_dir");
            }
            const Spektrafilm::DirScratchPlaneRoles& directDirAdmittedRoles =
                scratch.planeRoles;
            const Spektrafilm::DirScratchPlaneRoles& directDirAdmittedTargetRoles =
                scratch.targetPlaneRoles;
            directDirUsesSourceBuildCachedLogRaw =
                fused_alias_uses_source_build_cached_log_raw(
                    directUseFusedScannerPostSpatialDirHandoff,
                    directDirAdmittedRoles,
                    directDirAdmittedTargetRoles);
            if (directDirUsesSourceBuildCachedLogRaw &&
                !spatial_dir_required_cached_log_raw_present(
                    scratch,
                    directDirAdmittedRoles.cachedLogRawPlanes)) {
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
            const cudaError_t dirError = juicer_cuda_build_direct_spatial_dir(
                &run,
                dirBuildRequest);
            if (dirError != cudaSuccess) {
                throw_cuda_stage_fatal(
                    "direct_spatial_dir_launch",
                    "direct spatial DIR build failed",
                    dirError);
            }
            if (!directUseFusedScannerPostSpatialDirHandoff) {
                std::string stageError;
                if (!preparedFrame.stage_spatial_dir_cached_log_raw_for_final_develop(
                        focusedWorkspace,
                        _pCudaStream,
                        stageError)) {
                    throw_submission_fatal(
                        "direct_spatial_dir_cached_log_raw_stage",
                        "direct spatial DIR cached log raw staging failed",
                        stageError);
                }
            }
            const auto finalScratch = preparedFrame.spatial_dir_scratch(focusedWorkspace);
            const bool directDirRequiresCachedLogRaw =
                directDirUsesSourceBuildCachedLogRaw ||
                (!directUseFusedScannerPostSpatialDirHandoff &&
                 directDirAdmittedTargetRoles.cachedLogRawPlanes == 3);
            if (!finalScratch.filteredCorrectionY || !finalScratch.filteredCorrectionM ||
                !finalScratch.filteredCorrectionC) {
                throw_direct_restriction(
                    "MissingRequiredResource phase=3D-3 field=spatial_dir_filtered_correction");
            }
            if (directDirRequiresCachedLogRaw &&
                !spatial_dir_required_cached_log_raw_present(
                    finalScratch,
                    directDirAdmittedTargetRoles.cachedLogRawPlanes)) {
                throw_direct_restriction(
                    "MissingRequiredResource phase=3D-3 field=spatial_dir_cached_log_raw");
            }
            if (directDirRequiresCachedLogRaw && !directDirUsesSourceBuildCachedLogRaw) {
                const cudaError_t logRawError = cameraFilmLinearActive
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
                    throw_cuda_stage_fatal(
                        "direct_spatial_dir_cached_log_raw_launch",
                        "direct spatial DIR cached log raw build failed",
                        logRawError);
                }
            }
            bind_spatial_dir_final_develop_to_payload(run.filmDevelop, finalScratch);
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
            throw_submission_fatal("direct_scan_error_stage", "direct scan error stage failed", scanError);
        }
        cudaError_t launchError = cudaSuccess;
        if (directUseFocusedSplit &&
            !directUseFusedScannerPostSpatialDirHandoff) {
            launchError = cameraFilmLinearActive
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
                throw_cuda_stage_fatal(
                    "direct_capture_density_launch",
                    "direct capture-film density launch failed",
                    launchError);
            }
            directCaptureDensityReady = true;
            if (directSpatialDir.hash != 0) {
                run.filmDevelop.spatialDir.logRawB = nullptr;
                run.filmDevelop.spatialDir.logRawG = nullptr;
                run.filmDevelop.spatialDir.logRawR = nullptr;
            }
            if (grainStageActive) {
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
                    _pCudaStream);
                if (launchError != cudaSuccess) {
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
                    directEffectsWorkspace.filmDustTransmittance,
                    _pCudaStream);
                if (launchError != cudaSuccess) {
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
                throw_submission_fatal(
                    "direct_diffusion_release_after_use",
                    "direct diffusion resource release failed",
                    diffusionReleaseError);
            }
            directCameraDiffusion = {};
            directCameraFilmLinear = {};
        }

        if (gateOutputActive && !grainDebugActive &&
            directEffectsDescriptor->gateTransmittanceActive) {
            launchError = juicer_cuda_build_gate_defect_transmittance_focused(
                &directEffectsPayload,
                directEffectsWorkspace.gateTransmittance,
                directEffectsWorkspace.gateTransmittanceWidth,
                directEffectsWorkspace.gateTransmittanceHeight,
                _pCudaStream);
            if (launchError != cudaSuccess) {
                throw_cuda_stage_fatal(
                    "direct_gate_transmittance_launch",
                    "direct gate defect mask launch failed",
                    launchError);
            }
        }

        if (directUseFocusedSplit) {
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
                    throw_direct_restriction(
                        "MissingRequiredResource phase=8C field=scanner_post_effects");
                }
            }
            if (!grainDebugActive) {
                if (directUseFusedScannerPostSpatialDirHandoff) {
                    launchError = juicer_cuda_direct_focused_scan_linear_rgb(
                        &run,
                        directFocusedRgb.r,
                        directFocusedRgb.g,
                        directFocusedRgb.b,
                        _pCudaStream);
                } else {
                    if (!directCaptureDensityReady) {
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
                        directEffectsWorkspace.filmDustTransmittance,
                        _pCudaStream);
                }
                if (launchError != cudaSuccess) {
                    throw_cuda_stage_fatal(
                        "direct_scanner_linear_launch",
                        "direct scanner-linear launch failed",
                        launchError);
                }
            }
            if (directDirUsesSourceBuildCachedLogRaw) {
                run.filmDevelop.spatialDir.logRawB = nullptr;
                run.filmDevelop.spatialDir.logRawG = nullptr;
                run.filmDevelop.spatialDir.logRawR = nullptr;
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
                        directEffectsDescriptor->gateTransmittanceActive
                    ? directEffectsWorkspace.gateTransmittance
                    : nullptr,
                gateOutputActive && !grainDebugActive &&
                        directEffectsDescriptor->gateTransmittanceActive
                    ? directEffectsWorkspace.gateTransmittanceWidth
                    : 0,
                gateOutputActive && !grainDebugActive &&
                        directEffectsDescriptor->gateTransmittanceActive
                    ? directEffectsWorkspace.gateTransmittanceHeight
                    : 0,
                _pCudaStream);
            // Focused RGB can alias the spatial-DIR filtered planes, so scanner output is their final consumer.
            if (launchError == cudaSuccess && directSpatialDir.hash != 0) {
                clear_spatial_dir_final_develop_payload_bindings(run.filmDevelop);
            }
        } else {
            launchError = juicer_cuda_negative_direct_pipeline(&run, _pCudaStream);
        }
        if (launchError != cudaSuccess) {
            throw_cuda_stage_fatal("direct_negative_pipeline_launch", "direct negative pipeline launch failed", launchError);
        }
        if (directSpatialDir.hash != 0 && !directUseFocusedSplit) {
            run.filmDevelop.spatialDir.logRawB = nullptr;
            run.filmDevelop.spatialDir.logRawG = nullptr;
            run.filmDevelop.spatialDir.logRawR = nullptr;
        }
        if (!preparedFrame.finalize_scan_error_stage(run.scanStage.scanErrorFlag, _pCudaStream, scanError)) {
            const std::string diagnostic = halationExecutable
                                               ? directHalationCompletionDiagnostic(
                                                     "scan_error_finalize",
                                                     scanError)
                                               : scanError;
            throw_submission_fatal("direct_scan_error_finalize", "direct scan error finalize failed", diagnostic);
        }
        record_cuda_use(preparedFrame);
        std::string finishError;
        if (!preparedFrame.finish(_pCudaStream, finishError)) {
            const std::string diagnostic = halationExecutable
                                               ? directHalationCompletionDiagnostic(
                                                     "prepared_frame_finish",
                                                     finishError)
                                               : finishError;
            throw_submission_fatal("direct_prepared_frame_finish", "direct prepared frame finish failed", diagnostic);
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
                    &printRecipe->scannerOutput},
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
        Spektrafilm::SpatialDirDescriptor spatialDir{};
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

        std::optional<Spektrafilm::VisualGrainFrameDescriptor>
            printVisualGrainDescriptor;
        std::string printGrainDescriptorDiagnostic;
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
                _effectsGeometry,
                printRecipe->filmRaw.filmFormatLongEdgeMm,
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
        const bool captureDensityConsumerActive =
            grainStageActive || filmEffectsActive;
        const ScatterHalationFrameDescriptor* halationRequestDescriptor =
            _scatterHalationDescriptor ? &*_scatterHalationDescriptor : nullptr;

        JuicerProcess::Root::CudaFramePreparationRequest preparation{};
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
        preparation.scatterHalationDescriptor = halationRequestDescriptor;
        preparation.visualGrainDescriptor =
            printVisualGrainDescriptor;
        preparation.effectsDescriptor = printEffectsDescriptor;
        preparation.requestedWidth = width;
        preparation.requestedHeight = height;
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

        JuicerCuda::ScatterHalationPreparedView printHalation =
            preparedFrame.scatter_halation_resources();
        const bool halationExecutable = printHalation.descriptor != nullptr;
        const bool cameraFilmLinearActive =
            cameraDiffusionActive || halationExecutable;
        const bool printUseFocusedSplit =
            routeDiffusionActive || halationExecutable ||
            scannerPostEffects.active() || grainStageActive ||
            filmEffectsActive || gateOutputActive;
        const auto printHalationCompletionDiagnostic =
            [&](const char* boundary, const std::string& cudaStatus) {
                std::string diagnostic =
                    "ScatterHalationCompletionObservation route=";
                diagnostic += Spektrafilm::scan_route_label(
                    printRecipe->profileRoute.scanRoute);
                diagnostic +=
                    " domain=FilmLinearExposure component=pipeline film_profile_key=";
                diagnostic += printRecipe->profileRoute.filmProfileKey;
                diagnostic += " film_profile_asset_version_token=";
                diagnostic += std::to_string(
                    printRecipe->profileRoute.filmProfileAssetVersionToken);
                diagnostic +=
                    " backend=Exact in_flight_descriptor_recipe_hash=";
                diagnostic += std::to_string(
                    printHalation.descriptor
                        ? printHalation.descriptor->recipeHash
                        : 0);
                diagnostic += " device_id=" +
                              std::to_string(deviceContextKey.deviceId);
                diagnostic += " context=" +
                              std::to_string(static_cast<unsigned long long>(
                                  reinterpret_cast<std::uintptr_t>(
                                      deviceContextKey.contextOpaque)));
                diagnostic += " context_epoch=" +
                              std::to_string(snapshot.contextEpoch);
                diagnostic += " observing_completion_boundary=";
                diagnostic += boundary;
                diagnostic += " cuda_status=";
                diagnostic += cudaStatus;
                return diagnostic;
            };

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
                    throw_print_restriction(
                        enlargerBindingDiagnostic.c_str());
                }
            }
        } else if (diffusionPrepared.active) {
            throw_print_restriction(
                "ResourceDescriptorMismatch component=diffusion field=disabled_prepared_view");
        }

        const JuicerProcess::Root::PreparedCudaFrame::FocusedPreparedView prepared =
            preparedFrame.focused_resources();
        const JuicerProcess::Root::PreparedCudaFrame::PrintPreparedView preparedPrint =
            preparedFrame.print_resources();
        if (!prepared.active || !preparedPrint.active ||
            prepared.densityBoundsHash != printRecipe->densityBounds.hash ||
            prepared.scannerDescriptorHash != scannerDescriptor.hash ||
            prepared.selectedMethod != printRecipe->filmRaw.rgbToRawMethod) {
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
            throw_print_restriction(scannerDescriptorDiagnostic.c_str());
        }
        if (traceVerbose) {
            const PrintFilterRecipe& filters = printRecipe->print.filters;
            std::string msg;
            msg.reserve(512);
            msg = "event=focused_route_prepared recipeHash=";
            msg += std::to_string(printRecipe->hash);
            msg += " mainCmyCc(C/M/Y)=";
            msg += std::to_string(filters.mainCmyCc.c);
            msg += "/";
            msg += std::to_string(filters.mainCmyCc.m);
            msg += "/";
            msg += std::to_string(filters.mainCmyCc.y);
            msg += " preflashCmyCc(C/M/Y)=";
            msg += std::to_string(filters.preflashCmyCc.c);
            msg += "/";
            msg += std::to_string(filters.preflashCmyCc.m);
            msg += "/";
            msg += std::to_string(filters.preflashCmyCc.y);
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
                throw_print_restriction(detail_or_unknown(meterError));
            }
            preparedFrame.mark_auto_exposure_metered(
                JuicerProcess::Root::PreparedCudaFrame::AutoExposureMeteredResult{
                    autoExposureDescriptor.hash});
            autoExposureScaleDevice = buffers.deviceState.exposureScale;
        }

        JuicerCuda::FilmPayloadPack filmPayloads{};
        std::string payloadDiagnostic;
        if (!JuicerCuda::pack_film_payloads(
                printRecipe->filmRaw,
                printRecipe->filmDevelop,
                printRecipe->dirCouplers,
                printRecipe->enlargerFilmBounds,
                prepared.film,
                autoExposureScaleDevice,
                1.0f,
                filmPayloads,
                payloadDiagnostic)) {
            throw_print_restriction(payloadDiagnostic.c_str());
        }
        JuicerCuda::PrintCudaPayloadPack printPayloads{};
        if (!JuicerCuda::pack_print_cuda_payloads(
                printRecipe->print,
                preparedPrint,
                scannerCorrection.exposureScale,
                printPayloads,
                payloadDiagnostic)) {
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
        } else if (halationExecutable) {
            printCameraFilmLinear = printHalation.currentCarrier;
            const cudaError_t exposureError =
                juicer_cuda_print_camera_film_linear_exposure(
                    &run,
                    printCameraFilmLinear,
                    _pCudaStream);
            if (exposureError != cudaSuccess) {
                throw_cuda_stage_fatal(
                    "print_camera_film_linear_exposure_launch",
                    "print camera film-linear exposure launch failed",
                    exposureError);
            }
        }
        if (halationExecutable) {
            std::string halationDiagnostic;
            // bind_camera_diffusion_carrier_immediately
            if (!launch_scatter_halation_for_route(
                    printHalation,
                    cameraDiffusionActive
                        ? printCameraFilmLinear
                        : JuicerCuda::CameraFilmLinearExposurePlanes{},
                    cameraDiffusionActive && printCameraDiffusion.active,
                    Spektrafilm::scan_route_label(
                        printRecipe->profileRoute.scanRoute),
                    printRecipe->profileRoute.filmProfileKey,
                    printRecipe->profileRoute.filmProfileAssetVersionToken,
                    deviceContextKey,
                    snapshot.contextEpoch,
                    reinterpret_cast<cudaStream_t>(_pCudaStream),
                    halationDiagnostic)) {
                throw_submission_fatal(
                    "print_scatter_halation_launch",
                    "print scatter-halation launch failed",
                    halationDiagnostic);
            }
            // post_halation_camera_film_linear
        }

        const bool printUseFusedScannerPostSpatialDirHandoff =
            !routeDiffusionActive && !halationExecutable &&
            spatialDir.hash != 0 &&
            scannerPostEffects.active() &&
            spatialDir.approximation ==
                Spektrafilm::DirApproximationMarker::SpektrafilmStrict &&
            !captureDensityConsumerActive;
        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker focusedWorkspace =
            preparedFrame.workspace_lease();
        if (focusedWorkspace.active()) {
            std::string transitionError;
            if (!preparedFrame.checkpoint_large_scratch_transition(
                    focusedWorkspace,
                    _pCudaStream,
                    "print_large_scratch_transition",
                    transitionError)) {
                throw_submission_fatal(
                    "print_large_scratch_transition",
                    "print large scratch transition checkpoint failed",
                    transitionError);
            }
        }
        bool printDirUsesSourceBuildCachedLogRaw = false;
        bool printCaptureDensityReady = false;
        if (spatialDir.hash != 0) {
            std::string spatialError;
            if (!preparedFrame.prepare_spatial_dir_resources(
                    spatialDir,
                    focusedWorkspace,
                    _pCudaStream,
                    spatialError)) {
                throw_submission_fatal(
                    "print_spatial_dir_prepare",
                    "print spatial DIR preparation failed",
                    spatialError);
            }
        }
        JuicerProcess::Root::PreparedCudaFrame::CaptureFilmDensityWorkspaceView
            printCaptureDensity{};
        JuicerProcess::Root::PreparedCudaFrame::FocusedRgbWorkspaceView
            printFocusedRgb{};
        JuicerProcess::Root::PreparedCudaFrame::VisualGrainWorkspaceView
            printGrainWorkspace{};
        JuicerCuda::PreparedVisualGrainView printGrainResources{};
        JuicerProcess::Root::PreparedCudaFrame::ScannerWorkspaceView
            printEffectsWorkspace{};
        JuicerCuda::GrainPayload printGrainPayload{};
        JuicerCuda::GrainKernelPayload printGrainKernels{};
        JuicerCuda::FilmDefectsPayload printEffectsPayload{};
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
                throw_print_restriction(
                    effectsPayloadDiagnostic.empty()
                        ? "MissingRequiredResource phase=effects_route field=effects_binding"
                        : effectsPayloadDiagnostic.c_str());
            }
        } else if (preparedPrintEffectsDescriptor) {
            throw_print_restriction(
                "ResourceDescriptorMismatch phase=effects_route field=inactive_descriptor");
        }
        if (printUseFocusedSplit) {
            std::string focusedWorkspaceError;
            if (!preparedFrame.stage_optical_workspace(
                    focusedWorkspace,
                    _pCudaStream,
                    focusedWorkspaceError)) {
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
                throw_print_restriction(
                    "MissingRequiredResource phase=grain_route field=focused_triplet");
            }
            printEffectsWorkspace =
                preparedFrame.scanner_workspace(focusedWorkspace);
            if (!printEffectsWorkspace.active ||
                (printEffectsDescriptor.has_value() && printEffectsDescriptor->filmDust.slotProbability > 0.0f &&
                 !printEffectsWorkspace.filmDustTransmittance) ||
                (printEffectsDescriptor.has_value() &&
                 printEffectsDescriptor->gateTransmittanceActive &&
                 !printEffectsWorkspace.hasGateTransmittance)) {
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
                throw_print_restriction(
                    "MissingRequiredResource phase=4C field=prepared_spatial_dir");
            }
            const Spektrafilm::DirScratchPlaneRoles& printDirAdmittedRoles =
                scratch.planeRoles;
            const Spektrafilm::DirScratchPlaneRoles& printDirAdmittedTargetRoles =
                scratch.targetPlaneRoles;
            printDirUsesSourceBuildCachedLogRaw =
                fused_alias_uses_source_build_cached_log_raw(
                    printUseFusedScannerPostSpatialDirHandoff,
                    printDirAdmittedRoles,
                    printDirAdmittedTargetRoles);
            if (printDirUsesSourceBuildCachedLogRaw &&
                !spatial_dir_required_cached_log_raw_present(
                    scratch,
                    printDirAdmittedRoles.cachedLogRawPlanes)) {
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
            const cudaError_t dirError = juicer_cuda_build_print_spatial_dir(
                &run,
                dirBuildRequest);
            if (dirError != cudaSuccess) {
                throw_cuda_stage_fatal(
                    "print_spatial_dir_launch",
                    "print spatial DIR build failed",
                    dirError);
            }
            if (!printUseFusedScannerPostSpatialDirHandoff) {
                std::string stageError;
                if (!preparedFrame.stage_spatial_dir_cached_log_raw_for_final_develop(
                        focusedWorkspace,
                        _pCudaStream,
                        stageError)) {
                    throw_submission_fatal(
                        "print_spatial_dir_cached_log_raw_stage",
                        "print spatial DIR cached log raw staging failed",
                        stageError);
                }
            }
            const auto finalScratch = preparedFrame.spatial_dir_scratch(focusedWorkspace);
            const bool printDirRequiresCachedLogRaw =
                printDirUsesSourceBuildCachedLogRaw ||
                (!printUseFusedScannerPostSpatialDirHandoff &&
                 printDirAdmittedTargetRoles.cachedLogRawPlanes == 3);
            if (!finalScratch.filteredCorrectionY || !finalScratch.filteredCorrectionM ||
                !finalScratch.filteredCorrectionC) {
                throw_print_restriction(
                    "MissingRequiredResource phase=4C field=spatial_dir_filtered_correction");
            }
            if (printDirRequiresCachedLogRaw &&
                !spatial_dir_required_cached_log_raw_present(
                    finalScratch,
                    printDirAdmittedTargetRoles.cachedLogRawPlanes)) {
                throw_print_restriction(
                    "MissingRequiredResource phase=4C field=spatial_dir_cached_log_raw");
            }
            if (printDirRequiresCachedLogRaw && !printDirUsesSourceBuildCachedLogRaw) {
                const cudaError_t logRawError = cameraFilmLinearActive
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
                    throw_cuda_stage_fatal(
                        "print_spatial_dir_cached_log_raw_launch",
                        "print spatial DIR cached log raw build failed",
                        logRawError);
                }
            }
            bind_spatial_dir_final_develop_to_payload(run.filmDevelop, finalScratch);
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
            throw_submission_fatal(
                "print_scan_error_stage",
                "print scan error stage failed",
                scanError);
        }
        cudaError_t launchError = cudaSuccess;
        if (printUseFocusedSplit &&
            !printUseFusedScannerPostSpatialDirHandoff) {
            launchError = cameraFilmLinearActive
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
                throw_cuda_stage_fatal(
                    "print_capture_density_launch",
                    "print capture-film density launch failed",
                    launchError);
            }
            printCaptureDensityReady = true;
            if (spatialDir.hash != 0) {
                run.filmDevelop.spatialDir.logRawB = nullptr;
                run.filmDevelop.spatialDir.logRawG = nullptr;
                run.filmDevelop.spatialDir.logRawR = nullptr;
            }
            if (grainStageActive) {
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
                    _pCudaStream);
                if (launchError != cudaSuccess) {
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
                    printEffectsWorkspace.filmDustTransmittance,
                    _pCudaStream);
                if (launchError != cudaSuccess) {
                    throw_cuda_stage_fatal(
                        "print_film_effects_launch",
                        "print film defects launch failed",
                        launchError);
                }
            }
            if (!grainDebugActive) {
                if (enlargerDiffusionActive) {
                    if (!printEnlargerDiffusion.active) {
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
                            printEffectsWorkspace.filmDustTransmittance,
                            _pCudaStream);
                    if (launchError != cudaSuccess) {
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
                            printEffectsWorkspace.filmDustTransmittance,
                            _pCudaStream);
                    if (launchError != cudaSuccess) {
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
            printEffectsDescriptor->gateTransmittanceActive) {
            launchError = juicer_cuda_build_gate_defect_transmittance_focused(
                &printEffectsPayload,
                printEffectsWorkspace.gateTransmittance,
                printEffectsWorkspace.gateTransmittanceWidth,
                printEffectsWorkspace.gateTransmittanceHeight,
                _pCudaStream);
            if (launchError != cudaSuccess) {
                throw_cuda_stage_fatal(
                    "print_gate_transmittance_launch",
                    "print gate defect mask launch failed",
                    launchError);
            }
        }

        if (printUseFocusedSplit) {
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
                    throw_print_restriction(
                        "MissingRequiredResource phase=8C field=scanner_post_effects");
                }
            }
            const std::uint64_t glareSeed = Hash::hash_uint64_values(
                {printRecipe->hash,
                 snapshot.frameToken.value,
                 static_cast<std::uint64_t>(srcBounds.x1),
                 static_cast<std::uint64_t>(srcBounds.y1)});
            if (!grainDebugActive) {
                if (printUseFusedScannerPostSpatialDirHandoff) {
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
                        _pCudaStream);
                } else {
                    if (!printCaptureDensityReady) {
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
                    throw_cuda_stage_fatal(
                        "print_scanner_linear_launch",
                        "print scanner-linear launch failed",
                        launchError);
                }
            }
            if (printDirUsesSourceBuildCachedLogRaw) {
                run.filmDevelop.spatialDir.logRawB = nullptr;
                run.filmDevelop.spatialDir.logRawG = nullptr;
                run.filmDevelop.spatialDir.logRawR = nullptr;
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
                        printEffectsDescriptor->gateTransmittanceActive
                    ? printEffectsWorkspace.gateTransmittance
                    : nullptr,
                gateOutputActive && !grainDebugActive &&
                        printEffectsDescriptor->gateTransmittanceActive
                    ? printEffectsWorkspace.gateTransmittanceWidth
                    : 0,
                gateOutputActive && !grainDebugActive &&
                        printEffectsDescriptor->gateTransmittanceActive
                    ? printEffectsWorkspace.gateTransmittanceHeight
                    : 0,
                _pCudaStream);
            // Focused RGB can alias the spatial-DIR filtered planes, so scanner output is their final consumer.
            if (launchError == cudaSuccess && spatialDir.hash != 0) {
                clear_spatial_dir_final_develop_payload_bindings(run.filmDevelop);
            }
        } else {
            launchError = juicer_cuda_print_focused_pipeline(&run, _pCudaStream);
        }
        if (launchError != cudaSuccess) {
            throw_cuda_stage_fatal(
                "print_pipeline_launch",
                "focused print pipeline launch failed",
                launchError);
        }
        if (spatialDir.hash != 0 && !printUseFocusedSplit) {
            run.filmDevelop.spatialDir.logRawB = nullptr;
            run.filmDevelop.spatialDir.logRawG = nullptr;
            run.filmDevelop.spatialDir.logRawR = nullptr;
        }
        if (!preparedFrame.finalize_scan_error_stage(
                run.scanStage.scanErrorFlag,
                _pCudaStream,
                scanError)) {
            const std::string diagnostic = halationExecutable
                                               ? printHalationCompletionDiagnostic(
                                                     "scan_error_finalize",
                                                     scanError)
                                               : scanError;
            throw_submission_fatal(
                "print_scan_error_finalize",
                "print scan error finalize failed",
                diagnostic);
        }
        record_cuda_use(preparedFrame);
        std::string finishError;
        if (!preparedFrame.finish(_pCudaStream, finishError)) {
            const std::string diagnostic = halationExecutable
                                               ? printHalationCompletionDiagnostic(
                                                     "prepared_frame_finish",
                                                     finishError)
                                               : finishError;
            throw_submission_fatal(
                "print_prepared_frame_finish",
                "print prepared frame finish failed",
                diagnostic);
        }
        return;
    }

    trace_and_throw_cuda_policy_fatal(
        "CUDA unfocused launch blocked",
        "FocusedRenderStateRequiredAfterPhase4C");
}
