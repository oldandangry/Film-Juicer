// Native CUDA execution; the temporary host adapter owns publication and host access.

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <cctype>
#include <string>
#include <limits>
#include <optional>
#include <variant>

#include "FilmEffectsFrameDescriptors.h"
#include "GaussianSciPy.h"

#include <cuda_runtime.h>

#include "Cuda/Diffusion/JuicerCudaDiffusion.h"
#include "Cuda/JuicerCudaDriver.h"
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

#include "Cuda/JuicerCudaExecutor.h"
#include "FocusedRenderPayload.h"
#include "Logging.h"
#include "Hash.h"
#include "SpectralData.h"
#include "ColorTransforms.h"
#include "SpectralProcessing.h"
#include "ProcessRoot.h"
#include "Scanner.h"
#include "OutputColor.h"

namespace {
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
        const std::size_t stageIndex =
            expectedStage ==
                        Spektrafilm::DiffusionLinearStage::CameraFilmLinear ||
                    !frameSet.camera
                ? 0
                : 1;
        const auto& stageGeometry =
            prepared.executionDescriptor.stages[stageIndex];
        const std::size_t spectrumIndex =
            stageGeometry.spectrumKeyIndex;
        if (spectrumIndex >= prepared.spectrumCount ||
            spectrumIndex >= prepared.spectra.size() ||
            spectrumIndex >=
                prepared.executionDescriptor.spectrumKeys.size()) {
            diagnostic =
                "ResourceDescriptorMismatch component=diffusion stage=";
            diagnostic += stageLabel;
            diagnostic += " field=spectrum_index";
            return false;
        }

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
        out.geometry = stageGeometry;
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
        std::string_view filmProfileKey,
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

    bool spatial_dir_source_build_retains_cached_log_raw(
        const Spektrafilm::DirScratchPlaneRoles& roles,
        const Spektrafilm::DirScratchPlaneRoles& targetRoles) noexcept {
        // The three-channel source pass already computes log exposure. Populate
        // cache planes retained through final develop, including grain routes.
        return roles.rawCorrectionPlanes == 3 &&
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

    Spektrafilm::DirFrameExtent spatial_dir_extent_from_rect(const JuicerCuda::FrameRect& rect) {
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

    JuicerCuda::SpatialDirFilterOperator cuda_dir_filter_operator(
        Spektrafilm::DirReferenceOperator referenceOperator) noexcept {
        switch (referenceOperator) {
            case Spektrafilm::DirReferenceOperator::Identity:
                return JuicerCuda::SpatialDirFilterOperator::Identity;
            case Spektrafilm::DirReferenceOperator::SpektrafilmSmallFirReflect:
                return JuicerCuda::SpatialDirFilterOperator::FirReflect;
            case Spektrafilm::DirReferenceOperator::SpektrafilmLargeYvvReflect:
                return JuicerCuda::SpatialDirFilterOperator::YvvReflect;
            case Spektrafilm::DirReferenceOperator::None:
            default:
                return JuicerCuda::SpatialDirFilterOperator::None;
        }
    }

    void bind_spatial_dir_iir(
        JuicerCuda::SpatialDirFilterSpec& destination,
        const Spektrafilm::DirGaussianComponentPlan& component,
        const JuicerProcess::Root::PreparedCudaFrame::SpatialDirPreparedView::BoundaryView&
            boundary) noexcept {
        destination.iir.feedforward = component.iir.feedforward;
        std::copy(
            component.iir.feedback.begin(),
            component.iir.feedback.end(),
            std::begin(destination.iir.feedback));
        destination.horizontalBoundary.initialWeights = boundary.horizontalWeights;
        destination.horizontalBoundary.initialWeightLength = boundary.horizontalLength;
        destination.horizontalBoundary.terminalSize = boundary.horizontalTerminalSize;
        std::copy(
            boundary.horizontalTerminalMatrix.begin(),
            boundary.horizontalTerminalMatrix.end(),
            std::begin(destination.horizontalBoundary.terminalMatrix));
        destination.verticalBoundary.initialWeights = boundary.verticalWeights;
        destination.verticalBoundary.initialWeightLength = boundary.verticalLength;
        destination.verticalBoundary.terminalSize = boundary.verticalTerminalSize;
        std::copy(
            boundary.verticalTerminalMatrix.begin(),
            boundary.verticalTerminalMatrix.end(),
            std::begin(destination.verticalBoundary.terminalMatrix));
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
        msg += " scratch_tier=";
        msg += Spektrafilm::to_cstr(descriptor.scratchTier);
        msg += " target_scratch_tier=";
        msg += Spektrafilm::to_cstr(descriptor.targetScratchTier);
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
        const JuicerCuda::FrameRect& renderWindow,
        const JuicerCuda::FrameRect& sourceBounds,
        const JuicerCuda::FrameRect& fullFrameBounds) {
        if (!descriptor.has_value() || !descriptor->requiresFullFrame) {
            return true;
        }
        const Spektrafilm::VisualGrainFrameExtent& full =
            descriptor->fullFrameExtent;
        const auto matches = [&full](const JuicerCuda::FrameRect& rect) {
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
        const JuicerCuda::FrameRect& renderWindow,
        const JuicerCuda::FrameRect& sourceBounds,
        const JuicerCuda::FrameRect& fullFrameBounds) {
        if (!descriptor.has_value() || !descriptor->requiresFullFrame) {
            return true;
        }
        const Spektrafilm::FilmJuicerEffectsFrameExtent& full =
            descriptor->fullFrameExtent;
        const auto matches = [&full](const JuicerCuda::FrameRect& rect) {
            return rect.x1 == full.x && rect.y1 == full.y &&
                   rect.x2 - rect.x1 == full.width &&
                   rect.y2 - rect.y1 == full.height;
        };
        return matches(renderWindow) && matches(sourceBounds) &&
               matches(fullFrameBounds);
    }

} // namespace

namespace JuicerCuda {
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

    JuicerCuda::AutoExposurePreviewDescriptor make_auto_exposure_preview_descriptor(
        const JuicerCuda::FrameRect& sourceBounds,
        const JuicerCuda::FrameRect& meterBounds,
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

    ResourceManager::DeviceContextKey inspect_frame(
        const unsigned char* srcBase,
        unsigned char* dstBase,
        bool traceInfo) {
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
                throw JuicerCuda::ExecutionFailure{};
            }

            if (deviceId < 0) {
                int cur = -1;
                cudaError_t devErr = cudaGetDevice(&cur);
                if (devErr != cudaSuccess || cur < 0) {
                    JTRACE("CUDA", "FATAL: failed to determine CUDA device for OFX pointers");
                    throw JuicerCuda::ExecutionFailure{};
                }
                deviceId = cur;
            }

            cudaError_t setErr = cudaSetDevice(deviceId);
            if (setErr != cudaSuccess) {
                const char* msg = detail_or_unknown(cudaGetErrorString(setErr));
                trace_cuda_fatal_prefixed_if(
                    traceInfo,
                    CudaFailureTrace{"cudaSetDevice failed", msg});
                throw JuicerCuda::ExecutionFailure{};
            }

            std::string contextError;
            if (!JuicerCuda::query_current_cuda_context(contextOpaque, contextError)) {
                trace_cuda_fatal_prefixed_if(
                    traceInfo,
                    CudaFailureTrace{
                        "failed to capture CUDA context identity",
                        cstr_or_null_if_empty(contextError)});
                throw JuicerCuda::ExecutionFailure{};
            }
        }

        return {deviceId, contextOpaque};
    }

} // namespace JuicerCuda

namespace {
    struct ExecutionErrors {
        JuicerCuda::PendingContextLossRecovery& pendingContextLossRecovery;
        bool traceInfo = false;
        const char* restrictionPrefix = nullptr;
        void mark_context_loss_recovery(const char* stage, cudaError_t error, const std::string& detail) const {
            if (pendingContextLossRecovery.pending) {
                return;
            }
            if (!JuicerCuda::is_cuda_context_loss_signal(error, detail)) {
                return;
            }
            pendingContextLossRecovery.pending = true;
            pendingContextLossRecovery.error = error;
            pendingContextLossRecovery.stage = stage;
            pendingContextLossRecovery.detail = detail;
        }

        [[noreturn]] void fail_policy(const char* failurePrefix, const char* detail) const {
            trace_cuda_fatal_prefixed_if(
                traceInfo,
                CudaFailureTrace{
                    nonempty_cstr_or(failurePrefix, "CUDA render cannot continue"),
                    detail});
            throw JuicerCuda::ExecutionFailure{};
        }

        [[noreturn]] void fail_stage(const char* stageTag, const char* failurePrefix, cudaError_t errorCode) const {
            const char* errorMsg = cudaGetErrorString(errorCode);
            const char* detail = detail_or_unknown(errorMsg);
            mark_context_loss_recovery(nonempty_cstr_or(stageTag, "cuda_stage"), errorCode, detail);
            fail_policy(nonempty_cstr_or(failurePrefix, "CUDA stage failed"), detail);
        }

        [[noreturn]] void fail_submission(const char* stageTag, const char* failurePrefix, const std::string& error) const {
            mark_context_loss_recovery(
                nonempty_cstr_or(stageTag, "submission_stage"),
                cudaErrorUnknown,
                error);
            trace_cuda_fatal_prefixed_if(
                traceInfo,
                CudaFailureTrace{
                    nonempty_cstr_or(failurePrefix, "submission stage failed"),
                    cstr_or_null_if_empty(error)});
#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
            JuicerCuda::ExecutorTest::observe_classification(stageTag, pendingContextLossRecovery.pending);
#endif
            throw JuicerCuda::ExecutionFailure{};
        }

        [[noreturn]] void fail_route(const char* diagnostic) const {
            fail_policy(restrictionPrefix, diagnostic);
        }
    };

    struct RouteDescriptors {
        Scanner::ScannerSpectralLutDescriptor scanner{};
        Scanner::ScannerPostEffectsDescriptor post{};
        Spektrafilm::SpatialDirDescriptor dir{};
        std::optional<Spektrafilm::VisualGrainFrameDescriptor> grain;
        std::optional<Spektrafilm::FilmJuicerEffectsFrameDescriptor> effects;
    };

    struct PrintCorrectionSource {
        const RenderRecipe& recipe;
        const FocusedRenderPayload& payload;
    };

    struct ScheduleInput {
        const JuicerCuda::ExecutionFrame& frame;
        JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot;
        JuicerProcess::Root::PreparationIdentity identity;
        const RouteDescriptors& descriptors;
        JuicerCuda::FilmPayloadInput film;
        float filmRouteCorrectionScale;
        const VisualGrainRecipe& grain;
        const OutputGamutRecipe& outputGamut;
        std::uint64_t scannerBoundsHash;
        std::uint64_t recipeHash;
        bool cameraAutoEnabled;
        Spektrafilm::AutoExposureMethod meteringMethod;
        std::variant<const JuicerProcess::Root::CudaFramePreparationRequest*, const JuicerProcess::Root::PreparedFrameInput*> preparation;
        std::variant<Scanner::ScannerColorCorrectionDescriptor, PrintCorrectionSource> correction;
        PrintExposureRecipe printExposure{};
    };

    struct OpticalWorkspace {
        JuicerProcess::Root::PreparedCudaFrame::CaptureFilmDensityWorkspaceView density{};
        JuicerProcess::Root::PreparedCudaFrame::FocusedRgbWorkspaceView rgb{};
        JuicerProcess::Root::PreparedCudaFrame::ScannerWorkspaceView scanner{};
    };

    struct GrainBinding {
        JuicerProcess::Root::PreparedCudaFrame::VisualGrainWorkspaceView workspace{};
        JuicerCuda::GrainPayload payload{};
        JuicerCuda::GrainKernelPayload kernels{};
    };

    struct EffectsPayloads {
        JuicerCuda::FilmDefectsPayload film{};
        JuicerCuda::GateWeavePayload weave{};
    };

    void copy_scan_density_range_payload(JuicerCuda::ScannerDensityRangePayload& destination, const JuicerCuda::Resources::DeviceScanRange& source) {
        destination.mediumIsNegative = source.mediumIsNegative;
        copy_float3(destination.min_cmy, source.min_cmy);
        copy_float3(destination.inv_max_cmy, source.inv_max_cmy);
    }

    void pack_output_gamut_payload(
        JuicerCuda::ScanColorPayload& destination,
        const JuicerProcess::Root::PreparedCudaFrame::FocusedPreparedView& prepared,
        const OutputGamutRecipe& recipe,
        const ExecutionErrors& errors) {
        destination.outputGamutActive = 0;
        destination.outputGamutCmax = nullptr;
        if (!recipe.enabled) {
            return;
        }
        if (!prepared.outputGamutTransform ||
            !prepared.outputGamutCmax) {
            errors.fail_submission(
                "pack_output_gamut_payload",
                "CUDA output gamut payload binding failed",
                "ResourceDescriptorMismatch component=scan_color_payload field=output_gamut_identity");
        }
        destination.outputGamutCmax = prepared.outputGamutCmax;
        copy_float9(
            destination.outputGamutNativeRgbToD65Xyz,
            prepared.outputGamutTransform->nativeRgbToD65Xyz.data());
        copy_float9(
            destination.outputGamutD65XyzToNativeRgb,
            prepared.outputGamutTransform->d65XyzToNativeRgb.data());
        copy_float9(
            destination.outputGamutOklabXyzToLms,
            Gamut::kOklabXyzToLms.data());
        copy_float9(
            destination.outputGamutOklabLmsToXyz,
            Gamut::kOklabLmsToXyz.data());
        copy_float9(
            destination.outputGamutOklabLmsRootToLab,
            Gamut::kOklabLmsRootToLab.data());
        copy_float9(
            destination.outputGamutOklabLabToLmsRoot,
            Gamut::kOklabLabToLmsRoot.data());
        destination.outputGamutLightnessKnee[0] =
            recipe.lightnessKneeThreshold;
        destination.outputGamutLightnessKnee[1] =
            recipe.lightnessKneeLimit;
        destination.outputGamutLightnessKnee[2] =
            recipe.lightnessKneePower;
        destination.outputGamutChromaKnee[0] =
            recipe.chromaKneeThreshold;
        destination.outputGamutChromaKnee[1] =
            recipe.chromaKneeLimit;
        destination.outputGamutChromaKnee[2] =
            recipe.chromaKneePower;
        destination.outputGamutActive = 1;
    }

    void meter_camera_exposure(
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        JuicerCuda::FilmPayloadPack& filmPayloads,
        const JuicerCuda::ExecutionFrame& frame,
        Spektrafilm::AutoExposureMethod cameraMeteringMethod,
        const char* missingBuffersDiagnostic,
        const ExecutionErrors& errors) {
        const auto& autoExposureDescriptor = frame.autoExposureDescriptor;

        const auto buffers = preparedFrame.auto_exposure_buffers();
        if (!buffers.active) {
            errors.fail_route(missingBuffersDiagnostic);
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
                frame.stream,
                &meterError);
            if (weightsRc != 0) {
                errors.fail_route(detail_or_unknown(meterError));
            }
            preparedFrame.mark_auto_exposure_weights_built(
                JuicerProcess::Root::PreparedCudaFrame::AutoExposureWeightsExtent{
                    autoExposureDescriptor.previewWidth,
                    autoExposureDescriptor.previewHeight});
        }
        JuicerCuda::AutoExposureSourceFormat autoExposureSourceFormat{};
        autoExposureSourceFormat.componentCount = frame.components;
        autoExposureSourceFormat.filmRaw = filmPayloads.filmRaw;
        autoExposureSourceFormat.reconstruction =
            filmPayloads.filmExposure.reconstruction;
        const int meterRc = juicer_cuda_auto_exposure_meter_to_device(
            frame.sourceBase,
            static_cast<std::size_t>(frame.sourceRowBytes),
            autoExposureDescriptor,
            autoExposureSourceFormat,
            buffers.scratch,
            buffers.deviceState,
            frame.stream,
            &meterError);
        if (meterRc != 0) {
            errors.fail_route(detail_or_unknown(meterError));
        }
        preparedFrame.mark_auto_exposure_metered(
            JuicerProcess::Root::PreparedCudaFrame::AutoExposureMeteredResult{
                autoExposureDescriptor.hash});
        filmPayloads.filmExposure.exposureScaleDevice =
            buffers.deviceState.exposureScale;
    }

    RouteDescriptors build_direct_descriptors(
        const RenderRecipe& recipe,
        const FocusedRenderPayload& payload,
        const JuicerCuda::ExecutionFrame& frame,
        Scanner::ScannerColorCorrectionDescriptor& scannerCorrection,
        const ExecutionErrors& errors) {
        RouteDescriptors descriptors;
        const auto* directRecipe = &recipe;
        const auto* directPayload = &payload;
        const auto& win = frame.renderWindow;
        const auto& srcBounds = frame.sourceBounds;
        const int width = win.x2 - win.x1;
        const int height = win.y2 - win.y1;

        std::string scannerDescriptorDiagnostic;
        Scanner::DirectScannerSpectralLutDescriptorInput scannerDescriptorInput{};
        scannerDescriptorInput.profileRoute = &directRecipe->profileRoute;
        scannerDescriptorInput.densityBounds = &directRecipe->densityBounds;
        scannerDescriptorInput.scannerOutput = &directRecipe->scannerOutput;
        if (!Scanner::build_direct_scanner_spectral_lut_descriptor(
                scannerDescriptorInput,
                descriptors.scanner,
                scannerDescriptorDiagnostic)) {
            errors.fail_route(scannerDescriptorDiagnostic.c_str());
        }

        if (!Scanner::build_direct_scanner_color_correction_descriptor(
                *directRecipe,
                directPayload->scannerTables,
                scannerCorrection,
                scannerDescriptorDiagnostic)) {
            errors.fail_route(scannerDescriptorDiagnostic.c_str());
        }

        if (!Scanner::build_scanner_post_effects_descriptor(
                directRecipe->scannerOutput,
                descriptors.post,
                scannerDescriptorDiagnostic)) {
            errors.fail_route(scannerDescriptorDiagnostic.c_str());
        }

        const JuicerCuda::FrameRect directFullFrameRect =
            (frame.fullFrameExtent.x2 > frame.fullFrameExtent.x1 && frame.fullFrameExtent.y2 > frame.fullFrameExtent.y1)
                ? frame.fullFrameExtent
                : srcBounds;
        if (!Spektrafilm::build_spatial_dir_descriptor(
                directRecipe->dirCouplers,
                frame.pixelSizeUm,
                spatial_dir_extent_from_rect(win),
                spatial_dir_extent_from_rect(directFullFrameRect),
                "direct",
                descriptors.dir)) {
            trace_spatial_dir_descriptor_build("direct", descriptors.dir);
            errors.fail_route(
                "ResourceDescriptorMismatch phase=3D-3 field=spatial_dir_descriptor");
        }
        trace_spatial_dir_descriptor_build("direct", descriptors.dir);

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
                frame.pixelSizeUm,
                frame.timeFrames,
                frame.frameRate,
                frame.sessionSeed,
                static_cast<std::uint64_t>(frame.clipToken),
                descriptors.grain,
                directGrainDescriptorDiagnostic)) {
            errors.fail_route(
                directGrainDescriptorDiagnostic.c_str());
        }
        if (!visual_grain_full_frame_preflight(
                descriptors.grain,
                win,
                srcBounds,
                directFullFrameRect)) {
            errors.fail_route(
                "ResourceDescriptorMismatch phase=grain_preflight field=full_frame_extent");
        }

        std::string directEffectsDescriptorDiagnostic;
        if (!build_film_juicer_effects_descriptor_for_frame(
                directRecipe->filmJuicerEffects,
                {win.x1, win.y1, width, height},
                {directFullFrameRect.x1,
                 directFullFrameRect.y1,
                 directFullFrameRect.x2 - directFullFrameRect.x1,
                 directFullFrameRect.y2 - directFullFrameRect.y1},
                frame.effectsGeometry,
                directRecipe->filmRaw.filmFormatLongEdgeMm,
                frame.pixelSizeUm,
                frame.timeFrames,
                frame.frameRate,
                frame.sessionSeed,
                static_cast<std::uint64_t>(frame.clipToken),
                descriptors.effects,
                directEffectsDescriptorDiagnostic)) {
            errors.fail_route(
                directEffectsDescriptorDiagnostic.c_str());
        }
        if (!film_juicer_effects_full_frame_preflight(
                descriptors.effects,
                win,
                srcBounds,
                directFullFrameRect)) {
            errors.fail_route(
                "ResourceDescriptorMismatch phase=effects_preflight field=full_frame_extent");
        }

        return descriptors;
    }

    std::string halation_completion_diagnostic(
        const JuicerProcess::Root::PreparationIdentity& identity,
        const JuicerCuda::ScatterHalationPreparedView& halation,
        const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        const char* boundary,
        const std::string& cudaStatus) {
        const auto& deviceContextKey = snapshot.deviceContextKey;

        std::string diagnostic =
            "ScatterHalationCompletionObservation route=";
        diagnostic += Spektrafilm::scan_route_label(
            identity.scanRoute);
        diagnostic +=
            " domain=FilmLinearExposure component=pipeline film_profile_key=";
        diagnostic += identity.filmProfileKey;
        diagnostic += " film_profile_asset_version_token=";
        diagnostic += std::to_string(
            identity.filmProfileAssetVersionToken);
        diagnostic +=
            " backend=Exact in_flight_descriptor_recipe_hash=";
        diagnostic += std::to_string(
            halation.descriptor
                ? halation.descriptor->recipeHash
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
    }

    void bind_direct_diffusion(
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        const JuicerCuda::ExecutionFrame& frame,
        const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        DiffusionStageBinding& directCameraDiffusion,
        const ExecutionErrors& errors) {
        const auto diffusionPrepared = preparedFrame.diffusion_resources();

        if (frame.diffusionFrameSet) {
            if (!diffusionPrepared.active ||
                diffusionPrepared.executionDescriptor.frameSetHash !=
                    frame.diffusionFrameSet->hash ||
                diffusionPrepared.executionDescriptor.contextEpoch !=
                    snapshot.contextEpoch ||
                diffusionPrepared.spectrumCount == 0 ||
                diffusionPrepared.spectrumCount !=
                    diffusionPrepared.executionDescriptor.uniqueSpectrumCount) {
                errors.fail_route(
                    "MissingRequiredResource component=diffusion field=prepared_view");
            }
            std::string cameraBindingDiagnostic;
            if (!bind_diffusion_stage(
                    *frame.diffusionFrameSet,
                    Spektrafilm::DiffusionLinearStage::CameraFilmLinear,
                    diffusionPrepared,
                    directCameraDiffusion,
                    cameraBindingDiagnostic)) {
                errors.fail_route(cameraBindingDiagnostic.c_str());
            }
        } else if (diffusionPrepared.active) {
            errors.fail_route(
                "ResourceDescriptorMismatch component=diffusion field=disabled_prepared_view");
        }
    }

    JuicerCuda::CameraFilmLinearExposurePlanes expose_direct_camera(
        JuicerCuda::DirectPipelineRunParams& run,
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        const JuicerProcess::Root::PreparationIdentity& identity,
        const JuicerCuda::ExecutionFrame& frame,
        const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        bool cameraDiffusionActive,
        DiffusionStageBinding& directCameraDiffusion,
        JuicerCuda::ScatterHalationPreparedView& directHalation,
        const ExecutionErrors& errors) {
        const auto& deviceContextKey = snapshot.deviceContextKey;
        const bool halationExecutable = directHalation.descriptor != nullptr;
        JuicerCuda::CameraFilmLinearExposurePlanes directCameraFilmLinear{};
        if (cameraDiffusionActive) {
            if (!directCameraDiffusion.active) {
                errors.fail_route(
                    "MissingRequiredResource component=diffusion field=camera_stage_binding");
            }
            preparedFrame.mark_diffusion_work_enqueued();
            directCameraFilmLinear = camera_film_linear_planes(
                directCameraDiffusion.stagePlanes);
            const cudaError_t exposureError =
                juicer_cuda_direct_camera_film_linear_exposure(
                    &run,
                    directCameraFilmLinear,
                    frame.stream);
            if (exposureError != cudaSuccess) {
                errors.fail_stage(
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
                frame.stream);
            const JuicerCuda::Diffusion::LaunchResult diffusionResult =
                JuicerCuda::Diffusion::launch_stage(diffusionLaunch);
            if (!diffusionResult.ok()) {
                if (diffusionResult.api ==
                    JuicerCuda::Diffusion::FailureApi::Cuda) {
                    errors.fail_stage(
                        diffusionResult.stage,
                        "direct camera diffusion launch failed",
                        static_cast<cudaError_t>(diffusionResult.code));
                }
                const std::string diagnostic =
                    diffusion_launch_failure_diagnostic(diffusionResult);
                errors.fail_route(diagnostic.c_str());
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
                    frame.stream);
            if (exposureError != cudaSuccess) {
                errors.fail_stage(
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
                        identity.scanRoute),
                    identity.filmProfileKey,
                    identity.filmProfileAssetVersionToken,
                    deviceContextKey,
                    snapshot.contextEpoch,
                    reinterpret_cast<cudaStream_t>(frame.stream),
                    halationDiagnostic)) {
                errors.fail_submission(
                    "direct_scatter_halation_launch",
                    "direct scatter-halation launch failed",
                    halationDiagnostic);
            }
            // post_halation_camera_film_linear
        }

        return directCameraFilmLinear;
    }

    EffectsPayloads pack_effects_payloads(
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        const std::optional<Spektrafilm::FilmJuicerEffectsFrameDescriptor>& descriptor,
        const ExecutionErrors& errors) {
        EffectsPayloads effects;
        const Spektrafilm::FilmJuicerEffectsFrameDescriptor*
            preparedEffectsDescriptor =
                preparedFrame.film_juicer_effects_descriptor();
        if (descriptor.has_value()) {
            std::string effectsPayloadDiagnostic;
            if (!preparedEffectsDescriptor ||
                !JuicerCuda::pack_film_juicer_effects_payload(
                    *preparedEffectsDescriptor,
                    effects.film,
                    effects.weave,
                    effectsPayloadDiagnostic)) {
                errors.fail_route(
                    effectsPayloadDiagnostic.empty()
                        ? "MissingRequiredResource phase=effects_route field=effects_binding"
                        : effectsPayloadDiagnostic.c_str());
            }
        } else if (preparedEffectsDescriptor) {
            errors.fail_route(
                "ResourceDescriptorMismatch phase=effects_route field=inactive_descriptor");
        }

        return effects;
    }

    OpticalWorkspace bind_direct_optical_workspace(
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& focusedWorkspace,
        void* stream,
        const std::optional<Spektrafilm::FilmJuicerEffectsFrameDescriptor>& descriptor,
        const ExecutionErrors& errors) {
        OpticalWorkspace optical;

        std::string focusedWorkspaceError;
        if (!preparedFrame.stage_optical_workspace(
                focusedWorkspace,
                stream,
                focusedWorkspaceError)) {
            errors.fail_submission(
                "direct_focused_workspace_stage",
                "direct focused workspace staging failed",
                focusedWorkspaceError);
        }
        optical.density =
            preparedFrame.capture_film_density_workspace(
                focusedWorkspace);
        optical.rgb = preparedFrame.focused_rgb_workspace(
            focusedWorkspace);
        if (!optical.density.active || !optical.rgb.active ||
            optical.density.c != optical.rgb.r ||
            optical.density.m != optical.rgb.g ||
            optical.density.y != optical.rgb.b) {
            errors.fail_route(
                "MissingRequiredResource phase=grain_route field=focused_triplet");
        }
        optical.scanner =
            preparedFrame.scanner_workspace(focusedWorkspace);
        if (!optical.scanner.active ||
            (descriptor.has_value() && descriptor->filmDust.slotProbability > 0.0f &&
             !optical.scanner.filmDustTransmittance) ||
            (descriptor.has_value() &&
             descriptor->gateTransmittanceActive &&
             !optical.scanner.hasGateTransmittance)) {
            errors.fail_route(
                "MissingRequiredResource phase=effects_route field=effects_workspace");
        }

        return optical;
    }

    void bind_grain(
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& focusedWorkspace,
        const VisualGrainRecipe& recipe,
        GrainBinding& grain,
        const ExecutionErrors& errors) {
        const auto resources = preparedFrame.visual_grain_resources();
        grain.workspace = preparedFrame.visual_grain_workspace(
            focusedWorkspace);
        std::string grainPayloadDiagnostic;
        if (!resources.active ||
            !grain.workspace.active ||
            !JuicerCuda::pack_visual_grain_payload(
                recipe,
                resources,
                grain.payload,
                grain.kernels,
                grainPayloadDiagnostic)) {
            errors.fail_route(
                grainPayloadDiagnostic.empty()
                    ? "MissingRequiredResource phase=grain_route field=grain_binding"
                    : grainPayloadDiagnostic.c_str());
        }
        grain.payload.frameUniforms =
            grain.workspace.frameUniforms;
    }

    bool build_direct_dir(
        JuicerCuda::DirectPipelineRunParams& run,
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& focusedWorkspace,
        const Spektrafilm::SpatialDirDescriptor& descriptor,
        const JuicerCuda::CameraFilmLinearExposurePlanes& directCameraFilmLinear,
        bool cameraFilmLinearActive,
        void* stream,
        bool directUseFusedScannerPostSpatialDirHandoff,
        const ExecutionErrors& errors) {
        bool directDirUsesSourceBuildCachedLogRaw = false;

        const auto scratch = preparedFrame.spatial_dir_scratch(focusedWorkspace);
        const auto resources =
            preparedFrame.spatial_dir_resources(focusedWorkspace, descriptor.hash);
        if (!scratch.active || !resources.active) {
            errors.fail_route(
                "MissingRequiredResource phase=3D-3 field=prepared_spatial_dir");
        }
        const Spektrafilm::DirScratchPlaneRoles& directDirAdmittedRoles =
            scratch.planeRoles;
        const Spektrafilm::DirScratchPlaneRoles& directDirAdmittedTargetRoles =
            scratch.targetPlaneRoles;
        directDirUsesSourceBuildCachedLogRaw =
            spatial_dir_source_build_retains_cached_log_raw(
                directDirAdmittedRoles,
                directDirAdmittedTargetRoles);
        if (directDirUsesSourceBuildCachedLogRaw &&
            !spatial_dir_required_cached_log_raw_present(
                scratch,
                directDirAdmittedRoles.cachedLogRawPlanes)) {
            errors.fail_route(
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
        const Spektrafilm::DirGaussianComponentPlan& coreComponent =
            spatial_dir_component_or_empty(descriptor, 0);
        dirBuildRequest.gaussian.weight = coreComponent.weight;
        dirBuildRequest.gaussian.filterOperator =
            cuda_dir_filter_operator(coreComponent.referenceOperator);
        bind_spatial_dir_iir(
            dirBuildRequest.gaussian,
            coreComponent,
            resources.boundaries[0]);
        for (int tailIndex = 0; tailIndex < 3; ++tailIndex) {
            dirBuildRequest.tails[tailIndex].kernel = resources.exponential[tailIndex].weights;
            dirBuildRequest.tails[tailIndex].radius = resources.exponential[tailIndex].radius;
            dirBuildRequest.tails[tailIndex].sigma = resources.exponential[tailIndex].sigma;
            const Spektrafilm::DirGaussianComponentPlan& tailComponent =
                spatial_dir_component_or_empty(descriptor, tailIndex + 1);
            dirBuildRequest.tails[tailIndex].weight = tailComponent.weight;
            dirBuildRequest.tails[tailIndex].filterOperator =
                cuda_dir_filter_operator(tailComponent.referenceOperator);
            bind_spatial_dir_iir(
                dirBuildRequest.tails[tailIndex],
                tailComponent,
                resources.boundaries[static_cast<std::size_t>(tailIndex) + 1u]);
        }
        dirBuildRequest.streamOpaque = stream;
        const cudaError_t dirError = juicer_cuda_build_direct_spatial_dir(
            &run,
            dirBuildRequest);
        if (dirError != cudaSuccess) {
            errors.fail_stage(
                "direct_spatial_dir_launch",
                "direct spatial DIR build failed",
                dirError);
        }
        if (!directUseFusedScannerPostSpatialDirHandoff &&
            !directDirUsesSourceBuildCachedLogRaw) {
            std::string stageError;
            if (!preparedFrame.stage_spatial_dir_cached_log_raw_for_final_develop(
                    focusedWorkspace,
                    stream,
                    stageError)) {
                errors.fail_submission(
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
            errors.fail_route(
                "MissingRequiredResource phase=3D-3 field=spatial_dir_filtered_correction");
        }
        if (directDirRequiresCachedLogRaw &&
            !spatial_dir_required_cached_log_raw_present(
                finalScratch,
                directDirAdmittedTargetRoles.cachedLogRawPlanes)) {
            errors.fail_route(
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
                                                      stream)
                                                : juicer_cuda_build_direct_spatial_dir_cached_log_raw(
                                                      &run,
                                                      finalScratch.logRawB,
                                                      finalScratch.logRawG,
                                                      finalScratch.logRawR,
                                                      stream);
            if (logRawError != cudaSuccess) {
                errors.fail_stage(
                    "direct_spatial_dir_cached_log_raw_launch",
                    "direct spatial DIR cached log raw build failed",
                    logRawError);
            }
        }
        bind_spatial_dir_final_develop_to_payload(run.filmDevelop, finalScratch);

        return directDirUsesSourceBuildCachedLogRaw;
    }

    void pack_scan_stage(
        JuicerCuda::ScanStagePayload& scanStage,
        const JuicerProcess::Root::PreparedCudaFrame::FocusedPreparedView& prepared,
        const OutputGamutRecipe& outputGamut,
        const Scanner::ScannerColorCorrectionDescriptor& scannerCorrection,
        const ExecutionErrors& errors) {
        copy_scan_density_range_payload(
            scanStage.densityRange,
            *prepared.scanRange);
        scanStage.scanLutLog2PchipXYZ = prepared.scanLut->log2PchipXYZ;
        scanStage.scanLutPchipSlopeC = prepared.scanLut->slopeC;
        scanStage.scanLutPchipSlopeM = prepared.scanLut->slopeM;
        scanStage.scanLutPchipSlopeY = prepared.scanLut->slopeY;
        scanStage.scanLutPchipCellMin = prepared.scanLut->cellMin;
        scanStage.scanLutPchipCellMax = prepared.scanLut->cellMax;
        scanStage.scanLutRes = static_cast<int>(prepared.scanLut->res);
        const Scanner::ColorRuntime& color = *prepared.scannerColor;
        copy_float9(scanStage.scanColor.cat02, color.cat02);
        copy_float9(scanStage.scanColor.xyzToRgb, color.xyzToRgb);
        copy_float3(scanStage.scanColor.illuminantXYZ, color.illuminantXYZ);
        pack_output_gamut_payload(
            scanStage.scanColor,
            prepared,
            outputGamut,
            errors);
        scanStage.scanColor.encoding.outputColorSpaceIndex = OutputEncoding::toIndex(color.encoding.colorSpace);
        scanStage.scanColor.encoding.applyCctfEncoding = bool_to_i32(color.encoding.applyCctfEncoding);
        scanStage.scanColor.encoding.inputIsOutputSpace = bool_to_i32(color.encoding.inputIsOutputSpace);
        const auto& outputSpace = GeneratedColorSpaces::get(color.encoding.colorSpace);
        scanStage.scanColor.encoding.cctf.kind = static_cast<int>(outputSpace.cctf.kind);
        scanStage.scanColor.encoding.cctf.gamma = outputSpace.cctf.gamma;
        scanStage.scanColor.encoding.cctf.a = outputSpace.cctf.a;
        scanStage.scanColor.encoding.cctf.b = outputSpace.cctf.b;
        scanStage.scanColor.encoding.cctf.c = outputSpace.cctf.c;
        scanStage.scanColor.encoding.cctf.d = outputSpace.cctf.d;
        scanStage.scanColor.encoding.cctf.linearCutoff = outputSpace.cctf.linearCutoff;
        const OutputEncoding::Matrix3x3 dwgToOutput =
            OutputEncoding::dwg_to_output_matrix(color.encoding.colorSpace);
        copy_float9(scanStage.scanColor.encoding.dwgToOutput, dwgToOutput.m);
        scanStage.correctionActive = scannerCorrection.active ? 1 : 0;
        scanStage.correctionSlope = scannerCorrection.xyzSlope;
        scanStage.correctionOffset = scannerCorrection.xyzOffset;
    }

    void develop_direct_capture(
        JuicerCuda::DirectPipelineRunParams& run,
        const JuicerCuda::CameraFilmLinearExposurePlanes& directCameraFilmLinear,
        bool cameraFilmLinearActive,
        const Spektrafilm::SpatialDirDescriptor& descriptor,
        const OpticalWorkspace& optical,
        bool grainStageActive,
        const GrainBinding& grain,
        bool filmEffectsActive,
        const EffectsPayloads& effects,
        bool grainDebugActive,
        void* stream,
        const ExecutionErrors& errors) {
        cudaError_t launchError = cudaSuccess;
        const int width = run.width;
        const int height = run.height;

        launchError = cameraFilmLinearActive
                          ? juicer_cuda_direct_focused_capture_density_from_camera_film_linear(
                                &run,
                                directCameraFilmLinear,
                                optical.density.c,
                                optical.density.m,
                                optical.density.y,
                                stream)
                          : juicer_cuda_direct_focused_capture_density(
                                &run,
                                optical.density.c,
                                optical.density.m,
                                optical.density.y,
                                stream);
        if (launchError != cudaSuccess) {
            errors.fail_stage(
                "direct_capture_density_launch",
                "direct capture-film density launch failed",
                launchError);
        }

        if (descriptor.hash != 0) {
            run.filmDevelop.spatialDir.logRawB = nullptr;
            run.filmDevelop.spatialDir.logRawG = nullptr;
            run.filmDevelop.spatialDir.logRawR = nullptr;
        }
        if (grainStageActive) {
            launchError = juicer_cuda_apply_visual_grain(
                &grain.payload,
                &grain.kernels,
                width,
                height,
                optical.density.c,
                optical.density.m,
                optical.density.y,
                grain.workspace.filterTemp,
                grain.workspace.scaleWork,
                grain.workspace.deltaAccum,
                grain.workspace.layerWork,
                grain.workspace.sharedDelta,
                stream);
            if (launchError != cudaSuccess) {
                errors.fail_stage(
                    "direct_visual_grain_launch",
                    "direct visual grain launch failed",
                    launchError);
            }
        }
        if (filmEffectsActive && !grainDebugActive) {
            launchError = juicer_cuda_apply_film_defects(
                &effects.film,
                width,
                height,
                optical.density.c,
                optical.density.m,
                optical.density.y,
                optical.scanner.filmDustTransmittance,
                stream);
            if (launchError != cudaSuccess) {
                errors.fail_stage(
                    "direct_film_effects_launch",
                    "direct film defects launch failed",
                    launchError);
            }
        }
    }

    void scan_direct_output(
        JuicerCuda::DirectPipelineRunParams& run,
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& focusedWorkspace,
        const RouteDescriptors& descriptors,
        const OpticalWorkspace& optical,
        const EffectsPayloads& effects,
        bool directUseFocusedSplit,
        bool grainDebugActive,
        bool directUseFusedScannerPostSpatialDirHandoff,
        bool directCaptureDensityReady,
        bool directDirUsesSourceBuildCachedLogRaw,
        bool gateOutputActive,
        const JuicerCuda::ExecutionFrame& frame,
        const ExecutionErrors& errors) {
        cudaError_t launchError = cudaSuccess;

        if (directUseFocusedSplit) {
            JuicerProcess::Root::PreparedCudaFrame::ScannerPostEffectsPreparedView
                post{};
            if (descriptors.post.active() && !grainDebugActive) {
                post = preparedFrame.scanner_post_effects_resources(
                    focusedWorkspace,
                    descriptors.post.hash);
                if (!post.active ||
                    post.scratch.rgbR != optical.rgb.r ||
                    post.scratch.rgbG != optical.rgb.g ||
                    post.scratch.rgbB != optical.rgb.b) {
                    errors.fail_route(
                        "MissingRequiredResource phase=8C field=scanner_post_effects");
                }
            }
            if (!grainDebugActive) {
                if (directUseFusedScannerPostSpatialDirHandoff) {
                    launchError = juicer_cuda_direct_focused_scan_linear_rgb(
                        &run,
                        optical.rgb.r,
                        optical.rgb.g,
                        optical.rgb.b,
                        frame.stream);
                } else {
                    if (!directCaptureDensityReady) {
                        errors.fail_route(
                            "MissingRequiredResource phase=grain_route field=capture_density");
                    }
                    launchError = juicer_cuda_direct_focused_scan_linear_density_rgb(
                        &run,
                        optical.density.c,
                        optical.density.m,
                        optical.density.y,
                        optical.rgb.r,
                        optical.rgb.g,
                        optical.rgb.b,
                        optical.scanner.filmDustTransmittance,
                        frame.stream);
                }
                if (launchError != cudaSuccess) {
                    errors.fail_stage(
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
                optical.rgb.r,
                optical.rgb.g,
                optical.rgb.b,
                post.active ? post.scratch.tmp : nullptr,
                post.active ? post.lensBlur.weights : nullptr,
                post.active ? post.lensBlur.radius : 0,
                post.active ? post.unsharp.weights : nullptr,
                post.active ? post.unsharp.radius : 0,
                post.active ? descriptors.post.unsharpAmount : 0.0f,
                gateOutputActive && !grainDebugActive
                    ? &effects.film
                    : nullptr,
                gateOutputActive && !grainDebugActive
                    ? &effects.weave
                    : nullptr,
                gateOutputActive && !grainDebugActive &&
                        descriptors.effects->gateTransmittanceActive
                    ? optical.scanner.gateTransmittance
                    : nullptr,
                gateOutputActive && !grainDebugActive &&
                        descriptors.effects->gateTransmittanceActive
                    ? optical.scanner.gateTransmittanceWidth
                    : 0,
                gateOutputActive && !grainDebugActive &&
                        descriptors.effects->gateTransmittanceActive
                    ? optical.scanner.gateTransmittanceHeight
                    : 0,
                frame.stream);
            // Focused RGB can alias the spatial-DIR filtered planes, so scanner output is their final consumer.
            if (launchError == cudaSuccess && descriptors.dir.hash != 0) {
                clear_spatial_dir_final_develop_payload_bindings(run.filmDevelop);
            }
        } else {
            launchError = juicer_cuda_negative_direct_pipeline(&run, frame.stream);
        }
        if (launchError != cudaSuccess) {
            errors.fail_stage("direct_negative_pipeline_launch", "direct negative pipeline launch failed", launchError);
        }
        if (descriptors.dir.hash != 0 && !directUseFocusedSplit) {
            run.filmDevelop.spatialDir.logRawB = nullptr;
            run.filmDevelop.spatialDir.logRawG = nullptr;
            run.filmDevelop.spatialDir.logRawR = nullptr;
        }
    }

    void build_direct_gate_mask(const EffectsPayloads& effects, const OpticalWorkspace& optical, void* stream, const ExecutionErrors& errors) {
        cudaError_t launchError = cudaSuccess;

        launchError = juicer_cuda_build_gate_defect_transmittance_focused(
            &effects.film,
            optical.scanner.gateTransmittance,
            optical.scanner.gateTransmittanceWidth,
            optical.scanner.gateTransmittanceHeight,
            stream);
        if (launchError != cudaSuccess) {
            errors.fail_stage(
                "direct_gate_transmittance_launch",
                "direct gate defect mask launch failed",
                launchError);
        }
    }

    RouteDescriptors build_print_descriptors(const RenderRecipe& recipe, const JuicerCuda::ExecutionFrame& frame, const ExecutionErrors& errors) {
        RouteDescriptors descriptors;
        const auto* printRecipe = &recipe;
        const auto& win = frame.renderWindow;
        const auto& srcBounds = frame.sourceBounds;
        const int width = win.x2 - win.x1;
        const int height = win.y2 - win.y1;

        std::string scannerDescriptorDiagnostic;
        if (!Scanner::build_print_scanner_spectral_lut_descriptor(
                Scanner::PrintScannerSpectralLutDescriptorInput{
                    &printRecipe->profileRoute,
                    &printRecipe->densityBounds,
                    &printRecipe->scannerOutput},
                descriptors.scanner,
                scannerDescriptorDiagnostic)) {
            errors.fail_route(scannerDescriptorDiagnostic.c_str());
        }

        if (!Scanner::build_scanner_post_effects_descriptor(
                printRecipe->scannerOutput,
                descriptors.post,
                scannerDescriptorDiagnostic)) {
            errors.fail_route(scannerDescriptorDiagnostic.c_str());
        }

        const JuicerCuda::FrameRect printFullFrameRect =
            (frame.fullFrameExtent.x2 > frame.fullFrameExtent.x1 && frame.fullFrameExtent.y2 > frame.fullFrameExtent.y1)
                ? frame.fullFrameExtent
                : srcBounds;
        if (!Spektrafilm::build_spatial_dir_descriptor(
                printRecipe->dirCouplers,
                frame.pixelSizeUm,
                spatial_dir_extent_from_rect(win),
                spatial_dir_extent_from_rect(printFullFrameRect),
                "print",
                descriptors.dir)) {
            trace_spatial_dir_descriptor_build("print", descriptors.dir);
            errors.fail_route(
                "ResourceDescriptorMismatch phase=4C field=spatial_dir_descriptor");
        }
        trace_spatial_dir_descriptor_build("print", descriptors.dir);

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
                frame.pixelSizeUm,
                frame.timeFrames,
                frame.frameRate,
                frame.sessionSeed,
                static_cast<std::uint64_t>(frame.clipToken),
                descriptors.grain,
                printGrainDescriptorDiagnostic)) {
            errors.fail_route(
                printGrainDescriptorDiagnostic.c_str());
        }
        if (!visual_grain_full_frame_preflight(
                descriptors.grain,
                win,
                srcBounds,
                printFullFrameRect)) {
            errors.fail_route(
                "ResourceDescriptorMismatch phase=grain_preflight field=full_frame_extent");
        }

        std::string printEffectsDescriptorDiagnostic;
        if (!build_film_juicer_effects_descriptor_for_frame(
                printRecipe->filmJuicerEffects,
                {win.x1, win.y1, width, height},
                {printFullFrameRect.x1,
                 printFullFrameRect.y1,
                 printFullFrameRect.x2 - printFullFrameRect.x1,
                 printFullFrameRect.y2 - printFullFrameRect.y1},
                frame.effectsGeometry,
                printRecipe->filmRaw.filmFormatLongEdgeMm,
                frame.pixelSizeUm,
                frame.timeFrames,
                frame.frameRate,
                frame.sessionSeed,
                static_cast<std::uint64_t>(frame.clipToken),
                descriptors.effects,
                printEffectsDescriptorDiagnostic)) {
            errors.fail_route(
                printEffectsDescriptorDiagnostic.c_str());
        }
        if (!film_juicer_effects_full_frame_preflight(
                descriptors.effects,
                win,
                srcBounds,
                printFullFrameRect)) {
            errors.fail_route(
                "ResourceDescriptorMismatch phase=effects_preflight field=full_frame_extent");
        }

        return descriptors;
    }

    void bind_print_diffusion(
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        const JuicerCuda::ExecutionFrame& frame,
        const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        DiffusionStageBinding& printCameraDiffusion,
        DiffusionStageBinding& printEnlargerDiffusion,
        const ExecutionErrors& errors) {
        const auto diffusionPrepared = preparedFrame.diffusion_resources();

        if (frame.diffusionFrameSet) {
            if (!diffusionPrepared.active ||
                diffusionPrepared.executionDescriptor.frameSetHash !=
                    frame.diffusionFrameSet->hash ||
                diffusionPrepared.executionDescriptor.contextEpoch !=
                    snapshot.contextEpoch ||
                diffusionPrepared.spectrumCount == 0 ||
                diffusionPrepared.spectrumCount !=
                    diffusionPrepared.executionDescriptor.uniqueSpectrumCount) {
                errors.fail_route(
                    "MissingRequiredResource component=diffusion field=prepared_view");
            }
            if (frame.diffusionFrameSet->camera) {
                std::string cameraBindingDiagnostic;
                if (!bind_diffusion_stage(
                        *frame.diffusionFrameSet,
                        Spektrafilm::DiffusionLinearStage::CameraFilmLinear,
                        diffusionPrepared,
                        printCameraDiffusion,
                        cameraBindingDiagnostic)) {
                    errors.fail_route(
                        cameraBindingDiagnostic.c_str());
                }
            }
            if (frame.diffusionFrameSet->enlarger) {
                std::string enlargerBindingDiagnostic;
                if (!bind_diffusion_stage(
                        *frame.diffusionFrameSet,
                        Spektrafilm::DiffusionLinearStage::EnlargerPrintLinear,
                        diffusionPrepared,
                        printEnlargerDiffusion,
                        enlargerBindingDiagnostic)) {
                    errors.fail_route(
                        enlargerBindingDiagnostic.c_str());
                }
            }
        } else if (diffusionPrepared.active) {
            errors.fail_route(
                "ResourceDescriptorMismatch component=diffusion field=disabled_prepared_view");
        }
    }

    Scanner::ScannerColorCorrectionDescriptor build_print_correction(
        const RenderRecipe& recipe,
        const FocusedRenderPayload& payload,
        const JuicerProcess::Root::PreparedCudaFrame::PrintPreparedView& preparedPrint,
        bool traceVerbose,
        const ExecutionErrors& errors) {
        const auto* printRecipe = &recipe;
        const auto* printPayload = &payload;
        std::string scannerDescriptorDiagnostic;
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
            errors.fail_route(scannerDescriptorDiagnostic.c_str());
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

        return scannerCorrection;
    }

    JuicerCuda::CameraFilmLinearExposurePlanes expose_print_camera(
        JuicerCuda::PrintPipelineRunParams& run,
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        const JuicerProcess::Root::PreparationIdentity& identity,
        const JuicerCuda::ExecutionFrame& frame,
        const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        bool cameraDiffusionActive,
        DiffusionStageBinding& printCameraDiffusion,
        JuicerCuda::ScatterHalationPreparedView& printHalation,
        const ExecutionErrors& errors) {
        const auto& deviceContextKey = snapshot.deviceContextKey;
        const bool halationExecutable = printHalation.descriptor != nullptr;
        JuicerCuda::CameraFilmLinearExposurePlanes printCameraFilmLinear{};
        if (cameraDiffusionActive) {
            if (!printCameraDiffusion.active) {
                errors.fail_route(
                    "MissingRequiredResource component=diffusion field=camera_stage_binding");
            }
            preparedFrame.mark_diffusion_work_enqueued();
            printCameraFilmLinear = camera_film_linear_planes(
                printCameraDiffusion.stagePlanes);
            const cudaError_t exposureError =
                juicer_cuda_print_camera_film_linear_exposure(
                    &run,
                    printCameraFilmLinear,
                    frame.stream);
            if (exposureError != cudaSuccess) {
                errors.fail_stage(
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
                frame.stream);
            const JuicerCuda::Diffusion::LaunchResult diffusionResult =
                JuicerCuda::Diffusion::launch_stage(diffusionLaunch);
            if (!diffusionResult.ok()) {
                if (diffusionResult.api ==
                    JuicerCuda::Diffusion::FailureApi::Cuda) {
                    errors.fail_stage(
                        diffusionResult.stage,
                        "print camera diffusion launch failed",
                        static_cast<cudaError_t>(diffusionResult.code));
                }
                const std::string diagnostic =
                    diffusion_launch_failure_diagnostic(diffusionResult);
                errors.fail_route(diagnostic.c_str());
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
                    frame.stream);
            if (exposureError != cudaSuccess) {
                errors.fail_stage(
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
                        identity.scanRoute),
                    identity.filmProfileKey,
                    identity.filmProfileAssetVersionToken,
                    deviceContextKey,
                    snapshot.contextEpoch,
                    reinterpret_cast<cudaStream_t>(frame.stream),
                    halationDiagnostic)) {
                errors.fail_submission(
                    "print_scatter_halation_launch",
                    "print scatter-halation launch failed",
                    halationDiagnostic);
            }
            // post_halation_camera_film_linear
        }

        return printCameraFilmLinear;
    }

    OpticalWorkspace bind_print_optical_workspace(
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& focusedWorkspace,
        void* stream,
        const std::optional<Spektrafilm::FilmJuicerEffectsFrameDescriptor>& descriptor,
        const ExecutionErrors& errors) {
        OpticalWorkspace optical;

        std::string focusedWorkspaceError;
        if (!preparedFrame.stage_optical_workspace(
                focusedWorkspace,
                stream,
                focusedWorkspaceError)) {
            errors.fail_submission(
                "print_focused_workspace_stage",
                "print focused workspace staging failed",
                focusedWorkspaceError);
        }
        optical.density =
            preparedFrame.capture_film_density_workspace(
                focusedWorkspace);
        optical.rgb = preparedFrame.focused_rgb_workspace(
            focusedWorkspace);
        if (!optical.density.active || !optical.rgb.active ||
            optical.density.c != optical.rgb.r ||
            optical.density.m != optical.rgb.g ||
            optical.density.y != optical.rgb.b) {
            errors.fail_route(
                "MissingRequiredResource phase=grain_route field=focused_triplet");
        }
        optical.scanner =
            preparedFrame.scanner_workspace(focusedWorkspace);
        if (!optical.scanner.active ||
            (descriptor.has_value() && descriptor->filmDust.slotProbability > 0.0f &&
             !optical.scanner.filmDustTransmittance) ||
            (descriptor.has_value() &&
             descriptor->gateTransmittanceActive &&
             !optical.scanner.hasGateTransmittance)) {
            errors.fail_route(
                "MissingRequiredResource phase=effects_route field=effects_workspace");
        }

        return optical;
    }

    bool build_print_dir(
        JuicerCuda::PrintPipelineRunParams& run,
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& focusedWorkspace,
        const Spektrafilm::SpatialDirDescriptor& descriptor,
        const JuicerCuda::CameraFilmLinearExposurePlanes& printCameraFilmLinear,
        bool cameraFilmLinearActive,
        void* stream,
        bool printUseFusedScannerPostSpatialDirHandoff,
        const ExecutionErrors& errors) {
        bool printDirUsesSourceBuildCachedLogRaw = false;

        const auto scratch = preparedFrame.spatial_dir_scratch(focusedWorkspace);
        const auto resources =
            preparedFrame.spatial_dir_resources(focusedWorkspace, descriptor.hash);
        if (!scratch.active || !resources.active) {
            errors.fail_route(
                "MissingRequiredResource phase=4C field=prepared_spatial_dir");
        }
        const Spektrafilm::DirScratchPlaneRoles& printDirAdmittedRoles =
            scratch.planeRoles;
        const Spektrafilm::DirScratchPlaneRoles& printDirAdmittedTargetRoles =
            scratch.targetPlaneRoles;
        printDirUsesSourceBuildCachedLogRaw =
            spatial_dir_source_build_retains_cached_log_raw(
                printDirAdmittedRoles,
                printDirAdmittedTargetRoles);
        if (printDirUsesSourceBuildCachedLogRaw &&
            !spatial_dir_required_cached_log_raw_present(
                scratch,
                printDirAdmittedRoles.cachedLogRawPlanes)) {
            errors.fail_route(
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
        const Spektrafilm::DirGaussianComponentPlan& coreComponent =
            spatial_dir_component_or_empty(descriptor, 0);
        dirBuildRequest.gaussian.weight = coreComponent.weight;
        dirBuildRequest.gaussian.filterOperator =
            cuda_dir_filter_operator(coreComponent.referenceOperator);
        bind_spatial_dir_iir(
            dirBuildRequest.gaussian,
            coreComponent,
            resources.boundaries[0]);
        for (int tailIndex = 0; tailIndex < 3; ++tailIndex) {
            dirBuildRequest.tails[tailIndex].kernel = resources.exponential[tailIndex].weights;
            dirBuildRequest.tails[tailIndex].radius = resources.exponential[tailIndex].radius;
            dirBuildRequest.tails[tailIndex].sigma = resources.exponential[tailIndex].sigma;
            const Spektrafilm::DirGaussianComponentPlan& tailComponent =
                spatial_dir_component_or_empty(descriptor, tailIndex + 1);
            dirBuildRequest.tails[tailIndex].weight = tailComponent.weight;
            dirBuildRequest.tails[tailIndex].filterOperator =
                cuda_dir_filter_operator(tailComponent.referenceOperator);
            bind_spatial_dir_iir(
                dirBuildRequest.tails[tailIndex],
                tailComponent,
                resources.boundaries[static_cast<std::size_t>(tailIndex) + 1u]);
        }
        dirBuildRequest.streamOpaque = stream;
        const cudaError_t dirError = juicer_cuda_build_print_spatial_dir(
            &run,
            dirBuildRequest);
        if (dirError != cudaSuccess) {
            errors.fail_stage(
                "print_spatial_dir_launch",
                "print spatial DIR build failed",
                dirError);
        }
        if (!printUseFusedScannerPostSpatialDirHandoff &&
            !printDirUsesSourceBuildCachedLogRaw) {
            std::string stageError;
            if (!preparedFrame.stage_spatial_dir_cached_log_raw_for_final_develop(
                    focusedWorkspace,
                    stream,
                    stageError)) {
                errors.fail_submission(
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
            errors.fail_route(
                "MissingRequiredResource phase=4C field=spatial_dir_filtered_correction");
        }
        if (printDirRequiresCachedLogRaw &&
            !spatial_dir_required_cached_log_raw_present(
                finalScratch,
                printDirAdmittedTargetRoles.cachedLogRawPlanes)) {
            errors.fail_route(
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
                                                      stream)
                                                : juicer_cuda_build_print_spatial_dir_cached_log_raw(
                                                      &run,
                                                      finalScratch.logRawB,
                                                      finalScratch.logRawG,
                                                      finalScratch.logRawR,
                                                      stream);
            if (logRawError != cudaSuccess) {
                errors.fail_stage(
                    "print_spatial_dir_cached_log_raw_launch",
                    "print spatial DIR cached log raw build failed",
                    logRawError);
            }
        }
        bind_spatial_dir_final_develop_to_payload(run.filmDevelop, finalScratch);

        return printDirUsesSourceBuildCachedLogRaw;
    }

    void develop_print_capture(
        JuicerCuda::PrintPipelineRunParams& run,
        const JuicerCuda::CameraFilmLinearExposurePlanes& printCameraFilmLinear,
        bool cameraFilmLinearActive,
        const Spektrafilm::SpatialDirDescriptor& descriptor,
        const OpticalWorkspace& optical,
        bool grainStageActive,
        const GrainBinding& grain,
        bool filmEffectsActive,
        const EffectsPayloads& effects,
        bool grainDebugActive,
        void* stream,
        const ExecutionErrors& errors) {
        cudaError_t launchError = cudaSuccess;
        const int width = run.width;
        const int height = run.height;

        launchError = cameraFilmLinearActive
                          ? juicer_cuda_print_focused_capture_density_from_camera_film_linear(
                                &run,
                                printCameraFilmLinear,
                                optical.density.c,
                                optical.density.m,
                                optical.density.y,
                                stream)
                          : juicer_cuda_print_focused_capture_density(
                                &run,
                                optical.density.c,
                                optical.density.m,
                                optical.density.y,
                                stream);
        if (launchError != cudaSuccess) {
            errors.fail_stage(
                "print_capture_density_launch",
                "print capture-film density launch failed",
                launchError);
        }

        if (descriptor.hash != 0) {
            run.filmDevelop.spatialDir.logRawB = nullptr;
            run.filmDevelop.spatialDir.logRawG = nullptr;
            run.filmDevelop.spatialDir.logRawR = nullptr;
        }
        if (grainStageActive) {
            launchError = juicer_cuda_apply_visual_grain(
                &grain.payload,
                &grain.kernels,
                width,
                height,
                optical.density.c,
                optical.density.m,
                optical.density.y,
                grain.workspace.filterTemp,
                grain.workspace.scaleWork,
                grain.workspace.deltaAccum,
                grain.workspace.layerWork,
                grain.workspace.sharedDelta,
                stream);
            if (launchError != cudaSuccess) {
                errors.fail_stage(
                    "print_visual_grain_launch",
                    "print visual grain launch failed",
                    launchError);
            }
        }
        if (filmEffectsActive && !grainDebugActive) {
            launchError = juicer_cuda_apply_film_defects(
                &effects.film,
                width,
                height,
                optical.density.c,
                optical.density.m,
                optical.density.y,
                optical.scanner.filmDustTransmittance,
                stream);
            if (launchError != cudaSuccess) {
                errors.fail_stage(
                    "print_film_effects_launch",
                    "print film defects launch failed",
                    launchError);
            }
        }
    }

    void develop_print_medium(
        JuicerCuda::PrintPipelineRunParams& run,
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        DiffusionStageBinding& printEnlargerDiffusion,
        bool enlargerDiffusionActive,
        const OpticalWorkspace& optical,
        void* stream,
        const ExecutionErrors& errors) {
        cudaError_t launchError = cudaSuccess;

        if (enlargerDiffusionActive) {
            if (!printEnlargerDiffusion.active) {
                errors.fail_route(
                    "MissingRequiredResource component=diffusion field=enlarger_stage_binding");
            }
            preparedFrame.mark_diffusion_work_enqueued();
            JuicerCuda::EnlargerPrintLinearExposurePlanes
                printEnlargerLinear = enlarger_print_linear_planes(
                    printEnlargerDiffusion.stagePlanes);
            launchError =
                juicer_cuda_print_focused_enlarger_linear_exposure(
                    &run,
                    optical.density.c,
                    optical.density.m,
                    optical.density.y,
                    printEnlargerLinear,
                    optical.scanner.filmDustTransmittance,
                    stream);
            if (launchError != cudaSuccess) {
                errors.fail_stage(
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
                reinterpret_cast<cudaStream_t>(stream);
            const JuicerCuda::Diffusion::LaunchResult
                diffusionResult =
                    JuicerCuda::Diffusion::launch_stage(
                        diffusionLaunch);
            if (!diffusionResult.ok()) {
                if (diffusionResult.api ==
                    JuicerCuda::Diffusion::FailureApi::Cuda) {
                    errors.fail_stage(
                        diffusionResult.stage,
                        "print enlarger diffusion launch failed",
                        static_cast<cudaError_t>(
                            diffusionResult.code));
                }
                const std::string diagnostic =
                    diffusion_launch_failure_diagnostic(
                        diffusionResult);
                errors.fail_route(diagnostic.c_str());
            }
            // launch_stage rotates semantic plane pointers after each channel.
            printEnlargerLinear = enlarger_print_linear_planes(
                printEnlargerDiffusion.stagePlanes);
            launchError =
                juicer_cuda_print_focused_develop_from_enlarger_linear(
                    &run,
                    printEnlargerLinear,
                    optical.density.c,
                    optical.density.m,
                    optical.density.y,
                    stream);
            if (launchError != cudaSuccess) {
                errors.fail_stage(
                    "print_develop_from_enlarger_linear_launch",
                    "print development from enlarger-linear exposure failed",
                    launchError);
            }
        } else {
            launchError =
                juicer_cuda_print_focused_continue_from_capture_density(
                    &run,
                    optical.density.c,
                    optical.density.m,
                    optical.density.y,
                    optical.scanner.filmDustTransmittance,
                    stream);
            if (launchError != cudaSuccess) {
                errors.fail_stage(
                    "print_continuation_from_capture_density_launch",
                    "print continuation from capture density launch failed",
                    launchError);
            }
        }
    }

    void scan_print_output(
        JuicerCuda::PrintPipelineRunParams& run,
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& focusedWorkspace,
        const RouteDescriptors& descriptors,
        const OpticalWorkspace& optical,
        const EffectsPayloads& effects,
        bool printUseFocusedSplit,
        bool grainDebugActive,
        bool printUseFusedScannerPostSpatialDirHandoff,
        bool printCaptureDensityReady,
        bool printDirUsesSourceBuildCachedLogRaw,
        bool gateOutputActive,
        const JuicerCuda::ExecutionFrame& frame,
        std::uint64_t recipeHash,
        const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        const ExecutionErrors& errors) {
        cudaError_t launchError = cudaSuccess;
        const auto& srcBounds = frame.sourceBounds;
        if (printUseFocusedSplit) {
            JuicerProcess::Root::PreparedCudaFrame::ScannerPostEffectsPreparedView
                post{};
            if (descriptors.post.active() && !grainDebugActive) {
                post = preparedFrame.scanner_post_effects_resources(
                    focusedWorkspace,
                    descriptors.post.hash);
                if (!post.active ||
                    post.scratch.rgbR != optical.rgb.r ||
                    post.scratch.rgbG != optical.rgb.g ||
                    post.scratch.rgbB != optical.rgb.b) {
                    errors.fail_route(
                        "MissingRequiredResource phase=8C field=scanner_post_effects");
                }
            }
            const std::uint64_t glareSeed = Hash::hash_uint64_values(
                {recipeHash,
                 snapshot.frameToken.value,
                 static_cast<std::uint64_t>(srcBounds.x1),
                 static_cast<std::uint64_t>(srcBounds.y1)});
            if (!grainDebugActive) {
                if (printUseFusedScannerPostSpatialDirHandoff) {
                    launchError = juicer_cuda_print_focused_scan_linear_rgb(
                        &run,
                        optical.rgb.r,
                        optical.rgb.g,
                        optical.rgb.b,
                        post.scratch.tmp,
                        post.scratch.blurred,
                        srcBounds.x1,
                        srcBounds.y1,
                        glareSeed,
                        descriptors.post.glarePercent,
                        descriptors.post.glareRoughness,
                        post.glare.weights,
                        post.glare.radius,
                        frame.stream);
                } else {
                    if (!printCaptureDensityReady) {
                        errors.fail_route(
                            "MissingRequiredResource phase=grain_route field=capture_density");
                    }
                    launchError = juicer_cuda_print_focused_scan_linear_density_rgb(
                        &run,
                        optical.density.c,
                        optical.density.m,
                        optical.density.y,
                        optical.rgb.r,
                        optical.rgb.g,
                        optical.rgb.b,
                        post.active ? post.scratch.tmp : nullptr,
                        post.active ? post.scratch.blurred : nullptr,
                        srcBounds.x1,
                        srcBounds.y1,
                        glareSeed,
                        post.active ? descriptors.post.glarePercent : 0.0f,
                        post.active ? descriptors.post.glareRoughness : 0.0f,
                        post.active ? post.glare.weights : nullptr,
                        post.active ? post.glare.radius : 0,
                        frame.stream);
                }
                if (launchError != cudaSuccess) {
                    errors.fail_stage(
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
                optical.rgb.r,
                optical.rgb.g,
                optical.rgb.b,
                post.active ? post.scratch.tmp : nullptr,
                post.active ? post.lensBlur.weights : nullptr,
                post.active ? post.lensBlur.radius : 0,
                post.active ? post.unsharp.weights : nullptr,
                post.active ? post.unsharp.radius : 0,
                post.active ? descriptors.post.unsharpAmount : 0.0f,
                gateOutputActive && !grainDebugActive
                    ? &effects.film
                    : nullptr,
                gateOutputActive && !grainDebugActive
                    ? &effects.weave
                    : nullptr,
                gateOutputActive && !grainDebugActive &&
                        descriptors.effects->gateTransmittanceActive
                    ? optical.scanner.gateTransmittance
                    : nullptr,
                gateOutputActive && !grainDebugActive &&
                        descriptors.effects->gateTransmittanceActive
                    ? optical.scanner.gateTransmittanceWidth
                    : 0,
                gateOutputActive && !grainDebugActive &&
                        descriptors.effects->gateTransmittanceActive
                    ? optical.scanner.gateTransmittanceHeight
                    : 0,
                frame.stream);
            // Focused RGB can alias the spatial-DIR filtered planes, so scanner output is their final consumer.
            if (launchError == cudaSuccess && descriptors.dir.hash != 0) {
                clear_spatial_dir_final_develop_payload_bindings(run.filmDevelop);
            }
        } else {
            launchError = juicer_cuda_print_focused_pipeline(&run, frame.stream);
        }
        if (launchError != cudaSuccess) {
            errors.fail_stage(
                "print_pipeline_launch",
                "focused print pipeline launch failed",
                launchError);
        }
        if (descriptors.dir.hash != 0 && !printUseFocusedSplit) {
            run.filmDevelop.spatialDir.logRawB = nullptr;
            run.filmDevelop.spatialDir.logRawG = nullptr;
            run.filmDevelop.spatialDir.logRawR = nullptr;
        }
    }

    void build_print_gate_mask(const EffectsPayloads& effects, const OpticalWorkspace& optical, void* stream, const ExecutionErrors& errors) {
        cudaError_t launchError = cudaSuccess;

        launchError = juicer_cuda_build_gate_defect_transmittance_focused(
            &effects.film,
            optical.scanner.gateTransmittance,
            optical.scanner.gateTransmittanceWidth,
            optical.scanner.gateTransmittanceHeight,
            stream);
        if (launchError != cudaSuccess) {
            errors.fail_stage(
                "print_gate_transmittance_launch",
                "print gate defect mask launch failed",
                launchError);
        }
    }

} // namespace

namespace JuicerCuda {

    static void execute_direct_schedule(const ScheduleInput& input, PendingContextLossRecovery& recovery, const DirFailureMessage& dirFailureMessage) {
        const auto& frame = input.frame;
        auto& snapshot = input.snapshot;
        const auto& deviceContextKey = snapshot.deviceContextKey;
        const auto& win = frame.renderWindow;
        const int width = win.x2 - win.x1;
        const int height = win.y2 - win.y1;
        const auto& autoExposureDescriptor = frame.autoExposureDescriptor;
        const bool cameraAutoEnabled = input.cameraAutoEnabled;
        const auto cameraMeteringMethod = input.meteringMethod;
        const auto& descriptors = input.descriptors;
        const ExecutionErrors errors{recovery, frame.traceInfo, "CUDA direct route blocked"};
        JuicerProcess::Root::AutoExposureBufferRequest autoExposureBufferRequest{};
        autoExposureBufferRequest.enabled =
            cameraAutoEnabled &&
            (frame.components == 3 || frame.components == 4) &&
            autoExposureDescriptor.previewWidth > 0 &&
            autoExposureDescriptor.previewHeight > 0;
        autoExposureBufferRequest.descriptor = autoExposureDescriptor;
        const bool grainStageActive =
            descriptors.grain.has_value();
        const bool grainDebugActive =
            grainStageActive && input.grain.debugView != 0;
        const bool filmEffectsActive =
            descriptors.effects.has_value() &&
            descriptors.effects->filmActive;
        const bool gateOutputActive =
            descriptors.effects.has_value() &&
            descriptors.effects->gateOutputActive;
        const bool cameraDiffusionActive =
            frame.diffusionFrameSet.has_value() &&
            frame.diffusionFrameSet->camera.has_value();
        const bool captureDensityConsumerActive =
            grainStageActive || filmEffectsActive;
        const auto scannerCorrection = std::get<Scanner::ScannerColorCorrectionDescriptor>(input.correction);
        std::string directPrepareError;
        JuicerProcess::Root::PreparedCudaFrame preparedFrame =
            std::visit([&](const auto* preparation) {
                return JuicerProcess::root().prepare_cuda_frame(deviceContextKey, snapshot, *preparation, autoExposureBufferRequest, frame.stream, directPrepareError);
            },
                       input.preparation);
        if (!preparedFrame.active()) {
            errors.fail_submission(
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
            cameraFilmLinearActive || descriptors.post.active() ||
            grainStageActive || filmEffectsActive || gateOutputActive;
        DiffusionStageBinding directCameraDiffusion{};
        bind_direct_diffusion(preparedFrame, frame, snapshot, directCameraDiffusion, errors);
        const JuicerProcess::Root::PreparedCudaFrame::FocusedPreparedView prepared =
            preparedFrame.focused_resources();
        if (!prepared.active ||
            prepared.densityBoundsHash != input.scannerBoundsHash ||
            prepared.scannerDescriptorHash != descriptors.scanner.hash ||
            prepared.selectedMethod != input.film.method) {
            errors.fail_route("ResourceDescriptorMismatch phase=3C field=direct_prepared_view");
        }

        JuicerCuda::DirectPipelineRunParams run{};
        run.src = frame.source;
        run.srcRowBytes = static_cast<std::size_t>(frame.sourceRowBytes);
        run.dst = frame.destination;
        run.dstRowBytes = static_cast<std::size_t>(frame.destinationRowBytes);
        run.width = width;
        run.height = height;
        run.nComponents = frame.components;

        JuicerCuda::FilmPayloadPack directFilmPayloads{};
        std::string packDiagnostic;
        if (!JuicerCuda::pack_film_payloads(
                input.film,
                prepared.film,
                nullptr,
                input.filmRouteCorrectionScale,
                directFilmPayloads,
                packDiagnostic)) {
            errors.fail_route(packDiagnostic.c_str());
        }

        if (cameraAutoEnabled) {
            meter_camera_exposure(
                preparedFrame,
                directFilmPayloads,
                frame,
                cameraMeteringMethod,
                "MissingRequiredResource phase=3C field=auto_exposure_buffers",
                errors);
        }

        run.filmRaw = directFilmPayloads.filmRaw;
        run.filmExpose = directFilmPayloads.filmExposure;
        run.filmDevelop = directFilmPayloads.filmDevelop;
        auto directCameraFilmLinear = expose_direct_camera(
            run,
            preparedFrame,
            input.identity,
            frame,
            snapshot,
            cameraDiffusionActive,
            directCameraDiffusion,
            directHalation,
            errors);
        const bool directUseFusedScannerPostSpatialDirHandoff =
            !cameraFilmLinearActive && descriptors.dir.hash != 0 &&
            descriptors.post.active() &&
            !captureDensityConsumerActive;
        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker focusedWorkspace =
            preparedFrame.workspace_lease();
        if (focusedWorkspace.active()) {
            std::string transitionError;
            if (!preparedFrame.checkpoint_large_scratch_transition(
                    focusedWorkspace,
                    frame.stream,
                    "direct_large_scratch_transition",
                    transitionError)) {
                errors.fail_submission(
                    "direct_large_scratch_transition",
                    "direct large scratch transition checkpoint failed",
                    transitionError);
            }
        }
        bool directDirUsesSourceBuildCachedLogRaw = false;
        bool directCaptureDensityReady = false;
        if (descriptors.dir.hash != 0) {
            std::string spatialError;
            if (!preparedFrame.prepare_spatial_dir_resources(
                    descriptors.dir,
                    focusedWorkspace,
                    frame.stream,
                    spatialError)) {
                errors.fail_submission(
                    "direct_spatial_dir_prepare",
                    "direct spatial DIR preparation failed",
                    spatialError);
            }
        }
        const EffectsPayloads effects = pack_effects_payloads(preparedFrame, descriptors.effects, errors);
        OpticalWorkspace optical;
        if (directUseFocusedSplit) {
            optical = bind_direct_optical_workspace(preparedFrame, focusedWorkspace, frame.stream, descriptors.effects, errors);
        }
        GrainBinding grain;
        if (grainStageActive) {
            bind_grain(preparedFrame, focusedWorkspace, input.grain, grain, errors);
        }
        if (descriptors.dir.hash != 0) {
            directDirUsesSourceBuildCachedLogRaw = build_direct_dir(
                run,
                preparedFrame,
                focusedWorkspace,
                descriptors.dir,
                directCameraFilmLinear,
                cameraFilmLinearActive,
                frame.stream,
                directUseFusedScannerPostSpatialDirHandoff,
                errors);
        }
        pack_scan_stage(run.scanStage, prepared, input.outputGamut, scannerCorrection, errors);
        std::string scanError;
        if (
#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
            JuicerCuda::ExecutorTest::inject_scan_error(scanError) ||
#endif
            !preparedFrame.prepare_scan_error_stage(run.scanStage.scanErrorFlag, frame.stream, scanError)) {
            dirFailureMessage.deliver(dirFailureMessage.user, scanError);
            errors.fail_submission("direct_scan_error_stage", "direct scan error stage failed", scanError);
        }
        if (directUseFocusedSplit && !directUseFusedScannerPostSpatialDirHandoff) {
            develop_direct_capture(
                run,
                directCameraFilmLinear,
                cameraFilmLinearActive,
                descriptors.dir,
                optical,
                grainStageActive,
                grain,
                filmEffectsActive,
                effects,
                grainDebugActive,
                frame.stream,
                errors);
            directCaptureDensityReady = true;
        }
        if (cameraDiffusionActive) {
            std::string diffusionReleaseError;
            if (!preparedFrame.release_diffusion_resources_after_use(
                    frame.stream,
                    diffusionReleaseError)) {
                errors.fail_submission(
                    "direct_diffusion_release_after_use",
                    "direct diffusion resource release failed",
                    diffusionReleaseError);
            }
            directCameraDiffusion = {};
            directCameraFilmLinear = {};
        }

        if (gateOutputActive && !grainDebugActive && descriptors.effects->gateTransmittanceActive) {
            build_direct_gate_mask(effects, optical, frame.stream, errors);
        }
        scan_direct_output(
            run,
            preparedFrame,
            focusedWorkspace,
            descriptors,
            optical,
            effects,
            directUseFocusedSplit,
            grainDebugActive,
            directUseFusedScannerPostSpatialDirHandoff,
            directCaptureDensityReady,
            directDirUsesSourceBuildCachedLogRaw,
            gateOutputActive,
            frame,
            errors);
        if (!preparedFrame.finalize_scan_error_stage(run.scanStage.scanErrorFlag, frame.stream, scanError)) {
            const std::string diagnostic = halationExecutable
                                               ? halation_completion_diagnostic(input.identity, directHalation, snapshot, "scan_error_finalize", scanError)
                                               : scanError;
            errors.fail_submission("direct_scan_error_finalize", "direct scan error finalize failed", diagnostic);
        }
        std::string useError;
        if (!preparedFrame.record_use(frame.stream, useError)) {
            errors.fail_submission("prepared_frame_use_fence", "CUDA prepared-frame use fencing failed", useError);
        }
        std::string finishError;
        if (!preparedFrame.finish(frame.stream, finishError)) {
            const std::string diagnostic = halationExecutable
                                               ? halation_completion_diagnostic(input.identity, directHalation, snapshot, "prepared_frame_finish", finishError)
                                               : finishError;
            errors.fail_submission("direct_prepared_frame_finish", "direct prepared frame finish failed", diagnostic);
        }
        return;
    }

    static void execute_print_schedule(const ScheduleInput& input, PendingContextLossRecovery& recovery, const DirFailureMessage& dirFailureMessage) {
        const auto& frame = input.frame;
        auto& snapshot = input.snapshot;
        const auto& deviceContextKey = snapshot.deviceContextKey;
        const auto& win = frame.renderWindow;
        const int width = win.x2 - win.x1;
        const int height = win.y2 - win.y1;
        const auto& autoExposureDescriptor = frame.autoExposureDescriptor;
        const bool cameraAutoEnabled = input.cameraAutoEnabled;
        const auto cameraMeteringMethod = input.meteringMethod;
        const auto& descriptors = input.descriptors;
        const ExecutionErrors errors{recovery, frame.traceInfo, "CUDA print route blocked"};
        JuicerProcess::Root::AutoExposureBufferRequest autoExposureBufferRequest{};
        autoExposureBufferRequest.enabled =
            cameraAutoEnabled &&
            (frame.components == 3 || frame.components == 4) &&
            autoExposureDescriptor.previewWidth > 0 &&
            autoExposureDescriptor.previewHeight > 0;
        autoExposureBufferRequest.descriptor = autoExposureDescriptor;
        const bool grainStageActive =
            descriptors.grain.has_value();
        const bool grainDebugActive =
            grainStageActive && input.grain.debugView != 0;
        const bool filmEffectsActive =
            descriptors.effects.has_value() &&
            descriptors.effects->filmActive;
        const bool gateOutputActive =
            descriptors.effects.has_value() &&
            descriptors.effects->gateOutputActive;
        const bool cameraDiffusionActive =
            frame.diffusionFrameSet.has_value() &&
            frame.diffusionFrameSet->camera.has_value();
        const bool enlargerDiffusionActive =
            frame.diffusionFrameSet.has_value() &&
            frame.diffusionFrameSet->enlarger.has_value();
        const bool routeDiffusionActive =
            cameraDiffusionActive || enlargerDiffusionActive;
        const bool captureDensityConsumerActive =
            grainStageActive || filmEffectsActive;
        std::string prepareError;
        JuicerProcess::Root::PreparedCudaFrame preparedFrame =
            std::visit([&](const auto* preparation) {
                return JuicerProcess::root().prepare_cuda_frame(deviceContextKey, snapshot, *preparation, autoExposureBufferRequest, frame.stream, prepareError);
            },
                       input.preparation);
        if (!preparedFrame.active()) {
            errors.fail_submission(
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
            descriptors.post.active() || grainStageActive ||
            filmEffectsActive || gateOutputActive;
        DiffusionStageBinding printCameraDiffusion{};
        DiffusionStageBinding printEnlargerDiffusion{};
        bind_print_diffusion(preparedFrame, frame, snapshot, printCameraDiffusion, printEnlargerDiffusion, errors);
        const JuicerProcess::Root::PreparedCudaFrame::FocusedPreparedView prepared =
            preparedFrame.focused_resources();
        const JuicerProcess::Root::PreparedCudaFrame::PrintPreparedView preparedPrint =
            preparedFrame.print_resources();
        if (!prepared.active || !preparedPrint.active ||
            prepared.densityBoundsHash != input.scannerBoundsHash ||
            prepared.scannerDescriptorHash != descriptors.scanner.hash ||
            prepared.selectedMethod != input.film.method) {
            errors.fail_route(
                "ResourceDescriptorMismatch phase=4C field=print_prepared_view");
        }
        const auto scannerCorrection = std::visit([&](const auto& source) -> Scanner::ScannerColorCorrectionDescriptor {
            using Source = std::decay_t<decltype(source)>;
            if constexpr (std::is_same_v<Source, PrintCorrectionSource>) {
                return build_print_correction(source.recipe, source.payload, preparedPrint, frame.traceVerbose, errors);
            } else {
                return source;
            }
        },
                                                  input.correction);
        JuicerCuda::PrintPipelineRunParams run{};
        run.src = frame.source;
        run.srcRowBytes = static_cast<std::size_t>(frame.sourceRowBytes);
        run.dst = frame.destination;
        run.dstRowBytes = static_cast<std::size_t>(frame.destinationRowBytes);
        run.width = width;
        run.height = height;
        run.nComponents = frame.components;

        JuicerCuda::FilmPayloadPack filmPayloads{};
        std::string payloadDiagnostic;
        if (!JuicerCuda::pack_film_payloads(
                input.film,
                prepared.film,
                nullptr,
                input.filmRouteCorrectionScale,
                filmPayloads,
                payloadDiagnostic)) {
            errors.fail_route(payloadDiagnostic.c_str());
        }

        if (cameraAutoEnabled) {
            meter_camera_exposure(
                preparedFrame,
                filmPayloads,
                frame,
                cameraMeteringMethod,
                "MissingRequiredResource phase=4C field=auto_exposure_buffers",
                errors);
        }

        JuicerCuda::PrintCudaPayloadPack printPayloads{};
        if (!JuicerCuda::pack_print_cuda_payloads(
                input.printExposure,
                preparedPrint,
                scannerCorrection.exposureScale,
                printPayloads,
                payloadDiagnostic)) {
            errors.fail_route(payloadDiagnostic.c_str());
        }
        run.filmRaw = filmPayloads.filmRaw;
        run.filmExpose = filmPayloads.filmExposure;
        run.filmDevelop = filmPayloads.filmDevelop;
        run.printExpose = printPayloads.expose;
        run.printDevelop = printPayloads.develop;
        auto printCameraFilmLinear = expose_print_camera(
            run,
            preparedFrame,
            input.identity,
            frame,
            snapshot,
            cameraDiffusionActive,
            printCameraDiffusion,
            printHalation,
            errors);
        const bool printUseFusedScannerPostSpatialDirHandoff =
            !routeDiffusionActive && !halationExecutable &&
            descriptors.dir.hash != 0 &&
            descriptors.post.active() &&
            !captureDensityConsumerActive;
        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker focusedWorkspace =
            preparedFrame.workspace_lease();
        if (focusedWorkspace.active()) {
            std::string transitionError;
            if (!preparedFrame.checkpoint_large_scratch_transition(
                    focusedWorkspace,
                    frame.stream,
                    "print_large_scratch_transition",
                    transitionError)) {
                errors.fail_submission(
                    "print_large_scratch_transition",
                    "print large scratch transition checkpoint failed",
                    transitionError);
            }
        }
        bool printDirUsesSourceBuildCachedLogRaw = false;
        bool printCaptureDensityReady = false;
        if (descriptors.dir.hash != 0) {
            std::string spatialError;
            if (!preparedFrame.prepare_spatial_dir_resources(
                    descriptors.dir,
                    focusedWorkspace,
                    frame.stream,
                    spatialError)) {
                errors.fail_submission(
                    "print_spatial_dir_prepare",
                    "print spatial DIR preparation failed",
                    spatialError);
            }
        }
        const EffectsPayloads effects = pack_effects_payloads(preparedFrame, descriptors.effects, errors);
        OpticalWorkspace optical;
        if (printUseFocusedSplit) {
            optical = bind_print_optical_workspace(preparedFrame, focusedWorkspace, frame.stream, descriptors.effects, errors);
        }
        GrainBinding grain;
        if (grainStageActive) {
            bind_grain(preparedFrame, focusedWorkspace, input.grain, grain, errors);
        }
        if (descriptors.dir.hash != 0) {
            printDirUsesSourceBuildCachedLogRaw = build_print_dir(
                run,
                preparedFrame,
                focusedWorkspace,
                descriptors.dir,
                printCameraFilmLinear,
                cameraFilmLinearActive,
                frame.stream,
                printUseFusedScannerPostSpatialDirHandoff,
                errors);
        }
        pack_scan_stage(run.scanStage, prepared, input.outputGamut, scannerCorrection, errors);
        std::string scanError;
        if (
#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
            JuicerCuda::ExecutorTest::inject_scan_error(scanError) ||
#endif
            !preparedFrame.prepare_scan_error_stage(
                run.scanStage.scanErrorFlag,
                frame.stream,
                scanError)) {
            dirFailureMessage.deliver(dirFailureMessage.user, scanError);
            errors.fail_submission(
                "print_scan_error_stage",
                "print scan error stage failed",
                scanError);
        }
        if (printUseFocusedSplit && !printUseFusedScannerPostSpatialDirHandoff) {
            develop_print_capture(
                run,
                printCameraFilmLinear,
                cameraFilmLinearActive,
                descriptors.dir,
                optical,
                grainStageActive,
                grain,
                filmEffectsActive,
                effects,
                grainDebugActive,
                frame.stream,
                errors);
            printCaptureDensityReady = true;
            if (!grainDebugActive) {
                develop_print_medium(run, preparedFrame, printEnlargerDiffusion, enlargerDiffusionActive, optical, frame.stream, errors);
            }
        }
        if (routeDiffusionActive) {
            std::string diffusionReleaseError;
            if (!preparedFrame.release_diffusion_resources_after_use(
                    frame.stream,
                    diffusionReleaseError)) {
                errors.fail_submission(
                    "print_diffusion_release_after_use",
                    "print diffusion resource release failed",
                    diffusionReleaseError);
            }
            printCameraDiffusion = {};
            printEnlargerDiffusion = {};
            printCameraFilmLinear = {};
        }

        if (gateOutputActive && !grainDebugActive && descriptors.effects->gateTransmittanceActive) {
            build_print_gate_mask(effects, optical, frame.stream, errors);
        }
        scan_print_output(
            run,
            preparedFrame,
            focusedWorkspace,
            descriptors,
            optical,
            effects,
            printUseFocusedSplit,
            grainDebugActive,
            printUseFusedScannerPostSpatialDirHandoff,
            printCaptureDensityReady,
            printDirUsesSourceBuildCachedLogRaw,
            gateOutputActive,
            frame,
            input.recipeHash,
            snapshot,
            errors);
        if (!preparedFrame.finalize_scan_error_stage(
                run.scanStage.scanErrorFlag,
                frame.stream,
                scanError)) {
            const std::string diagnostic = halationExecutable
                                               ? halation_completion_diagnostic(input.identity, printHalation, snapshot, "scan_error_finalize", scanError)
                                               : scanError;
            errors.fail_submission(
                "print_scan_error_finalize",
                "print scan error finalize failed",
                diagnostic);
        }
        std::string useError;
        if (!preparedFrame.record_use(frame.stream, useError)) {
            errors.fail_submission("prepared_frame_use_fence", "CUDA prepared-frame use fencing failed", useError);
        }
        std::string finishError;
        if (!preparedFrame.finish(frame.stream, finishError)) {
            const std::string diagnostic = halationExecutable
                                               ? halation_completion_diagnostic(input.identity, printHalation, snapshot, "prepared_frame_finish", finishError)
                                               : finishError;
            errors.fail_submission(
                "print_prepared_frame_finish",
                "print prepared frame finish failed",
                diagnostic);
        }
        return;
    }
    void execute_direct(const DirectExecutionInput& input, PendingContextLossRecovery& recovery, const DirFailureMessage& dirFailureMessage) {
        const RenderRecipe* directRecipe = &input.recipe;
        const FocusedRenderPayload* directPayload = &input.payload;
        const auto& frame = input.frame;
        const auto& win = frame.renderWindow;
        const int width = win.x2 - win.x1;
        const int height = win.y2 - win.y1;
        const auto cameraMeteringMethod = input.recipe.filmRaw.autoExposureMethod;
        const ExecutionErrors errors{recovery, frame.traceInfo, "CUDA direct route blocked"};
        if (cameraMeteringMethod == Spektrafilm::AutoExposureMethod::Median) {
            errors.fail_route(Spektrafilm::kQuantizedMedianNotAcceptedForPhase3);
        }
        if (!(frame.components == 3 || frame.components == 4)) {
            errors.fail_route("UnsupportedDirectComponentCountForPhase3C");
        }
        Scanner::ScannerColorCorrectionDescriptor scannerCorrection{};
        const RouteDescriptors descriptors = build_direct_descriptors(*directRecipe, *directPayload, frame, scannerCorrection, errors);
        const ScatterHalationFrameDescriptor* halationRequestDescriptor =
            frame.scatterHalation ? &*frame.scatterHalation : nullptr;

        JuicerProcess::Root::CudaFramePreparationRequest directPreparation{};
        directPreparation.recipe = directRecipe;
        directPreparation.exposureTables = &directPayload->exposureTables;
        directPreparation.filmRawConfig = &directPayload->filmRawConfig;
        directPreparation.filmTcLut = directPayload->filmTcLut
                                          ? &*directPayload->filmTcLut
                                          : nullptr;
        directPreparation.scannerTables = &directPayload->scannerTables;
        directPreparation.scannerColor = &directPayload->scannerColor;
        directPreparation.scannerLutDescriptor = &descriptors.scanner;
        directPreparation.outputBoundaryTable =
            directPayload->outputBoundaryTable.get();
        directPreparation.scannerPostEffects = &descriptors.post;
        directPreparation.spatialDirDescriptor = &descriptors.dir;
        directPreparation.diffusionFrameSetDescriptor =
            frame.diffusionFrameSet
                ? &*frame.diffusionFrameSet
                : nullptr;
        directPreparation.scatterHalationDescriptor = halationRequestDescriptor;
        directPreparation.visualGrainDescriptor =
            descriptors.grain;
        directPreparation.effectsDescriptor = descriptors.effects;
        directPreparation.requestedWidth = width;
        directPreparation.requestedHeight = height;
        const ScheduleInput schedule{frame, input.snapshot, JuicerProcess::Root::preparation_identity(*directRecipe), descriptors, film_payload_input(directRecipe->filmRaw, directRecipe->filmDevelop, directRecipe->dirCouplers, directRecipe->densityBounds), scannerCorrection.exposureScale, directRecipe->visualGrain, directRecipe->scannerOutput.outputGamut, directRecipe->densityBounds.hash, directRecipe->hash, directRecipe->filmRaw.autoExposureEnabled, directRecipe->filmRaw.autoExposureMethod, &directPreparation, scannerCorrection, {}};
        execute_direct_schedule(schedule, recovery, dirFailureMessage);
    }

    void execute_print(const PrintExecutionInput& input, PendingContextLossRecovery& recovery, const DirFailureMessage& dirFailureMessage) {
        const RenderRecipe* printRecipe = &input.recipe;
        const FocusedRenderPayload* printPayload = &input.payload;
        const auto& frame = input.frame;
        const auto& win = frame.renderWindow;
        const int width = win.x2 - win.x1;
        const int height = win.y2 - win.y1;
        const ExecutionErrors errors{recovery, frame.traceInfo, "CUDA print route blocked"};
        if (!(frame.components == 3 || frame.components == 4)) {
            errors.fail_route("UnsupportedPrintComponentCountForPhase4C");
        }

        const RouteDescriptors descriptors = build_print_descriptors(*printRecipe, frame, errors);
        const ScatterHalationFrameDescriptor* halationRequestDescriptor =
            frame.scatterHalation ? &*frame.scatterHalation : nullptr;

        JuicerProcess::Root::CudaFramePreparationRequest preparation{};
        preparation.recipe = printRecipe;
        preparation.exposureTables = &printPayload->exposureTables;
        preparation.filmRawConfig = &printPayload->filmRawConfig;
        preparation.filmTcLut = printPayload->filmTcLut
                                    ? &*printPayload->filmTcLut
                                    : nullptr;
        preparation.printMainIlluminant = printPayload->printMainIlluminant
                                              ? &*printPayload->printMainIlluminant
                                              : nullptr;
        preparation.scannerTables = &printPayload->scannerTables;
        preparation.scannerColor = &printPayload->scannerColor;
        preparation.scannerLutDescriptor = &descriptors.scanner;
        preparation.outputBoundaryTable =
            printPayload->outputBoundaryTable.get();
        preparation.scannerPostEffects = &descriptors.post;
        preparation.spatialDirDescriptor = &descriptors.dir;
        preparation.diffusionFrameSetDescriptor =
            frame.diffusionFrameSet
                ? &*frame.diffusionFrameSet
                : nullptr;
        preparation.scatterHalationDescriptor = halationRequestDescriptor;
        preparation.visualGrainDescriptor =
            descriptors.grain;
        preparation.effectsDescriptor = descriptors.effects;
        preparation.requestedWidth = width;
        preparation.requestedHeight = height;
        const ScheduleInput schedule{frame, input.snapshot, JuicerProcess::Root::preparation_identity(*printRecipe), descriptors, film_payload_input(printRecipe->filmRaw, printRecipe->filmDevelop, printRecipe->dirCouplers, printRecipe->enlargerFilmBounds), 1.0f, printRecipe->visualGrain, printRecipe->scannerOutput.outputGamut, printRecipe->densityBounds.hash, printRecipe->hash, printRecipe->filmRaw.autoExposureEnabled, printRecipe->filmRaw.autoExposureMethod, &preparation, PrintCorrectionSource{*printRecipe, *printPayload}, printRecipe->print.exposure};
        execute_print_schedule(schedule, recovery, dirFailureMessage);
    }

    PreparedDescriptors describe_execution(const RenderRecipe& recipe, const FocusedRenderPayload& payload, const ExecutionFrame& frame) {
        PendingContextLossRecovery recovery;
        const ExecutionErrors errors{recovery, frame.traceInfo, "CUDA prepared descriptor construction failed"};
        PreparedDescriptors result;
        const auto descriptors = Spektrafilm::scan_route_is_print(recipe.profileRoute.scanRoute)
                                     ? build_print_descriptors(recipe, frame, errors)
                                     : build_direct_descriptors(recipe, payload, frame, result.correction, errors);
        result.route = recipe.profileRoute.scanRoute;
        result.capturePolarity = recipe.profileRoute.capturePolarity;
        result.scanner = descriptors.scanner;
        result.post = descriptors.post;
        result.spatialDir = descriptors.dir;
        result.grain = descriptors.grain;
        result.effects = descriptors.effects;
        result.grainRecipe = recipe.visualGrain;
        result.color = payload.scannerColor;
        result.outputGamut = recipe.scannerOutput.outputGamut;
        result.diffusion = frame.diffusionFrameSet;
        result.halation = frame.scatterHalation;
        return result;
    }

    void execute_prepared(const PreparedExecutionInput& input, PendingContextLossRecovery& recovery, const DirFailureMessage& dirFailureMessage) {
        const bool print = Spektrafilm::scan_route_is_print(input.descriptors.route);
        const ExecutionErrors errors{recovery, input.frame.traceInfo, print ? "CUDA print route blocked" : "CUDA direct route blocked"};
        if (!print && input.frame.autoExposureDescriptor.method == Spektrafilm::AutoExposureMethod::Median) {
            errors.fail_route(Spektrafilm::kQuantizedMedianNotAcceptedForPhase3);
        }
        if (!(input.frame.components == 3 || input.frame.components == 4)) {
            errors.fail_route(print ? "UnsupportedPrintComponentCountForPhase4C" : "UnsupportedDirectComponentCountForPhase3C");
        }
        const RouteDescriptors descriptors{input.descriptors.scanner, input.descriptors.post, input.descriptors.spatialDir, input.descriptors.grain, input.descriptors.effects};
        const ScheduleInput schedule{input.frame, input.snapshot, input.preparation.identity, descriptors, input.film, input.filmRouteCorrectionScale, input.descriptors.grainRecipe, input.descriptors.outputGamut, input.preparation.focused.densityBounds.hash, input.recipeHash, input.cameraAutoEnabled, input.frame.autoExposureDescriptor.method, &input.preparation, input.descriptors.correction, input.printExposure};
        if (Spektrafilm::scan_route_is_print(input.descriptors.route)) {
            execute_print_schedule(schedule, recovery, dirFailureMessage);
        } else {
            execute_direct_schedule(schedule, recovery, dirFailureMessage);
        }
    }

} // namespace JuicerCuda
