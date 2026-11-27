#include "Cuda/Executor/Stages.h"

#include "Cuda/Film/FilmLaunchers.h"
#include "Cuda/Scan/PostEffects.h"

namespace JuicerCuda {

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

    void check_diffusion_launch(
        Diffusion::LaunchResult result, const char* prefix, const ExecutionErrors& errors) {
        if (!result.ok()) {
            if (result.failure.status.api == FJ_API_CUDA_RUNTIME) {
                const auto* message = cudaGetErrorString(static_cast<cudaError_t>(result.failure.status.native_code));
                set_failure(result.failure, result.failure.status, detail_or_unknown(message));
                errors.fail_submission(result.stage, prefix, result.failure);
            }
            errors.fail_policy(errors.restrictionPrefix, std::move(result.failure));
        }
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
        JuicerCuda::Failure& failure) {
        auto& outDiagnostic = failure.diagnostic;
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
        if (result.stage != JuicerCuda::ScatterHalationLaunchStage::Binding) {
            failure.status = JuicerCuda::runtime_failure_status(result.status);
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

    struct CudaFailureTrace {
        const char* prefix = "CUDA operation failed";
        const char* detail = nullptr;
    };

    GrainModel grain_model(const ScheduleInput& input) {
        // The raw boundary has already rejected simultaneous descriptors.
        if (input.preparation.filmGrain)
            return GrainModel::Film;
        return input.descriptors.grain ? GrainModel::Current : GrainModel::Disabled;
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

    std::string halation_completion_diagnostic(
        const JuicerProcess::Root::PreparationIdentity& identity,
        const JuicerCuda::ScatterHalationPreparedView& halation,
        const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        std::uint64_t contextEpoch,
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
                      std::to_string(contextEpoch);
        diagnostic += " observing_completion_boundary=";
        diagnostic += boundary;
        diagnostic += " cuda_status=";
        diagnostic += cudaStatus;
        return diagnostic;
    }

    void bind_diffusion(
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        const JuicerCuda::ExecutionFrame& frame,
        DiffusionStageBinding& cameraDiffusion,
        DiffusionStageBinding* enlargerDiffusion,
        const ExecutionErrors& errors) {
        const auto diffusionPrepared = preparedFrame.diffusion_resources();

        if (frame.diffusionFrameSet) {
            if (!diffusionPrepared.active ||
                diffusionPrepared.executionDescriptor.frameSetHash !=
                    frame.diffusionFrameSet->hash ||
                diffusionPrepared.executionDescriptor.contextEpoch !=
                    preparedFrame.admitted_context_epoch() ||
                diffusionPrepared.spectrumCount == 0 ||
                diffusionPrepared.spectrumCount !=
                    diffusionPrepared.executionDescriptor.uniqueSpectrumCount) {
                errors.fail_route(
                    "MissingRequiredResource component=diffusion field=prepared_view");
            }
            if (!enlargerDiffusion || frame.diffusionFrameSet->camera) {
                std::string cameraBindingDiagnostic;
                if (!bind_diffusion_stage(
                        *frame.diffusionFrameSet,
                        Spektrafilm::DiffusionLinearStage::CameraFilmLinear,
                        diffusionPrepared,
                        cameraDiffusion,
                        cameraBindingDiagnostic)) {
                    errors.fail_route(
                        cameraBindingDiagnostic.c_str());
                }
            }
            if (enlargerDiffusion && frame.diffusionFrameSet->enlarger) {
                std::string enlargerBindingDiagnostic;
                if (!bind_diffusion_stage(
                        *frame.diffusionFrameSet,
                        Spektrafilm::DiffusionLinearStage::EnlargerPrintLinear,
                        diffusionPrepared,
                        *enlargerDiffusion,
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


    void append_highlight_identity(JuicerCuda::Failure& failure,
                                   const JuicerProcess::Root::PreparationIdentity& identity,
                                   std::uint64_t contextEpoch) {
        failure.diagnostic += " route=";
        failure.diagnostic += Spektrafilm::scan_route_key(identity.scanRoute);
        failure.diagnostic += " stock=";
        failure.diagnostic += identity.filmProfileKey;
        failure.diagnostic += " stock_asset=" + std::to_string(identity.filmProfileAssetVersionToken);
        failure.diagnostic += " boost_recipe=" + std::to_string(identity.highlightBoostHash);
        failure.diagnostic += " context_epoch=" + std::to_string(contextEpoch);
    }

    void enqueue_highlight_reconstruction(
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        JuicerCuda::CameraFilmLinearExposurePlanes carrier,
        const JuicerProcess::Root::PreparationIdentity& identity,
        const JuicerCuda::ExecutionFrame& frame,
        const ExecutionErrors& errors) {
        JuicerCuda::Failure failure;
        if (!preparedFrame.enqueue_highlight_boost(carrier, frame.stream, failure)) {
            append_highlight_identity(failure, identity, preparedFrame.admitted_context_epoch());
            errors.fail_submission("highlight_boost_launch", "highlight reconstruction submission failed", failure);
        }
    }

    void check_highlight_reconstruction(
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        const JuicerProcess::Root::PreparationIdentity& identity,
        const ExecutionErrors& errors) {
        JuicerCuda::Failure failure;
        if (!preparedFrame.check_highlight_boost(failure)) {
            append_highlight_identity(failure, identity, preparedFrame.admitted_context_epoch());
            errors.fail_submission("highlight_boost_result", "highlight reconstruction failed", failure);
        }
    }

    JuicerCuda::CameraFilmLinearExposurePlanes expose_camera(
        CaptureRoute route,
        JuicerCuda::FilmScanRunParams& run,
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        const JuicerProcess::Root::PreparationIdentity& identity,
        const JuicerCuda::ExecutionFrame& frame,
        const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        bool cameraDiffusionActive,
        DiffusionStageBinding& cameraDiffusion,
        JuicerCuda::ScatterHalationPreparedView& halation,
        const ExecutionErrors& errors) {
        const auto& deviceContextKey = snapshot.deviceContextKey;
        const bool halationExecutable = halation.descriptor != nullptr;
        const bool boostExecutable = identity.highlightBoostHash != 0;
        JuicerCuda::CameraFilmLinearExposurePlanes cameraFilmLinear{};
        if (cameraDiffusionActive) {
            if (!cameraDiffusion.active) {
                errors.fail_route(
                    "MissingRequiredResource component=diffusion field=camera_stage_binding");
            }
            preparedFrame.mark_diffusion_work_enqueued();
            cameraFilmLinear = camera_film_linear_planes(
                cameraDiffusion.stagePlanes);
            const cudaError_t exposureError =
                juicer_cuda_camera_film_linear_exposure(
                    &run,
                    cameraFilmLinear,
                    frame.stream);
            if (exposureError != cudaSuccess) {
                errors.fail_stage(
                    (route == CaptureRoute::Print ? "print_camera_film_linear_exposure_launch" : "direct_camera_film_linear_exposure_launch"),
                    (route == CaptureRoute::Print ? "print camera film-linear exposure launch failed" : "direct camera film-linear exposure launch failed"),
                    exposureError);
            }
            if (boostExecutable) {
                enqueue_highlight_reconstruction(preparedFrame, cameraFilmLinear, identity, frame, errors);
            }
            JuicerCuda::Diffusion::StageLaunchRequest diffusionLaunch{};
            diffusionLaunch.layout = cameraDiffusion.layout;
            diffusionLaunch.geometry = cameraDiffusion.geometry;
            diffusionLaunch.fullFrame = cameraDiffusion.fullFrame;
            diffusionLaunch.spectra = cameraDiffusion.spectra;
            diffusionLaunch.execution = cameraDiffusion.execution;
            diffusionLaunch.planes = &cameraDiffusion.stagePlanes;
            diffusionLaunch.stream = reinterpret_cast<cudaStream_t>(
                frame.stream);
            check_diffusion_launch(
                Diffusion::launch_stage(diffusionLaunch),
                route == CaptureRoute::Print ? "print camera diffusion launch failed" : "direct camera diffusion launch failed",
                errors);
            // launch_stage rotates semantic plane pointers after each channel.
            cameraFilmLinear = camera_film_linear_planes(
                cameraDiffusion.stagePlanes);
        } else if (halationExecutable || boostExecutable) {
            cameraFilmLinear = preparedFrame.camera_film_linear_carrier();
            const cudaError_t exposureError =
                juicer_cuda_camera_film_linear_exposure(
                    &run,
                    cameraFilmLinear,
                    frame.stream);
            if (exposureError != cudaSuccess) {
                errors.fail_stage(
                    (route == CaptureRoute::Print ? "print_camera_film_linear_exposure_launch" : "direct_camera_film_linear_exposure_launch"),
                    (route == CaptureRoute::Print ? "print camera film-linear exposure launch failed" : "direct camera film-linear exposure launch failed"),
                    exposureError);
            }
            if (boostExecutable) {
                enqueue_highlight_reconstruction(preparedFrame, cameraFilmLinear, identity, frame, errors);
            }
        }
        if (halationExecutable) {
            JuicerCuda::Failure halationDiagnostic;
            // bind_camera_diffusion_carrier_immediately
            if (!launch_scatter_halation_for_route(
                    halation,
                    cameraDiffusionActive
                        ? cameraFilmLinear
                        : JuicerCuda::CameraFilmLinearExposurePlanes{},
                    cameraDiffusionActive && cameraDiffusion.active,
                    Spektrafilm::scan_route_label(
                        identity.scanRoute),
                    identity.filmProfileKey,
                    identity.filmProfileAssetVersionToken,
                    deviceContextKey,
                    preparedFrame.admitted_context_epoch(),
                    reinterpret_cast<cudaStream_t>(frame.stream),
                    halationDiagnostic)) {
                errors.fail_submission(
                    (route == CaptureRoute::Print ? "print_scatter_halation_launch" : "direct_scatter_halation_launch"),
                    (route == CaptureRoute::Print ? "print scatter-halation launch failed" : "direct scatter-halation launch failed"),
                    halationDiagnostic);
            }
            // post_halation_camera_film_linear
        }

        return cameraFilmLinear;
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

    OpticalWorkspace bind_optical_workspace(
        CaptureRoute route,
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& focusedWorkspace,
        void* stream,
        const std::optional<Spektrafilm::FilmJuicerEffectsFrameDescriptor>& descriptor,
        const ExecutionErrors& errors) {
        OpticalWorkspace optical;

        JuicerCuda::Failure focusedWorkspaceError;
        if (!preparedFrame.stage_optical_workspace(
                focusedWorkspace,
                stream,
                focusedWorkspaceError)) {
            errors.fail_submission(
                (route == CaptureRoute::Print ? "print_focused_workspace_stage" : "direct_focused_workspace_stage"),
                (route == CaptureRoute::Print ? "print focused workspace staging failed" : "direct focused workspace staging failed"),
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

    void bind_film_grain(JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
                         const FjFilmGrain& input,
                         const JuicerCuda::FrameRect& window,
                         CaptureRoute route,
                         GrainBinding& grain,
                         const ExecutionErrors& errors) {
        grain.filmBank = preparedFrame.film_grain_bank();
        const bool direct = route == CaptureRoute::Direct;
        const char* stage = direct ? (route == CaptureRoute::Print ? "print_film_grain_binding" : "direct_film_grain_binding") : "print_film_grain_binding";
        if (!grain.filmBank) {
            errors.fail_grain_stage({stage, direct ? (route == CaptureRoute::Print ? "print Film grain binding failed: missing frame bank" : "direct Film grain binding failed: missing frame bank") : "print Film grain binding failed: missing frame bank"}, cudaErrorInvalidValue);
        }
        JuicerCuda::FilmGrain::bind_film_grain(
            input, {window.x1, window.y1, window.x2 - window.x1, window.y2 - window.y1}, input.sample_step[0] > 0.0 ? preparedFrame.film_proxy_workspace() : nullptr, grain.filmTables, grain.filmFrame);
    }

    void bind_grain(
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& focusedWorkspace,
        const JuicerCuda::VisualGrainPayloadInput& input,
        GrainBinding& grain,
        const ExecutionErrors& errors) {
        const auto resources = preparedFrame.visual_grain_resources();
        grain.workspace = preparedFrame.visual_grain_workspace(
            focusedWorkspace);
        std::string grainPayloadDiagnostic;
        if (!resources.active ||
            !grain.workspace.active ||
            !JuicerCuda::pack_visual_grain_payload(
                input,
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

    DirBuildResult build_spatial_dir(
        CaptureRoute route,
        JuicerCuda::FilmScanRunParams& run,
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& focusedWorkspace,
        const Spektrafilm::SpatialDirDescriptor& descriptor,
        const JuicerCuda::CameraFilmLinearExposurePlanes& cameraFilmLinear,
        bool cameraFilmLinearActive,
        void* stream,
        CaptureExecution execution,
        const ExecutionErrors& errors) {
        bool dirUsesSourceBuildCachedLogRaw = false;

        const auto scratch = preparedFrame.spatial_dir_scratch(focusedWorkspace);
        const auto resources =
            preparedFrame.spatial_dir_resources(focusedWorkspace, descriptor.hash);
        if (!scratch.active || !resources.active) {
            errors.fail_route(
                (route == CaptureRoute::Print ? "MissingRequiredResource phase=4C field=prepared_spatial_dir" : "MissingRequiredResource phase=3D-3 field=prepared_spatial_dir"));
        }
        const Spektrafilm::DirScratchPlaneRoles& dirAdmittedRoles =
            scratch.planeRoles;
        const Spektrafilm::DirScratchPlaneRoles& dirAdmittedTargetRoles =
            scratch.targetPlaneRoles;
        dirUsesSourceBuildCachedLogRaw =
            spatial_dir_source_build_retains_cached_log_raw(
                dirAdmittedRoles,
                dirAdmittedTargetRoles);
        if (dirUsesSourceBuildCachedLogRaw &&
            !spatial_dir_required_cached_log_raw_present(
                scratch,
                dirAdmittedRoles.cachedLogRawPlanes)) {
            errors.fail_route(
                (route == CaptureRoute::Print ? "MissingRequiredResource phase=4C field=spatial_dir_source_build_cached_log_raw" : "MissingRequiredResource phase=3D-3 field=spatial_dir_source_build_cached_log_raw"));
        }
        float* cachedLogRawB =
            dirUsesSourceBuildCachedLogRaw ? scratch.logRawB : nullptr;
        float* cachedLogRawG =
            dirUsesSourceBuildCachedLogRaw ? scratch.logRawG : nullptr;
        float* cachedLogRawR =
            dirUsesSourceBuildCachedLogRaw ? scratch.logRawR : nullptr;
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
        dirBuildRequest.cameraFilmLinear = cameraFilmLinear;
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
        const cudaError_t dirError = juicer_cuda_build_spatial_dir(
            &run,
            dirBuildRequest);
        if (dirError != cudaSuccess) {
            errors.fail_stage(
                (route == CaptureRoute::Print ? "print_spatial_dir_launch" : "direct_spatial_dir_launch"),
                (route == CaptureRoute::Print ? "print spatial DIR build failed" : "direct spatial DIR build failed"),
                dirError);
        }
        if (execution != CaptureExecution::FusedDirScanner &&
            !dirUsesSourceBuildCachedLogRaw) {
            JuicerCuda::Failure stageError;
            if (!preparedFrame.stage_spatial_dir_cached_log_raw_for_final_develop(
                    focusedWorkspace,
                    stream,
                    stageError)) {
                errors.fail_submission(
                    (route == CaptureRoute::Print ? "print_spatial_dir_cached_log_raw_stage" : "direct_spatial_dir_cached_log_raw_stage"),
                    (route == CaptureRoute::Print ? "print spatial DIR cached log raw staging failed" : "direct spatial DIR cached log raw staging failed"),
                    stageError);
            }
        }
        const auto finalScratch = preparedFrame.spatial_dir_scratch(focusedWorkspace);
        const bool dirRequiresCachedLogRaw =
            dirUsesSourceBuildCachedLogRaw ||
            (execution != CaptureExecution::FusedDirScanner &&
             dirAdmittedTargetRoles.cachedLogRawPlanes == 3);
        if (!finalScratch.filteredCorrectionY || !finalScratch.filteredCorrectionM ||
            !finalScratch.filteredCorrectionC) {
            errors.fail_route(
                (route == CaptureRoute::Print ? "MissingRequiredResource phase=4C field=spatial_dir_filtered_correction" : "MissingRequiredResource phase=3D-3 field=spatial_dir_filtered_correction"));
        }
        if (dirRequiresCachedLogRaw &&
            !spatial_dir_required_cached_log_raw_present(
                finalScratch,
                dirAdmittedTargetRoles.cachedLogRawPlanes)) {
            errors.fail_route(
                (route == CaptureRoute::Print ? "MissingRequiredResource phase=4C field=spatial_dir_cached_log_raw" : "MissingRequiredResource phase=3D-3 field=spatial_dir_cached_log_raw"));
        }
        if (dirRequiresCachedLogRaw && !dirUsesSourceBuildCachedLogRaw) {
            const cudaError_t logRawError = cameraFilmLinearActive
                                                ? juicer_cuda_build_spatial_dir_cached_log_raw_from_camera_film_linear(
                                                      &run,
                                                      cameraFilmLinear,
                                                      finalScratch.logRawB,
                                                      finalScratch.logRawG,
                                                      finalScratch.logRawR,
                                                      stream)
                                                : juicer_cuda_build_spatial_dir_cached_log_raw(
                                                      &run,
                                                      finalScratch.logRawB,
                                                      finalScratch.logRawG,
                                                      finalScratch.logRawR,
                                                      stream);
            if (logRawError != cudaSuccess) {
                errors.fail_stage(
                    (route == CaptureRoute::Print ? "print_spatial_dir_cached_log_raw_launch" : "direct_spatial_dir_cached_log_raw_launch"),
                    (route == CaptureRoute::Print ? "print spatial DIR cached log raw build failed" : "direct spatial DIR cached log raw build failed"),
                    logRawError);
            }
        }
        bind_spatial_dir_final_develop_to_payload(run.filmDevelop, finalScratch);

        return {dirUsesSourceBuildCachedLogRaw
                    ? (dirAdmittedRoles.cachedLogRawPlanes == 2 ? DirCache::SourceBg : DirCache::SourceBgr)
                : dirRequiresCachedLogRaw ? DirCache::StagedBgr
                                          : DirCache::Absent};
    }

    void apply_developed_grain(
        CaptureRoute route,
        GrainModel model,
        const GrainBinding& grain,
        const JuicerProcess::Root::PreparedCudaFrame::CaptureFilmDensityWorkspaceView& density,
        int width,
        int height,
        void* stream,
        const ExecutionErrors& errors) {
        if (model == GrainModel::Disabled)
            return;

        JuicerCuda::FilmGrain::FilmPlanes planes{density.c, density.m, density.y};
        const bool direct = route == CaptureRoute::Direct;
        if (model == GrainModel::Film) {
            const auto result = JuicerCuda::FilmGrain::launch_developed(
                grain.filmBank, grain.filmTables, grain.filmFrame, planes, static_cast<cudaStream_t>(stream));
            if (result.status != cudaSuccess) {
                const JuicerCuda::GrainFailureSite launch{
                    direct ? (route == CaptureRoute::Print ? "print_film_grain_launch" : "direct_film_grain_launch") : "print_film_grain_launch",
                    direct ? (route == CaptureRoute::Print ? "print Film grain launch failed" : "direct Film grain launch failed") : "print Film grain launch failed"};
                errors.fail_grain_stage(result.failure ? *result.failure : launch, result.status);
            }
            return;
        }
        const auto status = juicer_cuda_apply_visual_grain(
            &grain.payload,
            &grain.kernels,
            width,
            height,
            planes.cyan,
            planes.magenta,
            planes.yellow,
            grain.workspace.filterTemp,
            grain.workspace.scaleWork,
            grain.workspace.deltaAccum,
            grain.workspace.layerWork,
            grain.workspace.sharedDelta,
            stream);
        if (status != cudaSuccess) {
            errors.fail_grain_stage(
                {direct ? (route == CaptureRoute::Print ? "print_visual_grain_launch" : "direct_visual_grain_launch") : "print_visual_grain_launch",
                 direct ? (route == CaptureRoute::Print ? "print visual grain launch failed" : "direct visual grain launch failed") : "print visual grain launch failed"},
                status);
        }
    }

    void develop_capture(
        CaptureRoute route,
        JuicerCuda::FilmScanRunParams& run,
        const JuicerCuda::CameraFilmLinearExposurePlanes& cameraFilmLinear,
        bool cameraFilmLinearActive,
        const Spektrafilm::SpatialDirDescriptor& descriptor,
        const OpticalWorkspace& optical,
        GrainModel grainModel,
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
                          ? juicer_cuda_capture_density_from_camera_film_linear(
                                &run,
                                cameraFilmLinear,
                                optical.density.c,
                                optical.density.m,
                                optical.density.y,
                                stream)
                          : juicer_cuda_capture_density(
                                &run,
                                optical.density.c,
                                optical.density.m,
                                optical.density.y,
                                stream);
        if (launchError != cudaSuccess) {
            errors.fail_stage(
                (route == CaptureRoute::Print ? "print_capture_density_launch" : "direct_capture_density_launch"),
                (route == CaptureRoute::Print ? "print capture-film density launch failed" : "direct capture-film density launch failed"),
                launchError);
        }

        if (descriptor.hash != 0) {
            run.filmDevelop.spatialDir.logRawB = nullptr;
            run.filmDevelop.spatialDir.logRawG = nullptr;
            run.filmDevelop.spatialDir.logRawR = nullptr;
        }
        apply_developed_grain(route, grainModel, grain, optical.density, width, height, stream, errors);
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
                    (route == CaptureRoute::Print ? "print_film_effects_launch" : "direct_film_effects_launch"),
                    (route == CaptureRoute::Print ? "print film defects launch failed" : "direct film defects launch failed"),
                    launchError);
            }
        }
    }

    void build_gate_mask(CaptureRoute route, const EffectsPayloads& effects, const OpticalWorkspace& optical, void* stream, const ExecutionErrors& errors) {
        cudaError_t launchError = cudaSuccess;

        launchError = juicer_cuda_build_gate_defect_transmittance_focused(
            &effects.film,
            optical.scanner.gateTransmittance,
            optical.scanner.gateTransmittanceWidth,
            optical.scanner.gateTransmittanceHeight,
            stream);
        if (launchError != cudaSuccess) {
            errors.fail_stage(
                (route == CaptureRoute::Print ? "print_gate_transmittance_launch" : "direct_gate_transmittance_launch"),
                (route == CaptureRoute::Print ? "print gate defect mask launch failed" : "direct gate defect mask launch failed"),
                launchError);
        }
    }

    void bind_run_images(FilmScanRunParams& run, const ExecutionFrame& frame) {
        run.src = frame.source;
        run.srcRowBytes = static_cast<std::size_t>(frame.sourceRowBytes);
        run.dst = frame.destination;
        run.dstRowBytes = static_cast<std::size_t>(frame.destinationRowBytes);
        run.width = frame.renderWindow.x2 - frame.renderWindow.x1;
        run.height = frame.renderWindow.y2 - frame.renderWindow.y1;
        run.nComponents = frame.components;
    }

    void bind_run_film(FilmScanRunParams& run, const FilmPayloadPack& film) {
        run.filmRaw = film.filmRaw;
        run.filmExpose = film.filmExposure;
        run.filmDevelop = film.filmDevelop;
    }


    void annotate_halation_completion(
        Failure& failure,
        const JuicerProcess::Root::PreparationIdentity& identity,
        const ScatterHalationPreparedView& halation,
        const ResourceManager::SubmissionSnapshot& snapshot,
        std::uint64_t contextEpoch,
        const char* boundary) noexcept {
        try {
            failure.diagnostic = halation_completion_diagnostic(identity, halation, snapshot, contextEpoch, boundary, failure.diagnostic);
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }

} // namespace JuicerCuda
