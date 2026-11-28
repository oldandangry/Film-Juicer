#include "Cuda/Executor/Stages.h"


namespace JuicerCuda {

    void execute_direct_schedule(JuicerProcess::Root& root, const ScheduleInput& input, PendingContextLossRecovery& recovery, FjAbortCallback abortCallback) {
        const auto& frame = input.frame;
        const auto& requestedSnapshot = input.snapshot;
        const auto& deviceContextKey = requestedSnapshot.deviceContextKey;
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
        const GrainModel grainModel = grain_model(input);
        const bool grainDebugActive =
            descriptors.grain.has_value() && input.grain.coefficients.debug_view != 0;
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
            grainModel != GrainModel::Disabled || filmEffectsActive;
        const auto scannerCorrection = input.correction;
        JuicerCuda::Failure directPrepareError;
        check_abort(abortCallback);
        JuicerProcess::Root::PreparedCudaFrame preparedFrame =
            root.prepare_cuda_frame(deviceContextKey, requestedSnapshot, input.preparation, autoExposureBufferRequest, frame.stream, recovery, directPrepareError, input.hostPreparation);
        if (!preparedFrame.active()) {
            errors.fail_submission(
                preparedFrame.failure_stage_tag(),
                preparedFrame.failure_prefix(),
                directPrepareError);
        }

        check_abort(abortCallback);

        const auto& snapshot = preparedFrame.admitted_snapshot();

        JuicerCuda::ScatterHalationPreparedView directHalation =
            preparedFrame.scatter_halation_resources();
        const bool halationExecutable = directHalation.descriptor != nullptr;
        const bool boostExecutable = input.identity.highlightBoostHash != 0;
        const bool spatialOpticsActive = cameraDiffusionActive || halationExecutable;
        const bool cameraFilmLinearActive = spatialOpticsActive || boostExecutable;
        DiffusionStageBinding directCameraDiffusion{};
        bind_diffusion(preparedFrame, frame, directCameraDiffusion, nullptr, errors);
        const JuicerProcess::Root::PreparedCudaFrame::FocusedPreparedView prepared =
            preparedFrame.focused_resources();
        if (!prepared.active ||
            prepared.selectedMethod != input.film.method) {
            errors.fail_route("ResourceDescriptorMismatch phase=3C field=direct_prepared_view");
        }

        JuicerCuda::FilmScanRunParams run{};
        bind_run_images(run, frame);

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

        bind_run_film(run, directFilmPayloads);
        auto directCameraFilmLinear = expose_camera(CaptureRoute::Direct, run, preparedFrame, input.identity, frame, snapshot, cameraDiffusionActive, directCameraDiffusion, directHalation, errors);
        const CaptureExecution execution = (!spatialOpticsActive && descriptors.dir.hash != 0 && descriptors.post.active() && !captureDensityConsumerActive)
                                               ? CaptureExecution::FusedDirScanner
                                           : (spatialOpticsActive || descriptors.post.active() || captureDensityConsumerActive || gateOutputActive)
                                               ? CaptureExecution::MaterializedDensity
                                               : CaptureExecution::Monolithic;
        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker focusedWorkspace =
            preparedFrame.workspace_lease();
        if (focusedWorkspace.active()) {
            JuicerCuda::Failure transitionError;
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
        std::optional<DirBuildResult> dirBuild;
        if (descriptors.dir.hash != 0) {
            JuicerCuda::Failure spatialError;
            if (!preparedFrame.prepare_spatial_dir_resources(
                    descriptors.dir,
                    focusedWorkspace,
                    frame.stream,
                    spatialError,
                    input.hostPreparation)) {
                errors.fail_submission(
                    "direct_spatial_dir_prepare",
                    "direct spatial DIR preparation failed",
                    spatialError);
            }
        }
        const EffectsPayloads effects = pack_effects_payloads(preparedFrame, descriptors.effects, errors);
        OpticalWorkspace optical;
        if (execution != CaptureExecution::Monolithic) {
            optical = bind_optical_workspace(CaptureRoute::Direct, preparedFrame, focusedWorkspace, frame.stream, descriptors.effects, errors);
        }
        GrainBinding grain;
        if (grainModel == GrainModel::GrainV2) {
            bind_grain_v2(preparedFrame, *input.preparation.grainV2, frame.renderWindow, CaptureRoute::Direct, grain, errors);
        } else if (grainModel == GrainModel::Legacy) {
            bind_legacy_grain(preparedFrame, focusedWorkspace, input.grain, grain, errors);
        }
        if (descriptors.dir.hash != 0) {
            dirBuild = build_spatial_dir(CaptureRoute::Direct, run, preparedFrame, focusedWorkspace, descriptors.dir, directCameraFilmLinear, cameraFilmLinearActive, frame.stream, execution, errors);
        }
        pack_scan_stage(run.scanStage, prepared, input.outputGamut, scannerCorrection, errors);
        bind_scan_error_readback(CaptureRoute::Direct, preparedFrame, run.scanStage, frame.stream, errors);
        const ScannerInput scannerInput = [&]() -> ScannerInput {
            if (execution == CaptureExecution::Monolithic) {
                return MonolithicScan{directCameraFilmLinear};
            }
            if (execution == CaptureExecution::FusedDirScanner) {
                return FusedDirScanner{directCameraFilmLinear};
            }
            develop_capture(CaptureRoute::Direct, run, directCameraFilmLinear, cameraFilmLinearActive, descriptors.dir, optical, grainModel, grain, filmEffectsActive, effects, grainDebugActive, frame.stream, errors);
            return MaterializedDensity{optical.density};
        }();
        if (cameraDiffusionActive) {
            JuicerCuda::Failure diffusionReleaseError;
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
            build_gate_mask(CaptureRoute::Direct, effects, optical, frame.stream, errors);
        }
        scan_direct_output(
            run,
            *prepared.scannerSpectra,
            preparedFrame,
            focusedWorkspace,
            descriptors,
            optical,
            effects,
            scannerInput,
            grainDebugActive,
            dirBuild,
            gateOutputActive,
            frame,
            errors);
        finish_scan_error_readback(CaptureRoute::Direct, preparedFrame, run.scanStage, input.identity, directHalation, snapshot, frame.stream, errors);
        if (boostExecutable) {
            check_highlight_reconstruction(preparedFrame, input.identity, errors);
        }
        finish_frame(CaptureRoute::Direct, preparedFrame, input.identity, directHalation, snapshot, frame.stream, abortCallback, errors);
    }


} // namespace JuicerCuda
