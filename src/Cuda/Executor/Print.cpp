#include "Cuda/Executor/Stages.h"

#include "Cuda/Scan/PrintDevelop.h"

namespace JuicerCuda {

    static JuicerCuda::EnlargerPrintLinearExposurePlanes
    enlarger_print_linear_planes(
        const JuicerCuda::Diffusion::StagePlaneSet& planes) noexcept {
        return {
            planes.redSensitive,
            planes.greenSensitive,
            planes.blueSensitive,
            planes.rowStrideFloats};
    }

    static void develop_print_medium(
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
            check_diffusion_launch(Diffusion::launch_stage(diffusionLaunch), "print enlarger diffusion launch failed", errors);
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

    void execute_print_schedule(JuicerProcess::Root& root, const ScheduleInput& input, PendingContextLossRecovery& recovery, FjAbortCallback abortCallback) {
        const auto& frame = input.frame;
        const auto& requestedSnapshot = input.snapshot;
        const auto& deviceContextKey = requestedSnapshot.deviceContextKey;
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
        const bool enlargerDiffusionActive =
            frame.diffusionFrameSet.has_value() &&
            frame.diffusionFrameSet->enlarger.has_value();
        const bool routeDiffusionActive =
            cameraDiffusionActive || enlargerDiffusionActive;
        const bool captureDensityConsumerActive =
            grainModel != GrainModel::Disabled || filmEffectsActive;
        JuicerCuda::Failure prepareError;
        check_abort(abortCallback);
        JuicerProcess::Root::PreparedCudaFrame preparedFrame =
            root.prepare_cuda_frame(deviceContextKey, requestedSnapshot, input.preparation, autoExposureBufferRequest, frame.stream, recovery, prepareError, input.hostPreparation);
        if (!preparedFrame.active()) {
            errors.fail_submission(
                preparedFrame.failure_stage_tag(),
                preparedFrame.failure_prefix(),
                prepareError);
        }

        check_abort(abortCallback);

        const auto& snapshot = preparedFrame.admitted_snapshot();

        JuicerCuda::ScatterHalationPreparedView printHalation =
            preparedFrame.scatter_halation_resources();
        const bool halationExecutable = printHalation.descriptor != nullptr;
        const bool boostExecutable = input.identity.highlightBoostHash != 0;
        const bool cameraFilmLinearActive =
            cameraDiffusionActive || halationExecutable || boostExecutable;
        DiffusionStageBinding printCameraDiffusion{};
        DiffusionStageBinding printEnlargerDiffusion{};
        bind_diffusion(preparedFrame, frame, printCameraDiffusion, &printEnlargerDiffusion, errors);
        const JuicerProcess::Root::PreparedCudaFrame::FocusedPreparedView prepared =
            preparedFrame.focused_resources();
        const JuicerProcess::Root::PreparedCudaFrame::PrintPreparedView preparedPrint =
            preparedFrame.print_resources();
        if (!prepared.active || !preparedPrint.active ||
            prepared.selectedMethod != input.film.method) {
            errors.fail_route(
                "ResourceDescriptorMismatch phase=4C field=print_prepared_view");
        }
        const auto scannerCorrection = input.correction;
        JuicerCuda::PrintPipelineRunParams run{};
        bind_run_images(run, frame);

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
                input.preflashExposure,
                preparedPrint,
                scannerCorrection.exposureScale,
                printPayloads,
                payloadDiagnostic)) {
            errors.fail_route(payloadDiagnostic.c_str());
        }
        bind_run_film(run, filmPayloads);
        run.printExpose = printPayloads.expose;
        run.printDevelop = printPayloads.develop;
        auto printCameraFilmLinear = expose_camera(CaptureRoute::Print, run, preparedFrame, input.identity, frame, snapshot, cameraDiffusionActive, printCameraDiffusion, printHalation, errors);
        const CaptureExecution execution = (!routeDiffusionActive && !halationExecutable && descriptors.dir.hash != 0 && descriptors.post.active() && !captureDensityConsumerActive)
                                               ? CaptureExecution::FusedDirScanner
                                           : (routeDiffusionActive || halationExecutable || descriptors.post.active() || captureDensityConsumerActive || gateOutputActive)
                                               ? CaptureExecution::MaterializedDensity
                                               : CaptureExecution::Monolithic;
        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker focusedWorkspace =
            preparedFrame.workspace_lease();
        if (focusedWorkspace.active()) {
            JuicerCuda::Failure transitionError;
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
                    "print_spatial_dir_prepare",
                    "print spatial DIR preparation failed",
                    spatialError);
            }
        }
        const EffectsPayloads effects = pack_effects_payloads(preparedFrame, descriptors.effects, errors);
        OpticalWorkspace optical;
        if (execution != CaptureExecution::Monolithic) {
            optical = bind_optical_workspace(CaptureRoute::Print, preparedFrame, focusedWorkspace, frame.stream, descriptors.effects, errors);
        }
        GrainBinding grain;
        if (grainModel == GrainModel::GrainV2) {
            bind_grain_v2(preparedFrame, *input.preparation.grainV2, frame.renderWindow, CaptureRoute::Print, grain, errors);
        } else if (grainModel == GrainModel::Legacy) {
            bind_legacy_grain(preparedFrame, focusedWorkspace, input.grain, grain, errors);
        }
        if (descriptors.dir.hash != 0) {
            dirBuild = build_spatial_dir(CaptureRoute::Print, run, preparedFrame, focusedWorkspace, descriptors.dir, printCameraFilmLinear, cameraFilmLinearActive, frame.stream, execution, errors);
        }
        pack_scan_stage(run.scanStage, prepared, input.outputGamut, scannerCorrection, errors);
        bind_scan_error_readback(CaptureRoute::Print, preparedFrame, run.scanStage, frame.stream, errors);
        const ScannerInput scannerInput = [&]() -> ScannerInput {
            if (execution == CaptureExecution::Monolithic) {
                return MonolithicScan{printCameraFilmLinear};
            }
            if (execution == CaptureExecution::FusedDirScanner) {
                return FusedDirScanner{printCameraFilmLinear};
            }
            develop_capture(CaptureRoute::Print, run, printCameraFilmLinear, cameraFilmLinearActive, descriptors.dir, optical, grainModel, grain, filmEffectsActive, effects, grainDebugActive, frame.stream, errors);
            if (!grainDebugActive) {
                develop_print_medium(run, preparedFrame, printEnlargerDiffusion, enlargerDiffusionActive, optical, frame.stream, errors);
            }
            return MaterializedDensity{optical.density};
        }();
        if (routeDiffusionActive) {
            JuicerCuda::Failure diffusionReleaseError;
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
            build_gate_mask(CaptureRoute::Print, effects, optical, frame.stream, errors);
        }
        scan_print_output(
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
            input.glareHash,
            snapshot,
            errors);
        finish_scan_error_readback(CaptureRoute::Print, preparedFrame, run.scanStage, input.identity, printHalation, snapshot, frame.stream, errors);
        if (boostExecutable) {
            check_highlight_reconstruction(preparedFrame, input.identity, errors);
        }
        finish_frame(CaptureRoute::Print, preparedFrame, input.identity, printHalation, snapshot, frame.stream, abortCallback, errors);
    }

} // namespace JuicerCuda
