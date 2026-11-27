#include "Cuda/Executor/Stages.h"

#include "Cuda/Scan/SpectralScan.h"
#include "Cuda/Scan/PostEffects.h"

namespace JuicerCuda {

    void pack_scan_stage(
        JuicerCuda::ScanStagePayload& scanStage,
        const JuicerProcess::Root::PreparedCudaFrame::FocusedPreparedView& prepared,
        const OutputGamutRecipe& outputGamut,
        const Scanner::ScannerColorCorrectionDescriptor& scannerCorrection,
        const ExecutionErrors& errors) {
        if (!JuicerCuda::pack_scan_color(scanStage.scanColor, prepared, outputGamut)) {
            errors.fail_submission(
                "pack_scan_color",
                "CUDA scanner colour payload binding failed",
                JuicerCuda::Failure{{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0},
                                    "ResourceDescriptorMismatch component=scan_color_payload field=output_gamut_identity"});
        }
        scanStage.correctionActive = scannerCorrection.active ? 1 : 0;
        scanStage.correctionSlope = scannerCorrection.xyzSlope;
        scanStage.correctionOffset = scannerCorrection.xyzOffset;
    }

    void scan_direct_output(
        JuicerCuda::FilmScanRunParams& run,
        const JuicerCuda::ScannerSpectra& spectra,
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& focusedWorkspace,
        const RouteDescriptors& descriptors,
        const OpticalWorkspace& optical,
        const EffectsPayloads& effects,
        const ScannerInput& scannerInput,
        bool grainDebugActive,
        const std::optional<DirBuildResult>& dirBuild,
        bool gateOutputActive,
        const JuicerCuda::ExecutionFrame& frame,
        const ExecutionErrors& errors) {
        cudaError_t launchError = cudaSuccess;

        if (!std::holds_alternative<MonolithicScan>(scannerInput)) {
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
                if (const auto* fused = std::get_if<FusedDirScanner>(&scannerInput)) {
                    JTRACE_VERBOSE("CUDA_SCAN", "route=direct input=fused_dir");
                    launchError = juicer_cuda_direct_focused_scan_linear_rgb(
                        &run,
                        spectra,
                        optical.rgb.r,
                        optical.rgb.g,
                        optical.rgb.b,
                        frame.stream,
                        fused->cameraFilmLinear);
                } else {
                    const auto& density = std::get<MaterializedDensity>(scannerInput).planes;
                    JTRACE_VERBOSE("CUDA_SCAN", "route=direct input=developed_planes");
                    launchError = juicer_cuda_direct_focused_scan_linear_density_rgb(
                        &run,
                        spectra,
                        density.c,
                        density.m,
                        density.y,
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
            if (dirBuild && dirBuild->uses_source_cache()) {
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
            JTRACE_VERBOSE("CUDA_SCAN", "route=direct input=source");
            launchError = juicer_cuda_negative_direct_pipeline(
                &run,
                spectra,
                frame.stream,
                std::get<MonolithicScan>(scannerInput).cameraFilmLinear);
        }
        if (launchError != cudaSuccess) {
            errors.fail_stage("direct_negative_pipeline_launch", "direct negative pipeline launch failed", launchError);
        }
        if (descriptors.dir.hash != 0 && std::holds_alternative<MonolithicScan>(scannerInput)) {
            run.filmDevelop.spatialDir.logRawB = nullptr;
            run.filmDevelop.spatialDir.logRawG = nullptr;
            run.filmDevelop.spatialDir.logRawR = nullptr;
        }
    }

    void scan_print_output(
        JuicerCuda::PrintPipelineRunParams& run,
        const JuicerCuda::ScannerSpectra& spectra,
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& focusedWorkspace,
        const RouteDescriptors& descriptors,
        const OpticalWorkspace& optical,
        const EffectsPayloads& effects,
        const ScannerInput& scannerInput,
        bool grainDebugActive,
        const std::optional<DirBuildResult>& dirBuild,
        bool gateOutputActive,
        const JuicerCuda::ExecutionFrame& frame,
        std::uint64_t glareHash,
        const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        const ExecutionErrors& errors) {
        cudaError_t launchError = cudaSuccess;
        const auto& srcBounds = frame.sourceBounds;
        if (!std::holds_alternative<MonolithicScan>(scannerInput)) {
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
                {glareHash,
                 snapshot.frameToken.value,
                 static_cast<std::uint64_t>(srcBounds.x1),
                 static_cast<std::uint64_t>(srcBounds.y1)});
            if (!grainDebugActive) {
                if (const auto* fused = std::get_if<FusedDirScanner>(&scannerInput)) {
                    JTRACE_VERBOSE("CUDA_SCAN", "route=print input=fused_dir");
                    launchError = juicer_cuda_print_focused_scan_linear_rgb(
                        &run,
                        spectra,
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
                        frame.stream,
                        fused->cameraFilmLinear);
                } else {
                    const auto& density = std::get<MaterializedDensity>(scannerInput).planes;
                    JTRACE_VERBOSE("CUDA_SCAN", "route=print input=developed_planes");
                    launchError = juicer_cuda_print_focused_scan_linear_density_rgb(
                        &run,
                        spectra,
                        density.c,
                        density.m,
                        density.y,
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
            if (dirBuild && dirBuild->uses_source_cache()) {
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
            JTRACE_VERBOSE("CUDA_SCAN", "route=print input=source");
            launchError = juicer_cuda_print_focused_pipeline(
                &run,
                spectra,
                frame.stream,
                std::get<MonolithicScan>(scannerInput).cameraFilmLinear);
        }
        if (launchError != cudaSuccess) {
            errors.fail_stage(
                "print_pipeline_launch",
                "focused print pipeline launch failed",
                launchError);
        }
        if (descriptors.dir.hash != 0 && std::holds_alternative<MonolithicScan>(scannerInput)) {
            run.filmDevelop.spatialDir.logRawB = nullptr;
            run.filmDevelop.spatialDir.logRawG = nullptr;
            run.filmDevelop.spatialDir.logRawR = nullptr;
        }
    }

    void bind_scan_error_readback(
        CaptureRoute route,
        JuicerProcess::Root::PreparedCudaFrame& frame,
        ScanStagePayload& scan,
        void* stream,
        const ExecutionErrors& errors) {
        using Result = JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult;
        Failure failure;
        const auto result = frame.prepare_scan_error_stage(scan.scanErrorFlag, stream, failure);
        if (result != Result::Ready) {
            const bool print = route == CaptureRoute::Print;
            errors.fail_submission(print ? "print_scan_error_stage" : "direct_scan_error_stage",
                                   print ? "print scan error stage failed" : "direct scan error stage failed",
                                   failure,
                                   result == Result::DeferredDirFailure);
        }
    }

    void finish_scan_error_readback(
        CaptureRoute route,
        JuicerProcess::Root::PreparedCudaFrame& frame,
        ScanStagePayload& scan,
        const JuicerProcess::Root::PreparationIdentity& identity,
        const ScatterHalationPreparedView& halation,
        const ResourceManager::SubmissionSnapshot& snapshot,
        void* stream,
        const ExecutionErrors& errors) {
        Failure failure;
        if (!frame.finalize_scan_error_stage(scan.scanErrorFlag, stream, failure)) {
            if (halation.descriptor) {
                annotate_halation_completion(failure, identity, halation, snapshot, frame.admitted_context_epoch(), "scan_error_finalize");
            }
            const bool print = route == CaptureRoute::Print;
            errors.fail_submission(print ? "print_scan_error_finalize" : "direct_scan_error_finalize",
                                   print ? "print scan error finalize failed" : "direct scan error finalize failed",
                                   failure);
        }
    }


} // namespace JuicerCuda
