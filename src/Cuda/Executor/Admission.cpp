#include "Cuda/Executor/Stages.h"


namespace JuicerCuda {

    const char* nonempty_cstr_or(const char* value, const char* fallback) {
        return (value && value[0] != '\0') ? value : fallback;
    }

    const char* detail_or_unknown(const char* detail) {
        return nonempty_cstr_or(detail, "(unknown)");
    }

    inline const char* cstr_or_null_if_empty(const std::string& value) {
        return value.empty() ? nullptr : value.c_str();
    }

    inline int bool_to_i32(bool value) {
        return value ? 1 : 0;
    }
    struct CudaFailureTrace {
        const char* prefix = "CUDA operation failed";
        const char* detail = nullptr;
    };

    static void copy_float3(float dst[3], const float src[3]) {
        std::memcpy(dst, src, 3u * sizeof(float));
    }

    static void trace_cuda_fatal_prefixed_if(
        bool traceEnabled,
        const CudaFailureTrace& failure) noexcept try {
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
    } catch (...) {
        JuicerLogging::discard_current_exception();
    }

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

    void PendingContextLossRecovery::observe(const char* stageTag, const Failure& observed) noexcept {
        if (!lossStage && context_loss_status(observed).has_value()) {
            lossStage = stageTag;
        }
    }

    bool pack_scan_color(
        ScanColorPayload& destination,
        const JuicerProcess::Root::PreparedCudaFrame::FocusedPreparedView& prepared,
        const OutputGamutRecipe& recipe) {
        if (!prepared.scannerColor) {
            return false;
        }
        const Scanner::ColorRuntime& color = *prepared.scannerColor;
        copy_float9(destination.cat02, color.cat02);
        copy_float9(destination.xyzToRgb, color.xyzToRgb);
        copy_float3(destination.illuminantXYZ, color.illuminantXYZ);
        destination.encoding.applyCctfEncoding = bool_to_i32(color.encoding.applyCctfEncoding);
        destination.encoding.inputIsOutputSpace = bool_to_i32(color.encoding.inputIsOutputSpace);
        const auto& encoding = color.preparedEncoding;
        destination.encoding.cctf.kind = static_cast<int>(encoding.cctf_kind);
        destination.encoding.cctf.gamma = encoding.gamma;
        destination.encoding.cctf.a = encoding.a;
        destination.encoding.cctf.b = encoding.b;
        destination.encoding.cctf.c = encoding.c;
        destination.encoding.cctf.d = encoding.d;
        destination.encoding.cctf.linearCutoff = encoding.linear_cutoff;
        copy_float9(destination.encoding.dwgToOutput, encoding.dwg_to_output);
        destination.outputGamutActive = 0;
        destination.outputGamutCmax = nullptr;
        if (!recipe.enabled) {
            return true;
        }
        if (!prepared.outputGamutTransform ||
            !prepared.outputGamutCmax) {
            return false;
        }
        destination.outputGamutCmax = prepared.outputGamutCmax;
        copy_float9(
            destination.outputGamutNativeRgbToD65Xyz,
            prepared.outputGamutTransform->nativeRgbToD65Xyz.data());
        copy_float9(
            destination.outputGamutD65XyzToNativeRgb,
            prepared.outputGamutTransform->d65XyzToNativeRgb.data());
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
        return true;
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

        const auto width = std::max<std::int64_t>(0, static_cast<std::int64_t>(meterBounds.x2) - meterBounds.x1);
        const auto height = std::max<std::int64_t>(0, static_cast<std::int64_t>(meterBounds.y2) - meterBounds.y1);
        const auto longEdge = std::max(width, height);
        if (longEdge > std::numeric_limits<int>::max()) {
            return descriptor;
        }
        descriptor.previewWidth = static_cast<int>(width);
        descriptor.previewHeight = static_cast<int>(height);
        if (longEdge > JuicerCuda::AutoExposurePreviewDescriptor::kMaxLongEdge) {
            if (width >= height) {
                descriptor.previewWidth = JuicerCuda::AutoExposurePreviewDescriptor::kMaxLongEdge;
                descriptor.previewHeight = static_cast<int>(std::max<std::int64_t>(
                    1,
                    (height * JuicerCuda::AutoExposurePreviewDescriptor::kMaxLongEdge + width / 2) /
                        width));
            } else {
                descriptor.previewHeight = JuicerCuda::AutoExposurePreviewDescriptor::kMaxLongEdge;
                descriptor.previewWidth = static_cast<int>(std::max<std::int64_t>(
                    1,
                    (width * JuicerCuda::AutoExposurePreviewDescriptor::kMaxLongEdge + height / 2) /
                        height));
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


    void check_abort(FjAbortCallback callback) {
        if (!callback.query) {
            return;
        }
        switch (callback.query(callback.user)) {
            case FJ_ABORT_CONTINUE:
                return;
            case FJ_ABORT_REQUESTED:
                throw ExecutionFailure{{{FJ_STATUS_CANCELLED, FJ_API_NONE, 0}, "CUDA render cancelled"}};
            default:
                throw ExecutionFailure{{{FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0}, "invalid CUDA abort query result"}};
        }
    }

    void execute_prepared(JuicerProcess::Root& root, const PreparedExecutionInput& input, PendingContextLossRecovery& recovery, FjAbortCallback abortCallback) {
        const bool print = Spektrafilm::scan_route_is_print(input.descriptors.route);
        const ExecutionErrors errors{recovery, input.frame.traceInfo, print ? "CUDA print route blocked" : "CUDA direct route blocked"};
        if (!print && input.frame.autoExposureDescriptor.method == Spektrafilm::AutoExposureMethod::Median) {
            errors.fail_route(Spektrafilm::kQuantizedMedianNotAcceptedForPhase3);
        }
        if (!(input.frame.components == 3 || input.frame.components == 4)) {
            errors.fail_route(print ? "UnsupportedPrintComponentCountForPhase4C" : "UnsupportedDirectComponentCountForPhase3C");
        }
        if (!visual_grain_full_frame_preflight(input.descriptors.grain, input.frame.renderWindow, input.frame.sourceBounds, input.frame.fullFrameExtent) ||
            !film_juicer_effects_full_frame_preflight(input.descriptors.effects, input.frame.renderWindow, input.frame.sourceBounds, input.frame.fullFrameExtent)) {
            errors.fail_route("ResourceDescriptorMismatch component=prepared_frame field=full_frame_domain");
        }
        const RouteDescriptors descriptors{input.descriptors.post, input.descriptors.spatialDir, input.descriptors.grain, input.descriptors.effects};
        const ScheduleInput schedule{input.frame, input.snapshot, input.preparation.identity, descriptors, input.film, input.filmRouteCorrectionScale, input.descriptors.grainPayload, input.descriptors.outputGamut, input.glareHash, input.cameraAutoEnabled, input.frame.autoExposureDescriptor.method, input.preparation, input.descriptors.correction, input.printExposure, input.preflashExposure, input.hostPreparation};
        if (Spektrafilm::scan_route_is_print(input.descriptors.route)) {
            execute_print_schedule(root, schedule, recovery, abortCallback);
        } else {
            execute_direct_schedule(root, schedule, recovery, abortCallback);
        }
    }

    void finish_frame(
        CaptureRoute route,
        JuicerProcess::Root::PreparedCudaFrame& prepared,
        const JuicerProcess::Root::PreparationIdentity& identity,
        const ScatterHalationPreparedView& halation,
        const ResourceManager::SubmissionSnapshot& snapshot,
        void* stream,
        FjAbortCallback callback,
        const ExecutionErrors& errors) {
        check_abort(callback);
        Failure failure;
        if (!prepared.record_use(stream, failure)) {
            errors.fail_submission("prepared_frame_use_fence", "CUDA prepared-frame use fencing failed", failure);
        }
        if (!prepared.finish(stream, failure)) {
            if (halation.descriptor) {
                annotate_halation_completion(failure, identity, halation, snapshot, prepared.admitted_context_epoch(), "prepared_frame_finish");
            }
            const bool print = route == CaptureRoute::Print;
            errors.fail_submission(print ? "print_prepared_frame_finish" : "direct_prepared_frame_finish",
                                   print ? "print prepared frame finish failed" : "direct prepared frame finish failed",
                                   failure);
        }
    }

    [[noreturn]] void ExecutionErrors::fail_policy(const char* failurePrefix, JuicerCuda::Failure failure) const {
        trace_cuda_fatal_prefixed_if(
            traceInfo,
            CudaFailureTrace{
                nonempty_cstr_or(failurePrefix, "CUDA render cannot continue"),
                cstr_or_null_if_empty(failure.diagnostic)});
        throw JuicerCuda::ExecutionFailure{std::move(failure)};
    }

    [[noreturn]] void ExecutionErrors::fail_stage(const char* stageTag, const char* failurePrefix, cudaError_t errorCode) const {
        const char* detail = detail_or_unknown(cudaGetErrorString(errorCode));
        Failure failure;
        set_failure(failure, runtime_failure_status(errorCode), detail);
        pendingContextLossRecovery.observe(nonempty_cstr_or(stageTag, "cuda_stage"), failure);
        fail_policy(nonempty_cstr_or(failurePrefix, "CUDA stage failed"), std::move(failure));
    }

    [[noreturn]] void ExecutionErrors::fail_grain_stage(JuicerCuda::GrainFailureSite site, cudaError_t status) const {
        const char* detail = detail_or_unknown(cudaGetErrorString(status));
        std::array<char, 512> message{};
        const auto length = std::snprintf(message.data(), message.size(), "%s: %s", site.prefix, detail);
        JuicerCuda::Failure failure;
        JuicerCuda::set_failure(failure, JuicerCuda::runtime_failure_status(status), length < 0 ? site.prefix : message.data());
        pendingContextLossRecovery.observe(site.stage, failure);
        try {
            trace_cuda_fatal_prefixed_if(traceInfo, {site.prefix, detail});
        } catch (...) {
            // Diagnostic storage or tracing must not replace the CUDA status.
            JuicerLogging::discard_current_exception();
        }
        throw JuicerCuda::ExecutionFailure{std::move(failure)};
    }

    [[noreturn]] void ExecutionErrors::fail_submission(const char* stageTag, const char* failurePrefix, const JuicerCuda::Failure& failure, bool deferredDirError) const {
        pendingContextLossRecovery.observe(nonempty_cstr_or(stageTag, "submission_stage"), failure);
        trace_cuda_fatal_prefixed_if(
            traceInfo,
            CudaFailureTrace{
                nonempty_cstr_or(failurePrefix, "submission stage failed"),
                cstr_or_null_if_empty(failure.diagnostic)});
        Failure raised;
        set_failure(raised, failure.status, failure.diagnostic);
        raised.cleanupContextLoss = failure.cleanupContextLoss;
        throw ExecutionFailure{std::move(raised), deferredDirError};
    }

    [[noreturn]] void ExecutionErrors::fail_route(const char* diagnostic) const {
        Failure failure;
        set_failure(failure, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, detail_or_unknown(diagnostic));
        fail_policy(restrictionPrefix, std::move(failure));
    }

} // namespace JuicerCuda
