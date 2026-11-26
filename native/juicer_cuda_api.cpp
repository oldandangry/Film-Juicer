#include "juicer_cuda_api.h"
#include "juicer_cuda_owner.h"
#include "juicer_cuda_prepared.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_runtime.h>

#include "Cuda/JuicerCudaDriver.h"
#include "Cuda/JuicerCudaFailure.h"
#include "Cuda/JuicerCudaResourceOperations.h"
#include "ProcessRoot.h"
#include "Logging.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"

struct FjCuda final {
    enum class Lifecycle : std::uint8_t {
        Accepting,
        Closing,
        Closed,
        Blocked,
        Retained
    };

    ~FjCuda() {
    }

    FjStatus save_failure(FjStatus result, std::string_view diagnostic) noexcept {
        terminalStatus = result;
        FjErrorBuffer error{terminalDiagnostic.data(), terminalDiagnostic.size(), 0};
        JuicerCuda::write_status(result, diagnostic.empty() ? "CUDA terminal operation failed; native graph retained" : diagnostic, &error);
        terminalDiagnosticLength = error.length;
        lifecycle.store(Lifecycle::Blocked, std::memory_order_release);
        lifecycle.notify_all();
        return result;
    }

    FjStatus block(FjStatus result, std::string_view diagnostic) noexcept {
        auto state = lifecycle.load(std::memory_order_acquire);
        for (;;) {
            if (state == Lifecycle::Blocked || state == Lifecycle::Retained) {
                return terminalStatus;
            }
            if (state == Lifecycle::Closing) {
                lifecycle.wait(state, std::memory_order_acquire);
            } else if (lifecycle.compare_exchange_weak(state, Lifecycle::Closing, std::memory_order_acq_rel)) {
                break;
            }
            state = lifecycle.load(std::memory_order_acquire);
        }
        (void)root.stop_frame_preparation();
        return save_failure(result, diagnostic);
    }

    FjStatus shutdown() noexcept {
        auto state = lifecycle.load(std::memory_order_acquire);
        for (;;) {
            if (state == Lifecycle::Closed) {
                return {FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
            }
            if (state == Lifecycle::Blocked || state == Lifecycle::Retained) {
                return terminalStatus;
            }
            if (state == Lifecycle::Closing) {
                lifecycle.wait(state, std::memory_order_acquire);
            } else if (lifecycle.compare_exchange_weak(state, Lifecycle::Closing, std::memory_order_acq_rel)) {
                break;
            }
            state = lifecycle.load(std::memory_order_acquire);
        }
        // Closing rejects admission but does not publish a diagnostic. Only a
        // completed failure publishes immutable storage to acquire-readers,
        // including terminal callers whose owner-mutex acquisition failed.
        JuicerCuda::Failure failure;
        try {
            if (!root.stop_frame_preparation()) {
                return save_failure({FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "CUDA shutdown could not close frame admission");
            }
            if (root.shutdown(callMutex, failure)) {
                lifecycle.store(Lifecycle::Closed, std::memory_order_release);
                lifecycle.notify_all();
                return {FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
            }
            return save_failure(failure.status, failure.diagnostic.empty() ? "CUDA shutdown failed; native graph retained" : std::string_view(failure.diagnostic));
        } catch (const JuicerCuda::ExecutionFailure& detail) {
            return save_failure(detail.failure.status, detail.failure.diagnostic);
        } catch (const std::bad_alloc&) {
            return save_failure({FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0}, "CUDA shutdown allocation failed");
        } catch (const std::exception& detail) {
            return save_failure({FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, detail.what());
        } catch (...) {
            return save_failure({FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "CUDA shutdown failed with unknown exception");
        }
    }

    void retire_instance(std::uint64_t instanceToken) {
        root.retire_grain_static_instance(instanceToken);
    }

    std::atomic<Lifecycle> lifecycle{Lifecycle::Accepting};
    FjStatus terminalStatus{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
    std::array<char, 1024> terminalDiagnostic{};
    std::size_t terminalDiagnosticLength = 0;

    std::mutex callMutex;
    JuicerProcess::Root root;
};

namespace {

    std::mutex gOwnerMutex;
    constinit std::atomic<FjCuda*> gOwner{nullptr};
    thread_local bool gNativeCallActive = false;

    [[noreturn]] void reject_native_call(std::uint32_t category, std::string_view diagnostic) {
        JuicerCuda::Failure failure;
        JuicerCuda::set_failure(failure, {category, FJ_API_NONE, 0}, diagnostic);
        throw JuicerCuda::ExecutionFailure{std::move(failure)};
    }

    std::mutex& native_call_mutex(FjCuda* cuda) {
        JuicerCuda::check_native_call_admission(cuda);
        return cuda->callMutex;
    }


    bool invalid_render_arguments(const FjCudaContext* context, const FjFrame* frame, const FjSubmission* submission, const FjPreparedHostData* prepared) {
        return !context || !frame || !submission || !prepared ||
               context->device_id < 0 || context->context == 0 || context->epoch == 0 ||
               submission->instance_token == 0 || submission->submission_id == 0 ||
               !std::isfinite(frame->time_frames) || frame->time_frames < -0x1p63 || frame->time_frames >= 0x1p63 ||
               !std::isfinite(frame->frame_rate) || frame->frame_rate < 0 ||
               !std::isfinite(frame->pixel_size_um) || frame->pixel_size_um < 0;
    }

    bool valid_diagnostic(FjErrorBuffer* error) noexcept {
        if (!error) {
            return true;
        }
        error->length = 0;
        return error->capacity == 0 || (error->data && error->capacity <= static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()));
    }

    FjStatus status(uint32_t category, FjErrorBuffer* error, const char* message) noexcept {
        return JuicerCuda::write_status({category, FJ_API_NONE, 0}, message, error);
    }

    FjStatus terminal_admission_failure(FjCuda* cuda, FjStatus result, std::string_view diagnostic, bool consuming, FjErrorBuffer* error) noexcept {
        // The host excludes destruction while a handle is borrowed. Registration
        // establishes validity here even if acquiring gOwnerMutex failed.
        if (cuda != gOwner.load(std::memory_order_acquire)) {
            return JuicerCuda::write_status(result, diagnostic, error);
        }
        result = cuda->block(result, diagnostic);
        if (consuming) {
            cuda->lifecycle.store(FjCuda::Lifecycle::Retained, std::memory_order_release);
        }
        return JuicerCuda::write_status(result,
                                        {cuda->terminalDiagnostic.data(), cuda->terminalDiagnosticLength},
                                        error);
    }

    struct ImageBytes {
        std::uintptr_t first = 0;
        std::uintptr_t end = 0;
        std::size_t rowBytes = 0;
        std::size_t rows = 0;
        std::size_t stride = 0;
    };

    bool ordered_rect(const FjRect& rect) noexcept {
        return rect.x2 > rect.x1 && rect.y2 > rect.y1 &&
               static_cast<std::int64_t>(rect.x2) - rect.x1 <= std::numeric_limits<int>::max() &&
               static_cast<std::int64_t>(rect.y2) - rect.y1 <= std::numeric_limits<int>::max();
    }

    bool covers(const FjRect& bounds, const FjRect& window) noexcept {
        return bounds.x1 <= window.x1 && bounds.y1 <= window.y1 &&
               bounds.x2 >= window.x2 && bounds.y2 >= window.y2;
    }

    bool image_bytes(const FjImage& image, const FjRect& window, ImageBytes& pixels) noexcept {
        if (!image.address || image.depth != FJ_DEPTH_FLOAT32 ||
            (image.components != FJ_COMPONENTS_RGB && image.components != FJ_COMPONENTS_RGBA) ||
            !ordered_rect(image.bounds) || !covers(image.bounds, window) || image.row_bytes <= 0 ||
            image.address % alignof(float) != 0 || image.row_bytes % sizeof(float) != 0) {
            return false;
        }
        const auto width = static_cast<std::size_t>(static_cast<std::int64_t>(image.bounds.x2) - image.bounds.x1);
        const auto height = static_cast<std::size_t>(static_cast<std::int64_t>(image.bounds.y2) - image.bounds.y1);
        const auto stride = static_cast<std::size_t>(image.row_bytes);
        const auto pixelBytes = image.components * sizeof(float);
        constexpr auto limit = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
        if (width > limit / pixelBytes || width * pixelBytes > stride ||
            height - 1 > (limit - width * pixelBytes) / stride) {
            return false;
        }
        const auto extent = (height - 1) * stride + width * pixelBytes;
        if (extent > std::numeric_limits<std::uintptr_t>::max() - image.address) {
            return false;
        }
        pixels = {image.address, image.address + extent, width * pixelBytes, height, stride};
        return true;
    }

    bool overlapping_rows(const ImageBytes& source, const ImageBytes& destination) noexcept {
        std::size_t srcRow = 0;
        std::size_t dstRow = 0;
        while (srcRow < source.rows && dstRow < destination.rows) {
            const auto src = source.first + srcRow * source.stride;
            const auto dst = destination.first + dstRow * destination.stride;
            if (src < dst + destination.rowBytes && dst < src + source.rowBytes) {
                return true;
            }
            if (src + source.rowBytes <= dst) {
                srcRow += std::min(source.rows - srcRow, (dst - src - source.rowBytes) / source.stride + 1);
            } else {
                dstRow += std::min(destination.rows - dstRow, (src - dst - destination.rowBytes) / destination.stride + 1);
            }
        }
        return false;
    }

    FjStatus driver_failure(FjErrorBuffer* error, const std::string& message, int code) noexcept {
        return JuicerCuda::write_status(JuicerCuda::driver_failure_status(code), message, error);
    }


} // namespace

FjStatus fj_cuda_create(FjCuda** out_cuda, FjErrorBuffer* error) {
    if (out_cuda) {
        *out_cuda = nullptr;
    }
    if (!valid_diagnostic(error)) {
        return {FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0};
    }
    try {
        if (gNativeCallActive || !out_cuda) {
            return status(FJ_STATUS_UNSUPPORTED_INPUT, error, "invalid CUDA owner creation input");
        }
        auto candidate = std::make_unique<FjCuda>();
        bool published = false;
        {
            std::lock_guard<std::mutex> lock(gOwnerMutex);
            if (!gOwner.load(std::memory_order_acquire)) {
                *out_cuda = candidate.release();
                gOwner.store(*out_cuda, std::memory_order_release);
                published = true;
            }
        }
        // Rejected candidates release native metadata outside the registration lock.
        return status(published ? FJ_STATUS_SUCCESS : FJ_STATUS_PREPARATION_FAILURE, error, published ? "" : "CUDA runtime already has an owner");
    } catch (const JuicerCuda::ExecutionFailure& failure) {
        return JuicerCuda::write_status(failure.failure.status, failure.failure.diagnostic, error);
    } catch (const std::bad_alloc&) {
        return status(FJ_STATUS_ALLOCATION_FAILURE, error, "CUDA owner allocation failed");
    } catch (const std::exception& errorDetail) {
        return status(FJ_STATUS_INTERNAL_FAILURE, error, errorDetail.what());
    } catch (...) {
        return status(FJ_STATUS_INTERNAL_FAILURE, error, "CUDA owner creation failed");
    }
}

FjStatus fj_cuda_inspect(FjCuda* cuda, const FjFrame* frame, FjCudaContext* out_context, FjErrorBuffer* error) {
    if (!valid_diagnostic(error)) {
        return {FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0};
    }
    try {
        if (!cuda || !frame || !out_context) {
            return status(FJ_STATUS_UNSUPPORTED_INPUT, error, "invalid CUDA inspection arguments");
        }
        JuicerCuda::NativeCall call(cuda);
        return call.inspect(frame, out_context, error);
    } catch (const JuicerCuda::ExecutionFailure& failure) {
        return JuicerCuda::write_status(failure.failure.status, failure.failure.diagnostic, error);
    } catch (const std::bad_alloc&) {
        return status(FJ_STATUS_ALLOCATION_FAILURE, error, "CUDA inspection allocation failed");
    } catch (const std::exception& detail) {
        return status(FJ_STATUS_INTERNAL_FAILURE, error, detail.what());
    } catch (...) {
        return status(FJ_STATUS_INTERNAL_FAILURE, error, "CUDA inspection failed");
    }
}

static FjStatus inspect_frame(const FjFrame* frame, FjCudaContext* outContext, FjErrorBuffer* error, const FjCudaContext* expected = nullptr) {
    try {
        if (!frame || !outContext || (error && error->capacity && !error->data)) {
            return status(FJ_STATUS_UNSUPPORTED_INPUT, error, "invalid CUDA inspection arguments");
        }
        constexpr std::uint32_t knownFlags = FJ_FRAME_STREAM_PRESENT | FJ_FRAME_TRACE_INFO | FJ_FRAME_TRACE_VERBOSE;
        if ((frame->flags & ~knownFlags) != 0 ||
            (!(frame->flags & FJ_FRAME_STREAM_PRESENT) && frame->stream != 0) ||
            !ordered_rect(frame->render_window) || !ordered_rect(frame->full_frame_extent) ||
            frame->source.components != frame->destination.components) {
            return status(FJ_STATUS_UNSUPPORTED_INPUT, error, "invalid CUDA frame tags, stream or rectangle");
        }
        ImageBytes source, destination;
        if (!image_bytes(frame->source, frame->render_window, source)) {
            return status(FJ_STATUS_UNSUPPORTED_INPUT, error, "invalid CUDA source geometry, stride or address range");
        }
        if (!image_bytes(frame->destination, frame->render_window, destination)) {
            return status(FJ_STATUS_UNSUPPORTED_INPUT, error, "invalid CUDA destination geometry, stride or address range");
        }
        const auto& window = frame->render_window;
        const auto& image = frame->destination;
        // Per-image validation proved these subrange operations fit the full extent.
        const auto x = static_cast<std::size_t>(static_cast<std::int64_t>(window.x1) - image.bounds.x1);
        const auto y = static_cast<std::size_t>(static_cast<std::int64_t>(window.y1) - image.bounds.y1);
        const auto rows = static_cast<std::size_t>(static_cast<std::int64_t>(window.y2) - window.y1);
        const auto rowBytes = static_cast<std::size_t>(static_cast<std::int64_t>(window.x2) - window.x1) * image.components * sizeof(float);
        const auto first = image.address + y * destination.stride + x * image.components * sizeof(float);
        const ImageBytes destinationWindow{first, first + (rows - 1) * destination.stride + rowBytes, rowBytes, rows, destination.stride};
        // Auto exposure may read the complete source. Final scan may read source
        // pixels while writing destination pixels; no overlapping access is qualified.
        if (overlapping_rows(source, destinationWindow)) {
            return status(FJ_STATUS_UNSUPPORTED_INPUT, error, "overlapping CUDA source/destination pixel ranges");
        }
        if (expected && !JuicerCuda::ResourceManager::registry_accepts_epoch(
                            {expected->device_id, reinterpret_cast<void*>(expected->context)}, expected->epoch)) {
            return status(FJ_STATUS_PREPARATION_FAILURE, error, "CUDA render epoch is stale or retiring");
        }
        void* context = nullptr;
        int driverCode = 0;
        std::string diagnostic;
        if (!JuicerCuda::query_current_cuda_context(context, diagnostic, &driverCode)) {
            return driver_failure(error, diagnostic, driverCode);
        }
        outContext->context = reinterpret_cast<std::uintptr_t>(context);
        int device = -1;
        for (const auto& image : {source, destination}) {
            cudaPointerAttributes attributes{};
            const auto runtimeCode = cudaPointerGetAttributes(&attributes, reinterpret_cast<const void*>(image.first));
            if (runtimeCode != cudaSuccess) {
                return JuicerCuda::write_status(JuicerCuda::runtime_failure_status(runtimeCode), "cudaPointerGetAttributes failed", error);
            }
            if (attributes.type != cudaMemoryTypeDevice || attributes.device < 0) {
                return status(FJ_STATUS_UNSUPPORTED_INPUT, error, "CUDA image requires device memory");
            }
            if (device >= 0 && device != attributes.device) {
                return status(FJ_STATUS_UNSUPPORTED_INPUT, error, "source/destination CUDA device mismatch");
            }
            device = attributes.device;
            void* pointerContext = nullptr;
            if (!JuicerCuda::query_cuda_pointer_context(image.first, pointerContext, driverCode, diagnostic)) {
                return driver_failure(error, diagnostic, driverCode);
            }
            if (pointerContext != context) {
                return status(FJ_STATUS_UNSUPPORTED_INPUT, error, "CUDA image does not belong to the current context");
            }
            JuicerCuda::CudaAllocation allocation;
            if (!JuicerCuda::query_cuda_allocation(image.first, allocation, driverCode, diagnostic)) {
                return driver_failure(error, diagnostic, driverCode);
            }
            if (image.first < allocation.base || image.first - allocation.base > allocation.bytes ||
                image.end - image.first > allocation.bytes - (image.first - allocation.base)) {
                return status(FJ_STATUS_UNSUPPORTED_INPUT, error, "CUDA allocation does not cover image bounds");
            }
        }
        *outContext = FjCudaContext{device, reinterpret_cast<std::uintptr_t>(context), 0};
        return status(FJ_STATUS_SUCCESS, error, "");
    } catch (const std::bad_alloc&) {
        return status(FJ_STATUS_ALLOCATION_FAILURE, error, "CUDA inspection allocation failed");
    } catch (const std::exception& detail) {
        return status(FJ_STATUS_INTERNAL_FAILURE, error, detail.what());
    } catch (...) {
        return status(FJ_STATUS_INTERNAL_FAILURE, error, "CUDA inspection failed");
    }
}

FjStatus JuicerCuda::NativeCall::inspect(const FjFrame* frame, FjCudaContext* outContext, FjErrorBuffer* error) {
    FjCudaContext context{};
    const auto result = inspect_frame(frame, &context, error);
    if (result.category != FJ_STATUS_SUCCESS) {
        return result;
    }
    context.epoch = ResourceManager::registry_inspect_context(
        {context.device_id, reinterpret_cast<void*>(context.context)});
    if (context.epoch == 0) {
        return status(FJ_STATUS_PREPARATION_FAILURE, error, "CUDA context epoch is retiring");
    }
    *outContext = context;
    return result;
}

FjRenderOutcome fj_cuda_render(FjCuda* cuda, const FjCudaContext* context, const FjFrame* frame, const FjSubmission* submission, const FjPreparedHostData* prepared, const FjHostPreparation* host_preparation, FjAbortCallback abort_callback, FjErrorBuffer* error) {
    if (!valid_diagnostic(error)) {
        return {{FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0}, 0, {}};
    }
    try {
        // The fixture provider observes Rust's selection before the native gate.
        JuicerCuda::ResourceOperations::before_render_admission(cuda, context, submission);
        JuicerCuda::PendingContextLossRecovery recovery;
        std::string diagnostic;
        FjRenderOutcome outcome{};
        {
            JuicerCuda::NativeCall call(cuda);
            outcome = call.render(context, frame, submission, prepared, abort_callback, recovery, diagnostic, host_preparation);
            if (recovery.submissionAdmitted) {
                outcome.flags |= FJ_RENDER_ADMITTED;
            }
            if (recovery.lossStage) {
                const JuicerCuda::ResourceManager::DeviceContextKey key{
                    context->device_id, reinterpret_cast<void*>(context->context)};
                outcome.retirement = {*context, submission->instance_token, submission->submission_id};
                JuicerCuda::ResourceManager::registry_fail_epoch(key, outcome.retirement);
                outcome.flags |= FJ_RENDER_CONTEXT_LOSS_PENDING;
            }
        }
#if JUICER_DIAGNOSTICS_COMPILED
        if (recovery.submissionAdmitted && (frame->flags & FJ_FRAME_TRACE_VERBOSE) != 0) {
            try {
                JuicerLogging::write("CUDALIFE",
                                     "event=render_outcome snapshot_id=" + std::to_string(submission->submission_id) +
                                         " instance_token=" + std::to_string(submission->instance_token) +
                                         " status=" + std::to_string(outcome.status.category) +
                                         " scanner_hash=" + std::to_string(submission->scanner_hash) +
                                         " paper_version=" + std::to_string(prepared->print_profile_asset_version) +
                                         (recovery.lossStage ? std::string(" context_loss_stage=") + recovery.lossStage : ""),
                                     true);
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
        }
#endif
        JuicerCuda::write_status(outcome.status, diagnostic, error);
        return outcome;
    } catch (const JuicerCuda::ExecutionFailure& failure) {
        return {JuicerCuda::write_status(failure.failure.status, failure.failure.diagnostic, error), 0, {}};
    } catch (const std::bad_alloc&) {
        return {status(FJ_STATUS_ALLOCATION_FAILURE, error, "CUDA render allocation failed"), 0, {}};
    } catch (const std::exception& detail) {
        return {status(FJ_STATUS_INTERNAL_FAILURE, error, detail.what()), 0, {}};
    } catch (...) {
        return {status(FJ_STATUS_INTERNAL_FAILURE, error, "CUDA render failed with unknown exception"), 0, {}};
    }
}

FjRenderOutcome JuicerCuda::NativeCall::render(
    const FjCudaContext* context, const FjFrame* frame, const FjSubmission* submission, const FjPreparedHostData* prepared, FjAbortCallback abortCallback, PendingContextLossRecovery& recovery, std::string& diagnostic, const FjHostPreparation* hostPreparation) {
    diagnostic.clear();
    const auto result = [&](FjStatus selected, std::string_view message) -> FjRenderOutcome {
        try {
            diagnostic.assign(message);
        } catch (...) {
            diagnostic.clear();
        }
        return {selected, 0, {}};
    };
    try {
        if (invalid_render_arguments(context, frame, submission, prepared)) {
            return result({FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0}, "invalid CUDA render arguments");
        }
        auto admission = _cuda->root.begin_frame_preparation();
        if (!admission.active()) {
            return result({FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "CUDA frame preparation admission blocked");
        }
        FjCudaContext inspected{};
        std::array<char, 512> inspectionMessage{};
        FjErrorBuffer inspectionError{inspectionMessage.data(), inspectionMessage.size(), 0};
        const ResourceManager::DeviceContextKey key{context->device_id, reinterpret_cast<void*>(context->context)};
        const auto inspection = inspect_frame(frame, &inspected, &inspectionError, context);
        if (inspection.category != FJ_STATUS_SUCCESS) {
            // Attribute a revalidation failure only after the driver confirmed
            // the current context and the registry matched the inspected epoch.
            if (inspected.context == context->context) {
                recovery.observe("render_image_revalidation", {inspection, {}});
            }
            return result(inspection, {inspectionMessage.data(), inspectionError.length});
        }
        if (context->device_id != inspected.device_id || context->context != inspected.context) {
            return result({FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0}, "CUDA render context differs from inspected current context");
        }
        const auto sameRect = [](const FjRect& a, const FjRect& b) {
            return a.x1 == b.x1 && a.y1 == b.y1 && a.x2 == b.x2 && a.y2 == b.y2;
        };
        if (prepared->optics.flags != 0 &&
            (!sameRect(frame->source.bounds, frame->full_frame_extent) ||
             !sameRect(frame->destination.bounds, frame->full_frame_extent) ||
             !sameRect(frame->render_window, frame->full_frame_extent))) {
            return result({FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0}, "CUDA optics require the complete frame domain");
        }
        auto* const stream = reinterpret_cast<cudaStream_t>(frame->stream);
        if (stream && stream != cudaStreamLegacy && stream != cudaStreamPerThread) {
            void* streamContext = nullptr;
            int driverCode = 0;
            if (!JuicerCuda::query_cuda_stream_context(frame->stream, streamContext, driverCode, diagnostic)) {
                const auto failure = driver_failure_status(driverCode);
                recovery.observe("render_stream_revalidation", {failure, {}});
                return {failure, 0, {}};
            }
            if (reinterpret_cast<std::uintptr_t>(streamContext) != inspected.context) {
                return result({FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0}, "CUDA stream does not belong to the current context");
            }
        }
        const auto rect = [](const FjRect& value) -> JuicerCuda::FrameRect {
            return {value.x1, value.y1, value.x2, value.y2};
        };
        const auto address = [&](const FjImage& image) {
            const auto x = static_cast<std::ptrdiff_t>(frame->render_window.x1) - image.bounds.x1;
            const auto y = static_cast<std::ptrdiff_t>(frame->render_window.y1) - image.bounds.y1;
            return image.address + static_cast<std::uintptr_t>(y * image.row_bytes + x * image.components * sizeof(float));
        };
        const auto& geometry = frame->effects_geometry;
        const Spektrafilm::FilmJuicerEffectsGeometry effects{
            {geometry.pixel_definition.x, geometry.pixel_definition.y, geometry.pixel_definition.width, geometry.pixel_definition.height},
            geometry.canonical_x,
            geometry.canonical_y,
            geometry.canonical_width,
            geometry.canonical_height,
            geometry.scale_x,
            geometry.scale_y,
            geometry.pixel_aspect_ratio};
        const auto& meter = prepared->auto_exposure;
        const JuicerCuda::AutoExposurePreviewDescriptor metering{
            meter.source_bounds.x1, meter.source_bounds.y1, meter.source_bounds.x2, meter.source_bounds.y2, meter.meter_bounds.x1, meter.meter_bounds.y1, meter.meter_bounds.x2, meter.meter_bounds.y2, meter.preview_width, meter.preview_height, static_cast<Spektrafilm::AutoExposureMethod>(meter.method), meter.hash};
        const std::optional<Spektrafilm::DiffusionFrameSetDescriptor> diffusion;
        const std::optional<ScatterHalationFrameDescriptor> halation;
        const JuicerCuda::ExecutionFrame execution{
            rect(frame->source.bounds), rect(frame->render_window), rect(frame->full_frame_extent), reinterpret_cast<const unsigned char*>(frame->source.address), reinterpret_cast<const unsigned char*>(address(frame->source)), reinterpret_cast<unsigned char*>(address(frame->destination)), frame->source.row_bytes, frame->destination.row_bytes, static_cast<int>(frame->source.components), reinterpret_cast<void*>(frame->stream), diffusion, halation, effects, frame->pixel_size_um, frame->time_frames, frame->frame_rate, frame->session_seed, static_cast<std::uintptr_t>(frame->clip_token), metering, (frame->flags & FJ_FRAME_TRACE_INFO) != 0, (frame->flags & FJ_FRAME_TRACE_VERBOSE) != 0};
        JuicerCuda::ResourceManager::SubmissionSnapshot snapshot{
            {submission->instance_token}, {submission->frame_token}, submission->submission_id, key, {submission->upload_core_hash, submission->dir_hash, submission->scanner_hash, submission->auto_exposure_hash}, context->epoch};
        const auto complete = [&]() {
            Failure completion{{FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, {}};
            if (recovery.submissionAdmitted && (frame->flags & FJ_FRAME_STREAM_PRESENT) == 0) {
                const auto code = ResourceOperations::synchronize_stream(reinterpret_cast<cudaStream_t>(frame->stream));
                if (code != cudaSuccess) {
                    set_failure(completion, runtime_failure_status(code), cudaGetErrorString(code));
                    recovery.observe("absent_stream_completion", completion);
                }
            }
            return completion;
        };
        const auto completionResult = [&](Failure completion, std::uint32_t flags = 0) -> FjRenderOutcome {
            diagnostic = std::move(completion.diagnostic);
            return {completion.status, flags, {}};
        };
        try {
            if (!JuicerCuda::execute_prepared_host_data(_cuda->root, *prepared, execution, snapshot, recovery, abortCallback, diagnostic, hostPreparation)) {
                return {{FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0}, 0, {}};
            }
        } catch (const ExecutionFailure& failure) {
            recovery.observe("native_render", failure.failure);
            auto completion = complete();
            // A deferred report describes earlier work, independent of this
            // completion. A failed fence still keeps its own primary error.
            if ((failure.failure.status.category == FJ_STATUS_CANCELLED || failure.deferredDirError) &&
                completion.status.category != FJ_STATUS_SUCCESS) {
                return completionResult(std::move(completion), failure.deferredDirError ? FJ_RENDER_DEFERRED_DIR_ERROR : 0U);
            }
            throw;
        } catch (...) {
            auto completion = complete();
            if (completion.status.category != FJ_STATUS_SUCCESS) {
                return completionResult(std::move(completion));
            }
            throw;
        }
        return completionResult(complete());
    } catch (JuicerCuda::ExecutionFailure& failure) {
        diagnostic = std::move(failure.failure.diagnostic);
        return {failure.failure.status, failure.deferredDirError ? FJ_RENDER_DEFERRED_DIR_ERROR : 0U, {}};
    } catch (const std::bad_alloc&) {
        return result({FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0}, "CUDA render allocation failed");
    } catch (const std::exception& detail) {
        return result({FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, detail.what());
    } catch (...) {
        return result({FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "CUDA render failed with unknown exception");
    }
}

FjStatus fj_cuda_retire_context(FjCuda* cuda, const FjCudaRetirement* identity, FjErrorBuffer* error) {
    if (!valid_diagnostic(error)) {
        return {FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0};
    }
    if (!identity || identity->context.device_id < 0 || identity->context.context == 0 ||
        identity->context.epoch == 0 || ((identity->instance_token == 0) != (identity->submission_id == 0))) {
        return status(FJ_STATUS_UNSUPPORTED_INPUT, error, "invalid CUDA context retirement identity");
    }
    try {
        JuicerCuda::NativeCall call(cuda);
        JuicerCuda::Failure failure;
        if (!call.root().retire_context(*identity, failure)) {
            return JuicerCuda::write_status(failure.status, failure.diagnostic, error);
        }
        return status(FJ_STATUS_SUCCESS, error, "");
    } catch (const JuicerCuda::ExecutionFailure& failure) {
        return JuicerCuda::write_status(failure.failure.status, failure.failure.diagnostic, error);
    } catch (const std::bad_alloc&) {
        return status(FJ_STATUS_ALLOCATION_FAILURE, error, "CUDA context retirement allocation failed");
    } catch (...) {
        return status(FJ_STATUS_INTERNAL_FAILURE, error, "CUDA context retirement failed");
    }
}

FjStatus fj_cuda_metering(const FjRect* bounds, uint32_t method, FjAutoExposure* out, FjErrorBuffer* error) {
    if (out) {
        *out = {};
    }
    if (!valid_diagnostic(error)) {
        return {FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0};
    }
    if (!bounds || !out || !ordered_rect(*bounds) || method > FJ_METER_HIGHLIGHT_WEIGHTED) {
        return status(FJ_STATUS_UNSUPPORTED_INPUT, error, "invalid auto-exposure preview arguments");
    }
    const JuicerCuda::FrameRect rect{bounds->x1, bounds->y1, bounds->x2, bounds->y2};
    const auto preview = JuicerCuda::make_auto_exposure_preview_descriptor(
        rect, rect, static_cast<Spektrafilm::AutoExposureMethod>(method));
    *out = {*bounds, *bounds, preview.previewWidth, preview.previewHeight, method, preview.hash};
    return status(FJ_STATUS_SUCCESS, error, "");
}

FjStatus fj_cuda_retire_instance(FjCuda* cuda, uint64_t instance_token, FjErrorBuffer* error) {
    try {
        if (!cuda || instance_token == 0 || (error && error->capacity && !error->data)) {
            return status(FJ_STATUS_UNSUPPORTED_INPUT, error, "invalid CUDA instance retirement arguments");
        }
        JuicerCuda::NativeCall call(cuda);
        cuda->retire_instance(instance_token);
        return status(FJ_STATUS_SUCCESS, error, "");
    } catch (const JuicerCuda::ExecutionFailure& detail) {
        return JuicerCuda::write_status(detail.failure.status, detail.failure.diagnostic.empty() ? "CUDA instance retirement failed" : std::string_view(detail.failure.diagnostic), error);
    } catch (const std::bad_alloc&) {
        return status(FJ_STATUS_ALLOCATION_FAILURE, error, "CUDA instance retirement allocation failed");
    } catch (const std::exception& detail) {
        return status(FJ_STATUS_INTERNAL_FAILURE, error, detail.what());
    } catch (...) {
        return status(FJ_STATUS_INTERNAL_FAILURE, error, "CUDA instance retirement failed with unknown exception");
    }
}

FjStatus fj_cuda_check_terminal(FjCuda* cuda, FjErrorBuffer* error) {
    if (!valid_diagnostic(error)) {
        return {FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0};
    }
    if (!cuda || gNativeCallActive) {
        return status(FJ_STATUS_UNSUPPORTED_INPUT, error, "invalid or reentrant CUDA terminal precheck");
    }
    if (cuda != gOwner.load(std::memory_order_acquire)) {
        return status(FJ_STATUS_PREPARATION_FAILURE, error, "CUDA terminal precheck requires the registered owner");
    }
    return status(FJ_STATUS_SUCCESS, error, "");
}

FjStatus fj_cuda_shutdown(FjCuda* cuda, FjErrorBuffer* error) {
    if (!valid_diagnostic(error)) {
        return {FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0};
    }
    if (!cuda || gNativeCallActive) {
        return status(FJ_STATUS_UNSUPPORTED_INPUT, error, "invalid or reentrant CUDA shutdown");
    }
    try {
        FjStatus result{};
        {
            std::lock_guard<std::mutex> lock(gOwnerMutex);
            if (cuda != gOwner.load(std::memory_order_acquire)) {
                return status(FJ_STATUS_PREPARATION_FAILURE, error, "CUDA shutdown requires the registered owner");
            }
            result = cuda->shutdown();
        }
        return JuicerCuda::write_status(result,
                                        result.category == FJ_STATUS_SUCCESS ? std::string_view{} : std::string_view{cuda->terminalDiagnostic.data(), cuda->terminalDiagnosticLength},
                                        error);
    } catch (const std::bad_alloc&) {
        return terminal_admission_failure(cuda, {FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0}, "CUDA shutdown admission allocation failed", false, error);
    } catch (const std::exception& detail) {
        return terminal_admission_failure(cuda, {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, detail.what(), false, error);
    } catch (...) {
        return terminal_admission_failure(cuda, {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "CUDA shutdown admission failed", false, error);
    }
}

FjStatus fj_cuda_destroy(FjCuda* cuda, FjErrorBuffer* error) {
    const bool diagnosticValid = valid_diagnostic(error);
    FjErrorBuffer* nativeError = diagnosticValid ? error : nullptr;
    if (!cuda) {
        return status(FJ_STATUS_UNSUPPORTED_INPUT, nativeError, "CUDA destroy requires an owning handle");
    }
    // Registration precedes every dereference. Reentry retains the native graph
    // while the caller still holds its non-consuming host owner.
    if (cuda != gOwner.load(std::memory_order_acquire)) {
        return status(FJ_STATUS_PREPARATION_FAILURE, nativeError, "CUDA destroy requires the registered owner");
    }
    if (gNativeCallActive) {
        return terminal_admission_failure(cuda, {FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0}, "reentrant CUDA destroy retained the owner", true, nativeError);
    }
    // The Rust owner keeps Assets alive through this native terminal attempt.
    // Legal consuming use requires host callback/view exclusion before entry.
    FjStatus result{};
    bool closed = false;
    try {
        {
            std::lock_guard<std::mutex> lock(gOwnerMutex);
            result = cuda->shutdown();
            closed = result.category == FJ_STATUS_SUCCESS;
            if (closed) {
                gOwner.store(nullptr, std::memory_order_release);
            } else {
                cuda->lifecycle.store(FjCuda::Lifecycle::Retained, std::memory_order_release);
            }
        }
        if (!closed) {
            JuicerCuda::write_status(result, {cuda->terminalDiagnostic.data(), cuda->terminalDiagnosticLength}, nativeError);
        }
    } catch (const std::bad_alloc&) {
        result = terminal_admission_failure(cuda, {FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0}, "CUDA destroy admission allocation failed", true, nativeError);
    } catch (const std::exception& detail) {
        result = terminal_admission_failure(cuda, {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, detail.what(), true, nativeError);
    } catch (...) {
        result = terminal_admission_failure(cuda, {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "CUDA destroy admission failed", true, nativeError);
    }
    // Native metadata destruction stays outside every native lock. Uncertain
    // retention keeps only the native graph; the Rust owner handles host cleanup.
    if (closed) {
        delete cuda;
        return diagnosticValid ? status(FJ_STATUS_SUCCESS, nativeError, "")
                               : FjStatus{FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0};
    }
    return result;
}

namespace JuicerCuda {

    void check_native_call_admission(FjCuda* cuda) {
        if (!cuda || gNativeCallActive) {
            reject_native_call(FJ_STATUS_UNSUPPORTED_INPUT, "invalid or reentrant CUDA call");
        }
        if (cuda != gOwner.load(std::memory_order_acquire)) {
            reject_native_call(FJ_STATUS_PREPARATION_FAILURE, "CUDA call requires the registered owner");
        }
    }


    JuicerProcess::Root& borrow_root(FjCuda* cuda) {
        if (!cuda || cuda != gOwner.load(std::memory_order_acquire)) {
            reject_native_call(FJ_STATUS_PREPARATION_FAILURE, "Root borrow requires the supplied registered owner");
        }
        return cuda->root;
    }

    JuicerProcess::Root& NativeCall::root() const noexcept {
        return _cuda->root;
    }

    NativeCall::NativeCall(FjCuda* cuda)
        : _lock(native_call_mutex(cuda)), _cuda(cuda) {
        if (cuda->lifecycle.load(std::memory_order_acquire) != FjCuda::Lifecycle::Accepting) {
            reject_native_call(FJ_STATUS_PREPARATION_FAILURE, "CUDA owner admission is closed");
        }
        gNativeCallActive = true;
    }

    NativeCall::~NativeCall() {
        gNativeCallActive = false;
    }

} // namespace JuicerCuda

uint32_t fj_cuda_trace_flags(void) {
    try {
        return (JTRACE_ENABLED(1) ? FJ_FRAME_TRACE_INFO : 0u) |
               (JTRACE_ENABLED(3) ? FJ_FRAME_TRACE_VERBOSE : 0u);
    } catch (...) {
        return 0;
    }
}

void fj_cuda_trace_cutoff(FjStringView text) {
    try {
#if JUICER_DIAGNOSTICS_COMPILED
        const std::string_view message = text.data ? std::string_view(text.data, text.count) : std::string_view{};
        JTRACE("SPEKTRAFILM", std::string(message));
#else
        (void)text;
#endif
    } catch (...) {
        JuicerLogging::discard_current_exception();
    }
}

void fj_cuda_trace_failure(FjStringView text) {
    try {
#if JUICER_DIAGNOSTICS_COMPILED
        const std::string_view message = text.data ? std::string_view(text.data, text.count) : std::string_view{};
        JTRACE("CUDA", std::string(message));
#else
        (void)text;
#endif
    } catch (...) {
        JuicerLogging::discard_current_exception();
    }
}

void fj_cuda_trace_submission(const FjFrame* frame, const FjSubmission* submission, std::uint32_t reused) {
#if JUICER_DIAGNOSTICS_COMPILED
    if (!frame || !submission ||
        (frame->flags & FJ_FRAME_TRACE_VERBOSE) == 0) {
        return;
    }
    try {
        std::string message = "path=cuda action=";
        message += reused != 0 ? "reuse" : "new";
        message += " frame_token=";
        message += std::to_string(submission->frame_token);
        message += " snapshot_id=";
        message += std::to_string(submission->submission_id);
        message += " instance_token=";
        message += std::to_string(submission->instance_token);
        JTRACE_VERBOSE("MSSNP", message);
    } catch (...) {
        JuicerLogging::discard_current_exception();
    }
#else
    (void)frame;
    (void)submission;
    (void)reused;
#endif
}
