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
#include "ProcessRoot.h"
#include "RustAssetBridge.h"

struct FjCuda final {
    explicit FjCuda(const std::filesystem::path& dataDirectory)
        : root(dataDirectory) {
    }

    enum class Lifecycle : std::uint8_t {
        Accepting,
        Closing,
        Closed,
        Blocked,
        Retained
    };

    ~FjCuda() {
#if defined(JUICER_CUDA_TERMINAL_TEST_HOOK)
        JuicerCuda::TerminalTest::owner_destroyed();
#endif
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
#if defined(JUICER_CUDA_TERMINAL_TEST_HOOK)
            JuicerCuda::TerminalTest::before_shutdown();
#endif
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

    FjStatus release_host_caches(FjErrorBuffer* error) noexcept {
        auto state = hostCleanup.load(std::memory_order_acquire);
        for (;;) {
            if (state == HostCleanup::Complete) {
                return JuicerCuda::write_status(hostCleanupStatus, {hostCleanupDiagnostic.data(), hostCleanupDiagnosticLength}, error);
            }
            if (state == HostCleanup::Running) {
                hostCleanup.wait(state, std::memory_order_acquire);
            } else if (hostCleanup.compare_exchange_weak(state, HostCleanup::Running, std::memory_order_acq_rel)) {
                FjErrorBuffer retained{hostCleanupDiagnostic.data(), hostCleanupDiagnostic.size(), 0};
#if defined(JUICER_CUDA_TERMINAL_TEST_HOOK)
                JuicerCuda::TerminalTest::before_host_cleanup();
#endif
                hostCleanupStatus = root.release_process_host_services(&retained);
                hostCleanupDiagnosticLength = retained.length;
                hostCleanup.store(HostCleanup::Complete, std::memory_order_release);
                hostCleanup.notify_all();
            }
            state = hostCleanup.load(std::memory_order_acquire);
        }
    }

    std::unique_ptr<JuicerAssets::Library> detach_host_assets() noexcept {
        return root.detach_host_assets();
    }

    void retire_instance(std::uint64_t instanceToken) {
        root.retire_grain_static_instance(instanceToken);
    }

    std::atomic<Lifecycle> lifecycle{Lifecycle::Accepting};
    FjStatus terminalStatus{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
    std::array<char, 1024> terminalDiagnostic{};
    std::size_t terminalDiagnosticLength = 0;

    enum class HostCleanup : std::uint8_t {
        Pending,
        Running,
        Complete
    };
    std::atomic<HostCleanup> hostCleanup{HostCleanup::Pending};
    FjStatus hostCleanupStatus{FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
    std::array<char, 1024> hostCleanupDiagnostic{};
    std::size_t hostCleanupDiagnosticLength = 0;

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
               context->device_id < 0 || context->context == 0 ||
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

FjStatus fj_cuda_create(FjPathView data_directory, FjCuda** out_cuda, FjErrorBuffer* error) {
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
        const auto directory = JuicerAssets::copy_native_path(data_directory);
        auto candidate = std::make_unique<FjCuda>(directory);
#if defined(JUICER_CUDA_TERMINAL_TEST_HOOK)
        JuicerCuda::TerminalTest::after_construct();
#endif
        bool published = false;
        {
            std::lock_guard<std::mutex> lock(gOwnerMutex);
            if (!gOwner.load(std::memory_order_acquire)) {
                *out_cuda = candidate.release();
                gOwner.store(*out_cuda, std::memory_order_release);
                published = true;
            }
        }
        // Rejected candidates and partial construction release Rust outside locks.
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
    try {
        if (!cuda || !frame || !out_context || (error && error->capacity && !error->data)) {
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

FjStatus JuicerCuda::NativeCall::inspect(const FjFrame* frame, FjCudaContext* outContext, FjErrorBuffer* error) {
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
        void* context = nullptr;
        int driverCode = 0;
        std::string diagnostic;
        if (!JuicerCuda::query_current_cuda_context(context, diagnostic, &driverCode)) {
            return driver_failure(error, diagnostic, driverCode);
        }
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
        *outContext = FjCudaContext{device, reinterpret_cast<std::uintptr_t>(context)};
        return status(FJ_STATUS_SUCCESS, error, "");
    } catch (const std::bad_alloc&) {
        return status(FJ_STATUS_ALLOCATION_FAILURE, error, "CUDA inspection allocation failed");
    } catch (const std::exception& detail) {
        return status(FJ_STATUS_INTERNAL_FAILURE, error, detail.what());
    } catch (...) {
        return status(FJ_STATUS_INTERNAL_FAILURE, error, "CUDA inspection failed");
    }
}

FjRenderOutcome fj_cuda_render(FjCuda* cuda, const FjCudaContext* context, const FjFrame* frame, const FjSubmission* submission, const FjPreparedHostData* prepared, FjAbortCallback abort_callback, FjErrorBuffer* error) {
    try {
        if (!cuda || invalid_render_arguments(context, frame, submission, prepared) ||
            (error && error->capacity && !error->data)) {
            return {status(FJ_STATUS_UNSUPPORTED_INPUT, error, "invalid CUDA render arguments"), 0};
        }
        JuicerCuda::NativeCall call(cuda);
        JuicerCuda::PendingContextLossRecovery recovery;
        std::string diagnostic;
        const auto outcome = call.render(context, frame, submission, prepared, abort_callback, recovery, diagnostic);
        JuicerCuda::write_status(outcome.status, diagnostic, error);
        return outcome;
    } catch (const JuicerCuda::ExecutionFailure& failure) {
        return {JuicerCuda::write_status(failure.failure.status, failure.failure.diagnostic, error), 0};
    } catch (const std::bad_alloc&) {
        return {status(FJ_STATUS_ALLOCATION_FAILURE, error, "CUDA render allocation failed"), 0};
    } catch (const std::exception& detail) {
        return {status(FJ_STATUS_INTERNAL_FAILURE, error, detail.what()), 0};
    } catch (...) {
        return {status(FJ_STATUS_INTERNAL_FAILURE, error, "CUDA render failed with unknown exception"), 0};
    }
}

FjRenderOutcome JuicerCuda::NativeCall::render(
    const FjCudaContext* context, const FjFrame* frame, const FjSubmission* submission, const FjPreparedHostData* prepared, FjAbortCallback abortCallback, PendingContextLossRecovery& recovery, std::string& diagnostic) {
    diagnostic.clear();
    const auto result = [&](FjStatus selected, std::string_view message) -> FjRenderOutcome {
        diagnostic.assign(message);
        return {selected, 0};
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
        const auto inspection = inspect(frame, &inspected, &inspectionError);
        if (inspection.category != FJ_STATUS_SUCCESS) {
            diagnostic.assign(inspectionMessage.data(), inspectionError.length);
            return {inspection, 0};
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
        if (frame->stream != 0) {
            void* streamContext = nullptr;
            int driverCode = 0;
            if (!JuicerCuda::query_cuda_stream_context(frame->stream, streamContext, driverCode, diagnostic)) {
                return {driver_failure_status(driverCode), 0};
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
            {submission->instance_token}, {submission->frame_token}, submission->submission_id, {context->device_id, reinterpret_cast<void*>(context->context)}, {submission->upload_core_hash, submission->dir_hash, submission->scanner_hash, submission->auto_exposure_hash}};
        const auto complete = [&]() {
            const auto completion = (frame->flags & FJ_FRAME_STREAM_PRESENT) != 0
                                        ? cudaSuccess
                                        : cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(frame->stream));
            if (completion != cudaSuccess) {
                const Failure failure{runtime_failure_status(completion), cudaGetErrorString(completion)};
                if (!recovery.pending && context_loss(failure)) {
                    recovery.pending = true;
                    recovery.failure = failure;
                    recovery.stage = "absent_stream_completion";
                }
            }
            return completion;
        };
        try {
#if defined(JUICER_CUDA_RENDER_TEST_HOOK)
            JuicerCuda::RenderTest::before_execute();
#endif
            if (!JuicerCuda::execute_prepared_host_data(*prepared, execution, snapshot, recovery, abortCallback, diagnostic)) {
                return {{FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0}, 0};
            }
#if defined(JUICER_CUDA_RENDER_TEST_HOOK)
            JuicerCuda::RenderTest::after_execute();
#endif
        } catch (...) {
            const auto completion = complete();
            if (completion != cudaSuccess) {
                return result(runtime_failure_status(completion), cudaGetErrorString(completion));
            }
            throw;
        }
        const auto completion = complete();
        if (completion != cudaSuccess) {
            return result(runtime_failure_status(completion), cudaGetErrorString(completion));
        }
        return result({FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, "");
    } catch (JuicerCuda::ExecutionFailure& failure) {
        diagnostic = std::move(failure.failure.diagnostic);
        return {failure.failure.status, failure.deferredDirError ? FJ_RENDER_DEFERRED_DIR_ERROR : 0U};
    } catch (const std::bad_alloc&) {
        return result({FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0}, "CUDA render allocation failed");
    } catch (const std::exception& detail) {
        return result({FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, detail.what());
    } catch (...) {
        return result({FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "CUDA render failed with unknown exception");
    }
}

FjStatus fj_cuda_retire_instance(FjCuda* cuda, uint64_t instance_token, FjErrorBuffer* error) {
    try {
        if (!cuda || instance_token == 0 || (error && error->capacity && !error->data)) {
            return status(FJ_STATUS_UNSUPPORTED_INPUT, error, "invalid CUDA instance retirement arguments");
        }
        JuicerCuda::NativeCall call(cuda);
#if defined(JUICER_CUDA_TERMINAL_TEST_HOOK)
        JuicerCuda::TerminalTest::before_retire_instance();
#endif
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

FjStatus fj_cuda_shutdown(FjCuda* cuda, FjErrorBuffer* error) {
    if (!valid_diagnostic(error)) {
        return {FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0};
    }
    if (!cuda || gNativeCallActive) {
        return status(FJ_STATUS_UNSUPPORTED_INPUT, error, "invalid or reentrant CUDA shutdown");
    }
    try {
#if defined(JUICER_CUDA_TERMINAL_TEST_HOOK)
        JuicerCuda::TerminalTest::before_owner_lock();
#endif
        FjStatus result{};
        {
            std::lock_guard<std::mutex> lock(gOwnerMutex);
            if (cuda != gOwner.load(std::memory_order_acquire)) {
                return status(FJ_STATUS_PREPARATION_FAILURE, error, "CUDA shutdown requires the registered owner");
            }
            result = cuda->shutdown();
        }
        if (result.category == FJ_STATUS_SUCCESS) {
            return cuda->release_host_caches(error);
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
    FjErrorBuffer* nativeError = valid_diagnostic(error) ? error : nullptr;
    if (!cuda) {
        return status(FJ_STATUS_UNSUPPORTED_INPUT, nativeError, "CUDA destroy requires an owning handle");
    }
    // Registration precedes every dereference. Reentry cannot detach assets
    // borrowed by its active invocation; it retains the existing blocked graph.
    if (cuda != gOwner.load(std::memory_order_acquire)) {
        return status(FJ_STATUS_PREPARATION_FAILURE, nativeError, "CUDA destroy requires the registered owner");
    }
    if (gNativeCallActive) {
        return terminal_admission_failure(cuda, {FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0}, "reentrant CUDA destroy retained the owner", true, nativeError);
    }
    // Legal terminal use requires the host to exclude ALL asset readers/views,
    // independently of GPU draining. This move allocates nothing and precedes
    // throwable admission. Keep the whole Library through the native attempt.
    auto hostAssets = cuda->detach_host_assets();
    FjStatus result{};
    bool closed = false;
    try {
#if defined(JUICER_CUDA_TERMINAL_TEST_HOOK)
        JuicerCuda::TerminalTest::before_owner_lock();
#endif
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
    if (closed && cuda->hostCleanup.load(std::memory_order_acquire) == FjCuda::HostCleanup::Complete) {
        result = JuicerCuda::write_status(cuda->hostCleanupStatus, {cuda->hostCleanupDiagnostic.data(), cuda->hostCleanupDiagnosticLength}, nativeError);
    }
    // Neither Rust cleanup nor native metadata destruction runs under a lock.
    const auto hostResult = hostAssets ? hostAssets->close(result.category == FJ_STATUS_SUCCESS ? error : nullptr)
                                       : FjStatus{FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
    hostAssets.reset();
    if (closed) {
        delete cuda;
    }
    return result.category == FJ_STATUS_SUCCESS ? hostResult : result;
}

namespace JuicerProcess {

    Root& Root::instance() {
        FjCuda* cuda = gOwner.load(std::memory_order_acquire);
        if (!cuda) {
            throw std::logic_error("CUDA runtime has no owner");
        }
        return cuda->root;
    }

    Root& root() {
        return Root::instance();
    }


} // namespace JuicerProcess

namespace JuicerCuda {

    void check_native_call_admission(FjCuda* cuda) {
        if (!cuda || gNativeCallActive) {
            reject_native_call(FJ_STATUS_UNSUPPORTED_INPUT, "invalid or reentrant CUDA call");
        }
        if (cuda != gOwner.load(std::memory_order_acquire)) {
            reject_native_call(FJ_STATUS_PREPARATION_FAILURE, "CUDA call requires the registered owner");
        }
    }

#if defined(JUICER_NOISE_TEST_HOOK)
    bool calling_thread_native_gate_active() noexcept {
        return gNativeCallActive;
    }
#endif

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

    FjCuda* borrowed_owner() noexcept {
        return gOwner.load(std::memory_order_acquire);
    }

    Owner::~Owner() {
        close();
    }

    void Owner::create(const std::filesystem::path& dataDirectory) {
        if (_cuda) {
            throw std::logic_error("CUDA owner is already initialized");
        }
        std::array<char, 256> message{};
        FjErrorBuffer error{message.data(), message.size(), 0};
        const JuicerAssets::NativePathArgument argument(dataDirectory);
        const FjStatus result = fj_cuda_create(argument.view(), &_cuda, &error);
        if (result.category == FJ_STATUS_ALLOCATION_FAILURE) {
            throw std::bad_alloc();
        }
        if (result.category != FJ_STATUS_SUCCESS) {
            throw std::runtime_error(message.data());
        }
    }

    FjStatus Owner::close(FjErrorBuffer* error) noexcept {
        FjCuda* cuda = std::exchange(_cuda, nullptr);
        return cuda ? fj_cuda_destroy(cuda, error)
                    : write_status({FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, {}, error);
    }

} // namespace JuicerCuda
