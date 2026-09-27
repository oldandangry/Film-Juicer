#include "juicer_cuda_api.h"
#include "juicer_cuda_owner.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_runtime.h>

#include "Cuda/JuicerCudaDriver.h"
#include "Logging.h"
#include "ProcessRoot.h"

struct FjCuda final {
    explicit FjCuda(std::string dataDirectory)
        : root(std::move(dataDirectory)) {
    }

    bool close() noexcept {
        try {
            {
                std::lock_guard<std::mutex> lock(root._framePreparationMutex);
                if (root._shutdownRetireBlocked) {
                    return false;
                }
            }
            return root.shutdown();
        } catch (...) {
            JuicerLogging::discard_current_exception();
            return false;
        }
    }

    JuicerProcess::Root root;
};

namespace {

    std::mutex gOwnerMutex;
    constinit std::atomic<FjCuda*> gOwner{nullptr};

    FjStatus status(uint32_t category, FjErrorBuffer* error, const char* message) noexcept {
        if (error) {
            error->length = 0;
            if (error->capacity != 0 && error->data) {
                error->length = std::min(std::strlen(message), error->capacity - 1);
                std::memcpy(error->data, message, error->length);
                error->data[error->length] = '\0';
            }
        }
        return FjStatus{category, FJ_API_NONE, 0};
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
        auto result = status(code ? FJ_STATUS_CUDA_FAILURE : FJ_STATUS_PREPARATION_FAILURE, error, message.c_str());
        if (code) {
            result.api = FJ_API_CUDA_DRIVER;
            result.native_code = code;
        }
        return result;
    }

    // Removal: S2.D supplies typed shutdown/destroy and terminal retention.
    // Keep a failed graph registered and alive; a new runtime cannot coexist
    // with its registry. This bridge is not a successful C shutdown result.
    bool SF_TEMP_BRIDGE_release_cuda_owner(FjCuda* cuda) noexcept {
        if (!cuda) {
            return true;
        }
        try {
            std::lock_guard<std::mutex> lock(gOwnerMutex);
            if (gOwner.load(std::memory_order_acquire) != cuda || !cuda->close()) {
                JTRACE("MSLCY", "SF_TEMP_BRIDGE_release_cuda_owner retained; S2.D terminal qualification pending");
                return false;
            }
            gOwner.store(nullptr, std::memory_order_release);
            delete cuda;
            return true;
        } catch (...) {
            JuicerLogging::discard_current_exception();
            return false;
        }
    }

} // namespace

FjStatus fj_cuda_create(FjStringView data_directory, FjCuda** out_cuda, FjErrorBuffer* error) {
    if (out_cuda) {
        *out_cuda = nullptr;
    }
    try {
        if (!out_cuda || !data_directory.data || data_directory.count == 0 ||
            (error && error->capacity != 0 && !error->data) ||
            std::memchr(data_directory.data, '\0', data_directory.count)) {
            return status(FJ_STATUS_UNSUPPORTED_INPUT, error, "invalid CUDA owner creation input");
        }
        std::lock_guard<std::mutex> lock(gOwnerMutex);
        if (gOwner.load(std::memory_order_acquire)) {
            return status(FJ_STATUS_PREPARATION_FAILURE, error, "CUDA runtime already has an owner");
        }
        auto cuda = std::make_unique<FjCuda>(std::string(data_directory.data, data_directory.count));
        *out_cuda = cuda.release();
        gOwner.store(*out_cuda, std::memory_order_release);
        return status(FJ_STATUS_SUCCESS, error, "");
    } catch (const std::bad_alloc&) {
        return status(FJ_STATUS_ALLOCATION_FAILURE, error, "CUDA owner allocation failed");
    } catch (const std::exception& errorDetail) {
        return status(FJ_STATUS_PREPARATION_FAILURE, error, errorDetail.what());
    } catch (...) {
        return status(FJ_STATUS_INTERNAL_FAILURE, error, "CUDA owner creation failed");
    }
}

FjStatus fj_cuda_inspect(FjCuda* cuda, const FjFrame* frame, FjCudaContext* out_context, FjErrorBuffer* error) {
    try {
        if (!cuda || !frame || !out_context || (error && error->capacity && !error->data)) {
            return status(FJ_STATUS_UNSUPPORTED_INPUT, error, "invalid CUDA inspection arguments");
        }
        if (cuda != gOwner.load(std::memory_order_acquire)) {
            return status(FJ_STATUS_PREPARATION_FAILURE, error, "CUDA inspection requires the registered owner");
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
                auto result = status(FJ_STATUS_CUDA_FAILURE, error, "cudaPointerGetAttributes failed");
                result.api = FJ_API_CUDA_RUNTIME;
                result.native_code = static_cast<std::int32_t>(runtimeCode);
                return result;
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
        *out_context = FjCudaContext{device, reinterpret_cast<std::uintptr_t>(context)};
        return status(FJ_STATUS_SUCCESS, error, "");
    } catch (const std::bad_alloc&) {
        return status(FJ_STATUS_ALLOCATION_FAILURE, error, "CUDA inspection allocation failed");
    } catch (const std::exception& detail) {
        return status(FJ_STATUS_INTERNAL_FAILURE, error, detail.what());
    } catch (...) {
        return status(FJ_STATUS_INTERNAL_FAILURE, error, "CUDA inspection failed");
    }
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

    void shutdown_if_initialized() noexcept {
        FjCuda* cuda = gOwner.load(std::memory_order_acquire);
        if (cuda) {
            cuda->root.shutdown();
        }
    }

} // namespace JuicerProcess

namespace JuicerCuda {

    FjCuda* borrowed_owner() noexcept {
        return gOwner.load(std::memory_order_acquire);
    }

    Owner::~Owner() {
        close();
    }

    void Owner::create(std::string_view dataDirectory) {
        if (_cuda) {
            throw std::logic_error("CUDA owner is already initialized");
        }
        std::array<char, 256> message{};
        FjErrorBuffer error{message.data(), message.size(), 0};
        const FjStatus result = fj_cuda_create(
            FjStringView{dataDirectory.data(), dataDirectory.size()}, &_cuda, &error);
        if (result.category == FJ_STATUS_ALLOCATION_FAILURE) {
            throw std::bad_alloc();
        }
        if (result.category != FJ_STATUS_SUCCESS) {
            throw std::runtime_error(message.data());
        }
    }

    bool Owner::close() noexcept {
        return SF_TEMP_BRIDGE_release_cuda_owner(std::exchange(_cuda, nullptr));
    }

} // namespace JuicerCuda
