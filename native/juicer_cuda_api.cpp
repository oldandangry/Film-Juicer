#include "juicer_cuda_api.h"
#include "juicer_cuda_owner.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

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
