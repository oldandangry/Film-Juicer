#include "Cuda/JuicerCudaFailure.h"

#include <algorithm>
#include <cstring>

#include <cuda.h>
#include <cuda_runtime.h>

namespace JuicerCuda {

    FjStatus runtime_failure_status(std::int32_t code) noexcept {
        if (code == cudaSuccess) {
            return {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        }
        const auto category = code == cudaErrorMemoryAllocation
                                  ? FJ_STATUS_ALLOCATION_FAILURE
                              : code == cudaErrorContextIsDestroyed || code == cudaErrorDeviceUninitialized || code == cudaErrorCudartUnloading
                                  ? FJ_STATUS_CONTEXT_LOSS
                                  : FJ_STATUS_CUDA_FAILURE;
        return {category, FJ_API_CUDA_RUNTIME, code};
    }

    FjStatus driver_failure_status(std::int32_t code) noexcept {
        if (code == CUDA_SUCCESS) {
            return {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        }
        const auto category = code == CUDA_ERROR_OUT_OF_MEMORY
                                  ? FJ_STATUS_ALLOCATION_FAILURE
                              : code == CUDA_ERROR_CONTEXT_IS_DESTROYED || code == CUDA_ERROR_INVALID_CONTEXT || code == CUDA_ERROR_DEINITIALIZED
                                  ? FJ_STATUS_CONTEXT_LOSS
                                  : FJ_STATUS_CUDA_FAILURE;
        return {category, FJ_API_CUDA_DRIVER, code};
    }

    FjStatus cufft_failure_status(std::int32_t code) noexcept {
        if (code == 0) {
            return {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        }
        return {FJ_STATUS_CUFFT_FAILURE, FJ_API_CUFFT, code};
    }

    bool allocation_capacity_exhausted(const Failure& failure) noexcept {
        return failure.status.category == FJ_STATUS_ALLOCATION_FAILURE;
    }

    bool context_loss(const Failure& failure) noexcept {
        return failure.status.category == FJ_STATUS_CONTEXT_LOSS;
    }

    FjStatus write_status(FjStatus status, std::string_view diagnostic, FjErrorBuffer* error) noexcept {
        if (error) {
            error->length = 0;
            if (error->capacity && error->data) {
                error->length = std::min(diagnostic.size(), error->capacity - 1);
                if (error->length) {
                    std::memcpy(error->data, diagnostic.data(), error->length);
                }
                error->data[error->length] = '\0';
            }
        }
        return status;
    }


} // namespace JuicerCuda
