#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>
#include <cufft.h>

namespace JuicerCuda {
    struct Failure;
    struct Resources;

    struct HostToDeviceCopyRequest {
        const char* stage = nullptr;
        const char* label = nullptr;
    };

    bool enqueue_host_to_device_copy(
        Resources& resources,
        const HostToDeviceCopyRequest& request,
        void* dst,
        const void* src,
        std::size_t bytes,
        void* cudaStreamOpaque,
        Failure& outError) noexcept;

    namespace ResourceOperations {
        cudaError_t allocate_upload(void** destination, std::size_t bytes);
        cudaError_t create_upload_event(cudaEvent_t* event);
        cudaError_t copy_upload(void* destination, const void* source, std::size_t bytes, cudaStream_t stream);
        cudaError_t read_scan_error(int* host, const int* device, cudaStream_t stream);
        cudaError_t query_event(cudaEvent_t event) noexcept;
        cudaError_t record_event(cudaEvent_t event, cudaStream_t stream);
        cudaError_t synchronize_stream(cudaStream_t stream);
        cudaError_t synchronize_context() noexcept;
        cudaError_t free_device(void* pointer);
        cudaError_t free_host(void* pointer) noexcept;
        cudaError_t destroy_event(cudaEvent_t event) noexcept;
        cufftResult destroy_plan(cufftHandle plan) noexcept;
        cufftResult execute_spectrum_r2c(cufftHandle plan, cufftReal* input, cufftComplex* output) noexcept;
    } // namespace ResourceOperations
} // namespace JuicerCuda
