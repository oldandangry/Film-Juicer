#include "Cuda/JuicerCudaResourceOperations.h"

namespace JuicerCuda::ResourceOperations {
    void before_render_admission(FjCuda*, const FjCudaContext*, const FjSubmission*) {}

    cudaError_t allocate_upload(void** destination, std::size_t bytes) {
        return cudaMallocHost(destination, bytes);
    }

    cudaError_t create_upload_event(cudaEvent_t* event) {
        return cudaEventCreateWithFlags(event, cudaEventDisableTiming);
    }

    cudaError_t copy_upload(void* destination, const void* source, std::size_t bytes, cudaStream_t stream) {
        return cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream);
    }

    cudaError_t read_scan_error(int* host, const int* device, cudaStream_t stream) {
        return cudaMemcpyAsync(host, device, sizeof(*host), cudaMemcpyDeviceToHost, stream);
    }

    cudaError_t query_event(cudaEvent_t event) noexcept {
        return cudaEventQuery(event);
    }

    cudaError_t record_event(cudaEvent_t event, cudaStream_t stream) {
        return cudaEventRecord(event, stream);
    }

    cudaError_t synchronize_stream(cudaStream_t stream) {
        return cudaStreamSynchronize(stream);
    }

    cudaError_t synchronize_context() noexcept {
        return cudaDeviceSynchronize();
    }

    cudaError_t free_device(void* pointer) {
        return cudaFree(pointer);
    }

    cudaError_t free_host(void* pointer) noexcept {
        return cudaFreeHost(pointer);
    }

    cudaError_t destroy_event(cudaEvent_t event) noexcept {
        return cudaEventDestroy(event);
    }

    cufftResult destroy_plan(cufftHandle plan) noexcept {
        return cufftDestroy(plan);
    }

    cufftResult execute_spectrum_r2c(cufftHandle plan, cufftReal* input, cufftComplex* output) noexcept {
        return cufftExecR2C(plan, input, output);
    }
} // namespace JuicerCuda::ResourceOperations
