// Cuda/JuicerCudaSelfCheck.cu
//
// Runtime CUDA self-check (see header for intent + removal notes).
//
// This validates a minimal end-to-end CUDA path:
// - stream usage
// - device allocations
// - H2D + kernel + D2H on that stream
// - error propagation
//
#include <cuda_runtime.h>

#include <cmath>
#include <string>

namespace {

    __global__ void add_kernel(const float* a, const float* b, float* c, int n) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            c[idx] = a[idx] + b[idx];
        }
    }

    const char* set_error(std::string& storage, const char* msg) {
        storage = msg ? msg : "(unknown)";
        return storage.c_str();
    }

    const char* set_error_cuda(std::string& storage, const char* prefix, cudaError_t err) {
        const char* cudaMsg = cudaGetErrorString(err);
        storage.clear();
        if (prefix) {
            storage += prefix;
        }
        storage += (cudaMsg ? cudaMsg : "(unknown)");
        return storage.c_str();
    }

} // namespace

bool juicer_cuda_runtime_self_check(void* cudaStreamOpaque, const char** outError) {
    static std::string sError;
    if (outError) {
        *outError = nullptr;
    }

    const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

    constexpr int kN = 256;
    alignas(16) float hA[kN];
    alignas(16) float hB[kN];
    alignas(16) float hC[kN];
    for (int i = 0; i < kN; ++i) {
        hA[i] = 0.5f + 0.01f * static_cast<float>(i);
        hB[i] = -0.25f + 0.02f * static_cast<float>(i);
        hC[i] = 0.0f;
    }

    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;
    cudaError_t err = cudaSuccess;

    err = cudaMalloc(reinterpret_cast<void**>(&dA), sizeof(hA));
    if (err != cudaSuccess) {
        if (outError) *outError = set_error_cuda(sError, "cudaMalloc(dA) failed: ", err);
        return false;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&dB), sizeof(hB));
    if (err != cudaSuccess) {
        cudaFree(dA);
        if (outError) *outError = set_error_cuda(sError, "cudaMalloc(dB) failed: ", err);
        return false;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&dC), sizeof(hC));
    if (err != cudaSuccess) {
        cudaFree(dB);
        cudaFree(dA);
        if (outError) *outError = set_error_cuda(sError, "cudaMalloc(dC) failed: ", err);
        return false;
    }

    err = cudaMemcpyAsync(dA, hA, sizeof(hA), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(dC);
        cudaFree(dB);
        cudaFree(dA);
        if (outError) *outError = set_error_cuda(sError, "cudaMemcpyAsync(H2D A) failed: ", err);
        return false;
    }
    err = cudaMemcpyAsync(dB, hB, sizeof(hB), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(dC);
        cudaFree(dB);
        cudaFree(dA);
        if (outError) *outError = set_error_cuda(sError, "cudaMemcpyAsync(H2D B) failed: ", err);
        return false;
    }

    const int threads = 128;
    const int blocks = (kN + threads - 1) / threads;
    add_kernel<<<blocks, threads, 0, stream>>>(dA, dB, dC, kN);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(dC);
        cudaFree(dB);
        cudaFree(dA);
        if (outError) *outError = set_error_cuda(sError, "self-check kernel launch failed: ", err);
        return false;
    }

    err = cudaMemcpyAsync(hC, dC, sizeof(hC), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(dC);
        cudaFree(dB);
        cudaFree(dA);
        if (outError) *outError = set_error_cuda(sError, "cudaMemcpyAsync(D2H C) failed: ", err);
        return false;
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        cudaFree(dC);
        cudaFree(dB);
        cudaFree(dA);
        if (outError) *outError = set_error_cuda(sError, "cudaStreamSynchronize failed: ", err);
        return false;
    }

    cudaFree(dC);
    cudaFree(dB);
    cudaFree(dA);

    for (int i = 0; i < kN; ++i) {
        const float expected = hA[i] + hB[i];
        const float diff = std::fabs(hC[i] - expected);
        if (!std::isfinite(hC[i])) {
            if (outError) *outError = set_error(sError, "self-check validation failed: non-finite kernel result");
            return false;
        }
        if (!(diff <= 1e-6f)) {
            if (outError) *outError = set_error(sError, "self-check validation failed: wrong kernel result");
            return false;
        }
    }

    return true;
}
