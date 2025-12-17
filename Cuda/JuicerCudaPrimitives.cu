// Cuda/JuicerCudaPrimitives.cu
//
// Phase 2: core CUDA math primitives with CPU-parity semantics.
//
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

    __device__ __forceinline__ bool device_isfinite(float v) {
        return isfinite(v);
    }

    __device__ float sample_density_at_logE_device(const float* x, const float* y, int n, float logE, float gammaFactor) {
        if (!x || !y || n <= 0) {
            return 0.0f;
        }

        // Preserve NaN query behavior (matches CPU).
        if (!device_isfinite(logE)) {
            return nanf("");
        }

        const float gammaSafe = (device_isfinite(gammaFactor) && gammaFactor > 0.0f) ? gammaFactor : 1.0f;
        const float xq = logE * gammaSafe;

        int domainBegin = 0;
        while (domainBegin < n && !device_isfinite(x[domainBegin])) {
            ++domainBegin;
        }
        if (domainBegin >= n) {
            return 0.0f;
        }
        int domainEnd = n - 1;
        while (domainEnd > domainBegin && !device_isfinite(x[domainEnd])) {
            --domainEnd;
        }

        const float xmin = x[domainBegin];
        const float xmax = x[domainEnd];
        if (!device_isfinite(xmin) || !device_isfinite(xmax) || !(xmax >= xmin)) {
            return y[domainBegin];
        }

        if (xq <= xmin) {
            return y[domainBegin];
        }
        if (xq >= xmax) {
            return y[domainEnd];
        }

        int i1 = domainBegin + 1;
        while (i1 <= domainEnd && x[i1] < xq) {
            ++i1;
        }
        if (i1 > domainEnd) {
            return y[domainEnd];
        }

        const int i0 = i1 - 1;
        const float x0 = x[i0];
        const float x1 = x[i1];
        const float y0 = y[i0];
        const float y1 = y[i1];

        const float denom = x1 - x0;
        if (!(denom > 0.0f) || !device_isfinite(denom)) {
            return y0;
        }

        const float t = (xq - x0) / denom;
        return y0 + t * (y1 - y0);
    }

    __global__ void probe_density_curve_kernel(
        const float* x,
        const float* y,
        int n,
        float gammaFactor,
        const float* logE,
        int m,
        float* out)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < m) {
            out[idx] = sample_density_at_logE_device(x, y, n, logE[idx], gammaFactor);
        }
    }

} // namespace

extern "C" cudaError_t juicer_cuda_probe_density_curve(
    const float* dX,
    const float* dY,
    int n,
    float gammaFactor,
    const float* hLogE,
    int m,
    float* hOut,
    void* cudaStreamOpaque)
{
    if (!dX || !dY || n <= 0 || !hLogE || m <= 0 || !hOut) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

    float* dLogE = nullptr;
    float* dOut = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&dLogE), static_cast<size_t>(m) * sizeof(float));
    if (err != cudaSuccess) {
        return err;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&dOut), static_cast<size_t>(m) * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(dLogE);
        return err;
    }

    err = cudaMemcpyAsync(dLogE, hLogE, static_cast<size_t>(m) * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(dOut);
        cudaFree(dLogE);
        return err;
    }

    const int threads = 128;
    const int blocks = (m + threads - 1) / threads;
    probe_density_curve_kernel<<<blocks, threads, 0, stream>>>(dX, dY, n, gammaFactor, dLogE, m, dOut);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(dOut);
        cudaFree(dLogE);
        return err;
    }

    err = cudaMemcpyAsync(hOut, dOut, static_cast<size_t>(m) * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(dOut);
        cudaFree(dLogE);
        return err;
    }

    err = cudaStreamSynchronize(stream);
    cudaFree(dOut);
    cudaFree(dLogE);
    return err;
}

