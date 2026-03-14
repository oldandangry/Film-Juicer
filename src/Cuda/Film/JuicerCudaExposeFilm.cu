// Cuda/Film/JuicerCudaExposeFilm.cu
// Stage-aligned CUDA TU for film exposure kernels.
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

#include "Cuda/JuicerCudaPayloads.h"
#include "Cuda/Film/JuicerCudaFilmExposure.cuh"

__global__ void expose_film_raw_kernel(
    JuicerCuda::PipelineRunParams params,
    float* outB,
    float* outG,
    float* outR)
{
    if (!params.src || params.srcRowBytes == 0) {
        return;
    }
    if (!outB || !outG || !outR) {
        return;
    }

    const int nC = params.nComponents;
    if (!(nC == 3 || nC == 4)) {
        return;
    }

    const std::size_t pixelBytes = static_cast<std::size_t>(nC) * sizeof(float);
    for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < params.height; y += blockDim.y * gridDim.y) {
        const char* srcRow = reinterpret_cast<const char*>(params.src) + static_cast<std::size_t>(y) * params.srcRowBytes;
        for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < params.width; x += blockDim.x * gridDim.x) {
            const float* srcPix = reinterpret_cast<const float*>(srcRow + static_cast<std::size_t>(x) * pixelBytes);
            if (!srcPix) {
                continue;
            }

            const float rgbIn[3] = { srcPix[0], srcPix[1], srcPix[2] };
            float filmRaw[3] = { 0.0f, 0.0f, 0.0f };
            compute_film_raw_device(params, rgbIn, filmRaw);

            const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
            outB[idx] = filmRaw[0];
            outG[idx] = filmRaw[1];
            outR[idx] = filmRaw[2];
        }
    }
}

__global__ void halation_apply_kernel(float* inOut, const float* blurred, int n, float strength) {
    if (!inOut || !blurred) {
        return;
    }

    const float s = strength;
    if (!(s > 0.0f)) {
        return;
    }

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        const double a = static_cast<double>(inOut[idx]);
        const double b = static_cast<double>(blurred[idx]);
        const double out = (a + static_cast<double>(s) * b) / (1.0 + static_cast<double>(s));
        inOut[idx] = (isfinite(out) && !isnan(out)) ? static_cast<float>(out) : 0.0f;
    }
}
