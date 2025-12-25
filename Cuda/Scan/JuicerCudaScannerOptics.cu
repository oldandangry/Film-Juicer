// Cuda/Scan/JuicerCudaScannerOptics.cu
// Stage-aligned CUDA TU for scanner optics kernels.
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "Cuda/JuicerCudaKernelsUtil.cuh"
#include "openrand/philox.h"

namespace {

    // --- Scanner glare parity (matches ScannerOptics.cpp) ---
    struct GlareRngDevice {
        openrand::Philox rng;

        __device__ GlareRngDevice(std::uint64_t seed, std::uint32_t ctr0, std::uint32_t ctr1, std::uint32_t globalSeed)
            : rng(seed, ctr0, globalSeed, ctr1) {}

        __device__ __forceinline__ float normal() {
            float u1 = rng.rand<float>();
            u1 = fminf(fmaxf(u1, 1e-7f), 1.0f);
            const float u2 = rng.rand<float>();
            const float r = sqrtf(-2.0f * logf(u1));
            constexpr float kTwoPi = 6.28318530717958647692f;
            return r * cosf(kTwoPi * u2);
        }
    };

    __device__ __forceinline__ float lognormal_from_mean_std_device(float mean, float stddev, float normalSample) {
        const float m2 = mean * mean;
        const float s2 = stddev * stddev;
        const float sigmaSq = logf(1.0f + (s2 / m2));
        const float sigma = sqrtf(fmaxf(0.0f, sigmaSq));
        const float mu = logf(fmaxf(1e-12f, mean)) - 0.5f * sigmaSq;
        return expf(mu + sigma * normalSample);
    }

} // namespace

__global__ void optics_glare_generate_kernel(
    float* out,
    int width,
    int height,
    std::uint64_t glareSeed,
    std::uint64_t mediumId,
    int originX,
    int originY,
    float percent,
    float roughness)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    if (!out) {
        return;
    }
    if (!device_isfinite(percent) || !(percent > 0.0f) || !device_isfinite(roughness)) {
        out[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] = 0.0f;
        return;
    }

    const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
    const std::uint64_t absX = static_cast<std::uint64_t>(originX + x);
    const std::uint64_t absY = static_cast<std::uint64_t>(originY + y);

    GlareRngDevice rng(glareSeed,
        static_cast<std::uint32_t>(absX),
        static_cast<std::uint32_t>(absY),
        static_cast<std::uint32_t>(mediumId));
    const float n = rng.normal();

    const float mean = fmaxf(0.0f, percent);
    const float stddev = fmaxf(0.0f, roughness * percent);
    const float glare = lognormal_from_mean_std_device(mean, stddev, n);

    out[idx] = (device_isfinite(glare) && !isnan(glare)) ? glare : 0.0f;
}

__global__ void optics_blur_horizontal_kernel(
    const float* JUICER_RESTRICT in,
    float* out,
    int width,
    int height,
    const float* JUICER_RESTRICT k,
    int radius)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    if (!in || !out || !k || radius <= 0) {
        return;
    }

    double acc = 0.0;
    for (int j = -radius; j <= radius; ++j) {
        const int xx = reflect_index_repeat_device(x + j, width);
        const float v = in[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(xx)];
        const float w = k[j + radius];
        acc += static_cast<double>(v) * static_cast<double>(w);
    }
    out[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] =
        (isfinite(acc) && !isnan(acc)) ? static_cast<float>(acc) : 0.0f;
}

__global__ void optics_blur_vertical_kernel(
    const float* JUICER_RESTRICT in,
    float* out,
    int width,
    int height,
    const float* JUICER_RESTRICT k,
    int radius)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    if (!in || !out || !k || radius <= 0) {
        return;
    }

    double acc = 0.0;
    for (int j = -radius; j <= radius; ++j) {
        const int yy = reflect_index_repeat_device(y + j, height);
        const float v = in[static_cast<size_t>(yy) * static_cast<size_t>(width) + static_cast<size_t>(x)];
        const float w = k[j + radius];
        acc += static_cast<double>(v) * static_cast<double>(w);
    }
    out[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] =
        (isfinite(acc) && !isnan(acc)) ? static_cast<float>(acc) : 0.0f;
}

__global__ void optics_unsharp_combine_kernel(float* inOut, const float* JUICER_RESTRICT blurred, int n, float amount) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    if (!inOut || !blurred) {
        return;
    }
    const double v0 = static_cast<double>(inOut[idx]);
    const double vb = static_cast<double>(blurred[idx]);
    const double a = static_cast<double>(amount);
    const double v = v0 + a * (v0 - vb);
    inOut[idx] = (isfinite(v) && !isnan(v)) ? static_cast<float>(v) : 0.0f;
}
