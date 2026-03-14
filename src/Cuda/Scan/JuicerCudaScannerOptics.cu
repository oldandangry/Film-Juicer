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
    if (!out) {
        return;
    }
    if (!device_isfinite(percent) || !(percent > 0.0f) || !device_isfinite(roughness)) {
        return;
    }

    const float mean = fmaxf(0.0f, percent);
    const float stddev = fmaxf(0.0f, roughness * percent);
    for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < height; y += blockDim.y * gridDim.y) {
        for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < width; x += blockDim.x * gridDim.x) {
            const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            const std::uint64_t absX = static_cast<std::uint64_t>(originX + x);
            const std::uint64_t absY = static_cast<std::uint64_t>(originY + y);

            GlareRngDevice rng(glareSeed,
                               static_cast<std::uint32_t>(absX),
                               static_cast<std::uint32_t>(absY),
                               static_cast<std::uint32_t>(mediumId));
            const float n = rng.normal();
            const float glare = lognormal_from_mean_std_device(mean, stddev, n);

            out[idx] = (device_isfinite(glare) && !isnan(glare)) ? glare : 0.0f;
        }
    }
}

__global__ void optics_blur_horizontal_kernel(
    const float* JUICER_RESTRICT in,
    float* out,
    int width,
    int height,
    const float* JUICER_RESTRICT k,
    int radius)
{
    if (!in || !out || !k || radius <= 0) {
        return;
    }

    const int kLen = 2 * radius + 1;
    const int tileW = blockDim.x + 2 * radius;

    extern __shared__ float shared[];
    float* sWeights = shared;
    float* sTile = shared + kLen;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int tcount = blockDim.x * blockDim.y;
    const int yLocal = threadIdx.y;

    for (int i = tid; i < kLen; i += tcount) {
        sWeights[i] = k[i];
    }
    __syncthreads();

    for (int blockY = blockIdx.y * blockDim.y; blockY < height; blockY += blockDim.y * gridDim.y) {
        for (int blockX = blockIdx.x * blockDim.x; blockX < width; blockX += blockDim.x * gridDim.x) {
            const int x = blockX + threadIdx.x;
            const int y = blockY + threadIdx.y;
            const bool inBounds = (x < width && y < height);
            const int yLoad = blockY + yLocal;
            if (yLoad < height) {
                const size_t rowBase = static_cast<size_t>(yLoad) * static_cast<size_t>(width);
                for (int i = threadIdx.x; i < tileW; i += blockDim.x) {
                    const int xLoad = blockX + i - radius;
                    const int xx = reflect_index_repeat_device(xLoad, width);
                    sTile[yLocal * tileW + i] = in[rowBase + static_cast<size_t>(xx)];
                }
            }

            __syncthreads();

            if (inBounds) {
                double acc = 0.0;
                const int tileX = threadIdx.x + radius;
                const int tileRow = threadIdx.y * tileW;
                for (int j = -radius; j <= radius; ++j) {
                    const float v = sTile[tileRow + tileX + j];
                    const float w = sWeights[j + radius];
                    acc += static_cast<double>(v) * static_cast<double>(w);
                }

                out[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] =
                    (isfinite(acc) && !isnan(acc)) ? static_cast<float>(acc) : 0.0f;
            }

            __syncthreads();
        }
    }
}

__global__ void optics_blur_vertical_kernel(
    const float* JUICER_RESTRICT in,
    float* out,
    int width,
    int height,
    const float* JUICER_RESTRICT k,
    int radius)
{
    if (!in || !out || !k || radius <= 0) {
        return;
    }

    const int kLen = 2 * radius + 1;
    const int tileW = blockDim.x;
    const int tileH = blockDim.y + 2 * radius;

    extern __shared__ float shared[];
    float* sWeights = shared;
    float* sTile = shared + kLen;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int tcount = blockDim.x * blockDim.y;
    const int xLocal = threadIdx.x;

    for (int i = tid; i < kLen; i += tcount) {
        sWeights[i] = k[i];
    }
    __syncthreads();

    for (int blockY = blockIdx.y * blockDim.y; blockY < height; blockY += blockDim.y * gridDim.y) {
        for (int blockX = blockIdx.x * blockDim.x; blockX < width; blockX += blockDim.x * gridDim.x) {
            const int x = blockX + threadIdx.x;
            const int y = blockY + threadIdx.y;
            const bool inBounds = (x < width && y < height);
            const int xLoad = blockX + xLocal;
            if (xLoad < width) {
                for (int i = threadIdx.y; i < tileH; i += blockDim.y) {
                    const int yLoad = blockY + i - radius;
                    const int yy = reflect_index_repeat_device(yLoad, height);
                    sTile[i * tileW + xLocal] =
                        in[static_cast<size_t>(yy) * static_cast<size_t>(width) + static_cast<size_t>(xLoad)];
                }
            }

            __syncthreads();

            if (inBounds) {
                double acc = 0.0;
                const int tileY = threadIdx.y + radius;
                for (int j = -radius; j <= radius; ++j) {
                    const float v = sTile[(tileY + j) * tileW + xLocal];
                    const float w = sWeights[j + radius];
                    acc += static_cast<double>(v) * static_cast<double>(w);
                }

                out[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] =
                    (isfinite(acc) && !isnan(acc)) ? static_cast<float>(acc) : 0.0f;
            }

            __syncthreads();
        }
    }
}

__global__ void optics_unsharp_combine_kernel(float* inOut, const float* JUICER_RESTRICT blurred, int n, float amount) {
    if (!inOut || !blurred) {
        return;
    }
    const double a = static_cast<double>(amount);
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        const double v0 = static_cast<double>(inOut[idx]);
        const double vb = static_cast<double>(blurred[idx]);
        const double v = v0 + a * (v0 - vb);
        inOut[idx] = (isfinite(v) && !isnan(v)) ? static_cast<float>(v) : 0.0f;
    }
}

__global__ void optics_unsharp_vertical_combine_kernel(
    float* inOut,
    const float* JUICER_RESTRICT in,
    int width,
    int height,
    const float* JUICER_RESTRICT k,
    int radius,
    float amount)
{
    if (!inOut || !in || !k || radius <= 0) {
        return;
    }

    const int kLen = 2 * radius + 1;
    const int tileW = blockDim.x;
    const int tileH = blockDim.y + 2 * radius;

    extern __shared__ float shared[];
    float* sWeights = shared;
    float* sTile = shared + kLen;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int tcount = blockDim.x * blockDim.y;
    const int xLocal = threadIdx.x;

    for (int i = tid; i < kLen; i += tcount) {
        sWeights[i] = k[i];
    }
    __syncthreads();

    for (int blockY = blockIdx.y * blockDim.y; blockY < height; blockY += blockDim.y * gridDim.y) {
        for (int blockX = blockIdx.x * blockDim.x; blockX < width; blockX += blockDim.x * gridDim.x) {
            const int x = blockX + threadIdx.x;
            const int y = blockY + threadIdx.y;
            const bool inBounds = (x < width && y < height);
            const int xLoad = blockX + xLocal;
            if (xLoad < width) {
                for (int i = threadIdx.y; i < tileH; i += blockDim.y) {
                    const int yLoad = blockY + i - radius;
                    const int yy = reflect_index_repeat_device(yLoad, height);
                    sTile[i * tileW + xLocal] =
                        in[static_cast<size_t>(yy) * static_cast<size_t>(width) + static_cast<size_t>(xLoad)];
                }
            }

            __syncthreads();

            if (inBounds) {
                double acc = 0.0;
                const int tileY = threadIdx.y + radius;
                for (int j = -radius; j <= radius; ++j) {
                    const float v = sTile[(tileY + j) * tileW + xLocal];
                    const float w = sWeights[j + radius];
                    acc += static_cast<double>(v) * static_cast<double>(w);
                }
                const float blurred = (isfinite(acc) && !isnan(acc)) ? static_cast<float>(acc) : 0.0f;

                const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
                const double v0 = static_cast<double>(inOut[idx]);
                const double vb = static_cast<double>(blurred);
                const double a = static_cast<double>(amount);
                const double outV = v0 + a * (v0 - vb);
                inOut[idx] = (isfinite(outV) && !isnan(outV)) ? static_cast<float>(outV) : 0.0f;
            }

            __syncthreads();
        }
    }
}

__global__ void optics_halation_vertical_apply_kernel(
    float* inOut,
    const float* JUICER_RESTRICT in,
    int width,
    int height,
    const float* JUICER_RESTRICT k,
    int radius,
    float strength)
{
    if (!inOut || !in || !k || radius <= 0) {
        return;
    }

    const float s = strength;
    if (!(s > 0.0f)) {
        return;
    }

    const int kLen = 2 * radius + 1;
    const int tileW = blockDim.x;
    const int tileH = blockDim.y + 2 * radius;

    extern __shared__ float shared[];
    float* sWeights = shared;
    float* sTile = shared + kLen;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int tcount = blockDim.x * blockDim.y;
    const int xLocal = threadIdx.x;

    for (int i = tid; i < kLen; i += tcount) {
        sWeights[i] = k[i];
    }
    __syncthreads();

    for (int blockY = blockIdx.y * blockDim.y; blockY < height; blockY += blockDim.y * gridDim.y) {
        for (int blockX = blockIdx.x * blockDim.x; blockX < width; blockX += blockDim.x * gridDim.x) {
            const int x = blockX + threadIdx.x;
            const int y = blockY + threadIdx.y;
            const bool inBounds = (x < width && y < height);
            const int xLoad = blockX + xLocal;
            if (xLoad < width) {
                for (int i = threadIdx.y; i < tileH; i += blockDim.y) {
                    const int yLoad = blockY + i - radius;
                    const int yy = reflect_index_repeat_device(yLoad, height);
                    sTile[i * tileW + xLocal] =
                        in[static_cast<size_t>(yy) * static_cast<size_t>(width) + static_cast<size_t>(xLoad)];
                }
            }

            __syncthreads();

            if (inBounds) {
                double acc = 0.0;
                const int tileY = threadIdx.y + radius;
                for (int j = -radius; j <= radius; ++j) {
                    const float v = sTile[(tileY + j) * tileW + xLocal];
                    const float w = sWeights[j + radius];
                    acc += static_cast<double>(v) * static_cast<double>(w);
                }
                const float blurred = (isfinite(acc) && !isnan(acc)) ? static_cast<float>(acc) : 0.0f;

                const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
                const double a = static_cast<double>(inOut[idx]);
                const double b = static_cast<double>(blurred);
                const double sd = static_cast<double>(s);
                const double outV = (a + sd * b) / (1.0 + sd);
                inOut[idx] = (isfinite(outV) && !isnan(outV)) ? static_cast<float>(outV) : 0.0f;
            }

            __syncthreads();
        }
    }
}
