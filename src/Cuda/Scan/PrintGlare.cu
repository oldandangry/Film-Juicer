#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "Cuda/JuicerCudaDeviceHelpers.cuh"

#include "Cuda/Scan/PrintGlare.h"
#include "openrand/philox.h"

namespace {
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

    struct LognormalSampleDevice {
        float mean = 0.0f;
        float standardDeviation = 0.0f;
        float normalSample = 0.0f;
    };

    struct GlareGenerationInput {
        float* output = nullptr;
        int width = 0;
        int height = 0;
        std::uint64_t seed = 0;
        std::uint64_t mediumId = 0;
        int originX = 0;
        int originY = 0;
        float percent = 0.0f;
        float roughness = 0.0f;
    };

    __device__ __forceinline__ float lognormal_from_mean_std_device(LognormalSampleDevice sample) {
        const float m2 = sample.mean * sample.mean;
        const float s2 = sample.standardDeviation * sample.standardDeviation;
        const float sigmaSq = logf(1.0f + (s2 / m2));
        const float sigma = sqrtf(fmaxf(0.0f, sigmaSq));
        const float mu = logf(fmaxf(1e-12f, sample.mean)) - 0.5f * sigmaSq;
        return expf(mu + sigma * sample.normalSample);
    }
    __global__ void optics_glare_generate_kernel(GlareGenerationInput input) {
        float* out = input.output;
        const int width = input.width;
        const int height = input.height;
        const float percent = input.percent;
        const float roughness = input.roughness;
        if (!out || width <= 0 || height <= 0) {
            return;
        }
        if (!device_isfinite(percent) || !(percent > 0.0f) || !device_isfinite(roughness)) {
            return;
        }

        const float mean = fmaxf(0.0f, percent);
        const float stddev = fmaxf(0.0f, roughness * percent);
        const std::size_t widthCount = static_cast<std::size_t>(width);
        const std::size_t heightCount = static_cast<std::size_t>(height);
        const std::size_t yStep = static_cast<std::size_t>(blockDim.y) * static_cast<std::size_t>(gridDim.y);
        const std::size_t xStep = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
        for (std::size_t y = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(blockDim.y) + static_cast<std::size_t>(threadIdx.y); y < heightCount; y += yStep) {
            for (std::size_t x = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x); x < widthCount; x += xStep) {
                const std::size_t idx = y * widthCount + x;
                const std::int64_t absXSigned = static_cast<std::int64_t>(input.originX) + static_cast<std::int64_t>(x);
                const std::int64_t absYSigned = static_cast<std::int64_t>(input.originY) + static_cast<std::int64_t>(y);
                const std::uint64_t absX = static_cast<std::uint64_t>(absXSigned);
                const std::uint64_t absY = static_cast<std::uint64_t>(absYSigned);

                GlareRngDevice rng(input.seed,
                                   static_cast<std::uint32_t>(absX),
                                   static_cast<std::uint32_t>(absY),
                                   static_cast<std::uint32_t>(input.mediumId));
                const float n = rng.normal();
                LognormalSampleDevice sample{};
                sample.mean = mean;
                sample.standardDeviation = stddev;
                sample.normalSample = n;
                const float glare = lognormal_from_mean_std_device(sample);

                out[idx] = (device_isfinite(glare) && !isnan(glare)) ? glare : 0.0f;
            }
        }
    }

} // namespace

namespace JuicerCuda {
    cudaError_t launch_print_glare(
        const FocusedScannerPostEffectOptions& options,
        float* temporary,
        float* blurScratch,
        int width,
        int height,
        dim3 blocks,
        dim3 threads,
        cudaStream_t stream,
        const float*& field) {
        field = nullptr;
        if (!std::isfinite(options.glarePercent) || options.glarePercent <= 0.0F ||
            !std::isfinite(options.glareRoughness)) {
            return cudaSuccess;
        }
        const bool blur = options.glareRadius > 0 && options.glareKernel;
        if (!temporary || (blur && !blurScratch)) {
            return cudaErrorInvalidValue;
        }
        GlareGenerationInput glare{};
        glare.output = temporary;
        glare.width = width;
        glare.height = height;
        glare.mediumId = 1;
        glare.seed = options.glareSeed;
        glare.originX = options.glareOriginX;
        glare.originY = options.glareOriginY;
        glare.percent = options.glarePercent;
        glare.roughness = options.glareRoughness;
        optics_glare_generate_kernel<<<blocks, threads, 0, stream>>>(glare);
        auto status = cudaGetLastError();
        if (status == cudaSuccess && blur) {
            status = blur_plane_in_place(
                {temporary, blurScratch, width, height, options.glareKernel, options.glareRadius}, {blocks, threads, stream});
        }
        if (status == cudaSuccess) {
            field = temporary;
        }
        return status;
    }

} // namespace JuicerCuda
