// Cuda/Scan/JuicerCudaScanPipeline.cu
// Pipeline-aligned CUDA TU for print-development handoff and scan-stage kernels.
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "Cuda/JuicerCudaDeviceHelpers.cuh"
#include "openrand/philox.h"

namespace {

    JuicerCuda::PipelineRunParams focused_params_from_direct(
        const JuicerCuda::DirectPipelineRunParams& params) {
        JuicerCuda::PipelineRunParams out{};
        out.src = params.src;
        out.srcRowBytes = params.srcRowBytes;
        out.dst = params.dst;
        out.dstRowBytes = params.dstRowBytes;
        out.width = params.width;
        out.height = params.height;
        out.nComponents = params.nComponents;
        out.filmExpose = params.filmExpose;
        out.filmDevelop = params.filmDevelop;
        out.scanStage = params.scanStage;
        out.filmRaw = params.filmRaw;
        return out;
    }

    JuicerCuda::PipelineRunParams focused_params_from_print(
        const JuicerCuda::PrintPipelineRunParams& params) {
        JuicerCuda::PipelineRunParams out = focused_params_from_direct(
            JuicerCuda::DirectPipelineRunParams{
                params.src,
                params.srcRowBytes,
                params.dst,
                params.dstRowBytes,
                params.width,
                params.height,
                params.nComponents,
                params.filmExpose,
                params.filmDevelop,
                params.scanStage,
                params.filmRaw});
        out.printExpose = params.printExpose;
        out.printDevelop = params.printDevelop;
        return out;
    }

    __device__ __forceinline__ bool scan_log2_xyz_device(
        const JuicerCuda::ScanStagePayload& scanStage,
        const double D_norm[3],
        float log2XYZ[3]) {
        return sample_pchip_float_log2_scan_lut_device(scanStage, D_norm, log2XYZ);
    }

    __device__ __forceinline__ double clamp01d_device(double v) {
        if (v <= 0.0)
            return 0.0;
        if (v >= 1.0)
            return 1.0;
        return v;
    }

    __device__ __forceinline__ double encode_sRGBd_device(double v) {
        if (v <= 0.0031308) {
            return 12.92 * v;
        }
        return 1.055 * pow(v, 1.0 / 2.4) - 0.055;
    }

    __device__ __forceinline__ double encode_gammad_signed_device(double v, double exponent) {
        const double mag = pow(fabs(v), exponent);
        return copysign(mag, v);
    }

    __device__ __forceinline__ double encode_BT2020d_device(double v, const JuicerCuda::CctfPayload& cctf) {
        if (v < static_cast<double>(cctf.b)) {
            return v * 4.5;
        }
        return static_cast<double>(cctf.a) * pow(v, 0.45) - (static_cast<double>(cctf.a) - 1.0);
    }

    __device__ __forceinline__ double encode_ProPhotod_device(double v, const JuicerCuda::CctfPayload& cctf) {
        if (v < static_cast<double>(cctf.linearCutoff)) {
            return v * 16.0;
        }
        return pow(v, static_cast<double>(cctf.gamma));
    }

    __device__ __forceinline__ double encode_DaVinciIntermediated_device(double v, const JuicerCuda::CctfPayload& cctf) {
        const double linear = fmax(0.0, v);
        if (linear <= static_cast<double>(cctf.linearCutoff)) {
            return linear * static_cast<double>(cctf.d);
        }
        return (log2(linear + static_cast<double>(cctf.a)) + static_cast<double>(cctf.b)) * static_cast<double>(cctf.c);
    }

    __device__ __forceinline__ double encode_channel_double_device(const JuicerCuda::CctfPayload& cctf, double v) {
        switch (cctf.kind) {
            case 0:
                return v;
            case 1:
                return encode_gammad_signed_device(v, static_cast<double>(cctf.gamma));
            case 2:
                return encode_sRGBd_device(v);
            case 3:
                return encode_BT2020d_device(v, cctf);
            case 4:
                return encode_ProPhotod_device(v, cctf);
            case 5:
                return encode_DaVinciIntermediated_device(v, cctf);
            default:
                return clamp01d_device(v);
        }
    }

    __device__ __forceinline__ void apply_output_encoding_device(const JuicerCuda::OutputEncodingPayload& enc, double rgb[3]) {
        if (!rgb) {
            return;
        }

        double linear[3];
        if (enc.inputIsOutputSpace) {
            linear[0] = rgb[0];
            linear[1] = rgb[1];
            linear[2] = rgb[2];
        } else {
            linear[0] =
                static_cast<double>(enc.dwgToOutput[0]) * rgb[0] +
                static_cast<double>(enc.dwgToOutput[1]) * rgb[1] +
                static_cast<double>(enc.dwgToOutput[2]) * rgb[2];
            linear[1] =
                static_cast<double>(enc.dwgToOutput[3]) * rgb[0] +
                static_cast<double>(enc.dwgToOutput[4]) * rgb[1] +
                static_cast<double>(enc.dwgToOutput[5]) * rgb[2];
            linear[2] =
                static_cast<double>(enc.dwgToOutput[6]) * rgb[0] +
                static_cast<double>(enc.dwgToOutput[7]) * rgb[1] +
                static_cast<double>(enc.dwgToOutput[8]) * rgb[2];
        }

        if (enc.preserveLinearRange) {
            rgb[0] = linear[0];
            rgb[1] = linear[1];
            rgb[2] = linear[2];
            return;
        }

        if (enc.applyCctfEncoding) {
            rgb[0] = encode_channel_double_device(enc.cctf, linear[0]);
            rgb[1] = encode_channel_double_device(enc.cctf, linear[1]);
            rgb[2] = encode_channel_double_device(enc.cctf, linear[2]);
        } else {
            rgb[0] = linear[0];
            rgb[1] = linear[1];
            rgb[2] = linear[2];
        }

        rgb[0] = clamp01d_device(rgb[0]);
        rgb[1] = clamp01d_device(rgb[1]);
        rgb[2] = clamp01d_device(rgb[2]);
    }

    __device__ __forceinline__ void signal_scan_error_device(int* flag) {
        if (flag) {
            atomicExch(flag, 1);
        }
    }

    __device__ __forceinline__ void mat3_mul_vec_double_device(const float m9[9], const double v3[3], double out3[3]) {
        out3[0] =
            static_cast<double>(m9[0]) * v3[0] +
            static_cast<double>(m9[1]) * v3[1] +
            static_cast<double>(m9[2]) * v3[2];
        out3[1] =
            static_cast<double>(m9[3]) * v3[0] +
            static_cast<double>(m9[4]) * v3[1] +
            static_cast<double>(m9[5]) * v3[2];
        out3[2] =
            static_cast<double>(m9[6]) * v3[0] +
            static_cast<double>(m9[7]) * v3[1] +
            static_cast<double>(m9[8]) * v3[2];
    }

    // --- Scanner glare helpers ---
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

    __device__ __forceinline__ int reflect_index_repeat_wide_device(std::int64_t idx, int size) {
        if (size <= 1) {
            return 0;
        }
        while (idx < 0 || idx >= static_cast<std::int64_t>(size)) {
            if (idx < 0) {
                idx = -idx;
            } else {
                idx = 2 * static_cast<std::int64_t>(size) - idx - 2;
            }
        }
        return static_cast<int>(idx);
    }

} // namespace

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

template <typename Accumulator>
__global__ void optics_blur_horizontal_kernel(
    const float* JUICER_RESTRICT in,
    float* out,
    int width,
    int height,
    const float* JUICER_RESTRICT k,
    int radius) {
    if (!in || !out || !k || radius <= 0 || width <= 0 || height <= 0) {
        return;
    }

    const std::size_t radiusCount = static_cast<std::size_t>(radius);
    const std::size_t kLen = 2U * radiusCount + 1U;
    const std::size_t tileW = static_cast<std::size_t>(blockDim.x) + 2U * radiusCount;

    extern __shared__ float horizontalShared[];
    float* sWeights = horizontalShared;
    float* sTile = horizontalShared + kLen;

    const std::size_t tid = static_cast<std::size_t>(threadIdx.y) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    const std::size_t tcount = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockDim.y);
    const std::size_t yLocal = static_cast<std::size_t>(threadIdx.y);

    for (std::size_t i = tid; i < kLen; i += tcount) {
        sWeights[i] = k[i];
    }
    __syncthreads();

    const std::size_t widthCount = static_cast<std::size_t>(width);
    const std::size_t heightCount = static_cast<std::size_t>(height);
    const std::size_t blockYStep = static_cast<std::size_t>(blockDim.y) * static_cast<std::size_t>(gridDim.y);
    const std::size_t blockXStep = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    for (std::size_t blockY = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(blockDim.y); blockY < heightCount; blockY += blockYStep) {
        for (std::size_t blockX = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x); blockX < widthCount; blockX += blockXStep) {
            const std::size_t x = blockX + static_cast<std::size_t>(threadIdx.x);
            const std::size_t y = blockY + static_cast<std::size_t>(threadIdx.y);
            const bool inBounds = (x < widthCount && y < heightCount);
            const std::size_t yLoad = blockY + yLocal;
            if (yLoad < heightCount) {
                const std::size_t rowBase = yLoad * widthCount;
                for (std::size_t i = static_cast<std::size_t>(threadIdx.x); i < tileW; i += static_cast<std::size_t>(blockDim.x)) {
                    const std::int64_t xLoad = static_cast<std::int64_t>(blockX) + static_cast<std::int64_t>(i) - static_cast<std::int64_t>(radius);
                    const int xx = reflect_index_repeat_wide_device(xLoad, width);
                    sTile[yLocal * tileW + i] = in[rowBase + static_cast<size_t>(xx)];
                }
            }

            __syncthreads();

            if (inBounds) {
                Accumulator acc = 0;
                const std::size_t tileRow = static_cast<std::size_t>(threadIdx.y) * tileW;
                for (std::size_t kernelOffset = 0; kernelOffset < kLen; ++kernelOffset) {
                    const std::size_t tileX = static_cast<std::size_t>(threadIdx.x) + kernelOffset;
                    const float v = sTile[tileRow + tileX];
                    const float w = sWeights[kernelOffset];
                    acc += static_cast<Accumulator>(v) *
                           static_cast<Accumulator>(w);
                }

                out[y * widthCount + x] =
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
    int radius) {
    if (!in || !out || !k || radius <= 0 || width <= 0 || height <= 0) {
        return;
    }

    const std::size_t radiusCount = static_cast<std::size_t>(radius);
    const std::size_t kLen = 2U * radiusCount + 1U;
    const std::size_t tileW = static_cast<std::size_t>(blockDim.x);
    const std::size_t tileH = static_cast<std::size_t>(blockDim.y) + 2U * radiusCount;

    extern __shared__ float verticalShared[];
    float* sWeights = verticalShared;
    float* sTile = verticalShared + kLen;

    const std::size_t tid = static_cast<std::size_t>(threadIdx.y) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    const std::size_t tcount = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockDim.y);
    const std::size_t xLocal = static_cast<std::size_t>(threadIdx.x);

    for (std::size_t i = tid; i < kLen; i += tcount) {
        sWeights[i] = k[i];
    }
    __syncthreads();

    const std::size_t widthCount = static_cast<std::size_t>(width);
    const std::size_t heightCount = static_cast<std::size_t>(height);
    const std::size_t blockYStep = static_cast<std::size_t>(blockDim.y) * static_cast<std::size_t>(gridDim.y);
    const std::size_t blockXStep = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    for (std::size_t blockY = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(blockDim.y); blockY < heightCount; blockY += blockYStep) {
        for (std::size_t blockX = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x); blockX < widthCount; blockX += blockXStep) {
            const std::size_t x = blockX + static_cast<std::size_t>(threadIdx.x);
            const std::size_t y = blockY + static_cast<std::size_t>(threadIdx.y);
            const bool inBounds = (x < widthCount && y < heightCount);
            const std::size_t xLoad = blockX + xLocal;
            if (xLoad < widthCount) {
                for (std::size_t i = static_cast<std::size_t>(threadIdx.y); i < tileH; i += static_cast<std::size_t>(blockDim.y)) {
                    const std::int64_t yLoad = static_cast<std::int64_t>(blockY) + static_cast<std::int64_t>(i) - static_cast<std::int64_t>(radius);
                    const int yy = reflect_index_repeat_wide_device(yLoad, height);
                    sTile[i * tileW + xLocal] =
                        in[static_cast<std::size_t>(yy) * widthCount + xLoad];
                }
            }

            __syncthreads();

            if (inBounds) {
                double acc = 0.0;
                for (std::size_t kernelOffset = 0; kernelOffset < kLen; ++kernelOffset) {
                    const std::size_t tileY = static_cast<std::size_t>(threadIdx.y) + kernelOffset;
                    const float v = sTile[tileY * tileW + xLocal];
                    const float w = sWeights[kernelOffset];
                    acc += static_cast<double>(v) * static_cast<double>(w);
                }

                out[y * widthCount + x] =
                    (isfinite(acc) && !isnan(acc)) ? static_cast<float>(acc) : 0.0f;
            }

            __syncthreads();
        }
    }
}

struct GrainBlurVerticalAccumulateInput {
    const float* horizontalBlur = nullptr;
    float* output = nullptr;
    int width = 0;
    int height = 0;
    const float* kernel = nullptr;
    int radius = 0;
    float weight = 0.0f;
    int initialize = 0;
};

template <bool Weighted>
__device__ __forceinline__ void grain_blur_vertical_accumulate_device(
    GrainBlurVerticalAccumulateInput input,
    float* verticalShared) {
    const float* JUICER_RESTRICT in = input.horizontalBlur;
    float* dst = input.output;
    const int width = input.width;
    const int height = input.height;
    const float* JUICER_RESTRICT k = input.kernel;
    const int radius = input.radius;
    const float weight = input.weight;
    const int initialize = input.initialize;
    if (!in || !dst || !k || radius <= 0 || width <= 0 || height <= 0) {
        return;
    }

    const std::size_t radiusCount = static_cast<std::size_t>(radius);
    const std::size_t kLen = 2U * radiusCount + 1U;
    const std::size_t tileW = static_cast<std::size_t>(blockDim.x);
    const std::size_t tileH = static_cast<std::size_t>(blockDim.y) + 2U * radiusCount;

    float* sWeights = verticalShared;
    float* sTile = verticalShared + kLen;

    const std::size_t tid = static_cast<std::size_t>(threadIdx.y) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    const std::size_t tcount = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockDim.y);
    const std::size_t xLocal = static_cast<std::size_t>(threadIdx.x);

    for (std::size_t i = tid; i < kLen; i += tcount) {
        sWeights[i] = k[i];
    }
    __syncthreads();

    const std::size_t widthCount = static_cast<std::size_t>(width);
    const std::size_t heightCount = static_cast<std::size_t>(height);
    const std::size_t blockYStep = static_cast<std::size_t>(blockDim.y) * static_cast<std::size_t>(gridDim.y);
    const std::size_t blockXStep = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    const float finiteWeight = device_isfinite(weight) ? weight : 0.0f;
    for (std::size_t blockY = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(blockDim.y); blockY < heightCount; blockY += blockYStep) {
        for (std::size_t blockX = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x); blockX < widthCount; blockX += blockXStep) {
            const std::size_t x = blockX + static_cast<std::size_t>(threadIdx.x);
            const std::size_t y = blockY + static_cast<std::size_t>(threadIdx.y);
            const bool inBounds = (x < widthCount && y < heightCount);
            const std::size_t xLoad = blockX + xLocal;
            if (xLoad < widthCount) {
                for (std::size_t i = static_cast<std::size_t>(threadIdx.y); i < tileH; i += static_cast<std::size_t>(blockDim.y)) {
                    const std::int64_t yLoad = static_cast<std::int64_t>(blockY) + static_cast<std::int64_t>(i) - static_cast<std::int64_t>(radius);
                    const int yy = reflect_index_repeat_wide_device(yLoad, height);
                    sTile[i * tileW + xLocal] =
                        in[static_cast<std::size_t>(yy) * widthCount + xLoad];
                }
            }

            __syncthreads();

            if (inBounds) {
                float acc = 0.0f;
                for (std::size_t kernelOffset = 0; kernelOffset < kLen; ++kernelOffset) {
                    const std::size_t tileY = static_cast<std::size_t>(threadIdx.y) + kernelOffset;
                    const float v = sTile[tileY * tileW + xLocal];
                    const float w = sWeights[kernelOffset];
                    acc += v * w;
                }

                const float blurred =
                    (isfinite(acc) && !isnan(acc)) ? acc : 0.0f;
                const std::size_t index = y * widthCount + x;
                const float previous = initialize != 0 ? 0.0f : dst[index];
                float value = previous + blurred;
                if constexpr (Weighted) {
                    value = previous + finiteWeight * blurred;
                }
                dst[index] = device_isfinite(value) ? value : 0.0f;
            }

            __syncthreads();
        }
    }
}

__global__ void grain_blur_vertical_accumulate_kernel(
    GrainBlurVerticalAccumulateInput input) {
    extern __shared__ float grainLayerVerticalShared[];
    grain_blur_vertical_accumulate_device<false>(
        input,
        grainLayerVerticalShared);
}

__global__ void grain_blur_vertical_accumulate_weighted_kernel(
    GrainBlurVerticalAccumulateInput input) {
    extern __shared__ float grainWeightedVerticalShared[];
    grain_blur_vertical_accumulate_device<true>(
        input,
        grainWeightedVerticalShared);
}

struct VerticalUnsharpInput {
    float* output = nullptr;
    const float* horizontalBlur = nullptr;
    int width = 0;
    int height = 0;
    const float* kernel = nullptr;
    int radius = 0;
    float amount = 0.0f;
};

__global__ void optics_unsharp_vertical_combine_kernel(VerticalUnsharpInput input) {
    float* inOut = input.output;
    const float* JUICER_RESTRICT in = input.horizontalBlur;
    const int width = input.width;
    const int height = input.height;
    const float* JUICER_RESTRICT k = input.kernel;
    const int radius = input.radius;
    const float amount = input.amount;
    if (!inOut || !in || !k || radius <= 0 || width <= 0 || height <= 0) {
        return;
    }

    const std::size_t radiusCount = static_cast<std::size_t>(radius);
    const std::size_t kLen = 2U * radiusCount + 1U;
    const std::size_t tileW = static_cast<std::size_t>(blockDim.x);
    const std::size_t tileH = static_cast<std::size_t>(blockDim.y) + 2U * radiusCount;

    extern __shared__ float unsharpShared[];
    float* sWeights = unsharpShared;
    float* sTile = unsharpShared + kLen;

    const std::size_t tid = static_cast<std::size_t>(threadIdx.y) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    const std::size_t tcount = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockDim.y);
    const std::size_t xLocal = static_cast<std::size_t>(threadIdx.x);

    for (std::size_t i = tid; i < kLen; i += tcount) {
        sWeights[i] = k[i];
    }
    __syncthreads();

    const std::size_t widthCount = static_cast<std::size_t>(width);
    const std::size_t heightCount = static_cast<std::size_t>(height);
    const std::size_t blockYStep = static_cast<std::size_t>(blockDim.y) * static_cast<std::size_t>(gridDim.y);
    const std::size_t blockXStep = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    for (std::size_t blockY = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(blockDim.y); blockY < heightCount; blockY += blockYStep) {
        for (std::size_t blockX = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x); blockX < widthCount; blockX += blockXStep) {
            const std::size_t x = blockX + static_cast<std::size_t>(threadIdx.x);
            const std::size_t y = blockY + static_cast<std::size_t>(threadIdx.y);
            const bool inBounds = (x < widthCount && y < heightCount);
            const std::size_t xLoad = blockX + xLocal;
            if (xLoad < widthCount) {
                for (std::size_t i = static_cast<std::size_t>(threadIdx.y); i < tileH; i += static_cast<std::size_t>(blockDim.y)) {
                    const std::int64_t yLoad = static_cast<std::int64_t>(blockY) + static_cast<std::int64_t>(i) - static_cast<std::int64_t>(radius);
                    const int yy = reflect_index_repeat_wide_device(yLoad, height);
                    sTile[i * tileW + xLocal] =
                        in[static_cast<std::size_t>(yy) * widthCount + xLoad];
                }
            }

            __syncthreads();

            if (inBounds) {
                double acc = 0.0;
                for (std::size_t kernelOffset = 0; kernelOffset < kLen; ++kernelOffset) {
                    const std::size_t tileY = static_cast<std::size_t>(threadIdx.y) + kernelOffset;
                    const float v = sTile[tileY * tileW + xLocal];
                    const float w = sWeights[kernelOffset];
                    acc += static_cast<double>(v) * static_cast<double>(w);
                }
                const float blurred = (isfinite(acc) && !isnan(acc)) ? static_cast<float>(acc) : 0.0f;

                const std::size_t idx = y * widthCount + x;
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

// Film stage kernels are defined in their respective TUs.
__global__ void expose_film_raw_kernel(
    JuicerCuda::PipelineRunParams params,
    float* outB,
    float* outG,
    float* outR);
__global__ void develop_film_density_kernel(
    JuicerCuda::PipelineRunParams params,
    float* outC,
    float* outM,
    float* outY);
__global__ void develop_film_density_from_raw_kernel(
    JuicerCuda::PipelineRunParams params,
    const float* inB,
    const float* inG,
    const float* inR,
    float* outC,
    float* outM,
    float* outY);
__global__ void grain_clear_kernel(float* out, int n);
__global__ void grain_accumulate_kernel(
    float* dst,
    const float* src,
    int n,
    int initialize);
__global__ void grain_accumulate_weighted_kernel(
    float* dst,
    const float* src,
    int n,
    float weight,
    int initialize);
__global__ void grain_scale_kernel(float* inOut, int n, float scale);
__global__ void grain_reconstruct_kernel(
    float* inOutMean,
    const float* delta,
    int n,
    float scale,
    int addMean);
__global__ void grain_form_delta_kernel(
    float* inOut,
    const float* sub,
    int n,
    float bias);
__global__ void grain_subtract_kernel(float* inOut, const float* sub, int n, float amplitude);

struct GrainMixSharedInput {
    float* outputDelta = nullptr;
    const float* independentDelta = nullptr;
    const float* sharedDelta = nullptr;
    int count = 0;
    float sharedWeight = 0.0f;
    float independentWeight = 0.0f;
    float amplitude = 0.0f;
};

__global__ void grain_mix_shared_kernel(GrainMixSharedInput input) {
    if (input.count <= 0) {
        return;
    }
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    if (index >= static_cast<std::size_t>(input.count) || !input.outputDelta) {
        return;
    }
    const float sharedWeight = fminf(fmaxf(input.sharedWeight, 0.0f), 1.0f);
    const float independentWeight = fminf(fmaxf(input.independentWeight, 0.0f), 1.0f);
    float amplitude = device_isfinite(input.amplitude) ? input.amplitude : 1.0f;
    if (amplitude < 0.0f) {
        amplitude = 0.0f;
    }
    const float shared =
        (input.sharedDelta && sharedWeight > 0.0f) ? input.sharedDelta[index] : 0.0f;
    const float independent =
        (input.independentDelta && independentWeight > 0.0f) ? input.independentDelta[index] : 0.0f;
    const float value = amplitude * (sharedWeight * shared + independentWeight * independent);
    input.outputDelta[index] = device_isfinite(value) ? value : 0.0f;
}

struct GrainDebugAverageInput {
    float* outputR = nullptr;
    float* outputG = nullptr;
    float* outputB = nullptr;
    const float* input0 = nullptr;
    const float* input1 = nullptr;
    const float* input2 = nullptr;
    int count = 0;
    float offset = 0.0f;
    float scale = 0.0f;
};

__global__ void grain_debug_encode_avg3_kernel(GrainDebugAverageInput input) {
    if (input.count <= 0) {
        return;
    }
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    if (index >= static_cast<std::size_t>(input.count)) {
        return;
    }
    if (!input.outputR || !input.outputG || !input.outputB ||
        !input.input0 || !input.input1 || !input.input2) {
        return;
    }
    const float scale = device_isfinite(input.scale) ? input.scale : 1.0f;
    const float offset = device_isfinite(input.offset) ? input.offset : 0.0f;
    const float value0 = input.input0[index];
    const float value1 = input.input1[index];
    const float value2 = input.input2[index];
    float average = (value0 + value1 + value2) * (1.0f / 3.0f);
    if (!device_isfinite(average)) {
        average = 0.0f;
    }
    const float output = offset + average * scale;
    input.outputR[index] = output;
    input.outputG[index] = output;
    input.outputB[index] = output;
}
__global__ void grain_apply_simple_kernel(
    JuicerCuda::GrainPayload grain,
    int width,
    int height,
    const float* inDensity,
    float* outGrain,
    int channelIndex);
__global__ void grain_prepare_frame_uniforms_kernel(
    JuicerCuda::GrainPayload grain);
__global__ void grain_layer_kernel(
    JuicerCuda::GrainPayload grain,
    int width,
    int height,
    const float* inDensity,
    float* outGrain,
    int channelIndex,
    int sublayerIndex);
__global__ void grain_layer_triplet_kernel(
    JuicerCuda::GrainPayload grain,
    int width,
    int height,
    const float* inDensity,
    float* outGrain,
    int channelIndex);

namespace {

    __device__ __forceinline__ std::uint64_t splitmix64_device(std::uint64_t x) {
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    }

    __device__ __forceinline__ float hash_to_unit_device(std::uint64_t h) {
        constexpr float kInv = 1.0f / 16777216.0f; // 2^24
        return static_cast<float>((h >> 40) & 0xFFFFFFu) * kInv;
    }

    __device__ __forceinline__ float smoothstep_device(float edge0, float edge1, float x) {
        if (edge1 <= edge0) {
            return (x < edge0) ? 0.0f : 1.0f;
        }
        float t = (x - edge0) / (edge1 - edge0);
        t = fminf(fmaxf(t, 0.0f), 1.0f);
        return t * t * (3.0f - 2.0f * t);
    }

    struct DefectSampleLocation {
        float xMm = 0.0f;
        float yMm = 0.0f;
        float pixelSizeUm = 0.0f;
    };

    struct DustMaskParams {
        float amount = 0.0f;
        std::uint64_t seed = 0;
        float cellMm = 0.0f;
        float lambdaPerMm2 = 0.0f;
        float sizeUm = 0.0f;
        float strength = 0.0f;
        float brightMix = 0.0f;
        float brightScale = 0.0f;
    };

    struct ScratchMaskParams {
        float amount = 0.0f;
        std::uint64_t seed = 0;
        float cellMmX = 0.0f;
        float cellMmY = 0.0f;
        float lambdaPerMm = 0.0f;
        float widthUm = 0.0f;
        float strength = 0.0f;
        float maxAngleRad = 0.0f;
        float brightMix = 0.0f;
        float brightScale = 0.0f;
    };

    struct GateMaskParams {
        float dustAmount = 0.0f;
        float scratchAmount = 0.0f;
        std::uint64_t dustSeed = 0;
        std::uint64_t scratchSeed = 0;
    };

    __device__ __forceinline__ float dust_mask_device(
        DefectSampleLocation location,
        DustMaskParams params) {
        const float amount = params.amount;
        const float x = location.xMm;
        const float y = location.yMm;
        const float pixelSizeUm = location.pixelSizeUm;
        const std::uint64_t seed = params.seed;
        const float cellMm = params.cellMm;
        const float lambdaPerMm2 = params.lambdaPerMm2;
        const float sizeUm = params.sizeUm;
        const float strength = params.strength;
        const float brightMix = params.brightMix;
        const float brightScale = params.brightScale;
        if (!(amount > 0.0f) || !(pixelSizeUm > 0.0f) || !(cellMm > 0.0f)) {
            return 0.0f;
        }
        const float amountProb = fmaxf(amount, 0.0f);
        const float probScale = amountProb * amountProb;
        const float intensityScale = amountProb;
        const int cellX = static_cast<int>(floorf(x / cellMm));
        const int cellY = static_cast<int>(floorf(y / cellMm));
        std::uint64_t h = splitmix64_device(seed ^
                                            (static_cast<std::uint64_t>(cellX) * 0x8EBC6AF09C88C6E3ULL) ^
                                            (static_cast<std::uint64_t>(cellY) * 0x9E3779B97F4A7C15ULL));
        const float u = hash_to_unit_device(h);
        const float areaMm2 = cellMm * cellMm;
        const float p = 1.0f - expf(-lambdaPerMm2 * probScale * areaMm2);
        if (u >= p) {
            return 0.0f;
        }
        const float u1 = hash_to_unit_device(h ^ 0xBF58476D1CE4E5B9ULL);
        const float u2 = hash_to_unit_device(h ^ 0x94D049BB133111EBULL);
        const float u3 = hash_to_unit_device(h ^ 0xD6E8FEB86659FD93ULL);
        const float u4 = hash_to_unit_device(h ^ 0xA5A5A5A5A5A5A5A5ULL);
        const float u5 = hash_to_unit_device(h ^ 0x8EBC6AF09C88C6E3ULL);
        const float cx = (static_cast<float>(cellX) + u1) * cellMm;
        const float cy = (static_cast<float>(cellY) + u2) * cellMm;
        const float baseRadius = fmaxf(0.0005f, sizeUm * 0.001f);
        const float radius = baseRadius * (0.5f + 1.2f * u3);
        const float dx = x - cx;
        const float dy = y - cy;
        const float dist = sqrtf(dx * dx + dy * dy);
        const float edge = fmaxf(0.0005f, radius * 0.6f);
        const float mask = 1.0f - smoothstep_device(radius, radius + edge, dist);
        float intensity = strength * intensityScale * (0.5f + 0.5f * u4);
        if (u5 < brightMix) {
            intensity *= brightScale;
            return -mask * intensity;
        }
        return mask * intensity;
    }

    __device__ __forceinline__ float scratch_mask_device(
        DefectSampleLocation location,
        ScratchMaskParams params) {
        const float amount = params.amount;
        const float x = location.xMm;
        const float y = location.yMm;
        const float pixelSizeUm = location.pixelSizeUm;
        const std::uint64_t seed = params.seed;
        const float cellMmX = params.cellMmX;
        const float cellMmY = params.cellMmY;
        const float lambdaPerMm = params.lambdaPerMm;
        const float widthUm = params.widthUm;
        const float strength = params.strength;
        const float maxAngleRad = params.maxAngleRad;
        const float brightMix = params.brightMix;
        const float brightScale = params.brightScale;
        if (!(amount > 0.0f) || !(pixelSizeUm > 0.0f) || !(cellMmX > 0.0f) || !(cellMmY > 0.0f)) {
            return 0.0f;
        }
        const float amountProb = fmaxf(amount, 0.0f);
        const float probScale = amountProb * amountProb;
        const float intensityScale = amountProb;
        const int cellX = static_cast<int>(floorf(x / cellMmX));
        const int cellY = static_cast<int>(floorf(y / cellMmY));
        std::uint64_t h = splitmix64_device(seed ^
                                            (static_cast<std::uint64_t>(cellX) * 0xC6A4A7935BD1E995ULL) ^
                                            (static_cast<std::uint64_t>(cellY) * 0xD2B74407B1CE6E93ULL));
        const float u = hash_to_unit_device(h);
        const float p = 1.0f - expf(-lambdaPerMm * probScale * cellMmY);
        if (u >= p) {
            return 0.0f;
        }
        const float u1 = hash_to_unit_device(h ^ 0xBF58476D1CE4E5B9ULL);
        const float u2 = hash_to_unit_device(h ^ 0x94D049BB133111EBULL);
        const float u3 = hash_to_unit_device(h ^ 0xA5A5A5A5A5A5A5A5ULL);
        const float u4 = hash_to_unit_device(h ^ 0xD6E8FEB86659FD93ULL);
        const float u5 = hash_to_unit_device(h ^ 0x9E3779B97F4A7C15ULL);
        const float u6 = hash_to_unit_device(h ^ 0x8EBC6AF09C88C6E3ULL);
        const float cx = (static_cast<float>(cellX) + u1) * cellMmX;
        const float cy = (static_cast<float>(cellY) + u2) * cellMmY;
        const float baseWidth = fmaxf(0.0005f, widthUm * 0.001f);
        const float width = baseWidth * (0.6f + 1.4f * u3);
        const float halfLen = cellMmY * (0.35f + 0.4f * u4);
        const float angle = (u5 * 2.0f - 1.0f) * maxAngleRad;
        const float c = cosf(angle);
        const float s = sinf(angle);
        const float dx = x - cx;
        const float dy = y - cy;
        const float dist = fabsf(dx * c - dy * s);
        const float along = fabsf(dx * s + dy * c);
        if (along > halfLen) {
            return 0.0f;
        }
        const float edge = fmaxf(0.0005f, width * 0.8f);
        const float mask = 1.0f - smoothstep_device(width, width + edge, dist);
        float intensity = strength * intensityScale * (0.5f + 0.5f * u2);
        if (u6 < brightMix) {
            intensity *= brightScale;
            return -mask * intensity;
        }
        return mask * intensity;
    }

    __device__ __forceinline__ float gate_mask_device(
        DefectSampleLocation location,
        GateMaskParams params) {
        if (!(location.pixelSizeUm > 0.0f)) {
            return 0.0f;
        }
        constexpr float kGateDustCellUm = 600.0f;
        constexpr float kGateDustBaseProb = 0.01f;
        constexpr float kGateDustSizeUm = 28.0f;
        constexpr float kGateDustStrength = 0.35f;
        constexpr float kGateDustBrightMix = 0.15f;
        constexpr float kGateDustBrightScale = 0.5f;

        constexpr float kGateScratchCellUmX = 3500.0f;
        constexpr float kGateScratchCellUmY = 3500.0f;
        constexpr float kGateScratchBaseProb = 0.008f;
        constexpr float kGateScratchWidthUm = 12.0f;
        constexpr float kGateScratchStrength = 0.30f;
        constexpr float kGateScratchBrightMix = 0.08f;
        constexpr float kGateScratchBrightScale = 0.4f;
        constexpr float kGateScratchMaxAngle = 0.08726646f;

        const float gateDustCellMm = kGateDustCellUm * 0.001f;
        const float gateScratchCellMmX = kGateScratchCellUmX * 0.001f;
        const float gateScratchCellMmY = kGateScratchCellUmY * 0.001f;

        DustMaskParams dustParams{};
        dustParams.amount = params.dustAmount;
        dustParams.seed = params.dustSeed;
        dustParams.cellMm = gateDustCellMm;
        dustParams.lambdaPerMm2 = kGateDustBaseProb;
        dustParams.sizeUm = kGateDustSizeUm;
        dustParams.strength = kGateDustStrength;
        dustParams.brightMix = kGateDustBrightMix;
        dustParams.brightScale = kGateDustBrightScale;
        const float gateDust = dust_mask_device(location, dustParams);

        ScratchMaskParams scratchParams{};
        scratchParams.amount = params.scratchAmount;
        scratchParams.seed = params.scratchSeed;
        scratchParams.cellMmX = gateScratchCellMmX;
        scratchParams.cellMmY = gateScratchCellMmY;
        scratchParams.lambdaPerMm = kGateScratchBaseProb;
        scratchParams.widthUm = kGateScratchWidthUm;
        scratchParams.strength = kGateScratchStrength;
        scratchParams.maxAngleRad = kGateScratchMaxAngle;
        scratchParams.brightMix = kGateScratchBrightMix;
        scratchParams.brightScale = kGateScratchBrightScale;
        const float gateScratch = scratch_mask_device(location, scratchParams);
        float gateMask = gateDust + gateScratch;
        if (!device_isfinite(gateMask)) {
            gateMask = 0.0f;
        }
        return gateMask;
    }

    struct GateMaskSample {
        const float* mask = nullptr;
        int width = 0;
        int height = 0;
        float x = 0.0f;
        float y = 0.0f;
    };

    __device__ __forceinline__ float sample_gate_mask_device(GateMaskSample sample) {
        const float* mask = sample.mask;
        const int width = sample.width;
        const int height = sample.height;
        const float x = sample.x;
        const float y = sample.y;
        if (!mask || width <= 0 || height <= 0) {
            return 0.0f;
        }
        const float fx = fminf(fmaxf(x, 0.0f), static_cast<float>(width - 1));
        const float fy = fminf(fmaxf(y, 0.0f), static_cast<float>(height - 1));
        const int x0 = static_cast<int>(floorf(fx));
        const int y0 = static_cast<int>(floorf(fy));
        const int x1 = (x0 + 1 < width) ? (x0 + 1) : x0;
        const int y1 = (y0 + 1 < height) ? (y0 + 1) : y0;
        const float tx = fx - static_cast<float>(x0);
        const float ty = fy - static_cast<float>(y0);
        const int row0 = y0 * width;
        const int row1 = y1 * width;
        const float m00 = ldg_f(mask + row0 + x0);
        const float m10 = ldg_f(mask + row0 + x1);
        const float m01 = ldg_f(mask + row1 + x0);
        const float m11 = ldg_f(mask + row1 + x1);
        const float m0 = m00 + (m10 - m00) * tx;
        const float m1 = m01 + (m11 - m01) * tx;
        return m0 + (m1 - m0) * ty;
    }

} // namespace

__global__ void develop_print_density_kernel(
    JuicerCuda::PipelineRunParams params,
    float* ioC,
    float* ioM,
    float* ioY) {
    if (!params.printExpose.active) {
        return;
    }

    if (!ioC || !ioM || !ioY || params.width <= 0 || params.height <= 0) {
        return;
    }

    const std::size_t width = static_cast<std::size_t>(params.width);
    const std::size_t height = static_cast<std::size_t>(params.height);
    const std::size_t yStep = static_cast<std::size_t>(blockDim.y) * static_cast<std::size_t>(gridDim.y);
    const std::size_t xStep = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    for (std::size_t y = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(blockDim.y) + static_cast<std::size_t>(threadIdx.y); y < height; y += yStep) {
        for (std::size_t x = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x); x < width; x += xStep) {
            const std::size_t idx = y * width + x;
            float D_cmy[3] = {ioC[idx], ioM[idx], ioY[idx]};
            apply_print_pipeline_device(params.printExpose, params.printDevelop, D_cmy);
            ioC[idx] = D_cmy[0];
            ioM[idx] = D_cmy[1];
            ioY[idx] = D_cmy[2];
        }
    }
}

__global__ void build_enlarger_print_linear_exposure_kernel(
    JuicerCuda::PipelineRunParams params,
    const float* densityC,
    const float* densityM,
    const float* densityY,
    JuicerCuda::EnlargerPrintLinearExposurePlanes planes) {
    const std::size_t width = static_cast<std::size_t>(params.width);
    const std::size_t height = static_cast<std::size_t>(params.height);
    const std::size_t yStep =
        static_cast<std::size_t>(blockDim.y) *
        static_cast<std::size_t>(gridDim.y);
    const std::size_t xStep =
        static_cast<std::size_t>(blockDim.x) *
        static_cast<std::size_t>(gridDim.x);
    for (std::size_t y =
             static_cast<std::size_t>(blockIdx.y) *
                 static_cast<std::size_t>(blockDim.y) +
             static_cast<std::size_t>(threadIdx.y);
         y < height;
         y += yStep) {
        for (std::size_t x =
                 static_cast<std::size_t>(blockIdx.x) *
                     static_cast<std::size_t>(blockDim.x) +
                 static_cast<std::size_t>(threadIdx.x);
             x < width;
             x += xStep) {
            const std::size_t frameOffset = y * width + x;
            const float filmDensityCmy[3] = {
                densityC[frameOffset],
                densityM[frameOffset],
                densityY[frameOffset]};
            float innerRaw[3] = {0.0f, 0.0f, 0.0f};
            float linearForDiffusion[3] = {0.0f, 0.0f, 0.0f};
            if (!print_inner_raw_device(
                    params.printExpose,
                    filmDensityCmy,
                    innerRaw)) {
                continue;
            }
            print_linear_for_diffusion_device(
                params.printExpose,
                innerRaw,
                linearForDiffusion);
            const std::size_t planeOffset =
                y * planes.rowStrideFloats + x;
            planes.redSensitiveCForming[planeOffset] =
                linearForDiffusion[0];
            planes.greenSensitiveMForming[planeOffset] =
                linearForDiffusion[1];
            planes.blueSensitiveYForming[planeOffset] =
                linearForDiffusion[2];
        }
    }
}

__global__ void develop_print_density_from_enlarger_linear_kernel(
    JuicerCuda::PipelineRunParams params,
    JuicerCuda::EnlargerPrintLinearExposurePlanes planes,
    float* densityC,
    float* densityM,
    float* densityY) {
    const std::size_t width = static_cast<std::size_t>(params.width);
    const std::size_t height = static_cast<std::size_t>(params.height);
    const std::size_t yStep =
        static_cast<std::size_t>(blockDim.y) *
        static_cast<std::size_t>(gridDim.y);
    const std::size_t xStep =
        static_cast<std::size_t>(blockDim.x) *
        static_cast<std::size_t>(gridDim.x);
    for (std::size_t y =
             static_cast<std::size_t>(blockIdx.y) *
                 static_cast<std::size_t>(blockDim.y) +
             static_cast<std::size_t>(threadIdx.y);
         y < height;
         y += yStep) {
        for (std::size_t x =
                 static_cast<std::size_t>(blockIdx.x) *
                     static_cast<std::size_t>(blockDim.x) +
                 static_cast<std::size_t>(threadIdx.x);
             x < width;
             x += xStep) {
            const std::size_t frameOffset = y * width + x;
            const std::size_t planeOffset =
                y * planes.rowStrideFloats + x;
            const float diffusedRaw[3] = {
                planes.redSensitiveCForming[planeOffset],
                planes.greenSensitiveMForming[planeOffset],
                planes.blueSensitiveYForming[planeOffset]};
            float logPrint[3] = {0.0f, 0.0f, 0.0f};
            float printDensityCmy[3] = {0.0f, 0.0f, 0.0f};
            print_log_encode_diffused_device(diffusedRaw, logPrint);
            print_sample_density_curves_device(
                params.printDevelop,
                logPrint,
                printDensityCmy);
            densityC[frameOffset] = printDensityCmy[0];
            densityM[frameOffset] = printDensityCmy[1];
            densityY[frameOffset] = printDensityCmy[2];
        }
    }
}

namespace {

    __device__ __forceinline__ int clamp_index_device(int idx, int maxIndex) {
        if (idx < 0)
            return 0;
        if (idx > maxIndex)
            return maxIndex;
        return idx;
    }

    struct MitchellPlaneSample {
        const float* plane = nullptr;
        int width = 0;
        int height = 0;
        float x = 0.0f;
        float y = 0.0f;
    };

    __device__ __forceinline__ float sample_plane_mitchell_device(MitchellPlaneSample sample) {
        const float* JUICER_RESTRICT plane = sample.plane;
        const int width = sample.width;
        const int height = sample.height;
        float x = sample.x;
        float y = sample.y;
        if (!plane || width <= 0 || height <= 0) {
            return 0.0f;
        }

        const float maxX = static_cast<float>(width - 1);
        const float maxY = static_cast<float>(height - 1);
        x = fminf(fmaxf(x, 0.0f), maxX);
        y = fminf(fmaxf(y, 0.0f), maxY);

        const int ix = static_cast<int>(floorf(x));
        const int iy = static_cast<int>(floorf(y));
        const double tx = static_cast<double>(x - static_cast<float>(ix));
        const double ty = static_cast<double>(y - static_cast<float>(iy));

        double wx[4];
        double wy[4];
        wx[0] = mitchell_weight_device(tx + 1.0);
        wx[1] = mitchell_weight_device(tx);
        wx[2] = mitchell_weight_device(tx - 1.0);
        wx[3] = mitchell_weight_device(tx - 2.0);
        wy[0] = mitchell_weight_device(ty + 1.0);
        wy[1] = mitchell_weight_device(ty);
        wy[2] = mitchell_weight_device(ty - 1.0);
        wy[3] = mitchell_weight_device(ty - 2.0);

        const int maxXi = width - 1;
        const int maxYi = height - 1;
        double acc = 0.0;
        for (int j = 0; j < 4; ++j) {
            const int sy = clamp_index_device(iy + j - 1, maxYi);
            const double wyj = wy[j];
            const std::size_t row = static_cast<std::size_t>(sy) * static_cast<std::size_t>(width);
            for (int i = 0; i < 4; ++i) {
                const int sx = clamp_index_device(ix + i - 1, maxXi);
                acc += wyj * wx[i] * static_cast<double>(plane[row + static_cast<std::size_t>(sx)]);
            }
        }
        return static_cast<float>(acc);
    }

    template <typename Params>
    __device__ __forceinline__ bool load_spatial_dir_log_raw_for_final_develop_device(
        const Params& params,
        std::size_t pixelIndex,
        const float* rgbIn,
        float logE_raw[3]) {
        const JuicerCuda::FilmDevelopPayload& dev = params.filmDevelop;
        int cachedLogRawMask = 0;
        if (dev.spatialDir.logRawB) {
            logE_raw[0] = ldg_f(dev.spatialDir.logRawB + pixelIndex);
            cachedLogRawMask |= kJuicerLogRawMaskB;
        }
        if (dev.spatialDir.logRawG) {
            logE_raw[1] = ldg_f(dev.spatialDir.logRawG + pixelIndex);
            cachedLogRawMask |= kJuicerLogRawMaskG;
        }
        if (dev.spatialDir.logRawR) {
            logE_raw[2] = ldg_f(dev.spatialDir.logRawR + pixelIndex);
            cachedLogRawMask |= kJuicerLogRawMaskR;
        }
        const int missingLogRawMask = kJuicerLogRawMaskBgr & ~cachedLogRawMask;
        if (missingLogRawMask == 0) {
            return true;
        }

        if (rgbIn) {
            compute_logE_raw_selected_device(params, rgbIn, missingLogRawMask, logE_raw);
            return true;
        }

        const int nC = params.nComponents;
        if (!params.src || params.srcRowBytes == 0 || params.width <= 0 ||
            !(nC == 3 || nC == 4)) {
            return false;
        }

        const int y = static_cast<int>(pixelIndex / static_cast<std::size_t>(params.width));
        const int x =
            static_cast<int>(pixelIndex - static_cast<std::size_t>(y) *
                                              static_cast<std::size_t>(params.width));
        const std::size_t pixelBytes = static_cast<std::size_t>(nC) * sizeof(float);
        const char* srcRow =
            reinterpret_cast<const char*>(params.src) +
            static_cast<std::size_t>(y) * params.srcRowBytes;
        const float* srcPix =
            reinterpret_cast<const float*>(srcRow + static_cast<std::size_t>(x) * pixelBytes);

        const float sourceRgb[3] = {srcPix[0], srcPix[1], srcPix[2]};
        compute_logE_raw_selected_device(params, sourceRgb, missingLogRawMask, logE_raw);
        return true;
    }

    template <typename Params>
    __device__ __forceinline__ bool compute_capture_film_density_device(
        const Params& params,
        int x,
        int y,
        const float rgbIn[3],
        float densityCmy[3]) {
        const JuicerCuda::FilmDevelopPayload& dev = params.filmDevelop;
        const bool useSpatialDir =
            juicer_cuda_spatial_dir_filtered_correction_active_device(dev);

        if (useSpatialDir) {
            const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) +
                               static_cast<size_t>(x);
            float logE_raw[3] = {0.0f, 0.0f, 0.0f};
            if (!load_spatial_dir_log_raw_for_final_develop_device(
                    params,
                    idx,
                    rgbIn,
                    logE_raw)) {
                return false;
            }
            juicer_cuda_develop_dir_final_device(dev, logE_raw, idx, densityCmy);
            return true;
        }

        float logE_raw[3] = {0.0f, 0.0f, 0.0f};
        float logE_sanitized[3] = {0.0f, 0.0f, 0.0f};
        float layerPre[3] = {0.0f, 0.0f, 0.0f};
        FilmDevelopIntermediatesDevice intermediates{};
        intermediates.logERaw = logE_raw;
        intermediates.logESanitized = logE_sanitized;
        intermediates.layerPre = layerPre;
        compute_logE_and_layer_pre_device(params, rgbIn, intermediates);

        if (dev.dir.active) {
            float logE_corr[3] = {
                logE_sanitized[0],
                logE_sanitized[1],
                logE_sanitized[2]};
            apply_dir_runtime_logE_device(
                logE_corr,
                layerPre,
                dev.dir,
                dev.densB,
                dev.densG,
                dev.densR);

            const JuicerCuda::DeviceCurveView cB =
                dev.dirPrecorrected ? dev.dirDensB : dev.densB;
            const JuicerCuda::DeviceCurveView cG =
                dev.dirPrecorrected ? dev.dirDensG : dev.densG;
            const JuicerCuda::DeviceCurveView cR =
                dev.dirPrecorrected ? dev.dirDensR : dev.densR;

            densityCmy[2] =
                sample_density_at_logE_device(cB, logE_corr[0], dev.gammaFactorB);
            densityCmy[1] =
                sample_density_at_logE_device(cG, logE_corr[1], dev.gammaFactorG);
            densityCmy[0] =
                sample_density_at_logE_device(cR, logE_corr[2], dev.gammaFactorR);
        } else {
            // Map B/G/R layer densities to C/M/Y dyes.
            densityCmy[0] = layerPre[2];
            densityCmy[1] = layerPre[1];
            densityCmy[2] = layerPre[0];
        }
        return true;
    }

    template <typename Params>
    __global__ void pipeline_direct_kernel(Params params) {
        const JuicerCuda::ScanStagePayload& scan = params.scanStage;
        if (params.width <= 0 || params.height <= 0) {
            return;
        }
        const std::size_t xIndex = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
        const std::size_t yIndex = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(blockDim.y) + static_cast<std::size_t>(threadIdx.y);
        if (xIndex >= static_cast<std::size_t>(params.width) || yIndex >= static_cast<std::size_t>(params.height)) {
            return;
        }
        const int x = static_cast<int>(xIndex);
        const int y = static_cast<int>(yIndex);

        if (!params.src || !params.dst || params.srcRowBytes == 0 || params.dstRowBytes == 0) {
            return;
        }

        const int nC = params.nComponents;
        if (!(nC == 3 || nC == 4)) {
            return;
        }

        const std::size_t pixelBytes = static_cast<std::size_t>(nC) * sizeof(float);
        const char* srcRow = reinterpret_cast<const char*>(params.src) + static_cast<std::size_t>(y) * params.srcRowBytes;
        const float* srcPix = reinterpret_cast<const float*>(srcRow + static_cast<std::size_t>(x) * pixelBytes);

        const float rgbIn[3] = {srcPix[0], srcPix[1], srcPix[2]};

        float D_cmy[3] = {0.0f, 0.0f, 0.0f};
        if (!compute_capture_film_density_device(params, x, y, rgbIn, D_cmy)) {
            return;
        }

        if constexpr (requires { params.printExpose; params.printDevelop; }) {
            apply_print_pipeline_device(params.printExpose, params.printDevelop, D_cmy);
        }

        // Scan: normalize density -> logXYZ
        double D_norm[3];
        if (scan.scanTables.mediumIsNegative) {
            D_norm[0] = (static_cast<double>(D_cmy[0]) + static_cast<double>(scan.scanTables.min_cmy[0])) * static_cast<double>(scan.scanTables.inv_max_cmy[0]);
            D_norm[1] = (static_cast<double>(D_cmy[1]) + static_cast<double>(scan.scanTables.min_cmy[1])) * static_cast<double>(scan.scanTables.inv_max_cmy[1]);
            D_norm[2] = (static_cast<double>(D_cmy[2]) + static_cast<double>(scan.scanTables.min_cmy[2])) * static_cast<double>(scan.scanTables.inv_max_cmy[2]);
        } else {
            D_norm[0] = (static_cast<double>(D_cmy[0]) - static_cast<double>(scan.scanTables.min_cmy[0])) * static_cast<double>(scan.scanTables.inv_max_cmy[0]);
            D_norm[1] = (static_cast<double>(D_cmy[1]) - static_cast<double>(scan.scanTables.min_cmy[1])) * static_cast<double>(scan.scanTables.inv_max_cmy[1]);
            D_norm[2] = (static_cast<double>(D_cmy[2]) - static_cast<double>(scan.scanTables.min_cmy[2])) * static_cast<double>(scan.scanTables.inv_max_cmy[2]);
        }

        float log2XYZ[3] = {nanf(""), nanf(""), nanf("")};
        (void)scan_log2_xyz_device(scan, D_norm, log2XYZ);
        double xyz[3] = {
            static_cast<double>(exp2f(log2XYZ[0])),
            static_cast<double>(exp2f(log2XYZ[1])),
            static_cast<double>(exp2f(log2XYZ[2]))};

        if (scan.correctionActive) {
            const double correctedY = fmin(fmax(
                                               static_cast<double>(scan.correctionSlope) * xyz[1] +
                                                   static_cast<double>(scan.correctionOffset),
                                               0.0),
                                           1.0);
            const double scale = correctedY / (xyz[1] + 1e-10);
            xyz[0] *= scale;
            xyz[1] *= scale;
            xyz[2] *= scale;
        }
        const std::size_t idx = yIndex * static_cast<std::size_t>(params.width) + xIndex;
        if (scan.glarePercent) {
            const double glare = static_cast<double>(scan.glarePercent[idx]) * 0.01;
            xyz[0] += glare * static_cast<double>(scan.scanColor.illuminantXYZ[0]);
            xyz[1] += glare * static_cast<double>(scan.scanColor.illuminantXYZ[1]);
            xyz[2] += glare * static_cast<double>(scan.scanColor.illuminantXYZ[2]);
        }

        double adapted[3];
        mat3_mul_vec_double_device(scan.scanColor.cat02, xyz, adapted);
        double rgbOut[3];
        mat3_mul_vec_double_device(scan.scanColor.xyzToRgb, adapted, rgbOut);

        if (!isfinite(rgbOut[0]) || !isfinite(rgbOut[1]) || !isfinite(rgbOut[2])) {
            signal_scan_error_device(scan.scanErrorFlag);
            if (scan.linearRgbR && scan.linearRgbG && scan.linearRgbB) {
                scan.linearRgbR[idx] = 0.0f;
                scan.linearRgbG[idx] = 0.0f;
                scan.linearRgbB[idx] = 0.0f;
                return;
            }
            const std::size_t pixelBytes = static_cast<std::size_t>(nC) * sizeof(float);
            char* dstRow = reinterpret_cast<char*>(params.dst) + static_cast<std::size_t>(y) * params.dstRowBytes;
            float* dstPix = reinterpret_cast<float*>(dstRow + static_cast<std::size_t>(x) * pixelBytes);
            if (dstPix) {
                dstPix[0] = 0.0f;
                dstPix[1] = 0.0f;
                dstPix[2] = 0.0f;
                if (nC == 4) {
                    const char* srcRow = reinterpret_cast<const char*>(params.src) + static_cast<std::size_t>(y) * params.srcRowBytes;
                    const float* srcPix = reinterpret_cast<const float*>(srcRow + static_cast<std::size_t>(x) * pixelBytes);
                    dstPix[3] = srcPix ? srcPix[3] : 1.0f;
                }
            }
            return;
        }

        if (scan.linearRgbR && scan.linearRgbG && scan.linearRgbB) {
            scan.linearRgbR[idx] = static_cast<float>(rgbOut[0]);
            scan.linearRgbG[idx] = static_cast<float>(rgbOut[1]);
            scan.linearRgbB[idx] = static_cast<float>(rgbOut[2]);
            return;
        }

        // Output encoding + clamp
        apply_output_encoding_device(scan.scanColor.encoding, rgbOut);

        char* dstRow = reinterpret_cast<char*>(params.dst) + static_cast<std::size_t>(y) * params.dstRowBytes;
        float* dstPix = reinterpret_cast<float*>(dstRow + static_cast<std::size_t>(x) * pixelBytes);
        dstPix[0] = static_cast<float>(rgbOut[0]);
        dstPix[1] = static_cast<float>(rgbOut[1]);
        dstPix[2] = static_cast<float>(rgbOut[2]);
        if (nC == 4) {
            dstPix[3] = srcPix[3];
        }
    }

    template <typename Params>
    __global__ void focused_capture_film_density_kernel(
        Params params,
        float* outC,
        float* outM,
        float* outY) {
        if (!outC || !outM || !outY || !params.src || params.srcRowBytes == 0) {
            return;
        }

        const int nC = params.nComponents;
        if (!(nC == 3 || nC == 4)) {
            return;
        }

        if (params.width <= 0 || params.height <= 0) {
            return;
        }
        const std::size_t xIndex = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
        const std::size_t yIndex = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(blockDim.y) + static_cast<std::size_t>(threadIdx.y);
        if (xIndex >= static_cast<std::size_t>(params.width) || yIndex >= static_cast<std::size_t>(params.height)) {
            return;
        }
        const int x = static_cast<int>(xIndex);
        const int y = static_cast<int>(yIndex);

        const std::size_t idx = yIndex * static_cast<std::size_t>(params.width) + xIndex;
        const std::size_t pixelBytes = static_cast<std::size_t>(nC) * sizeof(float);
        const char* srcRow = reinterpret_cast<const char*>(params.src) +
                             static_cast<std::size_t>(y) * params.srcRowBytes;
        const float* srcPix = reinterpret_cast<const float*>(
            srcRow + static_cast<std::size_t>(x) * pixelBytes);
        const float rgbIn[3] = {srcPix[0], srcPix[1], srcPix[2]};

        float D_cmy[3] = {0.0f, 0.0f, 0.0f};
        if (!compute_capture_film_density_device(params, x, y, rgbIn, D_cmy)) {
            return;
        }
        outC[idx] = D_cmy[0];
        outM[idx] = D_cmy[1];
        outY[idx] = D_cmy[2];
    }

    struct ScanLinearDensityPlanes {
        const float* densityC = nullptr;
        const float* densityM = nullptr;
        const float* densityY = nullptr;
        const float* glarePercent = nullptr;
        float* outputR = nullptr;
        float* outputG = nullptr;
        float* outputB = nullptr;
    };

    __global__ void scan_linear_density_rgb_kernel(
        JuicerCuda::PipelineRunParams params,
        ScanLinearDensityPlanes planes) {
        const float* inC = planes.densityC;
        const float* inM = planes.densityM;
        const float* inY = planes.densityY;
        const float* glarePercent = planes.glarePercent;
        float* outR = planes.outputR;
        float* outG = planes.outputG;
        float* outB = planes.outputB;
        const JuicerCuda::ScanStagePayload& scan = params.scanStage;
        if (!inC || !inM || !inY || !outR || !outG || !outB || params.width <= 0 || params.height <= 0) {
            return;
        }
        const std::size_t width = static_cast<std::size_t>(params.width);
        const std::size_t height = static_cast<std::size_t>(params.height);
        const std::size_t yStep = static_cast<std::size_t>(blockDim.y) * static_cast<std::size_t>(gridDim.y);
        const std::size_t xStep = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
        for (std::size_t y = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(blockDim.y) + static_cast<std::size_t>(threadIdx.y); y < height; y += yStep) {
            for (std::size_t x = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x); x < width; x += xStep) {
                const std::size_t idx = y * width + x;
                const float D_cmy[3] = {inC[idx], inM[idx], inY[idx]};

                double D_norm[3];
                if (scan.scanTables.mediumIsNegative) {
                    D_norm[0] = (static_cast<double>(D_cmy[0]) +
                                 static_cast<double>(scan.scanTables.min_cmy[0])) *
                                static_cast<double>(scan.scanTables.inv_max_cmy[0]);
                    D_norm[1] = (static_cast<double>(D_cmy[1]) +
                                 static_cast<double>(scan.scanTables.min_cmy[1])) *
                                static_cast<double>(scan.scanTables.inv_max_cmy[1]);
                    D_norm[2] = (static_cast<double>(D_cmy[2]) +
                                 static_cast<double>(scan.scanTables.min_cmy[2])) *
                                static_cast<double>(scan.scanTables.inv_max_cmy[2]);
                } else {
                    D_norm[0] = (static_cast<double>(D_cmy[0]) -
                                 static_cast<double>(scan.scanTables.min_cmy[0])) *
                                static_cast<double>(scan.scanTables.inv_max_cmy[0]);
                    D_norm[1] = (static_cast<double>(D_cmy[1]) -
                                 static_cast<double>(scan.scanTables.min_cmy[1])) *
                                static_cast<double>(scan.scanTables.inv_max_cmy[1]);
                    D_norm[2] = (static_cast<double>(D_cmy[2]) -
                                 static_cast<double>(scan.scanTables.min_cmy[2])) *
                                static_cast<double>(scan.scanTables.inv_max_cmy[2]);
                }

                float log2XYZ[3] = {nanf(""), nanf(""), nanf("")};
                (void)scan_log2_xyz_device(scan, D_norm, log2XYZ);
                double xyz[3] = {
                    static_cast<double>(exp2f(log2XYZ[0])),
                    static_cast<double>(exp2f(log2XYZ[1])),
                    static_cast<double>(exp2f(log2XYZ[2]))};

                if (scan.correctionActive) {
                    const double correctedY = fmin(fmax(
                                                       static_cast<double>(scan.correctionSlope) * xyz[1] +
                                                           static_cast<double>(scan.correctionOffset),
                                                       0.0),
                                                   1.0);
                    const double scale = correctedY / (xyz[1] + 1e-10);
                    xyz[0] *= scale;
                    xyz[1] *= scale;
                    xyz[2] *= scale;
                }
                if (glarePercent) {
                    const double glare = static_cast<double>(glarePercent[idx]) * 0.01;
                    xyz[0] += glare * static_cast<double>(scan.scanColor.illuminantXYZ[0]);
                    xyz[1] += glare * static_cast<double>(scan.scanColor.illuminantXYZ[1]);
                    xyz[2] += glare * static_cast<double>(scan.scanColor.illuminantXYZ[2]);
                }

                double adapted[3];
                mat3_mul_vec_double_device(scan.scanColor.cat02, xyz, adapted);
                double rgbOut[3];
                mat3_mul_vec_double_device(scan.scanColor.xyzToRgb, adapted, rgbOut);

                if (!isfinite(rgbOut[0]) || !isfinite(rgbOut[1]) || !isfinite(rgbOut[2])) {
                    signal_scan_error_device(scan.scanErrorFlag);
                    outR[idx] = 0.0f;
                    outG[idx] = 0.0f;
                    outB[idx] = 0.0f;
                    continue;
                }

                outR[idx] = static_cast<float>(rgbOut[0]);
                outG[idx] = static_cast<float>(rgbOut[1]);
                outB[idx] = static_cast<float>(rgbOut[2]);
            }
        }
    }

    // CUDA dimensions and C/M/Y planes follow the focused ABI.
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    __global__ void apply_film_defects_kernel(
        JuicerCuda::GrainPayload grain,
        int width,
        int height,
        float* ioC,
        float* ioM,
        float* ioY) {
        if (!ioC || !ioM || !ioY || width <= 0 || height <= 0) {
            return;
        }
        const float dustAmount = grain.filmDustAmount;
        const float scratchAmount = grain.filmScratchAmount;
        if (!(dustAmount > 0.0f) && !(scratchAmount > 0.0f)) {
            return;
        }
        if (!(grain.pixelSizeUm > 0.0f)) {
            return;
        }

        const std::uint64_t sessionSeed = (grain.stbnSessionSeed != 0) ? grain.stbnSessionSeed : 1ULL;
        const std::uint64_t clipSeed = splitmix64_device(grain.clipToken ^ 0xD1B54A32D192ED03ULL);
        const std::uint64_t seedBase = sessionSeed ^ clipSeed;
        const std::uint64_t seedDust = splitmix64_device(seedBase ^ 0xF0D0C0B0A0908071ULL);
        const std::uint64_t seedScratch = splitmix64_device(seedBase ^ 0x8EBC6AF09C88C6E3ULL);

        const float time = static_cast<float>(grain.frameIndex) + grain.timeAlpha;
        const float rollPx = (grain.pitchPx > 0) ? static_cast<float>(grain.pitchPx) : 0.0f;

        constexpr float kDustCellUm = 400.0f;
        constexpr float kDustBaseProb = 0.02f;
        constexpr float kDustSizeUm = 25.0f;
        constexpr float kDustStrength = 0.45f;
        constexpr float kDustBrightMix = 0.20f;
        constexpr float kDustBrightScale = 0.5f;

        constexpr float kScratchCellUmX = 3500.0f;
        constexpr float kScratchCellUmY = 7000.0f;
        constexpr float kScratchBaseProb = 0.01f;
        constexpr float kScratchWidthUm = 15.0f;
        constexpr float kScratchStrength = 0.35f;
        constexpr float kScratchBrightMix = 0.10f;
        constexpr float kScratchBrightScale = 0.4f;
        constexpr float kScratchMaxAngle = 0.08726646f; // 5 deg

        const float pixelToMm = grain.pixelSizeUm * 0.001f;
        const float dustCellMm = kDustCellUm * 0.001f;
        const float scratchCellMmX = kScratchCellUmX * 0.001f;
        const float scratchCellMmY = kScratchCellUmY * 0.001f;
        const std::size_t widthCount = static_cast<std::size_t>(width);
        const std::size_t heightCount = static_cast<std::size_t>(height);
        const std::size_t yStep = static_cast<std::size_t>(blockDim.y) * static_cast<std::size_t>(gridDim.y);
        const std::size_t xStep = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
        for (std::size_t y = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(blockDim.y) + static_cast<std::size_t>(threadIdx.y); y < heightCount; y += yStep) {
            for (std::size_t x = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x); x < widthCount; x += xStep) {
                const std::int64_t absXSigned = static_cast<std::int64_t>(grain.originX) + static_cast<std::int64_t>(x);
                const std::int64_t absYSigned = static_cast<std::int64_t>(grain.originY) + static_cast<std::int64_t>(y);
                const float absX = static_cast<float>(absXSigned);
                const float absY = static_cast<float>(absYSigned);
                const float rollY = absY + rollPx * time;
                const float xMm = absX * pixelToMm;
                const float rollYMm = rollY * pixelToMm;

                DefectSampleLocation location{};
                location.xMm = xMm;
                location.yMm = rollYMm;
                location.pixelSizeUm = grain.pixelSizeUm;

                DustMaskParams dustParams{};
                dustParams.amount = dustAmount;
                dustParams.seed = seedDust;
                dustParams.cellMm = dustCellMm;
                dustParams.lambdaPerMm2 = kDustBaseProb;
                dustParams.sizeUm = kDustSizeUm;
                dustParams.strength = kDustStrength;
                dustParams.brightMix = kDustBrightMix;
                dustParams.brightScale = kDustBrightScale;
                const float dustMask = dust_mask_device(location, dustParams);

                ScratchMaskParams scratchParams{};
                scratchParams.amount = scratchAmount;
                scratchParams.seed = seedScratch;
                scratchParams.cellMmX = scratchCellMmX;
                scratchParams.cellMmY = scratchCellMmY;
                scratchParams.lambdaPerMm = kScratchBaseProb;
                scratchParams.widthUm = kScratchWidthUm;
                scratchParams.strength = kScratchStrength;
                scratchParams.maxAngleRad = kScratchMaxAngle;
                scratchParams.brightMix = kScratchBrightMix;
                scratchParams.brightScale = kScratchBrightScale;
                const float scratchMask = scratch_mask_device(location, scratchParams);
                float delta = dustMask + scratchMask;
                if (!device_isfinite(delta)) {
                    delta = 0.0f;
                }
                if (delta != 0.0f) {
                    const std::size_t idx = y * widthCount + x;
                    ioC[idx] = ioC[idx] + delta;
                    ioM[idx] = ioM[idx] + delta;
                    ioY[idx] = ioY[idx] + delta;
                }
            }
        }
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

    __global__ void gate_defect_mask_kernel(
        JuicerCuda::GrainPayload grain,
        float* outMask,
        int maskWidth,
        int maskHeight) {
        if (!outMask || maskWidth <= 0 || maskHeight <= 0) {
            return;
        }

        const float dustAmount = grain.gateDustAmount;
        const float scratchAmount = grain.gateScratchAmount;
        const bool gateActive = (dustAmount > 0.0f) || (scratchAmount > 0.0f);
        const bool pixelSizeValid = (grain.pixelSizeUm > 0.0f);

        const std::uint64_t sessionSeed = (grain.stbnSessionSeed != 0) ? grain.stbnSessionSeed : 1ULL;
        const std::uint64_t seedGateDust = splitmix64_device(sessionSeed ^ 0xA1B2C3D4E5F60718ULL);
        const std::uint64_t seedGateScratch = splitmix64_device(sessionSeed ^ 0xC6A4A7935BD1E995ULL);

        const float pixelToMm = grain.pixelSizeUm * 0.001f;
        const std::size_t width = static_cast<std::size_t>(maskWidth);
        const std::size_t height = static_cast<std::size_t>(maskHeight);
        const std::size_t yStep = static_cast<std::size_t>(blockDim.y) * static_cast<std::size_t>(gridDim.y);
        const std::size_t xStep = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
        for (std::size_t y = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(blockDim.y) + static_cast<std::size_t>(threadIdx.y); y < height; y += yStep) {
            for (std::size_t x = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x); x < width; x += xStep) {
                if (!gateActive || !pixelSizeValid) {
                    outMask[y * width + x] = 0.0f;
                    continue;
                }

                const float absX = static_cast<float>(grain.originX) + static_cast<float>(x) * 2.0f;
                const float absY = static_cast<float>(grain.originY) + static_cast<float>(y) * 2.0f;
                const float xMm = absX * pixelToMm;
                const float yMm = absY * pixelToMm;

                DefectSampleLocation location{};
                location.xMm = xMm;
                location.yMm = yMm;
                location.pixelSizeUm = grain.pixelSizeUm;
                GateMaskParams gateParams{};
                gateParams.dustAmount = dustAmount;
                gateParams.scratchAmount = scratchAmount;
                gateParams.dustSeed = seedGateDust;
                gateParams.scratchSeed = seedGateScratch;
                const float gateMask = gate_mask_device(location, gateParams);

                outMask[y * width + x] = gateMask;
            }
        }
    }

    template <typename Params>
    __global__ void focused_scan_output_encode_kernel(
        Params params,
        const float* rgbR,
        const float* rgbG,
        const float* rgbB,
        JuicerCuda::GrainPayload gateDefects,
        JuicerCuda::GateWeavePayload weave,
        const float* gateMask,
        int gateMaskWidth,
        int gateMaskHeight) {
        if (!params.src || !params.dst || !rgbR || !rgbG || !rgbB ||
            params.srcRowBytes == 0 || params.dstRowBytes == 0) {
            return;
        }
        const int nC = params.nComponents;
        if (!(nC == 3 || nC == 4)) {
            return;
        }
        const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= static_cast<unsigned int>(params.width) ||
            y >= static_cast<unsigned int>(params.height)) {
            return;
        }
        const std::size_t idx = static_cast<std::size_t>(y) *
                                    static_cast<std::size_t>(params.width) +
                                static_cast<std::size_t>(x);
        double rgbOut[3]{};
        if (weave.active != 0) {
            const float centerX = 0.5f * static_cast<float>(params.width - 1);
            const float centerY = 0.5f * static_cast<float>(params.height - 1);
            const float offsetX = static_cast<float>(x) - centerX;
            const float offsetY = static_cast<float>(y) - centerY;
            const float sourceX = weave.cosRot * offsetX -
                                  weave.sinRot * offsetY +
                                  centerX + weave.dxPx;
            const float sourceY = weave.sinRot * offsetX +
                                  weave.cosRot * offsetY +
                                  centerY + weave.dyPx;
            MitchellPlaneSample planeSample{};
            planeSample.width = params.width;
            planeSample.height = params.height;
            planeSample.x = sourceX;
            planeSample.y = sourceY;
            planeSample.plane = rgbR;
            rgbOut[0] = sample_plane_mitchell_device(planeSample);
            planeSample.plane = rgbG;
            rgbOut[1] = sample_plane_mitchell_device(planeSample);
            planeSample.plane = rgbB;
            rgbOut[2] = sample_plane_mitchell_device(planeSample);
        } else {
            rgbOut[0] = static_cast<double>(rgbR[idx]);
            rgbOut[1] = static_cast<double>(rgbG[idx]);
            rgbOut[2] = static_cast<double>(rgbB[idx]);
        }
        if (gateMask && gateMaskWidth > 0 && gateMaskHeight > 0) {
            const float absoluteX =
                static_cast<float>(gateDefects.originX) +
                static_cast<float>(x);
            const float absoluteY =
                static_cast<float>(gateDefects.originY) +
                static_cast<float>(y);
            const float maskX =
                (absoluteX - static_cast<float>(gateDefects.originX)) * 0.5f;
            const float maskY =
                (absoluteY - static_cast<float>(gateDefects.originY)) * 0.5f;
            GateMaskSample maskSample{};
            maskSample.mask = gateMask;
            maskSample.width = gateMaskWidth;
            maskSample.height = gateMaskHeight;
            maskSample.x = maskX;
            maskSample.y = maskY;
            float mask = sample_gate_mask_device(maskSample);
            if (device_isfinite(mask) && mask != 0.0f) {
                mask = fminf(fmaxf(mask, -0.5f), 0.95f);
                const double transmittance =
                    static_cast<double>(1.0f - mask);
                rgbOut[0] *= transmittance;
                rgbOut[1] *= transmittance;
                rgbOut[2] *= transmittance;
            }
        }
        apply_output_encoding_device(params.scanStage.scanColor.encoding, rgbOut);

        const std::size_t pixelBytes = static_cast<std::size_t>(nC) * sizeof(float);
        char* dstRow =
            reinterpret_cast<char*>(params.dst) + static_cast<std::size_t>(y) * params.dstRowBytes;
        float* dstPix =
            reinterpret_cast<float*>(dstRow + static_cast<std::size_t>(x) * pixelBytes);
        dstPix[0] = static_cast<float>(rgbOut[0]);
        dstPix[1] = static_cast<float>(rgbOut[1]);
        dstPix[2] = static_cast<float>(rgbOut[2]);
        if (nC == 4) {
            const char* srcRow =
                reinterpret_cast<const char*>(params.src) +
                static_cast<std::size_t>(y) * params.srcRowBytes;
            const float* srcPix =
                reinterpret_cast<const float*>(srcRow + static_cast<std::size_t>(x) * pixelBytes);
            dstPix[3] = srcPix[3];
        }
    }

} // namespace

extern "C" cudaError_t juicer_cuda_apply_film_defects(
    const JuicerCuda::GrainPayload* defects,
    int width,
    int height,
    float* densityC,
    float* densityM,
    float* densityY,
    void* cudaStreamOpaque) {
    if (!defects || !densityC || !densityM || !densityY ||
        width <= 0 || height <= 0) {
        return cudaErrorInvalidValue;
    }
    cudaStream_t stream = cudaStreamOpaque
                              ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque)
                              : nullptr;
    dim3 threads(32, 8);
    dim3 blocks(
        (static_cast<unsigned int>(width) + threads.x - 1) / threads.x,
        (static_cast<unsigned int>(height) + threads.y - 1) / threads.y);
    apply_film_defects_kernel<<<blocks, threads, 0, stream>>>(
        *defects,
        width,
        height,
        densityC,
        densityM,
        densityY);
    return cudaGetLastError();
}

extern "C" cudaError_t juicer_cuda_build_gate_defect_mask_focused(
    const JuicerCuda::GrainPayload* defects,
    float* gateMask,
    int gateMaskWidth,
    int gateMaskHeight,
    void* cudaStreamOpaque) {
    if (!defects || !gateMask || gateMaskWidth <= 0 || gateMaskHeight <= 0) {
        return cudaErrorInvalidValue;
    }
    cudaStream_t stream = cudaStreamOpaque
                              ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque)
                              : nullptr;
    dim3 threads(32, 8);
    dim3 blocks(
        (static_cast<unsigned int>(gateMaskWidth) + threads.x - 1) / threads.x,
        (static_cast<unsigned int>(gateMaskHeight) + threads.y - 1) / threads.y);
    gate_defect_mask_kernel<<<blocks, threads, 0, stream>>>(
        *defects,
        gateMask,
        gateMaskWidth,
        gateMaskHeight);
    return cudaGetLastError();
}

extern "C" cudaError_t juicer_cuda_negative_pipeline(
    const JuicerCuda::PipelineRunParams* hParams,
    void* cudaStreamOpaque) {
    if (!hParams) {
        return cudaErrorInvalidValue;
    }

    const JuicerCuda::PipelineRunParams params = *hParams;
    if (!params.src || !params.dst) {
        return cudaErrorInvalidValue;
    }
    if (params.width <= 0 || params.height <= 0) {
        return cudaSuccess;
    }
    if (!(params.nComponents == 3 || params.nComponents == 4)) {
        return cudaErrorInvalidValue;
    }
    if (params.srcRowBytes == 0 || params.dstRowBytes == 0) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

    dim3 threads(32, 8);
    dim3 blocks(
        (params.width + threads.x - 1) / threads.x,
        (params.height + threads.y - 1) / threads.y);
    pipeline_direct_kernel<<<blocks, threads, 0, stream>>>(params);
    return cudaGetLastError();
}

extern "C" cudaError_t juicer_cuda_negative_direct_pipeline(
    const JuicerCuda::DirectPipelineRunParams* hParams,
    void* cudaStreamOpaque) {
    if (!hParams) {
        return cudaErrorInvalidValue;
    }

    const JuicerCuda::DirectPipelineRunParams params = *hParams;
    if (!params.src || !params.dst) {
        return cudaErrorInvalidValue;
    }
    if (params.width <= 0 || params.height <= 0) {
        return cudaSuccess;
    }
    if (!(params.nComponents == 3 || params.nComponents == 4)) {
        return cudaErrorInvalidValue;
    }
    if (params.srcRowBytes == 0 || params.dstRowBytes == 0) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

    dim3 threads(32, 8);
    dim3 blocks(
        (params.width + threads.x - 1) / threads.x,
        (params.height + threads.y - 1) / threads.y);
    pipeline_direct_kernel<<<blocks, threads, 0, stream>>>(params);
    return cudaGetLastError();
}

struct FocusedScannerPostEffectOptions {
    float unsharpAmount = 0.0f;
    int glareOriginX = 0;
    int glareOriginY = 0;
    std::uint64_t glareSeed = 0;
    float glarePercent = 0.0f;
    float glareRoughness = 0.0f;
    const float* glareKernel = nullptr;
    int glareRadius = 0;
};

struct FocusedRgbPlanes {
    float* r = nullptr;
    float* g = nullptr;
    float* b = nullptr;
};

cudaError_t blur_plane_in_place(
    float* plane,
    float* tmp,
    int width,
    int height,
    const float* kernel,
    int radius,
    dim3 blocks,
    dim3 threads,
    cudaStream_t stream);

template <typename Params>
cudaError_t launch_focused_scan_linear_rgb(
    const Params* hParams,
    FocusedRgbPlanes rgb,
    float* dTmp,
    float* dScratchBlurred,
    const FocusedScannerPostEffectOptions& options,
    void* cudaStreamOpaque) {
    if (!hParams || !hParams->src || !hParams->dst || !rgb.r || !rgb.g || !rgb.b) {
        return cudaErrorInvalidValue;
    }
    Params params = *hParams;
    if (params.width <= 0 || params.height <= 0) {
        return cudaSuccess;
    }
    if (!(params.nComponents == 3 || params.nComponents == 4) ||
        params.srcRowBytes == 0 || params.dstRowBytes == 0) {
        return cudaErrorInvalidValue;
    }
    cudaStream_t stream =
        cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
    dim3 threads(32, 8);
    dim3 blocks(
        static_cast<unsigned int>((params.width + threads.x - 1) / threads.x),
        static_cast<unsigned int>((params.height + threads.y - 1) / threads.y));

    const bool doGlare =
        std::isfinite(static_cast<double>(options.glarePercent)) &&
        options.glarePercent > 0.0f &&
        std::isfinite(static_cast<double>(options.glareRoughness));
    if (doGlare) {
        if (!dTmp) {
            return cudaErrorInvalidValue;
        }
        if (options.glareRadius > 0 && options.glareKernel && !dScratchBlurred) {
            return cudaErrorInvalidValue;
        }
        const std::uint64_t mediumId = params.scanStage.scanTables.mediumIsNegative ? 0ULL : 1ULL;
        GlareGenerationInput glareInput{};
        glareInput.output = dTmp;
        glareInput.width = params.width;
        glareInput.height = params.height;
        glareInput.seed = options.glareSeed;
        glareInput.mediumId = mediumId;
        glareInput.originX = options.glareOriginX;
        glareInput.originY = options.glareOriginY;
        glareInput.percent = options.glarePercent;
        glareInput.roughness = options.glareRoughness;
        optics_glare_generate_kernel<<<blocks, threads, 0, stream>>>(glareInput);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            return error;
        }
        if (options.glareRadius > 0 && options.glareKernel) {
            error = blur_plane_in_place(
                dTmp,
                dScratchBlurred,
                params.width,
                params.height,
                options.glareKernel,
                options.glareRadius,
                blocks,
                threads,
                stream);
            if (error != cudaSuccess) {
                return error;
            }
        }
        params.scanStage.glarePercent = dTmp;
    }

    params.scanStage.linearRgbR = rgb.r;
    params.scanStage.linearRgbG = rgb.g;
    params.scanStage.linearRgbB = rgb.b;
    pipeline_direct_kernel<<<blocks, threads, 0, stream>>>(params);
    return cudaGetLastError();
}

template <typename Params>
cudaError_t launch_focused_capture_density(
    const Params* hParams,
    float* dDensityC,
    float* dDensityM,
    float* dDensityY,
    void* cudaStreamOpaque) {
    if (!hParams || !dDensityC || !dDensityM || !dDensityY) {
        return cudaErrorInvalidValue;
    }
    Params focusedParams = *hParams;
    if (!focusedParams.src) {
        return cudaErrorInvalidValue;
    }
    if (focusedParams.width <= 0 || focusedParams.height <= 0) {
        return cudaSuccess;
    }
    if (!(focusedParams.nComponents == 3 || focusedParams.nComponents == 4) ||
        focusedParams.srcRowBytes == 0) {
        return cudaErrorInvalidValue;
    }

    JuicerCuda::PipelineRunParams params = [&]() {
        if constexpr (requires { focusedParams.printExpose; focusedParams.printDevelop; }) {
            return focused_params_from_print(focusedParams);
        } else {
            return focused_params_from_direct(focusedParams);
        }
    }();
    cudaStream_t stream =
        cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
    dim3 threads(32, 8);
    dim3 blocks(
        (params.width + threads.x - 1) / threads.x,
        (params.height + threads.y - 1) / threads.y);

    focused_capture_film_density_kernel<<<blocks, threads, 0, stream>>>(
        params,
        dDensityC,
        dDensityM,
        dDensityY);
    return cudaGetLastError();
}

cudaError_t launch_focused_print_continuation_from_capture_density(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    float* dDensityC,
    float* dDensityM,
    float* dDensityY,
    void* cudaStreamOpaque) {
    if (!hParams || !dDensityC || !dDensityM || !dDensityY) {
        return cudaErrorInvalidValue;
    }
    if (hParams->width <= 0 || hParams->height <= 0) {
        return cudaSuccess;
    }

    JuicerCuda::PipelineRunParams params = focused_params_from_print(*hParams);
    cudaStream_t stream =
        cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
    dim3 threads(32, 8);
    dim3 blocks(
        (params.width + threads.x - 1) / threads.x,
        (params.height + threads.y - 1) / threads.y);
    develop_print_density_kernel<<<blocks, threads, 0, stream>>>(
        params,
        dDensityC,
        dDensityM,
        dDensityY);
    return cudaGetLastError();
}

bool valid_enlarger_print_linear_planes(
    const JuicerCuda::EnlargerPrintLinearExposurePlanes& planes,
    int width,
    int height) noexcept {
    if (!planes.redSensitiveCForming ||
        !planes.greenSensitiveMForming ||
        !planes.blueSensitiveYForming ||
        planes.redSensitiveCForming == planes.greenSensitiveMForming ||
        planes.redSensitiveCForming == planes.blueSensitiveYForming ||
        planes.greenSensitiveMForming == planes.blueSensitiveYForming ||
        width <= 0 || height <= 0 ||
        planes.rowStrideFloats < static_cast<std::size_t>(width)) {
        return false;
    }
    const std::size_t lastRow = static_cast<std::size_t>(height - 1);
    const std::size_t widthCount = static_cast<std::size_t>(width);
    return lastRow == 0 ||
           planes.rowStrideFloats <=
               (std::numeric_limits<std::size_t>::max() - widthCount) /
                   lastRow;
}

cudaError_t launch_focused_enlarger_print_linear_exposure(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    const float* dDensityC,
    const float* dDensityM,
    const float* dDensityY,
    JuicerCuda::EnlargerPrintLinearExposurePlanes planes,
    void* cudaStreamOpaque) {
    if (!hParams || !dDensityC || !dDensityM || !dDensityY) {
        return cudaErrorInvalidValue;
    }
    if (hParams->width <= 0 || hParams->height <= 0) {
        return cudaSuccess;
    }
    if (!hParams->printExpose.active ||
        !valid_enlarger_print_linear_planes(
            planes,
            hParams->width,
            hParams->height)) {
        return cudaErrorInvalidValue;
    }

    JuicerCuda::PipelineRunParams params = focused_params_from_print(*hParams);
    cudaStream_t stream =
        cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
    dim3 threads(32, 8);
    dim3 blocks(
        (params.width + threads.x - 1) / threads.x,
        (params.height + threads.y - 1) / threads.y);
    build_enlarger_print_linear_exposure_kernel<<<blocks, threads, 0, stream>>>(
        params,
        dDensityC,
        dDensityM,
        dDensityY,
        planes);
    return cudaGetLastError();
}

cudaError_t launch_focused_print_develop_from_enlarger_linear(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    JuicerCuda::EnlargerPrintLinearExposurePlanes planes,
    float* dDensityC,
    float* dDensityM,
    float* dDensityY,
    void* cudaStreamOpaque) {
    if (!hParams || !dDensityC || !dDensityM || !dDensityY) {
        return cudaErrorInvalidValue;
    }
    if (hParams->width <= 0 || hParams->height <= 0) {
        return cudaSuccess;
    }
    if (!hParams->printExpose.active ||
        !valid_enlarger_print_linear_planes(
            planes,
            hParams->width,
            hParams->height)) {
        return cudaErrorInvalidValue;
    }

    JuicerCuda::PipelineRunParams params = focused_params_from_print(*hParams);
    cudaStream_t stream =
        cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
    dim3 threads(32, 8);
    dim3 blocks(
        (params.width + threads.x - 1) / threads.x,
        (params.height + threads.y - 1) / threads.y);
    develop_print_density_from_enlarger_linear_kernel<<<blocks, threads, 0, stream>>>(
        params,
        planes,
        dDensityC,
        dDensityM,
        dDensityY);
    return cudaGetLastError();
}

template <typename Params>
cudaError_t launch_focused_scan_linear_density_rgb(
    const Params* hParams,
    const float* dDensityC,
    const float* dDensityM,
    const float* dDensityY,
    FocusedRgbPlanes rgb,
    float* dTmp,
    float* dScratchBlurred,
    const FocusedScannerPostEffectOptions& options,
    void* cudaStreamOpaque) {
    if (!hParams || !dDensityC || !dDensityM || !dDensityY || !rgb.r || !rgb.g || !rgb.b) {
        return cudaErrorInvalidValue;
    }
    Params focusedParams = *hParams;
    if (!focusedParams.src || !focusedParams.dst) {
        return cudaErrorInvalidValue;
    }
    if (focusedParams.width <= 0 || focusedParams.height <= 0) {
        return cudaSuccess;
    }
    if (!(focusedParams.nComponents == 3 || focusedParams.nComponents == 4) ||
        focusedParams.srcRowBytes == 0 || focusedParams.dstRowBytes == 0) {
        return cudaErrorInvalidValue;
    }

    JuicerCuda::PipelineRunParams params = [&]() {
        if constexpr (requires { focusedParams.printExpose; focusedParams.printDevelop; }) {
            return focused_params_from_print(focusedParams);
        } else {
            return focused_params_from_direct(focusedParams);
        }
    }();
    cudaStream_t stream =
        cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
    dim3 threads(32, 8);
    dim3 blocks(
        (params.width + threads.x - 1) / threads.x,
        (params.height + threads.y - 1) / threads.y);

    const float* glarePercent = nullptr;
    const bool doGlare =
        std::isfinite(static_cast<double>(options.glarePercent)) &&
        options.glarePercent > 0.0f &&
        std::isfinite(static_cast<double>(options.glareRoughness));
    if (doGlare) {
        if (!dTmp) {
            return cudaErrorInvalidValue;
        }
        if (options.glareRadius > 0 && options.glareKernel && !dScratchBlurred) {
            return cudaErrorInvalidValue;
        }
        const std::uint64_t mediumId = params.scanStage.scanTables.mediumIsNegative ? 0ULL : 1ULL;
        GlareGenerationInput glareInput{};
        glareInput.output = dTmp;
        glareInput.width = params.width;
        glareInput.height = params.height;
        glareInput.seed = options.glareSeed;
        glareInput.mediumId = mediumId;
        glareInput.originX = options.glareOriginX;
        glareInput.originY = options.glareOriginY;
        glareInput.percent = options.glarePercent;
        glareInput.roughness = options.glareRoughness;
        optics_glare_generate_kernel<<<blocks, threads, 0, stream>>>(glareInput);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            return error;
        }
        if (options.glareRadius > 0 && options.glareKernel) {
            error = blur_plane_in_place(
                dTmp,
                dScratchBlurred,
                params.width,
                params.height,
                options.glareKernel,
                options.glareRadius,
                blocks,
                threads,
                stream);
            if (error != cudaSuccess) {
                return error;
            }
        }
        glarePercent = dTmp;
    }

    ScanLinearDensityPlanes scanPlanes{};
    scanPlanes.densityC = dDensityC;
    scanPlanes.densityM = dDensityM;
    scanPlanes.densityY = dDensityY;
    scanPlanes.glarePercent = glarePercent;
    scanPlanes.outputR = rgb.r;
    scanPlanes.outputG = rgb.g;
    scanPlanes.outputB = rgb.b;
    scan_linear_density_rgb_kernel<<<blocks, threads, 0, stream>>>(params, scanPlanes);
    return cudaGetLastError();
}

cudaError_t blur_plane_in_place(
    float* plane,
    float* tmp,
    int width,
    int height,
    const float* kernel,
    int radius,
    dim3 blocks,
    dim3 threads,
    cudaStream_t stream) {
    if (!plane || !tmp || !kernel || radius <= 0) {
        return cudaSuccess;
    }
    const int kernelLength = 2 * radius + 1;
    const size_t horizontalShared =
        (static_cast<size_t>(kernelLength) +
         static_cast<size_t>(threads.y) * static_cast<size_t>(threads.x + 2 * radius)) *
        sizeof(float);
    const size_t verticalShared =
        (static_cast<size_t>(kernelLength) +
         static_cast<size_t>(threads.x) * static_cast<size_t>(threads.y + 2 * radius)) *
        sizeof(float);
    optics_blur_horizontal_kernel<double><<<
        blocks,
        threads,
        horizontalShared,
        stream>>>(
        plane,
        tmp,
        width,
        height,
        kernel,
        radius);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }
    optics_blur_vertical_kernel<<<blocks, threads, verticalShared, stream>>>(
        tmp,
        plane,
        width,
        height,
        kernel,
        radius);
    return cudaGetLastError();
}

template <typename Params>
cudaError_t launch_focused_scanner_post_output(
    const Params* hParams,
    FocusedRgbPlanes rgb,
    float* dTmp,
    const float* dLensBlurKernel,
    int lensBlurRadius,
    const float* dUnsharpKernel,
    int unsharpRadius,
    float unsharpAmount,
    const JuicerCuda::GrainPayload* gateDefects,
    const JuicerCuda::GateWeavePayload* weave,
    const float* gateMask,
    int gateMaskWidth,
    int gateMaskHeight,
    void* cudaStreamOpaque) {
    if (!hParams || !hParams->src || !hParams->dst || !rgb.r || !rgb.g || !rgb.b) {
        return cudaErrorInvalidValue;
    }
    Params params = *hParams;
    if (params.width <= 0 || params.height <= 0) {
        return cudaSuccess;
    }
    if (!(params.nComponents == 3 || params.nComponents == 4) ||
        params.srcRowBytes == 0 || params.dstRowBytes == 0) {
        return cudaErrorInvalidValue;
    }
    if ((gateMask != nullptr) !=
            (gateMaskWidth > 0 && gateMaskHeight > 0) ||
        (gateMask && !gateDefects)) {
        return cudaErrorInvalidValue;
    }
    cudaStream_t stream =
        cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
    dim3 threads(32, 8);
    dim3 blocks(
        static_cast<unsigned int>((params.width + threads.x - 1) / threads.x),
        static_cast<unsigned int>((params.height + threads.y - 1) / threads.y));

    auto blur_plane = [&](float* plane, const float* kernel, int radius) {
        return blur_plane_in_place(
            plane,
            dTmp,
            params.width,
            params.height,
            kernel,
            radius,
            blocks,
            threads,
            stream);
    };
    if (lensBlurRadius > 0 && dLensBlurKernel && !dTmp) {
        return cudaErrorInvalidValue;
    }
    if (unsharpRadius > 0 && dUnsharpKernel &&
        std::isfinite(static_cast<double>(unsharpAmount)) &&
        unsharpAmount > 0.0f &&
        !dTmp) {
        return cudaErrorInvalidValue;
    }

    cudaError_t error = cudaSuccess;
    const bool doLensBlur = lensBlurRadius > 0 && dLensBlurKernel;
    if (doLensBlur) {
        error = blur_plane(rgb.r, dLensBlurKernel, lensBlurRadius);
        if (error != cudaSuccess) {
            return error;
        }
        error = blur_plane(rgb.g, dLensBlurKernel, lensBlurRadius);
        if (error != cudaSuccess) {
            return error;
        }
        error = blur_plane(rgb.b, dLensBlurKernel, lensBlurRadius);
        if (error != cudaSuccess) {
            return error;
        }
    }

    const bool doUnsharp =
        unsharpRadius > 0 && dUnsharpKernel &&
        std::isfinite(static_cast<double>(unsharpAmount)) &&
        unsharpAmount > 0.0f;
    if (doUnsharp) {
        auto unsharp_plane_in_place = [&](float* plane) {
            const int kernelLength = 2 * unsharpRadius + 1;
            const size_t horizontalShared =
                (static_cast<size_t>(kernelLength) +
                 static_cast<size_t>(threads.y) *
                     static_cast<size_t>(threads.x + 2 * unsharpRadius)) *
                sizeof(float);
            const size_t verticalShared =
                (static_cast<size_t>(kernelLength) +
                 static_cast<size_t>(threads.x) *
                     static_cast<size_t>(threads.y + 2 * unsharpRadius)) *
                sizeof(float);
            optics_blur_horizontal_kernel<double><<<
                blocks,
                threads,
                horizontalShared,
                stream>>>(
                plane,
                dTmp,
                params.width,
                params.height,
                dUnsharpKernel,
                unsharpRadius);
            cudaError_t localError = cudaGetLastError();
            if (localError != cudaSuccess) {
                return localError;
            }
            VerticalUnsharpInput unsharpInput{};
            unsharpInput.output = plane;
            unsharpInput.horizontalBlur = dTmp;
            unsharpInput.width = params.width;
            unsharpInput.height = params.height;
            unsharpInput.kernel = dUnsharpKernel;
            unsharpInput.radius = unsharpRadius;
            unsharpInput.amount = unsharpAmount;
            optics_unsharp_vertical_combine_kernel<<<blocks, threads, verticalShared, stream>>>(unsharpInput);
            return cudaGetLastError();
        };
        error = unsharp_plane_in_place(rgb.r);
        if (error != cudaSuccess) {
            return error;
        }
        error = unsharp_plane_in_place(rgb.g);
        if (error != cudaSuccess) {
            return error;
        }
        error = unsharp_plane_in_place(rgb.b);
        if (error != cudaSuccess) {
            return error;
        }
    }
    focused_scan_output_encode_kernel<<<blocks, threads, 0, stream>>>(
        params,
        rgb.r,
        rgb.g,
        rgb.b,
        gateDefects ? *gateDefects : JuicerCuda::GrainPayload{},
        weave ? *weave : JuicerCuda::GateWeavePayload{},
        gateMask,
        gateMaskWidth,
        gateMaskHeight);
    return cudaGetLastError();
}

extern "C" cudaError_t juicer_cuda_direct_focused_capture_density(
    const JuicerCuda::DirectPipelineRunParams* hParams,
    float* dDensityC,
    float* dDensityM,
    float* dDensityY,
    void* cudaStreamOpaque) {
    return launch_focused_capture_density(
        hParams,
        dDensityC,
        dDensityM,
        dDensityY,
        cudaStreamOpaque);
}

extern "C" cudaError_t juicer_cuda_print_focused_capture_density(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    float* dDensityC,
    float* dDensityM,
    float* dDensityY,
    void* cudaStreamOpaque) {
    return launch_focused_capture_density(
        hParams,
        dDensityC,
        dDensityM,
        dDensityY,
        cudaStreamOpaque);
}

extern "C" cudaError_t juicer_cuda_print_focused_continue_from_capture_density(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    float* dDensityC,
    float* dDensityM,
    float* dDensityY,
    void* cudaStreamOpaque) {
    return launch_focused_print_continuation_from_capture_density(
        hParams,
        dDensityC,
        dDensityM,
        dDensityY,
        cudaStreamOpaque);
}

extern "C" cudaError_t juicer_cuda_print_focused_enlarger_linear_exposure(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    const float* dDensityC,
    const float* dDensityM,
    const float* dDensityY,
    JuicerCuda::EnlargerPrintLinearExposurePlanes planes,
    void* cudaStreamOpaque) {
    return launch_focused_enlarger_print_linear_exposure(
        hParams,
        dDensityC,
        dDensityM,
        dDensityY,
        planes,
        cudaStreamOpaque);
}

extern "C" cudaError_t
juicer_cuda_print_focused_develop_from_enlarger_linear(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    JuicerCuda::EnlargerPrintLinearExposurePlanes planes,
    float* dDensityC,
    float* dDensityM,
    float* dDensityY,
    void* cudaStreamOpaque) {
    return launch_focused_print_develop_from_enlarger_linear(
        hParams,
        planes,
        dDensityC,
        dDensityM,
        dDensityY,
        cudaStreamOpaque);
}

extern "C" cudaError_t juicer_cuda_direct_focused_scan_linear_density_rgb(
    const JuicerCuda::DirectPipelineRunParams* hParams,
    const float* dDensityC,
    const float* dDensityM,
    const float* dDensityY,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    void* cudaStreamOpaque) {
    return launch_focused_scan_linear_density_rgb(
        hParams,
        dDensityC,
        dDensityM,
        dDensityY,
        FocusedRgbPlanes{dRgbR, dRgbG, dRgbB},
        nullptr,
        nullptr,
        FocusedScannerPostEffectOptions{},
        cudaStreamOpaque);
}

extern "C" cudaError_t juicer_cuda_print_focused_scan_linear_density_rgb(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    const float* dDensityC,
    const float* dDensityM,
    const float* dDensityY,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    float* dScratchBlurred,
    int glareOriginX,
    int glareOriginY,
    std::uint64_t glareSeed,
    float glarePercent,
    float glareRoughness,
    const float* dGlareKernel,
    int glareRadius,
    void* cudaStreamOpaque) {
    return launch_focused_scan_linear_density_rgb(
        hParams,
        dDensityC,
        dDensityM,
        dDensityY,
        FocusedRgbPlanes{dRgbR, dRgbG, dRgbB},
        dTmp,
        dScratchBlurred,
        FocusedScannerPostEffectOptions{
            0.0f,
            glareOriginX,
            glareOriginY,
            glareSeed,
            glarePercent,
            glareRoughness,
            dGlareKernel,
            glareRadius},
        cudaStreamOpaque);
}

extern "C" cudaError_t juicer_cuda_direct_focused_scan_linear_rgb(
    const JuicerCuda::DirectPipelineRunParams* hParams,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    void* cudaStreamOpaque) {
    return launch_focused_scan_linear_rgb(
        hParams,
        FocusedRgbPlanes{dRgbR, dRgbG, dRgbB},
        nullptr,
        nullptr,
        FocusedScannerPostEffectOptions{},
        cudaStreamOpaque);
}

extern "C" cudaError_t juicer_cuda_print_focused_scan_linear_rgb(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    float* dScratchBlurred,
    int glareOriginX,
    int glareOriginY,
    std::uint64_t glareSeed,
    float glarePercent,
    float glareRoughness,
    const float* dGlareKernel,
    int glareRadius,
    void* cudaStreamOpaque) {
    return launch_focused_scan_linear_rgb(
        hParams,
        FocusedRgbPlanes{dRgbR, dRgbG, dRgbB},
        dTmp,
        dScratchBlurred,
        FocusedScannerPostEffectOptions{
            0.0f,
            glareOriginX,
            glareOriginY,
            glareSeed,
            glarePercent,
            glareRoughness,
            dGlareKernel,
            glareRadius},
        cudaStreamOpaque);
}

extern "C" cudaError_t juicer_cuda_direct_focused_scanner_post_output(
    const JuicerCuda::DirectPipelineRunParams* hParams,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    const float* dLensBlurKernel,
    int lensBlurRadius,
    const float* dUnsharpKernel,
    int unsharpRadius,
    float unsharpAmount,
    const JuicerCuda::GrainPayload* gateDefects,
    const JuicerCuda::GateWeavePayload* weave,
    const float* gateMask,
    int gateMaskWidth,
    int gateMaskHeight,
    void* cudaStreamOpaque) {
    return launch_focused_scanner_post_output(
        hParams,
        FocusedRgbPlanes{dRgbR, dRgbG, dRgbB},
        dTmp,
        dLensBlurKernel,
        lensBlurRadius,
        dUnsharpKernel,
        unsharpRadius,
        unsharpAmount,
        gateDefects,
        weave,
        gateMask,
        gateMaskWidth,
        gateMaskHeight,
        cudaStreamOpaque);
}

extern "C" cudaError_t juicer_cuda_print_focused_scanner_post_output(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    const float* dLensBlurKernel,
    int lensBlurRadius,
    const float* dUnsharpKernel,
    int unsharpRadius,
    float unsharpAmount,
    const JuicerCuda::GrainPayload* gateDefects,
    const JuicerCuda::GateWeavePayload* weave,
    const float* gateMask,
    int gateMaskWidth,
    int gateMaskHeight,
    void* cudaStreamOpaque) {
    return launch_focused_scanner_post_output(
        hParams,
        FocusedRgbPlanes{dRgbR, dRgbG, dRgbB},
        dTmp,
        dLensBlurKernel,
        lensBlurRadius,
        dUnsharpKernel,
        unsharpRadius,
        unsharpAmount,
        gateDefects,
        weave,
        gateMask,
        gateMaskWidth,
        gateMaskHeight,
        cudaStreamOpaque);
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters) The focused CUDA ABI and local schedule use fixed named bindings.
extern "C" cudaError_t juicer_cuda_apply_visual_grain(
    const JuicerCuda::GrainPayload* grain,
    const JuicerCuda::GrainKernelPayload* kernels,
    int width,
    int height,
    float* densityC,
    float* densityM,
    float* densityY,
    float* filterTemp,
    float* scaleWork,
    float* deltaAccum,
    float* layerWork,
    float* sharedDelta,
    void* cudaStreamOpaque) {
    if (!grain || !kernels) {
        return cudaErrorInvalidValue;
    }
    if (grain->active == 0) {
        return cudaSuccess;
    }
    if (width <= 0 || height <= 0 ||
        width > std::numeric_limits<int>::max() / height ||
        !densityC || !densityM || !densityY || !filterTemp ||
        !scaleWork || !deltaAccum ||
        (grain->debugView != 6 && !grain->frameUniforms)) {
        return cudaErrorInvalidValue;
    }

    const bool useSublayers = grain->sublayersActive != 0;
    if (useSublayers && !layerWork) {
        return cudaErrorInvalidValue;
    }
    for (int channel = 0; channel < 3; ++channel) {
        if (!std::isfinite(grain->densityMin[channel]) ||
            !std::isfinite(grain->densityMax[channel]) ||
            !(grain->densityMax[channel] > 0.0f) ||
            !std::isfinite(grain->nParticles[channel]) ||
            !(grain->nParticles[channel] > 0.0f) ||
            !std::isfinite(grain->odParticle[channel]) ||
            !(grain->odParticle[channel] > 0.0f)) {
            return cudaErrorInvalidValue;
        }
        if (!useSublayers) {
            continue;
        }
        const JuicerCuda::DeviceCurveView& baseCurve =
            grain->densityCurveCmy[channel];
        if (!baseCurve.x || !baseCurve.y || baseCurve.n <= 0 ||
            baseCurve.domainBegin < 0 ||
            baseCurve.domainEnd < baseCurve.domainBegin ||
            baseCurve.domainEnd >= baseCurve.n) {
            return cudaErrorInvalidValue;
        }
        for (int layer = 0; layer < 3; ++layer) {
            if (!grain->densityCurvesLayers[layer][channel] ||
                !std::isfinite(grain->densityMinLayers[layer][channel]) ||
                !std::isfinite(grain->densityMaxLayers[layer][channel]) ||
                !(grain->densityMaxLayers[layer][channel] > 0.0f) ||
                !std::isfinite(grain->nParticlesLayers[layer][channel]) ||
                !(grain->nParticlesLayers[layer][channel] > 0.0f) ||
                !std::isfinite(grain->odParticleLayers[layer][channel]) ||
                !(grain->odParticleLayers[layer][channel] > 0.0f)) {
                return cudaErrorInvalidValue;
            }
            if (kernels->dyeRadius[layer][channel] > 0 &&
                !kernels->dyeKernel[layer][channel]) {
                return cudaErrorInvalidValue;
            }
        }
    }

    constexpr std::uint64_t kSeedSaltFine = 0xA24BAED4963EE407ULL;
    constexpr std::uint64_t kSeedSaltMid = 0x9F6C1E6B2C4D7A13ULL;
    constexpr std::uint64_t kSeedSaltCoarse = 0x3C79AC492BA7B653ULL;
    constexpr std::uint64_t kSeedSaltShared = 0x7E8A1B9D3F2C65A1ULL;

    const float wFine = grain->sizeMixWeightFine;
    const float wMid = grain->sizeMixWeightMid;
    const float wCoarse = grain->sizeMixWeight;
    const float sizeMixScale = grain->sizeMixScale;
    const float sizeMixGain = grain->sizeMixGain;
    const float amplitude = grain->amplitude;
    if (!std::isfinite(wFine) || !std::isfinite(wMid) ||
        !std::isfinite(wCoarse) || wFine < 0.0f || wMid < 0.0f ||
        wCoarse < 0.0f ||
        std::fabs((wFine + wMid + wCoarse) - 1.0f) > 1e-5f ||
        !std::isfinite(sizeMixScale) || !(sizeMixScale >= 1.0f) ||
        !std::isfinite(sizeMixGain) || !(sizeMixGain > 0.0f) ||
        !std::isfinite(amplitude) || amplitude < 0.0f) {
        return cudaErrorInvalidValue;
    }
    if (wMid > 0.0f &&
        (!kernels->blurKernelMid || kernels->blurRadiusMid <= 0)) {
        return cudaErrorInvalidValue;
    }
    if (wCoarse > 0.0f &&
        (!kernels->blurKernelCoarse || kernels->blurRadiusCoarse <= 0)) {
        return cudaErrorInvalidValue;
    }

    const bool sharedChroma =
        (grain->debugView == 0 || grain->debugView == 1) &&
        grain->chromaMix < 0.999f && grain->chromaSharedWeight > 0.0f;
    if (sharedChroma && !sharedDelta) {
        return cudaErrorInvalidValue;
    }
    if (!std::isfinite(grain->chromaMix) || grain->chromaMix < 0.0f ||
        grain->chromaMix > 1.0f ||
        !std::isfinite(grain->chromaSharedWeight) ||
        grain->chromaSharedWeight < 0.0f ||
        !std::isfinite(grain->chromaIndWeight) ||
        grain->chromaIndWeight < 0.0f || grain->debugView < 0 ||
        grain->debugView > 6) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = cudaStreamOpaque
                              ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque)
                              : nullptr;
    const int total = width * height;
    const int threads1D = 256;
    const int blocks1D = (total + threads1D - 1) / threads1D;
    dim3 threads2D(32, 8);
    dim3 blocks2D(
        (width + threads2D.x - 1) / threads2D.x,
        (height + threads2D.y - 1) / threads2D.y);

    auto launch_grain_blur_horizontal = [&](float* plane,
                                            const float* kernel,
                                            int radius) -> cudaError_t {
        if (!plane || !kernel) {
            return cudaErrorInvalidValue;
        }
        const int kernelLength = 2 * radius + 1;
        const size_t horizontalSharedBytes =
            (static_cast<size_t>(kernelLength) +
             static_cast<size_t>(threads2D.y) *
                 static_cast<size_t>(threads2D.x + 2 * radius)) *
            sizeof(float);
        optics_blur_horizontal_kernel<float><<<
            blocks2D,
            threads2D,
            horizontalSharedBytes,
            stream>>>(plane, filterTemp, width, height, kernel, radius);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            return error;
        }
        return cudaSuccess;
    };

    auto generate_scale = [&](const JuicerCuda::GrainPayload& base,
                              const float* density,
                              int channel,
                              std::uint64_t salt,
                              float particleFactor,
                              const float* correlationKernel,
                              int correlationRadius,
                              bool applyCorrelation,
                              float weight,
                              bool initializeDeltaAccum) -> cudaError_t {
        JuicerCuda::GrainPayload pass = base;
        pass.seedBase ^= salt;
        pass.seedBaseNext ^= salt;
        pass.sizeMixWeightFine = 1.0f;
        pass.sizeMixWeight = 0.0f;
        pass.sizeMixWeightMid = 0.0f;
        pass.sizeMixScale = 1.0f;
        if (!std::isfinite(particleFactor) || !(particleFactor > 0.0f)) {
            return cudaErrorInvalidValue;
        }
        for (int scaleChannel = 0; scaleChannel < 3; ++scaleChannel) {
            pass.nParticles[scaleChannel] =
                base.nParticles[scaleChannel] / particleFactor;
            pass.odParticle[scaleChannel] =
                base.odParticle[scaleChannel] * particleFactor;
            for (int layer = 0; layer < 3; ++layer) {
                pass.nParticlesLayers[layer][scaleChannel] =
                    base.nParticlesLayers[layer][scaleChannel] /
                    particleFactor;
                pass.odParticleLayers[layer][scaleChannel] =
                    base.odParticleLayers[layer][scaleChannel] *
                    particleFactor;
            }
        }

        cudaError_t error = cudaSuccess;
        if (useSublayers) {
            const int dyeRadius0 = kernels->dyeRadius[0][channel];
            const int dyeRadius1 = kernels->dyeRadius[1][channel];
            const int dyeRadius2 = kernels->dyeRadius[2][channel];
            const float* dyeKernel0 = kernels->dyeKernel[0][channel];
            const float* dyeKernel1 = kernels->dyeKernel[1][channel];
            const float* dyeKernel2 = kernels->dyeKernel[2][channel];
            const bool dyeRadiiEqual =
                dyeRadius0 == dyeRadius1 && dyeRadius0 == dyeRadius2;
            const bool dyeKernelsEqual =
                dyeKernel0 == dyeKernel1 && dyeKernel0 == dyeKernel2;
            const bool dyeKernelPrepared =
                dyeRadius0 == 0 || dyeKernel0 != nullptr;
            const bool zeroDyeKernels =
                !dyeKernel0 && !dyeKernel1 && !dyeKernel2;
            const bool batchDyeTriplet =
                dyeRadiiEqual && dyeKernelsEqual && dyeKernelPrepared &&
                (dyeRadius0 > 0 || zeroDyeKernels);
            if (batchDyeTriplet) {
                grain_layer_triplet_kernel<<<blocks2D, threads2D, 0, stream>>>(
                    pass,
                    width,
                    height,
                    density,
                    layerWork,
                    channel);
                error = cudaGetLastError();
                if (error != cudaSuccess) {
                    return error;
                }
                if (dyeRadius0 > 0) {
                    error = launch_grain_blur_horizontal(
                        layerWork,
                        dyeKernel0,
                        dyeRadius0);
                    if (error != cudaSuccess) {
                        return error;
                    }
                    const int kernelLength = 2 * dyeRadius0 + 1;
                    const size_t verticalSharedBytes =
                        (static_cast<size_t>(kernelLength) +
                         static_cast<size_t>(threads2D.x) *
                             static_cast<size_t>(threads2D.y + 2 * dyeRadius0)) *
                        sizeof(float);
                    GrainBlurVerticalAccumulateInput finishInput{};
                    finishInput.horizontalBlur = filterTemp;
                    finishInput.output = scaleWork;
                    finishInput.width = width;
                    finishInput.height = height;
                    finishInput.kernel = dyeKernel0;
                    finishInput.radius = dyeRadius0;
                    finishInput.initialize = 1;
                    grain_blur_vertical_accumulate_kernel<<<
                        blocks2D,
                        threads2D,
                        verticalSharedBytes,
                        stream>>>(finishInput);
                    error = cudaGetLastError();
                    if (error != cudaSuccess) {
                        return error;
                    }
                } else {
                    grain_accumulate_kernel<<<blocks1D, threads1D, 0, stream>>>(
                        scaleWork,
                        layerWork,
                        total,
                        1);
                    error = cudaGetLastError();
                    if (error != cudaSuccess) {
                        return error;
                    }
                }
            } else {
                for (int layer = 0; layer < 3; ++layer) {
                    grain_layer_kernel<<<blocks2D, threads2D, 0, stream>>>(
                        pass,
                        width,
                        height,
                        density,
                        layerWork,
                        channel,
                        layer);
                    error = cudaGetLastError();
                    if (error != cudaSuccess) {
                        return error;
                    }
                    const float* dyeKernel = kernels->dyeKernel[layer][channel];
                    const int dyeRadius = kernels->dyeRadius[layer][channel];
                    if (dyeRadius > 0) {
                        error = launch_grain_blur_horizontal(
                            layerWork,
                            dyeKernel,
                            dyeRadius);
                        if (error != cudaSuccess) {
                            return error;
                        }
                        const int kernelLength = 2 * dyeRadius + 1;
                        const size_t verticalSharedBytes =
                            (static_cast<size_t>(kernelLength) +
                             static_cast<size_t>(threads2D.x) *
                                 static_cast<size_t>(threads2D.y + 2 * dyeRadius)) *
                            sizeof(float);
                        GrainBlurVerticalAccumulateInput finishInput{};
                        finishInput.horizontalBlur = filterTemp;
                        finishInput.output = scaleWork;
                        finishInput.width = width;
                        finishInput.height = height;
                        finishInput.kernel = dyeKernel;
                        finishInput.radius = dyeRadius;
                        finishInput.initialize = layer == 0 ? 1 : 0;
                        grain_blur_vertical_accumulate_kernel<<<
                            blocks2D,
                            threads2D,
                            verticalSharedBytes,
                            stream>>>(finishInput);
                        error = cudaGetLastError();
                        if (error != cudaSuccess) {
                            return error;
                        }
                    } else {
                        grain_accumulate_kernel<<<blocks1D, threads1D, 0, stream>>>(
                            scaleWork,
                            layerWork,
                            total,
                            layer == 0 ? 1 : 0);
                        error = cudaGetLastError();
                        if (error != cudaSuccess) {
                            return error;
                        }
                    }
                }
            }
        } else {
            grain_apply_simple_kernel<<<blocks2D, threads2D, 0, stream>>>(
                pass,
                width,
                height,
                density,
                scaleWork,
                channel);
            error = cudaGetLastError();
            if (error != cudaSuccess) {
                return error;
            }
        }

        const float densityBias = -base.densityMin[channel];
        if (densityBias != 0.0f) {
            grain_form_delta_kernel<<<blocks1D, threads1D, 0, stream>>>(
                scaleWork,
                density,
                total,
                densityBias);
            error = cudaGetLastError();
            if (error != cudaSuccess) {
                return error;
            }
        } else {
            grain_subtract_kernel<<<blocks1D, threads1D, 0, stream>>>(
                scaleWork,
                density,
                total,
                1.0f);
            error = cudaGetLastError();
            if (error != cudaSuccess) {
                return error;
            }
        }
        if (applyCorrelation && correlationRadius > 0) {
            error = launch_grain_blur_horizontal(
                scaleWork,
                correlationKernel,
                correlationRadius);
            if (error != cudaSuccess) {
                return error;
            }
            const int kernelLength = 2 * correlationRadius + 1;
            const size_t verticalSharedBytes =
                (static_cast<size_t>(kernelLength) +
                 static_cast<size_t>(threads2D.x) *
                     static_cast<size_t>(threads2D.y + 2 * correlationRadius)) *
                sizeof(float);
            GrainBlurVerticalAccumulateInput finishInput{};
            finishInput.horizontalBlur = filterTemp;
            finishInput.output = deltaAccum;
            finishInput.width = width;
            finishInput.height = height;
            finishInput.kernel = correlationKernel;
            finishInput.radius = correlationRadius;
            finishInput.weight = weight;
            finishInput.initialize = initializeDeltaAccum ? 1 : 0;
            grain_blur_vertical_accumulate_weighted_kernel<<<
                blocks2D,
                threads2D,
                verticalSharedBytes,
                stream>>>(finishInput);
            return cudaGetLastError();
        }
        grain_accumulate_weighted_kernel<<<
            blocks1D,
            threads1D,
            0,
            stream>>>(
            deltaAccum,
            scaleWork,
            total,
            weight,
            initializeDeltaAccum ? 1 : 0);
        return cudaGetLastError();
    };

    const float midFactor = std::sqrt(sizeMixScale);
    auto generate_mix = [&](const JuicerCuda::GrainPayload& base,
                            const float* density,
                            int channel) -> cudaError_t {
        cudaError_t error = cudaSuccess;
        bool initializeDeltaAccum = true;
        if (wFine > 0.0f) {
            error = generate_scale(
                base,
                density,
                channel,
                kSeedSaltFine,
                1.0f,
                kernels->blurKernel,
                kernels->blurRadius,
                kernels->blurRadius > 0,
                wFine,
                initializeDeltaAccum);
            if (error != cudaSuccess) {
                return error;
            }
            initializeDeltaAccum = false;
        }
        if (wMid > 0.0f) {
            error = generate_scale(
                base,
                density,
                channel,
                kSeedSaltMid,
                midFactor,
                kernels->blurKernelMid,
                kernels->blurRadiusMid,
                true,
                wMid,
                initializeDeltaAccum);
            if (error != cudaSuccess) {
                return error;
            }
            initializeDeltaAccum = false;
        }
        if (wCoarse > 0.0f) {
            error = generate_scale(
                base,
                density,
                channel,
                kSeedSaltCoarse,
                sizeMixScale,
                kernels->blurKernelCoarse,
                kernels->blurRadiusCoarse,
                true,
                wCoarse,
                initializeDeltaAccum);
            if (error != cudaSuccess) {
                return error;
            }
            initializeDeltaAccum = false;
        }
        grain_scale_kernel<<<blocks1D, threads1D, 0, stream>>>(
            deltaAccum,
            total,
            sizeMixGain);
        return cudaGetLastError();
    };

    float* density[3] = {densityC, densityM, densityY};
    cudaError_t error = cudaSuccess;
    if (grain->debugView == 6) {
        GrainDebugAverageInput debugInput{};
        debugInput.outputR = densityC;
        debugInput.outputG = densityM;
        debugInput.outputB = densityY;
        debugInput.input0 = densityC;
        debugInput.input1 = densityM;
        debugInput.input2 = densityY;
        debugInput.count = total;
        debugInput.offset = 0.0f;
        debugInput.scale = grain->debugScale * 4.0f;
        grain_debug_encode_avg3_kernel<<<blocks1D, threads1D, 0, stream>>>(debugInput);
        return cudaGetLastError();
    }

    grain_prepare_frame_uniforms_kernel<<<1, 1, 0, stream>>>(*grain);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }

    if (grain->debugView >= 2 && grain->debugView <= 5) {
        const bool coarse =
            grain->debugView == 3 || grain->debugView == 5;
        const bool raw = grain->debugView == 4 || grain->debugView == 5;
        for (int channel = 0; channel < 3; ++channel) {
            if (!coarse || wCoarse > 0.0f) {
                error = generate_scale(
                    *grain,
                    density[channel],
                    channel,
                    coarse ? kSeedSaltCoarse : kSeedSaltFine,
                    coarse ? sizeMixScale : 1.0f,
                    coarse ? kernels->blurKernelCoarse
                           : kernels->blurKernel,
                    coarse ? kernels->blurRadiusCoarse
                           : kernels->blurRadius,
                    !raw &&
                        (coarse ? kernels->blurRadiusCoarse > 0
                                : kernels->blurRadius > 0),
                    1.0f,
                    true);
                if (error != cudaSuccess) {
                    return error;
                }
            } else {
                grain_clear_kernel<<<blocks1D, threads1D, 0, stream>>>(
                    deltaAccum,
                    total);
                error = cudaGetLastError();
                if (error != cudaSuccess) {
                    return error;
                }
            }
            grain_reconstruct_kernel<<<blocks1D, threads1D, 0, stream>>>(
                density[channel],
                deltaAccum,
                total,
                amplitude,
                0);
            error = cudaGetLastError();
            if (error != cudaSuccess) {
                return error;
            }
        }
    } else {
        if (sharedChroma) {
            JuicerCuda::GrainPayload shared = *grain;
            shared.seedBase ^= kSeedSaltShared;
            shared.seedBaseNext ^= kSeedSaltShared;
            error = generate_mix(shared, densityM, 1);
            if (error != cudaSuccess) {
                return error;
            }
            error = cudaMemcpyAsync(
                sharedDelta,
                deltaAccum,
                static_cast<size_t>(total) * sizeof(float),
                cudaMemcpyDeviceToDevice,
                stream);
            if (error != cudaSuccess) {
                return error;
            }
        }
        for (int channel = 0; channel < 3; ++channel) {
            if (grain->chromaIndWeight > 0.0f || !sharedChroma) {
                error = generate_mix(*grain, density[channel], channel);
                if (error != cudaSuccess) {
                    return error;
                }
            } else {
                grain_clear_kernel<<<blocks1D, threads1D, 0, stream>>>(
                    deltaAccum,
                    total);
                error = cudaGetLastError();
                if (error != cudaSuccess) {
                    return error;
                }
            }
            if (sharedChroma) {
                GrainMixSharedInput mixInput{};
                mixInput.outputDelta = deltaAccum;
                mixInput.independentDelta = grain->chromaIndWeight > 0.0f ? deltaAccum : nullptr;
                mixInput.sharedDelta = sharedDelta;
                mixInput.count = total;
                mixInput.sharedWeight = grain->chromaSharedWeight;
                mixInput.independentWeight = grain->chromaIndWeight;
                mixInput.amplitude = amplitude;
                grain_mix_shared_kernel<<<blocks1D, threads1D, 0, stream>>>(mixInput);
                error = cudaGetLastError();
                if (error != cudaSuccess) {
                    return error;
                }
            }
            grain_reconstruct_kernel<<<blocks1D, threads1D, 0, stream>>>(
                density[channel],
                deltaAccum,
                total,
                sharedChroma ? 1.0f : amplitude,
                grain->debugView == 0 ? 1 : 0);
            error = cudaGetLastError();
            if (error != cudaSuccess) {
                return error;
            }
        }
    }

    if (grain->debugView != 0) {
        GrainDebugAverageInput debugInput{};
        debugInput.outputR = densityC;
        debugInput.outputG = densityM;
        debugInput.outputB = densityY;
        debugInput.input0 = densityC;
        debugInput.input1 = densityM;
        debugInput.input2 = densityY;
        debugInput.count = total;
        debugInput.offset = 0.5f;
        debugInput.scale = grain->debugScale;
        grain_debug_encode_avg3_kernel<<<blocks1D, threads1D, 0, stream>>>(debugInput);
        return cudaGetLastError();
    }
    return cudaSuccess;
}

// NOLINTEND(bugprone-easily-swappable-parameters)
extern "C" cudaError_t juicer_cuda_print_pipeline(
    const JuicerCuda::PipelineRunParams* hParams,
    void* cudaStreamOpaque) {
    if (!hParams) {
        return cudaErrorInvalidValue;
    }
    if (!hParams->printExpose.active) {
        return cudaErrorInvalidValue;
    }
    return juicer_cuda_negative_pipeline(hParams, cudaStreamOpaque);
}

extern "C" cudaError_t juicer_cuda_print_focused_pipeline(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    void* cudaStreamOpaque) {
    if (!hParams || !hParams->printExpose.active || !hParams->src || !hParams->dst) {
        return cudaErrorInvalidValue;
    }
    const JuicerCuda::PrintPipelineRunParams params = *hParams;
    if (params.width <= 0 || params.height <= 0) {
        return cudaSuccess;
    }
    if (!(params.nComponents == 3 || params.nComponents == 4) ||
        params.srcRowBytes == 0 || params.dstRowBytes == 0) {
        return cudaErrorInvalidValue;
    }
    cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
    dim3 threads(32, 8);
    dim3 blocks(
        (params.width + threads.x - 1) / threads.x,
        (params.height + threads.y - 1) / threads.y);
    pipeline_direct_kernel<<<blocks, threads, 0, stream>>>(params);
    return cudaGetLastError();
}
