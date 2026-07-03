// Cuda/Scan/JuicerCudaScanPipeline.cu
// Pipeline-aligned CUDA TU for print-development handoff and scan-stage kernels.
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "Cuda/JuicerCudaDirProfile.h"
#include "Cuda/JuicerCudaDeviceHelpers.cuh"
#include "openrand/philox.h"

namespace {

    struct CudaProfileStageTimer {
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        bool active = false;

        ~CudaProfileStageTimer() {
            destroy();
        }

        cudaError_t begin(cudaStream_t stream) {
            cudaError_t err = cudaEventCreateWithFlags(&start, cudaEventDefault);
            if (err != cudaSuccess) {
                destroy();
                return err;
            }
            err = cudaEventCreateWithFlags(&stop, cudaEventDefault);
            if (err != cudaSuccess) {
                destroy();
                return err;
            }
            err = cudaEventRecord(start, stream);
            if (err != cudaSuccess) {
                destroy();
                return err;
            }
            active = true;
            return cudaSuccess;
        }

        cudaError_t finish(cudaStream_t stream, JuicerCuda::SpatialDirStageProfile* stage, int launches) {
            if (!active) {
                return cudaErrorInvalidResourceHandle;
            }
            cudaError_t err = cudaEventRecord(stop, stream);
            if (err != cudaSuccess) {
                destroy();
                return err;
            }
            err = cudaEventSynchronize(stop);
            if (err != cudaSuccess) {
                destroy();
                return err;
            }
            float elapsedMs = 0.0f;
            err = cudaEventElapsedTime(&elapsedMs, start, stop);
            if (err == cudaSuccess && stage) {
                stage->launches += launches;
                stage->elapsedMs += elapsedMs;
            }
            destroy();
            return err;
        }

        void destroy() {
            if (start) {
                cudaEventDestroy(start);
                start = nullptr;
            }
            if (stop) {
                cudaEventDestroy(stop);
                stop = nullptr;
            }
            active = false;
        }
    };

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
        out.scanStage.scanErrorFlag = nullptr;
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

    JuicerCuda::PipelineRunParams profile_params_from_direct(
        const JuicerCuda::DirectPipelineRunParams& params) {
        JuicerCuda::PipelineRunParams out = focused_params_from_direct(params);
        out.scanStage.scanErrorFlag = nullptr;
        return out;
    }

    JuicerCuda::PipelineRunParams profile_params_from_print(
        const JuicerCuda::PrintPipelineRunParams& params) {
        JuicerCuda::PipelineRunParams out = focused_params_from_print(params);
        out.scanStage.scanErrorFlag = nullptr;
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

    __device__ __forceinline__ double encode_BT2020d_device(double v, double a, double b) {
        if (v < b) {
            return v * 4.5;
        }
        return a * pow(v, 0.45) - (a - 1.0);
    }

    __device__ __forceinline__ double encode_ProPhotod_device(double v, double threshold, double exponent) {
        if (v < threshold) {
            return v * 16.0;
        }
        return pow(v, exponent);
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
                return encode_BT2020d_device(v, static_cast<double>(cctf.a), static_cast<double>(cctf.b));
            case 4:
                return encode_ProPhotod_device(v, static_cast<double>(cctf.linearCutoff), static_cast<double>(cctf.gamma));
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
    float roughness) {
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
    int radius) {
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
    int radius) {
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
    float amount) {
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
    float strength) {
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

// Film stage kernels are defined in their respective TUs.
__global__ void expose_film_raw_kernel(
    JuicerCuda::PipelineRunParams params,
    float* outB,
    float* outG,
    float* outR);
__global__ void film_raw_max_kernel(
    const float* inB,
    const float* inG,
    const float* inR,
    int n,
    unsigned int* outMaxBits);
__global__ void highlight_boost_film_raw_kernel(
    JuicerCuda::HighlightBoostPayload boost,
    const unsigned int* maxRawBits,
    float* inOutB,
    float* inOutG,
    float* inOutR,
    int n);
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
__global__ void grain_accumulate_kernel(float* dst, const float* src, int n);
__global__ void grain_add_bias_kernel(float* inOut, int n, float bias);
__global__ void grain_multiply_kernel(float* inOut, const float* mult, int n);
__global__ void grain_subtract_kernel(float* inOut, const float* sub, int n, float amplitude);
__global__ void grain_mix_delta3_kernel(
    float* outDelta,
    const float* fineDelta,
    const float* midDelta,
    const float* coarseDelta,
    int n,
    float wMid,
    float wCoarse,
    float gain,
    float amplitude);
__global__ void grain_mix_shared_kernel(
    float* outDelta,
    const float* indDelta,
    const float* sharedDelta,
    int n,
    float wShared,
    float wInd,
    float amplitude);
__global__ void grain_debug_encode_avg3_kernel(
    float* outR,
    float* outG,
    float* outB,
    const float* in0,
    const float* in1,
    const float* in2,
    int n,
    float offset,
    float scale);
__global__ void grain_apply_simple_kernel(
    JuicerCuda::PipelineRunParams params,
    float* inOut,
    int channelIndex);
__global__ void grain_layer_kernel(
    JuicerCuda::PipelineRunParams params,
    const float* inDensity,
    float* outGrain,
    int channelIndex,
    int sublayerIndex);

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

    __device__ __forceinline__ float dust_mask_device(
        float amount,
        float x,
        float y,
        float pixelSizeUm,
        std::uint64_t seed,
        float cellMm,
        float lambdaPerMm2,
        float sizeUm,
        float strength,
        float brightMix,
        float brightScale) {
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
        float amount,
        float x,
        float y,
        float pixelSizeUm,
        std::uint64_t seed,
        float cellMmX,
        float cellMmY,
        float lambdaPerMm,
        float widthUm,
        float strength,
        float maxAngleRad,
        float brightMix,
        float brightScale) {
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
        float dustAmount,
        float scratchAmount,
        float x,
        float y,
        float pixelSizeUm,
        std::uint64_t seedDust,
        std::uint64_t seedScratch) {
        if (!(pixelSizeUm > 0.0f)) {
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

        const float gateDust = dust_mask_device(
            dustAmount, x, y, pixelSizeUm, seedDust, gateDustCellMm, kGateDustBaseProb, kGateDustSizeUm, kGateDustStrength, kGateDustBrightMix, kGateDustBrightScale);
        const float gateScratch = scratch_mask_device(
            scratchAmount, x, y, pixelSizeUm, seedScratch, gateScratchCellMmX, gateScratchCellMmY, kGateScratchBaseProb, kGateScratchWidthUm, kGateScratchStrength, kGateScratchMaxAngle, kGateScratchBrightMix, kGateScratchBrightScale);
        float gateMask = gateDust + gateScratch;
        if (!device_isfinite(gateMask)) {
            gateMask = 0.0f;
        }
        return gateMask;
    }

    __device__ __forceinline__ float sample_gate_mask_device(
        const float* mask,
        int width,
        int height,
        float x,
        float y) {
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

__global__ void halation_apply_kernel(
    float* inOut,
    const float* blurred,
    int n,
    float strength);

__global__ void develop_print_density_kernel(
    JuicerCuda::PipelineRunParams params,
    float* ioC,
    float* ioM,
    float* ioY) {
    if (!params.printExpose.active) {
        return;
    }

    if (!ioC || !ioM || !ioY) {
        return;
    }

    for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < params.height; y += blockDim.y * gridDim.y) {
        for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < params.width; x += blockDim.x * gridDim.x) {
            const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
            float D_cmy[3] = {ioC[idx], ioM[idx], ioY[idx]};
            apply_print_pipeline_device(params.printExpose, params.printDevelop, D_cmy);
            ioC[idx] = D_cmy[0];
            ioM[idx] = D_cmy[1];
            ioY[idx] = D_cmy[2];
        }
    }
}

__global__ void profile_print_develop_spectral_integrate_kernel(
    JuicerCuda::PipelineRunParams params,
    float* ioC,
    float* ioM,
    float* ioY) {
    if (!params.printExpose.active) {
        return;
    }

    if (!ioC || !ioM || !ioY) {
        return;
    }

    for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < params.height; y += blockDim.y * gridDim.y) {
        for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < params.width; x += blockDim.x * gridDim.x) {
            const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
            float D_cmy[3] = {ioC[idx], ioM[idx], ioY[idx]};
            float rawPrint[3] = {0.0f, 0.0f, 0.0f};
            (void)print_spectral_integrate_device(params.printExpose, D_cmy, rawPrint);
            ioC[idx] = rawPrint[0];
            ioM[idx] = rawPrint[1];
            ioY[idx] = rawPrint[2];
        }
    }
}

__global__ void profile_print_develop_exposure_scale_kernel(
    JuicerCuda::PipelineRunParams params,
    float* ioC,
    float* ioM,
    float* ioY) {
    if (!params.printExpose.active) {
        return;
    }

    if (!ioC || !ioM || !ioY) {
        return;
    }

    for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < params.height; y += blockDim.y * gridDim.y) {
        for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < params.width; x += blockDim.x * gridDim.x) {
            const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
            float rawPrint[3] = {ioC[idx], ioM[idx], ioY[idx]};
            print_apply_exposure_scale_device(params.printExpose, rawPrint);
            ioC[idx] = rawPrint[0];
            ioM[idx] = rawPrint[1];
            ioY[idx] = rawPrint[2];
        }
    }
}

__global__ void profile_print_develop_log_encode_kernel(
    JuicerCuda::PipelineRunParams params,
    float* ioC,
    float* ioM,
    float* ioY) {
    if (!params.printExpose.active) {
        return;
    }

    if (!ioC || !ioM || !ioY) {
        return;
    }

    for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < params.height; y += blockDim.y * gridDim.y) {
        for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < params.width; x += blockDim.x * gridDim.x) {
            const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
            const float rawPrint[3] = {ioC[idx], ioM[idx], ioY[idx]};
            float logPrint[3] = {0.0f, 0.0f, 0.0f};
            print_log_encode_device(rawPrint, logPrint);
            ioC[idx] = logPrint[0];
            ioM[idx] = logPrint[1];
            ioY[idx] = logPrint[2];
        }
    }
}

__global__ void profile_print_develop_density_curve_kernel(
    JuicerCuda::PipelineRunParams params,
    float* ioC,
    float* ioM,
    float* ioY) {
    if (!params.printExpose.active) {
        return;
    }

    if (!ioC || !ioM || !ioY) {
        return;
    }

    for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < params.height; y += blockDim.y * gridDim.y) {
        for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < params.width; x += blockDim.x * gridDim.x) {
            const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
            const float logPrint[3] = {ioC[idx], ioM[idx], ioY[idx]};
            float D_cmy[3] = {0.0f, 0.0f, 0.0f};
            print_sample_density_curves_device(params.printDevelop, logPrint, D_cmy);
            ioC[idx] = D_cmy[0];
            ioM[idx] = D_cmy[1];
            ioY[idx] = D_cmy[2];
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

    __device__ __forceinline__ float sample_plane_mitchell_device(
        const float* JUICER_RESTRICT plane,
        int width,
        int height,
        float x,
        float y) {
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
    __global__ void pipeline_direct_kernel(Params params) {
        const JuicerCuda::FilmDevelopPayload& dev = params.filmDevelop;
        const JuicerCuda::ScanStagePayload& scan = params.scanStage;
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        if (x >= params.width || y >= params.height) {
            return;
        }

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

        const bool useSpatialDir = juicer_cuda_spatial_dir_filtered_correction_active_device(dev);

        float D_cmy[3] = {0.0f, 0.0f, 0.0f};
        if (useSpatialDir) {
            const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
            float logE_raw[3] = {0.0f, 0.0f, 0.0f};
            if (juicer_cuda_spatial_dir_cached_log_raw_active_device(dev)) {
                juicer_cuda_load_spatial_dir_cached_log_raw_device(dev, idx, logE_raw);
            } else {
                float logE_sanitized[3] = {0.0f, 0.0f, 0.0f};
                float layerPre[3] = {0.0f, 0.0f, 0.0f};
                compute_logE_and_layer_pre_device(params, rgbIn, logE_raw, logE_sanitized, layerPre);
            }
            juicer_cuda_develop_dir_final_device(dev, logE_raw, idx, D_cmy);
        } else {
            float logE_raw[3] = {0.0f, 0.0f, 0.0f};
            float logE_sanitized[3] = {0.0f, 0.0f, 0.0f};
            float layerPre[3] = {0.0f, 0.0f, 0.0f};
            compute_logE_and_layer_pre_device(params, rgbIn, logE_raw, logE_sanitized, layerPre);

            if (dev.dir.active) {
                float logE_corr[3] = {logE_sanitized[0], logE_sanitized[1], logE_sanitized[2]};
                apply_dir_runtime_logE_device(logE_corr, layerPre, dev.dir, dev.densB, dev.densG, dev.densR);

                const JuicerCuda::DeviceCurveView cB = dev.dirPrecorrected ? dev.dirDensB : dev.densB;
                const JuicerCuda::DeviceCurveView cG = dev.dirPrecorrected ? dev.dirDensG : dev.densG;
                const JuicerCuda::DeviceCurveView cR = dev.dirPrecorrected ? dev.dirDensR : dev.densR;

                const float DY = sample_density_at_logE_device(cB, logE_corr[0], dev.gammaFactorB);
                const float DM = sample_density_at_logE_device(cG, logE_corr[1], dev.gammaFactorG);
                const float DC = sample_density_at_logE_device(cR, logE_corr[2], dev.gammaFactorR);

                D_cmy[0] = DC;
                D_cmy[1] = DM;
                D_cmy[2] = DY;
            } else {
                // Map B/G/R layer densities to C/M/Y dyes
                D_cmy[0] = layerPre[2];
                D_cmy[1] = layerPre[1];
                D_cmy[2] = layerPre[0];
            }
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
        const std::size_t idx =
            static_cast<std::size_t>(y) * static_cast<std::size_t>(params.width) +
            static_cast<std::size_t>(x);
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

    __global__ void scan_linear_rgb_kernel(
        JuicerCuda::PipelineRunParams params,
        const float* inC,
        const float* inM,
        const float* inY,
        const float* glarePercent,
        float* outR,
        float* outG,
        float* outB) {
        const JuicerCuda::ScanStagePayload& scan = params.scanStage;
        if (!inC || !inM || !inY || !outR || !outG || !outB) {
            return;
        }
        for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < params.height; y += blockDim.y * gridDim.y) {
            for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < params.width; x += blockDim.x * gridDim.x) {
                const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);

                const float D_cmy[3] = {inC[idx], inM[idx], inY[idx]};

                // Scan: normalize density -> logXYZ
                double D_norm[3];
                if (scan.scanTables.mediumIsNegative) {
                    D_norm[0] = (static_cast<double>(D_cmy[0]) + static_cast<double>(scan.scanTables.min_cmy[0])) * static_cast<double>(scan.scanTables.inv_max_cmy[0]);
                    D_norm[1] = (static_cast<double>(D_cmy[1]) + static_cast<double>(scan.scanTables.min_cmy[1])) * static_cast<double>(scan.scanTables.inv_max_cmy[1]);
                    D_norm[2] = (static_cast<double>(D_cmy[2]) + static_cast<double>(scan.scanTables.min_cmy[2])) * static_cast<double>(scan.scanTables.inv_max_cmy[2]);
                } else {
                    D_norm[0] = static_cast<double>(D_cmy[0]) * static_cast<double>(scan.scanTables.inv_max_cmy[0]);
                    D_norm[1] = static_cast<double>(D_cmy[1]) * static_cast<double>(scan.scanTables.inv_max_cmy[1]);
                    D_norm[2] = static_cast<double>(D_cmy[2]) * static_cast<double>(scan.scanTables.inv_max_cmy[2]);
                }

                float log2XYZ[3] = {nanf(""), nanf(""), nanf("")};
                (void)scan_log2_xyz_device(scan, D_norm, log2XYZ);
                double xyz[3] = {
                    static_cast<double>(exp2f(log2XYZ[0])),
                    static_cast<double>(exp2f(log2XYZ[1])),
                    static_cast<double>(exp2f(log2XYZ[2]))};

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

    __global__ void profile_focused_scan_linear_rgb_kernel(
        JuicerCuda::PipelineRunParams params,
        const float* inC,
        const float* inM,
        const float* inY,
        float* outR,
        float* outG,
        float* outB) {
        const JuicerCuda::ScanStagePayload& scan = params.scanStage;
        if (!inC || !inM || !inY || !outR || !outG || !outB) {
            return;
        }
        for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < params.height; y += blockDim.y * gridDim.y) {
            for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < params.width; x += blockDim.x * gridDim.x) {
                const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
                const float D_cmy[3] = {inC[idx], inM[idx], inY[idx]};

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

                double adapted[3];
                mat3_mul_vec_double_device(scan.scanColor.cat02, xyz, adapted);
                double rgbOut[3];
                mat3_mul_vec_double_device(scan.scanColor.xyzToRgb, adapted, rgbOut);

                if (!isfinite(rgbOut[0]) || !isfinite(rgbOut[1]) || !isfinite(rgbOut[2])) {
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

    __global__ void apply_print_pipeline_to_density_kernel(
        JuicerCuda::PipelineRunParams params,
        float* ioC,
        float* ioM,
        float* ioY) {
        if (!ioC || !ioM || !ioY || !params.printExpose.active) {
            return;
        }
        for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < params.height; y += blockDim.y * gridDim.y) {
            for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < params.width; x += blockDim.x * gridDim.x) {
                const size_t idx =
                    static_cast<size_t>(y) * static_cast<size_t>(params.width) +
                    static_cast<size_t>(x);
                float D_cmy[3] = {ioC[idx], ioM[idx], ioY[idx]};
                apply_print_pipeline_device(params.printExpose, params.printDevelop, D_cmy);
                ioC[idx] = D_cmy[0];
                ioM[idx] = D_cmy[1];
                ioY[idx] = D_cmy[2];
            }
        }
    }

    __global__ void scan_linear_density_rgb_kernel(
        JuicerCuda::PipelineRunParams params,
        const float* inC,
        const float* inM,
        const float* inY,
        const float* glarePercent,
        float* outR,
        float* outG,
        float* outB) {
        const JuicerCuda::ScanStagePayload& scan = params.scanStage;
        if (!inC || !inM || !inY || !outR || !outG || !outB) {
            return;
        }
        for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < params.height; y += blockDim.y * gridDim.y) {
            for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < params.width; x += blockDim.x * gridDim.x) {
                const size_t idx =
                    static_cast<size_t>(y) * static_cast<size_t>(params.width) +
                    static_cast<size_t>(x);
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

    __global__ void profile_focused_output_encode_kernel(
        JuicerCuda::PipelineRunParams params,
        float* ioR,
        float* ioG,
        float* ioB) {
        if (!ioR || !ioG || !ioB) {
            return;
        }
        const JuicerCuda::ScanStagePayload& scan = params.scanStage;
        for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < params.height; y += blockDim.y * gridDim.y) {
            for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < params.width; x += blockDim.x * gridDim.x) {
                const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
                double rgbOut[3] = {
                    static_cast<double>(ioR[idx]),
                    static_cast<double>(ioG[idx]),
                    static_cast<double>(ioB[idx])};
                apply_output_encoding_device(scan.scanColor.encoding, rgbOut);
                ioR[idx] = static_cast<float>(rgbOut[0]);
                ioG[idx] = static_cast<float>(rgbOut[1]);
                ioB[idx] = static_cast<float>(rgbOut[2]);
            }
        }
    }

    __global__ void apply_film_defects_kernel(
        JuicerCuda::PipelineRunParams params,
        float* ioC,
        float* ioM,
        float* ioY) {
        const JuicerCuda::GrainPayload& grain = params.grain;
        if (!ioC || !ioM || !ioY) {
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
        for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < params.height; y += blockDim.y * gridDim.y) {
            for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < params.width; x += blockDim.x * gridDim.x) {
                const float absX = static_cast<float>(grain.originX + x);
                const float absY = static_cast<float>(grain.originY + y);
                const float rollY = absY + rollPx * time;
                const float xMm = absX * pixelToMm;
                const float rollYMm = rollY * pixelToMm;

                const float dustMask = dust_mask_device(
                    dustAmount, xMm, rollYMm, grain.pixelSizeUm, seedDust, dustCellMm, kDustBaseProb, kDustSizeUm, kDustStrength, kDustBrightMix, kDustBrightScale);
                const float scratchMask = scratch_mask_device(
                    scratchAmount, xMm, rollYMm, grain.pixelSizeUm, seedScratch, scratchCellMmX, scratchCellMmY, kScratchBaseProb, kScratchWidthUm, kScratchStrength, kScratchMaxAngle, kScratchBrightMix, kScratchBrightScale);
                float delta = dustMask + scratchMask;
                if (!device_isfinite(delta)) {
                    delta = 0.0f;
                }
                if (delta != 0.0f) {
                    const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
                    ioC[idx] = ioC[idx] + delta;
                    ioM[idx] = ioM[idx] + delta;
                    ioY[idx] = ioY[idx] + delta;
                }
            }
        }
    }

    __global__ void gate_defect_mask_kernel(
        JuicerCuda::PipelineRunParams params,
        float* outMask,
        int maskWidth,
        int maskHeight) {
        const JuicerCuda::GrainPayload& grain = params.grain;
        if (!outMask) {
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
        for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < maskHeight; y += blockDim.y * gridDim.y) {
            for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < maskWidth; x += blockDim.x * gridDim.x) {
                if (!gateActive || !pixelSizeValid) {
                    outMask[static_cast<size_t>(y) * static_cast<size_t>(maskWidth) + static_cast<size_t>(x)] = 0.0f;
                    continue;
                }

                const float absX = static_cast<float>(grain.originX) + static_cast<float>(x) * 2.0f;
                const float absY = static_cast<float>(grain.originY) + static_cast<float>(y) * 2.0f;
                const float xMm = absX * pixelToMm;
                const float yMm = absY * pixelToMm;

                const float gateMask = gate_mask_device(
                    dustAmount,
                    scratchAmount,
                    xMm,
                    yMm,
                    grain.pixelSizeUm,
                    seedGateDust,
                    seedGateScratch);

                outMask[static_cast<size_t>(y) * static_cast<size_t>(maskWidth) + static_cast<size_t>(x)] = gateMask;
            }
        }
    }

    __global__ void scan_output_encode_kernel(
        JuicerCuda::PipelineRunParams params,
        const float* rgbR,
        const float* rgbG,
        const float* rgbB) {
        const JuicerCuda::ScanStagePayload& scan = params.scanStage;
        const JuicerCuda::GateWeavePayload& weave = params.gateWeave;
        const JuicerCuda::GrainPayload& grain = params.grain;

        if (!params.src || !params.dst || params.srcRowBytes == 0 || params.dstRowBytes == 0) {
            return;
        }
        if (!rgbR || !rgbG || !rgbB) {
            return;
        }

        const int nC = params.nComponents;
        if (!(nC == 3 || nC == 4)) {
            return;
        }
        const std::size_t pixelBytes = static_cast<std::size_t>(nC) * sizeof(float);
        const int debugView = grain.debugView;
        for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < params.height; y += blockDim.y * gridDim.y) {
            char* dstRow = reinterpret_cast<char*>(params.dst) + static_cast<std::size_t>(y) * params.dstRowBytes;
            const char* srcRow = reinterpret_cast<const char*>(params.src) + static_cast<std::size_t>(y) * params.srcRowBytes;
            for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < params.width; x += blockDim.x * gridDim.x) {
                const float absX = static_cast<float>(grain.originX + x);
                const float absY = static_cast<float>(grain.originY + y);
                double rgbOut[3];
                if (debugView != 0) {
                    const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
                    rgbOut[0] = static_cast<double>(rgbR[idx]);
                    rgbOut[1] = static_cast<double>(rgbG[idx]);
                    rgbOut[2] = static_cast<double>(rgbB[idx]);
                } else if (weave.active != 0) {
                    const float cx = 0.5f * static_cast<float>(params.width - 1);
                    const float cy = 0.5f * static_cast<float>(params.height - 1);
                    const float fx = static_cast<float>(x) - cx;
                    const float fy = static_cast<float>(y) - cy;
                    const float c = weave.cosRot;
                    const float s = weave.sinRot;
                    const float srcX = c * fx - s * fy + cx + weave.dxPx;
                    const float srcY = s * fx + c * fy + cy + weave.dyPx;
                    rgbOut[0] = static_cast<double>(sample_plane_mitchell_device(rgbR, params.width, params.height, srcX, srcY));
                    rgbOut[1] = static_cast<double>(sample_plane_mitchell_device(rgbG, params.width, params.height, srcX, srcY));
                    rgbOut[2] = static_cast<double>(sample_plane_mitchell_device(rgbB, params.width, params.height, srcX, srcY));
                } else {
                    const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
                    rgbOut[0] = static_cast<double>(rgbR[idx]);
                    rgbOut[1] = static_cast<double>(rgbG[idx]);
                    rgbOut[2] = static_cast<double>(rgbB[idx]);
                }
                if (debugView == 0) {
                    const std::uint64_t sessionSeed = (grain.stbnSessionSeed != 0) ? grain.stbnSessionSeed : 1ULL;
                    const std::uint64_t seedGateDust = splitmix64_device(sessionSeed ^ 0xA1B2C3D4E5F60718ULL);
                    const std::uint64_t seedGateScratch = splitmix64_device(sessionSeed ^ 0xC6A4A7935BD1E995ULL);
                    const float pixelToMm = grain.pixelSizeUm * 0.001f;
                    const float xMm = absX * pixelToMm;
                    const float yMm = absY * pixelToMm;
                    const bool gateActive = (grain.gateDustAmount > 0.0f) || (grain.gateScratchAmount > 0.0f);
                    if (gateActive) {
                        float gateMask = 0.0f;
                        if (grain.gateMask && grain.gateMaskWidth > 0 && grain.gateMaskHeight > 0) {
                            const float maskX = (absX - static_cast<float>(grain.originX)) * 0.5f;
                            const float maskY = (absY - static_cast<float>(grain.originY)) * 0.5f;
                            gateMask = sample_gate_mask_device(grain.gateMask, grain.gateMaskWidth, grain.gateMaskHeight, maskX, maskY);
                        } else {
                            gateMask = gate_mask_device(
                                grain.gateDustAmount,
                                grain.gateScratchAmount,
                                xMm,
                                yMm,
                                grain.pixelSizeUm,
                                seedGateDust,
                                seedGateScratch);
                        }
                        if (device_isfinite(gateMask) && gateMask != 0.0f) {
                            gateMask = fminf(fmaxf(gateMask, -0.5f), 0.95f);
                            const float trans = 1.0f - gateMask;
                            rgbOut[0] *= static_cast<double>(trans);
                            rgbOut[1] *= static_cast<double>(trans);
                            rgbOut[2] *= static_cast<double>(trans);
                        }
                    }
                    apply_output_encoding_device(scan.scanColor.encoding, rgbOut);
                }

                float* dstPix = reinterpret_cast<float*>(dstRow + static_cast<std::size_t>(x) * pixelBytes);
                dstPix[0] = static_cast<float>(rgbOut[0]);
                dstPix[1] = static_cast<float>(rgbOut[1]);
                dstPix[2] = static_cast<float>(rgbOut[2]);

                if (nC == 4) {
                    const float* srcPix = reinterpret_cast<const float*>(srcRow + static_cast<std::size_t>(x) * pixelBytes);
                    dstPix[3] = srcPix ? srcPix[3] : 1.0f;
                }
            }
        }
    }

    template <typename Params>
    __global__ void focused_scan_output_encode_kernel(
        Params params,
        const float* rgbR,
        const float* rgbG,
        const float* rgbB) {
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
        const std::size_t idx =
            static_cast<std::size_t>(y) * static_cast<std::size_t>(params.width) +
            static_cast<std::size_t>(x);
        double rgbOut[3] = {
            static_cast<double>(rgbR[idx]),
            static_cast<double>(rgbG[idx]),
            static_cast<double>(rgbB[idx])};
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

extern "C" cudaError_t juicer_cuda_build_gate_defect_mask(
    const JuicerCuda::PipelineRunParams* hParams,
    float* dGateMask,
    int gateWidth,
    int gateHeight,
    void* cudaStreamOpaque) {
    if (!hParams || !dGateMask) {
        return cudaErrorInvalidValue;
    }

    if (gateWidth <= 0 || gateHeight <= 0) {
        return cudaSuccess;
    }

    const JuicerCuda::PipelineRunParams params = *hParams;
    cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

    dim3 threads(32, 8);
    dim3 blocks(
        static_cast<unsigned int>((gateWidth + threads.x - 1) / threads.x),
        static_cast<unsigned int>((gateHeight + threads.y - 1) / threads.y));
    gate_defect_mask_kernel<<<blocks, threads, 0, stream>>>(params, dGateMask, gateWidth, gateHeight);
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
        optics_glare_generate_kernel<<<blocks, threads, 0, stream>>>(
            dTmp,
            params.width,
            params.height,
            options.glareSeed,
            mediumId,
            options.glareOriginX,
            options.glareOriginY,
            options.glarePercent,
            options.glareRoughness);
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
cudaError_t launch_focused_spatial_dir_final_develop_density(
    const Params* hParams,
    float* dDensityC,
    float* dDensityM,
    float* dDensityY,
    void* cudaStreamOpaque) {
    if (!hParams || !dDensityC || !dDensityM || !dDensityY) {
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
    const JuicerCuda::SpatialDirPayload& spatialDir = params.filmDevelop.spatialDir;
    if (!spatialDir.active || !spatialDir.corrY || !spatialDir.corrM ||
        !spatialDir.corrC) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream =
        cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
    dim3 threads(32, 8);
    dim3 blocks(
        static_cast<unsigned int>((params.width + threads.x - 1) / threads.x),
        static_cast<unsigned int>((params.height + threads.y - 1) / threads.y));

    develop_film_density_from_raw_kernel<<<blocks, threads, 0, stream>>>(
        params,
        dDensityC,
        dDensityM,
        dDensityY,
        dDensityC,
        dDensityM,
        dDensityY);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }

    if constexpr (requires { focusedParams.printExpose; focusedParams.printDevelop; }) {
        apply_print_pipeline_to_density_kernel<<<blocks, threads, 0, stream>>>(
            params,
            dDensityC,
            dDensityM,
            dDensityY);
        error = cudaGetLastError();
    }
    return error;
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
        static_cast<unsigned int>((params.width + threads.x - 1) / threads.x),
        static_cast<unsigned int>((params.height + threads.y - 1) / threads.y));

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
        optics_glare_generate_kernel<<<blocks, threads, 0, stream>>>(
            dTmp,
            params.width,
            params.height,
            options.glareSeed,
            mediumId,
            options.glareOriginX,
            options.glareOriginY,
            options.glarePercent,
            options.glareRoughness);
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

    scan_linear_density_rgb_kernel<<<blocks, threads, 0, stream>>>(
        params,
        dDensityC,
        dDensityM,
        dDensityY,
        glarePercent,
        rgb.r,
        rgb.g,
        rgb.b);
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
    optics_blur_horizontal_kernel<<<blocks, threads, horizontalShared, stream>>>(
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
    if (lensBlurRadius > 0 && dLensBlurKernel) {
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

    if (unsharpRadius > 0 && dUnsharpKernel &&
        std::isfinite(static_cast<double>(unsharpAmount)) &&
        unsharpAmount > 0.0f) {
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
            optics_blur_horizontal_kernel<<<blocks, threads, horizontalShared, stream>>>(
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
            optics_unsharp_vertical_combine_kernel<<<blocks, threads, verticalShared, stream>>>(
                plane,
                dTmp,
                params.width,
                params.height,
                dUnsharpKernel,
                unsharpRadius,
                unsharpAmount);
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
    focused_scan_output_encode_kernel<<<blocks, threads, 0, stream>>>(params, rgb.r, rgb.g, rgb.b);
    return cudaGetLastError();
}

template <typename Params>
cudaError_t profile_focused_pipeline_stages_impl(
    const Params* hParams,
    float* dPlane0,
    float* dPlane1,
    float* dPlane2,
    JuicerCuda::CompositePipelineProfile* profile,
    void* cudaStreamOpaque) {
    if (!hParams || !profile || !dPlane0 || !dPlane1 || !dPlane2) {
        return cudaErrorInvalidValue;
    }
    const Params focusedParams = *hParams;
    if (!focusedParams.src) {
        return cudaErrorInvalidValue;
    }
    if (focusedParams.width <= 0 || focusedParams.height <= 0) {
        *profile = JuicerCuda::CompositePipelineProfile{};
        profile->captured = 1;
        profile->profileKind = "focused_split_attribution";
        profile->profileNote = "empty_frame";
        return cudaSuccess;
    }
    if (!(focusedParams.nComponents == 3 || focusedParams.nComponents == 4) ||
        focusedParams.srcRowBytes == 0) {
        return cudaErrorInvalidValue;
    }

    const JuicerCuda::PipelineRunParams params = [&]() {
        if constexpr (requires { focusedParams.printExpose; focusedParams.printDevelop; }) {
            return profile_params_from_print(focusedParams);
        } else {
            return profile_params_from_direct(focusedParams);
        }
    }();

    cudaStream_t stream =
        cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
    dim3 threads(32, 8);
    dim3 blocks(
        static_cast<unsigned int>((params.width + threads.x - 1) / threads.x),
        static_cast<unsigned int>((params.height + threads.y - 1) / threads.y));

    *profile = JuicerCuda::CompositePipelineProfile{};
    profile->width = params.width;
    profile->height = params.height;
    profile->captured = 1;
    profile->profileKind = "focused_split_attribution";
    profile->profileNote = "scratch_only_print_develop_split_real_focused_pipeline_still_renders_frame";

    CudaProfileStageTimer totalTimer;
    cudaError_t err = totalTimer.begin(stream);
    if (err != cudaSuccess) {
        return err;
    }

    auto run_stage = [&](JuicerCuda::SpatialDirStageProfile& stage, int launches, auto launch) -> cudaError_t {
        CudaProfileStageTimer timer;
        cudaError_t stageErr = timer.begin(stream);
        if (stageErr != cudaSuccess) {
            return stageErr;
        }
        stageErr = launch();
        if (stageErr != cudaSuccess) {
            return stageErr;
        }
        profile->totalLaunches += launches;
        return timer.finish(stream, &stage, launches);
    };

    err = run_stage(profile->filmRaw, 1, [&]() {
        expose_film_raw_kernel<<<blocks, threads, 0, stream>>>(params, dPlane0, dPlane1, dPlane2);
        return cudaGetLastError();
    });
    if (err != cudaSuccess) {
        return err;
    }

    err = run_stage(profile->filmDevelop, 1, [&]() {
        develop_film_density_from_raw_kernel<<<blocks, threads, 0, stream>>>(
            params,
            dPlane0,
            dPlane1,
            dPlane2,
            dPlane0,
            dPlane1,
            dPlane2);
        return cudaGetLastError();
    });
    if (err != cudaSuccess) {
        return err;
    }

    if (params.printExpose.active) {
        JuicerCuda::PrintDevelopBreakdownProfile& printBreakdown =
            profile->printDevelopBreakdown;
        printBreakdown.captured = 1;
        printBreakdown.profileKind = "print_develop_split_attribution";
        printBreakdown.profileNote = "profile_only_four_kernel_split_total_is_diagnostic";

        CudaProfileStageTimer printTotalTimer;
        err = printTotalTimer.begin(stream);
        if (err != cudaSuccess) {
            return err;
        }

        err = run_stage(printBreakdown.spectralIntegrate, 1, [&]() {
            profile_print_develop_spectral_integrate_kernel<<<blocks, threads, 0, stream>>>(
                params,
                dPlane0,
                dPlane1,
                dPlane2);
            return cudaGetLastError();
        });
        if (err != cudaSuccess) {
            return err;
        }

        err = run_stage(printBreakdown.exposureScale, 1, [&]() {
            profile_print_develop_exposure_scale_kernel<<<blocks, threads, 0, stream>>>(
                params,
                dPlane0,
                dPlane1,
                dPlane2);
            return cudaGetLastError();
        });
        if (err != cudaSuccess) {
            return err;
        }

        err = run_stage(printBreakdown.logEncode, 1, [&]() {
            profile_print_develop_log_encode_kernel<<<blocks, threads, 0, stream>>>(
                params,
                dPlane0,
                dPlane1,
                dPlane2);
            return cudaGetLastError();
        });
        if (err != cudaSuccess) {
            return err;
        }

        err = run_stage(printBreakdown.densityCurve, 1, [&]() {
            profile_print_develop_density_curve_kernel<<<blocks, threads, 0, stream>>>(
                params,
                dPlane0,
                dPlane1,
                dPlane2);
            return cudaGetLastError();
        });
        if (err != cudaSuccess) {
            return err;
        }

        const int printLaunches =
            printBreakdown.spectralIntegrate.launches +
            printBreakdown.exposureScale.launches +
            printBreakdown.logEncode.launches +
            printBreakdown.densityCurve.launches;
        err = printTotalTimer.finish(stream, &profile->printDevelop, printLaunches);
        if (err != cudaSuccess) {
            return err;
        }
        printBreakdown.total = profile->printDevelop;
    }

    err = run_stage(profile->scannerLinear, 1, [&]() {
        profile_focused_scan_linear_rgb_kernel<<<blocks, threads, 0, stream>>>(
            params,
            dPlane0,
            dPlane1,
            dPlane2,
            dPlane0,
            dPlane1,
            dPlane2);
        return cudaGetLastError();
    });
    if (err != cudaSuccess) {
        return err;
    }

    err = run_stage(profile->outputEncode, 1, [&]() {
        profile_focused_output_encode_kernel<<<blocks, threads, 0, stream>>>(
            params,
            dPlane0,
            dPlane1,
            dPlane2);
        return cudaGetLastError();
    });
    if (err != cudaSuccess) {
        return err;
    }

    return totalTimer.finish(stream, &profile->total, profile->totalLaunches);
}

extern "C" cudaError_t juicer_cuda_profile_direct_focused_pipeline_stages(
    const JuicerCuda::DirectPipelineRunParams* hParams,
    float* dPlane0,
    float* dPlane1,
    float* dPlane2,
    JuicerCuda::CompositePipelineProfile* profile,
    void* cudaStreamOpaque) {
    return profile_focused_pipeline_stages_impl(
        hParams,
        dPlane0,
        dPlane1,
        dPlane2,
        profile,
        cudaStreamOpaque);
}

extern "C" cudaError_t juicer_cuda_profile_print_focused_pipeline_stages(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    float* dPlane0,
    float* dPlane1,
    float* dPlane2,
    JuicerCuda::CompositePipelineProfile* profile,
    void* cudaStreamOpaque) {
    return profile_focused_pipeline_stages_impl(
        hParams,
        dPlane0,
        dPlane1,
        dPlane2,
        profile,
        cudaStreamOpaque);
}

extern "C" cudaError_t juicer_cuda_direct_focused_spatial_dir_final_develop_density(
    const JuicerCuda::DirectPipelineRunParams* hParams,
    float* dDensityC,
    float* dDensityM,
    float* dDensityY,
    void* cudaStreamOpaque) {
    return launch_focused_spatial_dir_final_develop_density(
        hParams,
        dDensityC,
        dDensityM,
        dDensityY,
        cudaStreamOpaque);
}

extern "C" cudaError_t juicer_cuda_print_focused_spatial_dir_final_develop_density(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    float* dDensityC,
    float* dDensityM,
    float* dDensityY,
    void* cudaStreamOpaque) {
    return launch_focused_spatial_dir_final_develop_density(
        hParams,
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
        cudaStreamOpaque);
}

extern "C" cudaError_t juicer_cuda_negative_pipeline_optics(
    const JuicerCuda::PipelineRunParams* hParams,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    float* dScratchBlurred,
    float* dAux,
    float* dGrainTmp,
    float* dGrainTmpShared,
    float* dGrainTmpMid,
    float* dGrainTmpCoarse,
    const float* dLensBlurKernel,
    int lensBlurRadius,
    const float* dUnsharpKernel,
    int unsharpRadius,
    float unsharpAmount,
    int glareOriginX,
    int glareOriginY,
    std::uint64_t glareSeed,
    float glarePercent,
    float glareRoughness,
    const float* dGlareKernel,
    int glareRadius,
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
    if (!dRgbR || !dRgbG || !dRgbB || !dTmp) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

    dim3 threads2D(32, 8);
    dim3 blocks2D(
        static_cast<unsigned int>((params.width + threads2D.x - 1) / threads2D.x),
        static_cast<unsigned int>((params.height + threads2D.y - 1) / threads2D.y));

    auto blur_plane_in_place = [&](float* plane, float* tmpBuf, const float* k, int radius) -> cudaError_t {
        if (!plane || !tmpBuf || !k || radius <= 0) {
            return cudaSuccess;
        }
        const int kLen = 2 * radius + 1;
        const size_t shmemH = (static_cast<size_t>(kLen) +
                               static_cast<size_t>(threads2D.y) * static_cast<size_t>(threads2D.x + 2 * radius)) *
                              sizeof(float);
        const size_t shmemV = (static_cast<size_t>(kLen) +
                               static_cast<size_t>(threads2D.x) * static_cast<size_t>(threads2D.y + 2 * radius)) *
                              sizeof(float);

        optics_blur_horizontal_kernel<<<blocks2D, threads2D, shmemH, stream>>>(plane, tmpBuf, params.width, params.height, k, radius);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            return e;
        }
        optics_blur_vertical_kernel<<<blocks2D, threads2D, shmemV, stream>>>(tmpBuf, plane, params.width, params.height, k, radius);
        return cudaGetLastError();
    };

    expose_film_raw_kernel<<<blocks2D, threads2D, 0, stream>>>(params, dRgbR, dRgbG, dRgbB);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }

    const JuicerCuda::HighlightBoostPayload& highlightBoost = params.filmExpose.highlightBoost;
    if (highlightBoost.boostEv > 0.0f) {
        const int total = params.width * params.height;
        const int threads1D = 256;
        const int blocks1D = (total + threads1D - 1) / threads1D;
        unsigned int* maxRawBits = reinterpret_cast<unsigned int*>(dTmp);
        err = cudaMemsetAsync(maxRawBits, 0, sizeof(unsigned int), stream);
        if (err != cudaSuccess) {
            return err;
        }
        film_raw_max_kernel<<<blocks1D, threads1D, 0, stream>>>(dRgbR, dRgbG, dRgbB, total, maxRawBits);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }
        highlight_boost_film_raw_kernel<<<blocks1D, threads1D, 0, stream>>>(
            highlightBoost,
            maxRawBits,
            dRgbR,
            dRgbG,
            dRgbB,
            total);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }
    }

    const JuicerCuda::HalationPayload& halation = params.halation;
    const JuicerCuda::HalationKernelPayload& halKernels = params.halationKernels;
    const bool doHalation = (halation.active != 0) &&
                            (halation.primaryAmount[0] > 0.0f || halation.primaryAmount[1] > 0.0f || halation.primaryAmount[2] > 0.0f ||
                             halation.secondaryAmount[0] > 0.0f || halation.secondaryAmount[1] > 0.0f || halation.secondaryAmount[2] > 0.0f);
    if (doHalation) {
        auto apply_halation_pass = [&](float* plane, const float* k, int radius, float strength) -> cudaError_t {
            if (!plane || !k || radius <= 0 || !(strength > 0.0f)) {
                return cudaSuccess;
            }
            const int kLen = 2 * radius + 1;
            const size_t shmemH = (static_cast<size_t>(kLen) +
                                   static_cast<size_t>(threads2D.y) * static_cast<size_t>(threads2D.x + 2 * radius)) *
                                  sizeof(float);
            const size_t shmemV = (static_cast<size_t>(kLen) +
                                   static_cast<size_t>(threads2D.x) * static_cast<size_t>(threads2D.y + 2 * radius)) *
                                  sizeof(float);

            optics_blur_horizontal_kernel<<<blocks2D, threads2D, shmemH, stream>>>(plane, dTmp, params.width, params.height, k, radius);
            cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) {
                return e;
            }
            optics_halation_vertical_apply_kernel<<<blocks2D, threads2D, shmemV, stream>>>(plane, dTmp, params.width, params.height, k, radius, strength);
            return cudaGetLastError();
        };

        err = apply_halation_pass(dRgbR, halKernels.halationKernel[0], halKernels.halationRadius[0], halation.primaryAmount[0]);
        if (err != cudaSuccess)
            return err;
        err = apply_halation_pass(dRgbG, halKernels.halationKernel[1], halKernels.halationRadius[1], halation.primaryAmount[1]);
        if (err != cudaSuccess)
            return err;
        err = apply_halation_pass(dRgbB, halKernels.halationKernel[2], halKernels.halationRadius[2], halation.primaryAmount[2]);
        if (err != cudaSuccess)
            return err;

        err = apply_halation_pass(dRgbR, halKernels.scatteringKernel[0], halKernels.scatteringRadius[0], halation.secondaryAmount[0]);
        if (err != cudaSuccess)
            return err;
        err = apply_halation_pass(dRgbG, halKernels.scatteringKernel[1], halKernels.scatteringRadius[1], halation.secondaryAmount[1]);
        if (err != cudaSuccess)
            return err;
        err = apply_halation_pass(dRgbB, halKernels.scatteringKernel[2], halKernels.scatteringRadius[2], halation.secondaryAmount[2]);
        if (err != cudaSuccess)
            return err;
    }

    develop_film_density_from_raw_kernel<<<blocks2D, threads2D, 0, stream>>>(params, dRgbR, dRgbG, dRgbB, dRgbR, dRgbG, dRgbB);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }

    const JuicerCuda::GrainPayload& grain = params.grain;
    const JuicerCuda::GrainKernelPayload& grainKernels = params.grainKernels;
    const int debugView = grain.debugView;
    if (debugView == 6) {
        const int total = params.width * params.height;
        const int threads1D = 256;
        const int blocks1D = (total + threads1D - 1) / threads1D;

        grain_debug_encode_avg3_kernel<<<blocks1D, threads1D, 0, stream>>>(
            dRgbR,
            dRgbG,
            dRgbB,
            dRgbR,
            dRgbG,
            dRgbB,
            total,
            0.0f,
            grain.debugScale * 4.0f);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }
    } else if (grain.active != 0) {
        constexpr std::uint64_t kSeedSaltFine = 0xA24BAED4963EE407ULL;
        constexpr std::uint64_t kSeedSaltMid = 0x9F6C1E6B2C4D7A13ULL;
        constexpr std::uint64_t kSeedSaltCoarse = 0x3C79AC492BA7B653ULL;
        constexpr std::uint64_t kSeedSaltShared = 0x7E8A1B9D3F2C65A1ULL;

        const int total = params.width * params.height;
        const int threads1D = 256;
        const int blocks1D = (total + threads1D - 1) / threads1D;

        const bool useSublayers = (grain.sublayersActive != 0);
        const bool wantFineBlur = (grainKernels.blurKernel && grainKernels.blurRadius > 0);
        const float wCoarse = std::isfinite(grain.sizeMixWeight)
                                  ? fminf(fmaxf(grain.sizeMixWeight, 0.0f), 1.0f)
                                  : 0.0f;
        const float wMid = std::isfinite(grain.sizeMixWeightMid)
                               ? fminf(fmaxf(grain.sizeMixWeightMid, 0.0f), 1.0f)
                               : 0.0f;
        const float sizeMixScale = (std::isfinite(grain.sizeMixScale) && grain.sizeMixScale > 1.0f)
                                       ? grain.sizeMixScale
                                       : 1.0f;
        const float midScale = (sizeMixScale > 1.0f) ? sqrtf(sizeMixScale) : 1.0f;
        const bool hasMidKernel = (grainKernels.blurKernelMid && grainKernels.blurRadiusMid > 0);
        const bool hasCoarseKernel = (grainKernels.blurKernelCoarse && grainKernels.blurRadiusCoarse > 0);
        const bool wantMix = wantFineBlur &&
                             (sizeMixScale > 1.0f) &&
                             ((wMid > 0.0f && hasMidKernel) || (wCoarse > 0.0f && hasCoarseKernel)) &&
                             dGrainTmp && dGrainTmpMid && dGrainTmpCoarse;

        const bool needFine = (debugView == 0 || debugView == 1 || debugView == 2 || debugView == 4);
        const bool needMid = (debugView == 0 || debugView == 1) && wantMix && (wMid > 0.0f);
        const bool needCoarse = (debugView == 0 || debugView == 1 || debugView == 3 || debugView == 5) && wantMix && (wCoarse > 0.0f);
        const bool needMix = (debugView == 0 || debugView == 1) && wantMix;
        float amplitude = (std::isfinite(grain.amplitude) && grain.amplitude >= 0.0f) ? grain.amplitude : 1.0f;
        const bool applyAmpInMix = wantMix && (debugView == 0 || debugView == 1);
        const float chromaMix = std::isfinite(grain.chromaMix) ? grain.chromaMix : 1.0f;
        const float chromaSharedWeight = std::isfinite(grain.chromaSharedWeight) ? grain.chromaSharedWeight : 0.0f;
        const float chromaIndWeight = std::isfinite(grain.chromaIndWeight) ? grain.chromaIndWeight : 1.0f;
        const bool wantChromaMix = (debugView == 0 || debugView == 1) && (chromaMix < 0.999f);
        const float deltaAmp = (applyAmpInMix || wantChromaMix) ? 1.0f : amplitude;
        const float mixAmplitude = wantChromaMix ? 1.0f : amplitude;

        float* meanBuf = useSublayers ? dAux : (wantFineBlur ? dScratchBlurred : dTmp);
        if (!meanBuf || !dTmp) {
            return cudaErrorInvalidValue;
        }
        if (useSublayers && !dAux) {
            return cudaErrorInvalidValue;
        }
        if (!useSublayers && wantFineBlur && !dScratchBlurred) {
            return cudaErrorInvalidValue;
        }

        const size_t bytes = static_cast<size_t>(total) * sizeof(float);

        auto apply_bias = [&](float* plane, int channel) -> cudaError_t {
            const float bias = -grain.densityMin[channel];
            if (std::isfinite(bias) && bias != 0.0f) {
                grain_add_bias_kernel<<<blocks1D, threads1D, 0, stream>>>(plane, total, bias);
                return cudaGetLastError();
            }
            return cudaSuccess;
        };

        auto generate_channel_simple = [&](const JuicerCuda::PipelineRunParams& passParams, float* plane, int channel) -> cudaError_t {
            grain_apply_simple_kernel<<<blocks2D, threads2D, 0, stream>>>(passParams, plane, channel);
            return cudaGetLastError();
        };

        auto generate_channel_sublayers = [&](const JuicerCuda::PipelineRunParams& passParams, float* mean, float* plane, int channel) -> cudaError_t {
            if (!mean || !plane) {
                return cudaErrorInvalidValue;
            }
            if (!dScratchBlurred) {
                return cudaErrorInvalidValue;
            }

            grain_clear_kernel<<<blocks1D, threads1D, 0, stream>>>(plane, total);
            cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) {
                return e;
            }

            for (int sl = 0; sl < 3; ++sl) {
                grain_layer_kernel<<<blocks2D, threads2D, 0, stream>>>(passParams, mean, dTmp, channel, sl);
                e = cudaGetLastError();
                if (e != cudaSuccess) {
                    return e;
                }

                const float* dyeKernel = grainKernels.dyeKernel[sl][channel];
                const int dyeRadius = grainKernels.dyeRadius[sl][channel];
                if (dyeKernel && dyeRadius > 0) {
                    e = blur_plane_in_place(dTmp, dScratchBlurred, dyeKernel, dyeRadius);
                    if (e != cudaSuccess) {
                        return e;
                    }
                }

                grain_accumulate_kernel<<<blocks1D, threads1D, 0, stream>>>(plane, dTmp, total);
                e = cudaGetLastError();
                if (e != cudaSuccess) {
                    return e;
                }
            }

            return cudaSuccess;
        };

        auto apply_scale_params = [&](JuicerCuda::PipelineRunParams& passParams, float scaleFactor) {
            if (!(scaleFactor > 0.0f) || !std::isfinite(scaleFactor)) {
                return;
            }
            for (int c = 0; c < 3; ++c) {
                passParams.grain.nParticles[c] = params.grain.nParticles[c] / scaleFactor;
                passParams.grain.odParticle[c] = params.grain.odParticle[c] * scaleFactor;
            }
            for (int layer = 0; layer < 3; ++layer) {
                for (int c = 0; c < 3; ++c) {
                    passParams.grain.nParticlesLayers[layer][c] = params.grain.nParticlesLayers[layer][c] / scaleFactor;
                    passParams.grain.odParticleLayers[layer][c] = params.grain.odParticleLayers[layer][c] * scaleFactor;
                }
            }
        };

        auto generate_delta_mix = [&](const JuicerCuda::PipelineRunParams& baseParams,
                                      float* plane,
                                      int channel,
                                      float deltaAmpLocal,
                                      float mixAmplitudeLocal) -> cudaError_t {
            if (!plane) {
                return cudaErrorInvalidValue;
            }

            cudaError_t e = cudaMemcpyAsync(meanBuf, plane, bytes, cudaMemcpyDeviceToDevice, stream);
            if (e != cudaSuccess) {
                return e;
            }

            if (needFine) {
                JuicerCuda::PipelineRunParams fineParams = baseParams;
                fineParams.grain.seedBase ^= kSeedSaltFine;
                fineParams.grain.seedBaseNext ^= kSeedSaltFine;
                fineParams.grain.sizeMixWeight = 0.0f;
                fineParams.grain.sizeMixScale = 1.0f;

                if (useSublayers) {
                    e = generate_channel_sublayers(fineParams, meanBuf, plane, channel);
                } else {
                    e = generate_channel_simple(fineParams, plane, channel);
                }
                if (e != cudaSuccess) {
                    return e;
                }

                e = apply_bias(plane, channel);
                if (e != cudaSuccess) {
                    return e;
                }

                grain_subtract_kernel<<<blocks1D, threads1D, 0, stream>>>(plane, meanBuf, total, deltaAmpLocal);
                e = cudaGetLastError();
                if (e != cudaSuccess) {
                    return e;
                }

                if (wantFineBlur) {
                    e = blur_plane_in_place(plane, dTmp, grainKernels.blurKernel, grainKernels.blurRadius);
                    if (e != cudaSuccess) {
                        return e;
                    }
                }

                if (!needMix) {
                    return cudaSuccess;
                }

                e = cudaMemcpyAsync(dGrainTmp, plane, bytes, cudaMemcpyDeviceToDevice, stream);
                if (e != cudaSuccess) {
                    return e;
                }

                if (!useSublayers) {
                    e = cudaMemcpyAsync(plane, meanBuf, bytes, cudaMemcpyDeviceToDevice, stream);
                    if (e != cudaSuccess) {
                        return e;
                    }
                }
            }

            if (needMid) {
                JuicerCuda::PipelineRunParams midParams = baseParams;
                midParams.grain.seedBase ^= kSeedSaltMid;
                midParams.grain.seedBaseNext ^= kSeedSaltMid;
                midParams.grain.sizeMixWeight = 0.0f;
                midParams.grain.sizeMixWeightMid = 0.0f;
                midParams.grain.sizeMixScale = 1.0f;
                apply_scale_params(midParams, midScale);

                if (useSublayers) {
                    e = generate_channel_sublayers(midParams, meanBuf, plane, channel);
                } else {
                    e = generate_channel_simple(midParams, plane, channel);
                }
                if (e != cudaSuccess) {
                    return e;
                }

                e = apply_bias(plane, channel);
                if (e != cudaSuccess) {
                    return e;
                }

                grain_subtract_kernel<<<blocks1D, threads1D, 0, stream>>>(plane, meanBuf, total, deltaAmpLocal);
                e = cudaGetLastError();
                if (e != cudaSuccess) {
                    return e;
                }

                e = blur_plane_in_place(plane, dTmp, grainKernels.blurKernelMid, grainKernels.blurRadiusMid);
                if (e != cudaSuccess) {
                    return e;
                }

                if (needMix) {
                    e = cudaMemcpyAsync(dGrainTmpMid, plane, bytes, cudaMemcpyDeviceToDevice, stream);
                    if (e != cudaSuccess) {
                        return e;
                    }
                }

                if (!useSublayers) {
                    e = cudaMemcpyAsync(plane, meanBuf, bytes, cudaMemcpyDeviceToDevice, stream);
                    if (e != cudaSuccess) {
                        return e;
                    }
                }
            }

            if (needCoarse) {
                JuicerCuda::PipelineRunParams coarseParams = baseParams;
                coarseParams.grain.seedBase ^= kSeedSaltCoarse;
                coarseParams.grain.seedBaseNext ^= kSeedSaltCoarse;
                coarseParams.grain.sizeMixWeight = 0.0f;
                coarseParams.grain.sizeMixWeightMid = 0.0f;
                coarseParams.grain.sizeMixScale = 1.0f;
                apply_scale_params(coarseParams, sizeMixScale);

                if (useSublayers) {
                    e = generate_channel_sublayers(coarseParams, meanBuf, plane, channel);
                } else {
                    e = generate_channel_simple(coarseParams, plane, channel);
                }
                if (e != cudaSuccess) {
                    return e;
                }

                e = apply_bias(plane, channel);
                if (e != cudaSuccess) {
                    return e;
                }

                grain_subtract_kernel<<<blocks1D, threads1D, 0, stream>>>(plane, meanBuf, total, deltaAmpLocal);
                e = cudaGetLastError();
                if (e != cudaSuccess) {
                    return e;
                }

                e = blur_plane_in_place(plane, dTmp, grainKernels.blurKernelCoarse, grainKernels.blurRadiusCoarse);
                if (e != cudaSuccess) {
                    return e;
                }

                if (needMix) {
                    e = cudaMemcpyAsync(dGrainTmpCoarse, plane, bytes, cudaMemcpyDeviceToDevice, stream);
                    if (e != cudaSuccess) {
                        return e;
                    }
                }
            }

            if (needMix) {
                const float wM = wMid;
                const float wC = wCoarse;
                const float* midPtr = (wM > 0.0f) ? dGrainTmpMid : dGrainTmp;
                const float* coarsePtr = (wC > 0.0f) ? dGrainTmpCoarse : dGrainTmp;
                grain_mix_delta3_kernel<<<blocks1D, threads1D, 0, stream>>>(
                    plane,
                    dGrainTmp,
                    midPtr,
                    coarsePtr,
                    total,
                    wM,
                    wC,
                    grain.sizeMixGain,
                    mixAmplitudeLocal);
                e = cudaGetLastError();
                if (e != cudaSuccess) {
                    return e;
                }
            }

            return cudaSuccess;
        };

        auto process_channel = [&](float* plane, int channel) -> cudaError_t {
            if (!plane) {
                return cudaErrorInvalidValue;
            }

            cudaError_t e = cudaMemcpyAsync(meanBuf, plane, bytes, cudaMemcpyDeviceToDevice, stream);
            if (e != cudaSuccess) {
                return e;
            }

            if (needFine) {
                JuicerCuda::PipelineRunParams fineParams = params;
                fineParams.grain.seedBase ^= kSeedSaltFine;
                fineParams.grain.seedBaseNext ^= kSeedSaltFine;
                fineParams.grain.sizeMixWeight = 0.0f;
                fineParams.grain.sizeMixScale = 1.0f;

                if (useSublayers) {
                    e = generate_channel_sublayers(fineParams, meanBuf, plane, channel);
                } else {
                    e = generate_channel_simple(fineParams, plane, channel);
                }
                if (e != cudaSuccess) {
                    return e;
                }

                e = apply_bias(plane, channel);
                if (e != cudaSuccess) {
                    return e;
                }

                grain_subtract_kernel<<<blocks1D, threads1D, 0, stream>>>(plane, meanBuf, total, deltaAmp);
                e = cudaGetLastError();
                if (e != cudaSuccess) {
                    return e;
                }

                if (debugView != 4 && wantFineBlur) {
                    e = blur_plane_in_place(plane, dTmp, grainKernels.blurKernel, grainKernels.blurRadius);
                    if (e != cudaSuccess) {
                        return e;
                    }
                }

                if (debugView == 0 && !needMid && !needCoarse) {
                    grain_accumulate_kernel<<<blocks1D, threads1D, 0, stream>>>(plane, meanBuf, total);
                    return cudaGetLastError();
                }

                if (!needMid && !needCoarse) {
                    if (debugView == 3 || debugView == 5) {
                        grain_clear_kernel<<<blocks1D, threads1D, 0, stream>>>(plane, total);
                        return cudaGetLastError();
                    }
                    return cudaSuccess;
                }

                if (needMix) {
                    e = cudaMemcpyAsync(dGrainTmp, plane, bytes, cudaMemcpyDeviceToDevice, stream);
                    if (e != cudaSuccess) {
                        return e;
                    }
                }

                if (!useSublayers) {
                    e = cudaMemcpyAsync(plane, meanBuf, bytes, cudaMemcpyDeviceToDevice, stream);
                    if (e != cudaSuccess) {
                        return e;
                    }
                }
            } else if (debugView == 3 || debugView == 5) {
                if (!wantMix) {
                    grain_clear_kernel<<<blocks1D, threads1D, 0, stream>>>(plane, total);
                    return cudaGetLastError();
                }
            }

            if (needMid) {
                JuicerCuda::PipelineRunParams midParams = params;
                midParams.grain.seedBase ^= kSeedSaltMid;
                midParams.grain.seedBaseNext ^= kSeedSaltMid;
                midParams.grain.sizeMixWeight = 0.0f;
                midParams.grain.sizeMixWeightMid = 0.0f;
                midParams.grain.sizeMixScale = 1.0f;
                apply_scale_params(midParams, midScale);

                if (useSublayers) {
                    e = generate_channel_sublayers(midParams, meanBuf, plane, channel);
                } else {
                    e = generate_channel_simple(midParams, plane, channel);
                }
                if (e != cudaSuccess) {
                    return e;
                }

                e = apply_bias(plane, channel);
                if (e != cudaSuccess) {
                    return e;
                }

                grain_subtract_kernel<<<blocks1D, threads1D, 0, stream>>>(plane, meanBuf, total, deltaAmp);
                e = cudaGetLastError();
                if (e != cudaSuccess) {
                    return e;
                }

                e = blur_plane_in_place(plane, dTmp, grainKernels.blurKernelMid, grainKernels.blurRadiusMid);
                if (e != cudaSuccess) {
                    return e;
                }

                if (needMix) {
                    e = cudaMemcpyAsync(dGrainTmpMid, plane, bytes, cudaMemcpyDeviceToDevice, stream);
                    if (e != cudaSuccess) {
                        return e;
                    }
                }

                if (!useSublayers) {
                    e = cudaMemcpyAsync(plane, meanBuf, bytes, cudaMemcpyDeviceToDevice, stream);
                    if (e != cudaSuccess) {
                        return e;
                    }
                }
            }

            if (needCoarse) {
                JuicerCuda::PipelineRunParams coarseParams = params;
                coarseParams.grain.seedBase ^= kSeedSaltCoarse;
                coarseParams.grain.seedBaseNext ^= kSeedSaltCoarse;
                coarseParams.grain.sizeMixWeight = 0.0f;
                coarseParams.grain.sizeMixWeightMid = 0.0f;
                coarseParams.grain.sizeMixScale = 1.0f;
                apply_scale_params(coarseParams, sizeMixScale);

                if (useSublayers) {
                    e = generate_channel_sublayers(coarseParams, meanBuf, plane, channel);
                } else {
                    e = generate_channel_simple(coarseParams, plane, channel);
                }
                if (e != cudaSuccess) {
                    return e;
                }

                e = apply_bias(plane, channel);
                if (e != cudaSuccess) {
                    return e;
                }

                grain_subtract_kernel<<<blocks1D, threads1D, 0, stream>>>(plane, meanBuf, total, deltaAmp);
                e = cudaGetLastError();
                if (e != cudaSuccess) {
                    return e;
                }

                if (debugView != 5) {
                    e = blur_plane_in_place(plane, dTmp, grainKernels.blurKernelCoarse, grainKernels.blurRadiusCoarse);
                    if (e != cudaSuccess) {
                        return e;
                    }
                }

                if (debugView == 3 || debugView == 5) {
                    return cudaSuccess;
                }

                if (needMix) {
                    e = cudaMemcpyAsync(dGrainTmpCoarse, plane, bytes, cudaMemcpyDeviceToDevice, stream);
                    if (e != cudaSuccess) {
                        return e;
                    }
                }
            }

            if (needMix) {
                const float wM = wMid;
                const float wC = wCoarse;
                const float* midPtr = (wM > 0.0f) ? dGrainTmpMid : dGrainTmp;
                const float* coarsePtr = (wC > 0.0f) ? dGrainTmpCoarse : dGrainTmp;
                grain_mix_delta3_kernel<<<blocks1D, threads1D, 0, stream>>>(
                    plane,
                    dGrainTmp,
                    midPtr,
                    coarsePtr,
                    total,
                    wM,
                    wC,
                    grain.sizeMixGain,
                    mixAmplitude);
                e = cudaGetLastError();
                if (e != cudaSuccess) {
                    return e;
                }
            }

            if (debugView == 0) {
                grain_accumulate_kernel<<<blocks1D, threads1D, 0, stream>>>(plane, meanBuf, total);
                return cudaGetLastError();
            }

            return cudaSuccess;
        };

        if (wantChromaMix) {
            const bool needShared = (chromaSharedWeight > 0.0f);
            if (needShared && !dGrainTmpShared) {
                return cudaErrorInvalidValue;
            }
            const int sharedChannel = 1;
            if (needShared) {
                const float* sharedSrc = (sharedChannel == 0) ? dRgbR : (sharedChannel == 1 ? dRgbG : dRgbB);
                err = cudaMemcpyAsync(dGrainTmpShared, sharedSrc, bytes, cudaMemcpyDeviceToDevice, stream);
                if (err != cudaSuccess) {
                    return err;
                }
                JuicerCuda::PipelineRunParams sharedParams = params;
                sharedParams.grain.seedBase ^= kSeedSaltShared;
                sharedParams.grain.seedBaseNext ^= kSeedSaltShared;
                err = generate_delta_mix(sharedParams, dGrainTmpShared, sharedChannel, 1.0f, 1.0f);
                if (err != cudaSuccess) {
                    return err;
                }
            }

            auto process_channel_chroma = [&](float* plane, int channel) -> cudaError_t {
                if (!plane) {
                    return cudaErrorInvalidValue;
                }
                if (chromaIndWeight > 0.0f) {
                    cudaError_t e = generate_delta_mix(params, plane, channel, 1.0f, 1.0f);
                    if (e != cudaSuccess) {
                        return e;
                    }
                } else {
                    cudaError_t e = cudaMemcpyAsync(meanBuf, plane, bytes, cudaMemcpyDeviceToDevice, stream);
                    if (e != cudaSuccess) {
                        return e;
                    }
                }
                grain_mix_shared_kernel<<<blocks1D, threads1D, 0, stream>>>(
                    plane,
                    (chromaIndWeight > 0.0f) ? plane : nullptr,
                    dGrainTmpShared,
                    total,
                    chromaSharedWeight,
                    chromaIndWeight,
                    amplitude);
                cudaError_t e = cudaGetLastError();
                if (e != cudaSuccess) {
                    return e;
                }
                if (debugView == 0) {
                    grain_accumulate_kernel<<<blocks1D, threads1D, 0, stream>>>(plane, meanBuf, total);
                    return cudaGetLastError();
                }
                return cudaSuccess;
            };

            err = process_channel_chroma(dRgbR, 0);
            if (err != cudaSuccess)
                return err;
            err = process_channel_chroma(dRgbG, 1);
            if (err != cudaSuccess)
                return err;
            err = process_channel_chroma(dRgbB, 2);
            if (err != cudaSuccess)
                return err;
        } else {
            err = process_channel(dRgbR, 0);
            if (err != cudaSuccess)
                return err;
            err = process_channel(dRgbG, 1);
            if (err != cudaSuccess)
                return err;
            err = process_channel(dRgbB, 2);
            if (err != cudaSuccess)
                return err;
        }

        if (debugView != 0) {
            grain_debug_encode_avg3_kernel<<<blocks1D, threads1D, 0, stream>>>(
                dRgbR,
                dRgbG,
                dRgbB,
                dRgbR,
                dRgbG,
                dRgbB,
                total,
                0.5f,
                grain.debugScale);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                return err;
            }
        }
    }

    const bool doFilmDefects = (grain.filmDustAmount > 0.0f) || (grain.filmScratchAmount > 0.0f);
    if (grain.debugView == 0 && doFilmDefects) {
        apply_film_defects_kernel<<<blocks2D, threads2D, 0, stream>>>(params, dRgbR, dRgbG, dRgbB);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }
    }

    if (grain.debugView != 0) {
        scan_output_encode_kernel<<<blocks2D, threads2D, 0, stream>>>(params, dRgbR, dRgbG, dRgbB);
        return cudaGetLastError();
    }

    if (params.printExpose.active) {
        develop_print_density_kernel<<<blocks2D, threads2D, 0, stream>>>(params, dRgbR, dRgbG, dRgbB);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }
    }

    const bool doGlare =
        std::isfinite(static_cast<double>(glarePercent)) && (glarePercent > 0.0f) &&
        std::isfinite(static_cast<double>(glareRoughness));
    if (doGlare) {
        if (glareRadius > 0 && dGlareKernel && !dScratchBlurred) {
            return cudaErrorInvalidValue;
        }
        const std::uint64_t mediumId = params.scanStage.scanTables.mediumIsNegative ? 0ULL : 1ULL;
        optics_glare_generate_kernel<<<blocks2D, threads2D, 0, stream>>>(
            dTmp,
            params.width,
            params.height,
            glareSeed,
            mediumId,
            glareOriginX,
            glareOriginY,
            glarePercent,
            glareRoughness);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }
        if (glareRadius > 0 && dGlareKernel) {
            err = blur_plane_in_place(dTmp, dScratchBlurred, dGlareKernel, glareRadius);
            if (err != cudaSuccess)
                return err;
        }
    }

    scan_linear_rgb_kernel<<<blocks2D, threads2D, 0, stream>>>(params, dRgbR, dRgbG, dRgbB, doGlare ? dTmp : nullptr, dRgbR, dRgbG, dRgbB);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }

    if (lensBlurRadius > 0 && dLensBlurKernel) {
        err = blur_plane_in_place(dRgbR, dTmp, dLensBlurKernel, lensBlurRadius);
        if (err != cudaSuccess)
            return err;
        err = blur_plane_in_place(dRgbG, dTmp, dLensBlurKernel, lensBlurRadius);
        if (err != cudaSuccess)
            return err;
        err = blur_plane_in_place(dRgbB, dTmp, dLensBlurKernel, lensBlurRadius);
        if (err != cudaSuccess)
            return err;
    }

    const bool doUnsharp =
        (unsharpRadius > 0) && dUnsharpKernel &&
        std::isfinite(static_cast<double>(unsharpAmount)) && (unsharpAmount != 0.0f);
    if (doUnsharp) {
        auto unsharp_plane_in_place = [&](float* plane) -> cudaError_t {
            const int kLen = 2 * unsharpRadius + 1;
            const size_t shmemH = (static_cast<size_t>(kLen) +
                                   static_cast<size_t>(threads2D.y) * static_cast<size_t>(threads2D.x + 2 * unsharpRadius)) *
                                  sizeof(float);
            const size_t shmemV = (static_cast<size_t>(kLen) +
                                   static_cast<size_t>(threads2D.x) * static_cast<size_t>(threads2D.y + 2 * unsharpRadius)) *
                                  sizeof(float);

            optics_blur_horizontal_kernel<<<blocks2D, threads2D, shmemH, stream>>>(plane, dTmp, params.width, params.height, dUnsharpKernel, unsharpRadius);
            cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) {
                return e;
            }
            optics_unsharp_vertical_combine_kernel<<<blocks2D, threads2D, shmemV, stream>>>(plane, dTmp, params.width, params.height, dUnsharpKernel, unsharpRadius, unsharpAmount);
            return cudaGetLastError();
        };

        err = unsharp_plane_in_place(dRgbR);
        if (err != cudaSuccess)
            return err;
        err = unsharp_plane_in_place(dRgbG);
        if (err != cudaSuccess)
            return err;
        err = unsharp_plane_in_place(dRgbB);
        if (err != cudaSuccess)
            return err;
    }

    scan_output_encode_kernel<<<blocks2D, threads2D, 0, stream>>>(params, dRgbR, dRgbG, dRgbB);
    return cudaGetLastError();
}

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

extern "C" cudaError_t juicer_cuda_print_pipeline_optics(
    const JuicerCuda::PipelineRunParams* hParams,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    float* dScratchBlurred,
    float* dAux,
    float* dGrainTmp,
    float* dGrainTmpShared,
    float* dGrainTmpMid,
    float* dGrainTmpCoarse,
    const float* dLensBlurKernel,
    int lensBlurRadius,
    const float* dUnsharpKernel,
    int unsharpRadius,
    float unsharpAmount,
    int glareOriginX,
    int glareOriginY,
    std::uint64_t glareSeed,
    float glarePercent,
    float glareRoughness,
    const float* dGlareKernel,
    int glareRadius,
    void* cudaStreamOpaque) {
    if (!hParams) {
        return cudaErrorInvalidValue;
    }
    if (!hParams->printExpose.active) {
        return cudaErrorInvalidValue;
    }
    return juicer_cuda_negative_pipeline_optics(
        hParams,
        dRgbR,
        dRgbG,
        dRgbB,
        dTmp,
        dScratchBlurred,
        dAux,
        dGrainTmp,
        dGrainTmpShared,
        dGrainTmpMid,
        dGrainTmpCoarse,
        dLensBlurKernel,
        lensBlurRadius,
        dUnsharpKernel,
        unsharpRadius,
        unsharpAmount,
        glareOriginX,
        glareOriginY,
        glareSeed,
        glarePercent,
        glareRoughness,
        dGlareKernel,
        glareRadius,
        cudaStreamOpaque);
}
