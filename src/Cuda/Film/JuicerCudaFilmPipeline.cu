// Cuda/Film/JuicerCudaFilmPipeline.cu
// Pipeline-aligned CUDA TU for film exposure, development, and spatial DIR kernels.
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "Cuda/JuicerCudaDirProfile.h"
#include "Cuda/JuicerCudaDeviceHelpers.cuh"
#include "openrand/philox.h"

__global__ void expose_film_raw_kernel(
    JuicerCuda::PipelineRunParams params,
    float* outB,
    float* outG,
    float* outR) {
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
    if (params.width <= 0 || params.height <= 0) {
        return;
    }

    const std::size_t pixelBytes = static_cast<std::size_t>(nC) * sizeof(float);
    const std::size_t width = static_cast<std::size_t>(params.width);
    const std::size_t height = static_cast<std::size_t>(params.height);
    const std::size_t yStep = static_cast<std::size_t>(blockDim.y) * static_cast<std::size_t>(gridDim.y);
    const std::size_t xStep = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    for (std::size_t y = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(blockDim.y) + static_cast<std::size_t>(threadIdx.y); y < height; y += yStep) {
        const char* srcRow = reinterpret_cast<const char*>(params.src) + y * params.srcRowBytes;
        for (std::size_t x = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x); x < width; x += xStep) {
            const float* srcPix = reinterpret_cast<const float*>(srcRow + x * pixelBytes);
            if (!srcPix) {
                continue;
            }

            const float rgbIn[3] = {srcPix[0], srcPix[1], srcPix[2]};
            float filmRaw[3] = {0.0f, 0.0f, 0.0f};
            compute_film_raw_device(params, rgbIn, filmRaw);

            const std::size_t idx = y * width + x;
            outB[idx] = filmRaw[0];
            outG[idx] = filmRaw[1];
            outR[idx] = filmRaw[2];
        }
    }
}

// Optics blur kernels are defined in Cuda/Scan/JuicerCudaScannerOptics.cu.
__global__ void optics_blur_horizontal_kernel(
    const float* JUICER_RESTRICT in,
    float* out,
    int width,
    int height,
    const float* JUICER_RESTRICT k,
    int radius);
__global__ void optics_blur_vertical_kernel(
    const float* JUICER_RESTRICT in,
    float* out,
    int width,
    int height,
    const float* JUICER_RESTRICT k,
    int radius);

namespace {

    struct CudaProfileStageTimer {
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        bool active = false;

        ~CudaProfileStageTimer() {
            destroy();
        }

        cudaError_t begin(cudaStream_t stream) {
            if (active) {
                return cudaErrorInvalidValue;
            }
            cudaError_t err = cudaEventCreateWithFlags(&start, cudaEventDefault);
            if (err != cudaSuccess) {
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

        cudaError_t finish(cudaStream_t stream, JuicerCuda::SpatialDirStageProfile* stage) {
            if (!active) {
                return cudaSuccess;
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

    __device__ __forceinline__ int spatial_dir_reflect_index_device(int index, int size) {
        if (size <= 1) {
            return 0;
        }
        const int period = 2 * size;
        index %= period;
        if (index < 0) {
            index += period;
        }
        return index >= size ? period - 1 - index : index;
    }

    __device__ __forceinline__ void compute_dir_corrections_device(
        const JuicerCuda::DirPayload& dir,
        const float layerDensities[3],
        float outLayerCorrections[3]) {
        if (!outLayerCorrections) {
            return;
        }
        if (!dir.active) {
            outLayerCorrections[0] = 0.0f;
            outLayerCorrections[1] = 0.0f;
            outLayerCorrections[2] = 0.0f;
            return;
        }
        auto silver_density = [&](float density, float dmax) -> float {
            return dir.positive ? dmax - density : density;
        };

        const float nB = silver_density(layerDensities[0], dir.dMax[0]);
        const float nG = silver_density(layerDensities[1], dir.dMax[1]);
        const float nR = silver_density(layerDensities[2], dir.dMax[2]);

        float aY = dir.M[0] * nB + dir.M[3] * nG + dir.M[6] * nR;
        float aM = dir.M[1] * nB + dir.M[4] * nG + dir.M[7] * nR;
        float aC = dir.M[2] * nB + dir.M[5] * nG + dir.M[8] * nR;

        outLayerCorrections[0] = aY;
        outLayerCorrections[1] = aM;
        outLayerCorrections[2] = aC;
    }

    struct DirRawCorrectionOutputs {
        float* correctionY = nullptr;
        float* correctionM = nullptr;
        float* correctionC = nullptr;
        float* logRawB = nullptr;
        float* logRawG = nullptr;
        float* logRawR = nullptr;
    };

    template <typename Params>
    __global__ void dir_raw_correction_source_build_kernel(
        Params params,
        DirRawCorrectionOutputs outputs) {
        if (!params.src || params.srcRowBytes == 0) {
            return;
        }
        if (!outputs.correctionY || !outputs.correctionM || !outputs.correctionC) {
            return;
        }
        const bool cacheLogRaw = outputs.logRawB || outputs.logRawG || outputs.logRawR;

        const int nC = params.nComponents;
        if (!(nC == 3 || nC == 4)) {
            return;
        }

        const std::size_t pixelBytes = static_cast<std::size_t>(nC) * sizeof(float);
        const int yStart =
            static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) +
            static_cast<int>(threadIdx.y);
        const int yStep = static_cast<int>(blockDim.y) * static_cast<int>(gridDim.y);
        const int xStart =
            static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
            static_cast<int>(threadIdx.x);
        const int xStep = static_cast<int>(blockDim.x) * static_cast<int>(gridDim.x);
        for (int y = yStart; y < params.height; y += yStep) {
            const char* srcRow = reinterpret_cast<const char*>(params.src) + static_cast<std::size_t>(y) * params.srcRowBytes;
            for (int x = xStart; x < params.width; x += xStep) {
                const float* srcPix = reinterpret_cast<const float*>(srcRow + static_cast<std::size_t>(x) * pixelBytes);
                if (!srcPix) {
                    continue;
                }

                const float rgbIn[3] = {srcPix[0], srcPix[1], srcPix[2]};
                float logE_raw[3] = {0.0f, 0.0f, 0.0f};
                float logE_sanitized[3] = {0.0f, 0.0f, 0.0f};
                float layerPre[3] = {0.0f, 0.0f, 0.0f};
                FilmDevelopIntermediatesDevice intermediates{};
                intermediates.logERaw = logE_raw;
                intermediates.logESanitized = logE_sanitized;
                intermediates.layerPre = layerPre;
                compute_logE_and_layer_pre_device(params, rgbIn, intermediates);

                const float D_cmy[3] = {layerPre[2], layerPre[1], layerPre[0]};
                const float layerDensities[3] = {D_cmy[2], D_cmy[1], D_cmy[0]};

                float outCorr[3] = {0.0f, 0.0f, 0.0f};
                compute_dir_corrections_device(params.filmDevelop.dir, layerDensities, outCorr);

                const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
                outputs.correctionY[idx] = outCorr[0];
                outputs.correctionM[idx] = outCorr[1];
                outputs.correctionC[idx] = outCorr[2];
                if (cacheLogRaw) {
                    if (outputs.logRawB) {
                        outputs.logRawB[idx] = logE_raw[0];
                    }
                    if (outputs.logRawG) {
                        outputs.logRawG[idx] = logE_raw[1];
                    }
                    if (outputs.logRawR) {
                        outputs.logRawR[idx] = logE_raw[2];
                    }
                }
            }
        }
    }

    template <typename Params>
    __global__ void dir_raw_correction_channel_source_build_kernel(
        Params params,
        int correctionChannel,
        float* rawCorrection) {
        if (!params.src || params.srcRowBytes == 0 || !rawCorrection) {
            return;
        }
        if (correctionChannel < 0 || correctionChannel >= 3) {
            return;
        }

        const int nC = params.nComponents;
        if (!(nC == 3 || nC == 4)) {
            return;
        }

        const std::size_t pixelBytes = static_cast<std::size_t>(nC) * sizeof(float);
        const int yStart =
            static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) +
            static_cast<int>(threadIdx.y);
        const int yStep = static_cast<int>(blockDim.y) * static_cast<int>(gridDim.y);
        const int xStart =
            static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
            static_cast<int>(threadIdx.x);
        const int xStep = static_cast<int>(blockDim.x) * static_cast<int>(gridDim.x);
        for (int y = yStart; y < params.height; y += yStep) {
            const char* srcRow =
                reinterpret_cast<const char*>(params.src) +
                static_cast<std::size_t>(y) * params.srcRowBytes;
            for (int x = xStart; x < params.width; x += xStep) {
                const float* srcPix =
                    reinterpret_cast<const float*>(
                        srcRow + static_cast<std::size_t>(x) * pixelBytes);
                if (!srcPix) {
                    continue;
                }

                const float rgbIn[3] = {srcPix[0], srcPix[1], srcPix[2]};
                float logE_raw[3] = {0.0f, 0.0f, 0.0f};
                float logE_sanitized[3] = {0.0f, 0.0f, 0.0f};
                float layerPre[3] = {0.0f, 0.0f, 0.0f};
                FilmDevelopIntermediatesDevice intermediates{};
                intermediates.logERaw = logE_raw;
                intermediates.logESanitized = logE_sanitized;
                intermediates.layerPre = layerPre;
                compute_logE_and_layer_pre_device(params, rgbIn, intermediates);

                const float D_cmy[3] = {layerPre[2], layerPre[1], layerPre[0]};
                const float layerDensities[3] = {D_cmy[2], D_cmy[1], D_cmy[0]};

                float outCorr[3] = {0.0f, 0.0f, 0.0f};
                compute_dir_corrections_device(params.filmDevelop.dir, layerDensities, outCorr);

                const size_t idx =
                    static_cast<size_t>(y) * static_cast<size_t>(params.width) +
                    static_cast<size_t>(x);
                rawCorrection[idx] = outCorr[correctionChannel];
            }
        }
    }

    template <typename Params>
    __global__ void dir_cached_log_raw_build_kernel(
        Params params,
        float* logRawB,
        float* logRawG,
        float* logRawR) {
        if (!params.src || params.srcRowBytes == 0 || !logRawB || !logRawG || !logRawR) {
            return;
        }

        const int nC = params.nComponents;
        if (!(nC == 3 || nC == 4)) {
            return;
        }

        const std::size_t pixelBytes = static_cast<std::size_t>(nC) * sizeof(float);
        const int yStart =
            static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) +
            static_cast<int>(threadIdx.y);
        const int yStep = static_cast<int>(blockDim.y) * static_cast<int>(gridDim.y);
        const int xStart =
            static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
            static_cast<int>(threadIdx.x);
        const int xStep = static_cast<int>(blockDim.x) * static_cast<int>(gridDim.x);
        for (int y = yStart; y < params.height; y += yStep) {
            const char* srcRow = reinterpret_cast<const char*>(params.src) + static_cast<std::size_t>(y) * params.srcRowBytes;
            for (int x = xStart; x < params.width; x += xStep) {
                const float* srcPix = reinterpret_cast<const float*>(srcRow + static_cast<std::size_t>(x) * pixelBytes);
                if (!srcPix) {
                    continue;
                }

                const float rgbIn[3] = {srcPix[0], srcPix[1], srcPix[2]};
                float logE_raw[3] = {0.0f, 0.0f, 0.0f};
                float logE_sanitized[3] = {0.0f, 0.0f, 0.0f};
                float layerPre[3] = {0.0f, 0.0f, 0.0f};
                FilmDevelopIntermediatesDevice intermediates{};
                intermediates.logERaw = logE_raw;
                intermediates.logESanitized = logE_sanitized;
                intermediates.layerPre = layerPre;
                compute_logE_and_layer_pre_device(params, rgbIn, intermediates);

                const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
                logRawB[idx] = logE_raw[0];
                logRawG[idx] = logE_raw[1];
                logRawR[idx] = logE_raw[2];
            }
        }
    }

    __global__ void spatial_dir_blur_horizontal_reflect_kernel(
        const float* JUICER_RESTRICT in,
        float* out,
        int width,
        int height,
        const float* JUICER_RESTRICT kernel,
        int radius) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        if (x >= width || y >= height || !in || !out || !kernel || radius <= 0) {
            return;
        }
        double sum = 0.0;
        for (int offset = -radius; offset <= radius; ++offset) {
            const int sx = spatial_dir_reflect_index_device(x + offset, width);
            sum += static_cast<double>(
                       in[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(sx)]) *
                   static_cast<double>(kernel[offset + radius]);
        }
        out[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] =
            isfinite(sum) ? static_cast<float>(sum) : 0.0f;
    }

    __global__ void spatial_dir_blur_vertical_reflect_accumulate_kernel(
        const float* JUICER_RESTRICT in,
        float* inOut,
        int width,
        int height,
        const float* JUICER_RESTRICT kernel,
        int radius, // NOLINT(bugprone-easily-swappable-parameters)
        float weight,
        int initialize) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        if (x >= width || y >= height || !in || !inOut || !kernel || radius <= 0) {
            return;
        }
        double sum = 0.0;
        for (int offset = -radius; offset <= radius; ++offset) {
            const int sy = spatial_dir_reflect_index_device(y + offset, height);
            sum += static_cast<double>(
                       in[static_cast<size_t>(sy) * static_cast<size_t>(width) + static_cast<size_t>(x)]) *
                   static_cast<double>(kernel[offset + radius]);
        }
        const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
        const float blurred = isfinite(sum) ? static_cast<float>(sum) : 0.0f;
        const float weighted = blurred * weight;
        inOut[idx] = initialize ? weighted : inOut[idx] + weighted;
    }

    __device__ __forceinline__ const float* spatial_dir_select_input_channel_device(
        const float* p0,
        const float* p1,
        const float* p2,
        int channel) {
        return channel == 0 ? p0 : (channel == 1 ? p1 : p2);
    }

    __device__ __forceinline__ float* spatial_dir_select_output_channel_device(
        float* p0,
        float* p1,
        float* p2,
        int channel) {
        return channel == 0 ? p0 : (channel == 1 ? p1 : p2);
    }

    // Channel-fused YVV kernels intentionally group Y/M/C plane pointers and YVV coefficients.
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    __global__ void spatial_dir_iir_horizontal_channels_kernel(
        const float* inputY,
        const float* inputM,
        const float* inputC,
        float* outputY,
        float* outputM,
        float* outputC,
        int width,
        int height,
        double B,
        double B1,
        double B2,
        double B3) {
        const int y = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int channel = static_cast<int>(blockIdx.y);
        if (y >= height || channel >= 3 || width <= 0) {
            return;
        }
        const float* input = spatial_dir_select_input_channel_device(inputY, inputM, inputC, channel);
        float* output = spatial_dir_select_output_channel_device(outputY, outputM, outputC, channel);
        if (!input || !output) {
            return;
        }
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(width);
        double w1 = static_cast<double>(input[row]);
        double w2 = w1;
        double w3 = w1;
        for (int x = 0; x < width; ++x) {
            const size_t index = row + static_cast<size_t>(x);
            const double w = B * static_cast<double>(input[index]) + B1 * w1 + B2 * w2 + B3 * w3;
            output[index] = static_cast<float>(w);
            w3 = w2;
            w2 = w1;
            w1 = w;
        }
        double y1 = static_cast<double>(output[row + static_cast<size_t>(width - 1)]);
        double y2 = y1;
        double y3 = y1;
        for (int x = width - 1; x >= 0; --x) {
            const size_t index = row + static_cast<size_t>(x);
            const double value = B * static_cast<double>(output[index]) + B1 * y1 + B2 * y2 + B3 * y3;
            output[index] = static_cast<float>(value);
            y3 = y2;
            y2 = y1;
            y1 = value;
        }
    }

    __global__ void spatial_dir_iir_vertical_accumulate_channels_kernel(
        const float* inputY,
        const float* inputM,
        const float* inputC,
        float* forwardTempY,
        float* forwardTempM,
        float* forwardTempC,
        float* inOutY,
        float* inOutM,
        float* inOutC,
        int width,
        int height,
        double B,
        double B1,
        double B2,
        double B3,
        float weight,
        int initialize) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int channel = static_cast<int>(blockIdx.y);
        if (x >= width || channel >= 3 || height <= 0) {
            return;
        }
        const float* input = spatial_dir_select_input_channel_device(inputY, inputM, inputC, channel);
        float* forwardTemp = spatial_dir_select_output_channel_device(
            forwardTempY,
            forwardTempM,
            forwardTempC,
            channel);
        float* inOut = spatial_dir_select_output_channel_device(inOutY, inOutM, inOutC, channel);
        if (!input || !forwardTemp || !inOut) {
            return;
        }
        const size_t visualTop =
            static_cast<size_t>(height - 1) * static_cast<size_t>(width) + static_cast<size_t>(x);
        double w1 = static_cast<double>(input[visualTop]);
        double w2 = w1;
        double w3 = w1;
        for (int y = height - 1; y >= 0; --y) {
            const size_t index = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            const double w = B * static_cast<double>(input[index]) + B1 * w1 + B2 * w2 + B3 * w3;
            forwardTemp[index] = static_cast<float>(w);
            w3 = w2;
            w2 = w1;
            w1 = w;
        }
        double y1 = static_cast<double>(forwardTemp[x]);
        double y2 = y1;
        double y3 = y1;
        for (int y = 0; y < height; ++y) {
            const size_t index = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            const double value = B * static_cast<double>(forwardTemp[index]) + B1 * y1 + B2 * y2 + B3 * y3;
            const float blurred = isfinite(value) ? static_cast<float>(value) : 0.0f;
            const float weighted = blurred * weight;
            inOut[index] = initialize ? weighted : inOut[index] + weighted;
            y3 = y2;
            y2 = y1;
            y1 = value;
        }
    }

    __global__ void spatial_dir_iir_horizontal_single_kernel(
        const float* input,
        float* output,
        int width,
        int height,
        double B,
        double B1,
        double B2,
        double B3) {
        const int y = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        if (y >= height || width <= 0 || !input || !output) {
            return;
        }
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(width);
        double w1 = static_cast<double>(input[row]);
        double w2 = w1;
        double w3 = w1;
        for (int x = 0; x < width; ++x) {
            const size_t index = row + static_cast<size_t>(x);
            const double w = B * static_cast<double>(input[index]) + B1 * w1 + B2 * w2 + B3 * w3;
            output[index] = static_cast<float>(w);
            w3 = w2;
            w2 = w1;
            w1 = w;
        }
        double y1 = static_cast<double>(output[row + static_cast<size_t>(width - 1)]);
        double y2 = y1;
        double y3 = y1;
        for (int x = width - 1; x >= 0; --x) {
            const size_t index = row + static_cast<size_t>(x);
            const double value = B * static_cast<double>(output[index]) + B1 * y1 + B2 * y2 + B3 * y3;
            output[index] = static_cast<float>(value);
            y3 = y2;
            y2 = y1;
            y1 = value;
        }
    }

    __global__ void spatial_dir_iir_vertical_accumulate_single_kernel(
        const float* input,
        float* forwardTemp,
        float* inOut,
        int width,
        int height,
        double B,
        double B1,
        double B2,
        double B3,
        float weight,
        int initialize) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        if (x >= width || height <= 0 || !input || !forwardTemp || !inOut) {
            return;
        }
        const size_t visualTop =
            static_cast<size_t>(height - 1) * static_cast<size_t>(width) + static_cast<size_t>(x);
        double w1 = static_cast<double>(input[visualTop]);
        double w2 = w1;
        double w3 = w1;
        for (int y = height - 1; y >= 0; --y) {
            const size_t index = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            const double w = B * static_cast<double>(input[index]) + B1 * w1 + B2 * w2 + B3 * w3;
            forwardTemp[index] = static_cast<float>(w);
            w3 = w2;
            w2 = w1;
            w1 = w;
        }
        double y1 = static_cast<double>(forwardTemp[static_cast<size_t>(x)]);
        double y2 = y1;
        double y3 = y1;
        for (int y = 0; y < height; ++y) {
            const size_t index = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            const double value =
                B * static_cast<double>(forwardTemp[index]) + B1 * y1 + B2 * y2 + B3 * y3;
            const float blurred = isfinite(value) ? static_cast<float>(value) : 0.0f;
            const float weighted = blurred * weight;
            inOut[index] = initialize ? weighted : inOut[index] + weighted;
            y3 = y2;
            y2 = y1;
            y1 = value;
        }
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

} // namespace

template <typename Params>
cudaError_t build_spatial_dir_impl(
    const Params& params,
    JuicerCuda::SpatialDirBuildRequest request) {
    float* rawCorrectionY = request.planes.rawCorrectionY;
    float* rawCorrectionM = request.planes.rawCorrectionM;
    float* rawCorrectionC = request.planes.rawCorrectionC;
    float* filteredCorrectionY = request.planes.filteredCorrectionY;
    float* filteredCorrectionM = request.planes.filteredCorrectionM;
    float* filteredCorrectionC = request.planes.filteredCorrectionC;
    float* filterTemp = request.planes.filterTemp;
    float* filterTempM = request.planes.filterTempM;
    float* filterTempC = request.planes.filterTempC;
    float* logRawB = request.planes.logRawB;
    float* logRawG = request.planes.logRawG;
    float* logRawR = request.planes.logRawR;
    const float* dGaussianKernel = request.gaussian.kernel;
    const int gaussianRadius = request.gaussian.radius;
    const float gaussianSigma = request.gaussian.sigma;
    const float gaussianWeight = request.gaussian.weight;
    const float* dTailKernel0 = request.tails[0].kernel;
    const int tailRadius0 = request.tails[0].radius;
    const float tailSigma0 = request.tails[0].sigma;
    const float tailWeight0 = request.tails[0].weight;
    const float* dTailKernel1 = request.tails[1].kernel;
    const int tailRadius1 = request.tails[1].radius;
    const float tailSigma1 = request.tails[1].sigma;
    const float tailWeight1 = request.tails[1].weight;
    const float* dTailKernel2 = request.tails[2].kernel;
    const int tailRadius2 = request.tails[2].radius;
    const float tailSigma2 = request.tails[2].sigma;
    const float tailWeight2 = request.tails[2].weight;
    void* cudaStreamOpaque = request.streamOpaque;
    JuicerCuda::SpatialDirBuildProfile* profile = request.profile;
    if (!params.src || params.srcRowBytes == 0) {
        return cudaErrorInvalidValue;
    }
    if (params.width <= 0 || params.height <= 0) {
        return cudaSuccess;
    }
    if (!(params.nComponents == 3 || params.nComponents == 4)) {
        return cudaErrorInvalidValue;
    }
    const bool haveComponentStreamedYvvScratch =
        rawCorrectionY && !rawCorrectionM && !rawCorrectionC &&
        filteredCorrectionY && filteredCorrectionM && filteredCorrectionC && filterTemp &&
        !filterTempM && !filterTempC;
    const bool haveComponentStreamedScratch = haveComponentStreamedYvvScratch;
    if (!rawCorrectionY || (!haveComponentStreamedScratch && (!rawCorrectionM || !rawCorrectionC)) ||
        !filteredCorrectionY || !filteredCorrectionM || !filteredCorrectionC ||
        !filterTemp ||
        !(gaussianSigma > 0.0f) || !(gaussianWeight >= 0.0f)) {
        return cudaErrorInvalidValue;
    }
    const bool haveAliasedForwardYvvScratch =
        filterTempM && filterTempC;
    const bool haveLowScratchPairYvvScratch =
        filterTempM && !filterTempC;
    const bool haveSingleTempSequentialYvvScratch =
        rawCorrectionY && rawCorrectionM && rawCorrectionC &&
        filteredCorrectionY && filteredCorrectionM && filteredCorrectionC &&
        filterTemp && !filterTempM && !filterTempC;

    cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
    cudaError_t err = cudaSuccess;
    CudaProfileStageTimer totalTimer;
    if (profile) {
        *profile = JuicerCuda::SpatialDirBuildProfile{};
        profile->width = params.width;
        profile->height = params.height;
        profile->gaussianRadius = gaussianRadius;
        profile->gaussianSigma = gaussianSigma;
        profile->gaussianWeight = gaussianWeight;
        profile->tailRadius[0] = tailRadius0;
        profile->tailRadius[1] = tailRadius1;
        profile->tailRadius[2] = tailRadius2;
        profile->tailSigma[0] = tailSigma0;
        profile->tailSigma[1] = tailSigma1;
        profile->tailSigma[2] = tailSigma2;
        profile->tailWeight[0] = tailWeight0;
        profile->tailWeight[1] = tailWeight1;
        profile->tailWeight[2] = tailWeight2;
        err = totalTimer.begin(stream);
        if (err != cudaSuccess) {
            return err;
        }
    }

    auto finish_profile = [&](cudaError_t result) -> cudaError_t {
        cudaError_t finalErr = result;
        if (profile) {
            const cudaError_t totalErr = totalTimer.finish(stream, &profile->total);
            profile->total.launches = profile->totalLaunches;
            if (finalErr == cudaSuccess && totalErr != cudaSuccess) {
                finalErr = totalErr;
            }
        }
        return finalErr;
    };

    auto mark_launch = [&](JuicerCuda::SpatialDirStageProfile* stage, int* counter) {
        if (!profile) {
            return;
        }
        if (stage) {
            ++stage->launches;
        }
        if (counter) {
            ++(*counter);
        }
        ++profile->totalLaunches;
    };

    dim3 threads2D(32, 8);
    dim3 blocks2D(
        static_cast<unsigned int>((params.width + threads2D.x - 1) / threads2D.x),
        static_cast<unsigned int>((params.height + threads2D.y - 1) / threads2D.y));

    auto launch_corrections = [&]() -> cudaError_t {
        CudaProfileStageTimer timer;
        if (profile) {
            cudaError_t e = timer.begin(stream);
            if (e != cudaSuccess) {
                return e;
            }
        }
        DirRawCorrectionOutputs outputs{};
        outputs.correctionY = rawCorrectionY;
        outputs.correctionM = rawCorrectionM;
        outputs.correctionC = rawCorrectionC;
        outputs.logRawB = logRawB;
        outputs.logRawG = logRawG;
        outputs.logRawR = logRawR;
        dir_raw_correction_source_build_kernel<<<blocks2D, threads2D, 0, stream>>>(
            params,
            outputs);
        mark_launch(
            profile ? &profile->correction : nullptr,
            profile ? &profile->correctionLaunches : nullptr);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            return e;
        }
        return profile ? timer.finish(stream, &profile->correction) : cudaSuccess;
    };

    auto launch_streamed_channel_corrections = [&](int channel) -> cudaError_t {
        CudaProfileStageTimer timer;
        if (profile) {
            cudaError_t e = timer.begin(stream);
            if (e != cudaSuccess) {
                return e;
            }
        }
        dir_raw_correction_channel_source_build_kernel<<<blocks2D, threads2D, 0, stream>>>(
            params,
            channel,
            rawCorrectionY);
        mark_launch(
            profile ? &profile->correction : nullptr,
            profile ? &profile->correctionLaunches : nullptr);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            return e;
        }
        return profile ? timer.finish(stream, &profile->correction) : cudaSuccess;
    };

    if (!haveComponentStreamedScratch) {
        err = launch_corrections();
        if (err != cudaSuccess) {
            return finish_profile(err);
        }
    }

    auto accumulate_fir_plane = [&](const float* rawCorrection, float* filteredCorrection, const float* k, int r, float sigma, // NOLINT(bugprone-easily-swappable-parameters)
                                    float weight,
                                    bool initialize,
                                    JuicerCuda::SpatialDirStageProfile* stage,
                                    int* launchCounter)
        -> cudaError_t {
        if (!rawCorrection || !filteredCorrection || !(sigma > 0.0f) ||
            !(sigma < 3.0f) || !(weight >= 0.0f)) {
            return cudaErrorInvalidValue;
        }
        CudaProfileStageTimer timer;
        if (profile) {
            cudaError_t e = timer.begin(stream);
            if (e != cudaSuccess) {
                return e;
            }
        }
        if (!k || r <= 0) {
            return cudaErrorInvalidValue;
        }
        spatial_dir_blur_horizontal_reflect_kernel<<<blocks2D, threads2D, 0, stream>>>(
            rawCorrection, filterTemp, params.width, params.height, k, r);
        mark_launch(stage, launchCounter);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            return e;
        }
        spatial_dir_blur_vertical_reflect_accumulate_kernel<<<blocks2D, threads2D, 0, stream>>>(
            filterTemp,
            filteredCorrection,
            params.width,
            params.height,
            k,
            r,
            weight,
            initialize ? 1 : 0);
        mark_launch(stage, launchCounter);
        e = cudaGetLastError();
        if (e != cudaSuccess) {
            return e;
        }
        return profile ? timer.finish(stream, stage) : cudaSuccess;
    };

    auto accumulate_yvv_channels = [&](const float* raw0, const float* raw1, const float* raw2, float* filtered0, float* filtered1, float* filtered2, float sigma, float weight, bool initialize, JuicerCuda::SpatialDirStageProfile* stage, int* launchCounter) -> cudaError_t {
        if (!raw0 || !raw1 || !raw2 || !filtered0 || !filtered1 || !filtered2 ||
            !(sigma > 0.0f) || !(weight >= 0.0f) ||
            !haveAliasedForwardYvvScratch) {
            return cudaErrorInvalidValue;
        }
        CudaProfileStageTimer timer;
        if (profile) {
            cudaError_t e = timer.begin(stream);
            if (e != cudaSuccess) {
                return e;
            }
        }
        const double q = 0.98711 * static_cast<double>(sigma) - 0.96330;
        const double q2 = q * q;
        const double q3 = q2 * q;
        const double b0 = 1.57825 + 2.44413 * q + 1.4281 * q2 + 0.422205 * q3;
        const double b1 = 2.44413 * q + 2.85619 * q2 + 1.26661 * q3;
        const double b2 = -(1.4281 * q2 + 1.26661 * q3);
        const double b3 = 0.422205 * q3;
        const double B1 = b1 / b0;
        const double B2 = b2 / b0;
        const double B3 = b3 / b0;
        const double B = 1.0 - (b1 + b2 + b3) / b0;
        const int threads = 128;
        spatial_dir_iir_horizontal_channels_kernel<<<
            dim3(static_cast<unsigned int>((params.height + threads - 1) / threads), 3),
            threads,
            0,
            stream>>>(
            raw0,
            raw1,
            raw2,
            filterTemp,
            filterTempM,
            filterTempC,
            params.width,
            params.height,
            B,
            B1,
            B2,
            B3);
        mark_launch(stage, launchCounter);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            return e;
        }
        spatial_dir_iir_vertical_accumulate_channels_kernel<<<
            dim3(static_cast<unsigned int>((params.width + threads - 1) / threads), 3),
            threads,
            0,
            stream>>>(
            filterTemp,
            filterTempM,
            filterTempC,
            filterTemp,
            filterTempM,
            filterTempC,
            filtered0,
            filtered1,
            filtered2,
            params.width,
            params.height,
            B,
            B1,
            B2,
            B3,
            weight,
            initialize ? 1 : 0);
        mark_launch(stage, launchCounter);
        e = cudaGetLastError();
        if (e != cudaSuccess) {
            return e;
        }
        return profile ? timer.finish(stream, stage) : cudaSuccess;
    };

    const float* rawCorrections[3] = {rawCorrectionY, rawCorrectionM, rawCorrectionC};
    float* filteredCorrections[3] = {filteredCorrectionY, filteredCorrectionM, filteredCorrectionC};
    auto accumulate_yvv_low_scratch_pair = [&](float sigma, float weight, bool initialize, JuicerCuda::SpatialDirStageProfile* stage, int* launchCounter) -> cudaError_t {
        if (!rawCorrectionY || !rawCorrectionM || !rawCorrectionC ||
            !filteredCorrectionY || !filteredCorrectionM || !filteredCorrectionC ||
            !(sigma > 0.0f) || !(weight >= 0.0f) || !haveLowScratchPairYvvScratch) {
            return cudaErrorInvalidValue;
        }
        CudaProfileStageTimer timer;
        if (profile) {
            cudaError_t e = timer.begin(stream);
            if (e != cudaSuccess) {
                return e;
            }
        }
        const double q = 0.98711 * static_cast<double>(sigma) - 0.96330;
        const double q2 = q * q;
        const double q3 = q2 * q;
        const double b0 = 1.57825 + 2.44413 * q + 1.4281 * q2 + 0.422205 * q3;
        const double b1 = 2.44413 * q + 2.85619 * q2 + 1.26661 * q3;
        const double b2 = -(1.4281 * q2 + 1.26661 * q3);
        const double b3 = 0.422205 * q3;
        const double B1 = b1 / b0;
        const double B2 = b2 / b0;
        const double B3 = b3 / b0;
        const double B = 1.0 - (b1 + b2 + b3) / b0;
        const int threads = 128;
        const dim3 horizontalPairBlocks(
            static_cast<unsigned int>((params.height + threads - 1) / threads),
            2);
        const dim3 verticalPairBlocks(
            static_cast<unsigned int>((params.width + threads - 1) / threads),
            2);
        spatial_dir_iir_horizontal_channels_kernel<<<
            horizontalPairBlocks,
            threads,
            0,
            stream>>>(
            rawCorrectionY,
            rawCorrectionM,
            nullptr,
            filterTemp,
            filterTempM,
            nullptr,
            params.width,
            params.height,
            B,
            B1,
            B2,
            B3);
        mark_launch(stage, launchCounter);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            return e;
        }
        spatial_dir_iir_vertical_accumulate_channels_kernel<<<
            verticalPairBlocks,
            threads,
            0,
            stream>>>(
            filterTemp,
            filterTempM,
            nullptr,
            filterTemp,
            filterTempM,
            nullptr,
            filteredCorrectionY,
            filteredCorrectionM,
            nullptr,
            params.width,
            params.height,
            B,
            B1,
            B2,
            B3,
            weight,
            initialize ? 1 : 0);
        mark_launch(stage, launchCounter);
        e = cudaGetLastError();
        if (e != cudaSuccess) {
            return e;
        }

        const dim3 horizontalSingleBlocks(
            static_cast<unsigned int>((params.height + threads - 1) / threads));
        const dim3 verticalSingleBlocks(
            static_cast<unsigned int>((params.width + threads - 1) / threads));
        spatial_dir_iir_horizontal_single_kernel<<<
            horizontalSingleBlocks,
            threads,
            0,
            stream>>>(
            rawCorrectionC,
            filterTemp,
            params.width,
            params.height,
            B,
            B1,
            B2,
            B3);
        mark_launch(stage, launchCounter);
        e = cudaGetLastError();
        if (e != cudaSuccess) {
            return e;
        }
        spatial_dir_iir_vertical_accumulate_single_kernel<<<
            verticalSingleBlocks,
            threads,
            0,
            stream>>>(
            filterTemp,
            filterTemp,
            filteredCorrectionC,
            params.width,
            params.height,
            B,
            B1,
            B2,
            B3,
            weight,
            initialize ? 1 : 0);
        mark_launch(stage, launchCounter);
        e = cudaGetLastError();
        if (e != cudaSuccess) {
            return e;
        }
        return profile ? timer.finish(stream, stage) : cudaSuccess;
    };

    auto accumulate_yvv_sequential_single_channel = [&](float sigma, float weight, bool initialize, JuicerCuda::SpatialDirStageProfile* stage, int* launchCounter) -> cudaError_t {
        if (!(sigma > 0.0f) || !(weight >= 0.0f) ||
            !haveSingleTempSequentialYvvScratch) {
            return cudaErrorInvalidValue;
        }
        CudaProfileStageTimer timer;
        if (profile) {
            cudaError_t e = timer.begin(stream);
            if (e != cudaSuccess) {
                return e;
            }
        }
        const double q = 0.98711 * static_cast<double>(sigma) - 0.96330;
        const double q2 = q * q;
        const double q3 = q2 * q;
        const double b0 = 1.57825 + 2.44413 * q + 1.4281 * q2 + 0.422205 * q3;
        const double b1 = 2.44413 * q + 2.85619 * q2 + 1.26661 * q3;
        const double b2 = -(1.4281 * q2 + 1.26661 * q3);
        const double b3 = 0.422205 * q3;
        const double B1 = b1 / b0;
        const double B2 = b2 / b0;
        const double B3 = b3 / b0;
        const double B = 1.0 - (b1 + b2 + b3) / b0;
        const int threads = 128;
        const dim3 horizontalBlocks(
            static_cast<unsigned int>((params.height + threads - 1) / threads));
        const dim3 verticalBlocks(
            static_cast<unsigned int>((params.width + threads - 1) / threads));
        for (int channel = 0; channel < 3; ++channel) {
            spatial_dir_iir_horizontal_single_kernel<<<
                horizontalBlocks,
                threads,
                0,
                stream>>>(
                rawCorrections[channel],
                filterTemp,
                params.width,
                params.height,
                B,
                B1,
                B2,
                B3);
            mark_launch(stage, launchCounter);
            cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) {
                return e;
            }
            spatial_dir_iir_vertical_accumulate_single_kernel<<<
                verticalBlocks,
                threads,
                0,
                stream>>>(
                filterTemp,
                filterTemp,
                filteredCorrections[channel],
                params.width,
                params.height,
                B,
                B1,
                B2,
                B3,
                weight,
                initialize ? 1 : 0);
            mark_launch(stage, launchCounter);
            e = cudaGetLastError();
            if (e != cudaSuccess) {
                return e;
            }
        }
        return profile ? timer.finish(stream, stage) : cudaSuccess;
    };

    auto accumulate_yvv_component_streamed = [&](float* filteredCorrection, float sigma, float weight, bool initialize, JuicerCuda::SpatialDirStageProfile* stage, int* launchCounter) -> cudaError_t {
        if (!filteredCorrection || !(sigma > 0.0f) || !(weight >= 0.0f) ||
            !haveComponentStreamedYvvScratch) {
            return cudaErrorInvalidValue;
        }
        CudaProfileStageTimer timer;
        if (profile) {
            cudaError_t e = timer.begin(stream);
            if (e != cudaSuccess) {
                return e;
            }
        }
        const double q = 0.98711 * static_cast<double>(sigma) - 0.96330;
        const double q2 = q * q;
        const double q3 = q2 * q;
        const double b0 = 1.57825 + 2.44413 * q + 1.4281 * q2 + 0.422205 * q3;
        const double b1 = 2.44413 * q + 2.85619 * q2 + 1.26661 * q3;
        const double b2 = -(1.4281 * q2 + 1.26661 * q3);
        const double b3 = 0.422205 * q3;
        const double B1 = b1 / b0;
        const double B2 = b2 / b0;
        const double B3 = b3 / b0;
        const double B = 1.0 - (b1 + b2 + b3) / b0;
        const int threads = 128;
        const dim3 horizontalBlocks(
            static_cast<unsigned int>((params.height + threads - 1) / threads));
        const dim3 verticalBlocks(
            static_cast<unsigned int>((params.width + threads - 1) / threads));
        spatial_dir_iir_horizontal_single_kernel<<<
            horizontalBlocks,
            threads,
            0,
            stream>>>(
            rawCorrectionY,
            filterTemp,
            params.width,
            params.height,
            B,
            B1,
            B2,
            B3);
        mark_launch(stage, launchCounter);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            return e;
        }
        spatial_dir_iir_vertical_accumulate_single_kernel<<<
            verticalBlocks,
            threads,
            0,
            stream>>>(
            filterTemp,
            filterTemp,
            filteredCorrection,
            params.width,
            params.height,
            B,
            B1,
            B2,
            B3,
            weight,
            initialize ? 1 : 0);
        mark_launch(stage, launchCounter);
        e = cudaGetLastError();
        if (e != cudaSuccess) {
            return e;
        }
        return profile ? timer.finish(stream, stage) : cudaSuccess;
    };

    auto accumulate_component = [&](const float* k, int r, float sigma, float weight, bool initialize, JuicerCuda::SpatialDirStageProfile* stage, int* launchCounter)
        -> cudaError_t {
        if (sigma >= 3.0f) {
            if (haveAliasedForwardYvvScratch) {
                return accumulate_yvv_channels(
                    rawCorrectionY,
                    rawCorrectionM,
                    rawCorrectionC,
                    filteredCorrectionY,
                    filteredCorrectionM,
                    filteredCorrectionC,
                    sigma,
                    weight,
                    initialize,
                    stage,
                    launchCounter);
            }
            if (haveLowScratchPairYvvScratch) {
                return accumulate_yvv_low_scratch_pair(
                    sigma,
                    weight,
                    initialize,
                    stage,
                    launchCounter);
            }
            return accumulate_yvv_sequential_single_channel(
                sigma,
                weight,
                initialize,
                stage,
                launchCounter);
        }
        for (int channel = 0; channel < 3; ++channel) {
            cudaError_t componentErr = accumulate_fir_plane(
                rawCorrections[channel],
                filteredCorrections[channel],
                k,
                r,
                sigma,
                weight,
                initialize,
                stage,
                launchCounter);
            if (componentErr != cudaSuccess) {
                return componentErr;
            }
        }
        return cudaSuccess;
    };

    auto accumulate_streamed_component = [&](int channel, const float* k, int r, float sigma, float weight, bool initialize, JuicerCuda::SpatialDirStageProfile* stage, int* launchCounter)
        -> cudaError_t {
        if (channel < 0 || channel >= 3 || !(weight >= 0.0f)) {
            return cudaErrorInvalidValue;
        }
        float* filteredCorrection = filteredCorrections[static_cast<std::size_t>(channel)];
        if (sigma >= 3.0f) {
            return accumulate_yvv_component_streamed(
                filteredCorrection,
                sigma,
                weight,
                initialize,
                stage,
                launchCounter);
        }
        return accumulate_fir_plane(
            rawCorrectionY,
            filteredCorrection,
            k,
            r,
            sigma,
            weight,
            initialize,
            stage,
            launchCounter);
    };

    if (haveComponentStreamedYvvScratch) {
        for (int channel = 0; channel < 3; ++channel) {
            err = launch_streamed_channel_corrections(channel);
            if (err != cudaSuccess) {
                return finish_profile(err);
            }
            bool channelAccumulatorInitialized = false;
            if (gaussianWeight > 0.0f) {
                err = accumulate_streamed_component(
                    channel,
                    dGaussianKernel,
                    gaussianRadius,
                    gaussianSigma,
                    gaussianWeight,
                    true,
                    profile ? &profile->baseFilter : nullptr,
                    profile ? &profile->baseFilterLaunches : nullptr);
                if (err != cudaSuccess) {
                    return finish_profile(err);
                }
                channelAccumulatorInitialized = true;
            }
            const float* streamedTailKernels[3] = {dTailKernel0, dTailKernel1, dTailKernel2};
            const int streamedTailRadii[3] = {tailRadius0, tailRadius1, tailRadius2};
            const float streamedTailSigmas[3] = {tailSigma0, tailSigma1, tailSigma2};
            const float streamedTailWeights[3] = {tailWeight0, tailWeight1, tailWeight2};
            for (int component = 0; component < 3; ++component) {
                if (!(streamedTailWeights[component] > 0.0f)) {
                    continue;
                }
                if (!(streamedTailSigmas[component] > 0.0f) ||
                    (streamedTailSigmas[component] < 3.0f &&
                     (!streamedTailKernels[component] || streamedTailRadii[component] <= 0))) {
                    return finish_profile(cudaErrorInvalidValue);
                }
                const bool initializeComponent = !channelAccumulatorInitialized;
                err = accumulate_streamed_component(
                    channel,
                    streamedTailKernels[component],
                    streamedTailRadii[component],
                    streamedTailSigmas[component],
                    streamedTailWeights[component],
                    initializeComponent,
                    profile ? &profile->tailFilter[component] : nullptr,
                    profile ? &profile->tailFilterLaunches[component] : nullptr);
                if (err != cudaSuccess) {
                    return finish_profile(err);
                }
                channelAccumulatorInitialized = true;
            }
            if (!channelAccumulatorInitialized) {
                return finish_profile(cudaErrorInvalidValue);
            }
        }
        return finish_profile(cudaGetLastError());
    }

    bool accumulatorInitialized = false;
    if (gaussianWeight > 0.0f) {
        err = accumulate_component(
            dGaussianKernel,
            gaussianRadius,
            gaussianSigma,
            gaussianWeight,
            true,
            profile ? &profile->baseFilter : nullptr,
            profile ? &profile->baseFilterLaunches : nullptr);
        if (err != cudaSuccess)
            return finish_profile(err);
        accumulatorInitialized = true;
    }

    const float* tailKernels[3] = {dTailKernel0, dTailKernel1, dTailKernel2};
    const int tailRadii[3] = {tailRadius0, tailRadius1, tailRadius2};
    const float tailSigmas[3] = {tailSigma0, tailSigma1, tailSigma2};
    const float tailWeights[3] = {tailWeight0, tailWeight1, tailWeight2};
    for (int component = 0; component < 3; ++component) {
        if (!(tailWeights[component] > 0.0f)) {
            continue;
        }
        if (!(tailSigmas[component] > 0.0f) ||
            (tailSigmas[component] < 3.0f &&
             (!tailKernels[component] || tailRadii[component] <= 0))) {
            return finish_profile(cudaErrorInvalidValue);
        }
        const bool initializeComponent = !accumulatorInitialized;
        err = accumulate_component(
            tailKernels[component],
            tailRadii[component],
            tailSigmas[component],
            tailWeights[component],
            initializeComponent,
            profile ? &profile->tailFilter[component] : nullptr,
            profile ? &profile->tailFilterLaunches[component] : nullptr);
        if (err != cudaSuccess)
            return finish_profile(err);
        accumulatorInitialized = true;
    }
    if (!accumulatorInitialized) {
        return finish_profile(cudaErrorInvalidValue);
    }

    return finish_profile(cudaGetLastError());
}

template <typename Params>
cudaError_t build_spatial_dir_cached_log_raw_impl(
    const Params& params,
    float* logRawB,
    float* logRawG,
    float* logRawR,
    void* cudaStreamOpaque) {
    if (!params.src || params.srcRowBytes == 0 || !logRawB || !logRawG || !logRawR) {
        return cudaErrorInvalidValue;
    }
    if (params.width <= 0 || params.height <= 0) {
        return cudaSuccess;
    }
    if (!(params.nComponents == 3 || params.nComponents == 4)) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
    dim3 threads2D(32, 8);
    dim3 blocks2D(
        static_cast<unsigned int>((params.width + threads2D.x - 1) / threads2D.x),
        static_cast<unsigned int>((params.height + threads2D.y - 1) / threads2D.y));
    dir_cached_log_raw_build_kernel<<<blocks2D, threads2D, 0, stream>>>(
        params,
        logRawB,
        logRawG,
        logRawR);
    return cudaGetLastError();
}

extern "C" cudaError_t juicer_cuda_build_direct_spatial_dir_cached_log_raw(
    const JuicerCuda::DirectPipelineRunParams* hParams,
    float* logRawB,
    float* logRawG,
    float* logRawR,
    void* cudaStreamOpaque) {
    if (!hParams) {
        return cudaErrorInvalidValue;
    }
    return build_spatial_dir_cached_log_raw_impl(
        *hParams,
        logRawB,
        logRawG,
        logRawR,
        cudaStreamOpaque);
}

extern "C" cudaError_t juicer_cuda_build_direct_spatial_dir(
    const JuicerCuda::DirectPipelineRunParams* hParams,
    JuicerCuda::SpatialDirBuildRequest request) {
    if (!hParams) {
        return cudaErrorInvalidValue;
    }
    return build_spatial_dir_impl(*hParams, request);
}

extern "C" cudaError_t juicer_cuda_build_print_spatial_dir(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    JuicerCuda::SpatialDirBuildRequest request) {
    if (!hParams) {
        return cudaErrorInvalidValue;
    }
    return build_spatial_dir_impl(*hParams, request);
}

extern "C" cudaError_t juicer_cuda_build_print_spatial_dir_cached_log_raw(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    float* logRawB,
    float* logRawG,
    float* logRawR,
    void* cudaStreamOpaque) {
    if (!hParams) {
        return cudaErrorInvalidValue;
    }
    return build_spatial_dir_cached_log_raw_impl(
        *hParams,
        logRawB,
        logRawG,
        logRawR,
        cudaStreamOpaque);
}

namespace {

    struct LognormalSampleDevice {
        float mean = 0.0f;
        float standardDeviation = 0.0f;
        float normalSample = 0.0f;
    };

    struct WangLutEdgesDevice {
        int left = 0;
        int right = 0;
        int top = 0;
        int bottom = 0;
        int colorCount = 0;
    };

    struct NoiseCoordinateDevice {
        float x = 0.0f;
        float y = 0.0f;
    };

    struct StbnSampleLocationDevice {
        std::uint64_t absoluteX = 0;
        std::uint64_t absoluteY = 0;
        int offsetX = 0;
        int offsetY = 0;
        int frameOffset = 0;
    };

    struct GrainRngInitializationDevice {
        std::uint64_t seed = 0;
        std::uint32_t counterX = 0;
        std::uint32_t counterY = 0;
        std::uint64_t absoluteX = 0;
        std::uint64_t absoluteY = 0;
        int useStbn = 0;
        int frameOffset = 0;
    };

    struct LayerParticleModelSampleDevice {
        float density = 0.0f;
        float densityMaximum = 0.0f;
        float particleCount = 0.0f;
        float particleOpticalDensity = 0.0f;
        float uniformity = 0.0f;
        std::uint64_t seed = 0;
        std::uint64_t absoluteX = 0;
        std::uint64_t absoluteY = 0;
        const JuicerCuda::GrainPayload* grain = nullptr;
        int useStbn = 0;
        int frameOffset = 0;
    };

    __device__ __forceinline__ float lognormal_from_mean_std_device(LognormalSampleDevice sample) {
        const float m2 = sample.mean * sample.mean;
        const float s2 = sample.standardDeviation * sample.standardDeviation;
        const float sigmaSq = logf(1.0f + (s2 / m2));
        const float sigma = sqrtf(fmaxf(0.0f, sigmaSq));
        const float mu = logf(fmaxf(1e-12f, sample.mean)) - 0.5f * sigmaSq;
        return expf(mu + sigma * sample.normalSample);
    }

    __device__ __forceinline__ std::uint64_t splitmix64_device(std::uint64_t v) {
        v += 0x9E3779B97F4A7C15ULL;
        v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ULL;
        v = (v ^ (v >> 27)) * 0x94D049BB133111EBULL;
        return v ^ (v >> 31);
    }

    __device__ __forceinline__ float stbn_lookup_device(
        const JuicerCuda::GrainPayload& grain,
        int x,
        int y,
        int t) {
        if (!grain.stbn || grain.stbnWidth <= 0 || grain.stbnHeight <= 0 || grain.stbnFrames <= 0) {
            return 0.0f;
        }

        if (grain.stbnWidth > 0) {
            x = (x + grain.stbnOffsetX) % grain.stbnWidth;
            if (x < 0)
                x += grain.stbnWidth;
        }
        if (grain.stbnHeight > 0) {
            y = (y + grain.stbnOffsetY) % grain.stbnHeight;
            if (y < 0)
                y += grain.stbnHeight;
        }

        if (grain.stbnFrames > 0) {
            t = t % grain.stbnFrames;
            if (t < 0)
                t += grain.stbnFrames;
        }

        const std::size_t idx = (static_cast<std::size_t>(t) * static_cast<std::size_t>(grain.stbnHeight) + static_cast<std::size_t>(y)) *
                                    static_cast<std::size_t>(grain.stbnWidth) +
                                static_cast<std::size_t>(x);
        const std::uint8_t v = grain.stbn[idx];
        return (static_cast<float>(v) + 0.5f) * (1.0f / 256.0f);
    }

    __device__ __forceinline__ float smoothstep_device(float t) {
        return t * t * (3.0f - 2.0f * t);
    }

    __device__ __forceinline__ float hash01_device(int x, int y, std::uint64_t seed) {
        const std::uint64_t h = splitmix64_device(seed ^
                                                  (static_cast<std::uint64_t>(x) * 0x9E3779B97F4A7C15ULL) ^
                                                  (static_cast<std::uint64_t>(y) * 0xBF58476D1CE4E5B9ULL));
        constexpr float kInvU32 = 1.0f / 4294967296.0f;
        return static_cast<float>(static_cast<std::uint32_t>(h & 0xFFFFFFFFu)) * kInvU32;
    }

    __device__ __forceinline__ std::uint32_t hash_u32_device(int x, int y, std::uint64_t seed, std::uint64_t salt) {
        const std::uint64_t h = splitmix64_device(seed ^ salt ^
                                                  (static_cast<std::uint64_t>(x) * 0x9E3779B97F4A7C15ULL) ^
                                                  (static_cast<std::uint64_t>(y) * 0xBF58476D1CE4E5B9ULL));
        return static_cast<std::uint32_t>(h & 0xFFFFFFFFu);
    }

    __device__ __forceinline__ std::size_t wang_lut_index_device(WangLutEdgesDevice edges) {
        const std::size_t c = static_cast<std::size_t>(edges.colorCount);
        return (((static_cast<std::size_t>(edges.left) * c + static_cast<std::size_t>(edges.right)) * c +
                 static_cast<std::size_t>(edges.top)) *
                    c +
                static_cast<std::size_t>(edges.bottom));
    }

    __device__ __forceinline__ int wang_tile_id_device(const JuicerCuda::GrainPayload& grain, std::int64_t mx, std::int64_t my) {
        if (!grain.wangLut || grain.wangColors <= 0) {
            return -1;
        }
        const int colors = grain.wangColors;
        const std::uint64_t seed = (grain.clipToken != 0) ? grain.clipToken : grain.stbnSessionSeed;
        const std::uint64_t baseSeed = (seed != 0) ? seed : 1ULL;
        const std::uint32_t l = hash_u32_device(static_cast<int>(mx), static_cast<int>(my), baseSeed, 0xA5A5A5A5u) % colors;
        const std::uint32_t r = hash_u32_device(static_cast<int>(mx + 1), static_cast<int>(my), baseSeed, 0x5A5A5A5Au) % colors;
        const std::uint32_t t = hash_u32_device(static_cast<int>(mx), static_cast<int>(my), baseSeed, 0xC3C3C3C3u) % colors;
        const std::uint32_t b = hash_u32_device(static_cast<int>(mx), static_cast<int>(my + 1), baseSeed, 0x3C3C3C3Cu) % colors;
        WangLutEdgesDevice edges{};
        edges.left = static_cast<int>(l);
        edges.right = static_cast<int>(r);
        edges.top = static_cast<int>(t);
        edges.bottom = static_cast<int>(b);
        edges.colorCount = colors;
        const std::size_t idx = wang_lut_index_device(edges);
        return static_cast<int>(grain.wangLut[idx]);
    }

    __device__ __forceinline__ float wang_tile_sample_device(const JuicerCuda::GrainPayload& grain, int tileId, int x, int y) {
        if (!grain.wangTiles || grain.wangWidth <= 0 || grain.wangHeight <= 0 || grain.wangCount <= 0) {
            return 0.0f;
        }
        if (tileId < 0 || tileId >= grain.wangCount) {
            return 0.0f;
        }
        const int w = grain.wangWidth;
        const int h = grain.wangHeight;
        if (x < 0)
            x = 0;
        if (y < 0)
            y = 0;
        if (x >= w)
            x = w - 1;
        if (y >= h)
            y = h - 1;
        const std::size_t idx = (static_cast<std::size_t>(tileId) * static_cast<std::size_t>(h) + static_cast<std::size_t>(y)) *
                                    static_cast<std::size_t>(w) +
                                static_cast<std::size_t>(x);
        const std::uint8_t v = grain.wangTiles[idx];
        return (static_cast<float>(v) + 0.5f) * (1.0f / 256.0f);
    }

    __device__ __forceinline__ void wang_offsets_device(
        const JuicerCuda::GrainPayload& grain,
        std::uint64_t absX,
        std::uint64_t absY,
        int& outX,
        int& outY) {
        outX = 0;
        outY = 0;
        if (!grain.wangTiles || !grain.wangLut || grain.wangWidth <= 0 || grain.wangHeight <= 0 || grain.wangCount <= 0) {
            return;
        }
        if (!(grain.pixelSizeUm > 0.0f) || !(grain.wangCellMm > 0.0f)) {
            return;
        }

        const float xMm = static_cast<float>(absX) * (grain.pixelSizeUm * 0.001f);
        const float yMm = static_cast<float>(absY) * (grain.pixelSizeUm * 0.001f);
        const float invCell = 1.0f / grain.wangCellMm;
        const float cellX = xMm * invCell;
        const float cellY = yMm * invCell;
        const float cellFx = cellX - floorf(cellX);
        const float cellFy = cellY - floorf(cellY);
        const std::int64_t mx = static_cast<std::int64_t>(floorf(cellX));
        const std::int64_t my = static_cast<std::int64_t>(floorf(cellY));

        const int tileId = wang_tile_id_device(grain, mx, my);
        if (tileId < 0) {
            return;
        }

        const int w = grain.wangWidth;
        const int h = grain.wangHeight;
        int tx = static_cast<int>(floorf(cellFx * static_cast<float>(w)));
        int ty = static_cast<int>(floorf(cellFy * static_cast<float>(h)));
        if (tx >= w)
            tx = w - 1;
        if (ty >= h)
            ty = h - 1;
        const float v0 = wang_tile_sample_device(grain, tileId, tx, ty);
        const float v1 = wang_tile_sample_device(grain, tileId, (tx + (w >> 1)) % w, (ty + (h >> 1)) % h);

        const int warpX = (grain.stbnWidth > 0) ? ((grain.stbnWidth / 8) > 4 ? (grain.stbnWidth / 8) : 4) : 0;
        const int warpY = (grain.stbnHeight > 0) ? ((grain.stbnHeight / 8) > 4 ? (grain.stbnHeight / 8) : 4) : 0;
        if (warpX == 0 || warpY == 0) {
            return;
        }
        outX = static_cast<int>(floorf((v0 - 0.5f) * 2.0f * static_cast<float>(warpX)));
        outY = static_cast<int>(floorf((v1 - 0.5f) * 2.0f * static_cast<float>(warpY)));
    }

    __device__ __forceinline__ float value_noise_device(
        NoiseCoordinateDevice coordinate,
        std::uint64_t seed) {
        const int ix = static_cast<int>(floorf(coordinate.x));
        const int iy = static_cast<int>(floorf(coordinate.y));
        const float fx = coordinate.x - static_cast<float>(ix);
        const float fy = coordinate.y - static_cast<float>(iy);
        const float sx = smoothstep_device(fx);
        const float sy = smoothstep_device(fy);

        const float v00 = hash01_device(ix, iy, seed);
        const float v10 = hash01_device(ix + 1, iy, seed);
        const float v01 = hash01_device(ix, iy + 1, seed);
        const float v11 = hash01_device(ix + 1, iy + 1, seed);

        const float v0 = v00 + (v10 - v00) * sx;
        const float v1 = v01 + (v11 - v01) * sx;
        return v0 + (v1 - v0) * sy;
    }

    __device__ __forceinline__ float grain_breathing_factor_device(
        const JuicerCuda::GrainPayload& grain,
        std::uint64_t absX,
        std::uint64_t absY) {
        const float amp = grain.breathingAmplitude;
        const int period = grain.breathingPeriodFrames;
        if (!(amp > 0.0f) || period <= 0 || !(grain.pixelSizeUm > 0.0f)) {
            return 1.0f;
        }

        const float cellSmallUm = grain.breathingCellUmSmall;
        const float cellLargeUm = grain.breathingCellUmLarge;
        if (!(cellSmallUm > 0.0f) || !(cellLargeUm > 0.0f)) {
            return 1.0f;
        }

        const float time = static_cast<float>(grain.frameIndex) + grain.timeAlpha;
        const float periodF = static_cast<float>(period);
        const float stepF = floorf(time / periodF);
        const std::int64_t step = static_cast<std::int64_t>(stepF);
        const float frac = (time - stepF * periodF) / periodF;
        const float t = smoothstep_device(fminf(fmaxf(frac, 0.0f), 1.0f));

        const std::uint64_t seed = (grain.stbnSessionSeed != 0) ? grain.stbnSessionSeed : 1ULL;
        const std::uint64_t seedA = splitmix64_device(seed ^ (static_cast<std::uint64_t>(step) * 0x8EBC6AF09C88C6E3ULL));
        const std::uint64_t seedB = splitmix64_device(seed ^ (static_cast<std::uint64_t>(step + 1) * 0x8EBC6AF09C88C6E3ULL));

        const float invPixel = 1.0f / grain.pixelSizeUm;
        const float cellSmallPx = fmaxf(4.0f, cellSmallUm * invPixel);
        const float cellLargePx = fmaxf(4.0f, cellLargeUm * invPixel);

        float driftPx = 0.0f;
        if (grain.breathingDriftUmPerFrame > 0.0f) {
            driftPx = grain.breathingDriftUmPerFrame * invPixel;
        }
        float driftX = 0.0f;
        float driftY = 0.0f;
        if (driftPx > 0.0f) {
            const std::uint64_t h = splitmix64_device(seed ^ 0x6A09E667F3BCC909ULL);
            constexpr float kTwoPi = 6.28318530717958647692f;
            constexpr float kInvU32 = 1.0f / 4294967296.0f;
            const float angle = static_cast<float>(static_cast<std::uint32_t>(h & 0xFFFFFFFFu)) * kInvU32 * kTwoPi;
            driftX = cosf(angle) * driftPx;
            driftY = sinf(angle) * driftPx;
        }

        const float rollPx = (grain.pitchPx > 0) ? static_cast<float>(grain.pitchPx) : 0.0f;
        const float baseX = static_cast<float>(absX) + driftX * time;
        const float baseY = static_cast<float>(absY) + driftY * time + rollPx * time;

        const float xSmall = baseX / cellSmallPx;
        const float ySmall = baseY / cellSmallPx;
        const float xLarge = baseX / cellLargePx;
        const float yLarge = baseY / cellLargePx;

        NoiseCoordinateDevice smallCoordinate{};
        smallCoordinate.x = xSmall;
        smallCoordinate.y = ySmall;
        NoiseCoordinateDevice largeCoordinate{};
        largeCoordinate.x = xLarge;
        largeCoordinate.y = yLarge;
        const float noiseSmallA = value_noise_device(smallCoordinate, seedA ^ 0x9E3779B97F4A7C15ULL);
        const float noiseSmallB = value_noise_device(smallCoordinate, seedB ^ 0x9E3779B97F4A7C15ULL);
        const float noiseLargeA = value_noise_device(largeCoordinate, seedA ^ 0xBF58476D1CE4E5B9ULL);
        const float noiseLargeB = value_noise_device(largeCoordinate, seedB ^ 0xBF58476D1CE4E5B9ULL);

        const float noiseSmall = noiseSmallA + (noiseSmallB - noiseSmallA) * t;
        const float noiseLarge = noiseLargeA + (noiseLargeB - noiseLargeA) * t;
        const float mix = fminf(fmaxf(grain.breathingMix, 0.0f), 1.0f);
        const float noise = mix * noiseSmall + (1.0f - mix) * noiseLarge;

        const float n = noise * 2.0f - 1.0f;
        const float factor = 1.0f + amp * n;
        return (factor > 0.0f) ? factor : 0.0f;
    }

    __device__ __forceinline__ float grain_clump_factor_device(
        const JuicerCuda::GrainPayload& grain,
        std::uint64_t absX,
        std::uint64_t absY) {
        const float stddevSpatial = grain.microStructure[1] * 0.001f;
        if (!(stddevSpatial > 0.0f) || !(grain.pixelSizeUm > 0.0f)) {
            return 1.0f;
        }

        float cellUm = grain.microStructure[0];
        if (!(cellUm > 0.0f)) {
            cellUm = 0.0f;
        }
        const float invPixel = 1.0f / grain.pixelSizeUm;
        float cellPx = cellUm * invPixel;
        const float minCellPx = 4.0f;
        if (!(cellPx > minCellPx)) {
            cellPx = minCellPx;
        }

        const float time = static_cast<float>(grain.frameIndex) + grain.timeAlpha;

        const float rollPx = (grain.pitchPx > 0) ? static_cast<float>(grain.pitchPx) : 0.0f;
        const float baseXStatic = static_cast<float>(absX);
        const float baseYStatic = static_cast<float>(absY) + rollPx * time;
        const float xStatic = baseXStatic / cellPx;
        const float yStatic = baseYStatic / cellPx;

        std::uint64_t staticSeed = (grain.clipToken != 0) ? grain.clipToken : grain.stbnSessionSeed;
        if (staticSeed == 0) {
            staticSeed = 1ULL;
        }
        NoiseCoordinateDevice primaryCoordinate{};
        primaryCoordinate.x = xStatic;
        primaryCoordinate.y = yStatic;
        NoiseCoordinateDevice secondaryCoordinate{};
        secondaryCoordinate.x = xStatic + 19.19f;
        secondaryCoordinate.y = yStatic + 7.23f;
        const float u1S = value_noise_device(primaryCoordinate, staticSeed ^ 0x9E3779B97F4A7C15ULL);
        const float u2S = value_noise_device(secondaryCoordinate, staticSeed ^ 0xBF58476D1CE4E5B9ULL);
        float u1 = fminf(fmaxf(u1S, 1e-6f), 1.0f - 1e-6f);
        float u2 = fminf(fmaxf(u2S, 0.0f), 1.0f);
        const float r = sqrtf(-2.0f * logf(u1));
        constexpr float kTwoPi = 6.28318530717958647692f;
        const float nStatic = r * cosf(kTwoPi * u2);

        const float mix = fminf(fmaxf(grain.clumpTemporalMix, 0.0f), 1.0f);
        if (!(mix > 0.0f)) {
            LognormalSampleDevice staticSample{};
            staticSample.mean = 1.0f;
            staticSample.standardDeviation = stddevSpatial;
            staticSample.normalSample = nStatic;
            float staticVal = lognormal_from_mean_std_device(staticSample);
            if (!device_isfinite(staticVal)) {
                staticVal = 1.0f;
            }
            return staticVal;
        }

        int period = grain.clumpMorphPeriodFrames;
        if (period <= 0) {
            period = 1;
        }
        const float periodF = static_cast<float>(period);
        const float stepF = floorf(time / periodF);
        const std::int64_t step = static_cast<std::int64_t>(stepF);
        const float frac = (time - stepF * periodF) / periodF;
        const float t = smoothstep_device(fminf(fmaxf(frac, 0.0f), 1.0f));

        std::uint64_t temporalSeed = (grain.stbnSessionSeed != 0) ? grain.stbnSessionSeed : 1ULL;
        const std::uint64_t seedA = splitmix64_device(temporalSeed ^ (static_cast<std::uint64_t>(step) * 0xD2B74407B1CE6E93ULL));
        const std::uint64_t seedB = splitmix64_device(temporalSeed ^ (static_cast<std::uint64_t>(step + 1) * 0xD2B74407B1CE6E93ULL));

        const float u1A = hash01_device(0, 0, seedA ^ 0x9E3779B97F4A7C15ULL);
        const float u2A = hash01_device(1, 0, seedA ^ 0xBF58476D1CE4E5B9ULL);
        const float u1B = hash01_device(0, 0, seedB ^ 0x9E3779B97F4A7C15ULL);
        const float u2B = hash01_device(1, 0, seedB ^ 0xBF58476D1CE4E5B9ULL);

        u1 = u1A + (u1B - u1A) * t;
        u2 = u2A + (u2B - u2A) * t;
        u1 = fminf(fmaxf(u1, 1e-6f), 1.0f - 1e-6f);
        u2 = fminf(fmaxf(u2, 0.0f), 1.0f);

        const float rT = sqrtf(-2.0f * logf(u1));
        const float nTemporal = rT * cosf(kTwoPi * u2);

        constexpr float kClumpStrengthStdScale = 0.23597824f;
        const float strengthStd = mix * kClumpStrengthStdScale;
        LognormalSampleDevice strengthSample{};
        strengthSample.mean = 1.0f;
        strengthSample.standardDeviation = strengthStd;
        strengthSample.normalSample = nTemporal;
        float strength = lognormal_from_mean_std_device(strengthSample);
        const float strengthNorm = rsqrtf(1.0f + strengthStd * strengthStd);
        strength *= strengthNorm;
        const float stddev = stddevSpatial * strength;
        const float denom = 1.0f + stddev * stddev;
        const float numer = 1.0f + stddevSpatial * stddevSpatial;
        const float clumpRmsNorm = (denom > 0.0f) ? sqrtf(numer / denom) : 1.0f;

        LognormalSampleDevice clumpSample{};
        clumpSample.mean = 1.0f;
        clumpSample.standardDeviation = stddev;
        clumpSample.normalSample = nStatic;
        float clumpVal = lognormal_from_mean_std_device(clumpSample);
        if (!device_isfinite(clumpVal)) {
            clumpVal = 1.0f;
        }
        clumpVal *= clumpRmsNorm;

        return clumpVal;
    }

    __device__ __forceinline__ float stbn_sample_device(
        const JuicerCuda::GrainPayload& grain,
        StbnSampleLocationDevice location) {
        if (!grain.stbn || grain.stbnWidth <= 0 || grain.stbnHeight <= 0 || grain.stbnFrames <= 0) {
            return 0.0f;
        }

        const int t = grain.stbnFrame + location.frameOffset;
        std::int64_t x64 =
            static_cast<std::int64_t>(location.absoluteX) + static_cast<std::int64_t>(location.offsetX);
        std::int64_t y64 =
            static_cast<std::int64_t>(location.absoluteY) + static_cast<std::int64_t>(location.offsetY);
        if (grain.pitchPx > 0) {
            y64 += static_cast<std::int64_t>(grain.pitchPx) *
                   (static_cast<std::int64_t>(grain.frameIndex) +
                    static_cast<std::int64_t>(location.frameOffset));
        }

        if (grain.stbnWidth > 0) {
            const std::int64_t w = static_cast<std::int64_t>(grain.stbnWidth);
            x64 %= w;
            if (x64 < 0)
                x64 += w;
        }
        if (grain.stbnHeight > 0) {
            const std::int64_t h = static_cast<std::int64_t>(grain.stbnHeight);
            y64 %= h;
            if (y64 < 0)
                y64 += h;
        }

        const int x = static_cast<int>(x64);
        const int y = static_cast<int>(y64);
        return stbn_lookup_device(grain, x, y, t);
    }

    struct GrainRngDevice {
        openrand::Philox rng;
        const JuicerCuda::GrainPayload* grain = nullptr;
        std::uint64_t absX = 0;
        std::uint64_t absY = 0;
        std::uint64_t seed = 0;
        std::uint32_t drawIndex = 0;
        int useStbn = 0;
        int frameOffset = 0;
        int wangOffsetX = 0;
        int wangOffsetY = 0;

        __device__ GrainRngDevice(
            const JuicerCuda::GrainPayload* grain_,
            GrainRngInitializationDevice initialization)
            : rng(
                  initialization.seed,
                  initialization.counterX,
                  openrand::DEFAULT_GLOBAL_SEED,
                  initialization.counterY),
              grain(grain_),
              absX(initialization.absoluteX),
              absY(initialization.absoluteY),
              seed(initialization.seed),
              useStbn(initialization.useStbn),
              frameOffset(initialization.frameOffset) {
            if (useStbn && grain) {
                wang_offsets_device(*grain, absX, absY, wangOffsetX, wangOffsetY);
            }
        }

        __device__ __forceinline__ float uniform() {
            if (useStbn && grain) {
                const std::uint32_t draw = drawIndex++;
                const std::uint64_t h = splitmix64_device(seed + static_cast<std::uint64_t>(draw) * 0x9E3779B97F4A7C15ULL);
                const int offsetX = static_cast<int>(h & 0xFFFFu) + wangOffsetX;
                const int offsetY = static_cast<int>((h >> 16) & 0xFFFFu) + wangOffsetY;
                StbnSampleLocationDevice location{};
                location.absoluteX = absX;
                location.absoluteY = absY;
                location.offsetX = offsetX;
                location.offsetY = offsetY;
                location.frameOffset = frameOffset;
                return stbn_sample_device(*grain, location);
            }
            return rng.rand<float>();
        }

        __device__ __forceinline__ float normal() {
            float u1 = uniform();
            u1 = fminf(fmaxf(u1, 1e-7f), 1.0f);
            const float u2 = uniform();
            const float r = sqrtf(-2.0f * logf(u1));
            constexpr float kTwoPi = 6.28318530717958647692f;
            return r * cosf(kTwoPi * u2);
        }
    };

    __device__ __forceinline__ int fast_poisson_device(
        float lambda,
        GrainRngDevice& rng) {
        if (!device_isfinite(lambda) || !(lambda > 0.0f)) {
            return 0;
        }
        if (lambda < 30.0f) {
            const float L = expf(-lambda);
            float p = 1.0f;
            int k = 0;
            while (p > L && k < 1024) {
                ++k;
                const float u = rng.uniform();
                p *= fminf(fmaxf(u, 1e-7f), 1.0f);
            }
            return k - 1;
        }

        const float z = rng.normal();
        float sample = lambda + sqrtf(lambda) * z;
        if (!device_isfinite(sample)) {
            sample = 0.0f;
        }
        int sampleInt = static_cast<int>(floorf(sample + 0.5f));
        if (sampleInt < 0)
            sampleInt = 0;
        return sampleInt;
    }

    __device__ __forceinline__ int fast_binomial_device(
        int n,
        float p,
        GrainRngDevice& rng) {
        if (n <= 0) {
            return 0;
        }
        if (!device_isfinite(p)) {
            return 0;
        }
        if (p <= 0.0f) {
            return 0;
        }
        if (p >= 1.0f) {
            return n;
        }

        constexpr int kThreshold = 25;
        if (n < kThreshold) {
            int count = 0;
            for (int k = 0; k < n; ++k) {
                const float u = rng.uniform();
                if (u < p) {
                    ++count;
                }
            }
            return count;
        }

        const float mean = static_cast<float>(n) * p;
        const float var = static_cast<float>(n) * p * (1.0f - p);
        if (var > 10.0f) {
            const float z = rng.normal();
            float sample = mean + sqrtf(var) * z;
            if (!device_isfinite(sample)) {
                sample = 0.0f;
            }
            int approx = static_cast<int>(floorf(sample + 0.5f));
            if (approx < 0)
                approx = 0;
            if (approx > n)
                approx = n;
            return approx;
        }

        float u = rng.uniform();
        float cdf = 0.0f;
        float prob = powf(1.0f - p, static_cast<float>(n));
        if (!device_isfinite(prob) || prob <= 0.0f) {
            const float z = rng.normal();
            float sample = mean + sqrtf(fmaxf(0.0f, var)) * z;
            if (!device_isfinite(sample)) {
                sample = 0.0f;
            }
            int approx = static_cast<int>(floorf(sample + 0.5f));
            if (approx < 0)
                approx = 0;
            if (approx > n)
                approx = n;
            return approx;
        }

        int k = 0;
        while (cdf < u && k <= n) {
            cdf += prob;
            if (k < n) {
                const float denom = 1.0f - p;
                if (!(denom > 0.0f)) {
                    return n;
                }
                prob = prob * ((static_cast<float>(n - k) / static_cast<float>(k + 1)) * (p / denom));
            }
            ++k;
        }
        return k - 1;
    }

    __device__ __forceinline__ int poisson_sample_device(float lambda, GrainRngDevice& rng) {
        return fast_poisson_device(lambda, rng);
    }

    __device__ __forceinline__ int binomial_sample_device(int n, float p, GrainRngDevice& rng) {
        return fast_binomial_device(n, p, rng);
    }

    __device__ __forceinline__ float layer_particle_model_device(
        LayerParticleModelSampleDevice sample) {
        if (!sample.grain) {
            return 0.0f;
        }
        float density = sample.density;
        const float densityMax = sample.densityMaximum;
        const float nParticles = sample.particleCount;
        const float odParticle = sample.particleOpticalDensity;
        const float uniformity = sample.uniformity;
        const JuicerCuda::GrainPayload& grain = *sample.grain;
        if (!device_isfinite(density) || density < 0.0f) {
            density = 0.0f;
        }
        if (!device_isfinite(densityMax) || !(densityMax > 0.0f)) {
            return 0.0f;
        }
        if (!device_isfinite(nParticles) || !(nParticles > 0.0f)) {
            return 0.0f;
        }
        if (!device_isfinite(odParticle) || !(odParticle > 0.0f)) {
            return 0.0f;
        }

        float probability = density / densityMax;
        probability = fminf(fmaxf(probability, 1e-6f), 1.0f - 1e-6f);

        float uniform = device_isfinite(uniformity) ? fminf(fmaxf(uniformity, 0.0f), 1.0f) : 0.0f;
        float saturation = 1.0f - probability * uniform * (1.0f - 1e-6f);
        if (!device_isfinite(saturation) || saturation <= 0.0f) {
            saturation = 1e-6f;
        }

        GrainRngInitializationDevice initialization{};
        initialization.seed = sample.seed;
        initialization.counterX = static_cast<std::uint32_t>(sample.absoluteX);
        initialization.counterY = static_cast<std::uint32_t>(sample.absoluteY);
        initialization.absoluteX = sample.absoluteX;
        initialization.absoluteY = sample.absoluteY;
        initialization.useStbn = sample.useStbn;
        initialization.frameOffset = sample.frameOffset;
        GrainRngDevice rng(&grain, initialization);
        const float lambda = nParticles / saturation;
        const int seeds = poisson_sample_device(lambda, rng);
        const int grainCount = binomial_sample_device(seeds, probability, rng);
        float grainValue = static_cast<float>(grainCount) * odParticle * saturation;
        if (!device_isfinite(grainValue)) {
            grainValue = 0.0f;
        }
        return grainValue;
    }

    __device__ __forceinline__ float interp_density_layer_device(
        float density,
        const float* JUICER_RESTRICT x,
        const float* JUICER_RESTRICT y,
        int n,
        bool positiveFilm) {
        if (!x || !y || n <= 0) {
            return density;
        }
        if (!device_isfinite(density)) {
            return y[0];
        }
        const float query = positiveFilm ? -density : density;
        const auto axis_at = [x, positiveFilm](int index) {
            const float axisValue = ldg_f(x + index);
            return positiveFilm ? -axisValue : axisValue;
        };

        int domainBegin = 0;
        while (domainBegin < n && !device_isfinite(axis_at(domainBegin))) {
            ++domainBegin;
        }
        if (domainBegin >= n) {
            return density;
        }
        int domainEnd = n - 1;
        while (domainEnd > domainBegin && !device_isfinite(axis_at(domainEnd))) {
            --domainEnd;
        }

        const float xmin = axis_at(domainBegin);
        const float xmax = axis_at(domainEnd);
        if (!device_isfinite(xmin) || !device_isfinite(xmax) || !(xmax >= xmin)) {
            return ldg_f(y + domainBegin);
        }

        if (query <= xmin) {
            return ldg_f(y + domainBegin);
        }
        if (query >= xmax) {
            return ldg_f(y + domainEnd);
        }

        int i1 = domainBegin + 1;
        while (i1 <= domainEnd && axis_at(i1) < query) {
            ++i1;
        }
        if (i1 > domainEnd) {
            return ldg_f(y + domainEnd);
        }

        const int i0 = i1 - 1;
        const float x0 = axis_at(i0);
        const float x1 = axis_at(i1);
        const float y0 = ldg_f(y + i0);
        const float y1 = ldg_f(y + i1);

        const float denom = x1 - x0;
        if (!(denom > 0.0f) || !device_isfinite(denom)) {
            return y0;
        }

        const float t = (query - x0) / denom;
        return y0 + t * (y1 - y0);
    }

} // namespace

__global__ void develop_film_density_kernel(
    JuicerCuda::PipelineRunParams params,
    float* outC,
    float* outM,
    float* outY) {
    const JuicerCuda::FilmDevelopPayload& dev = params.filmDevelop;

    if (!params.src || params.srcRowBytes == 0) {
        return;
    }
    if (!outC || !outM || !outY) {
        return;
    }

    const int nC = params.nComponents;
    if (!(nC == 3 || nC == 4)) {
        return;
    }
    if (params.width <= 0 || params.height <= 0) {
        return;
    }

    const std::size_t pixelBytes = static_cast<std::size_t>(nC) * sizeof(float);
    const bool useSpatialDir = juicer_cuda_spatial_dir_filtered_correction_active_device(dev);
    const std::size_t width = static_cast<std::size_t>(params.width);
    const std::size_t height = static_cast<std::size_t>(params.height);
    const std::size_t yStep = static_cast<std::size_t>(blockDim.y) * static_cast<std::size_t>(gridDim.y);
    const std::size_t xStep = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    for (std::size_t y = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(blockDim.y) + static_cast<std::size_t>(threadIdx.y); y < height; y += yStep) {
        const char* srcRow = reinterpret_cast<const char*>(params.src) + y * params.srcRowBytes;
        for (std::size_t x = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x); x < width; x += xStep) {
            const float* srcPix = reinterpret_cast<const float*>(srcRow + x * pixelBytes);
            if (!srcPix) {
                continue;
            }

            const float rgbIn[3] = {srcPix[0], srcPix[1], srcPix[2]};
            float D_cmy[3] = {0.0f, 0.0f, 0.0f};
            if (useSpatialDir) {
                const std::size_t idx = y * width + x;
                float logE_raw[3] = {0.0f, 0.0f, 0.0f};
                if (juicer_cuda_spatial_dir_cached_log_raw_active_device(dev)) {
                    juicer_cuda_load_spatial_dir_cached_log_raw_device(dev, idx, logE_raw);
                } else {
                    float logE_sanitized[3] = {0.0f, 0.0f, 0.0f};
                    float layerPre[3] = {0.0f, 0.0f, 0.0f};
                    FilmDevelopIntermediatesDevice intermediates{};
                    intermediates.logERaw = logE_raw;
                    intermediates.logESanitized = logE_sanitized;
                    intermediates.layerPre = layerPre;
                    compute_logE_and_layer_pre_device(params, rgbIn, intermediates);
                }
                juicer_cuda_develop_dir_final_device(dev, logE_raw, idx, D_cmy);
            } else {
                float logE_raw[3] = {0.0f, 0.0f, 0.0f};
                float logE_sanitized[3] = {0.0f, 0.0f, 0.0f};
                float layerPre[3] = {0.0f, 0.0f, 0.0f};
                FilmDevelopIntermediatesDevice intermediates{};
                intermediates.logERaw = logE_raw;
                intermediates.logESanitized = logE_sanitized;
                intermediates.layerPre = layerPre;
                compute_logE_and_layer_pre_device(params, rgbIn, intermediates);

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

            const std::size_t idx = y * width + x;
            outC[idx] = D_cmy[0];
            outM[idx] = D_cmy[1];
            outY[idx] = D_cmy[2];
        }
    }
}

__global__ void grain_clear_kernel(float* out, int n) {
    if (n <= 0) {
        return;
    }
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    if (idx >= static_cast<std::size_t>(n)) {
        return;
    }
    if (!out) {
        return;
    }
    out[idx] = 0.0f;
}

__global__ void grain_accumulate_kernel(float* dst, const float* src, int n) {
    if (n <= 0) {
        return;
    }
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    if (idx >= static_cast<std::size_t>(n)) {
        return;
    }
    if (!dst || !src) {
        return;
    }
    const float v = dst[idx] + src[idx];
    dst[idx] = device_isfinite(v) ? v : 0.0f;
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters) Focused CUDA kernels use fixed launch bindings.
__global__ void grain_accumulate_weighted_kernel(
    float* dst,
    const float* src,
    int n,
    float weight) {
    if (n <= 0) {
        return;
    }
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    if (idx >= static_cast<std::size_t>(n) || !dst || !src) {
        return;
    }
    const float w = device_isfinite(weight) ? weight : 0.0f;
    const float v = dst[idx] + w * src[idx];
    dst[idx] = device_isfinite(v) ? v : 0.0f;
}

__global__ void grain_scale_kernel(float* inOut, int n, float scale) {
    if (n <= 0) {
        return;
    }
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    if (idx >= static_cast<std::size_t>(n) || !inOut) {
        return;
    }
    const float s = device_isfinite(scale) ? scale : 0.0f;
    const float v = inOut[idx] * s;
    inOut[idx] = device_isfinite(v) ? v : 0.0f;
}

__global__ void grain_reconstruct_kernel(
    float* inOutMean,
    const float* delta,
    int n,
    float scale,
    int addMean) {
    if (n <= 0) {
        return;
    }
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    if (idx >= static_cast<std::size_t>(n) || !inOutMean || !delta) {
        return;
    }
    const float s = device_isfinite(scale) ? scale : 0.0f;
    const float mean = addMean ? inOutMean[idx] : 0.0f;
    const float v = mean + s * delta[idx];
    inOutMean[idx] = device_isfinite(v) ? v : 0.0f;
}

__global__ void grain_add_bias_kernel(float* inOut, int n, float bias) {
    if (n <= 0) {
        return;
    }
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    if (idx >= static_cast<std::size_t>(n)) {
        return;
    }
    if (!inOut) {
        return;
    }
    const float v = inOut[idx] + bias;
    inOut[idx] = device_isfinite(v) ? v : 0.0f;
}

__global__ void grain_subtract_kernel(float* inOut, const float* sub, int n, float amplitude) {
    if (n <= 0) {
        return;
    }
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    if (idx >= static_cast<std::size_t>(n)) {
        return;
    }
    if (!inOut || !sub) {
        return;
    }
    float a = device_isfinite(amplitude) ? amplitude : 1.0f;
    if (a < 0.0f) {
        a = 0.0f;
    }
    const float v = (inOut[idx] - sub[idx]) * a;
    inOut[idx] = device_isfinite(v) ? v : 0.0f;
}
// NOLINTEND(bugprone-easily-swappable-parameters)
__global__ void grain_apply_simple_kernel(
    JuicerCuda::GrainPayload grain,
    int width,
    int height,
    const float* inDensity,
    float* outGrain,
    int channelIndex) {
    if (width <= 0 || height <= 0) {
        return;
    }
    const std::size_t x = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    const std::size_t y = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(blockDim.y) + static_cast<std::size_t>(threadIdx.y);
    if (x >= static_cast<std::size_t>(width) || y >= static_cast<std::size_t>(height)) {
        return;
    }
    if (!inDensity || !outGrain) {
        return;
    }
    if (channelIndex < 0 || channelIndex > 2) {
        return;
    }

    const std::size_t idx = y * static_cast<std::size_t>(width) + x;
    float density = inDensity[idx];

    const float densityMin = grain.densityMin[channelIndex];
    const float densityMax = grain.densityMax[channelIndex];
    const float nParticles = grain.nParticles[channelIndex];
    const float odParticle = grain.odParticle[channelIndex];
    const float uniformity = grain.uniformity[channelIndex];
    const int nSubLayers = (grain.nSubLayers > 0) ? grain.nSubLayers : 1;
    float mixWeight = grain.sizeMixWeight;
    float mixScale = grain.sizeMixScale;
    const float timeAlpha = grain.timeAlpha;
    const bool useRetime = (timeAlpha > 1e-6f && timeAlpha < 0.999999f);
    const bool wantNext = useRetime;

    if (!device_isfinite(densityMax) || !(densityMax > 0.0f) ||
        !device_isfinite(nParticles) || !(nParticles > 0.0f) ||
        !device_isfinite(odParticle) || !(odParticle > 0.0f)) {
        return;
    }

    const std::int64_t absXSigned = static_cast<std::int64_t>(grain.originX) + static_cast<std::int64_t>(x);
    const std::int64_t absYSigned = static_cast<std::int64_t>(grain.originY) + static_cast<std::int64_t>(y);
    const std::uint64_t absX = static_cast<std::uint64_t>(absXSigned);
    const std::uint64_t absY = static_cast<std::uint64_t>(absYSigned);

    density += densityMin;

    const int useStbn = (grain.stbn && grain.stbnWidth > 0 && grain.stbnHeight > 0 && grain.stbnFrames > 0) ? 1 : 0;
    const float clumpFactor = grain_clump_factor_device(grain, absX, absY);
    const float wCoarse = fminf(fmaxf(mixWeight, 0.0f), 1.0f);
    const float wFine = 1.0f - wCoarse;
    const bool useMix = (wCoarse > 0.0f) && (mixScale > 1.0f);

    float acc = 0.0f;
    float accNext = 0.0f;
    for (int sl = 0; sl < nSubLayers; ++sl) {
        const std::uint64_t seed = grain.seedBase ^ (static_cast<std::uint64_t>(channelIndex) + static_cast<std::uint64_t>(sl) * 10ULL);
        const std::uint64_t seedNext = grain.seedBaseNext ^ (static_cast<std::uint64_t>(channelIndex) + static_cast<std::uint64_t>(sl) * 10ULL);
        LayerParticleModelSampleDevice sample{};
        sample.density = density;
        sample.densityMaximum = densityMax;
        sample.uniformity = uniformity;
        sample.absoluteX = absX;
        sample.absoluteY = absY;
        sample.grain = &grain;
        sample.useStbn = useStbn;
        if (!useMix) {
            sample.particleCount = nParticles;
            sample.particleOpticalDensity = odParticle;
            sample.seed = seed;
            sample.frameOffset = 0;
            acc += layer_particle_model_device(sample);
            if (wantNext) {
                sample.seed = seedNext;
                sample.frameOffset = 1;
                accNext += layer_particle_model_device(sample);
            }
        } else {
            if (wFine > 0.0f) {
                sample.particleCount = nParticles;
                sample.particleOpticalDensity = odParticle * wFine;
                sample.seed = seed;
                sample.frameOffset = 0;
                acc += layer_particle_model_device(sample);
                if (wantNext) {
                    sample.seed = seedNext;
                    sample.frameOffset = 1;
                    accNext += layer_particle_model_device(sample);
                }
            }
            const float nParticlesCoarse = nParticles / mixScale;
            if (nParticlesCoarse > 0.0f) {
                sample.particleCount = nParticlesCoarse;
                sample.particleOpticalDensity = odParticle * wCoarse * mixScale;
                sample.seed = seed;
                sample.frameOffset = 0;
                acc += layer_particle_model_device(sample);
                if (wantNext) {
                    sample.seed = seedNext;
                    sample.frameOffset = 1;
                    accNext += layer_particle_model_device(sample);
                }
            }
        }
    }
    acc /= static_cast<float>(nSubLayers);
    if (wantNext) {
        accNext /= static_cast<float>(nSubLayers);
    }
    const float breathingFactor = grain_breathing_factor_device(grain, absX, absY);
    acc = density + (acc - density) * breathingFactor * clumpFactor;
    if (wantNext) {
        accNext = density + (accNext - density) * breathingFactor * clumpFactor;
    }
    if (useRetime) {
        const float delta0 = acc - density;
        const float delta1 = accNext - density;
        const float blend = delta0 + (delta1 - delta0) * timeAlpha;
        const float w0 = 1.0f - timeAlpha;
        const float denom = w0 * w0 + timeAlpha * timeAlpha + 1e-6f;
        const float norm = rsqrtf(denom);
        acc = density + blend * norm;
    }
    outGrain[idx] = device_isfinite(acc) ? acc : 0.0f;
}

__global__ void grain_layer_kernel(
    JuicerCuda::GrainPayload grain,
    int width,
    int height,
    const float* inDensity,
    float* outGrain,
    int channelIndex,
    int sublayerIndex) {
    if (width <= 0 || height <= 0) {
        return;
    }
    const std::size_t x = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    const std::size_t y = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(blockDim.y) + static_cast<std::size_t>(threadIdx.y);
    if (x >= static_cast<std::size_t>(width) || y >= static_cast<std::size_t>(height)) {
        return;
    }
    if (!inDensity || !outGrain) {
        return;
    }
    if (channelIndex < 0 || channelIndex > 2) {
        return;
    }
    if (sublayerIndex < 0 || sublayerIndex > 2) {
        return;
    }

    const std::size_t idx = y * static_cast<std::size_t>(width) + x;
    float density = inDensity[idx];

    const float densityMin = grain.densityMinLayers[sublayerIndex][channelIndex];
    const float densityMax = grain.densityMaxLayers[sublayerIndex][channelIndex];
    const float nParticles = grain.nParticlesLayers[sublayerIndex][channelIndex];
    const float odParticle = grain.odParticleLayers[sublayerIndex][channelIndex];
    const float uniformity = grain.uniformity[channelIndex];
    float mixWeight = grain.sizeMixWeight;
    float mixScale = grain.sizeMixScale;
    const float timeAlpha = grain.timeAlpha;
    const bool useRetime = (timeAlpha > 1e-6f && timeAlpha < 0.999999f);
    const bool wantNext = useRetime;

    if (!device_isfinite(densityMax) || !(densityMax > 0.0f) ||
        !device_isfinite(nParticles) || !(nParticles > 0.0f) ||
        !device_isfinite(odParticle) || !(odParticle > 0.0f)) {
        outGrain[idx] = 0.0f;
        return;
    }

    const std::int64_t absXSigned = static_cast<std::int64_t>(grain.originX) + static_cast<std::int64_t>(x);
    const std::int64_t absYSigned = static_cast<std::int64_t>(grain.originY) + static_cast<std::int64_t>(y);
    const std::uint64_t absX = static_cast<std::uint64_t>(absXSigned);
    const std::uint64_t absY = static_cast<std::uint64_t>(absYSigned);

    const JuicerCuda::DeviceCurveView curve = grain.densityCurveCmy[channelIndex];
    const float* layerCurve = grain.densityCurvesLayers[sublayerIndex][channelIndex];
    density = interp_density_layer_device(
        density,
        curve.y,
        layerCurve,
        curve.n,
        grain.positiveFilm != 0);
    density += densityMin;

    const std::uint64_t seed = grain.seedBase ^ (static_cast<std::uint64_t>(channelIndex) + static_cast<std::uint64_t>(sublayerIndex) * 10ULL);
    const int useStbn = (grain.stbn && grain.stbnWidth > 0 && grain.stbnHeight > 0 && grain.stbnFrames > 0) ? 1 : 0;
    const float clumpFactor = grain_clump_factor_device(grain, absX, absY);
    const float wCoarse = fminf(fmaxf(mixWeight, 0.0f), 1.0f);
    const float wFine = 1.0f - wCoarse;
    const bool useMix = (wCoarse > 0.0f) && (mixScale > 1.0f);

    float grainSample = 0.0f;
    float grainSampleNext = 0.0f;
    LayerParticleModelSampleDevice sample{};
    sample.density = density;
    sample.densityMaximum = densityMax;
    sample.uniformity = uniformity;
    sample.absoluteX = absX;
    sample.absoluteY = absY;
    sample.grain = &grain;
    sample.useStbn = useStbn;
    if (!useMix) {
        sample.particleCount = nParticles;
        sample.particleOpticalDensity = odParticle;
        sample.seed = seed;
        sample.frameOffset = 0;
        grainSample = layer_particle_model_device(sample);
        if (wantNext) {
            const std::uint64_t seedNext = grain.seedBaseNext ^ (static_cast<std::uint64_t>(channelIndex) + static_cast<std::uint64_t>(sublayerIndex) * 10ULL);
            sample.seed = seedNext;
            sample.frameOffset = 1;
            grainSampleNext = layer_particle_model_device(sample);
        }
    } else {
        if (wFine > 0.0f) {
            sample.particleCount = nParticles;
            sample.particleOpticalDensity = odParticle * wFine;
            sample.seed = seed;
            sample.frameOffset = 0;
            grainSample += layer_particle_model_device(sample);
            if (wantNext) {
                const std::uint64_t seedNext = grain.seedBaseNext ^ (static_cast<std::uint64_t>(channelIndex) + static_cast<std::uint64_t>(sublayerIndex) * 10ULL);
                sample.seed = seedNext;
                sample.frameOffset = 1;
                grainSampleNext += layer_particle_model_device(sample);
            }
        }
        const float nParticlesCoarse = nParticles / mixScale;
        if (nParticlesCoarse > 0.0f) {
            sample.particleCount = nParticlesCoarse;
            sample.particleOpticalDensity = odParticle * wCoarse * mixScale;
            sample.seed = seed;
            sample.frameOffset = 0;
            grainSample += layer_particle_model_device(sample);
            if (wantNext) {
                const std::uint64_t seedNext = grain.seedBaseNext ^ (static_cast<std::uint64_t>(channelIndex) + static_cast<std::uint64_t>(sublayerIndex) * 10ULL);
                sample.seed = seedNext;
                sample.frameOffset = 1;
                grainSampleNext += layer_particle_model_device(sample);
            }
        }
    }
    const float breathingFactor = grain_breathing_factor_device(grain, absX, absY);
    grainSample = density + (grainSample - density) * breathingFactor * clumpFactor;
    if (wantNext) {
        grainSampleNext = density + (grainSampleNext - density) * breathingFactor * clumpFactor;
    }
    if (useRetime) {
        const float delta0 = grainSample - density;
        const float delta1 = grainSampleNext - density;
        const float blend = delta0 + (delta1 - delta0) * timeAlpha;
        const float w0 = 1.0f - timeAlpha;
        const float denom = w0 * w0 + timeAlpha * timeAlpha + 1e-6f;
        const float norm = rsqrtf(denom);
        grainSample = density + blend * norm;
    }
    outGrain[idx] = device_isfinite(grainSample) ? grainSample : 0.0f;
}

__global__ void develop_film_density_from_raw_kernel(
    JuicerCuda::PipelineRunParams params,
    const float* inB,
    const float* inG,
    const float* inR,
    float* outC,
    float* outM,
    float* outY) {
    const JuicerCuda::FilmDevelopPayload& dev = params.filmDevelop;

    if (!inB || !inG || !inR || !outC || !outM || !outY) {
        return;
    }

    const int nC = params.nComponents;
    if (!(nC == 3 || nC == 4)) {
        return;
    }
    if (params.width <= 0 || params.height <= 0) {
        return;
    }

    const bool useSpatialDir = juicer_cuda_spatial_dir_filtered_correction_active_device(dev);
    const std::size_t width = static_cast<std::size_t>(params.width);
    const std::size_t height = static_cast<std::size_t>(params.height);
    const std::size_t yStep = static_cast<std::size_t>(blockDim.y) * static_cast<std::size_t>(gridDim.y);
    const std::size_t xStep = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    for (std::size_t y = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(blockDim.y) + static_cast<std::size_t>(threadIdx.y); y < height; y += yStep) {
        for (std::size_t x = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x); x < width; x += xStep) {
            const std::size_t idx = y * width + x;
            const float filmRaw[3] = {inB[idx], inG[idx], inR[idx]};

            float D_cmy[3] = {0.0f, 0.0f, 0.0f};
            if (useSpatialDir) {
                float logE_raw[3] = {0.0f, 0.0f, 0.0f};
                if (juicer_cuda_spatial_dir_cached_log_raw_active_device(dev)) {
                    juicer_cuda_load_spatial_dir_cached_log_raw_device(dev, idx, logE_raw);
                } else {
                    float logE_sanitized[3] = {0.0f, 0.0f, 0.0f};
                    float layerPre[3] = {0.0f, 0.0f, 0.0f};
                    FilmDevelopIntermediatesDevice intermediates{};
                    intermediates.logERaw = logE_raw;
                    intermediates.logESanitized = logE_sanitized;
                    intermediates.layerPre = layerPre;
                    compute_logE_from_film_raw_device(params, filmRaw, intermediates);
                }
                juicer_cuda_develop_dir_final_device(dev, logE_raw, idx, D_cmy);
            } else {
                float logE_raw[3] = {0.0f, 0.0f, 0.0f};
                float logE_sanitized[3] = {0.0f, 0.0f, 0.0f};
                float layerPre[3] = {0.0f, 0.0f, 0.0f};
                FilmDevelopIntermediatesDevice intermediates{};
                intermediates.logERaw = logE_raw;
                intermediates.logESanitized = logE_sanitized;
                intermediates.layerPre = layerPre;
                compute_logE_from_film_raw_device(params, filmRaw, intermediates);

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
                    D_cmy[0] = layerPre[2];
                    D_cmy[1] = layerPre[1];
                    D_cmy[2] = layerPre[0];
                }
            }

            outC[idx] = D_cmy[0];
            outM[idx] = D_cmy[1];
            outY[idx] = D_cmy[2];
        }
    }
}
