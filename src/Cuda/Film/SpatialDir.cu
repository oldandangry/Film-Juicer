#include "Cuda/Film/FilmLaunchers.h"
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "Cuda/JuicerCudaDeviceHelpers.cuh"
#include "Cuda/JuicerCudaPayloads.h"

#include "Cuda/Film/CameraFilmLinear.h"

namespace {

    using JuicerCuda::valid_camera_film_linear_planes;

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

    struct DirRawCorrectionOutputs {
        float* correctionY = nullptr;
        float* correctionM = nullptr;
        float* correctionC = nullptr;
        float* logRawB = nullptr;
        float* logRawG = nullptr;
        float* logRawR = nullptr;
    };

    __device__ __forceinline__ bool camera_film_linear_planes_active_device(
        const JuicerCuda::CameraFilmLinearExposurePlanes& planes) {
        return planes.redSensitive &&
               planes.greenSensitive &&
               planes.blueSensitive;
    }

    __device__ __forceinline__ void load_camera_film_linear_bgr_device(
        const JuicerCuda::CameraFilmLinearExposurePlanes& planes,
        int x,
        int y,
        float filmRawBgr[3]) {
        const std::size_t offset =
            static_cast<std::size_t>(y) * planes.rowStrideFloats +
            static_cast<std::size_t>(x);
        filmRawBgr[0] = ldg_f(planes.blueSensitive + offset);
        filmRawBgr[1] = ldg_f(planes.greenSensitive + offset);
        filmRawBgr[2] = ldg_f(planes.redSensitive + offset);
    }

    template <typename Params>
    __global__ void dir_raw_correction_source_build_kernel(
        Params params,
        DirRawCorrectionOutputs outputs,
        JuicerCuda::CameraFilmLinearExposurePlanes cameraFilmLinear) {
        const bool useCameraFilmLinear =
            camera_film_linear_planes_active_device(cameraFilmLinear);
        if (!useCameraFilmLinear &&
            (!params.src || params.srcRowBytes == 0)) {
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
            for (int x = xStart; x < params.width; x += xStep) {
                const float* srcPix = nullptr;
                if (!useCameraFilmLinear) {
                    const char* srcRow =
                        reinterpret_cast<const char*>(params.src) +
                        static_cast<std::size_t>(y) * params.srcRowBytes;
                    srcPix = reinterpret_cast<const float*>(
                        srcRow + static_cast<std::size_t>(x) * pixelBytes);
                }

                float logE_raw[3] = {0.0f, 0.0f, 0.0f};
                float logE_sanitized[3] = {0.0f, 0.0f, 0.0f};
                float layerPre[3] = {0.0f, 0.0f, 0.0f};
                FilmDevelopIntermediatesDevice intermediates{};
                intermediates.logERaw = logE_raw;
                intermediates.logESanitized = logE_sanitized;
                intermediates.layerPre = layerPre;
                if (useCameraFilmLinear) {
                    float filmRawBgr[3] = {0.0f, 0.0f, 0.0f};
                    load_camera_film_linear_bgr_device(
                        cameraFilmLinear,
                        x,
                        y,
                        filmRawBgr);
                    compute_logE_from_camera_film_linear_exposure_device(
                        params,
                        filmRawBgr,
                        intermediates);
                } else {
                    const float rgbIn[3] = {
                        srcPix[0],
                        srcPix[1],
                        srcPix[2]};
                    compute_logE_and_layer_pre_device(
                        params,
                        rgbIn,
                        intermediates);
                }

                const float D_cmy[3] = {layerPre[2], layerPre[1], layerPre[0]};
                const float layerDensities[3] = {D_cmy[2], D_cmy[1], D_cmy[0]};

                float outCorr[3] = {0.0f, 0.0f, 0.0f};
                const unsigned int failures = compute_dir_corrections_device(
                    params.filmDevelop.dir,
                    layerDensities,
                    outCorr);
                signal_dir_failure_device(params.scanStage.scanErrorFlag, failures);

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
        float* rawCorrection,
        JuicerCuda::CameraFilmLinearExposurePlanes cameraFilmLinear) {
        const bool useCameraFilmLinear =
            camera_film_linear_planes_active_device(cameraFilmLinear);
        if ((!useCameraFilmLinear &&
             (!params.src || params.srcRowBytes == 0)) ||
            !rawCorrection) {
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
            for (int x = xStart; x < params.width; x += xStep) {
                const float* srcPix = nullptr;
                if (!useCameraFilmLinear) {
                    const char* srcRow =
                        reinterpret_cast<const char*>(params.src) +
                        static_cast<std::size_t>(y) * params.srcRowBytes;
                    srcPix = reinterpret_cast<const float*>(
                        srcRow + static_cast<std::size_t>(x) * pixelBytes);
                }

                float logE_raw[3] = {0.0f, 0.0f, 0.0f};
                float logE_sanitized[3] = {0.0f, 0.0f, 0.0f};
                float layerPre[3] = {0.0f, 0.0f, 0.0f};
                FilmDevelopIntermediatesDevice intermediates{};
                intermediates.logERaw = logE_raw;
                intermediates.logESanitized = logE_sanitized;
                intermediates.layerPre = layerPre;
                if (useCameraFilmLinear) {
                    float filmRawBgr[3] = {0.0f, 0.0f, 0.0f};
                    load_camera_film_linear_bgr_device(
                        cameraFilmLinear,
                        x,
                        y,
                        filmRawBgr);
                    compute_logE_from_camera_film_linear_exposure_device(
                        params,
                        filmRawBgr,
                        intermediates);
                } else {
                    const float rgbIn[3] = {
                        srcPix[0],
                        srcPix[1],
                        srcPix[2]};
                    compute_logE_and_layer_pre_device(
                        params,
                        rgbIn,
                        intermediates);
                }

                const float D_cmy[3] = {layerPre[2], layerPre[1], layerPre[0]};
                const float layerDensities[3] = {D_cmy[2], D_cmy[1], D_cmy[0]};

                float outCorr[3] = {0.0f, 0.0f, 0.0f};
                const unsigned int failures = compute_dir_corrections_device(
                    params.filmDevelop.dir,
                    layerDensities,
                    outCorr);
                signal_dir_failure_device(params.scanStage.scanErrorFlag, failures);

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
        float* logRawR,
        JuicerCuda::CameraFilmLinearExposurePlanes cameraFilmLinear) {
        const bool useCameraFilmLinear =
            camera_film_linear_planes_active_device(cameraFilmLinear);
        if ((!useCameraFilmLinear &&
             (!params.src || params.srcRowBytes == 0)) ||
            !logRawB || !logRawG || !logRawR) {
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
            for (int x = xStart; x < params.width; x += xStep) {
                const float* srcPix = nullptr;
                if (!useCameraFilmLinear) {
                    const char* srcRow =
                        reinterpret_cast<const char*>(params.src) +
                        static_cast<std::size_t>(y) * params.srcRowBytes;
                    srcPix = reinterpret_cast<const float*>(
                        srcRow + static_cast<std::size_t>(x) * pixelBytes);
                }

                float logE_raw[3] = {0.0f, 0.0f, 0.0f};
                float logE_sanitized[3] = {0.0f, 0.0f, 0.0f};
                float layerPre[3] = {0.0f, 0.0f, 0.0f};
                FilmDevelopIntermediatesDevice intermediates{};
                intermediates.logERaw = logE_raw;
                intermediates.logESanitized = logE_sanitized;
                intermediates.layerPre = layerPre;
                if (useCameraFilmLinear) {
                    float filmRawBgr[3] = {0.0f, 0.0f, 0.0f};
                    load_camera_film_linear_bgr_device(
                        cameraFilmLinear,
                        x,
                        y,
                        filmRawBgr);
                    compute_logE_from_camera_film_linear_exposure_device(
                        params,
                        filmRawBgr,
                        intermediates);
                } else {
                    const float rgbIn[3] = {
                        srcPix[0],
                        srcPix[1],
                        srcPix[2]};
                    compute_logE_and_layer_pre_device(
                        params,
                        rgbIn,
                        intermediates);
                }

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

    enum class SpatialDirAccumulationMode : std::uint8_t {
        Initialize,
        Accumulate
    };

    __global__ void spatial_dir_identity_accumulate_kernel(
        const float* source,
        float weight,
        float* destination,
        std::size_t count,
        SpatialDirAccumulationMode mode) {
        const std::size_t index =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= count) {
            return;
        }
        const float weighted = source[index] * weight;
        destination[index] = mode == SpatialDirAccumulationMode::Initialize
                                 ? weighted
                                 : destination[index] + weighted;
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
        JuicerCuda::SpatialDirIirCoefficients coefficients,
        JuicerCuda::SpatialDirBoundarySpec boundary) {
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
        const double anchor = static_cast<double>(input[row]);
        double state[3] = {anchor, anchor, anchor};
        for (int sample = 0; sample < boundary.initialWeightLength; ++sample) {
            const double delta =
                static_cast<double>(input[row + static_cast<std::size_t>(sample)]) - anchor;
#pragma unroll
            for (int history = 0; history < 3; ++history) {
                state[history] +=
                    boundary.initialWeights[history * boundary.initialWeightLength + sample] *
                    delta;
            }
        }
        for (int x = 0; x < width; ++x) {
            const size_t index = row + static_cast<size_t>(x);
            const double value =
                coefficients.feedforward * static_cast<double>(input[index]) +
                coefficients.feedback[0] * state[0] +
                coefficients.feedback[1] * state[1] +
                coefficients.feedback[2] * state[2];
            output[index] = static_cast<float>(value);
            state[2] = state[1];
            state[1] = state[0];
            state[0] = value;
        }
        const int terminalSize = boundary.terminalSize;
        double terminal[3] = {0.0, 0.0, 0.0};
        const double last = state[0];
        for (int terminalRow = 0; terminalRow < terminalSize; ++terminalRow) {
            double value = last;
            for (int terminalColumn = 0; terminalColumn < terminalSize; ++terminalColumn) {
                value += boundary.terminalMatrix[terminalRow * terminalSize + terminalColumn] *
                         (state[terminalSize - 1 - terminalColumn] - last);
            }
            terminal[terminalRow] = value;
            output[row + static_cast<std::size_t>(
                             width - terminalSize + terminalRow)] =
                static_cast<float>(value);
        }
        for (int history = 0; history < terminalSize; ++history) {
            state[history] = terminal[history];
        }
        for (int x = width - terminalSize - 1; x >= 0; --x) {
            const size_t index = row + static_cast<size_t>(x);
            const double value =
                coefficients.feedforward * static_cast<double>(output[index]) +
                coefficients.feedback[0] * state[0] +
                coefficients.feedback[1] * state[1] +
                coefficients.feedback[2] * state[2];
            output[index] = static_cast<float>(value);
            state[2] = state[1];
            state[1] = state[0];
            state[0] = value;
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
        JuicerCuda::SpatialDirIirCoefficients coefficients,
        JuicerCuda::SpatialDirBoundarySpec boundary,
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
        const double anchor = static_cast<double>(input[visualTop]);
        double state[3] = {anchor, anchor, anchor};
        for (int sample = 0; sample < boundary.initialWeightLength; ++sample) {
            const size_t index = static_cast<size_t>(height - 1 - sample) *
                                     static_cast<size_t>(width) +
                                 static_cast<size_t>(x);
            const double delta = static_cast<double>(input[index]) - anchor;
#pragma unroll
            for (int history = 0; history < 3; ++history) {
                state[history] +=
                    boundary.initialWeights[history * boundary.initialWeightLength + sample] *
                    delta;
            }
        }
        for (int visualY = 0; visualY < height; ++visualY) {
            const int y = height - 1 - visualY;
            const size_t index = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            const double value =
                coefficients.feedforward * static_cast<double>(input[index]) +
                coefficients.feedback[0] * state[0] +
                coefficients.feedback[1] * state[1] +
                coefficients.feedback[2] * state[2];
            forwardTemp[index] = static_cast<float>(value);
            state[2] = state[1];
            state[1] = state[0];
            state[0] = value;
        }
        const int terminalSize = boundary.terminalSize;
        double terminal[3] = {0.0, 0.0, 0.0};
        const double last = state[0];
        for (int terminalRow = 0; terminalRow < terminalSize; ++terminalRow) {
            double value = last;
            for (int terminalColumn = 0; terminalColumn < terminalSize; ++terminalColumn) {
                value += boundary.terminalMatrix[terminalRow * terminalSize + terminalColumn] *
                         (state[terminalSize - 1 - terminalColumn] - last);
            }
            terminal[terminalRow] = value;
            const int visualY = height - terminalSize + terminalRow;
            const int y = height - 1 - visualY;
            const size_t index = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            const float weighted = static_cast<float>(value) * weight;
            inOut[index] = initialize ? weighted : inOut[index] + weighted;
        }
        for (int history = 0; history < terminalSize; ++history) {
            state[history] = terminal[history];
        }
        for (int visualY = height - terminalSize - 1; visualY >= 0; --visualY) {
            const int y = height - 1 - visualY;
            const size_t index = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            const double value =
                coefficients.feedforward * static_cast<double>(forwardTemp[index]) +
                coefficients.feedback[0] * state[0] +
                coefficients.feedback[1] * state[1] +
                coefficients.feedback[2] * state[2];
            const float weighted = static_cast<float>(value) * weight;
            inOut[index] = initialize ? weighted : inOut[index] + weighted;
            state[2] = state[1];
            state[1] = state[0];
            state[0] = value;
        }
    }

    __global__ void spatial_dir_iir_horizontal_single_kernel(
        const float* input,
        float* output,
        int width,
        int height,
        JuicerCuda::SpatialDirIirCoefficients coefficients,
        JuicerCuda::SpatialDirBoundarySpec boundary) {
        const int y = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        if (y >= height || width <= 0 || !input || !output) {
            return;
        }
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(width);
        const double anchor = static_cast<double>(input[row]);
        double state[3] = {anchor, anchor, anchor};
        for (int sample = 0; sample < boundary.initialWeightLength; ++sample) {
            const double delta =
                static_cast<double>(input[row + static_cast<std::size_t>(sample)]) - anchor;
#pragma unroll
            for (int history = 0; history < 3; ++history) {
                state[history] +=
                    boundary.initialWeights[history * boundary.initialWeightLength + sample] *
                    delta;
            }
        }
        for (int x = 0; x < width; ++x) {
            const size_t index = row + static_cast<size_t>(x);
            const double value =
                coefficients.feedforward * static_cast<double>(input[index]) +
                coefficients.feedback[0] * state[0] +
                coefficients.feedback[1] * state[1] +
                coefficients.feedback[2] * state[2];
            output[index] = static_cast<float>(value);
            state[2] = state[1];
            state[1] = state[0];
            state[0] = value;
        }
        const int terminalSize = boundary.terminalSize;
        double terminal[3] = {0.0, 0.0, 0.0};
        const double last = state[0];
        for (int terminalRow = 0; terminalRow < terminalSize; ++terminalRow) {
            double value = last;
            for (int terminalColumn = 0; terminalColumn < terminalSize; ++terminalColumn) {
                value += boundary.terminalMatrix[terminalRow * terminalSize + terminalColumn] *
                         (state[terminalSize - 1 - terminalColumn] - last);
            }
            terminal[terminalRow] = value;
            output[row + static_cast<std::size_t>(
                             width - terminalSize + terminalRow)] =
                static_cast<float>(value);
        }
        for (int history = 0; history < terminalSize; ++history) {
            state[history] = terminal[history];
        }
        for (int x = width - terminalSize - 1; x >= 0; --x) {
            const size_t index = row + static_cast<size_t>(x);
            const double value =
                coefficients.feedforward * static_cast<double>(output[index]) +
                coefficients.feedback[0] * state[0] +
                coefficients.feedback[1] * state[1] +
                coefficients.feedback[2] * state[2];
            output[index] = static_cast<float>(value);
            state[2] = state[1];
            state[1] = state[0];
            state[0] = value;
        }
    }

    __global__ void spatial_dir_iir_vertical_accumulate_single_kernel(
        const float* input,
        float* forwardTemp,
        float* inOut,
        int width,
        int height,
        JuicerCuda::SpatialDirIirCoefficients coefficients,
        JuicerCuda::SpatialDirBoundarySpec boundary,
        float weight,
        int initialize) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        if (x >= width || height <= 0 || !input || !forwardTemp || !inOut) {
            return;
        }
        const size_t visualTop =
            static_cast<size_t>(height - 1) * static_cast<size_t>(width) + static_cast<size_t>(x);
        const double anchor = static_cast<double>(input[visualTop]);
        double state[3] = {anchor, anchor, anchor};
        for (int sample = 0; sample < boundary.initialWeightLength; ++sample) {
            const size_t index = static_cast<size_t>(height - 1 - sample) *
                                     static_cast<size_t>(width) +
                                 static_cast<size_t>(x);
            const double delta = static_cast<double>(input[index]) - anchor;
#pragma unroll
            for (int history = 0; history < 3; ++history) {
                state[history] +=
                    boundary.initialWeights[history * boundary.initialWeightLength + sample] *
                    delta;
            }
        }
        for (int visualY = 0; visualY < height; ++visualY) {
            const int y = height - 1 - visualY;
            const size_t index = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            const double value =
                coefficients.feedforward * static_cast<double>(input[index]) +
                coefficients.feedback[0] * state[0] +
                coefficients.feedback[1] * state[1] +
                coefficients.feedback[2] * state[2];
            forwardTemp[index] = static_cast<float>(value);
            state[2] = state[1];
            state[1] = state[0];
            state[0] = value;
        }
        const int terminalSize = boundary.terminalSize;
        double terminal[3] = {0.0, 0.0, 0.0};
        const double last = state[0];
        for (int terminalRow = 0; terminalRow < terminalSize; ++terminalRow) {
            double value = last;
            for (int terminalColumn = 0; terminalColumn < terminalSize; ++terminalColumn) {
                value += boundary.terminalMatrix[terminalRow * terminalSize + terminalColumn] *
                         (state[terminalSize - 1 - terminalColumn] - last);
            }
            terminal[terminalRow] = value;
            const int visualY = height - terminalSize + terminalRow;
            const int y = height - 1 - visualY;
            const size_t index = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            const float weighted = static_cast<float>(value) * weight;
            inOut[index] = initialize ? weighted : inOut[index] + weighted;
        }
        for (int history = 0; history < terminalSize; ++history) {
            state[history] = terminal[history];
        }
        for (int visualY = height - terminalSize - 1; visualY >= 0; --visualY) {
            const int y = height - 1 - visualY;
            const size_t index = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            const double value =
                coefficients.feedforward * static_cast<double>(forwardTemp[index]) +
                coefficients.feedback[0] * state[0] +
                coefficients.feedback[1] * state[1] +
                coefficients.feedback[2] * state[2];
            const float weighted = static_cast<float>(value) * weight;
            inOut[index] = initialize ? weighted : inOut[index] + weighted;
            state[2] = state[1];
            state[1] = state[0];
            state[0] = value;
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
    const float gaussianWeight = request.gaussian.weight;
    void* cudaStreamOpaque = request.streamOpaque;
    if (params.width <= 0 || params.height <= 0) {
        return cudaSuccess;
    }
    const bool anyCameraFilmLinearPlane =
        request.cameraFilmLinear.redSensitive ||
        request.cameraFilmLinear.greenSensitive ||
        request.cameraFilmLinear.blueSensitive;
    const bool useCameraFilmLinear = valid_camera_film_linear_planes(
        request.cameraFilmLinear,
        params.width,
        params.height);
    if ((anyCameraFilmLinearPlane && !useCameraFilmLinear) ||
        (!useCameraFilmLinear &&
         (!params.src || params.srcRowBytes == 0 ||
          !(params.nComponents == 3 || params.nComponents == 4)))) {
        return cudaErrorInvalidValue;
    }
    const bool haveComponentStreamedYvvScratch =
        rawCorrectionY && !rawCorrectionM && !rawCorrectionC &&
        filteredCorrectionY && filteredCorrectionM && filteredCorrectionC && filterTemp &&
        !filterTempM && !filterTempC;
    const bool haveComponentStreamedScratch = haveComponentStreamedYvvScratch;
    if (!rawCorrectionY || (!haveComponentStreamedScratch && (!rawCorrectionM || !rawCorrectionC)) ||
        !filteredCorrectionY || !filteredCorrectionM || !filteredCorrectionC ||
        !filterTemp) {
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

    const JuicerCuda::SpatialDirFilterSpec* componentSpecs[4] = {
        &request.gaussian,
        &request.tails[0],
        &request.tails[1],
        &request.tails[2]};
    bool anyActiveComponent = false;
    for (int componentIndex = 0; componentIndex < 4; ++componentIndex) {
        const JuicerCuda::SpatialDirFilterSpec& component =
            *componentSpecs[componentIndex];
        if (!std::isfinite(component.weight) || component.weight < 0.0f) {
            return cudaErrorInvalidValue;
        }
        if (!(component.weight > 0.0f)) {
            continue;
        }
        anyActiveComponent = true;
        if (!std::isfinite(component.sigma) || component.sigma < 0.0f) {
            return cudaErrorInvalidValue;
        }
        switch (component.filterOperator) {
            case JuicerCuda::SpatialDirFilterOperator::Identity:
                if (component.radius != 0 || component.kernel) {
                    return cudaErrorInvalidValue;
                }
                break;
            case JuicerCuda::SpatialDirFilterOperator::FirReflect:
                if (!(component.sigma > 0.0f && component.sigma < 3.0f) ||
                    component.radius <= 0 || !component.kernel) {
                    return cudaErrorInvalidValue;
                }
                break;
            case JuicerCuda::SpatialDirFilterOperator::YvvReflect: {
                const double dc = component.iir.feedforward +
                                  component.iir.feedback[0] +
                                  component.iir.feedback[1] +
                                  component.iir.feedback[2];
                const auto validBoundary = [](const JuicerCuda::SpatialDirBoundarySpec& boundary,
                                              int length) {
                    return boundary.initialWeights &&
                           boundary.initialWeightLength > 0 &&
                           boundary.initialWeightLength <= length &&
                           boundary.terminalSize == (length < 3 ? length : 3);
                };
                if (!(component.sigma >= 3.0f) || component.kernel ||
                    !std::isfinite(component.iir.feedforward) ||
                    !(component.iir.feedforward > 0.0) ||
                    !std::isfinite(component.iir.feedback[0]) ||
                    !std::isfinite(component.iir.feedback[1]) ||
                    !std::isfinite(component.iir.feedback[2]) ||
                    !std::isfinite(dc) || std::abs(dc - 1.0) > 1.0e-12 ||
                    !validBoundary(component.horizontalBoundary, params.width) ||
                    !validBoundary(component.verticalBoundary, params.height)) {
                    return cudaErrorInvalidValue;
                }
                break;
            }
            case JuicerCuda::SpatialDirFilterOperator::None:
            default:
                return cudaErrorInvalidValue;
        }
    }
    if (!anyActiveComponent) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
    cudaError_t err = cudaSuccess;
    dim3 threads2D(32, 8);
    dim3 blocks2D(
        static_cast<unsigned int>((params.width + threads2D.x - 1) / threads2D.x),
        static_cast<unsigned int>((params.height + threads2D.y - 1) / threads2D.y));

    auto launch_corrections = [&]() -> cudaError_t {
        DirRawCorrectionOutputs outputs{};
        outputs.correctionY = rawCorrectionY;
        outputs.correctionM = rawCorrectionM;
        outputs.correctionC = rawCorrectionC;
        outputs.logRawB = logRawB;
        outputs.logRawG = logRawG;
        outputs.logRawR = logRawR;
        dir_raw_correction_source_build_kernel<<<blocks2D, threads2D, 0, stream>>>(
            params,
            outputs,
            request.cameraFilmLinear);
        return cudaGetLastError();
    };

    auto launch_streamed_channel_corrections = [&](int channel) -> cudaError_t {
        dir_raw_correction_channel_source_build_kernel<<<blocks2D, threads2D, 0, stream>>>(
            params,
            channel,
            rawCorrectionY,
            request.cameraFilmLinear);
        return cudaGetLastError();
    };

    if (!haveComponentStreamedScratch) {
        err = launch_corrections();
        if (err != cudaSuccess) {
            return err;
        }
    }

    auto accumulate_fir_plane = [&](const float* rawCorrection, float* filteredCorrection, const float* k, int r, float sigma, // NOLINT(bugprone-easily-swappable-parameters)
                                    float weight,
                                    bool initialize)
        -> cudaError_t {
        if (!rawCorrection || !filteredCorrection || !(sigma > 0.0f) ||
            !(sigma < 3.0f) || !(weight >= 0.0f)) {
            return cudaErrorInvalidValue;
        }
        if (!k || r <= 0) {
            return cudaErrorInvalidValue;
        }
        spatial_dir_blur_horizontal_reflect_kernel<<<blocks2D, threads2D, 0, stream>>>(
            rawCorrection, filterTemp, params.width, params.height, k, r);
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
        return cudaGetLastError();
    };

    auto accumulate_yvv_channels = [&](const float* raw0, const float* raw1, const float* raw2, float* filtered0, float* filtered1, float* filtered2, const JuicerCuda::SpatialDirFilterSpec& component, bool initialize) -> cudaError_t {
        if (!raw0 || !raw1 || !raw2 || !filtered0 || !filtered1 || !filtered2 ||
            !(component.weight >= 0.0f) ||
            !haveAliasedForwardYvvScratch) {
            return cudaErrorInvalidValue;
        }
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
            component.iir,
            component.horizontalBoundary);
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
            component.iir,
            component.verticalBoundary,
            component.weight,
            initialize ? 1 : 0);
        return cudaGetLastError();
    };

    const float* rawCorrections[3] = {rawCorrectionY, rawCorrectionM, rawCorrectionC};
    float* filteredCorrections[3] = {filteredCorrectionY, filteredCorrectionM, filteredCorrectionC};
    auto accumulate_yvv_low_scratch_pair = [&](const JuicerCuda::SpatialDirFilterSpec& component, bool initialize) -> cudaError_t {
        if (!rawCorrectionY || !rawCorrectionM || !rawCorrectionC ||
            !filteredCorrectionY || !filteredCorrectionM || !filteredCorrectionC ||
            !(component.weight >= 0.0f) || !haveLowScratchPairYvvScratch) {
            return cudaErrorInvalidValue;
        }
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
            component.iir,
            component.horizontalBoundary);
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
            component.iir,
            component.verticalBoundary,
            component.weight,
            initialize ? 1 : 0);
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
            component.iir,
            component.horizontalBoundary);
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
            component.iir,
            component.verticalBoundary,
            component.weight,
            initialize ? 1 : 0);
        return cudaGetLastError();
    };

    auto accumulate_yvv_sequential_single_channel = [&](const JuicerCuda::SpatialDirFilterSpec& component, bool initialize) -> cudaError_t {
        if (!(component.weight >= 0.0f) ||
            !haveSingleTempSequentialYvvScratch) {
            return cudaErrorInvalidValue;
        }
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
                component.iir,
                component.horizontalBoundary);
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
                component.iir,
                component.verticalBoundary,
                component.weight,
                initialize ? 1 : 0);
            e = cudaGetLastError();
            if (e != cudaSuccess) {
                return e;
            }
        }
        return cudaSuccess;
    };

    auto accumulate_yvv_component_streamed = [&](float* filteredCorrection, const JuicerCuda::SpatialDirFilterSpec& component, bool initialize) -> cudaError_t {
        if (!filteredCorrection || !(component.weight >= 0.0f) ||
            !haveComponentStreamedYvvScratch) {
            return cudaErrorInvalidValue;
        }
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
            component.iir,
            component.horizontalBoundary);
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
            component.iir,
            component.verticalBoundary,
            component.weight,
            initialize ? 1 : 0);
        return cudaGetLastError();
    };

    auto accumulate_component = [&](
                                    const JuicerCuda::SpatialDirFilterSpec& component,
                                    bool initialize)
        -> cudaError_t {
        if (component.filterOperator == JuicerCuda::SpatialDirFilterOperator::Identity) {
            const std::size_t count =
                static_cast<std::size_t>(params.width) * params.height;
            constexpr int threads = 256;
            const int blocks = static_cast<int>((count + threads - 1u) / threads);
            for (int channel = 0; channel < 3; ++channel) {
                spatial_dir_identity_accumulate_kernel<<<blocks, threads, 0, stream>>>(
                    rawCorrections[channel],
                    component.weight,
                    filteredCorrections[channel],
                    count,
                    initialize ? SpatialDirAccumulationMode::Initialize
                               : SpatialDirAccumulationMode::Accumulate);
                const cudaError_t componentError = cudaGetLastError();
                if (componentError != cudaSuccess) {
                    return componentError;
                }
            }
            return cudaSuccess;
        }
        if (component.filterOperator == JuicerCuda::SpatialDirFilterOperator::YvvReflect) {
            if (haveAliasedForwardYvvScratch) {
                return accumulate_yvv_channels(
                    rawCorrectionY,
                    rawCorrectionM,
                    rawCorrectionC,
                    filteredCorrectionY,
                    filteredCorrectionM,
                    filteredCorrectionC,
                    component,
                    initialize);
            }
            if (haveLowScratchPairYvvScratch) {
                return accumulate_yvv_low_scratch_pair(
                    component,
                    initialize);
            }
            return accumulate_yvv_sequential_single_channel(
                component,
                initialize);
        }
        if (component.filterOperator != JuicerCuda::SpatialDirFilterOperator::FirReflect) {
            return cudaErrorInvalidValue;
        }
        for (int channel = 0; channel < 3; ++channel) {
            cudaError_t componentErr = accumulate_fir_plane(
                rawCorrections[channel],
                filteredCorrections[channel],
                component.kernel,
                component.radius,
                component.sigma,
                component.weight,
                initialize);
            if (componentErr != cudaSuccess) {
                return componentErr;
            }
        }
        return cudaSuccess;
    };

    auto accumulate_streamed_component = [&](
                                             int channel,
                                             const JuicerCuda::SpatialDirFilterSpec& component,
                                             bool initialize)
        -> cudaError_t {
        if (channel < 0 || channel >= 3 || !(component.weight >= 0.0f)) {
            return cudaErrorInvalidValue;
        }
        float* filteredCorrection = filteredCorrections[static_cast<std::size_t>(channel)];
        if (component.filterOperator == JuicerCuda::SpatialDirFilterOperator::Identity) {
            const std::size_t count =
                static_cast<std::size_t>(params.width) * params.height;
            constexpr int threads = 256;
            const int blocks = static_cast<int>((count + threads - 1u) / threads);
            spatial_dir_identity_accumulate_kernel<<<blocks, threads, 0, stream>>>(
                rawCorrectionY,
                component.weight,
                filteredCorrection,
                count,
                initialize ? SpatialDirAccumulationMode::Initialize
                           : SpatialDirAccumulationMode::Accumulate);
            return cudaGetLastError();
        }
        if (component.filterOperator == JuicerCuda::SpatialDirFilterOperator::YvvReflect) {
            return accumulate_yvv_component_streamed(
                filteredCorrection,
                component,
                initialize);
        }
        if (component.filterOperator != JuicerCuda::SpatialDirFilterOperator::FirReflect) {
            return cudaErrorInvalidValue;
        }
        return accumulate_fir_plane(
            rawCorrectionY,
            filteredCorrection,
            component.kernel,
            component.radius,
            component.sigma,
            component.weight,
            initialize);
    };

    if (haveComponentStreamedYvvScratch) {
        for (int channel = 0; channel < 3; ++channel) {
            err = launch_streamed_channel_corrections(channel);
            if (err != cudaSuccess) {
                return err;
            }
            bool channelAccumulatorInitialized = false;
            if (gaussianWeight > 0.0f) {
                err = accumulate_streamed_component(
                    channel,
                    request.gaussian,
                    true);
                if (err != cudaSuccess) {
                    return err;
                }
                channelAccumulatorInitialized = true;
            }
            for (int component = 0; component < 3; ++component) {
                if (!(request.tails[component].weight > 0.0f)) {
                    continue;
                }
                const bool initializeComponent = !channelAccumulatorInitialized;
                err = accumulate_streamed_component(
                    channel,
                    request.tails[component],
                    initializeComponent);
                if (err != cudaSuccess) {
                    return err;
                }
                channelAccumulatorInitialized = true;
            }
            if (!channelAccumulatorInitialized) {
                return cudaErrorInvalidValue;
            }
        }
        return cudaGetLastError();
    }

    bool accumulatorInitialized = false;
    if (gaussianWeight > 0.0f) {
        err = accumulate_component(
            request.gaussian,
            true);
        if (err != cudaSuccess)
            return err;
        accumulatorInitialized = true;
    }

    for (int component = 0; component < 3; ++component) {
        if (!(request.tails[component].weight > 0.0f)) {
            continue;
        }
        const bool initializeComponent = !accumulatorInitialized;
        err = accumulate_component(
            request.tails[component],
            initializeComponent);
        if (err != cudaSuccess)
            return err;
        accumulatorInitialized = true;
    }
    if (!accumulatorInitialized) {
        return cudaErrorInvalidValue;
    }

    return cudaGetLastError();
}

template <typename Params>
cudaError_t build_spatial_dir_cached_log_raw_impl(
    const Params& params,
    JuicerCuda::CameraFilmLinearExposurePlanes cameraFilmLinear,
    float* logRawB,
    float* logRawG,
    float* logRawR,
    void* cudaStreamOpaque) {
    if (params.width <= 0 || params.height <= 0) {
        return cudaSuccess;
    }
    const bool anyCameraFilmLinearPlane =
        cameraFilmLinear.redSensitive ||
        cameraFilmLinear.greenSensitive ||
        cameraFilmLinear.blueSensitive;
    const bool useCameraFilmLinear = valid_camera_film_linear_planes(
        cameraFilmLinear,
        params.width,
        params.height);
    if (!logRawB || !logRawG || !logRawR ||
        (anyCameraFilmLinearPlane && !useCameraFilmLinear) ||
        (!useCameraFilmLinear &&
         (!params.src || params.srcRowBytes == 0 ||
          !(params.nComponents == 3 || params.nComponents == 4)))) {
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
        logRawR,
        cameraFilmLinear);
    return cudaGetLastError();
}

extern "C" cudaError_t juicer_cuda_build_spatial_dir_cached_log_raw(
    const JuicerCuda::FilmScanRunParams* hParams,
    float* logRawB,
    float* logRawG,
    float* logRawR,
    void* cudaStreamOpaque) {
    if (!hParams) {
        return cudaErrorInvalidValue;
    }
    return build_spatial_dir_cached_log_raw_impl(
        *hParams,
        {},
        logRawB,
        logRawG,
        logRawR,
        cudaStreamOpaque);
}

extern "C" cudaError_t
juicer_cuda_build_spatial_dir_cached_log_raw_from_camera_film_linear(
    const JuicerCuda::FilmScanRunParams* hParams,
    JuicerCuda::CameraFilmLinearExposurePlanes cameraFilmLinear,
    float* logRawB,
    float* logRawG,
    float* logRawR,
    void* cudaStreamOpaque) {
    if (!hParams) {
        return cudaErrorInvalidValue;
    }
    return build_spatial_dir_cached_log_raw_impl(
        *hParams,
        cameraFilmLinear,
        logRawB,
        logRawG,
        logRawR,
        cudaStreamOpaque);
}

extern "C" cudaError_t juicer_cuda_build_spatial_dir(
    const JuicerCuda::FilmScanRunParams* hParams,
    JuicerCuda::SpatialDirBuildRequest request) {
    if (!hParams) {
        return cudaErrorInvalidValue;
    }
    return build_spatial_dir_impl(*hParams, request);
}
