// Cuda/JuicerCudaAutoExposure.cu
//
// CUDA camera auto-exposure metering (center-weighted Gaussian mask).
//
// This is a parity-oriented port of the CPU meter in JuicerEffect.cpp, adapted for device pointers.
//
#include "Cuda/JuicerCudaResources.h"

#include <cuda_runtime.h>

#include <cmath>
#include <string>

#include "Cuda/JuicerCudaDeviceHelpers.cuh"

namespace {

    constexpr int kMedianHistogramBins = 2048;
    constexpr double kCameraMeterTargetY = 0.184;
    constexpr int kMethodCenterWeighted = static_cast<int>(Spektrafilm::AutoExposureMethod::CenterWeighted);
    constexpr int kMethodAverage = static_cast<int>(Spektrafilm::AutoExposureMethod::Average);
    constexpr int kMethodMedian = static_cast<int>(Spektrafilm::AutoExposureMethod::Median);
    constexpr int kMethodPartial = static_cast<int>(Spektrafilm::AutoExposureMethod::Partial);
    constexpr int kMethodMatrix = static_cast<int>(Spektrafilm::AutoExposureMethod::Matrix);
    constexpr int kMethodMultiZone = static_cast<int>(Spektrafilm::AutoExposureMethod::MultiZone);
    constexpr int kMethodHighlightWeighted = static_cast<int>(Spektrafilm::AutoExposureMethod::HighlightWeighted);

    struct PreviewAxisSampling {
        int previewSize = 0;
        int sourceSize = 0;
    };

    struct AutoExposureKernelInput {
        const unsigned char* sourceBase = nullptr;
        std::size_t sourceRowBytes = 0;
        JuicerCuda::AutoExposurePreviewDescriptor descriptor{};
        int nComponents = 0;
        int inputColorSpaceIndex = 0;
        int applyCctfDecoding = 0;
        Mat3 rgbToXYZ{};
    };

    struct MeteringSample {
        int method = kMethodCenterWeighted;
        int x = 0;
        int y = 0;
        int width = 0;
        int height = 0;
        float luminance = 0.0f;
    };

    struct MeteringWeightTables {
        const float* horizontal = nullptr;
        const float* vertical = nullptr;
    };

    struct HistogramBuffers {
        const unsigned int* maxYBits = nullptr;
        unsigned int* histogram = nullptr;
    };

    __device__ __forceinline__ float mulY(const Mat3& m, const float rgb[3]) {
        return m.m[3] * rgb[0] + m.m[4] * rgb[1] + m.m[5] * rgb[2];
    }

    __device__ __forceinline__ int preview_source_coordinate(
        int previewIndex,
        PreviewAxisSampling sampling) {
        if (sampling.previewSize <= 0 || sampling.sourceSize <= 0) {
            return 0;
        }
        const long long numerator =
            (2LL * static_cast<long long>(previewIndex) + 1LL) * static_cast<long long>(sampling.sourceSize);
        const int coordinate = static_cast<int>(numerator / (2LL * static_cast<long long>(sampling.previewSize)));
        return min(sampling.sourceSize - 1, max(0, coordinate));
    }

    __device__ __forceinline__ void metering_weights(
        MeteringSample sample,
        MeteringWeightTables tables,
        float outWeights[JuicerCudaAutoExposurePartial::kLaneCount]) {
        for (int lane = 0; lane < JuicerCudaAutoExposurePartial::kLaneCount; ++lane) {
            outWeights[lane] = 0.0f;
        }
        if (sample.method == kMethodAverage || sample.method == kMethodMedian) {
            outWeights[0] = 1.0f;
            return;
        }
        if (sample.method == kMethodCenterWeighted) {
            outWeights[0] = (tables.horizontal && tables.vertical)
                                ? tables.horizontal[sample.x] * tables.vertical[sample.y]
                                : 0.0f;
            return;
        }
        const int maxDimInt = (sample.width > sample.height) ? sample.width : sample.height;
        const float invMax = maxDimInt > 0 ? 1.0f / static_cast<float>(maxDimInt) : 0.0f;
        const float nx =
            ((static_cast<float>(sample.x) / static_cast<float>(sample.width)) - 0.5f) *
            static_cast<float>(sample.width) * invMax;
        const float ny =
            ((static_cast<float>(sample.y) / static_cast<float>(sample.height)) - 0.5f) *
            static_cast<float>(sample.height) * invMax;
        const float radius = sqrtf(nx * nx + ny * ny);
        if (sample.method == kMethodPartial) {
            outWeights[0] = radius < 0.15f ? 1.0f : 0.0f;
            outWeights[3] = 1.0f;
            return;
        }
        if (sample.method == kMethodMatrix) {
            constexpr int nRows = 5;
            constexpr int nCols = 5;
            const int cellHeight = sample.height / nRows;
            const int cellWidth = sample.width / nCols;
            if (cellHeight <= 0 || cellWidth <= 0) {
                return;
            }
            const int row = sample.y / cellHeight;
            const int col = sample.x / cellWidth;
            if (row >= nRows || col >= nCols) {
                return;
            }
            const float dy = (static_cast<float>(row) - 2.0f) / 2.0f;
            const float dx = (static_cast<float>(col) - 2.0f) / 2.0f;
            const float dist = sqrtf(dx * dx + dy * dy) / sqrtf(2.0f);
            outWeights[0] = 0.5f * (1.0f + cosf(3.14159265358979323846f * dist));
            return;
        }
        if (sample.method == kMethodMultiZone) {
            if (radius >= 0.0f && radius < 0.05f) {
                outWeights[0] = 1.0f;
            } else if (radius >= 0.05f && radius < 0.25f) {
                outWeights[1] = 1.0f;
            } else if (radius >= 0.25f && radius < 0.50f) {
                outWeights[2] = 1.0f;
            }
            outWeights[3] = 1.0f;
            return;
        }
        if (sample.method == kMethodHighlightWeighted) {
            outWeights[0] = sample.luminance * sample.luminance;
            outWeights[3] = 1.0f;
            return;
        }
    }

    __global__ void meter_max_Y_kernel(
        AutoExposureKernelInput input,
        unsigned int* outMaxBits) {
        const JuicerCuda::AutoExposurePreviewDescriptor descriptor = input.descriptor;
        const int width = descriptor.previewWidth;
        const int height = descriptor.previewHeight;
        if (width <= 0 || height <= 0) {
            return;
        }

        PreviewAxisSampling horizontalSampling{};
        horizontalSampling.previewSize = width;
        horizontalSampling.sourceSize = descriptor.meterX2 - descriptor.meterX1;
        PreviewAxisSampling verticalSampling{};
        verticalSampling.previewSize = height;
        verticalSampling.sourceSize = descriptor.meterY2 - descriptor.meterY1;

        InputCctfDecodingDevice decoding{};
        decoding.inputColorSpaceIndex = input.inputColorSpaceIndex;
        decoding.applyCctfDecoding = input.applyCctfDecoding;

        float localMax = 0.0f;
        const std::size_t pixelStrideBytes = static_cast<std::size_t>(input.nComponents) * sizeof(float);
        const std::size_t widthCount = static_cast<std::size_t>(width);
        const std::size_t heightCount = static_cast<std::size_t>(height);
        const std::size_t yStep = static_cast<std::size_t>(blockDim.y) * static_cast<std::size_t>(gridDim.y);
        const std::size_t xStep = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
        for (std::size_t y = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(blockDim.y) + static_cast<std::size_t>(threadIdx.y); y < heightCount; y += yStep) {
            const int py = descriptor.meterY1 + preview_source_coordinate(static_cast<int>(y), verticalSampling);
            if (py < descriptor.sourceY1 || py >= descriptor.sourceY2) {
                continue;
            }
            const unsigned char* rowPtr =
                input.sourceBase + static_cast<std::size_t>(py - descriptor.sourceY1) * input.sourceRowBytes;
            for (std::size_t x = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x); x < widthCount; x += xStep) {
                const int px = descriptor.meterX1 + preview_source_coordinate(static_cast<int>(x), horizontalSampling);
                if (px < descriptor.sourceX1 || px >= descriptor.sourceX2) {
                    continue;
                }
                const float* pix = reinterpret_cast<const float*>(
                    rowPtr + static_cast<std::size_t>(px - descriptor.sourceX1) * pixelStrideBytes);

                float inRgb[3] = {pix[0], pix[1], pix[2]};
                float lin[3];
                apply_input_cctf_decoding_device(decoding, inRgb, lin);

                const float Y = mulY(input.rgbToXYZ, lin);
                if (isfinite(Y)) {
                    localMax = fmaxf(localMax, fmaxf(0.0f, Y));
                }
            }
        }

        __shared__ float sMax[16 * 16];
        const std::size_t t = static_cast<std::size_t>(threadIdx.y) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
        sMax[t] = localMax;
        __syncthreads();

        const std::size_t count = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockDim.y);
        for (std::size_t stride = count / 2; stride > 0; stride /= 2) {
            if (t < stride) {
                sMax[t] = fmaxf(sMax[t], sMax[t + stride]);
            }
            __syncthreads();
        }

        if (t == 0) {
            atomicMax(outMaxBits, __float_as_uint(sMax[0]));
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

    __global__ void reset_auto_exposure_state_kernel(float* outExposureScale) {
        if (threadIdx.x != 0 || blockIdx.x != 0) {
            return;
        }
        if (outExposureScale) {
            *outExposureScale = 1.0f;
        }
    }

    __global__ void build_center_weight_x_kernel(
        int meterWidth,
        int meterHeight,
        float* outWX) {
        if (!outWX || meterWidth <= 0 || meterHeight <= 0) {
            return;
        }

        // Matches build_center_weight_mask() weighting.
        constexpr float sigma = 0.2f;
        const int maxDimInt = (meterWidth > meterHeight) ? meterWidth : meterHeight;
        const float maxDim = static_cast<float>(maxDimInt);
        const float invMax = (maxDim > 0.0f) ? (1.0f / maxDim) : 0.0f;
        const std::size_t meterWidthCount = static_cast<std::size_t>(meterWidth);
        const std::size_t step = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
        for (std::size_t x = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x); x < meterWidthCount; x += step) {
            const float nx = (static_cast<float>(static_cast<int>(x)) / static_cast<float>(meterWidth)) - 0.5f;
            const float normX = nx * static_cast<float>(meterWidth) * invMax;
            const float r2 = normX * normX;
            const float w = expf(-r2 / (2.0f * sigma * sigma));
            outWX[x] = w;
        }
    }

    __global__ void build_center_weight_y_kernel(
        int meterWidth,
        int meterHeight,
        float* outWY) {
        if (!outWY || meterWidth <= 0 || meterHeight <= 0) {
            return;
        }

        // Matches build_center_weight_mask() weighting.
        constexpr float sigma = 0.2f;
        const int maxDimInt = (meterWidth > meterHeight) ? meterWidth : meterHeight;
        const float maxDim = static_cast<float>(maxDimInt);
        const float invMax = (maxDim > 0.0f) ? (1.0f / maxDim) : 0.0f;
        const std::size_t meterHeightCount = static_cast<std::size_t>(meterHeight);
        const std::size_t step = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
        for (std::size_t y = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x); y < meterHeightCount; y += step) {
            const float ny = (static_cast<float>(static_cast<int>(y)) / static_cast<float>(meterHeight)) - 0.5f;
            const float normY = ny * static_cast<float>(meterHeight) * invMax;
            const float r2 = normY * normY;
            const float w = expf(-r2 / (2.0f * sigma * sigma));
            outWY[y] = w;
        }
    }

    __global__ void meter_center_weighted_Y_partials_kernel(
        AutoExposureKernelInput input,
        MeteringWeightTables weightTables,
        JuicerCudaAutoExposurePartial* outPartials) {
        const JuicerCuda::AutoExposurePreviewDescriptor descriptor = input.descriptor;
        const int width = descriptor.previewWidth;
        const int height = descriptor.previewHeight;
        if (width <= 0 || height <= 0) {
            return;
        }

        PreviewAxisSampling horizontalSampling{};
        horizontalSampling.previewSize = width;
        horizontalSampling.sourceSize = descriptor.meterX2 - descriptor.meterX1;
        PreviewAxisSampling verticalSampling{};
        verticalSampling.previewSize = height;
        verticalSampling.sourceSize = descriptor.meterY2 - descriptor.meterY1;

        InputCctfDecodingDevice decoding{};
        decoding.inputColorSpaceIndex = input.inputColorSpaceIndex;
        decoding.applyCctfDecoding = input.applyCctfDecoding;

        double localSumY[JuicerCudaAutoExposurePartial::kLaneCount] = {};
        double localSumW[JuicerCudaAutoExposurePartial::kLaneCount] = {};
        const std::size_t pixelStrideBytes = static_cast<std::size_t>(input.nComponents) * sizeof(float);
        const std::size_t widthCount = static_cast<std::size_t>(width);
        const std::size_t heightCount = static_cast<std::size_t>(height);
        const std::size_t yStep = static_cast<std::size_t>(blockDim.y) * static_cast<std::size_t>(gridDim.y);
        const std::size_t xStep = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
        for (std::size_t y = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(blockDim.y) + static_cast<std::size_t>(threadIdx.y); y < heightCount; y += yStep) {
            const int meterY = static_cast<int>(y);
            const int py = descriptor.meterY1 + preview_source_coordinate(meterY, verticalSampling);
            if (py < descriptor.sourceY1 || py >= descriptor.sourceY2) {
                continue;
            }
            const unsigned char* rowPtr =
                input.sourceBase + static_cast<std::size_t>(py - descriptor.sourceY1) * input.sourceRowBytes;
            for (std::size_t x = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x); x < widthCount; x += xStep) {
                const int meterX = static_cast<int>(x);
                const int px = descriptor.meterX1 + preview_source_coordinate(meterX, horizontalSampling);
                if (px < descriptor.sourceX1 || px >= descriptor.sourceX2) {
                    continue;
                }
                const float* pix = reinterpret_cast<const float*>(
                    rowPtr + static_cast<std::size_t>(px - descriptor.sourceX1) * pixelStrideBytes);

                float inRgb[3] = {pix[0], pix[1], pix[2]};
                float lin[3];
                apply_input_cctf_decoding_device(decoding, inRgb, lin);

                const float Y = mulY(input.rgbToXYZ, lin);
                if (isfinite(Y)) {
                    float weights[JuicerCudaAutoExposurePartial::kLaneCount];
                    MeteringSample sample{};
                    sample.method = static_cast<int>(descriptor.method);
                    sample.x = meterX;
                    sample.y = meterY;
                    sample.width = width;
                    sample.height = height;
                    sample.luminance = Y;
                    metering_weights(sample, weightTables, weights);
                    for (int lane = 0; lane < JuicerCudaAutoExposurePartial::kLaneCount; ++lane) {
                        float weight = weights[lane];
                        if (!isfinite(weight) || !(weight > 0.0f)) {
                            weight = 0.0f;
                        }
                        localSumY[lane] += static_cast<double>(Y) * static_cast<double>(weight);
                        localSumW[lane] += static_cast<double>(weight);
                    }
                }
            }
        }

        __shared__ double sY[JuicerCudaAutoExposurePartial::kLaneCount][16 * 16];
        __shared__ double sW[JuicerCudaAutoExposurePartial::kLaneCount][16 * 16];
        const std::size_t t = static_cast<std::size_t>(threadIdx.y) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
        for (int lane = 0; lane < JuicerCudaAutoExposurePartial::kLaneCount; ++lane) {
            sY[lane][t] = localSumY[lane];
            sW[lane][t] = localSumW[lane];
        }
        __syncthreads();

        const std::size_t count = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockDim.y);
        for (std::size_t stride = count / 2; stride > 0; stride /= 2) {
            if (t < stride) {
                for (int lane = 0; lane < JuicerCudaAutoExposurePartial::kLaneCount; ++lane) {
                    sY[lane][t] += sY[lane][t + stride];
                    sW[lane][t] += sW[lane][t + stride];
                }
            }
            __syncthreads();
        }

        if (t == 0 && outPartials) {
            const std::size_t blockId = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(gridDim.x) + static_cast<std::size_t>(blockIdx.x);
            for (int lane = 0; lane < JuicerCudaAutoExposurePartial::kLaneCount; ++lane) {
                outPartials[blockId].sumY[lane] = sY[lane][0];
                outPartials[blockId].sumW[lane] = sW[lane][0];
            }
        }
    }

    __global__ void reduce_auto_exposure_partials_kernel(
        const JuicerCudaAutoExposurePartial* inPartials,
        int n,
        JuicerCudaAutoExposurePartial* outPartials) {
        if (n <= 0) {
            return;
        }

        const std::size_t tid = static_cast<std::size_t>(threadIdx.x);
        double sumY[JuicerCudaAutoExposurePartial::kLaneCount] = {};
        double sumW[JuicerCudaAutoExposurePartial::kLaneCount] = {};
        const std::size_t count = static_cast<std::size_t>(n);
        const std::size_t blockWidth = static_cast<std::size_t>(blockDim.x);
        const std::size_t gridStride = blockWidth * static_cast<std::size_t>(gridDim.x) * 2U;
        for (std::size_t base = static_cast<std::size_t>(blockIdx.x) * blockWidth * 2U + tid; base < count; base += gridStride) {
            for (int lane = 0; lane < JuicerCudaAutoExposurePartial::kLaneCount; ++lane) {
                sumY[lane] += inPartials[base].sumY[lane];
                sumW[lane] += inPartials[base].sumW[lane];

                const std::size_t base2 = base + blockWidth;
                if (base2 < count) {
                    sumY[lane] += inPartials[base2].sumY[lane];
                    sumW[lane] += inPartials[base2].sumW[lane];
                }
            }
        }

        __shared__ double sY[JuicerCudaAutoExposurePartial::kLaneCount][256];
        __shared__ double sW[JuicerCudaAutoExposurePartial::kLaneCount][256];
        for (int lane = 0; lane < JuicerCudaAutoExposurePartial::kLaneCount; ++lane) {
            sY[lane][tid] = sumY[lane];
            sW[lane][tid] = sumW[lane];
        }
        __syncthreads();

        for (std::size_t stride = blockWidth / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                for (int lane = 0; lane < JuicerCudaAutoExposurePartial::kLaneCount; ++lane) {
                    sY[lane][tid] += sY[lane][tid + stride];
                    sW[lane][tid] += sW[lane][tid + stride];
                }
            }
            __syncthreads();
        }

        if (tid == 0 && outPartials) {
            for (int lane = 0; lane < JuicerCudaAutoExposurePartial::kLaneCount; ++lane) {
                outPartials[blockIdx.x].sumY[lane] = sY[lane][0];
                outPartials[blockIdx.x].sumW[lane] = sW[lane][0];
            }
        }
    }

    __global__ void finalize_auto_exposure_from_sums_kernel(
        const JuicerCudaAutoExposurePartial* sums,
        int meteringMethod,
        float* outExposureScale) {
        if (threadIdx.x != 0 || blockIdx.x != 0) {
            return;
        }

        double autoEV = 0.0;
        if (sums) {
            double Yexp = 0.0;
            bool hasExposure = false;
            if (meteringMethod == kMethodPartial) {
                const int lane = sums[0].sumW[0] > 0.0 ? 0 : 3;
                if (sums[0].sumW[lane] > 0.0) {
                    Yexp = sums[0].sumY[lane] / sums[0].sumW[lane];
                    hasExposure = true;
                }
            } else if (meteringMethod == kMethodMultiZone) {
                constexpr double ringWeights[3] = {0.50, 0.30, 0.20};
                double weightedSum = 0.0;
                double weightTotal = 0.0;
                for (int lane = 0; lane < 3; ++lane) {
                    if (sums[0].sumW[lane] > 0.0) {
                        weightedSum += ringWeights[lane] * sums[0].sumY[lane] / sums[0].sumW[lane];
                        weightTotal += ringWeights[lane];
                    }
                }
                if (weightTotal > 0.0) {
                    Yexp = weightedSum / weightTotal;
                    hasExposure = true;
                } else if (sums[0].sumW[3] > 0.0) {
                    Yexp = sums[0].sumY[3] / sums[0].sumW[3];
                    hasExposure = true;
                }
            } else if (meteringMethod == kMethodHighlightWeighted && sums[0].sumW[0] < 1e-12) {
                if (sums[0].sumW[3] > 0.0) {
                    Yexp = sums[0].sumY[3] / sums[0].sumW[3];
                    hasExposure = true;
                }
            } else if (sums[0].sumW[0] > 0.0) {
                Yexp = sums[0].sumY[0] / sums[0].sumW[0];
                hasExposure = true;
            }
            if (hasExposure && isfinite(Yexp)) {
                if ((Yexp > 0.0) && (kCameraMeterTargetY > 0.0)) {
                    const double exposureRatio = Yexp / kCameraMeterTargetY;
                    const double evComp = -log(exposureRatio) / log(2.0);
                    if (isfinite(evComp)) {
                        autoEV = evComp;
                    }
                }
            }
        }

        if (outExposureScale) {
            const double scale64 = exp2(autoEV);
            const float scale = (isfinite(scale64) && scale64 > 0.0) ? static_cast<float>(scale64) : 1.0f;
            *outExposureScale = scale;
        }
    }

    __global__ void meter_histogram_Y_bits_kernel(
        AutoExposureKernelInput input,
        HistogramBuffers buffers) {
        const JuicerCuda::AutoExposurePreviewDescriptor descriptor = input.descriptor;
        const int width = descriptor.previewWidth;
        const int height = descriptor.previewHeight;
        float maxY = 0.0f;
        if (buffers.maxYBits) {
            maxY = __uint_as_float(*buffers.maxYBits);
        }
        if (!(maxY > 0.0f) || !isfinite(maxY) || !buffers.histogram) {
            return;
        }
        if (width <= 0 || height <= 0) {
            return;
        }

        PreviewAxisSampling horizontalSampling{};
        horizontalSampling.previewSize = width;
        horizontalSampling.sourceSize = descriptor.meterX2 - descriptor.meterX1;
        PreviewAxisSampling verticalSampling{};
        verticalSampling.previewSize = height;
        verticalSampling.sourceSize = descriptor.meterY2 - descriptor.meterY1;

        InputCctfDecodingDevice decoding{};
        decoding.inputColorSpaceIndex = input.inputColorSpaceIndex;
        decoding.applyCctfDecoding = input.applyCctfDecoding;

        __shared__ unsigned int sHist[kMedianHistogramBins];
        const std::size_t t = static_cast<std::size_t>(threadIdx.y) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
        const std::size_t threads = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockDim.y);
        const std::size_t histogramBinCount = static_cast<std::size_t>(kMedianHistogramBins);
        for (std::size_t i = t; i < histogramBinCount; i += threads) {
            sHist[i] = 0U;
        }
        __syncthreads();

        const std::size_t pixelStrideBytes = static_cast<std::size_t>(input.nComponents) * sizeof(float);
        const std::size_t widthCount = static_cast<std::size_t>(width);
        const std::size_t heightCount = static_cast<std::size_t>(height);
        const std::size_t yStep = static_cast<std::size_t>(blockDim.y) * static_cast<std::size_t>(gridDim.y);
        const std::size_t xStep = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
        for (std::size_t y = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(blockDim.y) + static_cast<std::size_t>(threadIdx.y); y < heightCount; y += yStep) {
            const int py = descriptor.meterY1 + preview_source_coordinate(static_cast<int>(y), verticalSampling);
            if (py < descriptor.sourceY1 || py >= descriptor.sourceY2) {
                continue;
            }
            const unsigned char* rowPtr =
                input.sourceBase + static_cast<std::size_t>(py - descriptor.sourceY1) * input.sourceRowBytes;
            for (std::size_t x = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x); x < widthCount; x += xStep) {
                const int px = descriptor.meterX1 + preview_source_coordinate(static_cast<int>(x), horizontalSampling);
                if (px < descriptor.sourceX1 || px >= descriptor.sourceX2) {
                    continue;
                }
                const float* pix = reinterpret_cast<const float*>(
                    rowPtr + static_cast<std::size_t>(px - descriptor.sourceX1) * pixelStrideBytes);

                float inRgb[3] = {pix[0], pix[1], pix[2]};
                float lin[3];
                apply_input_cctf_decoding_device(decoding, inRgb, lin);

                float Y = mulY(input.rgbToXYZ, lin);
                if (isfinite(Y)) {
                    if (Y < 0.0f) {
                        Y = 0.0f;
                    }
                    const float norm = fminf(1.0f, Y / maxY);
                    const int bin = static_cast<int>(norm * static_cast<float>(kMedianHistogramBins - 1));
                    atomicAdd(&sHist[bin], 1U);
                }
            }
        }
        __syncthreads();

        for (std::size_t i = t; i < histogramBinCount; i += threads) {
            const unsigned int count = sHist[i];
            if (count != 0U) {
                atomicAdd(&buffers.histogram[i], count);
            }
        }
    }

    __global__ void finalize_auto_exposure_from_histogram_kernel(
        HistogramBuffers buffers,
        float* outExposureScale) {
        if (threadIdx.x != 0 || blockIdx.x != 0) {
            return;
        }

        float maxY = 0.0f;
        if (buffers.maxYBits) {
            maxY = __uint_as_float(*buffers.maxYBits);
        }
        if (!(maxY > 0.0f) || !isfinite(maxY) || !buffers.histogram) {
            if (outExposureScale) {
                *outExposureScale = 1.0f;
            }
            return;
        }

        unsigned long long totalCount = 0;
        for (int i = 0; i < kMedianHistogramBins; ++i) {
            totalCount += static_cast<unsigned long long>(buffers.histogram[i]);
        }
        if (totalCount == 0) {
            if (outExposureScale) {
                *outExposureScale = 1.0f;
            }
            return;
        }

        const unsigned long long target = totalCount / 2ULL;
        unsigned long long cumulative = 0;
        int medianBin = kMedianHistogramBins - 1;
        unsigned int countInBin = 0;
        for (int i = 0; i < kMedianHistogramBins; ++i) {
            const unsigned int count = buffers.histogram[i];
            if (cumulative + count > target) {
                medianBin = i;
                countInBin = count;
                break;
            }
            cumulative += count;
        }

        const double binWidth = static_cast<double>(maxY) / static_cast<double>(kMedianHistogramBins);
        double fraction = 0.0;
        if (countInBin > 0) {
            fraction = static_cast<double>(target - cumulative) / static_cast<double>(countInBin);
        }
        const double medianY = (static_cast<double>(medianBin) + fraction) * binWidth;

        double autoEV = 0.0;
        if ((medianY > 0.0) && isfinite(medianY) && (kCameraMeterTargetY > 0.0)) {
            const double exposureRatio = medianY / kCameraMeterTargetY;
            const double evComp = -log(exposureRatio) / log(2.0);
            if (isfinite(evComp)) {
                autoEV = evComp;
            }
        }

        if (outExposureScale) {
            const double scale64 = exp2(autoEV);
            const float scale = (isfinite(scale64) && scale64 > 0.0) ? static_cast<float>(scale64) : 1.0f;
            *outExposureScale = scale;
        }
    }

} // namespace

extern "C" int juicer_cuda_auto_exposure_meter_to_device(
    const void* srcDeviceBase,
    std::size_t srcRowBytes,
    JuicerCuda::AutoExposurePreviewDescriptor descriptor,
    JuicerCuda::AutoExposureSourceFormat sourceFormat,
    const float* rgbToXYZ9,
    JuicerCudaAutoExposureScratch scratch,
    JuicerCudaAutoExposureDeviceState outState,
    void* cudaStreamOpaque,
    const char** outErrorMsg) {
    thread_local std::string sError;
    if (outErrorMsg) {
        *outErrorMsg = nullptr;
    }

    if (!srcDeviceBase || srcRowBytes == 0 || !rgbToXYZ9) {
        if (outErrorMsg)
            *outErrorMsg = set_error(sError, "invalid arguments");
        return 1;
    }
    if (!(sourceFormat.componentCount == 3 || sourceFormat.componentCount == 4)) {
        if (outErrorMsg)
            *outErrorMsg = set_error(sError, "unsupported component count");
        return 2;
    }
    if (!outState.exposureScale) {
        if (outErrorMsg)
            *outErrorMsg = set_error(sError, "invalid output device state");
        return 3;
    }

    const int width = descriptor.previewWidth;
    const int height = descriptor.previewHeight;
    const int sourceWidth = descriptor.meterX2 - descriptor.meterX1;
    const int sourceHeight = descriptor.meterY2 - descriptor.meterY1;
    const int meteringMethod = static_cast<int>(descriptor.method);
    if (width <= 0 || height <= 0) {
        const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
        reset_auto_exposure_state_kernel<<<1, 1, 0, stream>>>(outState.exposureScale);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            if (outErrorMsg)
                *outErrorMsg = set_error_cuda(sError, "reset auto-exposure state failed: ", err);
            return 4;
        }
        return 0;
    }

    if (descriptor.sourceX2 < descriptor.sourceX1 ||
        descriptor.sourceY2 < descriptor.sourceY1 ||
        sourceWidth <= 0 ||
        sourceHeight <= 0 ||
        descriptor.sampling != JuicerCuda::AutoExposurePreviewDescriptor::Sampling::NearestNeighbor) {
        if (outErrorMsg)
            *outErrorMsg = set_error(sError, "invalid source bounds");
        return 5;
    }
    if (meteringMethod < kMethodCenterWeighted || meteringMethod > kMethodHighlightWeighted) {
        if (outErrorMsg)
            *outErrorMsg = set_error(sError, "unsupported auto-exposure method");
        return 18;
    }

    Mat3 m{};
    for (int i = 0; i < 9; ++i) {
        m.m[i] = rgbToXYZ9[i];
    }

    AutoExposureKernelInput kernelInput{};
    kernelInput.sourceBase = reinterpret_cast<const unsigned char*>(srcDeviceBase);
    kernelInput.sourceRowBytes = srcRowBytes;
    kernelInput.descriptor = descriptor;
    kernelInput.nComponents = sourceFormat.componentCount;
    kernelInput.inputColorSpaceIndex = sourceFormat.inputColorSpaceIndex;
    kernelInput.applyCctfDecoding = sourceFormat.applyCctfDecoding;
    kernelInput.rgbToXYZ = m;

    const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

    if (meteringMethod == kMethodMedian) {
        if (!scratch.maxYBits || !scratch.histogram) {
            if (outErrorMsg)
                *outErrorMsg = set_error(sError, "median metering scratch buffers missing");
            return 6;
        }

        cudaError_t err = cudaMemsetAsync(scratch.maxYBits, 0, sizeof(unsigned int), stream);
        if (err != cudaSuccess) {
            if (outErrorMsg)
                *outErrorMsg = set_error_cuda(sError, "cudaMemsetAsync(maxYBits) failed: ", err);
            return 7;
        }

        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        meter_max_Y_kernel<<<grid, block, 0, stream>>>(kernelInput, scratch.maxYBits);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            if (outErrorMsg)
                *outErrorMsg = set_error_cuda(sError, "meter_max_Y_kernel launch failed: ", err);
            return 8;
        }

        err = cudaMemsetAsync(scratch.histogram, 0, sizeof(unsigned int) * kMedianHistogramBins, stream);
        if (err != cudaSuccess) {
            if (outErrorMsg)
                *outErrorMsg = set_error_cuda(sError, "cudaMemsetAsync(histogram) failed: ", err);
            return 9;
        }

        HistogramBuffers histogramBuffers{};
        histogramBuffers.maxYBits = scratch.maxYBits;
        histogramBuffers.histogram = scratch.histogram;
        meter_histogram_Y_bits_kernel<<<grid, block, 0, stream>>>(kernelInput, histogramBuffers);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            if (outErrorMsg)
                *outErrorMsg = set_error_cuda(sError, "meter_histogram kernel launch failed: ", err);
            return 10;
        }

        finalize_auto_exposure_from_histogram_kernel<<<1, 1, 0, stream>>>(
            histogramBuffers,
            outState.exposureScale);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            if (outErrorMsg)
                *outErrorMsg = set_error_cuda(sError, "finalize histogram kernel launch failed: ", err);
            return 11;
        }
        return 0;
    }

    if (!scratch.partialsA || !scratch.partialsB || scratch.partialCapacity <= 0) {
        if (outErrorMsg)
            *outErrorMsg = set_error(sError, "center-weighted metering partial buffers missing");
        return 12;
    }
    if (meteringMethod == kMethodCenterWeighted && (!scratch.weightsX || !scratch.weightsY)) {
        if (outErrorMsg)
            *outErrorMsg = set_error(sError, "center-weighted metering weights missing");
        return 17;
    }

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    const std::size_t blocksCount = static_cast<std::size_t>(grid.x) * static_cast<std::size_t>(grid.y);
    if (blocksCount > static_cast<std::size_t>(scratch.partialCapacity)) {
        if (outErrorMsg)
            *outErrorMsg = set_error(sError, "center-weighted metering partial buffer capacity too small");
        return 13;
    }

    MeteringWeightTables weightTables{};
    weightTables.horizontal = scratch.weightsX;
    weightTables.vertical = scratch.weightsY;
    meter_center_weighted_Y_partials_kernel<<<grid, block, 0, stream>>>(
        kernelInput,
        weightTables,
        scratch.partialsA);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        if (outErrorMsg)
            *outErrorMsg = set_error_cuda(sError, "meter_center_weighted partials kernel launch failed: ", err);
        return 14;
    }

    const int reduceThreads = 256;
    int n = static_cast<int>(blocksCount);
    const JuicerCudaAutoExposurePartial* in = scratch.partialsA;
    JuicerCudaAutoExposurePartial* out = scratch.partialsB;
    while (n > 1) {
        const int blocks = (n + reduceThreads * 2 - 1) / (reduceThreads * 2);
        reduce_auto_exposure_partials_kernel<<<blocks, reduceThreads, 0, stream>>>(in, n, out);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            if (outErrorMsg)
                *outErrorMsg = set_error_cuda(sError, "reduce partials kernel launch failed: ", err);
            return 15;
        }
        n = blocks;
        const JuicerCudaAutoExposurePartial* nextIn = out;
        out = (JuicerCudaAutoExposurePartial*)in;
        in = nextIn;
    }

    finalize_auto_exposure_from_sums_kernel<<<1, 1, 0, stream>>>(
        in,
        meteringMethod,
        outState.exposureScale);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        if (outErrorMsg)
            *outErrorMsg = set_error_cuda(sError, "finalize sums kernel launch failed: ", err);
        return 16;
    }

    return 0;
}

extern "C" int juicer_cuda_auto_exposure_build_center_weight_tables(
    int meterWidth,
    int meterHeight,
    float* weightsX,
    float* weightsY,
    void* cudaStreamOpaque,
    const char** outErrorMsg) {
    thread_local std::string sError;
    if (outErrorMsg) {
        *outErrorMsg = nullptr;
    }

    if (meterWidth <= 0 || meterHeight <= 0) {
        if (outErrorMsg)
            *outErrorMsg = set_error(sError, "invalid meter dimensions");
        return 1;
    }
    if (!weightsX || !weightsY) {
        if (outErrorMsg)
            *outErrorMsg = set_error(sError, "invalid weights buffers");
        return 2;
    }

    const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
    const int threads = 256;
    const int blocksX = (meterWidth + threads - 1) / threads;
    build_center_weight_x_kernel<<<blocksX, threads, 0, stream>>>(meterWidth, meterHeight, weightsX);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        if (outErrorMsg)
            *outErrorMsg = set_error_cuda(sError, "build_center_weight_x kernel launch failed: ", err);
        return 3;
    }

    const int blocksY = (meterHeight + threads - 1) / threads;
    build_center_weight_y_kernel<<<blocksY, threads, 0, stream>>>(meterWidth, meterHeight, weightsY);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        if (outErrorMsg)
            *outErrorMsg = set_error_cuda(sError, "build_center_weight_y kernel launch failed: ", err);
        return 4;
    }

    return 0;
}
