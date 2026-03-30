// Cuda/JuicerCudaAutoExposure.cu
//
// CUDA camera auto-exposure metering (center-weighted Gaussian mask).
//
// This is a parity-oriented port of the CPU meter in JuicerEffect.cpp, adapted for device pointers.
//
#include "Cuda/JuicerCudaResources.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "Cuda/JuicerCudaDeviceHelpers.cuh"

namespace {

    constexpr int kMedianHistogramBins = 2048;
    constexpr double kCameraMeterTargetY = 0.184;

    struct DeviceAccumBuffers {
        double* sumY = nullptr;
        double* sumW = nullptr;
        unsigned int* maxYBits = nullptr;
        unsigned int* histogram = nullptr;
        int deviceId = -1;

        ~DeviceAccumBuffers() {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
            if (sumY) {
                cudaFree(sumY);
                sumY = nullptr;
            }
            if (sumW) {
                cudaFree(sumW);
                sumW = nullptr;
            }
            if (maxYBits) {
                cudaFree(maxYBits);
                maxYBits = nullptr;
            }
            if (histogram) {
                cudaFree(histogram);
                histogram = nullptr;
            }
#endif
        }
    };

    thread_local DeviceAccumBuffers gAccum;

    __device__ __forceinline__ float mulY(const Mat3& m, const float rgb[3]) {
        return m.m[3] * rgb[0] + m.m[4] * rgb[1] + m.m[5] * rgb[2];
    }

    __global__ void meter_center_weighted_Y_kernel(
        const unsigned char* srcBase,
        std::size_t srcRowBytes,
        int srcBoundsX1,
        int srcBoundsY1,
        int srcBoundsX2,
        int srcBoundsY2,
        int meterX1,
        int meterY1,
        int width,
        int height,
        int nComponents,
        int inputColorSpaceIndex,
        int applyCctfDecoding,
        Mat3 rgbToXYZ,
        double* outSumY,
        double* outSumW)
    {
        double localSumY = 0.0;
        double localSumW = 0.0;
        constexpr float sigma = 0.2f;
        const int maxDimInt = (width > height) ? width : height;
        const float maxDim = static_cast<float>(maxDimInt);
        const float invMax = (maxDim > 0.0f) ? (1.0f / maxDim) : 0.0f;
        const std::size_t pixelStrideBytes = static_cast<std::size_t>(nComponents) * sizeof(float);

        for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < height; y += blockDim.y * gridDim.y) {
            const int py = meterY1 + y;
            if (py < srcBoundsY1 || py >= srcBoundsY2) {
                continue;
            }
            const unsigned char* rowPtr = srcBase + static_cast<std::size_t>(py - srcBoundsY1) * srcRowBytes;
            const float ny = (static_cast<float>(y) / static_cast<float>(height)) - 0.5f;
            const float normY = ny * static_cast<float>(height) * invMax;
            for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < width; x += blockDim.x * gridDim.x) {
                const int px = meterX1 + x;
                if (px < srcBoundsX1 || px >= srcBoundsX2) {
                    continue;
                }
                const float* pix = reinterpret_cast<const float*>(rowPtr + static_cast<std::size_t>(px - srcBoundsX1) * pixelStrideBytes);

                float inRgb[3] = { pix[0], pix[1], pix[2] };
                float lin[3];
                apply_input_cctf_decoding_device(inputColorSpaceIndex, applyCctfDecoding, inRgb, lin);

                const float Y = mulY(rgbToXYZ, lin);
                if (isfinite(Y)) {
                    // Matches build_center_weight_mask() weighting.
                    const float nx = (static_cast<float>(x) / static_cast<float>(width)) - 0.5f;
                    const float normX = nx * static_cast<float>(width) * invMax;
                    const float r2 = normX * normX + normY * normY;
                    const float w = expf(-r2 / (2.0f * sigma * sigma));

                    localSumY += static_cast<double>(Y) * static_cast<double>(w);
                    localSumW += static_cast<double>(w);
                }
            }
        }

        __shared__ double sY[16 * 16];
        __shared__ double sW[16 * 16];
        const int t = threadIdx.y * blockDim.x + threadIdx.x;
        sY[t] = localSumY;
        sW[t] = localSumW;
        __syncthreads();

        int count = blockDim.x * blockDim.y;
        for (int stride = count / 2; stride > 0; stride /= 2) {
            if (t < stride) {
                sY[t] += sY[t + stride];
                sW[t] += sW[t + stride];
            }
            __syncthreads();
        }

        if (t == 0) {
            atomicAdd(outSumY, sY[0]);
            atomicAdd(outSumW, sW[0]);
        }
    }

    __global__ void meter_max_Y_kernel(
        const unsigned char* srcBase,
        std::size_t srcRowBytes,
        int srcBoundsX1,
        int srcBoundsY1,
        int srcBoundsX2,
        int srcBoundsY2,
        int meterX1,
        int meterY1,
        int width,
        int height,
        int nComponents,
        int inputColorSpaceIndex,
        int applyCctfDecoding,
        Mat3 rgbToXYZ,
        unsigned int* outMaxBits)
    {
        float localMax = 0.0f;
        const std::size_t pixelStrideBytes = static_cast<std::size_t>(nComponents) * sizeof(float);
        for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < height; y += blockDim.y * gridDim.y) {
            const int py = meterY1 + y;
            if (py < srcBoundsY1 || py >= srcBoundsY2) {
                continue;
            }
            const unsigned char* rowPtr = srcBase + static_cast<std::size_t>(py - srcBoundsY1) * srcRowBytes;
            for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < width; x += blockDim.x * gridDim.x) {
                const int px = meterX1 + x;
                if (px < srcBoundsX1 || px >= srcBoundsX2) {
                    continue;
                }
                const float* pix = reinterpret_cast<const float*>(rowPtr + static_cast<std::size_t>(px - srcBoundsX1) * pixelStrideBytes);

                float inRgb[3] = { pix[0], pix[1], pix[2] };
                float lin[3];
                apply_input_cctf_decoding_device(inputColorSpaceIndex, applyCctfDecoding, inRgb, lin);

                const float Y = mulY(rgbToXYZ, lin);
                if (isfinite(Y)) {
                    localMax = fmaxf(localMax, fmaxf(0.0f, Y));
                }
            }
        }

        __shared__ float sMax[16 * 16];
        const int t = threadIdx.y * blockDim.x + threadIdx.x;
        sMax[t] = localMax;
        __syncthreads();

        int count = blockDim.x * blockDim.y;
        for (int stride = count / 2; stride > 0; stride /= 2) {
            if (t < stride) {
                sMax[t] = fmaxf(sMax[t], sMax[t + stride]);
            }
            __syncthreads();
        }

        if (t == 0) {
            atomicMax(outMaxBits, __float_as_uint(sMax[0]));
        }
    }

    __global__ void meter_histogram_Y_kernel(
        const unsigned char* srcBase,
        std::size_t srcRowBytes,
        int srcBoundsX1,
        int srcBoundsY1,
        int srcBoundsX2,
        int srcBoundsY2,
        int meterX1,
        int meterY1,
        int width,
        int height,
        int nComponents,
        int inputColorSpaceIndex,
        int applyCctfDecoding,
        Mat3 rgbToXYZ,
        float maxY,
        unsigned int* histogram)
    {
        const std::size_t pixelStrideBytes = static_cast<std::size_t>(nComponents) * sizeof(float);
        for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < height; y += blockDim.y * gridDim.y) {
            const int py = meterY1 + y;
            if (py < srcBoundsY1 || py >= srcBoundsY2) {
                continue;
            }
            const unsigned char* rowPtr = srcBase + static_cast<std::size_t>(py - srcBoundsY1) * srcRowBytes;
            for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < width; x += blockDim.x * gridDim.x) {
                const int px = meterX1 + x;
                if (px < srcBoundsX1 || px >= srcBoundsX2) {
                    continue;
                }
                const float* pix = reinterpret_cast<const float*>(rowPtr + static_cast<std::size_t>(px - srcBoundsX1) * pixelStrideBytes);

                float inRgb[3] = { pix[0], pix[1], pix[2] };
                float lin[3];
                apply_input_cctf_decoding_device(inputColorSpaceIndex, applyCctfDecoding, inRgb, lin);

                float Y = mulY(rgbToXYZ, lin);
                if (isfinite(Y)) {
                    if (Y < 0.0f) {
                        Y = 0.0f;
                    }
                    if (maxY > 0.0f) {
                        const float norm = fminf(1.0f, Y / maxY);
                        const int bin = static_cast<int>(norm * static_cast<float>(kMedianHistogramBins - 1));
                        atomicAdd(&histogram[bin], 1U);
                    }
                }
            }
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

    __global__ void reset_auto_exposure_state_kernel(
        double sliderEV,
        double* outAutoEV,
        float* outExposureScale,
        int* outValid)
    {
        if (threadIdx.x != 0 || blockIdx.x != 0) {
            return;
        }
        if (outAutoEV) {
            *outAutoEV = 0.0;
        }
        if (outValid) {
            *outValid = 0;
        }
        if (outExposureScale) {
            const double scale64 = exp2(sliderEV);
            const float scale = (isfinite(scale64) && scale64 > 0.0) ? static_cast<float>(scale64) : 1.0f;
            *outExposureScale = scale;
        }
    }

    __global__ void update_auto_exposure_scale_kernel(
        double sliderEV,
        const double* autoEV,
        const int* valid,
        float* outExposureScale)
    {
        if (threadIdx.x != 0 || blockIdx.x != 0) {
            return;
        }
        const int v = valid ? *valid : 0;
        const double ae = (v != 0 && autoEV) ? *autoEV : 0.0;
        const double scale64 = exp2(ae + sliderEV);
        const float scale = (isfinite(scale64) && scale64 > 0.0) ? static_cast<float>(scale64) : 1.0f;
        if (outExposureScale) {
            *outExposureScale = scale;
        }
    }

    __global__ void build_center_weight_x_kernel(
        int meterWidth,
        int meterHeight,
        float* outWX)
    {
        if (!outWX || meterWidth <= 0 || meterHeight <= 0) {
            return;
        }

        // Matches build_center_weight_mask() weighting.
        constexpr float sigma = 0.2f;
        const int maxDimInt = (meterWidth > meterHeight) ? meterWidth : meterHeight;
        const float maxDim = static_cast<float>(maxDimInt);
        const float invMax = (maxDim > 0.0f) ? (1.0f / maxDim) : 0.0f;
        for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < meterWidth; x += blockDim.x * gridDim.x) {
            const float nx = (static_cast<float>(x) / static_cast<float>(meterWidth)) - 0.5f;
            const float normX = nx * static_cast<float>(meterWidth) * invMax;
            const float r2 = normX * normX;
            const float w = expf(-r2 / (2.0f * sigma * sigma));
            outWX[x] = w;
        }
    }

    __global__ void build_center_weight_y_kernel(
        int meterWidth,
        int meterHeight,
        float* outWY)
    {
        if (!outWY || meterWidth <= 0 || meterHeight <= 0) {
            return;
        }

        // Matches build_center_weight_mask() weighting.
        constexpr float sigma = 0.2f;
        const int maxDimInt = (meterWidth > meterHeight) ? meterWidth : meterHeight;
        const float maxDim = static_cast<float>(maxDimInt);
        const float invMax = (maxDim > 0.0f) ? (1.0f / maxDim) : 0.0f;
        for (int y = blockIdx.x * blockDim.x + threadIdx.x; y < meterHeight; y += blockDim.x * gridDim.x) {
            const float ny = (static_cast<float>(y) / static_cast<float>(meterHeight)) - 0.5f;
            const float normY = ny * static_cast<float>(meterHeight) * invMax;
            const float r2 = normY * normY;
            const float w = expf(-r2 / (2.0f * sigma * sigma));
            outWY[y] = w;
        }
    }

    __global__ void meter_center_weighted_Y_partials_kernel(
        const unsigned char* srcBase,
        std::size_t srcRowBytes,
        int srcBoundsX1,
        int srcBoundsY1,
        int srcBoundsX2,
        int srcBoundsY2,
        int meterX1,
        int meterY1,
        int width,
        int height,
        int nComponents,
        int inputColorSpaceIndex,
        int applyCctfDecoding,
        Mat3 rgbToXYZ,
        const float* weightsX,
        const float* weightsY,
        JuicerCudaAutoExposurePartial* outPartials)
    {
        double localSumY = 0.0;
        double localSumW = 0.0;
        const std::size_t pixelStrideBytes = static_cast<std::size_t>(nComponents) * sizeof(float);
        for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < height; y += blockDim.y * gridDim.y) {
            const int py = meterY1 + y;
            if (py < srcBoundsY1 || py >= srcBoundsY2) {
                continue;
            }
            const unsigned char* rowPtr = srcBase + static_cast<std::size_t>(py - srcBoundsY1) * srcRowBytes;
            for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < width; x += blockDim.x * gridDim.x) {
                const int px = meterX1 + x;
                if (px < srcBoundsX1 || px >= srcBoundsX2) {
                    continue;
                }
                const float* pix = reinterpret_cast<const float*>(rowPtr + static_cast<std::size_t>(px - srcBoundsX1) * pixelStrideBytes);

                float inRgb[3] = { pix[0], pix[1], pix[2] };
                float lin[3];
                apply_input_cctf_decoding_device(inputColorSpaceIndex, applyCctfDecoding, inRgb, lin);

                const float Y = mulY(rgbToXYZ, lin);
                if (isfinite(Y)) {
                    float w = 0.0f;
                    if (weightsX && weightsY) {
                        w = weightsX[x] * weightsY[y];
                    }
                    if (!isfinite(w) || !(w > 0.0f)) {
                        w = 0.0f;
                    }

                    localSumY += static_cast<double>(Y) * static_cast<double>(w);
                    localSumW += static_cast<double>(w);
                }
            }
        }

        __shared__ double sY[16 * 16];
        __shared__ double sW[16 * 16];
        const int t = threadIdx.y * blockDim.x + threadIdx.x;
        sY[t] = localSumY;
        sW[t] = localSumW;
        __syncthreads();

        int count = blockDim.x * blockDim.y;
        for (int stride = count / 2; stride > 0; stride /= 2) {
            if (t < stride) {
                sY[t] += sY[t + stride];
                sW[t] += sW[t + stride];
            }
            __syncthreads();
        }

        if (t == 0 && outPartials) {
            const int blockId = blockIdx.y * gridDim.x + blockIdx.x;
            outPartials[blockId].sumY = sY[0];
            outPartials[blockId].sumW = sW[0];
        }
    }

    __global__ void reduce_auto_exposure_partials_kernel(
        const JuicerCudaAutoExposurePartial* inPartials,
        int n,
        JuicerCudaAutoExposurePartial* outPartials)
    {
        const int tid = threadIdx.x;
        double sumY = 0.0;
        double sumW = 0.0;
        const int gridStride = blockDim.x * gridDim.x * 2;
        for (int base = (blockIdx.x * blockDim.x * 2) + tid; base < n; base += gridStride) {
            sumY += inPartials[base].sumY;
            sumW += inPartials[base].sumW;

            const int base2 = base + blockDim.x;
            if (base2 < n) {
                sumY += inPartials[base2].sumY;
                sumW += inPartials[base2].sumW;
            }
        }

        __shared__ double sY[256];
        __shared__ double sW[256];
        sY[tid] = sumY;
        sW[tid] = sumW;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                sY[tid] += sY[tid + stride];
                sW[tid] += sW[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0 && outPartials) {
            outPartials[blockIdx.x].sumY = sY[0];
            outPartials[blockIdx.x].sumW = sW[0];
        }
    }

    __global__ void finalize_auto_exposure_from_sums_kernel(
        const JuicerCudaAutoExposurePartial* sums,
        double sliderEV,
        double* outAutoEV,
        float* outExposureScale,
        int* outValid)
    {
        if (threadIdx.x != 0 || blockIdx.x != 0) {
            return;
        }

        double autoEV = 0.0;
        int valid = 0;
        if (sums) {
            const double sumY = sums[0].sumY;
            const double sumW = sums[0].sumW;
            if ((sumW > 0.0) && isfinite(sumW) && isfinite(sumY)) {
                const double Yexp = sumY / sumW;
                if ((Yexp > 0.0) && (kCameraMeterTargetY > 0.0)) {
                    const double exposureRatio = Yexp / kCameraMeterTargetY;
                    const double evComp = -log(exposureRatio) / log(2.0);
                    if (isfinite(evComp)) {
                        autoEV = evComp;
                        valid = 1;
                    }
                }
            }
        }

        if (outAutoEV) {
            *outAutoEV = autoEV;
        }
        if (outValid) {
            *outValid = valid;
        }

        if (outExposureScale) {
            const double scale64 = exp2(autoEV + sliderEV);
            const float scale = (isfinite(scale64) && scale64 > 0.0) ? static_cast<float>(scale64) : 1.0f;
            *outExposureScale = scale;
        }
    }

    __global__ void meter_histogram_Y_bits_kernel(
        const unsigned char* srcBase,
        std::size_t srcRowBytes,
        int srcBoundsX1,
        int srcBoundsY1,
        int srcBoundsX2,
        int srcBoundsY2,
        int meterX1,
        int meterY1,
        int width,
        int height,
        int nComponents,
        int inputColorSpaceIndex,
        int applyCctfDecoding,
        Mat3 rgbToXYZ,
        const unsigned int* maxYBits,
        unsigned int* histogram)
    {
        float maxY = 0.0f;
        if (maxYBits) {
            maxY = __uint_as_float(*maxYBits);
        }
        if (!(maxY > 0.0f) || !isfinite(maxY) || !histogram) {
            return;
        }

        __shared__ unsigned int sHist[kMedianHistogramBins];
        const int t = threadIdx.y * blockDim.x + threadIdx.x;
        const int threads = blockDim.x * blockDim.y;
        for (int i = t; i < kMedianHistogramBins; i += threads) {
            sHist[i] = 0U;
        }
        __syncthreads();

        const std::size_t pixelStrideBytes = static_cast<std::size_t>(nComponents) * sizeof(float);
        for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < height; y += blockDim.y * gridDim.y) {
            const int py = meterY1 + y;
            if (py < srcBoundsY1 || py >= srcBoundsY2) {
                continue;
            }
            const unsigned char* rowPtr = srcBase + static_cast<std::size_t>(py - srcBoundsY1) * srcRowBytes;
            for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < width; x += blockDim.x * gridDim.x) {
                const int px = meterX1 + x;
                if (px < srcBoundsX1 || px >= srcBoundsX2) {
                    continue;
                }
                const float* pix = reinterpret_cast<const float*>(rowPtr + static_cast<std::size_t>(px - srcBoundsX1) * pixelStrideBytes);

                float inRgb[3] = { pix[0], pix[1], pix[2] };
                float lin[3];
                apply_input_cctf_decoding_device(inputColorSpaceIndex, applyCctfDecoding, inRgb, lin);

                float Y = mulY(rgbToXYZ, lin);
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

        for (int i = t; i < kMedianHistogramBins; i += threads) {
            const unsigned int count = sHist[i];
            if (count != 0U) {
                atomicAdd(&histogram[i], count);
            }
        }
    }

    __global__ void finalize_auto_exposure_from_histogram_kernel(
        const unsigned int* maxYBits,
        const unsigned int* histogram,
        double sliderEV,
        double* outAutoEV,
        float* outExposureScale,
        int* outValid)
    {
        if (threadIdx.x != 0 || blockIdx.x != 0) {
            return;
        }

        float maxY = 0.0f;
        if (maxYBits) {
            maxY = __uint_as_float(*maxYBits);
        }
        if (!(maxY > 0.0f) || !isfinite(maxY) || !histogram) {
            if (outAutoEV) *outAutoEV = 0.0;
            if (outValid) *outValid = 0;
            if (outExposureScale) {
                const double scale64 = exp2(sliderEV);
                const float scale = (isfinite(scale64) && scale64 > 0.0) ? static_cast<float>(scale64) : 1.0f;
                *outExposureScale = scale;
            }
            return;
        }

        unsigned long long totalCount = 0;
        for (int i = 0; i < kMedianHistogramBins; ++i) {
            totalCount += static_cast<unsigned long long>(histogram[i]);
        }
        if (totalCount == 0) {
            if (outAutoEV) *outAutoEV = 0.0;
            if (outValid) *outValid = 0;
            if (outExposureScale) {
                const double scale64 = exp2(sliderEV);
                const float scale = (isfinite(scale64) && scale64 > 0.0) ? static_cast<float>(scale64) : 1.0f;
                *outExposureScale = scale;
            }
            return;
        }

        const unsigned long long target = totalCount / 2ULL;
        unsigned long long cumulative = 0;
        int medianBin = kMedianHistogramBins - 1;
        unsigned int countInBin = 0;
        for (int i = 0; i < kMedianHistogramBins; ++i) {
            const unsigned int count = histogram[i];
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
        int valid = 0;
        if ((medianY > 0.0) && isfinite(medianY) && (kCameraMeterTargetY > 0.0)) {
            const double exposureRatio = medianY / kCameraMeterTargetY;
            const double evComp = -log(exposureRatio) / log(2.0);
            if (isfinite(evComp)) {
                autoEV = evComp;
                valid = 1;
            }
        }

        if (outAutoEV) {
            *outAutoEV = autoEV;
        }
        if (outValid) {
            *outValid = valid;
        }
        if (outExposureScale) {
            const double scale64 = exp2(autoEV + sliderEV);
            const float scale = (isfinite(scale64) && scale64 > 0.0) ? static_cast<float>(scale64) : 1.0f;
            *outExposureScale = scale;
        }
    }

} // namespace

extern "C" int juicer_cuda_measure_center_weighted_Y(
    const void* srcDeviceBase,
    std::size_t srcRowBytes,
    int srcBoundsX1,
    int srcBoundsY1,
    int srcBoundsX2,
    int srcBoundsY2,
    int meterX1,
    int meterY1,
    int meterX2,
    int meterY2,
    int nComponents,
    int inputColorSpaceIndex,
    int applyCctfDecoding,
    const float* rgbToXYZ9,
    double* outY,
    void* cudaStreamOpaque,
    const char** outErrorMsg)
{
    thread_local std::string sError;
    if (outErrorMsg) {
        *outErrorMsg = nullptr;
    }
    if (outY) {
        *outY = 0.0;
    }

    if (!srcDeviceBase || srcRowBytes == 0 || !rgbToXYZ9 || !outY) {
        if (outErrorMsg) *outErrorMsg = set_error(sError, "invalid arguments");
        return 1;
    }
    if (!(nComponents == 3 || nComponents == 4)) {
        if (outErrorMsg) *outErrorMsg = set_error(sError, "unsupported component count");
        return 2;
    }

    const int width = meterX2 - meterX1;
    const int height = meterY2 - meterY1;
    if (width <= 0 || height <= 0) {
        *outY = 0.0;
        return 0;
    }

    if (srcBoundsX2 < srcBoundsX1 || srcBoundsY2 < srcBoundsY1) {
        if (outErrorMsg) *outErrorMsg = set_error(sError, "invalid source bounds");
        return 3;
    }

    Mat3 m{};
    for (int i = 0; i < 9; ++i) {
        m.m[i] = rgbToXYZ9[i];
    }

    const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

    int curDevice = -1;
    cudaError_t err = cudaGetDevice(&curDevice);
    if (err != cudaSuccess || curDevice < 0) {
        if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "cudaGetDevice failed: ", err);
        return 12;
    }

    if (gAccum.deviceId != curDevice) {
        if (gAccum.sumY) {
            cudaFree(gAccum.sumY);
            gAccum.sumY = nullptr;
        }
        if (gAccum.sumW) {
            cudaFree(gAccum.sumW);
            gAccum.sumW = nullptr;
        }
        gAccum.deviceId = curDevice;
    }

    if (!gAccum.sumY) {
        err = cudaMalloc(reinterpret_cast<void**>(&gAccum.sumY), sizeof(double));
        if (err != cudaSuccess) {
            if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "cudaMalloc(sumY) failed: ", err);
            return 4;
        }
    }
    if (!gAccum.sumW) {
        err = cudaMalloc(reinterpret_cast<void**>(&gAccum.sumW), sizeof(double));
        if (err != cudaSuccess) {
            if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "cudaMalloc(sumW) failed: ", err);
            return 5;
        }
    }

    double* dSumY = gAccum.sumY;
    double* dSumW = gAccum.sumW;

    err = cudaMemsetAsync(dSumY, 0, sizeof(double), stream);
    if (err != cudaSuccess) {
        if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "cudaMemsetAsync(sumY) failed: ", err);
        return 6;
    }
    err = cudaMemsetAsync(dSumW, 0, sizeof(double), stream);
    if (err != cudaSuccess) {
        if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "cudaMemsetAsync(sumW) failed: ", err);
        return 7;
    }

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    meter_center_weighted_Y_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const unsigned char*>(srcDeviceBase),
        srcRowBytes,
        srcBoundsX1,
        srcBoundsY1,
        srcBoundsX2,
        srcBoundsY2,
        meterX1,
        meterY1,
        width,
        height,
        nComponents,
        inputColorSpaceIndex,
        applyCctfDecoding,
        m,
        dSumY,
        dSumW);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "meter kernel launch failed: ", err);
        return 8;
    }

    double hSumY = 0.0;
    double hSumW = 0.0;
    err = cudaMemcpyAsync(&hSumY, dSumY, sizeof(double), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "cudaMemcpyAsync(sumY) failed: ", err);
        return 9;
    }
    err = cudaMemcpyAsync(&hSumW, dSumW, sizeof(double), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "cudaMemcpyAsync(sumW) failed: ", err);
        return 10;
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "cudaStreamSynchronize failed: ", err);
        return 11;
    }

    if (!(hSumW > 0.0) || !std::isfinite(hSumW) || !std::isfinite(hSumY)) {
        *outY = 0.0;
        return 0;
    }

    *outY = hSumY / hSumW;
    return 0;
}

extern "C" int juicer_cuda_measure_median_Y(
    const void* srcDeviceBase,
    std::size_t srcRowBytes,
    int srcBoundsX1,
    int srcBoundsY1,
    int srcBoundsX2,
    int srcBoundsY2,
    int meterX1,
    int meterY1,
    int meterX2,
    int meterY2,
    int nComponents,
    int inputColorSpaceIndex,
    int applyCctfDecoding,
    const float* rgbToXYZ9,
    double* outY,
    void* cudaStreamOpaque,
    const char** outErrorMsg)
{
    thread_local std::string sError;
    if (outErrorMsg) {
        *outErrorMsg = nullptr;
    }
    if (outY) {
        *outY = 0.0;
    }

    if (!srcDeviceBase || srcRowBytes == 0 || !rgbToXYZ9 || !outY) {
        if (outErrorMsg) *outErrorMsg = set_error(sError, "invalid arguments");
        return 1;
    }
    if (!(nComponents == 3 || nComponents == 4)) {
        if (outErrorMsg) *outErrorMsg = set_error(sError, "unsupported component count");
        return 2;
    }

    const int width = meterX2 - meterX1;
    const int height = meterY2 - meterY1;
    if (width <= 0 || height <= 0) {
        *outY = 0.0;
        return 0;
    }

    if (srcBoundsX2 < srcBoundsX1 || srcBoundsY2 < srcBoundsY1) {
        if (outErrorMsg) *outErrorMsg = set_error(sError, "invalid source bounds");
        return 3;
    }

    Mat3 m{};
    for (int i = 0; i < 9; ++i) {
        m.m[i] = rgbToXYZ9[i];
    }

    const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

    int curDevice = -1;
    cudaError_t err = cudaGetDevice(&curDevice);
    if (err != cudaSuccess || curDevice < 0) {
        if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "cudaGetDevice failed: ", err);
        return 12;
    }

    if (gAccum.deviceId != curDevice) {
        if (gAccum.sumY) {
            cudaFree(gAccum.sumY);
            gAccum.sumY = nullptr;
        }
        if (gAccum.sumW) {
            cudaFree(gAccum.sumW);
            gAccum.sumW = nullptr;
        }
        if (gAccum.maxYBits) {
            cudaFree(gAccum.maxYBits);
            gAccum.maxYBits = nullptr;
        }
        if (gAccum.histogram) {
            cudaFree(gAccum.histogram);
            gAccum.histogram = nullptr;
        }
        gAccum.deviceId = curDevice;
    }

    if (!gAccum.maxYBits) {
        err = cudaMalloc(reinterpret_cast<void**>(&gAccum.maxYBits), sizeof(unsigned int));
        if (err != cudaSuccess) {
            if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "cudaMalloc(maxYBits) failed: ", err);
            return 4;
        }
    }
    if (!gAccum.histogram) {
        err = cudaMalloc(reinterpret_cast<void**>(&gAccum.histogram), sizeof(unsigned int) * kMedianHistogramBins);
        if (err != cudaSuccess) {
            if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "cudaMalloc(histogram) failed: ", err);
            return 5;
        }
    }

    err = cudaMemsetAsync(gAccum.maxYBits, 0, sizeof(unsigned int), stream);
    if (err != cudaSuccess) {
        if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "cudaMemsetAsync(maxYBits) failed: ", err);
        return 6;
    }

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    meter_max_Y_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const unsigned char*>(srcDeviceBase),
        srcRowBytes,
        srcBoundsX1,
        srcBoundsY1,
        srcBoundsX2,
        srcBoundsY2,
        meterX1,
        meterY1,
        width,
        height,
        nComponents,
        inputColorSpaceIndex,
        applyCctfDecoding,
        m,
        gAccum.maxYBits);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "meter_max_Y_kernel launch failed: ", err);
        return 7;
    }

    unsigned int maxBits = 0;
    err = cudaMemcpyAsync(&maxBits, gAccum.maxYBits, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "cudaMemcpyAsync(maxYBits) failed: ", err);
        return 8;
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "cudaStreamSynchronize(maxYBits) failed: ", err);
        return 9;
    }

    float maxY = 0.0f;
    std::memcpy(&maxY, &maxBits, sizeof(float));
    if (!(maxY > 0.0f) || !isfinite(maxY)) {
        *outY = 0.0;
        return 0;
    }

    err = cudaMemsetAsync(gAccum.histogram, 0, sizeof(unsigned int) * kMedianHistogramBins, stream);
    if (err != cudaSuccess) {
        if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "cudaMemsetAsync(histogram) failed: ", err);
        return 10;
    }

    meter_histogram_Y_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const unsigned char*>(srcDeviceBase),
        srcRowBytes,
        srcBoundsX1,
        srcBoundsY1,
        srcBoundsX2,
        srcBoundsY2,
        meterX1,
        meterY1,
        width,
        height,
        nComponents,
        inputColorSpaceIndex,
        applyCctfDecoding,
        m,
        maxY,
        gAccum.histogram);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "meter_histogram_Y_kernel launch failed: ", err);
        return 11;
    }

    std::vector<unsigned int> histogram(kMedianHistogramBins);
    err = cudaMemcpyAsync(histogram.data(), gAccum.histogram, sizeof(unsigned int) * kMedianHistogramBins, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "cudaMemcpyAsync(histogram) failed: ", err);
        return 12;
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "cudaStreamSynchronize(histogram) failed: ", err);
        return 13;
    }

    unsigned long long totalCount = 0;
    for (unsigned int v : histogram) {
        totalCount += static_cast<unsigned long long>(v);
    }
    if (totalCount == 0) {
        *outY = 0.0;
        return 0;
    }

    const unsigned long long target = totalCount / 2ULL;
    unsigned long long cumulative = 0;
    int medianBin = kMedianHistogramBins - 1;
    unsigned int countInBin = 0;
    for (int i = 0; i < kMedianHistogramBins; ++i) {
        const unsigned int count = histogram[i];
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
    *outY = std::isfinite(medianY) ? medianY : 0.0;
    return 0;
}

extern "C" int juicer_cuda_auto_exposure_meter_to_device(
    const void* srcDeviceBase,
    std::size_t srcRowBytes,
    int srcBoundsX1,
    int srcBoundsY1,
    int srcBoundsX2,
    int srcBoundsY2,
    int meterX1,
    int meterY1,
    int meterX2,
    int meterY2,
    int nComponents,
    int inputColorSpaceIndex,
    int applyCctfDecoding,
    const float* rgbToXYZ9,
    int meteringMethod,
    double sliderEV,
    JuicerCudaAutoExposureScratch scratch,
    JuicerCudaAutoExposureDeviceState outState,
    void* cudaStreamOpaque,
    const char** outErrorMsg)
{
    thread_local std::string sError;
    if (outErrorMsg) {
        *outErrorMsg = nullptr;
    }

    if (!srcDeviceBase || srcRowBytes == 0 || !rgbToXYZ9) {
        if (outErrorMsg) *outErrorMsg = set_error(sError, "invalid arguments");
        return 1;
    }
    if (!(nComponents == 3 || nComponents == 4)) {
        if (outErrorMsg) *outErrorMsg = set_error(sError, "unsupported component count");
        return 2;
    }
    if (!outState.exposureScale || !outState.autoEV || !outState.valid) {
        if (outErrorMsg) *outErrorMsg = set_error(sError, "invalid output device state");
        return 3;
    }

    const int width = meterX2 - meterX1;
    const int height = meterY2 - meterY1;
    if (width <= 0 || height <= 0) {
        const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
        reset_auto_exposure_state_kernel<<<1, 1, 0, stream>>>(sliderEV, outState.autoEV, outState.exposureScale, outState.valid);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "reset auto-exposure state failed: ", err);
            return 4;
        }
        return 0;
    }

    if (srcBoundsX2 < srcBoundsX1 || srcBoundsY2 < srcBoundsY1) {
        if (outErrorMsg) *outErrorMsg = set_error(sError, "invalid source bounds");
        return 5;
    }

    Mat3 m{};
    for (int i = 0; i < 9; ++i) {
        m.m[i] = rgbToXYZ9[i];
    }

    const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

    if (meteringMethod == 1) {
        if (!scratch.maxYBits || !scratch.histogram) {
            if (outErrorMsg) *outErrorMsg = set_error(sError, "median metering scratch buffers missing");
            return 6;
        }

        cudaError_t err = cudaMemsetAsync(scratch.maxYBits, 0, sizeof(unsigned int), stream);
        if (err != cudaSuccess) {
            if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "cudaMemsetAsync(maxYBits) failed: ", err);
            return 7;
        }

        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        meter_max_Y_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const unsigned char*>(srcDeviceBase),
            srcRowBytes,
            srcBoundsX1,
            srcBoundsY1,
            srcBoundsX2,
            srcBoundsY2,
            meterX1,
            meterY1,
            width,
            height,
            nComponents,
            inputColorSpaceIndex,
            applyCctfDecoding,
            m,
            scratch.maxYBits);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "meter_max_Y_kernel launch failed: ", err);
            return 8;
        }

        err = cudaMemsetAsync(scratch.histogram, 0, sizeof(unsigned int) * kMedianHistogramBins, stream);
        if (err != cudaSuccess) {
            if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "cudaMemsetAsync(histogram) failed: ", err);
            return 9;
        }

        meter_histogram_Y_bits_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const unsigned char*>(srcDeviceBase),
            srcRowBytes,
            srcBoundsX1,
            srcBoundsY1,
            srcBoundsX2,
            srcBoundsY2,
            meterX1,
            meterY1,
            width,
            height,
            nComponents,
            inputColorSpaceIndex,
            applyCctfDecoding,
            m,
            scratch.maxYBits,
            scratch.histogram);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "meter_histogram kernel launch failed: ", err);
            return 10;
        }

        finalize_auto_exposure_from_histogram_kernel<<<1, 1, 0, stream>>>(
            scratch.maxYBits,
            scratch.histogram,
            sliderEV,
            outState.autoEV,
            outState.exposureScale,
            outState.valid);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "finalize histogram kernel launch failed: ", err);
            return 11;
        }
        return 0;
    }

    if (!scratch.partialsA || !scratch.partialsB || scratch.partialCapacity <= 0) {
        if (outErrorMsg) *outErrorMsg = set_error(sError, "center-weighted metering partial buffers missing");
        return 12;
    }
    if (!scratch.weightsX || !scratch.weightsY) {
        if (outErrorMsg) *outErrorMsg = set_error(sError, "center-weighted metering weights missing");
        return 17;
    }

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    const int blocksCount = static_cast<int>(grid.x * grid.y);
    if (blocksCount > scratch.partialCapacity) {
        if (outErrorMsg) *outErrorMsg = set_error(sError, "center-weighted metering partial buffer capacity too small");
        return 13;
    }

    meter_center_weighted_Y_partials_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const unsigned char*>(srcDeviceBase),
        srcRowBytes,
        srcBoundsX1,
        srcBoundsY1,
        srcBoundsX2,
        srcBoundsY2,
        meterX1,
        meterY1,
        width,
        height,
        nComponents,
        inputColorSpaceIndex,
        applyCctfDecoding,
        m,
        scratch.weightsX,
        scratch.weightsY,
        scratch.partialsA);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "meter_center_weighted partials kernel launch failed: ", err);
        return 14;
    }

    const int reduceThreads = 256;
    int n = blocksCount;
    const JuicerCudaAutoExposurePartial* in = scratch.partialsA;
    JuicerCudaAutoExposurePartial* out = scratch.partialsB;
    while (n > 1) {
        const int blocks = (n + reduceThreads * 2 - 1) / (reduceThreads * 2);
        reduce_auto_exposure_partials_kernel<<<blocks, reduceThreads, 0, stream>>>(in, n, out);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "reduce partials kernel launch failed: ", err);
            return 15;
        }
        n = blocks;
        const JuicerCudaAutoExposurePartial* nextIn = out;
        out = (JuicerCudaAutoExposurePartial*)in;
        in = nextIn;
    }

    finalize_auto_exposure_from_sums_kernel<<<1, 1, 0, stream>>>(
        in,
        sliderEV,
        outState.autoEV,
        outState.exposureScale,
        outState.valid);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "finalize sums kernel launch failed: ", err);
        return 16;
    }

    return 0;
}

extern "C" int juicer_cuda_auto_exposure_update_scale_to_device(
    double sliderEV,
    JuicerCudaAutoExposureDeviceState state,
    void* cudaStreamOpaque,
    const char** outErrorMsg)
{
    thread_local std::string sError;
    if (outErrorMsg) {
        *outErrorMsg = nullptr;
    }

    if (!state.exposureScale || !state.autoEV || !state.valid) {
        if (outErrorMsg) *outErrorMsg = set_error(sError, "invalid output device state");
        return 1;
    }

    const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
    update_auto_exposure_scale_kernel<<<1, 1, 0, stream>>>(sliderEV, state.autoEV, state.valid, state.exposureScale);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "auto-exposure scale update failed: ", err);
        return 2;
    }

    return 0;
}

extern "C" int juicer_cuda_auto_exposure_build_center_weight_tables(
    int meterWidth,
    int meterHeight,
    float* weightsX,
    float* weightsY,
    void* cudaStreamOpaque,
    const char** outErrorMsg)
{
    thread_local std::string sError;
    if (outErrorMsg) {
        *outErrorMsg = nullptr;
    }

    if (meterWidth <= 0 || meterHeight <= 0) {
        if (outErrorMsg) *outErrorMsg = set_error(sError, "invalid meter dimensions");
        return 1;
    }
    if (!weightsX || !weightsY) {
        if (outErrorMsg) *outErrorMsg = set_error(sError, "invalid weights buffers");
        return 2;
    }

    const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
    const int threads = 256;
    const int blocksX = (meterWidth + threads - 1) / threads;
    build_center_weight_x_kernel<<<blocksX, threads, 0, stream>>>(meterWidth, meterHeight, weightsX);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "build_center_weight_x kernel launch failed: ", err);
        return 3;
    }

    const int blocksY = (meterHeight + threads - 1) / threads;
    build_center_weight_y_kernel<<<blocksY, threads, 0, stream>>>(meterWidth, meterHeight, weightsY);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "build_center_weight_y kernel launch failed: ", err);
        return 4;
    }

    return 0;
}
