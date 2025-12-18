// Cuda/JuicerCudaAutoExposure.cu
//
// Phase 2: camera auto-exposure metering on CUDA (center-weighted Gaussian mask).
//
// This is a parity-oriented port of the CPU meter in JuicerEffect.cpp, adapted for device pointers.
//
#include "Cuda/JuicerCudaAutoExposure.h"

#include <cuda_runtime.h>

#include <cmath>
#include <string>

namespace {

    struct Mat3 {
        float m[9];
    };

    __device__ __forceinline__ float sanitize_channel(float v) {
        return isfinite(v) ? v : 0.0f;
    }

    __device__ __forceinline__ float decode_bt2020_channel(float v) {
        const float x = fmaxf(0.0f, sanitize_channel(v));
        constexpr float a = 1.09929681f;
        // Threshold is precomputed from: a * pow(b, 0.45) - (a - 1.0), with b = 0.01805397.
        constexpr float threshold = 0.0812428791f;
        if (x < threshold) {
            return x / 4.5f;
        }
        return powf((x + (a - 1.0f)) / a, 1.0f / 0.45f);
    }

    __device__ __forceinline__ void apply_input_cctf_decoding(int inputColorSpaceIndex, int applyCctfDecoding, const float inRgb[3], float outRgb[3]) {
        if (!applyCctfDecoding) {
            outRgb[0] = sanitize_channel(inRgb[0]);
            outRgb[1] = sanitize_channel(inRgb[1]);
            outRgb[2] = sanitize_channel(inRgb[2]);
            return;
        }

        // Mirrors Spectral::apply_input_cctf_decoding:
        // only BT.2020 has non-linear decoding; DWG and ACES are treated as linear.
        // InputColorSpace enum: 0=DWG, 1=BT2020, 2=ACES2065-1.
        if (inputColorSpaceIndex == 1) {
            outRgb[0] = decode_bt2020_channel(inRgb[0]);
            outRgb[1] = decode_bt2020_channel(inRgb[1]);
            outRgb[2] = decode_bt2020_channel(inRgb[2]);
            return;
        }

        outRgb[0] = sanitize_channel(inRgb[0]);
        outRgb[1] = sanitize_channel(inRgb[1]);
        outRgb[2] = sanitize_channel(inRgb[2]);
    }

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
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        double localSumY = 0.0;
        double localSumW = 0.0;

        if (x < width && y < height) {
            const int px = meterX1 + x;
            const int py = meterY1 + y;
            if (px >= srcBoundsX1 && px < srcBoundsX2 && py >= srcBoundsY1 && py < srcBoundsY2) {
                const std::size_t pixelStrideBytes = static_cast<std::size_t>(nComponents) * sizeof(float);
                const unsigned char* rowPtr = srcBase + static_cast<std::size_t>(py - srcBoundsY1) * srcRowBytes;
                const float* pix = reinterpret_cast<const float*>(rowPtr + static_cast<std::size_t>(px - srcBoundsX1) * pixelStrideBytes);

                float inRgb[3] = { pix[0], pix[1], pix[2] };
                float lin[3];
                apply_input_cctf_decoding(inputColorSpaceIndex, applyCctfDecoding, inRgb, lin);

                const float Y = mulY(rgbToXYZ, lin);
                if (isfinite(Y)) {
                // Matches build_center_weight_mask() weighting.
                constexpr float sigma = 0.2f;
                const float nx = (static_cast<float>(x) / static_cast<float>(width)) - 0.5f;
                const float ny = (static_cast<float>(y) / static_cast<float>(height)) - 0.5f;
                const int maxDimInt = (width > height) ? width : height;
                const float maxDim = static_cast<float>(maxDimInt);
                const float invMax = (maxDim > 0.0f) ? (1.0f / maxDim) : 0.0f;
                const float normX = nx * static_cast<float>(width) * invMax;
                const float normY = ny * static_cast<float>(height) * invMax;
                const float r2 = normX * normX + normY * normY;
                const float w = expf(-r2 / (2.0f * sigma * sigma));

                    localSumY = static_cast<double>(Y) * static_cast<double>(w);
                    localSumW = static_cast<double>(w);
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

    double* dSumY = nullptr;
    double* dSumW = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&dSumY), sizeof(double));
    if (err != cudaSuccess) {
        if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "cudaMalloc(sumY) failed: ", err);
        return 4;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&dSumW), sizeof(double));
    if (err != cudaSuccess) {
        cudaFree(dSumY);
        if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "cudaMalloc(sumW) failed: ", err);
        return 5;
    }

    err = cudaMemsetAsync(dSumY, 0, sizeof(double), stream);
    if (err != cudaSuccess) {
        cudaFree(dSumW);
        cudaFree(dSumY);
        if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "cudaMemsetAsync(sumY) failed: ", err);
        return 6;
    }
    err = cudaMemsetAsync(dSumW, 0, sizeof(double), stream);
    if (err != cudaSuccess) {
        cudaFree(dSumW);
        cudaFree(dSumY);
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
        cudaFree(dSumW);
        cudaFree(dSumY);
        if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "meter kernel launch failed: ", err);
        return 8;
    }

    double hSumY = 0.0;
    double hSumW = 0.0;
    err = cudaMemcpyAsync(&hSumY, dSumY, sizeof(double), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(dSumW);
        cudaFree(dSumY);
        if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "cudaMemcpyAsync(sumY) failed: ", err);
        return 9;
    }
    err = cudaMemcpyAsync(&hSumW, dSumW, sizeof(double), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(dSumW);
        cudaFree(dSumY);
        if (outErrorMsg) *outErrorMsg = set_error_cuda(sError, "cudaMemcpyAsync(sumW) failed: ", err);
        return 10;
    }
    err = cudaStreamSynchronize(stream);
    cudaFree(dSumW);
    cudaFree(dSumY);
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
