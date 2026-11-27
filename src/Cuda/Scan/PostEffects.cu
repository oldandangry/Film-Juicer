#include "Cuda/Scan/PostEffects.h"
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "Cuda/JuicerCudaDeviceHelpers.cuh"

#include "Cuda/JuicerCudaBlur.h"
#include "Cuda/JuicerCudaBlur.cuh"
#include "Cuda/Scan/OutputEncoding.cuh"

using JuicerCuda::apply_output_encoding_device;
using JuicerCuda::reflect_index_half_sample_wide_device;

namespace {

    struct GateTransmittanceSample {
        const float* mask = nullptr;
        int width = 0;
        int height = 0;
        float x = 0.0f;
        float y = 0.0f;
    };

    __device__ __forceinline__ float sample_gate_transmittance_device(GateTransmittanceSample sample) {
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
                        const int yy = reflect_index_half_sample_wide_device(yLoad, height);
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
        const float tx = x - static_cast<float>(ix);
        const float ty = y - static_cast<float>(iy);

        float wx[4];
        float wy[4];
        wx[0] = mitchell_weight_device(tx + 1.0f);
        wx[1] = mitchell_weight_device(tx);
        wx[2] = mitchell_weight_device(tx - 1.0f);
        wx[3] = mitchell_weight_device(tx - 2.0f);
        wy[0] = mitchell_weight_device(ty + 1.0f);
        wy[1] = mitchell_weight_device(ty);
        wy[2] = mitchell_weight_device(ty - 1.0f);
        wy[3] = mitchell_weight_device(ty - 2.0f);

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

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    __global__ void focused_scan_output_encode_kernel(
        const void* source,
        std::size_t sourceRowBytes,
        void* destination,
        std::size_t destinationRowBytes,
        int width,
        int height,
        int nComponents,
        JuicerCuda::OutputEncodingPayload encoding,
        const float* rgbR,
        const float* rgbG,
        const float* rgbB,
        int gateOffsetX,
        int gateOffsetY,
        JuicerCuda::GateWeavePayload weave,
        const float* gateTransmittance,
        int gateTransmittanceWidth,
        int gateTransmittanceHeight) {
        if (!source || !destination || !rgbR || !rgbG || !rgbB ||
            sourceRowBytes == 0 || destinationRowBytes == 0) {
            return;
        }
        const int nC = nComponents;
        if (!(nC == 3 || nC == 4)) {
            return;
        }
        const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= static_cast<unsigned int>(width) ||
            y >= static_cast<unsigned int>(height)) {
            return;
        }
        const std::size_t idx = static_cast<std::size_t>(y) *
                                    static_cast<std::size_t>(width) +
                                static_cast<std::size_t>(x);
        double rgbOut[3]{};
        if (weave.active != 0) {
            const float centerX = 0.5f * static_cast<float>(width - 1);
            const float centerY = 0.5f * static_cast<float>(height - 1);
            const float offsetX = static_cast<float>(x) - centerX;
            const float offsetY = static_cast<float>(y) - centerY;
            const float sourceX = weave.cosRot * offsetX -
                                  weave.sinRot * offsetY +
                                  centerX + weave.dxPx;
            const float sourceY = weave.sinRot * offsetX +
                                  weave.cosRot * offsetY +
                                  centerY + weave.dyPx;
            MitchellPlaneSample planeSample{};
            planeSample.width = width;
            planeSample.height = height;
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
        if (gateTransmittance && gateTransmittanceWidth > 0 && gateTransmittanceHeight > 0) {
            GateTransmittanceSample maskSample{};
            maskSample.mask = gateTransmittance;
            maskSample.width = gateTransmittanceWidth;
            maskSample.height = gateTransmittanceHeight;
            maskSample.x = (static_cast<float>(gateOffsetX + static_cast<int>(x))) * 0.5f - 0.25f;
            maskSample.y = (static_cast<float>(gateOffsetY + static_cast<int>(y))) * 0.5f - 0.25f;
            const double transmittance = static_cast<double>(sample_gate_transmittance_device(maskSample));
            rgbOut[0] *= transmittance;
            rgbOut[1] *= transmittance;
            rgbOut[2] *= transmittance;
        }

        apply_output_encoding_device(encoding, rgbOut);

        const std::size_t pixelBytes = static_cast<std::size_t>(nC) * sizeof(float);
        char* dstRow =
            reinterpret_cast<char*>(destination) + static_cast<std::size_t>(y) * destinationRowBytes;
        float* dstPix =
            reinterpret_cast<float*>(dstRow + static_cast<std::size_t>(x) * pixelBytes);
        dstPix[0] = static_cast<float>(rgbOut[0]);
        dstPix[1] = static_cast<float>(rgbOut[1]);
        dstPix[2] = static_cast<float>(rgbOut[2]);
        if (nC == 4) {
            const char* srcRow =
                reinterpret_cast<const char*>(source) +
                static_cast<std::size_t>(y) * sourceRowBytes;
            const float* srcPix =
                reinterpret_cast<const float*>(srcRow + static_cast<std::size_t>(x) * pixelBytes);
            dstPix[3] = srcPix[3];
        }
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

} // namespace

struct FocusedRgbPlanes {
    float* r = nullptr;
    float* g = nullptr;
    float* b = nullptr;
};

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
    const JuicerCuda::FilmDefectsPayload* gateDefects,
    const JuicerCuda::GateWeavePayload* weave,
    const float* gateTransmittance,
    int gateTransmittanceWidth,
    int gateTransmittanceHeight,
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
    if ((gateTransmittance != nullptr) !=
            (gateTransmittanceWidth > 0 && gateTransmittanceHeight > 0) ||
        (gateTransmittance && !gateDefects)) {
        return cudaErrorInvalidValue;
    }
    cudaStream_t stream =
        cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
    dim3 threads(32, 8);
    dim3 blocks(
        static_cast<unsigned int>((params.width + threads.x - 1) / threads.x),
        static_cast<unsigned int>((params.height + threads.y - 1) / threads.y));

    auto blur_plane = [&](float* plane, const float* kernel, int radius) {
        return JuicerCuda::blur_plane_in_place(
            {plane, dTmp, params.width, params.height, kernel, radius},
            {blocks, threads, stream});
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
            const size_t verticalShared =
                (static_cast<size_t>(kernelLength) +
                 static_cast<size_t>(threads.x) *
                     static_cast<size_t>(threads.y + 2 * unsharpRadius)) *
                sizeof(float);
            const auto localError = JuicerCuda::launch_horizontal_blur(
                {plane, dTmp, params.width, params.height, dUnsharpKernel, unsharpRadius},
                {blocks, threads, stream},
                JuicerCuda::BlurPrecision::Double);
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
        params.src,
        params.srcRowBytes,
        params.dst,
        params.dstRowBytes,
        params.width,
        params.height,
        params.nComponents,
        params.scanStage.scanColor.encoding,
        rgb.r,
        rgb.g,
        rgb.b,
        gateDefects ? gateDefects->roiOffsetX : 0,
        gateDefects ? gateDefects->roiOffsetY : 0,
        weave ? *weave : JuicerCuda::GateWeavePayload{},
        gateTransmittance,
        gateTransmittanceWidth,
        gateTransmittanceHeight);
    return cudaGetLastError();
}

extern "C" cudaError_t juicer_cuda_direct_focused_scanner_post_output(
    const JuicerCuda::FilmScanRunParams* hParams,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    const float* dLensBlurKernel,
    int lensBlurRadius,
    const float* dUnsharpKernel,
    int unsharpRadius,
    float unsharpAmount,
    const JuicerCuda::FilmDefectsPayload* gateDefects,
    const JuicerCuda::GateWeavePayload* weave,
    const float* gateTransmittance,
    int gateTransmittanceWidth,
    int gateTransmittanceHeight,
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
        gateTransmittance,
        gateTransmittanceWidth,
        gateTransmittanceHeight,
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
    const JuicerCuda::FilmDefectsPayload* gateDefects,
    const JuicerCuda::GateWeavePayload* weave,
    const float* gateTransmittance,
    int gateTransmittanceWidth,
    int gateTransmittanceHeight,
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
        gateTransmittance,
        gateTransmittanceWidth,
        gateTransmittanceHeight,
        cudaStreamOpaque);
}
