// Cuda/Dir/JuicerCudaSpatialDIR.cu
// Stage-aligned CUDA TU for spatial DIR kernels.
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

#include "Cuda/JuicerCudaPayloads.h"
#include "Cuda/JuicerCudaKernelsUtil.cuh"
#include "Cuda/Film/JuicerCudaFilmExposure.cuh"

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

    __device__ __forceinline__ void compute_dir_corrections_device(
        const JuicerCuda::DirPayload& dir,
        const float dYMC[3],
        float outYMC[3])
    {
        if (!outYMC) {
            return;
        }
        if (!dir.active) {
            outYMC[0] = 0.0f;
            outYMC[1] = 0.0f;
            outYMC[2] = 0.0f;
            return;
        }

        auto safe_norm = [](float D, float dmax) -> float {
            float Din = (!isfinite(D) || D < 0.0f) ? 0.0f : D;
            float m = (isfinite(dmax) && dmax > 1e-4f) ? dmax : 1.0f;
            float n = Din / m;
            if (!isfinite(n) || n < 0.0f) n = 0.0f;
            return n;
        };

        float nB = safe_norm(dYMC[0], dir.dMax[0]);
        float nG = safe_norm(dYMC[1], dir.dMax[1]);
        float nR = safe_norm(dYMC[2], dir.dMax[2]);

        auto high_boost = [&](float n) -> float {
            const float nb = n + dir.highShift * n * n;
            if (!isfinite(nb)) {
                return (n >= 0.0f && isfinite(n)) ? n : 0.0f;
            }
            return fmaxf(0.0f, nb);
        };
        nB = high_boost(nB);
        nG = high_boost(nG);
        nR = high_boost(nR);

        float aY = dir.M[0] * nB + dir.M[3] * nG + dir.M[6] * nR;
        float aM = dir.M[1] * nB + dir.M[4] * nG + dir.M[7] * nR;
        float aC = dir.M[2] * nB + dir.M[5] * nG + dir.M[8] * nR;

        if (!isfinite(aY)) aY = 0.0f;
        if (!isfinite(aM)) aM = 0.0f;
        if (!isfinite(aC)) aC = 0.0f;

        auto clamp_corr = [](float v) -> float {
            if (!isfinite(v)) return 0.0f;
            if (v < -10.0f) return -10.0f;
            if (v > 10.0f) return 10.0f;
            return v;
        };
        outYMC[0] = clamp_corr(aY);
        outYMC[1] = clamp_corr(aM);
        outYMC[2] = clamp_corr(aC);
    }

    __global__ void spatial_dir_corrections_kernel(
        JuicerCuda::PipelineRunParams params,
        float* corrY,
        float* corrM,
        float* corrC)
    {
        if (!params.src || params.srcRowBytes == 0) {
            return;
        }
        if (!corrY || !corrM || !corrC) {
            return;
        }

        const int nC = params.nComponents;
        if (!(nC == 3 || nC == 4)) {
            return;
        }

        const std::size_t pixelBytes = static_cast<std::size_t>(nC) * sizeof(float);
        for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < params.height; y += blockDim.y * gridDim.y) {
            const char* srcRow = reinterpret_cast<const char*>(params.src) + static_cast<std::size_t>(y) * params.srcRowBytes;
            for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < params.width; x += blockDim.x * gridDim.x) {
                const float* srcPix = reinterpret_cast<const float*>(srcRow + static_cast<std::size_t>(x) * pixelBytes);
                if (!srcPix) {
                    continue;
                }

                const float rgbIn[3] = { srcPix[0], srcPix[1], srcPix[2] };
                float logE_raw[3] = { 0.0f, 0.0f, 0.0f };
                float logE_sanitized[3] = { 0.0f, 0.0f, 0.0f };
                float layerPre[3] = { 0.0f, 0.0f, 0.0f };
                compute_logE_and_layer_pre_device(params, rgbIn, logE_raw, logE_sanitized, layerPre);

                const float D_cmy[3] = { layerPre[2], layerPre[1], layerPre[0] };
                const float dYMC[3] = { D_cmy[2], D_cmy[1], D_cmy[0] };

                float outCorr[3] = { 0.0f, 0.0f, 0.0f };
                compute_dir_corrections_device(params.filmDevelop.dir, dYMC, outCorr);

                const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
                corrY[idx] = outCorr[0];
                corrM[idx] = outCorr[1];
                corrC[idx] = outCorr[2];
            }
        }
    }

    __global__ void spatial_dir_blur_vertical_clamp_kernel(
        const float* JUICER_RESTRICT in,
        float* out,
        int width,
        int height,
        const float* JUICER_RESTRICT k,
        int radius)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (!in || !out || !k || radius <= 0) {
            return;
        }

        const bool inBounds = (x < width && y < height);
        const int kLen = 2 * radius + 1;
        const int tileW = blockDim.x;
        const int tileH = blockDim.y + 2 * radius;

        extern __shared__ float shared[];
        float* sWeights = shared;
        float* sTile = shared + kLen;

        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        const int tcount = blockDim.x * blockDim.y;

        for (int i = tid; i < kLen; i += tcount) {
            sWeights[i] = k[i];
        }

        const int blockX = blockIdx.x * blockDim.x;
        const int blockY = blockIdx.y * blockDim.y;
        const int xLocal = threadIdx.x;
        const int xLoad = blockX + xLocal;
        if (xLoad < width) {
            for (int i = threadIdx.y; i < tileH; i += blockDim.y) {
                const int yLoad = blockY + i - radius;
                const int yy = reflect_index_repeat_device(yLoad, height);
                sTile[i * tileW + xLocal] = in[static_cast<size_t>(yy) * static_cast<size_t>(width) + static_cast<size_t>(xLoad)];
            }
        }

        __syncthreads();

        if (!inBounds) {
            return;
        }

        double acc = 0.0;
        const int tileY = threadIdx.y + radius;
        for (int j = -radius; j <= radius; ++j) {
            const float v = sTile[(tileY + j) * tileW + xLocal];
            const float w = sWeights[j + radius];
            acc += static_cast<double>(v) * static_cast<double>(w);
        }

        float outV = (isfinite(acc) && !isnan(acc)) ? static_cast<float>(acc) : 0.0f;
        if (!isfinite(outV) || isnan(outV)) {
            outV = 0.0f;
        }
        if (outV < -10.0f) outV = -10.0f;
        if (outV > 10.0f) outV = 10.0f;

        out[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] = outV;
    }

} // namespace

extern "C" cudaError_t juicer_cuda_build_spatial_dir(
    const JuicerCuda::PipelineRunParams* hParams,
    float* dCorrY,
    float* dCorrM,
    float* dCorrC,
    float* dTmp,
    const float* dKernel,
    int radius,
    void* cudaStreamOpaque)
{
    if (!hParams) {
        return cudaErrorInvalidValue;
    }

    const JuicerCuda::PipelineRunParams params = *hParams;
    if (!params.src || params.srcRowBytes == 0) {
        return cudaErrorInvalidValue;
    }
    if (params.width <= 0 || params.height <= 0) {
        return cudaSuccess;
    }
    if (!(params.nComponents == 3 || params.nComponents == 4)) {
        return cudaErrorInvalidValue;
    }
    if (!dCorrY || !dCorrM || !dCorrC || !dTmp) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

    dim3 threads2D(32, 8);
    dim3 blocks2D(
        static_cast<unsigned int>((params.width + threads2D.x - 1) / threads2D.x),
        static_cast<unsigned int>((params.height + threads2D.y - 1) / threads2D.y));

    spatial_dir_corrections_kernel<<<blocks2D, threads2D, 0, stream>>>(params, dCorrY, dCorrM, dCorrC);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }

    auto blur_plane_in_place = [&](float* plane, float* tmpBuf, const float* k, int r) -> cudaError_t {
        if (!plane || !tmpBuf || !k || r <= 0) {
            return cudaSuccess;
        }
        const int kLen = 2 * r + 1;
        const size_t shmemH = (static_cast<size_t>(kLen) +
            static_cast<size_t>(threads2D.y) * static_cast<size_t>(threads2D.x + 2 * r)) * sizeof(float);
        const size_t shmemV = (static_cast<size_t>(kLen) +
            static_cast<size_t>(threads2D.x) * static_cast<size_t>(threads2D.y + 2 * r)) * sizeof(float);

        optics_blur_horizontal_kernel<<<blocks2D, threads2D, shmemH, stream>>>(plane, tmpBuf, params.width, params.height, k, r);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            return e;
        }
        spatial_dir_blur_vertical_clamp_kernel<<<blocks2D, threads2D, shmemV, stream>>>(tmpBuf, plane, params.width, params.height, k, r);
        return cudaGetLastError();
    };

    if (radius > 0 && dKernel) {
        err = blur_plane_in_place(dCorrY, dTmp, dKernel, radius);
        if (err != cudaSuccess) return err;
        err = blur_plane_in_place(dCorrM, dTmp, dKernel, radius);
        if (err != cudaSuccess) return err;
        err = blur_plane_in_place(dCorrC, dTmp, dKernel, radius);
        if (err != cudaSuccess) return err;
    }

    return cudaGetLastError();
}
