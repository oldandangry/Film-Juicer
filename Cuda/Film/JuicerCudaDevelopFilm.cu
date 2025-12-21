// Cuda/Film/JuicerCudaDevelopFilm.cu
// Stage-aligned CUDA TU for film development kernels.
#include <cuda_runtime.h>

#include <cstddef>

#include "Cuda/JuicerCudaPayloads.h"
#include "Cuda/JuicerCudaKernelsUtil.cuh"
#include "Cuda/Film/JuicerCudaFilmExposure.cuh"
#include "Cuda/Film/JuicerCudaFilmDevelop.cuh"

__global__ void develop_film_density_kernel(
    JuicerCuda::Phase3RunParams params,
    float* outC,
    float* outM,
    float* outY)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

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

    const std::size_t pixelBytes = static_cast<std::size_t>(nC) * sizeof(float);
    const char* srcRow = reinterpret_cast<const char*>(params.src) + static_cast<std::size_t>(y) * params.srcRowBytes;
    const float* srcPix = reinterpret_cast<const float*>(srcRow + static_cast<std::size_t>(x) * pixelBytes);
    if (!srcPix) {
        return;
    }

    const float rgbIn[3] = { srcPix[0], srcPix[1], srcPix[2] };

    float logE_raw[3] = { 0.0f, 0.0f, 0.0f };
    float logE_sanitized[3] = { 0.0f, 0.0f, 0.0f };
    float layerPre[3] = { 0.0f, 0.0f, 0.0f };
    compute_logE_and_layer_pre_device(params, rgbIn, logE_raw, logE_sanitized, layerPre);

    float D_cmy[3] = { 0.0f, 0.0f, 0.0f };
    const bool useSpatialDir =
        params.spatialDirActive &&
        params.spatialDirCorrY && params.spatialDirCorrM && params.spatialDirCorrC;
    if (useSpatialDir) {
        const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
        const float corrY = params.spatialDirCorrY[idx];
        const float corrM = params.spatialDirCorrM[idx];
        const float corrC = params.spatialDirCorrC[idx];

        float logE_corr[3] = {
            logE_raw[0] - corrY,
            logE_raw[1] - corrM,
            logE_raw[2] - corrC
        };

        const JuicerCuda::DeviceCurveView cB = params.dirPrecorrected ? params.dirDensB : params.densB;
        const JuicerCuda::DeviceCurveView cG = params.dirPrecorrected ? params.dirDensG : params.densG;
        const JuicerCuda::DeviceCurveView cR = params.dirPrecorrected ? params.dirDensR : params.densR;

        logE_corr[0] = sanitize_inf_logE_for_curve_device(logE_corr[0], cB.x, cB.n);
        logE_corr[1] = sanitize_inf_logE_for_curve_device(logE_corr[1], cG.x, cG.n);
        logE_corr[2] = sanitize_inf_logE_for_curve_device(logE_corr[2], cR.x, cR.n);

        const float DY = sample_density_at_logE_device(cB.x, cB.y, cB.n, logE_corr[0], params.gammaFactorB);
        const float DM = sample_density_at_logE_device(cG.x, cG.y, cG.n, logE_corr[1], params.gammaFactorG);
        const float DC = sample_density_at_logE_device(cR.x, cR.y, cR.n, logE_corr[2], params.gammaFactorR);

        D_cmy[0] = DC;
        D_cmy[1] = DM;
        D_cmy[2] = DY;
    }
    else if (params.dir.active) {
        float logE_corr[3] = { logE_sanitized[0], logE_sanitized[1], logE_sanitized[2] };
        apply_dir_runtime_logE_device(logE_corr, layerPre, params.dir, params.densB, params.densG, params.densR);

        const JuicerCuda::DeviceCurveView cB = params.dirPrecorrected ? params.dirDensB : params.densB;
        const JuicerCuda::DeviceCurveView cG = params.dirPrecorrected ? params.dirDensG : params.densG;
        const JuicerCuda::DeviceCurveView cR = params.dirPrecorrected ? params.dirDensR : params.densR;

        const float DY = sample_density_at_logE_device(cB.x, cB.y, cB.n, logE_corr[0], params.gammaFactorB);
        const float DM = sample_density_at_logE_device(cG.x, cG.y, cG.n, logE_corr[1], params.gammaFactorG);
        const float DC = sample_density_at_logE_device(cR.x, cR.y, cR.n, logE_corr[2], params.gammaFactorR);

        D_cmy[0] = DC;
        D_cmy[1] = DM;
        D_cmy[2] = DY;
    }
    else {
        // Map B/G/R layer densities to C/M/Y dyes
        D_cmy[0] = layerPre[2];
        D_cmy[1] = layerPre[1];
        D_cmy[2] = layerPre[0];
    }

    const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
    outC[idx] = D_cmy[0];
    outM[idx] = D_cmy[1];
    outY[idx] = D_cmy[2];
}
