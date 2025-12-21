// Cuda/JuicerCudaPrintPipeline.cuh
// Shared CUDA device helpers for the print pipeline.
#pragma once

#include "Cuda/JuicerCudaKernelsUtil.cuh"

static __device__ __forceinline__ float density_to_light_sample_agx_device(float density, float illuminant) {
    const double transmitted = pow(10.0, -static_cast<double>(density)) * static_cast<double>(illuminant);
    const float out = static_cast<float>(transmitted);
    return isnan(out) ? 0.0f : out;
}

static __device__ __forceinline__ void apply_print_pipeline_device(const JuicerCuda::Phase3RunParams& params, float D_cmy[3]) {
    if (!params.printActive || !D_cmy) {
        return;
    }

    const int K = params.printIllumK;
    if (K <= 0 || !params.printIllumFiltered) {
        D_cmy[0] = D_cmy[1] = D_cmy[2] = 0.0f;
        return;
    }

    const int negK = params.negTables.K;
    if (negK != K || !params.negTables.epsC || !params.negTables.epsM || !params.negTables.epsY) {
        D_cmy[0] = D_cmy[1] = D_cmy[2] = 0.0f;
        return;
    }

    const bool haveBaseline = (params.negTables.hasBaseline != 0) && params.negTables.baseMin;

    if (!params.printSensC.y || !params.printSensM.y || !params.printSensY.y) {
        D_cmy[0] = D_cmy[1] = D_cmy[2] = 0.0f;
        return;
    }
    if (params.printSensC.n < K || params.printSensM.n < K || params.printSensY.n < K) {
        D_cmy[0] = D_cmy[1] = D_cmy[2] = 0.0f;
        return;
    }

    // Negative density -> filtered enlarger light -> raw print exposures (C/M/Y).
    double accumC = 0.0;
    double accumM = 0.0;
    double accumY = 0.0;
    for (int i = 0; i < K; ++i) {
        const float baseD = haveBaseline ? ldg_f(params.negTables.baseMin + i) : 0.0f;
        const float densitySpectral =
            D_cmy[0] * ldg_f(params.negTables.epsC + i) +
            D_cmy[1] * ldg_f(params.negTables.epsM + i) +
            D_cmy[2] * ldg_f(params.negTables.epsY + i) +
            baseD;

        const float e = density_to_light_sample_agx_device(densitySpectral, ldg_f(params.printIllumFiltered + i));
        if (isnan(e)) {
            continue;
        }
        const double e64 = static_cast<double>(e);

        const float sC = ldg_f(params.printSensC.y + i);
        const float sM = ldg_f(params.printSensM.y + i);
        const float sY = ldg_f(params.printSensY.y + i);
        if (!isnan(sC)) accumC += e64 * static_cast<double>(sC);
        if (!isnan(sM)) accumM += e64 * static_cast<double>(sM);
        if (!isnan(sY)) accumY += e64 * static_cast<double>(sY);
    }

    float rawC = static_cast<float>(accumC);
    float rawM = static_cast<float>(accumM);
    float rawY = static_cast<float>(accumY);

    float expPrint = params.printExposure;
    if (!isfinite(expPrint)) {
        expPrint = 1.0f;
    }
    if (expPrint < 0.0f) {
        expPrint = 0.0f;
    }

    float kMid = params.printMidgrayFactor;
    if (!isfinite(kMid) || !(kMid > 0.0f)) {
        kMid = 1.0f;
    }

    const float rawScale = expPrint * kMid;
    rawC *= rawScale;
    rawM *= rawScale;
    rawY *= rawScale;

    const float preflash = params.printPreflashExposure;
    if (isfinite(preflash) && preflash > 0.0f) {
        rawC += params.printPreflashRaw[0] * preflash;
        rawM += params.printPreflashRaw[1] * preflash;
        rawY += params.printPreflashRaw[2] * preflash;
    }

    // RAW -> log10(raw + eps) -> print density curves.
    constexpr float kLogEps = 1e-10f;
    const float logC = log10f(rawC + kLogEps);
    const float logM = log10f(rawM + kLogEps);
    const float logY = log10f(rawY + kLogEps);

    D_cmy[0] = sample_density_at_logE_device(params.printDcC.x, params.printDcC.y, params.printDcC.n, logC, params.printGammaC);
    D_cmy[1] = sample_density_at_logE_device(params.printDcM.x, params.printDcM.y, params.printDcM.n, logM, params.printGammaM);
    D_cmy[2] = sample_density_at_logE_device(params.printDcY.x, params.printDcY.y, params.printDcY.n, logY, params.printGammaY);
}
