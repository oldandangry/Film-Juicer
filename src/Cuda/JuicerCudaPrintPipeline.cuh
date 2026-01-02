// Cuda/JuicerCudaPrintPipeline.cuh
// Shared CUDA device helpers for the print pipeline.
#pragma once

#include "Cuda/JuicerCudaKernelsUtil.cuh"

static __device__ __forceinline__ float density_to_light_sample_agx_device(float density, float illuminant) {
    // pow(10, -d) = exp2(-d * log2(10))
    constexpr double kLog2_10 = 3.32192809488736234787;
    const double transmitted = exp2(-static_cast<double>(density) * kLog2_10) * static_cast<double>(illuminant);
    const float out = static_cast<float>(transmitted);
    return isnan(out) ? 0.0f : out;
}

static __device__ __forceinline__ void apply_print_pipeline_device(
    const JuicerCuda::PrintExposePayload& expose,
    const JuicerCuda::PrintDevelopPayload& develop,
    float D_cmy[3]) {
    if (!expose.active || !D_cmy) {
        return;
    }

    const int K = expose.printIllumK;
    if (K <= 0 || !expose.printIllumFiltered) {
        D_cmy[0] = D_cmy[1] = D_cmy[2] = 0.0f;
        return;
    }

    const int negK = expose.negTables.K;
    if (negK != K || !expose.negTables.epsC || !expose.negTables.epsM || !expose.negTables.epsY) {
        D_cmy[0] = D_cmy[1] = D_cmy[2] = 0.0f;
        return;
    }

    const bool haveBaseline = (expose.negTables.hasBaseline != 0) && expose.negTables.baseMin;

    if (!expose.printSensC.y || !expose.printSensM.y || !expose.printSensY.y) {
        D_cmy[0] = D_cmy[1] = D_cmy[2] = 0.0f;
        return;
    }
    if (expose.printSensC.n < K || expose.printSensM.n < K || expose.printSensY.n < K) {
        D_cmy[0] = D_cmy[1] = D_cmy[2] = 0.0f;
        return;
    }

    // Negative density -> filtered enlarger light -> raw print exposures (C/M/Y).
    double accumC = 0.0;
    double accumM = 0.0;
    double accumY = 0.0;
    for (int i = 0; i < K; ++i) {
        const float sC = ldg_f(expose.printSensC.y + i);
        const float sM = ldg_f(expose.printSensM.y + i);
        const float sY = ldg_f(expose.printSensY.y + i);

        const bool activeC = !isnan(sC);
        const bool activeM = !isnan(sM);
        const bool activeY = !isnan(sY);
        if (!activeC && !activeM && !activeY) {
            continue;
        }

        const float baseD = haveBaseline ? ldg_f(expose.negTables.baseMin + i) : 0.0f;
        const float densitySpectral =
            D_cmy[0] * ldg_f(expose.negTables.epsC + i) +
            D_cmy[1] * ldg_f(expose.negTables.epsM + i) +
            D_cmy[2] * ldg_f(expose.negTables.epsY + i) +
            baseD;

        const float e = density_to_light_sample_agx_device(densitySpectral, ldg_f(expose.printIllumFiltered + i));
        const double e64 = static_cast<double>(e);

        if (activeC) accumC += e64 * static_cast<double>(sC);
        if (activeM) accumM += e64 * static_cast<double>(sM);
        if (activeY) accumY += e64 * static_cast<double>(sY);
    }

    float rawC = static_cast<float>(accumC);
    float rawM = static_cast<float>(accumM);
    float rawY = static_cast<float>(accumY);

    float expPrint = expose.printExposure;
    if (!isfinite(expPrint)) {
        expPrint = 1.0f;
    }
    if (expPrint < 0.0f) {
        expPrint = 0.0f;
    }

    float kMid = expose.printMidgrayFactor;
    if (!isfinite(kMid) || !(kMid > 0.0f)) {
        kMid = 1.0f;
    }

    const float rawScale = expPrint * kMid;
    rawC *= rawScale;
    rawM *= rawScale;
    rawY *= rawScale;

    const float preflash = expose.printPreflashExposure;
    if (isfinite(preflash) && preflash > 0.0f) {
        rawC += expose.printPreflashRaw[0] * preflash;
        rawM += expose.printPreflashRaw[1] * preflash;
        rawY += expose.printPreflashRaw[2] * preflash;
    }

    // RAW -> log10(raw + eps) -> print density curves.
    constexpr float kLogEps = 1e-10f;
    const float logC = log10f(rawC + kLogEps);
    const float logM = log10f(rawM + kLogEps);
    const float logY = log10f(rawY + kLogEps);

    D_cmy[0] = sample_density_at_logE_device(develop.printDcC, logC, develop.printGammaC);
    D_cmy[1] = sample_density_at_logE_device(develop.printDcM, logM, develop.printGammaM);
    D_cmy[2] = sample_density_at_logE_device(develop.printDcY, logY, develop.printGammaY);
}
