// Cuda/Film/JuicerCudaFilmDevelop.cuh
// Film development helpers shared by CUDA stages.
#pragma once

#include <cmath>

#include "Cuda/JuicerCudaPayloads.h"
#include "Cuda/JuicerCudaKernelsUtil.cuh"

static __device__ __forceinline__ float clamp_to_curve_domain_device(float logE, const JuicerCuda::DeviceCurveView& c) {
    if (!c.x || c.n <= 0) {
        return logE;
    }
    const float xmin = ldg_f(c.x);
    const float xmax = ldg_f(c.x + (c.n - 1));
    if (!isfinite(logE)) {
        return xmin;
    }
    float v = logE;
    v = fmaxf(v, xmin);
    v = fminf(v, xmax);
    return v;
}

static __device__ __forceinline__ void apply_dir_runtime_logE_device(
    float logE_BGR[3],
    const float layerD_BGR[3],
    const JuicerCuda::DirPayload& dir,
    const JuicerCuda::DeviceCurveView& densB,
    const JuicerCuda::DeviceCurveView& densG,
    const JuicerCuda::DeviceCurveView& densR)
{
    if (!logE_BGR || !layerD_BGR) {
        return;
    }
    if (!dir.active) {
        return;
    }

    auto safe_norm = [](float D, float dmax) -> float {
        float Din = (!isfinite(D) || D < 0.0f) ? 0.0f : D;
        float m = (isfinite(dmax) && dmax > 1e-4f) ? dmax : 1.0f;
        float n = Din / m;
        if (!isfinite(n) || n < 0.0f) n = 0.0f;
        return n;
    };

    float nB = safe_norm(layerD_BGR[0], dir.dMax[0]);
    float nG = safe_norm(layerD_BGR[1], dir.dMax[1]);
    float nR = safe_norm(layerD_BGR[2], dir.dMax[2]);

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
    aY = clamp_corr(aY);
    aM = clamp_corr(aM);
    aC = clamp_corr(aC);

    logE_BGR[0] -= aY;
    logE_BGR[1] -= aM;
    logE_BGR[2] -= aC;

    logE_BGR[0] = clamp_to_curve_domain_device(logE_BGR[0], densB);
    logE_BGR[1] = clamp_to_curve_domain_device(logE_BGR[1], densG);
    logE_BGR[2] = clamp_to_curve_domain_device(logE_BGR[2], densR);
}
