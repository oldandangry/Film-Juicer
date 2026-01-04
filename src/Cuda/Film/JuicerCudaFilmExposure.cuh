// Cuda/Film/JuicerCudaFilmExposure.cuh
// Film exposure helpers shared by CUDA stages.
#pragma once

#include <cmath>

#include "Cuda/JuicerCudaPayloads.h"
#include "Cuda/JuicerCudaKernelsUtil.cuh"

static __device__ __forceinline__ void convert_input_to_DWG_device(
    const JuicerCuda::FilmRawPayload& cfg,
    const float rgbIn[3],
    float rgbDWG[3],
    bool clampNonNegative)
{
    float linear[3];
    apply_input_cctf_decoding_device(cfg.inputColorSpaceIndex, cfg.applyCctfDecoding, rgbIn, linear);

    float XYZ[3];
    mat3_mul9_device(cfg.inputRGBToXYZ, linear, XYZ);

    const float* xyzPtr = XYZ;
    float adapted[3];
    if (cfg.applyInputChromaticAdapt) {
        mat3_mul9_device(cfg.inputXYZAdapt, XYZ, adapted);
        xyzPtr = adapted;
    }

    float dwg[3];
    if (clampNonNegative) {
        XYZ_to_DWG_linear_device(xyzPtr, dwg);
    }
    else {
        XYZ_to_DWG_linear_unclamped_device(xyzPtr, dwg);
    }
    rgbDWG[0] = dwg[0];
    rgbDWG[1] = dwg[1];
    rgbDWG[2] = dwg[2];
}

static __device__ void hanatos_layer_exposures_device(
    const float rgbDWG[3],
    const float* JUICER_RESTRICT hanatosLut,
    int hanatosN,
    const float refIllumWhiteXYZ[3],
    const float* JUICER_RESTRICT sensB,
    const float* JUICER_RESTRICT sensG,
    const float* JUICER_RESTRICT sensR,
    float E_out[3])
{
    if (!E_out) {
        return;
    }
    if (!hanatosLut || hanatosN <= 0 || !sensB || !sensG || !sensR) {
        E_out[0] = E_out[1] = E_out[2] = 0.0f;
        return;
    }

    // Convert DWG RGB to XYZ (D65).
    const float DWG_RGB_to_XYZ[9] = {
        0.70062239f,  0.14877482f,  0.10105872f,
        0.27411851f,  0.87363190f, -0.14775041f,
       -0.09896291f, -0.13789533f,  1.32591599f
    };

    float XYZ[3] = {
        DWG_RGB_to_XYZ[0] * rgbDWG[0] + DWG_RGB_to_XYZ[1] * rgbDWG[1] + DWG_RGB_to_XYZ[2] * rgbDWG[2],
        DWG_RGB_to_XYZ[3] * rgbDWG[0] + DWG_RGB_to_XYZ[4] * rgbDWG[1] + DWG_RGB_to_XYZ[5] * rgbDWG[2],
        DWG_RGB_to_XYZ[6] * rgbDWG[0] + DWG_RGB_to_XYZ[7] * rgbDWG[1] + DWG_RGB_to_XYZ[8] * rgbDWG[2]
    };
    XYZ[0] = device_sanitize_channel(XYZ[0]);
    XYZ[1] = device_sanitize_channel(XYZ[1]);
    XYZ[2] = device_sanitize_channel(XYZ[2]);

    const float D65[3] = { 0.950455f, 1.0f, 1.089058f };

    float refWhite[3] = {
        device_sanitize_channel(refIllumWhiteXYZ[0]),
        device_sanitize_channel(refIllumWhiteXYZ[1]),
        device_sanitize_channel(refIllumWhiteXYZ[2])
    };
    if (!(refWhite[1] > 0.0f)) {
        refWhite[0] = D65[0];
        refWhite[1] = D65[1];
        refWhite[2] = D65[2];
    }

    float adaptedXYZ[3];
    chromatic_adapt_XYZ_CAT02_device(XYZ, D65, refWhite, adaptedXYZ);
    adaptedXYZ[0] = device_sanitize_channel(adaptedXYZ[0]);
    adaptedXYZ[1] = device_sanitize_channel(adaptedXYZ[1]);
    adaptedXYZ[2] = device_sanitize_channel(adaptedXYZ[2]);

    const float sumXYZ = adaptedXYZ[0] + adaptedXYZ[1] + adaptedXYZ[2];
    const float denom = fmaxf(sumXYZ, 1e-10f);

    float x = adaptedXYZ[0] / denom;
    float y = adaptedXYZ[1] / denom;
    x = fminf(1.0f, fmaxf(0.0f, x));
    y = fminf(1.0f, fmaxf(0.0f, y));

    float qx, qy;
    tri2quad_device(x, y, qx, qy);

    const int N = hanatosN;
    const int K = 81;
    const float fx = fminf(1.0f, fmaxf(0.0f, qx)) * static_cast<float>(N - 1);
    const float fy = fminf(1.0f, fmaxf(0.0f, qy)) * static_cast<float>(N - 1);
    int x0 = static_cast<int>(floorf(fx));
    int y0 = static_cast<int>(floorf(fy));
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x0 > N - 1) x0 = N - 1;
    if (y0 > N - 1) y0 = N - 1;
    const int x1 = (x0 + 1 <= N - 1) ? (x0 + 1) : (N - 1);
    const int y1 = (y0 + 1 <= N - 1) ? (y0 + 1) : (N - 1);
    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);

    double Eb = 0.0;
    double Eg = 0.0;
    double Er = 0.0;
    for (int k = 0; k < K; ++k) {
        const float v00 = hanatos_bilinear_at(hanatosLut, N, K, x0, y0, k);
        const float v10 = hanatos_bilinear_at(hanatosLut, N, K, x1, y0, k);
        const float v01 = hanatos_bilinear_at(hanatosLut, N, K, x0, y1, k);
        const float v11 = hanatos_bilinear_at(hanatosLut, N, K, x1, y1, k);
        const float v0 = v00 * (1.0f - tx) + v10 * tx;
        const float v1 = v01 * (1.0f - tx) + v11 * tx;
        const float raw = v0 * (1.0f - ty) + v1 * ty;

        const float e = device_sanitize_channel(sumXYZ * raw);
        if (!device_isfinite(e)) {
            continue;
        }
        const double e64 = static_cast<double>(e);

        const float sb = ldg_f(sensB + k);
        const float sg = ldg_f(sensG + k);
        const float sr = ldg_f(sensR + k);
        if (isfinite(sb)) Eb += e64 * static_cast<double>(sb);
        if (isfinite(sg)) Eg += e64 * static_cast<double>(sg);
        if (isfinite(sr)) Er += e64 * static_cast<double>(sr);
    }

    E_out[0] = device_isfinite(static_cast<float>(Eb)) ? static_cast<float>(Eb) : 0.0f;
    E_out[1] = device_isfinite(static_cast<float>(Eg)) ? static_cast<float>(Eg) : 0.0f;
    E_out[2] = device_isfinite(static_cast<float>(Er)) ? static_cast<float>(Er) : 0.0f;
}

static __device__ void hanatos_integrated_exposures_device(
    const float rgbDWG[3],
    const float* JUICER_RESTRICT lutIntegrated,
    int hanatosN,
    const float refIllumWhiteXYZ[3],
    float E_out[3])
{
    if (!E_out) {
        return;
    }
    if (!lutIntegrated || hanatosN <= 0) {
        E_out[0] = E_out[1] = E_out[2] = 0.0f;
        return;
    }

    // Convert DWG RGB to XYZ (D65).
    const float DWG_RGB_to_XYZ[9] = {
        0.70062239f,  0.14877482f,  0.10105872f,
        0.27411851f,  0.87363190f, -0.14775041f,
       -0.09896291f, -0.13789533f,  1.32591599f
    };

    float XYZ[3] = {
        DWG_RGB_to_XYZ[0] * rgbDWG[0] + DWG_RGB_to_XYZ[1] * rgbDWG[1] + DWG_RGB_to_XYZ[2] * rgbDWG[2],
        DWG_RGB_to_XYZ[3] * rgbDWG[0] + DWG_RGB_to_XYZ[4] * rgbDWG[1] + DWG_RGB_to_XYZ[5] * rgbDWG[2],
        DWG_RGB_to_XYZ[6] * rgbDWG[0] + DWG_RGB_to_XYZ[7] * rgbDWG[1] + DWG_RGB_to_XYZ[8] * rgbDWG[2]
    };
    XYZ[0] = device_sanitize_channel(XYZ[0]);
    XYZ[1] = device_sanitize_channel(XYZ[1]);
    XYZ[2] = device_sanitize_channel(XYZ[2]);

    const float D65[3] = { 0.950455f, 1.0f, 1.089058f };

    float refWhite[3] = {
        device_sanitize_channel(refIllumWhiteXYZ[0]),
        device_sanitize_channel(refIllumWhiteXYZ[1]),
        device_sanitize_channel(refIllumWhiteXYZ[2])
    };
    if (!(refWhite[1] > 0.0f)) {
        refWhite[0] = D65[0];
        refWhite[1] = D65[1];
        refWhite[2] = D65[2];
    }

    float adaptedXYZ[3];
    chromatic_adapt_XYZ_CAT02_device(XYZ, D65, refWhite, adaptedXYZ);
    adaptedXYZ[0] = device_sanitize_channel(adaptedXYZ[0]);
    adaptedXYZ[1] = device_sanitize_channel(adaptedXYZ[1]);
    adaptedXYZ[2] = device_sanitize_channel(adaptedXYZ[2]);

    const float sumXYZ = adaptedXYZ[0] + adaptedXYZ[1] + adaptedXYZ[2];
    const float denom = fmaxf(sumXYZ, 1e-10f);

    float x = adaptedXYZ[0] / denom;
    float y = adaptedXYZ[1] / denom;
    x = fminf(1.0f, fmaxf(0.0f, x));
    y = fminf(1.0f, fmaxf(0.0f, y));

    float qx, qy;
    tri2quad_device(x, y, qx, qy);

    const int N = hanatosN;
    const float fx = fminf(1.0f, fmaxf(0.0f, qx)) * static_cast<float>(N - 1);
    const float fy = fminf(1.0f, fmaxf(0.0f, qy)) * static_cast<float>(N - 1);
    int x0 = static_cast<int>(floorf(fx));
    int y0 = static_cast<int>(floorf(fy));
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x0 > N - 1) x0 = N - 1;
    if (y0 > N - 1) y0 = N - 1;
    const int x1 = (x0 + 1 <= N - 1) ? (x0 + 1) : (N - 1);
    const int y1 = (y0 + 1 <= N - 1) ? (y0 + 1) : (N - 1);
    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);

    const float r00 = hanatos_integrated_at(lutIntegrated, N, x0, y0, 0);
    const float r10 = hanatos_integrated_at(lutIntegrated, N, x1, y0, 0);
    const float r01 = hanatos_integrated_at(lutIntegrated, N, x0, y1, 0);
    const float r11 = hanatos_integrated_at(lutIntegrated, N, x1, y1, 0);

    const float g00 = hanatos_integrated_at(lutIntegrated, N, x0, y0, 1);
    const float g10 = hanatos_integrated_at(lutIntegrated, N, x1, y0, 1);
    const float g01 = hanatos_integrated_at(lutIntegrated, N, x0, y1, 1);
    const float g11 = hanatos_integrated_at(lutIntegrated, N, x1, y1, 1);

    const float b00 = hanatos_integrated_at(lutIntegrated, N, x0, y0, 2);
    const float b10 = hanatos_integrated_at(lutIntegrated, N, x1, y0, 2);
    const float b01 = hanatos_integrated_at(lutIntegrated, N, x0, y1, 2);
    const float b11 = hanatos_integrated_at(lutIntegrated, N, x1, y1, 2);

    const float r0 = r00 * (1.0f - tx) + r10 * tx;
    const float r1 = r01 * (1.0f - tx) + r11 * tx;
    const float g0 = g00 * (1.0f - tx) + g10 * tx;
    const float g1 = g01 * (1.0f - tx) + g11 * tx;
    const float b0 = b00 * (1.0f - tx) + b10 * tx;
    const float b1 = b01 * (1.0f - tx) + b11 * tx;

    const float r = r0 * (1.0f - ty) + r1 * ty;
    const float g = g0 * (1.0f - ty) + g1 * ty;
    const float b = b0 * (1.0f - ty) + b1 * ty;

    const float rSafe = device_isfinite(r) ? r : 0.0f;
    const float gSafe = device_isfinite(g) ? g : 0.0f;
    const float bSafe = device_isfinite(b) ? b : 0.0f;

    // lutIntegrated stores R,G,B; map to E_out order B,G,R.
    E_out[0] = device_sanitize_channel(sumXYZ * bSafe);
    E_out[1] = device_sanitize_channel(sumXYZ * gSafe);
    E_out[2] = device_sanitize_channel(sumXYZ * rSafe);
}

static __device__ void tables_layer_exposures_device(
    const float rgbDWG[3],
    const float S_inv[9],
    const float refIllumWhiteXYZ[3],
    const float* JUICER_RESTRICT Ax,
    const float* JUICER_RESTRICT Ay,
    const float* JUICER_RESTRICT Az,
    const float* JUICER_RESTRICT sensB,
    const float* JUICER_RESTRICT sensG,
    const float* JUICER_RESTRICT sensR,
    float E_out[3])
{
    if (!E_out) {
        return;
    }
    if (!rgbDWG || !S_inv || !refIllumWhiteXYZ || !Ax || !Ay || !Az || !sensB || !sensG || !sensR) {
        E_out[0] = E_out[1] = E_out[2] = 0.0f;
        return;
    }

    const float DWG_RGB_to_XYZ[9] = {
        0.70062239f,  0.14877482f,  0.10105872f,
        0.27411851f,  0.87363190f, -0.14775041f,
       -0.09896291f, -0.13789533f,  1.32591599f
    };

    float XYZ[3] = {
        DWG_RGB_to_XYZ[0] * rgbDWG[0] + DWG_RGB_to_XYZ[1] * rgbDWG[1] + DWG_RGB_to_XYZ[2] * rgbDWG[2],
        DWG_RGB_to_XYZ[3] * rgbDWG[0] + DWG_RGB_to_XYZ[4] * rgbDWG[1] + DWG_RGB_to_XYZ[5] * rgbDWG[2],
        DWG_RGB_to_XYZ[6] * rgbDWG[0] + DWG_RGB_to_XYZ[7] * rgbDWG[1] + DWG_RGB_to_XYZ[8] * rgbDWG[2]
    };

    float sanitizedXYZ[3] = {
        device_sanitize_nonneg(XYZ[0]),
        device_sanitize_nonneg(XYZ[1]),
        device_sanitize_nonneg(XYZ[2])
    };

    const float D65[3] = { 0.950455f, 1.0f, 1.089058f };

    float refWhite[3] = {
        device_sanitize_nonneg(refIllumWhiteXYZ[0]),
        device_sanitize_nonneg(refIllumWhiteXYZ[1]),
        device_sanitize_nonneg(refIllumWhiteXYZ[2])
    };
    if (!(refWhite[1] > 0.0f)) {
        refWhite[0] = D65[0];
        refWhite[1] = D65[1];
        refWhite[2] = D65[2];
    }

    float adaptedXYZ[3];
    chromatic_adapt_XYZ_CAT02_device(sanitizedXYZ, D65, refWhite, adaptedXYZ);
    adaptedXYZ[0] = fmaxf(0.0f, adaptedXYZ[0]);
    adaptedXYZ[1] = fmaxf(0.0f, adaptedXYZ[1]);
    adaptedXYZ[2] = fmaxf(0.0f, adaptedXYZ[2]);

    const float targetScale = (adaptedXYZ[1] > 0.0f) ? adaptedXYZ[1] : sanitizedXYZ[1];

    float cx =
        S_inv[0] * adaptedXYZ[0] +
        S_inv[1] * adaptedXYZ[1] +
        S_inv[2] * adaptedXYZ[2];
    float cy =
        S_inv[3] * adaptedXYZ[0] +
        S_inv[4] * adaptedXYZ[1] +
        S_inv[5] * adaptedXYZ[2];
    float cz =
        S_inv[6] * adaptedXYZ[0] +
        S_inv[7] * adaptedXYZ[1] +
        S_inv[8] * adaptedXYZ[2];
    cx = fmaxf(0.0f, cx);
    cy = fmaxf(0.0f, cy);
    cz = fmaxf(0.0f, cz);

    double Y_recon = 0.0;
    double Eb = 0.0;
    double Eg = 0.0;
    double Er = 0.0;

    const int K = 81;
    for (int i = 0; i < K; ++i) {
        const float bx = fmaxf(0.0f, device_sanitize_nonneg(ldg_f(Ax + i)));
        const float by = fmaxf(0.0f, device_sanitize_nonneg(ldg_f(Ay + i)));
        const float bz = fmaxf(0.0f, device_sanitize_nonneg(ldg_f(Az + i)));
        const float Ei = fmaxf(1e-6f, cx * bx + cy * by + cz * bz);

        Y_recon += static_cast<double>(Ei) * static_cast<double>(by);

        const float sb = ldg_f(sensB + i);
        const float sg = ldg_f(sensG + i);
        const float sr = ldg_f(sensR + i);
        const double e64 = static_cast<double>(Ei);
        if (isfinite(sb)) Eb += e64 * static_cast<double>(sb);
        if (isfinite(sg)) Eg += e64 * static_cast<double>(sg);
        if (isfinite(sr)) Er += e64 * static_cast<double>(sr);
    }

    if (Y_recon > 1e-20 && targetScale > 0.0f) {
        const double s = static_cast<double>(targetScale) / Y_recon;
        Eb *= s;
        Eg *= s;
        Er *= s;
    }
    else if (!(targetScale > 0.0f)) {
        Eb = Eg = Er = 0.0;
    }

    // Mallett path uses applyDeltaLambda=true (Δλ=5nm) and exposureScale=1.0 for raw film exposure.
    const double dl = 5.0;
    E_out[0] = fmaxf(0.0f, static_cast<float>(Eb * dl));
    E_out[1] = fmaxf(0.0f, static_cast<float>(Eg * dl));
    E_out[2] = fmaxf(0.0f, static_cast<float>(Er * dl));
}

static __device__ __forceinline__ void compute_film_raw_device(
    const JuicerCuda::PipelineRunParams& params,
    const float rgbIn[3],
    float filmRaw[3])
{
    const JuicerCuda::FilmExposurePayload& expose = params.filmExpose;

    float E_raw[3] = { 0.0f, 0.0f, 0.0f };
    const bool allowHanatos = (params.filmRaw.spectralUpsamplingMode == 0);
    const bool spdReady = expose.tablesAx && expose.tablesAy && expose.tablesAz && expose.tablesK == 81;
    const bool useHanatos = allowHanatos &&
        spdReady &&
        expose.hanatosLut &&
        (expose.hanatosN > 0) &&
        (expose.sensB.n >= 81) &&
        (expose.sensG.n >= 81) &&
        (expose.sensR.n >= 81);
    const bool useHanatosIntegrated =
        useHanatos &&
        expose.hanatosLutIntegrated &&
        (expose.hanatosNIntegrated > 0) &&
        (expose.hanatosNIntegrated == expose.hanatosN);
    const bool canTables =
        spdReady &&
        expose.sensB.y && expose.sensG.y && expose.sensR.y &&
        (expose.sensB.n >= 81) &&
        (expose.sensG.n >= 81) &&
        (expose.sensR.n >= 81);

    float rgbDWG[3];
    convert_input_to_DWG_device(params.filmRaw, rgbIn, rgbDWG, !useHanatos);

    if (useHanatosIntegrated) {
        hanatos_integrated_exposures_device(
            rgbDWG,
            expose.hanatosLutIntegrated,
            expose.hanatosNIntegrated,
            params.filmRaw.refIllumWhiteXYZ,
            E_raw);
    }
    else if (useHanatos) {
        hanatos_layer_exposures_device(
            rgbDWG,
            expose.hanatosLut,
            expose.hanatosN,
            params.filmRaw.refIllumWhiteXYZ,
            expose.sensB.y,
            expose.sensG.y,
            expose.sensR.y,
            E_raw);
    }
    else if (canTables) {
        tables_layer_exposures_device(
            rgbDWG,
            expose.spdSInv,
            params.filmRaw.refIllumWhiteXYZ,
            expose.tablesAx,
            expose.tablesAy,
            expose.tablesAz,
            expose.sensB.y,
            expose.sensG.y,
            expose.sensR.y,
            E_raw);
    }

    float midgrayScale = params.filmRaw.midgrayScale;
    if (!isfinite(midgrayScale) || !(midgrayScale > 0.0f)) {
        midgrayScale = 1.0f;
    }

    float exposureScale = expose.exposureScale;
    if (!isfinite(exposureScale) || !(exposureScale > 0.0f)) {
        exposureScale = 1.0f;
    }
    if (expose.exposureScaleDevice) {
        float deviceScale = *expose.exposureScaleDevice;
        if (!isfinite(deviceScale) || !(deviceScale > 0.0f)) {
            deviceScale = 1.0f;
        }
        exposureScale = deviceScale;
    }

    for (int i = 0; i < 3; ++i) {
        float v = E_raw[i];
        if (!isfinite(v) || v < 0.0f) v = 0.0f;
        v = fmaxf(0.0f, v * midgrayScale);
        v = fmaxf(0.0f, v * exposureScale);
        filmRaw[i] = v;
    }
}

static __device__ __forceinline__ void compute_logE_raw_from_film_raw_device(
    const JuicerCuda::PipelineRunParams& params,
    const float filmRaw[3],
    float logE_raw[3])
{
    (void)params;
    constexpr float kLogEps = 1e-10f;
    logE_raw[0] = log10f(fmaxf(filmRaw[0], 0.0f) + kLogEps);
    logE_raw[1] = log10f(fmaxf(filmRaw[1], 0.0f) + kLogEps);
    logE_raw[2] = log10f(fmaxf(filmRaw[2], 0.0f) + kLogEps);
}

static __device__ __forceinline__ void compute_logE_raw_device(
    const JuicerCuda::PipelineRunParams& params,
    const float rgbIn[3],
    float logE_raw[3])
{
    float filmRaw[3] = { 0.0f, 0.0f, 0.0f };
    compute_film_raw_device(params, rgbIn, filmRaw);
    compute_logE_raw_from_film_raw_device(params, filmRaw, logE_raw);
}

static __device__ __forceinline__ void compute_logE_from_film_raw_device(
    const JuicerCuda::PipelineRunParams& params,
    const float filmRaw[3],
    float logE_raw[3],
    float logE_sanitized[3],
    float layerPre[3])
{
    const JuicerCuda::FilmDevelopPayload& develop = params.filmDevelop;

    compute_logE_raw_from_film_raw_device(params, filmRaw, logE_raw);

    logE_sanitized[0] = sanitize_inf_logE_for_curve_device(logE_raw[0], develop.densB);
    logE_sanitized[1] = sanitize_inf_logE_for_curve_device(logE_raw[1], develop.densG);
    logE_sanitized[2] = sanitize_inf_logE_for_curve_device(logE_raw[2], develop.densR);

    layerPre[0] = sample_density_at_logE_device(develop.densB, logE_sanitized[0], develop.gammaFactorB);
    layerPre[1] = sample_density_at_logE_device(develop.densG, logE_sanitized[1], develop.gammaFactorG);
    layerPre[2] = sample_density_at_logE_device(develop.densR, logE_sanitized[2], develop.gammaFactorR);
}

static __device__ __forceinline__ void compute_logE_and_layer_pre_device(
    const JuicerCuda::PipelineRunParams& params,
    const float rgbIn[3],
    float logE_raw[3],
    float logE_sanitized[3],
    float layerPre[3])
{
    float filmRaw[3] = { 0.0f, 0.0f, 0.0f };
    compute_film_raw_device(params, rgbIn, filmRaw);
    compute_logE_from_film_raw_device(params, filmRaw, logE_raw, logE_sanitized, layerPre);
}
