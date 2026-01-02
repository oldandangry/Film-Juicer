// Cuda/JuicerCudaKernelsUtil.cuh
// Shared CUDA device helpers (header-only).
#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "Cuda/JuicerCudaPayloads.h"

static __device__ __forceinline__ bool device_isfinite(float v) {
    return isfinite(v);
}

static __device__ __forceinline__ float device_sanitize_channel(float v) {
    return device_isfinite(v) ? v : 0.0f;
}

static __device__ __forceinline__ float device_sanitize_nonneg(float v) {
    return device_isfinite(v) ? fmaxf(0.0f, v) : 0.0f;
}

static __device__ __forceinline__ float ldg_f(const float* p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
    return __ldg(p);
#else
    return *p;
#endif
}

static __device__ __forceinline__ double ldg_d(const double* p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
    return __ldg(p);
#else
    return *p;
#endif
}

static __device__ __forceinline__ float sample_density_at_logE_device(
    const JuicerCuda::DeviceCurveView& curve,
    float logE,
    float gammaFactor)
{
    if (!curve.x || !curve.y || curve.n <= 0) {
        return 0.0f;
    }

    // Preserve NaN query behavior (matches CPU).
    if (!device_isfinite(logE)) {
        return nanf("");
    }

    const float gammaSafe = (device_isfinite(gammaFactor) && gammaFactor > 0.0f) ? gammaFactor : 1.0f;
    const float xq = logE * gammaSafe;

    int domainBegin = curve.domainBegin;
    int domainEnd = curve.domainEnd;
    if (domainBegin < 0) {
        domainBegin = 0;
    }
    if (domainEnd >= curve.n) {
        domainEnd = curve.n - 1;
    }
    if (domainBegin >= curve.n || domainEnd < domainBegin) {
        return 0.0f;
    }

    const float xmin = ldg_f(curve.x + domainBegin);
    const float xmax = ldg_f(curve.x + domainEnd);
    if (!device_isfinite(xmin) || !device_isfinite(xmax) || !(xmax >= xmin)) {
        return ldg_f(curve.y + domainBegin);
    }

    // fast_interp endpoints: literal y[0]/y[-1], even if NaN.
    if (xq <= xmin) {
        return ldg_f(curve.y + domainBegin);
    }
    if (xq >= xmax) {
        return ldg_f(curve.y + domainEnd);
    }

    // lower_bound: find first i1 in [domainBegin+1, domainEnd] where x[i1] >= xq.
    int left = domainBegin + 1;
    int right = domainEnd;
    int i1 = domainEnd + 1;
    while (left <= right) {
        const int mid = left + ((right - left) >> 1);
        const float xm = ldg_f(curve.x + mid);
        if (!(xm < xq)) {
            i1 = mid;
            right = mid - 1;
        }
        else {
            left = mid + 1;
        }
    }
    if (i1 > domainEnd) {
        return ldg_f(curve.y + domainEnd);
    }

    const int i0 = i1 - 1;
    const float x0 = ldg_f(curve.x + i0);
    const float x1 = ldg_f(curve.x + i1);
    const float y0 = ldg_f(curve.y + i0);
    const float y1 = ldg_f(curve.y + i1);

    const float denom = x1 - x0;
    if (!(denom > 0.0f) || !device_isfinite(denom)) {
        return y0;
    }

    const float t = (xq - x0) / denom;
    return y0 + t * (y1 - y0);
}

static __device__ __forceinline__ float sample_density_at_logE_device(
    const float* JUICER_RESTRICT x,
    const float* JUICER_RESTRICT y,
    int n,
    float logE,
    float gammaFactor)
{
    if (!x || !y || n <= 0) {
        return 0.0f;
    }

    int domainBegin = 0;
    while (domainBegin < n && !device_isfinite(ldg_f(x + domainBegin))) {
        ++domainBegin;
    }
    if (domainBegin >= n) {
        return 0.0f;
    }
    int domainEnd = n - 1;
    while (domainEnd > domainBegin && !device_isfinite(ldg_f(x + domainEnd))) {
        --domainEnd;
    }

    const JuicerCuda::DeviceCurveView curve = { x, y, n, domainBegin, domainEnd };
    return sample_density_at_logE_device(curve, logE, gammaFactor);
}

static __device__ __forceinline__ float sanitize_inf_logE_for_curve_device(float logE, const JuicerCuda::DeviceCurveView& curve) {
    if (device_isfinite(logE) || isnan(logE)) {
        return logE;
    }
    if (!curve.x || curve.n <= 0) {
        return logE;
    }

    int begin = curve.domainBegin;
    int end = curve.domainEnd;
    if (begin < 0) {
        begin = 0;
    }
    if (end >= curve.n) {
        end = curve.n - 1;
    }
    if (begin >= curve.n || end < begin) {
        return logE;
    }

    const float xmin = ldg_f(curve.x + begin);
    const float xmax = ldg_f(curve.x + end);
    if (!device_isfinite(xmin) || !device_isfinite(xmax) || !(xmax >= xmin)) {
        return logE;
    }
    return (logE > 0.0f) ? xmax : xmin;
}

static __device__ __forceinline__ float sanitize_inf_logE_for_curve_device(float logE, const float* JUICER_RESTRICT x, int n) {
    if (device_isfinite(logE) || isnan(logE)) {
        return logE;
    }
    if (!x || n <= 0) {
        return logE;
    }

    int begin = 0;
    while (begin < n && !device_isfinite(ldg_f(x + begin))) {
        ++begin;
    }
    if (begin >= n) {
        return logE;
    }
    int end = n - 1;
    while (end > begin && !device_isfinite(ldg_f(x + end))) {
        --end;
    }

    const JuicerCuda::DeviceCurveView curve = { x, /*y*/nullptr, n, begin, end };
    return sanitize_inf_logE_for_curve_device(logE, curve);
}

struct Mat3 {
    float m[9];
};

static __device__ __forceinline__ void mul3(const float m[9], const float v[3], float dst[3]) {
    dst[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
    dst[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
    dst[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
}

static __device__ __forceinline__ void chromatic_adapt_XYZ_CAT02_device(
    const float XYZ[3],
    const float srcWhiteXYZ[3],
    const float dstWhiteXYZ[3],
    float outXYZ[3])
{
    // Matches Spectral::chromatic_adapt_XYZ_CAT02 in ColorTransforms.h
    const float M[9] = {
         0.7328000f,  0.4296000f, -0.1624000f,
        -0.7036000f,  1.6975000f,  0.0061000f,
         0.0030000f,  0.0136000f,  0.9834000f
    };
    const float M_inv[9] = {
        1.0961238f, -0.2788690f,  0.1827452f,
        0.4543690f,  0.4735332f,  0.0720978f,
       -0.0096276f, -0.0056980f,  1.0153256f
    };

    float srcWhite[3] = {
        device_sanitize_nonneg(srcWhiteXYZ[0]),
        device_sanitize_nonneg(srcWhiteXYZ[1]),
        device_sanitize_nonneg(srcWhiteXYZ[2])
    };
    float dstWhite[3] = {
        device_sanitize_nonneg(dstWhiteXYZ[0]),
        device_sanitize_nonneg(dstWhiteXYZ[1]),
        device_sanitize_nonneg(dstWhiteXYZ[2])
    };

    const float srcY = (srcWhite[1] > 0.0f) ? srcWhite[1] : 1.0f;
    const float dstY = (dstWhite[1] > 0.0f) ? dstWhite[1] : 1.0f;
    const float srcScale = 1.0f / srcY;
    const float dstScale = 1.0f / dstY;
    srcWhite[0] *= srcScale; srcWhite[1] = 1.0f; srcWhite[2] *= srcScale;
    dstWhite[0] *= dstScale; dstWhite[1] = 1.0f; dstWhite[2] *= dstScale;

    float srcLMS[3];
    float dstLMS[3];
    float XYZ_LMS[3];
    mul3(M, srcWhite, srcLMS);
    mul3(M, dstWhite, dstLMS);
    mul3(M, XYZ, XYZ_LMS);

    const float scale0 = (srcLMS[0] > 1e-6f) ? (dstLMS[0] / srcLMS[0]) : 1.0f;
    const float scale1 = (srcLMS[1] > 1e-6f) ? (dstLMS[1] / srcLMS[1]) : 1.0f;
    const float scale2 = (srcLMS[2] > 1e-6f) ? (dstLMS[2] / srcLMS[2]) : 1.0f;

    float adaptedLMS[3] = {
        scale0 * XYZ_LMS[0],
        scale1 * XYZ_LMS[1],
        scale2 * XYZ_LMS[2]
    };
    mul3(M_inv, adaptedLMS, outXYZ);
}

static __device__ __forceinline__ float decode_bt2020_channel_device(float v) {
    const float x = fmaxf(0.0f, device_sanitize_channel(v));
    constexpr float a = 1.09929681f;
    constexpr float threshold = 0.0812428791f;
    if (x < threshold) {
        return x / 4.5f;
    }
    return powf((x + (a - 1.0f)) / a, 1.0f / 0.45f);
}

static __device__ __forceinline__ float decode_srgb_channel_device(float v) {
    const float x = fmaxf(0.0f, device_sanitize_channel(v));
    constexpr float threshold = 0.04045f;
    if (x <= threshold) {
        return x / 12.92f;
    }
    return powf((x + 0.055f) / 1.055f, 2.4f);
}

static __device__ __forceinline__ void apply_input_cctf_decoding_device(
    int inputColorSpaceIndex,
    int applyCctfDecoding,
    const float inRgb[3],
    float outRgb[3])
{
    if (!applyCctfDecoding) {
        outRgb[0] = device_sanitize_channel(inRgb[0]);
        outRgb[1] = device_sanitize_channel(inRgb[1]);
        outRgb[2] = device_sanitize_channel(inRgb[2]);
        return;
    }

    // InputColorSpace enum: 0=DWG, 1=BT2020, 2=ACES2065-1, 3=sRGB/Rec.709.
    if (inputColorSpaceIndex == 1) {
        outRgb[0] = decode_bt2020_channel_device(inRgb[0]);
        outRgb[1] = decode_bt2020_channel_device(inRgb[1]);
        outRgb[2] = decode_bt2020_channel_device(inRgb[2]);
        return;
    }
    if (inputColorSpaceIndex == 3) {
        outRgb[0] = decode_srgb_channel_device(inRgb[0]);
        outRgb[1] = decode_srgb_channel_device(inRgb[1]);
        outRgb[2] = decode_srgb_channel_device(inRgb[2]);
        return;
    }

    outRgb[0] = device_sanitize_channel(inRgb[0]);
    outRgb[1] = device_sanitize_channel(inRgb[1]);
    outRgb[2] = device_sanitize_channel(inRgb[2]);
}

static __device__ __forceinline__ void mat3_mul9_device(const float m[9], const float v[3], float out[3]) {
    out[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
    out[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
    out[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
}

static __device__ __forceinline__ void XYZ_to_DWG_linear_device(const float XYZ[3], float RGB[3]) {
    const float DWG_XYZ_to_RGB[9] = {
        1.51667204f, -0.28147805f, -0.14696363f,
       -0.46491710f,  1.25142378f,  0.17488461f,
        0.07578536f,  0.08076209f,  0.76034476f
    };
    float out[3];
    mat3_mul9_device(DWG_XYZ_to_RGB, XYZ, out);
    for (int i = 0; i < 3; ++i) {
        float v = out[i];
        if (!device_isfinite(v) || v < 0.0f) {
            v = 0.0f;
        }
        RGB[i] = v;
    }
}

static __device__ __forceinline__ void tri2quad_device(float tx, float ty, float& qx, float& qy) {
    // Matches Spectral::tri2quad in SpectralProcessing.h
    const float denom = fmaxf(1.0f - tx, 1e-10f);
    float x = (1.0f - tx);
    x = x * x;
    float y = ty / denom;
    qx = fminf(1.0f, fmaxf(0.0f, x));
    qy = fminf(1.0f, fmaxf(0.0f, y));
}

static __device__ __forceinline__ float hanatos_bilinear_at(
    const float* JUICER_RESTRICT lut,
    int N,
    int K,
    int x,
    int y,
    int k)
{
    const std::size_t idx = (static_cast<std::size_t>(x) * static_cast<std::size_t>(N) + static_cast<std::size_t>(y)) * static_cast<std::size_t>(K) + static_cast<std::size_t>(k);
    return ldg_f(lut + idx);
}

static __device__ __forceinline__ float hanatos_integrated_at(
    const float* JUICER_RESTRICT lut,
    int N,
    int x,
    int y,
    int c)
{
    const std::size_t idx = (static_cast<std::size_t>(x) * static_cast<std::size_t>(N) + static_cast<std::size_t>(y)) * 4u + static_cast<std::size_t>(c);
    return ldg_f(lut + idx);
}

static __device__ __forceinline__ int reflect_index_device(int idx, int size) {
    if (size <= 1) return 0;
    if (idx < 0) return -idx;
    if (idx >= size) return 2 * (size - 1) - idx;
    return idx;
}

static __device__ __forceinline__ double mitchell_weight_device(double t) {
    const double B = 1.0 / 3.0;
    const double C = 1.0 / 3.0;
    const double x = fabs(t);
    if (x < 1.0) {
        return (1.0 / 6.0) * ((12.0 - 9.0 * B - 6.0 * C) * x * x * x
            + (-18.0 + 12.0 * B + 6.0 * C) * x * x
            + (6.0 - 2.0 * B));
    }
    else if (x < 2.0) {
        return (1.0 / 6.0) * ((-B - 6.0 * C) * x * x * x
            + (6.0 * B + 30.0 * C) * x * x
            + (-12.0 * B - 48.0 * C) * x
            + (8.0 * B + 24.0 * C));
    }
    return 0.0;
}

static __device__ __forceinline__ double scan_lut_fetch_device(const double* JUICER_RESTRICT lut, int res, size_t limit, int xi, int yi, int zi, int c) {
    const size_t sRes = static_cast<size_t>(res);
    const size_t idx = (static_cast<size_t>(zi) * sRes + static_cast<size_t>(yi)) * sRes + static_cast<size_t>(xi);
    const size_t base = idx * 3u + static_cast<size_t>(c);
    if (base >= limit) {
        return 0.0;
    }
    return ldg_d(lut + base);
}

static __device__ __forceinline__ int reflect_index_repeat_device(int idx, int size) {
    if (size <= 1) {
        return 0;
    }
    while (idx < 0 || idx >= size) {
        if (idx < 0) {
            idx = -idx;
        }
        else {
            idx = 2 * size - idx - 2;
        }
    }
    return idx;
}

static __device__ __forceinline__ void sample_cubic_scan_lut_device(const double* JUICER_RESTRICT lut, int resRaw, const double D_norm[3], double out[3]) {
    if (!out) {
        return;
    }
    if (!lut || !D_norm) {
        out[0] = out[1] = out[2] = 0.0;
        return;
    }

    const int res = (resRaw > 1) ? resRaw : 1;
    const double scale = (res > 1) ? static_cast<double>(res - 1) : 1.0;
    const double fx = D_norm[0] * scale;
    const double fy = D_norm[1] * scale;
    const double fz = D_norm[2] * scale;

    const int xBase = static_cast<int>(floor(fx));
    const int yBase = static_cast<int>(floor(fy));
    const int zBase = static_cast<int>(floor(fz));
    const double tx = fx - static_cast<double>(xBase);
    const double ty = fy - static_cast<double>(yBase);
    const double tz = fz - static_cast<double>(zBase);

    double wx[4], wy[4], wz[4];
    wx[0] = mitchell_weight_device(tx + 1.0);
    wx[1] = mitchell_weight_device(tx);
    wx[2] = mitchell_weight_device(tx - 1.0);
    wx[3] = mitchell_weight_device(tx - 2.0);
    wy[0] = mitchell_weight_device(ty + 1.0);
    wy[1] = mitchell_weight_device(ty);
    wy[2] = mitchell_weight_device(ty - 1.0);
    wy[3] = mitchell_weight_device(ty - 2.0);
    wz[0] = mitchell_weight_device(tz + 1.0);
    wz[1] = mitchell_weight_device(tz);
    wz[2] = mitchell_weight_device(tz - 1.0);
    wz[3] = mitchell_weight_device(tz - 2.0);

    const size_t sRes = static_cast<size_t>(res);
    const size_t limit = sRes * sRes * sRes * 3u;

    double sum[3] = { 0.0, 0.0, 0.0 };
    double wsum = 0.0;
    for (int i = 0; i < 4; ++i) {
        const int xi = reflect_index_device(xBase - 1 + i, res);
        for (int j = 0; j < 4; ++j) {
            const int yj = reflect_index_device(yBase - 1 + j, res);
            for (int k = 0; k < 4; ++k) {
                const int zk = reflect_index_device(zBase - 1 + k, res);
                const double w = wx[i] * wy[j] * wz[k];
                wsum += w;
                sum[0] += w * scan_lut_fetch_device(lut, res, limit, xi, yj, zk, 0);
                sum[1] += w * scan_lut_fetch_device(lut, res, limit, xi, yj, zk, 1);
                sum[2] += w * scan_lut_fetch_device(lut, res, limit, xi, yj, zk, 2);
            }
        }
    }

    const double inv = (wsum != 0.0) ? (1.0 / wsum) : 0.0;
    out[0] = sum[0] * inv;
    out[1] = sum[1] * inv;
    out[2] = sum[2] * inv;
}
