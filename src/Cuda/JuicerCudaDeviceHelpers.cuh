// Cuda/JuicerCudaDeviceHelpers.cuh
// Shared CUDA device helpers for the film, scan, validation, and auto-exposure paths.
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
    float gammaFactor) {
    if (!curve.x || !curve.y || curve.n <= 0) {
        return 0.0f;
    }

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

    if (xq <= xmin) {
        return ldg_f(curve.y + domainBegin);
    }
    if (xq >= xmax) {
        return ldg_f(curve.y + domainEnd);
    }

    int left = domainBegin + 1;
    int right = domainEnd;
    int i1 = domainEnd + 1;
    while (left <= right) {
        const int mid = left + ((right - left) >> 1);
        const float xm = ldg_f(curve.x + mid);
        if (!(xm < xq)) {
            i1 = mid;
            right = mid - 1;
        } else {
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
    float gammaFactor) {
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

    const JuicerCuda::DeviceCurveView curve = {x, y, n, domainBegin, domainEnd};
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

    const JuicerCuda::DeviceCurveView curve = {x, nullptr, n, begin, end};
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
    float outXYZ[3]) {
    const float M[9] = {
        0.7328000f, 0.4296000f, -0.1624000f, -0.7036000f, 1.6975000f, 0.0061000f, 0.0030000f, 0.0136000f, 0.9834000f};
    const float M_inv[9] = {
        1.0961238f, -0.2788690f, 0.1827452f, 0.4543690f, 0.4735332f, 0.0720978f, -0.0096276f, -0.0056980f, 1.0153256f};

    float srcWhite[3] = {
        device_sanitize_nonneg(srcWhiteXYZ[0]),
        device_sanitize_nonneg(srcWhiteXYZ[1]),
        device_sanitize_nonneg(srcWhiteXYZ[2])};
    float dstWhite[3] = {
        device_sanitize_nonneg(dstWhiteXYZ[0]),
        device_sanitize_nonneg(dstWhiteXYZ[1]),
        device_sanitize_nonneg(dstWhiteXYZ[2])};

    const float srcY = (srcWhite[1] > 0.0f) ? srcWhite[1] : 1.0f;
    const float dstY = (dstWhite[1] > 0.0f) ? dstWhite[1] : 1.0f;
    const float srcScale = 1.0f / srcY;
    const float dstScale = 1.0f / dstY;
    srcWhite[0] *= srcScale;
    srcWhite[1] = 1.0f;
    srcWhite[2] *= srcScale;
    dstWhite[0] *= dstScale;
    dstWhite[1] = 1.0f;
    dstWhite[2] *= dstScale;

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
        scale2 * XYZ_LMS[2]};
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
    float outRgb[3]) {
    if (!applyCctfDecoding) {
        outRgb[0] = device_sanitize_channel(inRgb[0]);
        outRgb[1] = device_sanitize_channel(inRgb[1]);
        outRgb[2] = device_sanitize_channel(inRgb[2]);
        return;
    }

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
        1.51667204f, -0.28147805f, -0.14696363f, -0.46491710f, 1.25142378f, 0.17488461f, 0.07578536f, 0.08076209f, 0.76034476f};
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

static __device__ __forceinline__ void XYZ_to_DWG_linear_unclamped_device(const float XYZ[3], float RGB[3]) {
    const float DWG_XYZ_to_RGB[9] = {
        1.51667204f, -0.28147805f, -0.14696363f, -0.46491710f, 1.25142378f, 0.17488461f, 0.07578536f, 0.08076209f, 0.76034476f};
    float out[3];
    mat3_mul9_device(DWG_XYZ_to_RGB, XYZ, out);
    for (int i = 0; i < 3; ++i) {
        RGB[i] = device_sanitize_channel(out[i]);
    }
}

static __device__ __forceinline__ void tri2quad_device(float tx, float ty, float& qx, float& qy) {
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
    int k) {
    const std::size_t idx = (static_cast<std::size_t>(x) * static_cast<std::size_t>(N) + static_cast<std::size_t>(y)) * static_cast<std::size_t>(K) + static_cast<std::size_t>(k);
    return ldg_f(lut + idx);
}

static __device__ __forceinline__ float hanatos_integrated_at(
    const float* JUICER_RESTRICT lut,
    int N,
    int x,
    int y,
    int c) {
    const std::size_t idx = (static_cast<std::size_t>(x) * static_cast<std::size_t>(N) + static_cast<std::size_t>(y)) * 4u + static_cast<std::size_t>(c);
    return ldg_f(lut + idx);
}

static __device__ __forceinline__ int reflect_index_device(int idx, int size) {
    if (size <= 1)
        return 0;
    if (idx < 0)
        return -idx;
    if (idx >= size)
        return 2 * (size - 1) - idx;
    return idx;
}

static __device__ __forceinline__ double mitchell_weight_device(double t) {
    const double B = 1.0 / 3.0;
    const double C = 1.0 / 3.0;
    const double x = fabs(t);
    if (x < 1.0) {
        return (1.0 / 6.0) * ((12.0 - 9.0 * B - 6.0 * C) * x * x * x + (-18.0 + 12.0 * B + 6.0 * C) * x * x + (6.0 - 2.0 * B));
    } else if (x < 2.0) {
        return (1.0 / 6.0) * ((-B - 6.0 * C) * x * x * x + (6.0 * B + 30.0 * C) * x * x + (-12.0 * B - 48.0 * C) * x + (8.0 * B + 24.0 * C));
    }
    return 0.0;
}

static __device__ __forceinline__ void hanatos_cubic_coordinate_device(
    float normalized,
    int size,
    int& base,
    double& fraction) {
    double coordinate =
        static_cast<double>(fminf(1.0f, fmaxf(0.0f, normalized))) *
        static_cast<double>(size - 1);
    if (coordinate >= static_cast<double>(size - 1)) {
        base = size - 2;
        fraction = 1.0;
        return;
    }
    base = static_cast<int>(floor(coordinate));
    fraction = coordinate - static_cast<double>(base);
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
static __device__ __forceinline__ float sample_hanatos_spectrum_cubic_device(
    const float* JUICER_RESTRICT lut,
    int size,
    int samples,
    int sample,
    float tcC,
    float tcM) {
    int cBase = 0;
    int mBase = 0;
    double cFraction = 0.0;
    double mFraction = 0.0;
    hanatos_cubic_coordinate_device(tcC, size, cBase, cFraction);
    hanatos_cubic_coordinate_device(tcM, size, mBase, mFraction);

    const double wc[4] = {
        mitchell_weight_device(cFraction + 1.0),
        mitchell_weight_device(cFraction),
        mitchell_weight_device(cFraction - 1.0),
        mitchell_weight_device(cFraction - 2.0)};
    const double wm[4] = {
        mitchell_weight_device(mFraction + 1.0),
        mitchell_weight_device(mFraction),
        mitchell_weight_device(mFraction - 1.0),
        mitchell_weight_device(mFraction - 2.0)};

    double value = 0.0;
    double weightSum = 0.0;
    for (int dc = 0; dc < 4; ++dc) {
        const int c = reflect_index_device(cBase - 1 + dc, size);
        for (int dm = 0; dm < 4; ++dm) {
            const int m = reflect_index_device(mBase - 1 + dm, size);
            const double weight = wc[dc] * wm[dm];
            weightSum += weight;
            value += weight * static_cast<double>(hanatos_bilinear_at(lut, size, samples, c, m, sample));
        }
    }
    return static_cast<float>(weightSum != 0.0 ? value / weightSum : 0.0);
}

static __device__ __forceinline__ float sample_hanatos_integrated_cubic_device(
    const float* JUICER_RESTRICT lut,
    int size,
    int channel,
    float tcC,
    float tcM) {
    int cBase = 0;
    int mBase = 0;
    double cFraction = 0.0;
    double mFraction = 0.0;
    hanatos_cubic_coordinate_device(tcC, size, cBase, cFraction);
    hanatos_cubic_coordinate_device(tcM, size, mBase, mFraction);

    const double wc[4] = {
        mitchell_weight_device(cFraction + 1.0),
        mitchell_weight_device(cFraction),
        mitchell_weight_device(cFraction - 1.0),
        mitchell_weight_device(cFraction - 2.0)};
    const double wm[4] = {
        mitchell_weight_device(mFraction + 1.0),
        mitchell_weight_device(mFraction),
        mitchell_weight_device(mFraction - 1.0),
        mitchell_weight_device(mFraction - 2.0)};

    double value = 0.0;
    double weightSum = 0.0;
    for (int dc = 0; dc < 4; ++dc) {
        const int c = reflect_index_device(cBase - 1 + dc, size);
        for (int dm = 0; dm < 4; ++dm) {
            const int m = reflect_index_device(mBase - 1 + dm, size);
            const double weight = wc[dc] * wm[dm];
            weightSum += weight;
            value += weight * static_cast<double>(hanatos_integrated_at(lut, size, c, m, channel));
        }
    }
    return static_cast<float>(weightSum != 0.0 ? value / weightSum : 0.0);
}
// NOLINTEND(bugprone-easily-swappable-parameters)

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
        } else {
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

    double sum[3] = {0.0, 0.0, 0.0};
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

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
static __device__ __forceinline__ double scan_lut_fetch_cmy_device(
    const double* JUICER_RESTRICT lut,
    int res,
    int c,
    int m,
    int y,
    int output) {
    const size_t sRes = static_cast<size_t>(res);
    const size_t index =
        ((static_cast<size_t>(c) * sRes + static_cast<size_t>(m)) * sRes + static_cast<size_t>(y)) * 3u +
        static_cast<size_t>(output);
    return ldg_d(lut + index);
}

static __device__ __forceinline__ double scan_lut_fetch_cell_cmy_device(
    const double* JUICER_RESTRICT lut,
    int cellRes,
    int c,
    int m,
    int y,
    int output) {
    const size_t sRes = static_cast<size_t>(cellRes);
    const size_t index =
        ((static_cast<size_t>(c) * sRes + static_cast<size_t>(m)) * sRes + static_cast<size_t>(y)) * 3u +
        static_cast<size_t>(output);
    return ldg_d(lut + index);
}

static __device__ __forceinline__ void pchip_coordinate_device(
    double normalized,
    int res,
    int& base,
    double& fraction) {
    double coordinate = normalized * static_cast<double>(res - 1);
    coordinate = fmin(static_cast<double>(res - 1), fmax(0.0, coordinate));
    if (coordinate >= static_cast<double>(res - 1)) {
        base = res - 2;
        fraction = 1.0;
        return;
    }
    base = static_cast<int>(floor(coordinate));
    fraction = coordinate - static_cast<double>(base);
}

static __device__ __forceinline__ double hermite_value_device(
    double y0,
    double y1,
    double m0,
    double m1,
    double t) {
    const double t2 = t * t;
    const double t3 = t2 * t;
    const double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    const double h10 = t3 - 2.0 * t2 + t;
    const double h01 = -2.0 * t3 + 3.0 * t2;
    const double h11 = t3 - t2;
    return h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1;
}

static __device__ __forceinline__ double linear_mix_device(double v0, double v1, double t) {
    return v0 + t * (v1 - v0);
}

static __device__ __forceinline__ double bilinear_mix_device(
    double v00,
    double v10,
    double v01,
    double v11,
    double tc,
    double tm) {
    return linear_mix_device(
        linear_mix_device(v00, v10, tc),
        linear_mix_device(v01, v11, tc),
        tm);
}

static __device__ __forceinline__ void sample_pchip_scan_lut_device(
    const double* JUICER_RESTRICT values,
    const double* JUICER_RESTRICT slopeC,
    const double* JUICER_RESTRICT slopeM,
    const double* JUICER_RESTRICT slopeY,
    const double* JUICER_RESTRICT cellMin,
    const double* JUICER_RESTRICT cellMax,
    int res,
    const double normalizedCmy[3],
    double out[3]) {
    int c = 0;
    int m = 0;
    int y = 0;
    double tc = 0.0;
    double tm = 0.0;
    double ty = 0.0;
    pchip_coordinate_device(normalizedCmy[0], res, c, tc);
    pchip_coordinate_device(normalizedCmy[1], res, m, tm);
    pchip_coordinate_device(normalizedCmy[2], res, y, ty);

    const int cellRes = res - 1;
    for (int output = 0; output < 3; ++output) {
        const double v000 = hermite_value_device(
            scan_lut_fetch_cmy_device(values, res, c, m, y, output),
            scan_lut_fetch_cmy_device(values, res, c + 1, m, y, output),
            scan_lut_fetch_cmy_device(slopeC, res, c, m, y, output),
            scan_lut_fetch_cmy_device(slopeC, res, c + 1, m, y, output),
            tc);
        const double v010 = hermite_value_device(
            scan_lut_fetch_cmy_device(values, res, c, m + 1, y, output),
            scan_lut_fetch_cmy_device(values, res, c + 1, m + 1, y, output),
            scan_lut_fetch_cmy_device(slopeC, res, c, m + 1, y, output),
            scan_lut_fetch_cmy_device(slopeC, res, c + 1, m + 1, y, output),
            tc);
        const double v001 = hermite_value_device(
            scan_lut_fetch_cmy_device(values, res, c, m, y + 1, output),
            scan_lut_fetch_cmy_device(values, res, c + 1, m, y + 1, output),
            scan_lut_fetch_cmy_device(slopeC, res, c, m, y + 1, output),
            scan_lut_fetch_cmy_device(slopeC, res, c + 1, m, y + 1, output),
            tc);
        const double v011 = hermite_value_device(
            scan_lut_fetch_cmy_device(values, res, c, m + 1, y + 1, output),
            scan_lut_fetch_cmy_device(values, res, c + 1, m + 1, y + 1, output),
            scan_lut_fetch_cmy_device(slopeC, res, c, m + 1, y + 1, output),
            scan_lut_fetch_cmy_device(slopeC, res, c + 1, m + 1, y + 1, output),
            tc);

        const double sm00 = linear_mix_device(
            scan_lut_fetch_cmy_device(slopeM, res, c, m, y, output),
            scan_lut_fetch_cmy_device(slopeM, res, c + 1, m, y, output),
            tc);
        const double sm10 = linear_mix_device(
            scan_lut_fetch_cmy_device(slopeM, res, c, m + 1, y, output),
            scan_lut_fetch_cmy_device(slopeM, res, c + 1, m + 1, y, output),
            tc);
        const double sm01 = linear_mix_device(
            scan_lut_fetch_cmy_device(slopeM, res, c, m, y + 1, output),
            scan_lut_fetch_cmy_device(slopeM, res, c + 1, m, y + 1, output),
            tc);
        const double sm11 = linear_mix_device(
            scan_lut_fetch_cmy_device(slopeM, res, c, m + 1, y + 1, output),
            scan_lut_fetch_cmy_device(slopeM, res, c + 1, m + 1, y + 1, output),
            tc);
        const double vy0 = hermite_value_device(v000, v010, sm00, sm10, tm);
        const double vy1 = hermite_value_device(v001, v011, sm01, sm11, tm);

        const double sy0 = bilinear_mix_device(
            scan_lut_fetch_cmy_device(slopeY, res, c, m, y, output),
            scan_lut_fetch_cmy_device(slopeY, res, c + 1, m, y, output),
            scan_lut_fetch_cmy_device(slopeY, res, c, m + 1, y, output),
            scan_lut_fetch_cmy_device(slopeY, res, c + 1, m + 1, y, output),
            tc,
            tm);
        const double sy1 = bilinear_mix_device(
            scan_lut_fetch_cmy_device(slopeY, res, c, m, y + 1, output),
            scan_lut_fetch_cmy_device(slopeY, res, c + 1, m, y + 1, output),
            scan_lut_fetch_cmy_device(slopeY, res, c, m + 1, y + 1, output),
            scan_lut_fetch_cmy_device(slopeY, res, c + 1, m + 1, y + 1, output),
            tc,
            tm);

        const double interpolated = hermite_value_device(vy0, vy1, sy0, sy1, ty);
        const double minimum = scan_lut_fetch_cell_cmy_device(cellMin, cellRes, c, m, y, output);
        const double maximum = scan_lut_fetch_cell_cmy_device(cellMax, cellRes, c, m, y, output);
        out[output] = fmin(maximum, fmax(minimum, interpolated));
    }
}
// NOLINTEND(bugprone-easily-swappable-parameters)

static __device__ __forceinline__ void convert_input_to_DWG_device(
    const JuicerCuda::FilmRawPayload& cfg,
    const float rgbIn[3],
    float rgbDWG[3],
    bool clampNonNegative) {
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
    } else {
        XYZ_to_DWG_linear_unclamped_device(xyzPtr, dwg);
    }
    rgbDWG[0] = dwg[0];
    rgbDWG[1] = dwg[1];
    rgbDWG[2] = dwg[2];
}

static __device__ __forceinline__ void convert_input_to_sRGB_device(
    const JuicerCuda::FilmRawPayload& cfg,
    const float rgbIn[3],
    float rgbSRGB[3]) {
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

    const float XYZ_to_sRGB[9] = {
        3.2404542f, -1.5371385f, -0.4985314f, -0.9692660f, 1.8760108f, 0.0415560f, 0.0556434f, -0.2040259f, 1.0572252f};

    float srgb[3];
    mat3_mul9_device(XYZ_to_sRGB, xyzPtr, srgb);

    rgbSRGB[0] = device_isfinite(srgb[0]) ? srgb[0] : 0.0f;
    rgbSRGB[1] = device_isfinite(srgb[1]) ? srgb[1] : 0.0f;
    rgbSRGB[2] = device_isfinite(srgb[2]) ? srgb[2] : 0.0f;
}

static __device__ __forceinline__ void convert_input_to_working_xyz_device(
    const JuicerCuda::FilmRawPayload& cfg,
    const float rgbIn[3],
    float workingXYZ[3]) {
    float linear[3];
    apply_input_cctf_decoding_device(cfg.inputColorSpaceIndex, cfg.applyCctfDecoding, rgbIn, linear);

    float inputXYZ[3];
    mat3_mul9_device(cfg.inputRGBToXYZ, linear, inputXYZ);

    if (cfg.applyInputChromaticAdapt) {
        mat3_mul9_device(cfg.inputXYZAdapt, inputXYZ, workingXYZ);
    } else {
        workingXYZ[0] = inputXYZ[0];
        workingXYZ[1] = inputXYZ[1];
        workingXYZ[2] = inputXYZ[2];
    }

    workingXYZ[0] = device_sanitize_channel(workingXYZ[0]);
    workingXYZ[1] = device_sanitize_channel(workingXYZ[1]);
    workingXYZ[2] = device_sanitize_channel(workingXYZ[2]);
}

static __device__ void hanatos_layer_exposures_device(
    const float workingXYZ[3],
    const float* JUICER_RESTRICT hanatosLut,
    int hanatosN,
    const float refIllumWhiteXYZ[3],
    const float* JUICER_RESTRICT sensB,
    const float* JUICER_RESTRICT sensG,
    const float* JUICER_RESTRICT sensR,
    float E_out[3]) {
    if (!E_out) {
        return;
    }
    if (!hanatosLut || hanatosN <= 0 || !sensB || !sensG || !sensR) {
        E_out[0] = E_out[1] = E_out[2] = 0.0f;
        return;
    }

    float XYZ[3] = {
        device_sanitize_channel(workingXYZ[0]),
        device_sanitize_channel(workingXYZ[1]),
        device_sanitize_channel(workingXYZ[2])};

    const float D65[3] = {0.950455f, 1.0f, 1.089058f};

    float refWhite[3] = {
        device_sanitize_channel(refIllumWhiteXYZ[0]),
        device_sanitize_channel(refIllumWhiteXYZ[1]),
        device_sanitize_channel(refIllumWhiteXYZ[2])};
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
    double Eb = 0.0;
    double Eg = 0.0;
    double Er = 0.0;
    for (int k = 0; k < K; ++k) {
        const float raw = sample_hanatos_spectrum_cubic_device(hanatosLut, N, K, k, qx, qy);

        const float e = device_sanitize_channel(sumXYZ * raw);
        if (!device_isfinite(e)) {
            continue;
        }
        const double e64 = static_cast<double>(e);

        const float sb = ldg_f(sensB + k);
        const float sg = ldg_f(sensG + k);
        const float sr = ldg_f(sensR + k);
        if (isfinite(sb))
            Eb += e64 * static_cast<double>(sb);
        if (isfinite(sg))
            Eg += e64 * static_cast<double>(sg);
        if (isfinite(sr))
            Er += e64 * static_cast<double>(sr);
    }

    E_out[0] = device_isfinite(static_cast<float>(Eb)) ? static_cast<float>(Eb) : 0.0f;
    E_out[1] = device_isfinite(static_cast<float>(Eg)) ? static_cast<float>(Eg) : 0.0f;
    E_out[2] = device_isfinite(static_cast<float>(Er)) ? static_cast<float>(Er) : 0.0f;
}

static __device__ void hanatos_integrated_exposures_device(
    const float workingXYZ[3],
    const float* JUICER_RESTRICT lutIntegrated,
    int hanatosN,
    const float refIllumWhiteXYZ[3],
    float E_out[3]) {
    if (!E_out) {
        return;
    }
    if (!lutIntegrated || hanatosN <= 0) {
        E_out[0] = E_out[1] = E_out[2] = 0.0f;
        return;
    }

    float XYZ[3] = {
        device_sanitize_channel(workingXYZ[0]),
        device_sanitize_channel(workingXYZ[1]),
        device_sanitize_channel(workingXYZ[2])};

    const float D65[3] = {0.950455f, 1.0f, 1.089058f};

    float refWhite[3] = {
        device_sanitize_channel(refIllumWhiteXYZ[0]),
        device_sanitize_channel(refIllumWhiteXYZ[1]),
        device_sanitize_channel(refIllumWhiteXYZ[2])};
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
    const float r = sample_hanatos_integrated_cubic_device(lutIntegrated, N, 0, qx, qy);
    const float g = sample_hanatos_integrated_cubic_device(lutIntegrated, N, 1, qx, qy);
    const float b = sample_hanatos_integrated_cubic_device(lutIntegrated, N, 2, qx, qy);

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
    float E_out[3]) {
    if (!E_out) {
        return;
    }
    if (!rgbDWG || !S_inv || !refIllumWhiteXYZ || !Ax || !Ay || !Az || !sensB || !sensG || !sensR) {
        E_out[0] = E_out[1] = E_out[2] = 0.0f;
        return;
    }

    const float DWG_RGB_to_XYZ[9] = {
        0.70062239f, 0.14877482f, 0.10105872f, 0.27411851f, 0.87363190f, -0.14775041f, -0.09896291f, -0.13789533f, 1.32591599f};

    float XYZ[3] = {
        DWG_RGB_to_XYZ[0] * rgbDWG[0] + DWG_RGB_to_XYZ[1] * rgbDWG[1] + DWG_RGB_to_XYZ[2] * rgbDWG[2],
        DWG_RGB_to_XYZ[3] * rgbDWG[0] + DWG_RGB_to_XYZ[4] * rgbDWG[1] + DWG_RGB_to_XYZ[5] * rgbDWG[2],
        DWG_RGB_to_XYZ[6] * rgbDWG[0] + DWG_RGB_to_XYZ[7] * rgbDWG[1] + DWG_RGB_to_XYZ[8] * rgbDWG[2]};

    float sanitizedXYZ[3] = {
        device_sanitize_nonneg(XYZ[0]),
        device_sanitize_nonneg(XYZ[1]),
        device_sanitize_nonneg(XYZ[2])};

    const float D65[3] = {0.950455f, 1.0f, 1.089058f};

    float refWhite[3] = {
        device_sanitize_nonneg(refIllumWhiteXYZ[0]),
        device_sanitize_nonneg(refIllumWhiteXYZ[1]),
        device_sanitize_nonneg(refIllumWhiteXYZ[2])};
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
        if (isfinite(sb))
            Eb += e64 * static_cast<double>(sb);
        if (isfinite(sg))
            Eg += e64 * static_cast<double>(sg);
        if (isfinite(sr))
            Er += e64 * static_cast<double>(sr);
    }

    if (Y_recon > 1e-20 && targetScale > 0.0f) {
        const double s = static_cast<double>(targetScale) / Y_recon;
        Eb *= s;
        Eg *= s;
        Er *= s;
    } else if (!(targetScale > 0.0f)) {
        Eb = Eg = Er = 0.0;
    }

    // Table-based SPD path applies Δλ=5nm; exposure scale is applied later in the pipeline.
    const double dl = 5.0;
    E_out[0] = fmaxf(0.0f, static_cast<float>(Eb * dl));
    E_out[1] = fmaxf(0.0f, static_cast<float>(Eg * dl));
    E_out[2] = fmaxf(0.0f, static_cast<float>(Er * dl));
}

static __device__ void mallett_layer_exposures_device(
    const float rgbSRGB[3],
    const float* JUICER_RESTRICT mallettBasis,
    const float* JUICER_RESTRICT illum,
    int K,
    const float* JUICER_RESTRICT sensB,
    const float* JUICER_RESTRICT sensG,
    const float* JUICER_RESTRICT sensR,
    float E_out[3]) {
    if (!E_out) {
        return;
    }
    if (!rgbSRGB || !mallettBasis || !illum || !sensB || !sensG || !sensR || K <= 0) {
        E_out[0] = E_out[1] = E_out[2] = 0.0f;
        return;
    }

    const float r = fmaxf(0.0f, device_sanitize_channel(rgbSRGB[0]));
    const float g = fmaxf(0.0f, device_sanitize_channel(rgbSRGB[1]));
    const float b = fmaxf(0.0f, device_sanitize_channel(rgbSRGB[2]));

    double Eb = 0.0;
    double Eg = 0.0;
    double Er = 0.0;

    for (int i = 0; i < K; ++i) {
        const float illum_i = ldg_f(illum + i);
        const float b0 = ldg_f(mallettBasis + i * 3 + 0);
        const float b1 = ldg_f(mallettBasis + i * 3 + 1);
        const float b2 = ldg_f(mallettBasis + i * 3 + 2);
        const float spd = (r * b0 + g * b1 + b * b2) * illum_i;
        if (!device_isfinite(spd)) {
            continue;
        }
        const double e64 = static_cast<double>(spd);
        const float sb = ldg_f(sensB + i);
        const float sg = ldg_f(sensG + i);
        const float sr = ldg_f(sensR + i);
        if (isfinite(sb))
            Eb += e64 * static_cast<double>(sb);
        if (isfinite(sg))
            Eg += e64 * static_cast<double>(sg);
        if (isfinite(sr))
            Er += e64 * static_cast<double>(sr);
    }

    E_out[0] = device_isfinite(static_cast<float>(Eb)) ? static_cast<float>(Eb) : 0.0f;
    E_out[1] = device_isfinite(static_cast<float>(Eg)) ? static_cast<float>(Eg) : 0.0f;
    E_out[2] = device_isfinite(static_cast<float>(Er)) ? static_cast<float>(Er) : 0.0f;
}

template <typename Params>
static __device__ __forceinline__ void compute_film_raw_device(
    const Params& params,
    const float rgbIn[3],
    float filmRaw[3]) {
    const JuicerCuda::FilmExposurePayload& expose = params.filmExpose;

    float E_raw[3] = {0.0f, 0.0f, 0.0f};
    const bool allowHanatos = (params.filmRaw.spectralUpsamplingMode == 0);
    const bool allowMallett = (params.filmRaw.spectralUpsamplingMode != 0);
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
    const bool useMallett = allowMallett &&
                            spdReady &&
                            expose.tablesIllum &&
                            expose.mallettBasis &&
                            (expose.mallettBasisK == 81) &&
                            canTables;

    float autoExposureScale = 1.0f;
    if (expose.exposureScaleDevice) {
        const float deviceScale = *expose.exposureScaleDevice;
        if (isfinite(deviceScale) && deviceScale > 0.0f) {
            autoExposureScale = deviceScale;
        }
    }
    const float autoExposedRgb[3] = {
        rgbIn[0] * autoExposureScale,
        rgbIn[1] * autoExposureScale,
        rgbIn[2] * autoExposureScale};

    float rgbDWG[3] = {0.0f, 0.0f, 0.0f};
    float workingXYZ[3] = {0.0f, 0.0f, 0.0f};
    if (useHanatos) {
        convert_input_to_working_xyz_device(params.filmRaw, autoExposedRgb, workingXYZ);
    } else if (!useMallett) {
        convert_input_to_DWG_device(params.filmRaw, autoExposedRgb, rgbDWG, !useHanatos);
    }

    if (useHanatosIntegrated) {
        hanatos_integrated_exposures_device(
            workingXYZ,
            expose.hanatosLutIntegrated,
            expose.hanatosNIntegrated,
            params.filmRaw.refIllumWhiteXYZ,
            E_raw);
    } else if (useHanatos) {
        hanatos_layer_exposures_device(
            workingXYZ,
            expose.hanatosLut,
            expose.hanatosN,
            params.filmRaw.refIllumWhiteXYZ,
            expose.sensB.y,
            expose.sensG.y,
            expose.sensR.y,
            E_raw);
    } else if (useMallett) {
        float rgbSRGB[3];
        convert_input_to_sRGB_device(params.filmRaw, autoExposedRgb, rgbSRGB);
        mallett_layer_exposures_device(
            rgbSRGB,
            expose.mallettBasis,
            expose.tablesIllum,
            expose.tablesK,
            expose.sensB.y,
            expose.sensG.y,
            expose.sensR.y,
            E_raw);
    } else if (canTables) {
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

    float mallettGreenMidgrayScale = params.filmRaw.mallettGreenMidgrayScale;
    if (!isfinite(mallettGreenMidgrayScale) || !(mallettGreenMidgrayScale > 0.0f)) {
        mallettGreenMidgrayScale = 1.0f;
    }

    float manualExposureScale = expose.manualExposureScale;
    if (!isfinite(manualExposureScale) || !(manualExposureScale > 0.0f)) {
        manualExposureScale = 1.0f;
    }
    float routeCorrectionScale = expose.routeCorrectionScale;
    if (!isfinite(routeCorrectionScale) || !(routeCorrectionScale > 0.0f)) {
        routeCorrectionScale = 1.0f;
    }
    for (int i = 0; i < 3; ++i) {
        float v = E_raw[i];
        if (!isfinite(v) || v < 0.0f)
            v = 0.0f;
        if (useMallett) {
            v = fmaxf(0.0f, v * mallettGreenMidgrayScale);
        }
        v = fmaxf(0.0f, v * manualExposureScale * routeCorrectionScale);
        filmRaw[i] = v;
    }
}

template <typename Params>
static __device__ __forceinline__ void compute_logE_raw_from_film_raw_device(
    const Params& params,
    const float filmRaw[3],
    float logE_raw[3]) {
    (void)params;
    constexpr float kLogEps = 1e-10f;
    logE_raw[0] = log10f(fmaxf(filmRaw[0], 0.0f) + kLogEps);
    logE_raw[1] = log10f(fmaxf(filmRaw[1], 0.0f) + kLogEps);
    logE_raw[2] = log10f(fmaxf(filmRaw[2], 0.0f) + kLogEps);
}

template <typename Params>
static __device__ __forceinline__ void compute_logE_raw_device(
    const Params& params,
    const float rgbIn[3],
    float logE_raw[3]) {
    float filmRaw[3] = {0.0f, 0.0f, 0.0f};
    compute_film_raw_device(params, rgbIn, filmRaw);
    compute_logE_raw_from_film_raw_device(params, filmRaw, logE_raw);
}

template <typename Params>
static __device__ __forceinline__ void compute_logE_from_film_raw_device(
    const Params& params,
    const float filmRaw[3],
    float logE_raw[3],
    float logE_sanitized[3],
    float layerPre[3]) {
    const JuicerCuda::FilmDevelopPayload& develop = params.filmDevelop;

    compute_logE_raw_from_film_raw_device(params, filmRaw, logE_raw);

    logE_sanitized[0] = sanitize_inf_logE_for_curve_device(logE_raw[0], develop.densB);
    logE_sanitized[1] = sanitize_inf_logE_for_curve_device(logE_raw[1], develop.densG);
    logE_sanitized[2] = sanitize_inf_logE_for_curve_device(logE_raw[2], develop.densR);

    layerPre[0] = sample_density_at_logE_device(develop.densB, logE_sanitized[0], develop.gammaFactorB);
    layerPre[1] = sample_density_at_logE_device(develop.densG, logE_sanitized[1], develop.gammaFactorG);
    layerPre[2] = sample_density_at_logE_device(develop.densR, logE_sanitized[2], develop.gammaFactorR);
}

template <typename Params>
static __device__ __forceinline__ void compute_logE_and_layer_pre_device(
    const Params& params,
    const float rgbIn[3],
    float logE_raw[3],
    float logE_sanitized[3],
    float layerPre[3]) {
    float filmRaw[3] = {0.0f, 0.0f, 0.0f};
    compute_film_raw_device(params, rgbIn, filmRaw);
    compute_logE_from_film_raw_device(params, filmRaw, logE_raw, logE_sanitized, layerPre);
}

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

        if (activeC)
            accumC += e64 * static_cast<double>(sC);
        if (activeM)
            accumM += e64 * static_cast<double>(sM);
        if (activeY)
            accumY += e64 * static_cast<double>(sY);
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

    rawC *= kMid;
    rawM *= kMid;
    rawY *= kMid;

    const float preflash = expose.printPreflashExposure;
    if (isfinite(preflash) && preflash > 0.0f) {
        rawC += expose.printPreflashRaw[0] * preflash;
        rawM += expose.printPreflashRaw[1] * preflash;
        rawY += expose.printPreflashRaw[2] * preflash;
    }

    rawC *= expPrint;
    rawM *= expPrint;
    rawY *= expPrint;

    float routeCorrectionScale = expose.routeCorrectionScale;
    if (!isfinite(routeCorrectionScale) || !(routeCorrectionScale > 0.0f)) {
        routeCorrectionScale = 1.0f;
    }
    rawC *= routeCorrectionScale;
    rawM *= routeCorrectionScale;
    rawY *= routeCorrectionScale;

    // RAW -> log10(raw + eps) -> print density curves.
    constexpr float kLogEps = 1e-10f;
    const float logC = log10f(rawC + kLogEps);
    const float logM = log10f(rawM + kLogEps);
    const float logY = log10f(rawY + kLogEps);

    D_cmy[0] = sample_density_at_logE_device(develop.printDcC, logC, develop.printGammaC);
    D_cmy[1] = sample_density_at_logE_device(develop.printDcM, logM, develop.printGammaM);
    D_cmy[2] = sample_density_at_logE_device(develop.printDcY, logY, develop.printGammaY);
}

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
    const JuicerCuda::DeviceCurveView& densR) {
    if (!logE_BGR || !layerD_BGR) {
        return;
    }
    if (!dir.active) {
        return;
    }

    auto silver_density = [&](float density, float dmax) -> float {
        const float finiteDensity = isfinite(density) ? density : 0.0f;
        const float silver = dir.positive ? dmax - finiteDensity : finiteDensity;
        return isfinite(silver) ? fmaxf(0.0f, silver) : 0.0f;
    };

    const float nB = silver_density(layerD_BGR[0], dir.dMax[0]);
    const float nG = silver_density(layerD_BGR[1], dir.dMax[1]);
    const float nR = silver_density(layerD_BGR[2], dir.dMax[2]);

    float aY = dir.M[0] * nB + dir.M[3] * nG + dir.M[6] * nR;
    float aM = dir.M[1] * nB + dir.M[4] * nG + dir.M[7] * nR;
    float aC = dir.M[2] * nB + dir.M[5] * nG + dir.M[8] * nR;

    if (!isfinite(aY))
        aY = 0.0f;
    if (!isfinite(aM))
        aM = 0.0f;
    if (!isfinite(aC))
        aC = 0.0f;

    auto clamp_corr = [](float v) -> float {
        if (!isfinite(v))
            return 0.0f;
        if (v < -10.0f)
            return -10.0f;
        if (v > 10.0f)
            return 10.0f;
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
