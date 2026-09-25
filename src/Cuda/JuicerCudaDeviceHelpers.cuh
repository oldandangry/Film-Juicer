// Cuda/JuicerCudaDeviceHelpers.cuh
// Shared CUDA device helpers for the film, scan, validation, and auto-exposure paths.
#pragma once

#include <cuda_runtime.h>

#include <cfloat>
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

static __device__ __forceinline__ float sample_density_at_logE_device(
    const JuicerCuda::DeviceCurveView& curve,
    float logE,
    float gammaFactor) {
    if (!device_isfinite(logE)) {
        return nanf("");
    }

    const float xq = logE * gammaFactor;
    const int domainBegin = curve.domainBegin;
    const int domainEnd = curve.domainEnd;

    const float xmin = ldg_f(curve.x + domainBegin);
    const float xmax = ldg_f(curve.x + domainEnd);
    if (!(xmax >= xmin)) {
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
        if (xq < xm) {
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

static __device__ __forceinline__ float sanitize_inf_logE_for_curve_device(float logE, const JuicerCuda::DeviceCurveView& curve) {
    if (device_isfinite(logE) || isnan(logE)) {
        return logE;
    }
    const int begin = curve.domainBegin;
    const int end = curve.domainEnd;

    const float xmin = ldg_f(curve.x + begin);
    const float xmax = ldg_f(curve.x + end);
    if (!(xmax >= xmin)) {
        return logE;
    }
    return (logE > 0.0f) ? xmax : xmin;
}

struct Mat3 {
    float m[9];
};

struct ChromaticAdaptationDeviceInput {
    const float* xyz = nullptr;
    const float* sourceWhiteXYZ = nullptr;
    const float* destinationWhiteXYZ = nullptr;
};

struct InputCctfDecodingDevice {
    int inputColorSpaceIndex = 0;
    int applyCctfDecoding = 0;
};

struct FilmDevelopIntermediatesDevice {
    float* logERaw = nullptr;
    float* logESanitized = nullptr;
    float* layerPre = nullptr;
};

static __device__ __forceinline__ void mul3(const float m[9], const float v[3], float dst[3]) {
    dst[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
    dst[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
    dst[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
}

static __device__ __forceinline__ void chromatic_adapt_XYZ_CAT02_device(
    ChromaticAdaptationDeviceInput input,
    float outXYZ[3]) {
    const float M[9] = {
        0.7328000f, 0.4296000f, -0.1624000f, -0.7036000f, 1.6975000f, 0.0061000f, 0.0030000f, 0.0136000f, 0.9834000f};
    const float M_inv[9] = {
        1.0961238f, -0.2788690f, 0.1827452f, 0.4543690f, 0.4735332f, 0.0720978f, -0.0096276f, -0.0056980f, 1.0153256f};

    float srcWhite[3] = {
        device_sanitize_nonneg(input.sourceWhiteXYZ[0]),
        device_sanitize_nonneg(input.sourceWhiteXYZ[1]),
        device_sanitize_nonneg(input.sourceWhiteXYZ[2])};
    float dstWhite[3] = {
        device_sanitize_nonneg(input.destinationWhiteXYZ[0]),
        device_sanitize_nonneg(input.destinationWhiteXYZ[1]),
        device_sanitize_nonneg(input.destinationWhiteXYZ[2])};

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
    mul3(M, input.xyz, XYZ_LMS);

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
    InputCctfDecodingDevice config,
    const float inRgb[3],
    float outRgb[3]) {
    if (!config.applyCctfDecoding) {
        outRgb[0] = device_sanitize_channel(inRgb[0]);
        outRgb[1] = device_sanitize_channel(inRgb[1]);
        outRgb[2] = device_sanitize_channel(inRgb[2]);
        return;
    }

    if (config.inputColorSpaceIndex == 1) {
        outRgb[0] = decode_bt2020_channel_device(inRgb[0]);
        outRgb[1] = decode_bt2020_channel_device(inRgb[1]);
        outRgb[2] = decode_bt2020_channel_device(inRgb[2]);
        return;
    }
    if (config.inputColorSpaceIndex == 3) {
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

static __device__ __forceinline__ void tri2quad_device(float tx, float ty, float& qx, float& qy) {
    const float denom = fmaxf(1.0f - tx, 1e-10f);
    float x = (1.0f - tx);
    x = x * x;
    float y = ty / denom;
    qx = fminf(1.0f, fmaxf(0.0f, x));
    qy = fminf(1.0f, fmaxf(0.0f, y));
}

static __device__ __forceinline__ float film_tc_lut_at(
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

static __device__ __forceinline__ float mitchell_weight_device(float t) {
    constexpr float B = 1.0f / 3.0f;
    constexpr float C = 1.0f / 3.0f;
    const float x = fabsf(t);
    if (x < 1.0f) {
        return (1.0f / 6.0f) * ((12.0f - 9.0f * B - 6.0f * C) * x * x * x + (-18.0f + 12.0f * B + 6.0f * C) * x * x + (6.0f - 2.0f * B));
    } else if (x < 2.0f) {
        return (1.0f / 6.0f) * ((-B - 6.0f * C) * x * x * x + (6.0f * B + 30.0f * C) * x * x + (-12.0f * B - 48.0f * C) * x + (8.0f * B + 24.0f * C));
    }
    return 0.0f;
}

static __device__ __forceinline__ void film_tc_cubic_coordinate_device(
    float normalized,
    int size,
    int& base,
    float& fraction) {
    float coordinate =
        fminf(1.0f, fmaxf(0.0f, normalized)) * static_cast<float>(size - 1);
    if (coordinate >= static_cast<float>(size - 1)) {
        base = size - 2;
        fraction = 1.0f;
        return;
    }
    base = static_cast<int>(floorf(coordinate));
    fraction = coordinate - static_cast<float>(base);
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
static __device__ __forceinline__ float sample_film_tc_lut_cubic_device(
    const float* JUICER_RESTRICT lut,
    int size,
    int channel,
    float tcC,
    float tcM) {
    int cBase = 0;
    int mBase = 0;
    float cFraction = 0.0f;
    float mFraction = 0.0f;
    film_tc_cubic_coordinate_device(tcC, size, cBase, cFraction);
    film_tc_cubic_coordinate_device(tcM, size, mBase, mFraction);

    const float wc[4] = {
        mitchell_weight_device(cFraction + 1.0f),
        mitchell_weight_device(cFraction),
        mitchell_weight_device(cFraction - 1.0f),
        mitchell_weight_device(cFraction - 2.0f)};
    const float wm[4] = {
        mitchell_weight_device(mFraction + 1.0f),
        mitchell_weight_device(mFraction),
        mitchell_weight_device(mFraction - 1.0f),
        mitchell_weight_device(mFraction - 2.0f)};

    float value = 0.0f;
    float weightSum = 0.0f;
    for (int dc = 0; dc < 4; ++dc) {
        const int c = reflect_index_device(cBase - 1 + dc, size);
        for (int dm = 0; dm < 4; ++dm) {
            const int m = reflect_index_device(mBase - 1 + dm, size);
            const float weight = wc[dc] * wm[dm];
            weightSum += weight;
            value += weight * film_tc_lut_at(lut, size, c, m, channel);
        }
    }
    return weightSum != 0.0f ? value / weightSum : 0.0f;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
static __device__ __forceinline__ float scan_lut_fetch_float_cmy_device(
    const float* JUICER_RESTRICT lut,
    int res,
    int c,
    int m,
    int y,
    int output) {
    const size_t sRes = static_cast<size_t>(res);
    const size_t index =
        ((static_cast<size_t>(c) * sRes + static_cast<size_t>(m)) * sRes + static_cast<size_t>(y)) * 3u +
        static_cast<size_t>(output);
    return ldg_f(lut + index);
}

static __device__ __forceinline__ float scan_lut_fetch_float_cell_cmy_device(
    const float* JUICER_RESTRICT lut,
    int cellRes,
    int c,
    int m,
    int y,
    int output) {
    const size_t sRes = static_cast<size_t>(cellRes);
    const size_t index =
        ((static_cast<size_t>(c) * sRes + static_cast<size_t>(m)) * sRes + static_cast<size_t>(y)) * 3u +
        static_cast<size_t>(output);
    return ldg_f(lut + index);
}

static __device__ __forceinline__ void pchip_coordinate_float_device(
    float normalized,
    int res,
    int& base,
    float& fraction) {
    float coordinate = normalized * static_cast<float>(res - 1);
    coordinate = fminf(static_cast<float>(res - 1), fmaxf(0.0f, coordinate));
    if (coordinate >= static_cast<float>(res - 1)) {
        base = res - 2;
        fraction = 1.0f;
        return;
    }
    base = static_cast<int>(floorf(coordinate));
    fraction = coordinate - static_cast<float>(base);
}

static __device__ __forceinline__ float hermite_value_float_device(
    float y0,
    float y1,
    float m0,
    float m1,
    float t) {
    const float t2 = t * t;
    const float t3 = t2 * t;
    const float h00 = 2.0f * t3 - 3.0f * t2 + 1.0f;
    const float h10 = t3 - 2.0f * t2 + t;
    const float h01 = -2.0f * t3 + 3.0f * t2;
    const float h11 = t3 - t2;
    return h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1;
}

static __device__ __forceinline__ float linear_mix_float_device(float v0, float v1, float t) {
    return v0 + t * (v1 - v0);
}

static __device__ __forceinline__ float bilinear_mix_float_device(
    float v00,
    float v10,
    float v01,
    float v11,
    float tc,
    float tm) {
    return linear_mix_float_device(
        linear_mix_float_device(v00, v10, tc),
        linear_mix_float_device(v01, v11, tc),
        tm);
}

static __device__ __forceinline__ bool sample_pchip_float_log2_scan_lut_device(
    const JuicerCuda::ScanStagePayload& scanStage,
    const double normalizedCmy[3],
    float out[3]) {
    if (!scanStage.scanLutLog2PchipXYZ ||
        !scanStage.scanLutPchipSlopeC ||
        !scanStage.scanLutPchipSlopeM ||
        !scanStage.scanLutPchipSlopeY ||
        !scanStage.scanLutPchipCellMin ||
        !scanStage.scanLutPchipCellMax ||
        scanStage.scanLutRes < 2 ||
        !normalizedCmy ||
        !out ||
        !isfinite(normalizedCmy[0]) ||
        !isfinite(normalizedCmy[1]) ||
        !isfinite(normalizedCmy[2])) {
        return false;
    }

    int c = 0;
    int m = 0;
    int y = 0;
    float tc = 0.0f;
    float tm = 0.0f;
    float ty = 0.0f;
    pchip_coordinate_float_device(static_cast<float>(normalizedCmy[0]), scanStage.scanLutRes, c, tc);
    pchip_coordinate_float_device(static_cast<float>(normalizedCmy[1]), scanStage.scanLutRes, m, tm);
    pchip_coordinate_float_device(static_cast<float>(normalizedCmy[2]), scanStage.scanLutRes, y, ty);

    const int res = scanStage.scanLutRes;
    const int cellRes = res - 1;
    for (int output = 0; output < 3; ++output) {
        const float v000 = hermite_value_float_device(
            scan_lut_fetch_float_cmy_device(scanStage.scanLutLog2PchipXYZ, res, c, m, y, output),
            scan_lut_fetch_float_cmy_device(scanStage.scanLutLog2PchipXYZ, res, c + 1, m, y, output),
            scan_lut_fetch_float_cmy_device(scanStage.scanLutPchipSlopeC, res, c, m, y, output),
            scan_lut_fetch_float_cmy_device(scanStage.scanLutPchipSlopeC, res, c + 1, m, y, output),
            tc);
        const float v010 = hermite_value_float_device(
            scan_lut_fetch_float_cmy_device(scanStage.scanLutLog2PchipXYZ, res, c, m + 1, y, output),
            scan_lut_fetch_float_cmy_device(scanStage.scanLutLog2PchipXYZ, res, c + 1, m + 1, y, output),
            scan_lut_fetch_float_cmy_device(scanStage.scanLutPchipSlopeC, res, c, m + 1, y, output),
            scan_lut_fetch_float_cmy_device(scanStage.scanLutPchipSlopeC, res, c + 1, m + 1, y, output),
            tc);
        const float v001 = hermite_value_float_device(
            scan_lut_fetch_float_cmy_device(scanStage.scanLutLog2PchipXYZ, res, c, m, y + 1, output),
            scan_lut_fetch_float_cmy_device(scanStage.scanLutLog2PchipXYZ, res, c + 1, m, y + 1, output),
            scan_lut_fetch_float_cmy_device(scanStage.scanLutPchipSlopeC, res, c, m, y + 1, output),
            scan_lut_fetch_float_cmy_device(scanStage.scanLutPchipSlopeC, res, c + 1, m, y + 1, output),
            tc);
        const float v011 = hermite_value_float_device(
            scan_lut_fetch_float_cmy_device(scanStage.scanLutLog2PchipXYZ, res, c, m + 1, y + 1, output),
            scan_lut_fetch_float_cmy_device(scanStage.scanLutLog2PchipXYZ, res, c + 1, m + 1, y + 1, output),
            scan_lut_fetch_float_cmy_device(scanStage.scanLutPchipSlopeC, res, c, m + 1, y + 1, output),
            scan_lut_fetch_float_cmy_device(scanStage.scanLutPchipSlopeC, res, c + 1, m + 1, y + 1, output),
            tc);

        const float sm00 = linear_mix_float_device(
            scan_lut_fetch_float_cmy_device(scanStage.scanLutPchipSlopeM, res, c, m, y, output),
            scan_lut_fetch_float_cmy_device(scanStage.scanLutPchipSlopeM, res, c + 1, m, y, output),
            tc);
        const float sm10 = linear_mix_float_device(
            scan_lut_fetch_float_cmy_device(scanStage.scanLutPchipSlopeM, res, c, m + 1, y, output),
            scan_lut_fetch_float_cmy_device(scanStage.scanLutPchipSlopeM, res, c + 1, m + 1, y, output),
            tc);
        const float sm01 = linear_mix_float_device(
            scan_lut_fetch_float_cmy_device(scanStage.scanLutPchipSlopeM, res, c, m, y + 1, output),
            scan_lut_fetch_float_cmy_device(scanStage.scanLutPchipSlopeM, res, c + 1, m, y + 1, output),
            tc);
        const float sm11 = linear_mix_float_device(
            scan_lut_fetch_float_cmy_device(scanStage.scanLutPchipSlopeM, res, c, m + 1, y + 1, output),
            scan_lut_fetch_float_cmy_device(scanStage.scanLutPchipSlopeM, res, c + 1, m + 1, y + 1, output),
            tc);
        const float vy0 = hermite_value_float_device(v000, v010, sm00, sm10, tm);
        const float vy1 = hermite_value_float_device(v001, v011, sm01, sm11, tm);

        const float sy0 = bilinear_mix_float_device(
            scan_lut_fetch_float_cmy_device(scanStage.scanLutPchipSlopeY, res, c, m, y, output),
            scan_lut_fetch_float_cmy_device(scanStage.scanLutPchipSlopeY, res, c + 1, m, y, output),
            scan_lut_fetch_float_cmy_device(scanStage.scanLutPchipSlopeY, res, c, m + 1, y, output),
            scan_lut_fetch_float_cmy_device(scanStage.scanLutPchipSlopeY, res, c + 1, m + 1, y, output),
            tc,
            tm);
        const float sy1 = bilinear_mix_float_device(
            scan_lut_fetch_float_cmy_device(scanStage.scanLutPchipSlopeY, res, c, m, y + 1, output),
            scan_lut_fetch_float_cmy_device(scanStage.scanLutPchipSlopeY, res, c + 1, m, y + 1, output),
            scan_lut_fetch_float_cmy_device(scanStage.scanLutPchipSlopeY, res, c, m + 1, y + 1, output),
            scan_lut_fetch_float_cmy_device(scanStage.scanLutPchipSlopeY, res, c + 1, m + 1, y + 1, output),
            tc,
            tm);

        const float interpolated = hermite_value_float_device(vy0, vy1, sy0, sy1, ty);
        const float minimum =
            scan_lut_fetch_float_cell_cmy_device(scanStage.scanLutPchipCellMin, cellRes, c, m, y, output);
        const float maximum =
            scan_lut_fetch_float_cell_cmy_device(scanStage.scanLutPchipCellMax, cellRes, c, m, y, output);
        out[output] = fminf(maximum, fmaxf(minimum, interpolated));
    }
    return true;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

static __device__ __forceinline__ void convert_input_to_sRGB_device(
    const JuicerCuda::FilmRawPayload& cfg,
    const float rgbIn[3],
    float rgbSRGB[3]) {
    float linear[3];
    InputCctfDecodingDevice decoding{};
    decoding.inputColorSpaceIndex = cfg.inputColorSpaceIndex;
    decoding.applyCctfDecoding = cfg.applyCctfDecoding;
    apply_input_cctf_decoding_device(decoding, rgbIn, linear);

    float XYZ[3];
    mat3_mul9_device(cfg.inputRGBToXYZ, linear, XYZ);

    const float* xyzPtr = XYZ;
    float adapted[3];
    if (cfg.applyInputChromaticAdapt) {
        mat3_mul9_device(cfg.inputXYZAdapt, XYZ, adapted);
        xyzPtr = adapted;
    }

    float srgb[3];
    mat3_mul9_device(cfg.xyzToLinearSrgb, xyzPtr, srgb);

    rgbSRGB[0] = device_isfinite(srgb[0]) ? srgb[0] : 0.0f;
    rgbSRGB[1] = device_isfinite(srgb[1]) ? srgb[1] : 0.0f;
    rgbSRGB[2] = device_isfinite(srgb[2]) ? srgb[2] : 0.0f;
}

static __device__ __forceinline__ void convert_input_to_working_xyz_device(
    const JuicerCuda::FilmRawPayload& cfg,
    const float rgbIn[3],
    float workingXYZ[3]) {
    float linear[3];
    InputCctfDecodingDevice decoding{};
    decoding.inputColorSpaceIndex = cfg.inputColorSpaceIndex;
    decoding.applyCctfDecoding = cfg.applyCctfDecoding;
    apply_input_cctf_decoding_device(decoding, rgbIn, linear);

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

static __device__ __forceinline__ void film_tc_layer_exposures_device(
    const float workingXYZ[3],
    const float* JUICER_RESTRICT filmTcLut,
    int filmTcLutExtent,
    float E_out[3]) {
    if (!E_out) {
        return;
    }
    if (!filmTcLut || filmTcLutExtent != JuicerCuda::kFilmTcLutExtent) {
        E_out[0] = E_out[1] = E_out[2] = 0.0f;
        return;
    }

    float XYZ[3] = {
        device_sanitize_channel(workingXYZ[0]),
        device_sanitize_channel(workingXYZ[1]),
        device_sanitize_channel(workingXYZ[2])};

    const float sumXYZ = XYZ[0] + XYZ[1] + XYZ[2];
    const float denom = fmaxf(sumXYZ, 1e-10f);

    const float x = XYZ[0] / denom;
    const float y = XYZ[1] / denom;

    float qx, qy;
    tri2quad_device(x, y, qx, qy);

    const float r = sample_film_tc_lut_cubic_device(
        filmTcLut,
        filmTcLutExtent,
        0,
        qx,
        qy);
    const float g = sample_film_tc_lut_cubic_device(
        filmTcLut,
        filmTcLutExtent,
        1,
        qx,
        qy);
    const float b = sample_film_tc_lut_cubic_device(
        filmTcLut,
        filmTcLutExtent,
        2,
        qx,
        qy);

    const float rSafe = device_isfinite(r) ? r : 0.0f;
    const float gSafe = device_isfinite(g) ? g : 0.0f;
    const float bSafe = device_isfinite(b) ? b : 0.0f;

    // lutIntegrated stores R,G,B; map to E_out order B,G,R.
    E_out[0] = device_sanitize_channel(sumXYZ * bSafe);
    E_out[1] = device_sanitize_channel(sumXYZ * gSafe);
    E_out[2] = device_sanitize_channel(sumXYZ * rSafe);
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

    float Eb = 0.0f;
    float Eg = 0.0f;
    float Er = 0.0f;

    for (int i = 0; i < K; ++i) {
        const float illum_i = ldg_f(illum + i);
        const std::ptrdiff_t basisOffset = static_cast<std::ptrdiff_t>(i) * 3;
        const float b0 = ldg_f(mallettBasis + basisOffset);
        const float b1 = ldg_f(mallettBasis + basisOffset + 1);
        const float b2 = ldg_f(mallettBasis + basisOffset + 2);
        const float spd = (r * b0 + g * b1 + b * b2) * illum_i;
        if (!device_isfinite(spd)) {
            continue;
        }
        const float sb = ldg_f(sensB + i);
        const float sg = ldg_f(sensG + i);
        const float sr = ldg_f(sensR + i);
        if (isfinite(sb))
            Eb += spd * sb;
        if (isfinite(sg))
            Eg += spd * sg;
        if (isfinite(sr))
            Er += spd * sr;
    }

    E_out[0] = device_isfinite(Eb) ? fmaxf(0.0f, Eb) : 0.0f;
    E_out[1] = device_isfinite(Eg) ? fmaxf(0.0f, Eg) : 0.0f;
    E_out[2] = device_isfinite(Er) ? fmaxf(0.0f, Er) : 0.0f;
}

static __device__ __forceinline__ void reconstruct_film_raw_device(
    const JuicerCuda::FilmRawPayload& config,
    const JuicerCuda::FilmReconstructionPayload& reconstruction,
    const float rgbIn[3],
    float filmRaw[3]) {
    switch (config.rgbToRawMethod) {
        case JuicerCuda::kFilmRawMethodHanatos2025:
        case JuicerCuda::kFilmRawMethodArctic2026beta04: {
            float workingXYZ[3] = {};
            convert_input_to_working_xyz_device(config, rgbIn, workingXYZ);
            film_tc_layer_exposures_device(
                workingXYZ,
                reconstruction.filmTcLut,
                reconstruction.filmTcLutExtent,
                filmRaw);
            return;
        }
        case JuicerCuda::kFilmRawMethodMallett2019: {
            float rgbSRGB[3] = {};
            convert_input_to_sRGB_device(config, rgbIn, rgbSRGB);
            mallett_layer_exposures_device(
                rgbSRGB,
                reconstruction.mallettBasis,
                reconstruction.tablesIllum,
                reconstruction.tablesK,
                reconstruction.sensB.y,
                reconstruction.sensG.y,
                reconstruction.sensR.y,
                filmRaw);
            for (int channel = 0; channel < 3; ++channel) {
                filmRaw[channel] = fmaxf(
                    0.0f,
                    filmRaw[channel] * config.mallettGreenMidgrayScale);
            }
            return;
        }
    }
}

template <bool ApplyRouteCorrection, typename Params>
static __device__ __forceinline__ void compute_film_linear_exposure_device(
    const Params& params,
    const float rgbIn[3],
    float filmRaw[3]) {
    const JuicerCuda::FilmExposurePayload& expose = params.filmExpose;
    reconstruct_film_raw_device(
        params.filmRaw,
        expose.reconstruction,
        rgbIn,
        filmRaw);

    float autoExposureScale = 1.0f;
    if (expose.exposureScaleDevice) {
        const float deviceScale = *expose.exposureScaleDevice;
        if (isfinite(deviceScale) && deviceScale > 0.0f) {
            autoExposureScale = deviceScale;
        }
    }
    float exposureScale = autoExposureScale * expose.manualExposureScale;
    if constexpr (ApplyRouteCorrection) {
        exposureScale *= expose.routeCorrectionScale;
    }
    exposureScale =
        isfinite(exposureScale) && exposureScale > 0.0f ? exposureScale : 1.0f;
    for (int channel = 0; channel < 3; ++channel) {
        filmRaw[channel] = fmaxf(0.0f, filmRaw[channel] * exposureScale);
    }
}

template <typename Params>
static __device__ __forceinline__ void compute_camera_film_linear_exposure_device(
    const Params& params,
    const float rgbIn[3],
    float filmRaw[3]) {
    compute_film_linear_exposure_device<false>(params, rgbIn, filmRaw);
}

template <typename Params>
static __device__ __forceinline__ void apply_film_route_correction_device(
    const Params& params,
    float filmRaw[3]) {
    const float routeCorrectionScale = params.filmExpose.routeCorrectionScale;
    for (int i = 0; i < 3; ++i) {
        filmRaw[i] = fmaxf(0.0f, filmRaw[i] * routeCorrectionScale);
    }
}

template <typename Params>
static __device__ __forceinline__ void compute_film_raw_device(
    const Params& params,
    const float rgbIn[3],
    float filmRaw[3]) {
    compute_film_linear_exposure_device<true>(params, rgbIn, filmRaw);
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

static constexpr int kJuicerLogRawMaskB = 0x1;
static constexpr int kJuicerLogRawMaskG = 0x2;
static constexpr int kJuicerLogRawMaskR = 0x4;
static constexpr int kJuicerLogRawMaskBgr =
    kJuicerLogRawMaskB | kJuicerLogRawMaskG | kJuicerLogRawMaskR;

template <typename Params>
static __device__ __forceinline__ void compute_logE_raw_selected_device(
    const Params& params,
    const float rgbIn[3],
    int channelMask,
    float logE_raw[3]) {
    channelMask &= kJuicerLogRawMaskBgr;
    if (channelMask == 0) {
        return;
    }
    float fullLogRaw[3] = {0.0f, 0.0f, 0.0f};
    compute_logE_raw_device(params, rgbIn, fullLogRaw);
    if (channelMask & kJuicerLogRawMaskB) {
        logE_raw[0] = fullLogRaw[0];
    }
    if (channelMask & kJuicerLogRawMaskG) {
        logE_raw[1] = fullLogRaw[1];
    }
    if (channelMask & kJuicerLogRawMaskR) {
        logE_raw[2] = fullLogRaw[2];
    }
}

template <typename Params>
static __device__ __forceinline__ void compute_logE_from_film_raw_device(
    const Params& params,
    const float filmRaw[3],
    FilmDevelopIntermediatesDevice outputs) {
    const JuicerCuda::FilmDevelopPayload& develop = params.filmDevelop;

    compute_logE_raw_from_film_raw_device(params, filmRaw, outputs.logERaw);

    outputs.logESanitized[0] = sanitize_inf_logE_for_curve_device(outputs.logERaw[0], develop.densB);
    outputs.logESanitized[1] = sanitize_inf_logE_for_curve_device(outputs.logERaw[1], develop.densG);
    outputs.logESanitized[2] = sanitize_inf_logE_for_curve_device(outputs.logERaw[2], develop.densR);

    outputs.layerPre[0] =
        sample_density_at_logE_device(develop.densB, outputs.logESanitized[0], develop.gammaFactorB);
    outputs.layerPre[1] =
        sample_density_at_logE_device(develop.densG, outputs.logESanitized[1], develop.gammaFactorG);
    outputs.layerPre[2] =
        sample_density_at_logE_device(develop.densR, outputs.logESanitized[2], develop.gammaFactorR);
}

template <typename Params>
static __device__ __forceinline__ void compute_logE_from_camera_film_linear_exposure_device(
    const Params& params,
    const float cameraFilmLinear[3],
    FilmDevelopIntermediatesDevice outputs) {
    float correctedFilmRaw[3] = {
        cameraFilmLinear[0],
        cameraFilmLinear[1],
        cameraFilmLinear[2]};
    apply_film_route_correction_device(params, correctedFilmRaw);
    compute_logE_from_film_raw_device(params, correctedFilmRaw, outputs);
}

template <typename Params>
static __device__ __forceinline__ void compute_logE_and_layer_pre_device(
    const Params& params,
    const float rgbIn[3],
    FilmDevelopIntermediatesDevice outputs) {
    float filmRaw[3] = {0.0f, 0.0f, 0.0f};
    compute_film_raw_device(params, rgbIn, filmRaw);
    compute_logE_from_film_raw_device(params, filmRaw, outputs);
}

static __device__ __forceinline__ float density_to_light_sample_agx_device(float density, float illuminant) {
    constexpr float kLog2_10 = 3.32192809488736234787f;
    const float transmitted = exp2f(-density * kLog2_10) * illuminant;
    return isnan(transmitted) ? 0.0f : transmitted;
}

static __device__ __forceinline__ bool print_spectral_integrate_device(
    const JuicerCuda::PrintExposePayload& expose,
    const float D_cmy[3],
    float rawPrint[3]) {
    if (!D_cmy || !rawPrint) {
        return false;
    }

    const int K = expose.printIllumK;
    if (K <= 0 || !expose.printIllumFiltered) {
        rawPrint[0] = rawPrint[1] = rawPrint[2] = 0.0f;
        return false;
    }

    const int negK = expose.negTables.K;
    if (negK != K || !expose.negTables.epsC || !expose.negTables.epsM || !expose.negTables.epsY) {
        rawPrint[0] = rawPrint[1] = rawPrint[2] = 0.0f;
        return false;
    }

    const bool haveBaseline = (expose.negTables.hasBaseline != 0) && expose.negTables.baseDensityMin;

    if (!expose.printSensC.y || !expose.printSensM.y || !expose.printSensY.y) {
        rawPrint[0] = rawPrint[1] = rawPrint[2] = 0.0f;
        return false;
    }
    if (expose.printSensC.n < K || expose.printSensM.n < K || expose.printSensY.n < K) {
        rawPrint[0] = rawPrint[1] = rawPrint[2] = 0.0f;
        return false;
    }

    float accumC = 0.0f;
    float accumM = 0.0f;
    float accumY = 0.0f;
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

        const float baseD = haveBaseline ? ldg_f(expose.negTables.baseDensityMin + i) : 0.0f;
        const float densitySpectral =
            D_cmy[0] * ldg_f(expose.negTables.epsC + i) +
            D_cmy[1] * ldg_f(expose.negTables.epsM + i) +
            D_cmy[2] * ldg_f(expose.negTables.epsY + i) +
            baseD;

        const float e = density_to_light_sample_agx_device(densitySpectral, ldg_f(expose.printIllumFiltered + i));

        if (activeC)
            accumC += e * sC;
        if (activeM)
            accumM += e * sM;
        if (activeY)
            accumY += e * sY;
    }

    rawPrint[0] = accumC;
    rawPrint[1] = accumM;
    rawPrint[2] = accumY;
    return true;
}

static __device__ __forceinline__ void print_apply_exposure_scale_device(
    const JuicerCuda::PrintExposePayload& expose,
    float rawPrint[3]) {
    if (!rawPrint) {
        return;
    }

    const float expPrint = expose.printExposure;
    const float kMid = expose.printMidgrayFactor;

    rawPrint[0] *= kMid;
    rawPrint[1] *= kMid;
    rawPrint[2] *= kMid;

    const float preflash = expose.printPreflashExposure;
    if (isfinite(preflash) && preflash > 0.0f) {
        rawPrint[0] += expose.printPreflashRaw[0] * preflash;
        rawPrint[1] += expose.printPreflashRaw[1] * preflash;
        rawPrint[2] += expose.printPreflashRaw[2] * preflash;
    }

    rawPrint[0] *= expPrint;
    rawPrint[1] *= expPrint;
    rawPrint[2] *= expPrint;

    const float routeCorrectionScale = expose.routeCorrectionScale;
    rawPrint[0] *= routeCorrectionScale;
    rawPrint[1] *= routeCorrectionScale;
    rawPrint[2] *= routeCorrectionScale;
}

static __device__ __forceinline__ bool print_inner_raw_device(
    const JuicerCuda::PrintExposePayload& expose,
    const float filmDensityCmy[3],
    float innerRaw[3],
    float filmTransmittance) {
    if (!print_spectral_integrate_device(
            expose,
            filmDensityCmy,
            innerRaw)) {
        return false;
    }

    if (filmTransmittance != 1.0f) {
        innerRaw[0] *= filmTransmittance;
        innerRaw[1] *= filmTransmittance;
        innerRaw[2] *= filmTransmittance;
    }
    innerRaw[0] *= expose.printMidgrayFactor;
    innerRaw[1] *= expose.printMidgrayFactor;
    innerRaw[2] *= expose.printMidgrayFactor;
    if (expose.printPreflashExposure > 0.0f) {
        innerRaw[0] +=
            expose.printPreflashRaw[0] * expose.printPreflashExposure;
        innerRaw[1] +=
            expose.printPreflashRaw[1] * expose.printPreflashExposure;
        innerRaw[2] +=
            expose.printPreflashRaw[2] * expose.printPreflashExposure;
    }
    return true;
}

static __device__ __forceinline__ void print_linear_for_diffusion_device(
    const JuicerCuda::PrintExposePayload& expose,
    const float innerRaw[3],
    float linearForDiffusion[3]) {
    constexpr float kLogEps = 1e-10f;
    const float outerScale =
        expose.printExposure * expose.routeCorrectionScale;
    linearForDiffusion[0] =
        (fmaxf(innerRaw[0], 0.0f) + kLogEps) * outerScale;
    linearForDiffusion[1] =
        (fmaxf(innerRaw[1], 0.0f) + kLogEps) * outerScale;
    linearForDiffusion[2] =
        (fmaxf(innerRaw[2], 0.0f) + kLogEps) * outerScale;
}

static __device__ __forceinline__ void print_log_encode_device(
    const float rawPrint[3],
    float logPrint[3]) {
    if (!rawPrint || !logPrint) {
        return;
    }

    constexpr float kLogEps = 1e-10f;
    logPrint[0] = log10f(rawPrint[0] + kLogEps);
    logPrint[1] = log10f(rawPrint[1] + kLogEps);
    logPrint[2] = log10f(rawPrint[2] + kLogEps);
}

static __device__ __forceinline__ void print_log_encode_diffused_device(
    const float diffusedRaw[3],
    float logPrint[3]) {
    constexpr float kLogEps = 1e-10f;
    logPrint[0] = log10f(fmaxf(diffusedRaw[0], 0.0f) + kLogEps);
    logPrint[1] = log10f(fmaxf(diffusedRaw[1], 0.0f) + kLogEps);
    logPrint[2] = log10f(fmaxf(diffusedRaw[2], 0.0f) + kLogEps);
}

static __device__ __forceinline__ void print_sample_density_curves_device(
    const JuicerCuda::PrintDevelopPayload& develop,
    const float logPrint[3],
    float D_cmy[3]) {
    if (!logPrint || !D_cmy) {
        return;
    }

    D_cmy[0] = sample_density_at_logE_device(develop.printDcC, logPrint[0], 1.0f);
    D_cmy[1] = sample_density_at_logE_device(develop.printDcM, logPrint[1], 1.0f);
    D_cmy[2] = sample_density_at_logE_device(develop.printDcY, logPrint[2], 1.0f);
}

static __device__ __forceinline__ void apply_print_pipeline_device(
    const JuicerCuda::PrintExposePayload& expose,
    const JuicerCuda::PrintDevelopPayload& develop,
    float D_cmy[3],
    float filmTransmittance) {
    if (!expose.active || !D_cmy) {
        return;
    }

    float rawPrint[3] = {0.0f, 0.0f, 0.0f};
    if (!print_spectral_integrate_device(expose, D_cmy, rawPrint)) {
        D_cmy[0] = D_cmy[1] = D_cmy[2] = 0.0f;
        return;
    }

    if (filmTransmittance != 1.0f) {
        rawPrint[0] *= filmTransmittance;
        rawPrint[1] *= filmTransmittance;
        rawPrint[2] *= filmTransmittance;
    }
    print_apply_exposure_scale_device(expose, rawPrint);
    float logPrint[3] = {0.0f, 0.0f, 0.0f};
    print_log_encode_device(rawPrint, logPrint);
    print_sample_density_curves_device(develop, logPrint, D_cmy);
}

static __device__ __forceinline__ bool evaluate_langmuir_device(
    float value,
    float knee,
    float reference,
    float& output) {
    output = nanf("");
    if (!isfinite(value) || !isfinite(knee) || !(knee > 0.0f) ||
        !isfinite(reference)) {
        return false;
    }

    const float denominator = knee + value;
    const float kneePlusReference = knee + reference;
    const bool directProductSafe =
        isfinite(denominator) && denominator != 0.0f &&
        isfinite(kneePlusReference) &&
        (value == 0.0f ||
         fabsf(value) <= FLT_MAX / fabsf(kneePlusReference));
    if (directProductSafe) {
        output = value * kneePlusReference / denominator;
        return isfinite(output);
    }

    const float numeratorScale = 1.0f + reference / knee;
    if (!isfinite(numeratorScale)) {
        return false;
    }
    if (value >= 0.0f) {
        const float denominatorScale = 1.0f + value / knee;
        if (!isfinite(denominatorScale) || denominatorScale == 0.0f) {
            return false;
        }
        output = (value / denominatorScale) * numeratorScale;
        return isfinite(output);
    }

    // Retain the directly rounded signed denominator near the receiver pole.
    if (!isfinite(denominator) || denominator == 0.0f) {
        return false;
    }
    const float denominatorScale = denominator / knee;
    if (!isfinite(denominatorScale) || denominatorScale == 0.0f) {
        return false;
    }
    if (fabsf(value) <= FLT_MAX / fabsf(numeratorScale)) {
        output = value * numeratorScale / denominatorScale;
    } else {
        output = (value / denominatorScale) * numeratorScale;
    }
    return isfinite(output);
}

static __device__ __forceinline__ unsigned int compute_dir_corrections_device(
    const JuicerCuda::DirPayload& dir,
    const float layerDensities[3],
    float outLayerCorrections[3]) {
    if (!outLayerCorrections) {
        return 0x7u;
    }
    if (dir.mode == JuicerCuda::DirMode::Inactive) {
        outLayerCorrections[0] = 0.0f;
        outLayerCorrections[1] = 0.0f;
        outLayerCorrections[2] = 0.0f;
        return 0u;
    }

    float donors[3] = {
        dir.dMax[0] - layerDensities[0],
        dir.dMax[1] - layerDensities[1],
        dir.dMax[2] - layerDensities[2]};
    unsigned int failures = 0u;
    if (dir.mode == JuicerCuda::DirMode::NegativeDonorLangmuir) {
#pragma unroll
        for (int channel = 0; channel < 3; ++channel) {
            const float density = layerDensities[channel];
            if (!evaluate_langmuir_device(
                    density,
                    dir.donorK[channel],
                    dir.dRef[channel],
                    donors[channel])) {
                failures |= 1u << channel;
            }
        }
    }

    outLayerCorrections[0] = dir.M[0] * donors[0] + dir.M[3] * donors[1] + dir.M[6] * donors[2];
    outLayerCorrections[1] = dir.M[1] * donors[0] + dir.M[4] * donors[1] + dir.M[7] * donors[2];
    outLayerCorrections[2] = dir.M[2] * donors[0] + dir.M[5] * donors[1] + dir.M[8] * donors[2];
#pragma unroll
    for (int channel = 0; channel < 3; ++channel) {
        if (!isfinite(outLayerCorrections[channel])) {
            failures |= 1u << channel;
        }
    }
    return failures;
}

static __device__ __forceinline__ unsigned int finalize_dir_corrections_device(
    const JuicerCuda::DirPayload& dir,
    float corrections[3]) {
    if (dir.mode != JuicerCuda::DirMode::PositiveReceiverLangmuir) {
        return 0u;
    }
    unsigned int failures = 0u;
#pragma unroll
    for (int channel = 0; channel < 3; ++channel) {
        if (dir.receiverCRef[channel] == 0.0f) {
            corrections[channel] = 0.0f;
            continue;
        }
        const float value = corrections[channel];
        if (!evaluate_langmuir_device(
                value,
                dir.receiverKr[channel],
                dir.receiverCRef[channel],
                corrections[channel])) {
            failures |= 1u << channel;
        }
    }
    return failures;
}

static __device__ __forceinline__ void signal_dir_failure_device(
    int* status,
    unsigned int channelMask) {
    if (status && channelMask != 0u) {
        constexpr unsigned int kDirFailure = 1u << 8u;
        atomicOr(
            reinterpret_cast<unsigned int*>(status),
            kDirFailure | ((channelMask & 0x7u) << 9u));
    }
}

static __device__ __forceinline__ unsigned int apply_dir_runtime_logE_device(
    float logE_BGR[3],
    const float layerD_BGR[3],
    const JuicerCuda::DirPayload& dir) {
    if (!logE_BGR || !layerD_BGR) {
        return 0x7u;
    }
    if (dir.mode == JuicerCuda::DirMode::Inactive) {
        return 0u;
    }

    float corrections[3] = {0.0f, 0.0f, 0.0f};
    unsigned int failures =
        compute_dir_corrections_device(dir, layerD_BGR, corrections);
    failures |= finalize_dir_corrections_device(dir, corrections);
    if (failures != 0u) {
        return failures;
    }
    unsigned int subtractionFailures = 0u;
#pragma unroll
    for (int channel = 0; channel < 3; ++channel) {
        const float original = logE_BGR[channel];
        if (isnan(original)) {
            subtractionFailures |= 1u << channel;
            continue;
        }
        logE_BGR[channel] = original - corrections[channel];
        if ((!isinf(original) && !isfinite(logE_BGR[channel])) ||
            (isinf(original) &&
             (!isinf(logE_BGR[channel]) ||
              signbit(original) != signbit(logE_BGR[channel])))) {
            subtractionFailures |= 1u << channel;
        }
    }
    return subtractionFailures;
}

static __device__ __forceinline__ bool juicer_cuda_spatial_dir_filtered_correction_active_device(
    const JuicerCuda::FilmDevelopPayload& develop) {
    return develop.spatialDir.active &&
           develop.spatialDir.filteredCorrectionY &&
           develop.spatialDir.filteredCorrectionM &&
           develop.spatialDir.filteredCorrectionC;
}

static __device__ __forceinline__ bool juicer_cuda_spatial_dir_cached_log_raw_active_device(
    const JuicerCuda::FilmDevelopPayload& develop) {
    return develop.spatialDir.logRawB &&
           develop.spatialDir.logRawG &&
           develop.spatialDir.logRawR;
}

static __device__ __forceinline__ void juicer_cuda_load_spatial_dir_cached_log_raw_device(
    const JuicerCuda::FilmDevelopPayload& develop,
    std::size_t pixelIndex,
    float logRawBgr[3]) {
    logRawBgr[0] = ldg_f(develop.spatialDir.logRawB + pixelIndex);
    logRawBgr[1] = ldg_f(develop.spatialDir.logRawG + pixelIndex);
    logRawBgr[2] = ldg_f(develop.spatialDir.logRawR + pixelIndex);
}

static __device__ __forceinline__ unsigned int juicer_cuda_develop_dir_final_device(
    const JuicerCuda::FilmDevelopPayload& develop,
    const float logRawBgr[3],
    std::size_t pixelIndex,
    float densityCmy[3]) {
    // The focused schedule overwrites these correction planes with density.
    // They are mutable for this kernel and must not use the read-only __ldg path.
    const float filteredCorrectionY =
        develop.spatialDir.filteredCorrectionY[pixelIndex];
    const float filteredCorrectionM =
        develop.spatialDir.filteredCorrectionM[pixelIndex];
    const float filteredCorrectionC =
        develop.spatialDir.filteredCorrectionC[pixelIndex];

    float corrections[3] = {
        filteredCorrectionY,
        filteredCorrectionM,
        filteredCorrectionC};
    const unsigned int failures =
        finalize_dir_corrections_device(develop.dir, corrections);
    if (failures != 0u) {
        densityCmy[0] = densityCmy[1] = densityCmy[2] = nanf("");
        return failures;
    }
    float correctedLogRaw[3] = {0.0f, 0.0f, 0.0f};
    unsigned int subtractionFailures = 0u;
#pragma unroll
    for (int channel = 0; channel < 3; ++channel) {
        const float original = logRawBgr[channel];
        if (isnan(original)) {
            subtractionFailures |= 1u << channel;
            continue;
        }
        correctedLogRaw[channel] = original - corrections[channel];
        if ((!isinf(original) && !isfinite(correctedLogRaw[channel])) ||
            (isinf(original) &&
             (!isinf(correctedLogRaw[channel]) ||
              signbit(original) != signbit(correctedLogRaw[channel])))) {
            subtractionFailures |= 1u << channel;
        }
    }
    if (subtractionFailures != 0u) {
        densityCmy[0] = densityCmy[1] = densityCmy[2] = nanf("");
        return subtractionFailures;
    }

    const JuicerCuda::DeviceCurveView compensatedB = develop.dirDensB;
    const JuicerCuda::DeviceCurveView compensatedG = develop.dirDensG;
    const JuicerCuda::DeviceCurveView compensatedR = develop.dirDensR;

    correctedLogRaw[0] = sanitize_inf_logE_for_curve_device(correctedLogRaw[0], develop.densB);
    correctedLogRaw[1] = sanitize_inf_logE_for_curve_device(correctedLogRaw[1], develop.densG);
    correctedLogRaw[2] = sanitize_inf_logE_for_curve_device(correctedLogRaw[2], develop.densR);

    const float densityY = sample_density_at_logE_device(compensatedB, correctedLogRaw[0], develop.gammaFactorB);
    const float densityM = sample_density_at_logE_device(compensatedG, correctedLogRaw[1], develop.gammaFactorG);
    const float densityC = sample_density_at_logE_device(compensatedR, correctedLogRaw[2], develop.gammaFactorR);

    densityCmy[0] = densityC;
    densityCmy[1] = densityM;
    densityCmy[2] = densityY;
    return 0u;
}
