// Cuda/Scan/JuicerCudaScanOutput.cuh
// Output encoding + scan output helpers (header-only).
#pragma once

#include <cuda_runtime.h>

#include <cmath>

#include "Cuda/JuicerCudaPayloads.h"

static __device__ __forceinline__ double clamp01d_device(double v) {
    if (v <= 0.0) return 0.0;
    if (v >= 1.0) return 1.0;
    return v;
}

static __device__ __forceinline__ double encode_sRGBd_device(double v) {
    if (v <= 0.0031308) {
        return 12.92 * v;
    }
    return 1.055 * pow(v, 1.0 / 2.4) - 0.055;
}

static __device__ __forceinline__ double encode_gammad_signed_device(double v, double exponent) {
    const double mag = pow(fabs(v), exponent);
    return copysign(mag, v);
}

static __device__ __forceinline__ double encode_BT2020d_device(double v, double a, double b) {
    if (v < b) {
        return v * 4.5;
    }
    return a * pow(v, 0.45) - (a - 1.0);
}

static __device__ __forceinline__ double encode_ProPhotod_device(double v, double threshold, double exponent) {
    if (v < threshold) {
        return v * 16.0;
    }
    return pow(v, exponent);
}

static __device__ __forceinline__ double encode_DaVinciIntermediated_device(double v, const JuicerCuda::CctfPayload& cctf) {
    const double linear = fmax(0.0, v);
    if (linear <= static_cast<double>(cctf.linearCutoff)) {
        return linear * static_cast<double>(cctf.d);
    }
    // (log2(linear + a) + b) * c
    return (log2(linear + static_cast<double>(cctf.a)) + static_cast<double>(cctf.b)) * static_cast<double>(cctf.c);
}

static __device__ __forceinline__ double encode_channel_double_device(const JuicerCuda::CctfPayload& cctf, double v) {
    switch (cctf.kind) {
    case 0: // Linear
        return v;
    case 1: // Gamma
        return encode_gammad_signed_device(v, static_cast<double>(cctf.gamma));
    case 2: // SRGB
        return encode_sRGBd_device(v);
    case 3: // BT2020
        return encode_BT2020d_device(v, static_cast<double>(cctf.a), static_cast<double>(cctf.b));
    case 4: // ProPhoto
        return encode_ProPhotod_device(v, static_cast<double>(cctf.linearCutoff), static_cast<double>(cctf.gamma));
    case 5: // DaVinciIntermediate
        return encode_DaVinciIntermediated_device(v, cctf);
    default:
        return clamp01d_device(v);
    }
}

static __device__ __forceinline__ void apply_output_encoding_device(const JuicerCuda::OutputEncodingPayload& enc, double rgb[3]) {
    if (!rgb) {
        return;
    }

    double linear[3];
    if (enc.inputIsOutputSpace) {
        linear[0] = rgb[0];
        linear[1] = rgb[1];
        linear[2] = rgb[2];
    }
    else {
        linear[0] =
            static_cast<double>(enc.dwgToOutput[0]) * rgb[0] +
            static_cast<double>(enc.dwgToOutput[1]) * rgb[1] +
            static_cast<double>(enc.dwgToOutput[2]) * rgb[2];
        linear[1] =
            static_cast<double>(enc.dwgToOutput[3]) * rgb[0] +
            static_cast<double>(enc.dwgToOutput[4]) * rgb[1] +
            static_cast<double>(enc.dwgToOutput[5]) * rgb[2];
        linear[2] =
            static_cast<double>(enc.dwgToOutput[6]) * rgb[0] +
            static_cast<double>(enc.dwgToOutput[7]) * rgb[1] +
            static_cast<double>(enc.dwgToOutput[8]) * rgb[2];
    }

    if (enc.preserveLinearRange) {
        rgb[0] = linear[0];
        rgb[1] = linear[1];
        rgb[2] = linear[2];
        return; // no encoding, no clamp
    }

    if (enc.applyCctfEncoding) {
        rgb[0] = encode_channel_double_device(enc.cctf, linear[0]);
        rgb[1] = encode_channel_double_device(enc.cctf, linear[1]);
        rgb[2] = encode_channel_double_device(enc.cctf, linear[2]);
    }
    else {
        rgb[0] = linear[0];
        rgb[1] = linear[1];
        rgb[2] = linear[2];
    }

    // agx parity: encode first, then clip
    rgb[0] = clamp01d_device(rgb[0]);
    rgb[1] = clamp01d_device(rgb[1]);
    rgb[2] = clamp01d_device(rgb[2]);
}

static __device__ __forceinline__ void signal_scan_error_device(int* flag) {
    if (flag) {
        atomicExch(flag, 1);
    }
}

static __device__ __forceinline__ void mat3_mul_vec_double_device(const float m9[9], const double v3[3], double out3[3]) {
    out3[0] =
        static_cast<double>(m9[0]) * v3[0] +
        static_cast<double>(m9[1]) * v3[1] +
        static_cast<double>(m9[2]) * v3[2];
    out3[1] =
        static_cast<double>(m9[3]) * v3[0] +
        static_cast<double>(m9[4]) * v3[1] +
        static_cast<double>(m9[5]) * v3[2];
    out3[2] =
        static_cast<double>(m9[6]) * v3[0] +
        static_cast<double>(m9[7]) * v3[1] +
        static_cast<double>(m9[8]) * v3[2];
}
