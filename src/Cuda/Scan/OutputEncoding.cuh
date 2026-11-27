#pragma once

#include "Cuda/JuicerCudaDeviceHelpers.cuh"

namespace JuicerCuda {
    __device__ __forceinline__ double clamp01d_device(double v) {
        if (v <= 0.0)
            return 0.0;
        if (v >= 1.0)
            return 1.0;
        return v;
    }
    __device__ __forceinline__ float encode_sRGB_device(float v) {
        if (v <= 0.0031308f) {
            return 12.92f * v;
        }
        return 1.055f * powf(v, 1.0f / 2.4f) - 0.055f;
    }

    __device__ __forceinline__ float encode_gamma_signed_device(float v, float exponent) {
        const float mag = powf(fabsf(v), exponent);
        return copysignf(mag, v);
    }

    __device__ __forceinline__ float encode_BT2020_device(float v, const JuicerCuda::CctfPayload& cctf) {
        if (v < cctf.b) {
            return v * 4.5f;
        }
        return cctf.a * powf(v, 0.45f) - (cctf.a - 1.0f);
    }

    __device__ __forceinline__ float encode_ProPhoto_device(float v, const JuicerCuda::CctfPayload& cctf) {
        if (v < cctf.linearCutoff) {
            return v * 16.0f;
        }
        return powf(v, cctf.gamma);
    }

    __device__ __forceinline__ float encode_DaVinciIntermediate_device(float v, const JuicerCuda::CctfPayload& cctf) {
        const float linear = fmaxf(0.0f, v);
        if (linear <= cctf.linearCutoff) {
            return linear * cctf.d;
        }
        return (log2f(linear + cctf.a) + cctf.b) * cctf.c;
    }

    __device__ __forceinline__ float encode_channel_device(const JuicerCuda::CctfPayload& cctf, float v) {
        switch (cctf.kind) {
            case 0:
                return v;
            case 1:
                return encode_gamma_signed_device(v, cctf.gamma);
            case 2:
                return encode_sRGB_device(v);
            case 3:
                return encode_BT2020_device(v, cctf);
            case 4:
                return encode_ProPhoto_device(v, cctf);
            case 5:
                return encode_DaVinciIntermediate_device(v, cctf);
            default:
                return static_cast<float>(clamp01d_device(v));
        }
    }

    __device__ __forceinline__ void apply_output_encoding_device(const JuicerCuda::OutputEncodingPayload& enc, double rgb[3]) {
        if (!rgb) {
            return;
        }

        double linear[3];
        if (enc.inputIsOutputSpace) {
            linear[0] = rgb[0];
            linear[1] = rgb[1];
            linear[2] = rgb[2];
        } else {
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

        if (enc.applyCctfEncoding) {
            rgb[0] = encode_channel_device(enc.cctf, static_cast<float>(linear[0]));
            rgb[1] = encode_channel_device(enc.cctf, static_cast<float>(linear[1]));
            rgb[2] = encode_channel_device(enc.cctf, static_cast<float>(linear[2]));
        } else {
            rgb[0] = linear[0];
            rgb[1] = linear[1];
            rgb[2] = linear[2];
        }

        rgb[0] = clamp01d_device(rgb[0]);
        rgb[1] = clamp01d_device(rgb[1]);
        rgb[2] = clamp01d_device(rgb[2]);
    }

} // namespace JuicerCuda
