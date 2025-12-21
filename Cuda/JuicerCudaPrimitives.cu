// Cuda/JuicerCudaPrimitives.cu
//
// Phase 2: core CUDA math primitives with CPU-parity semantics.
//
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "Cuda/JuicerCudaPayloads.h"
#include "Cuda/JuicerCudaKernelsUtil.cuh"
#include "Cuda/JuicerCudaPrintPipeline.cuh"

// Optics kernels are defined in Cuda/Scan/JuicerCudaScannerOptics.cu.
__global__ void optics_glare_generate_kernel(
    float* out,
    int width,
    int height,
    std::uint64_t glareSeed,
    std::uint64_t mediumId,
    int originX,
    int originY,
    float percent,
    float roughness);
__global__ void optics_blur_horizontal_kernel(
    const float* JUICER_RESTRICT in,
    float* out,
    int width,
    int height,
    const float* JUICER_RESTRICT k,
    int radius);
__global__ void optics_blur_vertical_kernel(
    const float* JUICER_RESTRICT in,
    float* out,
    int width,
    int height,
    const float* JUICER_RESTRICT k,
    int radius);
__global__ void optics_unsharp_combine_kernel(
    float* inOut,
    const float* JUICER_RESTRICT blurred,
    int n,
    float amount);

namespace {

    __device__ __forceinline__ double clamp01d_device(double v) {
        if (v <= 0.0) return 0.0;
        if (v >= 1.0) return 1.0;
        return v;
    }

    __device__ __forceinline__ double encode_sRGBd_device(double v) {
        if (v <= 0.0031308) {
            return 12.92 * v;
        }
        return 1.055 * pow(v, 1.0 / 2.4) - 0.055;
    }

    __device__ __forceinline__ double encode_gammad_device(double v, double exponent) {
        return pow(v, exponent);
    }

    __device__ __forceinline__ double encode_BT2020d_device(double v, double a, double b) {
        if (v < b) {
            return v * 4.5;
        }
        return a * pow(v, 0.45) - (a - 1.0);
    }

    __device__ __forceinline__ double encode_ProPhotod_device(double v, double threshold, double exponent) {
        if (v < threshold) {
            return v * 16.0;
        }
        return pow(v, exponent);
    }

    __device__ __forceinline__ double encode_DaVinciIntermediated_device(double v, const JuicerCuda::CctfPayload& cctf) {
        const double linear = fmax(0.0, v);
        if (linear <= static_cast<double>(cctf.linearCutoff)) {
            return linear * static_cast<double>(cctf.d);
        }
        // (log2(linear + a) + b) * c
        return (log2(linear + static_cast<double>(cctf.a)) + static_cast<double>(cctf.b)) * static_cast<double>(cctf.c);
    }

    __device__ __forceinline__ double encode_channel_double_device(const JuicerCuda::CctfPayload& cctf, double v) {
        switch (cctf.kind) {
        case 0: // Linear
            return v;
        case 1: // Gamma
            return encode_gammad_device(v, static_cast<double>(cctf.gamma));
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

    __device__ __forceinline__ void apply_output_encoding_device(const JuicerCuda::OutputEncodingPayload& enc, double rgb[3]) {
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

    __device__ __forceinline__ void signal_scan_error_device(int* flag) {
        if (flag) {
            atomicExch(flag, 1);
        }
    }

    __device__ __forceinline__ void mat3_mul_vec_double_device(const float m9[9], const double v3[3], double out3[3]) {
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

    __device__ __forceinline__ void scan_spectral_to_log_xyz_device(
        const JuicerCuda::ScanTablesPayload& medium,
        const double D_norm[3],
        double logXYZ[3])
    {
        if (!D_norm || !logXYZ) {
            return;
        }
        if (!medium.epsC || !medium.epsM || !medium.epsY || !medium.Ax || !medium.Ay || !medium.Az || medium.K <= 0) {
            logXYZ[0] = logXYZ[1] = logXYZ[2] = nan("");
            return;
        }

        double D_denorm0;
        double D_denorm1;
        double D_denorm2;
        if (medium.mediumIsNegative) {
            D_denorm0 = D_norm[0] / static_cast<double>(medium.inv_max_cmy[0]) - static_cast<double>(medium.min_cmy[0]);
            D_denorm1 = D_norm[1] / static_cast<double>(medium.inv_max_cmy[1]) - static_cast<double>(medium.min_cmy[1]);
            D_denorm2 = D_norm[2] / static_cast<double>(medium.inv_max_cmy[2]) - static_cast<double>(medium.min_cmy[2]);
        }
        else {
            D_denorm0 = D_norm[0] / static_cast<double>(medium.inv_max_cmy[0]);
            D_denorm1 = D_norm[1] / static_cast<double>(medium.inv_max_cmy[1]);
            D_denorm2 = D_norm[2] / static_cast<double>(medium.inv_max_cmy[2]);
        }

        double X = 0.0;
        double Y = 0.0;
        double Z = 0.0;
        for (int i = 0; i < medium.K; ++i) {
            const double baseSpectral = (medium.hasBaseline && medium.baseMin)
                ? static_cast<double>(ldg_f(medium.baseMin + i))
                : 0.0;
            const double Dlambda =
                D_denorm0 * static_cast<double>(ldg_f(medium.epsC + i)) +
                D_denorm1 * static_cast<double>(ldg_f(medium.epsM + i)) +
                D_denorm2 * static_cast<double>(ldg_f(medium.epsY + i)) +
                baseSpectral;

            const double transmittance = pow(10.0, -Dlambda);

            const double ax = static_cast<double>(ldg_f(medium.Ax + i));
            const double ay = static_cast<double>(ldg_f(medium.Ay + i));
            const double az = static_cast<double>(ldg_f(medium.Az + i));

            if (isfinite(ax)) {
                const double out = transmittance * ax;
                if (!isnan(out)) X += out;
            }
            if (isfinite(ay)) {
                const double out = transmittance * ay;
                if (!isnan(out)) Y += out;
            }
            if (isfinite(az)) {
                const double out = transmittance * az;
                if (!isnan(out)) Z += out;
            }
        }

        const double invNormalization = static_cast<double>(medium.invYn);
        const double XYZ0 = X * invNormalization;
        const double XYZ1 = Y * invNormalization;
        const double XYZ2 = Z * invNormalization;

        constexpr double kEps = 1e-10;
        logXYZ[0] = log10(XYZ0 + kEps);
        logXYZ[1] = log10(XYZ1 + kEps);
        logXYZ[2] = log10(XYZ2 + kEps);
    }

    __device__ __forceinline__ void convert_input_to_DWG_device(
        const JuicerCuda::FilmRawPayload& cfg,
        const float rgbIn[3],
        float rgbDWG[3])
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
        XYZ_to_DWG_linear_device(xyzPtr, dwg);
        rgbDWG[0] = dwg[0];
        rgbDWG[1] = dwg[1];
        rgbDWG[2] = dwg[2];
    }

    __device__ void hanatos_layer_exposures_device(
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
        XYZ[0] = device_sanitize_nonneg(XYZ[0]);
        XYZ[1] = device_sanitize_nonneg(XYZ[1]);
        XYZ[2] = device_sanitize_nonneg(XYZ[2]);

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
        chromatic_adapt_XYZ_CAT02_device(XYZ, D65, refWhite, adaptedXYZ);
        adaptedXYZ[0] = device_sanitize_nonneg(adaptedXYZ[0]);
        adaptedXYZ[1] = device_sanitize_nonneg(adaptedXYZ[1]);
        adaptedXYZ[2] = device_sanitize_nonneg(adaptedXYZ[2]);

        const float sumXYZ = adaptedXYZ[0] + adaptedXYZ[1] + adaptedXYZ[2];
        const float safeSum = (sumXYZ > 0.0f) ? sumXYZ : 0.0f;
        if (!(safeSum > 0.0f)) {
            E_out[0] = E_out[1] = E_out[2] = 0.0f;
            return;
        }

        float x = 1.0f / 3.0f;
        float y = 1.0f / 3.0f;
        if (sumXYZ > 1e-12f) {
            x = adaptedXYZ[0] / sumXYZ;
            y = adaptedXYZ[1] / sumXYZ;
        }
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

            const float e = fmaxf(0.0f, safeSum * raw);
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

        E_out[0] = fmaxf(0.0f, static_cast<float>(Eb));
        E_out[1] = fmaxf(0.0f, static_cast<float>(Eg));
        E_out[2] = fmaxf(0.0f, static_cast<float>(Er));
    }

    __device__ void hanatos_integrated_exposures_device(
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
        XYZ[0] = device_sanitize_nonneg(XYZ[0]);
        XYZ[1] = device_sanitize_nonneg(XYZ[1]);
        XYZ[2] = device_sanitize_nonneg(XYZ[2]);

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
        chromatic_adapt_XYZ_CAT02_device(XYZ, D65, refWhite, adaptedXYZ);
        adaptedXYZ[0] = device_sanitize_nonneg(adaptedXYZ[0]);
        adaptedXYZ[1] = device_sanitize_nonneg(adaptedXYZ[1]);
        adaptedXYZ[2] = device_sanitize_nonneg(adaptedXYZ[2]);

        const float sumXYZ = adaptedXYZ[0] + adaptedXYZ[1] + adaptedXYZ[2];
        const float safeSum = (sumXYZ > 0.0f) ? sumXYZ : 0.0f;
        if (!(safeSum > 0.0f)) {
            E_out[0] = E_out[1] = E_out[2] = 0.0f;
            return;
        }

        float x = 1.0f / 3.0f;
        float y = 1.0f / 3.0f;
        if (sumXYZ > 1e-12f) {
            x = adaptedXYZ[0] / sumXYZ;
            y = adaptedXYZ[1] / sumXYZ;
        }
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
        E_out[0] = fmaxf(0.0f, safeSum * bSafe);
        E_out[1] = fmaxf(0.0f, safeSum * gSafe);
        E_out[2] = fmaxf(0.0f, safeSum * rSafe);
    }

    __device__ void tables_layer_exposures_device(
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

    __device__ __forceinline__ float clamp_to_curve_domain_device(float logE, const JuicerCuda::DeviceCurveView& c) {
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

    __device__ __forceinline__ void apply_dir_runtime_logE_device(
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

    __device__ __forceinline__ void compute_dir_corrections_device(
        const JuicerCuda::DirPayload& dir,
        const float dYMC[3],
        float outYMC[3])
    {
        if (!outYMC) {
            return;
        }
        if (!dir.active) {
            outYMC[0] = 0.0f;
            outYMC[1] = 0.0f;
            outYMC[2] = 0.0f;
            return;
        }

        auto safe_norm = [](float D, float dmax) -> float {
            float Din = (!isfinite(D) || D < 0.0f) ? 0.0f : D;
            float m = (isfinite(dmax) && dmax > 1e-4f) ? dmax : 1.0f;
            float n = Din / m;
            if (!isfinite(n) || n < 0.0f) n = 0.0f;
            return n;
        };

        float nB = safe_norm(dYMC[0], dir.dMax[0]);
        float nG = safe_norm(dYMC[1], dir.dMax[1]);
        float nR = safe_norm(dYMC[2], dir.dMax[2]);

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
        outYMC[0] = clamp_corr(aY);
        outYMC[1] = clamp_corr(aM);
        outYMC[2] = clamp_corr(aC);
    }

    __device__ __forceinline__ void compute_logE_and_layer_pre_device(
        const JuicerCuda::Phase3RunParams& params,
        const float rgbIn[3],
        float logE_raw[3],
        float logE_sanitized[3],
        float layerPre[3])
    {
        float rgbDWG[3];
        convert_input_to_DWG_device(params.filmRaw, rgbIn, rgbDWG);

        float E_raw[3] = { 0.0f, 0.0f, 0.0f };
        const bool allowHanatos = (params.filmRaw.spectralUpsamplingMode == 0);
        const bool spdReady = params.tablesAx && params.tablesAy && params.tablesAz && params.tablesK == 81;
        const bool useHanatos = allowHanatos &&
            spdReady &&
            params.hanatosLut &&
            (params.hanatosN > 0) &&
            (params.sensB.n >= 81) &&
            (params.sensG.n >= 81) &&
            (params.sensR.n >= 81);
        const bool useHanatosIntegrated =
            useHanatos &&
            params.hanatosLutIntegrated &&
            (params.hanatosNIntegrated > 0) &&
            (params.hanatosNIntegrated == params.hanatosN);
        const bool canTables =
            spdReady &&
            params.sensB.y && params.sensG.y && params.sensR.y &&
            (params.sensB.n >= 81) &&
            (params.sensG.n >= 81) &&
            (params.sensR.n >= 81);

        if (useHanatosIntegrated) {
            hanatos_integrated_exposures_device(
                rgbDWG,
                params.hanatosLutIntegrated,
                params.hanatosNIntegrated,
                params.filmRaw.refIllumWhiteXYZ,
                E_raw);
        }
        else if (useHanatos) {
            hanatos_layer_exposures_device(
                rgbDWG,
                params.hanatosLut,
                params.hanatosN,
                params.filmRaw.refIllumWhiteXYZ,
                params.sensB.y,
                params.sensG.y,
                params.sensR.y,
                E_raw);
        }
        else if (canTables) {
            tables_layer_exposures_device(
                rgbDWG,
                params.spdSInv,
                params.filmRaw.refIllumWhiteXYZ,
                params.tablesAx,
                params.tablesAy,
                params.tablesAz,
                params.sensB.y,
                params.sensG.y,
                params.sensR.y,
                E_raw);
        }

        float midgrayScale = params.filmRaw.midgrayScale;
        if (!isfinite(midgrayScale) || !(midgrayScale > 0.0f)) {
            midgrayScale = 1.0f;
        }

        float exposureScale = params.exposureScale;
        if (!isfinite(exposureScale) || !(exposureScale > 0.0f)) {
            exposureScale = 1.0f;
        }

        float filmRaw[3];
        for (int i = 0; i < 3; ++i) {
            float v = E_raw[i];
            if (!isfinite(v) || v < 0.0f) v = 0.0f;
            v = fmaxf(0.0f, v * midgrayScale);
            v = fmaxf(0.0f, v * exposureScale);
            filmRaw[i] = v;
        }

        constexpr float kLogEps = 1e-10f;
        logE_raw[0] = log10f(fmaxf(filmRaw[0], 0.0f) + kLogEps);
        logE_raw[1] = log10f(fmaxf(filmRaw[1], 0.0f) + kLogEps);
        logE_raw[2] = log10f(fmaxf(filmRaw[2], 0.0f) + kLogEps);

        logE_sanitized[0] = sanitize_inf_logE_for_curve_device(logE_raw[0], params.densB.x, params.densB.n);
        logE_sanitized[1] = sanitize_inf_logE_for_curve_device(logE_raw[1], params.densG.x, params.densG.n);
        logE_sanitized[2] = sanitize_inf_logE_for_curve_device(logE_raw[2], params.densR.x, params.densR.n);

        layerPre[0] = sample_density_at_logE_device(params.densB.x, params.densB.y, params.densB.n, logE_sanitized[0], params.gammaFactorB);
        layerPre[1] = sample_density_at_logE_device(params.densG.x, params.densG.y, params.densG.n, logE_sanitized[1], params.gammaFactorG);
        layerPre[2] = sample_density_at_logE_device(params.densR.x, params.densR.y, params.densR.n, logE_sanitized[2], params.gammaFactorR);
    }

    __global__ void spatial_dir_corrections_kernel(
        JuicerCuda::Phase3RunParams params,
        float* corrY,
        float* corrM,
        float* corrC)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= params.width || y >= params.height) {
            return;
        }
        if (!params.src || params.srcRowBytes == 0) {
            return;
        }
        if (!corrY || !corrM || !corrC) {
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

        const float D_cmy[3] = { layerPre[2], layerPre[1], layerPre[0] };
        const float dYMC[3] = { D_cmy[2], D_cmy[1], D_cmy[0] };

        float outCorr[3] = { 0.0f, 0.0f, 0.0f };
        compute_dir_corrections_device(params.dir, dYMC, outCorr);

        const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
        corrY[idx] = outCorr[0];
        corrM[idx] = outCorr[1];
        corrC[idx] = outCorr[2];
    }

    __global__ void spatial_dir_clamp_kernel(float* corrY, float* corrM, float* corrC, int n) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) {
            return;
        }
        auto scrub = [](float v) -> float {
            if (!isfinite(v) || isnan(v)) {
                return 0.0f;
            }
            if (v < -10.0f) return -10.0f;
            if (v > 10.0f) return 10.0f;
            return v;
        };
        if (corrY) {
            corrY[idx] = scrub(corrY[idx]);
        }
        if (corrM) {
            corrM[idx] = scrub(corrM[idx]);
        }
        if (corrC) {
            corrC[idx] = scrub(corrC[idx]);
        }
    }

    __global__ void phase3_negative_only_kernel(JuicerCuda::Phase3RunParams params) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= params.width || y >= params.height) {
            return;
        }

        if (!params.src || !params.dst || params.srcRowBytes == 0 || params.dstRowBytes == 0) {
            return;
        }

        const int nC = params.nComponents;
        if (!(nC == 3 || nC == 4)) {
            return;
        }

        const std::size_t pixelBytes = static_cast<std::size_t>(nC) * sizeof(float);
        const char* srcRow = reinterpret_cast<const char*>(params.src) + static_cast<std::size_t>(y) * params.srcRowBytes;
        const float* srcPix = reinterpret_cast<const float*>(srcRow + static_cast<std::size_t>(x) * pixelBytes);

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

        apply_print_pipeline_device(params, D_cmy);

        // Scan: normalize density -> logXYZ
        double D_norm[3];
        if (params.scan.mediumIsNegative) {
            D_norm[0] = (static_cast<double>(D_cmy[0]) + static_cast<double>(params.scan.min_cmy[0])) * static_cast<double>(params.scan.inv_max_cmy[0]);
            D_norm[1] = (static_cast<double>(D_cmy[1]) + static_cast<double>(params.scan.min_cmy[1])) * static_cast<double>(params.scan.inv_max_cmy[1]);
            D_norm[2] = (static_cast<double>(D_cmy[2]) + static_cast<double>(params.scan.min_cmy[2])) * static_cast<double>(params.scan.inv_max_cmy[2]);
        }
        else {
            D_norm[0] = static_cast<double>(D_cmy[0]) * static_cast<double>(params.scan.inv_max_cmy[0]);
            D_norm[1] = static_cast<double>(D_cmy[1]) * static_cast<double>(params.scan.inv_max_cmy[1]);
            D_norm[2] = static_cast<double>(D_cmy[2]) * static_cast<double>(params.scan.inv_max_cmy[2]);
        }

        double logXYZ[3] = { 0.0, 0.0, 0.0 };
        const bool D_norm_finite = isfinite(D_norm[0]) && isfinite(D_norm[1]) && isfinite(D_norm[2]);
        if (params.scannerUseLut && params.scanLutLogXYZ && params.scanLutRes > 0 && D_norm_finite) {
            sample_cubic_scan_lut_device(params.scanLutLogXYZ, params.scanLutRes, D_norm, logXYZ);
        }
        else {
            scan_spectral_to_log_xyz_device(params.scan, D_norm, logXYZ);
        }
        const double xyz[3] = {
            pow(10.0, logXYZ[0]),
            pow(10.0, logXYZ[1]),
            pow(10.0, logXYZ[2])
        };

        double adapted[3];
        mat3_mul_vec_double_device(params.scanColor.cat02, xyz, adapted);
        double rgbOut[3];
        mat3_mul_vec_double_device(params.scanColor.xyzToRgb, adapted, rgbOut);

        if (!isfinite(rgbOut[0]) || !isfinite(rgbOut[1]) || !isfinite(rgbOut[2])) {
            signal_scan_error_device(params.scanErrorFlag);
            const std::size_t pixelBytes = static_cast<std::size_t>(nC) * sizeof(float);
            char* dstRow = reinterpret_cast<char*>(params.dst) + static_cast<std::size_t>(y) * params.dstRowBytes;
            float* dstPix = reinterpret_cast<float*>(dstRow + static_cast<std::size_t>(x) * pixelBytes);
            if (dstPix) {
                dstPix[0] = 0.0f;
                dstPix[1] = 0.0f;
                dstPix[2] = 0.0f;
                if (nC == 4) {
                    const char* srcRow = reinterpret_cast<const char*>(params.src) + static_cast<std::size_t>(y) * params.srcRowBytes;
                    const float* srcPix = reinterpret_cast<const float*>(srcRow + static_cast<std::size_t>(x) * pixelBytes);
                    dstPix[3] = srcPix ? srcPix[3] : 1.0f;
                }
            }
            return;
        }

        // Output encoding + clamp
        apply_output_encoding_device(params.scanColor.encoding, rgbOut);

        char* dstRow = reinterpret_cast<char*>(params.dst) + static_cast<std::size_t>(y) * params.dstRowBytes;
        float* dstPix = reinterpret_cast<float*>(dstRow + static_cast<std::size_t>(x) * pixelBytes);
        dstPix[0] = static_cast<float>(rgbOut[0]);
        dstPix[1] = static_cast<float>(rgbOut[1]);
        dstPix[2] = static_cast<float>(rgbOut[2]);
        if (nC == 4) {
            dstPix[3] = srcPix[3];
        }
    }

    __global__ void phase3_stageA_linear_rgb_kernel(
        JuicerCuda::Phase3RunParams params,
        float* rgbR,
        float* rgbG,
        float* rgbB,
        const float* glarePercent)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= params.width || y >= params.height) {
            return;
        }

        if (!params.src || params.srcRowBytes == 0) {
            return;
        }
        if (!rgbR || !rgbG || !rgbB) {
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

        apply_print_pipeline_device(params, D_cmy);

        // Scan: normalize density -> logXYZ
        double D_norm[3];
        if (params.scan.mediumIsNegative) {
            D_norm[0] = (static_cast<double>(D_cmy[0]) + static_cast<double>(params.scan.min_cmy[0])) * static_cast<double>(params.scan.inv_max_cmy[0]);
            D_norm[1] = (static_cast<double>(D_cmy[1]) + static_cast<double>(params.scan.min_cmy[1])) * static_cast<double>(params.scan.inv_max_cmy[1]);
            D_norm[2] = (static_cast<double>(D_cmy[2]) + static_cast<double>(params.scan.min_cmy[2])) * static_cast<double>(params.scan.inv_max_cmy[2]);
        }
        else {
            D_norm[0] = static_cast<double>(D_cmy[0]) * static_cast<double>(params.scan.inv_max_cmy[0]);
            D_norm[1] = static_cast<double>(D_cmy[1]) * static_cast<double>(params.scan.inv_max_cmy[1]);
            D_norm[2] = static_cast<double>(D_cmy[2]) * static_cast<double>(params.scan.inv_max_cmy[2]);
        }

        double logXYZ[3] = { 0.0, 0.0, 0.0 };
        const bool D_norm_finite = isfinite(D_norm[0]) && isfinite(D_norm[1]) && isfinite(D_norm[2]);
        if (params.scannerUseLut && params.scanLutLogXYZ && params.scanLutRes > 0 && D_norm_finite) {
            sample_cubic_scan_lut_device(params.scanLutLogXYZ, params.scanLutRes, D_norm, logXYZ);
        }
        else {
            scan_spectral_to_log_xyz_device(params.scan, D_norm, logXYZ);
        }

        const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);

        double xyz[3] = {
            pow(10.0, logXYZ[0]),
            pow(10.0, logXYZ[1]),
            pow(10.0, logXYZ[2])
        };

        if (glarePercent) {
            const double glare = static_cast<double>(glarePercent[idx]) * 0.01;
            xyz[0] += glare * static_cast<double>(params.scanColor.illuminantXYZ[0]);
            xyz[1] += glare * static_cast<double>(params.scanColor.illuminantXYZ[1]);
            xyz[2] += glare * static_cast<double>(params.scanColor.illuminantXYZ[2]);
        }

        double adapted[3];
        mat3_mul_vec_double_device(params.scanColor.cat02, xyz, adapted);
        double rgbOut[3];
        mat3_mul_vec_double_device(params.scanColor.xyzToRgb, adapted, rgbOut);

        if (!isfinite(rgbOut[0]) || !isfinite(rgbOut[1]) || !isfinite(rgbOut[2])) {
            signal_scan_error_device(params.scanErrorFlag);
            rgbR[idx] = 0.0f;
            rgbG[idx] = 0.0f;
            rgbB[idx] = 0.0f;
            return;
        }

        rgbR[idx] = static_cast<float>(rgbOut[0]);
        rgbG[idx] = static_cast<float>(rgbOut[1]);
        rgbB[idx] = static_cast<float>(rgbOut[2]);
    }

    __global__ void phase3_stageD_encode_write_kernel(
        JuicerCuda::Phase3RunParams params,
        const float* rgbR,
        const float* rgbG,
        const float* rgbB)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= params.width || y >= params.height) {
            return;
        }

        if (!params.src || !params.dst || params.srcRowBytes == 0 || params.dstRowBytes == 0) {
            return;
        }
        if (!rgbR || !rgbG || !rgbB) {
            return;
        }

        const int nC = params.nComponents;
        if (!(nC == 3 || nC == 4)) {
            return;
        }

        const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
        double rgbOut[3] = {
            static_cast<double>(rgbR[idx]),
            static_cast<double>(rgbG[idx]),
            static_cast<double>(rgbB[idx])
        };
        apply_output_encoding_device(params.scanColor.encoding, rgbOut);

        const std::size_t pixelBytes = static_cast<std::size_t>(nC) * sizeof(float);
        char* dstRow = reinterpret_cast<char*>(params.dst) + static_cast<std::size_t>(y) * params.dstRowBytes;
        float* dstPix = reinterpret_cast<float*>(dstRow + static_cast<std::size_t>(x) * pixelBytes);
        dstPix[0] = static_cast<float>(rgbOut[0]);
        dstPix[1] = static_cast<float>(rgbOut[1]);
        dstPix[2] = static_cast<float>(rgbOut[2]);

        if (nC == 4) {
            const char* srcRow = reinterpret_cast<const char*>(params.src) + static_cast<std::size_t>(y) * params.srcRowBytes;
            const float* srcPix = reinterpret_cast<const float*>(srcRow + static_cast<std::size_t>(x) * pixelBytes);
            dstPix[3] = srcPix ? srcPix[3] : 1.0f;
        }
    }

} // namespace

extern "C" cudaError_t juicer_cuda_phase3_negative_only(
    const JuicerCuda::Phase3RunParams* hParams,
    void* cudaStreamOpaque)
{
    if (!hParams) {
        return cudaErrorInvalidValue;
    }

    const JuicerCuda::Phase3RunParams params = *hParams;
    if (!params.src || !params.dst) {
        return cudaErrorInvalidValue;
    }
    if (params.width <= 0 || params.height <= 0) {
        return cudaSuccess;
    }
    if (!(params.nComponents == 3 || params.nComponents == 4)) {
        return cudaErrorInvalidValue;
    }
    if (params.srcRowBytes == 0 || params.dstRowBytes == 0) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

    dim3 threads(32, 8);
    dim3 blocks(
        static_cast<unsigned int>((params.width + threads.x - 1) / threads.x),
        static_cast<unsigned int>((params.height + threads.y - 1) / threads.y));
    phase3_negative_only_kernel<<<blocks, threads, 0, stream>>>(params);
    return cudaGetLastError();
}

extern "C" cudaError_t juicer_cuda_build_spatial_dir(
    const JuicerCuda::Phase3RunParams* hParams,
    float* dCorrY,
    float* dCorrM,
    float* dCorrC,
    float* dTmp,
    const float* dKernel,
    int radius,
    void* cudaStreamOpaque)
{
    if (!hParams) {
        return cudaErrorInvalidValue;
    }

    const JuicerCuda::Phase3RunParams params = *hParams;
    if (!params.src || params.srcRowBytes == 0) {
        return cudaErrorInvalidValue;
    }
    if (params.width <= 0 || params.height <= 0) {
        return cudaSuccess;
    }
    if (!(params.nComponents == 3 || params.nComponents == 4)) {
        return cudaErrorInvalidValue;
    }
    if (!dCorrY || !dCorrM || !dCorrC || !dTmp) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

    dim3 threads2D(32, 8);
    dim3 blocks2D(
        static_cast<unsigned int>((params.width + threads2D.x - 1) / threads2D.x),
        static_cast<unsigned int>((params.height + threads2D.y - 1) / threads2D.y));

    spatial_dir_corrections_kernel<<<blocks2D, threads2D, 0, stream>>>(params, dCorrY, dCorrM, dCorrC);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }

    auto blur_plane_in_place = [&](float* plane, float* tmpBuf, const float* k, int r) -> cudaError_t {
        if (!plane || !tmpBuf || !k || r <= 0) {
            return cudaSuccess;
        }
        optics_blur_horizontal_kernel<<<blocks2D, threads2D, 0, stream>>>(plane, tmpBuf, params.width, params.height, k, r);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            return e;
        }
        optics_blur_vertical_kernel<<<blocks2D, threads2D, 0, stream>>>(tmpBuf, plane, params.width, params.height, k, r);
        return cudaGetLastError();
    };

    if (radius > 0 && dKernel) {
        err = blur_plane_in_place(dCorrY, dTmp, dKernel, radius);
        if (err != cudaSuccess) return err;
        err = blur_plane_in_place(dCorrM, dTmp, dKernel, radius);
        if (err != cudaSuccess) return err;
        err = blur_plane_in_place(dCorrC, dTmp, dKernel, radius);
        if (err != cudaSuccess) return err;
    }

    const int total = params.width * params.height;
    const int threads1D = 256;
    const int blocks1D = (total + threads1D - 1) / threads1D;
    spatial_dir_clamp_kernel<<<blocks1D, threads1D, 0, stream>>>(dCorrY, dCorrM, dCorrC, total);
    return cudaGetLastError();
}

extern "C" cudaError_t juicer_cuda_phase3_negative_only_optics(
    const JuicerCuda::Phase3RunParams* hParams,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    float* dScratchBlurred,
    const float* dLensBlurKernel,
    int lensBlurRadius,
    const float* dUnsharpKernel,
    int unsharpRadius,
    float unsharpAmount,
    int glareOriginX,
    int glareOriginY,
    std::uint64_t glareSeed,
    float glarePercent,
    float glareRoughness,
    const float* dGlareKernel,
    int glareRadius,
    void* cudaStreamOpaque)
{
    if (!hParams) {
        return cudaErrorInvalidValue;
    }

    const JuicerCuda::Phase3RunParams params = *hParams;
    if (!params.src || !params.dst) {
        return cudaErrorInvalidValue;
    }
    if (params.width <= 0 || params.height <= 0) {
        return cudaSuccess;
    }
    if (!(params.nComponents == 3 || params.nComponents == 4)) {
        return cudaErrorInvalidValue;
    }
    if (params.srcRowBytes == 0 || params.dstRowBytes == 0) {
        return cudaErrorInvalidValue;
    }
    if (!dRgbR || !dRgbG || !dRgbB || !dTmp) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

    dim3 threads2D(32, 8);
    dim3 blocks2D(
        static_cast<unsigned int>((params.width + threads2D.x - 1) / threads2D.x),
        static_cast<unsigned int>((params.height + threads2D.y - 1) / threads2D.y));

    auto blur_plane_in_place = [&](float* plane, float* tmpBuf, const float* k, int radius) -> cudaError_t {
        if (!plane || !tmpBuf || !k || radius <= 0) {
            return cudaSuccess;
        }
        optics_blur_horizontal_kernel<<<blocks2D, threads2D, 0, stream>>>(plane, tmpBuf, params.width, params.height, k, radius);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            return e;
        }
        optics_blur_vertical_kernel<<<blocks2D, threads2D, 0, stream>>>(tmpBuf, plane, params.width, params.height, k, radius);
        return cudaGetLastError();
    };

    const bool doGlare =
        std::isfinite(static_cast<double>(glarePercent)) && (glarePercent > 0.0f) &&
        std::isfinite(static_cast<double>(glareRoughness));
    if (doGlare) {
        const std::uint64_t mediumId = params.scan.mediumIsNegative ? 0ULL : 1ULL;
        optics_glare_generate_kernel<<<blocks2D, threads2D, 0, stream>>>(
            dTmp,
            params.width,
            params.height,
            glareSeed,
            mediumId,
            glareOriginX,
            glareOriginY,
            glarePercent,
            glareRoughness);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }
        if (glareRadius > 0 && dGlareKernel) {
            err = blur_plane_in_place(dTmp, dRgbR, dGlareKernel, glareRadius);
            if (err != cudaSuccess) return err;
        }
    }

    phase3_stageA_linear_rgb_kernel<<<blocks2D, threads2D, 0, stream>>>(params, dRgbR, dRgbG, dRgbB, doGlare ? dTmp : nullptr);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }

    if (lensBlurRadius > 0 && dLensBlurKernel) {
        err = blur_plane_in_place(dRgbR, dTmp, dLensBlurKernel, lensBlurRadius);
        if (err != cudaSuccess) return err;
        err = blur_plane_in_place(dRgbG, dTmp, dLensBlurKernel, lensBlurRadius);
        if (err != cudaSuccess) return err;
        err = blur_plane_in_place(dRgbB, dTmp, dLensBlurKernel, lensBlurRadius);
        if (err != cudaSuccess) return err;
    }

    const bool doUnsharp =
        (unsharpRadius > 0) && dUnsharpKernel &&
        std::isfinite(static_cast<double>(unsharpAmount)) && (unsharpAmount != 0.0f);
    if (doUnsharp) {
        if (!dScratchBlurred) {
            return cudaErrorInvalidValue;
        }

        const int total = params.width * params.height;
        const int threads1D = 256;
        const int blocks1D = (total + threads1D - 1) / threads1D;

        auto unsharp_plane_in_place = [&](float* plane) -> cudaError_t {
            optics_blur_horizontal_kernel<<<blocks2D, threads2D, 0, stream>>>(plane, dTmp, params.width, params.height, dUnsharpKernel, unsharpRadius);
            cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) {
                return e;
            }
            optics_blur_vertical_kernel<<<blocks2D, threads2D, 0, stream>>>(dTmp, dScratchBlurred, params.width, params.height, dUnsharpKernel, unsharpRadius);
            e = cudaGetLastError();
            if (e != cudaSuccess) {
                return e;
            }
            optics_unsharp_combine_kernel<<<blocks1D, threads1D, 0, stream>>>(plane, dScratchBlurred, total, unsharpAmount);
            return cudaGetLastError();
        };

        err = unsharp_plane_in_place(dRgbR);
        if (err != cudaSuccess) return err;
        err = unsharp_plane_in_place(dRgbG);
        if (err != cudaSuccess) return err;
        err = unsharp_plane_in_place(dRgbB);
        if (err != cudaSuccess) return err;
    }

    phase3_stageD_encode_write_kernel<<<blocks2D, threads2D, 0, stream>>>(params, dRgbR, dRgbG, dRgbB);
    return cudaGetLastError();
}

extern "C" cudaError_t juicer_cuda_phase5_print_pipeline(
    const JuicerCuda::Phase3RunParams* hParams,
    void* cudaStreamOpaque)
{
    if (!hParams) {
        return cudaErrorInvalidValue;
    }
    if (!hParams->printActive) {
        return cudaErrorInvalidValue;
    }
    return juicer_cuda_phase3_negative_only(hParams, cudaStreamOpaque);
}

extern "C" cudaError_t juicer_cuda_phase5_print_pipeline_optics(
    const JuicerCuda::Phase3RunParams* hParams,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    float* dScratchBlurred,
    const float* dLensBlurKernel,
    int lensBlurRadius,
    const float* dUnsharpKernel,
    int unsharpRadius,
    float unsharpAmount,
    int glareOriginX,
    int glareOriginY,
    std::uint64_t glareSeed,
    float glarePercent,
    float glareRoughness,
    const float* dGlareKernel,
    int glareRadius,
    void* cudaStreamOpaque)
{
    if (!hParams) {
        return cudaErrorInvalidValue;
    }
    if (!hParams->printActive) {
        return cudaErrorInvalidValue;
    }
    return juicer_cuda_phase3_negative_only_optics(
        hParams,
        dRgbR,
        dRgbG,
        dRgbB,
        dTmp,
        dScratchBlurred,
        dLensBlurKernel,
        lensBlurRadius,
        dUnsharpKernel,
        unsharpRadius,
        unsharpAmount,
        glareOriginX,
        glareOriginY,
        glareSeed,
        glarePercent,
        glareRoughness,
        dGlareKernel,
        glareRadius,
        cudaStreamOpaque);
}
