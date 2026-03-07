// SpectralProcessing.h
// Spectral processing operations: SPD reconstruction, table operations, integration, and math utilities

#pragma once

#include <cmath>
#include <algorithm>
#include <vector>
#include <array>
#include <mutex>
#include <atomic>
#include <cassert>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#if defined(JUICER_SPD_DEBUG) && (JUICER_SPD_DEBUG != 0)
#include <sstream>
#endif
#include "SpectralData.h"
#include "SpectralContext.h"
#include "NpyLoader.h"
#include "AkimaInterpolator.h"

namespace Spectral {

    // =========================================================================
    // Additional global references to SpectralContext (SpectralData.h has the base ones)
    // =========================================================================

    inline float& gDeltaLambda = context().deltaLambda;

    inline std::vector<float>& gEpsYTable = context().epsYTable;
    inline std::vector<float>& gEpsMTable = context().epsMTable;
    inline std::vector<float>& gEpsCTable = context().epsCTable;
    inline std::vector<float>& gXbarTable = context().xbarTable;
    inline std::vector<float>& gYbarTable = context().ybarTable;
    inline std::vector<float>& gZbarTable = context().zbarTable;
    inline std::vector<float>& gBaselineMinTable = context().baselineMinTable;
    inline std::vector<float>& gBaselineMidTable = context().baselineMidTable;
    inline std::vector<float>& gIllumTable = context().illumTable;

    inline std::vector<float>& gAx = context().Ax;
    inline std::vector<float>& gAy = context().Ay;
    inline std::vector<float>& gAz = context().Az;
    inline float& gYnNorm = context().ynNorm;
    inline float& gInvYn = context().invYn;

    inline PrecomputeStatus& gPrecomputeStatus = context().precomputeStatus;
    inline std::mutex& gPrecomputeMutex = context().precomputeMutex;

    inline constexpr float kLn10 = 2.302585092994046f;

    inline std::vector<float>& gLambda = context().lambda;

    inline Curve& gDensityCurveB = context().densityCurveB;
    inline Curve& gDensityCurveG = context().densityCurveG;
    inline Curve& gDensityCurveR = context().densityCurveR;

#if defined(JUICER_SPD_DEBUG) && (JUICER_SPD_DEBUG != 0)
#ifndef JUICER_SPD_DEBUG_MAX_SAMPLES
#define JUICER_SPD_DEBUG_MAX_SAMPLES 1
#endif

    struct SpdProbeBuffer {
        bool active = false;
        int sampleIndex = -1;
        float rgbInput[3] = { 0.0f, 0.0f, 0.0f };
        float rgbDWG[3] = { 0.0f, 0.0f, 0.0f };
        float qx = 0.0f;
        float qy = 0.0f;
        bool hasCoords = false;
        std::vector<float> rawLUT;
        float targetScale = 0.0f;
        bool hasTargetScale = false;
        float Y_recon = 0.0f;
        bool hasYrecon = false;
        float midgrayScale = 0.0f;
        bool hasMidgray = false;
        float cat02InputXYZ[3] = { 0.0f, 0.0f, 0.0f };
        float cat02OutputXYZ[3] = { 0.0f, 0.0f, 0.0f };
        float cat02SrcWhite[3] = { 0.0f, 0.0f, 0.0f };
        float cat02DstWhite[3] = { 0.0f, 0.0f, 0.0f };
        bool hasCat02 = false;
        bool applyDeltaLambdaFlag = false;
        bool hasDeltaLambda = false;
        std::vector<float> Ee;
    };

    inline std::atomic<int>& spd_probe_emitted_samples() {
        static std::atomic<int> emitted{ 0 };
        return emitted;
    }

    inline SpdProbeBuffer& spd_probe_buffer() {
        thread_local SpdProbeBuffer buffer;
        return buffer;
    }

    inline void spd_probe_reset() {
        spd_probe_emitted_samples().store(0, std::memory_order_release);
        spd_probe_buffer().active = false;
    }

    inline void copy_triplet_or_zero(const float* src, float dst[3]) {
        float* dstIt = dst;
        const float* srcIt = src;
        for (int i = 0; i < 3; ++i, ++dstIt) {
            *dstIt = srcIt ? *srcIt++ : 0.0f;
        }
    }

    inline bool spd_probe_begin_capture(const float rgbIn[3], const float rgbDWG[3], bool spdEnabled) {
        if (!spdEnabled) {
            return false;
        }

        auto& emitted = spd_probe_emitted_samples();
        int current = emitted.load(std::memory_order_relaxed);
        while (true) {
            if (current >= JUICER_SPD_DEBUG_MAX_SAMPLES) {
                return false;
            }
            if (emitted.compare_exchange_weak(current, current + 1, std::memory_order_acq_rel)) {
                break;
            }
        }

        const int sampleIndex = current;

        auto& buf = spd_probe_buffer();
        buf.active = true;
        buf.sampleIndex = sampleIndex;
        copy_triplet_or_zero(rgbIn, buf.rgbInput);
        copy_triplet_or_zero(rgbDWG, buf.rgbDWG);
        buf.targetScale = 0.0f;
        buf.hasTargetScale = false;
        buf.midgrayScale = 0.0f;
        buf.hasMidgray = false;
        buf.hasCat02 = false;
        buf.hasDeltaLambda = false;
        buf.Ee.clear();
        return true;
    }

    inline bool spd_probe_active() {
        return spd_probe_buffer().active;
    }

    inline void spd_probe_record_target_scale(float targetScale) {
        auto& buf = spd_probe_buffer();
        if (!buf.active) {
            return;
        }
        buf.targetScale = targetScale;
        buf.hasTargetScale = true;
    }

    inline void spd_probe_record_yrecon(double yrecon) {
        auto& buf = spd_probe_buffer();
        if (!buf.active) {
            return;
        }
        buf.Y_recon = static_cast<float>(yrecon);
        buf.hasYrecon = true;
    }

    inline void spd_probe_record_coords(float qx, float qy) {
        auto& buf = spd_probe_buffer();
        if (!buf.active) {
            return;
        }
        buf.qx = qx;
        buf.qy = qy;
        buf.hasCoords = true;
    }

    inline void spd_probe_record_raw_lut(const std::vector<float>& rawLUT) {
        auto& buf = spd_probe_buffer();
        if (!buf.active) {
            return;
        }
        buf.rawLUT = rawLUT;
    }

    inline void spd_probe_record_spectrum(const std::vector<float>& Ee) {
        auto& buf = spd_probe_buffer();
        if (!buf.active) {
            return;
        }
        buf.Ee = Ee;
    }

    inline void spd_probe_record_cat02(
        const float XYZ_in[3],
        const float XYZ_out[3],
        const float srcWhite[3],
        const float dstWhite[3])
    {
        auto& buf = spd_probe_buffer();
        if (!buf.active) {
            return;
        }
        copy_triplet_or_zero(XYZ_in, buf.cat02InputXYZ);
        copy_triplet_or_zero(XYZ_out, buf.cat02OutputXYZ);
        copy_triplet_or_zero(srcWhite, buf.cat02SrcWhite);
        copy_triplet_or_zero(dstWhite, buf.cat02DstWhite);
        buf.hasCat02 = true;
    }

    inline void spd_probe_record_delta_lambda(bool applyDeltaLambda) {
        auto& buf = spd_probe_buffer();
        if (!buf.active) {
            return;
        }
        buf.applyDeltaLambdaFlag = applyDeltaLambda;
        buf.hasDeltaLambda = true;
    }

    inline void spd_probe_finalize(float midgrayScale, const float E_afterMidgray[3]) {
        auto& buf = spd_probe_buffer();
        if (!buf.active) {
            return;
        }
        buf.midgrayScale = midgrayScale;
        buf.hasMidgray = true;

        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss.precision(6);
        oss << "sample=" << buf.sampleIndex;
        oss << " rgbIn=[" << buf.rgbInput[0] << "," << buf.rgbInput[1] << "," << buf.rgbInput[2] << "]";
        oss << " rgbDWG=[" << buf.rgbDWG[0] << "," << buf.rgbDWG[1] << "," << buf.rgbDWG[2] << "]";
        if (buf.hasCoords) {
            oss << " qx=" << buf.qx << " qy=" << buf.qy;
        }
        if (buf.hasTargetScale) {
            oss << " targetScale=" << buf.targetScale;
        }
        else {
            oss << " targetScale=<unset>";
        }
        if (buf.hasYrecon) {
            oss << " Y_recon=" << buf.Y_recon;
        }
        if (buf.hasMidgray) {
            oss << " midgrayScale=" << buf.midgrayScale;
        }
        if (E_afterMidgray) {
            oss << " E_midgray=[" << E_afterMidgray[0] << "," << E_afterMidgray[1] << "," << E_afterMidgray[2] << "]";
        }
        if (buf.hasCat02) {
            oss << " cat02XYZIn=[" << buf.cat02InputXYZ[0] << "," << buf.cat02InputXYZ[1] << "," << buf.cat02InputXYZ[2] << "]";
            oss << " cat02XYZOut=[" << buf.cat02OutputXYZ[0] << "," << buf.cat02OutputXYZ[1] << "," << buf.cat02OutputXYZ[2] << "]";
            oss << " cat02SrcWhite=[" << buf.cat02SrcWhite[0] << "," << buf.cat02SrcWhite[1] << "," << buf.cat02SrcWhite[2] << "]";
            oss << " cat02DstWhite=[" << buf.cat02DstWhite[0] << "," << buf.cat02DstWhite[1] << "," << buf.cat02DstWhite[2] << "]";
        }
        if (buf.hasDeltaLambda) {
            oss << " applyDeltaLambda=" << (buf.applyDeltaLambdaFlag ? 1 : 0);
        }
        if (!buf.rawLUT.empty()) {
            oss << " rawLUT=[";
            const size_t rawCount = buf.rawLUT.size();
            const float* rawData = buf.rawLUT.data();
            bool first = true;
            for (size_t i = 0; i < rawCount; ++i, ++rawData) {
                if (!first) {
                    oss << ",";
                }
                first = false;
                oss << *rawData;
            }
            oss << "]";
        }
        oss << " Ee=[";
        const size_t eeCount = buf.Ee.size();
        const float* eeData = buf.Ee.data();
        bool firstEe = true;
        for (size_t i = 0; i < eeCount; ++i, ++eeData) {
            if (!firstEe) {
                oss << ",";
            }
            firstEe = false;
            oss << *eeData;
        }
        oss << "]";
        JTRACE("SPDDBG", oss.str());
        buf.active = false;
    }
#else
    inline bool spd_probe_begin_capture(const float[3], const float[3], bool) { return false; }
    inline bool spd_probe_active() { return false; }
    inline void spd_probe_record_target_scale(float) {}
    inline void spd_probe_record_yrecon(double) {}
    inline void spd_probe_record_coords(float, float) {}
    inline void spd_probe_record_raw_lut(const std::vector<float>&) {}
    inline void spd_probe_record_spectrum(const std::vector<float>&) {}
    inline void spd_probe_record_cat02(const float[3], const float[3], const float[3], const float[3]) {}
    inline void spd_probe_record_delta_lambda(bool) {}
    inline void spd_probe_finalize(float, const float[3]) {}
    inline void spd_probe_reset() {}
#endif

    inline bool is_finite_sp(float value) {
        return std::isfinite(value);
    }

    inline bool is_finite_sp(double value) {
        return std::isfinite(value);
    }

    inline float sanitize_nonnegative_component(float value) {
        return (is_finite_sp(value) && value > 0.0f) ? value : 0.0f;
    }

    inline float sanitize_nonfinite_component(float value) {
        return is_finite_sp(value) ? value : 0.0f;
    }

    inline void sanitize_nonfinite_triplet(float values[3]) {
        float* valueIt = values;
        for (int i = 0; i < 3; ++i, ++valueIt) {
            *valueIt = sanitize_nonfinite_component(*valueIt);
        }
    }

    inline void sanitize_nonnegative_triplet_sp(float dst[3], const float src[3]) {
        float* dstIt = dst;
        const float* srcIt = src;
        for (int i = 0; i < 3; ++i, ++dstIt, ++srcIt) {
            *dstIt = sanitize_nonnegative_component(*srcIt);
        }
    }

    inline std::uint64_t hash_float_span_digest_sp(const float* values, std::size_t count) {
        const Hash::FloatSpanHash h = Hash::hash_float_span_with_nan_mask(values, count);
        const std::uint64_t fields[] = { h.valueHash, h.nanMaskHash };
        return Hash::hash_bytes(fields, sizeof(fields));
    }

    inline std::uint64_t hash_float_vector_digest_sp(const std::vector<float>& values) {
        return hash_float_span_digest_sp(values.data(), values.size());
    }

    inline std::uint64_t hash_float_scalar_digest_sp(float value) {
        return hash_float_span_digest_sp(&value, 1);
    }

    inline std::uint64_t hash_float_triplet_digest_sp(const float values[3]) {
        return hash_float_span_digest_sp(values, 3);
    }

    inline float sanitize_signed_width(float width, float minMagnitude = 1e-6f) {
        float safeWidth = width;
        if (!is_finite_sp(safeWidth)) {
            safeWidth = (safeWidth < 0.0f) ? -minMagnitude : minMagnitude;
        }
        if (std::fabs(safeWidth) < minMagnitude) {
            safeWidth = (safeWidth < 0.0f) ? -minMagnitude : minMagnitude;
        }
        return safeWidth;
    }

    inline void copy_triplet3(const float src[3], float dst[3]) {
        float* dstIt = dst;
        const float* srcIt = src;
        for (int i = 0; i < 3; ++i, ++dstIt, ++srcIt) {
            *dstIt = *srcIt;
        }
    }

    inline void clamp_triplet_nonnegative(float values[3]) {
        float* valueIt = values;
        for (int i = 0; i < 3; ++i, ++valueIt) {
            *valueIt = std::max(0.0f, *valueIt);
        }
    }

    inline void sanitize_ref_white_or_dwg(const float* white, float dst[3]) {
        const float fallback[3] = {
            gDWG_WhitePoint_XYZ[0],
            gDWG_WhitePoint_XYZ[1],
            gDWG_WhitePoint_XYZ[2]
        };
        const float* src = white ? white : fallback;
        float* dstIt = dst;
        const float* srcIt = src;
        const float* fallbackIt = fallback;
        for (int i = 0; i < 3; ++i, ++dstIt, ++srcIt, ++fallbackIt) {
            *dstIt = is_finite_sp(*srcIt) ? *srcIt : *fallbackIt;
        }
        if (!(dst[1] > 0.0f)) {
            copy_triplet3(fallback, dst);
        }
    }

    inline void set_identity_3x3(float matrix[9]) {
        std::fill_n(matrix, 9, 0.0f);
        matrix[0] = 1.0f;
        matrix[4] = 1.0f;
        matrix[8] = 1.0f;
    }

    inline void store_scaled_xyz(double X, double Y, double Z, float scale, float XYZ[3]) {
        XYZ[0] = static_cast<float>(X * scale);
        XYZ[1] = static_cast<float>(Y * scale);
        XYZ[2] = static_cast<float>(Z * scale);
    }

    inline void store_3x3_rowmajor(
        float matrix[9],
        double m00, double m01, double m02,
        double m10, double m11, double m12,
        double m20, double m21, double m22)
    {
        matrix[0] = static_cast<float>(m00);
        matrix[1] = static_cast<float>(m01);
        matrix[2] = static_cast<float>(m02);
        matrix[3] = static_cast<float>(m10);
        matrix[4] = static_cast<float>(m11);
        matrix[5] = static_cast<float>(m12);
        matrix[6] = static_cast<float>(m20);
        matrix[7] = static_cast<float>(m21);
        matrix[8] = static_cast<float>(m22);
    }

    inline bool determinant_near_zero(double determinant, double epsilon = 1e-20) {
        return std::fabs(determinant) < epsilon;
    }

    // -------------------------------------------------------------------------
    // 1. MATH UTILITIES (~100 lines)
    // -------------------------------------------------------------------------

    inline float sigmoid_erf(float x, float center, float width) {
        const float w = sanitize_signed_width(width);
        const float arg = (x - center) / w;
        const float val = static_cast<float>(std::erf(static_cast<double>(arg)));
        return val * 0.5f + 0.5f;
    }

    inline float gaussian(float x, float mu, float sigma) {
        const float t = (x - mu) / sigma;
        return std::exp(-0.5f * t * t);
    }

    inline std::vector<float> compute_band_pass_filter(
        const std::array<float, 3>& filterUV = { {1.0f, 410.0f, 8.0f} },
        const std::array<float, 3>& filterIR = { {1.0f, 675.0f, 15.0f} })
    {
        std::vector<float> bandPass;
        const int K = gShape.K;
        if (K <= 0 || gShape.wavelengths.size() != static_cast<size_t>(K)) {
            return bandPass;
        }

        bandPass.resize(static_cast<size_t>(K));

        const float ampUV = std::clamp(filterUV[0], 0.0f, 1.0f);
        const float ampIR = std::clamp(filterIR[0], 0.0f, 1.0f);
        const float wlUV = filterUV[1];
        const float wlIR = filterIR[1];
        if (ampUV <= 0.0f && ampIR <= 0.0f) {
            std::fill(bandPass.begin(), bandPass.end(), 1.0f);
            return bandPass;
        }

        const float widthUV = sanitize_signed_width(filterUV[2]);
        const float widthIR = -std::fabs(sanitize_signed_width(filterIR[2]));
        const float* wavelengths = gShape.wavelengths.data();
        float* outData = bandPass.data();

        for (int i = 0; i < K; ++i) {
            const float wl = wavelengths[i];
            const float filter_uv = 1.0f - ampUV + ampUV * sigmoid_erf(wl, wlUV, widthUV);
            const float filter_ir = 1.0f - ampIR + ampIR * sigmoid_erf(wl, wlIR, widthIR);
            outData[i] = filter_uv * filter_ir;
        }

        return bandPass;
    }

    inline float compute_delta_from_shape(const SpectralShape& s) {
        if (s.K <= 1 || s.wavelengths.size() < 2) return kDelta;
        // Estimate mean Δλ to be robust to tiny non-uniformities
        const float* wavelengths = s.wavelengths.data();
        double sum = 0.0;
        for (int i = 1; i < s.K; ++i) {
            sum += static_cast<double>(wavelengths[i] - wavelengths[i - 1]);
        }
        const double mean = sum / static_cast<double>(s.K - 1);
        return (mean > 0.0) ? static_cast<float>(mean) : kDelta;
    }

    inline float illuminant_E(float /*lambda*/) { return 1.0f; }

    inline void fill_viewing_illuminant_Ee(float gain, std::vector<float>& Ee_out) {
        const int K = gShape.K;
        Ee_out.resize(K);
        const float* illumData = gIllumTable.data();
        float* outData = Ee_out.data();
        for (int i = 0; i < K; ++i) {
            outData[i] = std::max(0.0f, gain * illumData[i]);
        }
    }

    // -------------------------------------------------------------------------
    // 2. SPD RECONSTRUCTION (~400 lines)
    // -------------------------------------------------------------------------

    // Forward declarations for chromatic adaptation
    inline void chromatic_adapt_XYZ_CAT02(
        const float XYZ[3],
        const float srcWhiteXYZ[3],
        const float dstWhiteXYZ[3],
        float outXYZ[3]);

    // Forward declarations for color space transforms
    inline void DWG_linear_to_XYZ(const float RGB[3], float XYZ[3]);
    inline void XYZ_to_DWG_linear(const float XYZ[3], float RGB[3]);

    // --- S-matrix inversion for CMF-based SPD reconstruction ---

    // Global variables for SPD reconstruction
    inline std::atomic<bool>& gSPDInit = context().spdInit;
    inline std::mutex gSPDMutex;
    inline float(&gS_inv)[9] = context().sInv; // row-major inverse of 3x3 S

    inline void compute_S_inverse_once() {
        if (gSPDInit.load(std::memory_order_acquire)) return;

        std::lock_guard<std::mutex> lock(gSPDMutex);
        if (gSPDInit.load(std::memory_order_acquire)) return;

        // Accumulate S over the fixed grid with Δλ = kDelta
        double Sxx = 0, Sxy = 0, Sxz = 0;
        double Syx = 0, Syy = 0, Syz = 0;
        double Szx = 0, Szy = 0, Szz = 0;

        for (int i = 0; i < gShape.K; ++i) {
            const float x = gAx[i];
            const float y = gAy[i];
            const float z = gAz[i];
            const float bx = std::max(0.0f, x);
            const float by = std::max(0.0f, y);
            const float bz = std::max(0.0f, z);

            Sxx += bx * x; Sxy += bx * y; Sxz += bx * z;
            Syx += by * x; Syy += by * y; Syz += by * z;
            Szx += bz * x; Szy += bz * y; Szz += bz * z;
        }
        const double dl = static_cast<double>(gDeltaLambda);
        Sxx *= dl; Sxy *= dl; Sxz *= dl;
        Syx *= dl; Syy *= dl; Syz *= dl;
        Szx *= dl; Szy *= dl; Szz *= dl;


        // Invert S via adjugate
        const double det =
            Sxx * (Syy * Szz - Syz * Szy) - Sxy * (Syx * Szz - Syz * Szx) + Sxz * (Syx * Szy - Syy * Szx);

        // Robustness: if ill‑conditioned, fall back to identity
        if (determinant_near_zero(det)) {
            set_identity_3x3(gS_inv);
            gSPDInit.store(true, std::memory_order_release);
            return;
        }

        const double invDet = 1.0 / det;
        const double invSxx = (Syy * Szz - Syz * Szy) * invDet;
        const double invSxy = (Sxz * Szy - Sxy * Szz) * invDet;
        const double invSxz = (Sxy * Syz - Sxz * Syy) * invDet;
        const double invSyx = (Syz * Szx - Syx * Szz) * invDet;
        const double invSyy = (Sxx * Szz - Sxz * Szx) * invDet;
        const double invSyz = (Sxz * Syx - Sxx * Syz) * invDet;
        const double invSzx = (Syx * Szy - Syy * Szx) * invDet;
        const double invSzy = (Sxy * Szx - Sxx * Szy) * invDet;
        const double invSzz = (Sxx * Syy - Sxy * Syx) * invDet;

        store_3x3_rowmajor(
            gS_inv,
            invSxx, invSxy, invSxz,
            invSyx, invSyy, invSyz,
            invSzx, invSzy, invSzz);

        gSPDInit.store(true, std::memory_order_release);
    }

    inline void compute_S_inverse_from_tables(const SpectralTables& T, float S_inv_out[9]) {
        // S = ∫ [max(0, CMF)]^T * [CMF] dλ under viewing axis in T
        double Sxx = 0, Sxy = 0, Sxz = 0, Syx = 0, Syy = 0, Syz = 0, Szx = 0, Szy = 0, Szz = 0;
        const int K = T.K;
        for (int i = 0; i < K; ++i) {
            const float x = T.Ax[i], y = T.Ay[i], z = T.Az[i];
            const float bx = std::max(0.0f, x);
            const float by = std::max(0.0f, y);
            const float bz = std::max(0.0f, z);
            Sxx += bx * x; Sxy += bx * y; Sxz += bx * z;
            Syx += by * x; Syy += by * y; Syz += by * z;
            Szx += bz * x; Szy += bz * y; Szz += bz * z;
        }
        const double dl = static_cast<double>(T.deltaLambda);
        Sxx *= dl; Sxy *= dl; Sxz *= dl; Syx *= dl; Syy *= dl; Syz *= dl; Szx *= dl; Szy *= dl; Szz *= dl;

        const double det = Sxx * (Syy * Szz - Syz * Szy) - Sxy * (Syx * Szz - Syz * Szx) + Sxz * (Syx * Szy - Syy * Szx);
        if (determinant_near_zero(det)) {
            set_identity_3x3(S_inv_out);
            return;
        }
        const double invDet = 1.0 / det;
        store_3x3_rowmajor(
            S_inv_out,
            (Syy * Szz - Syz * Szy) * invDet,
            (Sxz * Szy - Sxy * Szz) * invDet,
            (Sxy * Syz - Sxz * Syy) * invDet,
            (Syz * Szx - Syx * Szz) * invDet,
            (Sxx * Szz - Sxz * Szx) * invDet,
            (Sxz * Syx - Sxx * Syz) * invDet,
            (Syx * Szy - Syy * Szx) * invDet,
            (Sxy * Szx - Sxx * Szy) * invDet,
            (Sxx * Syy - Sxy * Syx) * invDet);
    }

    // --- CMF-based SPD reconstruction (global) ---

    inline void reconstruct_Ee_from_DWG_RGB(const float rgbDWG[3], std::vector<float>& Ee_out) {
        compute_S_inverse_once();

        float XYZ[3];
        DWG_linear_to_XYZ(rgbDWG, XYZ);
        clamp_triplet_nonnegative(XYZ);

        const float targetScale = XYZ[1];
        spd_probe_record_target_scale(targetScale);

        float cx = gS_inv[0] * XYZ[0] + gS_inv[1] * XYZ[1] + gS_inv[2] * XYZ[2];
        float cy = gS_inv[3] * XYZ[0] + gS_inv[4] * XYZ[1] + gS_inv[5] * XYZ[2];
        float cz = gS_inv[6] * XYZ[0] + gS_inv[7] * XYZ[1] + gS_inv[8] * XYZ[2];

        cx = std::max(0.0f, cx);
        cy = std::max(0.0f, cy);
        cz = std::max(0.0f, cz);

        const int K = gShape.K;
        Ee_out.resize(K);

        double Y_recon = 0.0;
        for (int i = 0; i < K; ++i) {
            const float bx = std::max(0.0f, gAx[i]);
            const float by = std::max(0.0f, gAy[i]);
            const float bz = std::max(0.0f, gAz[i]);

            const float Ei = std::max(1e-6f, cx * bx + cy * by + cz * bz);
            Ee_out[i] = Ei;
            Y_recon += static_cast<double>(Ei) * static_cast<double>(gAy[i]);
        }

        if (Y_recon > 1e-20 && targetScale > 0.0f) {
            const float s = static_cast<float>(static_cast<double>(targetScale) / Y_recon);
            for (int i = 0; i < K; ++i) {
                Ee_out[i] = std::max(0.0f, s * Ee_out[i]);
            }
        }
        else if (targetScale <= 0.0f) {
            std::fill(Ee_out.begin(), Ee_out.end(), 0.0f);
        }

        spd_probe_record_spectrum(Ee_out);
    }

    // Tri->quad mapping identical to agx-emulsion
    inline void tri2quad(float tx, float ty, float& qx, float& qy) {
        const float denom = std::max(1.0f - tx, 1e-10f);
        float x = (1.0f - tx);
        x = x * x;
        float y = ty / denom;
        qx = std::clamp(x, 0.0f, 1.0f);
        qy = std::clamp(y, 0.0f, 1.0f);
    }

    inline void quad2tri(float qx, float qy, float& tx, float& ty) {
        const float s = std::sqrt(std::max(0.0f, qx));
        tx = 1.0f - s;
        ty = qy * s;
    }

    inline float mitchell_weight(float x) {
        constexpr float B = 1.0f / 3.0f;
        constexpr float C = 1.0f / 3.0f;
        x = std::abs(x);
        if (x < 1.0f) {
            const float x2 = x * x;
            const float x3 = x2 * x;
            return ((12.0f - 9.0f * B - 6.0f * C) * x3 +
                (-18.0f + 12.0f * B + 6.0f * C) * x2 +
                (6.0f - 2.0f * B)) * (1.0f / 6.0f);
        }
        if (x < 2.0f) {
            const float x2 = x * x;
            const float x3 = x2 * x;
            return ((-B - 6.0f * C) * x3 +
                (6.0f * B + 30.0f * C) * x2 +
                (-12.0f * B - 48.0f * C) * x +
                (8.0f * B + 24.0f * C)) * (1.0f / 6.0f);
        }
        return 0.0f;
    }

    inline int reflect_index(int idx, int size) {
        if (size <= 0) return 0;
        while (idx < 0 || idx >= size) {
            if (idx < 0) {
                idx = -idx - 1;
            }
            else {
                idx = 2 * size - idx - 1;
            }
        }
        return idx;
    }

    inline void hanatos_linear_spectrum(float qx, float qy, std::vector<float>& Ee_out) {
        const int K = gShape.K;
        Ee_out.resize(K);

        if (gHanSpectra.size <= 0) {
            std::fill(Ee_out.begin(), Ee_out.end(), 0.0f);
            return;
        }

        const int N = gHanSpectra.size;
        const float fx = std::clamp(qx, 0.0f, 1.0f) * (N - 1);
        const float fy = std::clamp(qy, 0.0f, 1.0f) * (N - 1);
        const int x0 = std::clamp(static_cast<int>(std::floor(fx)), 0, N - 1);
        const int y0 = std::clamp(static_cast<int>(std::floor(fy)), 0, N - 1);
        const int x1 = std::min(x0 + 1, N - 1);
        const int y1 = std::min(y0 + 1, N - 1);
        const float tx = fx - static_cast<float>(x0);
        const float ty = fy - static_cast<float>(y0);

        // Bilinear interpolation matching scipy.interpolate.RegularGridInterpolator
        for (int k = 0; k < K; ++k) {
            const size_t idx00 = ((static_cast<size_t>(x0) * N + y0) * static_cast<size_t>(gHanSpectra.numSamples) + k);
            const size_t idx10 = ((static_cast<size_t>(x1) * N + y0) * static_cast<size_t>(gHanSpectra.numSamples) + k);
            const size_t idx01 = ((static_cast<size_t>(x0) * N + y1) * static_cast<size_t>(gHanSpectra.numSamples) + k);
            const size_t idx11 = ((static_cast<size_t>(x1) * N + y1) * static_cast<size_t>(gHanSpectra.numSamples) + k);

            const float v00 = gHanSpectra.data[idx00];
            const float v10 = gHanSpectra.data[idx10];
            const float v01 = gHanSpectra.data[idx01];
            const float v11 = gHanSpectra.data[idx11];

            // Bilinear: (1-tx)*(1-ty)*v00 + tx*(1-ty)*v10 + (1-tx)*ty*v01 + tx*ty*v11
            const float v0 = v00 * (1.0f - tx) + v10 * tx;
            const float v1 = v01 * (1.0f - tx) + v11 * tx;
            Ee_out[k] = v0 * (1.0f - ty) + v1 * ty;
        }
    }

    inline void hanatos_cubic_spectrum(float qx, float qy, std::vector<float>& Ee_out) {
        const int K = gShape.K;
        Ee_out.resize(K);

        if (gHanSpectra.size <= 0) {
            std::fill(Ee_out.begin(), Ee_out.end(), 0.0f);
            return;
        }

        const int N = gHanSpectra.size;
        const float fx = std::clamp(qx, 0.0f, 1.0f) * (N - 1);
        const float fy = std::clamp(qy, 0.0f, 1.0f) * (N - 1);
        const int baseX = std::clamp(static_cast<int>(std::floor(fx)), 0, N - 1);
        const int baseY = std::clamp(static_cast<int>(std::floor(fy)), 0, N - 1);
        const float tx = fx - static_cast<float>(baseX);
        const float ty = fy - static_cast<float>(baseY);

        auto at = [&](int i, int j, int k) -> float {
            size_t idx = ((static_cast<size_t>(i) * N + j) * static_cast<size_t>(gHanSpectra.numSamples) + k);
            return gHanSpectra.data[idx];
            };

        thread_local std::vector<float> tapBuffer;
        const size_t required = static_cast<size_t>(K) * 16;
        if (tapBuffer.size() != required) {
            tapBuffer.resize(required);
        }

        float wx[4];
        float wy[4];
        int xIndices[4];
        int yIndices[4];

        for (int i = 0; i < 4; ++i) {
            const int offset = i - 1;
            wx[i] = mitchell_weight(static_cast<float>(offset) - tx);
            xIndices[i] = reflect_index(baseX + offset, N);
            wy[i] = mitchell_weight(static_cast<float>(offset) - ty);
            yIndices[i] = reflect_index(baseY + offset, N);
        }

        int tap = 0;
        for (int j = 0; j < 4; ++j) {
            const int yIdx = yIndices[j];
            for (int i = 0; i < 4; ++i) {
                const int xIdx = xIndices[i];
                float* dest = tapBuffer.data() + static_cast<size_t>(tap) * static_cast<size_t>(K);
                for (int k = 0; k < K; ++k) {
                    dest[k] = at(xIdx, yIdx, k);
                }
                ++tap;
            }
        }

        std::fill(Ee_out.begin(), Ee_out.end(), 0.0f);

        float totalWeight = 0.0f;
        tap = 0;
        for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < 4; ++i) {
                const float w = wx[i] * wy[j];
                totalWeight += w;
                const float* src = tapBuffer.data() + static_cast<size_t>(tap) * static_cast<size_t>(K);
                for (int k = 0; k < K; ++k) {
                    Ee_out[k] += src[k] * w;
                }
                ++tap;
            }
        }

        if (totalWeight > 0.0f) {
            const float invW = 1.0f / totalWeight;
            for (int k = 0; k < K; ++k) {
                Ee_out[k] *= invW;
            }
        }
    }

    // Forward declarations for curve samplers (defined in section 5)
    inline float eps_yellow(float lambda);
    inline float eps_magenta(float lambda);
    inline float eps_cyan(float lambda);
    inline float cie_xbar(float lambda);
    inline float cie_ybar(float lambda);
    inline float cie_zbar(float lambda);

    // Reconstruct Ee via Hanatos spectra LUT from DWG RGB (match agx-emulsion behavior)
    // Per agx-emulsion parity: applies CAT02 chromatic adaptation from DWG D65 white point
    // to reference illuminant white point before computing xy chromaticity.
    inline void reconstruct_Ee_from_DWG_RGB_hanatos(
        const float rgbDWG[3],
        std::vector<float>& Ee_out,
        const float refIllumWhiteXYZ[3])
    {
        const int K = gShape.K;
        Ee_out.resize(K);

        if (!hanatos_available() || !hanatos_matches_reference_shape()) {
            reconstruct_Ee_from_DWG_RGB(rgbDWG, Ee_out);
            return;
        }

        // Convert DWG RGB to XYZ (DWG uses D65 white point)
        float XYZ[3];
        DWG_linear_to_XYZ(rgbDWG, XYZ);
        // agx-emulsion parity: keep signed XYZ; only sanitize non-finite components.
        sanitize_nonfinite_triplet(XYZ);

        float refWhiteXYZ[3];
        sanitize_ref_white_or_dwg(refIllumWhiteXYZ, refWhiteXYZ);

        // Apply CAT02 chromatic adaptation from D65 to reference illuminant.
        // This matches Python: colour.RGB_to_XYZ(..., illuminant=ref_illum, chromatic_adaptation_transform='CAT02')
        float adaptedXYZ[3];
        chromatic_adapt_XYZ_CAT02(XYZ, gDWG_WhitePoint_XYZ, refWhiteXYZ, adaptedXYZ);
        spd_probe_record_cat02(XYZ, adaptedXYZ, gDWG_WhitePoint_XYZ, refWhiteXYZ);

        // agx-emulsion parity:
        // - sanitize adaptedXYZ (non-finite -> 0), do not clamp negatives
        // - b = sum(adaptedXYZ) (signed)
        // - xy uses denom = max(b, 1e-10) and is then clamped to [0, 1]
        sanitize_nonfinite_triplet(adaptedXYZ);
        const float b = adaptedXYZ[0] + adaptedXYZ[1] + adaptedXYZ[2];
        const float bSafe = sanitize_nonfinite_component(b);
        spd_probe_record_target_scale(bSafe);

        const float denom = std::max(bSafe, 1e-10f);
        float x = adaptedXYZ[0] / denom;
        float y = adaptedXYZ[1] / denom;
        x = std::clamp(x, 0.0f, 1.0f);
        y = std::clamp(y, 0.0f, 1.0f);

        float qx, qy;
        tri2quad(x, y, qx, qy);
        spd_probe_record_coords(qx, qy);

        hanatos_linear_spectrum(qx, qy, Ee_out);  // Use bilinear to match Python's RegularGridInterpolator
        spd_probe_record_raw_lut(Ee_out);  // Log raw LUT spectrum BEFORE targetScale

        // Multiply by b to get final Ee spectrum (signed; matches Python).
        for (int i = 0; i < K; ++i) {
            Ee_out[i] = sanitize_nonfinite_component(bSafe * Ee_out[i]);
        }

        // Compute Y_recon for diagnostic logging (should match Python's ~3.95 for mid-gray)
        double Y_recon = 0.0;
        const bool hasYbar = (!gYBar.linear.empty() &&
            static_cast<int>(gYBar.linear.size()) == K);
        const bool hasLambda = (static_cast<int>(gShape.wavelengths.size()) == K);
        for (int i = 0; i < K; ++i) {
            const float lambda = hasLambda ? gShape.wavelengths[i] : (380.0f + 5.0f * i);
            const float ybar = hasYbar ? gYBar.linear[i] : cie_ybar(lambda);
            Y_recon += static_cast<double>(Ee_out[i]) * static_cast<double>(ybar);
        }
        spd_probe_record_yrecon(Y_recon);

        spd_probe_record_spectrum(Ee_out);
    }

    // -------------------------------------------------------------------------
    // 3. TABLE OPERATIONS (~150 lines)
    // -------------------------------------------------------------------------

    namespace precompute_impl {
        inline void precompute_spectral_tables_locked_body() {
            if (gShape.K <= 0 ||
                gShape.wavelengths.size() != static_cast<size_t>(gShape.K) ||
                !spectral_shape_matches_reference(gShape)) {
                JTRACE("SPECTRAL", "FATAL: gShape corrupted, expected 380-780@5nm");
                JTRACE("SPECTRAL", "Forcing rebuild to fixed grid");
                gShape = make_reference_spectral_shape();
                increment_shape_version();
            }

            // Set Δλ for all subsequent integrals
            gDeltaLambda = compute_delta_from_shape(gShape);

            // Ensure storage is allocated for current K
            const int K = gShape.K;
            gLambda.resize(K);
            gEpsYTable.resize(K);
            gEpsMTable.resize(K);
            gEpsCTable.resize(K);
            gXbarTable.resize(K);
            gYbarTable.resize(K);
            gZbarTable.resize(K);
            gBaselineMinTable.resize(K);
            gBaselineMidTable.resize(K);
            gIllumTable.resize(K);
            gAx.resize(K);
            gAy.resize(K);
            gAz.resize(K);

            // 1) Wavelength axis from gShape
            gLambda.assign(gShape.wavelengths.begin(), gShape.wavelengths.end());

            // 2) Sample all curves on the working grid (size-safe)
            const bool illumSizeMismatch = !gIlluminantCurve.linear.empty() &&
                ((int)gIlluminantCurve.linear.size() != K ||
                    gIlluminantCurve.lambda_nm.size() != gIlluminantCurve.linear.size());
            bool illumAxisMismatch = false;
            if (!illumSizeMismatch && !gIlluminantCurve.linear.empty()) {
                for (int i = 0; i < K; ++i) {
                    const float expected = gShape.wavelengths[i];
                    if (std::abs(gIlluminantCurve.lambda_nm[i] - expected) > 1e-3f) {
                        illumAxisMismatch = true;
                        break;
                    }
                }
            }
            const bool illumUseDirect = !illumSizeMismatch && !illumAxisMismatch;
            if ((illumSizeMismatch || illumAxisMismatch) && !gIlluminantCurve.linear.empty()) {
                JTRACE("SPECTRAL", "Illuminant axis/size mismatch; falling back to equal-energy for precompute");
            }
            const bool hasEpsYDirect = !gEpsY.linear.empty() && (int)gEpsY.linear.size() == K;
            const bool hasEpsMDirect = !gEpsM.linear.empty() && (int)gEpsM.linear.size() == K;
            const bool hasEpsCDirect = !gEpsC.linear.empty() && (int)gEpsC.linear.size() == K;
            const bool hasXbarDirect = !gXBar.linear.empty() && (int)gXBar.linear.size() == K;
            const bool hasYbarDirect = !gYBar.linear.empty() && (int)gYBar.linear.size() == K;
            const bool hasZbarDirect = !gZBar.linear.empty() && (int)gZBar.linear.size() == K;
            const bool hasBaseMinDirect = gHasBaseline && (int)gBaseMin.linear.size() == K;
            const bool hasBaseMidDirect = gHasBaseline && (int)gBaseMid.linear.size() == K;
            const bool hasIllumDirect = !gIlluminantCurve.linear.empty() && illumUseDirect;
            const float* epsYDirect = hasEpsYDirect ? gEpsY.linear.data() : nullptr;
            const float* epsMDirect = hasEpsMDirect ? gEpsM.linear.data() : nullptr;
            const float* epsCDirect = hasEpsCDirect ? gEpsC.linear.data() : nullptr;
            const float* xbarDirect = hasXbarDirect ? gXBar.linear.data() : nullptr;
            const float* ybarDirect = hasYbarDirect ? gYBar.linear.data() : nullptr;
            const float* zbarDirect = hasZbarDirect ? gZBar.linear.data() : nullptr;
            const float* baseMinDirect = hasBaseMinDirect ? gBaseMin.linear.data() : nullptr;
            const float* baseMidDirect = hasBaseMidDirect ? gBaseMid.linear.data() : nullptr;
            const float* illumDirect = hasIllumDirect ? gIlluminantCurve.linear.data() : nullptr;
            const float* lambdaData = gLambda.data();
            float* epsYTableData = gEpsYTable.data();
            float* epsMTableData = gEpsMTable.data();
            float* epsCTableData = gEpsCTable.data();
            float* xbarTableData = gXbarTable.data();
            float* ybarTableData = gYbarTable.data();
            float* zbarTableData = gZbarTable.data();
            float* baseMinTableData = gBaselineMinTable.data();
            float* baseMidTableData = gBaselineMidTable.data();
            float* illumTableData = gIllumTable.data();

            for (int i = 0; i < K; ++i) {
                const float l = lambdaData[i];

                // Measured dye extinctions: if pinned exactly to K, use direct indexing; otherwise sample by wavelength.
                if (epsYDirect) epsYTableData[i] = epsYDirect[i];
                else epsYTableData[i] = eps_yellow(l);

                if (epsMDirect) epsMTableData[i] = epsMDirect[i];
                else epsMTableData[i] = eps_magenta(l);

                if (epsCDirect) epsCTableData[i] = epsCDirect[i];
                else epsCTableData[i] = eps_cyan(l);

                // CMFs with the same size-safe rule
                if (xbarDirect) xbarTableData[i] = xbarDirect[i];
                else xbarTableData[i] = cie_xbar(l);

                if (ybarDirect) ybarTableData[i] = ybarDirect[i];
                else ybarTableData[i] = cie_ybar(l);

                if (zbarDirect) zbarTableData[i] = zbarDirect[i];
                else zbarTableData[i] = cie_zbar(l);

                // Baseline (size-safe: only direct index if sizes match)
                baseMinTableData[i] = baseMinDirect ? baseMinDirect[i] : 0.0f;
                baseMidTableData[i] = baseMidDirect ? baseMidDirect[i] : 0.0f;

                // Illuminant: only direct index if axis/size match; otherwise fallback to equal-energy for this sample
                illumTableData[i] = illumDirect ? illumDirect[i] : 1.0f; // equal-energy fallback
            }


            // 3) Precompute Ee*CMFs and Yn normalization
            float Yn = 0.0f;
            const float* illumData = gIllumTable.data();
            const float* xbarData = gXbarTable.data();
            const float* ybarData = gYbarTable.data();
            const float* zbarData = gZbarTable.data();
            float* axData = gAx.data();
            float* ayData = gAy.data();
            float* azData = gAz.data();
            for (int i = 0; i < K; ++i) {
                const float Ee = illumData[i];
                const float x = xbarData[i];
                const float y = ybarData[i];
                const float z = zbarData[i];

                axData[i] = Ee * x;
                ayData[i] = Ee * y;
                azData[i] = Ee * z;

                Yn += ayData[i];

            }

            gYnNorm = (Yn > 0.0f) ? Yn : 1.0f;
            gInvYn = 1.0f / gYnNorm;
            gSPDInit.store(false, std::memory_order_release);
        }

    } // namespace precompute_impl

    inline void precompute_spectral_tables() {
        bool dirty = gPrecomputeStatus.dirty.load(std::memory_order_acquire);
        uint64_t illumVersion = gPrecomputeStatus.illumVersion.load(std::memory_order_acquire);
        uint64_t lastIllum = gPrecomputeStatus.lastPrecomputeIllumVersion.load(std::memory_order_acquire);
        uint64_t shapeVersion = gPrecomputeStatus.shapeVersion.load(std::memory_order_acquire);
        uint64_t lastShape = gPrecomputeStatus.lastPrecomputeShapeVersion.load(std::memory_order_acquire);

        if (!dirty && illumVersion == lastIllum && shapeVersion == lastShape) {
            return;
        }

        std::unique_lock<std::mutex> lock(gPrecomputeMutex);

        dirty = gPrecomputeStatus.dirty.load(std::memory_order_acquire);
        illumVersion = gPrecomputeStatus.illumVersion.load(std::memory_order_acquire);
        shapeVersion = gPrecomputeStatus.shapeVersion.load(std::memory_order_acquire);
        lastIllum = gPrecomputeStatus.lastPrecomputeIllumVersion.load(std::memory_order_relaxed);
        lastShape = gPrecomputeStatus.lastPrecomputeShapeVersion.load(std::memory_order_relaxed);

        if (!dirty && illumVersion == lastIllum && shapeVersion == lastShape) {
            return;
        }

        precompute_impl::precompute_spectral_tables_locked_body();
        disable_hanatos_if_reference_mismatch();

        illumVersion = gPrecomputeStatus.illumVersion.load(std::memory_order_acquire);
        shapeVersion = gPrecomputeStatus.shapeVersion.load(std::memory_order_acquire);

        gPrecomputeStatus.lastPrecomputeIllumVersion.store(illumVersion, std::memory_order_release);
        gPrecomputeStatus.lastPrecomputeShapeVersion.store(shapeVersion, std::memory_order_release);
        gPrecomputeStatus.dirty.store(false, std::memory_order_release);
    }

    inline void ensure_precomputed_up_to_date() {
        const bool dirty = gPrecomputeStatus.dirty.load(std::memory_order_acquire);
        const auto illumVersion = gPrecomputeStatus.illumVersion.load(std::memory_order_acquire);
        const auto lastIllum = gPrecomputeStatus.lastPrecomputeIllumVersion.load(std::memory_order_acquire);
        const auto shapeVersion = gPrecomputeStatus.shapeVersion.load(std::memory_order_acquire);
        const auto lastShape = gPrecomputeStatus.lastPrecomputeShapeVersion.load(std::memory_order_acquire);

        if (!dirty && illumVersion == lastIllum && shapeVersion == lastShape) {
            return;
        }

        precompute_spectral_tables();
    }

    inline void build_tables_from_curves_non_global(
        const Curve& epsY, const Curve& epsM, const Curve& epsC,
        const Curve& xbar, const Curve& ybar, const Curve& zbar,
        const Curve& illumView,
        const Curve& baseMin, const Curve& baseMid, bool hasBaseline,
        float baselineMixReference,
        SpectralTables& T,
        std::uint64_t illuminantHash = 0)
    {
        (void)baselineMixReference;
        const int K = gShape.K;
        T.K = K;
        T.lambda.assign(gShape.wavelengths.begin(), gShape.wavelengths.end());
        T.deltaLambda = compute_delta_from_shape(gShape);

        T.epsY.resize(K);
        T.epsM.resize(K);
        T.epsC.resize(K);
        T.Xbar.resize(K);
        T.Ybar.resize(K);
        T.Zbar.resize(K);
        T.Ax.resize(K);
        T.Ay.resize(K);
        T.Az.resize(K);
        T.illum.resize(K);

        const bool hasIll = (!illumView.linear.empty() && (int)illumView.linear.size() == K);
        double Yn = 0.0;
        double sumAx = 0.0;
        double sumAy = 0.0;
        double sumAz = 0.0;
        const bool hasEpsY = (!epsY.linear.empty() && (int)epsY.linear.size() == K);
        const bool hasEpsM = (!epsM.linear.empty() && (int)epsM.linear.size() == K);
        const bool hasEpsC = (!epsC.linear.empty() && (int)epsC.linear.size() == K);
        const bool hasXbar = (!xbar.linear.empty() && (int)xbar.linear.size() == K);
        const bool hasYbar = (!ybar.linear.empty() && (int)ybar.linear.size() == K);
        const bool hasZbar = (!zbar.linear.empty() && (int)zbar.linear.size() == K);
        const float* epsYData = hasEpsY ? epsY.linear.data() : nullptr;
        const float* epsMData = hasEpsM ? epsM.linear.data() : nullptr;
        const float* epsCData = hasEpsC ? epsC.linear.data() : nullptr;
        const float* xbarData = hasXbar ? xbar.linear.data() : nullptr;
        const float* ybarData = hasYbar ? ybar.linear.data() : nullptr;
        const float* zbarData = hasZbar ? zbar.linear.data() : nullptr;
        const float* illumData = hasIll ? illumView.linear.data() : nullptr;
        const float* lambdaData = T.lambda.data();
        float* outEpsY = T.epsY.data();
        float* outEpsM = T.epsM.data();
        float* outEpsC = T.epsC.data();
        float* outXbar = T.Xbar.data();
        float* outYbar = T.Ybar.data();
        float* outZbar = T.Zbar.data();
        float* outIllum = T.illum.data();
        float* outAx = T.Ax.data();
        float* outAy = T.Ay.data();
        float* outAz = T.Az.data();
        for (int i = 0; i < K; ++i) {
            const float l = lambdaData[i];

            const float ey = epsYData ? epsYData[i] : eps_yellow(l);
            const float em = epsMData ? epsMData[i] : eps_magenta(l);
            const float ec = epsCData ? epsCData[i] : eps_cyan(l);
            outEpsY[i] = ey;
            outEpsM[i] = em;
            outEpsC[i] = ec;

            outXbar[i] = xbarData ? xbarData[i] : cie_xbar(l);
            outYbar[i] = ybarData ? ybarData[i] : cie_ybar(l);
            outZbar[i] = zbarData ? zbarData[i] : cie_zbar(l);

            const float Ee = illumData ? illumData[i] : 1.0f;
            outIllum[i] = Ee;
            outAx[i] = Ee * outXbar[i];
            outAy[i] = Ee * outYbar[i];
            outAz[i] = Ee * outZbar[i];
            Yn += outAy[i];
            sumAx += outAx[i];
            sumAy += outAy[i];
            sumAz += outAz[i];
        }
        T.invYn = (Yn > 0.0) ? (1.0f / (float)Yn) : 1.0f;
        const double scale = static_cast<double>(T.invYn);
        store_scaled_xyz(sumAx, sumAy, sumAz, static_cast<float>(scale), T.whiteXYZ);

        // Illuminant white point used for chromatic adaptation (normalized to Y=1).
        copy_triplet3(T.whiteXYZ, T.refIllumWhiteXYZ);

        T.hasBaseline = hasBaseline &&
            (int)baseMin.linear.size() == K;
        T.baseMin.assign(K, 0.0f);
        T.baseMid.assign(K, 0.0f);
        T.baselineMixReference = 0.0f;
        if (T.hasBaseline) {
            T.baseMin.assign(baseMin.linear.begin(), baseMin.linear.end());
            if ((int)baseMid.linear.size() == K) {
                T.baseMid.assign(baseMid.linear.begin(), baseMid.linear.end());
            }
        }

        T.illuminantHash = illuminantHash;
        if (T.illuminantHash == 0 && hasIll) {
            T.illuminantHash = hash_float_vector_digest_sp(illumView.linear);
        }
        const std::uint64_t tableFields[] = {
            T.illuminantHash,
            hash_float_vector_digest_sp(T.lambda),
            hash_float_scalar_digest_sp(T.deltaLambda),
            hash_float_scalar_digest_sp(T.invYn),
            hash_float_triplet_digest_sp(T.whiteXYZ),
            hash_float_triplet_digest_sp(T.refIllumWhiteXYZ),
            hash_float_vector_digest_sp(T.Ax),
            hash_float_vector_digest_sp(T.Ay),
            hash_float_vector_digest_sp(T.Az),
            hash_float_vector_digest_sp(T.illum),
            hash_float_vector_digest_sp(T.Xbar),
            hash_float_vector_digest_sp(T.Ybar),
            hash_float_vector_digest_sp(T.Zbar),
            hash_float_vector_digest_sp(T.epsC),
            hash_float_vector_digest_sp(T.epsM),
            hash_float_vector_digest_sp(T.epsY),
            hash_float_vector_digest_sp(T.baseMin),
            hash_float_vector_digest_sp(T.baseMid),
            hash_float_scalar_digest_sp(T.baselineMixReference),
            Hash::hash_bytes(&T.hasBaseline, sizeof(T.hasBaseline))
        };
        T.tablesHash = Hash::hash_bytes(tableFields, sizeof(tableFields));
    }

    inline void reconstruct_Ee_from_DWG_RGB_with_tables(
        const float rgbDWG[3],
        const SpectralTables& T,
        const float S_inv[9],
        std::vector<float>& Ee_out)
    {
        float XYZ[3];
        DWG_linear_to_XYZ(rgbDWG, XYZ);

        float sanitizedXYZ[3];
        sanitize_nonnegative_triplet_sp(sanitizedXYZ, XYZ);

        float refWhite[3];
        sanitize_nonnegative_triplet_sp(refWhite, T.refIllumWhiteXYZ);
        if (refWhite[1] <= 0.0f) {
            copy_triplet3(gDWG_WhitePoint_XYZ, refWhite);
        }

        float adaptedXYZ[3];
        chromatic_adapt_XYZ_CAT02(sanitizedXYZ, gDWG_WhitePoint_XYZ, refWhite, adaptedXYZ);
        spd_probe_record_cat02(sanitizedXYZ, adaptedXYZ, gDWG_WhitePoint_XYZ, refWhite);

        clamp_triplet_nonnegative(adaptedXYZ);

        const float targetScale = (adaptedXYZ[1] > 0.0f) ? adaptedXYZ[1] : sanitizedXYZ[1];
        spd_probe_record_target_scale(targetScale);

        float cx = S_inv[0] * adaptedXYZ[0] + S_inv[1] * adaptedXYZ[1] + S_inv[2] * adaptedXYZ[2];
        float cy = S_inv[3] * adaptedXYZ[0] + S_inv[4] * adaptedXYZ[1] + S_inv[5] * adaptedXYZ[2];
        float cz = S_inv[6] * adaptedXYZ[0] + S_inv[7] * adaptedXYZ[1] + S_inv[8] * adaptedXYZ[2];
        cx = std::max(0.0f, cx);
        cy = std::max(0.0f, cy);
        cz = std::max(0.0f, cz);

        const int K = T.K;
        Ee_out.resize(K);
        float* eeData = Ee_out.data();
        const float* axData = T.Ax.data();
        const float* ayData = T.Ay.data();
        const float* azData = T.Az.data();
        double Y_recon = 0.0;
        for (int i = 0; i < K; ++i) {
            const float bx = std::max(0.0f, axData[i]);
            const float by = std::max(0.0f, ayData[i]);
            const float bz = std::max(0.0f, azData[i]);
            const float Ei = std::max(1e-6f, cx * bx + cy * by + cz * bz);
            eeData[i] = Ei;
            Y_recon += static_cast<double>(Ei) * static_cast<double>(ayData[i]);
        }
        if (Y_recon > 1e-20 && targetScale > 0.0f) {
            const float s = static_cast<float>(static_cast<double>(targetScale) / Y_recon);
            float* eeIt = eeData;
            for (int i = 0; i < K; ++i, ++eeIt) {
                *eeIt = std::max(0.0f, s * (*eeIt));
            }
        }
        else if (targetScale <= 0.0f) {
            std::fill(Ee_out.begin(), Ee_out.end(), 0.0f);
        }

        spd_probe_record_spectrum(Ee_out);
    }

    // Forward declarations for exposure functions
    inline void rgbDWG_to_layerExposures(const float rgbDWG[3], float E[3], float exposureScale);
    inline void layerExposures_from_sceneSPD(
        const std::vector<float>& Ee,
        float E[3],
        float exposureScale,
        bool applyDeltaLambda = true);
    inline void layerExposures_from_sceneSPD_with_curves(
        const std::vector<float>& Ee,
        const Curve& sB, const Curve& sG, const Curve& sR,
        float E[3],
        float exposureScale,
        bool applyDeltaLambda = true);

    // Per-instance SPD exposure using per-instance sensitivity curves
    inline void rgbDWG_to_layerExposures_from_tables_with_curves(
        const float rgbDWG[3], float E[3], float exposureScale,
        const SpectralTables* T, const float* S_inv /* size 9 */,
        const Curve& sB, const Curve& sG, const Curve& sR,
        SpectralUpsamplingMode spectralUpsamplingMode = SpectralUpsamplingMode::PreferHanatos,
        const float refIllumWhiteXYZ[3] /* optional override */ = nullptr)
    {
        if (!T || !S_inv || T->K <= 0) {
            E[0] = E[1] = E[2] = 0.0f;
            return;
        }

        thread_local std::vector<float> Ee_scene;
        Ee_scene.resize(T->K);

        const bool allowHanatos = (spectralUpsamplingMode == SpectralUpsamplingMode::PreferHanatos);
        const bool useHanatos = allowHanatos && hanatos_available() && hanatos_matches_reference_shape();
        const float* adaptWhite = refIllumWhiteXYZ ? refIllumWhiteXYZ : T->refIllumWhiteXYZ;
        if (useHanatos) {
            // Pass reference illuminant white point for chromatic adaptation
            reconstruct_Ee_from_DWG_RGB_hanatos(rgbDWG, Ee_scene, adaptWhite);
        }
        else {
            reconstruct_Ee_from_DWG_RGB_with_tables(rgbDWG, *T, S_inv, Ee_scene);
        }

        // Integrate with per-instance sensitivity curves (no globals)
        layerExposures_from_sceneSPD_with_curves(Ee_scene, sB, sG, sR, E, exposureScale, !useHanatos);
    }

    // Integrate spectral irradiance under CMFs (stored as Spectral::Curve) to XYZ
    inline void Ee_to_XYZ_given_cmf(
        const std::vector<float>& Ee,
        const Spectral::Curve& xbar,
        const Spectral::Curve& ybar,
        const Spectral::Curve& zbar,
        float XYZ[3])
    {
        assert(Ee.size() == xbar.linear.size() &&
            Ee.size() == ybar.linear.size() &&
            Ee.size() == zbar.linear.size());

        double X = 0.0, Y = 0.0, Z = 0.0;
        const size_t count = Ee.size();
        const float* eeData = Ee.data();
        const float* xData = xbar.linear.data();
        const float* yData = ybar.linear.data();
        const float* zData = zbar.linear.data();
        for (size_t i = 0; i < count; ++i, ++eeData, ++xData, ++yData, ++zData) {
            const double E = *eeData;
            X += E * (*xData);
            Y += E * (*yData);
            Z += E * (*zData);
        }

        const float s = Spectral::gInvYn;
        store_scaled_xyz(X, Y, Z, s, XYZ);
    }

    // Integrate spectral irradiance with per-instance tables (viewing axis and normalization)
    inline void Ee_to_XYZ_given_tables(
        const SpectralTables& T,
        const std::vector<float>& Ee,
        float XYZ[3])
    {
        double X = 0.0, Y = 0.0, Z = 0.0;
        const int K = T.K;
        const int N = static_cast<int>(Ee.size());
        const int count = std::min(K, N);
        const float* eeData = Ee.data();
        const float* xData = T.Xbar.data();
        const float* yData = T.Ybar.data();
        const float* zData = T.Zbar.data();
        for (int i = 0; i < count; ++i, ++eeData, ++xData, ++yData, ++zData) {
            const float e = *eeData;
            X += static_cast<double>(e) * static_cast<double>(*xData);
            Y += static_cast<double>(e) * static_cast<double>(*yData);
            Z += static_cast<double>(e) * static_cast<double>(*zData);
        }
        const float s = T.invYn;
        store_scaled_xyz(X, Y, Z, s, XYZ);
    }

    // -------------------------------------------------------------------------
    // 5. CURVE SAMPLERS (~50 lines)
    // -------------------------------------------------------------------------

    // Dye extinction samplers
    inline float eps_yellow(float lambda) {
        if (!gEpsY.lambda_nm.empty()) return gEpsY.sample(lambda);
        return 1.4f * gaussian(lambda, 440.0f, 25.0f);
    }
    inline float eps_magenta(float lambda) {
        if (!gEpsM.lambda_nm.empty()) return gEpsM.sample(lambda);
        return 1.2f * gaussian(lambda, 540.0f, 30.0f);
    }
    inline float eps_cyan(float lambda) {
        if (!gEpsC.lambda_nm.empty()) return gEpsC.sample(lambda);
        return 1.1f * gaussian(lambda, 610.0f, 35.0f);
    }

    // CMF samplers
    inline float cie_xbar(float lambda) {
        if (!gXBar.lambda_nm.empty()) return gXBar.sample(lambda);
        return 1.0f * gaussian(lambda, 595.0f, 40.0f) + 0.25f * gaussian(lambda, 445.0f, 20.0f);
    }
    inline float cie_ybar(float lambda) {
        if (!gYBar.lambda_nm.empty()) return gYBar.sample(lambda);
        return 1.0f * gaussian(lambda, 555.0f, 30.0f);
    }
    inline float cie_zbar(float lambda) {
        if (!gZBar.lambda_nm.empty()) return gZBar.sample(lambda);
        return 1.2f * gaussian(lambda, 445.0f, 25.0f);
    }

    // Sensitivity samplers
    inline float sens_blue(float lambda) { return gSensBlue.sample(lambda); }
    inline float sens_green(float lambda) { return gSensGreen.sample(lambda); }
    inline float sens_red(float lambda) { return gSensRed.sample(lambda); }

    // Effective layer gain helper
    inline float effective_layer_gain(const Curve& c) {
        if (c.lambda_nm.empty()) return 1.0f;
        const int K = gShape.K;
        float num = 0.0f, den = 0.0f;
        const float* illumData = gIllumTable.data();
        const float* lambdaData = gLambda.data();
        for (int i = 0; i < K; ++i) {
            const float Ee = illumData[i];
            const float s = c.sample(lambdaData[i]);
            num += Ee * s;
            den += Ee;
        }
        return (den > 0.0f) ? (num / den) : 1.0f;
    }

} // namespace Spectral
