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
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <cstdint>
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

#if defined(JUICER_SPD_DEBUG)
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
        for (int i = 0; i < 3; ++i) {
            buf.rgbInput[i] = rgbIn ? rgbIn[i] : 0.0f;
            buf.rgbDWG[i] = rgbDWG ? rgbDWG[i] : 0.0f;
        }
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
        for (int i = 0; i < 3; ++i) {
            buf.cat02InputXYZ[i] = XYZ_in ? XYZ_in[i] : 0.0f;
            buf.cat02OutputXYZ[i] = XYZ_out ? XYZ_out[i] : 0.0f;
            buf.cat02SrcWhite[i] = srcWhite ? srcWhite[i] : 0.0f;
            buf.cat02DstWhite[i] = dstWhite ? dstWhite[i] : 0.0f;
        }
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
            for (size_t i = 0; i < buf.rawLUT.size(); ++i) {
                if (i > 0) oss << ",";
                oss << buf.rawLUT[i];
            }
            oss << "]";
        }
        oss << " Ee=[";
        for (size_t i = 0; i < buf.Ee.size(); ++i) {
            if (i > 0) oss << ",";
            oss << buf.Ee[i];
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

    // -------------------------------------------------------------------------
    // 1. MATH UTILITIES (~100 lines)
    // -------------------------------------------------------------------------

    inline float sigmoid_erf(float x, float center, float width) {
        const float minMag = 1e-6f;
        float w = width;
        if (!std::isfinite(w)) {
            w = (w < 0.0f) ? -minMag : minMag;
        }
        if (std::fabs(w) < minMag) {
            w = (w < 0.0f) ? -minMag : minMag;
        }
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

        auto safe_width = [](float w) {
            const float minMag = 1e-6f;
            if (!std::isfinite(w)) {
                return (w < 0.0f) ? -minMag : minMag;
            }
            if (std::fabs(w) < minMag) {
                return (w < 0.0f) ? -minMag : minMag;
            }
            return w;
            };

        const float widthUV = safe_width(filterUV[2]);
        const float widthIR = -std::fabs(safe_width(filterIR[2]));

        for (int i = 0; i < K; ++i) {
            const float wl = gShape.wavelengths[static_cast<size_t>(i)];
            const float filter_uv = 1.0f - ampUV + ampUV * sigmoid_erf(wl, wlUV, widthUV);
            const float filter_ir = 1.0f - ampIR + ampIR * sigmoid_erf(wl, wlIR, widthIR);
            bandPass[static_cast<size_t>(i)] = filter_uv * filter_ir;
        }

        return bandPass;
    }

    inline float compute_delta_from_shape(const SpectralShape& s) {
        if (s.K <= 1 || s.wavelengths.size() < 2) return kDelta;
        // Estimate mean Δλ to be robust to tiny non-uniformities
        double sum = 0.0;
        for (int i = 1; i < s.K; ++i) sum += static_cast<double>(s.wavelengths[i] - s.wavelengths[i - 1]);
        const double mean = sum / static_cast<double>(s.K - 1);
        return (mean > 0.0) ? static_cast<float>(mean) : kDelta;
    }

    inline float illuminant_E(float /*lambda*/) { return 1.0f; }

    inline void fill_viewing_illuminant_Ee(float gain, std::vector<float>& Ee_out) {
        const int K = gShape.K;
        Ee_out.resize(K);
        for (int i = 0; i < K; ++i) {
            Ee_out[i] = std::max(0.0f, gain * gIllumTable[i]);
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
        if (std::fabs(det) < 1e-20) {
            gS_inv[0] = 1; gS_inv[1] = 0; gS_inv[2] = 0;
            gS_inv[3] = 0; gS_inv[4] = 1; gS_inv[5] = 0;
            gS_inv[6] = 0; gS_inv[7] = 0; gS_inv[8] = 1;
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

        gS_inv[0] = static_cast<float>(invSxx);
        gS_inv[1] = static_cast<float>(invSxy);
        gS_inv[2] = static_cast<float>(invSxz);
        gS_inv[3] = static_cast<float>(invSyx);
        gS_inv[4] = static_cast<float>(invSyy);
        gS_inv[5] = static_cast<float>(invSyz);
        gS_inv[6] = static_cast<float>(invSzx);
        gS_inv[7] = static_cast<float>(invSzy);
        gS_inv[8] = static_cast<float>(invSzz);

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
        if (std::fabs(det) < 1e-20) {
            S_inv_out[0] = 1; S_inv_out[1] = 0; S_inv_out[2] = 0;
            S_inv_out[3] = 0; S_inv_out[4] = 1; S_inv_out[5] = 0;
            S_inv_out[6] = 0; S_inv_out[7] = 0; S_inv_out[8] = 1;
            return;
        }
        const double invDet = 1.0 / det;
        S_inv_out[0] = static_cast<float>((Syy * Szz - Syz * Szy) * invDet);
        S_inv_out[1] = static_cast<float>((Sxz * Szy - Sxy * Szz) * invDet);
        S_inv_out[2] = static_cast<float>((Sxy * Syz - Sxz * Syy) * invDet);
        S_inv_out[3] = static_cast<float>((Syz * Szx - Syx * Szz) * invDet);
        S_inv_out[4] = static_cast<float>((Sxx * Szz - Sxz * Szx) * invDet);
        S_inv_out[5] = static_cast<float>((Sxz * Syx - Sxx * Syz) * invDet);
        S_inv_out[6] = static_cast<float>((Syx * Szy - Syy * Szx) * invDet);
        S_inv_out[7] = static_cast<float>((Sxy * Szx - Sxx * Szy) * invDet);
        S_inv_out[8] = static_cast<float>((Sxx * Syy - Sxy * Syx) * invDet);
    }

    // --- CMF-based SPD reconstruction (global) ---

    inline void reconstruct_Ee_from_DWG_RGB(const float rgbDWG[3], std::vector<float>& Ee_out) {
        compute_S_inverse_once();

        float XYZ[3];
        DWG_linear_to_XYZ(rgbDWG, XYZ);
        XYZ[0] = std::max(0.0f, XYZ[0]);
        XYZ[1] = std::max(0.0f, XYZ[1]);
        XYZ[2] = std::max(0.0f, XYZ[2]);

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
        for (int i = 0; i < 3; ++i) {
            if (!std::isfinite(XYZ[i])) {
                XYZ[i] = 0.0f;
            }
        }

        auto sanitize_white = [](const float* white, float dst[3]) {
            const float fallback[3] = {
                gDWG_WhitePoint_XYZ[0],
                gDWG_WhitePoint_XYZ[1],
                gDWG_WhitePoint_XYZ[2]
            };
            const float* src = white ? white : fallback;
            for (int i = 0; i < 3; ++i) {
                const float v = src[i];
                dst[i] = (std::isfinite(v)) ? v : fallback[i];
            }
            if (!(dst[1] > 0.0f)) {
                dst[0] = fallback[0];
                dst[1] = fallback[1];
                dst[2] = fallback[2];
            }
            };

        float refWhiteXYZ[3];
        sanitize_white(refIllumWhiteXYZ, refWhiteXYZ);

        // Apply CAT02 chromatic adaptation from D65 to reference illuminant.
        // This matches Python: colour.RGB_to_XYZ(..., illuminant=ref_illum, chromatic_adaptation_transform='CAT02')
        float adaptedXYZ[3];
        chromatic_adapt_XYZ_CAT02(XYZ, gDWG_WhitePoint_XYZ, refWhiteXYZ, adaptedXYZ);
        spd_probe_record_cat02(XYZ, adaptedXYZ, gDWG_WhitePoint_XYZ, refWhiteXYZ);

        // agx-emulsion parity:
        // - sanitize adaptedXYZ (non-finite -> 0), do not clamp negatives
        // - b = sum(adaptedXYZ) (signed)
        // - xy uses denom = max(b, 1e-10) and is then clamped to [0, 1]
        for (int i = 0; i < 3; ++i) {
            if (!std::isfinite(adaptedXYZ[i])) {
                adaptedXYZ[i] = 0.0f;
            }
        }
        const float b = adaptedXYZ[0] + adaptedXYZ[1] + adaptedXYZ[2];
        const float bSafe = std::isfinite(b) ? b : 0.0f;
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
            float v = bSafe * Ee_out[i];
            if (!std::isfinite(v)) {
                v = 0.0f;
            }
            Ee_out[i] = v;
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

    namespace detail {
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
            for (int i = 0; i < K; ++i) {
                gLambda[i] = gShape.wavelengths[i];
            }

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

            for (int i = 0; i < K; ++i) {
                const float l = gLambda[i];

                // Measured dye extinctions: if pinned exactly to K, use direct indexing; otherwise sample by wavelength.
                if (!gEpsY.linear.empty() && (int)gEpsY.linear.size() == K) gEpsYTable[i] = gEpsY.linear[i];
                else gEpsYTable[i] = eps_yellow(l);

                if (!gEpsM.linear.empty() && (int)gEpsM.linear.size() == K) gEpsMTable[i] = gEpsM.linear[i];
                else gEpsMTable[i] = eps_magenta(l);

                if (!gEpsC.linear.empty() && (int)gEpsC.linear.size() == K) gEpsCTable[i] = gEpsC.linear[i];
                else gEpsCTable[i] = eps_cyan(l);

                // CMFs with the same size-safe rule
                if (!gXBar.linear.empty() && (int)gXBar.linear.size() == K) gXbarTable[i] = gXBar.linear[i];
                else gXbarTable[i] = cie_xbar(l);

                if (!gYBar.linear.empty() && (int)gYBar.linear.size() == K) gYbarTable[i] = gYBar.linear[i];
                else gYbarTable[i] = cie_ybar(l);

                if (!gZBar.linear.empty() && (int)gZBar.linear.size() == K) gZbarTable[i] = gZBar.linear[i];
                else gZbarTable[i] = cie_zbar(l);

                // Baseline (size-safe: only direct index if sizes match)
                gBaselineMinTable[i] = (gHasBaseline && (int)gBaseMin.linear.size() == K) ? gBaseMin.linear[i] : 0.0f;
                gBaselineMidTable[i] = (gHasBaseline && (int)gBaseMid.linear.size() == K) ? gBaseMid.linear[i] : 0.0f;

                // Illuminant: only direct index if axis/size match; otherwise fallback to equal-energy for this sample
                if (!gIlluminantCurve.linear.empty() && illumUseDirect) {
                    gIllumTable[i] = gIlluminantCurve.linear[i];
                }
                else {
                    gIllumTable[i] = illuminant_E(l); // equal-energy fallback
                }
            }


            // 3) Precompute Ee*CMFs and Yn normalization
            float Yn = 0.0f;
            for (int i = 0; i < K; ++i) {
                const float Ee = gIllumTable[i];
                const float x = gXbarTable[i];
                const float y = gYbarTable[i];
                const float z = gZbarTable[i];

                gAx[i] = Ee * x;
                gAy[i] = Ee * y;
                gAz[i] = Ee * z;

                Yn += gAy[i];

            }

            gYnNorm = (Yn > 0.0f) ? Yn : 1.0f;
            gInvYn = 1.0f / gYnNorm;
            gSPDInit.store(false, std::memory_order_release);
        }

    } // namespace detail

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

        detail::precompute_spectral_tables_locked_body();
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
        for (int i = 0; i < K; ++i) {
            const float l = T.lambda[i];

            const float ey = hasEpsY ? epsY.linear[i] : eps_yellow(l);
            const float em = hasEpsM ? epsM.linear[i] : eps_magenta(l);
            const float ec = hasEpsC ? epsC.linear[i] : eps_cyan(l);
            T.epsY[i] = ey;
            T.epsM[i] = em;
            T.epsC[i] = ec;

            T.Xbar[i] = (!xbar.linear.empty() && (int)xbar.linear.size() == K) ? xbar.linear[i] : cie_xbar(l);
            T.Ybar[i] = (!ybar.linear.empty() && (int)ybar.linear.size() == K) ? ybar.linear[i] : cie_ybar(l);
            T.Zbar[i] = (!zbar.linear.empty() && (int)zbar.linear.size() == K) ? zbar.linear[i] : cie_zbar(l);

            const float Ee = hasIll ? illumView.linear[i] : 1.0f;
            T.illum[i] = Ee;
            T.Ax[i] = Ee * T.Xbar[i];
            T.Ay[i] = Ee * T.Ybar[i];
            T.Az[i] = Ee * T.Zbar[i];
            Yn += T.Ay[i];
            sumAx += T.Ax[i];
            sumAy += T.Ay[i];
            sumAz += T.Az[i];
        }
        T.invYn = (Yn > 0.0) ? (1.0f / (float)Yn) : 1.0f;
        const double scale = static_cast<double>(T.invYn);
        T.whiteXYZ[0] = static_cast<float>(scale * sumAx);
        T.whiteXYZ[1] = static_cast<float>(scale * sumAy);
        T.whiteXYZ[2] = static_cast<float>(scale * sumAz);

        // Illuminant white point used for chromatic adaptation (normalized to Y=1).
        T.refIllumWhiteXYZ[0] = T.whiteXYZ[0];
        T.refIllumWhiteXYZ[1] = T.whiteXYZ[1];
        T.refIllumWhiteXYZ[2] = T.whiteXYZ[2];

        T.hasBaseline = hasBaseline &&
            (int)baseMin.linear.size() == K;
        T.baseMin.assign(K, 0.0f);
        T.baseMid.assign(K, 0.0f);
        if (T.hasBaseline) {
            for (int i = 0; i < K; ++i) {
                T.baseMin[i] = baseMin.linear[i];
            }
            if ((int)baseMid.linear.size() == K) {
                for (int i = 0; i < K; ++i) {
                    T.baseMid[i] = baseMid.linear[i];
                }
            }
            T.baselineMixReference = 0.0f;
        }
        else {
            T.baselineMixReference = 0.0f;
        }

        T.illuminantHash = illuminantHash;
        if (T.illuminantHash == 0 && hasIll &&
            static_cast<int>(illumView.linear.size()) == K) {
            const Hash::FloatSpanHash h = Hash::hash_float_span_with_nan_mask(
                illumView.linear.data(), illumView.linear.size());
            const std::uint64_t fields[] = { h.valueHash, h.nanMaskHash };
            T.illuminantHash = Hash::hash_bytes(fields, sizeof(fields));
        }

        auto hash_vec = [](const std::vector<float>& v) -> std::uint64_t {
            const Hash::FloatSpanHash h = Hash::hash_float_span_with_nan_mask(v.data(), v.size());
            const std::uint64_t fields[] = { h.valueHash, h.nanMaskHash };
            return Hash::hash_bytes(fields, sizeof(fields));
            };
        auto hash_scalar = [](float v) -> std::uint64_t {
            const Hash::FloatSpanHash h = Hash::hash_float_span_with_nan_mask(&v, 1);
            const std::uint64_t fields[] = { h.valueHash, h.nanMaskHash };
            return Hash::hash_bytes(fields, sizeof(fields));
            };
        auto hash_array3 = [](const float v[3]) -> std::uint64_t {
            const Hash::FloatSpanHash h = Hash::hash_float_span_with_nan_mask(v, 3);
            const std::uint64_t fields[] = { h.valueHash, h.nanMaskHash };
            return Hash::hash_bytes(fields, sizeof(fields));
            };
        const std::uint64_t tableFields[] = {
            T.illuminantHash,
            hash_vec(T.lambda),
            hash_scalar(T.deltaLambda),
            hash_scalar(T.invYn),
            hash_array3(T.whiteXYZ),
            hash_array3(T.refIllumWhiteXYZ),
            hash_vec(T.Ax),
            hash_vec(T.Ay),
            hash_vec(T.Az),
            hash_vec(T.illum),
            hash_vec(T.Xbar),
            hash_vec(T.Ybar),
            hash_vec(T.Zbar),
            hash_vec(T.epsC),
            hash_vec(T.epsM),
            hash_vec(T.epsY),
            hash_vec(T.baseMin),
            hash_vec(T.baseMid),
            hash_scalar(T.baselineMixReference),
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

        auto sanitize_component = [](float v) -> float {
            if (!std::isfinite(v)) {
                return 0.0f;
            }
            return std::max(0.0f, v);
            };

        float sanitizedXYZ[3] = {
            sanitize_component(XYZ[0]),
            sanitize_component(XYZ[1]),
            sanitize_component(XYZ[2])
        };

        float refWhite[3] = {
            sanitize_component(T.refIllumWhiteXYZ[0]),
            sanitize_component(T.refIllumWhiteXYZ[1]),
            sanitize_component(T.refIllumWhiteXYZ[2])
        };
        if (refWhite[1] <= 0.0f) {
            refWhite[0] = gDWG_WhitePoint_XYZ[0];
            refWhite[1] = gDWG_WhitePoint_XYZ[1];
            refWhite[2] = gDWG_WhitePoint_XYZ[2];
        }

        float adaptedXYZ[3];
        chromatic_adapt_XYZ_CAT02(sanitizedXYZ, gDWG_WhitePoint_XYZ, refWhite, adaptedXYZ);
        spd_probe_record_cat02(sanitizedXYZ, adaptedXYZ, gDWG_WhitePoint_XYZ, refWhite);

        adaptedXYZ[0] = std::max(0.0f, adaptedXYZ[0]);
        adaptedXYZ[1] = std::max(0.0f, adaptedXYZ[1]);
        adaptedXYZ[2] = std::max(0.0f, adaptedXYZ[2]);

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
        double Y_recon = 0.0;
        for (int i = 0; i < K; ++i) {
            const float bx = std::max(0.0f, T.Ax[i]);
            const float by = std::max(0.0f, T.Ay[i]);
            const float bz = std::max(0.0f, T.Az[i]);
            const float Ei = std::max(1e-6f, cx * bx + cy * by + cz * bz);
            Ee_out[i] = Ei;
            Y_recon += static_cast<double>(Ei) * static_cast<double>(T.Ay[i]);
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
        for (size_t i = 0; i < Ee.size(); ++i) {
            const double E = Ee[i];
            X += E * xbar.linear[i];
            Y += E * ybar.linear[i];
            Z += E * zbar.linear[i];
        }

        // Compute Δλ from the global spectral shape
        const float deltaLambda = (Spectral::gShape.lambdaMax - Spectral::gShape.lambdaMin)
            / float(Spectral::gShape.K - 1);
        const float s = Spectral::gInvYn;

        XYZ[0] = static_cast<float>(X * s);
        XYZ[1] = static_cast<float>(Y * s);
        XYZ[2] = static_cast<float>(Z * s);
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
        for (int i = 0; i < K; ++i) {
            const float e = (i < N) ? Ee[i] : 0.0f;
            X += static_cast<double>(e) * static_cast<double>(T.Xbar[i]);
            Y += static_cast<double>(e) * static_cast<double>(T.Ybar[i]);
            Z += static_cast<double>(e) * static_cast<double>(T.Zbar[i]);
        }
        const float s = T.invYn;
        XYZ[0] = static_cast<float>(X * s);
        XYZ[1] = static_cast<float>(Y * s);
        XYZ[2] = static_cast<float>(Z * s);
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
        float num = 0.0f, den = 0.0f;
        for (int i = 0; i < gShape.K; ++i) {
            const float Ee = gIllumTable[i];
            const float s = c.sample(gLambda[i]);
            num += Ee * s;
            den += Ee;
        }
        return (den > 0.0f) ? (num / den) : 1.0f;
    }

} // namespace Spectral
