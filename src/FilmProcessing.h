// FilmProcessing.h
// Film emulation operations: exposure computation, density curves, and masking
#pragma once

#include <cmath>
#include <algorithm>
#include <vector>
#include <array>
#include <limits>
#include "SpectralData.h"
#include "SpectralProcessing.h"

namespace Spectral {

    // =========================================================================
    // 1. NEGATIVE COUPLER PARAMETERS (~50 lines)
    // =========================================================================

    struct NegativeCouplerParams {
        // Per-dye maximum density (dynamic range before base; tweak per stock)
        float DmaxY, DmaxM, DmaxC;
        // Base (orange mask) densities
        float baseY, baseM, baseC;
        // Per-layer curve steepness (k) for log1p mapping
        float kB, kG, kR;
        // 3x3 masking matrix applied to [dY0, dM0, dC0] (post H-D)
        // Row-major: [ YY YM YC; MY MM MC; CY CM CC ]
        float mask[9];
        // Per-channel spectral masking adjustments (multiplicative + subtractive)
        float maskScale[3];
        float maskOffset[3];
    };

    // Default prototype negative (placeholder values; tune later)
    inline NegativeCouplerParams make_default_neg_params() {
        return {
            // Dmax per dye (kept conservative for now)
            1.10f, 1.00f, 1.30f,
            // Base mask densities (preserve your current baseline look)
            0.04f, 0.02f, 0.06f,
            // H-D curve steepness (larger = quicker rise toward Dmax)
            6.0f, 6.0f, 6.0f,
            // Masking (small off-diagonals; close to your current values)
            {
                 0.98f, -0.06f, -0.02f,
                -0.03f,  0.98f, -0.05f,
                -0.02f, -0.04f,  0.98f
            },
            // Spectral masking defaults
            { 1.0f, 1.0f, 1.0f },
            { 0.0f, 0.0f, 0.0f }
        };
    }

    // Simple H-D-like curve: monotonic, saturating, smooth
    inline float hd_curve(float E, float Dmax, float k) {
        E = std::max(0.0f, E);
        if (k <= 0.0f) return std::min(Dmax, E);
        const float denom = std::log1p(k);
        if (denom <= 0.0f) return 0.0f;
        const float t = std::log1p(k * E) / denom; // 0..1 for E in 0..1 (approx), saturates smoothly
        return Dmax * std::max(0.0f, t);
    }

    // =========================================================================
    // 2. EXPOSURE COMPUTATION (~350 lines)
    // =========================================================================

    // -------------------------------------------------------------------------
    // Matrix-based exposure path (current default)
    // -------------------------------------------------------------------------
    inline void rgbDWG_to_layerExposures(const float rgbDWG[3], float E[3], float exposureScale) {
        const float r = std::max(0.0f, rgbDWG[0]);
        const float g = std::max(0.0f, rgbDWG[1]);
        const float b = std::max(0.0f, rgbDWG[2]);
        const float M[9] = {
            0.05f, 0.09f, 0.75f,
            0.10f, 0.80f, 0.15f,
            0.85f, 0.12f, 0.05f
        };
        E[0] = exposureScale * (M[0] * r + M[1] * g + M[2] * b);
        E[1] = exposureScale * (M[3] * r + M[4] * g + M[5] * b);
        E[2] = exposureScale * (M[6] * r + M[7] * g + M[8] * b);
        E[0] = std::max(0.0f, E[0]);
        E[1] = std::max(0.0f, E[1]);
        E[2] = std::max(0.0f, E[2]);
    }

    // -------------------------------------------------------------------------
    // SPD-based exposure path (global sensitivities)
    // -------------------------------------------------------------------------

    // Compute layer exposures E = [E_B, E_G, E_R] by integrating a scene SPD Ee[i]
    // with the measured layer sensitivity curves on the fixed grid.
    inline void layerExposures_from_sceneSPD(
        const std::vector<float>& Ee,
        float E[3],
        float exposureScale,
        bool applyDeltaLambda)
    {
        const int K = gShape.K;
        double Eb = 0.0, Eg = 0.0, Er = 0.0;
        spd_probe_record_delta_lambda(applyDeltaLambda);

        for (int i = 0; i < K; ++i) {
            const float e = (i < static_cast<int>(Ee.size())) ? Ee[i] : 0.0f;
            if (!std::isfinite(e)) {
                continue;
            }

            const float sB = gSensBlue.linear[i];
            const float sG = gSensGreen.linear[i];
            const float sR = gSensRed.linear[i];

            const double e64 = static_cast<double>(e);
            if (std::isfinite(sB)) {
                Eb += e64 * static_cast<double>(sB);
            }
            if (std::isfinite(sG)) {
                Eg += e64 * static_cast<double>(sG);
            }
            if (std::isfinite(sR)) {
                Er += e64 * static_cast<double>(sR);
            }
        }

        const float safeScale = std::max(0.0f, exposureScale);
        const double delta = applyDeltaLambda ? static_cast<double>(gDeltaLambda) : 1.0;
        const double dl = delta * static_cast<double>(safeScale);
        E[0] = std::max(0.0f, static_cast<float>(Eb * dl));
        E[1] = std::max(0.0f, static_cast<float>(Eg * dl));
        E[2] = std::max(0.0f, static_cast<float>(Er * dl));
    }

    // Integrate Ee with per-instance sensitivities curves (non-global)
    inline void layerExposures_from_sceneSPD_with_curves(
        const std::vector<float>& Ee,
        const Spectral::Curve& sB,
        const Spectral::Curve& sG,
        const Spectral::Curve& sR,
        float E[3],
        float exposureScale,
        bool applyDeltaLambda)
    {
        const int K = gShape.K;
        double Eb = 0.0, Eg = 0.0, Er = 0.0;
        spd_probe_record_delta_lambda(applyDeltaLambda);
        for (int i = 0; i < K; ++i) {
            const float e = (i < static_cast<int>(Ee.size())) ? Ee[i] : 0.0f;
            if (!std::isfinite(e)) {
                continue;
            }

            const double e64 = static_cast<double>(e);
            const float sb = sB.linear.empty() ? 0.0f : sB.linear[i];
            const float sg = sG.linear.empty() ? 0.0f : sG.linear[i];
            const float sr = sR.linear.empty() ? 0.0f : sR.linear[i];

            if (std::isfinite(sb)) {
                Eb += e64 * static_cast<double>(sb);
            }
            if (std::isfinite(sg)) {
                Eg += e64 * static_cast<double>(sg);
            }
            if (std::isfinite(sr)) {
                Er += e64 * static_cast<double>(sr);
            }
        }
        const float safeScale = std::max(0.0f, exposureScale);
        const double delta = applyDeltaLambda ? static_cast<double>(gDeltaLambda) : 1.0;
        const double dl = delta * static_cast<double>(safeScale);
        E[0] = std::max(0.0f, static_cast<float>(Eb * dl));
        E[1] = std::max(0.0f, static_cast<float>(Eg * dl));
        E[2] = std::max(0.0f, static_cast<float>(Er * dl));
    }

    // -------------------------------------------------------------------------
    // Flexible dispatch between matrix and SPD paths
    // -------------------------------------------------------------------------

    // Toggle: choose how to get E from DWG RGB
    inline bool gUseSPDExposure = false; // false = current 3x3 matrix path; true = reconstruct SPD path

    // Compute E from DWG via either matrix mapping (current path) or SPD integration (new path)
    inline void rgbDWG_to_layerExposures_flex(const float rgbDWG[3], float E[3], float exposureScale) {
        if (!gUseSPDExposure) {
            rgbDWG_to_layerExposures(rgbDWG, E, exposureScale);
            return;
        }

        thread_local std::vector<float> Ee_scene;
        Ee_scene.resize(gShape.K);

        // SPD path: reconstruct the scene SPD per pixel then integrate with sensitivities
        const bool useHanatos = hanatos_available() && hanatos_matches_reference_shape();
        if (useHanatos) {
            // Global path: use D65 white point (no chromatic adaptation) for backward compatibility
            reconstruct_Ee_from_DWG_RGB_hanatos(rgbDWG, Ee_scene, gDWG_WhitePoint_XYZ);
        }
        else {
            reconstruct_Ee_from_DWG_RGB(rgbDWG, Ee_scene);
        }
        layerExposures_from_sceneSPD(Ee_scene, E, exposureScale, !useHanatos);
    }

    // -------------------------------------------------------------------------
    // Reference balancing (non-global variant)
    // -------------------------------------------------------------------------

    // Non-global variant: balances sensitivities and shifts B/R density curve domains
    // so that at logE = 0 their densities match the green curve's density.
    // Inputs: sensB/G/R_in and densB/G/R_in; Output: sensB/G/R_out and densB/G/R_out.
    // illumRef must be pinned to current gShape.
    inline void balance_negative_under_reference_non_global(
        const Curve& illumRef,
        const Curve& sensB_in, const Curve& sensG_in, const Curve& sensR_in,
        const Curve& densB_in, const Curve& densG_in, const Curve& densR_in,
        Curve& sensB_out, Curve& sensG_out, Curve& sensR_out,
        Curve& densB_out, Curve& densG_out, Curve& densR_out)
    {
        const int K = gShape.K;
        if (K <= 0 || illumRef.linear.size() != static_cast<size_t>(K)) {
            // Fallback: shallow copies
            sensB_out = sensB_in; sensG_out = sensG_in; sensR_out = sensR_in;
            densB_out = densB_in; densG_out = densG_in; densR_out = densR_in;
            return;
        }

        // Copy inputs
        sensB_out = sensB_in; sensG_out = sensG_in; sensR_out = sensR_in;
        densB_out = densB_in; densG_out = densG_in; densR_out = densR_in;

        // 1) Neutral exposures under reference illuminant
        auto neutral_exposure = [&](const Curve& s)->double {
            if (s.linear.size() != static_cast<size_t>(K)) return 1.0;
            double n = 0.0, d = 0.0;
            for (int i = 0; i < K; ++i) {
                const double Ee = (double)illumRef.linear[i];
                const double ss = (double)s.linear[i];
                n += Ee * ss;
                d += Ee;
            }
            return (d > 0.0) ? n / d : 1.0;
            };
        const double nB = neutral_exposure(sensB_in);
        const double nG = neutral_exposure(sensG_in);
        const double nR = neutral_exposure(sensR_in);

        if (nB <= 1e-20 || nG <= 1e-20 || nR <= 1e-20) return;

        const float corrB = (float)(nG / nB);
        const float corrR = (float)(nG / nR);

        // 2) Apply sensitivity scaling (in-place on copies)
        if (sensB_out.linear.size() == static_cast<size_t>(K))
            for (int i = 0; i < K; ++i) sensB_out.linear[i] *= corrB;
        if (sensR_out.linear.size() == static_cast<size_t>(K))
            for (int i = 0; i < K; ++i) sensR_out.linear[i] *= corrR;

        // 3) Shift B/R density domains to align with G at logE=0
        auto interp_density_at = [](const Curve& c, float x)->float {
            const size_t n = c.lambda_nm.size();
            if (n == 0) return 0.0f;
            if (x <= c.lambda_nm.front()) return c.linear.front();
            if (x >= c.lambda_nm.back())  return c.linear.back();
            size_t i1 = 1; while (i1 < n && c.lambda_nm[i1] < x) ++i1;
            const size_t i0 = i1 - 1;
            const float x0 = c.lambda_nm[i0], x1 = c.lambda_nm[i1];
            const float y0 = c.linear[i0], y1 = c.linear[i1];
            const float t = (x - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
            };
        auto find_logE_for_density_local = [&](const Curve& c, float targetY)->float {
            const size_t n = c.lambda_nm.size();
            if (n < 2) return 0.0f;
            for (size_t i = 1; i < n; ++i) {
                const float x0 = c.lambda_nm[i - 1], x1 = c.lambda_nm[i];
                const float y0 = c.linear[i - 1], y1 = c.linear[i];
                if ((y0 <= targetY && targetY <= y1) || (y1 <= targetY && targetY <= y0)) {
                    const float t = (y1 != y0) ? (targetY - y0) / (y1 - y0) : 0.0f;
                    return x0 + t * (x1 - x0);
                }
            }
            const float d0 = std::abs(targetY - (c.linear.empty() ? 0.0f : c.linear.front()));
            const float d1 = std::abs(targetY - (c.linear.empty() ? 0.0f : c.linear.back()));
            return (d0 < d1) ? (c.lambda_nm.empty() ? 0.0f : c.lambda_nm.front())
                : (c.lambda_nm.empty() ? 0.0f : c.lambda_nm.back());
            };

        const float targetG = interp_density_at(densG_out, 0.0f);

        const float shiftB = -find_logE_for_density_local(densB_out, targetG);
        const float shiftR = -find_logE_for_density_local(densR_out, targetG);

        for (float& x : densB_out.lambda_nm) x += shiftB;
        for (float& x : densR_out.lambda_nm) x += shiftR;
    }

    // =========================================================================
    // 3. DENSITY OPERATIONS (~200 lines)
    // =========================================================================

    // -------------------------------------------------------------------------
    // Density curve sampling (agx-emulsion fast_interp parity)
    // -------------------------------------------------------------------------
    inline float sample_density_at_logE(const Curve& c, float logE, float gammaFactor = 1.0f) {
        const size_t n = c.lambda_nm.size();
        if (n == 0 || c.linear.size() != n) {
            return 0.0f;
        }

        // agx-emulsion parity: NaNs in density curves are authored toe samples. Preserve them
        // through interpolation (fast_interp semantics) and let them become 0 transmitted light later.
        if (!std::isfinite(logE)) {
            return std::numeric_limits<float>::quiet_NaN();
        }

        const float gammaSafe = (std::isfinite(gammaFactor) && gammaFactor > 0.0f)
            ? gammaFactor
            : 1.0f;

        // agx scales the curve axis by dividing xa by gamma_factor; with fixed xa, scale query instead.
        const float xq = logE * gammaSafe;

        size_t domainBegin = 0;
        while (domainBegin < n && !std::isfinite(c.lambda_nm[domainBegin])) {
            ++domainBegin;
        }
        if (domainBegin == n) {
            return 0.0f;
        }

        size_t domainEnd = n - 1;
        while (domainEnd > domainBegin && !std::isfinite(c.lambda_nm[domainEnd])) {
            --domainEnd;
        }

        const float xmin = c.lambda_nm[domainBegin];
        const float xmax = c.lambda_nm[domainEnd];
        if (!std::isfinite(xmin) || !std::isfinite(xmax) || !(xmax >= xmin)) {
            return c.linear[domainBegin];
        }

        // fast_interp endpoints: literal y[0]/y[-1], even if NaN.
        if (xq <= xmin) {
            return c.linear[domainBegin];
        }
        if (xq >= xmax) {
            return c.linear[domainEnd];
        }

        size_t i1 = domainBegin + 1;
        while (i1 <= domainEnd && c.lambda_nm[i1] < xq) {
            ++i1;
        }
        if (i1 > domainEnd) {
            return c.linear[domainEnd];
        }

        const size_t i0 = i1 - 1;
        const float x0 = c.lambda_nm[i0];
        const float x1 = c.lambda_nm[i1];
        const float y0 = c.linear[i0];
        const float y1 = c.linear[i1];

        const float denom = x1 - x0;
        if (!(denom > 0.0f) || !std::isfinite(denom)) {
            return y0;
        }

        const float t = (xq - x0) / denom;
        return y0 + t * (y1 - y0);
    }

    // -------------------------------------------------------------------------
    // Exposure → Dyes conversion
    // -------------------------------------------------------------------------

    // Map layer exposures (E_B, E_G, E_R) to dye densities (D_Y, D_M, D_C)
    // 1) H-D curve per layer: Y<-B, M<-G, C<-R
    // 2) Apply masking matrix
    // 3) Add base densities
    inline void exposures_to_dyes_with_params(const float E[3], float D[3], const NegativeCouplerParams& negParams) {
        D[0] = std::max(0.0f, hd_curve(E[0], negParams.DmaxY, negParams.kB)); // Y <- B
        D[1] = std::max(0.0f, hd_curve(E[1], negParams.DmaxM, negParams.kG)); // M <- G
        D[2] = std::max(0.0f, hd_curve(E[2], negParams.DmaxC, negParams.kR)); // C <- R
    }
    // =========================================================================
    // 4. BASELINE MANAGEMENT (~50 lines)
    // =========================================================================

    inline float baseline_density(float lambda, float w) {
        if (!gHasBaseline) return 0.0f;
        const float d0 = gBaseMin.sample(lambda);
        const float d1 = gBaseMid.sample(lambda);
        const float t = std::clamp(w, 0.0f, 1.0f);
        return d0 + t * (d1 - d0);
    }

    // =========================================================================
    // 5. DIR MASKING / INTER-LAYER EFFECTS
    // =========================================================================

    // -------------------------------------------------------------------------
    // Masking coupler matrix
    // -------------------------------------------------------------------------
    inline void apply_masking_adjustments_with_params(const NegativeCouplerParams& negParams, float D[3]) {
        float masked[3] = { 0.0f, 0.0f, 0.0f };
        for (int i = 0; i < 3; ++i) {
            float scale = (i < 3) ? negParams.maskScale[i] : 1.0f;
            if (!std::isfinite(scale) || scale <= 0.0f) {
                scale = 1.0f;
            }
            float offset = (i < 3) ? negParams.maskOffset[i] : 0.0f;
            if (!std::isfinite(offset) || offset < 0.0f) {
                offset = 0.0f;
            }
            float v = D[i] * scale - offset;
            if (!std::isfinite(v) || v < 0.0f) {
                v = 0.0f;
            }
            masked[i] = v;
        }

        auto safe_coeff = [&](int idx, float fallback) {
            if (idx < 0 || idx >= 9) {
                return fallback;
            }
            const float coeff = negParams.mask[idx];
            if (!std::isfinite(coeff)) {
                return fallback;
            }
            return coeff;
            };

        auto safe_base = [](float base) {
            if (!std::isfinite(base) || base < 0.0f) {
                return 0.0f;
            }
            return base;
            };

        const float m00 = safe_coeff(0, 1.0f);
        const float m01 = safe_coeff(1, 0.0f);
        const float m02 = safe_coeff(2, 0.0f);
        const float m10 = safe_coeff(3, 0.0f);
        const float m11 = safe_coeff(4, 1.0f);
        const float m12 = safe_coeff(5, 0.0f);
        const float m20 = safe_coeff(6, 0.0f);
        const float m21 = safe_coeff(7, 0.0f);
        const float m22 = safe_coeff(8, 1.0f);

        const float baseY = safe_base(negParams.baseY);
        const float baseM = safe_base(negParams.baseM);
        const float baseC = safe_base(negParams.baseC);

        const float y = m00 * masked[0] + m01 * masked[1] + m02 * masked[2] + baseY;
        const float m = m10 * masked[0] + m11 * masked[1] + m12 * masked[2] + baseM;
        const float c = m20 * masked[0] + m21 * masked[1] + m22 * masked[2] + baseC;

        auto clamp_density = [](float v) {
            if (!std::isfinite(v) || v < 0.0f) {
                return 0.0f;
            }
            return v;
            };

        D[0] = clamp_density(y);
        D[1] = clamp_density(m);
        D[2] = clamp_density(c);
    }

} // namespace Spectral
