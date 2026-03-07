#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include "FilmProcessing.h"
#include "SpectralData.h"
#include "ParamNames.h"
#include "ofxImageEffect.h"
#include "ofxProperty.h"
#include "ofxParam.h"

// Couplers.h — REPLACE the entire stub namespace block under #ifndef JUICER_ENABLE_COUPLERS with this
#ifndef JUICER_ENABLE_COUPLERS
// Stubs: compiled out safely
namespace Couplers {
    // Keep identical signatures so call sites compile either way
    struct ApplyInput { float E[3]; float D[3]; };
    struct Runtime {
        bool  active = false;
        float M[3][3] = { {0,0,0},{0,0,0},{0,0,0} };
        float highShift = 0.0f;
        float dMax[3] = { 1.0f, 1.0f, 1.0f };
        float spatialSigmaMicrometers = 0.0f;
        float spatialSigmaPixels = 0.0f; // ABI parity with enabled build
    };

    inline void define_params(OfxImageEffectHandle, OfxImageEffectSuiteV1*, OfxPropertySuiteV1*, OfxParameterSuiteV1*) {}
    inline void on_param_changed(const char*) {}

    // IMPORTANT: keep the exact same signature as enabled build
    inline void maybe_precorrect_curves(OfxImageEffectHandle, OfxImageEffectSuiteV1*, OfxParameterSuiteV1*) {}

    inline void fetch_runtime(const OfxImageEffectHandle, OfxImageEffectSuiteV1*, OfxPropertySuiteV1*, OfxParameterSuiteV1*, Runtime&) {}
    inline void apply_runtime(ApplyInput&, const Runtime&) {}

    // Optional parity helpers used by the print/scanner paths
    struct ApplyInputLogE { float logE[3]; float D[3]; };
    inline void apply_runtime_logE(ApplyInputLogE&, const Runtime&) {}
    inline void apply_runtime_logE_with_curves(
        ApplyInputLogE&, const Runtime&,
        const Spectral::Curve&, const Spectral::Curve&, const Spectral::Curve&) {
    }

} // namespace Couplers

#else

namespace Couplers {

    // ------------------------------
    // OFX parameter names
    // ------------------------------
    static constexpr const char* kParamCouplersActive = "CouplersActive";
    static constexpr const char* kParamCouplersGroup = "Couplers";
    static constexpr const char* kParamCouplersAmount = "CouplersAmount";
    static constexpr const char* kParamCouplersAmountB = "CouplersAmountB";
    static constexpr const char* kParamCouplersAmountG = "CouplersAmountG";
    static constexpr const char* kParamCouplersAmountR = "CouplersAmountR";
    static constexpr const char* kParamCouplersLayerSigma = "CouplersLayerDiffusion";
    static constexpr const char* kParamCouplersHighExpShift = "CouplersHighExposureShift";
    static constexpr const char* kParamCouplersSpatialSigma = "CouplersSpatialDiffusion"; // micrometers

    inline float spatial_sigma_pixels_from_micrometers(float sigmaUm, double filmLongEdgeMm,
        double widthPixels, double heightPixels)
    {
        if (!(std::isfinite(sigmaUm)) || sigmaUm <= 0.0f) return 0.0f;
        if (!(std::isfinite(filmLongEdgeMm)) || filmLongEdgeMm <= 0.0) return 0.0f;

        const double longEdgePixels = std::max(widthPixels, heightPixels);
        if (!(std::isfinite(longEdgePixels)) || longEdgePixels <= 0.0) return 0.0f;

        const double filmLongEdgeUm = filmLongEdgeMm * 1000.0;
        if (!(std::isfinite(filmLongEdgeUm)) || filmLongEdgeUm <= 0.0) return 0.0f;

        const double pixelPitchUm = filmLongEdgeUm / longEdgePixels;
        if (!(std::isfinite(pixelPitchUm)) || pixelPitchUm <= 0.0) return 0.0f;

        double sigmaPixels = static_cast<double>(sigmaUm) / pixelPitchUm;
        if (!(std::isfinite(sigmaPixels)) || sigmaPixels <= 0.0) return 0.0f;
        if (sigmaPixels > 25.0) sigmaPixels = 25.0;
        return static_cast<float>(sigmaPixels);
    }


    // Orientation: M[inputLayer][outputLayer]. Input layers are ordered [B,G,R],
    // output axes correspond to [Y,M,C] corrections applied to Blue, Green, Red layer logE respectively.
    // Rows are normalized diffusion weights across outputs; each row is scaled by the per-input-layer amount.
    // Build 3x3 DIR matrix (layer-space diffusion, per-row scaling)
    inline void build_dir_matrix(float M[3][3], const float amountRGB[3], float layerSigma) {
        // Sanitize inputs
        const float sigma = std::isfinite(layerSigma) ? std::max(0.0f, layerSigma) : 0.0f;
        float amt[3] = { amountRGB[0], amountRGB[1], amountRGB[2] };
        for (int i = 0; i < 3; ++i) {
            if (!std::isfinite(amt[i])) amt[i] = 0.0f;
            if (amt[i] < 0.0f) amt[i] = 0.0f;
        }

        auto gauss = [&](int dx)->float {
            if (sigma <= 0.0f) return (dx == 0) ? 1.0f : 0.0f;
            const float s2 = sigma * sigma;
            return std::exp(-0.5f * (dx * dx) / s2);
            };
        for (int r = 0; r < 3; ++r) {
            float row[3]; float wsum = 0.0f;
            for (int c = 0; c < 3; ++c) { row[c] = gauss(c - r); wsum += row[c]; }
            if (wsum > 0.0f) for (int c = 0; c < 3; ++c) row[c] /= wsum;
            for (int c = 0; c < 3; ++c) M[r][c] = amt[r] * row[c];
        }

        // Scrub non-finite entries to ensure runtime stability
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                if (!std::isfinite(M[r][c])) M[r][c] = 0.0f;
            }
        }
    }

    // Pre-correct 1D density curves (before DIR) — agx parity
    inline void precorrect_density_curves_before_DIR_into(
        const float M[3][3], float highExpShift,
        const Spectral::Curve& inB, const Spectral::Curve& inG, const Spectral::Curve& inR,
        Spectral::Curve& outB, Spectral::Curve& outG, Spectral::Curve& outR)
    {
        using Spectral::Curve;

        // Copy-through by default
        outB = inB;
        outG = inG;
        outR = inR;

        const size_t N = std::min({ inB.lambda_nm.size(), inG.lambda_nm.size(), inR.lambda_nm.size() });
        if (N == 0) return;

        // Identity early-out if DIR matrix is exactly zero (agx parity; fixes ZeroMatrix test)
        bool zeroM = true;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                if (M[r][c] != 0.0f) { zeroM = false; break; }
        if (zeroM) return;

        // Original grids + densities
        std::vector<float> xB(N), xG(N), xR(N);
        std::vector<float> dY(N), dM(N), dC(N);
        for (size_t i = 0; i < N; ++i) {
            xB[i] = inB.lambda_nm[i]; dY[i] = inB.linear[i];
            xG[i] = inG.lambda_nm[i]; dM[i] = inG.linear[i];
            xR[i] = inR.lambda_nm[i]; dC[i] = inR.linear[i];
        }

        // Keep immutable query copies to prevent OOB after dedup (fixes crash)
        const std::vector<float> qB = xB, qG = xG, qR = xR;

        // Monotonicity guard (non-decreasing)
        auto mono = [](const std::vector<float>& X)->bool {
            if (X.empty() || !std::isfinite(X.front()) || !std::isfinite(X.back())) return false;
            for (size_t i = 1; i < X.size(); ++i) {
                if (!std::isfinite(X[i]) || X[i] < X[i - 1]) return false;
            }
            return true;
            };
        if (!(mono(qB) && mono(qG) && mono(qR))) return; // leave untouched

        // Normalize per-channel with optional high exposure quadratic boost
        auto vmax = [](const std::vector<float>& v)->float {
            float m = 0.0f; for (float t : v) if (std::isfinite(t)) m = std::max(m, t);
            return (m > 0.0f) ? m : 1.0f;
            };
        const float yMax = vmax(dY), mMax = vmax(dM), cMax = vmax(dC);
        auto safe_div = [](float a, float b)->float {
            if (!std::isfinite(a)) a = 0.0f;
            if (!(b > 1e-4f) || !std::isfinite(b)) b = 1.0f;
            return a / b;
            };
        auto boost = [highExpShift](float t)->float {
            // keep non-negative; agx-style quadratic shift
            t = std::clamp(t, 0.0f, 1.0f);
            float tb = t + highExpShift * t * t;
            if (!std::isfinite(tb)) tb = t;
            if (tb < 0.0f) tb = 0.0f;
            return tb;
            };

        std::vector<float> nB(N), nG(N), nR(N);
        for (size_t i = 0; i < N; ++i) {
            nB[i] = boost(safe_div(dY[i], yMax));
            nG[i] = boost(safe_div(dM[i], mMax));
            nR[i] = boost(safe_div(dC[i], cMax));
        }

        // Build shifted logE axes (pre-DIR exposures)
        std::vector<float> xShiftB(N), xShiftG(N), xShiftR(N);
        for (size_t i = 0; i < N; ++i) {
            const float aY = M[0][0] * nB[i] + M[1][0] * nG[i] + M[2][0] * nR[i];
            const float aM = M[0][1] * nB[i] + M[1][1] * nG[i] + M[2][1] * nR[i];
            const float aC = M[0][2] * nB[i] + M[1][2] * nG[i] + M[2][2] * nR[i];
            xShiftB[i] = qB[i] - aY;
            xShiftG[i] = qG[i] - aM;
            xShiftR[i] = qR[i] - aC;
        }

        auto sanitize_shifted_axis = [](std::vector<float>& X, std::vector<float>& Y)->bool {
            if (X.size() != Y.size() || X.empty()) return false;
            if (!std::isfinite(X.front()) || !std::isfinite(X.back())) return false;
            bool monotonic = true;
            for (size_t i = 1; i < X.size(); ++i) {
                if (!std::isfinite(X[i])) return false;
                if (X[i] < X[i - 1]) { monotonic = false; break; }
            }
            if (monotonic) return true; // already non-decreasing

            std::vector<size_t> order(X.size());
            std::iota(order.begin(), order.end(), size_t{ 0 });
            std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                return X[a] < X[b];
                });

            std::vector<float> Xsorted(X.size());
            std::vector<float> Ysorted(Y.size());
            float prev = X[order[0]];
            if (!std::isfinite(prev)) return false;
            Xsorted[0] = prev;
            Ysorted[0] = Y[order[0]];
            for (size_t k = 1; k < order.size(); ++k) {
                float xv = X[order[k]];
                if (!std::isfinite(xv)) return false;
                if (xv < prev) xv = prev; // enforce non-decreasing order
                Xsorted[k] = xv;
                Ysorted[k] = Y[order[k]];
                prev = xv;
            }
            X.swap(Xsorted);
            Y.swap(Ysorted);
            return true;
            };
        if (!sanitize_shifted_axis(xShiftB, dY)) return;
        if (!sanitize_shifted_axis(xShiftG, dM)) return;
        if (!sanitize_shifted_axis(xShiftR, dC)) return;

        // Dedup STRICT on interpolation data ONLY (not on the query grids)
        auto dedup_strict = [](const std::vector<float>& X, const std::vector<float>& Y,
            std::vector<float>& Xo, std::vector<float>& Yo) {
                Xo.clear(); Yo.clear();
                if (X.size() != Y.size() || X.empty()) return;
                Xo.reserve(X.size()); Yo.reserve(Y.size());
                float lastX = X[0];
                float lastY = std::isfinite(Y[0]) ? std::max(0.0f, Y[0]) : 0.0f;
                Xo.push_back(lastX); Yo.push_back(lastY);
                const float eps = 1e-6f;
                for (size_t i = 1; i < X.size(); ++i) {
                    float xi = X[i];
                    float yi = std::isfinite(Y[i]) ? std::max(0.0f, Y[i]) : 0.0f;
                    if (!std::isfinite(xi)) continue;
                    if (xi <= lastX + eps) {
                        // merge duplicates by keeping max density (monotone)
                        Yo.back() = std::max(Yo.back(), yi);
                    }
                    else {
                        Xo.push_back(xi);
                        Yo.push_back(yi);
                        lastX = xi;
                    }
                }
            };

        std::vector<float> XB, YB, XG, YG, XR, YR;
        dedup_strict(xShiftB, dY, XB, YB);
        dedup_strict(xShiftG, dM, XG, YG);
        dedup_strict(xShiftR, dC, XR, YR);
        if (XB.size() < 2 || XG.size() < 2 || XR.size() < 2) return; // nothing reliable to do

        // Inclusive clamp (NO inward bias; agx parity; fixes identity expectations)
        auto clamp_warp = [](float xq, float xmin, float xmax)->float {
            if (!std::isfinite(xq)) return xmin;
            return std::min(std::max(xq, xmin), xmax);
            };

        // Interpolator on dedup'ed data
        auto interp = [](const std::vector<float>& X, const std::vector<float>& Y, float xq)->float {
            const size_t N = X.size();
            if (N == 0 || Y.size() != N) return 0.0f;
            const float xmin = X.front(), xmax = X.back();
            if (!std::isfinite(xq)) xq = xmin;
            if (xq <= xmin) return Y.front();
            if (xq >= xmax) return Y.back();
            size_t i1 = size_t(std::lower_bound(X.begin(), X.end(), xq) - X.begin());
            if (i1 == 0) return Y.front();
            if (i1 >= N) return Y.back();
            size_t i0 = i1 - 1;
            float x0 = X[i0], x1 = X[i1];
            float y0 = Y[i0], y1 = Y[i1];
            if (!std::isfinite(x0) || !std::isfinite(x1) || !std::isfinite(y0) || !std::isfinite(y1)) {
                return std::isfinite(y0) ? y0 : 0.0f;
            }
            const float denom = (x1 - x0);
            if (!(denom > 0.0f) || !std::isfinite(denom)) return y0;
            const float t = (xq - x0) / denom;
            return y0 + t * (y1 - y0);
            };

        // Apply inverse warp: interpolate post-DIR densities over shifted axes at original grids
        std::vector<float> dYcorr(N), dMcorr(N), dCcorr(N);
        for (size_t i = 0; i < N; ++i) {
            const float xqY = clamp_warp(qB[i], XB.front(), XB.back());
            const float xqM = clamp_warp(qG[i], XG.front(), XG.back());
            const float xqC = clamp_warp(qR[i], XR.front(), XR.back());

            dYcorr[i] = interp(XB, YB, xqY);
            dMcorr[i] = interp(XG, YG, xqM);
            dCcorr[i] = interp(XR, YR, xqC);
        }
        // Enforce non-negative, monotone (non-decreasing) outputs to match agx behavior and tests
        auto sanitize_monotone_nonneg = [](std::vector<float>& v) {
            if (v.empty()) return;
            float prev = std::isfinite(v[0]) ? std::max(0.0f, v[0]) : 0.0f;
            v[0] = prev;
            for (size_t i = 1; i < v.size(); ++i) {
                float cur = std::isfinite(v[i]) ? std::max(0.0f, v[i]) : 0.0f;
                if (cur < prev) cur = prev;      // enforce monotone
                v[i] = cur;
                prev = cur;
            }
            };
        sanitize_monotone_nonneg(dYcorr);
        sanitize_monotone_nonneg(dMcorr);
        sanitize_monotone_nonneg(dCcorr);

        // Write back to outputs (sizes unchanged)
        for (size_t i = 0; i < N; ++i) {
            outB.linear[i] = dYcorr[i];
            outG.linear[i] = dMcorr[i];
            outR.linear[i] = dCcorr[i];
        }
    }

    // Define OFX params (no globals)
    inline void define_params(OfxImageEffectHandle effect,
        OfxImageEffectSuiteV1* effectSuite,
        OfxPropertySuiteV1* propSuite,
        OfxParameterSuiteV1* paramSuite) {
        OfxParamSetHandle ps = nullptr;
        effectSuite->getParamSet(effect, &ps);
        if (!ps) return;

        OfxPropertySetHandle p;

        OfxPropertySetHandle grp;
        paramSuite->paramDefine(ps, kOfxParamTypeGroup, kParamCouplersGroup, &grp);
        propSuite->propSetString(grp, kOfxPropLabel, 0, "DIR couplers");

        paramSuite->paramDefine(ps, kOfxParamTypeBoolean, kParamCouplersActive, &p);
        propSuite->propSetString(p, kOfxPropLabel, 0, "Active");
        propSuite->propSetInt(p, kOfxParamPropDefault, 0, 1);
        propSuite->propSetString(p, kOfxParamPropParent, 0, kParamCouplersGroup);

        paramSuite->paramDefine(ps, kOfxParamTypeDouble, kParamCouplersAmount, &p);
        propSuite->propSetString(p, kOfxPropLabel, 0, "Couplers amount");
        propSuite->propSetDouble(p, kOfxParamPropDefault, 0, 1.0);
        propSuite->propSetDouble(p, kOfxParamPropDisplayMin, 0, 0.0);
        propSuite->propSetDouble(p, kOfxParamPropDisplayMax, 0, 2.0);
        propSuite->propSetString(p, kOfxParamPropParent, 0, kParamCouplersGroup);

        paramSuite->paramDefine(ps, kOfxParamTypeDouble, kParamCouplersAmountR, &p);
        propSuite->propSetString(p, kOfxPropLabel, 0, "Couplers ratio R");
        propSuite->propSetDouble(p, kOfxParamPropDefault, 0, 1.0);
        propSuite->propSetDouble(p, kOfxParamPropDisplayMin, 0, 0.0);
        propSuite->propSetDouble(p, kOfxParamPropDisplayMax, 0, 1.0);
        propSuite->propSetString(p, kOfxParamPropParent, 0, kParamCouplersGroup);

        paramSuite->paramDefine(ps, kOfxParamTypeDouble, kParamCouplersAmountG, &p);
        propSuite->propSetString(p, kOfxPropLabel, 0, "Couplers ratio G");
        propSuite->propSetDouble(p, kOfxParamPropDefault, 0, 1.0);
        propSuite->propSetDouble(p, kOfxParamPropDisplayMin, 0, 0.0);
        propSuite->propSetDouble(p, kOfxParamPropDisplayMax, 0, 1.0);
        propSuite->propSetString(p, kOfxParamPropParent, 0, kParamCouplersGroup);

        paramSuite->paramDefine(ps, kOfxParamTypeDouble, kParamCouplersAmountB, &p);
        propSuite->propSetString(p, kOfxPropLabel, 0, "Couplers ratio B");
        propSuite->propSetDouble(p, kOfxParamPropDefault, 0, 1.0);
        propSuite->propSetDouble(p, kOfxParamPropDisplayMin, 0, 0.0);
        propSuite->propSetDouble(p, kOfxParamPropDisplayMax, 0, 1.0);
        propSuite->propSetString(p, kOfxParamPropParent, 0, kParamCouplersGroup);

        paramSuite->paramDefine(ps, kOfxParamTypeDouble, kParamCouplersLayerSigma, &p);
        propSuite->propSetString(p, kOfxPropLabel, 0, "Layer diffusion");
        propSuite->propSetDouble(p, kOfxParamPropDefault, 0, 2.0);
        propSuite->propSetDouble(p, kOfxParamPropDisplayMin, 0, 0.0);
        propSuite->propSetDouble(p, kOfxParamPropDisplayMax, 0, 4.0);
        propSuite->propSetString(p, kOfxParamPropParent, 0, kParamCouplersGroup);

        paramSuite->paramDefine(ps, kOfxParamTypeDouble, kParamCouplersHighExpShift, &p);
        propSuite->propSetString(p, kOfxPropLabel, 0, "High exposure shift");
        propSuite->propSetDouble(p, kOfxParamPropDefault, 0, 0.0);
        propSuite->propSetDouble(p, kOfxParamPropDisplayMin, 0, 0.0);
        propSuite->propSetDouble(p, kOfxParamPropDisplayMax, 0, 1.0);
        propSuite->propSetString(p, kOfxParamPropParent, 0, kParamCouplersGroup);

        // Spatial diffusion (xy Gaussian, micrometers on film long edge)
        paramSuite->paramDefine(ps, kOfxParamTypeDouble, kParamCouplersSpatialSigma, &p);
        propSuite->propSetString(p, kOfxPropLabel, 0, "Couplers spatial diffusion (\xC2\xB5m)");
        propSuite->propSetDouble(p, kOfxParamPropDefault, 0, 10.0);
        propSuite->propSetDouble(p, kOfxParamPropDisplayMin, 0, 0.0);
        propSuite->propSetDouble(p, kOfxParamPropDisplayMax, 0, 50.0);
        propSuite->propSetString(p, kOfxParamPropHint, 0,
            "Micrometers of DIR spatial diffusion; scaled using the Scanner film long edge parameter.");
        propSuite->propSetString(p, kOfxParamPropParent, 0, kParamCouplersGroup);

    }

    // Called from onParamChanged
    inline void on_param_changed(const char* name) {
        (void)name;
        // Coupler changes are consumed by WorkingState rebuild from ParamSnapshot.
        // Do not mutate process-global spectral precompute state from runtime param changes.
    }


    struct Runtime {
        bool  active = true;
        float M[3][3] = { {0,0,0},{0,0,0},{0,0,0} };
        float highShift = 0.0f;
        float dMax[3] = { 1.0f, 1.0f, 1.0f };
        float spatialSigmaMicrometers = 0.0f;
        float spatialSigmaPixels = 0.0f; // 0 = disabled, later used for optional xy Gaussian
    };


    inline void fetch_runtime(const OfxImageEffectHandle instance,
        OfxImageEffectSuiteV1* effectSuite,
        OfxPropertySuiteV1* propSuite,
        OfxParameterSuiteV1* paramSuite,
        Runtime& rt) {
        // Defaults
        rt.active = true;
        double amountSlider = 1.0;
        double ar = 1.0, ag = 1.0, ab = 1.0, sigma = 2.0, high = 0.0;
        double spatial = 10.0;
        double filmLongEdgeMm = 35.0;
        int activeInt = 1;

        OfxParamSetHandle ps = nullptr;
        effectSuite->getParamSet(instance, &ps);

        auto getD = [&](const char* nm, double& v) {
            OfxParamHandle h = nullptr;
            if (ps && paramSuite->paramGetHandle(ps, nm, &h, nullptr) == kOfxStatOK && h)
                paramSuite->paramGetValue(h, &v);
            };
        auto getB = [&](const char* nm, int& v) {
            OfxParamHandle h = nullptr;
            if (ps && paramSuite->paramGetHandle(ps, nm, &h, nullptr) == kOfxStatOK && h)
                paramSuite->paramGetValue(h, &v);
            };
        auto getOptionalDouble = [&](const char* nm, double& v) {
            OfxParamHandle h = nullptr;
            if (ps && paramSuite->paramGetHandle(ps, nm, &h, nullptr) == kOfxStatOK && h)
                paramSuite->paramGetValue(h, &v);
            };

        getB(kParamCouplersActive, activeInt);
        getD(kParamCouplersAmount, amountSlider);
        getD(kParamCouplersAmountR, ar);
        getD(kParamCouplersAmountG, ag);
        getD(kParamCouplersAmountB, ab);
        getD(kParamCouplersLayerSigma, sigma);
        getD(kParamCouplersHighExpShift, high);
        getD(kParamCouplersSpatialSigma, spatial);
        getOptionalDouble(JuicerParams::kCameraFilmFormatMm, filmLongEdgeMm);

        // Defensive fences (slider thrash can transiently produce NaN/Inf)
        auto f01 = [](double v)->float {
            if (!std::isfinite(v)) return 0.0f;
            if (v < 0.0) return 0.0f;
            if (v > 1.0) return 1.0f;
            return static_cast<float>(v);
            };
        auto fpos = [](double v, float maxv)->float {
            if (!std::isfinite(v) || v < 0.0) return 0.0f;
            return std::min(static_cast<float>(v), maxv);
            };
        auto fpos_uncapped = [](double v)->float {
            if (!std::isfinite(v) || v < 0.0) return 0.0f;
            float out = static_cast<float>(v);
            return std::isfinite(out) ? out : 0.0f;
            };

        // Spatial sigma (micrometers from UI; convert to pixels using film long edge)
        rt.spatialSigmaMicrometers = fpos(spatial, /*max*/ 50.0f);
        rt.spatialSigmaPixels = 0.0f;
        if (rt.spatialSigmaMicrometers > 0.0f) {
            double widthPixels = 0.0, heightPixels = 0.0;
            if (effectSuite && propSuite) {
                OfxPropertySetHandle effectProps = nullptr;
                if (effectSuite->getPropertySet(instance, &effectProps) == kOfxStatOK && effectProps) {
                    double projW = 0.0, projH = 0.0;
                    if (propSuite->propGetDouble(effectProps, kOfxImageEffectPropProjectSize, 0, &projW) == kOfxStatOK &&
                        propSuite->propGetDouble(effectProps, kOfxImageEffectPropProjectSize, 1, &projH) == kOfxStatOK) {
                        widthPixels = projW;
                        heightPixels = projH;
                    }
                    if ((widthPixels <= 0.0 || heightPixels <= 0.0)) {
                        double rod[4] = { 0.0, 0.0, 0.0, 0.0 };
                        bool haveRod = true;
                        for (int i = 0; i < 4; ++i) {
                            if (propSuite->propGetDouble(effectProps, kOfxImageEffectPropRegionOfDefinition, i, &rod[i]) != kOfxStatOK) {
                                haveRod = false;
                                break;
                            }
                        }
                        if (haveRod) {
                            widthPixels = rod[2] - rod[0];
                            heightPixels = rod[3] - rod[1];
                        }
                    }
                }
            }
            if (widthPixels > 0.0 && heightPixels > 0.0) {
                const double filmEdge =
                    (std::isfinite(filmLongEdgeMm) && filmLongEdgeMm > 0.0)
                    ? filmLongEdgeMm
                    : 36.0;
                rt.spatialSigmaPixels = spatial_sigma_pixels_from_micrometers(
                    rt.spatialSigmaMicrometers,
                    filmEdge,
                    widthPixels,
                    heightPixels);
            }
        }

        rt.active = (activeInt != 0);
        auto famp = [](double v)->float {
            if (!std::isfinite(v)) return 0.0f;
            if (v < 0.0) return 0.0f;
            if (v > 2.0) return 2.0f;
            return static_cast<float>(v);
            };

        const float ratio[3] = { f01(ab), f01(ag), f01(ar) };
        const float amountScale = famp(amountSlider);
        const float amount[3] = {
            amountScale * ratio[0],
            amountScale * ratio[1],
            amountScale * ratio[2]
        };
        const float sigmaF = fpos_uncapped(sigma);
        build_dir_matrix(rt.M, amount, sigmaF);
        rt.highShift = f01(high);


        // In enabled builds, dMax is supplied from per-instance WorkingState; avoid globals here.
        rt.dMax[0] = 1.0f;
        rt.dMax[1] = 1.0f;
        rt.dMax[2] = 1.0f;

    }

    struct ApplyInput {
        float E[3];   // layer exposures (B,G,R)
        float D[3];   // dye densities over B+F (Y,M,C)
    };

    inline void apply_runtime(ApplyInput& io, const Runtime& rt) {
        if (!rt.active) return;

        auto safe_norm = [](float D, float dmax)->float {
            float Din = (!std::isfinite(D) || D < 0.0f) ? 0.0f : D;
            float m = (std::isfinite(dmax) && dmax > 1e-6f) ? dmax : 1.0f;
            float n = Din / m;
            if (!std::isfinite(n) || n < 0.0f) n = 0.0f;
            return n;
            };

        float nB = safe_norm(io.D[0], rt.dMax[0]);
        float nG = safe_norm(io.D[1], rt.dMax[1]);
        float nR = safe_norm(io.D[2], rt.dMax[2]);
        nB += rt.highShift * nB * nB;
        nG += rt.highShift * nG * nG;
        nR += rt.highShift * nR * nR;

        const float aY = rt.M[0][0] * nB + rt.M[1][0] * nG + rt.M[2][0] * nR; // affects E_B
        const float aM = rt.M[0][1] * nB + rt.M[1][1] * nG + rt.M[2][1] * nR; // affects E_G
        const float aC = rt.M[0][2] * nB + rt.M[1][2] * nG + rt.M[2][2] * nR; // affects E_R

        const float sB = std::pow(10.0f, -aY);
        const float sG = std::pow(10.0f, -aM);
        const float sR = std::pow(10.0f, -aC);
        io.E[0] *= sB; io.E[1] *= sG; io.E[2] *= sR;
    }

    // LogE-domain couplers runtime (agx parity)
    struct ApplyInputLogE {
        float logE[3]; // per-layer log10 exposure, including any per-layer offsets used to sample curves
        float D[3];    // current layer densities B/G/R at those logE (used only to compute correction)
    };


    // Internal: compute ΔlogE (aY, aM, aC) from normalized densities
    // Note: M has shape [input layer (B,G,R)] x [output correction (Y,M,C)],
    // and corrections (aY,aM,aC) are subtracted from logE of Blue, Green, Red respectively,
    // matching agx-emulsion’s convention (input index = RGB, output index = per-layer correction).

    inline void compute_logE_corrections(const ApplyInputLogE& in, const Runtime& rt, float a_out[3]) {
        if (!rt.active) {
            a_out[0] = 0.0f;
            a_out[1] = 0.0f;
            a_out[2] = 0.0f;
            return;
        }
        auto safe_norm = [](float D, float dmax)->float {
            // Enforce non-negative inputs and robust maxima
            float Din = (!std::isfinite(D) || D < 0.0f) ? 0.0f : D;
            float m = (std::isfinite(dmax) && dmax > 1e-4f) ? dmax : 1.0f; // slightly higher floor
            float n = Din / m;
            if (!std::isfinite(n) || n < 0.0f) n = 0.0f;
            return n;
            };


        float nB = safe_norm(in.D[0], rt.dMax[0]);
        float nG = safe_norm(in.D[1], rt.dMax[1]);
        float nR = safe_norm(in.D[2], rt.dMax[2]);

        // high-exposure quadratic boost (agx: n += k * n^2)
        auto high_boost = [&](float n)->float {
            float nb = n + rt.highShift * n * n;
            if (!std::isfinite(nb)) {
                return (n >= 0.0f && std::isfinite(n)) ? n : 0.0f;
            }
            if (nb < 0.0f) nb = 0.0f;
            return nb;
            };
        nB = high_boost(nB);
        nG = high_boost(nG);
        nR = high_boost(nR);

        // M maps normalized densities [B,G,R] to per-layer logE corrections in [Y,M,C] axes
        float aY = rt.M[0][0] * nB + rt.M[1][0] * nG + rt.M[2][0] * nR; // affects Blue layer logE
        float aM = rt.M[0][1] * nB + rt.M[1][1] * nG + rt.M[2][1] * nR; // affects Green layer logE
        float aC = rt.M[0][2] * nB + rt.M[1][2] * nG + rt.M[2][2] * nR; // affects Red layer logE

        // Scrub non-finite corrections before bounding
        if (!std::isfinite(aY)) aY = 0.0f;
        if (!std::isfinite(aM)) aM = 0.0f;
        if (!std::isfinite(aC)) aC = 0.0f;

        // Bound corrections early to stabilize downstream interpolation and sampling.
        auto clamp_corr = [](float v)->float {
            if (!std::isfinite(v)) return 0.0f;
            if (v < -10.0f) return -10.0f;
            if (v > 10.0f)  return 10.0f;
            return v;
            };
        aY = clamp_corr(aY);
        aM = clamp_corr(aM);
        aC = clamp_corr(aC);

        a_out[0] = std::isfinite(aY) ? aY : 0.0f;
        a_out[1] = std::isfinite(aM) ? aM : 0.0f;
        a_out[2] = std::isfinite(aC) ? aC : 0.0f;
    }


    // Public: apply ΔlogE by subtraction (agx: log_raw_corrected = log_raw - correction)    
    inline void apply_runtime_logE_with_curves(
        ApplyInputLogE& io,
        const Runtime& rt,
        const Spectral::Curve& densB,
        const Spectral::Curve& densG,
        const Spectral::Curve& densR)
    {
        if (!rt.active) return;
        float a[3]; compute_logE_corrections(io, rt, a);
        for (float& v : a) {
            if (!std::isfinite(v)) v = 0.0f;
        }
        io.logE[0] -= a[0];
        io.logE[1] -= a[1];
        io.logE[2] -= a[2];

        auto clamp_to = [](float le, const Spectral::Curve& c)->float {
            if (c.lambda_nm.empty()) return le;
            const float xmin = c.lambda_nm.front();
            const float xmax = c.lambda_nm.back();
            if (!std::isfinite(le)) return xmin;
            return std::min(std::max(le, xmin), xmax);
            };
        io.logE[0] = clamp_to(io.logE[0], densB);
        io.logE[1] = clamp_to(io.logE[1], densG);
        io.logE[2] = clamp_to(io.logE[2], densR);
    }



} // namespace Couplers

#endif // JUICER_ENABLE_COUPLERS
