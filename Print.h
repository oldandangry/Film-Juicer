// Print.h
#pragma once
#include <vector>
#include <string>
#include <array>
#include <cmath>
#include <filesystem>
#include <initializer_list>
#include <utility>
#include "SpectralData.h"
#include "SpectralProcessing.h"
#include "ColorTransforms.h"
#include "Illuminants.h"
#include "WorkingState.h"
#include "ProfileJSONLoader.h"
#include "AkimaInterpolator.h"
#include "Couplers.h"
#include <sstream>
#include <algorithm>
#include <limits>
#include "Logging.h"

// Forward-declare WorkingState to avoid header cycles
struct WorkingState;

namespace Scanner {
    struct Params;
}

namespace Print {

    inline std::string make_data_subpath(
        const std::string& baseDir,
        std::initializer_list<const char*> segments)
    {
        std::filesystem::path path(baseDir);
        for (const char* seg : segments) {
            if (seg && *seg) {
                path /= seg;
            }
        }
        path = path.lexically_normal();
        path.make_preferred();
        return path.string();
    }

    constexpr float kEnlargerSteps = 170.0f;
    constexpr float kDefaultNeutralY = 0.9f;
    constexpr float kDefaultNeutralM = 0.5f;
    constexpr float kDefaultNeutralC = 0.35f;

    struct DensityCurves {
        std::vector<std::pair<double, double>> cyan;
        std::vector<std::pair<double, double>> magenta;
        std::vector<std::pair<double, double>> yellow;
        bool usedJson = false;
    };

    struct Profile;

    // Forward declaration to allow use before the inline definition later in the file.
    void print_densities_from_Eprint(const Profile& p, const float Eprint[3], float D_print[3]);

    inline float blend_dichroic_filter_linear(float curveVal, float normalizedAmount) {
        const float c = std::isfinite(curveVal) ? curveVal : 1.0f;
        const float a = std::isfinite(normalizedAmount) ? normalizedAmount : 0.0f;
        return 1.0f - (1.0f - c) * a;
    }

    inline float clamp_logE_to_curve(const Spectral::Curve& curve, float logE) {
        const size_t n = curve.lambda_nm.size();
        if (n == 0) {
            return std::isfinite(logE) ? logE : 0.0f;
        }

        size_t domainBegin = 0;
        while (domainBegin < n && !std::isfinite(curve.lambda_nm[domainBegin])) {
            ++domainBegin;
        }
        if (domainBegin == n) {
            return std::isfinite(logE) ? logE : 0.0f;
        }

        size_t domainEnd = n - 1;
        while (domainEnd > domainBegin && !std::isfinite(curve.lambda_nm[domainEnd])) {
            --domainEnd;
        }

        const float xmin = curve.lambda_nm[domainBegin];
        const float xmax = curve.lambda_nm[domainEnd];
        if (!std::isfinite(xmin) || !std::isfinite(xmax)) {
            return std::isfinite(logE) ? logE : 0.0f;
        }

        if (!(xmax >= xmin)) {
            return std::isfinite(logE) ? logE : xmin;
        }

        const float xq = std::isfinite(logE) ? logE : xmin;
        return std::clamp(xq, xmin, xmax);
    }

    inline float compose_dichroic_amount(float neutralAmount, float deltaAmount) {
        const float neutral = std::isfinite(neutralAmount)
            ? std::clamp(neutralAmount, 0.0f, 1.0f)
            : 0.0f;
        float deltaSteps = std::isfinite(deltaAmount) ? deltaAmount : 0.0f;
        const float shiftLimit = kEnlargerSteps;
        deltaSteps = std::clamp(deltaSteps, -shiftLimit, shiftLimit);
        const float totalSteps = neutral * kEnlargerSteps + deltaSteps;
        return totalSteps / kEnlargerSteps;
    }

    struct Profile {
        Spectral::Curve epsC, epsM, epsY;     // print dye extinction (OD/λ)
        Spectral::Curve dcC, dcM, dcY;        // logE -> D (C,M,Y)
        Spectral::Curve baseMin, baseMid;     // optional print baseline (D-min/mid)
        bool hasBaseline = false;
        bool glareRemoved = false;
        bool hasGlareCompensation = false;
        float glareCompensationFactor = 0.0f;
        float glareCompensationDensity = 1.2f;
        float glareCompensationTransition = 0.3f;
        Profiles::ProfileGlare glare;
        float logEOffC = 0.0f, logEOffM = 0.0f, logEOffY = 0.0f; // legacy per-channel logE offsets (unused)
        std::array<float, 3> gammaFactor{ {1.0f, 1.0f, 1.0f} };

        // Optional neutral density target for mid-scale metameric patch (agx parity)
        bool hasMidNeutralDensity = false;
        std::array<float, 3> midNeutralDensity{ {
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN()
        } };
        bool hasMidNeutralLogE = false;
        std::array<float, 3> midNeutralLogE{ {
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN()
        } };
        bool hasSensitivityCorrection = false;
        std::array<float, 3> sensitivityNeutralCorr{ {1.0f, 1.0f, 1.0f} };
        std::string referenceIlluminant;
        std::string viewingIlluminant;

        // Print paper spectral log-sensitivity (R/G/B on disk => C/M/Y mapping), pinned to shape (log10 domain in CSVs)
        Spectral::Curve sensY_log, sensM_log, sensC_log;
    };


    struct Params {
        bool bypass = false;     // if true, bypass print (show negative)
        float exposure = 1.0f;   // enlarger exposure scalar
        float preflashExposure = 0.0f; // additional uniform print exposure (linear scale)
        float yFilter = 0.0f;    // delta from neutral baseline in Durst steps (±170)
        float mFilter = 0.0f;

        // Whether print exposure compensation is enabled in the UI.
        bool exposureCompensationEnabled = false;
        // Slider EV scale (2^EV) used for the mid-gray probe when compensation is enabled.
        float exposureCompensationScale = 1.0f;
    };


    struct Runtime {
        Profile profile;
        Spectral::Curve illumEnlarger; // pinned to gShape
        Spectral::Curve illumView;     // pinned to gShape
        std::string referenceIlluminant;
        std::string viewingIlluminant;
        Profiles::ProfileGlare glare;
        DensityCurves densityCurvesRaw;

        // New: dichroic filter transmittance curves (normalized 0..1, pinned to gShape)
        Spectral::Curve filterY;
        Spectral::Curve filterM;
        Spectral::Curve filterC;
        // Neutral baseline scalars (0..1) used for compensation probes
        float neutralY = kDefaultNeutralY;
        float neutralM = kDefaultNeutralM;
        float neutralC = kDefaultNeutralC;

        bool hasMidNeutralDensity = false;
        std::array<float, 3> midNeutralDensity{ {
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN()
        } };
        bool hasMidNeutralLogE = false;
        std::array<float, 3> midNeutralLogE{ {
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN()
        } };
        bool hasSensitivityCorrection = false;
        std::array<float, 3> sensitivityNeutralCorr{ {1.0f, 1.0f, 1.0f} };
    };

    // Profile validity: spectral-domain curves must match current spectral shape (K).
    // Density curves (dcC/M/Y) are in log-exposure domain, so they only need to be non-empty.
    inline bool profile_is_valid(const Profile& p) {
        const int K = Spectral::gShape.K;
        if (K <= 0) return false;

        auto spectral_curve_ok = [K](const Spectral::Curve& c) -> bool {
            return !c.linear.empty() && static_cast<int>(c.linear.size()) == K;
            };
        auto logE_curve_ok = [](const Spectral::Curve& c) -> bool {
            return !c.linear.empty();
            };

        // Require spectral print dye EPS
        if (!spectral_curve_ok(p.epsC) ||
            !spectral_curve_ok(p.epsM) ||
            !spectral_curve_ok(p.epsY)) {
            return false;
        }
        // Require print paper density curves (logE -> D)
        if (!logE_curve_ok(p.dcC) ||
            !logE_curve_ok(p.dcM) ||
            !logE_curve_ok(p.dcY)) {
            return false;
        }
        // Require print paper spectral log-sensitivities pinned to shape
        const bool sensOK =
            spectral_curve_ok(p.sensY_log) &&
            spectral_curve_ok(p.sensM_log) &&
            spectral_curve_ok(p.sensC_log);
        if (!sensOK) {
            return false;
        }

        if (p.hasBaseline) {
            if (!spectral_curve_ok(p.baseMin) || !spectral_curve_ok(p.baseMid)) {
                return false;
            }
        }
        return true;
    }
    bool remove_glare_compensation_from_curves(Profile& profile, DensityCurves& curves);
    bool rebuild_density_curves(Profile& profile, const DensityCurves& curves);
    void recompute_mid_neutral(Profile& profile, Runtime* runtime = nullptr);
    void load_profile_from_dir(const std::string& dir, Profile& out,
        const std::string& jsonProfilePath = std::string(), Runtime* runtime = nullptr);

    // Build an illuminant pinned to shape from choice. Choices align with your UI (0:D65,1:D55,2:D50,3:TH-KG3-L,4:T,5:K75P,6:Equal)
    inline void build_illuminant_from_choice(int choice, Runtime& rt, const std::string& dataDir, bool forEnlarger) {
        auto log_failure = [&](const char* label, const char* extra = nullptr) {
            std::ostringstream oss;
            oss << "failed to load " << label << " illuminant";
            if (extra) {
                oss << " (" << extra << ")";
            }
            oss << "; selection=" << choice;
            JTRACE("ILLUM", oss.str());
            };

        auto build_or_log = [&](auto builder, const char* label) -> Spectral::Curve {
            try {
                return builder();
            }
            catch (const std::exception& e) {
                log_failure(label, e.what());
            }
            catch (...) {
                log_failure(label, "unknown error");
            }
            return Spectral::Curve{};
            };

        Spectral::Curve c;
        switch (choice) {
        case 0:
            c = build_or_log([&]() {
                return Spectral::build_curve_D65_pinned(
                    make_data_subpath(dataDir, { "illuminants", "D65.csv" }));
                }, "D65");
            break;
        case 1:
            c = build_or_log([&]() {
                return Spectral::build_curve_D55_pinned(
                    make_data_subpath(dataDir, { "illuminants", "D55.csv" }));
                }, "D55");
            break;
        case 2:
            c = build_or_log([&]() {
                return Spectral::build_curve_D50_pinned(
                    make_data_subpath(dataDir, { "illuminants", "D50.csv" }));
                }, "D50");
            break;
        case 3:
            c = build_or_log([&]() {
                return Spectral::build_curve_TH_KG3_L_pinned(
                    make_data_subpath(dataDir, { "filters", "heat_absorbing", "schott", "KG3.csv" }),
                    make_data_subpath(dataDir, { "filters", "lens_transmission", "canon", "canon_24_f28_is.csv" }));
                }, "TH-KG3-L");
            break;
        case 4:
            c = build_or_log([&]() {
                return Spectral::build_curve_T_pinned(
                    make_data_subpath(dataDir, { "illuminants", "T.csv" }));
                }, "T");
            break;
        case 5:
            c = build_or_log([&]() {
                return Spectral::build_curve_K75P_pinned(
                    make_data_subpath(dataDir, { "illuminants", "K75P.csv" }));
                }, "K75P");
            break;
        case 6:
            c = Spectral::build_curve_equal_energy_pinned();
            break;
        default:
            log_failure("unknown choice");
            break;
        }

        if (forEnlarger) {
            rt.illumEnlarger = std::move(c);
        }
        else {
            rt.illumView = std::move(c);
        }
    }

    inline void load_dichroic_filters_from_csvs(
        const std::string& dirYMC, Runtime& rt)
    {
        auto load_pairs_silent = [](const std::string& path) {
            try { return Spectral::load_csv_pairs(path); }
            catch (...) { return std::vector<std::pair<float, float>>{}; }
            };

        // Try Durst Digital Light first
        std::string yPath = dirYMC + "filter_y.csv";
        std::string mPath = dirYMC + "filter_m.csv";
        std::string cPath = dirYMC + "filter_c.csv";

        auto y_pairs = load_pairs_silent(yPath);
        auto m_pairs = load_pairs_silent(mPath);
        auto c_pairs = load_pairs_silent(cPath);


        // If not found, try Edmund Optics / Thorlabs fallbacks (same filenames under their dirs)
        if (y_pairs.empty() || m_pairs.empty() || c_pairs.empty()) {
            std::string alt1 = dirYMC; // allow caller to pass different vendor dirs if desired
            // Identity fallback will be used below if still empty.
        }

        Spectral::assign_reference_axis(rt.filterY.lambda_nm);
        Spectral::assign_reference_axis(rt.filterM.lambda_nm);
        Spectral::assign_reference_axis(rt.filterC.lambda_nm);

        rt.filterY.linear.resize(Spectral::gShape.K);
        rt.filterM.linear.resize(Spectral::gShape.K);
        rt.filterC.linear.resize(Spectral::gShape.K);

        auto sample_curve = [](const std::vector<std::pair<float, float>>& pairs,
            Spectral::Curve& dst) {
                dst.linear.assign((size_t)Spectral::gShape.K, 1.0f);
                if (pairs.size() < 2) {
                    return;
                }
                std::vector<float> x, y;
                x.reserve(pairs.size());
                y.reserve(pairs.size());
                for (const auto& p : pairs) {
                    if (!std::isfinite(p.first) || !std::isfinite(p.second)) continue;
                    x.push_back(p.first);
                    y.push_back(p.second);
                }
                if (x.size() < 2) {
                    return;
                }
                const auto [minIt, maxIt] = std::minmax_element(x.begin(), x.end());
                const float xmin = *minIt;
                const float xmax = *maxIt;
                const size_t idxMin = static_cast<size_t>(std::distance(x.begin(), minIt));
                const size_t idxMax = static_cast<size_t>(std::distance(x.begin(), maxIt));
                const float ymin = y[idxMin];
                const float ymax = y[idxMax];
                Interpolation::AkimaInterpolator interp;
                if (!interp.build(x, y)) {
                    return;
                }
                for (int i = 0; i < Spectral::gShape.K; ++i) {
                    const float wl = Spectral::gShape.wavelengths[i];
                    float v = interp.evaluate(wl);
                    if (!std::isfinite(v)) {
                        // Clamp to the nearest measured endpoint to mirror SciPy Akima extrapolation.
                        v = (wl < xmin) ? ymin : ymax;
                    }
                    const float normalized = std::isfinite(v) ? (v * 0.01f) : 0.0f;
                    dst.linear[(size_t)i] = normalized;
                }
            };

        sample_curve(y_pairs, rt.filterY);
        sample_curve(m_pairs, rt.filterM);
        sample_curve(c_pairs, rt.filterC);

        // Diagnostics
        {
            std::ostringstream oss;
            oss << "DICHROICS loaded K=" << Spectral::gShape.K
                << " Y/M/C first="
                << (rt.filterY.linear.empty() ? -1.0f : rt.filterY.linear.front()) << "/"
                << (rt.filterM.linear.empty() ? -1.0f : rt.filterM.linear.front()) << "/"
                << (rt.filterC.linear.empty() ? -1.0f : rt.filterC.linear.front());
            JTRACE("PRINT", oss.str());
        }
    }



    // Compute negative transmittance T_neg(λ) from over-B+F densities D_neg (CMY order) using per-instance data
    inline void negative_T_from_dyes(const WorkingState& ws,
        const float D_neg[3],
        std::vector<float>& Tneg_out)
    {
        // Use the instance's spectral length; do NOT gate off gShape here.
        const int K = ws.tablesView.K;
        Tneg_out.assign((size_t)std::max(K, 0), 1.0f);
        if (K <= 0) return;

        // Per-instance baseline (if available)
        const bool hasBL = ws.hasBaseline &&
            (int)ws.baseMin.linear.size() == K;

        // Access per-wavelength epsilon for Y/M/C negative dyes; lambdas fallback to 0 if missing.
        auto epsY_at = [&](int i)->float {
            return (i < (int)ws.tablesView.epsY.size()) ? ws.tablesView.epsY[i] : 0.0f;
            };
        auto epsM_at = [&](int i)->float {
            return (i < (int)ws.tablesView.epsM.size()) ? ws.tablesView.epsM[i] : 0.0f;
            };
        auto epsC_at = [&](int i)->float {
            return (i < (int)ws.tablesView.epsC.size()) ? ws.tablesView.epsC[i] : 0.0f;
            };

        for (int i = 0; i < K; ++i) {
            if (static_cast<size_t>(i) < ws.tablesView.epsValid.size() &&
                ws.tablesView.epsValid[static_cast<size_t>(i)] == 0) {
                Tneg_out[i] = 0.0f;
                continue;
            }
            if (hasBL &&
                static_cast<size_t>(i) < ws.tablesView.baseMinValid.size() &&
                ws.tablesView.baseMinValid[static_cast<size_t>(i)] == 0) {
                Tneg_out[i] = 0.0f;
                continue;
            }

            const float baseSpectral = hasBL &&
                static_cast<size_t>(i) < ws.tablesView.baseMin.size()
                ? ws.tablesView.baseMin[static_cast<size_t>(i)]
                : 0.0f;

            const float Dlambda = D_neg[0] * epsC_at(i) // C
                + D_neg[1] * epsM_at(i) // M
                + D_neg[2] * epsY_at(i) // Y
                + baseSpectral;

            if (!std::isfinite(Dlambda)) {
                Tneg_out[i] = 0.0f;
                continue;
            }

            Tneg_out[i] = std::exp(-Spectral::kLn10 * Dlambda);
        }
    }


    // MVP: derive print channel exposures E_print[3] from Ee_expose(λ) using print dye extinctions as proxies.
    // Channels are stored in C/M/Y order to match agx-emulsion's contract.
    inline void exposures_for_print_channels(const Runtime& rt,
        const std::vector<float>& Ee_expose,
        float Eprint[3])
    {
        const int K = Spectral::gShape.K;
        double Ec = 0.0, Em = 0.0, Ey = 0.0;

        for (int i = 0; i < K; ++i) {
            const float e = Ee_expose[i];
            if (!std::isfinite(e)) {
                continue;
            }

            const float lc = rt.profile.epsC.linear.empty() ? 0.0f : rt.profile.epsC.linear[i];
            const float lm = rt.profile.epsM.linear.empty() ? 0.0f : rt.profile.epsM.linear[i];
            const float ly = rt.profile.epsY.linear.empty() ? 0.0f : rt.profile.epsY.linear[i];

            const double e64 = static_cast<double>(e);
            if (std::isfinite(lc)) {
                Ec += e64 * static_cast<double>(lc);
            }
            if (std::isfinite(lm)) {
                Em += e64 * static_cast<double>(lm);
            }
            if (std::isfinite(ly)) {
                Ey += e64 * static_cast<double>(ly);
            }
        }

        Eprint[0] = std::max(0.0f, static_cast<float>(Ec)); // C
        Eprint[1] = std::max(0.0f, static_cast<float>(Em)); // M
        Eprint[2] = std::max(0.0f, static_cast<float>(Ey)); // Y
    }

    // Compute per-channel raw exposures via spectral sensitivity contraction (C/M/Y order).
    inline void raw_exposures_from_filtered_light(
        const Profile& p,
        const std::vector<float>& Ee_filtered,
        float raw[3])
    {
        raw[0] = raw[1] = raw[2] = 0.0f;
        const int K = Spectral::gShape.K;
        if (K <= 0) return;

        // Spectral::build_curve_on_reference_axis_from_log10_pairs already exponentiates the
        // authored log10 sensitivities, so Curve::linear stores linear samples pinned
        // to the active spectral shape. Avoid re-applying pow(10).
        const auto& sensY = p.sensY_log.linear;
        const auto& sensM = p.sensM_log.linear;
        const auto& sensC = p.sensC_log.linear;

        const size_t sizeY = sensY.size();
        const size_t sizeM = sensM.size();
        const size_t sizeC = sensC.size();
        const size_t n = std::min({ static_cast<size_t>(K), Ee_filtered.size(), sizeY, sizeM, sizeC });

        double accumC = 0.0;
        double accumM = 0.0;
        double accumY = 0.0;

        for (size_t i = 0; i < n; ++i) {
            const float e = Ee_filtered[i];
            if (!std::isfinite(e)) {
                continue;
            }

            const double e64 = static_cast<double>(e);
            const float sY = sensY[i];
            const float sM = sensM[i];
            const float sC = sensC[i];

            if (std::isfinite(sY)) {
                accumY += e64 * static_cast<double>(sY);
            }
            if (std::isfinite(sM)) {
                accumM += e64 * static_cast<double>(sM);
            }
            if (std::isfinite(sC)) {
                accumC += e64 * static_cast<double>(sC);
            }
        }

        raw[0] = std::max(0.0f, static_cast<float>(accumC)); // C
        raw[1] = std::max(0.0f, static_cast<float>(accumM)); // M
        raw[2] = std::max(0.0f, static_cast<float>(accumY)); // Y
    }

    inline void compute_preflash_raw(
        const Runtime& rt,
        const WorkingState& ws,
        std::vector<float>& Tpre,
        std::vector<float>& Ee_pre,
        float rawOut[3])
    {
        rawOut[0] = rawOut[1] = rawOut[2] = 0.0f;
        const int K = Spectral::gShape.K;
        if (K <= 0) return;
        if (ws.tablesView.K <= 0) return;

        const float Dbase[3] = { 0.0f, 0.0f, 0.0f };
        negative_T_from_dyes(ws, Dbase, Tpre);
        if ((int)Tpre.size() < K) {
            Tpre.resize((size_t)K, 1.0f);
        }

        Ee_pre.resize((size_t)K);
        const float yAmount = compose_dichroic_amount(rt.neutralY, 0.0f);
        const float mAmount = compose_dichroic_amount(rt.neutralM, 0.0f);
        const float cAmount = compose_dichroic_amount(rt.neutralC, 0.0f);

        for (int i = 0; i < K; ++i) {
            const float Ee = (rt.illumEnlarger.linear.size() > size_t(i))
                ? rt.illumEnlarger.linear[i]
                : 1.0f;
            const float t = Tpre[i];
            const float fY = blend_dichroic_filter_linear(rt.filterY.linear.empty() ? 1.0f : rt.filterY.linear[i], yAmount);
            const float fM = blend_dichroic_filter_linear(rt.filterM.linear.empty() ? 1.0f : rt.filterM.linear[i], mAmount);
            const float fC = blend_dichroic_filter_linear(rt.filterC.linear.empty() ? 1.0f : rt.filterC.linear[i], cAmount);
            const float fTotal = fY * fM * fC;
            Ee_pre[i] = std::max(0.0f, Ee * t * fTotal);
        }

        raw_exposures_from_filtered_light(rt.profile, Ee_pre, rawOut);
    }

    // Midgray compensation factor computed spectrally (AgX parity): always normalize RAW so a neutral mid-gray is
    // mapped to unity exposure. When the print exposure compensation toggle is enabled the camera exposure EV scale
    // is injected, otherwise the EV term is ignored. This uses the same negative development, enlarger illuminant
    // (including dichroic filters), and print paper sensitivities as the print leg.
    inline float compute_exposure_factor_midgray(
        const WorkingState& ws,
        const Runtime& rt,
        const Params& prm,
        const Couplers::Runtime& dirRT,
        float exposureCompScale)
    {
        // If slider EV scale is not meaningful, skip compensation.
        if (!std::isfinite(exposureCompScale) || exposureCompScale <= 0.0f) return 1.0f;

        // 1) Midgray DWG rgb at canonical brightness (AgX parity: constant 18.4% reflectance)
        const float rgbMid[3] = { 0.184f, 0.184f, 0.184f };

        // 2) DWG → per-layer exposures (negative leg); apply camera EV exactly once here.
        //    NOTE: Do not pre-scale rgbMid by cameraExposureScale — avoids double-applying EV.
        float E[3];
        const Spectral::SpectralTables* tablesSPD =
            (ws.spdReady && ws.tablesRef.K > 0) ? &ws.tablesRef : nullptr;
        // Per agx-emulsion parity: mid-gray probe must use same sensitivities as actual render.
        // Profiles contain pre-balanced sensitivities; no separate "before balance" state.
        const float exposureScale = prm.exposureCompensationEnabled ? exposureCompScale : 1.0f;
        Spectral::rgb_input_to_film_raw(
            rgbMid, E, exposureScale,
            ws.filmRaw,
            tablesSPD,
            (ws.spdReady ? ws.spdSInv : nullptr),
            ws.spdReady,
            ws.sensB, ws.sensG, ws.sensR);

        // 3) LogE sampling (offsets already baked into density curves), clamp to domain, sample negative densities
        const float logE[3] = {
            std::log10(std::max(0.0f, E[0]) + 1e-10f),
            std::log10(std::max(0.0f, E[1]) + 1e-10f),
            std::log10(std::max(0.0f, E[2]) + 1e-10f)
        };
        float D_neg[3];
        sample_negative_densities(ws, dirRT, logE, D_neg, DirSampleMode::BypassRuntime);

        // 4) Negative transmittance (baseline applied per stock)       
        thread_local std::vector<float> Tneg, Ee_expose, Ee_filtered;
        negative_T_from_dyes(ws, D_neg, Tneg);

        // 5) Enlarger illuminant exposure
        Ee_expose.resize(Spectral::gShape.K);
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            const float Ee = rt.illumEnlarger.linear.empty() ? 1.0f : rt.illumEnlarger.linear[i];
            Ee_expose[i] = std::max(0.0f, Ee * Tneg[i]);
        }

        // 6) Apply Y/M/C dichroic filters using Durst wheel linear blending (agx parity)
        Ee_filtered.resize(Spectral::gShape.K);
        const float yAmount = compose_dichroic_amount(rt.neutralY, prm.yFilter);
        const float mAmount = compose_dichroic_amount(rt.neutralM, prm.mFilter);
        const float cAmount = compose_dichroic_amount(rt.neutralC, 0.0f);
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            const float fY = blend_dichroic_filter_linear(rt.filterY.linear.empty() ? 1.0f : rt.filterY.linear[i], yAmount);
            const float fM = blend_dichroic_filter_linear(rt.filterM.linear.empty() ? 1.0f : rt.filterM.linear[i], mAmount);
            const float fC = blend_dichroic_filter_linear(rt.filterC.linear.empty() ? 1.0f : rt.filterC.linear[i], cAmount);
            const float fTotal = fY * fM * fC;
            Ee_filtered[i] = std::max(0.0f, Ee_expose[i] * fTotal);
        }

        // 7) RAW via print paper sensitivities (log domain → linear sensitivity)
        float raw[3];
        raw_exposures_from_filtered_light(rt.profile, Ee_filtered, raw);

        const float safeRawMid = std::max(1e-12f, raw[1]);
        const float baseFactor = std::isfinite(safeRawMid) && safeRawMid > 0.0f
            ? 1.0f / safeRawMid
            : 1.0f;

        if (std::fabs(baseFactor - 1.0f) > 0.05f) {
            std::ostringstream oss;
            oss << "midgray compensation factor=" << baseFactor
                << " rawMid=" << safeRawMid
                << " normalizedMidgray=" << ws.filmRaw.midgrayScale;
            JTRACE("PRINT", oss.str());
        }

        return baseFactor;
    }

    inline float interpolate_density_gamma(const Spectral::Curve& dc, float logE, float gammaFactor) {
        if (dc.lambda_nm.empty()) {
            return 0.0f;
        }

        const float gammaSafe = (std::isfinite(gammaFactor) && gammaFactor > 0.0f)
            ? gammaFactor
            : 1.0f;

        // Gamma applied inside sample_density_at_logE.
        return Spectral::sample_density_at_logE(dc, logE, gammaSafe);
    }

    // Build print densities from print exposures (C/M/Y order, parity with agx-emulsion)
    inline void print_densities_from_Eprint(const Profile& p, const float Eprint[3], float D_print[3]) {
        auto safe_log10 = [](float v)->float { return std::log10(std::max(0.0f, v) + 1e-10f); };
        float lEc = safe_log10(Eprint[0]);
        float lEm = safe_log10(Eprint[1]);
        float lEy = safe_log10(Eprint[2]);

        D_print[0] = interpolate_density_gamma(p.dcC, lEc, p.gammaFactor[0]);
        D_print[1] = interpolate_density_gamma(p.dcM, lEm, p.gammaFactor[1]);
        D_print[2] = interpolate_density_gamma(p.dcY, lEy, p.gammaFactor[2]);

        // Densities are optical densities (OD); clamp finite values to non-negative to prevent T>1.
        // agx-emulsion parity: do not convert NaN -> 0 density (NaN should become 0 transmitted light later).
        for (int i = 0; i < 3; ++i) {
            if (std::isfinite(D_print[i]) && D_print[i] < 0.0f) {
                D_print[i] = 0.0f;
            }
        }

    }


    inline void print_T_from_dyes(const Profile& p, const float D_print[3], std::vector<float>& Tprint_out) {
        const int K = Spectral::gShape.K;
        Tprint_out.resize(K);
        for (int i = 0; i < K; ++i) {
            const float baseSpectral = p.hasBaseline
                ? p.baseMin.linear[i]
                : 0.0f;
            const float Dlambda = D_print[0] * p.epsC.linear[i]
                + D_print[1] * p.epsM.linear[i]
                + D_print[2] * p.epsY.linear[i]
                + baseSpectral;
            if (!std::isfinite(Dlambda)) {
                Tprint_out[i] = 0.0f;
                continue;
            }
            Tprint_out[i] = std::exp(-Spectral::kLn10 * Dlambda);
        }
    }
    // >>> BEGIN INSERT: test-only probe helpers (Print.h) REMOVE AFTER USE
#ifdef JUICER_TESTS

    struct Probe {
        std::vector<float> Tneg, Ee_expose, Ee_filtered, Tprint, Ee_viewed;
        float raw[3]{ 0,0,0 };       // C/M/Y order
        float D_print[3]{ 0,0,0 };   // C/M/Y order
        float XYZ[3]{ 0,0,0 };
    };

    // Start from *negative* logE (already camera-exposed), then run the *print* leg.
    // This isolates the print path (filters → raw → DC curves → Tprint → viewing → XYZ → DWG).
    inline void simulate_print_from_logE_for_test(
        const float logE_neg[3],
        const Params& prm,
        const Runtime& rt,
        const Couplers::Runtime& dirRT,
        const WorkingState& ws,
        float exposureScale,              // not used when starting at logE
        float rgbOut[3],
        Probe* probe /*nullable*/)
    {
        if (ws.tablesPrint.K <= 0) {
            rgbOut[0] = rgbOut[1] = rgbOut[2] = 0.0f;
            if (probe) {
                probe->XYZ[0] = probe->XYZ[1] = probe->XYZ[2] = 0.0f;
            }
            return;
        }
        // 1) Sample negative densities from per-instance negative curves.
        float D_neg[3] = {
            Spectral::sample_density_at_logE(ws.densB, logE_neg[0], ws.gammaFactorB), // Y from B-layer curve
            Spectral::sample_density_at_logE(ws.densG, logE_neg[1], ws.gammaFactorG), // M from G-layer curve
            Spectral::sample_density_at_logE(ws.densR, logE_neg[2], ws.gammaFactorR)  // C from R-layer curve
        };

        // 2) Negative transmittance with per-stock baseline        
        thread_local std::vector<float> Tneg, Ee_expose, Ee_filtered, Tprint, Ee_viewed;
        thread_local std::vector<float> Tpreflash, Ee_preflash;
        negative_T_from_dyes(ws, D_neg, Tneg);

        // 3) Enlarger exposure (neutral illuminant here; print exposure happens later on RAW)
        Ee_expose.resize(Spectral::gShape.K);
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            const float Ee = rt.illumEnlarger.linear.empty() ? 1.0f : rt.illumEnlarger.linear[i];
            Ee_expose[i] = std::max(0.0f, Ee * Tneg[i]);
        }

        // 4) Apply neutral Y/M/C dichroic filters to enlarger light
        Ee_filtered.resize(Spectral::gShape.K);
        const float yAmount = compose_dichroic_amount(rt.neutralY, prm.yFilter);
        const float mAmount = compose_dichroic_amount(rt.neutralM, prm.mFilter);
        const float cAmount = compose_dichroic_amount(rt.neutralC, 0.0f);
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            const float fY = blend_dichroic_filter_linear(rt.filterY.linear.empty() ? 1.0f : rt.filterY.linear[i], yAmount);
            const float fM = blend_dichroic_filter_linear(rt.filterM.linear.empty() ? 1.0f : rt.filterM.linear[i], mAmount);
            const float fC = blend_dichroic_filter_linear(rt.filterC.linear.empty() ? 1.0f : rt.filterC.linear[i], cAmount);
            const float fTotal = fY * fM * fC;
            Ee_filtered[i] = std::max(0.0f, Ee_expose[i] * fTotal);
        }

        // 5) Contract to channel RAW using print sensitometries
        float raw[3] = { 0,0,0 };
        raw_exposures_from_filtered_light(rt.profile, Ee_filtered, raw);

        // Apply print exposure (agx parity)
        const float expPrint = std::isfinite(prm.exposure) ? std::max(0.0f, prm.exposure) : 1.0f;
        const float exposureCompScale = prm.exposureCompensationEnabled
            ? prm.exposureCompensationScale
            : 1.0f;

        const float kMid = compute_exposure_factor_midgray(ws, rt, prm, dirRT, exposureCompScale);
        const float rawScale = expPrint * kMid;
        raw[0] *= rawScale;
        raw[1] *= rawScale;
        raw[2] *= rawScale;

        if (std::isfinite(prm.preflashExposure) && prm.preflashExposure > 0.0f) {
            float rawPre[3];
            compute_preflash_raw(rt, ws, Tpreflash, Ee_preflash, rawPre);
            raw[0] += rawPre[0] * prm.preflashExposure;
            raw[1] += rawPre[1] * prm.preflashExposure;
            raw[2] += rawPre[2] * prm.preflashExposure;
        }

        // 6) RAW → print densities via DC curves (logE offsets already baked into curves)
        float D_print[3] = { 0,0,0 };
        print_densities_from_Eprint(rt.profile, raw, D_print);

        // 7) Print transmittance
        print_T_from_dyes(rt.profile, D_print, Tprint);

        // 8) Viewing illuminant and integration to XYZ
        Ee_viewed.resize(Spectral::gShape.K);
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            const float Ev = rt.illumView.linear.empty() ? 1.0f : rt.illumView.linear[i];
            Ee_viewed[i] = std::max(0.0f, Ev * Tprint[i]);
        }

        float XYZ[3] = { 0,0,0 };
        if (ws.tablesPrint.K > 0) {
            Spectral::Ee_to_XYZ_given_tables(ws.tablesPrint, Ee_viewed, XYZ);

            // XYZ → DWG with chromatic adaptation using print paper white
            Spectral::XYZ_to_DWG_linear_adapted(ws.tablesPrint, XYZ, rgbOut);
        }
        rgbOut[0] = std::max(0.0f, rgbOut[0]);
        rgbOut[1] = std::max(0.0f, rgbOut[1]);
        rgbOut[2] = std::max(0.0f, rgbOut[2]);

        if (probe) {
            probe->Tneg = Tneg;
            probe->Ee_expose = Ee_expose;
            probe->Ee_filtered = Ee_filtered;
            probe->Tprint = Tprint;
            probe->Ee_viewed = Ee_viewed;
            probe->raw[0] = raw[0]; probe->raw[1] = raw[1]; probe->raw[2] = raw[2];
            probe->D_print[0] = D_print[0]; probe->D_print[1] = D_print[1]; probe->D_print[2] = D_print[2];
            probe->XYZ[0] = XYZ[0]; probe->XYZ[1] = XYZ[1]; probe->XYZ[2] = XYZ[2];
        }
    }

#endif // JUICER_TESTS
    // <<< END INSERT REMOVE AFTER USE

    // Full pixel pipeline when print is active
    inline void simulate_print_pixel(const float rgbIn[3],
        const Params& prm,
        const Runtime& rt,
        const Couplers::Runtime& dirRT,
        const WorkingState& ws,
        float exposureScale,
        float kMid_spectral,
        float rgbOut[3])
    {
        if (ws.tablesPrint.K <= 0) {
            return;
        }


        // 1) Negative densities with DIR in logE domain (per-instance SPD vs Matrix)
        float E[3];
        const Spectral::SpectralTables* tablesSPD =
            (ws.spdReady && ws.tablesRef.K > 0) ? &ws.tablesRef : nullptr;
        // Per agx-emulsion parity: use same sensitivities everywhere.
        const float sExp = (std::isfinite(exposureScale) ? std::max(0.0f, exposureScale) : 1.0f);
        Spectral::rgb_input_to_film_raw(
            rgbIn, E, sExp,
            ws.filmRaw,
            tablesSPD,
            (ws.spdReady ? ws.spdSInv : nullptr),
            ws.spdReady,
            ws.sensB, ws.sensG, ws.sensR);



        float logE[3] = {
            std::log10(std::max(0.0f, E[0]) + 1e-10f),
            std::log10(std::max(0.0f, E[1]) + 1e-10f),
            std::log10(std::max(0.0f, E[2]) + 1e-10f)
        };

        float D_neg[3];
        sample_negative_densities(ws, dirRT, logE, D_neg);

        // 2) Negative transmittance with optional baseline blend    
        thread_local std::vector<float> Tneg, Ee_expose, Tprint, Ee_viewed;
        thread_local std::vector<float> Tpreflash, Ee_preflash;
        negative_T_from_dyes(ws, D_neg, Tneg);

        // 3) Ee_expose = Ee_enlarger * T_neg * exposure
        Ee_expose.resize(Spectral::gShape.K);
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            const float Ee = rt.illumEnlarger.linear.empty() ? 1.0f : rt.illumEnlarger.linear[i];
            Ee_expose[i] = std::max(0.0f, Ee * Tneg[i]);
        }

        // 4) Reduce to per-channel print exposures and apply neutral Y/M/C filters
        // Apply spectral Y/M/C dichroic filters to enlarger light before per-channel reduction
        thread_local std::vector<float> Ee_filtered;
        Ee_filtered.resize(Spectral::gShape.K);
        const float yAmount = compose_dichroic_amount(rt.neutralY, prm.yFilter);
        const float mAmount = compose_dichroic_amount(rt.neutralM, prm.mFilter);
        const float cAmount = compose_dichroic_amount(rt.neutralC, 0.0f);
        // Blend each wheel between identity (1.0) and its full transmittance curve
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            const float fY = blend_dichroic_filter_linear(rt.filterY.linear.empty() ? 1.0f : rt.filterY.linear[i], yAmount);
            const float fM = blend_dichroic_filter_linear(rt.filterM.linear.empty() ? 1.0f : rt.filterM.linear[i], mAmount);
            const float fC = blend_dichroic_filter_linear(rt.filterC.linear.empty() ? 1.0f : rt.filterC.linear[i], cAmount);
            const float fTotal = fY * fM * fC;
            Ee_filtered[i] = std::max(0.0f, Ee_expose[i] * fTotal);
        }

        // Compute per-channel raw via sensitivity contraction (agx parity)
        float raw[3];
        raw_exposures_from_filtered_light(rt.profile, Ee_filtered, raw);

        // Apply print exposure scaling to raw (agx: raw *= print_exposure)
        const float expPrint = std::isfinite(prm.exposure) ? std::max(0.0f, prm.exposure) : 1.0f;

        // Spectral midgray compensation (AgX parity): scale RAW vector by factor = 1 / RAW_midgray_green.
        // Prefer a caller-supplied precomputed factor (tile-constant) to avoid redundant probes.
        const float exposureCompScale = prm.exposureCompensationEnabled
            ? prm.exposureCompensationScale
            : 1.0f;
        const bool hasPrecomputedMid = std::isfinite(kMid_spectral) && kMid_spectral > 0.0f;
        const float kMid = hasPrecomputedMid
            ? kMid_spectral
            : compute_exposure_factor_midgray(ws, rt, prm, dirRT, exposureCompScale);

        const float rawScale = expPrint * kMid;
        raw[0] *= rawScale;
        raw[1] *= rawScale;
        raw[2] *= rawScale;

        if (std::isfinite(prm.preflashExposure) && prm.preflashExposure > 0.0f) {
            float rawPre[3];
            compute_preflash_raw(rt, ws, Tpreflash, Ee_preflash, rawPre);
            raw[0] += rawPre[0] * prm.preflashExposure;
            raw[1] += rawPre[1] * prm.preflashExposure;
            raw[2] += rawPre[2] * prm.preflashExposure;
        }


        // Map raw to print densities via DC curves (no runtime logE offsets)
        float D_print[3];
        print_densities_from_Eprint(rt.profile, raw, D_print);

        // 6) Print transmittance
        print_T_from_dyes(rt.profile, D_print, Tprint);

        // 7) Multiply print transmittance by viewing illuminant to get Ee_viewed
        Ee_viewed.resize(Spectral::gShape.K);
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            const float Ev = rt.illumView.linear.empty() ? 1.0f : rt.illumView.linear[i];
            Ee_viewed[i] = std::max(0.0f, Ev * Tprint[i]);
        }

        // Integrate to XYZ using per-instance viewing axis and normalization
        float XYZ[3];
        if (ws.tablesPrint.K > 0) {
            Spectral::Ee_to_XYZ_given_tables(ws.tablesPrint, Ee_viewed, XYZ);

            // XYZ -> DWG with chromatic adaptation using print paper tables
            Spectral::XYZ_to_DWG_linear_adapted(ws.tablesPrint, XYZ, rgbOut);
        }
        rgbOut[0] = std::max(0.0f, rgbOut[0]);
        rgbOut[1] = std::max(0.0f, rgbOut[1]);
        rgbOut[2] = std::max(0.0f, rgbOut[2]);

    }

} // namespace Print
