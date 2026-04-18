// Print.h
#pragma once
#include <vector>
#include <string>
#include <array>
#include <cstdint>
#include <cmath>
#include <utility>
#include "SpectralData.h"
#include "Illuminants.h"
#include "ProfileJSONLoader.h"
#include "ProcessRoot.h"
#include <sstream>
#include <algorithm>
#include <limits>

namespace Print {

    constexpr float kEnlargerSteps = 170.0f;
    constexpr float kDefaultNeutralY = 0.9f;
    constexpr float kDefaultNeutralM = 0.5f;
    constexpr float kDefaultNeutralC = 0.35f;
    constexpr std::uint64_t kDefaultNeutralFilterHash = 1ull;

    struct DensityCurves {
        std::vector<std::pair<double, double>> cyan;
        std::vector<std::pair<double, double>> magenta;
        std::vector<std::pair<double, double>> yellow;
        bool usedJson = false;
    };

    struct Profile;

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
        float logEOffC = 0.0f, logEOffM = 0.0f, logEOffY = 0.0f; // retained per-channel logE offsets (unused)
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
        float cFilter = 0.0f;

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
        std::uint64_t neutralFilterHash = kDefaultNeutralFilterHash;

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
            if (!spectral_curve_ok(p.baseMin)) {
                return false;
            }
            if (!p.baseMid.linear.empty() && !spectral_curve_ok(p.baseMid)) {
                return false;
            }
        }
        return true;
    }
    bool remove_glare_compensation_from_curves(Profile& profile, DensityCurves& curves);
    bool rebuild_density_curves(Profile& profile, const DensityCurves& curves);
    void recompute_mid_neutral(Profile& profile, Runtime* runtime = nullptr);
    void load_profile_from_dir(const std::string& dir, Profile& out, const std::string& jsonProfilePath = std::string(), Runtime* runtime = nullptr);

    // Build an illuminant pinned to shape from choice. Choices align with your UI (0:D65,1:D55,2:D50,3:TH-KG3-L,4:T,5:K75P,6:Equal)
    inline void build_illuminant_from_choice(int choice, Runtime& rt, const std::string&, bool forEnlarger) {
        const JuicerAssets::IlluminantFilterCurveSet& curveAssets =
            JuicerProcess::root().assets().illuminant_filter_curves();

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
            } catch (const std::exception& e) {
                log_failure(label, e.what());
            } catch (...) {
                log_failure(label, "unknown error");
            }
            return Spectral::Curve{};
        };

        Spectral::Curve c;
        switch (choice) {
            case 0:
                c = build_or_log([&]() {
                    return curveAssets.d65;
                },
                                 "D65");
                break;
            case 1:
                c = build_or_log([&]() {
                    return curveAssets.d55;
                },
                                 "D55");
                break;
            case 2:
                c = build_or_log([&]() {
                    return curveAssets.d50;
                },
                                 "D50");
                break;
            case 3:
                c = build_or_log([&]() {
                    return curveAssets.tungstenKg3Lens;
                },
                                 "TH-KG3-L");
                break;
            case 4:
                c = build_or_log([&]() {
                    return curveAssets.tungsten;
                },
                                 "T");
                break;
            case 5:
                c = build_or_log([&]() {
                    return curveAssets.kinoton75P;
                },
                                 "K75P");
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
        } else {
            rt.illumView = std::move(c);
        }
    }

    inline void load_dichroic_filters_from_assets(
        const JuicerAssets::DichroicFilterCurveSet& curves,
        Runtime& rt) {
        rt.filterY = curves.filterY;
        rt.filterM = curves.filterM;
        rt.filterC = curves.filterC;

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
} // namespace Print
