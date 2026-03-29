#include <algorithm>
#include <cmath>
#include <sstream>

#include "FilmProcessing.h"
#include "Logging.h"
#include "PipelineTypes.h"
#include "Print.h"
#include "WorkingState.h"

namespace Pipeline {

    struct ExposeFilmInputs {
        RgbLinear rgb;
        float exposureScale = 1.0f;
    };

    struct ExposeFilmOutputs {
        FilmRaw filmRaw;
    };

    class ExposeFilmStage {
    public:
        static bool run(const WorkingState& ws, const ExposeFilmInputs& in, ExposeFilmOutputs& out);
    };

    struct DevelopFilmInputs {
        FilmRaw filmRaw;
        const Couplers::Runtime* dirRuntime = nullptr;
        bool applyDirRuntime = true;

        bool useSpatialDIR = false;
        float spatialLogECorrectionsYMC[3] = { 0.0f, 0.0f, 0.0f };
    };

    struct DevelopFilmOutputs {
        FilmLogRaw filmLogRaw;
        NegativeDensityCMY negativeDensity;
    };

    class DevelopFilmStage {
    public:
        static bool run(const WorkingState& ws, const DevelopFilmInputs& in, DevelopFilmOutputs& out);
        static FilmLogRaw compute_log_raw(const FilmRaw& filmRaw);
    };

    namespace {

        inline float blend_dichroic_filter_linear(float curveVal, float normalizedAmount) {
            // agx-emulsion parity: do not treat non-finite curve samples as identity.
            // NaNs must propagate even when amount is 0 (NumPy semantics: NaN * 0 = NaN).
            const float a = std::isfinite(normalizedAmount) ? normalizedAmount : 0.0f;
            return 1.0f - (1.0f - curveVal) * a;
        }

        inline float compose_dichroic_amount(float neutralAmount, float deltaAmount) {
            const float neutral = std::isfinite(neutralAmount)
                ? std::clamp(neutralAmount, 0.0f, 1.0f)
                : 0.0f;
            float deltaSteps = std::isfinite(deltaAmount) ? deltaAmount : 0.0f;
            const float shiftLimit = Print::kEnlargerSteps;
            deltaSteps = std::clamp(deltaSteps, -shiftLimit, shiftLimit);
            const float totalSteps = neutral * Print::kEnlargerSteps + deltaSteps;
            return totalSteps / Print::kEnlargerSteps;
        }

        void negative_density_spectral_from_dyes(
            const WorkingState& ws,
            const float D_neg[3],
            std::vector<float>& density_out)
        {
            const int K = ws.tablesView.K;
            if (K <= 0) {
                density_out.clear();
                return;
            }
            density_out.resize(static_cast<size_t>(K));

            // Preserve the legacy gate used by the current render bridge.
            const bool hasBL = ws.hasBaseline && static_cast<int>(ws.baseMin.linear.size()) == K;

            auto epsY_at = [&](int i) {
                return (i < static_cast<int>(ws.tablesView.epsY.size())) ? ws.tablesView.epsY[i] : 0.0f;
                };
            auto epsM_at = [&](int i) {
                return (i < static_cast<int>(ws.tablesView.epsM.size())) ? ws.tablesView.epsM[i] : 0.0f;
                };
            auto epsC_at = [&](int i) {
                return (i < static_cast<int>(ws.tablesView.epsC.size())) ? ws.tablesView.epsC[i] : 0.0f;
                };

            for (int i = 0; i < K; ++i) {
                const float baseSpectral = hasBL &&
                        static_cast<size_t>(i) < ws.tablesView.baseMin.size()
                    ? ws.tablesView.baseMin[static_cast<size_t>(i)]
                    : 0.0f;

                density_out[i] = D_neg[0] * epsC_at(i) // C
                    + D_neg[1] * epsM_at(i) // M
                    + D_neg[2] * epsY_at(i) // Y
                    + baseSpectral;
            }
        }

        void build_enlarger_illuminant_filtered(
            const Print::Runtime& rt,
            float yShiftSteps,
            float mShiftSteps,
            float cShiftSteps,
            std::vector<float>& illuminant_out)
        {
            const int K = Spectral::gShape.K;
            if (K <= 0) {
                illuminant_out.clear();
                return;
            }
            illuminant_out.resize(static_cast<size_t>(K));

            const float yAmount = compose_dichroic_amount(rt.neutralY, yShiftSteps);
            const float mAmount = compose_dichroic_amount(rt.neutralM, mShiftSteps);
            const float cAmount = compose_dichroic_amount(rt.neutralC, cShiftSteps);

            for (int i = 0; i < K; ++i) {
                const float Ee = (rt.illumEnlarger.linear.size() > static_cast<size_t>(i))
                    ? rt.illumEnlarger.linear[static_cast<size_t>(i)]
                    : 1.0f;
                const float fY = blend_dichroic_filter_linear(
                    (rt.filterY.linear.size() > static_cast<size_t>(i)) ? rt.filterY.linear[static_cast<size_t>(i)] : 1.0f,
                    yAmount);
                const float fM = blend_dichroic_filter_linear(
                    (rt.filterM.linear.size() > static_cast<size_t>(i)) ? rt.filterM.linear[static_cast<size_t>(i)] : 1.0f,
                    mAmount);
                const float fC = blend_dichroic_filter_linear(
                    (rt.filterC.linear.size() > static_cast<size_t>(i)) ? rt.filterC.linear[static_cast<size_t>(i)] : 1.0f,
                    cAmount);
                illuminant_out[static_cast<size_t>(i)] = Ee * (fY * fM * fC);
            }
        }

        void ensure_cached_enlarger_illuminant_filtered(
            const Print::Runtime& rt,
            std::uint64_t wsBuildCounter,
            float yShiftSteps,
            float mShiftSteps,
            float cShiftSteps,
            PrintPipelineScratch& scratch)
        {
            const int shapeK = Spectral::gShape.K;
            if (shapeK <= 0) {
                scratch.enlargerIlluminantFilteredValid = false;
                scratch.enlargerIlluminantWsBuildCounter = 0;
                scratch.enlargerIlluminantNeutralFilterHash = 0;
                scratch.enlargerIlluminantShapeK = 0;
                scratch.Ee_expose.clear();
                return;
            }

            // Match compose_dichroic_amount behavior: treat non-finite delta steps as 0 so cache keys
            // do not thrash on NaN inputs (NaN != NaN).
            const float yKey = std::isfinite(yShiftSteps) ? yShiftSteps : 0.0f;
            const float mKey = std::isfinite(mShiftSteps) ? mShiftSteps : 0.0f;
            const float cKey = std::isfinite(cShiftSteps) ? cShiftSteps : 0.0f;
            const std::uint64_t neutralFilterHash =
                (rt.neutralFilterHash != 0) ? rt.neutralFilterHash : Print::kDefaultNeutralFilterHash;

            if (scratch.enlargerIlluminantFilteredValid &&
                scratch.enlargerIlluminantWsBuildCounter == wsBuildCounter &&
                scratch.enlargerIlluminantNeutralFilterHash == neutralFilterHash &&
                scratch.enlargerIlluminantYShiftSteps == yKey &&
                scratch.enlargerIlluminantMShiftSteps == mKey &&
                scratch.enlargerIlluminantCShiftSteps == cKey &&
                scratch.enlargerIlluminantShapeK == shapeK) {
                return;
            }

            build_enlarger_illuminant_filtered(rt, yKey, mKey, cKey, scratch.Ee_expose);
            scratch.enlargerIlluminantFilteredValid = true;
            scratch.enlargerIlluminantWsBuildCounter = wsBuildCounter;
            scratch.enlargerIlluminantNeutralFilterHash = neutralFilterHash;
            scratch.enlargerIlluminantYShiftSteps = yKey;
            scratch.enlargerIlluminantMShiftSteps = mKey;
            scratch.enlargerIlluminantCShiftSteps = cKey;
            scratch.enlargerIlluminantShapeK = shapeK;
        }

        void density_to_filtered_light_agx(
            const WorkingState& ws,
            const Print::Runtime& rt,
            float yShiftSteps,
            float mShiftSteps,
            float cShiftSteps,
            const float D_neg[3],
            std::vector<float>& tmp_density_spectral,
            std::vector<float>& tmp_illuminant_filtered,
            std::vector<float>& out_light)
        {
            negative_density_spectral_from_dyes(ws, D_neg, tmp_density_spectral);
            build_enlarger_illuminant_filtered(rt, yShiftSteps, mShiftSteps, cShiftSteps, tmp_illuminant_filtered);
            density_to_light_agx(tmp_density_spectral, tmp_illuminant_filtered, out_light);
        }

        void raw_exposures_from_filtered_light(
            const Print::Profile& p,
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
                if (std::isnan(e)) {
                    continue;
                }

                const double e64 = static_cast<double>(e);
                const float sY = sensY[i];
                const float sM = sensM[i];
                const float sC = sensC[i];

                if (!std::isnan(sY)) {
                    accumY += e64 * static_cast<double>(sY);
                }
                if (!std::isnan(sM)) {
                    accumM += e64 * static_cast<double>(sM);
                }
                if (!std::isnan(sC)) {
                    accumC += e64 * static_cast<double>(sC);
                }
            }

            raw[0] = static_cast<float>(accumC); // C
            raw[1] = static_cast<float>(accumM); // M
            raw[2] = static_cast<float>(accumY); // Y
        }

        void ensure_cached_preflash_raw(
            const WorkingState& ws,
            const Print::Runtime& rt,
            PrintPipelineScratch& scratch)
        {
            const int shapeK = Spectral::gShape.K;
            const std::uint64_t neutralFilterHash =
                (rt.neutralFilterHash != 0) ? rt.neutralFilterHash : Print::kDefaultNeutralFilterHash;
            if (shapeK <= 0 || ws.tablesView.K <= 0) {
                scratch.preflashRawValid = false;
                scratch.preflashWsBuildCounter = 0;
                scratch.preflashNeutralFilterHash = 0;
                scratch.preflashShapeK = 0;
                scratch.preflashRaw[0] = scratch.preflashRaw[1] = scratch.preflashRaw[2] = 0.0f;
                return;
            }

            if (scratch.preflashRawValid &&
                scratch.preflashWsBuildCounter == ws.buildCounter &&
                scratch.preflashNeutralFilterHash == neutralFilterHash &&
                scratch.preflashShapeK == shapeK) {
                return;
            }

            int computedShapeK = 0;
            if (!compute_preflash_raw(ws, rt, scratch.preflashRaw, computedShapeK)) {
                scratch.preflashRawValid = false;
                scratch.preflashWsBuildCounter = 0;
                scratch.preflashNeutralFilterHash = 0;
                scratch.preflashShapeK = 0;
                scratch.preflashRaw[0] = scratch.preflashRaw[1] = scratch.preflashRaw[2] = 0.0f;
                return;
            }

            scratch.preflashRawValid = true;
            scratch.preflashWsBuildCounter = ws.buildCounter;
            scratch.preflashNeutralFilterHash = neutralFilterHash;
            scratch.preflashShapeK = computedShapeK;
        }

    } // namespace

    bool compute_preflash_raw(
        const WorkingState& ws,
        const Print::Runtime& prt,
        float outRaw[3],
        int& outShapeK)
    {
        outShapeK = Spectral::gShape.K;
        if (!outRaw || outShapeK <= 0) {
            return false;
        }

        outRaw[0] = 0.0f;
        outRaw[1] = 0.0f;
        outRaw[2] = 0.0f;

        if (ws.tablesView.K <= 0 || ws.tablesView.K != outShapeK) {
            return false;
        }

        const Print::Profile& p = prt.profile;
        if (static_cast<int>(p.sensC_log.linear.size()) < outShapeK ||
            static_cast<int>(p.sensM_log.linear.size()) < outShapeK ||
            static_cast<int>(p.sensY_log.linear.size()) < outShapeK) {
            return false;
        }

        const bool hasBaseline =
            ws.hasBaseline &&
            static_cast<int>(ws.tablesView.baseMin.size()) == outShapeK;

        const float yAmount = compose_dichroic_amount(prt.neutralY, 0.0f);
        const float mAmount = compose_dichroic_amount(prt.neutralM, 0.0f);
        const float cAmount = compose_dichroic_amount(prt.neutralC, 0.0f);

        double accumC = 0.0;
        double accumM = 0.0;
        double accumY = 0.0;

        for (int i = 0; i < outShapeK; ++i) {
            const size_t idx = static_cast<size_t>(i);
            const float Ee = (prt.illumEnlarger.linear.size() > idx)
                ? prt.illumEnlarger.linear[idx]
                : 1.0f;
            const float fY = blend_dichroic_filter_linear(
                (prt.filterY.linear.size() > idx) ? prt.filterY.linear[idx] : 1.0f,
                yAmount);
            const float fM = blend_dichroic_filter_linear(
                (prt.filterM.linear.size() > idx) ? prt.filterM.linear[idx] : 1.0f,
                mAmount);
            const float fC = blend_dichroic_filter_linear(
                (prt.filterC.linear.size() > idx) ? prt.filterC.linear[idx] : 1.0f,
                cAmount);
            const float illumFiltered = Ee * (fY * fM * fC);

            const float baseDensity = hasBaseline ? ws.tablesView.baseMin[idx] : 0.0f;
            const double light = static_cast<double>(density_to_light_sample_agx(baseDensity, illumFiltered));

            const float sC = p.sensC_log.linear[idx];
            const float sM = p.sensM_log.linear[idx];
            const float sY = p.sensY_log.linear[idx];
            if (!std::isnan(sC)) {
                accumC += light * static_cast<double>(sC);
            }
            if (!std::isnan(sM)) {
                accumM += light * static_cast<double>(sM);
            }
            if (!std::isnan(sY)) {
                accumY += light * static_cast<double>(sY);
            }
        }

        outRaw[0] = static_cast<float>(accumC);
        outRaw[1] = static_cast<float>(accumM);
        outRaw[2] = static_cast<float>(accumY);
        return std::isfinite(outRaw[0]) && std::isfinite(outRaw[1]) && std::isfinite(outRaw[2]);
    }

    bool ExposePrintStage::run(
        const WorkingState& ws,
        const ExposePrintInputs& in,
        ExposePrintOutputs& out,
        PrintPipelineScratch& scratch)
    {
        out.printRaw = PrintRaw{};
        out.printLogRaw = PrintLogRaw{};

        if (!in.printRuntime || !in.printParams) {
            return false;
        }

        const Print::Runtime& prt = *in.printRuntime;
        const Print::Params& prm = *in.printParams;

        const int viewK = ws.tablesView.K;
        const int printK = ws.tablesPrint.K;
        const int shapeK = Spectral::gShape.K;
        if (viewK <= 0 || printK <= 0 || shapeK <= 0 || viewK != printK || viewK != shapeK) {
            return false;
        }

        const float D_cmy[3] = {
            in.negativeDensity.v[0],
            in.negativeDensity.v[1],
            in.negativeDensity.v[2]
        };

        ensure_cached_enlarger_illuminant_filtered(
            prt,
            ws.buildCounter,
            prm.yFilter,
            prm.mFilter,
            prm.cFilter,
            scratch);

        negative_density_spectral_from_dyes(ws, D_cmy, scratch.Tneg);
        density_to_light_agx(scratch.Tneg, scratch.Ee_expose, scratch.Ee_filtered);

        float raw[3];
        raw_exposures_from_filtered_light(prt.profile, scratch.Ee_filtered, raw);

        // Apply print exposure scaling to raw (agx: raw *= print_exposure)
        const float expPrint = std::isfinite(prm.exposure)
            ? std::max(0.0f, prm.exposure)
            : 1.0f;

        const float kMid = (std::isfinite(in.midgrayFactor) && in.midgrayFactor > 0.0f)
            ? in.midgrayFactor
            : 1.0f;

        const float rawScale = expPrint * kMid;
        raw[0] *= rawScale;
        raw[1] *= rawScale;
        raw[2] *= rawScale;

        if (std::isfinite(prm.preflashExposure) && prm.preflashExposure > 0.0f) {
            ensure_cached_preflash_raw(ws, prt, scratch);
            raw[0] += scratch.preflashRaw[0] * prm.preflashExposure;
            raw[1] += scratch.preflashRaw[1] * prm.preflashExposure;
            raw[2] += scratch.preflashRaw[2] * prm.preflashExposure;
        }

        out.printRaw.v[0] = raw[0];
        out.printRaw.v[1] = raw[1];
        out.printRaw.v[2] = raw[2];

        out.printLogRaw.v[0] = std::log10(raw[0] + 1e-10f);
        out.printLogRaw.v[1] = std::log10(raw[1] + 1e-10f);
        out.printLogRaw.v[2] = std::log10(raw[2] + 1e-10f);

        return true;
    }

    float ExposePrintStage::compute_midgray_factor(
        const WorkingState& ws,
        const Print::Runtime& printRuntime,
        const Print::Params& printParams,
        const Couplers::Runtime& dirRT,
        float exposureCompScale)
    {
        // If slider EV scale is not meaningful, skip compensation.
        if (!std::isfinite(exposureCompScale) || exposureCompScale <= 0.0f) return 1.0f;

        // 1) Midgray DWG rgb at canonical brightness (AgX parity: constant 18.4% reflectance)
        const float rgbMid[3] = { 0.184f, 0.184f, 0.184f };

	        // 2) DWG → per-layer exposures (negative leg); apply camera EV exactly once here.
	        //    NOTE: Do not pre-scale rgbMid by cameraExposureScale — avoids double-applying EV.
	        Pipeline::ExposeFilmInputs exposeIn{};
	        exposeIn.rgb.v[0] = rgbMid[0];
	        exposeIn.rgb.v[1] = rgbMid[1];
	        exposeIn.rgb.v[2] = rgbMid[2];
	        // Per agx-emulsion parity: mid-gray probe must use same exposure scaling semantics as render.
	        exposeIn.exposureScale = printParams.exposureCompensationEnabled ? exposureCompScale : 1.0f;

	        Pipeline::ExposeFilmOutputs exposeOut{};
	        if (!Pipeline::ExposeFilmStage::run(ws, exposeIn, exposeOut)) {
	            return 1.0f;
	        }

	        Pipeline::DevelopFilmInputs devIn{};
	        devIn.filmRaw = exposeOut.filmRaw;
	        devIn.dirRuntime = &dirRT;
	        devIn.applyDirRuntime = false; // midgray factor uses pre-DIR densities (legacy behavior)

	        Pipeline::DevelopFilmOutputs devOut{};
	        if (!Pipeline::DevelopFilmStage::run(ws, devIn, devOut)) {
	            return 1.0f;
	        }

	        const float D_neg[3] = {
	            devOut.negativeDensity.v[0],
	            devOut.negativeDensity.v[1],
	            devOut.negativeDensity.v[2]
	        };

	        // 4) Print illuminant + negative density -> transmitted light (agx parity: NaNs collapse to 0 here only).
	        static thread_local std::vector<float> density_spectral;
	        static thread_local std::vector<float> print_illuminant;
	        static thread_local std::vector<float> light;
        density_to_filtered_light_agx(
            ws, printRuntime,
            printParams.yFilter,
            printParams.mFilter,
            printParams.cFilter,
            D_neg,
            density_spectral,
            print_illuminant,
            light);

        // 5) RAW via print paper sensitivities (log domain → linear sensitivity)
        float raw[3];
        raw_exposures_from_filtered_light(printRuntime.profile, light, raw);

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

    bool DevelopPrintStage::run(const DevelopPrintInputs& in, DevelopPrintOutputs& out) {
        out.printDensity = PrintDensityCMY{};

        if (!in.printRuntime) {
            return false;
        }

        const Print::Profile& p = in.printRuntime->profile;

        auto interpolate_density_gamma = [](const Spectral::Curve& dc, float logE, float gammaFactor) {
            if (dc.lambda_nm.empty()) {
                return 0.0f;
            }

            const float gammaSafe = (std::isfinite(gammaFactor) && gammaFactor > 0.0f)
                ? gammaFactor
                : 1.0f;

            return Spectral::sample_density_at_logE(dc, logE, gammaSafe);
        };

        out.printDensity.v[0] = interpolate_density_gamma(p.dcC, in.printLogRaw.v[0], p.gammaFactor[0]);
        out.printDensity.v[1] = interpolate_density_gamma(p.dcM, in.printLogRaw.v[1], p.gammaFactor[1]);
        out.printDensity.v[2] = interpolate_density_gamma(p.dcY, in.printLogRaw.v[2], p.gammaFactor[2]);
        return true;
    }

} // namespace Pipeline
