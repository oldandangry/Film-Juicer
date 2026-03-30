#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

#include "ColorTransforms.h"
#include "FilmProcessing.h"
#include "Logging.h"
#include "mainProcessing.h"
#include "PipelineTypes.h"
#include "Print.h"
#include "JuicerState.h"

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

        inline bool finite_curve_domain(const Spectral::Curve& c, float& xmin, float& xmax) {
            const size_t n = c.lambda_nm.size();
            if (n == 0 || c.linear.size() != n) {
                xmin = 0.0f;
                xmax = 0.0f;
                return false;
            }

            size_t begin = 0;
            while (begin < n && !std::isfinite(c.lambda_nm[begin])) {
                ++begin;
            }
            if (begin == n) {
                xmin = 0.0f;
                xmax = 0.0f;
                return false;
            }

            size_t end = n - 1;
            while (end > begin && !std::isfinite(c.lambda_nm[end])) {
                --end;
            }

            xmin = c.lambda_nm[begin];
            xmax = c.lambda_nm[end];
            return std::isfinite(xmin) && std::isfinite(xmax) && (xmax >= xmin);
        }

        inline float sanitize_inf_logE_for_curve(float logE, const Spectral::Curve& c) {
            if (std::isfinite(logE) || std::isnan(logE)) {
                return logE;
            }

            float xmin = 0.0f, xmax = 0.0f;
            if (!finite_curve_domain(c, xmin, xmax)) {
                return logE;
            }

            if (logE > 0.0f) {
                return xmax;
            }
            return xmin;
        }

        inline void sample_negative_densities_spatial_dir(
            const WorkingState& ws,
            const float logE_BGR[3],
            float D_cmy_out[3])
        {
            const Spectral::Curve& cB = ws.dirPrecorrected ? ws.dirDensB : ws.densB;
            const Spectral::Curve& cG = ws.dirPrecorrected ? ws.dirDensG : ws.densG;
            const Spectral::Curve& cR = ws.dirPrecorrected ? ws.dirDensR : ws.densR;

            const float leB = sanitize_inf_logE_for_curve(logE_BGR[0], cB);
            const float leG = sanitize_inf_logE_for_curve(logE_BGR[1], cG);
            const float leR = sanitize_inf_logE_for_curve(logE_BGR[2], cR);

            const float D_Y = Spectral::sample_density_at_logE(cB, leB, ws.gammaFactorB);
            const float D_M = Spectral::sample_density_at_logE(cG, leG, ws.gammaFactorG);
            const float D_C = Spectral::sample_density_at_logE(cR, leR, ws.gammaFactorR);

            D_cmy_out[0] = D_C;
            D_cmy_out[1] = D_M;
            D_cmy_out[2] = D_Y;
        }

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

    bool ExposeFilmStage::run(const WorkingState& ws, const ExposeFilmInputs& in, ExposeFilmOutputs& out) {
        float rgbIn[3] = { in.rgb.v[0], in.rgb.v[1], in.rgb.v[2] };
        float E[3] = { 0.0f, 0.0f, 0.0f };

        const float exposureScaleSafe = (std::isfinite(in.exposureScale) && in.exposureScale > 0.0f)
            ? in.exposureScale
            : 1.0f;

        const Spectral::SpectralTables* tablesSPD =
            (ws.spdReady && ws.tablesRef.K > 0) ? &ws.tablesRef : nullptr;

        Spectral::rgb_input_to_film_raw(
            rgbIn, E, exposureScaleSafe,
            ws.filmRaw,
            tablesSPD,
            (ws.spdReady ? ws.spdSInv : nullptr),
            ws.spdReady,
            ws.sensB, ws.sensG, ws.sensR);

        out.filmRaw.v[0] = E[0];
        out.filmRaw.v[1] = E[1];
        out.filmRaw.v[2] = E[2];
        return true;
    }

    FilmLogRaw DevelopFilmStage::compute_log_raw(const FilmRaw& filmRaw) {
        FilmLogRaw out{};

        constexpr float kEps = 1e-10f;
        out.v[0] = std::log10(fmax_agx(filmRaw.v[0], 0.0f) + kEps);
        out.v[1] = std::log10(fmax_agx(filmRaw.v[1], 0.0f) + kEps);
        out.v[2] = std::log10(fmax_agx(filmRaw.v[2], 0.0f) + kEps);

        return out;
    }

    bool DevelopFilmStage::run(const WorkingState& ws, const DevelopFilmInputs& in, DevelopFilmOutputs& out) {
        out.filmLogRaw = compute_log_raw(in.filmRaw);
        out.negativeDensity = NegativeDensityCMY{};

        float logE[3] = { out.filmLogRaw.v[0], out.filmLogRaw.v[1], out.filmLogRaw.v[2] };

        if (in.useSpatialDIR) {
            logE[0] -= in.spatialLogECorrectionsYMC[0]; // Y -> Blue layer
            logE[1] -= in.spatialLogECorrectionsYMC[1]; // M -> Green layer
            logE[2] -= in.spatialLogECorrectionsYMC[2]; // C -> Red layer

            float D_cmy[3] = { 0.0f, 0.0f, 0.0f };
            sample_negative_densities_spatial_dir(ws, logE, D_cmy);
            out.negativeDensity.v[0] = D_cmy[0];
            out.negativeDensity.v[1] = D_cmy[1];
            out.negativeDensity.v[2] = D_cmy[2];
            return true;
        }

        const Couplers::Runtime* dirRT = in.dirRuntime ? in.dirRuntime : nullptr;
        Couplers::Runtime dummyRT{};
        const Couplers::Runtime& runtime = dirRT ? *dirRT : dummyRT;

        // Prevent +/-inf logE from collapsing to NaN in sample_density_at_logE (fast_interp parity).
        logE[0] = sanitize_inf_logE_for_curve(logE[0], ws.densB);
        logE[1] = sanitize_inf_logE_for_curve(logE[1], ws.densG);
        logE[2] = sanitize_inf_logE_for_curve(logE[2], ws.densR);

        float D_cmy[3] = { 0.0f, 0.0f, 0.0f };
        sample_negative_densities(
            ws,
            runtime,
            logE,
            D_cmy,
            in.applyDirRuntime ? DirSampleMode::ApplyRuntime : DirSampleMode::BypassRuntime);

        out.negativeDensity.v[0] = D_cmy[0];
        out.negativeDensity.v[1] = D_cmy[1];
        out.negativeDensity.v[2] = D_cmy[2];
        return true;
    }

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

        // 2) DWG -> per-layer exposures (negative leg); apply camera EV exactly once here.
        // NOTE: Do not pre-scale rgbMid by cameraExposureScale - avoids double-applying EV.
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

        // 5) RAW via print paper sensitivities (log domain -> linear sensitivity)
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

    PipelineRunner::PipelineRunner(const PipelineRunnerConfig& cfg) : cfg_(cfg) {}

    float PipelineRunner::compute_midgray_factor(
        const WorkingState& ws,
        const Print::Runtime& printRuntime,
        const Print::Params& printParams,
        const Couplers::Runtime& dirRT,
        float exposureCompScale)
    {
        return ExposePrintStage::compute_midgray_factor(
            ws,
            printRuntime,
            printParams,
            dirRT,
            exposureCompScale);
    }

    bool PipelineRunner::run_density_pixel(
        const WorkingState& ws,
        const DensityPixelInputs& in,
        DensityPixelOutputs& out) const
    {
        out = DensityPixelOutputs{};

        FilmRaw filmRaw{};
        if (in.useFilmRawOverride) {
            filmRaw = in.filmRawOverride;
        }
        else {
            ExposeFilmInputs exposeIn{};
            exposeIn.rgb = in.rgb;
            exposeIn.exposureScale = in.exposureScale;

            ExposeFilmOutputs exposeOut{};
            if (!ExposeFilmStage::run(ws, exposeIn, exposeOut)) {
                return false;
            }
            filmRaw = exposeOut.filmRaw;
        }

        DevelopFilmInputs devIn{};
        devIn.filmRaw = filmRaw;
        devIn.dirRuntime = in.dirRuntime;
        devIn.applyDirRuntime = in.applyDirRuntime;
        devIn.useSpatialDIR = in.useSpatialDIR;
        devIn.spatialLogECorrectionsYMC[0] = in.spatialLogECorrectionsYMC[0];
        devIn.spatialLogECorrectionsYMC[1] = in.spatialLogECorrectionsYMC[1];
        devIn.spatialLogECorrectionsYMC[2] = in.spatialLogECorrectionsYMC[2];

        DevelopFilmOutputs devOut{};
        if (!DevelopFilmStage::run(ws, devIn, devOut)) {
            return false;
        }

        out.filmRaw = filmRaw;
        out.filmLogRaw = devOut.filmLogRaw;
        out.negativeDensity = devOut.negativeDensity;
        out.medium = DensityMedium::Negative;

        if (!cfg_.enablePrint) {
            return true;
        }

        if (!in.printRuntime || !in.printParams || in.printParams->bypass) {
            return true;
        }
        if (!in.printScratch) {
            return false;
        }

        ExposePrintInputs exposePrintIn{};
        exposePrintIn.printRuntime = in.printRuntime;
        exposePrintIn.printParams = in.printParams;
        exposePrintIn.negativeDensity = devOut.negativeDensity;
        exposePrintIn.midgrayFactor = in.midgrayFactor;

        ExposePrintOutputs exposePrintOut{};
        if (!ExposePrintStage::run(ws, exposePrintIn, exposePrintOut, *in.printScratch)) {
            return false;
        }

        DevelopPrintInputs developPrintIn{};
        developPrintIn.printRuntime = in.printRuntime;
        developPrintIn.printLogRaw = exposePrintOut.printLogRaw;

        DevelopPrintOutputs developPrintOut{};
        if (!DevelopPrintStage::run(developPrintIn, developPrintOut)) {
            return false;
        }

        out.printDensity = developPrintOut.printDensity;
        out.medium = DensityMedium::Print;
        return true;
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

namespace SpatialDIR {

    namespace {

        inline void resize_noinit(std::vector<float>& v, size_t n) {
            // Keep capacity stable and avoid redundant full clears; hot-path writes overwrite every used sample.
            if (v.size() != n) {
                v.resize(n);
            }
        }

        void blurChannelSeparable(
            const std::vector<float>& src,
            std::vector<float>& tmp,
            std::vector<float>& dst,
            int width,
            int height,
            const std::vector<float>& k)
        {
            const size_t total = size_t(width) * size_t(height);
            if (total == 0) {
                tmp.clear();
                dst.clear();
                return;
            }

            resize_noinit(tmp, total);
            const int radius = int(k.size() / 2);
            const auto reflectIndex = [](int idx, int size) -> int {
                if (size <= 1) {
                    return 0;
                }
                while (idx < 0 || idx >= size) {
                    if (idx < 0) {
                        idx = -idx;
                    }
                    else {
                        idx = 2 * size - idx - 2;
                    }
                }
                return idx;
            };
            for (int y = 0; y < height; ++y) {
                const float* srow = &src[size_t(y * width)];
                float* trow = &tmp[size_t(y * width)];
                for (int x = 0; x < width; ++x) {
                    float acc = 0.0f;
                    for (int j = -radius; j <= radius; ++j) {
                        const int xx = reflectIndex(x + j, width);
                        acc += srow[xx] * k[size_t(j + radius)];
                    }
                    trow[x] = acc;
                }
            }
            resize_noinit(dst, total);
            for (int x = 0; x < width; ++x) {
                for (int y = 0; y < height; ++y) {
                    float acc = 0.0f;
                    for (int j = -radius; j <= radius; ++j) {
                        const int yy = reflectIndex(y + j, height);
                        acc += tmp[size_t(yy * width + x)] * k[size_t(j + radius)];
                    }
                    dst[size_t(y * width + x)] = acc;
                }
            }
        }

    } // namespace

    void buildSpatialDIRCorrections(
        int width,
        int height,
        const WorkingState& ws,
        const Couplers::Runtime& dirRT,
        float exposureScale,
        const Callbacks& callbacks,
        JuicerProc::SpatialDIRWorkspace& work,
        std::vector<float>& kernelCache)
    {
        const size_t total = size_t(width) * size_t(height);
        resize_noinit(work.filmRaw_B, total);
        resize_noinit(work.filmRaw_G, total);
        resize_noinit(work.filmRaw_R, total);
        resize_noinit(work.corrY, total);
        resize_noinit(work.corrM, total);
        resize_noinit(work.corrC, total);
        resize_noinit(work.corrYBlur, total);
        resize_noinit(work.corrMBlur, total);
        resize_noinit(work.corrCBlur, total);
        resize_noinit(work.tmp, total);

        if (!dirRT.active) {
            return;
        }

        if (!callbacks.fetchRGB || !callbacks.abortCheck) {
            JTRACE("DIR", "FATAL: SpatialDIR callbacks not provided");
            return;
        }

        const bool verboseDiagnostics = JTRACE_ENABLED(3);
        auto should_abort = [&]() -> bool {
            return callbacks.abortCheck(callbacks.user);
        };
        auto clear_workspace_on_abort = [&]() {
            std::fill(work.filmRaw_B.begin(), work.filmRaw_B.end(), 0.0f);
            std::fill(work.filmRaw_G.begin(), work.filmRaw_G.end(), 0.0f);
            std::fill(work.filmRaw_R.begin(), work.filmRaw_R.end(), 0.0f);
            std::fill(work.corrY.begin(), work.corrY.end(), 0.0f);
            std::fill(work.corrM.begin(), work.corrM.end(), 0.0f);
            std::fill(work.corrC.begin(), work.corrC.end(), 0.0f);
            std::fill(work.corrYBlur.begin(), work.corrYBlur.end(), 0.0f);
            std::fill(work.corrMBlur.begin(), work.corrMBlur.end(), 0.0f);
            std::fill(work.corrCBlur.begin(), work.corrCBlur.end(), 0.0f);
        };
        auto trace_abort_fast = [&](const char* stage) {
            if (!verboseDiagnostics) {
                return;
            }
            JTRACE_VERBOSE("MSCPU", std::string("path=spatial_dir event=abort_fast stage=") + stage);
        };

        // Per agx-emulsion parity: use same sensitivities everywhere (no separate "before balance" state).
        // Profiles contain pre-balanced sensitivities; spatial DIR and mid-gray must match pixel render.
        const Spectral::Curve& sensB_forExposure = ws.sensB;
        const Spectral::Curve& sensG_forExposure = ws.sensG;
        const Spectral::Curve& sensR_forExposure = ws.sensR;

        // DEBUG: Log WorkingState sensitivity curves before use
        if (verboseDiagnostics) {
            const int idx_450 = 14, idx_520 = 28, idx_650 = 54;
            std::ostringstream oss;
            oss << "WS_SENS_PREUSE: B[450nm]=" << (sensB_forExposure.linear.size() > idx_450 ? sensB_forExposure.linear[idx_450] : -999.0f)
                << " G[450nm]=" << (sensG_forExposure.linear.size() > idx_450 ? sensG_forExposure.linear[idx_450] : -999.0f)
                << " R[450nm]=" << (sensR_forExposure.linear.size() > idx_450 ? sensR_forExposure.linear[idx_450] : -999.0f)
                << " | B[520nm]=" << (sensB_forExposure.linear.size() > idx_520 ? sensB_forExposure.linear[idx_520] : -999.0f)
                << " G[520nm]=" << (sensG_forExposure.linear.size() > idx_520 ? sensG_forExposure.linear[idx_520] : -999.0f)
                << " R[520nm]=" << (sensR_forExposure.linear.size() > idx_520 ? sensR_forExposure.linear[idx_520] : -999.0f)
                << " | B[650nm]=" << (sensB_forExposure.linear.size() > idx_650 ? sensB_forExposure.linear[idx_650] : -999.0f)
                << " G[650nm]=" << (sensG_forExposure.linear.size() > idx_650 ? sensG_forExposure.linear[idx_650] : -999.0f)
                << " R[650nm]=" << (sensR_forExposure.linear.size() > idx_650 ? sensR_forExposure.linear[idx_650] : -999.0f);
            JTRACE_VERBOSE("SPECTRAL", oss.str());
        }

        // Pass A: sample exposures, convert to logE, compute DIR corrections per pixel.
        const int center_xx = width / 2;
        const int center_yy = height / 2;

        Pipeline::PipelineRunnerConfig runnerCfg{};
        runnerCfg.enablePrint = false;
        const Pipeline::PipelineRunner runner(runnerCfg);

        bool aborted = false;
        for (int yy = 0; yy < height; ++yy) {
            if (should_abort()) {
                aborted = true;
                break;
            }
            for (int xx = 0; xx < width; ++xx) {
                if ((xx & 63) == 0 && should_abort()) {
                    aborted = true;
                    break;
                }
                const size_t idx = size_t(yy) * size_t(width) + size_t(xx);
                float rgbIn[3] = { 0.0f, 0.0f, 0.0f };
                if (!callbacks.fetchRGB(callbacks.user, xx, yy, rgbIn)) {
                    work.filmRaw_B[idx] = work.filmRaw_G[idx] = work.filmRaw_R[idx] = 0.0f;
                    work.corrY[idx] = work.corrM[idx] = work.corrC[idx] = 0.0f;
                    continue;
                }

                // SPD DEBUG: Log input RGB for center pixel
                if (verboseDiagnostics && xx == center_xx && yy == center_yy) {
                    std::ostringstream oss;
                    oss << "INPUT_RGB tile_pixel(" << xx << "," << yy << "): R=" << rgbIn[0] << " G=" << rgbIn[1] << " B=" << rgbIn[2];
                    JTRACE_VERBOSE("SPECTRAL", oss.str());
                }

                Pipeline::DensityPixelInputs pxIn{};
                pxIn.rgb.v[0] = rgbIn[0];
                pxIn.rgb.v[1] = rgbIn[1];
                pxIn.rgb.v[2] = rgbIn[2];
                pxIn.exposureScale = exposureScale;
                pxIn.dirRuntime = &dirRT;
                pxIn.applyDirRuntime = false; // Pass A wants pre-DIR densities for correction computation.

                Pipeline::DensityPixelOutputs pxOut{};
                if (!runner.run_density_pixel(ws, pxIn, pxOut)) {
                    work.filmRaw_B[idx] = work.filmRaw_G[idx] = work.filmRaw_R[idx] = 0.0f;
                    work.corrY[idx] = work.corrM[idx] = work.corrC[idx] = 0.0f;
                    continue;
                }

                work.filmRaw_B[idx] = pxOut.filmRaw.v[0];
                work.filmRaw_G[idx] = pxOut.filmRaw.v[1];
                work.filmRaw_R[idx] = pxOut.filmRaw.v[2];

                // SPD DEBUG: Log film raw exposure (pre-log) for center pixel
                if (verboseDiagnostics && xx == center_xx && yy == center_yy) {
                    std::ostringstream oss;
                    oss << "FILM_RAW tile_pixel(" << xx << "," << yy << "): B=" << pxOut.filmRaw.v[0]
                        << " G=" << pxOut.filmRaw.v[1]
                        << " R=" << pxOut.filmRaw.v[2];
                    JTRACE_VERBOSE("SPECTRAL", oss.str());

                    // Log sensitivity curve values at key wavelengths
                    const int idx_450 = 14;  // (450-380)/5 = 14
                    const int idx_520 = 28;  // (520-380)/5 = 28
                    const int idx_650 = 54;  // (650-380)/5 = 54
                    if (!sensB_forExposure.linear.empty() && !sensG_forExposure.linear.empty() && !sensR_forExposure.linear.empty()) {
                        std::ostringstream oss1, oss2, oss3;
                        oss1 << "SENS_CURVES tile_pixel(" << xx << "," << yy << "): B[450nm]=" << sensB_forExposure.linear[idx_450]
                            << " G[450nm]=" << sensG_forExposure.linear[idx_450] << " R[450nm]=" << sensR_forExposure.linear[idx_450];
                        oss2 << "SENS_CURVES tile_pixel(" << xx << "," << yy << "): B[520nm]=" << sensB_forExposure.linear[idx_520]
                            << " G[520nm]=" << sensG_forExposure.linear[idx_520] << " R[520nm]=" << sensR_forExposure.linear[idx_520];
                        oss3 << "SENS_CURVES tile_pixel(" << xx << "," << yy << "): B[650nm]=" << sensB_forExposure.linear[idx_650]
                            << " G[650nm]=" << sensG_forExposure.linear[idx_650] << " R[650nm]=" << sensR_forExposure.linear[idx_650];
                        JTRACE_VERBOSE("SPECTRAL", oss1.str());
                        JTRACE_VERBOSE("SPECTRAL", oss2.str());
                        JTRACE_VERBOSE("SPECTRAL", oss3.str());
                    }
                }

                const float leB = pxOut.filmLogRaw.v[0];
                const float leG = pxOut.filmLogRaw.v[1];
                const float leR = pxOut.filmLogRaw.v[2];

                // Convert CMY -> YMC to match Couplers::ApplyInputLogE contract.
                const float D_Y = pxOut.negativeDensity.v[2];
                const float D_M = pxOut.negativeDensity.v[1];
                const float D_C = pxOut.negativeDensity.v[0];

                float aCorr[3];
                Couplers::ApplyInputLogE io{ { leB, leG, leR }, { D_Y, D_M, D_C } };
                Couplers::compute_logE_corrections(io, dirRT, aCorr);
                for (float& v : aCorr) {
                    if (!std::isfinite(v)) v = 0.0f;
                }
                work.corrY[idx] = aCorr[0];
                work.corrM[idx] = aCorr[1];
                work.corrC[idx] = aCorr[2];
            }
            if (aborted) {
                break;
            }
        }

        if (aborted || should_abort()) {
            trace_abort_fast("pre_blur");
            clear_workspace_on_abort();
            return;
        }

        // Blur corrections spatially (shared between preview + render path)
        kernelCache.clear();
        buildGaussianKernel(dirRT.spatialSigmaPixels, kernelCache);
        if (should_abort()) {
            trace_abort_fast("post_kernel");
            clear_workspace_on_abort();
            return;
        }
        blurChannelSeparable(work.corrY, work.tmp, work.corrYBlur, width, height, kernelCache);
        if (should_abort()) {
            trace_abort_fast("post_blur_y");
            clear_workspace_on_abort();
            return;
        }
        blurChannelSeparable(work.corrM, work.tmp, work.corrMBlur, width, height, kernelCache);
        if (should_abort()) {
            trace_abort_fast("post_blur_m");
            clear_workspace_on_abort();
            return;
        }
        blurChannelSeparable(work.corrC, work.tmp, work.corrCBlur, width, height, kernelCache);
        if (should_abort()) {
            trace_abort_fast("post_blur_c");
            clear_workspace_on_abort();
            return;
        }
        if (should_abort()) {
            trace_abort_fast("pre_clamp");
            clear_workspace_on_abort();
            return;
        }

        auto scrubClamp = [](std::vector<float>& v) {
            for (float& t : v) {
                if (!std::isfinite(t)) t = 0.0f;
                if (t < -10.0f) t = -10.0f;
                if (t > 10.0f) t = 10.0f;
            }
        };
        scrubClamp(work.corrYBlur);
        scrubClamp(work.corrMBlur);
        scrubClamp(work.corrCBlur);
    }

} // namespace SpatialDIR
