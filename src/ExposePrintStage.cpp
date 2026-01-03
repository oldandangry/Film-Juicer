#include "ExposePrintStage.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

#include "Couplers.h"
#include "DevelopFilmStage.h"
#include "ExposeFilmStage.h"
#include "Logging.h"
#include "Print.h"
#include "SpectralData.h"
#include "SpectralProcessing.h"
#include "WorkingState.h"

namespace Pipeline {
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
            density_out.assign(static_cast<size_t>(std::max(K, 0)), 0.0f);
            if (K <= 0) return;

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
            illuminant_out.assign(static_cast<size_t>(std::max(K, 0)), 0.0f);
            if (K <= 0) return;

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
            float yShiftSteps,
            float mShiftSteps,
            float cShiftSteps,
            PrintPipelineScratch& scratch)
        {
            const int shapeK = Spectral::gShape.K;
            if (shapeK <= 0) {
                scratch.enlargerIlluminantFilteredValid = false;
                scratch.enlargerIlluminantRuntime = nullptr;
                scratch.enlargerIlluminantShapeK = 0;
                scratch.Ee_expose.clear();
                return;
            }

            // Match compose_dichroic_amount behavior: treat non-finite delta steps as 0 so cache keys
            // do not thrash on NaN inputs (NaN != NaN).
            const float yKey = std::isfinite(yShiftSteps) ? yShiftSteps : 0.0f;
            const float mKey = std::isfinite(mShiftSteps) ? mShiftSteps : 0.0f;
            const float cKey = std::isfinite(cShiftSteps) ? cShiftSteps : 0.0f;

            if (scratch.enlargerIlluminantFilteredValid &&
                scratch.enlargerIlluminantRuntime == &rt &&
                scratch.enlargerIlluminantYShiftSteps == yKey &&
                scratch.enlargerIlluminantMShiftSteps == mKey &&
                scratch.enlargerIlluminantCShiftSteps == cKey &&
                scratch.enlargerIlluminantShapeK == shapeK) {
                return;
            }

            build_enlarger_illuminant_filtered(rt, yKey, mKey, cKey, scratch.Ee_expose);
            scratch.enlargerIlluminantFilteredValid = true;
            scratch.enlargerIlluminantRuntime = &rt;
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

        void compute_preflash_raw(
            const WorkingState& ws,
            const Print::Runtime& rt,
            PrintPipelineScratch& scratch,
            float rawOut[3])
        {
            rawOut[0] = rawOut[1] = rawOut[2] = 0.0f;
            const int K = Spectral::gShape.K;
            if (K <= 0) return;
            if (ws.tablesView.K <= 0) return;

            // agx-emulsion parity: preflash is computed via density_to_light(density_base, preflash_illuminant),
            // then contracted against print paper sensitivity (no clamp).
            const float Dbase[3] = { 0.0f, 0.0f, 0.0f };
            density_to_filtered_light_agx(
                ws, rt,
                /*yShiftSteps=*/0.0f,
                /*mShiftSteps=*/0.0f,
                /*cShiftSteps=*/0.0f,
                Dbase,
                scratch.Tpreflash,
                scratch.Ee_viewed,
                scratch.Ee_preflash);

            raw_exposures_from_filtered_light(rt.profile, scratch.Ee_preflash, rawOut);
        }

        void ensure_cached_preflash_raw(
            const WorkingState& ws,
            const Print::Runtime& rt,
            PrintPipelineScratch& scratch)
        {
            const int shapeK = Spectral::gShape.K;
            if (shapeK <= 0 || ws.tablesView.K <= 0) {
                scratch.preflashRawValid = false;
                scratch.preflashRuntime = nullptr;
                scratch.preflashWsBuildCounter = 0;
                scratch.preflashShapeK = 0;
                scratch.preflashRaw[0] = scratch.preflashRaw[1] = scratch.preflashRaw[2] = 0.0f;
                return;
            }

            if (scratch.preflashRawValid &&
                scratch.preflashRuntime == &rt &&
                scratch.preflashWsBuildCounter == ws.buildCounter &&
                scratch.preflashShapeK == shapeK) {
                return;
            }

            compute_preflash_raw(ws, rt, scratch, scratch.preflashRaw);
            scratch.preflashRawValid = true;
            scratch.preflashRuntime = &rt;
            scratch.preflashWsBuildCounter = ws.buildCounter;
            scratch.preflashShapeK = shapeK;
        }

    } // namespace

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
            prm.yFilter,
            prm.mFilter,
            /*cShiftSteps=*/0.0f,
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
	            /*cShiftSteps=*/0.0f,
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

} // namespace Pipeline
