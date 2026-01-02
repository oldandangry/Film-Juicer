#include "DevelopFilmStage.h"

#include <cmath>
#include <limits>

#include "WorkingState.h"

namespace Pipeline {

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

    } // namespace

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

} // namespace Pipeline
