#include "DevelopPrintStage.h"

#include <cmath>

#include "FilmProcessing.h"
#include "Print.h"

namespace Pipeline {

    namespace {

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

    } // namespace

    bool DevelopPrintStage::run(const DevelopPrintInputs& in, DevelopPrintOutputs& out) {
        out.printDensity = PrintDensityCMY{};

        if (!in.printRuntime) {
            return false;
        }

        const Print::Profile& p = in.printRuntime->profile;

        out.printDensity.v[0] = interpolate_density_gamma(p.dcC, in.printLogRaw.v[0], p.gammaFactor[0]);
        out.printDensity.v[1] = interpolate_density_gamma(p.dcM, in.printLogRaw.v[1], p.gammaFactor[1]);
        out.printDensity.v[2] = interpolate_density_gamma(p.dcY, in.printLogRaw.v[2], p.gammaFactor[2]);
        return true;
    }

} // namespace Pipeline
