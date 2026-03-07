#include "ScanStage.h"

#include <cmath>
#include <limits>

#include "Scanner.h"
#include "WorkingState.h"

namespace Pipeline {

    namespace {

        inline const Scanner::ScannerMediumRuntime* select_medium_runtime(
            const WorkingState& ws,
            DensityMedium medium)
        {
            return (medium == DensityMedium::Print)
                ? &ws.printMediumRuntime
                : &ws.negativeMediumRuntime;
        }

    } // namespace

    void ScanStage::normalize_density(
        const Scanner::ScannerMediumRuntime& medium,
        const float D_cmy[3],
        double D_norm[3])
    {
        if (!D_cmy || !D_norm) {
            return;
        }

        if (medium.medium == Scanner::ScannerMedium::Negative) {
            D_norm[0] = (static_cast<double>(D_cmy[0]) + static_cast<double>(medium.range.min_cmy[0])) * static_cast<double>(medium.range.inv_max_cmy[0]);
            D_norm[1] = (static_cast<double>(D_cmy[1]) + static_cast<double>(medium.range.min_cmy[1])) * static_cast<double>(medium.range.inv_max_cmy[1]);
            D_norm[2] = (static_cast<double>(D_cmy[2]) + static_cast<double>(medium.range.min_cmy[2])) * static_cast<double>(medium.range.inv_max_cmy[2]);
        }
        else {
            D_norm[0] = static_cast<double>(D_cmy[0]) * static_cast<double>(medium.range.inv_max_cmy[0]);
            D_norm[1] = static_cast<double>(D_cmy[1]) * static_cast<double>(medium.range.inv_max_cmy[1]);
            D_norm[2] = static_cast<double>(D_cmy[2]) * static_cast<double>(medium.range.inv_max_cmy[2]);
        }
    }

    void ScanStage::spectral_to_log_xyz(
        const Scanner::ScannerMediumRuntime& medium,
        const double D_norm[3],
        double logXYZ[3])
    {
        if (!D_norm || !logXYZ) {
            return;
        }

        const Spectral::SpectralTables* tables = medium.tables;
        if (!tables || tables->K <= 0) {
            logXYZ[0] = logXYZ[1] = logXYZ[2] = std::numeric_limits<double>::quiet_NaN();
            return;
        }

        double D_denorm[3];
        if (medium.medium == Scanner::ScannerMedium::Negative) {
            D_denorm[0] = D_norm[0] / static_cast<double>(medium.range.inv_max_cmy[0]) - static_cast<double>(medium.range.min_cmy[0]);
            D_denorm[1] = D_norm[1] / static_cast<double>(medium.range.inv_max_cmy[1]) - static_cast<double>(medium.range.min_cmy[1]);
            D_denorm[2] = D_norm[2] / static_cast<double>(medium.range.inv_max_cmy[2]) - static_cast<double>(medium.range.min_cmy[2]);
        }
        else {
            D_denorm[0] = D_norm[0] / static_cast<double>(medium.range.inv_max_cmy[0]);
            D_denorm[1] = D_norm[1] / static_cast<double>(medium.range.inv_max_cmy[1]);
            D_denorm[2] = D_norm[2] / static_cast<double>(medium.range.inv_max_cmy[2]);
        }

        const int K = tables->K;
        const float* epsC = tables->epsC.data();
        const float* epsM = tables->epsM.data();
        const float* epsY = tables->epsY.data();
        const float* Ax = tables->Ax.data();
        const float* Ay = tables->Ay.data();
        const float* Az = tables->Az.data();
        const float* baseMin = tables->baseMin.data();

        const bool useBaseline = tables->hasBaseline;

        double X = 0.0, Y = 0.0, Z = 0.0;
        for (int i = 0; i < K; ++i) {
            const double baseSpectral = (useBaseline) ? static_cast<double>(baseMin[i]) : 0.0;
            const double Dlambda = D_denorm[0] * static_cast<double>(epsC[i])
                + D_denorm[1] * static_cast<double>(epsM[i])
                + D_denorm[2] * static_cast<double>(epsY[i])
                + baseSpectral;

            const double transmittance = std::pow(10.0, -Dlambda);
            const double ax = static_cast<double>(Ax[i]);
            const double ay = static_cast<double>(Ay[i]);
            const double az = static_cast<double>(Az[i]);
            if (std::isfinite(ax)) {
                const double out = transmittance * ax;
                if (!std::isnan(out)) X += out;
            }
            if (std::isfinite(ay)) {
                const double out = transmittance * ay;
                if (!std::isnan(out)) Y += out;
            }
            if (std::isfinite(az)) {
                const double out = transmittance * az;
                if (!std::isnan(out)) Z += out;
            }
        }

        const double invNormalization = static_cast<double>(tables->invYn);
        const double XYZ[3] = {
            X * invNormalization,
            Y * invNormalization,
            Z * invNormalization
        };

        constexpr double kEps = 1e-10;
        // agx-emulsion parity: do not clamp XYZ before log.
        logXYZ[0] = std::log10(XYZ[0] + kEps);
        logXYZ[1] = std::log10(XYZ[1] + kEps);
        logXYZ[2] = std::log10(XYZ[2] + kEps);
    }

    bool ScanStage::run(const WorkingState& ws, const ScanInputs& in, ScanOutputs& out) {
        out.logXyz = LogXyz{};
        out.xyz = Xyz{};

        const Scanner::ScannerMediumRuntime* mediumRuntime = select_medium_runtime(ws, in.medium);
        if (!mediumRuntime) {
            return false;
        }
        if (!mediumRuntime->tables || mediumRuntime->tables->K <= 0) {
            return false;
        }

        float D_cmy[3];
        if (in.medium == DensityMedium::Print) {
            D_cmy[0] = in.printDensity.v[0];
            D_cmy[1] = in.printDensity.v[1];
            D_cmy[2] = in.printDensity.v[2];
        }
        else {
            D_cmy[0] = in.negativeDensity.v[0];
            D_cmy[1] = in.negativeDensity.v[1];
            D_cmy[2] = in.negativeDensity.v[2];
        }

        double D_norm[3];
        normalize_density(*mediumRuntime, D_cmy, D_norm);

        double logXYZ[3] = { 0.0, 0.0, 0.0 };
        spectral_to_log_xyz(*mediumRuntime, D_norm, logXYZ);

        out.logXyz.v[0] = logXYZ[0];
        out.logXyz.v[1] = logXYZ[1];
        out.logXyz.v[2] = logXYZ[2];

        out.xyz.v[0] = std::pow(10.0, out.logXyz.v[0]);
        out.xyz.v[1] = std::pow(10.0, out.logXyz.v[1]);
        out.xyz.v[2] = std::pow(10.0, out.logXyz.v[2]);
        return true;
    }

} // namespace Pipeline
