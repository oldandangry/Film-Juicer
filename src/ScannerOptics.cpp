// ScannerOptics.cpp

#include "JuicerState.h"

#include <cmath>
#include <cstdint>
#include <limits>

#include "ColorTransforms.h"
#include "Hash.h"
#include "Logging.h"
#include "OutputColor.h"
#include "SpectralProcessing.h"

namespace {

    std::uint64_t hash_color_runtime(
        const Scanner::ScannerMediumRuntime& medium,
        const OutputEncoding::Params& encoding,
        const GeneratedColorSpaces::ColorSpaceEntry& outSpace) {
        const std::uint64_t encHash = Hash::hash_uint64_values(
            {static_cast<std::uint64_t>(OutputEncoding::toIndex(encoding.colorSpace)),
             static_cast<std::uint64_t>(encoding.applyCctfEncoding),
             static_cast<std::uint64_t>(encoding.preserveLinearRange),
             static_cast<std::uint64_t>(encoding.inputIsOutputSpace)});
        return Hash::hash_uint64_values(
            {outSpace.hash,
             medium.illuminant.hash,
             static_cast<std::uint64_t>(medium.medium),
             encHash});
    }

} // namespace

namespace Scanner {

    void normalize_density(
        const ScannerMediumRuntime& medium,
        const float D_cmy[3],
        double D_norm[3]) {
        if (!D_cmy || !D_norm) {
            return;
        }

        if (medium.medium == ScannerMedium::Negative) {
            D_norm[0] =
                (static_cast<double>(D_cmy[0]) + static_cast<double>(medium.range.min_cmy[0])) *
                static_cast<double>(medium.range.inv_max_cmy[0]);
            D_norm[1] =
                (static_cast<double>(D_cmy[1]) + static_cast<double>(medium.range.min_cmy[1])) *
                static_cast<double>(medium.range.inv_max_cmy[1]);
            D_norm[2] =
                (static_cast<double>(D_cmy[2]) + static_cast<double>(medium.range.min_cmy[2])) *
                static_cast<double>(medium.range.inv_max_cmy[2]);
        } else {
            D_norm[0] =
                (static_cast<double>(D_cmy[0]) - static_cast<double>(medium.range.min_cmy[0])) *
                static_cast<double>(medium.range.inv_max_cmy[0]);
            D_norm[1] =
                (static_cast<double>(D_cmy[1]) - static_cast<double>(medium.range.min_cmy[1])) *
                static_cast<double>(medium.range.inv_max_cmy[1]);
            D_norm[2] =
                (static_cast<double>(D_cmy[2]) - static_cast<double>(medium.range.min_cmy[2])) *
                static_cast<double>(medium.range.inv_max_cmy[2]);
        }
    }

    void spectral_to_log_xyz(
        const ScannerMediumRuntime& medium,
        const double D_norm[3],
        double logXYZ[3]) {
        if (!D_norm || !logXYZ) {
            return;
        }

        const Spectral::SpectralTables* tables = medium.tables;
        if (!tables || tables->K <= 0) {
            logXYZ[0] = logXYZ[1] = logXYZ[2] = std::numeric_limits<double>::quiet_NaN();
            return;
        }

        double D_denorm[3];
        if (medium.medium == ScannerMedium::Negative) {
            D_denorm[0] =
                D_norm[0] / static_cast<double>(medium.range.inv_max_cmy[0]) -
                static_cast<double>(medium.range.min_cmy[0]);
            D_denorm[1] =
                D_norm[1] / static_cast<double>(medium.range.inv_max_cmy[1]) -
                static_cast<double>(medium.range.min_cmy[1]);
            D_denorm[2] =
                D_norm[2] / static_cast<double>(medium.range.inv_max_cmy[2]) -
                static_cast<double>(medium.range.min_cmy[2]);
        } else {
            D_denorm[0] =
                D_norm[0] / static_cast<double>(medium.range.inv_max_cmy[0]) +
                static_cast<double>(medium.range.min_cmy[0]);
            D_denorm[1] =
                D_norm[1] / static_cast<double>(medium.range.inv_max_cmy[1]) +
                static_cast<double>(medium.range.min_cmy[1]);
            D_denorm[2] =
                D_norm[2] / static_cast<double>(medium.range.inv_max_cmy[2]) +
                static_cast<double>(medium.range.min_cmy[2]);
        }

        const int K = tables->K;
        const float* epsC = tables->epsC.data();
        const float* epsM = tables->epsM.data();
        const float* epsY = tables->epsY.data();
        const float* Ax = tables->Ax.data();
        const float* Ay = tables->Ay.data();
        const float* Az = tables->Az.data();
        const float* baseDensityMin = tables->baseDensityMin.data();

        const bool useBaseline = tables->hasBaseline;

        double X = 0.0;
        double Y = 0.0;
        double Z = 0.0;
        for (int i = 0; i < K; ++i) {
            const double baseSpectral =
                useBaseline ? static_cast<double>(baseDensityMin[i]) : 0.0;
            const double Dlambda =
                D_denorm[0] * static_cast<double>(epsC[i]) +
                D_denorm[1] * static_cast<double>(epsM[i]) +
                D_denorm[2] * static_cast<double>(epsY[i]) + baseSpectral;

            const double transmittance = std::pow(10.0, -Dlambda);
            const double ax = static_cast<double>(Ax[i]);
            const double ay = static_cast<double>(Ay[i]);
            const double az = static_cast<double>(Az[i]);
            if (std::isfinite(ax)) {
                const double out = transmittance * ax;
                if (!std::isnan(out)) {
                    X += out;
                }
            }
            if (std::isfinite(ay)) {
                const double out = transmittance * ay;
                if (!std::isnan(out)) {
                    Y += out;
                }
            }
            if (std::isfinite(az)) {
                const double out = transmittance * az;
                if (!std::isnan(out)) {
                    Z += out;
                }
            }
        }

        const double invNormalization = static_cast<double>(tables->invYn);
        const double XYZ[3] = {
            X * invNormalization,
            Y * invNormalization,
            Z * invNormalization};

        constexpr double kEps = 1e-10;
        logXYZ[0] = std::log10(std::fmax(XYZ[0], 0.0) + kEps);
        logXYZ[1] = std::log10(std::fmax(XYZ[1], 0.0) + kEps);
        logXYZ[2] = std::log10(std::fmax(XYZ[2], 0.0) + kEps);
    }

} // namespace Scanner

namespace ScannerOptics {

    Scanner::ColorRuntime build_color_runtime(
        const Scanner::ScannerMediumRuntime& medium,
        const OutputEncoding::Params& outputEncoding) {
        Scanner::ColorRuntime rt{};
        if (medium.illuminant.hash == 0) {
            JTRACE("HASH", "FATAL: scanner illuminant hash invalid for color runtime");
            return rt;
        }
        const GeneratedColorSpaces::ColorSpaceEntry& outSpace =
            GeneratedColorSpaces::get(outputEncoding.colorSpace);
        if (outSpace.hash == 0) {
            JTRACE("HASH", "FATAL: generated color space hash invalid");
            return rt;
        }
        Spectral::ChromaticAdaptationWhites whites{};
        whites.source = medium.illuminant.whiteXYZ;
        whites.destination = outSpace.whiteXYZ;
        Spectral::Mat3 adapt = Spectral::build_chromatic_adaptation_matrix(whites);
        for (int i = 0; i < 9; ++i) {
            rt.cat02[i] = adapt.m[i];
        }

        for (int i = 0; i < 9; ++i) {
            rt.xyzToRgb[i] = outSpace.xyzToRgb[i];
        }
        rt.encoding = outputEncoding;
        rt.encoding.inputIsOutputSpace = true;
        rt.illuminantXYZ[0] = medium.illuminant.whiteXYZ[0];
        rt.illuminantXYZ[1] = medium.illuminant.whiteXYZ[1];
        rt.illuminantXYZ[2] = medium.illuminant.whiteXYZ[2];
        rt.hash = hash_color_runtime(medium, rt.encoding, outSpace);
        return rt;
    }

} // namespace ScannerOptics
