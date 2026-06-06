#pragma once

#include <cmath>
#include <cstdint>
#include <string>

#include "Cuda/JuicerCudaPayloads.h"
#include "RenderRecipe.h"

namespace JuicerCuda {

    struct DirectFilmPreparedView {
        DeviceCurveView finalSensB{};
        DeviceCurveView finalSensG{};
        DeviceCurveView finalSensR{};
        DeviceCurveView normalizedDensB{};
        DeviceCurveView normalizedDensG{};
        DeviceCurveView normalizedDensR{};
        const float* tablesAx = nullptr;
        const float* tablesAy = nullptr;
        const float* tablesAz = nullptr;
        const float* tablesIllum = nullptr;
        int tablesK = 0;
        float spdSInv[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        const float* hanatosLut = nullptr;
        int hanatosN = 0;
        const float* hanatosLutIntegrated = nullptr;
        int hanatosNIntegrated = 0;
        const float* mallettBasis = nullptr;
        int mallettBasisK = 0;
        float inputRGBToXYZ[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        float inputXYZAdapt[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        int applyInputChromaticAdapt = 0;
        float refIllumWhiteXYZ[3] = {0.950455f, 1.0f, 1.089058f};
        std::uint64_t finalSensitivityHash = 0;
        std::uint64_t normalizedDensityCurvesHash = 0;
    };

    struct DirectFilmPayloadPack {
        FilmRawPayload filmRaw{};
        FilmExposurePayload filmExposure{};
        FilmDevelopPayload filmDevelop{};
        std::uint64_t densityBoundsHash = 0;
    };

    inline void copy_direct_film_floats(float* dst, const float* src, int count) {
        for (int index = 0; index < count; ++index) {
            dst[index] = src[index];
        }
    }

    inline bool direct_film_curve_ready(const DeviceCurveView& curve, int expectedSamples) {
        return curve.x && curve.y && curve.n == expectedSamples &&
               curve.domainBegin >= 0 && curve.domainEnd >= curve.domainBegin &&
               curve.domainEnd < curve.n;
    }

    inline bool pack_direct_film_payloads(
        const FilmRawRecipe& filmRaw,
        const FilmDevelopRecipe& filmDevelop,
        const DensityBoundsRecipe& densityBounds,
        const DirectFilmPreparedView& prepared,
        const float* autoExposureScaleDevice,
        DirectFilmPayloadPack& out,
        std::string& diagnostic) {
        diagnostic.clear();
        out = DirectFilmPayloadPack{};
        if (filmRaw.finalSensitivityHash == 0 ||
            prepared.finalSensitivityHash != filmRaw.finalSensitivityHash) {
            diagnostic = "ResourceDescriptorMismatch phase=3B field=final_sensitivity";
            return false;
        }
        if (filmDevelop.normalizedDensityCurvesHash == 0 ||
            prepared.normalizedDensityCurvesHash != filmDevelop.normalizedDensityCurvesHash) {
            diagnostic = "ResourceDescriptorMismatch phase=3B field=normalized_density_curves";
            return false;
        }
        if (densityBounds.hash == 0) {
            diagnostic = "ResourceDescriptorMismatch phase=3B field=density_bounds";
            return false;
        }
        if (!direct_film_curve_ready(prepared.finalSensB, 81) ||
            !direct_film_curve_ready(prepared.finalSensG, 81) ||
            !direct_film_curve_ready(prepared.finalSensR, 81)) {
            diagnostic = "MissingRequiredResource phase=3B field=final_sensitivity_device_curves";
            return false;
        }
        const int densitySamples = static_cast<int>(filmDevelop.logExposure.size());
        if (densitySamples <= 0 ||
            !direct_film_curve_ready(prepared.normalizedDensB, densitySamples) ||
            !direct_film_curve_ready(prepared.normalizedDensG, densitySamples) ||
            !direct_film_curve_ready(prepared.normalizedDensR, densitySamples)) {
            diagnostic = "MissingRequiredResource phase=3B field=normalized_density_device_curves";
            return false;
        }
        if (!prepared.tablesAx || !prepared.tablesAy || !prepared.tablesAz ||
            !prepared.tablesIllum || prepared.tablesK != 81) {
            diagnostic = "MissingRequiredResource phase=3B field=spectral_tables";
            return false;
        }
        if (filmRaw.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Hanatos2025) {
            if (!prepared.hanatosLut || prepared.hanatosN <= 0) {
                diagnostic = "MissingRequiredResource phase=3B field=hanatos_lut";
                return false;
            }
        } else if (!prepared.mallettBasis || prepared.mallettBasisK != 81) {
            diagnostic = "MissingRequiredResource phase=3B field=mallett_basis";
            return false;
        }

        out.filmRaw.inputColorSpaceIndex = filmRaw.inputColorSpace;
        out.filmRaw.applyCctfDecoding = filmRaw.inputCctfDecoding ? 1 : 0;
        out.filmRaw.applyInputChromaticAdapt = prepared.applyInputChromaticAdapt;
        out.filmRaw.spectralUpsamplingMode =
            filmRaw.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Mallett2019 ? 1 : 0;
        out.filmRaw.mallettGreenMidgrayScale = filmRaw.mallettGreenMidgrayScale;
        copy_direct_film_floats(out.filmRaw.inputRGBToXYZ, prepared.inputRGBToXYZ, 9);
        copy_direct_film_floats(out.filmRaw.inputXYZAdapt, prepared.inputXYZAdapt, 9);
        copy_direct_film_floats(out.filmRaw.refIllumWhiteXYZ, prepared.refIllumWhiteXYZ, 3);

        const double manualScale = std::exp2(static_cast<double>(filmRaw.manualExposureCompensationEv));
        out.filmExposure.manualExposureScale =
            std::isfinite(manualScale) && manualScale > 0.0 ? static_cast<float>(manualScale) : 1.0f;
        out.filmExposure.exposureScaleDevice = autoExposureScaleDevice;
        out.filmExposure.highlightBoost.boostEv = filmRaw.highlightBoost.boostEv;
        out.filmExposure.highlightBoost.boostRange = filmRaw.highlightBoost.boostRange;
        out.filmExposure.highlightBoost.protectEv = filmRaw.highlightBoost.protectEv;
        out.filmExposure.sensB = prepared.finalSensB;
        out.filmExposure.sensG = prepared.finalSensG;
        out.filmExposure.sensR = prepared.finalSensR;
        out.filmExposure.tablesAx = prepared.tablesAx;
        out.filmExposure.tablesAy = prepared.tablesAy;
        out.filmExposure.tablesAz = prepared.tablesAz;
        out.filmExposure.tablesIllum = prepared.tablesIllum;
        out.filmExposure.tablesK = prepared.tablesK;
        copy_direct_film_floats(out.filmExposure.spdSInv, prepared.spdSInv, 9);
        if (filmRaw.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Hanatos2025) {
            out.filmExposure.hanatosLut = prepared.hanatosLut;
            out.filmExposure.hanatosN = prepared.hanatosN;
            out.filmExposure.hanatosLutIntegrated = prepared.hanatosLutIntegrated;
            out.filmExposure.hanatosNIntegrated = prepared.hanatosNIntegrated;
        } else {
            out.filmExposure.mallettBasis = prepared.mallettBasis;
            out.filmExposure.mallettBasisK = prepared.mallettBasisK;
        }

        out.filmDevelop.gammaFactorB = filmDevelop.densityCurveGamma[2];
        out.filmDevelop.gammaFactorG = filmDevelop.densityCurveGamma[1];
        out.filmDevelop.gammaFactorR = filmDevelop.densityCurveGamma[0];
        out.filmDevelop.densB = prepared.normalizedDensB;
        out.filmDevelop.densG = prepared.normalizedDensG;
        out.filmDevelop.densR = prepared.normalizedDensR;
        out.densityBoundsHash = densityBounds.hash;
        return true;
    }

} // namespace JuicerCuda
