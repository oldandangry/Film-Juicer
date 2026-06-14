#include "RenderRecipe.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "Hash.h"

namespace {

    template <typename T>
    void hash_value(std::uint64_t& hash, const T& value) {
        Hash::hash_bytes_update(hash, &value, sizeof(value));
    }

    void hash_string(std::uint64_t& hash, const std::string& value) {
        Hash::hash_bytes_update(hash, value.data(), value.size());
    }

    std::uint64_t hash_nan_preserving_floats(const float* values, std::size_t count) {
        const Hash::FloatSpanHash hashes = Hash::hash_float_span_with_nan_mask(values, count);
        return Hash::hash_uint64_values({hashes.valueHash, hashes.nanMaskHash});
    }

    bool compute_authored_extrema(
        const std::vector<std::array<float, 3>>& curves,
        std::array<float, 3>& outMin,
        std::array<float, 3>& outMax) {
        outMin.fill(std::numeric_limits<float>::infinity());
        outMax.fill(-std::numeric_limits<float>::infinity());
        std::array<bool, 3> found{{false, false, false}};
        for (const std::array<float, 3>& row : curves) {
            for (std::size_t channel = 0; channel < row.size(); ++channel) {
                const float value = row[channel];
                if (!std::isfinite(value)) {
                    continue;
                }
                outMin[channel] = std::min(outMin[channel], value);
                outMax[channel] = std::max(outMax[channel], value);
                found[channel] = true;
            }
        }
        return found[0] && found[1] && found[2];
    }

    bool normalize_density_curves(
        const std::vector<std::array<float, 3>>& authored,
        std::vector<std::array<float, 3>>& normalized,
        std::array<float, 3>& outMin,
        std::array<float, 3>& outMax) {
        if (!compute_authored_extrema(authored, outMin, outMax)) {
            return false;
        }
        normalized = authored;
        for (std::array<float, 3>& row : normalized) {
            for (std::size_t channel = 0; channel < row.size(); ++channel) {
                if (std::isfinite(row[channel])) {
                    row[channel] -= outMin[channel];
                }
            }
        }
        return true;
    }

    Spektrafilm::AutoExposureMethod auto_exposure_method_from_index(int index) {
        switch (index) {
            case 0:
                return Spektrafilm::AutoExposureMethod::CenterWeighted;
            case 1:
                return Spektrafilm::AutoExposureMethod::Average;
            case 2:
                return Spektrafilm::AutoExposureMethod::Median;
            case 3:
                return Spektrafilm::AutoExposureMethod::Partial;
            case 4:
                return Spektrafilm::AutoExposureMethod::Matrix;
            case 5:
                return Spektrafilm::AutoExposureMethod::MultiZone;
            case 6:
                return Spektrafilm::AutoExposureMethod::HighlightWeighted;
            default:
                return Spektrafilm::AutoExposureMethod::CenterWeighted;
        }
    }

    bool auto_exposure_method_index_valid(int index) {
        return index >= 0 && index <= 6;
    }

    std::array<float, 3> copy_filter_triplet(const std::array<double, 3>& values, bool active) {
        std::array<float, 3> out{{active ? static_cast<float>(values[0]) : 0.0f,
                                  static_cast<float>(values[1]),
                                  static_cast<float>(values[2])}};
        return out;
    }

    std::uint64_t hash_profile_route(const ProfileRoute& route) {
        std::uint64_t hash = Hash::kFnvOffset;
        hash_string(hash, route.filmProfileKey);
        hash_value(hash, route.filmProfileAssetVersionToken);
        hash_value(hash, route.scanRoute);
        hash_value(hash, route.captureSupport);
        hash_value(hash, route.captureStage);
        hash_value(hash, route.capturePolarity);
        hash_value(hash, route.captureUse);
        hash_value(hash, route.captureAntihalation);
        hash_value(hash, route.captureChannelModel);
        if (Spektrafilm::scan_route_is_print(route.scanRoute)) {
            hash_string(hash, route.printProfileKey);
            hash_value(hash, route.printProfileAssetVersionToken);
        }
        return hash;
    }

    std::uint64_t hash_film_raw_recipe(const FilmRawRecipe& recipe) {
        std::uint64_t hash = Hash::kFnvOffset;
        hash_value(hash, recipe.inputColorSpace);
        hash_value(hash, recipe.inputCctfDecoding);
        hash_value(hash, recipe.rgbToRawMethod);
        hash_value(hash, recipe.autoExposureEnabled);
        hash_value(hash, recipe.autoExposureMethod);
        hash_value(hash, recipe.manualExposureCompensationEv);
        hash_value(hash, recipe.filmFormatLongEdgeMm);
        hash_value(hash, recipe.cameraBandPass.active);
        Hash::hash_bytes_update(hash, recipe.cameraBandPass.uv.data(), sizeof(recipe.cameraBandPass.uv));
        Hash::hash_bytes_update(hash, recipe.cameraBandPass.ir.data(), sizeof(recipe.cameraBandPass.ir));
        hash_value(hash, recipe.highlightBoost.boostEv);
        hash_value(hash, recipe.highlightBoost.boostRange);
        hash_value(hash, recipe.highlightBoost.protectEv);
        hash_value(hash, recipe.finalSensitivityHash);
        if (recipe.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Hanatos2025) {
            hash_value(hash, recipe.hanatosLutHash);
            hash_string(hash, recipe.hanatos.referenceIlluminant);
        } else {
            hash_value(hash, recipe.mallettGreenMidgrayScale);
        }
        return hash;
    }

    float camera_filter_sample(float wavelength, const std::array<float, 3>& filter, bool uv) {
        const float amplitude = std::clamp(filter[0], 0.0f, 1.0f);
        if (!(amplitude > 0.0f)) {
            return 1.0f;
        }
        float width = filter[2];
        if (!std::isfinite(width) || std::abs(width) < 1e-6f) {
            width = uv ? 1e-6f : -1e-6f;
        }
        width = uv ? std::abs(width) : -std::abs(width);
        const float sigmoid = 0.5f * (std::erf((wavelength - filter[1]) / width) + 1.0f);
        return 1.0f - amplitude + amplitude * sigmoid;
    }

    float hanatos_window_sample(float wavelength, const std::array<float, 4>& params) {
        constexpr float kSqrt2 = 1.4142135623730950488f;
        const float uv = 0.5f * (1.0f + std::erf((wavelength - params[0]) / (params[1] * kSqrt2)));
        const float ir = 0.5f * (1.0f - std::erf((wavelength - params[2]) / (params[3] * kSqrt2)));
        return uv * ir;
    }

    bool hanatos_window_params_valid(const std::array<float, 4>& params) {
        return std::all_of(params.begin(), params.end(), [](float value) {
                   return std::isfinite(value);
               }) &&
               params[1] > 0.0f && params[3] > 0.0f;
    }

    std::uint64_t hash_hanatos_lut_recipe(const FilmRawRecipe& recipe) {
        std::uint64_t hash = Hash::kFnvOffset;
        constexpr std::uint32_t kSchemaVersion = 1u;
        hash_value(hash, kSchemaVersion);
        hash_value(hash, recipe.finalSensitivityHash);
        if (recipe.hanatos.spectralGaussianBlur > 0.0f) {
            hash_value(hash, recipe.hanatos.spectralGaussianBlur);
        }
        hash_value(hash, recipe.hanatos.applySurface);
        if (recipe.hanatos.applySurface) {
            Hash::hash_bytes_update(hash, recipe.hanatos.surfaceParams.data(), sizeof(recipe.hanatos.surfaceParams));
            hash_string(hash, recipe.hanatos.referenceIlluminant);
        }
        return hash;
    }

    bool derive_final_sensitivity(
        FilmRawRecipe& recipe,
        const std::array<float, 81>& referenceIlluminant) {
        std::array<double, 3> unfilteredResponse{};
        std::array<double, 3> filteredResponse{};
        std::array<float, 81> bandPass{};
        for (std::size_t wavelengthIndex = 0; wavelengthIndex < bandPass.size(); ++wavelengthIndex) {
            const float wavelength = 380.0f + 5.0f * static_cast<float>(wavelengthIndex);
            const float filter = recipe.cameraBandPass.active
                                     ? camera_filter_sample(wavelength, recipe.cameraBandPass.uv, true) *
                                           camera_filter_sample(wavelength, recipe.cameraBandPass.ir, false)
                                     : 1.0f;
            bandPass[wavelengthIndex] = std::isfinite(filter) ? std::max(0.0f, filter) : 0.0f;
            const float illuminant = referenceIlluminant[wavelengthIndex];
            if (!std::isfinite(illuminant) || illuminant < 0.0f) {
                return false;
            }
            for (std::size_t channel = 0; channel < 3; ++channel) {
                const float sensitivity = recipe.linearSensitivity[wavelengthIndex][channel];
                const double finiteSensitivity =
                    std::isfinite(sensitivity) ? std::max(0.0, static_cast<double>(sensitivity)) : 0.0;
                unfilteredResponse[channel] += finiteSensitivity * static_cast<double>(illuminant);
                filteredResponse[channel] +=
                    finiteSensitivity * static_cast<double>(bandPass[wavelengthIndex]) *
                    static_cast<double>(illuminant);
            }
        }

        std::array<double, 3> normalization{};
        for (std::size_t channel = 0; channel < normalization.size(); ++channel) {
            if (!(std::isfinite(unfilteredResponse[channel]) && unfilteredResponse[channel] > 0.0) ||
                !(std::isfinite(filteredResponse[channel]) && filteredResponse[channel] > 0.0)) {
                return false;
            }
            normalization[channel] = filteredResponse[channel] / unfilteredResponse[channel];
        }

        for (std::size_t wavelengthIndex = 0; wavelengthIndex < recipe.finalSensitivity.size(); ++wavelengthIndex) {
            for (std::size_t channel = 0; channel < 3; ++channel) {
                const float source = recipe.linearSensitivity[wavelengthIndex][channel];
                const double finiteSource =
                    std::isfinite(source) ? std::max(0.0, static_cast<double>(source)) : 0.0;
                const double derived =
                    finiteSource * static_cast<double>(bandPass[wavelengthIndex]) / normalization[channel];
                recipe.finalSensitivity[wavelengthIndex][channel] =
                    std::isfinite(derived) ? static_cast<float>(std::max(0.0, derived)) : 0.0f;
            }
        }

        if (recipe.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Hanatos2025 &&
            recipe.hanatos.applyWindow) {
            if (!hanatos_window_params_valid(recipe.hanatos.windowParams)) {
                return false;
            }

            std::array<double, 3> response{};
            std::array<double, 3> windowedResponse{};
            std::array<float, 81> window{};
            for (std::size_t wavelengthIndex = 0; wavelengthIndex < window.size(); ++wavelengthIndex) {
                const float wavelength = 380.0f + 5.0f * static_cast<float>(wavelengthIndex);
                const float sample = hanatos_window_sample(wavelength, recipe.hanatos.windowParams);
                if (!std::isfinite(sample) || sample < 0.0f) {
                    return false;
                }
                window[wavelengthIndex] = sample;
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    const double weighted =
                        static_cast<double>(recipe.finalSensitivity[wavelengthIndex][channel]) *
                        static_cast<double>(referenceIlluminant[wavelengthIndex]);
                    response[channel] += weighted;
                    windowedResponse[channel] += weighted * static_cast<double>(sample);
                }
            }

            std::array<double, 3> normalization{};
            for (std::size_t channel = 0; channel < normalization.size(); ++channel) {
                if (!(std::isfinite(response[channel]) && response[channel] > 0.0) ||
                    !(std::isfinite(windowedResponse[channel]) && windowedResponse[channel] > 0.0)) {
                    return false;
                }
                normalization[channel] = windowedResponse[channel] / response[channel];
            }

            for (std::size_t wavelengthIndex = 0; wavelengthIndex < recipe.finalSensitivity.size(); ++wavelengthIndex) {
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    const double adapted =
                        static_cast<double>(recipe.finalSensitivity[wavelengthIndex][channel]) *
                        static_cast<double>(window[wavelengthIndex]) /
                        normalization[channel];
                    if (!std::isfinite(adapted) || adapted < 0.0) {
                        return false;
                    }
                    recipe.finalSensitivity[wavelengthIndex][channel] = static_cast<float>(adapted);
                }
            }
        }

        recipe.finalSensitivityHash =
            Hash::hash_float_span(&recipe.finalSensitivity[0][0], recipe.finalSensitivity.size() * 3u);
        recipe.hanatosLutHash =
            recipe.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Hanatos2025
                ? hash_hanatos_lut_recipe(recipe)
                : 0;

        double greenMidgray = 0.0;
        for (std::size_t wavelengthIndex = 0; wavelengthIndex < recipe.finalSensitivity.size(); ++wavelengthIndex) {
            greenMidgray +=
                0.184 * static_cast<double>(referenceIlluminant[wavelengthIndex]) *
                static_cast<double>(recipe.finalSensitivity[wavelengthIndex][1]);
        }
        recipe.mallettGreenMidgrayScale =
            (std::isfinite(greenMidgray) && greenMidgray > 0.0)
                ? static_cast<float>(1.0 / greenMidgray)
                : 1.0f;
        return recipe.finalSensitivityHash != 0;
    }

    std::uint64_t hash_film_develop_recipe(const FilmDevelopRecipe& recipe) {
        std::uint64_t hash = Hash::kFnvOffset;
        hash_value(hash, recipe.polarity);
        hash_value(hash, recipe.authoredDensityCurvesHash);
        hash_value(hash, recipe.normalizedDensityCurvesHash);
        Hash::hash_bytes_update(hash, recipe.densityCurveGamma.data(), sizeof(recipe.densityCurveGamma));
        return hash;
    }

    bool finite_nonnegative(float value) {
        return std::isfinite(value) && value >= 0.0f;
    }

    float interp_clamped(
        float x,
        const std::vector<float>& xp,
        const std::vector<std::array<float, 3>>& values,
        std::size_t channel) {
        if (x <= xp.front()) {
            return values.front()[channel];
        }
        if (x >= xp.back()) {
            return values.back()[channel];
        }
        const auto upper = std::upper_bound(xp.begin(), xp.end(), x);
        const std::size_t hi = static_cast<std::size_t>(upper - xp.begin());
        const std::size_t lo = hi - 1u;
        const float span = xp[hi] - xp[lo];
        if (!(std::isfinite(span) && span > 0.0f)) {
            return values[lo][channel];
        }
        const float t = (x - xp[lo]) / span;
        return values[lo][channel] + t * (values[hi][channel] - values[lo][channel]);
    }

    std::uint64_t hash_dir_couplers_recipe(const DirCouplersRecipe& recipe) {
        if (!recipe.active) {
            return 0;
        }
        std::uint64_t hash = Hash::kFnvOffset;
        hash_value(hash, recipe.polarity);
        hash_value(hash, recipe.amount);
        hash_value(hash, recipe.inhibitionSameLayer);
        hash_value(hash, recipe.inhibitionInterlayer);
        Hash::hash_bytes_update(hash, recipe.gammaSameLayerRgb.data(), sizeof(recipe.gammaSameLayerRgb));
        Hash::hash_bytes_update(hash, recipe.gammaInterlayerRToGb.data(), sizeof(recipe.gammaInterlayerRToGb));
        Hash::hash_bytes_update(hash, recipe.gammaInterlayerGToRb.data(), sizeof(recipe.gammaInterlayerGToRb));
        Hash::hash_bytes_update(hash, recipe.gammaInterlayerBToRg.data(), sizeof(recipe.gammaInterlayerBToRg));
        hash_value(hash, recipe.precorrectedDensityCurvesHash);
        if (recipe.diffusionSizeUm > 0.0f) {
            hash_value(hash, recipe.diffusionSizeUm);
            hash_value(hash, recipe.diffusionTailWeight);
            if (recipe.diffusionTailWeight > 0.0f) {
                hash_value(hash, recipe.diffusionTailUm);
            }
        }
        return hash;
    }

    bool build_dir_couplers_recipe(
        const Profiles::ValidatedFilmProfile& profile,
        const FilmDevelopRecipe& develop,
        const DirCouplersControls& controls,
        DirCouplersRecipe& out) {
        out = DirCouplersRecipe{};
        if (!finite_nonnegative(controls.amount) ||
            !finite_nonnegative(controls.inhibitionSameLayer) ||
            !finite_nonnegative(controls.inhibitionInterlayer) ||
            !finite_nonnegative(controls.diffusionSizeUm) ||
            !finite_nonnegative(controls.diffusionTailUm) ||
            !std::isfinite(controls.diffusionTailWeight) ||
            controls.diffusionTailWeight < 0.0f || controls.diffusionTailWeight > 1.0f) {
            return false;
        }

        out.polarity = profile.info.type;
        out.active = controls.active && controls.amount > 0.0f;
        if (!out.active) {
            return true;
        }
        out.amount = controls.amount;
        out.inhibitionSameLayer = controls.inhibitionSameLayer;
        out.inhibitionInterlayer = controls.inhibitionInterlayer;
        out.gammaSameLayerRgb = profile.digest.gammaSamelayerRgb;
        out.gammaInterlayerRToGb = profile.digest.gammaInterlayerRToGb;
        out.gammaInterlayerGToRb = profile.digest.gammaInterlayerGToRb;
        out.gammaInterlayerBToRg = profile.digest.gammaInterlayerBToRg;
        out.diffusionSizeUm = controls.diffusionSizeUm;
        out.diffusionTailUm = controls.diffusionTailUm;
        out.diffusionTailWeight = controls.diffusionTailWeight;

        out.matrixRgb[0][0] = out.gammaSameLayerRgb[0] * out.inhibitionSameLayer;
        out.matrixRgb[1][1] = out.gammaSameLayerRgb[1] * out.inhibitionSameLayer;
        out.matrixRgb[2][2] = out.gammaSameLayerRgb[2] * out.inhibitionSameLayer;
        out.matrixRgb[0][1] = out.gammaInterlayerRToGb[0] * out.inhibitionInterlayer;
        out.matrixRgb[0][2] = out.gammaInterlayerRToGb[1] * out.inhibitionInterlayer;
        out.matrixRgb[1][0] = out.gammaInterlayerGToRb[0] * out.inhibitionInterlayer;
        out.matrixRgb[1][2] = out.gammaInterlayerGToRb[1] * out.inhibitionInterlayer;
        out.matrixRgb[2][0] = out.gammaInterlayerBToRg[0] * out.inhibitionInterlayer;
        out.matrixRgb[2][1] = out.gammaInterlayerBToRg[1] * out.inhibitionInterlayer;
        for (auto& row : out.matrixRgb) {
            for (float& value : row) {
                value *= out.amount;
            }
        }

        if (develop.logExposure.empty() ||
            develop.logExposure.size() != develop.normalizedDensityCurves.size()) {
            return false;
        }
        out.densityMaxRgb.fill(-std::numeric_limits<float>::infinity());
        for (const auto& row : develop.normalizedDensityCurves) {
            for (std::size_t channel = 0; channel < 3; ++channel) {
                if (std::isfinite(row[channel])) {
                    out.densityMaxRgb[channel] = std::max(out.densityMaxRgb[channel], row[channel]);
                }
            }
        }
        if (!std::all_of(out.densityMaxRgb.begin(), out.densityMaxRgb.end(), finite_nonnegative)) {
            return false;
        }

        std::array<std::vector<float>, 3> shiftedExposure;
        for (auto& axis : shiftedExposure) {
            axis.resize(develop.logExposure.size());
        }
        const bool positive = out.polarity == Spektrafilm::ProfilePolarity::Positive;
        for (std::size_t sample = 0; sample < develop.logExposure.size(); ++sample) {
            if (!std::isfinite(develop.logExposure[sample])) {
                return false;
            }
            for (std::size_t receiver = 0; receiver < 3; ++receiver) {
                double correction = 0.0;
                for (std::size_t donor = 0; donor < 3; ++donor) {
                    const float density = develop.normalizedDensityCurves[sample][donor];
                    if (!std::isfinite(density)) {
                        return false;
                    }
                    const float silver = positive ? out.densityMaxRgb[donor] - density : density;
                    correction += static_cast<double>(silver) *
                                  static_cast<double>(out.matrixRgb[donor][receiver]);
                }
                shiftedExposure[receiver][sample] =
                    develop.logExposure[sample] - static_cast<float>(correction);
                if (sample > 0u &&
                    !(shiftedExposure[receiver][sample] > shiftedExposure[receiver][sample - 1u])) {
                    return false;
                }
            }
        }

        out.precorrectedDensityCurves.resize(develop.normalizedDensityCurves.size());
        for (std::size_t sample = 0; sample < develop.logExposure.size(); ++sample) {
            for (std::size_t channel = 0; channel < 3; ++channel) {
                out.precorrectedDensityCurves[sample][channel] = interp_clamped(
                    develop.logExposure[sample],
                    shiftedExposure[channel],
                    develop.normalizedDensityCurves,
                    channel);
            }
        }
        out.precorrectedDensityCurvesHash = hash_nan_preserving_floats(
            &out.precorrectedDensityCurves[0][0],
            out.precorrectedDensityCurves.size() * 3u);
        out.hash = hash_dir_couplers_recipe(out);
        return out.precorrectedDensityCurvesHash != 0 && out.hash != 0;
    }

    std::uint64_t hash_density_bounds_recipe(const DensityBoundsRecipe& recipe) {
        std::uint64_t hash = Hash::kFnvOffset;
        hash_value(hash, recipe.route);
        hash_value(hash, recipe.medium);
        hash_value(hash, recipe.polarity);
        hash_value(hash, recipe.source);
        Hash::hash_bytes_update(hash, recipe.dataMinCmy.data(), sizeof(recipe.dataMinCmy));
        Hash::hash_bytes_update(hash, recipe.dataMaxCmy.data(), sizeof(recipe.dataMaxCmy));
        Hash::hash_bytes_update(hash, recipe.invSpanCmy.data(), sizeof(recipe.invSpanCmy));
        Hash::hash_bytes_update(hash, recipe.authoredMinCmy.data(), sizeof(recipe.authoredMinCmy));
        Hash::hash_bytes_update(hash, recipe.authoredMaxCmy.data(), sizeof(recipe.authoredMaxCmy));
        return hash;
    }

    std::uint64_t hash_scanner_output_recipe(const ScannerOutputRecipe& recipe) {
        std::uint64_t hash = Hash::kFnvOffset;
        hash_value(hash, recipe.route);
        hash_value(hash, recipe.medium);
        hash_value(hash, recipe.polarity);
        hash_string(hash, recipe.viewingIlluminant);
        hash_value(hash, recipe.lutResolution);
        hash_value(hash, recipe.outputColorSpace);
        hash_value(hash, recipe.outputCctfEncoding);
        hash_value(hash, recipe.outputLinearPassThrough);
        hash_value(hash, recipe.directGlareDisabled);
        hash_value(hash, recipe.lensBlurSigmaPx);
        hash_value(hash, recipe.unsharpSigmaPx);
        hash_value(hash, recipe.unsharpAmount);
        hash_value(hash, recipe.postEffectsDisposition);
        return hash;
    }

    bool build_direct_density_bounds(
        const Profiles::ValidatedFilmProfile& profile,
        const FilmDevelopRecipe& develop,
        const GrainContract& grain,
        Spektrafilm::ScanRoute route,
        DensityBoundsRecipe& out) {
        out = DensityBoundsRecipe{};
        out.route = route;
        out.medium = Spektrafilm::DensityMedium::Film;
        out.polarity = profile.info.type;
        out.source = Spektrafilm::DensityBoundsSource::DirectFilmGrainContractAndAuthoredCurves;
        out.authoredMinCmy = develop.authoredMinCmy;
        out.authoredMaxCmy = develop.authoredMaxCmy;
        for (std::size_t channel = 0; channel < out.dataMinCmy.size(); ++channel) {
            out.dataMinCmy[channel] = -grain.densityMinCmy[channel];
            out.dataMaxCmy[channel] = develop.authoredMaxCmy[channel];
            const float span = out.dataMaxCmy[channel] - out.dataMinCmy[channel];
            if (!(std::isfinite(span) && span > 0.0f)) {
                return false;
            }
            out.invSpanCmy[channel] = 1.0f / span;
        }
        out.hash = hash_density_bounds_recipe(out);
        return out.hash != 0;
    }

    bool finite_cmy(const CmyCcTriplet& values) {
        return std::isfinite(values.c) && std::isfinite(values.m) && std::isfinite(values.y);
    }

    void hash_cmy(std::uint64_t& hash, const CmyCcTriplet& values) {
        hash_value(hash, values.c);
        hash_value(hash, values.m);
        hash_value(hash, values.y);
    }

    Spektrafilm::PrintNormalizationMode print_normalization_mode(bool normalize, bool compensate) {
        if (normalize) {
            return compensate
                       ? Spektrafilm::PrintNormalizationMode::NormalizeAndCompensate
                       : Spektrafilm::PrintNormalizationMode::NormalizeOnly;
        }
        return compensate
                   ? Spektrafilm::PrintNormalizationMode::CompensationOnly
                   : Spektrafilm::PrintNormalizationMode::None;
    }

    Spektrafilm::PrintNormalizerExpression print_normalizer_expression(
        Spektrafilm::PrintNormalizationMode mode) {
        switch (mode) {
            case Spektrafilm::PrintNormalizationMode::None:
                return Spektrafilm::PrintNormalizerExpression::One;
            case Spektrafilm::PrintNormalizationMode::CompensationOnly:
                return Spektrafilm::PrintNormalizerExpression::FactorMidgrayCompOverFactorMidgray;
            case Spektrafilm::PrintNormalizationMode::NormalizeOnly:
                return Spektrafilm::PrintNormalizerExpression::FactorMidgray;
            case Spektrafilm::PrintNormalizationMode::NormalizeAndCompensate:
                return Spektrafilm::PrintNormalizerExpression::FactorMidgrayComp;
            default:
                return Spektrafilm::PrintNormalizerExpression::One;
        }
    }

    std::uint64_t hash_dichroic_resource_identity(const DichroicResourceIdentity& identity) {
        std::uint64_t hash = Hash::kFnvOffset;
        hash_value(hash, identity.set);
        hash_value(hash, identity.kind);
        hash_string(hash, identity.setKey);
        hash_value(hash, identity.percentTransmittanceDividedBy100);
        hash_value(hash, identity.duplicateWavelengthsKeepFirst);
        hash_value(hash, identity.akimaResampledToReferenceAxis);
        if (identity.kind == Spektrafilm::DichroicResourceKind::CustomAnalyticModel) {
            Hash::hash_bytes_update(hash, identity.customEdgesNm.data(), sizeof(identity.customEdgesNm));
            Hash::hash_bytes_update(hash, identity.customTransitionsNm.data(), sizeof(identity.customTransitionsNm));
        } else {
            for (std::size_t channel = 0; channel < identity.resourcePathsCmy.size(); ++channel) {
                hash_string(hash, identity.resourcePathsCmy[channel]);
                hash_value(hash, identity.resourceHashesCmy[channel]);
            }
        }
        return hash;
    }

    bool dichroic_resource_identity_valid(const DichroicResourceIdentity& identity) {
        if (identity.kind == Spektrafilm::DichroicResourceKind::CustomAnalyticModel) {
            return identity.set == Spektrafilm::DichroicFilterSet::Custom &&
                   identity.setKey == "custom" &&
                   std::all_of(identity.customEdgesNm.begin(), identity.customEdgesNm.end(), [](float value) {
                       return std::isfinite(value);
                   }) &&
                   std::all_of(identity.customTransitionsNm.begin(), identity.customTransitionsNm.end(), [](float value) {
                       return std::isfinite(value) && value > 0.0f;
                   });
        }

        if (identity.set == Spektrafilm::DichroicFilterSet::Custom ||
            identity.setKey.empty() ||
            !identity.percentTransmittanceDividedBy100 ||
            !identity.duplicateWavelengthsKeepFirst ||
            !identity.akimaResampledToReferenceAxis) {
            return false;
        }
        for (std::size_t channel = 0; channel < identity.resourcePathsCmy.size(); ++channel) {
            if (identity.resourcePathsCmy[channel].empty() || identity.resourceHashesCmy[channel] == 0) {
                return false;
            }
        }
        return true;
    }

    std::uint64_t hash_neutral_calibration(const NeutralCalibrationRecipe& calibration) {
        std::uint64_t hash = Hash::kFnvOffset;
        hash_value(hash, calibration.status);
        hash_string(hash, calibration.resourcePath);
        hash_string(hash, calibration.printProfileKey);
        hash_string(hash, calibration.printIlluminantKey);
        hash_string(hash, calibration.filmProfileKey);
        hash_value(hash, calibration.resourceHash);
        return hash;
    }

    std::uint64_t hash_print_filter_recipe(const PrintFilterRecipe& recipe) {
        std::uint64_t hash = Hash::kFnvOffset;
        hash_cmy(hash, recipe.neutralCmyCc);
        hash_cmy(hash, recipe.userCmyCc);
        hash_value(hash, recipe.filmJuicerMainCFilterShiftCc);
        hash_cmy(hash, recipe.mainCmyCc);
        hash_cmy(hash, recipe.preflashUserCmyCc);
        hash_cmy(hash, recipe.preflashCmyCc);
        hash_value(hash, recipe.dichroic.hash);
        hash_value(hash, recipe.neutralCalibration.hash);
        return hash;
    }

    std::uint64_t hash_print_exposure_recipe(const PrintExposureRecipe& recipe) {
        std::uint64_t hash = Hash::kFnvOffset;
        hash_value(hash, recipe.printExposure);
        hash_value(hash, recipe.preflashExposure);
        hash_value(hash, recipe.cameraExposureCompensationEv);
        hash_value(hash, recipe.normalizePrintExposure);
        hash_value(hash, recipe.printExposureCompensation);
        hash_value(hash, recipe.normalizationMode);
        hash_value(hash, recipe.normalizerExpression);
        hash_value(hash, recipe.scalingOrder);
        return hash;
    }

} // namespace

namespace Spektrafilm {

    DirectRecipeBuildResult build_direct_render_recipe(const DirectRecipeBuildInput& input) {
        DirectRecipeBuildResult result{};
        result.recipe = make_render_recipe(input.filmProfileKey, input.printProfileKey, input.scanRoute);
        if (scan_route_is_print(input.scanRoute)) {
            result.diagnostic = "UnsupportedMode phase=3A field=scan_route expected=direct";
            return result;
        }
        if (!input.filmProfile) {
            result.diagnostic = "MissingRequiredResource phase=3B field=film_profile";
            return result;
        }
        if (!input.directRoutePrintProfileExcluded || !input.directRouteNeutralCalibrationExcluded) {
            result.diagnostic = "ResourceDescriptorMismatch phase=3A direct route print/neutral exclusion";
            return result;
        }
        if (!input.referenceIlluminantValid) {
            result.diagnostic = "MissingRequiredResource phase=3B field=reference_illuminant";
            return result;
        }
        if (!auto_exposure_method_index_valid(input.cameraMeteringMethod)) {
            result.diagnostic = "UnsupportedMode phase=3B field=auto_exposure_method";
            return result;
        }

        const Profiles::ValidatedFilmProfile& profile = *input.filmProfile;
        const ScanRoute resolvedRoute = resolve_scan_route(profile.info.type, input.scanRoute);
        if (resolvedRoute != input.scanRoute ||
            profile.info.support != ProfileSupport::Film ||
            profile.info.stage != ProfileStage::Filming) {
            result.diagnostic = "UnsupportedMode phase=3A selected capture profile route mismatch";
            return result;
        }

        ProfileRoute& profileRoute = result.recipe.profileRoute;
        profileRoute.captureSupport = profile.info.support;
        profileRoute.captureStage = profile.info.stage;
        profileRoute.capturePolarity = profile.info.type;
        profileRoute.captureUse = profile.info.use;
        profileRoute.captureAntihalation = profile.info.antihalation;
        profileRoute.captureChannelModel = profile.info.channelModel;
        profileRoute.filmProfileAssetVersionToken = profile.assetVersionToken;
        profileRoute.printProfileAssetVersionToken = 0;
        profileRoute.filmProfile = input.filmProfile;
        profileRoute.printProfile.reset();
        profileRoute.directRoutePrintProfileExcluded = input.directRoutePrintProfileExcluded;
        profileRoute.directRouteNeutralCalibrationExcluded = input.directRouteNeutralCalibrationExcluded;
        profileRoute.hash = hash_profile_route(profileRoute);

        FilmRawRecipe& filmRaw = result.recipe.filmRaw;
        filmRaw.inputColorSpace = input.inputColorSpace;
        filmRaw.inputCctfDecoding = input.inputCctfDecoding;
        filmRaw.rgbToRawMethod = (input.spectralUpsamplingMode == 1)
                                     ? RgbToRawMethod::Mallett2019
                                     : RgbToRawMethod::Hanatos2025;
        filmRaw.autoExposureEnabled = input.cameraAutoExposureEnabled;
        filmRaw.autoExposureMethod = auto_exposure_method_from_index(input.cameraMeteringMethod);
        filmRaw.manualExposureCompensationEv = input.manualExposureCompensationEv;
        filmRaw.filmFormatLongEdgeMm = input.filmFormatLongEdgeMm;
        filmRaw.cameraBandPass.active = input.cameraFilterOverride;
        filmRaw.cameraBandPass.uv = copy_filter_triplet(input.cameraFilterUV, input.cameraFilterOverride);
        filmRaw.cameraBandPass.ir = copy_filter_triplet(input.cameraFilterIR, input.cameraFilterOverride);
        filmRaw.hanatos.applyWindow = profile.digest.hanatosRuntimeApplyWindowDefault;
        filmRaw.hanatos.applySurface = profile.digest.hanatosRuntimeApplySurfaceDefault;
        filmRaw.hanatos.spectralGaussianBlur = profile.digest.hanatosSpectralGaussianBlurDefault;
        filmRaw.hanatos.windowParams = profile.data.hanatos2025AdaptationWindowParams;
        filmRaw.hanatos.surfaceParams = profile.data.hanatos2025AdaptationSurfaceParams;
        filmRaw.hanatos.referenceIlluminant = profile.info.referenceIlluminant.value;
        filmRaw.linearSensitivity = profile.data.linearSensitivity;
        filmRaw.linearSensitivityHash =
            Hash::hash_float_span(&filmRaw.linearSensitivity[0][0], filmRaw.linearSensitivity.size() * 3u);
        if (filmRaw.rgbToRawMethod == RgbToRawMethod::Hanatos2025) {
            if (filmRaw.hanatos.applyWindow && !profile.data.hasHanatos2025AdaptationWindowParams) {
                result.diagnostic =
                    "MalformedRequiredProfileData phase=3D-2 field=data.hanatos2025_adaptation_window_params";
                return result;
            }
            if (filmRaw.hanatos.applySurface && !profile.data.hasHanatos2025AdaptationSurfaceParams) {
                result.diagnostic =
                    "MalformedRequiredProfileData phase=3D-2 field=data.hanatos2025_adaptation_surface_params";
                return result;
            }
            if (!std::isfinite(filmRaw.hanatos.spectralGaussianBlur) ||
                filmRaw.hanatos.spectralGaussianBlur < 0.0f) {
                result.diagnostic =
                    "MalformedRequiredProfileData phase=3D-2 field=settings.spectral_gaussian_blur";
                return result;
            }
        }
        if (!derive_final_sensitivity(filmRaw, input.referenceIlluminant)) {
            result.diagnostic = "MalformedRequiredProfileData phase=3B field=final_sensitivity";
            return result;
        }
        filmRaw.hash = hash_film_raw_recipe(filmRaw);
        if (filmRaw.linearSensitivityHash == 0 ||
            filmRaw.finalSensitivityHash == 0 ||
            (filmRaw.rgbToRawMethod == RgbToRawMethod::Hanatos2025 && filmRaw.hanatosLutHash == 0) ||
            filmRaw.hash == 0) {
            result.diagnostic = "MalformedRequiredProfileData phase=3B field=data.log_sensitivity";
            return result;
        }

        FilmDevelopRecipe& filmDevelop = result.recipe.filmDevelop;
        filmDevelop.polarity = profile.info.type;
        filmDevelop.logExposure = profile.data.logExposure;
        filmDevelop.authoredDensityCurves = profile.data.densityCurves;
        if (!normalize_density_curves(
                filmDevelop.authoredDensityCurves,
                filmDevelop.normalizedDensityCurves,
                filmDevelop.authoredMinCmy,
                filmDevelop.authoredMaxCmy)) {
            result.diagnostic = "MalformedRequiredProfileData phase=3A field=data.density_curves";
            return result;
        }
        if (filmDevelop.authoredDensityCurves.empty() ||
            filmDevelop.normalizedDensityCurves.empty()) {
            result.diagnostic = "MalformedRequiredProfileData phase=3A field=data.density_curves empty";
            return result;
        }
        filmDevelop.authoredDensityCurvesHash = hash_nan_preserving_floats(
            &filmDevelop.authoredDensityCurves[0][0],
            filmDevelop.authoredDensityCurves.size() * 3u);
        filmDevelop.normalizedDensityCurvesHash = hash_nan_preserving_floats(
            &filmDevelop.normalizedDensityCurves[0][0],
            filmDevelop.normalizedDensityCurves.size() * 3u);
        filmDevelop.hash = hash_film_develop_recipe(filmDevelop);
        if (filmDevelop.authoredDensityCurvesHash == 0 ||
            filmDevelop.normalizedDensityCurvesHash == 0 ||
            filmDevelop.hash == 0) {
            result.diagnostic = "MalformedRequiredProfileData phase=3A field=data.density_curves hash";
            return result;
        }

        if (!build_dir_couplers_recipe(
                profile,
                filmDevelop,
                input.dirCouplers,
                result.recipe.dirCouplers)) {
            result.diagnostic = "MalformedRequiredProfileData phase=3D-3 field=dir_couplers";
            return result;
        }

        if (!build_direct_density_bounds(
                profile,
                filmDevelop,
                input.grainContract,
                input.scanRoute,
                result.recipe.densityBounds)) {
            result.diagnostic = "MalformedRequiredProfileData phase=3A field=density_bounds";
            return result;
        }

        ScannerOutputRecipe& scanner = result.recipe.scannerOutput;
        scanner.route = input.scanRoute;
        scanner.medium = DensityMedium::Film;
        scanner.polarity = profile.info.type;
        scanner.viewingIlluminant = profile.info.viewingIlluminant.value;
        scanner.lutResolution = std::clamp(input.scannerLutResolution, 17u, 128u);
        scanner.outputColorSpace = input.outputColorSpace;
        scanner.outputCctfEncoding = input.outputCctfEncoding;
        scanner.outputLinearPassThrough = input.outputLinearPassThrough;
        scanner.directGlareDisabled = true;
        scanner.lensBlurSigmaPx = input.scannerLensBlurSigmaPx;
        scanner.unsharpSigmaPx = input.scannerUnsharpSigmaPx;
        scanner.unsharpAmount = input.scannerUnsharpAmount;
        const bool scannerPostEffectsIdentity =
            scanner.lensBlurSigmaPx <= 0.0f &&
            (scanner.unsharpSigmaPx <= 0.0f || scanner.unsharpAmount <= 0.0f);
        scanner.postEffectsDisposition = scannerPostEffectsIdentity
                                             ? ScannerPostEffectDisposition::Identity
                                             : ScannerPostEffectDisposition::BlockedNotImplementedForPhase3;
        scanner.blockingDiagnostic = scannerPostEffectsIdentity
                                         ? std::string()
                                         : kScannerPostEffectsNotImplementedForPhase3;
        scanner.hash = hash_scanner_output_recipe(scanner);

        result.recipe.directStructuralReady = true;
        result.recipe.directPixelAcceptance = false;
        if (result.recipe.dirCouplers.active) {
            result.recipe.hash = Hash::hash_uint64_values({profileRoute.hash,
                                                           filmRaw.hash,
                                                           filmDevelop.hash,
                                                           result.recipe.dirCouplers.hash,
                                                           result.recipe.densityBounds.hash,
                                                           scanner.hash});
        } else {
            result.recipe.hash = Hash::hash_uint64_values({profileRoute.hash,
                                                           filmRaw.hash,
                                                           filmDevelop.hash,
                                                           result.recipe.densityBounds.hash,
                                                           scanner.hash});
        }
        result.valid = result.recipe.hash != 0;
        if (!result.valid) {
            result.diagnostic = "ResourceDescriptorMismatch phase=3A field=render_recipe_hash";
        }
        return result;
    }

    PrintRecipeBuildResult build_print_render_recipe(const PrintRecipeBuildInput& input) {
        PrintRecipeBuildResult result{};
        result.recipe = make_render_recipe(input.filmProfileKey, input.printProfileKey, input.scanRoute);
        if (!scan_route_is_print(input.scanRoute)) {
            result.diagnostic = "UnsupportedMode phase=4A field=scan_route expected=print";
            return result;
        }
        if (!input.filmProfile || !input.printProfile) {
            result.diagnostic = "MissingRequiredResource phase=4A field=selected_profile";
            return result;
        }
        if (resolve_scan_route(input.filmProfile->info.type, input.scanRoute) != input.scanRoute ||
            input.filmProfile->info.stage != ProfileStage::Filming ||
            input.printProfile->digest.profileRole != Profiles::ProfileRole::Print ||
            input.printProfile->info.stage != ProfileStage::Printing) {
            result.diagnostic = "UnsupportedMode phase=4A selected profile route mismatch";
            return result;
        }
        if (input.neutralCalibrationHash == 0 ||
            input.printIlluminantKey.empty() ||
            !finite_cmy(input.currentNeutralCmyCc) ||
            !finite_cmy(input.calibratedNeutralCmyCc) ||
            !std::all_of(input.uiYmcCc.begin(), input.uiYmcCc.end(), [](float value) {
                return std::isfinite(value);
            }) ||
            !std::isfinite(input.preflashMFilterCc) || !std::isfinite(input.preflashYFilterCc) || !std::isfinite(input.printExposure) || !std::isfinite(input.preflashExposure) || !std::isfinite(input.cameraExposureCompensationEv)) {
            result.diagnostic = "ResourceDescriptorMismatch phase=4A field=print_recipe_input";
            return result;
        }

        ProfileRoute& route = result.recipe.profileRoute;
        route.captureSupport = input.filmProfile->info.support;
        route.captureStage = input.filmProfile->info.stage;
        route.capturePolarity = input.filmProfile->info.type;
        route.captureUse = input.filmProfile->info.use;
        route.captureAntihalation = input.filmProfile->info.antihalation;
        route.captureChannelModel = input.filmProfile->info.channelModel;
        route.filmProfileAssetVersionToken = input.filmProfile->assetVersionToken;
        route.printProfileAssetVersionToken = input.printProfile->assetVersionToken;
        route.filmProfile = input.filmProfile;
        route.printProfile = input.printProfile;
        route.hash = hash_profile_route(route);

        PrintRecipe& print = result.recipe.print;
        print.filters.dichroic = input.dichroic;
        if (!dichroic_resource_identity_valid(print.filters.dichroic)) {
            result.diagnostic = "ResourceDescriptorMismatch phase=4A field=dichroic_resource_identity";
            return result;
        }
        print.filters.dichroic.hash =
            hash_dichroic_resource_identity(print.filters.dichroic);
        if (print.filters.dichroic.hash == 0) {
            result.diagnostic = "ResourceDescriptorMismatch phase=4A field=dichroic_resource_identity";
            return result;
        }
        print.filters.neutralCalibration.status = input.neutralCalibrationStatus;
        print.filters.neutralCalibration.printProfileKey = input.printProfileKey;
        print.filters.neutralCalibration.printIlluminantKey = input.printIlluminantKey;
        print.filters.neutralCalibration.filmProfileKey = input.filmProfileKey;
        print.filters.neutralCalibration.resourceHash = input.neutralCalibrationResourceHash;
        print.filters.neutralCalibration.hash =
            Hash::hash_uint64_values({input.neutralCalibrationHash,
                                      hash_neutral_calibration(print.filters.neutralCalibration)});
        print.filters.neutralCmyCc =
            input.neutralCalibrationStatus == NeutralCalibrationStatus::Calibrated
                ? input.calibratedNeutralCmyCc
                : input.currentNeutralCmyCc;
        print.filters.userCmyCc =
            CmyCcTriplet{input.uiYmcCc[2], input.uiYmcCc[1], input.uiYmcCc[0]};
        print.filters.filmJuicerMainCFilterShiftCc = print.filters.userCmyCc.c;
        print.filters.mainCmyCc = CmyCcTriplet{
            print.filters.neutralCmyCc.c + print.filters.filmJuicerMainCFilterShiftCc,
            print.filters.neutralCmyCc.m + print.filters.userCmyCc.m,
            print.filters.neutralCmyCc.y + print.filters.userCmyCc.y};
        print.filters.preflashUserCmyCc =
            CmyCcTriplet{0.0f, input.preflashMFilterCc, input.preflashYFilterCc};
        print.filters.preflashCmyCc = CmyCcTriplet{
            print.filters.neutralCmyCc.c,
            print.filters.neutralCmyCc.m + input.preflashMFilterCc,
            print.filters.neutralCmyCc.y + input.preflashYFilterCc};
        print.filters.hash = hash_print_filter_recipe(print.filters);

        print.exposure.printExposure = input.printExposure;
        print.exposure.preflashExposure = input.preflashExposure;
        print.exposure.cameraExposureCompensationEv = input.cameraExposureCompensationEv;
        print.exposure.normalizePrintExposure = input.normalizePrintExposure;
        print.exposure.printExposureCompensation = input.printExposureCompensation;
        print.exposure.normalizationMode =
            print_normalization_mode(input.normalizePrintExposure, input.printExposureCompensation);
        print.exposure.normalizerExpression =
            print_normalizer_expression(print.exposure.normalizationMode);
        print.exposure.hash = hash_print_exposure_recipe(print.exposure);

        print.illuminant.key = input.printIlluminantKey;
        print.illuminant.hash = Hash::kFnvOffset;
        hash_string(print.illuminant.hash, print.illuminant.key);

        print.mediumHandoff.printProfileKey = input.printProfileKey;
        print.mediumHandoff.printProfileAssetVersionToken = input.printProfile->assetVersionToken;
        print.mediumHandoff.viewingIlluminant = input.printProfile->info.viewingIlluminant.value;
        {
            std::uint64_t hash = Hash::kFnvOffset;
            hash_value(hash, print.mediumHandoff.medium);
            hash_string(hash, print.mediumHandoff.printProfileKey);
            hash_value(hash, print.mediumHandoff.printProfileAssetVersionToken);
            hash_string(hash, print.mediumHandoff.viewingIlluminant);
            print.mediumHandoff.hash = hash;
        }

        print.hash = Hash::hash_uint64_values({print.filters.hash,
                                               print.exposure.hash,
                                               print.illuminant.hash,
                                               print.mediumHandoff.hash});
        result.recipe.printStructuralReady = true;
        result.recipe.hash = Hash::hash_uint64_values({route.hash, print.hash});
        result.valid = route.hash != 0 && print.filters.hash != 0 &&
                       print.exposure.hash != 0 && print.illuminant.hash != 0 &&
                       print.mediumHandoff.hash != 0 && print.hash != 0 &&
                       result.recipe.hash != 0;
        if (!result.valid) {
            result.diagnostic = "ResourceDescriptorMismatch phase=4A field=print_recipe_hash";
        }
        return result;
    }

    float print_exposure_normalizer(
        PrintNormalizationMode mode,
        float factorMidgray,
        float factorMidgrayComp) {
        switch (mode) {
            case PrintNormalizationMode::None:
                return 1.0f;
            case PrintNormalizationMode::CompensationOnly:
                return factorMidgrayComp / factorMidgray;
            case PrintNormalizationMode::NormalizeOnly:
                return factorMidgray;
            case PrintNormalizationMode::NormalizeAndCompensate:
                return factorMidgrayComp;
            default:
                return std::numeric_limits<float>::quiet_NaN();
        }
    }

    bool build_spatial_dir_descriptor(
        const DirCouplersRecipe& recipe,
        float pixelSizeUm,
        SpatialDirDescriptor& out) {
        out = SpatialDirDescriptor{};
        if (!recipe.active || !(recipe.diffusionSizeUm > 0.0f)) {
            return true;
        }
        if (!(std::isfinite(pixelSizeUm) && pixelSizeUm > 0.0f)) {
            return false;
        }
        out.dirRecipeHash = recipe.hash;
        out.gaussianSigmaPixels = recipe.diffusionSizeUm / pixelSizeUm;
        out.gaussianWeight = 1.0f - recipe.diffusionTailWeight;
        if (!(std::isfinite(out.gaussianSigmaPixels) && out.gaussianSigmaPixels > 0.0f) ||
            !finite_nonnegative(out.gaussianWeight)) {
            return false;
        }
        if (recipe.diffusionTailWeight > 0.0f) {
            const float tailSigmaPixels = recipe.diffusionTailUm / pixelSizeUm;
            if (!(std::isfinite(tailSigmaPixels) && tailSigmaPixels > 0.0f)) {
                return false;
            }
            for (std::size_t component = 0; component < out.exponentialSigmaPixels.size(); ++component) {
                out.exponentialSigmaPixels[component] =
                    tailSigmaPixels * SpatialDirDescriptor::kExponentialSigmaRatios[component];
                out.exponentialWeights[component] =
                    recipe.diffusionTailWeight * SpatialDirDescriptor::kExponentialAmplitudes[component];
            }
        }
        std::uint64_t hash = Hash::kFnvOffset;
        hash_value(hash, out.dirRecipeHash);
        hash_value(hash, out.gaussianSigmaPixels);
        hash_value(hash, out.gaussianWeight);
        Hash::hash_bytes_update(hash, out.exponentialSigmaPixels.data(), sizeof(out.exponentialSigmaPixels));
        Hash::hash_bytes_update(hash, out.exponentialWeights.data(), sizeof(out.exponentialWeights));
        out.hash = hash;
        return out.hash != 0;
    }

} // namespace Spektrafilm
