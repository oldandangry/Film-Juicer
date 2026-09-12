#include "RenderRecipe.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <numbers>
#include <string_view>

#include "Hash.h"

namespace {

    template <typename T>
    void hash_value(std::uint64_t& hash, const T& value) {
        Hash::hash_bytes_update(hash, &value, sizeof(value));
    }

    void hash_string(std::uint64_t& hash, const std::string& value) {
        Hash::hash_bytes_update(hash, value.data(), value.size());
    }

    double normal_cdf(double z) {
        return 0.5 * std::erfc(-z / std::numbers::sqrt2);
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

    void derive_grain_layer_axis_search_metadata(
        const Spektrafilm::FilmDevelopRecipe& filmDevelop,
        std::array<bool, 3>& outFinite,
        std::array<std::array<float, 16>, 3>& outBlockPrefixMax) {
        outFinite.fill(false);
        outBlockPrefixMax = {};
        if (filmDevelop.normalizedDensityCurves.size() != 256 ||
            (filmDevelop.polarity != Spektrafilm::ProfilePolarity::Negative &&
             filmDevelop.polarity != Spektrafilm::ProfilePolarity::Positive)) {
            return;
        }

        const bool positiveFilm =
            filmDevelop.polarity == Spektrafilm::ProfilePolarity::Positive;
        for (std::size_t channel = 0; channel < outFinite.size(); ++channel) {
            float prefixMaximum = -std::numeric_limits<float>::infinity();
            bool finite = true;
            for (std::size_t sample = 0;
                 sample < filmDevelop.normalizedDensityCurves.size();
                 ++sample) {
                float axisValue =
                    filmDevelop.normalizedDensityCurves[sample][channel];
                if (positiveFilm) {
                    axisValue = -axisValue;
                }
                if (!std::isfinite(axisValue)) {
                    finite = false;
                    break;
                }
                prefixMaximum = std::max(prefixMaximum, axisValue);
                if ((sample % 16) == 15) {
                    outBlockPrefixMax[channel][sample / 16] =
                        prefixMaximum;
                }
            }
            outFinite[channel] = finite;
        }
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
        hash_value(hash, route.capturePolarity);
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
        hash_value(hash, recipe.finalSensitivityHash);
        if (recipe.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Hanatos2025) {
            hash_value(hash, recipe.hanatosLutHash);
            hash_string(hash, recipe.hanatos.referenceIlluminant);
        } else {
            hash_value(hash, recipe.mallettGreenMidgrayScale);
        }
        return hash;
    }

    std::uint64_t hash_diffusion_filter_optics(const DiffusionFilterOpticsRecipe& recipe) {
        if (!recipe.resolved.active || recipe.resolved.hash == 0) {
            return 0;
        }
        std::uint64_t hash = Hash::kFnvOffset;
        hash_value(hash, recipe.domain);
        hash_value(hash, recipe.resolved.hash);
        return hash;
    }

    std::uint64_t hash_camera_lens_blur_optics(const CameraLensBlurOpticsRecipe& recipe) {
        if (!(recipe.sigmaUm > 0.0f)) {
            return 0;
        }
        std::uint64_t hash = Hash::kFnvOffset;
        hash_value(hash, recipe.sigmaUm);
        return hash;
    }

    bool build_spatial_optics_recipe(
        const Profiles::ValidatedFilmProfile& profile,
        const Spektrafilm::SpatialOpticsControls& controls,
        bool includePrintDomain,
        SpatialOptics& out,
        std::string& diagnostic) {
        out = SpatialOptics{};
        diagnostic.clear();

        const auto set_diffusion_diagnostic = [&](const char* component,
                                                  const std::string& resolvedDiagnostic) {
            constexpr const char kFieldMarker[] = "field=";
            const std::size_t fieldOffset = resolvedDiagnostic.find(kFieldMarker);
            const std::string field =
                fieldOffset == std::string::npos
                    ? "unknown"
                    : resolvedDiagnostic.substr(fieldOffset + sizeof(kFieldMarker) - 1);
            diagnostic = std::string("ResourceDescriptorMismatch phase=6A component=") +
                         component + " field=" + field;
        };

        DiffusionFilterOpticsRecipe& cameraDiffusion = out.cameraDiffusion;
        std::string resolvedDiagnostic;
        if (!Spektrafilm::resolve_diffusion_filter(
                controls.cameraDiffusion,
                cameraDiffusion.resolved,
                resolvedDiagnostic)) {
            set_diffusion_diagnostic("camera_diffusion", resolvedDiagnostic);
            return false;
        }
        if (cameraDiffusion.resolved.active) {
            cameraDiffusion.domain =
                Spektrafilm::SpatialOpticsDomain::FilmLinearExposure;
            cameraDiffusion.hash = hash_diffusion_filter_optics(cameraDiffusion);
        }

        CameraLensBlurOpticsRecipe& lensBlur = out.cameraLensBlur;
        if (!std::isfinite(controls.cameraLensBlurUm)) {
            diagnostic =
                "ResourceDescriptorMismatch phase=6A component=camera_lens_blur field=sigma_um";
            return false;
        }
        lensBlur.sigmaUm = std::max(controls.cameraLensBlurUm, 0.0f);
        if (lensBlur.sigmaUm > 0.0f) {
            lensBlur.hash = hash_camera_lens_blur_optics(lensBlur);
        }

        ScatterHalationOpticsRecipe& scatterHalation = out.scatterHalation;
        if (!Spektrafilm::resolve_scatter_halation_recipe(
                controls.scatterHalation,
                profile.digest,
                scatterHalation,
                diagnostic)) {
            return false;
        }

        DiffusionFilterOpticsRecipe& enlargerDiffusion = out.enlargerDiffusion;
        if (includePrintDomain) {
            if (!Spektrafilm::resolve_diffusion_filter(
                    controls.enlargerDiffusion,
                    enlargerDiffusion.resolved,
                    resolvedDiagnostic)) {
                set_diffusion_diagnostic("enlarger_diffusion", resolvedDiagnostic);
                return false;
            }
            if (enlargerDiffusion.resolved.active) {
                enlargerDiffusion.domain =
                    Spektrafilm::SpatialOpticsDomain::PrintLinearExposure;
                enlargerDiffusion.hash = hash_diffusion_filter_optics(enlargerDiffusion);
            }
        }

        std::uint64_t enabledHash = Hash::kFnvOffset;
        bool anyEnabled = false;
        const auto mix_component = [&](Spektrafilm::SpatialOpticsComponent component,
                                       std::uint64_t componentHash) {
            if (componentHash == 0) {
                return;
            }
            hash_value(enabledHash, component);
            hash_value(enabledHash, componentHash);
            anyEnabled = true;
        };
        mix_component(
            Spektrafilm::SpatialOpticsComponent::CameraDiffusion,
            cameraDiffusion.hash);
        mix_component(
            Spektrafilm::SpatialOpticsComponent::CameraLensBlur,
            lensBlur.hash);
        mix_component(
            Spektrafilm::SpatialOpticsComponent::InEmulsionScatterHalation,
            scatterHalation.hash);
        mix_component(
            Spektrafilm::SpatialOpticsComponent::EnlargerDiffusion,
            enlargerDiffusion.hash);
        out.hash = anyEnabled ? enabledHash : 0;
        return true;
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
        const std::array<std::array<float, 3>, 81>& linearSensitivity,
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
                const float sensitivity = linearSensitivity[wavelengthIndex][channel];
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
                const float source = linearSensitivity[wavelengthIndex][channel];
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

            std::array<double, 3> windowNormalization{};
            for (std::size_t channel = 0; channel < windowNormalization.size(); ++channel) {
                if (!(std::isfinite(response[channel]) && response[channel] > 0.0) ||
                    !(std::isfinite(windowedResponse[channel]) && windowedResponse[channel] > 0.0)) {
                    return false;
                }
                windowNormalization[channel] = windowedResponse[channel] / response[channel];
            }

            for (std::size_t wavelengthIndex = 0; wavelengthIndex < recipe.finalSensitivity.size(); ++wavelengthIndex) {
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    const double adapted =
                        static_cast<double>(recipe.finalSensitivity[wavelengthIndex][channel]) *
                        static_cast<double>(window[wavelengthIndex]) /
                        windowNormalization[channel];
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
        if (!(std::isfinite(greenMidgray) && greenMidgray > 0.0)) {
            return false;
        }
        recipe.mallettGreenMidgrayScale =
            static_cast<float>(1.0 / greenMidgray);
        if (!std::isfinite(recipe.mallettGreenMidgrayScale) ||
            !(recipe.mallettGreenMidgrayScale > 0.0f)) {
            return false;
        }
        return recipe.finalSensitivityHash != 0;
    }

    std::uint64_t hash_film_develop_recipe(const FilmDevelopRecipe& recipe) {
        std::uint64_t hash = Hash::kFnvOffset;
        hash_value(hash, recipe.polarity);
        hash_value(hash, recipe.normalizedDensityCurvesHash);
        hash_value(hash, recipe.densityCurvesLayersRequired);
        if (recipe.densityCurvesLayersRequired) {
            hash_value(hash, recipe.densityCurvesLayersHash);
        }
        Hash::hash_bytes_update(hash, recipe.densityCurveGamma.data(), sizeof(recipe.densityCurveGamma));
        return hash;
    }

    bool finite_nonnegative_grain_triplet(const std::array<float, 3>& values) {
        for (float value : values) {
            if (!std::isfinite(value) || value < 0.0f) {
                return false;
            }
        }
        return true;
    }

    bool finite_positive_triplet(const std::array<float, 3>& values) {
        for (float value : values) {
            if (!std::isfinite(value) || value <= 0.0f) {
                return false;
            }
        }
        return true;
    }

    bool finite_nonnegative_pair(const std::array<float, 2>& values) {
        for (float value : values) {
            if (!std::isfinite(value) || value < 0.0f) {
                return false;
            }
        }
        return true;
    }

    bool build_grain_contract(const GrainContract& input, GrainContract& out) {
        out = input;
        if (!finite_nonnegative_grain_triplet(out.densityMinCmy)) {
            return false;
        }
        for (float& value : out.densityMinCmy) {
            value = std::clamp(value, 0.0f, 1.0f);
        }
        return true;
    }

    std::uint64_t hash_visual_grain_recipe(const VisualGrainRecipe& recipe) {
        if (!recipe.active) {
            return 0;
        }
        std::uint64_t hash = Hash::kFnvOffset;
        hash_value(hash, recipe.active);
        hash_value(hash, recipe.sublayersActive);
        hash_value(hash, recipe.particleAreaUm2);
        Hash::hash_bytes_update(hash, recipe.particleScaleCmy.data(), sizeof(recipe.particleScaleCmy));
        Hash::hash_bytes_update(
            hash,
            recipe.particleScaleLayers.data(),
            sizeof(recipe.particleScaleLayers));
        Hash::hash_bytes_update(
            hash,
            recipe.visualParticleDensityMinCmy.data(),
            sizeof(recipe.visualParticleDensityMinCmy));
        Hash::hash_bytes_update(hash, recipe.uniformityCmy.data(), sizeof(recipe.uniformityCmy));
        hash_value(hash, recipe.correlationSigmaPx);
        hash_value(hash, recipe.dyeCloudBlurUm);
        Hash::hash_bytes_update(hash, recipe.microStructure.data(), sizeof(recipe.microStructure));
        hash_value(hash, recipe.nSubLayers);
        hash_value(hash, recipe.amplitude);
        hash_value(hash, recipe.chromaMix);
        hash_value(hash, recipe.chromaSharedWeight);
        hash_value(hash, recipe.chromaIndependentWeight);
        hash_value(hash, recipe.fineWeight);
        hash_value(hash, recipe.midWeight);
        hash_value(hash, recipe.coarseWeight);
        hash_value(hash, recipe.sizeMixScale);
        hash_value(hash, recipe.clumpTemporalMix);
        hash_value(hash, recipe.clumpMorphPeriodSec);
        hash_value(hash, recipe.debugView);
        hash_value(hash, recipe.densityCurvesLayersHash);
        return hash;
    }

    float normalize_effect_amount(float value) {
        return std::isfinite(value)
                   ? std::clamp(value, 0.0f, 10.0f)
                   : 0.0f;
    }

    double normalize_effect_amount(double value) {
        return std::isfinite(value)
                   ? std::clamp(value, 0.0, 10.0)
                   : 0.0;
    }

    void hash_defect_policy(std::uint64_t& hash, const DefectDustRecipe& policy) {
        if (!(policy.slotProbability > 0.0f)) {
            return;
        }
        hash_value(hash, policy.cellWidthMm);
        hash_value(hash, policy.cellHeightMm);
        hash_value(hash, policy.slotProbability);
        hash_value(hash, policy.softnessMinMm);
        hash_value(hash, policy.softnessMaxMm);
        hash_value(hash, policy.softnessSizeCapFraction);
        hash_value(hash, policy.supportXMm);
        hash_value(hash, policy.supportYMm);
        hash_value(hash, policy.fiberFraction);
        hash_value(hash, policy.fiberDriftFraction);
        hash_value(hash, policy.fiberFirstKnotMin);
        hash_value(hash, policy.fiberFirstKnotMax);
        hash_value(hash, policy.fiberSecondKnotMin);
        hash_value(hash, policy.fiberSecondKnotMax);
        hash_value(hash, policy.fiberInteriorWidthMinFraction);
        hash_value(hash, policy.fiberInteriorWidthMaxFraction);
        hash_value(hash, policy.diameterMinMm);
        hash_value(hash, policy.diameterBulkMaxMm);
        hash_value(hash, policy.diameterMaxMm);
        hash_value(hash, policy.diameterTailFraction);
        hash_value(hash, policy.fiberLengthMinMm);
        hash_value(hash, policy.fiberLengthMaxMm);
        hash_value(hash, policy.fiberWidthMinMm);
        hash_value(hash, policy.fiberWidthMaxMm);
        hash_value(hash, policy.opacityFaintCumulative);
        hash_value(hash, policy.opacityIntermediateCumulative);
        hash_value(hash, policy.compactOpacityMin);
        hash_value(hash, policy.compactOpacityFaintEnd);
        hash_value(hash, policy.compactOpacityIntermediateEnd);
        hash_value(hash, policy.compactOpacityMax);
        hash_value(hash, policy.fiberOpacityMin);
        hash_value(hash, policy.fiberOpacityFaintEnd);
        hash_value(hash, policy.fiberOpacityIntermediateEnd);
        hash_value(hash, policy.fiberOpacityMax);
        hash_value(hash, policy.compactDominantAspectMin);
        hash_value(hash, policy.compactDominantAspectMax);
        hash_value(hash, policy.compactSubsidiaryScaleMin);
        hash_value(hash, policy.compactSubsidiaryScaleMax);
        hash_value(hash, policy.compactSubsidiaryAspectMin);
        hash_value(hash, policy.compactSubsidiaryAspectMax);
        hash_value(hash, policy.compactSubsidiaryOffsetMax);
        hash_value(hash, policy.compactSubsidiaryAngleMaxRadians);
    }

    void hash_defect_policy(std::uint64_t& hash, const DefectScratchRecipe& policy) {
        if (!(policy.slotProbability > 0.0f)) {
            return;
        }
        hash_value(hash, policy.cellWidthMm);
        hash_value(hash, policy.cellHeightMm);
        hash_value(hash, policy.slotProbability);
        hash_value(hash, policy.softnessMinMm);
        hash_value(hash, policy.softnessMaxMm);
        hash_value(hash, policy.softnessSizeCapFraction);
        hash_value(hash, policy.supportXMm);
        hash_value(hash, policy.supportYMm);
        hash_value(hash, policy.lengthMinMm);
        hash_value(hash, policy.lengthBulkMaxMm);
        hash_value(hash, policy.lengthMaxMm);
        hash_value(hash, policy.lengthTailFraction);
        hash_value(hash, policy.widthMinMm);
        hash_value(hash, policy.widthBulkMaxMm);
        hash_value(hash, policy.widthMaxMm);
        hash_value(hash, policy.widthTailFraction);
        hash_value(hash, policy.driftFraction);
        hash_value(hash, policy.firstKnotMin);
        hash_value(hash, policy.firstKnotMax);
        hash_value(hash, policy.secondKnotMin);
        hash_value(hash, policy.secondKnotMax);
        hash_value(hash, policy.interiorWidthMinFraction);
        hash_value(hash, policy.interiorWidthMaxFraction);
        hash_value(hash, policy.interiorDepthMinFraction);
        hash_value(hash, policy.interiorDepthMaxFraction);
        hash_value(hash, policy.endpointAbruptProbability);
        hash_value(hash, policy.interruptionProbability);
        hash_value(hash, policy.gapCenterMin);
        hash_value(hash, policy.gapCenterMax);
        hash_value(hash, policy.gapSpanMin);
        hash_value(hash, policy.gapSpanMax);
        hash_value(hash, policy.scuffProbability);
        hash_value(hash, policy.scuffLengthMaxMm);
        hash_value(hash, policy.scuffAngleMaxRadians);
        hash_value(hash, policy.strengthMin);
        hash_value(hash, policy.strengthMax);
    }

    std::uint64_t hash_film_juicer_effects_recipe(const FilmJuicerEffectsRecipe& recipe) {
        if (!recipe.active) {
            return 0;
        }
        std::uint64_t hash = Hash::kFnvOffset;
        hash_value(hash, recipe.filmDust.slotProbability > 0.0f);
        hash_defect_policy(hash, recipe.filmDust);
        hash_value(hash, recipe.filmScratch.slotProbability > 0.0f);
        hash_defect_policy(hash, recipe.filmScratch);
        hash_value(hash, recipe.gateDust.slotProbability > 0.0f);
        hash_defect_policy(hash, recipe.gateDust);
        hash_value(hash, recipe.gateScratch.slotProbability > 0.0f);
        hash_defect_policy(hash, recipe.gateScratch);
        hash_value(hash, recipe.gateWeaveAmount);
        hash_value(hash, recipe.active);
        return hash;
    }

    bool build_film_juicer_effects_recipe(
        const Spektrafilm::FilmFoundationBuildInput& input,
        const VisualGrainRecipe& grain,
        FilmJuicerEffectsRecipe& out) {
        out = {};
        if (grain.active && grain.debugView != 0) {
            return true;
        }
        struct DustDefaults {
            float maximumRate;
            float softnessMinMm;
            float softnessMaxMm;
        };
        const auto dust = [](float amount, const DustDefaults& defaults) {
            DefectDustRecipe p{};
            if (!(amount > 0.0f)) {
                return p;
            }
            p.cellWidthMm = 0.5f;
            p.cellHeightMm = 0.5f;
            p.slotProbability = defaults.maximumRate * std::pow(amount / 10.0f, 2.2f) *
                                p.cellWidthMm * p.cellHeightMm / 2.0f;
            p.softnessMinMm = defaults.softnessMinMm;
            p.softnessMaxMm = defaults.softnessMaxMm;
            p.softnessSizeCapFraction = 0.25f;
            p.fiberFraction = 0.25f;
            p.fiberDriftFraction = 0.10f;
            p.fiberFirstKnotMin = 0.08f;
            p.fiberFirstKnotMax = 0.35f;
            p.fiberSecondKnotMin = 0.65f;
            p.fiberSecondKnotMax = 0.92f;
            p.fiberInteriorWidthMinFraction = 0.45f;
            p.fiberInteriorWidthMaxFraction = 1.0f;
            p.diameterMinMm = 0.008f;
            p.diameterBulkMaxMm = 0.080f;
            p.diameterMaxMm = 0.250f;
            p.diameterTailFraction = 0.20f;
            p.fiberLengthMinMm = 0.04f;
            p.fiberLengthMaxMm = 0.65f;
            p.fiberWidthMinMm = 0.003f;
            p.fiberWidthMaxMm = 0.025f;
            p.opacityFaintCumulative = 0.50f;
            p.opacityIntermediateCumulative = 0.85f;
            p.compactOpacityMin = 0.10f;
            p.compactOpacityFaintEnd = 0.45f;
            p.compactOpacityIntermediateEnd = 0.85f;
            p.compactOpacityMax = 0.99f;
            p.fiberOpacityMin = 0.08f;
            p.fiberOpacityFaintEnd = 0.35f;
            p.fiberOpacityIntermediateEnd = 0.75f;
            p.fiberOpacityMax = 0.97f;
            p.compactDominantAspectMin = 0.52f;
            p.compactDominantAspectMax = 0.95f;
            p.compactSubsidiaryScaleMin = 0.30f;
            p.compactSubsidiaryScaleMax = 0.75f;
            p.compactSubsidiaryAspectMin = 0.45f;
            p.compactSubsidiaryAspectMax = 1.0f;
            p.compactSubsidiaryOffsetMax = 0.68f;
            p.compactSubsidiaryAngleMaxRadians = 3.14159265359f;
            // Descriptor validation recomputes this bound, so keep its rounding explicit.
            p.supportXMm =
                std::fma(p.fiberLengthMaxMm, 0.5f + p.fiberDriftFraction, p.fiberWidthMaxMm);
            p.supportYMm = p.supportXMm;
            return p;
        };
        struct ScratchDefaults {
            std::array<float, 2> maximumLengthDensityAndStrength;
            float softnessMinMm;
            float softnessMaxMm;
        };
        const auto scratch = [](float amount, const ScratchDefaults& defaults) {
            const auto& maximumLengthDensityAndStrength = defaults.maximumLengthDensityAndStrength;
            const float maximumLengthDensity = maximumLengthDensityAndStrength[0];
            const float strengthMax = maximumLengthDensityAndStrength[1];
            DefectScratchRecipe p{};
            if (!(amount > 0.0f)) {
                return p;
            }
            p.cellWidthMm = 1.0f;
            p.cellHeightMm = 8.0f;
            p.lengthMinMm = 0.25f;
            p.lengthBulkMaxMm = 6.0f;
            p.lengthMaxMm = 24.0f;
            p.lengthTailFraction = 0.16f;
            // Each mixture branch uses u^3: E[u^3] = 1/4. Length is arc length.
            const float expectedLength =
                (1.0f - p.lengthTailFraction) * (p.lengthMinMm +
                                                 (p.lengthBulkMaxMm - p.lengthMinMm) * 0.25f) +
                p.lengthTailFraction * (p.lengthBulkMaxMm +
                                        (p.lengthMaxMm - p.lengthBulkMaxMm) * 0.25f);
            p.slotProbability = maximumLengthDensity * std::pow(amount / 10.0f, 2.2f) /
                                expectedLength * p.cellWidthMm * p.cellHeightMm / 2.0f;
            p.widthMinMm = 0.002f;
            p.widthBulkMaxMm = 0.025f;
            p.widthMaxMm = 0.080f;
            p.widthTailFraction = 0.18f;
            p.driftFraction = 0.025f;
            p.firstKnotMin = 0.06f;
            p.firstKnotMax = 0.28f;
            p.secondKnotMin = 0.72f;
            p.secondKnotMax = 0.94f;
            p.interiorWidthMinFraction = 0.35f;
            p.interiorWidthMaxFraction = 1.0f;
            p.interiorDepthMinFraction = 0.35f;
            p.interiorDepthMaxFraction = 1.0f;
            p.endpointAbruptProbability = 0.30f;
            p.interruptionProbability = 0.15f;
            p.gapCenterMin = 0.25f;
            p.gapCenterMax = 0.75f;
            p.gapSpanMin = 0.02f;
            p.gapSpanMax = 0.12f;
            p.scuffProbability = 0.10f;
            p.scuffLengthMaxMm = 0.75f;
            p.scuffAngleMaxRadians = 0.61086523820f;
            p.strengthMin = strengthMax * 0.15f;
            p.strengthMax = strengthMax;
            p.softnessMinMm = defaults.softnessMinMm;
            p.softnessMaxMm = defaults.softnessMaxMm;
            p.softnessSizeCapFraction = 0.25f;
            const float transportSupportX = p.lengthMaxMm * p.driftFraction + p.widthMaxMm;
            const float scuffSupportX = 0.5f * p.scuffLengthMaxMm * std::sin(p.scuffAngleMaxRadians) +
                                        p.scuffLengthMaxMm * p.driftFraction + p.widthMaxMm;
            const float scuffSupportY = 0.5f * p.scuffLengthMaxMm * std::cos(p.scuffAngleMaxRadians) +
                                        p.scuffLengthMaxMm * p.driftFraction + p.widthMaxMm;
            p.supportXMm = std::max(transportSupportX, scuffSupportX);
            p.supportYMm = std::max(p.lengthMaxMm * 0.5f + p.widthMaxMm, scuffSupportY);
            return p;
        };
        out.filmDust = dust(normalize_effect_amount(input.filmDustAmount), {0.80f, 0.00025f, 0.003f});
        out.gateDust = dust(normalize_effect_amount(input.gateDustAmount), {0.40f, 0.0005f, 0.005f});
        out.filmScratch = scratch(
            normalize_effect_amount(input.filmScratchAmount),
            {{0.080f, 0.95f}, 0.00015f, 0.0015f});
        out.gateScratch = scratch(
            normalize_effect_amount(input.gateScratchAmount),
            {{0.040f, 0.85f}, 0.00025f, 0.003f});
        out.gateWeaveAmount = normalize_effect_amount(input.gateWeaveAmount);
        for (float probability : {out.filmDust.slotProbability, out.gateDust.slotProbability, out.filmScratch.slotProbability, out.gateScratch.slotProbability}) {
            if (!std::isfinite(probability) || probability < 0.0f || probability >= 0.25f) {
                out = {};
                return false;
            }
        }
        out.active = out.filmDust.slotProbability > 0.0f || out.filmScratch.slotProbability > 0.0f ||
                     out.gateDust.slotProbability > 0.0f || out.gateScratch.slotProbability > 0.0f ||
                     out.gateWeaveAmount > 0.0;
        out.hash = hash_film_juicer_effects_recipe(out);
        return true;
    }

    bool build_visual_grain_recipe(
        const VisualGrainControls& input,
        const FilmDevelopRecipe& filmDevelop,
        VisualGrainRecipe& out) {
        out = VisualGrainRecipe{};
        if (!input.active) {
            out.hash = 0;
            return true;
        }
        if (!std::isfinite(input.particleAreaUm2) || input.particleAreaUm2 <= 0.0f ||
            !finite_positive_triplet(input.particleScaleCmy) ||
            !finite_positive_triplet(input.particleScaleLayers) ||
            !finite_nonnegative_grain_triplet(input.visualParticleDensityMinCmy) ||
            !finite_nonnegative_grain_triplet(input.uniformityCmy) ||
            !std::isfinite(input.correlationSigmaPx) || input.correlationSigmaPx < 0.0f ||
            !std::isfinite(input.dyeCloudBlurUm) || input.dyeCloudBlurUm < 0.0f ||
            !finite_nonnegative_pair(input.microStructure) ||
            input.nSubLayers <= 0 ||
            !std::isfinite(input.amplitude) || input.amplitude < 0.0f ||
            !std::isfinite(input.chromaMix) ||
            !std::isfinite(input.chromaSharedWeight) || input.chromaSharedWeight < 0.0f ||
            !std::isfinite(input.chromaIndependentWeight) ||
            input.chromaIndependentWeight < 0.0f ||
            !std::isfinite(input.coarseWeight) || input.coarseWeight < 0.0f ||
            !std::isfinite(input.midWeight) || input.midWeight < 0.0f ||
            !std::isfinite(input.sizeMixScale) || input.sizeMixScale < 1.0f ||
            !std::isfinite(input.clumpTemporalMix) || input.clumpTemporalMix < 0.0f ||
            !std::isfinite(input.clumpMorphPeriodSec) || input.clumpMorphPeriodSec < 0.0f ||
            (input.sublayersActive &&
             filmDevelop.densityCurvesLayersHash == 0)) {
            return false;
        }

        out.active = true;
        out.sublayersActive = input.sublayersActive;
        if (out.sublayersActive) {
            derive_grain_layer_axis_search_metadata(
                filmDevelop,
                out.grainLayerAxisFinite,
                out.grainLayerAxisBlockPrefixMax);
        }
        out.particleAreaUm2 = input.particleAreaUm2;
        out.particleScaleCmy = input.particleScaleCmy;
        out.particleScaleLayers = input.particleScaleLayers;
        out.visualParticleDensityMinCmy = input.visualParticleDensityMinCmy;
        out.uniformityCmy = input.uniformityCmy;
        for (float& value : out.visualParticleDensityMinCmy) {
            value = std::clamp(value, 0.0f, 1.0f);
        }
        for (float& value : out.uniformityCmy) {
            value = std::clamp(value, 0.0f, 1.0f);
        }
        out.correlationSigmaPx = input.correlationSigmaPx;
        out.dyeCloudBlurUm = input.dyeCloudBlurUm;
        out.microStructure = input.microStructure;
        out.nSubLayers = input.nSubLayers;
        out.amplitude = input.amplitude;
        out.chromaMix = std::clamp(input.chromaMix, 0.0f, 1.0f);
        out.chromaSharedWeight = std::clamp(input.chromaSharedWeight, 0.0f, 1.0f);
        out.chromaIndependentWeight =
            std::clamp(input.chromaIndependentWeight, 0.0f, 1.0f);

        out.midWeight = std::clamp(input.midWeight, 0.0f, 1.0f);
        out.coarseWeight = std::clamp(input.coarseWeight, 0.0f, 1.0f);
        out.fineWeight = std::max(0.0f, 1.0f - out.midWeight - out.coarseWeight);
        const float weightSum = out.fineWeight + out.midWeight + out.coarseWeight;
        if (!(weightSum > 0.0f) || !std::isfinite(weightSum)) {
            return false;
        }
        const float inverseWeightSum = 1.0f / weightSum;
        out.fineWeight *= inverseWeightSum;
        out.midWeight *= inverseWeightSum;
        out.coarseWeight *= inverseWeightSum;

        out.sizeMixScale = input.sizeMixScale;
        out.clumpTemporalMix = std::clamp(input.clumpTemporalMix, 0.0f, 0.30f);
        out.clumpMorphPeriodSec = std::clamp(input.clumpMorphPeriodSec, 5.0f, 60.0f);
        out.debugView = std::clamp(input.debugView, 0, 6);
        out.densityCurvesLayersHash =
            out.sublayersActive
                ? filmDevelop.densityCurvesLayersHash
                : 0;
        out.hash = hash_visual_grain_recipe(out);
        return out.hash != 0;
    }

    bool visual_grain_requires_density_layers(const VisualGrainControls& controls) {
        return controls.active && controls.sublayersActive;
    }

    std::uint64_t hash_density_curves_layers(
        const std::array<std::array<std::vector<float>, 3>, 3>& layers) {
        std::uint64_t hash = Hash::kFnvOffset;
        for (const auto& layer : layers) {
            for (const auto& channel : layer) {
                if (channel.empty()) {
                    return 0;
                }
                hash_value(hash, hash_nan_preserving_floats(channel.data(), channel.size()));
            }
        }
        return hash;
    }

    bool validate_density_curves_layers_shape(
        const std::array<std::array<std::vector<float>, 3>, 3>& layers,
        std::size_t expectedRows) {
        if (expectedRows == 0u) {
            return false;
        }
        for (const auto& layer : layers) {
            for (const auto& channel : layer) {
                if (channel.size() != expectedRows) {
                    return false;
                }
            }
        }
        return true;
    }

    bool finite_nonnegative(float value) {
        return std::isfinite(value) && value >= 0.0f;
    }

    bool dir_extent_valid(const Spektrafilm::DirFrameExtent& extent) noexcept {
        return extent.width > 0 && extent.height > 0;
    }

    bool dir_extents_match(
        const Spektrafilm::DirFrameExtent& a,
        const Spektrafilm::DirFrameExtent& b) noexcept {
        return a.x == b.x &&
               a.y == b.y &&
               a.width == b.width &&
               a.height == b.height;
    }

    Spektrafilm::DirReferenceOperator dir_reference_operator_for_sigma(float sigmaPixels) noexcept {
        if (!(sigmaPixels > 0.0f)) {
            return Spektrafilm::DirReferenceOperator::Identity;
        }
        return sigmaPixels >= 3.0f
                   ? Spektrafilm::DirReferenceOperator::SpektrafilmLargeYvvReplicate
                   : Spektrafilm::DirReferenceOperator::SpektrafilmSmallFirReflect;
    }

    Spektrafilm::DirFilterBackend dir_target_backend_for_operator(
        Spektrafilm::DirReferenceOperator referenceOperator) noexcept {
        switch (referenceOperator) {
            case Spektrafilm::DirReferenceOperator::Identity:
            case Spektrafilm::DirReferenceOperator::SpektrafilmSmallFirReflect:
                return Spektrafilm::DirFilterBackend::SmallFir;
            case Spektrafilm::DirReferenceOperator::SpektrafilmLargeYvvReplicate:
                return Spektrafilm::DirFilterBackend::StrictYvvChannelsAliasedForward;
            default:
                return Spektrafilm::DirFilterBackend::None;
        }
    }

    Spektrafilm::DirScratchTier dir_target_scratch_tier_for_operator(
        Spektrafilm::DirReferenceOperator referenceOperator) noexcept {
        return referenceOperator == Spektrafilm::DirReferenceOperator::SpektrafilmLargeYvvReplicate
                   ? Spektrafilm::DirScratchTier::Tier1IChannels
                   : Spektrafilm::DirScratchTier::Tier1F;
    }

    Spektrafilm::DirScratchPlaneRoles dir_target_plane_roles_for_tier(
        Spektrafilm::DirScratchTier scratchTier) noexcept {
        Spektrafilm::DirScratchPlaneRoles roles{};
        if (scratchTier == Spektrafilm::DirScratchTier::Tier1F ||
            scratchTier == Spektrafilm::DirScratchTier::Tier1IChannels ||
            scratchTier == Spektrafilm::DirScratchTier::Tier2) {
            roles.rawCorrectionPlanes = 3;
            roles.filteredCorrectionPlanes = 3;
            roles.filterTempPlanes =
                scratchTier == Spektrafilm::DirScratchTier::Tier1IChannels ? 3 : 1;
            if (scratchTier == Spektrafilm::DirScratchTier::Tier2) {
                roles.cachedLogRawPlanes = 3;
            }
        }
        return roles;
    }

    Spektrafilm::DirScratchPlaneRoles strict_yvv_channels_aliased_forward_plane_roles() noexcept {
        Spektrafilm::DirScratchPlaneRoles roles =
            dir_target_plane_roles_for_tier(Spektrafilm::DirScratchTier::Tier1IChannels);
        roles.cachedLogRawPlanes = 0;
        return roles;
    }

    bool dir_filter_plan_targets_aliased_forward_yvv(
        const Spektrafilm::DirFilterPlan& plan) noexcept {
        for (int component = 0; component < plan.componentCount; ++component) {
            const Spektrafilm::DirGaussianComponentPlan& componentPlan =
                plan.components[static_cast<std::size_t>(component)];
            if (componentPlan.weight > 0.0f &&
                componentPlan.targetBackend ==
                    Spektrafilm::DirFilterBackend::StrictYvvChannelsAliasedForward) {
                return true;
            }
        }
        return false;
    }

    bool dir_filter_plan_targets_strict_yvv(
        const Spektrafilm::DirFilterPlan& plan) noexcept {
        for (int component = 0; component < plan.componentCount; ++component) {
            const Spektrafilm::DirGaussianComponentPlan& componentPlan =
                plan.components[static_cast<std::size_t>(component)];
            if (componentPlan.weight > 0.0f &&
                componentPlan.targetBackend ==
                    Spektrafilm::DirFilterBackend::StrictYvvChannelsAliasedForward) {
                return true;
            }
        }
        return false;
    }

    Spektrafilm::DirScratchPlaneRoles dir_cached_log_raw_roles_for_base_tier(
        Spektrafilm::DirScratchTier baseTier) noexcept {
        Spektrafilm::DirScratchPlaneRoles roles = dir_target_plane_roles_for_tier(baseTier);
        if (roles.total_float_planes() > 0) {
            roles.cachedLogRawPlanes = 3;
        }
        return roles;
    }

    Spektrafilm::DirScratchTier dir_cached_log_raw_base_tier_for_descriptor(
        const SpatialDirDescriptor& descriptor) noexcept {
        return dir_filter_plan_targets_strict_yvv(descriptor.filterPlan)
                   ? Spektrafilm::DirScratchTier::Tier1IChannels
                   : Spektrafilm::DirScratchTier::Tier1F;
    }

    Spektrafilm::DirScratchPlaneRoles dir_target_plane_roles_for_descriptor(
        const SpatialDirDescriptor& descriptor) noexcept {
        if (descriptor.targetScratchTier == Spektrafilm::DirScratchTier::Tier2) {
            if (dir_filter_plan_targets_aliased_forward_yvv(descriptor.filterPlan)) {
                Spektrafilm::DirScratchPlaneRoles roles =
                    strict_yvv_channels_aliased_forward_plane_roles();
                roles.cachedLogRawPlanes = 3;
                return roles;
            }
            return dir_cached_log_raw_roles_for_base_tier(
                dir_cached_log_raw_base_tier_for_descriptor(descriptor));
        }
        return dir_target_plane_roles_for_tier(descriptor.targetScratchTier);
    }

    Spektrafilm::DirScratchTier dir_build_scratch_tier_for_descriptor(
        const SpatialDirDescriptor& descriptor) noexcept {
        if (descriptor.targetScratchTier == Spektrafilm::DirScratchTier::Tier2) {
            return dir_cached_log_raw_base_tier_for_descriptor(descriptor);
        }
        return descriptor.targetScratchTier;
    }

    Spektrafilm::DirScratchPlaneRoles dir_build_plane_roles_for_descriptor(
        const SpatialDirDescriptor& descriptor) noexcept {
        if (dir_filter_plan_targets_aliased_forward_yvv(descriptor.filterPlan)) {
            return strict_yvv_channels_aliased_forward_plane_roles();
        }
        return dir_target_plane_roles_for_tier(
            dir_build_scratch_tier_for_descriptor(descriptor));
    }

    void configure_cached_log_raw_descriptor(SpatialDirDescriptor& descriptor) noexcept {
        descriptor.targetScratchTier = Spektrafilm::DirScratchTier::Tier2;
        for (int component = 0; component < descriptor.filterPlan.componentCount; ++component) {
            Spektrafilm::DirGaussianComponentPlan& plan =
                descriptor.filterPlan.components[static_cast<std::size_t>(component)];
            if (plan.weight > 0.0f) {
                plan.targetScratchTier = Spektrafilm::DirScratchTier::Tier2;
            }
        }
    }

    struct DirComponentBuildInput {
        float sigmaPixels = 0.0f;
        float weight = 0.0f;
    };

    bool add_dir_component(
        Spektrafilm::DirFilterPlan& plan,
        DirComponentBuildInput input,
        Spektrafilm::DirScratchTier& targetScratchTier) noexcept {
        if (!(input.weight > 0.0f)) {
            return true;
        }
        if (plan.componentCount >= Spektrafilm::DirFilterPlan::kMaxComponents) {
            return false;
        }
        Spektrafilm::DirGaussianComponentPlan& component =
            plan.components[static_cast<std::size_t>(plan.componentCount)];
        component.sigmaPixels = input.sigmaPixels;
        component.weight = input.weight;
        component.referenceOperator = dir_reference_operator_for_sigma(input.sigmaPixels);
        component.targetBackend = dir_target_backend_for_operator(component.referenceOperator);
        component.backend = component.targetBackend;
        component.targetScratchTier = dir_target_scratch_tier_for_operator(component.referenceOperator);
        if (component.targetScratchTier == Spektrafilm::DirScratchTier::Tier1IChannels) {
            targetScratchTier = Spektrafilm::DirScratchTier::Tier1IChannels;
        } else if (targetScratchTier == Spektrafilm::DirScratchTier::Tier0) {
            targetScratchTier = Spektrafilm::DirScratchTier::Tier1F;
        }
        ++plan.componentCount;
        return true;
    }

    void hash_dir_extent(std::uint64_t& hash, const Spektrafilm::DirFrameExtent& extent) {
        hash_value(hash, extent.x);
        hash_value(hash, extent.y);
        hash_value(hash, extent.width);
        hash_value(hash, extent.height);
    }

    void hash_dir_plane_roles(std::uint64_t& hash, const Spektrafilm::DirScratchPlaneRoles& roles) {
        hash_value(hash, roles.rawCorrectionPlanes);
        hash_value(hash, roles.filteredCorrectionPlanes);
        hash_value(hash, roles.filterTempPlanes);
        hash_value(hash, roles.cachedLogRawPlanes);
    }

    void hash_dir_filter_plan(std::uint64_t& hash, const Spektrafilm::DirFilterPlan& plan) {
        hash_value(hash, plan.componentCount);
        for (int i = 0; i < plan.componentCount; ++i) {
            const Spektrafilm::DirGaussianComponentPlan& component =
                plan.components[static_cast<std::size_t>(i)];
            hash_value(hash, component.sigmaPixels);
            hash_value(hash, component.weight);
            hash_value(hash, component.referenceOperator);
            hash_value(hash, component.backend);
            hash_value(hash, component.targetBackend);
            hash_value(hash, component.targetScratchTier);
        }
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
        for (const auto& row : recipe.matrixRgb) {
            Hash::hash_bytes_update(hash, row.data(), sizeof(float) * row.size());
        }
        Hash::hash_bytes_update(hash, recipe.densityMaxRgb.data(), sizeof(recipe.densityMaxRgb));
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
            !finite_nonnegative(controls.diffusionSizeUm)) {
            return false;
        }

        const std::array<float, 3>& gammaSameLayerRgb =
            controls.gammaUseStock ? profile.digest.gammaSamelayerRgb : controls.gammaSameLayerRgb;
        const std::array<float, 2>& gammaInterlayerRToGb =
            controls.gammaUseStock ? profile.digest.gammaInterlayerRToGb : controls.gammaInterlayerRToGb;
        const std::array<float, 2>& gammaInterlayerGToRb =
            controls.gammaUseStock ? profile.digest.gammaInterlayerGToRb : controls.gammaInterlayerGToRb;
        const std::array<float, 2>& gammaInterlayerBToRg =
            controls.gammaUseStock ? profile.digest.gammaInterlayerBToRg : controls.gammaInterlayerBToRg;
        const auto finiteNonnegativeArray = [](const auto& values) {
            return std::all_of(values.begin(), values.end(), [](float value) {
                return finite_nonnegative(value);
            });
        };
        if (!finiteNonnegativeArray(gammaSameLayerRgb) ||
            !finiteNonnegativeArray(gammaInterlayerRToGb) ||
            !finiteNonnegativeArray(gammaInterlayerGToRb) ||
            !finiteNonnegativeArray(gammaInterlayerBToRg)) {
            return false;
        }

        out.polarity = profile.info.type;
        out.active = controls.active && controls.amount > 0.0f;
        if (!out.active) {
            return true;
        }
        out.diffusionSizeUm = controls.diffusionSizeUm;
        out.diffusionTailUm = 200.0f;
        out.diffusionTailWeight = 0.06f;

        out.matrixRgb[0][0] = gammaSameLayerRgb[0] * controls.inhibitionSameLayer;
        out.matrixRgb[1][1] = gammaSameLayerRgb[1] * controls.inhibitionSameLayer;
        out.matrixRgb[2][2] = gammaSameLayerRgb[2] * controls.inhibitionSameLayer;
        out.matrixRgb[0][1] = gammaInterlayerRToGb[0] * controls.inhibitionInterlayer;
        out.matrixRgb[0][2] = gammaInterlayerRToGb[1] * controls.inhibitionInterlayer;
        out.matrixRgb[1][0] = gammaInterlayerGToRb[0] * controls.inhibitionInterlayer;
        out.matrixRgb[1][2] = gammaInterlayerGToRb[1] * controls.inhibitionInterlayer;
        out.matrixRgb[2][0] = gammaInterlayerBToRg[0] * controls.inhibitionInterlayer;
        out.matrixRgb[2][1] = gammaInterlayerBToRg[1] * controls.inhibitionInterlayer;
        for (auto& row : out.matrixRgb) {
            for (float& value : row) {
                value *= controls.amount;
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
        hash_value(hash, recipe.blackCorrection);
        hash_value(hash, recipe.whiteCorrection);
        hash_value(hash, recipe.blackLevel);
        hash_value(hash, recipe.whiteLevel);
        hash_value(hash, recipe.glareActive);
        hash_value(hash, recipe.glarePercent);
        hash_value(hash, recipe.glareRoughness);
        hash_value(hash, recipe.glareBlurSigmaPx);
        hash_value(hash, recipe.lensBlurSigmaPx);
        hash_value(hash, recipe.unsharpSigmaPx);
        hash_value(hash, recipe.unsharpAmount);
        return hash;
    }

    bool build_film_density_bounds(
        const Profiles::ValidatedFilmProfile& profile,
        const FilmDevelopRecipe& develop,
        const GrainContract& grain,
        Spektrafilm::ScanRoute route,
        Spektrafilm::DensityBoundsSource source,
        DensityBoundsRecipe& out) {
        out = DensityBoundsRecipe{};
        out.route = route;
        out.medium = Spektrafilm::DensityMedium::Film;
        out.polarity = profile.info.type;
        out.source = source;
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

    bool build_print_density_bounds(
        const Profiles::ValidatedPrintProfile& profile,
        Spektrafilm::ScanRoute route,
        Spektrafilm::ProfilePolarity capturePolarity,
        DensityBoundsRecipe& out) {
        out = DensityBoundsRecipe{};
        out.route = route;
        out.medium = Spektrafilm::DensityMedium::Print;
        out.polarity = capturePolarity;
        out.source = Spektrafilm::DensityBoundsSource::PrintMediaAuthoredCurves;
        std::vector<std::array<float, 3>> normalized;
        if (!normalize_density_curves(
                profile.data.densityCurves,
                normalized,
                out.authoredMinCmy,
                out.authoredMaxCmy)) {
            return false;
        }
        for (std::size_t channel = 0; channel < out.dataMinCmy.size(); ++channel) {
            out.dataMinCmy[channel] = out.authoredMinCmy[channel];
            out.dataMaxCmy[channel] = out.authoredMaxCmy[channel];
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

    std::uint64_t hash_dichroic_filter_recipe(const DichroicFilterRecipe& recipe) {
        std::uint64_t hash = Hash::kFnvOffset;
        Hash::hash_bytes_update(hash, recipe.customEdgesNm.data(), sizeof(recipe.customEdgesNm));
        Hash::hash_bytes_update(hash, recipe.customTransitionsNm.data(), sizeof(recipe.customTransitionsNm));
        return hash;
    }

    std::uint64_t hash_print_filter_recipe(const PrintFilterRecipe& recipe) {
        std::uint64_t hash = Hash::kFnvOffset;
        hash_cmy(hash, recipe.mainCmyCc);
        hash_cmy(hash, recipe.preflashCmyCc);
        hash_value(hash, recipe.dichroic.hash);
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
        return hash;
    }

    constexpr std::uint64_t kDiffusionFrameFnvPrime = 0x100000001b3ULL;

    void fail_diffusion_frame_descriptor(
        std::string& diagnostic,
        std::string_view field) {
        diagnostic = "InvalidDiffusionFrameDescriptor field=";
        diagnostic.append(field);
    }

    void hash_diffusion_frame_byte(
        std::uint64_t& hash,
        std::uint8_t value) {
        hash ^= static_cast<std::uint64_t>(value);
        hash *= kDiffusionFrameFnvPrime;
    }

    void hash_diffusion_frame_u32_le(
        std::uint64_t& hash,
        std::uint32_t value) {
        for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
            hash_diffusion_frame_byte(
                hash,
                static_cast<std::uint8_t>((value >> (byte * 8)) & 0xffu));
        }
    }

    void hash_diffusion_frame_u64_le(
        std::uint64_t& hash,
        std::uint64_t value) {
        for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
            hash_diffusion_frame_byte(
                hash,
                static_cast<std::uint8_t>((value >> (byte * 8)) & 0xffu));
        }
    }

    void hash_diffusion_frame_tag(
        std::uint64_t& hash,
        std::string_view tag) {
        for (const char value : tag) {
            hash_diffusion_frame_byte(
                hash,
                static_cast<std::uint8_t>(value));
        }
        hash_diffusion_frame_byte(hash, 0);
    }

    void hash_diffusion_frame_double(
        std::uint64_t& hash,
        double value) {
        if (value == 0.0) {
            value = 0.0;
        }
        hash_diffusion_frame_u64_le(
            hash,
            std::bit_cast<std::uint64_t>(value));
    }

    void hash_diffusion_frame_domain(
        std::uint64_t& hash,
        const Spektrafilm::DiffusionFrameDomain& domain) {
        hash_diffusion_frame_u32_le(
            hash,
            static_cast<std::uint32_t>(domain.originX));
        hash_diffusion_frame_u32_le(
            hash,
            static_cast<std::uint32_t>(domain.originY));
        hash_diffusion_frame_u32_le(
            hash,
            static_cast<std::uint32_t>(domain.width));
        hash_diffusion_frame_u32_le(
            hash,
            static_cast<std::uint32_t>(domain.height));
    }

    bool valid_diffusion_frame_route(Spektrafilm::ScanRoute route) {
        using Spektrafilm::ScanRoute;
        switch (route) {
            case ScanRoute::NegativeDirectScan:
            case ScanRoute::NegativePrintScan:
            case ScanRoute::PositiveDirectScan:
            case ScanRoute::PositivePrintScan:
                return true;
            default:
                return false;
        }
    }

    bool build_diffusion_stage_frame_descriptor(
        const DiffusionFilterOpticsRecipe& component,
        Spektrafilm::DiffusionLinearStage stage,
        double pixelSizeUm,
        Spektrafilm::DiffusionFrameDomain fullFrame,
        Spektrafilm::DiffusionStageFrameDescriptor& out,
        std::string& diagnostic) {
        const bool cameraStage =
            stage == Spektrafilm::DiffusionLinearStage::CameraFilmLinear;
        const char* inactiveField = cameraStage
                                        ? "camera_recipe_component_hash"
                                        : "enlarger_recipe_component_hash";
        const char* resolvedField =
            cameraStage ? "camera_resolved_hash" : "enlarger_resolved_hash";
        const char* scatterField = cameraStage
                                       ? "camera_scatter_fraction"
                                       : "enlarger_scatter_fraction";
        const char* domainField =
            cameraStage ? "camera_linear_domain" : "enlarger_linear_domain";
        const char* sampleField =
            cameraStage ? "camera_sample" : "enlarger_sample";
        const char* stageHashField =
            cameraStage ? "camera_hash" : "enlarger_hash";

        if (component.hash == 0) {
            fail_diffusion_frame_descriptor(diagnostic, inactiveField);
            return false;
        }
        if (!component.resolved.active || component.resolved.hash == 0) {
            fail_diffusion_frame_descriptor(diagnostic, resolvedField);
            return false;
        }
        if (!std::isfinite(component.resolved.scatterFraction) ||
            component.resolved.scatterFraction <= 0.0 ||
            component.resolved.scatterFraction > 1.0) {
            fail_diffusion_frame_descriptor(diagnostic, scatterField);
            return false;
        }
        const Spektrafilm::SpatialOpticsDomain expectedDomain =
            cameraStage
                ? Spektrafilm::SpatialOpticsDomain::FilmLinearExposure
                : Spektrafilm::SpatialOpticsDomain::PrintLinearExposure;
        if (component.domain != expectedDomain) {
            fail_diffusion_frame_descriptor(diagnostic, domainField);
            return false;
        }

        Spektrafilm::DiffusionPsfSampleDescriptor sample{};
        std::string sampleDiagnostic;
        if (!Spektrafilm::build_diffusion_psf_sample_descriptor(
                component.resolved,
                pixelSizeUm,
                fullFrame.width,
                fullFrame.height,
                sample,
                sampleDiagnostic) ||
            sample.hash == 0 || sample.radiusPixels <= 0) {
            fail_diffusion_frame_descriptor(diagnostic, sampleField);
            return false;
        }

        out = Spektrafilm::DiffusionStageFrameDescriptor{};
        out.stage = stage;
        out.scatterFraction = component.resolved.scatterFraction;
        out.sample = sample;

        std::uint64_t hash = Hash::kFnvOffset;
        hash_diffusion_frame_tag(hash, "diffusion-stage-frame");
        hash_diffusion_frame_u32_le(
            hash,
            Spektrafilm::kDiffusionFrameDescriptorSchemaVersion);
        hash_diffusion_frame_byte(
            hash,
            static_cast<std::uint8_t>(out.stage));
        hash_diffusion_frame_double(hash, out.scatterFraction);
        hash_diffusion_frame_u64_le(hash, out.sample.hash);
        out.hash = hash;
        if (out.hash == 0) {
            fail_diffusion_frame_descriptor(diagnostic, stageHashField);
            return false;
        }
        return true;
    }

} // namespace

namespace Spektrafilm {
    std::array<double, 3> evaluate_print_density_sample(
        const Profiles::PrintDensityModel& model,
        double gammaFactor,
        ProfilePolarity polarity,
        double logExposure) {
        std::array<double, 3> density{};
        if (polarity != ProfilePolarity::Negative &&
            polarity != ProfilePolarity::Positive) {
            density.fill(std::numeric_limits<double>::quiet_NaN());
            return density;
        }
        for (std::size_t channel = 0; channel < density.size(); ++channel) {
            for (std::size_t layer = 0; layer < model.centers[channel].size(); ++layer) {
                double center = model.centers[channel][layer];
                double sigma = model.sigmas[channel][layer];
                if (gammaFactor != 1.0) {
                    center /= gammaFactor;
                    sigma = std::max(sigma / gammaFactor, 0.05);
                }
                double z = (logExposure - center) / sigma;
                if (polarity == ProfilePolarity::Positive) {
                    z = -z;
                }
                density[channel] +=
                    model.amplitudes[channel][layer] * normal_cdf(z);
            }
        }
        return density;
    }

    namespace {

        bool build_print_develop_recipe(
            const Profiles::ValidatedPrintProfile& profile,
            double gammaFactor,
            PrintDevelopRecipe& out) {
            out = PrintDevelopRecipe{};
            const std::size_t sampleCount = profile.sourceLogExposure.size();
            if (sampleCount == 0u ||
                sampleCount != profile.data.logExposure.size() ||
                sampleCount > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                (profile.info.type != ProfilePolarity::Negative &&
                 profile.info.type != ProfilePolarity::Positive)) {
                return false;
            }

            out.gammaFactor = gammaFactor;
            out.densityCurves.resize(sampleCount);
            for (std::size_t sample = 0; sample < sampleCount; ++sample) {
                const double sourceExposure = profile.sourceLogExposure[sample];
                const float uploadExposure = profile.data.logExposure[sample];
                if (!std::isfinite(sourceExposure) ||
                    !std::isfinite(uploadExposure) ||
                    static_cast<float>(sourceExposure) != uploadExposure ||
                    (sample > 0u &&
                     (sourceExposure < profile.sourceLogExposure[sample - 1u] ||
                      uploadExposure < profile.data.logExposure[sample - 1u]))) {
                    return false;
                }
                const std::array<double, 3> derived =
                    evaluate_print_density_sample(
                        profile.densityModel,
                        gammaFactor,
                        profile.info.type,
                        sourceExposure);
                for (std::size_t channel = 0; channel < derived.size(); ++channel) {
                    const float converted = static_cast<float>(derived[channel]);
                    if (!std::isfinite(derived[channel]) || !std::isfinite(converted)) {
                        return false;
                    }
                    out.densityCurves[sample][channel] = converted;
                }
            }

            out.densityCurvesHash = Hash::kFnvOffset;
            hash_value(out.densityCurvesHash, sampleCount);
            Hash::hash_bytes_update(
                out.densityCurvesHash,
                profile.data.logExposure.data(),
                sampleCount * sizeof(profile.data.logExposure.front()));
            Hash::hash_bytes_update(
                out.densityCurvesHash,
                &out.densityCurves[0][0],
                sampleCount * 3u * sizeof(float));
            return out.densityCurvesHash != 0;
        }

        bool build_film_foundation(
            const FilmFoundationBuildInput& input,
            RenderRecipe& recipe,
            std::string& diagnostic) {
            const auto fail = [&diagnostic](const char* message) {
                diagnostic = message;
                return false;
            };
            if (!input.filmProfile) {
                return fail("MissingRequiredResource phase=3B field=film_profile");
            }
            if (!input.referenceIlluminantValid) {
                return fail("MissingRequiredResource phase=3B field=reference_illuminant");
            }
            if (!auto_exposure_method_index_valid(input.cameraMeteringMethod)) {
                return fail("UnsupportedMode phase=3B field=auto_exposure_method");
            }
            if (!std::isfinite(input.filmGammaFactor) ||
                input.filmGammaFactor <
                    static_cast<float>(kFilmGammaFactorMinimum) ||
                input.filmGammaFactor >
                    static_cast<float>(kFilmGammaFactorMaximum)) {
                return fail(
                    "InvalidAuthoredControl component=film_development field=film_gamma_factor");
            }

            const Profiles::ValidatedFilmProfile& profile = *input.filmProfile;
            const ScanRoute resolvedRoute = resolve_scan_route(profile.info.type, input.scanRoute);
            if (resolvedRoute != input.scanRoute ||
                profile.info.support != ProfileSupport::Film ||
                profile.info.stage != ProfileStage::Filming) {
                return fail("UnsupportedMode phase=3A selected capture profile route mismatch");
            }

            ProfileRoute& profileRoute = recipe.profileRoute;
            profileRoute.capturePolarity = profile.info.type;
            profileRoute.filmProfileAssetVersionToken = profile.assetVersionToken;
            profileRoute.printProfileAssetVersionToken = 0;
            profileRoute.filmProfile = input.filmProfile;
            profileRoute.printProfile.reset();

            FilmRawRecipe& filmRaw = recipe.filmRaw;
            filmRaw.inputColorSpace = input.inputColorSpace;
            filmRaw.inputCctfDecoding = input.inputCctfDecoding;
            filmRaw.rgbToRawMethod = input.spectralUpsamplingMode == 1
                                         ? RgbToRawMethod::Mallett2019
                                         : RgbToRawMethod::Hanatos2025;
            filmRaw.autoExposureEnabled = input.cameraAutoExposureEnabled;
            filmRaw.autoExposureMethod = auto_exposure_method_from_index(input.cameraMeteringMethod);
            filmRaw.manualExposureCompensationEv = input.manualExposureCompensationEv;
            filmRaw.filmFormatLongEdgeMm = input.filmFormatLongEdgeMm;
            filmRaw.cameraBandPass.active = input.cameraFilterOverride;
            filmRaw.cameraBandPass.uv = copy_filter_triplet(input.cameraFilterUV, input.cameraFilterOverride);
            filmRaw.cameraBandPass.ir = copy_filter_triplet(input.cameraFilterIR, input.cameraFilterOverride);
            filmRaw.hanatos.applyWindow = input.applyHanatos2025AdaptationWindow;
            filmRaw.hanatos.applySurface = input.applyHanatos2025AdaptationSurface;
            filmRaw.hanatos.spectralGaussianBlur = profile.digest.hanatosSpectralGaussianBlurDefault;
            filmRaw.hanatos.windowParams = profile.data.hanatos2025AdaptationWindowParams;
            filmRaw.hanatos.surfaceParams = profile.data.hanatos2025AdaptationSurfaceParams;
            filmRaw.hanatos.referenceIlluminant = profile.info.referenceIlluminant.value;
            if (filmRaw.rgbToRawMethod == RgbToRawMethod::Hanatos2025) {
                if (filmRaw.hanatos.applyWindow &&
                    !profile.data.hasHanatos2025AdaptationWindowParams) {
                    return fail("MalformedRequiredProfileData phase=3D-2 field=data.hanatos2025_adaptation_window_params");
                }
                if (filmRaw.hanatos.applySurface &&
                    !profile.data.hasHanatos2025AdaptationSurfaceParams) {
                    return fail("MalformedRequiredProfileData phase=3D-2 field=data.hanatos2025_adaptation_surface_params");
                }
                if (!std::isfinite(filmRaw.hanatos.spectralGaussianBlur) ||
                    filmRaw.hanatos.spectralGaussianBlur < 0.0f) {
                    return fail("MalformedRequiredProfileData phase=3D-2 field=settings.spectral_gaussian_blur");
                }
            }
            if (!derive_final_sensitivity(
                    profile.data.linearSensitivity,
                    filmRaw,
                    input.referenceIlluminant)) {
                return fail("MalformedRequiredProfileData phase=3B field=final_sensitivity");
            }
            filmRaw.hash = hash_film_raw_recipe(filmRaw);
            if (filmRaw.finalSensitivityHash == 0 ||
                (filmRaw.rgbToRawMethod == RgbToRawMethod::Hanatos2025 &&
                 filmRaw.hanatosLutHash == 0) ||
                filmRaw.hash == 0) {
                return fail("MalformedRequiredProfileData phase=3B field=data.log_sensitivity");
            }

            if (!build_spatial_optics_recipe(
                    profile,
                    input.spatialOptics,
                    scan_route_is_print(input.scanRoute),
                    recipe.spatialOptics,
                    diagnostic)) {
                return false;
            }

            if (!build_grain_contract(input.grainContract, recipe.grainContract)) {
                return fail("MalformedRequiredProfileData phase=9A field=grain_contract");
            }

            FilmDevelopRecipe& filmDevelop = recipe.filmDevelop;
            filmDevelop.polarity = profile.info.type;
            filmDevelop.densityCurveGamma.fill(input.filmGammaFactor);
            filmDevelop.logExposure = profile.data.logExposure;
            filmDevelop.authoredDensityCurves = profile.data.densityCurves;
            if (!normalize_density_curves(
                    filmDevelop.authoredDensityCurves,
                    filmDevelop.normalizedDensityCurves,
                    filmDevelop.authoredMinCmy,
                    filmDevelop.authoredMaxCmy)) {
                return fail("MalformedRequiredProfileData phase=3A field=data.density_curves");
            }
            if (filmDevelop.authoredDensityCurves.empty() ||
                filmDevelop.normalizedDensityCurves.empty()) {
                return fail("MalformedRequiredProfileData phase=3A field=data.density_curves empty");
            }
            filmDevelop.normalizedDensityCurvesHash = hash_nan_preserving_floats(
                &filmDevelop.normalizedDensityCurves[0][0],
                filmDevelop.normalizedDensityCurves.size() * 3u);
            filmDevelop.densityCurvesLayersRequired =
                visual_grain_requires_density_layers(input.visualGrain);
            if (filmDevelop.densityCurvesLayersRequired) {
                if (profile.data.densityCurvesLayersMalformed) {
                    diagnostic =
                        profile.data.densityCurvesLayersDiagnostic.empty()
                            ? "MalformedRequiredProfileData phase=9B field=data.density_curves_layers"
                            : profile.data.densityCurvesLayersDiagnostic;
                    return false;
                }
                if (!profile.data.hasDensityCurvesLayers) {
                    return fail("MissingRequiredResource phase=9B field=data.density_curves_layers");
                }
                if (!validate_density_curves_layers_shape(
                        profile.data.densityCurvesLayers,
                        filmDevelop.logExposure.size())) {
                    return fail("MalformedRequiredProfileData phase=9B field=data.density_curves_layers shape");
                }
                filmDevelop.densityCurvesLayers = profile.data.densityCurvesLayers;
                filmDevelop.densityCurvesLayersHash =
                    hash_density_curves_layers(filmDevelop.densityCurvesLayers);
                if (filmDevelop.densityCurvesLayersHash == 0) {
                    return fail("MalformedRequiredProfileData phase=9B field=data.density_curves_layers hash");
                }
            }
            filmDevelop.hash = hash_film_develop_recipe(filmDevelop);
            if (filmDevelop.normalizedDensityCurvesHash == 0 ||
                filmDevelop.hash == 0) {
                return fail("MalformedRequiredProfileData phase=3A field=data.density_curves hash");
            }
            if (!build_visual_grain_recipe(
                    input.visualGrain,
                    filmDevelop,
                    recipe.visualGrain)) {
                return fail("MalformedRequiredProfileData phase=9A field=visual_grain");
            }
            if (!build_film_juicer_effects_recipe(input, recipe.visualGrain, recipe.filmJuicerEffects)) {
                return fail("ResourceDescriptorMismatch phase=effects_recipe field=physical_policy");
            }

            if (!build_dir_couplers_recipe(
                    profile,
                    filmDevelop,
                    input.dirCouplers,
                    recipe.dirCouplers)) {
                return fail("MalformedRequiredProfileData phase=3D-3 field=dirCouplers");
            }

            const bool printRoute = scan_route_is_print(input.scanRoute);
            DensityBoundsRecipe& filmBounds =
                printRoute ? recipe.enlargerFilmBounds : recipe.densityBounds;
            const DensityBoundsSource source = printRoute
                                                   ? DensityBoundsSource::EnlargerFilmGrainContractAndAuthoredCurves
                                                   : DensityBoundsSource::DirectFilmGrainContractAndAuthoredCurves;
            if (!build_film_density_bounds(
                    profile,
                    filmDevelop,
                    recipe.grainContract,
                    input.scanRoute,
                    source,
                    filmBounds)) {
                return fail("MalformedRequiredProfileData phase=3A field=density_bounds");
            }
            return true;
        }

        std::uint64_t append_optional_film_feature_hashes(
            std::uint64_t hash,
            const RenderRecipe& recipe) {
            if (recipe.spatialOptics.hash != 0) {
                hash = Hash::hash_uint64_values({hash, recipe.spatialOptics.hash});
            }
            if (recipe.visualGrain.hash != 0) {
                hash = Hash::hash_uint64_values({hash, recipe.visualGrain.hash});
            }
            if (recipe.filmJuicerEffects.hash != 0) {
                hash = Hash::hash_uint64_values({hash, recipe.filmJuicerEffects.hash});
            }
            return hash;
        }

    } // namespace

    DirectRecipeBuildResult build_direct_render_recipe(const DirectRecipeBuildInput& input) {
        DirectRecipeBuildResult result{};
        RenderRecipe& recipe = result.recipe;
        const FilmFoundationBuildInput& film = input.film;
        recipe =
            make_render_recipe(film.filmProfileKey, {}, film.scanRoute);
        if (scan_route_is_print(film.scanRoute)) {
            result.diagnostic =
                "UnsupportedMode phase=3A field=scan_route expected=direct";
            return result;
        }
        if (!std::isfinite(input.scannerBlackLevel) ||
            !std::isfinite(input.scannerWhiteLevel) ||
            !std::isfinite(input.scannerLensBlurSigmaPx) ||
            !std::isfinite(input.scannerUnsharpSigmaPx) ||
            !std::isfinite(input.scannerUnsharpAmount)) {
            result.diagnostic =
                "ResourceDescriptorMismatch phase=8 field=scanner_output";
            return result;
        }
        if (!build_film_foundation(film, recipe, result.diagnostic)) {
            return result;
        }

        ProfileRoute& profileRoute = recipe.profileRoute;
        profileRoute.hash = hash_profile_route(profileRoute);

        const Profiles::ValidatedFilmProfile& profile = *film.filmProfile;
        ScannerOutputRecipe& scanner = recipe.scannerOutput;
        scanner.route = film.scanRoute;
        scanner.medium = DensityMedium::Film;
        scanner.polarity = profile.info.type;
        scanner.viewingIlluminant = profile.info.viewingIlluminant.value;
        scanner.lutResolution =
            std::clamp(input.scannerLutResolution, 17u, 128u);
        scanner.outputColorSpace = input.outputColorSpace;
        scanner.outputCctfEncoding = input.outputCctfEncoding;
        scanner.blackCorrection = input.scannerBlackCorrection;
        scanner.whiteCorrection = input.scannerWhiteCorrection;
        scanner.blackLevel = input.scannerBlackLevel;
        scanner.whiteLevel = input.scannerWhiteLevel;
        scanner.lensBlurSigmaPx = input.scannerLensBlurSigmaPx;
        scanner.unsharpSigmaPx = input.scannerUnsharpSigmaPx;
        scanner.unsharpAmount = input.scannerUnsharpAmount;
        scanner.hash = hash_scanner_output_recipe(scanner);

        recipe.directStructuralReady = true;
        if (recipe.dirCouplers.active) {
            recipe.hash = Hash::hash_uint64_values(
                {profileRoute.hash,
                 recipe.filmRaw.hash,
                 recipe.filmDevelop.hash,
                 recipe.dirCouplers.hash,
                 recipe.densityBounds.hash,
                 scanner.hash});
        } else {
            recipe.hash = Hash::hash_uint64_values(
                {profileRoute.hash,
                 recipe.filmRaw.hash,
                 recipe.filmDevelop.hash,
                 recipe.densityBounds.hash,
                 scanner.hash});
        }
        recipe.hash =
            append_optional_film_feature_hashes(
                recipe.hash,
                recipe);
        result.valid = recipe.hash != 0;
        if (!result.valid) {
            result.diagnostic =
                "ResourceDescriptorMismatch phase=3A field=render_recipe_hash";
        }
        return result;
    }
    PrintRecipeBuildResult build_print_render_recipe(
        const PrintRecipeBuildInput& input) {
        PrintRecipeBuildResult result{};
        RenderRecipe& recipe = result.recipe;
        const FilmFoundationBuildInput& film = input.film;
        recipe = make_render_recipe(
            film.filmProfileKey,
            input.printProfileKey,
            film.scanRoute);
        if (!scan_route_is_print(film.scanRoute)) {
            result.diagnostic =
                "UnsupportedMode phase=4A field=scan_route expected=print";
            return result;
        }
        if (!film.filmProfile || !input.printProfile) {
            result.diagnostic =
                "MissingRequiredResource phase=4A field=selected_profile";
            return result;
        }
        if (!std::isfinite(input.printGammaFactor) ||
            input.printGammaFactor < kPrintGammaFactorMinimum ||
            input.printGammaFactor > kPrintGammaFactorMaximum) {
            result.diagnostic =
                "InvalidAuthoredControl component=print_development field=print_gamma_factor";
            return result;
        }
        if (input.printProfile->info.stage != ProfileStage::Printing) {
            result.diagnostic =
                "UnsupportedMode phase=4A selected profile route mismatch";
            return result;
        }
        if (input.printIlluminantKey.empty() ||
            !finite_cmy(input.currentNeutralCmyCc) ||
            !finite_cmy(input.calibratedNeutralCmyCc) ||
            !std::all_of(
                input.uiYmcCc.begin(),
                input.uiYmcCc.end(),
                [](float value) {
                    return std::isfinite(value);
                }) ||
            !std::isfinite(input.preflashMFilterCc) ||
            !std::isfinite(input.preflashYFilterCc) ||
            !std::isfinite(input.printExposure) ||
            !std::isfinite(input.preflashExposure) ||
            !std::isfinite(film.manualExposureCompensationEv) ||
            !std::isfinite(input.scannerBlackLevel) ||
            !std::isfinite(input.scannerWhiteLevel) ||
            !std::isfinite(input.glarePercent) ||
            !std::isfinite(input.glareRoughness) ||
            !std::isfinite(input.glareBlurSigmaPx) ||
            !std::isfinite(input.scannerLensBlurSigmaPx) ||
            !std::isfinite(input.scannerUnsharpSigmaPx) ||
            !std::isfinite(input.scannerUnsharpAmount)) {
            result.diagnostic =
                "ResourceDescriptorMismatch phase=4A field=print_recipe_input";
            return result;
        }
        if (!build_film_foundation(film, recipe, result.diagnostic)) {
            return result;
        }

        ProfileRoute& route = recipe.profileRoute;
        route.printProfileAssetVersionToken =
            input.printProfile->assetVersionToken;
        route.printProfile = input.printProfile;
        route.hash = hash_profile_route(route);

        if (!build_print_density_bounds(
                *input.printProfile,
                film.scanRoute,
                film.filmProfile->info.type,
                recipe.densityBounds)) {
            result.diagnostic =
                "MalformedRequiredProfileData phase=4C field=print_density_bounds";
            return result;
        }

        ScannerOutputRecipe& scanner = recipe.scannerOutput;
        scanner.route = film.scanRoute;
        scanner.medium = DensityMedium::Print;
        scanner.polarity = film.filmProfile->info.type;
        scanner.viewingIlluminant =
            input.printProfile->info.viewingIlluminant.value;
        scanner.lutResolution =
            std::clamp(input.scannerLutResolution, 17u, 128u);
        scanner.outputColorSpace = input.outputColorSpace;
        scanner.outputCctfEncoding = input.outputCctfEncoding;
        scanner.blackCorrection = input.scannerBlackCorrection;
        scanner.whiteCorrection = input.scannerWhiteCorrection;
        scanner.blackLevel = input.scannerBlackLevel;
        scanner.whiteLevel = input.scannerWhiteLevel;
        scanner.glareActive = input.glareActive;
        scanner.glarePercent = input.glarePercent;
        scanner.glareRoughness = input.glareRoughness;
        scanner.glareBlurSigmaPx = input.glareBlurSigmaPx;
        scanner.lensBlurSigmaPx = input.scannerLensBlurSigmaPx;
        scanner.unsharpSigmaPx = input.scannerUnsharpSigmaPx;
        scanner.unsharpAmount = input.scannerUnsharpAmount;
        scanner.hash = hash_scanner_output_recipe(scanner);

        PrintRecipe& print = recipe.print;
        if (!build_print_develop_recipe(
                *input.printProfile,
                input.printGammaFactor,
                print.develop)) {
            result.diagnostic =
                "MalformedRequiredProfileData phase=3B field=print_development_table";
            return result;
        }
        print.filters.dichroic = DichroicFilterRecipe{};
        print.filters.dichroic.hash = hash_dichroic_filter_recipe(print.filters.dichroic);
        const CmyCcTriplet neutralCmyCc =
            input.neutralCalibrationStatus ==
                    NeutralCalibrationStatus::Calibrated
                ? input.calibratedNeutralCmyCc
                : input.currentNeutralCmyCc;
        const CmyCcTriplet userCmyCc = {
            input.uiYmcCc[2],
            input.uiYmcCc[1],
            input.uiYmcCc[0]};
        print.filters.mainCmyCc = {
            neutralCmyCc.c + userCmyCc.c,
            neutralCmyCc.m + userCmyCc.m,
            neutralCmyCc.y + userCmyCc.y};
        print.filters.preflashCmyCc = {
            neutralCmyCc.c,
            neutralCmyCc.m + input.preflashMFilterCc,
            neutralCmyCc.y + input.preflashYFilterCc};
        print.filters.hash = hash_print_filter_recipe(print.filters);

        print.exposure.printExposure = input.printExposure;
        print.exposure.preflashExposure = input.preflashExposure;
        print.exposure.cameraExposureCompensationEv =
            film.manualExposureCompensationEv;
        print.exposure.normalizePrintExposure = input.normalizePrintExposure;
        print.exposure.printExposureCompensation =
            input.printExposureCompensation;
        print.exposure.normalizationMode = print_normalization_mode(
            input.normalizePrintExposure,
            input.printExposureCompensation);
        print.exposure.hash = hash_print_exposure_recipe(print.exposure);

        print.illuminant.key = input.printIlluminantKey;
        print.illuminant.hash = Hash::kFnvOffset;
        hash_string(print.illuminant.hash, print.illuminant.key);

        print.hash = Hash::hash_uint64_values(
            {print.filters.hash,
             print.exposure.hash,
             print.illuminant.hash,
             print.develop.densityCurvesHash,
             std::bit_cast<std::uint64_t>(print.develop.gammaFactor)});
        recipe.printStructuralReady = true;
        if (recipe.dirCouplers.active) {
            recipe.hash = Hash::hash_uint64_values(
                {route.hash,
                 recipe.filmRaw.hash,
                 recipe.filmDevelop.hash,
                 recipe.dirCouplers.hash,
                 recipe.enlargerFilmBounds.hash,
                 recipe.densityBounds.hash,
                 scanner.hash,
                 print.hash});
        } else {
            recipe.hash = Hash::hash_uint64_values(
                {route.hash,
                 recipe.filmRaw.hash,
                 recipe.filmDevelop.hash,
                 recipe.enlargerFilmBounds.hash,
                 recipe.densityBounds.hash,
                 scanner.hash,
                 print.hash});
        }
        recipe.hash =
            append_optional_film_feature_hashes(
                recipe.hash,
                recipe);
        result.valid = route.hash != 0 &&
                       print.filters.hash != 0 &&
                       print.exposure.hash != 0 &&
                       print.illuminant.hash != 0 &&
                       print.develop.densityCurvesHash != 0 &&
                       !print.develop.densityCurves.empty() &&
                       recipe.filmRaw.hash != 0 &&
                       recipe.filmDevelop.hash != 0 &&
                       (!recipe.dirCouplers.active ||
                        recipe.dirCouplers.hash != 0) &&
                       recipe.enlargerFilmBounds.hash != 0 &&
                       recipe.densityBounds.hash != 0 &&
                       scanner.hash != 0 &&
                       print.hash != 0 &&
                       recipe.hash != 0;
        if (!result.valid) {
            result.diagnostic =
                "ResourceDescriptorMismatch phase=4A field=print_recipe_hash";
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

    bool build_diffusion_frame_set_descriptor(
        const SpatialOptics& optics,
        ScanRoute route,
        double pixelSizeUm,
        DiffusionFrameDomain fullFrame,
        std::optional<DiffusionFrameSetDescriptor>& out,
        std::string& diagnostic) {
        out.reset();
        diagnostic.clear();

        const bool cameraActive = optics.cameraDiffusion.hash != 0;
        const bool enlargerActive = optics.enlargerDiffusion.hash != 0;
        if (!cameraActive && !enlargerActive) {
            return true;
        }
        if (!valid_diffusion_frame_route(route)) {
            fail_diffusion_frame_descriptor(diagnostic, "route");
            return false;
        }
        if (fullFrame.width < 2 || fullFrame.height < 2) {
            fail_diffusion_frame_descriptor(diagnostic, "full_frame_domain");
            return false;
        }
        if (!(std::isfinite(pixelSizeUm) && pixelSizeUm > 0.0)) {
            fail_diffusion_frame_descriptor(diagnostic, "pixel_size_um");
            return false;
        }
        if (enlargerActive && !scan_route_is_print(route)) {
            fail_diffusion_frame_descriptor(diagnostic, "enlarger_route");
            return false;
        }

        DiffusionFrameSetDescriptor descriptor{};
        descriptor.route = route;
        descriptor.fullFrame = fullFrame;
        if (cameraActive) {
            DiffusionStageFrameDescriptor camera{};
            if (!build_diffusion_stage_frame_descriptor(
                    optics.cameraDiffusion,
                    DiffusionLinearStage::CameraFilmLinear,
                    pixelSizeUm,
                    fullFrame,
                    camera,
                    diagnostic)) {
                return false;
            }
            descriptor.camera = camera;
        }
        if (enlargerActive) {
            DiffusionStageFrameDescriptor enlarger{};
            if (!build_diffusion_stage_frame_descriptor(
                    optics.enlargerDiffusion,
                    DiffusionLinearStage::EnlargerPrintLinear,
                    pixelSizeUm,
                    fullFrame,
                    enlarger,
                    diagnostic)) {
                return false;
            }
            descriptor.enlarger = enlarger;
        }

        std::uint64_t hash = Hash::kFnvOffset;
        hash_diffusion_frame_tag(hash, "diffusion-frame-set");
        hash_diffusion_frame_u32_le(
            hash,
            kDiffusionFrameDescriptorSchemaVersion);
        hash_diffusion_frame_byte(
            hash,
            static_cast<std::uint8_t>(descriptor.route));
        hash_diffusion_frame_domain(hash, descriptor.fullFrame);
        hash_diffusion_frame_byte(
            hash,
            descriptor.camera.has_value() ? 1u : 0u);
        if (descriptor.camera) {
            hash_diffusion_frame_u64_le(hash, descriptor.camera->hash);
        }
        hash_diffusion_frame_byte(
            hash,
            descriptor.enlarger.has_value() ? 1u : 0u);
        if (descriptor.enlarger) {
            hash_diffusion_frame_u64_le(hash, descriptor.enlarger->hash);
        }
        descriptor.hash = hash;
        if (descriptor.hash == 0) {
            fail_diffusion_frame_descriptor(diagnostic, "frame_set_hash");
            return false;
        }

        out = descriptor;
        return true;
    }

    bool build_spatial_dir_descriptor(
        const DirCouplersRecipe& recipe,
        float pixelSizeUm,
        Spektrafilm::DirFrameExtent renderExtent,
        Spektrafilm::DirFrameExtent fullFrameExtent,
        const char* traceRouteLabel,
        SpatialDirDescriptor& out) {
        out = SpatialDirDescriptor{};
        if (!recipe.active || !(recipe.diffusionSizeUm > 0.0f)) {
            return true;
        }
        if (!(std::isfinite(pixelSizeUm) && pixelSizeUm > 0.0f) ||
            !dir_extent_valid(renderExtent) ||
            !dir_extent_valid(fullFrameExtent)) {
            return false;
        }
        out.support = Spektrafilm::DirDescriptorSupport::Supported;
        out.dirRecipeHash = recipe.hash;
        out.sourceContract = Spektrafilm::DirSourceContract::FilmLogRawToInitialDensityCmy;
        out.boundaryMode = Spektrafilm::DirBoundaryMode::SpektrafilmReferencePerOperator;
        out.approximation = Spektrafilm::DirApproximationMarker::SpektrafilmStrict;
        out.renderExtent = renderExtent;
        out.fullFrameExtent = fullFrameExtent;
        out.filterDomainExtent = fullFrameExtent;
        out.traceRouteLabel = traceRouteLabel;
        out.gaussianSigmaPixels = recipe.diffusionSizeUm / pixelSizeUm;
        out.gaussianWeight = 1.0f - recipe.diffusionTailWeight;
        if (!(std::isfinite(out.gaussianSigmaPixels) && out.gaussianSigmaPixels > 0.0f) ||
            !finite_nonnegative(out.gaussianWeight)) {
            return false;
        }
        if (!dir_extents_match(renderExtent, fullFrameExtent)) {
            out.support = Spektrafilm::DirDescriptorSupport::UnsupportedPartialRenderWindow;
            out.scratchTier = Spektrafilm::DirScratchTier::Unsupported;
            out.planeRoles = Spektrafilm::DirScratchPlaneRoles{};
            return false;
        }
        out.targetScratchTier = Spektrafilm::DirScratchTier::Tier0;
        if (!add_dir_component(
                out.filterPlan,
                DirComponentBuildInput{out.gaussianSigmaPixels, out.gaussianWeight},
                out.targetScratchTier)) {
            return false;
        }
        if (recipe.diffusionTailWeight > 0.0f) {
            const float tailSigmaPixels = recipe.diffusionTailUm / pixelSizeUm;
            if (!(std::isfinite(tailSigmaPixels) && tailSigmaPixels > 0.0f)) {
                return false;
            }
            const std::size_t tailComponentCount = SpatialDirDescriptor::kExponentialSigmaRatios.size();
            for (std::size_t component = 0; component < tailComponentCount; ++component) {
                const float sigmaRatio = SpatialDirDescriptor::kExponentialSigmaRatios[component];
                const float amplitude = SpatialDirDescriptor::kExponentialAmplitudes[component];
                out.exponentialSigmaPixels[component] =
                    tailSigmaPixels * sigmaRatio;
                out.exponentialWeights[component] =
                    recipe.diffusionTailWeight * amplitude;
                if (!(std::isfinite(out.exponentialSigmaPixels[component]) &&
                      out.exponentialSigmaPixels[component] > 0.0f) ||
                    !finite_nonnegative(out.exponentialWeights[component])) {
                    return false;
                }
                if (!add_dir_component(
                        out.filterPlan,
                        DirComponentBuildInput{
                            out.exponentialSigmaPixels[component],
                            out.exponentialWeights[component]},
                        out.targetScratchTier)) {
                    return false;
                }
            }
        }
        if (out.filterPlan.componentCount <= 0 ||
            out.targetScratchTier == Spektrafilm::DirScratchTier::Tier0) {
            out.support = Spektrafilm::DirDescriptorSupport::UnsupportedScratchTier;
            return false;
        }
        configure_cached_log_raw_descriptor(out);
        out.targetPlaneRoles = dir_target_plane_roles_for_descriptor(out);
        out.scratchTier = dir_build_scratch_tier_for_descriptor(out);
        out.planeRoles = dir_build_plane_roles_for_descriptor(out);

        std::uint64_t hash = Hash::kFnvOffset;
        hash_value(hash, out.dirRecipeHash);
        hash_value(hash, pixelSizeUm);
        hash_value(hash, out.sourceContract);
        hash_value(hash, out.boundaryMode);
        hash_value(hash, out.scratchTier);
        hash_value(hash, out.targetScratchTier);
        hash_value(hash, out.approximation);
        hash_value(hash, out.support);
        hash_dir_extent(hash, out.renderExtent);
        hash_dir_extent(hash, out.fullFrameExtent);
        hash_dir_extent(hash, out.filterDomainExtent);
        hash_dir_filter_plan(hash, out.filterPlan);
        hash_dir_plane_roles(hash, out.planeRoles);
        hash_dir_plane_roles(hash, out.targetPlaneRoles);
        hash_value(hash, out.gaussianSigmaPixels);
        hash_value(hash, out.gaussianWeight);
        Hash::hash_bytes_update(hash, out.exponentialSigmaPixels.data(), sizeof(out.exponentialSigmaPixels));
        Hash::hash_bytes_update(hash, out.exponentialWeights.data(), sizeof(out.exponentialWeights));
        out.hash = hash;
        return out.hash != 0;
    }

    bool preflight_camera_lens_blur(
        const CameraLensBlurOpticsRecipe& recipe,
        std::string& outDiagnostic) {
        outDiagnostic.clear();
        if (recipe.hash == 0) {
            return true;
        }
        outDiagnostic = std::string(kExactOpticsNotImplementedForPhase6) +
                        " component=camera_lens_blur";
        return false;
    }

} // namespace Spektrafilm
