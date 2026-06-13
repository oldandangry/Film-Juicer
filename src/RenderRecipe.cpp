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
        result.recipe.hash = Hash::hash_uint64_values({profileRoute.hash,
                                                       filmRaw.hash,
                                                       filmDevelop.hash,
                                                       result.recipe.densityBounds.hash,
                                                       scanner.hash});
        result.valid = result.recipe.hash != 0;
        if (!result.valid) {
            result.diagnostic = "ResourceDescriptorMismatch phase=3A field=render_recipe_hash";
        }
        return result;
    }

} // namespace Spektrafilm
