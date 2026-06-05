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
            case 1:
                return Spektrafilm::AutoExposureMethod::Median;
            default:
                return Spektrafilm::AutoExposureMethod::CenterWeighted;
        }
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
        hash_value(hash, recipe.hanatos.applyWindow);
        hash_value(hash, recipe.hanatos.applySurface);
        hash_value(hash, recipe.hanatos.spectralGaussianBlur);
        Hash::hash_bytes_update(hash, recipe.hanatos.windowParams.data(), sizeof(recipe.hanatos.windowParams));
        Hash::hash_bytes_update(hash, recipe.hanatos.surfaceParams.data(), sizeof(recipe.hanatos.surfaceParams));
        hash_string(hash, recipe.hanatos.referenceIlluminant);
        hash_value(hash, recipe.highlightBoost.boostEv);
        hash_value(hash, recipe.highlightBoost.boostRange);
        hash_value(hash, recipe.highlightBoost.protectEv);
        hash_value(hash, recipe.linearSensitivityHash);
        return hash;
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
            out.dataMinCmy[channel] = -profile.digest.grainContract.density_min[channel];
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
            result.diagnostic = "MissingRequiredResource phase=3A field=film_profile";
            return result;
        }
        if (!input.directRoutePrintProfileExcluded || !input.directRouteNeutralCalibrationExcluded) {
            result.diagnostic = "ResourceDescriptorMismatch phase=3A direct route print/neutral exclusion";
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
        filmRaw.hash = hash_film_raw_recipe(filmRaw);
        if (filmRaw.linearSensitivityHash == 0 || filmRaw.hash == 0) {
            result.diagnostic = "MalformedRequiredProfileData phase=3A field=data.log_sensitivity";
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
        scanner.postEffectsDisposition = ScannerPostEffectDisposition::BlockedNotImplementedForPhase3;
        scanner.blockingDiagnostic = kScannerPostEffectsNotImplementedForPhase3;
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
