#include "RenderRecipe.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

#include "GamutCompression.h"
#include "RustGamutBridge.h"
#include "RustAssetBridge.h"
#include "RustFilmRecipeBridge.h"
#include "RustDiffusionBridge.h"
#include "RustDirBridge.h"
#include "Cuda/JuicerCudaExecutor.h"
#include "Cuda/JuicerCudaFailure.h"
#include "Hash.h"
#include "ColorTransforms.h"
#if defined(JUICER_ASSET_LOOKUP_TEST_HOOK)
#include "juicer_test_api.h"
#endif

namespace {

    template <typename T>
    void hash_value(std::uint64_t& hash, const T& value) {
        Hash::hash_bytes_update(hash, &value, sizeof(value));
    }

    void hash_string(std::uint64_t& hash, const std::string& value) {
        Hash::hash_bytes_update(hash, value.data(), value.size());
    }

    struct DensityCurveExtrema {
        std::array<float, 3> minimum{};
        std::array<float, 3> maximum{};
    };

    DensityCurveExtrema normalize_density_curves(
        const std::vector<std::array<float, 3>>& authored,
        std::vector<std::array<float, 3>>& normalized) {
        DensityCurveExtrema extrema{authored.front(), authored.front()};
        for (const std::array<float, 3>& row : authored) {
            for (std::size_t channel = 0; channel < row.size(); ++channel) {
                const float value = row[channel];
                extrema.minimum[channel] = std::min(extrema.minimum[channel], value);
                extrema.maximum[channel] = std::max(extrema.maximum[channel], value);
            }
        }
        normalized = authored;
        for (std::array<float, 3>& row : normalized) {
            for (std::size_t channel = 0; channel < row.size(); ++channel) {
                row[channel] -= extrema.minimum[channel];
            }
        }
        return extrema;
    }


    bool auto_exposure_method_index_valid(int index) {
        return index >= 0 && index <= 6;
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


    Spektrafilm::DirScratchTier dir_target_scratch_tier_for_operator(
        Spektrafilm::DirReferenceOperator referenceOperator) noexcept {
        return referenceOperator == Spektrafilm::DirReferenceOperator::SpektrafilmLargeYvvReflect
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

    bool dir_filter_plan_uses_yvv(
        const Spektrafilm::DirFilterPlan& plan) noexcept {
        for (int component = 0; component < plan.componentCount; ++component) {
            const Spektrafilm::DirGaussianComponentPlan& componentPlan =
                plan.components[static_cast<std::size_t>(component)];
            if (componentPlan.weight > 0.0f &&
                componentPlan.referenceOperator ==
                    Spektrafilm::DirReferenceOperator::SpektrafilmLargeYvvReflect) {
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
        return dir_filter_plan_uses_yvv(descriptor.filterPlan)
                   ? Spektrafilm::DirScratchTier::Tier1IChannels
                   : Spektrafilm::DirScratchTier::Tier1F;
    }

    Spektrafilm::DirScratchPlaneRoles dir_target_plane_roles_for_descriptor(
        const SpatialDirDescriptor& descriptor) noexcept {
        if (descriptor.targetScratchTier == Spektrafilm::DirScratchTier::Tier2) {
            if (dir_filter_plan_uses_yvv(descriptor.filterPlan)) {
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
        if (dir_filter_plan_uses_yvv(descriptor.filterPlan)) {
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
            hash_value(hash, component.radius);
            hash_value(hash, component.referenceOperator);
            hash_value(hash, component.targetScratchTier);
            if (component.referenceOperator ==
                Spektrafilm::DirReferenceOperator::SpektrafilmLargeYvvReflect) {
                hash_value(hash, component.iir.feedforward);
                for (double coefficient : component.iir.feedback) {
                    hash_value(hash, coefficient);
                }
                hash_value(hash, component.iir.normalizationDenominator);
                hash_value(hash, component.iir.feedforwardNumerator);
                for (double numerator : component.iir.feedbackNumerators) {
                    hash_value(hash, numerator);
                }
                hash_value(hash, component.boundaryTruncationAccuracy);
                hash_value(hash, component.boundaryCertificationTolerance);
                hash_value(hash, component.boundaryDerivationVersion);
            }
        }
    }

    bool finish_spatial_dir_descriptor(float pixelSizeUm, SpatialDirDescriptor& out) {
        Spektrafilm::resolve_spatial_dir_scratch(out);
        std::uint64_t hash = Hash::kFnvOffset;
        hash_value(hash, out.dirRecipeHash);
        hash_value(hash, pixelSizeUm);
        hash_value(hash, out.scratchTier);
        hash_value(hash, out.targetScratchTier);
        hash_value(hash, out.support);
        hash_dir_extent(hash, out.renderExtent);
        hash_dir_extent(hash, out.fullFrameExtent);
        hash_dir_extent(hash, out.filterDomainExtent);
        hash_dir_filter_plan(hash, out.filterPlan);
        hash_dir_plane_roles(hash, out.planeRoles);
        hash_dir_plane_roles(hash, out.targetPlaneRoles);
        out.hash = hash;
        return out.hash != 0;
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
        hash_value(hash, recipe.syntheticFilmReference.hash);
        hash_value(hash, recipe.outputGamut.hash);
        return hash;
    }

    bool build_output_gamut_recipe(
        int outputColorSpace,
        bool enabled,
        OutputGamutRecipe& out,
        std::string& diagnostic) {
        out = OutputGamutRecipe{};
        if (outputColorSpace < 0 ||
            outputColorSpace >=
                static_cast<int>(OutputEncoding::ColorSpace::Count)) {
            diagnostic =
                "ResourceDescriptorMismatch component=output_gamut_recipe field=output_color_space";
            return false;
        }
        out.enabled = enabled;
        out.outputColorSpace = outputColorSpace;
        if (!JuicerGamut::output_transform(
                OutputEncoding::colorSpaceFromIndex(outputColorSpace),
                out.transform,
                diagnostic)) {
            return false;
        }
        out.transformTableVersionHash =
            out.transform.contractHash;
        if (out.transformTableVersionHash == 0) {
            diagnostic =
                "ResourceDescriptorMismatch component=output_gamut_recipe field=table_contract_identity";
            return false;
        }
        std::uint64_t hash = Hash::kFnvOffset;
        hash_value(hash, out.enabled);
        hash_value(hash, out.lightnessKneeThreshold);
        hash_value(hash, out.lightnessKneeLimit);
        hash_value(hash, out.lightnessKneePower);
        hash_value(hash, out.chromaKneeThreshold);
        hash_value(hash, out.chromaKneeLimit);
        hash_value(hash, out.chromaKneePower);
        hash_value(hash, out.outputColorSpace);
        hash_value(hash, out.transformTableVersionHash);
        out.hash = hash;
        return out.hash != 0;
    }

    bool build_print_density_bounds(
        const Profiles::PrintProfile& profile,
        Spektrafilm::ScanRoute route,
        Spektrafilm::ProfilePolarity capturePolarity,
        DensityBoundsRecipe& out) {
        out = DensityBoundsRecipe{};
        out.route = route;
        out.medium = Spektrafilm::DensityMedium::Print;
        out.polarity = capturePolarity;
        out.source = Spektrafilm::DensityBoundsSource::PrintMediaAuthoredCurves;
        std::vector<std::array<float, 3>> normalized;
        const DensityCurveExtrema extrema = normalize_density_curves(
            profile.data.densityCurves,
            normalized);
        out.authoredMinCmy = extrema.minimum;
        out.authoredMaxCmy = extrema.maximum;
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


} // namespace

#ifdef JUICER_ASSET_LOOKUP_TEST_HOOK
namespace AssetLookupTest {


    float hanatos_window_sample_for_test(float wavelength, const std::array<float, 4>& params) {
        float result = 0.0f;
        std::array<char, 256> bytes{};
        FjErrorBuffer error{bytes.data(), bytes.size(), 0};
        const FjStatus status = fj_test_exposure_window_sample(wavelength, {params.data(), params.size()}, &result, &error);
        if (status.category != FJ_STATUS_SUCCESS) {
            throw std::runtime_error(std::string(bytes.data(), error.length));
        }
        return result;
    }

} // namespace AssetLookupTest
#endif

namespace Spektrafilm {
    namespace {

        bool build_print_develop_recipe(
            const JuicerAssets::PrintProfileSource& source,
            double gammaFactor,
            PrintDevelopRecipe& out) {
            out = PrintDevelopRecipe{};
            try {
                auto curves = source.sample_density_curves(gammaFactor);
                out.gammaFactor = gammaFactor;
                out.densityCurves = std::move(curves.totals);
                out.densityCurvesHash = curves.hash;
                return true;
            } catch (const JuicerCuda::ExecutionFailure& failure) {
                if (failure.failure.status.category != FJ_STATUS_PREPARATION_FAILURE) {
                    throw;
                }
                return false;
            }
        }

        bool build_film_foundation(
            FilmFoundationBuildInput& input,
            RenderRecipe& recipe,
            std::string& diagnostic) {
            const auto fail = [&diagnostic](const char* message) {
                diagnostic = message;
                return false;
            };
            if (!input.filmSource || !input.filmSource->profile()) {
                return fail("MissingRequiredResource phase=3B field=film_profile");
            }
            if (!input.referenceIlluminantValid) {
                return fail("MissingRequiredResource phase=3B field=reference_illuminant");
            }
            if (input.inputColorSpace < 0 ||
                input.inputColorSpace >=
                    static_cast<int>(Spectral::InputColorSpace::Count)) {
                return fail("UnsupportedMode phase=3B field=input_color_space");
            }
            if (input.spectralUpsamplingMode < 0 ||
                input.spectralUpsamplingMode > 2) {
                return fail("UnsupportedMode phase=3B field=spectral_upsampling_method");
            }
            if (!auto_exposure_method_index_valid(input.cameraMeteringMethod)) {
                return fail("UnsupportedMode phase=3B field=auto_exposure_method");
            }
            const Profiles::FilmProfile& profile = *input.filmSource->profile();

            ProfileRoute& profileRoute = recipe.profileRoute;
            profileRoute.capturePolarity = profile.info.type;
            profileRoute.filmProfileAssetVersionToken = profile.assetVersionToken;
            profileRoute.printProfileAssetVersionToken = 0;
            profileRoute.filmProfile = input.filmSource->profile();
            profileRoute.printProfile.reset();

            if (!input.filmSource->prepare_input(input, recipe.filmRaw, diagnostic)) {
                return false;
            }

            if (!JuicerAssets::prepare_spatial_optics(
                    input.spatialOptics,
                    profile.digest,
                    scan_route_is_print(input.scanRoute),
                    recipe.spatialOptics,
                    diagnostic)) {
                return false;
            }

            if (!JuicerAssets::prepare_film_density_minimum(input.grainContract, recipe.grainContract, diagnostic)) {
                return false;
            }

            recipe.filmDevelop = std::move(input.filmDevelop);
            const FilmDevelopRecipe& filmDevelop = recipe.filmDevelop;
            if (!JuicerAssets::prepare_visual_grain(
                    input.developmentHold,
                    input.visualGrain,
                    recipe.visualGrain,
                    diagnostic)) {
                return false;
            }
            const PhysicalEffectsControls effects{input.filmDustAmount, input.filmScratchAmount, input.gateDustAmount, input.gateScratchAmount, input.gateWeaveAmount};
            if (!JuicerAssets::prepare_physical_effects(effects, recipe.visualGrain, recipe.filmJuicerEffects, diagnostic)) {
                return false;
            }

            if (!JuicerAssets::prepare_film_dir(
                    input.developmentHold,
                    input.dirCouplers,
                    profile,
                    input.scanRoute,
                    filmDevelop.logExposure.size(),
                    recipe.dirCouplers,
                    diagnostic)) {
                if (diagnostic.empty()) {
                    return fail("MalformedRequiredProfileData phase=3D-3 field=dirCouplers");
                }
                return false;
            }

            const bool printRoute = scan_route_is_print(input.scanRoute);
            DensityBoundsRecipe& filmBounds =
                printRoute ? recipe.enlargerFilmBounds : recipe.densityBounds;
            if (!JuicerAssets::prepare_film_density_bounds(
                    input.developmentHold,
                    input.grainContract,
                    profile,
                    input.scanRoute,
                    filmBounds,
                    diagnostic)) {
                return false;
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

        std::uint64_t hash_print_recipe(const PrintRecipe& print) {
            return Hash::hash_uint64_values(
                {print.filters.hash,
                 print.exposure.hash,
                 print.illuminant.hash,
                 print.develop.densityCurvesHash,
                 std::bit_cast<std::uint64_t>(print.develop.gammaFactor),
                 print.balance.hash});
        }

        std::uint64_t hash_final_render_recipe(const RenderRecipe& recipe) {
            const bool printRoute = scan_route_is_print(recipe.profileRoute.scanRoute);
            std::uint64_t hash = 0;
            if (printRoute) {
                hash = recipe.dirCouplers.active
                           ? Hash::hash_uint64_values(
                                 {recipe.profileRoute.hash,
                                  recipe.filmRaw.hash,
                                  recipe.filmDevelop.hash,
                                  recipe.dirCouplers.hash,
                                  recipe.enlargerFilmBounds.hash,
                                  recipe.densityBounds.hash,
                                  recipe.scannerOutput.hash,
                                  recipe.print.hash})
                           : Hash::hash_uint64_values(
                                 {recipe.profileRoute.hash,
                                  recipe.filmRaw.hash,
                                  recipe.filmDevelop.hash,
                                  recipe.enlargerFilmBounds.hash,
                                  recipe.densityBounds.hash,
                                  recipe.scannerOutput.hash,
                                  recipe.print.hash});
            } else {
                hash = recipe.dirCouplers.active
                           ? Hash::hash_uint64_values(
                                 {recipe.profileRoute.hash,
                                  recipe.filmRaw.hash,
                                  recipe.filmDevelop.hash,
                                  recipe.dirCouplers.hash,
                                  recipe.densityBounds.hash,
                                  recipe.scannerOutput.hash})
                           : Hash::hash_uint64_values(
                                 {recipe.profileRoute.hash,
                                  recipe.filmRaw.hash,
                                  recipe.filmDevelop.hash,
                                  recipe.densityBounds.hash,
                                  recipe.scannerOutput.hash});
            }
            return append_optional_film_feature_hashes(hash, recipe);
        }

    } // namespace

    DirectRecipeBuildResult build_direct_render_recipe(DirectRecipeBuildInput input) {
        DirectRecipeBuildResult result{};
        RenderRecipe& recipe = result.recipe;
        FilmFoundationBuildInput& film = input.film;
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

        const Profiles::FilmProfile& profile = *film.filmSource->profile();
        ScannerOutputRecipe& scanner = recipe.scannerOutput;
        scanner.route = film.scanRoute;
        scanner.medium = DensityMedium::Film;
        scanner.polarity = profile.info.type;
        scanner.viewingIlluminant = profile.info.viewingIlluminant.value;
        scanner.lutResolution =
            std::clamp(input.scannerLutResolution, 17u, 128u);
        scanner.outputColorSpace = input.outputColorSpace;
        scanner.outputCctfEncoding = input.outputCctfEncoding;
        if (!build_output_gamut_recipe(
                input.outputColorSpace,
                input.outputGamutCompression,
                scanner.outputGamut,
                result.diagnostic)) {
            return result;
        }
        scanner.blackCorrection = input.scannerBlackCorrection;
        scanner.whiteCorrection = input.scannerWhiteCorrection;
        scanner.blackLevel = input.scannerBlackLevel;
        scanner.whiteLevel = input.scannerWhiteLevel;
        scanner.lensBlurSigmaPx = input.scannerLensBlurSigmaPx;
        scanner.unsharpSigmaPx = input.scannerUnsharpSigmaPx;
        scanner.unsharpAmount = input.scannerUnsharpAmount;
        scanner.hash = hash_scanner_output_recipe(scanner);

        recipe.hash = hash_final_render_recipe(recipe);
        result.valid = recipe.hash != 0;
        if (!result.valid) {
            result.diagnostic =
                "ResourceDescriptorMismatch phase=3A field=render_recipe_hash";
        }
        if (result.valid) {
            result.developmentHold = std::move(film.developmentHold);
        }
        return result;
    }
    PrintRecipeBuildResult build_print_render_recipe(
        PrintRecipeBuildInput input) {
        PrintRecipeBuildResult result{};
        RenderRecipe& recipe = result.recipe;
        FilmFoundationBuildInput& film = input.film;
        recipe = make_render_recipe(
            film.filmProfileKey,
            input.printProfileKey,
            film.scanRoute);
        if (!scan_route_is_print(film.scanRoute)) {
            result.diagnostic =
                "UnsupportedMode phase=4A field=scan_route expected=print";
            return result;
        }
        if (!film.filmSource || !film.filmSource->profile() || !input.printSource) {
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
        if (input.printSource->profile()->info.stage != ProfileStage::Printing) {
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
            input.printSource->profile()->assetVersionToken;
        route.printProfile = input.printSource->profile();
        route.hash = hash_profile_route(route);

        if (!build_print_density_bounds(
                *input.printSource->profile(),
                film.scanRoute,
                film.filmSource->profile()->info.type,
                recipe.densityBounds)) {
            result.diagnostic =
                "MalformedRequiredProfileData phase=4C field=print_density_bounds";
            return result;
        }

        ScannerOutputRecipe& scanner = recipe.scannerOutput;
        scanner.route = film.scanRoute;
        scanner.medium = DensityMedium::Print;
        scanner.polarity = film.filmSource->profile()->info.type;
        scanner.viewingIlluminant =
            input.printSource->profile()->info.viewingIlluminant.value;
        scanner.lutResolution =
            std::clamp(input.scannerLutResolution, 17u, 128u);
        scanner.outputColorSpace = input.outputColorSpace;
        scanner.outputCctfEncoding = input.outputCctfEncoding;
        if (!build_output_gamut_recipe(
                input.outputColorSpace,
                input.outputGamutCompression,
                scanner.outputGamut,
                result.diagnostic)) {
            return result;
        }
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
                *input.printSource,
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

        print.hash = hash_print_recipe(print);
        recipe.hash = hash_final_render_recipe(recipe);
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
        if (result.valid) {
            result.developmentHold = std::move(film.developmentHold);
        }
        return result;
    }

    bool finish_synthetic_reference_recipes(
        RenderRecipe& recipe,
        const JuicerAssets::DevelopmentHold& development,
        const Spectral::FilmTcLut* filmTcLut,
        const std::array<float, 81>& filmIlluminant,
        const std::array<float, 81>* filteredMainIlluminant,
        std::string& diagnostic) {
        diagnostic.clear();
        if (!recipe.profileRoute.filmProfile || !development.get()) {
            diagnostic = "ResourceDescriptorMismatch phase=3B field=synthetic_reference_inputs";
            return false;
        }
        SyntheticFilmReferenceRecipe reference{};
        if (!JuicerAssets::prepare_film_reference(development, recipe.filmRaw, filmTcLut, filmIlluminant, reference, diagnostic)) {
            return false;
        }
        recipe.scannerOutput.syntheticFilmReference = reference;
        recipe.scannerOutput.hash = hash_scanner_output_recipe(recipe.scannerOutput);
        if (scan_route_is_print(recipe.profileRoute.scanRoute)) {
            if (!recipe.profileRoute.printProfile || !filteredMainIlluminant) {
                diagnostic = "MissingRequiredResource phase=4A field=filtered_print_illuminant";
                return false;
            }
            PrintBalanceRecipe balance{};
            if (!JuicerAssets::prepare_print_balance(
                    recipe.profileRoute.filmProfile->data,
                    recipe.profileRoute.printProfile->data,
                    *filteredMainIlluminant,
                    reference,
                    recipe.print.exposure.normalizationMode,
                    balance,
                    diagnostic)) {
                return false;
            }
            recipe.print.balance = balance;
            recipe.print.hash = hash_print_recipe(recipe.print);
        }
        recipe.hash = hash_final_render_recipe(recipe);
        return reference.hash != 0 && recipe.scannerOutput.hash != 0 && recipe.hash != 0;
    }


    bool build_diffusion_frame_set_descriptor(
        const SpatialOptics& optics,
        ScanRoute route,
        double pixelSizeUm,
        DiffusionFrameDomain fullFrame,
        std::optional<DiffusionFrameSetDescriptor>& out,
        std::string& diagnostic) {
        JuicerCuda::Failure failure;
        const bool success = JuicerDiffusion::prepare_frame(optics, route, pixelSizeUm, fullFrame, out, failure);
        diagnostic = std::move(failure.diagnostic);
        return success;
    }

    void resolve_spatial_dir_scratch(SpatialDirDescriptor& descriptor) noexcept {
        configure_cached_log_raw_descriptor(descriptor);
        descriptor.targetPlaneRoles = dir_target_plane_roles_for_descriptor(descriptor);
        descriptor.scratchTier = dir_build_scratch_tier_for_descriptor(descriptor);
        descriptor.planeRoles = dir_build_plane_roles_for_descriptor(descriptor);
    }

    bool compose_spatial_dir_descriptor(
        const JuicerDir::MathOutcome& outcome, const char* traceRouteLabel, SpatialDirDescriptor& out) {
        out = SpatialDirDescriptor{};
        const auto& result = outcome.result();
        if (std::holds_alternative<JuicerDir::MathOutcome::Inactive>(result)) {
            return true;
        }
        const auto* active = std::get_if<JuicerDir::MathOutcome::Active>(&result);
        const auto* rejected = std::get_if<JuicerDir::MathOutcome::MathematicalFailure>(&result);
        if (!active && !rejected) {
            return false;
        }
        const auto& math = active ? active->math : rejected->failure.trace;
        if (math.present == 0) {
            return false;
        }
        const auto extent = [](const FjExtent& v) {
            return DirFrameExtent{v.x, v.y, v.width, v.height};
        };
        out.support = DirDescriptorSupport::Supported;
        out.dirRecipeHash = math.recipe_hash;
        out.renderExtent = extent(math.render_extent);
        out.fullFrameExtent = extent(math.full_frame_extent);
        out.filterDomainExtent = extent(math.filter_domain_extent);
        out.traceRouteLabel = traceRouteLabel;
        out.filterPlan.componentCount = static_cast<int>(math.component_count);
        for (std::size_t i = 0; i < out.filterPlan.components.size(); ++i) {
            const auto& source = math.components[i];
            auto& target = out.filterPlan.components[i];
            target.sigmaPixels = source.sigma_px;
            target.weight = source.weight;
            target.radius = source.radius;
            target.referenceOperator = static_cast<DirReferenceOperator>(source.reference_operator);
            target.iir.feedforward = source.feedforward;
            std::copy(std::begin(source.feedback), std::end(source.feedback), target.iir.feedback.begin());
            target.iir.normalizationDenominator = source.normalization_denominator;
            target.iir.feedforwardNumerator = source.feedforward_numerator;
            std::copy(std::begin(source.feedback_numerators), std::end(source.feedback_numerators), target.iir.feedbackNumerators.begin());
            target.boundaryTruncationAccuracy = source.boundary_truncation_accuracy;
            target.boundaryCertificationTolerance = source.boundary_certification_tolerance;
            target.boundaryDerivationVersion = source.boundary_derivation_version;
            if (i < math.component_count) {
                target.targetScratchTier = dir_target_scratch_tier_for_operator(target.referenceOperator);
                if (target.weight > 0.0f) {
                    if (target.targetScratchTier == DirScratchTier::Tier1IChannels) {
                        out.targetScratchTier = DirScratchTier::Tier1IChannels;
                    } else if (out.targetScratchTier == DirScratchTier::Tier0) {
                        out.targetScratchTier = DirScratchTier::Tier1F;
                    }
                }
            }
        }
        if (rejected) {
            if (rejected->failure.field == FJ_DIR_FRAME_FIELD_PARTIAL_RENDER_WINDOW) {
                out.support = DirDescriptorSupport::UnsupportedPartialRenderWindow;
                out.scratchTier = DirScratchTier::Unsupported;
            } else if (rejected->failure.field == FJ_DIR_FRAME_FIELD_EMPTY_PLAN) {
                out.support = DirDescriptorSupport::UnsupportedScratchTier;
            }
            return false;
        }
        return finish_spatial_dir_descriptor(outcome.input().pixel_size_um, out);
    }

    bool build_spatial_dir_descriptor(
        const DirCouplersRecipe& recipe,
        float pixelSizeUm,
        Spektrafilm::DirFrameExtent renderExtent,
        Spektrafilm::DirFrameExtent fullFrameExtent,
        const char* traceRouteLabel,
        SpatialDirDescriptor& out) {
        return compose_spatial_dir_descriptor(
            JuicerDir::prepare_math(recipe, pixelSizeUm, renderExtent, fullFrameExtent), traceRouteLabel, out);
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
