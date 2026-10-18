#include "RenderRecipe.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>

#include "GamutCompression.h"
#include "RustGamutBridge.h"
#include "RustAssetBridge.h"
#include "RustColorBridge.h"
#include "RustExposureBridge.h"
#include "Cuda/JuicerCudaExecutor.h"
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


    std::optional<float> sample_synthetic_density(
        float query,
        const std::vector<float>& axis,
        const std::vector<std::array<float, 3>>& curves,
        std::size_t channel) {
        if (std::isnan(query) || axis.empty() || axis.size() != curves.size() || channel >= 3u) {
            return std::nullopt;
        }
        if (query <= axis.front()) {
            return curves.front()[channel];
        }
        if (query >= axis.back()) {
            return curves.back()[channel];
        }
        // Complete profiles supply a nondecreasing axis; do not rescan or repair it here.
        const auto upper = std::upper_bound(axis.begin(), axis.end(), query);
        if (upper == axis.begin() || upper == axis.end()) {
            return std::nullopt;
        }
        const std::size_t hi = static_cast<std::size_t>(upper - axis.begin());
        const std::size_t lo = hi - 1u;
        const float span = axis[hi] - axis[lo];
        const float t = span > 0.0f ? (query - axis[lo]) / span : 0.0f;
        return curves[lo][channel] + t * (curves[hi][channel] - curves[lo][channel]);
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

    Spektrafilm::DirReferenceOperator dir_reference_operator_for_sigma(
        float sigmaPixels,
        int radius) noexcept {
        if (!(sigmaPixels > 0.0f) || radius == 0) {
            return Spektrafilm::DirReferenceOperator::Identity;
        }
        return sigmaPixels >= 3.0f
                   ? Spektrafilm::DirReferenceOperator::SpektrafilmLargeYvvReflect
                   : Spektrafilm::DirReferenceOperator::SpektrafilmSmallFirReflect;
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

    struct DirComponentBuildInput {
        float sigmaPixels = 0.0f;
        float weight = 0.0f;
    };

    bool build_dir_iir_coefficients(
        float sigmaPixels,
        Spektrafilm::DirGaussianComponentPlan::IirCoefficients& out) noexcept {
        constexpr double kA1 = 2.44413;
        constexpr double kA2 = 1.4281;
        constexpr double kA3 = 0.422205;
        constexpr double kFeedforwardNumerator = 1.57825;
        const double q = 0.98711 * static_cast<double>(sigmaPixels) - 0.96330;
        const double q2 = q * q;
        const double q3 = q2 * q;
        out.normalizationDenominator =
            kFeedforwardNumerator + kA1 * q + kA2 * q2 + kA3 * q3;
        out.feedforwardNumerator = kFeedforwardNumerator;
        out.feedbackNumerators = {
            kA1 * q + 2.0 * kA2 * q2 + 3.0 * kA3 * q3,
            -(kA2 * q2 + 3.0 * kA3 * q3),
            kA3 * q3};
        if (!(out.normalizationDenominator > 0.0) ||
            !std::isfinite(out.normalizationDenominator)) {
            return false;
        }
        out.feedforward = out.feedforwardNumerator / out.normalizationDenominator;
        for (std::size_t coefficient = 0; coefficient < out.feedback.size(); ++coefficient) {
            out.feedback[coefficient] =
                out.feedbackNumerators[coefficient] / out.normalizationDenominator;
        }
        const double dc = out.feedforward + out.feedback[0] +
                          out.feedback[1] + out.feedback[2];
        return std::isfinite(out.feedforward) && out.feedforward > 0.0 &&
               std::all_of(
                   out.feedback.begin(),
                   out.feedback.end(),
                   [](double value) {
                       return std::isfinite(value);
                   }) &&
               std::abs(dc - 1.0) <= 32.0 * std::numeric_limits<double>::epsilon();
    }

    bool add_dir_component(
        Spektrafilm::DirFilterPlan& plan,
        DirComponentBuildInput input,
        Spektrafilm::DirScratchTier& targetScratchTier) noexcept {
        if (plan.componentCount >= Spektrafilm::DirFilterPlan::kMaxComponents) {
            return false;
        }
        Spektrafilm::DirGaussianComponentPlan& component =
            plan.components[static_cast<std::size_t>(plan.componentCount)];
        component.sigmaPixels = input.sigmaPixels;
        component.weight = input.weight;
        const double radiusValue =
            3.0 * static_cast<double>(component.sigmaPixels) + 0.5;
        if (!std::isfinite(radiusValue) ||
            radiusValue > static_cast<double>(std::numeric_limits<int>::max())) {
            return false;
        }
        component.radius = static_cast<int>(radiusValue);
        component.referenceOperator =
            dir_reference_operator_for_sigma(input.sigmaPixels, component.radius);
        if (component.referenceOperator ==
                Spektrafilm::DirReferenceOperator::SpektrafilmLargeYvvReflect &&
            !build_dir_iir_coefficients(component.sigmaPixels, component.iir)) {
            return false;
        }
        component.targetScratchTier = dir_target_scratch_tier_for_operator(component.referenceOperator);
        if (component.weight > 0.0f) {
            if (component.targetScratchTier == Spektrafilm::DirScratchTier::Tier1IChannels) {
                targetScratchTier = Spektrafilm::DirScratchTier::Tier1IChannels;
            } else if (targetScratchTier == Spektrafilm::DirScratchTier::Tier0) {
                targetScratchTier = Spektrafilm::DirScratchTier::Tier1F;
            }
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
        const char* domainField =
            cameraStage ? "camera_linear_domain" : "enlarger_linear_domain";
        const char* sampleField =
            cameraStage ? "camera_sample" : "enlarger_sample";
        const char* stageHashField =
            cameraStage ? "camera_hash" : "enlarger_hash";

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
                sampleDiagnostic)) {
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

#ifdef JUICER_ASSET_LOOKUP_TEST_HOOK
namespace AssetLookupTest {

    std::optional<float> sample_synthetic_density_for_test(
        float query,
        const std::vector<float>& axis,
        const std::vector<std::array<float, 3>>& curves,
        std::size_t channel) {
        return sample_synthetic_density(query, axis, curves, channel);
    }

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

    bool build_filtered_print_illuminant(
        const PrintRecipe& recipe,
        const std::array<float, 81>& sourceIlluminant,
        const CmyCcTriplet& cmyCc,
        std::array<float, 81>& out,
        std::string& diagnostic) {
        diagnostic.clear();
        std::array<std::array<float, 81>, 3> filters{};
        for (std::size_t sample = 0; sample < out.size(); ++sample) {
            const float wavelength = 380.0f + 5.0f * static_cast<float>(sample);
            const float y =
                0.5f * std::erf((wavelength - recipe.filters.dichroic.customEdgesNm[0]) /
                                recipe.filters.dichroic.customTransitionsNm[0]) +
                0.5f;
            const float mErf =
                wavelength <= 550.0f
                    ? -std::erf((wavelength - recipe.filters.dichroic.customEdgesNm[1]) /
                                recipe.filters.dichroic.customTransitionsNm[1])
                    : std::erf((wavelength - recipe.filters.dichroic.customEdgesNm[2]) /
                               recipe.filters.dichroic.customTransitionsNm[2]);
            const float c =
                -0.5f * std::erf((wavelength - recipe.filters.dichroic.customEdgesNm[3]) /
                                 recipe.filters.dichroic.customTransitionsNm[3]) +
                0.5f;
            filters[0][sample] = c;
            filters[1][sample] = 0.5f * mErf + 0.5f;
            filters[2][sample] = y;
        }
        const std::array<float, 3> transmittance{{std::pow(10.0f, -cmyCc.c / 100.0f),
                                                  std::pow(10.0f, -cmyCc.m / 100.0f),
                                                  std::pow(10.0f, -cmyCc.y / 100.0f)}};
        float energy = 0.0f;
        for (std::size_t sample = 0; sample < out.size(); ++sample) {
            float total = sourceIlluminant[sample];
            for (std::size_t channel = 0; channel < filters.size(); ++channel) {
                total *= 1.0f - (1.0f - filters[channel][sample]) *
                                    (1.0f - transmittance[channel]);
            }
            if (!(std::isfinite(total) && total >= 0.0f)) {
                diagnostic =
                    "MalformedRequiredResource phase=4B field=filtered_print_illuminant";
                return false;
            }
            out[sample] = total;
            energy += total;
        }
        if (!(std::isfinite(energy) && energy > 0.0f)) {
            diagnostic =
                "MalformedRequiredResource phase=4B field=filtered_print_illuminant";
            return false;
        }
        return true;
    }

    bool finish_synthetic_reference_recipes(
        RenderRecipe& recipe,
        const Spectral::FilmTcLut* filmTcLut,
        const std::array<float, 81>& filmIlluminant,
        const std::array<float, 81>* filteredMainIlluminant,
        std::string& diagnostic) {
        diagnostic.clear();
        if (!recipe.profileRoute.filmProfile ||
            recipe.filmDevelop.logExposure.empty() ||
            recipe.filmDevelop.logExposure.size() !=
                recipe.filmDevelop.authoredDensityCurves.size()) {
            diagnostic =
                "ResourceDescriptorMismatch phase=3B field=synthetic_reference_inputs";
            return false;
        }

        auto reconstruct = [&](float exposureEv, std::array<float, 3>& out) {
            out = {};
            float source = 0.0f;
            std::string exposureDiagnostic;
            if (!JuicerExposure::reference_source(
                    exposureEv, source, exposureDiagnostic)) {
                return false;
            }
            if (recipe.filmRaw.rgbToRawMethod == RgbToRawMethod::Mallett2019) {
                const Spectral::MallettBasis& basis = Spectral::context().mallettBasis;
                if (basis.rows != 81 || basis.cols != 3 || basis.data.size() != 243u) {
                    return false;
                }
                return JuicerExposure::mallett_reference_raw(
                    basis,
                    filmIlluminant,
                    recipe.filmRaw.finalSensitivity,
                    source,
                    recipe.filmRaw.mallettGreenMidgrayScale,
                    out,
                    exposureDiagnostic);
            } else if (
                recipe.filmRaw.rgbToRawMethod == RgbToRawMethod::Hanatos2025 ||
                recipe.filmRaw.rgbToRawMethod == RgbToRawMethod::Arctic2026beta04) {
                if (!filmTcLut) {
                    return false;
                }
                const auto inputMatrices = JuicerColor::input_matrices(Spectral::InputColorSpace::SRGB_Rec709);
                const auto sourceXyz = JuicerColor::linear_srgb_to_xyz({source, source, source});
                const auto projectedXyz = JuicerColor::adapt_cat16(
                    sourceXyz,
                    Spectral::ChromaticAdaptationWhites{
                        inputMatrices.d65WhiteXYZ.data(),
                        recipe.filmRaw.projectionWhiteXYZ.data()});
                std::string samplingDiagnostic;
                if (!Spectral::sample_film_tc_lut(
                        *filmTcLut,
                        {projectedXyz[0], projectedXyz[1], projectedXyz[2]},
                        out,
                        samplingDiagnostic)) {
                    return false;
                }
            } else {
                return false;
            }
            return std::all_of(out.begin(), out.end(), [](float raw) {
                return std::isfinite(raw) && raw >= 0.0f;
            });
        };

        SyntheticFilmReferenceRecipe reference{};
        if (!reconstruct(0.0f, reference.baselineRawRgb) ||
            !reconstruct(
                recipe.filmRaw.manualExposureCompensationEv,
                reference.compensatedRawRgb)) {
            diagnostic =
                "MalformedRequiredResource phase=3B field=synthetic_reference_raw";
            return false;
        }
        if (recipe.filmRaw.autoExposureEnabled) {
            const float mean =
                (reference.baselineRawRgb[0] + reference.baselineRawRgb[1] +
                 reference.baselineRawRgb[2]) /
                3.0f;
            if (!(std::isfinite(mean) && mean > 0.0f)) {
                diagnostic =
                    "MalformedRequiredResource phase=3B field=synthetic_reference_meter";
                return false;
            }
            reference.baselineMeterScale = 1.0f / mean;
        }
        for (std::size_t channel = 0; channel < 3u; ++channel) {
            const float gamma = recipe.filmDevelop.densityCurveGamma[channel];
            if (!(std::isfinite(gamma) && gamma > 0.0f)) {
                diagnostic =
                    "ResourceDescriptorMismatch phase=3B field=film_density_curve_gamma";
                return false;
            }
            const float baselineLog = std::log10(
                reference.baselineRawRgb[channel] * reference.baselineMeterScale +
                1.0e-10f);
            const float compensatedLog = std::log10(
                reference.compensatedRawRgb[channel] * reference.baselineMeterScale +
                1.0e-10f);
            const auto baselineDensity = sample_synthetic_density(
                baselineLog * gamma,
                recipe.filmDevelop.logExposure,
                recipe.filmDevelop.authoredDensityCurves,
                channel);
            const auto compensatedDensity = sample_synthetic_density(
                compensatedLog * gamma,
                recipe.filmDevelop.logExposure,
                recipe.filmDevelop.authoredDensityCurves,
                channel);
            if (!baselineDensity || !compensatedDensity) {
                diagnostic =
                    "ResourceDescriptorMismatch phase=3B field=synthetic_reference_density_lookup";
                return false;
            }
            reference.baselineDensityCmy[channel] = *baselineDensity;
            reference.compensatedDensityCmy[channel] = *compensatedDensity;
        }
        reference.hash = Hash::kFnvOffset;
        Hash::hash_bytes_update(
            reference.hash,
            reference.baselineRawRgb.data(),
            sizeof(reference.baselineRawRgb));
        Hash::hash_bytes_update(
            reference.hash,
            reference.compensatedRawRgb.data(),
            sizeof(reference.compensatedRawRgb));
        Hash::hash_bytes_update(
            reference.hash,
            reference.baselineDensityCmy.data(),
            sizeof(reference.baselineDensityCmy));
        Hash::hash_bytes_update(
            reference.hash,
            reference.compensatedDensityCmy.data(),
            sizeof(reference.compensatedDensityCmy));
        hash_value(reference.hash, reference.baselineMeterScale);
        recipe.scannerOutput.syntheticFilmReference = reference;
        recipe.scannerOutput.hash = hash_scanner_output_recipe(recipe.scannerOutput);

        if (scan_route_is_print(recipe.profileRoute.scanRoute)) {
            if (!recipe.profileRoute.printProfile || !filteredMainIlluminant) {
                diagnostic =
                    "MissingRequiredResource phase=4A field=filtered_print_illuminant";
                return false;
            }
            const Profiles::FilmProfileSamples& film =
                recipe.profileRoute.filmProfile->data;
            const Profiles::PrintProfileSamples& print =
                recipe.profileRoute.printProfile->data;
            auto derive_print_raw = [&](
                                        const std::array<float, 3>& densityCmy,
                                        std::array<float, 3>& out) {
                std::array<double, 3> accumulated{};
                for (std::size_t sample = 0; sample < filteredMainIlluminant->size();
                     ++sample) {
                    double density = static_cast<double>(film.baseDensity[sample]);
                    for (std::size_t channel = 0; channel < 3u; ++channel) {
                        density += static_cast<double>(densityCmy[channel]) *
                                   static_cast<double>(film.channelDensity[sample][channel]);
                    }
                    const double light =
                        std::pow(10.0, -density) *
                        static_cast<double>((*filteredMainIlluminant)[sample]);
                    if (std::isnan(light)) {
                        continue;
                    }
                    for (std::size_t channel = 0; channel < 3u; ++channel) {
                        const float sensitivity = print.linearSensitivity[sample][channel];
                        if (std::isfinite(sensitivity)) {
                            accumulated[channel] +=
                                light * static_cast<double>(sensitivity);
                        }
                    }
                }
                for (std::size_t channel = 0; channel < 3u; ++channel) {
                    out[channel] = static_cast<float>(accumulated[channel]);
                    if (!(std::isfinite(out[channel]) && out[channel] >= 0.0f)) {
                        return false;
                    }
                }
                return true;
            };
            auto factor = [](const std::array<float, 3>& raw) {
                double meanLog = 0.0;
                for (float value : raw) {
                    meanLog += std::log(std::max(1.0e-10, static_cast<double>(value)));
                }
                return static_cast<float>(std::exp(-meanLog / 3.0));
            };
            PrintBalanceRecipe balance{};
            if (!derive_print_raw(
                    reference.baselineDensityCmy,
                    balance.baselinePrintRawRgb) ||
                !derive_print_raw(
                    reference.compensatedDensityCmy,
                    balance.compensatedPrintRawRgb)) {
                diagnostic =
                    "MalformedRequiredResource phase=4A field=print_balance_raw";
                return false;
            }
            balance.factorMidgray = factor(balance.baselinePrintRawRgb);
            balance.factorMidgrayComp = factor(balance.compensatedPrintRawRgb);
            balance.normalizer = print_exposure_normalizer(
                recipe.print.exposure.normalizationMode,
                balance.factorMidgray,
                balance.factorMidgrayComp);
            balance.filteredMainIlluminantHash =
                Hash::hash_float_span(*filteredMainIlluminant);
            balance.hash = Hash::hash_uint64_values(
                {reference.hash,
                 balance.filteredMainIlluminantHash,
                 Hash::hash_float_span(balance.baselinePrintRawRgb),
                 Hash::hash_float_span(balance.compensatedPrintRawRgb),
                 static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(
                     balance.factorMidgray)),
                 static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(
                     balance.factorMidgrayComp)),
                 static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(
                     balance.normalizer))});
            if (!(std::isfinite(balance.factorMidgray) &&
                  balance.factorMidgray > 0.0f &&
                  std::isfinite(balance.factorMidgrayComp) &&
                  balance.factorMidgrayComp > 0.0f &&
                  std::isfinite(balance.normalizer) && balance.normalizer > 0.0f &&
                  balance.filteredMainIlluminantHash != 0 && balance.hash != 0)) {
                diagnostic =
                    "MalformedRequiredResource phase=4A field=print_balance";
                return false;
            }
            recipe.print.balance = balance;
            recipe.print.hash = hash_print_recipe(recipe.print);
        }

        recipe.hash = hash_final_render_recipe(recipe);
        return reference.hash != 0 && recipe.scannerOutput.hash != 0 &&
               recipe.hash != 0;
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

    void resolve_spatial_dir_scratch(SpatialDirDescriptor& descriptor) noexcept {
        configure_cached_log_raw_descriptor(descriptor);
        descriptor.targetPlaneRoles = dir_target_plane_roles_for_descriptor(descriptor);
        descriptor.scratchTier = dir_build_scratch_tier_for_descriptor(descriptor);
        descriptor.planeRoles = dir_build_plane_roles_for_descriptor(descriptor);
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
        out.renderExtent = renderExtent;
        out.fullFrameExtent = fullFrameExtent;
        out.filterDomainExtent = fullFrameExtent;
        out.traceRouteLabel = traceRouteLabel;
        const float coreSigmaPixels = recipe.diffusionSizeUm / pixelSizeUm;
        const float coreWeight = 1.0f - recipe.diffusionTailWeight;
        if (!(std::isfinite(coreSigmaPixels) && coreSigmaPixels > 0.0f) ||
            !finite_nonnegative(coreWeight)) {
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
                DirComponentBuildInput{coreSigmaPixels, coreWeight},
                out.targetScratchTier)) {
            return false;
        }
        if (recipe.diffusionTailWeight > 0.0f) {
            const float tailSigmaPixels = recipe.diffusionTailUm / pixelSizeUm;
            if (!(std::isfinite(tailSigmaPixels) && tailSigmaPixels >= 0.0f)) {
                return false;
            }
            const std::size_t tailComponentCount = SpatialDirDescriptor::kExponentialSigmaRatios.size();
            const float tailAmplitudeSum =
                SpatialDirDescriptor::kExponentialAmplitudes[0] +
                SpatialDirDescriptor::kExponentialAmplitudes[1] +
                SpatialDirDescriptor::kExponentialAmplitudes[2];
            if (!(std::isfinite(tailAmplitudeSum) && tailAmplitudeSum > 0.0f)) {
                return false;
            }
            for (std::size_t component = 0; component < tailComponentCount; ++component) {
                const float sigmaRatio = SpatialDirDescriptor::kExponentialSigmaRatios[component];
                const float amplitude =
                    SpatialDirDescriptor::kExponentialAmplitudes[component] /
                    tailAmplitudeSum;
                const float componentSigmaPixels = tailSigmaPixels * sigmaRatio;
                const float componentWeight = recipe.diffusionTailWeight * amplitude;
                if (!(std::isfinite(componentSigmaPixels) &&
                      componentSigmaPixels >= 0.0f) ||
                    !finite_nonnegative(componentWeight)) {
                    return false;
                }
                if (!add_dir_component(
                        out.filterPlan,
                        DirComponentBuildInput{
                            componentSigmaPixels,
                            componentWeight},
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
        resolve_spatial_dir_scratch(out);

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
