#include "RenderValues.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <iterator>
#include <string>

#include "RustDirBridge.h"
#include "Hash.h"

namespace {

    template <typename T>
    void hash_value(std::uint64_t& hash, const T& value) {
        Hash::hash_bytes_update(hash, &value, sizeof(value));
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


} // namespace


namespace Spektrafilm {


    void resolve_spatial_dir_scratch(SpatialDirDescriptor& descriptor) noexcept {
        configure_cached_log_raw_descriptor(descriptor);
        descriptor.targetPlaneRoles = dir_target_plane_roles_for_descriptor(descriptor);
        descriptor.scratchTier = dir_build_scratch_tier_for_descriptor(descriptor);
        descriptor.planeRoles = dir_build_plane_roles_for_descriptor(descriptor);
    }

    namespace {
        bool empty_dir_frame_math(const FjDirFrameMath& math) {
            const auto zeroFloat = [](float value) {
                return std::bit_cast<std::uint32_t>(value) == 0;
            };
            const auto zeroDouble = [](double value) {
                return std::bit_cast<std::uint64_t>(value) == 0;
            };
            const auto zeroExtent = [](const FjExtent& value) {
                return value.x == 0 && value.y == 0 && value.width == 0 && value.height == 0;
            };
            if (math.present != 0 || math.component_count != 0 || math.recipe_hash != 0 ||
                !zeroExtent(math.render_extent) || !zeroExtent(math.full_frame_extent) ||
                !zeroExtent(math.filter_domain_extent)) {
                return false;
            }
            for (const FjDirGaussian& component : math.components) {
                if (!zeroFloat(component.sigma_px) || !zeroFloat(component.weight) || component.radius != 0 ||
                    component.reference_operator != 0 || !zeroDouble(component.feedforward) ||
                    !std::all_of(std::begin(component.feedback), std::end(component.feedback), zeroDouble) ||
                    !zeroDouble(component.normalization_denominator) ||
                    !zeroDouble(component.feedforward_numerator) ||
                    !std::all_of(std::begin(component.feedback_numerators), std::end(component.feedback_numerators), zeroDouble) ||
                    !zeroDouble(component.boundary_truncation_accuracy) ||
                    !zeroDouble(component.boundary_certification_tolerance) ||
                    component.boundary_derivation_version != 0) {
                    return false;
                }
            }
            return true;
        }

        bool bind_spatial_dir_math(
            const FjDirFrameMath& math,
            const char* traceRouteLabel,
            SpatialDirDescriptor& out) {
            if (math.present != 1 ||
                math.component_count > out.filterPlan.components.size() || !traceRouteLabel) {
                return false;
            }
            out = SpatialDirDescriptor{};
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
            return true;
        }
    } // namespace

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
        if (!bind_spatial_dir_math(math, traceRouteLabel, out)) {
            return false;
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

    bool compose_spatial_dir_descriptor(
        const FjDirFrameMath& math, float pixelSizeUm, const char* traceRouteLabel, SpatialDirDescriptor& out) {
        out = SpatialDirDescriptor{};
        if (math.present == 0) {
            return empty_dir_frame_math(math);
        }
        return math.component_count != 0 && bind_spatial_dir_math(math, traceRouteLabel, out) &&
               finish_spatial_dir_descriptor(pixelSizeUm, out);
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
