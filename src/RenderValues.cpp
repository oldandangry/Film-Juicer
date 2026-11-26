#include "RenderValues.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <iterator>

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
            plan.targetScratchTier = plan.weight > 0.0f ? Spektrafilm::DirScratchTier::Tier2 : dir_target_scratch_tier_for_operator(plan.referenceOperator);
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


} // namespace


namespace Spektrafilm {

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

    void resolve_spatial_dir_scratch(SpatialDirDescriptor& descriptor) noexcept {
        configure_cached_log_raw_descriptor(descriptor);
        descriptor.targetPlaneRoles = dir_target_plane_roles_for_descriptor(descriptor);
        descriptor.scratchTier = dir_build_scratch_tier_for_descriptor(descriptor);
        descriptor.planeRoles = dir_build_plane_roles_for_descriptor(descriptor);
    }

} // namespace Spektrafilm
