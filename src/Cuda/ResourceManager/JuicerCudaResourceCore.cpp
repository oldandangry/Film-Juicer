#include "JuicerCudaResourceCore.h"

#include <limits>

namespace JuicerCuda::ResourceManager {

    namespace {
        bool dir_roles_are_empty(const Spektrafilm::DirScratchPlaneRoles& roles) noexcept {
            return roles.rawCorrectionPlanes == 0 &&
                   roles.filteredCorrectionPlanes == 0 &&
                   roles.filterTempPlanes == 0 &&
                   roles.cachedLogRawPlanes == 0;
        }

        bool dir_layout_is_valid(const ActiveDirScratch::Layout& layout) noexcept {
            const auto& roles = layout.roles;
            if (layout.tier == Spektrafilm::DirScratchTier::Tier0) {
                return dir_roles_are_empty(roles);
            }
            return Spektrafilm::spatial_dir_roles_match_tier(layout.tier, roles);
        }
    } // namespace

    std::optional<ActiveDirScratch> ActiveDirScratch::create(
        std::uint64_t descriptorHash, const Layout& source, const Layout& target) noexcept {
        if (descriptorHash == 0 || source.tier == Spektrafilm::DirScratchTier::Tier0 ||
            !dir_layout_is_valid(source) || !dir_layout_is_valid(target)) {
            return std::nullopt;
        }
        ActiveDirScratch active;
        active._descriptorHash = descriptorHash;
        active._source = source;
        active._target = target;
        return active;
    }

    namespace {
        bool scratch_request_structure_is_valid(const ScratchRequestDescriptor& descriptor) noexcept {
            if (descriptor.needGateTransmittance && (descriptor.gateWidth <= 0 || descriptor.gateHeight <= 0 ||
                                                     static_cast<std::uint64_t>(descriptor.gateWidth) * descriptor.gateHeight >
                                                         std::numeric_limits<std::size_t>::max() / sizeof(float))) {
                return false;
            }
            if (!descriptor.has_any_family()) {
                return false;
            }
            if (descriptor.requestedWidth <= 0 || descriptor.requestedHeight <= 0) {
                return false;
            }
            if (!descriptor.needSpatialDir) {
                if (descriptor.spatialDirDescriptorHash != 0 ||
                    descriptor.spatialDirScratchTier != Spektrafilm::DirScratchTier::Tier0 ||
                    !dir_roles_are_empty(descriptor.spatialDirPlaneRoles) ||
                    descriptor.spatialDirTargetScratchTier != Spektrafilm::DirScratchTier::Tier0 ||
                    !dir_roles_are_empty(descriptor.spatialDirTargetPlaneRoles)) {
                    return false;
                }
            } else {
                if (descriptor.spatialDirDescriptorHash == 0 ||
                    descriptor.spatialDirScratchTier == Spektrafilm::DirScratchTier::Tier0 ||
                    !dir_layout_is_valid({descriptor.spatialDirScratchTier, descriptor.spatialDirPlaneRoles}) ||
                    !dir_layout_is_valid({descriptor.spatialDirTargetScratchTier, descriptor.spatialDirTargetPlaneRoles})) {
                    return false;
                }
            }
            if (!descriptor.needOptics &&
                (descriptor.needBlurred ||
                 descriptor.aliasScannerRgbFromSpatialDirFiltered ||
                 descriptor.needAux ||
                 descriptor.needGrainFrameUniforms ||
                 descriptor.needGrainLayerWork ||
                 descriptor.needGrainShared ||
                 descriptor.needGateTransmittance || descriptor.needFilmDustTransmittance)) {
                return false;
            }
            const bool spatialDirNeedsSharedTmp =
                descriptor.needSpatialDir &&
                descriptor.spatialDirPlaneRoles.filterTempPlanes > 0;
            if ((spatialDirNeedsSharedTmp && !descriptor.needSharedTmp) ||
                (descriptor.needSharedTmp &&
                 !descriptor.needOptics &&
                 !spatialDirNeedsSharedTmp)) {
                return false;
            }
            if (descriptor.aliasScannerRgbFromSpatialDirFiltered &&
                (!descriptor.needOptics ||
                 !descriptor.needSpatialDir ||
                 descriptor.spatialDirPlaneRoles.filteredCorrectionPlanes != 3 ||
                 (descriptor.spatialDirTargetPlaneRoles.cachedLogRawPlanes != 0 &&
                  descriptor.spatialDirTargetPlaneRoles.cachedLogRawPlanes != 2 &&
                  descriptor.spatialDirTargetPlaneRoles.cachedLogRawPlanes != 3))) {
                return false;
            }
            return true;
        }
    } // namespace

    bool scratch_request_descriptor_is_valid(const ScratchRequestDescriptor& descriptor) noexcept {
        if (!descriptor.has_any_family()) {
            return descriptor.spatialDirDescriptorHash == 0 &&
                   descriptor.spatialDirScratchTier == Spektrafilm::DirScratchTier::Tier0 &&
                   dir_roles_are_empty(descriptor.spatialDirPlaneRoles) &&
                   descriptor.spatialDirTargetScratchTier == Spektrafilm::DirScratchTier::Tier0 &&
                   dir_roles_are_empty(descriptor.spatialDirTargetPlaneRoles) &&
                   descriptor.requestedWidth == 0 && descriptor.requestedHeight == 0 &&
                   !descriptor.needBlurred && !descriptor.aliasScannerRgbFromSpatialDirFiltered &&
                   !descriptor.needAux && !descriptor.needSharedTmp && !descriptor.needGrainFrameUniforms &&
                   !descriptor.needGrainLayerWork && !descriptor.needGrainShared &&
                   !descriptor.needGateTransmittance && !descriptor.needFilmDustTransmittance &&
                   descriptor.gateWidth == 0 && descriptor.gateHeight == 0 && descriptor.generation == 0;
        }
        return scratch_request_structure_is_valid(descriptor) &&
               descriptor.generation == hash_scratch_request_descriptor(descriptor);
    }

    std::uint64_t hash_scratch_request_descriptor(const ScratchRequestDescriptor& descriptor) noexcept {
        constexpr std::uint64_t kFnvOffset = 1469598103934665603ull;
        constexpr std::uint64_t kFnvPrime = 1099511628211ull;
        std::uint64_t hash = kFnvOffset;
        auto mix = [&](std::uint64_t value) {
            hash ^= value;
            hash *= kFnvPrime;
        };
        mix(descriptor.needOptics ? 1ull : 0ull);
        mix(descriptor.needSpatialDir ? 1ull : 0ull);
        mix(descriptor.spatialDirDescriptorHash);
        mix(static_cast<std::uint64_t>(descriptor.spatialDirScratchTier));
        mix(static_cast<std::uint64_t>(descriptor.spatialDirTargetScratchTier));
        mix(static_cast<std::uint64_t>(descriptor.spatialDirPlaneRoles.rawCorrectionPlanes));
        mix(static_cast<std::uint64_t>(descriptor.spatialDirPlaneRoles.filteredCorrectionPlanes));
        mix(static_cast<std::uint64_t>(descriptor.spatialDirPlaneRoles.filterTempPlanes));
        mix(static_cast<std::uint64_t>(descriptor.spatialDirPlaneRoles.cachedLogRawPlanes));
        mix(static_cast<std::uint64_t>(descriptor.spatialDirTargetPlaneRoles.rawCorrectionPlanes));
        mix(static_cast<std::uint64_t>(descriptor.spatialDirTargetPlaneRoles.filteredCorrectionPlanes));
        mix(static_cast<std::uint64_t>(descriptor.spatialDirTargetPlaneRoles.filterTempPlanes));
        mix(static_cast<std::uint64_t>(descriptor.spatialDirTargetPlaneRoles.cachedLogRawPlanes));
        mix(static_cast<std::uint64_t>(descriptor.requestedWidth));
        mix(static_cast<std::uint64_t>(descriptor.requestedHeight));
        mix(descriptor.needBlurred ? 1ull : 0ull);
        mix(descriptor.aliasScannerRgbFromSpatialDirFiltered ? 1ull : 0ull);
        mix(descriptor.needAux ? 1ull : 0ull);
        mix(descriptor.needSharedTmp ? 1ull : 0ull);
        mix(descriptor.needGrainFrameUniforms ? 1ull : 0ull);
        mix(descriptor.needGrainLayerWork ? 1ull : 0ull);
        mix(descriptor.needGrainShared ? 1ull : 0ull);
        mix(descriptor.needGateTransmittance ? 1ull : 0ull);
        mix(descriptor.needFilmDustTransmittance ? 1ull : 0ull);
        mix(static_cast<std::uint64_t>(descriptor.gateWidth));
        mix(static_cast<std::uint64_t>(descriptor.gateHeight));
        return hash;
    }

    std::optional<ScratchRequestDescriptor> make_scratch_request_descriptor(
        const ScratchRequestExtent& extent,
        const DirScratchRequest& spatialDir,
        const ScratchRequestAttachments& attachments) noexcept {
        ScratchRequestDescriptor descriptor{};
        descriptor.needOptics = attachments.needOptics;
        if (const auto* active = std::get_if<ActiveDirScratch>(&spatialDir)) {
            descriptor.needSpatialDir = true;
            descriptor.spatialDirDescriptorHash = active->_descriptorHash;
            descriptor.spatialDirScratchTier = active->_source.tier;
            descriptor.spatialDirPlaneRoles = active->_source.roles;
            descriptor.spatialDirTargetScratchTier = active->_target.tier;
            descriptor.spatialDirTargetPlaneRoles = active->_target.roles;
        }
        descriptor.requestedWidth = extent.requestedWidth;
        descriptor.requestedHeight = extent.requestedHeight;
        descriptor.needBlurred = attachments.needBlurred;
        descriptor.aliasScannerRgbFromSpatialDirFiltered = attachments.aliasScannerRgbFromSpatialDirFiltered;
        descriptor.needAux = attachments.needAux;
        descriptor.needSharedTmp = attachments.needSharedTmp;
        descriptor.needGrainFrameUniforms = attachments.needGrainFrameUniforms;
        descriptor.needGrainLayerWork = attachments.needGrainLayerWork;
        descriptor.needGrainShared = attachments.needGrainShared;
        descriptor.needGateTransmittance = attachments.needGateTransmittance;
        descriptor.needFilmDustTransmittance = attachments.needFilmDustTransmittance;
        descriptor.gateWidth = descriptor.needGateTransmittance ? attachments.gateWidth : 0;
        descriptor.gateHeight = descriptor.needGateTransmittance ? attachments.gateHeight : 0;
        if (!scratch_request_structure_is_valid(descriptor)) {
            return std::nullopt;
        }
        descriptor.generation = hash_scratch_request_descriptor(descriptor);
        return descriptor;
    }

    WorkspaceLayout::WorkspaceLayout(ScratchRequestDescriptor allocation, ScannerStorage scanner, GrainStorage grain, Spektrafilm::VisualGrainScratchShape shape) noexcept
        : _allocation(allocation), _scanner(scanner), _grain(grain), _grainShape(shape) {
    }

    std::optional<WorkspaceLayout> WorkspaceLayout::create(
        const ScratchRequestExtent& extent, const DirScratchRequest& spatialDir, const ScratchRequestAttachments& attachments, Spektrafilm::VisualGrainScratchShape grainShape) noexcept {
        if (attachments.needAux || attachments.needGrainFrameUniforms ||
            attachments.needGrainLayerWork || attachments.needGrainShared) {
            return std::nullopt;
        }
        bool layers = false;
        bool shared = false;
        switch (grainShape) {
            case Spektrafilm::VisualGrainScratchShape::None:
            case Spektrafilm::VisualGrainScratchShape::Streamed:
                break;
            case Spektrafilm::VisualGrainScratchShape::StreamedShared:
                shared = true;
                break;
            case Spektrafilm::VisualGrainScratchShape::StreamedLayers:
                layers = true;
                break;
            case Spektrafilm::VisualGrainScratchShape::StreamedLayersShared:
                layers = true;
                shared = true;
                break;
            default:
                return std::nullopt;
        }
        const auto* dir = std::get_if<ActiveDirScratch>(&spatialDir);
        if (attachments.aliasScannerRgbFromSpatialDirFiltered &&
            (!dir || dir->_target.roles.filteredCorrectionPlanes != 3)) {
            return std::nullopt;
        }
        const bool grainActive = grainShape != Spektrafilm::VisualGrainScratchShape::None;
        const bool reuseRaw = grainActive && attachments.aliasScannerRgbFromSpatialDirFiltered && dir &&
                              dir->_source.roles.rawCorrectionPlanes == 3 && dir->_target.roles.rawCorrectionPlanes == 3;
        auto physical = attachments;
        if (grainActive) {
            physical.needOptics = true;
            physical.needSharedTmp = true;
            physical.needBlurred = true;
            physical.needGrainFrameUniforms = true;
            physical.needAux = !reuseRaw;
            physical.needGrainLayerWork = layers && !reuseRaw;
            physical.needGrainShared = shared && !reuseRaw;
        }
        const auto allocation = make_scratch_request_descriptor(extent, spatialDir, physical);
        if (!allocation) {
            return std::nullopt;
        }
        const auto scanner = !allocation->needOptics                             ? ScannerStorage::Absent
                             : allocation->aliasScannerRgbFromSpatialDirFiltered ? ScannerStorage::DirFiltered
                                                                                 : ScannerStorage::Dedicated;
        const auto grain = !grainActive ? GrainStorage::Absent : reuseRaw ? GrainStorage::DirRaw
                                                                          : GrainStorage::Dedicated;
        return WorkspaceLayout(*allocation, scanner, grain, grainShape);
    }

    const ScratchRequestDescriptor& WorkspaceLayout::allocation() const noexcept {
        return _allocation;
    }

    ScannerStorage WorkspaceLayout::scanner_storage() const noexcept {
        return _scanner;
    }

    GrainStorage WorkspaceLayout::grain_storage() const noexcept {
        return _grain;
    }

    Spektrafilm::VisualGrainScratchShape WorkspaceLayout::grain_shape() const noexcept {
        return _grainShape;
    }

    bool WorkspaceLayout::grain_uses_layers() const noexcept {
        return _grainShape == Spektrafilm::VisualGrainScratchShape::StreamedLayers ||
               _grainShape == Spektrafilm::VisualGrainScratchShape::StreamedLayersShared;
    }

    bool WorkspaceLayout::grain_uses_shared() const noexcept {
        return _grainShape == Spektrafilm::VisualGrainScratchShape::StreamedShared ||
               _grainShape == Spektrafilm::VisualGrainScratchShape::StreamedLayersShared;
    }

    std::optional<WorkspaceLayout> WorkspaceLayout::with_dir_layout(
        const ActiveDirScratch::Layout& source, const ActiveDirScratch::Layout& target) const noexcept {
        const auto dir = ActiveDirScratch::create(_allocation.spatialDirDescriptorHash, source, target);
        if (!dir) {
            return std::nullopt;
        }
        ScratchRequestAttachments attachments;
        attachments.needOptics = _allocation.needOptics;
        attachments.needBlurred = _allocation.needBlurred;
        attachments.aliasScannerRgbFromSpatialDirFiltered = _allocation.aliasScannerRgbFromSpatialDirFiltered;
        attachments.needSharedTmp = _allocation.needSharedTmp;
        attachments.needGateTransmittance = _allocation.needGateTransmittance;
        attachments.needFilmDustTransmittance = _allocation.needFilmDustTransmittance;
        attachments.gateWidth = _allocation.gateWidth;
        attachments.gateHeight = _allocation.gateHeight;
        return create({_allocation.requestedWidth, _allocation.requestedHeight}, *dir, attachments, _grainShape);
    }

    bool WorkspaceLayout::operator==(const WorkspaceLayout& other) const noexcept {
        const auto& a = _allocation;
        const auto& b = other._allocation;
        const auto rolesEqual = [](const auto& x, const auto& y) {
            return x.rawCorrectionPlanes == y.rawCorrectionPlanes && x.filteredCorrectionPlanes == y.filteredCorrectionPlanes &&
                   x.filterTempPlanes == y.filterTempPlanes && x.cachedLogRawPlanes == y.cachedLogRawPlanes;
        };
        return _scanner == other._scanner && _grain == other._grain && _grainShape == other._grainShape &&
               a.needOptics == b.needOptics && a.needSpatialDir == b.needSpatialDir &&
               a.spatialDirDescriptorHash == b.spatialDirDescriptorHash && a.spatialDirScratchTier == b.spatialDirScratchTier &&
               a.spatialDirTargetScratchTier == b.spatialDirTargetScratchTier &&
               rolesEqual(a.spatialDirPlaneRoles, b.spatialDirPlaneRoles) && rolesEqual(a.spatialDirTargetPlaneRoles, b.spatialDirTargetPlaneRoles) &&
               a.requestedWidth == b.requestedWidth && a.requestedHeight == b.requestedHeight &&
               a.needBlurred == b.needBlurred && a.aliasScannerRgbFromSpatialDirFiltered == b.aliasScannerRgbFromSpatialDirFiltered &&
               a.needAux == b.needAux && a.needSharedTmp == b.needSharedTmp && a.needGrainFrameUniforms == b.needGrainFrameUniforms &&
               a.needGrainLayerWork == b.needGrainLayerWork && a.needGrainShared == b.needGrainShared &&
               a.needGateTransmittance == b.needGateTransmittance && a.needFilmDustTransmittance == b.needFilmDustTransmittance &&
               a.gateWidth == b.gateWidth && a.gateHeight == b.gateHeight;
    }

} // namespace JuicerCuda::ResourceManager
