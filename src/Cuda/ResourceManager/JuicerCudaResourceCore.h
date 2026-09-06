// Cuda/ResourceManager/JuicerCudaResourceCore.h
//
// Shared types and policy helpers for CUDA resource ownership and submission.
#pragma once

#include <cstdint>
#include <cstddef>
#include <functional>
#include <limits>

#include "../../RenderRecipe.h"

namespace JuicerCuda {
    namespace ResourceManager {

        struct InstanceToken {
            std::uint64_t value = 0;
        };

        struct FrameToken {
            std::uint64_t value = 0;
        };

        struct DeviceContextKey {
            int deviceId = -1;
            void* contextOpaque = nullptr;

            bool operator==(const DeviceContextKey& other) const noexcept {
                return deviceId == other.deviceId && contextOpaque == other.contextOpaque;
            }
        };

        struct DeviceContextKeyHash {
            std::size_t operator()(const DeviceContextKey& key) const noexcept {
                const std::uintptr_t contextBits = reinterpret_cast<std::uintptr_t>(key.contextOpaque);
                const std::size_t hDevice = std::hash<int>{}(key.deviceId);
                const std::size_t hContext = std::hash<std::uintptr_t>{}(contextBits);
                return hDevice ^ (hContext + 0x9e3779b9u + (hDevice << 6u) + (hDevice >> 2u));
            }
        };

        struct KeyDigests {
            std::uint64_t uploadCoreHash = 0;
            std::uint64_t dirHash = 0;
            std::uint64_t scannerHash = 0;
            std::uint64_t autoExposureHash = 0;
        };

        struct ScratchRequestFamilies {
            bool needOptics = false;
            bool needSpatialDir = false;
        };

        struct ScratchRequestExtent {
            int requestedWidth = 0;
            int requestedHeight = 0;
        };

        struct ScratchRequestAttachments {
            bool needBlurred = false;
            bool aliasScannerRgbFromSpatialDirFiltered = false;
            bool needAux = false;
            bool needSharedTmp = false;
            bool needGrainFrameUniforms = false;
            bool needGrainLayerWork = false;
            bool needGrainShared = false;
            bool needGateTransmittance = false;
            bool needFilmDustTransmittance = false;
            int gateWidth = 0;
            int gateHeight = 0;
        };

        struct ScratchRequestBuildRequest {
            ScratchRequestFamilies families{};
            ScratchRequestExtent extent{};
            ScratchRequestAttachments attachments{};
            std::uint64_t spatialDirDescriptorHash = 0;
            Spektrafilm::DirScratchTier spatialDirScratchTier = Spektrafilm::DirScratchTier::Tier0;
            Spektrafilm::DirScratchPlaneRoles spatialDirPlaneRoles{};
            Spektrafilm::DirScratchTier spatialDirTargetScratchTier = Spektrafilm::DirScratchTier::Tier0;
            Spektrafilm::DirScratchPlaneRoles spatialDirTargetPlaneRoles{};
        };

        struct ScratchRequestDescriptor {
            bool needOptics = false;
            bool needSpatialDir = false;
            std::uint64_t spatialDirDescriptorHash = 0;
            Spektrafilm::DirScratchTier spatialDirScratchTier = Spektrafilm::DirScratchTier::Tier0;
            Spektrafilm::DirScratchPlaneRoles spatialDirPlaneRoles{};
            Spektrafilm::DirScratchTier spatialDirTargetScratchTier = Spektrafilm::DirScratchTier::Tier0;
            Spektrafilm::DirScratchPlaneRoles spatialDirTargetPlaneRoles{};
            int requestedWidth = 0;
            int requestedHeight = 0;
            bool needBlurred = false;
            bool aliasScannerRgbFromSpatialDirFiltered = false;
            bool needAux = false;
            bool needSharedTmp = false;
            bool needGrainFrameUniforms = false;
            bool needGrainLayerWork = false;
            bool needGrainShared = false;
            bool needGateTransmittance = false;
            bool needFilmDustTransmittance = false;
            int gateWidth = 0;
            int gateHeight = 0;
            std::uint64_t generation = 0;

            bool has_any_family() const noexcept {
                return needOptics || needSpatialDir;
            }
        };

        inline bool scratch_request_descriptor_is_valid(const ScratchRequestDescriptor& descriptor) noexcept {
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
                    descriptor.spatialDirPlaneRoles.total_float_planes() != 0 ||
                    descriptor.spatialDirTargetScratchTier != Spektrafilm::DirScratchTier::Tier0 ||
                    descriptor.spatialDirTargetPlaneRoles.total_float_planes() != 0) {
                    return false;
                }
            } else {
                if (descriptor.spatialDirDescriptorHash == 0 ||
                    descriptor.spatialDirScratchTier == Spektrafilm::DirScratchTier::Tier0 ||
                    !spatial_dir_roles_match_tier(
                        descriptor.spatialDirScratchTier,
                        descriptor.spatialDirPlaneRoles) ||
                    !spatial_dir_roles_match_tier(
                        descriptor.spatialDirTargetScratchTier,
                        descriptor.spatialDirTargetPlaneRoles)) {
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

        inline std::uint64_t hash_scratch_request_descriptor(const ScratchRequestDescriptor& descriptor) noexcept {
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

        inline ScratchRequestDescriptor make_scratch_request_descriptor(
            const ScratchRequestBuildRequest& request) noexcept {
            ScratchRequestDescriptor descriptor{};
            descriptor.needOptics = request.families.needOptics;
            descriptor.needSpatialDir = request.families.needSpatialDir;
            descriptor.spatialDirDescriptorHash =
                request.families.needSpatialDir ? request.spatialDirDescriptorHash : 0;
            descriptor.spatialDirScratchTier =
                request.families.needSpatialDir ? request.spatialDirScratchTier : Spektrafilm::DirScratchTier::Tier0;
            descriptor.spatialDirPlaneRoles =
                request.families.needSpatialDir ? request.spatialDirPlaneRoles : Spektrafilm::DirScratchPlaneRoles{};
            descriptor.spatialDirTargetScratchTier =
                request.families.needSpatialDir ? request.spatialDirTargetScratchTier : Spektrafilm::DirScratchTier::Tier0;
            descriptor.spatialDirTargetPlaneRoles =
                request.families.needSpatialDir ? request.spatialDirTargetPlaneRoles : Spektrafilm::DirScratchPlaneRoles{};
            descriptor.requestedWidth = request.extent.requestedWidth;
            descriptor.requestedHeight = request.extent.requestedHeight;
            descriptor.needBlurred = request.families.needOptics && request.attachments.needBlurred;
            descriptor.aliasScannerRgbFromSpatialDirFiltered =
                request.families.needOptics &&
                request.families.needSpatialDir &&
                request.attachments.aliasScannerRgbFromSpatialDirFiltered;
            descriptor.needAux = request.families.needOptics && request.attachments.needAux;
            descriptor.needSharedTmp = request.attachments.needSharedTmp;
            descriptor.needGrainFrameUniforms =
                request.families.needOptics &&
                request.attachments.needGrainFrameUniforms;
            descriptor.needGrainLayerWork = request.families.needOptics && request.attachments.needGrainLayerWork;
            descriptor.needGrainShared = request.families.needOptics && request.attachments.needGrainShared;
            descriptor.needGateTransmittance = request.families.needOptics && request.attachments.needGateTransmittance;
            descriptor.needFilmDustTransmittance = request.families.needOptics && request.attachments.needFilmDustTransmittance;
            descriptor.gateWidth = descriptor.needGateTransmittance ? request.attachments.gateWidth : 0;
            descriptor.gateHeight = descriptor.needGateTransmittance ? request.attachments.gateHeight : 0;
            descriptor.generation = hash_scratch_request_descriptor(descriptor);
            return descriptor;
        }

        struct ResolvedMemoryBudget {
            std::uint64_t deviceBudgetBytes = 0;
            std::uint64_t allocationCapBytes = 0;
            int deviceId = -1;
        };

        struct SubmissionSnapshot {
            InstanceToken instanceToken{};
            FrameToken frameToken{};
            std::uint64_t snapshotId = 0;
            DeviceContextKey deviceContextKey{};
            std::uint64_t contextEpoch = 1;
            KeyDigests keyDigests{};
        };

        struct SubmissionTransaction {
            std::uint64_t transactionId = 0;
            std::uint64_t leaseGeneration = 0;
            SubmissionSnapshot snapshot{};
            ResolvedMemoryBudget resolvedMemoryBudget{};
            bool active = false;
            bool committed = false;
        };

        std::uint64_t normalize_key_u64(std::uint64_t value) noexcept;

        KeyDigests make_key_digests(
            std::uint64_t uploadCoreHash,
            std::uint64_t dirHash,
            std::uint64_t scannerHash,
            std::uint64_t autoExposureHash) noexcept;

        KeyDigests normalize_key_digests(const KeyDigests& digests) noexcept;

    } // namespace ResourceManager
} // namespace JuicerCuda
