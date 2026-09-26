// Cuda/ResourceManager/JuicerCudaResourceCore.h
//
// Shared types and policy helpers for CUDA resource ownership and submission.
#pragma once

#include <cstdint>
#include <cstddef>
#include <functional>
#include <optional>
#include <variant>

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

        struct ScratchRequestExtent {
            int requestedWidth = 0;
            int requestedHeight = 0;
        };

        struct ScratchRequestAttachments {
            bool needOptics = false;
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

        class ActiveDirScratch {
        public:
            struct Layout {
                Spektrafilm::DirScratchTier tier = Spektrafilm::DirScratchTier::Tier0;
                Spektrafilm::DirScratchPlaneRoles roles{};
            };

            static std::optional<ActiveDirScratch> create(
                std::uint64_t descriptorHash,
                const Layout& source,
                const Layout& target) noexcept;

        private:
            ActiveDirScratch() = default;

            std::uint64_t _descriptorHash = 0;
            Layout _source;
            Layout _target;

            friend std::optional<ScratchRequestDescriptor> make_scratch_request_descriptor(
                const ScratchRequestExtent& extent,
                const std::variant<std::monostate, ActiveDirScratch>& spatialDir,
                const ScratchRequestAttachments& attachments) noexcept;
        };

        // Inactive DIR has no hash, tier or roles to contradict the selected state.
        using DirScratchRequest = std::variant<std::monostate, ActiveDirScratch>;

        std::optional<ScratchRequestDescriptor> make_scratch_request_descriptor(
            const ScratchRequestExtent& extent,
            const DirScratchRequest& spatialDir,
            const ScratchRequestAttachments& attachments) noexcept;

        bool scratch_request_descriptor_is_valid(const ScratchRequestDescriptor& descriptor) noexcept;
        std::uint64_t hash_scratch_request_descriptor(const ScratchRequestDescriptor& descriptor) noexcept;

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
