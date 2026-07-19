#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "DiffusionFrameDescriptor.h"
#include "DiffusionPlanLayout.h"

namespace Spektrafilm {

    inline constexpr std::uint32_t kDiffusionPrecisionSchemaVersion = 1;
    inline constexpr std::uint32_t kDiffusionCandidateTableVersion = 1;
    inline constexpr std::uint32_t kDiffusionSelectionPolicyVersion = 1;

    struct DiffusionExecutionProfileKey {
        int cudaCompileVersion = 0;
        int cudaRuntimeVersion = 0;
        int cudaDriverVersion = 0;
        int cufftVersion = 0;
        int computeCapabilityMajor = 0;
        int computeCapabilityMinor = 0;
        std::uint32_t precisionSchema = kDiffusionPrecisionSchemaVersion;
        std::uint32_t layoutSchema = kDiffusionPlanLayoutSchemaVersion;
        std::uint32_t candidateTableVersion = kDiffusionCandidateTableVersion;
        std::uint32_t selectionPolicyVersion = kDiffusionSelectionPolicyVersion;

        friend bool operator==(
            const DiffusionExecutionProfileKey&,
            const DiffusionExecutionProfileKey&) = default;
    };

    struct DiffusionExecutionCandidate {
        CandidateExtent extent;
        std::uint64_t transformBytes = 0;
        std::uint64_t maximumR2cWorkBytes = 0;
        std::uint64_t maximumC2rWorkBytes = 0;
        std::uint64_t acceptedSharedWorkBytes = 0;
        std::uint64_t acceptedPlanAllowanceBytes = 0;
        std::uint64_t pairMedianNanoseconds = 0;
        std::uint64_t oneFrameBytes = 0;
        std::uint64_t twoFrameBytes = 0;
    };

    struct DiffusionStageTileGeometry {
        DiffusionLinearStage stage = DiffusionLinearStage::CameraFilmLinear;
        std::uint64_t stageDescriptorHash = 0;
        double scatterFraction = 0.0;
        std::size_t spectrumKeyIndex = 0;
        int radiusPixels = 0;
        int validTileWidth = 0;
        int validTileHeight = 0;
        int tileCountX = 0;
        int tileCountY = 0;
    };

    struct DiffusionSpectrumKey {
        std::uint64_t sampleHash = 0;
        CandidateExtent extent;
        std::uint32_t layoutSchema = kDiffusionPlanLayoutSchemaVersion;
        std::uint64_t hash = 0;
    };

    struct DiffusionPlanKey {
        std::array<std::uint8_t, 32> profileDigest{};
        CandidateExtent extent;
        std::uint32_t layoutSchema = kDiffusionPlanLayoutSchemaVersion;
        std::uint64_t hash = 0;
    };

    struct DiffusionExecutionDescriptor {
        std::array<std::uint8_t, 32> profileKeyDigest{};
        std::array<std::uint8_t, 32> profileDigest{};
        std::uint64_t contextEpoch = 0;
        std::uint64_t selectionCapBytes = 0;
        std::uint64_t frameSetHash = 0;
        DiffusionFrameDomain fullFrame;
        CandidateExtent extent;
        PlanLayout layout;
        std::array<DiffusionStageTileGeometry, 2> stages{};
        std::array<DiffusionSpectrumKey, 2> spectrumKeys{};
        std::size_t stageCount = 0;
        std::size_t uniqueSpectrumCount = 0;
        DiffusionPlanKey planKey;
        std::uint64_t transformBufferBytes = 0;
        std::uint64_t spectrumPackageBytes = 0;
        std::uint64_t stagePlaneBytes = 0;
        std::uint64_t acceptedR2cWorkBytes = 0;
        std::uint64_t acceptedC2rWorkBytes = 0;
        std::uint64_t acceptedSharedWorkBytes = 0;
        std::uint64_t acceptedPlanAllowanceBytes = 0;
        std::uint64_t hash = 0;
    };

    struct SupportedDiffusionExecutionProfile {
        DiffusionExecutionProfileKey key;
        std::array<std::uint8_t, 32> profileKeyDigest{};
        std::array<std::uint8_t, 32> profileDigest{};
        std::uint64_t referenceCapBytes = 0;
        std::array<DiffusionExecutionCandidate, 10> candidates{};
    };

    struct DiffusionCandidateSelection {
        std::size_t candidateIndex = 0;
        std::array<DiffusionStageTileGeometry, 2> stages{};
        std::size_t stageCount = 0;
        std::uint64_t incrementalBytes = 0;
        std::uint64_t workloadNanoseconds = 0;
    };

    const SupportedDiffusionExecutionProfile&
    supported_diffusion_execution_profile() noexcept;

    bool diffusion_execution_profile_matches(
        const DiffusionExecutionProfileKey& observed,
        std::string& diagnostic);

    bool select_diffusion_execution_candidate(
        const SupportedDiffusionExecutionProfile& profile,
        const DiffusionFrameSetDescriptor& frameSet,
        std::uint64_t resolvedDeviceCapBytes,
        DiffusionCandidateSelection& out,
        std::string& diagnostic);

    bool build_diffusion_execution_descriptor(
        const DiffusionFrameSetDescriptor& frameSet,
        std::uint64_t contextEpoch,
        const DiffusionExecutionProfileKey& observedProfile,
        std::uint64_t resolvedDeviceCapBytes,
        DiffusionExecutionDescriptor& out,
        std::string& diagnostic);

    bool diffusion_plan_work_fits(
        const DiffusionExecutionDescriptor& descriptor,
        std::uint64_t actualR2cWorkBytes,
        std::uint64_t actualC2rWorkBytes,
        std::string& diagnostic);

} // namespace Spektrafilm
