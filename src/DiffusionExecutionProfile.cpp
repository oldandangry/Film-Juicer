#include "DiffusionExecutionProfile.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

namespace {

    constexpr std::uint64_t kFnvOffset = 0xcbf29ce484222325ULL;
    constexpr std::uint64_t kFnvPrime = 0x100000001b3ULL;
    constexpr std::string_view kAcceptedProfileKeyDigestHex =
        "63335b9008e3974c4206513750615f6cfa1515717a92e7c82ee15d8628c1b400";
    constexpr std::string_view kAcceptedProfileDigestHex =
        "fcdf3e9919114df8f48b06611939ec72dfd810309e393a1f5f1de6d27e4f7d31";

    consteval std::uint8_t hex_nibble(char value) {
        if (value >= '0' && value <= '9') {
            return static_cast<std::uint8_t>(value - '0');
        }
        if (value >= 'a' && value <= 'f') {
            return static_cast<std::uint8_t>(value - 'a' + 10);
        }
        return 0xffu;
    }

    consteval std::array<std::uint8_t, 32> decode_digest(std::string_view text) {
        std::array<std::uint8_t, 32> digest{};
        if (text.size() != digest.size() * 2) {
            return {};
        }
        for (std::size_t index = 0; index < digest.size(); ++index) {
            const std::uint8_t high = hex_nibble(text[index * 2]);
            const std::uint8_t low = hex_nibble(text[index * 2 + 1]);
            if (high > 0x0fu || low > 0x0fu) {
                return {};
            }
            digest[index] = static_cast<std::uint8_t>((high << 4u) | low);
        }
        return digest;
    }

    constexpr std::array<Spektrafilm::DiffusionExecutionCandidate, 10>
        kAcceptedCandidates{{{{1024, 1024}, 4202496, 0, 0, 0, 77594624, 32432, 673243136, 8021745664},
                             {{2048, 2048}, 16793600, 0, 0, 0, 69206016, 114080, 752992256, 8219017216},
                             {{3072, 3072}, 37773312, 0, 0, 0, 69206016, 396288, 899850240, 8575672320},
                             {{4096, 4096}, 67141632, 67141632, 67141632, 67141632, 69206016, 1088512, 1172570112, 9209217024},
                             {{4608, 4608}, 84971520, 84971520, 84971520, 84971520, 69206016, 2892800, 1315209216, 9547984896},
                             {{5120, 5120}, 104898560, 104898560, 104898560, 104898560, 69206016, 3808768, 1474625536, 9926598656},
                             {{6144, 6144}, 151044096, 151044096, 151044096, 151044096, 69206016, 6086144, 1843789824, 10803363840},
                             {{8192, 6144}, 201375744, 201375744, 201375744, 201375744, 69206016, 5111808, 2246443008, 11797561344},
                             {{6144, 8192}, 201392128, 201392128, 201392128, 201392128, 69206016, 7418880, 2246574080, 11797921792},
                             {{8192, 8192}, 268500992, 268500992, 268500992, 268500992, 69206016, 5972480, 2783444992, 13274316800}}};

    constexpr Spektrafilm::SupportedDiffusionExecutionProfile kAcceptedProfile{
        {13020,
         13020,
         13020,
         12300,
         8,
         9,
         Spektrafilm::kDiffusionPrecisionSchemaVersion,
         Spektrafilm::kDiffusionPlanLayoutSchemaVersion,
         Spektrafilm::kDiffusionCandidateTableVersion,
         Spektrafilm::kDiffusionSelectionPolicyVersion},
        decode_digest(kAcceptedProfileKeyDigestHex),
        decode_digest(kAcceptedProfileDigestHex),
        16902520832ULL,
        kAcceptedCandidates};

    void fail(std::string& diagnostic, std::string_view field) {
        diagnostic = "InvalidDiffusionExecutionDescriptor field=";
        diagnostic.append(field);
    }

    void profile_mismatch(std::string& diagnostic, std::string_view field) {
        diagnostic = "UnsupportedCudaExecutionProfile field=";
        diagnostic.append(field);
    }

    bool checked_add(
        std::uint64_t left,
        std::uint64_t right,
        std::uint64_t& out) noexcept {
        if (right > std::numeric_limits<std::uint64_t>::max() - left) {
            return false;
        }
        out = left + right;
        return true;
    }

    bool checked_multiply(
        std::uint64_t left,
        std::uint64_t right,
        std::uint64_t& out) noexcept {
        if (right != 0 && left > std::numeric_limits<std::uint64_t>::max() / right) {
            return false;
        }
        out = left * right;
        return true;
    }

    void hash_byte(std::uint64_t& hash, std::uint8_t value) noexcept {
        hash ^= static_cast<std::uint64_t>(value);
        hash *= kFnvPrime;
    }

    void hash_u32_le(std::uint64_t& hash, std::uint32_t value) noexcept {
        for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
            hash_byte(
                hash,
                static_cast<std::uint8_t>((value >> (byte * 8u)) & 0xffu));
        }
    }

    void hash_u64_le(std::uint64_t& hash, std::uint64_t value) noexcept {
        for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
            hash_byte(
                hash,
                static_cast<std::uint8_t>((value >> (byte * 8u)) & 0xffu));
        }
    }

    void hash_tag(std::uint64_t& hash, std::string_view tag) noexcept {
        for (const char value : tag) {
            hash_byte(hash, static_cast<std::uint8_t>(value));
        }
        hash_byte(hash, 0);
    }

    void hash_double(std::uint64_t& hash, double value) noexcept {
        if (value == 0.0) {
            value = 0.0;
        }
        hash_u64_le(hash, std::bit_cast<std::uint64_t>(value));
    }

    template <std::size_t Size>
    void hash_bytes(
        std::uint64_t& hash,
        const std::array<std::uint8_t, Size>& values) noexcept {
        for (const std::uint8_t value : values) {
            hash_byte(hash, value);
        }
    }

    void hash_extent(
        std::uint64_t& hash,
        const Spektrafilm::CandidateExtent& extent) noexcept {
        hash_u32_le(hash, static_cast<std::uint32_t>(extent.width));
        hash_u32_le(hash, static_cast<std::uint32_t>(extent.height));
    }

    void hash_domain(
        std::uint64_t& hash,
        const Spektrafilm::DiffusionFrameDomain& domain) noexcept {
        hash_u32_le(hash, static_cast<std::uint32_t>(domain.originX));
        hash_u32_le(hash, static_cast<std::uint32_t>(domain.originY));
        hash_u32_le(hash, static_cast<std::uint32_t>(domain.width));
        hash_u32_le(hash, static_cast<std::uint32_t>(domain.height));
    }

    void hash_layout(
        std::uint64_t& hash,
        const Spektrafilm::PlanLayout& layout) noexcept {
        hash_u32_le(hash, static_cast<std::uint32_t>(layout.width));
        hash_u32_le(hash, static_cast<std::uint32_t>(layout.height));
        hash_u32_le(hash, static_cast<std::uint32_t>(layout.complexWidth));
        hash_u32_le(hash, static_cast<std::uint32_t>(layout.physicalRealRowFloats));
        hash_u32_le(hash, static_cast<std::uint32_t>(layout.realDistance));
        hash_u32_le(hash, static_cast<std::uint32_t>(layout.complexDistance));
        hash_u64_le(hash, layout.transformBytes);
    }

    bool valid_route(Spektrafilm::ScanRoute route) noexcept {
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

    bool validate_stage(
        const Spektrafilm::DiffusionStageFrameDescriptor& stage,
        const Spektrafilm::DiffusionFrameSetDescriptor& frameSet,
        Spektrafilm::DiffusionLinearStage expectedStage,
        std::string& diagnostic) {
        if (stage.route != frameSet.route) {
            fail(diagnostic, "stage_route");
            return false;
        }
        if (stage.stage != expectedStage) {
            fail(diagnostic, "stage_order");
            return false;
        }
        if (stage.fullFrame != frameSet.fullFrame) {
            fail(diagnostic, "stage_full_frame");
            return false;
        }
        if (stage.recipeComponentHash == 0 || stage.hash == 0) {
            fail(diagnostic, "stage_hash");
            return false;
        }
        if (!std::isfinite(stage.scatterFraction) ||
            stage.scatterFraction <= 0.0 || stage.scatterFraction > 1.0) {
            fail(diagnostic, "stage_scatter_fraction");
            return false;
        }
        if (!std::isfinite(stage.pixelSizeUm) || stage.pixelSizeUm <= 0.0) {
            fail(diagnostic, "stage_pixel_size_um");
            return false;
        }
        if (stage.radiusPixels <= 0 ||
            stage.sample.radiusPixels != stage.radiusPixels || stage.sample.hash == 0) {
            fail(diagnostic, "stage_sample");
            return false;
        }
        return true;
    }

    bool validate_frame_set(
        const Spektrafilm::DiffusionFrameSetDescriptor& frameSet,
        std::array<const Spektrafilm::DiffusionStageFrameDescriptor*, 2>& stages,
        std::size_t& stageCount,
        std::string& diagnostic) {
        stages = {};
        stageCount = 0;
        if (!valid_route(frameSet.route)) {
            fail(diagnostic, "frame_route");
            return false;
        }
        if (frameSet.fullFrame.width < 2 || frameSet.fullFrame.height < 2) {
            fail(diagnostic, "frame_domain");
            return false;
        }
        if (frameSet.hash == 0) {
            fail(diagnostic, "frame_set_hash");
            return false;
        }
        if (frameSet.candidatePolicyVersion !=
            Spektrafilm::kDiffusionCandidatePolicyVersion) {
            fail(diagnostic, "candidate_policy_version");
            return false;
        }
        if (frameSet.planeRoles != Spektrafilm::kDiffusionSemanticPlaneRoles) {
            fail(diagnostic, "semantic_plane_roles");
            return false;
        }
        if (!frameSet.camera && !frameSet.enlarger) {
            fail(diagnostic, "active_stage_count");
            return false;
        }
        if (frameSet.enlarger && !Spektrafilm::scan_route_is_print(frameSet.route)) {
            fail(diagnostic, "enlarger_route");
            return false;
        }

        int maximumRadius = 0;
        if (frameSet.camera) {
            if (!validate_stage(
                    *frameSet.camera,
                    frameSet,
                    Spektrafilm::DiffusionLinearStage::CameraFilmLinear,
                    diagnostic)) {
                return false;
            }
            stages[stageCount++] = &*frameSet.camera;
            maximumRadius = frameSet.camera->radiusPixels;
        }
        if (frameSet.enlarger) {
            if (!validate_stage(
                    *frameSet.enlarger,
                    frameSet,
                    Spektrafilm::DiffusionLinearStage::EnlargerPrintLinear,
                    diagnostic)) {
                return false;
            }
            stages[stageCount++] = &*frameSet.enlarger;
            maximumRadius = std::max(maximumRadius, frameSet.enlarger->radiusPixels);
        }
        if (maximumRadius != frameSet.maximumRadiusPixels) {
            fail(diagnostic, "maximum_radius_pixels");
            return false;
        }

        std::uint64_t expectedWorkspace = 0;
        if (!checked_multiply(
                static_cast<std::uint64_t>(frameSet.fullFrame.width),
                static_cast<std::uint64_t>(frameSet.fullFrame.height),
                expectedWorkspace) ||
            !checked_multiply(expectedWorkspace, 4, expectedWorkspace) ||
            !checked_multiply(expectedWorkspace, sizeof(float), expectedWorkspace) ||
            expectedWorkspace != frameSet.workspaceBytes) {
            fail(diagnostic, "stage_plane_bytes");
            return false;
        }
        return true;
    }

    bool validate_candidate(
        const Spektrafilm::DiffusionExecutionCandidate& candidate,
        Spektrafilm::PlanLayout& layout,
        std::string& diagnostic) {
        if (!Spektrafilm::make_plan_layout(
                candidate.extent.width,
                candidate.extent.height,
                layout) ||
            layout.transformBytes != candidate.transformBytes) {
            fail(diagnostic, "candidate_transform_layout");
            return false;
        }
        if (candidate.acceptedSharedWorkBytes < candidate.maximumR2cWorkBytes ||
            candidate.acceptedSharedWorkBytes < candidate.maximumC2rWorkBytes) {
            fail(diagnostic, "candidate_shared_work_bytes");
            return false;
        }
        if (candidate.pairMedianNanoseconds == 0 ||
            candidate.oneFrameBytes == 0 || candidate.twoFrameBytes == 0 ||
            candidate.oneFrameBytes > candidate.twoFrameBytes) {
            fail(diagnostic, "candidate_measurement");
            return false;
        }
        return true;
    }

    struct EvaluatedCandidate {
        std::size_t candidateIndex = 0;
        Spektrafilm::CandidateExtent extent;
        std::array<Spektrafilm::DiffusionStageTileGeometry, 2> stages{};
        std::size_t stageCount = 0;
        std::uint64_t memoryBytes = 0;
        std::uint64_t workloadNanoseconds = 0;
    };

    bool lower_dimension_tie(
        const EvaluatedCandidate& left,
        const EvaluatedCandidate& right) noexcept {
        return left.extent.height < right.extent.height ||
               (left.extent.height == right.extent.height &&
                left.extent.width < right.extent.width);
    }

    bool lower_cost_tie(
        const EvaluatedCandidate& left,
        const EvaluatedCandidate& right) noexcept {
        if (left.memoryBytes != right.memoryBytes) {
            return left.memoryBytes < right.memoryBytes;
        }
        if (left.workloadNanoseconds != right.workloadNanoseconds) {
            return left.workloadNanoseconds < right.workloadNanoseconds;
        }
        return lower_dimension_tie(left, right);
    }

    bool evaluate_candidate(
        const Spektrafilm::DiffusionExecutionCandidate& candidate,
        std::size_t candidateIndex,
        const Spektrafilm::DiffusionFrameSetDescriptor& frameSet,
        const std::array<const Spektrafilm::DiffusionStageFrameDescriptor*, 2>& stages,
        std::size_t stageCount,
        EvaluatedCandidate& out,
        std::string& diagnostic) {
        Spektrafilm::PlanLayout layout{};
        if (!validate_candidate(candidate, layout, diagnostic)) {
            return false;
        }

        out = {};
        out.candidateIndex = candidateIndex;
        out.extent = candidate.extent;
        out.stageCount = stageCount;
        std::uint64_t totalTiles = 0;
        for (std::size_t index = 0; index < stageCount; ++index) {
            const auto& stage = *stages[index];
            std::uint64_t border = 0;
            if (!checked_multiply(
                    static_cast<std::uint64_t>(stage.radiusPixels), 2, border)) {
                fail(diagnostic, "candidate_radius_border");
                return false;
            }
            if (static_cast<std::uint64_t>(candidate.extent.width) <= border ||
                static_cast<std::uint64_t>(candidate.extent.height) <= border) {
                return true;
            }
            const std::uint64_t validWidth =
                static_cast<std::uint64_t>(candidate.extent.width) - border;
            const std::uint64_t validHeight =
                static_cast<std::uint64_t>(candidate.extent.height) - border;
            const std::uint64_t frameWidth =
                static_cast<std::uint64_t>(frameSet.fullFrame.width);
            const std::uint64_t frameHeight =
                static_cast<std::uint64_t>(frameSet.fullFrame.height);
            const std::uint64_t tileCountX =
                frameWidth / validWidth + (frameWidth % validWidth != 0 ? 1u : 0u);
            const std::uint64_t tileCountY =
                frameHeight / validHeight + (frameHeight % validHeight != 0 ? 1u : 0u);
            if (validWidth > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
                validHeight > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
                tileCountX > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
                tileCountY > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
                fail(diagnostic, "candidate_tile_geometry");
                return false;
            }
            std::uint64_t tileCount = 0;
            if (!checked_multiply(tileCountX, tileCountY, tileCount) ||
                !checked_add(totalTiles, tileCount, totalTiles)) {
                fail(diagnostic, "candidate_tile_count");
                return false;
            }
            out.stages[index] = {
                stage.stage,
                stage.hash,
                stage.scatterFraction,
                0,
                stage.radiusPixels,
                static_cast<int>(validWidth),
                static_cast<int>(validHeight),
                static_cast<int>(tileCountX),
                static_cast<int>(tileCountY)};
        }

        std::uint64_t channelTileCount = 0;
        if (!checked_multiply(totalTiles, 3, channelTileCount) ||
            !checked_multiply(
                candidate.pairMedianNanoseconds,
                channelTileCount,
                out.workloadNanoseconds)) {
            fail(diagnostic, "candidate_workload_nanoseconds");
            return false;
        }

        std::uint64_t spectrumAndTransformCount = 0;
        if (!checked_multiply(static_cast<std::uint64_t>(stageCount), 3, spectrumAndTransformCount) ||
            !checked_add(spectrumAndTransformCount, 1, spectrumAndTransformCount) ||
            !checked_multiply(
                spectrumAndTransformCount,
                candidate.transformBytes,
                out.memoryBytes) ||
            !checked_add(
                out.memoryBytes,
                candidate.acceptedSharedWorkBytes,
                out.memoryBytes) ||
            !checked_add(
                out.memoryBytes,
                candidate.acceptedPlanAllowanceBytes,
                out.memoryBytes)) {
            fail(diagnostic, "candidate_incremental_bytes");
            return false;
        }
        return true;
    }

    bool dominated_by(
        const EvaluatedCandidate& candidate,
        const EvaluatedCandidate& other) noexcept {
        return other.memoryBytes <= candidate.memoryBytes &&
               other.workloadNanoseconds <= candidate.workloadNanoseconds &&
               (other.memoryBytes < candidate.memoryBytes ||
                other.workloadNanoseconds < candidate.workloadNanoseconds);
    }

    struct WideUnsigned {
        std::uint64_t high = 0;
        std::uint64_t low = 0;
    };

    // Multiplication is commutative, so swapping these operands is harmless.
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    WideUnsigned multiply_wide(
        std::uint64_t left,
        std::uint64_t right) noexcept {
        constexpr std::uint64_t kLowMask = 0xffffffffULL;
        const std::uint64_t leftLow = left & kLowMask;
        const std::uint64_t leftHigh = left >> 32u;
        const std::uint64_t rightLow = right & kLowMask;
        const std::uint64_t rightHigh = right >> 32u;

        const std::uint64_t lowProduct = leftLow * rightLow;
        const std::uint64_t lowWord = lowProduct & kLowMask;
        const std::uint64_t firstCross =
            leftHigh * rightLow + (lowProduct >> 32u);
        const std::uint64_t firstCrossLow = firstCross & kLowMask;
        const std::uint64_t firstCrossHigh = firstCross >> 32u;
        const std::uint64_t secondCross =
            leftLow * rightHigh + firstCrossLow;

        return {
            leftHigh * rightHigh + firstCrossHigh +
                (secondCross >> 32u),
            (secondCross << 32u) + lowWord};
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

    int compare_wide(
        const WideUnsigned& left,
        const WideUnsigned& right) noexcept {
        if (left.high != right.high) {
            return left.high < right.high ? -1 : 1;
        }
        if (left.low != right.low) {
            return left.low < right.low ? -1 : 1;
        }
        return 0;
    }

    WideUnsigned subtract_wide(
        const WideUnsigned& greater,
        const WideUnsigned& lesser) noexcept {
        return {
            greater.high - lesser.high -
                (greater.low < lesser.low ? 1u : 0u),
            greater.low - lesser.low};
    }

    struct KneeNumerator {
        bool positive = false;
        WideUnsigned magnitude{};
    };

    KneeNumerator knee_numerator(
        const EvaluatedCandidate& candidate,
        const EvaluatedCandidate& lowestMemory,
        const EvaluatedCandidate& fastest) noexcept {
        const std::uint64_t timeGain =
            lowestMemory.workloadNanoseconds - candidate.workloadNanoseconds;
        const std::uint64_t memoryRange = fastest.memoryBytes - lowestMemory.memoryBytes;
        const std::uint64_t memoryCost = candidate.memoryBytes - lowestMemory.memoryBytes;
        const std::uint64_t timeRange =
            lowestMemory.workloadNanoseconds - fastest.workloadNanoseconds;
        const WideUnsigned left = multiply_wide(timeGain, memoryRange);
        const WideUnsigned right = multiply_wide(memoryCost, timeRange);
        const int order = compare_wide(left, right);
        return {
            order > 0,
            order >= 0 ? subtract_wide(left, right)
                       : subtract_wide(right, left)};
    }

    std::uint64_t hash_spectrum_key(
        std::uint64_t sampleHash,
        const Spektrafilm::CandidateExtent& extent) noexcept {
        std::uint64_t hash = kFnvOffset;
        hash_tag(hash, "diffusion-spectrum-key");
        hash_u32_le(hash, Spektrafilm::kDiffusionPlanLayoutSchemaVersion);
        hash_u64_le(hash, sampleHash);
        hash_extent(hash, extent);
        return hash;
    }

    std::uint64_t hash_plan_key(
        const std::array<std::uint8_t, 32>& profileDigest,
        const Spektrafilm::CandidateExtent& extent) noexcept {
        std::uint64_t hash = kFnvOffset;
        hash_tag(hash, "diffusion-plan-key");
        hash_u32_le(hash, Spektrafilm::kDiffusionPlanLayoutSchemaVersion);
        hash_bytes(hash, profileDigest);
        hash_extent(hash, extent);
        return hash;
    }

    std::uint64_t hash_execution_descriptor(
        const Spektrafilm::DiffusionExecutionDescriptor& descriptor) noexcept {
        std::uint64_t hash = kFnvOffset;
        hash_tag(hash, "diffusion-execution-descriptor");
        hash_u32_le(hash, Spektrafilm::kDiffusionFrameDescriptorSchemaVersion);
        hash_u32_le(hash, Spektrafilm::kDiffusionPlanLayoutSchemaVersion);
        hash_u32_le(hash, Spektrafilm::kDiffusionPrecisionSchemaVersion);
        hash_u32_le(hash, Spektrafilm::kDiffusionSelectionPolicyVersion);
        hash_bytes(hash, descriptor.profileKeyDigest);
        hash_bytes(hash, descriptor.profileDigest);
        hash_u64_le(hash, descriptor.contextEpoch);
        hash_u64_le(hash, descriptor.selectionCapBytes);
        hash_u64_le(hash, descriptor.frameSetHash);
        hash_domain(hash, descriptor.fullFrame);
        hash_extent(hash, descriptor.extent);
        hash_layout(hash, descriptor.layout);
        hash_u64_le(hash, static_cast<std::uint64_t>(descriptor.stageCount));
        for (std::size_t index = 0; index < descriptor.stageCount; ++index) {
            const auto& stage = descriptor.stages[index];
            hash_byte(hash, static_cast<std::uint8_t>(stage.stage));
            hash_u64_le(hash, stage.stageDescriptorHash);
            hash_double(hash, stage.scatterFraction);
            hash_u64_le(hash, static_cast<std::uint64_t>(stage.spectrumKeyIndex));
            hash_u32_le(hash, static_cast<std::uint32_t>(stage.radiusPixels));
            hash_u32_le(hash, static_cast<std::uint32_t>(stage.validTileWidth));
            hash_u32_le(hash, static_cast<std::uint32_t>(stage.validTileHeight));
            hash_u32_le(hash, static_cast<std::uint32_t>(stage.tileCountX));
            hash_u32_le(hash, static_cast<std::uint32_t>(stage.tileCountY));
        }
        hash_u64_le(hash, static_cast<std::uint64_t>(descriptor.uniqueSpectrumCount));
        for (std::size_t index = 0; index < descriptor.uniqueSpectrumCount; ++index) {
            const auto& key = descriptor.spectrumKeys[index];
            hash_u64_le(hash, key.sampleHash);
            hash_extent(hash, key.extent);
            hash_u32_le(hash, key.layoutSchema);
            hash_u64_le(hash, key.hash);
        }
        hash_u64_le(hash, descriptor.planKey.hash);
        hash_u64_le(hash, descriptor.transformBufferBytes);
        hash_u64_le(hash, descriptor.spectrumPackageBytes);
        hash_u64_le(hash, descriptor.stagePlaneBytes);
        hash_u64_le(hash, descriptor.acceptedR2cWorkBytes);
        hash_u64_le(hash, descriptor.acceptedC2rWorkBytes);
        hash_u64_le(hash, descriptor.acceptedSharedWorkBytes);
        hash_u64_le(hash, descriptor.acceptedPlanAllowanceBytes);
        return hash;
    }

} // namespace

namespace Spektrafilm {

    const SupportedDiffusionExecutionProfile&
    supported_diffusion_execution_profile() noexcept {
        return kAcceptedProfile;
    }

    bool diffusion_execution_profile_matches(
        const DiffusionExecutionProfileKey& observed,
        std::string& diagnostic) {
        diagnostic.clear();
        const auto& accepted = kAcceptedProfile.key;
        if (observed.cudaCompileVersion != accepted.cudaCompileVersion) {
            profile_mismatch(diagnostic, "cuda_compile_version");
        } else if (observed.cudaRuntimeVersion != accepted.cudaRuntimeVersion) {
            profile_mismatch(diagnostic, "cuda_runtime_version");
        } else if (observed.cudaDriverVersion != accepted.cudaDriverVersion) {
            profile_mismatch(diagnostic, "cuda_driver_version");
        } else if (observed.cufftVersion != accepted.cufftVersion) {
            profile_mismatch(diagnostic, "cufft_version");
        } else if (observed.computeCapabilityMajor != accepted.computeCapabilityMajor) {
            profile_mismatch(diagnostic, "compute_capability_major");
        } else if (observed.computeCapabilityMinor != accepted.computeCapabilityMinor) {
            profile_mismatch(diagnostic, "compute_capability_minor");
        } else if (observed.precisionSchema != accepted.precisionSchema) {
            profile_mismatch(diagnostic, "precision_schema");
        } else if (observed.layoutSchema != accepted.layoutSchema) {
            profile_mismatch(diagnostic, "layout_schema");
        } else if (observed.candidateTableVersion != accepted.candidateTableVersion) {
            profile_mismatch(diagnostic, "candidate_table_version");
        } else if (observed.selectionPolicyVersion != accepted.selectionPolicyVersion) {
            profile_mismatch(diagnostic, "selection_policy_version");
        }
        return diagnostic.empty();
    }

    bool select_diffusion_execution_candidate(
        const SupportedDiffusionExecutionProfile& profile,
        const DiffusionFrameSetDescriptor& frameSet,
        std::uint64_t resolvedDeviceCapBytes,
        DiffusionCandidateSelection& out,
        std::string& diagnostic) {
        out = {};
        diagnostic.clear();

        std::array<const DiffusionStageFrameDescriptor*, 2> stages{};
        std::size_t stageCount = 0;
        if (!validate_frame_set(frameSet, stages, stageCount, diagnostic)) {
            return false;
        }
        if (profile.referenceCapBytes == 0) {
            fail(diagnostic, "reference_cap_bytes");
            return false;
        }
        const std::uint64_t selectionCap =
            std::min(resolvedDeviceCapBytes, profile.referenceCapBytes);

        std::array<EvaluatedCandidate, 10> evaluated{};
        std::size_t evaluatedCount = 0;
        for (std::size_t index = 0; index < profile.candidates.size(); ++index) {
            const auto& candidate = profile.candidates[index];
            if (candidate.twoFrameBytes >= selectionCap) {
                continue;
            }
            EvaluatedCandidate value{};
            if (!evaluate_candidate(
                    candidate,
                    index,
                    frameSet,
                    stages,
                    stageCount,
                    value,
                    diagnostic)) {
                return false;
            }
            if (value.workloadNanoseconds == 0) {
                continue;
            }
            evaluated[evaluatedCount++] = value;
        }
        if (evaluatedCount == 0) {
            fail(diagnostic, "admissible_candidate");
            return false;
        }

        std::array<EvaluatedCandidate, 10> coalesced{};
        std::size_t coalescedCount = 0;
        for (std::size_t index = 0; index < evaluatedCount; ++index) {
            const auto& candidate = evaluated[index];
            std::size_t matching = coalescedCount;
            for (std::size_t existing = 0; existing < coalescedCount; ++existing) {
                if (coalesced[existing].memoryBytes == candidate.memoryBytes &&
                    coalesced[existing].workloadNanoseconds ==
                        candidate.workloadNanoseconds) {
                    matching = existing;
                    break;
                }
            }
            if (matching == coalescedCount) {
                coalesced[coalescedCount++] = candidate;
            } else if (lower_dimension_tie(candidate, coalesced[matching])) {
                coalesced[matching] = candidate;
            }
        }

        std::array<EvaluatedCandidate, 10> frontier{};
        std::size_t frontierCount = 0;
        for (std::size_t index = 0; index < coalescedCount; ++index) {
            bool dominated = false;
            for (std::size_t other = 0; other < coalescedCount; ++other) {
                if (index != other && dominated_by(coalesced[index], coalesced[other])) {
                    dominated = true;
                    break;
                }
            }
            if (!dominated) {
                frontier[frontierCount++] = coalesced[index];
            }
        }
        if (frontierCount == 0) {
            fail(diagnostic, "pareto_frontier");
            return false;
        }
        std::sort(
            frontier.begin(),
            frontier.begin() + static_cast<std::ptrdiff_t>(frontierCount),
            lower_cost_tie);

        const EvaluatedCandidate& lowestMemory = frontier[0];
        const EvaluatedCandidate* fastest = &frontier[0];
        for (std::size_t index = 1; index < frontierCount; ++index) {
            const auto& candidate = frontier[index];
            if (candidate.workloadNanoseconds < fastest->workloadNanoseconds ||
                (candidate.workloadNanoseconds == fastest->workloadNanoseconds &&
                 lower_cost_tie(candidate, *fastest))) {
                fastest = &candidate;
            }
        }

        const EvaluatedCandidate* selected = &lowestMemory;
        bool hasPositiveScore = false;
        WideUnsigned selectedMagnitude{};
        if (fastest->memoryBytes > lowestMemory.memoryBytes &&
            lowestMemory.workloadNanoseconds > fastest->workloadNanoseconds) {
            for (std::size_t index = 0; index < frontierCount; ++index) {
                const auto& candidate = frontier[index];
                const KneeNumerator numerator =
                    knee_numerator(candidate, lowestMemory, *fastest);
                const int magnitudeOrder =
                    compare_wide(numerator.magnitude, selectedMagnitude);
                if (numerator.positive &&
                    (!hasPositiveScore || magnitudeOrder > 0 ||
                     (magnitudeOrder == 0 &&
                      lower_cost_tie(candidate, *selected)))) {
                    selected = &candidate;
                    selectedMagnitude = numerator.magnitude;
                    hasPositiveScore = true;
                }
            }
        }

        out.candidateIndex = selected->candidateIndex;
        out.stages = selected->stages;
        out.stageCount = selected->stageCount;
        out.incrementalBytes = selected->memoryBytes;
        out.workloadNanoseconds = selected->workloadNanoseconds;
        return true;
    }

    bool build_diffusion_execution_descriptor(
        const DiffusionFrameSetDescriptor& frameSet,
        std::uint64_t contextEpoch,
        const DiffusionExecutionProfileKey& observedProfile,
        std::uint64_t resolvedDeviceCapBytes,
        DiffusionExecutionDescriptor& out,
        std::string& diagnostic) {
        out = {};
        diagnostic.clear();
        if (!diffusion_execution_profile_matches(observedProfile, diagnostic)) {
            return false;
        }
        if (contextEpoch == 0) {
            fail(diagnostic, "context_epoch");
            return false;
        }

        DiffusionCandidateSelection selection{};
        if (!select_diffusion_execution_candidate(
                kAcceptedProfile,
                frameSet,
                resolvedDeviceCapBytes,
                selection,
                diagnostic)) {
            return false;
        }
        const auto& candidate = kAcceptedProfile.candidates[selection.candidateIndex];
        PlanLayout layout{};
        if (!make_plan_layout(candidate.extent.width, candidate.extent.height, layout) ||
            layout.transformBytes != candidate.transformBytes) {
            fail(diagnostic, "selected_plan_layout");
            return false;
        }

        std::array<const DiffusionStageFrameDescriptor*, 2> sourceStages{};
        std::size_t sourceStageCount = 0;
        if (!validate_frame_set(
                frameSet,
                sourceStages,
                sourceStageCount,
                diagnostic) ||
            sourceStageCount != selection.stageCount) {
            if (diagnostic.empty()) {
                fail(diagnostic, "selected_stage_count");
            }
            return false;
        }

        out.profileKeyDigest = kAcceptedProfile.profileKeyDigest;
        out.profileDigest = kAcceptedProfile.profileDigest;
        out.contextEpoch = contextEpoch;
        out.selectionCapBytes =
            std::min(resolvedDeviceCapBytes, kAcceptedProfile.referenceCapBytes);
        out.frameSetHash = frameSet.hash;
        out.fullFrame = frameSet.fullFrame;
        out.extent = candidate.extent;
        out.layout = layout;
        out.stages = selection.stages;
        out.stageCount = selection.stageCount;

        for (std::size_t stageIndex = 0; stageIndex < sourceStageCount; ++stageIndex) {
            const std::uint64_t sampleHash = sourceStages[stageIndex]->sample.hash;
            std::size_t spectrumIndex = out.uniqueSpectrumCount;
            for (std::size_t keyIndex = 0; keyIndex < out.uniqueSpectrumCount; ++keyIndex) {
                const auto& key = out.spectrumKeys[keyIndex];
                if (key.sampleHash == sampleHash && key.extent == out.extent &&
                    key.layoutSchema == kDiffusionPlanLayoutSchemaVersion) {
                    spectrumIndex = keyIndex;
                    break;
                }
            }
            if (spectrumIndex == out.uniqueSpectrumCount) {
                auto& key = out.spectrumKeys[out.uniqueSpectrumCount++];
                key.sampleHash = sampleHash;
                key.extent = out.extent;
                key.layoutSchema = kDiffusionPlanLayoutSchemaVersion;
                key.hash = hash_spectrum_key(sampleHash, out.extent);
                if (key.hash == 0) {
                    fail(diagnostic, "spectrum_key_hash");
                    out = {};
                    return false;
                }
            }
            out.stages[stageIndex].spectrumKeyIndex = spectrumIndex;
        }

        out.planKey.profileDigest = out.profileDigest;
        out.planKey.extent = out.extent;
        out.planKey.layoutSchema = kDiffusionPlanLayoutSchemaVersion;
        out.planKey.hash = hash_plan_key(out.profileDigest, out.extent);
        out.transformBufferBytes = candidate.transformBytes;
        std::uint64_t spectrumPlaneCount = 0;
        if (!checked_multiply(
                static_cast<std::uint64_t>(out.uniqueSpectrumCount),
                3,
                spectrumPlaneCount) ||
            !checked_multiply(
                spectrumPlaneCount,
                out.transformBufferBytes,
                out.spectrumPackageBytes)) {
            fail(diagnostic, "spectrum_package_bytes");
            out = {};
            return false;
        }
        out.stagePlaneBytes = frameSet.workspaceBytes;
        out.acceptedR2cWorkBytes = candidate.maximumR2cWorkBytes;
        out.acceptedC2rWorkBytes = candidate.maximumC2rWorkBytes;
        out.acceptedSharedWorkBytes = candidate.acceptedSharedWorkBytes;
        out.acceptedPlanAllowanceBytes = candidate.acceptedPlanAllowanceBytes;
        out.hash = hash_execution_descriptor(out);
        if (out.planKey.hash == 0 || out.hash == 0) {
            fail(diagnostic, "execution_hash");
            out = {};
            return false;
        }
        return true;
    }

    bool diffusion_plan_work_fits(
        const DiffusionExecutionDescriptor& descriptor,
        std::uint64_t actualR2cWorkBytes,
        std::uint64_t actualC2rWorkBytes,
        std::string& diagnostic) {
        diagnostic.clear();
        if (actualR2cWorkBytes > descriptor.acceptedR2cWorkBytes) {
            fail(diagnostic, "actual_r2c_work_bytes");
            return false;
        }
        if (actualC2rWorkBytes > descriptor.acceptedC2rWorkBytes) {
            fail(diagnostic, "actual_c2r_work_bytes");
            return false;
        }
        if (std::max(actualR2cWorkBytes, actualC2rWorkBytes) >
            descriptor.acceptedSharedWorkBytes) {
            fail(diagnostic, "actual_shared_work_bytes");
            return false;
        }
        return true;
    }

} // namespace Spektrafilm
