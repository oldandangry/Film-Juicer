#include "DiffusionExecution.h"

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

    struct ExecutionCandidate {
        Spektrafilm::CandidateExtent extent;
        std::uint64_t reservedSharedWorkBytes = 0;
        std::uint64_t planAllowanceBytes = 0;
        std::uint64_t pairMedianNanoseconds = 0;
        std::uint64_t twoFrameBytes = 0;
    };

    struct CandidateSelection {
        std::size_t candidateIndex = 0;
        std::array<Spektrafilm::DiffusionStageTileGeometry, 2> stages{};
        std::array<const Spektrafilm::DiffusionStageFrameDescriptor*, 2>
            sourceStages{};
        std::size_t stageCount = 0;
        std::uint64_t stagePlaneBytes = 0;
    };

    struct ValidatedFrameSet {
        std::array<const Spektrafilm::DiffusionStageFrameDescriptor*, 2> stages{};
        std::size_t stageCount = 0;
        std::uint64_t stagePlaneBytes = 0;
    };

    constexpr std::array<ExecutionCandidate, 10>
        kCandidates{{{{1024, 1024}, 0, 77594624, 32432, 8021745664},
                     {{2048, 2048}, 0, 69206016, 114080, 8219017216},
                     {{3072, 3072}, 0, 69206016, 396288, 8575672320},
                     {{4096, 4096}, 67141632, 69206016, 1088512, 9209217024},
                     {{4608, 4608}, 84971520, 69206016, 2892800, 9547984896},
                     {{5120, 5120}, 104898560, 69206016, 3808768, 9926598656},
                     {{6144, 6144}, 151044096, 69206016, 6086144, 10803363840},
                     {{8192, 6144}, 201375744, 69206016, 5111808, 11797561344},
                     {{6144, 8192}, 201392128, 69206016, 7418880, 11797921792},
                     {{8192, 8192}, 268500992, 69206016, 5972480, 13274316800}}};

    void fail(std::string& diagnostic, std::string_view field) {
        diagnostic = "InvalidDiffusionExecutionDescriptor field=";
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

    void hash_extent(
        std::uint64_t& hash,
        const Spektrafilm::CandidateExtent& extent) noexcept {
        hash_u32_le(hash, static_cast<std::uint32_t>(extent.width));
        hash_u32_le(hash, static_cast<std::uint32_t>(extent.height));
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
        Spektrafilm::DiffusionLinearStage expectedStage,
        std::string& diagnostic) {
        if (stage.stage != expectedStage) {
            fail(diagnostic, "stage_order");
            return false;
        }
        if (stage.hash == 0) {
            fail(diagnostic, "stage_hash");
            return false;
        }
        if (!std::isfinite(stage.scatterFraction) ||
            stage.scatterFraction <= 0.0 || stage.scatterFraction > 1.0) {
            fail(diagnostic, "stage_scatter_fraction");
            return false;
        }
        if (!std::isfinite(stage.sample.pixelSizeUm) ||
            stage.sample.pixelSizeUm <= 0.0) {
            fail(diagnostic, "stage_pixel_size_um");
            return false;
        }
        if (stage.sample.radiusPixels <= 0 || stage.sample.hash == 0) {
            fail(diagnostic, "stage_sample");
            return false;
        }
        return true;
    }

    bool validate_frame_set(
        const Spektrafilm::DiffusionFrameSetDescriptor& frameSet,
        ValidatedFrameSet& out,
        std::string& diagnostic) {
        out = {};
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
        if (!frameSet.camera && !frameSet.enlarger) {
            fail(diagnostic, "active_stage_count");
            return false;
        }
        if (frameSet.enlarger && !Spektrafilm::scan_route_is_print(frameSet.route)) {
            fail(diagnostic, "enlarger_route");
            return false;
        }

        if (frameSet.camera) {
            if (!validate_stage(
                    *frameSet.camera,
                    Spektrafilm::DiffusionLinearStage::CameraFilmLinear,
                    diagnostic)) {
                return false;
            }
            out.stages[out.stageCount++] = &*frameSet.camera;
        }
        if (frameSet.enlarger) {
            if (!validate_stage(
                    *frameSet.enlarger,
                    Spektrafilm::DiffusionLinearStage::EnlargerPrintLinear,
                    diagnostic)) {
                return false;
            }
            out.stages[out.stageCount++] = &*frameSet.enlarger;
        }

        if (!checked_multiply(
                static_cast<std::uint64_t>(frameSet.fullFrame.width),
                static_cast<std::uint64_t>(frameSet.fullFrame.height),
                out.stagePlaneBytes) ||
            !checked_multiply(out.stagePlaneBytes, 4, out.stagePlaneBytes) ||
            !checked_multiply(
                out.stagePlaneBytes,
                sizeof(float),
                out.stagePlaneBytes)) {
            fail(diagnostic, "stage_plane_bytes");
            return false;
        }
        return true;
    }

    bool validate_candidate(
        const ExecutionCandidate& candidate,
        Spektrafilm::PlanLayout& layout,
        std::string& diagnostic) {
        if (!Spektrafilm::make_plan_layout(
                candidate.extent.width,
                candidate.extent.height,
                layout)) {
            fail(diagnostic, "candidate_transform_layout");
            return false;
        }
        if (candidate.pairMedianNanoseconds == 0 ||
            candidate.twoFrameBytes == 0) {
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
        const ExecutionCandidate& candidate,
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
                    static_cast<std::uint64_t>(stage.sample.radiusPixels),
                    2,
                    border)) {
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
                stage.sample.radiusPixels,
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
                layout.transformBytes,
                out.memoryBytes) ||
            !checked_add(
                out.memoryBytes,
                candidate.reservedSharedWorkBytes,
                out.memoryBytes) ||
            !checked_add(
                out.memoryBytes,
                candidate.planAllowanceBytes,
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
        hash_u64_le(hash, sampleHash);
        hash_extent(hash, extent);
        return hash;
    }

    std::uint64_t hash_plan_key(
        const Spektrafilm::CandidateExtent& extent) noexcept {
        std::uint64_t hash = kFnvOffset;
        hash_tag(hash, "diffusion-plan-key");
        hash_extent(hash, extent);
        return hash;
    }

    std::uint64_t hash_execution_descriptor(
        const Spektrafilm::DiffusionExecutionDescriptor& descriptor) noexcept {
        std::uint64_t hash = kFnvOffset;
        hash_tag(hash, "diffusion-execution-descriptor");
        hash_u64_le(hash, descriptor.contextEpoch);
        hash_u64_le(hash, descriptor.frameSetHash);
        hash_layout(hash, descriptor.layout);
        hash_u64_le(hash, static_cast<std::uint64_t>(descriptor.stageCount));
        for (std::size_t index = 0; index < descriptor.stageCount; ++index) {
            const auto& geometry = descriptor.stages[index];
            hash_byte(hash, static_cast<std::uint8_t>(geometry.stage));
            hash_u64_le(hash, geometry.stageDescriptorHash);
            hash_double(hash, geometry.scatterFraction);
            hash_u64_le(
                hash,
                static_cast<std::uint64_t>(geometry.spectrumKeyIndex));
            hash_u32_le(
                hash,
                static_cast<std::uint32_t>(geometry.radiusPixels));
            hash_u32_le(
                hash,
                static_cast<std::uint32_t>(geometry.validTileWidth));
            hash_u32_le(
                hash,
                static_cast<std::uint32_t>(geometry.validTileHeight));
            hash_u32_le(
                hash,
                static_cast<std::uint32_t>(geometry.tileCountX));
            hash_u32_le(
                hash,
                static_cast<std::uint32_t>(geometry.tileCountY));
        }
        hash_u64_le(hash, static_cast<std::uint64_t>(descriptor.uniqueSpectrumCount));
        for (std::size_t index = 0; index < descriptor.uniqueSpectrumCount; ++index) {
            const auto& key = descriptor.spectrumKeys[index];
            hash_u64_le(hash, key.sampleHash);
            hash_extent(hash, key.extent);
            hash_u64_le(hash, key.hash);
        }
        hash_u64_le(hash, descriptor.planKey.hash);
        hash_u64_le(hash, descriptor.stagePlaneBytes);
        hash_u64_le(hash, descriptor.reservedSharedWorkBytes);
        hash_u64_le(hash, descriptor.planAllowanceBytes);
        return hash;
    }

} // namespace

namespace Spektrafilm {

    static bool select_candidate(
        const std::array<ExecutionCandidate, 10>& candidates,
        const DiffusionFrameSetDescriptor& frameSet,
        std::uint64_t resolvedDeviceCapBytes,
        CandidateSelection& out,
        std::string& diagnostic) {
        out = {};
        diagnostic.clear();

        ValidatedFrameSet validated{};
        if (!validate_frame_set(frameSet, validated, diagnostic)) {
            return false;
        }

        std::array<EvaluatedCandidate, 10> evaluated{};
        std::size_t evaluatedCount = 0;
        for (std::size_t index = 0; index < candidates.size(); ++index) {
            const auto& candidate = candidates[index];
            if (candidate.twoFrameBytes >= resolvedDeviceCapBytes) {
                continue;
            }
            EvaluatedCandidate value{};
            if (!evaluate_candidate(
                    candidate,
                    index,
                    frameSet,
                    validated.stages,
                    validated.stageCount,
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
        out.sourceStages = validated.stages;
        out.stageCount = selected->stageCount;
        out.stagePlaneBytes = validated.stagePlaneBytes;
        return true;
    }

    // Both integer inputs are independently named and validated at this narrow boundary.
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    bool build_diffusion_execution_descriptor(
        const DiffusionFrameSetDescriptor& frameSet,
        std::uint64_t contextEpoch,
        std::uint64_t resolvedDeviceCapBytes,
        DiffusionExecutionDescriptor& out,
        std::string& diagnostic) {
        out = {};
        diagnostic.clear();
        if (contextEpoch == 0) {
            fail(diagnostic, "context_epoch");
            return false;
        }

        CandidateSelection selection{};
        if (!select_candidate(
                kCandidates,
                frameSet,
                resolvedDeviceCapBytes,
                selection,
                diagnostic)) {
            return false;
        }
        const auto& candidate = kCandidates[selection.candidateIndex];
        PlanLayout layout{};
        if (!make_plan_layout(candidate.extent.width, candidate.extent.height, layout)) {
            fail(diagnostic, "selected_plan_layout");
            return false;
        }

        out.contextEpoch = contextEpoch;
        out.frameSetHash = frameSet.hash;
        out.layout = layout;
        out.stages = selection.stages;
        out.stageCount = selection.stageCount;

        for (std::size_t stageIndex = 0; stageIndex < selection.stageCount; ++stageIndex) {
            const std::uint64_t sampleHash =
                selection.sourceStages[stageIndex]->sample.hash;
            std::size_t spectrumIndex = out.uniqueSpectrumCount;
            for (std::size_t keyIndex = 0; keyIndex < out.uniqueSpectrumCount; ++keyIndex) {
                const auto& key = out.spectrumKeys[keyIndex];
                if (key.sampleHash == sampleHash && key.extent == candidate.extent) {
                    spectrumIndex = keyIndex;
                    break;
                }
            }
            if (spectrumIndex == out.uniqueSpectrumCount) {
                auto& key = out.spectrumKeys[out.uniqueSpectrumCount++];
                key.sampleHash = sampleHash;
                key.extent = candidate.extent;
                key.hash = hash_spectrum_key(sampleHash, candidate.extent);
                if (key.hash == 0) {
                    fail(diagnostic, "spectrum_key_hash");
                    out = {};
                    return false;
                }
            }
            out.stages[stageIndex].spectrumKeyIndex = spectrumIndex;
        }

        out.planKey.extent = candidate.extent;
        out.planKey.hash = hash_plan_key(candidate.extent);
        out.stagePlaneBytes = selection.stagePlaneBytes;
        out.reservedSharedWorkBytes = candidate.reservedSharedWorkBytes;
        out.planAllowanceBytes = candidate.planAllowanceBytes;
        out.hash = hash_execution_descriptor(out);
        if (out.planKey.hash == 0 || out.hash == 0) {
            fail(diagnostic, "execution_hash");
            out = {};
            return false;
        }
        return true;
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

} // namespace Spektrafilm
