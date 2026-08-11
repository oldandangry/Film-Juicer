#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "RenderRecipe.h"

namespace Spektrafilm {

    struct CandidateExtent {
        int width = 0;
        int height = 0;

        friend bool operator==(const CandidateExtent&, const CandidateExtent&) = default;
    };

    struct PlanLayout {
        int width = 0;
        int height = 0;
        int complexWidth = 0;
        int physicalRealRowFloats = 0;
        int realDistance = 0;
        int complexDistance = 0;
        std::uint64_t transformBytes = 0;

        friend bool operator==(const PlanLayout&, const PlanLayout&) = default;
    };

    namespace DiffusionPlanLayoutDetail {

        inline bool checked_multiply(
            std::int64_t left,
            std::int64_t right,
            std::int64_t& out) noexcept {
            if (left < 0 || right < 0) {
                return false;
            }
            if (left != 0 && right > std::numeric_limits<std::int64_t>::max() / left) {
                return false;
            }
            out = left * right;
            return true;
        }

    } // namespace DiffusionPlanLayoutDetail

    inline bool make_plan_layout(int width, int height, PlanLayout& out) noexcept {
        out = {};
        if (width <= 0 || height <= 0) {
            return false;
        }

        const std::int64_t complexWidth = static_cast<std::int64_t>(width) / 2 + 1;
        std::int64_t physicalRealRowFloats = 0;
        if (!DiffusionPlanLayoutDetail::checked_multiply(
                complexWidth,
                2,
                physicalRealRowFloats)) {
            return false;
        }

        std::int64_t realDistance = 0;
        std::int64_t complexDistance = 0;
        if (!DiffusionPlanLayoutDetail::checked_multiply(
                height,
                physicalRealRowFloats,
                realDistance) ||
            !DiffusionPlanLayoutDetail::checked_multiply(
                height,
                complexWidth,
                complexDistance)) {
            return false;
        }

        constexpr std::int64_t kCufftIntMax = std::numeric_limits<int>::max();
        if (complexWidth > kCufftIntMax ||
            physicalRealRowFloats > kCufftIntMax ||
            realDistance > kCufftIntMax ||
            complexDistance > kCufftIntMax) {
            return false;
        }

        std::int64_t transformBytes = 0;
        if (!DiffusionPlanLayoutDetail::checked_multiply(
                realDistance,
                static_cast<std::int64_t>(sizeof(float)),
                transformBytes)) {
            return false;
        }

        out = {
            width,
            height,
            static_cast<int>(complexWidth),
            static_cast<int>(physicalRealRowFloats),
            static_cast<int>(realDistance),
            static_cast<int>(complexDistance),
            static_cast<std::uint64_t>(transformBytes)};
        return true;
    }

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
        std::uint64_t hash = 0;
    };

    struct DiffusionPlanKey {
        CandidateExtent extent;
        std::uint64_t hash = 0;
    };

    struct DiffusionExecutionDescriptor {
        std::uint64_t contextEpoch = 0;
        std::uint64_t frameSetHash = 0;
        PlanLayout layout;
        std::array<DiffusionStageTileGeometry, 2> stages{};
        std::array<DiffusionSpectrumKey, 2> spectrumKeys{};
        std::size_t stageCount = 0;
        std::size_t uniqueSpectrumCount = 0;
        DiffusionPlanKey planKey;
        std::uint64_t stagePlaneBytes = 0;
        std::uint64_t reservedSharedWorkBytes = 0;
        std::uint64_t planAllowanceBytes = 0;
        std::uint64_t hash = 0;
    };

    bool build_diffusion_execution_descriptor(
        const DiffusionFrameSetDescriptor& frameSet,
        std::uint64_t contextEpoch,
        std::uint64_t resolvedDeviceCapBytes,
        DiffusionExecutionDescriptor& out,
        std::string& diagnostic);

} // namespace Spektrafilm
