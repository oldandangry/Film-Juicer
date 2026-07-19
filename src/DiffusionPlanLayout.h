#pragma once

#include <array>
#include <cstdint>
#include <limits>

namespace Spektrafilm {

    inline constexpr std::uint32_t kDiffusionPlanLayoutSchemaVersion = 1;

    struct CandidateExtent {
        int width = 0;
        int height = 0;

        friend bool operator==(const CandidateExtent&, const CandidateExtent&) = default;
    };

    inline constexpr std::array<CandidateExtent, 10> kDiffusionCandidateExtents{{{1024, 1024},
                                                                                 {2048, 2048},
                                                                                 {3072, 3072},
                                                                                 {4096, 4096},
                                                                                 {4608, 4608},
                                                                                 {5120, 5120},
                                                                                 {6144, 6144},
                                                                                 {8192, 6144},
                                                                                 {6144, 8192},
                                                                                 {8192, 8192}}};

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

} // namespace Spektrafilm
