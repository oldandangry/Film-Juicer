#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace Spektrafilm {

    inline constexpr std::uint32_t kDiffusionHostBehaviorSchemaVersion = 1;

    enum class DiffusionFilterFamily : std::uint8_t {
        Glimmerglass,
        BlackProMist,
        ProMist,
        Cinebloom
    };

    struct DiffusionFilterResolvedParameters {
        bool active = false;
        DiffusionFilterFamily family = DiffusionFilterFamily::BlackProMist;
        double scatterFraction = 0.0;
        std::array<double, 3> groupWeightsCoreHaloBloom{};
        std::array<double, 3> groupCenterLambdaUm{};
        double effectiveWarmth = 0.0;
        double spatialScale = 0.0;
        std::uint64_t hash = 0;
    };

    struct DiffusionPsfComponents {
        static constexpr std::size_t kCount = 9;
        std::array<double, kCount> lambdaPixels{};
        std::array<double, kCount> redWeights{};
        std::array<double, kCount> greenWeights{};
        std::array<double, kCount> blueWeights{};
    };

    struct DiffusionPsfSampleDescriptor {
        DiffusionPsfComponents components;
        int radiusPixels = 0;
        std::uint64_t hash = 0;
    };

} // namespace Spektrafilm
