#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

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

    struct DiffusionPsfSampleDescriptor {
        DiffusionFilterFamily family = DiffusionFilterFamily::BlackProMist;
        std::array<double, 3> groupWeightsCoreHaloBloom{};
        std::array<double, 3> groupCenterLambdaUm{};
        double effectiveWarmth = 0.0;
        double spatialScale = 0.0;
        double pixelSizeUm = 0.0;
        int radiusPixels = 0;
        std::uint64_t hash = 0;
    };

    struct DiffusionPsfComponents {
        static constexpr std::size_t kCount = 9;

        std::array<double, kCount> lambdaPixels{};
        std::array<double, kCount> redWeights{};
        std::array<double, kCount> greenWeights{};
        std::array<double, kCount> blueWeights{};
    };

    bool expand_diffusion_psf_components(
        const DiffusionPsfSampleDescriptor& descriptor,
        DiffusionPsfComponents& out,
        std::string& diagnostic);

} // namespace Spektrafilm

namespace JuicerCuda {
    struct Failure;
}
namespace JuicerDiffusion {
    bool prepare_components(const Spektrafilm::DiffusionPsfSampleDescriptor&, Spektrafilm::DiffusionPsfComponents&, JuicerCuda::Failure&);
}
