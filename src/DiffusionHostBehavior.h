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

    struct DiffusionFilterAuthoredControls {
        bool active = false;
        DiffusionFilterFamily family = DiffusionFilterFamily::BlackProMist;
        double strength = 0.5;
        double spatialScale = 1.0;
        double haloWarmth = 0.0;
        double coreIntensity = 1.0;
        double coreSize = 1.0;
        double haloIntensity = 1.0;
        double haloSize = 1.0;
        double bloomIntensity = 1.0;
        double bloomSize = 1.0;
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

    std::uint64_t hash_diffusion_authored_controls(
        const DiffusionFilterAuthoredControls& controls);
    bool resolve_diffusion_filter(
        const DiffusionFilterAuthoredControls& authored,
        DiffusionFilterResolvedParameters& out,
        std::string& diagnostic);
    bool build_diffusion_psf_sample_descriptor(
        const DiffusionFilterResolvedParameters& resolved,
        double pixelSizeUm,
        int fullFrameWidth,
        int fullFrameHeight,
        DiffusionPsfSampleDescriptor& out,
        std::string& diagnostic);
    bool expand_diffusion_psf_components(
        const DiffusionPsfSampleDescriptor& descriptor,
        DiffusionPsfComponents& out,
        std::string& diagnostic);

} // namespace Spektrafilm
