#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>


struct ScatterHalationOpticsRecipe {
    bool scatterActive = false;
    bool backReflectionActive = false;
    float scatterAmount = 0.0f;
    float scatterSpatialScale = 0.0f;
    float halationSpatialScale = 0.0f;
    std::array<float, 3> halationFirstSigmaUm{};
    std::array<float, 3> totalStrength{};
    std::uint64_t hash = 0;
};

enum class ScatterHalationGaussianKind : std::uint8_t {
    Identity,
    FirReflect,
    YvvReplicate
};

struct ScatterHalationGaussianDescriptor {
    ScatterHalationGaussianKind kind = ScatterHalationGaussianKind::Identity;
    int radius = 0;
    std::array<float, 19> firWeights{};
    float B = 0.0f;
    float B1 = 0.0f;
    float B2 = 0.0f;
    float B3 = 0.0f;
};

struct ScatterHalationChannelDescriptor {
    ScatterHalationGaussianDescriptor core;
    std::array<ScatterHalationGaussianDescriptor, 3> tail{};
    std::array<ScatterHalationGaussianDescriptor, 3> bounce{};
    float totalStrength = 0.0f;
};

struct ScatterHalationFrameDescriptor {
    std::uint64_t recipeHash = 0;
    float scatterAmount = 0.0f;
    std::array<ScatterHalationChannelDescriptor, 3> channels{};
};

namespace Spektrafilm {
    using ::ScatterHalationChannelDescriptor;
    using ::ScatterHalationFrameDescriptor;
    using ::ScatterHalationGaussianDescriptor;
    using ::ScatterHalationGaussianKind;
    using ::ScatterHalationOpticsRecipe;

    inline constexpr std::array<float, 3> kScatterHalationTailWeights{{0.78f,
                                                                       0.65f,
                                                                       0.67f}};
    inline constexpr std::array<float, 3> kScatterHalationExponentialAmplitudes{{0.1633f,
                                                                                 0.6496f,
                                                                                 0.1870f}};
    inline constexpr std::array<float, 3> kScatterHalationBounceWeights{{4.0f / 7.0f,
                                                                         2.0f / 7.0f,
                                                                         1.0f / 7.0f}};

} // namespace Spektrafilm
