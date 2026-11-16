#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

inline constexpr std::size_t kHalationTailComponents = 3;
inline constexpr std::size_t kHalationMaxBounces = 5;

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
    YvvReplicate,
    YvvReplicateFactored
};

struct ScatterHalationGaussianDescriptor {
    ScatterHalationGaussianKind kind = ScatterHalationGaussianKind::Identity;
    int radius = 0;
    std::array<float, 19> firWeights{};
    // Legacy: [B, B1, B2, B3]. Factored: [h, g, t, 0].
    std::array<float, 4> yvv{};
};

struct ScatterHalationChannelDescriptor {
    ScatterHalationGaussianDescriptor core;
    std::array<ScatterHalationGaussianDescriptor, kHalationTailComponents> tail{};
    std::array<ScatterHalationGaussianDescriptor, kHalationMaxBounces> bounce{};
    float totalStrength = 0.0f;
    float tailWeight = 0.0f;
};

struct ScatterHalationFrameDescriptor {
    std::uint64_t recipeHash = 0;
    float scatterAmount = 0.0f;
    std::uint32_t bounceCount = 0;
    std::array<float, kHalationMaxBounces> bounceWeights{};
    bool renormalize = false;
    std::array<ScatterHalationChannelDescriptor, 3> channels{};
};

namespace Spektrafilm {
    using ::ScatterHalationChannelDescriptor;
    using ::ScatterHalationFrameDescriptor;
    using ::ScatterHalationGaussianDescriptor;
    using ::ScatterHalationGaussianKind;
    using ::ScatterHalationOpticsRecipe;

    inline constexpr std::array<float, 3> kScatterHalationExponentialAmplitudes{{0.1633f,
                                                                                 0.6496f,
                                                                                 0.1870f}};

} // namespace Spektrafilm
