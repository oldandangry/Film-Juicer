#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>

#include "ProfileAssets.h"

struct ScatterHalationRawControls {
    bool active = false;
    double scatterAmount = 1.0;
    double scatterSpatialScale = 1.0;
    double halationAmount = 1.0;
    double halationSpatialScale = 1.0;
};

struct ScatterHalationControls {
    bool active = false;
    float scatterAmount = 0.0f;
    float scatterSpatialScale = 0.0f;
    float halationAmount = 0.0f;
    float halationSpatialScale = 0.0f;
};

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
    using ::ScatterHalationControls;
    using ::ScatterHalationFrameDescriptor;
    using ::ScatterHalationGaussianDescriptor;
    using ::ScatterHalationGaussianKind;
    using ::ScatterHalationOpticsRecipe;
    using ::ScatterHalationRawControls;

    inline constexpr std::array<float, 3> kScatterHalationTailWeights{{0.78f,
                                                                       0.65f,
                                                                       0.67f}};
    inline constexpr std::array<float, 3> kScatterHalationExponentialAmplitudes{{0.1633f,
                                                                                 0.6496f,
                                                                                 0.1870f}};
    inline constexpr std::array<float, 3> kScatterHalationBounceWeights{{4.0f / 7.0f,
                                                                         2.0f / 7.0f,
                                                                         1.0f / 7.0f}};

    bool build_scatter_halation_controls(
        const ScatterHalationRawControls& raw,
        ScatterHalationControls& out,
        std::string& outDiagnostic);

    bool resolve_scatter_halation_recipe(
        const ScatterHalationControls& controls,
        const Profiles::ProfileDigest& profileDigest,
        ScatterHalationOpticsRecipe& out,
        std::string& outDiagnostic);

    bool build_scatter_halation_frame_descriptor(
        const ScatterHalationOpticsRecipe& recipe,
        float pixelSizeUm,
        std::optional<ScatterHalationFrameDescriptor>& out,
        std::string& outDiagnostic);
} // namespace Spektrafilm
