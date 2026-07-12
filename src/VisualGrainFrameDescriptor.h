#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "RenderRecipe.h"

namespace Spektrafilm {

    enum class VisualGrainScratchShape : std::uint8_t {
        None,
        Streamed,
        StreamedShared,
        StreamedLayers,
        StreamedLayersShared
    };

    struct VisualGrainFrameExtent {
        int x = 0;
        int y = 0;
        int width = 0;
        int height = 0;
    };

    struct VisualGrainGaussian {
        float sigmaPx = 0.0f;
        int radius = 0;
        std::uint64_t hash = 0;
    };

    struct VisualGrainFrameDescriptor {
        bool active = false;
        ProfilePolarity capturePolarity = ProfilePolarity::Unsupported;
        VisualGrainFrameExtent renderExtent{};
        VisualGrainFrameExtent fullFrameExtent{};
        float pixelSizeUm = 0.0f;
        std::int64_t frame0 = 0;
        float frameAlpha = 0.0f;
        std::uint64_t seedBase = 0;
        std::uint64_t seedBaseNext = 0;
        std::uint64_t sessionSeed = 0;
        std::uint64_t clipToken = 0;
        int pitchPx = 0;
        int breathingPeriodFrames = 0;
        int clumpMorphPeriodFrames = 0;
        float wangCellMm = 0.0f;
        float breathingAmplitude = 0.0f;
        float breathingCellUmSmall = 0.0f;
        float breathingCellUmLarge = 0.0f;
        float breathingMix = 0.0f;
        float breathingDriftUmPerFrame = 0.0f;
        float debugScale = 0.0f;
        std::array<float, 3> densityMaxCmy{};
        std::array<float, 3> nParticlesCmy{};
        std::array<float, 3> odParticleCmy{};
        std::array<std::array<float, 3>, 3> densityMinLayers{};
        std::array<std::array<float, 3>, 3> densityMaxLayers{};
        std::array<std::array<float, 3>, 3> nParticlesLayers{};
        std::array<std::array<float, 3>, 3> odParticleLayers{};
        std::array<VisualGrainGaussian, 3> correlation{};
        std::array<std::array<VisualGrainGaussian, 3>, 3> dyeCloud{};
        float effectiveFineWeight = 1.0f;
        float effectiveMidWeight = 0.0f;
        float effectiveCoarseWeight = 0.0f;
        float sizeMixGain = 1.0f;
        VisualGrainScratchShape scratchShape = VisualGrainScratchShape::None;
        bool requiresFullFrame = false;
        std::uint64_t recipeHash = 0;
        std::uint64_t densityCurvesLayersHash = 0;
        std::uint64_t staticNoiseVersion = 0;
        std::uint64_t hash = 0;
    };

    struct VisualGrainFrameDescriptorInput {
        const VisualGrainRecipe* recipe = nullptr;
        const FilmDevelopRecipe* filmDevelop = nullptr;
        ProfilePolarity capturePolarity = ProfilePolarity::Unsupported;
        VisualGrainFrameExtent renderExtent{};
        VisualGrainFrameExtent fullFrameExtent{};
        float pixelSizeUm = 0.0f;
        double frameTime = 0.0;
        double frameRate = 0.0;
        std::uint64_t sessionSeed = 0;
        std::uint64_t clipToken = 0;
        std::uint64_t staticNoiseVersion = 0;
    };

    bool build_visual_grain_frame_descriptor(
        const VisualGrainFrameDescriptorInput& input,
        VisualGrainFrameDescriptor& out,
        std::string& diagnostic);

} // namespace Spektrafilm
