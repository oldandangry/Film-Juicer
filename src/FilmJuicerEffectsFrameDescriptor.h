#pragma once

#include <cstdint>
#include <string>

#include "RenderRecipe.h"

namespace Spektrafilm {

    struct FilmJuicerEffectsFrameExtent final {
        int x = 0;
        int y = 0;
        int width = 0;
        int height = 0;
    };

    struct FilmJuicerEffectsFrameDescriptor final {
        FilmJuicerEffectsFrameExtent renderExtent{};
        FilmJuicerEffectsFrameExtent fullFrameExtent{};
        float pixelSizeUm = 0.0f;
        std::int64_t frame0 = 0;
        float frameAlpha = 0.0f;
        std::uint64_t sessionSeed = 0;
        std::uint64_t clipToken = 0;
        int pitchPx = 0;
        float filmDustAmount = 0.0f;
        float filmScratchAmount = 0.0f;
        float gateDustAmount = 0.0f;
        float gateScratchAmount = 0.0f;
        bool weaveActive = false;
        float weaveDxPx = 0.0f;
        float weaveDyPx = 0.0f;
        float weaveCosRot = 1.0f;
        float weaveSinRot = 0.0f;
        bool filmActive = false;
        bool gateMaskActive = false;
        bool gateOutputActive = false;
        bool requiresFullFrame = false;
        std::uint64_t recipeHash = 0;
        std::uint64_t hash = 0;
    };

    struct FilmJuicerEffectsFrameDescriptorInput final {
        const FilmJuicerEffectsRecipe* recipe = nullptr;
        FilmJuicerEffectsFrameExtent renderExtent{};
        FilmJuicerEffectsFrameExtent fullFrameExtent{};
        float pixelSizeUm = 0.0f;
        double frameTime = 0.0;
        double frameRate = 0.0;
        std::uint64_t sessionSeed = 0;
        std::uint64_t clipToken = 0;
    };

    bool build_film_juicer_effects_frame_descriptor(
        const FilmJuicerEffectsFrameDescriptorInput& input,
        FilmJuicerEffectsFrameDescriptor& out,
        std::string& diagnostic);

} // namespace Spektrafilm
