#pragma once
#include <array>
#include <string>
#include "FilmEffectsFrameDescriptors.h"
#include "juicer_test_api.h"

namespace Spektrafilm {
    struct FilmJuicerEffectsFrameDescriptorInput final {
        const FilmJuicerEffectsRecipe* recipe = nullptr;
        FilmJuicerEffectsFrameExtent renderExtent{};
        FilmJuicerEffectsFrameExtent fullFrameExtent{};
        float pixelSizeUm = 0.0f;
        double frameTime = 0.0;
        double frameRate = 0.0;
        std::uint64_t sessionSeed = 0;
        std::uint64_t clipToken = 0;
        FilmJuicerEffectsGeometry geometry{};
        float filmFormatLongEdgeMm = 0.0f;
    };

    bool build_film_juicer_effects_frame_descriptor(
        const FilmJuicerEffectsFrameDescriptorInput& input,
        FilmJuicerEffectsFrameDescriptor& out,
        std::string& diagnostic);

} // namespace Spektrafilm

namespace JuicerEffects {
    bool project_input(const Spektrafilm::FilmJuicerEffectsFrameDescriptorInput&, FjEffectsFrameInput&, std::string&);
    bool accept_result(const Spektrafilm::FilmJuicerEffectsFrameDescriptorInput&, FjStatus, const FjEffectsFrameMath&, const FjEffectsFrameFailure&, const FjErrorBuffer&, const std::array<char, 256>&, Spektrafilm::FilmJuicerEffectsFrameDescriptor&, std::string&);
    bool prepare_frame(const Spektrafilm::FilmJuicerEffectsFrameDescriptorInput&, Spektrafilm::FilmJuicerEffectsFrameDescriptor&, std::string&);
} // namespace JuicerEffects
