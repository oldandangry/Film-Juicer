#pragma once
#include <array>
#include <string>
#include "FilmEffectsFrameDescriptors.h"
#include "juicer_legacy_api.h"

namespace JuicerEffects {
    // SF_TEMP_BRIDGE_EFFECTS_FRAME: S4.E narrows inputs; remove at the last
    // native model caller, no later than S6.D. No input borrow survives the call.
    bool project_input(const Spektrafilm::FilmJuicerEffectsFrameDescriptorInput&, FjEffectsFrameInput&, std::string&);
    bool accept_result(const Spektrafilm::FilmJuicerEffectsFrameDescriptorInput&, FjStatus, const FjEffectsFrameMath&, const FjEffectsFrameFailure&, const FjErrorBuffer&, const std::array<char, 256>&, Spektrafilm::FilmJuicerEffectsFrameDescriptor&, std::string&);
    bool compose_frame_descriptor(
        const FjEffectsFrameMath&, const FjDust&, const FjScratch&, const FjDust&, const FjScratch&, Spektrafilm::FilmJuicerEffectsFrameDescriptor&, std::string&);
    bool prepare_frame(const Spektrafilm::FilmJuicerEffectsFrameDescriptorInput&, Spektrafilm::FilmJuicerEffectsFrameDescriptor&, std::string&);
} // namespace JuicerEffects
