#pragma once

#include <array>
#include <string>

#include "FilmEffectsFrameDescriptors.h"
#include "juicer_legacy_api.h"

namespace JuicerGrain {
    // SF_TEMP_BRIDGE_GRAIN_FRAME: remove at the last native model caller by S6.D;
    // S4.E narrows execution inputs. This projection retains no borrowed curves.
    bool project_input(const Spektrafilm::VisualGrainFrameDescriptorInput&, FjGrainFrameInput&, std::string&);
    bool admit_math(const Spektrafilm::VisualGrainFrameDescriptorInput&, const FjGrainFrameMath&);
    bool accept_result(const Spektrafilm::VisualGrainFrameDescriptorInput&, FjStatus, const FjGrainFrameMath&, const FjGrainFrameFailure&, const FjErrorBuffer&, const std::array<char, 256>&, Spektrafilm::VisualGrainFrameDescriptor&, std::string&);
    bool prepare_frame(const Spektrafilm::VisualGrainFrameDescriptorInput&, Spektrafilm::VisualGrainFrameDescriptor&, std::string&);
} // namespace JuicerGrain
