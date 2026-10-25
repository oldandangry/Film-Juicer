#pragma once

#include <optional>

#include "ScatterHalation.h"

namespace JuicerCuda {
    struct Failure;
}

namespace JuicerHalation {
    // SF_TEMP_BRIDGE_HALATION_FRAME: remove with the final native frame-preparation
    // caller, no later than S6.D. S4.E retains separate state-integration work.
    bool prepare_frame(const ScatterHalationOpticsRecipe&, float pixelSizeUm, std::optional<ScatterHalationFrameDescriptor>&, JuicerCuda::Failure&);
} // namespace JuicerHalation
