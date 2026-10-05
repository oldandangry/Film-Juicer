#pragma once

#include <array>

#include "ColorTransforms.h"

// FJ_TEMP_BRIDGE: CAT02/CAT16 host preparation; remove S4.E.
namespace JuicerColor {
    std::array<float, 9> cat16_matrix(const Spectral::ChromaticAdaptationWhites& whites);
    std::array<float, 3> adapt_cat16(const std::array<float, 3>& xyz, const Spectral::ChromaticAdaptationWhites& whites);
    std::array<float, 9> cat02_matrix(const Spectral::ChromaticAdaptationWhites& whites);
    std::array<float, 3> adapt_cat02(const std::array<float, 3>& xyz, const Spectral::ChromaticAdaptationWhites& whites);
} // namespace JuicerColor
