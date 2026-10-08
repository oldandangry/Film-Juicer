#pragma once

#include <array>
#include <string>

namespace Spectral {
    struct ReconstructionLut;
}
struct FilmRawRecipe;
namespace Spektrafilm {
    struct FilmFoundationBuildInput;
}

// FJ_TEMP_BRIDGE: native reference/sensitivity value binding; remove S4.E.
// Input owners remain live for each synchronous call. Only complete Rust values
// bind into native owners; this edge owns no mathematics, TC identity or cache.
namespace JuicerExposure {
    bool reference_white(const Spectral::ReconstructionLut& spectra, float spectralBlur, const std::array<float, 3>& whiteXyz, std::array<float, 81>& out, std::string& diagnostic);
    bool prepare_sensitivity(const std::array<std::array<float, 3>, 81>& linearSensitivity,
                             FilmRawRecipe& recipe,
                             const Spektrafilm::FilmFoundationBuildInput& input);
} // namespace JuicerExposure
