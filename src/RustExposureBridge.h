#pragma once

#include <array>
#include <span>
#include <string>

namespace Spectral {
    struct ReconstructionLut;
    struct MallettBasis;
    struct FilmRawConfig;
} // namespace Spectral
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
    bool mallett_midgray(const Spectral::MallettBasis& basis,
                         std::span<const float> illuminant,
                         const std::array<std::array<float, 3>, 81>& sensitivity,
                         Spectral::FilmRawConfig& config,
                         std::string& diagnostic);
    bool tc_midgray(float green, Spectral::FilmRawConfig& config, std::string& diagnostic);
    bool reference_source(float exposureEv, float& out, std::string& diagnostic);
    bool mallett_reference_raw(const Spectral::MallettBasis& basis,
                               std::span<const float> illuminant,
                               const std::array<std::array<float, 3>, 81>& sensitivity,
                               float source,
                               float greenScale,
                               std::array<float, 3>& out,
                               std::string& diagnostic);
} // namespace JuicerExposure
