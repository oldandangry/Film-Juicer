#pragma once

#include <array>
#include <cstdint>

namespace Spectral {
    enum class InputColorSpace : std::uint8_t;
    struct Mat3;
    struct ChromaticAdaptationWhites;
} // namespace Spectral

// FJ_TEMP_BRIDGE: host color preparation; remove S4.E.
namespace JuicerColor {
    struct InputMatrices {
        std::array<float, 9> rgbToXyz;
        std::array<float, 3> nominalWhiteXYZ;
        std::array<float, 9> xyzToLinearSrgb;
        std::array<float, 3> d65WhiteXYZ;
    };
    InputMatrices input_matrices(Spectral::InputColorSpace space);
    std::array<float, 3> linear_srgb_to_xyz(const std::array<float, 3>& rgb);
    std::array<float, 3> project_linear_rgb_to_xyz(const std::array<float, 3>& rgb, const Spectral::Mat3& rgbToXyz, const Spectral::Mat3& xyzAdapt);
    std::array<float, 9> cat16_matrix(const Spectral::ChromaticAdaptationWhites& whites);
    std::array<float, 3> adapt_cat16(const std::array<float, 3>& xyz, const Spectral::ChromaticAdaptationWhites& whites);
    std::array<float, 9> cat02_matrix(const Spectral::ChromaticAdaptationWhites& whites);
} // namespace JuicerColor
