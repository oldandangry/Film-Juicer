#pragma once

#include <array>
#include <cstdint>

#include "juicer_test_api.h"

namespace OutputEncoding {
    enum class ColorSpace : std::uint8_t;
}

namespace Spectral {
    // Independent input tags checked against the fixture ABI in color_fixture.cpp.
    enum class InputColorSpace : std::uint8_t {
        DaVinciWideGamut = 0,
        ITU_R_BT2020,
        ACES2065_1,
        SRGB_Rec709,
        Count
    };

    constexpr int input_color_space_to_index(InputColorSpace cs) {
        return static_cast<int>(cs);
    }

    constexpr InputColorSpace input_color_space_from_index(int index) {
        if (index < 0) {
            return InputColorSpace::DaVinciWideGamut;
        }
        const int maxIndex = static_cast<int>(InputColorSpace::Count) - 1;
        if (index > maxIndex) {
            return InputColorSpace::DaVinciWideGamut;
        }
        return static_cast<InputColorSpace>(index);
    }

    struct Mat3 {
        float m[9];
    };

    struct ChromaticAdaptationWhites {
        const float* source = nullptr;
        const float* destination = nullptr;
    };
} // namespace Spectral

namespace JuicerTestFixture::Color {
    FjOutputColorPreparation output_color(OutputEncoding::ColorSpace space);
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
} // namespace JuicerTestFixture::Color
