// ColorTransforms.h
// Color space transformations, chromatic adaptation, and input color space handling
#pragma once

#include <cstddef>
#include <cstdint>

#include "SpectralData.h"
#include "RustColorBridge.h"

namespace Spectral {

    // ============================================================================
    // Input Color Space (consolidated from ColorSpaces.h)
    // ============================================================================

    enum class InputColorSpace : std::uint8_t {
        DaVinciWideGamut = 0,
        ITU_R_BT2020,
        ACES2065_1,
        SRGB_Rec709,
        Count
    };

    inline constexpr const char* kInputColorSpaceLabels[] = {
        "DaVinci Wide Gamut",
        "ITU-R BT.2020",
        "ACES2065-1",
        "sRGB / Rec.709"};

    inline constexpr std::size_t kInputColorSpaceCount = static_cast<std::size_t>(InputColorSpace::Count);

    constexpr int inputColorSpaceToIndex(InputColorSpace cs) {
        return static_cast<int>(cs);
    }

    constexpr InputColorSpace inputColorSpaceFromIndex(int index) {
        if (index < 0) {
            return InputColorSpace::DaVinciWideGamut;
        }
        const int maxIndex = static_cast<int>(InputColorSpace::Count) - 1;
        if (index > maxIndex) {
            return InputColorSpace::DaVinciWideGamut;
        }
        return static_cast<InputColorSpace>(index);
    }

    // ============================================================================
    // Matrix Operations
    // ============================================================================

    struct Mat3 {
        float m[9];
    };

    inline Mat3 make_identity_mat3(float diag = 1.0f) {
        return Mat3{{diag, 0.0f, 0.0f, 0.0f, diag, 0.0f, 0.0f, 0.0f, diag}};
    }

    // ============================================================================
    // Chromatic Adaptation
    // ============================================================================

    struct ChromaticAdaptationWhites {
        const float* source = nullptr;
        const float* destination = nullptr;
    };

    // ============================================================================
    // Film Raw Input
    // ============================================================================

    struct FilmRawConfig {
        InputColorSpace inputColorSpace = InputColorSpace::DaVinciWideGamut;
        bool applyCctfDecoding = false;
        // Builders bind all three matrices before consuming this unready carrier.
        Mat3 inputRGBToXYZ = make_identity_mat3();
        Mat3 inputXYZAdapt = make_identity_mat3();
        Mat3 xyzToLinearSrgb = make_identity_mat3();
        bool applyInputChromaticAdapt = false;
        SpectralUpsamplingMode spectralUpsamplingMode = SpectralUpsamplingMode::PreferHanatos;
        float midgrayScale = 1.0f;
        float midgrayDWG[3] = {0.184f, 0.184f, 0.184f};
        float rawMidgray[3] = {1.0f, 1.0f, 1.0f};
        float rawMidgrayGreen = 1.0f;
        float inputWhiteXYZ[3] = {
            gDWG_WhitePoint_XYZ[0],
            gDWG_WhitePoint_XYZ[1],
            gDWG_WhitePoint_XYZ[2]};
        float workingWhiteXYZ[3] = {
            gDWG_WhitePoint_XYZ[0],
            gDWG_WhitePoint_XYZ[1],
            gDWG_WhitePoint_XYZ[2]};
        float refIllumWhiteXYZ[3] = {
            gDWG_WhitePoint_XYZ[0],
            gDWG_WhitePoint_XYZ[1],
            gDWG_WhitePoint_XYZ[2]};
        bool hasRefIllumWhite = false;
        bool valid = false;
    };

} // namespace Spectral
