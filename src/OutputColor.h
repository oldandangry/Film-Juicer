#pragma once

#include <cstddef>
#include <cstdint>

namespace OutputEncoding {

    enum class ColorSpace {
        sRGB = 0,
        DCI_P3,
        DisplayP3,
        AdobeRGB,
        ITU_R_BT2020,
        ProPhotoRGB,
        ACES2065_1,
        DaVinciWideGamutIntermediate,
        Rec709,
        Count
    };

    struct Params {
        ColorSpace colorSpace = ColorSpace::sRGB;
        bool applyCctfEncoding = true;
        bool preserveLinearRange = false;
        bool inputIsOutputSpace = false;
    };

    struct Matrix3x3 {
        float m[9];
    };

    inline constexpr const char* kColorSpaceLabels[] = {
        "sRGB",
        "DCI-P3",
        "Display P3",
        "Adobe RGB (1998)",
        "ITU-R BT.2020",
        "ProPhoto RGB",
        "ACES2065-1",
        "DaVinci Wide Gamut Intermediate",
        "Rec. 709"
    };

    inline constexpr std::size_t kColorSpaceCount = static_cast<std::size_t>(ColorSpace::Count);

    inline constexpr const char* labelFor(ColorSpace cs) {
        const std::size_t idx = static_cast<std::size_t>(cs);
        return idx < kColorSpaceCount ? kColorSpaceLabels[idx] : "sRGB";
    }

    inline constexpr int toIndex(ColorSpace cs) {
        return static_cast<int>(cs);
    }

    inline constexpr ColorSpace colorSpaceFromIndex(int index) {
        if (index < 0) return ColorSpace::sRGB;
        const int maxIndex = static_cast<int>(ColorSpace::Count) - 1;
        if (index > maxIndex) return ColorSpace::sRGB;
        return static_cast<ColorSpace>(index);
    }

    Matrix3x3 dwg_to_output_matrix(ColorSpace cs);
    void applyEncoding(const Params& params, float rgb[3]);
    void applyEncoding(const Params& params, double rgb[3]);
}

namespace GeneratedColorSpaces {

    enum class CctfKind {
        Linear = 0,
        Gamma,
        SRGB,
        BT2020,
        ProPhoto,
        DaVinciIntermediate
    };

    struct CctfParams {
        CctfKind kind = CctfKind::Linear;
        float gamma = 1.0f;
        float a = 0.0f;
        float b = 0.0f;
        float c = 0.0f;
        float d = 0.0f;
        float linearCutoff = 0.0f;
    };

    struct ColorSpaceEntry {
        const char* name = nullptr;
        float primaries[3][2]{ {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f} };
        float whiteXY[2]{ 0.0f, 0.0f };
        float whiteXYZ[3]{ 0.0f, 0.0f, 0.0f };
        float rgbToXyz[9]{ 0.0f };
        float xyzToRgb[9]{ 0.0f };
        CctfParams cctf{};
        std::uint64_t hash = 0;
    };

    // Lookup helpers (entries are stable and generated; returned references are lifetime static)
    const ColorSpaceEntry& get(OutputEncoding::ColorSpace cs);
    const ColorSpaceEntry& getDWG();

} // namespace GeneratedColorSpaces
