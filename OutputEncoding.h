#pragma once

#include <cstddef>

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

    void applyEncoding(const Params& params, float rgb[3]);
}