#pragma once

#include <cstdint>

namespace OutputEncoding {
    enum class ColorSpace : int;
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
