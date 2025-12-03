#include "GeneratedColorSpaces.h"

#include <array>
#include <algorithm>

#include "Hash.h"
#include "OutputEncoding.h"

namespace {

    using Entry = GeneratedColorSpaces::ColorSpaceEntry;
    using CctfKind = GeneratedColorSpaces::CctfKind;
    using OutputColorSpace = OutputEncoding::ColorSpace;

    constexpr float kD65WhiteXY[2] = { 0.3127f, 0.3290f };
    constexpr float kD60WhiteXY[2] = { 0.32168f, 0.33767f };
    constexpr float kD50WhiteXY[2] = { 0.34567f, 0.35850f };
    constexpr float kDCIWhiteXY[2] = { 0.3140f, 0.3510f };

    Entry gDWG = {
        "DaVinci Wide Gamut",
        {
            { 0.8000f, 0.3130f }, // R
            { 0.1682f, 0.9877f }, // G
            { 0.0790f,-0.1155f }  // B
        },
        { kD65WhiteXY[0], kD65WhiteXY[1] },
        { 0.950455f, 1.0f, 1.089058f },
        {
             0.70062239f,  0.14877482f,  0.10105872f,
             0.27411851f,  0.87363190f, -0.14775041f,
            -0.09896291f, -0.13789533f,  1.32591599f
        },
        {
             1.51667204f, -0.28147805f, -0.14696363f,
            -0.46491710f,  1.25142378f,  0.17488461f,
             0.07578536f,  0.08076209f,  0.76034476f
        },
        { CctfKind::Linear, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
        0
    };

    // Order must match OutputEncoding::ColorSpace
    Entry gEntries[static_cast<int>(OutputEncoding::ColorSpace::Count)] = {
        // sRGB
        {
            "sRGB",
            {
                { 0.6400f, 0.3300f },
                { 0.3000f, 0.6000f },
                { 0.1500f, 0.0600f }
            },
            { kD65WhiteXY[0], kD65WhiteXY[1] },
            { 0.950455f, 1.0f, 1.089058f },
            {
                0.4124564f, 0.3575761f, 0.1804375f,
                0.2126729f, 0.7151522f, 0.0721750f,
                0.0193339f, 0.1191920f, 0.9503041f
            },
            {
                 3.2404542f, -1.5371385f, -0.4985314f,
                -0.9692660f,  1.8760108f,  0.0415560f,
                 0.0556434f, -0.2040259f,  1.0572252f
            },
            { CctfKind::SRGB, 2.4f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
            0
        },
        // DCI-P3
        {
            "DCI-P3",
            {
                { 0.6800f, 0.3200f },
                { 0.2650f, 0.6900f },
                { 0.1500f, 0.0600f }
            },
            { kDCIWhiteXY[0], kDCIWhiteXY[1] },
            { 0.89458689f, 1.0f, 0.95441519f },
            {
                0.4451698f, 0.2771344f, 0.1722827f,
                0.2094917f, 0.7215953f, 0.0689131f,
                0.0000000f, 0.0470606f, 0.9073554f
            },
            {
                 2.7253940f, -1.0180030f, -0.4401632f,
                -0.7951680f,  1.6897321f,  0.0226472f,
                 0.0412419f, -0.0876390f,  1.1009294f
            },
            { CctfKind::Gamma, 1.0f / 2.6f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
            0
        },
        // Display P3
        {
            "Display P3",
            {
                { 0.6800f, 0.3200f },
                { 0.2650f, 0.6900f },
                { 0.1500f, 0.0600f }
            },
            { kD65WhiteXY[0], kD65WhiteXY[1] },
            { 0.950455f, 1.0f, 1.089058f },
            {
                0.4865709f, 0.2656676f, 0.1982173f,
                0.2289746f, 0.6917385f, 0.0792869f,
                0.0000000f, 0.0451134f, 1.0439444f
            },
            {
                 2.4934969f, -0.9313836f, -0.4027108f,
                -0.8294889f,  1.7626641f,  0.0236247f,
                 0.0358458f, -0.0761724f,  0.9568845f
            },
            { CctfKind::SRGB, 2.4f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
            0
        },
        // Adobe RGB
        {
            "Adobe RGB (1998)",
            {
                { 0.6400f, 0.3300f },
                { 0.2100f, 0.7100f },
                { 0.1500f, 0.0600f }
            },
            { kD65WhiteXY[0], kD65WhiteXY[1] },
            { 0.950455f, 1.0f, 1.089058f },
            {
                0.5767309f, 0.1855540f, 0.1881852f,
                0.2973769f, 0.6273491f, 0.0752741f,
                0.0270343f, 0.0706872f, 0.9911085f
            },
            {
                 2.0413690f, -0.5649464f, -0.3446944f,
                -0.9692660f,  1.8760108f,  0.0415560f,
                 0.0134474f, -0.1183897f,  1.0154096f
            },
            { CctfKind::Gamma, 1.0f / 2.19921875f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
            0
        },
        // ITU-R BT.2020
        {
            "ITU-R BT.2020",
            {
                { 0.7080f, 0.2920f },
                { 0.1700f, 0.7970f },
                { 0.1310f, 0.0460f }
            },
            { kD65WhiteXY[0], kD65WhiteXY[1] },
            { 0.950455f, 1.0f, 1.089058f },
            {
                0.6369580f, 0.1446169f, 0.1688809f,
                0.2627002f, 0.6779981f, 0.0593017f,
                0.0000000f, 0.0280727f, 1.0609851f
            },
            {
                 1.7166512f, -0.3556708f, -0.2533663f,
                -0.6666844f,  1.6164812f,  0.0157685f,
                 0.0176399f, -0.0427706f,  0.9421031f
            },
            { CctfKind::BT2020, 1.0f, 1.09929681f, 0.01805397f, 0.0f, 0.0f, 0.0f },
            0
        },
        // ProPhoto RGB
        {
            "ProPhoto RGB",
            {
                { 0.7347f, 0.2653f },
                { 0.1596f, 0.8404f },
                { 0.0366f, 0.0001f }
            },
            { kD50WhiteXY[0], kD50WhiteXY[1] },
            { 0.96367000f, 1.0f, 0.82518800f },
            {
                0.7976749f, 0.1351917f, 0.0313534f,
                0.2880402f, 0.7118741f, 0.0000857f,
                0.0000000f, 0.0000000f, 0.8252100f
            },
            {
                 1.3459433f, -0.2556075f, -0.0511118f,
                -0.5445989f,  1.5081673f,  0.0205351f,
                 0.0000000f,  0.0000000f,  1.2118128f
            },
            { CctfKind::ProPhoto, 1.0f / 1.8f, 0.0f, 0.0f, 0.0f, 0.0f, 0.001953125f },
            0
        },
        // ACES2065-1
        {
            "ACES2065-1",
            {
                { 0.7347f, 0.2653f },
                { 0.0000f, 1.0000f },
                { 0.0001f,-0.0770f }
            },
            { kD60WhiteXY[0], kD60WhiteXY[1] },
            { 0.95264608f, 1.0f, 1.00882518f },
            {
                0.9525524f, 0.0000000f, 0.0000937f,
                0.3439664f, 0.7281661f,-0.0721325f,
                0.0000000f, 0.0000000f, 1.0088252f
            },
            {
                 1.0498110f, 0.0000000f, -0.0000974f,
                -0.4959030f, 1.3733130f,  0.0982400f,
                 0.0000000f, 0.0000000f,  0.9912520f
            },
            { CctfKind::Linear, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
            0
        },
        // DaVinci Wide Gamut Intermediate (uses DWG primaries/white; log CCTF)
        {
            "DaVinci Wide Gamut Intermediate",
            {
                { 0.8000f, 0.3130f },
                { 0.1682f, 0.9877f },
                { 0.0790f,-0.1155f }
            },
            { kD65WhiteXY[0], kD65WhiteXY[1] },
            { 0.950455f, 1.0f, 1.089058f },
            {
                 0.70062239f,  0.14877482f,  0.10105872f,
                 0.27411851f,  0.87363190f, -0.14775041f,
                -0.09896291f, -0.13789533f,  1.32591599f
            },
            {
                 1.51667204f, -0.28147805f, -0.14696363f,
                -0.46491710f,  1.25142378f,  0.17488461f,
                 0.07578536f,  0.08076209f,  0.76034476f
            },
            { CctfKind::DaVinciIntermediate, 1.0f, 0.0075f, 7.0f, 0.07329248f, 10.44426855f, 0.00262409f },
            0
        },
        // Rec.709
        {
            "Rec. 709",
            {
                { 0.6400f, 0.3300f },
                { 0.3000f, 0.6000f },
                { 0.1500f, 0.0600f }
            },
            { kD65WhiteXY[0], kD65WhiteXY[1] },
            { 0.950455f, 1.0f, 1.089058f },
            {
                0.4124564f, 0.3575761f, 0.1804375f,
                0.2126729f, 0.7151522f, 0.0721750f,
                0.0193339f, 0.1191920f, 0.9503041f
            },
            {
                 3.2404542f, -1.5371385f, -0.4985314f,
                -0.9692660f,  1.8760108f,  0.0415560f,
                 0.0556434f, -0.2040259f,  1.0572252f
            },
            { CctfKind::Gamma, 1.0f / 2.4f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
            0
        }
    };

    std::uint64_t hash_entry(const Entry& e) {
        std::array<float, 11> baseFloats = {
            e.primaries[0][0], e.primaries[0][1],
            e.primaries[1][0], e.primaries[1][1],
            e.primaries[2][0], e.primaries[2][1],
            e.whiteXY[0], e.whiteXY[1],
            e.whiteXYZ[0], e.whiteXYZ[1], e.whiteXYZ[2]
        };
        const std::uint64_t baseHash = Hash::hash_float_span(baseFloats.data(), baseFloats.size());
        const std::uint64_t rgbToXyzHash = Hash::hash_float_span(e.rgbToXyz, 9);
        const std::uint64_t xyzToRgbHash = Hash::hash_float_span(e.xyzToRgb, 9);
        std::array<float, 6> cctfFloats = {
            e.cctf.gamma, e.cctf.a, e.cctf.b, e.cctf.c, e.cctf.d, e.cctf.linearCutoff
        };
        const std::uint64_t cctfHash = Hash::hash_float_span(cctfFloats.data(), cctfFloats.size());
        const std::uint64_t fields[] = {
            baseHash,
            rgbToXyzHash,
            xyzToRgbHash,
            static_cast<std::uint64_t>(e.cctf.kind),
            cctfHash
        };
        return Hash::hash_bytes(fields, sizeof(fields));
    }

    void ensure_hashes() {
        if (gDWG.hash == 0) {
            gDWG.hash = hash_entry(gDWG);
        }
        for (Entry& e : gEntries) {
            if (e.hash == 0) {
                e.hash = hash_entry(e);
            }
        }
    }

} // namespace

namespace GeneratedColorSpaces {

    const ColorSpaceEntry& get(OutputEncoding::ColorSpace cs) {
        ensure_hashes();
        const int idx = static_cast<int>(cs);
        const int maxIndex = static_cast<int>(OutputEncoding::ColorSpace::Count) - 1;
        if (idx < 0 || idx > maxIndex) {
            return gEntries[0];
        }
        return gEntries[idx];
    }

    const ColorSpaceEntry& getDWG() {
        ensure_hashes();
        return gDWG;
    }

} // namespace GeneratedColorSpaces
