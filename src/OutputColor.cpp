#include "OutputColor.h"

#include <algorithm>
#include <array>
#include <cmath>

#include "Hash.h"

namespace {

    using Entry = GeneratedColorSpaces::ColorSpaceEntry;
    using CctfKind = GeneratedColorSpaces::CctfKind;

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
            { 0.95045593f, 1.0f, 1.08905775f },
            {
                0.41240000f, 0.35760000f, 0.18050000f,
                0.21260000f, 0.71520000f, 0.07220000f,
                0.01930000f, 0.11920000f, 0.95050000f
            },
            {
                 3.24060000f, -1.53720000f, -0.49860000f,
                -0.96890000f,  1.87580000f,  0.04150000f,
                 0.05570000f, -0.20400000f,  1.05700000f
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
            { 0.89458689f, 1.0f, 0.95441595f },
            {
                0.44516982f, 0.27713441f, 0.17228267f,
                0.20949168f, 0.72159525f, 0.06891307f,
               -0.00000000f, 0.04706056f, 0.90735539f
            },
            {
                 2.72539403f, -1.01800301f, -0.44016320f,
                -0.79516803f,  1.68973205f,  0.02264719f,
                 0.04124189f, -0.08763902f,  1.10092938f
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
            { 0.95045593f, 1.0f, 1.08905775f },
            {
                0.48657095f, 0.26566769f, 0.19821729f,
                0.22897456f, 0.69173852f, 0.07928691f,
                -0.00000000f, 0.04511338f, 1.04394437f
            },
            {
                 2.49349691f, -0.93138362f, -0.40271078f,
                -0.82948897f,  1.76266406f,  0.02362469f,
                 0.03584583f, -0.07617239f,  0.95688452f
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
            { 0.95045593f, 1.0f, 1.08905775f },
            {
                0.57667000f, 0.18556000f, 0.18823000f,
                0.29734000f, 0.62736000f, 0.07529000f,
                0.02703000f, 0.07069000f, 0.99134000f
            },
            {
                 2.04159000f, -0.56501000f, -0.34473000f,
                -0.96924000f,  1.87597000f,  0.04156000f,
                 0.01344000f, -0.11836000f,  1.01517000f
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
            { 0.95045593f, 1.0f, 1.08905775f },
            {
                0.63695805f, 0.14461690f, 0.16888098f,
                0.26270021f, 0.67799807f, 0.05930172f,
                0.00000000f, 0.02807269f, 1.06098506f
            },
            {
                 1.71665119f, -0.35567078f, -0.25336628f,
                -0.66668435f,  1.61648124f,  0.01576855f,
                 0.01763986f, -0.04277061f, 0.94210312f
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
            { 0.96429568f, 1.0f, 0.82510460f },
            {
                0.79770000f, 0.13520000f, 0.03130000f,
                0.28800000f, 0.71190000f, 0.00010000f,
                0.00000000f, 0.00000000f, 0.82490000f
            },
            {
                 1.34600000f, -0.25560000f, -0.05110000f,
                -0.54460000f,  1.50820000f,  0.02050000f,
                 0.00000000f,  0.00000000f,  1.21230000f
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
            { 0.95264607f, 1.0f, 1.00882518f },
            {
                0.95255240f, 0.00000000f,  0.00009368f,
                0.34396645f, 0.72816610f, -0.07213255f,
                0.00000000f, 0.00000000f, 1.00882518f
            },
            {
                1.04981102f, 0.00000000f, -0.00009748f,
               -0.49590302f, 1.37331305f,  0.09824004f,
                0.00000000f, 0.00000000f,  0.99125202f
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

namespace OutputEncoding {
    namespace {

        struct Mat3 {
            float m[9];
            void mul(const float v[3], float out[3]) const {
                out[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
                out[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
                out[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
            }
        };

        inline Mat3 mat_mul(const float a[9], const float b[9]) {
            Mat3 out{};
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    out.m[r * 3 + c] =
                        a[r * 3 + 0] * b[0 * 3 + c] +
                        a[r * 3 + 1] * b[1 * 3 + c] +
                        a[r * 3 + 2] * b[2 * 3 + c];
                }
            }
            return out;
        }

        inline float clamp01(float v) {
            if (v <= 0.0f) return 0.0f;
            if (v >= 1.0f) return 1.0f;
            return v;
        }

        inline float encode_gamma_signed(float v, float exponent) {
            const float mag = static_cast<float>(std::pow(std::abs(v), exponent));
            return std::copysign(mag, v);
        }

        inline float encode_sRGB(float v) {
            if (v <= 0.0031308f) {
                return 12.92f * v;
            }
            return 1.055f * static_cast<float>(std::pow(v, 1.0f / 2.4f)) - 0.055f;
        }

        inline float encode_BT2020(float v, float a, float b) {
            if (v < b) {
                return v * 4.5f;
            }
            return a * static_cast<float>(std::pow(v, 0.45f)) - (a - 1.0f);
        }

        inline float encode_ProPhoto(float v, float threshold, float exponent) {
            if (v < threshold) {
                return v * 16.0f;
            }
            return static_cast<float>(std::pow(v, exponent));
        }

        inline float encode_DaVinciIntermediate(float v, const GeneratedColorSpaces::CctfParams& cctf) {
            const float linear = std::max(0.0f, v);
            if (linear <= cctf.linearCutoff) {
                return linear * cctf.d;
            }
            return (static_cast<float>(std::log2(linear + cctf.a)) + cctf.b) * cctf.c;
        }

        inline float encode_channel(const GeneratedColorSpaces::CctfParams& cctf, float v) {
            using Kind = GeneratedColorSpaces::CctfKind;
            switch (cctf.kind) {
            case Kind::Linear:
                return v;
            case Kind::Gamma:
                return encode_gamma_signed(v, cctf.gamma);
            case Kind::SRGB:
                return encode_sRGB(v);
            case Kind::BT2020:
                return encode_BT2020(v, cctf.a, cctf.b);
            case Kind::ProPhoto:
                return encode_ProPhoto(v, cctf.linearCutoff, cctf.gamma);
            case Kind::DaVinciIntermediate:
                return encode_DaVinciIntermediate(v, cctf);
            default:
                return clamp01(v);
            }
        }

        inline double clamp01d(double v) {
            if (v <= 0.0) return 0.0;
            if (v >= 1.0) return 1.0;
            return v;
        }

        inline double encode_gammad_signed(double v, double exponent) {
            const double mag = std::pow(std::abs(v), exponent);
            return std::copysign(mag, v);
        }

        inline double encode_sRGBd(double v) {
            if (v <= 0.0031308) {
                return 12.92 * v;
            }
            return 1.055 * std::pow(v, 1.0 / 2.4) - 0.055;
        }

        inline double encode_BT2020d(double v, double a, double b) {
            if (v < b) {
                return v * 4.5;
            }
            return a * std::pow(v, 0.45) - (a - 1.0);
        }

        inline double encode_ProPhotod(double v, double threshold, double exponent) {
            if (v < threshold) {
                return v * 16.0;
            }
            return std::pow(v, exponent);
        }

        inline double encode_DaVinciIntermediated(double v, const GeneratedColorSpaces::CctfParams& cctf) {
            const double linear = std::max(0.0, v);
            if (linear <= static_cast<double>(cctf.linearCutoff)) {
                return linear * static_cast<double>(cctf.d);
            }
            return (std::log2(linear + static_cast<double>(cctf.a)) + static_cast<double>(cctf.b)) * static_cast<double>(cctf.c);
        }

        inline double encode_channel_double(const GeneratedColorSpaces::CctfParams& cctf, double v) {
            using Kind = GeneratedColorSpaces::CctfKind;
            switch (cctf.kind) {
            case Kind::Linear:
                return v;
            case Kind::Gamma:
                return encode_gammad_signed(v, static_cast<double>(cctf.gamma));
            case Kind::SRGB:
                return encode_sRGBd(v);
            case Kind::BT2020:
                return encode_BT2020d(v, static_cast<double>(cctf.a), static_cast<double>(cctf.b));
            case Kind::ProPhoto:
                return encode_ProPhotod(v, static_cast<double>(cctf.linearCutoff), static_cast<double>(cctf.gamma));
            case Kind::DaVinciIntermediate:
                return encode_DaVinciIntermediated(v, cctf);
            default:
                return clamp01d(v);
            }
        }

    } // namespace

    Matrix3x3 dwg_to_output_matrix(ColorSpace cs) {
        const auto& dwg = GeneratedColorSpaces::getDWG();
        const auto& out = GeneratedColorSpaces::get(cs);
        Matrix3x3 result{};
        Mat3 mat = mat_mul(out.xyzToRgb, dwg.rgbToXyz);
        for (int i = 0; i < 9; ++i) {
            result.m[i] = mat.m[i];
        }
        return result;
    }

    void applyEncoding(const Params& params, float rgb[3]) {
        const auto& outSpace = GeneratedColorSpaces::get(params.colorSpace);
        float linear[3];
        if (params.inputIsOutputSpace) {
            linear[0] = rgb[0];
            linear[1] = rgb[1];
            linear[2] = rgb[2];
        }
        else {
            const Matrix3x3 m = dwg_to_output_matrix(params.colorSpace);
            Mat3 tmp{ { m.m[0], m.m[1], m.m[2], m.m[3], m.m[4], m.m[5], m.m[6], m.m[7], m.m[8] } };
            tmp.mul(rgb, linear);
        }

        if (params.preserveLinearRange) {
            rgb[0] = linear[0];
            rgb[1] = linear[1];
            rgb[2] = linear[2];
            return;
        }

        if (params.applyCctfEncoding) {
            rgb[0] = encode_channel(outSpace.cctf, linear[0]);
            rgb[1] = encode_channel(outSpace.cctf, linear[1]);
            rgb[2] = encode_channel(outSpace.cctf, linear[2]);
        }
        else {
            rgb[0] = linear[0];
            rgb[1] = linear[1];
            rgb[2] = linear[2];
        }

        // Single post-encoding clamp (agx parity: encode first, then clip)
        rgb[0] = clamp01(rgb[0]);
        rgb[1] = clamp01(rgb[1]);
        rgb[2] = clamp01(rgb[2]);
    }

    void applyEncoding(const Params& params, double rgb[3]) {
        const auto& outSpace = GeneratedColorSpaces::get(params.colorSpace);
        double linear[3];
        if (params.inputIsOutputSpace) {
            linear[0] = rgb[0];
            linear[1] = rgb[1];
            linear[2] = rgb[2];
        }
        else {
            const Matrix3x3 m = dwg_to_output_matrix(params.colorSpace);
            double mat[9] = {
                static_cast<double>(m.m[0]), static_cast<double>(m.m[1]), static_cast<double>(m.m[2]),
                static_cast<double>(m.m[3]), static_cast<double>(m.m[4]), static_cast<double>(m.m[5]),
                static_cast<double>(m.m[6]), static_cast<double>(m.m[7]), static_cast<double>(m.m[8])
            };
            linear[0] = mat[0] * rgb[0] + mat[1] * rgb[1] + mat[2] * rgb[2];
            linear[1] = mat[3] * rgb[0] + mat[4] * rgb[1] + mat[5] * rgb[2];
            linear[2] = mat[6] * rgb[0] + mat[7] * rgb[1] + mat[8] * rgb[2];
        }

        if (params.preserveLinearRange) {
            rgb[0] = linear[0];
            rgb[1] = linear[1];
            rgb[2] = linear[2];
            return;
        }

        if (params.applyCctfEncoding) {
            rgb[0] = encode_channel_double(outSpace.cctf, linear[0]);
            rgb[1] = encode_channel_double(outSpace.cctf, linear[1]);
            rgb[2] = encode_channel_double(outSpace.cctf, linear[2]);
        }
        else {
            rgb[0] = linear[0];
            rgb[1] = linear[1];
            rgb[2] = linear[2];
        }

        rgb[0] = clamp01d(rgb[0]);
        rgb[1] = clamp01d(rgb[1]);
        rgb[2] = clamp01d(rgb[2]);
    }

} // namespace OutputEncoding
