// ColorTransforms.h
// Color space transformations, chromatic adaptation, and input color space handling
#pragma once

#include <cmath>
#include <algorithm>
#include <cstddef>
#include "SpectralData.h"

// Forward declarations
namespace Spectral {
    struct SpectralTables;
    struct Curve;
    inline void rgbDWG_to_layerExposures(const float rgbDWG[3], float E[3], float exposureScale);
    inline void rgbDWG_to_layerExposures_from_tables_with_curves(
        const float rgbDWG[3], float E[3], float exposureScale,
        const SpectralTables* tables, const float* S_inv,
        const Curve& sB, const Curve& sG, const Curve& sR,
        SpectralUpsamplingMode spectralUpsamplingMode,
        const float refIllumWhiteXYZ[3]);
    inline void XYZ_to_DWG_linear(const float XYZ[3], float RGB[3]);

#if defined(JUICER_SPD_DEBUG)
    bool spd_probe_begin_capture(const float rgbIn[3], const float rgbDWG[3], bool spdEnabled);
    void spd_probe_finalize(float midgrayScale, const float E_afterMidgray[3]);
#endif
}

namespace Spectral {

    // ============================================================================
    // Input Color Space (consolidated from ColorSpaces.h)
    // ============================================================================

    enum class InputColorSpace {
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
        "sRGB / Rec.709"
    };

    inline constexpr std::size_t kInputColorSpaceCount = static_cast<std::size_t>(InputColorSpace::Count);

    inline constexpr const char* inputColorSpaceLabel(InputColorSpace cs) {
        const std::size_t idx = static_cast<std::size_t>(cs);
        return idx < kInputColorSpaceCount ? kInputColorSpaceLabels[idx] : "DaVinci Wide Gamut";
    }

    inline constexpr int inputColorSpaceToIndex(InputColorSpace cs) {
        return static_cast<int>(cs);
    }

    inline constexpr InputColorSpace inputColorSpaceFromIndex(int index) {
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
        inline void mul(const float v[3], float out[3]) const {
            out[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
            out[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
            out[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
        }
        inline Mat3 inverse(float fallback = 1.0f) const {
            const float det =
                m[0] * (m[4] * m[8] - m[5] * m[7]) -
                m[1] * (m[3] * m[8] - m[5] * m[6]) +
                m[2] * (m[3] * m[7] - m[4] * m[6]);

            const float eps = 1e-6f;
            if (!std::isfinite(det) || std::fabs(det) <= eps) {
                return Mat3{ {
                    fallback, 0.0f,    0.0f,
                    0.0f,    fallback, 0.0f,
                    0.0f,    0.0f,    fallback
                } };
            }

            const float invDet = 1.0f / det;
            Mat3 inv{ {
                (m[4] * m[8] - m[5] * m[7]) * invDet,
                (m[2] * m[7] - m[1] * m[8]) * invDet,
                (m[1] * m[5] - m[2] * m[4]) * invDet,
                (m[5] * m[6] - m[3] * m[8]) * invDet,
                (m[0] * m[8] - m[2] * m[6]) * invDet,
                (m[2] * m[3] - m[0] * m[5]) * invDet,
                (m[3] * m[7] - m[4] * m[6]) * invDet,
                (m[1] * m[6] - m[0] * m[7]) * invDet,
                (m[0] * m[4] - m[1] * m[3]) * invDet
            } };
            return inv;
        }
    };

    inline Mat3 make_identity_mat3(float diag = 1.0f) {
        return Mat3{ {
            diag, 0.0f, 0.0f,
            0.0f, diag, 0.0f,
            0.0f, 0.0f, diag
        } };
    }

    // Canonical RGB�XYZ matrices
    inline constexpr Mat3 kRGB_to_XYZ_DWG = { {
        0.70062239f,  0.14877482f,  0.10105872f,
        0.27411851f,  0.87363190f, -0.14775041f,
       -0.09896291f, -0.13789533f,  1.32591599f
    } };

    inline constexpr Mat3 kRGB_to_XYZ_BT2020 = { {
        0.63695806f,  0.14461690f,  0.16888097f,
        0.26270020f,  0.67799807f,  0.05930172f,
        0.00000000f,  0.02807269f,  1.06098509f
    } };

    inline constexpr Mat3 kRGB_to_XYZ_ACES2065 = { {
        0.95255238f,  0.00000000f,  0.00009368f,
        0.34396645f,  0.72816610f, -0.07213255f,
        0.00000000f,  0.00000000f,  1.00882518f
    } };

    inline constexpr Mat3 kRGB_to_XYZ_sRGB_Rec709 = { {
        0.4124564f, 0.3575761f, 0.1804375f,
        0.2126729f, 0.7151522f, 0.0721750f,
        0.0193339f, 0.1191920f, 0.9503041f
    } };

    // Global DWG�XYZ matrices
    inline Mat3 gDWG_RGB_to_XYZ = { {
        0.70062239f,  0.14877482f,  0.10105872f,
        0.27411851f,  0.87363190f, -0.14775041f,
       -0.09896291f, -0.13789533f,  1.32591599f
    } };
    inline Mat3 gDWG_XYZ_to_RGB = { {
        1.51667204f, -0.28147805f, -0.14696363f,
       -0.46491710f,  1.25142378f,  0.17488461f,
        0.07578536f,  0.08076209f,  0.76034476f
    } };

    // ============================================================================
    // Color Space Helpers
    // ============================================================================

    inline Mat3 matrix_input_rgb_to_xyz(InputColorSpace cs) {
        switch (cs) {
        case InputColorSpace::DaVinciWideGamut:
            return kRGB_to_XYZ_DWG;
        case InputColorSpace::ITU_R_BT2020:
            return kRGB_to_XYZ_BT2020;
        case InputColorSpace::ACES2065_1:
            return kRGB_to_XYZ_ACES2065;
        case InputColorSpace::SRGB_Rec709:
            return kRGB_to_XYZ_sRGB_Rec709;
        default:
            return kRGB_to_XYZ_DWG;
        }
    }

    inline void input_colorspace_white_xyz(InputColorSpace cs, float whiteXYZ[3]) {
        switch (cs) {
        case InputColorSpace::DaVinciWideGamut:
        case InputColorSpace::ITU_R_BT2020:
        case InputColorSpace::SRGB_Rec709:
            whiteXYZ[0] = gDWG_WhitePoint_XYZ[0];
            whiteXYZ[1] = gDWG_WhitePoint_XYZ[1];
            whiteXYZ[2] = gDWG_WhitePoint_XYZ[2];
            break;
        case InputColorSpace::ACES2065_1:
            whiteXYZ[0] = 0.95264608f;
            whiteXYZ[1] = 1.0f;
            whiteXYZ[2] = 1.00882518f;
            break;
        default:
            whiteXYZ[0] = gDWG_WhitePoint_XYZ[0];
            whiteXYZ[1] = gDWG_WhitePoint_XYZ[1];
            whiteXYZ[2] = gDWG_WhitePoint_XYZ[2];
            break;
        }
    }

    inline float sanitize_channel(float v) {
        return std::isfinite(v) ? v : 0.0f;
    }

    inline float decode_BT2020_channel(float v) {
        const float x = std::max(0.0f, sanitize_channel(v));
        constexpr float a = 1.09929681f;
        constexpr float b = 0.01805397f;
        constexpr float threshold = 0.0812428791f; // Precomputed from a * pow(b, 0.45f) - (a - 1.0f)
        if (x < threshold) {
            return x / 4.5f;
        }
        return static_cast<float>(std::pow((x + (a - 1.0f)) / a, 1.0f / 0.45f));
    }

    inline float decode_sRGB_channel(float v) {
        const float x = std::max(0.0f, sanitize_channel(v));
        constexpr float threshold = 0.04045f;
        if (x <= threshold) {
            return x / 12.92f;
        }
        return static_cast<float>(std::pow((x + 0.055f) / 1.055f, 2.4f));
    }

    inline void apply_input_cctf_decoding(InputColorSpace cs, bool decode, const float in[3], float out[3]) {
        if (!decode) {
            out[0] = sanitize_channel(in[0]);
            out[1] = sanitize_channel(in[1]);
            out[2] = sanitize_channel(in[2]);
            return;
        }

        switch (cs) {
        case InputColorSpace::ITU_R_BT2020:
            out[0] = decode_BT2020_channel(in[0]);
            out[1] = decode_BT2020_channel(in[1]);
            out[2] = decode_BT2020_channel(in[2]);
            break;
        case InputColorSpace::SRGB_Rec709:
            out[0] = decode_sRGB_channel(in[0]);
            out[1] = decode_sRGB_channel(in[1]);
            out[2] = decode_sRGB_channel(in[2]);
            break;
        case InputColorSpace::DaVinciWideGamut:
        case InputColorSpace::ACES2065_1:
        default:
            out[0] = sanitize_channel(in[0]);
            out[1] = sanitize_channel(in[1]);
            out[2] = sanitize_channel(in[2]);
            break;
        }
    }

    // ============================================================================
    // Chromatic Adaptation
    // ============================================================================

    // CAT02/Von Kries based chromatic adaptation (matches colour.XYZ_to_RGB default)
    inline void chromatic_adapt_XYZ_CAT02(
        const float XYZ[3],
        const float srcWhiteXYZ[3],
        const float dstWhiteXYZ[3],
        float outXYZ[3])
    {
        static const float M[9] = {
             0.7328000f,  0.4296000f, -0.1624000f,
            -0.7036000f,  1.6975000f,  0.0061000f,
             0.0030000f,  0.0136000f,  0.9834000f
        };
        static const float M_inv[9] = {
            1.0961238f, -0.2788690f,  0.1827452f,
            0.4543690f,  0.4735332f,  0.0720978f,
           -0.0096276f, -0.0056980f,  1.0153256f
        };

        auto mul3 = [](const float m[9], const float v[3], float dst[3]) {
            dst[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
            dst[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
            dst[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
            };

        auto sanitize = [](float v) -> float {
            return std::isfinite(v) ? std::max(0.0f, v) : 0.0f;
            };

        float srcWhite[3] = {
            sanitize(srcWhiteXYZ[0]),
            sanitize(srcWhiteXYZ[1]),
            sanitize(srcWhiteXYZ[2])
        };
        float dstWhite[3] = {
            sanitize(dstWhiteXYZ[0]),
            sanitize(dstWhiteXYZ[1]),
            sanitize(dstWhiteXYZ[2])
        };

        const float srcY = (srcWhite[1] > 0.0f) ? srcWhite[1] : 1.0f;
        const float dstY = (dstWhite[1] > 0.0f) ? dstWhite[1] : 1.0f;
        const float srcScale = 1.0f / srcY;
        const float dstScale = 1.0f / dstY;
        srcWhite[0] *= srcScale; srcWhite[1] = 1.0f; srcWhite[2] *= srcScale;
        dstWhite[0] *= dstScale; dstWhite[1] = 1.0f; dstWhite[2] *= dstScale;

        float srcLMS[3];
        float dstLMS[3];
        float XYZ_LMS[3];
        mul3(M, srcWhite, srcLMS);
        mul3(M, dstWhite, dstLMS);
        mul3(M, XYZ, XYZ_LMS);

        float scale[3];
        scale[0] = (srcLMS[0] > 1e-6f) ? (dstLMS[0] / srcLMS[0]) : 1.0f;
        scale[1] = (srcLMS[1] > 1e-6f) ? (dstLMS[1] / srcLMS[1]) : 1.0f;
        scale[2] = (srcLMS[2] > 1e-6f) ? (dstLMS[2] / srcLMS[2]) : 1.0f;

        float adaptedLMS[3] = {
            scale[0] * XYZ_LMS[0],
            scale[1] * XYZ_LMS[1],
            scale[2] * XYZ_LMS[2]
        };

        mul3(M_inv, adaptedLMS, outXYZ);
    }

    inline Mat3 build_chromatic_adaptation_matrix(const float srcWhite[3], const float dstWhite[3]) {
        Mat3 adapt = make_identity_mat3();
        for (int col = 0; col < 3; ++col) {
            float basis[3] = { 0.0f, 0.0f, 0.0f };
            basis[col] = 1.0f;
            float adapted[3];
            chromatic_adapt_XYZ_CAT02(basis, srcWhite, dstWhite, adapted);
            for (int row = 0; row < 3; ++row) {
                adapt.m[row * 3 + col] = adapted[row];
            }
        }
        return adapt;
    }

    // ============================================================================
    // Film Raw Input
    // ============================================================================

    struct FilmRawConfig {
        InputColorSpace inputColorSpace = InputColorSpace::DaVinciWideGamut;
        bool applyCctfDecoding = false;
        Mat3 inputRGBToXYZ = make_identity_mat3();
        Mat3 inputXYZAdapt = make_identity_mat3();
        bool applyInputChromaticAdapt = false;
        SpectralUpsamplingMode spectralUpsamplingMode = SpectralUpsamplingMode::PreferHanatos;
        float midgrayScale = 1.0f;
        float midgrayDWG[3] = { 0.184f, 0.184f, 0.184f };
        float rawMidgray[3] = { 1.0f, 1.0f, 1.0f };
        float rawMidgrayGreen = 1.0f;
        float inputWhiteXYZ[3] = {
            gDWG_WhitePoint_XYZ[0],
            gDWG_WhitePoint_XYZ[1],
            gDWG_WhitePoint_XYZ[2]
        };
        float workingWhiteXYZ[3] = {
            gDWG_WhitePoint_XYZ[0],
            gDWG_WhitePoint_XYZ[1],
            gDWG_WhitePoint_XYZ[2]
        };
        float refIllumWhiteXYZ[3] = {
            gDWG_WhitePoint_XYZ[0],
            gDWG_WhitePoint_XYZ[1],
            gDWG_WhitePoint_XYZ[2]
        };
        bool hasRefIllumWhite = false;
        bool valid = false;
    };

    inline bool whites_approximately_equal(const float a[3], const float b[3]) {
        auto scale = [](float v) {
            return std::max(1.0f, std::fabs(v));
            };
        for (int i = 0; i < 3; ++i) {
            const float diff = std::fabs(a[i] - b[i]);
            if (diff > 1e-4f * scale(a[i]) || diff > 1e-4f * scale(b[i])) {
                return false;
            }
        }
        return true;
    }

    inline void prepare_film_raw_config(FilmRawConfig& cfg) {
        cfg.inputRGBToXYZ = matrix_input_rgb_to_xyz(cfg.inputColorSpace);
        input_colorspace_white_xyz(cfg.inputColorSpace, cfg.inputWhiteXYZ);
        cfg.workingWhiteXYZ[0] = gDWG_WhitePoint_XYZ[0];
        cfg.workingWhiteXYZ[1] = gDWG_WhitePoint_XYZ[1];
        cfg.workingWhiteXYZ[2] = gDWG_WhitePoint_XYZ[2];

        cfg.applyInputChromaticAdapt = !whites_approximately_equal(cfg.inputWhiteXYZ, cfg.workingWhiteXYZ);
        if (cfg.applyInputChromaticAdapt) {
            cfg.inputXYZAdapt = build_chromatic_adaptation_matrix(cfg.inputWhiteXYZ, cfg.workingWhiteXYZ);
            bool finite = true;
            for (float m : cfg.inputXYZAdapt.m) {
                if (!std::isfinite(m)) {
                    finite = false;
                    break;
                }
            }
            if (!finite) {
                cfg.inputXYZAdapt = make_identity_mat3();
                cfg.applyInputChromaticAdapt = false;
            }
        }
        else {
            cfg.inputXYZAdapt = make_identity_mat3();
        }

        cfg.valid = true;
    }

    inline void convert_input_rgb_to_DWG(
        const FilmRawConfig& cfg,
        const float rgbIn[3],
        float rgbDWG[3],
        float* outXYZ = nullptr)
    {
        float linear[3];
        apply_input_cctf_decoding(cfg.inputColorSpace, cfg.applyCctfDecoding, rgbIn, linear);

        float XYZ[3];
        cfg.inputRGBToXYZ.mul(linear, XYZ);

        const float* xyzPtr = XYZ;
        float adaptedXYZ[3];
        if (cfg.applyInputChromaticAdapt) {
            cfg.inputXYZAdapt.mul(XYZ, adaptedXYZ);
            xyzPtr = adaptedXYZ;
        }

        if (outXYZ) {
            outXYZ[0] = std::max(0.0f, xyzPtr[0]);
            outXYZ[1] = std::max(0.0f, xyzPtr[1]);
            outXYZ[2] = std::max(0.0f, xyzPtr[2]);
        }

        float dwgLinear[3];
        XYZ_to_DWG_linear(xyzPtr, dwgLinear);

        for (int i = 0; i < 3; ++i) {
            float v = dwgLinear[i];
            if (!std::isfinite(v)) {
                v = 0.0f;
            }
            if (v < 0.0f) {
                v = 0.0f;
            }
            rgbDWG[i] = v;
        }
    }

    inline void compute_film_raw_midgray(
        FilmRawConfig& cfg,
        const SpectralTables* tablesSPD,
        const float* S_inv,
        const Curve& sB,
        const Curve& sG,
        const Curve& sR)
    {
        const float rgbMid[3] = { 0.184f, 0.184f, 0.184f };
        float rgbMidDWG[3];
        convert_input_rgb_to_DWG(cfg, rgbMid, rgbMidDWG);
        cfg.midgrayDWG[0] = rgbMidDWG[0];
        cfg.midgrayDWG[1] = rgbMidDWG[1];
        cfg.midgrayDWG[2] = rgbMidDWG[2];

        const bool spdReady = tablesSPD && S_inv && tablesSPD->K > 0;
        float E[3] = { 0.0f, 0.0f, 0.0f };
        if (!spdReady) {
            cfg.rawMidgray[0] = cfg.rawMidgray[1] = cfg.rawMidgray[2] = 0.0f;
            cfg.rawMidgrayGreen = 1.0f;
            cfg.midgrayScale = 1.0f;
            return;
        }

        rgbDWG_to_layerExposures_from_tables_with_curves(
            rgbMidDWG, E, 1.0f,
            tablesSPD,
            S_inv,
            sB, sG, sR,
            cfg.spectralUpsamplingMode,
            cfg.refIllumWhiteXYZ);

        cfg.rawMidgray[0] = E[0];
        cfg.rawMidgray[1] = E[1];
        cfg.rawMidgray[2] = E[2];
        const float safeGreen = (std::isfinite(E[1]) && E[1] > 1e-9f) ? E[1] : 1.0f;
        cfg.rawMidgrayGreen = safeGreen;
        cfg.midgrayScale = (std::isfinite(safeGreen) && safeGreen > 1e-9f) ? (1.0f / safeGreen) : 1.0f;
        if (!std::isfinite(cfg.midgrayScale) || cfg.midgrayScale <= 0.0f) {
            cfg.midgrayScale = 1.0f;
        }
    }

    inline void rgb_input_to_film_raw(
        const float rgbIn[3],
        float E[3],
        float exposureScale,
        const FilmRawConfig& cfg,
        const SpectralTables* tablesSPD,
        const float* S_inv,
        bool useSPD,
        const Curve& sB,
        const Curve& sG,
        const Curve& sR)
    {
        const bool spdReady = useSPD && tablesSPD && S_inv && tablesSPD->K > 0;
        float rgbDWG[3];
        float xyzWorking[3];
        convert_input_rgb_to_DWG(cfg, rgbIn, rgbDWG, xyzWorking);

#if defined(JUICER_SPD_DEBUG)
        spd_probe_begin_capture(rgbIn, rgbDWG, spdReady);
#endif

        float normScale = cfg.midgrayScale;
        if (spdReady) {
            rgbDWG_to_layerExposures_from_tables_with_curves(
                rgbDWG, E, 1.0f,
                tablesSPD,
                S_inv,
                sB, sG, sR,
                cfg.spectralUpsamplingMode,
                cfg.refIllumWhiteXYZ);

            normScale = cfg.midgrayScale;
            E[0] = std::max(0.0f, E[0] * normScale);
            E[1] = std::max(0.0f, E[1] * normScale);
            E[2] = std::max(0.0f, E[2] * normScale);
        }
        else {
            E[0] = E[1] = E[2] = 0.0f;
        }

#if defined(JUICER_SPD_DEBUG)
        spd_probe_finalize(normScale, E);
#endif

        const float sExp = (std::isfinite(exposureScale) && exposureScale > 0.0f) ? exposureScale : 1.0f;
        if (sExp != 1.0f) {
            E[0] = std::max(0.0f, E[0] * sExp);
            E[1] = std::max(0.0f, E[1] * sExp);
            E[2] = std::max(0.0f, E[2] * sExp);
        }
    }

    // ============================================================================
    // DWG � XYZ Transforms
    // ============================================================================

    inline void XYZ_to_DWG_linear_adapted(
        const SpectralTables& tables,
        const float XYZ[3],
        float RGB[3])
    {
        float srcWhite[3] = {
            tables.whiteXYZ[0],
            tables.whiteXYZ[1],
            tables.whiteXYZ[2]
        };
        const float sum = srcWhite[0] + srcWhite[1] + srcWhite[2];
        if (!std::isfinite(sum) || sum <= 0.0f) {
            srcWhite[0] = gDWG_WhitePoint_XYZ[0];
            srcWhite[1] = gDWG_WhitePoint_XYZ[1];
            srcWhite[2] = gDWG_WhitePoint_XYZ[2];
        }

        float adaptedXYZ[3];
        chromatic_adapt_XYZ_CAT02(XYZ, srcWhite, gDWG_WhitePoint_XYZ, adaptedXYZ);
        gDWG_XYZ_to_RGB.mul(adaptedXYZ, RGB);
    }

    inline void XYZ_to_DWG_linear(const float XYZ[3], float RGB[3]) {
        gDWG_XYZ_to_RGB.mul(XYZ, RGB);
        RGB[0] = std::max(0.0f, RGB[0]);
        RGB[1] = std::max(0.0f, RGB[1]);
        RGB[2] = std::max(0.0f, RGB[2]);
    }

    inline void DWG_linear_to_XYZ(const float RGB[3], float XYZ[3]) {
        gDWG_RGB_to_XYZ.mul(RGB, XYZ);
    }

    inline float neutral_blend_weight_from_DWG_rgb(const float rgbDWG[3]) {
        // Y = row 2 of DWG_RGB_to_XYZ dot rgb
        const float Y = std::max(0.0f,
            gDWG_RGB_to_XYZ.m[3] * rgbDWG[0] +
            gDWG_RGB_to_XYZ.m[4] * rgbDWG[1] +
            gDWG_RGB_to_XYZ.m[5] * rgbDWG[2]);
        const float w = Y / 0.18f;
        return std::clamp(w, 0.0f, 1.0f);
    }

} // namespace Spectral
