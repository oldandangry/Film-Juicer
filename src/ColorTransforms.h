// ColorTransforms.h
// Color space transformations, chromatic adaptation, and input color space handling
#pragma once

#include <cmath>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include "SpectralData.h"

// Forward declarations
namespace Spectral {
    inline bool is_finite(float v);
    inline bool is_finite(double v);
    struct SpectralTables;
    struct Curve;
    inline void rgbDWG_to_layerExposures_from_tables_with_curves(
        const float rgbDWG[3], float E[3], float exposureScale, const SpectralTables* tables, const float* S_inv, const Curve& sB, const Curve& sG, const Curve& sR, SpectralUpsamplingMode spectralUpsamplingMode, const float refIllumWhiteXYZ[3]);

#if defined(JUICER_SPD_DEBUG) && (JUICER_SPD_DEBUG != 0)
    bool spd_probe_begin_capture(const float rgbIn[3], const float rgbDWG[3], bool spdEnabled);
    void spd_probe_finalize(float midgrayScale, const float E_afterMidgray[3]);
#endif
} // namespace Spectral

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

    constexpr const char* inputColorSpaceLabel(InputColorSpace cs) {
        const std::size_t idx = static_cast<std::size_t>(cs);
        return idx < kInputColorSpaceCount ? kInputColorSpaceLabels[idx] : "DaVinci Wide Gamut";
    }

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

    inline void mul_3x3_vec3(const float m[9], const float v[3], float out[3]) {
        out[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
        out[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
        out[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
    }

    struct Mat3 {
        float m[9];
        void mul(const float v[3], float out[3]) const {
            mul_3x3_vec3(m, v, out);
        }
        Mat3 inverse(float fallback = 1.0f) const {
            const float det =
                m[0] * (m[4] * m[8] - m[5] * m[7]) -
                m[1] * (m[3] * m[8] - m[5] * m[6]) +
                m[2] * (m[3] * m[7] - m[4] * m[6]);

            const float eps = 1e-6f;
            if (!is_finite(det) || std::fabs(det) <= eps) {
                return Mat3{{fallback, 0.0f, 0.0f, 0.0f, fallback, 0.0f, 0.0f, 0.0f, fallback}};
            }

            const float invDet = 1.0f / det;
            Mat3 inv{{(m[4] * m[8] - m[5] * m[7]) * invDet,
                      (m[2] * m[7] - m[1] * m[8]) * invDet,
                      (m[1] * m[5] - m[2] * m[4]) * invDet,
                      (m[5] * m[6] - m[3] * m[8]) * invDet,
                      (m[0] * m[8] - m[2] * m[6]) * invDet,
                      (m[2] * m[3] - m[0] * m[5]) * invDet,
                      (m[3] * m[7] - m[4] * m[6]) * invDet,
                      (m[1] * m[6] - m[0] * m[7]) * invDet,
                      (m[0] * m[4] - m[1] * m[3]) * invDet}};
            return inv;
        }
    };

    inline bool mat3_has_only_finite(const Mat3& matrix) {
        const float* mIt = matrix.m;
        const float* const mEnd = mIt + 9;
        for (; mIt < mEnd; ++mIt) {
            if (!is_finite(*mIt)) {
                return false;
            }
        }
        return true;
    }

    inline Mat3 make_identity_mat3(float diag = 1.0f) {
        return Mat3{{diag, 0.0f, 0.0f, 0.0f, diag, 0.0f, 0.0f, 0.0f, diag}};
    }

    inline void copy_triplet(float dst[3], const float src[3]) {
        float* dstIt = dst;
        const float* srcIt = src;
        for (int i = 0; i < 3; ++i, ++dstIt, ++srcIt) {
            *dstIt = *srcIt;
        }
    }

    inline void assign_triplet(float dst[3], float v0, float v1, float v2) {
        dst[0] = v0;
        dst[1] = v1;
        dst[2] = v2;
    }

    inline void scale_triplet_nonnegative_inplace(float values[3], float scale) {
        float* valueIt = values;
        for (int i = 0; i < 3; ++i, ++valueIt) {
            *valueIt = std::max(0.0f, *valueIt * scale);
        }
    }

    inline void clamp_triplet_nonnegative_inplace(float values[3]) {
        float* valueIt = values;
        for (int i = 0; i < 3; ++i, ++valueIt) {
            *valueIt = std::max(0.0f, *valueIt);
        }
    }

    inline void copy_triplet_sanitized(float dst[3], const float src[3], bool clampNonNegative) {
        const float* srcIt = src;
        float* dstIt = dst;
        for (int i = 0; i < 3; ++i, ++srcIt, ++dstIt) {
            const float value = *srcIt;
            float out = is_finite(value) ? value : 0.0f;
            if (clampNonNegative && out < 0.0f) {
                out = 0.0f;
            }
            *dstIt = out;
        }
    }

    // Canonical RGB�XYZ matrices
    inline constexpr Mat3 kRGB_to_XYZ_DWG = {{0.70062239f, 0.14877482f, 0.10105872f, 0.27411851f, 0.87363190f, -0.14775041f, -0.09896291f, -0.13789533f, 1.32591599f}};

    inline constexpr Mat3 kRGB_to_XYZ_BT2020 = {{0.63695806f, 0.14461690f, 0.16888097f, 0.26270020f, 0.67799807f, 0.05930172f, 0.00000000f, 0.02807269f, 1.06098509f}};

    inline constexpr Mat3 kRGB_to_XYZ_ACES2065 = {{0.95255238f, 0.00000000f, 0.00009368f, 0.34396645f, 0.72816610f, -0.07213255f, 0.00000000f, 0.00000000f, 1.00882518f}};

    inline constexpr Mat3 kRGB_to_XYZ_sRGB_Rec709 = {{0.4124564f, 0.3575761f, 0.1804375f, 0.2126729f, 0.7151522f, 0.0721750f, 0.0193339f, 0.1191920f, 0.9503041f}};

    // Global DWG�XYZ matrices
    inline Mat3 gDWG_RGB_to_XYZ = {{0.70062239f, 0.14877482f, 0.10105872f, 0.27411851f, 0.87363190f, -0.14775041f, -0.09896291f, -0.13789533f, 1.32591599f}};
    inline Mat3 gDWG_XYZ_to_RGB = {{1.51667204f, -0.28147805f, -0.14696363f, -0.46491710f, 1.25142378f, 0.17488461f, 0.07578536f, 0.08076209f, 0.76034476f}};

    inline void XYZ_to_DWG_linear(const float XYZ[3], float RGB[3]) {
        gDWG_XYZ_to_RGB.mul(XYZ, RGB);
        clamp_triplet_nonnegative_inplace(RGB);
    }

    inline void DWG_linear_to_XYZ(const float RGB[3], float XYZ[3]) {
        gDWG_RGB_to_XYZ.mul(RGB, XYZ);
    }

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
        static constexpr float kACES2065WhiteXYZ[3] = {0.95264608f, 1.0f, 1.00882518f};
        switch (cs) {
            case InputColorSpace::DaVinciWideGamut:
            case InputColorSpace::ITU_R_BT2020:
            case InputColorSpace::SRGB_Rec709:
                copy_triplet(whiteXYZ, gDWG_WhitePoint_XYZ);
                break;
            case InputColorSpace::ACES2065_1:
                copy_triplet(whiteXYZ, kACES2065WhiteXYZ);
                break;
            default:
                copy_triplet(whiteXYZ, gDWG_WhitePoint_XYZ);
                break;
        }
    }

    inline bool is_finite(float v) {
        return std::isfinite(v);
    }

    inline bool is_finite(double v) {
        return std::isfinite(v);
    }

    inline float sanitize_channel(float v) {
        return is_finite(v) ? v : 0.0f;
    }

    inline float sanitize_nonnegative_channel(float v) {
        return std::max(0.0f, sanitize_channel(v));
    }

    inline void sanitize_triplet(float out[3], const float in[3]) {
        float* outIt = out;
        const float* inIt = in;
        for (int i = 0; i < 3; ++i, ++outIt, ++inIt) {
            *outIt = sanitize_channel(*inIt);
        }
    }

    inline void sanitize_nonnegative_triplet(float out[3], const float in[3]) {
        float* outIt = out;
        const float* inIt = in;
        for (int i = 0; i < 3; ++i, ++outIt, ++inIt) {
            *outIt = sanitize_nonnegative_channel(*inIt);
        }
    }

    inline void normalize_triplet_to_unit_y(float values[3]) {
        const float y = (values[1] > 0.0f) ? values[1] : 1.0f;
        const float invY = 1.0f / y;
        float* valueIt = values;
        for (int i = 0; i < 3; ++i, ++valueIt) {
            *valueIt *= invY;
        }
        values[1] = 1.0f;
    }

    inline bool triplet_has_positive_finite_sum(const float values[3]) {
        const float sum = values[0] + values[1] + values[2];
        return is_finite(sum) && sum > 0.0f;
    }

    inline void sanitize_white_or_dwg(const float in[3], float out[3]) {
        sanitize_nonnegative_triplet(out, in);
        if (!triplet_has_positive_finite_sum(out)) {
            copy_triplet(out, gDWG_WhitePoint_XYZ);
        }
    }

    inline float sanitize_raw_midgray_green_or_one(float value) {
        const float sanitized = sanitize_channel(value);
        return (sanitized > 1e-9f) ? sanitized : 1.0f;
    }

    inline float sanitize_exposure_scale_or_one(float value) {
        const float sanitized = sanitize_channel(value);
        return (sanitized > 0.0f) ? sanitized : 1.0f;
    }

    inline bool is_positive_finite(float v) {
        return is_finite(v) && v > 0.0f;
    }

    inline float finite_to_float_or_zero(double v) {
        return is_finite(v) ? static_cast<float>(v) : 0.0f;
    }

    inline void accumulate_if_finite(double& acc, double lhs, float rhs) {
        if (is_finite(rhs)) {
            acc += lhs * static_cast<double>(rhs);
        }
    }

    inline float decode_BT2020_nonnegative(float x) {
        constexpr float a = 1.09929681f;
        constexpr float invA = 1.0f / a;
        constexpr float threshold = 0.0812428791f; // Precomputed from a * pow(b, 0.45f) - (a - 1.0f)
        constexpr float invGamma = 2.2222222222f;  // 1 / 0.45
        if (x < threshold) {
            return x / 4.5f;
        }
        return std::pow((x + (a - 1.0f)) * invA, invGamma);
    }

    inline float decode_BT2020_channel(float v) {
        const float x = sanitize_nonnegative_channel(v);
        return decode_BT2020_nonnegative(x);
    }

    inline float decode_sRGB_nonnegative(float x) {
        constexpr float threshold = 0.04045f;
        constexpr float invScale = 1.0f / 1.055f;
        if (x <= threshold) {
            return x / 12.92f;
        }
        return std::pow((x + 0.055f) * invScale, 2.4f);
    }

    inline float decode_sRGB_channel(float v) {
        const float x = sanitize_nonnegative_channel(v);
        return decode_sRGB_nonnegative(x);
    }

    inline void apply_input_cctf_decoding(InputColorSpace cs, bool decode, const float in[3], float out[3]) {
        if (!decode) {
            sanitize_triplet(out, in);
            return;
        }

        switch (cs) {
            case InputColorSpace::ITU_R_BT2020: {
                assign_triplet(
                    out,
                    decode_BT2020_channel(in[0]),
                    decode_BT2020_channel(in[1]),
                    decode_BT2020_channel(in[2]));
                break;
            }
            case InputColorSpace::SRGB_Rec709: {
                assign_triplet(
                    out,
                    decode_sRGB_channel(in[0]),
                    decode_sRGB_channel(in[1]),
                    decode_sRGB_channel(in[2]));
                break;
            }
            case InputColorSpace::DaVinciWideGamut:
            case InputColorSpace::ACES2065_1:
            default:
                sanitize_triplet(out, in);
                break;
        }
    }

    // ============================================================================
    // Chromatic Adaptation
    // ============================================================================

    struct ChromaticAdaptationWhites {
        const float* source = nullptr;
        const float* destination = nullptr;
    };

    // CAT02/Von Kries based chromatic adaptation (matches colour.XYZ_to_RGB default)
    inline void chromatic_adapt_XYZ_CAT02(
        const float XYZ[3],
        const ChromaticAdaptationWhites& whites,
        float outXYZ[3]) {
        static const float M[9] = {
            0.7328000f, 0.4296000f, -0.1624000f, -0.7036000f, 1.6975000f, 0.0061000f, 0.0030000f, 0.0136000f, 0.9834000f};
        static const float M_inv[9] = {
            1.0961238f, -0.2788690f, 0.1827452f, 0.4543690f, 0.4735332f, 0.0720978f, -0.0096276f, -0.0056980f, 1.0153256f};

        float srcWhite[3];
        float dstWhite[3];
        sanitize_nonnegative_triplet(srcWhite, whites.source);
        sanitize_nonnegative_triplet(dstWhite, whites.destination);
        normalize_triplet_to_unit_y(srcWhite);
        normalize_triplet_to_unit_y(dstWhite);

        float srcLMS[3];
        float dstLMS[3];
        float XYZ_LMS[3];
        mul_3x3_vec3(M, srcWhite, srcLMS);
        mul_3x3_vec3(M, dstWhite, dstLMS);
        mul_3x3_vec3(M, XYZ, XYZ_LMS);

        float adaptedLMS[3];
        const float* srcLMSIt = srcLMS;
        const float* dstLMSIt = dstLMS;
        const float* xyzLMSIt = XYZ_LMS;
        float* adaptedIt = adaptedLMS;
        for (int i = 0; i < 3; ++i, ++srcLMSIt, ++dstLMSIt, ++xyzLMSIt, ++adaptedIt) {
            const float scale = (*srcLMSIt > 1e-6f) ? (*dstLMSIt / *srcLMSIt) : 1.0f;
            *adaptedIt = scale * *xyzLMSIt;
        }

        mul_3x3_vec3(M_inv, adaptedLMS, outXYZ);
    }

    inline Mat3 build_chromatic_adaptation_matrix(const ChromaticAdaptationWhites& whites) {
        Mat3 adapt = make_identity_mat3();
        for (int col = 0; col < 3; ++col) {
            float basis[3] = {0.0f, 0.0f, 0.0f};
            basis[col] = 1.0f;
            float adapted[3];
            chromatic_adapt_XYZ_CAT02(basis, whites, adapted);
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

    inline bool whites_approximately_equal(const ChromaticAdaptationWhites& whites) {
        auto scale = [](float v) {
            return std::max(1.0f, std::fabs(v));
        };
        const float* aIt = whites.source;
        const float* bIt = whites.destination;
        for (int i = 0; i < 3; ++i, ++aIt, ++bIt) {
            const float av = *aIt;
            const float bv = *bIt;
            const float diff = std::fabs(av - bv);
            if (diff > 1e-4f * scale(av) || diff > 1e-4f * scale(bv)) {
                return false;
            }
        }
        return true;
    }

    inline void prepare_film_raw_config(FilmRawConfig& cfg) {
        cfg.inputRGBToXYZ = matrix_input_rgb_to_xyz(cfg.inputColorSpace);
        input_colorspace_white_xyz(cfg.inputColorSpace, cfg.inputWhiteXYZ);
        copy_triplet(cfg.workingWhiteXYZ, gDWG_WhitePoint_XYZ);
        sanitize_white_or_dwg(cfg.inputWhiteXYZ, cfg.inputWhiteXYZ);
        sanitize_white_or_dwg(cfg.workingWhiteXYZ, cfg.workingWhiteXYZ);

        ChromaticAdaptationWhites whites{};
        whites.source = cfg.inputWhiteXYZ;
        whites.destination = cfg.workingWhiteXYZ;
        cfg.applyInputChromaticAdapt = !whites_approximately_equal(whites);
        if (cfg.applyInputChromaticAdapt) {
            cfg.inputXYZAdapt = build_chromatic_adaptation_matrix(whites);
            if (!mat3_has_only_finite(cfg.inputXYZAdapt)) {
                cfg.inputXYZAdapt = make_identity_mat3();
                cfg.applyInputChromaticAdapt = false;
            }
        } else {
            cfg.inputXYZAdapt = make_identity_mat3();
        }

        cfg.valid = true;
    }

    inline void convert_input_rgb_to_DWG(
        const FilmRawConfig& cfg,
        const float rgbIn[3],
        float rgbDWG[3],
        float* outXYZ = nullptr,
        bool clampNonNegative = true) {
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
            copy_triplet_sanitized(outXYZ, xyzPtr, clampNonNegative);
        }

        float dwgLinear[3];
        if (clampNonNegative) {
            XYZ_to_DWG_linear(xyzPtr, dwgLinear);
        } else {
            gDWG_XYZ_to_RGB.mul(xyzPtr, dwgLinear);
        }
        copy_triplet_sanitized(rgbDWG, dwgLinear, clampNonNegative);
    }

    inline void convert_input_rgb_to_sRGB_linear(
        const FilmRawConfig& cfg,
        const float rgbIn[3],
        float rgbSRGB[3],
        float* outXYZ = nullptr) {
        float linear[3];
        apply_input_cctf_decoding(cfg.inputColorSpace, cfg.applyCctfDecoding, rgbIn, linear);

        float XYZ[3];
        cfg.inputRGBToXYZ.mul(linear, XYZ);

        const float* xyzPtr = XYZ;
        float adapted[3];
        if (cfg.applyInputChromaticAdapt) {
            cfg.inputXYZAdapt.mul(XYZ, adapted);
            xyzPtr = adapted;
        }

        if (outXYZ) {
            copy_triplet_sanitized(outXYZ, xyzPtr, false);
        }

        static const Mat3 kXYZ_to_sRGB = kRGB_to_XYZ_sRGB_Rec709.inverse();
        float rgbLinear[3];
        kXYZ_to_sRGB.mul(xyzPtr, rgbLinear);
        copy_triplet_sanitized(rgbSRGB, rgbLinear, false);
    }

    inline bool mallett_basis_ready_for_tables(
        const SpectralTables* tables,
        const Curve& sB,
        const Curve& sG,
        const Curve& sR) {
        if (!tables || tables->K <= 0) {
            return false;
        }
        if (!mallett_available() || !mallett_basis_matches_reference_shape()) {
            return false;
        }
        const int K = tables->K;
        if (static_cast<int>(tables->illum.size()) != K) {
            return false;
        }
        if (static_cast<int>(sB.linear.size()) != K ||
            static_cast<int>(sG.linear.size()) != K ||
            static_cast<int>(sR.linear.size()) != K) {
            return false;
        }
        const size_t expected = static_cast<size_t>(K) * 3;
        if (gMallettBasis.data.size() != expected) {
            return false;
        }
        return true;
    }

    inline void mallett2019_exposures_from_linear_srgb(
        const float lrgb[3],
        const SpectralTables& tables,
        const Curve& sB,
        const Curve& sG,
        const Curve& sR,
        float E[3]) {
        const int K = tables.K;
        double Eb = 0.0;
        double Eg = 0.0;
        double Er = 0.0;
        const float r = sanitize_nonnegative_channel(lrgb[0]);
        const float g = sanitize_nonnegative_channel(lrgb[1]);
        const float b = sanitize_nonnegative_channel(lrgb[2]);
        const float* basis = gMallettBasis.data.data();
        const float* illumIt = tables.illum.data();
        const float* sensBIt = sB.linear.data();
        const float* sensGIt = sG.linear.data();
        const float* sensRIt = sR.linear.data();

        for (int i = 0; i < K; ++i, ++illumIt, ++sensBIt, ++sensGIt, ++sensRIt, basis += 3) {
            const float illum = *illumIt;
            const float b0 = basis[0];
            const float b1 = basis[1];
            const float b2 = basis[2];
            const float spd = (r * b0 + g * b1 + b * b2) * illum;
            if (!is_finite(spd)) {
                continue;
            }
            const double e64 = static_cast<double>(spd);
            const float sb = *sensBIt;
            const float sg = *sensGIt;
            const float sr = *sensRIt;
            accumulate_if_finite(Eb, e64, sb);
            accumulate_if_finite(Eg, e64, sg);
            accumulate_if_finite(Er, e64, sr);
        }

        assign_triplet(
            E,
            finite_to_float_or_zero(Eb),
            finite_to_float_or_zero(Eg),
            finite_to_float_or_zero(Er));
    }

    struct SpectralReconstructionPath {
        bool spdReady = false;
        bool useHanatos = false;
        bool useMallett = false;
    };

    inline SpectralReconstructionPath select_spectral_reconstruction_path(
        const FilmRawConfig& cfg,
        const SpectralTables* tablesSPD,
        const float* S_inv,
        bool useSPD,
        const Curve& sB,
        const Curve& sG,
        const Curve& sR) {
        SpectralReconstructionPath path{};
        path.spdReady = useSPD && tablesSPD && S_inv && tablesSPD->K > 0;
        if (!path.spdReady) {
            return path;
        }
        const bool allowHanatos = (cfg.spectralUpsamplingMode == SpectralUpsamplingMode::PreferHanatos);
        path.useHanatos = allowHanatos && hanatos_available() && hanatos_matches_reference_shape();
        path.useMallett =
            (cfg.spectralUpsamplingMode == SpectralUpsamplingMode::ForceMallett) &&
            mallett_basis_ready_for_tables(tablesSPD, sB, sG, sR);
        return path;
    }

    struct ReconstructionRgbInputs {
        const float* input = nullptr;
        const float* davinciWideGamut = nullptr;
    };

    inline void compute_layer_exposures_from_reconstruction_path(
        const FilmRawConfig& cfg,
        const ReconstructionRgbInputs& rgb,
        const SpectralTables* tablesSPD,
        const float* S_inv,
        const Curve& sB,
        const Curve& sG,
        const Curve& sR,
        const SpectralReconstructionPath& path,
        float E[3]) {
        if (!path.spdReady || !tablesSPD || !S_inv) {
            std::fill_n(E, 3, 0.0f);
            return;
        }

        if (path.useMallett) {
            float rgbSRGB[3];
            convert_input_rgb_to_sRGB_linear(cfg, rgb.input, rgbSRGB, nullptr);
            mallett2019_exposures_from_linear_srgb(rgbSRGB, *tablesSPD, sB, sG, sR, E);
            return;
        }

        rgbDWG_to_layerExposures_from_tables_with_curves(
            rgb.davinciWideGamut,
            E,
            1.0f,
            tablesSPD,
            S_inv,
            sB,
            sG,
            sR,
            cfg.spectralUpsamplingMode,
            cfg.refIllumWhiteXYZ);
    }

    inline void compute_film_raw_midgray(
        FilmRawConfig& cfg,
        const SpectralTables* tablesSPD,
        const float* S_inv,
        const Curve& sB,
        const Curve& sG,
        const Curve& sR) {
        const float rgbMid[3] = {0.184f, 0.184f, 0.184f};
        float rgbMidDWG[3];
        const SpectralReconstructionPath path = select_spectral_reconstruction_path(
            cfg,
            tablesSPD,
            S_inv,
            true,
            sB,
            sG,
            sR);
        convert_input_rgb_to_DWG(cfg, rgbMid, rgbMidDWG, nullptr, !path.useHanatos);
        copy_triplet(cfg.midgrayDWG, rgbMidDWG);

        float E[3] = {0.0f, 0.0f, 0.0f};
        if (!path.spdReady) {
            std::fill_n(cfg.rawMidgray, 3, 0.0f);
            cfg.rawMidgrayGreen = 1.0f;
            cfg.midgrayScale = 1.0f;
            return;
        }

        ReconstructionRgbInputs rgbInputs{};
        rgbInputs.input = rgbMid;
        rgbInputs.davinciWideGamut = rgbMidDWG;
        compute_layer_exposures_from_reconstruction_path(
            cfg,
            rgbInputs,
            tablesSPD,
            S_inv,
            sB,
            sG,
            sR,
            path,
            E);

        copy_triplet(cfg.rawMidgray, E);
        const float safeGreen = sanitize_raw_midgray_green_or_one(E[1]);
        cfg.rawMidgrayGreen = safeGreen;
        cfg.midgrayScale = 1.0f / safeGreen;
        if (!is_positive_finite(cfg.midgrayScale)) {
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
        const Curve& sR) {
        const SpectralReconstructionPath path = select_spectral_reconstruction_path(
            cfg,
            tablesSPD,
            S_inv,
            useSPD,
            sB,
            sG,
            sR);
        float rgbDWG[3];
        float xyzWorking[3];
        convert_input_rgb_to_DWG(cfg, rgbIn, rgbDWG, xyzWorking, !path.useHanatos);

#if defined(JUICER_SPD_DEBUG) && (JUICER_SPD_DEBUG != 0)
        spd_probe_begin_capture(rgbIn, rgbDWG, path.spdReady);
#endif

#if defined(JUICER_SPD_DEBUG) && (JUICER_SPD_DEBUG != 0)
        float normScale = cfg.midgrayScale;
#endif
        if (path.spdReady) {
            ReconstructionRgbInputs rgbInputs{};
            rgbInputs.input = rgbIn;
            rgbInputs.davinciWideGamut = rgbDWG;
            compute_layer_exposures_from_reconstruction_path(
                cfg,
                rgbInputs,
                tablesSPD,
                S_inv,
                sB,
                sG,
                sR,
                path,
                E);
            scale_triplet_nonnegative_inplace(E, cfg.midgrayScale);
        } else {
            std::fill_n(E, 3, 0.0f);
        }

#if defined(JUICER_SPD_DEBUG) && (JUICER_SPD_DEBUG != 0)
        spd_probe_finalize(normScale, E);
#endif

        const float sExp = sanitize_exposure_scale_or_one(exposureScale);
        if (sExp != 1.0f) {
            scale_triplet_nonnegative_inplace(E, sExp);
        }
    }

    // ============================================================================
    // DWG � XYZ Transforms
    // ============================================================================

    inline void XYZ_to_DWG_linear_adapted(
        const SpectralTables& tables,
        const float XYZ[3],
        float RGB[3]) {
        float srcWhite[3];
        sanitize_white_or_dwg(tables.whiteXYZ, srcWhite);

        float adaptedXYZ[3];
        ChromaticAdaptationWhites whites{};
        whites.source = srcWhite;
        whites.destination = gDWG_WhitePoint_XYZ;
        chromatic_adapt_XYZ_CAT02(XYZ, whites, adaptedXYZ);
        gDWG_XYZ_to_RGB.mul(adaptedXYZ, RGB);
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
