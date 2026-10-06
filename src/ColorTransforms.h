// ColorTransforms.h
// Color space transformations, chromatic adaptation, and input color space handling
#pragma once

#include <cmath>
#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "SpectralData.h"
#include "RustColorBridge.h"

// Forward declarations
namespace Spectral {
    inline bool is_finite(float v);
    inline bool is_finite(double v);
    struct SpectralTables;
    struct Curve;
    inline void rgbDWG_to_layerExposures_from_tables_with_curves(
        const float rgbDWG[3], float E[3], float exposureScale, const SpectralTables* tables, const float* S_inv, const Curve& sB, const Curve& sG, const Curve& sR, SpectralUpsamplingMode spectralUpsamplingMode, const float refIllumWhiteXYZ[3]);

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

    inline float sanitize_raw_midgray_green_or_one(float value) {
        const float sanitized = sanitize_channel(value);
        return (sanitized > 1e-9f) ? sanitized : 1.0f;
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
            const auto converted = JuicerColor::input_to_linear_srgb(
                cfg, {rgb.input[0], rgb.input[1], rgb.input[2]});
            mallett2019_exposures_from_linear_srgb(converted.rgb.data(), *tablesSPD, sB, sG, sR, E);
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
        const auto converted = JuicerColor::input_to_dwg(
            cfg, {rgbMid[0], rgbMid[1], rgbMid[2]}, !path.useHanatos);
        std::copy(converted.rgb.begin(), converted.rgb.end(), rgbMidDWG);
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

} // namespace Spectral
