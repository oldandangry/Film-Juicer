// SpectralProcessing.h
// Spectral processing operations: SPD reconstruction, table operations, integration, and math utilities

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "ColorTransforms.h"
#include "Hash.h"
#include "RustColorBridge.h"
#include "SpectralData.h"

struct FilmRawRecipe;
namespace Spectral {

    bool build_film_tc_lut(
        const ::FilmRawRecipe& recipe,
        const ReconstructionLut& spectra,
        const std::array<float, kNumSamples>& referenceIlluminant,
        FilmTcLut& out,
        std::string& diagnostic);

    std::array<float, 3> sample_film_tc_lut(
        const FilmTcLut& lut,
        const std::array<float, 3>& projectedXYZ);

    inline bool is_finite_sp(float value) {
        return std::isfinite(value);
    }

    inline float sanitize_nonnegative_component(float value) {
        return (is_finite_sp(value) && value > 0.0f) ? value : 0.0f;
    }

    inline float sanitize_nonfinite_component(float value) {
        return is_finite_sp(value) ? value : 0.0f;
    }

    inline void sanitize_nonfinite_triplet(float values[3]) {
        float* valueIt = values;
        for (int i = 0; i < 3; ++i, ++valueIt) {
            *valueIt = sanitize_nonfinite_component(*valueIt);
        }
    }

    inline void sanitize_nonnegative_triplet_sp(float dst[3], const float src[3]) {
        float* dstIt = dst;
        const float* srcIt = src;
        for (int i = 0; i < 3; ++i, ++dstIt, ++srcIt) {
            *dstIt = sanitize_nonnegative_component(*srcIt);
        }
    }

    inline void copy_triplet3(const float src[3], float dst[3]) {
        float* dstIt = dst;
        const float* srcIt = src;
        for (int i = 0; i < 3; ++i, ++dstIt, ++srcIt) {
            *dstIt = *srcIt;
        }
    }

    inline void clamp_triplet_nonnegative(float values[3]) {
        float* valueIt = values;
        for (int i = 0; i < 3; ++i, ++valueIt) {
            *valueIt = std::max(0.0f, *valueIt);
        }
    }

    inline void sanitize_ref_white_or_dwg(const float* white, float dst[3]) {
        const float fallback[3] = {
            gDWG_WhitePoint_XYZ[0],
            gDWG_WhitePoint_XYZ[1],
            gDWG_WhitePoint_XYZ[2]};
        const float* src = white ? white : fallback;
        float* dstIt = dst;
        const float* srcIt = src;
        const float* fallbackIt = fallback;
        for (int i = 0; i < 3; ++i, ++dstIt, ++srcIt, ++fallbackIt) {
            *dstIt = is_finite_sp(*srcIt) ? *srcIt : *fallbackIt;
        }
        if (!(dst[1] > 0.0f)) {
            copy_triplet3(fallback, dst);
        }
    }

    // Tri->quad mapping identical to agx-emulsion
    inline void tri2quad(float tx, float ty, float& qx, float& qy) {
        const float denom = std::max(1.0f - tx, 1e-10f);
        float x = (1.0f - tx);
        x = x * x;
        float y = ty / denom;
        qx = std::clamp(x, 0.0f, 1.0f);
        qy = std::clamp(y, 0.0f, 1.0f);
    }

    inline void hanatos_linear_spectrum(float qx, float qy, std::vector<float>& Ee_out) {
        const int K = gShape.K;
        Ee_out.resize(K);

        if (gHanSpectra.size <= 0) {
            std::fill(Ee_out.begin(), Ee_out.end(), 0.0f);
            return;
        }

        const int N = gHanSpectra.size;
        const float gridMax = static_cast<float>(N - 1);
        const float fx = std::clamp(qx, 0.0f, 1.0f) * gridMax;
        const float fy = std::clamp(qy, 0.0f, 1.0f) * gridMax;
        const int x0 = std::clamp(static_cast<int>(std::floor(fx)), 0, N - 1);
        const int y0 = std::clamp(static_cast<int>(std::floor(fy)), 0, N - 1);
        const int x1 = std::min(x0 + 1, N - 1);
        const int y1 = std::min(y0 + 1, N - 1);
        const float tx = fx - static_cast<float>(x0);
        const float ty = fy - static_cast<float>(y0);

        // Bilinear interpolation matching scipy.interpolate.RegularGridInterpolator
        for (int k = 0; k < K; ++k) {
            const size_t idx00 = ((static_cast<size_t>(x0) * N + y0) * static_cast<size_t>(gHanSpectra.numSamples) + k);
            const size_t idx10 = ((static_cast<size_t>(x1) * N + y0) * static_cast<size_t>(gHanSpectra.numSamples) + k);
            const size_t idx01 = ((static_cast<size_t>(x0) * N + y1) * static_cast<size_t>(gHanSpectra.numSamples) + k);
            const size_t idx11 = ((static_cast<size_t>(x1) * N + y1) * static_cast<size_t>(gHanSpectra.numSamples) + k);

            const float v00 = gHanSpectra.data[idx00];
            const float v10 = gHanSpectra.data[idx10];
            const float v01 = gHanSpectra.data[idx01];
            const float v11 = gHanSpectra.data[idx11];

            // Bilinear: (1-tx)*(1-ty)*v00 + tx*(1-ty)*v10 + (1-tx)*ty*v01 + tx*ty*v11
            const float v0 = v00 * (1.0f - tx) + v10 * tx;
            const float v1 = v01 * (1.0f - tx) + v11 * tx;
            Ee_out[k] = v0 * (1.0f - ty) + v1 * ty;
        }
    }

    // Reconstruct Ee via Hanatos spectra LUT from DWG RGB (match agx-emulsion behavior)
    // Per agx-emulsion parity: applies CAT02 chromatic adaptation from DWG D65 white point
    // to reference illuminant white point before computing xy chromaticity.
    inline void reconstruct_Ee_from_DWG_RGB_hanatos(
        const float rgbDWG[3],
        std::vector<float>& Ee_out,
        const float refIllumWhiteXYZ[3]) {
        const int K = gShape.K;
        Ee_out.resize(K);

        // Convert DWG RGB to XYZ (DWG uses D65 white point)
        float XYZ[3];
        const auto convertedXyz = JuicerColor::dwg_to_xyz({rgbDWG[0], rgbDWG[1], rgbDWG[2]});
        std::copy(convertedXyz.begin(), convertedXyz.end(), XYZ);
        // agx-emulsion parity: keep signed XYZ; only sanitize non-finite components.
        sanitize_nonfinite_triplet(XYZ);

        float refWhiteXYZ[3];
        sanitize_ref_white_or_dwg(refIllumWhiteXYZ, refWhiteXYZ);

        // Apply CAT02 chromatic adaptation from D65 to reference illuminant.
        // This matches Python: colour.RGB_to_XYZ(..., illuminant=ref_illum, chromatic_adaptation_transform='CAT02')
        ChromaticAdaptationWhites whites{};
        whites.source = gDWG_WhitePoint_XYZ;
        whites.destination = refWhiteXYZ;
        auto adaptedXYZ = JuicerColor::adapt_cat02({XYZ[0], XYZ[1], XYZ[2]}, whites);

        // agx-emulsion parity:
        // - sanitize adaptedXYZ (non-finite -> 0), do not clamp negatives
        // - b = sum(adaptedXYZ) (signed)
        // - xy uses denom = max(b, 1e-10) and is then clamped to [0, 1]
        sanitize_nonfinite_triplet(adaptedXYZ.data());
        const float b = adaptedXYZ[0] + adaptedXYZ[1] + adaptedXYZ[2];
        const float bSafe = sanitize_nonfinite_component(b);

        const float denom = std::max(bSafe, 1e-10f);
        float x = adaptedXYZ[0] / denom;
        float y = adaptedXYZ[1] / denom;
        x = std::clamp(x, 0.0f, 1.0f);
        y = std::clamp(y, 0.0f, 1.0f);

        float qx, qy;
        tri2quad(x, y, qx, qy);

        hanatos_linear_spectrum(qx, qy, Ee_out); // Use bilinear to match Python's RegularGridInterpolator

        // Multiply by b to get final Ee spectrum (signed; matches Python).
        for (int i = 0; i < K; ++i) {
            Ee_out[i] = sanitize_nonfinite_component(bSafe * Ee_out[i]);
        }
    }

    inline void reconstruct_Ee_from_DWG_RGB_with_tables(
        const float rgbDWG[3],
        const SpectralTables& T,
        const float S_inv[9],
        std::vector<float>& Ee_out) {
        float XYZ[3];
        const auto convertedXyz = JuicerColor::dwg_to_xyz({rgbDWG[0], rgbDWG[1], rgbDWG[2]});
        std::copy(convertedXyz.begin(), convertedXyz.end(), XYZ);

        float sanitizedXYZ[3];
        sanitize_nonnegative_triplet_sp(sanitizedXYZ, XYZ);

        float refWhite[3];
        sanitize_nonnegative_triplet_sp(refWhite, T.refIllumWhiteXYZ);
        if (refWhite[1] <= 0.0f) {
            copy_triplet3(gDWG_WhitePoint_XYZ, refWhite);
        }

        ChromaticAdaptationWhites whites{};
        whites.source = gDWG_WhitePoint_XYZ;
        whites.destination = refWhite;
        auto adaptedXYZ = JuicerColor::adapt_cat02({sanitizedXYZ[0], sanitizedXYZ[1], sanitizedXYZ[2]}, whites);

        clamp_triplet_nonnegative(adaptedXYZ.data());

        const float targetScale = (adaptedXYZ[1] > 0.0f) ? adaptedXYZ[1] : sanitizedXYZ[1];

        float cx = S_inv[0] * adaptedXYZ[0] + S_inv[1] * adaptedXYZ[1] + S_inv[2] * adaptedXYZ[2];
        float cy = S_inv[3] * adaptedXYZ[0] + S_inv[4] * adaptedXYZ[1] + S_inv[5] * adaptedXYZ[2];
        float cz = S_inv[6] * adaptedXYZ[0] + S_inv[7] * adaptedXYZ[1] + S_inv[8] * adaptedXYZ[2];
        cx = std::max(0.0f, cx);
        cy = std::max(0.0f, cy);
        cz = std::max(0.0f, cz);

        const int K = T.K;
        Ee_out.resize(K);
        float* eeData = Ee_out.data();
        const float* axData = T.Ax.data();
        const float* ayData = T.Ay.data();
        const float* azData = T.Az.data();
        double Y_recon = 0.0;
        for (int i = 0; i < K; ++i) {
            const float bx = std::max(0.0f, axData[i]);
            const float by = std::max(0.0f, ayData[i]);
            const float bz = std::max(0.0f, azData[i]);
            const float Ei = std::max(1e-6f, cx * bx + cy * by + cz * bz);
            eeData[i] = Ei;
            Y_recon += static_cast<double>(Ei) * static_cast<double>(ayData[i]);
        }
        if (Y_recon > 1e-20 && targetScale > 0.0f) {
            const float s = static_cast<float>(static_cast<double>(targetScale) / Y_recon);
            float* eeIt = eeData;
            for (int i = 0; i < K; ++i, ++eeIt) {
                *eeIt = std::max(0.0f, s * (*eeIt));
            }
        } else if (targetScale <= 0.0f) {
            std::fill(Ee_out.begin(), Ee_out.end(), 0.0f);
        }
    }

    // Per-instance SPD integration used by host preparation and CUDA parity probes.
    inline void layerExposures_from_sceneSPD_with_curves(
        const std::vector<float>& Ee,
        const Curve& sB,
        const Curve& sG,
        const Curve& sR,
        float E[3],
        float exposureScale,
        bool applyDeltaLambda = true) {
        const int K = gShape.K;
        double exposureBlue = 0.0;
        double exposureGreen = 0.0;
        double exposureRed = 0.0;
        for (int i = 0; i < K; ++i) {
            const float irradiance =
                i < static_cast<int>(Ee.size()) ? Ee[i] : 0.0f;
            if (!std::isfinite(irradiance)) {
                continue;
            }

            const double irradiance64 = static_cast<double>(irradiance);
            const float blue = sB.linear.empty() ? 0.0f : sB.linear[i];
            const float green = sG.linear.empty() ? 0.0f : sG.linear[i];
            const float red = sR.linear.empty() ? 0.0f : sR.linear[i];

            if (std::isfinite(blue)) {
                exposureBlue += irradiance64 * static_cast<double>(blue);
            }
            if (std::isfinite(green)) {
                exposureGreen += irradiance64 * static_cast<double>(green);
            }
            if (std::isfinite(red)) {
                exposureRed += irradiance64 * static_cast<double>(red);
            }
        }
        const float safeScale = std::max(0.0f, exposureScale);
        const double delta =
            applyDeltaLambda ? static_cast<double>(kDelta) : 1.0;
        const double scale = delta * static_cast<double>(safeScale);
        E[0] = std::max(
            0.0f,
            static_cast<float>(exposureBlue * scale));
        E[1] = std::max(
            0.0f,
            static_cast<float>(exposureGreen * scale));
        E[2] = std::max(
            0.0f,
            static_cast<float>(exposureRed * scale));
    }

    // Per-instance SPD exposure using per-instance sensitivity curves
    inline void rgbDWG_to_layerExposures_from_tables_with_curves(
        const float rgbDWG[3], float E[3], float exposureScale, const SpectralTables* T, const float* S_inv /* size 9 */, const Curve& sB, const Curve& sG, const Curve& sR, SpectralUpsamplingMode spectralUpsamplingMode = SpectralUpsamplingMode::PreferHanatos, const float refIllumWhiteXYZ[3] /* optional override */ = nullptr) {
        if (!T || !S_inv || T->K <= 0) {
            E[0] = E[1] = E[2] = 0.0f;
            return;
        }

        thread_local std::vector<float> Ee_scene;
        Ee_scene.resize(T->K);

        const bool allowHanatos = (spectralUpsamplingMode == SpectralUpsamplingMode::PreferHanatos);
        const bool useHanatos = allowHanatos && hanatos_available() && hanatos_matches_reference_shape();
        const float* adaptWhite = refIllumWhiteXYZ ? refIllumWhiteXYZ : T->refIllumWhiteXYZ;
        if (useHanatos) {
            // Pass reference illuminant white point for chromatic adaptation
            reconstruct_Ee_from_DWG_RGB_hanatos(rgbDWG, Ee_scene, adaptWhite);
        } else {
            reconstruct_Ee_from_DWG_RGB_with_tables(rgbDWG, *T, S_inv, Ee_scene);
        }

        // Integrate with per-instance sensitivity curves (no globals)
        layerExposures_from_sceneSPD_with_curves(Ee_scene, sB, sG, sR, E, exposureScale, !useHanatos);
    }

} // namespace Spectral
