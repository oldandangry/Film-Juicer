// SpectralProcessing.h
// Spectral processing operations: SPD reconstruction, table operations, integration, and math utilities

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "ColorTransforms.h"
#include "Hash.h"
#include "SpectralData.h"

namespace Spectral {

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

    inline std::uint64_t hash_float_span_digest_sp(const float* values, std::size_t count) {
        const Hash::FloatSpanHash h = Hash::hash_float_span_with_nan_mask(values, count);
        const std::uint64_t fields[] = {h.valueHash, h.nanMaskHash};
        return Hash::hash_bytes(fields, sizeof(fields));
    }

    inline std::uint64_t hash_float_vector_digest_sp(const std::vector<float>& values) {
        return hash_float_span_digest_sp(values.data(), values.size());
    }

    inline std::uint64_t hash_float_scalar_digest_sp(float value) {
        return hash_float_span_digest_sp(&value, 1);
    }

    inline std::uint64_t hash_float_triplet_digest_sp(const float values[3]) {
        return hash_float_span_digest_sp(values, 3);
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

    inline void set_identity_3x3(float matrix[9]) {
        std::fill_n(matrix, 9, 0.0f);
        matrix[0] = 1.0f;
        matrix[4] = 1.0f;
        matrix[8] = 1.0f;
    }

    struct RowMajor3x3d {
        double m00 = 0.0;
        double m01 = 0.0;
        double m02 = 0.0;
        double m10 = 0.0;
        double m11 = 0.0;
        double m12 = 0.0;
        double m20 = 0.0;
        double m21 = 0.0;
        double m22 = 0.0;
    };

    inline void store_3x3_rowmajor(float matrix[9], const RowMajor3x3d& values) {
        matrix[0] = static_cast<float>(values.m00);
        matrix[1] = static_cast<float>(values.m01);
        matrix[2] = static_cast<float>(values.m02);
        matrix[3] = static_cast<float>(values.m10);
        matrix[4] = static_cast<float>(values.m11);
        matrix[5] = static_cast<float>(values.m12);
        matrix[6] = static_cast<float>(values.m20);
        matrix[7] = static_cast<float>(values.m21);
        matrix[8] = static_cast<float>(values.m22);
    }

    inline bool determinant_near_zero(double determinant, double epsilon = 1e-20) {
        return std::fabs(determinant) < epsilon;
    }

    inline void compute_S_inverse_from_tables(const SpectralTables& T, float S_inv_out[9]) {
        // S = ∫ [max(0, CMF)]^T * [CMF] dλ under viewing axis in T
        double Sxx = 0, Sxy = 0, Sxz = 0, Syx = 0, Syy = 0, Syz = 0, Szx = 0, Szy = 0, Szz = 0;
        const int K = T.K;
        for (int i = 0; i < K; ++i) {
            const float x = T.Ax[i], y = T.Ay[i], z = T.Az[i];
            const float bx = std::max(0.0f, x);
            const float by = std::max(0.0f, y);
            const float bz = std::max(0.0f, z);
            Sxx += bx * x;
            Sxy += bx * y;
            Sxz += bx * z;
            Syx += by * x;
            Syy += by * y;
            Syz += by * z;
            Szx += bz * x;
            Szy += bz * y;
            Szz += bz * z;
        }
        const double dl = static_cast<double>(T.deltaLambda);
        Sxx *= dl;
        Sxy *= dl;
        Sxz *= dl;
        Syx *= dl;
        Syy *= dl;
        Syz *= dl;
        Szx *= dl;
        Szy *= dl;
        Szz *= dl;

        const double det = Sxx * (Syy * Szz - Syz * Szy) - Sxy * (Syx * Szz - Syz * Szx) + Sxz * (Syx * Szy - Syy * Szx);
        if (determinant_near_zero(det)) {
            set_identity_3x3(S_inv_out);
            return;
        }
        const double invDet = 1.0 / det;
        store_3x3_rowmajor(
            S_inv_out,
            RowMajor3x3d{
                (Syy * Szz - Syz * Szy) * invDet,
                (Sxz * Szy - Sxy * Szz) * invDet,
                (Sxy * Syz - Sxz * Syy) * invDet,
                (Syz * Szx - Syx * Szz) * invDet,
                (Sxx * Szz - Sxz * Szx) * invDet,
                (Sxz * Syx - Sxx * Syz) * invDet,
                (Syx * Szy - Syy * Szx) * invDet,
                (Sxy * Szx - Sxx * Szy) * invDet,
                (Sxx * Syy - Sxy * Syx) * invDet});
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
        DWG_linear_to_XYZ(rgbDWG, XYZ);
        // agx-emulsion parity: keep signed XYZ; only sanitize non-finite components.
        sanitize_nonfinite_triplet(XYZ);

        float refWhiteXYZ[3];
        sanitize_ref_white_or_dwg(refIllumWhiteXYZ, refWhiteXYZ);

        // Apply CAT02 chromatic adaptation from D65 to reference illuminant.
        // This matches Python: colour.RGB_to_XYZ(..., illuminant=ref_illum, chromatic_adaptation_transform='CAT02')
        float adaptedXYZ[3];
        ChromaticAdaptationWhites whites{};
        whites.source = gDWG_WhitePoint_XYZ;
        whites.destination = refWhiteXYZ;
        chromatic_adapt_XYZ_CAT02(XYZ, whites, adaptedXYZ);

        // agx-emulsion parity:
        // - sanitize adaptedXYZ (non-finite -> 0), do not clamp negatives
        // - b = sum(adaptedXYZ) (signed)
        // - xy uses denom = max(b, 1e-10) and is then clamped to [0, 1]
        sanitize_nonfinite_triplet(adaptedXYZ);
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

    inline void build_tables_from_curves_non_global(
        const Curve& epsY, const Curve& epsM, const Curve& epsC, const Curve& xbar, const Curve& ybar, const Curve& zbar, const Curve& illumView, const Curve& baseDensityMin, const Curve& baseDensityMid, bool hasBaseline, float densityBaselineMixReference, SpectralTables& T, std::uint64_t illuminantHash = 0) {
        (void)densityBaselineMixReference;
        const int K = gShape.K;
        T.K = K;
        T.lambda.assign(gShape.wavelengths.begin(), gShape.wavelengths.end());
        T.deltaLambda = kDelta;

        T.epsY.resize(K);
        T.epsM.resize(K);
        T.epsC.resize(K);
        T.Xbar.resize(K);
        T.Ybar.resize(K);
        T.Zbar.resize(K);
        T.Ax.resize(K);
        T.Ay.resize(K);
        T.Az.resize(K);
        T.illum.resize(K);

        const bool hasIll = (!illumView.linear.empty() && (int)illumView.linear.size() == K);
        double Yn = 0.0;
        double sumAx = 0.0;
        double sumAy = 0.0;
        double sumAz = 0.0;
        const bool hasEpsY = (!epsY.linear.empty() && (int)epsY.linear.size() == K);
        const bool hasEpsM = (!epsM.linear.empty() && (int)epsM.linear.size() == K);
        const bool hasEpsC = (!epsC.linear.empty() && (int)epsC.linear.size() == K);
        const bool hasXbar = (!xbar.linear.empty() && (int)xbar.linear.size() == K);
        const bool hasYbar = (!ybar.linear.empty() && (int)ybar.linear.size() == K);
        const bool hasZbar = (!zbar.linear.empty() && (int)zbar.linear.size() == K);
        const float* epsYData = hasEpsY ? epsY.linear.data() : nullptr;
        const float* epsMData = hasEpsM ? epsM.linear.data() : nullptr;
        const float* epsCData = hasEpsC ? epsC.linear.data() : nullptr;
        const float* xbarData = hasXbar ? xbar.linear.data() : nullptr;
        const float* ybarData = hasYbar ? ybar.linear.data() : nullptr;
        const float* zbarData = hasZbar ? zbar.linear.data() : nullptr;
        const float* illumData = hasIll ? illumView.linear.data() : nullptr;
        const float* lambdaData = T.lambda.data();
        float* outEpsY = T.epsY.data();
        float* outEpsM = T.epsM.data();
        float* outEpsC = T.epsC.data();
        float* outXbar = T.Xbar.data();
        float* outYbar = T.Ybar.data();
        float* outZbar = T.Zbar.data();
        float* outIllum = T.illum.data();
        float* outAx = T.Ax.data();
        float* outAy = T.Ay.data();
        float* outAz = T.Az.data();
        const auto gaussian = [](float wavelength, float center, float sigma) {
            const float offset = (wavelength - center) / sigma;
            return std::exp(-0.5f * offset * offset);
        };
        for (int i = 0; i < K; ++i) {
            const float l = lambdaData[i];

            const float ey = epsYData ? epsYData[i] : 1.4f * gaussian(l, 440.0f, 25.0f);
            const float em = epsMData ? epsMData[i] : 1.2f * gaussian(l, 540.0f, 30.0f);
            const float ec = epsCData ? epsCData[i] : 1.1f * gaussian(l, 610.0f, 35.0f);
            outEpsY[i] = ey;
            outEpsM[i] = em;
            outEpsC[i] = ec;

            outXbar[i] = xbarData
                             ? xbarData[i]
                             : gaussian(l, 595.0f, 40.0f) +
                                   0.25f * gaussian(l, 445.0f, 20.0f);
            outYbar[i] = ybarData ? ybarData[i] : gaussian(l, 555.0f, 30.0f);
            outZbar[i] = zbarData ? zbarData[i] : 1.2f * gaussian(l, 445.0f, 25.0f);

            const float Ee = illumData ? illumData[i] : 1.0f;
            outIllum[i] = Ee;
            outAx[i] = Ee * outXbar[i];
            outAy[i] = Ee * outYbar[i];
            outAz[i] = Ee * outZbar[i];
            Yn += outAy[i];
            sumAx += outAx[i];
            sumAy += outAy[i];
            sumAz += outAz[i];
        }
        T.invYn = (Yn > 0.0) ? (1.0f / (float)Yn) : 1.0f;
        const double scale = static_cast<double>(T.invYn);
        T.whiteXYZ[0] = static_cast<float>(sumAx * scale);
        T.whiteXYZ[1] = static_cast<float>(sumAy * scale);
        T.whiteXYZ[2] = static_cast<float>(sumAz * scale);

        // Illuminant white point used for chromatic adaptation (normalized to Y=1).
        copy_triplet3(T.whiteXYZ, T.refIllumWhiteXYZ);

        T.hasBaseline = hasBaseline &&
                        (int)baseDensityMin.linear.size() == K;
        T.baseDensityMin.assign(K, 0.0f);
        T.baseDensityMid.assign(K, 0.0f);
        T.densityBaselineMixReference = 0.0f;
        if (T.hasBaseline) {
            T.baseDensityMin.assign(baseDensityMin.linear.begin(), baseDensityMin.linear.end());
            if ((int)baseDensityMid.linear.size() == K) {
                T.baseDensityMid.assign(baseDensityMid.linear.begin(), baseDensityMid.linear.end());
            }
        }

        T.illuminantHash = illuminantHash;
        if (T.illuminantHash == 0 && hasIll) {
            T.illuminantHash = hash_float_vector_digest_sp(illumView.linear);
        }
        const std::uint64_t tableFields[] = {
            T.illuminantHash,
            hash_float_vector_digest_sp(T.lambda),
            hash_float_scalar_digest_sp(T.deltaLambda),
            hash_float_scalar_digest_sp(T.invYn),
            hash_float_triplet_digest_sp(T.whiteXYZ),
            hash_float_triplet_digest_sp(T.refIllumWhiteXYZ),
            hash_float_vector_digest_sp(T.Ax),
            hash_float_vector_digest_sp(T.Ay),
            hash_float_vector_digest_sp(T.Az),
            hash_float_vector_digest_sp(T.illum),
            hash_float_vector_digest_sp(T.Xbar),
            hash_float_vector_digest_sp(T.Ybar),
            hash_float_vector_digest_sp(T.Zbar),
            hash_float_vector_digest_sp(T.epsC),
            hash_float_vector_digest_sp(T.epsM),
            hash_float_vector_digest_sp(T.epsY),
            hash_float_vector_digest_sp(T.baseDensityMin),
            hash_float_vector_digest_sp(T.baseDensityMid),
            hash_float_scalar_digest_sp(T.densityBaselineMixReference),
            Hash::hash_bytes(&T.hasBaseline, sizeof(T.hasBaseline))};
        T.tablesHash = Hash::hash_bytes(tableFields, sizeof(tableFields));
    }

    inline void reconstruct_Ee_from_DWG_RGB_with_tables(
        const float rgbDWG[3],
        const SpectralTables& T,
        const float S_inv[9],
        std::vector<float>& Ee_out) {
        float XYZ[3];
        DWG_linear_to_XYZ(rgbDWG, XYZ);

        float sanitizedXYZ[3];
        sanitize_nonnegative_triplet_sp(sanitizedXYZ, XYZ);

        float refWhite[3];
        sanitize_nonnegative_triplet_sp(refWhite, T.refIllumWhiteXYZ);
        if (refWhite[1] <= 0.0f) {
            copy_triplet3(gDWG_WhitePoint_XYZ, refWhite);
        }

        float adaptedXYZ[3];
        ChromaticAdaptationWhites whites{};
        whites.source = gDWG_WhitePoint_XYZ;
        whites.destination = refWhite;
        chromatic_adapt_XYZ_CAT02(sanitizedXYZ, whites, adaptedXYZ);

        clamp_triplet_nonnegative(adaptedXYZ);

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
