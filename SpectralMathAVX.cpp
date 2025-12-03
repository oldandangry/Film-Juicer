#include "SpectralProcessing.h"
#include "SpectralData.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>

namespace Spectral {

#if defined(__AVX2__)

    // Accurate exp implementation adapted from the Cephes vector routine (public domain).
    __m256 exp_ps(__m256 x) {
        const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
        const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
        const __m256 log2ef = _mm256_set1_ps(1.44269504088896341f);
        const __m256 half = _mm256_set1_ps(0.5f);
        const __m256 one = _mm256_set1_ps(1.0f);
        const __m256 c1 = _mm256_set1_ps(0.693359375f);
        const __m256 c2 = _mm256_set1_ps(-2.12194440e-4f);
        const __m256 p0 = _mm256_set1_ps(1.9875691500E-4f);
        const __m256 p1 = _mm256_set1_ps(1.3981999507E-3f);
        const __m256 p2 = _mm256_set1_ps(8.3334519073E-3f);
        const __m256 p3 = _mm256_set1_ps(4.1665795894E-2f);
        const __m256 p4 = _mm256_set1_ps(1.6666665459E-1f);
        const __m256 p5 = _mm256_set1_ps(5.0000001201E-1f);

        __m256 x_clamped = _mm256_max_ps(_mm256_min_ps(x, exp_hi), exp_lo);

        __m256 fx = _mm256_fmadd_ps(x_clamped, log2ef, half);
        __m256i emm0 = _mm256_cvttps_epi32(fx);
        __m256 tmp = _mm256_cvtepi32_ps(emm0);

        __m256 mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
        mask = _mm256_and_ps(mask, one);
        fx = _mm256_sub_ps(tmp, mask);

        __m256 x_reduced = _mm256_fnmadd_ps(fx, c1, x_clamped);
        x_reduced = _mm256_fnmadd_ps(fx, c2, x_reduced);

        __m256 z = _mm256_mul_ps(x_reduced, x_reduced);

        __m256 y = p0;
        y = _mm256_fmadd_ps(y, x_reduced, p1);
        y = _mm256_fmadd_ps(y, x_reduced, p2);
        y = _mm256_fmadd_ps(y, x_reduced, p3);
        y = _mm256_fmadd_ps(y, x_reduced, p4);
        y = _mm256_fmadd_ps(y, x_reduced, p5);

        y = _mm256_fmadd_ps(y, z, x_reduced);
        y = _mm256_add_ps(y, one);

        emm0 = _mm256_cvttps_epi32(fx);
        emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(0x7f));
        emm0 = _mm256_slli_epi32(emm0, 23);

        __m256 pow2n = _mm256_castsi256_ps(emm0);
        return _mm256_mul_ps(y, pow2n);
    }

    float hsum_avx(__m256 v) {
        __m128 vlow = _mm256_castps256_ps128(v);
        __m128 vhigh = _mm256_extractf128_ps(v, 1);
        __m128 vsum = _mm_add_ps(vlow, vhigh);
        vsum = _mm_hadd_ps(vsum, vsum);
        vsum = _mm_hadd_ps(vsum, vsum);
        float out;
        _mm_store_ss(&out, vsum);
        return out;
    }

    void integrate_dyes_to_XYZ_avx2(
        float dY, float dM, float dC,
        const float* epsY, const float* epsM, const float* epsC,
        const float* Ax, const float* Ay, const float* Az,
        int K,
        const BaselineCtx& base,
        float XYZ_out[3])
    {
        const __m256 kLN10 = _mm256_set1_ps(kLn10);
        const __m256 dYv = _mm256_set1_ps(dY);
        const __m256 dMv = _mm256_set1_ps(dM);
        const __m256 dCv = _mm256_set1_ps(dC);

        __m256 sumX = _mm256_setzero_ps();
        __m256 sumY = _mm256_setzero_ps();
        __m256 sumZ = _mm256_setzero_ps();

        int i = 0;
        const int step = 8;
        for (; i + step <= K; i += step) {
            __m256 eY = _mm256_loadu_ps(epsY + i);
            __m256 eM = _mm256_loadu_ps(epsM + i);
            __m256 eC = _mm256_loadu_ps(epsC + i);

            __m256 Dlambda = _mm256_fmadd_ps(dYv, eY, _mm256_fmadd_ps(dMv, eM, _mm256_mul_ps(dCv, eC)));

            if (base.hasBaseline && base.baseMin) {
                __m256 bMin = _mm256_loadu_ps(base.baseMin + i);
                Dlambda = _mm256_add_ps(Dlambda, bMin);
            }

            __m256 valid = _mm256_cmp_ps(Dlambda, Dlambda, _CMP_ORD_Q);
            __m256 T = exp_ps(_mm256_mul_ps(_mm256_set1_ps(-1.0f), _mm256_mul_ps(kLN10, Dlambda)));
            T = _mm256_and_ps(T, valid);

            __m256 Xv = _mm256_mul_ps(T, _mm256_loadu_ps(Ax + i));
            __m256 Yv = _mm256_mul_ps(T, _mm256_loadu_ps(Ay + i));
            __m256 Zv = _mm256_mul_ps(T, _mm256_loadu_ps(Az + i));

            sumX = _mm256_add_ps(sumX, Xv);
            sumY = _mm256_add_ps(sumY, Yv);
            sumZ = _mm256_add_ps(sumZ, Zv);
        }

        float X = hsum_avx(sumX);
        float Y = hsum_avx(sumY);
        float Z = hsum_avx(sumZ);

        for (; i < K; ++i) {
            float Dlambda = dY * epsY[i] + dM * epsM[i] + dC * epsC[i];
            if (base.hasBaseline && base.baseMin) {
                Dlambda += base.baseMin[i];
            }
            if (!std::isfinite(Dlambda)) {
                continue;
            }
            float T = std::exp(-kLn10 * Dlambda);
            X += T * Ax[i];
            Y += T * Ay[i];
            Z += T * Az[i];
        }

        float scale = gInvYn;
        XYZ_out[0] = X * scale;
        XYZ_out[1] = Y * scale;
        XYZ_out[2] = Z * scale;
    }

#endif // __AVX2__

} // namespace Spectral
