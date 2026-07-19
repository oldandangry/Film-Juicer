// FilmProcessing.h
// Film density-curve data and sampling shared by host preparation and validation.
#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

#include "SpectralData.h"

namespace Spectral {

    struct NegativeCouplerParams {
        float DmaxY;
        float DmaxM;
        float DmaxC;
        float baseY;
        float baseM;
        float baseC;
        float kB;
        float kG;
        float kR;
        float mask[9];
        float maskScale[3];
        float maskOffset[3];
    };

    inline float sample_density_at_logE(
        const Curve& curve,
        float logExposure,
        float gammaFactor = 1.0f) {
        const std::size_t count = curve.lambda_nm.size();
        if (count == 0 || curve.linear.size() != count) {
            return 0.0f;
        }

        // Spektrafilm fast_interp parity preserves authored NaN toe samples.
        if (!std::isfinite(logExposure)) {
            return std::numeric_limits<float>::quiet_NaN();
        }

        const float gammaSafe =
            std::isfinite(gammaFactor) && gammaFactor > 0.0f
                ? gammaFactor
                : 1.0f;
        const float query = logExposure * gammaSafe;

        std::size_t domainBegin = 0;
        while (domainBegin < count &&
               !std::isfinite(curve.lambda_nm[domainBegin])) {
            ++domainBegin;
        }
        if (domainBegin == count) {
            return 0.0f;
        }

        std::size_t domainEnd = count - 1;
        while (domainEnd > domainBegin &&
               !std::isfinite(curve.lambda_nm[domainEnd])) {
            --domainEnd;
        }

        const float minimum = curve.lambda_nm[domainBegin];
        const float maximum = curve.lambda_nm[domainEnd];
        if (!std::isfinite(minimum) || !std::isfinite(maximum) ||
            !(maximum >= minimum)) {
            return curve.linear[domainBegin];
        }

        if (query <= minimum) {
            return curve.linear[domainBegin];
        }
        if (query >= maximum) {
            return curve.linear[domainEnd];
        }

        std::size_t upper = domainBegin + 1;
        while (upper <= domainEnd && curve.lambda_nm[upper] < query) {
            ++upper;
        }
        if (upper > domainEnd) {
            return curve.linear[domainEnd];
        }

        const std::size_t lower = upper - 1;
        const float x0 = curve.lambda_nm[lower];
        const float x1 = curve.lambda_nm[upper];
        const float y0 = curve.linear[lower];
        const float y1 = curve.linear[upper];

        const float denominator = x1 - x0;
        if (!(denominator > 0.0f) || !std::isfinite(denominator)) {
            return y0;
        }

        const float t = (query - x0) / denominator;
        return y0 + t * (y1 - y0);
    }

} // namespace Spectral
