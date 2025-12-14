// GaussianSciPy.h
#pragma once

#include <cmath>

namespace JuicerGaussian {

    constexpr double kSciPyGaussianTruncate = 4.0;

    inline int scipy_gaussian_radius(double sigma, double truncate = kSciPyGaussianTruncate) {
        if (!std::isfinite(sigma) || !(sigma > 0.0)) {
            return 0;
        }
        if (!std::isfinite(truncate) || !(truncate > 0.0)) {
            return 0;
        }
        const double r = truncate * sigma + 0.5;
        if (!(r > 0.0) || !std::isfinite(r)) {
            return 0;
        }
        return static_cast<int>(r);
    }

    inline int scipy_gaussian_radius(float sigma, float truncate = static_cast<float>(kSciPyGaussianTruncate)) {
        if (!std::isfinite(sigma) || !(sigma > 0.0f)) {
            return 0;
        }
        if (!std::isfinite(truncate) || !(truncate > 0.0f)) {
            return 0;
        }
        const float r = truncate * sigma + 0.5f;
        if (!(r > 0.0f) || !std::isfinite(r)) {
            return 0;
        }
        return static_cast<int>(r);
    }

} // namespace JuicerGaussian

