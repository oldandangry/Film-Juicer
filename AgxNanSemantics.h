#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "AkimaInterpolator.h"

// Shared primitives for consistent NaN handling.

template <typename T>
inline T nan_to_num_agx(T v)
{
    if (std::isnan(v)) {
        return static_cast<T>(0);
    }
    if (!std::isfinite(v)) {
        if (v > static_cast<T>(0)) {
            return std::numeric_limits<T>::max();
        }
        return std::numeric_limits<T>::lowest();
    }
    return v;
}

template <typename T>
inline T fmax_agx(T a, T b)
{
    // NumPy np.fmax behavior: return the non-NaN input when one side is NaN.
    return std::fmax(a, b);
}

inline void density_to_light_agx(
    const std::vector<float>& density_spectral,
    const std::vector<float>& illuminant,
    std::vector<float>& out_light)
{
    const size_t n = std::min(density_spectral.size(), illuminant.size());
    out_light.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const double density = static_cast<double>(density_spectral[i]);
        const double illum = static_cast<double>(illuminant[i]);
        const double transmitted = std::pow(10.0, -density) * illum;
        const float out = static_cast<float>(transmitted);
        out_light[i] = std::isnan(out) ? 0.0f : out;
    }
}

namespace AgxNanInternal {

    struct IndexedPair {
        float lambda = 0.0f;
        float value = 0.0f;
        size_t originalIndex = 0;
    };

    inline void dedup_pairs_keep_first_by_wavelength(
        const std::vector<std::pair<float, float>>& inPairs,
        std::vector<float>& outX,
        std::vector<float>& outY)
    {
        outX.clear();
        outY.clear();

        std::vector<IndexedPair> temp;
        temp.reserve(inPairs.size());
        for (size_t i = 0; i < inPairs.size(); ++i) {
            const float lambda = inPairs[i].first;
            const float value = inPairs[i].second;
            if (!std::isfinite(lambda) || !std::isfinite(value)) {
                continue;
            }
            temp.push_back({ lambda, value, i });
        }
        if (temp.empty()) {
            return;
        }

        std::sort(temp.begin(), temp.end(),
            [](const IndexedPair& a, const IndexedPair& b) {
                if (a.lambda != b.lambda) return a.lambda < b.lambda;
                return a.originalIndex < b.originalIndex;
            });

        constexpr float kLambdaDedupEps = 1e-5f;
        float currentLambda = temp[0].lambda;
        float currentValue = temp[0].value;
        size_t currentFirstIndex = temp[0].originalIndex;

        auto flush = [&]() {
            outX.push_back(currentLambda);
            outY.push_back(currentValue);
        };

        for (size_t i = 1; i < temp.size(); ++i) {
            const float lambda = temp[i].lambda;
            const float value = temp[i].value;
            const size_t idx = temp[i].originalIndex;

            if (std::abs(lambda - currentLambda) <= kLambdaDedupEps) {
                if (idx < currentFirstIndex) {
                    currentFirstIndex = idx;
                    currentValue = value;
                }
                continue;
            }

            flush();
            currentLambda = lambda;
            currentValue = value;
            currentFirstIndex = idx;
        }
        flush();
    }

} // namespace AgxNanInternal

inline std::vector<std::pair<float, float>> akima_resample_agx(
    const std::vector<std::pair<float, float>>& pairs,
    const float* axis_nm,
    size_t axis_count)
{
    std::vector<std::pair<float, float>> out;
    if (!axis_nm || axis_count == 0) {
        return out;
    }

    std::vector<float> xs;
    std::vector<float> ys;
    AgxNanInternal::dedup_pairs_keep_first_by_wavelength(pairs, xs, ys);
    if (xs.size() < 2) {
        return out;
    }

    Interpolation::AkimaInterpolator akima;
    if (!akima.build(xs, ys)) {
        return out;
    }

    out.reserve(axis_count);
    for (size_t i = 0; i < axis_count; ++i) {
        const float lambda = axis_nm[i];
        const float value = akima.evaluate(lambda); // out-of-domain => NaN
        out.emplace_back(lambda, value);
    }
    return out;
}
