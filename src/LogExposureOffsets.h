#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <utility>
#include <vector>

#include "FilmProcessing.h"

namespace RebuildWorkingState {
    namespace detail {
        inline std::optional<float> invert_density_curve(const Spectral::Curve& curve, float targetDensity) {
            if (!std::isfinite(targetDensity)) {
                return std::nullopt;
            }

            const size_t n = std::min(curve.linear.size(), curve.lambda_nm.size());
            if (n == 0) {
                return std::nullopt;
            }

            std::vector<std::pair<float, float>> samples;
            samples.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                const float logE = curve.lambda_nm[i];
                const float density = curve.linear[i];
                if (std::isfinite(logE) && std::isfinite(density)) {
                    samples.emplace_back(logE, density);
                }
            }

            if (samples.empty()) {
                return std::nullopt;
            }

            std::sort(samples.begin(), samples.end(), [](const auto& a, const auto& b) {
                return a.first < b.first;
                });

            if (samples.size() == 1) {
                return samples.front().first;
            }

            auto between = [](float value, float a, float b) {
                return (value >= a && value <= b) || (value <= a && value >= b);
                };

            const float firstDensity = samples.front().second;
            const float lastDensity = samples.back().second;
            const bool increasing = lastDensity >= firstDensity;

            if (increasing) {
                if (targetDensity <= firstDensity) {
                    return samples.front().first;
                }
                if (targetDensity >= lastDensity) {
                    return samples.back().first;
                }
            }
            else {
                if (targetDensity >= firstDensity) {
                    return samples.front().first;
                }
                if (targetDensity <= lastDensity) {
                    return samples.back().first;
                }
            }

            float prevLogE = samples.front().first;
            float prevDensity = samples.front().second;
            for (size_t i = 1; i < samples.size(); ++i) {
                const float logE = samples[i].first;
                const float density = samples[i].second;
                if (!std::isfinite(logE) || !std::isfinite(density)) {
                    prevLogE = logE;
                    prevDensity = density;
                    continue;
                }
                if (between(targetDensity, prevDensity, density)) {
                    if (density == prevDensity) {
                        return logE;
                    }
                    const float t = std::clamp((targetDensity - prevDensity) / (density - prevDensity), 0.0f, 1.0f);
                    return prevLogE + t * (logE - prevLogE);
                }
                prevLogE = logE;
                prevDensity = density;
            }

            return samples.back().first;
        }
    } // namespace detail

    inline std::array<float, 3> compute_mid_neutral_logE_offsets_rgb(
        const Spectral::Curve& densR,
        const Spectral::Curve& densG,
        const Spectral::Curve& densB,
        const std::array<float, 3>& densityMidRGB)
    {
        std::array<float, 3> result{ {0.0f, 0.0f, 0.0f} };

        if (auto logER = detail::invert_density_curve(densR, densityMidRGB[0])) {
            result[0] = *logER;
        }
        if (auto logEG = detail::invert_density_curve(densG, densityMidRGB[1])) {
            result[1] = *logEG;
        }
        if (auto logEB = detail::invert_density_curve(densB, densityMidRGB[2])) {
            result[2] = *logEB;
        }

        return result;
    }
} // namespace RebuildWorkingState