#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace Spectral {

    inline constexpr float kLambdaMin = 380.0f;
    inline constexpr float kLambdaMax = 780.0f;
    inline constexpr float kDelta = 5.0f;
    inline constexpr int kNumSamples = static_cast<int>((kLambdaMax - kLambdaMin) / kDelta) + 1;
    static_assert(kNumSamples == 81, "Spectral grid must be 380..780 nm at 5 nm.");

    // Log-exposure axis (matches agx-emulsion LOG_EXPOSURE np.linspace(-3,4,256))
    inline constexpr float kLogExposureMin = -3.0f;
    inline constexpr float kLogExposureMax = 4.0f;
    inline constexpr int kLogExposureSamples = 256;
    inline constexpr float kLogExposureDelta =
        (kLogExposureMax - kLogExposureMin) / static_cast<float>(kLogExposureSamples - 1);

    struct PrecomputeStatus {
        std::atomic<uint64_t> illumVersion{ 0 };
        std::atomic<uint64_t> lastPrecomputeIllumVersion{ ~uint64_t{ 0 } };
        std::atomic<bool> dirty{ true };
        std::atomic<uint64_t> shapeVersion{ 0 };
        std::atomic<uint64_t> lastPrecomputeShapeVersion{ ~uint64_t{ 0 } };
        std::atomic<uint64_t> mixVersion{ 0 };
    };

    struct BaselineCtx {
        bool hasBaseline;
        const float* baseMin;
        const float* baseMid;
        float mix;
    };

    struct Curve {
        std::vector<float> lambda_nm;
        std::vector<float> linear;

        inline void build_from_log10_pairs(const std::vector<std::pair<float, float>>& samples) {
            lambda_nm.clear();
            linear.clear();
            lambda_nm.reserve(samples.size());
            linear.reserve(samples.size());
            float peak = 0.0f;
            for (auto& p : samples) {
                lambda_nm.push_back(p.first);
                float lin = std::pow(10.0f, p.second);
                if (!std::isfinite(lin) || lin < 0.0f) {
                    lin = 0.0f;
                }
                linear.push_back(lin);
                if (lin > peak) peak = lin;
            }
            if (peak > 0.0f) {
                for (auto& v : linear) v /= peak;
            }
        }

        inline void build_from_linear_pairs(const std::vector<std::pair<float, float>>& samples) {
            lambda_nm.clear();
            linear.clear();
            if (samples.empty()) return;

            std::vector<std::pair<float, float>> s = samples;
            std::sort(s.begin(), s.end(), [](auto& a, auto& b) { return a.first < b.first; });

            lambda_nm.reserve(s.size());
            linear.reserve(s.size());
            for (auto& p : s) {
                lambda_nm.push_back(p.first);
                linear.push_back(p.second);
            }
        }

        inline float sample(float lambda) const {
            const size_t n = lambda_nm.size();
            if (n == 0) return 0.0f;
            if (lambda <= lambda_nm.front()) return linear.front();
            if (lambda >= lambda_nm.back()) return linear.back();
            size_t i1 = 1;
            while (i1 < n && lambda_nm[i1] < lambda) ++i1;
            size_t i0 = i1 - 1;
            float x0 = lambda_nm[i0], x1 = lambda_nm[i1];
            float y0 = linear[i0], y1 = linear[i1];
            if (!(std::isfinite(x0) && std::isfinite(x1)) || x1 <= x0) {
                return y0;
            }
            if (!std::isfinite(y0)) {
                y0 = std::isfinite(y1) ? y1 : 0.0f;
            }
            if (!std::isfinite(y1)) {
                y1 = y0;
            }
            float t = (lambda - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }
    };

    struct DirRuntimeSnapshot {
        bool active = false;
        float M[3][3] = { {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f} };
        float highShift = 0.0f;
        float dMax[3] = { 1.0f, 1.0f, 1.0f };
    };

    struct SpectralShape {
        static constexpr float lambdaMin = kLambdaMin;
        static constexpr float lambdaMax = kLambdaMax;
        static constexpr float delta = kDelta;
        static constexpr int K = kNumSamples;

        std::array<float, kNumSamples> wavelengths{};

        constexpr SpectralShape() : wavelengths{} {
            for (int i = 0; i < kNumSamples; ++i) {
                wavelengths[static_cast<size_t>(i)] = lambdaMin + delta * static_cast<float>(i);
            }
        }
    };

} // namespace Spectral
