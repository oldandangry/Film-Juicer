// Canonical spectral axis and native upload values.
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace Spectral {

    inline constexpr float kLambdaMin = 380.0f;
    inline constexpr float kLambdaMax = 780.0f;
    inline constexpr float kDelta = 5.0f;
    inline constexpr int kNumSamples = static_cast<int>((kLambdaMax - kLambdaMin) / kDelta) + 1;
    static_assert(kNumSamples == 81, "Spectral grid must be 380..780 nm at 5 nm.");

    inline constexpr int kFilmTcLutExtent = 192;
    inline constexpr int kFilmTcLutChannels = 4;

    struct Curve {
        std::vector<float> lambda_nm;
        std::vector<float> linear;
    };

    struct SpectralShape {
        static constexpr float lambdaMin = kLambdaMin;
        static constexpr float lambdaMax = kLambdaMax;
        static constexpr float delta = kDelta;
        static constexpr int K = kNumSamples;

        std::array<float, kNumSamples> wavelengths{};

        constexpr SpectralShape() : wavelengths{} {
            for (int i = 0; i < kNumSamples; ++i) {
                wavelengths[static_cast<std::size_t>(i)] = lambdaMin + delta * static_cast<float>(i);
            }
        }
    };

    inline constexpr SpectralShape kReferenceShape{};

    // DWG working space white point (D65)
    inline constexpr float gDWG_WhitePoint_XYZ[3] = {
        0.950455f, 1.0f, 1.089058f};

    // Spectral upsampling / SPD reconstruction selection.
    // "Mallett" identifies the Mallett 2019 sRGB basis reconstruction.
    enum class SpectralUpsamplingMode : std::uint8_t {
        PreferHanatos = 0,
        ForceMallett = 1,
        Arctic2026beta04 = 2
    };

    inline bool spectral_shape_matches_reference(const SpectralShape& s) {
        for (int i = 0; i < SpectralShape::K; ++i) {
            const float expected = kLambdaMin + static_cast<float>(i) * kDelta;
            if (std::abs(s.wavelengths[static_cast<std::size_t>(i)] - expected) > 1e-3f) {
                return false;
            }
        }
        return true;
    }

    inline void assign_reference_axis(std::vector<float>& lambda) {
        lambda.assign(kReferenceShape.wavelengths.begin(), kReferenceShape.wavelengths.end());
    }

} // namespace Spectral
