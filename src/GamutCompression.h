#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <string>

#include "OutputColor.h"

namespace Spectral {
    struct Curve;
    struct FilmTcLut;
} // namespace Spectral

namespace Gamut {

    inline constexpr int kInputHullDirectionCount = 1024;
    inline constexpr int kOutputBoundaryLightnessCount = 64;
    inline constexpr int kOutputBoundaryHueCount = 720;
    inline constexpr float kOutputBoundaryMinimumLightness = 0.02f;
    inline constexpr float kOutputBoundaryMaximumLightness = 1.0f;
    inline constexpr float kOutputBoundaryCubeSlack = 1.0e-6f;
    inline constexpr float kOutputBoundaryMaximumChroma = 2.0f;
    inline constexpr std::size_t kOutputBoundaryValueCount =
        static_cast<std::size_t>(kOutputBoundaryLightnessCount) *
        static_cast<std::size_t>(kOutputBoundaryHueCount);

    struct OutputGamutTransform {
        std::array<float, 9> nativeRgbToD65Xyz{};
        std::array<float, 9> d65XyzToNativeRgb{};
        OutputEncoding::ColorSpace outputColorSpace =
            OutputEncoding::ColorSpace::sRGB;
        std::uint64_t hash = 0;
        bool valid = false;
    };

    struct OutputBoundaryTable {
        std::array<float, kOutputBoundaryValueCount> cmax{};
        std::uint64_t transformHash = 0;
        std::uint64_t contractHash = 0;
        std::uint64_t hash = 0;
        bool valid = false;
        std::string diagnostic;
    };

    // Pinned colour-science 0.4.6 OkLab matrices, shared by host construction
    // and the CUDA scalar path added at the execution boundary.
    inline constexpr std::array<float, 9> kOklabXyzToLms = {
        0.8189330101f,
        0.3618667424f,
        -0.1288597137f,
        0.0329845436f,
        0.9293118715f,
        0.0361456387f,
        0.0482003018f,
        0.2643662691f,
        0.6338517070f};
    inline constexpr std::array<float, 9> kOklabLmsToXyz = {
        1.2270138511f,
        -0.5577999807f,
        0.2812561490f,
        -0.0405801784f,
        1.1122568696f,
        -0.0716766787f,
        -0.0763812845f,
        -0.4214819784f,
        1.5861632204f};
    inline constexpr std::array<float, 9> kOklabLmsRootToLab = {
        0.2104542553f,
        0.7936177850f,
        -0.0040720468f,
        1.9779984951f,
        -2.4285922050f,
        0.4505937099f,
        0.0259040371f,
        0.7827717662f,
        -0.8086757660f};
    inline constexpr std::array<float, 9> kOklabLabToLmsRoot = {
        1.0f,
        0.3963377774f,
        0.2158037573f,
        1.0f,
        -0.1055613458f,
        -0.0638541728f,
        1.0f,
        -0.0894841775f,
        -1.2914855480f};

#if defined(__CUDACC__)
#define JUICER_GAMUT_HOST_DEVICE __host__ __device__
#else
#define JUICER_GAMUT_HOST_DEVICE
#endif

    JUICER_GAMUT_HOST_DEVICE inline float signed_cube_root(float value) {
        return value < 0.0f ? -cbrtf(-value) : cbrtf(value);
    }

    JUICER_GAMUT_HOST_DEVICE inline void multiply_float3x3(
        const float matrix[9],
        const float input[3],
        float output[3]) {
        output[0] = matrix[0] * input[0] + matrix[1] * input[1] +
                    matrix[2] * input[2];
        output[1] = matrix[3] * input[0] + matrix[4] * input[1] +
                    matrix[5] * input[2];
        output[2] = matrix[6] * input[0] + matrix[7] * input[1] +
                    matrix[8] * input[2];
    }

    struct OutputKneeParameters {
        float threshold = 0.95f;
        float limit = 1.0f;
        float power = 1.6f;
    };

    struct OutputBoundaryPosition {
        float lightness = 0.0f;
        float hue = 0.0f;
    };

    struct OutputCompressionView {
        const float* cmax = nullptr;
        const float* nativeRgbToD65Xyz = nullptr;
        const float* d65XyzToNativeRgb = nullptr;
        const float* oklabXyzToLms = nullptr;
        const float* oklabLmsToXyz = nullptr;
        const float* oklabLmsRootToLab = nullptr;
        const float* oklabLabToLmsRoot = nullptr;
        OutputKneeParameters lightnessKnee{};
        OutputKneeParameters chromaKnee{};
    };

    JUICER_GAMUT_HOST_DEVICE inline float output_reinhard_knee(
        float value,
        const OutputKneeParameters& knee) {
        if (!(value > knee.threshold)) {
            return value;
        }
        const float span = knee.limit - knee.threshold;
        const float normalized = (value - knee.threshold) / span;
        return knee.threshold +
               span * normalized /
                   powf(
                       1.0f + powf(normalized, knee.power),
                       1.0f / knee.power);
    }

    JUICER_GAMUT_HOST_DEVICE inline float sample_output_boundary(
        const float* cmax,
        const OutputBoundaryPosition& position) {
        constexpr float kPi = 3.14159265358979323846f;
        constexpr float kTwoPi = 2.0f * kPi;
        const float clampedLightness = fminf(
            fmaxf(position.lightness, kOutputBoundaryMinimumLightness),
            kOutputBoundaryMaximumLightness);
        const float lightnessPosition =
            (clampedLightness - kOutputBoundaryMinimumLightness) /
            (kOutputBoundaryMaximumLightness -
             kOutputBoundaryMinimumLightness) *
            static_cast<float>(kOutputBoundaryLightnessCount - 1);
        const int lightness0 = static_cast<int>(floorf(lightnessPosition));
        const int lightness1 =
            lightness0 + 1 < kOutputBoundaryLightnessCount
                ? lightness0 + 1
                : lightness0;
        const float lightnessFraction =
            lightnessPosition - static_cast<float>(lightness0);
        float wrappedHue = fmodf(position.hue + kPi, kTwoPi);
        if (wrappedHue < 0.0f) {
            wrappedHue += kTwoPi;
        }
        const float huePosition =
            wrappedHue / kTwoPi *
            static_cast<float>(kOutputBoundaryHueCount);
        const int hue0 =
            static_cast<int>(floorf(huePosition)) %
            kOutputBoundaryHueCount;
        const int hue1 = (hue0 + 1) % kOutputBoundaryHueCount;
        const float hueFraction = huePosition - floorf(huePosition);
        const int row0 = lightness0 * kOutputBoundaryHueCount;
        const int row1 = lightness1 * kOutputBoundaryHueCount;
        const float first =
            cmax[row0 + hue0] +
            hueFraction * (cmax[row0 + hue1] - cmax[row0 + hue0]);
        const float second =
            cmax[row1 + hue0] +
            hueFraction * (cmax[row1 + hue1] - cmax[row1 + hue0]);
        return first + lightnessFraction * (second - first);
    }

    JUICER_GAMUT_HOST_DEVICE inline void compress_output_oklch(
        const OutputCompressionView& view,
        float rgb[3]) {
        float xyz[3];
        multiply_float3x3(view.nativeRgbToD65Xyz, rgb, xyz);
        float lms[3];
        multiply_float3x3(view.oklabXyzToLms, xyz, lms);
        lms[0] = signed_cube_root(lms[0]);
        lms[1] = signed_cube_root(lms[1]);
        lms[2] = signed_cube_root(lms[2]);
        float lab[3];
        multiply_float3x3(view.oklabLmsRootToLab, lms, lab);
        lab[0] = output_reinhard_knee(lab[0], view.lightnessKnee);
        const float chroma = hypotf(lab[1], lab[2]);
        if (chroma > 0.0f) {
            const float hue = atan2f(lab[2], lab[1]);
            const float boundary = fmaxf(
                sample_output_boundary(
                    view.cmax,
                    OutputBoundaryPosition{lab[0], hue}),
                1.0e-9f);
            const float compressed = output_reinhard_knee(
                                         chroma / boundary,
                                         view.chromaKnee) *
                                     boundary;
            const float scale = compressed / chroma;
            lab[1] *= scale;
            lab[2] *= scale;
        }
        multiply_float3x3(view.oklabLabToLmsRoot, lab, lms);
        lms[0] = lms[0] * lms[0] * lms[0];
        lms[1] = lms[1] * lms[1] * lms[1];
        lms[2] = lms[2] * lms[2] * lms[2];
        multiply_float3x3(view.oklabLmsToXyz, lms, xyz);
        multiply_float3x3(view.d65XyzToNativeRgb, xyz, rgb);
    }

#undef JUICER_GAMUT_HOST_DEVICE

    bool build_output_gamut_transform(
        OutputEncoding::ColorSpace outputColorSpace,
        OutputGamutTransform& out,
        std::string& diagnostic);

    bool build_output_boundary_table(
        const OutputGamutTransform& transform,
        OutputBoundaryTable& out);

    std::uint64_t output_boundary_contract_hash(
        const OutputGamutTransform& transform);

    std::array<float, 3> native_rgb_to_oklab(
        const OutputGamutTransform& transform,
        const std::array<float, 3>& rgb);

    std::array<float, 3> oklab_to_native_rgb(
        const OutputGamutTransform& transform,
        const std::array<float, 3>& lab);

    struct InputCompressionHull {
        std::array<std::array<float, 2>, kInputHullDirectionCount + 1> xy{};
        std::array<float, 2> center{};
        std::uint64_t hash = 0;
        bool valid = false;
        std::string diagnostic;
    };

    bool build_input_compression_hull(
        const Spectral::Curve& xBar,
        const Spectral::Curve& yBar,
        const Spectral::Curve& zBar,
        const Spectral::Curve& d65,
        InputCompressionHull& out);

    bool compress_input_xy(
        const InputCompressionHull& hull,
        const std::array<float, 2>& xy,
        std::array<float, 2>& out);

    bool remap_film_tc_lut_for_input_compression(
        const InputCompressionHull& hull,
        const Spectral::FilmTcLut& source,
        Spectral::FilmTcLut& out,
        std::string& diagnostic);

} // namespace Gamut
