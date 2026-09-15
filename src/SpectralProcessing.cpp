#include "SpectralProcessing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "GamutCompression.h"
#include "GaussianSciPy.h"
#include "RenderRecipe.h"

namespace Spectral {
    namespace {

        constexpr std::size_t kFilmTcValueCount =
            static_cast<std::size_t>(FilmTcLut::kSize) *
            static_cast<std::size_t>(FilmTcLut::kSize) *
            FilmTcLut::kChannels;

        struct SurfaceCoordinates {
            float tcC = 0.0f;
            float tcM = 0.0f;
            float centerC = 0.0f;
            float centerM = 0.0f;
        };

        int gaussian_reflect_index(int index, int size) {
            if (size <= 1) {
                return 0;
            }
            while (index < 0 || index >= size) {
                index = index < 0 ? -index - 1 : 2 * size - index - 1;
            }
            return index;
        }

        int mitchell_reflect_index(int index, int size) {
            if (size <= 1) {
                return 0;
            }
            if (index < 0) {
                return -index;
            }
            if (index >= size) {
                return 2 * (size - 1) - index;
            }
            return index;
        }

        bool spectra_shape_valid(const NpySpectraLUT& spectra) {
            return spectra.size == FilmTcLut::kSize &&
                   spectra.numSamples == kNumSamples &&
                   spectra.data.size() ==
                       static_cast<std::size_t>(FilmTcLut::kSize) *
                           static_cast<std::size_t>(FilmTcLut::kSize) *
                           static_cast<std::size_t>(kNumSamples) &&
                   std::all_of(
                       spectra.data.begin(),
                       spectra.data.end(),
                       [](float value) {
                           return std::isfinite(value);
                       });
        }

        bool build_gaussian_kernel(
            float sigma,
            std::vector<float>& kernel) {
            const int radius = JuicerGaussian::scipy_gaussian_radius(sigma);
            if (radius <= 0) {
                return false;
            }
            kernel.resize(static_cast<std::size_t>(radius) * 2u + 1u);
            float kernelSum = 0.0f;
            const float sigmaSquared = sigma * sigma;
            for (std::size_t kernelIndex = 0; kernelIndex < kernel.size(); ++kernelIndex) {
                const int offset = static_cast<int>(kernelIndex) - radius;
                const float weight =
                    std::exp(-0.5f * static_cast<float>(offset * offset) / sigmaSquared);
                kernel[kernelIndex] = weight;
                kernelSum += weight;
            }
            if (!(std::isfinite(kernelSum) && kernelSum > 0.0f)) {
                return false;
            }
            for (float& weight : kernel) {
                weight /= kernelSum;
            }
            return true;
        }

        bool build_blurred_hanatos_spectra(
            const NpySpectraLUT& source,
            float sigma,
            std::vector<float>& out,
            std::string& diagnostic) {
            if (!spectra_shape_valid(source) || !std::isfinite(sigma) || sigma < 0.0f) {
                diagnostic =
                    "MalformedRequiredResource component=film_tc_lut method=hanatos2025 requirement=finite_192x192x81_spectra_and_blur";
                return false;
            }
            if (!(sigma > 0.0f)) {
                out = source.data;
                return true;
            }
            std::vector<float> kernel;
            if (!build_gaussian_kernel(sigma, kernel)) {
                diagnostic =
                    "MalformedRequiredResource component=film_tc_lut method=hanatos2025 requirement=finite_blur_kernel";
                return false;
            }
            const int radius = static_cast<int>(kernel.size() / 2u);

            out.assign(source.data.size(), 0.0f);
            constexpr int size = FilmTcLut::kSize;
            for (int c = 0; c < size; ++c) {
                for (int m = 0; m < size; ++m) {
                    const std::size_t base =
                        (static_cast<std::size_t>(c) * static_cast<std::size_t>(size) +
                         static_cast<std::size_t>(m)) *
                        static_cast<std::size_t>(kNumSamples);
                    for (int sample = 0; sample < kNumSamples; ++sample) {
                        float value = 0.0f;
                        for (std::size_t kernelIndex = 0; kernelIndex < kernel.size(); ++kernelIndex) {
                            const int offset = static_cast<int>(kernelIndex) - radius;
                            const int reflected = gaussian_reflect_index(
                                sample + offset,
                                kNumSamples);
                            value += kernel[kernelIndex] *
                                     source.data[base + static_cast<std::size_t>(reflected)];
                        }
                        if (!std::isfinite(value)) {
                            diagnostic =
                                "MalformedRequiredResource component=film_tc_lut method=hanatos2025 requirement=finite_blurred_spectra";
                            return false;
                        }
                        out[base + static_cast<std::size_t>(sample)] = value;
                    }
                }
            }
            return true;
        }

        float mitchell_weight(float t) {
            constexpr float b = 1.0f / 3.0f;
            constexpr float c = 1.0f / 3.0f;
            const float x = std::abs(t);
            if (x < 1.0f) {
                return (1.0f / 6.0f) *
                       ((12.0f - 9.0f * b - 6.0f * c) * x * x * x +
                        (-18.0f + 12.0f * b + 6.0f * c) * x * x +
                        (6.0f - 2.0f * b));
            }
            if (x < 2.0f) {
                return (1.0f / 6.0f) *
                       ((-b - 6.0f * c) * x * x * x +
                        (6.0f * b + 30.0f * c) * x * x +
                        (-12.0f * b - 48.0f * c) * x +
                        (8.0f * b + 24.0f * c));
            }
            return 0.0f;
        }

        void mitchell_coordinate(
            float normalized,
            int size,
            int& base,
            float& fraction) {
            const float value =
                std::clamp(normalized, 0.0f, 1.0f) * static_cast<float>(size - 1);
            if (value >= static_cast<float>(size - 1)) {
                base = size - 2;
                fraction = 1.0f;
                return;
            }
            base = static_cast<int>(std::floor(value));
            fraction = value - static_cast<float>(base);
        }

        template <typename Sample>
        float sample_mitchell_2d(
            int size,
            float tcC,
            float tcM,
            Sample&& sample) {
            int cBase = 0;
            int mBase = 0;
            float cFraction = 0.0f;
            float mFraction = 0.0f;
            mitchell_coordinate(tcC, size, cBase, cFraction);
            mitchell_coordinate(tcM, size, mBase, mFraction);
            const std::array<float, 4> cWeights{
                mitchell_weight(cFraction + 1.0f),
                mitchell_weight(cFraction),
                mitchell_weight(cFraction - 1.0f),
                mitchell_weight(cFraction - 2.0f)};
            const std::array<float, 4> mWeights{
                mitchell_weight(mFraction + 1.0f),
                mitchell_weight(mFraction),
                mitchell_weight(mFraction - 1.0f),
                mitchell_weight(mFraction - 2.0f)};
            float value = 0.0f;
            float weightSum = 0.0f;
            for (int dc = 0; dc < 4; ++dc) {
                const int c = mitchell_reflect_index(cBase - 1 + dc, size);
                for (int dm = 0; dm < 4; ++dm) {
                    const int m = mitchell_reflect_index(mBase - 1 + dm, size);
                    const float weight =
                        cWeights[static_cast<std::size_t>(dc)] *
                        mWeights[static_cast<std::size_t>(dm)];
                    weightSum += weight;
                    value += weight * sample(c, m);
                }
            }
            return weightSum != 0.0f ? value / weightSum : 0.0f;
        }

        float sample_spectrum(
            const std::vector<float>& spectra,
            const std::array<float, 2>& tc,
            int spectralSample) {
            return sample_mitchell_2d(
                FilmTcLut::kSize,
                tc[0],
                tc[1],
                [&](int c, int m) {
                    const std::size_t offset =
                        (static_cast<std::size_t>(c) *
                             static_cast<std::size_t>(FilmTcLut::kSize) +
                         static_cast<std::size_t>(m)) *
                            static_cast<std::size_t>(kNumSamples) +
                        static_cast<std::size_t>(spectralSample);
                    return spectra[offset];
                });
        }

        float sample_integrated(
            const FilmTcLut& lut,
            const std::array<float, 2>& tc,
            int channel) {
            return sample_mitchell_2d(
                FilmTcLut::kSize,
                tc[0],
                tc[1],
                [&](int c, int m) {
                    const std::size_t offset =
                        (static_cast<std::size_t>(c) *
                             static_cast<std::size_t>(FilmTcLut::kSize) +
                         static_cast<std::size_t>(m)) *
                            FilmTcLut::kChannels +
                        static_cast<std::size_t>(channel);
                    return lut.rgba[offset];
                });
        }

        float eval_hanatos_surface(
            const std::array<float, 15>& params,
            const SurfaceCoordinates& coordinates) {
            const float x = coordinates.tcC - coordinates.centerC;
            const float y = coordinates.tcM - coordinates.centerM;
            const float x2 = x * x;
            const float y2 = y * y;
            const float x3 = x2 * x;
            const float y3 = y2 * y;
            const float raw =
                params[1] * x + params[2] * y + params[3] * x2 +
                params[4] * y2 + params[5] * x * y + params[6] * x3 +
                params[7] * y3 + params[8] * x2 * y + params[9] * x * y2 +
                params[10] * x2 * x2 + params[11] * y2 * y2 +
                params[12] * x3 * y + params[13] * x2 * y2 +
                params[14] * x * y3;
            constexpr float maximumStops = 2.0f;
            const float normalized = raw / maximumStops;
            return raw / std::sqrt(1.0f + normalized * normalized);
        }

        bool projected_white_to_tc(
            const std::array<float, 3>& whiteXYZ,
            float& tcC,
            float& tcM) {
            const float sum = whiteXYZ[0] + whiteXYZ[1] + whiteXYZ[2];
            if (!(std::isfinite(sum) && sum > 0.0f)) {
                return false;
            }
            tri2quad(whiteXYZ[0] / sum, whiteXYZ[1] / sum, tcC, tcM);
            return std::isfinite(tcC) && std::isfinite(tcM);
        }

    } // namespace

    bool build_hanatos_reconstructed_reference_white(
        const NpySpectraLUT& spectra,
        float spectralGaussianBlur,
        const std::array<float, 3>& referenceWhiteXYZ,
        std::array<float, kNumSamples>& out,
        std::string& diagnostic) {
        diagnostic.clear();
        out = {};
        if (!spectra_shape_valid(spectra) ||
            !std::isfinite(spectralGaussianBlur) ||
            spectralGaussianBlur < 0.0f) {
            diagnostic =
                "MalformedRequiredResource component=hanatos_window requirement=finite_192x192x81_spectra_and_blur";
            return false;
        }
        float tcC = 0.0f;
        float tcM = 0.0f;
        if (!projected_white_to_tc(referenceWhiteXYZ, tcC, tcM)) {
            diagnostic =
                "MalformedRequiredResource component=hanatos_window requirement=finite_reference_white";
            return false;
        }

        std::array<float, kNumSamples> sampled{};
        for (int sample = 0; sample < kNumSamples; ++sample) {
            const float value =
                sample_spectrum(spectra.data, {tcC, tcM}, sample);
            if (!std::isfinite(value)) {
                diagnostic =
                    "MalformedRequiredResource component=hanatos_window requirement=finite_reconstructed_reference_white";
                return false;
            }
            sampled[static_cast<std::size_t>(sample)] = value;
        }
        if (!(spectralGaussianBlur > 0.0f)) {
            out = sampled;
            return true;
        }

        std::vector<float> kernel;
        if (!build_gaussian_kernel(spectralGaussianBlur, kernel)) {
            diagnostic =
                "MalformedRequiredResource component=hanatos_window requirement=finite_blur_kernel";
            return false;
        }
        const int radius = static_cast<int>(kernel.size() / 2u);
        for (int sample = 0; sample < kNumSamples; ++sample) {
            float value = 0.0f;
            for (std::size_t kernelIndex = 0; kernelIndex < kernel.size(); ++kernelIndex) {
                const int offset = static_cast<int>(kernelIndex) - radius;
                const int reflected = gaussian_reflect_index(
                    sample + offset,
                    kNumSamples);
                value += kernel[kernelIndex] *
                         sampled[static_cast<std::size_t>(reflected)];
            }
            if (!std::isfinite(value)) {
                diagnostic =
                    "MalformedRequiredResource component=hanatos_window requirement=finite_reconstructed_reference_white";
                return false;
            }
            out[static_cast<std::size_t>(sample)] = value;
        }
        return true;
    }

    bool build_film_tc_lut(
        const ::FilmRawRecipe& recipe,
        const NpySpectraLUT& spectra,
        const std::array<float, kNumSamples>& referenceIlluminant,
        FilmTcLut& out,
        std::string& diagnostic) {
        diagnostic.clear();
        out = FilmTcLut{};
        if (!spectra_shape_valid(spectra) || recipe.tcLutHash == 0 ||
            (recipe.rgbToRawMethod != Spektrafilm::RgbToRawMethod::Hanatos2025 &&
             recipe.rgbToRawMethod != Spektrafilm::RgbToRawMethod::Arctic2026beta04)) {
            diagnostic =
                "MalformedRequiredResource component=film_tc_lut requirement=selected_finite_192x192x81_spectra";
            return false;
        }
        if (!std::all_of(
                referenceIlluminant.begin(),
                referenceIlluminant.end(),
                [](float value) {
                    return std::isfinite(value) && value >= 0.0f;
                })) {
            diagnostic =
                "MalformedRequiredResource component=film_tc_lut requirement=finite_nonnegative_reference_illuminant";
            return false;
        }

        std::vector<float> hanatosSpectra;
        const std::vector<float>* selectedSpectra = &spectra.data;
        if (recipe.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Hanatos2025 &&
            recipe.hanatos.spectralGaussianBlur > 0.0f) {
            if (!build_blurred_hanatos_spectra(
                    spectra,
                    recipe.hanatos.spectralGaussianBlur,
                    hanatosSpectra,
                    diagnostic)) {
                return false;
            }
            selectedSpectra = &hanatosSpectra;
        }

        std::array<float, 3> neutralDenominator{{1.0f, 1.0f, 1.0f}};
        if (recipe.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Arctic2026beta04) {
            float tcC = 0.0f;
            float tcM = 0.0f;
            if (!projected_white_to_tc(recipe.projectionWhiteXYZ, tcC, tcM)) {
                diagnostic =
                    "MalformedRequiredResource component=film_tc_lut method=arctic2026beta04 requirement=finite_d65_projection_white";
                return false;
            }
            neutralDenominator.fill(0.0f);
            for (int sample = 0; sample < kNumSamples; ++sample) {
                const float reflectance = sample_spectrum(
                    *selectedSpectra,
                    {tcC, tcM},
                    sample);
                const float relit =
                    reflectance * referenceIlluminant[static_cast<std::size_t>(sample)];
                for (int channel = 0; channel < 3; ++channel) {
                    neutralDenominator[static_cast<std::size_t>(channel)] +=
                        relit * recipe.finalSensitivity[static_cast<std::size_t>(sample)]
                                                       [static_cast<std::size_t>(channel)];
                }
            }
            for (float denominator : neutralDenominator) {
                if (!(std::isfinite(denominator) && denominator > 0.0f)) {
                    diagnostic =
                        "MalformedRequiredResource component=film_tc_lut method=arctic2026beta04 requirement=finite_positive_neutral_response";
                    return false;
                }
            }
        }

        FilmTcLut integrated;
        integrated.rgba.assign(kFilmTcValueCount, 0.0f);
        constexpr int size = FilmTcLut::kSize;
        float surfaceCenterC = 0.0f;
        float surfaceCenterM = 0.0f;
        if (recipe.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Hanatos2025 &&
            recipe.hanatos.applySurface &&
            !projected_white_to_tc(
                recipe.projectionWhiteXYZ,
                surfaceCenterC,
                surfaceCenterM)) {
            diagnostic =
                "MalformedRequiredResource component=film_tc_lut method=hanatos2025 requirement=finite_surface_reference_white";
            return false;
        }
        for (int c = 0; c < size; ++c) {
            const float tcC = static_cast<float>(c) / static_cast<float>(size - 1);
            for (int m = 0; m < size; ++m) {
                const float tcM = static_cast<float>(m) / static_cast<float>(size - 1);
                std::array<float, 3> raw{};
                const std::size_t spectralBase =
                    (static_cast<std::size_t>(c) * static_cast<std::size_t>(size) +
                     static_cast<std::size_t>(m)) *
                    static_cast<std::size_t>(kNumSamples);
                for (int sample = 0; sample < kNumSamples; ++sample) {
                    float energy = (*selectedSpectra)[spectralBase + static_cast<std::size_t>(sample)];
                    if (recipe.rgbToRawMethod ==
                        Spektrafilm::RgbToRawMethod::Arctic2026beta04) {
                        energy *= referenceIlluminant[static_cast<std::size_t>(sample)];
                    }
                    for (int channel = 0; channel < 3; ++channel) {
                        raw[static_cast<std::size_t>(channel)] +=
                            energy * recipe.finalSensitivity[static_cast<std::size_t>(sample)]
                                                            [static_cast<std::size_t>(channel)];
                    }
                }
                for (int channel = 0; channel < 3; ++channel) {
                    float value = raw[static_cast<std::size_t>(channel)] /
                                  neutralDenominator[static_cast<std::size_t>(channel)];
                    if (recipe.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Hanatos2025 &&
                        recipe.hanatos.applySurface) {
                        value *= std::exp2(eval_hanatos_surface(
                            recipe.hanatos.surfaceParams[static_cast<std::size_t>(channel)],
                            {tcC, tcM, surfaceCenterC, surfaceCenterM}));
                    }
                    if (!std::isfinite(value)) {
                        diagnostic =
                            "MalformedRequiredResource component=film_tc_lut requirement=finite_integrated_samples";
                        return false;
                    }
                    const std::size_t outputBase =
                        (static_cast<std::size_t>(c) * static_cast<std::size_t>(size) +
                         static_cast<std::size_t>(m)) *
                        FilmTcLut::kChannels;
                    integrated.rgba[outputBase + static_cast<std::size_t>(channel)] = value;
                }
            }
        }

        if (recipe.inputCompressionActive) {
            if (!recipe.inputCompressionHull ||
                !Gamut::remap_film_tc_lut_for_input_compression(
                    *recipe.inputCompressionHull,
                    integrated,
                    out,
                    diagnostic)) {
                if (diagnostic.empty()) {
                    diagnostic =
                        "MissingRequiredResource component=film_tc_lut requirement=input_compression_hull";
                }
                return false;
            }
        } else {
            out = std::move(integrated);
        }
        return out.rgba.size() == kFilmTcValueCount;
    }

    std::array<float, 3> sample_film_tc_lut(
        const FilmTcLut& lut,
        const std::array<float, 3>& projectedXYZ) {
        std::array<float, 3> out{};
        if (lut.rgba.size() != kFilmTcValueCount) {
            return out;
        }
        const float xValue =
            std::isfinite(projectedXYZ[0]) ? projectedXYZ[0] : 0.0f;
        const float yValue =
            std::isfinite(projectedXYZ[1]) ? projectedXYZ[1] : 0.0f;
        const float zValue =
            std::isfinite(projectedXYZ[2]) ? projectedXYZ[2] : 0.0f;
        const float brightness = xValue + yValue + zValue;
        const float denominator = std::max(brightness, 1e-10f);
        float tcC = 0.0f;
        float tcM = 0.0f;
        tri2quad(xValue / denominator, yValue / denominator, tcC, tcM);
        for (int channel = 0; channel < 3; ++channel) {
            const float sample = sample_integrated(lut, {tcC, tcM}, channel);
            const float value = brightness * sample;
            out[static_cast<std::size_t>(channel)] =
                std::isfinite(value) ? value : 0.0f;
        }
        return out;
    }

} // namespace Spectral
