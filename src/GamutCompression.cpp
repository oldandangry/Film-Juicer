#include "GamutCompression.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>

#include "ColorTransforms.h"
#include "Hash.h"
#include "SpectralData.h"
#include "SpectralProcessing.h"

namespace Gamut {
    namespace {

        constexpr int kLocusSampleCount = 65;
        constexpr double kTwoPi = 6.283185307179586476925286766559;
        constexpr double kParallelTolerance = 1e-12;
        constexpr double kVertexTolerance = 1e-9;
        constexpr double kMinimumRadius = 1e-12;
        constexpr double kHullDetail = 5.0;
        constexpr double kKneeThreshold = 0.815;
        constexpr double kKneeLimit = 1.0;
        constexpr double kKneePower = 1.2;
        constexpr std::uint32_t kInputCompressionVersion = 1u;
        constexpr std::uint32_t kOutputTransformVersion = 1u;
        constexpr std::uint32_t kOutputBoundaryVersion = 1u;
        constexpr int kOutputBoundaryBisections = 20;
        constexpr double kDerivativeTolerance = 1.0e-14;

        using Point = std::array<double, 2>;
        using Locus = std::array<Point, kLocusSampleCount + 1>;
        using Matrix = std::array<double, 9>;

        struct Cubic {
            double coefficient[4]{};
        };

        struct Ray {
            Point origin;
            Point direction;
        };

        Matrix multiply_matrix(const Matrix& left, const Matrix& right) {
            Matrix result{};
            for (int row = 0; row < 3; ++row) {
                const std::size_t rowOffset =
                    static_cast<std::size_t>(row) * 3u;
                for (int column = 0; column < 3; ++column) {
                    const std::size_t columnIndex =
                        static_cast<std::size_t>(column);
                    for (int inner = 0; inner < 3; ++inner) {
                        const std::size_t innerIndex =
                            static_cast<std::size_t>(inner);
                        result[rowOffset + columnIndex] +=
                            left[rowOffset + innerIndex] *
                            right[innerIndex * 3u + columnIndex];
                    }
                }
            }
            return result;
        }

        bool finite_matrix(const Matrix& matrix) {
            return std::all_of(
                matrix.begin(),
                matrix.end(),
                [](double value) {
                    return std::isfinite(value);
                });
        }

        double evaluate_cubic(const Cubic& cubic, double chroma) {
            return ((cubic.coefficient[3] * chroma + cubic.coefficient[2]) *
                        chroma +
                    cubic.coefficient[1]) *
                       chroma +
                   cubic.coefficient[0];
        }

        bool channel_inside_cube(double value) {
            constexpr double slack =
                static_cast<double>(kOutputBoundaryCubeSlack);
            return std::isfinite(value) && value >= -slack &&
                   value <= 1.0 + slack;
        }

        void append_interior_derivative_roots(
            const Cubic& cubic,
            std::array<double, 4>& partitions,
            int& partitionCount) {
            const double a = 3.0 * cubic.coefficient[3];
            const double b = 2.0 * cubic.coefficient[2];
            const double c = cubic.coefficient[1];
            auto append = [&](double root) {
                if (std::isfinite(root) && root > 0.0 &&
                    root < static_cast<double>(kOutputBoundaryMaximumChroma)) {
                    partitions[static_cast<std::size_t>(partitionCount++)] = root;
                }
            };
            const double scale = std::max({std::abs(a), std::abs(b), std::abs(c), 1.0});
            if (std::abs(a) <= kDerivativeTolerance * scale) {
                if (std::abs(b) > kDerivativeTolerance * scale) {
                    append(-c / b);
                }
                return;
            }
            const double discriminant = b * b - 4.0 * a * c;
            if (!(discriminant >= 0.0)) {
                return;
            }
            const double squareRoot = std::sqrt(discriminant);
            const double q = -0.5 * (b + std::copysign(squareRoot, b));
            if (std::abs(q) <= kDerivativeTolerance * scale) {
                append(-b / (2.0 * a));
                return;
            }
            append(q / a);
            append(c / q);
        }

        double first_channel_exit(const Cubic& cubic) {
            if (!channel_inside_cube(evaluate_cubic(cubic, 0.0))) {
                return 0.0;
            }
            std::array<double, 4> partitions{
                0.0,
                static_cast<double>(kOutputBoundaryMaximumChroma),
                0.0,
                0.0};
            int partitionCount = 2;
            append_interior_derivative_roots(
                cubic,
                partitions,
                partitionCount);
            for (int index = 1; index < partitionCount; ++index) {
                const double value = partitions[static_cast<std::size_t>(index)];
                int insertion = index;
                while (insertion > 0 &&
                       value < partitions[static_cast<std::size_t>(insertion - 1)]) {
                    partitions[static_cast<std::size_t>(insertion)] =
                        partitions[static_cast<std::size_t>(insertion - 1)];
                    --insertion;
                }
                partitions[static_cast<std::size_t>(insertion)] = value;
            }
            const auto uniqueEnd = std::unique(
                partitions.begin(),
                partitions.begin() + partitionCount,
                [](double left, double right) {
                    return std::abs(left - right) <= 1.0e-12;
                });
            partitionCount = static_cast<int>(uniqueEnd - partitions.begin());
            for (int partition = 1; partition < partitionCount; ++partition) {
                double lower = partitions[static_cast<std::size_t>(partition - 1)];
                double upper = partitions[static_cast<std::size_t>(partition)];
                const double upperValue = evaluate_cubic(cubic, upper);
                if (channel_inside_cube(upperValue)) {
                    continue;
                }
                for (int iteration = 0;
                     iteration < kOutputBoundaryBisections;
                     ++iteration) {
                    const double middle = 0.5 * (lower + upper);
                    if (channel_inside_cube(evaluate_cubic(cubic, middle))) {
                        lower = middle;
                    } else {
                        upper = middle;
                    }
                }
                return lower;
            }
            return std::numeric_limits<double>::infinity();
        }

        std::array<Cubic, 3> build_boundary_cubics(
            const Matrix& lmsToNativeRgb,
            double lightness,
            const std::array<double, 2>& hueDirection) {
            std::array<double, 3> u{};
            std::array<double, 3> v{};
            for (int row = 0; row < 3; ++row) {
                const std::size_t rowOffset =
                    static_cast<std::size_t>(row) * 3u;
                u[static_cast<std::size_t>(row)] =
                    static_cast<double>(kOklabLabToLmsRoot[rowOffset]) *
                    lightness;
                v[static_cast<std::size_t>(row)] =
                    static_cast<double>(kOklabLabToLmsRoot[rowOffset + 1u]) *
                        hueDirection[0] +
                    static_cast<double>(kOklabLabToLmsRoot[rowOffset + 2u]) *
                        hueDirection[1];
            }
            std::array<Cubic, 3> cubics{};
            for (int channel = 0; channel < 3; ++channel) {
                Cubic& cubic = cubics[static_cast<std::size_t>(channel)];
                const std::size_t channelOffset =
                    static_cast<std::size_t>(channel) * 3u;
                for (int component = 0; component < 3; ++component) {
                    const double weight = lmsToNativeRgb[channelOffset + static_cast<std::size_t>(component)];
                    const double componentU = u[static_cast<std::size_t>(component)];
                    const double componentV = v[static_cast<std::size_t>(component)];
                    cubic.coefficient[0] += weight * componentU * componentU * componentU;
                    cubic.coefficient[1] +=
                        weight * 3.0 * componentU * componentU * componentV;
                    cubic.coefficient[2] +=
                        weight * 3.0 * componentU * componentV * componentV;
                    cubic.coefficient[3] +=
                        weight * componentV * componentV * componentV;
                }
            }
            return cubics;
        }

        std::array<float, 3> multiply_float_matrix_vector(
            const std::array<float, 9>& matrix,
            const std::array<float, 3>& value) {
            return {
                matrix[0] * value[0] + matrix[1] * value[1] + matrix[2] * value[2],
                matrix[3] * value[0] + matrix[4] * value[1] + matrix[5] * value[2],
                matrix[6] * value[0] + matrix[7] * value[1] + matrix[8] * value[2]};
        }

        double ray_polygon_distance(
            const Ray& ray,
            const auto& polygon) {
            double minimum = std::numeric_limits<double>::infinity();
            for (std::size_t index = 0; index + 1 < polygon.size(); ++index) {
                const double ax = static_cast<double>(polygon[index][0]);
                const double ay = static_cast<double>(polygon[index][1]);
                const double ex =
                    static_cast<double>(polygon[index + 1][0]) - ax;
                const double ey =
                    static_cast<double>(polygon[index + 1][1]) - ay;
                const double denominator =
                    ray.direction[0] * ey - ray.direction[1] * ex;
                if (std::abs(denominator) <= kParallelTolerance) {
                    continue;
                }
                const double ox = ray.origin[0] - ax;
                const double oy = ray.origin[1] - ay;
                const double distance = (-ox * ey + oy * ex) / denominator;
                const double edgePosition =
                    (-ox * ray.direction[1] + oy * ray.direction[0]) /
                    denominator;
                if (distance > kVertexTolerance &&
                    edgePosition >= -kVertexTolerance &&
                    edgePosition <= 1.0 + kVertexTolerance) {
                    minimum = std::min(minimum, distance);
                }
            }
            return minimum;
        }

        bool curve_has_reference_samples(const Spectral::Curve& curve) {
            return curve.linear.size() ==
                       static_cast<std::size_t>(Spectral::kNumSamples) &&
                   curve.lambda_nm.size() ==
                       static_cast<std::size_t>(Spectral::kNumSamples);
        }

        bool build_locus(
            const Spectral::Curve& xBar,
            const Spectral::Curve& yBar,
            const Spectral::Curve& zBar,
            Locus& out) {
            if (!curve_has_reference_samples(xBar) ||
                !curve_has_reference_samples(yBar) ||
                !curve_has_reference_samples(zBar)) {
                return false;
            }
            for (int index = 0; index < kLocusSampleCount; ++index) {
                const double x = static_cast<double>(xBar.linear[index]);
                const double y = static_cast<double>(yBar.linear[index]);
                const double z = static_cast<double>(zBar.linear[index]);
                const double sum = x + y + z;
                if (!(std::isfinite(x) && std::isfinite(y) &&
                      std::isfinite(z) && std::isfinite(sum) &&
                      sum > kMinimumRadius)) {
                    return false;
                }
                out[static_cast<std::size_t>(index)] = {x / sum, y / sum};
            }
            out.back() = out.front();
            return true;
        }

        bool build_d65_center(
            const Spectral::Curve& xBar,
            const Spectral::Curve& yBar,
            const Spectral::Curve& zBar,
            const Spectral::Curve& d65,
            Point& out) {
            if (!curve_has_reference_samples(d65)) {
                return false;
            }
            double xyz[3]{};
            for (int index = 0; index < Spectral::kNumSamples; ++index) {
                const double illuminant =
                    static_cast<double>(d65.linear[static_cast<std::size_t>(index)]);
                if (!std::isfinite(illuminant)) {
                    return false;
                }
                xyz[0] += illuminant *
                          static_cast<double>(xBar.linear[static_cast<std::size_t>(index)]);
                xyz[1] += illuminant *
                          static_cast<double>(yBar.linear[static_cast<std::size_t>(index)]);
                xyz[2] += illuminant *
                          static_cast<double>(zBar.linear[static_cast<std::size_t>(index)]);
            }
            const double sum = xyz[0] + xyz[1] + xyz[2];
            if (!(std::isfinite(sum) && sum > kMinimumRadius)) {
                return false;
            }
            out = {xyz[0] / sum, xyz[1] / sum};
            return std::isfinite(out[0]) && std::isfinite(out[1]);
        }

        void fourier_lowpass(
            const std::array<double, kInputHullDirectionCount>& source,
            std::array<double, kInputHullDirectionCount>& out) {
            using Complex = std::complex<double>;
            std::array<Complex, kInputHullDirectionCount> coefficients{};
            for (int mode = 0; mode < kInputHullDirectionCount; ++mode) {
                Complex coefficient{};
                for (int sample = 0; sample < kInputHullDirectionCount; ++sample) {
                    const double angle =
                        -kTwoPi * static_cast<double>(mode * sample) /
                        static_cast<double>(kInputHullDirectionCount);
                    coefficient += source[static_cast<std::size_t>(sample)] *
                                   Complex(std::cos(angle), std::sin(angle));
                }
                const int signedMode =
                    mode <= kInputHullDirectionCount / 2
                        ? mode
                        : mode - kInputHullDirectionCount;
                const double normalizedMode =
                    static_cast<double>(signedMode) / kHullDetail;
                coefficients[static_cast<std::size_t>(mode)] =
                    coefficient * std::exp(-0.5 * normalizedMode * normalizedMode);
            }
            for (int sample = 0; sample < kInputHullDirectionCount; ++sample) {
                Complex value{};
                for (int mode = 0; mode < kInputHullDirectionCount; ++mode) {
                    const double angle =
                        kTwoPi * static_cast<double>(mode * sample) /
                        static_cast<double>(kInputHullDirectionCount);
                    value += coefficients[static_cast<std::size_t>(mode)] *
                             Complex(std::cos(angle), std::sin(angle));
                }
                out[static_cast<std::size_t>(sample)] =
                    value.real() / static_cast<double>(kInputHullDirectionCount);
            }
        }

        double reinhard_knee(double distance) {
            if (!(distance > kKneeThreshold)) {
                return distance;
            }
            const double scale = kKneeLimit - kKneeThreshold;
            const double normalized = (distance - kKneeThreshold) / scale;
            const double compressed =
                normalized /
                std::pow(1.0 + std::pow(normalized, kKneePower),
                         1.0 / kKneePower);
            return kKneeThreshold + scale * compressed;
        }

        float bilinear_sample(
            const Spectral::FilmTcLut& source,
            const std::array<float, 2>& tc,
            int channel) {
            constexpr int size = Spectral::FilmTcLut::kSize;
            const float cPosition =
                std::clamp(tc[0], 0.0f, 1.0f) * static_cast<float>(size - 1);
            const float mPosition =
                std::clamp(tc[1], 0.0f, 1.0f) * static_cast<float>(size - 1);
            const int c0 = static_cast<int>(std::floor(cPosition));
            const int m0 = static_cast<int>(std::floor(mPosition));
            const int c1 = std::min(c0 + 1, size - 1);
            const int m1 = std::min(m0 + 1, size - 1);
            const float cFraction = cPosition - static_cast<float>(c0);
            const float mFraction = mPosition - static_cast<float>(m0);
            const auto value = [&](int c, int m) {
                const std::size_t offset =
                    (static_cast<std::size_t>(c) * static_cast<std::size_t>(size) +
                     static_cast<std::size_t>(m)) *
                        Spectral::FilmTcLut::kChannels +
                    static_cast<std::size_t>(channel);
                return source.rgba[offset];
            };
            const float row0 =
                value(c0, m0) + cFraction * (value(c1, m0) - value(c0, m0));
            const float row1 =
                value(c0, m1) + cFraction * (value(c1, m1) - value(c0, m1));
            return row0 + mFraction * (row1 - row0);
        }

    } // namespace

    bool build_output_gamut_transform(
        OutputEncoding::ColorSpace outputColorSpace,
        OutputGamutTransform& out,
        std::string& diagnostic) {
        out = OutputGamutTransform{};
        diagnostic.clear();
        const int outputIndex = OutputEncoding::toIndex(outputColorSpace);
        if (outputIndex < 0 ||
            outputIndex >= static_cast<int>(OutputEncoding::ColorSpace::Count)) {
            diagnostic =
                "ResourceDescriptorMismatch component=output_gamut_transform field=output_color_space";
            return false;
        }
        const GeneratedColorSpaces::ColorSpaceEntry& output =
            GeneratedColorSpaces::get(outputColorSpace);
        std::array<double, 3> nativeWhite{};
        for (int channel = 0; channel < 3; ++channel) {
            nativeWhite[static_cast<std::size_t>(channel)] =
                static_cast<double>(output.whiteXYZ[channel]) /
                static_cast<double>(output.whiteXYZ[1]);
        }
        const GeneratedColorSpaces::ColorSpaceEntry& d65Space =
            GeneratedColorSpaces::get(OutputEncoding::ColorSpace::sRGB);
        float normalizedNativeWhite[3]{};
        for (int channel = 0; channel < 3; ++channel) {
            normalizedNativeWhite[channel] =
                static_cast<float>(nativeWhite[static_cast<std::size_t>(channel)]);
        }
        const Spectral::ChromaticAdaptationWhites entryWhites{
            normalizedNativeWhite,
            d65Space.whiteXYZ};
        const Spectral::ChromaticAdaptationWhites exitWhites{
            d65Space.whiteXYZ,
            normalizedNativeWhite};
        const Spectral::Mat3 entryAdapt =
            Spectral::build_chromatic_adaptation_matrix_CAT16(entryWhites);
        const Spectral::Mat3 exitAdapt =
            Spectral::build_chromatic_adaptation_matrix_CAT16(exitWhites);
        Matrix entryAdaptMatrix{};
        Matrix exitAdaptMatrix{};
        Matrix rgbToXyz{};
        Matrix xyzToRgb{};
        for (std::size_t index = 0; index < rgbToXyz.size(); ++index) {
            rgbToXyz[index] = static_cast<double>(output.rgbToXyz[index]);
            xyzToRgb[index] = static_cast<double>(output.xyzToRgb[index]);
            entryAdaptMatrix[index] = static_cast<double>(entryAdapt.m[index]);
            exitAdaptMatrix[index] = static_cast<double>(exitAdapt.m[index]);
        }
        const Matrix entry = multiply_matrix(
            entryAdaptMatrix,
            rgbToXyz);
        const Matrix exit = multiply_matrix(
            xyzToRgb,
            exitAdaptMatrix);
        if (!finite_matrix(entry) || !finite_matrix(exit)) {
            diagnostic =
                "MalformedRequiredResource component=output_gamut_transform requirement=finite_native_white_cat16";
            return false;
        }
        for (std::size_t index = 0; index < entry.size(); ++index) {
            out.nativeRgbToD65Xyz[index] = static_cast<float>(entry[index]);
            out.d65XyzToNativeRgb[index] = static_cast<float>(exit[index]);
        }
        out.outputColorSpace = outputColorSpace;
        std::uint64_t hash = Hash::kFnvOffset;
        Hash::hash_bytes_update(
            hash,
            &kOutputTransformVersion,
            sizeof(kOutputTransformVersion));
        Hash::hash_bytes_update(hash, &outputColorSpace, sizeof(outputColorSpace));
        Hash::hash_bytes_update(
            hash,
            output.rgbToXyz,
            sizeof(output.rgbToXyz));
        Hash::hash_bytes_update(
            hash,
            output.xyzToRgb,
            sizeof(output.xyzToRgb));
        Hash::hash_bytes_update(
            hash,
            output.whiteXYZ,
            sizeof(output.whiteXYZ));
        Hash::hash_bytes_update(
            hash,
            out.nativeRgbToD65Xyz.data(),
            sizeof(out.nativeRgbToD65Xyz));
        Hash::hash_bytes_update(
            hash,
            out.d65XyzToNativeRgb.data(),
            sizeof(out.d65XyzToNativeRgb));
        Hash::hash_bytes_update(
            hash,
            kOklabXyzToLms.data(),
            sizeof(kOklabXyzToLms));
        Hash::hash_bytes_update(
            hash,
            kOklabLmsToXyz.data(),
            sizeof(kOklabLmsToXyz));
        Hash::hash_bytes_update(
            hash,
            kOklabLmsRootToLab.data(),
            sizeof(kOklabLmsRootToLab));
        Hash::hash_bytes_update(
            hash,
            kOklabLabToLmsRoot.data(),
            sizeof(kOklabLabToLmsRoot));
        out.hash = hash;
        out.valid = out.hash != 0;
        if (!out.valid) {
            diagnostic =
                "ResourceDescriptorMismatch component=output_gamut_transform field=identity";
        }
        return out.valid;
    }

    std::uint64_t output_boundary_contract_hash(
        const OutputGamutTransform& transform) {
        if (!transform.valid || transform.hash == 0) {
            return 0;
        }
        std::uint64_t hash = Hash::kFnvOffset;
        Hash::hash_bytes_update(
            hash,
            &kOutputBoundaryVersion,
            sizeof(kOutputBoundaryVersion));
        Hash::hash_bytes_update(
            hash,
            &transform.hash,
            sizeof(transform.hash));
        Hash::hash_bytes_update(
            hash,
            &kOutputBoundaryLightnessCount,
            sizeof(kOutputBoundaryLightnessCount));
        Hash::hash_bytes_update(
            hash,
            &kOutputBoundaryHueCount,
            sizeof(kOutputBoundaryHueCount));
        Hash::hash_bytes_update(
            hash,
            &kOutputBoundaryBisections,
            sizeof(kOutputBoundaryBisections));
        Hash::hash_bytes_update(
            hash,
            &kOutputBoundaryMinimumLightness,
            sizeof(kOutputBoundaryMinimumLightness));
        Hash::hash_bytes_update(
            hash,
            &kOutputBoundaryMaximumLightness,
            sizeof(kOutputBoundaryMaximumLightness));
        Hash::hash_bytes_update(
            hash,
            &kOutputBoundaryCubeSlack,
            sizeof(kOutputBoundaryCubeSlack));
        Hash::hash_bytes_update(
            hash,
            &kOutputBoundaryMaximumChroma,
            sizeof(kOutputBoundaryMaximumChroma));
        return hash;
    }

    bool build_output_boundary_table(
        const OutputGamutTransform& transform,
        OutputBoundaryTable& out) {
        out = OutputBoundaryTable{};
        out.contractHash = output_boundary_contract_hash(transform);
        if (out.contractHash == 0) {
            out.diagnostic =
                "ResourceDescriptorMismatch component=output_boundary_table field=transform";
            return false;
        }
        Matrix exit{};
        Matrix lmsToXyz{};
        for (std::size_t index = 0; index < exit.size(); ++index) {
            exit[index] = static_cast<double>(
                transform.d65XyzToNativeRgb[index]);
            lmsToXyz[index] =
                static_cast<double>(kOklabLmsToXyz[index]);
        }
        const Matrix lmsToNativeRgb = multiply_matrix(exit, lmsToXyz);
        std::array<std::array<double, 2>, kOutputBoundaryHueCount>
            hueDirections{};
        for (int hueIndex = 0; hueIndex < kOutputBoundaryHueCount; ++hueIndex) {
            const double hue =
                -0.5 * kTwoPi +
                kTwoPi * static_cast<double>(hueIndex) /
                    static_cast<double>(kOutputBoundaryHueCount);
            hueDirections[static_cast<std::size_t>(hueIndex)] = {
                std::cos(hue), std::sin(hue)};
        }
        for (int lightnessIndex = 0;
             lightnessIndex < kOutputBoundaryLightnessCount;
             ++lightnessIndex) {
            const double lightness =
                static_cast<double>(kOutputBoundaryMinimumLightness) +
                (static_cast<double>(kOutputBoundaryMaximumLightness) -
                 static_cast<double>(kOutputBoundaryMinimumLightness)) *
                    static_cast<double>(lightnessIndex) /
                    static_cast<double>(kOutputBoundaryLightnessCount - 1);
            for (int hueIndex = 0; hueIndex < kOutputBoundaryHueCount; ++hueIndex) {
                const auto& hueDirection =
                    hueDirections[static_cast<std::size_t>(hueIndex)];
                const std::array<Cubic, 3> cubics =
                    build_boundary_cubics(
                        lmsToNativeRgb,
                        lightness,
                        hueDirection);
                bool endpointOutside = false;
                bool neutralInside = true;
                double firstExit = std::numeric_limits<double>::infinity();
                for (const Cubic& cubic : cubics) {
                    neutralInside = neutralInside &&
                                    channel_inside_cube(
                                        evaluate_cubic(cubic, 0.0));
                    endpointOutside = endpointOutside ||
                                      !channel_inside_cube(evaluate_cubic(
                                          cubic,
                                          static_cast<double>(
                                              kOutputBoundaryMaximumChroma)));
                    firstExit = std::min(firstExit, first_channel_exit(cubic));
                }
                if (!endpointOutside) {
                    out.diagnostic =
                        "MalformedRequiredResource component=output_boundary_table requirement=c2_outside output_space=" +
                        std::to_string(OutputEncoding::toIndex(
                            transform.outputColorSpace)) +
                        " lightness_index=" + std::to_string(lightnessIndex) +
                        " hue_index=" + std::to_string(hueIndex);
                    return false;
                }
                if (!neutralInside) {
                    firstExit = 0.0;
                }
                if (!(std::isfinite(firstExit) && firstExit >= 0.0 &&
                      firstExit <= static_cast<double>(
                                       kOutputBoundaryMaximumChroma))) {
                    out.diagnostic =
                        "MalformedRequiredResource component=output_boundary_table requirement=finite_first_exit";
                    return false;
                }
                out.cmax[static_cast<std::size_t>(lightnessIndex) *
                             static_cast<std::size_t>(kOutputBoundaryHueCount) +
                         static_cast<std::size_t>(hueIndex)] =
                    static_cast<float>(firstExit);
            }
        }
        out.transformHash = transform.hash;
        std::uint64_t hash = Hash::kFnvOffset;
        Hash::hash_bytes_update(hash, &out.contractHash, sizeof(out.contractHash));
        Hash::hash_bytes_update(hash, out.cmax.data(), sizeof(out.cmax));
        out.hash = hash;
        out.valid = out.hash != 0;
        if (!out.valid) {
            out.diagnostic =
                "ResourceDescriptorMismatch component=output_boundary_table field=identity";
        }
        return out.valid;
    }

    std::array<float, 3> native_rgb_to_oklab(
        const OutputGamutTransform& transform,
        const std::array<float, 3>& rgb) {
        const std::array<float, 3> xyz =
            multiply_float_matrix_vector(transform.nativeRgbToD65Xyz, rgb);
        const std::array<float, 3> lms =
            multiply_float_matrix_vector(kOklabXyzToLms, xyz);
        const std::array<float, 3> root = {
            signed_cube_root(lms[0]),
            signed_cube_root(lms[1]),
            signed_cube_root(lms[2])};
        return multiply_float_matrix_vector(kOklabLmsRootToLab, root);
    }

    std::array<float, 3> oklab_to_native_rgb(
        const OutputGamutTransform& transform,
        const std::array<float, 3>& lab) {
        std::array<float, 3> root =
            multiply_float_matrix_vector(kOklabLabToLmsRoot, lab);
        for (float& component : root) {
            component = component * component * component;
        }
        const std::array<float, 3> xyz =
            multiply_float_matrix_vector(kOklabLmsToXyz, root);
        return multiply_float_matrix_vector(transform.d65XyzToNativeRgb, xyz);
    }

    bool build_input_compression_hull(
        const Spectral::Curve& xBar,
        const Spectral::Curve& yBar,
        const Spectral::Curve& zBar,
        const Spectral::Curve& d65,
        InputCompressionHull& out) {
        out = InputCompressionHull{};
        Locus locus{};
        Point center{};
        if (!build_locus(xBar, yBar, zBar, locus) ||
            !build_d65_center(xBar, yBar, zBar, d65, center)) {
            out.diagnostic =
                "MalformedRequiredResource component=input_gamut_hull requirement=cie1931_380_700_and_d65";
            return false;
        }

        std::array<double, kInputHullDirectionCount> reach{};
        for (int index = 0; index < kInputHullDirectionCount; ++index) {
            const double angle =
                kTwoPi * static_cast<double>(index) /
                static_cast<double>(kInputHullDirectionCount);
            const Point direction{std::cos(angle), std::sin(angle)};
            reach[static_cast<std::size_t>(index)] =
                ray_polygon_distance(Ray{center, direction}, locus);
            if (!(std::isfinite(reach[static_cast<std::size_t>(index)]) &&
                  reach[static_cast<std::size_t>(index)] > 0.0)) {
                out.diagnostic =
                    "MalformedRequiredResource component=input_gamut_hull requirement=finite_locus_ray_intersections";
                return false;
            }
        }

        std::array<double, kInputHullDirectionCount> smoothed{};
        fourier_lowpass(reach, smoothed);
        double inscribedScale = std::numeric_limits<double>::infinity();
        for (int index = 0; index < kInputHullDirectionCount; ++index) {
            double& radius = smoothed[static_cast<std::size_t>(index)];
            radius = std::max(radius, kMinimumRadius);
            inscribedScale = std::min(
                inscribedScale,
                reach[static_cast<std::size_t>(index)] / radius);
        }
        if (!(std::isfinite(inscribedScale) && inscribedScale > 0.0)) {
            out.diagnostic =
                "MalformedRequiredResource component=input_gamut_hull requirement=finite_inscribed_scale";
            return false;
        }

        out.center = {static_cast<float>(center[0]), static_cast<float>(center[1])};
        for (int index = 0; index < kInputHullDirectionCount; ++index) {
            const double angle =
                kTwoPi * static_cast<double>(index) /
                static_cast<double>(kInputHullDirectionCount);
            const double radius =
                smoothed[static_cast<std::size_t>(index)] * inscribedScale;
            out.xy[static_cast<std::size_t>(index)] = {
                static_cast<float>(center[0] + radius * std::cos(angle)),
                static_cast<float>(center[1] + radius * std::sin(angle))};
        }
        out.xy.back() = out.xy.front();
        std::uint64_t hash = Hash::kFnvOffset;
        Hash::hash_bytes_update(hash, &kInputCompressionVersion, sizeof(kInputCompressionVersion));
        Hash::hash_bytes_update(hash, out.center.data(), sizeof(out.center));
        Hash::hash_bytes_update(hash, out.xy.data(), sizeof(out.xy));
        out.hash = hash;
        out.valid = out.hash != 0;
        if (!out.valid) {
            out.diagnostic =
                "MalformedRequiredResource component=input_gamut_hull requirement=stable_identity";
        }
        return out.valid;
    }

    bool compress_input_xy(
        const InputCompressionHull& hull,
        const std::array<float, 2>& xy,
        std::array<float, 2>& out) {
        out = {};
        if (!hull.valid || hull.hash == 0 ||
            !std::isfinite(xy[0]) || !std::isfinite(xy[1])) {
            return false;
        }
        const Point center{
            static_cast<double>(hull.center[0]),
            static_cast<double>(hull.center[1])};
        const double deltaX = static_cast<double>(xy[0]) - center[0];
        const double deltaY = static_cast<double>(xy[1]) - center[1];
        const double distance = std::hypot(deltaX, deltaY);
        if (distance < kVertexTolerance) {
            out = xy;
            return true;
        }
        const Point direction{deltaX / distance, deltaY / distance};
        const double boundary =
            ray_polygon_distance(Ray{center, direction}, hull.xy);
        if (!(std::isfinite(boundary) && boundary > kMinimumRadius)) {
            return false;
        }
        const double compressedDistance =
            reinhard_knee(distance / boundary) * boundary;
        const double resultX = center[0] + direction[0] * compressedDistance;
        const double resultY = center[1] + direction[1] * compressedDistance;
        if (!(std::isfinite(resultX) && std::isfinite(resultY))) {
            return false;
        }
        out = {static_cast<float>(resultX), static_cast<float>(resultY)};
        return true;
    }

    bool remap_film_tc_lut_for_input_compression(
        const InputCompressionHull& hull,
        const Spectral::FilmTcLut& source,
        Spectral::FilmTcLut& out,
        std::string& diagnostic) {
        diagnostic.clear();
        constexpr int size = Spectral::FilmTcLut::kSize;
        constexpr std::size_t expected =
            static_cast<std::size_t>(size) * static_cast<std::size_t>(size) *
            Spectral::FilmTcLut::kChannels;
        if (!hull.valid || hull.hash == 0 || source.rgba.size() != expected) {
            diagnostic =
                "MalformedRequiredResource component=input_gamut_remap requirement=valid_hull_and_192x192x4_tc_lut";
            return false;
        }
        Spectral::FilmTcLut remapped;
        remapped.rgba.resize(expected);
        for (int c = 0; c < size; ++c) {
            const float tcC = static_cast<float>(c) / static_cast<float>(size - 1);
            const float root = std::sqrt(tcC);
            for (int m = 0; m < size; ++m) {
                const float tcM = static_cast<float>(m) / static_cast<float>(size - 1);
                const std::array<float, 2> xy{1.0f - root, tcM * root};
                std::array<float, 2> compressed{};
                if (!compress_input_xy(hull, xy, compressed)) {
                    diagnostic =
                        "MalformedRequiredResource component=input_gamut_remap requirement=finite_compressed_coordinates";
                    return false;
                }
                float sampleC = 0.0f;
                float sampleM = 0.0f;
                Spectral::tri2quad(
                    compressed[0],
                    compressed[1],
                    sampleC,
                    sampleM);
                const std::size_t base =
                    (static_cast<std::size_t>(c) * static_cast<std::size_t>(size) +
                     static_cast<std::size_t>(m)) *
                    Spectral::FilmTcLut::kChannels;
                for (int channel = 0; channel < 3; ++channel) {
                    const float value =
                        bilinear_sample(source, {sampleC, sampleM}, channel);
                    if (!std::isfinite(value)) {
                        diagnostic =
                            "MalformedRequiredResource component=input_gamut_remap requirement=finite_remapped_tc_lut";
                        return false;
                    }
                    remapped.rgba[base + static_cast<std::size_t>(channel)] = value;
                }
                remapped.rgba[base + 3u] = 0.0f;
            }
        }
        out = std::move(remapped);
        return true;
    }

} // namespace Gamut
