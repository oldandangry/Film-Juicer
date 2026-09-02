#include "ScatterHalation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "Hash.h"

namespace {
    bool convert_authored_control(
        double raw,
        const char* field,
        float& out,
        std::string& outDiagnostic) {
        const char* reason = nullptr;
        if (!std::isfinite(raw)) {
            reason = "non_finite";
        } else if (raw < 0.0) {
            reason = "below_minimum";
        } else if (raw > 2.0) {
            reason = "above_maximum";
        }
        if (reason) {
            outDiagnostic = "InvalidAuthoredControl component=scatter_halation field=";
            outDiagnostic += field;
            outDiagnostic += " reason=";
            outDiagnostic += reason;
            return false;
        }

        const float narrowed = static_cast<float>(raw);
        out = narrowed == 0.0f ? 0.0f : narrowed;
        return true;
    }

    constexpr std::array<double, 3> kScatterCoreUm{{2.2, 2.0, 1.6}};
    constexpr std::array<double, 3> kScatterTailUm{{9.3, 9.7, 9.1}};
    constexpr std::array<double, 3> kExponentialSigmaRatios{{0.5360,
                                                             1.5236,
                                                             2.7684}};
    constexpr std::array<double, 3> kBounceSigmaMultipliers{{1.0,
                                                             1.4142135623730950488,
                                                             1.7320508075688772935}};
    constexpr double kMinimumActiveSigmaPixels = 1.0e-6;
    constexpr double kFirDispatchSigmaPixels = 3.0;

    bool any_positive(const std::array<float, 3>& values) {
        return std::any_of(values.begin(), values.end(), [](float value) {
            return value > 0.0f;
        });
    }

    float canonical_float(float value) {
        return value == 0.0f ? 0.0f : value;
    }

    template <typename T>
    void hash_value(std::uint64_t& hash, const T& value) {
        Hash::hash_bytes_update(hash, &value, sizeof(value));
    }

    std::uint64_t hash_scatter_halation_recipe(
        const ScatterHalationOpticsRecipe& recipe) {
        if (!recipe.scatterActive && !recipe.backReflectionActive) {
            return 0;
        }
        std::uint64_t hash = Hash::kFnvOffset;
        hash_value(hash, recipe.scatterActive);
        if (recipe.scatterActive) {
            hash_value(hash, recipe.scatterAmount);
            hash_value(hash, recipe.scatterSpatialScale);
        }
        hash_value(hash, recipe.backReflectionActive);
        if (recipe.backReflectionActive) {
            hash_value(hash, recipe.halationSpatialScale);
            Hash::hash_bytes_update(
                hash,
                recipe.halationFirstSigmaUm.data(),
                sizeof(recipe.halationFirstSigmaUm));
            Hash::hash_bytes_update(
                hash,
                recipe.totalStrength.data(),
                sizeof(recipe.totalStrength));
        }
        return hash == 0 ? 1 : hash;
    }

    bool fail_descriptor(
        const char* field,
        double sigmaPixels,
        std::string& outDiagnostic) {
        outDiagnostic =
            "ResourceDescriptorMismatch component=scatter_halation field=";
        outDiagnostic += field;
        outDiagnostic += " sigma_pixels=";
        outDiagnostic += std::to_string(sigmaPixels);
        return false;
    }

    bool build_gaussian_descriptor(
        double sigmaPixels,
        const char* field,
        ScatterHalationGaussianDescriptor& out,
        std::string& outDiagnostic) {
        out = ScatterHalationGaussianDescriptor{};
        if (!std::isfinite(sigmaPixels)) {
            return fail_descriptor(field, sigmaPixels, outDiagnostic);
        }
        if (!(sigmaPixels > 0.0)) {
            return true;
        }

        if (sigmaPixels < kFirDispatchSigmaPixels) {
            const int radius = static_cast<int>(3.0 * sigmaPixels + 0.5);
            if (radius == 0) {
                return true;
            }
            if (radius < 0 || radius > 9) {
                return fail_descriptor(field, sigmaPixels, outDiagnostic);
            }

            std::array<double, 19> weights{};
            double sum = 0.0;
            const int count = 2 * radius + 1;
            for (int index = 0; index < count; ++index) {
                const double x = static_cast<double>(index - radius);
                const double value =
                    std::exp(-(x * x) / (2.0 * sigmaPixels * sigmaPixels));
                if (!std::isfinite(value)) {
                    return fail_descriptor(field, sigmaPixels, outDiagnostic);
                }
                weights[static_cast<std::size_t>(index)] = value;
                sum += value;
            }
            if (!std::isfinite(sum) || !(sum > 0.0)) {
                return fail_descriptor(field, sigmaPixels, outDiagnostic);
            }

            out.kind = ScatterHalationGaussianKind::FirReflect;
            out.radius = radius;
            for (int index = 0; index < count; ++index) {
                const float narrowed = static_cast<float>(
                    weights[static_cast<std::size_t>(index)] / sum);
                if (!std::isfinite(narrowed)) {
                    return fail_descriptor(field, sigmaPixels, outDiagnostic);
                }
                out.firWeights[static_cast<std::size_t>(index)] =
                    canonical_float(narrowed);
            }
            return true;
        }

        const double q = 0.98711 * sigmaPixels - 0.96330;
        const double q2 = q * q;
        const double q3 = q2 * q;
        const double b0 =
            1.57825 + 2.44413 * q + 1.4281 * q2 + 0.422205 * q3;
        const double b1 = 2.44413 * q + 2.85619 * q2 + 1.26661 * q3;
        const double b2 = -(1.4281 * q2 + 1.26661 * q3);
        const double b3 = 0.422205 * q3;
        const std::array<double, 4> coefficients{{1.0 - (b1 + b2 + b3) / b0,
                                                  b1 / b0,
                                                  b2 / b0,
                                                  b3 / b0}};
        if (!std::all_of(
                coefficients.begin(),
                coefficients.end(),
                [](double value) {
                    return std::isfinite(value);
                })) {
            return fail_descriptor(field, sigmaPixels, outDiagnostic);
        }

        out.kind = ScatterHalationGaussianKind::YvvReplicate;
        out.B = static_cast<float>(coefficients[0]);
        out.B1 = static_cast<float>(coefficients[1]);
        out.B2 = static_cast<float>(coefficients[2]);
        out.B3 = static_cast<float>(coefficients[3]);
        if (!std::isfinite(out.B) || !std::isfinite(out.B1) ||
            !std::isfinite(out.B2) || !std::isfinite(out.B3)) {
            return fail_descriptor(field, sigmaPixels, outDiagnostic);
        }
        return true;
    }
} // namespace

namespace Spektrafilm {
    bool build_scatter_halation_controls(
        const ScatterHalationRawControls& raw,
        ScatterHalationControls& out,
        std::string& outDiagnostic) {
        out = ScatterHalationControls{};
        outDiagnostic.clear();
        if (!raw.active) {
            return true;
        }

        ScatterHalationControls converted{};
        converted.active = true;
        if (!convert_authored_control(
                raw.scatterAmount,
                "scatter_amount",
                converted.scatterAmount,
                outDiagnostic) ||
            !convert_authored_control(
                raw.scatterSpatialScale,
                "scatter_spatial_scale",
                converted.scatterSpatialScale,
                outDiagnostic) ||
            !convert_authored_control(
                raw.halationAmount,
                "halation_amount",
                converted.halationAmount,
                outDiagnostic) ||
            !convert_authored_control(
                raw.halationSpatialScale,
                "halation_spatial_scale",
                converted.halationSpatialScale,
                outDiagnostic)) {
            return false;
        }

        out = converted;
        return true;
    }

    bool resolve_scatter_halation_recipe(
        const ScatterHalationControls& controls,
        const Profiles::ProfileDigest& profileDigest,
        ScatterHalationOpticsRecipe& out,
        std::string& outDiagnostic) {
        out = ScatterHalationOpticsRecipe{};
        outDiagnostic.clear();
        if (!controls.active) {
            return true;
        }

        const bool fixedScatterSupport =
            std::any_of(kScatterCoreUm.begin(), kScatterCoreUm.end(), [](double value) {
                return value > 0.0;
            }) ||
            std::any_of(kScatterTailUm.begin(), kScatterTailUm.end(), [](double value) {
                return value > 0.0;
            });
        out.scatterActive = controls.scatterAmount > 0.0f &&
                            controls.scatterSpatialScale > 0.0f &&
                            fixedScatterSupport;
        if (out.scatterActive) {
            out.scatterAmount = canonical_float(controls.scatterAmount);
            out.scatterSpatialScale = canonical_float(controls.scatterSpatialScale);
        }

        out.backReflectionActive = controls.halationAmount > 0.0f &&
                                   controls.halationSpatialScale > 0.0f &&
                                   any_positive(profileDigest.halationPrimaryAmount) &&
                                   any_positive(profileDigest.halationFirstSigmaUm);
        if (out.backReflectionActive) {
            out.halationSpatialScale =
                canonical_float(controls.halationSpatialScale);
            out.halationFirstSigmaUm = profileDigest.halationFirstSigmaUm;
            for (std::size_t channel = 0; channel < out.totalStrength.size(); ++channel) {
                const double strength =
                    static_cast<double>(profileDigest.halationPrimaryAmount[channel]) *
                    static_cast<double>(controls.halationAmount);
                if (!std::isfinite(strength) || strength < 0.0 ||
                    strength > static_cast<double>(std::numeric_limits<float>::max())) {
                    out = ScatterHalationOpticsRecipe{};
                    outDiagnostic =
                        "ResourceDescriptorMismatch component=scatter_halation "
                        "field=total_strength channel=" +
                        std::to_string(channel);
                    return false;
                }
                const float narrowed = static_cast<float>(strength);
                if (!std::isfinite(narrowed) || narrowed < 0.0f) {
                    out = ScatterHalationOpticsRecipe{};
                    outDiagnostic =
                        "ResourceDescriptorMismatch component=scatter_halation "
                        "field=total_strength channel=" +
                        std::to_string(channel);
                    return false;
                }
                out.totalStrength[channel] = canonical_float(narrowed);
            }
            if (!any_positive(out.totalStrength)) {
                out.backReflectionActive = false;
                out.halationSpatialScale = 0.0f;
                out.halationFirstSigmaUm = {};
                out.totalStrength = {};
            }
        }

        out.hash = hash_scatter_halation_recipe(out);
        return true;
    }

    bool build_scatter_halation_frame_descriptor(
        const ScatterHalationOpticsRecipe& recipe,
        float pixelSizeUm,
        std::optional<ScatterHalationFrameDescriptor>& out,
        std::string& outDiagnostic) {
        out.reset();
        outDiagnostic.clear();
        if (recipe.hash == 0) {
            return true;
        }
        if (!std::isfinite(pixelSizeUm) || !(pixelSizeUm > 0.0f)) {
            outDiagnostic =
                "ResourceDescriptorMismatch component=scatter_halation "
                "field=pixel_size_um";
            return false;
        }

        ScatterHalationFrameDescriptor descriptor{};
        descriptor.recipeHash = recipe.hash;
        descriptor.scatterAmount =
            recipe.scatterActive ? canonical_float(recipe.scatterAmount) : 0.0f;
        const double pixelSize = static_cast<double>(pixelSizeUm);
        for (std::size_t channel = 0; channel < descriptor.channels.size(); ++channel) {
            ScatterHalationChannelDescriptor& channelDescriptor =
                descriptor.channels[channel];
            if (recipe.scatterActive) {
                const double scale = static_cast<double>(recipe.scatterSpatialScale);
                const double coreSigma = std::max(
                    kScatterCoreUm[channel] * scale / pixelSize,
                    kMinimumActiveSigmaPixels);
                if (!build_gaussian_descriptor(
                        coreSigma,
                        "scatter_core",
                        channelDescriptor.core,
                        outDiagnostic)) {
                    return false;
                }
                for (std::size_t index = 0; index < channelDescriptor.tail.size(); ++index) {
                    const double tailSigma = std::max(
                        kScatterTailUm[channel] * kExponentialSigmaRatios[index] *
                            scale / pixelSize,
                        kMinimumActiveSigmaPixels);
                    if (!build_gaussian_descriptor(
                            tailSigma,
                            "scatter_tail",
                            channelDescriptor.tail[index],
                            outDiagnostic)) {
                        return false;
                    }
                }
            }

            channelDescriptor.totalStrength = recipe.backReflectionActive
                                                  ? canonical_float(
                                                        recipe.totalStrength[channel])
                                                  : 0.0f;
            if (recipe.backReflectionActive) {
                const double firstSigma =
                    static_cast<double>(recipe.halationFirstSigmaUm[channel]);
                const double scale = static_cast<double>(recipe.halationSpatialScale);
                for (std::size_t index = 0; index < channelDescriptor.bounce.size(); ++index) {
                    const double bounceSigma = std::max(
                        firstSigma * kBounceSigmaMultipliers[index] * scale /
                            pixelSize,
                        kMinimumActiveSigmaPixels);
                    if (!build_gaussian_descriptor(
                            bounceSigma,
                            "back_reflection_bounce",
                            channelDescriptor.bounce[index],
                            outDiagnostic)) {
                        return false;
                    }
                }
            }
        }

        if (descriptor.recipeHash == 0) {
            outDiagnostic =
                "ResourceDescriptorMismatch component=scatter_halation "
                "field=recipe_hash";
            return false;
        }
        out = descriptor;
        return true;
    }
} // namespace Spektrafilm
