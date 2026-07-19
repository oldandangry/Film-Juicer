#include "DiffusionHostBehavior.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <utility>

namespace {

    constexpr std::uint64_t kFnvOffset = 0xcbf29ce484222325ULL;
    constexpr std::uint64_t kFnvPrime = 0x100000001b3ULL;
    constexpr double kMinimumScale = 1.0e-6;
    constexpr double kMaximumScatterFraction = 0.99;
    constexpr std::array<double, 3> kHaloWarmthAxis{{1.30, 0.15, -1.45}};
    constexpr std::array<double, 5> kStrengthBreakpoints{{0.125, 0.25, 0.5, 1.0, 2.0}};
    constexpr std::array<double, 5> kStrengthFractions{{0.10, 0.20, 0.35, 0.55, 0.75}};

    struct GroupConfig {
        double centerLambdaUm = 0.0;
        double spread = 1.0;
        std::size_t componentCount = 1;
        double alpha = 3.0;
    };

    struct FamilyConfig {
        GroupConfig core;
        GroupConfig halo;
        GroupConfig bloom;
        std::array<double, 3> groupWeights{};
        double haloWarmthBase = 0.0;
        double scatterGain = 1.0;
    };

    constexpr std::array<FamilyConfig, 4> kFamilyConfigs{{FamilyConfig{
                                                              GroupConfig{10.0, 1.5, 2, 3.0},
                                                              GroupConfig{50.0, 2.0, 3, 3.0},
                                                              GroupConfig{260.0, 2.5, 4, 3.2},
                                                              {0.60, 0.30, 0.10},
                                                              0.0,
                                                              0.65},
                                                          FamilyConfig{
                                                              GroupConfig{16.0, 1.5, 2, 3.0},
                                                              GroupConfig{95.0, 2.0, 3, 3.0},
                                                              GroupConfig{380.0, 2.5, 4, 3.5},
                                                              {0.40, 0.47, 0.13},
                                                              0.65,
                                                              0.75},
                                                          FamilyConfig{
                                                              GroupConfig{14.0, 1.5, 2, 3.0},
                                                              GroupConfig{150.0, 2.0, 3, 3.0},
                                                              GroupConfig{650.0, 2.5, 4, 2.9},
                                                              {0.28, 0.42, 0.30},
                                                              0.40,
                                                              1.05},
                                                          FamilyConfig{
                                                              GroupConfig{20.0, 1.5, 2, 3.0},
                                                              GroupConfig{200.0, 2.0, 3, 3.0},
                                                              GroupConfig{1000.0, 2.5, 4, 2.5},
                                                              {0.22, 0.30, 0.48},
                                                              0.85,
                                                              1.00}}};

    struct ExpandedGroup {
        std::array<double, 4> lambdasUm{};
        std::array<double, 4> weights{};
        std::size_t count = 0;
    };

    void fail(std::string& diagnostic, std::string_view field) {
        diagnostic = "InvalidDiffusionHostBehavior field=";
        diagnostic.append(field);
    }

    const FamilyConfig* family_config(Spektrafilm::DiffusionFilterFamily family) {
        using Spektrafilm::DiffusionFilterFamily;
        switch (family) {
            case DiffusionFilterFamily::Glimmerglass:
                return &kFamilyConfigs[0];
            case DiffusionFilterFamily::BlackProMist:
                return &kFamilyConfigs[1];
            case DiffusionFilterFamily::ProMist:
                return &kFamilyConfigs[2];
            case DiffusionFilterFamily::Cinebloom:
                return &kFamilyConfigs[3];
            default:
                return nullptr;
        }
    }

    void hash_byte(std::uint64_t& hash, std::uint8_t value) {
        hash ^= static_cast<std::uint64_t>(value);
        hash *= kFnvPrime;
    }

    void hash_u64_le(std::uint64_t& hash, std::uint64_t value) {
        for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
            hash_byte(hash, static_cast<std::uint8_t>((value >> (byte * 8)) & 0xffu));
        }
    }

    void hash_u32_le(std::uint64_t& hash, std::uint32_t value) {
        for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
            hash_byte(hash, static_cast<std::uint8_t>((value >> (byte * 8)) & 0xffu));
        }
    }

    void hash_tag(std::uint64_t& hash, std::string_view tag) {
        for (const char value : tag) {
            hash_byte(hash, static_cast<std::uint8_t>(value));
        }
        hash_byte(hash, 0);
    }

    void hash_bool(std::uint64_t& hash, bool value) {
        hash_byte(hash, value ? 1u : 0u);
    }

    void hash_family(std::uint64_t& hash, Spektrafilm::DiffusionFilterFamily family) {
        hash_byte(hash, static_cast<std::uint8_t>(family));
    }

    void hash_schema(std::uint64_t& hash) {
        hash_u32_le(hash, Spektrafilm::kDiffusionHostBehaviorSchemaVersion);
    }

    void hash_double(std::uint64_t& hash, double value) {
        if (value == 0.0) {
            value = 0.0;
        }
        hash_u64_le(hash, std::bit_cast<std::uint64_t>(value));
    }

    template <std::size_t N>
    void hash_double_array(std::uint64_t& hash, const std::array<double, N>& values) {
        for (const double value : values) {
            hash_double(hash, value);
        }
    }

    std::uint64_t resolved_hash(
        const Spektrafilm::DiffusionFilterResolvedParameters& resolved) {
        if (!resolved.active) {
            return 0;
        }
        std::uint64_t hash = kFnvOffset;
        hash_tag(hash, "diffusion-resolved");
        hash_schema(hash);
        hash_family(hash, resolved.family);
        hash_double(hash, resolved.scatterFraction);
        hash_double_array(hash, resolved.groupWeightsCoreHaloBloom);
        hash_double_array(hash, resolved.groupCenterLambdaUm);
        hash_double(hash, resolved.effectiveWarmth);
        hash_double(hash, resolved.radiusScale);
        hash_double(hash, resolved.samplingScale);
        return hash;
    }

    std::uint64_t sample_descriptor_hash(
        const Spektrafilm::DiffusionPsfSampleDescriptor& descriptor) {
        std::uint64_t hash = kFnvOffset;
        hash_tag(hash, "diffusion-psf-sample");
        hash_schema(hash);
        hash_family(hash, descriptor.family);
        hash_double_array(hash, descriptor.groupWeightsCoreHaloBloom);
        hash_double_array(hash, descriptor.groupCenterLambdaUm);
        hash_double(hash, descriptor.effectiveWarmth);
        hash_double(hash, descriptor.samplingScale);
        hash_double(hash, descriptor.pixelSizeUm);
        hash_u32_le(hash, static_cast<std::uint32_t>(descriptor.radiusPixels));
        return hash;
    }

    bool validate_authored_finite(
        const Spektrafilm::DiffusionFilterAuthoredControls& authored,
        std::string& diagnostic) {
        const std::array<std::pair<double, std::string_view>, 9> values{{{authored.strength, "strength"},
                                                                         {authored.spatialScale, "spatial_scale"},
                                                                         {authored.haloWarmth, "halo_warmth"},
                                                                         {authored.coreIntensity, "core_intensity"},
                                                                         {authored.coreSize, "core_size"},
                                                                         {authored.haloIntensity, "halo_intensity"},
                                                                         {authored.haloSize, "halo_size"},
                                                                         {authored.bloomIntensity, "bloom_intensity"},
                                                                         {authored.bloomSize, "bloom_size"}}};
        for (const auto& [value, field] : values) {
            if (!std::isfinite(value)) {
                fail(diagnostic, field);
                return false;
            }
        }
        return true;
    }

    bool validate_resolved(
        const Spektrafilm::DiffusionFilterResolvedParameters& resolved,
        std::string& diagnostic) {
        if (!resolved.active || resolved.hash == 0) {
            fail(diagnostic, "resolved_inactive");
            return false;
        }
        if (!family_config(resolved.family)) {
            fail(diagnostic, "family");
            return false;
        }
        if (!std::isfinite(resolved.scatterFraction) || resolved.scatterFraction <= 0.0 ||
            resolved.scatterFraction > kMaximumScatterFraction) {
            fail(diagnostic, "scatter_fraction");
            return false;
        }
        double weightTotal = 0.0;
        for (const double weight : resolved.groupWeightsCoreHaloBloom) {
            if (!std::isfinite(weight) || weight < 0.0) {
                fail(diagnostic, "group_weights");
                return false;
            }
            weightTotal += weight;
        }
        if (!(std::isfinite(weightTotal) && weightTotal > 0.0)) {
            fail(diagnostic, "group_weights");
            return false;
        }
        for (const double center : resolved.groupCenterLambdaUm) {
            if (!(std::isfinite(center) && center > 0.0)) {
                fail(diagnostic, "group_centers_um");
                return false;
            }
        }
        if (!std::isfinite(resolved.effectiveWarmth) || resolved.effectiveWarmth < -1.5 ||
            resolved.effectiveWarmth > 1.5) {
            fail(diagnostic, "effective_warmth");
            return false;
        }
        if (!(std::isfinite(resolved.radiusScale) && resolved.radiusScale > 0.0)) {
            fail(diagnostic, "radius_scale");
            return false;
        }
        if (!(std::isfinite(resolved.samplingScale) && resolved.samplingScale > 0.0)) {
            fail(diagnostic, "sampling_scale");
            return false;
        }
        if (resolved_hash(resolved) != resolved.hash) {
            fail(diagnostic, "resolved_hash");
            return false;
        }
        return true;
    }

    bool validate_sample_descriptor(
        const Spektrafilm::DiffusionPsfSampleDescriptor& descriptor,
        std::string& diagnostic) {
        if (descriptor.hash == 0) {
            fail(diagnostic, "sample_descriptor_inactive");
            return false;
        }
        if (!family_config(descriptor.family)) {
            fail(diagnostic, "family");
            return false;
        }
        double weightTotal = 0.0;
        for (const double weight : descriptor.groupWeightsCoreHaloBloom) {
            if (!std::isfinite(weight) || weight < 0.0) {
                fail(diagnostic, "group_weights");
                return false;
            }
            weightTotal += weight;
        }
        if (!(std::isfinite(weightTotal) && weightTotal > 0.0)) {
            fail(diagnostic, "group_weights");
            return false;
        }
        for (const double center : descriptor.groupCenterLambdaUm) {
            if (!(std::isfinite(center) && center > 0.0)) {
                fail(diagnostic, "group_centers_um");
                return false;
            }
        }
        if (!std::isfinite(descriptor.effectiveWarmth) || descriptor.effectiveWarmth < -1.5 ||
            descriptor.effectiveWarmth > 1.5) {
            fail(diagnostic, "effective_warmth");
            return false;
        }
        if (!(std::isfinite(descriptor.samplingScale) && descriptor.samplingScale > 0.0)) {
            fail(diagnostic, "sampling_scale");
            return false;
        }
        if (!(std::isfinite(descriptor.pixelSizeUm) && descriptor.pixelSizeUm > 0.0)) {
            fail(diagnostic, "pixel_size_um");
            return false;
        }
        if (descriptor.radiusPixels <= 0) {
            fail(diagnostic, "radius_pixels");
            return false;
        }
        if (sample_descriptor_hash(descriptor) != descriptor.hash) {
            fail(diagnostic, "sample_descriptor_hash");
            return false;
        }
        return true;
    }

    double strength_to_scatter(double strength, const FamilyConfig& family) {
        if (strength <= 0.0) {
            return 0.0;
        }
        const double logStrength = std::log2(std::max(strength, kMinimumScale));
        const double firstBreak = std::log2(kStrengthBreakpoints.front());
        const double lastBreak = std::log2(kStrengthBreakpoints.back());
        double baseFraction = kStrengthFractions.front();
        if (logStrength >= lastBreak) {
            baseFraction = kStrengthFractions.back();
        } else if (logStrength > firstBreak) {
            for (std::size_t index = 0; index + 1 < kStrengthBreakpoints.size(); ++index) {
                const double lower = std::log2(kStrengthBreakpoints[index]);
                const double upper = std::log2(kStrengthBreakpoints[index + 1]);
                if (logStrength <= upper) {
                    const double t = (logStrength - lower) / (upper - lower);
                    baseFraction = kStrengthFractions[index] +
                                   t * (kStrengthFractions[index + 1] -
                                        kStrengthFractions[index]);
                    break;
                }
            }
        }
        return std::clamp(baseFraction * family.scatterGain, 0.0, kMaximumScatterFraction);
    }

    bool expand_group(
        const GroupConfig& config,
        double resolvedCenterLambdaUm,
        bool bloom,
        ExpandedGroup& out) {
        out = ExpandedGroup{};
        if (!(std::isfinite(resolvedCenterLambdaUm) && resolvedCenterLambdaUm > 0.0) ||
            config.componentCount == 0 || config.componentCount > out.lambdasUm.size()) {
            return false;
        }
        out.count = config.componentCount;
        if (config.componentCount == 1 || config.spread <= 1.0) {
            out.lambdasUm[0] = resolvedCenterLambdaUm;
            out.weights[0] = 1.0;
            return true;
        }

        const double logLow = std::log(resolvedCenterLambdaUm / config.spread);
        const double logHigh = std::log(resolvedCenterLambdaUm * config.spread);
        if (!(std::isfinite(logLow) && std::isfinite(logHigh))) {
            return false;
        }
        for (std::size_t index = 0; index < config.componentCount; ++index) {
            double logLambda = logLow;
            if (index + 1 == config.componentCount) {
                logLambda = logHigh;
            } else if (index != 0) {
                const double t = static_cast<double>(index) /
                                 static_cast<double>(config.componentCount - 1);
                logLambda = logLow + (logHigh - logLow) * t;
            }
            out.lambdasUm[index] = std::exp(logLambda);
            out.weights[index] = bloom
                                     ? std::pow(out.lambdasUm[index], 2.0 - config.alpha)
                                     : 1.0;
            if (!(std::isfinite(out.lambdasUm[index]) && out.lambdasUm[index] > 0.0) ||
                !(std::isfinite(out.weights[index]) && out.weights[index] >= 0.0)) {
                return false;
            }
        }
        double total = 0.0;
        for (std::size_t index = 0; index < config.componentCount; ++index) {
            total += out.weights[index];
        }
        if (!(std::isfinite(total) && total > 0.0)) {
            return false;
        }
        for (std::size_t index = 0; index < config.componentCount; ++index) {
            out.weights[index] /= total;
        }
        return true;
    }

    bool halo_channel_weights(
        const ExpandedGroup& halo,
        double warmth,
        std::array<std::array<double, 4>, 3>& out) {
        out = {};
        if (halo.count == 0 || halo.count > halo.weights.size()) {
            return false;
        }
        if (halo.count < 2) {
            for (auto& channel : out) {
                channel = halo.weights;
            }
            return true;
        }

        std::array<double, 4> gradient{};
        double targetTotal = 0.0;
        double weightedGradient = 0.0;
        for (std::size_t index = 0; index < halo.count; ++index) {
            gradient[index] = -1.0 +
                              2.0 * static_cast<double>(index) /
                                  static_cast<double>(halo.count - 1);
            targetTotal += halo.weights[index];
            weightedGradient += halo.weights[index] * gradient[index];
        }
        if (!(std::isfinite(targetTotal) && targetTotal > 0.0)) {
            return false;
        }
        weightedGradient /= targetTotal;
        for (std::size_t index = 0; index < halo.count; ++index) {
            gradient[index] -= weightedGradient;
        }

        for (std::size_t channel = 0; channel < out.size(); ++channel) {
            double channelTotal = 0.0;
            for (std::size_t index = 0; index < halo.count; ++index) {
                out[channel][index] = std::max(
                    halo.weights[index] *
                        (1.0 + warmth * kHaloWarmthAxis[channel] * gradient[index]),
                    0.0);
                channelTotal += out[channel][index];
            }
            if (channelTotal > 0.0) {
                const double scale = targetTotal / channelTotal;
                for (std::size_t index = 0; index < halo.count; ++index) {
                    out[channel][index] *= scale;
                }
            } else {
                out[channel] = halo.weights;
            }
        }
        return true;
    }

    bool finite_component_output(const Spektrafilm::DiffusionPsfComponents& components) {
        for (std::size_t index = 0; index < components.kCount; ++index) {
            if (!(std::isfinite(components.lambdaPixels[index]) &&
                  components.lambdaPixels[index] > 0.0) ||
                !(std::isfinite(components.redWeights[index]) &&
                  components.redWeights[index] >= 0.0) ||
                !(std::isfinite(components.greenWeights[index]) &&
                  components.greenWeights[index] >= 0.0) ||
                !(std::isfinite(components.blueWeights[index]) &&
                  components.blueWeights[index] >= 0.0)) {
                return false;
            }
        }
        return true;
    }

} // namespace

namespace Spektrafilm {

    std::uint64_t hash_diffusion_authored_controls(
        const DiffusionFilterAuthoredControls& controls) {
        std::uint64_t hash = kFnvOffset;
        hash_tag(hash, "diffusion-authored");
        hash_schema(hash);
        hash_bool(hash, controls.active);
        hash_family(hash, controls.family);
        hash_double(hash, controls.strength);
        hash_double(hash, controls.spatialScale);
        hash_double(hash, controls.haloWarmth);
        hash_double(hash, controls.coreIntensity);
        hash_double(hash, controls.coreSize);
        hash_double(hash, controls.haloIntensity);
        hash_double(hash, controls.haloSize);
        hash_double(hash, controls.bloomIntensity);
        hash_double(hash, controls.bloomSize);
        return hash;
    }

    bool resolve_diffusion_filter(
        const DiffusionFilterAuthoredControls& authored,
        DiffusionFilterResolvedParameters& out,
        std::string& diagnostic) {
        out = DiffusionFilterResolvedParameters{};
        diagnostic.clear();

        const FamilyConfig* config = family_config(authored.family);
        if (!config) {
            fail(diagnostic, "family");
            return false;
        }
        if (!validate_authored_finite(authored, diagnostic)) {
            return false;
        }
        if (!authored.active || authored.strength <= 0.0 || authored.spatialScale <= 0.0) {
            return true;
        }

        out.active = true;
        out.family = authored.family;
        out.scatterFraction = strength_to_scatter(authored.strength, *config);
        if (!(std::isfinite(out.scatterFraction) && out.scatterFraction > 0.0)) {
            fail(diagnostic, "scatter_fraction");
            out = DiffusionFilterResolvedParameters{};
            return false;
        }

        const std::array<double, 3> intensityMultipliers{{std::max(authored.coreIntensity, 0.0),
                                                          std::max(authored.haloIntensity, 0.0),
                                                          std::max(authored.bloomIntensity, 0.0)}};
        std::array<double, 3> weightedGroups{};
        double weightedTotal = 0.0;
        for (std::size_t group = 0; group < weightedGroups.size(); ++group) {
            weightedGroups[group] = config->groupWeights[group] * intensityMultipliers[group];
            weightedTotal += weightedGroups[group];
        }

        if (weightedTotal <= 0.0) {
            out.groupWeightsCoreHaloBloom = config->groupWeights;
            out.groupCenterLambdaUm = {
                config->core.centerLambdaUm,
                config->halo.centerLambdaUm,
                config->bloom.centerLambdaUm};
        } else {
            if (!std::isfinite(weightedTotal)) {
                fail(diagnostic, "group_weights");
                out = DiffusionFilterResolvedParameters{};
                return false;
            }
            for (std::size_t group = 0; group < weightedGroups.size(); ++group) {
                out.groupWeightsCoreHaloBloom[group] = weightedGroups[group] / weightedTotal;
            }
            out.groupCenterLambdaUm = {
                config->core.centerLambdaUm * std::max(authored.coreSize, kMinimumScale),
                config->halo.centerLambdaUm * std::max(authored.haloSize, kMinimumScale),
                config->bloom.centerLambdaUm * std::max(authored.bloomSize, kMinimumScale)};
        }

        for (const double center : out.groupCenterLambdaUm) {
            if (!(std::isfinite(center) && center > 0.0)) {
                fail(diagnostic, "group_centers_um");
                out = DiffusionFilterResolvedParameters{};
                return false;
            }
        }
        out.effectiveWarmth = std::clamp(
            config->haloWarmthBase + authored.haloWarmth,
            -1.5,
            1.5);
        out.radiusScale = authored.spatialScale;
        out.samplingScale = std::max(authored.spatialScale, kMinimumScale);
        out.hash = resolved_hash(out);
        if (out.hash == 0) {
            fail(diagnostic, "resolved_hash");
            out = DiffusionFilterResolvedParameters{};
            return false;
        }
        return true;
    }

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    bool build_diffusion_psf_sample_descriptor(
        const DiffusionFilterResolvedParameters& resolved,
        double pixelSizeUm,
        int fullFrameWidth,
        int fullFrameHeight,
        DiffusionPsfSampleDescriptor& out,
        std::string& diagnostic) {
        out = DiffusionPsfSampleDescriptor{};
        diagnostic.clear();
        if (!validate_resolved(resolved, diagnostic)) {
            return false;
        }
        if (!(std::isfinite(pixelSizeUm) && pixelSizeUm > 0.0)) {
            fail(diagnostic, "pixel_size_um");
            return false;
        }
        if (fullFrameWidth <= 0 || fullFrameHeight <= 0) {
            fail(diagnostic, "full_frame_extent");
            return false;
        }

        const FamilyConfig* config = family_config(resolved.family);
        const double bloomMaximumLambdaUm =
            resolved.groupCenterLambdaUm[2] * config->bloom.spread;
        const double requestedRadius = std::ceil(std::max(
            8.0 * bloomMaximumLambdaUm * resolved.radiusScale / pixelSizeUm,
            5.0));
        if (!(std::isfinite(requestedRadius) && requestedRadius > 0.0)) {
            fail(diagnostic, "radius_pixels");
            return false;
        }
        const int radiusCap =
            std::max(std::min(fullFrameWidth, fullFrameHeight) / 2 - 1, 1);
        const int radius = requestedRadius >= static_cast<double>(radiusCap)
                               ? radiusCap
                               : static_cast<int>(requestedRadius);
        if (radius <= 0) {
            fail(diagnostic, "radius_pixels");
            return false;
        }

        out.family = resolved.family;
        out.groupWeightsCoreHaloBloom = resolved.groupWeightsCoreHaloBloom;
        out.groupCenterLambdaUm = resolved.groupCenterLambdaUm;
        out.effectiveWarmth = resolved.effectiveWarmth;
        out.samplingScale = resolved.samplingScale;
        out.pixelSizeUm = pixelSizeUm;
        out.radiusPixels = radius;
        out.hash = sample_descriptor_hash(out);
        if (out.hash == 0) {
            fail(diagnostic, "sample_descriptor_hash");
            out = DiffusionPsfSampleDescriptor{};
            return false;
        }
        return true;
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

    bool expand_diffusion_psf_components(
        const DiffusionPsfSampleDescriptor& descriptor,
        DiffusionPsfComponents& out,
        std::string& diagnostic) {
        out = DiffusionPsfComponents{};
        diagnostic.clear();
        if (!validate_sample_descriptor(descriptor, diagnostic)) {
            return false;
        }
        const FamilyConfig* config = family_config(descriptor.family);
        ExpandedGroup core{};
        ExpandedGroup halo{};
        ExpandedGroup bloom{};
        if (!expand_group(config->core, descriptor.groupCenterLambdaUm[0], false, core) ||
            !expand_group(config->halo, descriptor.groupCenterLambdaUm[1], false, halo) ||
            !expand_group(config->bloom, descriptor.groupCenterLambdaUm[2], true, bloom)) {
            fail(diagnostic, "component_expansion");
            return false;
        }
        if (core.count != 2 || halo.count != 3 || bloom.count != 4) {
            fail(diagnostic, "component_count");
            return false;
        }

        std::array<std::array<double, 4>, 3> haloWeightsRgb{};
        if (!halo_channel_weights(halo, descriptor.effectiveWarmth, haloWeightsRgb)) {
            fail(diagnostic, "halo_channel_weights");
            return false;
        }

        const auto lambda_pixels = [&](double lambdaUm) {
            return std::max(
                lambdaUm * descriptor.samplingScale / descriptor.pixelSizeUm,
                kMinimumScale);
        };
        for (std::size_t index = 0; index < core.count; ++index) {
            out.lambdaPixels[index] = lambda_pixels(core.lambdasUm[index]);
            const double weight = descriptor.groupWeightsCoreHaloBloom[0] * core.weights[index];
            out.redWeights[index] = weight;
            out.greenWeights[index] = weight;
            out.blueWeights[index] = weight;
        }
        for (std::size_t index = 0; index < halo.count; ++index) {
            const std::size_t outputIndex = index + core.count;
            out.lambdaPixels[outputIndex] = lambda_pixels(halo.lambdasUm[index]);
            out.redWeights[outputIndex] =
                descriptor.groupWeightsCoreHaloBloom[1] * haloWeightsRgb[0][index];
            out.greenWeights[outputIndex] =
                descriptor.groupWeightsCoreHaloBloom[1] * haloWeightsRgb[1][index];
            out.blueWeights[outputIndex] =
                descriptor.groupWeightsCoreHaloBloom[1] * haloWeightsRgb[2][index];
        }
        for (std::size_t index = 0; index < bloom.count; ++index) {
            const std::size_t outputIndex = index + core.count + halo.count;
            out.lambdaPixels[outputIndex] = lambda_pixels(bloom.lambdasUm[index]);
            const double weight = descriptor.groupWeightsCoreHaloBloom[2] * bloom.weights[index];
            out.redWeights[outputIndex] = weight;
            out.greenWeights[outputIndex] = weight;
            out.blueWeights[outputIndex] = weight;
        }

        if (!finite_component_output(out)) {
            fail(diagnostic, "component_output");
            out = DiffusionPsfComponents{};
            return false;
        }
        return true;
    }

} // namespace Spektrafilm
