#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numbers>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "DiffusionHostBehavior.h"
#include "reference_cases.h"

namespace {

    constexpr double kHostToleranceScale = 5.0e-12;
    constexpr int kLargeFrameExtent = 65536;

    int gFailures = 0;

    double tolerance(double reference) {
        return kHostToleranceScale * std::max(1.0, std::abs(reference));
    }

    void fail_text(std::string_view name, std::string_view field, std::string_view detail) {
        std::cerr << "FAIL case=" << name << " field=" << field << " " << detail << '\n';
        ++gFailures;
    }

    void check_true(bool condition, std::string_view name, std::string_view field) {
        if (!condition) {
            fail_text(name, field, "expected=true actual=false");
        }
    }

    void check_false(bool condition, std::string_view name, std::string_view field) {
        if (condition) {
            fail_text(name, field, "expected=false actual=true");
        }
    }

    void check_equal(
        std::uint64_t actual,
        std::uint64_t reference,
        std::string_view name,
        std::string_view field) {
        if (actual != reference) {
            std::cerr << "FAIL case=" << name << " field=" << field
                      << " expected=" << reference << " actual=" << actual << '\n';
            ++gFailures;
        }
    }

    void check_equal(
        int actual,
        int reference,
        std::string_view name,
        std::string_view field) {
        if (actual != reference) {
            std::cerr << "FAIL case=" << name << " field=" << field
                      << " expected=" << reference << " actual=" << actual << '\n';
            ++gFailures;
        }
    }

    void check_close(
        double actual,
        double reference,
        std::string_view name,
        std::string_view field) {
        const double allowed = tolerance(reference);
        if (!std::isfinite(actual) || std::abs(actual - reference) > allowed) {
            std::cerr.precision(17);
            std::cerr << "FAIL case=" << name << " field=" << field
                      << " expected=" << reference << " actual=" << actual
                      << " tolerance=" << allowed << '\n';
            ++gFailures;
        }
    }

    template <std::size_t N>
    void check_array_close(
        const std::array<double, N>& actual,
        const std::array<double, N>& reference,
        std::string_view name,
        std::string_view field) {
        for (std::size_t index = 0; index < N; ++index) {
            check_close(
                actual[index],
                reference[index],
                name,
                std::string(field) + "[" + std::to_string(index) + "]");
        }
    }

    Spektrafilm::DiffusionFilterAuthoredControls authored_controls(
        const DiffusionHostReference::AuthoredInput& input) {
        Spektrafilm::DiffusionFilterAuthoredControls out{};
        out.active = input.active;
        out.family = static_cast<Spektrafilm::DiffusionFilterFamily>(input.familyIndex);
        out.strength = input.strength;
        out.spatialScale = input.spatialScale;
        out.haloWarmth = input.haloWarmth;
        out.coreIntensity = input.coreIntensity;
        out.coreSize = input.coreSize;
        out.haloIntensity = input.haloIntensity;
        out.haloSize = input.haloSize;
        out.bloomIntensity = input.bloomIntensity;
        out.bloomSize = input.bloomSize;
        return out;
    }

    Spektrafilm::DiffusionFilterAuthoredControls default_authored(std::uint8_t familyIndex = 1) {
        Spektrafilm::DiffusionFilterAuthoredControls out{};
        out.active = true;
        out.family = static_cast<Spektrafilm::DiffusionFilterFamily>(familyIndex);
        return out;
    }

    bool resolve(
        const Spektrafilm::DiffusionFilterAuthoredControls& authored,
        Spektrafilm::DiffusionFilterResolvedParameters& resolved,
        std::string_view name) {
        std::string diagnostic;
        if (!Spektrafilm::resolve_diffusion_filter(authored, resolved, diagnostic)) {
            fail_text(name, "resolve", diagnostic.empty() ? "no diagnostic" : diagnostic);
            return false;
        }
        return true;
    }

    bool describe(
        const Spektrafilm::DiffusionFilterResolvedParameters& resolved,
        double pixelSizeUm,
        int width,
        int height,
        Spektrafilm::DiffusionPsfSampleDescriptor& descriptor,
        std::string_view name) {
        std::string diagnostic;
        if (!Spektrafilm::build_diffusion_psf_sample_descriptor(
                resolved,
                pixelSizeUm,
                width,
                height,
                descriptor,
                diagnostic)) {
            fail_text(name, "sample_descriptor", diagnostic.empty() ? "no diagnostic" : diagnostic);
            return false;
        }
        return true;
    }

    bool expand(
        const Spektrafilm::DiffusionPsfSampleDescriptor& descriptor,
        Spektrafilm::DiffusionPsfComponents& components,
        std::string_view name) {
        std::string diagnostic;
        if (!Spektrafilm::expand_diffusion_psf_components(descriptor, components, diagnostic)) {
            fail_text(name, "components", diagnostic.empty() ? "no diagnostic" : diagnostic);
            return false;
        }
        return true;
    }

    const std::array<double, Spektrafilm::DiffusionPsfComponents::kCount>& channel_weights(
        const Spektrafilm::DiffusionPsfComponents& components,
        int channel) {
        if (channel == 0) {
            return components.redWeights;
        }
        if (channel == 1) {
            return components.greenWeights;
        }
        return components.blueWeights;
    }

    bool resolve_describe_expand(
        const Spektrafilm::DiffusionFilterAuthoredControls& authored,
        double pixelSizeUm,
        int width,
        int height,
        Spektrafilm::DiffusionFilterResolvedParameters& resolved,
        Spektrafilm::DiffusionPsfSampleDescriptor& descriptor,
        Spektrafilm::DiffusionPsfComponents& components,
        std::string_view name) {
        return resolve(authored, resolved, name) &&
               describe(resolved, pixelSizeUm, width, height, descriptor, name) &&
               expand(descriptor, components, name);
    }

    void test_strength_cases() {
        for (const auto& row : DiffusionHostReference::kStrengthCases) {
            Spektrafilm::DiffusionFilterResolvedParameters resolved{};
            if (!resolve(authored_controls(row.authored), resolved, row.name)) {
                continue;
            }
            check_close(resolved.scatterFraction, row.scatterFraction, row.name, "scatterFraction");
            check_true(
                resolved.active == (row.scatterFraction > 0.0),
                row.name,
                "active_identity");
            if (row.scatterFraction > 0.0) {
                check_true(resolved.hash != 0, row.name, "resolved_hash_nonzero");
            } else {
                check_equal(resolved.hash, std::uint64_t{0}, row.name, "resolved_hash_zero");
            }
        }
    }

    void test_family_cases() {
        for (const auto& row : DiffusionHostReference::kFamilyCases) {
            auto authored = default_authored(row.familyIndex);
            Spektrafilm::DiffusionFilterResolvedParameters resolved{};
            Spektrafilm::DiffusionPsfSampleDescriptor descriptor{};
            Spektrafilm::DiffusionPsfComponents components{};
            if (!resolve_describe_expand(
                    authored,
                    1.0,
                    kLargeFrameExtent,
                    kLargeFrameExtent,
                    resolved,
                    descriptor,
                    components,
                    row.name)) {
                continue;
            }
            check_array_close(
                resolved.groupWeightsCoreHaloBloom,
                row.groupWeights,
                row.name,
                "groupWeights");
            check_array_close(
                resolved.groupCenterLambdaUm,
                row.groupCentersUm,
                row.name,
                "groupCentersUm");
            check_close(resolved.effectiveWarmth, row.baseWarmth, row.name, "baseWarmth");
            for (std::size_t index = 0; index < 2; ++index) {
                check_close(components.lambdaPixels[index], row.coreLambdasUm[index], row.name, "coreLambda");
            }
            for (std::size_t index = 0; index < 3; ++index) {
                check_close(components.lambdaPixels[index + 2], row.haloLambdasUm[index], row.name, "haloLambda");
            }
            for (std::size_t index = 0; index < 4; ++index) {
                check_close(components.lambdaPixels[index + 5], row.bloomLambdasUm[index], row.name, "bloomLambda");
            }

            authored.haloWarmth = -row.baseWarmth;
            if (!resolve_describe_expand(
                    authored,
                    1.0,
                    kLargeFrameExtent,
                    kLargeFrameExtent,
                    resolved,
                    descriptor,
                    components,
                    row.name)) {
                continue;
            }
            for (std::size_t index = 0; index < 2; ++index) {
                check_close(
                    components.redWeights[index] / row.groupWeights[0],
                    row.coreWeights[index],
                    row.name,
                    "coreInternalWeight");
            }
            for (std::size_t index = 0; index < 3; ++index) {
                check_close(
                    components.redWeights[index + 2] / row.groupWeights[1],
                    row.haloWeights[index],
                    row.name,
                    "haloInternalWeight");
            }
            for (std::size_t index = 0; index < 4; ++index) {
                check_close(
                    components.redWeights[index + 5] / row.groupWeights[2],
                    row.bloomWeights[index],
                    row.name,
                    "bloomInternalWeight");
            }
        }
    }

    void test_warmth_cases() {
        for (const auto& row : DiffusionHostReference::kWarmthCases) {
            Spektrafilm::DiffusionFilterResolvedParameters resolved{};
            Spektrafilm::DiffusionPsfSampleDescriptor descriptor{};
            Spektrafilm::DiffusionPsfComponents components{};
            if (!resolve_describe_expand(
                    authored_controls(row.authored),
                    1.0,
                    kLargeFrameExtent,
                    kLargeFrameExtent,
                    resolved,
                    descriptor,
                    components,
                    row.name)) {
                continue;
            }
            check_close(resolved.effectiveWarmth, row.effectiveWarmth, row.name, "effectiveWarmth");
            const double haloGroupWeight = resolved.groupWeightsCoreHaloBloom[1];
            for (int channel = 0; channel < 3; ++channel) {
                const auto& weights = channel_weights(components, channel);
                for (std::size_t index = 0; index < 3; ++index) {
                    check_close(
                        weights[index + 2] / haloGroupWeight,
                        row.channelWeightsRgb[static_cast<std::size_t>(channel)][index],
                        row.name,
                        "haloChannelWeight");
                }
            }
        }
    }

    void test_advanced_cases() {
        for (const auto& row : DiffusionHostReference::kAdvancedCases) {
            auto authored = authored_controls(row.authored);
            Spektrafilm::DiffusionFilterResolvedParameters resolved{};
            Spektrafilm::DiffusionPsfSampleDescriptor descriptor{};
            Spektrafilm::DiffusionPsfComponents components{};
            if (!resolve_describe_expand(
                    authored,
                    1.0,
                    kLargeFrameExtent,
                    kLargeFrameExtent,
                    resolved,
                    descriptor,
                    components,
                    row.name)) {
                continue;
            }
            check_array_close(
                resolved.groupWeightsCoreHaloBloom,
                row.groupWeights,
                row.name,
                "groupWeights");
            check_array_close(
                resolved.groupCenterLambdaUm,
                row.groupCentersUm,
                row.name,
                "groupCentersUm");
            check_array_close(components.lambdaPixels, row.lambdasUm, row.name, "lambdaPixels");

            authored.haloWarmth = -row.familyBaseWarmth;
            if (!resolve_describe_expand(
                    authored,
                    1.0,
                    kLargeFrameExtent,
                    kLargeFrameExtent,
                    resolved,
                    descriptor,
                    components,
                    row.name)) {
                continue;
            }
            const auto& groups = row.groupWeights;
            if (groups[0] > 0.0) {
                for (std::size_t index = 0; index < 2; ++index) {
                    check_close(
                        components.redWeights[index] / groups[0],
                        row.coreWeights[index],
                        row.name,
                        "coreInternalWeight");
                }
            }
            if (groups[1] > 0.0) {
                for (std::size_t index = 0; index < 3; ++index) {
                    check_close(
                        components.redWeights[index + 2] / groups[1],
                        row.haloWeights[index],
                        row.name,
                        "haloInternalWeight");
                }
            }
            if (groups[2] > 0.0) {
                for (std::size_t index = 0; index < 4; ++index) {
                    check_close(
                        components.redWeights[index + 5] / groups[2],
                        row.bloomWeights[index],
                        row.name,
                        "bloomInternalWeight");
                }
            }
        }
    }

    void test_radius_cases() {
        for (const auto& row : DiffusionHostReference::kRadiusCases) {
            Spektrafilm::DiffusionFilterResolvedParameters resolved{};
            Spektrafilm::DiffusionPsfSampleDescriptor descriptor{};
            if (!resolve(authored_controls(row.authored), resolved, row.name) ||
                !describe(
                    resolved,
                    row.pixelSizeUm,
                    row.width,
                    row.height,
                    descriptor,
                    row.name)) {
                continue;
            }
            check_equal(descriptor.radiusPixels, row.radiusPixels, row.name, "radiusPixels");
        }
    }

    std::vector<double> sample_psf(
        const Spektrafilm::DiffusionPsfComponents& components,
        int height,
        int width) {
        std::vector<double> values(
            static_cast<std::size_t>(height) * static_cast<std::size_t>(width) * 3,
            0.0);
        std::array<double, 3> channelSums{};
        const int centerY = height / 2;
        const int centerX = width / 2;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const double dx = static_cast<double>(x - centerX);
                const double dy = static_cast<double>(y - centerY);
                const double radius = std::sqrt(dx * dx + dy * dy);
                for (int channel = 0; channel < 3; ++channel) {
                    const auto& weights = channel_weights(components, channel);
                    double sample = 0.0;
                    for (std::size_t component = 0;
                         component < Spektrafilm::DiffusionPsfComponents::kCount;
                         ++component) {
                        const double lambda = components.lambdaPixels[component];
                        sample += weights[component] * std::exp(-radius / lambda) /
                                  (2.0 * std::numbers::pi_v<double> * lambda * lambda);
                    }
                    const std::size_t offset =
                        (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                         static_cast<std::size_t>(x)) *
                            3 +
                        static_cast<std::size_t>(channel);
                    values[offset] = sample;
                    channelSums[static_cast<std::size_t>(channel)] += sample;
                }
            }
        }
        for (std::size_t offset = 0; offset < values.size(); offset += 3) {
            for (std::size_t channel = 0; channel < 3; ++channel) {
                values[offset + channel] /= channelSums[channel];
            }
        }
        return values;
    }

    void test_psf_cases() {
        for (const auto& row : DiffusionHostReference::kPsfCases) {
            Spektrafilm::DiffusionFilterResolvedParameters resolved{};
            Spektrafilm::DiffusionPsfSampleDescriptor descriptor{};
            Spektrafilm::DiffusionPsfComponents components{};
            if (!resolve_describe_expand(
                    authored_controls(row.authored),
                    row.pixelSizeUm,
                    kLargeFrameExtent,
                    kLargeFrameExtent,
                    resolved,
                    descriptor,
                    components,
                    row.name)) {
                continue;
            }
            const std::vector<double> actual = sample_psf(components, row.height, row.width);
            for (std::size_t offset = 0; offset < actual.size(); ++offset) {
                check_close(
                    actual[offset],
                    DiffusionHostReference::kPsfValues[row.valueOffset + offset],
                    row.name,
                    "psf");
            }
        }
    }

    void test_spatial_scale_floor_contract() {
        constexpr double kRawBelowFloor = 5.0e-7;
        constexpr double kFloor = 1.0e-6;
        constexpr double kTinyPixelSizeUm = 1.0e-6;
        constexpr int kScaleFloorFrameExtent = 1'000'000;

        auto belowFloorAuthored = default_authored();
        belowFloorAuthored.spatialScale = kRawBelowFloor;
        Spektrafilm::DiffusionFilterResolvedParameters belowFloorResolved{};
        Spektrafilm::DiffusionPsfSampleDescriptor belowFloorDescriptor{};
        Spektrafilm::DiffusionPsfComponents belowFloorComponents{};
        if (!resolve_describe_expand(
                belowFloorAuthored,
                kTinyPixelSizeUm,
                kScaleFloorFrameExtent,
                kScaleFloorFrameExtent,
                belowFloorResolved,
                belowFloorDescriptor,
                belowFloorComponents,
                "spatial-scale:below-floor")) {
            return;
        }

        auto atFloorAuthored = default_authored();
        atFloorAuthored.spatialScale = kFloor;
        Spektrafilm::DiffusionFilterResolvedParameters atFloorResolved{};
        Spektrafilm::DiffusionPsfSampleDescriptor atFloorDescriptor{};
        Spektrafilm::DiffusionPsfComponents atFloorComponents{};
        if (!resolve_describe_expand(
                atFloorAuthored,
                kTinyPixelSizeUm,
                kScaleFloorFrameExtent,
                kScaleFloorFrameExtent,
                atFloorResolved,
                atFloorDescriptor,
                atFloorComponents,
                "spatial-scale:at-floor")) {
            return;
        }

        check_close(
            belowFloorResolved.spatialScale,
            kRawBelowFloor,
            "spatial-scale:below-floor",
            "resolvedRaw");
        check_close(
            atFloorResolved.spatialScale,
            kFloor,
            "spatial-scale:at-floor",
            "resolvedRaw");
        check_close(
            belowFloorDescriptor.spatialScale,
            kRawBelowFloor,
            "spatial-scale:below-floor",
            "descriptorRaw");
        check_close(
            atFloorDescriptor.spatialScale,
            kFloor,
            "spatial-scale:at-floor",
            "descriptorRaw");
        check_true(
            belowFloorDescriptor.radiusPixels < atFloorDescriptor.radiusPixels,
            "spatial-scale:below-floor",
            "rawRadiusIsNotFloored");
        check_array_close(
            belowFloorComponents.lambdaPixels,
            atFloorComponents.lambdaPixels,
            "spatial-scale:floor",
            "componentWidthsUseFloor");
        check_true(
            belowFloorResolved.hash != atFloorResolved.hash,
            "spatial-scale:floor",
            "resolvedRawIdentity");
        check_true(
            belowFloorDescriptor.hash != atFloorDescriptor.hash,
            "spatial-scale:floor",
            "sampleRawIdentity");

        Spektrafilm::DiffusionFilterResolvedParameters ordinaryResolved{};
        Spektrafilm::DiffusionPsfSampleDescriptor ordinaryDescriptor{};
        Spektrafilm::DiffusionPsfComponents ordinaryComponents{};
        if (resolve_describe_expand(
                default_authored(),
                1.0,
                kLargeFrameExtent,
                kLargeFrameExtent,
                ordinaryResolved,
                ordinaryDescriptor,
                ordinaryComponents,
                "spatial-scale:ordinary")) {
            check_close(
                ordinaryResolved.spatialScale,
                1.0,
                "spatial-scale:ordinary",
                "resolvedRaw");
            check_close(
                ordinaryDescriptor.spatialScale,
                1.0,
                "spatial-scale:ordinary",
                "descriptorRaw");
        }
    }

    Spektrafilm::DiffusionFilterResolvedParameters resolved_for_hash(
        const Spektrafilm::DiffusionFilterAuthoredControls& authored,
        std::string_view name) {
        Spektrafilm::DiffusionFilterResolvedParameters resolved{};
        resolve(authored, resolved, name);
        return resolved;
    }

    Spektrafilm::DiffusionPsfSampleDescriptor descriptor_for_hash(
        const Spektrafilm::DiffusionFilterAuthoredControls& authored,
        double pixelSizeUm,
        int width,
        int height,
        std::string_view name) {
        auto resolved = resolved_for_hash(authored, name);
        Spektrafilm::DiffusionPsfSampleDescriptor descriptor{};
        describe(resolved, pixelSizeUm, width, height, descriptor, name);
        return descriptor;
    }

    void test_authored_field_hash_coverage() {
        using Controls = Spektrafilm::DiffusionFilterAuthoredControls;
        using NumericField = double Controls::*;
        const std::array<std::pair<NumericField, std::string_view>, 9> numericFields{{{&Controls::strength, "strength"},
                                                                                      {&Controls::spatialScale, "spatialScale"},
                                                                                      {&Controls::haloWarmth, "haloWarmth"},
                                                                                      {&Controls::coreIntensity, "coreIntensity"},
                                                                                      {&Controls::coreSize, "coreSize"},
                                                                                      {&Controls::haloIntensity, "haloIntensity"},
                                                                                      {&Controls::haloSize, "haloSize"},
                                                                                      {&Controls::bloomIntensity, "bloomIntensity"},
                                                                                      {&Controls::bloomSize, "bloomSize"}}};

        const auto verify_stage = [&](std::string_view stage) {
            const Controls base = default_authored();
            const std::uint64_t baseHash =
                Spektrafilm::hash_diffusion_authored_controls(base);
            const auto expect_hash_change = [&](const Controls& changed, std::string_view field) {
                check_true(
                    Spektrafilm::hash_diffusion_authored_controls(changed) != baseHash,
                    std::string("authored-fields:") + std::string(stage),
                    field);
            };

            Controls changed = base;
            changed.active = false;
            expect_hash_change(changed, "active");
            changed = base;
            changed.family = Spektrafilm::DiffusionFilterFamily::ProMist;
            expect_hash_change(changed, "family");
            for (const auto& [field, name] : numericFields) {
                changed = base;
                changed.*field += 1.0e-6;
                expect_hash_change(changed, name);
            }

            Controls zeroControls{};
            for (const auto& [field, name] : numericFields) {
                zeroControls.*field = 0.0;
                Controls negativeZero = zeroControls;
                negativeZero.*field = -0.0;
                check_equal(
                    Spektrafilm::hash_diffusion_authored_controls(zeroControls),
                    Spektrafilm::hash_diffusion_authored_controls(negativeZero),
                    std::string("authored-zero:") + std::string(stage),
                    name);
            }

            Controls inactive = base;
            inactive.active = false;
            Controls inactiveAdvancedEdit = inactive;
            inactiveAdvancedEdit.bloomSize += 1.0e-6;
            check_true(
                Spektrafilm::hash_diffusion_authored_controls(inactive) !=
                    Spektrafilm::hash_diffusion_authored_controls(inactiveAdvancedEdit),
                std::string("authored-inactive:") + std::string(stage),
                "advancedEdit");

            check_true(
                baseHash != Spektrafilm::hash_diffusion_authored_controls(inactive),
                std::string("authored-active:") + std::string(stage),
                "activeToInactive");
            Spektrafilm::DiffusionFilterResolvedParameters resolved{};
            if (resolve(inactive, resolved, "authored-active-to-inactive")) {
                check_false(resolved.active, "authored-active-to-inactive", stage);
                check_equal(
                    resolved.hash,
                    std::uint64_t{0},
                    "authored-active-to-inactive",
                    stage);
            }
        };

        verify_stage("camera");
        verify_stage("enlarger");
    }

    void test_hash_relationships() {
        auto base = default_authored();

        auto negativeZero = base;
        negativeZero.haloWarmth = -0.0;
        check_equal(
            Spektrafilm::hash_diffusion_authored_controls(base),
            Spektrafilm::hash_diffusion_authored_controls(negativeZero),
            "hash:-zero",
            "authoredHash");

        auto tinyEdit = base;
        tinyEdit.strength += 1.0e-6;
        check_true(
            Spektrafilm::hash_diffusion_authored_controls(base) !=
                Spektrafilm::hash_diffusion_authored_controls(tinyEdit),
            "hash:sub-1e-4",
            "authoredHash");

        auto proportional = base;
        proportional.coreIntensity = 2.0;
        proportional.haloIntensity = 2.0;
        proportional.bloomIntensity = 2.0;
        check_equal(
            resolved_for_hash(base, "hash:proportional").hash,
            resolved_for_hash(proportional, "hash:proportional").hash,
            "hash:proportional",
            "resolvedHash");

        auto allZero = base;
        allZero.coreIntensity = 0.0;
        allZero.haloIntensity = 0.0;
        allZero.bloomIntensity = 0.0;
        allZero.coreSize = 0.25;
        allZero.haloSize = 2.0;
        allZero.bloomSize = 4.0;
        check_equal(
            resolved_for_hash(base, "hash:all-zero").hash,
            resolved_for_hash(allZero, "hash:all-zero").hash,
            "hash:all-zero",
            "resolvedHash");

        auto strengthTwo = base;
        strengthTwo.strength = 2.0;
        auto strengthFour = base;
        strengthFour.strength = 4.0;
        check_equal(
            resolved_for_hash(strengthTwo, "hash:endpoint").hash,
            resolved_for_hash(strengthFour, "hash:endpoint").hash,
            "hash:endpoint",
            "resolvedHash");

        auto strengthOne = base;
        strengthOne.strength = 1.0;
        check_true(
            resolved_for_hash(base, "hash:strength").hash !=
                resolved_for_hash(strengthOne, "hash:strength").hash,
            "hash:strength",
            "resolvedHashChanges");
        check_equal(
            descriptor_for_hash(base, 1000.0, 1920, 1080, "hash:strength").hash,
            descriptor_for_hash(strengthOne, 1000.0, 1920, 1080, "hash:strength").hash,
            "hash:strength",
            "sampleHashReused");

        const auto baseDescriptor =
            descriptor_for_hash(base, 1000.0, 1920, 1080, "hash:sample-fields");
        const auto expect_sample_hash_change = [&](
                                                   const auto& authored,
                                                   double pixelSize,
                                                   int width,
                                                   int height,
                                                   std::string_view field) {
            const auto changed = descriptor_for_hash(
                authored,
                pixelSize,
                width,
                height,
                "hash:sample-fields");
            check_true(
                changed.hash != baseDescriptor.hash,
                "hash:sample-fields",
                field);
        };

        expect_sample_hash_change(base, 900.0, 1920, 1080, "pixelSize");
        expect_sample_hash_change(base, 1000.0, 11, 11, "radius");
        auto changedWarmth = base;
        changedWarmth.haloWarmth = 0.2;
        expect_sample_hash_change(changedWarmth, 1000.0, 1920, 1080, "warmth");
        auto changedWeights = base;
        changedWeights.coreIntensity = 2.0;
        expect_sample_hash_change(changedWeights, 1000.0, 1920, 1080, "weights");
        auto changedCenters = base;
        changedCenters.coreSize = 1.1;
        expect_sample_hash_change(changedCenters, 1000.0, 1920, 1080, "centers");
        auto changedScale = base;
        changedScale.spatialScale = 1.1;
        expect_sample_hash_change(changedScale, 1000.0, 1920, 1080, "spatialScale");
    }

    void test_invalid_and_identity_inputs() {
        auto authored = default_authored();
        authored.family = static_cast<Spektrafilm::DiffusionFilterFamily>(255);
        Spektrafilm::DiffusionFilterResolvedParameters resolved{};
        std::string diagnostic;
        check_false(
            Spektrafilm::resolve_diffusion_filter(authored, resolved, diagnostic),
            "invalid:family",
            "resolve");

        const std::array<double Spektrafilm::DiffusionFilterAuthoredControls::*, 9> fields{{&Spektrafilm::DiffusionFilterAuthoredControls::strength,
                                                                                            &Spektrafilm::DiffusionFilterAuthoredControls::spatialScale,
                                                                                            &Spektrafilm::DiffusionFilterAuthoredControls::haloWarmth,
                                                                                            &Spektrafilm::DiffusionFilterAuthoredControls::coreIntensity,
                                                                                            &Spektrafilm::DiffusionFilterAuthoredControls::coreSize,
                                                                                            &Spektrafilm::DiffusionFilterAuthoredControls::haloIntensity,
                                                                                            &Spektrafilm::DiffusionFilterAuthoredControls::haloSize,
                                                                                            &Spektrafilm::DiffusionFilterAuthoredControls::bloomIntensity,
                                                                                            &Spektrafilm::DiffusionFilterAuthoredControls::bloomSize}};
        const std::array<double, 3> invalidValues{{std::numeric_limits<double>::quiet_NaN(),
                                                   std::numeric_limits<double>::infinity(),
                                                   -std::numeric_limits<double>::infinity()}};
        for (std::size_t field = 0; field < fields.size(); ++field) {
            for (std::size_t value = 0; value < invalidValues.size(); ++value) {
                authored = default_authored();
                authored.*fields[field] = invalidValues[value];
                diagnostic.clear();
                check_false(
                    Spektrafilm::resolve_diffusion_filter(authored, resolved, diagnostic),
                    "invalid:numeric",
                    std::string("field=") + std::to_string(field) +
                        " value=" + std::to_string(value));
            }
        }

        const auto check_identity = [&](
                                        Spektrafilm::DiffusionFilterAuthoredControls identity,
                                        std::string_view name) {
            Spektrafilm::DiffusionFilterResolvedParameters actual{};
            if (resolve(identity, actual, name)) {
                check_false(actual.active, name, "active");
                check_equal(actual.hash, std::uint64_t{0}, name, "hash");
            }
        };
        authored = default_authored();
        authored.active = false;
        check_identity(authored, "identity:inactive");
        authored = default_authored();
        authored.strength = 0.0;
        check_identity(authored, "identity:strength-zero");
        authored = default_authored();
        authored.strength = -1.0;
        check_identity(authored, "identity:strength-negative");
        authored = default_authored();
        authored.spatialScale = 0.0;
        check_identity(authored, "identity:scale-zero");
        authored = default_authored();
        authored.spatialScale = -1.0;
        check_identity(authored, "identity:scale-negative");

        authored = default_authored();
        if (resolve(authored, resolved, "invalid:descriptor")) {
            Spektrafilm::DiffusionPsfSampleDescriptor descriptor{};
            diagnostic.clear();
            check_false(
                Spektrafilm::build_diffusion_psf_sample_descriptor(
                    resolved,
                    0.0,
                    1920,
                    1080,
                    descriptor,
                    diagnostic),
                "invalid:descriptor",
                "pixelSizeZero");
            check_false(
                Spektrafilm::build_diffusion_psf_sample_descriptor(
                    resolved,
                    1.0,
                    0,
                    1080,
                    descriptor,
                    diagnostic),
                "invalid:descriptor",
                "widthZero");
            check_false(
                Spektrafilm::build_diffusion_psf_sample_descriptor(
                    resolved,
                    1.0,
                    1920,
                    -1,
                    descriptor,
                    diagnostic),
                "invalid:descriptor",
                "heightNegative");
        }

        Spektrafilm::DiffusionPsfSampleDescriptor inactiveDescriptor{};
        Spektrafilm::DiffusionPsfComponents components{};
        diagnostic.clear();
        check_false(
            Spektrafilm::expand_diffusion_psf_components(
                inactiveDescriptor,
                components,
                diagnostic),
            "invalid:components",
            "inactiveDescriptor");
    }

} // namespace

int main() {
    test_strength_cases();
    test_family_cases();
    test_warmth_cases();
    test_advanced_cases();
    test_radius_cases();
    test_psf_cases();
    test_spatial_scale_floor_contract();
    test_authored_field_hash_coverage();
    test_hash_relationships();
    test_invalid_and_identity_inputs();

    if (gFailures != 0) {
        std::cerr << "HOST_BEHAVIOR_CONTRACT=FAIL failures=" << gFailures << '\n';
        return 1;
    }
    std::cout << "HOST_BEHAVIOR_CONTRACT=PASS"
              << " strength=" << DiffusionHostReference::kStrengthCases.size()
              << " families=" << DiffusionHostReference::kFamilyCases.size()
              << " warmth=" << DiffusionHostReference::kWarmthCases.size()
              << " advanced=" << DiffusionHostReference::kAdvancedCases.size()
              << " authored_fields=22"
              << " radius=" << DiffusionHostReference::kRadiusCases.size()
              << " psf=" << DiffusionHostReference::kPsfCases.size()
              << '\n';
    return 0;
}
