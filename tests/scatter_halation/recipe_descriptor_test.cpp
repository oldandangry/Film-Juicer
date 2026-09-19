#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include "gtest/gtest.h"

#include "ProcessRoot.h"
#include "ProfileCatalog.h"
#include "ScatterHalation.h"
#include "SpectralData.h"

namespace {

    std::uint32_t float_bits(float value) {
        return std::bit_cast<std::uint32_t>(value);
    }

    Profiles::ProfileDigest profile_digest(
        float sigma,
        std::array<float, 3> strength) {
        Profiles::ProfileDigest digest;
        digest.halationFirstSigmaUm = {sigma, sigma, sigma};
        digest.halationPrimaryAmount = strength;
        return digest;
    }

    ScatterHalationControls controls_or_fail(
        const ScatterHalationRawControls& raw) {
        ScatterHalationControls controls;
        std::string diagnostic;
        EXPECT_TRUE(Spektrafilm::build_scatter_halation_controls(
            raw,
            controls,
            diagnostic))
            << diagnostic;
        return controls;
    }

    ScatterHalationOpticsRecipe recipe_or_fail(
        const ScatterHalationControls& controls,
        const Profiles::ProfileDigest& digest) {
        ScatterHalationOpticsRecipe recipe;
        std::string diagnostic;
        EXPECT_TRUE(Spektrafilm::resolve_scatter_halation_recipe(
            controls,
            digest,
            recipe,
            diagnostic))
            << diagnostic;
        return recipe;
    }

    TEST(ScatterHalationControls, RetainsFloat32BoundsAndIdentity) {
        for (double value : {0.0, 1.0, 1.25, 2.0}) {
            SCOPED_TRACE(value);
            ScatterHalationRawControls raw{true, value, value, value, value};
            ScatterHalationControls controls;
            std::string diagnostic;
            ASSERT_TRUE(Spektrafilm::build_scatter_halation_controls(
                raw,
                controls,
                diagnostic))
                << diagnostic;
            const std::uint32_t expected =
                value == 0.0 ? 0u : float_bits(static_cast<float>(value));
            EXPECT_TRUE(controls.active);
            EXPECT_EQ(float_bits(controls.scatterAmount), expected);
            EXPECT_EQ(float_bits(controls.scatterSpatialScale), expected);
            EXPECT_EQ(float_bits(controls.halationAmount), expected);
            EXPECT_EQ(float_bits(controls.halationSpatialScale), expected);
        }

        const float retained = 1.0f;
        const double sameFloat = std::nextafter(
            static_cast<double>(retained),
            static_cast<double>(std::nextafter(retained, 2.0f)));
        const auto first = controls_or_fail({true, 1.0, 1.0, 1.0, 1.0});
        const auto same = controls_or_fail({true, sameFloat, 1.0, 1.0, 1.0});
        const auto distinct = controls_or_fail(
            {true,
             static_cast<double>(std::nextafter(retained, 2.0f)),
             1.0,
             1.0,
             1.0});
        const auto digest = profile_digest(65.0f, {0.08f, 0.02f, 0.0f});
        const auto firstRecipe = recipe_or_fail(first, digest);
        const auto sameRecipe = recipe_or_fail(same, digest);
        const auto distinctRecipe = recipe_or_fail(distinct, digest);
        EXPECT_EQ(firstRecipe.hash, sameRecipe.hash);
        EXPECT_NE(firstRecipe.hash, distinctRecipe.hash);
    }

    TEST(ScatterHalationControls, RejectsInvalidActiveValues) {
        for (double value : {
                 std::numeric_limits<double>::quiet_NaN(),
                 std::numeric_limits<double>::infinity(),
                 -std::numeric_limits<double>::infinity(),
                 -0.01,
                 std::nextafter(2.0, 3.0)}) {
            SCOPED_TRACE(value);
            ScatterHalationRawControls raw;
            raw.active = true;
            raw.scatterAmount = value;
            ScatterHalationControls controls;
            std::string diagnostic;
            EXPECT_FALSE(Spektrafilm::build_scatter_halation_controls(
                raw,
                controls,
                diagnostic));
            EXPECT_FALSE(diagnostic.empty());
        }

        ScatterHalationRawControls inactive;
        inactive.scatterAmount = std::numeric_limits<double>::quiet_NaN();
        inactive.scatterSpatialScale = -1.0;
        inactive.halationAmount = std::numeric_limits<double>::infinity();
        inactive.halationSpatialScale = std::nextafter(2.0, 3.0);
        ScatterHalationControls controls;
        std::string diagnostic;
        ASSERT_TRUE(Spektrafilm::build_scatter_halation_controls(
            inactive,
            controls,
            diagnostic))
            << diagnostic;
        EXPECT_FALSE(controls.active);
        EXPECT_EQ(float_bits(controls.scatterAmount), 0u);
        EXPECT_EQ(float_bits(controls.scatterSpatialScale), 0u);
        EXPECT_EQ(float_bits(controls.halationAmount), 0u);
        EXPECT_EQ(float_bits(controls.halationSpatialScale), 0u);
    }

    TEST(ScatterHalationRecipe, ResolvesProfileMappingsAndZeroWork) {
        struct Mapping {
            float sigma;
            std::array<float, 3> strength;
        };
        constexpr std::array<Mapping, 6> mappings{{{65.0f, {0.015f, 0.005f, 0.0f}},
                                                   {65.0f, {0.08f, 0.02f, 0.0f}},
                                                   {65.0f, {0.3f, 0.1f, 0.015f}},
                                                   {50.0f, {0.015f, 0.005f, 0.0f}},
                                                   {50.0f, {0.08f, 0.02f, 0.0f}},
                                                   {50.0f, {0.3f, 0.1f, 0.015f}}}};
        const auto active = controls_or_fail({true, 1.0, 1.0, 1.0, 1.0});
        for (const Mapping& mapping : mappings) {
            SCOPED_TRACE(mapping.sigma);
            const auto digest = profile_digest(mapping.sigma, mapping.strength);
            const auto recipe = recipe_or_fail(active, digest);
            EXPECT_TRUE(recipe.scatterActive);
            EXPECT_TRUE(recipe.backReflectionActive);
            EXPECT_NE(recipe.hash, 0u);
            EXPECT_EQ(recipe.halationFirstSigmaUm, digest.halationFirstSigmaUm);
            for (std::size_t channel = 0; channel < mapping.strength.size(); ++channel) {
                EXPECT_EQ(
                    float_bits(recipe.totalStrength[channel]),
                    float_bits(mapping.strength[channel]));
            }
        }

        const ScatterHalationControls inactive;
        const auto identity = recipe_or_fail(
            inactive,
            profile_digest(65.0f, {0.08f, 0.02f, 0.0f}));
        EXPECT_FALSE(identity.scatterActive);
        EXPECT_FALSE(identity.backReflectionActive);
        EXPECT_EQ(identity.hash, 0u);
    }

    TEST(ScatterHalationRecipe, KeepsScatterAndReflectionIndependent) {
        const auto digest = profile_digest(65.0f, {0.08f, 0.02f, 0.0f});
        const auto scatterOnly = recipe_or_fail(
            controls_or_fail({true, 1.0, 1.0, 0.0, 1.0}),
            digest);
        EXPECT_TRUE(scatterOnly.scatterActive);
        EXPECT_FALSE(scatterOnly.backReflectionActive);
        EXPECT_NE(scatterOnly.hash, 0u);

        const auto reflectionOnly = recipe_or_fail(
            controls_or_fail({true, 0.0, 1.0, 1.0, 1.0}),
            digest);
        EXPECT_FALSE(reflectionOnly.scatterActive);
        EXPECT_TRUE(reflectionOnly.backReflectionActive);
        EXPECT_NE(reflectionOnly.hash, 0u);
        EXPECT_NE(scatterOnly.hash, reflectionOnly.hash);
    }

    TEST(ScatterHalationDescriptor, BindsRecipeAndSelectsReferenceOperators) {
        const auto recipe = recipe_or_fail(
            controls_or_fail({true, 1.0, 1.0, 1.0, 1.0}),
            profile_digest(65.0f, {0.08f, 0.02f, 0.0f}));
        std::optional<ScatterHalationFrameDescriptor> descriptor;
        std::string diagnostic;
        ASSERT_TRUE(Spektrafilm::build_scatter_halation_frame_descriptor(
            recipe,
            1.0f,
            descriptor,
            diagnostic))
            << diagnostic;
        ASSERT_TRUE(descriptor);
        EXPECT_EQ(descriptor->recipeHash, recipe.hash);
        EXPECT_EQ(
            descriptor->channels[0].core.kind,
            ScatterHalationGaussianKind::FirReflect);
        EXPECT_EQ(
            descriptor->channels[0].tail[0].kind,
            ScatterHalationGaussianKind::YvvReplicate);
        EXPECT_EQ(
            descriptor->channels[0].bounce[0].kind,
            ScatterHalationGaussianKind::YvvReplicate);

        ScatterHalationOpticsRecipe identity;
        descriptor.emplace();
        diagnostic.clear();
        EXPECT_TRUE(Spektrafilm::build_scatter_halation_frame_descriptor(
            identity,
            std::numeric_limits<float>::quiet_NaN(),
            descriptor,
            diagnostic))
            << diagnostic;
        EXPECT_FALSE(descriptor);
    }

    TEST(ScatterHalationBootstrap, LoadsPublishedProfilesWithoutCuda) {
        JuicerProcess::root().ensure_bootstrap();
        const Spektrafilm::ProfileCatalog& catalog =
            JuicerProcess::root().assets().spektrafilm_profile_catalog();
        ASSERT_TRUE(catalog.valid) << catalog.failure;
        EXPECT_TRUE(catalog.defaultFilmPresent);
        EXPECT_TRUE(catalog.defaultPrintPresent);

        const auto profile = JuicerProcess::root().assets().selected_film_profile_for_key(
            Spektrafilm::kDefaultFilmProfileKey);
        ASSERT_TRUE(profile);
        EXPECT_EQ(profile->info.stock, Spektrafilm::kDefaultFilmProfileKey);
        EXPECT_EQ(profile->data.wavelengths.front(), 380.0f);
        EXPECT_EQ(profile->data.wavelengths.back(), 780.0f);
        EXPECT_TRUE(Spectral::spectral_shape_matches_reference(Spectral::gShape));
        EXPECT_TRUE(Spectral::hanatos_available());
        EXPECT_TRUE(Spectral::mallett_available());
    }

} // namespace

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    JuicerProcess::shutdown_if_initialized();
    return result;
}
