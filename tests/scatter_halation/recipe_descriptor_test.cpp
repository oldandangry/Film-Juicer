#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "gtest_cases.h"
#include "gtest/gtest.h"

#include "FilmEffectsFrameDescriptors.h"
#include "../ffi/owner_fixture.h"
#include "ScatterHalation.h"
#include "spatial_optics_fixture.h"
#include "SpectralData.h"
#include "juicer_test_api.h"

namespace {

    class ScatterHalationBootstrapTest : public testing::Test {
    protected:
        static void SetUpTestSuite() {
            runtime.create(JUICER_TEST_RESOURCE_DIR);
        }

        static void TearDownTestSuite() {
            EXPECT_EQ(runtime.close().category, FJ_STATUS_SUCCESS);
        }

        static JuicerTestSupport::RuntimeFixture runtime;
    };

    JuicerTestSupport::RuntimeFixture ScatterHalationBootstrapTest::runtime;

    std::uint32_t float_bits(float value) {
        return std::bit_cast<std::uint32_t>(value);
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
        const std::array<float, 3>& firstSigmaUm,
        const std::array<float, 3>& primaryAmount) {
        ScatterHalationOpticsRecipe recipe;
        std::string diagnostic;
        EXPECT_TRUE(JuicerTestSupport::prepare_scatter_halation(
            controls,
            firstSigmaUm,
            primaryAmount,
            recipe,
            diagnostic))
            << diagnostic;
        return recipe;
    }

    Spektrafilm::FilmJuicerEffectsRecipe effects_recipe_for_test(
        int activationMask) {
        Spektrafilm::FilmJuicerEffectsRecipe recipe{};
        recipe.active = true;
        recipe.hash = 0x1234u;
        if ((activationMask & 1) != 0) {
            recipe.filmDust.cellWidthMm = 0.5f;
            recipe.filmDust.cellHeightMm = 0.5f;
            recipe.filmDust.slotProbability = 0.01f;
            recipe.filmDust.softnessMaxMm = 0.003f;
            recipe.filmDust.supportXMm = 0.5f;
            recipe.filmDust.supportYMm = 0.5f;
        }
        if ((activationMask & 2) != 0) {
            recipe.filmScratch.cellWidthMm = 1.0f;
            recipe.filmScratch.cellHeightMm = 8.0f;
            recipe.filmScratch.slotProbability = 0.01f;
            recipe.filmScratch.softnessMaxMm = 0.001f;
            recipe.filmScratch.supportXMm = 1.0f;
            recipe.filmScratch.supportYMm = 12.0f;
        }
        if ((activationMask & 4) != 0) {
            recipe.gateDust.cellWidthMm = 0.5f;
            recipe.gateDust.cellHeightMm = 0.5f;
            recipe.gateDust.slotProbability = 0.01f;
            recipe.gateDust.softnessMaxMm = 0.005f;
            recipe.gateDust.supportXMm = 0.5f;
            recipe.gateDust.supportYMm = 0.5f;
        }
        if ((activationMask & 8) != 0) {
            recipe.gateScratch.cellWidthMm = 1.0f;
            recipe.gateScratch.cellHeightMm = 8.0f;
            recipe.gateScratch.slotProbability = 0.01f;
            recipe.gateScratch.softnessMaxMm = 0.003f;
            recipe.gateScratch.supportXMm = 1.0f;
            recipe.gateScratch.supportYMm = 12.0f;
        }
        if ((activationMask & 16) != 0) {
            recipe.gateWeaveAmount = 1.0;
        }
        return recipe;
    }

    Spektrafilm::FilmJuicerEffectsFrameDescriptorInput effects_input_for_test(
        const Spektrafilm::FilmJuicerEffectsRecipe& recipe,
        Spektrafilm::FilmJuicerEffectsFrameExtent renderExtent = {0, 0, 640, 480},
        Spektrafilm::FilmJuicerEffectsFrameExtent fullFrameExtent = {0, 0, 640, 480}) {
        Spektrafilm::FilmJuicerEffectsGeometry geometry;
        geometry.pixelDefinition = fullFrameExtent;
        geometry.canonicalWidth = 35.0;
        geometry.canonicalHeight = 20.0;
        geometry.scaleX = 1.0;
        geometry.scaleY = 1.0;
        geometry.pixelAspectRatio = 1.0;
        return {&recipe,
                renderExtent,
                fullFrameExtent,
                10.0f,
                0.0,
                24.0,
                1u,
                2u,
                geometry,
                35.0f};
    }

    bool build_effects_descriptor_for_test(
        const Spektrafilm::FilmJuicerEffectsFrameDescriptorInput& input,
        Spektrafilm::FilmJuicerEffectsFrameDescriptor& descriptor) {
        std::string diagnostic;
        return JuicerTestSupport::prepare_effects_frame(
            input,
            descriptor,
            diagnostic);
    }

    TEST(FilmEffectsDescriptor, ConstructsEachActivationAndAllOffIdentity) {
        Spektrafilm::FilmJuicerEffectsRecipe identity{};
        auto input = effects_input_for_test(identity);
        Spektrafilm::FilmJuicerEffectsFrameDescriptor descriptor;
        ASSERT_TRUE(build_effects_descriptor_for_test(input, descriptor));
        EXPECT_EQ(descriptor.hash, 0u);
        EXPECT_EQ(descriptor.recipeHash, 0u);
        EXPECT_FALSE(descriptor.filmActive);
        EXPECT_FALSE(descriptor.gateOutputActive);

        JuicerTests::GtestCohort activation_masks_cases("activation-masks", 5u);
        std::size_t activation_masks_index = 0;
        for (const int activationMask : {1, 2, 4, 8, 16}) {
            auto activation_masks_case = activation_masks_cases.case_event(std::to_string(activation_masks_index++));
            SCOPED_TRACE(activationMask);
            const auto recipe = effects_recipe_for_test(activationMask);
            input = effects_input_for_test(recipe);
            ASSERT_TRUE(build_effects_descriptor_for_test(input, descriptor));
            EXPECT_NE(descriptor.hash, 0u);
            EXPECT_EQ(descriptor.recipeHash, recipe.hash);
            EXPECT_EQ(descriptor.filmActive, (activationMask & 3) != 0);
            EXPECT_EQ(descriptor.gateTransmittanceActive, (activationMask & 12) != 0);
            EXPECT_EQ(descriptor.weaveActive, (activationMask & 16) != 0);
            EXPECT_EQ(
                descriptor.gateOutputActive,
                (activationMask & 28) != 0);
            EXPECT_EQ(descriptor.requiresFullFrame, (activationMask & 16) != 0);

            activation_masks_case.passed();
        }
    }

    TEST(FilmEffectsDescriptor, KeepsGeometryAndPhysicalStepRejectionAtConstruction) {
        const auto recipe = effects_recipe_for_test(1);
        Spektrafilm::FilmJuicerEffectsFrameDescriptor descriptor;

        auto contained = effects_input_for_test(recipe, {32, 24, 320, 240});
        ASSERT_TRUE(build_effects_descriptor_for_test(contained, descriptor));
        EXPECT_EQ(descriptor.roiOffsetX, 32);
        EXPECT_EQ(descriptor.roiOffsetY, 24);

        const auto outside = effects_input_for_test(recipe, {-1, 0, 320, 240});
        EXPECT_FALSE(build_effects_descriptor_for_test(outside, descriptor));

        const auto weaveRecipe = effects_recipe_for_test(16);
        const auto partialWeave = effects_input_for_test(weaveRecipe, {32, 24, 320, 240});
        EXPECT_FALSE(build_effects_descriptor_for_test(partialWeave, descriptor));

        auto extremeExtent = effects_input_for_test(
            recipe,
            {0, 0, std::numeric_limits<int>::max(), 64},
            {0, 0, std::numeric_limits<int>::max(), 64});
        EXPECT_FALSE(build_effects_descriptor_for_test(extremeExtent, descriptor));

        auto extremeCoordinate = effects_input_for_test(
            recipe,
            {std::numeric_limits<int>::max(), 0, 64, 64},
            {std::numeric_limits<int>::max(), 0, 64, 64});
        extremeCoordinate.geometry.scaleX = 1.0e-12;
        EXPECT_FALSE(build_effects_descriptor_for_test(extremeCoordinate, descriptor));

        auto nonfiniteStep = effects_input_for_test(recipe);
        nonfiniteStep.geometry.scaleX = std::numeric_limits<double>::denorm_min();
        EXPECT_FALSE(build_effects_descriptor_for_test(nonfiniteStep, descriptor));
    }

    TEST(ScatterHalationControls, RetainsFloat32BoundsAndIdentity) {
        JuicerTests::GtestCohort control_values_cases("control-values", 4u);
        std::size_t control_values_index = 0;
        for (double value : {0.0, 1.0, 1.25, 2.0}) {
            auto control_values_case = control_values_cases.case_event(std::to_string(control_values_index++));
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

            control_values_case.passed();
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
        const std::array<float, 3> firstSigmaUm{65.0f, 65.0f, 65.0f};
        const std::array<float, 3> primaryAmount{0.08f, 0.02f, 0.0f};
        const auto firstRecipe = recipe_or_fail(first, firstSigmaUm, primaryAmount);
        const auto sameRecipe = recipe_or_fail(same, firstSigmaUm, primaryAmount);
        const auto distinctRecipe = recipe_or_fail(distinct, firstSigmaUm, primaryAmount);
        EXPECT_EQ(firstRecipe.hash, sameRecipe.hash);
        EXPECT_NE(firstRecipe.hash, distinctRecipe.hash);
    }

    TEST(ScatterHalationControls, RejectsInvalidActiveValues) {
        JuicerTests::GtestCohort invalid_values_cases("invalid-values", 5u);
        std::size_t invalid_values_index = 0;
        for (double value : {
                 std::numeric_limits<double>::quiet_NaN(),
                 std::numeric_limits<double>::infinity(),
                 -std::numeric_limits<double>::infinity(),
                 -0.01,
                 std::nextafter(2.0, 3.0)}) {
            auto invalid_values_case = invalid_values_cases.case_event(std::to_string(invalid_values_index++));
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

            invalid_values_case.passed();
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
        JuicerTests::GtestCohort profile_mappings_cases("profile-mappings", std::size(mappings));
        std::size_t profile_mappings_index = 0;
        for (const Mapping& mapping : mappings) {
            auto profile_mappings_case = profile_mappings_cases.case_event(std::to_string(profile_mappings_index++));
            SCOPED_TRACE(mapping.sigma);
            const std::array<float, 3> firstSigmaUm{mapping.sigma, mapping.sigma, mapping.sigma};
            const auto recipe = recipe_or_fail(active, firstSigmaUm, mapping.strength);
            EXPECT_TRUE(recipe.scatterActive);
            EXPECT_TRUE(recipe.backReflectionActive);
            EXPECT_NE(recipe.hash, 0u);
            EXPECT_EQ(recipe.halationFirstSigmaUm, firstSigmaUm);
            for (std::size_t channel = 0; channel < mapping.strength.size(); ++channel) {
                EXPECT_EQ(
                    float_bits(recipe.totalStrength[channel]),
                    float_bits(mapping.strength[channel]));
            }

            profile_mappings_case.passed();
        }

        const ScatterHalationControls inactive;
        const auto identity = recipe_or_fail(
            inactive,
            std::array<float, 3>{65.0f, 65.0f, 65.0f},
            std::array<float, 3>{0.08f, 0.02f, 0.0f});
        EXPECT_FALSE(identity.scatterActive);
        EXPECT_FALSE(identity.backReflectionActive);
        EXPECT_EQ(identity.hash, 0u);
    }

    TEST(ScatterHalationRecipe, KeepsScatterAndReflectionIndependent) {
        const auto scatterOnly = recipe_or_fail(
            controls_or_fail({true, 1.0, 1.0, 0.0, 1.0}),
            std::array<float, 3>{65.0f, 65.0f, 65.0f},
            std::array<float, 3>{0.08f, 0.02f, 0.0f});
        EXPECT_TRUE(scatterOnly.scatterActive);
        EXPECT_FALSE(scatterOnly.backReflectionActive);
        EXPECT_NE(scatterOnly.hash, 0u);

        const auto reflectionOnly = recipe_or_fail(
            controls_or_fail({true, 0.0, 1.0, 1.0, 1.0}),
            std::array<float, 3>{65.0f, 65.0f, 65.0f},
            std::array<float, 3>{0.08f, 0.02f, 0.0f});
        EXPECT_FALSE(reflectionOnly.scatterActive);
        EXPECT_TRUE(reflectionOnly.backReflectionActive);
        EXPECT_NE(reflectionOnly.hash, 0u);
        EXPECT_NE(scatterOnly.hash, reflectionOnly.hash);
    }

    TEST(ScatterHalationDescriptor, BindsRecipeAndSelectsReferenceOperators) {
        const auto recipe = recipe_or_fail(
            controls_or_fail({true, 1.0, 1.0, 1.0, 1.0}),
            std::array<float, 3>{65.0f, 65.0f, 65.0f},
            std::array<float, 3>{0.08f, 0.02f, 0.0f});
        std::optional<ScatterHalationFrameDescriptor> descriptor;
        std::string diagnostic;
        ASSERT_TRUE(JuicerTestSupport::prepare_halation_frame(
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
        EXPECT_TRUE(JuicerTestSupport::prepare_halation_frame(
            identity,
            std::numeric_limits<float>::quiet_NaN(),
            descriptor,
            diagnostic))
            << diagnostic;
        EXPECT_FALSE(descriptor);
    }

    TEST_F(ScatterHalationBootstrapTest, LoadsPublishedProfilesWithoutCuda) {
        FjCatalog* acquiredCatalog = nullptr;
        FjCatalogCounts counts{};
        ASSERT_EQ(fj_test_catalog_acquire(runtime.borrow().assets, &acquiredCatalog, &counts, nullptr).category, FJ_STATUS_SUCCESS);
        const auto releaseCatalog = [](FjCatalog* held) {
            (void)fj_test_catalog_release(held, nullptr);
        };
        std::unique_ptr<FjCatalog, decltype(releaseCatalog)> catalog(acquiredCatalog, releaseCatalog);
        ASSERT_TRUE(catalog);
        const std::string key = "kodak_portra_400";
        const std::string printKey = "kodak_portra_endura";
        bool defaultFilmPresent = false, defaultPrintPresent = false;
        for (const auto role : {FJ_PROFILE_ROLE_FILM, FJ_PROFILE_ROLE_PRINT}) {
            const auto count = role == FJ_PROFILE_ROLE_FILM ? counts.film_count : counts.print_count;
            for (std::size_t i = 0; i < count; ++i) {
                FjCatalogEntryView entry{};
                ASSERT_EQ(fj_test_catalog_entry(catalog.get(), role, i, &entry, nullptr).category, FJ_STATUS_SUCCESS);
                const std::string_view entryKey(entry.key.data, entry.key.count);
                if (role == FJ_PROFILE_ROLE_FILM && entryKey == key)
                    defaultFilmPresent = true;
                if (role == FJ_PROFILE_ROLE_PRINT && entryKey == printKey)
                    defaultPrintPresent = true;
            }
        }
        EXPECT_TRUE(defaultFilmPresent);
        EXPECT_TRUE(defaultPrintPresent);

        FjFilmProfile* acquired = nullptr;
        ASSERT_EQ(fj_test_assets_film_acquire(runtime.borrow().assets, {key.data(), key.size()}, &acquired, nullptr).category, FJ_STATUS_SUCCESS);
        const auto releaseProfile = [](FjFilmProfile* held) {
            (void)fj_test_assets_film_release(held, nullptr);
        };
        std::unique_ptr<FjFilmProfile, decltype(releaseProfile)> profile(acquired, releaseProfile);
        ASSERT_TRUE(profile);
        FjFilmProfileView view{};
        ASSERT_EQ(fj_test_assets_film_view(profile.get(), &view, nullptr).category, FJ_STATUS_SUCCESS);
        EXPECT_EQ(std::string(view.stock.data, view.stock.count), key);
        ASSERT_EQ(view.wavelengths.count, 81u);
        EXPECT_EQ(view.wavelengths.data[0], 380.0f);
        EXPECT_EQ(view.wavelengths.data[80], 780.0f);
        EXPECT_TRUE(Spectral::spectral_shape_matches_reference(Spectral::kReferenceShape));
    }


} // namespace

int main(int argc, char** argv) noexcept {
    try {
        testing::InitGoogleTest(&argc, argv);
        if (::testing::GTEST_FLAG(list_tests))
            return RUN_ALL_TESTS();
        JuicerTests::CaseLog case_log;
        const int result = RUN_ALL_TESTS();
        case_log.finish();
        return result;
    } catch (const std::exception& error) {
        std::fprintf(
            stderr,
            "FATAL scatter/halation recipe test: %s\n",
            error.what());
    } catch (...) {
        std::fputs(
            "FATAL scatter/halation recipe test: unknown exception\n",
            stderr);
    }
    return 2;
}
