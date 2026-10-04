#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "asset_lookup_test.h"
#include "RenderRecipe.h"
#include "Scanner.h"
#include "SpectralData.h"

namespace {

    constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
    constexpr float kInfinity = std::numeric_limits<float>::infinity();

    TEST(AssetLookup, SyntheticFiniteInterpolationAndEndpoints) {
        const std::vector<float> axis{0.0f, 2.0f, 4.0f};
        const std::vector<std::array<float, 3>> curves{{0.0f, 4.0f, 8.0f}, {4.0f, 8.0f, 12.0f}, {8.0f, 12.0f, 16.0f}};
        for (std::size_t channel = 0; channel < 3u; ++channel) {
            const float offset = 4.0f * static_cast<float>(channel);
            EXPECT_EQ(AssetLookupTest::sample_synthetic_density_for_test(1.0f, axis, curves, channel), 2.0f + offset);
            for (float query : {-1.0f, 0.0f}) {
                EXPECT_EQ(AssetLookupTest::sample_synthetic_density_for_test(query, axis, curves, channel), offset);
            }
            for (float query : {4.0f, 5.0f}) {
                EXPECT_EQ(AssetLookupTest::sample_synthetic_density_for_test(query, axis, curves, channel), 8.0f + offset);
            }
        }
    }

    TEST(AssetLookup, ScannerFiniteAscendingAndDescending) {
        for (float query : {-1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f}) {
            const float expected = query < 0.0f ? 0.0f : (query > 4.0f ? 8.0f : query * 2.0f);
            EXPECT_EQ(AssetLookupTest::interp_clamped_monotonic_for_test(query, {0.0f, 2.0f, 4.0f}, {0.0f, 4.0f, 8.0f}), expected);
            EXPECT_EQ(AssetLookupTest::interp_clamped_monotonic_for_test(query, {4.0f, 2.0f, 0.0f}, {8.0f, 4.0f, 0.0f}), expected);
        }
    }

    TEST(AssetLookup, SingletonAndDistinctDuplicateSelection) {
        for (float query : {-kInfinity, 1.0f, 2.0f, 3.0f, kInfinity}) {
            EXPECT_EQ(AssetLookupTest::sample_synthetic_density_for_test(query, {2.0f}, {{7.0f, 8.0f, 9.0f}}, 0u), 7.0f);
            EXPECT_EQ(AssetLookupTest::interp_clamped_monotonic_for_test(query, {2.0f}, {7.0f}), 7.0f);
        }
        const std::vector<float> axis{0.0f, 2.0f, 2.0f, 4.0f};
        const std::vector<std::array<float, 3>> curves{{0.0f, 0.0f, 0.0f}, {4.0f, 4.0f, 4.0f}, {8.0f, 8.0f, 8.0f}, {12.0f, 12.0f, 12.0f}};
        EXPECT_EQ(AssetLookupTest::sample_synthetic_density_for_test(2.0f, axis, curves, 0u), 8.0f);
        EXPECT_EQ(AssetLookupTest::sample_synthetic_density_for_test(3.0f, axis, curves, 0u), 10.0f);
        EXPECT_EQ(AssetLookupTest::interp_clamped_monotonic_for_test(2.0f, axis, {0.0f, 4.0f, 8.0f, 12.0f}), 4.0f);
        EXPECT_EQ(AssetLookupTest::interp_clamped_monotonic_for_test(3.0f, axis, {0.0f, 4.0f, 8.0f, 12.0f}), 10.0f);
        EXPECT_EQ(AssetLookupTest::interp_clamped_monotonic_for_test(2.0f, {4.0f, 2.0f, 2.0f, 0.0f}, {12.0f, 8.0f, 4.0f, 0.0f}), 4.0f);
        EXPECT_EQ(AssetLookupTest::interp_clamped_monotonic_for_test(3.0f, {4.0f, 2.0f, 2.0f, 0.0f}, {12.0f, 8.0f, 4.0f, 0.0f}), 10.0f);
        EXPECT_EQ(AssetLookupTest::sample_synthetic_density_for_test(2.0f, {2.0f, 2.0f}, {{7.0f, 7.0f, 7.0f}, {9.0f, 9.0f, 9.0f}}, 0u), 7.0f);
    }

    TEST(AssetLookup, OrderedInfinitiesDoNotImplyFiniteOutput) {
        const std::vector<float> axis{-kInfinity, 0.0f, kInfinity};
        const std::vector<std::array<float, 3>> curves{{1.0f, 1.0f, 1.0f}, {2.0f, 2.0f, 2.0f}, {3.0f, 3.0f, 3.0f}};
        for (float query : {-kInfinity, 1.0f, kInfinity}) {
            const float expected = query == -kInfinity ? 1.0f : (query == kInfinity ? 3.0f : 2.0f);
            EXPECT_EQ(AssetLookupTest::sample_synthetic_density_for_test(query, axis, curves, 0u), expected);
            EXPECT_EQ(AssetLookupTest::interp_clamped_monotonic_for_test(query, axis, {1.0f, 2.0f, 3.0f}), expected);
            EXPECT_EQ(AssetLookupTest::interp_clamped_monotonic_for_test(query, {kInfinity, 0.0f, -kInfinity}, {3.0f, 2.0f, 1.0f}), expected);
        }
        const auto synthetic = AssetLookupTest::sample_synthetic_density_for_test(-1.0f, axis, curves, 0u);
        const auto scanner = AssetLookupTest::interp_clamped_monotonic_for_test(-1.0f, axis, {1.0f, 2.0f, 3.0f});
        ASSERT_TRUE(synthetic);
        ASSERT_TRUE(scanner);
        EXPECT_TRUE(std::isnan(*synthetic));
        EXPECT_TRUE(std::isnan(*scanner));
    }

    TEST(AssetLookup, RejectsQueryShapeAndChannelBeforeAccess) {
        const std::vector<std::array<float, 3>> curves{{1.0f, 2.0f, 3.0f}};
        EXPECT_FALSE(AssetLookupTest::sample_synthetic_density_for_test(kNan, {0.0f}, curves, 0u));
        EXPECT_FALSE(AssetLookupTest::sample_synthetic_density_for_test(0.0f, {}, {}, 0u));
        EXPECT_FALSE(AssetLookupTest::sample_synthetic_density_for_test(0.0f, {0.0f, 1.0f}, curves, 0u));
        EXPECT_FALSE(AssetLookupTest::sample_synthetic_density_for_test(0.0f, {0.0f}, curves, 3u));
        EXPECT_FALSE(AssetLookupTest::interp_clamped_monotonic_for_test(kNan, {0.0f}, {1.0f}));
        EXPECT_FALSE(AssetLookupTest::interp_clamped_monotonic_for_test(0.0f, {}, {}));
        EXPECT_FALSE(AssetLookupTest::interp_clamped_monotonic_for_test(0.0f, {0.0f}, {}));
        // Scanner uses a linear traversal, so these cases have no partition precondition.
        EXPECT_FALSE(AssetLookupTest::interp_clamped_monotonic_for_test(1.0f, {kNan}, {1.0f}));
        EXPECT_FALSE(AssetLookupTest::interp_clamped_monotonic_for_test(1.0f, {0.0f, kNan, 2.0f}, {0.0f, 1.0f, 2.0f}));
        Profiles::FilmProfileSamples data;
        EXPECT_FALSE(AssetLookupTest::sample_density_curve_for_test(0.0f, data, 0u));
        data.logExposure = {0.0f};
        EXPECT_FALSE(AssetLookupTest::sample_density_curve_for_test(0.0f, data, 0u));
        data.densityCurves = curves;
        EXPECT_FALSE(AssetLookupTest::sample_density_curve_for_test(kNan, data, 0u));
        EXPECT_FALSE(AssetLookupTest::sample_density_curve_for_test(0.0f, data, 3u));
        EXPECT_EQ(AssetLookupTest::sample_density_curve_for_test(0.0f, data, 2u), 3.0f);
    }

    struct ScannerCorrectionFixture {
        RenderRecipe recipe;
        std::shared_ptr<Profiles::FilmProfile> film = std::make_shared<Profiles::FilmProfile>();
        std::shared_ptr<Profiles::PrintProfile> print = std::make_shared<Profiles::PrintProfile>();
        Spectral::SpectralTables tables;
        std::array<float, 81> illuminant{};
        std::array<float, 3> preflash{};

        ScannerCorrectionFixture() {
            recipe.profileRoute.filmProfile = film;
            recipe.profileRoute.printProfile = print;
            recipe.scannerOutput.blackCorrection = true;
            recipe.scannerOutput.whiteCorrection = true;
            recipe.scannerOutput.blackLevel = 0.0f;
            recipe.scannerOutput.whiteLevel = 1.0f;
            recipe.scannerOutput.syntheticFilmReference.hash = 1u;
            recipe.scannerOutput.syntheticFilmReference.baselineDensityCmy = {1.0f, 1.0f, 1.0f};
            recipe.densityBounds.dataMaxCmy = {3.0f, 3.0f, 3.0f};
            recipe.densityBounds.invSpanCmy = {1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f};
            recipe.densityBounds.hash = 1u;
            recipe.enlargerFilmBounds.dataMaxCmy = {3.0f, 3.0f, 3.0f};
            film->data.logExposure = {-2.0f, 0.0f, 2.0f};
            recipe.filmDevelop.authoredDensityCurves = {{3.0f, 3.0f, 3.0f}, {1.5f, 1.5f, 1.5f}, {0.0f, 0.0f, 0.0f}};
            print->data.logExposure = {-2.0f, 0.0f, 2.0f};
            print->data.densityCurves = {{0.0f, 0.0f, 0.0f}, {1.5f, 1.5f, 1.5f}, {3.0f, 3.0f, 3.0f}};
            for (std::size_t sample = 0; sample < illuminant.size(); ++sample) {
                film->data.channelDensity[sample] = {1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f};
                print->data.linearSensitivity[sample] = {1.0f / 81.0f, 1.0f / 81.0f, 1.0f / 81.0f};
                illuminant[sample] = 1.0f;
            }
            tables.K = 1;
            tables.epsC = tables.epsM = tables.epsY = {1.0f / 3.0f};
            tables.Ax = tables.Ay = tables.Az = {1.0f};
        }

        Scanner::PrintCorrectionDerivationInput print_input() const {
            return {&recipe, &tables, illuminant.data(), 81, preflash.data(), 1.0f};
        }
    };

    TEST(AssetLookup, DirectBuilderPropagatesLookupFailure) {
        ScannerCorrectionFixture fixture;
        fixture.recipe.profileRoute.scanRoute = Spektrafilm::ScanRoute::PositiveDirectScan;
        Scanner::ScannerColorCorrectionDescriptor descriptor;
        std::string diagnostic;
        ASSERT_TRUE(Scanner::build_direct_scanner_color_correction_descriptor(fixture.recipe, fixture.tables, descriptor, diagnostic)) << diagnostic;
        EXPECT_NE(descriptor.hash, 0u);
        fixture.recipe.filmDevelop.authoredDensityCurves.clear();
        EXPECT_FALSE(Scanner::build_direct_scanner_color_correction_descriptor(fixture.recipe, fixture.tables, descriptor, diagnostic));
        EXPECT_EQ(diagnostic, "ResourceDescriptorMismatch phase=8B direct density lookup");
        EXPECT_EQ(descriptor.hash, 0u);
        EXPECT_FALSE(descriptor.active);
    }

    TEST(AssetLookup, PrintBuilderPropagatesForwardLookupFailure) {
        ScannerCorrectionFixture fixture;
        fixture.recipe.profileRoute.scanRoute = Spektrafilm::ScanRoute::NegativePrintScan;
        fixture.recipe.densityBounds.medium = Spektrafilm::DensityMedium::Print;
        Scanner::ScannerColorCorrectionDescriptor descriptor;
        std::string diagnostic;
        ASSERT_TRUE(Scanner::build_print_scanner_color_correction_descriptor(fixture.print_input(), descriptor, diagnostic)) << diagnostic;
        EXPECT_NE(descriptor.hash, 0u);
        fixture.print->data.logExposure.clear();
        EXPECT_FALSE(Scanner::build_print_scanner_color_correction_descriptor(fixture.print_input(), descriptor, diagnostic));
        EXPECT_EQ(diagnostic, "ResourceDescriptorMismatch phase=8B print reference density lookup");
        EXPECT_EQ(descriptor.hash, 0u);
        EXPECT_FALSE(descriptor.active);
    }

    class ScopedMallettBasis {
    public:
        ScopedMallettBasis() : previous_(std::move(Spectral::context().mallettBasis)) {
            auto& basis = Spectral::context().mallettBasis;
            basis.rows = 81;
            basis.cols = 3;
            basis.data.assign(243u, 1.0f / 3.0f);
        }
        ~ScopedMallettBasis() {
            Spectral::context().mallettBasis = std::move(previous_);
        }
        ScopedMallettBasis(const ScopedMallettBasis&) = delete;
        ScopedMallettBasis& operator=(const ScopedMallettBasis&) = delete;

    private:
        Spectral::MallettBasis previous_;
    };

    TEST(AssetLookup, SyntheticBuilderRejectsNaNQueryBeforePublication) {
        const ScopedMallettBasis basis;
        RenderRecipe recipe;
        recipe.profileRoute.filmProfile = std::make_shared<Profiles::FilmProfile>();
        recipe.profileRoute.scanRoute = Spektrafilm::ScanRoute::NegativeDirectScan;
        recipe.filmRaw.rgbToRawMethod = Spektrafilm::RgbToRawMethod::Mallett2019;
        recipe.filmRaw.mallettGreenMidgrayScale = 1.0f;
        recipe.filmDevelop.logExposure = {0.0f};
        recipe.filmDevelop.authoredDensityCurves = {{2.0f, 4.0f, 6.0f}};
        std::array<float, 81> illuminant{};
        illuminant.fill(1.0f);
        for (auto& sensitivity : recipe.filmRaw.finalSensitivity) {
            sensitivity = {1.0f, 1.0f, 1.0f};
        }
        std::string diagnostic;
        ASSERT_TRUE(Spektrafilm::finish_synthetic_reference_recipes(recipe, nullptr, illuminant, nullptr, diagnostic)) << diagnostic;
        EXPECT_EQ(recipe.scannerOutput.syntheticFilmReference.baselineDensityCmy, (std::array<float, 3>{2.0f, 4.0f, 6.0f}));
        const auto referenceHash = recipe.scannerOutput.syntheticFilmReference.hash;
        const auto scannerHash = recipe.scannerOutput.hash;
        const auto recipeHash = recipe.hash;
        ASSERT_NE(referenceHash, 0u);
        // Positive subnormal mean makes the existing reciprocal overflow. Zero
        // channels then produce NaN lookup queries through 0 * infinity.
        recipe.filmRaw.autoExposureEnabled = true;
        for (auto& sensitivity : recipe.filmRaw.finalSensitivity) {
            sensitivity = {0.0f, 1.0e-40f, 0.0f};
        }
        EXPECT_FALSE(Spektrafilm::finish_synthetic_reference_recipes(recipe, nullptr, illuminant, nullptr, diagnostic));
        EXPECT_EQ(diagnostic, "ResourceDescriptorMismatch phase=3B field=synthetic_reference_density_lookup");
        EXPECT_EQ(recipe.scannerOutput.syntheticFilmReference.hash, referenceHash);
        EXPECT_EQ(recipe.scannerOutput.hash, scannerHash);
        EXPECT_EQ(recipe.hash, recipeHash);
    }

    TEST(AssetLookup, HanatosNegativeWidthReference) {
        const std::array<float, 4> params{380.0f, -100.0f, 780.0f, 100.0f};
        double response = 0.0;
        for (int sample = 0; sample < 81; ++sample) {
            const float window = AssetLookupTest::hanatos_window_sample_for_test(380.0f + 5.0f * static_cast<float>(sample), params);
            ASSERT_TRUE(std::isfinite(window));
            EXPECT_GE(window, 0.0f);
            response += window;
        }
        EXPECT_GT(response, 0.0);
        const float first = AssetLookupTest::hanatos_window_sample_for_test(380.0f, params);
        const float last = AssetLookupTest::hanatos_window_sample_for_test(780.0f, params);
        EXPECT_GT(first, 0.25f);
        EXPECT_LT(first, 0.5f);
        EXPECT_GT(last, 0.0f);
        EXPECT_LT(last, 0.25f);
    }

    TEST(AssetLookup, HanatosF64WidthNarrowsToInfinityReference) {
        const std::array<double, 4> authored{380.0, 1.0e40, 780.0, 100.0};
        std::array<float, 4> params{};
        for (std::size_t index = 0; index < params.size(); ++index) {
            params[index] = static_cast<float>(authored[index]);
        }
        ASSERT_EQ(params[1], kInfinity);
        double response = 0.0;
        for (int sample = 0; sample < 81; ++sample) {
            const float window = AssetLookupTest::hanatos_window_sample_for_test(380.0f + 5.0f * static_cast<float>(sample), params);
            ASSERT_TRUE(std::isfinite(window));
            EXPECT_GE(window, 0.0f);
            response += window;
        }
        EXPECT_GT(response, 0.0);
        EXPECT_EQ(AssetLookupTest::hanatos_window_sample_for_test(780.0f, params), 0.25f);
    }

    TEST(AssetLookup, HanatosZeroWidthComputedNaNReference) {
        const std::array<float, 4> params{380.0f, 0.0f, 780.0f, 100.0f};
        EXPECT_TRUE(std::isnan(AssetLookupTest::hanatos_window_sample_for_test(380.0f, params)));
        for (int sample = 1; sample < 81; ++sample) {
            const float window = AssetLookupTest::hanatos_window_sample_for_test(380.0f + 5.0f * static_cast<float>(sample), params);
            EXPECT_TRUE(std::isfinite(window));
            EXPECT_GT(window, 0.0f);
        }
        EXPECT_EQ(AssetLookupTest::hanatos_window_sample_for_test(780.0f, params), 0.5f);
    }

} // namespace
