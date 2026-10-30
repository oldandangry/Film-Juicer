#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "asset_lookup_test.h"
#include "juicer_cuda_owner.h"
#include "juicer_test_api.h"
#include "JuicerState.h"
#include "ProcessRoot.h"
#include "ProfileAssets.h"

namespace AssetLookupTest {
    std::optional<float> sample_reference_density(float query, const std::vector<float>& axis, const std::vector<std::array<float, 3>>& curves, std::size_t channel) {
        std::vector<float> flat;
        for (const auto& row : curves)
            flat.insert(flat.end(), row.begin(), row.end());
        float out = 0;
        std::uint32_t present = 0;
        const auto status = fj_test_film_reference_density_lookup(query, {axis.data(), axis.size()}, {flat.data(), flat.size()}, channel < 3 ? static_cast<std::uint32_t>(channel) : 3u, &out, &present);
        if (status.category != FJ_STATUS_SUCCESS || status.api != FJ_API_NONE || status.native_code != 0 || present > 1)
            throw std::runtime_error("film reference lookup fixture transport");
        return present ? std::optional<float>{out} : std::nullopt;
    }
    std::optional<float> interp_clamped_monotonic_for_test(float query, const std::vector<float>& axis, const std::vector<float>& values) {
        float out = 0;
        std::uint32_t present = 0;
        const auto status = fj_test_scanner_lookup(query, {axis.data(), axis.size()}, {values.data(), values.size()}, &out, &present);
        if (status.category != FJ_STATUS_SUCCESS || status.api != FJ_API_NONE || status.native_code != 0 || present > 1)
            throw std::runtime_error("scanner lookup fixture transport");
        return present ? std::optional<float>{out} : std::nullopt;
    }
    std::optional<float> sample_density_curve_for_test(float query, const Profiles::FilmProfileSamples& data, std::size_t channel) {
        std::vector<float> curves;
        for (const auto& row : data.densityCurves)
            for (float value : row)
                curves.push_back(value);
        float out = 0;
        std::uint32_t present = 0;
        const auto status = fj_test_scanner_density_curve(query, {data.logExposure.data(), data.logExposure.size()}, {curves.data(), curves.size()}, channel < 3 ? static_cast<std::uint32_t>(channel) : 3u, &out, &present);
        if (status.category != FJ_STATUS_SUCCESS || status.api != FJ_API_NONE || status.native_code != 0 || present > 1)
            throw std::runtime_error("scanner density lookup fixture transport");
        return present ? std::optional<float>{out} : std::nullopt;
    }
} // namespace AssetLookupTest

namespace {

    constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
    constexpr float kInfinity = std::numeric_limits<float>::infinity();

    FjFloatSpan span(const float* data, std::size_t count) {
        return {data, count};
    }

    void require_status(FjStatus value, std::uint32_t category) {
        if (value.category != category || value.api != FJ_API_NONE || value.native_code != 0)
            throw std::runtime_error("closed lookup fixture status");
    }

    struct FilmReferenceOwner {
        FjFilmProfile* profile = nullptr;
        FjFilmDevelopmentOwner* development = nullptr;

        FilmReferenceOwner() {
            const std::string root = JUICER_LOOKUP_RESOURCE_DIR;
            const std::string key = "kodak_portra_400";
            try {
                require_status(fj_test_film_profile_acquire({root.data(), root.size()}, {key.data(), key.size()}, &profile, nullptr), FJ_STATUS_SUCCESS);
                if (!profile)
                    throw std::runtime_error("film profile facade returned no owner");
                require_status(fj_test_film_development_acquire(profile, 0, 1.0f, 0, &development, nullptr), FJ_STATUS_SUCCESS);
                if (!development)
                    throw std::runtime_error("film development facade returned no owner");
                const auto released = fj_test_film_profile_release(profile, nullptr);
                profile = nullptr;
                require_status(released, FJ_STATUS_SUCCESS);
            } catch (...) {
                if (development) {
                    (void)fj_test_film_development_release(development, nullptr);
                    development = nullptr;
                }
                if (profile) {
                    (void)fj_test_film_profile_release(profile, nullptr);
                    profile = nullptr;
                }
                throw;
            }
        }

        ~FilmReferenceOwner() {
            if (development && fj_test_film_development_release(development, nullptr).category != FJ_STATUS_SUCCESS)
                std::terminate();
            if (profile && fj_test_film_profile_release(profile, nullptr).category != FJ_STATUS_SUCCESS)
                std::terminate();
        }
        FilmReferenceOwner(const FilmReferenceOwner&) = delete;
        FilmReferenceOwner& operator=(const FilmReferenceOwner&) = delete;
    };

    struct ReferenceValue {
        std::array<std::uint32_t, 13> scalar_words{};
        std::uint64_t hash = 0;
        friend bool operator==(const ReferenceValue&, const ReferenceValue&) = default;
    };

    ReferenceValue reference_value(const FjFilmReference& value) {
        ReferenceValue out;
        std::size_t cursor = 0;
        for (const auto* row : {value.baseline_raw_rgb, value.compensated_raw_rgb, value.baseline_density_cmy, value.compensated_density_cmy}) {
            for (std::size_t channel = 0; channel < 3; ++channel)
                out.scalar_words[cursor++] = std::bit_cast<std::uint32_t>(row[channel]);
        }
        out.scalar_words[cursor] = std::bit_cast<std::uint32_t>(value.meter_scale);
        out.hash = value.hash;
        return out;
    }

    struct PublishedReference {
        std::uint64_t reference_hash = 0;
        std::uint64_t scanner_recipe_hash = 0;
        std::uint64_t recipe_hash = 0;
        ReferenceValue value;
        friend bool operator==(const PublishedReference&, const PublishedReference&) = default;
    };

    PublishedReference published_reference(const Spektrafilm::RenderRecipe& recipe) {
        return {
            recipe.scannerOutput.syntheticFilmReference.hash,
            recipe.scannerOutput.hash,
            recipe.hash,
            {std::array{
                 std::bit_cast<std::uint32_t>(recipe.scannerOutput.syntheticFilmReference.baselineRawRgb[0]),
                 std::bit_cast<std::uint32_t>(recipe.scannerOutput.syntheticFilmReference.baselineRawRgb[1]),
                 std::bit_cast<std::uint32_t>(recipe.scannerOutput.syntheticFilmReference.baselineRawRgb[2]),
                 std::bit_cast<std::uint32_t>(recipe.scannerOutput.syntheticFilmReference.compensatedRawRgb[0]),
                 std::bit_cast<std::uint32_t>(recipe.scannerOutput.syntheticFilmReference.compensatedRawRgb[1]),
                 std::bit_cast<std::uint32_t>(recipe.scannerOutput.syntheticFilmReference.compensatedRawRgb[2]),
                 std::bit_cast<std::uint32_t>(recipe.scannerOutput.syntheticFilmReference.baselineDensityCmy[0]),
                 std::bit_cast<std::uint32_t>(recipe.scannerOutput.syntheticFilmReference.baselineDensityCmy[1]),
                 std::bit_cast<std::uint32_t>(recipe.scannerOutput.syntheticFilmReference.baselineDensityCmy[2]),
                 std::bit_cast<std::uint32_t>(recipe.scannerOutput.syntheticFilmReference.compensatedDensityCmy[0]),
                 std::bit_cast<std::uint32_t>(recipe.scannerOutput.syntheticFilmReference.compensatedDensityCmy[1]),
                 std::bit_cast<std::uint32_t>(recipe.scannerOutput.syntheticFilmReference.compensatedDensityCmy[2]),
                 std::bit_cast<std::uint32_t>(recipe.scannerOutput.syntheticFilmReference.baselineMeterScale)},
             recipe.scannerOutput.syntheticFilmReference.hash}};
    }

    ParamSnapshot lookup_controls() {
        ParamSnapshot params;
        params.filmProfileKey = "kodak_portra_400";
        params.printProfileKey = "kodak_portra_endura";
        params.scanRoute = Spektrafilm::ScanRoute::NegativeDirectScan;
        params.spectralUpsamplingMode = static_cast<int>(Spektrafilm::RgbToRawMethod::Mallett2019);
        params.cameraExposureCompensationEv = 0.75;
        params.cameraAutoExposureEnabled = 0;
        params.inputCompressionEnabled = 0;
        return params;
    }

    void set_pending(InstanceState& state, const ParamSnapshot& params) {
        std::lock_guard lock(state.pending.m);
        state.pending.value = PendingParamsState::Valid{params, hash_params(params)};
    }

    bool admitted(const PendingRenderAdmissionResult& result) {
        return result.status == PendingRenderAdmissionStatus::AdmittedDirect && bool(result.directState);
    }

    void set_lookup_fault(std::uint32_t mode) {
        require_status(fj_test_film_reference_lookup_fault(mode), FJ_STATUS_SUCCESS);
    }

    TEST(AssetLookup, SyntheticFiniteInterpolationAndEndpoints) {
        const std::vector<float> axis{0.0f, 2.0f, 4.0f};
        const std::vector<std::array<float, 3>> curves{{0.0f, 4.0f, 8.0f}, {4.0f, 8.0f, 12.0f}, {8.0f, 12.0f, 16.0f}};
        for (std::size_t channel = 0; channel < 3u; ++channel) {
            const float offset = 4.0f * static_cast<float>(channel);
            EXPECT_EQ(AssetLookupTest::sample_reference_density(1.0f, axis, curves, channel), 2.0f + offset);
            for (float query : {-1.0f, 0.0f}) {
                EXPECT_EQ(AssetLookupTest::sample_reference_density(query, axis, curves, channel), offset);
            }
            for (float query : {4.0f, 5.0f}) {
                EXPECT_EQ(AssetLookupTest::sample_reference_density(query, axis, curves, channel), 8.0f + offset);
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
            EXPECT_EQ(AssetLookupTest::sample_reference_density(query, {2.0f}, {{7.0f, 8.0f, 9.0f}}, 0u), 7.0f);
            EXPECT_EQ(AssetLookupTest::interp_clamped_monotonic_for_test(query, {2.0f}, {7.0f}), 7.0f);
        }
        const std::vector<float> axis{0.0f, 2.0f, 2.0f, 4.0f};
        const std::vector<std::array<float, 3>> curves{{0.0f, 0.0f, 0.0f}, {4.0f, 4.0f, 4.0f}, {8.0f, 8.0f, 8.0f}, {12.0f, 12.0f, 12.0f}};
        EXPECT_EQ(AssetLookupTest::sample_reference_density(2.0f, axis, curves, 0u), 8.0f);
        EXPECT_EQ(AssetLookupTest::sample_reference_density(3.0f, axis, curves, 0u), 10.0f);
        EXPECT_EQ(AssetLookupTest::interp_clamped_monotonic_for_test(2.0f, axis, {0.0f, 4.0f, 8.0f, 12.0f}), 4.0f);
        EXPECT_EQ(AssetLookupTest::interp_clamped_monotonic_for_test(3.0f, axis, {0.0f, 4.0f, 8.0f, 12.0f}), 10.0f);
        EXPECT_EQ(AssetLookupTest::interp_clamped_monotonic_for_test(2.0f, {4.0f, 2.0f, 2.0f, 0.0f}, {12.0f, 8.0f, 4.0f, 0.0f}), 4.0f);
        EXPECT_EQ(AssetLookupTest::interp_clamped_monotonic_for_test(3.0f, {4.0f, 2.0f, 2.0f, 0.0f}, {12.0f, 8.0f, 4.0f, 0.0f}), 10.0f);
        EXPECT_EQ(AssetLookupTest::sample_reference_density(2.0f, {2.0f, 2.0f}, {{7.0f, 7.0f, 7.0f}, {9.0f, 9.0f, 9.0f}}, 0u), 7.0f);
        for (float query : {-0.0f, 0.0f}) {
            EXPECT_EQ(AssetLookupTest::sample_reference_density(query, {-0.0f, 0.0f}, {{7.0f, 7.0f, 7.0f}, {9.0f, 9.0f, 9.0f}}, 0u), 7.0f);
            EXPECT_EQ(AssetLookupTest::sample_reference_density(query, {-1.0f, -0.0f, 0.0f, 1.0f}, curves, 0u), 8.0f);
        }
    }

    TEST(AssetLookup, OrderedInfinitiesDoNotImplyFiniteOutput) {
        const std::vector<float> axis{-kInfinity, 0.0f, kInfinity};
        const std::vector<std::array<float, 3>> curves{{1.0f, 1.0f, 1.0f}, {2.0f, 2.0f, 2.0f}, {3.0f, 3.0f, 3.0f}};
        for (float query : {-kInfinity, 1.0f, kInfinity}) {
            const float expected = query == -kInfinity ? 1.0f : (query == kInfinity ? 3.0f : 2.0f);
            EXPECT_EQ(AssetLookupTest::sample_reference_density(query, axis, curves, 0u), expected);
            EXPECT_EQ(AssetLookupTest::interp_clamped_monotonic_for_test(query, axis, {1.0f, 2.0f, 3.0f}), expected);
            EXPECT_EQ(AssetLookupTest::interp_clamped_monotonic_for_test(query, {kInfinity, 0.0f, -kInfinity}, {3.0f, 2.0f, 1.0f}), expected);
        }
        const auto synthetic = AssetLookupTest::sample_reference_density(-1.0f, axis, curves, 0u);
        const auto scanner = AssetLookupTest::interp_clamped_monotonic_for_test(-1.0f, axis, {1.0f, 2.0f, 3.0f});
        ASSERT_TRUE(synthetic);
        ASSERT_TRUE(scanner);
        EXPECT_TRUE(std::isnan(*synthetic));
        EXPECT_TRUE(std::isnan(*scanner));
    }

    TEST(AssetLookup, RejectsQueryShapeAndChannelBeforeAccess) {
        const std::vector<std::array<float, 3>> curves{{1.0f, 2.0f, 3.0f}};
        EXPECT_FALSE(AssetLookupTest::sample_reference_density(kNan, {0.0f}, curves, 0u));
        EXPECT_FALSE(AssetLookupTest::sample_reference_density(0.0f, {}, {}, 0u));
        EXPECT_FALSE(AssetLookupTest::sample_reference_density(0.0f, {0.0f, 1.0f}, curves, 0u));
        EXPECT_FALSE(AssetLookupTest::sample_reference_density(0.0f, {0.0f}, curves, 3u));
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

    TEST(AssetLookup, DirectBuilderPropagatesLookupFailure) {
        const std::array<float, 1> eps{1.0f / 3.0f}, weightedY{1.0f};
        FjScannerCorrectionTables tables{
            span(eps.data(), eps.size()), span(eps.data(), eps.size()), span(eps.data(), eps.size()), span(weightedY.data(), weightedY.size()), {nullptr, 0}, 1, 1.0f, 0};
        const std::array<float, 81> filmBase{};
        const std::array<float, 3> axis{-2.0f, 0.0f, 2.0f};
        const std::array<float, 9> curves{3.0f, 3.0f, 3.0f, 1.5f, 1.5f, 1.5f, 0.0f, 0.0f, 0.0f};
        FjScannerDirectCorrectionInput input{};
        input.tables = &tables;
        input.film_base_density = span(filmBase.data(), filmBase.size());
        input.film_log_exposure = span(axis.data(), axis.size());
        input.film_authored_density_curves_cmy = span(curves.data(), curves.size());
        std::fill_n(input.minimum_density_cmy, 3, 0.0f);
        std::fill_n(input.maximum_density_cmy, 3, 3.0f);
        std::fill_n(input.inverse_span_cmy, 3, 1.0f / 3.0f);
        std::fill_n(input.baseline_density_cmy, 3, 1.0f);
        input.reference_hash = 1;
        FjScannerCorrectionParams params{2, 3, 0.0f, 1.0f};
        FjScannerColorCorrection correction{};
        FjScannerCorrectionFailure failure{};
        FjScannerCorrectionTrace trace{};
        std::array<char, 256> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
        auto status = fj_test_scanner_direct_correction_prepare(&params, &input, &correction, &failure, &trace, &error);
        ASSERT_EQ(status.category, FJ_STATUS_SUCCESS);
        EXPECT_EQ(status.api, FJ_API_NONE);
        EXPECT_EQ(status.native_code, 0);
        EXPECT_NE(correction.hash, 0u);
        input.film_authored_density_curves_cmy = {nullptr, 0};
        status = fj_test_scanner_direct_correction_prepare(&params, &input, &correction, &failure, &trace, &error);
        EXPECT_EQ(status.category, FJ_STATUS_PREPARATION_FAILURE);
        EXPECT_EQ(status.api, FJ_API_NONE);
        EXPECT_EQ(status.native_code, 0);
        EXPECT_EQ(failure.reason, FJ_SCANNER_CORRECTION_REASON_DIRECT_LOOKUP);
        EXPECT_EQ(std::string(diagnostic.data(), error.length), "ResourceDescriptorMismatch phase=8B direct density lookup");
        EXPECT_EQ(correction.hash, 0u);
        EXPECT_EQ(correction.active, 0u);
    }

    TEST(AssetLookup, PrintBuilderPropagatesForwardLookupFailure) {
        const std::array<float, 1> eps{1.0f / 3.0f}, weightedY{1.0f};
        FjScannerCorrectionTables tables{
            span(eps.data(), eps.size()), span(eps.data(), eps.size()), span(eps.data(), eps.size()), span(weightedY.data(), weightedY.size()), {nullptr, 0}, 1, 1.0f, 0};
        const std::array<float, 81> filmBase{}, printBase{}, illuminant = [] {
            std::array<float, 81> values{};
            values.fill(1.0f);
            return values;
        }();
        std::array<float, 243> filmChannelDensity{}, printSensitivity{};
        filmChannelDensity.fill(1.0f / 3.0f);
        printSensitivity.fill(1.0f / 81.0f);
        const std::array<float, 3> axis{-2.0f, 0.0f, 2.0f};
        const std::array<float, 9> curves{0.0f, 0.0f, 0.0f, 1.5f, 1.5f, 1.5f, 3.0f, 3.0f, 3.0f};
        FjScannerPrintCorrectionInput input{};
        input.tables = &tables;
        input.film_base_density = span(filmBase.data(), filmBase.size());
        input.film_channel_density_cmy = span(filmChannelDensity.data(), filmChannelDensity.size());
        input.print_sensitivity_cmy = span(printSensitivity.data(), printSensitivity.size());
        input.filtered_main_illuminant = span(illuminant.data(), illuminant.size());
        input.print_base_density = span(printBase.data(), printBase.size());
        input.print_log_exposure = span(axis.data(), axis.size());
        input.print_density_curves_cmy = span(curves.data(), curves.size());
        std::fill_n(input.minimum_density_cmy, 3, 0.0f);
        std::fill_n(input.inverse_span_cmy, 3, 1.0f / 3.0f);
        std::fill_n(input.film_minimum_density_cmy, 3, 0.0f);
        std::fill_n(input.film_maximum_density_cmy, 3, 3.0f);
        input.normalizer = 1.0f;
        input.medium = 1;
        FjScannerCorrectionParams params{1, 3, 0.0f, 1.0f};
        FjScannerColorCorrection correction{};
        FjScannerCorrectionFailure failure{};
        FjScannerCorrectionTrace trace{};
        std::array<char, 256> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
        auto status = fj_test_scanner_print_correction_prepare(&params, &input, &correction, &failure, &trace, &error);
        ASSERT_EQ(status.category, FJ_STATUS_SUCCESS);
        EXPECT_EQ(status.api, FJ_API_NONE);
        EXPECT_EQ(status.native_code, 0);
        EXPECT_NE(correction.hash, 0u);
        input.print_log_exposure = {nullptr, 0};
        status = fj_test_scanner_print_correction_prepare(&params, &input, &correction, &failure, &trace, &error);
        EXPECT_EQ(status.category, FJ_STATUS_PREPARATION_FAILURE);
        EXPECT_EQ(status.api, FJ_API_NONE);
        EXPECT_EQ(status.native_code, 0);
        EXPECT_EQ(failure.reason, FJ_SCANNER_CORRECTION_REASON_PRINT_REFERENCE_LOOKUP);
        EXPECT_EQ(std::string(diagnostic.data(), error.length), "ResourceDescriptorMismatch phase=8B print reference density lookup");
        EXPECT_EQ(correction.hash, 0u);
        EXPECT_EQ(correction.active, 0u);
    }

    TEST(AssetLookup, SyntheticBuilderRejectsNaNQueryBeforePublication) {
        FilmReferenceOwner owners;
        std::array<float, 243> basis{}, sensitivity{};
        basis.fill(1.0f / 3.0f);
        sensitivity.fill(1.0f);
        std::array<float, 81> illuminant{};
        illuminant.fill(1.0f);
        FjFilmReferenceInput input{};
        input.method = 1;
        input.mallett_green_scale = 1.0f;
        input.basis_rgb = span(basis.data(), basis.size());
        input.film_illuminant = span(illuminant.data(), illuminant.size());
        input.sensitivity_rgb = span(sensitivity.data(), sensitivity.size());
        FjFilmReference successful{};
        FjFilmReferenceFailure failure{};
        std::array<char, 256> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
        FjStatus result = fj_test_film_reference_prepare(owners.development, &input, &successful, &failure, &error);
        ASSERT_EQ(result.category, FJ_STATUS_SUCCESS);
        ASSERT_EQ(result.api, FJ_API_NONE);
        ASSERT_EQ(result.native_code, 0);
        ASSERT_NE(successful.hash, 0u);
        const ReferenceValue retained = reference_value(successful);

        // Positive subnormal mean makes the existing reciprocal overflow. Zero
        // channels then produce NaN lookup queries through 0 * infinity.
        input.auto_exposure_enabled = 1;
        for (std::size_t sample = 0; sample < 81; ++sample) {
            sensitivity[3 * sample] = 0.0f;
            sensitivity[3 * sample + 1] = 1.0e-40f;
            sensitivity[3 * sample + 2] = 0.0f;
        }
        FjFilmReference failed{};
        failed.meter_scale = 99.0f;
        failed.hash = 99;
        failed.baseline_raw_rgb[0] = failed.baseline_raw_rgb[1] = failed.baseline_raw_rgb[2] = 99.0f;
        result = fj_test_film_reference_prepare(owners.development, &input, &failed, &failure, &error);
        EXPECT_EQ(result.category, FJ_STATUS_PREPARATION_FAILURE);
        EXPECT_EQ(result.api, FJ_API_NONE);
        EXPECT_EQ(result.native_code, 0);
        EXPECT_EQ(failure.reason, FJ_FILM_REFERENCE_DENSITY_LOOKUP);
        EXPECT_EQ(failure.exposure, 1u);
        EXPECT_EQ(failure.operation, 8u);
        EXPECT_EQ(failure.channel, 1u);
        EXPECT_EQ(std::string(diagnostic.data(), error.length), "ResourceDescriptorMismatch phase=3B field=synthetic_reference_density_lookup");
        EXPECT_EQ(reference_value(failed), ReferenceValue{});
        EXPECT_EQ(reference_value(successful), retained);

        JuicerCuda::Owner processOwner;
        processOwner.create(std::filesystem::path(JUICER_LOOKUP_RESOURCE_DIR));
        JuicerProcess::root().ensure_bootstrap();
        auto controls = lookup_controls();
        // An earlier development-construction failure must leave the later
        // baseline/channel-zero lookup fault armed and unconsumed.
        InstanceState early;
        auto invalid = controls;
        invalid.filmGammaFactor = 0.0f;
        set_pending(early, invalid);
        ASSERT_NO_THROW(set_lookup_fault(1));
        const auto early_failure = admit_pending_render_state(early);
        EXPECT_EQ(early_failure.status, PendingRenderAdmissionStatus::RebuildFailed);
        EXPECT_EQ(fj_test_film_reference_lookup_fault_consumed(), 0u);
        EXPECT_EQ(fj_test_film_reference_lookup_fault_consumed(), 0u);
        ASSERT_NO_THROW(set_lookup_fault(0));

        InstanceState state;
        set_pending(state, controls);
        const auto initial = admit_pending_render_state(state);
        ASSERT_TRUE(admitted(initial));
        const auto old_state = initial.directState;
        const PublishedReference old_value = published_reference(old_state->recipe);
        ASSERT_NE(old_value.reference_hash, 0u);
        ASSERT_NE(old_value.scanner_recipe_hash, 0u);
        ASSERT_NE(old_value.recipe_hash, 0u);
        const std::uint64_t previous_hash = state.lastHash.load();
        const std::uint64_t previous_counter = state.buildCounterNext.load();

        auto changed = controls;
        changed.cameraExposureCompensationEv += 0.25;
        set_pending(state, changed);
        ASSERT_NO_THROW(set_lookup_fault(1));
        const auto unsupported_mode = fj_test_film_reference_lookup_fault(2);
        EXPECT_EQ(unsupported_mode.category, FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(unsupported_mode.api, FJ_API_NONE);
        EXPECT_EQ(unsupported_mode.native_code, 0);
        const auto rejected = admit_pending_render_state(state);
        EXPECT_EQ(fj_test_film_reference_lookup_fault_consumed(), 1u);
        EXPECT_EQ(fj_test_film_reference_lookup_fault_consumed(), 0u);
        EXPECT_EQ(rejected.status, PendingRenderAdmissionStatus::RebuildFailed);
        EXPECT_FALSE(rejected.directState);
        EXPECT_EQ(rejected.diagnostic, "ResourceDescriptorMismatch phase=3B field=synthetic_reference_density_lookup");
        EXPECT_FALSE(JuicerAtomic::load_shared_ptr(&state.activeDirectState));
        EXPECT_EQ(state.lastHash.load(), previous_hash);
        EXPECT_EQ(state.buildCounterNext.load(), previous_counter);
        EXPECT_EQ(published_reference(old_state->recipe), old_value);

        set_pending(state, changed);
        const auto retry = admit_pending_render_state(state);
        ASSERT_TRUE(admitted(retry));
        EXPECT_NE(retry.directState, old_state);
        EXPECT_EQ(state.lastHash.load(), hash_params(changed));
        EXPECT_EQ(state.buildCounterNext.load(), previous_counter + 1);
        const auto reused = admit_pending_render_state(state);
        ASSERT_TRUE(admitted(reused));
        EXPECT_EQ(reused.directState, retry.directState);
        EXPECT_EQ(state.buildCounterNext.load(), previous_counter + 1);
        ASSERT_EQ(JuicerProcess::root().assets().release_cached_payloads().category, FJ_STATUS_SUCCESS);
        EXPECT_EQ(published_reference(old_state->recipe), old_value);
        EXPECT_EQ(processOwner.close().category, FJ_STATUS_SUCCESS);
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
