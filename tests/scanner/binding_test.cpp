#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <span>
#include <string>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "juicer_cuda_prepared.h"

namespace {
    template <std::size_t N>
    void read_binding_words(const nlohmann::json& source, std::span<float, N> output) {
        const auto words = source.get<std::array<std::uint32_t, N>>();
        std::ranges::transform(words, output.begin(), [](std::uint32_t value) {
            return std::bit_cast<float>(value);
        });
    }

    std::uint64_t fixture_binding_digest(const nlohmann::json& row) {
        FjPreparedHostData source{};
        source.route = row.at("route");
        const bool print = source.route == FJ_ROUTE_NEGATIVE_PRINT || source.route == FJ_ROUTE_POSITIVE_PRINT;
        const auto& key = row.at("key").get_ref<const std::string&>();
        auto& profileKey = print ? source.print_profile_key : source.film_profile_key;
        profileKey.data = key.data();
        profileKey.count = key.size();
        (print ? source.print_profile_asset_version : source.film_profile_asset_version) = row.at("asset_token");
        std::array<std::array<float, 81>, 6> lanes{};
        std::array<float, 81> base{};
        auto& spectra = source.scanner_spectra;
        const std::array spanSlots{&spectra.dye_c, &spectra.dye_m, &spectra.dye_y, &spectra.weighted_x, &spectra.weighted_y, &spectra.weighted_z};
        for (std::size_t i = 0; i < lanes.size(); ++i) {
            read_binding_words(row.at("lanes").at(i), std::span{lanes[i]});
            *spanSlots[i] = {lanes[i].data(), lanes[i].size()};
        }
        if (!row.at("base").is_null()) {
            read_binding_words(row.at("base"), std::span{base});
            spectra.base_density = {base.data(), base.size()};
        }
        spectra.inverse_y_normalization = std::bit_cast<float>(row.at("inverse_y").get<std::uint32_t>());
        auto& color = source.output_color;
        color.color_space = row.at("output_space");
        color.flags = row.at("flags");
        read_binding_words(row.at("white"), std::span{color.illuminant_xyz});
        read_binding_words(row.at("cat02"), std::span{color.cat02});
        read_binding_words(row.at("xyz_to_rgb"), std::span{color.xyz_to_rgb});
        std::array<float, 6> coefficients{};
        read_binding_words(row.at("coefficients"), std::span{coefficients});
        const auto [gamma, a, b, c, d, cutoff] = coefficients;
        color.encoding = {row.at("encoding_schema"), row.at("encoding_kind"), gamma, a, b, c, d, cutoff, {}};
        read_binding_words(row.at("dwg_to_output"), std::span{color.encoding.dwg_to_output});
        return JuicerCuda::spectral_binding_digest(source);
    }

    TEST(ScannerBinding, SharedSpectralBindingKnownAnswers) {
        std::ifstream stream(FJ_SCANNER_BINDING_FIXTURE);
        ASSERT_TRUE(stream);
        const auto fixture = nlohmann::json::parse(stream);
        ASSERT_EQ(fixture.at("schema"), "scanner-binding-fixture-v1");
        const auto& rows = fixture.at("vectors");
        ASSERT_EQ(rows.size(), 9U);
        const auto original = fixture_binding_digest(rows.at(0));
        ASSERT_EQ(rows.at(0).at("id"), "direct-absent-base");
        for (const auto& row : rows) {
            SCOPED_TRACE(row.at("id").get<std::string>());
            ASSERT_EQ(row.at("schema_word"), 1U);
            const auto actual = fixture_binding_digest(row);
            EXPECT_EQ(actual, row.at("digest_u64").get<std::uint64_t>());
            if (row.at("id") == "direct-absent-base" || row.at("id") == "canonical-equivalent")
                EXPECT_EQ(actual, original);
            else
                EXPECT_NE(actual, original);
        }
    }
} // namespace
