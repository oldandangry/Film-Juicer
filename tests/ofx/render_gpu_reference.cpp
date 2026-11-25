#include "render_gpu_fixture.h"
#include "render_assertions.h"

#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>

namespace {
    using Json = nlohmann::json;
    using ofx_test::require_render;

    Json source_words(const ofx_test::RenderReferenceCase& row) {
        auto words = Json::array();
        words.get_ref<Json::array_t&>().reserve(35U * static_cast<std::size_t>(row.components));
        for (int y = 0; y < 5; ++y)
            for (int x = 0; x < 7; ++x)
                for (int c = 0; c < row.components; ++c) {
                    const float value = c == 3 ? 0.125F + static_cast<float>((x + y) % 4) * 0.25F
                                               : static_cast<float>((x + 3 * y + 7 * c) % 19) / 19.0F;
                    words.push_back(std::bit_cast<std::uint32_t>(value));
                }
        return words;
    }

    ofx_test::RenderPixels decode_rows(const Json& rows) {
        require_render(rows.is_array() && rows.size() == ofx_test::kRenderReferenceCases.size(), "spectral render fixture membership");
        ofx_test::RenderPixels result;
        for (std::size_t index = 0; index < ofx_test::kRenderReferenceCases.size(); ++index) {
            const auto& test = ofx_test::kRenderReferenceCases[index];
            const auto& row = rows.at(index);
            const auto input = source_words(test);
            require_render(row.at("id") == test.id && row.at("shape") == Json::array({5, 7, test.components}) &&
                               row.at("route") == test.resolvedRoute && row.at("reconstruction") == test.reconstruction &&
                               row.at("film") == test.film && row.at("paper") == (test.resolvedRoute % 2U ? "kodak_portra_endura" : "") &&
                               row.at("source_origin") == Json::array({test.origin.x, test.origin.y}) &&
                               row.at("grain") == 0 && row.at("glare") == 0 && row.at("unsharp_radius") == 0 &&
                               row.at("frame_rate") == 24 && row.at("frame_token") == 37 &&
                               row.at("output_color_flags") == 7 && row.at("source_bits") == input,
                           "spectral render fixture order or controls");
            const auto& output = row.at("output_bits");
            require_render(output.is_array() && output.size() == input.size(), "spectral render fixture word count");
            std::vector<float> pixels;
            pixels.reserve(output.size());
            for (const auto& word : output) {
                require_render(word.is_number_unsigned() && word.get<std::uint64_t>() <= std::numeric_limits<std::uint32_t>::max(), "spectral render fixture word type");
                pixels.push_back(std::bit_cast<float>(word.get<std::uint32_t>()));
            }
            RenderAssertions::compare_pixels(test.id, pixels, pixels);
            if (test.components == 4)
                for (std::size_t alpha = 3; alpha < output.size(); alpha += 4)
                    require_render(output[alpha] == input[alpha], "spectral render fixture alpha");
            result.emplace(test.id, std::move(pixels));
        }
        return result;
    }
} // namespace

namespace ofx_test {
    void configure_reference_render(RenderHost& fixture, DeviceImages& images, const RenderReferenceCase& test) {
        fixture.configure(images, test.origin);
        fixture.host().set_value(fixture.effect(), "ScanRoute", {std::string(test.route)});
        fixture.host().set_value(fixture.effect(), "FilmProfileKey", {std::string(test.film)});
        fixture.host().set_value(fixture.effect(), "SpectralUpsampling", {test.reconstruction});
        fixture.host().set_value(fixture.effect(), "GlareActive", {0});
    }

    RenderPixels read_render_reference(const char* path, const std::string& preset, bool check_controls) {
        std::ifstream stream(path);
        require_render(static_cast<bool>(stream), "spectral render fixture unavailable");
        const auto fixture = Json::parse(stream);
        const std::string referencePreset = preset.ends_with("-release") ? preset.substr(0, preset.size() - 8) + "-debug" : preset;
        require_render(fixture.at("schema") == "spectral-render-gpu-v1" && fixture.at("preset") == preset &&
                           fixture.at("reference_preset") == referencePreset,
                       "spectral render fixture identity");
        const auto& rows = fixture.at("rows");
        auto result = decode_rows(rows);
        if (check_controls) {
            for (bool reorder : {false, true}) {
                auto altered = rows;
                if (reorder)
                    std::swap(altered.at(1), altered.at(3));
                else
                    altered.at(1).at("id") = "altered-row";
                bool rejected = false;
                try {
                    (void)decode_rows(altered);
                } catch (const std::runtime_error&) {
                    rejected = true;
                }
                require_render(rejected, "altered spectral render fixture row accepted");
            }
        }
        return result;
    }
} // namespace ofx_test
