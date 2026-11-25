#pragma once

#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../ffi/print_illuminant_fixture.h"

namespace ofx_test {
    inline void compare_grain_capture(const std::filesystem::path& archive, const std::string& live_words) {
        std::ifstream expected(archive);
        if (!expected)
            throw std::runtime_error("grain capture archive unavailable");
        std::vector<std::string> original;
        original.reserve(198 * 38 + 36 * 34);
        std::vector<std::size_t> widths;
        widths.reserve(234);
        for (std::string line; std::getline(expected, line);) {
            std::istringstream words(line);
            const auto start = original.size();
            for (std::string word; words >> word;)
                original.push_back(word);
            const auto width = original.size() - start;
            const auto capture = widths.size() < 198;
            if (width != (capture ? 38u : 34u) || original.at(start) != (capture ? "GRAIN_CAPTURE" : "GRAIN_EDIT_CAPTURE"))
                throw std::runtime_error("invalid archived grain capture row");
            widths.push_back(width);
        }
        const auto frozen = PrintIlluminantFixtures::apply(archive, original);
        if (widths.size() != 234 || frozen.size() != original.size())
            throw std::runtime_error("archived grain capture membership");
        const auto recipe_word = [](const std::string& word) {
            std::uint64_t value = 0;
            const auto* const end = word.data() + word.size();
            const auto parsed = std::from_chars(word.data(), end, value);
            return parsed.ec == std::errc{} && parsed.ptr == end && value != 0;
        };
        std::istringstream actual(live_words);
        std::size_t index = 0;
        for (std::size_t row = 0; row < widths.size(); ++row) {
            for (std::size_t token = 0; token < widths[row]; ++token) {
                const auto frozen_word = frozen.at(index++).get<std::string>();
                std::string observed;
                const bool full_recipe = token + 1 == widths[row];
                if (!(actual >> observed) ||
                    (full_recipe ? !(recipe_word(frozen_word) && recipe_word(observed)) : observed != frozen_word))
                    throw std::runtime_error("grain capture differs at row " + std::to_string(row) +
                                             " word " + std::to_string(token));
            }
        }
        std::string extra;
        if (actual >> extra)
            throw std::runtime_error("extra grain capture word");
    }
} // namespace ofx_test
