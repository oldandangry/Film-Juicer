#pragma once

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace PrintIlluminantFixtures {
    // Earlier captures retain their bytes. Only explicitly adopted downstream
    // products receive ordered test/replace operations for the approved erff policy.
    inline nlohmann::json apply(const std::filesystem::path& path, const nlohmann::json& original) {
        const auto tests = std::filesystem::path(__FILE__).parent_path().parent_path();
        static const nlohmann::json updates = [tests] {
            std::ifstream stream(tests / "print/illuminant-fixtures/product-updates.json");
            const auto parsed = nlohmann::json::parse(stream);
            if (parsed.at("schema") != 1 || parsed.at("policy") != "print-erff-0.2.15-2026-10-20")
                throw std::runtime_error("unsupported print illuminant expectation policy");
            return parsed;
        }();
        const auto name = std::filesystem::relative(path, tests).generic_string();
        const auto& fixtures = updates.at("fixtures");
        if (!fixtures.contains(name))
            return original;
        const auto& patch = fixtures.at(name);
        if (patch.size() % 2 != 0)
            throw std::runtime_error("incomplete print illuminant expectation supplement");
        for (std::size_t i = 0; i < patch.size(); i += 2)
            if (patch[i].at("op") != "test" || patch[i + 1].at("op") != "replace" || patch[i].at("path") != patch[i + 1].at("path"))
                throw std::runtime_error("unbounded print illuminant expectation supplement");
        return original.patch(patch);
    }
} // namespace PrintIlluminantFixtures
