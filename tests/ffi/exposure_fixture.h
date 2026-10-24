#pragma once

#include <bit>
#include <cstdint>
#include <vector>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

// Logical capture names select provenance and supplements independently of the
// shared physical payload. Expand frozen expected arrays before applying patches.
inline nlohmann::json load_shared_fixture(const std::filesystem::path& logicalPath, const char* manifestName) {
    std::ifstream manifestFile(logicalPath.parent_path() / manifestName);
    const auto manifest = nlohmann::json::parse(manifestFile);
    const auto name = logicalPath.filename().string();
    const auto payload = manifest.at("payloads").at(name).get<std::string>();
    std::ifstream source(logicalPath.parent_path() / payload);
    auto fixture = nlohmann::json::parse(source);
    if (manifest.contains("contexts")) {
        std::ifstream contextFile(logicalPath.parent_path() / manifest.at("contexts").at(name).get<std::string>());
        const auto context = nlohmann::json::parse(contextFile);
        for (const auto& [key, value] : context.items()) {
            if (fixture.contains(key))
                throw std::runtime_error("shared fixture context overlaps payload");
            fixture[key] = value;
        }
    }
    if (fixture.contains("expected_blocks")) {
        std::ifstream blocksFile(logicalPath.parent_path() / fixture.at("expected_blocks").get<std::string>());
        const auto blocks = nlohmann::json::parse(blocksFile);
        if (blocks.at("schema_version") != 1)
            throw std::runtime_error("unsupported shared expected-block schema");
        for (const char* group : {"recipes", "products"}) {
            if (!fixture.contains(group))
                continue;
            for (auto& row : fixture.at(group)) {
                if (!row.contains("expected"))
                    continue;
                auto& expected = row.at("expected");
                for (const char* field : {"scanner_tables", "exposure_tables", "sensitivity", "final_sensitivity"}) {
                    const auto reference = std::string(field) + "_record";
                    if (expected.contains(reference)) {
                        const auto& index = expected.at(reference);
                        if (!index.is_number_integer() || (!index.is_number_unsigned() && index.get<std::int64_t>() < 0))
                            throw std::runtime_error("invalid expected block reference");
                        const auto offset = index.get<std::uint64_t>();
                        const auto& records = blocks.at(field);
                        if (offset >= records.size())
                            throw std::runtime_error("invalid expected block reference");
                        expected[field] = records.at(static_cast<std::size_t>(offset));
                        expected.erase(reference);
                    }
                }
            }
        }
        fixture.erase("expected_blocks");
    }
    return fixture;
}

inline nlohmann::json load_exposure_product_fixture(const std::filesystem::path& path) {
    std::ifstream source(path);
    auto fixture = nlohmann::json::parse(source);
    std::ifstream values(path.parent_path() / "product-values.json");
    auto records = nlohmann::json::parse(values);
    if (records.at("schema_version") != 1)
        throw std::runtime_error("unsupported A5 product fixture schema");
    fixture["expected_records"] = std::move(records.at("expected_records"));
    return fixture;
}

// Earlier slice fixtures retain their bytes. A5 independently replayed native
// products supply exact test/replace operations for the approved erff decision.
inline nlohmann::json load_exposure_qualified_fixture(const std::filesystem::path& path) {
    const auto original = [&] {
        if (path.parent_path().filename() == "color") {
            const auto name = path.filename().string();
            const auto* manifest = name.starts_with("input-color-") ? "input-color-manifest.json" : name.starts_with("cat02-") ? "cat02-manifest.json"
                                                                                                                               : "manifest.json";
            return load_shared_fixture(path, manifest);
        }
        std::ifstream source(path);
        return nlohmann::json::parse(source);
    }();
    const auto supplement = path.parent_path().parent_path() / "exposure" / "updates" / path.parent_path().filename() / path.filename();
    std::ifstream update(supplement);
    if (!update)
        throw std::runtime_error("missing qualified A5 fixture supplement: " + supplement.string());
    const auto packet = nlohmann::json::parse(update);
    if (packet.at("original_fixture").get<std::string>() != path.filename().string())
        throw std::runtime_error("wrong A5 fixture supplement");
    return original.patch(packet.at("patch"));
}

// Exact independent Gaussian words for real scanner/grain/FIR-DIR device readers.
inline std::vector<std::uint32_t> gaussian_kernel_fixture_words(const std::filesystem::path& logical, int radius, float sigma) {
    const auto fixture = load_shared_fixture(logical, "manifest.json");
    for (const auto& row : fixture.at("cases")) {
        if (row.at("radius") == radius && row.at("sigma_bits") == std::bit_cast<std::uint32_t>(sigma)) {
            auto words = row.at("weights").get<std::vector<std::uint32_t>>();
            if (words.size() != static_cast<std::size_t>(radius) * 2u + 1u || row.at("count").get<std::size_t>() != words.size())
                throw std::runtime_error("Gaussian frozen metadata mismatch");
            return words;
        }
    }
    throw std::runtime_error("Gaussian actual kernel has no frozen expectation");
}
