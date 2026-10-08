#pragma once

#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

// Earlier slice fixtures retain their bytes. A5 independently replayed native
// products supply exact test/replace operations for the approved erff decision.
inline nlohmann::json load_exposure_qualified_fixture(const std::filesystem::path& path) {
    std::ifstream source(path);
    const auto original = nlohmann::json::parse(source);
    const auto supplement = path.parent_path().parent_path() / "exposure" / "updates" / path.parent_path().filename() / path.filename();
    std::ifstream update(supplement);
    if (!update)
        throw std::runtime_error("missing qualified A5 fixture supplement: " + supplement.string());
    const auto packet = nlohmann::json::parse(update);
    if (packet.at("original_fixture").get<std::string>() != path.filename().string())
        throw std::runtime_error("wrong A5 fixture supplement");
    return original.patch(packet.at("patch"));
}
