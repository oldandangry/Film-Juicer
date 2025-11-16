#pragma once
#include <string>
#include <tuple>

// Minimal JSON loader for enlarger_neutral_ymc_filters.json.
// Extracts (Y,M,C) triple for a given paperKey, illuminantKey, negativeKey.
// Returns true if values were found; false otherwise.
// Path example: gDataDir + "profiles\\enlarger_neutral_ymc_filters.json"
bool load_enlarger_neutral_filters(
    const std::string& jsonPath,
    const std::string& paperKey,        // e.g. "kodak_2383_uc" or "kodak_2393_uc"
    const std::string& illuminantKey,   // e.g. "TH-KG3-L"
    const std::string& negativeKey,     // e.g. "kodak_vision3_250d_uc"
    std::tuple<float, float, float>& outYMC // filled with (Y,M,C)
);
