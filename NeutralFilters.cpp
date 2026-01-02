#include "NeutralFilters.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <string>

#include "nlohmann/json.hpp"

namespace {
    using Json = nlohmann::json;

    std::string to_lower(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
            });
        return s;
    }

    const Json* find_child_case_insensitive(const Json& node, const std::string& key) {
        if (!node.is_object()) {
            return nullptr;
        }

        auto it = node.find(key);
        if (it != node.end()) {
            return &(*it);
        }

        std::string keyLower = to_lower(key);
        for (auto iter = node.cbegin(); iter != node.cend(); ++iter) {
            if (to_lower(iter.key()) == keyLower) {
                return &(*iter);
            }
        }
        return nullptr;
    }

    bool parse_array_triplet(const Json& arrNode, std::tuple<float, float, float>& outYMC) {
        if (!arrNode.is_array() || arrNode.size() < 3) {
            return false;
        }
        float vals[3] = {};
        for (size_t i = 0; i < 3; ++i) {
            const Json& element = arrNode[i];
            if (!(element.is_number_float() || element.is_number_integer())) {
                return false;
            }
            float v = static_cast<float>(element.get<double>());
            if (!std::isfinite(v)) {
                return false;
            }
            v = std::clamp(v, 0.0f, 1.0f);
            vals[i] = v;
        }

        outYMC = std::make_tuple(vals[0], vals[1], vals[2]);
        return true;
    }
}

bool load_enlarger_neutral_filters(
    const std::string& jsonPath,
    const std::string& paperKey,
    const std::string& illuminantKey,
    const std::string& negativeKey,
    std::tuple<float, float, float>& outYMC)
{
    if (paperKey.empty() || illuminantKey.empty() || negativeKey.empty()) {
        return false;
    }

    std::ifstream file(jsonPath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    Json root = Json::parse(file, nullptr, false);
    if (root.is_discarded()) {
        return false;
    }

    const Json* paperNode = find_child_case_insensitive(root, paperKey);
    if (!paperNode) {
        return false;
    }

    const Json* illuminantNode = find_child_case_insensitive(*paperNode, illuminantKey);
    if (!illuminantNode) {
        return false;
    }

    const Json* negativeNode = find_child_case_insensitive(*illuminantNode, negativeKey);
    if (!negativeNode) {
        return false;
    }
    return parse_array_triplet(*negativeNode, outYMC);
}
