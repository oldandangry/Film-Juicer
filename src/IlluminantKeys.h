#pragma once

#include <algorithm>
#include <cctype>
#include <initializer_list>
#include <string>
#include <string_view>

namespace IlluminantKeys {

    inline std::string normalize(std::string_view raw) {
        std::string result;
        result.reserve(raw.size());
        bool lastWasHyphen = true;
        for (char c : raw) {
            const unsigned char uc = static_cast<unsigned char>(c);
            if (std::isalnum(uc)) {
                result.push_back(static_cast<char>(std::toupper(uc)));
                lastWasHyphen = false;
            } else if (c == '-' || c == '_' || std::isspace(uc)) {
                if (!result.empty() && !lastWasHyphen) {
                    result.push_back('-');
                    lastWasHyphen = true;
                }
            }
        }
        while (!result.empty() && result.back() == '-') {
            result.pop_back();
        }
        if (result.empty()) {
            result.reserve(raw.size());
            for (char c : raw) {
                const unsigned char uc = static_cast<unsigned char>(c);
                if (!std::isspace(uc)) {
                    result.push_back(static_cast<char>(std::toupper(uc)));
                }
            }
        }
        return result;
    }

    inline bool equals(std::string_view a, std::string_view b) {
        return normalize(a) == normalize(b);
    }

    inline bool matches_any(std::string_view value, std::initializer_list<std::string_view> keys) {
        const std::string norm = normalize(value);
        for (std::string_view key : keys) {
            if (norm == normalize(key)) {
                return true;
            }
        }
        return false;
    }

} // namespace IlluminantKeys
