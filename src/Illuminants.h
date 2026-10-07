// Illuminants.h
#pragma once
#include <cctype>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>

#include "SpectralData.h"

namespace IlluminantKeys {

    // Canonicalizes user/profile illuminant labels so lookup code can compare a small
    // normalized vocabulary instead of carrying casing and separator variants everywhere.
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


namespace JuicerAssets {
    class CsvRows;
}
struct FjIlluminantLens;
// FJ_TEMP_BRIDGE: native illuminant value/lens projection; remove S4.E.
namespace JuicerIlluminant {
    class Lens final {
    public:
        ~Lens();
        Lens(Lens&& lens) noexcept;
        Lens(const Lens&) = delete;
        Lens& operator=(const Lens&) = delete;
        Lens& operator=(Lens&&) = delete;

    private:
        friend std::optional<Lens> prepare_lens(const JuicerAssets::CsvRows&, std::string_view);
        friend Spectral::Curve finish_lens(Lens, const JuicerAssets::CsvRows&, std::string_view);
        explicit Lens(FjIlluminantLens* owner) noexcept;
        FjIlluminantLens* _owner;
    };
    Spectral::Curve from_samples(const JuicerAssets::CsvRows& rows, std::string_view label);
    Spectral::Curve blackbody(float temperatureKelvin);
    Spectral::Curve equal_energy();
    Spectral::Curve tungsten_kg3(const JuicerAssets::CsvRows& rows, std::string_view label);
    std::optional<Lens> prepare_lens(const JuicerAssets::CsvRows& rows, std::string_view label);
    Spectral::Curve finish_lens(Lens lens, const JuicerAssets::CsvRows& rows, std::string_view label);
} // namespace JuicerIlluminant
