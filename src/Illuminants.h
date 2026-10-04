// Illuminants.h
#pragma once
#include <algorithm>
#include <cctype>
#include <cmath>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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


namespace Spectral {

    // --------------------------
    // Physics: Planck blackbody
    // --------------------------
    struct PlanckBlackbodySample {
        float wavelengthNm = 0.0f;
        float temperatureKelvin = 0.0f;
    };

    inline float planck_blackbody(const PlanckBlackbodySample& sample) {
        // Spectral radiance up to a scale factor; we normalize later.
        // lambda in meters
        const double lambda_m = static_cast<double>(sample.wavelengthNm) * 1e-9;
        const double c = 2.99792458e8;
        const double h = 6.62607015e-34;
        const double k = 1.380649e-23;

        const double c1 = 2.0 * h * c * c;
        const double c2 = h * c / k;
        const double denom = std::exp(c2 / (lambda_m * static_cast<double>(sample.temperatureKelvin))) - 1.0;
        const double L = (denom > 0.0) ? c1 / (std::pow(lambda_m, 5) * denom) : 0.0;
        return static_cast<float>(L);
    }

    Curve build_illuminant_curve(const std::vector<std::pair<float, float>>& pairs, std::string_view label);
    Curve build_tungsten_kg3_curve(const std::vector<std::pair<float, float>>& pairs, std::string_view label);

    // Complete call-local 3200 K/KG3 preparation permits conditional lens acquisition.
    class TungstenKg3LensInput final {
    public:
        TungstenKg3LensInput(TungstenKg3LensInput&&) = default;
        TungstenKg3LensInput& operator=(TungstenKg3LensInput&&) = default;

    private:
        friend std::optional<TungstenKg3LensInput> prepare_tungsten_kg3_lens_input(
            const std::vector<std::pair<float, float>>&, std::string_view);
        friend Curve build_tungsten_kg3_lens_curve(TungstenKg3LensInput,
                                                   const std::vector<std::pair<float, float>>&,
                                                   std::string_view);
        TungstenKg3LensInput(std::vector<float> bb, std::vector<std::pair<float, float>> filter);
        std::vector<float> blackbody;
        std::vector<std::pair<float, float>> kg3;
    };
    std::optional<TungstenKg3LensInput> prepare_tungsten_kg3_lens_input(
        const std::vector<std::pair<float, float>>& pairs, std::string_view label);
    Curve build_tungsten_kg3_lens_curve(TungstenKg3LensInput input,
                                        const std::vector<std::pair<float, float>>& lens,
                                        std::string_view label);

    inline Spectral::Curve build_curve_equal_energy_pinned() {
        Spectral::Curve c;
        Spectral::assign_reference_axis(c.lambda_nm);
        c.linear.assign(Spectral::gShape.K, 1.0f);
        return c;
    }


} // namespace Spectral
