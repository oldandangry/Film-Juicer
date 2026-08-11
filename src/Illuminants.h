// Illuminants.h
#pragma once
#include <algorithm>
#include <cctype>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <sstream>
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

    inline bool csv_pairs_cover_reference_band(
        const std::vector<std::pair<float, float>>& pairs,
        std::string_view label) {
        if (pairs.empty()) {
            return false;
        }

        float minLambda = std::numeric_limits<float>::infinity();
        float maxLambda = -std::numeric_limits<float>::infinity();
        for (const auto& sample : pairs) {
            if (!std::isfinite(sample.first)) {
                continue;
            }
            minLambda = std::min(minLambda, sample.first);
            maxLambda = std::max(maxLambda, sample.first);
        }

        if (!(std::isfinite(minLambda) && std::isfinite(maxLambda))) {
            return false;
        }

        constexpr float kMarginNm = 10.0f;
        const float minNeeded = Spectral::kLambdaMin - kMarginNm;
        const float maxNeeded = Spectral::kLambdaMax + kMarginNm;
        const bool coversMin = (minLambda <= minNeeded);
        const bool coversMax = (maxLambda >= maxNeeded);

        if (!coversMin || !coversMax) {
            if (JTRACE_ENABLED(1)) {
                std::ostringstream oss;
                oss << "CSV coverage warning (" << label << "): ";
                if (!coversMin) {
                    oss << "start=" << minLambda << "nm (need <= " << minNeeded << "nm)";
                }
                if (!coversMax) {
                    if (!coversMin) {
                        oss << ", ";
                    }
                    oss << "end=" << maxLambda << "nm (need >= " << maxNeeded << "nm)";
                }
                JTRACE("ILLUM", oss.str());
            }
        }

        return coversMin && coversMax;
    }

    inline bool csv_pairs_match_reference_axis(
        const std::vector<std::pair<float, float>>& pairs,
        std::string_view label) {
        if (Spectral::samples_follow_reference_axis(pairs)) {
            return true;
        }
        if (JTRACE_ENABLED(1)) {
            std::ostringstream oss;
            oss << "Illuminant CSV axis mismatch (" << label << "): expected agx reference grid";
            JTRACE("ILLUM", oss.str());
        }
        return false;
    }

    inline bool csv_pairs_mean_power_normalized(
        const std::vector<std::pair<float, float>>& pairs,
        std::string_view label) {
        if (pairs.size() != static_cast<size_t>(Spectral::SpectralShape::K)) {
            if (JTRACE_ENABLED(1)) {
                std::ostringstream oss;
                oss << "Illuminant CSV sample count mismatch (" << label << "): expected "
                    << Spectral::SpectralShape::K << " samples";
                JTRACE("ILLUM", oss.str());
            }
            return false;
        }

        double sum = 0.0;
        for (const auto& sample : pairs) {
            if (!std::isfinite(sample.second)) {
                if (JTRACE_ENABLED(1)) {
                    std::ostringstream oss;
                    oss << "Illuminant CSV contains non-finite sample (" << label << ")";
                    JTRACE("ILLUM", oss.str());
                }
                return false;
            }
            sum += static_cast<double>(sample.second);
        }

        const double mean = sum / static_cast<double>(pairs.size());
        constexpr double kMeanTolerance = 1e-5;
        if (!std::isfinite(mean) || std::abs(mean - 1.0) > kMeanTolerance) {
            if (JTRACE_ENABLED(1)) {
                std::ostringstream oss;
                oss << "Illuminant CSV mean-power mismatch (" << label << "): mean=" << mean;
                JTRACE("ILLUM", oss.str());
            }
            return false;
        }
        return true;
    }

    // --------------------------
    // Builders for illuminants
    // --------------------------
    inline Spectral::Curve build_curve_from_csv_pinned(const std::string& csvPath) {
        Spectral::Curve c;
        std::vector<std::pair<float, float>> pairs;
        try {
            pairs = Spectral::load_csv_pairs(csvPath);
        } catch (...) {
            pairs.clear();
        }
        if (pairs.empty()) {
            if (JTRACE_ENABLED(1)) {
                JTRACE("ILLUM", std::string("Failed to load illuminant CSV: ") + csvPath);
            }
            // Return empty curve to signal failure
            return c;
        }
        if (!csv_pairs_match_reference_axis(pairs, csvPath)) {
            return c;
        }
        if (!csv_pairs_mean_power_normalized(pairs, csvPath)) {
            return c;
        }
        Spectral::assign_reference_axis(c.lambda_nm);
        c.linear.resize(Spectral::gShape.K);
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            c.linear[static_cast<size_t>(i)] = pairs[static_cast<size_t>(i)].second;
        }
        return c;
    }

    inline Spectral::Curve build_curve_D65_pinned(const std::string& csvPath) {
        return build_curve_from_csv_pinned(csvPath);
    }

    inline Spectral::Curve build_curve_D55_pinned(const std::string& csvPath) {
        return build_curve_from_csv_pinned(csvPath);
    }

    inline Spectral::Curve build_curve_D50_pinned(const std::string& csvPath) {
        return build_curve_from_csv_pinned(csvPath);
    }

    inline Spectral::Curve build_curve_T_pinned(const std::string& csvPath) {
        return build_curve_from_csv_pinned(csvPath);
    }

    inline Spectral::Curve build_curve_K75P_pinned(const std::string& csvPath) {
        return build_curve_from_csv_pinned(csvPath);
    }

    inline Spectral::Curve build_curve_TH_KG3_L_pinned(
        const std::string& kg3CsvPath, const std::string& lensCsvPath) {
        Spectral::Curve c;
        // 1) 3200K blackbody
        std::vector<float> bb(Spectral::gShape.K);
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            PlanckBlackbodySample sample{};
            sample.wavelengthNm = Spectral::gShape.wavelengths[i];
            sample.temperatureKelvin = 3200.0f;
            bb[i] = planck_blackbody(sample);
        }

        // 2) KG3 filter (resampled, no fallback)
        std::vector<std::pair<float, float>> kg3_pairs;
        try {
            kg3_pairs = Spectral::load_csv_pairs(kg3CsvPath);
        } catch (...) {
            kg3_pairs.clear();
        }
        if (kg3_pairs.empty()) {
            if (JTRACE_ENABLED(1)) {
                JTRACE("ILLUM", std::string("Failed to load KG3 filter CSV: ") + kg3CsvPath);
            }
            c.lambda_nm.clear();
            c.linear.clear();
            return c;
        }
        csv_pairs_cover_reference_band(kg3_pairs, kg3CsvPath);
        auto kg3_pinned = Spectral::resample_pairs_akima_to_reference_axis(kg3_pairs);
        if (kg3_pinned.empty()) {
            if (JTRACE_ENABLED(1)) {
                JTRACE("ILLUM", std::string("KG3 filter resample failed for: ") + kg3CsvPath);
            }
            c.lambda_nm.clear();
            c.linear.clear();
            return c;
        }

        // 3) Lens transmission (resampled, no fallback)
        std::vector<std::pair<float, float>> lens_pairs;
        try {
            lens_pairs = Spectral::load_csv_pairs(lensCsvPath);
        } catch (...) {
            lens_pairs.clear();
        }
        if (lens_pairs.empty()) {
            if (JTRACE_ENABLED(1)) {
                JTRACE("ILLUM", std::string("Failed to load lens transmission CSV: ") + lensCsvPath);
            }
            c.lambda_nm.clear();
            c.linear.clear();
            return c;
        }
        csv_pairs_cover_reference_band(lens_pairs, lensCsvPath);
        auto lens_pinned = Spectral::resample_pairs_akima_to_reference_axis(lens_pairs);
        if (lens_pinned.empty()) {
            if (JTRACE_ENABLED(1)) {
                JTRACE("ILLUM", std::string("Lens transmission resample failed for: ") + lensCsvPath);
            }
            c.lambda_nm.clear();
            c.linear.clear();
            return c;
        }

        // 4) Multiply and mean-power normalize
        std::vector<float> combined(Spectral::gShape.K);
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            combined[i] = bb[i] * kg3_pinned[i].second * lens_pinned[i].second;
        }
        Spectral::mean_power_normalize(combined);

        // 5) Pin to shape without touching globals
        Spectral::assign_reference_axis(c.lambda_nm);
        c.linear = std::move(combined);
        return c;
    }

    inline Spectral::Curve build_curve_TH_KG3_pinned(const std::string& kg3CsvPath) {
        Spectral::Curve c;
        std::vector<std::pair<float, float>> kg3_pairs;
        try {
            kg3_pairs = Spectral::load_csv_pairs(kg3CsvPath);
        } catch (...) {
            kg3_pairs.clear();
        }
        if (kg3_pairs.empty() || !csv_pairs_cover_reference_band(kg3_pairs, kg3CsvPath)) {
            return c;
        }
        const auto kg3_pinned = Spectral::resample_pairs_akima_to_reference_axis(kg3_pairs);
        if (kg3_pinned.size() != static_cast<std::size_t>(Spectral::gShape.K)) {
            return c;
        }

        std::vector<float> combined(static_cast<std::size_t>(Spectral::gShape.K));
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            PlanckBlackbodySample sample{};
            sample.wavelengthNm = Spectral::gShape.wavelengths[i];
            sample.temperatureKelvin = 3400.0f;
            combined[static_cast<std::size_t>(i)] =
                planck_blackbody(sample) *
                kg3_pinned[static_cast<std::size_t>(i)].second;
        }
        Spectral::mean_power_normalize(combined);
        Spectral::assign_reference_axis(c.lambda_nm);
        c.linear = std::move(combined);
        return c;
    }


    inline Spectral::Curve build_curve_equal_energy_pinned() {
        Spectral::Curve c;
        Spectral::assign_reference_axis(c.lambda_nm);
        c.linear.assign(Spectral::gShape.K, 1.0f);
        return c;
    }


} // namespace Spectral
