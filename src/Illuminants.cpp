#include "Illuminants.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <utility>

namespace Spectral {
    namespace {
        bool csv_pairs_cover_reference_band(
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

        bool csv_pairs_match_reference_axis(
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

        bool csv_pairs_mean_power_normalized(
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

    } // namespace

    Curve build_illuminant_curve(const std::vector<std::pair<float, float>>& pairs, std::string_view label) {
        Spectral::Curve c;
        if (pairs.empty()) {
            if (JTRACE_ENABLED(1)) {
                JTRACE("ILLUM", std::string("Failed to load illuminant CSV: ") + std::string(label));
            }
            // Return empty curve to signal failure
            return c;
        }
        if (!csv_pairs_match_reference_axis(pairs, label)) {
            return c;
        }
        if (!csv_pairs_mean_power_normalized(pairs, label)) {
            return c;
        }
        Spectral::assign_reference_axis(c.lambda_nm);
        c.linear.resize(Spectral::gShape.K);
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            c.linear[static_cast<size_t>(i)] = pairs[static_cast<size_t>(i)].second;
        }
        return c;
    }

    Curve build_tungsten_kg3_curve(const std::vector<std::pair<float, float>>& kg3_pairs, std::string_view label) {
        Spectral::Curve c;
        if (kg3_pairs.empty() || !csv_pairs_cover_reference_band(kg3_pairs, label)) {
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


    TungstenKg3LensInput::TungstenKg3LensInput(std::vector<float> bb, std::vector<std::pair<float, float>> filter)
        : blackbody(std::move(bb)), kg3(std::move(filter)) {}

    std::optional<TungstenKg3LensInput> prepare_tungsten_kg3_lens_input(
        const std::vector<std::pair<float, float>>& kg3_pairs, std::string_view label) {
        // 1) 3200K blackbody
        std::vector<float> bb(Spectral::gShape.K);
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            PlanckBlackbodySample sample{};
            sample.wavelengthNm = Spectral::gShape.wavelengths[i];
            sample.temperatureKelvin = 3200.0f;
            bb[i] = planck_blackbody(sample);
        }

        // 2) KG3 filter (resampled, no fallback)
        if (kg3_pairs.empty()) {
            if (JTRACE_ENABLED(1)) {
                JTRACE("ILLUM", std::string("Failed to load KG3 filter CSV: ") + std::string(label));
            }
            return std::nullopt;
        }
        csv_pairs_cover_reference_band(kg3_pairs, label);
        auto kg3_pinned = Spectral::resample_pairs_akima_to_reference_axis(kg3_pairs);
        if (kg3_pinned.empty()) {
            if (JTRACE_ENABLED(1)) {
                JTRACE("ILLUM", std::string("KG3 filter resample failed for: ") + std::string(label));
            }
            return std::nullopt;
        }

        return TungstenKg3LensInput{std::move(bb), std::move(kg3_pinned)};
    }

    Curve build_tungsten_kg3_lens_curve(TungstenKg3LensInput input,
                                        const std::vector<std::pair<float, float>>& lens_pairs,
                                        std::string_view label) {
        Curve c;
        // 3) Lens transmission (resampled, no fallback)
        if (lens_pairs.empty()) {
            if (JTRACE_ENABLED(1)) {
                JTRACE("ILLUM", std::string("Failed to load lens transmission CSV: ") + std::string(label));
            }
            c.lambda_nm.clear();
            c.linear.clear();
            return c;
        }
        csv_pairs_cover_reference_band(lens_pairs, label);
        auto lens_pinned = Spectral::resample_pairs_akima_to_reference_axis(lens_pairs);
        if (lens_pinned.empty()) {
            if (JTRACE_ENABLED(1)) {
                JTRACE("ILLUM", std::string("Lens transmission resample failed for: ") + std::string(label));
            }
            c.lambda_nm.clear();
            c.linear.clear();
            return c;
        }

        // 4) Multiply and mean-power normalize
        std::vector<float> combined(Spectral::gShape.K);
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            combined[i] = input.blackbody[i] * input.kg3[i].second * lens_pinned[i].second;
        }
        Spectral::mean_power_normalize(combined);

        // 5) Pin to shape without touching globals
        Spectral::assign_reference_axis(c.lambda_nm);
        c.linear = std::move(combined);
        return c;
    }

} // namespace Spectral
