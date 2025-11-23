// SpectralData.h
// Data structures, curves, I/O operations, and global setters for spectral processing
#pragma once

#include <cmath>
#include <algorithm>
#include <vector>
#include <array>
#include <utility>
#include <string>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <initializer_list>
#include <cstdint>
#include "SpectralContext.h"
#include "AkimaInterpolator.h"
#include "Logging.h"
#include "Hash.h"

namespace Spectral {

    // =========================================================================
    // Global references to SpectralContext (for default parameters and inline functions)
    // =========================================================================

    inline SpectralShape& gShape = context().shape;
    inline Curve& gIlluminantCurve = context().illuminantCurve;
    inline Curve& gSensBlue = context().sensBlue;
    inline Curve& gSensGreen = context().sensGreen;
    inline Curve& gSensRed = context().sensRed;
    inline Curve& gEpsY = context().epsY;
    inline Curve& gEpsM = context().epsM;
    inline Curve& gEpsC = context().epsC;
    inline Curve& gXBar = context().xBar;
    inline Curve& gYBar = context().yBar;
    inline Curve& gZBar = context().zBar;
    inline Curve& gBaseMin = context().baseMin;
    inline Curve& gBaseMid = context().baseMid;
    inline bool& gHasBaseline = context().hasBaseline;

    // =========================================================================
    // Constants
    // =========================================================================

    // DWG working space white point (D65)
    inline constexpr float gDWG_WhitePoint_XYZ[3] = {
        0.950455f, 1.0f, 1.089058f
    };

    // ============================================================================
    // SpectralTables: Per-instance spectral tables (consolidated from SpectralTables.h)
    // ============================================================================

    struct SpectralTables {
        // Wavelength axis
        std::vector<float> lambda;
        int   K = 0;
        float deltaLambda = 5.0f;
        float invYn = 1.0f;
        float whiteXYZ[3] = { 0.0f, 0.0f, 0.0f };

        // Reference illuminant white point (for chromatic adaptation in SPD reconstruction)
        float refIllumWhiteXYZ[3] = { 0.95047f, 1.0f, 1.08883f };  // D65 default

        // Illuminant-weighted CMFs (Ax, Ay, Az) and raw CMFs
        std::vector<float> Ax, Ay, Az;
        std::vector<float> Xbar, Ybar, Zbar;

        // Dye extinction tables
        std::vector<float> epsY, epsM, epsC;

        // Baseline (optional) and flag
        std::vector<float> baseMin, baseMid;
        bool hasBaseline = false;

        // Reference density used to compute baseline interpolation mix (0 => use baseMin).
        float baselineMixReference = 0.0f;

        // Hashes used for scanner caches (agx parity)
        std::uint64_t illuminantHash = 0;
        std::uint64_t tablesHash = 0;
    };

    // ============================================================================
    // CMFTriplets: Color matching function triplets
    // ============================================================================

    struct CMFTriplets {
        std::vector<std::pair<float, float>> xbar;
        std::vector<std::pair<float, float>> ybar;
        std::vector<std::pair<float, float>> zbar;
    };

    // ============================================================================
    // Helper/Utility Functions
    // ============================================================================

    inline void log_spectral_warning(const std::string& message) {
        JTRACE("SPECTRAL", "WARN: " + message);
    }

    inline void log_resample_failure(const char* context,
        std::initializer_list<std::pair<const char*, bool>> states) {
        std::ostringstream oss;
        oss << context;
        if (!states.size()) {
            log_spectral_warning(oss.str());
            return;
        }
        oss << " (";
        bool first = true;
        for (const auto& state : states) {
            if (!first) oss << ", ";
            first = false;
            oss << state.first << '=' << (state.second ? "ok" : "empty");
        }
        oss << ')';
        log_spectral_warning(oss.str());
    }

    inline void mean_power_normalize(std::vector<float>& spd) {
        if (spd.empty()) return;
        double sum = 0.0;
        for (float v : spd) sum += static_cast<double>(v);
        double mean = sum / static_cast<double>(spd.size());
        if (mean > 0.0) {
            for (float& v : spd) v = static_cast<float>(v / mean);
        }
    }

    // ============================================================================
    // Curve Sampling and Resampling Functions
    // ============================================================================

    // Linear sample arbitrary (lambda, value) pairs at 'lambda' with endpoint clamp.
    inline float sample_linear_pairs(const std::vector<std::pair<float, float>>& pairs, float lambda) {
        const size_t n = pairs.size();
        if (n == 0) return 0.0f;
        if (n == 1) return pairs.front().second;
        if (lambda <= pairs.front().first) return pairs.front().second;
        if (lambda >= pairs.back().first)  return pairs.back().second;
        size_t i1 = 1;
        while (i1 < n && pairs[i1].first < lambda) ++i1;
        const size_t i0 = i1 - 1;
        const float x0 = pairs[i0].first;
        const float x1 = pairs[i1].first;
        if (!(std::isfinite(x0) && std::isfinite(x1)) || x1 <= x0) {
            return pairs[i0].second;
        }
        float y0 = pairs[i0].second;
        float y1 = pairs[i1].second;
        if (!std::isfinite(y0)) {
            y0 = std::isfinite(y1) ? y1 : 0.0f;
        }
        if (!std::isfinite(y1)) {
            y1 = y0;
        }
        const float t = (lambda - x0) / (x1 - x0);
        return y0 + t * (y1 - y0);
    }

    enum class ReferenceResampleKernel {
        Linear,
        Akima
    };

    inline void sanitize_pairs_for_resample(
        const std::vector<std::pair<float, float>>& inPairs,
        std::vector<std::pair<float, float>>& sanitized) {
        sanitized.clear();
        if (inPairs.empty()) return;

        std::vector<std::pair<float, float>> sorted = inPairs;
        std::sort(sorted.begin(), sorted.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

        constexpr float kLambdaDedupEps = 1e-5f;
        sanitized.reserve(sorted.size());
        for (const auto& sample : sorted) {
            if (!std::isfinite(sample.first) || !std::isfinite(sample.second)) {
                continue;
            }
            if (!sanitized.empty() &&
                std::abs(sample.first - sanitized.back().first) <= kLambdaDedupEps) {
                sanitized.back().second = sample.second;
                continue;
            }
            sanitized.emplace_back(sample.first, sample.second);
        }
    }

    inline bool samples_follow_reference_axis(const std::vector<std::pair<float, float>>& samples) {
        if (samples.size() != static_cast<size_t>(SpectralShape::K)) {
            return false;
        }
        constexpr float kAxisMatchTolerance = 1e-3f;
        for (size_t i = 0; i < samples.size(); ++i) {
            if (std::abs(samples[i].first - gShape.wavelengths[i]) > kAxisMatchTolerance) {
                return false;
            }
        }
        return true;
    }

    inline std::vector<std::pair<float, float>> resample_pairs_to_axis_impl(
        const std::vector<std::pair<float, float>>& inPairs,
        ReferenceResampleKernel kernel)
    {
        std::vector<std::pair<float, float>> out;
        if (inPairs.empty()) {
            return out;
        }
        const SpectralShape& axis = gShape;

        std::vector<std::pair<float, float>> sanitized;
        sanitize_pairs_for_resample(inPairs, sanitized);
        if (sanitized.empty()) {
            return out;
        }

        if (samples_follow_reference_axis(sanitized)) {
            return sanitized;
        }

        const bool useAkima = (kernel == ReferenceResampleKernel::Akima);
        Interpolation::AkimaInterpolator akima;
        bool akimaOk = false;
        if (useAkima && sanitized.size() >= 2) {
            std::vector<float> xs;
            std::vector<float> ys;
            xs.reserve(sanitized.size());
            ys.reserve(sanitized.size());
            for (const auto& sample : sanitized) {
                xs.push_back(sample.first);
                ys.push_back(sample.second);
            }
            akimaOk = akima.build(xs, ys);
        }

        out.reserve(static_cast<size_t>(axis.K));
        for (int i = 0; i < axis.K; ++i) {
            const float lambda = axis.wavelengths[i];
            float value = sample_linear_pairs(sanitized, lambda);
            if (useAkima && akimaOk) {
                const float akimaValue = akima.evaluate(lambda);
                if (std::isfinite(akimaValue)) {
                    value = akimaValue;
                }
            }
            if (!std::isfinite(value)) {
                return {};
            }
            out.emplace_back(lambda, value);
        }
        return out;
    }

    inline std::vector<std::pair<float, float>> resample_pairs_linear_to_reference_axis(
        const std::vector<std::pair<float, float>>& inPairs) {
        return resample_pairs_to_axis_impl(inPairs, ReferenceResampleKernel::Linear);
    }

    inline std::vector<std::pair<float, float>> resample_pairs_akima_to_reference_axis(
        const std::vector<std::pair<float, float>>& inPairs) {
        return resample_pairs_to_axis_impl(inPairs, ReferenceResampleKernel::Akima);
    }

    // Build a curve pinned to the reference axis from linear pairs.
    inline bool build_curve_on_reference_axis_from_linear_pairs(
        Curve& curve,
        const std::vector<std::pair<float, float>>& pairs,
        ReferenceResampleKernel kernel = ReferenceResampleKernel::Linear)
    {
        const bool useAkima = (kernel == ReferenceResampleKernel::Akima);
        std::vector<std::pair<float, float>> resampled =
            useAkima
            ? resample_pairs_akima_to_reference_axis(pairs)
            : resample_pairs_linear_to_reference_axis(pairs);

        curve.lambda_nm.clear();
        curve.linear.clear();
        if (resampled.empty()) {
            return false;
        }

        curve.lambda_nm.reserve(resampled.size());
        curve.linear.reserve(resampled.size());
        for (const auto& sample : resampled) {
            const float lambda = sample.first;
            const float value = sample.second;
            if (!std::isfinite(lambda) || !std::isfinite(value)) {
                curve.lambda_nm.clear();
                curve.linear.clear();
                return false;
            }
            curve.lambda_nm.push_back(lambda);
            curve.linear.push_back(value < 0.0f ? 0.0f : value);
        }
        return !curve.linear.empty();
    }

    // Build a curve pinned to the reference axis from log10 pairs (log → linear).
    inline bool build_curve_on_reference_axis_from_log10_pairs(
        Curve& curve,
        const std::vector<std::pair<float, float>>& log10pairs,
        ReferenceResampleKernel kernel = ReferenceResampleKernel::Linear)
    {
        curve.lambda_nm.clear();
        curve.linear.clear();

        if (log10pairs.empty()) {
            return false;
        }

        std::vector<std::pair<float, float>> linearPairs;
        linearPairs.reserve(log10pairs.size());
        for (const auto& sample : log10pairs) {
            const float lambda = sample.first;
            if (!std::isfinite(lambda)) {
                continue;
            }
            float linear = 0.0f;
            if (std::isfinite(sample.second)) {
                linear = std::pow(10.0f, sample.second);
                if (!std::isfinite(linear) || linear < 0.0f) {
                    linear = 0.0f;
                }
            }
            linearPairs.emplace_back(lambda, linear);
        }

        if (linearPairs.empty()) {
            return false;
        }

        const bool ok = build_curve_on_reference_axis_from_linear_pairs(
            curve, linearPairs, kernel);
        if (!ok) {
            curve.lambda_nm.clear();
            curve.linear.clear();
        }
        return ok;
    }

    inline void sort_and_build(Curve& curve, const std::vector<std::pair<float, float>>& pairs) {
        if (pairs.empty()) {
            curve.lambda_nm.clear();
            curve.linear.clear();
            return;
        }
        std::vector<std::pair<float, float>> sorted = pairs;
        std::sort(sorted.begin(), sorted.end(),
            [](auto& a, auto& b) { return a.first < b.first; });

        std::vector<std::pair<float, float>> filtered;
        filtered.reserve(sorted.size());
        for (auto& p : sorted) {
            if (!std::isfinite(p.first) || !std::isfinite(p.second)) {
                continue;
            }
            filtered.emplace_back(p.first, p.second < 0.0f ? 0.0f : p.second);
        }

        if (filtered.empty()) {
            curve.lambda_nm.clear();
            curve.linear.clear();
            return;
        }

        curve.lambda_nm.resize(filtered.size());
        curve.linear.resize(filtered.size());
        for (size_t i = 0; i < filtered.size(); ++i) {
            curve.lambda_nm[i] = filtered[i].first;
            curve.linear[i] = filtered[i].second;
        }
    }

    // ============================================================================
    // CSV I/O Functions
    // ============================================================================

    // Utility: Load wavelength/value pairs from a CSV file
    inline std::vector<std::pair<float, float>> load_csv_pairs(const std::string& path) {
        std::vector<std::pair<float, float>> data;
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open CSV: " + path);
        }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            // Strip comments starting at # or ;
            auto strip_comment = [&](char c) {
                size_t p = line.find(c);
                if (p != std::string::npos) line.erase(p);
                };
            strip_comment('#');
            strip_comment(';');
            std::istringstream ss(line);
            float x = 0.0f, y = 0.0f;
            if (!(ss >> x)) continue;
            // Skip optional comma/semicolon
            while (ss.peek() == ',' || ss.peek() == ';') ss.get();
            if (!(ss >> y)) continue;
            data.emplace_back(x, y);
        }
        return data;
    }

    inline CMFTriplets load_csv_triplets(const std::string& path) {
        CMFTriplets out;
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open CSV: " + path);
        }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            auto strip_comment = [&](char c) {
                size_t p = line.find(c);
                if (p != std::string::npos) line.erase(p);
                };
            strip_comment('#');
            strip_comment(';');

            std::istringstream ss(line);
            float l = 0.0f, xv = 0.0f, yv = 0.0f, zv = 0.0f;

            if (!(ss >> l)) continue;
            while (ss.peek() == ',' || ss.peek() == ';') ss.get();

            if (!(ss >> xv)) continue;
            while (ss.peek() == ',' || ss.peek() == ';') ss.get();

            if (!(ss >> yv)) continue;
            while (ss.peek() == ',' || ss.peek() == ';') ss.get();

            if (!(ss >> zv)) continue;

            out.xbar.emplace_back(l, xv);
            out.ybar.emplace_back(l, yv);
            out.zbar.emplace_back(l, zv);
        }
        return out;
    }

    // Load a single-column CSV of wavelengths (nm). Ignores comments (# or ;) and empty lines.
    inline std::vector<float> load_csv_single(const std::string& path) {
        std::vector<float> data;
        std::ifstream file(path);
        if (!file.is_open()) {
            // Return empty to allow fallback without throwing
            return data;
        }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            // Strip comments
            auto strip_comment = [&](char c) {
                size_t p = line.find(c);
                if (p != std::string::npos) line.erase(p);
                };
            strip_comment('#');
            strip_comment(';');
            std::istringstream ss(line);
            float v = 0.0f;
            if (!(ss >> v)) continue;
            data.push_back(v);
        }
        return data;
    }

    // ============================================================================
    // SpectralShape Management
    // ============================================================================

    inline SpectralShape make_reference_spectral_shape() {
        return SpectralShape{};
    }

    inline bool spectral_shape_matches_reference(const SpectralShape& s) {
        for (int i = 0; i < SpectralShape::K; ++i) {
            const float expected = kLambdaMin + static_cast<float>(i) * kDelta;
            if (std::abs(s.wavelengths[static_cast<size_t>(i)] - expected) > 1e-3f) {
                return false;
            }
        }
        return true;
    }

    inline void assign_reference_axis(std::vector<float>& lambda) {
        lambda.assign(gShape.wavelengths.begin(), gShape.wavelengths.end());
    }

    inline void lock_shape_to_reference_axis() {
        gShape = make_reference_spectral_shape();
        increment_shape_version();
        mark_spectral_tables_dirty();
    }

    inline bool cmf_triplets_match_reference_axis(const CMFTriplets& cmf) {
        const auto matches_reference = [](const std::vector<std::pair<float, float>>& axis) {
            if (axis.size() != static_cast<size_t>(SpectralShape::K)) {
                return false;
            }
            for (int i = 0; i < SpectralShape::K; ++i) {
                const float lambda = axis[static_cast<size_t>(i)].first;
                const float expected = kLambdaMin + static_cast<float>(i) * kDelta;
                if (std::abs(lambda - expected) > 1e-3f) {
                    return false;
                }
            }
            return true;
            };

        return matches_reference(cmf.xbar) &&
            matches_reference(cmf.ybar) &&
            matches_reference(cmf.zbar);
    }

    // ============================================================================
    // Global Curve Setters
    // ============================================================================

    // Install a custom illuminant from wavelength/value pairs (linear power)
    inline void set_illuminant_from_pairs(const std::vector<std::pair<float, float>>& pairs) {
        // Build pinned curve
        Curve newCurve;
        const bool ok = build_curve_on_reference_axis_from_linear_pairs(newCurve, pairs);
        if (!ok) {
            log_spectral_warning("Illuminant resample failed (all samples filtered)");
            return;
        }

        // If identical to current gIlluminantCurve, skip dirtying
        const bool sameSize =
            newCurve.linear.size() == gIlluminantCurve.linear.size() &&
            newCurve.lambda_nm.size() == gIlluminantCurve.lambda_nm.size();

        bool identical = sameSize;
        if (identical) {
            // Compare lambda axis first (shapes should match)
            for (size_t i = 0; i < newCurve.lambda_nm.size(); ++i) {
                if (newCurve.lambda_nm[i] != gIlluminantCurve.lambda_nm[i]) {
                    identical = false; break;
                }
            }
            // Compare spectrum with a tight epsilon (mean‑power normalization should make this exact or very close)
            if (identical) {
                constexpr float eps = 1e-6f;
                for (size_t i = 0; i < newCurve.linear.size(); ++i) {
                    if (std::fabs(newCurve.linear[i] - gIlluminantCurve.linear[i]) > eps) {
                        identical = false; break;
                    }
                }
            }
        }

        if (identical) {
            return; // no change
        }

        // Install and dirty
        context().illuminantCurve = std::move(newCurve);
        mark_spectral_tables_dirty();
    }

    inline void set_layer_sensitivities(
        const std::vector<std::pair<float, float>>& blue_log10,
        const std::vector<std::pair<float, float>>& green_log10,
        const std::vector<std::pair<float, float>>& red_log10)
    {
        const bool bOk = build_curve_on_reference_axis_from_log10_pairs(gSensBlue, blue_log10);
        const bool gOk = build_curve_on_reference_axis_from_log10_pairs(gSensGreen, green_log10);
        const bool rOk = build_curve_on_reference_axis_from_log10_pairs(gSensRed, red_log10);
        if (!(bOk && gOk && rOk)) {
            log_resample_failure("Layer sensitivity resample failed",
                { {"B", bOk}, {"G", gOk}, {"R", rOk} });
        }
    }

    // --- Dye extinction curves ---
    inline void set_dye_extinctions_linear(
        const std::vector<std::pair<float, float>>& y_linear,
        const std::vector<std::pair<float, float>>& m_linear,
        const std::vector<std::pair<float, float>>& c_linear) {
        const bool yOk = build_curve_on_reference_axis_from_linear_pairs(gEpsY, y_linear);
        const bool mOk = build_curve_on_reference_axis_from_linear_pairs(gEpsM, m_linear);
        const bool cOk = build_curve_on_reference_axis_from_linear_pairs(gEpsC, c_linear);
        if (!(yOk && mOk && cOk)) {
            log_resample_failure("Dye extinction resample failed",
                { {"Y", yOk}, {"M", mOk}, {"C", cOk} });
        }
    }

    // set_dye_extinctions_log10 is NOT for agx-emulsion dye_density_* CSVs.
    // Those CSVs are linear optical densities already.
    inline void set_dye_extinctions_log10(
        const std::vector<std::pair<float, float>>& y_log10,
        const std::vector<std::pair<float, float>>& m_log10,
        const std::vector<std::pair<float, float>>& c_log10) {
        const bool yOk = build_curve_on_reference_axis_from_log10_pairs(gEpsY, y_log10);
        const bool mOk = build_curve_on_reference_axis_from_log10_pairs(gEpsM, m_log10);
        const bool cOk = build_curve_on_reference_axis_from_log10_pairs(gEpsC, c_log10);
        if (!(yOk && mOk && cOk)) {
            log_resample_failure("Dye extinction log10 resample failed",
                { {"Y", yOk}, {"M", mOk}, {"C", cOk} });
        }
    }

    // -------------------------------------------------------------------------
    // CIE 1931 CMFs (Gaussian fallback).
    // -------------------------------------------------------------------------
    inline void set_cie_1931_2deg_cmf(
        const std::vector<std::pair<float, float>>& xbar,
        const std::vector<std::pair<float, float>>& ybar,
        const std::vector<std::pair<float, float>>& zbar)
    {
        const bool xOk = build_curve_on_reference_axis_from_linear_pairs(gXBar, xbar);
        const bool yOk = build_curve_on_reference_axis_from_linear_pairs(gYBar, ybar);
        const bool zOk = build_curve_on_reference_axis_from_linear_pairs(gZBar, zbar);
        if (!(xOk && yOk && zOk)) {
            log_resample_failure("CIE 1931 CMF resample failed",
                { {"x", xOk}, {"y", yOk}, {"z", zOk} });
        }
    }

    // --- Baseline spectral densities (global, not per-dye) ---
    inline void set_negative_baseline_linear(
        const std::vector<std::pair<float, float>>& min_linear,
        const std::vector<std::pair<float, float>>& mid_linear)
    {
        const bool minOk = build_curve_on_reference_axis_from_linear_pairs(gBaseMin, min_linear);
        const bool midOk = build_curve_on_reference_axis_from_linear_pairs(gBaseMid, mid_linear);
        if (!(minOk && midOk)) {
            log_resample_failure("Baseline resample failed",
                { {"min", minOk}, {"mid", midOk} });
        }
        gHasBaseline = !gBaseMin.lambda_nm.empty() && !gBaseMid.lambda_nm.empty();
    }

    inline void set_negative_baseline_log10(
        const std::vector<std::pair<float, float>>& min_log10,
        const std::vector<std::pair<float, float>>& mid_log10)
    {
        const bool minOk = build_curve_on_reference_axis_from_log10_pairs(gBaseMin, min_log10);
        const bool midOk = build_curve_on_reference_axis_from_log10_pairs(gBaseMid, mid_log10);
        if (!(minOk && midOk)) {
            log_resample_failure("Baseline log10 resample failed",
                { {"min", minOk}, {"mid", midOk} });
        }
        gHasBaseline = !gBaseMin.lambda_nm.empty() && !gBaseMid.lambda_nm.empty();
    }

} // namespace Spectral
