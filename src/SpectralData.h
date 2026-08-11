// SpectralData.h
// Canonical spectral data, immutable table structures, resource I/O, and narrow resampling
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "Logging.h"
#include "NpyLoader.h"

namespace Spectral {

    inline constexpr float kLambdaMin = 380.0f;
    inline constexpr float kLambdaMax = 780.0f;
    inline constexpr float kDelta = 5.0f;
    inline constexpr int kNumSamples = static_cast<int>((kLambdaMax - kLambdaMin) / kDelta) + 1;
    static_assert(kNumSamples == 81, "Spectral grid must be 380..780 nm at 5 nm.");

    struct Curve {
        std::vector<float> lambda_nm;
        std::vector<float> linear;
    };

    struct SpectralShape {
        static constexpr float lambdaMin = kLambdaMin;
        static constexpr float lambdaMax = kLambdaMax;
        static constexpr float delta = kDelta;
        static constexpr int K = kNumSamples;

        std::array<float, kNumSamples> wavelengths{};

        constexpr SpectralShape() : wavelengths{} {
            for (int i = 0; i < kNumSamples; ++i) {
                wavelengths[static_cast<size_t>(i)] = lambdaMin + delta * static_cast<float>(i);
            }
        }
    };

    struct SpectralContext {
        SpectralShape shape;
        Curve xBar, yBar, zBar;
        std::atomic<bool> hanatosAvailable{false};
        NpySpectraLUT hanSpectra;
        std::atomic<bool> mallettAvailable{false};
        NpyFloat2D mallettBasis;
    };

    inline SpectralContext& context() {
        static SpectralContext ctx{};
        return ctx;
    }

} // namespace Spectral

namespace SpectralResampleDetail {

    struct IndexedPair {
        float lambda = 0.0f;
        float value = 0.0f;
        size_t originalIndex = 0;
    };

    inline void dedup_pairs_keep_first_by_wavelength(
        const std::vector<std::pair<float, float>>& inPairs,
        std::vector<float>& outX,
        std::vector<float>& outY) {
        outX.clear();
        outY.clear();

        std::vector<IndexedPair> temp;
        temp.reserve(inPairs.size());
        for (size_t i = 0; i < inPairs.size(); ++i) {
            const float lambda = inPairs[i].first;
            const float value = inPairs[i].second;
            if (!std::isfinite(lambda) || !std::isfinite(value)) {
                continue;
            }
            temp.push_back({lambda, value, i});
        }
        if (temp.empty()) {
            return;
        }

        std::sort(temp.begin(), temp.end(), [](const IndexedPair& a, const IndexedPair& b) {
            if (a.lambda != b.lambda)
                return a.lambda < b.lambda;
            return a.originalIndex < b.originalIndex;
        });

        float currentLambda = temp[0].lambda;
        float currentValue = temp[0].value;
        size_t currentFirstIndex = temp[0].originalIndex;

        auto flush = [&]() {
            outX.push_back(currentLambda);
            outY.push_back(currentValue);
        };

        for (size_t i = 1; i < temp.size(); ++i) {
            const float lambda = temp[i].lambda;
            const float value = temp[i].value;
            const size_t idx = temp[i].originalIndex;

            if (lambda == currentLambda) {
                if (idx < currentFirstIndex) {
                    currentFirstIndex = idx;
                    currentValue = value;
                }
                continue;
            }

            flush();
            currentLambda = lambda;
            currentValue = value;
            currentFirstIndex = idx;
        }
        flush();
    }

    inline std::vector<std::pair<float, float>> scalar_akima_resample(
        const std::vector<float>& x,
        const std::vector<float>& y,
        const float* axisNm,
        size_t axisCount) {
        std::vector<std::pair<float, float>> out;
        if (x.size() < 2 || x.size() != y.size()) {
            return out;
        }

        std::vector<double> abscissae(x.begin(), x.end());
        std::vector<double> ordinates(y.begin(), y.end());
        const std::size_t sampleCount = abscissae.size();
        for (std::size_t index = 0; index < sampleCount; ++index) {
            if (!std::isfinite(abscissae[index]) || !std::isfinite(ordinates[index])) {
                return out;
            }
            if (index > 0 && abscissae[index] <= abscissae[index - 1]) {
                return out;
            }
        }

        std::vector<double> slopes(sampleCount, 0.0);
        if (sampleCount == 2) {
            const double interval = abscissae[1] - abscissae[0];
            const double slope = (ordinates[1] - ordinates[0]) / interval;
            slopes[0] = slope;
            slopes[1] = slope;
        } else {
            const std::size_t extendedSlopeCount = sampleCount + 3;
            std::vector<double> intervals(sampleCount - 1, 0.0);
            for (std::size_t index = 0; index + 1 < sampleCount; ++index) {
                intervals[index] = abscissae[index + 1] - abscissae[index];
            }

            std::vector<double> extendedSlopes(extendedSlopeCount, 0.0);
            std::vector<double> defaultSlopes(sampleCount, 0.0);
            std::vector<double> slopeDifferences(extendedSlopeCount - 1, 0.0);
            std::vector<double> forwardWeights(sampleCount, 0.0);
            std::vector<double> backwardWeights(sampleCount, 0.0);
            std::vector<double> weightSums(sampleCount, 0.0);

            auto calculate_weights = [&]() {
                std::fill(extendedSlopes.begin(), extendedSlopes.end(), 0.0);
                for (std::size_t index = 0; index + 1 < sampleCount; ++index) {
                    extendedSlopes[index + 2] =
                        (ordinates[index + 1] - ordinates[index]) / intervals[index];
                }
                extendedSlopes[1] =
                    2.0 * extendedSlopes[2] - extendedSlopes[3];
                extendedSlopes[0] =
                    2.0 * extendedSlopes[1] - extendedSlopes[2];
                extendedSlopes[extendedSlopeCount - 2] =
                    2.0 * extendedSlopes[extendedSlopeCount - 3] -
                    extendedSlopes[extendedSlopeCount - 4];
                extendedSlopes[extendedSlopeCount - 1] =
                    2.0 * extendedSlopes[extendedSlopeCount - 2] -
                    extendedSlopes[extendedSlopeCount - 3];

                for (std::size_t index = 0; index < sampleCount; ++index) {
                    defaultSlopes[index] =
                        0.5 * (extendedSlopes[index] + extendedSlopes[index + 3]);
                }
                for (std::size_t index = 0; index + 1 < extendedSlopeCount; ++index) {
                    slopeDifferences[index] =
                        std::abs(extendedSlopes[index + 1] - extendedSlopes[index]);
                }

                double maximumWeightSum = 0.0;
                for (std::size_t index = 0; index < sampleCount; ++index) {
                    forwardWeights[index] = slopeDifferences[index + 2];
                    backwardWeights[index] = slopeDifferences[index];
                    weightSums[index] =
                        forwardWeights[index] + backwardWeights[index];
                    maximumWeightSum = std::max(maximumWeightSum, weightSums[index]);
                }
                return maximumWeightSum;
            };

            constexpr double kBreakMultiplier = 1e-9;
            const double cutoff = kBreakMultiplier * calculate_weights();
            for (std::size_t index = 0; index < sampleCount; ++index) {
                double slope = defaultSlopes[index];
                if (weightSums[index] > cutoff) {
                    const double numerator =
                        backwardWeights[index] *
                        (extendedSlopes[index + 2] - extendedSlopes[index + 1]);
                    slope = extendedSlopes[index + 1] + numerator / weightSums[index];
                }
                slopes[index] = slope;
            }
        }

        const std::size_t segmentCount = sampleCount - 1;
        std::vector<double> coefficients(segmentCount * 4, 0.0);
        for (std::size_t segment = 0; segment < segmentCount; ++segment) {
            const double interval = abscissae[segment + 1] - abscissae[segment];
            const double inverseInterval = 1.0 / interval;
            const double inverseIntervalSquared = inverseInterval * inverseInterval;
            const double delta =
                (ordinates[segment + 1] - ordinates[segment]) * inverseInterval;
            const std::size_t coefficientIndex = segment * 4;
            coefficients[coefficientIndex] = ordinates[segment];
            coefficients[coefficientIndex + 1] = slopes[segment];
            coefficients[coefficientIndex + 2] =
                (3.0 * delta - 2.0 * slopes[segment] - slopes[segment + 1]) *
                inverseInterval;
            coefficients[coefficientIndex + 3] =
                (slopes[segment] + slopes[segment + 1] - 2.0 * delta) *
                inverseIntervalSquared;
        }

        out.reserve(axisCount);
        for (size_t index = 0; index < axisCount; ++index) {
            const float wavelength = axisNm[index];
            const double wavelength64 = static_cast<double>(wavelength);
            float value = std::numeric_limits<float>::quiet_NaN();
            if (wavelength64 >= abscissae.front() && wavelength64 <= abscissae.back()) {
                auto upper = std::upper_bound(abscissae.begin(), abscissae.end(), wavelength64);
                std::size_t upperIndex =
                    static_cast<std::size_t>(std::distance(abscissae.begin(), upper));
                if (upperIndex == 0) {
                    upperIndex = 1;
                }
                if (upperIndex >= abscissae.size()) {
                    upperIndex = abscissae.size() - 1;
                }
                const std::size_t segment = upperIndex - 1;
                const double offset = wavelength64 - abscissae[segment];
                const double* segmentCoefficients = &coefficients[segment * 4];
                double interpolated = segmentCoefficients[3];
                interpolated = interpolated * offset + segmentCoefficients[2];
                interpolated = interpolated * offset + segmentCoefficients[1];
                interpolated = interpolated * offset + segmentCoefficients[0];
                value = static_cast<float>(interpolated);
            }
            out.emplace_back(wavelength, value);
        }
        return out;
    }

} // namespace SpectralResampleDetail

inline std::vector<std::pair<float, float>> akima_resample_agx(
    const std::vector<std::pair<float, float>>& pairs,
    const float* axis_nm,
    size_t axis_count) {
    std::vector<std::pair<float, float>> out;
    if (!axis_nm || axis_count == 0) {
        return out;
    }

    std::vector<float> xs;
    std::vector<float> ys;
    SpectralResampleDetail::dedup_pairs_keep_first_by_wavelength(pairs, xs, ys);
    if (xs.size() < 2) {
        return out;
    }

    return SpectralResampleDetail::scalar_akima_resample(xs, ys, axis_nm, axis_count);
}

namespace Spectral {

    inline SpectralShape& gShape = context().shape;
    inline Curve& gXBar = context().xBar;
    inline Curve& gYBar = context().yBar;
    inline Curve& gZBar = context().zBar;

    inline bool spectral_shape_matches_reference(const SpectralShape& s);

    // =========================================================================
    // Hanatos 2025 LUT availability (shared across modules)
    // =========================================================================

    inline std::atomic<bool>& gHanatosAvailable = context().hanatosAvailable;

    inline bool hanatos_available() {
        return gHanatosAvailable.load(std::memory_order_acquire);
    }

    inline void set_hanatos_available(bool available) {
        gHanatosAvailable.store(available, std::memory_order_release);
    }

    inline NpySpectraLUT& gHanSpectra = context().hanSpectra;
    inline std::atomic<bool>& gMallettAvailable = context().mallettAvailable;
    inline NpyFloat2D& gMallettBasis = context().mallettBasis;

    inline bool hanatos_matches_reference_shape() {
        if (gHanSpectra.size <= 0) {
            return false;
        }
        if (gHanSpectra.numSamples != Spectral::kNumSamples) {
            return false;
        }
        if (!spectral_shape_matches_reference(gShape)) {
            return false;
        }
        return gShape.K == gHanSpectra.numSamples;
    }

    inline void disable_hanatos_if_reference_mismatch() {
        if (hanatos_available() && !hanatos_matches_reference_shape()) {
            JTRACE("HANATOS", "Disabling spectral LUT: reference axis mismatch");
            set_hanatos_available(false);
        }
    }

    inline void load_hanatos_spectra_lut(const std::string& path) {
        bool success = load_npy_spectra_lut(path, gHanSpectra);
        set_hanatos_available(success && gHanSpectra.size > 0 && gHanSpectra.numSamples > 0);
        if (hanatos_available() && !hanatos_matches_reference_shape()) {
            disable_hanatos_if_reference_mismatch();
        }
    }

    inline bool mallett_available() {
        return gMallettAvailable.load(std::memory_order_acquire);
    }

    inline void set_mallett_available(bool available) {
        gMallettAvailable.store(available, std::memory_order_release);
    }

    inline bool mallett_basis_matches_reference_shape() {
        if (gMallettBasis.rows != Spectral::kNumSamples) {
            return false;
        }
        if (gMallettBasis.cols != 3) {
            return false;
        }
        return true;
    }

    inline void load_mallett2019_basis_npy(const std::string& path) {
        NpyFloat2D basis;
        bool success = load_npy_float2d(path, basis);
        if (success) {
            if (basis.rows == Spectral::kNumSamples && basis.cols == 3) {
                gMallettBasis = std::move(basis);
                set_mallett_available(true);
                return;
            }
            if (basis.rows == 3 && basis.cols == Spectral::kNumSamples) {
                NpyFloat2D transposed;
                transposed.rows = Spectral::kNumSamples;
                transposed.cols = 3;
                transposed.data.resize(static_cast<size_t>(transposed.rows) * transposed.cols);
                for (int r = 0; r < basis.rows; ++r) {
                    for (int c = 0; c < basis.cols; ++c) {
                        transposed.data[static_cast<size_t>(c) * 3 + r] =
                            basis.data[static_cast<size_t>(r) * basis.cols + c];
                    }
                }
                gMallettBasis = std::move(transposed);
                set_mallett_available(true);
                return;
            }
        }
        gMallettBasis = NpyFloat2D{};
        set_mallett_available(false);
    }

    // =========================================================================
    // Constants
    // =========================================================================

    // DWG working space white point (D65)
    inline constexpr float gDWG_WhitePoint_XYZ[3] = {
        0.950455f, 1.0f, 1.089058f};

    // Spectral upsampling / SPD reconstruction selection.
    // Note: "Mallett" refers to the Mallett 2019 sRGB basis reconstruction (when Hanatos LUT is not selected/available).
    enum class SpectralUpsamplingMode : std::uint8_t {
        PreferHanatos = 0,
        ForceMallett = 1
    };

    // ============================================================================
    // SpectralTables: Per-instance spectral tables (consolidated from SpectralTables.h)
    // ============================================================================

    struct SpectralTables {
        // Wavelength axis
        std::vector<float> lambda;
        int K = 0;
        float deltaLambda = 5.0f;
        float invYn = 1.0f;
        float whiteXYZ[3] = {0.0f, 0.0f, 0.0f};

        // Reference illuminant white point (for chromatic adaptation in SPD reconstruction)
        float refIllumWhiteXYZ[3] = {0.95047f, 1.0f, 1.08883f}; // D65 default

        // Illuminant-weighted CMFs (Ax, Ay, Az) and raw CMFs
        std::vector<float> Ax, Ay, Az;
        std::vector<float> Xbar, Ybar, Zbar;
        std::vector<float> illum; // Illuminant SPD used to build Ax/Ay/Az.

        // Dye extinction tables
        std::vector<float> epsY, epsM, epsC;

        // Baseline (optional) and flag
        std::vector<float> baseDensityMin, baseDensityMid;
        bool hasBaseline = false;

        // Reference density used to compute baseline interpolation mix (0 => use baseDensityMin).
        float densityBaselineMixReference = 0.0f;

        // Hashes used for scanner caches
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
        (void)message;
        if (JTRACE_ENABLED(1)) {
            JTRACE("SPECTRAL", "WARN: " + message);
        }
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
            if (!first)
                oss << ", ";
            first = false;
            oss << state.first << '=' << (state.second ? "ok" : "empty");
        }
        oss << ')';
        log_spectral_warning(oss.str());
    }

    inline void mean_power_normalize(std::vector<float>& spd) {
        if (spd.empty())
            return;
        double sum = 0.0;
        for (float v : spd)
            sum += static_cast<double>(v);
        double mean = sum / static_cast<double>(spd.size());
        if (mean > 0.0) {
            for (float& v : spd)
                v = static_cast<float>(v / mean);
        }
    }

    // ============================================================================
    // Curve Sampling and Resampling Functions
    // ============================================================================

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

    inline std::vector<std::pair<float, float>> resample_pairs_akima_to_reference_axis(
        const std::vector<std::pair<float, float>>& inPairs) {
        std::vector<std::pair<float, float>> out;
        if (inPairs.empty()) {
            return out;
        }

        if (samples_follow_reference_axis(inPairs)) {
            return inPairs;
        }

        // Out-of-domain evaluation must yield NaN (no extrapolation, no endpoint clamp),
        // matching SciPy Akima with extrapolate=False / extrapolate=None semantics.
        const SpectralShape& axis = gShape;
        return akima_resample_agx(inPairs, axis.wavelengths.data(), static_cast<size_t>(axis.K));
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
            if (line.empty())
                continue;
            // Strip comments starting at # or ;
            auto strip_comment = [&](char c) {
                size_t p = line.find(c);
                if (p != std::string::npos)
                    line.erase(p);
            };
            strip_comment('#');
            strip_comment(';');
            std::istringstream ss(line);
            float x = 0.0f, y = 0.0f;
            if (!(ss >> x))
                continue;
            // Skip optional comma/semicolon
            while (ss.peek() == ',' || ss.peek() == ';')
                ss.get();
            if (!(ss >> y))
                continue;
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
            if (line.empty())
                continue;
            auto strip_comment = [&](char c) {
                size_t p = line.find(c);
                if (p != std::string::npos)
                    line.erase(p);
            };
            strip_comment('#');
            strip_comment(';');

            std::istringstream ss(line);
            float l = 0.0f, xv = 0.0f, yv = 0.0f, zv = 0.0f;

            if (!(ss >> l))
                continue;
            while (ss.peek() == ',' || ss.peek() == ';')
                ss.get();

            if (!(ss >> xv))
                continue;
            while (ss.peek() == ',' || ss.peek() == ';')
                ss.get();

            if (!(ss >> yv))
                continue;
            while (ss.peek() == ',' || ss.peek() == ';')
                ss.get();

            if (!(ss >> zv))
                continue;

            out.xbar.emplace_back(l, xv);
            out.ybar.emplace_back(l, yv);
            out.zbar.emplace_back(l, zv);
        }
        return out;
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

    inline void set_cie_1931_2deg_cmf(
        const std::vector<std::pair<float, float>>& xbar,
        const std::vector<std::pair<float, float>>& ybar,
        const std::vector<std::pair<float, float>>& zbar) {
        const auto assign_cmf = [](Curve& curve,
                                   const std::vector<std::pair<float, float>>& samples) {
            curve.lambda_nm.clear();
            curve.linear.clear();
            if (!samples_follow_reference_axis(samples)) {
                return false;
            }
            curve.lambda_nm.reserve(samples.size());
            curve.linear.reserve(samples.size());
            for (const auto& sample : samples) {
                if (!std::isfinite(sample.first)) {
                    curve.lambda_nm.clear();
                    curve.linear.clear();
                    return false;
                }
                curve.lambda_nm.push_back(sample.first);
                curve.linear.push_back(sample.second);
            }
            return true;
        };
        const bool xOk = assign_cmf(gXBar, xbar);
        const bool yOk = assign_cmf(gYBar, ybar);
        const bool zOk = assign_cmf(gZBar, zbar);
        if (!(xOk && yOk && zOk)) {
            log_resample_failure("CIE 1931 CMF resample failed",
                                 {{"x", xOk}, {"y", yOk}, {"z", zOk}});
        }
    }

} // namespace Spectral
