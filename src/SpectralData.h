// SpectralData.h
// Canonical spectral data and immutable table structures.
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "Logging.h"

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

    struct FilmTcLut {
        static constexpr int kSize = 192;
        static constexpr int kChannels = 4;

        // Fixed C-order [192][192][4]; RGB are followed by one padding channel.
        std::vector<float> rgba;
    };

    // FJ_TEMP_BRIDGE: independent spectral source storage; remove S4.E.
    struct ReconstructionLut {
        int size = 0;
        int numSamples = 0;
        std::vector<float> data;
        std::uint64_t assetHash = 0;
    };

    struct MallettBasis {
        int rows = 0;
        int cols = 0;
        std::vector<float> data;
    };

    struct SpectralContext {
        SpectralShape shape;
        Curve xBar, yBar, zBar;
        std::atomic<bool> hanatosAvailable{false};
        ReconstructionLut hanSpectra;
        std::atomic<bool> arcticAvailable{false};
        ReconstructionLut arcticSpectra;
        std::atomic<bool> mallettAvailable{false};
        MallettBasis mallettBasis;
    };

    inline SpectralContext& context() {
        static SpectralContext ctx{};
        return ctx;
    }

} // namespace Spectral

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

    inline ReconstructionLut& gHanSpectra = context().hanSpectra;
    inline std::atomic<bool>& gMallettAvailable = context().mallettAvailable;
    inline MallettBasis& gMallettBasis = context().mallettBasis;

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

    inline bool arctic_available() {
        return context().arcticAvailable.load(std::memory_order_acquire);
    }

    inline void set_arctic_available(bool available) {
        context().arcticAvailable.store(available, std::memory_order_release);
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
        ForceMallett = 1,
        Arctic2026beta04 = 2
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

    // ============================================================================
    // CSV I/O Functions
    // ============================================================================

    // Utility: Load wavelength/value pairs from a CSV file
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

    // Construction commits all three curves under bootstrap reader exclusion.
    void set_cie_1931_2deg_cmf(
        const std::vector<std::pair<float, float>>& xbar,
        const std::vector<std::pair<float, float>>& ybar,
        const std::vector<std::pair<float, float>>& zbar);

} // namespace Spectral
