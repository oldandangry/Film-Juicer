// SpectralData.h
// Data structures, curves, I/O operations, and global setters for spectral processing
#pragma once

#include <cmath>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "AkimaInterpolator.h"
#include "NpyLoader.h"
#include "Logging.h"
#include "Hash.h"

namespace Spectral {

    inline constexpr float kLambdaMin = 380.0f;
    inline constexpr float kLambdaMax = 780.0f;
    inline constexpr float kDelta = 5.0f;
    inline constexpr int kNumSamples = static_cast<int>((kLambdaMax - kLambdaMin) / kDelta) + 1;
    static_assert(kNumSamples == 81, "Spectral grid must be 380..780 nm at 5 nm.");

    inline constexpr float kLogExposureMin = -3.0f;
    inline constexpr float kLogExposureMax = 4.0f;
    inline constexpr int kLogExposureSamples = 256;
    inline constexpr float kLogExposureDelta =
        (kLogExposureMax - kLogExposureMin) / static_cast<float>(kLogExposureSamples - 1);

    struct PrecomputeStatus {
        std::atomic<uint64_t> illumVersion{0};
        std::atomic<uint64_t> lastPrecomputeIllumVersion{~uint64_t{0}};
        std::atomic<bool> dirty{true};
        std::atomic<uint64_t> shapeVersion{0};
        std::atomic<uint64_t> lastPrecomputeShapeVersion{~uint64_t{0}};
        std::atomic<uint64_t> mixVersion{0};
    };

    struct BaselineCtx {
        bool hasBaseline;
        const float* baseMin;
        const float* baseMid;
        float mix;
    };

    struct Curve {
        std::vector<float> lambda_nm;
        std::vector<float> linear;

        void build_from_log10_pairs(const std::vector<std::pair<float, float>>& samples) {
            lambda_nm.clear();
            linear.clear();
            lambda_nm.reserve(samples.size());
            linear.reserve(samples.size());
            float peak = 0.0f;
            for (auto& p : samples) {
                lambda_nm.push_back(p.first);
                float lin = std::pow(10.0f, p.second);
                if (!std::isfinite(lin) || lin < 0.0f) {
                    lin = 0.0f;
                }
                linear.push_back(lin);
                if (lin > peak)
                    peak = lin;
            }
            if (peak > 0.0f) {
                for (auto& v : linear)
                    v /= peak;
            }
        }

        void build_from_linear_pairs(const std::vector<std::pair<float, float>>& samples) {
            lambda_nm.clear();
            linear.clear();
            if (samples.empty())
                return;

            std::vector<std::pair<float, float>> s = samples;
            std::sort(s.begin(), s.end(), [](auto& a, auto& b) {
                return a.first < b.first;
            });

            lambda_nm.reserve(s.size());
            linear.reserve(s.size());
            for (auto& p : s) {
                lambda_nm.push_back(p.first);
                linear.push_back(p.second);
            }
        }

        float sample(float lambda) const {
            const size_t n = lambda_nm.size();
            if (n == 0)
                return 0.0f;
            if (lambda <= lambda_nm.front())
                return linear.front();
            if (lambda >= lambda_nm.back())
                return linear.back();
            size_t i1 = 1;
            while (i1 < n && lambda_nm[i1] < lambda)
                ++i1;
            size_t i0 = i1 - 1;
            float x0 = lambda_nm[i0], x1 = lambda_nm[i1];
            float y0 = linear[i0], y1 = linear[i1];
            if (!(std::isfinite(x0) && std::isfinite(x1)) || x1 <= x0) {
                return y0;
            }
            if (!std::isfinite(y0)) {
                y0 = std::isfinite(y1) ? y1 : 0.0f;
            }
            if (!std::isfinite(y1)) {
                y1 = y0;
            }
            float t = (lambda - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }
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

    enum class SpectralMutationStage : std::uint8_t {
        None = 0,
        Bootstrap = 1,
        Rebuild = 2
    };

    inline const char* to_cstr(SpectralMutationStage stage) noexcept {
        switch (stage) {
            case SpectralMutationStage::Bootstrap:
                return "bootstrap";
            case SpectralMutationStage::Rebuild:
                return "rebuild";
            case SpectralMutationStage::None:
            default:
                return "none";
        }
    }

    class SpectralMutationScope {
    public:
        SpectralMutationScope(SpectralMutationStage stage, const char* owner = nullptr) noexcept;
        ~SpectralMutationScope() noexcept;

        bool active() const noexcept {
            return _active;
        }

    private:
        SpectralMutationStage _stage = SpectralMutationStage::None;
        const char* _owner = nullptr;
        bool _active = false;
    };

    bool spectral_mutation_scope_active() noexcept;
    SpectralMutationStage spectral_mutation_stage() noexcept;
    bool require_spectral_mutation_scope(const char* action) noexcept;

    struct SpectralContext {
        PrecomputeStatus precomputeStatus;
        std::mutex precomputeMutex;
        int lastIllumChoice = -1;

        SpectralShape shape;
        std::vector<float> epsYTable;
        std::vector<float> epsMTable;
        std::vector<float> epsCTable;
        std::vector<float> xbarTable;
        std::vector<float> ybarTable;
        std::vector<float> zbarTable;
        std::vector<float> baselineMinTable;
        std::vector<float> baselineMidTable;
        std::vector<float> illumTable;
        std::vector<float> Ax;
        std::vector<float> Ay;
        std::vector<float> Az;
        std::vector<float> lambda;

        float ynNorm = 1.0f;
        float invYn = 1.0f;
        float deltaLambda = kDelta;

        Curve illuminantCurve;
        Curve sensBlue, sensGreen, sensRed;
        Curve densityCurveB, densityCurveG, densityCurveR;
        Curve epsY, epsM, epsC;
        Curve xBar, yBar, zBar;
        Curve baseMin, baseMid;
        bool hasBaseline = false;

        std::atomic<bool> hanatosAvailable{false};
        NpySpectraLUT hanSpectra;
        std::atomic<bool> mallettAvailable{false};
        NpyFloat2D mallettBasis;

        std::atomic<bool> spdInit{false};
        float sInv[9] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    };

    namespace detail {
        struct SpectralMutationTlsState {
            int depth = 0;
            SpectralMutationStage stage = SpectralMutationStage::None;
            const char* owner = nullptr;
        };

        inline thread_local SpectralMutationTlsState gSpectralMutationTlsState{};

        inline void trace_spectral_mutation_scope_event(
            const char* event,
            SpectralMutationStage stage,
            const char* owner,
            const char* action = nullptr) noexcept {
            try {
                if (!JTRACE_ENABLED(2)) {
                    return;
                }

                std::string msg = std::string("event=") + (event ? event : "unknown") + " stage=" + to_cstr(stage) + " depth=" + std::to_string(gSpectralMutationTlsState.depth);
                if (owner && owner[0] != '\0') {
                    msg += " owner=";
                    msg += owner;
                }
                if (action && action[0] != '\0') {
                    msg += " action=";
                    msg += action;
                }

                JTRACE_LEVEL(2, "MSPEC", msg);
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
        }
    } // namespace detail

    inline SpectralMutationScope::SpectralMutationScope(
        SpectralMutationStage stage,
        const char* owner) noexcept
        : _stage(stage), _owner(owner) {
        if (stage == SpectralMutationStage::None) {
            return;
        }

        if (detail::gSpectralMutationTlsState.depth <= 0) {
            detail::gSpectralMutationTlsState.depth = 1;
            detail::gSpectralMutationTlsState.stage = stage;
            detail::gSpectralMutationTlsState.owner = owner;
            _active = true;
            detail::trace_spectral_mutation_scope_event("scope_enter", stage, owner);
            return;
        }

        if (detail::gSpectralMutationTlsState.stage == stage) {
            ++detail::gSpectralMutationTlsState.depth;
            _active = true;
            detail::trace_spectral_mutation_scope_event("scope_reenter", stage, owner);
            return;
        }

        detail::trace_spectral_mutation_scope_event(
            "scope_rejected_nested_stage_mismatch",
            detail::gSpectralMutationTlsState.stage,
            detail::gSpectralMutationTlsState.owner,
            owner);
    }

    inline SpectralMutationScope::~SpectralMutationScope() noexcept {
        if (!_active) {
            return;
        }

        if (detail::gSpectralMutationTlsState.depth > 0) {
            --detail::gSpectralMutationTlsState.depth;
        }

        if (detail::gSpectralMutationTlsState.depth <= 0) {
            detail::gSpectralMutationTlsState.depth = 0;
            detail::gSpectralMutationTlsState.stage = SpectralMutationStage::None;
            detail::gSpectralMutationTlsState.owner = nullptr;
        }

        detail::trace_spectral_mutation_scope_event("scope_exit", _stage, _owner);
    }

    inline bool spectral_mutation_scope_active() noexcept {
        return detail::gSpectralMutationTlsState.depth > 0 &&
               detail::gSpectralMutationTlsState.stage != SpectralMutationStage::None;
    }

    inline SpectralMutationStage spectral_mutation_stage() noexcept {
        if (!spectral_mutation_scope_active()) {
            return SpectralMutationStage::None;
        }
        return detail::gSpectralMutationTlsState.stage;
    }

    inline bool require_spectral_mutation_scope(const char* action) noexcept {
        if (spectral_mutation_scope_active()) {
            return true;
        }

        detail::trace_spectral_mutation_scope_event(
            "mutation_guard_violation",
            SpectralMutationStage::None,
            nullptr,
            action);
        return false;
    }

    inline SpectralContext& context() {
        static SpectralContext ctx{};
        return ctx;
    }

    inline void mark_mixing_dirty() {
        if (!require_spectral_mutation_scope("mark_mixing_dirty")) {
            return;
        }
        context().precomputeStatus.mixVersion.fetch_add(1, std::memory_order_acq_rel);
    }

    inline void mark_spectral_tables_dirty() {
        if (!require_spectral_mutation_scope("mark_spectral_tables_dirty")) {
            return;
        }
        context().precomputeStatus.dirty.store(true, std::memory_order_release);
    }

    inline void increment_illum_version() {
        if (!require_spectral_mutation_scope("increment_illum_version")) {
            return;
        }
        context().precomputeStatus.illumVersion.fetch_add(1, std::memory_order_acq_rel);
    }

    inline void increment_shape_version() {
        if (!require_spectral_mutation_scope("increment_shape_version")) {
            return;
        }
        context().precomputeStatus.shapeVersion.fetch_add(1, std::memory_order_acq_rel);
    }

} // namespace Spectral

// AGX-compatible numeric helpers shared by the CPU spectral and print pipeline code.
template <typename T>
inline T nan_to_num_agx(T v) {
    if (std::isnan(v)) {
        return static_cast<T>(0);
    }
    if (!std::isfinite(v)) {
        if (v > static_cast<T>(0)) {
            return std::numeric_limits<T>::max();
        }
        return std::numeric_limits<T>::lowest();
    }
    return v;
}

template <typename T>
inline T fmax_agx(T a, T b) {
    // Match NumPy np.fmax: prefer the non-NaN operand when only one side is NaN.
    return std::fmax(a, b);
}

inline float density_to_light_sample_agx(float density, float illuminant) {
    const double transmitted = std::pow(10.0, -static_cast<double>(density)) * static_cast<double>(illuminant);
    const float out = static_cast<float>(transmitted);
    return std::isnan(out) ? 0.0f : out;
}

inline double density_to_light_sample_agx(double density, double illuminant) {
    const double out = std::pow(10.0, -density) * illuminant;
    return std::isnan(out) ? 0.0 : out;
}

inline void density_to_light_agx(
    const std::vector<float>& density_spectral,
    const std::vector<float>& illuminant,
    std::vector<float>& out_light) {
    const size_t n = std::min(density_spectral.size(), illuminant.size());
    out_light.resize(n);
    for (size_t i = 0; i < n; ++i) {
        out_light[i] = density_to_light_sample_agx(density_spectral[i], illuminant[i]);
    }
}

namespace AgxNanInternal {

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

} // namespace AgxNanInternal

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
    AgxNanInternal::dedup_pairs_keep_first_by_wavelength(pairs, xs, ys);
    if (xs.size() < 2) {
        return out;
    }

    Interpolation::AkimaInterpolator akima;
    if (!akima.build(xs, ys)) {
        return out;
    }

    out.reserve(axis_count);
    for (size_t i = 0; i < axis_count; ++i) {
        const float lambda = axis_nm[i];
        const float value = akima.evaluate(lambda);
        out.emplace_back(lambda, value);
    }
    return out;
}

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
    enum class SpectralUpsamplingMode : int {
        PreferHanatos = 0,
        ForceMallett = 1
    };

    inline SpectralUpsamplingMode spectral_upsampling_mode_from_index(int index) {
        return (index == static_cast<int>(SpectralUpsamplingMode::ForceMallett))
                   ? SpectralUpsamplingMode::ForceMallett
                   : SpectralUpsamplingMode::PreferHanatos;
    }

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
        std::vector<float> baseMin, baseMid;
        bool hasBaseline = false;

        // Reference density used to compute baseline interpolation mix (0 => use baseMin).
        float baselineMixReference = 0.0f;

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

    // Linear sample arbitrary (lambda, value) pairs at 'lambda' with endpoint clamp.
    // Mirrors NumPy np.interp semantics: values are not "repaired" if non-finite.
    inline float sample_linear_pairs(const std::vector<std::pair<float, float>>& pairs, float lambda) {
        const size_t n = pairs.size();
        if (n == 0)
            return 0.0f;
        if (n == 1)
            return pairs.front().second;
        if (lambda <= pairs.front().first)
            return pairs.front().second;
        if (lambda >= pairs.back().first)
            return pairs.back().second;

        size_t i1 = 1;
        while (i1 < n && pairs[i1].first < lambda)
            ++i1;
        const size_t i0 = i1 > 0 ? (i1 - 1) : 0;

        const float x0 = pairs[i0].first;
        const float x1 = pairs[i1].first;
        if (!(std::isfinite(x0) && std::isfinite(x1)) || !(x1 > x0)) {
            return pairs[i0].second;
        }

        const float y0 = pairs[i0].second;
        const float y1 = pairs[i1].second;
        const float t = (lambda - x0) / (x1 - x0);
        return y0 + t * (y1 - y0);
    }

    enum class ReferenceResampleKernel {
        Linear,
        Akima
    };

    inline void sanitize_pairs_for_resample_linear(
        const std::vector<std::pair<float, float>>& inPairs,
        std::vector<std::pair<float, float>>& sanitized) {
        // agx-emulsion sorts by wavelength and removes duplicates with "first occurrence wins"
        // semantics (np.unique(..., return_index=True) after sorting).
        sanitized.clear();
        if (inPairs.empty())
            return;

        struct IndexedPair {
            float lambda = 0.0f;
            float value = 0.0f;
            size_t originalIndex = 0;
        };

        std::vector<IndexedPair> temp;
        temp.reserve(inPairs.size());
        for (size_t i = 0; i < inPairs.size(); ++i) {
            const float lambda = inPairs[i].first;
            if (!std::isfinite(lambda)) {
                continue;
            }
            temp.push_back({lambda, inPairs[i].second, i});
        }
        if (temp.empty())
            return;

        std::sort(temp.begin(), temp.end(), [](const IndexedPair& a, const IndexedPair& b) {
            if (a.lambda != b.lambda)
                return a.lambda < b.lambda;
            return a.originalIndex < b.originalIndex;
        });

        sanitized.reserve(temp.size());
        float lastLambda = temp.front().lambda;
        sanitized.emplace_back(lastLambda, temp.front().value);
        for (size_t i = 1; i < temp.size(); ++i) {
            const float lambda = temp[i].lambda;
            if (lambda == lastLambda) {
                continue; // keep first occurrence
            }
            lastLambda = lambda;
            sanitized.emplace_back(lambda, temp[i].value);
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

    inline std::vector<std::pair<float, float>> resample_pairs_linear_to_reference_axis(
        const std::vector<std::pair<float, float>>& inPairs) {
        std::vector<std::pair<float, float>> out;
        if (inPairs.empty()) {
            return out;
        }

        if (samples_follow_reference_axis(inPairs)) {
            return inPairs;
        }

        std::vector<std::pair<float, float>> sanitized;
        sanitize_pairs_for_resample_linear(inPairs, sanitized);
        if (sanitized.empty()) {
            return out;
        }

        const SpectralShape& axis = gShape;
        out.reserve(static_cast<size_t>(axis.K));
        for (int i = 0; i < axis.K; ++i) {
            const float lambda = axis.wavelengths[i];
            const float value = sample_linear_pairs(sanitized, lambda);
            out.emplace_back(lambda, value);
        }
        return out;
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

    // Build a curve pinned to the reference axis from linear pairs.
    inline bool build_curve_on_reference_axis_from_linear_pairs(
        Curve& curve,
        const std::vector<std::pair<float, float>>& pairs,
        ReferenceResampleKernel kernel = ReferenceResampleKernel::Linear) {
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
            if (!std::isfinite(lambda)) {
                curve.lambda_nm.clear();
                curve.linear.clear();
                return false;
            }
            curve.lambda_nm.push_back(lambda);
            curve.linear.push_back(value);
        }
        return !curve.linear.empty();
    }

    // Build a curve pinned to the reference axis from log10 pairs (log → linear).
    inline bool build_curve_on_reference_axis_from_log10_pairs(
        Curve& curve,
        const std::vector<std::pair<float, float>>& log10pairs,
        ReferenceResampleKernel kernel = ReferenceResampleKernel::Linear) {
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

    // Build a curve from axis-aligned samples without any sanitize/sort/clamp.
    inline bool build_curve_on_reference_axis_from_aligned_pairs(
        Curve& curve,
        const std::vector<std::pair<float, float>>& pairs,
        bool clampNegative = false) {
        if (pairs.size() != static_cast<size_t>(SpectralShape::K)) {
            return false;
        }
        curve.lambda_nm.resize(pairs.size());
        curve.linear.resize(pairs.size());
        for (size_t i = 0; i < pairs.size(); ++i) {
            const float expected = gShape.wavelengths[i];
            if (std::abs(pairs[i].first - expected) > 1e-6f) {
                return false;
            }
            curve.lambda_nm[i] = expected;
            float v = pairs[i].second;
            if (clampNegative && std::isfinite(v) && v < 0.0f) {
                v = 0.0f;
            }
            curve.linear[i] = v;
        }
        return true;
    }

    inline bool build_curve_on_reference_axis_from_log10_aligned_pairs(
        Curve& curve,
        const std::vector<std::pair<float, float>>& logPairs) {
        if (logPairs.size() != static_cast<size_t>(SpectralShape::K)) {
            return false;
        }
        curve.lambda_nm.resize(logPairs.size());
        curve.linear.resize(logPairs.size());
        for (size_t i = 0; i < logPairs.size(); ++i) {
            const float expected = gShape.wavelengths[i];
            if (std::abs(logPairs[i].first - expected) > 1e-6f) {
                return false;
            }
            curve.lambda_nm[i] = expected;
            const float logV = logPairs[i].second;
            float lin = 0.0f;
            if (std::isfinite(logV)) {
                lin = std::pow(10.0f, logV);
                if (!std::isfinite(lin) || lin < 0.0f) {
                    lin = 0.0f;
                }
            }
            curve.linear[i] = lin; // NaNs become 0 per np.nan_to_num parity
        }
        return true;
    }

    inline void sort_and_build(Curve& curve, const std::vector<std::pair<float, float>>& pairs) {
        if (pairs.empty()) {
            curve.lambda_nm.clear();
            curve.linear.clear();
            return;
        }
        std::vector<std::pair<float, float>> sorted = pairs;
        std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) {
            return a.first < b.first;
        });

        std::vector<std::pair<float, float>> filtered;
        filtered.reserve(sorted.size());
        for (auto& p : sorted) {
            if (!std::isfinite(p.first) || !std::isfinite(p.second)) {
                continue;
            }
            filtered.emplace_back(p.first, p.second);
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
            if (line.empty())
                continue;
            // Strip comments
            auto strip_comment = [&](char c) {
                size_t p = line.find(c);
                if (p != std::string::npos)
                    line.erase(p);
            };
            strip_comment('#');
            strip_comment(';');
            std::istringstream ss(line);
            float v = 0.0f;
            if (!(ss >> v))
                continue;
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

    inline const std::vector<float>& reference_log_exposure_axis() {
        static const std::vector<float> axis = []() {
            std::vector<float> v;
            v.reserve(kLogExposureSamples);
            for (int i = 0; i < kLogExposureSamples; ++i) {
                v.push_back(kLogExposureMin + static_cast<float>(i) * kLogExposureDelta);
            }
            return v;
        }();
        return axis;
    }

    inline bool log_exposure_axis_matches_reference(const std::vector<float>& axis) {
        if (axis.size() != static_cast<size_t>(kLogExposureSamples)) {
            return false;
        }
        for (int i = 0; i < kLogExposureSamples; ++i) {
            const float expected = kLogExposureMin + static_cast<float>(i) * kLogExposureDelta;
            if (std::abs(axis[static_cast<size_t>(i)] - expected) > 1e-6f) {
                return false;
            }
        }
        return true;
    }

    inline bool build_curve_on_log_exposure_axis(
        Curve& curve,
        const std::vector<std::pair<float, float>>& pairs,
        bool clampNegative = false) {
        if (pairs.size() != static_cast<size_t>(kLogExposureSamples)) {
            return false;
        }
        // Require monotonic increasing logE to avoid broken interpolation.
        for (size_t i = 1; i < pairs.size(); ++i) {
            if (!(pairs[i].first > pairs[i - 1].first)) {
                return false;
            }
        }

        curve.lambda_nm.resize(pairs.size());
        curve.linear.resize(pairs.size());
        for (size_t i = 0; i < pairs.size(); ++i) {
            curve.lambda_nm[i] = pairs[i].first;
            float v = pairs[i].second;
            if (clampNegative && std::isfinite(v) && v < 0.0f) {
                v = 0.0f;
            }
            curve.linear[i] = v;
        }
        return true;
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
                    identical = false;
                    break;
                }
            }
            // Compare spectrum with a tight epsilon (mean‑power normalization should make this exact or very close)
            if (identical) {
                constexpr float eps = 1e-6f;
                for (size_t i = 0; i < newCurve.linear.size(); ++i) {
                    if (std::fabs(newCurve.linear[i] - gIlluminantCurve.linear[i]) > eps) {
                        identical = false;
                        break;
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
        const std::vector<std::pair<float, float>>& red_log10) {
        const bool bOk = build_curve_on_reference_axis_from_log10_pairs(gSensBlue, blue_log10);
        const bool gOk = build_curve_on_reference_axis_from_log10_pairs(gSensGreen, green_log10);
        const bool rOk = build_curve_on_reference_axis_from_log10_pairs(gSensRed, red_log10);
        if (!(bOk && gOk && rOk)) {
            log_resample_failure("Layer sensitivity resample failed",
                                 {{"B", bOk}, {"G", gOk}, {"R", rOk}});
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
                                 {{"Y", yOk}, {"M", mOk}, {"C", cOk}});
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
                                 {{"Y", yOk}, {"M", mOk}, {"C", cOk}});
        }
    }

    // -------------------------------------------------------------------------
    // CIE 1931 CMFs (Gaussian fallback).
    // -------------------------------------------------------------------------
    inline void set_cie_1931_2deg_cmf(
        const std::vector<std::pair<float, float>>& xbar,
        const std::vector<std::pair<float, float>>& ybar,
        const std::vector<std::pair<float, float>>& zbar) {
        const bool xOk = build_curve_on_reference_axis_from_linear_pairs(gXBar, xbar);
        const bool yOk = build_curve_on_reference_axis_from_linear_pairs(gYBar, ybar);
        const bool zOk = build_curve_on_reference_axis_from_linear_pairs(gZBar, zbar);
        if (!(xOk && yOk && zOk)) {
            log_resample_failure("CIE 1931 CMF resample failed",
                                 {{"x", xOk}, {"y", yOk}, {"z", zOk}});
        }
    }

    // --- Baseline spectral densities (global, not per-dye) ---
    inline void set_negative_baseline_linear(
        const std::vector<std::pair<float, float>>& min_linear,
        const std::vector<std::pair<float, float>>& mid_linear) {
        const bool minOk = build_curve_on_reference_axis_from_linear_pairs(gBaseMin, min_linear);
        const bool midOk = build_curve_on_reference_axis_from_linear_pairs(gBaseMid, mid_linear);
        if (!(minOk && midOk)) {
            log_resample_failure("Baseline resample failed",
                                 {{"min", minOk}, {"mid", midOk}});
        }
        gHasBaseline = !gBaseMin.lambda_nm.empty() && !gBaseMid.lambda_nm.empty();
    }

    inline void set_negative_baseline_log10(
        const std::vector<std::pair<float, float>>& min_log10,
        const std::vector<std::pair<float, float>>& mid_log10) {
        const bool minOk = build_curve_on_reference_axis_from_log10_pairs(gBaseMin, min_log10);
        const bool midOk = build_curve_on_reference_axis_from_log10_pairs(gBaseMid, mid_log10);
        if (!(minOk && midOk)) {
            log_resample_failure("Baseline log10 resample failed",
                                 {{"min", minOk}, {"mid", midOk}});
        }
        gHasBaseline = !gBaseMin.lambda_nm.empty() && !gBaseMid.lambda_nm.empty();
    }

} // namespace Spectral
