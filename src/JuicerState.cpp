#include "JuicerState.h"

#include "Couplers.h"

namespace RebuildWorkingState {

    struct NegativeReuseContext {
        std::uint64_t activeBuildCounter = 0;
        std::uint64_t lastHash = 0;
        int lastFilmStock = 0;
        int lastEnlargerIll = 0;
    };

    inline bool dir_runtime_equivalent(const Couplers::Runtime& a, const Couplers::Runtime& b) {
        if (a.active != b.active) return false;
        if (a.highShift != b.highShift) return false;
        if (a.spatialSigmaMicrometers != b.spatialSigmaMicrometers) return false;
        if (a.spatialSigmaPixels != b.spatialSigmaPixels) return false;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                if (a.M[r][c] != b.M[r][c]) {
                    return false;
                }
            }
        }
        return true;
    }

    inline bool negative_inputs_match(const NegativeReuseContext& ctx,
        int filmStockIndex,
        int enlargerIll)
    {
        return (ctx.lastHash != 0) &&
            (ctx.lastFilmStock == filmStockIndex) &&
            (ctx.lastEnlargerIll == enlargerIll);
    }

    inline bool can_reuse_negative_params(const NegativeReuseContext& ctx,
        const WorkingState* prev,
        const WorkingState& candidate,
        int filmStockIndex,
        int enlargerIll)
    {
        if (!prev) {
            return false;
        }
        if (prev->buildCounter == 0) {
            return false;
        }
        if (prev->buildCounter != ctx.activeBuildCounter) {
            return false;
        }
        if (!negative_inputs_match(ctx, filmStockIndex, enlargerIll)) {
            return false;
        }
        if (!dir_runtime_equivalent(prev->dirRT, candidate.dirRT)) {
            return false;
        }
        if (prev->dirPrecorrected != candidate.dirPrecorrected) {
            return false;
        }
        return true;
    }

} // namespace RebuildWorkingState

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
void JuicerCudaResourcesDeleter::operator()(JuicerCuda::Resources* resources) const noexcept {
    JuicerCuda::destroy(resources);
}
#endif

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <mutex>
#include <sstream>
#include <system_error>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cstdlib>
#include <cstdint>
#include <iterator>

#include "nlohmann/json.hpp"

#include "Illuminants.h"

namespace WorkingStateSharing {

    struct WorkingStateCorePayload {
        Spectral::Curve densB;
        Spectral::Curve densG;
        Spectral::Curve densR;
        std::array<std::array<std::vector<float>, 3>, 3> densityCurvesLayers{};
        bool hasDensityCurvesLayers = false;

        Profiles::GrainMetadata grain;
        Profiles::HalationMetadata halation;
        Profiles::ProfileGlare negativeGlare;
        Profiles::ProfileGlare printGlare;

        Spectral::Curve sensB;
        Spectral::Curve sensG;
        Spectral::Curve sensR;
        Spectral::Curve negSensB;
        Spectral::Curve negSensG;
        Spectral::Curve negSensR;

        Spectral::Curve baseMin;
        Spectral::Curve baseMid;
        bool hasBaseline = false;
        float baselineMixReference = 0.0f;
        float printBaselineMixReference = 0.0f;
        float gammaFactorB = 1.0f;
        float gammaFactorG = 1.0f;
        float gammaFactorR = 1.0f;

        Spectral::SpectralTables tablesView;
        Spectral::SpectralTables tablesPrint;
        Spectral::SpectralTables tablesRef;
        Spectral::SpectralTables tablesScan;

        Scanner::ScannerIlluminant negativeScannerIlluminant;
        Scanner::ScannerDensityRange negativeDensityRange;

        Scanner::ScannerIlluminant printScannerIlluminant;
        Scanner::ScannerDensityRange printDensityRange;

        bool negativeScannerValid = false;
        bool printScannerValid = false;
        bool printGlareCompensated = false;

        float spdSInv[9] = { 1,0,0, 0,1,0, 0,0,1 };
        bool spdReady = false;
        Spectral::FilmRawConfig filmRaw;
        std::shared_ptr<const Print::Runtime> printRT;
        Spectral::NegativeCouplerParams negParams{};
        std::uint64_t coreShareHash = 0;
    };

    inline void capture_working_state_core_payload(const WorkingState& in, WorkingStateCorePayload& out) {
        out.densB = in.densB;
        out.densG = in.densG;
        out.densR = in.densR;
        out.densityCurvesLayers = in.densityCurvesLayers;
        out.hasDensityCurvesLayers = in.hasDensityCurvesLayers;

        out.grain = in.grain;
        out.halation = in.halation;
        out.negativeGlare = in.negativeGlare;
        out.printGlare = in.printGlare;

        out.sensB = in.sensB;
        out.sensG = in.sensG;
        out.sensR = in.sensR;
        out.negSensB = in.negSensB;
        out.negSensG = in.negSensG;
        out.negSensR = in.negSensR;

        out.baseMin = in.baseMin;
        out.baseMid = in.baseMid;
        out.hasBaseline = in.hasBaseline;
        out.baselineMixReference = in.baselineMixReference;
        out.printBaselineMixReference = in.printBaselineMixReference;
        out.gammaFactorB = in.gammaFactorB;
        out.gammaFactorG = in.gammaFactorG;
        out.gammaFactorR = in.gammaFactorR;

        out.tablesView = in.tablesView;
        out.tablesPrint = in.tablesPrint;
        out.tablesRef = in.tablesRef;
        out.tablesScan = in.tablesScan;

        out.negativeScannerIlluminant = in.negativeScannerIlluminant;
        out.negativeDensityRange = in.negativeDensityRange;

        out.printScannerIlluminant = in.printScannerIlluminant;
        out.printDensityRange = in.printDensityRange;

        out.negativeScannerValid = in.negativeScannerValid;
        out.printScannerValid = in.printScannerValid;
        out.printGlareCompensated = in.printGlareCompensated;

        for (int i = 0; i < 9; ++i) {
            out.spdSInv[i] = in.spdSInv[i];
        }
        out.spdReady = in.spdReady;
        out.filmRaw = in.filmRaw;
        out.printRT = in.printRT;
        out.negParams = in.negParams;
        out.coreShareHash = in.coreShareHash;
    }

    inline void apply_working_state_core_payload(const WorkingStateCorePayload& in, WorkingState& out) {
        out.densB = in.densB;
        out.densG = in.densG;
        out.densR = in.densR;
        out.densityCurvesLayers = in.densityCurvesLayers;
        out.hasDensityCurvesLayers = in.hasDensityCurvesLayers;

        out.grain = in.grain;
        out.halation = in.halation;
        out.negativeGlare = in.negativeGlare;
        out.printGlare = in.printGlare;

        out.sensB = in.sensB;
        out.sensG = in.sensG;
        out.sensR = in.sensR;
        out.negSensB = in.negSensB;
        out.negSensG = in.negSensG;
        out.negSensR = in.negSensR;

        out.baseMin = in.baseMin;
        out.baseMid = in.baseMid;
        out.hasBaseline = in.hasBaseline;
        out.baselineMixReference = in.baselineMixReference;
        out.printBaselineMixReference = in.printBaselineMixReference;
        out.gammaFactorB = in.gammaFactorB;
        out.gammaFactorG = in.gammaFactorG;
        out.gammaFactorR = in.gammaFactorR;

        out.tablesView = in.tablesView;
        out.tablesPrint = in.tablesPrint;
        out.tablesRef = in.tablesRef;
        out.tablesScan = in.tablesScan;

        out.negativeScannerIlluminant = in.negativeScannerIlluminant;
        out.negativeDensityRange = in.negativeDensityRange;

        out.printScannerIlluminant = in.printScannerIlluminant;
        out.printDensityRange = in.printDensityRange;

        out.negativeScannerValid = in.negativeScannerValid;
        out.printScannerValid = in.printScannerValid;
        out.printGlareCompensated = in.printGlareCompensated;

        for (int i = 0; i < 9; ++i) {
            out.spdSInv[i] = in.spdSInv[i];
        }
        out.spdReady = in.spdReady;
        out.filmRaw = in.filmRaw;
        out.printRT = in.printRT;
        out.negParams = in.negParams;
        out.coreShareHash = in.coreShareHash;
    }

    struct WorkingStateCoreShared {
        std::uint64_t keyHash = 0;
        std::uint64_t identity = 0;
        std::shared_ptr<const WorkingStateCorePayload> payload;
    };

    struct AcquireCoreSharedResult {
        std::shared_ptr<WorkingStateCoreShared> sharedCore;
        std::uint64_t keyHash = 0;
        std::uint64_t identity = 0;
        std::uint32_t cacheEntries = 0;
        bool hit = false;
        bool inserted = false;
        bool payloadPresent = false;
        bool payloadBackfilled = false;
    };

    class WorkingStateCoreSharedCache final {
    public:
        static WorkingStateCoreSharedCache& instance() noexcept {
            static WorkingStateCoreSharedCache cache;
            return cache;
        }

        AcquireCoreSharedResult acquire_or_create(
            std::uint64_t keyHash,
            std::shared_ptr<const WorkingStateCorePayload> insertPayload = nullptr)
        {
            AcquireCoreSharedResult out{};
            out.keyHash = keyHash;
            if (keyHash == 0) {
                return out;
            }

            std::lock_guard<std::mutex> lock(mutex_);
            ++touchSequence_;
            prune_expired_locked();

            auto it = entries_.find(keyHash);
            if (it != entries_.end()) {
                std::shared_ptr<WorkingStateCoreShared> shared = it->second.shared.lock();
                if (shared) {
                    if (!shared->payload && insertPayload) {
                        shared->payload = std::move(insertPayload);
                        out.payloadBackfilled = true;
                    }
                    it->second.lastTouchSequence = touchSequence_;
                    out.sharedCore = std::move(shared);
                    out.identity = out.sharedCore->identity;
                    out.hit = true;
                    out.payloadPresent = (out.sharedCore->payload != nullptr);
                    out.cacheEntries = static_cast<std::uint32_t>(entries_.size());
                    return out;
                }
                entries_.erase(it);
            }

            auto created = std::make_shared<WorkingStateCoreShared>();
            created->keyHash = keyHash;
            created->identity = ++identitySequence_;
            created->payload = std::move(insertPayload);

            CacheEntry entry{};
            entry.shared = created;
            entry.lastTouchSequence = touchSequence_;
            entries_[keyHash] = std::move(entry);
            trim_to_cap_locked();

            out.sharedCore = std::move(created);
            out.identity = out.sharedCore->identity;
            out.inserted = true;
            out.payloadPresent = (out.sharedCore->payload != nullptr);
            out.cacheEntries = static_cast<std::uint32_t>(entries_.size());
            return out;
        }

    private:
        struct CacheEntry {
            std::weak_ptr<WorkingStateCoreShared> shared;
            std::uint64_t lastTouchSequence = 0;
        };

        static constexpr std::size_t kMaxEntries = 256;

        void prune_expired_locked() {
            for (auto it = entries_.begin(); it != entries_.end();) {
                if (it->second.shared.expired()) {
                    it = entries_.erase(it);
                }
                else {
                    ++it;
                }
            }
        }

        void trim_to_cap_locked() {
            if (entries_.size() <= kMaxEntries) {
                return;
            }
            while (entries_.size() > kMaxEntries) {
                auto victim = std::min_element(
                    entries_.begin(),
                    entries_.end(),
                    [](const auto& a, const auto& b) {
                        return a.second.lastTouchSequence < b.second.lastTouchSequence;
                    });
                if (victim == entries_.end()) {
                    break;
                }
                entries_.erase(victim);
            }
        }

        std::mutex mutex_;
        std::unordered_map<std::uint64_t, CacheEntry> entries_;
        std::uint64_t touchSequence_ = 0;
        std::uint64_t identitySequence_ = 0;
    };

    inline AcquireCoreSharedResult acquire_or_create_shared_core(
        std::uint64_t keyHash,
        std::shared_ptr<const WorkingStateCorePayload> insertPayload = nullptr)
    {
        return WorkingStateCoreSharedCache::instance().acquire_or_create(keyHash, std::move(insertPayload));
    }

} // namespace WorkingStateSharing

namespace RebuildWorkingState {
    namespace curve_inversion {

        static std::optional<float> invert_density_curve(const Spectral::Curve& curve, float targetDensity) {
            if (!std::isfinite(targetDensity)) {
                return std::nullopt;
            }

            const size_t n = std::min(curve.linear.size(), curve.lambda_nm.size());
            if (n == 0) {
                return std::nullopt;
            }

            std::vector<std::pair<float, float>> samples;
            samples.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                const float logE = curve.lambda_nm[i];
                const float density = curve.linear[i];
                if (std::isfinite(logE) && std::isfinite(density)) {
                    samples.emplace_back(logE, density);
                }
            }

            if (samples.empty()) {
                return std::nullopt;
            }

            std::sort(samples.begin(), samples.end(), [](const auto& a, const auto& b) {
                return a.first < b.first;
                });

            if (samples.size() == 1) {
                return samples.front().first;
            }

            auto between = [](float value, float a, float b) {
                return (value >= a && value <= b) || (value <= a && value >= b);
                };

            const float firstDensity = samples.front().second;
            const float lastDensity = samples.back().second;
            const bool increasing = lastDensity >= firstDensity;

            if (increasing) {
                if (targetDensity <= firstDensity) {
                    return samples.front().first;
                }
                if (targetDensity >= lastDensity) {
                    return samples.back().first;
                }
            }
            else {
                if (targetDensity >= firstDensity) {
                    return samples.front().first;
                }
                if (targetDensity <= lastDensity) {
                    return samples.back().first;
                }
            }

            float prevLogE = samples.front().first;
            float prevDensity = samples.front().second;
            for (size_t i = 1; i < samples.size(); ++i) {
                const float logE = samples[i].first;
                const float density = samples[i].second;
                if (!std::isfinite(logE) || !std::isfinite(density)) {
                    prevLogE = logE;
                    prevDensity = density;
                    continue;
                }
                if (between(targetDensity, prevDensity, density)) {
                    if (density == prevDensity) {
                        return logE;
                    }
                    const float t = std::clamp((targetDensity - prevDensity) / (density - prevDensity), 0.0f, 1.0f);
                    return prevLogE + t * (logE - prevLogE);
                }
                prevLogE = logE;
                prevDensity = density;
            }

            return samples.back().first;
        }

    } // namespace curve_inversion

    static std::array<float, 3> compute_mid_neutral_logE_offsets_rgb(
        const Spectral::Curve& densR,
        const Spectral::Curve& densG,
        const Spectral::Curve& densB,
        const std::array<float, 3>& densityMidRGB)
    {
        std::array<float, 3> result{ {0.0f, 0.0f, 0.0f} };

        if (auto logER = curve_inversion::invert_density_curve(densR, densityMidRGB[0])) {
            result[0] = *logER;
        }
        if (auto logEG = curve_inversion::invert_density_curve(densG, densityMidRGB[1])) {
            result[1] = *logEG;
        }
        if (auto logEB = curve_inversion::invert_density_curve(densB, densityMidRGB[2])) {
            result[2] = *logEB;
        }

        return result;
    }
} // namespace RebuildWorkingState

namespace {
    namespace fs = std::filesystem;

    struct FilmStockDefinition {
        std::string optionLabel;
        std::string jsonKey;
    };

    struct PrintPaperDefinition {
        std::string optionLabel;
        std::string folderName;
        std::string jsonKey;
    };

    using FilmStockFallback = std::array<FilmStockDefinition, 5>;
    using PrintPaperFallback = std::array<PrintPaperDefinition, 2>;

    static const FilmStockFallback kFallbackFilmStocks{ {
        { "Vision3 250D", "kodak_vision3_250d_uc" },
        { "Vision3 50D",  "kodak_vision3_50d_uc" },
        { "Vision3 200T", "kodak_vision3_200t_uc" },
        { "Vision3 500T", "kodak_vision3_500t_uc" },
        { "Portra 400",   "kodak_portra_400_auc" }
    } };

    static const PrintPaperFallback kFallbackPrintPapers{ {
        { "2383", "kodak_2383", "kodak_2383_uc" },
        { "2393", "kodak_2393", "kodak_2393_uc" }
    } };

    std::vector<FilmStockDefinition>& film_stock_definitions() {
        static std::vector<FilmStockDefinition> defs;
        return defs;
    }

    std::vector<PrintPaperDefinition>& print_paper_definitions() {
        static std::vector<PrintPaperDefinition> defs;
        return defs;
    }

    std::once_flag gProfileCatalogOnce;

    struct FilterCatalog {
        std::vector<std::string> paperKeys;
        std::vector<std::string> filmKeys;
        std::unordered_set<std::string> paperKeySet;
        std::unordered_set<std::string> filmKeySet;
    };

    void trace_working_state_core_share(
        const WorkingStateSharing::AcquireCoreSharedResult& result,
        std::uint64_t buildCounter,
        const char* path)
    {
        const bool traceLevel2 = JTRACE_ENABLED(2);
        if (!traceLevel2 || result.keyHash == 0) {
            return;
        }
        const char* pathLabel = path ? path : "unknown";
        std::string msg;
        msg.reserve(256);
        msg = "event=core_share_shell";
        msg += " path=";
        msg += pathLabel;
        msg += " build=";
        msg += std::to_string(buildCounter);
        msg += " core_share_hash=";
        msg += std::to_string(result.keyHash);
        msg += " core_share_identity=";
        msg += std::to_string(result.identity);
        msg += " cache_hit=";
        msg += std::to_string(result.hit ? 1 : 0);
        msg += " cache_inserted=";
        msg += std::to_string(result.inserted ? 1 : 0);
        msg += " payload_present=";
        msg += std::to_string(result.payloadPresent ? 1 : 0);
        msg += " payload_backfilled=";
        msg += std::to_string(result.payloadBackfilled ? 1 : 0);
        msg += " cache_entries=";
        msg += std::to_string(static_cast<unsigned long long>(result.cacheEntries));
        JTRACE("MSWSC", msg);
    }

    inline void copy_float3(float dst[3], const float src[3]) {
        std::memcpy(dst, src, 3u * sizeof(float));
    }

    struct RebuildStateSnapshot {
        BaseState base;
        std::shared_ptr<const WorkingState> activeWorkingState;
        std::uint64_t activeBuildCounter = 0;
        ParamSnapshot lastParams;
        std::uint64_t lastHash = 0;
        Print::Runtime printRT;
        std::string dataDir;
        bool baseLoaded = false;
        IlluminantOverrideFlags illuminantOverride;
        std::string filmReferenceIlluminant;
    };

    inline RebuildStateSnapshot snapshot_rebuild_state_locked(InstanceState& state) {
        RebuildStateSnapshot snapshot{};
        snapshot.base = state.base;
        snapshot.activeWorkingState = JuicerAtomic::load_shared_ptr(&state.activeWorkingState);
        snapshot.activeBuildCounter = state.activeBuildCounter;
        snapshot.lastParams = state.lastParams;
        snapshot.lastHash = state.lastHash.load(std::memory_order_acquire);
        snapshot.printRT = state.printRT;
        snapshot.dataDir = state.dataDir;
        snapshot.baseLoaded = state.baseLoaded;
        snapshot.illuminantOverride = state.illuminantOverride;
        snapshot.filmReferenceIlluminant = state.filmReferenceIlluminant;
        return snapshot;
    }

    inline void publish_rebuilt_working_state(
        InstanceState& state,
        const ParamSnapshot& params,
        std::shared_ptr<WorkingState> next,
        bool invalidateSpatialSigmaCache) {
        const std::uint64_t buildCounter = next ? next->buildCounter : 0;
        const std::uint64_t fullHash = next ? next->fullHash : 0;
        std::lock_guard<std::mutex> lock(state.m);
        if (invalidateSpatialSigmaCache) {
            state.spatialSigmaCacheValid.store(false, std::memory_order_release);
        }
        JuicerAtomic::store_shared_ptr(&state.activeWorkingState, std::shared_ptr<const WorkingState>(std::move(next)));
        state.activeBuildCounter = buildCounter;
        state.lastParams = params;
        state.lastHash.store(fullHash, std::memory_order_release);
    }

    inline bool is_finite(float value) {
        return std::isfinite(value);
    }

    inline bool is_finite(double value) {
        return std::isfinite(value);
    }

    inline double clamp_finite_or(double value, double fallback, double lo, double hi) {
        const double candidate = is_finite(value) ? value : fallback;
        return std::clamp(candidate, lo, hi);
    }

    inline float clamp_coupler_ratio(double value) {
        return static_cast<float>(clamp_finite_or(value, 0.0, 0.0, 1.0));
    }

    inline float clamp_coupler_amount(double value) {
        return static_cast<float>(clamp_finite_or(value, 0.0, 0.0, 2.0));
    }

    inline bool is_positive_finite(float value);
    inline bool is_positive_finite(double value);

    inline float sanitize_positive_or(float value, float fallback) {
        return is_positive_finite(value) ? value : fallback;
    }

    inline bool is_positive_finite(float value) {
        return is_finite(value) && value > 0.0f;
    }

    inline bool is_positive_finite(double value) {
        return is_finite(value) && value > 0.0;
    }

    inline float sanitize_nonnegative_or(float value, float fallback) {
        if (!is_finite(value)) {
            return fallback;
        }
        return std::max(0.0f, value);
    }

    inline float sanitize_nonnegative_clamped_or(float value, float fallback, float hi) {
        return std::clamp(sanitize_nonnegative_or(value, fallback), 0.0f, hi);
    }

    inline float finite_or_fallback(float value, float fallback) {
        return is_finite(value) ? value : fallback;
    }

    inline float sanitize_abs_positive_or_nan(float value, float minMagnitude = 1e-6f) {
        if (!is_finite(value)) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        const float magnitude = std::fabs(value);
        return (magnitude > minMagnitude) ? magnitude : std::numeric_limits<float>::quiet_NaN();
    }

    inline bool curve_has_nonfinite_samples(const Spectral::Curve& curve) {
        const float* valueIt = curve.linear.data();
        const float* const valueEnd = valueIt + curve.linear.size();
        for (; valueIt < valueEnd; ++valueIt) {
            if (!is_finite(*valueIt)) {
                return true;
            }
        }
        return false;
    }

    inline float curve_max_clamped_or_default(
        const Spectral::Curve& curve,
        float fallback = 1.0f,
        float minValue = 1e-4f,
        float maxValue = 1000.0f) {
        float maximum = 0.0f;
        const float* values = curve.linear.data();
        const float* const valuesEnd = values + curve.linear.size();
        for (; values < valuesEnd; ++values) {
            const float value = *values;
            if (is_finite(value) && value > maximum) {
                maximum = value;
            }
        }
        if (!is_finite(maximum) || maximum <= minValue) {
            maximum = fallback;
        }
        return std::min(maximum, maxValue);
    }

    inline void clamp_negative_finite_curve_samples(Spectral::Curve& curve) {
        float* sample = curve.linear.data();
        const float* const sampleEnd = sample + curve.linear.size();
        for (; sample < sampleEnd; ++sample) {
            if (is_finite(*sample) && *sample < 0.0f) {
                *sample = 0.0f;
            }
        }
    }

    inline void scale_finite_curve_samples(Spectral::Curve& curve, float scale, bool clampNonnegative = false) {
        float* sample = curve.linear.data();
        const float* const sampleEnd = sample + curve.linear.size();
        for (; sample < sampleEnd; ++sample) {
            if (!is_finite(*sample)) {
                continue;
            }
            *sample *= scale;
            if (clampNonnegative && *sample < 0.0f) {
                *sample = 0.0f;
            }
        }
    }

    inline void copy_3x3_and_append_rhs(const double matrix3x3[3][3], const double rhs[3], double augmented[3][4]) {
        const double* rhsIt = rhs;
        double(*dstRow)[4] = augmented;
        const double(*srcRow)[3] = matrix3x3;
        for (int r = 0; r < 3; ++r, ++rhsIt, ++dstRow, ++srcRow) {
            std::memcpy(*dstRow, *srcRow, 3u * sizeof(double));
            (*dstRow)[3] = *rhsIt;
        }
    }

    inline uint64_t hash_mix(uint64_t h, uint64_t v) {
        h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }

    template <typename MixFn, typename TValue>
    inline void mix_hash_field(uint64_t& h, TValue value, const MixFn& mix) {
        h = mix(h, static_cast<uint64_t>(value));
    }

    template <typename MixFn>
    inline void mix_hash_field_scaled(uint64_t& h, double value, double scale, const MixFn& mix) {
        h = mix(h, static_cast<uint64_t>(value * scale));
    }

    template <typename MixFn>
    inline void mix_hash_field_scaled_rounded_if_finite(
        uint64_t& h,
        double value,
        double scale,
        const MixFn& mix) {
        if (!is_finite(value)) {
            return;
        }
        const int64_t scaled = static_cast<int64_t>(std::llround(value * scale));
        h = mix(h, static_cast<uint64_t>(scaled));
    }

    template <typename MixFn>
    inline void mix_glare_print_hash_fields(uint64_t& h, const ParamSnapshot& p, const MixFn& mix) {
        mix_hash_field_scaled(h, p.glareCompRemovalFactor, 10000.0, mix);
        mix_hash_field_scaled(h, p.glareCompRemovalDensity, 10000.0, mix);
        mix_hash_field_scaled(h, p.glareCompRemovalTransition, 10000.0, mix);
        mix_hash_field_scaled(h, p.printDminFactor, 10000.0, mix);
    }

    template <typename MixFn>
    inline void mix_camera_filter_hash(uint64_t& h, const ParamSnapshot& p, const MixFn& mix) {
        mix_hash_field(h, p.cameraFilterOverride ? 1 : 0, mix);
        if (!p.cameraFilterOverride) {
            return;
        }

        auto mix_triplet = [&](const std::array<double, 3>& triplet) {
            const double* values = triplet.data();
            const double* const valuesEnd = values + triplet.size();
            for (; values < valuesEnd; ++values) {
                mix_hash_field_scaled_rounded_if_finite(h, *values, 10000.0, mix);
            }
        };
        mix_triplet(p.cameraFilterUV);
        mix_triplet(p.cameraFilterIR);
    }

    template <typename MixFn>
    inline void mix_profile_selection_hash_fields(uint64_t& h, const ParamSnapshot& p, const MixFn& mix) {
        mix_hash_field(h, p.filmStockIndex, mix);
        mix_hash_field(h, p.printPaperIndex, mix);
        mix_hash_field(h, p.spectralUpsamplingMode, mix);
        mix_hash_field(h, p.refIll, mix);
        mix_hash_field(h, p.enlIll, mix);
        mix_hash_field(h, p.enlDichroicSet, mix);
    }

    template <typename MixFn>
    inline void mix_output_encoding_hash_fields(uint64_t& h, const ParamSnapshot& p, const MixFn& mix) {
        mix_hash_field(h, p.scannerLutResolution, mix);
        mix_hash_field(h, p.inputColorSpace, mix);
        mix_hash_field(h, p.inputCctfDecoding, mix);
        mix_hash_field(h, p.outputColorSpace, mix);
        mix_hash_field(h, p.outputCctfEncoding, mix);
        mix_hash_field(h, p.outputLinearPassThrough, mix);
    }

    template <typename MixFn>
    inline void mix_coupler_hash_fields(uint64_t& h, const ParamSnapshot& p, const MixFn& mix) {
        mix_hash_field(h, p.couplersActive, mix);
        mix_hash_field_scaled(h, p.couplersAmount, 10000.0, mix);
        mix_hash_field_scaled(h, p.ratioR, 10000.0, mix);
        mix_hash_field_scaled(h, p.ratioG, 10000.0, mix);
        mix_hash_field_scaled(h, p.ratioB, 10000.0, mix);
        mix_hash_field_scaled(h, p.sigma, 10000.0, mix);
        mix_hash_field_scaled(h, p.high, 10000.0, mix);
        mix_hash_field_scaled(h, p.spatialSigmaMicrometers, 10000.0, mix);
    }

    inline void sanitize_dir_matrix(float matrix[3][3]) {
        float* valueIt = &matrix[0][0];
        const float* const valueEnd = valueIt + 9;
        for (; valueIt < valueEnd; ++valueIt) {
            float value = *valueIt;
            if (!is_finite(value)) value = 0.0f;
            if (value < -10.0f) value = -10.0f;
            if (value > 10.0f) value = 10.0f;
            *valueIt = value;
        }
    }

    inline void sanitize_dir_dmax(float dMax[3], float mirror[3]) {
        float* valueIt = dMax;
        float* mirrorIt = mirror;
        const float* const valueEnd = valueIt + 3;
        for (; valueIt < valueEnd; ++valueIt, ++mirrorIt) {
            float value = *valueIt;
            if (!is_finite(value) || value <= 1e-4f) value = 1.0f;
            if (value > 1000.0f) value = 1000.0f;
            *valueIt = value;
            *mirrorIt = value;
        }
    }

    inline void build_dir_matrix_fallback(float matrix[3][3], const float amountValues[3], float layerSigma) {
        const float sigma = sanitize_nonnegative_or(layerSigma, 0.0f);
        float amount[3] = { amountValues[0], amountValues[1], amountValues[2] };
        const float sigmaCapped = std::min(sigma, 3.0f);
        float* amountIt = amount;
        for (int i = 0; i < 3; ++i, ++amountIt) {
            *amountIt = sanitize_nonnegative_clamped_or(*amountIt, 0.0f, 1.0f);
        }

        auto gauss = [sigmaCapped](int dx) -> float {
            if (sigmaCapped <= 0.0f) {
                return (dx == 0) ? 1.0f : 0.0f;
            }
            const float s2 = sigmaCapped * sigmaCapped;
            return std::exp(-0.5f * (dx * dx) / s2);
            };

        for (int row = 0; row < 3; ++row) {
            float kernelRow[3];
            float rowWeightSum = 0.0f;
            for (int col = 0; col < 3; ++col) {
                kernelRow[col] = gauss(col - row);
                rowWeightSum += kernelRow[col];
            }
            if (rowWeightSum > 0.0f) {
                for (float& v : kernelRow) {
                    v /= rowWeightSum;
                }
            }
            float* dst = matrix[row];
            const float* src = kernelRow;
            const float amountRow = amount[row];
            for (int col = 0; col < 3; ++col, ++dst, ++src) {
                *dst = amountRow * *src;
            }
        }

        sanitize_dir_matrix(matrix);
    }

    void recompute_working_state_dir_overlay(const RebuildStateSnapshot& snapshot, const ParamSnapshot& P, WorkingState& target) {
        (void)snapshot;
        const int effectiveCouplersActive = (P.couplersActive != 0) ? 1 : 0;
        const double effectiveCouplersAmount = clamp_finite_or(P.couplersAmount, kFactoryCouplersAmount, 0.0, 2.0);
        const double effectiveRatioB = clamp_finite_or(P.ratioB, kFactoryCouplersRatioB, 0.0, 1.0);
        const double effectiveRatioG = clamp_finite_or(P.ratioG, kFactoryCouplersRatioG, 0.0, 1.0);
        const double effectiveRatioR = clamp_finite_or(P.ratioR, kFactoryCouplersRatioR, 0.0, 1.0);
        const double effectiveCouplersSigma = clamp_finite_or(P.sigma, kFactoryCouplersSigma, 0.0, 4.0);
        const double effectiveCouplersHigh = clamp_finite_or(P.high, kFactoryCouplersHigh, 0.0, 1.0);
        const double effectiveSpatialSigma =
            clamp_finite_or(P.spatialSigmaMicrometers, kFactoryCouplersSpatialSigma, 0.0, 50.0);

        const std::array<float, 3> densityMaxPostDir{
            curve_max_clamped_or_default(target.densB),
            curve_max_clamped_or_default(target.densG),
            curve_max_clamped_or_default(target.densR)
        };

        bool precorrectApplied = false;
        Couplers::Runtime dirRT{};
        dirRT.active = (effectiveCouplersActive != 0);
        {
            const float amountScale = clamp_coupler_amount(effectiveCouplersAmount);
            const float amount[3] = {
                amountScale * clamp_coupler_ratio(effectiveRatioB),
                amountScale * clamp_coupler_ratio(effectiveRatioG),
                amountScale * clamp_coupler_ratio(effectiveRatioR)
            };
#ifdef JUICER_ENABLE_COUPLERS
            Couplers::build_dir_matrix(dirRT.M, amount, static_cast<float>(effectiveCouplersSigma));
#else
            build_dir_matrix_fallback(dirRT.M, amount, static_cast<float>(effectiveCouplersSigma));
#endif
            dirRT.highShift = static_cast<float>(effectiveCouplersHigh);
            dirRT.spatialSigmaMicrometers = static_cast<float>(effectiveSpatialSigma);
            dirRT.spatialSigmaPixels = 0.0f;

#ifdef JUICER_ENABLE_COUPLERS
            if (dirRT.active) {
                if (curve_has_nonfinite_samples(target.densB) ||
                    curve_has_nonfinite_samples(target.densG) ||
                    curve_has_nonfinite_samples(target.densR)) {
                    precorrectApplied = false;
                }
                else {
                    Spectral::Curve densB_corr, densG_corr, densR_corr;
                    Couplers::precorrect_density_curves_before_DIR_into(
                        dirRT.M, dirRT.highShift,
                        target.densB, target.densG, target.densR,
                        densB_corr, densG_corr, densR_corr);
                    target.dirDensB = std::move(densB_corr);
                    target.dirDensG = std::move(densG_corr);
                    target.dirDensR = std::move(densR_corr);
                    precorrectApplied = true;
                }
            }
#endif
        }

        copy_float3(dirRT.dMax, densityMaxPostDir.data());
        target.dirRT = dirRT;
        target.dirPrecorrected = precorrectApplied;
        if (!precorrectApplied) {
            target.dirDensB = target.densB;
            target.dirDensG = target.densG;
            target.dirDensR = target.densR;
        }
        copy_float3(target.dMax, dirRT.dMax);
        target.negParams.DmaxY = target.dMax[0];
        target.negParams.DmaxM = target.dMax[1];
        target.negParams.DmaxC = target.dMax[2];

        sanitize_dir_matrix(target.dirRT.M);
        sanitize_dir_dmax(target.dMax, target.dirRT.dMax);
    }

    bool rebuild_working_state_scanner_output_runtime(const ParamSnapshot& P, WorkingState& target) {
        const bool printRuntimeOk = target.printScannerValid;
        if (!target.negativeScannerValid) {
            JTRACE("HASH", "FATAL: negative scanner runtime marked invalid");
            return false;
        }

        OutputEncoding::Params scannerEncoding{};
        scannerEncoding.colorSpace = OutputEncoding::colorSpaceFromIndex(P.outputColorSpace);
        scannerEncoding.applyCctfEncoding = (P.outputCctfEncoding != 0);
        scannerEncoding.preserveLinearRange = (P.outputLinearPassThrough != 0);
        scannerEncoding.inputIsOutputSpace = true;

        const std::uint32_t lutRes =
            static_cast<std::uint32_t>(std::clamp(P.scannerLutResolution, 17, 128));
        const std::uint64_t negGlareHash = Scanner::hash_glare(target.negativeGlare);
        const std::uint64_t printGlareHash = printRuntimeOk ? Scanner::hash_glare(target.printGlare) : 0;
        if (negGlareHash == 0) {
            JTRACE("HASH", "FATAL: failed to hash negative glare parameters");
            return false;
        }
        if (printRuntimeOk && printGlareHash == 0) {
            JTRACE("HASH", "FATAL: failed to hash print glare parameters");
            return false;
        }
        if (target.tablesScan.tablesHash == 0) {
            JTRACE("HASH", "FATAL: scanner table hash invalid for negative medium");
            return false;
        }
        if (printRuntimeOk && target.tablesPrint.tablesHash == 0) {
            JTRACE("HASH", "FATAL: scanner table hash invalid for print medium");
            return false;
        }

        target.negativeMediumRuntime = Scanner::ScannerMediumRuntime{};
        target.negativeMediumRuntime.medium = Scanner::ScannerMedium::Negative;
        target.negativeMediumRuntime.tables = (target.tablesScan.K > 0) ? &target.tablesScan : nullptr;
        target.negativeMediumRuntime.range = target.negativeDensityRange;
        target.negativeMediumRuntime.illuminant = target.negativeScannerIlluminant;
        target.negativeMediumRuntime.glare = target.negativeGlare;

        target.printMediumRuntime = Scanner::ScannerMediumRuntime{};
        target.printMediumRuntime.medium = Scanner::ScannerMedium::Print;
        target.printMediumRuntime.tables = (target.tablesPrint.K > 0) ? &target.tablesPrint : nullptr;
        target.printMediumRuntime.range = target.printDensityRange;
        target.printMediumRuntime.illuminant = target.printScannerIlluminant;
        target.printMediumRuntime.glare = target.printGlare;

        target.negativeColorRuntime = ScannerOptics::build_color_runtime(
            target.negativeMediumRuntime,
            scannerEncoding);
        if (target.negativeColorRuntime.hash == 0) {
            JTRACE("HASH", "FATAL: scanner color runtime hash invalid for negative medium");
            return false;
        }
        if (printRuntimeOk) {
            target.printColorRuntime = ScannerOptics::build_color_runtime(
                target.printMediumRuntime,
                scannerEncoding);
            if (target.printColorRuntime.hash == 0) {
                JTRACE("HASH", "FATAL: scanner color runtime hash invalid for print medium");
                return false;
            }
        }
        else {
            target.printColorRuntime = Scanner::ColorRuntime{};
        }

        target.negativeStaticKey = Scanner::ScannerStaticKey{};
        target.negativeStaticKey.medium = Scanner::ScannerMedium::Negative;
        target.negativeStaticKey.tablesHash = target.tablesScan.tablesHash;
        target.negativeStaticKey.densityRangeHash = target.negativeDensityRange.digest;
        target.negativeStaticKey.glareHash = negGlareHash;
        target.negativeStaticKey.colorRuntimeHash = target.negativeColorRuntime.hash;
        target.negativeStaticKey.lutResolution = lutRes;
        Scanner::finalize_static_key(target.negativeStaticKey);

        target.printStaticKey = Scanner::ScannerStaticKey{};
        if (printRuntimeOk) {
            target.printStaticKey.medium = Scanner::ScannerMedium::Print;
            target.printStaticKey.tablesHash = target.tablesPrint.tablesHash;
            target.printStaticKey.densityRangeHash = target.printDensityRange.digest;
            target.printStaticKey.glareHash = printGlareHash;
            target.printStaticKey.colorRuntimeHash = target.printColorRuntime.hash;
            target.printStaticKey.lutResolution = lutRes;
            Scanner::finalize_static_key(target.printStaticKey);
        }

        target.negativeMediumRuntime.color = &target.negativeColorRuntime;
        target.negativeMediumRuntime.staticKey = target.negativeStaticKey;

        target.printMediumRuntime.color = printRuntimeOk ? &target.printColorRuntime : nullptr;
        target.printMediumRuntime.staticKey = target.printStaticKey;

        const float glareCompensationFactor = target.printRT
            ? target.printRT->profile.glare.compensationRemovalFactor
            : target.printGlare.compensationRemovalFactor;
        target.printGlareCompensated = (printRuntimeOk && glareCompensationFactor > 0.0f);
        return true;
    }

    std::string sanitize_identifier(const std::string& value) {
        std::string out;
        out.reserve(value.size());
        const char* inData = value.data();
        const char* const inEnd = inData + value.size();
        for (; inData < inEnd; ++inData) {
            unsigned char uc = static_cast<unsigned char>(*inData);
            if (std::isalnum(uc)) {
                out.push_back(static_cast<char>(std::tolower(uc)));
            }
        }
        return out;
    }

    bool equals_ignore_case(const std::string& a, const std::string& b) {
        if (a.size() != b.size()) {
            return false;
        }
        const char* aData = a.data();
        const char* bData = b.data();
        const char* const aEnd = aData + a.size();
        for (; aData < aEnd; ++aData, ++bData) {
            if (std::tolower(static_cast<unsigned char>(*aData)) !=
                std::tolower(static_cast<unsigned char>(*bData))) {
                return false;
            }
        }
        return true;
    }

    FilterCatalog load_filter_catalog(const fs::path& filterPath) {
        FilterCatalog catalog;
        std::error_code ec;
        if (!fs::exists(filterPath, ec) || fs::is_directory(filterPath, ec)) {
            return catalog;
        }

        std::ifstream file(filterPath, std::ios::binary);
        if (!file.is_open()) {
            return catalog;
        }

        nlohmann::json root = nlohmann::json::parse(file, nullptr, false);
        if (root.is_discarded() || !root.is_object()) {
            return catalog;
        }
        const size_t paperCount = root.size();
        catalog.paperKeys.reserve(paperCount);
        catalog.paperKeySet.reserve(paperCount);
        catalog.filmKeys.reserve(paperCount * 4);
        catalog.filmKeySet.reserve(paperCount * 4);

        for (auto it = root.begin(); it != root.end(); ++it) {
            if (!it.value().is_object()) {
                continue;
            }
            const std::string paperKey = it.key();
            if (catalog.paperKeySet.insert(paperKey).second) {
                catalog.paperKeys.emplace_back(paperKey);
            }
            for (auto illumIt = it.value().begin(); illumIt != it.value().end(); ++illumIt) {
                if (!illumIt.value().is_object()) {
                    continue;
                }
                for (auto filmIt = illumIt.value().begin(); filmIt != illumIt.value().end(); ++filmIt) {
                    const std::string filmKey = filmIt.key();
                    if (catalog.filmKeySet.insert(filmKey).second) {
                        catalog.filmKeys.emplace_back(filmKey);
                    }
                }
            }
        }
        return catalog;
    }

    void populate_profile_catalogs() {
        auto& filmDefs = film_stock_definitions();
        auto& paperDefs = print_paper_definitions();
        const bool traceCatalog = JTRACE_ENABLED(1);
        filmDefs.clear();
        paperDefs.clear();

        fs::path base = fs::path(gDataDir);
        fs::path profilesDir = base / "profiles";
        fs::path paperDir = base / "paper";

        std::unordered_map<std::string, Profiles::ProfileInfoSummary> infoByKey;
        std::vector<std::string> missingFilmKeys;
        std::vector<std::string> missingPaperKeys;
        std::error_code ec;
        if (!gDataDir.empty() && fs::exists(profilesDir, ec) && fs::is_directory(profilesDir, ec)) {
            for (fs::directory_iterator it(profilesDir, ec); !ec && it != fs::directory_iterator(); it.increment(ec)) {
                if (!it->is_regular_file(ec)) {
                    continue;
                }
                if (it->path().extension() != ".json") {
                    continue;
                }
                Profiles::ProfileInfoSummary info;
                const std::string jsonPath = it->path().string();
                if (Profiles::load_profile_info(jsonPath, info)) {
                    infoByKey[info.stock] = std::move(info);
                }
            }
        }
        filmDefs.reserve(infoByKey.size());
        paperDefs.reserve(infoByKey.size());

        FilterCatalog filters = load_filter_catalog(profilesDir / "enlarger_neutral_ymc_filters.json");
        if (traceCatalog) {
            missingFilmKeys.reserve(filters.filmKeys.size());
            missingPaperKeys.reserve(filters.paperKeys.size());
        }

        auto pushFilm = [&](const std::string& key) {
            auto it = infoByKey.find(key);
            if (it == infoByKey.end()) {
                if (traceCatalog) {
                    missingFilmKeys.push_back(key + " (profile missing)");
                }
                return;
            }
            if (!equals_ignore_case(it->second.type, "negative")) {
                if (traceCatalog) {
                    std::string reason = key + " (type='" + it->second.type + "')";
                    missingFilmKeys.push_back(std::move(reason));
                }
                return;
            }
            std::string label = it->second.name.empty() ? it->second.stock : it->second.name;
            filmDefs.emplace_back(FilmStockDefinition{ std::move(label), it->second.stock });
            };

        for (const std::string& key : filters.filmKeys) {
            pushFilm(key);
        }

        if (filmDefs.empty()) {
            for (const auto& pair : infoByKey) {
                if (!equals_ignore_case(pair.second.type, "negative")) {
                    continue;
                }
                std::string label = pair.second.name.empty() ? pair.second.stock : pair.second.name;
                filmDefs.emplace_back(FilmStockDefinition{ std::move(label), pair.second.stock });
            }
            std::sort(filmDefs.begin(), filmDefs.end(),
                [](const FilmStockDefinition& a, const FilmStockDefinition& b) {
                    return a.optionLabel < b.optionLabel;
                });
        }

        if (filmDefs.empty()) {
            if (traceCatalog) {
                if (!missingFilmKeys.empty()) {
                    std::ostringstream oss;
                    oss << "catalog fallback: film profiles unavailable for keys: ";
                    const size_t missingCount = missingFilmKeys.size();
                    const std::string* missingData = missingFilmKeys.data();
                    for (size_t i = 0; i < missingCount; ++i, ++missingData) {
                        if (i > 0) {
                            oss << ", ";
                        }
                        oss << *missingData;
                    }
                    JTRACE("CATALOG", oss.str());
                }
                else {
                    JTRACE("CATALOG", "catalog fallback: no film profiles discovered; using defaults");
                }
            }
            filmDefs.assign(kFallbackFilmStocks.begin(), kFallbackFilmStocks.end());
        }

        struct PrintFolderInfo {
            std::string name;
            std::string sanitized;
            bool used = false;
        };

        std::vector<PrintFolderInfo> folders;
        if (!gDataDir.empty() && fs::exists(paperDir, ec) && fs::is_directory(paperDir, ec)) {
            for (fs::directory_iterator it(paperDir, ec); !ec && it != fs::directory_iterator(); it.increment(ec)) {
                if (it->is_directory(ec)) {
                    std::string folder = it->path().filename().string();
                    if (!folder.empty()) {
                        folders.emplace_back(PrintFolderInfo{ folder, sanitize_identifier(folder), false });
                    }
                }
            }
        }

        auto claimFolder = [&](const Profiles::ProfileInfoSummary& info, const std::string& key) -> std::string {
            std::string keySan = sanitize_identifier(key);
            std::string nameSan = sanitize_identifier(info.name);
            size_t bestScore = 0;
            int bestIndex = -1;
            PrintFolderInfo* folderData = folders.data();
            const size_t folderCount = folders.size();
            PrintFolderInfo* folderIt = folderData;
            for (size_t i = 0; i < folderCount; ++i, ++folderIt) {
                if (folderIt->used) {
                    continue;
                }
                const std::string& folderSan = folderIt->sanitized;
                if (folderSan.empty()) {
                    continue;
                }
                size_t score = 0;
                bool match = false;
                if (!keySan.empty() && keySan.find(folderSan) != std::string::npos) {
                    match = true;
                    score = folderSan.size() * 4;
                }
                if (!match && !nameSan.empty() && nameSan.find(folderSan) != std::string::npos) {
                    match = true;
                    score = folderSan.size() * 3;
                }
                if (!match && !keySan.empty() && folderSan.find(keySan) != std::string::npos) {
                    match = true;
                    score = keySan.size() * 2;
                }
                if (!match && !nameSan.empty() && folderSan.find(nameSan) != std::string::npos) {
                    match = true;
                    score = nameSan.size();
                }
                if (match && score > bestScore) {
                    bestScore = score;
                    bestIndex = static_cast<int>(i);
                }
            }
            if (bestIndex >= 0) {
                folderData[bestIndex].used = true;
                return folderData[bestIndex].name;
            }
            return {};
            };

        auto pushPaper = [&](const std::string& key) {
            auto it = infoByKey.find(key);
            if (it == infoByKey.end()) {
                if (traceCatalog) {
                    missingPaperKeys.push_back(key + " (profile missing)");
                }
                return;
            }
            const auto& info = it->second;
            if (!equals_ignore_case(info.type, "paper")) {
                if (traceCatalog) {
                    std::string reason = key + " (type='" + info.type + "')";
                    missingPaperKeys.push_back(std::move(reason));
                }
                return;
            }
            std::string folder = claimFolder(info, key);
            std::string label = info.name.empty() ? key : info.name;
            paperDefs.emplace_back(PrintPaperDefinition{ std::move(label), std::move(folder), key });
            };

        for (const std::string& key : filters.paperKeys) {
            pushPaper(key);
        }

        if (paperDefs.empty()) {
            for (auto& folderInfo : folders) {
                if (folderInfo.used || folderInfo.sanitized.empty()) {
                    continue;
                }
                std::string bestKey;
                Profiles::ProfileInfoSummary bestInfo;
                size_t bestScore = 0;
                for (const auto& pair : infoByKey) {
                    if (!equals_ignore_case(pair.second.type, "paper")) {
                        continue;
                    }
                    std::string keySan = sanitize_identifier(pair.first);
                    std::string nameSan = sanitize_identifier(pair.second.name);
                    size_t score = 0;
                    bool match = false;
                    if (!keySan.empty() && keySan.find(folderInfo.sanitized) != std::string::npos) {
                        match = true;
                        score = folderInfo.sanitized.size() * 4;
                    }
                    if (!match && !nameSan.empty() && nameSan.find(folderInfo.sanitized) != std::string::npos) {
                        match = true;
                        score = folderInfo.sanitized.size() * 3;
                    }
                    if (!match && !keySan.empty() && folderInfo.sanitized.find(keySan) != std::string::npos) {
                        match = true;
                        score = keySan.size() * 2;
                    }
                    if (!match && !nameSan.empty() && folderInfo.sanitized.find(nameSan) != std::string::npos) {
                        match = true;
                        score = nameSan.size();
                    }
                    if (match && score > bestScore) {
                        bestScore = score;
                        bestKey = pair.first;
                        bestInfo = pair.second;
                    }
                }
                if (!bestKey.empty()) {
                    folderInfo.used = true;
                    std::string label = folderInfo.name;
                    paperDefs.emplace_back(PrintPaperDefinition{ std::move(label), folderInfo.name, bestKey });
                }
            }
        }

        if (paperDefs.empty()) {
            if (traceCatalog) {
                if (!missingPaperKeys.empty()) {
                    std::ostringstream oss;
                    oss << "catalog fallback: print profiles unavailable for keys: ";
                    const size_t missingCount = missingPaperKeys.size();
                    const std::string* missingData = missingPaperKeys.data();
                    for (size_t i = 0; i < missingCount; ++i, ++missingData) {
                        if (i > 0) {
                            oss << ", ";
                        }
                        oss << *missingData;
                    }
                    JTRACE("CATALOG", oss.str());
                }
                else {
                    JTRACE("CATALOG", "catalog fallback: no print profiles discovered; using defaults");
                }
            }
            paperDefs.assign(kFallbackPrintPapers.begin(), kFallbackPrintPapers.end());
        }
    }

    void ensure_profile_catalogs() {
        std::call_once(gProfileCatalogOnce, populate_profile_catalogs);
        auto& films = film_stock_definitions();
        auto& papers = print_paper_definitions();
        if (films.empty()) {
            films.assign(kFallbackFilmStocks.begin(), kFallbackFilmStocks.end());
        }
        if (papers.empty()) {
            papers.assign(kFallbackPrintPapers.begin(), kFallbackPrintPapers.end());
        }
    }

    static const FilmStockDefinition& film_stock_for_index(int filmIndex) {
        ensure_profile_catalogs();
        auto& films = film_stock_definitions();
        if (films.empty()) {
            static const FilmStockDefinition dummy{ "", "" };
            return dummy;
        }
        if (filmIndex < 0 || filmIndex >= static_cast<int>(films.size())) {
            filmIndex = 0;
        }
        return films[filmIndex];
    }

    static const PrintPaperDefinition& print_paper_for_index(int index) {
        ensure_profile_catalogs();
        auto& papers = print_paper_definitions();
        if (papers.empty()) {
            static const PrintPaperDefinition dummy{ "", "", "" };
            return dummy;
        }
        if (index < 0 || index >= static_cast<int>(papers.size())) {
            index = 0;
        }
        return papers[index];
    }

    inline bool approx_equal(double a, double b, double eps = 1e-6) {
        return std::fabs(a - b) <= eps;
    }

    inline bool triplet_is_finite(const float values[3]) {
        return is_finite(values[0]) &&
            is_finite(values[1]) &&
            is_finite(values[2]);
    }

    inline bool normalized_white_triplet_is_valid(const float whiteXYZ[3], double yTolerance = 1e-4) {
        return triplet_is_finite(whiteXYZ) &&
            whiteXYZ[1] > 0.0f &&
            approx_equal(static_cast<double>(whiteXYZ[1]), 1.0, yTolerance);
    }

    static Spectral::Curve build_blackbody_curve(float temperature) {
        Spectral::Curve curve;
        if (!(temperature > 0.0f)) {
            return curve;
        }
        const int K = Spectral::gShape.K;
        Spectral::assign_reference_axis(curve.lambda_nm);
        curve.linear.resize(static_cast<size_t>(K));
        const float* wavelengths = Spectral::gShape.wavelengths.data();
        float* outLinear = curve.linear.data();
        for (int i = 0; i < K; ++i) {
            outLinear[i] = Spectral::planck_blackbody(
                wavelengths[i],
                temperature);
        }
        Spectral::mean_power_normalize(curve.linear);
        return curve;
    }

    static std::string make_data_subpath(
        const std::string& baseDir,
        std::initializer_list<std::string_view> segments)
    {
        fs::path path(baseDir);
        for (std::string_view seg : segments) {
            if (!seg.empty()) {
                path /= seg;
            }
        }
        path = path.lexically_normal();
        path.make_preferred();
        return path.string();
    }

    static Spectral::Curve build_illuminant_from_string(
        const std::string& dataDir,
        const std::string& source)
    {
        const std::string normalized = IlluminantKeys::normalize(source);

        auto build_or_log = [&](auto builder, const char* label) -> Spectral::Curve {
            try {
                return builder();
            }
            catch (const std::exception& e) {
                std::ostringstream oss;
                oss << "failed to load illuminant '" << source << "' (" << label
                    << "): " << e.what();
                JTRACE("ILLUM", oss.str());
            }
            catch (...) {
                std::ostringstream oss;
                oss << "failed to load illuminant '" << source << "' (" << label
                    << "): unknown error";
                JTRACE("ILLUM", oss.str());
            }
            return Spectral::Curve{};
            };

        if (IlluminantKeys::matches_any(normalized, { "D65" })) {
            return build_or_log([&]() {
                return Spectral::build_curve_D65_pinned(
                    make_data_subpath(dataDir, { "illuminants", "D65.csv" }));
                }, "D65");
        }
        if (IlluminantKeys::matches_any(normalized, { "D55" })) {
            return build_or_log([&]() {
                return Spectral::build_curve_D55_pinned(
                    make_data_subpath(dataDir, { "illuminants", "D55.csv" }));
                }, "D55");
        }
        if (IlluminantKeys::matches_any(normalized, { "D50" })) {
            return build_or_log([&]() {
                return Spectral::build_curve_D50_pinned(
                    make_data_subpath(dataDir, { "illuminants", "D50.csv" }));
                }, "D50");
        }
        if (IlluminantKeys::matches_any(normalized, { "TH-KG3-L", "THKG3L", "TH-KG3L" })) {
            return build_or_log([&]() {
                return Spectral::build_curve_TH_KG3_L_pinned(
                    make_data_subpath(dataDir, { "filters", "heat_absorbing", "schott", "KG3.csv" }),
                    make_data_subpath(dataDir, { "filters", "lens_transmission", "canon", "canon_24_f28_is.csv" }));
                }, "TH-KG3-L");
        }
        if (IlluminantKeys::matches_any(normalized, { "T", "INCANDESCENT" })) {
            return build_or_log([&]() {
                return Spectral::build_curve_T_pinned(
                    make_data_subpath(dataDir, { "illuminants", "T.csv" }));
                }, "T");
        }
        if (IlluminantKeys::matches_any(normalized, { "K75P", "KINOTON75P" })) {
            return build_or_log([&]() {
                return Spectral::build_curve_K75P_pinned(
                    make_data_subpath(dataDir, { "illuminants", "K75P.csv" }));
                }, "K75P");
        }
        if (IlluminantKeys::matches_any(normalized, { "EQUAL", "EQUALENERGY", "EQUAL-ENERGY" })) {
            return Spectral::build_curve_equal_energy_pinned();
        }

        if (normalized.size() > 2 && normalized[0] == 'B' && normalized[1] == 'B') {
            const char* start = normalized.c_str() + 2;
            while (*start != '\0' && std::isspace(static_cast<unsigned char>(*start))) {
                ++start;
            }
            char* endPtr = nullptr;
            const double temperature = std::strtod(start, &endPtr);
            while (endPtr && *endPtr != '\0' && std::isspace(static_cast<unsigned char>(*endPtr))) {
                ++endPtr;
            }
            if (start != endPtr && endPtr && *endPtr == '\0' && temperature > 0.0) {
                return build_blackbody_curve(static_cast<float>(temperature));
            }
        }

        if (!normalized.empty()) {
            std::ostringstream oss;
            oss << "unknown illuminant string '" << source << "'; returning empty curve";
            JTRACE("ILLUM", oss.str());
        }
        return Spectral::Curve{};
    }

    static bool curve_matches_reference_axis(const Spectral::Curve& curve) {
        const size_t expected = static_cast<size_t>(Spectral::gShape.K);
        if (curve.linear.size() != expected || curve.lambda_nm.size() != expected) {
            return false;
        }
        const float* lambdaData = curve.lambda_nm.data();
        const float* axisData = Spectral::gShape.wavelengths.data();
        constexpr float kAxisMatchTolerance = 1e-3f;
        const float* const lambdaEnd = lambdaData + expected;
        for (; lambdaData < lambdaEnd; ++lambdaData, ++axisData) {
            const float lambda = *lambdaData;
            if (!is_finite(lambda) ||
                std::abs(lambda - *axisData) > kAxisMatchTolerance) {
                return false;
            }
        }
        return true;
    }

    static bool build_scanner_illuminant(
        const std::string& dataDir,
        const std::string& source,
        const char* label,
        Scanner::ScannerIlluminant& out)
    {
        out = Scanner::ScannerIlluminant{};
        if (source.empty()) {
            std::ostringstream oss;
            oss << "FATAL: missing viewing illuminant for " << label;
            JTRACE("ILLUM", oss.str());
            return false;
        }

        Spectral::Curve curve = build_illuminant_from_string(dataDir, source);
        if (!curve_matches_reference_axis(curve)) {
            std::ostringstream oss;
            oss << "FATAL: viewing illuminant '" << source
                << "' for " << label << " not pinned to agx axis";
            JTRACE("ILLUM", oss.str());
            return false;
        }

        const int K = Spectral::gShape.K;
        const auto& xBar = Spectral::gXBar.linear;
        const auto& yBar = Spectral::gYBar.linear;
        const auto& zBar = Spectral::gZBar.linear;
        if (xBar.size() != static_cast<size_t>(K) ||
            yBar.size() != static_cast<size_t>(K) ||
            zBar.size() != static_cast<size_t>(K)) {
            std::ostringstream oss;
            oss << "FATAL: CMFs unavailable for " << label << " (K mismatch)";
            JTRACE("ILLUM", oss.str());
            return false;
        }

        double sumX = 0.0;
        double sumY = 0.0;
        double sumZ = 0.0;
        const float* spdData = curve.linear.data();
        const float* xData = xBar.data();
        const float* yData = yBar.data();
        const float* zData = zBar.data();
        for (int i = 0; i < K; ++i) {
            const float spd = spdData[i];
            const float xb = xData[i];
            const float yb = yData[i];
            const float zb = zData[i];
            if (!(is_finite(spd) && is_finite(xb) && is_finite(yb) && is_finite(zb))) {
                std::ostringstream oss;
                oss << "FATAL: non-finite CMF/SPD sample in " << label << " illuminant";
                JTRACE("ILLUM", oss.str());
                return false;
            }
            sumX += static_cast<double>(spd) * static_cast<double>(xb);
            sumY += static_cast<double>(spd) * static_cast<double>(yb);
            sumZ += static_cast<double>(spd) * static_cast<double>(zb);
        }

        if (!is_positive_finite(sumY)) {
            std::ostringstream oss;
            oss << "FATAL: invalid luminance sum for " << label << " (Yn=" << sumY << ")";
            JTRACE("ILLUM", oss.str());
            return false;
        }

        out.curve = std::move(curve);
        out.normalization = static_cast<float>(sumY);
        const double invYn = 1.0 / sumY;
        const float whiteXYZ[3] = {
            static_cast<float>(sumX * invYn),
            1.0f,
            static_cast<float>(sumZ * invYn)
        };
        copy_float3(out.whiteXYZ, whiteXYZ);

        const double whiteSum = sumX + sumY + sumZ;
        if (!is_positive_finite(whiteSum)) {
            JTRACE("ILLUM", "FATAL: invalid white sum while building scanner illuminant");
            return false;
        }
        out.whiteXY[0] = static_cast<float>(sumX / whiteSum);
        out.whiteXY[1] = static_cast<float>(sumY / whiteSum);

        constexpr int kReferenceAxisSamples = 81;
        const size_t sampleCount = out.curve.linear.size();
        if (K == kReferenceAxisSamples && sampleCount == static_cast<size_t>(kReferenceAxisSamples)) {
            float hashSamples[kReferenceAxisSamples + 1];
            std::memcpy(
                hashSamples,
                out.curve.linear.data(),
                static_cast<size_t>(kReferenceAxisSamples) * sizeof(float));
            hashSamples[kReferenceAxisSamples] = out.normalization;
            out.hash = Hash::hash_float_span(hashSamples, static_cast<size_t>(kReferenceAxisSamples + 1));
        }
        else {
            std::vector<float> hashSamples(sampleCount + 1);
            if (sampleCount > 0) {
                std::memcpy(
                    hashSamples.data(),
                    out.curve.linear.data(),
                    sampleCount * sizeof(float));
            }
            hashSamples[sampleCount] = out.normalization;
            out.hash = Hash::hash_float_span(hashSamples.data(), hashSamples.size());
        }
        if (out.hash == 0) {
            std::ostringstream oss;
            oss << "FATAL: failed to hash viewing illuminant for " << label;
            JTRACE("ILLUM", oss.str());
            return false;
        }
        return true;
    }

    static bool nanmax_curve(const Spectral::Curve& curve, float& outMax) {
        if (curve.linear.empty()) {
            return false;
        }
        double m = -std::numeric_limits<double>::infinity();
        bool found = false;
        const float* values = curve.linear.data();
        const float* const valuesEnd = values + curve.linear.size();
        for (; values < valuesEnd; ++values) {
            const float v = *values;
            if (is_finite(v)) {
                m = std::max(m, static_cast<double>(v));
                found = true;
            }
        }
        if (!found || !is_finite(m)) {
            return false;
        }
        outMax = static_cast<float>(m);
        return is_finite(outMax);
    }

    inline void add_triplet(float dst[3], const float lhs[3], const float rhs[3]) {
        float* dstIt = dst;
        const float* lhsIt = lhs;
        const float* rhsIt = rhs;
        for (int i = 0; i < 3; ++i, ++dstIt, ++lhsIt, ++rhsIt) {
            *dstIt = *lhsIt + *rhsIt;
        }
    }

    inline bool finalize_density_range(
        Scanner::ScannerDensityRange& range,
        const char* invalidRangeMessage,
        const char* hashFailMessage) {
        const float* maxCmyIt = range.max_cmy;
        float* invMaxCmyIt = range.inv_max_cmy;
        for (int i = 0; i < 3; ++i, ++maxCmyIt, ++invMaxCmyIt) {
            const float v = *maxCmyIt;
            if (!is_positive_finite(v)) {
                JTRACE("BUILD", invalidRangeMessage);
                return false;
            }
            *invMaxCmyIt = 1.0f / v;
        }

        float hashVals[6] = {
            range.min_cmy[0], range.min_cmy[1], range.min_cmy[2],
            range.max_cmy[0], range.max_cmy[1], range.max_cmy[2]
        };
        range.digest = Hash::hash_float_span(hashVals, std::size(hashVals));
        if (range.digest == 0) {
            JTRACE("HASH", hashFailMessage);
            return false;
        }
        return true;
    }

    static bool compute_negative_density_range(
        const Spectral::Curve& densB,
        const Spectral::Curve& densG,
        const Spectral::Curve& densR,
        const Profiles::GrainMetadata& grain,
        Scanner::ScannerDensityRange& outRange)
    {
        outRange = Scanner::ScannerDensityRange{};
        const float* densityMinIt = grain.densityMin.data();
        float* minCmyIt = outRange.min_cmy;
        for (int i = 0; i < 3; ++i, ++densityMinIt, ++minCmyIt) {
            const float v = *densityMinIt;
            if (!is_finite(v)) {
                JTRACE("BUILD", "FATAL: non-finite grain density_min for negative medium");
                return false;
            }
            *minCmyIt = v;
        }

        float maxC = 0.0f, maxM = 0.0f, maxY = 0.0f;
        const bool okC = nanmax_curve(densR, maxC);
        const bool okM = nanmax_curve(densG, maxM);
        const bool okY = nanmax_curve(densB, maxY);
        if (!(okC && okM && okY)) {
            JTRACE("BUILD", "FATAL: failed to capture negative density maxima (post-glare)");
            return false;
        }

        const float maxCmy[3] = { maxC, maxM, maxY };
        add_triplet(outRange.max_cmy, maxCmy, outRange.min_cmy);
        return finalize_density_range(
            outRange,
            "FATAL: invalid negative density range (non-positive max)",
            "FATAL: failed to hash negative density range");
    }

    static bool compute_print_density_range(
        const Print::Profile& profile,
        Scanner::ScannerDensityRange& outRange)
    {
        outRange = Scanner::ScannerDensityRange{};
        float maxC = 0.0f, maxM = 0.0f, maxY = 0.0f;
        const bool okC = nanmax_curve(profile.dcC, maxC);
        const bool okM = nanmax_curve(profile.dcM, maxM);
        const bool okY = nanmax_curve(profile.dcY, maxY);
        if (!(okC && okM && okY)) {
            JTRACE("BUILD", "FATAL: failed to capture print density maxima (post-glare)");
            return false;
        }

        const float maxCmy[3] = { maxC, maxM, maxY };
        copy_float3(outRange.max_cmy, maxCmy);
        return finalize_density_range(
            outRange,
            "FATAL: invalid print density range (non-positive max)",
            "FATAL: failed to hash print density range");
    }
}

uint64_t hash_params(const ParamSnapshot& p) {
    uint64_t h = 0;
    mix_profile_selection_hash_fields(h, p, hash_mix);
    mix_glare_print_hash_fields(h, p, hash_mix);
    mix_coupler_hash_fields(h, p, hash_mix);
    mix_output_encoding_hash_fields(h, p, hash_mix);
    mix_camera_filter_hash(h, p, hash_mix);
    return h;
}

uint64_t hash_params_core(const ParamSnapshot& p) {
    uint64_t h = 0;
    mix_profile_selection_hash_fields(h, p, hash_mix);
    mix_glare_print_hash_fields(h, p, hash_mix);
    mix_output_encoding_hash_fields(h, p, hash_mix);
    mix_camera_filter_hash(h, p, hash_mix);
    return h;
}

static uint64_t hash_params_upload_core(const ParamSnapshot& p) {
    uint64_t h = 0;
    mix_profile_selection_hash_fields(h, p, hash_mix);
    mix_glare_print_hash_fields(h, p, hash_mix);
    mix_camera_filter_hash(h, p, hash_mix);
    return h;
}

uint64_t hash_params_dir(const ParamSnapshot& p) {
    uint64_t h = 0;
    mix_coupler_hash_fields(h, p, hash_mix);
    return h;
}

std::string print_dir_for_index(int index) {
    const PrintPaperDefinition& paper = print_paper_for_index(index);
    if (paper.folderName.empty() || gDataDir.empty()) {
        return {};
    }

    std::filesystem::path base = std::filesystem::path(gDataDir);
    std::filesystem::path dir = base / "paper" / paper.folderName;
    dir.make_preferred();
    std::string result = dir.string();
#ifdef _WIN32
    const char separator = '\\';
#else
    const char separator = '/';
#endif
    if (!result.empty() && result.back() != separator) {
        result.push_back(separator);
    }
    return result;
}

const char* print_paper_json_key_for_index(int index) {
    const PrintPaperDefinition& paper = print_paper_for_index(index);
    return paper.jsonKey.empty() ? nullptr : paper.jsonKey.c_str();
}

const char* negative_json_key_for_stock_index(int filmIndex) {
    const FilmStockDefinition& stock = film_stock_for_index(filmIndex);
    return stock.jsonKey.empty() ? nullptr : stock.jsonKey.c_str();
}

int film_stock_option_count() {
    ensure_profile_catalogs();
    return static_cast<int>(film_stock_definitions().size());
}

const char* film_stock_option_label(int index) {
    const FilmStockDefinition& stock = film_stock_for_index(index);
    return stock.optionLabel.empty() ? "" : stock.optionLabel.c_str();
}

int print_paper_option_count() {
    ensure_profile_catalogs();
    return static_cast<int>(print_paper_definitions().size());
}

const char* print_paper_option_label(int index) {
    const PrintPaperDefinition& paper = print_paper_for_index(index);
    return paper.optionLabel.empty() ? "" : paper.optionLabel.c_str();
}

bool load_film_stock_into_base(int filmIndex, InstanceState& S) {
    const FilmStockDefinition& stock = film_stock_for_index(filmIndex);
    const bool stockTraceEnabled = JTRACE_ENABLED(1);
    JTRACE_SCOPE("STOCK", "load_film_stock_into_base");
    auto trace_stock_key = [&](const char* prefix) {
        if (stockTraceEnabled) {
            std::string msg;
            msg.reserve((prefix ? std::strlen(prefix) : 0u) + stock.jsonKey.size());
            msg = prefix ? prefix : "";
            msg += stock.jsonKey;
            JTRACE("STOCK", msg);
        }
    };
    if (stockTraceEnabled) {
        trace_stock_key("load_film_stock_into_base: ");
    }

    std::vector<std::pair<float, float>> c_data;
    std::vector<std::pair<float, float>> m_data;
    std::vector<std::pair<float, float>> y_data;

    std::vector<std::pair<float, float>> r_sens;
    std::vector<std::pair<float, float>> g_sens;
    std::vector<std::pair<float, float>> b_sens;

    std::vector<std::pair<float, float>> dmin;
    std::vector<std::pair<float, float>> dmid;

    std::vector<std::pair<float, float>> dc_r;
    std::vector<std::pair<float, float>> dc_g;
    std::vector<std::pair<float, float>> dc_b;
    std::array<std::array<std::vector<std::pair<float, float>>, 3>, 3> dc_layers{};

    S.base.densityMidNeutral.clear();
    S.base.logExposureMidNeutral.clear();
    S.base.referenceIlluminant.clear();
    S.base.viewingIlluminant.clear();
    S.filmReferenceIlluminant.clear();
    S.base.dyeDensityMinFactor = 1.0f;
    S.base.cameraFilterUV = { {1.0f, 410.0f, 8.0f} };
    S.base.cameraFilterIR = { {1.0f, 675.0f, 15.0f} };
    S.base.cameraFilterDefined = false;
    S.base.grain = Profiles::GrainMetadata{};
    S.base.halation = Profiles::HalationMetadata{};
    S.base.glare = Profiles::ProfileGlare{};
    S.base.hasDensityCurvesLayers = false;
    for (auto& layer : S.base.densityCurvesLayers) {
        for (auto& ch : layer) {
            ch.clear();
        }
    }

    if (stock.jsonKey.empty()) {
        JTRACE("STOCK", "film stock missing JSON key; cannot load profile");
        return false;
    }
    const std::string jsonPath = data_dir_string("profiles", stock.jsonKey + ".json");
    Profiles::AgxFilmProfile profile;
    if (!Profiles::load_agx_film_profile_json(jsonPath, profile)) {
        trace_stock_key("failed to load agx profile json: ");
        return false;
    }

    c_data = std::move(profile.dyeC);
    m_data = std::move(profile.dyeM);
    y_data = std::move(profile.dyeY);
    r_sens = std::move(profile.logSensR);
    g_sens = std::move(profile.logSensG);
    b_sens = std::move(profile.logSensB);
    dc_r = std::move(profile.densityCurveR);
    dc_g = std::move(profile.densityCurveG);
    dc_b = std::move(profile.densityCurveB);
    dmin = std::move(profile.baseMin);
    dmid = std::move(profile.baseMid);
    dc_layers = std::move(profile.densityCurvesLayers);
    if (is_finite(profile.dyeDensityMinFactor) && profile.dyeDensityMinFactor >= 0.0f) {
        S.base.dyeDensityMinFactor = profile.dyeDensityMinFactor;
    }
    if (profile.hasGammaFactor) {
        S.base.gammaFactor = profile.gammaFactor;
    }
    S.base.densityMidNeutral = std::move(profile.densityMidNeutral);
    S.base.logExposureMidNeutral = std::move(profile.logExposureMidNeutral);
    S.base.referenceIlluminant = profile.referenceIlluminant;
    S.base.viewingIlluminant = profile.viewingIlluminant;
    S.filmReferenceIlluminant = S.base.referenceIlluminant;
    S.base.dirCouplers = profile.dirCouplers;
    S.base.cameraFilterUV = profile.cameraFilterUV;
    S.base.cameraFilterIR = profile.cameraFilterIR;
    S.base.cameraFilterDefined = profile.hasCameraFilterUV || profile.hasCameraFilterIR;
    if (profile.dirCouplers.hasData && is_finite(profile.dirCouplers.diffusionSizeUm)) {
        const float spatialSigmaUm = std::clamp(profile.dirCouplers.diffusionSizeUm, 0.0f, 50.0f);
        S.couplerProfileSpatialSigmaMicrometers = static_cast<double>(spatialSigmaUm);
        S.couplerProfileSpatialSigmaValid = true;
    }
    else {
        S.couplerProfileSpatialSigmaMicrometers = 0.0;
        S.couplerProfileSpatialSigmaValid = false;
    }
    S.base.maskingCouplers = profile.maskingCouplers;
    S.base.grain = profile.grain;
    S.base.halation = profile.halation;
    S.base.glare = profile.glare;
    if (profile.hasDensityCurvesLayers) {
        S.base.hasDensityCurvesLayers = true;
        const size_t layerCount = dc_layers.size();
        for (size_t layer = 0; layer < layerCount; ++layer) {
            auto& dstLayer = S.base.densityCurvesLayers[layer];
            const auto& srcLayer = dc_layers[layer];
            const size_t channelCount = srcLayer.size();
            for (size_t ch = 0; ch < channelCount; ++ch) {
                auto& dstCurve = dstLayer[ch];
                const auto& srcCurve = srcLayer[ch];
                dstCurve.clear();
                dstCurve.reserve(srcCurve.size());
                const auto* srcSamples = srcCurve.data();
                const auto* const srcEnd = srcSamples + srcCurve.size();
                for (; srcSamples < srcEnd; ++srcSamples) {
                    dstCurve.emplace_back(srcSamples->second);
                }
            }
        }
    }
    if (stockTraceEnabled) {
        trace_stock_key("loaded agx profile json: ");
        if (!dc_r.empty() && !dc_g.empty() && !dc_b.empty()) {
            std::ostringstream oss;
            oss << "density curves loaded from JSON '" << stock.jsonKey << "' samples R/G/B="
                << dc_r.size() << "/" << dc_g.size() << "/" << dc_b.size();
            JTRACE("STOCK", oss.str());
        }
        else {
            trace_stock_key("density curves missing in JSON profile: ");
        }
    }

    if (stockTraceEnabled) {
        auto sz = [](const auto& v) { return static_cast<int>(v.size()); };
        std::ostringstream oss;
        oss << "c/m/y=" << sz(c_data) << "/" << sz(m_data) << "/" << sz(y_data)
            << " sens r/g/b=" << sz(r_sens) << "/" << sz(g_sens) << "/" << sz(b_sens)
            << " dens r/g/b=" << sz(dc_r) << "/" << sz(dc_g) << "/" << sz(dc_b)
            << " base min/mid=" << sz(dmin) << "/" << sz(dmid);
        JTRACE("STOCK", oss.str());
    }

    if (c_data.empty()) JTRACE("STOCK", "dye_density_c missing/empty");
    if (m_data.empty()) JTRACE("STOCK", "dye_density_m missing/empty");
    if (y_data.empty()) JTRACE("STOCK", "dye_density_y missing/empty");
    if (r_sens.empty()) JTRACE("STOCK", "log_sensitivity_r missing/empty");
    if (g_sens.empty()) JTRACE("STOCK", "log_sensitivity_g missing/empty");
    if (b_sens.empty()) JTRACE("STOCK", "log_sensitivity_b missing/empty");
    if (dc_r.empty()) JTRACE("STOCK", "density_curve_r missing/empty");
    if (dc_g.empty()) JTRACE("STOCK", "density_curve_g missing/empty");
    if (dc_b.empty()) JTRACE("STOCK", "density_curve_b missing/empty");

    const bool okCore =
        !c_data.empty() && !m_data.empty() && !y_data.empty() &&
        !r_sens.empty() && !g_sens.empty() && !b_sens.empty() &&
        !dc_r.empty() && !dc_g.empty() && !dc_b.empty();
    if (!okCore) {
        JTRACE("STOCK", "okCore=false (required assets missing)");
        JTRACE("STOCK", "film stock load aborted due to missing JSON density curves");
        return false;
    }

    const bool epsYOk = Spectral::build_curve_on_reference_axis_from_aligned_pairs(S.base.epsY, y_data);
    const bool epsMOk = Spectral::build_curve_on_reference_axis_from_aligned_pairs(S.base.epsM, m_data);
    const bool epsCOk = Spectral::build_curve_on_reference_axis_from_aligned_pairs(S.base.epsC, c_data);

    const bool sensBOk = Spectral::build_curve_on_reference_axis_from_log10_aligned_pairs(S.base.sensB, b_sens);
    const bool sensGOk = Spectral::build_curve_on_reference_axis_from_log10_aligned_pairs(S.base.sensG, g_sens);
    const bool sensROk = Spectral::build_curve_on_reference_axis_from_log10_aligned_pairs(S.base.sensR, r_sens);

    const bool densBOk = Spectral::build_curve_on_log_exposure_axis(S.base.densB, dc_b);
    const bool densGOk = Spectral::build_curve_on_log_exposure_axis(S.base.densG, dc_g);
    const bool densROk = Spectral::build_curve_on_log_exposure_axis(S.base.densR, dc_r);

    const bool dyeOk = epsYOk && epsMOk && epsCOk;
    const bool sensOk = sensBOk && sensGOk && sensROk;
    const bool densOk = densBOk && densGOk && densROk;

    if (!dyeOk && stockTraceEnabled) {
        std::ostringstream oss;
        oss << "dye epsilon resample failure (Y=" << (epsYOk ? "ok" : "empty")
            << ", M=" << (epsMOk ? "ok" : "empty")
            << ", C=" << (epsCOk ? "ok" : "empty") << ")";
        JTRACE("STOCK", oss.str());
    }
    if (!sensOk && stockTraceEnabled) {
        std::ostringstream oss;
        oss << "log sensitivity resample failure (B=" << (sensBOk ? "ok" : "empty")
            << ", G=" << (sensGOk ? "ok" : "empty")
            << ", R=" << (sensROk ? "ok" : "empty") << ")";
        JTRACE("STOCK", oss.str());
    }

    if (!densOk && stockTraceEnabled) {
        std::ostringstream oss;
        oss << "density curve build failure (B=" << (densBOk ? "ok" : "empty")
            << ", G=" << (densGOk ? "ok" : "empty")
            << ", R=" << (densROk ? "ok" : "empty") << ")";
        JTRACE("STOCK", oss.str());
    }

    if (!(dyeOk && sensOk && densOk)) {
        if (stockTraceEnabled) {
            std::ostringstream fatal;
            fatal << "FATAL: missing spectral data (film profile '" << stock.jsonKey << "')";
            JTRACE("STOCK", fatal.str());
        }
        return false;
    }

    auto subtract_baseline_floor = [](Spectral::Curve& curve) {
        if (curve.linear.empty()) return;
        float minVal = FLT_MAX;
        const float* inData = curve.linear.data();
        const float* const inEnd = inData + curve.linear.size();
        for (; inData < inEnd; ++inData) {
            const float v = *inData;
            if (is_finite(v) && v < minVal) {
                minVal = v;
            }
        }
        if (!is_finite(minVal) || minVal == FLT_MAX || minVal == 0.0f) {
            return;
        }
        float* outData = curve.linear.data();
        const float* const outEnd = outData + curve.linear.size();
        for (; outData < outEnd; ++outData) {
            float& v = *outData;
            // agx-emulsion parity (density curves): preserve authored NaNs through sampling; do not
            // convert NaN -> 0 density (which would lift shadows). agx does `curve -= nanmin(curve)`.
            if (is_finite(v)) {
                v -= minVal;
                // Guard against tiny negatives from float error; keep NaNs untouched.
                if (v < 0.0f) {
                    v = 0.0f;
                }
            }
        }
        };
    subtract_baseline_floor(S.base.densB);
    subtract_baseline_floor(S.base.densG);
    subtract_baseline_floor(S.base.densR);

    bool baseMinOk = false;
    if (!dmin.empty()) {
        baseMinOk = Spectral::build_curve_on_reference_axis_from_aligned_pairs(S.base.baseMin, dmin);
        if (!baseMinOk) {
            S.base.baseMin.lambda_nm.clear();
            S.base.baseMin.linear.clear();
        }
    }
    else {
        S.base.baseMin.lambda_nm.clear();
        S.base.baseMin.linear.clear();
    }

    bool baseMidOk = true;
    if (!dmid.empty()) {
        baseMidOk = Spectral::build_curve_on_reference_axis_from_aligned_pairs(S.base.baseMid, dmid);
        if (!baseMidOk) {
            S.base.baseMid.lambda_nm.clear();
            S.base.baseMid.linear.clear();
        }
    }
    else {
        S.base.baseMid.lambda_nm.clear();
        S.base.baseMid.linear.clear();
    }

    S.base.hasBaseline = baseMinOk && !S.base.baseMin.linear.empty();
    if (!baseMinOk && stockTraceEnabled) {
        std::ostringstream oss;
        oss << "baseline resample failure (min=" << (baseMinOk ? "ok" : "empty") << ")";
        JTRACE("STOCK", oss.str());
    }
    else if (!baseMidOk && !dmid.empty() && stockTraceEnabled) {
        std::ostringstream oss;
        oss << "baseline resample warning (mid=" << (baseMidOk ? "ok" : "empty") << ")";
        JTRACE("STOCK", oss.str());
    }

    if (stockTraceEnabled) {
        std::ostringstream oss;
        oss << "epsY/M/C K=" << static_cast<int>(S.base.epsY.linear.size())
            << "/" << static_cast<int>(S.base.epsM.linear.size())
            << "/" << static_cast<int>(S.base.epsC.linear.size())
            << " sensB/G/R K=" << static_cast<int>(S.base.sensB.linear.size())
            << "/" << static_cast<int>(S.base.sensG.linear.size())
            << "/" << static_cast<int>(S.base.sensR.linear.size())
            << " densB/G/R K=" << static_cast<int>(S.base.densB.linear.size())
            << "/" << static_cast<int>(S.base.densG.linear.size())
            << "/" << static_cast<int>(S.base.densR.linear.size())
            << " baseMin/baseMid K=" << static_cast<int>(S.base.baseMin.linear.size())
            << "/" << static_cast<int>(S.base.baseMid.linear.size())
            << " hasBaseline=" << (S.base.hasBaseline ? 1 : 0);
        JTRACE("STOCK", oss.str());
        JTRACE("STOCK", S.base.hasBaseline ? "loaded OK; baseline=1" : "loaded OK; baseline=0");
    }

    return true;
}

void rebuild_working_state(OfxImageEffectHandle instance, InstanceState& S, const ParamSnapshot& P) {
    Spectral::SpectralMutationScope mutationScope(
        Spectral::SpectralMutationStage::Rebuild,
        "rebuild_working_state");
    (void)mutationScope;

    auto sanitize_curve = [](Spectral::Curve& c) {
        float* values = c.linear.data();
        const float* const valuesEnd = values + c.linear.size();
        for (; values < valuesEnd; ++values) {
            float& v = *values;
            v = sanitize_nonnegative_or(v, 0.0f);
        }
        };

    auto estimate_base_dyes = [](const Spectral::Curve& baseCurve,
        const Spectral::Curve& epsY,
        const Spectral::Curve& epsM,
        const Spectral::Curve& epsC) -> std::array<float, 3>
        {
            std::array<float, 3> result{ 0.0f, 0.0f, 0.0f };
            const size_t K = baseCurve.linear.size();
            if (K == 0 || epsY.linear.size() != K || epsM.linear.size() != K || epsC.linear.size() != K) {
                return result;
            }

            double ATA[3][3] = { {0.0,0.0,0.0}, {0.0,0.0,0.0}, {0.0,0.0,0.0} };
            double ATb[3] = { 0.0, 0.0, 0.0 };

            const float* yData = epsY.linear.data();
            const float* mData = epsM.linear.data();
            const float* cData = epsC.linear.data();
            const float* bData = baseCurve.linear.data();
            for (size_t i = 0; i < K; ++i, ++yData, ++mData, ++cData, ++bData) {
                const double ay = *yData;
                const double am = *mData;
                const double ac = *cData;
                const double b = *bData;
                if (!is_finite(ay) || !is_finite(am) || !is_finite(ac) || !is_finite(b)) {
                    continue;
                }
                const double vec[3] = { ay, am, ac };
                for (int r = 0; r < 3; ++r) {
                    ATb[r] += vec[r] * b;
                    for (int c = 0; c < 3; ++c) {
                        ATA[r][c] += vec[r] * vec[c];
                    }
                }
            }

            double mat[3][4];
            copy_3x3_and_append_rhs(ATA, ATb, mat);

            for (int i = 0; i < 3; ++i) {
                int pivot = i;
                double maxAbs = std::fabs(mat[i][i]);
                for (int r = i + 1; r < 3; ++r) {
                    const double absVal = std::fabs(mat[r][i]);
                    if (absVal > maxAbs) {
                        maxAbs = absVal;
                        pivot = r;
                    }
                }
                if (maxAbs < 1e-9) {
                    return result;
                }
                if (pivot != i) {
                    for (int c = i; c < 4; ++c) {
                        std::swap(mat[i][c], mat[pivot][c]);
                    }
                }
                const double inv = 1.0 / mat[i][i];
                for (int c = i; c < 4; ++c) {
                    mat[i][c] *= inv;
                }
                for (int r = 0; r < 3; ++r) {
                    if (r == i) continue;
                    const double factor = mat[r][i];
                    for (int c = i; c < 4; ++c) {
                        mat[r][c] -= factor * mat[i][c];
                    }
                }
            }

            float* outData = result.data();
            for (int i = 0; i < 3; ++i, ++outData) {
                float v = static_cast<float>(mat[i][3]);
                *outData = sanitize_nonnegative_or(v, 0.0f);
            }

            return result;
        };

    JTRACE_SCOPE("BUILD", "rebuild_working_state");

    std::unique_lock<std::mutex> rebuildLock(S.rebuildMutex);
    std::shared_ptr<WorkingState> next = std::make_shared<WorkingState>();
    WorkingState* target = next.get();
    const bool buildTraceEnabled = JTRACE_ENABLED(1);
    const bool printTraceEnabled = JTRACE_ENABLED(3);
    RebuildStateSnapshot snapshot{};
    {
        std::lock_guard<std::mutex> stateLock(S.m);
        snapshot = snapshot_rebuild_state_locked(S);
    }
    const BaseState& base = snapshot.base;
    Print::Runtime printRT = snapshot.printRT;
    const std::string& dataDir = snapshot.dataDir;

    if (buildTraceEnabled) {
        std::ostringstream oss;
        oss << "enter with baseLoaded=" << (snapshot.baseLoaded ? 1 : 0);
        JTRACE("BUILD", oss.str());
    }

    auto trace_print_working_state_commit = [&]() {
        if (!printTraceEnabled) {
            return;
        }
        const char* paperKey = print_paper_json_key_for_index(P.printPaperIndex);
        const char* filmKey = negative_json_key_for_stock_index(P.filmStockIndex);
        const std::uintptr_t prtPtr = reinterpret_cast<std::uintptr_t>(target->printRT.get());
        const float neutralY = target->printRT ? target->printRT->neutralY : 0.0f;
        const float neutralM = target->printRT ? target->printRT->neutralM : 0.0f;
        const float neutralC = target->printRT ? target->printRT->neutralC : 0.0f;
        const char* paperLabel = paperKey ? paperKey : "<null>";
        const char* filmLabel = filmKey ? filmKey : "<null>";
        const char* printRef = target->printRT ? target->printRT->referenceIlluminant.c_str() : "<null>";
        const char* printView = target->printRT ? target->printRT->viewingIlluminant.c_str() : "<null>";
        std::string msg;
        msg.reserve(256);
        msg = "working state commit build=";
        msg += std::to_string(target->buildCounter);
        msg += " paper=";
        msg += paperLabel;
        msg += " film=";
        msg += filmLabel;
        msg += " printRT=";
        msg += std::to_string(prtPtr);
        msg += " neutralY/M/C=";
        msg += std::to_string(neutralY);
        msg += "/";
        msg += std::to_string(neutralM);
        msg += "/";
        msg += std::to_string(neutralC);
        msg += " printRef=";
        msg += printRef;
        msg += " printView=";
        msg += printView;
        JTRACE_VERBOSE("PRINTDBG", msg);
    };

    const std::uint64_t coreShareHash = hash_params_core(P);
    WorkingStateSharing::AcquireCoreSharedResult coreShare =
        WorkingStateSharing::acquire_or_create_shared_core(coreShareHash);
    const WorkingStateSharing::AcquireCoreSharedResult coreShareInitial = coreShare;
    if (coreShare.sharedCore && coreShare.sharedCore->payload) {
        WorkingStateSharing::apply_working_state_core_payload(*coreShare.sharedCore->payload, *target);
        target->coreShareHash = coreShareHash;
        target->sharedCore = coreShare.sharedCore;

        recompute_working_state_dir_overlay(snapshot, P, *target);
        if (rebuild_working_state_scanner_output_runtime(P, *target)) {
            target->fullHash = hash_params(P);
            target->uploadCoreHash = hash_params_upload_core(P);
            target->coreHash = coreShareHash;
            target->coreShareHash = coreShareHash;
            target->dirHash = hash_params_dir(P);
            target->buildCounter = S.buildCounterNext.fetch_add(1, std::memory_order_relaxed) + 1;
            trace_working_state_core_share(coreShare, target->buildCounter, "full_rebuild_payload_fast");

            trace_print_working_state_commit();

            if (buildTraceEnabled) {
                std::ostringstream oss;
                oss << "WorkingState build #" << target->buildCounter;
                JTRACE("BUILD", oss.str());
            }

            publish_rebuilt_working_state(S, P, next, /*invalidateSpatialSigmaCache*/true);
            if (buildTraceEnabled) {
                std::ostringstream oss;
                oss << "activeWorkingState swapped; buildCounter=" << static_cast<long long>(target->buildCounter);
                JTRACE("BUILD", oss.str());
            }
            return;
        }
        JTRACE("MSWSC", "event=core_share_fastpath_fallback reason=scanner_runtime_rebuild_failed");
    }

    Print::build_illuminant_from_choice(P.enlIll, printRT, dataDir, /*forEnlarger*/true);
    Scanner::ScannerIlluminant printScannerIlluminant;
    if (!build_scanner_illuminant(dataDir, printRT.viewingIlluminant, "print viewing", printScannerIlluminant)) {
        return;
    }
    printRT.illumView = printScannerIlluminant.curve;
    if (buildTraceEnabled) {
        std::ostringstream oss;
        oss << "Enl illum K=" << static_cast<int>(printRT.illumEnlarger.linear.size())
            << " View illum K=" << static_cast<int>(printRT.illumView.linear.size());
        JTRACE("BUILD", oss.str());
    }

    Scanner::ScannerIlluminant negativeScannerIlluminant;
    if (!build_scanner_illuminant(dataDir, base.viewingIlluminant, "negative viewing", negativeScannerIlluminant)) {
        return;
    }
    Scanner::ScannerDensityRange negativeDensityRange;
    bool negativeRangeOk = false;

    Spectral::Curve epsY = base.epsY;
    Spectral::Curve epsM = base.epsM;
    Spectral::Curve epsC = base.epsC;
    Spectral::Curve sensB = base.sensB;
    Spectral::Curve sensG = base.sensG;
    Spectral::Curve sensR = base.sensR;
    Spectral::Curve densB = base.densB;
    Spectral::Curve densG = base.densG;
    Spectral::Curve densR = base.densR;
    Spectral::Curve dirDensB = densB;
    Spectral::Curve dirDensG = densG;
    Spectral::Curve dirDensR = densR;
    Spectral::Curve baseMin = base.baseMin;
    Spectral::Curve baseMid = base.baseMid;
    Spectral::Curve illumRef;
    bool hasRefIlluminant = false;
    const bool hasBaseline = base.hasBaseline;
    const float dyeDensityMinScale =
        (is_finite(base.dyeDensityMinFactor) && base.dyeDensityMinFactor >= 0.0f)
        ? base.dyeDensityMinFactor
        : 1.0f;
    if (hasBaseline && !approx_equal(dyeDensityMinScale, 1.0f)) {
        scale_finite_curve_samples(baseMin, dyeDensityMinScale, true);
    }
    if (hasBaseline) {
        clamp_negative_finite_curve_samples(baseMin);
        clamp_negative_finite_curve_samples(baseMid);
    }
    // agx-emulsion parity: baseline NaNs are preserved in working-state curves and handled as
    // "0 contribution" during integration via SpectralTables baseline validity masks.


    // Per agx-emulsion parity: film profiles contain sensitivities that are ALREADY balanced
    // during profile generation (profiles/balance.py). Runtime rebalancing creates spectral
    // errors. We load the reference illuminant for metadata/debugging purposes only.
    {
        Print::Runtime tmpRT;

        Spectral::Curve profileRefIll;
        if (!snapshot.illuminantOverride.reference && !snapshot.filmReferenceIlluminant.empty()) {
            profileRefIll = build_illuminant_from_string(dataDir, snapshot.filmReferenceIlluminant);
        }

        if (!profileRefIll.linear.empty() &&
            static_cast<int>(profileRefIll.linear.size()) == Spectral::gShape.K)
        {
            tmpRT.illumView = profileRefIll;
        }
        else {
            Print::build_illuminant_from_choice(P.refIll, tmpRT, dataDir, /*forEnlarger*/false);
        }

        illumRef = tmpRT.illumView;
        hasRefIlluminant =
            (!illumRef.linear.empty() && (int)illumRef.linear.size() == Spectral::gShape.K);

        // NOTE: Removed runtime balancing. Profiles are pre-balanced; rebalancing at runtime
        // violates agx-emulsion parity and causes red/magenta color shifts. See claude-review.md.
        if (hasRefIlluminant) {
            JTRACE("BUILD", "loaded reference illuminant for metadata (no runtime balancing applied)");
        }
        else {
            JTRACE("BUILD", "reference illuminant failed to load; SPD reconstruction will be disabled");
        }
    }

    {
        auto to_triplet = [](const std::array<double, 3>& src, const std::array<float, 3>& fallback) {
            std::array<float, 3> out = fallback;
            const double* srcData = src.data();
            float* outData = out.data();
            const double* const srcEnd = srcData + out.size();
            for (; srcData < srcEnd; ++srcData, ++outData) {
                const double v = *srcData;
                if (is_finite(v)) {
                    *outData = static_cast<float>(v);
                }
            }
            return out;
            };

        std::array<float, 3> filterUV = base.cameraFilterUV;
        std::array<float, 3> filterIR = base.cameraFilterIR;
        if (P.cameraFilterOverride) {
            filterUV = to_triplet(P.cameraFilterUV, filterUV);
            filterIR = to_triplet(P.cameraFilterIR, filterIR);
        }

        const float ampUV = std::clamp(filterUV[0], 0.0f, 1.0f);
        const float ampIR = std::clamp(filterIR[0], 0.0f, 1.0f);
        if (ampUV > 0.0f || ampIR > 0.0f) {
            const std::vector<float> bandPass = Spectral::compute_band_pass_filter(filterUV, filterIR);
            if (bandPass.size() == static_cast<size_t>(Spectral::gShape.K)) {
                const size_t bandPassCount = bandPass.size();
                const float* bandPassData = bandPass.data();
                auto applyFilter = [bandPassCount, bandPassData](Spectral::Curve& curve) {
                    if (curve.linear.size() != bandPassCount) {
                        return;
                    }
                    float* curveData = curve.linear.data();
                    const float* bandData = bandPassData;
                    for (size_t i = 0; i < bandPassCount; ++i, ++curveData, ++bandData) {
                        *curveData *= *bandData;
                    }
                    };

                // Apply UV/IR band-pass filters to sensitivities (agx-emulsion parity)
                applyFilter(sensB);
                applyFilter(sensG);
                applyFilter(sensR);
            }
        }
    }

    sanitize_curve(sensB);
    sanitize_curve(sensG);
    sanitize_curve(sensR);

    Spectral::Curve densBForCalibration = densB;
    Spectral::Curve densGForCalibration = densG;
    Spectral::Curve densRForCalibration = densR;
    std::array<float, 3> densityMaxPostDir{
        curve_max_clamped_or_default(densBForCalibration),
        curve_max_clamped_or_default(densGForCalibration),
        curve_max_clamped_or_default(densRForCalibration)
    };

    const Profiles::DirCouplersProfile& dirCfg = base.dirCouplers;

    const int effectiveCouplersActive = (P.couplersActive != 0) ? 1 : 0;
    const double effectiveCouplersAmount = clamp_finite_or(P.couplersAmount, kFactoryCouplersAmount, 0.0, 2.0);
    const double effectiveRatioB = clamp_finite_or(P.ratioB, kFactoryCouplersRatioB, 0.0, 1.0);
    const double effectiveRatioG = clamp_finite_or(P.ratioG, kFactoryCouplersRatioG, 0.0, 1.0);
    const double effectiveRatioR = clamp_finite_or(P.ratioR, kFactoryCouplersRatioR, 0.0, 1.0);
    const double effectiveCouplersSigma = clamp_finite_or(P.sigma, kFactoryCouplersSigma, 0.0, 4.0);
    const double effectiveCouplersHigh = clamp_finite_or(P.high, kFactoryCouplersHigh, 0.0, 1.0);
    const double effectiveSpatialSigma =
        clamp_finite_or(P.spatialSigmaMicrometers, kFactoryCouplersSpatialSigma, 0.0, 50.0);

    bool precorrectApplied = false;
    Couplers::Runtime dirRT{};
    dirRT.active = (effectiveCouplersActive != 0);
    {
        const float amountScale = clamp_coupler_amount(effectiveCouplersAmount);
        const float amount[3] = {
            amountScale * clamp_coupler_ratio(effectiveRatioB),
            amountScale * clamp_coupler_ratio(effectiveRatioG),
            amountScale * clamp_coupler_ratio(effectiveRatioR)
        };
#ifdef JUICER_ENABLE_COUPLERS
        Couplers::build_dir_matrix(dirRT.M, amount, static_cast<float>(effectiveCouplersSigma));
#else
        build_dir_matrix_fallback(dirRT.M, amount, static_cast<float>(effectiveCouplersSigma));
#endif
        dirRT.highShift = static_cast<float>(effectiveCouplersHigh);
        dirRT.spatialSigmaMicrometers = static_cast<float>(effectiveSpatialSigma);
        dirRT.spatialSigmaPixels = 0.0f;

#ifdef JUICER_ENABLE_COUPLERS
        if (dirRT.active) {
            // agx-emulsion parity: density curves may contain intentional toe NaNs. DIR pre-correction
            // must not "heal" them into 0 densities; if authored NaNs exist, skip pre-correction and
            // let NaNs propagate through sampling to "0 transmitted light" downstream.
            if (curve_has_nonfinite_samples(densB) ||
                curve_has_nonfinite_samples(densG) ||
                curve_has_nonfinite_samples(densR)) {
                precorrectApplied = false;
                JTRACE("BUILD", "precorrect: skipped (density curves contain non-finite samples; NaN toe parity)");
            }
            else {
                Spectral::Curve densB_corr, densG_corr, densR_corr;
                Couplers::precorrect_density_curves_before_DIR_into(
                    dirRT.M, dirRT.highShift,
                    densB, densG, densR,
                    densB_corr, densG_corr, densR_corr);
                dirDensB = std::move(densB_corr);
                dirDensG = std::move(densG_corr);
                dirDensR = std::move(densR_corr);
                precorrectApplied = true;
            }
        }
#else
        (void)dirRT;
#endif

        copy_float3(dirRT.dMax, densityMaxPostDir.data());
        if (buildTraceEnabled) {
            std::ostringstream oss;
            oss << "DIR active=" << (dirRT.active ? 1 : 0)
                << " sigma=" << static_cast<float>(effectiveCouplersSigma) << " high=" << static_cast<float>(effectiveCouplersHigh)
                << " dMax=" << dirRT.dMax[0] << "," << dirRT.dMax[1] << "," << dirRT.dMax[2]
                << " precorrect=" << (precorrectApplied ? 1 : 0);
            JTRACE("BUILD", oss.str());
        }
    }
    copy_float3(dirRT.dMax, densityMaxPostDir.data());

    Spectral::NegativeCouplerParams negParams;
    negParams.DmaxY = dirRT.dMax[0];
    negParams.DmaxM = dirRT.dMax[1];
    negParams.DmaxC = dirRT.dMax[2];
    negParams.baseY = 0.0f;
    negParams.baseM = 0.0f;
    negParams.baseC = 0.0f;
    negParams.kB = 6.0f;
    negParams.kG = 6.0f;
    negParams.kR = 6.0f;
    {
        const float defaultMask[9] = {
            0.98f, -0.06f, -0.02f,
           -0.03f,  0.98f, -0.05f,
           -0.02f, -0.04f,  0.98f };
        std::copy(std::begin(defaultMask), std::end(defaultMask), negParams.mask);
    }

    if (base.hasBaseline) {
        const auto baseCoeffs = estimate_base_dyes(baseMin, epsY, epsM, epsC);
        negParams.baseY = baseCoeffs[0];
        negParams.baseM = baseCoeffs[1];
        negParams.baseC = baseCoeffs[2];
    }

    if (dirCfg.hasData) {
        auto computeK = [&](int idx) -> float {
            float amount = is_finite(static_cast<double>(dirCfg.amount))
                ? static_cast<float>(dirCfg.amount)
                : 1.0f;
            if (amount <= 0.0f) {
                amount = 0.1f;
            }
            const float ratio = sanitize_positive_or(dirCfg.ratioRGB[idx], 1.0f);
            float k = 6.0f * amount * ratio;
            k = sanitize_positive_or(k, 6.0f);
            return std::clamp(k, 1.0f, 24.0f);
            };
        negParams.kB = computeK(0);
        negParams.kG = computeK(1);
        negParams.kR = computeK(2);
    }

    const bool hasMaskingData = base.maskingCouplers.hasData;
    if (dirCfg.hasData || hasMaskingData) {
        float amountRGB[3] = { 1.0f, 1.0f, 1.0f };
        if (dirCfg.hasData) {
            const float amount = sanitize_nonnegative_or(static_cast<float>(dirCfg.amount), 1.0f);
            float* amountRgbIt = amountRGB;
            const float* ratioIt = dirCfg.ratioRGB.data();
            for (int i = 0; i < 3; ++i, ++amountRgbIt, ++ratioIt) {
                const float ratio = sanitize_nonnegative_or(*ratioIt, 1.0f);
                *amountRgbIt = std::clamp(amount * ratio, 0.0f, 1.0f);
            }
        }

        float dirMatrix[3][3] = { {0.0f,0.0f,0.0f},{0.0f,0.0f,0.0f},{0.0f,0.0f,0.0f} };
#ifdef JUICER_ENABLE_COUPLERS
        Couplers::build_dir_matrix(dirMatrix, amountRGB, dirCfg.hasData ? dirCfg.diffusionInterlayer : 0.0f);
#else
        build_dir_matrix_fallback(dirMatrix, amountRGB, dirCfg.hasData ? dirCfg.diffusionInterlayer : 0.0f);
#endif

        std::array<float, 3> maskScaleCh{ {1.0f, 1.0f, 1.0f} };
        std::array<float, 3> maskOffsetCh{ {0.0f, 0.0f, 0.0f} };
        if (hasMaskingData) {
            const auto& maskProfile = base.maskingCouplers;
            const auto& lambda = Spectral::gShape.wavelengths;
            const size_t K = lambda.size();
            const Spectral::Curve* epsCurves[3] = { &base.epsY, &base.epsM, &base.epsC };
            const float effectiveness = 1.0f;
            for (int ch = 0; ch < 3; ++ch) {
                const Spectral::Curve* eps = epsCurves[ch];
                if (!eps || eps->linear.size() != K || K == 0) {
                    continue;
                }
                const float cross = (maskProfile.crossOverPoints.size() > static_cast<size_t>(ch))
                    ? maskProfile.crossOverPoints[ch]
                    : std::numeric_limits<float>::quiet_NaN();
                const float widthRaw = (maskProfile.transitionWidths.size() > static_cast<size_t>(ch))
                    ? maskProfile.transitionWidths[ch]
                    : std::numeric_limits<float>::quiet_NaN();
                const float width = sanitize_abs_positive_or_nan(widthRaw);

                double weightSum = 0.0;
                double scaleSum = 0.0;
                double offsetSum = 0.0;
                const float* epsData = eps->linear.data();
                const float* lambdaData = lambda.data();
                for (size_t i = 0; i < K; ++i, ++epsData, ++lambdaData) {
                    const float weight = *epsData;
                    if (!is_finite(weight) || weight <= 0.0f) {
                        continue;
                    }
                    const float lambda_nm = *lambdaData;
                    float scaleSpectral = 1.0f;
                    if (is_finite(cross) && is_finite(width)) {
                        const float t = (lambda_nm - cross) / width;
                        scaleSpectral = (std::erf(t) + 1.0f + effectiveness) / (2.0f + effectiveness);
                    }
                    double gaussSpectral = 0.0;
                    const auto& gaussians = maskProfile.gaussianModel[ch];
                    const std::array<float, 3>* triData = gaussians.data();
                    const size_t triCount = gaussians.size();
                    for (size_t triIdx = 0; triIdx < triCount; ++triIdx, ++triData) {
                        float mu = (*triData)[0];
                        float sigma = (*triData)[1];
                        float amp = (*triData)[2];
                        if (!is_finite(mu) || !is_finite(sigma) || !is_finite(amp)) {
                            continue;
                        }
                        sigma = sanitize_abs_positive_or_nan(sigma);
                        if (!is_finite(sigma)) {
                            continue;
                        }
                        const float s = (lambda_nm - mu) / sigma;
                        gaussSpectral += static_cast<double>(amp) * std::exp(-0.5 * static_cast<double>(s) * static_cast<double>(s));
                    }
                    weightSum += static_cast<double>(weight);
                    scaleSum += static_cast<double>(weight) * static_cast<double>(scaleSpectral);
                    offsetSum += static_cast<double>(weight) * gaussSpectral;
                }
                if (weightSum > 0.0) {
                    float scaleAvg = static_cast<float>(scaleSum / weightSum);
                    float offsetAvg = static_cast<float>(offsetSum / weightSum);
                    scaleAvg = sanitize_positive_or(scaleAvg, 1.0f);
                    offsetAvg = sanitize_nonnegative_or(offsetAvg, 0.0f);
                    maskScaleCh[ch] = scaleAvg;
                    maskOffsetCh[ch] = offsetAvg;
                }
            }
        }

        float* scaleSrc = maskScaleCh.data();
        float* offsetSrc = maskOffsetCh.data();
        float* scaleDst = negParams.maskScale;
        float* offsetDst = negParams.maskOffset;
        for (int i = 0; i < 3; ++i, ++scaleSrc, ++offsetSrc, ++scaleDst, ++offsetDst) {
            float scale = sanitize_positive_or(*scaleSrc, 1.0f);
            float offset = sanitize_nonnegative_or(*offsetSrc, 0.0f);
            *scaleDst = scale;
            *offsetDst = offset;
        }

        const float maskScaleBase = 0.25f;
        const float maskScaleOffset = hasMaskingData ? 0.05f : 0.0f;
        for (int r = 0; r < 3; ++r) {
            float maskStrength = hasMaskingData
                ? ((1.0f - maskScaleCh[r]) + maskOffsetCh[r])
                : 0.0f;
            maskStrength = sanitize_nonnegative_or(maskStrength, 0.0f);
            float amountRow = sanitize_nonnegative_clamped_or(
                dirCfg.hasData ? amountRGB[r] : 1.0f, 0.0f, 1.0f);
            float scale = maskScaleBase * (maskStrength + maskScaleOffset);
            if (dirCfg.hasData) {
                scale *= amountRow;
            }
            scale = std::clamp(scale, 0.0f, 0.25f);
            const float* dirRow = dirMatrix[r];
            float* maskRow = negParams.mask + static_cast<size_t>(r) * 3u;
            for (int c = 0; c < 3; ++c, ++dirRow, ++maskRow) {
                const float val = finite_or_fallback(*dirRow, (r == c) ? 1.0f : 0.0f);
                const float delta = scale * val;
                if (r == c) {
                    const float diag = sanitize_nonnegative_or(1.0f - delta, 1.0f);
                    *maskRow = diag;
                }
                else {
                    const float off = std::clamp(finite_or_fallback(-delta, 0.0f), -1.0f, 1.0f);
                    *maskRow = off;
                }
            }
        }
    }

    target->negativeScannerValid = false;
    target->printScannerValid = false;
    target->printGlareCompensated = false;

    target->negParams = negParams;
    target->grain = base.grain;
    target->halation = base.halation;
    target->negativeGlare = base.glare;
    target->hasDensityCurvesLayers = base.hasDensityCurvesLayers;
    const size_t layerCount = target->densityCurvesLayers.size();
    for (size_t layer = 0; layer < layerCount; ++layer) {
        auto& dstLayer = target->densityCurvesLayers[layer];
        const auto& srcLayer = base.densityCurvesLayers[layer];
        const size_t channelCount = dstLayer.size();
        auto* dstChannel = dstLayer.data();
        const auto* srcChannel = srcLayer.data();
        for (size_t ch = 0; ch < channelCount; ++ch, ++dstChannel, ++srcChannel) {
            if (target->hasDensityCurvesLayers) {
                *dstChannel = *srcChannel;
            }
            else {
                dstChannel->clear();
            }
        }
    }

    auto average_positive = [](const auto& values) -> float {
        float sum = 0.0f;
        int count = 0;
        const float* data = values.data();
        const float* const dataEnd = data + values.size();
        for (; data < dataEnd; ++data) {
            const float v = *data;
            if (is_finite(v) && v > 1e-6f) {
                sum += v;
                ++count;
            }
        }
        return (count > 0) ? (sum / static_cast<float>(count)) : 0.0f;
        };

    const float baselineMixReference = !base.densityMidNeutral.empty()
        ? average_positive(base.densityMidNeutral)
        : 0.0f;
    const float printBaselineMixReference =
        (printRT.profile.hasBaseline && printRT.hasMidNeutralDensity)
        ? average_positive(printRT.midNeutralDensity)
        : 0.0f;

        Spectral::build_tables_from_curves_non_global(
            /*epsY*/ epsY, /*epsM*/ epsM, /*epsC*/ epsC,
            /*xbar*/ Spectral::gXBar, /*ybar*/ Spectral::gYBar, /*zbar*/ Spectral::gZBar,
            /*illumView*/ printRT.illumView,
            /*baseMin*/ baseMin, /*baseMid*/ baseMid, /*hasBaseline*/ hasBaseline,
            baselineMixReference,
            target->tablesView,
            printScannerIlluminant.hash);

    if (hasRefIlluminant) {
        Spectral::build_tables_from_curves_non_global(
            /*epsY*/ epsY, /*epsM*/ epsM, /*epsC*/ epsC,
            /*xbar*/ Spectral::gXBar, /*ybar*/ Spectral::gYBar, /*zbar*/ Spectral::gZBar,
            /*illumView*/ illumRef,
            /*baseMin*/ baseMin, /*baseMid*/ baseMid, /*hasBaseline*/ hasBaseline,
            baselineMixReference,
            target->tablesRef);

        const bool validWhite =
            triplet_is_finite(target->tablesRef.whiteXYZ) &&
            target->tablesRef.whiteXYZ[1] > 0.0f;
        if (!validWhite) {
            JTRACE("BUILD", "reference illuminant produced invalid white XYZ; disabling SPD for safety");
            target->tablesRef = Spectral::SpectralTables{};
            hasRefIlluminant = false;
        }
    }
    else {
        target->tablesRef = Spectral::SpectralTables{};
    }

    std::shared_ptr<Print::Runtime> printRuntimeCopy = std::make_shared<Print::Runtime>(printRT);
    Print::Profile printProfile = printRuntimeCopy->profile;
    Print::DensityCurves printCurves = printRuntimeCopy->densityCurvesRaw;
    Scanner::ScannerDensityRange printDensityRange;
    bool printRangeOk = false;
    bool printDensityOk = !printCurves.cyan.empty() &&
        !printCurves.magenta.empty() &&
        !printCurves.yellow.empty();
    bool printRuntimeOk = false;
    if (printDensityOk) {
        const float factor = static_cast<float>(clamp_finite_or(P.glareCompRemovalFactor, 0.0, 0.0, 1.0));
        const float density = static_cast<float>(clamp_finite_or(P.glareCompRemovalDensity, 1.2, 0.0, 3.0));
        const float transition = static_cast<float>(clamp_finite_or(P.glareCompRemovalTransition, 0.3, 0.0, 2.0));

        printProfile.glare.compensationRemovalFactor = factor;
        printProfile.glare.compensationRemovalDensity = density;
        printProfile.glare.compensationRemovalTransition = transition;
        printProfile.glareCompensationFactor = factor;
        printProfile.glareCompensationDensity = density;
        printProfile.glareCompensationTransition = transition;
        printProfile.hasGlareCompensation = (factor > 0.0f);

        if (factor > 0.0f) {
            const bool removed = Print::remove_glare_compensation_from_curves(printProfile, printCurves);
            if (!removed) {
                JTRACE("PRINT", "FATAL: failed to remove viewing glare compensation from print curves");
                printDensityOk = false;
            }
        }
    }
    if (printDensityOk) {
        printDensityOk = Print::rebuild_density_curves(printProfile, printCurves);
        if (printDensityOk) {
            printRangeOk = compute_print_density_range(printProfile, printDensityRange);
            printDensityOk = printRangeOk;
        }
    }
    Print::recompute_mid_neutral(printProfile, printRuntimeCopy.get());
    printRuntimeCopy->profile = printProfile;
    printRuntimeCopy->glare = printProfile.glare;

    if (printDensityOk &&
        Print::profile_is_valid(printProfile) &&
        printRuntimeCopy->illumView.linear.size() == static_cast<size_t>(Spectral::gShape.K))
    {
        const float printDminFactor = static_cast<float>(clamp_finite_or(P.printDminFactor, 0.4, 0.0, 1.0));
        if (printProfile.hasBaseline && !approx_equal(printDminFactor, 1.0f)) {
            scale_finite_curve_samples(printProfile.baseMin, printDminFactor);
        }
        if (printProfile.hasBaseline) {
            clamp_negative_finite_curve_samples(printProfile.baseMin);
            clamp_negative_finite_curve_samples(printProfile.baseMid);
        }
        printRuntimeCopy->profile = printProfile;
        Spectral::build_tables_from_curves_non_global(
            /*epsY*/ printProfile.epsY,
            /*epsM*/ printProfile.epsM,
            /*epsC*/ printProfile.epsC,
            /*xbar*/ Spectral::gXBar, /*ybar*/ Spectral::gYBar, /*zbar*/ Spectral::gZBar,
            /*illumView*/ printRuntimeCopy->illumView,
            /*baseMin*/ printProfile.baseMin,
            /*baseMid*/ printProfile.baseMid,
            /*hasBaseline*/ printProfile.hasBaseline,
            printBaselineMixReference,
            target->tablesPrint,
            printScannerIlluminant.hash);
        printRuntimeOk = true;
    }
    else {
        if (!printDensityOk) {
            JTRACE("BUILD", "FATAL: missing spectral data (print profile) after glare processing");
        }
        else {
            JTRACE("BUILD", "FATAL: print profile invalid or viewing illuminant missing");
        }
        target->tablesPrint = Spectral::SpectralTables{};
    }

    Spectral::Curve illumScan = negativeScannerIlluminant.curve;
    if (illumScan.linear.size() != static_cast<size_t>(Spectral::gShape.K)) {
        JTRACE("BUILD", "FATAL: scanner illuminant for negative medium is invalid");
        return;
    }
    Spectral::build_tables_from_curves_non_global(
        /*epsY*/ epsY, /*epsM*/ epsM, /*epsC*/ epsC,
        /*xbar*/ Spectral::gXBar, /*ybar*/ Spectral::gYBar, /*zbar*/ Spectral::gZBar,
        /*illumView*/ illumScan,
        /*baseMin*/ baseMin, /*baseMid*/ baseMid, /*hasBaseline*/ hasBaseline,
        baselineMixReference,
        target->tablesScan,
        negativeScannerIlluminant.hash);

    if (hasRefIlluminant && target->tablesRef.K > 0) {
        const bool validWhite =
            normalized_white_triplet_is_valid(target->tablesRef.whiteXYZ);

        const bool validRefWhite =
            normalized_white_triplet_is_valid(target->tablesRef.refIllumWhiteXYZ);

        if (validWhite && validRefWhite) {
            Spectral::compute_S_inverse_from_tables(target->tablesRef, target->spdSInv);
            target->spdReady = true;
        }
        else {
            target->spdReady = false;
            JTRACE("BUILD", "Reference white XYZ validation failed; disabling SPD exposure");
        }
    }
    else {
        target->spdReady = false;
    }

    if (target->tablesView.K <= 0) {
        JTRACE("BUILD", "tablesView not ready; will cause wsReady=0 in render.");
    }
    if (hasRefIlluminant) {
        if (!target->spdReady || target->tablesRef.K <= 0) {
            JTRACE("BUILD", "tablesRef or spdSInv not ready; SPD exposure disabled.");
        }
    }
    else {
        JTRACE("BUILD", "reference illuminant missing; SPD exposure disabled.");
    }

    {
        auto all_finite_curve = [](const Spectral::Curve& c)->bool {
            const float* values = c.linear.data();
            const float* const valuesEnd = values + c.linear.size();
            for (; values < valuesEnd; ++values) {
                const float v = *values;
                if (!is_finite(v)) {
                    return false;
                }
            }
            return true;
            };
        auto density_curve_ok = [](const Spectral::Curve& c)->bool {
            // agx-emulsion parity: density curves may contain toe NaNs; allow NaNs but reject
            // infinities and require at least one finite sample for calibration.
            bool anyFinite = false;
            const float* values = c.linear.data();
            const float* const valuesEnd = values + c.linear.size();
            for (; values < valuesEnd; ++values) {
                const float v = *values;
                if (std::isinf(v)) {
                    return false;
                }
                if (is_finite(v)) {
                    anyFinite = true;
                }
            }
            return anyFinite;
            };

        const bool ok_dens = density_curve_ok(densB) && density_curve_ok(densG) && density_curve_ok(densR);
        const bool ok_sens = all_finite_curve(sensB) && all_finite_curve(sensG) && all_finite_curve(sensR);
        const bool baseMidOk = baseMid.linear.empty() ||
            static_cast<int>(baseMid.linear.size()) == Spectral::gShape.K;
        const bool ok_base = !hasBaseline ||
            (static_cast<int>(baseMin.linear.size()) == Spectral::gShape.K && baseMidOk);
        const bool ok_tables =
            (target->tablesView.K == Spectral::gShape.K) &&
            (target->tablesScan.K == Spectral::gShape.K) &&
            (!target->spdReady || target->tablesRef.K == Spectral::gShape.K);

        if (buildTraceEnabled) {
            std::ostringstream oss;
            oss << "pre-Ecal: ok_dens=" << (ok_dens ? 1 : 0)
                << " ok_sens=" << (ok_sens ? 1 : 0)
                << " ok_base=" << (ok_base ? 1 : 0)
                << " ok_tables=" << (ok_tables ? 1 : 0)
                << " spdReady=" << (target->spdReady ? 1 : 0);
            JTRACE("BUILD", oss.str());
        }

        if (!(ok_dens && ok_sens && ok_base && ok_tables)) {
            JTRACE("BUILD", "pre-Ecal: invalid inputs; aborting rebuild to avoid crash");
            return;
        }
    }

    {
        bool ok_spd = true;
        const float* spdInvIt = target->spdSInv;
        const float* const spdInvEnd = spdInvIt + 9;
        for (; spdInvIt != spdInvEnd; ++spdInvIt) {
            if (!is_finite(*spdInvIt)) { ok_spd = false; break; }
        }
        const bool ok_invYn =
            is_positive_finite(target->tablesView.invYn) &&
            is_positive_finite(target->tablesScan.invYn) &&
            (!target->spdReady || is_positive_finite(target->tablesRef.invYn));

        if (buildTraceEnabled) {
            std::ostringstream oss;
            oss << "pre-Ecal: ok_spd=" << (ok_spd ? 1 : 0)
                << " ok_invYn=" << (ok_invYn ? 1 : 0);
            JTRACE("BUILD", oss.str());
        }

        if (!ok_spd || !ok_invYn) {
            JTRACE("BUILD", "pre-Ecal: invalid S_inv or invYn; aborting rebuild");
            return;
        }
    }

    std::array<float, 3> densityMidRGB{ {0.0f, 0.0f, 0.0f} };
    bool hasDensityMid = !base.densityMidNeutral.empty();
    if (hasDensityMid) {
        float seed = 0.0f;
        if (is_finite(base.densityMidNeutral.front())) {
            seed = base.densityMidNeutral.front();
        }
        densityMidRGB.fill(seed);
        const size_t count = std::min<size_t>(static_cast<size_t>(3), base.densityMidNeutral.size());
        const float* midNeutralData = base.densityMidNeutral.data();
        float* densityMidData = densityMidRGB.data();
        for (size_t i = 0; i < count; ++i, ++midNeutralData, ++densityMidData) {
            const float v = *midNeutralData;
            if (is_finite(v)) {
                *densityMidData = v;
            }
        }
    }
    std::array<float, 3> densityOffsets{ {0.0f, 0.0f, 0.0f} };
    if (hasDensityMid) {
        densityOffsets = RebuildWorkingState::compute_mid_neutral_logE_offsets_rgb(
            densRForCalibration, densGForCalibration, densBForCalibration, densityMidRGB);
    }

    std::array<float, 3> logMidRGB{ {0.0f, 0.0f, 0.0f} };
    bool hasLogEMid = !base.logExposureMidNeutral.empty();
    if (hasLogEMid) {
        float seed = 0.0f;
        bool seedValid = false;
        if (is_finite(base.logExposureMidNeutral.front())) {
            seed = base.logExposureMidNeutral.front();
            seedValid = true;
        }
        logMidRGB.fill(seed);
        size_t finiteCount = seedValid ? 1u : 0u;
        const size_t count = std::min<size_t>(static_cast<size_t>(3), base.logExposureMidNeutral.size());
        const float* logMidData = base.logExposureMidNeutral.data();
        float* outLogMidData = logMidRGB.data();
        for (size_t i = 0; i < count; ++i, ++logMidData, ++outLogMidData) {
            const float v = *logMidData;
            if (is_finite(v)) {
                *outLogMidData = v;
                ++finiteCount;
            }
        }
        if (!seedValid && finiteCount == 0u) {
            hasLogEMid = false;
            logMidRGB = { {0.0f, 0.0f, 0.0f} };
        }
    }

    std::array<float, 3> offsetsRGB{ {0.0f, 0.0f, 0.0f} };
    bool usingLogEMetadata = false;
    if (hasLogEMid) {
        offsetsRGB = logMidRGB;
        usingLogEMetadata = true;
    }
    else if (hasDensityMid) {
        offsetsRGB = densityOffsets;
    }

    const float offR = offsetsRGB[0];
    const float offG = offsetsRGB[1];
    const float offB = offsetsRGB[2];

    if (buildTraceEnabled) {
        std::ostringstream oss;
        if (usingLogEMetadata) {
            oss << "negative logE offsets B/G/R=" << offB << "/" << offG << "/" << offR
                << " (log_exposure_midscale_neutral)";
            if (hasDensityMid) {
                oss << " density_midscale_neutral=" << densityOffsets[2] << "/"
                    << densityOffsets[1] << "/" << densityOffsets[0];
            }
        }
        else if (hasDensityMid) {
            oss << "negative logE offsets B/G/R=" << offB << "/" << offG << "/" << offR
                << " (density_midscale_neutral)";
        }
        else {
            oss << "negative logE offsets B/G/R=" << offB << "/" << offG << "/" << offR
                << " (no midscale metadata)";
        }
        JTRACE("BUILD", oss.str());
    }

    target->densB = std::move(densB);
    target->densG = std::move(densG);
    target->densR = std::move(densR);
    if (precorrectApplied) {
        target->dirDensB = std::move(dirDensB);
        target->dirDensG = std::move(dirDensG);
        target->dirDensR = std::move(dirDensR);
    }
    else {
        target->dirDensB = target->densB;
        target->dirDensG = target->densG;
        target->dirDensR = target->densR;
    }
    target->sensB = std::move(sensB);
    target->sensG = std::move(sensG);
    target->sensR = std::move(sensR);
    // negSensB/G/R removed: profiles are pre-balanced, no "before balance" state needed.
    // Per agx-emulsion parity, sensitivities are used as-is from profiles.
    target->negSensB.linear.clear();
    target->negSensG.linear.clear();
    target->negSensR.linear.clear();
    target->baseMin = std::move(baseMin);
    target->baseMid = std::move(baseMid);
    target->hasBaseline = hasBaseline;
    target->baselineMixReference = baselineMixReference;
    target->printBaselineMixReference = printBaselineMixReference;

    // Copy per-channel gamma factors for density curve interpolation (agx-emulsion parity)
    target->gammaFactorB = base.gammaFactor[0];
    target->gammaFactorG = base.gammaFactor[1];
    target->gammaFactorR = base.gammaFactor[2];

    target->dirRT = dirRT;
    target->dirPrecorrected = precorrectApplied;
    copy_float3(target->dMax, dirRT.dMax);

    negativeRangeOk = compute_negative_density_range(
        target->densB, target->densG, target->densR, target->grain, negativeDensityRange);
    if (!negativeRangeOk) {
        return;
    }

    {
        RebuildWorkingState::NegativeReuseContext reuseCtx;
        reuseCtx.activeBuildCounter = snapshot.activeBuildCounter;
        reuseCtx.lastHash = snapshot.lastHash;
        reuseCtx.lastFilmStock = snapshot.lastParams.filmStockIndex;
        reuseCtx.lastEnlargerIll = snapshot.lastParams.enlIll;

        const std::shared_ptr<const WorkingState> prev = snapshot.activeWorkingState;
        if (RebuildWorkingState::can_reuse_negative_params(
            reuseCtx, prev.get(), *target, P.filmStockIndex, P.enlIll)) {
            // Only reuse metadata that is guaranteed to be identical. Density
            // curves and dMax are left untouched so freshly computed values stay
            // active after rebuilds (agx-emulsion parity for stock/illuminant swaps).
            target->negParams = prev->negParams;
        }
    }

    {
        sanitize_dir_matrix(target->dirRT.M);
        sanitize_dir_dmax(target->dMax, target->dirRT.dMax);
    }

    target->filmRaw = Spectral::FilmRawConfig{};
    target->filmRaw.inputColorSpace = Spectral::inputColorSpaceFromIndex(P.inputColorSpace);
    target->filmRaw.applyCctfDecoding = (P.inputCctfDecoding != 0);
    target->filmRaw.spectralUpsamplingMode = Spectral::spectral_upsampling_mode_from_index(P.spectralUpsamplingMode);
    Spectral::prepare_film_raw_config(target->filmRaw);

    if (target->spdReady && target->tablesRef.K > 0) {
        std::memcpy(
            target->filmRaw.refIllumWhiteXYZ,
            target->tablesRef.refIllumWhiteXYZ,
            3u * sizeof(float));
        target->filmRaw.hasRefIllumWhite = true;
    }
    else {
        copy_float3(target->filmRaw.refIllumWhiteXYZ, Spectral::gDWG_WhitePoint_XYZ);
        target->filmRaw.hasRefIllumWhite = false;
    }

    // Per agx-emulsion parity: use the same sensitivities everywhere (no separate "before balance" state).
    // Profiles contain pre-balanced sensitivities; mid-gray computation must match actual render.
    const Spectral::SpectralTables* tablesSPD =
        (target->spdReady && target->tablesRef.K > 0) ? &target->tablesRef : nullptr;
    const float* sInv = target->spdReady ? target->spdSInv : nullptr;
    Spectral::compute_film_raw_midgray(
        target->filmRaw,
        tablesSPD,
        sInv,
        target->sensB,
        target->sensG,
        target->sensR);

    if (buildTraceEnabled) {
        std::ostringstream oss;
        oss << "film raw midgray scale=" << target->filmRaw.midgrayScale
            << " rawMidGreen=" << target->filmRaw.rawMidgrayGreen;
        JTRACE("BUILD", oss.str());
    }

    target->printRT = std::move(printRuntimeCopy);
    target->printGlare = target->printRT ? target->printRT->glare : Profiles::ProfileGlare{};
    target->negativeScannerIlluminant = negativeScannerIlluminant;
    target->printScannerIlluminant = printScannerIlluminant;
    target->negativeDensityRange = negativeDensityRange;
    target->printDensityRange = printDensityRange;

    target->negativeScannerValid = true;
    target->printScannerValid = printRuntimeOk;
    target->printGlareCompensated = (printRuntimeOk && printProfile.glare.compensationRemovalFactor > 0.0f);
    if (!rebuild_working_state_scanner_output_runtime(P, *target)) {
        return;
    }

    target->fullHash = hash_params(P);
    target->uploadCoreHash = hash_params_upload_core(P);
    target->coreHash = coreShareHash;
    target->coreShareHash = coreShareHash;
    target->dirHash = hash_params_dir(P);
    target->buildCounter = S.buildCounterNext.fetch_add(1, std::memory_order_relaxed) + 1;
    {
        auto corePayload = std::make_shared<WorkingStateSharing::WorkingStateCorePayload>();
        WorkingStateSharing::capture_working_state_core_payload(*target, *corePayload);
        const WorkingStateSharing::AcquireCoreSharedResult coreShareSeed =
            WorkingStateSharing::acquire_or_create_shared_core(target->coreShareHash, std::move(corePayload));
        target->sharedCore = coreShareSeed.sharedCore;
        trace_working_state_core_share(coreShareInitial, target->buildCounter, "full_rebuild_shell_acquire");
        trace_working_state_core_share(coreShareSeed, target->buildCounter, "full_rebuild");
    }
    trace_print_working_state_commit();

    if (buildTraceEnabled) {
        std::ostringstream oss;
        oss << "WorkingState build #" << target->buildCounter;
        JTRACE("BUILD", oss.str());
    }

    publish_rebuilt_working_state(S, P, next, /*invalidateSpatialSigmaCache*/true);
    if (buildTraceEnabled) {
        std::ostringstream oss;
        oss << "activeWorkingState swapped; buildCounter=" << static_cast<long long>(target->buildCounter);
        JTRACE("BUILD", oss.str());
    }
}

void rebuild_working_state_couplers_only(OfxImageEffectHandle instance, InstanceState& S, const ParamSnapshot& P) {
    (void)instance;

#ifndef JUICER_ENABLE_COUPLERS
    rebuild_working_state(instance, S, P);
    return;
#else
    Spectral::SpectralMutationScope mutationScope(
        Spectral::SpectralMutationStage::Rebuild,
        "rebuild_working_state_couplers_only");
    (void)mutationScope;

    JTRACE_SCOPE("BUILD", "rebuild_working_state_couplers_only");

    std::unique_lock<std::mutex> rebuildLock(S.rebuildMutex);
    std::shared_ptr<WorkingState> next = std::make_shared<WorkingState>();
    WorkingState* target = next.get();
    RebuildStateSnapshot snapshot{};
    {
        std::lock_guard<std::mutex> stateLock(S.m);
        snapshot = snapshot_rebuild_state_locked(S);
    }
    const std::uint64_t coreShareHash = hash_params_core(P);
    WorkingStateSharing::AcquireCoreSharedResult coreShare =
        WorkingStateSharing::acquire_or_create_shared_core(coreShareHash);
    const WorkingStateSharing::AcquireCoreSharedResult coreShareInitial = coreShare;
    std::shared_ptr<const WorkingState> src;
    if (!(coreShare.sharedCore && coreShare.sharedCore->payload)) {
        src = snapshot.activeWorkingState;
        if (!src || src->buildCounter == 0) {
            rebuildLock.unlock();
            rebuild_working_state(instance, S, P);
            return;
        }
        auto payloadSeed = std::make_shared<WorkingStateSharing::WorkingStateCorePayload>();
        WorkingStateSharing::capture_working_state_core_payload(*src, *payloadSeed);
        coreShare = WorkingStateSharing::acquire_or_create_shared_core(coreShareHash, std::move(payloadSeed));
    }
    if (coreShare.sharedCore && coreShare.sharedCore->payload) {
        WorkingStateSharing::apply_working_state_core_payload(*coreShare.sharedCore->payload, *target);
    }
    else {
        WorkingStateSharing::WorkingStateCorePayload fallbackPayload{};
        WorkingStateSharing::capture_working_state_core_payload(*src, fallbackPayload);
        WorkingStateSharing::apply_working_state_core_payload(fallbackPayload, *target);
    }
    target->coreShareHash = coreShareHash;
    target->sharedCore = coreShare.sharedCore;
    recompute_working_state_dir_overlay(snapshot, P, *target);
    if (!rebuild_working_state_scanner_output_runtime(P, *target)) {
        rebuildLock.unlock();
        rebuild_working_state(instance, S, P);
        return;
    }

    target->fullHash = hash_params(P);
    target->uploadCoreHash = hash_params_upload_core(P);
    target->coreHash = coreShareHash;
    target->coreShareHash = coreShareHash;
    target->dirHash = hash_params_dir(P);
    target->buildCounter = S.buildCounterNext.fetch_add(1, std::memory_order_relaxed) + 1;
    target->sharedCore = coreShare.sharedCore;
    trace_working_state_core_share(coreShareInitial, target->buildCounter, "couplers_only_shell_acquire");
    trace_working_state_core_share(coreShare, target->buildCounter, "couplers_only");

    publish_rebuilt_working_state(S, P, next, /*invalidateSpatialSigmaCache*/true);
#endif
}
