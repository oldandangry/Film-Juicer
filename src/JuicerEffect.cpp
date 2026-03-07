#include "JuicerEffect.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <atomic>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "JuicerState.h"
#include "ColorTransforms.h"
#include "Couplers.h"
#include "Illuminants.h"
#include "IlluminantKeys.h"
#include "NeutralFilters.h"
#include "OutputEncoding.h"
#include "Print.h"
#include "ParamNames.h"
#include "Scanner.h"
#include "SpectralData.h"
#include "Logging.h"
#include "Hash.h"
#include "mainProcessing.h"
#include "WorkingState.h"

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#endif

namespace {
    static std::once_flag gSpectralGlobalsOnce;

    enum class MeteringMethod : int {
        CenterWeighted = 0,
        Median = 1
    };

    const char* dichroic_dir_name_for_choice(int choice) {
        switch (choice) {
        case 1: return "thorlabs";
        case 2: return "edmund_optics";
        case 0:
        default: return "durst_digital_light";
        }
    }

    const char* enlarger_neutral_filters_json_for_choice(int choice) {
        switch (choice) {
        case 1: return "enlarger_neutral_ymc_filters_thorlabs.json";
        case 2: return "enlarger_neutral_ymc_filters_edmund.json";
        case 0:
        default: return "enlarger_neutral_ymc_filters.json";
        }
    }

    std::string profile_json_path_for_key_or_empty(const char* jsonKey) {
        if (!jsonKey) {
            return {};
        }
        std::string profileName(jsonKey);
        profileName += ".json";
        return data_dir_string("profiles", profileName);
    }

    inline const char* cstr_or_default_if_null(const char* value, const char* fallback);
    static int illuminant_choice_index_from_string(const std::string& value);

    void trace_dichroic_load_failure(
        const char* operation,
        const std::string& dichroicDir,
        const char* detail,
        const char* fallbackState) {

        if (!JTRACE_ENABLED(1)) {
            return;
        }
        const char* errorDetail = cstr_or_default_if_null(detail, "unknown error");
        std::string msg;
        msg.reserve(160 + dichroicDir.size());
        msg = cstr_or_default_if_null(operation, "dichroic load failed");
        msg += " at '";
        msg += dichroicDir;
        msg += "' (";
        msg += errorDetail;
        msg += "); ";
        msg += cstr_or_default_if_null(fallbackState, "using identity filters");
        JTRACE("PRINT", msg);
    }

    bool try_load_dichroic_filters(
        int dichroicSetChoice,
        Print::Runtime& runtime,
        const char* operation,
        const char* fallbackState) {
        const std::string dichroicDir = ensure_trailing_separator(
            data_dir_string("filters", "dichroics", dichroic_dir_name_for_choice(dichroicSetChoice)));
        try {
            Print::load_dichroic_filters_from_csvs(dichroicDir, runtime);
            return true;
        }
        catch (const std::exception& ex) {
            trace_dichroic_load_failure(operation, dichroicDir, ex.what(), fallbackState);
        }
        catch (...) {
            trace_dichroic_load_failure(operation, dichroicDir, nullptr, fallbackState);
        }
        return false;
    }

    inline bool nearly_equal_double(double a, double b) {
        const double diff = std::fabs(a - b);
        const double scale = std::max({ 1.0, std::fabs(a), std::fabs(b) });
        return diff <= scale * 1e-9;
    }

    inline bool is_finite(float value) {
        return std::isfinite(value);
    }

    inline bool is_finite(double value) {
        return std::isfinite(value);
    }

    inline int bool_to_i32(bool value) {
        return value ? 1 : 0;
    }

    inline const char* cstr_or_default_if_null(const char* value, const char* fallback) {
        return value ? value : fallback;
    }

    inline std::size_t cstr_len_or_zero(const char* value) {
        return value ? std::strlen(value) : 0u;
    }

    inline const char* cstr_or_default_if_empty(const std::string& value, const char* fallback) {
        return value.empty() ? fallback : value.c_str();
    }

    inline bool requires_nonfloat_copy(OFX::BitDepthEnum depth, int nComponents) {
        return depth != OFX::eBitDepthFloat || nComponents == 0;
    }

    inline bool param_events_suppressed(const InstanceState* state) {
        return state && state->suppressParamEvents;
    }

    inline bool bootstrap_in_progress(const InstanceState* state) {
        return state && state->inBootstrap;
    }

    inline void trace_changed_param_gate(
        bool traceInfo,
        const std::string& paramName,
        const char* prefix) {
        if (!traceInfo) {
            return;
        }
        std::string msg;
        msg.reserve(cstr_len_or_zero(prefix) + paramName.size() + 1);
        msg = cstr_or_default_if_null(prefix, "");
        msg += paramName;
        msg.push_back('\'');
        JTRACE("BUILD", msg);
    }

    inline bool param_name_is(const char* changedName, const char* expected) {
        return changedName && expected && (std::strcmp(changedName, expected) == 0);
    }

    inline bool param_name_is(const std::string& paramName, const char* expected) {
        return expected && (paramName == expected);
    }

    inline bool user_edit_param_is(bool userEdit, const std::string& paramName, const char* expected) {
        return userEdit && param_name_is(paramName, expected);
    }

    inline const std::string& first_nonempty_or(
        const std::string& primary,
        const std::string& secondary,
        const std::string& fallback) {
        if (!primary.empty()) {
            return primary;
        }
        if (!secondary.empty()) {
            return secondary;
        }
        return fallback;
    }

    inline std::uint64_t instance_token_or_zero(const InstanceState* state) {
        return state ? state->instanceToken : 0ull;
    }

    inline std::uint64_t working_state_build_counter_or_zero(const WorkingState* ws) {
        return ws ? ws->buildCounter : 0ull;
    }

    inline std::uint64_t working_state_build_counter_or_zero(const std::shared_ptr<const WorkingState>& ws) {
        return ws ? ws->buildCounter : 0ull;
    }

    inline std::uint64_t working_state_full_hash_or_zero(const std::shared_ptr<const WorkingState>& ws) {
        return ws ? ws->fullHash : 0ull;
    }

    inline std::uint64_t working_state_core_hash_or_zero(const std::shared_ptr<const WorkingState>& ws) {
        return ws ? ws->coreHash : 0ull;
    }

    inline std::uint64_t working_state_dir_hash_or_zero(const std::shared_ptr<const WorkingState>& ws) {
        return ws ? ws->dirHash : 0ull;
    }

    inline std::shared_ptr<const WorkingState> load_active_working_state_if(const InstanceState* state) {
        return state ? JuicerAtomic::load_shared_ptr(&state->activeWorkingState) : nullptr;
    }

    inline std::uint32_t frame_bounds_version_or_zero(const InstanceState* state) {
        return state ? state->frameBoundsVersion.load(std::memory_order_acquire) : 0u;
    }

    inline float print_runtime_value_or_zero(const Print::Runtime* runtime, float Print::Runtime::*field) {
        return runtime ? (runtime->*field) : 0.0f;
    }

    inline float scale_if_enabled_or_one(bool enabled, float scale) {
        return enabled ? scale : 1.0f;
    }

    inline void append_ymc_triplet(std::string& msg, float y, float m, float c) {
        msg += std::to_string(y);
        msg += "/";
        msg += std::to_string(m);
        msg += "/";
        msg += std::to_string(c);
    }

    inline std::string join_keys_csv_or_none(const std::vector<std::string>& keys) {
        std::string combined;
        size_t reserveHint = 0;
        const std::string* keyData = keys.data();
        const size_t keyCount = keys.size();
        for (size_t i = 0; i < keyCount; ++i, ++keyData) {
            reserveHint += keyData->size() + 1;
        }
        combined.reserve(reserveHint);
        keyData = keys.data();
        for (size_t i = 0; i < keyCount; ++i, ++keyData) {
            if (!combined.empty()) {
                combined += ",";
            }
            combined += *keyData;
        }
        if (combined.empty()) {
            combined = "<none>";
        }
        return combined;
    }

    inline std::string neutral_filter_prereq_context(
        const char* paperKey,
        const char* negativeKey,
        const std::string& illumChoices) {
        std::string msg;
        msg.reserve(96 + illumChoices.size());
        msg = "paper=";
        msg += cstr_or_default_if_null(paperKey, "<unset>");
        msg += " negative=";
        msg += cstr_or_default_if_null(negativeKey, "<unset>");
        msg += " illum_choices=";
        msg += illumChoices;
        return msg;
    }

    inline std::string neutral_filter_missing_context(
        const char* paperKey,
        const char* negativeKey,
        const std::string& illumKeys) {
        std::string msg;
        msg.reserve(96 + illumKeys.size());
        msg = "paper=";
        msg += cstr_or_default_if_null(paperKey, "<unset>");
        msg += " illuminant_keys=";
        msg += illumKeys;
        msg += " negative=";
        msg += cstr_or_default_if_null(negativeKey, "<unset>");
        return msg;
    }

    struct ProfileKeyLabels {
        const char* paperKey = nullptr;
        const char* filmKey = nullptr;
        const char* paperLabel = "<null>";
        const char* filmLabel = "<null>";
    };

    inline ProfileKeyLabels resolve_profile_key_labels(int printPaperIndex, int filmStockIndex) {
        ProfileKeyLabels labels{};
        labels.paperKey = print_paper_json_key_for_index(printPaperIndex);
        labels.filmKey = negative_json_key_for_stock_index(filmStockIndex);
        labels.paperLabel = cstr_or_default_if_null(labels.paperKey, "<null>");
        labels.filmLabel = cstr_or_default_if_null(labels.filmKey, "<null>");
        return labels;
    }

    inline ProfileKeyLabels resolve_profile_key_labels(const ParamSnapshot& snapshot) {
        return resolve_profile_key_labels(snapshot.printPaperIndex, snapshot.filmStockIndex);
    }

    struct PrintProfileLoadInputs {
        ProfileKeyLabels labels{};
        std::string printDir;
        std::string printProfileJson;
    };

    inline void load_print_profile_into_runtime(
        const std::string& printDir,
        const std::string& printProfileJson,
        Print::Runtime& runtime,
        bool moveMidNeutralVectors);

    inline PrintProfileLoadInputs build_print_profile_load_inputs(const ParamSnapshot& snapshot) {
        PrintProfileLoadInputs inputs{};
        inputs.labels = resolve_profile_key_labels(snapshot);
        inputs.printDir = print_dir_for_index(snapshot.printPaperIndex);
        inputs.printProfileJson = profile_json_path_for_key_or_empty(inputs.labels.paperKey);
        return inputs;
    }

    inline PrintProfileLoadInputs load_print_profile_for_snapshot(
        const ParamSnapshot& snapshot,
        Print::Runtime& runtime,
        bool moveMidNeutralVectors) {
        PrintProfileLoadInputs inputs = build_print_profile_load_inputs(snapshot);
        load_print_profile_into_runtime(
            inputs.printDir,
            inputs.printProfileJson,
            runtime,
            moveMidNeutralVectors);
        return inputs;
    }

    inline void sync_print_runtime_mid_neutral_from_profile(Print::Runtime& runtime, bool moveVectors) {
        runtime.hasMidNeutralDensity = runtime.profile.hasMidNeutralDensity;
        runtime.hasMidNeutralLogE = runtime.profile.hasMidNeutralLogE;
        if (moveVectors) {
            runtime.midNeutralDensity = std::move(runtime.profile.midNeutralDensity);
            runtime.midNeutralLogE = std::move(runtime.profile.midNeutralLogE);
        }
        else {
            runtime.midNeutralDensity = runtime.profile.midNeutralDensity;
            runtime.midNeutralLogE = runtime.profile.midNeutralLogE;
        }
    }

    inline void load_print_profile_into_runtime(
        const std::string& printDir,
        const std::string& printProfileJson,
        Print::Runtime& runtime,
        bool moveMidNeutralVectors) {
        Print::load_profile_from_dir(printDir, runtime.profile, printProfileJson, &runtime);
        sync_print_runtime_mid_neutral_from_profile(runtime, moveMidNeutralVectors);
    }

    struct PendingStateSnapshot {
        ParamSnapshot params{};
        std::uint64_t fullHash = 0ull;
        std::uint64_t coreHash = 0ull;
        std::uint64_t dirHash = 0ull;
    };

    inline PendingStateSnapshot load_pending_state_snapshot(InstanceState& state) {
        PendingStateSnapshot snapshot{};
        std::lock_guard<std::mutex> lock(state.pending.m);
        snapshot.params = state.pending.params;
        snapshot.fullHash = state.pending.fullHash;
        snapshot.coreHash = state.pending.coreHash;
        snapshot.dirHash = state.pending.dirHash;
        return snapshot;
    }

    inline void store_pending_state_snapshot(
        InstanceState& state,
        const ParamSnapshot& params,
        std::uint64_t fullHash,
        std::uint64_t coreHash,
        std::uint64_t dirHash) {
        std::lock_guard<std::mutex> lock(state.pending.m);
        state.pending.params = params;
        state.pending.fullHash = fullHash;
        state.pending.coreHash = coreHash;
        state.pending.dirHash = dirHash;
    }

    inline bool pending_rebuild_required(const PendingStateSnapshot& pending, std::uint64_t builtFullHash) {
        return pending.fullHash != 0 && pending.fullHash != builtFullHash;
    }

    inline bool pending_dir_only_rebuild(
        const PendingStateSnapshot& pending,
        std::uint64_t builtCoreHash,
        std::uint64_t builtDirHash) {
        return (pending.coreHash != 0) && (builtCoreHash != 0) &&
            (pending.coreHash == builtCoreHash) &&
            (pending.dirHash != 0) && (pending.dirHash != builtDirHash);
    }

    inline void rebuild_pending_working_state(
        JuicerEffect& effect,
        InstanceState& state,
        const PendingStateSnapshot& pending,
        bool dirOnly) {
        if (dirOnly) {
            rebuild_working_state_couplers_only(effect.getHandle(), state, pending.params);
        }
        else {
            rebuild_working_state(effect.getHandle(), state, pending.params);
        }
    }

    inline void rebuild_pending_state_if_needed(JuicerEffect& effect, InstanceState& state) {
        const PendingStateSnapshot pending = load_pending_state_snapshot(state);
        const std::shared_ptr<const WorkingState> wsCur = load_active_working_state_if(&state);
        const std::uint64_t builtFullHash = working_state_full_hash_or_zero(wsCur);
        if (!pending_rebuild_required(pending, builtFullHash)) {
            return;
        }
        const std::uint64_t builtCoreHash = working_state_core_hash_or_zero(wsCur);
        const std::uint64_t builtDirHash = working_state_dir_hash_or_zero(wsCur);
        const bool dirOnly = pending_dir_only_rebuild(pending, builtCoreHash, builtDirHash);
        rebuild_pending_working_state(effect, state, pending, dirOnly);
    }

    inline bool is_grain_preset_input_param(const std::string& paramName) {
        return param_name_is(paramName, JuicerParams::kGrainAmplitude) ||
            param_name_is(paramName, JuicerParams::kGrainBlur) ||
            param_name_is(paramName, JuicerParams::kGrainSharpness) ||
            param_name_is(paramName, JuicerParams::kGrainChroma) ||
            param_name_is(paramName, JuicerParams::kGrainTexture) ||
            param_name_is(paramName, JuicerParams::kGrainSublayersActive) ||
            param_name_is(paramName, JuicerParams::kGrainParticleAreaUm2) ||
            param_name_is(paramName, JuicerParams::kGrainParticleScaleMaster) ||
            param_name_is(paramName, JuicerParams::kGrainParticleScaleLayersMaster) ||
            param_name_is(paramName, JuicerParams::kGrainDensityMinMaster) ||
            param_name_is(paramName, JuicerParams::kGrainUniformityMaster) ||
            param_name_is(paramName, JuicerParams::kGrainParticleScale) ||
            param_name_is(paramName, JuicerParams::kGrainParticleScaleLayers) ||
            param_name_is(paramName, JuicerParams::kGrainDensityMin) ||
            param_name_is(paramName, JuicerParams::kGrainUniformity) ||
            param_name_is(paramName, JuicerParams::kGrainBlurDyeCloudsUm) ||
            param_name_is(paramName, JuicerParams::kGrainSizeMixWeight) ||
            param_name_is(paramName, JuicerParams::kGrainSizeMixWeightMid) ||
            param_name_is(paramName, JuicerParams::kGrainSizeMixScale) ||
            param_name_is(paramName, JuicerParams::kGrainMicroStructure) ||
            param_name_is(paramName, JuicerParams::kGrainClumpTemporalMix) ||
            param_name_is(paramName, JuicerParams::kGrainClumpMorphPeriodSec);
    }

    enum class GrainRatioMasterSelector {
        None = 0,
        Scale,
        ScaleLayers,
        DensityMin,
        Uniformity
    };

    inline GrainRatioMasterSelector grain_ratio_master_selector(const std::string& paramName) {
        if (param_name_is(paramName, JuicerParams::kGrainParticleScaleMaster)) {
            return GrainRatioMasterSelector::Scale;
        }
        if (param_name_is(paramName, JuicerParams::kGrainParticleScaleLayersMaster)) {
            return GrainRatioMasterSelector::ScaleLayers;
        }
        if (param_name_is(paramName, JuicerParams::kGrainDensityMinMaster)) {
            return GrainRatioMasterSelector::DensityMin;
        }
        if (param_name_is(paramName, JuicerParams::kGrainUniformityMaster)) {
            return GrainRatioMasterSelector::Uniformity;
        }
        return GrainRatioMasterSelector::None;
    }

    enum class HalationMasterSelector {
        None = 0,
        Strength,
        SizeUm,
        ScatteringStrength,
        ScatteringSizeUm
    };

    inline HalationMasterSelector halation_master_selector(const std::string& paramName) {
        if (param_name_is(paramName, JuicerParams::kHalationStrengthMaster)) {
            return HalationMasterSelector::Strength;
        }
        if (param_name_is(paramName, JuicerParams::kHalationSizeUmMaster)) {
            return HalationMasterSelector::SizeUm;
        }
        if (param_name_is(paramName, JuicerParams::kHalationScatteringStrengthMaster)) {
            return HalationMasterSelector::ScatteringStrength;
        }
        if (param_name_is(paramName, JuicerParams::kHalationScatteringSizeUmMaster)) {
            return HalationMasterSelector::ScatteringSizeUm;
        }
        return HalationMasterSelector::None;
    }

#ifdef JUICER_ENABLE_COUPLERS
    enum class CouplerParamKind {
        None = 0,
        Active,
        Amount,
        RatioB,
        RatioG,
        RatioR,
        Sigma,
        High,
        SpatialSigma
    };

    inline CouplerParamKind coupler_param_kind(const char* changedName) {
        using namespace Couplers;
        if (param_name_is(changedName, kParamCouplersActive)) {
            return CouplerParamKind::Active;
        }
        if (param_name_is(changedName, kParamCouplersAmount)) {
            return CouplerParamKind::Amount;
        }
        if (param_name_is(changedName, kParamCouplersAmountB)) {
            return CouplerParamKind::RatioB;
        }
        if (param_name_is(changedName, kParamCouplersAmountG)) {
            return CouplerParamKind::RatioG;
        }
        if (param_name_is(changedName, kParamCouplersAmountR)) {
            return CouplerParamKind::RatioR;
        }
        if (param_name_is(changedName, kParamCouplersLayerSigma)) {
            return CouplerParamKind::Sigma;
        }
        if (param_name_is(changedName, kParamCouplersHighExpShift)) {
            return CouplerParamKind::High;
        }
        if (param_name_is(changedName, kParamCouplersSpatialSigma)) {
            return CouplerParamKind::SpatialSigma;
        }
        return CouplerParamKind::None;
    }

    inline void mark_dirty_release(std::atomic<bool>& dirtyFlag) {
        dirtyFlag.store(true, std::memory_order_release);
    }

    inline bool mark_coupler_dirty_if_known(InstanceState& state, const char* changedName) {
        switch (coupler_param_kind(changedName)) {
        case CouplerParamKind::Active:
            mark_dirty_release(state.couplerDirty.active);
            return true;
        case CouplerParamKind::Amount:
            mark_dirty_release(state.couplerDirty.amount);
            return true;
        case CouplerParamKind::RatioB:
            mark_dirty_release(state.couplerDirty.ratioB);
            return true;
        case CouplerParamKind::RatioG:
            mark_dirty_release(state.couplerDirty.ratioG);
            return true;
        case CouplerParamKind::RatioR:
            mark_dirty_release(state.couplerDirty.ratioR);
            return true;
        case CouplerParamKind::Sigma:
            mark_dirty_release(state.couplerDirty.sigma);
            return true;
        case CouplerParamKind::High:
            mark_dirty_release(state.couplerDirty.high);
            return true;
        case CouplerParamKind::SpatialSigma:
            mark_dirty_release(state.couplerDirty.spatialSigma);
            return true;
        case CouplerParamKind::None:
        default:
            return false;
        }
    }

    inline bool is_coupler_param_name(const char* changedName) {
        return coupler_param_kind(changedName) != CouplerParamKind::None;
    }
#endif

    inline bool should_refresh_print_illuminant(bool printReloaded, bool filmReloaded) {
        return printReloaded || filmReloaded;
    }

    inline bool should_apply_neutral_after_reload(bool printReloaded, bool dichroicReloaded) {
        return printReloaded || dichroicReloaded;
    }

    inline bool auto_exposure_cache_param_changed(const std::string& paramName) {
        return param_name_is(paramName, kParamCameraAutoExposure) ||
            param_name_is(paramName, JuicerParams::kCameraMeteringMethod);
    }

    inline void update_print_illuminant_runtime(
        const ParamSnapshot& snapshot,
        Print::Runtime& runtime,
        const std::string& dataDir) {
        Print::build_illuminant_from_choice(snapshot.enlIll, runtime, dataDir, /*forEnlarger*/true);
    }

    struct ChangedParamFlags {
        bool hasName = false;
        bool referenceIlluminant = false;
        bool enlargerIlluminant = false;
        bool printPaper = false;
        bool enlargerDichroicSet = false;
        bool filmStock = false;
        bool couplerParam = false;
    };

    inline ChangedParamFlags classify_changed_param(const char* changedName) {
        ChangedParamFlags flags{};
        flags.hasName = (changedName != nullptr);
        if (!flags.hasName) {
            return flags;
        }

        flags.referenceIlluminant = param_name_is(changedName, kParamReferenceIlluminant);
        flags.enlargerIlluminant = param_name_is(changedName, kParamEnlargerIlluminant);
        flags.printPaper = param_name_is(changedName, kParamPrintPaper);
        flags.enlargerDichroicSet = param_name_is(changedName, kParamEnlargerDichroicSet);
        flags.filmStock = param_name_is(changedName, kParamFilmStock);
#ifdef JUICER_ENABLE_COUPLERS
        flags.couplerParam = is_coupler_param_name(changedName);
#endif
        return flags;
    }

    inline void mark_illuminant_override_if_changed(InstanceState& state, const ChangedParamFlags& changed) {
        if (changed.referenceIlluminant) {
            state.illuminantOverride.reference = true;
        }
        else if (changed.enlargerIlluminant) {
            state.illuminantOverride.enlarger = true;
        }
    }

    inline double sanitize_finite_clamped(double value, double fallback, double minValue, double maxValue) {
        if (!is_finite(value)) return fallback;
        return std::clamp(value, minValue, maxValue);
    }

    inline double sanitize_finite_or(double value, double fallback) {
        return is_finite(value) ? value : fallback;
    }

    inline double read_double_param_or(OFX::DoubleParam* param, double fallback) {
        double value = fallback;
        if (param) {
            param->getValue(value);
        }
        return value;
    }

    inline int read_int_param_or(OFX::IntParam* param, int fallback) {
        int value = fallback;
        if (param) {
            param->getValue(value);
        }
        return value;
    }

    inline double sanitize_positive_finite_or(double value, double fallback) {
        return (is_finite(value) && value > 0.0) ? value : fallback;
    }

    inline bool is_positive_finite(double value) {
        return is_finite(value) && value > 0.0;
    }

    inline float sanitize_nonnegative_finite_or(float value, float fallback) {
        return (is_finite(value) && value >= 0.0f) ? value : fallback;
    }

    inline double read_sanitized_double(
        OFX::DoubleParam* param,
        double fallback,
        double minValue,
        double maxValue)
    {
        double value = fallback;
        if (param) {
            param->getValue(value);
        }
        return sanitize_finite_clamped(value, fallback, minValue, maxValue);
    }

    inline float read_sanitized_float(
        OFX::DoubleParam* param,
        float fallback,
        double minValue,
        double maxValue) {
        return static_cast<float>(read_sanitized_double(
            param,
            static_cast<double>(fallback),
            minValue,
            maxValue));
    }

    inline bool read_bool_param_or(OFX::BooleanParam* param, bool fallback) {
        bool value = fallback;
        if (param) {
            param->getValue(value);
        }
        return value;
    }

    inline int read_choice_param_or(OFX::ChoiceParam* param, int fallback) {
        int value = fallback;
        if (param) {
            param->getValue(value);
        }
        return value;
    }

    inline int read_choice_param_clamped(
        OFX::ChoiceParam* param,
        int fallback,
        int minValue,
        int maxValue) {
        return std::clamp(read_choice_param_or(param, fallback), minValue, maxValue);
    }

    inline std::array<double, 3> read_double3_param_or(
        OFX::Double3DParam* param,
        const std::array<double, 3>& fallback) {
        std::array<double, 3> values = fallback;
        if (param) {
            param->getValue(values[0], values[1], values[2]);
        }
        return values;
    }

    inline std::array<double, 2> read_double2_param_or(
        OFX::Double2DParam* param,
        const std::array<double, 2>& fallback) {
        std::array<double, 2> values = fallback;
        if (param) {
            param->getValue(values[0], values[1]);
        }
        return values;
    }

    inline void set_bool_param_if(OFX::BooleanParam* param, bool value) {
        if (param) {
            param->setValue(value);
        }
    }

    inline void set_double_param_if(OFX::DoubleParam* param, double value) {
        if (param) {
            param->setValue(value);
        }
    }

    inline void set_choice_param_if(OFX::ChoiceParam* param, int value) {
        if (param) {
            param->setValue(value);
        }
    }

    inline bool apply_illuminant_choice_from_source(
        OFX::ChoiceParam* param,
        int& currentIndex,
        bool overrideFlag,
        const std::string& source) {
        if (overrideFlag || !param) {
            return false;
        }
        const int mapped = illuminant_choice_index_from_string(source);
        if (mapped < 0 || currentIndex == mapped) {
            return false;
        }
        set_choice_param_if(param, mapped);
        currentIndex = mapped;
        return true;
    }

    inline void set_double2_param_if(
        OFX::Double2DParam* param,
        double x,
        double y) {
        if (param) {
            param->setValue(x, y);
        }
    }

    inline void set_double3_param_if(
        OFX::Double3DParam* param,
        double x,
        double y,
        double z) {
        if (param) {
            param->setValue(x, y, z);
        }
    }

    inline void set_choice_label_if(OFX::ChoiceParam* param, const std::string& label) {
        if (param) {
            param->setLabel(label);
        }
    }

    inline void set_double_param_hint_if(OFX::DoubleParam* param, const std::string& hint) {
        if (param) {
            param->setHint(hint);
        }
    }

    inline void set_double_param_hint_if(OFX::DoubleParam* param, const char* hint) {
        if (param) {
            param->setHint(hint);
        }
    }

    inline void set_double_param_enabled_if(OFX::DoubleParam* param, bool enabled) {
        if (param) {
            param->setEnabled(enabled);
        }
    }

    inline std::array<double, 3> read_sanitized_double3(
        OFX::Double3DParam* param,
        const std::array<double, 3>& defaults,
        double minValue,
        double maxValue)
    {
        std::array<double, 3> values = defaults;
        if (param) {
            param->getValue(values[0], values[1], values[2]);
        }
        double* valueIt = values.data();
        const double* defaultIt = defaults.data();
        for (int i = 0; i < 3; ++i, ++valueIt, ++defaultIt) {
            *valueIt = sanitize_finite_clamped(*valueIt, *defaultIt, minValue, maxValue);
        }
        return values;
    }

    inline std::array<double, 2> read_sanitized_double2(
        OFX::Double2DParam* param,
        const std::array<double, 2>& defaults,
        double minValue,
        double maxValue)
    {
        std::array<double, 2> values = defaults;
        if (param) {
            param->getValue(values[0], values[1]);
        }
        double* valueIt = values.data();
        const double* defaultIt = defaults.data();
        for (int i = 0; i < 2; ++i, ++valueIt, ++defaultIt) {
            *valueIt = sanitize_finite_clamped(*valueIt, *defaultIt, minValue, maxValue);
        }
        return values;
    }

    template <size_t N>
    inline std::array<double, N> filled_double_array(double value) {
        std::array<double, N> out{};
        out.fill(value);
        return out;
    }

    template <size_t N>
    inline void cast_array(std::array<float, N>& dst, const std::array<double, N>& src) {
        float* dstIt = dst.data();
        const float* const dstEnd = dstIt + N;
        const double* srcIt = src.data();
        for (; dstIt < dstEnd; ++dstIt, ++srcIt) {
            *dstIt = static_cast<float>(*srcIt);
        }
    }

    template <size_t N>
    inline double mean_array(const std::array<double, N>& values) {
        double sum = 0.0;
        const double* valueIt = values.data();
        const double* const valueEnd = valueIt + N;
        for (; valueIt < valueEnd; ++valueIt) {
            sum += *valueIt;
        }
        return sum / static_cast<double>(N);
    }

    template <size_t N>
    inline std::array<double, N> sanitize_scaled_float_array_to_double(
        const std::array<float, N>& src,
        const std::array<double, N>& fallback,
        double scale,
        double minValue,
        double maxValue) {
        std::array<double, N> out{};
        double* outIt = out.data();
        const double* fallbackIt = fallback.data();
        const float* srcIt = src.data();
        const float* const srcEnd = srcIt + N;
        for (; srcIt < srcEnd; ++srcIt, ++outIt, ++fallbackIt) {
            const double scaled = static_cast<double>(*srcIt) * scale;
            *outIt = sanitize_finite_clamped(scaled, *fallbackIt, minValue, maxValue);
        }
        return out;
    }

    inline int pixel_component_count(OFX::PixelComponentEnum comps) {
        switch (comps) {
        case OFX::ePixelComponentRGBA: return 4;
        case OFX::ePixelComponentRGB: return 3;
        case OFX::ePixelComponentAlpha: return 1;
        default: return 0;
        }
    }

    inline bool span_x_within_bounds(int xStart, int xEnd, const OfxRectI& bounds) {
        return xStart >= bounds.x1 && xEnd <= bounds.x2;
    }

    inline bool row_has_full_coverage(const OfxRectI& bounds, int xStart, int xEnd, int y) {
        return span_x_within_bounds(xStart, xEnd, bounds) &&
            y >= bounds.y1 && y < bounds.y2;
    }

    template <typename T>
    inline T* row_ptr_if_fully_covered(
        OFX::Image* image,
        const OfxRectI& bounds,
        int xStart,
        int xEnd,
        int y) {
        if (!row_has_full_coverage(bounds, xStart, xEnd, y)) {
            return nullptr;
        }
        return reinterpret_cast<T*>(image->getPixelAddress(xStart, y));
    }

    inline const float* row_start_if_covered(
        OFX::Image* image,
        const OfxRectI& srcBounds,
        const OfxRectI& meterBounds,
        int y) {
        return row_ptr_if_fully_covered<const float>(
            image,
            srcBounds,
            meterBounds.x1,
            meterBounds.x2,
            y);
    }

    template <typename T>
    inline T* pixel_ptr(OFX::Image* image, int x, int y) {
        return reinterpret_cast<T*>(image->getPixelAddress(x, y));
    }

    inline void decode_input_pixel_linear(
        const float* pix,
        bool singleComponent,
        Spectral::InputColorSpace inputColorSpace,
        bool applyCctfDecoding,
        float linear[3]) {
        if (singleComponent) {
            const float gray = pix[0];
            const float grayRgb[3] = { gray, gray, gray };
            Spectral::apply_input_cctf_decoding(inputColorSpace, applyCctfDecoding, grayRgb, linear);
            return;
        }
        Spectral::apply_input_cctf_decoding(inputColorSpace, applyCctfDecoding, pix, linear);
    }

    inline bool decode_pixel_luma_if_finite(
        const float* pix,
        bool singleComponent,
        Spectral::InputColorSpace inputColorSpace,
        bool applyCctfDecoding,
        const Spectral::Mat3& rgbToXYZ,
        float& outY) {
        float linear[3];
        decode_input_pixel_linear(
            pix,
            singleComponent,
            inputColorSpace,
            applyCctfDecoding,
            linear);
        float XYZ[3];
        rgbToXYZ.mul(linear, XYZ);
        const float Y = XYZ[1];
        if (!is_finite(Y)) {
            return false;
        }
        outY = Y;
        return true;
    }

    inline void accumulate_weighted_luma_if_finite(
        const float* pix,
        bool singleComponent,
        Spectral::InputColorSpace inputColorSpace,
        bool applyCctfDecoding,
        const Spectral::Mat3& rgbToXYZ,
        double weight,
        double& sumY,
        double& sumMask) {
        float Y = 0.0f;
        if (!decode_pixel_luma_if_finite(
            pix,
            singleComponent,
            inputColorSpace,
            applyCctfDecoding,
            rgbToXYZ,
            Y)) {
            return;
        }
        sumY += static_cast<double>(Y) * weight;
        sumMask += weight;
    }

    inline void set_optional_sum_mask(double* outSumMask, double value) {
        if (outSumMask) {
            *outSumMask = value;
        }
    }

    inline double weighted_mean_or_zero(double sumY, double sumMask) {
        return (sumMask > 0.0) ? (sumY / sumMask) : 0.0;
    }

    inline double finalize_weighted_metering(double sumY, double sumMask, double* outSumMask) {
        set_optional_sum_mask(outSumMask, sumMask);
        return weighted_mean_or_zero(sumY, sumMask);
    }

    inline double return_zero_weighted_metering(double* outSumMask) {
        set_optional_sum_mask(outSumMask, 0.0);
        return 0.0;
    }

    inline float clamp_nonnegative(float value) {
        return (value < 0.0f) ? 0.0f : value;
    }

    inline void append_nonnegative_luma_if_finite(
        const float* pix,
        bool singleComponent,
        Spectral::InputColorSpace inputColorSpace,
        bool applyCctfDecoding,
        const Spectral::Mat3& rgbToXYZ,
        std::vector<float>& values) {
        float Y = 0.0f;
        if (!decode_pixel_luma_if_finite(
            pix,
            singleComponent,
            inputColorSpace,
            applyCctfDecoding,
            rgbToXYZ,
            Y)) {
            return;
        }
        values.emplace_back(clamp_nonnegative(Y));
    }

    inline void accumulate_weighted_image_pixel_if_present(
        OFX::Image* image,
        int x,
        int y,
        bool singleComponent,
        Spectral::InputColorSpace inputColorSpace,
        bool applyCctfDecoding,
        const Spectral::Mat3& rgbToXYZ,
        double weight,
        double& sumY,
        double& sumMask) {
        const float* pix = pixel_ptr<const float>(image, x, y);
        if (!pix) {
            return;
        }
        accumulate_weighted_luma_if_finite(
            pix,
            singleComponent,
            inputColorSpace,
            applyCctfDecoding,
            rgbToXYZ,
            weight,
            sumY,
            sumMask);
    }

    inline void append_nonnegative_image_pixel_luma_if_present(
        OFX::Image* image,
        int x,
        int y,
        bool singleComponent,
        Spectral::InputColorSpace inputColorSpace,
        bool applyCctfDecoding,
        const Spectral::Mat3& rgbToXYZ,
        std::vector<float>& values) {
        const float* pix = pixel_ptr<const float>(image, x, y);
        if (!pix) {
            return;
        }
        append_nonnegative_luma_if_finite(
            pix,
            singleComponent,
            inputColorSpace,
            applyCctfDecoding,
            rgbToXYZ,
            values);
    }

    inline double gaussian_weight(double normX, double normY, double invSigmaDenom) {
        const double r2 = normX * normX + normY * normY;
        return std::exp(-r2 * invSigmaDenom);
    }

    struct CenterWeightGeometry {
        double invWidth = 0.0;
        double invHeight = 0.0;
        double scaleX = 0.0;
        double scaleY = 0.0;
        double invSigmaDenom = 0.0;
    };

    inline bool build_center_weight_geometry(
        int width,
        int height,
        double sigma,
        CenterWeightGeometry& out) {
        if (width <= 0 || height <= 0 || !is_finite(sigma) || sigma <= 0.0) {
            return false;
        }
        const double maxDim = static_cast<double>(std::max(width, height));
        const double invMax = (maxDim > 0.0) ? (1.0 / maxDim) : 0.0;
        const double sigmaDenom = 2.0 * sigma * sigma;
        if (!is_finite(sigmaDenom) || sigmaDenom <= 0.0) {
            return false;
        }
        out.invWidth = 1.0 / static_cast<double>(width);
        out.invHeight = 1.0 / static_cast<double>(height);
        out.scaleX = static_cast<double>(width) * invMax;
        out.scaleY = static_cast<double>(height) * invMax;
        out.invSigmaDenom = 1.0 / sigmaDenom;
        return true;
    }

    inline double centered_norm_coordinate(int offset, double invExtent) {
        return static_cast<double>(offset) * invExtent - 0.5;
    }

    inline double gaussian_weight_from_nx(
        double nx,
        double scaleX,
        double normY,
        double invSigmaDenom) {
        const double normX = nx * scaleX;
        return gaussian_weight(normX, normY, invSigmaDenom);
    }

    inline float finite_exp2_scale(double ev) {
        const float scale = static_cast<float>(std::exp2(ev));
        return is_finite(scale) ? scale : 1.0f;
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

    inline bool has_nonzero_finite_dir_matrix(const float matrix[3][3]) {
        const float* valueIt = &matrix[0][0];
        const float* const valueEnd = valueIt + 9;
        for (; valueIt < valueEnd; ++valueIt) {
            const float value = *valueIt;
            if (is_finite(value) && value != 0.0f) {
                return true;
            }
        }
        return false;
    }

    constexpr std::uint64_t kAutoExposureMaskCacheMaxBytes = 96ull * 1024ull * 1024ull;
    constexpr std::size_t kAutoExposureMedianScratchMaxSamples =
        static_cast<std::size_t>(kAutoExposureMaskCacheMaxBytes / sizeof(float));
    static std::atomic<std::uint64_t> gAutoExposureMaskCacheResidentBytes{ 0 };

    inline bool auto_exposure_mask_cache_eligible(std::uint64_t requestedMaskBytes) {
        return (requestedMaskBytes > 0) && (requestedMaskBytes <= kAutoExposureMaskCacheMaxBytes);
    }

    std::vector<float>& auto_exposure_median_scratch() {
        thread_local std::vector<float> scratch;
        return scratch;
    }

    inline std::uint64_t mask_bytes_for_dimensions(int width, int height) {
        if (width <= 0 || height <= 0) {
            return 0;
        }
        const std::uint64_t w = static_cast<std::uint64_t>(width);
        const std::uint64_t h = static_cast<std::uint64_t>(height);
        if (w > (std::numeric_limits<std::uint64_t>::max() / h)) {
            return 0;
        }
        const std::uint64_t samples = w * h;
        const std::uint64_t sampleBytes = static_cast<std::uint64_t>(sizeof(double));
        if (samples > (std::numeric_limits<std::uint64_t>::max() / sampleBytes)) {
            return 0;
        }
        return samples * sampleBytes;
    }

    inline void update_auto_exposure_mask_resident_bytes(
        std::uint64_t previousBytes,
        std::uint64_t nextBytes) {

        if (nextBytes > previousBytes) {
            gAutoExposureMaskCacheResidentBytes.fetch_add(nextBytes - previousBytes, std::memory_order_relaxed);
        }
        else if (previousBytes > nextBytes) {
            const std::uint64_t delta = previousBytes - nextBytes;
            std::uint64_t observed = gAutoExposureMaskCacheResidentBytes.load(std::memory_order_relaxed);
            while (true) {
                const std::uint64_t updated = (observed > delta) ? (observed - delta) : 0;
                if (gAutoExposureMaskCacheResidentBytes.compare_exchange_weak(
                    observed,
                    updated,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed)) {
                    break;
                }
            }
        }
    }

    inline void trace_auto_exposure_mask_cache_event(
        const InstanceState* state,
        const char* event,
        int width,
        int height,
        std::uint64_t requestedBytes,
        std::uint64_t cachedBytes,
        std::uint64_t previousCachedBytes,
        const char* reason) {

        if (!JTRACE_ENABLED(2)) {
            return;
        }
        const std::uint64_t residentBytes =
            gAutoExposureMaskCacheResidentBytes.load(std::memory_order_relaxed);
        std::string msg;
        msg.reserve(256);
        msg = "event=";
        msg += cstr_or_default_if_null(event, "unknown");
        msg += " instance_token=";
        msg += std::to_string(instance_token_or_zero(state));
        msg += " width=";
        msg += std::to_string(width);
        msg += " height=";
        msg += std::to_string(height);
        msg += " requested_bytes=";
        msg += std::to_string(requestedBytes);
        msg += " previous_cached_bytes=";
        msg += std::to_string(previousCachedBytes);
        msg += " cached_bytes=";
        msg += std::to_string(cachedBytes);
        msg += " resident_bytes=";
        msg += std::to_string(residentBytes);
        msg += " cap_bytes=";
        msg += std::to_string(kAutoExposureMaskCacheMaxBytes);
        if (reason && reason[0] != '\0') {
            msg += " reason=";
            msg += reason;
        }
        JTRACE_LEVEL(2, "MSAEM", msg);
    }

    static double build_center_weight_mask(int width, int height, double sigma, std::vector<double>& outMask) {
        outMask.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
        CenterWeightGeometry geometry{};
        if (!build_center_weight_geometry(width, height, sigma, geometry)) {
            std::fill(outMask.begin(), outMask.end(), 0.0);
            return 0.0;
        }

        double sumMask = 0.0;
        for (int y = 0; y < height; ++y) {
            double* row = outMask.data() + static_cast<size_t>(y) * static_cast<size_t>(width);
            double* rowIt = row;
            const double ny = centered_norm_coordinate(y, geometry.invHeight);
            const double normY = ny * geometry.scaleY;
            for (int x = 0; x < width; ++x) {
                const double nx = centered_norm_coordinate(x, geometry.invWidth);
                const double w = gaussian_weight_from_nx(nx, geometry.scaleX, normY, geometry.invSigmaDenom);
                *rowIt++ = w;
                sumMask += w;
            }
        }
        return sumMask;
    }

    static double measure_center_weighted_Y_DWG_cached(
        OFX::Image* img,
        const OfxRectI& bounds,
        double sigma,
        InstanceState* state,
        double renderScaleX,
        double renderScaleY,
        std::uintptr_t clipToken,
        Spectral::InputColorSpace inputColorSpace,
        const Spectral::Mat3& rgbToXYZ,
        bool applyCctfDecoding) {

        if (!img) {
            return 0.0;
        }

        const int width = bounds.x2 - bounds.x1;
        const int height = bounds.y2 - bounds.y1;
        if (width <= 0 || height <= 0) {
            return 0.0;
        }
        const OfxRectI srcBounds = img->getBounds();
        const int nComponents = pixel_component_count(img->getPixelComponents());
        if (nComponents <= 0) {
            return 0.0;
        }
        const std::size_t pixelStride = static_cast<std::size_t>(nComponents);
        const bool singleComponent = (nComponents == 1);
        auto accumulate_weighted_Y = [&](const float* pix, double weight, double& sumY, double& sumMask) {
            accumulate_weighted_luma_if_finite(
                pix,
                singleComponent,
                inputColorSpace,
                applyCctfDecoding,
                rgbToXYZ,
                weight,
                sumY,
                sumMask);
            };

        auto accumulateYFromMask = [&](const std::vector<double>& mask, double* outSumMask) {
            double sumY = 0.0;
            double sumMask = 0.0;
            for (int yy = bounds.y1; yy < bounds.y2; ++yy) {
                const size_t rowOffset = static_cast<size_t>(yy - bounds.y1) * static_cast<size_t>(width);
                const double* maskRow = mask.data() + rowOffset;
                const float* rowPix = row_start_if_covered(img, srcBounds, bounds, yy);
                const double* maskIt = maskRow;
                if (rowPix) {
                    const float* rowPixIt = rowPix;
                    for (int xOff = 0; xOff < width; ++xOff) {
                        const double w = *maskIt++;
                        accumulate_weighted_Y(rowPixIt, w, sumY, sumMask);
                        rowPixIt += pixelStride;
                    }
                    continue;
                }
                for (int xOff = 0; xOff < width; ++xOff) {
                    const int xx = bounds.x1 + xOff;
                    const double w = *maskIt++;
                    accumulate_weighted_image_pixel_if_present(
                        img,
                        xx,
                        yy,
                        singleComponent,
                        inputColorSpace,
                        applyCctfDecoding,
                        rgbToXYZ,
                        w,
                        sumY,
                        sumMask);
                }
            }
            return finalize_weighted_metering(sumY, sumMask, outSumMask);
        };

        auto accumulateYUncached = [&](double* outSumMask) {
            CenterWeightGeometry geometry{};
            if (!build_center_weight_geometry(width, height, sigma, geometry)) {
                return return_zero_weighted_metering(outSumMask);
            }

            double sumY = 0.0;
            double sumMask = 0.0;
            for (int yy = bounds.y1; yy < bounds.y2; ++yy) {
                const int localY = yy - bounds.y1;
                const double ny = centered_norm_coordinate(localY, geometry.invHeight);
                const double normY = ny * geometry.scaleY;
                const float* rowPix = row_start_if_covered(img, srcBounds, bounds, yy);
                double nx = -0.5;
                if (rowPix) {
                    const float* rowPixIt = rowPix;
                    for (int xOff = 0; xOff < width; ++xOff) {
                        const double w = gaussian_weight_from_nx(nx, geometry.scaleX, normY, geometry.invSigmaDenom);
                        accumulate_weighted_Y(rowPixIt, w, sumY, sumMask);
                        rowPixIt += pixelStride;
                        nx += geometry.invWidth;
                    }
                    continue;
                }
                for (int xOff = 0; xOff < width; ++xOff) {
                    const int xx = bounds.x1 + xOff;
                    const double w = gaussian_weight_from_nx(nx, geometry.scaleX, normY, geometry.invSigmaDenom);
                    accumulate_weighted_image_pixel_if_present(
                        img,
                        xx,
                        yy,
                        singleComponent,
                        inputColorSpace,
                        applyCctfDecoding,
                        rgbToXYZ,
                        w,
                        sumY,
                        sumMask);
                    nx += geometry.invWidth;
                }
            }
            return finalize_weighted_metering(sumY, sumMask, outSumMask);
        };

        if (!state) {
            return accumulateYUncached(nullptr);
        }

        const std::uint64_t requestedMaskBytes = mask_bytes_for_dimensions(width, height);
        const bool cacheEligible = auto_exposure_mask_cache_eligible(requestedMaskBytes);

        if (!cacheEligible) {
            bool emitBypassTrace = false;
            std::uint64_t previousCachedBytes = 0;
            {
                std::lock_guard<std::mutex> lock(state->autoExposureMutex);
                previousCachedBytes = state->autoExposureMaskCachedBytes;
                const bool wasBypass = state->autoExposureMaskPolicyBypass;
                const std::uint64_t previousRequestedBytes = state->autoExposureMaskLastRequestedBytes;

                if (previousCachedBytes > 0) {
                    update_auto_exposure_mask_resident_bytes(previousCachedBytes, 0);
                }
                state->autoExposureMaskWeights.reset();
                state->autoExposureMaskCachedBytes = 0;
                state->autoExposureMaskValid = false;
                state->autoExposureMaskWidth = width;
                state->autoExposureMaskHeight = height;
                state->autoExposureMaskSigma = sigma;
                state->autoExposureMaskRenderScaleX = renderScaleX;
                state->autoExposureMaskRenderScaleY = renderScaleY;
                state->autoExposureMaskClipToken = clipToken;
                state->autoExposureMaskSum = 0.0;
                state->autoExposureMaskPolicyBypass = true;
                state->autoExposureMaskLastRequestedBytes = requestedMaskBytes;

                emitBypassTrace = (previousCachedBytes > 0)
                    || !wasBypass
                    || (previousRequestedBytes != requestedMaskBytes);
            }

            if (emitBypassTrace) {
                trace_auto_exposure_mask_cache_event(
                    state,
                    "cache_bypass",
                    width,
                    height,
                    requestedMaskBytes,
                    0,
                    previousCachedBytes,
                    (requestedMaskBytes == 0) ? "overflow_or_invalid" : "over_cap");
            }

            double effectiveSumMask = 0.0;
            const double measuredY = accumulateYUncached(&effectiveSumMask);
            {
                std::lock_guard<std::mutex> lock(state->autoExposureMutex);
                if (state->autoExposureMaskPolicyBypass &&
                    state->autoExposureMaskLastRequestedBytes == requestedMaskBytes) {
                    state->autoExposureMaskSum = effectiveSumMask;
                }
            }
            return measuredY;
        }

        const size_t expectedMaskSize = static_cast<size_t>(width) * static_cast<size_t>(height);
        auto needsMaskRebuild = [&](const InstanceState& s) -> bool {
            const std::shared_ptr<const std::vector<double>>& weights = s.autoExposureMaskWeights;
            return !s.autoExposureMaskValid
                || s.autoExposureMaskWidth != width
                || s.autoExposureMaskHeight != height
                || !nearly_equal_double(s.autoExposureMaskSigma, sigma)
                || !nearly_equal_double(s.autoExposureMaskRenderScaleX, renderScaleX)
                || !nearly_equal_double(s.autoExposureMaskRenderScaleY, renderScaleY)
                || s.autoExposureMaskClipToken != clipToken
                || !weights
                || weights->size() != expectedMaskSize;
        };

        std::shared_ptr<const std::vector<double>> maskSnapshot;
        bool maskValid = false;
        bool rebuildMask = false;
        {
            std::lock_guard<std::mutex> lock(state->autoExposureMutex);
            rebuildMask = needsMaskRebuild(*state);
            if (!rebuildMask) {
                maskSnapshot = state->autoExposureMaskWeights;
                maskValid = state->autoExposureMaskValid && static_cast<bool>(maskSnapshot);
                if (!maskValid) {
                    state->autoExposureMaskSum = 0.0;
                }
            }
        }

        if (rebuildMask) {
            auto rebuiltMask = std::make_shared<std::vector<double>>();
            const double sumMask = build_center_weight_mask(width, height, sigma, *rebuiltMask);
            const bool rebuiltValid = sumMask > 0.0;
            bool emitStoreTrace = false;
            std::uint64_t previousCachedBytes = 0;
            {
                std::lock_guard<std::mutex> lock(state->autoExposureMutex);
                if (needsMaskRebuild(*state)) {
                    previousCachedBytes = state->autoExposureMaskCachedBytes;
                    update_auto_exposure_mask_resident_bytes(previousCachedBytes, requestedMaskBytes);
                    state->autoExposureMaskWidth = width;
                    state->autoExposureMaskHeight = height;
                    state->autoExposureMaskSigma = sigma;
                    state->autoExposureMaskRenderScaleX = renderScaleX;
                    state->autoExposureMaskRenderScaleY = renderScaleY;
                    state->autoExposureMaskClipToken = clipToken;
                    state->autoExposureMaskWeights = rebuiltMask;
                    state->autoExposureMaskCachedBytes = requestedMaskBytes;
                    state->autoExposureMaskValid = rebuiltValid;
                    state->autoExposureMaskSum = rebuiltValid ? sumMask : 0.0;
                    emitStoreTrace = (previousCachedBytes != requestedMaskBytes) || state->autoExposureMaskPolicyBypass;
                    state->autoExposureMaskPolicyBypass = false;
                    state->autoExposureMaskLastRequestedBytes = requestedMaskBytes;
                }
                maskSnapshot = state->autoExposureMaskWeights;
                maskValid = state->autoExposureMaskValid && static_cast<bool>(maskSnapshot);
                if (!maskValid) {
                    state->autoExposureMaskSum = 0.0;
                }
            }

            if (emitStoreTrace) {
                trace_auto_exposure_mask_cache_event(
                    state,
                    "cache_store",
                    width,
                    height,
                    requestedMaskBytes,
                    requestedMaskBytes,
                    previousCachedBytes,
                    rebuiltValid ? "rebuilt" : "rebuilt_invalid");
            }
        }

        if (!maskValid || !maskSnapshot || maskSnapshot->size() != expectedMaskSize) {
            return 0.0;
        }

        double effectiveSumMask = 0.0;
        const double measuredY = accumulateYFromMask(*maskSnapshot, &effectiveSumMask);
        {
            std::lock_guard<std::mutex> lock(state->autoExposureMutex);
            if (state->autoExposureMaskWeights == maskSnapshot) {
                state->autoExposureMaskSum = effectiveSumMask;
            }
        }
        return measuredY;
    }

    static double measure_median_Y_DWG(
        OFX::Image* img,
        const OfxRectI& bounds,
        Spectral::InputColorSpace inputColorSpace,
        const Spectral::Mat3& rgbToXYZ,
        bool applyCctfDecoding) {

        if (!img) {
            return 0.0;
        }

        const int width = bounds.x2 - bounds.x1;
        const int height = bounds.y2 - bounds.y1;
        if (width <= 0 || height <= 0) {
            return 0.0;
        }
        const OfxRectI srcBounds = img->getBounds();

        const size_t total = static_cast<size_t>(width) * static_cast<size_t>(height);
        const int nComponents = pixel_component_count(img->getPixelComponents());
        if (nComponents <= 0) {
            return 0.0;
        }
        const std::size_t pixelStride = static_cast<std::size_t>(nComponents);
        const bool singleComponent = (nComponents == 1);
        std::vector<float>* valuesPtr = nullptr;
        std::vector<float> localValues;
        if (total <= kAutoExposureMedianScratchMaxSamples) {
            std::vector<float>& scratch = auto_exposure_median_scratch();
            scratch.clear();
            scratch.reserve(total);
            valuesPtr = &scratch;
        }
        else {
            localValues.reserve(total);
            valuesPtr = &localValues;
        }
        std::vector<float>& values = *valuesPtr;
        for (int yy = bounds.y1; yy < bounds.y2; ++yy) {
            const float* rowPix = row_start_if_covered(img, srcBounds, bounds, yy);
            if (rowPix) {
                const float* rowPixIt = rowPix;
                for (int xOff = 0; xOff < width; ++xOff) {
                    append_nonnegative_luma_if_finite(
                        rowPixIt,
                        singleComponent,
                        inputColorSpace,
                        applyCctfDecoding,
                        rgbToXYZ,
                        values);
                    rowPixIt += pixelStride;
                }
                continue;
            }
            for (int xOff = 0; xOff < width; ++xOff) {
                const int x = bounds.x1 + xOff;
                append_nonnegative_image_pixel_luma_if_present(
                    img,
                    x,
                    yy,
                    singleComponent,
                    inputColorSpace,
                    applyCctfDecoding,
                    rgbToXYZ,
                    values);
            }
        }

        if (values.empty()) {
            return 0.0;
        }

        const size_t n = values.size();
        const size_t mid = n / 2;
        auto midIt = values.begin() + static_cast<std::ptrdiff_t>(mid);
        std::nth_element(values.begin(), midIt, values.end());
        const float high = *midIt;
        if ((n & 1U) != 0U) {
            return static_cast<double>(high);
        }

        const float low = *std::max_element(values.begin(), midIt);
        return static_cast<double>((low + high) * 0.5f);
    }

    void init_spectral_globals_once() {
        Spectral::SpectralMutationScope mutationScope(
            Spectral::SpectralMutationStage::Bootstrap,
            "init_spectral_globals_once");
        (void)mutationScope;

        try {
            Spectral::lock_shape_to_reference_axis();
            const auto cmf = Spectral::load_csv_triplets(data_dir_string("cie1931_2deg.csv"));
            if (!Spectral::cmf_triplets_match_reference_axis(cmf)) {
                JTRACE("INIT", "FATAL: CMF wavelengths do not match 380-780@5nm grid");
                throw std::runtime_error("CMF grid mismatch");
            }
            Spectral::set_cie_1931_2deg_cmf(cmf.xbar, cmf.ybar, cmf.zbar);
            Spectral::ensure_precomputed_up_to_date();
            Spectral::disable_hanatos_if_reference_mismatch();
        }
        catch (...) {
            // Leave globals as-is; instance guards will pass-through if shape is invalid.
        }

        // Hanatos LUT: load once and set availability flag atomically.
        try {
            const std::string lutPath = data_dir_string("luts", "spectral_upsampling", "irradiance_xy_tc.npy");
            Spectral::load_hanatos_spectra_lut(lutPath);
        }
        catch (...) {
            Spectral::set_hanatos_available(false);
        }

        // Mallett 2019 basis: load once for sRGB basis reconstruction.
        try {
            const std::string basisPath = data_dir_string("luts", "spectral_upsampling", "mallett2019_basis.npy");
            Spectral::load_mallett2019_basis_npy(basisPath);
        }
        catch (...) {
            Spectral::set_mallett_available(false);
        }

        // KG3 filter fallback: set once if not present; safe idempotently.
        std::vector<std::pair<float, float>> kg3_pairs_raw;
        try { kg3_pairs_raw = Spectral::load_csv_pairs(data_dir_string("filters", "heat_absorbing", "schott", "KG3.csv")); }
        catch (...) { kg3_pairs_raw.clear(); }
        if (kg3_pairs_raw.empty()) {
            kg3_pairs_raw = {
                { Spectral::gShape.lambdaMin, 1.0f },
                { Spectral::gShape.lambdaMax, 1.0f }
            };
        }
        Spectral::set_filter_KG3_from_pairs(kg3_pairs_raw);
    }


    static std::vector<std::string> enlarger_illuminant_keys_for_choice(int choice) {
        switch (choice) {
        case 0: return { "D65", "d65" };
        case 1: return { "D55", "d55" };
        case 2: return { "D50", "d50" };
        case 3: return { "TH-KG3-L", "th-kg3-l" };
        case 4: return { "T", "t", "Incandescent", "incandescent" };
        case 5: return { "K75P", "k75p", "Kinoton75P", "Kinoton 75P", "kinoton75p", "kinoton_75p" };
        case 6: return { "EqualEnergy", "equal_energy", "Equal energy" };
        default: break;
        }
        return {};
    }

    static int illuminant_choice_index_from_string(const std::string& value) {
        const std::string normalized = IlluminantKeys::normalize(value);
        if (normalized.empty()) {
            return -1;
        }
        constexpr int kIlluminantChoiceCount = 7;
        for (int choice = 0; choice < kIlluminantChoiceCount; ++choice) {
            const auto keys = enlarger_illuminant_keys_for_choice(choice);
            const std::string* keyData = keys.data();
            const size_t keyCount = keys.size();
            for (size_t i = 0; i < keyCount; ++i, ++keyData) {
                if (IlluminantKeys::normalize(*keyData) == normalized) {
                    return choice;
                }
            }
        }
        return -1;
    }

    class ScopedParamEventSuppression {
    public:
        explicit ScopedParamEventSuppression(InstanceState* state)
            : _state(state) {
            if (_state) {
                _previous = _state->suppressParamEvents;
                _state->suppressParamEvents = true;
            }
        }

        ~ScopedParamEventSuppression() {
            if (_state) {
                _state->suppressParamEvents = _previous;
            }
        }

    private:
        InstanceState* _state = nullptr;
        bool _previous = false;
    };

} // namespace

JuicerEffect::ExposureParams JuicerEffect::gatherExposureParams() const {
    ExposureParams params{};
    double exposureSliderEV = read_double_param_or(_pExposure, 0.0);
    exposureSliderEV = sanitize_finite_or(exposureSliderEV, 0.0);
    params.sliderEV = exposureSliderEV;
    params.sliderScale = finite_exp2_scale(exposureSliderEV);
    const bool cameraAuto = read_bool_param_or(_pCameraAutoExposure, true);
    params.cameraAutoEnabled = cameraAuto;
    const int meteringMethod = read_choice_param_or(_pCameraMeteringMethod, 0);
    params.meteringMethod = meteringMethod;
    return params;
}

Scanner::Options JuicerEffect::gatherScannerOptions() const {
    Scanner::Options opts{};
    double blurSigma = read_double_param_or(_pScannerLensBlur, static_cast<double>(opts.lensBlurSigmaPx));
    blurSigma = sanitize_finite_or(blurSigma, 0.55);
    blurSigma = std::clamp(blurSigma, 0.0, 10.0);
    opts.lensBlurSigmaPx = static_cast<float>(blurSigma);

    std::array<double, 2> unsharp = read_double2_param_or(
        _pScannerUnsharp,
        { { static_cast<double>(opts.unsharpSigmaPx), static_cast<double>(opts.unsharpAmount) } });
    double unsharpSigma = unsharp[0];
    double unsharpAmount = unsharp[1];
    unsharpSigma = sanitize_finite_or(unsharpSigma, 0.7);
    unsharpAmount = sanitize_finite_or(unsharpAmount, 1.0);
    unsharpSigma = std::clamp(unsharpSigma, 0.0, 5.0);
    unsharpAmount = std::clamp(unsharpAmount, 0.0, 3.0);
    opts.unsharpSigmaPx = static_cast<float>(unsharpSigma);
    opts.unsharpAmount = static_cast<float>(unsharpAmount);
    return opts;
}

Scanner::Settings JuicerEffect::gatherScannerSettings() const {
    Scanner::Settings settings{};
    const bool useLut = read_bool_param_or(_pScannerUseLut, settings.useLut);
    settings.useLut = useLut;
    int lutRes = read_int_param_or(_pScannerLutResolution, static_cast<int>(settings.lutResolution));
    if (lutRes < 17 || lutRes > 128) {
        lutRes = 17;
    }
    settings.lutResolution = static_cast<std::uint32_t>(lutRes);
    return settings;
}

Print::Params JuicerEffect::gatherPrintParams() const {
    Print::Params params{};
    const bool bypass = read_bool_param_or(_pPrintBypass, true);
    const double pexp = read_double_param_or(_pPrintExposure, 1.0);
    const double preflash = read_double_param_or(_pPrintPreflash, 0.0);
    const double y = read_double_param_or(_pEnlargerY, 0.0);
    const double m = read_double_param_or(_pEnlargerM, 0.0);
    const double c = read_double_param_or(_pEnlargerC, 0.0);
    auto clampShift = [](double v) -> double {
        if (!is_finite(v)) return 0.0;
        const double limit = static_cast<double>(Print::kEnlargerSteps);
        return std::clamp(v, -limit, limit);
        };
    params.bypass = bypass;
    params.exposure = static_cast<float>(pexp);
    params.preflashExposure = static_cast<float>(preflash);
    params.yFilter = static_cast<float>(clampShift(y));
    params.mFilter = static_cast<float>(clampShift(m));
    params.cFilter = static_cast<float>(clampShift(c));
    return params;
}

Profiles::HalationMetadata JuicerEffect::gatherHalationUi() const {
    Profiles::HalationMetadata halation{};

    halation.active = read_bool_param_or(_pHalationActive, false);

    const std::array<double, 3> strengthPercent =
        read_sanitized_double3(_pHalationStrength, { {3.0, 0.30, 0.10} }, 0.0, 100.0);
    const std::array<double, 3> sizeUm =
        read_sanitized_double3(_pHalationSizeUm, { {200.0, 200.0, 200.0} }, 0.0, 1000.0);
    const std::array<double, 3> scatterStrengthPercent =
        read_sanitized_double3(_pHalationScatteringStrength, { {1.0, 2.0, 4.0} }, 0.0, 100.0);
    const std::array<double, 3> scatterSizeUm =
        read_sanitized_double3(_pHalationScatteringSizeUm, { {30.0, 20.0, 15.0} }, 0.0, 1000.0);

    cast_array(halation.sizeUm, sizeUm);
    cast_array(halation.scatteringSizeUm, scatterSizeUm);
    const float scale = 0.01f;
    float* strengthIt = halation.strength.data();
    float* scatterStrengthIt = halation.scatteringStrength.data();
    const double* strengthSrc = strengthPercent.data();
    const double* scatterStrengthSrc = scatterStrengthPercent.data();
    for (int i = 0; i < 3; ++i, ++strengthIt, ++scatterStrengthIt, ++strengthSrc, ++scatterStrengthSrc) {
        *strengthIt = static_cast<float>(*strengthSrc) * scale;
        *scatterStrengthIt = static_cast<float>(*scatterStrengthSrc) * scale;
    }

    return halation;
}

void JuicerEffect::applyHalationProfileDefaults() {
    if (!_state) {
        return;
    }
    if (!_state->baseLoaded) {
        return;
    }

    const Profiles::HalationMetadata& halationCfg = _state->base.halation;

    const std::array<double, 3> currentStrength =
        read_double3_param_or(_pHalationStrength, { { 0.0, 0.0, 0.0 } });
    const std::array<double, 3> currentSize =
        read_double3_param_or(_pHalationSizeUm, { { 0.0, 0.0, 0.0 } });
    const std::array<double, 3> currentScatterStrength =
        read_double3_param_or(_pHalationScatteringStrength, { { 0.0, 0.0, 0.0 } });
    const std::array<double, 3> currentScatterSize =
        read_double3_param_or(_pHalationScatteringSizeUm, { { 0.0, 0.0, 0.0 } });

    const std::array<double, 3> strengthPct = sanitize_scaled_float_array_to_double(
        halationCfg.strength,
        currentStrength,
        100.0,
        0.0,
        100.0);
    const std::array<double, 3> sizeUm = sanitize_scaled_float_array_to_double(
        halationCfg.sizeUm,
        currentSize,
        1.0,
        0.0,
        1000.0);
    const std::array<double, 3> scatterStrengthPct = sanitize_scaled_float_array_to_double(
        halationCfg.scatteringStrength,
        currentScatterStrength,
        100.0,
        0.0,
        100.0);
    const std::array<double, 3> scatterSizeUm = sanitize_scaled_float_array_to_double(
        halationCfg.scatteringSizeUm,
        currentScatterSize,
        1.0,
        0.0,
        1000.0);

    const double strengthMaster = mean_array(strengthPct);
    const double sizeMaster = mean_array(sizeUm);
    const double scatterStrengthMaster = mean_array(scatterStrengthPct);
    const double scatterSizeMaster = mean_array(scatterSizeUm);

    const ScopedParamEventSuppression suppressEvents(_state.get());

    set_double3_param_if(_pHalationStrength, strengthPct[0], strengthPct[1], strengthPct[2]);
    set_double3_param_if(_pHalationSizeUm, sizeUm[0], sizeUm[1], sizeUm[2]);
    set_double3_param_if(
        _pHalationScatteringStrength,
        scatterStrengthPct[0], scatterStrengthPct[1], scatterStrengthPct[2]);
    set_double3_param_if(_pHalationScatteringSizeUm, scatterSizeUm[0], scatterSizeUm[1], scatterSizeUm[2]);
    set_double_param_if(_pHalationStrengthMaster, strengthMaster);
    set_double_param_if(_pHalationSizeUmMaster, sizeMaster);
    set_double_param_if(_pHalationScatteringStrengthMaster, scatterStrengthMaster);
    set_double_param_if(_pHalationScatteringSizeUmMaster, scatterSizeMaster);

    _halationStrengthMasterLast = strengthMaster;
    _halationSizeUmMasterLast = sizeMaster;
    _halationScatteringStrengthMasterLast = scatterStrengthMaster;
    _halationScatteringSizeUmMasterLast = scatterSizeMaster;
}

namespace {
    struct GrainPresetDefaults {
        double amountEV = -1.20;
        double sizePx = 0.50;
        double sharpness = 0.5;
        double chroma = 0.3;
        double texture = 0.55;
        double particleAreaUm2 = 0.25;
        double sizeMixScale = 19.0;
        double densityMinMaster = 0.08;
        double uniformityMaster = 0.97;
        double sizeMixWeight = std::numeric_limits<double>::quiet_NaN();
        double microCell = std::numeric_limits<double>::quiet_NaN();
        double microSigma = std::numeric_limits<double>::quiet_NaN();
        double particleScaleMaster = 1.48;
        double particleScaleLayersMaster = 1.922;
        bool sublayersActive = true;
    };

    static GrainPresetDefaults grain_preset_defaults(int presetIndex) {
        GrainPresetDefaults d;
        switch (presetIndex) {
        case 0: // Fine
            d.amountEV = -1.396;
            d.sizePx = 0.615;
            d.sharpness = 0.50;
            d.chroma = 0.00;
            d.texture = 0.63;
            d.particleAreaUm2 = 0.25;
            d.sizeMixScale = 31.3;
            d.densityMinMaster = 0.08;
            d.uniformityMaster = 0.97;
            d.sizeMixWeight = 0.119;
            d.microCell = 60.0;
            d.microSigma = 181.2;
            d.particleScaleMaster = 1.48;
            d.particleScaleLayersMaster = 1.99;
            d.sublayersActive = true;
            break;
        case 2: // Coarse
            d.amountEV = -0.57;
            d.sizePx = 0.56;
            d.sharpness = 0.50;
            d.chroma = 0.50;
            d.texture = 0.35;
            d.particleAreaUm2 = 0.33;
            d.sizeMixScale = 16.0;
            d.densityMinMaster = 0.09;
            d.uniformityMaster = 0.96;
            d.sublayersActive = true;
            break;
        case 1: // Medium
        default:
            break;
        }
        return d;
    }

    inline double grain_lerp(double a, double b, double t) {
        return a + (b - a) * t;
    }

    static std::array<double, 3> default_particle_scale_ratio() {
        return { {0.8, 1.0, 2.0} };
    }

    static std::array<double, 3> default_particle_scale_layers_ratio() {
        return { {2.5, 1.0, 0.5} };
    }

    static std::array<double, 3> default_density_min_ratio() {
        return { {0.07, 0.08, 0.12} };
    }

    static std::array<double, 3> default_uniformity_ratio() {
        return { {0.97, 0.97, 0.99} };
    }

    static void normalize_ratio(std::array<double, 3>& values) {
        double* data = values.data();
        double sum = 0.0;
        const double* const dataEnd = data + values.size();
        for (; data < dataEnd; ++data) {
            const double v = *data;
            if (!is_positive_finite(v)) {
                values = { {1.0, 1.0, 1.0} };
                return;
            }
            sum += v;
        }
        const double mean = sum / 3.0;
        if (!is_positive_finite(mean)) {
            values = { {1.0, 1.0, 1.0} };
            return;
        }
        double* outData = values.data();
        const double* const outEnd = outData + values.size();
        for (; outData < outEnd; ++outData) {
            *outData /= mean;
        }
    }

    struct GrainRatioSet {
        std::array<double, 3> scale{};
        std::array<double, 3> scaleLayers{};
        std::array<double, 3> densityMin{};
        std::array<double, 3> uniformity{};
    };

    struct GrainAdvancedDefaults {
        double blurDyeClouds = 0.0;
        double sizeMixWeight = 0.0;
        double microCell = 0.0;
        double microSigma = 0.0;
    };

    static GrainAdvancedDefaults compute_grain_advanced_defaults(
        const GrainPresetDefaults& preset,
        double sharpness,
        double texture) {
        GrainAdvancedDefaults defaults{};
        defaults.blurDyeClouds = grain_lerp(1.40, 0.60, sharpness);
        defaults.sizeMixWeight = sanitize_finite_or(
            preset.sizeMixWeight,
            grain_lerp(0.072, 0.38, texture));
        defaults.microCell = sanitize_finite_or(
            preset.microCell,
            grain_lerp(50.0, 70.0, texture));
        defaults.microSigma = sanitize_finite_or(
            preset.microSigma,
            grain_lerp(140.0, 200.0, texture));
        return defaults;
    }

    static GrainRatioSet normalized_default_grain_ratios() {
        GrainRatioSet ratios{
            default_particle_scale_ratio(),
            default_particle_scale_layers_ratio(),
            default_density_min_ratio(),
            default_uniformity_ratio()
        };
        normalize_ratio(ratios.scale);
        normalize_ratio(ratios.scaleLayers);
        normalize_ratio(ratios.densityMin);
        normalize_ratio(ratios.uniformity);
        return ratios;
    }

    inline void set_scaled_triplet(
        OFX::Double3DParam* param,
        double master,
        const std::array<double, 3>& ratio) {
        if (!param) {
            return;
        }
        param->setValue(
            master * ratio[0],
            master * ratio[1],
            master * ratio[2]);
    }

    inline void set_scaled_triplet_clamped(
        OFX::Double3DParam* param,
        double master,
        const std::array<double, 3>& ratio,
        double lo,
        double hi) {
        if (!param) {
            return;
        }
        param->setValue(
            std::clamp(master * ratio[0], lo, hi),
            std::clamp(master * ratio[1], lo, hi),
            std::clamp(master * ratio[2], lo, hi));
    }

}

Profiles::GrainMetadata JuicerEffect::gatherGrainUi() const {
    Profiles::GrainMetadata grain{};

    grain.active = read_bool_param_or(_pGrainActive, false);

    const int presetIndex = read_choice_param_clamped(_pGrainPreset, 1, 0, 2);
    const GrainPresetDefaults preset = grain_preset_defaults(presetIndex);

    grain.sublayersActive = read_bool_param_or(_pGrainSublayersActive, preset.sublayersActive);

    const double amountEV = read_sanitized_double(_pGrainAmplitude, preset.amountEV, -3.0, 3.0);
    const double amplitude = std::exp2(amountEV);
    grain.amplitude = static_cast<float>(amplitude);

    grain.blur = read_sanitized_float(_pGrainBlur, static_cast<float>(preset.sizePx), 0.20, 2.00);

    const double sharpness = read_sanitized_double(_pGrainSharpness, preset.sharpness, 0.0, 1.0);

    const double chroma = read_sanitized_double(_pGrainChroma, preset.chroma, 0.0, 1.0);

    const double texture = read_sanitized_double(_pGrainTexture, preset.texture, 0.0, 1.0);

    const GrainAdvancedDefaults advancedDefaults = compute_grain_advanced_defaults(
        preset,
        sharpness,
        texture);

    grain.agxParticleAreaUm2 = read_sanitized_float(
        _pGrainParticleAreaUm2,
        static_cast<float>(preset.particleAreaUm2),
        0.0,
        10.0);

    grain.sizeMixScale = read_sanitized_float(
        _pGrainSizeMixScale,
        static_cast<float>(preset.sizeMixScale),
        1.0,
        50.0);

    grain.sizeMixWeight = read_sanitized_float(
        _pGrainSizeMixWeight,
        static_cast<float>(advancedDefaults.sizeMixWeight),
        0.0,
        1.0);

    grain.sizeMixWeightMid = read_sanitized_float(_pGrainSizeMixWeightMid, 0.0f, 0.0, 1.0);

    grain.blurDyeCloudsUm = read_sanitized_float(
        _pGrainBlurDyeCloudsUm,
        static_cast<float>(advancedDefaults.blurDyeClouds),
        0.0,
        10.0);

    grain.chroma = static_cast<float>(chroma);
    grain.chromaSharedWeight = static_cast<float>(std::sqrt(std::max(0.0, 1.0 - chroma)));
    grain.chromaIndWeight = static_cast<float>(std::sqrt(std::max(0.0, chroma)));
    const std::array<double, 3> defaultParticleScale = filled_double_array<3>(preset.particleScaleMaster);
    const std::array<double, 3> defaultParticleScaleLayers = filled_double_array<3>(preset.particleScaleLayersMaster);
    const std::array<double, 3> defaultDensityMin = filled_double_array<3>(preset.densityMinMaster);
    const std::array<double, 3> defaultUniformity = filled_double_array<3>(preset.uniformityMaster);
    const std::array<double, 3> particleScale =
        read_sanitized_double3(_pGrainParticleScale, defaultParticleScale, 0.0, 10.0);
    const std::array<double, 3> particleScaleLayers =
        read_sanitized_double3(_pGrainParticleScaleLayers, defaultParticleScaleLayers, 0.0, 10.0);
    const std::array<double, 3> densityMin =
        read_sanitized_double3(_pGrainDensityMin, defaultDensityMin, 0.0, 1.0);
    const std::array<double, 3> uniformity =
        read_sanitized_double3(_pGrainUniformity, defaultUniformity, 0.0, 1.0);
    cast_array(grain.agxParticleScale, particleScale);
    cast_array(grain.agxParticleScaleLayers, particleScaleLayers);
    cast_array(grain.densityMin, densityMin);
    cast_array(grain.uniformity, uniformity);

    grain.clumpTemporalMix = read_sanitized_float(_pGrainClumpTemporalMix, 0.30f, 0.0, 0.30);

    grain.clumpMorphPeriodSec = read_sanitized_float(_pGrainClumpMorphPeriodSec, 8.0f, 5.0, 60.0);

    std::array<double, 2> microStructure = { {
        advancedDefaults.microCell,
        advancedDefaults.microSigma
    } };
    microStructure = read_sanitized_double2(_pGrainMicroStructure, microStructure, 0.0, 1000.0);
    cast_array(grain.microStructure, microStructure);

    grain.breathingDebug = read_bool_param_or(_pGrainBreathingDebug, false);

    grain.debugView = read_choice_param_clamped(_pGrainDebugView, 0, 0, 6);

    grain.filmDustAmount = read_sanitized_float(_pFilmDustAmount, 0.0f, 0.0, 10.0);

    grain.gateDustAmount = read_sanitized_float(_pGateDustAmount, 0.0f, 0.0, 10.0);

    grain.filmScratchAmount = read_sanitized_float(_pFilmScratchAmount, 0.0f, 0.0, 10.0);

    grain.gateScratchAmount = read_sanitized_float(_pGateScratchAmount, 0.0f, 0.0, 10.0);

    grain.nSubLayers = 1;
    return grain;
}

void JuicerEffect::applyGrainPresetDefaults(int presetIndex) {
    const GrainPresetDefaults preset = grain_preset_defaults(presetIndex);
    const ScopedParamEventSuppression suppressEvents(_state.get());

    const GrainAdvancedDefaults advancedDefaults = compute_grain_advanced_defaults(
        preset,
        preset.sharpness,
        preset.texture);

    const GrainRatioSet ratios = normalized_default_grain_ratios();

    set_double_param_if(_pGrainAmplitude, preset.amountEV);
    set_double_param_if(_pGrainBlur, preset.sizePx);
    set_double_param_if(_pGrainSharpness, preset.sharpness);
    set_double_param_if(_pGrainChroma, preset.chroma);
    set_double_param_if(_pGrainTexture, preset.texture);
    set_bool_param_if(_pGrainSublayersActive, preset.sublayersActive);

    set_double_param_if(_pGrainParticleAreaUm2, preset.particleAreaUm2);
    if (_pGrainParticleScaleMaster) {
        set_double_param_if(_pGrainParticleScaleMaster, preset.particleScaleMaster);
        _grainParticleScaleMasterLast = preset.particleScaleMaster;
    }
    set_scaled_triplet_clamped(_pGrainParticleScale, preset.particleScaleMaster, ratios.scale, 0.0, 10.0);
    if (_pGrainParticleScaleLayersMaster) {
        set_double_param_if(_pGrainParticleScaleLayersMaster, preset.particleScaleLayersMaster);
        _grainParticleScaleLayersMasterLast = preset.particleScaleLayersMaster;
    }
    set_scaled_triplet_clamped(_pGrainParticleScaleLayers, preset.particleScaleLayersMaster, ratios.scaleLayers, 0.0, 10.0);
    if (_pGrainDensityMinMaster) {
        set_double_param_if(_pGrainDensityMinMaster, preset.densityMinMaster);
        _grainDensityMinMasterLast = preset.densityMinMaster;
    }
    set_scaled_triplet_clamped(_pGrainDensityMin, preset.densityMinMaster, ratios.densityMin, 0.0, 1.0);
    if (_pGrainUniformityMaster) {
        set_double_param_if(_pGrainUniformityMaster, preset.uniformityMaster);
        _grainUniformityMasterLast = preset.uniformityMaster;
    }
    set_scaled_triplet_clamped(_pGrainUniformity, preset.uniformityMaster, ratios.uniformity, 0.0, 1.0);
    set_double_param_if(_pGrainBlurDyeCloudsUm, std::clamp(advancedDefaults.blurDyeClouds, 0.0, 10.0));
    set_double_param_if(_pGrainSizeMixWeight, std::clamp(advancedDefaults.sizeMixWeight, 0.0, 1.0));
    set_double_param_if(_pGrainSizeMixWeightMid, 0.0);
    set_double_param_if(_pGrainSizeMixScale, std::clamp(preset.sizeMixScale, 1.0, 50.0));
    set_double2_param_if(
        _pGrainMicroStructure,
        std::clamp(advancedDefaults.microCell, 0.0, 1000.0),
        std::clamp(advancedDefaults.microSigma, 0.0, 1000.0));
    set_double_param_if(_pGrainClumpTemporalMix, 0.30);
    set_double_param_if(_pGrainClumpMorphPeriodSec, 8.0);

    updateGrainPresetLabel(false);
    updateGrainChromaEnabled();
}

void JuicerEffect::resetGrainAdvancedControls() {
    const int presetIndex = read_choice_param_clamped(_pGrainPreset, 1, 0, 2);
    const GrainPresetDefaults preset = grain_preset_defaults(presetIndex);

    const double sharpness = read_sanitized_double(_pGrainSharpness, preset.sharpness, 0.0, 1.0);

    const double texture = read_sanitized_double(_pGrainTexture, preset.texture, 0.0, 1.0);

    const GrainAdvancedDefaults advancedDefaults = compute_grain_advanced_defaults(
        preset,
        sharpness,
        texture);

    const double particleArea = preset.particleAreaUm2;
    const double sizeMixScale = preset.sizeMixScale;
    const double densityMinMaster = preset.densityMinMaster;
    const double uniformityMaster = preset.uniformityMaster;
    const double particleScaleMaster = preset.particleScaleMaster;
    const double particleScaleLayersMaster = preset.particleScaleLayersMaster;
    const double clumpTemporalMix = 0.30;
    const double clumpMorphPeriodSec = 8.0;

    const GrainRatioSet ratios = normalized_default_grain_ratios();
    const ScopedParamEventSuppression suppressEvents(_state.get());

    set_double_param_if(_pGrainParticleAreaUm2, particleArea);
    if (_pGrainParticleScaleMaster) {
        set_double_param_if(_pGrainParticleScaleMaster, particleScaleMaster);
        _grainParticleScaleMasterLast = particleScaleMaster;
    }
    if (_pGrainParticleScale) {
        set_scaled_triplet(_pGrainParticleScale, particleScaleMaster, ratios.scale);
    }
    if (_pGrainParticleScaleLayersMaster) {
        set_double_param_if(_pGrainParticleScaleLayersMaster, particleScaleLayersMaster);
        _grainParticleScaleLayersMasterLast = particleScaleLayersMaster;
    }
    if (_pGrainParticleScaleLayers) {
        set_scaled_triplet(_pGrainParticleScaleLayers, particleScaleLayersMaster, ratios.scaleLayers);
    }
    if (_pGrainDensityMinMaster) {
        set_double_param_if(_pGrainDensityMinMaster, densityMinMaster);
        _grainDensityMinMasterLast = densityMinMaster;
    }
    if (_pGrainDensityMin) {
        set_scaled_triplet(_pGrainDensityMin, densityMinMaster, ratios.densityMin);
    }
    if (_pGrainUniformityMaster) {
        set_double_param_if(_pGrainUniformityMaster, uniformityMaster);
        _grainUniformityMasterLast = uniformityMaster;
    }
    if (_pGrainUniformity) {
        set_scaled_triplet(_pGrainUniformity, uniformityMaster, ratios.uniformity);
    }
    set_double_param_if(_pGrainBlurDyeCloudsUm, advancedDefaults.blurDyeClouds);
    set_double_param_if(_pGrainSizeMixWeight, advancedDefaults.sizeMixWeight);
    set_double_param_if(_pGrainSizeMixWeightMid, 0.0);
    set_double_param_if(_pGrainSizeMixScale, sizeMixScale);
    set_double2_param_if(_pGrainMicroStructure, advancedDefaults.microCell, advancedDefaults.microSigma);
    set_double_param_if(_pGrainClumpTemporalMix, clumpTemporalMix);
    set_double_param_if(_pGrainClumpMorphPeriodSec, clumpMorphPeriodSec);

    updateGrainChromaEnabled();
}

void JuicerEffect::updateGrainPresetLabel(bool custom) {
    _grainPresetCustom = custom;
    const std::string label = custom
        ? (_grainPresetLabel + " (Custom)")
        : _grainPresetLabel;
    set_choice_label_if(_pGrainPreset, label);
}

void JuicerEffect::updateGrainChromaEnabled() {
    const bool perChannelDirty = false;
    set_double_param_enabled_if(_pGrainChroma, !perChannelDirty);
    if (perChannelDirty) {
        set_double_param_hint_if(_pGrainChroma, "Chroma disabled when per-channel overrides are active.");
    }
    else {
        set_double_param_hint_if(_pGrainChroma, _grainChromaHint);
    }
}

Profiles::ProfileGlare JuicerEffect::gatherGlareUi() const {
    Profiles::ProfileGlare glare{};

    glare.active = read_bool_param_or(_pGlareActive, true);

    glare.percent = read_sanitized_float(_pGlarePercent, 0.10f, 0.0, 1.0);

    glare.roughness = read_sanitized_float(_pGlareRoughness, 0.4f, 0.0, 1.0);

    glare.blur = read_sanitized_float(_pGlareBlurSigmaPx, 0.5f, 0.0, 10.0);

    glare.compensationRemovalFactor = read_sanitized_float(_pGlareCompRemovalFactor, 0.0f, 0.0, 1.0);

    glare.compensationRemovalDensity = read_sanitized_float(_pGlareCompRemovalDensity, 1.2f, 0.0, 3.0);

    glare.compensationRemovalTransition = read_sanitized_float(_pGlareCompRemovalTransition, 0.3f, 0.0, 2.0);

    return glare;
}

OutputEncoding::Params JuicerEffect::gatherOutputEncodingParams() const {
    OutputEncoding::Params params{};
    const int csIndex = read_choice_param_or(
        _pOutputColorSpace,
        OutputEncoding::toIndex(OutputEncoding::ColorSpace::sRGB));
    const bool applyCctf = read_bool_param_or(_pOutputCctfEncoding, true);
    const bool preserveLinear = read_bool_param_or(_pOutputLinearPassThrough, false);
    params.colorSpace = OutputEncoding::colorSpaceFromIndex(csIndex);
    params.applyCctfEncoding = applyCctf;
    params.preserveLinearRange = preserveLinear;
    return params;
}

JuicerEffect::AutoExposureResult JuicerEffect::computeAutoExposure(
    const OFX::RenderArguments& args,
    OFX::Image* srcImg,
    const OfxRectI& fullBounds,
    const ExposureParams& exposureParams) const {

    AutoExposureResult result{};
    result.exposureScale = 1.0f;
    result.autoEV = 0.0;

    if (!srcImg) {
        result.exposureScale = finite_exp2_scale(exposureParams.sliderEV);
        return result;
    }
    if (!exposureParams.cameraAutoEnabled) {
        result.exposureScale = finite_exp2_scale(exposureParams.sliderEV);
        return result;
    }

    InstanceState* state = _state.get();
    const bool isCudaRender = args.isEnabledCudaRender;

    auto rect_equal = [](const OfxRectI& a, const OfxRectI& b) {
        return a.x1 == b.x1 && a.y1 == b.y1 && a.x2 == b.x2 && a.y2 == b.y2;
        };

    OfxRectI meterBounds = fullBounds;
    if (_src) {
        try {
            const OfxRectD rod = _src->getRegionOfDefinition(args.time);
            const double rodWidth = rod.x2 - rod.x1;
            const double rodHeight = rod.y2 - rod.y1;
            const bool hasRodDimensions =
                sanitize_positive_finite_or(rodWidth, 0.0) > 0.0 &&
                sanitize_positive_finite_or(rodHeight, 0.0) > 0.0;
            if (hasRodDimensions) {
                meterBounds.x1 = static_cast<int>(std::floor(rod.x1));
                meterBounds.y1 = static_cast<int>(std::floor(rod.y1));
                meterBounds.x2 = static_cast<int>(std::ceil(rod.x2));
                meterBounds.y2 = static_cast<int>(std::ceil(rod.y2));
            }
        }
        catch (...) {
            // Ignore failures; fall back to full bounds.
        }
    }

    if (state) {
        std::lock_guard<std::mutex> cacheLock(state->autoExposureMutex);
        state->autoExposureCanonicalBounds = meterBounds;
        state->autoExposureCanonicalValid = true;
    }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    // CUDA path: metering + exposure scale are computed and applied entirely on the GPU to avoid
    // forcing a stream synchronization just to read back Y/EV on the CPU.
    if (isCudaRender) {
        result.autoEV = 0.0;
        result.exposureScale = 1.0f;
        return result;
    }
#endif

    const std::shared_ptr<const WorkingState> wsCur = load_active_working_state_if(state);
    const uint64_t wsBuildCounter = working_state_build_counter_or_zero(wsCur);

    // Camera auto-exposure always meters against AgX's fixed 18.4% target (independent of scanner target tweaks).
    constexpr double kCameraMeterTargetY = 0.184;
    constexpr double kInvLn2 = 1.44269504088896340736;

    const double sigma = 0.2;
    const double renderScaleX = sanitize_positive_finite_or(args.renderScale.x, 1.0);
    const double renderScaleY = sanitize_positive_finite_or(args.renderScale.y, 1.0);
    const std::uintptr_t clipToken = reinterpret_cast<std::uintptr_t>(_src);

    const int inputColorSpaceIndex = read_choice_param_or(
        _pInputColorSpace,
        Spectral::inputColorSpaceToIndex(Spectral::InputColorSpace::DaVinciWideGamut));
    const bool applyInputCctfDecoding = read_bool_param_or(_pInputCctfDecoding, false);

    double autoEV = 0.0;
    bool haveCachedAutoEV = false;
    const int meteringMethod = exposureParams.meteringMethod;
    if (state) {
        std::lock_guard<std::mutex> cacheLock(state->autoExposureMutex);
        if (state->autoExposureCacheValid &&
            state->autoExposureCacheIsCudaRender == isCudaRender &&
            state->autoExposureCacheAutoEnabled &&
            state->autoExposureCacheMeteringMethod == meteringMethod &&
            nearly_equal_double(state->autoExposureCacheTime, args.time) &&
            state->autoExposureCacheBuildCounter == wsBuildCounter &&
            rect_equal(state->autoExposureCacheBounds, meterBounds) &&
            nearly_equal_double(state->autoExposureCacheRenderScaleX, renderScaleX) &&
            nearly_equal_double(state->autoExposureCacheRenderScaleY, renderScaleY) &&
            state->autoExposureCacheClipToken == clipToken &&
            state->autoExposureCacheInputColorSpaceIndex == inputColorSpaceIndex &&
            state->autoExposureCacheApplyCctfDecoding == applyInputCctfDecoding) {
            autoEV = state->autoExposureCacheEV;
            haveCachedAutoEV = true;
        }
    }

    if (!haveCachedAutoEV) {
        bool measurementValid = false;
        double evComp = 0.0;
        double Yexp = 0.0;
        const Spectral::InputColorSpace inputColorSpace =
            Spectral::inputColorSpaceFromIndex(inputColorSpaceIndex);
        const Spectral::Mat3 inputRgbToXYZ = Spectral::matrix_input_rgb_to_xyz(inputColorSpace);
        if (meteringMethod == static_cast<int>(MeteringMethod::Median)) {
            Yexp = measure_median_Y_DWG(
                srcImg,
                meterBounds,
                inputColorSpace,
                inputRgbToXYZ,
                applyInputCctfDecoding);
        }
        else {
            Yexp = measure_center_weighted_Y_DWG_cached(
                srcImg,
                meterBounds,
                sigma,
                state,
                renderScaleX,
                renderScaleY,
                clipToken,
                inputColorSpace,
                inputRgbToXYZ,
                applyInputCctfDecoding);
        }
        const bool canComputeEv = (Yexp > 0.0 && kCameraMeterTargetY > 0.0);
        if (canComputeEv) {
            const double exposureRatio = Yexp / kCameraMeterTargetY;
            evComp = -std::log(exposureRatio) * kInvLn2;
        }
        if (!is_finite(evComp)) {
            evComp = 0.0;
            measurementValid = false;
        }
        else {
            measurementValid = canComputeEv;
        }
        autoEV = evComp;

        if (state) {
            std::lock_guard<std::mutex> cacheLock(state->autoExposureMutex);
            state->autoExposureCacheValid = measurementValid;
            state->autoExposureCacheIsCudaRender = isCudaRender;
            state->autoExposureCacheTime = args.time;
            state->autoExposureCacheAutoEnabled = true;
            state->autoExposureCacheMeteringMethod = meteringMethod;
            state->autoExposureCacheBuildCounter = wsBuildCounter;
            state->autoExposureCacheBounds = meterBounds;
            state->autoExposureCacheEV = autoEV;
            state->autoExposureCacheRenderScaleX = renderScaleX;
            state->autoExposureCacheRenderScaleY = renderScaleY;
            state->autoExposureCacheClipToken = clipToken;
            state->autoExposureCacheInputColorSpaceIndex = inputColorSpaceIndex;
            state->autoExposureCacheApplyCctfDecoding = applyInputCctfDecoding;
        }
    }

    const double sliderEV = exposureParams.sliderEV;
    const double totalEV = autoEV + sliderEV;
    result.autoEV = autoEV;
    result.exposureScale = finite_exp2_scale(totalEV);
    return result;
}

#ifdef JUICER_ENABLE_COUPLERS
Couplers::Runtime JuicerEffect::prepareCouplers(
    const OFX::RenderArguments& args,
    int fullWidth,
    int fullHeight,
    float pixelSizeUm) const {

    Couplers::Runtime dirRT{};
    if (!(_state && _state->baseLoaded)) {
        return dirRT;
    }

    const std::shared_ptr<const WorkingState> wsCur = load_active_working_state_if(_state.get());
    if (wsCur && wsCur->buildCounter > 0) {
        dirRT = wsCur->dirRT;
        float* dMaxIt = dirRT.dMax;
        for (int i = 0; i < 3; ++i, ++dMaxIt) {
            float v = static_cast<float>(sanitize_positive_finite_or(*dMaxIt, 1.0));
            if (v > 1000.0f) v = 1000.0f;
            *dMaxIt = v;
        }
        sanitize_dir_matrix(dirRT.M);
    }

    float sigmaPixels = 0.0f;
    const float sigmaMicrometers = dirRT.spatialSigmaMicrometers;
    const bool hasPixelSize = sanitize_positive_finite_or(pixelSizeUm, 0.0) > 0.0;
    if (sigmaMicrometers > 0.0f && hasPixelSize) {
        sigmaPixels = sigmaMicrometers / pixelSizeUm;
        sigmaPixels = sanitize_nonnegative_finite_or(sigmaPixels, 0.0f);
    }
    else {
        // Fallback to legacy geometry if pixelSizeUm was not available
        const double filmFormat = read_double_param_or(_pCameraFilmFormat, 35.0);
        const double filmLongEdgeMm = sanitize_positive_finite_or(filmFormat, 35.0);

        const double widthPx = static_cast<double>(fullWidth);
        const double heightPx = static_cast<double>(fullHeight);
        const double longEdgePx = std::max(widthPx, heightPx);

        if (sigmaMicrometers > 0.0f && longEdgePx > 0.0 && filmLongEdgeMm > 0.0) {
            sigmaPixels = Couplers::spatial_sigma_pixels_from_micrometers(
                sigmaMicrometers,
                filmLongEdgeMm,
                widthPx,
                heightPx);
            sigmaPixels = sanitize_nonnegative_finite_or(sigmaPixels, 0.0f);
        }
    }
    dirRT.spatialSigmaPixels = sigmaPixels;

    auto dir_has_effect = [](const Couplers::Runtime& rt) -> bool {
        if (!rt.active) {
            return false;
        }
        return has_nonzero_finite_dir_matrix(rt.M);
    };

    if (!dir_has_effect(dirRT)) {
        // Provably zero-effect DIR (e.g. amount==0). Treat as inactive so we can skip the
        // extra DIR sampling path and any spatial-DIR build work without changing results.
        dirRT.active = false;
    }

    return dirRT;
}
#endif

JuicerEffect::WorkingStateInfo JuicerEffect::prepareWorkingState() const {
    WorkingStateInfo info{};
    if (!(_state && _state->baseLoaded)) {
        return info;
    }

    info.workingState = load_active_working_state_if(_state.get());

    const WorkingState* ws = info.workingState.get();
    if (ws && ws->buildCounter > 0 && ws->printRT) {
        info.printRuntime = ws->printRT.get();
    }

    const bool baselineReady = (!ws || !ws->hasBaseline) ||
        (ws->baseMin.linear.size() == static_cast<size_t>(Spectral::gShape.K));

    const bool negativeScannerReady = ws && ws->negativeScannerValid &&
        ws->negativeMediumRuntime.staticKey.hash != 0 &&
        ws->negativeMediumRuntime.range.digest != 0 &&
        ws->negativeMediumRuntime.tables &&
        ws->negativeMediumRuntime.tables->K == Spectral::gShape.K;

    info.workingStateReady = (ws && ws->buildCounter > 0 &&
        ws->tablesView.K == Spectral::gShape.K &&
        ws->tablesView.epsY.size() == static_cast<size_t>(Spectral::gShape.K) &&
        ws->tablesView.epsM.size() == static_cast<size_t>(Spectral::gShape.K) &&
        ws->tablesView.epsC.size() == static_cast<size_t>(Spectral::gShape.K) &&
        baselineReady &&
        !ws->densB.lambda_nm.empty() && !ws->densB.linear.empty() &&
        !ws->densG.lambda_nm.empty() && !ws->densG.linear.empty() &&
        !ws->densR.lambda_nm.empty() && !ws->densR.linear.empty() &&
        negativeScannerReady);

    const Print::Runtime* prt = info.printRuntime;
    const bool printScannerReady = ws && ws->printScannerValid &&
        ws->printMediumRuntime.staticKey.hash != 0 &&
        ws->printMediumRuntime.range.digest != 0 &&
        ws->printMediumRuntime.tables &&
        ws->printMediumRuntime.tables->K == Spectral::gShape.K;

    info.printRuntimeReady =
        (prt != nullptr) &&
        Print::profile_is_valid(prt->profile) &&
        prt->illumView.linear.size() == static_cast<size_t>(Spectral::gShape.K) &&
        prt->illumEnlarger.linear.size() == static_cast<size_t>(Spectral::gShape.K) &&
        (ws && ws->tablesPrint.K == Spectral::gShape.K) &&
        info.workingStateReady &&
        printScannerReady;

    return info;
}

JuicerEffect::JuicerEffect(OfxImageEffectHandle handle)
    : OFX::ImageEffect(handle)
{
    // Cache clips (wrappers) for Step 2; safe even if render still uses legacy path.
    try {
        _src = fetchClip(kOfxImageEffectSimpleSourceClipName); // "Source"
        _dst = fetchClip(kOfxImageEffectOutputClipName);       // "Output"
    }
    catch (...) {
        _src = nullptr;
        _dst = nullptr;
    }

    // Cache parameter handles (wrappers)
    try {
        _pExposure = fetchDoubleParam(kParamExposure);
        _pCameraAutoExposure = fetchBooleanParam(kParamCameraAutoExposure);
        _pCameraFilmFormat = fetchDoubleParam(JuicerParams::kCameraFilmFormatMm);
        _pCameraMeteringMethod = fetchChoiceParam(JuicerParams::kCameraMeteringMethod);
        _pFilmStock = fetchChoiceParam(kParamFilmStock);
        _pSpectralMode = fetchChoiceParam(kParamSpectralMode);
        _pPrintPaper = fetchChoiceParam(kParamPrintPaper);
        _pRefIll = fetchChoiceParam("ReferenceIlluminant");
        _pEnlIll = fetchChoiceParam("EnlargerIlluminant");
        _pEnlDichroicSet = fetchChoiceParam(kParamEnlargerDichroicSet);
        _pInputColorSpace = fetchChoiceParam(JuicerParams::kInputColorSpace);
        _pInputCctfDecoding = fetchBooleanParam(JuicerParams::kInputCctfDecoding);
        _pOutputColorSpace = fetchChoiceParam(kParamOutputColorSpace);
        _pOutputCctfEncoding = fetchBooleanParam(kParamOutputCctfEncoding);
        _pOutputLinearPassThrough = fetchBooleanParam(kParamOutputLinearPassThrough);


#ifdef JUICER_ENABLE_COUPLERS
        _pCouplersActive = fetchBooleanParam(Couplers::kParamCouplersActive);
        _pCouplersAmount = fetchDoubleParam(Couplers::kParamCouplersAmount);
        _pCouplersAmountR = fetchDoubleParam(Couplers::kParamCouplersAmountR);
        _pCouplersAmountG = fetchDoubleParam(Couplers::kParamCouplersAmountG);
        _pCouplersAmountB = fetchDoubleParam(Couplers::kParamCouplersAmountB);
        _pCouplersSigma = fetchDoubleParam(Couplers::kParamCouplersLayerSigma);
        _pCouplersHigh = fetchDoubleParam(Couplers::kParamCouplersHighExpShift);
        _pCouplersSpatialSigma = fetchDoubleParam(Couplers::kParamCouplersSpatialSigma);
#endif

        _pScannerLensBlur = fetchDoubleParam(JuicerParams::kScannerLensBlurSigmaPx);
        _pScannerUnsharp = fetchDouble2DParam(JuicerParams::kScannerUnsharpMask);
        _pScannerUseLut = fetchBooleanParam(JuicerParams::kScannerUseLut);
        _pScannerLutResolution = fetchIntParam(JuicerParams::kScannerLutResolution);

        _pPrintBypass = fetchBooleanParam("PrintBypass");
        _pPrintExposure = fetchDoubleParam("PrintExposure");
        _pPrintPreflash = fetchDoubleParam("PrintPreflash");
        _pPrintExposureComp = fetchBooleanParam("PrintExposureCompensation");
        _pEnlargerY = fetchDoubleParam("EnlargerY");
        _pEnlargerM = fetchDoubleParam("EnlargerM");
        _pEnlargerC = fetchDoubleParam("EnlargerC");

        _pHalationActive = fetchBooleanParam(JuicerParams::kHalationActive);
        _pHalationStrengthMaster = fetchDoubleParam(JuicerParams::kHalationStrengthMaster);
        _pHalationSizeUmMaster = fetchDoubleParam(JuicerParams::kHalationSizeUmMaster);
        _pHalationScatteringStrengthMaster = fetchDoubleParam(JuicerParams::kHalationScatteringStrengthMaster);
        _pHalationScatteringSizeUmMaster = fetchDoubleParam(JuicerParams::kHalationScatteringSizeUmMaster);
        _pHalationRevertToStock = fetchPushButtonParam(JuicerParams::kHalationRevertToStock);
        _pHalationStrength = fetchDouble3DParam(JuicerParams::kHalationStrength);
        _pHalationSizeUm = fetchDouble3DParam(JuicerParams::kHalationSizeUm);
        _pHalationScatteringStrength = fetchDouble3DParam(JuicerParams::kHalationScatteringStrength);
        _pHalationScatteringSizeUm = fetchDouble3DParam(JuicerParams::kHalationScatteringSizeUm);

        _pGrainActive = fetchBooleanParam(JuicerParams::kGrainActive);
        _pGrainSublayersActive = fetchBooleanParam(JuicerParams::kGrainSublayersActive);
        _pGrainPreset = fetchChoiceParam(JuicerParams::kGrainPreset);
        _pGrainParticleAreaUm2 = fetchDoubleParam(JuicerParams::kGrainParticleAreaUm2);
        _pGrainAmplitude = fetchDoubleParam(JuicerParams::kGrainAmplitude);
        _pGrainSharpness = fetchDoubleParam(JuicerParams::kGrainSharpness);
        _pGrainChroma = fetchDoubleParam(JuicerParams::kGrainChroma);
        _pGrainTexture = fetchDoubleParam(JuicerParams::kGrainTexture);
        _pGrainParticleScaleMaster = fetchDoubleParam(JuicerParams::kGrainParticleScaleMaster);
        _pGrainParticleScaleLayersMaster = fetchDoubleParam(JuicerParams::kGrainParticleScaleLayersMaster);
        _pGrainDensityMinMaster = fetchDoubleParam(JuicerParams::kGrainDensityMinMaster);
        _pGrainUniformityMaster = fetchDoubleParam(JuicerParams::kGrainUniformityMaster);
        _pGrainParticleScale = fetchDouble3DParam(JuicerParams::kGrainParticleScale);
        _pGrainParticleScaleLayers = fetchDouble3DParam(JuicerParams::kGrainParticleScaleLayers);
        _pGrainDensityMin = fetchDouble3DParam(JuicerParams::kGrainDensityMin);
        _pGrainUniformity = fetchDouble3DParam(JuicerParams::kGrainUniformity);
        _pGrainBlur = fetchDoubleParam(JuicerParams::kGrainBlur);
        _pGrainBlurDyeCloudsUm = fetchDoubleParam(JuicerParams::kGrainBlurDyeCloudsUm);
        _pGrainSizeMixWeight = fetchDoubleParam(JuicerParams::kGrainSizeMixWeight);
        _pGrainSizeMixWeightMid = fetchDoubleParam(JuicerParams::kGrainSizeMixWeightMid);
        _pGrainSizeMixScale = fetchDoubleParam(JuicerParams::kGrainSizeMixScale);
        _pGrainClumpTemporalMix = fetchDoubleParam(JuicerParams::kGrainClumpTemporalMix);
        _pGrainClumpMorphPeriodSec = fetchDoubleParam(JuicerParams::kGrainClumpMorphPeriodSec);
        _pGrainBreathingDebug = fetchBooleanParam(JuicerParams::kGrainBreathingDebug);
        _pGrainDebugView = fetchChoiceParam(JuicerParams::kGrainDebugView);
        _pGrainMicroStructure = fetchDouble2DParam(JuicerParams::kGrainMicroStructure);
        _pGrainResetAdvanced = fetchPushButtonParam(JuicerParams::kGrainResetAdvanced);
        _pGateWeaveAmount = fetchDoubleParam(JuicerParams::kGateWeaveAmount);
        _pFilmDustAmount = fetchDoubleParam(JuicerParams::kFilmDustAmount);
        _pGateDustAmount = fetchDoubleParam(JuicerParams::kGateDustAmount);
        _pFilmScratchAmount = fetchDoubleParam(JuicerParams::kFilmScratchAmount);
        _pGateScratchAmount = fetchDoubleParam(JuicerParams::kGateScratchAmount);

        _pGlareActive = fetchBooleanParam(JuicerParams::kGlareActive);
        _pGlarePercent = fetchDoubleParam(JuicerParams::kGlarePercent);
        _pGlareRoughness = fetchDoubleParam(JuicerParams::kGlareRoughness);
        _pGlareBlurSigmaPx = fetchDoubleParam(JuicerParams::kGlareBlurSigmaPx);
        _pGlareCompRemovalFactor = fetchDoubleParam(JuicerParams::kGlareCompensationRemovalFactor);
        _pGlareCompRemovalDensity = fetchDoubleParam(JuicerParams::kGlareCompensationRemovalDensity);
        _pGlareCompRemovalTransition = fetchDoubleParam(JuicerParams::kGlareCompensationRemovalTransition);
        _pPrintDminFactor = fetchDoubleParam(JuicerParams::kPrintDminFactor);
    }
    catch (...) {
        // Safe: any missing param will remain nullptr and defaults are used in snapshot/usage paths.
    }

    if (_pGrainPreset) {
        std::string label;
        _pGrainPreset->getLabel(label);
        if (!label.empty()) {
            _grainPresetLabel = label;
        }
    }
    if (_pGrainChroma) {
        const std::string hint = _pGrainChroma->getHint();
        if (!hint.empty()) {
            _grainChromaHint = hint;
        }
    }

    auto initMasterCache = [](OFX::DoubleParam* param, double& outValue) {
        if (!param) {
            outValue = std::numeric_limits<double>::quiet_NaN();
            return;
        }
        double v = 0.0;
        param->getValue(v);
        outValue = v;
    };
    initMasterCache(_pHalationStrengthMaster, _halationStrengthMasterLast);
    initMasterCache(_pHalationSizeUmMaster, _halationSizeUmMasterLast);
    initMasterCache(_pHalationScatteringStrengthMaster, _halationScatteringStrengthMasterLast);
    initMasterCache(_pHalationScatteringSizeUmMaster, _halationScatteringSizeUmMasterLast);
    initMasterCache(_pGrainParticleScaleMaster, _grainParticleScaleMasterLast);
    initMasterCache(_pGrainParticleScaleLayersMaster, _grainParticleScaleLayersMasterLast);
    initMasterCache(_pGrainDensityMinMaster, _grainDensityMinMasterLast);
    initMasterCache(_pGrainUniformityMaster, _grainUniformityMasterLast);

    // Own per-instance state
    _state = std::make_unique<InstanceState>();
    _state->dataDir = ensure_trailing_separator(data_dir_string());
    JuicerAtomic::store_shared_ptr(&_state->activeWorkingState, std::shared_ptr<const WorkingState>{});
    _state->activeBuildCounter = 0;
    {
        const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        const std::uint64_t seedFields[2] = {
            static_cast<std::uint64_t>(now),
            static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(this))
        };
        std::uint64_t seed = Hash::hash_bytes(seedFields, sizeof(seedFields));
        if (seed == 0) {
            seed = 1;
        }
        _state->sessionSeed = seed;
        _state->instanceToken = seed;
    }

    // Defer heavy bootstrap until first param change
}

JuicerEffect::~JuicerEffect() {
    std::uint64_t releasedMaskBytes = 0;
    if (_state) {
        std::lock_guard<std::mutex> lock(_state->autoExposureMutex);
        releasedMaskBytes = _state->autoExposureMaskCachedBytes;
        _state->autoExposureMaskWeights.reset();
        _state->autoExposureMaskCachedBytes = 0;
        _state->autoExposureMaskValid = false;
        _state->autoExposureMaskSum = 0.0;
    }
    if (releasedMaskBytes > 0) {
        update_auto_exposure_mask_resident_bytes(releasedMaskBytes, 0);
        trace_auto_exposure_mask_cache_event(
            _state.get(),
            "cache_release",
            0,
            0,
            0,
            0,
            releasedMaskBytes,
            "instance_destroy");
    }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    if (_state) {
        const bool traceInfo = JTRACE_ENABLED(1);
        std::vector<JuicerCuda::ResourceManager::DeviceContextKey> keys;
        {
            std::lock_guard<std::mutex> lock(_state->cudaMutex);
            keys.reserve(_state->cudaByDevice.size());
            for (const auto& entry : _state->cudaByDevice) {
                keys.emplace_back(entry.first);
            }
        }
        const JuicerCuda::ResourceManager::DeviceContextKey* keyData = keys.data();
        const size_t keyCount = keys.size();
        for (size_t i = 0; i < keyCount; ++i, ++keyData) {
            const auto& key = *keyData;
            std::string retireError;
            const bool retireOk = JuicerCuda::ResourceManager::command_retire_context_idle(key, retireError);
            if (!retireOk || !retireError.empty()) {
                if (traceInfo) {
                    const std::uintptr_t contextBits = reinterpret_cast<std::uintptr_t>(key.contextOpaque);
                    std::string msg;
                    msg.reserve(192);
                    msg = "teardown_retire_idle_failed device_id=";
                    msg += std::to_string(key.deviceId);
                    msg += " context=";
                    msg += std::to_string(contextBits);
                    msg += " accepted=";
                    msg += std::to_string(bool_to_i32(retireOk));
                    if (!retireError.empty()) {
                        msg += " error=";
                        msg += retireError;
                    }
                    JTRACE("MSLCY", msg);
                }
            }
        }
    }
#endif
    _state.reset();
}

void JuicerEffect::render(const OFX::RenderArguments& args) {
    // Fetch images via wrappers
    std::unique_ptr<OFX::Image> srcImg(_src ? _src->fetchImage(args.time) : nullptr);
    std::unique_ptr<OFX::Image> dstImg(_dst ? _dst->fetchImage(args.time) : nullptr);
    if (!srcImg || !dstImg) return;

#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
    // CUDA-only mode: reject CPU/OpenCL/Metal renders. During development this stays disabled so
    // we can fall back to the CPU pipeline while CUDA parity is still in progress.
    if (!args.isEnabledCudaRender) {
        JTRACE("CUDA", "JUICER_CUDA_ONLY: rejecting non-CUDA render request");
        throw OFX::Exception::Suite(kOfxStatErrUnsupported);
    }
#endif

    // Components and depth
    const OFX::PixelComponentEnum comps = srcImg->getPixelComponents();
    const OFX::BitDepthEnum depth = srcImg->getPixelDepth();

    const int nComponents = pixel_component_count(comps);
    const bool traceVerbose = JTRACE_ENABLED(3);

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    if (args.isEnabledCudaRender) {
        // CUDA renders use device pointers; avoid CPU pixel reads (auto-exposure, non-float copies, etc.).
        if (requires_nonfloat_copy(depth, nComponents)) {
            throw OFX::Exception::Suite(kOfxStatErrUnsupported);
        }
    }
#else
    if (args.isEnabledCudaRender) {
        throw OFX::Exception::Suite(kOfxStatErrUnsupported);
    }
#endif

    if (requires_nonfloat_copy(depth, nComponents)) {
        JuicerProc::copyNonFloatRect(srcImg.get(), dstImg.get());
        return;
    }

    const OfxRectI fullBounds = srcImg->getBounds();
    if (_state) {
        std::lock_guard<std::mutex> lock(_state->m);
        const OfxRectI prev = _state->cachedFrameBounds;
        const bool changed = prev.x1 != fullBounds.x1 || prev.y1 != fullBounds.y1 ||
            prev.x2 != fullBounds.x2 || prev.y2 != fullBounds.y2;
        if (changed) {
            _state->cachedFrameBounds = fullBounds;
            std::uint32_t next = _state->frameBoundsVersion.load(std::memory_order_relaxed);
            next = (next == std::numeric_limits<std::uint32_t>::max()) ? next : (next + 1U);
            if (next == 0) {
                next = 1;
            }
            _state->frameBoundsVersion.store(next, std::memory_order_release);
        }
    }

    // ROI: args.renderWindow if provided; otherwise use image bounds
    OfxRectI roi = args.renderWindow;
    if (roi.x1 == roi.x2 && roi.y1 == roi.y2) {
        roi = fullBounds;
    }
    const int width = roi.x2 - roi.x1;
    const int height = roi.y2 - roi.y1;
    if (width <= 0 || height <= 0) return;
    const int fullWidth = fullBounds.x2 - fullBounds.x1;
    const int fullHeight = fullBounds.y2 - fullBounds.y1;
    const bool fullFrame = (roi.x1 == fullBounds.x1 && roi.y1 == fullBounds.y1 &&
        roi.x2 == fullBounds.x2 && roi.y2 == fullBounds.y2);
    if (!fullFrame) {
        JTRACE("RENDER", "FATAL: render window must match full frame; tiles/ROIs are unsupported");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    const double filmFormat = read_double_param_or(_pCameraFilmFormat, 35.0);
    const double filmFormatMm = sanitize_positive_finite_or(filmFormat, 35.0);
    const double longEdgePx = static_cast<double>(std::max(fullWidth, fullHeight));
    float pixelSizeUm = 0.0f;
    if (filmFormatMm > 0.0 && longEdgePx > 0.0) {
        pixelSizeUm = static_cast<float>((filmFormatMm * 1000.0) / longEdgePx);
    }

    // Ensure bootstrap has run before we rely on parameter state
    if (_state && !_state->baseLoaded) {
        if (!_state->inBootstrap) {
            const ScopedParamEventSuppression guard(_state.get());
            bootstrap_after_attach();
        }
    }

    // Coalesce parameter-driven WorkingState rebuilds on the render thread to keep UI callbacks fast.
    if (_state && _state->baseLoaded) {
        rebuild_pending_state_if_needed(*this, *_state);
    }
    const ExposureParams exposureParams = gatherExposureParams();
    const Scanner::Options scannerOptions = gatherScannerOptions();
    const Scanner::Settings scannerSettings = gatherScannerSettings();
    Print::Params printParams = gatherPrintParams();
    const Profiles::HalationMetadata halationUi = gatherHalationUi();
    const Profiles::GrainMetadata grainUi = gatherGrainUi();
    const Profiles::ProfileGlare glareUi = gatherGlareUi();
    const double gateWeaveAmount = read_double_param_or(_pGateWeaveAmount, 1.0);
    OutputEncoding::Params outputEncodingParams = gatherOutputEncodingParams();

    const AutoExposureResult autoExposure = computeAutoExposure(
        args,
        srcImg.get(),
        fullBounds,
        exposureParams);

#ifdef JUICER_ENABLE_COUPLERS
    Couplers::Runtime dirRT = prepareCouplers(args, fullWidth, fullHeight, pixelSizeUm);
#else
    Couplers::Runtime dirRT{};
#endif

    WorkingStateInfo wsInfo = prepareWorkingState();
    std::shared_ptr<const WorkingState> wsHold = wsInfo.workingState;
    const WorkingState* ws = wsHold.get();
    const Print::Runtime* prt = wsInfo.printRuntime;
    const bool wsReady = wsInfo.workingStateReady;
    const bool printReady = wsInfo.printRuntimeReady;
    if (traceVerbose) {
        ParamSnapshot Pdbg = snapshotParams();
        const ProfileKeyLabels labels = resolve_profile_key_labels(Pdbg);
        const std::uintptr_t prtPtr = reinterpret_cast<std::uintptr_t>(prt);
        const std::uint64_t buildCounter = working_state_build_counter_or_zero(ws);
        const float neutralY = print_runtime_value_or_zero(prt, &Print::Runtime::neutralY);
        const float neutralM = print_runtime_value_or_zero(prt, &Print::Runtime::neutralM);
        const float neutralC = print_runtime_value_or_zero(prt, &Print::Runtime::neutralC);
        std::string msg;
        msg.reserve(256);
        msg = "render print state build=";
        msg += std::to_string(buildCounter);
        msg += " paper=";
        msg += labels.paperLabel;
        msg += " film=";
        msg += labels.filmLabel;
        msg += " printRT=";
        msg += std::to_string(prtPtr);
        msg += " neutralY/M/C=";
        append_ymc_triplet(msg, neutralY, neutralM, neutralC);
        msg += " yFilter=";
        msg += std::to_string(printParams.yFilter);
        msg += " mFilter=";
        msg += std::to_string(printParams.mFilter);
        msg += " cFilter=";
        msg += std::to_string(printParams.cFilter);
        msg += " bypass=";
        msg += std::to_string(bool_to_i32(printParams.bypass));
        JTRACE_VERBOSE("PRINTDBG", msg);
    }
    if (!wsReady) {
        JTRACE("BUILD", "FATAL: working state not ready; aborting render");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    if (!printParams.bypass && !printReady) {
        JTRACE("PRINT", "FATAL: print runtime not ready while print path requested");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    // --- Print exposure compensation via spectral mid-gray probe (agx parity) ---
    {
        const bool printComp = read_bool_param_or(_pPrintExposureComp, false);

        printParams.exposureCompensationEnabled = printComp;
        printParams.exposureCompensationScale = scale_if_enabled_or_one(printComp, exposureParams.sliderScale);
    }

    // Tile-based multithreaded processing via OFX::ImageProcessor
    JuicerProcessor proc(*this);
    proc.setSrcDst(srcImg.get(), dstImg.get());
    proc.setComponents(nComponents);
    proc.setScannerOptions(scannerOptions);
    proc.setScannerSettings(scannerSettings);
    proc.setPrintParams(printParams);
    proc.setHalationOverride(halationUi);
    proc.setGrainOverride(grainUi);
    proc.setGateWeaveAmount(gateWeaveAmount);
    proc.setPrintGlareOverride(glareUi);
    proc.setDirRuntime(dirRT);
    proc.setWorkingState(ws, wsReady);
    proc.setPrintRuntime(prt, printReady);
    proc.setInstanceState(_state.get());
    const std::uint32_t frameVersion = frame_bounds_version_or_zero(_state.get());
    proc.setFrameBoundsVersion(frameVersion);
    proc.setPixelSizeUm(pixelSizeUm);
    // Per agx-emulsion parity: autoExposure.exposureScale already encodes 2^(autoEV + sliderEV).
    float filmExposureScale = static_cast<float>(sanitize_positive_finite_or(autoExposure.exposureScale, 1.0));
    proc.setExposure(filmExposureScale);
    proc.setCameraAutoExposure(exposureParams.cameraAutoEnabled, exposureParams.meteringMethod, exposureParams.sliderEV);
    proc.setOutputEncoding(outputEncodingParams);
    const std::uintptr_t renderClipToken = reinterpret_cast<std::uintptr_t>(_src);
    proc.setClipToken(renderClipToken);
    proc.setFrameRate(getFrameRate());
    proc.setFrameTime(args.time);
    proc.setRenderWindowRect(roi);
    proc.setGPURenderArgs(args);
    proc.setRenderHints(args.interactiveRenderStatus, args.renderQualityDraft, args.sequentialRenderStatus);

    // Dispatch to support library's threaded/tiled CPU path
    proc.process();
}

void JuicerEffect::changedParam(const OFX::InstanceChangedArgs& args, const std::string& paramName) {
    const bool traceInfo = JTRACE_ENABLED(1);

    // Suppress recursion while we are programmatically setting params
    if (param_events_suppressed(_state.get())) {
        trace_changed_param_gate(traceInfo, paramName, "changedParam suppressed for '");
        return;
    }
    if (bootstrap_in_progress(_state.get())) {
        trace_changed_param_gate(traceInfo, paramName, "changedParam ignored during bootstrap for '");
        return;
    }
    if (_state && auto_exposure_cache_param_changed(paramName)) {
        std::lock_guard<std::mutex> cacheLock(_state->autoExposureMutex);
        _state->autoExposureCacheValid = false;
    }
    if (param_name_is(paramName, JuicerParams::kHalationRevertToStock)) {
        applyHalationProfileDefaults();
        onParamsPossiblyChanged(paramName.c_str());
        return;
    }

    const bool userEdit = (args.reason == OFX::eChangeUserEdit);
    if (user_edit_param_is(userEdit, paramName, JuicerParams::kGrainPreset)) {
        const int presetIndex = read_choice_param_clamped(_pGrainPreset, 1, 0, 2);
        applyGrainPresetDefaults(presetIndex);
        onParamsPossiblyChanged(paramName.c_str());
        return;
    }
    if (user_edit_param_is(userEdit, paramName, JuicerParams::kGrainResetAdvanced)) {
        resetGrainAdvancedControls();
        onParamsPossiblyChanged(paramName.c_str());
        return;
    }

    auto has_master_triplet_params = [this](OFX::DoubleParam* masterParam, OFX::Double3DParam* advParam) -> bool {
        return masterParam && advParam && _state;
    };

    auto read_master_value = [&](OFX::DoubleParam* masterParam, double lo, double hi, double& master) -> bool {
        if (!masterParam) {
            return false;
        }
        master = read_double_param_or(masterParam, master);
        if (!is_finite(master)) {
            return false;
        }
        master = std::clamp(master, lo, hi);
        return true;
    };

    auto apply_master_delta = [&](OFX::DoubleParam* masterParam,
        OFX::Double3DParam* advParam,
        double& masterCache,
        double lo,
        double hi) {
        if (!has_master_triplet_params(masterParam, advParam)) {
            return;
        }
        double master = 0.0;
        if (!read_master_value(masterParam, lo, hi, master)) {
            return;
        }
        double prev = masterCache;
        if (!is_finite(prev)) {
            prev = master;
        }
        const double delta = master - prev;
        std::array<double, 3> values{ {0.0, 0.0, 0.0} };
        advParam->getValue(values[0], values[1], values[2]);
        for (double& value : values) {
            value = sanitize_finite_clamped(value, master, lo, hi);
        }
        if (delta != 0.0) {
            for (double& value : values) {
                value = std::clamp(value + delta, lo, hi);
            }
            const ScopedParamEventSuppression suppressEvents(_state.get());
            set_double3_param_if(advParam, values[0], values[1], values[2]);
        }
        masterCache = master;
    };

    auto apply_ratio_master = [&](OFX::DoubleParam* masterParam,
        OFX::Double3DParam* advParam,
        const std::array<double, 3>& fallbackRatio,
        double& masterCache,
        double lo,
        double hi) {
        if (!has_master_triplet_params(masterParam, advParam)) {
            return;
        }
        double master = 0.0;
        if (!read_master_value(masterParam, lo, hi, master)) {
            return;
        }

        std::array<double, 3> values{ {0.0, 0.0, 0.0} };
        advParam->getValue(values[0], values[1], values[2]);
        for (double& value : values) {
            value = sanitize_finite_clamped(value, master, lo, hi);
        }

        std::array<double, 3> ratio = fallbackRatio;
        const double mean = (values[0] + values[1] + values[2]) / 3.0;
        if (is_positive_finite(mean)) {
            double* ratioIt = ratio.data();
            const double* valueIt = values.data();
            for (int i = 0; i < 3; ++i, ++ratioIt, ++valueIt) {
                *ratioIt = *valueIt / mean;
            }
        }

        double* valueIt = values.data();
        const double* ratioIt = ratio.data();
        for (int i = 0; i < 3; ++i, ++valueIt, ++ratioIt) {
            *valueIt = std::clamp(master * *ratioIt, lo, hi);
        }
        const ScopedParamEventSuppression suppressEvents(_state.get());
        set_double3_param_if(advParam, values[0], values[1], values[2]);
        masterCache = master;
    };

    if (userEdit && is_grain_preset_input_param(paramName)) {
        updateGrainPresetLabel(true);
    }

    updateGrainChromaEnabled();

    switch (halation_master_selector(paramName)) {
    case HalationMasterSelector::Strength:
        apply_master_delta(_pHalationStrengthMaster, _pHalationStrength, _halationStrengthMasterLast, 0.0, 100.0);
        break;
    case HalationMasterSelector::SizeUm:
        apply_master_delta(_pHalationSizeUmMaster, _pHalationSizeUm, _halationSizeUmMasterLast, 0.0, 1000.0);
        break;
    case HalationMasterSelector::ScatteringStrength:
        apply_master_delta(_pHalationScatteringStrengthMaster, _pHalationScatteringStrength, _halationScatteringStrengthMasterLast, 0.0, 100.0);
        break;
    case HalationMasterSelector::ScatteringSizeUm:
        apply_master_delta(_pHalationScatteringSizeUmMaster, _pHalationScatteringSizeUm, _halationScatteringSizeUmMasterLast, 0.0, 1000.0);
        break;
    case HalationMasterSelector::None:
        if (userEdit) {
            const GrainRatioMasterSelector grainMaster = grain_ratio_master_selector(paramName);
            if (grainMaster != GrainRatioMasterSelector::None) {
                const GrainRatioSet ratios = normalized_default_grain_ratios();

                switch (grainMaster) {
                case GrainRatioMasterSelector::Scale:
                    apply_ratio_master(_pGrainParticleScaleMaster, _pGrainParticleScale, ratios.scale, _grainParticleScaleMasterLast, 0.0, 10.0);
                    break;
                case GrainRatioMasterSelector::ScaleLayers:
                    apply_ratio_master(_pGrainParticleScaleLayersMaster, _pGrainParticleScaleLayers, ratios.scaleLayers, _grainParticleScaleLayersMasterLast, 0.0, 10.0);
                    break;
                case GrainRatioMasterSelector::DensityMin:
                    apply_ratio_master(_pGrainDensityMinMaster, _pGrainDensityMin, ratios.densityMin, _grainDensityMinMasterLast, 0.0, 1.0);
                    break;
                case GrainRatioMasterSelector::Uniformity:
                    apply_ratio_master(_pGrainUniformityMaster, _pGrainUniformity, ratios.uniformity, _grainUniformityMasterLast, 0.0, 1.0);
                    break;
                case GrainRatioMasterSelector::None:
                    break;
                }
            }
        }
        break;
    }

    if (user_edit_param_is(userEdit, paramName, JuicerParams::kGrainSharpness) && _pGrainSharpness && _pGrainBlurDyeCloudsUm && _state) {
        double sharpness = read_double_param_or(_pGrainSharpness, 0.5);
        if (is_finite(sharpness)) {
            sharpness = sanitize_finite_clamped(sharpness, 0.5, 0.0, 1.0);
            const double blurDyeClouds = std::clamp(grain_lerp(1.40, 0.60, sharpness), 0.0, 10.0);
            const ScopedParamEventSuppression suppressEvents(_state.get());
            set_double_param_if(_pGrainBlurDyeCloudsUm, blurDyeClouds);
        }
    }
    if (user_edit_param_is(userEdit, paramName, JuicerParams::kGrainTexture) && _pGrainTexture && _pGrainSizeMixWeight && _pGrainMicroStructure && _state) {
        double texture = read_double_param_or(_pGrainTexture, 0.55);
        if (is_finite(texture)) {
            texture = sanitize_finite_clamped(texture, 0.55, 0.0, 1.0);
            const double sizeMixWeight = std::clamp(grain_lerp(0.072, 0.38, texture), 0.0, 1.0);
            const double microCell = std::clamp(grain_lerp(50.0, 70.0, texture), 0.0, 1000.0);
            const double microSigma = std::clamp(grain_lerp(140.0, 200.0, texture), 0.0, 1000.0);
            const ScopedParamEventSuppression suppressEvents(_state.get());
            set_double_param_if(_pGrainSizeMixWeight, sizeMixWeight);
            set_double2_param_if(_pGrainMicroStructure, microCell, microSigma);
        }
    }
    onParamsPossiblyChanged(paramName.c_str());
}

ParamSnapshot JuicerEffect::snapshotParams() const {
    ParamSnapshot P;
    auto read_bool_as_int = [](auto* param, bool fallback) -> int {
        return bool_to_i32(read_bool_param_or(param, fallback));
    };

    P.filmStockIndex = read_choice_param_or(_pFilmStock, P.filmStockIndex);
    P.printPaperIndex = read_choice_param_or(_pPrintPaper, P.printPaperIndex);
    P.spectralUpsamplingMode = read_choice_param_or(_pSpectralMode, P.spectralUpsamplingMode);
    P.refIll = read_choice_param_or(_pRefIll, P.refIll);
    P.enlIll = read_choice_param_or(_pEnlIll, P.enlIll);
    P.enlDichroicSet = read_choice_param_or(_pEnlDichroicSet, P.enlDichroicSet);

    P.glareCompRemovalFactor =
        read_sanitized_double(_pGlareCompRemovalFactor, P.glareCompRemovalFactor, 0.0, 1.0);
    P.glareCompRemovalDensity =
        read_sanitized_double(_pGlareCompRemovalDensity, P.glareCompRemovalDensity, 0.0, 3.0);
    P.glareCompRemovalTransition =
        read_sanitized_double(_pGlareCompRemovalTransition, P.glareCompRemovalTransition, 0.0, 2.0);
    P.printDminFactor =
        read_sanitized_double(_pPrintDminFactor, P.printDminFactor, 0.0, 1.0);
    P.inputColorSpace = read_choice_param_or(_pInputColorSpace, P.inputColorSpace);
    P.inputCctfDecoding = read_bool_as_int(_pInputCctfDecoding, false);
#ifdef JUICER_ENABLE_COUPLERS
    P.couplersActive = read_bool_as_int(_pCouplersActive, true);
    P.couplersAmount = read_double_param_or(_pCouplersAmount, P.couplersAmount);
    P.ratioR = read_double_param_or(_pCouplersAmountR, P.ratioR);
    P.ratioG = read_double_param_or(_pCouplersAmountG, P.ratioG);
    P.ratioB = read_double_param_or(_pCouplersAmountB, P.ratioB);
    P.sigma = read_double_param_or(_pCouplersSigma, P.sigma);
    P.high = read_double_param_or(_pCouplersHigh, P.high);
    P.spatialSigmaMicrometers = read_double_param_or(_pCouplersSpatialSigma, 0.0);
#endif
    P.scannerLensBlurSigmaPx = read_double_param_or(_pScannerLensBlur, P.scannerLensBlurSigmaPx);
    P.scannerUnsharpMask = read_double2_param_or(
        _pScannerUnsharp,
        P.scannerUnsharpMask);
    P.scannerUseLut = read_bool_as_int(_pScannerUseLut, true);
    P.scannerLutResolution = read_int_param_or(_pScannerLutResolution, P.scannerLutResolution);
    P.outputColorSpace = read_choice_param_or(_pOutputColorSpace, P.outputColorSpace);
    P.outputCctfEncoding = read_bool_as_int(_pOutputCctfEncoding, true);
    P.outputLinearPassThrough = read_bool_as_int(_pOutputLinearPassThrough, false);
    return P;
}

void JuicerEffect::bootstrap_after_attach() {
    // Initialize Spectral globals exactly once per process.
    std::call_once(gSpectralGlobalsOnce, init_spectral_globals_once);
    JTRACE("BUILD", "spectral globals ensured once; proceeding to profile and film stock load");
    // Suppress re-entrant param events during bootstrap
    _state->inBootstrap = true;
    _state->suppressParamEvents = true;

    _state->printRT = Print::Runtime{};
    ParamSnapshot P = snapshotParams();

    // Load selected print paper profile
    (void)load_print_profile_for_snapshot(P, _state->printRT, /*moveMidNeutralVectors*/false);

    // Load film stock before applying metadata-driven illuminant defaults
    _state->baseLoaded = load_film_stock_into_base(P.filmStockIndex, *_state);
    if (_state->baseLoaded) {
        applyHalationProfileDefaults();
    }

    // Apply metadata-driven illuminant defaults and rebuild runtime illuminants
    applyMetadataIlluminantDefaults(P);
    update_print_illuminant_runtime(P, _state->printRT, _state->dataDir);

    // Load dichroic filters (set selection controls which vendor curves are used).
    // Identity fallback is already handled in loader via 1.0 curves.
    (void)try_load_dichroic_filters(
        P.enlDichroicSet,
        _state->printRT,
        "dichroic load failed",
        "using identity filters");

    applyNeutralFilters(P);

    if (_state->baseLoaded) {
#ifdef JUICER_ENABLE_COUPLERS
        applyCouplerProfileDefaults(P);
#endif
        rebuild_working_state(this->getHandle(), *_state, P);
    }
    else {
        JTRACE("STOCK", "bootstrap: failed to load film stock; deferring rebuild");
    }

    // Re-enable changedParam handling now that bootstrap is complete
    _state->suppressParamEvents = false;
    _state->inBootstrap = false;
}

void JuicerEffect::applyNeutralFilters(const ParamSnapshot& P) {
    if (!_state) {
        return;
    }
    const bool traceInfo = JTRACE_ENABLED(1);

    const ProfileKeyLabels labels = resolve_profile_key_labels(P);
    const char* paperKey = labels.paperKey;
    const char* negativeKey = labels.filmKey;
    const std::vector<std::string> illumKeys = enlarger_illuminant_keys_for_choice(P.enlIll);

    if (!(paperKey && negativeKey && !illumKeys.empty())) {
        if (traceInfo) {
            JTRACE(
                "PRINT",
                "Neutral filter lookup prerequisites missing: "
                + neutral_filter_prereq_context(paperKey, negativeKey, join_keys_csv_or_none(illumKeys)));
        }
        throw std::runtime_error("Neutral filter metadata incomplete for current selection");
    }

    float neutralY = Print::kDefaultNeutralY;
    float neutralM = Print::kDefaultNeutralM;
    float neutralC = Print::kDefaultNeutralC;
    bool loaded = false;

    const std::string jsonPathPrimary = data_dir_string("profiles", enlarger_neutral_filters_json_for_choice(P.enlDichroicSet));
    const std::string jsonPathFallback = data_dir_string("profiles", "enlarger_neutral_ymc_filters.json");
    std::tuple<float, float, float> ymc{};
    std::string selectedDbVersionHash;
    const std::string* illumKeyData = illumKeys.data();
    const size_t illumKeyCount = illumKeys.size();
    for (size_t i = 0; i < illumKeyCount; ++i, ++illumKeyData) {
        const std::string& illumKey = *illumKeyData;
        if (illumKey.empty()) {
            continue;
        }
        if (load_enlarger_neutral_filters(jsonPathPrimary, paperKey, illumKey, negativeKey, ymc, NeutralFilterThreadClass::Control, &selectedDbVersionHash) ||
            (jsonPathPrimary != jsonPathFallback &&
                load_enlarger_neutral_filters(jsonPathFallback, paperKey, illumKey, negativeKey, ymc, NeutralFilterThreadClass::Control, &selectedDbVersionHash))) {
            neutralY = std::clamp(std::get<0>(ymc), 0.0f, 1.0f);
            neutralM = std::clamp(std::get<1>(ymc), 0.0f, 1.0f);
            neutralC = std::clamp(std::get<2>(ymc), 0.0f, 1.0f);
            loaded = true;
            if (traceInfo) {
                std::string msg;
                msg.reserve(192);
                msg = "Neutral filters loaded for ";
                msg += illumKey;
                msg += " Y/M/C=";
                append_ymc_triplet(msg, neutralY, neutralM, neutralC);
                msg += " db_version_hash=";
                msg += cstr_or_default_if_empty(selectedDbVersionHash, "none");
                JTRACE("PRINT", msg);
            }
            break;
        }
    }

    if (!loaded) {
        if (traceInfo) {
            JTRACE(
                "PRINT",
                "Neutral filters missing for "
                + neutral_filter_missing_context(paperKey, negativeKey, join_keys_csv_or_none(illumKeys))
                + "; aborting print path");
        }
        throw std::runtime_error("Neutral filter database entry not found");
    }

    std::uint64_t neutralFilterHash = Print::kDefaultNeutralFilterHash;
    if (!selectedDbVersionHash.empty()) {
        neutralFilterHash = Hash::hash_bytes(selectedDbVersionHash.data(), selectedDbVersionHash.size());
        if (neutralFilterHash == 0) {
            neutralFilterHash = Print::kDefaultNeutralFilterHash;
        }
    }

    _state->printRT.neutralY = neutralY;
    _state->printRT.neutralM = neutralM;
    _state->printRT.neutralC = neutralC;
    _state->printRT.neutralFilterHash = neutralFilterHash;
    // Preserve user-entered enlarger offsets and exposure toggle; neutral baselines update independently.

}

bool JuicerEffect::applyMetadataIlluminantDefaults(ParamSnapshot& P) {
    if (!_state) {
        return false;
    }

    bool changed = false;

    const std::string& filmRef = !_state->filmReferenceIlluminant.empty()
        ? _state->filmReferenceIlluminant
        : _state->base.referenceIlluminant;
    const std::string& printRef = _state->printRT.referenceIlluminant;
    const std::string& printView = _state->printRT.viewingIlluminant;

    const std::string& refSource = first_nonempty_or(filmRef, printRef, printView);
    const std::string& enlSource = first_nonempty_or(printRef, filmRef, printView);

    const ScopedParamEventSuppression suppressEvents(_state.get());
    changed = apply_illuminant_choice_from_source(
        _pRefIll,
        P.refIll,
        _state->illuminantOverride.reference,
        refSource) || changed;
    changed = apply_illuminant_choice_from_source(
        _pEnlIll,
        P.enlIll,
        _state->illuminantOverride.enlarger,
        enlSource) || changed;

    if (changed) {
        P = snapshotParams();
    }

    return changed;
}

#ifdef JUICER_ENABLE_COUPLERS
void JuicerEffect::applyCouplerProfileDefaults(ParamSnapshot& P) {
    if (!_state) {
        return;
    }

    const Profiles::DirCouplersProfile& dirCfg = _state->base.dirCouplers;
    if (!dirCfg.hasData) {
        return;
    }

    const ScopedParamEventSuppression suppressEvents(_state.get());

    auto apply_clean_double = [&](bool dirty,
                                  double source,
                                  double fallback,
                                  double lo,
                                  double hi,
                                  OFX::DoubleParam* param,
                                  double& target) {
        if (dirty) {
            return;
        }
        const double value = sanitize_finite_clamped(source, fallback, lo, hi);
        set_double_param_if(param, value);
        target = value;
    };

    if (!_state->couplerDirty.active.load(std::memory_order_acquire)) {
        const bool active = dirCfg.active;
        set_bool_param_if(_pCouplersActive, active);
        P.couplersActive = bool_to_i32(active);
    }

    apply_clean_double(
        _state->couplerDirty.amount.load(std::memory_order_acquire),
        static_cast<double>(dirCfg.amount),
        P.couplersAmount,
        0.0,
        2.0,
        _pCouplersAmount,
        P.couplersAmount);

    apply_clean_double(
        _state->couplerDirty.ratioB.load(std::memory_order_acquire),
        static_cast<double>(dirCfg.ratioRGB[0]),
        P.ratioB,
        0.0,
        1.0,
        _pCouplersAmountB,
        P.ratioB);

    apply_clean_double(
        _state->couplerDirty.ratioG.load(std::memory_order_acquire),
        static_cast<double>(dirCfg.ratioRGB[1]),
        P.ratioG,
        0.0,
        1.0,
        _pCouplersAmountG,
        P.ratioG);

    apply_clean_double(
        _state->couplerDirty.ratioR.load(std::memory_order_acquire),
        static_cast<double>(dirCfg.ratioRGB[2]),
        P.ratioR,
        0.0,
        1.0,
        _pCouplersAmountR,
        P.ratioR);

    apply_clean_double(
        _state->couplerDirty.sigma.load(std::memory_order_acquire),
        static_cast<double>(dirCfg.diffusionInterlayer),
        P.sigma,
        0.0,
        4.0,
        _pCouplersSigma,
        P.sigma);

    apply_clean_double(
        _state->couplerDirty.high.load(std::memory_order_acquire),
        static_cast<double>(dirCfg.highExposureShift),
        P.high,
        0.0,
        1.0,
        _pCouplersHigh,
        P.high);

    const float profileSpatialSigma = _state->couplerProfileSpatialSigmaValid
        ? static_cast<float>(_state->couplerProfileSpatialSigmaMicrometers)
        : dirCfg.diffusionSizeUm;
    apply_clean_double(
        _state->couplerDirty.spatialSigma.load(std::memory_order_acquire),
        static_cast<double>(profileSpatialSigma),
        P.spatialSigmaMicrometers,
        0.0,
        50.0,
        _pCouplersSpatialSigma,
        P.spatialSigmaMicrometers);

}
#endif

void JuicerEffect::onParamsPossiblyChanged(const char* changedNameOrNull) {
    if (!_state) return;
    const bool traceVerbose = JTRACE_ENABLED(3);
    // Suppress re-entrant param handling while programmatic changes are in flight
    if (_state->suppressParamEvents) {
        JTRACE("BUILD", "onParamsPossiblyChanged suppressed");
        return;
    }

    // If bootstrap hasn’t run yet, run it once now
    if (!_state->baseLoaded) {
        bootstrap_after_attach();
    }

    ParamSnapshot P = snapshotParams();
    const ChangedParamFlags changed = classify_changed_param(changedNameOrNull);
    if (traceVerbose) {
        const ProfileKeyLabels labels = resolve_profile_key_labels(P);
        const std::shared_ptr<const WorkingState> wsDbg = load_active_working_state_if(_state.get());
        const std::uint64_t activeBuild = working_state_build_counter_or_zero(wsDbg);
        const std::uint64_t lastHash = _state->lastHash.load(std::memory_order_acquire);
        std::string msg;
        msg.reserve(224);
        msg = "params change name=";
        msg += cstr_or_default_if_null(changedNameOrNull, "<null>");
        msg += " printIndex=";
        msg += std::to_string(P.printPaperIndex);
        msg += " printKey=";
        msg += labels.paperLabel;
        msg += " filmIndex=";
        msg += std::to_string(P.filmStockIndex);
        msg += " filmKey=";
        msg += labels.filmLabel;
        msg += " activeBuild=";
        msg += std::to_string(activeBuild);
        msg += " lastHash=";
        msg += std::to_string(lastHash);
        JTRACE_VERBOSE("PRINTDBG", msg);
    }
#ifdef JUICER_ENABLE_COUPLERS
    if (changed.hasName) {
        (void)mark_coupler_dirty_if_known(*_state, changedNameOrNull);
    }
#endif

    // Track user overrides for illuminant choices.
    mark_illuminant_override_if_changed(*_state, changed);

    auto trace_neutral_filters_applied = [&](const char* reloadSource) {
        if (!traceVerbose) {
            return;
        }
        const ProfileKeyLabels labels = resolve_profile_key_labels(P);
        std::string msg;
        msg.reserve(192);
        msg = "neutral filters applied (";
        msg += cstr_or_default_if_null(reloadSource, "unspecified");
        msg += ") paper=";
        msg += labels.paperLabel;
        msg += " film=";
        msg += labels.filmLabel;
        msg += " Y/M/C=";
        append_ymc_triplet(msg, _state->printRT.neutralY, _state->printRT.neutralM, _state->printRT.neutralC);
        JTRACE_VERBOSE("PRINTDBG", msg);
    };

    bool neutralApplied = false;
    auto apply_neutral_filters_with_trace = [&](const char* reloadSource) {
        applyNeutralFilters(P);
        neutralApplied = true;
        trace_neutral_filters_applied(reloadSource);
    };

    bool printReloaded = false;
    if (changed.printPaper) {
        const PrintProfileLoadInputs printLoad =
            load_print_profile_for_snapshot(P, _state->printRT, /*moveMidNeutralVectors*/true);
        const ProfileKeyLabels& labels = printLoad.labels;
        if (traceVerbose) {
            std::string msg;
            msg.reserve(256);
            msg = "print reload key=";
            msg += labels.paperLabel;
            msg += " dir=";
            msg += printLoad.printDir;
            msg += " json=";
            msg += printLoad.printProfileJson;
            msg += " ref=";
            msg += _state->printRT.referenceIlluminant;
            msg += " view=";
            msg += _state->printRT.viewingIlluminant;
            JTRACE_VERBOSE("PRINTDBG", msg);
        }

        // Reload dichroic filters (vendor selection controls which curves are used).
        (void)try_load_dichroic_filters(
            P.enlDichroicSet,
            _state->printRT,
            "dichroic reload failed",
            "identity filters remain active");

        printReloaded = true;
    }

    bool dichroicReloaded = false;
    if (changed.enlargerDichroicSet) {
        dichroicReloaded = try_load_dichroic_filters(
            P.enlDichroicSet,
            _state->printRT,
            "dichroic reload failed",
            "identity filters remain active");
    }

    bool filmReloaded = false;
    if (changed.filmStock) {
        _state->baseLoaded = load_film_stock_into_base(P.filmStockIndex, *_state);
        filmReloaded = _state->baseLoaded;
    }
    if (filmReloaded) {
        applyHalationProfileDefaults();
    }

    if (should_refresh_print_illuminant(printReloaded, filmReloaded)) {
        applyMetadataIlluminantDefaults(P);
        update_print_illuminant_runtime(P, _state->printRT, _state->dataDir);
    }

    if (should_apply_neutral_after_reload(printReloaded, dichroicReloaded)) {
        apply_neutral_filters_with_trace("print/dichroic");
    }
    if (filmReloaded && !neutralApplied) {
        apply_neutral_filters_with_trace("film");
    }

    if (changed.enlargerIlluminant) {
        applyNeutralFilters(P);
    }

    // Rebuild if any effective param changed
    if (_state->baseLoaded) {
#ifdef JUICER_ENABLE_COUPLERS
        applyCouplerProfileDefaults(P);
#endif
    }

    const std::uint64_t fullHash = hash_params(P);
    const std::uint64_t coreHash = hash_params_core(P);
    const std::uint64_t dirHash = hash_params_dir(P);
    store_pending_state_snapshot(*_state, P, fullHash, coreHash, dirHash);

#ifdef JUICER_ENABLE_COUPLERS
    if (changed.couplerParam) {
        Couplers::on_param_changed(changedNameOrNull);
    }
#endif
}
