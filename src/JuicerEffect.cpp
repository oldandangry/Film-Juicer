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
#include "Illuminants.h"
#include "OutputColor.h"
#include "Print.h"
#include "ParamNames.h"
#include "ProcessRoot.h"
#include "Scanner.h"
#include "SpectralData.h"
#include "Logging.h"
#include "Hash.h"
#include "mainProcessing.h"

namespace {
    enum class MeteringMethod : int {
        CenterWeighted = 0,
        Average = 1,
        Median = 2,
        Partial = 3,
        Matrix = 4,
        MultiZone = 5,
        HighlightWeighted = 6
    };

    inline const char* cstr_or_default_if_null(const char* value, const char* fallback);

    struct RenderFatalTrace {
        const char* tag = nullptr;
        const char* message = nullptr;
    };

    [[noreturn]] inline void trace_and_throw_render_fatal(const RenderFatalTrace& fatal) {
        JTRACE(fatal.tag, cstr_or_default_if_null(fatal.message, "fatal render error"));
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    inline bool nearly_equal_double(double a, double b) {
        const double diff = std::fabs(a - b);
        const double scale = std::max({1.0, std::fabs(a), std::fabs(b)});
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

#if JUICER_DIAGNOSTICS_COMPILED
    inline std::size_t cstr_len_or_zero(const char* value) {
        return value ? std::strlen(value) : 0u;
    }
#endif

    inline bool requires_nonfloat_copy(OFX::BitDepthEnum depth, int nComponents) {
        return depth != OFX::eBitDepthFloat || nComponents == 0;
    }

    inline bool param_events_suppressed(const InstanceState* state) {
        return state && state->suppressParamEvents;
    }

    inline bool bootstrap_in_progress(const InstanceState* state) {
        return state && state->inBootstrap;
    }

    inline bool has_loaded_base_state(const InstanceState* state) {
        return state && state->baseLoaded;
    }

    inline bool bootstrap_needed(const InstanceState* state) {
        return state && !state->baseLoaded;
    }

    inline void trace_changed_param_gate(
        bool traceInfo,
        const std::string& paramName,
        const char* prefix) {
#if JUICER_DIAGNOSTICS_COMPILED
        if (!traceInfo) {
            return;
        }
        std::string msg;
        msg.reserve(cstr_len_or_zero(prefix) + paramName.size() + 1);
        msg = cstr_or_default_if_null(prefix, "");
        msg += paramName;
        msg.push_back('\'');
        JTRACE("BUILD", msg);
#else
        (void)traceInfo;
        (void)paramName;
        (void)prefix;
#endif
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

    inline std::uint64_t instance_token_or_zero(const InstanceState* state) {
        return state ? state->instanceToken : 0ull;
    }

    struct SessionTokenSnapshot {
        std::uint64_t sessionSeed = 1;
        std::uint64_t instanceToken = 1;
    };

    inline SessionTokenSnapshot snapshot_session_tokens(const InstanceState* state) {
        SessionTokenSnapshot snapshot{};
        if (state && state->sessionSeed != 0) {
            snapshot.sessionSeed = state->sessionSeed;
        }
        if (state && state->instanceToken != 0) {
            snapshot.instanceToken = state->instanceToken;
        } else {
            snapshot.instanceToken = snapshot.sessionSeed;
        }
        return snapshot;
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

    inline std::shared_ptr<const DirectRenderState> load_active_direct_state_if(const InstanceState* state) {
        return state ? JuicerAtomic::load_shared_ptr(&state->activeDirectState) : nullptr;
    }

    inline std::shared_ptr<const PrintRenderState> load_active_print_state_if(const InstanceState* state) {
        return state ? JuicerAtomic::load_shared_ptr(&state->activePrintState) : nullptr;
    }

    struct ProfileKeyLabels {
        const char* paperKey = nullptr;
        const char* filmKey = nullptr;
        const char* paperLabel = "<null>";
        const char* filmLabel = "<null>";
    };

    inline ProfileKeyLabels resolve_profile_key_labels(const ParamSnapshot& snapshot) {
        ProfileKeyLabels labels{};
        labels.paperKey = snapshot.printProfileKey.empty() ? nullptr : snapshot.printProfileKey.c_str();
        labels.filmKey = snapshot.filmProfileKey.empty() ? nullptr : snapshot.filmProfileKey.c_str();
        labels.paperLabel = cstr_or_default_if_null(labels.paperKey, "<null>");
        labels.filmLabel = cstr_or_default_if_null(labels.filmKey, "<null>");
        return labels;
    }

    inline bool load_direct_film_profile_into_base_locked(
        const ParamSnapshot& snapshot,
        InstanceState& state) {
        const JuicerAssets::SelectedProfileResult selected =
            JuicerProcess::root().assets().selected_profiles_for_route(
                JuicerAssets::SelectedProfileRequest{
                    snapshot.filmProfileKey,
                    snapshot.printProfileKey,
                    snapshot.scanRoute});
        const bool validDirectSelection = selected.valid && selected.filmProfile;
        if (!validDirectSelection) {
            JTRACE(
                "SPEKTRAFILM",
                selected.diagnostic.empty()
                    ? "MissingRequiredResource phase=3A field=selected_direct_film_profile"
                    : selected.diagnostic);
            std::lock_guard<std::mutex> lock(state.m);
            state.baseLoaded = false;
            return false;
        }

        std::lock_guard<std::mutex> lock(state.m);
        state.baseLoaded =
            load_selected_spektrafilm_film_profile_into_base(*selected.filmProfile, state);
        return state.baseLoaded;
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

    struct PendingStateHashes {
        std::uint64_t fullHash = 0;
        std::uint64_t coreHash = 0;
        std::uint64_t dirHash = 0;
    };

    inline void store_pending_state_snapshot(
        InstanceState& state,
        const ParamSnapshot& params,
        const PendingStateHashes& hashes) {
        std::lock_guard<std::mutex> lock(state.pending.m);
        state.pending.params = params;
        state.pending.fullHash = hashes.fullHash;
        state.pending.coreHash = hashes.coreHash;
        state.pending.dirHash = hashes.dirHash;
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
        } else {
            rebuild_working_state(effect.getHandle(), state, pending.params);
        }
    }

    inline void rebuild_pending_state_if_needed(JuicerEffect& effect, InstanceState& state) {
        const PendingStateSnapshot pending = load_pending_state_snapshot(state);
        if (Spektrafilm::scan_route_is_print(pending.params.scanRoute)) {
            const std::uint64_t builtFullHash =
                state.lastHash.load(std::memory_order_acquire);
            if (pending_rebuild_required(pending, builtFullHash)) {
                (void)rebuild_print_render_state(state, pending.params);
            }
            return;
        }

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

    inline bool is_coupler_param_name(const char* changedName) {
        return param_name_is(changedName, JuicerParams::kDirCouplersActive) ||
               param_name_is(changedName, JuicerParams::kDirCouplersAmount);
    }

    inline bool auto_exposure_cache_param_changed(const std::string& paramName) {
        return param_name_is(paramName, kParamCameraAutoExposure) ||
               param_name_is(paramName, JuicerParams::kCameraMeteringMethod);
    }

    inline void invalidate_auto_exposure_cache_if_needed(
        InstanceState* state,
        const std::string& paramName) {
        if (!state || !auto_exposure_cache_param_changed(paramName)) {
            return;
        }
        std::lock_guard<std::mutex> cacheLock(state->autoExposureMutex);
        state->autoExposureCacheValid = false;
    }

    inline bool halation_revert_param_changed(const std::string& paramName) {
        return param_name_is(paramName, JuicerParams::kHalationRevertToStock);
    }

    inline bool changed_param_suppressed(const InstanceState* state) {
        return param_events_suppressed(state);
    }

    inline bool changed_param_bootstrap_blocked(const InstanceState* state) {
        return bootstrap_in_progress(state);
    }

    inline bool grain_preset_param_changed_by_user(bool userEdit, const std::string& paramName) {
        return user_edit_param_is(userEdit, paramName, JuicerParams::kGrainPreset);
    }

    inline bool grain_reset_advanced_param_changed_by_user(bool userEdit, const std::string& paramName) {
        return user_edit_param_is(userEdit, paramName, JuicerParams::kGrainResetAdvanced);
    }

    template <typename MediumRuntimeT>
    inline bool medium_runtime_ready_on_reference_axis(const MediumRuntimeT& runtime) {
        return runtime.staticKey.hash != 0 &&
               runtime.range.digest != 0 &&
               runtime.tables &&
               runtime.tables->K == Spectral::gShape.K;
    }

    inline bool negative_scanner_runtime_ready(const WorkingState& ws) {
        return ws.negativeScannerValid &&
               medium_runtime_ready_on_reference_axis(ws.negativeMediumRuntime);
    }

    inline bool print_scanner_runtime_ready(const WorkingState& ws) {
        return ws.printScannerValid &&
               medium_runtime_ready_on_reference_axis(ws.printMediumRuntime);
    }

    inline bool working_tables_view_ready(const WorkingState& ws) {
        const size_t k = static_cast<size_t>(Spectral::gShape.K);
        return ws.tablesView.K == Spectral::gShape.K &&
               ws.tablesView.epsY.size() == k &&
               ws.tablesView.epsM.size() == k &&
               ws.tablesView.epsC.size() == k;
    }

    inline bool working_density_curves_ready(const WorkingState& ws) {
        return !ws.densB.lambda_nm.empty() && !ws.densB.linear.empty() &&
               !ws.densG.lambda_nm.empty() && !ws.densG.linear.empty() &&
               !ws.densR.lambda_nm.empty() && !ws.densR.linear.empty();
    }

    inline bool working_baseline_ready(const WorkingState& ws) {
        return !ws.hasBaseline ||
               ws.baseMin.linear.size() == static_cast<size_t>(Spectral::gShape.K);
    }

    inline bool print_runtime_illuminants_ready(const Print::Runtime& runtime) {
        const size_t k = static_cast<size_t>(Spectral::gShape.K);
        return runtime.illumView.linear.size() == k &&
               runtime.illumEnlarger.linear.size() == k;
    }

    inline bool working_state_ready(const WorkingState* ws) {
        return ws &&
               ws->buildCounter > 0 &&
               working_tables_view_ready(*ws) &&
               working_baseline_ready(*ws) &&
               working_density_curves_ready(*ws) &&
               negative_scanner_runtime_ready(*ws);
    }

    inline bool print_runtime_ready(
        const WorkingState* ws,
        const Print::Runtime* prt,
        bool workingStateReady) {
        return (prt != nullptr) &&
               Print::profile_is_valid(prt->profile) &&
               print_runtime_illuminants_ready(*prt) &&
               ws &&
               ws->tablesPrint.K == Spectral::gShape.K &&
               workingStateReady &&
               print_scanner_runtime_ready(*ws);
    }

    struct ChangedParamFlags {
        bool referenceIlluminant = false;
        bool enlargerIlluminant = false;
        bool printProfile = false;
        bool enlargerDichroicSet = false;
        bool filmProfile = false;
        bool couplerParam = false;
    };

    inline ChangedParamFlags classify_changed_param(const char* changedName) {
        ChangedParamFlags flags{};
        if (changedName == nullptr) {
            return flags;
        }

        flags.referenceIlluminant = param_name_is(changedName, kParamReferenceIlluminant);
        flags.enlargerIlluminant = param_name_is(changedName, kParamEnlargerIlluminant);
        flags.printProfile = param_name_is(changedName, JuicerParams::kPrintProfileKey);
        flags.enlargerDichroicSet = param_name_is(changedName, kParamEnlargerDichroicSet);
        flags.filmProfile = param_name_is(changedName, JuicerParams::kFilmProfileKey);
        flags.couplerParam = is_coupler_param_name(changedName);
        return flags;
    }

    inline void mark_illuminant_override_if_changed(InstanceState& state, const ChangedParamFlags& changed) {
        if (changed.referenceIlluminant) {
            state.illuminantOverride.reference = true;
        } else if (changed.enlargerIlluminant) {
            state.illuminantOverride.enlarger = true;
        }
    }

    inline void trace_param_change_verbose_if(
        bool traceVerbose,
        const ParamSnapshot& snapshot,
        const InstanceState& state,
        const char* changedNameOrNull) {
        if (!traceVerbose) {
            return;
        }
        const ProfileKeyLabels labels = resolve_profile_key_labels(snapshot);
        const std::shared_ptr<const WorkingState> wsDbg = load_active_working_state_if(&state);
        const std::uint64_t activeBuild = working_state_build_counter_or_zero(wsDbg);
        const std::uint64_t lastHash = state.lastHash.load(std::memory_order_acquire);
        std::string msg;
        msg.reserve(224);
        msg = "params change name=";
        msg += cstr_or_default_if_null(changedNameOrNull, "<null>");
        msg += " printKey=";
        msg += labels.paperLabel;
        msg += " filmKey=";
        msg += labels.filmLabel;
        msg += " activeBuild=";
        msg += std::to_string(activeBuild);
        msg += " lastHash=";
        msg += std::to_string(lastHash);
        JTRACE_VERBOSE("PRINTDBG", msg);
    }

    inline void store_pending_hashes_for_snapshot(InstanceState& state, const ParamSnapshot& snapshot) {
        store_pending_state_snapshot(
            state,
            snapshot,
            PendingStateHashes{
                hash_params(snapshot),
                hash_params_core(snapshot),
                hash_params_dir(snapshot)});
    }

    inline bool reload_film_stock_if_requested(
        bool requested,
        const ParamSnapshot& snapshot,
        InstanceState& state) {
        if (!requested) {
            return false;
        }
        return load_direct_film_profile_into_base_locked(snapshot, state);
    }

    inline bool should_skip_param_change_due_to_suppression(const InstanceState* state) {
        return param_events_suppressed(state);
    }

    inline bool should_bootstrap_for_param_change(const InstanceState* state) {
        return bootstrap_needed(state);
    }

    template <typename ApplyFn>
    inline void apply_when_film_reloaded(bool filmReloaded, ApplyFn&& applyFn) {
        if (!filmReloaded) {
            return;
        }
        std::forward<ApplyFn>(applyFn)();
    }

    inline double sanitize_finite_clamped(double value, double fallback, double minValue, double maxValue) {
        if (!is_finite(value))
            return fallback;
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

    inline int read_bool_param_as_i32(OFX::BooleanParam* param, bool fallback) {
        bool value = fallback;
        if (param) {
            param->getValue(value);
        }
        return bool_to_i32(value);
    }

    inline double sanitize_positive_finite_or(double value, double fallback) {
        return (is_finite(value) && value > 0.0) ? value : fallback;
    }

    inline double read_camera_film_format_mm_or_default(OFX::DoubleParam* param) {
        return sanitize_positive_finite_or(read_double_param_or(param, 35.0), 35.0);
    }

    inline bool is_positive_finite(double value) {
        return is_finite(value) && value > 0.0;
    }

    inline double read_sanitized_double(
        OFX::DoubleParam* param,
        double fallback,
        double minValue,
        double maxValue) {
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

    inline double read_sanitized_unit_double(OFX::DoubleParam* param, double fallback) {
        return read_sanitized_double(param, fallback, 0.0, 1.0);
    }

    inline float read_sanitized_unit_float(OFX::DoubleParam* param, float fallback) {
        return read_sanitized_float(param, fallback, 0.0, 1.0);
    }

    inline float read_sanitized_nonnegative_float(
        OFX::DoubleParam* param,
        float fallback,
        double maxValue) {
        return read_sanitized_float(param, fallback, 0.0, maxValue);
    }

    inline float read_sanitized_0_to_10_float(OFX::DoubleParam* param, float fallback) {
        return read_sanitized_nonnegative_float(param, fallback, 10.0);
    }

    struct GrainClumpControls {
        float temporalMix = 0.30f;
        float morphPeriodSec = 8.0f;
    };

    inline GrainClumpControls read_grain_clump_controls(
        OFX::DoubleParam* temporalMixParam,
        OFX::DoubleParam* morphPeriodSecParam) {
        GrainClumpControls controls{};
        controls.temporalMix = read_sanitized_float(temporalMixParam, 0.30f, 0.0, 0.30);
        controls.morphPeriodSec = read_sanitized_float(morphPeriodSecParam, 8.0f, 5.0, 60.0);
        return controls;
    }

    struct GrainSurfaceArtifacts {
        float filmDustAmount = 0.0f;
        float gateDustAmount = 0.0f;
        float filmScratchAmount = 0.0f;
        float gateScratchAmount = 0.0f;
    };

    inline GrainSurfaceArtifacts read_grain_surface_artifacts(
        OFX::DoubleParam* filmDustParam,
        OFX::DoubleParam* gateDustParam,
        OFX::DoubleParam* filmScratchParam,
        OFX::DoubleParam* gateScratchParam) {
        GrainSurfaceArtifacts artifacts{};
        artifacts.filmDustAmount = read_sanitized_0_to_10_float(filmDustParam, 0.0f);
        artifacts.gateDustAmount = read_sanitized_0_to_10_float(gateDustParam, 0.0f);
        artifacts.filmScratchAmount = read_sanitized_0_to_10_float(filmScratchParam, 0.0f);
        artifacts.gateScratchAmount = read_sanitized_0_to_10_float(gateScratchParam, 0.0f);
        return artifacts;
    }

    struct GlareCompensationUiValues {
        float factor = 0.0f;
        float density = 1.2f;
        float transition = 0.3f;
    };

    inline GlareCompensationUiValues read_glare_compensation_ui_values(
        OFX::DoubleParam* factorParam,
        OFX::DoubleParam* densityParam,
        OFX::DoubleParam* transitionParam) {
        GlareCompensationUiValues values{};
        values.factor = read_sanitized_unit_float(factorParam, 0.0f);
        values.density = read_sanitized_float(densityParam, 1.2f, 0.0, 3.0);
        values.transition = read_sanitized_float(transitionParam, 0.3f, 0.0, 2.0);
        return values;
    }

    struct GlareCompensationSnapshotValues {
        double factor = 0.0;
        double density = 1.2;
        double transition = 0.3;
    };

    inline GlareCompensationSnapshotValues read_glare_compensation_snapshot_values(
        OFX::DoubleParam* factorParam,
        OFX::DoubleParam* densityParam,
        OFX::DoubleParam* transitionParam,
        double factorFallback,
        double densityFallback,
        double transitionFallback) {
        GlareCompensationSnapshotValues values{};
        values.factor = read_sanitized_unit_double(factorParam, factorFallback);
        values.density = read_sanitized_double(densityParam, densityFallback, 0.0, 3.0);
        values.transition = read_sanitized_double(transitionParam, transitionFallback, 0.0, 2.0);
        return values;
    }

    inline bool read_finite_unit_interval(
        OFX::DoubleParam* param,
        double fallback,
        double& valueOut) {
        valueOut = read_double_param_or(param, fallback);
        if (!is_finite(valueOut)) {
            return false;
        }
        valueOut = sanitize_finite_clamped(valueOut, fallback, 0.0, 1.0);
        return true;
    }

    inline bool grain_user_edit_active(
        bool userEdit,
        const std::string& paramName,
        const char* expectedParam,
        const InstanceState* state,
        OFX::DoubleParam* sourceParam) {
        return user_edit_param_is(userEdit, paramName, expectedParam) &&
               state &&
               sourceParam;
    }

    inline void set_grain_chroma_weights(float chroma, float& sharedWeightOut, float& indWeightOut) {
        sharedWeightOut = std::sqrt(std::max(0.0f, 1.0f - chroma));
        indWeightOut = std::sqrt(std::max(0.0f, chroma));
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

    inline std::string read_str_choice_param_or(OFX::StrChoiceParam* param, const std::string& fallback) {
        std::string value = fallback;
        if (param) {
            param->getValue(value);
        }
        return value.empty() ? fallback : value;
    }

    inline Spektrafilm::ProfilePolarity capture_profile_polarity_for_key(const std::string& filmProfileKey) {
        const Spektrafilm::ProfileCatalog& catalog =
            JuicerProcess::root().assets().spektrafilm_profile_catalog();
        for (const Spektrafilm::ProfileCatalogEntry& entry : catalog.filmProfiles) {
            if (entry.key == filmProfileKey) {
                return entry.polarity;
            }
        }
        return Spektrafilm::ProfilePolarity::Negative;
    }

    inline Spektrafilm::ScanRoute read_resolved_scan_route(
        OFX::StrChoiceParam* scanRouteParam,
        const std::string& filmProfileKey) {
        const Spektrafilm::ProfilePolarity capturePolarity =
            capture_profile_polarity_for_key(filmProfileKey);
        const Spektrafilm::ScanRoute defaultRoute =
            Spektrafilm::default_scan_route_for_polarity(capturePolarity);
        const std::string routeKey =
            read_str_choice_param_or(scanRouteParam, Spektrafilm::scan_route_key(defaultRoute));
        const Spektrafilm::ScanRoute userRouteSelection =
            Spektrafilm::scan_route_from_key_or(routeKey, defaultRoute);
        return Spektrafilm::resolve_scan_route(capturePolarity, userRouteSelection);
    }

    inline int read_choice_param_clamped(
        OFX::ChoiceParam* param,
        int fallback,
        int minValue,
        int maxValue) {
        return std::clamp(read_choice_param_or(param, fallback), minValue, maxValue);
    }

    inline int sanitize_scanner_lut_resolution_or_default(int value) {
        return (value < 17 || value > 128) ? 17 : value;
    }

    inline std::uint32_t read_scanner_lut_resolution_or_default(
        OFX::IntParam* param,
        std::uint32_t fallback) {
        const int fallbackI32 = static_cast<int>(fallback);
        const int value = read_int_param_or(param, fallbackI32);
        return static_cast<std::uint32_t>(sanitize_scanner_lut_resolution_or_default(value));
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

    struct ScannerUnsharpPair {
        float sigmaPx = 0.7f;
        float amount = 1.0f;
    };

    inline float read_scanner_blur_sigma_px_or_default(OFX::DoubleParam* param, float fallback) {
        const double blur = sanitize_finite_clamped(
            read_double_param_or(param, static_cast<double>(fallback)),
            0.55,
            0.0,
            10.0);
        return static_cast<float>(blur);
    }

    inline ScannerUnsharpPair read_scanner_unsharp_or_default(
        OFX::Double2DParam* param,
        float sigmaFallback,
        float amountFallback) {
        const std::array<double, 2> unsharp = read_double2_param_or(
            param,
            {{static_cast<double>(sigmaFallback), static_cast<double>(amountFallback)}});
        const double sigma = sanitize_finite_clamped(unsharp[0], 0.7, 0.0, 5.0);
        const double amount = sanitize_finite_clamped(unsharp[1], 1.0, 0.0, 3.0);
        ScannerUnsharpPair out{};
        out.sigmaPx = static_cast<float>(sigma);
        out.amount = static_cast<float>(amount);
        return out;
    }

    inline double sanitize_enlarger_filter_shift_or_zero(double value) {
        if (!is_finite(value)) {
            return 0.0;
        }
        const double limit = static_cast<double>(Print::kEnlargerSteps);
        return std::clamp(value, -limit, limit);
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

    inline void set_str_choice_param_if(OFX::StrChoiceParam* param, const char* value) {
        if (param && value) {
            param->setValue(value);
        }
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
        double maxValue) {
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

    inline std::array<double, 3> read_sanitized_triplet_from_master(
        OFX::Double3DParam* param,
        double masterValue,
        double minValue,
        double maxValue) {
        std::array<double, 3> defaults{};
        defaults.fill(masterValue);
        return read_sanitized_double3(
            param,
            defaults,
            minValue,
            maxValue);
    }

    inline std::array<double, 2> read_sanitized_double2(
        OFX::Double2DParam* param,
        const std::array<double, 2>& defaults,
        double minValue,
        double maxValue) {
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

    inline void assign_grain_triplet_controls(
        Profiles::GrainMetadata& grain,
        const std::array<double, 3>& particleScale,
        const std::array<double, 3>& particleScaleLayers,
        const std::array<double, 3>& densityMin,
        const std::array<double, 3>& uniformity) {
        cast_array(grain.agxParticleScale, particleScale);
        cast_array(grain.agxParticleScaleLayers, particleScaleLayers);
        cast_array(grain.densityMin, densityMin);
        cast_array(grain.uniformity, uniformity);
    }

    inline void read_scanner_snapshot_values(
        OFX::DoubleParam* scannerLensBlurParam,
        OFX::Double2DParam* scannerUnsharpParam,
        OFX::BooleanParam* scannerUseLutParam,
        OFX::IntParam* scannerLutResolutionParam,
        ParamSnapshot& snapshot) {
        snapshot.scannerLensBlurSigmaPx = read_double_param_or(
            scannerLensBlurParam,
            snapshot.scannerLensBlurSigmaPx);
        snapshot.scannerUnsharpMask = read_double2_param_or(
            scannerUnsharpParam,
            snapshot.scannerUnsharpMask);
        snapshot.scannerUseLut = read_bool_param_as_i32(scannerUseLutParam, true);
        snapshot.scannerLutResolution = read_int_param_or(
            scannerLutResolutionParam,
            snapshot.scannerLutResolution);
    }

    inline void read_output_snapshot_values(
        OFX::ChoiceParam* outputColorSpaceParam,
        OFX::BooleanParam* outputCctfEncodingParam,
        OFX::BooleanParam* outputLinearPassThroughParam,
        ParamSnapshot& snapshot) {
        snapshot.outputColorSpace = read_choice_param_or(outputColorSpaceParam, snapshot.outputColorSpace);
        snapshot.outputCctfEncoding = read_bool_param_as_i32(outputCctfEncodingParam, true);
        snapshot.outputLinearPassThrough = read_bool_param_as_i32(outputLinearPassThroughParam, false);
    }

    inline void read_profile_snapshot_choices(
        OFX::StrChoiceParam* filmProfileKeyParam,
        OFX::StrChoiceParam* printProfileKeyParam,
        OFX::StrChoiceParam* scanRouteParam,
        OFX::ChoiceParam* spectralModeParam,
        OFX::ChoiceParam* refIlluminantParam,
        OFX::ChoiceParam* enlargerIlluminantParam,
        OFX::ChoiceParam* enlargerDichroicSetParam,
        ParamSnapshot& snapshot) {
        snapshot.filmProfileKey = read_str_choice_param_or(filmProfileKeyParam, snapshot.filmProfileKey);
        snapshot.printProfileKey = read_str_choice_param_or(printProfileKeyParam, snapshot.printProfileKey);
        snapshot.scanRoute = read_resolved_scan_route(scanRouteParam, snapshot.filmProfileKey);
        snapshot.spectralUpsamplingMode = read_choice_param_or(spectralModeParam, snapshot.spectralUpsamplingMode);
        snapshot.refIll = read_choice_param_or(refIlluminantParam, snapshot.refIll);
        snapshot.enlIll = read_choice_param_or(enlargerIlluminantParam, snapshot.enlIll);
        snapshot.enlDichroicSet = read_choice_param_or(enlargerDichroicSetParam, snapshot.enlDichroicSet);
    }

    inline void read_print_recipe_snapshot_values(
        OFX::DoubleParam* printExposureParam,
        OFX::DoubleParam* printPreflashParam,
        OFX::BooleanParam* printExposureCompensationParam,
        OFX::DoubleParam* enlargerYParam,
        OFX::DoubleParam* enlargerMParam,
        OFX::DoubleParam* enlargerCParam,
        ParamSnapshot& snapshot) {
        snapshot.printExposure = sanitize_finite_or(
            read_double_param_or(printExposureParam, snapshot.printExposure),
            snapshot.printExposure);
        snapshot.printPreflashExposure = sanitize_finite_or(
            read_double_param_or(printPreflashParam, snapshot.printPreflashExposure),
            snapshot.printPreflashExposure);
        snapshot.printExposureCompensation =
            read_bool_param_as_i32(printExposureCompensationParam, true);
        snapshot.normalizePrintExposure = 1;
        snapshot.printUiYmcCc = {
            sanitize_finite_or(read_double_param_or(enlargerYParam, 0.0), 0.0),
            sanitize_finite_or(read_double_param_or(enlargerMParam, 0.0), 0.0),
            sanitize_finite_or(read_double_param_or(enlargerCParam, 0.0), 0.0)};
    }

    inline void read_input_snapshot_values(
        OFX::ChoiceParam* inputColorSpaceParam,
        OFX::BooleanParam* inputCctfDecodingParam,
        ParamSnapshot& snapshot) {
        snapshot.inputColorSpace = read_choice_param_or(inputColorSpaceParam, snapshot.inputColorSpace);
        snapshot.inputCctfDecoding = read_bool_param_as_i32(inputCctfDecodingParam, false);
    }

    inline void read_coupler_snapshot_values(
        OFX::BooleanParam* couplersActiveParam,
        OFX::DoubleParam* couplersAmountParam,
        ParamSnapshot& snapshot) {
        snapshot.couplersActive = read_bool_param_as_i32(couplersActiveParam, true);
        snapshot.couplersAmount = read_double_param_or(couplersAmountParam, snapshot.couplersAmount);
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
            case OFX::ePixelComponentRGBA:
                return 4;
            case OFX::ePixelComponentRGB:
                return 3;
            case OFX::ePixelComponentAlpha:
                return 1;
            default:
                return 0;
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

    constexpr std::uint64_t kAutoExposureMaskCacheMaxBytes = 96ull * 1024ull * 1024ull;
    constexpr std::size_t kAutoExposureMedianScratchMaxSamples =
        static_cast<std::size_t>(kAutoExposureMaskCacheMaxBytes / sizeof(float));
    static std::atomic<std::uint64_t> gAutoExposureMaskCacheResidentBytes{0};

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
        } else if (previousBytes > nextBytes) {
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
                        float linear[3];
                        if (singleComponent) {
                            const float gray = rowPixIt[0];
                            const float grayRgb[3] = {gray, gray, gray};
                            Spectral::apply_input_cctf_decoding(inputColorSpace, applyCctfDecoding, grayRgb, linear);
                        } else {
                            Spectral::apply_input_cctf_decoding(inputColorSpace, applyCctfDecoding, rowPixIt, linear);
                        }
                        float XYZ[3];
                        rgbToXYZ.mul(linear, XYZ);
                        const float Y = XYZ[1];
                        if (is_finite(Y)) {
                            sumY += static_cast<double>(Y) * w;
                            sumMask += w;
                        }
                        rowPixIt += pixelStride;
                    }
                    continue;
                }
                for (int xOff = 0; xOff < width; ++xOff) {
                    const int xx = bounds.x1 + xOff;
                    const double w = *maskIt++;
                    const float* pix = reinterpret_cast<const float*>(img->getPixelAddress(xx, yy));
                    if (!pix) {
                        continue;
                    }
                    float linear[3];
                    if (singleComponent) {
                        const float gray = pix[0];
                        const float grayRgb[3] = {gray, gray, gray};
                        Spectral::apply_input_cctf_decoding(inputColorSpace, applyCctfDecoding, grayRgb, linear);
                    } else {
                        Spectral::apply_input_cctf_decoding(inputColorSpace, applyCctfDecoding, pix, linear);
                    }
                    float XYZ[3];
                    rgbToXYZ.mul(linear, XYZ);
                    const float Y = XYZ[1];
                    if (!is_finite(Y)) {
                        continue;
                    }
                    sumY += static_cast<double>(Y) * w;
                    sumMask += w;
                }
            }
            if (outSumMask) {
                *outSumMask = sumMask;
            }
            return (sumMask > 0.0) ? (sumY / sumMask) : 0.0;
        };

        auto accumulateYUncached = [&](double* outSumMask) {
            CenterWeightGeometry geometry{};
            if (!build_center_weight_geometry(width, height, sigma, geometry)) {
                if (outSumMask) {
                    *outSumMask = 0.0;
                }
                return 0.0;
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
                        float linear[3];
                        if (singleComponent) {
                            const float gray = rowPixIt[0];
                            const float grayRgb[3] = {gray, gray, gray};
                            Spectral::apply_input_cctf_decoding(inputColorSpace, applyCctfDecoding, grayRgb, linear);
                        } else {
                            Spectral::apply_input_cctf_decoding(inputColorSpace, applyCctfDecoding, rowPixIt, linear);
                        }
                        float XYZ[3];
                        rgbToXYZ.mul(linear, XYZ);
                        const float Y = XYZ[1];
                        if (is_finite(Y)) {
                            sumY += static_cast<double>(Y) * w;
                            sumMask += w;
                        }
                        rowPixIt += pixelStride;
                        nx += geometry.invWidth;
                    }
                    continue;
                }
                for (int xOff = 0; xOff < width; ++xOff) {
                    const int xx = bounds.x1 + xOff;
                    const double w = gaussian_weight_from_nx(nx, geometry.scaleX, normY, geometry.invSigmaDenom);
                    const float* pix = reinterpret_cast<const float*>(img->getPixelAddress(xx, yy));
                    if (pix) {
                        float linear[3];
                        if (singleComponent) {
                            const float gray = pix[0];
                            const float grayRgb[3] = {gray, gray, gray};
                            Spectral::apply_input_cctf_decoding(inputColorSpace, applyCctfDecoding, grayRgb, linear);
                        } else {
                            Spectral::apply_input_cctf_decoding(inputColorSpace, applyCctfDecoding, pix, linear);
                        }
                        float XYZ[3];
                        rgbToXYZ.mul(linear, XYZ);
                        const float Y = XYZ[1];
                        if (is_finite(Y)) {
                            sumY += static_cast<double>(Y) * w;
                            sumMask += w;
                        }
                    }
                    nx += geometry.invWidth;
                }
            }
            if (outSumMask) {
                *outSumMask = sumMask;
            }
            return (sumMask > 0.0) ? (sumY / sumMask) : 0.0;
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

                emitBypassTrace = (previousCachedBytes > 0) || !wasBypass || (previousRequestedBytes != requestedMaskBytes);
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
            return !s.autoExposureMaskValid || s.autoExposureMaskWidth != width || s.autoExposureMaskHeight != height || !nearly_equal_double(s.autoExposureMaskSigma, sigma) || !nearly_equal_double(s.autoExposureMaskRenderScaleX, renderScaleX) || !nearly_equal_double(s.autoExposureMaskRenderScaleY, renderScaleY) || s.autoExposureMaskClipToken != clipToken || !weights || weights->size() != expectedMaskSize;
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
        } else {
            localValues.reserve(total);
            valuesPtr = &localValues;
        }
        std::vector<float>& values = *valuesPtr;
        for (int yy = bounds.y1; yy < bounds.y2; ++yy) {
            const float* rowPix = row_start_if_covered(img, srcBounds, bounds, yy);
            if (rowPix) {
                const float* rowPixIt = rowPix;
                for (int xOff = 0; xOff < width; ++xOff) {
                    float linear[3];
                    if (singleComponent) {
                        const float gray = rowPixIt[0];
                        const float grayRgb[3] = {gray, gray, gray};
                        Spectral::apply_input_cctf_decoding(inputColorSpace, applyCctfDecoding, grayRgb, linear);
                    } else {
                        Spectral::apply_input_cctf_decoding(inputColorSpace, applyCctfDecoding, rowPixIt, linear);
                    }
                    float XYZ[3];
                    rgbToXYZ.mul(linear, XYZ);
                    float Y = XYZ[1];
                    if (is_finite(Y)) {
                        values.emplace_back((Y < 0.0f) ? 0.0f : Y);
                    }
                    rowPixIt += pixelStride;
                }
                continue;
            }
            for (int xOff = 0; xOff < width; ++xOff) {
                const int x = bounds.x1 + xOff;
                const float* pix = reinterpret_cast<const float*>(img->getPixelAddress(x, yy));
                if (!pix) {
                    continue;
                }
                float linear[3];
                if (singleComponent) {
                    const float gray = pix[0];
                    const float grayRgb[3] = {gray, gray, gray};
                    Spectral::apply_input_cctf_decoding(inputColorSpace, applyCctfDecoding, grayRgb, linear);
                } else {
                    Spectral::apply_input_cctf_decoding(inputColorSpace, applyCctfDecoding, pix, linear);
                }
                float XYZ[3];
                rgbToXYZ.mul(linear, XYZ);
                float Y = XYZ[1];
                if (is_finite(Y)) {
                    values.emplace_back((Y < 0.0f) ? 0.0f : Y);
                }
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

    class ScopedBootstrapState {
    public:
        explicit ScopedBootstrapState(InstanceState* state)
            : _state(state) {
            if (_state) {
                _previousSuppression = _state->suppressParamEvents;
                _previousBootstrap = _state->inBootstrap;
                _state->suppressParamEvents = true;
                _state->inBootstrap = true;
            }
        }

        ~ScopedBootstrapState() {
            if (_state) {
                _state->suppressParamEvents = _previousSuppression;
                _state->inBootstrap = _previousBootstrap;
            }
        }

    private:
        InstanceState* _state = nullptr;
        bool _previousSuppression = false;
        bool _previousBootstrap = false;
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
    opts.lensBlurSigmaPx = read_scanner_blur_sigma_px_or_default(
        _pScannerLensBlur,
        opts.lensBlurSigmaPx);

    const ScannerUnsharpPair unsharp = read_scanner_unsharp_or_default(
        _pScannerUnsharp,
        opts.unsharpSigmaPx,
        opts.unsharpAmount);
    opts.unsharpSigmaPx = unsharp.sigmaPx;
    opts.unsharpAmount = unsharp.amount;
    return opts;
}

Scanner::Settings JuicerEffect::gatherScannerSettings() const {
    Scanner::Settings settings{};
    const bool useLut = read_bool_param_or(_pScannerUseLut, settings.useLut);
    settings.useLut = useLut;
    settings.lutResolution = read_scanner_lut_resolution_or_default(
        _pScannerLutResolution,
        settings.lutResolution);
    return settings;
}

Print::Params JuicerEffect::gatherPrintParams() const {
    Print::Params params{};
    const ParamSnapshot snapshot = snapshotParams();
    const double pexp = read_double_param_or(_pPrintExposure, 1.0);
    const double preflash = read_double_param_or(_pPrintPreflash, 0.0);
    const double y = read_double_param_or(_pEnlargerY, 0.0);
    const double m = read_double_param_or(_pEnlargerM, 0.0);
    const double c = read_double_param_or(_pEnlargerC, 0.0);
    params.bypass = !Spektrafilm::scan_route_is_print(snapshot.scanRoute);
    params.exposure = static_cast<float>(pexp);
    params.preflashExposure = static_cast<float>(preflash);
    params.yFilter = static_cast<float>(sanitize_enlarger_filter_shift_or_zero(y));
    params.mFilter = static_cast<float>(sanitize_enlarger_filter_shift_or_zero(m));
    params.cFilter = static_cast<float>(sanitize_enlarger_filter_shift_or_zero(c));
    return params;
}

Profiles::HalationMetadata JuicerEffect::gatherHalationUi() const {
    Profiles::HalationMetadata halation{};

    halation.active = read_bool_param_or(_pHalationActive, false);

    const std::array<double, 3> strengthPercent =
        read_sanitized_double3(_pHalationStrength, {{3.0, 0.30, 0.10}}, 0.0, 100.0);
    const std::array<double, 3> sizeUm =
        read_sanitized_double3(_pHalationSizeUm, {{200.0, 200.0, 200.0}}, 0.0, 1000.0);
    const std::array<double, 3> scatterStrengthPercent =
        read_sanitized_double3(_pHalationScatteringStrength, {{1.0, 2.0, 4.0}}, 0.0, 100.0);
    const std::array<double, 3> scatterSizeUm =
        read_sanitized_double3(_pHalationScatteringSizeUm, {{30.0, 20.0, 15.0}}, 0.0, 1000.0);

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
    if (!has_loaded_base_state(_state.get())) {
        return;
    }

    const Profiles::HalationMetadata& halationCfg = _state->base.halation;

    const std::array<double, 3> currentStrength =
        read_double3_param_or(_pHalationStrength, {{0.0, 0.0, 0.0}});
    const std::array<double, 3> currentSize =
        read_double3_param_or(_pHalationSizeUm, {{0.0, 0.0, 0.0}});
    const std::array<double, 3> currentScatterStrength =
        read_double3_param_or(_pHalationScatteringStrength, {{0.0, 0.0, 0.0}});
    const std::array<double, 3> currentScatterSize =
        read_double3_param_or(_pHalationScatteringSizeUm, {{0.0, 0.0, 0.0}});

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
        scatterStrengthPct[0],
        scatterStrengthPct[1],
        scatterStrengthPct[2]);
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
        return {{0.8, 1.0, 2.0}};
    }

    static std::array<double, 3> default_particle_scale_layers_ratio() {
        return {{2.5, 1.0, 0.5}};
    }

    static std::array<double, 3> default_density_min_ratio() {
        return {{0.07, 0.08, 0.12}};
    }

    static std::array<double, 3> default_uniformity_ratio() {
        return {{0.97, 0.97, 0.99}};
    }

    static void normalize_ratio(std::array<double, 3>& values) {
        double* data = values.data();
        double sum = 0.0;
        const double* const dataEnd = data + values.size();
        for (; data < dataEnd; ++data) {
            const double v = *data;
            if (!is_positive_finite(v)) {
                values = {{1.0, 1.0, 1.0}};
                return;
            }
            sum += v;
        }
        const double mean = sum / 3.0;
        if (!is_positive_finite(mean)) {
            values = {{1.0, 1.0, 1.0}};
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
            default_uniformity_ratio()};
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

} // namespace

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

    const double sharpness = read_sanitized_unit_double(_pGrainSharpness, preset.sharpness);

    const double chroma = read_sanitized_unit_double(_pGrainChroma, preset.chroma);

    const double texture = read_sanitized_unit_double(_pGrainTexture, preset.texture);

    const GrainAdvancedDefaults advancedDefaults = compute_grain_advanced_defaults(
        preset,
        sharpness,
        texture);

    grain.agxParticleAreaUm2 = read_sanitized_0_to_10_float(
        _pGrainParticleAreaUm2,
        static_cast<float>(preset.particleAreaUm2));

    grain.sizeMixScale = read_sanitized_float(
        _pGrainSizeMixScale,
        static_cast<float>(preset.sizeMixScale),
        1.0,
        50.0);

    grain.sizeMixWeight = read_sanitized_unit_float(
        _pGrainSizeMixWeight,
        static_cast<float>(advancedDefaults.sizeMixWeight));

    grain.sizeMixWeightMid = read_sanitized_unit_float(_pGrainSizeMixWeightMid, 0.0f);

    grain.blurDyeCloudsUm = read_sanitized_0_to_10_float(
        _pGrainBlurDyeCloudsUm,
        static_cast<float>(advancedDefaults.blurDyeClouds));

    grain.chroma = static_cast<float>(chroma);
    set_grain_chroma_weights(
        grain.chroma,
        grain.chromaSharedWeight,
        grain.chromaIndWeight);
    const std::array<double, 3> particleScale =
        read_sanitized_triplet_from_master(_pGrainParticleScale, preset.particleScaleMaster, 0.0, 10.0);
    const std::array<double, 3> particleScaleLayers =
        read_sanitized_triplet_from_master(_pGrainParticleScaleLayers, preset.particleScaleLayersMaster, 0.0, 10.0);
    const std::array<double, 3> densityMin =
        read_sanitized_triplet_from_master(_pGrainDensityMin, preset.densityMinMaster, 0.0, 1.0);
    const std::array<double, 3> uniformity =
        read_sanitized_triplet_from_master(_pGrainUniformity, preset.uniformityMaster, 0.0, 1.0);
    assign_grain_triplet_controls(
        grain,
        particleScale,
        particleScaleLayers,
        densityMin,
        uniformity);

    const GrainClumpControls clumpControls =
        read_grain_clump_controls(_pGrainClumpTemporalMix, _pGrainClumpMorphPeriodSec);
    grain.clumpTemporalMix = clumpControls.temporalMix;
    grain.clumpMorphPeriodSec = clumpControls.morphPeriodSec;

    std::array<double, 2> microStructure = {{advancedDefaults.microCell,
                                             advancedDefaults.microSigma}};
    microStructure = read_sanitized_double2(_pGrainMicroStructure, microStructure, 0.0, 1000.0);
    cast_array(grain.microStructure, microStructure);

    grain.breathingDebug = read_bool_param_or(_pGrainBreathingDebug, false);

    grain.debugView = read_choice_param_clamped(_pGrainDebugView, 0, 0, 6);

    const GrainSurfaceArtifacts artifacts = read_grain_surface_artifacts(
        _pFilmDustAmount,
        _pGateDustAmount,
        _pFilmScratchAmount,
        _pGateScratchAmount);
    grain.filmDustAmount = artifacts.filmDustAmount;
    grain.gateDustAmount = artifacts.gateDustAmount;
    grain.filmScratchAmount = artifacts.filmScratchAmount;
    grain.gateScratchAmount = artifacts.gateScratchAmount;

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

    const double sharpness = read_sanitized_unit_double(_pGrainSharpness, preset.sharpness);

    const double texture = read_sanitized_unit_double(_pGrainTexture, preset.texture);

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
    } else {
        set_double_param_hint_if(_pGrainChroma, _grainChromaHint);
    }
}

Profiles::ProfileGlare JuicerEffect::gatherGlareUi() const {
    Profiles::ProfileGlare glare{};

    glare.active = read_bool_param_or(_pGlareActive, true);

    glare.percent = read_sanitized_unit_float(_pGlarePercent, 0.10f);

    glare.roughness = read_sanitized_unit_float(_pGlareRoughness, 0.4f);

    glare.blur = read_sanitized_0_to_10_float(_pGlareBlurSigmaPx, 0.5f);

    const GlareCompensationUiValues compensation = read_glare_compensation_ui_values(
        _pGlareCompRemovalFactor,
        _pGlareCompRemovalDensity,
        _pGlareCompRemovalTransition);
    glare.compensationRemovalFactor = compensation.factor;
    glare.compensationRemovalDensity = compensation.density;
    glare.compensationRemovalTransition = compensation.transition;

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

// SF_TEMP_BRIDGE_CPUAutoExposure owner=Phase4-print-route:
// reason=legacy non-direct metering; allowed=no product call site while non-direct render is
// blocked; output_impact=blocked print route; hash_impact=none; resource_impact=CPU source reads;
// removal=Phase4 print-route metering cutover.
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
        } catch (...) {
            // Ignore failures; fall back to full bounds.
            meterBounds = fullBounds;
        }
    }
    result.meterBounds = meterBounds;
    result.meterBoundsValid = true;

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
        } else {
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
        } else {
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

JuicerEffect::WorkingStateInfo JuicerEffect::prepareWorkingState() const {
    WorkingStateInfo info{};
    if (!has_loaded_base_state(_state.get())) {
        return info;
    }

    info.workingState = load_active_working_state_if(_state.get());

    const WorkingState* ws = info.workingState.get();
    if (ws && ws->buildCounter > 0 && ws->printRT) {
        info.printRuntime = ws->printRT.get();
    }

    info.workingStateReady = working_state_ready(ws);

    const Print::Runtime* prt = info.printRuntime;
    info.printRuntimeReady = print_runtime_ready(ws, prt, info.workingStateReady);

    return info;
}

JuicerEffect::JuicerEffect(OfxImageEffectHandle handle)
    : OFX::ImageEffect(handle) {
    // Cache clips (wrappers) for Step 2; safe even if render still uses the existing path.
    try {
        _src = fetchClip(kOfxImageEffectSimpleSourceClipName); // "Source"
        _dst = fetchClip(kOfxImageEffectOutputClipName);       // "Output"
    } catch (...) {
        _src = nullptr;
        _dst = nullptr;
    }

    // Cache parameter handles (wrappers)
    try {
        _pExposure = fetchDoubleParam(kParamExposure);
        _pCameraAutoExposure = fetchBooleanParam(kParamCameraAutoExposure);
        _pCameraFilmFormat = fetchDoubleParam(JuicerParams::kCameraFilmFormatMm);
        _pCameraMeteringMethod = fetchChoiceParam(JuicerParams::kCameraMeteringMethod);
        _pFilmProfileKey = fetchStrChoiceParam(JuicerParams::kFilmProfileKey);
        _pSpectralMode = fetchChoiceParam(kParamSpectralMode);
        _pPrintProfileKey = fetchStrChoiceParam(JuicerParams::kPrintProfileKey);
        _pRefIll = fetchChoiceParam("ReferenceIlluminant");
        _pEnlIll = fetchChoiceParam("EnlargerIlluminant");
        _pEnlDichroicSet = fetchChoiceParam(kParamEnlargerDichroicSet);
        _pInputColorSpace = fetchChoiceParam(JuicerParams::kInputColorSpace);
        _pInputCctfDecoding = fetchBooleanParam(JuicerParams::kInputCctfDecoding);
        _pScanRoute = fetchStrChoiceParam(JuicerParams::kParamScanRoute);
        _pOutputColorSpace = fetchChoiceParam(kParamOutputColorSpace);
        _pOutputCctfEncoding = fetchBooleanParam(kParamOutputCctfEncoding);
        _pOutputLinearPassThrough = fetchBooleanParam(kParamOutputLinearPassThrough);


        _pCouplersActive = fetchBooleanParam(JuicerParams::kDirCouplersActive);
        _pCouplersAmount = fetchDoubleParam(JuicerParams::kDirCouplersAmount);

        _pScannerLensBlur = fetchDoubleParam(JuicerParams::kScannerLensBlurSigmaPx);
        _pScannerUnsharp = fetchDouble2DParam(JuicerParams::kScannerUnsharpMask);
        _pScannerUseLut = fetchBooleanParam(JuicerParams::kScannerUseLut);
        _pScannerLutResolution = fetchIntParam(JuicerParams::kScannerLutResolution);

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
    } catch (...) {
        // Safe: any missing param will remain nullptr and defaults are used in snapshot/usage paths.
        JTRACE("PARAM", "parameter cache bootstrap incomplete; defaults will be used for missing handles");
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
    JuicerAtomic::store_shared_ptr(&_state->activeDirectState, std::shared_ptr<const DirectRenderState>{});
    JuicerAtomic::store_shared_ptr(&_state->activePrintState, std::shared_ptr<const PrintRenderState>{});
    _state->activeBuildCounter = 0;
    {
        const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        const std::uint64_t seedFields[2] = {
            static_cast<std::uint64_t>(now),
            static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(this))};
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
    try {
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
    } catch (...) {
        JuicerLogging::discard_current_exception();
    }

    try {
        _state.reset();
    } catch (...) {
        JuicerLogging::discard_current_exception();
    }
}

[[noreturn]] void JuicerEffect::throw_spektrafilm_phase1a_render_cutoff(const OFX::RenderArguments& args) const {
    const std::string filmKey = read_str_choice_param_or(_pFilmProfileKey, Spektrafilm::kDefaultFilmProfileKey);
    const std::string printKey = read_str_choice_param_or(_pPrintProfileKey, Spektrafilm::kDefaultPrintProfileKey);
    const char* backend = args.isEnabledCudaRender ? "cuda" : "non-cuda";

    std::string msg;
    msg.reserve(192 + filmKey.size() + printKey.size());
    msg = "FATAL: SpektrafilmPixelPipelineNotImplementedForPhase1A at JuicerEffect::render";
    msg += " backend=";
    msg += backend;
    msg += " filmProfileKey=";
    msg += filmKey;
    msg += " printProfileKey=";
    msg += printKey;
    JTRACE("SPEKTRAFILM", msg);
    throw OFX::Exception::Suite(kOfxStatErrFatal);
}

void JuicerEffect::render(const OFX::RenderArguments& args) {
    const Spektrafilm::ScanRoute requestedRoute = Spektrafilm::scan_route_from_key_or(
        read_str_choice_param_or(_pScanRoute, Spektrafilm::scan_route_key(Spektrafilm::kDefaultScanRoute)),
        Spektrafilm::kDefaultScanRoute);
    const bool cudaRoute = args.isEnabledCudaRender;
    const bool directCudaRoute = cudaRoute && !Spektrafilm::scan_route_is_print(requestedRoute);
    if (!cudaRoute) {
        throw_spektrafilm_phase1a_render_cutoff(args);
    }
#if JUICER_DIAGNOSTICS_COMPILED
    if (JTRACE_ENABLED(1)) {
        std::string msg = "plugin=com.juicer.Juicer render backend=cuda route=";
        msg += Spektrafilm::scan_route_key(requestedRoute);
        JTRACE("PHASE3D", msg);
    }
#endif

    auto framePreparation = JuicerProcess::root().begin_frame_preparation();
    if (!framePreparation.active()) {
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    // Fetch images via wrappers
    std::unique_ptr<OFX::Image> srcImg(_src ? _src->fetchImage(args.time) : nullptr);
    std::unique_ptr<OFX::Image> dstImg(_dst ? _dst->fetchImage(args.time) : nullptr);
    if (!srcImg || !dstImg)
        return;

#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
    // CUDA-only mode: reject CPU/OpenCL/Metal renders with an explicit failure.
    if (!args.isEnabledCudaRender) {
        JTRACE("CUDA", "FATAL: JUICER_CUDA_ONLY rejected non-CUDA render request");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
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
            JTRACE("CUDA", "FATAL: CUDA render requested with unsupported non-float copy path");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
    }
#else
    if (args.isEnabledCudaRender) {
        JTRACE("CUDA", "FATAL: CUDA render requested but the CUDA backend is unavailable in this build");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
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
    if (width <= 0 || height <= 0)
        return;
    const int fullWidth = fullBounds.x2 - fullBounds.x1;
    const int fullHeight = fullBounds.y2 - fullBounds.y1;
    const bool fullFrame = (roi.x1 == fullBounds.x1 && roi.y1 == fullBounds.y1 &&
                            roi.x2 == fullBounds.x2 && roi.y2 == fullBounds.y2);
    if (!fullFrame && !directCudaRoute) {
        JTRACE("RENDER", "FATAL: render window must match full frame; tiles/ROIs are unsupported");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    const double longEdgePx = static_cast<double>(std::max(fullWidth, fullHeight));

    // Ensure bootstrap has run before we rely on parameter state
    if (bootstrap_needed(_state.get())) {
        if (!bootstrap_in_progress(_state.get())) {
            const ScopedParamEventSuppression guard(_state.get());
            bootstrap_after_attach();
        }
    }

    // Coalesce parameter-driven WorkingState rebuilds on the render thread to keep UI callbacks fast.
    if (has_loaded_base_state(_state.get())) {
        rebuild_pending_state_if_needed(*this, *_state);
    }
    const bool printRoute = Spektrafilm::scan_route_is_print(requestedRoute);
    const std::shared_ptr<const DirectRenderState> directState =
        printRoute ? nullptr : load_active_direct_state_if(_state.get());
    const std::shared_ptr<const PrintRenderState> printState =
        printRoute ? load_active_print_state_if(_state.get()) : nullptr;
    if ((!printRoute &&
         (!directState || directState->buildCounter == 0 || !directState->recipe.directStructuralReady)) ||
        (printRoute &&
         (!printState || printState->buildCounter == 0 || !printState->recipe.printStructuralReady))) {
        trace_and_throw_render_fatal(RenderFatalTrace{"BUILD", "FATAL: focused render state not ready; aborting render"});
    }
    const RenderRecipe& focusedRecipe = printRoute ? printState->recipe : directState->recipe;
    if (printRoute && focusedRecipe.scannerOutput.glareActive) {
        trace_and_throw_render_fatal(
            RenderFatalTrace{"SPEKTRAFILM", Spektrafilm::kGlareNotImplementedForPhase4});
    }
    if (printRoute &&
        focusedRecipe.scannerOutput.postEffectsDisposition !=
            Spektrafilm::ScannerPostEffectDisposition::Identity) {
        trace_and_throw_render_fatal(
            RenderFatalTrace{"SPEKTRAFILM", Spektrafilm::kScannerPostEffectsNotImplementedForPhase4});
    }
    const double filmFormatMm = focusedRecipe.filmRaw.filmFormatLongEdgeMm;
    const float pixelSizeUm =
        (filmFormatMm > 0.0 && longEdgePx > 0.0)
            ? static_cast<float>((filmFormatMm * 1000.0) / longEdgePx)
            : 0.0f;
    if (traceVerbose) {
        std::string msg = printRoute ? "render print state build=" : "render direct state build=";
        msg += std::to_string(printRoute ? printState->buildCounter : directState->buildCounter);
        msg += " recipe_hash=";
        msg += std::to_string(focusedRecipe.hash);
        JTRACE_VERBOSE("PHASE3D", msg);
    }

    // Tile-based multithreaded processing via OFX::ImageProcessor
    JuicerProcessor proc(*this);
    JuicerProcessor::SourceDestinationImages images{};
    images.src = srcImg.get();
    images.dst = dstImg.get();
    proc.setSrcDst(images);
    proc.setInstanceState(_state.get());
    const SessionTokenSnapshot sessionTokens = snapshot_session_tokens(_state.get());
    const std::uintptr_t renderClipToken = reinterpret_cast<std::uintptr_t>(_src);

    if (printRoute) {
        JuicerProcessor::PrintFrameRequest frameRequest{};
        frameRequest.state = printState;
        frameRequest.components = nComponents;
        frameRequest.renderWindow = roi;
        frameRequest.fullFrameExtent = fullBounds;
        frameRequest.sessionSeed = sessionTokens.sessionSeed;
        frameRequest.instanceToken = sessionTokens.instanceToken;
        frameRequest.clipToken = renderClipToken;
        frameRequest.frameTime = args.time;
        frameRequest.frameRate = getFrameRate();
        frameRequest.pixelSizeUm = pixelSizeUm;
        proc.setPrintFrameRequest(frameRequest);
    } else {
        JuicerProcessor::DirectFrameRequest frameRequest{};
        frameRequest.state = directState;
        frameRequest.components = nComponents;
        frameRequest.renderWindow = roi;
        frameRequest.fullFrameExtent = fullBounds;
        frameRequest.sessionSeed = sessionTokens.sessionSeed;
        frameRequest.instanceToken = sessionTokens.instanceToken;
        frameRequest.clipToken = renderClipToken;
        frameRequest.frameTime = args.time;
        frameRequest.frameRate = getFrameRate();
        frameRequest.pixelSizeUm = pixelSizeUm;
        proc.setDirectFrameRequest(frameRequest);
    }
    proc.setGPURenderArgs(args);

    // Dispatch to support library's threaded/tiled CPU path
    proc.process();
}

void JuicerEffect::changedParam(const OFX::InstanceChangedArgs& args, const std::string& paramName) {
    const bool traceInfo = JTRACE_ENABLED(1);

    // Suppress recursion while we are programmatically setting params
    if (changed_param_suppressed(_state.get())) {
        trace_changed_param_gate(traceInfo, paramName, "changedParam suppressed for '");
        return;
    }
    if (changed_param_bootstrap_blocked(_state.get())) {
        trace_changed_param_gate(traceInfo, paramName, "changedParam ignored during bootstrap for '");
        return;
    }
    invalidate_auto_exposure_cache_if_needed(_state.get(), paramName);

    auto apply_action_then_rebuild = [&](const auto& applyFn) {
        applyFn();
        onParamsPossiblyChanged(paramName.c_str());
    };

    const bool userEdit = (args.reason == OFX::eChangeUserEdit);

    if (param_name_is(paramName, JuicerParams::kFilmProfileKey)) {
        const std::string filmKey =
            read_str_choice_param_or(_pFilmProfileKey, Spektrafilm::kDefaultFilmProfileKey);
        const Spektrafilm::ProfilePolarity polarity = capture_profile_polarity_for_key(filmKey);
        const Spektrafilm::ScanRoute defaultRoute = Spektrafilm::default_scan_route_for_polarity(polarity);
        const ScopedParamEventSuppression suppressEvents(_state.get());
        set_str_choice_param_if(_pScanRoute, Spektrafilm::scan_route_key(defaultRoute));
    }

    auto should_apply_halation_revert_defaults = [&]() {
        return halation_revert_param_changed(paramName);
    };

    auto should_apply_grain_preset_defaults = [&]() {
        return grain_preset_param_changed_by_user(userEdit, paramName);
    };

    auto should_apply_grain_reset_advanced_defaults = [&]() {
        return grain_reset_advanced_param_changed_by_user(userEdit, paramName);
    };

    auto resolve_grain_preset_index_for_user_edit = [&]() {
        return read_choice_param_clamped(_pGrainPreset, 1, 0, 2);
    };

    auto apply_grain_preset_defaults_then_rebuild = [&](int presetIndex) {
        apply_action_then_rebuild([&]() {
            applyGrainPresetDefaults(presetIndex);
        });
    };

    auto apply_grain_reset_advanced_then_rebuild = [&]() {
        apply_action_then_rebuild([&]() {
            resetGrainAdvancedControls();
        });
    };

    auto process_immediate_rebuild_actions = [&]() -> bool {
        if (should_apply_halation_revert_defaults()) {
            apply_action_then_rebuild([&]() {
                applyHalationProfileDefaults();
            });
            return true;
        }
        if (should_apply_grain_preset_defaults()) {
            const int presetIndex = resolve_grain_preset_index_for_user_edit();
            apply_grain_preset_defaults_then_rebuild(presetIndex);
            return true;
        }
        if (should_apply_grain_reset_advanced_defaults()) {
            apply_grain_reset_advanced_then_rebuild();
            return true;
        }
        return false;
    };

    if (process_immediate_rebuild_actions()) {
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

    auto sanitize_triplet_values = [&](std::array<double, 3>& values, double fallback, double lo, double hi) {
        for (double& value : values) {
            value = sanitize_finite_clamped(value, fallback, lo, hi);
        }
    };

    auto set_triplet_suppressed = [&](OFX::Double3DParam* advParam, const std::array<double, 3>& values) {
        if (!advParam || !_state) {
            return;
        }
        const ScopedParamEventSuppression suppressEvents(_state.get());
        set_double3_param_if(advParam, values[0], values[1], values[2]);
    };

    auto should_apply_master_delta = [&](OFX::DoubleParam* masterParam, OFX::Double3DParam* advParam) {
        return has_master_triplet_params(masterParam, advParam);
    };

    auto should_apply_ratio_master = [&](OFX::DoubleParam* masterParam, OFX::Double3DParam* advParam) {
        return has_master_triplet_params(masterParam, advParam);
    };

    auto read_master_value_for_update = [&](OFX::DoubleParam* masterParam, double lo, double hi, double& master) {
        return read_master_value(masterParam, lo, hi, master);
    };

    auto sanitize_triplet_values_for_master =
        [&](std::array<double, 3>& values, double fallback, double lo, double hi) {
            sanitize_triplet_values(values, fallback, lo, hi);
        };

    auto compute_triplet_mean = [&](const std::array<double, 3>& values) {
        return (values[0] + values[1] + values[2]) / 3.0;
    };

    auto write_triplet_from_ratio =
        [&](std::array<double, 3>& values, double master, const std::array<double, 3>& ratio, double lo, double hi) {
            double* valueIt = values.data();
            const double* ratioIt = ratio.data();
            for (int i = 0; i < 3; ++i, ++valueIt, ++ratioIt) {
                *valueIt = std::clamp(master * *ratioIt, lo, hi);
            }
        };

    auto update_master_cache = [&](double& masterCache, double master) {
        masterCache = master;
    };

    auto apply_master_delta = [&](OFX::DoubleParam* masterParam,
                                  OFX::Double3DParam* advParam,
                                  double& masterCache,
                                  double lo,
                                  double hi) {
        if (!should_apply_master_delta(masterParam, advParam)) {
            return;
        }
        double master = 0.0;
        if (!read_master_value_for_update(masterParam, lo, hi, master)) {
            return;
        }
        double prev = masterCache;
        if (!is_finite(prev)) {
            prev = master;
        }
        const double delta = master - prev;
        std::array<double, 3> values{{0.0, 0.0, 0.0}};
        advParam->getValue(values[0], values[1], values[2]);
        sanitize_triplet_values_for_master(values, master, lo, hi);
        if (delta != 0.0) {
            for (double& value : values) {
                value = std::clamp(value + delta, lo, hi);
            }
            set_triplet_suppressed(advParam, values);
        }
        update_master_cache(masterCache, master);
    };

    auto apply_ratio_master = [&](OFX::DoubleParam* masterParam,
                                  OFX::Double3DParam* advParam,
                                  const std::array<double, 3>& fallbackRatio,
                                  double& masterCache,
                                  double lo,
                                  double hi) {
        if (!should_apply_ratio_master(masterParam, advParam)) {
            return;
        }
        double master = 0.0;
        if (!read_master_value_for_update(masterParam, lo, hi, master)) {
            return;
        }

        std::array<double, 3> values{{0.0, 0.0, 0.0}};
        advParam->getValue(values[0], values[1], values[2]);
        sanitize_triplet_values_for_master(values, master, lo, hi);

        std::array<double, 3> ratio = fallbackRatio;
        const double mean = compute_triplet_mean(values);
        if (is_positive_finite(mean)) {
            double* ratioIt = ratio.data();
            const double* valueIt = values.data();
            for (int i = 0; i < 3; ++i, ++ratioIt, ++valueIt) {
                *ratioIt = *valueIt / mean;
            }
        }
        write_triplet_from_ratio(values, master, ratio, lo, hi);
        set_triplet_suppressed(advParam, values);
        update_master_cache(masterCache, master);
    };

    struct MasterTripletUpdateBinding {
        OFX::DoubleParam* masterParam = nullptr;
        OFX::Double3DParam* tripletParam = nullptr;
        double* masterCache = nullptr;
        double lo = 0.0;
        double hi = 0.0;
    };

    auto master_binding_ready = [](const MasterTripletUpdateBinding& binding) {
        return binding.masterParam && binding.tripletParam && binding.masterCache;
    };

    auto resolve_halation_master_update = [&]() -> MasterTripletUpdateBinding {
        switch (halation_master_selector(paramName)) {
            case HalationMasterSelector::Strength:
                return {_pHalationStrengthMaster, _pHalationStrength, &_halationStrengthMasterLast, 0.0, 100.0};
            case HalationMasterSelector::SizeUm:
                return {_pHalationSizeUmMaster, _pHalationSizeUm, &_halationSizeUmMasterLast, 0.0, 1000.0};
            case HalationMasterSelector::ScatteringStrength:
                return {_pHalationScatteringStrengthMaster, _pHalationScatteringStrength, &_halationScatteringStrengthMasterLast, 0.0, 100.0};
            case HalationMasterSelector::ScatteringSizeUm:
                return {_pHalationScatteringSizeUmMaster, _pHalationScatteringSizeUm, &_halationScatteringSizeUmMasterLast, 0.0, 1000.0};
            case HalationMasterSelector::None:
            default:
                return {};
        }
    };

    struct RatioMasterUpdateBinding {
        OFX::DoubleParam* masterParam = nullptr;
        OFX::Double3DParam* tripletParam = nullptr;
        const std::array<double, 3>* fallbackRatio = nullptr;
        double* masterCache = nullptr;
        double lo = 0.0;
        double hi = 0.0;
    };

    auto ratio_binding_ready = [](const RatioMasterUpdateBinding& binding) {
        return binding.masterParam &&
               binding.tripletParam &&
               binding.masterCache &&
               binding.fallbackRatio;
    };

    auto resolve_grain_ratio_update =
        [&](GrainRatioMasterSelector selector, const GrainRatioSet& ratios) -> RatioMasterUpdateBinding {
        switch (selector) {
            case GrainRatioMasterSelector::Scale:
                return {_pGrainParticleScaleMaster, _pGrainParticleScale, &ratios.scale, &_grainParticleScaleMasterLast, 0.0, 10.0};
            case GrainRatioMasterSelector::ScaleLayers:
                return {_pGrainParticleScaleLayersMaster, _pGrainParticleScaleLayers, &ratios.scaleLayers, &_grainParticleScaleLayersMasterLast, 0.0, 10.0};
            case GrainRatioMasterSelector::DensityMin:
                return {_pGrainDensityMinMaster, _pGrainDensityMin, &ratios.densityMin, &_grainDensityMinMasterLast, 0.0, 1.0};
            case GrainRatioMasterSelector::Uniformity:
                return {_pGrainUniformityMaster, _pGrainUniformity, &ratios.uniformity, &_grainUniformityMasterLast, 0.0, 1.0};
            case GrainRatioMasterSelector::None:
            default:
                return {};
        }
    };

    auto try_read_grain_user_edit_unit =
        [&](const char* expectedParam, OFX::DoubleParam* sourceParam, double fallback, double& valueOut) -> bool {
        if (!grain_user_edit_active(userEdit, paramName, expectedParam, _state.get(), sourceParam)) {
            return false;
        }
        return read_finite_unit_interval(sourceParam, fallback, valueOut);
    };

    auto with_param_event_suppression = [&](const auto& applyFn) {
        if (!_state) {
            return;
        }
        const ScopedParamEventSuppression suppressEvents(_state.get());
        applyFn();
    };

    struct GrainTextureLinkedValues {
        double sizeMixWeight = 0.0;
        double microCell = 0.0;
        double microSigma = 0.0;
    };

    auto compute_grain_texture_linked_values = [&](double texture) -> GrainTextureLinkedValues {
        GrainTextureLinkedValues values{};
        values.sizeMixWeight = std::clamp(grain_lerp(0.072, 0.38, texture), 0.0, 1.0);
        values.microCell = std::clamp(grain_lerp(50.0, 70.0, texture), 0.0, 1000.0);
        values.microSigma = std::clamp(grain_lerp(140.0, 200.0, texture), 0.0, 1000.0);
        return values;
    };

    auto apply_grain_linked_unit_edit =
        [&](const char* expectedParam, OFX::DoubleParam* sourceParam, double fallback, const auto& applyFn) {
            double value = fallback;
            if (!try_read_grain_user_edit_unit(expectedParam, sourceParam, fallback, value)) {
                return;
            }
            applyFn(value);
        };

    auto has_grain_blur_dye_target = [&]() {
        return _pGrainBlurDyeCloudsUm != nullptr;
    };

    auto has_grain_texture_link_targets = [&]() {
        return _pGrainSizeMixWeight && _pGrainMicroStructure;
    };

    auto should_apply_grain_sharpness_linked_update = [&]() {
        return has_grain_blur_dye_target();
    };

    auto should_apply_grain_texture_linked_update = [&]() {
        return has_grain_texture_link_targets();
    };

    auto write_grain_blur_dye_clouds = [&](double blurDyeClouds) {
        with_param_event_suppression([&]() {
            set_double_param_if(_pGrainBlurDyeCloudsUm, blurDyeClouds);
        });
    };

    auto write_grain_texture_linked_values = [&](const GrainTextureLinkedValues& linked) {
        with_param_event_suppression([&]() {
            set_double_param_if(_pGrainSizeMixWeight, linked.sizeMixWeight);
            set_double2_param_if(_pGrainMicroStructure, linked.microCell, linked.microSigma);
        });
    };

    auto should_mark_grain_preset_custom = [&]() {
        return userEdit && is_grain_preset_input_param(paramName);
    };

    auto should_process_grain_preset_custom_label = [&]() {
        return should_mark_grain_preset_custom();
    };

    auto should_try_grain_ratio_master_update = [&](const MasterTripletUpdateBinding& binding) {
        return userEdit && !master_binding_ready(binding);
    };

    auto resolve_grain_ratio_master_for_fallback = [&]() {
        return grain_ratio_master_selector(paramName);
    };

    auto apply_ratio_master_if_selected = [&](GrainRatioMasterSelector grainMaster) {
        if (grainMaster == GrainRatioMasterSelector::None) {
            return;
        }
        const GrainRatioSet ratios = normalized_default_grain_ratios();
        const RatioMasterUpdateBinding ratioBinding = resolve_grain_ratio_update(grainMaster, ratios);
        if (ratio_binding_ready(ratioBinding)) {
            apply_ratio_master(
                ratioBinding.masterParam,
                ratioBinding.tripletParam,
                *ratioBinding.fallbackRatio,
                *ratioBinding.masterCache,
                ratioBinding.lo,
                ratioBinding.hi);
        }
    };

    auto apply_grain_sharpness_linked_update = [&]() {
        if (!should_apply_grain_sharpness_linked_update()) {
            return;
        }
        apply_grain_linked_unit_edit(
            JuicerParams::kGrainSharpness,
            _pGrainSharpness,
            0.5,
            [&](double sharpness) {
                const double blurDyeClouds = std::clamp(grain_lerp(1.40, 0.60, sharpness), 0.0, 10.0);
                write_grain_blur_dye_clouds(blurDyeClouds);
            });
    };

    auto apply_grain_texture_linked_update = [&]() {
        if (!should_apply_grain_texture_linked_update()) {
            return;
        }
        apply_grain_linked_unit_edit(
            JuicerParams::kGrainTexture,
            _pGrainTexture,
            0.55,
            [&](double texture) {
                const GrainTextureLinkedValues linked = compute_grain_texture_linked_values(texture);
                write_grain_texture_linked_values(linked);
            });
    };

    auto should_apply_any_grain_linked_updates = [&]() {
        return should_apply_grain_sharpness_linked_update() || should_apply_grain_texture_linked_update();
    };

    auto apply_grain_linked_updates = [&]() {
        if (!should_apply_any_grain_linked_updates()) {
            return;
        }
        apply_grain_sharpness_linked_update();
        apply_grain_texture_linked_update();
    };

    auto should_apply_halation_master_delta = [&](const MasterTripletUpdateBinding& binding) {
        return master_binding_ready(binding);
    };

    auto should_apply_ratio_master_fallback = [&](const MasterTripletUpdateBinding& binding) {
        return should_try_grain_ratio_master_update(binding);
    };

    auto notify_param_change_rebuild = [&]() {
        onParamsPossiblyChanged(paramName.c_str());
    };

    auto apply_master_triplet_or_ratio_fallback = [&](const MasterTripletUpdateBinding& binding) {
        if (should_apply_halation_master_delta(binding)) {
            apply_master_delta(
                binding.masterParam,
                binding.tripletParam,
                *binding.masterCache,
                binding.lo,
                binding.hi);
            return;
        }
        if (should_apply_ratio_master_fallback(binding)) {
            const GrainRatioMasterSelector grainMaster = resolve_grain_ratio_master_for_fallback();
            apply_ratio_master_if_selected(grainMaster);
        }
    };

    auto finalize_changed_param_update = [&](const MasterTripletUpdateBinding& binding) {
        apply_master_triplet_or_ratio_fallback(binding);
        apply_grain_linked_updates();
        notify_param_change_rebuild();
    };

    if (should_process_grain_preset_custom_label()) {
        updateGrainPresetLabel(true);
    }

    updateGrainChromaEnabled();

    const MasterTripletUpdateBinding halationBinding = resolve_halation_master_update();
    finalize_changed_param_update(halationBinding);
}

ParamSnapshot JuicerEffect::snapshotParams() const {
    ParamSnapshot P;
    read_profile_snapshot_choices(
        _pFilmProfileKey,
        _pPrintProfileKey,
        _pScanRoute,
        _pSpectralMode,
        _pRefIll,
        _pEnlIll,
        _pEnlDichroicSet,
        P);
    read_print_recipe_snapshot_values(
        _pPrintExposure,
        _pPrintPreflash,
        _pPrintExposureComp,
        _pEnlargerY,
        _pEnlargerM,
        _pEnlargerC,
        P);
    const JuicerAssets::SelectedProfileResult selectedProfiles =
        JuicerProcess::root().assets().selected_profiles_for_route(
            JuicerAssets::SelectedProfileRequest{
                P.filmProfileKey,
                P.printProfileKey,
                P.scanRoute});
    P.filmProfileAssetVersionToken =
        selectedProfiles.filmProfile ? selectedProfiles.filmProfile->assetVersionToken : 0;
    P.printProfileAssetVersionToken =
        selectedProfiles.printProfile ? selectedProfiles.printProfile->assetVersionToken : 0;
    P.directRoutePrintProfileExcluded = selectedProfiles.directRoutePrintProfileExcluded;
    P.directRouteNeutralCalibrationExcluded =
        selectedProfiles.directRouteNeutralCalibrationExcluded;

    const GlareCompensationSnapshotValues compensation = read_glare_compensation_snapshot_values(
        _pGlareCompRemovalFactor,
        _pGlareCompRemovalDensity,
        _pGlareCompRemovalTransition,
        P.glareCompRemovalFactor,
        P.glareCompRemovalDensity,
        P.glareCompRemovalTransition);
    P.glareCompRemovalFactor = compensation.factor;
    P.glareCompRemovalDensity = compensation.density;
    P.glareCompRemovalTransition = compensation.transition;
    const Profiles::ProfileGlare glare = gatherGlareUi();
    P.glareActive = glare.active;
    P.glarePercent = glare.percent;
    P.glareRoughness = glare.roughness;
    P.glareBlurSigmaPx = glare.blur;
    P.printDminFactor =
        read_sanitized_unit_double(_pPrintDminFactor, P.printDminFactor);
    read_input_snapshot_values(_pInputColorSpace, _pInputCctfDecoding, P);
    const ExposureParams exposure = gatherExposureParams();
    P.cameraAutoExposureEnabled = exposure.cameraAutoEnabled ? 1 : 0;
    P.cameraMeteringMethod = exposure.meteringMethod;
    P.cameraExposureCompensationEv = exposure.sliderEV;
    P.cameraFilmFormatLongEdgeMm = read_camera_film_format_mm_or_default(_pCameraFilmFormat);
    P.exactScatterHalationActive = read_bool_param_or(_pHalationActive, false) ? 1 : 0;
    read_coupler_snapshot_values(_pCouplersActive, _pCouplersAmount, P);
    read_scanner_snapshot_values(
        _pScannerLensBlur,
        _pScannerUnsharp,
        _pScannerUseLut,
        _pScannerLutResolution,
        P);
    read_output_snapshot_values(
        _pOutputColorSpace,
        _pOutputCctfEncoding,
        _pOutputLinearPassThrough,
        P);
    return P;
}

void JuicerEffect::bootstrap_after_attach() {
    // Initialize Spectral globals exactly once per process.
    JuicerProcess::root().ensure_bootstrap();
    JTRACE("BUILD", "spectral globals ensured once; proceeding to profile and film stock load");
    const ScopedBootstrapState bootstrapState(_state.get());
    ParamSnapshot P = snapshotParams();
    const bool printRoute = Spektrafilm::scan_route_is_print(P.scanRoute);
    store_pending_hashes_for_snapshot(*_state, P);

    // Both accepted routes consume the selected validated Spektrafilm film payload.
    load_direct_film_profile_into_base_locked(P, *_state);
    if (has_loaded_base_state(_state.get())) {
        applyHalationProfileDefaults();
    }

    if (printRoute) {
        JTRACE_VERBOSE(
            "SPEKTRAFILM",
            "phase=4C print bootstrap excludes legacy Print::Runtime, dichroic reload, and neutral-filter bootstrap");
    } else {
        JTRACE_VERBOSE(
            "SPEKTRAFILM",
            "phase=3A direct route excludes print profile, dichroic, and neutral-calibration bootstrap");
    }

    if (has_loaded_base_state(_state.get())) {
        store_pending_hashes_for_snapshot(*_state, P);
        if (printRoute) {
            (void)rebuild_print_render_state(*_state, P);
        } else {
            rebuild_working_state(this->getHandle(), *_state, P);
        }
    } else {
        JTRACE("STOCK", "bootstrap: failed to load film stock; deferring rebuild");
    }
}

#if 0
// Phase 4C hard block: accepted print bootstrap/publication owns neutral policy in RenderRecipe.
void JuicerEffect::applyNeutralFilters(const ParamSnapshot& P, Print::Runtime& runtime) {
    if (!_state) {
        return;
    }
#if JUICER_DIAGNOSTICS_COMPILED
    const bool traceInfo = JTRACE_ENABLED(1);
#endif

    const JuicerAssets::PrintRuntimeAssetSet printAssets =
        JuicerProcess::root().assets().print_runtime_assets_for_profile_keys(
            JuicerAssets::PrintRuntimeProfileKeyChoices{
                P.filmProfileKey,
                P.printProfileKey,
                P.enlDichroicSet});
    const char* paperKey = printAssets.printPaper.jsonKey.empty()
                               ? nullptr
                               : printAssets.printPaper.jsonKey.c_str();
    const char* negativeKey = printAssets.filmStock.jsonKey.empty()
                                  ? nullptr
                                  : printAssets.filmStock.jsonKey.c_str();
    const std::vector<std::string> illumKeys = enlarger_illuminant_keys_for_choice(P.enlIll);

    if (!(paperKey && negativeKey && !illumKeys.empty())) {
#if JUICER_DIAGNOSTICS_COMPILED
        if (traceInfo) {
            JTRACE(
                "PRINT",
                "Neutral filter lookup prerequisites missing: " + neutral_filter_prereq_context(paperKey, negativeKey, join_keys_csv_or_none(illumKeys)));
        }
#endif
        throw std::runtime_error("Neutral filter metadata incomplete for current selection");
    }

    float neutralY = Print::kDefaultNeutralY;
    float neutralM = Print::kDefaultNeutralM;
    float neutralC = Print::kDefaultNeutralC;
    bool loaded = false;

    const JuicerAssets::NeutralFilterDatabaseAsset& neutralDb = printAssets.neutralFilters;
    std::tuple<float, float, float> ymc{};
    std::string selectedDbVersionHash;
    const std::string* illumKeyData = illumKeys.data();
    const size_t illumKeyCount = illumKeys.size();
    for (size_t i = 0; i < illumKeyCount; ++i, ++illumKeyData) {
        const std::string& illumKey = *illumKeyData;
        if (illumKey.empty()) {
            continue;
        }
        const JuicerAssets::NeutralFilterLookupResult lookup =
            JuicerProcess::root().assets().lookup_neutral_filters(
                neutralDb,
                JuicerAssets::NeutralFilterLookupKey{
                    printAssets.printPaper.jsonKey,
                    illumKey,
                    printAssets.filmStock.jsonKey},
                JuicerAssets::NeutralFilterLookupThread::Control);
        if (lookup.found) {
            ymc = lookup.ymc;
            selectedDbVersionHash = lookup.selectedDbVersionHash;
            neutralY = std::clamp(std::get<0>(ymc), 0.0f, 1.0f);
            neutralM = std::clamp(std::get<1>(ymc), 0.0f, 1.0f);
            neutralC = std::clamp(std::get<2>(ymc), 0.0f, 1.0f);
            loaded = true;
#if JUICER_DIAGNOSTICS_COMPILED
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
#endif
            break;
        }
    }

    if (!loaded) {
#if JUICER_DIAGNOSTICS_COMPILED
        if (traceInfo) {
            JTRACE(
                "PRINT",
                "Neutral filters missing for " + neutral_filter_missing_context(paperKey, negativeKey, join_keys_csv_or_none(illumKeys)) + "; aborting print path");
        }
#endif
        throw std::runtime_error("Neutral filter database entry not found");
    }

    std::uint64_t neutralFilterHash = Print::kDefaultNeutralFilterHash;
    if (!selectedDbVersionHash.empty()) {
        neutralFilterHash = Hash::hash_bytes(selectedDbVersionHash.data(), selectedDbVersionHash.size());
        if (neutralFilterHash == 0) {
            neutralFilterHash = Print::kDefaultNeutralFilterHash;
        }
    }

    runtime.neutralY = neutralY;
    runtime.neutralM = neutralM;
    runtime.neutralC = neutralC;
    runtime.neutralFilterHash = neutralFilterHash;
    // Preserve user-entered enlarger offsets and exposure toggle; neutral baselines update independently.
}

bool JuicerEffect::applyMetadataIlluminantDefaults(ParamSnapshot& P, const Print::Runtime& runtime) {
    if (!_state) {
        return false;
    }

    bool changed = false;

    const std::string& filmRef = !_state->filmReferenceIlluminant.empty()
                                     ? _state->filmReferenceIlluminant
                                     : _state->base.referenceIlluminant;
    const std::string& printRef = runtime.referenceIlluminant;
    const std::string& printView = runtime.viewingIlluminant;

    const std::string refSource = first_nonempty_or(filmRef, printRef, printView);
    const std::string enlSource = first_nonempty_or(printRef, filmRef, printView);

    const ScopedParamEventSuppression suppressEvents(_state.get());
    changed = apply_illuminant_choice_from_source(
                  _pRefIll,
                  P.refIll,
                  _state->illuminantOverride.reference,
                  refSource) ||
              changed;
    changed = apply_illuminant_choice_from_source(
                  _pEnlIll,
                  P.enlIll,
                  _state->illuminantOverride.enlarger,
                  enlSource) ||
              changed;

    if (changed) {
        P = snapshotParams();
    }

    return changed;
}
#endif

// SF_PHASE5_BLOCKED_LiveOfxDirProfileFollow owner=Phase5;
// allowed_call_sites=none; output_hash_resource_impact=none;
// cleanup_symbol=SF_PHASE5_BLOCKED_LiveOfxDirProfileFollow; disposition=delete_with_legacy_renderer.
#if 0
void JuicerEffect::initializeCouplerParamsFromProfileIfNeeded(ParamSnapshot& P) {
    if (!_state) {
        return;
    }

    const int initVersion = read_int_param_or(_pCouplersInitVersion, 0);
    if (initVersion >= kDirCouplersInitVersionCurrent) {
        return;
    }

    const ScopedParamEventSuppression suppressEvents(_state.get());

    const Profiles::DirCouplersProfile& dirCfg = _state->base.dirCouplers;
    CouplerProfileDefaults factoryDefaults{};
    factoryDefaults.active = (kFactoryCouplersActive != 0);
    factoryDefaults.amount = kFactoryCouplersAmount;
    factoryDefaults.ratioB = kFactoryCouplersRatioB;
    factoryDefaults.ratioG = kFactoryCouplersRatioG;
    factoryDefaults.ratioR = kFactoryCouplersRatioR;
    factoryDefaults.sigma = kFactoryCouplersSigma;
    factoryDefaults.high = kFactoryCouplersHigh;
    factoryDefaults.spatialSigmaMicrometers = kFactoryCouplersSpatialSigma;

    if (!dirCfg.hasData) {
        set_int_param_if(_pCouplersFollowMask, infer_coupler_follow_stock_mask(P, factoryDefaults));
        set_int_param_if(_pCouplersInitVersion, kDirCouplersInitVersionCurrent);
        return;
    }

    const CouplerProfileDefaults profileDefaults = build_coupler_profile_defaults(
        dirCfg,
        _state->couplerProfileSpatialSigmaValid,
        _state->couplerProfileSpatialSigmaMicrometers,
        P);

    const int profileFollowMask = infer_coupler_follow_stock_mask(P, profileDefaults);
    const bool matchesProfileDefaults = profileFollowMask == kCouplerFollowStockAllMask;
    const bool matchesFactoryDefaults =
        infer_coupler_follow_stock_mask(P, factoryDefaults) == kCouplerFollowStockAllMask;

    int followMask = profileFollowMask;
    // Legacy instances without the init-version param can only be distinguished by their
    // visible values: untouched factory defaults get the stock-profile initialization once,
    // while any other restored values are preserved as authored state.
    if (!matchesProfileDefaults && matchesFactoryDefaults) {
        applyCouplerProfileDefaults(P);
        followMask = kCouplerFollowStockAllMask;
    }

    set_int_param_if(_pCouplersFollowMask, sanitize_coupler_follow_stock_mask(followMask));
    set_int_param_if(_pCouplersInitVersion, kDirCouplersInitVersionCurrent);
}

void JuicerEffect::syncCouplerParamsFromProfileFollowMask(ParamSnapshot& P) {
    if (!_state) {
        return;
    }

    const Profiles::DirCouplersProfile& dirCfg = _state->base.dirCouplers;
    if (!dirCfg.hasData) {
        return;
    }

    const int followMask = sanitize_coupler_follow_stock_mask(
        read_int_param_or(_pCouplersFollowMask, kCouplerFollowStockAllMask));
    if (followMask == 0) {
        return;
    }

    const ScopedParamEventSuppression suppressEvents(_state.get());
    const CouplerProfileDefaults profileDefaults = build_coupler_profile_defaults(
        dirCfg,
        _state->couplerProfileSpatialSigmaValid,
        _state->couplerProfileSpatialSigmaMicrometers,
        P);

    auto apply_if_following = [&](CouplerParamKind kind,
                                  double source,
                                  double fallback,
                                  double lo,
                                  double hi,
                                  OFX::DoubleParam* param,
                                  double& target) {
        if (!coupler_follow_stock_enabled(followMask, kind)) {
            return;
        }
        const double value = sanitize_finite_clamped(source, fallback, lo, hi);
        set_double_param_if(param, value);
        target = value;
    };

    if (coupler_follow_stock_enabled(followMask, CouplerParamKind::Active)) {
        set_bool_param_if(_pCouplersActive, profileDefaults.active);
        P.couplersActive = bool_to_i32(profileDefaults.active);
    }

    apply_if_following(
        CouplerParamKind::Amount,
        profileDefaults.amount,
        P.couplersAmount,
        0.0,
        2.0,
        _pCouplersAmount,
        P.couplersAmount);
    apply_if_following(
        CouplerParamKind::RatioB,
        profileDefaults.ratioB,
        P.ratioB,
        0.0,
        1.0,
        _pCouplersAmountB,
        P.ratioB);
    apply_if_following(
        CouplerParamKind::RatioG,
        profileDefaults.ratioG,
        P.ratioG,
        0.0,
        1.0,
        _pCouplersAmountG,
        P.ratioG);
    apply_if_following(
        CouplerParamKind::RatioR,
        profileDefaults.ratioR,
        P.ratioR,
        0.0,
        1.0,
        _pCouplersAmountR,
        P.ratioR);
    apply_if_following(
        CouplerParamKind::Sigma,
        profileDefaults.sigma,
        P.sigma,
        0.0,
        4.0,
        _pCouplersSigma,
        P.sigma);
    apply_if_following(
        CouplerParamKind::High,
        profileDefaults.high,
        P.high,
        0.0,
        1.0,
        _pCouplersHigh,
        P.high);
    apply_if_following(
        CouplerParamKind::SpatialSigma,
        profileDefaults.spatialSigmaMicrometers,
        P.spatialSigmaMicrometers,
        0.0,
        50.0,
        _pCouplersSpatialSigma,
        P.spatialSigmaMicrometers);
}

void JuicerEffect::clearCouplerFollowStockForParam(const char* changedNameOrNull) {
    if (!_state || !_pCouplersFollowMask) {
        return;
    }

    const CouplerParamKind kind = coupler_param_kind(changedNameOrNull);
    const int bit = coupler_follow_stock_bit(kind);
    if (bit == 0) {
        return;
    }

    const int followMask = sanitize_coupler_follow_stock_mask(read_int_param_or(_pCouplersFollowMask, 0));
    if ((followMask & bit) == 0) {
        return;
    }

    const ScopedParamEventSuppression suppressEvents(_state.get());
    set_int_param_if(_pCouplersFollowMask, followMask & ~bit);
}

void JuicerEffect::applyCouplerProfileDefaults(ParamSnapshot& P) {
    if (!_state) {
        return;
    }

    const Profiles::DirCouplersProfile& dirCfg = _state->base.dirCouplers;
    if (!dirCfg.hasData) {
        return;
    }

    const ScopedParamEventSuppression suppressEvents(_state.get());

    auto apply_profile_double = [&](double source,
                                    double fallback,
                                    double lo,
                                    double hi,
                                    OFX::DoubleParam* param,
                                    double& target) {
        const double value = sanitize_finite_clamped(source, fallback, lo, hi);
        set_double_param_if(param, value);
        target = value;
    };

    const bool active = dirCfg.active;
    set_bool_param_if(_pCouplersActive, active);
    P.couplersActive = bool_to_i32(active);

    apply_profile_double(
        static_cast<double>(dirCfg.amount),
        P.couplersAmount,
        0.0,
        2.0,
        _pCouplersAmount,
        P.couplersAmount);

    apply_profile_double(
        static_cast<double>(dirCfg.ratioRGB[0]),
        P.ratioB,
        0.0,
        1.0,
        _pCouplersAmountB,
        P.ratioB);

    apply_profile_double(
        static_cast<double>(dirCfg.ratioRGB[1]),
        P.ratioG,
        0.0,
        1.0,
        _pCouplersAmountG,
        P.ratioG);

    apply_profile_double(
        static_cast<double>(dirCfg.ratioRGB[2]),
        P.ratioR,
        0.0,
        1.0,
        _pCouplersAmountR,
        P.ratioR);

    apply_profile_double(
        static_cast<double>(dirCfg.diffusionInterlayer),
        P.sigma,
        0.0,
        4.0,
        _pCouplersSigma,
        P.sigma);

    apply_profile_double(
        static_cast<double>(dirCfg.highExposureShift),
        P.high,
        0.0,
        1.0,
        _pCouplersHigh,
        P.high);

    const float profileSpatialSigma = _state->couplerProfileSpatialSigmaValid
                                          ? static_cast<float>(_state->couplerProfileSpatialSigmaMicrometers)
                                          : dirCfg.diffusionSizeUm;
    apply_profile_double(
        static_cast<double>(profileSpatialSigma),
        P.spatialSigmaMicrometers,
        0.0,
        50.0,
        _pCouplersSpatialSigma,
        P.spatialSigmaMicrometers);
}
#endif

void JuicerEffect::onParamsPossiblyChanged(const char* changedNameOrNull) {
    if (!_state)
        return;
    const bool traceVerbose = JTRACE_ENABLED(3);
    // Suppress re-entrant param handling while programmatic changes are in flight
    if (should_skip_param_change_due_to_suppression(_state.get())) {
        JTRACE("BUILD", "onParamsPossiblyChanged suppressed");
        return;
    }

    // If bootstrap hasn’t run yet, run it once now
    if (should_bootstrap_for_param_change(_state.get())) {
        bootstrap_after_attach();
    }

    ParamSnapshot P = snapshotParams();
    const ChangedParamFlags changed = classify_changed_param(changedNameOrNull);
    trace_param_change_verbose_if(traceVerbose, P, *_state, changedNameOrNull);

    // Track user overrides for illuminant choices.
    mark_illuminant_override_if_changed(*_state, changed);

    const bool filmReloaded =
        reload_film_stock_if_requested(changed.filmProfile, P, *_state);
    apply_when_film_reloaded(filmReloaded, [&]() {
        applyHalationProfileDefaults();
    });

    if (Spektrafilm::scan_route_is_print(P.scanRoute)) {
        JTRACE_VERBOSE(
            "SPEKTRAFILM",
            "phase=4C parameter change queued focused print recipe publication without legacy Print::Runtime");
    }

    store_pending_hashes_for_snapshot(*_state, P);
}
