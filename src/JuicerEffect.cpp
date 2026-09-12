#include "JuicerEffect.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <limits>
#include <utility>
#include <vector>

#include "JuicerState.h"
#include "ColorTransforms.h"
#include "DiffusionHostBehavior.h"
#include "Illuminants.h"
#include "ParamNames.h"
#include "ProcessRoot.h"
#include "ScatterHalation.h"
#include "SpectralData.h"
#include "SpectralProcessing.h"
#include "Logging.h"
#include "Hash.h"
#include "mainProcessing.h"

namespace {
    inline const char* cstr_or_default_if_null(const char* value, const char* fallback);

    struct RenderFatalTrace {
        const char* tag = nullptr;
        const char* message = nullptr;
    };

    [[noreturn]] inline void trace_and_throw_render_fatal(const RenderFatalTrace& fatal) {
        JTRACE(fatal.tag, cstr_or_default_if_null(fatal.message, "fatal render error"));
        throw OFX::Exception::Suite(kOfxStatErrFatal);
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

    inline std::shared_ptr<const DirectRenderState> load_active_direct_state_if(const InstanceState* state) {
        return state ? JuicerAtomic::load_shared_ptr(&state->activeDirectState) : nullptr;
    }

    inline std::shared_ptr<const PrintRenderState> load_active_print_state_if(const InstanceState* state) {
        return state ? JuicerAtomic::load_shared_ptr(&state->activePrintState) : nullptr;
    }

    struct ProfileKeyLabels {
        const char* printProfileKey = nullptr;
        const char* filmKey = nullptr;
        const char* paperLabel = "<null>";
        const char* filmLabel = "<null>";
    };

    inline ProfileKeyLabels resolve_profile_key_labels(const ParamSnapshot& snapshot) {
        ProfileKeyLabels labels{};
        labels.printProfileKey = snapshot.printProfileKey.empty() ? nullptr : snapshot.printProfileKey.c_str();
        labels.filmKey = snapshot.filmProfileKey.empty() ? nullptr : snapshot.filmProfileKey.c_str();
        labels.paperLabel = cstr_or_default_if_null(labels.printProfileKey, "<null>");
        labels.filmLabel = cstr_or_default_if_null(labels.filmKey, "<null>");
        return labels;
    }

    inline bool pending_snapshot_acquisition_needed(InstanceState& state) {
        std::lock_guard<std::mutex> lock(state.pending.m);
        return std::holds_alternative<PendingParamsState::Uninitialized>(
            state.pending.value);
    }

    inline void store_pending_valid_snapshot(
        InstanceState& state,
        const ParamSnapshot& snapshot) {
        PendingParamsState::Valid valid;
        valid.params = snapshot;
        valid.fullHash = hash_params(snapshot);
        std::lock_guard<std::mutex> lock(state.pending.m);
        state.pending.value = std::move(valid);
    }

    inline void store_pending_invalid_snapshot(
        InstanceState& state,
        std::string diagnostic) {
        PendingParamsState::InvalidSnapshotControls invalid;
        invalid.diagnostic = std::move(diagnostic);
        std::lock_guard<std::mutex> lock(state.pending.m);
        state.pending.value = std::move(invalid);
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

    enum class GrainRatioMasterSelector : std::uint8_t {
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

    inline bool is_coupler_gamma_numeric_param_name(const std::string& paramName) {
        return param_name_is(paramName, JuicerParams::kDirCouplersGammaSameLayerRgb) ||
               param_name_is(paramName, JuicerParams::kDirCouplersGammaInterlayerRToGb) ||
               param_name_is(paramName, JuicerParams::kDirCouplersGammaInterlayerGToRb) ||
               param_name_is(paramName, JuicerParams::kDirCouplersGammaInterlayerBToRg);
    }

    inline bool changed_param_suppressed(const InstanceState* state) {
        return param_events_suppressed(state);
    }

    inline bool grain_preset_param_changed_by_user(bool userEdit, const std::string& paramName) {
        return user_edit_param_is(userEdit, paramName, JuicerParams::kGrainPreset);
    }

    inline bool grain_reset_advanced_param_changed_by_user(bool userEdit, const std::string& paramName) {
        return user_edit_param_is(userEdit, paramName, JuicerParams::kGrainResetAdvanced);
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
        std::uint64_t activeBuild = 0;
        if (Spektrafilm::scan_route_is_print(snapshot.scanRoute)) {
            const std::shared_ptr<const PrintRenderState> active = load_active_print_state_if(&state);
            activeBuild = active ? active->buildCounter : 0;
        } else {
            const std::shared_ptr<const DirectRenderState> active = load_active_direct_state_if(&state);
            activeBuild = active ? active->buildCounter : 0;
        }
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

    inline bool read_camera_film_format_mm(
        OFX::DoubleParam* param,
        float& out,
        std::string& outDiagnostic) {
        if (!param) {
            out = 35.0f;
            return true;
        }

        double raw = 35.0;
        param->getValue(raw);
        const char* reason = nullptr;
        if (!is_finite(raw)) {
            reason = "non_finite";
        } else if (raw < 8.0) {
            reason = "below_minimum";
        } else if (raw > 120.0) {
            reason = "above_maximum";
        }
        if (reason) {
            outDiagnostic =
                "InvalidAuthoredControl component=camera field=film_format_mm reason=";
            outDiagnostic += reason;
            return false;
        }
        out = static_cast<float>(raw);
        return true;
    }

    inline bool is_positive_finite(double value) {
        return is_finite(value) && value > 0.0;
    }

    struct SanitizedDoubleRange {
        double minimum = 0.0;
        double maximum = 0.0;
    };

    inline double read_sanitized_double(
        OFX::DoubleParam* param,
        double fallback,
        const SanitizedDoubleRange& range) {
        double value = fallback;
        if (param) {
            param->getValue(value);
        }
        if (!is_finite(value)) {
            return fallback;
        }
        return std::clamp(value, range.minimum, range.maximum);
    }

    inline float read_sanitized_float(
        OFX::DoubleParam* param,
        float fallback,
        double minValue,
        double maxValue) {
        return static_cast<float>(read_sanitized_double(
            param,
            static_cast<double>(fallback),
            SanitizedDoubleRange{minValue, maxValue}));
    }

    inline double read_sanitized_unit_double(OFX::DoubleParam* param, double fallback) {
        return read_sanitized_double(param, fallback, SanitizedDoubleRange{0.0, 1.0});
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

    struct GlareCompensationParams {
        OFX::DoubleParam* factor = nullptr;
        OFX::DoubleParam* density = nullptr;
        OFX::DoubleParam* transition = nullptr;
    };

    struct GlareCompensationSnapshotValues {
        double factor = 0.0;
        double density = 1.2;
        double transition = 0.3;
    };

    inline GlareCompensationSnapshotValues read_glare_compensation_snapshot_values(
        const GlareCompensationParams& params,
        const GlareCompensationSnapshotValues& fallback) {
        GlareCompensationSnapshotValues values{};
        values.factor = read_sanitized_unit_double(params.factor, fallback.factor);
        values.density = read_sanitized_double(
            params.density,
            fallback.density,
            SanitizedDoubleRange{0.0, 3.0});
        values.transition = read_sanitized_double(
            params.transition,
            fallback.transition,
            SanitizedDoubleRange{0.0, 2.0});
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
        valueOut = std::clamp(valueOut, 0.0, 1.0);
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

    struct GrainChromaWeights {
        float shared = 0.0f;
        float independent = 0.0f;
    };

    inline GrainChromaWeights compute_grain_chroma_weights(float chroma) {
        GrainChromaWeights weights{};
        weights.shared = std::sqrt(std::max(0.0f, 1.0f - chroma));
        weights.independent = std::sqrt(std::max(0.0f, chroma));
        return weights;
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

    inline Spektrafilm::DiffusionFilterFamily read_diffusion_family(
        OFX::ChoiceParam* param) {
        switch (read_choice_param_clamped(param, 1, 0, 3)) {
            case 0:
                return Spektrafilm::DiffusionFilterFamily::Glimmerglass;
            case 2:
                return Spektrafilm::DiffusionFilterFamily::ProMist;
            case 3:
                return Spektrafilm::DiffusionFilterFamily::Cinebloom;
            case 1:
            default:
                return Spektrafilm::DiffusionFilterFamily::BlackProMist;
        }
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
            if (!is_finite(*valueIt)) {
                *valueIt = *defaultIt;
            } else {
                *valueIt = std::clamp(*valueIt, minValue, maxValue);
            }
        }
        return values;
    }

    inline std::array<double, 3> read_sanitized_grain_scale_triplet(
        OFX::Double3DParam* param,
        double masterValue) {
        std::array<double, 3> defaults{};
        defaults.fill(masterValue);
        return read_sanitized_double3(param, defaults, 0.0, 10.0);
    }

    inline std::array<double, 3> read_sanitized_grain_unit_triplet(
        OFX::Double3DParam* param,
        double masterValue) {
        std::array<double, 3> defaults{};
        defaults.fill(masterValue);
        return read_sanitized_double3(param, defaults, 0.0, 1.0);
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
            if (!is_finite(*valueIt)) {
                *valueIt = *defaultIt;
            } else {
                *valueIt = std::clamp(*valueIt, minValue, maxValue);
            }
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
        Spektrafilm::VisualGrainControls& grain,
        const std::array<double, 3>& particleScale,
        const std::array<double, 3>& particleScaleLayers,
        const std::array<double, 3>& densityMin,
        const std::array<double, 3>& uniformity) {
        cast_array(grain.particleScaleCmy, particleScale);
        cast_array(grain.particleScaleLayers, particleScaleLayers);
        cast_array(grain.visualParticleDensityMinCmy, densityMin);
        cast_array(grain.uniformityCmy, uniformity);
    }

    inline void read_scanner_snapshot_values(
        OFX::DoubleParam* scannerLensBlurParam,
        OFX::Double2DParam* scannerUnsharpParam,
        OFX::BooleanParam* scannerBlackCorrectionParam,
        OFX::BooleanParam* scannerWhiteCorrectionParam,
        OFX::DoubleParam* scannerBlackLevelParam,
        OFX::DoubleParam* scannerWhiteLevelParam,
        OFX::BooleanParam* scannerUseLutParam,
        OFX::IntParam* scannerLutResolutionParam,
        ParamSnapshot& snapshot) {
        snapshot.scannerLensBlurSigmaPx = read_double_param_or(
            scannerLensBlurParam,
            snapshot.scannerLensBlurSigmaPx);
        snapshot.scannerUnsharpMask = read_double2_param_or(
            scannerUnsharpParam,
            snapshot.scannerUnsharpMask);
        snapshot.scannerBlackCorrection =
            read_bool_param_as_i32(scannerBlackCorrectionParam, false);
        snapshot.scannerWhiteCorrection =
            read_bool_param_as_i32(scannerWhiteCorrectionParam, false);
        snapshot.scannerBlackLevel =
            read_sanitized_unit_double(scannerBlackLevelParam, snapshot.scannerBlackLevel);
        snapshot.scannerWhiteLevel =
            read_sanitized_unit_double(scannerWhiteLevelParam, snapshot.scannerWhiteLevel);
        snapshot.scannerUseLut = read_bool_param_as_i32(scannerUseLutParam, true);
        snapshot.scannerLutResolution = read_int_param_or(
            scannerLutResolutionParam,
            snapshot.scannerLutResolution);
    }

    inline void read_output_snapshot_values(
        OFX::ChoiceParam* outputColorSpaceParam,
        OFX::BooleanParam* outputCctfEncodingParam,
        ParamSnapshot& snapshot) {
        snapshot.outputColorSpace = read_choice_param_or(outputColorSpaceParam, snapshot.outputColorSpace);
        snapshot.outputCctfEncoding = read_bool_param_as_i32(outputCctfEncodingParam, true);
    }

    struct ProfileSnapshotChoiceParams {
        OFX::StrChoiceParam* filmProfileKey = nullptr;
        OFX::StrChoiceParam* printProfileKey = nullptr;
        OFX::StrChoiceParam* scanRoute = nullptr;
        OFX::ChoiceParam* spectralMode = nullptr;
        OFX::ChoiceParam* referenceIlluminant = nullptr;
        OFX::ChoiceParam* enlargerIlluminant = nullptr;
    };

    inline void read_profile_snapshot_choices(
        const ProfileSnapshotChoiceParams& params,
        ParamSnapshot& snapshot) {
        snapshot.filmProfileKey = read_str_choice_param_or(params.filmProfileKey, snapshot.filmProfileKey);
        snapshot.printProfileKey = read_str_choice_param_or(params.printProfileKey, snapshot.printProfileKey);
        snapshot.scanRoute = read_resolved_scan_route(params.scanRoute, snapshot.filmProfileKey);
        snapshot.spectralUpsamplingMode = read_choice_param_or(params.spectralMode, snapshot.spectralUpsamplingMode);
        snapshot.refIll = read_choice_param_or(params.referenceIlluminant, snapshot.refIll);
        snapshot.enlIll = read_choice_param_or(params.enlargerIlluminant, snapshot.enlIll);
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
        OFX::BooleanParam* hanatos2025AdaptationWindowParam,
        OFX::BooleanParam* hanatos2025AdaptationSurfaceParam,
        ParamSnapshot& snapshot) {
        snapshot.inputColorSpace = read_choice_param_or(inputColorSpaceParam, snapshot.inputColorSpace);
        snapshot.inputCctfDecoding = read_bool_param_as_i32(inputCctfDecodingParam, false);
        snapshot.hanatos2025AdaptationWindow =
            read_bool_param_as_i32(hanatos2025AdaptationWindowParam, true);
        snapshot.hanatos2025AdaptationSurface =
            read_bool_param_as_i32(hanatos2025AdaptationSurfaceParam, false);
    }

    inline void read_coupler_snapshot_values(
        OFX::BooleanParam* couplersActiveParam,
        OFX::DoubleParam* couplersAmountParam,
        OFX::DoubleParam* inhibitionSameLayerParam,
        OFX::DoubleParam* inhibitionInterlayerParam,
        OFX::DoubleParam* diffusionSizeUmParam,
        OFX::BooleanParam* gammaUseStockParam,
        OFX::Double3DParam* gammaSameLayerRgbParam,
        OFX::Double2DParam* gammaInterlayerRToGbParam,
        OFX::Double2DParam* gammaInterlayerGToRbParam,
        OFX::Double2DParam* gammaInterlayerBToRgParam,
        ParamSnapshot& snapshot) {
        snapshot.couplersActive = read_bool_param_as_i32(couplersActiveParam, true);
        snapshot.couplersAmount = read_double_param_or(couplersAmountParam, snapshot.couplersAmount);
        snapshot.couplersInhibitionSameLayer =
            read_double_param_or(inhibitionSameLayerParam, snapshot.couplersInhibitionSameLayer);
        snapshot.couplersInhibitionInterlayer =
            read_double_param_or(inhibitionInterlayerParam, snapshot.couplersInhibitionInterlayer);
        snapshot.couplersDiffusionSizeUm =
            read_double_param_or(diffusionSizeUmParam, snapshot.couplersDiffusionSizeUm);
        snapshot.couplersGammaUseStock = read_bool_param_as_i32(gammaUseStockParam, true);
        snapshot.couplersGammaSameLayerRgb =
            read_double3_param_or(gammaSameLayerRgbParam, snapshot.couplersGammaSameLayerRgb);
        snapshot.couplersGammaInterlayerRToGb =
            read_double2_param_or(gammaInterlayerRToGbParam, snapshot.couplersGammaInterlayerRToGb);
        snapshot.couplersGammaInterlayerGToRb =
            read_double2_param_or(gammaInterlayerGToRbParam, snapshot.couplersGammaInterlayerGToRb);
        snapshot.couplersGammaInterlayerBToRg =
            read_double2_param_or(gammaInterlayerBToRgParam, snapshot.couplersGammaInterlayerBToRg);
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

    inline float finite_exp2_scale(double ev) {
        const float scale = static_cast<float>(std::exp2(ev));
        return is_finite(scale) ? scale : 1.0f;
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

Spektrafilm::VisualGrainControls JuicerEffect::gatherGrainUi() const {
    Spektrafilm::VisualGrainControls grain{};

    grain.active = read_bool_param_or(_pGrainActive, false);

    const int presetIndex = read_choice_param_clamped(_pGrainPreset, 1, 0, 2);
    const GrainPresetDefaults preset = grain_preset_defaults(presetIndex);

    grain.sublayersActive = read_bool_param_or(_pGrainSublayersActive, preset.sublayersActive);

    const double amountEV = read_sanitized_double(
        _pGrainAmplitude,
        preset.amountEV,
        SanitizedDoubleRange{-3.0, 3.0});
    const double amplitude = std::exp2(amountEV);
    grain.amplitude = static_cast<float>(amplitude);

    grain.correlationSigmaPx = read_sanitized_float(
        _pGrainBlur,
        static_cast<float>(preset.sizePx),
        0.20,
        2.00);

    const double sharpness = read_sanitized_unit_double(_pGrainSharpness, preset.sharpness);

    const double chroma = read_sanitized_unit_double(_pGrainChroma, preset.chroma);

    const double texture = read_sanitized_unit_double(_pGrainTexture, preset.texture);

    const GrainAdvancedDefaults advancedDefaults = compute_grain_advanced_defaults(
        preset,
        sharpness,
        texture);

    grain.particleAreaUm2 = read_sanitized_0_to_10_float(
        _pGrainParticleAreaUm2,
        static_cast<float>(preset.particleAreaUm2));

    grain.sizeMixScale = read_sanitized_float(
        _pGrainSizeMixScale,
        static_cast<float>(preset.sizeMixScale),
        1.0,
        50.0);

    grain.coarseWeight = read_sanitized_unit_float(
        _pGrainSizeMixWeight,
        static_cast<float>(advancedDefaults.sizeMixWeight));

    grain.midWeight = read_sanitized_unit_float(_pGrainSizeMixWeightMid, 0.0f);

    grain.dyeCloudBlurUm = read_sanitized_0_to_10_float(
        _pGrainBlurDyeCloudsUm,
        static_cast<float>(advancedDefaults.blurDyeClouds));

    grain.chromaMix = static_cast<float>(chroma);
    const GrainChromaWeights chromaWeights = compute_grain_chroma_weights(grain.chromaMix);
    grain.chromaSharedWeight = chromaWeights.shared;
    grain.chromaIndependentWeight = chromaWeights.independent;
    const std::array<double, 3> particleScale =
        read_sanitized_grain_scale_triplet(_pGrainParticleScale, preset.particleScaleMaster);
    const std::array<double, 3> particleScaleLayers =
        read_sanitized_grain_scale_triplet(_pGrainParticleScaleLayers, preset.particleScaleLayersMaster);
    const std::array<double, 3> densityMin =
        read_sanitized_grain_unit_triplet(_pGrainDensityMin, preset.densityMinMaster);
    const std::array<double, 3> uniformity =
        read_sanitized_grain_unit_triplet(_pGrainUniformity, preset.uniformityMaster);
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

    grain.debugView = read_choice_param_clamped(_pGrainDebugView, 0, 0, 6);

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

Spektrafilm::DiffusionFilterAuthoredControls JuicerEffect::gatherDiffusionUi(
    const DiffusionUiParams& params) const {
    Spektrafilm::DiffusionFilterAuthoredControls controls{};
    controls.active = read_bool_param_or(params.enabled, controls.active);
    controls.family = read_diffusion_family(params.family);
    controls.strength = read_double_param_or(params.strength, controls.strength);
    controls.spatialScale = read_double_param_or(params.spatialScale, controls.spatialScale);
    controls.haloWarmth = read_double_param_or(params.haloWarmth, controls.haloWarmth);
    controls.coreIntensity = read_double_param_or(params.coreIntensity, controls.coreIntensity);
    controls.coreSize = read_double_param_or(params.coreSize, controls.coreSize);
    controls.haloIntensity = read_double_param_or(params.haloIntensity, controls.haloIntensity);
    controls.haloSize = read_double_param_or(params.haloSize, controls.haloSize);
    controls.bloomIntensity = read_double_param_or(params.bloomIntensity, controls.bloomIntensity);
    controls.bloomSize = read_double_param_or(params.bloomSize, controls.bloomSize);
    return controls;
}

void JuicerEffect::updateDiffusionControlState() {
    const bool cameraEnabled = read_bool_param_or(_cameraDiffusionUi.enabled, false);
    const std::string filmProfileKey =
        read_str_choice_param_or(_pFilmProfileKey, Spektrafilm::kDefaultFilmProfileKey);
    const Spektrafilm::ScanRoute selectedRoute =
        read_resolved_scan_route(_pScanRoute, filmProfileKey);
    const bool printRoute = Spektrafilm::scan_route_is_print(selectedRoute);
    const bool printEnabled = read_bool_param_or(_printDiffusionUi.enabled, false);

    auto setControlState = [](auto* param, bool visible, bool enabled) {
        if (!param) {
            return;
        }
        param->setIsSecret(!visible);
        param->setEnabled(enabled);
    };
    auto setStageControlState = [&](const DiffusionUiParams& params,
                                    bool visible,
                                    bool enabled) {
        setControlState(params.enabled, visible, visible);
        setControlState(params.family, visible, visible && enabled);
        setControlState(params.strength, visible, visible && enabled);
        setControlState(params.spatialScale, visible, visible && enabled);
        setControlState(params.haloWarmth, visible, visible && enabled);
        setControlState(params.coreIntensity, visible, visible && enabled);
        setControlState(params.coreSize, visible, visible && enabled);
        setControlState(params.haloIntensity, visible, visible && enabled);
        setControlState(params.haloSize, visible, visible && enabled);
        setControlState(params.bloomIntensity, visible, visible && enabled);
        setControlState(params.bloomSize, visible, visible && enabled);
    };

    setStageControlState(_cameraDiffusionUi, true, cameraEnabled);
    setStageControlState(_printDiffusionUi, printRoute, printEnabled);
}

void JuicerEffect::updateGammaControlState() {
    const std::string filmProfileKey =
        read_str_choice_param_or(
            _pFilmProfileKey,
            Spektrafilm::kDefaultFilmProfileKey);
    const Spektrafilm::ScanRoute selectedRoute =
        read_resolved_scan_route(_pScanRoute, filmProfileKey);
    if (_pPrintGammaFactor) {
        _pPrintGammaFactor->setEnabled(
            Spektrafilm::scan_route_is_print(selectedRoute));
    }
}

void JuicerEffect::applyDirGammaProfileDefaults() {
    if (!_state || !read_bool_param_or(_pCouplersGammaUseStock, true)) {
        return;
    }

    const std::string filmKey =
        read_str_choice_param_or(_pFilmProfileKey, Spektrafilm::kDefaultFilmProfileKey);
    const std::shared_ptr<const Profiles::ValidatedFilmProfile> profile =
        JuicerProcess::root().assets().selected_film_profile_for_key(filmKey);
    if (!profile) {
        JTRACE("SPEKTRAFILM", "MissingRequiredResource field=dir_gamma_profile_defaults");
        return;
    }

    const Profiles::ProfileDigest& digest = profile->digest;
    const auto finiteNonnegative = [](float value) {
        return std::isfinite(value) && value >= 0.0f;
    };
    const bool valid =
        std::all_of(digest.gammaSamelayerRgb.begin(), digest.gammaSamelayerRgb.end(), finiteNonnegative) &&
        std::all_of(digest.gammaInterlayerRToGb.begin(), digest.gammaInterlayerRToGb.end(), finiteNonnegative) &&
        std::all_of(digest.gammaInterlayerGToRb.begin(), digest.gammaInterlayerGToRb.end(), finiteNonnegative) &&
        std::all_of(digest.gammaInterlayerBToRg.begin(), digest.gammaInterlayerBToRg.end(), finiteNonnegative);
    if (!valid) {
        JTRACE("SPEKTRAFILM", "MalformedRequiredProfileData field=dir_gamma_profile_defaults");
        return;
    }

    const ScopedParamEventSuppression suppressEvents(_state.get());
    set_double3_param_if(
        _pCouplersGammaSameLayerRgb,
        digest.gammaSamelayerRgb[0],
        digest.gammaSamelayerRgb[1],
        digest.gammaSamelayerRgb[2]);
    set_double2_param_if(
        _pCouplersGammaInterlayerRToGb,
        digest.gammaInterlayerRToGb[0],
        digest.gammaInterlayerRToGb[1]);
    set_double2_param_if(
        _pCouplersGammaInterlayerGToRb,
        digest.gammaInterlayerGToRb[0],
        digest.gammaInterlayerGToRb[1]);
    set_double2_param_if(
        _pCouplersGammaInterlayerBToRg,
        digest.gammaInterlayerBToRg[0],
        digest.gammaInterlayerBToRg[1]);
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

    const auto fetchOptionalBooleanParam = [this](const char* name) {
        try {
            return fetchBooleanParam(name);
        } catch (...) {
            JuicerLogging::discard_current_exception();
            return static_cast<OFX::BooleanParam*>(nullptr);
        }
    };
    const auto fetchOptionalDoubleParam = [this](const char* name) {
        try {
            return fetchDoubleParam(name);
        } catch (...) {
            JuicerLogging::discard_current_exception();
            return static_cast<OFX::DoubleParam*>(nullptr);
        }
    };

    // Cache parameter handles (wrappers)
    try {
        _pExposure = fetchDoubleParam(kParamExposure);
        _pCameraAutoExposure = fetchBooleanParam(kParamCameraAutoExposure);
        _pCameraFilmFormatPreset =
            fetchChoiceParam(JuicerParams::kCameraFilmFormatPreset);
        _pCameraFilmFormat = fetchDoubleParam(JuicerParams::kCameraFilmFormatMm);
        _pCameraMeteringMethod = fetchChoiceParam(JuicerParams::kCameraMeteringMethod);
        _pFilmProfileKey = fetchStrChoiceParam(JuicerParams::kFilmProfileKey);
        _pSpectralMode = fetchChoiceParam(kParamSpectralMode);
        _pPrintProfileKey = fetchStrChoiceParam(JuicerParams::kPrintProfileKey);
        _pRefIll = fetchChoiceParam("ReferenceIlluminant");
        _pEnlIll = fetchChoiceParam("EnlargerIlluminant");
        _pInputColorSpace = fetchChoiceParam(JuicerParams::kInputColorSpace);
        _pInputCctfDecoding = fetchBooleanParam(JuicerParams::kInputCctfDecoding);
        _pHanatos2025AdaptationWindow =
            fetchBooleanParam(JuicerParams::kHanatos2025AdaptationWindow);
        _pHanatos2025AdaptationSurface =
            fetchBooleanParam(JuicerParams::kHanatos2025AdaptationSurface);
        _pScanRoute = fetchStrChoiceParam(JuicerParams::kParamScanRoute);
        _pFilmGammaFactor = fetchDoubleParam(JuicerParams::kFilmGammaFactor);
        _pPrintGammaFactor = fetchDoubleParam(JuicerParams::kPrintGammaFactor);
        _pOutputColorSpace = fetchChoiceParam(kParamOutputColorSpace);
        _pOutputCctfEncoding = fetchBooleanParam(kParamOutputCctfEncoding);


        _pCouplersActive = fetchBooleanParam(JuicerParams::kDirCouplersActive);
        _pCouplersAmount = fetchDoubleParam(JuicerParams::kDirCouplersAmount);
        _pCouplersInhibitionSameLayer =
            fetchDoubleParam(JuicerParams::kDirCouplersInhibitionSameLayer);
        _pCouplersInhibitionInterlayer =
            fetchDoubleParam(JuicerParams::kDirCouplersInhibitionInterlayer);
        _pCouplersDiffusionSizeUm =
            fetchDoubleParam(JuicerParams::kDirCouplersDiffusionSizeUm);
        _pCouplersGammaUseStock =
            fetchBooleanParam(JuicerParams::kDirCouplersGammaUseStock);
        _pCouplersGammaSameLayerRgb =
            fetchDouble3DParam(JuicerParams::kDirCouplersGammaSameLayerRgb);
        _pCouplersGammaInterlayerRToGb =
            fetchDouble2DParam(JuicerParams::kDirCouplersGammaInterlayerRToGb);
        _pCouplersGammaInterlayerGToRb =
            fetchDouble2DParam(JuicerParams::kDirCouplersGammaInterlayerGToRb);
        _pCouplersGammaInterlayerBToRg =
            fetchDouble2DParam(JuicerParams::kDirCouplersGammaInterlayerBToRg);

        _pScannerLensBlur = fetchDoubleParam(JuicerParams::kScannerLensBlurSigmaPx);
        _pScannerUnsharp = fetchDouble2DParam(JuicerParams::kScannerUnsharpMask);
        _pScannerBlackCorrection =
            fetchBooleanParam(JuicerParams::kScannerBlackCorrection);
        _pScannerWhiteCorrection =
            fetchBooleanParam(JuicerParams::kScannerWhiteCorrection);
        _pScannerBlackLevel = fetchDoubleParam(JuicerParams::kScannerBlackLevel);
        _pScannerWhiteLevel = fetchDoubleParam(JuicerParams::kScannerWhiteLevel);
        _pScannerUseLut = fetchBooleanParam(JuicerParams::kScannerUseLut);
        _pScannerLutResolution = fetchIntParam(JuicerParams::kScannerLutResolution);

        _pPrintExposure = fetchDoubleParam("PrintExposure");
        _pPrintPreflash = fetchDoubleParam("PrintPreflash");
        _pPrintExposureComp = fetchBooleanParam("PrintExposureCompensation");
        _pEnlargerY = fetchDoubleParam("EnlargerY");
        _pEnlargerM = fetchDoubleParam("EnlargerM");
        _pEnlargerC = fetchDoubleParam("EnlargerC");

        _pHalationActive = fetchOptionalBooleanParam(JuicerParams::kHalationActive);
        _pHalationScatterAmount =
            fetchOptionalDoubleParam(JuicerParams::kHalationScatterAmount);
        _pHalationScatterSpatialScale =
            fetchOptionalDoubleParam(JuicerParams::kHalationScatterSpatialScale);
        _pHalationAmount = fetchOptionalDoubleParam(JuicerParams::kHalationAmount);
        _pHalationSpatialScale =
            fetchOptionalDoubleParam(JuicerParams::kHalationSpatialScale);

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
        _pGrainDebugView = fetchChoiceParam(JuicerParams::kGrainDebugView);
        _pGrainMicroStructure = fetchDouble2DParam(JuicerParams::kGrainMicroStructure);
        _pGrainResetAdvanced = fetchPushButtonParam(JuicerParams::kGrainResetAdvanced);
        _pGateWeaveAmount = fetchDoubleParam(JuicerParams::kGateWeaveAmount);
        _pFilmDustAmount = fetchDoubleParam(JuicerParams::kFilmDustAmount);
        _pGateDustAmount = fetchDoubleParam(JuicerParams::kGateDustAmount);
        _pFilmScratchAmount = fetchDoubleParam(JuicerParams::kFilmScratchAmount);
        _pGateScratchAmount = fetchDoubleParam(JuicerParams::kGateScratchAmount);

        _cameraDiffusionUi.enabled =
            fetchBooleanParam(JuicerParams::kCameraDiffusionEnabled);
        _cameraDiffusionUi.family =
            fetchChoiceParam(JuicerParams::kCameraDiffusionFamily);
        _cameraDiffusionUi.strength =
            fetchDoubleParam(JuicerParams::kCameraDiffusionStrength);
        _cameraDiffusionUi.spatialScale =
            fetchDoubleParam(JuicerParams::kCameraDiffusionSpatialScale);
        _cameraDiffusionUi.haloWarmth =
            fetchDoubleParam(JuicerParams::kCameraDiffusionHaloWarmth);
        _cameraDiffusionUi.coreIntensity =
            fetchDoubleParam(JuicerParams::kCameraDiffusionCoreIntensity);
        _cameraDiffusionUi.coreSize =
            fetchDoubleParam(JuicerParams::kCameraDiffusionCoreSize);
        _cameraDiffusionUi.haloIntensity =
            fetchDoubleParam(JuicerParams::kCameraDiffusionHaloIntensity);
        _cameraDiffusionUi.haloSize =
            fetchDoubleParam(JuicerParams::kCameraDiffusionHaloSize);
        _cameraDiffusionUi.bloomIntensity =
            fetchDoubleParam(JuicerParams::kCameraDiffusionBloomIntensity);
        _cameraDiffusionUi.bloomSize =
            fetchDoubleParam(JuicerParams::kCameraDiffusionBloomSize);

        _printDiffusionUi.enabled =
            fetchBooleanParam(JuicerParams::kPrintDiffusionEnabled);
        _printDiffusionUi.family =
            fetchChoiceParam(JuicerParams::kPrintDiffusionFamily);
        _printDiffusionUi.strength =
            fetchDoubleParam(JuicerParams::kPrintDiffusionStrength);
        _printDiffusionUi.spatialScale =
            fetchDoubleParam(JuicerParams::kPrintDiffusionSpatialScale);
        _printDiffusionUi.haloWarmth =
            fetchDoubleParam(JuicerParams::kPrintDiffusionHaloWarmth);
        _printDiffusionUi.coreIntensity =
            fetchDoubleParam(JuicerParams::kPrintDiffusionCoreIntensity);
        _printDiffusionUi.coreSize =
            fetchDoubleParam(JuicerParams::kPrintDiffusionCoreSize);
        _printDiffusionUi.haloIntensity =
            fetchDoubleParam(JuicerParams::kPrintDiffusionHaloIntensity);
        _printDiffusionUi.haloSize =
            fetchDoubleParam(JuicerParams::kPrintDiffusionHaloSize);
        _printDiffusionUi.bloomIntensity =
            fetchDoubleParam(JuicerParams::kPrintDiffusionBloomIntensity);
        _printDiffusionUi.bloomSize =
            fetchDoubleParam(JuicerParams::kPrintDiffusionBloomSize);

        _pGlareActive = fetchBooleanParam(JuicerParams::kGlareActive);
        _pGlarePercent = fetchDoubleParam(JuicerParams::kGlarePercent);
        _pGlareRoughness = fetchDoubleParam(JuicerParams::kGlareRoughness);
        _pGlareBlurSigmaPx = fetchDoubleParam(JuicerParams::kGlareBlurSigmaPx);
        _pGlareCompRemovalFactor = fetchDoubleParam(JuicerParams::kPrintShadowCompensationFactor);
        _pGlareCompRemovalDensity = fetchDoubleParam(JuicerParams::kPrintShadowCompensationDensity);
        _pGlareCompRemovalTransition = fetchDoubleParam(JuicerParams::kPrintShadowCompensationTransition);
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
    initMasterCache(_pGrainParticleScaleMaster, _grainParticleScaleMasterLast);
    initMasterCache(_pGrainParticleScaleLayersMaster, _grainParticleScaleLayersMasterLast);
    initMasterCache(_pGrainDensityMinMaster, _grainDensityMinMasterLast);
    initMasterCache(_pGrainUniformityMaster, _grainUniformityMasterLast);

    // Own per-instance state
    _state = std::make_unique<InstanceState>();
    JuicerAtomic::store_shared_ptr(&_state->activeDirectState, std::shared_ptr<const DirectRenderState>{});
    JuicerAtomic::store_shared_ptr(&_state->activePrintState, std::shared_ptr<const PrintRenderState>{});
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

    updateDiffusionControlState();
    updateGammaControlState();

    // Defer heavy bootstrap until first param change
}

JuicerEffect::~JuicerEffect() {
    try {
        if (_state) {
            JuicerProcess::root().retire_grain_static_instance(
                _state->instanceToken);
        }
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
    const bool cudaRoute = args.isEnabledCudaRender;
    if (!cudaRoute) {
        throw_spektrafilm_phase1a_render_cutoff(args);
    }
    auto framePreparation = JuicerProcess::root().begin_frame_preparation();
    if (!framePreparation.active()) {
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    // Fetch images via wrappers
    std::unique_ptr<OFX::Image> srcImg(_src ? _src->fetchImage(args.time) : nullptr);
    std::unique_ptr<OFX::Image> dstImg(_dst ? _dst->fetchImage(args.time) : nullptr);
    if (!srcImg || !dstImg)
        return;

    // Components
    const OFX::PixelComponentEnum comps = srcImg->getPixelComponents();
    const int nComponents = pixel_component_count(comps);
    const bool traceVerbose = JTRACE_ENABLED(3);

    const OFX::BitDepthEnum depth = srcImg->getPixelDepth();
    if (args.isEnabledCudaRender) {
        // CUDA renders use device pointers; avoid host-side image reads for metering or copy paths.
        if (requires_nonfloat_copy(depth, nComponents)) {
            JTRACE("CUDA", "FATAL: CUDA render requested with unsupported non-float copy path");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
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

    PendingRenderAdmissionResult admission = admit_pending_render_state(*_state);
    if (admission.status == PendingRenderAdmissionStatus::NeedsSnapshotAcquisition) {
        initialize_pending_render_state();
        admission = admit_pending_render_state(*_state);
    }
    switch (admission.status) {
        case PendingRenderAdmissionStatus::NeedsSnapshotAcquisition:
            trace_and_throw_render_fatal(RenderFatalTrace{
                "BUILD",
                "FATAL: focused render snapshot acquisition did not publish state"});
        case PendingRenderAdmissionStatus::InvalidSnapshotControls:
        case PendingRenderAdmissionStatus::RebuildFailed:
            trace_and_throw_render_fatal(RenderFatalTrace{
                "BUILD",
                admission.diagnostic.empty()
                    ? "FATAL: focused render admission failed"
                    : admission.diagnostic.c_str()});
        case PendingRenderAdmissionStatus::AdmittedDirect:
        case PendingRenderAdmissionStatus::AdmittedPrint:
            break;
    }

    const bool printRoute =
        admission.status == PendingRenderAdmissionStatus::AdmittedPrint;
    const bool directCudaRoute = cudaRoute && !printRoute;
    if (!fullFrame && !directCudaRoute) {
        JTRACE("RENDER", "FATAL: render window must match full frame; tiles/ROIs are unsupported");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    const double longEdgePx = static_cast<double>(std::max(fullWidth, fullHeight));
    const std::shared_ptr<const DirectRenderState> directState = admission.directState;
    const std::shared_ptr<const PrintRenderState> printState = admission.printState;
    const RenderRecipe& focusedRecipe = printRoute ? printState->recipe : directState->recipe;
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
        JTRACE_VERBOSE("RENDER", msg);
    }

    std::optional<Spektrafilm::DiffusionFrameSetDescriptor> diffusionFrameSet;
    std::string diffusionDiagnostic;
    if (!Spektrafilm::build_diffusion_frame_set_descriptor(
            focusedRecipe.spatialOptics,
            focusedRecipe.profileRoute.scanRoute,
            static_cast<double>(pixelSizeUm),
            Spektrafilm::DiffusionFrameDomain{
                fullBounds.x1,
                fullBounds.y1,
                fullWidth,
                fullHeight},
            diffusionFrameSet,
            diffusionDiagnostic)) {
        JTRACE("SPEKTRAFILM", diffusionDiagnostic);
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    std::optional<ScatterHalationFrameDescriptor> scatterHalationDescriptor;
    std::string scatterHalationDiagnostic;
    if (!Spektrafilm::build_scatter_halation_frame_descriptor(
            focusedRecipe.spatialOptics.scatterHalation,
            pixelSizeUm,
            scatterHalationDescriptor,
            scatterHalationDiagnostic)) {
        JTRACE("SPEKTRAFILM", scatterHalationDiagnostic);
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    Spektrafilm::FilmJuicerEffectsGeometry effectsGeometry{};
    if (focusedRecipe.filmJuicerEffects.active) {
        const auto definition = srcImg->getRegionOfDefinition();
        const auto canonical = _src->getRegionOfDefinition(args.time);
        const auto scale = srcImg->getRenderScale();
        effectsGeometry.pixelDefinition = {definition.x1, definition.y1, definition.x2 - definition.x1, definition.y2 - definition.y1};
        effectsGeometry.canonicalX = canonical.x1;
        effectsGeometry.canonicalY = canonical.y1;
        effectsGeometry.canonicalWidth = canonical.x2 - canonical.x1;
        effectsGeometry.canonicalHeight = canonical.y2 - canonical.y1;
        effectsGeometry.scaleX = scale.x;
        effectsGeometry.scaleY = scale.y;
        effectsGeometry.pixelAspectRatio = srcImg->getPixelAspectRatio();
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
        frameRequest.diffusionFrameSet = diffusionFrameSet;
        frameRequest.scatterHalation = scatterHalationDescriptor;
        frameRequest.components = nComponents;
        frameRequest.renderWindow = roi;
        frameRequest.effectsGeometry = effectsGeometry;
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
        frameRequest.diffusionFrameSet = diffusionFrameSet;
        frameRequest.scatterHalation = scatterHalationDescriptor;
        frameRequest.components = nComponents;
        frameRequest.renderWindow = roi;
        frameRequest.effectsGeometry = effectsGeometry;
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

    auto apply_action_then_rebuild = [&](const auto& applyFn) {
        applyFn();
        onParamsPossiblyChanged(paramName.c_str());
    };

    const bool userEdit = (args.reason == OFX::eChangeUserEdit);

    if (userEdit && param_name_is(paramName, JuicerParams::kCameraFilmFormatPreset)) {
        const int presetIndex = read_choice_param_clamped(
            _pCameraFilmFormatPreset,
            0,
            0,
            static_cast<int>(JuicerParams::kCameraFilmFormatPresets.size()));
        if (presetIndex > 0) {
            const ScopedParamEventSuppression suppressEvents(_state.get());
            set_double_param_if(
                _pCameraFilmFormat,
                JuicerParams::kCameraFilmFormatPresets[static_cast<std::size_t>(presetIndex - 1)]
                    .longEdgeMm);
        }
    } else if (userEdit && param_name_is(paramName, JuicerParams::kCameraFilmFormatMm)) {
        const ScopedParamEventSuppression suppressEvents(_state.get());
        if (_pCameraFilmFormatPreset) {
            _pCameraFilmFormatPreset->setValue(0);
        }
    }

    if (userEdit && is_coupler_gamma_numeric_param_name(paramName)) {
        const ScopedParamEventSuppression suppressEvents(_state.get());
        set_bool_param_if(_pCouplersGammaUseStock, false);
    }

    if (param_name_is(paramName, JuicerParams::kFilmProfileKey)) {
        const std::string filmKey =
            read_str_choice_param_or(_pFilmProfileKey, Spektrafilm::kDefaultFilmProfileKey);
        const Spektrafilm::ProfilePolarity polarity = capture_profile_polarity_for_key(filmKey);
        const Spektrafilm::ScanRoute defaultRoute = Spektrafilm::default_scan_route_for_polarity(polarity);
        const ScopedParamEventSuppression suppressEvents(_state.get());
        set_str_choice_param_if(_pScanRoute, Spektrafilm::scan_route_key(defaultRoute));
    }

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
            value = is_finite(value) ? std::clamp(value, lo, hi) : fallback;
        }
    };

    auto set_triplet_suppressed = [&](OFX::Double3DParam* advParam, const std::array<double, 3>& values) {
        if (!advParam || !_state) {
            return;
        }
        const ScopedParamEventSuppression suppressEvents(_state.get());
        set_double3_param_if(advParam, values[0], values[1], values[2]);
    };

    auto should_apply_ratio_master = [&](OFX::DoubleParam* masterParam, OFX::Double3DParam* advParam) {
        return has_master_triplet_params(masterParam, advParam);
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
        if (!read_master_value(masterParam, lo, hi, master)) {
            return;
        }

        std::array<double, 3> values{{0.0, 0.0, 0.0}};
        advParam->getValue(values[0], values[1], values[2]);
        sanitize_triplet_values(values, master, lo, hi);

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
        masterCache = master;
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

    if (should_process_grain_preset_custom_label()) {
        updateGrainPresetLabel(true);
    }

    updateGrainChromaEnabled();

    const GrainRatioMasterSelector grainMaster =
        userEdit ? grain_ratio_master_selector(paramName) : GrainRatioMasterSelector::None;
    if (grainMaster != GrainRatioMasterSelector::None) {
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
    }
    apply_grain_linked_updates();
    updateDiffusionControlState();
    updateGammaControlState();
    onParamsPossiblyChanged(paramName.c_str());
}

bool JuicerEffect::snapshotParams(
    ParamSnapshot& out,
    std::string& outDiagnostic) const {
    ParamSnapshot P;
    outDiagnostic.clear();

    ScatterHalationRawControls rawControls{};
    rawControls.active = read_bool_param_or(_pHalationActive, false);
    if (rawControls.active) {
        rawControls.scatterAmount = read_double_param_or(_pHalationScatterAmount, 1.0);
        rawControls.scatterSpatialScale =
            read_double_param_or(_pHalationScatterSpatialScale, 1.0);
        rawControls.halationAmount = read_double_param_or(_pHalationAmount, 1.0);
        rawControls.halationSpatialScale =
            read_double_param_or(_pHalationSpatialScale, 1.0);
    }
    if (!Spektrafilm::build_scatter_halation_controls(
            rawControls,
            P.scatterHalationControls,
            outDiagnostic)) {
        return false;
    }
    if (!read_camera_film_format_mm(
            _pCameraFilmFormat,
            P.cameraFilmFormatLongEdgeMm,
            outDiagnostic)) {
        return false;
    }

    ProfileSnapshotChoiceParams profileChoiceParams{};
    profileChoiceParams.filmProfileKey = _pFilmProfileKey;
    profileChoiceParams.printProfileKey = _pPrintProfileKey;
    profileChoiceParams.scanRoute = _pScanRoute;
    profileChoiceParams.spectralMode = _pSpectralMode;
    profileChoiceParams.referenceIlluminant = _pRefIll;
    profileChoiceParams.enlargerIlluminant = _pEnlIll;
    read_profile_snapshot_choices(profileChoiceParams, P);
    if (!_pFilmGammaFactor || !_pPrintGammaFactor) {
        outDiagnostic =
            "MissingRequiredParameter component=tuning field=gamma_factor";
        return false;
    }
    double authoredFilmGammaFactor = 1.0;
    double authoredPrintGammaFactor = 1.0;
    _pFilmGammaFactor->getValue(authoredFilmGammaFactor);
    _pPrintGammaFactor->getValue(authoredPrintGammaFactor);
    if (!set_gamma_snapshot_values(
            authoredFilmGammaFactor,
            authoredPrintGammaFactor,
            P,
            outDiagnostic)) {
        return false;
    }
    P.cameraDiffusion = gatherDiffusionUi(_cameraDiffusionUi);
    P.enlargerDiffusion = gatherDiffusionUi(_printDiffusionUi);
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

    GlareCompensationParams compensationParams{};
    compensationParams.factor = _pGlareCompRemovalFactor;
    compensationParams.density = _pGlareCompRemovalDensity;
    compensationParams.transition = _pGlareCompRemovalTransition;
    GlareCompensationSnapshotValues compensationFallback{};
    compensationFallback.factor = P.printShadowCompensationFactor;
    compensationFallback.density = P.printShadowCompensationDensity;
    compensationFallback.transition = P.printShadowCompensationTransition;
    const GlareCompensationSnapshotValues compensation =
        read_glare_compensation_snapshot_values(compensationParams, compensationFallback);
    P.printShadowCompensationFactor = compensation.factor;
    P.printShadowCompensationDensity = compensation.density;
    P.printShadowCompensationTransition = compensation.transition;
    P.glareActive = read_bool_param_or(_pGlareActive, true);
    P.glarePercent = read_sanitized_unit_float(_pGlarePercent, 0.03f);
    P.glareRoughness = read_sanitized_unit_float(_pGlareRoughness, 0.7f);
    P.glareBlurSigmaPx =
        read_sanitized_0_to_10_float(_pGlareBlurSigmaPx, 0.5f);
    read_input_snapshot_values(
        _pInputColorSpace,
        _pInputCctfDecoding,
        _pHanatos2025AdaptationWindow,
        _pHanatos2025AdaptationSurface,
        P);
    const ExposureParams exposure = gatherExposureParams();
    P.cameraAutoExposureEnabled = exposure.cameraAutoEnabled ? 1 : 0;
    P.cameraMeteringMethod = exposure.meteringMethod;
    P.cameraExposureCompensationEv = exposure.sliderEV;
    P.grainControls = gatherGrainUi();
    const GrainSurfaceArtifacts artifacts = read_grain_surface_artifacts(
        _pFilmDustAmount,
        _pGateDustAmount,
        _pFilmScratchAmount,
        _pGateScratchAmount);
    P.filmDustAmount = artifacts.filmDustAmount;
    P.filmScratchAmount = artifacts.filmScratchAmount;
    P.gateDustAmount = artifacts.gateDustAmount;
    P.gateScratchAmount = artifacts.gateScratchAmount;
    P.gateWeaveAmount = read_sanitized_double(
        _pGateWeaveAmount,
        0.0,
        SanitizedDoubleRange{0.0, 10.0});
    read_coupler_snapshot_values(
        _pCouplersActive,
        _pCouplersAmount,
        _pCouplersInhibitionSameLayer,
        _pCouplersInhibitionInterlayer,
        _pCouplersDiffusionSizeUm,
        _pCouplersGammaUseStock,
        _pCouplersGammaSameLayerRgb,
        _pCouplersGammaInterlayerRToGb,
        _pCouplersGammaInterlayerGToRb,
        _pCouplersGammaInterlayerBToRg,
        P);
    read_scanner_snapshot_values(
        _pScannerLensBlur,
        _pScannerUnsharp,
        _pScannerBlackCorrection,
        _pScannerWhiteCorrection,
        _pScannerBlackLevel,
        _pScannerWhiteLevel,
        _pScannerUseLut,
        _pScannerLutResolution,
        P);
    read_output_snapshot_values(
        _pOutputColorSpace,
        _pOutputCctfEncoding,
        P);
    out = std::move(P);
    return true;
}

void JuicerEffect::initialize_pending_render_state() {
    if (!_state) {
        return;
    }
    JuicerProcess::root().ensure_bootstrap();
    JTRACE("BUILD", "spectral globals ensured; publishing initial pending render state");
    applyDirGammaProfileDefaults();
    ParamSnapshot params;
    std::string diagnostic;
    if (snapshotParams(params, diagnostic)) {
        store_pending_valid_snapshot(*_state, params);
    } else {
        store_pending_invalid_snapshot(*_state, std::move(diagnostic));
    }
}

void JuicerEffect::onParamsPossiblyChanged(const char* changedNameOrNull) {
    if (!_state) {
        return;
    }
    if (param_events_suppressed(_state.get())) {
        JTRACE("BUILD", "onParamsPossiblyChanged suppressed");
        return;
    }
    if (pending_snapshot_acquisition_needed(*_state)) {
        initialize_pending_render_state();
        return;
    }

    const bool filmProfileChanged =
        param_name_is(changedNameOrNull, JuicerParams::kFilmProfileKey);
    const bool stockGammaEnabled =
        param_name_is(changedNameOrNull, JuicerParams::kDirCouplersGammaUseStock) &&
        read_bool_param_or(_pCouplersGammaUseStock, true);
    if (filmProfileChanged || stockGammaEnabled) {
        applyDirGammaProfileDefaults();
    }

    ParamSnapshot params;
    std::string diagnostic;
    if (!snapshotParams(params, diagnostic)) {
        store_pending_invalid_snapshot(*_state, std::move(diagnostic));
        return;
    }
    trace_param_change_verbose_if(
        JTRACE_ENABLED(3),
        params,
        *_state,
        changedNameOrNull);
    if (Spektrafilm::scan_route_is_print(params.scanRoute)) {
        JTRACE_VERBOSE(
            "SPEKTRAFILM",
            "parameter change queued focused print recipe publication");
    }
    store_pending_valid_snapshot(*_state, params);
}
