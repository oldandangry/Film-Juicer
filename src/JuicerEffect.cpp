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
#include "PreparedState.h"
#include "RustFrameBridge.h"
#include "ColorTransforms.h"
#include "DiffusionHostBehavior.h"
#include "ParamNames.h"
#include "Cuda/JuicerCudaExecutor.h"
#include "Cuda/JuicerCudaFailure.h"
#include "juicer_cuda_owner.h"
#include "ScatterHalation.h"
#include "SpectralData.h"
#include "Logging.h"
#include "Hash.h"
#include "mainProcessing.h"

#if defined(JUICER_FRAME_GEOMETRY_TEST_HOOK)
namespace JuicerFrame::FrameGeometryTest {
    void instance_retired(const FjInstance*, std::uint64_t, FjStatus) noexcept;
}
#endif

namespace {
    Spektrafilm::DiffusionFrameDomain frame_domain(const OfxRectI& bounds) {
        JuicerFrame::Domain domain{};
        JuicerFrame::Failure failure;
        if (!JuicerFrame::domain_from_bounds({bounds.x1, bounds.y1, bounds.x2, bounds.y2}, domain, failure)) {
#if JUICER_DIAGNOSTICS_COMPILED
            if (failure.reason == JuicerFrame::FailureReason::DomainRange) {
                JTRACE("CUDA", "FATAL: frame descriptor dimensions exceed signed integer range");
            } else {
                JTRACE("CUDA", failure.delivery.diagnostic);
            }
#endif
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        return {domain.x, domain.y, domain.width, domain.height};
    }

#if JUICER_DIAGNOSTICS_COMPILED
    inline const char* cstr_or_default_if_null(const char* value, const char* fallback);
#endif

    struct RenderFatalTrace {
        const char* tag = nullptr;
        const char* message = nullptr;
    };

    [[noreturn]] inline void trace_and_throw_render_fatal(const RenderFatalTrace& fatal) {
#if JUICER_DIAGNOSTICS_COMPILED
        JTRACE(fatal.tag, cstr_or_default_if_null(fatal.message, "fatal render error"));
#else
        (void)fatal;
#endif
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

#if JUICER_DIAGNOSTICS_COMPILED
    inline const char* cstr_or_default_if_null(const char* value, const char* fallback) {
        return value ? value : fallback;
    }

    inline std::size_t cstr_len_or_zero(const char* value) {
        return value ? std::strlen(value) : 0u;
    }
#endif

    inline bool requires_nonfloat_copy(OFX::BitDepthEnum depth, int nComponents) {
        return depth != OFX::eBitDepthFloat || nComponents == 0;
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

    inline bool grain_preset_param_changed_by_user(bool userEdit, const std::string& paramName) {
        return user_edit_param_is(userEdit, paramName, JuicerParams::kGrainPreset);
    }

    inline bool grain_reset_advanced_param_changed_by_user(bool userEdit, const std::string& paramName) {
        return user_edit_param_is(userEdit, paramName, JuicerParams::kGrainResetAdvanced);
    }

    inline double sanitize_finite_or(double value, double fallback) {
        return is_finite(value) ? value : fallback;
    }

    inline double read_double_param(OFX::DoubleParam* param) {
        double value = 0.0;
        param->getValue(value);
        return value;
    }

    inline int read_int_param(OFX::IntParam* param) {
        int value = 0;
        param->getValue(value);
        return value;
    }

    inline int read_bool_param_as_i32(OFX::BooleanParam* param) {
        bool value = false;
        param->getValue(value);
        return bool_to_i32(value);
    }

    inline bool read_camera_film_format_mm(
        OFX::DoubleParam* param,
        float& out,
        std::string& outDiagnostic) {
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
        param->getValue(value);
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
        double& valueOut) {
        valueOut = read_double_param(param);
        if (!is_finite(valueOut)) {
            return false;
        }
        valueOut = std::clamp(valueOut, 0.0, 1.0);
        return true;
    }

    inline bool grain_user_edit_active(
        bool userEdit,
        const std::string& paramName,
        const char* expectedParam) {
        return user_edit_param_is(userEdit, paramName, expectedParam);
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

    inline bool read_bool_param(OFX::BooleanParam* param) {
        bool value = false;
        param->getValue(value);
        return value;
    }

    inline int read_choice_param(OFX::ChoiceParam* param) {
        int value = 0;
        param->getValue(value);
        return value;
    }

    inline std::string read_str_choice_param_or_empty(OFX::StrChoiceParam* param, const std::string& fallback) {
        std::string value;
        param->getValue(value);
        return value.empty() ? fallback : value;
    }

    inline Spektrafilm::ProfilePolarity capture_profile_polarity_for_key(
        const Spektrafilm::ProfileCatalog& catalog,
        const std::string& filmProfileKey) {
        for (const Spektrafilm::ProfileCatalogEntry& entry : catalog.filmProfiles) {
            if (entry.key == filmProfileKey) {
                return entry.polarity;
            }
        }
        return Spektrafilm::ProfilePolarity::Negative;
    }

    inline bool read_resolved_scan_route(
        const Spektrafilm::ProfileCatalog& catalog,
        OFX::StrChoiceParam* scanRouteParam,
        const std::string& filmProfileKey,
        Spektrafilm::ScanRoute& outRoute,
        std::string& outDiagnostic) {
        const Spektrafilm::ProfilePolarity capturePolarity =
            capture_profile_polarity_for_key(catalog, filmProfileKey);
        const Spektrafilm::ScanRoute defaultRoute =
            Spektrafilm::default_scan_route_for_polarity(capturePolarity);
        const std::string routeKey =
            read_str_choice_param_or_empty(scanRouteParam, Spektrafilm::scan_route_key(defaultRoute));
        const Spektrafilm::ScanRoute userRouteSelection =
            Spektrafilm::scan_route_from_key_or(routeKey, defaultRoute);
        return Spektrafilm::resolve_scan_route(
            capturePolarity, userRouteSelection, outRoute, outDiagnostic);
    }

    inline int read_choice_param_clamped(
        OFX::ChoiceParam* param,
        int minValue,
        int maxValue) {
        return std::clamp(read_choice_param(param), minValue, maxValue);
    }

    inline Spektrafilm::DiffusionFilterFamily read_diffusion_family(
        OFX::ChoiceParam* param) {
        switch (read_choice_param_clamped(param, 0, 3)) {
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

    inline std::array<double, 3> read_double3_param(OFX::Double3DParam* param) {
        std::array<double, 3> values{};
        param->getValue(values[0], values[1], values[2]);
        return values;
    }

    inline std::array<double, 2> read_double2_param(OFX::Double2DParam* param) {
        std::array<double, 2> values{};
        param->getValue(values[0], values[1]);
        return values;
    }

    void require_instance_success(FjStatus status) {
        if (status.category == FJ_STATUS_SUCCESS) {
            return;
        }
        if (status.category == FJ_STATUS_ALLOCATION_FAILURE) {
            throw std::bad_alloc();
        }
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    // Callback provenance is thread-local; publication ordering stays in Rust.
    class CaptureAction {
    public:
        CaptureAction(RenderInstance& instance, std::uint32_t mode, bool claimInitial = false)
            : instance(instance), previous(current) {
            FjErrorBuffer error{nullptr, 0, 0};
            FjCaptureStart start{};
            if (claimInitial) {
                require_instance_success(instance.begin_capture(FJ_CAPTURE_INITIAL, start, error));
                initial = start.kind == FJ_CAPTURE_STARTED;
            }
            if (!claimInitial || !initial) {
                require_instance_success(instance.begin_capture(mode, start, error));
            }
            if (start.kind == FJ_CAPTURE_STARTED) {
                ticket = start.capture;
                armed = true;
            }
            current = this;
        }
        ~CaptureAction() {
            fail(FJ_CAPTURE_HOST_FAILURE, "parameter action abandoned");
            current = previous;
        }
        CaptureAction(const CaptureAction&) = delete;
        CaptureAction& operator=(const CaptureAction&) = delete;

        std::uint32_t fail(std::uint32_t kind, const char* diagnostic) noexcept {
            if (!armed) {
                return completion;
            }
            FjErrorBuffer error{nullptr, 0, 0};
            std::uint32_t update = FJ_UPDATE_NONE;
            const FjStringView text{diagnostic, std::strlen(diagnostic)};
            (void)instance.invalidate(ticket, kind, text, update, error);
            armed = false;
            completion = update;
            return update;
        }
        bool is_current() const {
            if (!ticket) {
                return false;
            }
            FjErrorBuffer error{nullptr, 0, 0};
            std::uint32_t result = 0;
            require_instance_success(instance.capture_is_current(ticket, result, error));
            return result != 0;
        }

        static thread_local CaptureAction* current;
        RenderInstance& instance;
        std::uint64_t ticket = 0;
        bool armed = false;
        bool initial = false;
        bool refresh = false;
        bool physicalWriteFailed = false;
        std::uint32_t completion = FJ_UPDATE_IGNORED;

    private:
        CaptureAction* previous;
    };
    thread_local CaptureAction* CaptureAction::current = nullptr;

    // UI-only constructor calls have a context with no capture ticket.
    class HostEditScope {
    public:
        explicit HostEditScope(RenderInstance& instance)
            : instance(instance), action(CaptureAction::current), previous(current) {
            current = this;
        }
        ~HostEditScope() {
            current = previous;
        }
        static thread_local HostEditScope* current;
        RenderInstance& instance;
        CaptureAction* action;

    private:
        HostEditScope* previous;
    };
    thread_local HostEditScope* HostEditScope::current = nullptr;

    class HostCallScope {
    public:
        explicit HostCallScope(RenderInstance& instance)
            : instance(instance), previous(current) {
            current = this;
        }
        ~HostCallScope() {
            current = previous;
        }
        static thread_local HostCallScope* current;
        RenderInstance& instance;

    private:
        HostCallScope* previous;
    };
    thread_local HostCallScope* HostCallScope::current = nullptr;

    template <typename Setter>
    void registered_host_write(Setter&& setter) {
        CaptureAction* action = CaptureAction::current;
        if (!action || !action->ticket) {
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        FjErrorBuffer error{nullptr, 0, 0};
        std::optional<HostWrite> write;
        std::uint32_t kind = FJ_HOST_WRITE_NONE;
        require_instance_success(action->instance.begin_host_write(action->ticket, kind, write, error));
        if (kind == FJ_HOST_WRITE_IGNORED) {
            return;
        }
        if (!write) {
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        auto finish = [&](std::uint32_t failure, const char* text) {
            std::uint32_t refresh = 0;
            const FjStatus status = write->finish(failure, {text, std::strlen(text)}, refresh, error);
            write.reset();
            action->refresh = action->refresh || refresh != 0;
            return status;
        };
        try {
            const HostCallScope hostCall(action->instance);
            setter();
        } catch (const std::bad_alloc&) {
            action->physicalWriteFailed = true;
            (void)finish(FJ_CAPTURE_ALLOCATION_FAILURE, "host parameter write allocation failed");
            throw;
        } catch (...) {
            action->physicalWriteFailed = true;
            (void)finish(FJ_CAPTURE_HOST_FAILURE, "host parameter write failed");
            throw;
        }
        require_instance_success(finish(FJ_HOST_WRITE_SUCCEEDED, ""));
    }

    template <typename Property>
    void host_property(Property&& property) {
        CaptureAction* action = CaptureAction::current;
        HostEditScope* scope = HostEditScope::current;
        RenderInstance* instance = scope && scope->action == action ? &scope->instance : (action ? &action->instance : nullptr);
        if (action && instance == &action->instance && !action->is_current()) {
            return;
        }
        if (!instance) {
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        const HostCallScope hostCall(*instance);
        property();
    }

    inline void set_bool_param(OFX::BooleanParam* param, bool value) {
        registered_host_write([&]() {
            param->setValue(value);
        });
    }

    inline void set_double_param(OFX::DoubleParam* param, double value) {
        registered_host_write([&]() {
            param->setValue(value);
        });
    }

    inline void set_str_choice_param(OFX::StrChoiceParam* param, const char* value) {
        registered_host_write([&]() {
            param->setValue(value);
        });
    }

    inline void set_double2_param(
        OFX::Double2DParam* param,
        double x,
        double y) {
        registered_host_write([&]() {
            param->setValue(x, y);
        });
    }

    inline void set_double3_param(
        OFX::Double3DParam* param,
        double x,
        double y,
        double z) {
        registered_host_write([&]() {
            param->setValue(x, y, z);
        });
    }

    inline void set_choice_label(OFX::ChoiceParam* param, const std::string& label) {
        host_property([&]() {
            param->setLabel(label);
        });
    }

    inline void set_double_param_hint(OFX::DoubleParam* param, const std::string& hint) {
        host_property([&]() {
            param->setHint(hint);
        });
    }

    inline void set_double_param_hint(OFX::DoubleParam* param, const char* hint) {
        host_property([&]() {
            param->setHint(hint);
        });
    }

    inline void set_double_param_enabled(OFX::DoubleParam* param, bool enabled) {
        host_property([&]() {
            param->setEnabled(enabled);
        });
    }

    inline std::array<double, 3> read_sanitized_double3(
        OFX::Double3DParam* param,
        const std::array<double, 3>& defaults,
        double minValue,
        double maxValue) {
        std::array<double, 3> values = defaults;
        param->getValue(values[0], values[1], values[2]);
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
        param->getValue(values[0], values[1]);
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
        OFX::IntParam* scannerLutResolutionParam,
        ParamSnapshot& snapshot) {
        snapshot.scannerLensBlurSigmaPx = read_double_param(scannerLensBlurParam);
        snapshot.scannerUnsharpMask = read_double2_param(scannerUnsharpParam);
        snapshot.scannerBlackCorrection =
            read_bool_param_as_i32(scannerBlackCorrectionParam);
        snapshot.scannerWhiteCorrection =
            read_bool_param_as_i32(scannerWhiteCorrectionParam);
        snapshot.scannerBlackLevel =
            read_sanitized_unit_double(scannerBlackLevelParam, snapshot.scannerBlackLevel);
        snapshot.scannerWhiteLevel =
            read_sanitized_unit_double(scannerWhiteLevelParam, snapshot.scannerWhiteLevel);
        snapshot.scannerLutResolution = read_int_param(scannerLutResolutionParam);
    }

    inline void read_output_snapshot_values(
        OFX::ChoiceParam* outputColorSpaceParam,
        OFX::BooleanParam* outputCctfEncodingParam,
        OFX::BooleanParam* outputGamutCompressionParam,
        ParamSnapshot& snapshot) {
        snapshot.outputColorSpace = read_choice_param(outputColorSpaceParam);
        snapshot.outputCctfEncoding = read_bool_param_as_i32(outputCctfEncodingParam);
        snapshot.outputGamutCompressionEnabled =
            read_bool_param_as_i32(outputGamutCompressionParam);
    }

    struct ProfileSnapshotChoiceParams {
        OFX::StrChoiceParam* filmProfileKey = nullptr;
        OFX::StrChoiceParam* printProfileKey = nullptr;
        OFX::StrChoiceParam* scanRoute = nullptr;
        OFX::ChoiceParam* spectralMode = nullptr;
        OFX::ChoiceParam* referenceIlluminant = nullptr;
        OFX::ChoiceParam* enlargerIlluminant = nullptr;
    };

    inline bool read_profile_snapshot_choices(
        const Spektrafilm::ProfileCatalog& catalog,
        const ProfileSnapshotChoiceParams& params,
        ParamSnapshot& snapshot,
        std::string& outDiagnostic) {
        snapshot.filmProfileKey = read_str_choice_param_or_empty(params.filmProfileKey, snapshot.filmProfileKey);
        snapshot.printProfileKey = read_str_choice_param_or_empty(params.printProfileKey, snapshot.printProfileKey);
        if (!read_resolved_scan_route(
                catalog,
                params.scanRoute,
                snapshot.filmProfileKey,
                snapshot.scanRoute,
                outDiagnostic)) {
            return false;
        }
        snapshot.spectralUpsamplingMode = read_choice_param(params.spectralMode);
        snapshot.refIll = read_choice_param(params.referenceIlluminant);
        snapshot.enlIll = read_choice_param(params.enlargerIlluminant);
        return true;
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
            read_double_param(printExposureParam),
            snapshot.printExposure);
        snapshot.printPreflashExposure = sanitize_finite_or(
            read_double_param(printPreflashParam),
            snapshot.printPreflashExposure);
        snapshot.printExposureCompensation =
            read_bool_param_as_i32(printExposureCompensationParam);
        snapshot.normalizePrintExposure = 1;
        snapshot.printUiYmcCc = {
            sanitize_finite_or(read_double_param(enlargerYParam), 0.0),
            sanitize_finite_or(read_double_param(enlargerMParam), 0.0),
            sanitize_finite_or(read_double_param(enlargerCParam), 0.0)};
    }

    inline void read_input_snapshot_values(
        OFX::ChoiceParam* inputColorSpaceParam,
        OFX::BooleanParam* inputCctfDecodingParam,
        OFX::BooleanParam* inputCompressionParam,
        OFX::BooleanParam* hanatos2025AdaptationWindowParam,
        OFX::BooleanParam* hanatos2025AdaptationSurfaceParam,
        ParamSnapshot& snapshot) {
        snapshot.inputColorSpace = read_choice_param(inputColorSpaceParam);
        snapshot.inputCctfDecoding = read_bool_param_as_i32(inputCctfDecodingParam);
        snapshot.inputCompressionEnabled =
            read_bool_param_as_i32(inputCompressionParam);
        snapshot.hanatos2025AdaptationWindow =
            read_bool_param_as_i32(hanatos2025AdaptationWindowParam);
        snapshot.hanatos2025AdaptationSurface =
            read_bool_param_as_i32(hanatos2025AdaptationSurfaceParam);
    }

    inline bool read_coupler_snapshot_values(
        OFX::BooleanParam* couplersActiveParam,
        OFX::DoubleParam* couplersAmountParam,
        OFX::DoubleParam* inhibitionSameLayerParam,
        OFX::DoubleParam* inhibitionInterlayerParam,
        OFX::DoubleParam* diffusionSizeUmParam,
        OFX::Double3DParam* donorKParam,
        OFX::Double3DParam* receiverKParam,
        OFX::DoubleParam* diffusionTailUmParam,
        OFX::DoubleParam* diffusionTailWeightParam,
        OFX::BooleanParam* gammaUseStockParam,
        OFX::Double3DParam* gammaSameLayerRgbParam,
        OFX::Double2DParam* gammaInterlayerRToGbParam,
        OFX::Double2DParam* gammaInterlayerGToRbParam,
        OFX::Double2DParam* gammaInterlayerBToRgParam,
        Spektrafilm::ProfilePolarity polarity,
        ParamSnapshot& snapshot,
        std::string& diagnostic) {
        Spektrafilm::DirCouplersControls controls{};
        controls.active = read_bool_param(couplersActiveParam);
        controls.gammaUseStock = read_bool_param(gammaUseStockParam);
        const auto narrow = [&](double authored,
                                double minimum,
                                double maximum,
                                const char* field,
                                float& destination) {
            if (!std::isfinite(authored) ||
                authored < minimum ||
                authored > maximum) {
                diagnostic = std::string("InvalidSnapshotControls component=dir field=") +
                             field + " requirement=finite_supported_range";
                return false;
            }
            destination = static_cast<float>(authored);
            if (!std::isfinite(destination)) {
                diagnostic = std::string("InvalidSnapshotControls component=dir field=") +
                             field + " requirement=float_representable";
                return false;
            }
            return true;
        };
        if (!controls.active) {
            snapshot.dirCouplers = controls;
            return true;
        }
        if (!narrow(
                read_double_param(couplersAmountParam),
                0.0,
                2.0,
                "amount",
                controls.amount)) {
            return false;
        }
        if (controls.amount == 0.0f) {
            snapshot.dirCouplers = controls;
            return true;
        }
        if (!narrow(read_double_param(inhibitionSameLayerParam), 0.0, 2.0, "inhibition_same", controls.inhibitionSameLayer) ||
            !narrow(read_double_param(inhibitionInterlayerParam), 0.0, 2.0, "inhibition_inter", controls.inhibitionInterlayer) ||
            !narrow(read_double_param(diffusionSizeUmParam), 0.0, 200.0, "diffusion_core_um", controls.diffusionSizeUm)) {
            return false;
        }
        if (!controls.gammaUseStock) {
            const auto same = read_double3_param(gammaSameLayerRgbParam);
            const auto rToGb = read_double2_param(gammaInterlayerRToGbParam);
            const auto gToRb = read_double2_param(gammaInterlayerGToRbParam);
            const auto bToRg = read_double2_param(gammaInterlayerBToRgParam);
            for (std::size_t channel = 0; channel < 3u; ++channel) {
                if (!narrow(same[channel], 0.0, 4.0, "gamma_same", controls.gammaSameLayerRgb[channel])) {
                    return false;
                }
            }
            for (std::size_t channel = 0; channel < 2u; ++channel) {
                if (!narrow(rToGb[channel], 0.0, 4.0, "gamma_r_to_gb", controls.gammaInterlayerRToGb[channel]) ||
                    !narrow(gToRb[channel], 0.0, 4.0, "gamma_g_to_rb", controls.gammaInterlayerGToRb[channel]) ||
                    !narrow(bToRg[channel], 0.0, 4.0, "gamma_b_to_rg", controls.gammaInterlayerBToRg[channel])) {
                    return false;
                }
            }
        }
        OFX::Double3DParam* consumedKParam =
            polarity == Spektrafilm::ProfilePolarity::Positive
                ? receiverKParam
                : donorKParam;
        const auto authoredK = read_double3_param(consumedKParam);
        auto& consumedK = polarity == Spektrafilm::ProfilePolarity::Positive
                              ? controls.langmuirReceiverKRgb
                              : controls.langmuirDonorKRgb;
        for (std::size_t channel = 0; channel < 3u; ++channel) {
            if (!narrow(
                    authoredK[channel],
                    0.1,
                    static_cast<double>(std::numeric_limits<float>::max()),
                    polarity == Spektrafilm::ProfilePolarity::Positive
                        ? "receiver_k"
                        : "donor_k",
                    consumedK[channel])) {
                return false;
            }
        }
        if (controls.diffusionSizeUm > 0.0f) {
            if (!narrow(
                    read_double_param(diffusionTailWeightParam),
                    0.0,
                    1.0,
                    "diffusion_tail_weight",
                    controls.diffusionTailWeight)) {
                return false;
            }
            if (controls.diffusionTailWeight > 0.0f &&
                !narrow(
                    read_double_param(diffusionTailUmParam),
                    0.0,
                    static_cast<double>(std::numeric_limits<float>::max()),
                    "diffusion_tail_um",
                    controls.diffusionTailUm)) {
                return false;
            }
        }
        snapshot.dirCouplers = controls;
        return true;
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


} // namespace

JuicerEffect::ExposureParams JuicerEffect::gatherExposureParams() const {
    ExposureParams params{};
    double exposureSliderEV = read_double_param(_pExposure);
    exposureSliderEV = sanitize_finite_or(exposureSliderEV, 0.0);
    params.sliderEV = exposureSliderEV;
    params.sliderScale = finite_exp2_scale(exposureSliderEV);
    const bool cameraAuto = read_bool_param(_pCameraAutoExposure);
    params.cameraAutoEnabled = cameraAuto;
    const int meteringMethod = read_choice_param(_pCameraMeteringMethod);
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
        set_double3_param(param, master * ratio[0], master * ratio[1], master * ratio[2]);
    }

    inline void set_scaled_triplet_clamped(
        OFX::Double3DParam* param,
        double master,
        const std::array<double, 3>& ratio,
        double lo,
        double hi) {
        set_double3_param(param, std::clamp(master * ratio[0], lo, hi), std::clamp(master * ratio[1], lo, hi), std::clamp(master * ratio[2], lo, hi));
    }

} // namespace

// S6.B/C transfers these resolved EV/chroma/default calculations; current OFX getter/edit mechanics stay native.
Spektrafilm::VisualGrainControls JuicerEffect::gatherGrainUi() const {
    Spektrafilm::VisualGrainControls grain{};

    grain.active = read_bool_param(_pGrainActive);

    const int presetIndex = read_choice_param_clamped(_pGrainPreset, 0, 2);
    const GrainPresetDefaults preset = grain_preset_defaults(presetIndex);

    grain.sublayersActive = read_bool_param(_pGrainSublayersActive);

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

    grain.debugView = read_choice_param_clamped(_pGrainDebugView, 0, 6);

    grain.nSubLayers = 1;
    return grain;
}

void JuicerEffect::applyGrainPresetDefaults(int presetIndex) {
    const GrainPresetDefaults preset = grain_preset_defaults(presetIndex);
    const HostEditScope suppressEvents(_instance);

    const GrainAdvancedDefaults advancedDefaults = compute_grain_advanced_defaults(
        preset,
        preset.sharpness,
        preset.texture);

    const GrainRatioSet ratios = normalized_default_grain_ratios();

    set_double_param(_pGrainAmplitude, preset.amountEV);
    set_double_param(_pGrainBlur, preset.sizePx);
    set_double_param(_pGrainSharpness, preset.sharpness);
    set_double_param(_pGrainChroma, preset.chroma);
    set_double_param(_pGrainTexture, preset.texture);
    set_bool_param(_pGrainSublayersActive, preset.sublayersActive);

    set_double_param(_pGrainParticleAreaUm2, preset.particleAreaUm2);
    set_double_param(_pGrainParticleScaleMaster, preset.particleScaleMaster);
    cacheGrainMaster(_grainParticleScaleMasterLast, preset.particleScaleMaster);
    set_scaled_triplet_clamped(_pGrainParticleScale, preset.particleScaleMaster, ratios.scale, 0.0, 10.0);
    set_double_param(_pGrainParticleScaleLayersMaster, preset.particleScaleLayersMaster);
    cacheGrainMaster(_grainParticleScaleLayersMasterLast, preset.particleScaleLayersMaster);
    set_scaled_triplet_clamped(_pGrainParticleScaleLayers, preset.particleScaleLayersMaster, ratios.scaleLayers, 0.0, 10.0);
    set_double_param(_pGrainDensityMinMaster, preset.densityMinMaster);
    cacheGrainMaster(_grainDensityMinMasterLast, preset.densityMinMaster);
    set_scaled_triplet_clamped(_pGrainDensityMin, preset.densityMinMaster, ratios.densityMin, 0.0, 1.0);
    set_double_param(_pGrainUniformityMaster, preset.uniformityMaster);
    cacheGrainMaster(_grainUniformityMasterLast, preset.uniformityMaster);
    set_scaled_triplet_clamped(_pGrainUniformity, preset.uniformityMaster, ratios.uniformity, 0.0, 1.0);
    set_double_param(_pGrainBlurDyeCloudsUm, std::clamp(advancedDefaults.blurDyeClouds, 0.0, 10.0));
    set_double_param(_pGrainSizeMixWeight, std::clamp(advancedDefaults.sizeMixWeight, 0.0, 1.0));
    set_double_param(_pGrainSizeMixWeightMid, 0.0);
    set_double_param(_pGrainSizeMixScale, std::clamp(preset.sizeMixScale, 1.0, 50.0));
    set_double2_param(
        _pGrainMicroStructure,
        std::clamp(advancedDefaults.microCell, 0.0, 1000.0),
        std::clamp(advancedDefaults.microSigma, 0.0, 1000.0));
    set_double_param(_pGrainClumpTemporalMix, 0.30);
    set_double_param(_pGrainClumpMorphPeriodSec, 8.0);

    updateGrainPresetLabel(false);
    updateGrainChromaEnabled();
}

void JuicerEffect::resetGrainAdvancedControls() {
    const int presetIndex = read_choice_param_clamped(_pGrainPreset, 0, 2);
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
    const HostEditScope suppressEvents(_instance);

    set_double_param(_pGrainParticleAreaUm2, particleArea);
    set_double_param(_pGrainParticleScaleMaster, particleScaleMaster);
    cacheGrainMaster(_grainParticleScaleMasterLast, particleScaleMaster);
    set_scaled_triplet(_pGrainParticleScale, particleScaleMaster, ratios.scale);
    set_double_param(_pGrainParticleScaleLayersMaster, particleScaleLayersMaster);
    cacheGrainMaster(_grainParticleScaleLayersMasterLast, particleScaleLayersMaster);
    set_scaled_triplet(_pGrainParticleScaleLayers, particleScaleLayersMaster, ratios.scaleLayers);
    set_double_param(_pGrainDensityMinMaster, densityMinMaster);
    cacheGrainMaster(_grainDensityMinMasterLast, densityMinMaster);
    set_scaled_triplet(_pGrainDensityMin, densityMinMaster, ratios.densityMin);
    set_double_param(_pGrainUniformityMaster, uniformityMaster);
    cacheGrainMaster(_grainUniformityMasterLast, uniformityMaster);
    set_scaled_triplet(_pGrainUniformity, uniformityMaster, ratios.uniformity);
    set_double_param(_pGrainBlurDyeCloudsUm, advancedDefaults.blurDyeClouds);
    set_double_param(_pGrainSizeMixWeight, advancedDefaults.sizeMixWeight);
    set_double_param(_pGrainSizeMixWeightMid, 0.0);
    set_double_param(_pGrainSizeMixScale, sizeMixScale);
    set_double2_param(_pGrainMicroStructure, advancedDefaults.microCell, advancedDefaults.microSigma);
    set_double_param(_pGrainClumpTemporalMix, clumpTemporalMix);
    set_double_param(_pGrainClumpMorphPeriodSec, clumpMorphPeriodSec);

    updateGrainChromaEnabled();
}

void JuicerEffect::cacheGrainMaster(double& target, double value) {
    if (CaptureAction::current && &CaptureAction::current->instance == &_instance && !CaptureAction::current->is_current()) {
        return;
    }
    std::lock_guard<std::mutex> lock(_uiMutex);
    target = value;
}

void JuicerEffect::updateGrainPresetLabel(bool custom) {
    if (CaptureAction::current && &CaptureAction::current->instance == &_instance && !CaptureAction::current->is_current()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(_uiMutex);
        _grainPresetCustom = custom;
    }
    const std::string label = custom
                                  ? (_grainPresetLabel + " (Custom)")
                                  : _grainPresetLabel;
    set_choice_label(_pGrainPreset, label);
}

void JuicerEffect::updateGrainChromaEnabled() {
    const bool perChannelDirty = false;
    set_double_param_enabled(_pGrainChroma, !perChannelDirty);
    if (perChannelDirty) {
        set_double_param_hint(_pGrainChroma, "Chroma disabled when per-channel overrides are active.");
    } else {
        set_double_param_hint(_pGrainChroma, _grainChromaHint);
    }
}

Spektrafilm::DiffusionFilterAuthoredControls JuicerEffect::gatherDiffusionUi(
    const DiffusionUiParams& params) const {
    Spektrafilm::DiffusionFilterAuthoredControls controls{};
    controls.active = read_bool_param(params.enabled);
    controls.family = read_diffusion_family(params.family);
    controls.strength = read_double_param(params.strength);
    controls.spatialScale = read_double_param(params.spatialScale);
    controls.haloWarmth = read_double_param(params.haloWarmth);
    controls.coreIntensity = read_double_param(params.coreIntensity);
    controls.coreSize = read_double_param(params.coreSize);
    controls.haloIntensity = read_double_param(params.haloIntensity);
    controls.haloSize = read_double_param(params.haloSize);
    controls.bloomIntensity = read_double_param(params.bloomIntensity);
    controls.bloomSize = read_double_param(params.bloomSize);
    return controls;
}

bool JuicerEffect::resolveDiffusionControlRoute(
    Spektrafilm::ScanRoute& outRoute,
    std::string& outDiagnostic) {
    const std::string filmProfileKey =
        read_str_choice_param_or_empty(_pFilmProfileKey, Spektrafilm::kDefaultFilmProfileKey);
    return read_resolved_scan_route(_runtime.spektrafilm_profile_catalog(), _pScanRoute, filmProfileKey, outRoute, outDiagnostic);
}

void JuicerEffect::updateDiffusionControlState(Spektrafilm::ScanRoute selectedRoute) {
    const bool cameraEnabled = read_bool_param(_cameraDiffusionUi.enabled);
    const bool printRoute = Spektrafilm::scan_route_is_print(selectedRoute);
    const bool printEnabled = read_bool_param(_printDiffusionUi.enabled);

    auto setControlState = [](auto* param, bool visible, bool enabled) {
        host_property([&]() {
            param->setIsSecret(!visible);
        });
        host_property([&]() {
            param->setEnabled(enabled);
        });
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

bool JuicerEffect::resolveGammaControlRoute(
    Spektrafilm::ScanRoute& outRoute,
    std::string& outFilmProfileKey,
    std::string& outDiagnostic) {
    outFilmProfileKey =
        read_str_choice_param_or_empty(
            _pFilmProfileKey,
            Spektrafilm::kDefaultFilmProfileKey);
    return read_resolved_scan_route(_runtime.spektrafilm_profile_catalog(), _pScanRoute, outFilmProfileKey, outRoute, outDiagnostic);
}

void JuicerEffect::updateGammaControlState(
    Spektrafilm::ScanRoute selectedRoute,
    const std::string&) {
    host_property([&]() {
        _pPrintGammaFactor->setEnabled(
            Spektrafilm::scan_route_is_print(selectedRoute));
    });
    const bool dirActive = read_bool_param(_pCouplersActive);
    const double coreUm = read_double_param(_pCouplersDiffusionSizeUm);
    const double tailWeight = read_double_param(_pCouplersDiffusionTailWeight);
    const bool positive =
        Spektrafilm::scan_route_metadata(selectedRoute).capturePolarity ==
        Spektrafilm::ProfilePolarity::Positive;
    host_property([&]() {
        _pCouplersLangmuirDonorKRgb->setEnabled(dirActive && !positive);
    });
    host_property([&]() {
        _pCouplersLangmuirReceiverKRgb->setEnabled(dirActive && positive);
    });
    host_property([&]() {
        _pCouplersDiffusionTailWeight->setEnabled(dirActive && coreUm > 0.0);
    });
    host_property([&]() {
        _pCouplersDiffusionTailUm->setEnabled(
            dirActive && coreUm > 0.0 && tailWeight > 0.0);
    });
}

void JuicerEffect::updateSpectralControlState() {
    const int mode = read_choice_param(_pSpectralMode);
    host_property([&]() {
        _pInputCompression->setEnabled(mode != 1);
    });
    const bool hanatos = mode == 0;
    host_property([&]() {
        _pHanatos2025AdaptationWindow->setEnabled(hanatos);
    });
    host_property([&]() {
        _pHanatos2025AdaptationSurface->setEnabled(hanatos);
    });
}

void JuicerEffect::applyDirGammaProfileDefaults() {
    if (!read_bool_param(_pCouplersGammaUseStock)) {
        return;
    }

    const std::string filmKey =
        read_str_choice_param_or_empty(_pFilmProfileKey, Spektrafilm::kDefaultFilmProfileKey);
    FjFilmDigest digest{};
    std::array<char, 1024> message{};
    FjErrorBuffer error{message.data(), message.size(), 0};
    const FjStatus status = _runtime.film_defaults(filmKey, digest, error);
    if (status.category != FJ_STATUS_SUCCESS) {
        if (status.category == FJ_STATUS_PREPARATION_FAILURE) {
            JTRACE("SPEKTRAFILM", "MissingRequiredResource field=dir_gamma_profile_defaults");
            return;
        }
        if (status.category == FJ_STATUS_ALLOCATION_FAILURE) {
            throw std::bad_alloc();
        }
        JuicerCuda::Failure failure;
        JuicerCuda::set_failure(
            failure,
            status,
            std::string_view(message.data(), error.length));
        throw JuicerCuda::ExecutionFailure{std::move(failure)};
    }

    const HostEditScope suppressEvents(_instance);
    set_double3_param(
        _pCouplersGammaSameLayerRgb,
        digest.gamma_samelayer_rgb[0],
        digest.gamma_samelayer_rgb[1],
        digest.gamma_samelayer_rgb[2]);
    set_double2_param(
        _pCouplersGammaInterlayerRToGb,
        digest.gamma_interlayer_r_to_gb[0],
        digest.gamma_interlayer_r_to_gb[1]);
    set_double2_param(
        _pCouplersGammaInterlayerGToRb,
        digest.gamma_interlayer_g_to_rb[0],
        digest.gamma_interlayer_g_to_rb[1]);
    set_double2_param(
        _pCouplersGammaInterlayerBToRg,
        digest.gamma_interlayer_b_to_rg[0],
        digest.gamma_interlayer_b_to_rg[1]);
}

JuicerEffect::JuicerEffect(OfxImageEffectHandle handle, JuicerProcess::ProcessRuntime& runtime) try
    : OFX
    ::ImageEffect(handle),
        _runtime(runtime) {
        _src = fetchClip(kOfxImageEffectSimpleSourceClipName);
        _dst = fetchClip(kOfxImageEffectOutputClipName);

        // Cache parameter handles (wrappers)
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
        _pInputCompression = fetchBooleanParam(JuicerParams::kInputCompression);
        _pHanatos2025AdaptationWindow =
            fetchBooleanParam(JuicerParams::kHanatos2025AdaptationWindow);
        _pHanatos2025AdaptationSurface =
            fetchBooleanParam(JuicerParams::kHanatos2025AdaptationSurface);
        _pScanRoute = fetchStrChoiceParam(JuicerParams::kParamScanRoute);
        _pFilmGammaFactor = fetchDoubleParam(JuicerParams::kFilmGammaFactor);
        _pPrintGammaFactor = fetchDoubleParam(JuicerParams::kPrintGammaFactor);
        _pOutputColorSpace = fetchChoiceParam(kParamOutputColorSpace);
        _pOutputCctfEncoding = fetchBooleanParam(kParamOutputCctfEncoding);
        _pOutputGamutCompression =
            fetchBooleanParam(JuicerParams::kOutputGamutCompression);

        _pCouplersActive = fetchBooleanParam(JuicerParams::kDirCouplersActive);
        _pCouplersAmount = fetchDoubleParam(JuicerParams::kDirCouplersAmount);
        _pCouplersInhibitionSameLayer =
            fetchDoubleParam(JuicerParams::kDirCouplersInhibitionSameLayer);
        _pCouplersInhibitionInterlayer =
            fetchDoubleParam(JuicerParams::kDirCouplersInhibitionInterlayer);
        _pCouplersDiffusionSizeUm =
            fetchDoubleParam(JuicerParams::kDirCouplersDiffusionSizeUm);
        _pCouplersLangmuirDonorKRgb =
            fetchDouble3DParam(JuicerParams::kDirCouplersLangmuirDonorKRgb);
        _pCouplersLangmuirReceiverKRgb =
            fetchDouble3DParam(JuicerParams::kDirCouplersLangmuirReceiverKRgb);
        _pCouplersDiffusionTailUm =
            fetchDoubleParam(JuicerParams::kDirCouplersDiffusionTailUm);
        _pCouplersDiffusionTailWeight =
            fetchDoubleParam(JuicerParams::kDirCouplersDiffusionTailWeight);
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
        _pScannerLutResolution = fetchIntParam(JuicerParams::kScannerLutResolution);

        _pPrintExposure = fetchDoubleParam("PrintExposure");
        _pPrintPreflash = fetchDoubleParam("PrintPreflash");
        _pPrintExposureComp = fetchBooleanParam("PrintExposureCompensation");
        _pEnlargerY = fetchDoubleParam("EnlargerY");
        _pEnlargerM = fetchDoubleParam("EnlargerM");
        _pEnlargerC = fetchDoubleParam("EnlargerC");

        _pHalationActive = fetchBooleanParam(JuicerParams::kHalationActive);
        _pHalationScatterAmount =
            fetchDoubleParam(JuicerParams::kHalationScatterAmount);
        _pHalationScatterSpatialScale =
            fetchDoubleParam(JuicerParams::kHalationScatterSpatialScale);
        _pHalationAmount = fetchDoubleParam(JuicerParams::kHalationAmount);
        _pHalationSpatialScale =
            fetchDoubleParam(JuicerParams::kHalationSpatialScale);

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

        std::string grainPresetLabel;
        _pGrainPreset->getLabel(grainPresetLabel);
        if (!grainPresetLabel.empty()) {
            _grainPresetLabel = grainPresetLabel;
        }
        const std::string grainChromaHint = _pGrainChroma->getHint();
        if (!grainChromaHint.empty()) {
            _grainChromaHint = grainChromaHint;
        }

        auto initMasterCache = [&](OFX::DoubleParam* param, double& outValue) {
            double v = 0.0;
            param->getValue(v);
            cacheGrainMaster(outValue, v);
        };
        initMasterCache(_pGrainParticleScaleMaster, _grainParticleScaleMasterLast);
        initMasterCache(_pGrainParticleScaleLayersMaster, _grainParticleScaleLayersMasterLast);
        initMasterCache(_pGrainDensityMinMaster, _grainDensityMinMasterLast);
        initMasterCache(_pGrainUniformityMaster, _grainUniformityMasterLast);

        // SF_TEMP_BRIDGE_FRAME_IDENTITY_INPUTS: clock/address conversion moves at S6.
        {
            const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            std::uint64_t seedFields[2]{};
            seedFields[0] = static_cast<std::uint64_t>(now);
            seedFields[1] = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(this));
            std::uint64_t seed = Hash::hash_bytes(seedFields, sizeof(seedFields));
            if (seed == 0) {
                seed = 1;
            }
            std::array<char, 512> text{};
            FjErrorBuffer error{text.data(), text.size(), 0};
            require_instance_success(_instance.bind_identity(seed, seed, error));
        }

        const HostEditScope uiContext(_instance);
        std::string routeDiagnostic;
        std::string gammaFilmProfileKey;
        Spektrafilm::ScanRoute diffusionRoute{};
        Spektrafilm::ScanRoute gammaRoute{};
        // Both route reads must succeed before either helper changes route-dependent UI.
        if (!resolveDiffusionControlRoute(diffusionRoute, routeDiagnostic) ||
            !resolveGammaControlRoute(gammaRoute, gammaFilmProfileKey, routeDiagnostic)) {
            JTRACE("BUILD", routeDiagnostic);
            try {
                sendMessage(OFX::Message::eMessageError, "FilmJuicerRouteResolution", routeDiagnostic);
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        updateDiffusionControlState(diffusionRoute);
        updateGammaControlState(gammaRoute, gammaFilmProfileKey);
        updateSpectralControlState();

        // Defer initial pending render state publication until the first parameter change.
    }
catch (const RenderInstance::CreationFailure&) {
    throw OFX::Exception::Suite(kOfxStatErrFatal);
}

JuicerEffect::~JuicerEffect() {
    try {
        std::array<char, 512> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
        const FjRuntimeView runtimeView = _runtime.borrow();
        std::uint32_t initialized = 0;
        std::uint64_t seed = 0;
        std::uint64_t token = 0;
        const auto identity = fj_legacy_instance_identity(_instance.borrow(), &initialized, &seed, &token, &error);
        if (identity.category != FJ_STATUS_SUCCESS) {
            JTRACE("MSLCY", diagnostic.data());
            return;
        }
        if (initialized == 0) {
            return;
        }
        const auto result = fj_cuda_retire_instance(runtimeView.cuda, token, &error);
#if defined(JUICER_FRAME_GEOMETRY_TEST_HOOK)
        JuicerFrame::FrameGeometryTest::instance_retired(_instance.borrow(), token, result);
#endif
        if (result.category != FJ_STATUS_SUCCESS) {
            JTRACE("MSLCY", diagnostic.data());
        }
    } catch (...) {
        JuicerLogging::discard_current_exception();
    }
}

[[noreturn]] void JuicerEffect::throw_spektrafilm_phase1a_render_cutoff(const OFX::RenderArguments& args) const {
    const std::string filmKey = read_str_choice_param_or_empty(_pFilmProfileKey, Spektrafilm::kDefaultFilmProfileKey);
    const std::string printKey = read_str_choice_param_or_empty(_pPrintProfileKey, Spektrafilm::kDefaultPrintProfileKey);
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

#if defined(JUICER_FRAME_GEOMETRY_TEST_HOOK)
namespace JuicerFrame::FrameGeometryTest {
    void diffusion_frame(double, Spektrafilm::ScanRoute, Spektrafilm::DiffusionFrameDomain, bool, const std::optional<Spektrafilm::DiffusionFrameSetDescriptor>&);
    void halation_frame(float, bool, const std::optional<ScatterHalationFrameDescriptor>&);
    void prepared_policy(const FjPreparedFramePolicy&, std::uint32_t);
    void full_domain(const Spektrafilm::DiffusionFrameDomain&);
    void scalar_pixel(float);
    void effects_domain(const Spektrafilm::FilmJuicerEffectsGeometry&);
} // namespace JuicerFrame::FrameGeometryTest
#endif

void JuicerEffect::render(const OFX::RenderArguments& args) {
    const bool cudaRoute = args.isEnabledCudaRender;
    if (!cudaRoute) {
        throw_spektrafilm_phase1a_render_cutoff(args);
    }
    const FjRuntimeView runtimeView = _runtime.borrow();
    auto framePreparation = JuicerCuda::borrow_root(runtimeView.cuda).begin_frame_preparation();
    if (!framePreparation.active()) {
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    // Fetch images via wrappers
    std::unique_ptr<OFX::Image> srcImg(_src->fetchImage(args.time));
    std::unique_ptr<OFX::Image> dstImg(_dst->fetchImage(args.time));
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

    // ROI: args.renderWindow if provided; otherwise use image bounds
    OfxRectI roi = args.renderWindow;
    if (roi.x1 == roi.x2 && roi.y1 == roi.y2) {
        roi = fullBounds;
    }
    if (roi.x2 <= roi.x1 || roi.y2 <= roi.y1)
        return;
    const auto fullDomain = frame_domain(fullBounds);
#if defined(JUICER_FRAME_GEOMETRY_TEST_HOOK)
    JuicerFrame::FrameGeometryTest::full_domain(fullDomain);
#endif
    const bool fullFrame = (roi.x1 == fullBounds.x1 && roi.y1 == fullBounds.y1 &&
                            roi.x2 == fullBounds.x2 && roi.y2 == fullBounds.y2);

    std::array<char, 2048> admissionText{};
    FjErrorBuffer admissionError{admissionText.data(), admissionText.size(), 0};
    std::uint32_t admissionKind = FJ_ADMISSION_NONE;
    std::optional<PreparedState> admitted;
    const FjAssets* assets = runtimeView.assets;
    FjStatus admissionStatus = _instance.admit(assets, admissionKind, admitted, admissionError);
    if (admissionKind == FJ_ADMISSION_NEEDS_CAPTURE || admissionKind == FJ_ADMISSION_NEEDS_REFRESH) {
        acquire_render_capture(admissionKind == FJ_ADMISSION_NEEDS_CAPTURE ? FJ_CAPTURE_INITIAL : FJ_CAPTURE_REFRESH);
        admissionStatus = _instance.admit(assets, admissionKind, admitted, admissionError);
    }
    if (admissionKind == FJ_ADMISSION_NEEDS_CAPTURE || admissionKind == FJ_ADMISSION_NEEDS_REFRESH ||
        admissionKind == FJ_ADMISSION_CAPTURING) {
        throw OFX::Exception::Suite(kOfxStatFailed);
    }
    if (admissionStatus.category != FJ_STATUS_SUCCESS || admissionKind != FJ_ADMISSION_READY || !admitted) {
        if (admissionStatus.category == FJ_STATUS_ALLOCATION_FAILURE) {
            throw std::bad_alloc();
        }
        try {
            std::string deliveryText(admissionText.data(), admissionError.length);
            for (std::size_t position = 0; (position = deliveryText.find('%', position)) != std::string::npos; position += 2u) {
                deliveryText.insert(position, 1u, '%');
            }
            sendMessage(OFX::Message::eMessageError, "FilmJuicerRenderAdmission", deliveryText);
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
        trace_and_throw_render_fatal(RenderFatalTrace{"BUILD", admissionText.data()});
    }
    const PreparedState& prepared = *admitted;
    std::array<char, 2048> stateText{};
    FjErrorBuffer stateError{stateText.data(), stateText.size(), 0};
    FjPreparedStateView stateView{};
    require_instance_success(prepared.view(stateView, stateError));
    if (stateView.route > static_cast<std::uint32_t>(Spektrafilm::ScanRoute::PositivePrintScan)) {
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
    const bool printRoute = Spektrafilm::scan_route_is_print(static_cast<Spektrafilm::ScanRoute>(stateView.route));
    const bool directCudaRoute = cudaRoute && !printRoute;
    if (!fullFrame && !directCudaRoute) {
        JTRACE("RENDER", "FATAL: render window must match full frame; tiles/ROIs are unsupported");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    const FjExtent fullExtent{
        fullDomain.originX,
        fullDomain.originY,
        fullDomain.width,
        fullDomain.height};
    FjPreparedFramePolicy framePolicy{};
    stateError.length = 0;
    const FjStatus policyStatus = prepared.frame_policy(fullExtent, framePolicy, stateError);
    if (policyStatus.category != FJ_STATUS_SUCCESS) {
        JTRACE("CUDA", std::string(stateText.data(), stateError.length));
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
#if defined(JUICER_FRAME_GEOMETRY_TEST_HOOK)
    JuicerFrame::FrameGeometryTest::prepared_policy(framePolicy, stateView.route);
#endif
#if defined(JUICER_FRAME_GEOMETRY_TEST_HOOK)
    JuicerFrame::FrameGeometryTest::scalar_pixel(framePolicy.geometry.pixel_size_um);
#endif
    if (traceVerbose) {
        std::string msg = printRoute ? "render print state build=" : "render direct state build=";
        msg += std::to_string(prepared.build());
        msg += " recipe_hash=";
        msg += std::to_string(framePolicy.recipe_hash);
        JTRACE_VERBOSE("RENDER", msg);
    }

    Spektrafilm::FilmJuicerEffectsGeometry effectsGeometry{};
    if (stateView.effects.active != 0) {
        const auto definition = srcImg->getRegionOfDefinition();
        const auto canonical = _src->getRegionOfDefinition(args.time);
        const auto scale = srcImg->getRenderScale();
        const auto definitionDomain = frame_domain(definition);
        effectsGeometry.pixelDefinition = {definitionDomain.originX, definitionDomain.originY, definitionDomain.width, definitionDomain.height};
        effectsGeometry.canonicalX = canonical.x1;
        effectsGeometry.canonicalY = canonical.y1;
        effectsGeometry.canonicalWidth = canonical.x2 - canonical.x1;
        effectsGeometry.canonicalHeight = canonical.y2 - canonical.y1;
        effectsGeometry.scaleX = scale.x;
        effectsGeometry.scaleY = scale.y;
        effectsGeometry.pixelAspectRatio = srcImg->getPixelAspectRatio();
#if defined(JUICER_FRAME_GEOMETRY_TEST_HOOK)
        JuicerFrame::FrameGeometryTest::effects_domain(effectsGeometry);
#endif
    }

    // The callback retains image leases through inspection and CUDA submission.
    JuicerProcessor proc(*this, runtimeView);
    JuicerProcessor::SourceDestinationImages images{};
    images.src = srcImg.get();
    images.dst = dstImg.get();
    proc.setSrcDst(images);
    // SF_TEMP_BRIDGE_FRAME_IDENTITY_INPUTS: actual OFXS clip conversion moves at S6.
    const std::uintptr_t renderClipToken = reinterpret_cast<std::uintptr_t>(_src);

    if (printRoute) {
        JuicerProcessor::PrintFrameRequest frameRequest{};
        frameRequest.renderState = &prepared;
        frameRequest.framePolicy = framePolicy;
        frameRequest.renderWindow = roi;
        frameRequest.effectsGeometry = effectsGeometry;
        frameRequest.fullFrameExtent = fullBounds;
        frameRequest.instance = _instance.borrow();
        frameRequest.clipToken = renderClipToken;
        frameRequest.frameTime = args.time;
        frameRequest.frameRate = getFrameRate();
        proc.setPrintFrameRequest(frameRequest);
    } else {
        JuicerProcessor::DirectFrameRequest frameRequest{};
        frameRequest.renderState = &prepared;
        frameRequest.framePolicy = framePolicy;
        frameRequest.renderWindow = roi;
        frameRequest.effectsGeometry = effectsGeometry;
        frameRequest.fullFrameExtent = fullBounds;
        frameRequest.instance = _instance.borrow();
        frameRequest.clipToken = renderClipToken;
        frameRequest.frameTime = args.time;
        frameRequest.frameRate = getFrameRate();
        proc.setDirectFrameRequest(frameRequest);
    }
    proc.setGPURenderArgs(args);

    // Dispatch synchronously while the prepared state and Runtime borrows remain live.
    proc.process();
}

bool JuicerEffect::plugin_edit_suppressed() const noexcept {
    return HostCallScope::current && &HostCallScope::current->instance == &_instance;
}

void JuicerEffect::changedParam(const OFX::InstanceChangedArgs& args, const std::string& paramName) {
    const bool traceInfo = JTRACE_ENABLED(1);

    // Suppress recursion while we are programmatically setting params
    if (args.reason == OFX::eChangePluginEdit && plugin_edit_suppressed()) {
        trace_changed_param_gate(traceInfo, paramName, "changedParam suppressed for '");
        return;
    }

    bool refresh = false;
    {
        CaptureAction action(_instance, FJ_CAPTURE_CHANGE, true);
        try {
            [&]() {
                auto apply_action_then_rebuild = [&](const auto& applyFn) {
                    applyFn();
                    onParamsPossiblyChanged(paramName.c_str());
                };

                const bool userEdit = (args.reason == OFX::eChangeUserEdit);

                if (userEdit && param_name_is(paramName, JuicerParams::kCameraFilmFormatPreset)) {
                    const int presetIndex = read_choice_param_clamped(
                        _pCameraFilmFormatPreset,
                        0,
                        static_cast<int>(JuicerParams::kCameraFilmFormatPresets.size()));
                    if (presetIndex > 0) {
                        const HostEditScope suppressEvents(_instance);
                        set_double_param(
                            _pCameraFilmFormat,
                            JuicerParams::kCameraFilmFormatPresets[static_cast<std::size_t>(presetIndex - 1)]
                                .longEdgeMm);
                    }
                } else if (userEdit && param_name_is(paramName, JuicerParams::kCameraFilmFormatMm)) {
                    const HostEditScope suppressEvents(_instance);
                    registered_host_write([&]() {
                        _pCameraFilmFormatPreset->setValue(0);
                    });
                }

                if (userEdit && is_coupler_gamma_numeric_param_name(paramName)) {
                    const HostEditScope suppressEvents(_instance);
                    set_bool_param(_pCouplersGammaUseStock, false);
                }

                if (param_name_is(paramName, JuicerParams::kFilmProfileKey)) {
                    const std::string filmKey =
                        read_str_choice_param_or_empty(_pFilmProfileKey, Spektrafilm::kDefaultFilmProfileKey);
                    const Spektrafilm::ProfilePolarity polarity = capture_profile_polarity_for_key(_runtime.spektrafilm_profile_catalog(), filmKey);
                    const Spektrafilm::ScanRoute defaultRoute = Spektrafilm::default_scan_route_for_polarity(polarity);
                    const HostEditScope suppressEvents(_instance);
                    set_str_choice_param(_pScanRoute, Spektrafilm::scan_route_key(defaultRoute));
                }

                auto should_apply_grain_preset_defaults = [&]() {
                    return grain_preset_param_changed_by_user(userEdit, paramName);
                };

                auto should_apply_grain_reset_advanced_defaults = [&]() {
                    return grain_reset_advanced_param_changed_by_user(userEdit, paramName);
                };

                auto resolve_grain_preset_index_for_user_edit = [&]() {
                    return read_choice_param_clamped(_pGrainPreset, 0, 2);
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

                auto read_master_value = [&](OFX::DoubleParam* masterParam, double lo, double hi, double& master) -> bool {
                    master = read_double_param(masterParam);
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
                    const HostEditScope suppressEvents(_instance);
                    set_double3_param(advParam, values[0], values[1], values[2]);
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
                    cacheGrainMaster(masterCache, master);
                };

                struct RatioMasterUpdateBinding {
                    OFX::DoubleParam* masterParam = nullptr;
                    OFX::Double3DParam* tripletParam = nullptr;
                    const std::array<double, 3>* fallbackRatio = nullptr;
                    double* masterCache = nullptr;
                    double lo = 0.0;
                    double hi = 0.0;
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
                    [&](const char* expectedParam, OFX::DoubleParam* sourceParam, double& valueOut) -> bool {
                    if (!grain_user_edit_active(userEdit, paramName, expectedParam)) {
                        return false;
                    }
                    return read_finite_unit_interval(sourceParam, valueOut);
                };

                auto with_param_event_suppression = [&](const auto& applyFn) {
                    const HostEditScope suppressEvents(_instance);
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
                    [&](const char* expectedParam, OFX::DoubleParam* sourceParam, const auto& applyFn) {
                        double value = 0.0;
                        if (!try_read_grain_user_edit_unit(expectedParam, sourceParam, value)) {
                            return;
                        }
                        applyFn(value);
                    };

                auto write_grain_blur_dye_clouds = [&](double blurDyeClouds) {
                    with_param_event_suppression([&]() {
                        set_double_param(_pGrainBlurDyeCloudsUm, blurDyeClouds);
                    });
                };

                auto write_grain_texture_linked_values = [&](const GrainTextureLinkedValues& linked) {
                    with_param_event_suppression([&]() {
                        set_double_param(_pGrainSizeMixWeight, linked.sizeMixWeight);
                        set_double2_param(_pGrainMicroStructure, linked.microCell, linked.microSigma);
                    });
                };

                auto should_mark_grain_preset_custom = [&]() {
                    return userEdit && is_grain_preset_input_param(paramName);
                };

                auto should_process_grain_preset_custom_label = [&]() {
                    return should_mark_grain_preset_custom();
                };

                auto apply_grain_sharpness_linked_update = [&]() {
                    apply_grain_linked_unit_edit(
                        JuicerParams::kGrainSharpness,
                        _pGrainSharpness,
                        [&](double sharpness) {
                            const double blurDyeClouds = std::clamp(grain_lerp(1.40, 0.60, sharpness), 0.0, 10.0);
                            write_grain_blur_dye_clouds(blurDyeClouds);
                        });
                };

                auto apply_grain_texture_linked_update = [&]() {
                    apply_grain_linked_unit_edit(
                        JuicerParams::kGrainTexture,
                        _pGrainTexture,
                        [&](double texture) {
                            const GrainTextureLinkedValues linked = compute_grain_texture_linked_values(texture);
                            write_grain_texture_linked_values(linked);
                        });
                };

                auto apply_grain_linked_updates = [&]() {
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
                    apply_ratio_master(
                        ratioBinding.masterParam,
                        ratioBinding.tripletParam,
                        *ratioBinding.fallbackRatio,
                        *ratioBinding.masterCache,
                        ratioBinding.lo,
                        ratioBinding.hi);
                }
                apply_grain_linked_updates();
                std::string routeDiagnostic;
                std::string gammaFilmProfileKey;
                Spektrafilm::ScanRoute diffusionRoute{};
                Spektrafilm::ScanRoute gammaRoute{};
                if (!resolveDiffusionControlRoute(diffusionRoute, routeDiagnostic) ||
                    !resolveGammaControlRoute(gammaRoute, gammaFilmProfileKey, routeDiagnostic)) {
                    CaptureAction::current->fail(FJ_CAPTURE_INVALID_CONTROLS, routeDiagnostic.c_str());
                    return;
                }
                updateDiffusionControlState(diffusionRoute);
                updateGammaControlState(gammaRoute, gammaFilmProfileKey);
                updateSpectralControlState();
                onParamsPossiblyChanged(paramName.c_str());
            }();
        } catch (const std::bad_alloc&) {
            action.fail(FJ_CAPTURE_ALLOCATION_FAILURE, "host parameter action allocation failed");
            throw;
        } catch (...) {
            action.fail(FJ_CAPTURE_HOST_FAILURE, "host parameter action failed");
            throw;
        }
        refresh = action.refresh;
    }
    if (refresh) {
        acquire_render_capture(FJ_CAPTURE_REFRESH);
    }
}

bool JuicerEffect::snapshotParams(
    ParamSnapshot& out,
    std::string& outDiagnostic) const {
    ParamSnapshot P;
    outDiagnostic.clear();

    ScatterHalationRawControls rawControls{};
    rawControls.active = read_bool_param(_pHalationActive);
    if (rawControls.active) {
        rawControls.scatterAmount = read_double_param(_pHalationScatterAmount);
        rawControls.scatterSpatialScale =
            read_double_param(_pHalationScatterSpatialScale);
        rawControls.halationAmount = read_double_param(_pHalationAmount);
        rawControls.halationSpatialScale =
            read_double_param(_pHalationSpatialScale);
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
    if (!read_profile_snapshot_choices(_runtime.spektrafilm_profile_catalog(), profileChoiceParams, P, outDiagnostic)) {
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
    P.glareActive = read_bool_param(_pGlareActive);
    P.glarePercent = read_sanitized_unit_float(_pGlarePercent, 0.03f);
    P.glareRoughness = read_sanitized_unit_float(_pGlareRoughness, 0.7f);
    P.glareBlurSigmaPx =
        read_sanitized_0_to_10_float(_pGlareBlurSigmaPx, 0.5f);
    read_input_snapshot_values(
        _pInputColorSpace,
        _pInputCctfDecoding,
        _pInputCompression,
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
    if (!read_coupler_snapshot_values(
            _pCouplersActive,
            _pCouplersAmount,
            _pCouplersInhibitionSameLayer,
            _pCouplersInhibitionInterlayer,
            _pCouplersDiffusionSizeUm,
            _pCouplersLangmuirDonorKRgb,
            _pCouplersLangmuirReceiverKRgb,
            _pCouplersDiffusionTailUm,
            _pCouplersDiffusionTailWeight,
            _pCouplersGammaUseStock,
            _pCouplersGammaSameLayerRgb,
            _pCouplersGammaInterlayerRToGb,
            _pCouplersGammaInterlayerGToRb,
            _pCouplersGammaInterlayerBToRg,
            Spektrafilm::scan_route_metadata(P.scanRoute).capturePolarity,
            P,
            outDiagnostic)) {
        return false;
    }
    read_scanner_snapshot_values(
        _pScannerLensBlur,
        _pScannerUnsharp,
        _pScannerBlackCorrection,
        _pScannerWhiteCorrection,
        _pScannerBlackLevel,
        _pScannerWhiteLevel,
        _pScannerLutResolution,
        P);
    read_output_snapshot_values(
        _pOutputColorSpace,
        _pOutputCctfEncoding,
        _pOutputGamutCompression,
        P);
    out = std::move(P);
    return true;
}

void JuicerEffect::acquire_render_capture(std::uint32_t mode) {
    bool refresh = false;
    {
        CaptureAction action(_instance, mode);
        if (!action.ticket) {
            return;
        }
        try {
            if (mode == FJ_CAPTURE_INITIAL) {
                applyDirGammaProfileDefaults();
            }
            finish_capture();
        } catch (const std::bad_alloc&) {
            if (action.fail(FJ_CAPTURE_ALLOCATION_FAILURE, "host parameter capture allocation failed") != FJ_UPDATE_IGNORED || action.physicalWriteFailed) {
                throw;
            }
        } catch (...) {
            if (action.fail(FJ_CAPTURE_HOST_FAILURE, "host parameter capture failed") != FJ_UPDATE_IGNORED || action.physicalWriteFailed) {
                throw;
            }
        }
        refresh = action.refresh;
    }
    if (refresh) {
        acquire_render_capture(FJ_CAPTURE_REFRESH);
    }
}

void JuicerEffect::finish_capture() {
    CaptureAction& action = *CaptureAction::current;
    FjErrorBuffer error{nullptr, 0, 0};
    FjSnapshotStart start{};
    require_instance_success(_instance.begin_snapshot(action.ticket, start, error));
    if (start.kind == FJ_SNAPSHOT_DEFERRED || start.kind == FJ_SNAPSHOT_IGNORED) {
        action.armed = false;
        action.completion = start.kind == FJ_SNAPSHOT_DEFERRED ? FJ_UPDATE_DEFERRED : FJ_UPDATE_IGNORED;
        return;
    }
    if (start.kind != FJ_SNAPSHOT_READY) {
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
    ParamSnapshot params;
    std::string diagnostic;
    if (!snapshotParams(params, diagnostic)) {
        action.fail(FJ_CAPTURE_INVALID_CONTROLS, diagnostic.c_str());
        return;
    }
    // SF_TEMP_BRIDGE_CAPTURED_INPUT_IDENTITY: S6 moves this definition with capture.
    const std::uint64_t identity = hash_params(params);
    std::uint32_t update = FJ_UPDATE_NONE;
    const FjStatus status = _instance.install(action.ticket, start.write_revision, params, identity, update, error);
    if (update == FJ_UPDATE_APPLIED || update == FJ_UPDATE_IGNORED || update == FJ_UPDATE_DEFERRED) {
        action.armed = false;
        action.completion = update;
    }
    require_instance_success(status);
}

void JuicerEffect::onParamsPossiblyChanged(const char* changedNameOrNull) {
    CaptureAction& action = *CaptureAction::current;
    if (action.initial) {
        applyDirGammaProfileDefaults();
        finish_capture();
        return;
    }
    const bool filmProfileChanged = param_name_is(changedNameOrNull, JuicerParams::kFilmProfileKey);
    const bool stockGammaEnabled = param_name_is(changedNameOrNull, JuicerParams::kDirCouplersGammaUseStock) &&
                                   read_bool_param(_pCouplersGammaUseStock);
    if (filmProfileChanged || stockGammaEnabled) {
        applyDirGammaProfileDefaults();
    }
    finish_capture();
}
