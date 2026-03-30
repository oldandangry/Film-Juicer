#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

#include "Logging.h"
#include "SpectralTypes.h"
#include "NpyLoader.h"

namespace Spectral {

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

        bool active() const noexcept { return _active; }

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

        std::atomic<bool> hanatosAvailable{ false };
        NpySpectraLUT hanSpectra;
        std::atomic<bool> mallettAvailable{ false };
        NpyFloat2D mallettBasis;

        std::atomic<bool> spdInit{ false };
        float sInv[9] = { 1.0f,0.0f,0.0f, 0.0f,1.0f,0.0f, 0.0f,0.0f,1.0f };
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
            const char* action = nullptr) {

            if (!JTRACE_ENABLED(2)) {
                return;
            }

            std::string msg = std::string("event=") + (event ? event : "unknown")
                + " stage=" + to_cstr(stage)
                + " depth=" + std::to_string(gSpectralMutationTlsState.depth);
            if (owner && owner[0] != '\0') {
                msg += " owner=";
                msg += owner;
            }
            if (action && action[0] != '\0') {
                msg += " action=";
                msg += action;
            }

            JTRACE_LEVEL(2, "MSPEC", msg);
        }
    } // namespace detail

    inline SpectralMutationScope::SpectralMutationScope(
        SpectralMutationStage stage,
        const char* owner) noexcept
        : _stage(stage)
        , _owner(owner) {

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
