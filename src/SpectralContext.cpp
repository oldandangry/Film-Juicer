#include "SpectralContext.h"
#include "Logging.h"

#include <string>

namespace Spectral {

    namespace {
        struct SpectralMutationTlsState {
            int depth = 0;
            SpectralMutationStage stage = SpectralMutationStage::None;
            const char* owner = nullptr;
        };

        thread_local SpectralMutationTlsState gSpectralMutationTlsState{};

        void trace_spectral_mutation_scope_event(
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
    } // namespace

    const char* to_cstr(SpectralMutationStage stage) noexcept {
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

    SpectralMutationScope::SpectralMutationScope(
        SpectralMutationStage stage,
        const char* owner) noexcept
        : _stage(stage)
        , _owner(owner) {

        if (stage == SpectralMutationStage::None) {
            return;
        }

        if (gSpectralMutationTlsState.depth <= 0) {
            gSpectralMutationTlsState.depth = 1;
            gSpectralMutationTlsState.stage = stage;
            gSpectralMutationTlsState.owner = owner;
            _active = true;
            trace_spectral_mutation_scope_event("scope_enter", stage, owner);
            return;
        }

        if (gSpectralMutationTlsState.stage == stage) {
            ++gSpectralMutationTlsState.depth;
            _active = true;
            trace_spectral_mutation_scope_event("scope_reenter", stage, owner);
            return;
        }

        trace_spectral_mutation_scope_event(
            "scope_rejected_nested_stage_mismatch",
            gSpectralMutationTlsState.stage,
            gSpectralMutationTlsState.owner,
            owner);
    }

    SpectralMutationScope::~SpectralMutationScope() noexcept {
        if (!_active) {
            return;
        }

        if (gSpectralMutationTlsState.depth > 0) {
            --gSpectralMutationTlsState.depth;
        }

        if (gSpectralMutationTlsState.depth <= 0) {
            gSpectralMutationTlsState.depth = 0;
            gSpectralMutationTlsState.stage = SpectralMutationStage::None;
            gSpectralMutationTlsState.owner = nullptr;
        }

        trace_spectral_mutation_scope_event("scope_exit", _stage, _owner);
    }

    bool spectral_mutation_scope_active() noexcept {
        return gSpectralMutationTlsState.depth > 0 &&
            gSpectralMutationTlsState.stage != SpectralMutationStage::None;
    }

    SpectralMutationStage spectral_mutation_stage() noexcept {
        if (!spectral_mutation_scope_active()) {
            return SpectralMutationStage::None;
        }
        return gSpectralMutationTlsState.stage;
    }

    bool require_spectral_mutation_scope(const char* action) noexcept {
        if (spectral_mutation_scope_active()) {
            return true;
        }

        trace_spectral_mutation_scope_event(
            "mutation_guard_violation",
            SpectralMutationStage::None,
            nullptr,
            action);
        return false;
    }

    SpectralContext& context() {
        static SpectralContext ctx{};
        return ctx;
    }

    void mark_mixing_dirty() {
        if (!require_spectral_mutation_scope("mark_mixing_dirty")) {
            return;
        }
        context().precomputeStatus.mixVersion.fetch_add(1, std::memory_order_acq_rel);
    }

    void mark_spectral_tables_dirty() {
        if (!require_spectral_mutation_scope("mark_spectral_tables_dirty")) {
            return;
        }
        context().precomputeStatus.dirty.store(true, std::memory_order_release);
    }

    void increment_illum_version() {
        if (!require_spectral_mutation_scope("increment_illum_version")) {
            return;
        }
        context().precomputeStatus.illumVersion.fetch_add(1, std::memory_order_acq_rel);
    }

    void increment_shape_version() {
        if (!require_spectral_mutation_scope("increment_shape_version")) {
            return;
        }
        context().precomputeStatus.shapeVersion.fetch_add(1, std::memory_order_acq_rel);
    }

} // namespace Spectral
