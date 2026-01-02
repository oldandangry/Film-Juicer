#include "SpectralContext.h"

namespace Spectral {

    SpectralContext& context() {
        static SpectralContext ctx{};
        return ctx;
    }

    void mark_mixing_dirty() {
        context().precomputeStatus.mixVersion.fetch_add(1, std::memory_order_acq_rel);
    }

    void mark_spectral_tables_dirty() {
        context().precomputeStatus.dirty.store(true, std::memory_order_release);
    }

    void increment_illum_version() {
        context().precomputeStatus.illumVersion.fetch_add(1, std::memory_order_acq_rel);
    }

    void increment_shape_version() {
        context().precomputeStatus.shapeVersion.fetch_add(1, std::memory_order_acq_rel);
    }

    DirRuntimeSnapshot get_dir_runtime_snapshot() {
#ifdef JUICER_ENABLE_COUPLERS
        return context().dirRuntimeSnapshot.load(std::memory_order_acquire);
#else
        return context().dirRuntimeSnapshot.load(std::memory_order_relaxed);
#endif
    }

    void set_dir_runtime_snapshot(const DirRuntimeSnapshot& snap) {
#ifdef JUICER_ENABLE_COUPLERS
        context().dirRuntimeSnapshot.store(snap, std::memory_order_release);
#else
        (void)snap;
#endif
    }

} // namespace Spectral
