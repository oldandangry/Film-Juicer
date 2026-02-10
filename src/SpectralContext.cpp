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

} // namespace Spectral
