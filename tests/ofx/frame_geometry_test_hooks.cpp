#include "frame_geometry_test_hooks.h"

#include <stdexcept>

namespace JuicerFrame::FrameGeometryTest {
    namespace {
        thread_local Observation observed;
        void phase(unsigned value) {
            if (observed.count >= observed.phases.size())
                throw std::runtime_error("frame geometry observation exceeds three phases");
            observed.phases[observed.count++] = value;
        }
    } // namespace
    void reset() {
        observed = {};
    }
    Observation snapshot() {
        return observed;
    }
    void full_domain(const Spektrafilm::DiffusionFrameDomain& domain) {
        phase(1);
        observed.fullDomain = domain;
    }
    void scalar_pixel(float pixel) {
        phase(2);
        observed.pixelSizeUm = pixel;
    }
    void effects_domain(const Spektrafilm::FilmJuicerEffectsGeometry& geometry) {
        phase(3);
        observed.effects = geometry;
    }
} // namespace JuicerFrame::FrameGeometryTest
