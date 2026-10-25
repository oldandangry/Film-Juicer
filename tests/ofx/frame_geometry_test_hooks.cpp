#include "frame_geometry_test_hooks.h"

#include <stdexcept>

namespace JuicerFrame::FrameGeometryTest {
    namespace {
        thread_local Observation observed;
        thread_local HalationObservation halation;
        void phase(unsigned value) {
            if (observed.count >= observed.phases.size())
                throw std::runtime_error("frame geometry observation exceeds three phases");
            observed.phases[observed.count++] = value;
        }
    } // namespace
    void reset() {
        observed = {};
        halation = {};
    }
    Observation snapshot() {
        return observed;
    }
    HalationObservation halation_snapshot() {
        return halation;
    }
    void halation_frame(float pixelSizeUm, bool success, const std::optional<ScatterHalationFrameDescriptor>& descriptor) {
        halation = {halation.count + 1, pixelSizeUm, success, descriptor};
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
