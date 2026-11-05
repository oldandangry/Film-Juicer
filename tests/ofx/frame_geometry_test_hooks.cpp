#include "frame_geometry_test_hooks.h"

#include <stdexcept>

namespace JuicerFrame::FrameGeometryTest {
    namespace {
        thread_local Observation observed;
        thread_local HalationObservation halation;
        thread_local DiffusionObservation diffusion;
        thread_local PreparedPolicyObservation prepared;
        void phase(unsigned value) {
            if (observed.count >= observed.phases.size())
                throw std::runtime_error("frame geometry observation exceeds three phases");
            observed.phases[observed.count++] = value;
        }
    } // namespace
    void reset() {
        observed = {};
        halation = {};
        diffusion = {};
        prepared = {};
    }
    Observation snapshot() {
        return observed;
    }
    HalationObservation halation_snapshot() {
        return halation;
    }
    DiffusionObservation diffusion_snapshot() {
        return diffusion;
    }
    PreparedPolicyObservation prepared_policy_snapshot() {
        return prepared;
    }
    void prepared_policy(const FjPreparedFramePolicy& policy, std::uint32_t route) {
        if (observed.count != 1 || prepared.count != 0)
            throw std::runtime_error("prepared policy observation must follow full-domain observation exactly once");
        prepared = {1, route, policy};
    }
    void diffusion_frame(double pixelSizeUm, Spektrafilm::ScanRoute route, Spektrafilm::DiffusionFrameDomain domain, bool success, const std::optional<Spektrafilm::DiffusionFrameSetDescriptor>& descriptor) {
        if (observed.count != 2 || halation.count != 0)
            throw std::runtime_error("diffusion observation must follow scalar geometry and precede halation");
        diffusion = {diffusion.count + 1, pixelSizeUm, route, domain, success, descriptor};
    }
    void halation_frame(float pixelSizeUm, bool success, const std::optional<ScatterHalationFrameDescriptor>& descriptor) {
        if (diffusion.count != 1 || !diffusion.success)
            throw std::runtime_error("halation observation must follow successful diffusion construction");
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
        if (prepared.count != 1 || observed.count != 2)
            throw std::runtime_error("effects observation must follow prepared frame policy and pixel observation");
        phase(3);
        observed.effects = geometry;
    }
} // namespace JuicerFrame::FrameGeometryTest
