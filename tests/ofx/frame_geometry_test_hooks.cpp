#include "frame_geometry_test_hooks.h"

#include <algorithm>
#include <stdexcept>

#include "../ffi/juicer_test_api.h"

namespace JuicerFrame::FrameGeometryTest {
    namespace {
        FjOfxFrameObservation observe() {
            FjOfxFrameObservation value{};
            if (fj_test_ofx_frame_observe(&value) != 1)
                throw std::runtime_error("cannot inspect Rust callback frame stages");
            return value;
        }
    } // namespace
    InstanceRetirementObservation instance_retirement_snapshot() {
        FjOfxRetirementObservation value{};
        if (fj_test_ofx_retirement_observe(&value) != 1)
            throw std::runtime_error("cannot inspect OFX instance retirement");
        return {value.count, value.identity_live != 0, value.token, value.status};
    }
    void reset_instance_retirement() {
        fj_test_ofx_retirement_reset();
    }
    void reset() {
        fj_test_ofx_frame_reset();
    }
    Observation snapshot() {
        const auto value = observe();
        Observation result{};
        std::copy(std::begin(value.phases), std::end(value.phases), result.phases.begin());
        result.count = value.count;
        result.fullDomain = {value.full_frame.x, value.full_frame.y, value.full_frame.width, value.full_frame.height};
        result.pixelSizeUm = value.policy.geometry.pixel_size_um;
        if (value.effects_count != 0) {
            const auto& source = value.effects;
            result.effects = {{source.pixel_definition.x, source.pixel_definition.y, source.pixel_definition.width, source.pixel_definition.height},
                              source.canonical_x,
                              source.canonical_y,
                              source.canonical_width,
                              source.canonical_height,
                              source.scale_x,
                              source.scale_y,
                              source.pixel_aspect_ratio};
        }
        return result;
    }
    HalationObservation halation_snapshot() {
        HalationObservation result{};
        result.count = observe().halation_transport_calls;
        return result;
    }
    DiffusionObservation diffusion_snapshot() {
        DiffusionObservation result{};
        result.count = observe().diffusion_transport_calls;
        return result;
    }
    PreparedPolicyObservation prepared_policy_snapshot() {
        const auto value = observe();
        return {value.policy_count, value.route, value.policy};
    }
} // namespace JuicerFrame::FrameGeometryTest
