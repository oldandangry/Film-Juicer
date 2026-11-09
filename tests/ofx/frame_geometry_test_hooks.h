#pragma once

#include <array>

#include "FilmEffectsFrameDescriptors.h"
#include "RenderValues.h"
#include "juicer_cuda_frame_values.h"

namespace JuicerFrame::FrameGeometryTest {
    struct Observation {
        std::array<unsigned, 3> phases{};
        unsigned count = 0;
        Spektrafilm::DiffusionFrameDomain fullDomain{};
        float pixelSizeUm = 0;
        Spektrafilm::FilmJuicerEffectsGeometry effects{};
    };
    struct HalationObservation {
        unsigned count = 0;
        float pixelSizeUm = 0;
        bool success = false;
        std::optional<ScatterHalationFrameDescriptor> descriptor;
    };
    struct DiffusionObservation {
        unsigned count = 0;
        double pixelSizeUm = 0;
        Spektrafilm::ScanRoute route = Spektrafilm::kDefaultScanRoute;
        Spektrafilm::DiffusionFrameDomain fullDomain{};
        bool success = false;
        std::optional<Spektrafilm::DiffusionFrameSetDescriptor> descriptor;
    };
    struct PreparedPolicyObservation {
        unsigned count = 0;
        std::uint32_t route = 0;
        FjPreparedFramePolicy policy{};
    };
    struct InstanceRetirementObservation {
        unsigned count = 0;
        bool identityLive = false;
        std::uint64_t token = 0;
        FjStatus status{};
    };
    InstanceRetirementObservation instance_retirement_snapshot();
    void reset_instance_retirement();
    void reset();
    Observation snapshot();
    HalationObservation halation_snapshot();
    DiffusionObservation diffusion_snapshot();
    PreparedPolicyObservation prepared_policy_snapshot();
} // namespace JuicerFrame::FrameGeometryTest
