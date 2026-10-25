#pragma once

#include <array>

#include "FilmEffectsFrameDescriptors.h"
#include "RenderRecipe.h"

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
    void reset();
    Observation snapshot();
    HalationObservation halation_snapshot();
    DiffusionObservation diffusion_snapshot();
    void diffusion_frame(double pixelSizeUm, Spektrafilm::ScanRoute, Spektrafilm::DiffusionFrameDomain, bool success, const std::optional<Spektrafilm::DiffusionFrameSetDescriptor>&);
    void halation_frame(float pixelSizeUm, bool success, const std::optional<ScatterHalationFrameDescriptor>&);
    void full_domain(const Spektrafilm::DiffusionFrameDomain&);
    void scalar_pixel(float);
    void effects_domain(const Spektrafilm::FilmJuicerEffectsGeometry&);
} // namespace JuicerFrame::FrameGeometryTest
