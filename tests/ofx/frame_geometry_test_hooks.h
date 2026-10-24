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
    void reset();
    Observation snapshot();
    void full_domain(const Spektrafilm::DiffusionFrameDomain&);
    void scalar_pixel(float);
    void effects_domain(const Spektrafilm::FilmJuicerEffectsGeometry&);
} // namespace JuicerFrame::FrameGeometryTest
