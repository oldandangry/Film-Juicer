#pragma once

#include <optional>

#include "DiffusionHostBehavior.h"
#include "RenderRecipe.h"

namespace JuicerCuda {
    struct Failure;
}

namespace JuicerDiffusion {
    // SF_TEMP_BRIDGE_DIFFUSION_FRAME: sample/frame projections retire with the
    // last native preparation caller, no later than S6.D. The native-resource
    // component projection retires with S4.E's resolved execution conversion.
    bool prepare_sample(const Spektrafilm::DiffusionFilterResolvedParameters&, double pixelSizeUm, int width, int height, Spektrafilm::DiffusionPsfSampleDescriptor&, JuicerCuda::Failure&);
    bool prepare_components(const Spektrafilm::DiffusionPsfSampleDescriptor&, Spektrafilm::DiffusionPsfComponents&, JuicerCuda::Failure&);
    bool prepare_frame(const SpatialOptics&, Spektrafilm::ScanRoute, double pixelSizeUm, Spektrafilm::DiffusionFrameDomain, std::optional<Spektrafilm::DiffusionFrameSetDescriptor>&, JuicerCuda::Failure&);
} // namespace JuicerDiffusion
