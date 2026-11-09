#pragma once

#include <optional>

#include "DiffusionHostBehavior.h"
#include "RenderValues.h"

namespace JuicerCuda {
    struct Failure;
}

namespace JuicerDiffusion {
    bool prepare_sample(const Spektrafilm::DiffusionFilterResolvedParameters&, double pixelSizeUm, int width, int height, Spektrafilm::DiffusionPsfSampleDescriptor&, JuicerCuda::Failure&);
    bool prepare_frame(const SpatialOptics&, Spektrafilm::ScanRoute, double pixelSizeUm, Spektrafilm::DiffusionFrameDomain, std::optional<Spektrafilm::DiffusionFrameSetDescriptor>&, JuicerCuda::Failure&);
} // namespace JuicerDiffusion

namespace Spektrafilm {
    struct DiffusionFilterAuthoredControls {
        bool active = false;
        DiffusionFilterFamily family = DiffusionFilterFamily::BlackProMist;
        double strength = 0.5;
        double spatialScale = 1.0;
        double haloWarmth = 0.0;
        double coreIntensity = 1.0;
        double coreSize = 1.0;
        double haloIntensity = 1.0;
        double haloSize = 1.0;
        double bloomIntensity = 1.0;
        double bloomSize = 1.0;
    };

    std::uint64_t hash_diffusion_authored_controls(
        const DiffusionFilterAuthoredControls& controls);
    bool build_diffusion_psf_sample_descriptor(
        const DiffusionFilterResolvedParameters& resolved,
        double pixelSizeUm,
        int fullFrameWidth,
        int fullFrameHeight,
        DiffusionPsfSampleDescriptor& out,
        std::string& diagnostic);

} // namespace Spektrafilm
