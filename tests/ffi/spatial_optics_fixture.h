#pragma once

#include "DiffusionHostBehavior.h"
#include "ScatterHalation.h"

namespace JuicerTestSupport {
    bool prepare_diffusion_filter(
        const Spektrafilm::DiffusionFilterAuthoredControls& authored,
        Spektrafilm::DiffusionFilterResolvedParameters& out,
        std::string& diagnostic);
    bool prepare_scatter_halation(
        const Spektrafilm::ScatterHalationControls& controls,
        const Profiles::ProfileDigest& profile,
        ScatterHalationOpticsRecipe& out,
        std::string& diagnostic);
} // namespace JuicerTestSupport
