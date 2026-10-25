#pragma once

#include <nlohmann/json.hpp>

#include "DiffusionHostBehavior.h"
#include "ScatterHalation.h"

namespace JuicerTestSupport {
    bool matches_halation_frame(const std::optional<ScatterHalationFrameDescriptor>&, const nlohmann::json& expected);
    bool prepare_halation_frame(
        const ScatterHalationOpticsRecipe& recipe, float pixelSizeUm, std::optional<ScatterHalationFrameDescriptor>& out, std::string& diagnostic);
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
