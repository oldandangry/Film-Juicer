#pragma once

#include <nlohmann/json.hpp>

#include "DiffusionHostBehavior.h"
#include "ScatterHalation.h"
#include "RenderRecipe.h"

namespace JuicerTestSupport {
    bool matches_diffusion_frame(const std::optional<Spektrafilm::DiffusionFrameSetDescriptor>&, const nlohmann::json& expected);
    bool prepare_diffusion_sample(const Spektrafilm::DiffusionFilterResolvedParameters&, double pixelSizeUm, int width, int height, Spektrafilm::DiffusionPsfSampleDescriptor&, std::string& diagnostic);
    bool prepare_diffusion_components(const Spektrafilm::DiffusionPsfSampleDescriptor&, Spektrafilm::DiffusionPsfComponents&, std::string& diagnostic);
    bool prepare_diffusion_frame(const SpatialOptics&, Spektrafilm::ScanRoute, double pixelSizeUm, Spektrafilm::DiffusionFrameDomain, std::optional<Spektrafilm::DiffusionFrameSetDescriptor>&, std::string& diagnostic);
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
