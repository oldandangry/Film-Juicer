#pragma once

#include <nlohmann/json.hpp>

#include "DiffusionHostBehavior.h"
#include "ScatterHalation.h"
#include "RenderRecipe.h"
#include "FilmEffectsFrameDescriptors.h"

namespace JuicerTestSupport {
    bool prepare_effects_frame(const Spektrafilm::FilmJuicerEffectsFrameDescriptorInput&, Spektrafilm::FilmJuicerEffectsFrameDescriptor&, std::string&);
    Spektrafilm::FilmJuicerEffectsFrameDescriptorInput effects_frame_input(const nlohmann::json&, Spektrafilm::FilmJuicerEffectsRecipe&);
    nlohmann::json effects_frame_fields(const Spektrafilm::FilmJuicerEffectsFrameDescriptor&);

    // Literal characterization only; existing production recipes go directly to prepare_grain_frame.
    Spektrafilm::VisualGrainFrameDescriptorInput grain_frame_input(const nlohmann::json&, Spektrafilm::VisualGrainRecipe&, Spektrafilm::FilmDevelopRecipe&);
    bool prepare_grain_frame(const Spektrafilm::VisualGrainFrameDescriptorInput&, Spektrafilm::VisualGrainFrameDescriptor&, std::string&);
    nlohmann::json grain_frame_fields(const Spektrafilm::VisualGrainFrameDescriptor&);
    nlohmann::json dir_frame_fields(const SpatialDirDescriptor&);
    bool prepare_dir_frame(const DirCouplersRecipe&, float pixelSizeUm, Spektrafilm::DirFrameExtent renderExtent, Spektrafilm::DirFrameExtent fullFrameExtent, const char* traceRouteLabel, SpatialDirDescriptor&);
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
