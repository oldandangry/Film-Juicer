#pragma once

#include <array>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "DiffusionHostBehavior.h"
#include "RenderControls.h"
#include "RenderValues.h"
#include "ScatterHalation.h"
#include "FilmEffectsFrameDescriptors.h"
#include "juicer_cuda_frame_values.h"
#include "juicer_test_api.h"

struct FjDirFrameInput;

namespace JuicerTestSupport {
    bool prepare_effects_frame(const Spektrafilm::FilmJuicerEffectsFrameDescriptorInput&, Spektrafilm::FilmJuicerEffectsFrameDescriptor&, std::string&);
    Spektrafilm::FilmJuicerEffectsFrameDescriptorInput effects_frame_input(const nlohmann::json&, Spektrafilm::FilmJuicerEffectsRecipe&);
    nlohmann::json effects_frame_fields(const Spektrafilm::FilmJuicerEffectsFrameDescriptor&);

    // Storage is populated before bind() creates a synchronous raw borrow.
    struct GrainFrameInput {
        std::array<std::array<std::vector<float>, 3>, 3> layers;
        FjGrainFrameInput frame{};
        FjGrainCoefficients coefficients{};

        FjGrainFrameInput bind() const noexcept;
    };
    GrainFrameInput grain_frame_input(const nlohmann::json&, const FjGrainCoefficients& seed = {});
    // Returned layer spans borrow the complete state's retained owner.
    FjGrainFrameInput grain_frame_input(const FjPreparedStateView&, const FjGrainFrameInput& facts);
    FjGrainCoefficients grain_coefficients(const FjVisualGrainRecipe&);
    bool prepare_grain_frame(const GrainFrameInput&, Spektrafilm::VisualGrainFrameDescriptor&, std::string&);

    nlohmann::json grain_frame_fields(const Spektrafilm::VisualGrainFrameDescriptor&);
    nlohmann::json dir_frame_fields(const SpatialDirDescriptor&);
    bool prepare_dir_frame(const FjDirFrameInput&, const char* traceRouteLabel, SpatialDirDescriptor&);
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
        const std::array<float, 3>& firstSigmaUm,
        const std::array<float, 3>& primaryAmount,
        ScatterHalationOpticsRecipe& out,
        std::string& diagnostic);
} // namespace JuicerTestSupport
