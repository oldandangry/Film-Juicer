#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>

#include "RenderRecipe.h"

namespace Spektrafilm {

    inline constexpr std::uint32_t kDiffusionFrameDescriptorSchemaVersion = 1;
    inline constexpr std::uint32_t kDiffusionCandidatePolicyVersion = 1;

    enum class DiffusionLinearStage : std::uint8_t {
        CameraFilmLinear,
        EnlargerPrintLinear
    };

    enum class DiffusionSemanticPlaneRole : std::uint8_t {
        RedSensitiveExposure,
        GreenSensitiveExposure,
        BlueSensitiveExposure,
        SequentialOutputAuxiliary
    };

    inline constexpr std::array<DiffusionSemanticPlaneRole, 4>
        kDiffusionSemanticPlaneRoles{{DiffusionSemanticPlaneRole::RedSensitiveExposure,
                                      DiffusionSemanticPlaneRole::GreenSensitiveExposure,
                                      DiffusionSemanticPlaneRole::BlueSensitiveExposure,
                                      DiffusionSemanticPlaneRole::SequentialOutputAuxiliary}};

    struct DiffusionFrameDomain {
        int originX = 0;
        int originY = 0;
        int width = 0;
        int height = 0;

        friend bool operator==(
            const DiffusionFrameDomain&,
            const DiffusionFrameDomain&) = default;
    };

    struct DiffusionStageFrameDescriptor {
        ScanRoute route = kDefaultScanRoute;
        DiffusionLinearStage stage = DiffusionLinearStage::CameraFilmLinear;
        std::uint64_t recipeComponentHash = 0;
        double scatterFraction = 0.0;
        double pixelSizeUm = 0.0;
        DiffusionFrameDomain fullFrame;
        int radiusPixels = 0;
        DiffusionPsfSampleDescriptor sample;
        std::uint64_t hash = 0;
    };

    struct DiffusionFrameSetDescriptor {
        ScanRoute route = kDefaultScanRoute;
        DiffusionFrameDomain fullFrame;
        std::optional<DiffusionStageFrameDescriptor> camera;
        std::optional<DiffusionStageFrameDescriptor> enlarger;
        int maximumRadiusPixels = 0;
        std::uint32_t candidatePolicyVersion = kDiffusionCandidatePolicyVersion;
        std::array<DiffusionSemanticPlaneRole, 4> planeRoles =
            kDiffusionSemanticPlaneRoles;
        std::uint64_t workspaceBytes = 0;
        std::uint64_t hash = 0;
    };

    bool build_diffusion_frame_set_descriptor(
        const SpatialOptics& optics,
        ScanRoute route,
        double pixelSizeUm,
        DiffusionFrameDomain fullFrame,
        std::optional<DiffusionFrameSetDescriptor>& out,
        std::string& diagnostic);

    bool diffusion_full_frame_matches(
        const DiffusionFrameSetDescriptor& descriptor,
        DiffusionFrameDomain renderWindow,
        DiffusionFrameDomain sourceBounds,
        DiffusionFrameDomain requestFullFrame) noexcept;

} // namespace Spektrafilm
