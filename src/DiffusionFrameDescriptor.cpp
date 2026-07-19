#include "DiffusionFrameDescriptor.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

namespace {

    constexpr std::uint64_t kFnvOffset = 0xcbf29ce484222325ULL;
    constexpr std::uint64_t kFnvPrime = 0x100000001b3ULL;

    void fail(std::string& diagnostic, std::string_view field) {
        diagnostic = "InvalidDiffusionFrameDescriptor field=";
        diagnostic.append(field);
    }

    void hash_byte(std::uint64_t& hash, std::uint8_t value) {
        hash ^= static_cast<std::uint64_t>(value);
        hash *= kFnvPrime;
    }

    void hash_u32_le(std::uint64_t& hash, std::uint32_t value) {
        for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
            hash_byte(
                hash,
                static_cast<std::uint8_t>((value >> (byte * 8)) & 0xffu));
        }
    }

    void hash_u64_le(std::uint64_t& hash, std::uint64_t value) {
        for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
            hash_byte(
                hash,
                static_cast<std::uint8_t>((value >> (byte * 8)) & 0xffu));
        }
    }

    void hash_tag(std::uint64_t& hash, std::string_view tag) {
        for (const char value : tag) {
            hash_byte(hash, static_cast<std::uint8_t>(value));
        }
        hash_byte(hash, 0);
    }

    void hash_double(std::uint64_t& hash, double value) {
        if (value == 0.0) {
            value = 0.0;
        }
        hash_u64_le(hash, std::bit_cast<std::uint64_t>(value));
    }

    void hash_domain(
        std::uint64_t& hash,
        const Spektrafilm::DiffusionFrameDomain& domain) {
        hash_u32_le(hash, static_cast<std::uint32_t>(domain.originX));
        hash_u32_le(hash, static_cast<std::uint32_t>(domain.originY));
        hash_u32_le(hash, static_cast<std::uint32_t>(domain.width));
        hash_u32_le(hash, static_cast<std::uint32_t>(domain.height));
    }

    bool valid_route(Spektrafilm::ScanRoute route) {
        using Spektrafilm::ScanRoute;
        switch (route) {
            case ScanRoute::NegativeDirectScan:
            case ScanRoute::NegativePrintScan:
            case ScanRoute::PositiveDirectScan:
            case ScanRoute::PositivePrintScan:
                return true;
            default:
                return false;
        }
    }

    bool checked_multiply(
        std::uint64_t left,
        std::uint64_t right,
        std::uint64_t& out) {
        if (right != 0 && left > std::numeric_limits<std::uint64_t>::max() / right) {
            return false;
        }
        out = left * right;
        return true;
    }

    bool build_stage_descriptor(
        const DiffusionFilterOpticsRecipe& component,
        Spektrafilm::ScanRoute route,
        Spektrafilm::DiffusionLinearStage stage,
        double pixelSizeUm,
        Spektrafilm::DiffusionFrameDomain fullFrame,
        Spektrafilm::DiffusionStageFrameDescriptor& out,
        std::string& diagnostic) {
        const bool cameraStage =
            stage == Spektrafilm::DiffusionLinearStage::CameraFilmLinear;
        const char* inactiveField = cameraStage
                                        ? "camera_recipe_component_hash"
                                        : "enlarger_recipe_component_hash";
        const char* resolvedField = cameraStage
                                        ? "camera_resolved_hash"
                                        : "enlarger_resolved_hash";
        const char* scatterField = cameraStage
                                       ? "camera_scatter_fraction"
                                       : "enlarger_scatter_fraction";
        const char* domainField = cameraStage
                                      ? "camera_linear_domain"
                                      : "enlarger_linear_domain";
        const char* sampleField = cameraStage ? "camera_sample" : "enlarger_sample";
        const char* stageHashField = cameraStage ? "camera_hash" : "enlarger_hash";

        if (component.hash == 0) {
            fail(diagnostic, inactiveField);
            return false;
        }
        if (!component.resolved.active || component.resolved.hash == 0) {
            fail(diagnostic, resolvedField);
            return false;
        }
        if (!std::isfinite(component.resolved.scatterFraction) ||
            component.resolved.scatterFraction <= 0.0 ||
            component.resolved.scatterFraction > 1.0) {
            fail(diagnostic, scatterField);
            return false;
        }
        const Spektrafilm::SpatialOpticsDomain expectedDomain =
            cameraStage ? Spektrafilm::SpatialOpticsDomain::FilmLinearExposure
                        : Spektrafilm::SpatialOpticsDomain::PrintLinearExposure;
        if (component.policy.domain != expectedDomain) {
            fail(diagnostic, domainField);
            return false;
        }

        Spektrafilm::DiffusionPsfSampleDescriptor sample{};
        std::string sampleDiagnostic;
        if (!Spektrafilm::build_diffusion_psf_sample_descriptor(
                component.resolved,
                pixelSizeUm,
                fullFrame.width,
                fullFrame.height,
                sample,
                sampleDiagnostic) ||
            sample.hash == 0 || sample.radiusPixels <= 0) {
            fail(diagnostic, sampleField);
            return false;
        }

        out = Spektrafilm::DiffusionStageFrameDescriptor{};
        out.route = route;
        out.stage = stage;
        out.recipeComponentHash = component.hash;
        out.scatterFraction = component.resolved.scatterFraction;
        out.pixelSizeUm = pixelSizeUm;
        out.fullFrame = fullFrame;
        out.radiusPixels = sample.radiusPixels;
        out.sample = sample;

        std::uint64_t hash = kFnvOffset;
        hash_tag(hash, "diffusion-stage-frame");
        hash_u32_le(hash, Spektrafilm::kDiffusionFrameDescriptorSchemaVersion);
        hash_byte(hash, static_cast<std::uint8_t>(out.route));
        hash_byte(hash, static_cast<std::uint8_t>(out.stage));
        hash_u64_le(hash, out.recipeComponentHash);
        hash_double(hash, out.scatterFraction);
        hash_double(hash, out.pixelSizeUm);
        hash_domain(hash, out.fullFrame);
        hash_u32_le(hash, static_cast<std::uint32_t>(out.radiusPixels));
        hash_u64_le(hash, out.sample.hash);
        out.hash = hash;
        if (out.hash == 0) {
            fail(diagnostic, stageHashField);
            return false;
        }
        return true;
    }

} // namespace

namespace Spektrafilm {

    bool build_diffusion_frame_set_descriptor(
        const SpatialOptics& optics,
        ScanRoute route,
        double pixelSizeUm,
        DiffusionFrameDomain fullFrame,
        std::optional<DiffusionFrameSetDescriptor>& out,
        std::string& diagnostic) {
        out.reset();
        diagnostic.clear();

        const bool cameraActive = optics.cameraDiffusion.hash != 0;
        const bool enlargerActive = optics.enlargerDiffusion.hash != 0;
        if (!cameraActive && !enlargerActive) {
            return true;
        }
        if (!valid_route(route)) {
            fail(diagnostic, "route");
            return false;
        }
        if (fullFrame.width < 2 || fullFrame.height < 2) {
            fail(diagnostic, "full_frame_domain");
            return false;
        }
        if (!(std::isfinite(pixelSizeUm) && pixelSizeUm > 0.0)) {
            fail(diagnostic, "pixel_size_um");
            return false;
        }
        if (enlargerActive && !scan_route_is_print(route)) {
            fail(diagnostic, "enlarger_route");
            return false;
        }

        DiffusionFrameSetDescriptor descriptor{};
        descriptor.route = route;
        descriptor.fullFrame = fullFrame;
        if (cameraActive) {
            DiffusionStageFrameDescriptor camera{};
            if (!build_stage_descriptor(
                    optics.cameraDiffusion,
                    route,
                    DiffusionLinearStage::CameraFilmLinear,
                    pixelSizeUm,
                    fullFrame,
                    camera,
                    diagnostic)) {
                return false;
            }
            descriptor.camera = camera;
            descriptor.maximumRadiusPixels = camera.radiusPixels;
        }
        if (enlargerActive) {
            DiffusionStageFrameDescriptor enlarger{};
            if (!build_stage_descriptor(
                    optics.enlargerDiffusion,
                    route,
                    DiffusionLinearStage::EnlargerPrintLinear,
                    pixelSizeUm,
                    fullFrame,
                    enlarger,
                    diagnostic)) {
                return false;
            }
            descriptor.enlarger = enlarger;
            descriptor.maximumRadiusPixels =
                std::max(descriptor.maximumRadiusPixels, enlarger.radiusPixels);
        }

        std::uint64_t workspaceBytes = 0;
        if (!checked_multiply(
                static_cast<std::uint64_t>(fullFrame.width),
                static_cast<std::uint64_t>(fullFrame.height),
                workspaceBytes) ||
            !checked_multiply(workspaceBytes, 4, workspaceBytes) ||
            !checked_multiply(workspaceBytes, sizeof(float), workspaceBytes)) {
            fail(diagnostic, "workspace_bytes");
            return false;
        }
        descriptor.workspaceBytes = workspaceBytes;

        std::uint64_t hash = kFnvOffset;
        hash_tag(hash, "diffusion-frame-set");
        hash_u32_le(hash, kDiffusionFrameDescriptorSchemaVersion);
        hash_u32_le(hash, descriptor.candidatePolicyVersion);
        hash_byte(hash, static_cast<std::uint8_t>(descriptor.route));
        hash_domain(hash, descriptor.fullFrame);
        hash_byte(hash, descriptor.camera.has_value() ? 1u : 0u);
        if (descriptor.camera) {
            hash_u64_le(hash, descriptor.camera->hash);
        }
        hash_byte(hash, descriptor.enlarger.has_value() ? 1u : 0u);
        if (descriptor.enlarger) {
            hash_u64_le(hash, descriptor.enlarger->hash);
        }
        hash_u32_le(hash, static_cast<std::uint32_t>(descriptor.maximumRadiusPixels));
        for (const DiffusionSemanticPlaneRole role : descriptor.planeRoles) {
            hash_byte(hash, static_cast<std::uint8_t>(role));
        }
        hash_u64_le(hash, descriptor.workspaceBytes);
        descriptor.hash = hash;
        if (descriptor.hash == 0) {
            fail(diagnostic, "frame_set_hash");
            return false;
        }

        out = descriptor;
        return true;
    }

    bool diffusion_full_frame_matches(
        const DiffusionFrameSetDescriptor& descriptor,
        DiffusionFrameDomain renderWindow,
        DiffusionFrameDomain sourceBounds,
        DiffusionFrameDomain requestFullFrame) noexcept {
        return renderWindow == descriptor.fullFrame &&
               sourceBounds == descriptor.fullFrame &&
               requestFullFrame == descriptor.fullFrame;
    }

} // namespace Spektrafilm
