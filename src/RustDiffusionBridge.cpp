#include "RustDiffusionBridge.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <string>
#include <string_view>

#include "Cuda/JuicerCudaFailure.h"
#include "juicer_legacy_api.h"

namespace JuicerDiffusion {
    namespace {
        bool is_positive_zero(double value) {
            return std::bit_cast<std::uint64_t>(value) == 0;
        }
        template <std::size_t N>
        bool cleared(const double (&values)[N]) {
            return std::all_of(std::begin(values), std::end(values), is_positive_zero);
        }
        bool cleared(const FjDiffusionSample& v) {
            return v.family == 0 && cleared(v.group_weights) && cleared(v.group_lambda_um) && is_positive_zero(v.warmth) &&
                   is_positive_zero(v.spatial_scale) && is_positive_zero(v.pixel_size_um) && v.radius_px == 0 && v.hash == 0;
        }
        bool cleared(const FjDiffusionComponents& v) {
            return cleared(v.lambda_px) && cleared(v.red_weights) && cleared(v.green_weights) && cleared(v.blue_weights);
        }
        bool cleared(const FjDiffusion& v) {
            return v.family == 0 && is_positive_zero(v.scatter_fraction) && cleared(v.group_weights) && cleared(v.group_lambda_um) &&
                   is_positive_zero(v.warmth) && is_positive_zero(v.spatial_scale) && is_positive_zero(v.pixel_size_um) && v.radius_px == 0 && v.sample_hash == 0 && v.hash == 0;
        }
        bool cleared(const FjDiffusionFrameSet& v) {
            return v.route == 0 && v.flags == 0 && v.full_frame.x == 0 && v.full_frame.y == 0 && v.full_frame.width == 0 && v.full_frame.height == 0 &&
                   cleared(v.camera) && cleared(v.enlarger) && v.hash == 0;
        }
        bool reject(std::string_view operation, JuicerCuda::Failure& failure) {
            JuicerCuda::set_failure(failure, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "diffusion " + std::string(operation) + " projection invalid");
            return false;
        }
        std::string_view field_text(unsigned operation, const FjDiffusionFailure& v) {
            constexpr std::array sampleFields{"", "pixel_size_um", "full_frame_extent", "family", "radius_pixels", "sample_descriptor_hash", "sample_descriptor_inactive", "group_weights", "group_centers_um", "effective_warmth", "spatial_scale", "component_expansion", "component_count", "halo_channel_weights", "component_output"};
            if (operation == 1)
                return v.stage == 0 && v.field >= 1 && v.field <= 5 ? sampleFields[v.field] : "";
            if (operation == 2)
                return v.stage == 0 && v.field >= 1 && v.field <= 14 && v.field != 2 ? sampleFields[v.field] : "";
            if (v.stage == 0) {
                switch (v.field) {
                    case 1:
                        return "pixel_size_um";
                    case 15:
                        return "route";
                    case 16:
                        return "full_frame_domain";
                    case 17:
                        return "enlarger_route";
                    case 21:
                        return "frame_set_hash";
                    default:
                        return "";
                }
            }
            if (v.stage != 1 && v.stage != 2)
                return "";
            switch (v.field) {
                case 18:
                    return v.stage == 1 ? "camera_linear_domain" : "enlarger_linear_domain";
                case 19:
                    return v.stage == 1 ? "camera_sample" : "enlarger_sample";
                case 20:
                    return v.stage == 1 ? "camera_hash" : "enlarger_hash";
                default:
                    return "";
            }
        }
        bool envelope(unsigned operation, FjStatus status, const FjDiffusionFailure& v, const FjErrorBuffer& error, const std::array<char, 256>& text) {
            constexpr std::array categories{FJ_STATUS_SUCCESS, FJ_STATUS_UNSUPPORTED_INPUT, FJ_STATUS_PREPARATION_FAILURE, FJ_STATUS_INTERNAL_FAILURE};
            constexpr std::array raw{"invalid diffusion sample arguments", "invalid diffusion components arguments", "invalid diffusion frame arguments"};
            constexpr std::array panic{"diffusion sample preparation panicked", "diffusion components preparation panicked", "diffusion frame preparation panicked"};
            if (v.reason >= categories.size() || status.category != categories[v.reason] || status.api != FJ_API_NONE || status.native_code != 0 ||
                error.data != text.data() || error.capacity != text.size() || error.length >= text.size() || text[error.length] != '\0')
                return false;
            std::string expected;
            if (v.reason == FJ_DIFFUSION_REASON_PREPARATION) {
                const auto field = field_text(operation, v);
                if (field.empty())
                    return false;
                expected = operation == 3 ? "InvalidDiffusionFrameDescriptor field=" : "InvalidDiffusionHostBehavior field=";
                expected += field;
            } else {
                if (v.stage != 0 || v.field != 0)
                    return false;
                if (v.reason == FJ_DIFFUSION_REASON_RAW)
                    expected = raw[operation - 1];
                if (v.reason == FJ_DIFFUSION_REASON_PANIC)
                    expected = panic[operation - 1];
            }
            return std::string_view(text.data(), error.length) == expected;
        }
        Spektrafilm::DiffusionPsfSampleDescriptor project(const FjDiffusionSample& v) {
            Spektrafilm::DiffusionPsfSampleDescriptor out;
            out.family = static_cast<Spektrafilm::DiffusionFilterFamily>(v.family);
            std::copy(std::begin(v.group_weights), std::end(v.group_weights), out.groupWeightsCoreHaloBloom.begin());
            std::copy(std::begin(v.group_lambda_um), std::end(v.group_lambda_um), out.groupCenterLambdaUm.begin());
            out.effectiveWarmth = v.warmth;
            out.spatialScale = v.spatial_scale;
            out.pixelSizeUm = v.pixel_size_um;
            out.radiusPixels = v.radius_px;
            out.hash = v.hash;
            return out;
        }
        FjDiffusionRecipe bind(const DiffusionFilterOpticsRecipe& v) {
            FjDiffusionRecipe out{};
            out.domain = static_cast<std::uint32_t>(v.domain);
            const auto& r = v.resolved;
            out.active = r.active ? 1u : 0u;
            out.family = static_cast<std::uint32_t>(r.family);
            out.scatter_fraction = r.scatterFraction;
            std::copy(r.groupWeightsCoreHaloBloom.begin(), r.groupWeightsCoreHaloBloom.end(), out.group_weights);
            std::copy(r.groupCenterLambdaUm.begin(), r.groupCenterLambdaUm.end(), out.group_lambda_um);
            out.warmth = r.effectiveWarmth;
            out.spatial_scale = r.spatialScale;
            out.resolved_hash = r.hash;
            out.hash = v.hash;
            return out;
        }
        Spektrafilm::DiffusionStageFrameDescriptor project(const FjDiffusion& v, Spektrafilm::DiffusionLinearStage stage) {
            FjDiffusionSample sample{};
            sample.family = v.family;
            std::copy(std::begin(v.group_weights), std::end(v.group_weights), sample.group_weights);
            std::copy(std::begin(v.group_lambda_um), std::end(v.group_lambda_um), sample.group_lambda_um);
            sample.warmth = v.warmth;
            sample.spatial_scale = v.spatial_scale;
            sample.pixel_size_um = v.pixel_size_um;
            sample.radius_px = v.radius_px;
            sample.hash = v.sample_hash;
            return {stage, v.scatter_fraction, project(sample), v.hash};
        }
        bool associated(const FjDiffusion& v, const FjDiffusionRecipe& input, bool present) {
            return present ? v.family <= 3 && v.family == input.family && v.radius_px > 0 && v.sample_hash != 0 && v.hash != 0 : cleared(v);
        }
    } // namespace

    // Keep the retained public sample builder's scalar order at this mechanical boundary.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    bool prepare_sample(const Spektrafilm::DiffusionFilterResolvedParameters& resolved, double pixelSizeUm, int width, int height, Spektrafilm::DiffusionPsfSampleDescriptor& out, JuicerCuda::Failure& failure) {
        out = {};
        failure = {};
        FjDiffusionSampleInput input{};
        input.family = static_cast<std::uint32_t>(resolved.family);
        std::copy(resolved.groupWeightsCoreHaloBloom.begin(), resolved.groupWeightsCoreHaloBloom.end(), input.group_weights);
        std::copy(resolved.groupCenterLambdaUm.begin(), resolved.groupCenterLambdaUm.end(), input.group_lambda_um);
        input.warmth = resolved.effectiveWarmth;
        input.spatial_scale = resolved.spatialScale;
        input.pixel_size_um = pixelSizeUm;
        input.width = width;
        input.height = height;
        FjDiffusionSample wire{};
        FjDiffusionFailure rejected{};
        std::array<char, 256> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        const auto status = fj_legacy_diffusion_sample_prepare(&input, &wire, &rejected, &error);
        if (!envelope(1, status, rejected, error, text))
            return reject("sample", failure);
        if (rejected.reason != 0) {
            if (!cleared(wire))
                return reject("sample", failure);
            JuicerCuda::set_failure(failure, status, std::string_view(text.data(), error.length));
            return false;
        }
        if (wire.family > 3 || wire.family != input.family || wire.radius_px <= 0 || wire.hash == 0)
            return reject("sample", failure);
        out = project(wire);
        return true;
    }

    bool prepare_components(const Spektrafilm::DiffusionPsfSampleDescriptor& sample, Spektrafilm::DiffusionPsfComponents& out, JuicerCuda::Failure& failure) {
        out = {};
        failure = {};
        FjDiffusionSample input{};
        input.family = static_cast<std::uint32_t>(sample.family);
        std::copy(sample.groupWeightsCoreHaloBloom.begin(), sample.groupWeightsCoreHaloBloom.end(), input.group_weights);
        std::copy(sample.groupCenterLambdaUm.begin(), sample.groupCenterLambdaUm.end(), input.group_lambda_um);
        input.warmth = sample.effectiveWarmth;
        input.spatial_scale = sample.spatialScale;
        input.pixel_size_um = sample.pixelSizeUm;
        input.radius_px = sample.radiusPixels;
        input.hash = sample.hash;
        FjDiffusionComponents wire{};
        FjDiffusionFailure rejected{};
        std::array<char, 256> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        const auto status = fj_legacy_diffusion_components_prepare(&input, &wire, &rejected, &error);
        if (!envelope(2, status, rejected, error, text))
            return reject("components", failure);
        if (rejected.reason != 0) {
            if (!cleared(wire))
                return reject("components", failure);
            JuicerCuda::set_failure(failure, status, std::string_view(text.data(), error.length));
            return false;
        }
        std::copy(std::begin(wire.lambda_px), std::end(wire.lambda_px), out.lambdaPixels.begin());
        std::copy(std::begin(wire.red_weights), std::end(wire.red_weights), out.redWeights.begin());
        std::copy(std::begin(wire.green_weights), std::end(wire.green_weights), out.greenWeights.begin());
        std::copy(std::begin(wire.blue_weights), std::end(wire.blue_weights), out.blueWeights.begin());
        return true;
    }

    bool prepare_frame(const SpatialOptics& optics, Spektrafilm::ScanRoute route, double pixelSizeUm, Spektrafilm::DiffusionFrameDomain domain, std::optional<Spektrafilm::DiffusionFrameSetDescriptor>& out, JuicerCuda::Failure& failure) {
        out.reset();
        failure = {};
        FjDiffusionFrameInput input{static_cast<std::uint32_t>(route), {domain.originX, domain.originY, domain.width, domain.height}, pixelSizeUm, bind(optics.cameraDiffusion), bind(optics.enlargerDiffusion)};
        FjDiffusionFrameSet wire{};
        FjDiffusionFailure rejected{};
        std::array<char, 256> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        const auto status = fj_legacy_diffusion_frame_prepare(&input, &wire, &rejected, &error);
        if (!envelope(3, status, rejected, error, text))
            return reject("frame", failure);
        if (rejected.reason != 0) {
            if (!cleared(wire))
                return reject("frame", failure);
            JuicerCuda::set_failure(failure, status, std::string_view(text.data(), error.length));
            return false;
        }
        const unsigned flags = (input.camera.hash != 0 ? 1u : 0u) | (input.enlarger.hash != 0 ? 2u : 0u);
        if (flags == 0)
            return cleared(wire) ? true : reject("frame", failure);
        if (wire.flags != flags || wire.route > 3 || wire.route != input.route || wire.hash == 0 ||
            wire.full_frame.x != domain.originX || wire.full_frame.y != domain.originY || wire.full_frame.width != domain.width || wire.full_frame.height != domain.height ||
            !associated(wire.camera, input.camera, (flags & 1) != 0) || !associated(wire.enlarger, input.enlarger, (flags & 2) != 0))
            return reject("frame", failure);
        Spektrafilm::DiffusionFrameSetDescriptor descriptor;
        descriptor.route = route;
        descriptor.fullFrame = domain;
        descriptor.hash = wire.hash;
        if (flags & 1)
            descriptor.camera = project(wire.camera, Spektrafilm::DiffusionLinearStage::CameraFilmLinear);
        if (flags & 2)
            descriptor.enlarger = project(wire.enlarger, Spektrafilm::DiffusionLinearStage::EnlargerPrintLinear);
        out = descriptor;
        return true;
    }
} // namespace JuicerDiffusion
