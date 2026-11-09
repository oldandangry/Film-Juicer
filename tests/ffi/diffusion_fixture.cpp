#include "diffusion_fixture.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include "Cuda/JuicerCudaFailure.h"
#include "juicer_test_api.h"


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
        const auto status = fj_test_diffusion_sample_prepare(&input, &wire, &rejected, &error);
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


    bool prepare_frame(const SpatialOptics& optics, Spektrafilm::ScanRoute route, double pixelSizeUm, Spektrafilm::DiffusionFrameDomain domain, std::optional<Spektrafilm::DiffusionFrameSetDescriptor>& out, JuicerCuda::Failure& failure) {
        out.reset();
        failure = {};
        FjDiffusionFrameInput input{static_cast<std::uint32_t>(route), {domain.originX, domain.originY, domain.width, domain.height}, pixelSizeUm, bind(optics.cameraDiffusion), bind(optics.enlargerDiffusion)};
        FjDiffusionFrameSet wire{};
        FjDiffusionFailure rejected{};
        std::array<char, 256> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        const auto status = fj_test_diffusion_frame_prepare(&input, &wire, &rejected, &error);
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


namespace {

    constexpr std::uint64_t kFnvOffset = 0xcbf29ce484222325ULL;
    constexpr std::uint64_t kFnvPrime = 0x100000001b3ULL;

    void hash_byte(std::uint64_t& hash, std::uint8_t value) {
        hash ^= static_cast<std::uint64_t>(value);
        hash *= kFnvPrime;
    }

    void hash_u64_le(std::uint64_t& hash, std::uint64_t value) {
        for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
            hash_byte(hash, static_cast<std::uint8_t>((value >> (byte * 8)) & 0xffu));
        }
    }

    void hash_u32_le(std::uint64_t& hash, std::uint32_t value) {
        for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
            hash_byte(hash, static_cast<std::uint8_t>((value >> (byte * 8)) & 0xffu));
        }
    }

    void hash_tag(std::uint64_t& hash, std::string_view tag) {
        for (const char value : tag) {
            hash_byte(hash, static_cast<std::uint8_t>(value));
        }
        hash_byte(hash, 0);
    }

    void hash_bool(std::uint64_t& hash, bool value) {
        hash_byte(hash, value ? 1u : 0u);
    }

    void hash_family(std::uint64_t& hash, Spektrafilm::DiffusionFilterFamily family) {
        hash_byte(hash, static_cast<std::uint8_t>(family));
    }

    void hash_schema(std::uint64_t& hash) {
        hash_u32_le(hash, Spektrafilm::kDiffusionHostBehaviorSchemaVersion);
    }

    void hash_double(std::uint64_t& hash, double value) {
        if (value == 0.0) {
            value = 0.0;
        }
        hash_u64_le(hash, std::bit_cast<std::uint64_t>(value));
    }

} // namespace

namespace Spektrafilm {

    std::uint64_t hash_diffusion_authored_controls(
        const DiffusionFilterAuthoredControls& controls) {
        std::uint64_t hash = kFnvOffset;
        hash_tag(hash, "diffusion-authored");
        hash_schema(hash);
        hash_bool(hash, controls.active);
        hash_family(hash, controls.family);
        hash_double(hash, controls.strength);
        hash_double(hash, controls.spatialScale);
        hash_double(hash, controls.haloWarmth);
        hash_double(hash, controls.coreIntensity);
        hash_double(hash, controls.coreSize);
        hash_double(hash, controls.haloIntensity);
        hash_double(hash, controls.haloSize);
        hash_double(hash, controls.bloomIntensity);
        hash_double(hash, controls.bloomSize);
        return hash;
    }

    // Retain the public sample builder's scalar argument order.
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    bool build_diffusion_psf_sample_descriptor(
        const DiffusionFilterResolvedParameters& resolved,
        double pixelSizeUm,
        int fullFrameWidth,
        int fullFrameHeight,
        DiffusionPsfSampleDescriptor& out,
        std::string& diagnostic) {
        JuicerCuda::Failure failure;
        const bool success = JuicerDiffusion::prepare_sample(
            resolved, pixelSizeUm, fullFrameWidth, fullFrameHeight, out, failure);
        diagnostic = std::move(failure.diagnostic);
        return success;
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)


} // namespace Spektrafilm
