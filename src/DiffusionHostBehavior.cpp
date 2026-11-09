#include "DiffusionHostBehavior.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include "Cuda/JuicerCudaFailure.h"
#include "juicer_diffusion_api.h"

namespace JuicerDiffusion {
    namespace {
        bool is_positive_zero(double value) {
            return std::bit_cast<std::uint64_t>(value) == 0;
        }
        template <std::size_t N>
        bool cleared(const double (&values)[N]) {
            return std::all_of(std::begin(values), std::end(values), is_positive_zero);
        }
        bool cleared(const FjDiffusionComponents& v) {
            return cleared(v.lambda_px) && cleared(v.red_weights) && cleared(v.green_weights) && cleared(v.blue_weights);
        }
        bool reject(std::string_view operation, JuicerCuda::Failure& failure) {
            JuicerCuda::set_failure(failure, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "diffusion " + std::string(operation) + " projection invalid");
            return false;
        }
        bool envelope(FjStatus status, const FjDiffusionFailure& v, const FjErrorBuffer& error, const std::array<char, 256>& text) {
            constexpr std::array categories{FJ_STATUS_SUCCESS, FJ_STATUS_UNSUPPORTED_INPUT, FJ_STATUS_PREPARATION_FAILURE, FJ_STATUS_INTERNAL_FAILURE};
            constexpr std::array fields{"", "pixel_size_um", "full_frame_extent", "family", "radius_pixels", "sample_descriptor_hash", "sample_descriptor_inactive", "group_weights", "group_centers_um", "effective_warmth", "spatial_scale", "component_expansion", "component_count", "halo_channel_weights", "component_output"};
            if (v.reason >= categories.size() || status.category != categories[v.reason] || status.api != FJ_API_NONE || status.native_code != 0 ||
                error.data != text.data() || error.capacity != text.size() || error.length >= text.size() || text[error.length] != '\0' || v.stage != 0)
                return false;
            std::string expected;
            if (v.reason == FJ_DIFFUSION_REASON_PREPARATION) {
                if (v.field < 1 || v.field > 14 || v.field == 2)
                    return false;
                expected = "InvalidDiffusionHostBehavior field=";
                expected += fields[v.field];
            } else {
                if (v.field != 0)
                    return false;
                if (v.reason == FJ_DIFFUSION_REASON_RAW)
                    expected = "invalid diffusion components arguments";
                if (v.reason == FJ_DIFFUSION_REASON_PANIC)
                    expected = "diffusion components preparation panicked";
            }
            return std::string_view(text.data(), error.length) == expected;
        }
    } // namespace
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
        const auto status = fj_diffusion_components_prepare(&input, &wire, &rejected, &error);
        if (!envelope(status, rejected, error, text))
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
} // namespace JuicerDiffusion

namespace Spektrafilm {
    bool expand_diffusion_psf_components(
        const DiffusionPsfSampleDescriptor& descriptor,
        DiffusionPsfComponents& out,
        std::string& diagnostic) {
        JuicerCuda::Failure failure;
        const bool success = JuicerDiffusion::prepare_components(descriptor, out, failure);
        diagnostic = std::move(failure.diagnostic);
        return success;
    }
} // namespace Spektrafilm
