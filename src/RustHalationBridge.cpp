#include "RustHalationBridge.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <string>
#include <string_view>

#include "Cuda/JuicerCudaFailure.h"
#include "juicer_legacy_api.h"

namespace JuicerHalation {
    namespace {
        bool reject_projection(JuicerCuda::Failure& failure) {
            JuicerCuda::set_failure(failure, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "scatter/halation frame projection invalid");
            return false;
        }
        bool has_gaussian_shape(const FjHalationGaussian& v) {
            return (v.kind == FJ_HALATION_IDENTITY && v.radius == 0) ||
                   (v.kind == FJ_HALATION_FIR_REFLECT && v.radius >= 1 && v.radius <= 9) ||
                   (v.kind == FJ_HALATION_YVV_REPLICATE && v.radius == 0);
        }
        bool is_positive_zero(float value) {
            return std::bit_cast<std::uint32_t>(value) == 0;
        }
        bool is_cleared(const FjHalationGaussian& v) {
            return v.kind == FJ_HALATION_IDENTITY && v.radius == 0 && is_positive_zero(v.feedforward) &&
                   std::all_of(std::begin(v.fir_weights), std::end(v.fir_weights), is_positive_zero) &&
                   std::all_of(std::begin(v.feedback), std::end(v.feedback), is_positive_zero);
        }
        bool is_cleared(const FjScatterHalation& v) {
            if (v.recipe_hash != 0 || !is_positive_zero(v.scatter_amount))
                return false;
            for (const auto& c : v.channels) {
                if (!is_positive_zero(c.total_strength) || !is_cleared(c.core))
                    return false;
                for (const auto* group : {&c.tail, &c.bounce})
                    for (const auto& g : *group)
                        if (!is_cleared(g))
                            return false;
            }
            return true;
        }
        ScatterHalationGaussianDescriptor project(const FjHalationGaussian& v) {
            ScatterHalationGaussianDescriptor out;
            out.kind = static_cast<ScatterHalationGaussianKind>(v.kind);
            out.radius = v.radius;
            std::copy(std::begin(v.fir_weights), std::end(v.fir_weights), out.firWeights.begin());
            out.B = v.feedforward;
            out.B1 = v.feedback[0];
            out.B2 = v.feedback[1];
            out.B3 = v.feedback[2];
            return out;
        }
    } // namespace

    bool prepare_frame(const ScatterHalationOpticsRecipe& recipe, float pixelSizeUm, std::optional<ScatterHalationFrameDescriptor>& out, JuicerCuda::Failure& rejected) {
        out.reset();
        rejected = {};
        FjHalationFrameInput input{};
        auto& h = input.recipe;
        h.scatter_active = recipe.scatterActive ? 1u : 0u;
        h.back_reflection_active = recipe.backReflectionActive ? 1u : 0u;
        h.scatter_amount = recipe.scatterAmount;
        h.scatter_spatial_scale = recipe.scatterSpatialScale;
        h.halation_spatial_scale = recipe.halationSpatialScale;
        std::copy(recipe.halationFirstSigmaUm.begin(), recipe.halationFirstSigmaUm.end(), h.first_sigma_um);
        std::copy(recipe.totalStrength.begin(), recipe.totalStrength.end(), h.total_strength);
        h.hash = recipe.hash;
        input.pixel_size_um = pixelSizeUm;
        FjScatterHalation wire{};
        FjHalationFrameFailure failure{};
        std::array<char, 256> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        const auto status = fj_legacy_scatter_halation_frame_prepare(&input, &wire, &failure, &error);
        constexpr std::array<std::uint32_t, 5> categories{FJ_STATUS_SUCCESS, FJ_STATUS_UNSUPPORTED_INPUT, FJ_STATUS_PREPARATION_FAILURE, FJ_STATUS_PREPARATION_FAILURE, FJ_STATUS_INTERNAL_FAILURE};
        constexpr std::array messages{"", "invalid scatter/halation frame arguments", "ResourceDescriptorMismatch component=scatter_halation field=pixel_size_um", "", "scatter/halation frame preparation panicked"};
        constexpr std::array fields{"", "", "ResourceDescriptorMismatch component=scatter_halation field=scatter_core", "ResourceDescriptorMismatch component=scatter_halation field=scatter_tail", "ResourceDescriptorMismatch component=scatter_halation field=back_reflection_bounce"};
        if (failure.reason >= categories.size() || status.category != categories[failure.reason] || status.api != FJ_API_NONE || status.native_code != 0 ||
            error.data != text.data() || error.capacity != text.size() || error.length >= text.size() || text[error.length] != '\0')
            return reject_projection(rejected);
        const bool gaussian = failure.reason == FJ_HALATION_FRAME_GAUSSIAN;
        if (gaussian) {
            if (failure.field < 2 || failure.field > 4 || failure.channel >= 3 ||
                (failure.field == 2 ? failure.index != 3 : failure.index >= 3))
                return reject_projection(rejected);
        } else if (failure.field != (failure.reason == FJ_HALATION_FRAME_PIXEL_SIZE ? 1u : 0u) ||
                   failure.channel != 3 || failure.index != 3 || std::bit_cast<std::uint64_t>(failure.sigma_px) != 0) {
            return reject_projection(rejected);
        }
        if (std::string_view(text.data(), error.length) != (gaussian ? fields[failure.field] : messages[failure.reason]))
            return reject_projection(rejected);
        if (failure.reason != FJ_HALATION_FRAME_NONE) {
            if (!is_cleared(wire))
                return reject_projection(rejected);
            std::string diagnostic(text.data(), error.length);
            if (gaussian)
                diagnostic += " sigma_pixels=" + std::to_string(failure.sigma_px);
            JuicerCuda::set_failure(rejected, status, diagnostic);
            return false;
        }
        if (wire.recipe_hash != recipe.hash)
            return reject_projection(rejected);
        if (recipe.hash == 0)
            return is_cleared(wire) ? true : reject_projection(rejected);
        for (const auto& c : wire.channels) {
            if (!has_gaussian_shape(c.core))
                return reject_projection(rejected);
            for (const auto* group : {&c.tail, &c.bounce})
                for (const auto& g : *group)
                    if (!has_gaussian_shape(g))
                        return reject_projection(rejected);
        }
        ScatterHalationFrameDescriptor descriptor;
        descriptor.recipeHash = wire.recipe_hash;
        descriptor.scatterAmount = wire.scatter_amount;
        for (std::size_t channel = 0; channel < descriptor.channels.size(); ++channel) {
            const auto& source = wire.channels[channel];
            auto& target = descriptor.channels[channel];
            target.core = project(source.core);
            target.totalStrength = source.total_strength;
            for (std::size_t index = 0; index < target.tail.size(); ++index) {
                target.tail[index] = project(source.tail[index]);
                target.bounce[index] = project(source.bounce[index]);
            }
        }
        out = descriptor;
        return true;
    }
} // namespace JuicerHalation
