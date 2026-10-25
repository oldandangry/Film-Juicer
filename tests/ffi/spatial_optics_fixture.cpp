#include "spatial_optics_fixture.h"

#include <algorithm>
#include <array>
#include <bit>
#include <stdexcept>

#include "juicer_test_api.h"
#include "Cuda/JuicerCudaFailure.h"
#include "RustHalationBridge.h"
#include "RustDirBridge.h"

namespace {
    FjSpatialOpticsParams defaults() {
        FjSpatialOpticsParams input{};
        input.camera_diffusion.family = 1;
        input.enlarger_diffusion.family = 1;
        return input;
    }

    bool prepare(const FjSpatialOpticsParams& input, FjSpatialOpticsRecipe& out, std::string& diagnostic) {
        diagnostic.clear();
        FjSpatialOpticsFailure failure{};
        const FjStatus status = fj_test_spatial_optics_prepare(&input, &out, &failure, nullptr);
        if (status.category == FJ_STATUS_SUCCESS) {
            return true;
        }
        if (status.category != FJ_STATUS_PREPARATION_FAILURE) {
            throw std::runtime_error("spatial optics fixture boundary failure");
        }
        constexpr std::array<const char*, 16> fields{{"family", "strength", "spatial_scale", "halo_warmth", "core_intensity", "core_size", "halo_intensity", "halo_size", "bloom_intensity", "bloom_size", "scatter_fraction", "group_weights", "group_centers_um", "resolved_hash", "sigma_um", "total_strength"}};
        if (failure.field == 0 || failure.field > fields.size()) {
            throw std::runtime_error("spatial optics fixture unknown field");
        }
        if (failure.reason == FJ_OPTICS_REASON_DIFFUSION) {
            diagnostic = "InvalidDiffusionHostBehavior field=";
        } else if (failure.reason == FJ_OPTICS_REASON_HALATION) {
            diagnostic = "ResourceDescriptorMismatch component=scatter_halation field=";
        } else {
            throw std::runtime_error("spatial optics fixture unexpected component");
        }
        diagnostic += fields[failure.field - 1];
        if (failure.channel < 3) {
            diagnostic += " channel=" + std::to_string(failure.channel);
        }
        return false;
    }
} // namespace

namespace JuicerTestSupport {
    nlohmann::json dir_frame_fields(const Spektrafilm::DirFrameExtent& input) {
        return {input.x, input.y, input.width, input.height};
    }

    nlohmann::json dir_frame_fields(const Spektrafilm::DirScratchPlaneRoles& input) {
        return {input.rawCorrectionPlanes, input.filteredCorrectionPlanes, input.filterTempPlanes, input.cachedLogRawPlanes};
    }

    nlohmann::json dir_frame_fields(const Spektrafilm::DirGaussianComponentPlan& input) {
        nlohmann::json feedback = nlohmann::json::array(), numerators = nlohmann::json::array();
        for (double value : input.iir.feedback)
            feedback.push_back(std::bit_cast<std::uint64_t>(value));
        for (double value : input.iir.feedbackNumerators)
            numerators.push_back(std::bit_cast<std::uint64_t>(value));
        return {{"sigma", std::bit_cast<std::uint32_t>(input.sigmaPixels)},
                {"weight", std::bit_cast<std::uint32_t>(input.weight)},
                {"radius", input.radius},
                {"operator", static_cast<unsigned>(input.referenceOperator)},
                {"target", static_cast<unsigned>(input.targetScratchTier)},
                {"feedforward", std::bit_cast<std::uint64_t>(input.iir.feedforward)},
                {"feedback", feedback},
                {"denominator", std::bit_cast<std::uint64_t>(input.iir.normalizationDenominator)},
                {"numerator", std::bit_cast<std::uint64_t>(input.iir.feedforwardNumerator)},
                {"numerators", numerators},
                {"accuracy", std::bit_cast<std::uint64_t>(input.boundaryTruncationAccuracy)},
                {"tolerance", std::bit_cast<std::uint64_t>(input.boundaryCertificationTolerance)},
                {"version", input.boundaryDerivationVersion}};
    }

    nlohmann::json dir_frame_fields(const SpatialDirDescriptor& input) {
        nlohmann::json components = nlohmann::json::array();
        for (const auto& component : input.filterPlan.components)
            components.push_back(dir_frame_fields(component));
        return {{"recipe_hash", input.dirRecipeHash},
                {"scratch", static_cast<unsigned>(input.scratchTier)},
                {"support", static_cast<unsigned>(input.support)},
                {"render", dir_frame_fields(input.renderExtent)},
                {"full", dir_frame_fields(input.fullFrameExtent)},
                {"filter", dir_frame_fields(input.filterDomainExtent)},
                {"count", input.filterPlan.componentCount},
                {"components", components},
                {"roles", dir_frame_fields(input.planeRoles)},
                {"target", static_cast<unsigned>(input.targetScratchTier)},
                {"target_roles", dir_frame_fields(input.targetPlaneRoles)},
                {"label", input.traceRouteLabel ? nlohmann::json(input.traceRouteLabel) : nlohmann::json(nullptr)},
                {"hash", input.hash}};
    }

    bool prepare_dir_frame(const DirCouplersRecipe& recipe, float pixelSizeUm, Spektrafilm::DirFrameExtent renderExtent, Spektrafilm::DirFrameExtent fullFrameExtent, const char* traceRouteLabel, SpatialDirDescriptor& out) {
        const FjDirFrameInput input{
            recipe.active ? 1u : 0u, recipe.diffusionSizeUm, recipe.diffusionTailUm, recipe.diffusionTailWeight, pixelSizeUm, {renderExtent.x, renderExtent.y, renderExtent.width, renderExtent.height}, {fullFrameExtent.x, fullFrameExtent.y, fullFrameExtent.width, fullFrameExtent.height}, recipe.hash};
        FjDirFrameMath math{};
        FjDirFrameFailure failure{};
        std::array<char, 256> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        const auto status = fj_test_dir_frame_prepare(&input, &math, &failure, &error);
        const auto outcome = JuicerDir::MathOutcome::admit(input, status, math, failure, error, text);
        return Spektrafilm::compose_spatial_dir_descriptor(outcome, traceRouteLabel, out);
    }
    bool matches_halation_frame(const std::optional<ScatterHalationFrameDescriptor>& frame, const nlohmann::json& expected) {
        if (!frame || expected.is_null())
            return !frame && expected.is_null();
        if (frame->recipeHash != expected.at("recipe_hash").get<std::uint64_t>() ||
            std::bit_cast<std::uint32_t>(frame->scatterAmount) != expected.at("scatter_amount").get<std::uint32_t>())
            return false;
        const auto gaussian = [](const ScatterHalationGaussianDescriptor& actual, const nlohmann::json& frozen) {
            if (static_cast<unsigned>(actual.kind) != frozen.at("kind").get<unsigned>() || actual.radius != frozen.at("radius").get<int>())
                return false;
            for (std::size_t index = 0; index < actual.firWeights.size(); ++index)
                if (std::bit_cast<std::uint32_t>(actual.firWeights[index]) != frozen.at("weights")[index].get<std::uint32_t>())
                    return false;
            const std::array yvv{actual.B, actual.B1, actual.B2, actual.B3};
            for (std::size_t index = 0; index < yvv.size(); ++index)
                if (std::bit_cast<std::uint32_t>(yvv[index]) != frozen.at("yvv")[index].get<std::uint32_t>())
                    return false;
            return true;
        };
        for (std::size_t channel = 0; channel < frame->channels.size(); ++channel) {
            const auto& actual = frame->channels[channel];
            const auto& frozen = expected.at("channels")[channel];
            if (std::bit_cast<std::uint32_t>(actual.totalStrength) != frozen.at("strength").get<std::uint32_t>() || !gaussian(actual.core, frozen.at("core")))
                return false;
            for (std::size_t index = 0; index < 3; ++index)
                if (!gaussian(actual.tail[index], frozen.at("tail")[index]) || !gaussian(actual.bounce[index], frozen.at("bounce")[index]))
                    return false;
        }
        return true;
    }
    bool prepare_halation_frame(const ScatterHalationOpticsRecipe& recipe, float pixelSizeUm, std::optional<ScatterHalationFrameDescriptor>& out, std::string& diagnostic) {
        JuicerCuda::Failure failure;
        const bool success = JuicerHalation::prepare_frame(recipe, pixelSizeUm, out, failure);
        diagnostic = failure.diagnostic;
        return success;
    }
    bool prepare_diffusion_filter(
        const Spektrafilm::DiffusionFilterAuthoredControls& authored,
        Spektrafilm::DiffusionFilterResolvedParameters& out,
        std::string& diagnostic) {
        out = {};
        auto input = defaults();
        input.camera_diffusion = {authored.active ? 1u : 0u,
                                  static_cast<std::uint32_t>(authored.family),
                                  authored.strength,
                                  authored.spatialScale,
                                  authored.haloWarmth,
                                  authored.coreIntensity,
                                  authored.coreSize,
                                  authored.haloIntensity,
                                  authored.haloSize,
                                  authored.bloomIntensity,
                                  authored.bloomSize};
        FjSpatialOpticsRecipe result{};
        if (!prepare(input, result, diagnostic)) {
            return false;
        }
        const auto& v = result.camera_diffusion;
        out.active = v.active != 0;
        out.family = static_cast<Spektrafilm::DiffusionFilterFamily>(v.family);
        out.scatterFraction = v.scatter_fraction;
        std::copy_n(v.group_weights, 3, out.groupWeightsCoreHaloBloom.begin());
        std::copy_n(v.group_lambda_um, 3, out.groupCenterLambdaUm.begin());
        out.effectiveWarmth = v.warmth;
        out.spatialScale = v.spatial_scale;
        out.hash = v.resolved_hash;
        return true;
    }

    bool prepare_scatter_halation(
        const Spektrafilm::ScatterHalationControls& controls,
        const Profiles::ProfileDigest& profile,
        ScatterHalationOpticsRecipe& out,
        std::string& diagnostic) {
        out = {};
        auto input = defaults();
        auto& h = input.scatter_halation;
        h.active = controls.active ? 1u : 0u;
        h.scatter_amount = controls.scatterAmount;
        h.scatter_spatial_scale = controls.scatterSpatialScale;
        h.halation_amount = controls.halationAmount;
        h.halation_spatial_scale = controls.halationSpatialScale;
        std::copy_n(profile.halationFirstSigmaUm.begin(), 3, h.first_sigma_um);
        std::copy_n(profile.halationPrimaryAmount.begin(), 3, h.primary_amount);
        FjSpatialOpticsRecipe result{};
        if (!prepare(input, result, diagnostic)) {
            return false;
        }
        const auto& v = result.scatter_halation;
        out.scatterActive = v.scatter_active != 0;
        out.backReflectionActive = v.back_reflection_active != 0;
        out.scatterAmount = v.scatter_amount;
        out.scatterSpatialScale = v.scatter_spatial_scale;
        out.halationSpatialScale = v.halation_spatial_scale;
        std::copy_n(v.first_sigma_um, 3, out.halationFirstSigmaUm.begin());
        std::copy_n(v.total_strength, 3, out.totalStrength.begin());
        out.hash = v.hash;
        return true;
    }
} // namespace JuicerTestSupport

namespace {
    Spektrafilm::DiffusionPsfSampleDescriptor diffusion_sample(const FjDiffusionSample& v) {
        Spektrafilm::DiffusionPsfSampleDescriptor out;
        out.family = static_cast<Spektrafilm::DiffusionFilterFamily>(v.family);
        std::copy_n(v.group_weights, 3, out.groupWeightsCoreHaloBloom.begin());
        std::copy_n(v.group_lambda_um, 3, out.groupCenterLambdaUm.begin());
        out.effectiveWarmth = v.warmth;
        out.spatialScale = v.spatial_scale;
        out.pixelSizeUm = v.pixel_size_um;
        out.radiusPixels = v.radius_px;
        out.hash = v.hash;
        return out;
    }
    FjDiffusionSample diffusion_sample(const Spektrafilm::DiffusionPsfSampleDescriptor& v) {
        FjDiffusionSample out{};
        out.family = static_cast<std::uint32_t>(v.family);
        std::copy(v.groupWeightsCoreHaloBloom.begin(), v.groupWeightsCoreHaloBloom.end(), out.group_weights);
        std::copy(v.groupCenterLambdaUm.begin(), v.groupCenterLambdaUm.end(), out.group_lambda_um);
        out.warmth = v.effectiveWarmth;
        out.spatial_scale = v.spatialScale;
        out.pixel_size_um = v.pixelSizeUm;
        out.radius_px = v.radiusPixels;
        out.hash = v.hash;
        return out;
    }
    FjDiffusionRecipe diffusion_recipe(const DiffusionFilterOpticsRecipe& v) {
        FjDiffusionRecipe out{};
        out.domain = static_cast<std::uint32_t>(v.domain);
        out.active = v.resolved.active ? 1u : 0u;
        out.family = static_cast<std::uint32_t>(v.resolved.family);
        out.scatter_fraction = v.resolved.scatterFraction;
        std::copy(v.resolved.groupWeightsCoreHaloBloom.begin(), v.resolved.groupWeightsCoreHaloBloom.end(), out.group_weights);
        std::copy(v.resolved.groupCenterLambdaUm.begin(), v.resolved.groupCenterLambdaUm.end(), out.group_lambda_um);
        out.warmth = v.resolved.effectiveWarmth;
        out.spatial_scale = v.resolved.spatialScale;
        out.resolved_hash = v.resolved.hash;
        out.hash = v.hash;
        return out;
    }
    bool diffusion_succeeded(FjStatus status, const FjDiffusionFailure& failure, const FjErrorBuffer& error, std::string& diagnostic) {
        diagnostic.assign(error.data, error.length);
        if (status.api != 0 || status.native_code != 0 || (status.category != 0 && status.category != 3 && status.category != 8) ||
            (status.category == 0 && (failure.reason != 0 || failure.stage != 0 || failure.field != 0)))
            throw std::runtime_error("diffusion fixture envelope invalid");
        return status.category == 0;
    }
    Spektrafilm::DiffusionStageFrameDescriptor diffusion_stage(const FjDiffusion& v, Spektrafilm::DiffusionLinearStage stage) {
        FjDiffusionSample sample{};
        sample.family = v.family;
        std::copy_n(v.group_weights, 3, sample.group_weights);
        std::copy_n(v.group_lambda_um, 3, sample.group_lambda_um);
        sample.warmth = v.warmth;
        sample.spatial_scale = v.spatial_scale;
        sample.pixel_size_um = v.pixel_size_um;
        sample.radius_px = v.radius_px;
        sample.hash = v.sample_hash;
        return {stage, v.scatter_fraction, diffusion_sample(sample), v.hash};
    }
} // namespace

namespace JuicerTestSupport {
    // Mirror the native sample builder so fixture adoption changes only its producer.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    bool prepare_diffusion_sample(const Spektrafilm::DiffusionFilterResolvedParameters& r, double pixelSizeUm, int width, int height, Spektrafilm::DiffusionPsfSampleDescriptor& out, std::string& diagnostic) {
        out = {};
        FjDiffusionSampleInput input{};
        input.family = static_cast<std::uint32_t>(r.family);
        std::copy(r.groupWeightsCoreHaloBloom.begin(), r.groupWeightsCoreHaloBloom.end(), input.group_weights);
        std::copy(r.groupCenterLambdaUm.begin(), r.groupCenterLambdaUm.end(), input.group_lambda_um);
        input.warmth = r.effectiveWarmth;
        input.spatial_scale = r.spatialScale;
        input.pixel_size_um = pixelSizeUm;
        input.width = width;
        input.height = height;
        FjDiffusionSample wire{};
        FjDiffusionFailure failure{};
        std::array<char, 256> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        const auto status = fj_test_diffusion_sample_prepare(&input, &wire, &failure, &error);
        if (!diffusion_succeeded(status, failure, error, diagnostic))
            return false;
        out = diffusion_sample(wire);
        return true;
    }
    bool prepare_diffusion_components(const Spektrafilm::DiffusionPsfSampleDescriptor& sample, Spektrafilm::DiffusionPsfComponents& out, std::string& diagnostic) {
        out = {};
        const auto input = diffusion_sample(sample);
        FjDiffusionComponents wire{};
        FjDiffusionFailure failure{};
        std::array<char, 256> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        const auto status = fj_test_diffusion_components_prepare(&input, &wire, &failure, &error);
        if (!diffusion_succeeded(status, failure, error, diagnostic))
            return false;
        std::copy_n(wire.lambda_px, 9, out.lambdaPixels.begin());
        std::copy_n(wire.red_weights, 9, out.redWeights.begin());
        std::copy_n(wire.green_weights, 9, out.greenWeights.begin());
        std::copy_n(wire.blue_weights, 9, out.blueWeights.begin());
        return true;
    }
    bool prepare_diffusion_frame(const SpatialOptics& optics, Spektrafilm::ScanRoute route, double pixelSizeUm, Spektrafilm::DiffusionFrameDomain domain, std::optional<Spektrafilm::DiffusionFrameSetDescriptor>& out, std::string& diagnostic) {
        out.reset();
        const FjDiffusionFrameInput input{static_cast<std::uint32_t>(route), {domain.originX, domain.originY, domain.width, domain.height}, pixelSizeUm, diffusion_recipe(optics.cameraDiffusion), diffusion_recipe(optics.enlargerDiffusion)};
        FjDiffusionFrameSet wire{};
        FjDiffusionFailure failure{};
        std::array<char, 256> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        const auto status = fj_test_diffusion_frame_prepare(&input, &wire, &failure, &error);
        if (!diffusion_succeeded(status, failure, error, diagnostic))
            return false;
        if (wire.flags == 0)
            return true;
        Spektrafilm::DiffusionFrameSetDescriptor frame;
        frame.route = static_cast<Spektrafilm::ScanRoute>(wire.route);
        frame.fullFrame = {wire.full_frame.x, wire.full_frame.y, wire.full_frame.width, wire.full_frame.height};
        frame.hash = wire.hash;
        if (wire.flags & 1)
            frame.camera = diffusion_stage(wire.camera, Spektrafilm::DiffusionLinearStage::CameraFilmLinear);
        if (wire.flags & 2)
            frame.enlarger = diffusion_stage(wire.enlarger, Spektrafilm::DiffusionLinearStage::EnlargerPrintLinear);
        out = frame;
        return true;
    }
    bool matches_diffusion_frame(const std::optional<Spektrafilm::DiffusionFrameSetDescriptor>& frame, const nlohmann::json& expected) {
        if (!frame || expected.is_null())
            return !frame && expected.is_null();
        const auto stage = [](const std::optional<Spektrafilm::DiffusionStageFrameDescriptor>& actual, const nlohmann::json& frozen) {
            if (!actual || frozen.is_null())
                return !actual && frozen.is_null();
            const auto& sample = actual->sample;
            const auto& f = frozen.at("sample");
            if (static_cast<unsigned>(actual->stage) != frozen.at("stage").get<unsigned>() || actual->hash != frozen.at("hash").get<std::uint64_t>() ||
                std::bit_cast<std::uint64_t>(actual->scatterFraction) != frozen.at("scatter").get<std::uint64_t>() ||
                static_cast<unsigned>(sample.family) != f.at("family").get<unsigned>() || sample.hash != f.at("hash").get<std::uint64_t>() || sample.radiusPixels != f.at("radius").get<int>() ||
                std::bit_cast<std::uint64_t>(sample.effectiveWarmth) != f.at("warmth").get<std::uint64_t>() || std::bit_cast<std::uint64_t>(sample.spatialScale) != f.at("scale").get<std::uint64_t>() || std::bit_cast<std::uint64_t>(sample.pixelSizeUm) != f.at("pixel").get<std::uint64_t>())
                return false;
            for (std::size_t i = 0; i < 3; ++i)
                if (std::bit_cast<std::uint64_t>(sample.groupWeightsCoreHaloBloom[i]) != f.at("weights")[i].get<std::uint64_t>() || std::bit_cast<std::uint64_t>(sample.groupCenterLambdaUm[i]) != f.at("centers")[i].get<std::uint64_t>())
                    return false;
            return true;
        };
        const auto& d = frame->fullFrame;
        return static_cast<unsigned>(frame->route) == expected.at("route").get<unsigned>() && frame->hash == expected.at("hash").get<std::uint64_t>() &&
               nlohmann::json({d.originX, d.originY, d.width, d.height}) == expected.at("extent") && stage(frame->camera, expected.at("camera")) && stage(frame->enlarger, expected.at("enlarger"));
    }
} // namespace JuicerTestSupport
