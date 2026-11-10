#include "effects_fixture.h"
#include "spatial_optics_fixture.h"

#include <algorithm>
#include <array>
#include <bit>
#include <stdexcept>

#include "juicer_test_api.h"
#include "Cuda/JuicerCudaFailure.h"
#include "halation_fixture.h"
#include "dir_fixture.h"

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

    bool prepare_dir_frame(const FjDirFrameInput& input, const char* traceRouteLabel, SpatialDirDescriptor& out) {
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
        const std::array<float, 3>& firstSigmaUm,
        const std::array<float, 3>& primaryAmount,
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
        std::copy(firstSigmaUm.begin(), firstSigmaUm.end(), h.first_sigma_um);
        std::copy(primaryAmount.begin(), primaryAmount.end(), h.primary_amount);
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

namespace JuicerTestSupport {
    nlohmann::json grain_frame_fields(const Spektrafilm::VisualGrainFrameDescriptor& v) {
        using Json = nlohmann::json;
        Json out;
        out["active"] = v.active;
        out["capture_polarity"] = static_cast<unsigned>(v.capturePolarity);
        out["render_extent"] = {v.renderExtent.x, v.renderExtent.y, v.renderExtent.width, v.renderExtent.height};
        out["full_frame_extent"] = {v.fullFrameExtent.x, v.fullFrameExtent.y, v.fullFrameExtent.width, v.fullFrameExtent.height};
        out["pixel_size_um"] = std::bit_cast<std::uint32_t>(v.pixelSizeUm);
        out["frame_alpha"] = std::bit_cast<std::uint32_t>(v.frameAlpha);
        out["wang_cell_mm"] = std::bit_cast<std::uint32_t>(v.wangCellMm);
        out["breathing_amplitude"] = std::bit_cast<std::uint32_t>(v.breathingAmplitude);
        out["breathing_cell_um_small"] = std::bit_cast<std::uint32_t>(v.breathingCellUmSmall);
        out["breathing_cell_um_large"] = std::bit_cast<std::uint32_t>(v.breathingCellUmLarge);
        out["breathing_mix"] = std::bit_cast<std::uint32_t>(v.breathingMix);
        out["breathing_drift_um_per_frame"] = std::bit_cast<std::uint32_t>(v.breathingDriftUmPerFrame);
        out["debug_scale"] = std::bit_cast<std::uint32_t>(v.debugScale);
        out["size_mix_gain"] = std::bit_cast<std::uint32_t>(v.sizeMixGain);
        out["frame0"] = v.frame0;
        out["seed_base"] = v.seedBase;
        out["seed_base_next"] = v.seedBaseNext;
        out["session_seed"] = v.sessionSeed;
        out["clip_token"] = v.clipToken;
        out["pitch_px"] = v.pitchPx;
        out["breathing_period_frames"] = v.breathingPeriodFrames;
        out["clump_morph_period_frames"] = v.clumpMorphPeriodFrames;
        out["recipe_hash"] = v.recipeHash;
        out["density_layers_hash"] = v.densityCurvesLayersHash;
        out["hash"] = v.hash;
        out["scratch_shape"] = static_cast<unsigned>(v.scratchShape);
        out["requires_full_frame"] = v.requiresFullFrame;
        out["size_mix_weights"] = {std::bit_cast<std::uint32_t>(v.effectiveFineWeight), std::bit_cast<std::uint32_t>(v.effectiveMidWeight), std::bit_cast<std::uint32_t>(v.effectiveCoarseWeight)};
        const auto words = [](const auto& values) {
            Json result = Json::array();
            for (float f : values)
                result.push_back(std::bit_cast<std::uint32_t>(f));
            return result;
        };
        out["density_max_cmy"] = words(v.densityMaxCmy);
        out["n_particles_cmy"] = words(v.nParticlesCmy);
        out["od_particle_cmy"] = words(v.odParticleCmy);
        out["density_min_layers"] = Json::array();
        for (const auto& layer : v.densityMinLayers)
            out["density_min_layers"].push_back(words(layer));
        out["density_max_layers"] = Json::array();
        for (const auto& layer : v.densityMaxLayers)
            out["density_max_layers"].push_back(words(layer));
        out["n_particles_layers"] = Json::array();
        for (const auto& layer : v.nParticlesLayers)
            out["n_particles_layers"].push_back(words(layer));
        out["od_particle_layers"] = Json::array();
        for (const auto& layer : v.odParticleLayers)
            out["od_particle_layers"].push_back(words(layer));
        const auto gaussian = [](const auto& g) -> Json {
            return {std::bit_cast<std::uint32_t>(g.sigmaPx), g.radius, g.hash};
        };
        out["correlation"] = Json::array();
        for (const auto& g : v.correlation)
            out["correlation"].push_back(gaussian(g));
        out["dye_cloud"] = Json::array();
        for (const auto& layer : v.dyeCloud) {
            Json row = Json::array();
            for (const auto& g : layer)
                row.push_back(gaussian(g));
            out["dye_cloud"].push_back(row);
        }
        return out;
    }
} // namespace JuicerTestSupport

namespace JuicerTestSupport {
    FjGrainFrameInput GrainFrameInput::bind() const noexcept {
        FjGrainFrameInput bound = frame;
        for (std::size_t layer = 0; layer < layers.size(); ++layer) {
            for (std::size_t channel = 0; channel < layers[layer].size(); ++channel) {
                const auto& samples = layers[layer][channel];
                bound.layer_curves[layer][channel] = {samples.empty() ? nullptr : samples.data(), samples.size()};
            }
        }
        return bound;
    }

    namespace {
        std::uint32_t grain_u32(const nlohmann::json& value) {
            return value.is_boolean() ? static_cast<std::uint32_t>(value.get<bool>()) : value.get<std::uint32_t>();
        }

        float grain_float(const nlohmann::json& value) {
            return std::bit_cast<float>(value.get<std::uint32_t>());
        }

        double grain_double(const nlohmann::json& value) {
            return std::bit_cast<double>(value.get<std::uint64_t>());
        }

        FjExtent grain_extent(const nlohmann::json& value) {
            return {value.at(0).get<std::int32_t>(), value.at(1).get<std::int32_t>(), value.at(2).get<std::int32_t>(), value.at(3).get<std::int32_t>()};
        }
    } // namespace

    GrainFrameInput grain_frame_input(const nlohmann::json& value, const FjGrainCoefficients& seed) {
        GrainFrameInput input;
        input.coefficients = seed;
        auto& frame = input.frame;
        frame.active = grain_u32(value.at("active"));
        frame.sublayers_active = grain_u32(value.at("sublayers_active"));
        frame.layer_curves_required = grain_u32(value.at("layer_curves_required"));
        frame.capture_polarity = value.at("capture_polarity").get<std::uint32_t>();
        frame.film_polarity = value.at("film_polarity").get<std::uint32_t>();
        frame.sublayer_count = value.at("sublayer_count").get<std::int32_t>();
        frame.pixel_size_um = grain_float(value.at("pixel_size_um"));
        frame.particle_area_um2 = grain_float(value.at("particle_area_um2"));
        for (std::size_t channel = 0; channel < 3; ++channel) {
            frame.particle_scale_cmy[channel] = grain_float(value.at("particle_scale_cmy")[channel]);
            frame.particle_scale_layers[channel] = grain_float(value.at("particle_scale_layers")[channel]);
            frame.particle_density_min_cmy[channel] = grain_float(value.at("particle_density_min_cmy")[channel]);
            frame.authored_max_cmy[channel] = grain_float(value.at("authored_max_cmy")[channel]);
        }
        frame.correlation_sigma_px = grain_float(value.at("correlation_sigma_px"));
        frame.dye_cloud_blur_um = grain_float(value.at("dye_cloud_blur_um"));
        frame.size_mix_scale = grain_float(value.at("size_mix_scale"));
        frame.fine_weight = grain_float(value.at("fine_weight"));
        frame.mid_weight = grain_float(value.at("mid_weight"));
        frame.coarse_weight = grain_float(value.at("coarse_weight"));
        frame.clump_morph_period_sec = grain_float(value.at("clump_morph_period_sec"));
        frame.render_extent = grain_extent(value.at("render_extent"));
        frame.full_frame_extent = grain_extent(value.at("full_frame_extent"));
        frame.frame_time = grain_double(value.at("frame_time"));
        frame.frame_rate = grain_double(value.at("frame_rate"));
        frame.session_seed = value.at("session_seed").get<std::uint64_t>();
        frame.clip_token = value.at("clip_token").get<std::uint64_t>();
        frame.recipe_hash = value.at("recipe_hash").get<std::uint64_t>();
        frame.recipe_layers_hash = value.at("recipe_layers_hash").get<std::uint64_t>();
        frame.film_layers_hash = value.at("film_layers_hash").get<std::uint64_t>();
        for (std::size_t layer = 0; layer < input.layers.size(); ++layer) {
            for (std::size_t channel = 0; channel < input.layers[layer].size(); ++channel) {
                auto& samples = input.layers[layer][channel];
                for (const auto& sample : value.at("layer_curves")[layer][channel])
                    samples.push_back(grain_float(sample));
            }
        }

        auto& coefficients = input.coefficients;
        if (value.contains("active")) {
            coefficients.flags &= ~FJ_GRAIN_ACTIVE;
            if (frame.active)
                coefficients.flags |= FJ_GRAIN_ACTIVE;
        }
        if (value.contains("sublayers_active")) {
            coefficients.flags &= ~FJ_GRAIN_SUBLAYERS;
            if (frame.sublayers_active)
                coefficients.flags |= FJ_GRAIN_SUBLAYERS;
        }
        if (frame.active) {
            if (value.contains("debug_view"))
                coefficients.debug_view = value.at("debug_view").get<std::uint32_t>();
            if (value.contains("sublayer_count"))
                coefficients.sublayer_count = static_cast<std::size_t>(frame.sublayer_count);
            if (value.contains("size_mix_scale"))
                coefficients.size_mix_scale = frame.size_mix_scale;
            if (value.contains("chroma_mix"))
                coefficients.chroma_mix = grain_float(value.at("chroma_mix"));
            if (value.contains("shared_weight"))
                coefficients.chroma_shared_weight = grain_float(value.at("shared_weight"));
            if (value.contains("particle_density_min_cmy"))
                std::copy_n(frame.particle_density_min_cmy, 3, coefficients.particle_density_min_cmy);
        }
        return input;
    }

    FjGrainFrameInput grain_frame_input(const FjPreparedStateView& view, const FjGrainFrameInput& facts) {
        FjGrainFrameInput input = facts;
        const auto& recipe = view.grain;
        input.active = recipe.active;
        input.sublayers_active = recipe.sublayers_active;
        input.layer_curves_required = recipe.sublayers_active;
        input.capture_polarity = view.polarity;
        input.film_polarity = view.polarity;
        input.sublayer_count = recipe.sublayer_count;
        input.particle_area_um2 = recipe.particle_area_um2;
        std::copy_n(recipe.particle_scale_cmy, 3, input.particle_scale_cmy);
        std::copy_n(recipe.particle_scale_layers, 3, input.particle_scale_layers);
        std::copy_n(recipe.particle_density_min_cmy, 3, input.particle_density_min_cmy);
        const auto& bounds = (view.route & 1u) ? view.enlarger_film_bounds : view.scanner_bounds;
        std::copy_n(bounds.maximum_cmy, 3, input.authored_max_cmy);
        input.correlation_sigma_px = recipe.correlation_sigma_px;
        input.dye_cloud_blur_um = recipe.dye_cloud_blur_um;
        input.size_mix_scale = recipe.size_mix_scale;
        input.fine_weight = recipe.fine_weight;
        input.mid_weight = recipe.mid_weight;
        input.coarse_weight = recipe.coarse_weight;
        input.clump_morph_period_sec = recipe.clump_morph_period_sec;
        input.recipe_hash = recipe.hash;
        input.recipe_layers_hash = recipe.layers_hash;
        input.film_layers_hash = view.film_development.density_layers_hash;
        for (std::size_t layer = 0; layer < 3; ++layer)
            for (std::size_t channel = 0; channel < 3; ++channel)
                input.layer_curves[layer][channel] = view.film_development.density_layers[layer][channel];
        return input;
    }

    FjGrainCoefficients grain_coefficients(const FjVisualGrainRecipe& recipe) {
        FjGrainCoefficients out{};
        if (!recipe.active)
            return out;
        out.flags = FJ_GRAIN_ACTIVE | (recipe.sublayers_active ? FJ_GRAIN_SUBLAYERS : 0u);
        out.debug_view = recipe.debug_view;
        out.sublayer_count = static_cast<std::size_t>(recipe.sublayer_count);
        std::copy_n(recipe.micro_structure, 2, out.micro_structure);
        out.clump_temporal_mix = recipe.clump_temporal_mix;
        out.size_mix_scale = recipe.size_mix_scale;
        out.amplitude = recipe.amplitude;
        out.chroma_mix = recipe.chroma_mix;
        out.chroma_shared_weight = recipe.chroma_shared_weight;
        out.chroma_independent_weight = recipe.chroma_independent_weight;
        std::copy_n(recipe.particle_density_min_cmy, 3, out.particle_density_min_cmy);
        std::copy_n(recipe.uniformity_cmy, 3, out.uniformity_cmy);
        for (std::size_t channel = 0; channel < 3; ++channel) {
            if (recipe.axis_finite_cmy[channel])
                out.flags |= FJ_GRAIN_AXIS_FINITE_C << channel;
            std::copy_n(recipe.axis_block_prefix_max_cmy[channel], 16, out.layer_axis_block_prefix_max[channel]);
        }
        return out;
    }

    bool prepare_grain_frame(const GrainFrameInput& input, Spektrafilm::VisualGrainFrameDescriptor& out, std::string& diagnostic) {
        out = {};
        diagnostic.clear();
        const FjGrainFrameInput raw = input.bind();
        FjGrainFrameMath math{};
        FjGrainFrameFailure failure{};
        std::array<char, 256> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        const FjStatus status = fj_test_grain_frame_prepare(&raw, &math, &failure, &error);
        if (status.api != FJ_API_NONE || status.native_code != 0 || error.length >= error.capacity || text[error.length] != '\0') {
            diagnostic = "ResourceDescriptorMismatch phase=grain_descriptor field=rust_result";
            return false;
        }
        if (status.category != FJ_STATUS_SUCCESS) {
            diagnostic.assign(text.data(), error.length);
            return false;
        }
        if (error.length != 0 || failure.reason != 0 || failure.field != 0 || failure.layer != 3 || failure.channel != 3) {
            diagnostic = "ResourceDescriptorMismatch phase=grain_descriptor field=rust_result";
            return false;
        }
        const bool rawRelations = input.frame.active
                                      ? math.present == 1 && math.capture_polarity == raw.capture_polarity &&
                                            math.render_extent.x == raw.render_extent.x && math.render_extent.y == raw.render_extent.y &&
                                            math.render_extent.width == raw.render_extent.width && math.render_extent.height == raw.render_extent.height &&
                                            math.full_frame_extent.x == raw.full_frame_extent.x && math.full_frame_extent.y == raw.full_frame_extent.y &&
                                            math.full_frame_extent.width == raw.full_frame_extent.width && math.full_frame_extent.height == raw.full_frame_extent.height &&
                                            std::bit_cast<std::uint32_t>(math.pixel_size_um) == std::bit_cast<std::uint32_t>(raw.pixel_size_um) &&
                                            math.session_seed == raw.session_seed && math.clip_token == raw.clip_token &&
                                            math.recipe_hash == raw.recipe_hash &&
                                            math.density_layers_hash == (raw.sublayers_active ? raw.film_layers_hash : 0)
                                      : math.present == 0 && math.recipe_hash == 0 && math.density_layers_hash == 0;
        if (!rawRelations) {
            diagnostic = "ResourceDescriptorMismatch phase=grain_descriptor field=rust_result";
            return false;
        }
        if (!Spektrafilm::compose_visual_grain_frame_descriptor(
                input.coefficients, static_cast<Spektrafilm::ProfilePolarity>(raw.capture_polarity), math, out, diagnostic)) {
            out = {};
            diagnostic = "ResourceDescriptorMismatch phase=grain_descriptor field=rust_result";
            return false;
        }
        return true;
    }
} // namespace JuicerTestSupport

namespace JuicerTestSupport {
    bool prepare_effects_frame(const Spektrafilm::FilmJuicerEffectsFrameDescriptorInput& input, Spektrafilm::FilmJuicerEffectsFrameDescriptor& out, std::string& diagnostic) {
        out = {};
        FjEffectsFrameInput raw{};
        if (!JuicerEffects::project_input(input, raw, diagnostic))
            return false;
        FjEffectsFrameMath math{};
        FjEffectsFrameFailure failure{};
        std::array<char, 256> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        const auto status = fj_test_effects_frame_prepare(&raw, &math, &failure, &error);
        return JuicerEffects::accept_result(input, status, math, failure, error, text, out, diagnostic);
    }
} // namespace JuicerTestSupport

namespace JuicerTestSupport {
    namespace {
        float effects_scalar(const nlohmann::json& v) {
            return std::bit_cast<float>(v.get<std::uint32_t>());
        }
        double effects_wide(const nlohmann::json& v) {
            return std::bit_cast<double>(v.get<std::uint64_t>());
        }
        Spektrafilm::FilmJuicerEffectsFrameExtent effects_extent(const nlohmann::json& v) {
            return {v.at(0), v.at(1), v.at(2), v.at(3)};
        }
        DefectDustRecipe effects_dust(const nlohmann::json& v) {
            DefectDustRecipe out;
            out.cellWidthMm = effects_scalar(v.at("cell_width_mm"));
            out.cellHeightMm = effects_scalar(v.at("cell_height_mm"));
            out.slotProbability = effects_scalar(v.at("slot_probability"));
            out.softnessMinMm = effects_scalar(v.at("softness_min_mm"));
            out.softnessMaxMm = effects_scalar(v.at("softness_max_mm"));
            out.softnessSizeCapFraction = effects_scalar(v.at("softness_size_cap_fraction"));
            out.supportXMm = effects_scalar(v.at("support_x_mm"));
            out.supportYMm = effects_scalar(v.at("support_y_mm"));
            out.fiberFraction = effects_scalar(v.at("fiber_fraction"));
            out.fiberDriftFraction = effects_scalar(v.at("fiber_drift_fraction"));
            out.fiberFirstKnotMin = effects_scalar(v.at("fiber_first_knot_min"));
            out.fiberFirstKnotMax = effects_scalar(v.at("fiber_first_knot_max"));
            out.fiberSecondKnotMin = effects_scalar(v.at("fiber_second_knot_min"));
            out.fiberSecondKnotMax = effects_scalar(v.at("fiber_second_knot_max"));
            out.fiberInteriorWidthMinFraction = effects_scalar(v.at("fiber_interior_width_min_fraction"));
            out.fiberInteriorWidthMaxFraction = effects_scalar(v.at("fiber_interior_width_max_fraction"));
            out.diameterMinMm = effects_scalar(v.at("diameter_min_mm"));
            out.diameterBulkMaxMm = effects_scalar(v.at("diameter_bulk_max_mm"));
            out.diameterMaxMm = effects_scalar(v.at("diameter_max_mm"));
            out.diameterTailFraction = effects_scalar(v.at("diameter_tail_fraction"));
            out.fiberLengthMinMm = effects_scalar(v.at("fiber_length_min_mm"));
            out.fiberLengthMaxMm = effects_scalar(v.at("fiber_length_max_mm"));
            out.fiberWidthMinMm = effects_scalar(v.at("fiber_width_min_mm"));
            out.fiberWidthMaxMm = effects_scalar(v.at("fiber_width_max_mm"));
            out.opacityFaintCumulative = effects_scalar(v.at("opacity_faint_cumulative"));
            out.opacityIntermediateCumulative = effects_scalar(v.at("opacity_intermediate_cumulative"));
            out.compactOpacityMin = effects_scalar(v.at("compact_opacity_min"));
            out.compactOpacityFaintEnd = effects_scalar(v.at("compact_opacity_faint_end"));
            out.compactOpacityIntermediateEnd = effects_scalar(v.at("compact_opacity_intermediate_end"));
            out.compactOpacityMax = effects_scalar(v.at("compact_opacity_max"));
            out.fiberOpacityMin = effects_scalar(v.at("fiber_opacity_min"));
            out.fiberOpacityFaintEnd = effects_scalar(v.at("fiber_opacity_faint_end"));
            out.fiberOpacityIntermediateEnd = effects_scalar(v.at("fiber_opacity_intermediate_end"));
            out.fiberOpacityMax = effects_scalar(v.at("fiber_opacity_max"));
            out.compactDominantAspectMin = effects_scalar(v.at("compact_dominant_aspect_min"));
            out.compactDominantAspectMax = effects_scalar(v.at("compact_dominant_aspect_max"));
            out.compactSubsidiaryScaleMin = effects_scalar(v.at("compact_subsidiary_scale_min"));
            out.compactSubsidiaryScaleMax = effects_scalar(v.at("compact_subsidiary_scale_max"));
            out.compactSubsidiaryAspectMin = effects_scalar(v.at("compact_subsidiary_aspect_min"));
            out.compactSubsidiaryAspectMax = effects_scalar(v.at("compact_subsidiary_aspect_max"));
            out.compactSubsidiaryOffsetMax = effects_scalar(v.at("compact_subsidiary_offset_max"));
            out.compactSubsidiaryAngleMaxRadians = effects_scalar(v.at("compact_subsidiary_angle_max_radians"));
            return out;
        }
        DefectScratchRecipe effects_scratch(const nlohmann::json& v) {
            DefectScratchRecipe out;
            out.cellWidthMm = effects_scalar(v.at("cell_width_mm"));
            out.cellHeightMm = effects_scalar(v.at("cell_height_mm"));
            out.slotProbability = effects_scalar(v.at("slot_probability"));
            out.softnessMinMm = effects_scalar(v.at("softness_min_mm"));
            out.softnessMaxMm = effects_scalar(v.at("softness_max_mm"));
            out.softnessSizeCapFraction = effects_scalar(v.at("softness_size_cap_fraction"));
            out.supportXMm = effects_scalar(v.at("support_x_mm"));
            out.supportYMm = effects_scalar(v.at("support_y_mm"));
            out.lengthMinMm = effects_scalar(v.at("length_min_mm"));
            out.lengthBulkMaxMm = effects_scalar(v.at("length_bulk_max_mm"));
            out.lengthMaxMm = effects_scalar(v.at("length_max_mm"));
            out.lengthTailFraction = effects_scalar(v.at("length_tail_fraction"));
            out.widthMinMm = effects_scalar(v.at("width_min_mm"));
            out.widthBulkMaxMm = effects_scalar(v.at("width_bulk_max_mm"));
            out.widthMaxMm = effects_scalar(v.at("width_max_mm"));
            out.widthTailFraction = effects_scalar(v.at("width_tail_fraction"));
            out.driftFraction = effects_scalar(v.at("drift_fraction"));
            out.firstKnotMin = effects_scalar(v.at("first_knot_min"));
            out.firstKnotMax = effects_scalar(v.at("first_knot_max"));
            out.secondKnotMin = effects_scalar(v.at("second_knot_min"));
            out.secondKnotMax = effects_scalar(v.at("second_knot_max"));
            out.interiorWidthMinFraction = effects_scalar(v.at("interior_width_min_fraction"));
            out.interiorWidthMaxFraction = effects_scalar(v.at("interior_width_max_fraction"));
            out.interiorDepthMinFraction = effects_scalar(v.at("interior_depth_min_fraction"));
            out.interiorDepthMaxFraction = effects_scalar(v.at("interior_depth_max_fraction"));
            out.endpointAbruptProbability = effects_scalar(v.at("endpoint_abrupt_probability"));
            out.interruptionProbability = effects_scalar(v.at("interruption_probability"));
            out.gapCenterMin = effects_scalar(v.at("gap_center_min"));
            out.gapCenterMax = effects_scalar(v.at("gap_center_max"));
            out.gapSpanMin = effects_scalar(v.at("gap_span_min"));
            out.gapSpanMax = effects_scalar(v.at("gap_span_max"));
            out.scuffProbability = effects_scalar(v.at("scuff_probability"));
            out.scuffLengthMaxMm = effects_scalar(v.at("scuff_length_max_mm"));
            out.scuffAngleMaxRadians = effects_scalar(v.at("scuff_angle_max_radians"));
            out.strengthMin = effects_scalar(v.at("strength_min"));
            out.strengthMax = effects_scalar(v.at("strength_max"));
            return out;
        }
    } // namespace
    Spektrafilm::FilmJuicerEffectsFrameDescriptorInput effects_frame_input(const nlohmann::json& v, Spektrafilm::FilmJuicerEffectsRecipe& recipe) {
        Spektrafilm::FilmJuicerEffectsFrameDescriptorInput frame;
        recipe.active = v.at("active");
        recipe.hash = v.at("recipe_hash");
        recipe.gateWeaveAmount = effects_wide(v.at("gate_weave_amount"));
        recipe.filmDust = effects_dust(v.at("film_dust"));
        recipe.filmScratch = effects_scratch(v.at("film_scratch"));
        recipe.gateDust = effects_dust(v.at("gate_dust"));
        recipe.gateScratch = effects_scratch(v.at("gate_scratch"));
        frame.recipe = v.value("missing_recipe", false) ? nullptr : &recipe;
        frame.renderExtent = effects_extent(v.at("render_extent"));
        frame.fullFrameExtent = effects_extent(v.at("full_frame_extent"));
        frame.geometry.pixelDefinition = effects_extent(v.at("pixel_definition"));
        frame.pixelSizeUm = effects_scalar(v.at("pixel_size_um"));
        frame.filmFormatLongEdgeMm = effects_scalar(v.at("film_format_long_edge_mm"));
        frame.frameTime = effects_wide(v.at("frame_time"));
        frame.frameRate = effects_wide(v.at("frame_rate"));
        frame.sessionSeed = v.at("session_seed");
        frame.clipToken = v.at("clip_token");
        frame.geometry.canonicalX = effects_wide(v.at("canonical_x"));
        frame.geometry.canonicalY = effects_wide(v.at("canonical_y"));
        frame.geometry.canonicalWidth = effects_wide(v.at("canonical_width"));
        frame.geometry.canonicalHeight = effects_wide(v.at("canonical_height"));
        frame.geometry.scaleX = effects_wide(v.at("scale_x"));
        frame.geometry.scaleY = effects_wide(v.at("scale_y"));
        frame.geometry.pixelAspectRatio = effects_wide(v.at("pixel_aspect_ratio"));
        return frame;
    }
    namespace {
        nlohmann::json effects_policy(const DefectDustRecipe& v) {
            return nlohmann::json::array({std::bit_cast<std::uint32_t>(v.cellWidthMm), std::bit_cast<std::uint32_t>(v.cellHeightMm), std::bit_cast<std::uint32_t>(v.slotProbability), std::bit_cast<std::uint32_t>(v.softnessMinMm), std::bit_cast<std::uint32_t>(v.softnessMaxMm), std::bit_cast<std::uint32_t>(v.softnessSizeCapFraction), std::bit_cast<std::uint32_t>(v.supportXMm), std::bit_cast<std::uint32_t>(v.supportYMm), std::bit_cast<std::uint32_t>(v.fiberFraction), std::bit_cast<std::uint32_t>(v.fiberDriftFraction), std::bit_cast<std::uint32_t>(v.fiberFirstKnotMin), std::bit_cast<std::uint32_t>(v.fiberFirstKnotMax), std::bit_cast<std::uint32_t>(v.fiberSecondKnotMin), std::bit_cast<std::uint32_t>(v.fiberSecondKnotMax), std::bit_cast<std::uint32_t>(v.fiberInteriorWidthMinFraction), std::bit_cast<std::uint32_t>(v.fiberInteriorWidthMaxFraction), std::bit_cast<std::uint32_t>(v.diameterMinMm), std::bit_cast<std::uint32_t>(v.diameterBulkMaxMm), std::bit_cast<std::uint32_t>(v.diameterMaxMm), std::bit_cast<std::uint32_t>(v.diameterTailFraction), std::bit_cast<std::uint32_t>(v.fiberLengthMinMm), std::bit_cast<std::uint32_t>(v.fiberLengthMaxMm), std::bit_cast<std::uint32_t>(v.fiberWidthMinMm), std::bit_cast<std::uint32_t>(v.fiberWidthMaxMm), std::bit_cast<std::uint32_t>(v.opacityFaintCumulative), std::bit_cast<std::uint32_t>(v.opacityIntermediateCumulative), std::bit_cast<std::uint32_t>(v.compactOpacityMin), std::bit_cast<std::uint32_t>(v.compactOpacityFaintEnd), std::bit_cast<std::uint32_t>(v.compactOpacityIntermediateEnd), std::bit_cast<std::uint32_t>(v.compactOpacityMax), std::bit_cast<std::uint32_t>(v.fiberOpacityMin), std::bit_cast<std::uint32_t>(v.fiberOpacityFaintEnd), std::bit_cast<std::uint32_t>(v.fiberOpacityIntermediateEnd), std::bit_cast<std::uint32_t>(v.fiberOpacityMax), std::bit_cast<std::uint32_t>(v.compactDominantAspectMin), std::bit_cast<std::uint32_t>(v.compactDominantAspectMax), std::bit_cast<std::uint32_t>(v.compactSubsidiaryScaleMin), std::bit_cast<std::uint32_t>(v.compactSubsidiaryScaleMax), std::bit_cast<std::uint32_t>(v.compactSubsidiaryAspectMin), std::bit_cast<std::uint32_t>(v.compactSubsidiaryAspectMax), std::bit_cast<std::uint32_t>(v.compactSubsidiaryOffsetMax), std::bit_cast<std::uint32_t>(v.compactSubsidiaryAngleMaxRadians)});
        }
        nlohmann::json effects_policy(const DefectScratchRecipe& v) {
            return nlohmann::json::array({std::bit_cast<std::uint32_t>(v.cellWidthMm), std::bit_cast<std::uint32_t>(v.cellHeightMm), std::bit_cast<std::uint32_t>(v.slotProbability), std::bit_cast<std::uint32_t>(v.softnessMinMm), std::bit_cast<std::uint32_t>(v.softnessMaxMm), std::bit_cast<std::uint32_t>(v.softnessSizeCapFraction), std::bit_cast<std::uint32_t>(v.supportXMm), std::bit_cast<std::uint32_t>(v.supportYMm), std::bit_cast<std::uint32_t>(v.lengthMinMm), std::bit_cast<std::uint32_t>(v.lengthBulkMaxMm), std::bit_cast<std::uint32_t>(v.lengthMaxMm), std::bit_cast<std::uint32_t>(v.lengthTailFraction), std::bit_cast<std::uint32_t>(v.widthMinMm), std::bit_cast<std::uint32_t>(v.widthBulkMaxMm), std::bit_cast<std::uint32_t>(v.widthMaxMm), std::bit_cast<std::uint32_t>(v.widthTailFraction), std::bit_cast<std::uint32_t>(v.driftFraction), std::bit_cast<std::uint32_t>(v.firstKnotMin), std::bit_cast<std::uint32_t>(v.firstKnotMax), std::bit_cast<std::uint32_t>(v.secondKnotMin), std::bit_cast<std::uint32_t>(v.secondKnotMax), std::bit_cast<std::uint32_t>(v.interiorWidthMinFraction), std::bit_cast<std::uint32_t>(v.interiorWidthMaxFraction), std::bit_cast<std::uint32_t>(v.interiorDepthMinFraction), std::bit_cast<std::uint32_t>(v.interiorDepthMaxFraction), std::bit_cast<std::uint32_t>(v.endpointAbruptProbability), std::bit_cast<std::uint32_t>(v.interruptionProbability), std::bit_cast<std::uint32_t>(v.gapCenterMin), std::bit_cast<std::uint32_t>(v.gapCenterMax), std::bit_cast<std::uint32_t>(v.gapSpanMin), std::bit_cast<std::uint32_t>(v.gapSpanMax), std::bit_cast<std::uint32_t>(v.scuffProbability), std::bit_cast<std::uint32_t>(v.scuffLengthMaxMm), std::bit_cast<std::uint32_t>(v.scuffAngleMaxRadians), std::bit_cast<std::uint32_t>(v.strengthMin), std::bit_cast<std::uint32_t>(v.strengthMax)});
        }
    } // namespace
    nlohmann::json effects_frame_fields(const Spektrafilm::FilmJuicerEffectsFrameDescriptor& v) {
        nlohmann::json origins = nlohmann::json::array();
        for (const auto& o : v.origins)
            origins.push_back({o.cellX, o.cellY, std::bit_cast<std::uint32_t>(o.localXMm), std::bit_cast<std::uint32_t>(o.localYMm)});
        return {{"render_extent", {v.renderExtent.x, v.renderExtent.y, v.renderExtent.width, v.renderExtent.height}},
                {"full_frame_extent", {v.fullFrameExtent.x, v.fullFrameExtent.y, v.fullFrameExtent.width, v.fullFrameExtent.height}},
                {"film_dust", effects_policy(v.filmDust)},
                {"film_scratch", effects_policy(v.filmScratch)},
                {"gate_dust", effects_policy(v.gateDust)},
                {"gate_scratch", effects_policy(v.gateScratch)},
                {"origins", origins},
                {"sample_step_x_mm", std::bit_cast<std::uint32_t>(v.sampleStepXMm)},
                {"sample_step_y_mm", std::bit_cast<std::uint32_t>(v.sampleStepYMm)},
                {"roi_offset_x", v.roiOffsetX},
                {"roi_offset_y", v.roiOffsetY},
                {"gate_width", v.gateWidth},
                {"gate_height", v.gateHeight},
                {"session_seed", v.sessionSeed},
                {"clip_token", v.clipToken},
                {"weave_active", v.weaveActive},
                {"weave_dx_px", std::bit_cast<std::uint32_t>(v.weaveDxPx)},
                {"weave_dy_px", std::bit_cast<std::uint32_t>(v.weaveDyPx)},
                {"weave_cos_rot", std::bit_cast<std::uint32_t>(v.weaveCosRot)},
                {"weave_sin_rot", std::bit_cast<std::uint32_t>(v.weaveSinRot)},
                {"film_active", v.filmActive},
                {"gate_transmittance_active", v.gateTransmittanceActive},
                {"gate_output_active", v.gateOutputActive},
                {"requires_full_frame", v.requiresFullFrame},
                {"recipe_hash", v.recipeHash},
                {"hash", v.hash}};
    }
} // namespace JuicerTestSupport
