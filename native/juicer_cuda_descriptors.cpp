#include "juicer_cuda_descriptors.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

    static_assert(static_cast<std::uint32_t>(Spektrafilm::ScanRoute::NegativeDirectScan) == FJ_ROUTE_NEGATIVE_DIRECT);
    static_assert(static_cast<std::uint32_t>(Spektrafilm::ScanRoute::NegativePrintScan) == FJ_ROUTE_NEGATIVE_PRINT);
    static_assert(static_cast<std::uint32_t>(Spektrafilm::ScanRoute::PositiveDirectScan) == FJ_ROUTE_POSITIVE_DIRECT);
    static_assert(static_cast<std::uint32_t>(Spektrafilm::ScanRoute::PositivePrintScan) == FJ_ROUTE_POSITIVE_PRINT);
    static_assert(static_cast<std::uint32_t>(Spektrafilm::ProfilePolarity::Negative) == FJ_POLARITY_NEGATIVE);
    static_assert(static_cast<std::uint32_t>(Spektrafilm::ProfilePolarity::Positive) == FJ_POLARITY_POSITIVE);
    static_assert(static_cast<std::uint32_t>(Spektrafilm::DiffusionFilterFamily::Glimmerglass) == FJ_DIFFUSION_GLIMMERGLASS);
    static_assert(static_cast<std::uint32_t>(Spektrafilm::DiffusionFilterFamily::BlackProMist) == FJ_DIFFUSION_BLACK_PRO_MIST);
    static_assert(static_cast<std::uint32_t>(Spektrafilm::DiffusionFilterFamily::ProMist) == FJ_DIFFUSION_PRO_MIST);
    static_assert(static_cast<std::uint32_t>(Spektrafilm::DiffusionFilterFamily::Cinebloom) == FJ_DIFFUSION_CINEBLOOM);
    static_assert(static_cast<std::uint32_t>(Spektrafilm::DirReferenceOperator::None) == FJ_DIR_OPERATOR_NONE);
    static_assert(static_cast<std::uint32_t>(Spektrafilm::DirReferenceOperator::Identity) == FJ_DIR_OPERATOR_IDENTITY);
    static_assert(static_cast<std::uint32_t>(Spektrafilm::DirReferenceOperator::SpektrafilmSmallFirReflect) == FJ_DIR_OPERATOR_FIR_REFLECT);
    static_assert(static_cast<std::uint32_t>(Spektrafilm::DirReferenceOperator::SpektrafilmLargeYvvReflect) == FJ_DIR_OPERATOR_YVV_REFLECT);
    static_assert(static_cast<std::uint32_t>(ScatterHalationGaussianKind::Identity) == FJ_HALATION_IDENTITY);
    static_assert(static_cast<std::uint32_t>(ScatterHalationGaussianKind::FirReflect) == FJ_HALATION_FIR_REFLECT);
    static_assert(static_cast<std::uint32_t>(ScatterHalationGaussianKind::YvvReplicate) == FJ_HALATION_YVV_REPLICATE);
    static_assert(static_cast<std::uint32_t>(OutputEncoding::ColorSpace::sRGB) == FJ_OUTPUT_SRGB);
    static_assert(static_cast<std::uint32_t>(OutputEncoding::ColorSpace::DCI_P3) == FJ_OUTPUT_DCI_P3);
    static_assert(static_cast<std::uint32_t>(OutputEncoding::ColorSpace::DisplayP3) == FJ_OUTPUT_DISPLAY_P3);
    static_assert(static_cast<std::uint32_t>(OutputEncoding::ColorSpace::AdobeRGB) == FJ_OUTPUT_ADOBE_RGB);
    static_assert(static_cast<std::uint32_t>(OutputEncoding::ColorSpace::ITU_R_BT2020) == FJ_OUTPUT_BT2020);
    static_assert(static_cast<std::uint32_t>(OutputEncoding::ColorSpace::ProPhotoRGB) == FJ_OUTPUT_PROPHOTO_RGB);
    static_assert(static_cast<std::uint32_t>(OutputEncoding::ColorSpace::ACES2065_1) == FJ_OUTPUT_ACES2065_1);
    static_assert(static_cast<std::uint32_t>(OutputEncoding::ColorSpace::DaVinciWideGamutIntermediate) == FJ_OUTPUT_DWG_INTERMEDIATE);
    static_assert(static_cast<std::uint32_t>(OutputEncoding::ColorSpace::Rec709) == FJ_OUTPUT_REC709);

    bool invalid_descriptor(std::string& diagnostic, const char* field) {
        diagnostic = "MalformedPreparedHostData field=";
        diagnostic += field;
        return false;
    }

    template <typename Extent>
    Extent decode_extent(const FjExtent& source) {
        return {source.x, source.y, source.width, source.height};
    }

    DefectDustRecipe decode_dust(const FjDust& source) {
        DefectDustRecipe out{};
        out.cellWidthMm = source.cell_width_mm;
        out.cellHeightMm = source.cell_height_mm;
        out.slotProbability = source.slot_probability;
        out.softnessMinMm = source.softness_min_mm;
        out.softnessMaxMm = source.softness_max_mm;
        out.softnessSizeCapFraction = source.softness_size_cap_fraction;
        out.supportXMm = source.support_x_mm;
        out.supportYMm = source.support_y_mm;
        out.fiberFraction = source.fiber_fraction;
        out.fiberDriftFraction = source.fiber_drift_fraction;
        out.fiberFirstKnotMin = source.fiber_first_knot_min;
        out.fiberFirstKnotMax = source.fiber_first_knot_max;
        out.fiberSecondKnotMin = source.fiber_second_knot_min;
        out.fiberSecondKnotMax = source.fiber_second_knot_max;
        out.fiberInteriorWidthMinFraction = source.fiber_interior_width_min_fraction;
        out.fiberInteriorWidthMaxFraction = source.fiber_interior_width_max_fraction;
        out.diameterMinMm = source.diameter_min_mm;
        out.diameterBulkMaxMm = source.diameter_bulk_max_mm;
        out.diameterMaxMm = source.diameter_max_mm;
        out.diameterTailFraction = source.diameter_tail_fraction;
        out.fiberLengthMinMm = source.fiber_length_min_mm;
        out.fiberLengthMaxMm = source.fiber_length_max_mm;
        out.fiberWidthMinMm = source.fiber_width_min_mm;
        out.fiberWidthMaxMm = source.fiber_width_max_mm;
        out.opacityFaintCumulative = source.opacity_faint_cumulative;
        out.opacityIntermediateCumulative = source.opacity_intermediate_cumulative;
        out.compactOpacityMin = source.compact_opacity_min;
        out.compactOpacityFaintEnd = source.compact_opacity_faint_end;
        out.compactOpacityIntermediateEnd = source.compact_opacity_intermediate_end;
        out.compactOpacityMax = source.compact_opacity_max;
        out.fiberOpacityMin = source.fiber_opacity_min;
        out.fiberOpacityFaintEnd = source.fiber_opacity_faint_end;
        out.fiberOpacityIntermediateEnd = source.fiber_opacity_intermediate_end;
        out.fiberOpacityMax = source.fiber_opacity_max;
        out.compactDominantAspectMin = source.compact_dominant_aspect_min;
        out.compactDominantAspectMax = source.compact_dominant_aspect_max;
        out.compactSubsidiaryScaleMin = source.compact_subsidiary_scale_min;
        out.compactSubsidiaryScaleMax = source.compact_subsidiary_scale_max;
        out.compactSubsidiaryAspectMin = source.compact_subsidiary_aspect_min;
        out.compactSubsidiaryAspectMax = source.compact_subsidiary_aspect_max;
        out.compactSubsidiaryOffsetMax = source.compact_subsidiary_offset_max;
        out.compactSubsidiaryAngleMaxRadians = source.compact_subsidiary_angle_max_radians;
        return out;
    }

    DefectScratchRecipe decode_scratch(const FjScratch& source) {
        DefectScratchRecipe out{};
        out.cellWidthMm = source.cell_width_mm;
        out.cellHeightMm = source.cell_height_mm;
        out.slotProbability = source.slot_probability;
        out.softnessMinMm = source.softness_min_mm;
        out.softnessMaxMm = source.softness_max_mm;
        out.softnessSizeCapFraction = source.softness_size_cap_fraction;
        out.supportXMm = source.support_x_mm;
        out.supportYMm = source.support_y_mm;
        out.lengthMinMm = source.length_min_mm;
        out.lengthBulkMaxMm = source.length_bulk_max_mm;
        out.lengthMaxMm = source.length_max_mm;
        out.lengthTailFraction = source.length_tail_fraction;
        out.widthMinMm = source.width_min_mm;
        out.widthBulkMaxMm = source.width_bulk_max_mm;
        out.widthMaxMm = source.width_max_mm;
        out.widthTailFraction = source.width_tail_fraction;
        out.driftFraction = source.drift_fraction;
        out.firstKnotMin = source.first_knot_min;
        out.firstKnotMax = source.first_knot_max;
        out.secondKnotMin = source.second_knot_min;
        out.secondKnotMax = source.second_knot_max;
        out.interiorWidthMinFraction = source.interior_width_min_fraction;
        out.interiorWidthMaxFraction = source.interior_width_max_fraction;
        out.interiorDepthMinFraction = source.interior_depth_min_fraction;
        out.interiorDepthMaxFraction = source.interior_depth_max_fraction;
        out.endpointAbruptProbability = source.endpoint_abrupt_probability;
        out.interruptionProbability = source.interruption_probability;
        out.gapCenterMin = source.gap_center_min;
        out.gapCenterMax = source.gap_center_max;
        out.gapSpanMin = source.gap_span_min;
        out.gapSpanMax = source.gap_span_max;
        out.scuffProbability = source.scuff_probability;
        out.scuffLengthMaxMm = source.scuff_length_max_mm;
        out.scuffAngleMaxRadians = source.scuff_angle_max_radians;
        out.strengthMin = source.strength_min;
        out.strengthMax = source.strength_max;
        return out;
    }

    bool decode_halation_gaussian(
        const FjHalationGaussian& source,
        ScatterHalationGaussianDescriptor& out,
        std::string& diagnostic) {
        if (source.kind > FJ_HALATION_YVV_REPLICATE || source.radius < 0 ||
            (source.kind == FJ_HALATION_FIR_REFLECT && source.radius > 9)) {
            return invalid_descriptor(diagnostic, "optics.scatter_halation.gaussian");
        }
        out.kind = static_cast<ScatterHalationGaussianKind>(source.kind);
        out.radius = source.radius;
        std::copy_n(source.fir_weights, out.firWeights.size(), out.firWeights.begin());
        out.B = source.feedforward;
        out.B1 = source.feedback[0];
        out.B2 = source.feedback[1];
        out.B3 = source.feedback[2];
        return true;
    }

    bool decode_diffusion(
        const FjDiffusion& source,
        Spektrafilm::DiffusionLinearStage stage,
        Spektrafilm::DiffusionStageFrameDescriptor& out,
        std::string& diagnostic) {
        if (source.family > FJ_DIFFUSION_CINEBLOOM ||
            source.hash == 0 || source.sample_hash == 0) {
            return invalid_descriptor(diagnostic, "optics.diffusion");
        }
        out.stage = stage;
        out.scatterFraction = source.scatter_fraction;
        out.hash = source.hash;
        out.sample.family = static_cast<Spektrafilm::DiffusionFilterFamily>(source.family);
        std::copy_n(source.group_weights, 3, out.sample.groupWeightsCoreHaloBloom.begin());
        std::copy_n(source.group_lambda_um, 3, out.sample.groupCenterLambdaUm.begin());
        out.sample.effectiveWarmth = source.warmth;
        out.sample.spatialScale = source.spatial_scale;
        out.sample.pixelSizeUm = source.pixel_size_um;
        out.sample.radiusPixels = source.radius_px;
        out.sample.hash = source.sample_hash;
        return true;
    }

    bool decode_optics(
        const FjOptics& source,
        JuicerCuda::PreparedDescriptors& out,
        std::string& diagnostic) {
        constexpr std::uint32_t kOpticsFlags =
            FJ_OPTICS_CAMERA_DIFFUSION | FJ_OPTICS_ENLARGER_DIFFUSION | FJ_OPTICS_SCATTER_HALATION;
        if ((source.flags & ~kOpticsFlags) != 0 ||
            (!Spektrafilm::scan_route_is_print(out.route) &&
             (source.flags & FJ_OPTICS_ENLARGER_DIFFUSION) != 0)) {
            return invalid_descriptor(diagnostic, "optics.flags");
        }
        const bool camera = (source.flags & FJ_OPTICS_CAMERA_DIFFUSION) != 0;
        const bool enlarger = (source.flags & FJ_OPTICS_ENLARGER_DIFFUSION) != 0;
        if (camera || enlarger) {
            if (source.diffusion_hash == 0) {
                return invalid_descriptor(diagnostic, "optics.diffusion_hash");
            }
            Spektrafilm::DiffusionFrameSetDescriptor descriptor{};
            descriptor.route = out.route;
            descriptor.fullFrame = {source.full_frame.x, source.full_frame.y, source.full_frame.width, source.full_frame.height};
            descriptor.hash = source.diffusion_hash;
            if (camera) {
                Spektrafilm::DiffusionStageFrameDescriptor stage{};
                if (!decode_diffusion(source.camera, Spektrafilm::DiffusionLinearStage::CameraFilmLinear, stage, diagnostic)) {
                    return false;
                }
                descriptor.camera = stage;
            }
            if (enlarger) {
                Spektrafilm::DiffusionStageFrameDescriptor stage{};
                if (!decode_diffusion(source.enlarger, Spektrafilm::DiffusionLinearStage::EnlargerPrintLinear, stage, diagnostic)) {
                    return false;
                }
                descriptor.enlarger = stage;
            }
            out.diffusion = descriptor;
        } else if (source.diffusion_hash != 0) {
            return invalid_descriptor(diagnostic, "optics.inactive_diffusion");
        }
        if ((source.flags & FJ_OPTICS_SCATTER_HALATION) != 0) {
            if (source.scatter_halation.recipe_hash == 0) {
                return invalid_descriptor(diagnostic, "optics.scatter_halation.recipe_hash");
            }
            ScatterHalationFrameDescriptor descriptor{};
            descriptor.recipeHash = source.scatter_halation.recipe_hash;
            descriptor.scatterAmount = source.scatter_halation.scatter_amount;
            for (std::size_t channel = 0; channel < descriptor.channels.size(); ++channel) {
                const FjHalationChannel& input = source.scatter_halation.channels[channel];
                ScatterHalationChannelDescriptor& output = descriptor.channels[channel];
                output.totalStrength = input.total_strength;
                if (!decode_halation_gaussian(input.core, output.core, diagnostic)) {
                    return false;
                }
                for (std::size_t component = 0; component < output.tail.size(); ++component) {
                    if (!decode_halation_gaussian(input.tail[component], output.tail[component], diagnostic) ||
                        !decode_halation_gaussian(input.bounce[component], output.bounce[component], diagnostic)) {
                        return false;
                    }
                }
            }
            out.halation = descriptor;
        } else if (source.scatter_halation.recipe_hash != 0) {
            return invalid_descriptor(diagnostic, "optics.inactive_scatter_halation");
        }
        return true;
    }

    bool decode_spatial_dir(
        const FjSpatialDir& source,
        JuicerCuda::PreparedDescriptors& out,
        std::string& diagnostic) {
        if (source.component_count > static_cast<std::size_t>(Spektrafilm::DirFilterPlan::kMaxComponents)) {
            return invalid_descriptor(diagnostic, "spatial_dir.component_count");
        }
        if (source.hash == 0) {
            return (source.component_count == 0 && source.recipe_hash == 0) ||
                   invalid_descriptor(diagnostic, "spatial_dir.inactive");
        }
        if (source.component_count == 0 || source.recipe_hash == 0) {
            return invalid_descriptor(diagnostic, "spatial_dir.identity");
        }
        Spektrafilm::SpatialDirDescriptor& descriptor = out.spatialDir;
        descriptor.dirRecipeHash = source.recipe_hash;
        descriptor.support = Spektrafilm::DirDescriptorSupport::Supported;
        descriptor.renderExtent = decode_extent<Spektrafilm::DirFrameExtent>(source.render_extent);
        descriptor.fullFrameExtent = decode_extent<Spektrafilm::DirFrameExtent>(source.full_frame_extent);
        descriptor.filterDomainExtent = decode_extent<Spektrafilm::DirFrameExtent>(source.filter_domain_extent);
        descriptor.traceRouteLabel = Spektrafilm::scan_route_is_print(out.route) ? "print" : "direct";
        descriptor.hash = source.hash;
        descriptor.filterPlan.componentCount = static_cast<int>(source.component_count);
        for (std::size_t index = 0; index < source.component_count; ++index) {
            const FjDirGaussian& input = source.components[index];
            if (input.reference_operator == FJ_DIR_OPERATOR_NONE ||
                input.reference_operator > FJ_DIR_OPERATOR_YVV_REFLECT) {
                return invalid_descriptor(diagnostic, "spatial_dir.reference_operator");
            }
            Spektrafilm::DirGaussianComponentPlan& output = descriptor.filterPlan.components[index];
            output.sigmaPixels = input.sigma_px;
            output.weight = input.weight;
            output.radius = input.radius;
            output.referenceOperator = static_cast<Spektrafilm::DirReferenceOperator>(input.reference_operator);
            output.iir.feedforward = input.feedforward;
            std::copy_n(input.feedback, 3, output.iir.feedback.begin());
            output.iir.normalizationDenominator = input.normalization_denominator;
            output.iir.feedforwardNumerator = input.feedforward_numerator;
            std::copy_n(input.feedback_numerators, 3, output.iir.feedbackNumerators.begin());
            output.boundaryTruncationAccuracy = input.boundary_truncation_accuracy;
            output.boundaryCertificationTolerance = input.boundary_certification_tolerance;
            output.boundaryDerivationVersion = input.boundary_derivation_version;
        }
        Spektrafilm::resolve_spatial_dir_scratch(descriptor);
        return true;
    }

    Spektrafilm::VisualGrainGaussian decode_grain_gaussian(const FjGrainGaussian& source) {
        return {source.sigma_px, source.radius, source.hash};
    }

    bool decode_grain(
        const FjVisualGrain& source,
        JuicerCuda::PreparedDescriptors& out,
        std::string& diagnostic) {
        constexpr std::uint32_t kGrainFlags =
            FJ_GRAIN_ACTIVE | FJ_GRAIN_SUBLAYERS | FJ_GRAIN_FULL_FRAME |
            FJ_GRAIN_AXIS_FINITE_C | FJ_GRAIN_AXIS_FINITE_M | FJ_GRAIN_AXIS_FINITE_Y;
        if ((source.flags & ~kGrainFlags) != 0 ||
            source.debug_view > FJ_GRAIN_DEBUG_MEAN_DENSITY ||
            source.sublayer_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            return invalid_descriptor(diagnostic, "grain.flags_or_count");
        }
        if ((source.flags & FJ_GRAIN_ACTIVE) == 0) {
            return (source.flags == 0 && source.hash == 0 && source.recipe_hash == 0) ||
                   invalid_descriptor(diagnostic, "grain.inactive");
        }
        if (source.hash == 0 || source.recipe_hash == 0) {
            return invalid_descriptor(diagnostic, "grain.identity");
        }
        Spektrafilm::VisualGrainRecipe& recipe = out.grainRecipe;
        recipe.active = true;
        recipe.sublayersActive = (source.flags & FJ_GRAIN_SUBLAYERS) != 0;
        recipe.debugView = static_cast<int>(source.debug_view);
        recipe.nSubLayers = static_cast<int>(source.sublayer_count);
        recipe.hash = source.recipe_hash;
        recipe.densityCurvesLayersHash = source.density_layers_hash;
        std::copy_n(source.micro_structure, 2, recipe.microStructure.begin());
        recipe.clumpTemporalMix = source.clump_temporal_mix;
        recipe.sizeMixScale = source.size_mix_scale;
        recipe.amplitude = source.amplitude;
        recipe.chromaMix = source.chroma_mix;
        recipe.chromaSharedWeight = source.chroma_shared_weight;
        recipe.chromaIndependentWeight = source.chroma_independent_weight;
        std::copy_n(source.particle_density_min_cmy, 3, recipe.visualParticleDensityMinCmy.begin());
        std::copy_n(source.uniformity_cmy, 3, recipe.uniformityCmy.begin());
        constexpr std::uint32_t kAxisFlags[] = {
            FJ_GRAIN_AXIS_FINITE_C, FJ_GRAIN_AXIS_FINITE_M, FJ_GRAIN_AXIS_FINITE_Y};
        for (std::size_t channel = 0; channel < recipe.grainLayerAxisFinite.size(); ++channel) {
            recipe.grainLayerAxisFinite[channel] = (source.flags & kAxisFlags[channel]) != 0;
            std::copy_n(source.layer_axis_block_prefix_max[channel], 16, recipe.grainLayerAxisBlockPrefixMax[channel].begin());
        }
        Spektrafilm::VisualGrainFrameDescriptor descriptor{};
        descriptor.active = true;
        descriptor.capturePolarity = out.capturePolarity;
        descriptor.renderExtent = decode_extent<Spektrafilm::VisualGrainFrameExtent>(source.render_extent);
        descriptor.fullFrameExtent = decode_extent<Spektrafilm::VisualGrainFrameExtent>(source.full_frame_extent);
        descriptor.pixelSizeUm = source.pixel_size_um;
        descriptor.frame0 = source.frame0;
        descriptor.frameAlpha = source.frame_alpha;
        descriptor.seedBase = source.seed_base;
        descriptor.seedBaseNext = source.seed_base_next;
        descriptor.sessionSeed = source.session_seed;
        descriptor.clipToken = source.clip_token;
        descriptor.pitchPx = source.pitch_px;
        descriptor.breathingPeriodFrames = source.breathing_period_frames;
        descriptor.clumpMorphPeriodFrames = source.clump_morph_period_frames;
        descriptor.wangCellMm = source.wang_cell_mm;
        descriptor.breathingAmplitude = source.breathing_amplitude;
        descriptor.breathingCellUmSmall = source.breathing_cell_um_small;
        descriptor.breathingCellUmLarge = source.breathing_cell_um_large;
        descriptor.breathingMix = source.breathing_mix;
        descriptor.breathingDriftUmPerFrame = source.breathing_drift_um_per_frame;
        descriptor.debugScale = source.debug_scale;
        descriptor.effectiveFineWeight = source.size_mix_weights[0];
        descriptor.effectiveMidWeight = source.size_mix_weights[1];
        descriptor.effectiveCoarseWeight = source.size_mix_weights[2];
        descriptor.sizeMixGain = source.size_mix_gain;
        descriptor.requiresFullFrame = (source.flags & FJ_GRAIN_FULL_FRAME) != 0;
        descriptor.recipeHash = source.recipe_hash;
        descriptor.densityCurvesLayersHash = source.density_layers_hash;
        descriptor.hash = source.hash;
        descriptor.scratchShape = Spektrafilm::visual_grain_scratch_shape(recipe);
        std::copy_n(source.density_max_cmy, 3, descriptor.densityMaxCmy.begin());
        std::copy_n(source.n_particles_cmy, 3, descriptor.nParticlesCmy.begin());
        std::copy_n(source.od_particle_cmy, 3, descriptor.odParticleCmy.begin());
        for (std::size_t layer = 0; layer < descriptor.dyeCloud.size(); ++layer) {
            descriptor.correlation[layer] = decode_grain_gaussian(source.correlation[layer]);
            std::copy_n(source.density_min_layers[layer], 3, descriptor.densityMinLayers[layer].begin());
            std::copy_n(source.density_max_layers[layer], 3, descriptor.densityMaxLayers[layer].begin());
            std::copy_n(source.n_particles_layers[layer], 3, descriptor.nParticlesLayers[layer].begin());
            std::copy_n(source.od_particle_layers[layer], 3, descriptor.odParticleLayers[layer].begin());
            for (std::size_t channel = 0; channel < descriptor.dyeCloud[layer].size(); ++channel) {
                descriptor.dyeCloud[layer][channel] = decode_grain_gaussian(source.dye_cloud[layer][channel]);
            }
        }
        out.grain = descriptor;
        return true;
    }

    bool decode_effects(
        const FjFilmEffects& source,
        JuicerCuda::PreparedDescriptors& out,
        std::string& diagnostic) {
        constexpr std::uint32_t kEffectsFlags =
            FJ_EFFECTS_WEAVE | FJ_EFFECTS_FILM | FJ_EFFECTS_GATE_TRANSMITTANCE |
            FJ_EFFECTS_GATE_OUTPUT | FJ_EFFECTS_FULL_FRAME;
        if ((source.flags & ~kEffectsFlags) != 0) {
            return invalid_descriptor(diagnostic, "effects.flags");
        }
        if ((source.flags & (FJ_EFFECTS_FILM | FJ_EFFECTS_GATE_OUTPUT)) == 0) {
            return (source.flags == 0 && source.hash == 0 && source.recipe_hash == 0) ||
                   invalid_descriptor(diagnostic, "effects.inactive");
        }
        if (source.hash == 0 || source.recipe_hash == 0) {
            return invalid_descriptor(diagnostic, "effects.identity");
        }
        Spektrafilm::FilmJuicerEffectsFrameDescriptor descriptor{};
        descriptor.renderExtent = decode_extent<Spektrafilm::FilmJuicerEffectsFrameExtent>(source.render_extent);
        descriptor.fullFrameExtent = decode_extent<Spektrafilm::FilmJuicerEffectsFrameExtent>(source.full_frame_extent);
        descriptor.filmDust = decode_dust(source.film_dust);
        descriptor.filmScratch = decode_scratch(source.film_scratch);
        descriptor.gateDust = decode_dust(source.gate_dust);
        descriptor.gateScratch = decode_scratch(source.gate_scratch);
        for (std::size_t index = 0; index < descriptor.origins.size(); ++index) {
            const FjDefectOrigin& origin = source.origins[index];
            descriptor.origins[index] = {origin.cell_x, origin.cell_y, origin.local_x_mm, origin.local_y_mm};
        }
        descriptor.sampleStepXMm = source.sample_step_x_mm;
        descriptor.sampleStepYMm = source.sample_step_y_mm;
        descriptor.roiOffsetX = source.roi_offset_x;
        descriptor.roiOffsetY = source.roi_offset_y;
        descriptor.gateWidth = source.gate_width;
        descriptor.gateHeight = source.gate_height;
        descriptor.sessionSeed = source.session_seed;
        descriptor.clipToken = source.clip_token;
        descriptor.weaveActive = (source.flags & FJ_EFFECTS_WEAVE) != 0;
        descriptor.weaveDxPx = source.weave_dx_px;
        descriptor.weaveDyPx = source.weave_dy_px;
        descriptor.weaveCosRot = source.weave_cos_rot;
        descriptor.weaveSinRot = source.weave_sin_rot;
        descriptor.filmActive = (source.flags & FJ_EFFECTS_FILM) != 0;
        descriptor.gateTransmittanceActive = (source.flags & FJ_EFFECTS_GATE_TRANSMITTANCE) != 0;
        descriptor.gateOutputActive = (source.flags & FJ_EFFECTS_GATE_OUTPUT) != 0;
        descriptor.requiresFullFrame = (source.flags & FJ_EFFECTS_FULL_FRAME) != 0;
        descriptor.recipeHash = source.recipe_hash;
        descriptor.hash = source.hash;
        out.effects = descriptor;
        return true;
    }

} // namespace

namespace JuicerCuda {

    bool decode_prepared_descriptors(
        const FjPreparedHostData& source,
        PreparedDescriptors& out,
        std::string& diagnostic) {
        out = {};
        diagnostic.clear();
        if (source.route > FJ_ROUTE_POSITIVE_PRINT ||
            source.capture_polarity > FJ_POLARITY_POSITIVE) {
            return invalid_descriptor(diagnostic, "route_or_polarity");
        }
        PreparedDescriptors decoded{};
        decoded.route = static_cast<Spektrafilm::ScanRoute>(source.route);
        decoded.capturePolarity = static_cast<Spektrafilm::ProfilePolarity>(source.capture_polarity);
        if (Spektrafilm::scan_route_metadata(decoded.route).capturePolarity != decoded.capturePolarity) {
            return invalid_descriptor(diagnostic, "route_polarity");
        }
        if ((source.scanner_correction.flags & ~FJ_CORRECTION_ACTIVE) != 0 ||
            (source.scanner_post.flags & ~FJ_POST_GLARE) != 0) {
            return invalid_descriptor(diagnostic, "scanner.flags");
        }
        decoded.scanner.route = decoded.route;
        decoded.scanner.medium = Spektrafilm::scan_route_is_print(decoded.route)
                                     ? Scanner::ScannedMediumKind::Print
                                     : Scanner::ScannedMediumKind::Film;
        decoded.scanner.polarity = decoded.capturePolarity;
        decoded.scanner.densityBoundsHash = source.scanner_lut.density_bounds_hash;
        decoded.scanner.lutResolution = source.scanner_lut.resolution;
        decoded.scanner.hash = source.scanner_lut.hash;
        decoded.correction.route = decoded.route;
        decoded.correction.active = (source.scanner_correction.flags & FJ_CORRECTION_ACTIVE) != 0;
        decoded.correction.xyzSlope = source.scanner_correction.xyz_slope;
        decoded.correction.xyzOffset = source.scanner_correction.xyz_offset;
        decoded.correction.exposureScale = source.scanner_correction.exposure_scale;
        decoded.post.route = decoded.route;
        decoded.post.glareActive = (source.scanner_post.flags & FJ_POST_GLARE) != 0;
        decoded.post.glarePercent = source.scanner_post.glare_percent;
        decoded.post.glareRoughness = source.scanner_post.glare_roughness;
        decoded.post.glareBlurSigmaPx = source.scanner_post.glare_blur_sigma_px;
        decoded.post.glareBlurRadius = source.scanner_post.glare_blur_radius;
        decoded.post.lensBlurSigmaPx = source.scanner_post.lens_blur_sigma_px;
        decoded.post.lensBlurRadius = source.scanner_post.lens_blur_radius;
        decoded.post.unsharpSigmaPx = source.scanner_post.unsharp_sigma_px;
        decoded.post.unsharpRadius = source.scanner_post.unsharp_radius;
        decoded.post.unsharpAmount = source.scanner_post.unsharp_amount;
        decoded.post.hash = source.scanner_post.hash;
        constexpr std::uint32_t kColorFlags =
            FJ_COLOR_ENCODE_CCTF | FJ_COLOR_INPUT_IS_OUTPUT_SPACE | FJ_COLOR_GAMUT_COMPRESSION;
        const FjOutputColor& color = source.output_color;
        if (color.color_space > FJ_OUTPUT_REC709 || (color.flags & ~kColorFlags) != 0) {
            return invalid_descriptor(diagnostic, "output_color.tags");
        }
        std::copy_n(color.cat02, 9, decoded.color.cat02);
        std::copy_n(color.xyz_to_rgb, 9, decoded.color.xyzToRgb);
        std::copy_n(color.illuminant_xyz, 3, decoded.color.illuminantXYZ);
        decoded.color.encoding.colorSpace = static_cast<OutputEncoding::ColorSpace>(color.color_space);
        decoded.color.encoding.applyCctfEncoding = (color.flags & FJ_COLOR_ENCODE_CCTF) != 0;
        decoded.color.encoding.inputIsOutputSpace = (color.flags & FJ_COLOR_INPUT_IS_OUTPUT_SPACE) != 0;
        decoded.color.outputGamutRecipeHash = color.gamut_recipe_hash;
        OutputGamutRecipe& gamut = decoded.outputGamut;
        gamut.enabled = (color.flags & FJ_COLOR_GAMUT_COMPRESSION) != 0;
        gamut.outputColorSpace = static_cast<int>(color.color_space);
        gamut.hash = color.gamut_recipe_hash;
        gamut.lightnessKneeThreshold = color.lightness_knee[0];
        gamut.lightnessKneeLimit = color.lightness_knee[1];
        gamut.lightnessKneePower = color.lightness_knee[2];
        gamut.chromaKneeThreshold = color.chroma_knee[0];
        gamut.chromaKneeLimit = color.chroma_knee[1];
        gamut.chromaKneePower = color.chroma_knee[2];
        std::copy_n(color.native_rgb_to_d65_xyz, 9, gamut.transform.nativeRgbToD65Xyz.begin());
        std::copy_n(color.d65_xyz_to_native_rgb, 9, gamut.transform.d65XyzToNativeRgb.begin());
        gamut.transform.outputColorSpace = decoded.color.encoding.colorSpace;
        if (!decode_optics(source.optics, decoded, diagnostic) ||
            !decode_spatial_dir(source.spatial_dir, decoded, diagnostic) ||
            !decode_grain(source.grain, decoded, diagnostic) ||
            !decode_effects(source.effects, decoded, diagnostic)) {
            return false;
        }
        out = decoded;
        return true;
    }

} // namespace JuicerCuda
