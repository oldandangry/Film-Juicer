#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include <gtest/gtest.h>

#include "prepared_descriptors.h"

namespace {

    // Fixed projection contract only; complete spectral spans and CUDA admission have separate fixtures.
    FjPreparedHostData make_fixed_input() {
        FjPreparedHostData input{};
        input.route = FJ_ROUTE_NEGATIVE_PRINT;
        input.capture_polarity = FJ_POLARITY_NEGATIVE;
        input.scanner_lut = {101, 17, 102};
        input.scanner_correction = {FJ_CORRECTION_ACTIVE, 1.25f, -0.125f, 0.75f};
        input.scanner_post = {
            FJ_POST_GLARE, 2.5f, 0.375f, 1.25f, 4, 0.75f, 2, 1.5f, 5, 0.625f, 103};
        input.output_color.color_space = FJ_OUTPUT_DISPLAY_P3;
        input.output_color.flags =
            FJ_COLOR_ENCODE_CCTF | FJ_COLOR_INPUT_IS_OUTPUT_SPACE | FJ_COLOR_GAMUT_COMPRESSION;
        input.output_color.cat02[1] = 0.125f;
        input.output_color.cat02[3] = -0.25f;
        input.output_color.xyz_to_rgb[2] = 0.375f;
        input.output_color.xyz_to_rgb[6] = -0.5f;
        input.output_color.illuminant_xyz[0] = 0.95f;
        input.output_color.illuminant_xyz[1] = 1.0f;
        input.output_color.illuminant_xyz[2] = 1.08f;
        input.output_color.native_rgb_to_d65_xyz[5] = 0.625f;
        input.output_color.d65_xyz_to_native_rgb[7] = -0.75f;
        input.output_color.lightness_knee[0] = 0.9f;
        input.output_color.lightness_knee[1] = 1.0f;
        input.output_color.lightness_knee[2] = 1.6f;
        input.output_color.chroma_knee[0] = 0.8f;
        input.output_color.chroma_knee[1] = 1.125f;
        input.output_color.chroma_knee[2] = 1.75f;
        input.output_color.gamut_recipe_hash = 104;

        input.optics.flags =
            FJ_OPTICS_CAMERA_DIFFUSION | FJ_OPTICS_ENLARGER_DIFFUSION | FJ_OPTICS_SCATTER_HALATION;
        input.optics.full_frame = {-3, 7, 64, 48};
        input.optics.diffusion_hash = 201;
        input.optics.camera = {
            FJ_DIFFUSION_BLACK_PRO_MIST, 0.25, {0.5, 0.3, 0.2}, {1.25, 4.5, 12.0}, 0.125, 1.5, 5.0, 9, 202, 203};
        input.optics.enlarger = {
            FJ_DIFFUSION_CINEBLOOM, 0.125, {0.6, 0.25, 0.15}, {2.5, 6.0, 20.0}, -0.25, 0.75, 7.5, 11, 204, 205};
        input.optics.scatter_halation.recipe_hash = 206;
        input.optics.scatter_halation.scatter_amount = 0.375f;
        input.optics.scatter_halation.channels[0].total_strength = 0.125f;
        input.optics.scatter_halation.channels[1].total_strength = 0.25f;
        input.optics.scatter_halation.channels[2].total_strength = 0.5f;
        auto& core = input.optics.scatter_halation.channels[0].core;
        core.kind = FJ_HALATION_FIR_REFLECT;
        core.radius = 1;
        core.fir_weights[0] = 0.25f;
        core.fir_weights[1] = 0.5f;
        core.fir_weights[2] = 0.25f;
        auto& tail = input.optics.scatter_halation.channels[1].tail[2];
        tail.kind = FJ_HALATION_YVV_REPLICATE;
        tail.feedforward = 0.625f;
        tail.feedback[0] = 0.5f;
        tail.feedback[1] = -0.1875f;
        tail.feedback[2] = 0.0625f;
        input.optics.scatter_halation.channels[2].bounce[1] = core;

        input.spatial_dir.render_extent = {-3, 7, 64, 48};
        input.spatial_dir.full_frame_extent = {-3, 7, 64, 48};
        input.spatial_dir.filter_domain_extent = {-3, 7, 64, 48};
        input.spatial_dir.component_count = 2;
        input.spatial_dir.recipe_hash = 301;
        input.spatial_dir.hash = 302;
        input.spatial_dir.components[0].sigma_px = 0.75f;
        input.spatial_dir.components[0].weight = 0.75f;
        input.spatial_dir.components[0].radius = 2;
        input.spatial_dir.components[0].reference_operator = FJ_DIR_OPERATOR_FIR_REFLECT;
        auto& yvv = input.spatial_dir.components[1];
        yvv.sigma_px = 4.0f;
        yvv.weight = 0.25f;
        yvv.radius = 12;
        yvv.reference_operator = FJ_DIR_OPERATOR_YVV_REFLECT;
        yvv.feedforward = 0.123456789012345;
        yvv.feedback[0] = 1.25;
        yvv.feedback[1] = -0.5;
        yvv.feedback[2] = 0.126543210987655;
        yvv.normalization_denominator = 12.75;
        yvv.feedforward_numerator = 1.57825;
        yvv.feedback_numerators[0] = 15.9375;
        yvv.feedback_numerators[1] = -6.375;
        yvv.feedback_numerators[2] = 1.61342594;
        yvv.boundary_truncation_accuracy = 1.0e-11;
        yvv.boundary_certification_tolerance = 5.0e-7;
        yvv.boundary_derivation_version = 1;

        input.grain.flags = FJ_GRAIN_ACTIVE | FJ_GRAIN_SUBLAYERS | FJ_GRAIN_FULL_FRAME |
                            FJ_GRAIN_AXIS_FINITE_C | FJ_GRAIN_AXIS_FINITE_Y;
        input.grain.debug_view = FJ_GRAIN_DEBUG_DELTA_MIX;
        input.grain.render_extent = {-3, 7, 64, 48};
        input.grain.full_frame_extent = {-3, 7, 64, 48};
        input.grain.pixel_size_um = 5.0f;
        input.grain.frame0 = -17;
        input.grain.frame_alpha = 0.375f;
        input.grain.seed_base = 401;
        input.grain.seed_base_next = 402;
        input.grain.session_seed = 403;
        input.grain.clip_token = 404;
        input.grain.pitch_px = 2;
        input.grain.breathing_period_frames = 240;
        input.grain.clump_morph_period_frames = 192;
        input.grain.wang_cell_mm = 0.125f;
        input.grain.breathing_amplitude = 0.0625f;
        input.grain.breathing_cell_um_small = 32.0f;
        input.grain.breathing_cell_um_large = 128.0f;
        input.grain.breathing_mix = 0.25f;
        input.grain.breathing_drift_um_per_frame = 0.5f;
        input.grain.debug_scale = 4.0f;
        input.grain.density_max_cmy[0] = 1.5f;
        input.grain.density_max_cmy[1] = 2.5f;
        input.grain.density_max_cmy[2] = 3.5f;
        input.grain.n_particles_cmy[0] = 16.0f;
        input.grain.n_particles_cmy[2] = 64.0f;
        input.grain.od_particle_cmy[1] = 0.03125f;
        input.grain.density_min_layers[1][2] = 0.0625f;
        input.grain.density_max_layers[2][0] = 2.25f;
        input.grain.n_particles_layers[0][1] = 128.0f;
        input.grain.od_particle_layers[2][1] = 0.015625f;
        input.grain.correlation[0] = {0.75f, 2, 405};
        input.grain.correlation[2] = {1.5f, 5, 406};
        input.grain.dye_cloud[1][2] = {0.5f, 2, 407};
        input.grain.size_mix_weights[0] = 0.5f;
        input.grain.size_mix_weights[1] = 0.375f;
        input.grain.size_mix_weights[2] = 0.125f;
        input.grain.size_mix_gain = 1.25f;
        input.grain.sublayer_count = 3;
        input.grain.micro_structure[0] = 0.2f;
        input.grain.micro_structure[1] = 30.0f;
        input.grain.clump_temporal_mix = 0.25f;
        input.grain.size_mix_scale = 1.75f;
        input.grain.amplitude = 0.875f;
        input.grain.chroma_mix = 0.25f;
        input.grain.chroma_shared_weight = 0.75f;
        input.grain.chroma_independent_weight = 0.5f;
        input.grain.particle_density_min_cmy[0] = 0.07f;
        input.grain.particle_density_min_cmy[2] = 0.12f;
        input.grain.uniformity_cmy[1] = 0.97f;
        input.grain.layer_axis_block_prefix_max[0][15] = 2.125f;
        input.grain.layer_axis_block_prefix_max[2][4] = 3.25f;
        input.grain.recipe_hash = 408;
        input.grain.density_layers_hash = 409;
        input.grain.hash = 410;

        input.effects.render_extent = {-3, 7, 64, 48};
        input.effects.full_frame_extent = {-3, 7, 64, 48};
        input.effects.flags = FJ_EFFECTS_WEAVE | FJ_EFFECTS_FILM |
                              FJ_EFFECTS_GATE_TRANSMITTANCE | FJ_EFFECTS_GATE_OUTPUT | FJ_EFFECTS_FULL_FRAME;
        input.effects.film_dust.cell_width_mm = 0.25f;
        input.effects.film_dust.support_x_mm = 0.125f;
        input.effects.film_dust.fiber_interior_width_max_fraction = 0.875f;
        input.effects.gate_dust.cell_width_mm = 0.5f;
        input.effects.gate_dust.compact_subsidiary_angle_max_radians = 0.375f;
        input.effects.film_scratch.interior_width_min_fraction = 0.25f;
        input.effects.film_scratch.interior_width_max_fraction = 0.75f;
        input.effects.film_scratch.interior_depth_min_fraction = 0.125f;
        input.effects.film_scratch.interior_depth_max_fraction = 0.875f;
        input.effects.gate_scratch.scuff_angle_max_radians = 0.625f;
        input.effects.gate_scratch.strength_max = 0.9375f;
        input.effects.origins[0] = {-1234567890123LL, 7123456789012LL, 0.125f, 0.25f};
        input.effects.origins[3] = {4321, -9876, 0.5f, 0.75f};
        input.effects.sample_step_x_mm = 0.005f;
        input.effects.sample_step_y_mm = 0.0075f;
        input.effects.roi_offset_x = -3;
        input.effects.roi_offset_y = 7;
        input.effects.gate_width = 68;
        input.effects.gate_height = 54;
        input.effects.session_seed = 501;
        input.effects.clip_token = 502;
        input.effects.weave_dx_px = -0.25f;
        input.effects.weave_dy_px = 0.125f;
        input.effects.weave_cos_rot = 0.999f;
        input.effects.weave_sin_rot = 0.03125f;
        input.effects.recipe_hash = 503;
        input.effects.hash = 504;
        return input;
    }

    TEST(PreparedProjection, RoutesKeepCapturePolarityAndSelectedMedium) {
        struct RouteCase {
            std::uint32_t route;
            std::uint32_t polarity;
            Spektrafilm::ScanRoute nativeRoute;
            Spektrafilm::ProfilePolarity nativePolarity;
            Scanner::ScannedMediumKind medium;
        };
        const RouteCase cases[] = {
            {FJ_ROUTE_NEGATIVE_DIRECT, FJ_POLARITY_NEGATIVE, Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::ProfilePolarity::Negative, Scanner::ScannedMediumKind::Film},
            {FJ_ROUTE_NEGATIVE_PRINT, FJ_POLARITY_NEGATIVE, Spektrafilm::ScanRoute::NegativePrintScan, Spektrafilm::ProfilePolarity::Negative, Scanner::ScannedMediumKind::Print},
            {FJ_ROUTE_POSITIVE_DIRECT, FJ_POLARITY_POSITIVE, Spektrafilm::ScanRoute::PositiveDirectScan, Spektrafilm::ProfilePolarity::Positive, Scanner::ScannedMediumKind::Film},
            {FJ_ROUTE_POSITIVE_PRINT, FJ_POLARITY_POSITIVE, Spektrafilm::ScanRoute::PositivePrintScan, Spektrafilm::ProfilePolarity::Positive, Scanner::ScannedMediumKind::Print}};
        for (const auto& item : cases) {
            SCOPED_TRACE(item.route);
            FjPreparedHostData input{};
            input.route = item.route;
            input.capture_polarity = item.polarity;
            JuicerCuda::PreparedDescriptors actual;
            std::string diagnostic;
            ASSERT_TRUE(JuicerCuda::decode_prepared_descriptors(input, actual, diagnostic)) << diagnostic;
            EXPECT_EQ(actual.route, item.nativeRoute);
            EXPECT_EQ(actual.capturePolarity, item.nativePolarity);
            EXPECT_EQ(actual.scanner.medium, item.medium);
            EXPECT_EQ(actual.scanner.polarity, item.nativePolarity);
            EXPECT_FALSE(actual.diffusion);
            EXPECT_FALSE(actual.halation);
            EXPECT_FALSE(actual.grain);
            EXPECT_FALSE(actual.effects);
            EXPECT_EQ(actual.spatialDir.hash, 0u);
        }
    }

    TEST(PreparedProjection, ScannerAndColorPreserveCoefficientsAndIdentities) {
        const FjPreparedHostData input = make_fixed_input();
        JuicerCuda::PreparedDescriptors actual;
        std::string diagnostic;
        ASSERT_TRUE(JuicerCuda::decode_prepared_descriptors(input, actual, diagnostic)) << diagnostic;
        EXPECT_EQ(actual.scanner.densityBoundsHash, 101u);
        EXPECT_EQ(actual.scanner.lutResolution, 17u);
        EXPECT_EQ(actual.scanner.hash, 102u);
        EXPECT_TRUE(actual.correction.active);
        EXPECT_FLOAT_EQ(actual.correction.xyzSlope, 1.25f);
        EXPECT_FLOAT_EQ(actual.correction.xyzOffset, -0.125f);
        EXPECT_FLOAT_EQ(actual.correction.exposureScale, 0.75f);
        EXPECT_TRUE(actual.post.glareActive);
        EXPECT_FLOAT_EQ(actual.post.glarePercent, 2.5f);
        EXPECT_FLOAT_EQ(actual.post.glareRoughness, 0.375f);
        EXPECT_EQ(actual.post.glareBlurRadius, 4);
        EXPECT_EQ(actual.post.lensBlurRadius, 2);
        EXPECT_EQ(actual.post.unsharpRadius, 5);
        EXPECT_FLOAT_EQ(actual.post.unsharpAmount, 0.625f);
        EXPECT_EQ(actual.post.hash, 103u);
        EXPECT_EQ(actual.color.encoding.colorSpace, OutputEncoding::ColorSpace::DisplayP3);
        EXPECT_TRUE(actual.color.encoding.applyCctfEncoding);
        EXPECT_TRUE(actual.color.encoding.inputIsOutputSpace);
        EXPECT_FLOAT_EQ(actual.color.cat02[1], 0.125f);
        EXPECT_FLOAT_EQ(actual.color.cat02[3], -0.25f);
        EXPECT_FLOAT_EQ(actual.color.xyzToRgb[2], 0.375f);
        EXPECT_FLOAT_EQ(actual.color.xyzToRgb[6], -0.5f);
        EXPECT_FLOAT_EQ(actual.color.illuminantXYZ[2], 1.08f);
        EXPECT_EQ(actual.color.outputGamutRecipeHash, 104u);
        EXPECT_TRUE(actual.outputGamut.enabled);
        EXPECT_EQ(actual.outputGamut.hash, 104u);
        EXPECT_FLOAT_EQ(actual.outputGamut.transform.nativeRgbToD65Xyz[5], 0.625f);
        EXPECT_FLOAT_EQ(actual.outputGamut.transform.d65XyzToNativeRgb[7], -0.75f);
        EXPECT_FLOAT_EQ(actual.outputGamut.lightnessKneePower, 1.6f);
        EXPECT_FLOAT_EQ(actual.outputGamut.chromaKneeThreshold, 0.8f);
        EXPECT_FLOAT_EQ(actual.outputGamut.chromaKneeLimit, 1.125f);
    }

    TEST(PreparedProjection, OpticsKeepStageAndRgbChannelOrder) {
        const FjPreparedHostData input = make_fixed_input();
        JuicerCuda::PreparedDescriptors actual;
        std::string diagnostic;
        ASSERT_TRUE(JuicerCuda::decode_prepared_descriptors(input, actual, diagnostic)) << diagnostic;
        ASSERT_TRUE(actual.diffusion);
        ASSERT_TRUE(actual.diffusion->camera);
        ASSERT_TRUE(actual.diffusion->enlarger);
        EXPECT_EQ(actual.diffusion->hash, 201u);
        EXPECT_EQ(actual.diffusion->fullFrame.originX, -3);
        EXPECT_EQ(actual.diffusion->fullFrame.height, 48);
        const auto& camera = *actual.diffusion->camera;
        const auto& enlarger = *actual.diffusion->enlarger;
        EXPECT_EQ(camera.stage, Spektrafilm::DiffusionLinearStage::CameraFilmLinear);
        EXPECT_EQ(enlarger.stage, Spektrafilm::DiffusionLinearStage::EnlargerPrintLinear);
        EXPECT_EQ(camera.sample.family, Spektrafilm::DiffusionFilterFamily::BlackProMist);
        EXPECT_EQ(enlarger.sample.family, Spektrafilm::DiffusionFilterFamily::Cinebloom);
        EXPECT_DOUBLE_EQ(camera.sample.groupWeightsCoreHaloBloom[1], 0.3);
        EXPECT_DOUBLE_EQ(camera.sample.groupCenterLambdaUm[2], 12.0);
        EXPECT_DOUBLE_EQ(enlarger.sample.effectiveWarmth, -0.25);
        EXPECT_EQ(camera.sample.hash, 202u);
        EXPECT_EQ(camera.hash, 203u);
        EXPECT_EQ(enlarger.sample.hash, 204u);
        EXPECT_EQ(enlarger.hash, 205u);
        ASSERT_TRUE(actual.halation);
        EXPECT_EQ(actual.halation->recipeHash, 206u);
        EXPECT_FLOAT_EQ(actual.halation->scatterAmount, 0.375f);
        EXPECT_FLOAT_EQ(actual.halation->channels[0].totalStrength, 0.125f);
        EXPECT_FLOAT_EQ(actual.halation->channels[1].totalStrength, 0.25f);
        EXPECT_FLOAT_EQ(actual.halation->channels[2].totalStrength, 0.5f);
        EXPECT_EQ(actual.halation->channels[0].core.kind, ScatterHalationGaussianKind::FirReflect);
        EXPECT_FLOAT_EQ(actual.halation->channels[0].core.firWeights[1], 0.5f);
        const auto& tail = actual.halation->channels[1].tail[2];
        EXPECT_EQ(tail.kind, ScatterHalationGaussianKind::YvvReplicate);
        EXPECT_FLOAT_EQ(tail.B, 0.625f);
        EXPECT_FLOAT_EQ(tail.B1, 0.5f);
        EXPECT_FLOAT_EQ(tail.B2, -0.1875f);
        EXPECT_FLOAT_EQ(tail.B3, 0.0625f);
        EXPECT_EQ(actual.halation->channels[2].bounce[1].radius, 1);
    }

    TEST(PreparedProjection, DirKeepsDoubleCoefficientsAndNativeScratchPolicy) {
        const FjPreparedHostData input = make_fixed_input();
        JuicerCuda::PreparedDescriptors actual;
        std::string diagnostic;
        ASSERT_TRUE(JuicerCuda::decode_prepared_descriptors(input, actual, diagnostic)) << diagnostic;
        const auto& dir = actual.spatialDir;
        EXPECT_EQ(dir.dirRecipeHash, 301u);
        EXPECT_EQ(dir.hash, 302u);
        EXPECT_EQ(dir.renderExtent.x, -3);
        EXPECT_EQ(dir.filterDomainExtent.height, 48);
        EXPECT_EQ(dir.filterPlan.componentCount, 2);
        EXPECT_EQ(dir.filterPlan.components[0].referenceOperator, Spektrafilm::DirReferenceOperator::SpektrafilmSmallFirReflect);
        const auto& yvv = dir.filterPlan.components[1];
        EXPECT_EQ(yvv.referenceOperator, Spektrafilm::DirReferenceOperator::SpektrafilmLargeYvvReflect);
        EXPECT_DOUBLE_EQ(yvv.iir.feedforward, 0.123456789012345);
        EXPECT_DOUBLE_EQ(yvv.iir.feedback[1], -0.5);
        EXPECT_DOUBLE_EQ(yvv.iir.normalizationDenominator, 12.75);
        EXPECT_DOUBLE_EQ(yvv.iir.feedforwardNumerator, 1.57825);
        EXPECT_DOUBLE_EQ(yvv.iir.feedbackNumerators[2], 1.61342594);
        EXPECT_DOUBLE_EQ(yvv.boundaryTruncationAccuracy, 1.0e-11);
        EXPECT_DOUBLE_EQ(yvv.boundaryCertificationTolerance, 5.0e-7);
        EXPECT_EQ(yvv.boundaryDerivationVersion, 1u);
        EXPECT_EQ(dir.scratchTier, Spektrafilm::DirScratchTier::Tier1IChannels);
        EXPECT_EQ(dir.targetScratchTier, Spektrafilm::DirScratchTier::Tier2);
        EXPECT_EQ(dir.planeRoles.filterTempPlanes, 3);
        EXPECT_EQ(dir.planeRoles.cachedLogRawPlanes, 0);
        EXPECT_EQ(dir.targetPlaneRoles.cachedLogRawPlanes, 3);
        EXPECT_STREQ(dir.traceRouteLabel, "print");
    }

    TEST(PreparedProjection, GrainKeepsTemporalLayerAndCmyDistinctions) {
        const FjPreparedHostData input = make_fixed_input();
        JuicerCuda::PreparedDescriptors actual;
        std::string diagnostic;
        ASSERT_TRUE(JuicerCuda::decode_prepared_descriptors(input, actual, diagnostic)) << diagnostic;
        ASSERT_TRUE(actual.grain);
        const auto& grain = *actual.grain;
        const auto& recipe = actual.grainRecipe;
        EXPECT_EQ(grain.frame0, -17);
        EXPECT_FLOAT_EQ(grain.frameAlpha, 0.375f);
        EXPECT_EQ(grain.seedBase, 401u);
        EXPECT_EQ(grain.seedBaseNext, 402u);
        EXPECT_EQ(grain.sessionSeed, 403u);
        EXPECT_EQ(grain.clipToken, 404u);
        EXPECT_EQ(grain.capturePolarity, Spektrafilm::ProfilePolarity::Negative);
        EXPECT_EQ(grain.scratchShape, Spektrafilm::VisualGrainScratchShape::StreamedLayersShared);
        EXPECT_TRUE(grain.requiresFullFrame);
        EXPECT_FLOAT_EQ(grain.densityMaxCmy[0], 1.5f);
        EXPECT_FLOAT_EQ(grain.densityMaxCmy[2], 3.5f);
        EXPECT_FLOAT_EQ(grain.nParticlesCmy[2], 64.0f);
        EXPECT_FLOAT_EQ(grain.odParticleCmy[1], 0.03125f);
        EXPECT_FLOAT_EQ(grain.densityMinLayers[1][2], 0.0625f);
        EXPECT_FLOAT_EQ(grain.densityMaxLayers[2][0], 2.25f);
        EXPECT_FLOAT_EQ(grain.nParticlesLayers[0][1], 128.0f);
        EXPECT_FLOAT_EQ(grain.odParticleLayers[2][1], 0.015625f);
        EXPECT_EQ(grain.correlation[2].hash, 406u);
        EXPECT_EQ(grain.dyeCloud[1][2].hash, 407u);
        EXPECT_EQ(grain.recipeHash, 408u);
        EXPECT_EQ(grain.densityCurvesLayersHash, 409u);
        EXPECT_EQ(grain.hash, 410u);
        EXPECT_TRUE(recipe.active);
        EXPECT_TRUE(recipe.sublayersActive);
        EXPECT_EQ(recipe.nSubLayers, 3);
        EXPECT_TRUE(recipe.grainLayerAxisFinite[0]);
        EXPECT_FALSE(recipe.grainLayerAxisFinite[1]);
        EXPECT_TRUE(recipe.grainLayerAxisFinite[2]);
        EXPECT_FLOAT_EQ(recipe.grainLayerAxisBlockPrefixMax[0][15], 2.125f);
        EXPECT_FLOAT_EQ(recipe.grainLayerAxisBlockPrefixMax[2][4], 3.25f);
        EXPECT_FLOAT_EQ(recipe.visualParticleDensityMinCmy[2], 0.12f);
        EXPECT_FLOAT_EQ(recipe.microStructure[0], 0.2f);
        EXPECT_FLOAT_EQ(recipe.microStructure[1], 30.0f);
        EXPECT_FLOAT_EQ(recipe.chromaSharedWeight, 0.75f);
        EXPECT_FLOAT_EQ(recipe.chromaIndependentWeight, 0.5f);
    }

    TEST(PreparedProjection, EffectsKeepFilmGateAndPhysicalCoordinatesDistinct) {
        const FjPreparedHostData input = make_fixed_input();
        JuicerCuda::PreparedDescriptors actual;
        std::string diagnostic;
        ASSERT_TRUE(JuicerCuda::decode_prepared_descriptors(input, actual, diagnostic)) << diagnostic;
        ASSERT_TRUE(actual.effects);
        const auto& effects = *actual.effects;
        EXPECT_FLOAT_EQ(effects.filmDust.cellWidthMm, 0.25f);
        EXPECT_FLOAT_EQ(effects.gateDust.cellWidthMm, 0.5f);
        EXPECT_FLOAT_EQ(effects.filmDust.fiberInteriorWidthMaxFraction, 0.875f);
        EXPECT_FLOAT_EQ(effects.gateDust.compactSubsidiaryAngleMaxRadians, 0.375f);
        EXPECT_FLOAT_EQ(effects.filmScratch.interiorWidthMinFraction, 0.25f);
        EXPECT_FLOAT_EQ(effects.filmScratch.interiorWidthMaxFraction, 0.75f);
        EXPECT_FLOAT_EQ(effects.filmScratch.interiorDepthMinFraction, 0.125f);
        EXPECT_FLOAT_EQ(effects.filmScratch.interiorDepthMaxFraction, 0.875f);
        EXPECT_FLOAT_EQ(effects.gateScratch.scuffAngleMaxRadians, 0.625f);
        EXPECT_EQ(effects.origins[0].cellX, -1234567890123LL);
        EXPECT_EQ(effects.origins[0].cellY, 7123456789012LL);
        EXPECT_EQ(effects.origins[3].cellY, -9876);
        EXPECT_FLOAT_EQ(effects.origins[3].localYMm, 0.75f);
        EXPECT_FLOAT_EQ(effects.sampleStepYMm, 0.0075f);
        EXPECT_EQ(effects.roiOffsetX, -3);
        EXPECT_EQ(effects.gateWidth, 68);
        EXPECT_EQ(effects.sessionSeed, 501u);
        EXPECT_EQ(effects.clipToken, 502u);
        EXPECT_TRUE(effects.filmActive);
        EXPECT_TRUE(effects.gateTransmittanceActive);
        EXPECT_TRUE(effects.gateOutputActive);
        EXPECT_TRUE(effects.weaveActive);
        EXPECT_TRUE(effects.requiresFullFrame);
        EXPECT_FLOAT_EQ(effects.weaveDxPx, -0.25f);
        EXPECT_FLOAT_EQ(effects.weaveSinRot, 0.03125f);
        EXPECT_EQ(effects.recipeHash, 503u);
        EXPECT_EQ(effects.hash, 504u);
    }

    TEST(PreparedProjection, EncodeUsesNativeValuesAndPreservesBorrowedTables) {
        JuicerCuda::PreparedDescriptors source{};
        source.route = Spektrafilm::ScanRoute::PositiveDirectScan;
        source.capturePolarity = Spektrafilm::ProfilePolarity::Positive;
        source.scanner.densityBoundsHash = 601;
        source.scanner.lutResolution = 33;
        source.scanner.hash = 602;
        source.correction.active = true;
        source.correction.xyzOffset = -0.0625f;
        source.post.glareBlurRadius = 7;
        source.post.hash = 603;
        source.color.encoding = {OutputEncoding::ColorSpace::Rec709, false, true};
        source.color.cat02[7] = -0.375f;
        source.color.xyzToRgb[1] = 0.875f;
        source.outputGamut.enabled = true;
        source.outputGamut.hash = 604;
        source.outputGamut.chromaKneePower = 1.875f;
        source.outputGamut.transform.nativeRgbToD65Xyz[2] = -0.625f;
        source.diffusion.emplace();
        source.diffusion->fullFrame = {-5, 11, 80, 40};
        source.diffusion->hash = 605;
        source.diffusion->camera.emplace();
        source.diffusion->camera->stage = Spektrafilm::DiffusionLinearStage::CameraFilmLinear;
        source.diffusion->camera->sample.family = Spektrafilm::DiffusionFilterFamily::ProMist;
        source.diffusion->camera->sample.groupCenterLambdaUm[1] = 6.75;
        source.diffusion->camera->sample.hash = 606;
        source.diffusion->camera->hash = 607;
        source.halation.emplace();
        source.halation->recipeHash = 608;
        source.halation->channels[2].bounce[0].kind = ScatterHalationGaussianKind::YvvReplicate;
        source.halation->channels[2].bounce[0].B2 = -0.3125f;
        source.spatialDir.hash = 609;
        source.spatialDir.dirRecipeHash = 610;
        source.spatialDir.filterPlan.componentCount = 1;
        source.spatialDir.filterPlan.components[0].referenceOperator = Spektrafilm::DirReferenceOperator::Identity;
        source.spatialDir.filterPlan.components[0].iir.feedbackNumerators[1] = -7.25;
        source.grain.emplace();
        source.grain->active = true;
        source.grain->seedBaseNext = 611;
        source.grain->densityMinLayers[2][0] = 0.09375f;
        source.grain->dyeCloud[0][2] = {1.25f, 4, 612};
        source.grain->recipeHash = 613;
        source.grain->hash = 614;
        source.grainRecipe.active = true;
        source.grainRecipe.nSubLayers = 2;
        source.grainRecipe.grainLayerAxisFinite[1] = true;
        source.grainRecipe.grainLayerAxisBlockPrefixMax[1][14] = 2.875f;
        source.grainRecipe.chromaIndependentWeight = 0.625f;
        source.effects.emplace();
        source.effects->filmActive = true;
        source.effects->filmScratch.interiorDepthMaxFraction = 0.8125f;
        source.effects->gateDust.supportYMm = 0.1875f;
        source.effects->origins[2] = {-4321, 1234, 0.0625f, 0.125f};
        source.effects->recipeHash = 615;
        source.effects->hash = 616;

        const std::array<float, 3> borrowed{{0.125f, 0.25f, 0.5f}};
        FjPreparedHostData actual{};
        actual.output_color.gamut_cmax = {borrowed.data(), borrowed.size()};
        actual.output_color.gamut_table_hash = 617;
        actual.film_development.log_exposure = {borrowed.data(), borrowed.size()};
        JuicerCudaTest::encode_prepared_descriptors(source, actual);
        EXPECT_EQ(actual.route, FJ_ROUTE_POSITIVE_DIRECT);
        EXPECT_EQ(actual.capture_polarity, FJ_POLARITY_POSITIVE);
        EXPECT_EQ(actual.scanner_lut.density_bounds_hash, 601u);
        EXPECT_EQ(actual.scanner_lut.resolution, 33u);
        EXPECT_EQ(actual.scanner_lut.hash, 602u);
        EXPECT_EQ(actual.scanner_correction.flags, FJ_CORRECTION_ACTIVE);
        EXPECT_FLOAT_EQ(actual.scanner_correction.xyz_offset, -0.0625f);
        EXPECT_EQ(actual.scanner_post.glare_blur_radius, 7);
        EXPECT_EQ(actual.scanner_post.hash, 603u);
        EXPECT_EQ(actual.output_color.color_space, FJ_OUTPUT_REC709);
        EXPECT_EQ(actual.output_color.flags, FJ_COLOR_INPUT_IS_OUTPUT_SPACE | FJ_COLOR_GAMUT_COMPRESSION);
        EXPECT_FLOAT_EQ(actual.output_color.cat02[7], -0.375f);
        EXPECT_FLOAT_EQ(actual.output_color.xyz_to_rgb[1], 0.875f);
        EXPECT_FLOAT_EQ(actual.output_color.chroma_knee[2], 1.875f);
        EXPECT_FLOAT_EQ(actual.output_color.native_rgb_to_d65_xyz[2], -0.625f);
        EXPECT_EQ(actual.output_color.gamut_recipe_hash, 604u);
        EXPECT_EQ(actual.output_color.gamut_cmax.data, borrowed.data());
        EXPECT_EQ(actual.output_color.gamut_cmax.count, 3u);
        EXPECT_EQ(actual.output_color.gamut_table_hash, 617u);
        EXPECT_EQ(actual.film_development.log_exposure.data, borrowed.data());
        EXPECT_EQ(actual.optics.flags, FJ_OPTICS_CAMERA_DIFFUSION | FJ_OPTICS_SCATTER_HALATION);
        EXPECT_EQ(actual.optics.full_frame.x, -5);
        EXPECT_EQ(actual.optics.camera.family, FJ_DIFFUSION_PRO_MIST);
        EXPECT_DOUBLE_EQ(actual.optics.camera.group_lambda_um[1], 6.75);
        EXPECT_EQ(actual.optics.camera.sample_hash, 606u);
        EXPECT_EQ(actual.optics.camera.hash, 607u);
        EXPECT_EQ(actual.optics.scatter_halation.recipe_hash, 608u);
        EXPECT_EQ(actual.optics.scatter_halation.channels[2].bounce[0].kind, FJ_HALATION_YVV_REPLICATE);
        EXPECT_FLOAT_EQ(actual.optics.scatter_halation.channels[2].bounce[0].feedback[1], -0.3125f);
        EXPECT_EQ(actual.spatial_dir.hash, 609u);
        EXPECT_EQ(actual.spatial_dir.component_count, 1u);
        EXPECT_EQ(actual.spatial_dir.components[0].reference_operator, FJ_DIR_OPERATOR_IDENTITY);
        EXPECT_DOUBLE_EQ(actual.spatial_dir.components[0].feedback_numerators[1], -7.25);
        EXPECT_EQ(actual.grain.flags, FJ_GRAIN_ACTIVE | FJ_GRAIN_AXIS_FINITE_M);
        EXPECT_EQ(actual.grain.seed_base_next, 611u);
        EXPECT_FLOAT_EQ(actual.grain.density_min_layers[2][0], 0.09375f);
        EXPECT_EQ(actual.grain.dye_cloud[0][2].hash, 612u);
        EXPECT_EQ(actual.grain.recipe_hash, 613u);
        EXPECT_EQ(actual.grain.hash, 614u);
        EXPECT_EQ(actual.grain.sublayer_count, 2u);
        EXPECT_FLOAT_EQ(actual.grain.layer_axis_block_prefix_max[1][14], 2.875f);
        EXPECT_FLOAT_EQ(actual.grain.chroma_independent_weight, 0.625f);
        EXPECT_EQ(actual.effects.flags, FJ_EFFECTS_FILM);
        EXPECT_FLOAT_EQ(actual.effects.film_scratch.interior_depth_max_fraction, 0.8125f);
        EXPECT_FLOAT_EQ(actual.effects.gate_dust.support_y_mm, 0.1875f);
        EXPECT_EQ(actual.effects.origins[2].cell_x, -4321);
        EXPECT_EQ(actual.effects.hash, 616u);
    }

    TEST(PreparedProjection, EncodeClearsUnselectedFixedFamilies) {
        JuicerCuda::PreparedDescriptors source{};
        source.route = Spektrafilm::ScanRoute::NegativeDirectScan;
        source.capturePolarity = Spektrafilm::ProfilePolarity::Negative;
        FjPreparedHostData actual = make_fixed_input();
        JuicerCudaTest::encode_prepared_descriptors(source, actual);
        EXPECT_EQ(actual.optics.flags, 0u);
        EXPECT_EQ(actual.optics.diffusion_hash, 0u);
        EXPECT_EQ(actual.optics.scatter_halation.recipe_hash, 0u);
        EXPECT_EQ(actual.spatial_dir.component_count, 0u);
        EXPECT_EQ(actual.spatial_dir.hash, 0u);
        EXPECT_EQ(actual.grain.flags, 0u);
        EXPECT_EQ(actual.grain.hash, 0u);
        EXPECT_EQ(actual.effects.flags, 0u);
        EXPECT_EQ(actual.effects.hash, 0u);
    }

    TEST(PreparedProjection, MalformedForeignDescriptorsClearPreviousOutput) {
        struct InvalidCase {
            const char* name;
            void (*mutate)(FjPreparedHostData&);
        };
        const InvalidCase cases[] = {
            {"unknown route", [](FjPreparedHostData& value) {
                 value.route = 4;
             }},
            {"unsupported polarity", [](FjPreparedHostData& value) {
                 value.capture_polarity = FJ_POLARITY_UNSUPPORTED;
             }},
            {"route polarity mismatch", [](FjPreparedHostData& value) {
                 value.capture_polarity = FJ_POLARITY_POSITIVE;
             }},
            {"correction flags", [](FjPreparedHostData& value) {
                 value.scanner_correction.flags = 2;
             }},
            {"post flags", [](FjPreparedHostData& value) {
                 value.scanner_post.flags = 2;
             }},
            {"color enum", [](FjPreparedHostData& value) {
                 value.output_color.color_space = 9;
             }},
            {"color flags", [](FjPreparedHostData& value) {
                 value.output_color.flags = 8;
             }},
            {"optics flags", [](FjPreparedHostData& value) {
                 value.optics.flags = 8;
             }},
            {"direct enlarger", [](FjPreparedHostData& value) {
                 value.route = FJ_ROUTE_NEGATIVE_DIRECT;
             }},
            {"diffusion identity", [](FjPreparedHostData& value) {
                 value.optics.diffusion_hash = 0;
             }},
            {"inactive diffusion", [](FjPreparedHostData& value) {
                 value.optics.flags = FJ_OPTICS_SCATTER_HALATION;
             }},
            {"camera enum", [](FjPreparedHostData& value) {
                 value.optics.camera.family = 4;
             }},
            {"enlarger enum", [](FjPreparedHostData& value) {
                 value.optics.enlarger.family = 4;
             }},
            {"camera sample identity", [](FjPreparedHostData& value) {
                 value.optics.camera.sample_hash = 0;
             }},
            {"enlarger stage identity", [](FjPreparedHostData& value) {
                 value.optics.enlarger.hash = 0;
             }},
            {"halation identity", [](FjPreparedHostData& value) {
                 value.optics.scatter_halation.recipe_hash = 0;
             }},
            {"inactive halation", [](FjPreparedHostData& value) {
                 value.optics.flags &= ~FJ_OPTICS_SCATTER_HALATION;
             }},
            {"halation enum", [](FjPreparedHostData& value) {
                 value.optics.scatter_halation.channels[0].core.kind = 3;
             }},
            {"halation negative radius", [](FjPreparedHostData& value) {
                 value.optics.scatter_halation.channels[0].core.radius = -1;
             }},
            {"halation FIR storage", [](FjPreparedHostData& value) {
                 value.optics.scatter_halation.channels[0].core.radius = 10;
             }},
            {"DIR count", [](FjPreparedHostData& value) {
                 value.spatial_dir.component_count = 5;
             }},
            {"DIR empty active", [](FjPreparedHostData& value) {
                 value.spatial_dir.component_count = 0;
             }},
            {"DIR recipe identity", [](FjPreparedHostData& value) {
                 value.spatial_dir.recipe_hash = 0;
             }},
            {"DIR inactive identity", [](FjPreparedHostData& value) {
                 value.spatial_dir.hash = 0;
             }},
            {"DIR absent operator", [](FjPreparedHostData& value) {
                 value.spatial_dir.components[0].reference_operator = FJ_DIR_OPERATOR_NONE;
             }},
            {"DIR operator enum", [](FjPreparedHostData& value) {
                 value.spatial_dir.components[0].reference_operator = 4;
             }},
            {"grain flags", [](FjPreparedHostData& value) {
                 value.grain.flags |= 64;
             }},
            {"grain debug enum", [](FjPreparedHostData& value) {
                 value.grain.debug_view = 7;
             }},
            {"grain count narrowing", [](FjPreparedHostData& value) {
                 value.grain.sublayer_count = static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1;
             }},
            {"grain inactive identity", [](FjPreparedHostData& value) {
                 value.grain.flags = 0;
             }},
            {"grain recipe identity", [](FjPreparedHostData& value) {
                 value.grain.recipe_hash = 0;
             }},
            {"grain frame identity", [](FjPreparedHostData& value) {
                 value.grain.hash = 0;
             }},
            {"effects flags", [](FjPreparedHostData& value) {
                 value.effects.flags |= 32;
             }},
            {"effects inactive identity", [](FjPreparedHostData& value) {
                 value.effects.flags = 0;
             }},
            {"effects recipe identity", [](FjPreparedHostData& value) {
                 value.effects.recipe_hash = 0;
             }},
            {"effects frame identity", [](FjPreparedHostData& value) {
                 value.effects.hash = 0;
             }}};
        for (const auto& item : cases) {
            SCOPED_TRACE(item.name);
            FjPreparedHostData input = make_fixed_input();
            JuicerCuda::PreparedDescriptors actual;
            std::string diagnostic;
            ASSERT_TRUE(JuicerCuda::decode_prepared_descriptors(input, actual, diagnostic)) << diagnostic;
            item.mutate(input);
            EXPECT_FALSE(JuicerCuda::decode_prepared_descriptors(input, actual, diagnostic));
            EXPECT_FALSE(diagnostic.empty());
            EXPECT_FALSE(actual.diffusion);
            EXPECT_FALSE(actual.halation);
            EXPECT_FALSE(actual.grain);
            EXPECT_FALSE(actual.effects);
            EXPECT_FALSE(actual.grainRecipe.active);
            EXPECT_EQ(actual.spatialDir.hash, 0u);
            EXPECT_EQ(actual.scanner.hash, 0u);
        }
    }

} // namespace
