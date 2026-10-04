#include "CudaRenderProjection.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>
#include <memory>
#include <span>
#include <string_view>
#include <utility>

#include "juicer_cuda_owner.h"
#include "Cuda/JuicerCudaHostViews.h"
#include "FocusedRenderPayload.h"
#include "ResourceAssetLibrary.h"
#include "SpectralProcessing.h"

namespace {

    template <typename Extent>
    FjExtent encode_extent(const Extent& source) {
        return {source.x, source.y, source.width, source.height};
    }

    FjDust encode_dust(const DefectDustRecipe& source) {
        FjDust out{};
        out.cell_width_mm = source.cellWidthMm;
        out.cell_height_mm = source.cellHeightMm;
        out.slot_probability = source.slotProbability;
        out.softness_min_mm = source.softnessMinMm;
        out.softness_max_mm = source.softnessMaxMm;
        out.softness_size_cap_fraction = source.softnessSizeCapFraction;
        out.support_x_mm = source.supportXMm;
        out.support_y_mm = source.supportYMm;
        out.fiber_fraction = source.fiberFraction;
        out.fiber_drift_fraction = source.fiberDriftFraction;
        out.fiber_first_knot_min = source.fiberFirstKnotMin;
        out.fiber_first_knot_max = source.fiberFirstKnotMax;
        out.fiber_second_knot_min = source.fiberSecondKnotMin;
        out.fiber_second_knot_max = source.fiberSecondKnotMax;
        out.fiber_interior_width_min_fraction = source.fiberInteriorWidthMinFraction;
        out.fiber_interior_width_max_fraction = source.fiberInteriorWidthMaxFraction;
        out.diameter_min_mm = source.diameterMinMm;
        out.diameter_bulk_max_mm = source.diameterBulkMaxMm;
        out.diameter_max_mm = source.diameterMaxMm;
        out.diameter_tail_fraction = source.diameterTailFraction;
        out.fiber_length_min_mm = source.fiberLengthMinMm;
        out.fiber_length_max_mm = source.fiberLengthMaxMm;
        out.fiber_width_min_mm = source.fiberWidthMinMm;
        out.fiber_width_max_mm = source.fiberWidthMaxMm;
        out.opacity_faint_cumulative = source.opacityFaintCumulative;
        out.opacity_intermediate_cumulative = source.opacityIntermediateCumulative;
        out.compact_opacity_min = source.compactOpacityMin;
        out.compact_opacity_faint_end = source.compactOpacityFaintEnd;
        out.compact_opacity_intermediate_end = source.compactOpacityIntermediateEnd;
        out.compact_opacity_max = source.compactOpacityMax;
        out.fiber_opacity_min = source.fiberOpacityMin;
        out.fiber_opacity_faint_end = source.fiberOpacityFaintEnd;
        out.fiber_opacity_intermediate_end = source.fiberOpacityIntermediateEnd;
        out.fiber_opacity_max = source.fiberOpacityMax;
        out.compact_dominant_aspect_min = source.compactDominantAspectMin;
        out.compact_dominant_aspect_max = source.compactDominantAspectMax;
        out.compact_subsidiary_scale_min = source.compactSubsidiaryScaleMin;
        out.compact_subsidiary_scale_max = source.compactSubsidiaryScaleMax;
        out.compact_subsidiary_aspect_min = source.compactSubsidiaryAspectMin;
        out.compact_subsidiary_aspect_max = source.compactSubsidiaryAspectMax;
        out.compact_subsidiary_offset_max = source.compactSubsidiaryOffsetMax;
        out.compact_subsidiary_angle_max_radians = source.compactSubsidiaryAngleMaxRadians;
        return out;
    }

    FjScratch encode_scratch(const DefectScratchRecipe& source) {
        FjScratch out{};
        out.cell_width_mm = source.cellWidthMm;
        out.cell_height_mm = source.cellHeightMm;
        out.slot_probability = source.slotProbability;
        out.softness_min_mm = source.softnessMinMm;
        out.softness_max_mm = source.softnessMaxMm;
        out.softness_size_cap_fraction = source.softnessSizeCapFraction;
        out.support_x_mm = source.supportXMm;
        out.support_y_mm = source.supportYMm;
        out.length_min_mm = source.lengthMinMm;
        out.length_bulk_max_mm = source.lengthBulkMaxMm;
        out.length_max_mm = source.lengthMaxMm;
        out.length_tail_fraction = source.lengthTailFraction;
        out.width_min_mm = source.widthMinMm;
        out.width_bulk_max_mm = source.widthBulkMaxMm;
        out.width_max_mm = source.widthMaxMm;
        out.width_tail_fraction = source.widthTailFraction;
        out.drift_fraction = source.driftFraction;
        out.first_knot_min = source.firstKnotMin;
        out.first_knot_max = source.firstKnotMax;
        out.second_knot_min = source.secondKnotMin;
        out.second_knot_max = source.secondKnotMax;
        out.interior_width_min_fraction = source.interiorWidthMinFraction;
        out.interior_width_max_fraction = source.interiorWidthMaxFraction;
        out.interior_depth_min_fraction = source.interiorDepthMinFraction;
        out.interior_depth_max_fraction = source.interiorDepthMaxFraction;
        out.endpoint_abrupt_probability = source.endpointAbruptProbability;
        out.interruption_probability = source.interruptionProbability;
        out.gap_center_min = source.gapCenterMin;
        out.gap_center_max = source.gapCenterMax;
        out.gap_span_min = source.gapSpanMin;
        out.gap_span_max = source.gapSpanMax;
        out.scuff_probability = source.scuffProbability;
        out.scuff_length_max_mm = source.scuffLengthMaxMm;
        out.scuff_angle_max_radians = source.scuffAngleMaxRadians;
        out.strength_min = source.strengthMin;
        out.strength_max = source.strengthMax;
        return out;
    }

    FjHalationGaussian encode_halation_gaussian(const ScatterHalationGaussianDescriptor& source) {
        FjHalationGaussian out{};
        out.kind = static_cast<std::uint32_t>(source.kind);
        out.radius = source.radius;
        std::copy(source.firWeights.begin(), source.firWeights.end(), out.fir_weights);
        out.feedforward = source.B;
        out.feedback[0] = source.B1;
        out.feedback[1] = source.B2;
        out.feedback[2] = source.B3;
        return out;
    }

    FjDiffusion encode_diffusion(const Spektrafilm::DiffusionStageFrameDescriptor& source) {
        FjDiffusion out{};
        out.family = static_cast<std::uint32_t>(source.sample.family);
        out.scatter_fraction = source.scatterFraction;
        std::copy(source.sample.groupWeightsCoreHaloBloom.begin(), source.sample.groupWeightsCoreHaloBloom.end(), out.group_weights);
        std::copy(source.sample.groupCenterLambdaUm.begin(), source.sample.groupCenterLambdaUm.end(), out.group_lambda_um);
        out.warmth = source.sample.effectiveWarmth;
        out.spatial_scale = source.sample.spatialScale;
        out.pixel_size_um = source.sample.pixelSizeUm;
        out.radius_px = source.sample.radiusPixels;
        out.sample_hash = source.sample.hash;
        out.hash = source.hash;
        return out;
    }

    void encode_optics(const JuicerCuda::PreparedDescriptors& source, FjOptics& out) {
        out = {};
        if (source.diffusion) {
            const auto& descriptor = *source.diffusion;
            out.full_frame = {descriptor.fullFrame.originX, descriptor.fullFrame.originY, descriptor.fullFrame.width, descriptor.fullFrame.height};
            out.diffusion_hash = descriptor.hash;
            if (descriptor.camera) {
                out.flags |= FJ_OPTICS_CAMERA_DIFFUSION;
                out.camera = encode_diffusion(*descriptor.camera);
            }
            if (descriptor.enlarger) {
                out.flags |= FJ_OPTICS_ENLARGER_DIFFUSION;
                out.enlarger = encode_diffusion(*descriptor.enlarger);
            }
        }
        if (source.halation) {
            out.flags |= FJ_OPTICS_SCATTER_HALATION;
            out.scatter_halation.recipe_hash = source.halation->recipeHash;
            out.scatter_halation.scatter_amount = source.halation->scatterAmount;
            for (std::size_t channel = 0; channel < source.halation->channels.size(); ++channel) {
                const auto& input = source.halation->channels[channel];
                FjHalationChannel& output = out.scatter_halation.channels[channel];
                output.total_strength = input.totalStrength;
                output.core = encode_halation_gaussian(input.core);
                for (std::size_t component = 0; component < input.tail.size(); ++component) {
                    output.tail[component] = encode_halation_gaussian(input.tail[component]);
                    output.bounce[component] = encode_halation_gaussian(input.bounce[component]);
                }
            }
        }
    }

    void encode_spatial_dir(const Spektrafilm::SpatialDirDescriptor& source, FjSpatialDir& out) {
        out = {};
        if (source.hash == 0) {
            return;
        }
        out.render_extent = encode_extent(source.renderExtent);
        out.full_frame_extent = encode_extent(source.fullFrameExtent);
        out.filter_domain_extent = encode_extent(source.filterDomainExtent);
        out.component_count = static_cast<std::size_t>(source.filterPlan.componentCount);
        out.recipe_hash = source.dirRecipeHash;
        out.hash = source.hash;
        for (std::size_t index = 0; index < out.component_count; ++index) {
            const auto& input = source.filterPlan.components[index];
            FjDirGaussian& output = out.components[index];
            output.sigma_px = input.sigmaPixels;
            output.weight = input.weight;
            output.radius = input.radius;
            output.reference_operator = static_cast<std::uint32_t>(input.referenceOperator);
            output.feedforward = input.iir.feedforward;
            std::copy(input.iir.feedback.begin(), input.iir.feedback.end(), output.feedback);
            output.normalization_denominator = input.iir.normalizationDenominator;
            output.feedforward_numerator = input.iir.feedforwardNumerator;
            std::copy(input.iir.feedbackNumerators.begin(), input.iir.feedbackNumerators.end(), output.feedback_numerators);
            output.boundary_truncation_accuracy = input.boundaryTruncationAccuracy;
            output.boundary_certification_tolerance = input.boundaryCertificationTolerance;
            output.boundary_derivation_version = input.boundaryDerivationVersion;
        }
    }

    FjGrainGaussian encode_grain_gaussian(const Spektrafilm::VisualGrainGaussian& source) {
        return {source.sigmaPx, source.radius, source.hash};
    }

    void encode_grain(const JuicerCuda::PreparedDescriptors& source, FjVisualGrain& out) {
        out = {};
        if (!source.grain) {
            return;
        }
        const auto& descriptor = *source.grain;
        const auto& recipe = source.grainRecipe;
        out.flags = FJ_GRAIN_ACTIVE;
        if (recipe.sublayersActive) {
            out.flags |= FJ_GRAIN_SUBLAYERS;
        }
        if (descriptor.requiresFullFrame) {
            out.flags |= FJ_GRAIN_FULL_FRAME;
        }
        constexpr std::uint32_t kAxisFlags[] = {
            FJ_GRAIN_AXIS_FINITE_C, FJ_GRAIN_AXIS_FINITE_M, FJ_GRAIN_AXIS_FINITE_Y};
        for (std::size_t channel = 0; channel < recipe.grainLayerAxisFinite.size(); ++channel) {
            if (recipe.grainLayerAxisFinite[channel]) {
                out.flags |= kAxisFlags[channel];
            }
            std::copy(recipe.grainLayerAxisBlockPrefixMax[channel].begin(),
                      recipe.grainLayerAxisBlockPrefixMax[channel].end(),
                      out.layer_axis_block_prefix_max[channel]);
        }
        out.debug_view = static_cast<std::uint32_t>(recipe.debugView);
        out.render_extent = encode_extent(descriptor.renderExtent);
        out.full_frame_extent = encode_extent(descriptor.fullFrameExtent);
        out.pixel_size_um = descriptor.pixelSizeUm;
        out.frame0 = descriptor.frame0;
        out.frame_alpha = descriptor.frameAlpha;
        out.seed_base = descriptor.seedBase;
        out.seed_base_next = descriptor.seedBaseNext;
        out.session_seed = descriptor.sessionSeed;
        out.clip_token = descriptor.clipToken;
        out.pitch_px = descriptor.pitchPx;
        out.breathing_period_frames = descriptor.breathingPeriodFrames;
        out.clump_morph_period_frames = descriptor.clumpMorphPeriodFrames;
        out.wang_cell_mm = descriptor.wangCellMm;
        out.breathing_amplitude = descriptor.breathingAmplitude;
        out.breathing_cell_um_small = descriptor.breathingCellUmSmall;
        out.breathing_cell_um_large = descriptor.breathingCellUmLarge;
        out.breathing_mix = descriptor.breathingMix;
        out.breathing_drift_um_per_frame = descriptor.breathingDriftUmPerFrame;
        out.debug_scale = descriptor.debugScale;
        out.size_mix_weights[0] = descriptor.effectiveFineWeight;
        out.size_mix_weights[1] = descriptor.effectiveMidWeight;
        out.size_mix_weights[2] = descriptor.effectiveCoarseWeight;
        out.size_mix_gain = descriptor.sizeMixGain;
        out.sublayer_count = static_cast<std::size_t>(recipe.nSubLayers);
        std::copy(recipe.microStructure.begin(), recipe.microStructure.end(), out.micro_structure);
        out.clump_temporal_mix = recipe.clumpTemporalMix;
        out.size_mix_scale = recipe.sizeMixScale;
        out.amplitude = recipe.amplitude;
        out.chroma_mix = recipe.chromaMix;
        out.chroma_shared_weight = recipe.chromaSharedWeight;
        out.chroma_independent_weight = recipe.chromaIndependentWeight;
        std::copy(recipe.visualParticleDensityMinCmy.begin(), recipe.visualParticleDensityMinCmy.end(), out.particle_density_min_cmy);
        std::copy(recipe.uniformityCmy.begin(), recipe.uniformityCmy.end(), out.uniformity_cmy);
        std::copy(descriptor.densityMaxCmy.begin(), descriptor.densityMaxCmy.end(), out.density_max_cmy);
        std::copy(descriptor.nParticlesCmy.begin(), descriptor.nParticlesCmy.end(), out.n_particles_cmy);
        std::copy(descriptor.odParticleCmy.begin(), descriptor.odParticleCmy.end(), out.od_particle_cmy);
        for (std::size_t layer = 0; layer < descriptor.dyeCloud.size(); ++layer) {
            out.correlation[layer] = encode_grain_gaussian(descriptor.correlation[layer]);
            std::copy(descriptor.densityMinLayers[layer].begin(), descriptor.densityMinLayers[layer].end(), out.density_min_layers[layer]);
            std::copy(descriptor.densityMaxLayers[layer].begin(), descriptor.densityMaxLayers[layer].end(), out.density_max_layers[layer]);
            std::copy(descriptor.nParticlesLayers[layer].begin(), descriptor.nParticlesLayers[layer].end(), out.n_particles_layers[layer]);
            std::copy(descriptor.odParticleLayers[layer].begin(), descriptor.odParticleLayers[layer].end(), out.od_particle_layers[layer]);
            for (std::size_t channel = 0; channel < descriptor.dyeCloud[layer].size(); ++channel) {
                out.dye_cloud[layer][channel] = encode_grain_gaussian(descriptor.dyeCloud[layer][channel]);
            }
        }
        out.recipe_hash = descriptor.recipeHash;
        out.density_layers_hash = descriptor.densityCurvesLayersHash;
        out.hash = descriptor.hash;
    }

    void encode_effects(const JuicerCuda::PreparedDescriptors& source, FjFilmEffects& out) {
        out = {};
        if (!source.effects) {
            return;
        }
        const auto& descriptor = *source.effects;
        out.render_extent = encode_extent(descriptor.renderExtent);
        out.full_frame_extent = encode_extent(descriptor.fullFrameExtent);
        out.film_dust = encode_dust(descriptor.filmDust);
        out.film_scratch = encode_scratch(descriptor.filmScratch);
        out.gate_dust = encode_dust(descriptor.gateDust);
        out.gate_scratch = encode_scratch(descriptor.gateScratch);
        for (std::size_t index = 0; index < descriptor.origins.size(); ++index) {
            const auto& origin = descriptor.origins[index];
            out.origins[index] = {origin.cellX, origin.cellY, origin.localXMm, origin.localYMm};
        }
        out.sample_step_x_mm = descriptor.sampleStepXMm;
        out.sample_step_y_mm = descriptor.sampleStepYMm;
        out.roi_offset_x = descriptor.roiOffsetX;
        out.roi_offset_y = descriptor.roiOffsetY;
        out.gate_width = descriptor.gateWidth;
        out.gate_height = descriptor.gateHeight;
        out.session_seed = descriptor.sessionSeed;
        out.clip_token = descriptor.clipToken;
        if (descriptor.weaveActive) {
            out.flags |= FJ_EFFECTS_WEAVE;
        }
        if (descriptor.filmActive) {
            out.flags |= FJ_EFFECTS_FILM;
        }
        if (descriptor.gateTransmittanceActive) {
            out.flags |= FJ_EFFECTS_GATE_TRANSMITTANCE;
        }
        if (descriptor.gateOutputActive) {
            out.flags |= FJ_EFFECTS_GATE_OUTPUT;
        }
        if (descriptor.requiresFullFrame) {
            out.flags |= FJ_EFFECTS_FULL_FRAME;
        }
        out.weave_dx_px = descriptor.weaveDxPx;
        out.weave_dy_px = descriptor.weaveDyPx;
        out.weave_cos_rot = descriptor.weaveCosRot;
        out.weave_sin_rot = descriptor.weaveSinRot;
        out.recipe_hash = descriptor.recipeHash;
        out.hash = descriptor.hash;
    }


    void encode_prepared_descriptors(
        const JuicerCuda::PreparedDescriptors& source,
        FjPreparedHostData& out) {
        out.route = static_cast<std::uint32_t>(source.route);
        out.capture_polarity = static_cast<std::uint32_t>(source.capturePolarity);
        out.scanner_lut = {};
        out.scanner_lut.density_bounds_hash = source.scanner.densityBoundsHash;
        out.scanner_lut.resolution = source.scanner.lutResolution;
        out.scanner_lut.hash = source.scanner.hash;
        out.scanner_correction = {};
        out.scanner_correction.flags = source.correction.active ? FJ_CORRECTION_ACTIVE : 0;
        out.scanner_correction.xyz_slope = source.correction.xyzSlope;
        out.scanner_correction.xyz_offset = source.correction.xyzOffset;
        out.scanner_correction.exposure_scale = source.correction.exposureScale;
        out.scanner_post = {};
        out.scanner_post.flags = source.post.glareActive ? FJ_POST_GLARE : 0;
        out.scanner_post.glare_percent = source.post.glarePercent;
        out.scanner_post.glare_roughness = source.post.glareRoughness;
        out.scanner_post.glare_blur_sigma_px = source.post.glareBlurSigmaPx;
        out.scanner_post.glare_blur_radius = source.post.glareBlurRadius;
        out.scanner_post.lens_blur_sigma_px = source.post.lensBlurSigmaPx;
        out.scanner_post.lens_blur_radius = source.post.lensBlurRadius;
        out.scanner_post.unsharp_sigma_px = source.post.unsharpSigmaPx;
        out.scanner_post.unsharp_radius = source.post.unsharpRadius;
        out.scanner_post.unsharp_amount = source.post.unsharpAmount;
        out.scanner_post.hash = source.post.hash;
        FjOutputColor& color = out.output_color;
        std::copy_n(source.color.cat02, 9, color.cat02);
        std::copy_n(source.color.xyzToRgb, 9, color.xyz_to_rgb);
        std::copy_n(source.color.illuminantXYZ, 3, color.illuminant_xyz);
        color.color_space = static_cast<std::uint32_t>(source.color.encoding.colorSpace);
        color.flags = 0;
        if (source.color.encoding.applyCctfEncoding) {
            color.flags |= FJ_COLOR_ENCODE_CCTF;
        }
        if (source.color.encoding.inputIsOutputSpace) {
            color.flags |= FJ_COLOR_INPUT_IS_OUTPUT_SPACE;
        }
        if (source.outputGamut.enabled) {
            color.flags |= FJ_COLOR_GAMUT_COMPRESSION;
        }
        std::copy(source.outputGamut.transform.nativeRgbToD65Xyz.begin(),
                  source.outputGamut.transform.nativeRgbToD65Xyz.end(),
                  color.native_rgb_to_d65_xyz);
        std::copy(source.outputGamut.transform.d65XyzToNativeRgb.begin(),
                  source.outputGamut.transform.d65XyzToNativeRgb.end(),
                  color.d65_xyz_to_native_rgb);
        color.lightness_knee[0] = source.outputGamut.lightnessKneeThreshold;
        color.lightness_knee[1] = source.outputGamut.lightnessKneeLimit;
        color.lightness_knee[2] = source.outputGamut.lightnessKneePower;
        color.chroma_knee[0] = source.outputGamut.chromaKneeThreshold;
        color.chroma_knee[1] = source.outputGamut.chromaKneeLimit;
        color.chroma_knee[2] = source.outputGamut.chromaKneePower;
        color.gamut_recipe_hash = source.outputGamut.hash;
        encode_optics(source, out.optics);
        encode_spatial_dir(source.spatialDir, out.spatial_dir);
        encode_grain(source, out.grain);
        encode_effects(source, out.effects);
    }

    FjFloatSpan project(std::span<const float> source) {
        return source.empty() ? FjFloatSpan{} : FjFloatSpan{source.data(), source.size()};
    }

    FjFloatSpan project(const JuicerCuda::ThreeChannelSamplesView& source) {
        // Transport the contiguous scalar representation without indexing it as
        // a float array. The native decoder reads fixed triplets through bytes.
        const auto bytes = source.object_bytes();
        return bytes.empty() ? FjFloatSpan{} : FjFloatSpan{reinterpret_cast<const float*>(bytes.data()), source.scalar_count()};
    }

    FjByteSpan project(std::span<const std::uint8_t> source) {
        return source.empty() ? FjByteSpan{} : FjByteSpan{source.data(), source.size()};
    }

    FjStringView project(std::string_view source) {
        return source.empty() ? FjStringView{} : FjStringView{source.data(), source.size()};
    }

    void encode_bounds(const DensityBoundsRecipe& source, FjDensityBounds& out) {
        std::copy(source.dataMinCmy.begin(), source.dataMinCmy.end(), out.min_cmy);
        std::copy(source.invSpanCmy.begin(), source.invSpanCmy.end(), out.inv_span_cmy);
        out.hash = source.hash;
    }

    void encode_film(const RenderRecipe& recipe, const FocusedRenderPayload& payload, const JuicerCuda::FocusedRouteResourceInput& focused, FjPreparedHostData& out) {
        auto& exposure = out.film_exposure;
        exposure.input_color_space = static_cast<std::uint32_t>(recipe.filmRaw.inputColorSpace);
        exposure.method = static_cast<std::uint32_t>(recipe.filmRaw.rgbToRawMethod);
        exposure.flags = (recipe.filmRaw.inputCctfDecoding ? FJ_FILM_DECODE_CCTF : 0u) |
                         (payload.filmRawConfig.applyInputChromaticAdapt ? FJ_FILM_ADAPT_INPUT : 0u) |
                         (recipe.filmRaw.autoExposureEnabled ? FJ_FILM_AUTO_EXPOSURE : 0u);
        exposure.manual_exposure_ev = recipe.filmRaw.manualExposureCompensationEv;
        exposure.route_correction_scale = Spektrafilm::scan_route_is_print(recipe.profileRoute.scanRoute)
                                              ? 1.0f
                                              : out.scanner_correction.exposure_scale;
        exposure.mallett_green_midgray_scale = recipe.filmRaw.mallettGreenMidgrayScale;
        std::copy_n(payload.filmRawConfig.inputRGBToXYZ.m, 9, exposure.input_rgb_to_xyz);
        std::copy_n(payload.filmRawConfig.inputXYZAdapt.m, 9, exposure.input_xyz_adapt);
        std::copy_n(payload.filmRawConfig.xyzToLinearSrgb.m, 9, exposure.xyz_to_linear_srgb);
        exposure.sensitivity_rgb = project(focused.filmRaw.finalSensitivityRgb);
        if (recipe.filmRaw.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Mallett2019) {
            exposure.mallett_illuminant = project(focused.exposureIlluminant);
            exposure.mallett_basis = project(focused.mallettBasis);
        } else {
            exposure.tc_lut_rgba = project(focused.filmTcLut);
        }
        exposure.sensitivity_hash = recipe.filmRaw.finalSensitivityHash;
        exposure.tc_lut_hash = recipe.filmRaw.tcLutHash;
        exposure.hash = recipe.filmRaw.hash;
        auto& development = out.film_development;
        development.log_exposure = project(focused.filmDevelop.logExposure);
        development.density_rgb = project(focused.filmDevelop.normalizedDensityCurvesRgb);
        if (focused.wantDensityLayers) {
            for (std::size_t layer = 0; layer < 3; ++layer) {
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    development.density_layers[layer][channel] = project(focused.filmDevelop.densityCurvesLayers[layer][channel]);
                }
            }
            development.density_layers_hash = recipe.filmDevelop.densityCurvesLayersHash;
        }
        std::copy(recipe.filmDevelop.densityCurveGamma.begin(), recipe.filmDevelop.densityCurveGamma.end(), development.gamma_rgb);
        development.density_curves_hash = recipe.filmDevelop.normalizedDensityCurvesHash;
        development.hash = recipe.filmDevelop.hash;
        if (recipe.dirCouplers.active) {
            auto& dir = out.dir_couplers;
            dir.mode = static_cast<std::uint32_t>(recipe.dirCouplers.nonlinearMode);
            for (std::size_t channel = 0; channel < 3; ++channel) {
                std::copy(recipe.dirCouplers.matrixRgb[channel].begin(), recipe.dirCouplers.matrixRgb[channel].end(), dir.matrix_rgb + channel * 3u);
                dir.compensated_axes_rgb[channel] = project(focused.dirCouplers.compensatedDensityCurveAxesRgb[channel]);
            }
            std::copy(recipe.dirCouplers.densityMaxRgb.begin(), recipe.dirCouplers.densityMaxRgb.end(), dir.density_max_rgb);
            std::copy(recipe.dirCouplers.densityRefRgb.begin(), recipe.dirCouplers.densityRefRgb.end(), dir.density_ref_rgb);
            std::copy(recipe.dirCouplers.donorKRgb.begin(), recipe.dirCouplers.donorKRgb.end(), dir.donor_k_rgb);
            std::copy(recipe.dirCouplers.receiverCRefRgb.begin(), recipe.dirCouplers.receiverCRefRgb.end(), dir.receiver_c_ref_rgb);
            std::copy(recipe.dirCouplers.receiverKrRgb.begin(), recipe.dirCouplers.receiverKrRgb.end(), dir.receiver_kr_rgb);
            dir.compensated_axes_hash = recipe.dirCouplers.compensatedDensityCurveAxesHash;
            dir.hash = recipe.dirCouplers.hash;
        }
    }

    void encode_print(const JuicerCuda::PrintResourceInput& source, const PrintExposureRecipe& exposure, FjPrint& out) {
        out.film_density_cmy = project(source.filmChannelDensityCmy);
        out.film_base_density = project(source.filmBaseDensity);
        out.sensitivity_cmy = project(source.printSensitivityCmy);
        out.log_exposure = project(source.printLogExposure);
        out.density_cmy = project(source.printDensityCurvesCmy);
        out.main_illuminant = project(source.mainIlluminant);
        const auto& descriptors = source.descriptors;
        if (descriptors.preflashActive) {
            out.flags = FJ_PRINT_PREFLASH;
            out.preflash_illuminant = project(source.preflashIlluminant);
            std::copy(source.preflashRawCmy.begin(), source.preflashRawCmy.end(), out.preflash_raw_cmy);
        }
        out.main_cc_cmy[0] = descriptors.mainIlluminant.cmyCc.c;
        out.main_cc_cmy[1] = descriptors.mainIlluminant.cmyCc.m;
        out.main_cc_cmy[2] = descriptors.mainIlluminant.cmyCc.y;
        out.preflash_cc_cmy[0] = descriptors.preflashIlluminant.cmyCc.c;
        out.preflash_cc_cmy[1] = descriptors.preflashIlluminant.cmyCc.m;
        out.preflash_cc_cmy[2] = descriptors.preflashIlluminant.cmyCc.y;
        out.exposure = exposure.printExposure;
        out.preflash_exposure = exposure.preflashExposure;
        out.normalizer = source.normalizer;
        out.film_density_hash = descriptors.filmDensityTables.hash;
        out.profile_tables_hash = descriptors.profileTables.hash;
        out.main_illuminant_hash = descriptors.mainIlluminant.hash;
        out.preflash_illuminant_hash = descriptors.preflashIlluminant.hash;
        out.preflash_raw_hash = descriptors.preflashRaw.hash;
        out.balance_hash = descriptors.balance.hash;
        out.hash = descriptors.hash;
    }

} // namespace

FjRenderOutcome JuicerCuda::project_and_render(
    NativeCall& call, const RenderRecipe& recipe, const FocusedRenderPayload& payload, const ExecutionFrame& frame, const FjFrame& rawFrame, const ResourceManager::SubmissionSnapshot& snapshot, PendingContextLossRecovery& recovery, FjAbortCallback abortCallback, const StaticNoiseInput* noise, const std::exception_ptr& noiseFailure, std::string& diagnostic) {
    try {
        diagnostic.clear();
        JuicerCuda::PreparedDescriptors preparedDescriptors = JuicerCuda::describe_execution(recipe, payload, frame);
        JuicerCuda::FocusedRouteResourceInput focused;
        const JuicerCuda::FocusedRouteResourcePreparation request{
            &recipe, &payload.exposureTables, &payload.filmRawConfig, payload.filmTcLut ? &*payload.filmTcLut : nullptr, &payload.scannerTables, &payload.scannerColor, &preparedDescriptors.scanner, payload.outputBoundaryTable.get()};
        if (!JuicerCuda::build_focused_route_resource_input(request, focused, diagnostic)) {
            return {{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, 0};
        }
        const bool print = Spektrafilm::scan_route_is_print(recipe.profileRoute.scanRoute);
        std::array<float, 81> preflashIlluminant{};
        JuicerCuda::PrintResourceInput printInput;
        if (print) {
            if (!JuicerCuda::build_print_resource_input(
                    {&recipe, &JuicerProcess::root().assets(), payload.printMainIlluminant ? &*payload.printMainIlluminant : nullptr},
                    printInput,
                    preflashIlluminant,
                    diagnostic)) {
                return {{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, 0};
            }
            const Scanner::PrintCorrectionDerivationInput correction{
                &recipe, &payload.scannerTables, printInput.mainIlluminant.data(), 81, printInput.preflashRawCmy.data(), printInput.normalizer};
            if (!Scanner::build_print_scanner_color_correction_descriptor(correction, preparedDescriptors.correction, diagnostic)) {
                return {{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, 0};
            }
        }
        if (recipe.visualGrain.active) {
#if defined(JUICER_NOISE_TEST_HOOK)
            JuicerAssets::NoiseTest::projection_step();
#endif
            if (noiseFailure) {
                std::rethrow_exception(noiseFailure);
            }
            if (!noise) {
                diagnostic = "MissingRequiredResource phase=grain_static field=noise";
                return {{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, 0};
            }
        }
        FjPreparedHostData prepared{};
        encode_prepared_descriptors(preparedDescriptors, prepared);
        prepared.film_profile_key = project(recipe.profileRoute.filmProfileKey);
        prepared.print_profile_key = print ? project(recipe.profileRoute.printProfileKey) : FjStringView{};
        prepared.film_profile_asset_version = recipe.profileRoute.filmProfileAssetVersionToken;
        prepared.print_profile_asset_version = print ? recipe.profileRoute.printProfileAssetVersionToken : 0;
        prepared.recipe_hash = recipe.hash;
        encode_film(recipe, payload, focused, prepared);
        encode_bounds(recipe.densityBounds, prepared.scanner_bounds);
        if (print) {
            encode_bounds(recipe.enlargerFilmBounds, prepared.enlarger_film_bounds);
            encode_print(printInput, recipe.print.exposure, prepared.print);
        }
        const auto& scanner = focused.scannerTables;
        prepared.scanner_spectra = {project(scanner.epsY), project(scanner.epsM), project(scanner.epsC), project(scanner.Ax), project(scanner.Ay), project(scanner.Az), scanner.hasBaseline ? project(scanner.baseDensityMin) : FjFloatSpan{}, scanner.invYn};
        if (focused.outputGamut.enabled) {
            prepared.output_color.gamut_cmax = project(focused.outputGamut.cmax);
            prepared.output_color.gamut_table_hash = focused.outputGamut.tableHash;
        }
        if (recipe.visualGrain.active) {
            prepared.noise = {project(noise->stbn), noise->stbnWidth, noise->stbnHeight, noise->stbnFrames, project(noise->wangTiles), project(noise->wangLut), noise->wangWidth, noise->wangHeight, static_cast<std::size_t>(noise->wangCount), noise->wangColors};
        }
        const auto& meter = frame.autoExposureDescriptor;
        prepared.auto_exposure = {{meter.sourceX1, meter.sourceY1, meter.sourceX2, meter.sourceY2},
                                  {meter.meterX1, meter.meterY1, meter.meterX2, meter.meterY2},
                                  meter.previewWidth,
                                  meter.previewHeight,
                                  static_cast<std::uint32_t>(meter.method),
                                  meter.hash};
        const FjCudaContext context{snapshot.deviceContextKey.deviceId, reinterpret_cast<std::uintptr_t>(snapshot.deviceContextKey.contextOpaque)};
        const FjSubmission submission{snapshot.instanceToken.value, snapshot.frameToken.value, snapshot.snapshotId, snapshot.keyDigests.uploadCoreHash, snapshot.keyDigests.dirHash, snapshot.keyDigests.scannerHash, snapshot.keyDigests.autoExposureHash};
        return call.render(&context, &rawFrame, &submission, &prepared, abortCallback, recovery, diagnostic);
    } catch (ExecutionFailure& failure) {
        diagnostic = std::move(failure.failure.diagnostic);
        return {failure.failure.status, failure.deferredDirError ? FJ_RENDER_DEFERRED_DIR_ERROR : 0U};
    } catch (const std::bad_alloc&) {
        diagnostic = "CUDA render allocation failed";
        return {{FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0}, 0};
    } catch (const std::exception& detail) {
        diagnostic = detail.what();
        return {{FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, 0};
    } catch (...) {
        diagnostic = "CUDA render projection failed with unknown exception";
        return {{FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, 0};
    }
}
