#include "prepared_state_native.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "ColorTransforms.h"
#include "Cuda/JuicerCudaHostViews.h"
#include "Cuda/JuicerCudaResources.h"
#include "FocusedRenderPayload.h"
#include "Illuminants.h"
#include "JuicerState.h"
#include "ProcessRoot.h"
#include "RenderRecipe.h"
#include "ResourceAssetLibrary.h"
#include "Scanner.h"
#include "SpectralData.h"

namespace {

    template <std::size_t N>
    FjFloatSpan float_span(const std::array<float, N>& values) {
        return {values.data(), values.size()};
    }

    FjFloatSpan float_span(const std::vector<float>& values) {
        return {values.data(), values.size()};
    }

    template <class Rows>
    FjFloatSpan triplet_span(const Rows& rows) {
        using Row = typename Rows::value_type;
        static_assert(sizeof(Row) == 3 * sizeof(float));
        return {rows.empty() ? nullptr : reinterpret_cast<const float*>(rows.data()), rows.size() * 3};
    }

    void copy3(const std::array<float, 3>& source, float (&destination)[3]) {
        std::copy(source.begin(), source.end(), destination);
    }

    FjFilmDensityBounds density_bounds(const DensityBoundsRecipe& source) {
        FjFilmDensityBounds out{};
        out.route = static_cast<std::uint32_t>(source.route);
        out.medium = static_cast<std::uint32_t>(source.medium);
        out.polarity = static_cast<std::uint32_t>(source.polarity);
        out.source = static_cast<std::uint32_t>(source.source);
        copy3(source.dataMinCmy, out.minimum_cmy);
        copy3(source.dataMaxCmy, out.maximum_cmy);
        copy3(source.invSpanCmy, out.inv_span_cmy);
        copy3(source.authoredMinCmy, out.authored_minimum_cmy);
        copy3(source.authoredMaxCmy, out.authored_maximum_cmy);
        out.hash = source.hash;
        return out;
    }

    FjDiffusionRecipe diffusion(const DiffusionFilterOpticsRecipe& source) {
        FjDiffusionRecipe out{};
        out.domain = static_cast<std::uint32_t>(source.domain);
        out.active = source.resolved.active ? 1u : 0u;
        out.family = static_cast<std::uint32_t>(source.resolved.family);
        out.scatter_fraction = source.resolved.scatterFraction;
        std::copy(source.resolved.groupWeightsCoreHaloBloom.begin(), source.resolved.groupWeightsCoreHaloBloom.end(), out.group_weights);
        std::copy(source.resolved.groupCenterLambdaUm.begin(), source.resolved.groupCenterLambdaUm.end(), out.group_lambda_um);
        out.warmth = source.resolved.effectiveWarmth;
        out.spatial_scale = source.resolved.spatialScale;
        out.resolved_hash = source.resolved.hash;
        out.hash = source.hash;
        return out;
    }

    FjSpatialOpticsRecipe spatial_optics(const SpatialOptics& source) {
        FjSpatialOpticsRecipe out{};
        out.camera_diffusion = diffusion(source.cameraDiffusion);
        out.camera_lens_blur.sigma_um = source.cameraLensBlur.sigmaUm;
        out.camera_lens_blur.hash = source.cameraLensBlur.hash;
        const auto& halation = source.scatterHalation;
        out.scatter_halation.scatter_active = halation.scatterActive ? 1u : 0u;
        out.scatter_halation.back_reflection_active = halation.backReflectionActive ? 1u : 0u;
        out.scatter_halation.scatter_amount = halation.scatterAmount;
        out.scatter_halation.scatter_spatial_scale = halation.scatterSpatialScale;
        out.scatter_halation.halation_spatial_scale = halation.halationSpatialScale;
        std::copy(halation.halationFirstSigmaUm.begin(), halation.halationFirstSigmaUm.end(), out.scatter_halation.first_sigma_um);
        std::copy(halation.totalStrength.begin(), halation.totalStrength.end(), out.scatter_halation.total_strength);
        out.scatter_halation.hash = halation.hash;
        out.enlarger_diffusion = diffusion(source.enlargerDiffusion);
        out.hash = source.hash;
        return out;
    }

    FjDust dust(const DefectDustRecipe& source) {
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

    FjScratch scratch(const DefectScratchRecipe& source) {
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

    FjPreparedSpectralView spectral_view(const Spectral::SpectralTables& source) {
        FjPreparedSpectralView out{};
        out.lambda_nm = float_span(source.lambda);
        out.illuminant = float_span(source.illum);
        out.observer_xyz[0] = float_span(source.Xbar);
        out.observer_xyz[1] = float_span(source.Ybar);
        out.observer_xyz[2] = float_span(source.Zbar);
        out.weighted_xyz[0] = float_span(source.Ax);
        out.weighted_xyz[1] = float_span(source.Ay);
        out.weighted_xyz[2] = float_span(source.Az);
        out.dyes_cmy[0] = float_span(source.epsC);
        out.dyes_cmy[1] = float_span(source.epsM);
        out.dyes_cmy[2] = float_span(source.epsY);
        out.baseline_min = float_span(source.baseDensityMin);
        out.baseline_mid = float_span(source.baseDensityMid);
        out.inv_yn = source.invYn;
        copy3({source.whiteXYZ[0], source.whiteXYZ[1], source.whiteXYZ[2]}, out.white_xyz);
        out.has_baseline = source.hasBaseline ? 1u : 0u;
        out.illuminant_hash = source.illuminantHash;
        out.hash = source.tablesHash;
        return out;
    }

    FjScannerColorCorrection scanner_correction(const Scanner::ScannerColorCorrectionDescriptor& source) {
        FjScannerColorCorrection out{};
        out.route = static_cast<std::uint32_t>(source.route);
        out.active = source.active ? 1u : 0u;
        out.black_correction = source.blackCorrection ? 1u : 0u;
        out.white_correction = source.whiteCorrection ? 1u : 0u;
        out.target_black_linear = source.targetBlackLinear;
        out.target_white_linear = source.targetWhiteLinear;
        out.reference_black_y = source.referenceBlackY;
        out.reference_white_y = source.referenceWhiteY;
        out.xyz_slope = source.xyzSlope;
        out.xyz_offset = source.xyzOffset;
        out.exposure_scale = source.exposureScale;
        out.schema_version = source.schemaVersion;
        out.hash = source.hash;
        return out;
    }

    FjScannerPostEffectsDescriptor scanner_post(const Scanner::ScannerPostEffectsDescriptor& source) {
        FjScannerPostEffectsDescriptor out{};
        out.route = static_cast<std::uint32_t>(source.route);
        out.glare_active = source.glareActive ? 1u : 0u;
        out.glare_percent = source.glarePercent;
        out.glare_roughness = source.glareRoughness;
        out.glare_blur_sigma_px = source.glareBlurSigmaPx;
        out.glare_blur_radius = source.glareBlurRadius;
        out.lens_blur_sigma_px = source.lensBlurSigmaPx;
        out.lens_blur_radius = source.lensBlurRadius;
        out.unsharp_sigma_px = source.unsharpSigmaPx;
        out.unsharp_radius = source.unsharpRadius;
        out.unsharp_amount = source.unsharpAmount;
        out.schema = source.schemaVersion;
        out.hash = source.hash;
        return out;
    }

    FjScannerLutDescriptor scanner_lut(const Scanner::ScannerSpectralLutDescriptor& source) {
        FjScannerLutDescriptor out{};
        out.route = static_cast<std::uint32_t>(source.route);
        out.medium = static_cast<std::uint32_t>(source.medium);
        out.capture_polarity = static_cast<std::uint32_t>(source.polarity);
        out.resolution = source.lutResolution;
        out.density_bounds_hash = source.densityBoundsHash;
        out.channel_density_hash = source.channelDensityHash;
        out.base_density_hash = source.baseDensityHash;
        out.viewing_illuminant_hash = source.scanIlluminantHash;
        out.observer_hash = source.observerHash;
        out.hash = source.hash;
        return out;
    }

    FjScannerColor scanner_color(const Scanner::ColorRuntime& source) {
        FjScannerColor out{};
        std::copy_n(source.cat02, 9, out.cat02);
        std::copy_n(source.xyzToRgb, 9, out.xyz_to_rgb);
        std::copy_n(source.illuminantXYZ, 3, out.illuminant_xyz);
        out.output_space = static_cast<std::uint32_t>(source.encoding.colorSpace);
        out.flags = (source.encoding.applyCctfEncoding ? FJ_COLOR_ENCODE_CCTF : 0u) |
                    (source.encoding.inputIsOutputSpace ? FJ_COLOR_INPUT_IS_OUTPUT_SPACE : 0u);
        out.encoding = source.preparedEncoding;
        out.active_gamut_recipe_hash = source.outputGamutRecipeHash;
        out.hash = source.hash;
        return out;
    }

    FjOutputColor output_color(const Scanner::ColorRuntime& color, const OutputGamutRecipe& gamut, const Gamut::OutputBoundaryHold* boundary) {
        FjOutputColor out{};
        std::copy_n(color.cat02, 9, out.cat02);
        std::copy_n(color.xyzToRgb, 9, out.xyz_to_rgb);
        std::copy_n(color.illuminantXYZ, 3, out.illuminant_xyz);
        out.color_space = static_cast<std::uint32_t>(color.encoding.colorSpace);
        out.flags = (color.encoding.applyCctfEncoding ? FJ_COLOR_ENCODE_CCTF : 0u) |
                    (color.encoding.inputIsOutputSpace ? FJ_COLOR_INPUT_IS_OUTPUT_SPACE : 0u) |
                    (gamut.enabled ? FJ_COLOR_GAMUT_COMPRESSION : 0u);
        std::copy(gamut.transform.nativeRgbToD65Xyz.begin(), gamut.transform.nativeRgbToD65Xyz.end(), out.native_rgb_to_d65_xyz);
        std::copy(gamut.transform.d65XyzToNativeRgb.begin(), gamut.transform.d65XyzToNativeRgb.end(), out.d65_xyz_to_native_rgb);
        out.lightness_knee[0] = gamut.lightnessKneeThreshold;
        out.lightness_knee[1] = gamut.lightnessKneeLimit;
        out.lightness_knee[2] = gamut.lightnessKneePower;
        out.chroma_knee[0] = gamut.chromaKneeThreshold;
        out.chroma_knee[1] = gamut.chromaKneeLimit;
        out.chroma_knee[2] = gamut.chromaKneePower;
        if (boundary) {
            out.gamut_cmax = {boundary->cmax.data(), boundary->cmax.size()};
            out.gamut_table_hash = boundary->hash;
        }
        out.gamut_recipe_hash = gamut.hash;
        out.encoding = color.preparedEncoding;
        return out;
    }

    FjVisualGrainRecipe visual_grain(const VisualGrainRecipe& source) {
        FjVisualGrainRecipe out{};
        out.active = source.active ? 1u : 0u;
        out.sublayers_active = source.sublayersActive ? 1u : 0u;
        for (std::size_t channel = 0; channel < 3; ++channel) {
            out.axis_finite_cmy[channel] = source.grainLayerAxisFinite[channel] ? 1u : 0u;
            std::copy(source.grainLayerAxisBlockPrefixMax[channel].begin(), source.grainLayerAxisBlockPrefixMax[channel].end(), out.axis_block_prefix_max_cmy[channel]);
        }
        out.particle_area_um2 = source.particleAreaUm2;
        copy3(source.particleScaleCmy, out.particle_scale_cmy);
        copy3(source.particleScaleLayers, out.particle_scale_layers);
        copy3(source.visualParticleDensityMinCmy, out.particle_density_min_cmy);
        copy3(source.uniformityCmy, out.uniformity_cmy);
        std::copy(source.microStructure.begin(), source.microStructure.end(), out.micro_structure);
        out.correlation_sigma_px = source.correlationSigmaPx;
        out.dye_cloud_blur_um = source.dyeCloudBlurUm;
        out.sublayer_count = source.nSubLayers;
        out.amplitude = source.amplitude;
        out.chroma_mix = source.chromaMix;
        out.chroma_shared_weight = source.chromaSharedWeight;
        out.chroma_independent_weight = source.chromaIndependentWeight;
        out.fine_weight = source.fineWeight;
        out.mid_weight = source.midWeight;
        out.coarse_weight = source.coarseWeight;
        out.size_mix_scale = source.sizeMixScale;
        out.clump_temporal_mix = source.clumpTemporalMix;
        out.clump_morph_period_sec = source.clumpMorphPeriodSec;
        out.debug_view = source.debugView;
        out.layers_hash = source.densityCurvesLayersHash;
        out.hash = source.hash;
        return out;
    }

} // namespace

bool PreparedStateOracle::SF_TEMP_BRIDGE_native_prepared_view(
    const FocusedRenderStateBuildProduct& product,
    const ParamSnapshot& params,
    FjPreparedStateView& out,
    PreparedStateOracle::NativeBacking& backing,
    std::string& diagnostic) {
    diagnostic.clear();
    out = {};
    backing.preflash = {};
    backing.print_source_illuminant = {};

    const auto& recipe = product.recipe;
    const auto& payload = product.payload;
    const bool print = Spektrafilm::scan_route_is_print(recipe.profileRoute.scanRoute);

    out.film_profile_key = {recipe.profileRoute.filmProfileKey.data(), recipe.profileRoute.filmProfileKey.size()};
    out.print_profile_key = {recipe.profileRoute.printProfileKey.data(), recipe.profileRoute.printProfileKey.size()};
    out.film_profile_asset_version = recipe.profileRoute.filmProfileAssetVersionToken;
    out.print_profile_asset_version = recipe.profileRoute.printProfileAssetVersionToken;
    out.profile_route_hash = recipe.profileRoute.hash;
    out.recipe_hash = recipe.hash;
    out.upload_core_hash = payload.uploadCoreHash;
    out.scanner_hash = payload.scannerHash;
    out.route = static_cast<std::uint32_t>(recipe.profileRoute.scanRoute);
    out.polarity = static_cast<std::uint32_t>(recipe.profileRoute.capturePolarity);

    const auto& input = recipe.filmRaw;
    auto& input_policy = out.input_policy;
    input_policy.meter = static_cast<std::uint32_t>(input.autoExposureMethod);
    input_policy.band_pass_active = input.cameraBandPass.active ? 1u : 0u;
    input_policy.apply_hanatos_window = input.hanatos.applyWindow ? 1u : 0u;
    input_policy.apply_hanatos_surface = input.hanatos.applySurface ? 1u : 0u;
    input_policy.projection_illuminant = input.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Hanatos2025
                                             ? FJ_FILM_INPUT_ILLUMINANT_PROFILE_REFERENCE
                                             : FJ_FILM_INPUT_ILLUMINANT_D65;
    input_policy.input_compression_active = input.inputCompressionActive ? 1u : 0u;
    input_policy.film_format_long_edge_mm = input.filmFormatLongEdgeMm;
    copy3(input.cameraBandPass.uv, input_policy.uv);
    copy3(input.cameraBandPass.ir, input_policy.ir);
    input_policy.hanatos_spectral_blur = input.hanatos.spectralGaussianBlur;
    copy3(input.inputNominalWhiteXYZ, input_policy.nominal_white_xyz);
    copy3(input.projectionWhiteXYZ, input_policy.projection_white_xyz);
    input_policy.hanatos_window = {input.hanatos.windowParams.data(), input.hanatos.windowParams.size()};
    input_policy.hanatos_surface_rgb = {input.hanatos.surfaceParams[0].data(), 45};
    input_policy.input_hull_hash = input.inputCompressionHullHash;
    input_policy.tc_source_hash = input.tcSourceAssetHash;

    auto& scanner_policy = out.scanner_policy;
    const auto& scan = recipe.scannerOutput;
    scanner_policy.black_correction = scan.blackCorrection ? 1u : 0u;
    scanner_policy.white_correction = scan.whiteCorrection ? 1u : 0u;
    scanner_policy.glare_active = scan.glareActive ? 1u : 0u;
    scanner_policy.black_level = scan.blackLevel;
    scanner_policy.white_level = scan.whiteLevel;
    scanner_policy.glare_percent = scan.glarePercent;
    scanner_policy.glare_roughness = scan.glareRoughness;
    scanner_policy.glare_blur_sigma_px = scan.glareBlurSigmaPx;
    scanner_policy.lens_blur_sigma_px = scan.lensBlurSigmaPx;
    scanner_policy.unsharp_sigma_px = scan.unsharpSigmaPx;
    scanner_policy.unsharp_amount = scan.unsharpAmount;

    auto& exposure = out.film_exposure;
    exposure.input_color_space = static_cast<std::uint32_t>(input.inputColorSpace);
    exposure.method = static_cast<std::uint32_t>(input.rgbToRawMethod);
    exposure.flags = (input.inputCctfDecoding ? FJ_FILM_DECODE_CCTF : 0u) |
                     (payload.filmRawConfig.applyInputChromaticAdapt ? FJ_FILM_ADAPT_INPUT : 0u) |
                     (input.autoExposureEnabled ? FJ_FILM_AUTO_EXPOSURE : 0u);
    exposure.manual_exposure_ev = input.manualExposureCompensationEv;
    std::copy(input.inputRgbToXyz.begin(), input.inputRgbToXyz.end(), exposure.input_rgb_to_xyz);
    std::copy(input.inputXyzAdapt.begin(), input.inputXyzAdapt.end(), exposure.input_xyz_adapt);
    std::copy(input.xyzToLinearSrgb.begin(), input.xyzToLinearSrgb.end(), exposure.xyz_to_linear_srgb);
    exposure.sensitivity_rgb = triplet_span(input.finalSensitivity);
    exposure.sensitivity_hash = input.finalSensitivityHash;
    exposure.tc_lut_hash = input.tcLutHash;
    exposure.hash = input.hash;
    exposure.mallett_green_midgray_scale = input.mallettGreenMidgrayScale;
    if (payload.filmTcLut)
        exposure.tc_lut_rgba = {payload.filmTcLut->samples().data(), payload.filmTcLut->samples().size()};
    if (input.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Mallett2019) {
        exposure.mallett_illuminant = float_span(payload.exposureTables.illum);
        exposure.mallett_basis = float_span(Spectral::gMallettBasis.data);
    }

    const auto& develop = recipe.filmDevelop;
    out.film_development.log_exposure = float_span(develop.logExposure);
    out.film_development.density_rgb = triplet_span(develop.normalizedDensityCurves);
    for (std::size_t layer = 0; layer < 3; ++layer)
        for (std::size_t channel = 0; channel < 3; ++channel)
            out.film_development.density_layers[layer][channel] = float_span(develop.densityCurvesLayers[layer][channel]);
    copy3(develop.densityCurveGamma, out.film_development.gamma_rgb);
    out.film_development.density_curves_hash = develop.normalizedDensityCurvesHash;
    out.film_development.density_layers_hash = develop.densityCurvesLayersHash;
    out.film_development.hash = develop.hash;

    const auto& dir = recipe.dirCouplers;
    auto& dir_view = out.dir_recipe;
    dir_view.polarity = static_cast<std::uint32_t>(dir.polarity);
    dir_view.mode = static_cast<std::uint32_t>(dir.nonlinearMode);
    for (std::size_t row = 0; row < 3; ++row)
        std::copy(dir.matrixRgb[row].begin(), dir.matrixRgb[row].end(), dir_view.matrix_rgb + row * 3);
    dir_view.diffusion_size_um = dir.diffusionSizeUm;
    dir_view.diffusion_tail_um = dir.diffusionTailUm;
    dir_view.diffusion_tail_weight = dir.diffusionTailWeight;
    copy3(dir.densityMaxRgb, dir_view.density_max_rgb);
    copy3(dir.densityRefRgb, dir_view.density_ref_rgb);
    copy3(dir.donorKRgb, dir_view.donor_k_rgb);
    copy3(dir.receiverCRefRgb, dir_view.receiver_c_ref_rgb);
    copy3(dir.receiverKrRgb, dir_view.receiver_kr_rgb);
    for (std::size_t channel = 0; channel < 3; ++channel)
        dir_view.compensated_axes_rgb[channel] = float_span(dir.compensatedDensityCurveAxesRgb[channel]);
    dir_view.compensated_axes_hash = dir.compensatedDensityCurveAxesHash;
    dir_view.hash = dir.hash;

    copy3(recipe.grainContract.densityMinCmy, out.grain_minimum.cmy);
    out.scanner_bounds = density_bounds(recipe.densityBounds);
    if (print)
        out.enlarger_film_bounds = density_bounds(recipe.enlargerFilmBounds);
    out.film_reference.hash = scan.syntheticFilmReference.hash;
    copy3(scan.syntheticFilmReference.baselineRawRgb, out.film_reference.baseline_raw_rgb);
    copy3(scan.syntheticFilmReference.compensatedRawRgb, out.film_reference.compensated_raw_rgb);
    copy3(scan.syntheticFilmReference.baselineDensityCmy, out.film_reference.baseline_density_cmy);
    copy3(scan.syntheticFilmReference.compensatedDensityCmy, out.film_reference.compensated_density_cmy);
    out.film_reference.meter_scale = scan.syntheticFilmReference.baselineMeterScale;

    out.optics = spatial_optics(recipe.spatialOptics);
    out.grain = visual_grain(recipe.visualGrain);
    auto& effects = out.effects;
    const auto& effects_recipe = recipe.filmJuicerEffects;
    effects.film_dust = dust(effects_recipe.filmDust);
    effects.film_scratch = scratch(effects_recipe.filmScratch);
    effects.gate_dust = dust(effects_recipe.gateDust);
    effects.gate_scratch = scratch(effects_recipe.gateScratch);
    effects.gate_weave_amount = effects_recipe.gateWeaveAmount;
    effects.active = effects_recipe.active ? 1u : 0u;
    effects.hash = effects_recipe.hash;

    out.exposure_tables = spectral_view(payload.exposureTables);
    out.scanner_tables = spectral_view(payload.scannerTables);
    std::copy(payload.spdSInv.begin(), payload.spdSInv.end(), out.s_inverse);
    std::copy(payload.filmRawConfig.midgrayDWG, payload.filmRawConfig.midgrayDWG + 3, out.midgray_dwg_rgb);
    std::copy(payload.filmRawConfig.rawMidgray, payload.filmRawConfig.rawMidgray + 3, out.raw_midgray);
    out.raw_green = payload.filmRawConfig.rawMidgrayGreen;
    out.midgray_scale = payload.filmRawConfig.midgrayScale;
    out.raw_midgray_bgr = params.spectralUpsamplingMode == 1 ? 1u : 0u;

    if (input.inputCompressionHull) {
        out.input_hull.center_xy[0] = input.inputCompressionHull->center[0];
        out.input_hull.center_xy[1] = input.inputCompressionHull->center[1];
        out.input_hull.xy = {input.inputCompressionHull->xy.data(), input.inputCompressionHull->xy.size() * 2};
        out.input_hull.hash = input.inputCompressionHull->hash;
        out.input_hull.entry_identity = input.inputCompressionHull->entryIdentity;
    }
    if (payload.outputBoundaryTable) {
        out.output_boundary.cmax = {payload.outputBoundaryTable->cmax.data(), payload.outputBoundaryTable->cmax.size()};
        out.output_boundary.transform_hash = payload.outputBoundaryTable->transformHash;
        out.output_boundary.contract_hash = payload.outputBoundaryTable->contractHash;
        out.output_boundary.hash = payload.outputBoundaryTable->hash;
        out.output_boundary.entry_identity = payload.outputBoundaryTable->entryIdentity;
    }

    Scanner::ScannerSpectralLutDescriptor lut;
    const bool lut_ok = print
                            ? Scanner::build_print_scanner_spectral_lut_descriptor({&recipe.profileRoute, &recipe.densityBounds, &scan}, lut, diagnostic)
                            : Scanner::build_direct_scanner_spectral_lut_descriptor({&recipe.profileRoute, &recipe.densityBounds, &scan}, lut, diagnostic);
    if (!lut_ok)
        return false;
    out.scanner_lut = scanner_lut(lut);
    out.scanner_color = scanner_color(payload.scannerColor);
    Scanner::ScannerPostEffectsDescriptor post;
    if (!Scanner::build_scanner_post_effects_descriptor(scan, post, diagnostic))
        return false;
    out.scanner_post_effects = scanner_post(post);
    out.output_color = output_color(payload.scannerColor, scan.outputGamut, payload.outputBoundaryTable.get());

    out.scanner_recipe_hash = scan.hash;
    out.print_recipe_hash = recipe.print.hash;
    out.print_filters_hash = recipe.print.filters.hash;
    out.print_exposure_hash = recipe.print.exposure.hash;
    out.print_illuminant_hash = recipe.print.illuminant.hash;
    out.print_curves_hash = recipe.print.develop.densityCurvesHash;
    if (!print) {
        Scanner::ScannerColorCorrectionDescriptor correction;
        if (!Scanner::build_direct_scanner_color_correction_descriptor(recipe, payload.scannerTables, correction, diagnostic))
            return false;
        out.scanner_correction = scanner_correction(correction);
        out.film_exposure.route_correction_scale = correction.exposureScale;
        return true;
    }
    if (!recipe.profileRoute.filmProfile || !recipe.profileRoute.printProfile || !payload.printMainIlluminant) {
        diagnostic = "MissingRequiredResource phase=I1 field=print_profile_projection";
        return false;
    }

    auto& assets = JuicerProcess::root().assets();
    const auto& film_data = recipe.profileRoute.filmProfile->data;
    const auto& print_data = recipe.profileRoute.printProfile->data;
    out.print.density_cmy = triplet_span(recipe.print.develop.densityCurves);
    out.print.film_density_cmy = triplet_span(film_data.channelDensity);
    out.print.film_base_density = float_span(film_data.baseDensity);
    out.print.sensitivity_cmy = triplet_span(print_data.linearSensitivity);
    out.print.log_exposure = float_span(print_data.logExposure);
    out.print_gamma = recipe.print.develop.gammaFactor;
    out.print_curves_hash = recipe.print.develop.densityCurvesHash;
    out.print_filters_hash = recipe.print.filters.hash;
    out.print_exposure_hash = recipe.print.exposure.hash;
    out.print_illuminant_hash = recipe.print.illuminant.hash;

    const auto calibration = assets.neutral_print_calibration(recipe.profileRoute.printProfile->info.stock, recipe.print.illuminant.key, recipe.profileRoute.filmProfile->info.stock);
    if (calibration.status == JuicerAssets::NeutralPrintCalibrationStatus::Malformed) {
        diagnostic = "MalformedNeutralPrintCalibration phase=4A field=cmy_cc";
        return false;
    }
    out.calibration = static_cast<std::uint32_t>(calibration.status);
    constexpr std::array<float, 3> kCurrentNeutral{0.0f, 65.0f, 55.0f};
    copy3(calibration.status == JuicerAssets::NeutralPrintCalibrationStatus::Found ? calibration.cmyCc : kCurrentNeutral, out.neutral_cmy_cc);

    const auto illuminant_curves = assets.illuminant_filter_curves();
    if (!illuminant_curves) {
        diagnostic = "MissingRequiredResource phase=4A field=illuminant_filter_curves";
        return false;
    }
    std::span<const float> source_light;
    Spectral::Curve equal_energy_illuminant;
    const auto& light_key = recipe.print.illuminant.key;
    if (light_key == "D65")
        source_light = illuminant_curves->d65.linear;
    else if (light_key == "D55")
        source_light = illuminant_curves->d55.linear;
    else if (light_key == "D50")
        source_light = illuminant_curves->d50.linear;
    else if (light_key == "T")
        source_light = illuminant_curves->tungsten.linear;
    else if (light_key == "K75P")
        source_light = illuminant_curves->kinoton75P.linear;
    else if (light_key == "TH-KG3")
        source_light = illuminant_curves->tungstenKg3.linear;
    else if (light_key == "TH-KG3-L")
        source_light = illuminant_curves->tungstenKg3Lens.linear;
    else if (light_key == "EQUAL") {
        equal_energy_illuminant = JuicerIlluminant::equal_energy();
        source_light = equal_energy_illuminant.linear;
    } else {
        diagnostic = "UnsupportedInput phase=I1 field=print_source_illuminant";
        return false;
    }
    if (source_light.size() != 81) {
        diagnostic = "ResourceDescriptorMismatch phase=I1 field=print_source_illuminant";
        return false;
    }
    std::copy(source_light.begin(), source_light.end(), backing.print_source_illuminant.begin());
    out.print_source_illuminant = {backing.print_source_illuminant.data(), backing.print_source_illuminant.size()};

    JuicerCuda::PrintResourceInput print_resources{};
    if (!JuicerCuda::build_print_resource_input({&recipe, &JuicerProcess::root().assets(), &*payload.printMainIlluminant}, print_resources, backing.preflash, diagnostic))
        return false;
    const auto& resources = print_resources;
    const auto& descriptors = resources.descriptors;
    out.print.main_illuminant = {resources.mainIlluminant.data(), resources.mainIlluminant.size()};
    if (descriptors.preflashActive)
        out.print.preflash_illuminant = {resources.preflashIlluminant.data(), resources.preflashIlluminant.size()};
    out.print.flags = descriptors.preflashActive ? FJ_PRINT_PREFLASH : 0u;
    const auto& filters = recipe.print.filters;
    out.print.main_cc_cmy[0] = filters.mainCmyCc.c;
    out.print.main_cc_cmy[1] = filters.mainCmyCc.m;
    out.print.main_cc_cmy[2] = filters.mainCmyCc.y;
    if (descriptors.preflashActive) {
        out.print.preflash_cc_cmy[0] = filters.preflashCmyCc.c;
        out.print.preflash_cc_cmy[1] = filters.preflashCmyCc.m;
        out.print.preflash_cc_cmy[2] = filters.preflashCmyCc.y;
    }
    out.print.exposure = recipe.print.exposure.printExposure;
    out.print.preflash_exposure = recipe.print.exposure.preflashExposure;
    copy3(resources.preflashRawCmy, out.print.preflash_raw_cmy);
    out.print.normalizer = resources.normalizer;
    out.print.film_density_hash = descriptors.filmDensityTables.hash;
    out.print.profile_tables_hash = descriptors.profileTables.hash;
    out.print.main_illuminant_hash = descriptors.mainIlluminant.hash;
    out.print.preflash_illuminant_hash = descriptors.preflashIlluminant.hash;
    out.print.preflash_raw_hash = descriptors.preflashRaw.hash;
    out.print.balance_hash = descriptors.balance.hash;
    out.print.hash = descriptors.hash;

    auto& balance = out.print_balance;
    copy3(recipe.print.balance.baselinePrintRawRgb, balance.baseline_raw_rgb);
    copy3(recipe.print.balance.compensatedPrintRawRgb, balance.compensated_raw_rgb);
    balance.factor_midgray = recipe.print.balance.factorMidgray;
    balance.factor_midgray_comp = recipe.print.balance.factorMidgrayComp;
    balance.normalizer = recipe.print.balance.normalizer;
    balance.filtered_main_illuminant_hash = recipe.print.balance.filteredMainIlluminantHash;
    balance.hash = recipe.print.balance.hash;
    out.print_filter_main_cmy_cc[0] = filters.mainCmyCc.c;
    out.print_filter_main_cmy_cc[1] = filters.mainCmyCc.m;
    out.print_filter_main_cmy_cc[2] = filters.mainCmyCc.y;
    out.print_filter_preflash_cmy_cc[0] = filters.preflashCmyCc.c;
    out.print_filter_preflash_cmy_cc[1] = filters.preflashCmyCc.m;
    out.print_filter_preflash_cmy_cc[2] = filters.preflashCmyCc.y;
    out.print_normalization_mode = static_cast<std::uint32_t>(recipe.print.exposure.normalizationMode);
    out.normalize_print_exposure = recipe.print.exposure.normalizePrintExposure ? 1u : 0u;
    out.print_exposure_compensation = recipe.print.exposure.printExposureCompensation ? 1u : 0u;

    Scanner::ScannerColorCorrectionDescriptor correction;
    if (!Scanner::build_print_scanner_color_correction_descriptor(
            {&recipe, &payload.scannerTables, payload.printMainIlluminant->data(), 81, resources.preflashRawCmy.data(), resources.normalizer},
            correction,
            diagnostic))
        return false;
    out.scanner_correction = scanner_correction(correction);
    out.film_exposure.route_correction_scale = 1.0f;
    out.print_recipe_hash = recipe.print.hash;
    out.print_filters_hash = recipe.print.filters.hash;
    out.print_exposure_hash = recipe.print.exposure.hash;
    out.print_illuminant_hash = recipe.print.illuminant.hash;
    out.print_curves_hash = recipe.print.develop.densityCurvesHash;
    return true;
}
