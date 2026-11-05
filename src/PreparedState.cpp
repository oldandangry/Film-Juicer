#include "PreparedState.h"

#include <algorithm>
#include <cstdint>

#include "JuicerState.h"
#include "juicer_cuda_call_scope.h"

FjPreparationParams preparation_params(const ParamSnapshot& p) {
    FjPreparationParams out{};
    out.print_exposure = p.printExposure;
    out.print_preflash_exposure = p.printPreflashExposure;
    out.print_gamma_factor = p.printGammaFactor;
    out.preflash_m_filter_cc = p.preflashMFilterCc;
    out.preflash_y_filter_cc = p.preflashYFilterCc;
    out.print_shadow_compensation_factor = p.printShadowCompensationFactor;
    out.print_shadow_compensation_density = p.printShadowCompensationDensity;
    out.print_shadow_compensation_transition = p.printShadowCompensationTransition;
    out.glare_percent = p.glarePercent;
    out.glare_roughness = p.glareRoughness;
    out.glare_blur_sigma_px = p.glareBlurSigmaPx;
    out.camera_exposure_compensation_ev = p.cameraExposureCompensationEv;
    out.camera_film_format_long_edge_mm = p.cameraFilmFormatLongEdgeMm;
    out.film_gamma_factor = p.filmGammaFactor;
    out.scanner_lens_blur_sigma_px = p.scannerLensBlurSigmaPx;
    out.scanner_black_level = p.scannerBlackLevel;
    out.scanner_white_level = p.scannerWhiteLevel;
    out.gate_weave_amount = p.gateWeaveAmount;
    std::copy(p.scannerUnsharpMask.begin(), p.scannerUnsharpMask.end(), out.scanner_unsharp_mask);
    std::copy(p.printUiYmcCc.begin(), p.printUiYmcCc.end(), out.print_ui_ymc_cc);
    std::copy(p.cameraFilterUV.begin(), p.cameraFilterUV.end(), out.camera_filter_uv);
    std::copy(p.cameraFilterIR.begin(), p.cameraFilterIR.end(), out.camera_filter_ir);
    const auto diffusion = [](const Spektrafilm::DiffusionFilterAuthoredControls& controls) {
        return FjDiffusionParams{controls.active ? 1u : 0u, static_cast<std::uint32_t>(controls.family), controls.strength, controls.spatialScale, controls.haloWarmth, controls.coreIntensity, controls.coreSize, controls.haloIntensity, controls.haloSize, controls.bloomIntensity, controls.bloomSize};
    };
    out.camera_diffusion = diffusion(p.cameraDiffusion);
    out.enlarger_diffusion = diffusion(p.enlargerDiffusion);
    const auto& dir = p.dirCouplers;
    out.dir_couplers = {dir.active ? 1u : 0u, dir.amount, dir.inhibitionSameLayer, dir.inhibitionInterlayer, dir.diffusionSizeUm, dir.diffusionTailUm, dir.diffusionTailWeight, dir.gammaUseStock ? 1u : 0u, {}, {}, {}, {}, {}, {}};
    std::copy(dir.gammaSameLayerRgb.begin(), dir.gammaSameLayerRgb.end(), out.dir_couplers.gamma_same_layer_rgb);
    std::copy(dir.gammaInterlayerRToGb.begin(), dir.gammaInterlayerRToGb.end(), out.dir_couplers.gamma_interlayer_r_to_gb);
    std::copy(dir.gammaInterlayerGToRb.begin(), dir.gammaInterlayerGToRb.end(), out.dir_couplers.gamma_interlayer_g_to_rb);
    std::copy(dir.gammaInterlayerBToRg.begin(), dir.gammaInterlayerBToRg.end(), out.dir_couplers.gamma_interlayer_b_to_rg);
    std::copy(dir.langmuirDonorKRgb.begin(), dir.langmuirDonorKRgb.end(), out.dir_couplers.langmuir_donor_k_rgb);
    std::copy(dir.langmuirReceiverKRgb.begin(), dir.langmuirReceiverKRgb.end(), out.dir_couplers.langmuir_receiver_k_rgb);
    out.film_profile_key = {p.filmProfileKey.data(), p.filmProfileKey.size()};
    out.print_profile_key = {p.printProfileKey.data(), p.printProfileKey.size()};
    out.spectral_upsampling_mode = p.spectralUpsamplingMode;
    out.ref_ill = p.refIll;
    out.enl_ill = p.enlIll;
    out.normalize_print_exposure = p.normalizePrintExposure;
    out.print_exposure_compensation = p.printExposureCompensation;
    out.input_color_space = p.inputColorSpace;
    out.input_cctf_decoding = p.inputCctfDecoding;
    out.input_compression_enabled = p.inputCompressionEnabled;
    out.hanatos2025_adaptation_window = p.hanatos2025AdaptationWindow;
    out.hanatos2025_adaptation_surface = p.hanatos2025AdaptationSurface;
    out.camera_auto_exposure_enabled = p.cameraAutoExposureEnabled;
    out.camera_metering_method = p.cameraMeteringMethod;
    const auto& halation = p.scatterHalationControls;
    out.scatter_halation = {halation.active ? 1u : 0u, halation.scatterAmount, halation.scatterSpatialScale, halation.halationAmount, halation.halationSpatialScale};
    out.scanner_black_correction = p.scannerBlackCorrection;
    out.scanner_white_correction = p.scannerWhiteCorrection;
    out.scanner_lut_resolution = p.scannerLutResolution;
    out.output_color_space = p.outputColorSpace;
    out.output_cctf_encoding = p.outputCctfEncoding;
    out.output_gamut_compression_enabled = p.outputGamutCompressionEnabled;
    const auto& grain = p.grainControls;
    out.grain = {grain.active ? 1u : 0u, grain.sublayersActive ? 1u : 0u, grain.particleAreaUm2, {}, {}, {}, {}, grain.correlationSigmaPx, grain.dyeCloudBlurUm, {}, grain.nSubLayers, grain.amplitude, grain.chromaMix, grain.chromaSharedWeight, grain.chromaIndependentWeight, grain.coarseWeight, grain.midWeight, grain.sizeMixScale, grain.clumpTemporalMix, grain.clumpMorphPeriodSec, grain.debugView};
    std::copy(grain.particleScaleCmy.begin(), grain.particleScaleCmy.end(), out.grain.particle_scale_cmy);
    std::copy(grain.particleScaleLayers.begin(), grain.particleScaleLayers.end(), out.grain.particle_scale_layers);
    std::copy(grain.visualParticleDensityMinCmy.begin(), grain.visualParticleDensityMinCmy.end(), out.grain.particle_density_min_cmy);
    std::copy(grain.uniformityCmy.begin(), grain.uniformityCmy.end(), out.grain.uniformity_cmy);
    std::copy(grain.microStructure.begin(), grain.microStructure.end(), out.grain.micro_structure);
    out.film_dust_amount = p.filmDustAmount;
    out.film_scratch_amount = p.filmScratchAmount;
    out.gate_dust_amount = p.gateDustAmount;
    out.gate_scratch_amount = p.gateScratchAmount;
    out.scan_route = static_cast<std::uint32_t>(p.scanRoute);
    out.glare_active = p.glareActive ? 1u : 0u;
    out.camera_filter_override = p.cameraFilterOverride ? 1u : 0u;
    return out;
}

PreparedState::PreparedState(FjPreparedState* owner, AdoptTag) noexcept : owner_(owner) {}

PreparedState::PreparedState(PreparedState&& other) noexcept : owner_(other.owner_) {
    other.owner_ = nullptr;
}

PreparedState& PreparedState::operator=(PreparedState&& other) noexcept {
    if (this != &other) {
        release();
        owner_ = other.owner_;
        other.owner_ = nullptr;
    }
    return *this;
}

PreparedState::~PreparedState() {
    release();
}

void PreparedState::release() noexcept {
    if (!owner_) {
        return;
    }
    FjPreparedState* owner = owner_;
    owner_ = nullptr;
    FjErrorBuffer ignored{nullptr, 0, 0};
    (void)fj_legacy_prepared_state_release(owner, &ignored);
}

FjStatus PreparedState::view(FjPreparedStateView& out, FjErrorBuffer& error) const {
    return fj_legacy_prepared_state_view(owner_, &out, &error);
}

FjStatus PreparedState::frame_policy(
    const FjExtent& extent, FjPreparedFramePolicy& out, FjErrorBuffer& error) const {
    return fj_legacy_prepared_state_frame_policy(owner_, &extent, &out, &error);
}

FjCallCompletionResult PreparedState::render(
    FjAssets* assets, FjCudaCallAccess access, const FjFrame& frame, const FjPreparedFramePolicy& policy, FjSubmissionAccess submission, FjAbortCallback abort, FjErrorBuffer& error) const {
    return fj_legacy_prepared_state_render(
        assets, owner_, access, &frame, &policy, submission, abort, &error);
}

FjCallCompletionResult PreparedState::reject_missing_instance(
    FjAssets* assets, FjCudaCallAccess access, const FjFrame& frame, FjAbortCallback abort, FjErrorBuffer& error) const {
    return fj_legacy_prepared_unavailable(
        assets, owner_, FJ_UNAVAILABLE_INSTANCE, access, &frame, abort, &error);
}

FjCallCompletionResult reject_missing_render_state(
    FjAssets* assets, FjCudaCallAccess access, const FjFrame& frame, FjAbortCallback abort, FjErrorBuffer& error) {
    return fj_legacy_prepared_unavailable(
        assets, nullptr, FJ_UNAVAILABLE_STATE, access, &frame, abort, &error);
}

FjStatus prepare_state(
    FjAssets* assets,
    const ParamSnapshot& snapshot,
    std::optional<PreparedState>& out,
    FjErrorBuffer& error) {
    out.reset();

    const FjPreparationParams params = preparation_params(snapshot);
    FjPreparedState* owner = nullptr;
    const FjStatus status = fj_legacy_prepared_state_create(assets, &params, &owner, &error);
    if (status.category != FJ_STATUS_SUCCESS) {
        if (owner) {
            FjErrorBuffer ignored{nullptr, 0, 0};
            (void)fj_legacy_prepared_state_release(owner, &ignored);
        }
        return status;
    }
    if (!owner) {
        return {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
    }

    out.emplace(PreparedState(owner, PreparedState::AdoptTag{}));
    return status;
}
