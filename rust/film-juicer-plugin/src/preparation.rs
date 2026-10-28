//! Temporary held-state and derived-entry foreign edge. Both production and
//! the feature-only facade call these private operations, never each other's exports.

use std::mem::{align_of, size_of};
use std::sync::Arc;

use film_juicer_core::assets::{Assets, Illuminants, InputHull};
use film_juicer_core::{color, gamut, recipe};
use film_juicer_core::{film, optics, profile, route::ScanRoute};
use film_juicer_core::{effects, scanner, spectral};
use crate::cuda::sys;
use crate::asset_bridge::{self, Failure, FjAssets};
use crate::film_recipe_bridge::{
    FjFilmDirParams, FjVisualGrainParams, FjFilmDirView, FjFilmDensityMinimum, FjFilmDensityBounds,
    FjFilmReference, FjVisualGrainRecipe, FjPhysicalEffectsRecipe,
};
use crate::optics_bridge::{
    FjDiffusionParams, FjSpatialOpticsRecipe, FjDiffusionRecipe, FjLensBlurRecipe, FjHalationRecipe,
};
use crate::scanner_bridge::{
    FjScannerLutDescriptor, FjScannerColor, FjScannerColorCorrection,
    FjScannerPostEffectsDescriptor,
};
use crate::print_bridge::FjPrintBalance;
use crate::cuda::sys::{
    FjErrorBuffer, FjFloatSpan, FjStatus, FJ_API_NONE, FJ_STATUS_SUCCESS,
    FJ_STATUS_UNSUPPORTED_INPUT, FJ_STATUS_PREPARATION_FAILURE, FJ_STATUS_ALLOCATION_FAILURE,
    FJ_STATUS_INTERNAL_FAILURE,
};

#[repr(C)]
pub(crate) struct FjIlluminantsView {
    pub(crate) d65: FjFloatSpan,
    pub(crate) d55: FjFloatSpan,
    pub(crate) d50: FjFloatSpan,
    pub(crate) tungsten: FjFloatSpan,
    pub(crate) k75p: FjFloatSpan,
    pub(crate) tungsten_kg3: FjFloatSpan,
    pub(crate) tungsten_kg3_lens: FjFloatSpan,
    pub(crate) entry_identity: usize,
}
#[repr(C)]
pub(crate) struct FjInputHullView {
    pub(crate) center_xy: [f32; 2],
    pub(crate) xy: FjFloatSpan,
    pub(crate) hash: u64,
    pub(crate) entry_identity: usize,
}
#[repr(C)]
pub(crate) struct FjOutputBoundaryView {
    pub(crate) cmax: FjFloatSpan,
    pub(crate) transform_hash: u64,
    pub(crate) contract_hash: u64,
    pub(crate) hash: u64,
    pub(crate) entry_identity: usize,
}
#[repr(C)]
pub(crate) struct FjOutputBoundarySource {
    pub(crate) output_space: u32,
    pub(crate) d65_xyz_to_native_rgb: [f32; 9],
    pub(crate) transform_hash: u64,
}
pub(crate) struct FjIlluminantsOwner {
    entry: Arc<Illuminants>,
}
pub(crate) struct FjInputHullOwner {
    entry: Arc<InputHull>,
}
pub(crate) struct FjOutputBoundaryOwner {
    entry: Arc<film_juicer_core::assets::OutputBoundary>,
}

#[repr(C)]
pub(crate) struct FjPreparationHalation {
    pub(crate) active: u32,
    pub(crate) scatter_amount: f32,
    pub(crate) scatter_spatial_scale: f32,
    pub(crate) halation_amount: f32,
    pub(crate) halation_spatial_scale: f32,
}
#[repr(C)]
pub(crate) struct FjPreparationParams {
    pub(crate) print_exposure: f64,
    pub(crate) print_preflash_exposure: f64,
    pub(crate) print_gamma_factor: f64,
    pub(crate) preflash_m_filter_cc: f64,
    pub(crate) preflash_y_filter_cc: f64,
    pub(crate) print_shadow_compensation_factor: f64,
    pub(crate) print_shadow_compensation_density: f64,
    pub(crate) print_shadow_compensation_transition: f64,
    pub(crate) glare_percent: f64,
    pub(crate) glare_roughness: f64,
    pub(crate) glare_blur_sigma_px: f64,
    pub(crate) camera_exposure_compensation_ev: f64,
    pub(crate) camera_film_format_long_edge_mm: f32,
    pub(crate) film_gamma_factor: f32,
    pub(crate) scanner_lens_blur_sigma_px: f64,
    pub(crate) scanner_black_level: f64,
    pub(crate) scanner_white_level: f64,
    pub(crate) gate_weave_amount: f64,
    pub(crate) scanner_unsharp_mask: [f64; 2],
    pub(crate) print_ui_ymc_cc: [f64; 3],
    pub(crate) camera_filter_uv: [f64; 3],
    pub(crate) camera_filter_ir: [f64; 3],
    pub(crate) camera_diffusion: FjDiffusionParams,
    pub(crate) enlarger_diffusion: FjDiffusionParams,
    pub(crate) dir_couplers: FjFilmDirParams,
    pub(crate) film_profile_key: crate::cuda::sys::FjStringView,
    pub(crate) print_profile_key: crate::cuda::sys::FjStringView,
    pub(crate) spectral_upsampling_mode: i32,
    pub(crate) ref_ill: i32,
    pub(crate) enl_ill: i32,
    pub(crate) normalize_print_exposure: i32,
    pub(crate) print_exposure_compensation: i32,
    pub(crate) input_color_space: i32,
    pub(crate) input_cctf_decoding: i32,
    pub(crate) input_compression_enabled: i32,
    pub(crate) hanatos2025_adaptation_window: i32,
    pub(crate) hanatos2025_adaptation_surface: i32,
    pub(crate) camera_auto_exposure_enabled: i32,
    pub(crate) camera_metering_method: i32,
    pub(crate) scatter_halation: FjPreparationHalation,
    pub(crate) scanner_black_correction: i32,
    pub(crate) scanner_white_correction: i32,
    pub(crate) scanner_lut_resolution: i32,
    pub(crate) output_color_space: i32,
    pub(crate) output_cctf_encoding: i32,
    pub(crate) output_gamut_compression_enabled: i32,
    pub(crate) grain: FjVisualGrainParams,
    pub(crate) film_dust_amount: f32,
    pub(crate) film_scratch_amount: f32,
    pub(crate) gate_dust_amount: f32,
    pub(crate) gate_scratch_amount: f32,
    pub(crate) scan_route: u32,
    pub(crate) glare_active: u32,
    pub(crate) camera_filter_override: u32,
}

pub(crate) struct FjPreparedState {
    // A fallible substantial allocation, moved once after complete preparation.
    // The foreign Box owns only this small header; projections never self-point.
    state: Vec<recipe::RenderState>,
}
#[cfg(feature = "test-support")]
static LIVE_STATES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "test-support")]
impl Drop for FjPreparedState {
    fn drop(&mut self) {
        LIVE_STATES.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}
#[cfg(feature = "test-support")]
pub(crate) fn live_states() -> usize {
    LIVE_STATES.load(std::sync::atomic::Ordering::Relaxed)
}
#[cfg(feature = "test-support")]
#[repr(C)]
pub(crate) struct FjPreparationTrace {
    pub(crate) count: u32,
    pub(crate) stages: [u32; 24],
}

#[repr(C)]
pub(crate) struct FjPreparedSpectralView {
    pub(crate) lambda_nm: FjFloatSpan,
    pub(crate) illuminant: FjFloatSpan,
    pub(crate) observer_xyz: [FjFloatSpan; 3],
    pub(crate) weighted_xyz: [FjFloatSpan; 3],
    pub(crate) dyes_cmy: [FjFloatSpan; 3],
    pub(crate) baseline_min: FjFloatSpan,
    pub(crate) baseline_mid: FjFloatSpan,
    pub(crate) inv_yn: f32,
    pub(crate) white_xyz: [f32; 3],
    pub(crate) has_baseline: u32,
    pub(crate) illuminant_hash: u64,
    pub(crate) hash: u64,
}
#[repr(C)]
pub(crate) struct FjPreparedInputPolicy {
    pub(crate) meter: u32,
    pub(crate) band_pass_active: u32,
    pub(crate) apply_hanatos_window: u32,
    pub(crate) apply_hanatos_surface: u32,
    pub(crate) projection_illuminant: u32,
    pub(crate) input_compression_active: u32,
    pub(crate) film_format_long_edge_mm: f32,
    pub(crate) uv: [f32; 3],
    pub(crate) ir: [f32; 3],
    pub(crate) hanatos_spectral_blur: f32,
    pub(crate) nominal_white_xyz: [f32; 3],
    pub(crate) projection_white_xyz: [f32; 3],
    pub(crate) hanatos_window: FjFloatSpan,
    pub(crate) hanatos_surface_rgb: FjFloatSpan,
    pub(crate) input_hull_hash: u64,
    pub(crate) tc_source_hash: u64,
}
#[repr(C)]
pub(crate) struct FjPreparedScannerPolicy {
    pub(crate) black_correction: u32,
    pub(crate) white_correction: u32,
    pub(crate) glare_active: u32,
    pub(crate) black_level: f32,
    pub(crate) white_level: f32,
    pub(crate) glare_percent: f32,
    pub(crate) glare_roughness: f32,
    pub(crate) glare_blur_sigma_px: f32,
    pub(crate) lens_blur_sigma_px: f32,
    pub(crate) unsharp_sigma_px: f32,
    pub(crate) unsharp_amount: f32,
}
#[repr(C)]
pub(crate) struct FjPreparedStateView {
    pub(crate) film_profile_key: crate::cuda::sys::FjStringView,
    pub(crate) print_profile_key: crate::cuda::sys::FjStringView,
    pub(crate) film_profile_asset_version: u64,
    pub(crate) print_profile_asset_version: u64,
    pub(crate) profile_route_hash: u64,
    pub(crate) recipe_hash: u64,
    pub(crate) upload_core_hash: u64,
    pub(crate) scanner_hash: u64,
    pub(crate) route: u32,
    pub(crate) polarity: u32,
    pub(crate) input_policy: FjPreparedInputPolicy,
    pub(crate) scanner_policy: FjPreparedScannerPolicy,
    pub(crate) film_exposure: crate::cuda::sys::FjFilmExposure,
    pub(crate) film_development: crate::cuda::sys::FjFilmDevelopment,
    pub(crate) dir_recipe: FjFilmDirView,
    pub(crate) grain_minimum: FjFilmDensityMinimum,
    pub(crate) scanner_bounds: FjFilmDensityBounds,
    pub(crate) enlarger_film_bounds: FjFilmDensityBounds,
    pub(crate) film_reference: FjFilmReference,
    pub(crate) optics: FjSpatialOpticsRecipe,
    pub(crate) grain: FjVisualGrainRecipe,
    pub(crate) effects: FjPhysicalEffectsRecipe,
    pub(crate) exposure_tables: FjPreparedSpectralView,
    pub(crate) scanner_tables: FjPreparedSpectralView,
    pub(crate) s_inverse: [f32; 9],
    pub(crate) midgray_dwg_rgb: [f32; 3],
    pub(crate) raw_midgray: [f32; 3],
    pub(crate) raw_green: f32,
    pub(crate) midgray_scale: f32,
    pub(crate) raw_midgray_bgr: u32,
    pub(crate) input_hull: FjInputHullView,
    pub(crate) output_boundary: FjOutputBoundaryView,
    pub(crate) scanner_lut: FjScannerLutDescriptor,
    pub(crate) scanner_color: FjScannerColor,
    pub(crate) scanner_correction: FjScannerColorCorrection,
    pub(crate) scanner_post_effects: FjScannerPostEffectsDescriptor,
    pub(crate) output_color: crate::cuda::sys::FjOutputColor,
    pub(crate) print: crate::cuda::sys::FjPrint,
    pub(crate) print_gamma: f64,
    pub(crate) print_filter_main_cmy_cc: [f32; 3],
    pub(crate) print_filter_preflash_cmy_cc: [f32; 3],
    pub(crate) print_normalization_mode: u32,
    pub(crate) normalize_print_exposure: u32,
    pub(crate) print_exposure_compensation: u32,
    pub(crate) calibration: u32,
    pub(crate) neutral_cmy_cc: [f32; 3],
    pub(crate) print_source_illuminant: FjFloatSpan,
    pub(crate) print_balance: FjPrintBalance,
    pub(crate) scanner_recipe_hash: u64,
    pub(crate) print_recipe_hash: u64,
    pub(crate) print_filters_hash: u64,
    pub(crate) print_exposure_hash: u64,
    pub(crate) print_illuminant_hash: u64,
    pub(crate) print_curves_hash: u64,
}

fn flag(value: u32) -> Result<bool, Failure> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(Failure::Input("invalid preparation boolean representation")),
    }
}
fn diffusion(input: &FjDiffusionParams) -> Result<optics::DiffusionParams, Failure> {
    Ok(optics::DiffusionParams {
        active: flag(input.active)?,
        family: input.family,
        strength: input.strength,
        spatial_scale: input.spatial_scale,
        halo_warmth: input.halo_warmth,
        core_intensity: input.core_intensity,
        core_size: input.core_size,
        halo_intensity: input.halo_intensity,
        halo_size: input.halo_size,
        bloom_intensity: input.bloom_intensity,
        bloom_size: input.bloom_size,
    })
}
/// # Safety
/// The length-delimited UTF-8 key is immutable, initialized and live until return.
unsafe fn key(input: &crate::cuda::sys::FjStringView) -> Result<String, Failure> {
    if input.count == 0 {
        return Ok(String::new());
    }
    if input.data.is_null()
        || input.count > isize::MAX as usize
        || (input.data as usize).checked_add(input.count).is_none()
    {
        return Err(Failure::Input("invalid preparation key extent"));
    }
    // SAFETY: The caller authorizes the checked immutable byte extent.
    let bytes = unsafe { std::slice::from_raw_parts(input.data.cast::<u8>(), input.count) };
    if bytes.contains(&0) {
        return Err(Failure::Input("preparation key contains NUL"));
    }
    let decoded =
        std::str::from_utf8(bytes).map_err(|_| Failure::Input("preparation key is not UTF-8"))?;
    let mut owned = String::new();
    owned
        .try_reserve_exact(decoded.len())
        .map_err(Failure::Capacity)?;
    owned.push_str(decoded);
    Ok(owned)
}
/// # Safety
/// input is one initialized immutable snapshot and its borrowed keys retain
/// their complete authorized extents until this synchronous conversion returns.
unsafe fn params(input: &FjPreparationParams) -> Result<recipe::Params, Failure> {
    let FjPreparationParams {
        print_exposure,
        print_preflash_exposure,
        print_gamma_factor,
        preflash_m_filter_cc,
        preflash_y_filter_cc,
        print_shadow_compensation_factor,
        print_shadow_compensation_density,
        print_shadow_compensation_transition,
        glare_percent,
        glare_roughness,
        glare_blur_sigma_px,
        camera_exposure_compensation_ev,
        camera_film_format_long_edge_mm,
        film_gamma_factor,
        scanner_lens_blur_sigma_px,
        scanner_black_level,
        scanner_white_level,
        gate_weave_amount,
        scanner_unsharp_mask,
        print_ui_ymc_cc,
        camera_filter_uv,
        camera_filter_ir,
        camera_diffusion,
        enlarger_diffusion,
        dir_couplers,
        film_profile_key,
        print_profile_key,
        spectral_upsampling_mode,
        ref_ill,
        enl_ill,
        normalize_print_exposure,
        print_exposure_compensation,
        input_color_space,
        input_cctf_decoding,
        input_compression_enabled,
        hanatos2025_adaptation_window,
        hanatos2025_adaptation_surface,
        camera_auto_exposure_enabled,
        camera_metering_method,
        scatter_halation,
        scanner_black_correction,
        scanner_white_correction,
        scanner_lut_resolution,
        output_color_space,
        output_cctf_encoding,
        output_gamut_compression_enabled,
        grain,
        film_dust_amount,
        film_scratch_amount,
        gate_dust_amount,
        gate_scratch_amount,
        scan_route,
        glare_active,
        camera_filter_override,
    } = input;
    let scan_route = match scan_route {
        0 => ScanRoute::NegativeDirect,
        1 => ScanRoute::NegativePrint,
        2 => ScanRoute::PositiveDirect,
        3 => ScanRoute::PositivePrint,
        _ => return Err(Failure::Input("invalid preparation route tag")),
    };
    // SAFETY: Only these two authorized key extents are borrowed while copying
    // the captured snapshot. No foreign storage is retained in safe Params.
    let (film_profile_key, print_profile_key) =
        unsafe { (key(film_profile_key)?, key(print_profile_key)?) };
    Ok(recipe::Params {
        print_exposure: *print_exposure,
        print_preflash_exposure: *print_preflash_exposure,
        print_gamma_factor: *print_gamma_factor,
        preflash_m_filter_cc: *preflash_m_filter_cc,
        preflash_y_filter_cc: *preflash_y_filter_cc,
        print_shadow_compensation_factor: *print_shadow_compensation_factor,
        print_shadow_compensation_density: *print_shadow_compensation_density,
        print_shadow_compensation_transition: *print_shadow_compensation_transition,
        glare_percent: *glare_percent,
        glare_roughness: *glare_roughness,
        glare_blur_sigma_px: *glare_blur_sigma_px,
        camera_exposure_compensation_ev: *camera_exposure_compensation_ev,
        camera_film_format_long_edge_mm: *camera_film_format_long_edge_mm,
        film_gamma_factor: *film_gamma_factor,
        scanner_lens_blur_sigma_px: *scanner_lens_blur_sigma_px,
        scanner_black_level: *scanner_black_level,
        scanner_white_level: *scanner_white_level,
        gate_weave_amount: *gate_weave_amount,
        scanner_unsharp_mask: *scanner_unsharp_mask,
        print_ui_ymc_cc: *print_ui_ymc_cc,
        camera_filter_uv: *camera_filter_uv,
        camera_filter_ir: *camera_filter_ir,
        camera_diffusion: diffusion(camera_diffusion)?,
        enlarger_diffusion: diffusion(enlarger_diffusion)?,
        dir_couplers: film::dir::Params {
            active: flag(dir_couplers.active)?,
            amount: dir_couplers.amount,
            inhibition_same_layer: dir_couplers.inhibition_same_layer,
            inhibition_interlayer: dir_couplers.inhibition_interlayer,
            diffusion_size_um: dir_couplers.diffusion_size_um,
            diffusion_tail_um: dir_couplers.diffusion_tail_um,
            diffusion_tail_weight: dir_couplers.diffusion_tail_weight,
            gamma_use_stock: flag(dir_couplers.gamma_use_stock)?,
            gamma_same_layer_rgb: dir_couplers.gamma_same_layer_rgb,
            gamma_interlayer_r_to_gb: dir_couplers.gamma_interlayer_r_to_gb,
            gamma_interlayer_g_to_rb: dir_couplers.gamma_interlayer_g_to_rb,
            gamma_interlayer_b_to_rg: dir_couplers.gamma_interlayer_b_to_rg,
            langmuir_donor_k_rgb: dir_couplers.langmuir_donor_k_rgb,
            langmuir_receiver_k_rgb: dir_couplers.langmuir_receiver_k_rgb,
        },
        film_profile_key,
        print_profile_key,
        spectral_upsampling_mode: *spectral_upsampling_mode,
        ref_ill: *ref_ill,
        enl_ill: *enl_ill,
        normalize_print_exposure: *normalize_print_exposure,
        print_exposure_compensation: *print_exposure_compensation,
        input_color_space: *input_color_space,
        input_cctf_decoding: *input_cctf_decoding,
        input_compression_enabled: *input_compression_enabled,
        hanatos2025_adaptation_window: *hanatos2025_adaptation_window,
        hanatos2025_adaptation_surface: *hanatos2025_adaptation_surface,
        camera_auto_exposure_enabled: *camera_auto_exposure_enabled,
        camera_metering_method: *camera_metering_method,
        scatter_halation: recipe::HalationControls {
            active: flag(scatter_halation.active)?,
            scatter_amount: scatter_halation.scatter_amount,
            scatter_spatial_scale: scatter_halation.scatter_spatial_scale,
            halation_amount: scatter_halation.halation_amount,
            halation_spatial_scale: scatter_halation.halation_spatial_scale,
        },
        scanner_black_correction: *scanner_black_correction,
        scanner_white_correction: *scanner_white_correction,
        scanner_lut_resolution: *scanner_lut_resolution,
        output_color_space: *output_color_space,
        output_cctf_encoding: *output_cctf_encoding,
        output_gamut_compression_enabled: *output_gamut_compression_enabled,
        grain: film::grain::Params {
            active: flag(grain.active)?,
            sublayers_active: flag(grain.sublayers_active)?,
            particle_area_um2: grain.particle_area_um2,
            particle_scale_cmy: grain.particle_scale_cmy,
            particle_scale_layers: grain.particle_scale_layers,
            particle_density_min_cmy: grain.particle_density_min_cmy,
            uniformity_cmy: grain.uniformity_cmy,
            correlation_sigma_px: grain.correlation_sigma_px,
            dye_cloud_blur_um: grain.dye_cloud_blur_um,
            micro_structure: grain.micro_structure,
            sublayer_count: grain.sublayer_count,
            amplitude: grain.amplitude,
            chroma_mix: grain.chroma_mix,
            chroma_shared_weight: grain.chroma_shared_weight,
            chroma_independent_weight: grain.chroma_independent_weight,
            coarse_weight: grain.coarse_weight,
            mid_weight: grain.mid_weight,
            size_mix_scale: grain.size_mix_scale,
            clump_temporal_mix: grain.clump_temporal_mix,
            clump_morph_period_sec: grain.clump_morph_period_sec,
            debug_view: grain.debug_view,
        },
        film_dust_amount: *film_dust_amount,
        film_scratch_amount: *film_scratch_amount,
        gate_dust_amount: *gate_dust_amount,
        gate_scratch_amount: *gate_scratch_amount,
        scan_route,
        glare_active: flag(*glare_active)?,
        camera_filter_override: flag(*camera_filter_override)?,
    })
}

fn construct(assets: &Assets, params: &recipe::Params) -> Result<FjPreparedState, Failure> {
    let completed = recipe::prepare(assets, params).map_err(|error| match error {
        recipe::PrepareError::Asset(error) => Failure::Asset(error),
        error => Failure::Preparation(error),
    })?;
    let mut state = Vec::new();
    state.try_reserve_exact(1).map_err(Failure::Capacity)?;
    state.push(completed);
    #[cfg(feature = "test-support")]
    LIVE_STATES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(FjPreparedState { state })
}

/// # Safety
/// Inputs, borrowed keys, exclusive owner output and optional diagnostics obey
/// the single snapshot/held-state section of juicer_legacy_api.h.
pub(crate) unsafe fn state_create(
    input_assets: *const FjAssets,
    input_params: *const FjPreparationParams,
    out: *mut *mut FjPreparedState,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: acquire clears/admit output before borrowing either live input.
    unsafe {
        acquire(out, error, || {
            if !aligned(input_params) {
                return Err(Failure::Input("invalid preparation snapshot record"));
            }
            let params = params(&*input_params)?;
            construct(assets(input_assets)?, &params)
        })
    }
}
/// # Safety
/// The caller transfers one matching live state handle and excludes every view
/// borrow; diagnostic rejection still consumes that one hold.
pub(crate) unsafe fn state_release(
    owner: *mut FjPreparedState,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The common release consumes the caller-authorized matching Box.
    unsafe { release(owner, error) }
}

fn route_tag(route: ScanRoute) -> u32 {
    match route {
        ScanRoute::NegativeDirect => 0,
        ScanRoute::NegativePrint => 1,
        ScanRoute::PositiveDirect => 2,
        ScanRoute::PositivePrint => 3,
    }
}
fn polarity_tag(polarity: profile::Polarity) -> u32 {
    match polarity {
        profile::Polarity::Negative => 0,
        profile::Polarity::Positive => 1,
    }
}
fn text(value: &str) -> sys::FjStringView {
    sys::FjStringView {
        data: if value.is_empty() {
            std::ptr::null()
        } else {
            value.as_ptr().cast()
        },
        count: value.len(),
    }
}
fn spectral_projection(tables: &spectral::Tables) -> FjPreparedSpectralView {
    FjPreparedSpectralView {
        lambda_nm: floats(tables.lambda_nm()),
        illuminant: floats(tables.illuminant()),
        observer_xyz: std::array::from_fn(|c| floats(&tables.observer_xyz()[c])),
        weighted_xyz: std::array::from_fn(|c| floats(&tables.weighted_xyz()[c])),
        dyes_cmy: std::array::from_fn(|c| floats(&tables.dyes_cmy()[c])),
        baseline_min: floats(tables.baseline_min()),
        baseline_mid: floats(tables.baseline_mid()),
        inv_yn: tables.inv_yn(),
        white_xyz: *tables.white_xyz(),
        has_baseline: u32::from(tables.has_baseline()),
        illuminant_hash: tables.illuminant_hash(),
        hash: tables.hash(),
    }
}
fn film_bounds(bounds: &film::density::Bounds) -> FjFilmDensityBounds {
    FjFilmDensityBounds {
        route: route_tag(bounds.route()),
        medium: 0,
        polarity: polarity_tag(bounds.polarity()),
        source: match bounds.source() {
            film::density::Source::DirectFilm => 0,
            film::density::Source::EnlargerFilm => 1,
        },
        minimum_cmy: *bounds.minimum_cmy(),
        maximum_cmy: *bounds.maximum_cmy(),
        inv_span_cmy: *bounds.inv_span_cmy(),
        authored_minimum_cmy: *bounds.authored_minimum_cmy(),
        authored_maximum_cmy: *bounds.authored_maximum_cmy(),
        hash: bounds.hash(),
    }
}
fn correction_projection(correction: &scanner::Correction) -> FjScannerColorCorrection {
    FjScannerColorCorrection {
        route: u32::from(correction.route()),
        active: u32::from(correction.active()),
        black_correction: u32::from(correction.black_correction()),
        white_correction: u32::from(correction.white_correction()),
        target_black_linear: correction.target_black_linear(),
        target_white_linear: correction.target_white_linear(),
        reference_black_y: correction.reference_black_y(),
        reference_white_y: correction.reference_white_y(),
        xyz_slope: correction.xyz_slope(),
        xyz_offset: correction.xyz_offset(),
        exposure_scale: correction.exposure_scale(),
        schema_version: correction.schema_version(),
        hash: correction.hash(),
    }
}
fn output_encoding(color: &scanner::Color) -> sys::FjOutputEncoding {
    let cctf = color.cctf();
    sys::FjOutputEncoding {
        schema: sys::FJ_OUTPUT_ENCODING_SCHEMA,
        cctf_kind: cctf.kind as u32,
        gamma: cctf.gamma,
        a: cctf.a,
        b: cctf.b,
        c: cctf.c,
        d: cctf.d,
        linear_cutoff: cctf.linear_cutoff,
        dwg_to_output: *color.dwg_to_output(),
    }
}
fn diffusion_projection(value: &optics::Diffusion) -> FjDiffusionRecipe {
    FjDiffusionRecipe {
        domain: match value.domain() {
            optics::Domain::FilmLinearExposure => 0,
            optics::Domain::PrintLinearExposure => 1,
        },
        active: u32::from(value.active()),
        family: match value.family() {
            optics::Family::Glimmerglass => 0,
            optics::Family::BlackProMist => 1,
            optics::Family::ProMist => 2,
            optics::Family::Cinebloom => 3,
        },
        scatter_fraction: value.scatter_fraction(),
        group_weights: *value.group_weights(),
        group_lambda_um: *value.group_lambda_um(),
        warmth: value.warmth(),
        spatial_scale: value.spatial_scale(),
        resolved_hash: value.resolved_hash(),
        hash: value.hash(),
    }
}
fn optics_projection(value: &optics::Recipe) -> FjSpatialOpticsRecipe {
    let lens = value.camera_lens_blur();
    let halation = value.scatter_halation();
    FjSpatialOpticsRecipe {
        camera_diffusion: diffusion_projection(value.camera_diffusion()),
        camera_lens_blur: FjLensBlurRecipe {
            sigma_um: lens.sigma_um(),
            hash: lens.hash(),
        },
        scatter_halation: FjHalationRecipe {
            scatter_active: u32::from(halation.scatter_active()),
            back_reflection_active: u32::from(halation.back_reflection_active()),
            scatter_amount: halation.scatter_amount(),
            scatter_spatial_scale: halation.scatter_spatial_scale(),
            halation_spatial_scale: halation.halation_spatial_scale(),
            first_sigma_um: *halation.first_sigma_um(),
            total_strength: *halation.total_strength(),
            hash: halation.hash(),
        },
        enlarger_diffusion: diffusion_projection(value.enlarger_diffusion()),
        hash: value.hash(),
    }
}
fn grain_projection(value: &film::grain::Recipe) -> FjVisualGrainRecipe {
    FjVisualGrainRecipe {
        active: u32::from(value.active()),
        sublayers_active: u32::from(value.sublayers_active()),
        axis_finite_cmy: value.axis_finite_cmy().map(u32::from),
        axis_block_prefix_max_cmy: *value.axis_block_prefix_max_cmy(),
        particle_area_um2: value.particle_area_um2(),
        particle_scale_cmy: *value.particle_scale_cmy(),
        particle_scale_layers: *value.particle_scale_layers(),
        particle_density_min_cmy: *value.particle_density_min_cmy(),
        uniformity_cmy: *value.uniformity_cmy(),
        correlation_sigma_px: value.correlation_sigma_px(),
        dye_cloud_blur_um: value.dye_cloud_blur_um(),
        micro_structure: *value.micro_structure(),
        sublayer_count: value.sublayer_count(),
        amplitude: value.amplitude(),
        chroma_mix: value.chroma_mix(),
        chroma_shared_weight: value.chroma_shared_weight(),
        chroma_independent_weight: value.chroma_independent_weight(),
        fine_weight: value.fine_weight(),
        mid_weight: value.mid_weight(),
        coarse_weight: value.coarse_weight(),
        size_mix_scale: value.size_mix_scale(),
        clump_temporal_mix: value.clump_temporal_mix(),
        clump_morph_period_sec: value.clump_morph_period_sec(),
        debug_view: value.debug_view(),
        layers_hash: value.layers_hash(),
        hash: value.hash(),
    }
}
fn dust_projection(value: &effects::Dust) -> sys::FjDust {
    sys::FjDust {
        cell_width_mm: value.cell_width_mm(),
        cell_height_mm: value.cell_height_mm(),
        slot_probability: value.slot_probability(),
        softness_min_mm: value.softness_min_mm(),
        softness_max_mm: value.softness_max_mm(),
        softness_size_cap_fraction: value.softness_size_cap_fraction(),
        support_x_mm: value.support_x_mm(),
        support_y_mm: value.support_y_mm(),
        fiber_fraction: value.fiber_fraction(),
        fiber_drift_fraction: value.fiber_drift_fraction(),
        fiber_first_knot_min: value.fiber_first_knot_min(),
        fiber_first_knot_max: value.fiber_first_knot_max(),
        fiber_second_knot_min: value.fiber_second_knot_min(),
        fiber_second_knot_max: value.fiber_second_knot_max(),
        fiber_interior_width_min_fraction: value.fiber_interior_width_min_fraction(),
        fiber_interior_width_max_fraction: value.fiber_interior_width_max_fraction(),
        diameter_min_mm: value.diameter_min_mm(),
        diameter_bulk_max_mm: value.diameter_bulk_max_mm(),
        diameter_max_mm: value.diameter_max_mm(),
        diameter_tail_fraction: value.diameter_tail_fraction(),
        fiber_length_min_mm: value.fiber_length_min_mm(),
        fiber_length_max_mm: value.fiber_length_max_mm(),
        fiber_width_min_mm: value.fiber_width_min_mm(),
        fiber_width_max_mm: value.fiber_width_max_mm(),
        opacity_faint_cumulative: value.opacity_faint_cumulative(),
        opacity_intermediate_cumulative: value.opacity_intermediate_cumulative(),
        compact_opacity_min: value.compact_opacity_min(),
        compact_opacity_faint_end: value.compact_opacity_faint_end(),
        compact_opacity_intermediate_end: value.compact_opacity_intermediate_end(),
        compact_opacity_max: value.compact_opacity_max(),
        fiber_opacity_min: value.fiber_opacity_min(),
        fiber_opacity_faint_end: value.fiber_opacity_faint_end(),
        fiber_opacity_intermediate_end: value.fiber_opacity_intermediate_end(),
        fiber_opacity_max: value.fiber_opacity_max(),
        compact_dominant_aspect_min: value.compact_dominant_aspect_min(),
        compact_dominant_aspect_max: value.compact_dominant_aspect_max(),
        compact_subsidiary_scale_min: value.compact_subsidiary_scale_min(),
        compact_subsidiary_scale_max: value.compact_subsidiary_scale_max(),
        compact_subsidiary_aspect_min: value.compact_subsidiary_aspect_min(),
        compact_subsidiary_aspect_max: value.compact_subsidiary_aspect_max(),
        compact_subsidiary_offset_max: value.compact_subsidiary_offset_max(),
        compact_subsidiary_angle_max_radians: value.compact_subsidiary_angle_max_radians(),
    }
}
fn scratch_projection(value: &effects::Scratch) -> sys::FjScratch {
    sys::FjScratch {
        cell_width_mm: value.cell_width_mm(),
        cell_height_mm: value.cell_height_mm(),
        slot_probability: value.slot_probability(),
        softness_min_mm: value.softness_min_mm(),
        softness_max_mm: value.softness_max_mm(),
        softness_size_cap_fraction: value.softness_size_cap_fraction(),
        support_x_mm: value.support_x_mm(),
        support_y_mm: value.support_y_mm(),
        length_min_mm: value.length_min_mm(),
        length_bulk_max_mm: value.length_bulk_max_mm(),
        length_max_mm: value.length_max_mm(),
        length_tail_fraction: value.length_tail_fraction(),
        width_min_mm: value.width_min_mm(),
        width_bulk_max_mm: value.width_bulk_max_mm(),
        width_max_mm: value.width_max_mm(),
        width_tail_fraction: value.width_tail_fraction(),
        drift_fraction: value.drift_fraction(),
        first_knot_min: value.first_knot_min(),
        first_knot_max: value.first_knot_max(),
        second_knot_min: value.second_knot_min(),
        second_knot_max: value.second_knot_max(),
        interior_width_min_fraction: value.interior_width_min_fraction(),
        interior_width_max_fraction: value.interior_width_max_fraction(),
        interior_depth_min_fraction: value.interior_depth_min_fraction(),
        interior_depth_max_fraction: value.interior_depth_max_fraction(),
        endpoint_abrupt_probability: value.endpoint_abrupt_probability(),
        interruption_probability: value.interruption_probability(),
        gap_center_min: value.gap_center_min(),
        gap_center_max: value.gap_center_max(),
        gap_span_min: value.gap_span_min(),
        gap_span_max: value.gap_span_max(),
        scuff_probability: value.scuff_probability(),
        scuff_length_max_mm: value.scuff_length_max_mm(),
        scuff_angle_max_radians: value.scuff_angle_max_radians(),
        strength_min: value.strength_min(),
        strength_max: value.strength_max(),
    }
}

fn project_state(owner: &FjPreparedState) -> FjPreparedStateView {
    let state = &owner.state[0];
    let film = state.film_profile();
    let input = state.input();
    let development = state.development();
    let scanner = state.scanner();
    let scanner_params = scanner.params();
    let color = scanner.color();
    let correction = match state.route() {
        recipe::PreparedRoute::Direct(direct) => direct.correction(),
        recipe::PreparedRoute::Print(print) => print.correction(),
    };
    // SAFETY: This foreign-only aggregate contains integers, floats and raw spans,
    // all zero-valid. Zero denotes the absent route/family; no owning Rust value
    // or self-reference is stored here. Defined fields are filled from complete
    // owners below; all large arrays remain borrowed, including on warm views.
    let mut out: FjPreparedStateView = unsafe { std::mem::zeroed() };
    out.film_profile_key = text(state.film_key());
    out.film_profile_asset_version = film.asset_token();
    out.profile_route_hash = state.profile_route_hash();
    out.recipe_hash = state.hash();
    out.upload_core_hash = state.upload_core_hash();
    out.scanner_hash = state.scanner_hash();
    out.route = route_tag(development.route());
    out.polarity = polarity_tag(development.polarity());
    out.input_policy = FjPreparedInputPolicy {
        meter: match input.meter() {
            film::input::Meter::CenterWeighted => 0,
            film::input::Meter::Average => 1,
            film::input::Meter::Median => 2,
            film::input::Meter::Partial => 3,
            film::input::Meter::Matrix => 4,
            film::input::Meter::MultiZone => 5,
            film::input::Meter::HighlightWeighted => 6,
        },
        band_pass_active: u32::from(input.band_pass_active()),
        apply_hanatos_window: u32::from(input.apply_hanatos_window()),
        apply_hanatos_surface: u32::from(input.apply_hanatos_surface()),
        projection_illuminant: match input.projection_illuminant() {
            film::input::ProjectionIlluminant::ProfileReference => 0,
            film::input::ProjectionIlluminant::D65 => 1,
        },
        input_compression_active: u32::from(input.input_compression_active()),
        film_format_long_edge_mm: input.film_format_long_edge_mm(),
        uv: *input.uv(),
        ir: *input.ir(),
        hanatos_spectral_blur: input.hanatos_spectral_blur(),
        nominal_white_xyz: *input.color().nominal_white_xyz(),
        projection_white_xyz: *input.color().projection_white_xyz(),
        hanatos_window: floats(input.hanatos_window()),
        hanatos_surface_rgb: floats(input.hanatos_surface_rgb().as_flattened()),
        input_hull_hash: input.input_hull_hash(),
        tc_source_hash: input.tc_source_hash(),
    };
    out.scanner_policy = FjPreparedScannerPolicy {
        black_correction: u32::from(scanner_params.black_correction),
        white_correction: u32::from(scanner_params.white_correction),
        glare_active: u32::from(scanner_params.glare_active),
        black_level: scanner_params.black_level,
        white_level: scanner_params.white_level,
        glare_percent: scanner_params.glare_percent,
        glare_roughness: scanner_params.glare_roughness,
        glare_blur_sigma_px: scanner_params.glare_blur_sigma_px,
        lens_blur_sigma_px: scanner_params.lens_blur_sigma_px,
        unsharp_sigma_px: scanner_params.unsharp_sigma_px,
        unsharp_amount: scanner_params.unsharp_amount,
    };
    out.film_exposure = sys::FjFilmExposure {
        input_color_space: match input.color().space() {
            color::InputSpace::DavinciWideGamut => 0,
            color::InputSpace::Bt2020 => 1,
            color::InputSpace::Aces2065_1 => 2,
            color::InputSpace::SrgbRec709 => 3,
        },
        method: match input.method() {
            film::input::Method::Hanatos2025 => 0,
            film::input::Method::Mallett2019 => 1,
            film::input::Method::Arctic2026Beta04 => 2,
        },
        flags: (if input.color().decode_cctf() {
            sys::FJ_FILM_DECODE_CCTF
        } else {
            0
        }) | sys::FJ_FILM_ADAPT_INPUT
            | (if input.auto_exposure_enabled() {
                sys::FJ_FILM_AUTO_EXPOSURE
            } else {
                0
            }),
        manual_exposure_ev: input.manual_exposure_ev(),
        route_correction_scale: if matches!(state.route(), recipe::PreparedRoute::Direct(_)) {
            correction.exposure_scale()
        } else {
            1.0
        },
        mallett_green_midgray_scale: input.sensitivity().mallett_green_scale(),
        input_rgb_to_xyz: *input.color().rgb_to_xyz(),
        input_xyz_adapt: *input.color().xyz_adapt(),
        xyz_to_linear_srgb: *input.color().xyz_to_linear_srgb(),
        sensitivity_rgb: floats(input.sensitivity().values_rgb().as_flattened()),
        tc_lut_rgba: state
            .reconstruction()
            .tc_lut()
            .map_or_else(|| floats(&[]), |lut| floats(lut.samples())),
        mallett_illuminant: if state.reconstruction().mallett_basis().is_some() {
            floats(state.exposure_tables().illuminant())
        } else {
            floats(&[])
        },
        mallett_basis: state
            .reconstruction()
            .mallett_basis()
            .map_or_else(|| floats(&[]), |basis| floats(basis.as_flattened())),
        sensitivity_hash: input.sensitivity().hash(),
        tc_lut_hash: input.tc_hash(),
        hash: input.hash(),
    };
    out.film_development = sys::FjFilmDevelopment {
        log_exposure: floats(development.log_exposure()),
        density_rgb: floats(development.normalized_curves_cmy().as_flattened()),
        density_layers: std::array::from_fn(|layer| {
            std::array::from_fn(|channel| {
                floats(
                    development
                        .density_curves_layers()
                        .map_or(&[], |layers| &layers[layer][channel]),
                )
            })
        }),
        gamma_rgb: *development.gamma_rgb(),
        density_curves_hash: development.normalized_curves_hash(),
        density_layers_hash: development.layers_hash(),
        hash: development.hash(),
    };
    let dir = state.dir();
    out.dir_recipe = FjFilmDirView {
        polarity: dir.polarity().map_or(2, polarity_tag),
        mode: match dir.mode() {
            film::dir::Mode::Inactive => 0,
            film::dir::Mode::NegativeDonorLangmuir => 1,
            film::dir::Mode::PositiveReceiverLangmuir => 2,
        },
        matrix_rgb: std::array::from_fn(|i| dir.matrix_rgb()[i / 3][i % 3]),
        diffusion_size_um: dir.diffusion_size_um(),
        diffusion_tail_um: dir.diffusion_tail_um(),
        diffusion_tail_weight: dir.diffusion_tail_weight(),
        density_max_rgb: *dir.density_max_rgb(),
        density_ref_rgb: *dir.density_ref_rgb(),
        donor_k_rgb: *dir.donor_k_rgb(),
        receiver_c_ref_rgb: *dir.receiver_c_ref_rgb(),
        receiver_kr_rgb: *dir.receiver_kr_rgb(),
        compensated_axes_rgb: std::array::from_fn(|c| floats(&dir.compensated_axes_rgb()[c])),
        compensated_axes_hash: dir.compensated_axes_hash(),
        hash: dir.hash(),
    };
    out.grain_minimum = FjFilmDensityMinimum {
        cmy: *state.minimum().cmy(),
    };
    out.scanner_bounds = film_bounds(state.film_bounds());
    let reference = state.reference();
    out.film_reference = FjFilmReference {
        baseline_raw_rgb: *reference.baseline_raw_rgb(),
        compensated_raw_rgb: *reference.compensated_raw_rgb(),
        baseline_density_cmy: *reference.baseline_density_cmy(),
        compensated_density_cmy: *reference.compensated_density_cmy(),
        meter_scale: reference.meter_scale(),
        hash: reference.hash(),
    };
    out.optics = optics_projection(state.optics());
    out.grain = grain_projection(state.grain());
    let effects = state.effects();
    out.effects = FjPhysicalEffectsRecipe {
        film_dust: dust_projection(effects.film_dust()),
        film_scratch: scratch_projection(effects.film_scratch()),
        gate_dust: dust_projection(effects.gate_dust()),
        gate_scratch: scratch_projection(effects.gate_scratch()),
        gate_weave_amount: effects.gate_weave_amount(),
        active: u32::from(effects.active()),
        hash: effects.hash(),
    };
    out.exposure_tables = spectral_projection(state.exposure_tables());
    out.scanner_tables = spectral_projection(scanner.tables());
    out.s_inverse = *state.s_inverse();
    out.midgray_dwg_rgb = *state.reconstruction().midgray_dwg_rgb();
    if let Some(rgb) = state.reconstruction().raw_midgray_rgb() {
        out.raw_midgray = *rgb;
    }
    if let Some(bgr) = state.reconstruction().raw_midgray_bgr() {
        out.raw_midgray = *bgr;
        out.raw_midgray_bgr = 1;
    }
    out.raw_green = state.reconstruction().normalization().raw_green();
    out.midgray_scale = state.reconstruction().normalization().scale();
    if let Some(hull) = state.input_hull() {
        out.input_hull = hull_projection(hull);
    }
    if let Some(boundary) = scanner.gamut().boundary() {
        out.output_boundary = boundary_projection(boundary);
    }
    let lut = scanner.lut();
    out.scanner_lut = FjScannerLutDescriptor {
        route: route_tag(lut.route()),
        medium: u32::from(lut.medium()),
        capture_polarity: u32::from(lut.capture_polarity()),
        resolution: lut.resolution(),
        density_bounds_hash: lut.density_bounds_hash(),
        channel_density_hash: lut.channel_density_hash(),
        base_density_hash: lut.base_density_hash(),
        viewing_illuminant_hash: lut.viewing_illuminant_hash(),
        observer_hash: lut.observer_hash(),
        hash: lut.hash(),
    };
    out.scanner_color = FjScannerColor {
        cat02: *color.cat02(),
        xyz_to_rgb: *color.xyz_to_rgb(),
        illuminant_xyz: *color.illuminant_xyz(),
        output_space: u32::from(color.output_space()),
        flags: sys::FJ_COLOR_INPUT_IS_OUTPUT_SPACE | u32::from(color.apply_cctf()),
        encoding: output_encoding(color),
        active_gamut_recipe_hash: color.active_gamut_recipe_hash(),
        hash: color.hash(),
    };
    out.scanner_correction = correction_projection(correction);
    let post = scanner.post_effects();
    let fields = post.fields();
    let values = fields.values();
    let radii = fields.radii();
    out.scanner_post_effects = FjScannerPostEffectsDescriptor {
        route: route_tag(fields.route()),
        glare_active: u32::from(fields.glare_active()),
        glare_percent: values[0],
        glare_roughness: values[1],
        glare_blur_sigma_px: values[2],
        glare_blur_radius: radii[0],
        lens_blur_sigma_px: values[3],
        lens_blur_radius: radii[1],
        unsharp_sigma_px: values[4],
        unsharp_radius: radii[2],
        unsharp_amount: values[5],
        schema: fields.schema(),
        hash: post.hash(),
    };
    let gamut = scanner.gamut();
    out.output_color = sys::FjOutputColor {
        cat02: *color.cat02(),
        xyz_to_rgb: *color.xyz_to_rgb(),
        illuminant_xyz: *color.illuminant_xyz(),
        color_space: u32::from(color.output_space()),
        flags: sys::FJ_COLOR_INPUT_IS_OUTPUT_SPACE
            | (if color.apply_cctf() {
                sys::FJ_COLOR_ENCODE_CCTF
            } else {
                0
            })
            | (if gamut.enabled() {
                sys::FJ_COLOR_GAMUT_COMPRESSION
            } else {
                0
            }),
        native_rgb_to_d65_xyz: *gamut.transform().native_rgb_to_d65_xyz(),
        d65_xyz_to_native_rgb: *gamut.transform().d65_xyz_to_native_rgb(),
        lightness_knee: *gamut.lightness_knee(),
        chroma_knee: *gamut.chroma_knee(),
        gamut_cmax: gamut
            .boundary()
            .map_or_else(|| floats(&[]), |boundary| floats(boundary.cmax())),
        gamut_table_hash: gamut.boundary().map_or(0, |boundary| boundary.hash()),
        gamut_recipe_hash: gamut.hash(),
        encoding: output_encoding(color),
    };
    out.scanner_recipe_hash = scanner.hash();
    if let recipe::PreparedRoute::Print(print) = state.route() {
        let profile = print.profile();
        let exposure = print.exposure().params();
        let descriptors = print.resource_descriptors();
        out.print_profile_key = text(print.key());
        out.print_profile_asset_version = profile.asset_token();
        out.enlarger_film_bounds = film_bounds(state.film_bounds());
        out.scanner_bounds = FjFilmDensityBounds {
            route: out.route,
            medium: 1,
            polarity: out.polarity,
            source: 2,
            minimum_cmy: *print.bounds().minimum_cmy(),
            maximum_cmy: *print.bounds().maximum_cmy(),
            inv_span_cmy: *print.bounds().inv_span_cmy(),
            authored_minimum_cmy: *print.bounds().minimum_cmy(),
            authored_maximum_cmy: *print.bounds().maximum_cmy(),
            hash: print.bounds().hash(),
        };
        out.print = sys::FjPrint {
            film_density_cmy: floats(film.tables().channel_density().as_flattened()),
            film_base_density: floats(film.tables().base_density()),
            sensitivity_cmy: floats(profile.tables().linear_sensitivity().as_flattened()),
            log_exposure: floats(profile.tables().interpolation_log_exposure()),
            density_cmy: floats(print.curves().totals().as_flattened()),
            main_illuminant: floats(print.main_illuminant().samples()),
            preflash_illuminant: print.preflash().map_or_else(
                || floats(&[]),
                |preflash| floats(preflash.illuminant().samples()),
            ),
            flags: if print.preflash().is_some() {
                sys::FJ_PRINT_PREFLASH
            } else {
                0
            },
            main_cc_cmy: *print.filters().main_cmy_cc(),
            preflash_cc_cmy: if print.preflash().is_some() {
                *print.filters().preflash_cmy_cc()
            } else {
                [0.0; 3]
            },
            exposure: exposure.print_exposure,
            preflash_exposure: exposure.preflash_exposure,
            preflash_raw_cmy: print
                .preflash()
                .map_or([0.0; 3], |preflash| *preflash.raw().cmy()),
            normalizer: print.balance().normalizer(),
            film_density_hash: descriptors.film_density_hash(),
            profile_tables_hash: descriptors.profile_tables_hash(),
            main_illuminant_hash: descriptors.main_illuminant_hash(),
            preflash_illuminant_hash: descriptors.preflash_illuminant_hash(),
            preflash_raw_hash: descriptors.preflash_raw_hash(),
            balance_hash: descriptors.balance_hash(),
            hash: descriptors.hash(),
        };
        out.print_gamma = print.gamma();
        out.print_filter_main_cmy_cc = *print.filters().main_cmy_cc();
        out.print_filter_preflash_cmy_cc = *print.filters().preflash_cmy_cc();
        out.print_normalization_mode = u32::from(print.exposure().normalization_mode());
        out.normalize_print_exposure = u32::from(exposure.normalize);
        out.print_exposure_compensation = u32::from(exposure.compensate);
        out.neutral_cmy_cc = *print.neutral_cmy_cc();
        match print.calibration() {
            recipe::Calibration::MissingFile => out.calibration = 0,
            recipe::Calibration::MissingEntry => out.calibration = 1,
            recipe::Calibration::Calibrated(_) => out.calibration = 2,
        }
        out.print_source_illuminant = floats(print.source_illuminant());
        let balance = print.balance();
        out.print_balance = FjPrintBalance {
            baseline_raw_rgb: *balance.baseline_raw_rgb(),
            compensated_raw_rgb: *balance.compensated_raw_rgb(),
            factor_midgray: balance.factor_midgray(),
            factor_midgray_comp: balance.factor_midgray_comp(),
            normalizer: balance.normalizer(),
            filtered_main_illuminant_hash: balance.filtered_main_illuminant_hash(),
            hash: balance.hash(),
        };
        out.print_recipe_hash = print.hash();
        out.print_filters_hash = print.filters().hash();
        out.print_exposure_hash = print.exposure().hash();
        out.print_illuminant_hash = print.illuminant_hash();
        out.print_curves_hash = print.curves().hash();
    }
    out
}

/// # Safety
/// A live retained state bounds every projected span. Output and optional
/// diagnostic storage remain exclusive and disjoint through synchronous return.
pub(crate) unsafe fn state_view(
    owner: *const FjPreparedState,
    out: *mut FjPreparedStateView,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The complete zero-valid raw view borrows only this live state.
    unsafe { view(owner, out, error, project_state) }
}

fn aligned<T>(pointer: *const T) -> bool {
    !pointer.is_null()
        && (pointer as usize).is_multiple_of(align_of::<T>())
        && (pointer as usize).checked_add(size_of::<T>()).is_some()
}
fn status(category: u32) -> FjStatus {
    FjStatus {
        category,
        api: FJ_API_NONE,
        native_code: 0,
    }
}
fn floats(values: &[f32]) -> FjFloatSpan {
    FjFloatSpan {
        data: if values.is_empty() {
            std::ptr::null()
        } else {
            values.as_ptr()
        },
        count: values.len(),
    }
}

/// # Safety
/// Optional initialized diagnostic records and any nonempty backing are
/// exclusive and disjoint from the operation's authorized foreign storage.
unsafe fn run<T>(
    error: *mut FjErrorBuffer,
    operation: impl FnOnce() -> Result<T, Failure>,
) -> Result<T, FjStatus> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if !error.is_null() {
            if !aligned(error) {
                drop(operation);
                return Err(status(FJ_STATUS_UNSUPPORTED_INPUT));
            }
            // SAFETY: The caller authorizes this initialized aligned record;
            // zero capacity deliberately leaves its data field unread.
            let capacity = unsafe { (*error).capacity };
            if capacity != 0 {
                // SAFETY: Positive capacity authorizes the initialized data field.
                let data = unsafe { (*error).data };
                if data.is_null()
                    || capacity > isize::MAX as usize
                    || (data as usize).checked_add(capacity).is_none()
                {
                    // SAFETY: Only the authorized record length is cleared for
                    // malformed backing; that backing is never accessed.
                    unsafe { (*error).length = 0 };
                    drop(operation);
                    return Err(status(FJ_STATUS_UNSUPPORTED_INPUT));
                }
            }
        }
        // SAFETY: Validated diagnostic extent and original caller disjointness
        // permit the existing panic-contained writer to execute this operation.
        unsafe { asset_bridge::run(error, operation) }
    }))
    .unwrap_or(Err(status(FJ_STATUS_INTERNAL_FAILURE)))
}

/// # Safety
/// A matching live Assets allocation remains borrowed for the caller's scope;
/// the caller excludes destruction. Alignment does not establish pointer validity.
unsafe fn assets<'a>(pointer: *const FjAssets) -> Result<&'a Assets, Failure> {
    let pointer = pointer.cast::<Assets>();
    if !aligned(pointer) {
        return Err(Failure::Input("invalid preparation Assets owner"));
    }
    // SAFETY: The caller authorizes the complete initialized live Assets allocation.
    Ok(unsafe { &*pointer })
}

/// # Safety
/// Both live Assets owners are borrowed synchronously; the destination is
/// quiescent except for retained immutable views of its older snapshot.
#[cfg(feature = "test-support")]
pub(crate) unsafe fn calibration_source(
    destination: *const FjAssets,
    source: *const FjAssets,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The caller authorizes both matching owner borrows through return.
    unsafe {
        run(error, || {
            assets(destination)?
                .replace_calibration_source_for_test(assets(source)?)
                .map_err(Failure::Asset)
        })
    }
    .map_or_else(std::convert::identity, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// The matching Assets and exclusive diagnostics obey the temporary test edge.
#[cfg(feature = "test-support")]
pub(crate) unsafe fn hull_operation_fault(
    owner: *const FjAssets,
    fault: u32,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Shared owner and checked diagnostics stay live through return.
    unsafe {
        run(error, || {
            use film_juicer_core::{assets::testing::HullFault, gamut::HullError};
            let fault = match fault {
                0 => None,
                1 => Some(HullFault::Unsupported),
                2 => Some(HullFault::Panic),
                3 => Some(HullFault::Preparation(HullError::LocusAndCenter)),
                4 => Some(HullFault::Preparation(HullError::RayIntersections)),
                5 => Some(HullFault::Preparation(HullError::InscribedScale)),
                6 => Some(HullFault::Preparation(HullError::Identity)),
                _ => return Err(Failure::Input("unsupported input hull fault")),
            };
            assets(owner)?.arm_hull_fault(fault);
            Ok(())
        })
    }
    .map_or_else(std::convert::identity, |()| status(FJ_STATUS_SUCCESS))
}
/// # Safety
/// Matching shared owner and exclusive initialized scalar/diagnostics obey cache_read.
#[cfg(feature = "test-support")]
pub(crate) unsafe fn hull_operation_pending(
    owner: *const FjAssets,
    out: *mut u32,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Shared read clears the admitted exclusive output before fallible work.
    unsafe {
        cache_read(owner, out, error, |owner| {
            Ok(u32::from(owner.hull_fault_pending()))
        })
    }
}

/// # Safety
/// The matching Assets owner remains live; diagnostics obey the held-state contract.
#[cfg(feature = "test-support")]
pub(crate) unsafe fn illuminant_operation_fault(
    owner: *const FjAssets,
    operation: u32,
    fault: u32,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Shared Assets and exclusively writable diagnostic per the test edge.
    unsafe {
        run(error, || {
            use film_juicer_core::assets::testing::{IlluminantFault, IlluminantOperation};
            let operation = match operation {
                1 => IlluminantOperation::Samples,
                4 => IlluminantOperation::TungstenKg3,
                5 => IlluminantOperation::LensPrepare,
                6 => IlluminantOperation::LensFinish,
                _ => {
                    return Err(Failure::Input(
                        "unsupported cache-owned illuminant operation",
                    ));
                }
            };
            let fault = match fault {
                0 => None,
                1 => Some(IlluminantFault::Unsupported),
                2 => Some(IlluminantFault::Panic),
                3 => Some(IlluminantFault::Capacity),
                _ => return Err(Failure::Input("unsupported cache-owned illuminant fault")),
            };
            assets(owner)?.arm_illuminant_operation(operation, fault);
            Ok(())
        })
    }
    .map_or_else(std::convert::identity, |()| status(FJ_STATUS_SUCCESS))
}
/// # Safety
/// Matching shared owner and exclusive scalar output/diagnostic obey cache_read.
#[cfg(feature = "test-support")]
pub(crate) unsafe fn illuminant_operation_consumed(
    owner: *const FjAssets,
    out: *mut u32,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The shared owner and exclusive scalar output follow the test edge.
    unsafe {
        cache_read(owner, out, error, |owner| {
            Ok(owner.illuminant_operation_fault_consumed())
        })
    }
}

/// # Safety
/// Shared matching Assets and exclusive diagnostics obey the temporary edge.
#[cfg(feature = "test-support")]
pub(crate) unsafe fn curve_fault(
    owner: *const FjAssets,
    source: u32,
    fault: u32,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The caller authorizes the checked matching owner borrow.
    unsafe {
        run(error, || {
            let source = crate::asset_bridge::csv_source(source)?;
            if matches!(
                source,
                film_juicer_core::assets::CsvSource::Kg3
                    | film_juicer_core::assets::CsvSource::Canon24F28Is
            ) {
                return Err(Failure::Input(
                    "unsupported derived-curve observation source",
                ));
            }
            let fault = match fault {
                0 => None,
                1 => Some(film_juicer_core::assets::CurveFault::Capacity),
                2 => Some(film_juicer_core::assets::CurveFault::Panic),
                3 => Some(film_juicer_core::assets::CurveFault::RowIntegrity),
                _ => return Err(Failure::Input("unsupported derived-curve fault")),
            };
            assets(owner)?
                .arm_illuminant_curve_fault(source, fault)
                .map_err(Failure::Asset)
        })
    }
    .map_or_else(std::convert::identity, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// The matching live owner and exclusive initialized output/error records are disjoint.
#[cfg(feature = "test-support")]
pub(crate) unsafe fn curve_fault_pending(
    owner: *const FjAssets,
    out: *mut u32,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    if !aligned(out) {
        return status(FJ_STATUS_UNSUPPORTED_INPUT);
    }
    // SAFETY: The caller authorizes this admitted exclusive scalar slot.
    unsafe {
        out.write(0);
    }
    // SAFETY: The matching owner and scalar output remain live and disjoint.
    unsafe {
        run(error, || {
            let pending = assets(owner)?
                .illuminant_curve_fault_pending()
                .map_err(Failure::Asset)?;
            out.write(u32::from(pending));
            Ok(())
        })
    }
    .map_or_else(std::convert::identity, |()| status(FJ_STATUS_SUCCESS))
}

#[cfg(feature = "test-support")]
#[repr(C)]
#[derive(Default)]
pub(crate) struct FjCacheMetrics {
    allocations: u64,
    derivations: u64,
    destructions: u64,
    live_bytes: u64,
    peak_bytes: u64,
    reentries: u64,
    reentry_failures: u64,
}
#[cfg(feature = "test-support")]
impl From<film_juicer_core::assets::testing::CacheMetrics> for FjCacheMetrics {
    fn from(m: film_juicer_core::assets::testing::CacheMetrics) -> Self {
        Self {
            allocations: m.allocations,
            derivations: m.derivations,
            destructions: m.destructions,
            live_bytes: m.live_bytes,
            peak_bytes: m.peak_bytes,
            reentries: m.reentries,
            reentry_failures: m.reentry_failures,
        }
    }
}

#[cfg(feature = "test-support")]
pub(crate) struct FjCacheObservationOwner {
    observation: film_juicer_core::assets::testing::CacheObservation,
}

/// # Safety
/// Matching Assets/output/error records obey acquire's consuming-hold contract.
#[cfg(feature = "test-support")]
pub(crate) unsafe fn cache_observation_acquire(
    owner: *const FjAssets,
    out: *mut *mut FjCacheObservationOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Shared acquire clears/admit output before borrowing the owner.
    unsafe {
        acquire(out, error, || {
            Ok(FjCacheObservationOwner {
                observation: assets(owner)?.cache_observation(),
            })
        })
    }
}
/// # Safety
/// The live observation bounds the read; the output/error records are exclusive.
#[cfg(feature = "test-support")]
pub(crate) unsafe fn cache_observation_metrics(
    owner: *const FjCacheObservationOwner,
    family: u32,
    out: *mut FjCacheMetrics,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The checked view clears output before admitting either input.
    unsafe {
        view_result(owner, out, error, |owner| {
            Ok(owner.observation.metrics(cache_family(family)?).into())
        })
    }
}
/// # Safety
/// Same observation/output contract as cache_observation_metrics. entry is only
/// an identity word and is never dereferenced.
#[cfg(feature = "test-support")]
pub(crate) unsafe fn cache_observation_entry_live(
    owner: *const FjCacheObservationOwner,
    family: u32,
    entry: usize,
    out: *mut u32,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The checked view clears output before admitting either input.
    unsafe {
        view_result(owner, out, error, |owner| {
            Ok(u32::from(
                owner.observation.entry_live(cache_family(family)?, entry),
            ))
        })
    }
}
/// # Safety
/// Consume one matching observation after excluding concurrent reads.
#[cfg(feature = "test-support")]
pub(crate) unsafe fn cache_observation_release(
    owner: *mut FjCacheObservationOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Shared release consumes the matching Box, including invalid diagnostics.
    unsafe { release(owner, error) }
}

/// # Safety
/// Matching live Assets and optional diagnostics obey the cache contract.
#[cfg(feature = "test-support")]
pub(crate) unsafe fn hull_entry_fault(
    owner: *const FjAssets,
    active: u32,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Checked matching owner and diagnostics are borrowed through return.
    unsafe {
        run(error, || {
            if active > 1 {
                return Err(Failure::Input("invalid hull entry fault"));
            }
            assets(owner)?.arm_hull_entry_reservation(active != 0);
            Ok(())
        })
    }
    .map_or_else(std::convert::identity, |()| status(FJ_STATUS_SUCCESS))
}
/// # Safety
/// Matching owner and exclusive output/error records obey cache_read's contract.
#[cfg(feature = "test-support")]
pub(crate) unsafe fn hull_entry_fault_consumed(
    owner: *const FjAssets,
    out: *mut u32,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Shared reader clears and admits this exclusive scalar slot.
    unsafe {
        cache_read(owner, out, error, |owner| {
            Ok(u32::from(owner.take_hull_entry_reservation_consumed()))
        })
    }
}
/// # Safety
/// Matching live Assets and optional diagnostics obey the cache contract. Only
/// the cold operation's local CMF copy is varied; no published source is mutated.
#[cfg(feature = "test-support")]
pub(crate) unsafe fn hull_source_tail(
    owner: *const FjAssets,
    active: u32,
    add: f32,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Checked matching owner and diagnostics are borrowed through return.
    unsafe {
        run(error, || {
            if active > 1 {
                return Err(Failure::Input("invalid hull source observation"));
            }
            assets(owner)?.set_hull_source_tail_add((active != 0).then_some(add));
            Ok(())
        })
    }
    .map_or_else(std::convert::identity, |()| status(FJ_STATUS_SUCCESS))
}

#[cfg(feature = "test-support")]
fn cache_family(value: u32) -> Result<film_juicer_core::assets::testing::CacheFamily, Failure> {
    use film_juicer_core::assets::testing::CacheFamily;
    match value {
        0 => Ok(CacheFamily::Illuminants),
        1 => Ok(CacheFamily::InputHull),
        2 => Ok(CacheFamily::OutputBoundary),
        _ => Err(Failure::Input("invalid cache observation family")),
    }
}

/// # Safety
/// The matching owner, exclusive output, and optional error obey the same
/// initialization/alignment/disjointness contract as the family view operations.
#[cfg(feature = "test-support")]
unsafe fn cache_read<T: Default>(
    owner: *const FjAssets,
    out: *mut T,
    error: *mut FjErrorBuffer,
    read: impl FnOnce(&Assets) -> Result<T, Failure>,
) -> FjStatus {
    if !aligned(out) {
        return status(FJ_STATUS_UNSUPPORTED_INPUT);
    }
    // SAFETY: An admitted exclusive initialized output slot is authorized.
    unsafe {
        out.write(T::default());
    }
    // SAFETY: Checked matching owner and diagnostic/output storage live through return.
    unsafe {
        run(error, || {
            out.write(read(assets(owner)?)?);
            Ok(())
        })
    }
    .map_or_else(std::convert::identity, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// Matching owner/output/error records follow cache_read's contract.
#[cfg(feature = "test-support")]
pub(crate) unsafe fn cache_metrics(
    owner: *const FjAssets,
    family: u32,
    out: *mut FjCacheMetrics,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The shared reader admits and clears the exclusive output.
    unsafe {
        cache_read(owner, out, error, |owner| {
            Ok(owner.cache_metrics(cache_family(family)?).into())
        })
    }
}
/// # Safety
/// Matching owner/output/error records follow cache_read's contract.
#[cfg(feature = "test-support")]
pub(crate) unsafe fn output_cache_generation(
    owner: *const FjAssets,
    out: *mut u64,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The shared reader admits and clears the exclusive scalar output.
    unsafe {
        cache_read(owner, out, error, |owner| {
            owner.output_cache_generation().map_err(Failure::Asset)
        })
    }
}
/// # Safety
/// Matching owner/output/error records follow cache_read's contract.
#[cfg(feature = "test-support")]
pub(crate) unsafe fn cache_fault_consumed(
    owner: *const FjAssets,
    family: u32,
    out: *mut u64,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The shared reader admits and clears the exclusive scalar output.
    unsafe {
        cache_read(owner, out, error, |owner| {
            Ok(owner.take_cache_fault_consumed(cache_family(family)?))
        })
    }
}
/// # Safety
/// Matching owner/output/error records follow cache_read's contract. entry is an
/// observation word only; no pointed-to storage is dereferenced by this operation.
#[cfg(feature = "test-support")]
pub(crate) unsafe fn cache_entry_live(
    owner: *const FjAssets,
    family: u32,
    entry: usize,
    out: *mut u32,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The shared reader admits and clears the exclusive scalar output.
    unsafe {
        cache_read(owner, out, error, |owner| {
            Ok(u32::from(
                owner.cache_entry_live(cache_family(family)?, entry),
            ))
        })
    }
}
/// # Safety
/// The matching owner is live; callers exclude concurrent cache operations when
/// setting the actual generation. Diagnostics obey the family contract.
#[cfg(feature = "test-support")]
pub(crate) unsafe fn output_cache_set_generation(
    owner: *const FjAssets,
    generation: u64,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Checked owner borrow and diagnostics remain live through return.
    unsafe {
        run(error, || {
            assets(owner)?
                .set_output_cache_generation(generation)
                .map_err(Failure::Asset)
        })
    }
    .map_or_else(std::convert::identity, |()| status(FJ_STATUS_SUCCESS))
}
/// # Safety
/// The matching owner is live through return; diagnostics obey the family contract.
#[cfg(feature = "test-support")]
pub(crate) unsafe fn cache_fault(
    owner: *const FjAssets,
    family: u32,
    fault: u32,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Checked owner borrow and diagnostics remain live through return.
    unsafe {
        run(error, || {
            use film_juicer_core::assets::testing::CacheFault;
            let fault = match fault {
                0 => None,
                1 => Some(CacheFault::Unsupported),
                2 => Some(CacheFault::Panic),
                3 => Some(CacheFault::Capacity),
                4 | 5 => Some(CacheFault::Preparation),
                _ => return Err(Failure::Input("invalid cache observation fault")),
            };
            assets(owner)?.arm_cache_fault(cache_family(family)?, fault);
            Ok(())
        })
    }
    .map_or_else(std::convert::identity, |()| status(FJ_STATUS_SUCCESS))
}
/// # Safety
/// The matching owner is live through return; diagnostics obey the family contract.
#[cfg(feature = "test-support")]
pub(crate) unsafe fn cache_clear_failures(
    owner: *const FjAssets,
    illuminants: u32,
    late: u32,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Checked owner borrow and diagnostics remain live through return.
    unsafe {
        run(error, || {
            if illuminants > 1 || late > 1 {
                return Err(Failure::Input("invalid cache cleanup fault"));
            }
            assets(owner)?.arm_cache_clear_failures(illuminants != 0, late != 0);
            Ok(())
        })
    }
    .map_or_else(std::convert::identity, |()| status(FJ_STATUS_SUCCESS))
}
/// # Safety
/// The matching owner is live through return; diagnostics obey the family contract.
#[cfg(feature = "test-support")]
pub(crate) unsafe fn cache_drop_reentry(
    owner: *const FjAssets,
    family: u32,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Checked owner borrow and diagnostics remain live through return.
    unsafe {
        run(error, || {
            assets(owner)?.arm_cache_drop_reentry(cache_family(family)?);
            Ok(())
        })
    }
    .map_or_else(std::convert::identity, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// The output authorizes one exclusive aligned slot, disjoint from diagnostic
/// and all operation inputs. Captured foreign borrows last until return.
unsafe fn acquire<T>(
    out: *mut *mut T,
    error: *mut FjErrorBuffer,
    operation: impl FnOnce() -> Result<T, Failure>,
) -> FjStatus {
    let valid = aligned(out);
    if valid {
        // SAFETY: The caller authorizes this exclusive output slot.
        unsafe { out.write(std::ptr::null_mut()) };
    }
    // SAFETY: The foreign diagnostic and output obey the documented disjoint contract.
    unsafe {
        run(error, || {
            if !valid {
                return Err(Failure::Input("invalid preparation owner output"));
            }
            let owner = Box::new(operation()?);
            // SAFETY: Success transfers exactly this new Box into the cleared slot.
            out.write(Box::into_raw(owner));
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// owner is NULL or one uniquely transferred matching Box. No view borrow or
/// operation overlaps release. Diagnostics follow the shared foreign contract.
unsafe fn release<T>(owner: *mut T, error: *mut FjErrorBuffer) -> FjStatus {
    if !owner.is_null() && !aligned(owner) {
        return status(FJ_STATUS_UNSUPPORTED_INPUT);
    }
    let owner = if owner.is_null() {
        None
    } else {
        // SAFETY: The caller transfers this live matching allocation once.
        Some(unsafe { Box::from_raw(owner) })
    };
    // SAFETY: Capturing the consumed owner drops it inside containment even when
    // diagnostic admission rejects the operation without invoking its body.
    unsafe {
        run(error, move || {
            drop(owner);
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// R is a zero-valid raw C record. out authorizes its complete exclusive extent;
/// owner is a matching live immutable allocation retained through all uses of R.
/// Its projected pointer fields borrow owner and never confer detached ownership.
unsafe fn view<T, R>(
    owner: *const T,
    out: *mut R,
    error: *mut FjErrorBuffer,
    project: impl FnOnce(&T) -> R,
) -> FjStatus {
    // SAFETY: The same initialized/disjoint owner and output borrows are forwarded.
    unsafe { view_result(owner, out, error, |owner| Ok(project(owner))) }
}

/// # Safety
/// Same initialized owner and zero-valid exclusive output contract as view.
unsafe fn view_result<T, R>(
    owner: *const T,
    out: *mut R,
    error: *mut FjErrorBuffer,
    project: impl FnOnce(&T) -> Result<R, Failure>,
) -> FjStatus {
    let valid = aligned(out);
    if valid {
        // SAFETY: Every call instantiates R with a numeric/span-only zero-valid
        // C record, and the caller authorizes its exclusive writable extent.
        unsafe { out.write(std::mem::zeroed()) };
    }
    // SAFETY: The foreign records/diagnostic have the documented disjoint extent.
    let result = unsafe {
        run(error, || {
            if !valid || !aligned(owner) {
                return Err(Failure::Input("invalid preparation view records"));
            }
            // SAFETY: The caller retains the matching initialized owner. The result
            // borrows only that allocation or its own retained immutable subowners.
            out.write(project(&*owner)?);
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS));
    if result.category != FJ_STATUS_SUCCESS && valid {
        // SAFETY: Clearing restores every defined field even after a contained
        // projection failure; R is the same admitted zero-valid foreign record.
        unsafe { out.write(std::mem::zeroed()) };
    }
    result
}

fn illuminants_projection(owner: &FjIlluminantsOwner) -> FjIlluminantsView {
    let curve =
        |values: Option<&[f32; 81]>| values.map_or_else(|| floats(&[]), |values| floats(values));
    FjIlluminantsView {
        d65: curve(owner.entry.d65()),
        d55: curve(owner.entry.d55()),
        d50: curve(owner.entry.d50()),
        tungsten: curve(owner.entry.tungsten()),
        k75p: curve(owner.entry.k75p()),
        tungsten_kg3: curve(owner.entry.tungsten_kg3()),
        tungsten_kg3_lens: curve(owner.entry.tungsten_kg3_lens()),
        entry_identity: Arc::as_ptr(&owner.entry) as usize,
    }
}
fn hull_projection(entry: &InputHull) -> FjInputHullView {
    FjInputHullView {
        center_xy: *entry.hull().center(),
        xy: floats(entry.hull().xy().as_flattened()),
        hash: entry.hull().hash(),
        entry_identity: entry.hull() as *const gamut::Hull as usize,
    }
}
fn boundary_projection(entry: &gamut::OutputBoundary) -> FjOutputBoundaryView {
    FjOutputBoundaryView {
        cmax: floats(entry.cmax()),
        transform_hash: entry.transform_hash(),
        contract_hash: entry.contract_hash(),
        hash: entry.hash(),
        entry_identity: entry as *const gamut::OutputBoundary as usize,
    }
}

/// # Safety
/// Live borrowed Assets, exclusive owner output and disjoint diagnostic storage
/// remain valid through this synchronous operation.
pub(crate) unsafe fn illuminants_acquire(
    input: *const FjAssets,
    out: *mut *mut FjIlluminantsOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Forward the same admitted storage to the shared operation.
    unsafe {
        illuminants_acquire_with(
            input,
            out,
            error,
            #[cfg(feature = "test-support")]
            None,
        )
    }
}

#[cfg(feature = "test-support")]
#[repr(C)]
pub(crate) struct FjIlluminantCacheHooks {
    before_source: Option<unsafe extern "C" fn(u32, u32) -> u32>,
    before_publication: Option<unsafe extern "C" fn(usize, usize) -> u32>,
}
#[cfg(feature = "test-support")]
struct IlluminantObserver<'a>(&'a FjIlluminantCacheHooks);
#[cfg(feature = "test-support")]
fn illuminant_observer_status(
    category: u32,
    label: &'static str,
) -> Result<(), film_juicer_core::assets::DerivedError> {
    if category == FJ_STATUS_ALLOCATION_FAILURE {
        return Err(film_juicer_core::assets::DerivedError::Illuminant {
            label,
            error: film_juicer_core::illuminant::test_support::capacity_failure()
                .expect_err("unaddressable scratch reservation must fail"),
        });
    }
    observer_status(category, "illuminant cache observation failed")
}
#[cfg(feature = "test-support")]
impl film_juicer_core::assets::testing::IlluminantObserver for IlluminantObserver<'_> {
    fn before_source(
        &self,
        source: film_juicer_core::assets::CsvSource,
        lens_prepared: bool,
    ) -> Result<(), film_juicer_core::assets::DerivedError> {
        let result = if let Some(callback) = self.0.before_source {
            // SAFETY: The synchronous test callback contains every exception and
            // retains neither the callback table nor an owner/source borrow.
            unsafe {
                callback(
                    crate::asset_bridge::csv_source_tag(source),
                    u32::from(lens_prepared),
                )
            }
        } else {
            0
        };
        illuminant_observer_status(result, "source acquisition")
    }
    fn before_publication(
        &self,
        entry: usize,
        bytes: usize,
    ) -> Result<(), film_juicer_core::assets::DerivedError> {
        let result = if let Some(callback) = self.0.before_publication {
            // SAFETY: The same synchronous, exception-containing C callback contract.
            unsafe { callback(entry, bytes) }
        } else {
            0
        };
        illuminant_observer_status(result, "candidate publication")
    }
}

/// # Safety
/// The callback table is immutable/live until synchronous acquire returns. Its
/// callbacks contain all native exceptions and run outside owner cache locks.
#[cfg(feature = "test-support")]
pub(crate) unsafe fn illuminants_acquire_observed(
    input: *const FjAssets,
    hooks: *const FjIlluminantCacheHooks,
    out: *mut *mut FjIlluminantsOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The shared operation clears output before callback-table admission.
    unsafe { illuminants_acquire_with(input, out, error, Some(hooks)) }
}

/// # Safety
/// All inputs and exclusive outputs obey illuminants_acquire's foreign contract.
unsafe fn illuminants_acquire_with(
    input: *const FjAssets,
    out: *mut *mut FjIlluminantsOwner,
    error: *mut FjErrorBuffer,
    #[cfg(feature = "test-support")] hooks: Option<*const FjIlluminantCacheHooks>,
) -> FjStatus {
    // SAFETY: acquire admits the output before borrowing the authorized Assets.
    unsafe {
        acquire(out, error, || {
            let owner = assets(input)?;
            #[cfg(feature = "test-support")]
            if let Some(hooks) = hooks {
                if !aligned(hooks) {
                    return Err(Failure::Input("invalid illuminant cache observation"));
                }
                return Ok(FjIlluminantsOwner {
                    entry: owner
                        .illuminants_observed(&IlluminantObserver(&*hooks))
                        .map_err(Failure::Asset)?,
                });
            }
            Ok(FjIlluminantsOwner {
                entry: owner.illuminants().map_err(Failure::Asset)?,
            })
        })
    }
}
/// # Safety
/// The live owner bounds every returned borrow; output/diagnostic are exclusive.
pub(crate) unsafe fn illuminants_view(
    owner: *const FjIlluminantsOwner,
    out: *mut FjIlluminantsView,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The numeric/span-only view is zero-valid and borrows this owner.
    unsafe { view(owner, out, error, illuminants_projection) }
}
/// # Safety
/// Consume exactly one matching owner after excluding its outstanding borrows.
pub(crate) unsafe fn illuminants_release(
    owner: *mut FjIlluminantsOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: release consumes the caller's matching allocation once.
    unsafe { release(owner, error) }
}
/// # Safety
/// Live borrowed Assets and exclusive disjoint output/diagnostic storage.
pub(crate) unsafe fn hull_acquire(
    input: *const FjAssets,
    out: *mut *mut FjInputHullOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Forward the same admitted storage to the shared operation.
    unsafe {
        hull_acquire_with(
            input,
            out,
            error,
            #[cfg(feature = "test-support")]
            None,
        )
    }
}
#[cfg(feature = "test-support")]
#[repr(C)]
pub(crate) struct FjInputHullCacheHooks {
    before_build: Option<unsafe extern "C" fn(usize, usize) -> u32>,
}
#[cfg(feature = "test-support")]
struct HullObserver<'a>(&'a FjInputHullCacheHooks);
#[cfg(feature = "test-support")]
impl film_juicer_core::assets::testing::HullObserver for HullObserver<'_> {
    fn before_build(
        &self,
        entry: usize,
        bytes: usize,
    ) -> Result<(), film_juicer_core::assets::DerivedError> {
        let category = if let Some(callback) = self.0.before_build {
            // SAFETY: This synchronous callback contains all native exceptions;
            // the entry identity is compared only and never dereferenced.
            unsafe { callback(entry, bytes) }
        } else {
            0
        };
        observer_status(category, "input hull cache observation failed")
    }
}
/// # Safety
/// The callback table is immutable/live until acquire returns. Callbacks contain
/// every native exception and run outside owner locks; entry words are opaque.
#[cfg(feature = "test-support")]
pub(crate) unsafe fn hull_acquire_observed(
    input: *const FjAssets,
    hooks: *const FjInputHullCacheHooks,
    out: *mut *mut FjInputHullOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Shared acquire clears output before callback-table admission.
    unsafe { hull_acquire_with(input, out, error, Some(hooks)) }
}
/// # Safety
/// All inputs and exclusive outputs obey hull_acquire's foreign contract.
unsafe fn hull_acquire_with(
    input: *const FjAssets,
    out: *mut *mut FjInputHullOwner,
    error: *mut FjErrorBuffer,
    #[cfg(feature = "test-support")] hooks: Option<*const FjInputHullCacheHooks>,
) -> FjStatus {
    // SAFETY: acquire admits output before borrowing the authorized Assets.
    unsafe {
        acquire(out, error, || {
            let owner = assets(input)?;
            #[cfg(feature = "test-support")]
            if let Some(hooks) = hooks {
                if !aligned(hooks) {
                    return Err(Failure::Input("invalid input hull cache observation"));
                }
                return Ok(FjInputHullOwner {
                    entry: owner
                        .input_hull_observed(&HullObserver(&*hooks))
                        .map_err(Failure::Asset)?,
                });
            }
            Ok(FjInputHullOwner {
                entry: owner.input_hull().map_err(Failure::Asset)?,
            })
        })
    }
}
/// # Safety
/// Every result pointer expires with the live owner; outputs remain disjoint.
pub(crate) unsafe fn hull_view(
    owner: *const FjInputHullOwner,
    out: *mut FjInputHullView,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: This zero-valid C record borrows only retained hull backing.
    unsafe { view(owner, out, error, |owner| hull_projection(&owner.entry)) }
}
/// # Safety
/// Transfer one matching live owner once, with no outstanding view borrow.
pub(crate) unsafe fn hull_release(
    owner: *mut FjInputHullOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: release consumes exactly the caller-authorized Box.
    unsafe { release(owner, error) }
}
/// # Safety
/// source is initialized and immutable; Assets stays live; output and diagnostic
/// extents are exclusive, aligned and disjoint from both inputs.
pub(crate) unsafe fn boundary_acquire(
    input: *const FjAssets,
    source: *const FjOutputBoundarySource,
    expected: u64,
    out: *mut *mut FjOutputBoundaryOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Forward the same admitted foreign storage to the shared operation.
    unsafe {
        boundary_acquire_with(
            input,
            source,
            expected,
            out,
            error,
            #[cfg(feature = "test-support")]
            None,
        )
    }
}

#[cfg(feature = "test-support")]
#[repr(C)]
pub(crate) struct FjOutputCacheHooks {
    after_miss: Option<unsafe extern "C" fn(u64) -> u32>,
    after_selection: Option<unsafe extern "C" fn(u32, u64) -> u32>,
}

#[cfg(feature = "test-support")]
struct OutputObserver<'a>(&'a FjOutputCacheHooks);
#[cfg(feature = "test-support")]
impl film_juicer_core::assets::testing::OutputObserver for OutputObserver<'_> {
    fn after_miss(&self, generation: u64) -> Result<(), film_juicer_core::assets::DerivedError> {
        let result = if let Some(callback) = self.0.after_miss {
            // SAFETY: The synchronous test call supplies a valid C callback
            // which contains every native exception and retains no borrow.
            unsafe { callback(generation) }
        } else {
            0
        };
        observer_status(result, "output cache observation failed")
    }
    fn after_selection(
        &self,
        warm: bool,
        generation: u64,
    ) -> Result<(), film_juicer_core::assets::DerivedError> {
        let result = if let Some(callback) = self.0.after_selection {
            // SAFETY: The same synchronous, exception-containing C callback contract.
            unsafe { callback(u32::from(warm), generation) }
        } else {
            0
        };
        observer_status(result, "output cache observation failed")
    }
}
#[cfg(feature = "test-support")]
fn observer_status(
    category: u32,
    detail: &'static str,
) -> Result<(), film_juicer_core::assets::DerivedError> {
    use film_juicer_core::assets::DerivedError;
    match category {
        0 => Ok(()),
        2 => Err(DerivedError::Unsupported(detail)),
        3 => Err(DerivedError::Resource(detail)),
        4 => Err(DerivedError::Allocation(detail)),
        _ => Err(DerivedError::Internal(detail)),
    }
}

/// # Safety
/// The source and callback table are immutable through synchronous return.
/// Callbacks contain all foreign exceptions and are invoked outside cache locks.
#[cfg(feature = "test-support")]
pub(crate) unsafe fn boundary_acquire_observed(
    input: *const FjAssets,
    source: *const FjOutputBoundarySource,
    expected: u64,
    hooks: *const FjOutputCacheHooks,
    out: *mut *mut FjOutputBoundaryOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The shared edge clears output before callback-table admission.
    unsafe { boundary_acquire_with(input, source, expected, out, error, Some(hooks)) }
}

/// # Safety
/// All borrowed inputs and exclusive outputs obey boundary_acquire's contract.
unsafe fn boundary_acquire_with(
    input: *const FjAssets,
    source: *const FjOutputBoundarySource,
    expected: u64,
    out: *mut *mut FjOutputBoundaryOwner,
    error: *mut FjErrorBuffer,
    #[cfg(feature = "test-support")] hooks: Option<*const FjOutputCacheHooks>,
) -> FjStatus {
    // SAFETY: acquire clears/admit outputs before inspecting either input.
    unsafe {
        acquire(out, error, || {
            if !aligned(source) {
                return Err(Failure::Input("invalid output boundary source"));
            }
            let source = &*source;
            let space = color::OutputSpace::try_from(source.output_space)
                .map_err(|_| Failure::Input("invalid output boundary space"))?;
            let owner = assets(input)?;
            let input = gamut::BoundaryInput {
                space,
                d65_xyz_to_native_rgb: &source.d65_xyz_to_native_rgb,
                transform_hash: source.transform_hash,
            };
            #[cfg(feature = "test-support")]
            if let Some(hooks) = hooks {
                if !aligned(hooks) {
                    return Err(Failure::Input("invalid output cache observation"));
                }
                return Ok(FjOutputBoundaryOwner {
                    entry: owner
                        .output_boundary_observed(input, expected, &OutputObserver(&*hooks))
                        .map_err(Failure::Asset)?,
                });
            }
            Ok(FjOutputBoundaryOwner {
                entry: owner
                    .output_boundary(input, expected)
                    .map_err(Failure::Asset)?,
            })
        })
    }
}
/// # Safety
/// A live immutable matching owner bounds all span uses; output is exclusive.
pub(crate) unsafe fn boundary_view(
    owner: *const FjOutputBoundaryOwner,
    out: *mut FjOutputBoundaryView,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The zero-valid C view borrows only the retained boundary entry.
    unsafe {
        view(owner, out, error, |owner| {
            boundary_projection(owner.entry.boundary())
        })
    }
}
/// # Safety
/// Consume one matching live hold with no overlapping view uses.
pub(crate) unsafe fn boundary_release(
    owner: *mut FjOutputBoundaryOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The common release consumes exactly this acquired Box.
    unsafe { release(owner, error) }
}

macro_rules! cache_export {
    ($name:ident($($arg:ident: $ty:ty),*) => $body:expr) => {
        /// # Safety
        /// The matching acquire/view/release contract in juicer_legacy_api.h
        /// authorizes every live input, exclusive output and diagnostic extent.
        #[unsafe(no_mangle)]
        unsafe extern "C" fn $name($($arg: $ty),*) -> FjStatus {
            // SAFETY: This export forwards exactly the caller's documented contract.
            unsafe { $body }
        }
    };
}
cache_export!(fj_legacy_assets_illuminants_acquire(assets: *const FjAssets, out: *mut *mut FjIlluminantsOwner, error: *mut FjErrorBuffer) => illuminants_acquire(assets, out, error));
cache_export!(fj_legacy_illuminants_view(owner: *const FjIlluminantsOwner, out: *mut FjIlluminantsView, error: *mut FjErrorBuffer) => illuminants_view(owner, out, error));
cache_export!(fj_legacy_illuminants_release(owner: *mut FjIlluminantsOwner, error: *mut FjErrorBuffer) => illuminants_release(owner, error));
cache_export!(fj_legacy_assets_input_hull_acquire(assets: *const FjAssets, out: *mut *mut FjInputHullOwner, error: *mut FjErrorBuffer) => hull_acquire(assets, out, error));
cache_export!(fj_legacy_input_hull_view(owner: *const FjInputHullOwner, out: *mut FjInputHullView, error: *mut FjErrorBuffer) => hull_view(owner, out, error));
cache_export!(fj_legacy_input_hull_release(owner: *mut FjInputHullOwner, error: *mut FjErrorBuffer) => hull_release(owner, error));
cache_export!(fj_legacy_assets_output_boundary_acquire(assets: *const FjAssets, source: *const FjOutputBoundarySource, expected: u64, out: *mut *mut FjOutputBoundaryOwner, error: *mut FjErrorBuffer) => boundary_acquire(assets, source, expected, out, error));
cache_export!(fj_legacy_output_boundary_view(owner: *const FjOutputBoundaryOwner, out: *mut FjOutputBoundaryView, error: *mut FjErrorBuffer) => boundary_view(owner, out, error));
cache_export!(fj_legacy_output_boundary_release(owner: *mut FjOutputBoundaryOwner, error: *mut FjErrorBuffer) => boundary_release(owner, error));

cache_export!(fj_legacy_prepared_state_create(assets: *mut FjAssets, params: *const FjPreparationParams, out: *mut *mut FjPreparedState, error: *mut FjErrorBuffer) => state_create(assets, params, out, error));
cache_export!(fj_legacy_prepared_state_view(owner: *const FjPreparedState, out: *mut FjPreparedStateView, error: *mut FjErrorBuffer) => state_view(owner, out, error));
cache_export!(fj_legacy_prepared_state_release(owner: *mut FjPreparedState, error: *mut FjErrorBuffer) => state_release(owner, error));

pub(crate) fn failure_category(error: &recipe::PrepareError) -> u32 {
    use recipe::PrepareError as E;
    match error {
        E::Control(
            recipe::ControlError::InputColorSpace
            | recipe::ControlError::ReconstructionMethod
            | recipe::ControlError::Meter
            | recipe::ControlError::OutputColorSpace
            | recipe::ControlError::PrintStage,
        ) => FJ_STATUS_UNSUPPORTED_INPUT,
        E::Development(
            film_juicer_core::film::Error::InvalidGamma
            | film_juicer_core::film::Error::RouteMismatch,
        ) => FJ_STATUS_UNSUPPORTED_INPUT,
        E::Development(film_juicer_core::film::Error::Allocation)
        | E::ReferenceWhite(film_juicer_core::reconstruction::Error::AllocationFailure)
        | E::Dir {
            error:
                film_juicer_core::film::dir::Error::Allocation
                | film_juicer_core::film::dir::Error::Size,
            ..
        }
        | E::Tc(film_juicer_core::reconstruction::TcError::AllocationFailure)
        | E::PrintDevelopment(film_juicer_core::profile::PrintDensityError::Capacity) => {
            FJ_STATUS_ALLOCATION_FAILURE
        }
        E::Input(film_juicer_core::film::input::Error::Panic(_)) => FJ_STATUS_INTERNAL_FAILURE,
        #[cfg(feature = "test-support")]
        E::Input(film_juicer_core::film::input::Error::Boundary(_, category)) => *category,
        #[cfg(feature = "test-support")]
        E::Reference(film_juicer_core::film::reference::Error::Dependency { category, .. }) => {
            match category {
                film_juicer_core::film::reference::Category::Unsupported => {
                    FJ_STATUS_UNSUPPORTED_INPUT
                }
                film_juicer_core::film::reference::Category::Allocation => {
                    FJ_STATUS_ALLOCATION_FAILURE
                }
            }
        }
        #[cfg(feature = "test-support")]
        E::ScannerColor(
            film_juicer_core::scanner::ColorError::OutputBoundary
            | film_juicer_core::scanner::ColorError::Cat02Boundary,
        ) => FJ_STATUS_UNSUPPORTED_INPUT,
        // Assets errors are translated through the same existing asset boundary
        // before being wrapped, retaining their exact classification.
        E::Asset(_) => FJ_STATUS_INTERNAL_FAILURE,
        _ => FJ_STATUS_PREPARATION_FAILURE,
    }
}

#[cfg(feature = "test-support")]
pub(crate) fn layout_facts() -> &'static [usize] {
    macro_rules! facts {
        ($($ty:ty => [$($field:ident),*]),* $(,)?) => {
            &[$(size_of::<$ty>(), align_of::<$ty>(), $(std::mem::offset_of!($ty, $field),)*)*]
        };
    }
    const FACTS: &[usize] = facts!(
        FjIlluminantsView => [d65, d55, d50, tungsten, k75p, tungsten_kg3, tungsten_kg3_lens, entry_identity],
        FjInputHullView => [center_xy, xy, hash, entry_identity],
        FjOutputBoundaryView => [cmax, transform_hash, contract_hash, hash, entry_identity],
        FjOutputBoundarySource => [output_space, d65_xyz_to_native_rgb, transform_hash],
        FjPreparationHalation => [active, scatter_amount, scatter_spatial_scale, halation_amount, halation_spatial_scale],
        FjPreparationParams => [print_exposure, print_preflash_exposure, print_gamma_factor, preflash_m_filter_cc, preflash_y_filter_cc, print_shadow_compensation_factor, print_shadow_compensation_density, print_shadow_compensation_transition, glare_percent, glare_roughness, glare_blur_sigma_px, camera_exposure_compensation_ev, camera_film_format_long_edge_mm, film_gamma_factor, scanner_lens_blur_sigma_px, scanner_black_level, scanner_white_level, gate_weave_amount, scanner_unsharp_mask, print_ui_ymc_cc, camera_filter_uv, camera_filter_ir, camera_diffusion, enlarger_diffusion, dir_couplers, film_profile_key, print_profile_key, spectral_upsampling_mode, ref_ill, enl_ill, normalize_print_exposure, print_exposure_compensation, input_color_space, input_cctf_decoding, input_compression_enabled, hanatos2025_adaptation_window, hanatos2025_adaptation_surface, camera_auto_exposure_enabled, camera_metering_method, scatter_halation, scanner_black_correction, scanner_white_correction, scanner_lut_resolution, output_color_space, output_cctf_encoding, output_gamut_compression_enabled, grain, film_dust_amount, film_scratch_amount, gate_dust_amount, gate_scratch_amount, scan_route, glare_active, camera_filter_override],
        FjPreparedSpectralView => [lambda_nm, illuminant, observer_xyz, weighted_xyz, dyes_cmy, baseline_min, baseline_mid, inv_yn, white_xyz, has_baseline, illuminant_hash, hash],
        FjPreparedInputPolicy => [meter, band_pass_active, apply_hanatos_window, apply_hanatos_surface, projection_illuminant, input_compression_active, film_format_long_edge_mm, uv, ir, hanatos_spectral_blur, nominal_white_xyz, projection_white_xyz, hanatos_window, hanatos_surface_rgb, input_hull_hash, tc_source_hash],
        FjPreparedScannerPolicy => [black_correction, white_correction, glare_active, black_level, white_level, glare_percent, glare_roughness, glare_blur_sigma_px, lens_blur_sigma_px, unsharp_sigma_px, unsharp_amount],
        FjPreparedStateView => [film_profile_key, print_profile_key, film_profile_asset_version, print_profile_asset_version, profile_route_hash, recipe_hash, upload_core_hash, scanner_hash, route, polarity, input_policy, scanner_policy, film_exposure, film_development, dir_recipe, grain_minimum, scanner_bounds, enlarger_film_bounds, film_reference, optics, grain, effects, exposure_tables, scanner_tables, s_inverse, midgray_dwg_rgb, raw_midgray, raw_green, midgray_scale, raw_midgray_bgr, input_hull, output_boundary, scanner_lut, scanner_color, scanner_correction, scanner_post_effects, output_color, print, print_gamma, print_filter_main_cmy_cc, print_filter_preflash_cmy_cc, print_normalization_mode, normalize_print_exposure, print_exposure_compensation, calibration, neutral_cmy_cc, print_source_illuminant, print_balance, scanner_recipe_hash, print_recipe_hash, print_filters_hash, print_exposure_hash, print_illuminant_hash, print_curves_hash],
        FjPreparationTrace => [count, stages],
        FjOutputCacheHooks => [after_miss, after_selection],
        FjCacheMetrics => [allocations, derivations, destructions, live_bytes, peak_bytes, reentries, reentry_failures],
        FjIlluminantCacheHooks => [before_source, before_publication],
        FjInputHullCacheHooks => [before_build]
    );
    FACTS
}
