//! SF_TEMP_BRIDGE_FILM_DIR_DENSITY: synchronous family owners and fixed projections; remove S4.E.

use std::fmt::{self, Write};
use std::mem::{align_of, size_of};
use std::panic::{catch_unwind, AssertUnwindSafe};
use film_juicer_core::effects;
use film_juicer_core::film::{Development, density, dir, grain};
use film_juicer_core::profile::Polarity;
use film_juicer_core::route::ScanRoute;
use crate::film_bridge::FjFilmDevelopmentOwner;
use crate::cuda::sys::{
    FjDust, FjScratch, FjErrorBuffer, FjFloatSpan, FjStatus, FJ_API_NONE, FJ_STATUS_SUCCESS,
    FJ_STATUS_UNSUPPORTED_INPUT, FJ_STATUS_PREPARATION_FAILURE, FJ_STATUS_ALLOCATION_FAILURE,
    FJ_STATUS_INTERNAL_FAILURE,
};

pub(crate) const FJ_FILM_DIR_INACTIVE: u32 = 0;

pub(crate) const FJ_FILM_DIR_NEGATIVE_DONOR_LANGMUIR: u32 = 1;

pub(crate) const FJ_FILM_DIR_POSITIVE_RECEIVER_LANGMUIR: u32 = 2;

pub(crate) const FJ_FILM_RECIPE_REASON_NONE: u32 = 0;

pub(crate) const FJ_FILM_RECIPE_RAW: u32 = 1;

pub(crate) const FJ_FILM_RECIPE_MINIMUM: u32 = 2;

pub(crate) const FJ_FILM_RECIPE_MATRIX: u32 = 3;

pub(crate) const FJ_FILM_RECIPE_DENSITY_MAX: u32 = 4;

pub(crate) const FJ_FILM_RECIPE_RECEIVER_K: u32 = 5;

pub(crate) const FJ_FILM_RECIPE_DONOR_K: u32 = 6;

pub(crate) const FJ_FILM_RECIPE_DONOR_LANGMUIR: u32 = 7;

pub(crate) const FJ_FILM_RECIPE_RECEIVER_LANGMUIR: u32 = 8;

pub(crate) const FJ_FILM_RECIPE_COMPENSATED_AXIS: u32 = 9;

pub(crate) const FJ_FILM_RECIPE_IDENTITY: u32 = 10;

pub(crate) const FJ_FILM_RECIPE_BOUNDS_SPAN: u32 = 11;

pub(crate) const FJ_FILM_RECIPE_SIZE: u32 = 12;

pub(crate) const FJ_FILM_RECIPE_ALLOCATION: u32 = 13;

pub(crate) const FJ_FILM_RECIPE_PANIC: u32 = 14;

pub(crate) const FJ_FILM_RECIPE_INTEGRITY: u32 = 15;

pub(crate) const FJ_FILM_RECIPE_LOG_EXPOSURE: u32 = 16;

pub(crate) const FJ_FILM_RECIPE_GRAIN_CONTROLS: u32 = 17;
pub(crate) const FJ_FILM_RECIPE_GRAIN_LAYERS: u32 = 18;
pub(crate) const FJ_FILM_RECIPE_EFFECTS_PROBABILITY: u32 = 19;

#[repr(C)]
pub(crate) struct FjFilmRecipeFailure {
    pub(crate) reason: u32,
    pub(crate) channel: u32,
    pub(crate) sample: u64,
}

#[repr(C)]
pub(crate) struct FjFilmDensityMinimum {
    pub(crate) cmy: [f32; 3],
}

#[repr(C)]
pub(crate) struct FjFilmDirParams {
    pub(crate) active: u32,
    pub(crate) amount: f32,
    pub(crate) inhibition_same_layer: f32,
    pub(crate) inhibition_interlayer: f32,
    pub(crate) diffusion_size_um: f32,
    pub(crate) diffusion_tail_um: f32,
    pub(crate) diffusion_tail_weight: f32,
    pub(crate) gamma_use_stock: u32,
    pub(crate) gamma_same_layer_rgb: [f32; 3],
    pub(crate) gamma_interlayer_r_to_gb: [f32; 2],
    pub(crate) gamma_interlayer_g_to_rb: [f32; 2],
    pub(crate) gamma_interlayer_b_to_rg: [f32; 2],
    pub(crate) langmuir_donor_k_rgb: [f32; 3],
    pub(crate) langmuir_receiver_k_rgb: [f32; 3],
}

#[repr(C)]
pub(crate) struct FjFilmDirView {
    pub(crate) polarity: u32,
    pub(crate) mode: u32,
    pub(crate) matrix_rgb: [f32; 9],
    pub(crate) diffusion_size_um: f32,
    pub(crate) diffusion_tail_um: f32,
    pub(crate) diffusion_tail_weight: f32,
    pub(crate) density_max_rgb: [f32; 3],
    pub(crate) density_ref_rgb: [f32; 3],
    pub(crate) donor_k_rgb: [f32; 3],
    pub(crate) receiver_c_ref_rgb: [f32; 3],
    pub(crate) receiver_kr_rgb: [f32; 3],
    pub(crate) compensated_axes_rgb: [FjFloatSpan; 3],
    pub(crate) compensated_axes_hash: u64,
    pub(crate) hash: u64,
}

#[repr(C)]
pub(crate) struct FjFilmDensityBounds {
    pub(crate) route: u32,
    pub(crate) medium: u32,
    pub(crate) polarity: u32,
    pub(crate) source: u32,
    pub(crate) minimum_cmy: [f32; 3],
    pub(crate) maximum_cmy: [f32; 3],
    pub(crate) inv_span_cmy: [f32; 3],
    pub(crate) authored_minimum_cmy: [f32; 3],
    pub(crate) authored_maximum_cmy: [f32; 3],
    pub(crate) hash: u64,
}

#[repr(C)]
pub(crate) struct FjFilmDirOwner {
    _opaque: [u8; 0],
}

// SF_TEMP_BRIDGE_VISUAL_GRAIN: fixed completed grain transport; remove S4.E.
#[repr(C)]
pub(crate) struct FjVisualGrainParams {
    pub(crate) active: u32,
    pub(crate) sublayers_active: u32,
    pub(crate) particle_area_um2: f32,
    pub(crate) particle_scale_cmy: [f32; 3],
    pub(crate) particle_scale_layers: [f32; 3],
    pub(crate) particle_density_min_cmy: [f32; 3],
    pub(crate) uniformity_cmy: [f32; 3],
    pub(crate) correlation_sigma_px: f32,
    pub(crate) dye_cloud_blur_um: f32,
    pub(crate) micro_structure: [f32; 2],
    pub(crate) sublayer_count: i32,
    pub(crate) amplitude: f32,
    pub(crate) chroma_mix: f32,
    pub(crate) chroma_shared_weight: f32,
    pub(crate) chroma_independent_weight: f32,
    pub(crate) coarse_weight: f32,
    pub(crate) mid_weight: f32,
    pub(crate) size_mix_scale: f32,
    pub(crate) clump_temporal_mix: f32,
    pub(crate) clump_morph_period_sec: f32,
    pub(crate) debug_view: i32,
}

#[repr(C)]
pub(crate) struct FjVisualGrainRecipe {
    pub(crate) active: u32,
    pub(crate) sublayers_active: u32,
    pub(crate) axis_finite_cmy: [u32; 3],
    pub(crate) axis_block_prefix_max_cmy: [[f32; 16]; 3],
    pub(crate) particle_area_um2: f32,
    pub(crate) particle_scale_cmy: [f32; 3],
    pub(crate) particle_scale_layers: [f32; 3],
    pub(crate) particle_density_min_cmy: [f32; 3],
    pub(crate) uniformity_cmy: [f32; 3],
    pub(crate) correlation_sigma_px: f32,
    pub(crate) dye_cloud_blur_um: f32,
    pub(crate) micro_structure: [f32; 2],
    pub(crate) sublayer_count: i32,
    pub(crate) amplitude: f32,
    pub(crate) chroma_mix: f32,
    pub(crate) chroma_shared_weight: f32,
    pub(crate) chroma_independent_weight: f32,
    pub(crate) fine_weight: f32,
    pub(crate) mid_weight: f32,
    pub(crate) coarse_weight: f32,
    pub(crate) size_mix_scale: f32,
    pub(crate) clump_temporal_mix: f32,
    pub(crate) clump_morph_period_sec: f32,
    pub(crate) debug_view: i32,
    pub(crate) layers_hash: u64,
    pub(crate) hash: u64,
}

// SF_TEMP_BRIDGE_PHYSICAL_EFFECTS: fixed owned result projection; remove S4.E.
#[repr(C)]
pub(crate) struct FjPhysicalEffectsParams {
    pub(crate) film_dust_amount: f32,
    pub(crate) film_scratch_amount: f32,
    pub(crate) gate_dust_amount: f32,
    pub(crate) gate_scratch_amount: f32,
    pub(crate) gate_weave_amount: f64,
    pub(crate) grain_active: u32,
    pub(crate) grain_debug_view: i32,
}

#[repr(C)]
pub(crate) struct FjPhysicalEffectsRecipe {
    pub(crate) film_dust: FjDust,
    pub(crate) film_scratch: FjScratch,
    pub(crate) gate_dust: FjDust,
    pub(crate) gate_scratch: FjScratch,
    pub(crate) gate_weave_amount: f64,
    pub(crate) active: u32,
    pub(crate) hash: u64,
}

fn project_dust(policy: &effects::Dust) -> FjDust {
    FjDust {
        cell_width_mm: policy.cell_width_mm(),
        cell_height_mm: policy.cell_height_mm(),
        slot_probability: policy.slot_probability(),
        softness_min_mm: policy.softness_min_mm(),
        softness_max_mm: policy.softness_max_mm(),
        softness_size_cap_fraction: policy.softness_size_cap_fraction(),
        support_x_mm: policy.support_x_mm(),
        support_y_mm: policy.support_y_mm(),
        fiber_fraction: policy.fiber_fraction(),
        fiber_drift_fraction: policy.fiber_drift_fraction(),
        fiber_first_knot_min: policy.fiber_first_knot_min(),
        fiber_first_knot_max: policy.fiber_first_knot_max(),
        fiber_second_knot_min: policy.fiber_second_knot_min(),
        fiber_second_knot_max: policy.fiber_second_knot_max(),
        fiber_interior_width_min_fraction: policy.fiber_interior_width_min_fraction(),
        fiber_interior_width_max_fraction: policy.fiber_interior_width_max_fraction(),
        diameter_min_mm: policy.diameter_min_mm(),
        diameter_bulk_max_mm: policy.diameter_bulk_max_mm(),
        diameter_max_mm: policy.diameter_max_mm(),
        diameter_tail_fraction: policy.diameter_tail_fraction(),
        fiber_length_min_mm: policy.fiber_length_min_mm(),
        fiber_length_max_mm: policy.fiber_length_max_mm(),
        fiber_width_min_mm: policy.fiber_width_min_mm(),
        fiber_width_max_mm: policy.fiber_width_max_mm(),
        opacity_faint_cumulative: policy.opacity_faint_cumulative(),
        opacity_intermediate_cumulative: policy.opacity_intermediate_cumulative(),
        compact_opacity_min: policy.compact_opacity_min(),
        compact_opacity_faint_end: policy.compact_opacity_faint_end(),
        compact_opacity_intermediate_end: policy.compact_opacity_intermediate_end(),
        compact_opacity_max: policy.compact_opacity_max(),
        fiber_opacity_min: policy.fiber_opacity_min(),
        fiber_opacity_faint_end: policy.fiber_opacity_faint_end(),
        fiber_opacity_intermediate_end: policy.fiber_opacity_intermediate_end(),
        fiber_opacity_max: policy.fiber_opacity_max(),
        compact_dominant_aspect_min: policy.compact_dominant_aspect_min(),
        compact_dominant_aspect_max: policy.compact_dominant_aspect_max(),
        compact_subsidiary_scale_min: policy.compact_subsidiary_scale_min(),
        compact_subsidiary_scale_max: policy.compact_subsidiary_scale_max(),
        compact_subsidiary_aspect_min: policy.compact_subsidiary_aspect_min(),
        compact_subsidiary_aspect_max: policy.compact_subsidiary_aspect_max(),
        compact_subsidiary_offset_max: policy.compact_subsidiary_offset_max(),
        compact_subsidiary_angle_max_radians: policy.compact_subsidiary_angle_max_radians(),
    }
}

fn project_scratch(policy: &effects::Scratch) -> FjScratch {
    FjScratch {
        cell_width_mm: policy.cell_width_mm(),
        cell_height_mm: policy.cell_height_mm(),
        slot_probability: policy.slot_probability(),
        softness_min_mm: policy.softness_min_mm(),
        softness_max_mm: policy.softness_max_mm(),
        softness_size_cap_fraction: policy.softness_size_cap_fraction(),
        support_x_mm: policy.support_x_mm(),
        support_y_mm: policy.support_y_mm(),
        length_min_mm: policy.length_min_mm(),
        length_bulk_max_mm: policy.length_bulk_max_mm(),
        length_max_mm: policy.length_max_mm(),
        length_tail_fraction: policy.length_tail_fraction(),
        width_min_mm: policy.width_min_mm(),
        width_bulk_max_mm: policy.width_bulk_max_mm(),
        width_max_mm: policy.width_max_mm(),
        width_tail_fraction: policy.width_tail_fraction(),
        drift_fraction: policy.drift_fraction(),
        first_knot_min: policy.first_knot_min(),
        first_knot_max: policy.first_knot_max(),
        second_knot_min: policy.second_knot_min(),
        second_knot_max: policy.second_knot_max(),
        interior_width_min_fraction: policy.interior_width_min_fraction(),
        interior_width_max_fraction: policy.interior_width_max_fraction(),
        interior_depth_min_fraction: policy.interior_depth_min_fraction(),
        interior_depth_max_fraction: policy.interior_depth_max_fraction(),
        endpoint_abrupt_probability: policy.endpoint_abrupt_probability(),
        interruption_probability: policy.interruption_probability(),
        gap_center_min: policy.gap_center_min(),
        gap_center_max: policy.gap_center_max(),
        gap_span_min: policy.gap_span_min(),
        gap_span_max: policy.gap_span_max(),
        scuff_probability: policy.scuff_probability(),
        scuff_length_max_mm: policy.scuff_length_max_mm(),
        scuff_angle_max_radians: policy.scuff_angle_max_radians(),
        strength_min: policy.strength_min(),
        strength_max: policy.strength_max(),
    }
}

/// # Safety
/// Initialized live aligned params and exclusive output/failure/diagnostic
/// extents are mutually disjoint through synchronous return. No borrow escapes.
pub(crate) unsafe fn prepare_effects(
    params: *const FjPhysicalEffectsParams,
    out: *mut FjPhysicalEffectsRecipe,
    failure: *mut FjFilmRecipeFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // Consume at raw entry, including independently rejected output storage.
    #[cfg(feature = "test-support")]
    let fault = faults::take(8);
    // SAFETY: execute independently admits outputs before borrowing input.
    unsafe {
        execute(out, failure, error, || {
            if !valid_extent(params, 1) || (*params).grain_active > 1 {
                return Err(Failure::raw());
            }
            let input = &*params;
            let params = effects::Params {
                film_dust_amount: input.film_dust_amount,
                film_scratch_amount: input.film_scratch_amount,
                gate_dust_amount: input.gate_dust_amount,
                gate_scratch_amount: input.gate_scratch_amount,
                gate_weave_amount: input.gate_weave_amount,
                grain_active: input.grain_active != 0,
                grain_debug_view: input.grain_debug_view,
            };
            #[cfg(feature = "test-support")]
            match fault {
                1 => panic!("physical effects fixture panic"),
                2 => effects::test_support::substitute_next_probability(f32::NAN),
                3 => effects::test_support::substitute_next_probability(-1.0),
                4 => effects::test_support::substitute_next_probability(0.25),
                _ => {}
            }
            let recipe = effects::Recipe::new(params).map_err(|effects::Error::Probability| {
                Failure::new(
                    FJ_STATUS_PREPARATION_FAILURE,
                    FJ_FILM_RECIPE_EFFECTS_PROBABILITY,
                )
            })?;
            Ok(FjPhysicalEffectsRecipe {
                film_dust: project_dust(recipe.film_dust()),
                film_scratch: project_scratch(recipe.film_scratch()),
                gate_dust: project_dust(recipe.gate_dust()),
                gate_scratch: project_scratch(recipe.gate_scratch()),
                gate_weave_amount: recipe.gate_weave_amount(),
                active: u32::from(recipe.active()),
                hash: recipe.hash(),
            })
        })
    }
}

/// # Safety
/// The checked operation's live, initialized, aligned, disjoint record contract.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_physical_effects_prepare(
    params: *const FjPhysicalEffectsParams,
    out_recipe: *mut FjPhysicalEffectsRecipe,
    out_failure: *mut FjFilmRecipeFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Forward the identical synchronous raw contract.
    unsafe { prepare_effects(params, out_recipe, out_failure, error) }
}
const _: unsafe extern "C" fn(
    *const FjPhysicalEffectsParams,
    *mut FjPhysicalEffectsRecipe,
    *mut FjFilmRecipeFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_legacy_physical_effects_prepare;
const _: () = assert!(
    size_of::<FjPhysicalEffectsParams>() == 32 && align_of::<FjPhysicalEffectsParams>() == 8
);
const _: () = assert!(
    size_of::<FjPhysicalEffectsRecipe>() == 648 && align_of::<FjPhysicalEffectsRecipe>() == 8
);

fn status(category: u32) -> FjStatus {
    FjStatus {
        category,
        api: FJ_API_NONE,
        native_code: 0,
    }
}
fn valid_extent<T>(pointer: *const T, count: usize) -> bool {
    !pointer.is_null()
        && (pointer as usize).is_multiple_of(align_of::<T>())
        && count.checked_mul(size_of::<T>()).is_some_and(|bytes| {
            bytes <= isize::MAX as usize && (pointer as usize).checked_add(bytes).is_some()
        })
}
struct Diagnostic {
    record: *mut FjErrorBuffer,
    data: *mut u8,
    capacity: usize,
    length: usize,
}
impl Write for Diagnostic {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        if self.capacity != 0 {
            let count = text.len().min(self.capacity - 1 - self.length);
            // SAFETY: Admission establishes exclusive backing with this checked extent.
            unsafe {
                std::ptr::copy_nonoverlapping(text.as_ptr(), self.data.add(self.length), count);
                self.length += count;
                self.data.add(self.length).write(0);
            }
        }
        if !self.record.is_null() {
            // SAFETY: The exclusively writable admitted record stays live until return.
            unsafe {
                (*self.record).length = self.length;
            }
        }
        Ok(())
    }
}
impl Diagnostic {
    /// # Safety
    /// The optional record/backing are initialized, live, exclusive and disjoint.
    unsafe fn admit(record: *mut FjErrorBuffer) -> Result<Self, ()> {
        let mut writer = Self {
            record,
            data: std::ptr::null_mut(),
            capacity: 0,
            length: 0,
        };
        if record.is_null() {
            return Ok(writer);
        }
        if !valid_extent(record, 1) {
            return Err(());
        }
        // SAFETY: Capacity and writable length are initialized; zero capacity never reads data.
        unsafe {
            (*record).length = 0;
            writer.capacity = (*record).capacity;
            if writer.capacity != 0 {
                let data = (*record).data.cast::<u8>();
                if !valid_extent(data, writer.capacity) {
                    return Err(());
                }
                writer.data = data;
                data.write(0);
            }
        }
        Ok(writer)
    }
}
struct Failure {
    category: u32,
    record: FjFilmRecipeFailure,
}
impl Failure {
    fn new(category: u32, reason: u32) -> Self {
        Self {
            category,
            record: FjFilmRecipeFailure {
                reason,
                channel: 3,
                sample: u64::MAX,
            },
        }
    }
    fn raw() -> Self {
        Self::new(FJ_STATUS_UNSUPPORTED_INPUT, FJ_FILM_RECIPE_RAW)
    }
    fn dir(error: dir::Error) -> Self {
        let (reason, channel, sample, category) = match error {
            dir::Error::Matrix => (
                FJ_FILM_RECIPE_MATRIX,
                3,
                u64::MAX,
                FJ_STATUS_PREPARATION_FAILURE,
            ),
            dir::Error::LogExposure { sample } => (
                FJ_FILM_RECIPE_LOG_EXPOSURE,
                3,
                sample as u64,
                FJ_STATUS_PREPARATION_FAILURE,
            ),
            dir::Error::DensityMaximum => (
                FJ_FILM_RECIPE_DENSITY_MAX,
                3,
                u64::MAX,
                FJ_STATUS_PREPARATION_FAILURE,
            ),
            dir::Error::ReceiverK { channel } => (
                FJ_FILM_RECIPE_RECEIVER_K,
                channel as u32,
                u64::MAX,
                FJ_STATUS_PREPARATION_FAILURE,
            ),
            dir::Error::DonorK { channel } => (
                FJ_FILM_RECIPE_DONOR_K,
                channel as u32,
                u64::MAX,
                FJ_STATUS_PREPARATION_FAILURE,
            ),
            dir::Error::DonorLangmuir { channel, sample } => (
                FJ_FILM_RECIPE_DONOR_LANGMUIR,
                channel as u32,
                sample as u64,
                FJ_STATUS_PREPARATION_FAILURE,
            ),
            dir::Error::ReceiverLangmuir { channel, sample } => (
                FJ_FILM_RECIPE_RECEIVER_LANGMUIR,
                channel as u32,
                sample as u64,
                FJ_STATUS_PREPARATION_FAILURE,
            ),
            dir::Error::CompensatedAxis { channel, sample } => (
                FJ_FILM_RECIPE_COMPENSATED_AXIS,
                channel as u32,
                sample as u64,
                FJ_STATUS_PREPARATION_FAILURE,
            ),
            dir::Error::Size => (
                FJ_FILM_RECIPE_SIZE,
                3,
                u64::MAX,
                FJ_STATUS_ALLOCATION_FAILURE,
            ),
            dir::Error::Allocation => (
                FJ_FILM_RECIPE_ALLOCATION,
                3,
                u64::MAX,
                FJ_STATUS_ALLOCATION_FAILURE,
            ),
            dir::Error::ZeroIdentity => (
                FJ_FILM_RECIPE_IDENTITY,
                3,
                u64::MAX,
                FJ_STATUS_PREPARATION_FAILURE,
            ),
        };
        Self {
            category,
            record: FjFilmRecipeFailure {
                reason,
                channel,
                sample,
            },
        }
    }
    fn grain(error: grain::Error) -> Self {
        Self::new(
            FJ_STATUS_PREPARATION_FAILURE,
            match error {
                grain::Error::Controls => FJ_FILM_RECIPE_GRAIN_CONTROLS,
                grain::Error::MissingLayers => FJ_FILM_RECIPE_GRAIN_LAYERS,
                grain::Error::ZeroIdentity => FJ_FILM_RECIPE_IDENTITY,
            },
        )
    }
    fn density(error: density::Error) -> Self {
        let mut failure = Self::new(
            FJ_STATUS_PREPARATION_FAILURE,
            match error {
                density::Error::Minimum => FJ_FILM_RECIPE_MINIMUM,
                density::Error::Span { .. } => FJ_FILM_RECIPE_BOUNDS_SPAN,
                density::Error::ZeroIdentity => FJ_FILM_RECIPE_IDENTITY,
            },
        );
        if let density::Error::Span { channel } = error {
            failure.record.channel = channel as u32;
        }
        failure
    }
}

/// # Safety
/// T is one of this module's plain numerical/nullable-pointer wire results, valid
/// when zeroed. Caller authorizes live initialized output/failure/diagnostic
/// extents, mutually disjoint and disjoint from all reachable input/owner storage.
unsafe fn execute<T>(
    out: *mut T,
    failure: *mut FjFilmRecipeFailure,
    error: *mut FjErrorBuffer,
    operation: impl FnOnce() -> Result<T, Failure>,
) -> FjStatus {
    let valid_out = valid_extent(out, 1);
    let valid_failure = valid_extent(failure, 1);
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Each independently authorized result is cleared, including nullable spans.
        unsafe {
            if valid_out {
                out.write_bytes(0, 1);
            }
            if valid_failure {
                failure.write(FjFilmRecipeFailure {
                    reason: FJ_FILM_RECIPE_REASON_NONE,
                    channel: 3,
                    sample: u64::MAX,
                });
            }
        }
        // SAFETY: Same exclusive diagnostic contract, admitted before semantic input.
        let diagnostic = unsafe { Diagnostic::admit(error) };
        let rejected = diagnostic.is_err() || !valid_out || !valid_failure;
        let mut diagnostic = diagnostic.ok();
        let value = if rejected {
            Err(Failure::raw())
        } else {
            operation()
        };
        match value {
            Ok(value) => {
                // SAFETY: Complete result and success metadata commit only to admitted outputs.
                unsafe {
                    out.write(value);
                    failure.write(FjFilmRecipeFailure {
                        reason: FJ_FILM_RECIPE_REASON_NONE,
                        channel: 3,
                        sample: u64::MAX,
                    });
                }
                status(FJ_STATUS_SUCCESS)
            }
            Err(rejection) => {
                if let Some(writer) = &mut diagnostic {
                    #[cfg(feature = "test-support")]
                    if faults::take(4) == 1 {
                        panic!("film recipe diagnostic fixture panic");
                    }
                    let _ = write!(
                        writer,
                        "film recipe reason={} channel={} sample={}",
                        rejection.record.reason, rejection.record.channel, rejection.record.sample
                    );
                }
                if valid_failure {
                    // SAFETY: Independent valid failure storage receives the closed typed error.
                    unsafe {
                        failure.write(rejection.record);
                    }
                }
                status(rejection.category)
            }
        }
    }));
    result.unwrap_or_else(|_| {
        // SAFETY: These extents were admitted independently; panic cannot publish partial output.
        unsafe {
            if valid_out {
                out.write_bytes(0, 1);
            }
            if valid_failure {
                failure
                    .write(Failure::new(FJ_STATUS_INTERNAL_FAILURE, FJ_FILM_RECIPE_PANIC).record);
            }
        }
        status(FJ_STATUS_INTERNAL_FAILURE)
    })
}

impl FjVisualGrainParams {
    fn params(&self) -> Result<grain::Params, Failure> {
        if self.active > 1 || self.sublayers_active > 1 {
            return Err(Failure::raw());
        }
        Ok(grain::Params {
            active: self.active != 0,
            sublayers_active: self.sublayers_active != 0,
            particle_area_um2: self.particle_area_um2,
            particle_scale_cmy: self.particle_scale_cmy,
            particle_scale_layers: self.particle_scale_layers,
            particle_density_min_cmy: self.particle_density_min_cmy,
            uniformity_cmy: self.uniformity_cmy,
            correlation_sigma_px: self.correlation_sigma_px,
            dye_cloud_blur_um: self.dye_cloud_blur_um,
            micro_structure: self.micro_structure,
            sublayer_count: self.sublayer_count,
            amplitude: self.amplitude,
            chroma_mix: self.chroma_mix,
            chroma_shared_weight: self.chroma_shared_weight,
            chroma_independent_weight: self.chroma_independent_weight,
            coarse_weight: self.coarse_weight,
            mid_weight: self.mid_weight,
            size_mix_scale: self.size_mix_scale,
            clump_temporal_mix: self.clump_temporal_mix,
            clump_morph_period_sec: self.clump_morph_period_sec,
            debug_view: self.debug_view,
        })
    }
}
impl FjVisualGrainRecipe {
    fn new(recipe: &grain::Recipe) -> Self {
        Self {
            active: u32::from(recipe.active()),
            sublayers_active: u32::from(recipe.sublayers_active()),
            axis_finite_cmy: (*recipe.axis_finite_cmy()).map(u32::from),
            axis_block_prefix_max_cmy: *recipe.axis_block_prefix_max_cmy(),
            particle_area_um2: recipe.particle_area_um2(),
            particle_scale_cmy: *recipe.particle_scale_cmy(),
            particle_scale_layers: *recipe.particle_scale_layers(),
            particle_density_min_cmy: *recipe.particle_density_min_cmy(),
            uniformity_cmy: *recipe.uniformity_cmy(),
            correlation_sigma_px: recipe.correlation_sigma_px(),
            dye_cloud_blur_um: recipe.dye_cloud_blur_um(),
            micro_structure: *recipe.micro_structure(),
            sublayer_count: recipe.sublayer_count(),
            amplitude: recipe.amplitude(),
            chroma_mix: recipe.chroma_mix(),
            chroma_shared_weight: recipe.chroma_shared_weight(),
            chroma_independent_weight: recipe.chroma_independent_weight(),
            fine_weight: recipe.fine_weight(),
            mid_weight: recipe.mid_weight(),
            coarse_weight: recipe.coarse_weight(),
            size_mix_scale: recipe.size_mix_scale(),
            clump_temporal_mix: recipe.clump_temporal_mix(),
            clump_morph_period_sec: recipe.clump_morph_period_sec(),
            debug_view: recipe.debug_view(),
            layers_hash: recipe.layers_hash(),
            hash: recipe.hash(),
        }
    }
}

/// # Safety
/// One matching immutable live Development hold, borrowed only through return.
/// Initialized aligned readable params and exclusive writable outputs/diagnostics
/// have checked extents, are disjoint, and exclude concurrent owner release.
pub(crate) unsafe fn prepare_grain(
    development: *const FjFilmDevelopmentOwner,
    params: *const FjVisualGrainParams,
    out: *mut FjVisualGrainRecipe,
    failure: *mut FjFilmRecipeFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // This const-initialized Cell has no destructor; consume the test-only fault
    // once at entry, including calls rejected during result/diagnostic admission.
    #[cfg(feature = "test-support")]
    let fault = faults::take(7);
    // SAFETY: execute admits each result independently before any input borrow.
    unsafe {
        execute(out, failure, error, || {
            let development = development.cast::<Development>();
            if !valid_extent(development, 1) || !valid_extent(params, 1) {
                return Err(Failure::raw());
            }
            let params = (*params).params()?;
            #[cfg(feature = "test-support")]
            if fault == 1 {
                panic!("visual grain fixture panic");
            }
            #[cfg(feature = "test-support")]
            if fault == 2 {
                grain::test_support::zero_next_identity();
            }
            let recipe = grain::Recipe::new(&*development, params).map_err(Failure::grain)?;
            Ok(FjVisualGrainRecipe::new(&recipe))
        })
    }
}

/// # Safety
/// Matching live Development and initialized aligned disjoint call-local records.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_visual_grain_prepare(
    development: *const FjFilmDevelopmentOwner,
    params: *const FjVisualGrainParams,
    out_recipe: *mut FjVisualGrainRecipe,
    out_failure: *mut FjFilmRecipeFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Forward the identical synchronous raw contract without retaining a borrow.
    unsafe { prepare_grain(development, params, out_recipe, out_failure, error) }
}
const _: unsafe extern "C" fn(
    *const FjFilmDevelopmentOwner,
    *const FjVisualGrainParams,
    *mut FjVisualGrainRecipe,
    *mut FjFilmRecipeFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_legacy_visual_grain_prepare;
const _: () =
    assert!(size_of::<FjVisualGrainParams>() == 120 && align_of::<FjVisualGrainParams>() == 4);
const _: () = assert!(
    size_of::<FjVisualGrainRecipe>() == 344
        && align_of::<FjVisualGrainRecipe>() == 8
        && std::mem::offset_of!(FjVisualGrainRecipe, layers_hash) == 328
        && std::mem::offset_of!(FjVisualGrainRecipe, hash) == 336
);

impl FjFilmDirParams {
    fn params(&self) -> Result<dir::Params, Failure> {
        if self.active > 1 || self.gamma_use_stock > 1 {
            return Err(Failure::raw());
        }
        Ok(dir::Params {
            active: self.active != 0,
            amount: self.amount,
            inhibition_same_layer: self.inhibition_same_layer,
            inhibition_interlayer: self.inhibition_interlayer,
            diffusion_size_um: self.diffusion_size_um,
            diffusion_tail_um: self.diffusion_tail_um,
            diffusion_tail_weight: self.diffusion_tail_weight,
            gamma_use_stock: self.gamma_use_stock != 0,
            gamma_same_layer_rgb: self.gamma_same_layer_rgb,
            gamma_interlayer_r_to_gb: self.gamma_interlayer_r_to_gb,
            gamma_interlayer_g_to_rb: self.gamma_interlayer_g_to_rb,
            gamma_interlayer_b_to_rg: self.gamma_interlayer_b_to_rg,
            langmuir_donor_k_rgb: self.langmuir_donor_k_rgb,
            langmuir_receiver_k_rgb: self.langmuir_receiver_k_rgb,
        })
    }
}
impl FjFilmDirView {
    fn new(recipe: &dir::Recipe) -> Self {
        Self {
            polarity: match recipe.polarity() {
                Some(Polarity::Negative) => 0,
                Some(Polarity::Positive) => 1,
                None => 2,
            },
            mode: match recipe.mode() {
                dir::Mode::Inactive => FJ_FILM_DIR_INACTIVE,
                dir::Mode::NegativeDonorLangmuir => FJ_FILM_DIR_NEGATIVE_DONOR_LANGMUIR,
                dir::Mode::PositiveReceiverLangmuir => FJ_FILM_DIR_POSITIVE_RECEIVER_LANGMUIR,
            },
            matrix_rgb: std::array::from_fn(|i| recipe.matrix_rgb()[i / 3][i % 3]),
            diffusion_size_um: recipe.diffusion_size_um(),
            diffusion_tail_um: recipe.diffusion_tail_um(),
            diffusion_tail_weight: recipe.diffusion_tail_weight(),
            density_max_rgb: *recipe.density_max_rgb(),
            density_ref_rgb: *recipe.density_ref_rgb(),
            donor_k_rgb: *recipe.donor_k_rgb(),
            receiver_c_ref_rgb: *recipe.receiver_c_ref_rgb(),
            receiver_kr_rgb: *recipe.receiver_kr_rgb(),
            compensated_axes_rgb: std::array::from_fn(|i| {
                let axis = &recipe.compensated_axes_rgb()[i];
                FjFloatSpan {
                    data: if axis.is_empty() {
                        std::ptr::null()
                    } else {
                        axis.as_ptr()
                    },
                    count: axis.len(),
                }
            }),
            compensated_axes_hash: recipe.compensated_axes_hash(),
            hash: recipe.hash(),
        }
    }
}
impl FjFilmDensityBounds {
    fn new(bounds: &density::Bounds) -> Self {
        Self {
            route: match bounds.route() {
                ScanRoute::NegativeDirect => 0,
                ScanRoute::NegativePrint => 1,
                ScanRoute::PositiveDirect => 2,
                ScanRoute::PositivePrint => 3,
            },
            medium: 0,
            polarity: match bounds.polarity() {
                Polarity::Negative => 0,
                Polarity::Positive => 1,
            },
            source: match bounds.source() {
                density::Source::DirectFilm => 0,
                density::Source::EnlargerFilm => 1,
            },
            minimum_cmy: *bounds.minimum_cmy(),
            maximum_cmy: *bounds.maximum_cmy(),
            inv_span_cmy: *bounds.inv_span_cmy(),
            authored_minimum_cmy: *bounds.authored_minimum_cmy(),
            authored_maximum_cmy: *bounds.authored_maximum_cmy(),
            hash: bounds.hash(),
        }
    }
}

/// # Safety
/// Initialized/disjoint live records/backing with caller-authorized extents.
pub(crate) unsafe fn minimum(
    requested: *const FjFilmDensityMinimum,
    out: *mut FjFilmDensityMinimum,
    failure: *mut FjFilmRecipeFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: All records follow the immediate synchronous execute contract.
    unsafe {
        execute(out, failure, error, || {
            if !valid_extent(requested, 1) {
                return Err(Failure::raw());
            }
            let minimum = density::Minimum::new((*requested).cmy).map_err(Failure::density)?;
            Ok(FjFilmDensityMinimum {
                cmy: *minimum.cmy(),
            })
        })
    }
}
/// # Safety
/// Live development hold and initialized/disjoint records/backing until return.
pub(crate) unsafe fn acquire(
    development: *const FjFilmDevelopmentOwner,
    params: *const FjFilmDirParams,
    out: *mut *mut FjFilmDirOwner,
    failure: *mut FjFilmRecipeFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Borrow the completed development only after raw extent admission.
    unsafe {
        execute(out, failure, error, || {
            let development = development.cast::<Development>();
            if !valid_extent(development, 1) || !valid_extent(params, 1) {
                return Err(Failure::raw());
            }
            let params = (*params).params()?;
            #[cfg(feature = "test-support")]
            if faults::take(0) == 1 {
                panic!("film DIR acquire fixture panic");
            }
            let recipe = dir::Recipe::new(&*development, params).map_err(Failure::dir)?;
            let owner = Box::new(recipe);
            #[cfg(feature = "test-support")]
            faults::LIVE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // No fallible work follows raw ownership transfer before execute commits it.
            Ok(Box::into_raw(owner).cast())
        })
    }
}
/// # Safety
/// A matching live immutable DIR owner; exclusive disjoint output/diagnostics.
pub(crate) unsafe fn view(
    owner: *const FjFilmDirOwner,
    out: *mut FjFilmDirView,
    failure: *mut FjFilmRecipeFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The borrow expires at consuming release; no reference escapes this call.
    unsafe {
        execute(out, failure, error, || {
            let recipe = owner.cast::<dir::Recipe>();
            if !valid_extent(recipe, 1) {
                return Err(Failure::raw());
            }
            #[cfg(feature = "test-support")]
            if faults::take(1) == 1 {
                panic!("film DIR view fixture panic");
            }
            Ok(FjFilmDirView::new(&*recipe))
        })
    }
}
/// # Safety
/// Matching NULL/live DIR owner, consumed once, excluding every read/view.
/// Diagnostics obey the existing asset-owner family's optional-buffer contract.
pub(crate) unsafe fn release(owner: *mut FjFilmDirOwner, error: *mut FjErrorBuffer) -> FjStatus {
    if !owner.is_null() && !valid_extent(owner.cast::<dir::Recipe>(), 1) {
        // SAFETY: A numerically malformed non-handle owns nothing. Diagnostics
        // retain the existing owner-family optional-buffer contract.
        return unsafe {
            crate::asset_bridge::run(error, || {
                Err::<(), _>(crate::asset_bridge::Failure::Input(
                    "malformed film DIR release owner",
                ))
            })
        }
        .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS));
    }
    let owned = if owner.is_null() {
        None
    } else {
        #[cfg(feature = "test-support")]
        faults::LIVE.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        // SAFETY: The caller transfers one matching Box before diagnostic rejection.
        Some(unsafe { Box::from_raw(owner.cast::<dir::Recipe>()) })
    };
    // SAFETY: Captured ownership is destroyed inside containment even when diagnostics reject.
    unsafe {
        crate::asset_bridge::run(error, move || {
            #[cfg(feature = "test-support")]
            if faults::take(2) == 1 {
                panic!("film DIR release fixture panic");
            }
            drop(owned);
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}
/// # Safety
/// Live development and unchanged original minimum request; initialized/disjoint
/// exclusive outputs/diagnostics. No owner or input span is retained.
pub(crate) unsafe fn bounds(
    development: *const FjFilmDevelopmentOwner,
    requested: *const FjFilmDensityMinimum,
    out: *mut FjFilmDensityBounds,
    failure: *mut FjFilmRecipeFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Fresh raw request is admitted through the one safe minimum constructor.
    unsafe {
        execute(out, failure, error, || {
            let development = development.cast::<Development>();
            if !valid_extent(development, 1) || !valid_extent(requested, 1) {
                return Err(Failure::raw());
            }
            let minimum = density::Minimum::new((*requested).cmy).map_err(Failure::density)?;
            #[cfg(feature = "test-support")]
            if faults::take(3) == 1 {
                panic!("film density bounds fixture panic");
            }
            let bounds = density::Bounds::new(&*development, &minimum).map_err(Failure::density)?;
            Ok(FjFilmDensityBounds::new(&bounds))
        })
    }
}

#[cfg(feature = "test-support")]
pub(crate) mod faults {
    use std::cell::Cell;
    use std::sync::atomic::AtomicUsize;
    thread_local! {pub(super) static NEXT:Cell<(u32,u32)>=const{Cell::new((0,0))};}
    pub(super) static LIVE: AtomicUsize = AtomicUsize::new(0);
    pub(crate) fn take(stage: u32) -> u32 {
        let (selected, fault) = NEXT.get();
        if selected == stage {
            NEXT.set((0, 0));
            fault
        } else {
            0
        }
    }
}

#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
extern "C" fn fj_test_film_recipe_fault(stage: u32, fault: u32) -> FjStatus {
    if stage == 5 && fault <= 3 {
        dir::test_support::reject_next(fault);
        return status(FJ_STATUS_SUCCESS);
    }
    if (stage == 10 || stage == 11) && fault <= 1 {
        faults::NEXT.set((stage, fault));
        return status(FJ_STATUS_SUCCESS);
    }
    if stage == 9 && fault <= 2 {
        faults::NEXT.set((stage, fault));
        return status(FJ_STATUS_SUCCESS);
    }
    if stage == 8 && fault <= 4 {
        faults::NEXT.set((stage, fault));
        return status(FJ_STATUS_SUCCESS);
    }
    if stage == 7 && fault <= 2 {
        faults::NEXT.set((stage, fault));
        return status(FJ_STATUS_SUCCESS);
    }
    if stage == 6 && fault == 1 {
        density::test_support::zero_next_identity();
        return status(FJ_STATUS_SUCCESS);
    }
    if stage > 4 || fault > 1 {
        return status(FJ_STATUS_UNSUPPORTED_INPUT);
    }
    faults::NEXT.set((stage, fault));
    status(FJ_STATUS_SUCCESS)
}
#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
extern "C" fn fj_test_film_dir_live_owners() -> usize {
    faults::LIVE.load(std::sync::atomic::Ordering::Relaxed)
}

/// # Safety
/// Matching live owners; caller-authorized initialized/aligned/disjoint input, output and diagnostic extents.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_film_density_minimum_prepare(
    requested: *const FjFilmDensityMinimum,
    out_minimum: *mut FjFilmDensityMinimum,
    out_failure: *mut FjFilmRecipeFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Forward the identical immediate raw call contract.
    unsafe { minimum(requested, out_minimum, out_failure, error) }
}

/// # Safety
/// Matching live owners; caller-authorized initialized/aligned/disjoint input, output and diagnostic extents.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_film_dir_acquire(
    development: *const FjFilmDevelopmentOwner,
    params: *const FjFilmDirParams,
    out_dir: *mut *mut FjFilmDirOwner,
    out_failure: *mut FjFilmRecipeFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Forward the identical immediate raw call contract.
    unsafe { acquire(development, params, out_dir, out_failure, error) }
}

/// # Safety
/// Matching live owners; caller-authorized initialized/aligned/disjoint input, output and diagnostic extents.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_film_dir_view(
    dir: *const FjFilmDirOwner,
    out_view: *mut FjFilmDirView,
    out_failure: *mut FjFilmRecipeFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Forward the identical immediate raw call contract.
    unsafe { view(dir, out_view, out_failure, error) }
}

/// # Safety
/// Matching live owners; caller-authorized initialized/aligned/disjoint input, output and diagnostic extents.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_film_dir_release(
    dir: *mut FjFilmDirOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Forward the identical immediate raw call contract.
    unsafe { release(dir, error) }
}

/// # Safety
/// Matching live owners; caller-authorized initialized/aligned/disjoint input, output and diagnostic extents.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_film_density_bounds_prepare(
    development: *const FjFilmDevelopmentOwner,
    requested: *const FjFilmDensityMinimum,
    out_bounds: *mut FjFilmDensityBounds,
    out_failure: *mut FjFilmRecipeFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Forward the identical immediate raw call contract.
    unsafe { bounds(development, requested, out_bounds, out_failure, error) }
}

// The closed wire contract reserves integrity for detecting impossible envelopes.
const _: () = assert!(FJ_FILM_RECIPE_INTEGRITY == 15);

#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
/// # Safety
/// Count is NULL or a live aligned exclusively writable usize slot. Returned
/// facts have immutable static lifetime and cannot be freed by the caller.
unsafe extern "C" fn fj_test_film_recipe_abi_facts(count: *mut usize) -> *const usize {
    const FACTS: &[usize] = &[
        std::mem::size_of::<FjFilmRecipeFailure>(),
        std::mem::align_of::<FjFilmRecipeFailure>(),
        std::mem::offset_of!(FjFilmRecipeFailure, reason),
        std::mem::offset_of!(FjFilmRecipeFailure, channel),
        std::mem::offset_of!(FjFilmRecipeFailure, sample),
        std::mem::size_of::<FjFilmDensityMinimum>(),
        std::mem::align_of::<FjFilmDensityMinimum>(),
        std::mem::offset_of!(FjFilmDensityMinimum, cmy),
        std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
        std::mem::size_of::<FjFilmDirParams>(),
        std::mem::align_of::<FjFilmDirParams>(),
        std::mem::offset_of!(FjFilmDirParams, active),
        std::mem::offset_of!(FjFilmDirParams, amount),
        std::mem::offset_of!(FjFilmDirParams, inhibition_same_layer),
        std::mem::offset_of!(FjFilmDirParams, inhibition_interlayer),
        std::mem::offset_of!(FjFilmDirParams, diffusion_size_um),
        std::mem::offset_of!(FjFilmDirParams, diffusion_tail_um),
        std::mem::offset_of!(FjFilmDirParams, diffusion_tail_weight),
        std::mem::offset_of!(FjFilmDirParams, gamma_use_stock),
        std::mem::offset_of!(FjFilmDirParams, gamma_same_layer_rgb),
        std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDirParams, gamma_interlayer_r_to_gb),
        std::mem::size_of::<[f32; 2]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDirParams, gamma_interlayer_g_to_rb),
        std::mem::size_of::<[f32; 2]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDirParams, gamma_interlayer_b_to_rg),
        std::mem::size_of::<[f32; 2]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDirParams, langmuir_donor_k_rgb),
        std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDirParams, langmuir_receiver_k_rgb),
        std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
        std::mem::size_of::<FjFilmDirView>(),
        std::mem::align_of::<FjFilmDirView>(),
        std::mem::offset_of!(FjFilmDirView, polarity),
        std::mem::offset_of!(FjFilmDirView, mode),
        std::mem::offset_of!(FjFilmDirView, matrix_rgb),
        std::mem::size_of::<[f32; 9]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDirView, diffusion_size_um),
        std::mem::offset_of!(FjFilmDirView, diffusion_tail_um),
        std::mem::offset_of!(FjFilmDirView, diffusion_tail_weight),
        std::mem::offset_of!(FjFilmDirView, density_max_rgb),
        std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDirView, density_ref_rgb),
        std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDirView, donor_k_rgb),
        std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDirView, receiver_c_ref_rgb),
        std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDirView, receiver_kr_rgb),
        std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDirView, compensated_axes_rgb),
        std::mem::size_of::<[FjFloatSpan; 3]>() / std::mem::size_of::<FjFloatSpan>(),
        std::mem::offset_of!(FjFilmDirView, compensated_axes_hash),
        std::mem::offset_of!(FjFilmDirView, hash),
        std::mem::size_of::<FjFilmDensityBounds>(),
        std::mem::align_of::<FjFilmDensityBounds>(),
        std::mem::offset_of!(FjFilmDensityBounds, route),
        std::mem::offset_of!(FjFilmDensityBounds, medium),
        std::mem::offset_of!(FjFilmDensityBounds, polarity),
        std::mem::offset_of!(FjFilmDensityBounds, source),
        std::mem::offset_of!(FjFilmDensityBounds, minimum_cmy),
        std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDensityBounds, maximum_cmy),
        std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDensityBounds, inv_span_cmy),
        std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDensityBounds, authored_minimum_cmy),
        std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDensityBounds, authored_maximum_cmy),
        std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDensityBounds, hash),
        FJ_FILM_DIR_INACTIVE as usize,
        FJ_FILM_DIR_NEGATIVE_DONOR_LANGMUIR as usize,
        FJ_FILM_DIR_POSITIVE_RECEIVER_LANGMUIR as usize,
        FJ_FILM_RECIPE_REASON_NONE as usize,
        FJ_FILM_RECIPE_RAW as usize,
        FJ_FILM_RECIPE_MINIMUM as usize,
        FJ_FILM_RECIPE_MATRIX as usize,
        FJ_FILM_RECIPE_DENSITY_MAX as usize,
        FJ_FILM_RECIPE_RECEIVER_K as usize,
        FJ_FILM_RECIPE_DONOR_K as usize,
        FJ_FILM_RECIPE_DONOR_LANGMUIR as usize,
        FJ_FILM_RECIPE_RECEIVER_LANGMUIR as usize,
        FJ_FILM_RECIPE_COMPENSATED_AXIS as usize,
        FJ_FILM_RECIPE_IDENTITY as usize,
        FJ_FILM_RECIPE_BOUNDS_SPAN as usize,
        FJ_FILM_RECIPE_SIZE as usize,
        FJ_FILM_RECIPE_ALLOCATION as usize,
        FJ_FILM_RECIPE_PANIC as usize,
        FJ_FILM_RECIPE_INTEGRITY as usize,
        FJ_FILM_RECIPE_LOG_EXPOSURE as usize,
        size_of::<FjVisualGrainParams>(),
        align_of::<FjVisualGrainParams>(),
        std::mem::offset_of!(FjVisualGrainParams, active),
        std::mem::offset_of!(FjVisualGrainParams, sublayers_active),
        std::mem::offset_of!(FjVisualGrainParams, particle_area_um2),
        std::mem::offset_of!(FjVisualGrainParams, particle_scale_cmy),
        size_of::<[f32; 3]>() / size_of::<f32>(),
        std::mem::offset_of!(FjVisualGrainParams, particle_scale_layers),
        size_of::<[f32; 3]>() / size_of::<f32>(),
        std::mem::offset_of!(FjVisualGrainParams, particle_density_min_cmy),
        size_of::<[f32; 3]>() / size_of::<f32>(),
        std::mem::offset_of!(FjVisualGrainParams, uniformity_cmy),
        size_of::<[f32; 3]>() / size_of::<f32>(),
        std::mem::offset_of!(FjVisualGrainParams, correlation_sigma_px),
        std::mem::offset_of!(FjVisualGrainParams, dye_cloud_blur_um),
        std::mem::offset_of!(FjVisualGrainParams, micro_structure),
        size_of::<[f32; 2]>() / size_of::<f32>(),
        std::mem::offset_of!(FjVisualGrainParams, sublayer_count),
        std::mem::offset_of!(FjVisualGrainParams, amplitude),
        std::mem::offset_of!(FjVisualGrainParams, chroma_mix),
        std::mem::offset_of!(FjVisualGrainParams, chroma_shared_weight),
        std::mem::offset_of!(FjVisualGrainParams, chroma_independent_weight),
        std::mem::offset_of!(FjVisualGrainParams, coarse_weight),
        std::mem::offset_of!(FjVisualGrainParams, mid_weight),
        std::mem::offset_of!(FjVisualGrainParams, size_mix_scale),
        std::mem::offset_of!(FjVisualGrainParams, clump_temporal_mix),
        std::mem::offset_of!(FjVisualGrainParams, clump_morph_period_sec),
        std::mem::offset_of!(FjVisualGrainParams, debug_view),
        size_of::<FjVisualGrainRecipe>(),
        align_of::<FjVisualGrainRecipe>(),
        std::mem::offset_of!(FjVisualGrainRecipe, active),
        std::mem::offset_of!(FjVisualGrainRecipe, sublayers_active),
        std::mem::offset_of!(FjVisualGrainRecipe, axis_finite_cmy),
        size_of::<[u32; 3]>() / size_of::<u32>(),
        std::mem::offset_of!(FjVisualGrainRecipe, axis_block_prefix_max_cmy),
        size_of::<[[f32; 16]; 3]>() / size_of::<[f32; 16]>(),
        size_of::<[f32; 16]>() / size_of::<f32>(),
        std::mem::offset_of!(FjVisualGrainRecipe, particle_area_um2),
        std::mem::offset_of!(FjVisualGrainRecipe, particle_scale_cmy),
        size_of::<[f32; 3]>() / size_of::<f32>(),
        std::mem::offset_of!(FjVisualGrainRecipe, particle_scale_layers),
        size_of::<[f32; 3]>() / size_of::<f32>(),
        std::mem::offset_of!(FjVisualGrainRecipe, particle_density_min_cmy),
        size_of::<[f32; 3]>() / size_of::<f32>(),
        std::mem::offset_of!(FjVisualGrainRecipe, uniformity_cmy),
        size_of::<[f32; 3]>() / size_of::<f32>(),
        std::mem::offset_of!(FjVisualGrainRecipe, correlation_sigma_px),
        std::mem::offset_of!(FjVisualGrainRecipe, dye_cloud_blur_um),
        std::mem::offset_of!(FjVisualGrainRecipe, micro_structure),
        size_of::<[f32; 2]>() / size_of::<f32>(),
        std::mem::offset_of!(FjVisualGrainRecipe, sublayer_count),
        std::mem::offset_of!(FjVisualGrainRecipe, amplitude),
        std::mem::offset_of!(FjVisualGrainRecipe, chroma_mix),
        std::mem::offset_of!(FjVisualGrainRecipe, chroma_shared_weight),
        std::mem::offset_of!(FjVisualGrainRecipe, chroma_independent_weight),
        std::mem::offset_of!(FjVisualGrainRecipe, fine_weight),
        std::mem::offset_of!(FjVisualGrainRecipe, mid_weight),
        std::mem::offset_of!(FjVisualGrainRecipe, coarse_weight),
        std::mem::offset_of!(FjVisualGrainRecipe, size_mix_scale),
        std::mem::offset_of!(FjVisualGrainRecipe, clump_temporal_mix),
        std::mem::offset_of!(FjVisualGrainRecipe, clump_morph_period_sec),
        std::mem::offset_of!(FjVisualGrainRecipe, debug_view),
        std::mem::offset_of!(FjVisualGrainRecipe, layers_hash),
        std::mem::offset_of!(FjVisualGrainRecipe, hash),
        FJ_FILM_RECIPE_GRAIN_CONTROLS as usize,
        FJ_FILM_RECIPE_GRAIN_LAYERS as usize,
        size_of::<FjPhysicalEffectsParams>(),
        align_of::<FjPhysicalEffectsParams>(),
        std::mem::offset_of!(FjPhysicalEffectsParams, film_dust_amount),
        std::mem::offset_of!(FjPhysicalEffectsParams, film_scratch_amount),
        std::mem::offset_of!(FjPhysicalEffectsParams, gate_dust_amount),
        std::mem::offset_of!(FjPhysicalEffectsParams, gate_scratch_amount),
        std::mem::offset_of!(FjPhysicalEffectsParams, gate_weave_amount),
        std::mem::offset_of!(FjPhysicalEffectsParams, grain_active),
        std::mem::offset_of!(FjPhysicalEffectsParams, grain_debug_view),
        size_of::<FjPhysicalEffectsRecipe>(),
        align_of::<FjPhysicalEffectsRecipe>(),
        std::mem::offset_of!(FjPhysicalEffectsRecipe, film_dust),
        std::mem::offset_of!(FjPhysicalEffectsRecipe, film_scratch),
        std::mem::offset_of!(FjPhysicalEffectsRecipe, gate_dust),
        std::mem::offset_of!(FjPhysicalEffectsRecipe, gate_scratch),
        std::mem::offset_of!(FjPhysicalEffectsRecipe, gate_weave_amount),
        std::mem::offset_of!(FjPhysicalEffectsRecipe, active),
        std::mem::offset_of!(FjPhysicalEffectsRecipe, hash),
        size_of::<FjDust>(),
        align_of::<FjDust>(),
        std::mem::offset_of!(FjDust, cell_width_mm),
        std::mem::offset_of!(FjDust, cell_height_mm),
        std::mem::offset_of!(FjDust, slot_probability),
        std::mem::offset_of!(FjDust, softness_min_mm),
        std::mem::offset_of!(FjDust, softness_max_mm),
        std::mem::offset_of!(FjDust, softness_size_cap_fraction),
        std::mem::offset_of!(FjDust, support_x_mm),
        std::mem::offset_of!(FjDust, support_y_mm),
        std::mem::offset_of!(FjDust, fiber_fraction),
        std::mem::offset_of!(FjDust, fiber_drift_fraction),
        std::mem::offset_of!(FjDust, fiber_first_knot_min),
        std::mem::offset_of!(FjDust, fiber_first_knot_max),
        std::mem::offset_of!(FjDust, fiber_second_knot_min),
        std::mem::offset_of!(FjDust, fiber_second_knot_max),
        std::mem::offset_of!(FjDust, fiber_interior_width_min_fraction),
        std::mem::offset_of!(FjDust, fiber_interior_width_max_fraction),
        std::mem::offset_of!(FjDust, diameter_min_mm),
        std::mem::offset_of!(FjDust, diameter_bulk_max_mm),
        std::mem::offset_of!(FjDust, diameter_max_mm),
        std::mem::offset_of!(FjDust, diameter_tail_fraction),
        std::mem::offset_of!(FjDust, fiber_length_min_mm),
        std::mem::offset_of!(FjDust, fiber_length_max_mm),
        std::mem::offset_of!(FjDust, fiber_width_min_mm),
        std::mem::offset_of!(FjDust, fiber_width_max_mm),
        std::mem::offset_of!(FjDust, opacity_faint_cumulative),
        std::mem::offset_of!(FjDust, opacity_intermediate_cumulative),
        std::mem::offset_of!(FjDust, compact_opacity_min),
        std::mem::offset_of!(FjDust, compact_opacity_faint_end),
        std::mem::offset_of!(FjDust, compact_opacity_intermediate_end),
        std::mem::offset_of!(FjDust, compact_opacity_max),
        std::mem::offset_of!(FjDust, fiber_opacity_min),
        std::mem::offset_of!(FjDust, fiber_opacity_faint_end),
        std::mem::offset_of!(FjDust, fiber_opacity_intermediate_end),
        std::mem::offset_of!(FjDust, fiber_opacity_max),
        std::mem::offset_of!(FjDust, compact_dominant_aspect_min),
        std::mem::offset_of!(FjDust, compact_dominant_aspect_max),
        std::mem::offset_of!(FjDust, compact_subsidiary_scale_min),
        std::mem::offset_of!(FjDust, compact_subsidiary_scale_max),
        std::mem::offset_of!(FjDust, compact_subsidiary_aspect_min),
        std::mem::offset_of!(FjDust, compact_subsidiary_aspect_max),
        std::mem::offset_of!(FjDust, compact_subsidiary_offset_max),
        std::mem::offset_of!(FjDust, compact_subsidiary_angle_max_radians),
        size_of::<FjScratch>(),
        align_of::<FjScratch>(),
        std::mem::offset_of!(FjScratch, cell_width_mm),
        std::mem::offset_of!(FjScratch, cell_height_mm),
        std::mem::offset_of!(FjScratch, slot_probability),
        std::mem::offset_of!(FjScratch, softness_min_mm),
        std::mem::offset_of!(FjScratch, softness_max_mm),
        std::mem::offset_of!(FjScratch, softness_size_cap_fraction),
        std::mem::offset_of!(FjScratch, support_x_mm),
        std::mem::offset_of!(FjScratch, support_y_mm),
        std::mem::offset_of!(FjScratch, length_min_mm),
        std::mem::offset_of!(FjScratch, length_bulk_max_mm),
        std::mem::offset_of!(FjScratch, length_max_mm),
        std::mem::offset_of!(FjScratch, length_tail_fraction),
        std::mem::offset_of!(FjScratch, width_min_mm),
        std::mem::offset_of!(FjScratch, width_bulk_max_mm),
        std::mem::offset_of!(FjScratch, width_max_mm),
        std::mem::offset_of!(FjScratch, width_tail_fraction),
        std::mem::offset_of!(FjScratch, drift_fraction),
        std::mem::offset_of!(FjScratch, first_knot_min),
        std::mem::offset_of!(FjScratch, first_knot_max),
        std::mem::offset_of!(FjScratch, second_knot_min),
        std::mem::offset_of!(FjScratch, second_knot_max),
        std::mem::offset_of!(FjScratch, interior_width_min_fraction),
        std::mem::offset_of!(FjScratch, interior_width_max_fraction),
        std::mem::offset_of!(FjScratch, interior_depth_min_fraction),
        std::mem::offset_of!(FjScratch, interior_depth_max_fraction),
        std::mem::offset_of!(FjScratch, endpoint_abrupt_probability),
        std::mem::offset_of!(FjScratch, interruption_probability),
        std::mem::offset_of!(FjScratch, gap_center_min),
        std::mem::offset_of!(FjScratch, gap_center_max),
        std::mem::offset_of!(FjScratch, gap_span_min),
        std::mem::offset_of!(FjScratch, gap_span_max),
        std::mem::offset_of!(FjScratch, scuff_probability),
        std::mem::offset_of!(FjScratch, scuff_length_max_mm),
        std::mem::offset_of!(FjScratch, scuff_angle_max_radians),
        std::mem::offset_of!(FjScratch, strength_min),
        std::mem::offset_of!(FjScratch, strength_max),
        FJ_FILM_RECIPE_EFFECTS_PROBABILITY as usize,
        size_of::<crate::optics_bridge::FjDiffusionParams>(),
        align_of::<crate::optics_bridge::FjDiffusionParams>(),
        std::mem::offset_of!(crate::optics_bridge::FjDiffusionParams, active),
        std::mem::offset_of!(crate::optics_bridge::FjDiffusionParams, family),
        std::mem::offset_of!(crate::optics_bridge::FjDiffusionParams, strength),
        std::mem::offset_of!(crate::optics_bridge::FjDiffusionParams, spatial_scale),
        std::mem::offset_of!(crate::optics_bridge::FjDiffusionParams, halo_warmth),
        std::mem::offset_of!(crate::optics_bridge::FjDiffusionParams, core_intensity),
        std::mem::offset_of!(crate::optics_bridge::FjDiffusionParams, core_size),
        std::mem::offset_of!(crate::optics_bridge::FjDiffusionParams, halo_intensity),
        std::mem::offset_of!(crate::optics_bridge::FjDiffusionParams, halo_size),
        std::mem::offset_of!(crate::optics_bridge::FjDiffusionParams, bloom_intensity),
        std::mem::offset_of!(crate::optics_bridge::FjDiffusionParams, bloom_size),
        size_of::<crate::optics_bridge::FjHalationParams>(),
        align_of::<crate::optics_bridge::FjHalationParams>(),
        std::mem::offset_of!(crate::optics_bridge::FjHalationParams, active),
        std::mem::offset_of!(crate::optics_bridge::FjHalationParams, scatter_amount),
        std::mem::offset_of!(
            crate::optics_bridge::FjHalationParams,
            scatter_spatial_scale
        ),
        std::mem::offset_of!(crate::optics_bridge::FjHalationParams, halation_amount),
        std::mem::offset_of!(
            crate::optics_bridge::FjHalationParams,
            halation_spatial_scale
        ),
        std::mem::offset_of!(crate::optics_bridge::FjHalationParams, first_sigma_um),
        size_of::<[f32; 3]>() / size_of::<f32>(),
        std::mem::offset_of!(crate::optics_bridge::FjHalationParams, primary_amount),
        size_of::<[f32; 3]>() / size_of::<f32>(),
        size_of::<crate::optics_bridge::FjSpatialOpticsParams>(),
        align_of::<crate::optics_bridge::FjSpatialOpticsParams>(),
        std::mem::offset_of!(
            crate::optics_bridge::FjSpatialOpticsParams,
            camera_diffusion
        ),
        std::mem::offset_of!(
            crate::optics_bridge::FjSpatialOpticsParams,
            camera_lens_blur_um
        ),
        std::mem::offset_of!(
            crate::optics_bridge::FjSpatialOpticsParams,
            scatter_halation
        ),
        std::mem::offset_of!(
            crate::optics_bridge::FjSpatialOpticsParams,
            enlarger_diffusion
        ),
        std::mem::offset_of!(crate::optics_bridge::FjSpatialOpticsParams, print_domain),
        size_of::<crate::optics_bridge::FjDiffusionRecipe>(),
        align_of::<crate::optics_bridge::FjDiffusionRecipe>(),
        std::mem::offset_of!(crate::optics_bridge::FjDiffusionRecipe, domain),
        std::mem::offset_of!(crate::optics_bridge::FjDiffusionRecipe, active),
        std::mem::offset_of!(crate::optics_bridge::FjDiffusionRecipe, family),
        std::mem::offset_of!(crate::optics_bridge::FjDiffusionRecipe, scatter_fraction),
        std::mem::offset_of!(crate::optics_bridge::FjDiffusionRecipe, group_weights),
        size_of::<[f64; 3]>() / size_of::<f64>(),
        std::mem::offset_of!(crate::optics_bridge::FjDiffusionRecipe, group_lambda_um),
        size_of::<[f64; 3]>() / size_of::<f64>(),
        std::mem::offset_of!(crate::optics_bridge::FjDiffusionRecipe, warmth),
        std::mem::offset_of!(crate::optics_bridge::FjDiffusionRecipe, spatial_scale),
        std::mem::offset_of!(crate::optics_bridge::FjDiffusionRecipe, resolved_hash),
        std::mem::offset_of!(crate::optics_bridge::FjDiffusionRecipe, hash),
        size_of::<crate::optics_bridge::FjLensBlurRecipe>(),
        align_of::<crate::optics_bridge::FjLensBlurRecipe>(),
        std::mem::offset_of!(crate::optics_bridge::FjLensBlurRecipe, sigma_um),
        std::mem::offset_of!(crate::optics_bridge::FjLensBlurRecipe, hash),
        size_of::<crate::optics_bridge::FjHalationRecipe>(),
        align_of::<crate::optics_bridge::FjHalationRecipe>(),
        std::mem::offset_of!(crate::optics_bridge::FjHalationRecipe, scatter_active),
        std::mem::offset_of!(
            crate::optics_bridge::FjHalationRecipe,
            back_reflection_active
        ),
        std::mem::offset_of!(crate::optics_bridge::FjHalationRecipe, scatter_amount),
        std::mem::offset_of!(
            crate::optics_bridge::FjHalationRecipe,
            scatter_spatial_scale
        ),
        std::mem::offset_of!(
            crate::optics_bridge::FjHalationRecipe,
            halation_spatial_scale
        ),
        std::mem::offset_of!(crate::optics_bridge::FjHalationRecipe, first_sigma_um),
        size_of::<[f32; 3]>() / size_of::<f32>(),
        std::mem::offset_of!(crate::optics_bridge::FjHalationRecipe, total_strength),
        size_of::<[f32; 3]>() / size_of::<f32>(),
        std::mem::offset_of!(crate::optics_bridge::FjHalationRecipe, hash),
        size_of::<crate::optics_bridge::FjSpatialOpticsRecipe>(),
        align_of::<crate::optics_bridge::FjSpatialOpticsRecipe>(),
        std::mem::offset_of!(
            crate::optics_bridge::FjSpatialOpticsRecipe,
            camera_diffusion
        ),
        std::mem::offset_of!(
            crate::optics_bridge::FjSpatialOpticsRecipe,
            camera_lens_blur
        ),
        std::mem::offset_of!(
            crate::optics_bridge::FjSpatialOpticsRecipe,
            scatter_halation
        ),
        std::mem::offset_of!(
            crate::optics_bridge::FjSpatialOpticsRecipe,
            enlarger_diffusion
        ),
        std::mem::offset_of!(crate::optics_bridge::FjSpatialOpticsRecipe, hash),
        size_of::<crate::optics_bridge::FjSpatialOpticsFailure>(),
        align_of::<crate::optics_bridge::FjSpatialOpticsFailure>(),
        std::mem::offset_of!(crate::optics_bridge::FjSpatialOpticsFailure, reason),
        std::mem::offset_of!(crate::optics_bridge::FjSpatialOpticsFailure, component),
        std::mem::offset_of!(crate::optics_bridge::FjSpatialOpticsFailure, field),
        std::mem::offset_of!(crate::optics_bridge::FjSpatialOpticsFailure, channel),
        crate::optics_bridge::FJ_OPTICS_REASON_NONE as usize,
        crate::optics_bridge::FJ_OPTICS_REASON_RAW as usize,
        crate::optics_bridge::FJ_OPTICS_REASON_DIFFUSION as usize,
        crate::optics_bridge::FJ_OPTICS_REASON_LENS_BLUR as usize,
        crate::optics_bridge::FJ_OPTICS_REASON_HALATION as usize,
        crate::optics_bridge::FJ_OPTICS_REASON_PANIC as usize,
        crate::optics_bridge::FJ_OPTICS_COMPONENT_NONE as usize,
        crate::optics_bridge::FJ_OPTICS_COMPONENT_CAMERA_DIFFUSION as usize,
        crate::optics_bridge::FJ_OPTICS_COMPONENT_CAMERA_LENS_BLUR as usize,
        crate::optics_bridge::FJ_OPTICS_COMPONENT_SCATTER_HALATION as usize,
        crate::optics_bridge::FJ_OPTICS_COMPONENT_ENLARGER_DIFFUSION as usize,
        crate::optics_bridge::FJ_OPTICS_FIELD_NONE as usize,
        crate::optics_bridge::FJ_OPTICS_FIELD_FAMILY as usize,
        crate::optics_bridge::FJ_OPTICS_FIELD_STRENGTH as usize,
        crate::optics_bridge::FJ_OPTICS_FIELD_SPATIAL_SCALE as usize,
        crate::optics_bridge::FJ_OPTICS_FIELD_HALO_WARMTH as usize,
        crate::optics_bridge::FJ_OPTICS_FIELD_CORE_INTENSITY as usize,
        crate::optics_bridge::FJ_OPTICS_FIELD_CORE_SIZE as usize,
        crate::optics_bridge::FJ_OPTICS_FIELD_HALO_INTENSITY as usize,
        crate::optics_bridge::FJ_OPTICS_FIELD_HALO_SIZE as usize,
        crate::optics_bridge::FJ_OPTICS_FIELD_BLOOM_INTENSITY as usize,
        crate::optics_bridge::FJ_OPTICS_FIELD_BLOOM_SIZE as usize,
        crate::optics_bridge::FJ_OPTICS_FIELD_SCATTER_FRACTION as usize,
        crate::optics_bridge::FJ_OPTICS_FIELD_GROUP_WEIGHTS as usize,
        crate::optics_bridge::FJ_OPTICS_FIELD_GROUP_CENTERS as usize,
        crate::optics_bridge::FJ_OPTICS_FIELD_RESOLVED_HASH as usize,
        crate::optics_bridge::FJ_OPTICS_FIELD_SIGMA_UM as usize,
        crate::optics_bridge::FJ_OPTICS_FIELD_TOTAL_STRENGTH as usize,
        size_of::<crate::print_bridge::FjPrintIlluminantParams>(),
        align_of::<crate::print_bridge::FjPrintIlluminantParams>(),
        std::mem::offset_of!(crate::print_bridge::FjPrintIlluminantParams, edges_nm),
        4,
        std::mem::offset_of!(crate::print_bridge::FjPrintIlluminantParams, transitions_nm),
        4,
        std::mem::offset_of!(crate::print_bridge::FjPrintIlluminantParams, cmy_cc),
        3,
        size_of::<crate::print_bridge::FjPrintIlluminantFailure>(),
        align_of::<crate::print_bridge::FjPrintIlluminantFailure>(),
        std::mem::offset_of!(crate::print_bridge::FjPrintIlluminantFailure, reason),
        std::mem::offset_of!(crate::print_bridge::FjPrintIlluminantFailure, sample),
        size_of::<crate::asset_illuminant::FjIlluminant>(),
        align_of::<crate::asset_illuminant::FjIlluminant>(),
        std::mem::offset_of!(crate::asset_illuminant::FjIlluminant, samples),
        81,
        size_of::<FjFloatSpan>(),
        align_of::<FjFloatSpan>(),
        std::mem::offset_of!(FjFloatSpan, data),
        std::mem::offset_of!(FjFloatSpan, count),
        crate::print_bridge::FJ_PRINT_ILLUMINANT_REASON_NONE as usize,
        crate::print_bridge::FJ_PRINT_ILLUMINANT_REASON_RAW as usize,
        crate::print_bridge::FJ_PRINT_ILLUMINANT_REASON_SAMPLE as usize,
        crate::print_bridge::FJ_PRINT_ILLUMINANT_REASON_ENERGY as usize,
        crate::print_bridge::FJ_PRINT_ILLUMINANT_REASON_PANIC as usize,
        size_of::<crate::print_bridge::FjPrintPreflashInput>(),
        align_of::<crate::print_bridge::FjPrintPreflashInput>(),
        std::mem::offset_of!(crate::print_bridge::FjPrintPreflashInput, film_base_density),
        std::mem::offset_of!(
            crate::print_bridge::FjPrintPreflashInput,
            filtered_illuminant
        ),
        std::mem::offset_of!(
            crate::print_bridge::FjPrintPreflashInput,
            print_sensitivity_cmy
        ),
        size_of::<crate::print_bridge::FjPrintPreflashRaw>(),
        align_of::<crate::print_bridge::FjPrintPreflashRaw>(),
        std::mem::offset_of!(crate::print_bridge::FjPrintPreflashRaw, cmy),
        3,
        size_of::<crate::print_bridge::FjPrintPreflashFailure>(),
        align_of::<crate::print_bridge::FjPrintPreflashFailure>(),
        std::mem::offset_of!(crate::print_bridge::FjPrintPreflashFailure, reason),
        std::mem::offset_of!(crate::print_bridge::FjPrintPreflashFailure, channel),
        crate::print_bridge::FJ_PRINT_PREFLASH_REASON_NONE as usize,
        crate::print_bridge::FJ_PRINT_PREFLASH_REASON_RAW as usize,
        crate::print_bridge::FJ_PRINT_PREFLASH_REASON_CHANNEL as usize,
        crate::print_bridge::FJ_PRINT_PREFLASH_REASON_PANIC as usize,
    ];
    if !count.is_null() {
        // SAFETY: Facade caller authorizes the exclusive count output.
        unsafe {
            count.write(FACTS.len());
        }
    }
    FACTS.as_ptr()
}
