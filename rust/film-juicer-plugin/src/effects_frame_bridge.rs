//! SF_TEMP_BRIDGE_EFFECTS_FRAME: synchronous resolved literal transport.
//! S4.E narrows inputs; remove at the last native model caller by S6.D.

use std::mem::{align_of, size_of};
use std::panic::{AssertUnwindSafe, catch_unwind};

use film_juicer_core::effects::frame::{self, Frame};
use film_juicer_core::frame::Domain;
use crate::cuda::sys::{FjDefectOrigin, FjDust, FjScratch, FjExtent, FjErrorBuffer, FjStatus};

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjEffectsFrameInput {
    pub(crate) active: u32,
    pub(crate) render_extent: FjExtent,
    pub(crate) pixel_definition: FjExtent,
    pub(crate) pixel_size_um: f32,
    pub(crate) film_format_long_edge_mm: f32,
    pub(crate) canonical_x: f64,
    pub(crate) canonical_y: f64,
    pub(crate) canonical_width: f64,
    pub(crate) canonical_height: f64,
    pub(crate) scale_x: f64,
    pub(crate) scale_y: f64,
    pub(crate) pixel_aspect_ratio: f64,
    pub(crate) frame_time: f64,
    pub(crate) frame_rate: f64,
    pub(crate) gate_weave_amount: f64,
    pub(crate) session_seed: u64,
    pub(crate) clip_token: u64,
    pub(crate) recipe_hash: u64,
    pub(crate) film_dust: FjDust,
    pub(crate) film_scratch: FjScratch,
    pub(crate) gate_dust: FjDust,
    pub(crate) gate_scratch: FjScratch,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjEffectsFrameMath {
    pub(crate) present: u32,
    pub(crate) flags: u32,
    pub(crate) render_extent: FjExtent,
    pub(crate) full_frame_extent: FjExtent,
    pub(crate) origins: [FjDefectOrigin; 4],
    pub(crate) sample_step_x_mm: f32,
    pub(crate) sample_step_y_mm: f32,
    pub(crate) roi_offset_x: i32,
    pub(crate) roi_offset_y: i32,
    pub(crate) gate_width: i32,
    pub(crate) gate_height: i32,
    pub(crate) session_seed: u64,
    pub(crate) clip_token: u64,
    pub(crate) weave_dx_px: f32,
    pub(crate) weave_dy_px: f32,
    pub(crate) weave_cos_rot: f32,
    pub(crate) weave_sin_rot: f32,
    pub(crate) recipe_hash: u64,
    pub(crate) hash: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjEffectsFrameFailure {
    pub(crate) reason: u32,
}

const _: () = {
    assert!(size_of::<FjEffectsFrameInput>() == 776 && align_of::<FjEffectsFrameInput>() == 8);
    assert!(size_of::<FjEffectsFrameMath>() == 208 && align_of::<FjEffectsFrameMath>() == 8);
    assert!(size_of::<FjEffectsFrameFailure>() == 4 && align_of::<FjEffectsFrameFailure>() == 4);
    assert!(std::mem::offset_of!(FjEffectsFrameInput, active) == 0);
    assert!(std::mem::offset_of!(FjEffectsFrameInput, render_extent) == 4);
    assert!(std::mem::offset_of!(FjEffectsFrameInput, pixel_definition) == 20);
    assert!(std::mem::offset_of!(FjEffectsFrameInput, pixel_size_um) == 36);
    assert!(std::mem::offset_of!(FjEffectsFrameInput, film_format_long_edge_mm) == 40);
    assert!(std::mem::offset_of!(FjEffectsFrameInput, canonical_x) == 48);
    assert!(std::mem::offset_of!(FjEffectsFrameInput, canonical_y) == 56);
    assert!(std::mem::offset_of!(FjEffectsFrameInput, canonical_width) == 64);
    assert!(std::mem::offset_of!(FjEffectsFrameInput, canonical_height) == 72);
    assert!(std::mem::offset_of!(FjEffectsFrameInput, scale_x) == 80);
    assert!(std::mem::offset_of!(FjEffectsFrameInput, scale_y) == 88);
    assert!(std::mem::offset_of!(FjEffectsFrameInput, pixel_aspect_ratio) == 96);
    assert!(std::mem::offset_of!(FjEffectsFrameInput, frame_time) == 104);
    assert!(std::mem::offset_of!(FjEffectsFrameInput, frame_rate) == 112);
    assert!(std::mem::offset_of!(FjEffectsFrameInput, gate_weave_amount) == 120);
    assert!(std::mem::offset_of!(FjEffectsFrameInput, session_seed) == 128);
    assert!(std::mem::offset_of!(FjEffectsFrameInput, clip_token) == 136);
    assert!(std::mem::offset_of!(FjEffectsFrameInput, recipe_hash) == 144);
    assert!(std::mem::offset_of!(FjEffectsFrameInput, film_dust) == 152);
    assert!(std::mem::offset_of!(FjEffectsFrameInput, film_scratch) == 320);
    assert!(std::mem::offset_of!(FjEffectsFrameInput, gate_dust) == 464);
    assert!(std::mem::offset_of!(FjEffectsFrameInput, gate_scratch) == 632);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, present) == 0);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, flags) == 4);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, render_extent) == 8);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, full_frame_extent) == 24);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, origins) == 40);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, sample_step_x_mm) == 136);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, sample_step_y_mm) == 140);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, roi_offset_x) == 144);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, roi_offset_y) == 148);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, gate_width) == 152);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, gate_height) == 156);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, session_seed) == 160);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, clip_token) == 168);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, weave_dx_px) == 176);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, weave_dy_px) == 180);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, weave_cos_rot) == 184);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, weave_sin_rot) == 188);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, recipe_hash) == 192);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, hash) == 200);
    assert!(std::mem::offset_of!(FjEffectsFrameFailure, reason) == 0);
};
const _: fn(FjEffectsFrameInput) = |v| {
    let _: u32 = v.active;
    let _: FjExtent = v.render_extent;
    let _: FjExtent = v.pixel_definition;
    let _: f32 = v.pixel_size_um;
    let _: f32 = v.film_format_long_edge_mm;
    let _: f64 = v.canonical_x;
    let _: f64 = v.canonical_y;
    let _: f64 = v.canonical_width;
    let _: f64 = v.canonical_height;
    let _: f64 = v.scale_x;
    let _: f64 = v.scale_y;
    let _: f64 = v.pixel_aspect_ratio;
    let _: f64 = v.frame_time;
    let _: f64 = v.frame_rate;
    let _: f64 = v.gate_weave_amount;
    let _: u64 = v.session_seed;
    let _: u64 = v.clip_token;
    let _: u64 = v.recipe_hash;
    let _: FjDust = v.film_dust;
    let _: FjScratch = v.film_scratch;
    let _: FjDust = v.gate_dust;
    let _: FjScratch = v.gate_scratch;
};
const _: fn(FjEffectsFrameMath) = |v| {
    let _: u32 = v.present;
    let _: u32 = v.flags;
    let _: FjExtent = v.render_extent;
    let _: FjExtent = v.full_frame_extent;
    let _: [FjDefectOrigin; 4] = v.origins;
    let _: f32 = v.sample_step_x_mm;
    let _: f32 = v.sample_step_y_mm;
    let _: i32 = v.roi_offset_x;
    let _: i32 = v.roi_offset_y;
    let _: i32 = v.gate_width;
    let _: i32 = v.gate_height;
    let _: u64 = v.session_seed;
    let _: u64 = v.clip_token;
    let _: f32 = v.weave_dx_px;
    let _: f32 = v.weave_dy_px;
    let _: f32 = v.weave_cos_rot;
    let _: f32 = v.weave_sin_rot;
    let _: u64 = v.recipe_hash;
    let _: u64 = v.hash;
};
const _: fn(FjEffectsFrameFailure) = |v| {
    let _: u32 = v.reason;
};
const EMPTY_EXTENT: FjExtent = FjExtent {
    x: 0,
    y: 0,
    width: 0,
    height: 0,
};
const EMPTY_ORIGIN: FjDefectOrigin = FjDefectOrigin {
    cell_x: 0,
    cell_y: 0,
    local_x_mm: 0.0,
    local_y_mm: 0.0,
};
const EMPTY_MATH: FjEffectsFrameMath = FjEffectsFrameMath {
    present: 0,
    flags: 0,
    render_extent: EMPTY_EXTENT,
    full_frame_extent: EMPTY_EXTENT,
    origins: [EMPTY_ORIGIN; 4],
    sample_step_x_mm: 0.0,
    sample_step_y_mm: 0.0,
    roi_offset_x: 0,
    roi_offset_y: 0,
    gate_width: 0,
    gate_height: 0,
    session_seed: 0,
    clip_token: 0,
    weave_dx_px: 0.0,
    weave_dy_px: 0.0,
    weave_cos_rot: 0.0,
    weave_sin_rot: 0.0,
    recipe_hash: 0,
    hash: 0,
};
const EMPTY_FAILURE: FjEffectsFrameFailure = FjEffectsFrameFailure { reason: 0 };
fn valid<T>(pointer: *const T) -> bool {
    !pointer.is_null()
        && (pointer as usize).is_multiple_of(align_of::<T>())
        && size_of::<T>() <= isize::MAX as usize
        && (pointer as usize).checked_add(size_of::<T>()).is_some()
}
struct Diagnostic {
    record: *mut FjErrorBuffer,
    data: *mut u8,
    capacity: usize,
}

impl Diagnostic {
    /// # Safety
    /// Optional initialized record and nonempty backing are live/exclusive/disjoint.
    unsafe fn admit(record: *mut FjErrorBuffer) -> Result<Self, ()> {
        let mut writer = Self {
            record,
            data: std::ptr::null_mut(),
            capacity: 0,
        };
        if record.is_null() {
            return Ok(writer);
        }
        if !valid(record) {
            return Err(());
        }
        // SAFETY: Independent numeric admission precedes authorized initialized record access.
        unsafe {
            (*record).length = 0;
            writer.capacity = (*record).capacity;
            if writer.capacity != 0 {
                writer.data = (*record).data.cast();
                if writer.data.is_null()
                    || writer.capacity > isize::MAX as usize
                    || (writer.data as usize)
                        .checked_add(writer.capacity)
                        .is_none()
                {
                    return Err(());
                }
                writer.data.write(0);
            }
        }
        Ok(writer)
    }

    fn write(&mut self, text: &str) {
        let count = text.len().min(self.capacity.saturating_sub(1));
        // SAFETY: Admitted live/exclusive backing has capacity; count leaves room for NUL.
        unsafe {
            if self.capacity != 0 {
                std::ptr::copy_nonoverlapping(text.as_ptr(), self.data, count);
                self.data.add(count).write(0);
            }
            if !self.record.is_null() {
                (*self.record).length = count;
            }
        }
    }
}

fn domain(v: FjExtent) -> Domain {
    Domain {
        x: v.x,
        y: v.y,
        width: v.width,
        height: v.height,
    }
}
fn dust(v: FjDust) -> frame::DustInput {
    frame::DustInput {
        cell_width_mm: v.cell_width_mm,
        cell_height_mm: v.cell_height_mm,
        slot_probability: v.slot_probability,
        softness_min_mm: v.softness_min_mm,
        softness_max_mm: v.softness_max_mm,
        softness_size_cap_fraction: v.softness_size_cap_fraction,
        support_x_mm: v.support_x_mm,
        support_y_mm: v.support_y_mm,
        fiber_fraction: v.fiber_fraction,
        fiber_drift_fraction: v.fiber_drift_fraction,
        fiber_first_knot_min: v.fiber_first_knot_min,
        fiber_first_knot_max: v.fiber_first_knot_max,
        fiber_second_knot_min: v.fiber_second_knot_min,
        fiber_second_knot_max: v.fiber_second_knot_max,
        fiber_interior_width_min_fraction: v.fiber_interior_width_min_fraction,
        fiber_interior_width_max_fraction: v.fiber_interior_width_max_fraction,
        diameter_min_mm: v.diameter_min_mm,
        diameter_bulk_max_mm: v.diameter_bulk_max_mm,
        diameter_max_mm: v.diameter_max_mm,
        diameter_tail_fraction: v.diameter_tail_fraction,
        fiber_length_min_mm: v.fiber_length_min_mm,
        fiber_length_max_mm: v.fiber_length_max_mm,
        fiber_width_min_mm: v.fiber_width_min_mm,
        fiber_width_max_mm: v.fiber_width_max_mm,
        opacity_faint_cumulative: v.opacity_faint_cumulative,
        opacity_intermediate_cumulative: v.opacity_intermediate_cumulative,
        compact_opacity_min: v.compact_opacity_min,
        compact_opacity_faint_end: v.compact_opacity_faint_end,
        compact_opacity_intermediate_end: v.compact_opacity_intermediate_end,
        compact_opacity_max: v.compact_opacity_max,
        fiber_opacity_min: v.fiber_opacity_min,
        fiber_opacity_faint_end: v.fiber_opacity_faint_end,
        fiber_opacity_intermediate_end: v.fiber_opacity_intermediate_end,
        fiber_opacity_max: v.fiber_opacity_max,
        compact_dominant_aspect_min: v.compact_dominant_aspect_min,
        compact_dominant_aspect_max: v.compact_dominant_aspect_max,
        compact_subsidiary_scale_min: v.compact_subsidiary_scale_min,
        compact_subsidiary_scale_max: v.compact_subsidiary_scale_max,
        compact_subsidiary_aspect_min: v.compact_subsidiary_aspect_min,
        compact_subsidiary_aspect_max: v.compact_subsidiary_aspect_max,
        compact_subsidiary_offset_max: v.compact_subsidiary_offset_max,
        compact_subsidiary_angle_max_radians: v.compact_subsidiary_angle_max_radians,
    }
}
fn scratch(v: FjScratch) -> frame::ScratchInput {
    frame::ScratchInput {
        cell_width_mm: v.cell_width_mm,
        cell_height_mm: v.cell_height_mm,
        slot_probability: v.slot_probability,
        softness_min_mm: v.softness_min_mm,
        softness_max_mm: v.softness_max_mm,
        softness_size_cap_fraction: v.softness_size_cap_fraction,
        support_x_mm: v.support_x_mm,
        support_y_mm: v.support_y_mm,
        length_min_mm: v.length_min_mm,
        length_bulk_max_mm: v.length_bulk_max_mm,
        length_max_mm: v.length_max_mm,
        length_tail_fraction: v.length_tail_fraction,
        width_min_mm: v.width_min_mm,
        width_bulk_max_mm: v.width_bulk_max_mm,
        width_max_mm: v.width_max_mm,
        width_tail_fraction: v.width_tail_fraction,
        drift_fraction: v.drift_fraction,
        first_knot_min: v.first_knot_min,
        first_knot_max: v.first_knot_max,
        second_knot_min: v.second_knot_min,
        second_knot_max: v.second_knot_max,
        interior_width_min_fraction: v.interior_width_min_fraction,
        interior_width_max_fraction: v.interior_width_max_fraction,
        interior_depth_min_fraction: v.interior_depth_min_fraction,
        interior_depth_max_fraction: v.interior_depth_max_fraction,
        endpoint_abrupt_probability: v.endpoint_abrupt_probability,
        interruption_probability: v.interruption_probability,
        gap_center_min: v.gap_center_min,
        gap_center_max: v.gap_center_max,
        gap_span_min: v.gap_span_min,
        gap_span_max: v.gap_span_max,
        scuff_probability: v.scuff_probability,
        scuff_length_max_mm: v.scuff_length_max_mm,
        scuff_angle_max_radians: v.scuff_angle_max_radians,
        strength_min: v.strength_min,
        strength_max: v.strength_max,
    }
}
fn input(v: FjEffectsFrameInput) -> frame::Input {
    frame::Input {
        active: v.active != 0,
        render_extent: domain(v.render_extent),
        pixel_definition: domain(v.pixel_definition),
        pixel_size_um: v.pixel_size_um,
        film_format_long_edge_mm: v.film_format_long_edge_mm,
        canonical_x: v.canonical_x,
        canonical_y: v.canonical_y,
        canonical_width: v.canonical_width,
        canonical_height: v.canonical_height,
        scale_x: v.scale_x,
        scale_y: v.scale_y,
        pixel_aspect_ratio: v.pixel_aspect_ratio,
        frame_time: v.frame_time,
        frame_rate: v.frame_rate,
        gate_weave_amount: v.gate_weave_amount,
        session_seed: v.session_seed,
        clip_token: v.clip_token,
        recipe_hash: v.recipe_hash,
        film_dust: dust(v.film_dust),
        film_scratch: scratch(v.film_scratch),
        gate_dust: dust(v.gate_dust),
        gate_scratch: scratch(v.gate_scratch),
    }
}

fn project(v: &Frame) -> FjEffectsFrameMath {
    let extent = |e: Domain| FjExtent {
        x: e.x,
        y: e.y,
        width: e.width,
        height: e.height,
    };
    FjEffectsFrameMath {
        present: 1,
        flags: u32::from(v.weave_active())
            | (u32::from(v.film_active()) << 1)
            | (u32::from(v.gate_transmittance_active()) << 2)
            | (u32::from(v.gate_output_active()) << 3)
            | (u32::from(v.requires_full_frame()) << 4),
        render_extent: extent(v.render_extent()),
        full_frame_extent: extent(v.full_frame_extent()),
        origins: v.origins().map(|o| FjDefectOrigin {
            cell_x: o.cell_x,
            cell_y: o.cell_y,
            local_x_mm: o.local_x_mm,
            local_y_mm: o.local_y_mm,
        }),
        sample_step_x_mm: v.sample_step_x_mm(),
        sample_step_y_mm: v.sample_step_y_mm(),
        roi_offset_x: v.roi_offset_x(),
        roi_offset_y: v.roi_offset_y(),
        gate_width: v.gate_width(),
        gate_height: v.gate_height(),
        session_seed: v.session_seed(),
        clip_token: v.clip_token(),
        weave_dx_px: v.weave()[0],
        weave_dy_px: v.weave()[1],
        weave_cos_rot: v.weave()[2],
        weave_sin_rot: v.weave()[3],
        recipe_hash: v.recipe_hash(),
        hash: v.hash(),
    }
}
fn status(category: u32) -> FjStatus {
    FjStatus {
        category,
        api: 0,
        native_code: 0,
    }
}

/// # Safety
/// Independently admitted writable records are initialized, live, exclusive and disjoint.
unsafe fn clear(out: *mut FjEffectsFrameMath, failure: *mut FjEffectsFrameFailure) {
    if valid(out) {
        // SAFETY: Numeric admission and caller's storage authorization cover this fixed write.
        unsafe {
            out.write(EMPTY_MATH);
        }
    }
    if valid(failure) {
        // SAFETY: This peer is independently admitted even if another record is invalid.
        unsafe {
            failure.write(EMPTY_FAILURE);
        }
    }
}

/// # Safety
/// Input and writable peers have initialized, aligned, live, mutually disjoint storage.
/// Optional diagnostic storage is exclusive for this synchronous call. Nothing escapes.
pub(crate) unsafe fn prepare(
    input_ptr: *const FjEffectsFrameInput,
    out: *mut FjEffectsFrameMath,
    failure: *mut FjEffectsFrameFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    #[cfg(feature = "test-support")]
    let fault = faults::take();
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Each admitted writable peer satisfies the entry's storage contract.
        unsafe {
            clear(out, failure);
        }
        // SAFETY: Optional diagnostic record and nonempty backing obey the entry contract.
        let mut diagnostic = unsafe { Diagnostic::admit(error) };
        let admitted = valid(input_ptr) && valid(out) && valid(failure) && diagnostic.is_ok();
        let raw = if admitted {
            // SAFETY: Admitted initialized fixed input is copied only during this call.
            Some(unsafe { input_ptr.read() })
        } else {
            None
        };
        if !admitted || raw.is_some_and(|v| v.active > 1) {
            if valid(failure) {
                // SAFETY: Independent admission authorizes this complete failure write.
                unsafe {
                    failure.write(FjEffectsFrameFailure { reason: 1 });
                }
            }
            if let Ok(writer) = &mut diagnostic {
                writer.write("physical effects frame arguments are invalid");
            }
            return status(2);
        }
        #[cfg(feature = "test-support")]
        if fault == 1 {
            panic!("physical effects frame fixture entry panic");
        }
        match Frame::new(input(raw.unwrap())) {
            Ok(frame) => {
                let wire = frame.as_ref().map_or(EMPTY_MATH, project);
                #[cfg(feature = "test-support")]
                let wire = faults::corrupt(wire, fault);
                // SAFETY: Fully constructed fixed output is written to admitted exclusive storage.
                unsafe {
                    out.write(wire);
                }
                status(0)
            }
            Err(rejected) => {
                let (reason, message) = match rejected {
                    frame::Error::InactiveIdentity => (2, ""),
                    frame::Error::GeometryOrPolicy => (
                        3,
                        "ResourceDescriptorMismatch phase=effects_descriptor field=geometry_or_policy",
                    ),
                };
                // SAFETY: Independently admitted failure peer; math remains canonical zero.
                unsafe {
                    failure.write(FjEffectsFrameFailure { reason });
                }
                diagnostic.as_mut().unwrap().write(message);
                status(3)
            }
        }
    }));
    match result {
        Ok(result) => result,
        Err(_) => {
            // SAFETY: Panic recovery repeats only independent fixed-record admission and bounded writes.
            unsafe {
                clear(out, failure);
                if valid(failure) {
                    failure.write(FjEffectsFrameFailure { reason: 4 });
                }
                if let Ok(mut writer) = Diagnostic::admit(error) {
                    writer.write("physical effects frame preparation panicked");
                }
            }
            status(8)
        }
    }
}

#[unsafe(no_mangle)]
/// # Safety
/// Same initialized/live/exclusive/disjoint synchronous storage contract as prepare.
unsafe extern "C" fn fj_legacy_effects_frame_prepare(
    input: *const FjEffectsFrameInput,
    out: *mut FjEffectsFrameMath,
    failure: *mut FjEffectsFrameFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The checked implementation contains admission and whole-operation unwind.
    unsafe { prepare(input, out, failure, error) }
}

const _: unsafe extern "C" fn(
    *const FjEffectsFrameInput,
    *mut FjEffectsFrameMath,
    *mut FjEffectsFrameFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_legacy_effects_frame_prepare;

#[cfg(feature = "test-support")]
pub(crate) mod faults {
    use std::cell::Cell;
    use super::{FjEffectsFrameMath, FjStatus, status};
    thread_local! { static PENDING: Cell<u32> = const { Cell::new(0) }; }
    pub(crate) fn arm(code: u32) -> FjStatus {
        if code > 6 {
            return status(2);
        }
        PENDING.with(|pending| pending.set(code));
        status(0)
    }
    pub(super) fn take() -> u32 {
        PENDING.with(|pending| pending.replace(0))
    }
    pub(super) fn corrupt(mut wire: FjEffectsFrameMath, fault: u32) -> FjEffectsFrameMath {
        match fault {
            2 => wire.present = 2,
            3 => wire.flags |= 32,
            4 => wire.session_seed ^= 1,
            5 => wire.origins[0].cell_x = 4503599627370496,
            6 if wire.present == 0 => wire.weave_cos_rot = 1.0,
            _ => {}
        }
        wire
    }
}
