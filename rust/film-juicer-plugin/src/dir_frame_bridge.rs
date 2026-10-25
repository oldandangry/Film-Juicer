//! SF_TEMP_BRIDGE_DIR_FRAME: fixed, synchronous mathematical transport. Retire
//! with its last native model caller by S6.D; S4.E narrows execution inputs.

use std::mem::{align_of, size_of};
use std::panic::{AssertUnwindSafe, catch_unwind};

use film_juicer_core::film::dir::frame::{self, Component, Operator, Plan};
use film_juicer_core::frame::Domain;

use crate::cuda::sys::{FjDirGaussian, FjErrorBuffer, FjExtent, FjStatus};

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjDirFrameInput {
    pub(crate) active: u32,
    pub(crate) diffusion_size_um: f32,
    pub(crate) diffusion_tail_um: f32,
    pub(crate) diffusion_tail_weight: f32,
    pub(crate) pixel_size_um: f32,
    pub(crate) render_extent: FjExtent,
    pub(crate) full_frame_extent: FjExtent,
    pub(crate) recipe_hash: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjDirFrameMath {
    pub(crate) present: u32,
    pub(crate) component_count: u32,
    pub(crate) recipe_hash: u64,
    pub(crate) render_extent: FjExtent,
    pub(crate) full_frame_extent: FjExtent,
    pub(crate) filter_domain_extent: FjExtent,
    pub(crate) components: [FjDirGaussian; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjDirFrameFailure {
    pub(crate) reason: u32,
    pub(crate) field: u32,
    pub(crate) component: u32,
    pub(crate) trace: FjDirFrameMath,
}

const EMPTY_EXTENT: FjExtent = FjExtent {
    x: 0,
    y: 0,
    width: 0,
    height: 0,
};
const EMPTY_COMPONENT: FjDirGaussian = FjDirGaussian {
    sigma_px: 0.0,
    weight: 0.0,
    radius: 0,
    reference_operator: 0,
    feedforward: 0.0,
    feedback: [0.0; 3],
    normalization_denominator: 0.0,
    feedforward_numerator: 0.0,
    feedback_numerators: [0.0; 3],
    boundary_truncation_accuracy: 0.0,
    boundary_certification_tolerance: 0.0,
    boundary_derivation_version: 0,
};
const EMPTY_MATH: FjDirFrameMath = FjDirFrameMath {
    present: 0,
    component_count: 0,
    recipe_hash: 0,
    render_extent: EMPTY_EXTENT,
    full_frame_extent: EMPTY_EXTENT,
    filter_domain_extent: EMPTY_EXTENT,
    components: [EMPTY_COMPONENT; 4],
};
const EMPTY_FAILURE: FjDirFrameFailure = FjDirFrameFailure {
    reason: 0,
    field: 0,
    component: 4,
    trace: EMPTY_MATH,
};

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
fn extent(v: Domain) -> FjExtent {
    FjExtent {
        x: v.x,
        y: v.y,
        width: v.width,
        height: v.height,
    }
}

fn component(v: &Component) -> FjDirGaussian {
    let c = v.coefficients();
    FjDirGaussian {
        sigma_px: v.sigma_px(),
        weight: v.weight(),
        radius: v.radius(),
        reference_operator: match v.operator() {
            Operator::None => 0,
            Operator::Identity => 1,
            Operator::FirReflect => 2,
            Operator::YvvReflect => 3,
        },
        feedforward: c[0],
        feedback: [c[1], c[2], c[3]],
        normalization_denominator: c[4],
        feedforward_numerator: c[5],
        feedback_numerators: [c[6], c[7], c[8]],
        boundary_truncation_accuracy: v.boundary_accuracy(),
        boundary_certification_tolerance: v.boundary_tolerance(),
        boundary_derivation_version: v.boundary_version(),
    }
}

fn projection(
    recipe_hash: u64,
    render: Domain,
    full: Domain,
    count: usize,
    components: &[Component; 4],
) -> FjDirFrameMath {
    FjDirFrameMath {
        present: 1,
        component_count: count as u32,
        recipe_hash,
        render_extent: extent(render),
        full_frame_extent: extent(full),
        filter_domain_extent: extent(full),
        components: components.map(|c| component(&c)),
    }
}

const MESSAGES: [&str; 13] = [
    "",
    "SpatialDirFrameFailure field=pixel_size_um",
    "SpatialDirFrameFailure field=render_extent",
    "SpatialDirFrameFailure field=full_frame_extent",
    "SpatialDirFrameFailure field=partial_render_window",
    "SpatialDirFrameFailure field=core",
    "SpatialDirFrameFailure field=tail",
    "SpatialDirFrameFailure field=component_values",
    "SpatialDirFrameFailure field=component_capacity",
    "SpatialDirFrameFailure field=radius",
    "SpatialDirFrameFailure field=iir_denominator",
    "SpatialDirFrameFailure field=iir_output",
    "SpatialDirFrameFailure field=empty_plan",
];

fn status(category: u32) -> FjStatus {
    FjStatus {
        category,
        api: 0,
        native_code: 0,
    }
}

/// # Safety
/// Independently authorized initialized/aligned/live writable peers remain exclusive/disjoint.
unsafe fn clear(out: *mut FjDirFrameMath, failure: *mut FjDirFrameFailure) {
    // SAFETY: Each admitted peer may be written independently of all other arguments.
    unsafe {
        if valid(out) {
            out.write(EMPTY_MATH);
        }
        if valid(failure) {
            failure.write(EMPTY_FAILURE);
        }
    }
}

/// # Safety
/// Required records are initialized/aligned/live, writable peers and optional
/// diagnostics exclusive/mutually disjoint/disjoint from input. No pointer escapes.
pub(crate) unsafe fn prepare(
    input: *const FjDirFrameInput,
    out: *mut FjDirFrameMath,
    failure: *mut FjDirFrameFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Independent output admission shares the entry's foreign storage contract.
        unsafe {
            clear(out, failure);
        }
        // SAFETY: The optional diagnostic record/backing have the same live exclusive contract.
        let mut diagnostic = unsafe { Diagnostic::admit(error) };
        let admitted = valid(input) && valid(out) && valid(failure) && diagnostic.is_ok();
        let input = if admitted {
            // SAFETY: Numeric admission succeeded; caller authorizes this initialized aligned copy.
            Some(unsafe { input.read() })
        } else {
            None
        };
        if !admitted || input.is_some_and(|v| v.active > 1) {
            if valid(failure) {
                // SAFETY: Independently admitted failure remains writable despite peer rejection.
                unsafe {
                    failure.write(FjDirFrameFailure {
                        reason: 1,
                        ..EMPTY_FAILURE
                    });
                }
            }
            if let Ok(writer) = &mut diagnostic {
                writer.write("invalid spatial DIR frame arguments");
            }
            return status(2);
        }
        let input = input.unwrap();
        #[cfg(feature = "test-support")]
        let fault = dir_frame_faults::take();
        #[cfg(feature = "test-support")]
        if fault == 1 {
            panic!("spatial DIR frame fixture panic");
        }
        let result = Plan::new(frame::Input {
            active: input.active != 0,
            diffusion_size_um: input.diffusion_size_um,
            diffusion_tail_um: input.diffusion_tail_um,
            diffusion_tail_weight: input.diffusion_tail_weight,
            pixel_size_um: input.pixel_size_um,
            render_extent: domain(input.render_extent),
            full_frame_extent: domain(input.full_frame_extent),
            recipe_hash: input.recipe_hash,
        });
        match result {
            Ok(plan) => {
                let wire = plan.as_ref().map_or(EMPTY_MATH, |p| {
                    projection(
                        p.recipe_hash(),
                        p.render_extent(),
                        p.full_frame_extent(),
                        p.component_count(),
                        p.components(),
                    )
                });
                #[cfg(feature = "test-support")]
                let wire = dir_frame_faults::corrupt(wire, fault);
                // SAFETY: All peers were admitted and the fixed completed projection owns no pointer.
                unsafe {
                    out.write(wire);
                    #[cfg(feature = "test-support")]
                    if fault == 2 && wire.present != 0 {
                        (*failure).reason = 1;
                    }
                }
                status(0)
            }
            Err(rejected) => {
                let trace = rejected.trace().map_or(EMPTY_MATH, |p| {
                    projection(
                        p.recipe_hash(),
                        p.render_extent(),
                        p.full_frame_extent(),
                        p.component_count(),
                        p.components(),
                    )
                });
                let details = FjDirFrameFailure {
                    reason: 2,
                    field: rejected.field() as u32,
                    component: rejected.component() as u32,
                    trace,
                };
                #[cfg(feature = "test-support")]
                let details = dir_frame_faults::corrupt_trace(details, fault);
                // SAFETY: Admitted exclusive failure storage receives diagnostic progress, never a Plan.
                unsafe {
                    failure.write(details);
                }
                diagnostic
                    .as_mut()
                    .unwrap()
                    .write(MESSAGES[rejected.field() as usize]);
                status(3)
            }
        }
    }));
    match result {
        Ok(result) => result,
        Err(_) => {
            // SAFETY: Recovery repeats independent admission and bounded non-allocating writes only.
            unsafe {
                clear(out, failure);
                if valid(failure) {
                    failure.write(FjDirFrameFailure {
                        reason: 3,
                        ..EMPTY_FAILURE
                    });
                }
                if let Ok(mut writer) = Diagnostic::admit(error) {
                    writer.write("spatial DIR frame preparation panicked");
                }
            }
            status(8)
        }
    }
}

#[unsafe(no_mangle)]
/// # Safety
/// Same initialized/live/aligned and exclusive/disjoint obligations as prepare.
unsafe extern "C" fn fj_legacy_dir_frame_prepare(
    input: *const FjDirFrameInput,
    out: *mut FjDirFrameMath,
    failure: *mut FjDirFrameFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The sole checked body contains admission, construction and projection panics.
    unsafe { prepare(input, out, failure, error) }
}

#[cfg(feature = "test-support")]
pub(crate) mod dir_frame_faults {
    use std::cell::Cell;
    use super::*;
    thread_local! { static PENDING: Cell<u32> = const { Cell::new(0) }; }
    pub(crate) fn arm(fault: u32) -> FjStatus {
        if fault > 6 {
            return status(2);
        }
        PENDING.with(|p| p.set(fault));
        status(0)
    }
    pub(super) fn take() -> u32 {
        PENDING.with(|p| p.replace(0))
    }
    pub(super) fn corrupt(mut wire: FjDirFrameMath, fault: u32) -> FjDirFrameMath {
        if wire.present != 0 {
            match fault {
                3 => wire.component_count = 5,
                4 => wire.components[0].reference_operator = 99,
                5 => wire.components[0].radius = -1,
                _ => {}
            }
        }
        wire
    }
    pub(super) fn corrupt_trace(mut failure: FjDirFrameFailure, fault: u32) -> FjDirFrameFailure {
        if fault == 6 {
            failure.trace.component_count = 5;
        }
        failure
    }
}

const _: unsafe extern "C" fn(
    *const FjDirFrameInput,
    *mut FjDirFrameMath,
    *mut FjDirFrameFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_legacy_dir_frame_prepare;
const _: () = {
    assert!(size_of::<FjDirFrameInput>() == 64 && align_of::<FjDirFrameInput>() == 8);
    assert!(std::mem::offset_of!(FjDirFrameInput, active) == 0);
    assert!(std::mem::offset_of!(FjDirFrameInput, diffusion_size_um) == 4);
    assert!(std::mem::offset_of!(FjDirFrameInput, diffusion_tail_um) == 8);
    assert!(std::mem::offset_of!(FjDirFrameInput, diffusion_tail_weight) == 12);
    assert!(std::mem::offset_of!(FjDirFrameInput, pixel_size_um) == 16);
    assert!(std::mem::offset_of!(FjDirFrameInput, render_extent) == 20);
    assert!(std::mem::offset_of!(FjDirFrameInput, full_frame_extent) == 36);
    assert!(std::mem::offset_of!(FjDirFrameInput, recipe_hash) == 56);
    assert!(size_of::<FjDirFrameMath>() == 512 && align_of::<FjDirFrameMath>() == 8);
    assert!(std::mem::offset_of!(FjDirFrameMath, present) == 0);
    assert!(std::mem::offset_of!(FjDirFrameMath, component_count) == 4);
    assert!(std::mem::offset_of!(FjDirFrameMath, recipe_hash) == 8);
    assert!(std::mem::offset_of!(FjDirFrameMath, render_extent) == 16);
    assert!(std::mem::offset_of!(FjDirFrameMath, full_frame_extent) == 32);
    assert!(std::mem::offset_of!(FjDirFrameMath, filter_domain_extent) == 48);
    assert!(std::mem::offset_of!(FjDirFrameMath, components) == 64);
    assert!(size_of::<FjDirFrameFailure>() == 528 && align_of::<FjDirFrameFailure>() == 8);
    assert!(std::mem::offset_of!(FjDirFrameFailure, reason) == 0);
    assert!(std::mem::offset_of!(FjDirFrameFailure, field) == 4);
    assert!(std::mem::offset_of!(FjDirFrameFailure, component) == 8);
    assert!(std::mem::offset_of!(FjDirFrameFailure, trace) == 16);
    assert!(size_of::<FjExtent>() == 16 && align_of::<FjExtent>() == 4);
    assert!(std::mem::offset_of!(FjExtent, x) == 0);
    assert!(std::mem::offset_of!(FjExtent, y) == 4);
    assert!(std::mem::offset_of!(FjExtent, width) == 8);
    assert!(std::mem::offset_of!(FjExtent, height) == 12);
    assert!(size_of::<FjDirGaussian>() == 112 && align_of::<FjDirGaussian>() == 8);
    assert!(std::mem::offset_of!(FjDirGaussian, sigma_px) == 0);
    assert!(std::mem::offset_of!(FjDirGaussian, weight) == 4);
    assert!(std::mem::offset_of!(FjDirGaussian, radius) == 8);
    assert!(std::mem::offset_of!(FjDirGaussian, reference_operator) == 12);
    assert!(std::mem::offset_of!(FjDirGaussian, feedforward) == 16);
    assert!(std::mem::offset_of!(FjDirGaussian, feedback) == 24);
    assert!(std::mem::offset_of!(FjDirGaussian, normalization_denominator) == 48);
    assert!(std::mem::offset_of!(FjDirGaussian, feedforward_numerator) == 56);
    assert!(std::mem::offset_of!(FjDirGaussian, feedback_numerators) == 64);
    assert!(std::mem::offset_of!(FjDirGaussian, boundary_truncation_accuracy) == 88);
    assert!(std::mem::offset_of!(FjDirGaussian, boundary_certification_tolerance) == 96);
    assert!(std::mem::offset_of!(FjDirGaussian, boundary_derivation_version) == 104);
    assert!(
        size_of::<crate::cuda::sys::FjSpatialDir>() == 520
            && align_of::<crate::cuda::sys::FjSpatialDir>() == 8
    );
    assert!(std::mem::offset_of!(crate::cuda::sys::FjSpatialDir, render_extent) == 0);
    assert!(std::mem::offset_of!(crate::cuda::sys::FjSpatialDir, full_frame_extent) == 16);
    assert!(std::mem::offset_of!(crate::cuda::sys::FjSpatialDir, filter_domain_extent) == 32);
    assert!(std::mem::offset_of!(crate::cuda::sys::FjSpatialDir, component_count) == 48);
    assert!(std::mem::offset_of!(crate::cuda::sys::FjSpatialDir, components) == 56);
    assert!(std::mem::offset_of!(crate::cuda::sys::FjSpatialDir, recipe_hash) == 504);
    assert!(std::mem::offset_of!(crate::cuda::sys::FjSpatialDir, hash) == 512);
};
const _: fn(FjDirFrameInput) = |v| {
    let _: u32 = v.active;
    let _: f32 = v.diffusion_size_um;
    let _: f32 = v.diffusion_tail_um;
    let _: f32 = v.diffusion_tail_weight;
    let _: f32 = v.pixel_size_um;
    let _: FjExtent = v.render_extent;
    let _: FjExtent = v.full_frame_extent;
    let _: u64 = v.recipe_hash;
};
const _: fn(FjDirFrameMath) = |v| {
    let _: u32 = v.present;
    let _: u32 = v.component_count;
    let _: u64 = v.recipe_hash;
    let _: FjExtent = v.render_extent;
    let _: FjExtent = v.full_frame_extent;
    let _: FjExtent = v.filter_domain_extent;
    let _: [FjDirGaussian; 4] = v.components;
};
const _: fn(FjDirFrameFailure) = |v| {
    let _: u32 = v.reason;
    let _: u32 = v.field;
    let _: u32 = v.component;
    let _: FjDirFrameMath = v.trace;
};
const _: fn(FjExtent) = |v| {
    let _: i32 = v.x;
    let _: i32 = v.y;
    let _: i32 = v.width;
    let _: i32 = v.height;
};
const _: fn(FjDirGaussian) = |v| {
    let _: f32 = v.sigma_px;
    let _: f32 = v.weight;
    let _: i32 = v.radius;
    let _: u32 = v.reference_operator;
    let _: f64 = v.feedforward;
    let _: [f64; 3] = v.feedback;
    let _: f64 = v.normalization_denominator;
    let _: f64 = v.feedforward_numerator;
    let _: [f64; 3] = v.feedback_numerators;
    let _: f64 = v.boundary_truncation_accuracy;
    let _: f64 = v.boundary_certification_tolerance;
    let _: u32 = v.boundary_derivation_version;
};
const _: fn(crate::cuda::sys::FjSpatialDir) = |v| {
    let _: FjExtent = v.render_extent;
    let _: FjExtent = v.full_frame_extent;
    let _: FjExtent = v.filter_domain_extent;
    let _: usize = v.component_count;
    let _: [FjDirGaussian; 4] = v.components;
    let _: u64 = v.recipe_hash;
    let _: u64 = v.hash;
};
