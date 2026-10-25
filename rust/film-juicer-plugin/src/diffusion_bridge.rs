//! SF_TEMP_BRIDGE_DIFFUSION_FRAME: sample/frame projection retires by S6.D;
//! native-resource component conversion retires with S4.E execution inputs.

use std::mem::{align_of, size_of};
use std::panic::{AssertUnwindSafe, catch_unwind};

use film_juicer_core::optics::diffusion_frame::{self, Components, FrameSet, Sample, SampleFields};

use crate::cuda::sys::{FjDiffusion, FjErrorBuffer, FjExtent, FjStatus};
use crate::optics_bridge::FjDiffusionRecipe;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct FjDiffusionSampleInput {
    pub(crate) family: u32,
    pub(crate) group_weights: [f64; 3],
    pub(crate) group_lambda_um: [f64; 3],
    pub(crate) warmth: f64,
    pub(crate) spatial_scale: f64,
    pub(crate) pixel_size_um: f64,
    pub(crate) width: i32,
    pub(crate) height: i32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct FjDiffusionSample {
    pub(crate) family: u32,
    pub(crate) group_weights: [f64; 3],
    pub(crate) group_lambda_um: [f64; 3],
    pub(crate) warmth: f64,
    pub(crate) spatial_scale: f64,
    pub(crate) pixel_size_um: f64,
    pub(crate) radius_px: i32,
    pub(crate) hash: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct FjDiffusionComponents {
    pub(crate) lambda_px: [f64; 9],
    pub(crate) red_weights: [f64; 9],
    pub(crate) green_weights: [f64; 9],
    pub(crate) blue_weights: [f64; 9],
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjDiffusionFrameInput {
    pub(crate) route: u32,
    pub(crate) full_frame: FjExtent,
    pub(crate) pixel_size_um: f64,
    pub(crate) camera: FjDiffusionRecipe,
    pub(crate) enlarger: FjDiffusionRecipe,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjDiffusionFrameSet {
    pub(crate) route: u32,
    pub(crate) flags: u32,
    pub(crate) full_frame: FjExtent,
    pub(crate) camera: FjDiffusion,
    pub(crate) enlarger: FjDiffusion,
    pub(crate) hash: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct FjDiffusionFailure {
    pub(crate) reason: u32,
    pub(crate) stage: u32,
    pub(crate) field: u32,
}

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
    /// The optional record and nonempty backing are initialized/live/exclusive/disjoint.
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
        // SAFETY: Independent numeric admission precedes authorized live record access.
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
        // SAFETY: Admission establishes live exclusive backing; count leaves room for NUL.
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

#[derive(Clone, Copy)]
enum Operation {
    Sample = 1,
    Components = 2,
    Frame = 3,
}
impl Operation {
    fn raw_text(self) -> &'static str {
        match self {
            Self::Sample => "invalid diffusion sample arguments",
            Self::Components => "invalid diffusion components arguments",
            Self::Frame => "invalid diffusion frame arguments",
        }
    }
    fn panic_text(self) -> &'static str {
        match self {
            Self::Sample => "diffusion sample preparation panicked",
            Self::Components => "diffusion components preparation panicked",
            Self::Frame => "diffusion frame preparation panicked",
        }
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
/// Required records are initialized/aligned/live; writable peers and diagnostics
/// are exclusive and disjoint. `empty` clears every semantic field of fixed O.
/// Numeric admission does not establish provenance. No pointer escapes.
unsafe fn checked<I: Copy, O: Copy>(
    operation: Operation,
    input: *const I,
    out: *mut O,
    failure: *mut FjDiffusionFailure,
    error: *mut FjErrorBuffer,
    empty: O,
    build: fn(I, u32) -> Result<O, diffusion_frame::Error>,
) -> FjStatus {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Each independently admitted peer is caller-authorized live writable storage.
        unsafe {
            if valid(out) {
                out.write(empty);
            }
            if valid(failure) {
                failure.write(FjDiffusionFailure::default());
            }
        }
        // SAFETY: Diagnostic storage shares the entry's foreign borrow obligations.
        let mut writer = unsafe { Diagnostic::admit(error) };
        if !valid(input) || !valid(out) || !valid(failure) || writer.is_err() {
            if valid(failure) {
                // SAFETY: Independent admission of failure does not depend on its peers.
                unsafe {
                    failure.write(FjDiffusionFailure {
                        reason: 1,
                        stage: 0,
                        field: 0,
                    });
                }
            }
            if let Ok(writer) = &mut writer {
                writer.write(operation.raw_text());
            }
            return status(2);
        }
        #[cfg(feature = "test-support")]
        let fault = diffusion_frame_faults::take(operation as u32);
        #[cfg(not(feature = "test-support"))]
        let fault = 0;
        #[cfg(feature = "test-support")]
        if fault == 1 {
            panic!("diffusion construction fixture panic");
        }
        // SAFETY: All numeric admissions succeeded; caller supplies live aligned initialized input.
        match build(unsafe { input.read() }, fault) {
            Ok(wire) => {
                // SAFETY: Completed fixed output commits to the admitted exclusive slot.
                unsafe {
                    out.write(wire);
                }
                #[cfg(feature = "test-support")]
                if fault == 2 {
                    // SAFETY: The same admitted exclusive failure slot remains live.
                    unsafe {
                        failure.write(FjDiffusionFailure {
                            reason: 1,
                            stage: 0,
                            field: 0,
                        });
                    }
                }
                status(0)
            }
            Err(rejected) => {
                // SAFETY: Both writable peers were independently admitted; discard partial output.
                unsafe {
                    out.write(empty);
                    failure.write(FjDiffusionFailure {
                        reason: 2,
                        stage: rejected.stage(),
                        field: rejected.field() as u32,
                    });
                }
                writer.as_mut().unwrap().write(rejected.diagnostic());
                status(3)
            }
        }
    }));
    match result {
        Ok(status) => status,
        Err(_) => {
            // SAFETY: Recovery repeats independent numeric admission and bounded fixed writes.
            unsafe {
                if valid(out) {
                    out.write(empty);
                }
                if valid(failure) {
                    failure.write(FjDiffusionFailure {
                        reason: 3,
                        stage: 0,
                        field: 0,
                    });
                }
                if let Ok(mut writer) = Diagnostic::admit(error) {
                    writer.write(operation.panic_text());
                }
            }
            status(8)
        }
    }
}

fn sample_projection(fields: SampleFields) -> FjDiffusionSample {
    FjDiffusionSample {
        family: fields.family,
        group_weights: fields.group_weights,
        group_lambda_um: fields.group_lambda_um,
        warmth: fields.warmth,
        spatial_scale: fields.spatial_scale,
        pixel_size_um: fields.pixel_size_um,
        radius_px: fields.radius_px,
        hash: fields.hash,
    }
}
fn sample_fields(wire: FjDiffusionSample) -> SampleFields {
    SampleFields {
        family: wire.family,
        group_weights: wire.group_weights,
        group_lambda_um: wire.group_lambda_um,
        warmth: wire.warmth,
        spatial_scale: wire.spatial_scale,
        pixel_size_um: wire.pixel_size_um,
        radius_px: wire.radius_px,
        hash: wire.hash,
    }
}
fn sample_build(
    input: FjDiffusionSampleInput,
    _fault: u32,
) -> Result<FjDiffusionSample, diffusion_frame::Error> {
    let sample = Sample::new(diffusion_frame::SampleInput {
        family: input.family,
        group_weights: input.group_weights,
        group_lambda_um: input.group_lambda_um,
        warmth: input.warmth,
        spatial_scale: input.spatial_scale,
        pixel_size_um: input.pixel_size_um,
        width: input.width,
        height: input.height,
    })?;
    let wire = sample_projection(sample.fields());
    #[cfg(feature = "test-support")]
    let wire = diffusion_frame_faults::sample(wire, _fault);
    Ok(wire)
}
fn components_build(
    input: FjDiffusionSample,
    _fault: u32,
) -> Result<FjDiffusionComponents, diffusion_frame::Error> {
    let components = Components::new(sample_fields(input))?;
    Ok(FjDiffusionComponents {
        lambda_px: *components.lambda_px(),
        red_weights: *components.red_weights(),
        green_weights: *components.green_weights(),
        blue_weights: *components.blue_weights(),
    })
}
fn stage_input(wire: FjDiffusionRecipe) -> diffusion_frame::StageInput {
    diffusion_frame::StageInput {
        domain: wire.domain,
        family: wire.family,
        group_weights: wire.group_weights,
        group_lambda_um: wire.group_lambda_um,
        warmth: wire.warmth,
        spatial_scale: wire.spatial_scale,
        scatter_fraction: wire.scatter_fraction,
        hash: wire.hash,
    }
}
fn empty_stage() -> FjDiffusion {
    FjDiffusion {
        family: 0,
        scatter_fraction: 0.0,
        group_weights: [0.0; 3],
        group_lambda_um: [0.0; 3],
        warmth: 0.0,
        spatial_scale: 0.0,
        pixel_size_um: 0.0,
        radius_px: 0,
        sample_hash: 0,
        hash: 0,
    }
}
fn empty_frame() -> FjDiffusionFrameSet {
    FjDiffusionFrameSet {
        route: 0,
        flags: 0,
        full_frame: FjExtent {
            x: 0,
            y: 0,
            width: 0,
            height: 0,
        },
        camera: empty_stage(),
        enlarger: empty_stage(),
        hash: 0,
    }
}
fn stage_projection(stage: &diffusion_frame::Stage) -> FjDiffusion {
    let fields = stage.sample().fields();
    FjDiffusion {
        family: fields.family,
        scatter_fraction: stage.scatter_fraction(),
        group_weights: fields.group_weights,
        group_lambda_um: fields.group_lambda_um,
        warmth: fields.warmth,
        spatial_scale: fields.spatial_scale,
        pixel_size_um: fields.pixel_size_um,
        radius_px: fields.radius_px,
        sample_hash: fields.hash,
        hash: stage.hash(),
    }
}
fn frame_build(
    input: FjDiffusionFrameInput,
    _fault: u32,
) -> Result<FjDiffusionFrameSet, diffusion_frame::Error> {
    let f = input.full_frame;
    let frame = FrameSet::new(diffusion_frame::FrameInput {
        route: input.route,
        full_frame: [f.x, f.y, f.width, f.height],
        pixel_size_um: input.pixel_size_um,
        camera: stage_input(input.camera),
        enlarger: stage_input(input.enlarger),
    })?;
    let mut wire = empty_frame();
    if let Some(frame) = frame {
        wire.route = frame.route();
        wire.full_frame = input.full_frame;
        wire.flags =
            u32::from(frame.camera().is_some()) | (u32::from(frame.enlarger().is_some()) << 1);
        wire.camera = frame.camera().map_or_else(empty_stage, stage_projection);
        wire.enlarger = frame.enlarger().map_or_else(empty_stage, stage_projection);
        wire.hash = frame.hash();
    }
    #[cfg(feature = "test-support")]
    let wire = diffusion_frame_faults::frame(wire, _fault);
    Ok(wire)
}

/// # Safety
/// Initialized/live/aligned records; writable peers/backing exclusive and disjoint.
pub(crate) unsafe fn sample_prepare(
    input: *const FjDiffusionSampleInput,
    out: *mut FjDiffusionSample,
    failure: *mut FjDiffusionFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Delegates the complete synchronous foreign contract to the checked operation.
    unsafe {
        checked(
            Operation::Sample,
            input,
            out,
            failure,
            error,
            FjDiffusionSample::default(),
            sample_build,
        )
    }
}
/// # Safety
/// Initialized/live/aligned records; writable peers/backing exclusive and disjoint.
pub(crate) unsafe fn components_prepare(
    input: *const FjDiffusionSample,
    out: *mut FjDiffusionComponents,
    failure: *mut FjDiffusionFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Delegates the complete synchronous foreign contract to the checked operation.
    unsafe {
        checked(
            Operation::Components,
            input,
            out,
            failure,
            error,
            FjDiffusionComponents::default(),
            components_build,
        )
    }
}
/// # Safety
/// Initialized/live/aligned records; writable peers/backing exclusive and disjoint.
pub(crate) unsafe fn frame_prepare(
    input: *const FjDiffusionFrameInput,
    out: *mut FjDiffusionFrameSet,
    failure: *mut FjDiffusionFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Delegates the complete synchronous foreign contract to the checked operation.
    unsafe {
        checked(
            Operation::Frame,
            input,
            out,
            failure,
            error,
            empty_frame(),
            frame_build,
        )
    }
}

#[cfg(feature = "test-support")]
pub(crate) mod diffusion_frame_faults {
    use super::*;
    use std::cell::Cell;
    thread_local! { static NEXT: Cell<(u32,u32)> = const { Cell::new((0,0)) }; }
    pub(crate) fn arm(operation: u32, fault: u32) -> FjStatus {
        if !(1..=3).contains(&operation)
            || fault > 6
            || (operation == 2 && fault > 2)
            || (fault == 6 && operation != 3)
        {
            return status(2);
        }
        NEXT.set(if fault == 0 {
            (0, 0)
        } else {
            (operation, fault)
        });
        status(0)
    }
    pub(super) fn take(operation: u32) -> u32 {
        let (selected, fault) = NEXT.get();
        if selected == operation {
            NEXT.set((0, 0));
            fault
        } else {
            0
        }
    }
    pub(super) fn sample(mut wire: FjDiffusionSample, fault: u32) -> FjDiffusionSample {
        match fault {
            3 => wire.family = 99,
            4 => wire.radius_px = 0,
            5 => wire.hash = 0,
            _ => {}
        }
        wire
    }
    pub(super) fn frame(mut wire: FjDiffusionFrameSet, fault: u32) -> FjDiffusionFrameSet {
        let stage = if wire.flags & 1 != 0 {
            &mut wire.camera
        } else {
            &mut wire.enlarger
        };
        match fault {
            3 => stage.family = 99,
            4 => stage.radius_px = 0,
            5 => stage.sample_hash = 0,
            6 => wire.flags = 99,
            _ => {}
        }
        wire
    }
}

#[unsafe(no_mangle)]
/// # Safety
/// Same live/aligned/exclusive/disjoint record obligations as sample_prepare.
unsafe extern "C" fn fj_legacy_diffusion_sample_prepare(
    input: *const FjDiffusionSampleInput,
    out: *mut FjDiffusionSample,
    failure: *mut FjDiffusionFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The export shares the complete checked synchronous operation.
    unsafe { sample_prepare(input, out, failure, error) }
}

#[unsafe(no_mangle)]
/// # Safety
/// Same live/aligned/exclusive/disjoint record obligations as components_prepare.
unsafe extern "C" fn fj_legacy_diffusion_components_prepare(
    input: *const FjDiffusionSample,
    out: *mut FjDiffusionComponents,
    failure: *mut FjDiffusionFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The export shares the complete checked synchronous operation.
    unsafe { components_prepare(input, out, failure, error) }
}

#[unsafe(no_mangle)]
/// # Safety
/// Same live/aligned/exclusive/disjoint record obligations as frame_prepare.
unsafe extern "C" fn fj_legacy_diffusion_frame_prepare(
    input: *const FjDiffusionFrameInput,
    out: *mut FjDiffusionFrameSet,
    failure: *mut FjDiffusionFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The export shares the complete checked synchronous operation.
    unsafe { frame_prepare(input, out, failure, error) }
}

const _: () = {
    assert!(size_of::<FjDiffusionSampleInput>() == 88 && align_of::<FjDiffusionSampleInput>() == 8);
    assert!(std::mem::offset_of!(FjDiffusionSampleInput, family) == 0);
    assert!(std::mem::offset_of!(FjDiffusionSampleInput, group_weights) == 8);
    assert!(std::mem::offset_of!(FjDiffusionSampleInput, group_lambda_um) == 32);
    assert!(std::mem::offset_of!(FjDiffusionSampleInput, warmth) == 56);
    assert!(std::mem::offset_of!(FjDiffusionSampleInput, spatial_scale) == 64);
    assert!(std::mem::offset_of!(FjDiffusionSampleInput, pixel_size_um) == 72);
    assert!(std::mem::offset_of!(FjDiffusionSampleInput, width) == 80);
    assert!(std::mem::offset_of!(FjDiffusionSampleInput, height) == 84);
};
const _: fn(FjDiffusionSampleInput) = |v| {
    let _: u32 = v.family;
    let _: [f64; 3] = v.group_weights;
    let _: [f64; 3] = v.group_lambda_um;
    let _: f64 = v.warmth;
    let _: f64 = v.spatial_scale;
    let _: f64 = v.pixel_size_um;
    let _: i32 = v.width;
    let _: i32 = v.height;
};
const _: () = {
    assert!(size_of::<FjDiffusionSample>() == 96 && align_of::<FjDiffusionSample>() == 8);
    assert!(std::mem::offset_of!(FjDiffusionSample, family) == 0);
    assert!(std::mem::offset_of!(FjDiffusionSample, group_weights) == 8);
    assert!(std::mem::offset_of!(FjDiffusionSample, group_lambda_um) == 32);
    assert!(std::mem::offset_of!(FjDiffusionSample, warmth) == 56);
    assert!(std::mem::offset_of!(FjDiffusionSample, spatial_scale) == 64);
    assert!(std::mem::offset_of!(FjDiffusionSample, pixel_size_um) == 72);
    assert!(std::mem::offset_of!(FjDiffusionSample, radius_px) == 80);
    assert!(std::mem::offset_of!(FjDiffusionSample, hash) == 88);
};
const _: fn(FjDiffusionSample) = |v| {
    let _: u32 = v.family;
    let _: [f64; 3] = v.group_weights;
    let _: [f64; 3] = v.group_lambda_um;
    let _: f64 = v.warmth;
    let _: f64 = v.spatial_scale;
    let _: f64 = v.pixel_size_um;
    let _: i32 = v.radius_px;
    let _: u64 = v.hash;
};
const _: () = {
    assert!(size_of::<FjDiffusionComponents>() == 288 && align_of::<FjDiffusionComponents>() == 8);
    assert!(std::mem::offset_of!(FjDiffusionComponents, lambda_px) == 0);
    assert!(std::mem::offset_of!(FjDiffusionComponents, red_weights) == 72);
    assert!(std::mem::offset_of!(FjDiffusionComponents, green_weights) == 144);
    assert!(std::mem::offset_of!(FjDiffusionComponents, blue_weights) == 216);
};
const _: fn(FjDiffusionComponents) = |v| {
    let _: [f64; 9] = v.lambda_px;
    let _: [f64; 9] = v.red_weights;
    let _: [f64; 9] = v.green_weights;
    let _: [f64; 9] = v.blue_weights;
};
const _: () = {
    assert!(size_of::<FjDiffusionFrameInput>() == 240 && align_of::<FjDiffusionFrameInput>() == 8);
    assert!(std::mem::offset_of!(FjDiffusionFrameInput, route) == 0);
    assert!(std::mem::offset_of!(FjDiffusionFrameInput, full_frame) == 4);
    assert!(std::mem::offset_of!(FjDiffusionFrameInput, pixel_size_um) == 24);
    assert!(std::mem::offset_of!(FjDiffusionFrameInput, camera) == 32);
    assert!(std::mem::offset_of!(FjDiffusionFrameInput, enlarger) == 136);
};
const _: fn(FjDiffusionFrameInput) = |v| {
    let _: u32 = v.route;
    let _: FjExtent = v.full_frame;
    let _: f64 = v.pixel_size_um;
    let _: FjDiffusionRecipe = v.camera;
    let _: FjDiffusionRecipe = v.enlarger;
};
const _: () = {
    assert!(size_of::<FjDiffusionFrameSet>() == 256 && align_of::<FjDiffusionFrameSet>() == 8);
    assert!(std::mem::offset_of!(FjDiffusionFrameSet, route) == 0);
    assert!(std::mem::offset_of!(FjDiffusionFrameSet, flags) == 4);
    assert!(std::mem::offset_of!(FjDiffusionFrameSet, full_frame) == 8);
    assert!(std::mem::offset_of!(FjDiffusionFrameSet, camera) == 24);
    assert!(std::mem::offset_of!(FjDiffusionFrameSet, enlarger) == 136);
    assert!(std::mem::offset_of!(FjDiffusionFrameSet, hash) == 248);
};
const _: fn(FjDiffusionFrameSet) = |v| {
    let _: u32 = v.route;
    let _: u32 = v.flags;
    let _: FjExtent = v.full_frame;
    let _: FjDiffusion = v.camera;
    let _: FjDiffusion = v.enlarger;
    let _: u64 = v.hash;
};
const _: () = {
    assert!(size_of::<FjDiffusionFailure>() == 12 && align_of::<FjDiffusionFailure>() == 4);
    assert!(std::mem::offset_of!(FjDiffusionFailure, reason) == 0);
    assert!(std::mem::offset_of!(FjDiffusionFailure, stage) == 4);
    assert!(std::mem::offset_of!(FjDiffusionFailure, field) == 8);
};
const _: fn(FjDiffusionFailure) = |v| {
    let _: u32 = v.reason;
    let _: u32 = v.stage;
    let _: u32 = v.field;
};
const _: unsafe extern "C" fn(
    *const FjDiffusionSampleInput,
    *mut FjDiffusionSample,
    *mut FjDiffusionFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_legacy_diffusion_sample_prepare;
const _: unsafe extern "C" fn(
    *const FjDiffusionSample,
    *mut FjDiffusionComponents,
    *mut FjDiffusionFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_legacy_diffusion_components_prepare;
const _: unsafe extern "C" fn(
    *const FjDiffusionFrameInput,
    *mut FjDiffusionFrameSet,
    *mut FjDiffusionFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_legacy_diffusion_frame_prepare;

const _: () = {
    assert!(size_of::<FjDiffusion>() == 112 && align_of::<FjDiffusion>() == 8);
    assert!(std::mem::offset_of!(FjDiffusion, family) == 0);
    assert!(std::mem::offset_of!(FjDiffusion, scatter_fraction) == 8);
    assert!(std::mem::offset_of!(FjDiffusion, group_weights) == 16);
    assert!(std::mem::offset_of!(FjDiffusion, group_lambda_um) == 40);
    assert!(std::mem::offset_of!(FjDiffusion, warmth) == 64);
    assert!(std::mem::offset_of!(FjDiffusion, spatial_scale) == 72);
    assert!(std::mem::offset_of!(FjDiffusion, pixel_size_um) == 80);
    assert!(std::mem::offset_of!(FjDiffusion, radius_px) == 88);
    assert!(std::mem::offset_of!(FjDiffusion, sample_hash) == 96);
    assert!(std::mem::offset_of!(FjDiffusion, hash) == 104);
};
const _: fn(FjDiffusion) = |v| {
    let _: u32 = v.family;
    let _: f64 = v.scatter_fraction;
    let _: [f64; 3] = v.group_weights;
    let _: [f64; 3] = v.group_lambda_um;
    let _: f64 = v.warmth;
    let _: f64 = v.spatial_scale;
    let _: f64 = v.pixel_size_um;
    let _: i32 = v.radius_px;
    let _: u64 = v.sample_hash;
    let _: u64 = v.hash;
};

const _: () = {
    assert!(diffusion_frame::Field::PixelSizeUm as u32 == 1);
    assert!(diffusion_frame::Field::FullFrameExtent as u32 == 2);
    assert!(diffusion_frame::Field::Family as u32 == 3);
    assert!(diffusion_frame::Field::RadiusPixels as u32 == 4);
    assert!(diffusion_frame::Field::SampleDescriptorHash as u32 == 5);
    assert!(diffusion_frame::Field::SampleDescriptorInactive as u32 == 6);
    assert!(diffusion_frame::Field::GroupWeights as u32 == 7);
    assert!(diffusion_frame::Field::GroupCentersUm as u32 == 8);
    assert!(diffusion_frame::Field::EffectiveWarmth as u32 == 9);
    assert!(diffusion_frame::Field::SpatialScale as u32 == 10);
    assert!(diffusion_frame::Field::ComponentExpansion as u32 == 11);
    assert!(diffusion_frame::Field::ComponentCount as u32 == 12);
    assert!(diffusion_frame::Field::HaloChannelWeights as u32 == 13);
    assert!(diffusion_frame::Field::ComponentOutput as u32 == 14);
    assert!(diffusion_frame::Field::Route as u32 == 15);
    assert!(diffusion_frame::Field::FullFrameDomain as u32 == 16);
    assert!(diffusion_frame::Field::EnlargerRoute as u32 == 17);
    assert!(diffusion_frame::Field::LinearDomain as u32 == 18);
    assert!(diffusion_frame::Field::StageSample as u32 == 19);
    assert!(diffusion_frame::Field::StageHash as u32 == 20);
    assert!(diffusion_frame::Field::FrameSetHash as u32 == 21);
    assert!(
        Operation::Sample as u32 == 1
            && Operation::Components as u32 == 2
            && Operation::Frame as u32 == 3
    );
};
