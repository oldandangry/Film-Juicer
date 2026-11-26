//! Completed diffusion math, native component expansion and checked fixture transport.

use std::mem::{align_of, size_of};
use std::panic::{AssertUnwindSafe, catch_unwind};

use film_juicer_core::optics::diffusion_frame::{self, Components, SampleFields};

use crate::cuda::sys::{FjDiffusion, FjErrorBuffer, FjStatus};

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
    input: *const I,
    out: *mut O,
    failure: *mut FjDiffusionFailure,
    error: *mut FjErrorBuffer,
    empty: O,
    build: fn(I) -> Result<O, diffusion_frame::Error>,
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
                writer.write("invalid diffusion components arguments");
            }
            return status(2);
        }
        // SAFETY: All numeric admissions succeeded; caller supplies live aligned initialized input.
        match build(unsafe { input.read() }) {
            Ok(wire) => {
                // SAFETY: Completed fixed output commits to the admitted exclusive slot.
                unsafe {
                    out.write(wire);
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
                    writer.write("diffusion components preparation panicked");
                }
            }
            status(8)
        }
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
fn components_build(
    input: FjDiffusionSample,
) -> Result<FjDiffusionComponents, diffusion_frame::Error> {
    let components = Components::new(sample_fields(input))?;
    Ok(FjDiffusionComponents {
        lambda_px: *components.lambda_px(),
        red_weights: *components.red_weights(),
        green_weights: *components.green_weights(),
        blue_weights: *components.blue_weights(),
    })
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
            input,
            out,
            failure,
            error,
            FjDiffusionComponents::default(),
            components_build,
        )
    }
}

#[unsafe(no_mangle)]
/// # Safety
/// Same live/aligned/exclusive/disjoint record obligations as components_prepare.
unsafe extern "C" fn fj_diffusion_components_prepare(
    input: *const FjDiffusionSample,
    out: *mut FjDiffusionComponents,
    failure: *mut FjDiffusionFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The export shares the complete checked synchronous operation.
    unsafe { components_prepare(input, out, failure, error) }
}

const _: () = {};
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
const _: () = {};
const _: () = {};
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
    *const FjDiffusionSample,
    *mut FjDiffusionComponents,
    *mut FjDiffusionFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_diffusion_components_prepare;
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
};
