//! SF_TEMP_BRIDGE_FRAME_GEOMETRY: callback-local values; remove with the final
//! native OFX preparation consumer, no later than S6.D. S4.E owns state bridges.

use std::mem::{align_of, size_of};
use std::panic::{AssertUnwindSafe, catch_unwind};

use film_juicer_core::frame::{Bounds, Domain, Geometry};

use crate::cuda::sys::{
    FJ_API_NONE, FJ_STATUS_INTERNAL_FAILURE, FJ_STATUS_PREPARATION_FAILURE, FJ_STATUS_SUCCESS,
    FJ_STATUS_UNSUPPORTED_INPUT, FjErrorBuffer, FjExtent, FjRect, FjStatus,
};

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjFrameGeometryInput {
    pub(crate) full_frame: FjExtent,
    pub(crate) film_format_long_edge_mm: f32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjFrameGeometry {
    pub(crate) full_frame: FjExtent,
    pub(crate) pixel_size_um: f32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjFrameGeometryFailure {
    pub(crate) reason: u32,
}

fn valid<T>(pointer: *const T) -> bool {
    !pointer.is_null()
        && (pointer as usize).is_multiple_of(align_of::<T>())
        && (pointer as usize).checked_add(size_of::<T>()).is_some()
}

struct Diagnostic {
    record: *mut FjErrorBuffer,
    data: *mut u8,
    capacity: usize,
}
impl Diagnostic {
    /// # Safety
    /// Optional initialized live record and nonempty backing are exclusive and disjoint.
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
        // SAFETY: Numeric record admission precedes the caller-authorized live access.
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
        // SAFETY: Admission established live disjoint backing, bounded copy and NUL.
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
enum Failure {
    Raw = 1,
    Range = 2,
    Panic = 3,
}
impl Failure {
    fn status(self) -> FjStatus {
        FjStatus {
            category: match self {
                Self::Raw => FJ_STATUS_UNSUPPORTED_INPUT,
                Self::Range => FJ_STATUS_PREPARATION_FAILURE,
                Self::Panic => FJ_STATUS_INTERNAL_FAILURE,
            },
            api: FJ_API_NONE,
            native_code: 0,
        }
    }
    fn message(self) -> &'static str {
        match self {
            Self::Raw => "invalid frame geometry arguments",
            Self::Range => "frame descriptor dimensions exceed signed integer range",
            Self::Panic => "frame geometry preparation panicked",
        }
    }
}

/// # Safety
/// T is a fixed frame output with valid all-zero fields and no owned storage.
/// Each numerically valid writable peer is initialized/live/exclusive and disjoint.
unsafe fn clear<T>(out: *mut T, failure: *mut FjFrameGeometryFailure) {
    // SAFETY: Clear each independently admitted peer even when the other is malformed.
    unsafe {
        if valid(out) {
            std::ptr::write_bytes(out, 0, 1);
        }
        if valid(failure) {
            (*failure).reason = 0;
        }
    }
}

/// # Safety
/// T is the same zero-valid fixed frame output as clear; foreign records retain
/// the entry operation’s admitted live/exclusive/disjoint obligations.
unsafe fn reject<T>(
    out: *mut T,
    failure: *mut FjFrameGeometryFailure,
    error: *mut FjErrorBuffer,
    reason: Failure,
) -> FjStatus {
    // SAFETY: Independent numeric checks precede every authorized writable access.
    unsafe {
        clear(out, failure);
        if valid(failure) {
            (*failure).reason = reason as u32;
        }
        if let Ok(mut diagnostic) = Diagnostic::admit(error) {
            diagnostic.write(reason.message());
        }
    }
    reason.status()
}

fn project(domain: Domain) -> FjExtent {
    FjExtent {
        x: domain.x,
        y: domain.y,
        width: domain.width,
        height: domain.height,
    }
}
fn domain(extent: FjExtent) -> Domain {
    Domain {
        x: extent.x,
        y: extent.y,
        width: extent.width,
        height: extent.height,
    }
}

/// # Safety
/// Required records authorize initialized/live aligned storage; writable records
/// and optional diagnostic backing are exclusive/disjoint through synchronous return.
pub(crate) unsafe fn prepare_domain(
    input: *const FjRect,
    out: *mut FjExtent,
    failure: *mut FjFrameGeometryFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Every peer is independently cleared before raw/diagnostic admission.
        unsafe {
            clear(out, failure);
            let _diagnostic = Diagnostic::admit(error).map_err(|_| Failure::Raw)?;
            if !valid(input) || !valid(out) || !valid(failure) {
                return Err(Failure::Raw);
            }
            let input = input.read();
            #[cfg(feature = "test-support")]
            faults::check(0);
            let result = Domain::from_bounds(Bounds {
                x1: input.x1,
                y1: input.y1,
                x2: input.x2,
                y2: input.y2,
            })
            .map_err(|_| Failure::Range)?;
            out.write(project(result));
            Ok(())
        }
    }));
    match result {
        Ok(Ok(())) => FjStatus {
            category: FJ_STATUS_SUCCESS,
            api: FJ_API_NONE,
            native_code: 0,
        },
        // SAFETY: Rejection preserves the original entry's admitted foreign obligations.
        Ok(Err(reason)) => unsafe { reject(out, failure, error, reason) },
        // SAFETY: Whole-operation unwind is contained; clear all independently valid peers.
        Err(_) => unsafe { reject(out, failure, error, Failure::Panic) },
    }
}

/// # Safety
/// Same fixed-record live/exclusive/disjoint obligations as prepare_domain.
pub(crate) unsafe fn prepare_geometry(
    input: *const FjFrameGeometryInput,
    out: *mut FjFrameGeometry,
    failure: *mut FjFrameGeometryFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Raw admission precedes input copying and safe scalar construction.
        unsafe {
            clear(out, failure);
            let _diagnostic = Diagnostic::admit(error).map_err(|_| Failure::Raw)?;
            if !valid(input) || !valid(out) || !valid(failure) {
                return Err(Failure::Raw);
            }
            let input = input.read();
            #[cfg(feature = "test-support")]
            faults::check(1);
            let result = Geometry::new(domain(input.full_frame), input.film_format_long_edge_mm);
            out.write(FjFrameGeometry {
                full_frame: project(result.domain()),
                pixel_size_um: result.pixel_size_um(),
            });
            Ok(())
        }
    }));
    match result {
        Ok(Ok(())) => FjStatus {
            category: FJ_STATUS_SUCCESS,
            api: FJ_API_NONE,
            native_code: 0,
        },
        // SAFETY: Rejection preserves the entry's fixed-record caller obligations.
        Ok(Err(reason)) => unsafe { reject(out, failure, error, reason) },
        // SAFETY: Panic does not escape C; independently clear any completed projection.
        Err(_) => unsafe { reject(out, failure, error, Failure::Panic) },
    }
}

#[unsafe(no_mangle)]
/// # Safety
/// Same synchronous foreign storage contract as prepare_domain.
unsafe extern "C" fn fj_legacy_frame_domain_prepare(
    input: *const FjRect,
    out: *mut FjExtent,
    failure: *mut FjFrameGeometryFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The exported caller supplies the checked body's storage obligations.
    unsafe { prepare_domain(input, out, failure, error) }
}
#[unsafe(no_mangle)]
/// # Safety
/// Same synchronous foreign storage contract as prepare_geometry.
unsafe extern "C" fn fj_legacy_frame_geometry_prepare(
    input: *const FjFrameGeometryInput,
    out: *mut FjFrameGeometry,
    failure: *mut FjFrameGeometryFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The exported caller supplies the checked body's storage obligations.
    unsafe { prepare_geometry(input, out, failure, error) }
}

#[cfg(feature = "test-support")]
pub(crate) mod faults {
    use std::cell::Cell;
    use super::*;
    thread_local! { static PANIC: Cell<[bool; 2]> = const { Cell::new([false; 2]) }; }
    pub(crate) fn arm(operation: u32, fault: u32) -> FjStatus {
        if operation > 1 || fault > 1 {
            return Failure::Raw.status();
        }
        PANIC.with(|slot| {
            let mut armed = slot.get();
            armed[operation as usize] = fault == 1;
            slot.set(armed);
        });
        FjStatus {
            category: FJ_STATUS_SUCCESS,
            api: FJ_API_NONE,
            native_code: 0,
        }
    }
    pub(super) fn check(operation: usize) {
        PANIC.with(|slot| {
            let mut armed = slot.get();
            let panic = armed[operation];
            armed[operation] = false;
            slot.set(armed);
            assert!(!panic, "frame geometry fixture panic");
        });
    }
}

const _: () = {
    assert!(size_of::<FjFrameGeometryInput>() == 20 && align_of::<FjFrameGeometryInput>() == 4);
    assert!(std::mem::offset_of!(FjFrameGeometryInput, full_frame) == 0);
    assert!(std::mem::offset_of!(FjFrameGeometryInput, film_format_long_edge_mm) == 16);
    assert!(size_of::<FjFrameGeometry>() == 20 && align_of::<FjFrameGeometry>() == 4);
    assert!(std::mem::offset_of!(FjFrameGeometry, full_frame) == 0);
    assert!(std::mem::offset_of!(FjFrameGeometry, pixel_size_um) == 16);
    assert!(size_of::<FjFrameGeometryFailure>() == 4 && align_of::<FjFrameGeometryFailure>() == 4);
    assert!(std::mem::offset_of!(FjFrameGeometryFailure, reason) == 0);
};
const _: unsafe extern "C" fn(
    *const FjRect,
    *mut FjExtent,
    *mut FjFrameGeometryFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_legacy_frame_domain_prepare;
const _: unsafe extern "C" fn(
    *const FjFrameGeometryInput,
    *mut FjFrameGeometry,
    *mut FjFrameGeometryFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_legacy_frame_geometry_prepare;

const _: fn(FjRect, FjExtent, FjFrameGeometryInput, FjFrameGeometry, FjFrameGeometryFailure) =
    |bounds, extent, input, out, failure| {
        let _: [i32; 4] = [bounds.x1, bounds.y1, bounds.x2, bounds.y2];
        let _: [i32; 4] = [extent.x, extent.y, extent.width, extent.height];
        let _: FjExtent = input.full_frame;
        let _: f32 = input.film_format_long_edge_mm;
        let _: FjExtent = out.full_frame;
        let _: f32 = out.pixel_size_um;
        let _: u32 = failure.reason;
    };
