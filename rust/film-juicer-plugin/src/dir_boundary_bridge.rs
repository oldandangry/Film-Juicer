//! SF_TEMP_BRIDGE_DIR_BOUNDARY: literal native model transport. S4.E narrows
//! execution inputs; remove at the last native model caller, no later than S6.D.

use std::mem::{align_of, size_of};
use std::panic::{AssertUnwindSafe, catch_unwind};

use film_juicer_core::film::dir::boundary::{Boundary, Error, Input};

use crate::cuda::sys::{FjDirGaussian, FjErrorBuffer, FjStatus};

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjDirBoundaryInput {
    pub(crate) component: FjDirGaussian,
    pub(crate) width: i32,
    pub(crate) height: i32,
}

#[repr(C)]
pub(crate) struct FjDirBoundaryOwner {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjDirBoundaryView {
    pub(crate) weights: *const f64,
    pub(crate) count: usize,
    pub(crate) identity: u64,
    pub(crate) width: i32,
    pub(crate) height: i32,
    pub(crate) horizontal_length: i32,
    pub(crate) vertical_length: i32,
    pub(crate) horizontal_terminal_size: i32,
    pub(crate) vertical_terminal_size: i32,
    pub(crate) horizontal_terminal: [f64; 9],
    pub(crate) vertical_terminal: [f64; 9],
}

const EMPTY_VIEW: FjDirBoundaryView = FjDirBoundaryView {
    weights: std::ptr::null(),
    count: 0,
    identity: 0,
    width: 0,
    height: 0,
    horizontal_length: 0,
    vertical_length: 0,
    horizontal_terminal_size: 0,
    vertical_terminal_size: 0,
    horizontal_terminal: [0.0; 9],
    vertical_terminal: [0.0; 9],
};

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjDirBoundaryFailure {
    pub(crate) reason: u32,
    pub(crate) axis: u32,
    pub(crate) row: u32,
    pub(crate) length: i32,
    pub(crate) sigma_px: f32,
    pub(crate) sum: f64,
    pub(crate) tolerance: f64,
}

fn failure(reason: u32) -> FjDirBoundaryFailure {
    FjDirBoundaryFailure {
        reason,
        axis: 0,
        row: 3,
        length: 0,
        sigma_px: 0.0,
        sum: 0.0,
        tolerance: 0.0,
    }
}

impl From<Error> for FjDirBoundaryFailure {
    fn from(value: Error) -> Self {
        Self {
            reason: value.reason() as u32,
            axis: value.axis(),
            row: value.row(),
            length: value.length(),
            sigma_px: value.sigma_px(),
            sum: value.sum(),
            tolerance: value.tolerance(),
        }
    }
}

const MESSAGES: [&str; 14] = [
    "",
    "spatial DIR reflected-boundary arguments are invalid",
    "spatial DIR reflected-boundary request is invalid",
    "spatial DIR reflected-boundary axis size is invalid",
    "spatial DIR reflected-boundary policy is invalid",
    "spatial DIR causal reflected-boundary solve is singular",
    "spatial DIR causal reflected-boundary weights are nonfinite",
    "spatial DIR causal reflected-boundary certification failed",
    "spatial DIR reflected terminal system escaped its tail",
    "spatial DIR reflected terminal solve is singular",
    "spatial DIR reflected terminal certification failed",
    "spatial DIR reflected-boundary storage size is invalid",
    "spatial DIR reflected-boundary allocation failed",
    "spatial DIR reflected-boundary preparation panicked",
];

fn status(reason: u32) -> FjStatus {
    FjStatus {
        category: match reason {
            0 => 0,
            1 => 2,
            12 => 4,
            13 => 8,
            _ => 3,
        },
        api: 0,
        native_code: 0,
    }
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
    /// The optional initialized record and nonempty backing are live/exclusive/disjoint.
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
        // SAFETY: Independent numeric admission precedes caller-authorized record access.
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
        // SAFETY: Admitted exclusive backing has capacity; count reserves its NUL.
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

/// # Safety
/// Independently authorized writable outputs/diagnostics are live, aligned and
/// mutually disjoint, including backing and input. Admission proves no provenance.
unsafe fn call<T: Copy>(
    out: *mut T,
    empty: T,
    details: *mut FjDirBoundaryFailure,
    error: *mut FjErrorBuffer,
    release: bool,
    body: impl FnOnce() -> Result<T, FjDirBoundaryFailure>,
) -> FjStatus {
    let valid_out = valid(out);
    let valid_details = valid(details);
    // SAFETY: Clear each independently admitted peer before other admission.
    unsafe {
        if valid_out {
            out.write(empty);
        }
        if valid_details {
            details.write(failure(0));
        }
    }
    // SAFETY: Optional diagnostics have the same authorized storage/disjointness contract.
    let mut diagnostic = unsafe { Diagnostic::admit(error) };
    let result = if !valid_out || !valid_details || diagnostic.is_err() {
        Err(failure(1))
    } else {
        catch_unwind(AssertUnwindSafe(body)).unwrap_or_else(|_| Err(failure(13)))
    };
    match result {
        Ok(value) => {
            // SAFETY: Successful independent admission established exclusive output storage.
            unsafe {
                out.write(value);
            }
            status(0)
        }
        Err(rejected) => {
            if valid_details {
                // SAFETY: This peer was independently admitted, even on another raw failure.
                unsafe {
                    details.write(rejected);
                }
            }
            if let Ok(writer) = &mut diagnostic {
                writer.write(if release && rejected.reason == 13 {
                    "spatial DIR reflected-boundary release panicked"
                } else {
                    MESSAGES[rejected.reason as usize]
                });
            }
            status(rejected.reason)
        }
    }
}

fn project(raw: &FjDirBoundaryInput) -> Result<Input, FjDirBoundaryFailure> {
    let c = &raw.component;
    if !(c.reference_operator == 3 && c.weight > 0.0) {
        return Err(failure(2));
    }
    Input::new(
        [
            c.feedforward,
            c.feedback[0],
            c.feedback[1],
            c.feedback[2],
            c.normalization_denominator,
            c.feedforward_numerator,
            c.feedback_numerators[0],
            c.feedback_numerators[1],
            c.feedback_numerators[2],
        ],
        raw.width,
        raw.height,
        c.boundary_truncation_accuracy,
        c.boundary_certification_tolerance,
        c.boundary_derivation_version,
        c.sigma_px,
    )
    .map_err(Into::into)
}

fn take_fault(operation: usize) -> u32 {
    #[cfg(feature = "test-support")]
    {
        faults::take(operation)
    }
    #[cfg(not(feature = "test-support"))]
    {
        let _ = operation;
        0
    }
}
fn panic_fault(fault: u32) {
    #[cfg(feature = "test-support")]
    if fault == 1 {
        panic!("spatial DIR boundary fixture panic");
    }
    #[cfg(not(feature = "test-support"))]
    let _ = fault;
}

/// # Safety
/// Input and disjoint writable peers are initialized/live/aligned through return.
pub(crate) unsafe fn describe(
    input: *const FjDirBoundaryInput,
    out: *mut u64,
    details: *mut FjDirBoundaryFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    let fault = take_fault(0);
    // SAFETY: Common admission clears peers before authorizing the immutable input borrow.
    unsafe {
        call(out, 0, details, error, false, || {
            if !valid(input) {
                return Err(failure(1));
            }
            panic_fault(fault);
            project(&*input).map(|input| input.key().value())
        })
    }
}

/// # Safety
/// Input/peers are live/aligned/disjoint; one successful owner transfers to release.
pub(crate) unsafe fn acquire(
    input: *const FjDirBoundaryInput,
    out: *mut *mut FjDirBoundaryOwner,
    details: *mut FjDirBoundaryFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    let fault = take_fault(1);
    #[cfg(feature = "test-support")]
    faults::ACQUIRES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // SAFETY: Common admission precedes the authorized immutable input borrow and ownership transfer.
    unsafe {
        call(out, std::ptr::null_mut(), details, error, false, || {
            if !valid(input) {
                return Err(failure(1));
            }
            panic_fault(fault);
            let input = project(&*input)?;
            #[cfg(feature = "test-support")]
            let _reservation =
                film_juicer_core::film::dir::boundary::test_support::Reservation::new(
                    if (2..=8).contains(&fault) {
                        fault - 1
                    } else {
                        0
                    },
                );
            let boundary = Boundary::new(input).map_err(FjDirBoundaryFailure::from)?;
            // The small Box retains the established infallible allocator-abort limitation.
            let owner = Box::new(boundary);
            #[cfg(feature = "test-support")]
            faults::LIVE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(Box::into_raw(owner).cast())
        })
    }
}

/// # Safety
/// Borrow a legitimate live owner excluding release; exclusive peers are disjoint.
/// The immutable storage and all returned views expire at matching release.
pub(crate) unsafe fn view(
    owner: *const FjDirBoundaryOwner,
    out: *mut FjDirBoundaryView,
    details: *mut FjDirBoundaryFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    let fault = take_fault(2);
    #[cfg(feature = "test-support")]
    faults::VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // SAFETY: Common admission and caller-guaranteed owner provenance precede the borrow.
    unsafe {
        call(out, EMPTY_VIEW, details, error, false, || {
            if !valid(owner.cast::<Boundary>()) {
                return Err(failure(1));
            }
            panic_fault(fault);
            let boundary = &*owner.cast::<Boundary>();
            let result = FjDirBoundaryView {
                weights: boundary.weights().as_ptr(),
                count: boundary.weights().len(),
                identity: boundary.key().value(),
                width: boundary.width(),
                height: boundary.height(),
                horizontal_length: boundary.horizontal().length(),
                vertical_length: boundary.vertical().length(),
                horizontal_terminal_size: boundary.horizontal().terminal_size(),
                vertical_terminal_size: boundary.vertical().terminal_size(),
                horizontal_terminal: *boundary.horizontal().terminal(),
                vertical_terminal: *boundary.vertical().terminal(),
            };
            #[cfg(feature = "test-support")]
            let result = faults::corrupt(result, fault);
            Ok(result)
        })
    }
}

/// # Safety
/// NULL or one matching live owner transfers exactly once, excluding all readers.
/// Malformed diagnostics and contained unwind still consume an admitted owner.
pub(crate) unsafe fn release(
    owner: *mut FjDirBoundaryOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    let fault = take_fault(3);
    #[cfg(feature = "test-support")]
    faults::RELEASES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let admitted = owner.is_null() || valid(owner.cast::<Boundary>());
    let owned = if admitted && !owner.is_null() {
        #[cfg(feature = "test-support")]
        faults::LIVE.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        // SAFETY: The caller transfers this exact Box before any fallible diagnostic work.
        Some(unsafe { Box::from_raw(owner.cast::<Boundary>()) })
    } else {
        None
    };
    let mut out = 0_u32;
    let mut details = failure(0);
    // SAFETY: Local peers and caller-authorized diagnostics satisfy common admission.
    // The captured Box drops even if malformed diagnostics reject before invocation.
    unsafe {
        call(&mut out, 0, &mut details, error, true, move || {
            if !admitted {
                return Err(failure(1));
            }
            panic_fault(fault);
            drop(owned);
            Ok(0)
        })
    }
}

/// # Safety
/// Identical initialized, live, disjoint record contract to describe.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_dir_boundary_describe(
    input: *const FjDirBoundaryInput,
    out: *mut u64,
    details: *mut FjDirBoundaryFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Forward the unchanged checked contract.
    unsafe { describe(input, out, details, error) }
}
/// # Safety
/// Identical storage and ownership-transfer contract to acquire.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_dir_boundary_acquire(
    input: *const FjDirBoundaryInput,
    out: *mut *mut FjDirBoundaryOwner,
    details: *mut FjDirBoundaryFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Forward the unchanged checked contract.
    unsafe { acquire(input, out, details, error) }
}
/// # Safety
/// Identical legitimate-owner borrow and storage-expiry contract to view.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_dir_boundary_view(
    owner: *const FjDirBoundaryOwner,
    out: *mut FjDirBoundaryView,
    details: *mut FjDirBoundaryFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Forward the unchanged checked contract.
    unsafe { view(owner, out, details, error) }
}
/// # Safety
/// Identical consume-once ownership and diagnostic contract to release.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_dir_boundary_release(
    owner: *mut FjDirBoundaryOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Forward the unchanged checked contract.
    unsafe { release(owner, error) }
}

#[cfg(feature = "test-support")]
pub(crate) mod faults {
    use super::*;
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU64, Ordering};
    thread_local! { static PENDING:Cell<[u32;4]>=const { Cell::new([0;4]) }; }
    pub(crate) static ACQUIRES: AtomicU64 = AtomicU64::new(0);
    pub(crate) static VIEWS: AtomicU64 = AtomicU64::new(0);
    pub(crate) static RELEASES: AtomicU64 = AtomicU64::new(0);
    pub(crate) static LIVE: AtomicU64 = AtomicU64::new(0);
    pub(crate) fn arm(operation: u32, fault: u32) -> FjStatus {
        if operation > 3
            || !(fault <= 1
                || operation == 1 && (2..=8).contains(&fault)
                || operation == 2 && (9..=12).contains(&fault))
        {
            return status(1);
        }
        let mut pending = PENDING.get();
        pending[operation as usize] = fault;
        PENDING.set(pending);
        status(0)
    }
    pub(super) fn take(operation: usize) -> u32 {
        let mut pending = PENDING.get();
        let fault = pending[operation];
        pending[operation] = 0;
        PENDING.set(pending);
        fault
    }
    pub(super) fn corrupt(mut view: FjDirBoundaryView, fault: u32) -> FjDirBoundaryView {
        match fault {
            9 => view.count += 1,
            10 => view.horizontal_length = i32::MAX,
            11 => view.identity ^= 1,
            12 => view.horizontal_terminal_size = 0,
            _ => {}
        }
        view
    }
    pub(crate) fn counts() -> [u64; 4] {
        [
            ACQUIRES.load(Ordering::Relaxed),
            VIEWS.load(Ordering::Relaxed),
            RELEASES.load(Ordering::Relaxed),
            LIVE.load(Ordering::Relaxed),
        ]
    }
}

const _: () = {
    assert!(size_of::<FjDirBoundaryInput>() == 120 && align_of::<FjDirBoundaryInput>() == 8);
    assert!(size_of::<FjDirBoundaryView>() == 192 && align_of::<FjDirBoundaryView>() == 8);
    assert!(size_of::<FjDirBoundaryFailure>() == 40 && align_of::<FjDirBoundaryFailure>() == 8);
};

const _: () = {
    assert!(size_of::<FjDirBoundaryInput>() == 120 && align_of::<FjDirBoundaryInput>() == 8);
    assert!(std::mem::offset_of!(FjDirBoundaryInput, component) == 0);
    assert!(std::mem::offset_of!(FjDirBoundaryInput, width) == 112);
    assert!(std::mem::offset_of!(FjDirBoundaryInput, height) == 116);
    assert!(size_of::<FjDirBoundaryView>() == 192 && align_of::<FjDirBoundaryView>() == 8);
    assert!(std::mem::offset_of!(FjDirBoundaryView, weights) == 0);
    assert!(std::mem::offset_of!(FjDirBoundaryView, count) == 8);
    assert!(std::mem::offset_of!(FjDirBoundaryView, identity) == 16);
    assert!(std::mem::offset_of!(FjDirBoundaryView, width) == 24);
    assert!(std::mem::offset_of!(FjDirBoundaryView, height) == 28);
    assert!(std::mem::offset_of!(FjDirBoundaryView, horizontal_length) == 32);
    assert!(std::mem::offset_of!(FjDirBoundaryView, vertical_length) == 36);
    assert!(std::mem::offset_of!(FjDirBoundaryView, horizontal_terminal_size) == 40);
    assert!(std::mem::offset_of!(FjDirBoundaryView, vertical_terminal_size) == 44);
    assert!(std::mem::offset_of!(FjDirBoundaryView, horizontal_terminal) == 48);
    assert!(std::mem::offset_of!(FjDirBoundaryView, vertical_terminal) == 120);
    assert!(size_of::<FjDirBoundaryFailure>() == 40 && align_of::<FjDirBoundaryFailure>() == 8);
    assert!(std::mem::offset_of!(FjDirBoundaryFailure, reason) == 0);
    assert!(std::mem::offset_of!(FjDirBoundaryFailure, axis) == 4);
    assert!(std::mem::offset_of!(FjDirBoundaryFailure, row) == 8);
    assert!(std::mem::offset_of!(FjDirBoundaryFailure, length) == 12);
    assert!(std::mem::offset_of!(FjDirBoundaryFailure, sigma_px) == 16);
    assert!(std::mem::offset_of!(FjDirBoundaryFailure, sum) == 24);
    assert!(std::mem::offset_of!(FjDirBoundaryFailure, tolerance) == 32);
};

const _: fn(FjDirBoundaryInput) = |v| {
    let _: FjDirGaussian = v.component;
    let _: i32 = v.width;
    let _: i32 = v.height;
};
const _: fn(FjDirBoundaryView) = |v| {
    let _: *const f64 = v.weights;
    let _: usize = v.count;
    let _: u64 = v.identity;
    let _: i32 = v.width;
    let _: i32 = v.height;
    let _: i32 = v.horizontal_length;
    let _: i32 = v.vertical_length;
    let _: i32 = v.horizontal_terminal_size;
    let _: i32 = v.vertical_terminal_size;
    let _: [f64; 9] = v.horizontal_terminal;
    let _: [f64; 9] = v.vertical_terminal;
};
const _: fn(FjDirBoundaryFailure) = |v| {
    let _: u32 = v.reason;
    let _: u32 = v.axis;
    let _: u32 = v.row;
    let _: i32 = v.length;
    let _: f32 = v.sigma_px;
    let _: f64 = v.sum;
    let _: f64 = v.tolerance;
};
