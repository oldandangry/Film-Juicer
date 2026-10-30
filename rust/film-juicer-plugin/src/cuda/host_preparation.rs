//! Call-local backing for the native scanner-LUT and Gaussian miss callbacks.

use std::cell::UnsafeCell;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;

use crate::cuda::{edge, sys};
use crate::{optics_bridge, scanner_bridge};

#[derive(Default)]
struct Backing {
    scanner_luts: Vec<scanner_bridge::HostLutCandidate>,
    gaussians: Vec<optics_bridge::HostGaussianCandidate>,
}

/// One render's candidate arrays. Native staging copies any asynchronous upload
/// source before the render callback returns, so these owners never enter a cache.
pub(crate) struct Call {
    backing: UnsafeCell<Backing>,
    thread: PhantomData<Rc<()>>,
}

impl Call {
    pub(crate) fn new() -> Self {
        Self {
            backing: UnsafeCell::new(Backing::default()),
            thread: PhantomData,
        }
    }

    pub(super) fn raw(&self) -> edge::FjHostPreparation {
        edge::FjHostPreparation {
            user: self.backing.get().cast(),
            scanner_lut: Some(scanner_lut),
            gaussian: Some(gaussian),
        }
    }
}

fn valid<T>(pointer: *const T) -> bool {
    !pointer.is_null()
        && (pointer as usize).is_multiple_of(std::mem::align_of::<T>())
        && std::mem::size_of::<T>() <= isize::MAX as usize
        && (pointer as usize)
            .checked_add(std::mem::size_of::<T>())
            .is_some()
}

/// Synchronous native callback that prepares a scanner LUT into call-local backing.
///
/// # Safety
/// The native caller must provide ABI-valid pointers for the callback records and
/// keep the context supplied by `Call::raw` alive through this invocation.
unsafe extern "C" fn scanner_lut(
    user: *mut c_void,
    input: *const scanner_bridge::FjScannerLutInput,
    out: *mut scanner_bridge::FjScannerLutView,
    failure: *mut scanner_bridge::FjScannerLutFailure,
    error: *mut sys::FjErrorBuffer,
) -> sys::FjStatus {
    let acquire_fault = scanner_bridge::take_host_lut_acquire_fault();
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: The callback's native caller provides valid ABI records; the checked
        // helper validates their extents before dereferencing them or the call context.
        unsafe { scanner_lut_call(user, input, out, failure, error, acquire_fault) }
    }));
    result.unwrap_or_else(|_| {
        // SAFETY: The native callback supplies writable output records when valid.
        unsafe { scanner_bridge::fail_host_lut_view(out, failure, 6) };
        write_diagnostic(error, "scanner LUT preparation panicked");
        edge::status(sys::FJ_STATUS_INTERNAL_FAILURE)
    })
}

/// Validate scanner callback records and populate this call's candidate backing.
///
/// # Safety
/// Each non-null input/output/failure/error pointer must be aligned and reference
/// initialized/readable or writable storage for its declared C record. `user`
/// must be null/invalid (which is rejected) or refer to this call's live,
/// exclusively accessed `Backing`; pointer-shape checks cannot validate allocation provenance.
unsafe fn scanner_lut_call(
    user: *mut c_void,
    input: *const scanner_bridge::FjScannerLutInput,
    out: *mut scanner_bridge::FjScannerLutView,
    failure: *mut scanner_bridge::FjScannerLutFailure,
    error: *mut sys::FjErrorBuffer,
    acquire_fault: u32,
) -> sys::FjStatus {
    if !valid(user.cast::<Backing>()) {
        // SAFETY: The native callback contract supplies either null or one writable view.
        unsafe { scanner_bridge::fail_host_lut_view(out, failure, 1) };
        write_diagnostic(error, "invalid host preparation callback context");
        return edge::status(sys::FJ_STATUS_UNSUPPORTED_INPUT);
    }
    // SAFETY: `user` points to this render's `UnsafeCell<Backing>` and the native
    // callback is synchronous and serialized within the one render call.
    let backing = unsafe { &mut *user.cast::<Backing>() };
    // SAFETY: Candidate vectors remain owned by this call through native return.
    unsafe {
        scanner_bridge::prepare_host_lut(
            input,
            out,
            failure,
            error,
            &mut backing.scanner_luts,
            acquire_fault,
        )
    }
}

/// Synchronous native callback that prepares a Gaussian kernel into call-local backing.
///
/// # Safety
/// The native caller must provide ABI-valid pointers for the callback records and
/// keep the context supplied by `Call::raw` alive through this invocation.
unsafe extern "C" fn gaussian(
    user: *mut c_void,
    input: *const optics_bridge::FjGaussianKernelInput,
    out: *mut optics_bridge::FjGaussianKernelView,
    failure: *mut optics_bridge::FjGaussianKernelFailure,
    error: *mut sys::FjErrorBuffer,
) -> sys::FjStatus {
    let acquire_fault = optics_bridge::take_host_gaussian_acquire_fault();
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: The callback's native caller provides valid ABI records; the checked
        // helper validates their extents before dereferencing them or the call context.
        unsafe { gaussian_call(user, input, out, failure, error, acquire_fault) }
    }));
    result.unwrap_or_else(|_| {
        // SAFETY: The native callback supplies writable output records when valid.
        unsafe { optics_bridge::fail_host_gaussian_view(out, failure, 5) };
        write_diagnostic(error, "Gaussian kernel preparation panicked");
        edge::status(sys::FJ_STATUS_INTERNAL_FAILURE)
    })
}

/// Validate Gaussian callback records and populate this call's candidate backing.
///
/// # Safety
/// Each non-null input/output/failure/error pointer must be aligned and reference
/// initialized/readable or writable storage for its declared C record. `user`
/// must either be null/invalid (which is rejected) or refer to this call's live,
/// exclusively accessed `Backing`.
unsafe fn gaussian_call(
    user: *mut c_void,
    input: *const optics_bridge::FjGaussianKernelInput,
    out: *mut optics_bridge::FjGaussianKernelView,
    failure: *mut optics_bridge::FjGaussianKernelFailure,
    error: *mut sys::FjErrorBuffer,
    acquire_fault: u32,
) -> sys::FjStatus {
    if !valid(user.cast::<Backing>()) {
        // SAFETY: The native callback contract supplies either null or one writable view.
        unsafe { optics_bridge::fail_host_gaussian_view(out, failure, 1) };
        write_diagnostic(error, "invalid host preparation callback context");
        return edge::status(sys::FJ_STATUS_UNSUPPORTED_INPUT);
    }
    // SAFETY: `user` points to this render's `UnsafeCell<Backing>` and the native
    // callback is synchronous and serialized within the one render call.
    let backing = unsafe { &mut *user.cast::<Backing>() };
    // SAFETY: Candidate vectors remain owned by this call through native return.
    unsafe {
        optics_bridge::prepare_host_gaussian(
            input,
            out,
            failure,
            error,
            &mut backing.gaussians,
            acquire_fault,
        )
    }
}

fn write_diagnostic(error: *mut sys::FjErrorBuffer, message: &str) {
    // SAFETY: The callback ABI requires a valid error buffer when non-null;
    // Diagnostic::admit checks its pointer/length contract before writing.
    if let Ok(mut diagnostic) = unsafe { edge::Diagnostic::admit(error) } {
        diagnostic.write(message);
    }
}

#[cfg(feature = "test-support")]
pub(crate) type Consumer = unsafe extern "C" fn(
    *mut c_void,
    *const edge::FjHostPreparation,
    *mut sys::FjErrorBuffer,
) -> sys::FjStatus;

/// Invoke one test consumer with the same call-local callback pair used by render.
///
/// # Safety
/// `user` and any storage reached by `consumer` remain live, initialized and
/// disjoint for the synchronous consumer call. The consumer must not retain the
/// callback table or any view returned by it.
#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn fj_test_with_host_preparation(
    user: *mut c_void,
    consumer: Option<Consumer>,
    error: *mut sys::FjErrorBuffer,
) -> sys::FjStatus {
    let consumer = {
        // SAFETY: The caller supplies null or live writable diagnostic storage;
        // admission validates its pointer, extent and backing before use.
        let mut diagnostic = match unsafe { edge::Diagnostic::admit(error) } {
            Ok(diagnostic) => diagnostic,
            Err(()) => return edge::status(sys::FJ_STATUS_UNSUPPORTED_INPUT),
        };
        if user.is_null() {
            diagnostic.write("invalid host preparation user");
            return edge::status(sys::FJ_STATUS_UNSUPPORTED_INPUT);
        }
        let Some(consumer) = consumer else {
            diagnostic.write("missing host preparation consumer");
            return edge::status(sys::FJ_STATUS_UNSUPPORTED_INPUT);
        };
        consumer
    };

    let call = Call::new();
    let preparation = call.raw();
    // SAFETY: The admitted consumer borrows this stack table and its live Call
    // only synchronously, serially on this thread, and contains foreign exceptions.
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        consumer(user, &preparation, error)
    })) {
        Ok(status) => status,
        Err(_) => {
            // SAFETY: The same caller-owned diagnostic record remains live after
            // the contained Rust unwind; admission checks its backing again.
            if let Ok(mut diagnostic) = unsafe { edge::Diagnostic::admit(error) } {
                diagnostic.write("host preparation test consumer panicked");
            }
            edge::status(sys::FJ_STATUS_INTERNAL_FAILURE)
        }
    }
}
