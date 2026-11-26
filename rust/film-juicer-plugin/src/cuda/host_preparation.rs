//! Call-local backing for native Gaussian cache misses.

use std::cell::UnsafeCell;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;

use crate::cuda::{edge, sys};
use crate::cuda::optics;

#[derive(Default)]
pub(super) struct Backing {
    pub(super) boundaries: Vec<film_juicer_core::film::dir::boundary::Boundary>,
    gaussians: Vec<film_juicer_core::optics::GaussianKernel>,
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

    pub(super) fn raw(&self) -> sys::FjHostPreparation {
        sys::FjHostPreparation {
            user: self.backing.get().cast(),
            gaussian: Some(gaussian),
            dir_boundary: Some(super::dir_boundary::prepare),
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

/// Synchronous native callback that prepares a Gaussian kernel into call-local backing.
///
/// # Safety
/// The native caller must provide ABI-valid pointers for the callback records and
/// keep the context supplied by `Call::raw` alive through this invocation.
unsafe extern "C" fn gaussian(
    user: *mut c_void,
    input: *const sys::FjGaussianKernelInput,
    out: *mut sys::FjGaussianKernelView,
    failure: *mut sys::FjGaussianKernelFailure,
    error: *mut sys::FjErrorBuffer,
) -> sys::FjStatus {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: The callback's native caller provides valid ABI records; the checked
        // helper validates their extents before dereferencing them or the call context.
        unsafe { gaussian_call(user, input, out, failure, error) }
    }));
    result.unwrap_or_else(|_| {
        // SAFETY: The native callback supplies writable output records when valid.
        unsafe { optics::fail_host_gaussian_view(out, failure, 5) };
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
    input: *const sys::FjGaussianKernelInput,
    out: *mut sys::FjGaussianKernelView,
    failure: *mut sys::FjGaussianKernelFailure,
    error: *mut sys::FjErrorBuffer,
) -> sys::FjStatus {
    if !valid(user.cast::<Backing>()) {
        // SAFETY: The native callback contract supplies either null or one writable view.
        unsafe { optics::fail_host_gaussian_view(out, failure, 1) };
        write_diagnostic(error, "invalid host preparation callback context");
        return edge::status(sys::FJ_STATUS_UNSUPPORTED_INPUT);
    }
    // SAFETY: `user` points to this render's `UnsafeCell<Backing>` and the native
    // callback is synchronous and serialized within the one render call.
    let backing = unsafe { &mut *user.cast::<Backing>() };
    // SAFETY: Candidate vectors remain owned by this call through native return.
    unsafe { optics::prepare_host_gaussian(input, out, failure, error, &mut backing.gaussians) }
}

fn write_diagnostic(error: *mut sys::FjErrorBuffer, message: &str) {
    // SAFETY: The callback ABI requires a valid error buffer when non-null;
    // Diagnostic::admit checks its pointer/length contract before writing.
    if let Ok(mut diagnostic) = unsafe { edge::Diagnostic::admit(error) } {
        diagnostic.write(message);
    }
}
