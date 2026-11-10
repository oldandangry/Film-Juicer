//! Private process construction, synchronous projection and consuming C edge.

#![expect(
    clippy::result_large_err,
    reason = "the foreign terminal edge preserves fixed diagnostics through consuming failure cleanup"
)]

use std::mem::{align_of, size_of};
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::asset_bridge::FjAssets;
use crate::cuda::{Failure, sys};
use crate::runtime::Runtime;

#[repr(C)]
pub(crate) struct FjRuntime {
    _opaque: [u8; 0],
}

#[repr(C)]
pub(crate) struct FjRuntimeView {
    assets: *const FjAssets,
    cuda: *mut sys::FjCuda,
}

fn valid<T>(pointer: *const T) -> bool {
    !pointer.is_null()
        && (pointer as usize).is_multiple_of(align_of::<T>())
        && (pointer as usize).checked_add(size_of::<T>()).is_some()
}

fn failure(message: &'static str) -> Failure {
    Failure::new(sys::FJ_STATUS_UNSUPPORTED_INPUT, message)
}

struct Diagnostic {
    record: *mut sys::FjErrorBuffer,
    bytes: *mut u8,
    capacity: usize,
}

impl Diagnostic {
    /// # Safety
    /// Nonnull record and nonempty backing are initialized, aligned, exclusive,
    /// live and disjoint from the other call storage through return.
    unsafe fn admit(record: *mut sys::FjErrorBuffer) -> Result<Self, Failure> {
        let mut diagnostic = Self {
            record,
            bytes: std::ptr::null_mut(),
            capacity: 0,
        };
        if record.is_null() {
            return Ok(diagnostic);
        }
        if !valid(record) {
            return Err(failure("invalid runtime diagnostic record"));
        }
        // SAFETY: The caller authorizes initialized capacity and writable length.
        unsafe {
            std::ptr::addr_of_mut!((*record).length).write(0);
            diagnostic.capacity = std::ptr::addr_of!((*record).capacity).read();
            if diagnostic.capacity == 0 {
                return Ok(diagnostic);
            }
            diagnostic.bytes = std::ptr::addr_of!((*record).data).read().cast();
        }
        if diagnostic.bytes.is_null()
            || diagnostic.capacity > isize::MAX as usize
            || (diagnostic.bytes as usize)
                .checked_add(diagnostic.capacity)
                .is_none()
        {
            return Err(failure("invalid runtime diagnostic extent"));
        }
        // SAFETY: Authorized nonempty backing is exclusive and addressable.
        unsafe {
            diagnostic.bytes.write(0);
        }
        Ok(diagnostic)
    }

    fn result(&mut self, result: Result<(), Failure>) -> sys::FjStatus {
        let Err(failure) = result else {
            return sys::FjStatus {
                category: sys::FJ_STATUS_SUCCESS,
                api: sys::FJ_API_NONE,
                native_code: 0,
            };
        };
        if self.capacity != 0 {
            let bytes = failure.bytes();
            let count = bytes.len().min(self.capacity - 1);
            // SAFETY: Admission establishes this exclusive writable extent.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.bytes, count);
                self.bytes.add(count).write(0);
                std::ptr::addr_of_mut!((*self.record).length).write(count);
            }
        }
        failure.status
    }
}

fn contained(operation: impl FnOnce() -> Result<(), Failure>) -> Result<(), Failure> {
    catch_unwind(AssertUnwindSafe(operation)).unwrap_or_else(|_| {
        Err(Failure::new(
            sys::FJ_STATUS_INTERNAL_FAILURE,
            "runtime boundary panic",
        ))
    })
}

/// # Safety
/// A valid pointer names this live Runtime, immutable until the synchronous call
/// returns. The host excludes terminal close and retains every callback borrow.
unsafe fn runtime<'a>(pointer: *const FjRuntime) -> Result<&'a Runtime, Failure> {
    if !valid(pointer.cast::<Runtime>()) {
        return Err(failure("invalid runtime owner"));
    }
    // SAFETY: Numeric admission precedes the caller-authorized shared borrow.
    Ok(unsafe { &*pointer.cast::<Runtime>() })
}

/// # Safety
/// Path/output/diagnostics meet the private C contract. Output is a disjoint live
/// initialized owning slot; native path backing is immutable through return.
/// Path bytes are borrowed only through return; calls are synchronous on the caller
/// thread and perform no host reentry.
pub(crate) unsafe fn create(
    path: sys::FjPathView,
    out: *mut *mut FjRuntime,
    error: *mut sys::FjErrorBuffer,
) -> sys::FjStatus {
    if valid(out) {
        // SAFETY: Required output authorizes one initialized exclusive slot.
        unsafe {
            out.write(std::ptr::null_mut());
        }
    }
    // SAFETY: The caller supplies the optional diagnostic contract.
    let mut diagnostic = match unsafe { Diagnostic::admit(error) } {
        Ok(value) => value,
        Err(error) => return error.status,
    };
    diagnostic.result(contained(|| {
        if !valid(out) {
            return Err(failure("invalid runtime output"));
        }
        // SAFETY: Existing path admission preserves native representation before
        // any host construction/native registration and rejects malformed input.
        let path = unsafe { crate::asset_bridge::path(path) }
            .map_err(|error| Failure::new(error.category(), error))?;
        let owner = Box::new(Runtime::new(path)?);
        // SAFETY: Complete construction is covered by RAII until publication.
        unsafe {
            out.write(Box::into_raw(owner).cast());
        }
        Ok(())
    }))
}

/// # Safety
/// The Runtime is live and immutable. The exclusive initialized output/error
/// storage is disjoint; both projected borrows expire before terminal close.
/// Calls are synchronous on the caller thread, retain no raw inputs, and perform
/// no host reentry.
pub(crate) unsafe fn borrow(
    owner: *const FjRuntime,
    out: *mut FjRuntimeView,
    error: *mut sys::FjErrorBuffer,
) -> sys::FjStatus {
    if valid(out) {
        // SAFETY: Output is an exclusive initialized record, cleared on failure.
        unsafe {
            out.write(FjRuntimeView {
                assets: std::ptr::null(),
                cuda: std::ptr::null_mut(),
            });
        }
    }
    // SAFETY: Optional diagnostics are caller-authorized exclusive storage.
    let mut diagnostic = match unsafe { Diagnostic::admit(error) } {
        Ok(value) => value,
        Err(error) => return error.status,
    };
    diagnostic.result(contained(|| {
        if !valid(out) {
            return Err(failure("invalid runtime view output"));
        }
        // SAFETY: Matching live Runtime remains borrowed throughout this call.
        let runtime = unsafe { runtime(owner) }?;
        let view = FjRuntimeView {
            assets: std::ptr::from_ref(runtime.assets()).cast(),
            cuda: runtime.native().raw(),
        };
        // SAFETY: The callback-local projection authorizes no owning operation.
        unsafe {
            out.write(view);
        }
        Ok(())
    }))
}

/// # Safety
/// Owner is live shared input and diagnostics are exclusive/disjoint. This
/// precheck establishes neither terminal exclusion nor an admission reservation.
/// Calls are synchronous on the caller thread, retain no raw pointers, and perform
/// no host reentry.
pub(crate) unsafe fn check_close(
    owner: *const FjRuntime,
    error: *mut sys::FjErrorBuffer,
) -> sys::FjStatus {
    // SAFETY: Optional diagnostics obey the C edge contract.
    let mut diagnostic = match unsafe { Diagnostic::admit(error) } {
        Ok(value) => value,
        Err(error) => return error.status,
    };
    diagnostic.result(contained(|| {
        // SAFETY: Owner remains live shared input through the non-consuming call.
        unsafe { runtime(owner) }?.check_close()
    }))
}

/// # Safety
/// Same shared owner/error contract as check_close. Shutdown retains ownership
/// and all projected borrow lifetimes on both success and failure. Calls are
/// synchronous on the caller thread, retain no raw pointers, and perform no host
/// reentry.
pub(crate) unsafe fn shutdown(
    owner: *const FjRuntime,
    error: *mut sys::FjErrorBuffer,
) -> sys::FjStatus {
    // SAFETY: Optional diagnostics obey the C edge contract.
    let mut diagnostic = match unsafe { Diagnostic::admit(error) } {
        Ok(value) => value,
        Err(error) => return error.status,
    };
    diagnostic.result(contained(|| {
        // SAFETY: The live owner is retained throughout native and host cleanup.
        unsafe { runtime(owner) }?.shutdown()
    }))
}

/// # Safety
/// Owner is a matching live owning allocation taken once after successful
/// precheck under host exclusion of ALL callbacks/readers/views/native calls.
/// Diagnostics meet the private storage contract; malformed shape still consumes.
/// Calls are synchronous on the caller thread and perform no host reentry.
pub(crate) unsafe fn destroy(
    owner: *mut FjRuntime,
    error: *mut sys::FjErrorBuffer,
) -> sys::FjStatus {
    if !valid(owner.cast::<Runtime>()) {
        return failure("invalid consuming runtime owner").status;
    }
    // SAFETY: Legal terminal ownership is transferred before diagnostic admission.
    let owner = unsafe { Box::from_raw(owner.cast::<Runtime>()) };
    // SAFETY: A malformed optional buffer is not accessed and cannot stop consume.
    let diagnostic = unsafe { Diagnostic::admit(error) };
    let result = contained(move || (*owner).close());
    match diagnostic {
        Ok(mut diagnostic) => diagnostic.result(result),
        Err(error) => result.map_or_else(|failure| failure.status, |()| error.status),
    }
}

// Signature consumers keep the real edge checked in Rust test builds, whose
// executables have no native CUDA engine. Production/test-support staticlibs
// retain the C export names; no substitute native implementation is linked.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_view_matches_native_c11_layout() {
        assert_eq!(size_of::<FjRuntimeView>(), 16);
        assert_eq!(align_of::<FjRuntimeView>(), 8);
        assert_eq!(std::mem::offset_of!(FjRuntimeView, assets), 0);
        assert_eq!(std::mem::offset_of!(FjRuntimeView, cuda), 8);
    }
}
