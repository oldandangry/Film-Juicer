//! Sole owning native handle. Raw operations stay at this foreign edge.

#![expect(
    clippy::result_large_err,
    reason = "native failures retain category/API/code and a bounded diagnostic without allocation"
)]

use std::ptr::NonNull;

use super::{Failure, sys};

impl Failure {
    fn buffer(&mut self) -> sys::FjErrorBuffer {
        sys::FjErrorBuffer {
            data: self.diagnostic.as_mut_ptr().cast(),
            capacity: self.diagnostic.len(),
            length: 0,
        }
    }

    fn result(mut self, status: sys::FjStatus, length: usize) -> Result<(), Self> {
        self.status = status;
        self.length = length.min(1023);
        if status.category == sys::FJ_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(self)
        }
    }
}

pub(crate) struct Owner {
    handle: Option<NonNull<sys::FjCuda>>,
}

// SAFETY: This owner carries no callback tables or thread-affine context. Native
// registration, terminal operations and ordinary calls serialize their own state.
// CUDA operations still require the caller's current context; only the owning
// handle may move/share, while callback capabilities remain non-Send. Rust borrows
// exclude consuming close during any shared use; the C edge requires host exclusion.
unsafe impl Send for Owner {}
// SAFETY: Shared access only invokes native checked/serialized operations, never
// exposes mutable Rust state or transfers a callback/context capability.
unsafe impl Sync for Owner {}

impl Owner {
    pub(crate) fn new() -> Result<Self, Failure> {
        let mut handle = std::ptr::null_mut();
        let mut failure = Failure::new(sys::FJ_STATUS_SUCCESS, "");
        let mut error = failure.buffer();
        // SAFETY: Output and diagnostic are initialized disjoint local storage.
        let status = unsafe { sys::fj_cuda_create(&mut handle, &mut error) };
        failure.result(status, error.length)?;
        let handle = NonNull::new(handle).ok_or_else(|| {
            Failure::new(
                sys::FJ_STATUS_INTERNAL_FAILURE,
                "native creation returned no owner",
            )
        })?;
        Ok(Self {
            handle: Some(handle),
        })
    }

    pub(crate) fn check_close(&self) -> Result<(), Failure> {
        let mut failure = Failure::new(sys::FJ_STATUS_SUCCESS, "");
        let mut error = failure.buffer();
        // SAFETY: The live owner is retained through this non-consuming query.
        let status = unsafe { sys::fj_cuda_check_terminal(self.raw(), &mut error) };
        failure.result(status, error.length)
    }

    pub(crate) fn retire_instance(&self, token: u64) -> Result<(), Failure> {
        let mut failure = Failure::new(sys::FJ_STATUS_SUCCESS, "");
        let mut error = failure.buffer();
        // SAFETY: The Runtime retains this native owner through synchronous
        // retirement; the immutable token was bound once by its Instance.
        let status = unsafe { sys::fj_cuda_retire_instance(self.raw(), token, &mut error) };
        failure.result(status, error.length)
    }

    pub(crate) fn shutdown(&self) -> Result<(), Failure> {
        let mut failure = Failure::new(sys::FJ_STATUS_SUCCESS, "");
        let mut error = failure.buffer();
        // SAFETY: Native shutdown borrows this live owner and serializes retirement.
        let status = unsafe { sys::fj_cuda_shutdown(self.raw(), &mut error) };
        failure.result(status, error.length)
    }

    pub(crate) fn close(mut self) -> Result<(), Failure> {
        self.destroy()
    }

    pub(crate) fn raw(&self) -> *mut sys::FjCuda {
        self.handle.expect("unconsumed native owner").as_ptr()
    }

    fn destroy(&mut self) -> Result<(), Failure> {
        let Some(handle) = self.handle.take() else {
            return Ok(());
        };
        let mut failure = Failure::new(sys::FJ_STATUS_SUCCESS, "");
        let mut error = failure.buffer();
        // SAFETY: The owning handle is taken before consuming destruction.
        // Native uncertainty retains native metadata only; Drop never retries.
        let status = unsafe { sys::fj_cuda_destroy(handle.as_ptr(), &mut error) };
        failure.result(status, error.length)
    }
}

impl Drop for Owner {
    fn drop(&mut self) {
        let _ = self.destroy();
    }
}
