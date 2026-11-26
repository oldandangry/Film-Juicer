//! Shipping OFX render sequencing across the flat CUDA boundary.

use super::*;
use crate::{instance::Instance, runtime::Runtime};
use film_juicer_core::recipe::RenderState;
use std::panic::{AssertUnwindSafe, catch_unwind};

pub(crate) fn trace_flags() -> u32 {
    // SAFETY: This diagnostic query takes no owner and invokes no host code.
    unsafe { sys::fj_cuda_trace_flags() }
}

pub(crate) fn trace_cutoff(bytes: &[u8]) {
    trace(sys::fj_cuda_trace_cutoff, bytes);
}

fn trace(sink: unsafe extern "C" fn(sys::FjStringView), bytes: &[u8]) {
    // SAFETY: The synchronous diagnostic sink retains no pointer.
    unsafe {
        sink(sys::FjStringView {
            data: bytes.as_ptr().cast(),
            count: bytes.len(),
        })
    };
}

pub(crate) struct Completion<'call> {
    outcome: &'call sys::FjRenderOutcome,
    diagnostic: &'call str,
    fatal: bool,
}

impl Completion<'_> {
    pub(crate) fn deferred_dir_error(&self) -> bool {
        self.outcome.flags & sys::FJ_RENDER_DEFERRED_DIR_ERROR != 0
    }

    pub(crate) fn context_loss_pending(&self) -> bool {
        self.outcome.flags & sys::FJ_RENDER_CONTEXT_LOSS_PENDING != 0
    }

    pub(crate) fn diagnostic(&self) -> &str {
        self.diagnostic
    }

    pub(crate) fn fatal(&self) -> bool {
        self.fatal
            || !matches!(
                self.outcome.status.category,
                sys::FJ_STATUS_SUCCESS | sys::FJ_STATUS_CANCELLED
            )
    }

    pub(crate) fn trace_failure(&self) {
        trace(sys::fj_cuda_trace_failure, self.diagnostic.as_bytes());
    }
}

/// # Safety
/// Frame addresses come from callback-local OFX image leases. The leases and
/// abort context remain valid on this thread until return.
#[expect(
    clippy::too_many_arguments,
    reason = "the edge borrows the existing callback owners and frame facts"
)]
pub(crate) unsafe fn render_callback(
    runtime: &Runtime,
    state: &RenderState,
    raw: &sys::FjFrame,
    policy: &FramePolicy,
    instance: &Instance,
    frame_token: u64,
    query: sys::FjAbortCallback,
    complete: impl FnOnce(Completion<'_>) -> bool,
) -> bool {
    let (outcome, diagnostic, fatal) = match catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: The caller retains the OFX image leases and abort backing.
        let frame = unsafe { frame(raw, frame_token) }?;
        let abort = abort(query);
        crate::cuda::render::render(
            runtime.native(),
            runtime.assets(),
            state,
            &frame,
            policy,
            instance,
            &abort,
        )
    })) {
        Ok(Ok(result)) => result,
        Ok(Err(failure)) => (failed_outcome(failure.status), failure.diagnostic, true),
        Err(_) => (
            failed_outcome(status(sys::FJ_STATUS_INTERNAL_FAILURE)),
            "prepared render panicked".to_owned(),
            true,
        ),
    };

    // The native gate and all Rust preparation borrows have ended. The host
    // may reenter while the faulted epoch remains marked in the registry.
    let fatal = catch_unwind(AssertUnwindSafe(|| {
        complete(Completion {
            outcome: &outcome,
            diagnostic: &diagnostic,
            fatal,
        })
    }))
    .unwrap_or(true);

    if outcome.flags & sys::FJ_RENDER_CONTEXT_LOSS_PENDING != 0 {
        let identity = outcome.retirement;
        if let Err(failure) = runtime.native().retire_context(&identity) {
            trace(sys::fj_cuda_trace_failure, failure.diagnostic.as_bytes());
        }
        let context = identity.context;
        if let Err(error) = instance.invalidate_submission(
            identity.instance_token,
            context.device_id,
            context.context,
            context.epoch,
            identity.submission_id,
        ) {
            trace(sys::fj_cuda_trace_failure, error.to_string().as_bytes());
        }
    }
    fatal
}
