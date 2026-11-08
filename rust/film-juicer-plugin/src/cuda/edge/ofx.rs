//! Immediate stack CallScope connection for the real Rust OFX callback.

use super::*;
use crate::{instance::Instance, runtime::Runtime};
use film_juicer_core::recipe::RenderState;

#[repr(C)]
#[derive(Clone, Copy)]
struct FjCudaCompletion {
    result: FjCallCompletionResult,
    retirement: u32,
    invalidation: sys::FjStatus,
    latch_cleared: u32,
}

const _: () = {
    assert!(std::mem::size_of::<FjCudaCompletion>() == 40);
    assert!(std::mem::align_of::<FjCudaCompletion>() == 4);
    assert!(std::mem::offset_of!(FjCudaCompletion, retirement) == 20);
    assert!(std::mem::offset_of!(FjCudaCompletion, invalidation) == 24);
    assert!(std::mem::offset_of!(FjCudaCompletion, latch_cleared) == 36);
};

unsafe extern "C" {
    fn fj_cuda_call_trace_flags() -> u32;
    fn fj_cuda_call_trace_cutoff(text: sys::FjStringView);
    fn fj_cuda_call_trace_failure(text: sys::FjStringView);
    fn fj_cuda_call(
        owner: *mut sys::FjCuda,
        route: u32,
        run: unsafe extern "C" fn(*mut c_void, FjCudaCallAccess) -> FjCallCompletionResult,
        complete: unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *const FjCallCompletionInput,
        ) -> FjCudaCompletion,
        user: *mut c_void,
    ) -> FjCallCompletionResult;
    fn fj_cuda_call_retire_context(scope: *mut c_void, input: *const FjCallCompletionInput) -> u32;
}

pub(crate) fn trace_flags() -> u32 {
    // SAFETY: The existing native diagnostic sink takes no owner or borrowed
    // arguments and contains exceptions. It performs no CUDA/host callback work.
    unsafe { fj_cuda_call_trace_flags() }
}
pub(crate) fn trace_cutoff(bytes: &[u8]) {
    // SAFETY: This immutable CPU diagnostic backing lives through the immediate
    // no-throw sink call on this thread; native never retains its pointer.
    unsafe {
        fj_cuda_call_trace_cutoff(sys::FjStringView {
            data: bytes.as_ptr().cast(),
            count: bytes.len(),
        });
    }
}

pub(crate) struct Completion<'call> {
    input: &'call FjCallCompletionInput,
    scope: *mut c_void,
    diagnostic: &'call str,
    recovery: &'call std::cell::Cell<(u32, sys::FjStatus, u32)>,
    _thread: PhantomData<Rc<()>>,
}
impl Completion<'_> {
    pub(crate) fn deferred_dir_error(&self) -> bool {
        self.input.outcome.flags & sys::FJ_RENDER_DEFERRED_DIR_ERROR != 0
    }
    pub(crate) fn diagnostic(&self) -> &str {
        self.diagnostic
    }
    #[cfg(feature = "test-support")]
    pub(crate) fn status(&self) -> u32 {
        self.input.outcome.status.category
    }
    pub(crate) fn fatal(&self) -> bool {
        self.input.force_fatal != 0
            || self.input.abort_failed != 0
            || (self.input.render_outcome_valid != 0
                && !matches!(
                    self.input.outcome.status.category,
                    sys::FJ_STATUS_SUCCESS | sys::FJ_STATUS_CANCELLED
                ))
    }
    pub(crate) fn trace_failure(&self) {
        // SAFETY: CallScope owns this length-delimited diagnostic through the
        // synchronous completion. The diagnostic sink neither retains it nor
        // invokes host callbacks; this runs after release and before recovery.
        unsafe {
            fj_cuda_call_trace_failure(self.input.diagnostic);
        }
    }
    pub(crate) fn recover(self, instance: &Instance) {
        // SAFETY: Only the stack native completion introduces this capability,
        // after the gate is released. The scope/input live through immediate
        // return, on this thread; native retains no Rust/host callback or owner.
        let retirement = unsafe { fj_cuda_call_retire_context(self.scope, self.input) };
        if retirement & 1 != 0 {
            let context = self.input.inspected_context;
            let result = instance.invalidate_submission(context.device_id, context.context);
            self.recovery.set((
                retirement,
                status(if result.is_ok() {
                    sys::FJ_STATUS_SUCCESS
                } else {
                    sys::FJ_STATUS_INTERNAL_FAILURE
                }),
                u32::from(result.unwrap_or(false)),
            ));
        }
    }
}

struct Call<'a, F> {
    runtime: &'a Runtime,
    state: &'a RenderState,
    frame: &'a sys::FjFrame,
    policy: &'a FramePolicy,
    instance: &'a Instance,
    frame_token: u64,
    abort: sys::FjAbortCallback,
    complete: std::cell::UnsafeCell<F>,
}
/// # Safety
/// Only the synchronous native entry may invoke this callback, once on the
/// originating thread. `user` points to its live `Call<F>` with retained owners
/// and image backing; `access` belongs to that same stack CallScope. Nested
/// completion may mutate only the disjoint UnsafeCell callback field. Neither
/// native code nor this callback retains the supplied storage after return.
unsafe extern "C" fn run<F: for<'call> FnMut(Completion<'call>) -> bool>(
    user: *mut c_void,
    access: FjCudaCallAccess,
) -> FjCallCompletionResult {
    let fallback = FjCallCompletionResult {
        outcome: sys::FjRenderOutcome {
            status: status(sys::FJ_STATUS_INTERNAL_FAILURE),
            flags: 0,
        },
        fatal: 1,
    };
    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: fj_cuda_call invokes this once serially on this thread. The
        // shared Call retains immutable inputs; nested completion mutates only
        // its UnsafeCell field, which run never reads or borrows.
        let call = unsafe { &*user.cast::<Call<'_, F>>() };
        let Ok(mut owner) = OwnerBorrow::admit_access(access) else {
            return fallback;
        };
        // SAFETY: The caller's image leases/backing retain this immutable frame
        // and abort context. Native inspects addresses; Rust never reads pixels.
        let Ok(frame) = (unsafe { frame(call.frame, call.frame_token) }) else {
            return fallback;
        };
        let abort = abort(call.abort);
        let result = crate::cuda::render(
            &mut owner,
            call.runtime.assets(),
            call.state,
            &frame,
            call.policy,
            call.instance,
            &abort,
        );
        match result {
            Ok(outcome) => outcome.completion,
            Err(error) => error.completion,
        }
    }))
    .unwrap_or(fallback)
}
/// # Safety
/// The matching native CallScope invokes this once, serially on the originating
/// thread after releasing its render gate. `user` retains the same live `Call<F>`;
/// `scope` and initialized `input`, including readable diagnostic backing, live
/// through return. Only this callback mutates the disjoint UnsafeCell field; the
/// higher-ranked completion cannot retain or transfer the native capability.
unsafe extern "C" fn complete<F: for<'call> FnMut(Completion<'call>) -> bool>(
    user: *mut c_void,
    scope: *mut c_void,
    input: *const FjCallCompletionInput,
) -> FjCudaCompletion {
    let fallback = FjCallCompletionResult {
        outcome: sys::FjRenderOutcome {
            status: status(sys::FJ_STATUS_INTERNAL_FAILURE),
            flags: 0,
        },
        fatal: 1,
    };
    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Native completion provides initialized immutable input and a
        // live stack scope after release. It runs once serially; only complete's
        // UnsafeCell field is accessed mutably, disjoint from run's shared inputs.
        let call = unsafe { &*user.cast::<Call<'_, F>>() };
        // SAFETY: Native completion supplies this initialized immutable record through return.
        let input = unsafe { &*input };
        let diagnostic = if input.diagnostic.count == 0 {
            ""
        } else {
            // SAFETY: CallScope supplies this length-delimited borrowed native
            // diagnostic through this completion only; no host call retains it.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    input.diagnostic.data.cast::<u8>(),
                    input.diagnostic.count,
                )
            };
            std::str::from_utf8(bytes).unwrap_or("native render diagnostic is not UTF-8")
        };
        let recovery = std::cell::Cell::new((0, status(sys::FJ_STATUS_SUCCESS), 0));
        let completion = Completion {
            input,
            scope,
            diagnostic,
            recovery: &recovery,
            _thread: PhantomData,
        };
        // SAFETY: Exactly one synchronous completion may mutate this callback.
        let fatal = unsafe { &mut *call.complete.get() }(completion);
        let (retirement, invalidation, latch_cleared) = recovery.get();
        FjCudaCompletion {
            result: FjCallCompletionResult {
                outcome: input.outcome,
                fatal: u32::from(fatal),
            },
            retirement,
            invalidation,
            latch_cleared,
        }
    }))
    .unwrap_or(FjCudaCompletion {
        result: fallback,
        retirement: 0,
        invalidation: status(sys::FJ_STATUS_INTERNAL_FAILURE),
        latch_cleared: 0,
    })
}

/// # Safety
/// Frame addresses originate in live callback-local OFX image leases. Runtime,
/// complete state, frame, policy, Instance and abort user stay alive unchanged on
/// this thread until return. Abort callbacks contain panics; native never retains
/// borrowed storage and copies asynchronous inputs before returning.
#[expect(
    clippy::too_many_arguments,
    reason = "immediate projection borrows the existing complete owners and callback-local facts"
)]
pub(crate) unsafe fn render_callback<F: for<'call> FnMut(Completion<'call>) -> bool>(
    runtime: &Runtime,
    state: &RenderState,
    frame: &sys::FjFrame,
    policy: &FramePolicy,
    instance: &Instance,
    frame_token: u64,
    abort: sys::FjAbortCallback,
    completion: F,
) -> bool {
    let mut call = Call {
        runtime,
        state,
        frame,
        policy,
        instance,
        frame_token,
        abort,
        complete: std::cell::UnsafeCell::new(completion),
    };
    let route = match state.development().route() {
        film_juicer_core::route::ScanRoute::NegativeDirect => sys::FJ_ROUTE_NEGATIVE_DIRECT,
        film_juicer_core::route::ScanRoute::NegativePrint => sys::FJ_ROUTE_NEGATIVE_PRINT,
        film_juicer_core::route::ScanRoute::PositiveDirect => sys::FJ_ROUTE_POSITIVE_DIRECT,
        film_juicer_core::route::ScanRoute::PositivePrint => sys::FJ_ROUTE_POSITIVE_PRINT,
    };
    // SAFETY: The retained Runtime excludes consuming close; stack Call and all
    // borrowed owners survive this synchronous entry and nested completion. The
    // callback/continuation types match the C header and catch all Rust panics.
    let result = unsafe {
        fj_cuda_call(
            runtime.native().raw(),
            route,
            run::<F>,
            complete::<F>,
            std::ptr::from_mut(&mut call).cast(),
        )
    };
    result.fatal != 0
}
