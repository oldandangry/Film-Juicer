//! Bounded fixture observations at the actual module boundary.

mod control_read;

use std::cell::Cell;
use std::sync::{Condvar, Mutex};
use std::time::Duration;

use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

thread_local! { static INSTANCE_FAULT: Cell<u32> = const { Cell::new(0) }; }
thread_local! { static CLOCK_TOKEN: Cell<Option<u64>> = const { Cell::new(None) }; }
pub(in crate::ofx) fn clock_token() -> Option<u64> {
    CLOCK_TOKEN.take()
}

#[cfg_attr(not(test), unsafe(no_mangle))]
extern "C" fn fj_test_ofx_clock_token(clock: u64) {
    CLOCK_TOKEN.set(Some(clock));
}

#[cfg_attr(not(test), unsafe(no_mangle))]
extern "C" fn fj_test_ofx_seed(clock: u64, address: u64) -> u64 {
    super::super::instance::seed(clock, address)
}

/// # Safety
/// Effect is live under the OFX callback contract. The three outputs are
/// nonnull, aligned, disjoint writable objects of their declared types.
#[cfg_attr(not(test), unsafe(no_mangle))]
unsafe extern "C" fn fj_test_ofx_instance_identity(
    effect: sys::OfxImageEffectHandle,
    seed: *mut u64,
    token: *mut u64,
    address: *mut usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        let admission = super::super::admit_action(true).ok()?;
        let callback = Callback::new(&admission, effect, std::ptr::null_mut());
        let owner = callback.instance().ok()?;
        let identity = owner.state.identity()?;
        // SAFETY: The fixture supplies three disjoint, live output objects of
        // these exact widths, and a live effect under the callback contract.
        unsafe {
            seed.write(identity.session_seed);
            token.write(identity.instance_token);
            address.write(std::ptr::from_ref(owner) as usize);
        }
        Some(1)
    }))
    .ok()
    .flatten()
    .unwrap_or(0)
}
static INSTANCES_CREATED: AtomicU64 = AtomicU64::new(0);
static INSTANCES_RECLAIMED: AtomicU64 = AtomicU64::new(0);
static INSTANCES_RETIRED: AtomicU64 = AtomicU64::new(0);

pub(in crate::ofx) fn instance_fault(kind: u32) -> bool {
    if INSTANCE_FAULT.get() == kind {
        INSTANCE_FAULT.set(0);
        true
    } else {
        false
    }
}
pub(super) fn instance_created() {
    INSTANCES_CREATED.fetch_add(1, Ordering::Relaxed);
}
pub(super) fn instance_reclaimed() {
    INSTANCES_RECLAIMED.fetch_add(1, Ordering::Relaxed);
}
pub(super) fn instance_retired() {
    INSTANCES_RETIRED.fetch_add(1, Ordering::Relaxed);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FjOfxRetirementObservation {
    count: u32,
    identity_live: u32,
    token: u64,
    status: crate::cuda::sys::FjStatus,
}
const EMPTY_RETIREMENT: FjOfxRetirementObservation = FjOfxRetirementObservation {
    count: 0,
    identity_live: 0,
    token: 0,
    status: crate::cuda::sys::FjStatus {
        category: 0,
        api: 0,
        native_code: 0,
    },
};
thread_local! {
    static RETIREMENT: Cell<FjOfxRetirementObservation> = const { Cell::new(EMPTY_RETIREMENT) };
}
pub(in crate::ofx) fn record_retirement(
    token: u64,
    identity_live: bool,
    status: crate::cuda::sys::FjStatus,
) {
    let count = RETIREMENT.get().count.wrapping_add(1);
    RETIREMENT.set(FjOfxRetirementObservation {
        count,
        identity_live: u32::from(identity_live),
        token,
        status,
    });
}
#[cfg_attr(not(test), unsafe(no_mangle))]
extern "C" fn fj_test_ofx_retirement_reset() {
    RETIREMENT.set(EMPTY_RETIREMENT);
}

/// # Safety
/// Out authorizes one live, aligned, exclusive observation record through return.
#[cfg_attr(not(test), unsafe(no_mangle))]
unsafe extern "C" fn fj_test_ofx_retirement_observe(out: *mut FjOfxRetirementObservation) -> i32 {
    if out.is_null()
        || !(out as usize).is_multiple_of(std::mem::align_of::<FjOfxRetirementObservation>())
        || (out as usize)
            .checked_add(std::mem::size_of::<FjOfxRetirementObservation>())
            .is_none()
    {
        return 0;
    }
    // SAFETY: Numeric admission supplements the caller's live exclusive extent;
    // this pointer-free copy retains neither owner nor output and calls no host.
    unsafe { out.write(RETIREMENT.get()) };
    1
}
const _: extern "C" fn() = fj_test_ofx_retirement_reset;
const _: unsafe extern "C" fn(*mut FjOfxRetirementObservation) -> i32 =
    fj_test_ofx_retirement_observe;

#[cfg_attr(not(test), unsafe(no_mangle))]
extern "C" fn fj_test_ofx_instance_control(kind: u32) {
    INSTANCE_FAULT.set(kind);
}

/// # Safety
/// The three outputs are nonnull, aligned, disjoint writable u64 objects.
#[cfg_attr(not(test), unsafe(no_mangle))]
unsafe extern "C" fn fj_test_ofx_instance_counts(
    created: *mut u64,
    reclaimed: *mut u64,
    retired: *mut u64,
) {
    // SAFETY: The native fixture supplies three live disjoint u64 output slots.
    unsafe {
        created.write(INSTANCES_CREATED.load(Ordering::Relaxed));
        reclaimed.write(INSTANCES_RECLAIMED.load(Ordering::Relaxed));
        retired.write(INSTANCES_RETIRED.load(Ordering::Relaxed));
    }
}

/// # Safety
/// Effect is null or live under the OFX callback contract. Output and error
/// storage obey the existing observation ABI for this synchronous call.
#[cfg_attr(not(test), unsafe(no_mangle))]
unsafe extern "C" fn fj_test_ofx_instance_observe(
    effect: sys::OfxImageEffectHandle,
    out: *mut *mut crate::instance_bridge::FjInstanceObservation,
    error: *mut crate::cuda::sys::FjErrorBuffer,
) -> crate::cuda::sys::FjStatus {
    observe_or_admit(effect, true, |owner| {
        // SAFETY: Fixture supplies ordinary existing observation output/error storage.
        unsafe { crate::instance_bridge::observe_safe(&owner.state, out, error) }
    })
}

/// # Safety
/// Effect is null or live under the OFX callback contract. Output and error
/// storage obey the existing prepared-state ABI for this synchronous call.
#[cfg_attr(not(test), unsafe(no_mangle))]
unsafe extern "C" fn fj_test_ofx_instance_admit(
    effect: sys::OfxImageEffectHandle,
    out: *mut *mut crate::preparation::FjPreparedState,
    error: *mut crate::cuda::sys::FjErrorBuffer,
) -> crate::cuda::sys::FjStatus {
    observe_or_admit(effect, false, |owner| {
        // SAFETY: Fixture supplies ordinary existing prepared output/error storage.
        unsafe {
            crate::instance_bridge::admit_safe(
                &owner.state,
                owner.loaded.runtime().assets(),
                out,
                error,
            )
        }
    })
}

/// # Safety
/// Effect is live through the guarded call. Initialized output/error extents are
/// aligned, exclusive and disjoint; success transfers one weak observation only.
#[cfg_attr(not(test), unsafe(no_mangle))]
unsafe extern "C" fn fj_test_ofx_lifetime_acquire(
    effect: sys::OfxImageEffectHandle,
    out: *mut *mut crate::instance_bridge::FjOfxLifetime,
    error: *mut crate::cuda::sys::FjErrorBuffer,
) -> crate::cuda::sys::FjStatus {
    observe_or_admit(effect, true, |owner| {
        // SAFETY: The action guard retains the actual OFX owner while its weak
        // signal is copied; caller extents obey the common facade contract.
        unsafe {
            crate::instance_bridge::ofx_lifetime_create(
                std::sync::Arc::downgrade(&owner.fixture_lifetime),
                out,
                error,
            )
        }
    })
}
const _: unsafe extern "C" fn(
    sys::OfxImageEffectHandle,
    *mut *mut crate::instance_bridge::FjOfxLifetime,
    *mut crate::cuda::sys::FjErrorBuffer,
) -> crate::cuda::sys::FjStatus = fj_test_ofx_lifetime_acquire;

/// # Safety
/// Effect is a live instance under the OFX callback exclusion contract.
#[cfg_attr(not(test), unsafe(no_mangle))]
unsafe extern "C" fn fj_test_ofx_host_call_active(effect: sys::OfxImageEffectHandle) -> i32 {
    let mut observed = None;
    let _ = observe_or_admit(effect, true, |owner| {
        observed = Some(owner.host_call_active());
        crate::cuda::sys::FjStatus {
            category: 0,
            api: 0,
            native_code: 0,
        }
    });
    observed.map_or(-1, i32::from)
}
const _: unsafe extern "C" fn(sys::OfxImageEffectHandle) -> i32 = fj_test_ofx_host_call_active;

/// # Safety
/// Effect is live until this synchronous call returns. Output and diagnostic
/// extents are initialized, aligned, exclusive and disjoint. A Ready output
/// transfers one complete state hold; it retains no OFX or host pointer.
#[cfg_attr(not(test), unsafe(no_mangle))]
unsafe extern "C" fn fj_test_ofx_instance_admission(
    effect: sys::OfxImageEffectHandle,
    out: *mut crate::instance_bridge::FjAdmission,
    error: *mut crate::cuda::sys::FjErrorBuffer,
) -> crate::cuda::sys::FjStatus {
    observe_or_admit(effect, false, |owner| {
        // SAFETY: Action admission excludes owner destruction; fixture storage
        // obeys the same checked admission contract as the raw Instance facade.
        unsafe {
            crate::instance_bridge::admit_safe_full(
                &owner.state,
                owner.loaded.runtime().assets(),
                out,
                error,
            )
        }
    })
}

/// # Safety
/// Effect remains live; all outputs and diagnostics are aligned, initialized,
/// exclusive and mutually disjoint through return. No output owns the Instance.
#[cfg_attr(not(test), unsafe(no_mangle))]
unsafe extern "C" fn fj_test_ofx_instance_submission(
    effect: sys::OfxImageEffectHandle,
    valid: *mut u32,
    out: *mut crate::cuda::sys::FjSubmission,
    context: *mut crate::cuda::sys::FjCudaContext,
    error: *mut crate::cuda::sys::FjErrorBuffer,
) -> crate::cuda::sys::FjStatus {
    observe_or_admit(effect, true, |owner| {
        // SAFETY: The guarded owner and disjoint outputs remain live through the
        // checked projection; observing its latch does not select a submission.
        unsafe { crate::instance_bridge::submission_safe(&owner.state, valid, out, context, error) }
    })
}

/// # Safety
/// Effect remains live; output and diagnostics are initialized, aligned,
/// exclusive and disjoint until synchronous return. No pointer is retained.
#[cfg_attr(not(test), unsafe(no_mangle))]
unsafe extern "C" fn fj_test_ofx_capture_is_current(
    effect: sys::OfxImageEffectHandle,
    capture: u64,
    out: *mut u32,
    error: *mut crate::cuda::sys::FjErrorBuffer,
) -> crate::cuda::sys::FjStatus {
    observe_or_admit(effect, true, |owner| {
        // SAFETY: Guarded immutable Instance and exclusive caller extents obey
        // the immediate checked observation's borrow/non-retention contract.
        unsafe { crate::instance_bridge::capture_current_safe(&owner.state, capture, out, error) }
    })
}

const _: unsafe extern "C" fn(
    sys::OfxImageEffectHandle,
    *mut crate::instance_bridge::FjAdmission,
    *mut crate::cuda::sys::FjErrorBuffer,
) -> crate::cuda::sys::FjStatus = fj_test_ofx_instance_admission;
const _: unsafe extern "C" fn(
    sys::OfxImageEffectHandle,
    *mut u32,
    *mut crate::cuda::sys::FjSubmission,
    *mut crate::cuda::sys::FjCudaContext,
    *mut crate::cuda::sys::FjErrorBuffer,
) -> crate::cuda::sys::FjStatus = fj_test_ofx_instance_submission;
const _: unsafe extern "C" fn(
    sys::OfxImageEffectHandle,
    u64,
    *mut u32,
    *mut crate::cuda::sys::FjErrorBuffer,
) -> crate::cuda::sys::FjStatus = fj_test_ofx_capture_is_current;

/// # Safety
/// Live effect/module, held state and all raw records remain live through the
/// synchronous call and completion. Output/error backing is exclusive/disjoint.
/// Callbacks run serially on this thread, contain exceptions and retain nothing;
/// native staging owns every queued read before return. The effect's real safe
/// Instance supplies identity/submission; no raw Instance capability escapes.
#[cfg_attr(not(test), unsafe(no_mangle))]
unsafe extern "C" fn fj_test_ofx_prepared_state_render(
    effect: sys::OfxImageEffectHandle,
    state: *const crate::preparation::FjPreparedState,
    access: crate::cuda::edge::FjCudaCallAccess,
    frame: *const crate::cuda::sys::FjFrame,
    policy: *const crate::cuda::edge::FjPreparedFramePolicy,
    frame_token: u64,
    abort: crate::cuda::sys::FjAbortCallback,
    error: *mut crate::cuda::sys::FjErrorBuffer,
) -> crate::cuda::edge::FjCallCompletionResult {
    let mut result = None;
    let status = observe_or_admit(effect, false, |owner| {
        // SAFETY: The action admission retains the actual OFX owner and Runtime;
        // the fixture admits every synchronous raw peer/callback under the same
        // held-render contract. Only immutable backing crosses native callbacks.
        let completed = unsafe {
            crate::cuda::edge::prepared_render_for_instance(
                std::ptr::from_ref(owner.loaded.runtime().assets()).cast(),
                state,
                access,
                crate::cuda::edge::PreparedFrameInput {
                    frame,
                    policy,
                    token: frame_token,
                },
                &owner.state,
                abort,
                error,
            )
        };
        let status = completed.outcome.status;
        result = Some(completed);
        status
    });
    result.unwrap_or(crate::cuda::edge::FjCallCompletionResult {
        outcome: crate::cuda::sys::FjRenderOutcome { status, flags: 0 },
        fatal: 1,
    })
}
const _: unsafe extern "C" fn(
    sys::OfxImageEffectHandle,
    *const crate::preparation::FjPreparedState,
    crate::cuda::edge::FjCudaCallAccess,
    *const crate::cuda::sys::FjFrame,
    *const crate::cuda::edge::FjPreparedFramePolicy,
    u64,
    crate::cuda::sys::FjAbortCallback,
    *mut crate::cuda::sys::FjErrorBuffer,
) -> crate::cuda::edge::FjCallCompletionResult = fj_test_ofx_prepared_state_render;

fn observe_or_admit(
    effect: sys::OfxImageEffectHandle,
    terminal: bool,
    operation: impl FnOnce(&super::super::instance::Owner) -> crate::cuda::sys::FjStatus,
) -> crate::cuda::sys::FjStatus {
    catch_unwind(AssertUnwindSafe(|| {
        if effect.is_null() {
            return None;
        }
        let admission = super::super::admit_action(terminal).ok()?;
        let callback = Callback::new(&admission, effect, std::ptr::null_mut());
        Some(operation(callback.instance().ok()?))
    }))
    .ok()
    .flatten()
    .unwrap_or(crate::cuda::sys::FjStatus {
        category: crate::cuda::sys::FJ_STATUS_INTERNAL_FAILURE,
        api: 0,
        native_code: 0,
    })
}

#[cfg_attr(not(test), unsafe(no_mangle))]
extern "C" fn fj_test_ofx_clock_conversion(counter: i64, frequency: i64) -> u64 {
    crate::platform::qpc_nanoseconds(counter, frequency).unwrap_or(0)
}

thread_local! {
    static PANIC: Cell<u32> = const { Cell::new(0) };
    static PAUSES: Cell<u32> = const { Cell::new(0) };
    static CONTEXT_FAULT: Cell<u32> = const { Cell::new(0) };
}

pub(super) fn context_fault(mode: u32) -> bool {
    if CONTEXT_FAULT.get() == mode {
        CONTEXT_FAULT.set(0);
        true
    } else {
        false
    }
}

#[cfg_attr(not(test), unsafe(no_mangle))]
extern "C" fn fj_test_ofx_context_fault(mode: u32) {
    CONTEXT_FAULT.set(mode);
}

struct Pause {
    stage: u32,
    released: bool,
    timed_out: bool,
}

static PAUSE: Mutex<Pause> = Mutex::new(Pause {
    stage: 0,
    released: false,
    timed_out: false,
});
static CHANGED: Condvar = Condvar::new();

// One worker/fixture owns this bounded gate. The TLS mask prevents overlap
// probes on the controlling thread from pausing at the worker's observations.
pub(in crate::ofx) fn pause(stage: u32) {
    if PAUSES.get() & stage == 0 {
        return;
    }
    PAUSES.set(PAUSES.get() & !stage);
    let mut gate = PAUSE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    gate.stage = stage;
    gate.released = false;
    CHANGED.notify_all();
    let (mut gate, timeout) = CHANGED
        .wait_timeout_while(gate, Duration::from_secs(10), |gate| !gate.released)
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if timeout.timed_out() && !gate.released {
        gate.timed_out = true;
    }
    gate.stage = 0;
    CHANGED.notify_all();
}

#[cfg_attr(not(test), unsafe(no_mangle))]
extern "C" fn fj_test_ofx_arm_pause(stages: u32) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        *PAUSE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Pause {
            stage: 0,
            released: false,
            timed_out: false,
        };
        PAUSES.set(stages);
    }));
}

#[cfg_attr(not(test), unsafe(no_mangle))]
extern "C" fn fj_test_ofx_wait_pause(stage: u32) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        let gate = PAUSE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (gate, _) = CHANGED
            .wait_timeout_while(gate, Duration::from_secs(10), |gate| {
                gate.stage != stage && !gate.timed_out
            })
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        i32::from(gate.stage == stage && !gate.timed_out)
    }))
    .unwrap_or(0)
}

#[cfg_attr(not(test), unsafe(no_mangle))]
extern "C" fn fj_test_ofx_release_pause() -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        let mut gate = PAUSE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        gate.released = true;
        CHANGED.notify_all();
        i32::from(!gate.timed_out)
    }))
    .unwrap_or(0)
}

pub(super) fn checkpoint(stage: u32) {
    if PANIC.get() == stage {
        PANIC.set(0);
        panic!("OFX callback checkpoint panic");
    }
}

#[cfg_attr(not(test), unsafe(no_mangle))]
extern "C" fn fj_test_ofx_panic(stage: u32) {
    PANIC.set(stage);
}

#[cfg_attr(not(test), unsafe(no_mangle))]
extern "C" fn fj_test_ofx_native() -> *mut crate::cuda::sys::FjCuda {
    catch_unwind(AssertUnwindSafe(|| {
        let state = super::super::state();
        match &*state {
            super::super::Module::Loaded(loaded) | super::super::Module::InvalidLoaded(loaded) => {
                loaded.runtime.native().raw()
            }
            _ => std::ptr::null_mut(),
        }
    }))
    .unwrap_or(std::ptr::null_mut())
}

#[cfg_attr(not(test), unsafe(no_mangle))]
extern "C" fn fj_test_ofx_assets() -> *const c_void {
    catch_unwind(AssertUnwindSafe(|| {
        let state = super::super::state();
        match &*state {
            super::super::Module::Loaded(loaded) | super::super::Module::InvalidLoaded(loaded) => {
                std::ptr::from_ref(loaded.runtime.assets()).cast()
            }
            _ => std::ptr::null(),
        }
    }))
    .unwrap_or(std::ptr::null())
}

#[cfg_attr(not(test), unsafe(no_mangle))]
extern "C" fn fj_test_ofx_runtime_shutdown() -> crate::cuda::sys::FjStatus {
    catch_unwind(AssertUnwindSafe(|| {
        let Ok(admission) = super::super::admit_action(true) else {
            return crate::cuda::sys::FjStatus {
                category: crate::cuda::sys::FJ_STATUS_UNSUPPORTED_INPUT,
                api: 0,
                native_code: 0,
            };
        };
        match admission.runtime().shutdown() {
            Ok(()) => crate::cuda::sys::FjStatus {
                category: 0,
                api: 0,
                native_code: 0,
            },
            Err(failure) => failure.status,
        }
    }))
    .unwrap_or(crate::cuda::sys::FjStatus {
        category: crate::cuda::sys::FJ_STATUS_INTERNAL_FAILURE,
        api: 0,
        native_code: 0,
    })
}
const _: extern "C" fn() -> crate::cuda::sys::FjStatus = fj_test_ofx_runtime_shutdown;

/// # Safety
/// Facts is an initialized immutable native array of count size_t values.
#[cfg_attr(not(test), unsafe(no_mangle))]
unsafe extern "C" fn fj_test_ofx_platform_abi(facts: *const usize, count: usize) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if facts.is_null() || count != crate::platform::abi().len() {
            return 0;
        }
        // SAFETY: Native C11 fixture supplies exactly this live immutable array.
        i32::from(unsafe { std::slice::from_raw_parts(facts, count) } == crate::platform::abi())
    }))
    .unwrap_or(0)
}

/// # Safety
/// Input obeys FjPathView's native representation; output has capacity writable
/// native path units and is disjoint. Zero capacity permits a null output.
#[cfg_attr(not(test), unsafe(no_mangle))]
unsafe extern "C" fn fj_test_ofx_path(
    mode: u32,
    input: crate::cuda::sys::FjPathView,
    output: *mut c_void,
    capacity: usize,
) -> usize {
    catch_unwind(AssertUnwindSafe(|| {
        let path = if mode == 1 {
            // SAFETY: Fixture supplies the established native path view contract.
            let Ok(path) = (unsafe { crate::asset_bridge::path(input) }) else {
                return usize::MAX;
            };
            path
        } else {
            std::path::PathBuf::new()
        };
        let path = crate::platform::characterize(mode, &path);
        #[cfg(unix)]
        let units = {
            use std::os::unix::ffi::OsStrExt;
            path.as_os_str().as_bytes()
        };
        #[cfg(windows)]
        let storage = {
            use std::os::windows::ffi::OsStrExt;
            path.as_os_str().encode_wide().collect::<Vec<_>>()
        };
        #[cfg(windows)]
        let units = storage.as_slice();
        if units.len() <= capacity && !units.is_empty() {
            if output.is_null() {
                return usize::MAX;
            }
            // SAFETY: Caller authorizes capacity native units; lengths checked
            // before copying, and the temporary source has disjoint storage.
            unsafe {
                std::ptr::copy_nonoverlapping(units.as_ptr(), output.cast(), units.len());
            }
        }
        units.len()
    }))
    .unwrap_or(usize::MAX)
}

const _: extern "C" fn(u64) = fj_test_ofx_clock_token;
const _: extern "C" fn(u64, u64) -> u64 = fj_test_ofx_seed;
const _: unsafe extern "C" fn(sys::OfxImageEffectHandle, *mut u64, *mut u64, *mut usize) -> i32 =
    fj_test_ofx_instance_identity;
const _: extern "C" fn(u32) = fj_test_ofx_instance_control;
const _: unsafe extern "C" fn(*mut u64, *mut u64, *mut u64) = fj_test_ofx_instance_counts;
const _: unsafe extern "C" fn(
    sys::OfxImageEffectHandle,
    *mut *mut crate::instance_bridge::FjInstanceObservation,
    *mut crate::cuda::sys::FjErrorBuffer,
) -> crate::cuda::sys::FjStatus = fj_test_ofx_instance_observe;
const _: unsafe extern "C" fn(
    sys::OfxImageEffectHandle,
    *mut *mut crate::preparation::FjPreparedState,
    *mut crate::cuda::sys::FjErrorBuffer,
) -> crate::cuda::sys::FjStatus = fj_test_ofx_instance_admit;
const _: extern "C" fn(i64, i64) -> u64 = fj_test_ofx_clock_conversion;
const _: extern "C" fn(u32) = fj_test_ofx_panic;
const _: extern "C" fn(u32) = fj_test_ofx_context_fault;
const _: extern "C" fn(u32) = fj_test_ofx_arm_pause;
const _: extern "C" fn(u32) -> i32 = fj_test_ofx_wait_pause;
const _: extern "C" fn() -> i32 = fj_test_ofx_release_pause;
const _: extern "C" fn() -> *mut crate::cuda::sys::FjCuda = fj_test_ofx_native;
const _: extern "C" fn() -> *const c_void = fj_test_ofx_assets;
const _: unsafe extern "C" fn(*const usize, usize) -> i32 = fj_test_ofx_platform_abi;
const _: unsafe extern "C" fn(u32, crate::cuda::sys::FjPathView, *mut c_void, usize) -> usize =
    fj_test_ofx_path;

// Render controls are thread-local and one-shot at the named actual consumer.
thread_local! {
    static RENDER_FAULT: Cell<u32> = const { Cell::new(0) };
    static RENDER_TOKENS: Cell<Option<(u64, u64, u64)>> = const { Cell::new(None) };
    static RENDER_OBSERVATION: Cell<RenderObservation> = const { Cell::new(RenderObservation::EMPTY) };
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RenderObservation {
    frame_time: f64,
    frame_rate: f64,
    frame_token: u64,
    session_seed: u64,
    instance_token: u64,
    clip_token: u64,
    native_entry_count: u64,
    submission_id: u64,
    input_identity: u64,
    build_number: u64,
    backing_identity: usize,
    full_bounds: [i32; 4],
    render_bounds: [i32; 4],
    pixel_size_um: f32,
    status: u32,
    fatal: u32,
}
impl RenderObservation {
    const EMPTY: Self = Self {
        frame_time: 0.0,
        frame_rate: 0.0,
        frame_token: 0,
        session_seed: 0,
        instance_token: 0,
        clip_token: 0,
        native_entry_count: 0,
        submission_id: 0,
        input_identity: 0,
        build_number: 0,
        backing_identity: 0,
        full_bounds: [0; 4],
        render_bounds: [0; 4],
        pixel_size_um: 0.0,
        status: 0,
        fatal: 0,
    };
}
pub(in crate::ofx) fn render_fault(kind: u32) -> bool {
    if RENDER_FAULT.get() == kind {
        RENDER_FAULT.set(0);
        true
    } else {
        false
    }
}
pub(in crate::ofx) fn render_tokens() -> Option<(u64, u64, u64)> {
    RENDER_TOKENS.get()
}
pub(in crate::ofx) fn begin_render() {
    RENDER_OBSERVATION.set(RenderObservation::EMPTY);
}
pub(in crate::ofx) fn render_state(state: &film_juicer_core::state::Admission) {
    RENDER_OBSERVATION.with(|cell| {
        let mut value = cell.get();
        value.input_identity = state.input_identity();
        value.build_number = state.build();
        value.backing_identity = std::sync::Arc::as_ptr(state.shared_state()) as usize;
        cell.set(value);
    });
}
pub(in crate::ofx) fn render_frame(
    frame: &crate::cuda::sys::FjFrame,
    token: u64,
    instance: &crate::instance::Instance,
) {
    RENDER_OBSERVATION.with(|cell| {
        let mut value = cell.get();
        value.frame_time = frame.time_frames;
        value.frame_rate = frame.frame_rate;
        value.frame_token = token;
        value.session_seed = frame.session_seed;
        value.instance_token = instance
            .identity()
            .map_or(0, |identity| identity.instance_token);
        value.clip_token = frame.clip_token;
        value.native_entry_count += 1;
        value.full_bounds = [
            frame.full_frame_extent.x1,
            frame.full_frame_extent.y1,
            frame.full_frame_extent.x2,
            frame.full_frame_extent.y2,
        ];
        value.render_bounds = [
            frame.render_window.x1,
            frame.render_window.y1,
            frame.render_window.x2,
            frame.render_window.y2,
        ];
        value.pixel_size_um = frame.pixel_size_um;
        cell.set(value);
    });
}
pub(in crate::ofx) fn render_complete(
    status: u32,
    fatal: bool,
    instance: &crate::instance::Instance,
) {
    RENDER_OBSERVATION.with(|cell| {
        let mut value = cell.get();
        value.status = status;
        value.fatal = u32::from(fatal);
        value.submission_id = instance
            .observe_submission()
            .ok()
            .flatten()
            .map_or(0, |selection| selection.id);
        cell.set(value);
    });
}
#[cfg_attr(not(test), unsafe(no_mangle))]
extern "C" fn fj_test_ofx_render_control(kind: u32) {
    RENDER_FAULT.set(kind);
}
#[cfg_attr(not(test), unsafe(no_mangle))]
extern "C" fn fj_test_ofx_render_tokens(session: u64, instance: u64, clip: u64) {
    RENDER_TOKENS.set(if session == 0 && instance == 0 && clip == 0 {
        None
    } else {
        Some((session, instance, clip))
    });
}
/// # Safety
/// Effect is live; output is a nonnull aligned writable observation. This read
/// is synchronous on the same callback thread and retains no output pointer.
#[cfg_attr(not(test), unsafe(no_mangle))]
unsafe extern "C" fn fj_test_ofx_render_observe(
    _effect: sys::OfxImageEffectHandle,
    output: *mut RenderObservation,
) -> i32 {
    if output.is_null() {
        return 0;
    }
    // SAFETY: The fixture supplies the stated live output slot.
    unsafe {
        output.write(RENDER_OBSERVATION.get());
    }
    1
}

const _: extern "C" fn(u32) = fj_test_ofx_render_control;
const _: extern "C" fn(u64, u64, u64) = fj_test_ofx_render_tokens;
const _: unsafe extern "C" fn(sys::OfxImageEffectHandle, *mut RenderObservation) -> i32 =
    fj_test_ofx_render_observe;
