//! Bounded fixture observations at the actual module boundary.

use std::cell::Cell;
use std::sync::{Condvar, Mutex};
use std::time::Duration;

use super::*;

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

/// # Safety
/// Facts is an initialized immutable native array of six size_t values.
#[cfg_attr(not(test), unsafe(no_mangle))]
unsafe extern "C" fn fj_test_ofx_platform_abi(facts: *const usize, count: usize) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if facts.is_null() || count != 6 {
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
