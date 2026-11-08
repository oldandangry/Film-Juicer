//! Candidate module lifecycle; shipping discovery still selects the C++ adapter.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex};

use crate::runtime::Runtime;

mod actions;
mod default_actions;
mod render;
mod params;
mod controls;
mod instance;
#[allow(
    unsafe_code,
    reason = "the OFX edge admits binary-lived host tables and creates callback-scoped capabilities"
)]
pub(crate) mod host;

use host::{Admission, Binding, Status, Suites};

struct Loaded {
    binding: Binding,
    suites: Suites,
    runtime: Arc<Runtime>,
}

enum Module {
    Unbound,
    Bound(Binding),
    Invalid(Option<Binding>),
    Loading(Binding),
    InvalidLoading(Binding),
    Loaded(Loaded),
    InvalidLoaded(Loaded),
    Closing(Binding),
    InvalidClosing(Binding),
}

static MODULE: Mutex<Module> = Mutex::new(Module::Unbound);

fn state() -> std::sync::MutexGuard<'static, Module> {
    // No foreign call or fallible construction holds this lock. A boundary
    // panic marks the recovered state invalid before another ordinary action.
    MODULE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn invalidated(previous: Module) -> Module {
    match previous {
        Module::Unbound => Module::Invalid(None),
        Module::Bound(binding) | Module::Invalid(Some(binding)) => Module::Invalid(Some(binding)),
        Module::Invalid(None) => Module::Invalid(None),
        Module::Loading(binding) | Module::InvalidLoading(binding) => {
            Module::InvalidLoading(binding)
        }
        Module::Loaded(loaded) | Module::InvalidLoaded(loaded) => Module::InvalidLoaded(loaded),
        Module::Closing(binding) | Module::InvalidClosing(binding) => {
            Module::InvalidClosing(binding)
        }
    }
}

fn bind(binding: impl FnOnce() -> Option<Binding>) {
    let mut state = state();
    // Reading the host record invokes no foreign code. Contain a void failure
    // and mark only this binding mutation before releasing the module lock.
    match catch_unwind(AssertUnwindSafe(binding)) {
        Ok(binding)
            if matches!(
                *state,
                Module::Unbound | Module::Bound(_) | Module::Invalid(_)
            ) =>
        {
            *state = binding.map_or(Module::Invalid(None), Module::Bound);
        }
        _ => *state = invalidated(std::mem::replace(&mut *state, Module::Unbound)),
    }
}

fn load() -> Result<(), Status> {
    let binding = {
        let mut state = state();
        let binding = match &*state {
            Module::Bound(binding) | Module::Invalid(Some(binding)) => *binding,
            Module::Unbound | Module::Invalid(None) => return Err(Status::BAD_HANDLE),
            _ => return Err(Status::FAILED),
        };
        *state = Module::Loading(binding);
        binding
    };
    // The reservation remains exclusive through construction unwind AND complete
    // rollback. No outer panic mapper may mutate state after it is released.
    let result = catch_unwind(AssertUnwindSafe(|| {
        let suites = binding.suites()?;
        let directory = crate::platform::resource_directory();
        if directory.as_os_str().is_empty() {
            return Err(Status::FAILED);
        }
        let runtime = Runtime::new(directory).map_err(Status::runtime)?;
        let loaded = Loaded {
            binding,
            suites,
            runtime: Arc::new(runtime),
        };
        #[cfg(feature = "test-support")]
        host::test_support::pause(4);
        let mut state = state();
        if matches!(*state, Module::Loading(_)) {
            *state = Module::Loaded(loaded);
            return Ok(());
        }
        drop(state);
        #[cfg(feature = "test-support")]
        host::test_support::pause(8);
        drop(loaded);
        Err(Status::FAILED)
    }));
    match result {
        Ok(Ok(())) => Ok(()),
        result => {
            // Local owners have finished unwinding/dropping. Only this action
            // can release Loading/InvalidLoading, so no later lifecycle exists.
            *state() = Module::Invalid(Some(binding));
            match result {
                Ok(Err(error)) => Err(error),
                _ => Err(Status::FAILED),
            }
        }
    }
}

fn fail_admission(admission: &Admission) {
    let mut state = state();
    if matches!(&*state, Module::Loaded(loaded) if admission.owns(&loaded.runtime)) {
        *state = invalidated(std::mem::replace(&mut *state, Module::Unbound));
    }
}

fn admit() -> Result<Admission, Status> {
    admit_action(false)
}

fn admit_action(terminal: bool) -> Result<Admission, Status> {
    let state = state();
    match &*state {
        Module::Loaded(loaded) => Ok(Admission::new(loaded.suites, Arc::clone(&loaded.runtime))),
        Module::InvalidLoaded(loaded) if terminal => {
            Ok(Admission::new(loaded.suites, Arc::clone(&loaded.runtime)))
        }
        Module::Unbound | Module::Invalid(None) => Err(Status::BAD_HANDLE),
        _ => Err(Status::FAILED),
    }
}

fn unload() -> Result<(), Status> {
    let (loaded, invalid) = {
        let mut state = state();
        match &*state {
            Module::Loaded(loaded) | Module::InvalidLoaded(loaded)
                if Arc::strong_count(&loaded.runtime) != 1 =>
            {
                return Err(Status::FATAL);
            }
            Module::Loading(_)
            | Module::InvalidLoading(_)
            | Module::Closing(_)
            | Module::InvalidClosing(_) => {
                return Err(Status::FATAL);
            }
            _ => {}
        }
        let previous = std::mem::replace(&mut *state, Module::Unbound);
        let (loaded, invalid) = match previous {
            Module::Loaded(loaded) => (loaded, false),
            Module::InvalidLoaded(loaded) => (loaded, true),
            Module::Bound(binding) | Module::Invalid(Some(binding)) => {
                *state = Module::Bound(binding);
                return Ok(());
            }
            Module::Unbound | Module::Invalid(None) => return Ok(()),
            _ => unreachable!("transitional unload rejected before taking ownership"),
        };
        *state = Module::Closing(loaded.binding);
        (loaded, invalid)
    };
    if !matches!(
        catch_unwind(AssertUnwindSafe(|| loaded.runtime.check_close().is_ok())),
        Ok(true)
    ) {
        let mut state = state();
        *state = if invalid || matches!(*state, Module::InvalidClosing(_)) {
            Module::InvalidLoaded(loaded)
        } else {
            Module::Loaded(loaded)
        };
        return Err(Status::FATAL);
    }
    // Closing excludes new admission and the count check excluded existing
    // callbacks. Keep the same owner even if that invariant is ever violated.
    let runtime = match Arc::try_unwrap(loaded.runtime) {
        Ok(runtime) => runtime,
        Err(runtime) => {
            *state() = Module::InvalidLoaded(Loaded { runtime, ..loaded });
            return Err(Status::FATAL);
        }
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        runtime.close().map_err(|_| Status::FATAL)
    }))
    .map_err(|_| Status::FATAL)
    .and_then(|result| result);
    let mut state = state();
    *state = if matches!(*state, Module::InvalidClosing(_)) {
        Module::Invalid(Some(loaded.binding))
    } else {
        Module::Bound(loaded.binding)
    };
    result
}

#[allow(
    unsafe_code,
    dead_code,
    nonstandard_style,
    clippy::allow_attributes_without_reason,
    reason = "bindgen preserves foreign declarations and emits layout assertions with generated lint attributes; complete suites precede host-adapter consumers"
)]
pub(super) mod sys;

#[cfg(feature = "test-support")]
#[allow(
    unsafe_code,
    reason = "the native ABI fixture borrows immutable fact tables and exercises synchronous foreign callbacks and variadic fields"
)]
mod abi_tests;
