//! Safe host-instance composition. Publication and submission stay in Instance.

#![forbid(unsafe_code)]

use std::cell::{Cell, RefCell};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use film_juicer_core::{profile, state};
use state::{
    CaptureFailure, CaptureFailureKind, CaptureMode, CaptureStart, FrameInput, FrameInstall, Update,
};

use super::param_id::ParamId;
use super::controls::{self, CaptureError};
use super::host::{Admission, Callback, InstanceHandles, ParameterAccess, Status};

pub(crate) struct Owner {
    pub(super) loaded: Admission,
    pub(super) handles: InstanceHandles,
    pub(super) state: crate::instance::Instance,
    setup_failed: AtomicBool,
    grain_label: Vec<u8>,
    chroma_hint: Vec<u8>,
}

// Host-call provenance is local to the synchronous callback chain. An unrelated
// user/time event or another instance must remain admitted during nested edits.
thread_local! { static HOST_CALL: Cell<Option<usize>> = const { Cell::new(None) }; }
thread_local! { static RENDER_READ: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) }; }
struct RenderRead(usize);
impl RenderRead {
    fn enter(owner: usize) -> Result<Option<Self>, Status> {
        RENDER_READ.with(|stack| {
            let mut stack = stack.borrow_mut();
            if stack.contains(&owner) {
                return Ok(None);
            }
            stack.try_reserve(1).map_err(|_| Status::MEMORY)?;
            stack.push(owner);
            Ok(Some(Self(owner)))
        })
    }
}
impl Drop for RenderRead {
    fn drop(&mut self) {
        RENDER_READ.with(|stack| {
            let owner = stack.borrow_mut().pop();
            debug_assert_eq!(owner, Some(self.0));
        });
    }
}

struct HostCall(Option<usize>);
impl HostCall {
    fn new(owner: &Owner) -> Self {
        Self(HOST_CALL.replace(Some(std::ptr::from_ref(owner) as usize)))
    }
}
impl Drop for HostCall {
    fn drop(&mut self) {
        HOST_CALL.set(self.0);
    }
}

fn state_status(_: state::Error) -> Status {
    Status::FATAL
}

pub(super) fn seed(clock: u64, address: u64) -> u64 {
    let mut fields = [0u8; 16];
    fields[..8].copy_from_slice(&clock.to_ne_bytes());
    fields[8..].copy_from_slice(&address.to_ne_bytes());
    let seed = film_juicer_core::hash::bytes(&fields);
    if seed == 0 { 1 } else { seed }
}

struct Capture<'a> {
    owner: &'a Owner,
    ticket: u64,
    initial: bool,
    armed: Cell<bool>,
    completion: Cell<Update>,
    refresh: Cell<bool>,
    physical_write_failed: Cell<bool>,
    read_revision: Cell<Option<u64>>,
    action_pending: Cell<bool>,
}
impl<'a> Capture<'a> {
    fn new(owner: &'a Owner, mode: CaptureMode, claim_initial: bool) -> Result<Self, Status> {
        let first = if claim_initial {
            owner
                .state
                .begin_capture(CaptureMode::Initial)
                .map_err(state_status)?
        } else {
            CaptureStart::NotStarted
        };
        let initial = matches!(first, CaptureStart::Started { .. });
        let start = if initial {
            first
        } else {
            owner.state.begin_capture(mode).map_err(state_status)?
        };
        let ticket = match start {
            CaptureStart::Started { capture } => capture,
            CaptureStart::NotStarted => 0,
        };
        Ok(Self {
            owner,
            ticket,
            initial,
            armed: Cell::new(ticket != 0),
            completion: Cell::new(Update::Ignored),
            refresh: Cell::new(false),
            physical_write_failed: Cell::new(false),
            read_revision: Cell::new(None),
            action_pending: Cell::new(
                initial || matches!(mode, CaptureMode::Initial | CaptureMode::Change),
            ),
        })
    }
    fn current(&self) -> Result<bool, Status> {
        if self.ticket == 0 {
            Ok(false)
        } else {
            self.owner
                .state
                .capture_is_current(self.ticket)
                .map_err(state_status)
        }
    }
    fn complete(&self, update: Update) {
        self.completion.set(update);
        self.armed.set(false);
    }
    fn fail(&self, kind: CaptureFailureKind, text: &str) -> Result<Update, Status> {
        let action = self.action_pending.replace(false) && self.ticket != 0;
        if !self.armed.replace(false) && !action {
            return Ok(self.completion.get());
        }
        let failure = CaptureFailure::new(kind, text);
        let update = if action {
            self.owner.state.invalidate_action(self.ticket, failure)
        } else {
            self.owner.state.invalidate(self.ticket, failure)
        }
        .map_err(state_status)?;
        self.completion.set(update);
        Ok(update)
    }
}
impl Drop for Capture<'_> {
    fn drop(&mut self) {
        let _ = self.fail(CaptureFailureKind::Host, "parameter action abandoned");
    }
}

pub(crate) struct Controls<'a> {
    callback: &'a Callback<'a>,
    owner: &'a Owner,
    capture: Option<&'a Capture<'a>>,
}
impl Controls<'_> {
    pub(super) fn begin_read(&self) -> Result<(), Status> {
        let capture = self.capture.ok_or(Status::FATAL)?;
        self.owner.begin_read(capture).map(|_| ())
    }
    fn access(&self) -> ParameterAccess<'_> {
        self.callback.parameter_access(&self.owner.handles)
    }
    pub(super) fn double(&self, name: ParamId) -> Result<f64, Status> {
        self.access().double(name)
    }
    pub(super) fn integer(&self, name: ParamId) -> Result<i32, Status> {
        self.access().integer(name)
    }
    pub(super) fn double2(&self, name: ParamId) -> Result<[f64; 2], Status> {
        self.access().double2(name)
    }
    pub(super) fn double3(&self, name: ParamId) -> Result<[f64; 3], Status> {
        self.access().double3(name)
    }
    pub(super) fn string_bytes(&self, name: ParamId) -> Result<Vec<u8>, Status> {
        self.access().string_bytes(name)
    }
    fn write(&self, operation: impl FnOnce() -> Result<(), Status>) -> Result<(), Status> {
        let capture = self.capture.ok_or(Status::FATAL)?;
        let write = match self
            .owner
            .state
            .begin_host_write(capture.ticket)
            .map_err(state_status)?
        {
            state::HostWriteStart::Started(write) => write,
            state::HostWriteStart::Ignored => return Ok(()),
        };
        let outcome = {
            let _call = HostCall::new(self.owner);
            operation()
        };
        let failure = outcome.as_ref().err().map(|status| {
            CaptureFailure::new(
                if status.code() == super::sys::kOfxStatErrMemory {
                    CaptureFailureKind::Allocation
                } else {
                    CaptureFailureKind::Host
                },
                if status.code() == super::sys::kOfxStatErrMemory {
                    "host parameter write allocation failed"
                } else {
                    "host parameter write failed"
                },
            )
        });
        if failure.is_some() {
            capture.physical_write_failed.set(true);
        }
        let end = write
            .finish(failure.map_or(Ok(()), Err))
            .map_err(state_status);
        if matches!(end, Ok(state::WriteEnd::RefreshRequired)) {
            capture.refresh.set(true);
        }
        outcome?;
        end.map(|_| ())
    }
    pub(super) fn set_double(&self, name: ParamId, value: f64) -> Result<(), Status> {
        self.write(|| self.access().set_double(name, value))
    }
    pub(super) fn set_integer(&self, name: ParamId, value: i32) -> Result<(), Status> {
        self.write(|| self.access().set_integer(name, value))
    }
    pub(super) fn set_double2(&self, name: ParamId, value: [f64; 2]) -> Result<(), Status> {
        self.write(|| self.access().set_double2(name, value))
    }
    pub(super) fn set_double3(&self, name: ParamId, value: [f64; 3]) -> Result<(), Status> {
        self.write(|| self.access().set_double3(name, value))
    }
    pub(super) fn set_string_bytes(&self, name: ParamId, value: &[u8]) -> Result<(), Status> {
        self.write(|| self.access().set_string_bytes(name, value))
    }
    fn property(&self, operation: impl FnOnce() -> Result<(), Status>) -> Result<(), Status> {
        if let Some(capture) = self.capture
            && !self
                .owner
                .state
                .property_is_current(capture.ticket)
                .map_err(state_status)?
        {
            return Ok(());
        }
        let _call = HostCall::new(self.owner);
        operation()
    }
    pub(super) fn set_enabled(&self, name: ParamId, value: bool) -> Result<(), Status> {
        self.property(|| self.access().set_enabled(name, value))
    }
    pub(super) fn set_secret(&self, name: ParamId, value: bool) -> Result<(), Status> {
        self.property(|| self.access().set_secret(name, value))
    }
    pub(super) fn set_label(&self, name: ParamId, value: &[u8]) -> Result<(), Status> {
        self.property(|| self.access().set_label(name, value))
    }
    pub(super) fn set_hint(&self, name: ParamId, value: &[u8]) -> Result<(), Status> {
        self.property(|| self.access().set_hint(name, value))
    }
    pub(super) fn base_grain_label(&self) -> &[u8] {
        &self.owner.grain_label
    }
    pub(super) fn grain_chroma_hint(&self) -> &[u8] {
        &self.owner.chroma_hint
    }
    pub(super) fn catalog(&self) -> Result<Arc<profile::Catalog>, Status> {
        self.owner
            .loaded
            .runtime()
            .assets()
            .catalog()
            .map_err(|_| Status::FAILED)
    }
    pub(super) fn film_defaults(
        &self,
        key: &[u8],
    ) -> Result<Option<profile::FilmProcessingDefaults>, Status> {
        let key = std::str::from_utf8(key).map_err(|_| Status::FATAL)?;
        let film = match self.owner.loaded.runtime().assets().film(key) {
            Ok(film) => film,
            Err(error) => {
                use film_juicer_core::assets::AssetError;
                use film_juicer_core::profile::ProfileCompletionErrorKind;

                return match error {
                    AssetError::ProfileCompletion(error)
                        if error.kind == ProfileCompletionErrorKind::Capacity =>
                    {
                        Err(Status::MEMORY)
                    }
                    AssetError::ProfileCompletion(_)
                    | AssetError::MissingProfile { .. }
                    | AssetError::ProfileDecode(_) => Ok(None),
                    _ => Err(Status::FATAL),
                };
            }
        };
        let defaults = film.processing_defaults();
        Ok(Some(profile::FilmProcessingDefaults {
            gamma_samelayer_rgb: defaults.gamma_samelayer_rgb,
            gamma_interlayer_r_to_gb: defaults.gamma_interlayer_r_to_gb,
            gamma_interlayer_g_to_rb: defaults.gamma_interlayer_g_to_rb,
            gamma_interlayer_b_to_rg: defaults.gamma_interlayer_b_to_rg,
            halation_first_sigma_um: defaults.halation_first_sigma_um,
            halation_primary_amount: defaults.halation_primary_amount,
            hanatos_spectral_gaussian_blur_default: defaults.hanatos_spectral_gaussian_blur_default,
        }))
    }
}

impl Owner {
    pub(super) fn new(callback: &Callback<'_>, handles: InstanceHandles) -> Result<Self, Status> {
        let access = callback.parameter_access(&handles);
        let label = access.label(ParamId::GrainPreset)?;
        let hint = access.hint(ParamId::GrainChroma)?;
        for name in [
            ParamId::GrainParticleScaleMaster,
            ParamId::GrainParticleScaleLayersMaster,
            ParamId::GrainDensityMinMaster,
            ParamId::GrainUniformityMaster,
        ] {
            let _ = access.double(name)?;
        }
        let grain_label = if label.is_empty() {
            super::host::copy_bytes(b"Grain Preset")?
        } else {
            label
        };
        let chroma_hint = if hint.is_empty() {
            super::host::copy_bytes(b"0 = achromatic, 1 = independent RGB grain.")?
        } else {
            hint
        };
        Ok(Self {
            loaded: callback.retain(),
            handles,
            state: crate::instance::Instance::new(),
            setup_failed: AtomicBool::new(false),
            grain_label,
            chroma_hint,
        })
    }
    pub(super) fn bind_address(&self, address: usize) -> Result<(), Status> {
        let now = crate::platform::clock_nanoseconds();
        let seed = seed(now.ok_or(Status::FATAL)?, address as u64);
        self.state
            .bind_identity(seed, seed)
            .map_err(|_| Status::FATAL)
    }
    pub(super) fn fail(&self) {
        self.setup_failed.store(true, Ordering::Release);
        self.state.block();
    }
    pub(super) fn require_live(&self) -> Result<(), Status> {
        if self.setup_failed.load(Ordering::Acquire) {
            Err(Status::FATAL)
        } else {
            Ok(())
        }
    }
    pub(super) fn host_call<T>(&self, operation: impl FnOnce() -> T) -> T {
        let _call = HostCall::new(self);
        operation()
    }
    pub(super) fn initialize(&self, callback: &Callback<'_>) -> Result<(), Status> {
        controls::initialize(&Controls {
            callback,
            owner: self,
            capture: None,
        })
    }
    pub(super) fn retire(&self) -> Result<(), Status> {
        if let Some(identity) = self.state.identity() {
            let result = self
                .loaded
                .runtime()
                .native()
                .retire_instance(identity.instance_token);
            result.map_err(|_| Status::FATAL)?;
        }
        Ok(())
    }
    pub(super) fn changed(
        &self,
        callback: &Callback<'_>,
        name: &[u8],
        plugin: bool,
        user: bool,
    ) -> Result<(), Status> {
        self.require_live()?;
        if plugin && HOST_CALL.get() == Some(std::ptr::from_ref(self) as usize) {
            return Ok(());
        }
        let capture = Capture::new(self, CaptureMode::Change, true)?;
        let refresh;
        {
            let controls = Controls {
                callback,
                owner: self,
                capture: Some(&capture),
            };
            let result: Result<(), Status> = (|| {
                // Preserve host-call/failure order where defaults can still
                // write values after UI. Other changes seal before the UI tail.
                let read_before_ui = !capture.initial
                    && name != b"FilmProfileKey"
                    && name != b"CouplersGammaUseStock";
                let _early = controls::change(&controls, name, user, read_before_ui)?;
                if capture.initial
                    || name == b"FilmProfileKey"
                    || (name == b"CouplersGammaUseStock"
                        && controls.integer(ParamId::CouplersGammaUseStock)? != 0)
                {
                    controls::apply_dir_defaults(&controls)?;
                }
                self.finish(&controls, &capture)?;
                Ok(())
            })();
            if let Err(status) = result {
                let kind = if status.code() == super::sys::kOfxStatErrMemory {
                    CaptureFailureKind::Allocation
                } else {
                    CaptureFailureKind::Host
                };
                capture.fail(
                    kind,
                    if kind == CaptureFailureKind::Allocation {
                        "host parameter action allocation failed"
                    } else {
                        "host parameter action failed"
                    },
                )?;
                return Err(status);
            }
            refresh = capture.refresh.get();
        }
        drop(capture);
        if refresh {
            self.refresh(callback)?;
        }
        Ok(())
    }
    pub(super) fn render_capture(
        &self,
        callback: &Callback<'_>,
        initial: bool,
    ) -> Result<Option<FrameInput<'_>>, Status> {
        if !initial {
            return self.refresh(callback);
        }
        let capture = Capture::new(self, CaptureMode::Initial, false)?;
        if capture.ticket == 0 {
            return Ok(None);
        }
        let controls = Controls {
            callback,
            owner: self,
            capture: Some(&capture),
        };
        let result = (|| {
            controls::apply_dir_defaults(&controls)?;
            self.finish(&controls, &capture)
        })();
        let input = match result {
            Ok(input) => input,
            Err(status) => {
                let update = capture.fail(
                    if status.code() == super::sys::kOfxStatErrMemory {
                        CaptureFailureKind::Allocation
                    } else {
                        CaptureFailureKind::Host
                    },
                    "host parameter capture failed",
                )?;
                if update != Update::Ignored || capture.physical_write_failed.get() {
                    return Err(status);
                }
                None
            }
        };
        let refresh = capture.refresh.get();
        drop(capture);
        if refresh {
            return self.refresh(callback);
        }
        Ok(input)
    }
    fn refresh(&self, callback: &Callback<'_>) -> Result<Option<FrameInput<'_>>, Status> {
        self.read_capture(callback, CaptureMode::Refresh)
    }
    pub(super) fn render_read(
        &self,
        callback: &Callback<'_>,
    ) -> Result<Option<FrameInput<'_>>, Status> {
        let Some(_guard) = RenderRead::enter(std::ptr::from_ref(self) as usize)? else {
            return Ok(None);
        };
        self.read_capture(callback, CaptureMode::RenderRead)
    }
    fn read_capture(
        &self,
        callback: &Callback<'_>,
        mode: CaptureMode,
    ) -> Result<Option<FrameInput<'_>>, Status> {
        let capture = Capture::new(self, mode, false)?;
        if capture.ticket == 0 {
            return Ok(None);
        }
        let controls = Controls {
            callback,
            owner: self,
            capture: Some(&capture),
        };
        match self.finish(&controls, &capture) {
            Ok(input) => Ok(input),
            Err(status) => {
                let kind = if status.code() == super::sys::kOfxStatErrMemory {
                    CaptureFailureKind::Allocation
                } else {
                    CaptureFailureKind::Host
                };
                let update = capture.fail(
                    kind,
                    if kind == CaptureFailureKind::Allocation {
                        "host parameter capture allocation failed"
                    } else {
                        "host parameter capture failed"
                    },
                )?;
                if update != Update::Ignored || capture.physical_write_failed.get() {
                    return Err(status);
                }
                Ok(None)
            }
        }
    }
    fn begin_read(&self, capture: &Capture<'_>) -> Result<Option<u64>, Status> {
        if !capture.armed.get() {
            return Ok(None);
        }
        if let Some(revision) = capture.read_revision.get() {
            return Ok(Some(revision));
        }
        match self
            .state
            .begin_snapshot(capture.ticket)
            .map_err(state_status)?
        {
            state::SnapshotStart::Ready { write_revision } => {
                capture.read_revision.set(Some(write_revision));
                Ok(Some(write_revision))
            }
            state::SnapshotStart::Ignored => {
                capture.complete(Update::Ignored);
                Ok(None)
            }
            state::SnapshotStart::Deferred => {
                capture.complete(Update::Deferred);
                Ok(None)
            }
        }
    }
    fn finish(
        &self,
        controls: &Controls<'_>,
        capture: &Capture<'_>,
    ) -> Result<Option<FrameInput<'_>>, Status> {
        // UI/action failures still belong to the latest real change. Once its
        // full parameter read starts, a fresher render read can supersede it.
        capture.action_pending.set(false);
        let Some(revision) = self.begin_read(capture)? else {
            return Ok(None);
        };
        self.finish_read(controls, capture, revision)
    }
    fn finish_read(
        &self,
        controls: &Controls<'_>,
        capture: &Capture<'_>,
        revision: u64,
    ) -> Result<Option<FrameInput<'_>>, Status> {
        let captured = match controls::capture(controls) {
            Ok(captured) => captured,
            Err(CaptureError::Host(status)) => return Err(status),
            Err(CaptureError::Invalid(text)) => {
                capture.fail(CaptureFailureKind::InvalidControls, &text)?;
                return Ok(None);
            }
        };
        if !capture.current()? {
            capture.complete(Update::Ignored);
            return Ok(None);
        }
        match captured.admit() {
            Ok((params, identity)) => {
                match self
                    .state
                    .install_frame(capture.ticket, revision, params, identity)
                    .map_err(state_status)?
                {
                    FrameInstall::Applied(input) => {
                        capture.complete(Update::Applied);
                        Ok(Some(input))
                    }
                    FrameInstall::Ignored => {
                        capture.complete(Update::Ignored);
                        Ok(None)
                    }
                    FrameInstall::Deferred => {
                        capture.complete(Update::Deferred);
                        Ok(None)
                    }
                }
            }
            Err(_) => {
                let update = self
                    .state
                    .fail_snapshot(
                        capture.ticket,
                        revision,
                        CaptureFailure::new(
                            CaptureFailureKind::InvalidControls,
                            "preparation key is not UTF-8",
                        ),
                    )
                    .map_err(state_status)?;
                capture.complete(update);
                if update == Update::Applied {
                    Err(Status::FATAL)
                } else {
                    Ok(None)
                }
            }
        }
    }
}

#[cfg(test)]
mod render_read_tests {
    use super::RenderRead;

    #[test]
    fn nested_render_read_is_bounded_and_guard_releases_on_unwind() {
        struct IntentionalUnwind;
        let panic = std::panic::catch_unwind(|| {
            let _outer = RenderRead::enter(1).ok().flatten().expect("first read");
            let _other = RenderRead::enter(2).ok().flatten().expect("other owner");
            assert!(RenderRead::enter(1).ok().flatten().is_none());
            assert!(RenderRead::enter(2).ok().flatten().is_none());
            std::panic::panic_any(IntentionalUnwind);
        })
        .expect_err("intentional guard unwind");
        assert!(panic.is::<IntentionalUnwind>());
        assert!(RenderRead::enter(1).ok().flatten().is_some());
        assert!(RenderRead::enter(2).ok().flatten().is_some());
    }
}
