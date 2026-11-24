//! Complete-state publication and capture admission, independent of host and CUDA ownership.

#![expect(
    clippy::result_large_err,
    reason = "Capture failure recording and propagation retain the bounded diagnostic inline so unwind invalidation never allocates."
)]

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::recipe::{Params, PrepareError, RenderState};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CaptureFailureKind {
    InvalidControls,
    Host,
    Allocation,
}

/// Failure recording, including unwind cleanup, never allocates diagnostic storage.
#[derive(Clone, Debug)]
pub struct CaptureFailure {
    kind: CaptureFailureKind,
    diagnostic: [u8; 2047],
    len: usize,
}

impl CaptureFailure {
    pub fn new(kind: CaptureFailureKind, diagnostic: &str) -> Self {
        let diagnostic = if diagnostic.is_empty() {
            match kind {
                CaptureFailureKind::InvalidControls => "InvalidSnapshotControls",
                CaptureFailureKind::Host => "host parameter capture failed",
                CaptureFailureKind::Allocation => "parameter capture allocation failed",
            }
        } else {
            diagnostic
        };
        let mut len = diagnostic.len().min(2047);
        while !diagnostic.is_char_boundary(len) {
            len -= 1;
        }
        let mut failure = Self {
            kind,
            diagnostic: [0; 2047],
            len,
        };
        failure.diagnostic[..len].copy_from_slice(&diagnostic.as_bytes()[..len]);
        failure
    }

    pub fn kind(&self) -> CaptureFailureKind {
        self.kind
    }
    pub fn diagnostic(&self) -> &str {
        std::str::from_utf8(&self.diagnostic[..self.len]).expect("bounded UTF-8 diagnostic")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CaptureMode {
    Initial,
    Change,
    Refresh,
    RenderRead,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CaptureStart {
    Started { capture: u64 },
    NotStarted,
}
pub enum HostWriteStart {
    Started(HostWrite),
    Ignored,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SnapshotStart {
    Ready { write_revision: u64 },
    Deferred,
    Ignored,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WriteEnd {
    Complete,
    RefreshRequired,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Update {
    Applied,
    Ignored,
    Deferred,
}

#[derive(Clone, Copy)]
enum CapturePhase {
    Unread,
    Reading { revision: u64 },
}

#[derive(Clone, Copy)]
enum FailureOrigin {
    ReadOnly,
    Action,
    PhysicalWrite,
}

impl CaptureMode {
    fn failure_origin(self) -> FailureOrigin {
        match self {
            Self::Refresh | Self::RenderRead => FailureOrigin::ReadOnly,
            Self::Initial | Self::Change => FailureOrigin::Action,
        }
    }
}

// Each alternative owns its completion obligation; no validity/hash sentinel.
#[expect(
    clippy::large_enum_variant,
    reason = "Capture failure recording and propagation retain the bounded diagnostic inline so unwind invalidation never allocates."
)]
enum Pending {
    Uninitialized,
    Capturing {
        ticket: u64,
        mode: CaptureMode,
        phase: CapturePhase,
    },
    AwaitingWrites,
    NeedsRefresh,
    Valid {
        params: Arc<Params>,
        identity: u64,
    },
    Failed {
        failure: CaptureFailure,
        origin: FailureOrigin,
    },
    Blocked,
}

#[derive(Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "Capture failure recording and propagation retain the bounded diagnostic inline so unwind invalidation never allocates."
)]
pub enum Error {
    Uninitialized,
    Capturing,
    RefreshRequired,
    CaptureFailed(CaptureFailure),
    InvalidCaptureTicket,
    InvalidCaptureOperation,
    Preparation(PrepareError),
    Poisoned,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Uninitialized => f.write_str("render snapshot is uninitialized"),
            Self::Capturing => f.write_str("render snapshot capture in progress"),
            Self::RefreshRequired => f.write_str("render snapshot needs a fresh read"),
            Self::CaptureFailed(failure) => f.write_str(failure.diagnostic()),
            Self::InvalidCaptureTicket => f.write_str("invalid capture ticket"),
            Self::InvalidCaptureOperation => f.write_str("invalid capture phase or write revision"),
            Self::Preparation(error) => error.fmt(f),
            Self::Poisoned => f.write_str("render publication synchronization blocked or poisoned"),
        }
    }
}
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Preparation(error) => Some(error),
            _ => None,
        }
    }
}

pub struct Admission {
    state: Arc<RenderState>,
    input_identity: u64,
    build: u64,
}

/// Immutable photographic input selected for one frame.
pub struct FrameInput<'a> {
    publication: &'a Publication,
    params: Arc<Params>,
    identity: u64,
}
impl FrameInput<'_> {
    #[cfg(feature = "grain-preview")]
    pub fn film_grain_selected(&self) -> bool {
        self.params.film_grain.is_some()
    }

    #[cfg(feature = "test-support")]
    pub fn input_identity(&self) -> u64 {
        self.identity
    }
}

pub enum FrameInstall<'a> {
    Applied(FrameInput<'a>),
    Ignored,
    Deferred,
}

impl FrameInstall<'_> {
    fn into_update(self) -> Update {
        match self {
            Self::Applied(_) => Update::Applied,
            Self::Ignored => Update::Ignored,
            Self::Deferred => Update::Deferred,
        }
    }
}
impl Admission {
    pub fn state(&self) -> &RenderState {
        &self.state
    }
    pub fn input_identity(&self) -> u64 {
        self.input_identity
    }
    pub fn build(&self) -> u64 {
        self.build
    }
    #[cfg(feature = "test-support")]
    pub fn shared_state(&self) -> &Arc<RenderState> {
        &self.state
    }
    pub fn into_state(self) -> Arc<RenderState> {
        self.state
    }
}

struct Published {
    identity: u64,
    state: Arc<RenderState>,
    build: u64,
}
impl Published {
    fn matches(
        &self,
        identity: u64,
        #[cfg(feature = "grain-preview")] film_raster: Option<crate::film::grain::preview::Raster>,
    ) -> bool {
        #[cfg(feature = "grain-preview")]
        let same_geometry = self.state.film_grain().map(|grain| grain.raster()) == film_raster;
        #[cfg(not(feature = "grain-preview"))]
        let same_geometry = true;
        self.identity == identity && same_geometry
    }

    fn admission(&self) -> Admission {
        Admission {
            state: Arc::clone(&self.state),
            input_identity: self.identity,
            build: self.build,
        }
    }
}
struct State {
    pending: Pending,
    published: Option<Published>,
    issued: u64,
    last_action_ticket: u64,
    write_revision: u64,
    outstanding_writes: u64,
    build: u64,
}
impl State {
    fn current(&self, capture: u64) -> Result<bool, Error> {
        if capture == 0 || capture > self.issued {
            return Err(Error::InvalidCaptureTicket);
        }
        Ok(matches!(self.pending, Pending::Capturing { ticket, .. } if ticket == capture))
    }
    fn input(&self) -> Result<(&Arc<Params>, u64), Error> {
        match &self.pending {
            Pending::Blocked => Err(Error::Poisoned),
            Pending::Failed { failure, .. } => Err(Error::CaptureFailed(failure.clone())),
            _ if self.outstanding_writes != 0 => Err(Error::Capturing),
            Pending::Uninitialized => Err(Error::Uninitialized),
            Pending::Capturing { .. } | Pending::AwaitingWrites => Err(Error::Capturing),
            Pending::NeedsRefresh => Err(Error::RefreshRequired),
            Pending::Valid { params, identity } => Ok((params, *identity)),
        }
    }
}

/// The rebuild gate precedes the short state lock. Host calls, allocation,
/// preparation and destruction of retired data happen outside the state lock.
pub struct Publication {
    state: Mutex<State>,
    rebuild: Mutex<()>,
}
impl Default for Publication {
    fn default() -> Self {
        Self::new()
    }
}
impl Publication {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(State {
                pending: Pending::Uninitialized,
                published: None,
                issued: 0,
                last_action_ticket: 0,
                write_revision: 0,
                outstanding_writes: 0,
                build: 0,
            }),
            rebuild: Mutex::new(()),
        }
    }
    fn lock(&self) -> Result<MutexGuard<'_, State>, Error> {
        if self.rebuild.is_poisoned() {
            return Err(Error::Poisoned);
        }
        let state = self.state.lock().map_err(|_| Error::Poisoned)?;
        if matches!(state.pending, Pending::Blocked) {
            return Err(Error::Poisoned);
        }
        Ok(state)
    }
    pub fn block(&self) {
        if let Ok(mut state) = self.state.lock() {
            let retired = std::mem::replace(&mut state.pending, Pending::Blocked);
            drop(state);
            drop(retired);
        }
    }
    pub fn begin_capture(&self, mode: CaptureMode) -> Result<CaptureStart, Error> {
        let mut state = self.lock()?;
        if (mode == CaptureMode::Initial && !matches!(state.pending, Pending::Uninitialized))
            || (mode == CaptureMode::Refresh
                && (!matches!(state.pending, Pending::NeedsRefresh)
                    || state.outstanding_writes != 0))
            || (mode == CaptureMode::RenderRead
                && (state.outstanding_writes != 0
                    || !matches!(state.pending, Pending::Capturing {
                        mode: CaptureMode::Initial | CaptureMode::Change | CaptureMode::Refresh,
                        phase: CapturePhase::Reading { revision }, ..
                    } if revision == state.write_revision)))
        {
            return Ok(CaptureStart::NotStarted);
        }
        let Some(ticket) = state.issued.checked_add(1) else {
            let retired = std::mem::replace(&mut state.pending, Pending::Blocked);
            drop(state);
            drop(retired);
            return Err(Error::Poisoned);
        };
        state.issued = ticket;
        if matches!(mode, CaptureMode::Initial | CaptureMode::Change) {
            state.last_action_ticket = ticket;
        }
        let retired = std::mem::replace(
            &mut state.pending,
            Pending::Capturing {
                ticket,
                mode,
                phase: CapturePhase::Unread,
            },
        );
        drop(state);
        drop(retired);
        Ok(CaptureStart::Started { capture: ticket })
    }
    pub fn capture_is_current(&self, capture: u64) -> Result<bool, Error> {
        self.lock()?.current(capture)
    }
    pub fn property_is_current(&self, capture: u64) -> Result<bool, Error> {
        let state = self.lock()?;
        state.current(capture)?;
        Ok(state.last_action_ticket == capture
            && !matches!(
                state.pending,
                Pending::Failed {
                    origin: FailureOrigin::Action | FailureOrigin::PhysicalWrite,
                    ..
                }
            ))
    }
    pub fn begin_host_write(self: &Arc<Self>, capture: u64) -> Result<HostWriteStart, Error> {
        let mut state = self.lock()?;
        if !state.current(capture)? {
            return Ok(HostWriteStart::Ignored);
        }
        if !matches!(
            state.pending,
            Pending::Capturing {
                mode: CaptureMode::Initial | CaptureMode::Change,
                phase: CapturePhase::Unread,
                ..
            }
        ) {
            return Err(Error::InvalidCaptureOperation);
        }
        let counts = state
            .outstanding_writes
            .checked_add(1)
            .zip(state.write_revision.checked_add(1));
        let Some((writes, revision)) = counts else {
            let retired = std::mem::replace(&mut state.pending, Pending::Blocked);
            drop(state);
            drop(retired);
            return Err(Error::Poisoned);
        };
        state.outstanding_writes = writes;
        state.write_revision = revision;
        Ok(HostWriteStart::Started(HostWrite {
            publication: Some(Arc::clone(self)),
        }))
    }
    pub fn begin_snapshot(&self, capture: u64) -> Result<SnapshotStart, Error> {
        let mut state = self.lock()?;
        if !state.current(capture)? {
            return Ok(SnapshotStart::Ignored);
        }
        let Pending::Capturing {
            mode,
            phase: CapturePhase::Unread,
            ..
        } = state.pending
        else {
            return Err(Error::InvalidCaptureOperation);
        };
        if state.outstanding_writes != 0 {
            state.pending = Pending::AwaitingWrites;
            return Ok(SnapshotStart::Deferred);
        }
        let revision = state.write_revision;
        state.pending = Pending::Capturing {
            ticket: capture,
            mode,
            phase: CapturePhase::Reading { revision },
        };
        Ok(SnapshotStart::Ready {
            write_revision: revision,
        })
    }
    pub fn install(
        &self,
        capture: u64,
        write_revision: u64,
        params: Params,
        input_identity: u64,
    ) -> Result<Update, Error> {
        self.install_shared(capture, write_revision, Arc::new(params), input_identity)
    }
    fn install_shared(
        &self,
        capture: u64,
        write_revision: u64,
        params: Arc<Params>,
        identity: u64,
    ) -> Result<Update, Error> {
        self.install_frame_shared(capture, write_revision, params, identity)
            .map(FrameInstall::into_update)
    }
    pub fn install_frame(
        &self,
        capture: u64,
        write_revision: u64,
        params: Params,
        identity: u64,
    ) -> Result<FrameInstall<'_>, Error> {
        self.install_frame_shared(capture, write_revision, Arc::new(params), identity)
    }
    fn install_frame_shared(
        &self,
        capture: u64,
        write_revision: u64,
        params: Arc<Params>,
        identity: u64,
    ) -> Result<FrameInstall<'_>, Error> {
        let mut state = self.lock()?;
        if !state.current(capture)? {
            return Ok(FrameInstall::Ignored);
        }
        if !matches!(state.pending, Pending::Capturing { phase: CapturePhase::Reading { revision }, .. }
            if revision == write_revision)
        {
            return Err(Error::InvalidCaptureOperation);
        }
        if state.write_revision != write_revision || state.outstanding_writes != 0 {
            state.pending = if state.outstanding_writes == 0 {
                Pending::NeedsRefresh
            } else {
                Pending::AwaitingWrites
            };
            return Ok(FrameInstall::Deferred);
        }
        state.pending = Pending::Valid {
            params: Arc::clone(&params),
            identity,
        };
        Ok(FrameInstall::Applied(FrameInput {
            publication: self,
            params,
            identity,
        }))
    }
    pub fn frame_input(&self) -> Result<FrameInput<'_>, Error> {
        let state = self.lock()?;
        let (params, identity) = state.input()?;
        Ok(FrameInput {
            publication: self,
            params: Arc::clone(params),
            identity,
        })
    }
    /// A valid snapshot whose foreign decoding failed still completes its exact
    /// reading phase. Protocol errors must leave the capture obligation armed.
    pub fn fail_snapshot(
        &self,
        capture: u64,
        write_revision: u64,
        failure: CaptureFailure,
    ) -> Result<Update, Error> {
        let mut state = self.lock()?;
        if !state.current(capture)? {
            return Ok(Update::Ignored);
        }
        let Pending::Capturing {
            mode,
            phase: CapturePhase::Reading { revision },
            ..
        } = state.pending
        else {
            return Err(Error::InvalidCaptureOperation);
        };
        if revision != write_revision {
            return Err(Error::InvalidCaptureOperation);
        }
        state.pending = Pending::Failed {
            failure,
            origin: mode.failure_origin(),
        };
        Ok(Update::Applied)
    }
    pub fn invalidate(&self, capture: u64, failure: CaptureFailure) -> Result<Update, Error> {
        let mut state = self.lock()?;
        if !state.current(capture)? {
            return Ok(Update::Ignored);
        }
        let Pending::Capturing { mode, .. } = state.pending else {
            unreachable!("current ticket must own a capture")
        };
        state.pending = Pending::Failed {
            failure,
            origin: mode.failure_origin(),
        };
        Ok(Update::Applied)
    }
    pub fn invalidate_action(
        &self,
        capture: u64,
        failure: CaptureFailure,
    ) -> Result<Update, Error> {
        let mut state = self.lock()?;
        state.current(capture)?;
        if state.last_action_ticket != capture || matches!(state.pending, Pending::Failed { .. }) {
            return Ok(Update::Ignored);
        }
        state.pending = Pending::Failed {
            failure,
            origin: FailureOrigin::Action,
        };
        Ok(Update::Applied)
    }
    fn finish_write(&self, outcome: Result<(), CaptureFailure>) -> Result<WriteEnd, Error> {
        let mut state = self.lock()?;
        let Some(revision) = state.write_revision.checked_add(1) else {
            let retired = std::mem::replace(&mut state.pending, Pending::Blocked);
            drop(state);
            drop(retired);
            return Err(Error::Poisoned);
        };
        // Only a guard created by begin_host_write can reach this operation.
        state.outstanding_writes -= 1;
        state.write_revision = revision;
        if let Err(failure) = outcome {
            if !matches!(state.pending, Pending::Failed { .. }) {
                let retired = std::mem::replace(
                    &mut state.pending,
                    Pending::Failed {
                        failure,
                        origin: FailureOrigin::PhysicalWrite,
                    },
                );
                drop(state);
                drop(retired);
            }
            return Ok(WriteEnd::Complete);
        }
        if state.outstanding_writes == 0 && matches!(state.pending, Pending::AwaitingWrites) {
            state.pending = Pending::NeedsRefresh;
            return Ok(WriteEnd::RefreshRequired);
        }
        Ok(WriteEnd::Complete)
    }
    pub fn admit_frame(
        &self,
        input: FrameInput<'_>,
        prepare: impl FnOnce(&Params) -> Result<RenderState, PrepareError>,
    ) -> Result<Arc<RenderState>, Error> {
        self.admit_frame_with(
            input,
            prepare,
            #[cfg(feature = "grain-preview")]
            None,
        )
    }

    #[cfg(feature = "grain-preview")]
    pub fn admit_preview_frame(
        &self,
        input: FrameInput<'_>,
        frame: crate::recipe::PreviewFrame,
        prepare: impl FnOnce(&Params) -> Result<RenderState, PrepareError>,
    ) -> Result<Arc<RenderState>, Error> {
        self.admit_frame_with(input, prepare, Some(frame))
    }

    fn admit_frame_with(
        &self,
        input: FrameInput<'_>,
        prepare: impl FnOnce(&Params) -> Result<RenderState, PrepareError>,
        #[cfg(feature = "grain-preview")] frame: Option<crate::recipe::PreviewFrame>,
    ) -> Result<Arc<RenderState>, Error> {
        if !std::ptr::eq(input.publication, self) {
            return Err(Error::InvalidCaptureOperation);
        }
        let prepare = || {
            prepare(&input.params).map(Arc::new).map_err(|error| {
                if error.is_panic() {
                    self.block();
                }
                Error::Preparation(error)
            })
        };
        #[cfg(feature = "grain-preview")]
        let film_raster = match frame.filter(|_| input.film_grain_selected()) {
            // This render owns the grain-free state. Leave the timeline
            // publication and the geometry it was prepared for in place.
            Some(crate::recipe::PreviewFrame::Thumbnail) => return prepare(),
            Some(crate::recipe::PreviewFrame::Raster(raster)) => Some(raster),
            None => None,
        };
        let _rebuild = self.rebuild.lock().map_err(|_| Error::Poisoned)?;
        {
            let state = self.lock()?;
            if let Some(published) = &state.published
                && published.matches(
                    input.identity,
                    #[cfg(feature = "grain-preview")]
                    film_raster,
                )
            {
                return Ok(Arc::clone(&published.state));
            }
        }
        let candidate = prepare()?;
        let mut state = self.lock()?;
        if matches!(state.input(), Ok((_, identity)) if identity == input.identity) {
            let Some(build) = state.build.checked_add(1) else {
                let retired = std::mem::replace(&mut state.pending, Pending::Blocked);
                drop(state);
                drop(retired);
                return Err(Error::Poisoned);
            };
            state.build = build;
            let retired = state.published.replace(Published {
                identity: input.identity,
                state: Arc::clone(&candidate),
                build,
            });
            drop(state);
            drop(retired);
        }
        Ok(candidate)
    }
    pub fn admit(
        &self,
        mut prepare: impl FnMut(&Params) -> Result<RenderState, PrepareError>,
    ) -> Result<Admission, Error> {
        // An unfinished host action never waits for a builder or reentrant host work.
        {
            self.lock()?.input()?;
        }
        #[cfg(test)]
        tests::observe_rebuild_contention(&self.rebuild);
        let _rebuild = self.rebuild.lock().map_err(|_| Error::Poisoned)?;
        loop {
            let (params, identity) = {
                let state = self.lock()?;
                let (params, identity) = state.input()?;
                if let Some(published) = &state.published
                    && published.matches(
                        identity,
                        #[cfg(feature = "grain-preview")]
                        None,
                    )
                {
                    return Ok(published.admission());
                }
                (Arc::clone(params), identity)
            };
            let outcome = prepare(&params);
            if outcome.as_ref().err().is_some_and(PrepareError::is_panic) {
                let Err(error) = outcome else {
                    unreachable!("panic classification requires a preparation error")
                };
                // A caught leaf unwind is still terminal even if this build's
                // snapshot became obsolete while preparation was running.
                self.block();
                return Err(Error::Preparation(error));
            }
            let outcome = outcome.map(Arc::new);
            let mut state = self.lock()?;
            if !matches!(state.input(), Ok((_, current)) if current == identity) {
                drop(state);
                drop(outcome);
                continue;
            }
            match outcome {
                Ok(candidate) => {
                    let Some(build) = state.build.checked_add(1) else {
                        let retired = std::mem::replace(&mut state.pending, Pending::Blocked);
                        drop(state);
                        drop(retired);
                        drop(candidate);
                        return Err(Error::Poisoned);
                    };
                    state.build = build;
                    let published = Published {
                        identity,
                        state: candidate,
                        build,
                    };
                    let admitted = published.admission();
                    let retired = state.published.replace(published);
                    drop(state);
                    drop(retired);
                    return Ok(admitted);
                }
                Err(error) => {
                    drop(state);
                    return Err(Error::Preparation(error));
                }
            }
        }
    }
}

/// Read-only fixture holds from the actual owner, excluded from production.
#[cfg(feature = "test-support")]
pub struct Observation {
    pub pending: ObservedPending,
    pub published: Option<Admission>,
    pub issued: u64,
    pub write_revision: u64,
    pub outstanding_writes: u64,
}
#[cfg(feature = "test-support")]
#[expect(
    clippy::large_enum_variant,
    reason = "Fixture observations preserve the same inline nonallocating failure diagnostic as pending state."
)]
pub enum ObservedPending {
    Uninitialized,
    Capturing,
    AwaitingWrites,
    NeedsRefresh,
    Valid { params: Arc<Params>, identity: u64 },
    Failed(CaptureFailure),
    Blocked,
}
#[cfg(feature = "test-support")]
impl Publication {
    pub fn observe(&self) -> Result<Observation, Error> {
        let state = self.state.lock().map_err(|_| Error::Poisoned)?;
        let pending = match &state.pending {
            Pending::Uninitialized => ObservedPending::Uninitialized,
            Pending::Capturing { .. } => ObservedPending::Capturing,
            Pending::AwaitingWrites => ObservedPending::AwaitingWrites,
            Pending::NeedsRefresh => ObservedPending::NeedsRefresh,
            Pending::Valid { params, identity } => ObservedPending::Valid {
                params: Arc::clone(params),
                identity: *identity,
            },
            Pending::Failed { failure, .. } => ObservedPending::Failed(failure.clone()),
            Pending::Blocked => ObservedPending::Blocked,
        };
        Ok(Observation {
            pending,
            published: state.published.as_ref().map(Published::admission),
            issued: state.issued,
            write_revision: state.write_revision,
            outstanding_writes: state.outstanding_writes,
        })
    }
}

/// One registered physical host write. Drop records abandonment without calling
/// the host, allocating text, refreshing controls or recovering a poisoned mutex.
pub struct HostWrite {
    publication: Option<Arc<Publication>>,
}
impl HostWrite {
    pub fn finish(mut self, outcome: Result<(), CaptureFailure>) -> Result<WriteEnd, Error> {
        self.publication
            .take()
            .expect("live host write")
            .finish_write(outcome)
    }
}
impl Drop for HostWrite {
    fn drop(&mut self) {
        if let Some(publication) = self.publication.take() {
            let _ = publication.finish_write(Err(CaptureFailure::new(
                CaptureFailureKind::Host,
                "registered host write abandoned",
            )));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use super::*;
    use crate::film;
    use crate::recipe::{self, PreparedRoute};
    use crate::recipe::test_support::{assets, output_boundary_weak, params};
    use crate::route::ScanRoute;
    use crate::test_cases::RowCases;

    enum TestInput {
        Valid { params: Arc<Params>, identity: u64 },
        InvalidSnapshotControls { diagnostic: Arc<str> },
        Capturing,
    }
    impl Publication {
        // Existing twelve acceptance cases now complete real capture operations.
        fn stage(&self, input: TestInput) -> Result<(), Error> {
            let CaptureStart::Started { capture } = self.begin_capture(CaptureMode::Change)? else {
                unreachable!()
            };
            match input {
                TestInput::Valid { params, identity } => {
                    let SnapshotStart::Ready { write_revision } = self.begin_snapshot(capture)?
                    else {
                        unreachable!()
                    };
                    assert_eq!(
                        self.install_shared(capture, write_revision, params, identity)?,
                        Update::Applied
                    );
                }
                TestInput::InvalidSnapshotControls { diagnostic } => {
                    assert_eq!(
                        self.invalidate(
                            capture,
                            CaptureFailure::new(CaptureFailureKind::InvalidControls, &diagnostic)
                        )?,
                        Update::Applied
                    );
                }
                TestInput::Capturing => {}
            }
            Ok(())
        }
        fn admit_state(
            &self,
            prepare: impl FnMut(&Params) -> Result<RenderState, PrepareError>,
        ) -> Result<Arc<RenderState>, Error> {
            self.admit(prepare).map(Admission::into_state)
        }
    }

    const DIAGNOSTIC: &str = "InvalidSnapshotControls field=camera_film_format_long_edge_mm";
    const RENDEZVOUS: Duration = Duration::from_secs(10);

    thread_local! {
        static REBUILD_WAITER: Cell<Option<mpsc::SyncSender<()>>> = const { Cell::new(None) };
    }

    // One-shot observation inside admission: the second caller must encounter
    // the real gate while the first builder is still paused, before its release.
    pub(super) fn observe_rebuild_contention(rebuild: &Mutex<()>) {
        if let Some(waiting) = REBUILD_WAITER.take() {
            assert!(matches!(
                rebuild.try_lock(),
                Err(std::sync::TryLockError::WouldBlock)
            ));
            waiting.send(()).expect("contention observer disconnected");
        }
    }

    fn valid(identity: u64, params: Params) -> TestInput {
        TestInput::Valid {
            params: Arc::new(params),
            identity,
        }
    }

    fn direct() -> Params {
        params(ScanRoute::NegativeDirect, 1)
    }

    fn invalid() -> TestInput {
        TestInput::InvalidSnapshotControls {
            diagnostic: DIAGNOSTIC.into(),
        }
    }

    fn assert_invalid(result: Result<Arc<RenderState>, Error>) {
        assert!(
            matches!(result, Err(Error::CaptureFailed(failure)) if failure.kind() == CaptureFailureKind::InvalidControls && failure.diagnostic() == DIAGNOSTIC)
        );
    }

    // Small exact reads observe real retained arrays without copying prepared tables.
    fn backing_words(state: &RenderState) -> [u32; 4] {
        [
            state.development().normalized_curves_cmy()[0][0].to_bits(),
            state.exposure_tables().weighted_xyz()[1][40].to_bits(),
            state.scanner().tables().weighted_xyz()[2][50].to_bits(),
            state.scanner().gamut().boundary().unwrap().cmax()[100].to_bits(),
        ]
    }

    #[test]
    fn uninitialized_does_not_build() {
        let publication = Publication::new();
        assert!(matches!(
            publication.admit_state(|_| panic!("uninitialized input must not build")),
            Err(Error::Uninitialized)
        ));
        assert!(publication.state.lock().unwrap().published.is_none());
    }

    #[test]
    fn matching_identity_reuses_state() {
        let assets = assets();
        let publication = Publication::new();
        publication.stage(valid(11, direct())).unwrap();
        let mut builds = 0;
        let first = publication
            .admit_state(|input| {
                builds += 1;
                assert_eq!(builds, 1, "equivalent pending input must not rebuild");
                // Equivalent input after an intervening update still admits this build.
                publication
                    .stage(valid(12, params(ScanRoute::NegativePrint, 1)))
                    .unwrap();
                publication.stage(valid(11, direct())).unwrap();
                recipe::prepare(&assets, input)
            })
            .unwrap();
        assert_eq!(builds, 1);
        let curves = first.development().normalized_curves_cmy().as_ptr();
        let tables = first.scanner().tables().weighted_xyz().as_ptr();
        let boundary = first.scanner().gamut().boundary().unwrap().cmax().as_ptr();
        let words = backing_words(&first);
        publication.stage(valid(11, direct())).unwrap();
        let same = publication
            .admit_state(|_| panic!("same input rebuilt"))
            .unwrap();
        publication
            .stage(valid(12, params(ScanRoute::PositivePrint, 1)))
            .unwrap();
        publication.stage(valid(11, direct())).unwrap();
        let back = publication
            .admit_state(|_| panic!("equivalent input rebuilt"))
            .unwrap();
        assert!(Arc::ptr_eq(&first, &same));
        assert!(Arc::ptr_eq(&first, &back));
        assert_eq!(back.development().normalized_curves_cmy().as_ptr(), curves);
        assert_eq!(back.scanner().tables().weighted_xyz().as_ptr(), tables);
        assert_eq!(
            back.scanner().gamut().boundary().unwrap().cmax().as_ptr(),
            boundary
        );
        assert_eq!(backing_words(&back), words);

        // Print-only captured controls can differ while direct completed recipes agree.
        let mut changed = direct();
        changed.print_exposure = 2.0;
        publication.stage(valid(13, changed)).unwrap();
        let changed = publication
            .admit_state(|input| recipe::prepare(&assets, input))
            .unwrap();
        assert!(!Arc::ptr_eq(&first, &changed));
        assert_eq!(first.hash(), changed.hash());
    }

    #[cfg(feature = "grain-preview")]
    #[test]
    fn film_preview_geometry_retains_matching_tables_and_bounds() {
        let assets = assets();
        for route in [ScanRoute::NegativeDirect, ScanRoute::NegativePrint] {
            let publication = Publication::new();
            let film_params = |mm| {
                let mut captured = params(route, 1);
                captured.camera_film_format_long_edge_mm = mm;
                captured.film_grain = Some(film::grain::preview::Controls {
                    amount_ev: 3.0,
                    texel_pitch_um: 4.0,
                    chroma_fraction: 1.0,
                });
                captured
            };
            publication.stage(valid(1, film_params(36.0))).unwrap();
            let admit = |long_edge_px| {
                let frame = recipe::PreviewFrame::Raster(crate::film::grain::preview::Raster {
                    long_edge_px,
                    full_grid: None,
                });
                publication.admit_preview_frame(
                    publication.frame_input().unwrap(),
                    frame,
                    |params| recipe::prepare_for_frame(&assets, params, frame),
                )
            };
            let landscape = admit(4832).unwrap();
            let thumbnail = recipe::PreviewFrame::Thumbnail;
            let clean = publication
                .admit_preview_frame(publication.frame_input().unwrap(), thumbnail, |params| {
                    recipe::prepare_for_frame(&assets, params, thumbnail)
                })
                .unwrap();
            assert!(clean.film_grain().is_none() && !clean.grain().active());
            assert!(Arc::ptr_eq(&landscape, &admit(4832).unwrap()));
            let portrait = admit(7418).unwrap();
            let landscape_grain = landscape.film_grain().unwrap();
            let portrait_grain = portrait.film_grain().unwrap();
            assert_eq!(
                landscape_grain.table().footprint.to_bits(),
                1.8625828_f32.to_bits()
            );
            assert_eq!(
                portrait_grain.table().footprint.to_bits(),
                1.2132651_f32.to_bits()
            );
            assert_ne!(
                landscape.film_v1_bounds().unwrap().hash(),
                portrait.film_v1_bounds().unwrap().hash()
            );
            assert_eq!(landscape.scanner_hash(), portrait.scanner_hash());
            assert_ne!(landscape.hash(), portrait.hash());
            assert!(matches!(
                admit(36001),
                Err(Error::Preparation(PrepareError::FilmGrainFootprint { .. }))
            ));
            assert!(Arc::ptr_eq(&portrait, &admit(7418).unwrap()));
            publication.stage(valid(2, film_params(11.66))).unwrap();
            let current = admit(4832).unwrap();
            assert_eq!(
                current.film_grain().unwrap().table().footprint,
                (f64::from(11.66_f32) * 1000.0 / 4832.0 / 4.0) as f32
            );
            assert_ne!(
                landscape.film_v1_bounds().unwrap().hash(),
                current.film_v1_bounds().unwrap().hash()
            );
            assert_eq!(landscape_grain.table().footprint, 1.8625828);
            assert_eq!(landscape_grain.raster().long_edge_px, 4832);
        }
    }

    #[cfg(feature = "grain-preview")]
    #[test]
    fn film_thumbnail_validates_authored_selection_without_publishing() {
        use film::grain::preview::{Controls, Raster};
        use recipe::{ControlError, PreviewFrame};

        let assets = assets();
        let controls = Controls {
            amount_ev: 0.0,
            texel_pitch_um: 1.0,
            chroma_fraction: 1.0,
        };
        let raster = Raster {
            long_edge_px: 4832,
            full_grid: None,
        };
        let timeline = PreviewFrame::Raster(raster);
        let thumbnail = PreviewFrame::Thumbnail;
        for (route, positive_route) in [
            (ScanRoute::NegativeDirect, ScanRoute::PositiveDirect),
            (ScanRoute::NegativePrint, ScanRoute::PositivePrint),
        ] {
            let publication = Publication::new();
            let authored = || {
                let mut captured = params(route, 1);
                captured.grain.active = true;
                captured.film_grain = Some(controls);
                captured
            };
            let admit = |frame| {
                publication.admit_preview_frame(
                    publication.frame_input().unwrap(),
                    frame,
                    |params| recipe::prepare_for_frame(&assets, params, frame),
                )
            };
            publication.stage(valid(1, authored())).unwrap();
            let held = admit(timeline).unwrap();
            assert!(held.film_grain().unwrap().raster() == raster);
            assert!(!held.grain().active());
            let words = backing_words(&held);
            let assert_timeline_published = || {
                let state = publication.state.lock().unwrap();
                let published = state.published.as_ref().unwrap();
                assert_eq!((published.identity, published.build), (1, 1));
                assert!(Arc::ptr_eq(&published.state, &held));
                assert!(held.film_grain().unwrap().raster() == raster);
                assert_eq!(backing_words(&held), words);
            };

            let clean = admit(thumbnail).unwrap();
            let off = recipe::prepare(&assets, &params(route, 1)).unwrap();
            assert_eq!(clean.development().route(), route);
            assert!(clean.film_grain().is_none() && !clean.grain().active());
            assert!(clean.film_v1_bounds().is_none() && off.film_v1_bounds().is_none());
            assert!(held.film_v1_bounds().is_some());
            assert_timeline_published();
            assert!(Arc::ptr_eq(&held, &admit(timeline).unwrap()));

            let invalid = [
                (
                    "stock",
                    Params {
                        film_profile_key: "kodak_gold_200".into(),
                        ..authored()
                    },
                ),
                (
                    "amount",
                    Params {
                        film_grain: Some(Controls {
                            amount_ev: 4.0,
                            ..controls
                        }),
                        ..authored()
                    },
                ),
                (
                    "pitch",
                    Params {
                        film_grain: Some(Controls {
                            texel_pitch_um: 0.25,
                            ..controls
                        }),
                        ..authored()
                    },
                ),
                (
                    "chroma",
                    Params {
                        film_grain: Some(Controls {
                            chroma_fraction: -0.1,
                            ..controls
                        }),
                        ..authored()
                    },
                ),
                (
                    "format",
                    Params {
                        camera_film_format_long_edge_mm: 7.0,
                        ..authored()
                    },
                ),
                (
                    "route",
                    Params {
                        scan_route: positive_route,
                        ..authored()
                    },
                ),
            ];
            for (index, (field, captured)) in invalid.into_iter().enumerate() {
                let identity = index as u64 + 2;
                publication.stage(valid(identity, captured)).unwrap();
                assert_eq!(publication.frame_input().unwrap().identity, identity);
                for (name, frame) in [("thumbnail", thumbnail), ("timeline", timeline)] {
                    assert!(
                        matches!(
                            admit(frame),
                            Err(Error::Preparation(PrepareError::Control(
                                ControlError::FilmGrain
                            )))
                        ),
                        "{route:?} {name}: expected Film control rejection for {field}"
                    );
                    assert_timeline_published();
                }
            }
            publication.stage(valid(1, authored())).unwrap();
            assert!(Arc::ptr_eq(&held, &admit(timeline).unwrap()));
            assert_timeline_published();

            publication
                .stage(valid(
                    8,
                    Params {
                        film_profile_key: "kodak_gold_200".into(),
                        ..params(route, 1)
                    },
                ))
                .unwrap();
            for frame in [timeline, thumbnail] {
                let gold = admit(frame).unwrap();
                assert_eq!(gold.development().route(), route);
                assert!(gold.film_grain().is_none() && !gold.grain().active());
            }
        }
    }

    #[cfg(feature = "grain-preview")]
    #[test]
    fn film_proxy_publication_uses_full_grid_and_encloses_its_densities() {
        use crate::film::grain::preview::{ProxyGrid, Raster, Sampling};
        let assets = assets();
        for route in [ScanRoute::NegativeDirect, ScanRoute::NegativePrint] {
            let publication = Publication::new();
            let mut captured = params(route, 1);
            captured.camera_film_format_long_edge_mm = 36.0;
            captured.film_grain = Some(film::grain::preview::Controls {
                amount_ev: 0.0,
                texel_pitch_um: 0.5,
                chroma_fraction: 0.24,
            });
            publication.stage(valid(1, captured)).unwrap();
            let admit = |long_edge_px, full_grid| {
                let frame = recipe::PreviewFrame::Raster(Raster {
                    long_edge_px,
                    full_grid,
                });
                publication
                    .admit_preview_frame(publication.frame_input().unwrap(), frame, |params| {
                        recipe::prepare_for_frame(&assets, params, frame)
                    })
                    .unwrap()
            };
            let full = admit(4832, None);
            let grid = ProxyGrid {
                extent: [4832.0, 2016.0],
                step: [8.0; 2],
                origin: [0.0; 2],
                size: [604, 252],
            };
            let eighth = admit(604, Some(grid));
            assert!(std::ptr::eq(
                full.film_grain().unwrap().table(),
                eighth.film_grain().unwrap().table(),
            ));
            assert_eq!(
                full.film_v1_bounds().unwrap().hash(),
                eighth.film_v1_bounds().unwrap().hash()
            );
            assert_eq!(full.scanner_hash(), eighth.scanner_hash());
            assert!(Arc::ptr_eq(&eighth, &admit(604, Some(grid))));
            let other = admit(
                604,
                Some(ProxyGrid {
                    extent: [2416.0, 1008.0],
                    step: [4.0; 2],
                    ..grid
                }),
            );
            assert!(!Arc::ptr_eq(&eighth, &other));
            assert_ne!(
                eighth.film_grain().unwrap().table().footprint,
                other.film_grain().unwrap().table().footprint,
            );
            assert_eq!(eighth.film_grain().unwrap().footprint(), 72000.0 / 604.0);
            assert_eq!(admit(4832, None).hash(), full.hash());

            publication
                .stage(valid(
                    2,
                    Params {
                        camera_film_format_long_edge_mm: 120.0,
                        film_grain: Some(film::grain::preview::Controls {
                            amount_ev: 0.0,
                            texel_pitch_um: 0.5,
                            chroma_fraction: 0.24,
                        }),
                        ..params(route, 1)
                    },
                ))
                .unwrap();
            let native = admit(4832, None);
            let same_pitch = admit(
                4832,
                Some(ProxyGrid {
                    extent: [9664.0, 4032.0],
                    step: [2.0; 2],
                    origin: [0.0; 2],
                    size: [4832, 2016],
                }),
            );
            let native_grain = native.film_grain().unwrap();
            let proxy_grain = same_pitch.film_grain().unwrap();
            assert_eq!(native_grain.footprint(), proxy_grain.footprint());
            assert!(matches!(
                native_grain.sampling(),
                Sampling::Calibrated([0, 0, 0, 0])
            ));
            assert!(matches!(proxy_grain.sampling(), Sampling::Integrated(_)));
            assert_eq!(
                native_grain.table().footprint,
                proxy_grain.table().footprint * 2.0
            );
            assert_ne!(
                native.film_v1_bounds().unwrap().hash(),
                same_pitch.film_v1_bounds().unwrap().hash()
            );
            assert_eq!(admit(4832, None).hash(), native.hash());
        }
    }

    #[test]
    fn zero_identity_is_valid() {
        let assets = assets();
        let publication = Publication::new();
        publication.stage(valid(0, direct())).unwrap();
        let mut builds = 0;
        let first = publication
            .admit_state(|input| {
                builds += 1;
                recipe::prepare(&assets, input)
            })
            .unwrap();
        let second = publication
            .admit_state(|_| panic!("zero identity rebuilt"))
            .unwrap();
        assert_eq!(builds, 1);
        assert!(Arc::ptr_eq(&first, &second));
        publication.stage(invalid()).unwrap();
        assert_invalid(publication.admit_state(|_| panic!("invalid input built")));
        publication.stage(TestInput::Capturing).unwrap();
        assert!(matches!(
            publication.admit_state(|_| panic!("uninitialized input built")),
            Err(Error::Capturing)
        ));
    }

    #[test]
    fn route_changes_publish_complete_states() {
        let assets = assets();
        let publication = Publication::new();
        let routes = [
            ("negative-direct", ScanRoute::NegativeDirect),
            ("negative-print", ScanRoute::NegativePrint),
            ("positive-direct", ScanRoute::PositiveDirect),
            ("positive-print", ScanRoute::PositivePrint),
        ];
        let mut cases = RowCases::new("publication-routes", routes.map(|(id, _)| id));
        let mut holds = Vec::new();
        for (index, (id, route)) in routes.into_iter().enumerate() {
            let case = cases.case(id);
            publication
                .stage(valid(index as u64, params(route, 1)))
                .unwrap();
            let state = publication
                .admit_state(|input| recipe::prepare(&assets, input))
                .unwrap();
            assert_eq!(state.development().route(), route);
            assert_ne!(state.hash(), 0);
            assert_ne!(state.reference().hash(), 0);
            assert_ne!(state.scanner().hash(), 0);
            assert_ne!(state.scanner_hash(), 0);
            assert!(!state.development().normalized_curves_cmy().is_empty());
            match (route, state.route()) {
                (
                    ScanRoute::NegativeDirect | ScanRoute::PositiveDirect,
                    PreparedRoute::Direct(_),
                ) => {}
                (
                    ScanRoute::NegativePrint | ScanRoute::PositivePrint,
                    PreparedRoute::Print(print),
                ) => {
                    assert_ne!(print.balance().hash(), 0);
                    assert_eq!(
                        print.curves().totals().len(),
                        print.profile().tables().interpolation_log_exposure().len()
                    );
                }
                _ => panic!("wrong completed route"),
            }
            let words = backing_words(&state);
            holds.push((state, route, words));
            for (held, route, words) in &holds {
                assert_eq!(held.development().route(), *route);
                assert_eq!(backing_words(held), *words);
            }
            case.passed();
        }
        cases.finish();
    }

    #[test]
    fn invalid_controls_block_admission() {
        let assets = assets();
        let publication = Publication::new();
        publication.stage(valid(1, direct())).unwrap();
        let held = publication
            .admit_state(|input| recipe::prepare(&assets, input))
            .unwrap();
        let words = backing_words(&held);
        publication.stage(invalid()).unwrap();
        assert_invalid(publication.admit_state(|_| panic!("invalid input built")));
        assert_eq!(backing_words(&held), words);
        publication.stage(valid(1, direct())).unwrap();
        assert!(Arc::ptr_eq(
            &held,
            &publication
                .admit_state(|_| panic!("recovery rebuilt equivalent input"))
                .unwrap()
        ));
        publication
            .stage(valid(2, params(ScanRoute::NegativePrint, 1)))
            .unwrap();
        let recovered = publication
            .admit_state(|input| recipe::prepare(&assets, input))
            .unwrap();
        assert_eq!(recovered.development().route(), ScanRoute::NegativePrint);
        assert!(!Arc::ptr_eq(&held, &recovered));
    }

    #[test]
    fn failed_preparation_blocks_old_state() {
        let assets = assets();
        let publication = Publication::new();
        publication.stage(valid(1, direct())).unwrap();
        let held = publication
            .admit_state(|input| recipe::prepare(&assets, input))
            .unwrap();
        let mut bad = direct();
        bad.film_gamma_factor = 0.0;
        publication.stage(valid(2, bad)).unwrap();
        assert!(matches!(
            publication.admit_state(|input| recipe::prepare(&assets, input)),
            Err(Error::Preparation(PrepareError::Development(
                film::Error::InvalidGamma
            )))
        ));
        assert!(Arc::ptr_eq(
            &held,
            &publication
                .state
                .lock()
                .unwrap()
                .published
                .as_ref()
                .unwrap()
                .state
        ));
        publication.stage(valid(3, direct())).unwrap();
        let recovered = publication
            .admit_state(|input| recipe::prepare(&assets, input))
            .unwrap();
        assert!(!Arc::ptr_eq(&held, &recovered));
        assert_eq!(backing_words(&held), backing_words(&recovered));
    }

    #[test]
    fn typed_panics_block_current_and_obsolete_publications() {
        let mut cases = RowCases::new("publication-typed-panic", ["current", "obsolete"]);
        for scenario in ["current", "obsolete"] {
            let case = cases.case(scenario);
            let assets = assets();
            let publication = Publication::new();
            publication.stage(valid(1, direct())).unwrap();
            let held = publication
                .admit_state(|input| recipe::prepare(&assets, input))
                .unwrap();
            let words = backing_words(&held);
            publication
                .stage(valid(2, params(ScanRoute::PositivePrint, 1)))
                .unwrap();

            let result = publication.admit(|_| {
                if scenario == "obsolete" {
                    publication
                        .stage(valid(3, params(ScanRoute::NegativePrint, 1)))
                        .unwrap();
                }
                Err(PrepareError::Input(film::input::Error::Panic(
                    film::input::Operation::Sensitivity,
                )))
            });
            let Err(Error::Preparation(error)) = result else {
                panic!("typed leaf panic must retain its producer error")
            };
            assert!(error.is_panic());
            assert_eq!(error.to_string(), "asset boundary panic");
            let state = publication.state.lock().unwrap();
            assert!(matches!(state.pending, Pending::Blocked));
            assert_eq!(state.published.as_ref().unwrap().identity, 1);
            assert_eq!(state.published.as_ref().unwrap().build, 1);
            assert!(Arc::ptr_eq(&state.published.as_ref().unwrap().state, &held));
            drop(state);
            assert_eq!(backing_words(&held), words);
            assert!(matches!(
                publication.admit(|_| panic!("blocked publication prepared")),
                Err(Error::Poisoned)
            ));
            assert!(matches!(
                publication.begin_capture(CaptureMode::Change),
                Err(Error::Poisoned)
            ));
            case.passed();
        }
        cases.finish();
    }

    #[cfg(feature = "test-support")]
    #[test]
    fn current_contents_reservation_failure_preserves_and_reuses_last_build() {
        let assets = assets();
        let publication = Publication::new();
        publication.stage(valid(1, direct())).unwrap();
        let held = publication
            .admit_state(|input| recipe::prepare(&assets, input))
            .unwrap();
        let held_weak = Arc::downgrade(&held);
        let boundary_weak = output_boundary_weak(&held);
        let old_words = backing_words(&held);

        publication
            .stage(valid(2, params(ScanRoute::PositivePrint, 1)))
            .unwrap();
        let result = publication.admit_state(|input| {
            recipe::testing::arm_contents_reservation_failure();
            recipe::prepare(&assets, input)
        });
        assert!(matches!(
            result,
            Err(Error::Preparation(PrepareError::ContentsAllocation(_)))
        ));
        assert!(recipe::testing::contents_reservation_failure_consumed());
        let state = publication.state.lock().unwrap();
        assert!(matches!(state.pending, Pending::Valid { identity: 2, .. }));
        assert_eq!(state.published.as_ref().unwrap().identity, 1);
        assert_eq!(state.published.as_ref().unwrap().build, 1);
        assert!(Arc::ptr_eq(&state.published.as_ref().unwrap().state, &held));
        drop(state);
        assert_eq!(backing_words(&held), old_words);

        publication.stage(valid(1, direct())).unwrap();
        let reused = publication
            .admit_state(|_| panic!("return to A must reuse its committed state"))
            .unwrap();
        assert!(Arc::ptr_eq(&held, &reused));

        publication
            .stage(valid(2, params(ScanRoute::PositivePrint, 1)))
            .unwrap();
        let replacement = publication
            .admit_state(|input| recipe::prepare(&assets, input))
            .unwrap();
        assert_eq!(
            publication
                .state
                .lock()
                .unwrap()
                .published
                .as_ref()
                .unwrap()
                .build,
            2
        );
        assert_eq!(backing_words(&held), old_words);
        assets.release_cached_payloads().unwrap();
        drop(assets);
        drop(replacement);
        drop(publication);
        drop(held);
        drop(reused);
        assert!(held_weak.upgrade().is_none());
        assert!(boundary_weak.upgrade().is_none());
    }

    #[cfg(feature = "test-support")]
    #[test]
    fn obsolete_contents_reservation_failure_rechecks_then_commits_new_input() {
        let assets = assets();
        let publication = Publication::new();
        publication.stage(valid(1, direct())).unwrap();
        let held = publication
            .admit_state(|input| recipe::prepare(&assets, input))
            .unwrap();
        let old_words = backing_words(&held);
        publication
            .stage(valid(2, params(ScanRoute::PositivePrint, 1)))
            .unwrap();

        let mut attempts = 0;
        let admitted = publication
            .admit_state(|input| {
                attempts += 1;
                if attempts == 1 {
                    recipe::testing::arm_contents_reservation_failure();
                }
                let result = recipe::prepare(&assets, input);
                if attempts == 1 {
                    assert!(matches!(&result, Err(PrepareError::ContentsAllocation(_))));
                    publication
                        .stage(valid(3, params(ScanRoute::NegativePrint, 1)))
                        .unwrap();
                }
                result
            })
            .unwrap();
        assert_eq!(attempts, 2);
        assert!(recipe::testing::contents_reservation_failure_consumed());
        assert_eq!(admitted.development().route(), ScanRoute::NegativePrint);
        assert_eq!(backing_words(&held), old_words);
        let observed = publication.observe().unwrap();
        assert_eq!(observed.published.as_ref().unwrap().input_identity(), 3);
        assert_eq!(observed.published.as_ref().unwrap().build(), 2);
        assert!(Arc::ptr_eq(
            observed.published.as_ref().unwrap().shared_state(),
            &admitted
        ));
    }

    #[test]
    fn obsolete_success_is_not_published() {
        let assets = assets();
        let publication = Publication::new();
        let input = Arc::new(direct());
        let input_weak = Arc::downgrade(&input);
        publication
            .stage(TestInput::Valid {
                params: input,
                identity: 1,
            })
            .unwrap();
        let mut discarded = None;
        let mut builds = 0;
        let admitted = publication
            .admit_state(|input| {
                builds += 1;
                assert!(
                    builds <= 2,
                    "obsolete success must retry only the current input"
                );
                let state = recipe::prepare(&assets, input)?;
                if builds == 1 {
                    discarded = Some(output_boundary_weak(&state));
                    assets.release_cached_payloads().unwrap();
                    let mut next = params(ScanRoute::PositiveDirect, 1);
                    next.output_gamut_compression_enabled = 0;
                    publication.stage(valid(2, next)).unwrap();
                    assert_eq!(input.scan_route, ScanRoute::NegativeDirect);
                    assert!(input_weak.upgrade().is_some());
                } else {
                    assert!(input_weak.upgrade().is_none());
                    assert!(discarded.as_ref().unwrap().upgrade().is_none());
                }
                Ok(state)
            })
            .unwrap();
        assert_eq!(builds, 2);
        assert_eq!(admitted.development().route(), ScanRoute::PositiveDirect);
        assert!(discarded.unwrap().upgrade().is_none());
        assert_eq!(Arc::strong_count(&admitted), 2);
        assert!(Arc::ptr_eq(
            &admitted,
            &publication
                .admit_state(|_| panic!("current state rebuilt"))
                .unwrap()
        ));
    }

    #[test]
    fn obsolete_failure_rechecks_pending() {
        let mut cases = RowCases::new(
            "publication-obsolete-failure",
            ["valid", "invalid", "capturing"],
        );
        for next in ["valid", "invalid", "capturing"] {
            let case = cases.case(next);
            let assets = assets();
            let publication = Publication::new();
            publication.stage(valid(1, direct())).unwrap();
            let held = publication
                .admit_state(|input| recipe::prepare(&assets, input))
                .unwrap();
            let mut bad = direct();
            bad.film_gamma_factor = 0.0;
            publication.stage(valid(2, bad)).unwrap();
            let mut builds = 0;
            let result = publication.admit_state(|input| {
                builds += 1;
                assert!(
                    builds <= 2,
                    "obsolete failure must retry only the current input"
                );
                let result = recipe::prepare(&assets, input);
                if builds == 1 {
                    assert!(matches!(
                        &result,
                        Err(PrepareError::Development(film::Error::InvalidGamma))
                    ));
                    publication
                        .stage(match next {
                            "valid" => valid(3, direct()),
                            "invalid" => invalid(),
                            "capturing" => TestInput::Capturing,
                            _ => unreachable!(),
                        })
                        .unwrap();
                }
                result
            });
            match next {
                "valid" => {
                    assert_eq!(builds, 2);
                    let current = result.unwrap();
                    assert!(!Arc::ptr_eq(&held, &current));
                    assert_eq!(backing_words(&held), backing_words(&current));
                }
                "invalid" => {
                    assert_eq!(builds, 1);
                    assert_invalid(result);
                }
                "capturing" => {
                    assert_eq!(builds, 1);
                    assert!(matches!(result, Err(Error::Capturing)));
                }
                _ => unreachable!(),
            }
            case.passed();
        }
        cases.finish();
    }

    #[test]
    fn invalidated_build_is_not_published() {
        let assets = assets();
        let publication = Publication::new();
        publication.stage(valid(1, direct())).unwrap();
        let held = publication
            .admit_state(|input| recipe::prepare(&assets, input))
            .unwrap();
        publication
            .stage(valid(2, params(ScanRoute::NegativePrint, 1)))
            .unwrap();
        let mut builds = 0;
        assert_invalid(publication.admit_state(|input| {
            builds += 1;
            assert_eq!(builds, 1, "invalid input must stop preparation retries");
            let state = recipe::prepare(&assets, input)?;
            publication.stage(invalid()).unwrap();
            Ok(state)
        }));
        assert_eq!(builds, 1);
        let state = publication.state.lock().unwrap();
        let saved = state.published.as_ref().unwrap();
        assert_eq!(saved.identity, 1);
        assert!(Arc::ptr_eq(&saved.state, &held));
    }

    #[test]
    fn admitted_holds_survive_replacement_and_cache_release() {
        let assets = assets();
        let publication = Publication::new();
        publication.stage(valid(1, direct())).unwrap();
        let held = publication
            .admit_state(|input| recipe::prepare(&assets, input))
            .unwrap();
        let weak = Arc::downgrade(&held);
        let boundary = output_boundary_weak(&held);
        let words = backing_words(&held);
        let address = held.scanner().gamut().boundary().unwrap().cmax().as_ptr();
        publication.stage(invalid()).unwrap();
        assert_invalid(publication.admit_state(|_| panic!("invalid input built")));
        publication
            .stage(valid(2, params(ScanRoute::PositivePrint, 1)))
            .unwrap();
        let replacement = publication
            .admit_state(|input| recipe::prepare(&assets, input))
            .unwrap();
        let replacement_weak = Arc::downgrade(&replacement);
        assets.release_cached_payloads().unwrap();
        drop(assets);
        drop(publication);
        drop(replacement);
        assert!(replacement_weak.upgrade().is_none());
        assert!(weak.upgrade().is_some());
        assert!(boundary.upgrade().is_some());
        assert_eq!(backing_words(&held), words);
        assert_eq!(
            held.scanner().gamut().boundary().unwrap().cmax().as_ptr(),
            address
        );
        assert_eq!(Arc::strong_count(&held), 1);
        drop(held);
        assert!(weak.upgrade().is_none());
        assert!(boundary.upgrade().is_none());
    }

    #[test]
    fn concurrent_admission_builds_once() {
        let assets = assets();
        let publication = Publication::new();
        publication.stage(valid(1, direct())).unwrap();
        let builds = AtomicUsize::new(0);
        let (started_tx, started_rx) = mpsc::sync_channel(1);
        let (release_tx, release_rx) = mpsc::sync_channel(1);
        let (waiting_tx, waiting_rx) = mpsc::sync_channel(1);
        let (first, second) = thread::scope(|scope| {
            let publication = &publication;
            let assets = &assets;
            let builds = &builds;
            let first = scope.spawn(move || {
                publication.admit_state(|input| {
                    builds.fetch_add(1, Ordering::SeqCst);
                    started_tx.send(()).unwrap();
                    release_rx
                        .recv_timeout(RENDEZVOUS)
                        .expect("first builder was not released");
                    recipe::prepare(assets, input)
                })
            });
            started_rx
                .recv_timeout(RENDEZVOUS)
                .expect("first builder did not start");
            let second = scope.spawn(move || {
                REBUILD_WAITER.set(Some(waiting_tx));
                publication.admit_state(|input| {
                    builds.fetch_add(1, Ordering::SeqCst);
                    recipe::prepare(assets, input)
                })
            });
            waiting_rx
                .recv_timeout(RENDEZVOUS)
                .expect("second admission did not encounter the held rebuild gate");
            publication
                .stage(valid(2, params(ScanRoute::NegativePrint, 1)))
                .unwrap();
            publication.stage(valid(1, direct())).unwrap();
            release_tx.send(()).unwrap();
            let first = first.join();
            let second = second.join();
            (first.unwrap().unwrap(), second.unwrap().unwrap())
        });
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(backing_words(&first), backing_words(&second));
    }

    #[test]
    fn poisoned_state_blocks_admission() {
        let mut cases = RowCases::new("publication-poison", ["state", "rebuild"]);
        for target in ["state", "rebuild"] {
            let case = cases.case(target);
            let assets = assets();
            let publication = Publication::new();
            publication.stage(valid(1, direct())).unwrap();
            let held = publication
                .admit_state(|input| recipe::prepare(&assets, input))
                .unwrap();
            let words = backing_words(&held);
            let panic = catch_unwind(AssertUnwindSafe(|| match target {
                "state" => {
                    let _state = publication.state.lock().unwrap();
                    panic!("intentional state poison");
                }
                "rebuild" => {
                    publication.stage(valid(2, direct())).unwrap();
                    let _ = publication.admit_state(|_| {
                        publication.stage(valid(1, direct())).unwrap();
                        panic!("intentional rebuild poison");
                    });
                }
                _ => unreachable!(),
            }));
            assert!(panic.is_err());
            assert!(matches!(
                publication.admit_state(|_| panic!("poisoned admission built")),
                Err(Error::Poisoned)
            ));
            if target == "state" {
                assert!(matches!(
                    publication.stage(valid(3, direct())),
                    Err(Error::Poisoned)
                ));
            } else {
                assert!(publication.rebuild.is_poisoned());
                let state = publication.state.lock().unwrap();
                let saved = state.published.as_ref().unwrap();
                assert_eq!(saved.identity, 1);
                assert!(Arc::ptr_eq(&saved.state, &held));
            }
            assert_eq!(backing_words(&held), words);
            case.passed();
        }
        cases.finish();
    }
    fn begin(publication: &Publication, mode: CaptureMode) -> u64 {
        let CaptureStart::Started { capture } = publication.begin_capture(mode).unwrap() else {
            panic!("capture must start")
        };
        capture
    }
    fn reading(publication: &Publication, capture: u64) -> u64 {
        let SnapshotStart::Ready { write_revision } = publication.begin_snapshot(capture).unwrap()
        else {
            panic!("snapshot must be readable")
        };
        write_revision
    }
    fn write(publication: &Arc<Publication>, capture: u64) -> HostWrite {
        let HostWriteStart::Started(write) = publication.begin_host_write(capture).unwrap() else {
            panic!("host write must be granted")
        };
        write
    }

    #[test]
    fn frame_input_survives_new_change_without_replacing_current_cache() {
        let assets = assets();
        let publication = Publication::new();
        let first = begin(&publication, CaptureMode::Change);
        let revision = reading(&publication, first);
        let FrameInstall::Applied(input) = publication
            .install_frame(first, revision, direct(), 41)
            .unwrap()
        else {
            panic!("frame input must be installed")
        };
        let calls = AtomicUsize::new(0);
        let frame = publication
            .admit_frame(input, |params| {
                calls.fetch_add(1, Ordering::SeqCst);
                let newer = begin(&publication, CaptureMode::Change);
                assert!(matches!(publication.frame_input(), Err(Error::Capturing)));
                assert!(newer > first);
                recipe::prepare(&assets, params)
            })
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(frame.development().route(), ScanRoute::NegativeDirect);
        {
            let state = publication.state.lock().unwrap();
            assert!(state.published.is_none());
            assert_eq!(state.build, 0);
        }
        let newer = publication.state.lock().unwrap().issued;
        let revision = reading(&publication, newer);
        publication
            .install(newer, revision, params(ScanRoute::NegativePrint, 1), 42)
            .unwrap();
        let newer = publication
            .admit_frame(publication.frame_input().unwrap(), |params| {
                recipe::prepare(&assets, params)
            })
            .unwrap();
        assert_eq!(newer.development().route(), ScanRoute::NegativePrint);
        let state = publication.state.lock().unwrap();
        assert_eq!(state.published.as_ref().unwrap().identity, 42);
        assert_eq!(state.build, 1);
        assert!(!Arc::ptr_eq(&frame, &newer));
    }

    #[test]
    fn frame_input_failure_and_cache_identity_are_explicit() {
        let assets = assets();
        let publication = Publication::new();
        let ticket = begin(&publication, CaptureMode::Change);
        let revision = reading(&publication, ticket);
        publication.install(ticket, revision, direct(), 1).unwrap();
        let cached = publication
            .admit_frame(publication.frame_input().unwrap(), |params| {
                recipe::prepare(&assets, params)
            })
            .unwrap();
        let reused = publication
            .admit_frame(publication.frame_input().unwrap(), |_| {
                panic!("matching frame rebuilt")
            })
            .unwrap();
        assert!(Arc::ptr_eq(&cached, &reused));
        let ticket = begin(&publication, CaptureMode::Change);
        let revision = reading(&publication, ticket);
        let mut bad = direct();
        bad.film_gamma_factor = 0.0;
        let FrameInstall::Applied(input) =
            publication.install_frame(ticket, revision, bad, 2).unwrap()
        else {
            panic!("bad frame input not installed")
        };
        let newer = begin(&publication, CaptureMode::Change);
        assert!(matches!(
            publication.admit_frame(input, |params| recipe::prepare(&assets, params)),
            Err(Error::Preparation(PrepareError::Development(
                film::Error::InvalidGamma
            )))
        ));
        assert!(matches!(publication.frame_input(), Err(Error::Capturing)));
        let state = publication.state.lock().unwrap();
        assert_eq!(state.published.as_ref().unwrap().identity, 1);
        assert!(state.current(newer).unwrap());
    }

    #[test]
    fn frame_input_rejects_another_publication_before_cache_or_preparation() {
        let first = Publication::new();
        let second = Publication::new();
        let ticket = begin(&first, CaptureMode::Change);
        let revision = reading(&first, ticket);
        let FrameInstall::Applied(input) =
            first.install_frame(ticket, revision, direct(), 1).unwrap()
        else {
            panic!("frame input not installed")
        };
        assert!(matches!(
            second.admit_frame(input, |_| panic!("cross-owner preparation")),
            Err(Error::InvalidCaptureOperation)
        ));
        let state = second.state.lock().unwrap();
        assert!(state.published.is_none());
        assert_eq!(state.build, 0);
    }

    #[test]
    fn obsolete_frame_keeps_newer_cache_and_panic_blocks_publication() {
        let assets = assets();
        let publication = Publication::new();
        let old = begin(&publication, CaptureMode::Change);
        let revision = reading(&publication, old);
        let FrameInstall::Applied(input) = publication
            .install_frame(old, revision, direct(), 1)
            .unwrap()
        else {
            panic!("old input not installed")
        };
        let newer = begin(&publication, CaptureMode::Change);
        let revision = reading(&publication, newer);
        publication
            .install(newer, revision, params(ScanRoute::NegativePrint, 1), 2)
            .unwrap();
        let newer_state = publication
            .admit_frame(publication.frame_input().unwrap(), |params| {
                recipe::prepare(&assets, params)
            })
            .unwrap();
        let old_state = publication
            .admit_frame(input, |params| recipe::prepare(&assets, params))
            .unwrap();
        assert_eq!(old_state.development().route(), ScanRoute::NegativeDirect);
        assert_eq!(newer_state.development().route(), ScanRoute::NegativePrint);
        {
            let state = publication.state.lock().unwrap();
            assert_eq!(state.build, 1);
            assert_eq!(state.published.as_ref().unwrap().identity, 2);
            assert!(Arc::ptr_eq(
                &state.published.as_ref().unwrap().state,
                &newer_state
            ));
        }
        let current = begin(&publication, CaptureMode::Change);
        let revision = reading(&publication, current);
        publication.install(current, revision, direct(), 3).unwrap();
        let input = publication.frame_input().unwrap();
        begin(&publication, CaptureMode::Change);
        assert!(matches!(
            publication.admit_frame(input, |_| Err(PrepareError::Input(
                film::input::Error::Panic(film::input::Operation::Sensitivity)
            ))),
            Err(Error::Preparation(_))
        ));
        assert!(matches!(publication.frame_input(), Err(Error::Poisoned)));
    }

    #[test]
    fn frame_install_preserves_checks_and_action_identity() {
        let publication = Arc::new(Publication::new());
        assert!(matches!(
            publication.property_is_current(0),
            Err(Error::InvalidCaptureTicket)
        ));
        let action = begin(&publication, CaptureMode::Change);
        let revision = reading(&publication, action);
        assert!(matches!(
            publication.install_frame(action, revision + 1, direct(), 1),
            Err(Error::InvalidCaptureOperation)
        ));
        let reader = begin(&publication, CaptureMode::RenderRead);
        assert!(publication.property_is_current(action).unwrap());
        assert!(!publication.property_is_current(reader).unwrap());
        assert!(matches!(
            publication.install_frame(action, revision, direct(), 1),
            Ok(FrameInstall::Ignored)
        ));
        let reader_revision = reading(&publication, reader);
        assert!(matches!(
            publication.install_frame(reader, reader_revision, direct(), 2),
            Ok(FrameInstall::Applied(_))
        ));
        assert!(publication.property_is_current(action).unwrap());
        let next = begin(&publication, CaptureMode::Change);
        assert!(!publication.property_is_current(action).unwrap());
        let writer = write(&publication, next);
        let newest = begin(&publication, CaptureMode::Change);
        assert_eq!(
            publication.begin_snapshot(newest).unwrap(),
            SnapshotStart::Deferred
        );
        assert!(matches!(
            publication.install_frame(newest, 0, direct(), 3),
            Ok(FrameInstall::Ignored)
        ));
        writer
            .finish(Err(CaptureFailure::new(
                CaptureFailureKind::Host,
                "write failed",
            )))
            .unwrap();
        assert!(!publication.property_is_current(newest).unwrap());
        assert!(matches!(
            publication.frame_input(),
            Err(Error::CaptureFailed(_))
        ));

        let capture = begin(&publication, CaptureMode::Change);
        let revision = reading(&publication, capture);
        publication.state.lock().unwrap().write_revision += 1;
        assert!(matches!(
            publication.install_frame(capture, revision, direct(), 4),
            Ok(FrameInstall::Deferred)
        ));
        assert!(matches!(
            publication.frame_input(),
            Err(Error::RefreshRequired)
        ));
    }

    #[test]
    fn action_failure_survives_render_read_but_not_newer_action_or_failure() {
        let publication = Arc::new(Publication::new());
        let action = begin(&publication, CaptureMode::Change);
        reading(&publication, action);
        let reader = begin(&publication, CaptureMode::RenderRead);
        let revision = reading(&publication, reader);
        assert!(matches!(
            publication.install_frame(reader, revision, direct(), 1),
            Ok(FrameInstall::Applied(_))
        ));
        assert_eq!(
            publication
                .invalidate(
                    reader,
                    CaptureFailure::new(CaptureFailureKind::Host, "stale reader")
                )
                .unwrap(),
            Update::Ignored
        );
        assert_eq!(
            publication
                .invalidate_action(
                    action,
                    CaptureFailure::new(CaptureFailureKind::Host, "action UI failed")
                )
                .unwrap(),
            Update::Applied
        );
        match publication.frame_input() {
            Err(Error::CaptureFailed(failure)) => {
                assert_eq!(failure.diagnostic(), "action UI failed")
            }
            _ => panic!("current action failure was lost"),
        }

        let action = begin(&publication, CaptureMode::Change);
        reading(&publication, action);
        let reader = begin(&publication, CaptureMode::RenderRead);
        reading(&publication, reader);
        let newer = begin(&publication, CaptureMode::Change);
        assert_eq!(
            publication
                .invalidate_action(
                    action,
                    CaptureFailure::new(CaptureFailureKind::Host, "old action")
                )
                .unwrap(),
            Update::Ignored
        );
        assert!(publication.capture_is_current(newer).unwrap());
        let writer = write(&publication, newer);
        writer
            .finish(Err(CaptureFailure::new(
                CaptureFailureKind::Host,
                "physical failure",
            )))
            .unwrap();
        assert_eq!(
            publication
                .invalidate_action(
                    newer,
                    CaptureFailure::new(CaptureFailureKind::Host, "later action failure")
                )
                .unwrap(),
            Update::Ignored
        );
        match publication.frame_input() {
            Err(Error::CaptureFailed(failure)) => {
                assert_eq!(failure.diagnostic(), "physical failure")
            }
            _ => panic!("physical failure was replaced"),
        }
    }

    #[test]
    fn read_only_failure_keeps_latest_action_ui_and_rejects_frames() {
        for snapshot_failure in [false, true] {
            let publication = Arc::new(Publication::new());
            let action = begin(&publication, CaptureMode::Change);
            reading(&publication, action);
            let reader = begin(&publication, CaptureMode::RenderRead);
            let revision = reading(&publication, reader);
            let failure = CaptureFailure::new(CaptureFailureKind::Host, "reader failed");
            let outcome = if snapshot_failure {
                publication.fail_snapshot(reader, revision, failure)
            } else {
                publication.invalidate(reader, failure)
            };
            assert_eq!(outcome.unwrap(), Update::Applied);
            assert!(publication.property_is_current(action).unwrap());
            assert!(!publication.property_is_current(reader).unwrap());
            match publication.frame_input() {
                Err(Error::CaptureFailed(failure)) => {
                    assert_eq!(failure.kind(), CaptureFailureKind::Host);
                    assert_eq!(failure.diagnostic(), "reader failed");
                }
                _ => panic!("read failure must block frames"),
            }
            assert_eq!(
                publication
                    .invalidate_action(
                        action,
                        CaptureFailure::new(CaptureFailureKind::Host, "later action failure")
                    )
                    .unwrap(),
                Update::Ignored
            );
            match publication.frame_input() {
                Err(Error::CaptureFailed(failure)) => {
                    assert_eq!(failure.diagnostic(), "reader failed")
                }
                _ => panic!("first failure must remain"),
            }
            let newer = begin(&publication, CaptureMode::Change);
            assert!(!publication.property_is_current(action).unwrap());
            assert!(publication.property_is_current(newer).unwrap());
            let writer = write(&publication, newer);
            writer
                .finish(Err(CaptureFailure::new(
                    CaptureFailureKind::Allocation,
                    "physical write failed",
                )))
                .unwrap();
            assert!(!publication.property_is_current(newer).unwrap());
            assert_eq!(
                publication
                    .invalidate_action(
                        newer,
                        CaptureFailure::new(CaptureFailureKind::Host, "later failure")
                    )
                    .unwrap(),
                Update::Ignored
            );
            match publication.frame_input() {
                Err(Error::CaptureFailed(failure)) => {
                    assert_eq!(failure.kind(), CaptureFailureKind::Allocation);
                    assert_eq!(failure.diagnostic(), "physical write failed");
                }
                _ => panic!("physical failure must remain"),
            }
        }
    }

    #[test]
    fn failed_refresh_preserves_latest_action_ui_obligation() {
        let publication = Arc::new(Publication::new());
        let old = begin(&publication, CaptureMode::Change);
        let writer = write(&publication, old);
        let action = begin(&publication, CaptureMode::Change);
        assert_eq!(
            publication.begin_snapshot(action).unwrap(),
            SnapshotStart::Deferred
        );
        assert_eq!(writer.finish(Ok(())).unwrap(), WriteEnd::RefreshRequired);
        let refresh = begin(&publication, CaptureMode::Refresh);
        let revision = reading(&publication, refresh);
        assert_eq!(
            publication
                .fail_snapshot(
                    refresh,
                    revision,
                    CaptureFailure::new(CaptureFailureKind::Host, "refresh failed")
                )
                .unwrap(),
            Update::Applied
        );
        assert!(publication.property_is_current(action).unwrap());
        assert!(!publication.property_is_current(refresh).unwrap());
        assert!(matches!(
            publication.frame_input(),
            Err(Error::CaptureFailed(_))
        ));
    }

    #[test]
    fn render_read_supersedes_only_reading_and_preserves_newer_outcomes() {
        let publication = Arc::new(Publication::new());
        assert_eq!(
            publication.begin_capture(CaptureMode::RenderRead).unwrap(),
            CaptureStart::NotStarted
        );
        let old = begin(&publication, CaptureMode::Change);
        assert_eq!(
            publication.begin_capture(CaptureMode::RenderRead).unwrap(),
            CaptureStart::NotStarted
        );
        let old_revision = reading(&publication, old);
        let fresh = begin(&publication, CaptureMode::RenderRead);
        assert!(fresh > old);
        assert_eq!(
            publication.begin_capture(CaptureMode::RenderRead).unwrap(),
            CaptureStart::NotStarted
        );
        assert!(matches!(
            publication.begin_host_write(fresh),
            Err(Error::InvalidCaptureOperation)
        ));
        let revision = reading(&publication, fresh);
        assert_eq!(revision, old_revision);
        let mut fresh_params = direct();
        fresh_params.camera_exposure_compensation_ev = 0.75;
        assert_eq!(
            publication
                .install(fresh, revision, fresh_params, 17)
                .unwrap(),
            Update::Applied
        );
        assert_eq!(
            publication.install(old, old_revision, direct(), 9).unwrap(),
            Update::Ignored
        );
        assert_eq!(
            publication
                .fail_snapshot(
                    old,
                    old_revision,
                    CaptureFailure::new(CaptureFailureKind::Host, "old read")
                )
                .unwrap(),
            Update::Ignored
        );
        assert_eq!(
            publication
                .invalidate(
                    old,
                    CaptureFailure::new(CaptureFailureKind::Host, "old getter")
                )
                .unwrap(),
            Update::Ignored
        );
        let admission = publication
            .admit(|params| recipe::prepare(&assets(), params))
            .unwrap();
        assert_eq!(admission.input_identity(), 17);
        {
            let state = publication.state.lock().unwrap();
            let Pending::Valid { params, .. } = &state.pending else {
                panic!("fresh input missing")
            };
            assert_eq!(params.camera_exposure_compensation_ev, 0.75);
        }

        let old = begin(&publication, CaptureMode::Change);
        reading(&publication, old);
        let reader = begin(&publication, CaptureMode::RenderRead);
        let reader_revision = reading(&publication, reader);
        let newer = begin(&publication, CaptureMode::Change);
        let newer_revision = reading(&publication, newer);
        assert_eq!(
            publication
                .install(newer, newer_revision, direct(), 23)
                .unwrap(),
            Update::Applied
        );
        assert_eq!(
            publication
                .install(reader, reader_revision, direct(), 18)
                .unwrap(),
            Update::Ignored
        );
        assert_eq!(
            publication
                .invalidate(
                    reader,
                    CaptureFailure::new(CaptureFailureKind::Host, "obsolete reader")
                )
                .unwrap(),
            Update::Ignored
        );
        assert_eq!(
            publication
                .admit(|params| recipe::prepare(&assets(), params))
                .unwrap()
                .input_identity(),
            23
        );
    }

    #[test]
    fn render_read_refuses_writes_failure_revision_mismatch_and_overflow() {
        let publication = Arc::new(Publication::new());
        let old = begin(&publication, CaptureMode::Change);
        let pending_write = write(&publication, old);
        let latest = begin(&publication, CaptureMode::Change);
        assert_eq!(
            publication.begin_snapshot(latest).unwrap(),
            SnapshotStart::Deferred
        );
        assert_eq!(
            publication.begin_capture(CaptureMode::RenderRead).unwrap(),
            CaptureStart::NotStarted
        );
        pending_write
            .finish(Err(CaptureFailure::new(
                CaptureFailureKind::Host,
                "physical write failed",
            )))
            .unwrap();
        assert!(matches!(
            publication.admit(|_| panic!("failed write admitted")),
            Err(Error::CaptureFailed(_))
        ));
        assert_eq!(
            publication.begin_capture(CaptureMode::RenderRead).unwrap(),
            CaptureStart::NotStarted
        );

        let old = begin(&publication, CaptureMode::Change);
        reading(&publication, old);
        {
            let mut state = publication.state.lock().unwrap();
            state.write_revision += 1;
        }
        assert_eq!(
            publication.begin_capture(CaptureMode::RenderRead).unwrap(),
            CaptureStart::NotStarted
        );
        let old = begin(&publication, CaptureMode::Change);
        reading(&publication, old);
        publication.state.lock().unwrap().issued = u64::MAX;
        assert!(matches!(
            publication.begin_capture(CaptureMode::RenderRead),
            Err(Error::Poisoned)
        ));
        assert!(matches!(
            publication.admit(|_| panic!("overflow admitted")),
            Err(Error::Poisoned)
        ));
    }

    #[test]
    fn render_read_superseded_by_write_preserves_completion_and_failure() {
        for fails in [false, true] {
            let publication = Arc::new(Publication::new());
            let old = begin(&publication, CaptureMode::Change);
            reading(&publication, old);
            let reader = begin(&publication, CaptureMode::RenderRead);
            let reader_revision = reading(&publication, reader);
            let newer = begin(&publication, CaptureMode::Change);
            let writer = write(&publication, newer);
            assert_eq!(
                publication.begin_snapshot(newer).unwrap(),
                SnapshotStart::Deferred
            );
            assert_eq!(
                publication
                    .install(reader, reader_revision, direct(), 1)
                    .unwrap(),
                Update::Ignored
            );
            assert_eq!(
                publication
                    .fail_snapshot(
                        reader,
                        reader_revision,
                        CaptureFailure::new(CaptureFailureKind::Host, "late reader failure")
                    )
                    .unwrap(),
                Update::Ignored
            );
            assert_eq!(
                publication
                    .invalidate(
                        reader,
                        CaptureFailure::new(CaptureFailureKind::Host, "late reader drop")
                    )
                    .unwrap(),
                Update::Ignored
            );
            if fails {
                assert_eq!(
                    writer
                        .finish(Err(CaptureFailure::new(
                            CaptureFailureKind::Host,
                            "partial write"
                        )))
                        .unwrap(),
                    WriteEnd::Complete
                );
                match publication.admit(|_| panic!("failed write admitted")) {
                    Err(Error::CaptureFailed(failure)) => {
                        assert_eq!(failure.diagnostic(), "partial write")
                    }
                    _ => panic!("physical write failure lost"),
                }
                assert_eq!(
                    publication.begin_capture(CaptureMode::RenderRead).unwrap(),
                    CaptureStart::NotStarted
                );
            } else {
                assert_eq!(writer.finish(Ok(())).unwrap(), WriteEnd::RefreshRequired);
                assert!(matches!(
                    publication.admit(|_| panic!("refresh pending admitted")),
                    Err(Error::RefreshRequired)
                ));
                let refresh = begin(&publication, CaptureMode::Refresh);
                let refresh_revision = reading(&publication, refresh);
                let fresh = begin(&publication, CaptureMode::RenderRead);
                let revision = reading(&publication, fresh);
                assert_eq!(revision, refresh_revision);
                assert_eq!(
                    publication.install(fresh, revision, direct(), 29).unwrap(),
                    Update::Applied
                );
                assert_eq!(
                    publication
                        .install(refresh, refresh_revision, direct(), 28)
                        .unwrap(),
                    Update::Ignored
                );
                assert_eq!(
                    publication
                        .admit(|params| recipe::prepare(&assets(), params))
                        .unwrap()
                        .input_identity(),
                    29
                );
            }
        }
    }

    #[test]
    fn render_read_supersedes_a_read_on_another_thread() {
        let publication = Arc::new(Publication::new());
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (resume_tx, resume_rx) = std::sync::mpsc::channel();
        let worker_publication = Arc::clone(&publication);
        let worker = std::thread::spawn(move || {
            let old = begin(&worker_publication, CaptureMode::Change);
            let old_revision = reading(&worker_publication, old);
            ready_tx.send(()).unwrap();
            resume_rx.recv().unwrap();
            worker_publication
                .install(old, old_revision, direct(), 1)
                .unwrap()
        });
        ready_rx.recv().unwrap();
        let reader = begin(&publication, CaptureMode::RenderRead);
        let revision = reading(&publication, reader);
        assert_eq!(
            publication.install(reader, revision, direct(), 2).unwrap(),
            Update::Applied
        );
        resume_tx.send(()).unwrap();
        assert_eq!(worker.join().unwrap(), Update::Ignored);
        assert_eq!(
            publication
                .admit(|params| recipe::prepare(&assets(), params))
                .unwrap()
                .input_identity(),
            2
        );
    }

    #[test]
    fn ordered_capture_completion_and_admission_metadata() {
        let assets = assets();
        let publication = Publication::new();
        let initial = begin(&publication, CaptureMode::Initial);
        assert!(matches!(
            publication.admit(|_| panic!("capturing input built")),
            Err(Error::Capturing)
        ));
        let revision = reading(&publication, initial);
        let newer = begin(&publication, CaptureMode::Change);
        assert_eq!(
            publication.begin_capture(CaptureMode::Initial).unwrap(),
            CaptureStart::NotStarted
        );
        let newer_revision = reading(&publication, newer);
        assert_eq!(newer_revision, 0);
        assert_eq!(
            publication
                .install(newer, newer_revision, direct(), 0)
                .unwrap(),
            Update::Applied
        );
        let held = publication
            .admit(|params| recipe::prepare(&assets, params))
            .unwrap();
        assert_eq!(held.input_identity(), 0);
        assert_eq!(held.build(), 1);
        assert_eq!(
            publication
                .install(initial, revision, direct(), 12)
                .unwrap(),
            Update::Ignored
        );
        assert_eq!(
            publication
                .invalidate(
                    initial,
                    CaptureFailure::new(CaptureFailureKind::Host, "obsolete getter")
                )
                .unwrap(),
            Update::Ignored
        );
        let reused = publication
            .admit(|_| panic!("obsolete completion caused rebuild"))
            .unwrap();
        assert_eq!(reused.build(), 1);
        assert!(Arc::ptr_eq(&held.state, &reused.state));
        let bad = begin(&publication, CaptureMode::Change);
        let bad_revision = reading(&publication, bad);
        let mut invalid = direct();
        invalid.film_gamma_factor = 0.0;
        publication.install(bad, bad_revision, invalid, 1).unwrap();
        assert!(matches!(
            publication.admit(|params| recipe::prepare(&assets, params)),
            Err(Error::Preparation(_))
        ));
        let back = begin(&publication, CaptureMode::Change);
        let back_revision = reading(&publication, back);
        publication
            .install(back, back_revision, direct(), 0)
            .unwrap();
        assert_eq!(
            publication
                .admit(|_| panic!("A recovery rebuilt"))
                .unwrap()
                .build(),
            1
        );
        let replacement = begin(&publication, CaptureMode::Change);
        let replacement_revision = reading(&publication, replacement);
        publication
            .install(
                replacement,
                replacement_revision,
                params(ScanRoute::PositiveDirect, 1),
                2,
            )
            .unwrap();
        let replacement = publication
            .admit(|params| recipe::prepare(&assets, params))
            .unwrap();
        assert_eq!(replacement.build(), 2);
        assert_eq!(held.input_identity(), 0);
        assert_eq!(held.build(), 1);
    }

    #[test]
    fn registered_writes_defer_and_reconcile_actual_completion() {
        let cases = [
            "late-success",
            "late-partial-failure",
            "abandoned",
            "first-finishes-first",
            "last-finishes-first",
        ];
        let mut rows = RowCases::new("publication-host-writes", cases);
        for id in cases {
            let row = rows.case(id);
            let publication = Arc::new(Publication::new());
            let old = begin(&publication, CaptureMode::Initial);
            let first = write(&publication, old);
            let latest = begin(&publication, CaptureMode::Change);
            let second = write(&publication, latest);
            assert!(matches!(
                publication.begin_host_write(old).unwrap(),
                HostWriteStart::Ignored
            ));
            assert_eq!(
                publication.begin_snapshot(latest).unwrap(),
                SnapshotStart::Deferred
            );
            assert!(matches!(
                publication.admit(|_| panic!("outstanding write admitted")),
                Err(Error::Capturing)
            ));
            assert_eq!(
                publication.begin_capture(CaptureMode::Refresh).unwrap(),
                CaptureStart::NotStarted
            );
            if id == "last-finishes-first" {
                assert_eq!(second.finish(Ok(())).unwrap(), WriteEnd::Complete);
                assert_eq!(first.finish(Ok(())).unwrap(), WriteEnd::RefreshRequired);
            } else {
                match id {
                    "late-partial-failure" => {
                        first
                            .finish(Err(CaptureFailure::new(
                                CaptureFailureKind::Host,
                                "late partial failure",
                            )))
                            .unwrap();
                    }
                    "abandoned" => drop(first),
                    _ => {
                        assert_eq!(first.finish(Ok(())).unwrap(), WriteEnd::Complete);
                    }
                }
                let end = second.finish(Ok(())).unwrap();
                assert_eq!(
                    end,
                    if matches!(id, "late-partial-failure" | "abandoned") {
                        WriteEnd::Complete
                    } else {
                        WriteEnd::RefreshRequired
                    }
                );
            }
            assert_eq!(publication.state.lock().unwrap().outstanding_writes, 0);
            if matches!(id, "late-partial-failure" | "abandoned") {
                assert!(matches!(
                    publication.admit(|_| panic!("failed write admitted")),
                    Err(Error::CaptureFailed(_))
                ));
                assert_eq!(
                    publication.begin_capture(CaptureMode::Refresh).unwrap(),
                    CaptureStart::NotStarted
                );
                let recover = begin(&publication, CaptureMode::Change);
                let revision = reading(&publication, recover);
                publication.install(recover, revision, direct(), 4).unwrap();
            } else {
                assert!(matches!(
                    publication.admit(|_| panic!("refresh missing")),
                    Err(Error::RefreshRequired)
                ));
                let capture = if id == "first-finishes-first" {
                    let change = begin(&publication, CaptureMode::Change);
                    assert_eq!(
                        publication.begin_capture(CaptureMode::Refresh).unwrap(),
                        CaptureStart::NotStarted
                    );
                    change
                } else {
                    let refresh = begin(&publication, CaptureMode::Refresh);
                    assert!(matches!(
                        publication.begin_host_write(refresh),
                        Err(Error::InvalidCaptureOperation)
                    ));
                    refresh
                };
                let revision = reading(&publication, capture);
                publication.install(capture, revision, direct(), 5).unwrap();
            }
            row.passed();
        }
        rows.finish();
    }

    #[test]
    fn snapshot_phase_rejection_and_failure_precedence() {
        let publication = Arc::new(Publication::new());
        let ticket = begin(&publication, CaptureMode::Change);
        assert!(matches!(
            publication.capture_is_current(0),
            Err(Error::InvalidCaptureTicket)
        ));
        assert!(matches!(
            publication.begin_snapshot(ticket + 1),
            Err(Error::InvalidCaptureTicket)
        ));
        assert!(matches!(
            publication.install(ticket, 0, direct(), 1),
            Err(Error::InvalidCaptureOperation)
        ));
        let revision = reading(&publication, ticket);
        assert_eq!(revision, 0);
        assert!(matches!(
            publication.begin_snapshot(ticket),
            Err(Error::InvalidCaptureOperation)
        ));
        assert!(matches!(
            publication.begin_host_write(ticket),
            Err(Error::InvalidCaptureOperation)
        ));
        assert!(matches!(
            publication.install(ticket, 1, direct(), 1),
            Err(Error::InvalidCaptureOperation)
        ));
        assert!(publication.capture_is_current(ticket).unwrap());
        publication.install(ticket, revision, direct(), 1).unwrap();
        assert_eq!(
            publication.install(ticket, 999, direct(), 9).unwrap(),
            Update::Ignored
        );
        let first = begin(&publication, CaptureMode::Change);
        let one = write(&publication, first);
        let latest = begin(&publication, CaptureMode::Change);
        let two = write(&publication, latest);
        publication.begin_snapshot(latest).unwrap();
        one.finish(Err(CaptureFailure::new(
            CaptureFailureKind::Allocation,
            "first failure",
        )))
        .unwrap();
        two.finish(Err(CaptureFailure::new(
            CaptureFailureKind::Host,
            "second failure",
        )))
        .unwrap();
        let Err(Error::CaptureFailed(failure)) = publication.admit(|_| panic!("failure erased"))
        else {
            panic!("failure missing")
        };
        assert_eq!(failure.kind(), CaptureFailureKind::Allocation);
        assert_eq!(failure.diagnostic(), "first failure");
        assert_eq!(
            CaptureFailure::new(CaptureFailureKind::Host, "").diagnostic(),
            "host parameter capture failed"
        );
        let preview = CaptureFailure::new(CaptureFailureKind::Host, &"é".repeat(2048));
        assert_eq!(preview.diagnostic().len(), 2046);
    }

    #[test]
    fn exhausted_ordering_and_blocked_guards_do_not_wrap() {
        let cases = [
            "ticket",
            "revision-start",
            "count",
            "revision-finish",
            "blocked-drop",
            "poisoned-write-drop",
        ];
        let mut rows = RowCases::new("publication-exhaustion", cases);
        for id in cases {
            let row = rows.case(id);
            let publication = Arc::new(Publication::new());
            let capture = begin(&publication, CaptureMode::Change);
            match id {
                "ticket" => {
                    publication.state.lock().unwrap().issued = u64::MAX;
                    assert!(matches!(
                        publication.begin_capture(CaptureMode::Change),
                        Err(Error::Poisoned)
                    ));
                }
                "revision-start" => {
                    publication.state.lock().unwrap().write_revision = u64::MAX;
                    assert!(matches!(
                        publication.begin_host_write(capture),
                        Err(Error::Poisoned)
                    ));
                }
                "count" => {
                    publication.state.lock().unwrap().outstanding_writes = u64::MAX;
                    assert!(matches!(
                        publication.begin_host_write(capture),
                        Err(Error::Poisoned)
                    ));
                }
                "revision-finish" => {
                    let guard = write(&publication, capture);
                    publication.state.lock().unwrap().write_revision = u64::MAX;
                    assert!(matches!(guard.finish(Ok(())), Err(Error::Poisoned)));
                }
                "blocked-drop" => {
                    let guard = write(&publication, capture);
                    publication.block();
                    drop(guard);
                }
                "poisoned-write-drop" => {
                    let guard = write(&publication, capture);
                    assert!(
                        catch_unwind(AssertUnwindSafe(|| {
                            let _state = publication.state.lock().unwrap();
                            panic!("intentional outstanding-write state poison");
                        }))
                        .is_err()
                    );
                    // Drop must be non-panicking and must not recover the poisoned lock.
                    drop(guard);
                    assert!(publication.state.is_poisoned());
                }
                _ => unreachable!(),
            }
            assert!(matches!(
                publication.admit(|_| panic!("blocked owner built")),
                Err(Error::Poisoned)
            ));
            assert!(matches!(
                publication.begin_capture(CaptureMode::Change),
                Err(Error::Poisoned)
            ));
            row.passed();
        }
        rows.finish();
    }

    #[test]
    fn build_counter_exhaustion_keeps_last_committed_publication() {
        let assets = assets();
        let publication = Publication::new();
        publication.stage(valid(1, direct())).unwrap();
        let held = publication
            .admit_state(|input| recipe::prepare(&assets, input))
            .unwrap();
        let words = backing_words(&held);
        publication
            .stage(valid(2, params(ScanRoute::NegativePrint, 1)))
            .unwrap();
        publication.state.lock().unwrap().build = u64::MAX;
        assert!(matches!(
            publication.admit_state(|input| recipe::prepare(&assets, input)),
            Err(Error::Poisoned)
        ));
        let state = publication.state.lock().unwrap();
        assert!(matches!(state.pending, Pending::Blocked));
        assert_eq!(state.published.as_ref().unwrap().identity, 1);
        assert_eq!(state.published.as_ref().unwrap().build, 1);
        assert!(Arc::ptr_eq(&state.published.as_ref().unwrap().state, &held));
        assert_eq!(state.build, u64::MAX);
        drop(state);
        assert_eq!(backing_words(&held), words);
    }
}
