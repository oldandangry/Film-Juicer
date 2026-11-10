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
    Failed(CaptureFailure),
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
            Pending::Failed(failure) => Err(Error::CaptureFailed(failure.clone())),
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
        let mut state = self.lock()?;
        if !state.current(capture)? {
            return Ok(Update::Ignored);
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
            return Ok(Update::Deferred);
        }
        state.pending = Pending::Valid { params, identity };
        Ok(Update::Applied)
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
        if !matches!(state.pending, Pending::Capturing { phase: CapturePhase::Reading { revision }, .. }
            if revision == write_revision)
        {
            return Err(Error::InvalidCaptureOperation);
        }
        state.pending = Pending::Failed(failure);
        Ok(Update::Applied)
    }
    pub fn invalidate(&self, capture: u64, failure: CaptureFailure) -> Result<Update, Error> {
        let mut state = self.lock()?;
        if !state.current(capture)? {
            return Ok(Update::Ignored);
        }
        state.pending = Pending::Failed(failure);
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
            if !matches!(state.pending, Pending::Failed(_)) {
                let retired = std::mem::replace(&mut state.pending, Pending::Failed(failure));
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
                    && published.identity == identity
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
            Pending::Failed(failure) => ObservedPending::Failed(failure.clone()),
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
            assert_ne!(state.scanner().lut().hash(), 0);
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
