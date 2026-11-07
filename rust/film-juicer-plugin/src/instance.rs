//! Safe per-effect publication orchestration. Host and CUDA owners stay outside.

#![forbid(unsafe_code)]
#![expect(
    clippy::result_large_err,
    reason = "Instance propagates Publication errors with an inline bounded diagnostic; failure recording cannot allocate."
)]

use std::sync::Arc;

use film_juicer_core::assets::Assets;
use film_juicer_core::{recipe, state};

use crate::submission::{self, Submission};

pub(crate) struct Instance {
    publication: Arc<state::Publication>,
    submission: Submission,
}

impl Instance {
    pub(crate) fn new() -> Self {
        Self {
            publication: Arc::new(state::Publication::new()),
            submission: Submission::new(),
        }
    }
    pub(crate) fn bind_identity(&self, seed: u64, token: u64) -> Result<(), submission::Error> {
        self.submission.bind_identity(seed, token)
    }
    pub(crate) fn identity(&self) -> Option<submission::Identity> {
        self.submission.identity()
    }
    pub(crate) fn select_submission(
        &self,
        key: submission::Key,
    ) -> Result<submission::Selection, submission::Error> {
        self.submission.select(key)
    }
    pub(crate) fn invalidate_submission(
        &self,
        device_id: i32,
        context: usize,
    ) -> Result<bool, submission::Error> {
        self.submission.invalidate(device_id, context)
    }
    #[cfg(feature = "test-support")]
    pub(crate) fn observe_submission(
        &self,
    ) -> Result<Option<submission::Selection>, submission::Error> {
        self.submission.observe()
    }
    #[cfg(test)]
    pub(crate) fn poison_submission_for_test(&self) {
        self.submission.poison_for_test();
    }
    pub(crate) fn begin_capture(
        &self,
        mode: state::CaptureMode,
    ) -> Result<state::CaptureStart, state::Error> {
        self.publication.begin_capture(mode)
    }
    pub(crate) fn capture_is_current(&self, capture: u64) -> Result<bool, state::Error> {
        self.publication.capture_is_current(capture)
    }
    pub(crate) fn begin_host_write(
        &self,
        capture: u64,
    ) -> Result<state::HostWriteStart, state::Error> {
        self.publication.begin_host_write(capture)
    }
    pub(crate) fn begin_snapshot(
        &self,
        capture: u64,
    ) -> Result<state::SnapshotStart, state::Error> {
        self.publication.begin_snapshot(capture)
    }
    pub(crate) fn install(
        &self,
        capture: u64,
        write_revision: u64,
        params: recipe::Params,
        input_identity: u64,
    ) -> Result<state::Update, state::Error> {
        self.publication
            .install(capture, write_revision, params, input_identity)
    }
    pub(crate) fn fail_snapshot(
        &self,
        capture: u64,
        write_revision: u64,
        failure: state::CaptureFailure,
    ) -> Result<state::Update, state::Error> {
        self.publication
            .fail_snapshot(capture, write_revision, failure)
    }
    pub(crate) fn invalidate(
        &self,
        capture: u64,
        failure: state::CaptureFailure,
    ) -> Result<state::Update, state::Error> {
        self.publication.invalidate(capture, failure)
    }
    pub(crate) fn admit(&self, assets: &Assets) -> Result<state::Admission, state::Error> {
        self.publication.admit(|params| {
            #[cfg(feature = "test-support")]
            testing::before_prepare();
            let outcome = recipe::prepare(assets, params);
            #[cfg(feature = "test-support")]
            testing::after_prepare();
            outcome
        })
    }
    #[cfg(feature = "test-support")]
    pub(crate) fn observe(&self) -> Result<state::Observation, state::Error> {
        self.publication.observe()
    }
    pub(crate) fn block(&self) {
        self.publication.block();
    }
    pub(crate) fn close(self) {}
}

#[cfg(feature = "test-support")]
pub(crate) mod testing {
    use std::cell::Cell;
    thread_local! {
        static BEFORE_PREPARE: Cell<Option<fn()>> = const { Cell::new(None) };
        static AFTER_PREPARE: Cell<Option<fn()>> = const { Cell::new(None) };
    }
    pub(crate) fn set_prepare_hooks(before: Option<fn()>, after: Option<fn()>) {
        BEFORE_PREPARE.set(before);
        AFTER_PREPARE.set(after);
    }
    pub(super) fn before_prepare() {
        if let Some(hook) = BEFORE_PREPARE.get() {
            hook();
        }
    }
    pub(super) fn after_prepare() {
        if let Some(hook) = AFTER_PREPARE.get() {
            hook();
        }
    }
}
