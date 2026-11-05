//! Ordered rendering of one held complete state through borrowed native access.
#![forbid(unsafe_code)]

use std::panic::{AssertUnwindSafe, catch_unwind};

use film_juicer_core::{
    assets::{AssetError, Assets},
    film::input::Meter,
    recipe::RenderState,
};

use crate::asset_noise::NoiseOwner;

#[cfg(feature = "test-support")]
thread_local! {
    static SHORT_NOISE_VIEW: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static SHORT_NOISE_VIEW_CONSUMED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(feature = "test-support")]
pub(crate) fn set_short_noise_view(active: bool) {
    SHORT_NOISE_VIEW.set(active);
    SHORT_NOISE_VIEW_CONSUMED.set(false);
}

#[cfg(feature = "test-support")]
pub(crate) fn short_noise_view_consumed() -> bool {
    SHORT_NOISE_VIEW_CONSUMED.get()
}

#[cfg(feature = "test-support")]
fn shorten_noise_view(
    mut view: crate::asset_noise::NoiseView<'_>,
) -> crate::asset_noise::NoiseView<'_> {
    if SHORT_NOISE_VIEW.replace(false) {
        SHORT_NOISE_VIEW_CONSUMED.set(true);
        // Corrupt only the call-local literal boundary input. The immutable
        // validated source and its retained owner are unchanged.
        view.stbn = &view.stbn[..view.stbn.len() - 1];
    }
    view
}

use super::{
    Abort, Frame, FramePolicy, OwnerBorrow, RenderError, RenderOutcome, SubmissionLatch,
    edge::{self, Failure},
    sys,
};

#[cfg(feature = "test-support")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct NoiseCallObservation {
    pub sequence: u32,
    pub precheck_order: u32,
    pub acquisition_order: u32,
    pub admission_order: u32,
    pub inspection_order: u32,
    pub release_order: u32,
    pub complete_order: u32,
    pub drop_order: u32,
    pub native_render_order: u32,
    pub acquisition_attempts: u32,
    pub successful_results: u32,
    pub preparation_failures: u32,
    pub allocation_failures: u32,
    pub internal_failures: u32,
    pub held_at_complete: u32,
    pub complete_calls: u32,
    pub drops_after_complete: u32,
    pub native_render_calls: u32,
    pub arc_references_at_complete: usize,
}

#[cfg(feature = "test-support")]
thread_local! {
    static NOISE_CALL_OBSERVATION: std::cell::Cell<NoiseCallObservation> = const {
        std::cell::Cell::new(NoiseCallObservation {
            sequence: 0,
            precheck_order: 0,
            acquisition_order: 0,
            admission_order: 0,
            inspection_order: 0,
            release_order: 0,
            complete_order: 0,
            drop_order: 0,
            native_render_order: 0,
            acquisition_attempts: 0,
            successful_results: 0,
            preparation_failures: 0,
            allocation_failures: 0,
            internal_failures: 0,
            held_at_complete: 0,
            complete_calls: 0,
            drops_after_complete: 0,
            native_render_calls: 0,
            arc_references_at_complete: 0,
        })
    };
}

#[cfg(feature = "test-support")]
fn observe_stage(stage: fn(&mut NoiseCallObservation, u32)) {
    NOISE_CALL_OBSERVATION.with(|cell| {
        let mut observation = cell.get();
        observation.sequence += 1;
        let order = observation.sequence;
        stage(&mut observation, order);
        cell.set(observation);
    });
}

#[cfg(feature = "test-support")]
fn order_precheck(observation: &mut NoiseCallObservation, order: u32) {
    observation.precheck_order = order;
}

#[cfg(feature = "test-support")]
fn order_admission(observation: &mut NoiseCallObservation, order: u32) {
    observation.admission_order = order;
}

#[cfg(feature = "test-support")]
fn order_inspection(observation: &mut NoiseCallObservation, order: u32) {
    observation.inspection_order = order;
}

#[cfg(feature = "test-support")]
fn order_release(observation: &mut NoiseCallObservation, order: u32) {
    observation.release_order = order;
}

#[cfg(feature = "test-support")]
pub(crate) fn reset_noise_call_observation() {
    NOISE_CALL_OBSERVATION.with(|cell| cell.set(NoiseCallObservation::default()));
}

#[cfg(feature = "test-support")]
pub(crate) fn noise_call_observation() -> NoiseCallObservation {
    NOISE_CALL_OBSERVATION.with(std::cell::Cell::get)
}

#[cfg(feature = "test-support")]
fn observe_complete(noise: Option<&Result<NoiseOwner, AssetError>>) {
    NOISE_CALL_OBSERVATION.with(|cell| {
        let mut observation = cell.get();
        observation.complete_calls += 1;
        if let Some(noise) = noise {
            observation.held_at_complete += 1;
            observation.arc_references_at_complete = match noise {
                Ok(owner) => owner.arc_references(),
                Err(AssetError::Noise(error)) => std::sync::Arc::strong_count(error),
                Err(_) => 0,
            };
        }
        observation.sequence += 1;
        observation.complete_order = observation.sequence;
        cell.set(observation);
    });
}

#[cfg(feature = "test-support")]
fn observe_native_render() {
    NOISE_CALL_OBSERVATION.with(|cell| {
        let mut observation = cell.get();
        observation.native_render_calls += 1;
        observation.sequence += 1;
        observation.native_render_order = observation.sequence;
        cell.set(observation);
    });
}

#[cfg(feature = "test-support")]
fn observe_noise_drop(dropped: bool) {
    if dropped {
        NOISE_CALL_OBSERVATION.with(|cell| {
            let mut observation = cell.get();
            observation.drops_after_complete += 1;
            observation.sequence += 1;
            observation.drop_order = observation.sequence;
            cell.set(observation);
        });
    }
}

#[cfg(feature = "test-support")]
fn observe_noise_attempt() {
    NOISE_CALL_OBSERVATION.with(|cell| {
        let mut observation = cell.get();
        observation.acquisition_attempts += 1;
        observation.sequence += 1;
        observation.acquisition_order = observation.sequence;
        cell.set(observation);
    });
}

#[cfg(feature = "test-support")]
fn observe_noise_result(result: &Result<NoiseOwner, AssetError>) {
    NOISE_CALL_OBSERVATION.with(|cell| {
        let mut observation = cell.get();
        match result {
            Ok(_) => observation.successful_results += 1,
            Err(AssetError::Noise(error)) => {
                match crate::asset_bridge::noise_category(error.kind()) {
                    sys::FJ_STATUS_PREPARATION_FAILURE => observation.preparation_failures += 1,
                    sys::FJ_STATUS_ALLOCATION_FAILURE => observation.allocation_failures += 1,
                    _ => observation.internal_failures += 1,
                }
            }
            Err(_) => observation.internal_failures += 1,
        }
        cell.set(observation);
    });
}

struct CallGuard<'owner, 'call> {
    owner: &'owner mut OwnerBorrow<'call>,
    noise: Option<Result<NoiseOwner, AssetError>>,
    admitted: bool,
    completed: bool,
    render_outcome_valid: bool,
    abort_failed: bool,
    recovery_scope_entered: bool,
    inspected: Option<sys::FjCudaContext>,
    force_fatal: bool,
}

impl<'owner, 'call> CallGuard<'owner, 'call> {
    fn new(owner: &'owner mut OwnerBorrow<'call>) -> Self {
        Self {
            owner,
            noise: None,
            admitted: false,
            completed: false,
            render_outcome_valid: false,
            abort_failed: false,
            recovery_scope_entered: false,
            inspected: None,
            force_fatal: true,
        }
    }

    fn complete(
        &mut self,
        outcome: sys::FjRenderOutcome,
        diagnostic: &str,
    ) -> edge::FjCallCompletionResult {
        if self.admitted {
            self.admitted = false;
            self.owner.release();
            #[cfg(feature = "test-support")]
            observe_stage(order_release);
        }
        self.completed = true;
        #[cfg(feature = "test-support")]
        observe_complete(self.noise.as_ref());
        self.owner.complete(&edge::FjCallCompletionInput {
            outcome,
            diagnostic: sys::FjStringView {
                data: diagnostic.as_ptr().cast(),
                count: diagnostic.len(),
            },
            render_outcome_valid: u32::from(self.render_outcome_valid),
            abort_failed: u32::from(self.abort_failed),
            recovery_scope_entered: u32::from(self.recovery_scope_entered),
            has_inspected_context: u32::from(self.inspected.is_some()),
            inspected_context: self.inspected.unwrap_or(sys::FjCudaContext {
                device_id: 0,
                context: 0,
            }),
            force_fatal: u32::from(self.force_fatal),
        })
    }

    fn finish(
        mut self,
        result: Result<(sys::FjRenderOutcome, String), Failure>,
    ) -> Result<RenderOutcome, RenderError> {
        let (outcome, diagnostic) = result.unwrap_or_else(|failure| {
            (
                sys::FjRenderOutcome {
                    status: failure.status,
                    flags: 0,
                },
                failure.diagnostic,
            )
        });
        let completion = self.complete(outcome, &diagnostic);
        // The no-throw native continuation has completed delivery and recovery.
        // A cached source/error may still retain its allocation after this clone.
        let dropped_noise = self.noise.take();
        #[cfg(feature = "test-support")]
        let had_noise = dropped_noise.is_some();
        drop(dropped_noise);
        #[cfg(feature = "test-support")]
        observe_noise_drop(had_noise);
        if completion.outcome.status.category == sys::FJ_STATUS_SUCCESS
            || completion.outcome.status.category == sys::FJ_STATUS_CANCELLED
        {
            Ok(RenderOutcome {
                completion,
                diagnostic,
            })
        } else {
            Err(RenderError {
                completion,
                diagnostic,
            })
        }
    }

    fn abort(&mut self, abort: &Abort<'_>) -> Result<bool, Failure> {
        match abort.query() {
            Ok(requested) => Ok(requested),
            Err(failure) => {
                self.abort_failed = true;
                Err(failure)
            }
        }
    }

    fn enter(
        &mut self,
        assets: &Assets,
        state: Option<&RenderState>,
        frame: &Frame<'_>,
    ) -> Result<sys::FjAutoExposure, Failure> {
        let preview = if let Some(state) = state {
            self.owner.preview(frame, meter(state.input().meter()))?
        } else {
            empty_preview()
        };
        if state.is_some_and(|state| state.grain().active()) {
            self.owner.precheck()?;
            #[cfg(feature = "test-support")]
            observe_stage(order_precheck);
            #[cfg(feature = "test-support")]
            observe_noise_attempt();
            let noise = assets.noise().map(NoiseOwner::new);
            #[cfg(feature = "test-support")]
            observe_noise_result(&noise);
            self.noise = Some(noise);
        }
        self.owner.admit()?;
        self.admitted = true;
        #[cfg(feature = "test-support")]
        observe_stage(order_admission);
        let inspected = self.owner.inspect(frame);
        #[cfg(feature = "test-support")]
        observe_stage(order_inspection);
        self.inspected = Some(inspected?);
        Ok(preview)
    }
}

impl Drop for CallGuard<'_, '_> {
    fn drop(&mut self) {
        if !self.completed {
            self.force_fatal = true;
            self.complete(
                sys::FjRenderOutcome {
                    status: edge::status(sys::FJ_STATUS_INTERNAL_FAILURE),
                    flags: 0,
                },
                "prepared render unwound",
            );
        }
        let dropped_noise = self.noise.take();
        #[cfg(feature = "test-support")]
        let had_noise = dropped_noise.is_some();
        drop(dropped_noise);
        #[cfg(feature = "test-support")]
        observe_noise_drop(had_noise);
    }
}

fn meter(value: Meter) -> u32 {
    match value {
        Meter::CenterWeighted => sys::FJ_METER_CENTER_WEIGHTED,
        Meter::Average => sys::FJ_METER_AVERAGE,
        Meter::Median => sys::FJ_METER_MEDIAN,
        Meter::Partial => sys::FJ_METER_PARTIAL,
        Meter::Matrix => sys::FJ_METER_MATRIX,
        Meter::MultiZone => sys::FJ_METER_MULTI_ZONE,
        Meter::HighlightWeighted => sys::FJ_METER_HIGHLIGHT_WEIGHTED,
    }
}

fn empty_preview() -> sys::FjAutoExposure {
    sys::FjAutoExposure {
        source_bounds: sys::FjRect {
            x1: 0,
            y1: 0,
            x2: 0,
            y2: 0,
        },
        meter_bounds: sys::FjRect {
            x1: 0,
            y1: 0,
            x2: 0,
            y2: 0,
        },
        preview_width: 0,
        preview_height: 0,
        method: 0,
        hash: 0,
    }
}

fn cancelled() -> (sys::FjRenderOutcome, String) {
    (
        sys::FjRenderOutcome {
            status: edge::status(sys::FJ_STATUS_CANCELLED),
            flags: 0,
        },
        String::new(),
    )
}

fn same_domain(rect: sys::FjRect, extent: sys::FjExtent) -> bool {
    rect.x1 == extent.x
        && rect.y1 == extent.y
        && rect.x2.checked_sub(rect.x1) == Some(extent.width)
        && rect.y2.checked_sub(rect.y1) == Some(extent.height)
}

fn preflight(
    state: &RenderState,
    frame: &sys::FjFrame,
    policy: &FramePolicy,
) -> Result<(), Failure> {
    let policy = policy.raw();
    let optics = &policy.optics;
    let full = policy.geometry.full_frame;
    if optics.flags != 0
        && (full.width <= 0
            || full.height <= 0
            || !same_domain(frame.render_window, full)
            || !same_domain(frame.source.bounds, full)
            || !same_domain(frame.destination.bounds, full))
    {
        return Err(Failure::new(
            sys::FJ_STATUS_PREPARATION_FAILURE,
            "ResourceDescriptorMismatch component=scatter_halation field=full_frame_domain",
        ));
    }
    if optics.flags & (sys::FJ_OPTICS_CAMERA_DIFFUSION | sys::FJ_OPTICS_ENLARGER_DIFFUSION) != 0
        && (!same_domain(frame.render_window, optics.full_frame)
            || !same_domain(frame.source.bounds, optics.full_frame)
            || !same_domain(frame.full_frame_extent, optics.full_frame))
    {
        return Err(Failure::new(
            sys::FJ_STATUS_PREPARATION_FAILURE,
            "ResourceDescriptorMismatch component=diffusion field=full_frame_domain",
        ));
    }
    if state.optics().camera_lens_blur().sigma_um() > 0.0 {
        return Err(Failure::new(
            sys::FJ_STATUS_PREPARATION_FAILURE,
            "ExactOpticsNotImplementedForPhase6 component=camera_lens_blur",
        ));
    }
    Ok(())
}

fn noise_failure(error: &AssetError) -> Failure {
    let category = match error {
        AssetError::Noise(error) => crate::asset_bridge::noise_category(error.kind()),
        _ => sys::FJ_STATUS_INTERNAL_FAILURE,
    };
    Failure::new(category, error.to_string())
}

pub(crate) fn render(
    owner: &mut OwnerBorrow<'_>,
    assets: &Assets,
    state: &RenderState,
    frame: &Frame<'_>,
    policy: &FramePolicy,
    latch: &mut SubmissionLatch<'_>,
    abort: &Abort<'_>,
) -> Result<RenderOutcome, RenderError> {
    let mut guard = CallGuard::new(owner);
    let result = catch_unwind(AssertUnwindSafe(|| {
        let preview = guard.enter(assets, Some(state), frame)?;
        if guard.abort(abort)? {
            guard.force_fatal = false;
            return Ok(cancelled());
        }
        preflight(state, frame.raw(), policy)?;
        guard.recovery_scope_entered = true;
        guard.force_fatal = false;
        if guard.abort(abort)? {
            return Ok(cancelled());
        }
        let context = guard.inspected.expect("successful inspection");
        let submission = latch
            .select(
                &context,
                sys::FjSubmission {
                    instance_token: 0,
                    frame_token: 0,
                    submission_id: 0,
                    upload_core_hash: state.upload_core_hash(),
                    dir_hash: state.dir().hash(),
                    scanner_hash: state.scanner_hash(),
                    auto_exposure_hash: preview.hash,
                },
            )
            .inspect_err(|_| {
                guard.force_fatal = true;
            })?;
        guard.render_outcome_valid = true;
        let math = super::frame::prepare(state, frame.raw(), policy.raw())?;
        let descriptors = guard.owner.descriptors(&math)?;
        let noise = match guard.noise.as_ref() {
            Some(Ok(owner)) => Some(owner.view()),
            Some(Err(error)) => return Err(noise_failure(error)),
            None => None,
        };
        #[cfg(feature = "test-support")]
        let noise = noise.map(shorten_noise_view);
        let prepared = super::projection::host_data(
            state,
            &policy.raw().optics,
            &descriptors,
            &preview,
            noise.as_ref(),
        )
        .map_err(|diagnostic| Failure::new(sys::FJ_STATUS_PREPARATION_FAILURE, diagnostic))?;
        let host_preparation = super::host_preparation::Call::new();
        #[cfg(feature = "test-support")]
        observe_native_render();
        Ok(guard.owner.execute(
            &context,
            frame,
            &submission,
            &prepared,
            abort,
            &host_preparation,
        ))
    }))
    .unwrap_or_else(|_| {
        guard.force_fatal = true;
        Err(Failure::new(
            sys::FJ_STATUS_INTERNAL_FAILURE,
            "prepared render panicked",
        ))
    });
    guard.finish(result)
}

pub(super) enum UnavailableKind {
    MissingState,
    MissingInstance,
}

pub(super) fn reject_unavailable(
    owner: &mut OwnerBorrow<'_>,
    assets: &Assets,
    state: Option<&RenderState>,
    frame: &Frame<'_>,
    abort: &Abort<'_>,
    kind: UnavailableKind,
) -> Result<RenderOutcome, RenderError> {
    let mut guard = CallGuard::new(owner);
    let result = catch_unwind(AssertUnwindSafe(|| {
        guard.enter(assets, state, frame)?;
        if guard.abort(abort)? {
            guard.force_fatal = false;
            return Ok(cancelled());
        }
        Err(Failure::new(
            sys::FJ_STATUS_PREPARATION_FAILURE,
            match kind {
                UnavailableKind::MissingState => {
                    "FATAL: render state unavailable; cannot serve CUDA render"
                }
                UnavailableKind::MissingInstance => {
                    "FATAL: instance state missing; cannot serve CUDA render"
                }
            },
        ))
    }))
    .unwrap_or_else(|_| {
        Err(Failure::new(
            sys::FJ_STATUS_INTERNAL_FAILURE,
            "unavailable render panicked",
        ))
    });
    guard.finish(result)
}
