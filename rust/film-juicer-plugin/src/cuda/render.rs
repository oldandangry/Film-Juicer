//! Ordered rendering of one held complete state through borrowed native access.
#![forbid(unsafe_code)]

use film_juicer_core::{
    assets::{AssetError, Assets},
    film::input::Meter,
    recipe::RenderState,
};
use crate::{instance::Instance, submission};
use super::{
    edge::{self, Abort, Failure, Frame, FramePolicy},
    sys, Owner,
};

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

fn cancelled() -> (sys::FjRenderOutcome, String, bool) {
    (
        edge::failed_outcome(edge::status(sys::FJ_STATUS_CANCELLED)),
        String::new(),
        false,
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
    preflight_camera_lens_blur(state.optics().camera_lens_blur().sigma_um())
}

pub(super) fn preflight_camera_lens_blur(sigma_um: f32) -> Result<(), Failure> {
    if sigma_um > 0.0 {
        return Err(Failure::new(
            sys::FJ_STATUS_PREPARATION_FAILURE,
            "ExactOpticsNotImplementedForPhase6 component=camera_lens_blur",
        ));
    }
    Ok(())
}

fn noise_failure(error: &AssetError) -> Failure {
    let category = match error {
        AssetError::Noise(error) => crate::asset_noise::noise_category(error.kind()),
        _ => sys::FJ_STATUS_INTERNAL_FAILURE,
    };
    Failure::new(category, error.to_string())
}

pub(super) fn render(
    owner: &Owner,
    assets: &Assets,
    state: &RenderState,
    frame: &Frame<'_>,
    policy: &FramePolicy,
    instance: &Instance,
    abort: &Abort<'_>,
) -> Result<(sys::FjRenderOutcome, String, bool), Failure> {
    let grain_v2 = super::frame::grain_v2(state, frame.raw())?;
    let preview = edge::metering(frame, meter(state.input().meter()))?;
    let context = owner.inspect(frame)?;
    if abort.query()? {
        return Ok(cancelled());
    }
    preflight(state, frame.raw(), policy)?;
    if abort.query()? {
        return Ok(cancelled());
    }
    let selection = instance
        .select_submission(submission::Key {
            frame_token: frame.frame_token(),
            device_id: context.device_id,
            context: context.context,
            epoch: context.epoch,
            upload_core: state.upload_core_hash(),
            dir: state.dir().hash(),
            scanner: state.scanner_hash(),
            auto_exposure: preview.hash,
        })
        .map_err(|error| Failure::new(sys::FJ_STATUS_INTERNAL_FAILURE, error.to_string()))?;
    let reservation = &selection.reservation;
    let selected = sys::FjSubmission {
        instance_token: instance
            .identity()
            .expect("selected initialized identity")
            .instance_token,
        frame_token: reservation.key.frame_token,
        submission_id: reservation.id,
        upload_core_hash: reservation.key.upload_core,
        dir_hash: reservation.key.dir,
        scanner_hash: reservation.key.scanner,
        auto_exposure_hash: reservation.key.auto_exposure,
    };
    let descriptors = super::frame::prepare(state, frame.raw())?;
    let noise = state
        .legacy_grain()
        .active()
        .then(|| assets.noise())
        .transpose()
        .map_err(|error| noise_failure(&error))?;
    let noise = noise.map(crate::asset_noise::NoiseOwner::new);
    let noise_view = noise.as_ref().map(crate::asset_noise::NoiseOwner::view);
    let prepared = super::projection::host_data(
        state,
        &policy.raw().optics,
        &descriptors,
        &preview,
        noise_view.as_ref(),
    )
    .map_err(|diagnostic| Failure::new(sys::FJ_STATUS_PREPARATION_FAILURE, diagnostic))?;
    let prepared = sys::FjPreparedHostData {
        film_grain: grain_v2
            .as_ref()
            .map_or(std::ptr::null(), std::ptr::from_ref),
        ..prepared
    };
    let host_preparation = super::host_preparation::Call::new();
    let (outcome, mut diagnostic) = owner.execute(
        &context,
        frame,
        &selected,
        &prepared,
        abort,
        &host_preparation,
    );
    let mut fatal = false;
    if outcome.flags & sys::FJ_RENDER_ADMITTED != 0 {
        if let Err(error) = instance.publish_submission(reservation) {
            fatal = true;
            if diagnostic.is_empty() {
                diagnostic = error.to_string();
            }
        }
        edge::trace_submission(frame, &selected, selection.reused);
    }
    Ok((outcome, diagnostic, fatal))
}
