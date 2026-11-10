//! Render sequencing over callback-local leases and existing complete owners.
#![forbid(unsafe_code)]

use std::fmt::{self, Write};

use film_juicer_core::{
    frame::{Bounds, Domain},
    route::ScanRoute,
    state,
};
use super::host::{Callback, Status};
use crate::cuda;

// Admission delivery is best-effort. Formatting must not introduce an
// infallible allocation before the fallible host-message copy.
struct AdmissionText(String);

impl Write for AdmissionText {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        self.0.try_reserve(text.len()).map_err(|_| fmt::Error)?;
        self.0.push_str(text);
        Ok(())
    }
}

fn domain(bounds: [i32; 4]) -> Result<Domain, Status> {
    Domain::from_bounds(Bounds {
        x1: bounds[0],
        y1: bounds[1],
        x2: bounds[2],
        y2: bounds[3],
    })
    .map_err(|_| Status::FATAL)
}

fn frame_position(time: f64) -> Result<(f64, u64), Status> {
    let time = if time.is_finite() { time } else { 0.0 };
    // Binary64 INT64_MAX rounds to the excluded upper endpoint. These exact
    // powers of two define the floor-to-i64 domain without a saturating cast.
    if !(-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&time) {
        return Err(Status::FATAL);
    }
    Ok((time, time.floor() as i64 as u64))
}

fn allocation(error: &state::Error) -> bool {
    match error {
        state::Error::CaptureFailed(failure) => {
            failure.kind() == state::CaptureFailureKind::Allocation
        }
        state::Error::Preparation(error) => {
            crate::preparation::failure_category(error) == cuda::sys::FJ_STATUS_ALLOCATION_FAILURE
        }
        _ => false,
    }
}

fn cutoff(callback: &Callback<'_>, owner: &super::instance::Owner) -> Result<(), Status> {
    let access = callback.parameter_access(&owner.handles);
    let film = access.profile_key(c"FilmProfileKey", b"kodak_portra_400")?;
    let print = access.profile_key(c"PrintProfileKey", b"kodak_portra_endura")?;
    let capacity = 192usize
        .checked_add(film.len())
        .and_then(|size| size.checked_add(print.len()))
        .ok_or(Status::MEMORY)?;
    let mut diagnostic = Vec::new();
    #[cfg(feature = "test-support")]
    if super::host::test_support::render_fault(3) {
        return Err(Status::MEMORY);
    }
    diagnostic
        .try_reserve_exact(capacity)
        .map_err(|_| Status::MEMORY)?;
    diagnostic.extend_from_slice(b"FATAL: SpektrafilmPixelPipelineNotImplementedForPhase1A at JuicerEffect::render backend=non-cuda filmProfileKey=");
    diagnostic.extend_from_slice(&film);
    diagnostic.extend_from_slice(b" printProfileKey=");
    diagnostic.extend_from_slice(&print);
    // The historical cutoff constructs its diagnostic even when logging is off.
    // Its owned bytes expire here; no message, capture or image work follows.
    cuda::edge::trace_cutoff(&diagnostic);
    Err(Status::FATAL)
}

pub(super) fn run(callback: &Callback<'_>) -> Result<(), Status> {
    let owner = callback.instance()?;
    owner.require_live()?;
    #[cfg(feature = "test-support")]
    super::host::test_support::begin_render();
    let args = callback.render_args()?;
    if !args.cuda {
        return cutoff(callback, owner);
    }
    #[cfg(feature = "test-support")]
    if super::host::test_support::render_fault(4) {
        panic!("OFX render action panic");
    }
    // Declaration order makes Output release before Source on every return.
    let source = callback.fetch_image(&owner.handles, 0, args.time)?;
    let destination = callback.fetch_image(&owner.handles, 1, args.time)?;
    let (Some(source), Some(destination)) = (&source, &destination) else {
        return Ok(());
    };
    let trace_flags = cuda::edge::trace_flags();
    if !source.float_path {
        return Err(Status::FATAL);
    }
    let mut window = args.window;
    if window[0] == window[2] && window[1] == window[3] {
        window = source.bounds;
    }
    if window[2] <= window[0] || window[3] <= window[1] {
        return Ok(());
    }
    let full = domain(source.bounds)?;
    #[cfg(feature = "test-support")]
    crate::frame_observer::record_full_domain(full);
    let assets = owner.loaded.runtime().assets();
    let mut admitted = owner.state.admit(assets);
    if matches!(
        admitted,
        Err(state::Error::Uninitialized | state::Error::RefreshRequired)
    ) {
        owner.render_capture(
            callback,
            matches!(admitted, Err(state::Error::Uninitialized)),
        )?;
        admitted = owner.state.admit(assets);
    }
    let admitted = match admitted {
        Ok(admitted) => admitted,
        Err(
            state::Error::Uninitialized | state::Error::RefreshRequired | state::Error::Capturing,
        ) => return Err(Status::FAILED),
        Err(error) if allocation(&error) => return Err(Status::MEMORY),
        Err(error) => {
            let mut text = AdmissionText(String::new());
            if write!(text, "{error}").is_ok() {
                callback.message(c"FilmJuicerRenderAdmission", &text.0);
            }
            return Err(Status::FATAL);
        }
    };
    let state = admitted.state();
    let print = matches!(
        state.development().route(),
        ScanRoute::NegativePrint | ScanRoute::PositivePrint
    );
    if print && window != source.bounds {
        return Err(Status::FATAL);
    }
    let policy = cuda::FramePolicy::prepare(state, full).map_err(|_| Status::FATAL)?;
    #[cfg(feature = "test-support")]
    policy.observe_callback(state.development().route() as u32);
    let mut effects = cuda::sys::FjEffectsGeometry {
        pixel_definition: cuda::sys::FjExtent {
            x: 0,
            y: 0,
            width: 0,
            height: 0,
        },
        canonical_x: 0.0,
        canonical_y: 0.0,
        canonical_width: 0.0,
        canonical_height: 0.0,
        scale_x: 0.0,
        scale_y: 0.0,
        pixel_aspect_ratio: 0.0,
    };
    if state.effects().active() {
        let canonical = callback.canonical_region(&owner.handles, args.time)?;
        let definition = domain(source.definition)?;
        effects = cuda::sys::FjEffectsGeometry {
            pixel_definition: cuda::sys::FjExtent {
                x: definition.x,
                y: definition.y,
                width: definition.width,
                height: definition.height,
            },
            canonical_x: canonical[0],
            canonical_y: canonical[1],
            canonical_width: canonical[2] - canonical[0],
            canonical_height: canonical[3] - canonical[1],
            scale_x: source.scale[0],
            scale_y: source.scale[1],
            pixel_aspect_ratio: source.pixel_aspect_ratio,
        };
        #[cfg(feature = "test-support")]
        crate::frame_observer::record_effects_geometry(effects);
    }
    let frame_rate = callback.frame_rate(&owner.handles)?;
    let frame_rate = if frame_rate.is_finite() && frame_rate > 0.0 {
        frame_rate
    } else {
        0.0
    };
    let (time, frame_token) = frame_position(args.time)?;
    #[cfg(feature = "test-support")]
    super::host::test_support::render_state(&admitted);
    callback.render_native(
        owner,
        state,
        source,
        destination,
        &args,
        window,
        &policy,
        time,
        frame_token,
        frame_rate,
        effects,
        trace_flags,
    )
}

#[cfg(test)]
mod tests {
    use super::frame_position;
    use crate::test_cases::RowCases;

    #[test]
    fn frame_positions() {
        // Independently frozen binary64 inputs and literal token expectations.
        // No out-of-domain native conversion is executed as an oracle.
        let cases = [
            ("lower_outside", 0xc3e0_0000_0000_0001, None),
            (
                "lower_boundary",
                0xc3e0_0000_0000_0000,
                Some(0x8000_0000_0000_0000),
            ),
            (
                "lower_inside",
                0xc3df_ffff_ffff_ffff,
                Some(0x8000_0000_0000_0400),
            ),
            (
                "upper_inside",
                0x43df_ffff_ffff_ffff,
                Some(0x7fff_ffff_ffff_fc00),
            ),
            ("upper_boundary", 0x43e0_0000_0000_0000, None),
            ("upper_outside", 0x43e0_0000_0000_0001, None),
            ("finite_max", 0x7fef_ffff_ffff_ffff, None),
            ("finite_min", 0xffef_ffff_ffff_ffff, None),
            ("positive_zero", 0x0000_0000_0000_0000, Some(0)),
            ("negative_zero", 0x8000_0000_0000_0000, Some(0)),
            ("negative_half", 0xbfe0_0000_0000_0000, Some(u64::MAX)),
            (
                "below_negative_one",
                0xbff0_0000_0000_0001,
                Some(u64::MAX - 1),
            ),
            ("below_positive_one", 0x3fef_ffff_ffff_ffff, Some(0)),
            ("nan", 0x7ff8_0000_0000_0000, Some(0)),
            ("positive_infinity", 0x7ff0_0000_0000_0000, Some(0)),
            ("negative_infinity", 0xfff0_0000_0000_0000, Some(0)),
        ];
        let mut rows = RowCases::new("ofx-frame-positions", cases.map(|(id, _, _)| id));
        for (id, bits, expected) in cases {
            let row = rows.case(id);
            {
                let input = f64::from_bits(bits);
                let actual = frame_position(input);
                match expected {
                    None => assert_eq!(
                        actual.unwrap_err().code(),
                        super::super::sys::kOfxStatErrFatal
                    ),
                    Some(token) => {
                        let (time, actual) = actual.unwrap();
                        assert_eq!(actual, token);
                        assert_eq!(time.to_bits(), if input.is_finite() { bits } else { 0 });
                    }
                }
            }
            row.passed();
        }
        rows.finish();
    }
}
