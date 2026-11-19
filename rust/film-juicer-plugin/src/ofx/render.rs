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

#[cfg(feature = "grain-preview")]
fn complete_frame_mapping(
    args: &super::host::RenderArgs,
    source: &super::host::Image<'_>,
    destination: &super::host::Image<'_>,
    window: [i32; 4],
    canonical: [f64; 4],
) -> bool {
    let [x1, y1, x2, y2] = canonical;
    let [scale_x, scale_y] = args.scale;
    let standard_scale =
        scale_x == scale_y && source.scale == args.scale && destination.scale == args.scale;
    let resized =
        scale_x < 1.0 && scale_y < 1.0 && source.scale == [1.0; 2] && destination.scale == [1.0; 2];
    // A dimension-derived action scale can multiply back one ulp past an
    // integer edge. Recognize that exact quotient on pre-resized images;
    // genuinely fractional edges still need outward rounding.
    let edge = |canonical: f64, scale: f64, outward: fn(f64) -> f64| {
        let scaled = canonical * scale;
        let nearest = scaled.round();
        if resized && canonical != 0.0 && nearest / canonical == scale {
            nearest
        } else {
            outward(scaled)
        }
    };
    let [left, top, right, bottom] = source.definition;
    canonical.into_iter().all(f64::is_finite)
        && x1 < x2
        && y1 < y2
        && scale_x.is_finite()
        && scale_y.is_finite()
        && scale_x > 0.0
        && scale_y > 0.0
        && (standard_scale || resized)
        && source.pixel_aspect_ratio == 1.0
        && destination.pixel_aspect_ratio == 1.0
        && edge(x1, scale_x, f64::floor) == f64::from(left)
        && edge(x2, scale_x, f64::ceil) == f64::from(right)
        && edge(y1, scale_y, f64::floor) == f64::from(top)
        && edge(y2, scale_y, f64::ceil) == f64::from(bottom)
        && destination.definition == source.definition
        && source.bounds == source.definition
        && destination.bounds == source.definition
        && window == source.definition
}

#[cfg(feature = "grain-preview")]
fn film_geometry_error(
    callback: &Callback<'_>,
    args: &super::host::RenderArgs,
    source: &super::host::Image<'_>,
    destination: &super::host::Image<'_>,
    window: [i32; 4],
    canonical: [f64; 4],
) -> Status {
    let mut text = AdmissionText(String::new());
    if write!(
        text,
        "Film v1 requires complete square-pixel source and output images that agree with the action's canonical mapping. canonical={canonical:?} action_scale={:?} source_scale={:?} output_scale={:?} source_definition={:?} output_definition={:?} source_bounds={:?} output_bounds={:?} window={window:?} source_par={} output_par={} time={}",
        args.scale,
        source.scale,
        destination.scale,
        source.definition,
        destination.definition,
        source.bounds,
        destination.bounds,
        source.pixel_aspect_ratio,
        destination.pixel_aspect_ratio,
        args.time,
    )
    .is_ok()
    {
        callback.message(c"FilmJuicerFilmGrainGeometry", &text.0);
    }
    Status::FAILED
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
    let mut input = owner.state.frame_input();
    if matches!(
        input,
        Err(state::Error::Uninitialized | state::Error::RefreshRequired)
    ) {
        let captured =
            owner.render_capture(callback, matches!(input, Err(state::Error::Uninitialized)))?;
        input = match captured {
            Some(input) => Ok(input),
            None => owner.state.frame_input(),
        };
    }
    if matches!(input, Err(state::Error::Capturing)) {
        let captured = owner.render_read(callback)?;
        input = match captured {
            Some(input) => Ok(input),
            None => owner.state.frame_input(),
        };
    }
    #[cfg(feature = "grain-preview")]
    let film_selected = input
        .as_ref()
        .is_ok_and(|input| input.film_grain_selected());
    #[cfg(feature = "grain-preview")]
    let film_thumbnail = film_selected && callback.thumbnail(&owner.handles);
    #[cfg(feature = "grain-preview")]
    let film_canonical = if film_selected && !film_thumbnail {
        let canonical = callback.canonical_region(&owner.handles, args.time)?;
        if !complete_frame_mapping(&args, source, destination, window, canonical)
            || !(-1_000_000.0..=1_000_000.0).contains(&args.time)
        {
            return Err(film_geometry_error(
                callback,
                &args,
                source,
                destination,
                window,
                canonical,
            ));
        }
        Some(canonical)
    } else {
        None
    };
    #[cfg(feature = "grain-preview")]
    let preview_frame = if film_thumbnail {
        film_juicer_core::recipe::PreviewFrame::Thumbnail
    } else {
        use film_juicer_core::film::grain::preview::{ProxyGrid, Raster};
        let full_grid = film_canonical
            .filter(|_| args.scale.into_iter().all(|scale| scale < 1.0))
            .map(|canonical| {
                let minimum = [canonical[0].floor(), canonical[1].floor()];
                let extent = std::array::from_fn(|axis| canonical[axis + 2].ceil() - minimum[axis]);
                let step = args.scale.map(f64::recip);
                let origin = std::array::from_fn(|axis| {
                    f64::from(source.definition[axis]) / args.scale[axis] - minimum[axis]
                });
                ProxyGrid {
                    extent,
                    step,
                    origin,
                }
            });
        film_juicer_core::recipe::PreviewFrame::Raster(Raster {
            long_edge_px: full.width.max(full.height),
            full_grid,
        })
    };
    #[cfg(not(feature = "grain-preview"))]
    let film_canonical = None;
    #[cfg(feature = "test-support")]
    let input_identity = input.as_ref().map_or(0, state::FrameInput::input_identity);
    let admitted = match input {
        Ok(input) => owner.state.admit_frame(
            input,
            assets,
            #[cfg(feature = "grain-preview")]
            preview_frame,
        ),
        Err(error) => Err(error),
    };
    let admitted = match admitted {
        Ok(admitted) => admitted,
        Err(
            state::Error::Uninitialized | state::Error::RefreshRequired | state::Error::Capturing,
        ) => return Err(Status::FAILED),
        Err(error) if allocation(&error) => return Err(Status::MEMORY),
        Err(error) => {
            let mut text = AdmissionText(String::new());
            let diagnostic = write!(text, "{error}");
            #[cfg(feature = "grain-preview")]
            let diagnostic = diagnostic.and_then(|()| {
                if matches!(
                    error,
                    state::Error::Preparation(
                        film_juicer_core::recipe::PrepareError::FilmGrainFootprint { .. }
                    )
                ) {
                    write!(
                        text,
                        " (rendered={}x{}, action_scale={:?})",
                        full.width, full.height, args.scale
                    )
                } else {
                    Ok(())
                }
            });
            if diagnostic.is_ok() {
                callback.message(c"FilmJuicerRenderAdmission", &text.0);
            }
            return Err(Status::FATAL);
        }
    };
    let state = admitted.as_ref();
    let print = matches!(
        state.development().route(),
        ScanRoute::NegativePrint | ScanRoute::PositivePrint
    );
    if print && window != source.bounds {
        return Err(Status::FATAL);
    }
    if state.optics().highlight_boost().is_some()
        && (source.definition != source.bounds
            || destination.bounds != source.bounds
            || window != source.bounds)
    {
        callback.message(
            c"FilmJuicerHighlightBoost",
            "Highlight reconstruction requires the full source region, output image and render window.",
        );
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
        let canonical = match film_canonical {
            Some(region) => region,
            None => callback.canonical_region(&owner.handles, args.time)?,
        };
        let scale = source.scale;
        #[cfg(feature = "grain-preview")]
        let scale = if film_canonical.is_some()
            || complete_frame_mapping(&args, source, destination, window, canonical)
        {
            args.scale
        } else {
            scale
        };
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
            scale_x: scale[0],
            scale_y: scale[1],
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
    super::host::test_support::render_state(&admitted, input_identity, &owner.state);
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
