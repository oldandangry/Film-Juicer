//! Render sequencing over callback-local leases and existing complete owners.
#![forbid(unsafe_code)]

use std::fmt::{self, Write};

use film_juicer_core::{
    frame::{Bounds, Domain},
    route::ScanRoute,
    state,
};
use super::param_id::ParamId;
use super::host::{Callback, Status};
use crate::cuda;
use film_juicer_core::film::grain::grain_v2::{ProxyGrid, Raster, Selection};

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
struct CompleteFrameMapping {
    canonical: [f64; 4],
    candidate_full_grid: Option<ProxyGrid>,
}
struct AxisSupport {
    span: [f64; 2],
    empty: [bool; 2],
}
fn proxy_axis_support(
    bounds: [f64; 2],
    minimum: f64,
    extent: f64,
    scale: f64,
) -> Option<AxisSupport> {
    let [lower, upper] = bounds;
    let count = upper - lower;
    let origin = lower / scale - minimum;
    let step = scale.recip();
    let end = count.mul_add(step, origin);
    if count < 1.0
        || !origin.is_finite()
        || !end.is_finite()
        || !extent.is_finite()
        || extent <= 0.0
    {
        return None;
    }
    let edge = |i: f64| i.mul_add(step, origin).clamp(0.0, extent);
    let span = [edge(0.0), edge(count)];
    let empty = [edge(1.0) == span[0], edge(count - 1.0) == span[1]];
    // Match the kernel's clamped edges, including a positive neighbor for
    // each empty end. Native repeats this support check for raw callers.
    (span[0] < span[1]
        && (!empty[0] || (count > 1.0 && edge(2.0) > span[0]))
        && (!empty[1] || (count > 1.0 && edge(count - 2.0) < span[1])))
        .then_some(AxisSupport { span, empty })
}
fn complete_frame_mapping(
    args: &super::host::RenderArgs,
    source: &super::host::Image<'_>,
    destination: &super::host::Image<'_>,
    window: [i32; 4],
    canonical: [f64; 4],
) -> Option<CompleteFrameMapping> {
    let [scale_x, scale_y] = args.scale;
    let standard_scale =
        scale_x == scale_y && source.scale == args.scale && destination.scale == args.scale;
    let reduced = scale_x < 1.0 && scale_y < 1.0;
    let resized = reduced && source.scale == [1.0; 2] && destination.scale == [1.0; 2];
    if !(standard_scale || resized)
        || source.pixel_aspect_ratio != 1.0
        || destination.pixel_aspect_ratio != 1.0
        || destination.definition != source.definition
        || source.bounds != source.definition
        || destination.bounds != source.definition
        || window != source.definition
    {
        return None;
    }
    complete_frame_mapping_for_bounds(args.scale, source.definition, canonical, standard_scale)
}

fn complete_frame_mapping_for_bounds(
    scale: [f64; 2],
    definition: [i32; 4],
    canonical: [f64; 4],
    standard_scale: bool,
) -> Option<CompleteFrameMapping> {
    let [x1, y1, x2, y2] = canonical;
    let reduced = scale[0] < 1.0 && scale[1] < 1.0;
    if !canonical.into_iter().all(f64::is_finite)
        || x1 >= x2
        || y1 >= y2
        || !scale
            .into_iter()
            .all(|scale| scale.is_finite() && scale > 0.0)
    {
        return None;
    }
    let minimum = [x1.floor(), y1.floor()];
    let extent = [x2.ceil() - minimum[0], y2.ceil() - minimum[1]];
    for axis in 0..2 {
        let scale = scale[axis];
        let endpoints = [canonical[axis], canonical[axis + 2]];
        let scaled = endpoints.map(|endpoint| endpoint * scale);
        let outward = [scaled[0].floor(), scaled[1].ceil()];
        let observed = [f64::from(definition[axis]), f64::from(definition[axis + 2])];
        if !reduced {
            if observed != outward {
                return None;
            }
            continue;
        }
        let tight = std::array::from_fn(|edge| {
            let nearest = scaled[edge].round();
            if endpoints[edge] != 0.0 && nearest / endpoints[edge] == scale {
                nearest
            } else {
                outward[edge]
            }
        });
        let actual = proxy_axis_support(observed, minimum[axis], extent[axis], scale)?;
        let nominal = proxy_axis_support(outward, minimum[axis], extent[axis], scale);
        // Dropping a leading cell re-anchors the grid. Compare support at
        // BOTH ends after that change, even if the trailing bound did not move.
        let removable: [bool; 2] = std::array::from_fn(|edge| {
            (outward[edge] - tight[edge]).abs() == 1.0
                && nominal.as_ref().is_some_and(|nominal| {
                    nominal.empty[edge]
                        && actual.span[0] <= nominal.span[0]
                        && actual.span[1] >= nominal.span[1]
                })
        });
        if standard_scale {
            if !(0..2).all(|edge| {
                observed[edge] == outward[edge]
                    || (removable[edge] && observed[edge] == tight[edge])
            }) {
                return None;
            }
        } else if observed != tight {
            return None;
        }
    }
    Some(CompleteFrameMapping {
        canonical,
        candidate_full_grid: reduced.then(|| ProxyGrid {
            extent,
            step: scale.map(f64::recip),
            origin: std::array::from_fn(|axis| {
                f64::from(definition[axis]) / scale[axis] - minimum[axis]
            }),
            size: std::array::from_fn(|axis| definition[axis + 2] - definition[axis]),
        }),
    })
}
fn grain_v2_geometry_error(
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
        "Grain v2 requires complete square-pixel source and output images that agree with the action's canonical mapping. canonical={canonical:?} action_scale={:?} source_scale={:?} output_scale={:?} source_definition={:?} output_definition={:?} source_bounds={:?} output_bounds={:?} window={window:?} source_par={} output_par={} time={}",
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
fn grain_v2_time_error(callback: &Callback<'_>, time: f64) -> Status {
    let range = cuda::FILM_GRAIN_TIME_RANGE;
    let mut text = AdmissionText(String::new());
    if write!(
        text,
        "Grain v2 frame time must be finite and within {} to {} frames, inclusive. time={time}",
        range.start(),
        range.end(),
    )
    .is_ok()
    {
        callback.message(c"FilmJuicerFilmGrainTime", &text.0);
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

fn allocation_status(error: &state::Error) -> Option<Status> {
    let allocation = match error {
        state::Error::CaptureFailed(failure) => {
            failure.kind() == state::CaptureFailureKind::Allocation
        }
        state::Error::Preparation(error) => {
            failure_category(error) == cuda::sys::FJ_STATUS_ALLOCATION_FAILURE
        }
        _ => false,
    };
    allocation.then_some(Status::MEMORY)
}

fn cutoff(callback: &Callback<'_>, owner: &super::instance::Owner) -> Result<(), Status> {
    let access = callback.parameter_access(&owner.handles);
    let film = access.profile_key(ParamId::FilmProfileKey, b"kodak_portra_400")?;
    let print = access.profile_key(ParamId::PrintProfileKey, b"kodak_portra_endura")?;
    let capacity = 192usize
        .checked_add(film.len())
        .and_then(|size| size.checked_add(print.len()))
        .ok_or(Status::MEMORY)?;
    let mut diagnostic = Vec::new();
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
    let args = callback.render_args()?;
    if !args.cuda {
        return cutoff(callback, owner);
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
    let rendered_extent = domain(source.bounds)?;
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
    let film = input
        .as_ref()
        .map_or(Selection::Legacy, state::FrameInput::grain_v2);
    let film_active = film.controls().is_some();
    let film_thumbnail = film_active && callback.thumbnail(&owner.handles);
    let film_mapping = if film_active && !film_thumbnail {
        if !cuda::FILM_GRAIN_TIME_RANGE.contains(&args.time) {
            return Err(grain_v2_time_error(callback, args.time));
        }
        let canonical = callback.canonical_region(&owner.handles, args.time)?;
        let Some(mapping) = complete_frame_mapping(&args, source, destination, window, canonical)
        else {
            return Err(grain_v2_geometry_error(
                callback,
                &args,
                source,
                destination,
                window,
                canonical,
            ));
        };
        Some(mapping)
    } else {
        None
    };

    let film_frame = if film_thumbnail {
        film_juicer_core::recipe::GrainV2Frame::Thumbnail
    } else {
        film_juicer_core::recipe::GrainV2Frame::Raster(Raster {
            rendered_long_edge_px: rendered_extent.width.max(rendered_extent.height),
            candidate_full_grid: film_mapping
                .as_ref()
                .and_then(|mapping| mapping.candidate_full_grid),
        })
    };
    let film_canonical = film_mapping.map(|mapping| mapping.canonical);
    let admitted = match input {
        Ok(input) => owner.state.admit_frame(input, assets, film_frame),
        Err(error) => Err(error),
    };
    let admitted = match admitted {
        Ok(admitted) => admitted,
        Err(
            state::Error::Uninitialized | state::Error::RefreshRequired | state::Error::Capturing,
        ) => return Err(Status::FAILED),
        Err(error) => {
            if let Some(status) = allocation_status(&error) {
                return Err(status);
            }
            let mut text = AdmissionText(String::new());
            let diagnostic = write!(text, "{error}");
            let diagnostic = diagnostic.and_then(|()| {
                if matches!(
                    error,
                    state::Error::Preparation(
                        film_juicer_core::recipe::PrepareError::GrainV2Footprint { .. }
                    )
                ) {
                    write!(
                        text,
                        " (rendered={}x{}, action_scale={:?})",
                        rendered_extent.width, rendered_extent.height, args.scale
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
    let policy = cuda::FramePolicy::prepare(state, rendered_extent).map_err(|_| Status::FATAL)?;
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
        // Pre-resized preview images can report unit image scales. Dust and
        // scratches use the action scale when it describes the complete raster.

        let scale = if film.selected()
            && (film_canonical.is_some()
                || complete_frame_mapping(&args, source, destination, window, canonical).is_some())
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
    }
    let frame_rate = callback.frame_rate(&owner.handles)?;
    let frame_rate = if frame_rate.is_finite() && frame_rate > 0.0 {
        frame_rate
    } else {
        0.0
    };
    let (time, frame_token) = frame_position(args.time)?;
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

fn failure_category(error: &film_juicer_core::recipe::PrepareError) -> u32 {
    use crate::cuda::sys::*;
    use film_juicer_core::assets::{AssetError, GrainV2Error};
    use film_juicer_core::recipe::{self, PrepareError as E};
    match error {
        E::ContentsAllocation(_) => FJ_STATUS_ALLOCATION_FAILURE,
        E::Asset(AssetError::GrainV2(
            _,
            GrainV2Error::Allocation(_) | GrainV2Error::OwnerAllocation,
        )) => FJ_STATUS_ALLOCATION_FAILURE,
        E::Control(recipe::ControlError::GrainV2) | E::GrainV2Footprint { .. } => {
            FJ_STATUS_UNSUPPORTED_INPUT
        }
        E::Control(
            recipe::ControlError::InputColorSpace
            | recipe::ControlError::ReconstructionMethod
            | recipe::ControlError::Meter
            | recipe::ControlError::OutputColorSpace
            | recipe::ControlError::PrintStage,
        ) => FJ_STATUS_UNSUPPORTED_INPUT,
        E::Development(
            film_juicer_core::film::Error::InvalidGamma
            | film_juicer_core::film::Error::RouteMismatch,
        ) => FJ_STATUS_UNSUPPORTED_INPUT,
        E::Development(film_juicer_core::film::Error::Allocation)
        | E::ReferenceWhite(film_juicer_core::reconstruction::Error::AllocationFailure)
        | E::Dir {
            error:
                film_juicer_core::film::dir::Error::Allocation
                | film_juicer_core::film::dir::Error::Size,
            ..
        }
        | E::Tc(film_juicer_core::reconstruction::TcError::AllocationFailure)
        | E::PrintDevelopment(film_juicer_core::profile::PrintDensityError::Capacity) => {
            FJ_STATUS_ALLOCATION_FAILURE
        }
        E::Input(film_juicer_core::film::input::Error::Panic(_)) => FJ_STATUS_INTERNAL_FAILURE,
        // Assets errors are translated through the same existing asset boundary
        // before being wrapped, retaining their exact classification.
        E::Asset(_) => FJ_STATUS_INTERNAL_FAILURE,
        _ => FJ_STATUS_PREPARATION_FAILURE,
    }
}

#[cfg(test)]
mod tests {
    use super::frame_position;

    #[test]
    fn portra_allocation_status() {
        use super::{Status, allocation_status, state};
        use crate::ofx::sys::kOfxStatErrMemory;
        use film_juicer_core::assets::{AssetError, GrainV2Error, GrainV2Stock};
        use film_juicer_core::recipe::PrepareError;

        let status = |error| {
            allocation_status(&state::Error::Preparation(PrepareError::Asset(
                AssetError::GrainV2(GrainV2Stock::Portra400, error),
            )))
            .map(Status::code)
        };
        let reserve = Vec::<u8>::new().try_reserve_exact(usize::MAX).unwrap_err();
        assert_eq!(
            status(GrainV2Error::Allocation(reserve)),
            Some(kOfxStatErrMemory)
        );
        assert_eq!(
            status(GrainV2Error::OwnerAllocation),
            Some(kOfxStatErrMemory)
        );
        assert!(status(GrainV2Error::Invalid("word count")).is_none());
    }

    #[test]
    fn grain_v2_aperture_geometry_matches_native_wire_vectors() {
        use film_juicer_core::{
            assets::Assets,
            capture,
            film::grain::grain_v2::{Controls, ProxyGrid, Raster, Selection},
            recipe::{self, GrainV2Frame},
            route::ScanRoute,
        };
        use std::path::Path;

        let mut rows = include_str!("../../../../tests/grain/film-aperture-geometry.txt").lines();
        assert_eq!(rows.next(), Some("film-aperture-geometry-v1"));
        let assets = Assets::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Resources"));
        let mut count = 0;
        for row in rows {
            let mut fields = row.split_whitespace();
            let name = fields.next().unwrap();
            let print = fields.next().unwrap() == "1";
            let canonical: [f64; 4] =
                std::array::from_fn(|_| fields.next().unwrap().parse().unwrap());
            let bounds: [i32; 4] = std::array::from_fn(|_| fields.next().unwrap().parse().unwrap());
            let scale: [f64; 2] = std::array::from_fn(|_| fields.next().unwrap().parse().unwrap());
            let integrated = fields.next().unwrap() == "1";
            let pitch: f32 = fields.next().unwrap().parse().unwrap();
            let extent: [f64; 2] = std::array::from_fn(|_| fields.next().unwrap().parse().unwrap());
            let step: [f64; 2] = std::array::from_fn(|_| fields.next().unwrap().parse().unwrap());
            let origin: [f64; 2] = std::array::from_fn(|_| fields.next().unwrap().parse().unwrap());
            let border: [u8; 4] =
                std::array::from_fn(|_| fields.next().map_or(0, |s| s.parse().unwrap()));
            assert_eq!(fields.next(), None, "{name}");
            let size: [i32; 2] = std::array::from_fn(|axis| bounds[axis + 2] - bounds[axis]);
            let candidate_full_grid = if name == "border-direct" {
                // Exercise the settled explicit raster case for an empty leading cell.
                Some(ProxyGrid {
                    extent: [3840., 16.],
                    step: [4.; 2],
                    origin: [0., -4.],
                    size,
                })
            } else {
                super::complete_frame_mapping_for_bounds(
                    scale,
                    bounds,
                    canonical,
                    scale[0] == scale[1],
                )
                .unwrap_or_else(|| panic!("invalid OFX geometry for {name}"))
                .candidate_full_grid
            };
            let raster = Raster {
                rendered_long_edge_px: size[0].max(size[1]),
                candidate_full_grid,
            };
            let mut params = capture::defaults();
            params.film_profile_key = "kodak_portra_400".into();
            params.print_profile_key = "kodak_portra_endura".into();
            params.scan_route = if print {
                ScanRoute::NegativePrint
            } else {
                ScanRoute::NegativeDirect
            };
            params.camera_film_format_long_edge_mm = 36.0;
            params.grain_v2 = Selection::GrainV2(Controls {
                amount_ev: 0.0,
                texel_pitch_um: 1.0,
                chroma_fraction: 1.0,
            });
            let state = recipe::prepare_for_frame(&assets, &params, GrainV2Frame::Raster(raster))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            use crate::cuda::{self, sys};
            let [x1, y1, x2, y2] = bounds;
            let rect = sys::FjRect { x1, y1, x2, y2 };
            let image = sys::FjImage {
                address: 0,
                bounds: rect,
                row_bytes: 0,
                components: 4,
                depth: 0,
            };
            let raw = sys::FjFrame {
                source: image,
                destination: image,
                render_window: rect,
                full_frame_extent: rect,
                stream: 0,
                flags: 0,
                pixel_size_um: 0.,
                time_frames: 37.375,
                frame_rate: 24.,
                session_seed: 42,
                clip_token: 0x1234,
                effects_geometry: sys::FjEffectsGeometry {
                    pixel_definition: sys::FjExtent {
                        x: x1,
                        y: y1,
                        width: size[0],
                        height: size[1],
                    },
                    canonical_x: canonical[0],
                    canonical_y: canonical[1],
                    canonical_width: canonical[2] - canonical[0],
                    canonical_height: canonical[3] - canonical[1],
                    scale_x: scale[0],
                    scale_y: scale[1],
                    pixel_aspect_ratio: 1.,
                },
            };
            let film = cuda::projected_grain_v2(&state, &raw)
                .unwrap_or_else(|| panic!("Film ABI projection inactive for {name}"));
            assert_eq!(
                (film.seed, film.next_seed),
                (1_764_135_136, 195_565_897),
                "{name}"
            );
            assert_eq!(film.temporal_fraction, 0.375, "{name}");
            assert_eq!(film.pitch, 37.5, "{name}");
            assert_eq!(film.sample_pitch, pitch, "{name}");
            assert_eq!(film.calibrated_border, border, "{name}");
            assert_eq!(film.sample_extent, extent, "{name}");
            assert_eq!(film.sample_step, step, "{name}");
            assert_eq!(film.sample_origin, origin, "{name}");
            assert_eq!(film.sample_step[0] > 0., integrated, "{name}");
            count += 1;
        }
        assert_eq!(count, 8);
    }

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
        let rows_ids: Vec<_> = IntoIterator::into_iter(cases.map(|(id, _, _)| id))
            .map(|id| AsRef::<str>::as_ref(&id).to_owned())
            .collect();
        assert!(!rows_ids.is_empty(), "ofx-frame-positions");
        let rows_count = rows_ids.len();
        let mut rows_remaining = rows_ids
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(rows_remaining.len(), rows_count, "ofx-frame-positions");
        for (id, bits, expected) in cases {
            let row = AsRef::<str>::as_ref(&(id)).to_owned();
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
            assert!(
                rows_remaining.remove(&row),
                "unexpected or repeated comparison: {row}"
            );
        }
        assert!(
            rows_remaining.is_empty(),
            "missing comparisons: {rows_remaining:?}"
        );
    }
}
