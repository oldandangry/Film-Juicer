//! Callback-local frame descriptors from the admitted complete Rust state.

#![forbid(unsafe_code)]

use std::panic::{AssertUnwindSafe, catch_unwind};

use film_juicer_core::{
    effects,
    film::{dir, grain},
    frame::{Bounds, Domain},
    profile::Polarity,
    recipe,
    route::ScanRoute,
};

use super::{edge, sys};
use film_juicer_core::film::grain::film_v1::Sampling;
pub(super) fn film_grain(
    state: &recipe::RenderState,
    frame: &sys::FjFrame,
) -> Result<Option<sys::FjFilmGrain>, edge::Failure> {
    let Some(recipe) = state.film_grain() else {
        return Ok(None);
    };
    let rendered_extent = checked_extent(frame.full_frame_extent, "full_frame_extent")?;
    if recipe.raster().rendered_long_edge_px != rendered_extent.width.max(rendered_extent.height) {
        return Err(fail(
            sys::FJ_STATUS_UNSUPPORTED_INPUT,
            "Film v1 frame geometry does not match its prepared density bounds",
        ));
    }
    let (seed, next_seed, fraction) =
        film_seed_pair(frame.time_frames, frame.session_seed, frame.clip_token)?;
    let table = recipe.table();
    let records = std::array::from_fn(|index| {
        let source = &recipe.bank().records()[index];
        sys::FjFilmGrainRecord {
            minimum_gross: source.minimum_gross,
            maximum_gross: source.maximum_gross,
            offset: source.offset,
            mean_light: source.mean_light,
            mean_density: table.mean_density[index],
            correlation: table.correlation[index],
        }
    });
    let sums = recipe.bank().sums();
    let (aperture_grid, calibrated_border) = match recipe.sampling() {
        Sampling::Calibrated(border) => (None, border),
        Sampling::Integrated(grid) => (Some(grid), [0; 4]),
    };
    Ok(Some(sys::FjFilmGrain {
        bank_hash: recipe.bank().identity(),
        sums: sys::FjFloatSpan {
            data: sums.as_ptr(),
            count: sums.len(),
        },
        records,
        seed,
        next_seed,
        pitch: recipe.rendered_footprint_texels(),
        temporal_fraction: fraction,
        gain: recipe.gain(),
        chroma_fraction: recipe.chroma_fraction(),
        sample_pitch: table.footprint,
        calibrated_border,
        sample_extent: aperture_grid.map_or([0.0; 2], |grid| grid.extent),
        sample_step: aperture_grid.map_or([0.0; 2], |grid| grid.step),
        sample_origin: aperture_grid.map_or([0.0; 2], |grid| grid.origin),
    }))
}
fn film_seed_pair(time: f64, session: u64, clip: u64) -> Result<(u32, u32, f32), edge::Failure> {
    if !super::FILM_GRAIN_TIME_RANGE.contains(&time) {
        return Err(fail(
            sys::FJ_STATUS_UNSUPPORTED_INPUT,
            "Film v1 frame time must be finite and within the supported range",
        ));
    }
    let time_floor = time.floor();
    let mut index = time_floor as i64;
    let mut fraction = (time - time_floor) as f32;
    // A Float32 endpoint represents the next integral frame, not a blend.
    if fraction == 1.0 {
        index += 1;
        fraction = 0.0;
    }
    let seed = film_seed(session, clip, index);
    let next_seed = if fraction == 0.0 {
        seed
    } else {
        film_seed(session, clip, index + 1)
    };
    Ok((seed, next_seed, fraction))
}
fn film_seed(session: u64, clip: u64, index: i64) -> u32 {
    let mut word = (session ^ clip.rotate_left(17)).wrapping_add(
        (index as u64)
            .wrapping_add(1)
            .wrapping_mul(0x9e3779b97f4a7c15),
    );
    word = (word ^ (word >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    word = (word ^ (word >> 27)).wrapping_mul(0x94d049bb133111eb);
    (word ^ (word >> 31)) as u32
}

fn fail(category: u32, diagnostic: impl Into<String>) -> edge::Failure {
    edge::Failure::new(category, diagnostic)
}

fn checked_extent(rect: sys::FjRect, field: &str) -> Result<Domain, edge::Failure> {
    let result = catch_unwind(AssertUnwindSafe(|| {
        crate::cuda::geometry::checked_domain(Bounds {
            x1: rect.x1,
            y1: rect.y1,
            x2: rect.x2,
            y2: rect.y2,
        })
    }))
    .map_err(|_| {
        fail(
            sys::FJ_STATUS_INTERNAL_FAILURE,
            "frame geometry preparation panicked",
        )
    })?;
    result.map_err(|_| {
        fail(
            sys::FJ_STATUS_PREPARATION_FAILURE,
            format!("ResourceDescriptorMismatch phase=frame_descriptor field={field}"),
        )
    })
}

fn polarity(route: ScanRoute) -> Polarity {
    match route {
        ScanRoute::NegativeDirect | ScanRoute::NegativePrint => Polarity::Negative,
        ScanRoute::PositiveDirect | ScanRoute::PositivePrint => Polarity::Positive,
    }
}

#[expect(
    clippy::result_large_err,
    reason = "contain the core's bounded four-slot failure trace without a fallible heap owner"
)]
fn dir_math(
    state: &recipe::RenderState,
    frame: &sys::FjFrame,
    render: Domain,
    full: Domain,
) -> Result<sys::FjSpatialDir, edge::Failure> {
    let source = state.dir();
    let input = dir::frame::Input {
        active: source.active(),
        diffusion_size_um: source.diffusion_size_um(),
        diffusion_tail_um: source.diffusion_tail_um(),
        diffusion_tail_weight: source.diffusion_tail_weight(),
        pixel_size_um: frame.pixel_size_um,
        render_extent: render,
        full_frame_extent: full,
        recipe_hash: source.hash(),
    };
    let result = catch_unwind(AssertUnwindSafe(|| dir::frame::Plan::new(input))).map_err(|_| {
        fail(
            sys::FJ_STATUS_INTERNAL_FAILURE,
            "spatial DIR frame preparation panicked",
        )
    })?;
    match result {
        Ok(Some(plan)) => crate::cuda::dir_frame::projection(
            plan.recipe_hash(),
            plan.render_extent(),
            plan.full_frame_extent(),
            plan.component_count(),
            plan.components(),
        ),
        Ok(None) => Ok(crate::cuda::dir_frame::empty_math()),
        Err(error) => Err(fail(
            sys::FJ_STATUS_PREPARATION_FAILURE,
            crate::cuda::dir_frame::failure_diagnostic(error.field()),
        )),
    }
}

fn grain_math(
    state: &recipe::RenderState,
    frame: &sys::FjFrame,
    render: Domain,
    full: Domain,
) -> Result<sys::FjVisualGrain, edge::Failure> {
    let source = state.grain();
    if !source.active() {
        // Match the caller's inactive optional-descriptor omission.
        return Ok(crate::cuda::grain_frame::empty());
    }
    let development = state.development();
    let curves = development.density_curves_layers();
    let input = grain::frame::Input {
        active: source.active(),
        sublayers_active: source.sublayers_active(),
        layer_curves_required: development.layers_required(),
        capture_polarity: Some(polarity(development.route())),
        film_polarity: Some(development.polarity()),
        sublayer_count: source.sublayer_count(),
        pixel_size_um: frame.pixel_size_um,
        particle_area_um2: source.particle_area_um2(),
        particle_scale_cmy: *source.particle_scale_cmy(),
        particle_scale_layers: *source.particle_scale_layers(),
        particle_density_min_cmy: *source.particle_density_min_cmy(),
        authored_max_cmy: *development.maximum_cmy(),
        correlation_sigma_px: source.correlation_sigma_px(),
        dye_cloud_blur_um: source.dye_cloud_blur_um(),
        size_mix_scale: source.size_mix_scale(),
        fine_weight: source.fine_weight(),
        mid_weight: source.mid_weight(),
        coarse_weight: source.coarse_weight(),
        clump_morph_period_sec: source.clump_morph_period_sec(),
        render_extent: render,
        full_frame_extent: full,
        frame_time: frame.time_frames,
        frame_rate: frame.frame_rate,
        session_seed: frame.session_seed,
        clip_token: frame.clip_token,
        recipe_hash: source.hash(),
        recipe_layers_hash: source.layers_hash(),
        film_layers_hash: development.layers_hash(),
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        grain::frame::Plan::new(input, |layer, channel| {
            Ok::<&[f32], grain::frame::LayerAccessError>(
                curves.map_or(&[], |all| all[layer][channel].as_slice()),
            )
        })
    }))
    .map_err(|_| {
        fail(
            sys::FJ_STATUS_INTERNAL_FAILURE,
            "visual grain frame preparation panicked",
        )
    })?;
    match result {
        Ok(Some(plan)) => Ok(crate::cuda::grain_frame::projection(&plan, source)),
        Ok(None) => Ok(crate::cuda::grain_frame::empty()),
        Err(error) => {
            let raw = error.layer_access();
            let message = if raw {
                "visual grain frame arguments are invalid"
            } else {
                crate::cuda::grain_frame::failure_diagnostic(error.field())
            };
            Err(fail(
                if raw {
                    sys::FJ_STATUS_UNSUPPORTED_INPUT
                } else {
                    sys::FJ_STATUS_PREPARATION_FAILURE
                },
                message,
            ))
        }
    }
}

fn effects_math(
    state: &recipe::RenderState,
    frame: &sys::FjFrame,
    render: Domain,
) -> Result<sys::FjFilmEffects, edge::Failure> {
    let recipe = state.effects();
    if !recipe.active() {
        // Preserve the native caller's inactive omission: no frame math runs.
        return Ok(crate::cuda::effects_frame::empty());
    }
    let geometry = frame.effects_geometry;
    let input = effects::frame::Input {
        render_extent: render,
        pixel_definition: Domain {
            x: geometry.pixel_definition.x,
            y: geometry.pixel_definition.y,
            width: geometry.pixel_definition.width,
            height: geometry.pixel_definition.height,
        },
        pixel_size_um: frame.pixel_size_um,
        film_format_long_edge_mm: state.input().film_format_long_edge_mm(),
        canonical_x: geometry.canonical_x,
        canonical_y: geometry.canonical_y,
        canonical_width: geometry.canonical_width,
        canonical_height: geometry.canonical_height,
        scale_x: geometry.scale_x,
        scale_y: geometry.scale_y,
        pixel_aspect_ratio: geometry.pixel_aspect_ratio,
        frame_time: frame.time_frames,
        frame_rate: frame.frame_rate,
        session_seed: frame.session_seed,
        clip_token: frame.clip_token,
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        effects::frame::Frame::new(input, recipe)
    }))
    .map_err(|_| {
        fail(
            sys::FJ_STATUS_INTERNAL_FAILURE,
            "physical effects frame preparation panicked",
        )
    })?;
    match result {
        Ok(Some(value)) => Ok(crate::cuda::effects_frame::project(&value)),
        Ok(None) => Ok(crate::cuda::effects_frame::empty()),
        Err(effects::frame::Error::InactiveIdentity) => {
            Err(fail(sys::FJ_STATUS_PREPARATION_FAILURE, ""))
        }
        Err(effects::frame::Error::GeometryOrPolicy) => Err(fail(
            sys::FJ_STATUS_PREPARATION_FAILURE,
            "ResourceDescriptorMismatch phase=effects_descriptor field=geometry_or_policy",
        )),
    }
}

pub(super) fn prepare(
    state: &recipe::RenderState,
    frame: &sys::FjFrame,
) -> Result<edge::FrameDescriptors, edge::Failure> {
    let render = checked_extent(frame.render_window, "render_extent")?;
    let full = checked_extent(frame.full_frame_extent, "full_frame_extent")?;
    Ok(edge::FrameDescriptors {
        spatial_dir: dir_math(state, frame, render, full)?,
        grain: grain_math(state, frame, render, full)?,
        effects: effects_math(state, frame, render)?,
    })
}

#[cfg(test)]
mod tests {
    use super::{film_seed_pair, sys};

    fn assert_film_seed_pair(time: f64, expected: (u32, u32, u32)) {
        let (seed, next_seed, fraction) =
            film_seed_pair(time, 17, 23).unwrap_or_else(|error| panic!("{}", error.diagnostic));
        assert_eq!(
            (seed, next_seed, fraction.to_bits()),
            expected,
            "time={time}"
        );
    }

    #[test]
    fn film_seed_pair_carries_rounded_endpoints() {
        // Independent words from the C++ reference integer mapping.
        for (time, seed) in [
            (0.29 * 100.0, 2_010_141_349),
            (37.999_999_99, 857_065_782),
            (-1e-8, 3_099_808_472),
            (-1.000_000_01, 1_701_653_862),
            (1.0 - 2.0_f64.powi(-25), 1_854_463_680),
        ] {
            assert_film_seed_pair(time, (seed, seed, 0));
        }
    }

    #[test]
    fn film_seed_pair_preserves_integral_and_fractional_words() {
        let below_tie = (1.0_f64 - 2.0_f64.powi(-25)).next_down();
        for (time, seed, next_seed, fraction) in [
            (-1.0, 1_701_653_862, 1_701_653_862, 0),
            (-0.5, 1_701_653_862, 3_099_808_472, 0x3f00_0000),
            (-0.0, 3_099_808_472, 3_099_808_472, 0),
            (0.0, 3_099_808_472, 3_099_808_472, 0),
            (below_tie, 3_099_808_472, 1_854_463_680, 0x3f7f_ffff),
            (1.0, 1_854_463_680, 1_854_463_680, 0),
            (28.5, 1_138_692_069, 2_010_141_349, 0x3f00_0000),
            (28.999_99, 1_138_692_069, 2_010_141_349, 0x3f7f_ff58),
            (29.0, 2_010_141_349, 2_010_141_349, 0),
            (37.375, 2_044_520_963, 857_065_782, 0x3ec0_0000),
            (38.0, 857_065_782, 857_065_782, 0),
        ] {
            assert_film_seed_pair(time, (seed, next_seed, fraction));
        }
    }

    #[test]
    fn film_seed_pair_checks_original_time_at_both_limits() {
        let lower = -1_000_000.0_f64;
        let upper = 1_000_000.0_f64;
        assert_film_seed_pair(lower, (1_360_295_736, 1_360_295_736, 0));
        assert_film_seed_pair(upper, (134_912_488, 134_912_488, 0));
        for time in [
            lower.next_down(),
            upper.next_up(),
            f64::NAN,
            f64::NEG_INFINITY,
            f64::INFINITY,
        ] {
            let error = film_seed_pair(time, 17, 23).unwrap_err();
            assert_eq!(error.status.category, sys::FJ_STATUS_UNSUPPORTED_INPUT);
            assert!(
                error.diagnostic.starts_with("Film v1 frame time"),
                "time={time}: {}",
                error.diagnostic,
            );
        }
    }
}
