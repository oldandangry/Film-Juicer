//! SF_TEMP_BRIDGE_FRAME_MATH: callback-local late frame descriptors; remove with
//! the final native preparation consumer, no later than S6.D.
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

fn fail(category: u32, diagnostic: impl Into<String>) -> edge::Failure {
    edge::Failure::new(category, diagnostic)
}

fn checked_extent(rect: sys::FjRect, field: &str) -> Result<Domain, edge::Failure> {
    let result = catch_unwind(AssertUnwindSafe(|| {
        crate::frame_bridge::checked_domain(Bounds {
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

fn grain_coefficients(recipe: &grain::Recipe) -> edge::FjGrainCoefficients {
    let finite = recipe.axis_finite_cmy();
    let flags = u32::from(recipe.active())
        | (u32::from(recipe.sublayers_active()) << 1)
        | (u32::from(finite[0]) << 3)
        | (u32::from(finite[1]) << 4)
        | (u32::from(finite[2]) << 5);
    edge::FjGrainCoefficients {
        flags,
        debug_view: recipe.debug_view() as u32,
        sublayer_count: recipe.sublayer_count() as usize,
        micro_structure: *recipe.micro_structure(),
        clump_temporal_mix: recipe.clump_temporal_mix(),
        size_mix_scale: recipe.size_mix_scale(),
        amplitude: recipe.amplitude(),
        chroma_mix: recipe.chroma_mix(),
        chroma_shared_weight: recipe.chroma_shared_weight(),
        chroma_independent_weight: recipe.chroma_independent_weight(),
        particle_density_min_cmy: *recipe.particle_density_min_cmy(),
        uniformity_cmy: *recipe.uniformity_cmy(),
        layer_axis_block_prefix_max: *recipe.axis_block_prefix_max_cmy(),
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
) -> Result<crate::dir_frame_bridge::FjDirFrameMath, edge::Failure> {
    let source = state.dir();
    #[cfg(feature = "test-support")]
    let fault = crate::dir_frame_bridge::dir_frame_faults::take();
    #[cfg(not(feature = "test-support"))]
    let fault = 0;
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
    let result = catch_unwind(AssertUnwindSafe(|| {
        crate::dir_frame_bridge::checked_plan(input, fault)
    }))
    .map_err(|_| {
        fail(
            sys::FJ_STATUS_INTERNAL_FAILURE,
            "spatial DIR frame preparation panicked",
        )
    })?;
    match result {
        Ok(Some(plan)) => Ok(crate::dir_frame_bridge::projection(
            plan.recipe_hash(),
            plan.render_extent(),
            plan.full_frame_extent(),
            plan.component_count(),
            plan.components(),
        )),
        Ok(None) => Ok(crate::dir_frame_bridge::empty_math()),
        Err(error) => Err(fail(
            sys::FJ_STATUS_PREPARATION_FAILURE,
            crate::dir_frame_bridge::failure_diagnostic(error.field()),
        )),
    }
}

fn grain_math(
    state: &recipe::RenderState,
    frame: &sys::FjFrame,
    render: Domain,
    full: Domain,
) -> Result<crate::grain_frame_bridge::FjGrainFrameMath, edge::Failure> {
    let source = state.grain();
    if !source.active() {
        // Match the caller's inactive optional-descriptor omission.
        return Ok(crate::grain_frame_bridge::empty_math());
    }
    #[cfg(feature = "test-support")]
    let fault = crate::grain_frame_bridge::faults::take();
    #[cfg(not(feature = "test-support"))]
    let fault = 0;
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
        crate::grain_frame_bridge::checked_plan(
            input,
            |layer, channel| {
                Ok::<&[f32], grain::frame::LayerAccessError>(
                    curves.map_or(&[], |all| all[layer][channel].as_slice()),
                )
            },
            fault,
        )
    }))
    .map_err(|_| {
        fail(
            sys::FJ_STATUS_INTERNAL_FAILURE,
            "visual grain frame preparation panicked",
        )
    })?;
    match result {
        Ok(Some(plan)) => Ok(crate::grain_frame_bridge::projection(&plan)),
        Ok(None) => Ok(crate::grain_frame_bridge::empty_math()),
        Err(error) => {
            let raw = error.layer_access();
            let message = if raw {
                "visual grain frame arguments are invalid"
            } else {
                crate::grain_frame_bridge::failure_diagnostic(error.field())
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
) -> Result<crate::effects_frame_bridge::FjEffectsFrameMath, edge::Failure> {
    let recipe = state.effects();
    if !recipe.active() {
        // Preserve the native caller's inactive omission: no frame math runs.
        return Ok(crate::effects_frame_bridge::empty_math());
    }
    #[cfg(feature = "test-support")]
    let fault = crate::effects_frame_bridge::faults::take();
    #[cfg(not(feature = "test-support"))]
    let fault = 0;
    let geometry = frame.effects_geometry;
    let input = effects::frame::Input {
        active: recipe.active(),
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
        gate_weave_amount: recipe.gate_weave_amount(),
        session_seed: frame.session_seed,
        clip_token: frame.clip_token,
        recipe_hash: recipe.hash(),
        film_dust: effects::frame::DustInput::from(recipe.film_dust()),
        film_scratch: effects::frame::ScratchInput::from(recipe.film_scratch()),
        gate_dust: effects::frame::DustInput::from(recipe.gate_dust()),
        gate_scratch: effects::frame::ScratchInput::from(recipe.gate_scratch()),
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        crate::effects_frame_bridge::checked_frame(input, fault)
    }))
    .map_err(|_| {
        fail(
            sys::FJ_STATUS_INTERNAL_FAILURE,
            "physical effects frame preparation panicked",
        )
    })?;
    match result {
        Ok(Some(value)) => Ok(crate::effects_frame_bridge::project(&value)),
        Ok(None) => Ok(crate::effects_frame_bridge::empty_math()),
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
    policy: &edge::FjPreparedFramePolicy,
) -> Result<edge::FjFrameMathInput, edge::Failure> {
    let render = checked_extent(frame.render_window, "render_extent")?;
    let full = checked_extent(frame.full_frame_extent, "full_frame_extent")?;
    let _ = policy;
    let dir = dir_math(state, frame, render, full)?;
    let grain = grain_math(state, frame, render, full)?;
    let effects = effects_math(state, frame, render)?;
    let grain_recipe = state.grain();
    let effects_recipe = state.effects();

    Ok(edge::FjFrameMathInput {
        dir,
        grain,
        effects,
        grain_coefficients: grain_coefficients(grain_recipe),
        film_dust: crate::film_recipe_bridge::project_dust(effects_recipe.film_dust()),
        gate_dust: crate::film_recipe_bridge::project_dust(effects_recipe.gate_dust()),
        film_scratch: crate::film_recipe_bridge::project_scratch(effects_recipe.film_scratch()),
        gate_scratch: crate::film_recipe_bridge::project_scratch(effects_recipe.gate_scratch()),
    })
}
