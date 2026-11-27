//! Projection of physical effects from the resolved frame and recipe.

use film_juicer_core::{
    effects::{self, frame::Frame},
    frame::Domain,
};
use super::{
    effects_projection::{project_dust, project_scratch},
    sys::{FjDefectOrigin, FjExtent, FjFilmEffects},
};

pub(super) fn project(v: &Frame, recipe: &effects::Recipe) -> FjFilmEffects {
    let extent = |e: Domain| FjExtent {
        x: e.x,
        y: e.y,
        width: e.width,
        height: e.height,
    };
    FjFilmEffects {
        film_dust: project_dust(recipe.film_dust()),
        gate_dust: project_dust(recipe.gate_dust()),
        film_scratch: project_scratch(recipe.film_scratch()),
        gate_scratch: project_scratch(recipe.gate_scratch()),
        flags: u32::from(v.weave_active())
            | (u32::from(v.film_active()) << 1)
            | (u32::from(v.gate_transmittance_active()) << 2)
            | (u32::from(v.gate_output_active()) << 3)
            | (u32::from(v.requires_full_frame()) << 4),
        render_extent: extent(v.render_extent()),
        full_frame_extent: extent(v.full_frame_extent()),
        origins: v.origins().map(|o| FjDefectOrigin {
            cell_x: o.cell_x,
            cell_y: o.cell_y,
            local_x_mm: o.local_x_mm,
            local_y_mm: o.local_y_mm,
        }),
        sample_step_x_mm: v.sample_step_x_mm(),
        sample_step_y_mm: v.sample_step_y_mm(),
        roi_offset_x: v.roi_offset_x(),
        roi_offset_y: v.roi_offset_y(),
        gate_width: v.gate_width(),
        gate_height: v.gate_height(),
        session_seed: v.session_seed(),
        clip_token: v.clip_token(),
        weave_dx_px: v.weave()[0],
        weave_dy_px: v.weave()[1],
        weave_cos_rot: v.weave()[2],
        weave_sin_rot: v.weave()[3],
        recipe_hash: v.recipe_hash(),
        hash: v.hash(),
    }
}

pub(super) fn empty() -> FjFilmEffects {
    super::edge::inactive_effects()
}
