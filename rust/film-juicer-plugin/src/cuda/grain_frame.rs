//! Projection of the resolved Legacy grain plan and its recipe.

use film_juicer_core::{
    film::grain::{
        Recipe,
        frame::{Field, Gaussian, Plan},
    },
    frame::Domain,
};
use super::sys::{self, FjExtent, FjGrainGaussian, FjVisualGrain};

fn extent(v: Domain) -> FjExtent {
    FjExtent {
        x: v.x,
        y: v.y,
        width: v.width,
        height: v.height,
    }
}
fn gaussian(v: &Gaussian) -> FjGrainGaussian {
    FjGrainGaussian {
        sigma_px: v.sigma_px(),
        radius: v.radius(),
        hash: v.hash(),
    }
}
pub(super) fn projection(p: &Plan, recipe: &Recipe) -> FjVisualGrain {
    let finite = recipe.axis_finite_cmy();
    FjVisualGrain {
        flags: sys::FJ_GRAIN_ACTIVE
            | (u32::from(recipe.sublayers_active()) * sys::FJ_GRAIN_SUBLAYERS)
            | (u32::from(p.requires_full_frame()) * sys::FJ_GRAIN_FULL_FRAME)
            | (u32::from(finite[0]) * sys::FJ_GRAIN_AXIS_FINITE_C)
            | (u32::from(finite[1]) * sys::FJ_GRAIN_AXIS_FINITE_M)
            | (u32::from(finite[2]) * sys::FJ_GRAIN_AXIS_FINITE_Y),
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
        render_extent: extent(p.render_extent()),
        full_frame_extent: extent(p.full_frame_extent()),
        pixel_size_um: p.pixel_size_um(),
        frame0: p.frame0(),
        frame_alpha: p.frame_alpha(),
        seed_base: p.seed_base(),
        seed_base_next: p.seed_base_next(),
        session_seed: p.session_seed(),
        clip_token: p.clip_token(),
        pitch_px: p.pitch_px(),
        breathing_period_frames: p.breathing_period_frames(),
        clump_morph_period_frames: p.clump_morph_period_frames(),
        wang_cell_mm: p.wang_cell_mm(),
        breathing_amplitude: p.breathing_amplitude(),
        breathing_cell_um_small: p.breathing_cell_um_small(),
        breathing_cell_um_large: p.breathing_cell_um_large(),
        breathing_mix: p.breathing_mix(),
        breathing_drift_um_per_frame: p.breathing_drift_um_per_frame(),
        debug_scale: p.debug_scale(),
        density_max_cmy: *p.density_max_cmy(),
        n_particles_cmy: *p.n_particles_cmy(),
        od_particle_cmy: *p.od_particle_cmy(),
        density_min_layers: *p.density_min_layers(),
        density_max_layers: *p.density_max_layers(),
        n_particles_layers: *p.n_particles_layers(),
        od_particle_layers: *p.od_particle_layers(),
        correlation: p.correlation().each_ref().map(gaussian),
        dye_cloud: p
            .dye_cloud()
            .each_ref()
            .map(|row| row.each_ref().map(gaussian)),
        size_mix_weights: *p.size_mix_weights(),
        size_mix_gain: p.size_mix_gain(),
        recipe_hash: p.recipe_hash(),
        density_layers_hash: p.density_layers_hash(),
    }
}
const MESSAGES: [&str; 14] = [
    "",
    "ResourceDescriptorMismatch phase=grain_descriptor field=frame_input",
    "ResourceDescriptorMismatch phase=grain_descriptor field=density_layers",
    "UnsupportedMode phase=grain_descriptor field=frame_time",
    "UnsupportedMode phase=grain_descriptor field=period_frames",
    "ResourceDescriptorMismatch phase=grain_descriptor field=pixel_area",
    "ResourceDescriptorMismatch phase=grain_descriptor field=particle_model",
    "ResourceDescriptorMismatch phase=grain_descriptor field=particle_count",
    "ResourceDescriptorMismatch phase=grain_descriptor field=size_mix",
    "MalformedRequiredProfileData phase=grain_descriptor field=density_layers",
    "MalformedRequiredProfileData phase=grain_descriptor field=density_layer_total",
    "ResourceDescriptorMismatch phase=grain_descriptor field=layer_particle_area",
    "ResourceDescriptorMismatch phase=grain_descriptor field=layer_particle_count",
    "UnsupportedMode phase=grain_descriptor field=partial_frame",
];

pub(crate) fn failure_diagnostic(field: Field) -> &'static str {
    MESSAGES[field as usize]
}

pub(super) fn empty() -> FjVisualGrain {
    super::edge::inactive_grain()
}
