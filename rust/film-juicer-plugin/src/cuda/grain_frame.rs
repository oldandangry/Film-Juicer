//! Projection of the admitted visual grain plan.

use std::mem::{align_of, size_of};
use film_juicer_core::film::grain::frame::{Field, Gaussian, Plan};
use film_juicer_core::frame::Domain;
use film_juicer_core::profile::Polarity;
use crate::cuda::sys::{FjExtent, FjGrainGaussian};

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjGrainFrameMath {
    pub(crate) present: u32,
    pub(crate) capture_polarity: u32,
    pub(crate) render_extent: FjExtent,
    pub(crate) full_frame_extent: FjExtent,
    pub(crate) pixel_size_um: f32,
    pub(crate) frame0: i64,
    pub(crate) frame_alpha: f32,
    pub(crate) seed_base: u64,
    pub(crate) seed_base_next: u64,
    pub(crate) session_seed: u64,
    pub(crate) clip_token: u64,
    pub(crate) pitch_px: i32,
    pub(crate) breathing_period_frames: i32,
    pub(crate) clump_morph_period_frames: i32,
    pub(crate) wang_cell_mm: f32,
    pub(crate) breathing_amplitude: f32,
    pub(crate) breathing_cell_um_small: f32,
    pub(crate) breathing_cell_um_large: f32,
    pub(crate) breathing_mix: f32,
    pub(crate) breathing_drift_um_per_frame: f32,
    pub(crate) debug_scale: f32,
    pub(crate) density_max_cmy: [f32; 3],
    pub(crate) n_particles_cmy: [f32; 3],
    pub(crate) od_particle_cmy: [f32; 3],
    pub(crate) density_min_layers: [[f32; 3]; 3],
    pub(crate) density_max_layers: [[f32; 3]; 3],
    pub(crate) n_particles_layers: [[f32; 3]; 3],
    pub(crate) od_particle_layers: [[f32; 3]; 3],
    pub(crate) correlation: [FjGrainGaussian; 3],
    pub(crate) dye_cloud: [[FjGrainGaussian; 3]; 3],
    pub(crate) size_mix_weights: [f32; 3],
    pub(crate) size_mix_gain: f32,
    pub(crate) requires_full_frame: u32,
    pub(crate) recipe_hash: u64,
    pub(crate) density_layers_hash: u64,
}
const EMPTY_EXTENT: FjExtent = FjExtent {
    x: 0,
    y: 0,
    width: 0,
    height: 0,
};

const _: () = {
    assert!(size_of::<FjGrainFrameMath>() == 552 && align_of::<FjGrainFrameMath>() == 8);
    assert!(std::mem::offset_of!(FjGrainFrameMath, present) == 0);
    assert!(std::mem::offset_of!(FjGrainFrameMath, capture_polarity) == 4);
    assert!(std::mem::offset_of!(FjGrainFrameMath, render_extent) == 8);
    assert!(std::mem::offset_of!(FjGrainFrameMath, full_frame_extent) == 24);
    assert!(std::mem::offset_of!(FjGrainFrameMath, pixel_size_um) == 40);
    assert!(std::mem::offset_of!(FjGrainFrameMath, frame0) == 48);
    assert!(std::mem::offset_of!(FjGrainFrameMath, frame_alpha) == 56);
    assert!(std::mem::offset_of!(FjGrainFrameMath, seed_base) == 64);
    assert!(std::mem::offset_of!(FjGrainFrameMath, seed_base_next) == 72);
    assert!(std::mem::offset_of!(FjGrainFrameMath, session_seed) == 80);
    assert!(std::mem::offset_of!(FjGrainFrameMath, clip_token) == 88);
    assert!(std::mem::offset_of!(FjGrainFrameMath, pitch_px) == 96);
    assert!(std::mem::offset_of!(FjGrainFrameMath, breathing_period_frames) == 100);
    assert!(std::mem::offset_of!(FjGrainFrameMath, clump_morph_period_frames) == 104);
    assert!(std::mem::offset_of!(FjGrainFrameMath, wang_cell_mm) == 108);
    assert!(std::mem::offset_of!(FjGrainFrameMath, breathing_amplitude) == 112);
    assert!(std::mem::offset_of!(FjGrainFrameMath, breathing_cell_um_small) == 116);
    assert!(std::mem::offset_of!(FjGrainFrameMath, breathing_cell_um_large) == 120);
    assert!(std::mem::offset_of!(FjGrainFrameMath, breathing_mix) == 124);
    assert!(std::mem::offset_of!(FjGrainFrameMath, breathing_drift_um_per_frame) == 128);
    assert!(std::mem::offset_of!(FjGrainFrameMath, debug_scale) == 132);
    assert!(std::mem::offset_of!(FjGrainFrameMath, density_max_cmy) == 136);
    assert!(std::mem::offset_of!(FjGrainFrameMath, n_particles_cmy) == 148);
    assert!(std::mem::offset_of!(FjGrainFrameMath, od_particle_cmy) == 160);
    assert!(std::mem::offset_of!(FjGrainFrameMath, density_min_layers) == 172);
    assert!(std::mem::offset_of!(FjGrainFrameMath, density_max_layers) == 208);
    assert!(std::mem::offset_of!(FjGrainFrameMath, n_particles_layers) == 244);
    assert!(std::mem::offset_of!(FjGrainFrameMath, od_particle_layers) == 280);
    assert!(std::mem::offset_of!(FjGrainFrameMath, correlation) == 320);
    assert!(std::mem::offset_of!(FjGrainFrameMath, dye_cloud) == 368);
    assert!(std::mem::offset_of!(FjGrainFrameMath, size_mix_weights) == 512);
    assert!(std::mem::offset_of!(FjGrainFrameMath, size_mix_gain) == 524);
    assert!(std::mem::offset_of!(FjGrainFrameMath, requires_full_frame) == 528);
    assert!(std::mem::offset_of!(FjGrainFrameMath, recipe_hash) == 536);
    assert!(std::mem::offset_of!(FjGrainFrameMath, density_layers_hash) == 544);
};
const _: fn(FjGrainFrameMath) = |v| {
    let _: u32 = v.present;
    let _: u32 = v.capture_polarity;
    let _: FjExtent = v.render_extent;
    let _: FjExtent = v.full_frame_extent;
    let _: f32 = v.pixel_size_um;
    let _: i64 = v.frame0;
    let _: f32 = v.frame_alpha;
    let _: u64 = v.seed_base;
    let _: u64 = v.seed_base_next;
    let _: u64 = v.session_seed;
    let _: u64 = v.clip_token;
    let _: i32 = v.pitch_px;
    let _: i32 = v.breathing_period_frames;
    let _: i32 = v.clump_morph_period_frames;
    let _: f32 = v.wang_cell_mm;
    let _: f32 = v.breathing_amplitude;
    let _: f32 = v.breathing_cell_um_small;
    let _: f32 = v.breathing_cell_um_large;
    let _: f32 = v.breathing_mix;
    let _: f32 = v.breathing_drift_um_per_frame;
    let _: f32 = v.debug_scale;
    let _: [f32; 3] = v.density_max_cmy;
    let _: [f32; 3] = v.n_particles_cmy;
    let _: [f32; 3] = v.od_particle_cmy;
    let _: [[f32; 3]; 3] = v.density_min_layers;
    let _: [[f32; 3]; 3] = v.density_max_layers;
    let _: [[f32; 3]; 3] = v.n_particles_layers;
    let _: [[f32; 3]; 3] = v.od_particle_layers;
    let _: [FjGrainGaussian; 3] = v.correlation;
    let _: [[FjGrainGaussian; 3]; 3] = v.dye_cloud;
    let _: [f32; 3] = v.size_mix_weights;
    let _: f32 = v.size_mix_gain;
    let _: u32 = v.requires_full_frame;
    let _: u64 = v.recipe_hash;
    let _: u64 = v.density_layers_hash;
};
const EMPTY_GAUSSIAN: FjGrainGaussian = FjGrainGaussian {
    sigma_px: 0.0,
    radius: 0,
    hash: 0,
};
const EMPTY_MATH: FjGrainFrameMath = FjGrainFrameMath {
    present: 0,
    capture_polarity: 0,
    render_extent: EMPTY_EXTENT,
    full_frame_extent: EMPTY_EXTENT,
    pixel_size_um: 0.0,
    frame0: 0,
    frame_alpha: 0.0,
    seed_base: 0,
    seed_base_next: 0,
    session_seed: 0,
    clip_token: 0,
    pitch_px: 0,
    breathing_period_frames: 0,
    clump_morph_period_frames: 0,
    wang_cell_mm: 0.0,
    breathing_amplitude: 0.0,
    breathing_cell_um_small: 0.0,
    breathing_cell_um_large: 0.0,
    breathing_mix: 0.0,
    breathing_drift_um_per_frame: 0.0,
    debug_scale: 0.0,
    density_max_cmy: [0.0; 3],
    n_particles_cmy: [0.0; 3],
    od_particle_cmy: [0.0; 3],
    density_min_layers: [[0.0; 3]; 3],
    density_max_layers: [[0.0; 3]; 3],
    n_particles_layers: [[0.0; 3]; 3],
    od_particle_layers: [[0.0; 3]; 3],
    correlation: [EMPTY_GAUSSIAN; 3],
    dye_cloud: [[EMPTY_GAUSSIAN; 3]; 3],
    size_mix_weights: [0.0; 3],
    size_mix_gain: 0.0,
    requires_full_frame: 0,
    recipe_hash: 0,
    density_layers_hash: 0,
};

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
pub(crate) fn projection(p: &Plan) -> FjGrainFrameMath {
    FjGrainFrameMath {
        present: 1,
        capture_polarity: match p.capture_polarity() {
            Polarity::Negative => 0,
            Polarity::Positive => 1,
        },
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
        requires_full_frame: u32::from(p.requires_full_frame()),
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

pub(crate) fn empty_math() -> FjGrainFrameMath {
    EMPTY_MATH
}
