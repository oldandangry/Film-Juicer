//! SF_TEMP_BRIDGE_GRAIN_FRAME: fixed synchronous mathematical transport.
//! Remove at its last native model caller by S6.D; S4.E narrows execution inputs.

use std::mem::{align_of, size_of};
use std::panic::{AssertUnwindSafe, catch_unwind};
use film_juicer_core::film::grain::frame::{self, Field, Gaussian, LayerAccessError, Plan};
use film_juicer_core::frame::Domain;
use film_juicer_core::profile::Polarity;
use crate::cuda::sys::{FjErrorBuffer, FjExtent, FjFloatSpan, FjGrainGaussian, FjStatus};

/// The shared checked grain-plan operation. The layer reader stays lazy: the
/// injected layer panic occurs only when the consumer requests a real cell.
pub(crate) fn checked_plan<'a, F>(
    input: frame::Input,
    mut read_layer: F,
    fault: u32,
) -> Result<Option<Plan>, frame::Error>
where
    F: FnMut(usize, usize) -> Result<&'a [f32], LayerAccessError>,
{
    #[cfg(feature = "test-support")]
    if fault == 1 {
        panic!("visual grain frame fixture entry panic");
    }
    let plan = Plan::new(input, |row, channel| {
        #[cfg(feature = "test-support")]
        if fault == 2 {
            panic!("visual grain frame fixture layer panic");
        }
        read_layer(row, channel)
    });
    let _ = fault;
    plan
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjGrainFrameInput {
    pub(crate) active: u32,
    pub(crate) sublayers_active: u32,
    pub(crate) layer_curves_required: u32,
    pub(crate) capture_polarity: u32,
    pub(crate) film_polarity: u32,
    pub(crate) sublayer_count: i32,
    pub(crate) pixel_size_um: f32,
    pub(crate) particle_area_um2: f32,
    pub(crate) particle_scale_cmy: [f32; 3],
    pub(crate) particle_scale_layers: [f32; 3],
    pub(crate) particle_density_min_cmy: [f32; 3],
    pub(crate) authored_max_cmy: [f32; 3],
    pub(crate) correlation_sigma_px: f32,
    pub(crate) dye_cloud_blur_um: f32,
    pub(crate) size_mix_scale: f32,
    pub(crate) fine_weight: f32,
    pub(crate) mid_weight: f32,
    pub(crate) coarse_weight: f32,
    pub(crate) clump_morph_period_sec: f32,
    pub(crate) render_extent: FjExtent,
    pub(crate) full_frame_extent: FjExtent,
    pub(crate) frame_time: f64,
    pub(crate) frame_rate: f64,
    pub(crate) session_seed: u64,
    pub(crate) clip_token: u64,
    pub(crate) recipe_hash: u64,
    pub(crate) recipe_layers_hash: u64,
    pub(crate) film_layers_hash: u64,
    pub(crate) layer_curves: [[FjFloatSpan; 3]; 3],
}
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
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjGrainFrameFailure {
    pub(crate) reason: u32,
    pub(crate) field: u32,
    pub(crate) layer: u32,
    pub(crate) channel: u32,
}
const EMPTY_EXTENT: FjExtent = FjExtent {
    x: 0,
    y: 0,
    width: 0,
    height: 0,
};

const _: () = {
    assert!(size_of::<FjGrainFrameInput>() == 344 && align_of::<FjGrainFrameInput>() == 8);
    assert!(std::mem::offset_of!(FjGrainFrameInput, active) == 0);
    assert!(std::mem::offset_of!(FjGrainFrameInput, sublayers_active) == 4);
    assert!(std::mem::offset_of!(FjGrainFrameInput, layer_curves_required) == 8);
    assert!(std::mem::offset_of!(FjGrainFrameInput, capture_polarity) == 12);
    assert!(std::mem::offset_of!(FjGrainFrameInput, film_polarity) == 16);
    assert!(std::mem::offset_of!(FjGrainFrameInput, sublayer_count) == 20);
    assert!(std::mem::offset_of!(FjGrainFrameInput, pixel_size_um) == 24);
    assert!(std::mem::offset_of!(FjGrainFrameInput, particle_area_um2) == 28);
    assert!(std::mem::offset_of!(FjGrainFrameInput, particle_scale_cmy) == 32);
    assert!(std::mem::offset_of!(FjGrainFrameInput, particle_scale_layers) == 44);
    assert!(std::mem::offset_of!(FjGrainFrameInput, particle_density_min_cmy) == 56);
    assert!(std::mem::offset_of!(FjGrainFrameInput, authored_max_cmy) == 68);
    assert!(std::mem::offset_of!(FjGrainFrameInput, correlation_sigma_px) == 80);
    assert!(std::mem::offset_of!(FjGrainFrameInput, dye_cloud_blur_um) == 84);
    assert!(std::mem::offset_of!(FjGrainFrameInput, size_mix_scale) == 88);
    assert!(std::mem::offset_of!(FjGrainFrameInput, fine_weight) == 92);
    assert!(std::mem::offset_of!(FjGrainFrameInput, mid_weight) == 96);
    assert!(std::mem::offset_of!(FjGrainFrameInput, coarse_weight) == 100);
    assert!(std::mem::offset_of!(FjGrainFrameInput, clump_morph_period_sec) == 104);
    assert!(std::mem::offset_of!(FjGrainFrameInput, render_extent) == 108);
    assert!(std::mem::offset_of!(FjGrainFrameInput, full_frame_extent) == 124);
    assert!(std::mem::offset_of!(FjGrainFrameInput, frame_time) == 144);
    assert!(std::mem::offset_of!(FjGrainFrameInput, frame_rate) == 152);
    assert!(std::mem::offset_of!(FjGrainFrameInput, session_seed) == 160);
    assert!(std::mem::offset_of!(FjGrainFrameInput, clip_token) == 168);
    assert!(std::mem::offset_of!(FjGrainFrameInput, recipe_hash) == 176);
    assert!(std::mem::offset_of!(FjGrainFrameInput, recipe_layers_hash) == 184);
    assert!(std::mem::offset_of!(FjGrainFrameInput, film_layers_hash) == 192);
    assert!(std::mem::offset_of!(FjGrainFrameInput, layer_curves) == 200);
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
    assert!(size_of::<FjGrainFrameFailure>() == 16 && align_of::<FjGrainFrameFailure>() == 4);
    assert!(std::mem::offset_of!(FjGrainFrameFailure, reason) == 0);
    assert!(std::mem::offset_of!(FjGrainFrameFailure, field) == 4);
    assert!(std::mem::offset_of!(FjGrainFrameFailure, layer) == 8);
    assert!(std::mem::offset_of!(FjGrainFrameFailure, channel) == 12);
};
const _: fn(FjGrainFrameInput) = |v| {
    let _: u32 = v.active;
    let _: u32 = v.sublayers_active;
    let _: u32 = v.layer_curves_required;
    let _: u32 = v.capture_polarity;
    let _: u32 = v.film_polarity;
    let _: i32 = v.sublayer_count;
    let _: f32 = v.pixel_size_um;
    let _: f32 = v.particle_area_um2;
    let _: [f32; 3] = v.particle_scale_cmy;
    let _: [f32; 3] = v.particle_scale_layers;
    let _: [f32; 3] = v.particle_density_min_cmy;
    let _: [f32; 3] = v.authored_max_cmy;
    let _: f32 = v.correlation_sigma_px;
    let _: f32 = v.dye_cloud_blur_um;
    let _: f32 = v.size_mix_scale;
    let _: f32 = v.fine_weight;
    let _: f32 = v.mid_weight;
    let _: f32 = v.coarse_weight;
    let _: f32 = v.clump_morph_period_sec;
    let _: FjExtent = v.render_extent;
    let _: FjExtent = v.full_frame_extent;
    let _: f64 = v.frame_time;
    let _: f64 = v.frame_rate;
    let _: u64 = v.session_seed;
    let _: u64 = v.clip_token;
    let _: u64 = v.recipe_hash;
    let _: u64 = v.recipe_layers_hash;
    let _: u64 = v.film_layers_hash;
    let _: [[FjFloatSpan; 3]; 3] = v.layer_curves;
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
const _: fn(FjGrainFrameFailure) = |v| {
    let _: u32 = v.reason;
    let _: u32 = v.field;
    let _: u32 = v.layer;
    let _: u32 = v.channel;
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
const EMPTY_FAILURE: FjGrainFrameFailure = FjGrainFrameFailure {
    reason: 0,
    field: 0,
    layer: 3,
    channel: 3,
};

fn valid<T>(pointer: *const T) -> bool {
    !pointer.is_null()
        && (pointer as usize).is_multiple_of(align_of::<T>())
        && size_of::<T>() <= isize::MAX as usize
        && (pointer as usize).checked_add(size_of::<T>()).is_some()
}
struct Diagnostic {
    record: *mut FjErrorBuffer,
    data: *mut u8,
    capacity: usize,
}

impl Diagnostic {
    /// # Safety
    /// Optional initialized record and nonempty backing are live/exclusive/disjoint.
    unsafe fn admit(record: *mut FjErrorBuffer) -> Result<Self, ()> {
        let mut writer = Self {
            record,
            data: std::ptr::null_mut(),
            capacity: 0,
        };
        if record.is_null() {
            return Ok(writer);
        }
        if !valid(record) {
            return Err(());
        }
        // SAFETY: Independent numeric admission precedes authorized initialized record access.
        unsafe {
            (*record).length = 0;
            writer.capacity = (*record).capacity;
            if writer.capacity != 0 {
                writer.data = (*record).data.cast();
                if writer.data.is_null()
                    || writer.capacity > isize::MAX as usize
                    || (writer.data as usize)
                        .checked_add(writer.capacity)
                        .is_none()
                {
                    return Err(());
                }
                writer.data.write(0);
            }
        }
        Ok(writer)
    }

    fn write(&mut self, text: &str) {
        let count = text.len().min(self.capacity.saturating_sub(1));
        // SAFETY: Admitted live/exclusive backing has capacity; count leaves room for NUL.
        unsafe {
            if self.capacity != 0 {
                std::ptr::copy_nonoverlapping(text.as_ptr(), self.data, count);
                self.data.add(count).write(0);
            }
            if !self.record.is_null() {
                (*self.record).length = count;
            }
        }
    }
}

fn domain(v: FjExtent) -> Domain {
    Domain {
        x: v.x,
        y: v.y,
        width: v.width,
        height: v.height,
    }
}
fn extent(v: Domain) -> FjExtent {
    FjExtent {
        x: v.x,
        y: v.y,
        width: v.width,
        height: v.height,
    }
}
fn polarity(tag: u32) -> Option<Polarity> {
    match tag {
        0 => Some(Polarity::Negative),
        1 => Some(Polarity::Positive),
        _ => None,
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
fn status(category: u32) -> FjStatus {
    FjStatus {
        category,
        api: 0,
        native_code: 0,
    }
}

/// # Safety
/// Authorized nonnull peers are aligned, initialized, live, exclusive and disjoint.
unsafe fn clear(out: *mut FjGrainFrameMath, failure: *mut FjGrainFrameFailure) {
    if valid(out) {
        // SAFETY: Independent numeric admission and the caller authorize this complete write.
        unsafe {
            out.write(EMPTY_MATH);
        }
    }
    if valid(failure) {
        // SAFETY: Failure storage is independently writable despite rejection of another peer.
        unsafe {
            failure.write(EMPTY_FAILURE);
        }
    }
}

/// # Safety
/// Nonempty span backing is initialized, aligned, live and immutable for `'a`.
/// The caller confines the borrow to synchronous preparation; it cannot escape.
unsafe fn layer<'a>(span: FjFloatSpan) -> Result<&'a [f32], LayerAccessError> {
    if span.count == 0 {
        return Ok(&[]);
    }
    let bytes = span
        .count
        .checked_mul(size_of::<f32>())
        .ok_or(LayerAccessError::Invalid)?;
    if !valid(span.data)
        || bytes > isize::MAX as usize
        || (span.data as usize).checked_add(bytes).is_none()
    {
        return Err(LayerAccessError::Invalid);
    }
    // SAFETY: Checked numeric extent plus caller's live immutable backing obligation.
    Ok(unsafe { std::slice::from_raw_parts(span.data, span.count) })
}

/// # Safety
/// Fixed input is initialized/live/aligned; all consumed nonempty curve spans are
/// initialized/live/immutable for this call. Outputs and optional diagnostics are
/// exclusive and disjoint from inputs and each other. No pointer survives return.
pub(crate) unsafe fn prepare(
    input: *const FjGrainFrameInput,
    out: *mut FjGrainFrameMath,
    failure: *mut FjGrainFrameFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // Consume once even when raw admission rejects, or the layer reader is never reached.
    #[cfg(feature = "test-support")]
    let fault = faults::take();
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Each peer is independently admitted under the entry's storage contract.
        unsafe {
            clear(out, failure);
        }
        // SAFETY: The optional initialized record and its exclusive backing obey the entry contract.
        let mut diagnostic = unsafe { Diagnostic::admit(error) };
        let admitted = valid(input) && valid(out) && valid(failure) && diagnostic.is_ok();
        let input = if admitted {
            // SAFETY: Admitted initialized input is authorized for a synchronous fixed copy.
            Some(unsafe { input.read() })
        } else {
            None
        };
        if !admitted || input.is_some_and(|v| v.active > 1) {
            if valid(failure) {
                // SAFETY: Independently admitted exclusive failure peer.
                unsafe {
                    failure.write(FjGrainFrameFailure {
                        reason: 1,
                        ..EMPTY_FAILURE
                    });
                }
            }
            if let Ok(writer) = &mut diagnostic {
                writer.write("visual grain frame arguments are invalid");
            }
            return status(2);
        }
        let input = input.unwrap();
        let result = checked_plan(
            frame::Input {
                active: input.active != 0,
                sublayers_active: input.sublayers_active != 0,
                layer_curves_required: input.sublayers_active <= 1
                    && input.layer_curves_required == 1,
                capture_polarity: polarity(input.capture_polarity),
                film_polarity: polarity(input.film_polarity),
                sublayer_count: input.sublayer_count,
                pixel_size_um: input.pixel_size_um,
                particle_area_um2: input.particle_area_um2,
                particle_scale_cmy: input.particle_scale_cmy,
                particle_scale_layers: input.particle_scale_layers,
                particle_density_min_cmy: input.particle_density_min_cmy,
                authored_max_cmy: input.authored_max_cmy,
                correlation_sigma_px: input.correlation_sigma_px,
                dye_cloud_blur_um: input.dye_cloud_blur_um,
                size_mix_scale: input.size_mix_scale,
                fine_weight: input.fine_weight,
                mid_weight: input.mid_weight,
                coarse_weight: input.coarse_weight,
                clump_morph_period_sec: input.clump_morph_period_sec,
                render_extent: domain(input.render_extent),
                full_frame_extent: domain(input.full_frame_extent),
                frame_time: input.frame_time,
                frame_rate: input.frame_rate,
                session_seed: input.session_seed,
                clip_token: input.clip_token,
                recipe_hash: input.recipe_hash,
                recipe_layers_hash: input.recipe_layers_hash,
                film_layers_hash: input.film_layers_hash,
            },
            |row, channel| {
                // SAFETY: Borrow is created only for the requested cell, after all earlier checks.
                unsafe { layer(input.layer_curves[row][channel]) }
            },
            {
                #[cfg(feature = "test-support")]
                {
                    fault
                }
                #[cfg(not(feature = "test-support"))]
                {
                    0
                }
            },
        );
        match result {
            Ok(plan) => {
                let wire = plan.as_ref().map_or(EMPTY_MATH, projection);
                #[cfg(feature = "test-support")]
                let wire = faults::corrupt(wire, fault);
                // SAFETY: All peers were admitted; complete pointer-free output commits once.
                unsafe {
                    out.write(wire);
                }
                status(0)
            }
            Err(rejected) => {
                // Reserved tags are raw failures only at their actually consumed stage.
                // Force an invalid sublayer tag to stop at identity, before any layer borrow.
                let reserved = match rejected.field() {
                    Field::FrameInput => {
                        input.recipe_hash != 0
                            && (input.capture_polarity > 2
                                || (input.capture_polarity < 2 && input.film_polarity > 2))
                    }
                    Field::DensityLayers => {
                        input.sublayers_active > 1 || input.layer_curves_required > 1
                    }
                    _ => false,
                };
                let raw = reserved || rejected.layer_access();
                let details = if reserved {
                    FjGrainFrameFailure {
                        reason: 1,
                        ..EMPTY_FAILURE
                    }
                } else {
                    FjGrainFrameFailure {
                        reason: if raw { 1 } else { 2 },
                        field: rejected.field() as u32,
                        layer: rejected.layer() as u32,
                        channel: rejected.channel() as u32,
                    }
                };
                // SAFETY: Independently admitted exclusive failure peer, never a partial math result.
                unsafe {
                    failure.write(details);
                }
                diagnostic.as_mut().unwrap().write(if raw {
                    "visual grain frame arguments are invalid"
                } else {
                    MESSAGES[rejected.field() as usize]
                });
                status(if raw { 2 } else { 3 })
            }
        }
    }));
    match result {
        Ok(result) => result,
        Err(_) => {
            // SAFETY: Recovery uses independent numeric admission and fixed bounded writes only.
            unsafe {
                clear(out, failure);
                if valid(failure) {
                    failure.write(FjGrainFrameFailure {
                        reason: 3,
                        ..EMPTY_FAILURE
                    });
                }
                if let Ok(mut writer) = Diagnostic::admit(error) {
                    writer.write("visual grain frame preparation panicked");
                }
            }
            status(8)
        }
    }
}

#[unsafe(no_mangle)]
/// # Safety
/// Same initialized/live/disjoint and synchronous borrowing contract as prepare.
unsafe extern "C" fn fj_legacy_grain_frame_prepare(
    input: *const FjGrainFrameInput,
    out: *mut FjGrainFrameMath,
    failure: *mut FjGrainFrameFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The single checked implementation contains admission and whole-operation unwind.
    unsafe { prepare(input, out, failure, error) }
}

#[cfg(feature = "test-support")]
pub(crate) mod faults {
    use std::cell::Cell;
    use super::{FjGrainFrameMath, FjStatus, status};
    thread_local! { static PENDING: Cell<u32> = const { Cell::new(0) }; }
    pub(crate) fn arm(fault: u32) -> FjStatus {
        if fault > 6 {
            return status(2);
        }
        PENDING.with(|pending| pending.set(fault));
        status(0)
    }
    pub(crate) fn take() -> u32 {
        PENDING.with(|pending| pending.replace(0))
    }
    pub(super) fn corrupt(mut wire: FjGrainFrameMath, fault: u32) -> FjGrainFrameMath {
        match fault {
            3 => wire.present = 2,
            4 => wire.recipe_hash ^= 1,
            5 => wire.correlation[0].radius = 76,
            6 => wire.requires_full_frame ^= 1,
            _ => {}
        }
        wire
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn input(curves: &[[[f32; 1]; 3]; 3]) -> FjGrainFrameInput {
        // SAFETY: Every member is a number or a nullable raw span; all-zero is
        // a valid initialized transport representation, without an owning pointer.
        let mut v: FjGrainFrameInput = unsafe { std::mem::zeroed() };
        v.active = 1;
        v.sublayers_active = 1;
        v.layer_curves_required = 1;
        v.sublayer_count = 3;
        v.pixel_size_um = 1.0;
        v.particle_area_um2 = 1.0;
        v.particle_scale_cmy = [1.0; 3];
        v.particle_scale_layers = [1.0; 3];
        v.particle_density_min_cmy = [0.1; 3];
        v.authored_max_cmy = [1.0; 3];
        v.size_mix_scale = 1.0;
        v.fine_weight = 1.0;
        v.clump_morph_period_sec = 8.0;
        v.render_extent = FjExtent {
            x: 0,
            y: 0,
            width: 64,
            height: 48,
        };
        v.full_frame_extent = v.render_extent;
        v.frame_rate = 24.0;
        v.recipe_hash = 1;
        v.recipe_layers_hash = 7;
        v.film_layers_hash = 7;
        for (layer, row) in curves.iter().enumerate() {
            for (channel, curve) in row.iter().enumerate() {
                v.layer_curves[layer][channel] = FjFloatSpan {
                    data: curve.as_ptr(),
                    count: 1,
                };
            }
        }
        v
    }
    #[test]
    fn r1_precedes_later_channel_total_and_clears_math() {
        let curves = [
            [[0.0], [0.0], [1.0]],
            [[1.0], [0.0], [1.0]],
            [[1.0], [0.0], [1.0]],
        ];
        let v = input(&curves);
        let mut out = EMPTY_MATH;
        out.present = 99;
        let mut failure = EMPTY_FAILURE;
        let mut bytes = [0_i8; 128];
        let mut error = FjErrorBuffer {
            data: bytes.as_mut_ptr(),
            capacity: bytes.len(),
            length: 99,
        };
        // SAFETY: Fixed inputs and all nine singleton backings remain initialized
        // and immutable; output/failure/diagnostics are live exclusive disjoint locals.
        let status = unsafe { prepare(&v, &mut out, &mut failure, &mut error) };
        assert_eq!((status.category, status.api, status.native_code), (3, 0, 0));
        assert_eq!(
            (
                failure.reason,
                failure.field,
                failure.layer,
                failure.channel
            ),
            (2, 12, 0, 0)
        );
        assert_eq!(out.present, 0);
        assert_eq!(out.n_particles_layers, [[0.0; 3]; 3]);
        assert_eq!(out.recipe_hash, 0);
        assert_eq!(error.length, MESSAGES[12].len());
        assert_eq!(
            &bytes[..error.length],
            MESSAGES[12]
                .as_bytes()
                .iter()
                .map(|b| *b as i8)
                .collect::<Vec<_>>()
        );
    }
    #[test]
    fn inactive_ignores_tags_and_rejection_clears_independent_peer() {
        let curves = [[[1.0]; 3]; 3];
        let mut v = input(&curves);
        v.active = 0;
        v.sublayers_active = 99;
        v.capture_polarity = 99;
        v.frame_time = f64::NAN;
        let mut out = EMPTY_MATH;
        let mut failure = EMPTY_FAILURE;
        // SAFETY: Live initialized fixed locals are exclusive/disjoint; ignored spans
        // remain legitimate backing; the optional diagnostic is absent.
        let status = unsafe { prepare(&v, &mut out, &mut failure, std::ptr::null_mut()) };
        assert_eq!(status.category, 0);
        assert_eq!(
            (
                failure.reason,
                failure.field,
                failure.layer,
                failure.channel
            ),
            (0, 0, 3, 3)
        );
        failure.field = 99;
        // SAFETY: The admitted failure peer is a live exclusive local; null peers
        // are rejected numerically without access, including a missing output.
        let status = unsafe {
            prepare(
                std::ptr::null(),
                std::ptr::null_mut(),
                &mut failure,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(status.category, 2);
        assert_eq!(
            (
                failure.reason,
                failure.field,
                failure.layer,
                failure.channel
            ),
            (1, 0, 3, 3)
        );
    }
}
