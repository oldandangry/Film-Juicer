//! Projection of the admitted effects frame.

use std::mem::{align_of, size_of};

use film_juicer_core::effects::frame::Frame;
use film_juicer_core::frame::Domain;
use crate::cuda::sys::{FjDefectOrigin, FjExtent};

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjEffectsFrameMath {
    pub(crate) present: u32,
    pub(crate) flags: u32,
    pub(crate) render_extent: FjExtent,
    pub(crate) full_frame_extent: FjExtent,
    pub(crate) origins: [FjDefectOrigin; 4],
    pub(crate) sample_step_x_mm: f32,
    pub(crate) sample_step_y_mm: f32,
    pub(crate) roi_offset_x: i32,
    pub(crate) roi_offset_y: i32,
    pub(crate) gate_width: i32,
    pub(crate) gate_height: i32,
    pub(crate) session_seed: u64,
    pub(crate) clip_token: u64,
    pub(crate) weave_dx_px: f32,
    pub(crate) weave_dy_px: f32,
    pub(crate) weave_cos_rot: f32,
    pub(crate) weave_sin_rot: f32,
    pub(crate) recipe_hash: u64,
    pub(crate) hash: u64,
}

const _: () = {
    assert!(size_of::<FjEffectsFrameMath>() == 208 && align_of::<FjEffectsFrameMath>() == 8);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, present) == 0);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, flags) == 4);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, render_extent) == 8);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, full_frame_extent) == 24);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, origins) == 40);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, sample_step_x_mm) == 136);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, sample_step_y_mm) == 140);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, roi_offset_x) == 144);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, roi_offset_y) == 148);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, gate_width) == 152);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, gate_height) == 156);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, session_seed) == 160);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, clip_token) == 168);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, weave_dx_px) == 176);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, weave_dy_px) == 180);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, weave_cos_rot) == 184);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, weave_sin_rot) == 188);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, recipe_hash) == 192);
    assert!(std::mem::offset_of!(FjEffectsFrameMath, hash) == 200);
};
const _: fn(FjEffectsFrameMath) = |v| {
    let _: u32 = v.present;
    let _: u32 = v.flags;
    let _: FjExtent = v.render_extent;
    let _: FjExtent = v.full_frame_extent;
    let _: [FjDefectOrigin; 4] = v.origins;
    let _: f32 = v.sample_step_x_mm;
    let _: f32 = v.sample_step_y_mm;
    let _: i32 = v.roi_offset_x;
    let _: i32 = v.roi_offset_y;
    let _: i32 = v.gate_width;
    let _: i32 = v.gate_height;
    let _: u64 = v.session_seed;
    let _: u64 = v.clip_token;
    let _: f32 = v.weave_dx_px;
    let _: f32 = v.weave_dy_px;
    let _: f32 = v.weave_cos_rot;
    let _: f32 = v.weave_sin_rot;
    let _: u64 = v.recipe_hash;
    let _: u64 = v.hash;
};
const EMPTY_EXTENT: FjExtent = FjExtent {
    x: 0,
    y: 0,
    width: 0,
    height: 0,
};
const EMPTY_ORIGIN: FjDefectOrigin = FjDefectOrigin {
    cell_x: 0,
    cell_y: 0,
    local_x_mm: 0.0,
    local_y_mm: 0.0,
};
const EMPTY_MATH: FjEffectsFrameMath = FjEffectsFrameMath {
    present: 0,
    flags: 0,
    render_extent: EMPTY_EXTENT,
    full_frame_extent: EMPTY_EXTENT,
    origins: [EMPTY_ORIGIN; 4],
    sample_step_x_mm: 0.0,
    sample_step_y_mm: 0.0,
    roi_offset_x: 0,
    roi_offset_y: 0,
    gate_width: 0,
    gate_height: 0,
    session_seed: 0,
    clip_token: 0,
    weave_dx_px: 0.0,
    weave_dy_px: 0.0,
    weave_cos_rot: 0.0,
    weave_sin_rot: 0.0,
    recipe_hash: 0,
    hash: 0,
};

pub(crate) fn project(v: &Frame) -> FjEffectsFrameMath {
    let extent = |e: Domain| FjExtent {
        x: e.x,
        y: e.y,
        width: e.width,
        height: e.height,
    };
    FjEffectsFrameMath {
        present: 1,
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

pub(crate) fn empty_math() -> FjEffectsFrameMath {
    EMPTY_MATH
}
