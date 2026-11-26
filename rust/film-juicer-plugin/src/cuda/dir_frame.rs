//! Projection of the admitted spatial DIR plan.

use std::mem::{align_of, size_of};

use film_juicer_core::film::dir::frame::{self, Component, Operator};
use film_juicer_core::frame::Domain;

use crate::cuda::sys::{FjDirGaussian, FjExtent};

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjDirFrameMath {
    pub(crate) present: u32,
    pub(crate) component_count: u32,
    pub(crate) recipe_hash: u64,
    pub(crate) render_extent: FjExtent,
    pub(crate) full_frame_extent: FjExtent,
    pub(crate) filter_domain_extent: FjExtent,
    pub(crate) components: [FjDirGaussian; 4],
}

const EMPTY_EXTENT: FjExtent = FjExtent {
    x: 0,
    y: 0,
    width: 0,
    height: 0,
};
const EMPTY_COMPONENT: FjDirGaussian = FjDirGaussian {
    sigma_px: 0.0,
    weight: 0.0,
    radius: 0,
    reference_operator: 0,
    feedforward: 0.0,
    feedback: [0.0; 3],
    normalization_denominator: 0.0,
    feedforward_numerator: 0.0,
    feedback_numerators: [0.0; 3],
    boundary_truncation_accuracy: 0.0,
    boundary_certification_tolerance: 0.0,
    boundary_derivation_version: 0,
};
const EMPTY_MATH: FjDirFrameMath = FjDirFrameMath {
    present: 0,
    component_count: 0,
    recipe_hash: 0,
    render_extent: EMPTY_EXTENT,
    full_frame_extent: EMPTY_EXTENT,
    filter_domain_extent: EMPTY_EXTENT,
    components: [EMPTY_COMPONENT; 4],
};

fn extent(v: Domain) -> FjExtent {
    FjExtent {
        x: v.x,
        y: v.y,
        width: v.width,
        height: v.height,
    }
}

fn component(v: &Component) -> FjDirGaussian {
    let c = v.coefficients();
    FjDirGaussian {
        sigma_px: v.sigma_px(),
        weight: v.weight(),
        radius: v.radius(),
        reference_operator: match v.operator() {
            Operator::None => 0,
            Operator::Identity => 1,
            Operator::FirReflect => 2,
            Operator::YvvReflect => 3,
        },
        feedforward: c[0],
        feedback: [c[1], c[2], c[3]],
        normalization_denominator: c[4],
        feedforward_numerator: c[5],
        feedback_numerators: [c[6], c[7], c[8]],
        boundary_truncation_accuracy: v.boundary_accuracy(),
        boundary_certification_tolerance: v.boundary_tolerance(),
        boundary_derivation_version: v.boundary_version(),
    }
}

pub(crate) fn projection(
    recipe_hash: u64,
    render: Domain,
    full: Domain,
    count: usize,
    components: &[Component; 4],
) -> FjDirFrameMath {
    FjDirFrameMath {
        present: 1,
        component_count: count as u32,
        recipe_hash,
        render_extent: extent(render),
        full_frame_extent: extent(full),
        filter_domain_extent: extent(full),
        components: components.map(|c| component(&c)),
    }
}

pub(crate) fn failure_diagnostic(field: frame::Field) -> &'static str {
    MESSAGES[field as usize]
}

pub(crate) fn empty_math() -> FjDirFrameMath {
    EMPTY_MATH
}

const MESSAGES: [&str; 13] = [
    "",
    "SpatialDirFrameFailure field=pixel_size_um",
    "SpatialDirFrameFailure field=render_extent",
    "SpatialDirFrameFailure field=full_frame_extent",
    "SpatialDirFrameFailure field=partial_render_window",
    "SpatialDirFrameFailure field=core",
    "SpatialDirFrameFailure field=tail",
    "SpatialDirFrameFailure field=component_values",
    "SpatialDirFrameFailure field=component_capacity",
    "SpatialDirFrameFailure field=radius",
    "SpatialDirFrameFailure field=iir_denominator",
    "SpatialDirFrameFailure field=iir_output",
    "SpatialDirFrameFailure field=empty_plan",
];

const _: () = {
    assert!(size_of::<FjDirFrameMath>() == 512 && align_of::<FjDirFrameMath>() == 8);
    assert!(std::mem::offset_of!(FjDirFrameMath, present) == 0);
    assert!(std::mem::offset_of!(FjDirFrameMath, component_count) == 4);
    assert!(std::mem::offset_of!(FjDirFrameMath, recipe_hash) == 8);
    assert!(std::mem::offset_of!(FjDirFrameMath, render_extent) == 16);
    assert!(std::mem::offset_of!(FjDirFrameMath, full_frame_extent) == 32);
    assert!(std::mem::offset_of!(FjDirFrameMath, filter_domain_extent) == 48);
    assert!(std::mem::offset_of!(FjDirFrameMath, components) == 64);
    assert!(size_of::<FjExtent>() == 16 && align_of::<FjExtent>() == 4);
    assert!(std::mem::offset_of!(FjExtent, x) == 0);
    assert!(std::mem::offset_of!(FjExtent, y) == 4);
    assert!(std::mem::offset_of!(FjExtent, width) == 8);
    assert!(std::mem::offset_of!(FjExtent, height) == 12);
    assert!(size_of::<FjDirGaussian>() == 112 && align_of::<FjDirGaussian>() == 8);
    assert!(std::mem::offset_of!(FjDirGaussian, sigma_px) == 0);
    assert!(std::mem::offset_of!(FjDirGaussian, weight) == 4);
    assert!(std::mem::offset_of!(FjDirGaussian, radius) == 8);
    assert!(std::mem::offset_of!(FjDirGaussian, reference_operator) == 12);
    assert!(std::mem::offset_of!(FjDirGaussian, feedforward) == 16);
    assert!(std::mem::offset_of!(FjDirGaussian, feedback) == 24);
    assert!(std::mem::offset_of!(FjDirGaussian, normalization_denominator) == 48);
    assert!(std::mem::offset_of!(FjDirGaussian, feedforward_numerator) == 56);
    assert!(std::mem::offset_of!(FjDirGaussian, feedback_numerators) == 64);
    assert!(std::mem::offset_of!(FjDirGaussian, boundary_truncation_accuracy) == 88);
    assert!(std::mem::offset_of!(FjDirGaussian, boundary_certification_tolerance) == 96);
    assert!(std::mem::offset_of!(FjDirGaussian, boundary_derivation_version) == 104);
    assert!(
        size_of::<crate::cuda::sys::FjSpatialDir>() == 520
            && align_of::<crate::cuda::sys::FjSpatialDir>() == 8
    );
    assert!(std::mem::offset_of!(crate::cuda::sys::FjSpatialDir, render_extent) == 0);
    assert!(std::mem::offset_of!(crate::cuda::sys::FjSpatialDir, full_frame_extent) == 16);
    assert!(std::mem::offset_of!(crate::cuda::sys::FjSpatialDir, filter_domain_extent) == 32);
    assert!(std::mem::offset_of!(crate::cuda::sys::FjSpatialDir, component_count) == 48);
    assert!(std::mem::offset_of!(crate::cuda::sys::FjSpatialDir, components) == 56);
    assert!(std::mem::offset_of!(crate::cuda::sys::FjSpatialDir, recipe_hash) == 504);
    assert!(std::mem::offset_of!(crate::cuda::sys::FjSpatialDir, hash) == 512);
};
const _: fn(FjDirFrameMath) = |v| {
    let _: u32 = v.present;
    let _: u32 = v.component_count;
    let _: u64 = v.recipe_hash;
    let _: FjExtent = v.render_extent;
    let _: FjExtent = v.full_frame_extent;
    let _: FjExtent = v.filter_domain_extent;
    let _: [FjDirGaussian; 4] = v.components;
};
const _: fn(FjExtent) = |v| {
    let _: i32 = v.x;
    let _: i32 = v.y;
    let _: i32 = v.width;
    let _: i32 = v.height;
};
const _: fn(FjDirGaussian) = |v| {
    let _: f32 = v.sigma_px;
    let _: f32 = v.weight;
    let _: i32 = v.radius;
    let _: u32 = v.reference_operator;
    let _: f64 = v.feedforward;
    let _: [f64; 3] = v.feedback;
    let _: f64 = v.normalization_denominator;
    let _: f64 = v.feedforward_numerator;
    let _: [f64; 3] = v.feedback_numerators;
    let _: f64 = v.boundary_truncation_accuracy;
    let _: f64 = v.boundary_certification_tolerance;
    let _: u32 = v.boundary_derivation_version;
};
const _: fn(crate::cuda::sys::FjSpatialDir) = |v| {
    let _: FjExtent = v.render_extent;
    let _: FjExtent = v.full_frame_extent;
    let _: FjExtent = v.filter_domain_extent;
    let _: usize = v.component_count;
    let _: [FjDirGaussian; 4] = v.components;
    let _: u64 = v.recipe_hash;
    let _: u64 = v.hash;
};
