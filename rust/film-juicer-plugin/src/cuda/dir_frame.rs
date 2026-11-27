//! Projection of the admitted spatial DIR plan.

use film_juicer_core::film::dir::frame::{self, Component, Operator};
use film_juicer_core::frame::Domain;

use super::{
    edge::Failure,
    sys::{
        FjDirBoundaryInput, FjDirGaussian, FjExtent, FjSpatialDir, FJ_STATUS_PREPARATION_FAILURE,
    },
};

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
    boundary_identity: 0,
};
const EMPTY_MATH: FjSpatialDir = FjSpatialDir {
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
        boundary_identity: 0,
    }
}

pub(crate) fn projection(
    recipe_hash: u64,
    render: Domain,
    full: Domain,
    count: usize,
    components: &[Component; 4],
) -> Result<FjSpatialDir, Failure> {
    let mut projected = components.each_ref().map(component);
    for item in &mut projected[..count] {
        if item.reference_operator == super::sys::FJ_DIR_OPERATOR_YVV_REFLECT && item.weight > 0.0 {
            let input = FjDirBoundaryInput {
                component: *item,
                width: full.width,
                height: full.height,
            };
            item.boundary_identity = super::dir_boundary::project(&input)
                .map_err(|_| {
                    Failure::new(
                        FJ_STATUS_PREPARATION_FAILURE,
                        "spatial DIR reflected-boundary request is invalid",
                    )
                })?
                .key()
                .value();
        }
    }
    Ok(FjSpatialDir {
        component_count: count,
        recipe_hash,
        render_extent: extent(render),
        full_frame_extent: extent(full),
        filter_domain_extent: extent(full),
        components: projected,
    })
}

pub(crate) fn failure_diagnostic(field: frame::Field) -> &'static str {
    MESSAGES[field as usize]
}

pub(crate) fn empty_math() -> FjSpatialDir {
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
