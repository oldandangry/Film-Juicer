//! Fixed intermediate observations for the independent input-hull fixtures.

use super::{Hull, HullError, HullInput};

pub struct Trace {
    pub locus_center: Option<([[f64; 2]; 66], [f64; 2])>,
    pub reach: Option<[f64; 1024]>,
    pub forward: [[f64; 2]; 1024],
    pub coefficients: [[f64; 2]; 1024],
    pub inverse: [[f64; 2]; 1024],
    pub smoothed: Option<[f64; 1024]>,
    pub floored: Option<[f64; 1024]>,
    pub scale: Option<f64>,
}

pub fn capture(input: HullInput<'_>) -> (Result<Hull, HullError>, Trace) {
    let mut trace = Trace {
        locus_center: None,
        reach: None,
        forward: [[0.0; 2]; 1024],
        coefficients: [[0.0; 2]; 1024],
        inverse: [[0.0; 2]; 1024],
        smoothed: None,
        floored: None,
        scale: None,
    };
    let result = super::build_hull(input, Some(&mut trace));
    (result, trace)
}

pub fn locus_ray(center: [f64; 2], direction: [f64; 2], polygon: &[[f64; 2]; 66]) -> f64 {
    super::ray_polygon_distance(center, direction, polygon)
}

pub fn hull_ray(center: [f64; 2], direction: [f64; 2], polygon: &[[f32; 2]; 1025]) -> f64 {
    super::ray_polygon_distance(center, direction, polygon)
}

pub fn output_constants() -> [[f32; 9]; 4] {
    [
        super::OKLAB_XYZ_TO_LMS,
        super::OKLAB_LMS_TO_XYZ,
        super::OKLAB_LMS_ROOT_TO_LAB,
        super::OKLAB_LAB_TO_LMS_ROOT,
    ]
}
pub fn output_direction(hue_index: usize) -> [f64; 3] {
    let hue = -0.5 * super::TWO_PI + super::TWO_PI * hue_index as f64 / 720.0;
    [hue, hue.cos(), hue.sin()]
}
pub fn output_cube(value: f64) -> bool {
    super::channel_inside_cube(value)
}
pub fn output_evaluation(coefficients: &[f64; 4], chroma: f64) -> f64 {
    super::evaluate_cubic(coefficients, chroma)
}
pub fn output_primitive(coefficients: &[f64; 4]) -> ([f64; 4], usize, f64) {
    let mut partitions = [0.0, 2.0, 0.0, 0.0];
    let mut count = 2;
    super::append_derivative_roots(coefficients, &mut partitions, &mut count);
    (partitions, count, super::first_channel_exit(coefficients))
}
pub fn output_lms_matrix(exit: &[f32; 9]) -> [f64; 9] {
    super::multiply_matrix(
        &exit.map(f64::from),
        &super::OKLAB_LMS_TO_XYZ.map(f64::from),
    )
}
pub fn output_cubics(exit: &[f32; 9], lightness: f64, direction: [f64; 2]) -> [[f64; 4]; 3] {
    super::boundary_cubics(&output_lms_matrix(exit), lightness, direction)
}
pub fn output_reserve_failure() -> super::BoundaryError {
    super::boundary_storage(usize::MAX).unwrap_err()
}

pub fn output_allocation_bytes(boundary: &super::OutputBoundary) -> usize {
    boundary.cmax.capacity() * std::mem::size_of::<f32>()
}
