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
