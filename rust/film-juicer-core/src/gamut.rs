//! Borrowed input hull and TC remapping; hull construction/cache stays at its owner.

use crate::reconstruction::{self, TcError};

#[derive(Clone, Copy)]
pub struct InputHull<'a> {
    center: [f32; 2],
    xy: &'a [[f32; 2]; 1025],
}
impl<'a> InputHull<'a> {
    pub fn new(center: [f32; 2], xy: &'a [[f32; 2]; 1025]) -> Self {
        Self { center, xy }
    }
}

#[expect(
    clippy::manual_range_contains,
    reason = "preserve the native ray's ordered edge comparisons and explicit thresholds"
)]
fn ray_polygon_distance(center: [f64; 2], direction: [f64; 2], polygon: &[[f32; 2]; 1025]) -> f64 {
    let mut minimum = f64::INFINITY;
    for index in 0..1024 {
        let ax = f64::from(polygon[index][0]);
        let ay = f64::from(polygon[index][1]);
        let ex = f64::from(polygon[index + 1][0]) - ax;
        let ey = f64::from(polygon[index + 1][1]) - ay;
        let denominator = direction[0] * ey - direction[1] * ex;
        if denominator.abs() <= 1e-12 {
            continue;
        }
        let ox = center[0] - ax;
        let oy = center[1] - ay;
        let distance = (-ox * ey + oy * ex) / denominator;
        let edge_position = (-ox * direction[1] + oy * direction[0]) / denominator;
        if distance > 1e-9
            && edge_position >= -1e-9
            && edge_position <= 1.0 + 1e-9
            && distance < minimum
        {
            minimum = distance;
        }
    }
    minimum
}
#[expect(
    clippy::neg_cmp_op_on_partial_ord,
    reason = "the native knee bypass includes unordered comparisons"
)]
fn knee(distance: f64) -> f64 {
    if !(distance > 0.815) {
        return distance;
    }
    let scale = 1.0 - 0.815;
    let normalized = (distance - 0.815) / scale;
    let compressed = normalized / (1.0 + normalized.powf(1.2)).powf(1.0 / 1.2);
    0.815 + scale * compressed
}
fn compress(hull: InputHull<'_>, xy: [f32; 2]) -> Result<[f32; 2], TcError> {
    if !xy.iter().all(|v| v.is_finite()) {
        return Err(TcError::UnusableCompressedCoordinates);
    }
    let center = hull.center.map(f64::from);
    let dx = f64::from(xy[0]) - center[0];
    let dy = f64::from(xy[1]) - center[1];
    let distance = dx.hypot(dy);
    if distance < 1e-9 {
        return Ok(xy);
    }
    let direction = [dx / distance, dy / distance];
    let boundary = ray_polygon_distance(center, direction, hull.xy);
    if !boundary.is_finite() || boundary <= 1e-12 {
        return Err(TcError::UnusableCompressedCoordinates);
    }
    let compressed_distance = knee(distance / boundary) * boundary;
    let result = [
        center[0] + direction[0] * compressed_distance,
        center[1] + direction[1] * compressed_distance,
    ];
    if !result.iter().all(|v| v.is_finite()) {
        return Err(TcError::UnusableCompressedCoordinates);
    }
    Ok(result.map(|v| v as f32))
}
fn bilinear(samples: &[f32], tc: [f32; 2], channel: usize) -> Result<f32, TcError> {
    let cp = tc[0].clamp(0.0, 1.0) * 191.0;
    let mp = tc[1].clamp(0.0, 1.0) * 191.0;
    if !cp.is_finite() || !mp.is_finite() {
        return Err(TcError::UnusableSamplingCoordinates);
    }
    let c0 = cp.floor() as usize;
    let m0 = mp.floor() as usize;
    let c1 = (c0 + 1).min(191);
    let m1 = (m0 + 1).min(191);
    let cf = cp - c0 as f32;
    let mf = mp - m0 as f32;
    let value = |c: usize, m: usize| samples[(c * 192 + m) * 4 + channel];
    let row0 = value(c0, m0) + cf * (value(c1, m0) - value(c0, m0));
    let row1 = value(c0, m1) + cf * (value(c1, m1) - value(c0, m1));
    Ok(row0 + mf * (row1 - row0))
}
pub(crate) fn remap_tc_lut(
    samples: &[f32; reconstruction::TC_LUT_SAMPLE_COUNT],
    hull: InputHull<'_>,
) -> Result<Vec<f32>, TcError> {
    let mut output = reconstruction::tc_scratch(reconstruction::TC_LUT_SAMPLE_COUNT)?;
    for c in 0..192 {
        let root = (c as f32 / 191.0).sqrt();
        for m in 0..192 {
            let xy = [1.0 - root, (m as f32 / 191.0) * root];
            let compressed = compress(hull, xy)?;
            let tc = reconstruction::tri2quad(compressed[0], compressed[1]);
            for channel in 0..3 {
                let value = bilinear(samples, tc, channel)?;
                if !value.is_finite() {
                    return Err(TcError::NonfiniteRemappedSample);
                }
                output[(c * 192 + m) * 4 + channel] = value;
            }
        }
    }
    Ok(output)
}
