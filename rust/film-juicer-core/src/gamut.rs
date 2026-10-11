//! Input hull construction and borrowed TC remapping.

#[cfg(feature = "test-support")]
pub mod fixtures;

use crate::hash;
use crate::reconstruction::{self, TcError};

#[expect(
    clippy::approx_constant,
    reason = "preserve the native hull's f64 literal and angle evaluation"
)]
const TWO_PI: f64 = 6.283185307179586476925286766559;
const DIRECTIONS: usize = 1024;
const MINIMUM_RADIUS: f64 = 1e-12;

#[derive(Clone, Copy)]
pub struct HullInput<'a> {
    pub x: &'a [f32; 81],
    pub y: &'a [f32; 81],
    pub z: &'a [f32; 81],
    pub d65: &'a [f32; 81],
}

/// Complete owned coordinates and their raw IEEE content identity. Construction
/// preserves the numerical producer's checks, without new narrowed-value gates.
pub struct Hull {
    center: [f32; 2],
    xy: [[f32; 2]; 1025],
    hash: u64,
}
impl Hull {
    pub fn center(&self) -> &[f32; 2] {
        &self.center
    }
    pub fn xy(&self) -> &[[f32; 2]; 1025] {
        &self.xy
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HullError {
    LocusAndCenter,
    RayIntersections,
    InscribedScale,
    Identity,
}
impl std::fmt::Display for HullError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let requirement = match self {
            Self::LocusAndCenter => "cie1931_380_700_and_d65",
            Self::RayIntersections => "finite_locus_ray_intersections",
            Self::InscribedScale => "finite_inscribed_scale",
            Self::Identity => "stable_identity",
        };
        write!(
            formatter,
            "MalformedRequiredResource component=input_gamut_hull requirement={requirement}"
        )
    }
}
impl std::error::Error for HullError {}

fn locus(input: HullInput<'_>) -> Result<[[f64; 2]; 66], HullError> {
    let mut points = [[0.0; 2]; 66];
    for (index, point) in points[..65].iter_mut().enumerate() {
        let x = f64::from(input.x[index]);
        let y = f64::from(input.y[index]);
        let z = f64::from(input.z[index]);
        let sum = x + y + z;
        if !(x.is_finite()
            && y.is_finite()
            && z.is_finite()
            && sum.is_finite()
            && sum > MINIMUM_RADIUS)
        {
            return Err(HullError::LocusAndCenter);
        }
        *point = [x / sum, y / sum];
    }
    points[65] = points[0];
    Ok(points)
}

fn d65_center(input: HullInput<'_>) -> Result<[f64; 2], HullError> {
    let mut xyz = [0.0; 3];
    for index in 0..81 {
        let illuminant = f64::from(input.d65[index]);
        if !illuminant.is_finite() {
            return Err(HullError::LocusAndCenter);
        }
        xyz[0] += illuminant * f64::from(input.x[index]);
        xyz[1] += illuminant * f64::from(input.y[index]);
        xyz[2] += illuminant * f64::from(input.z[index]);
    }
    let sum = xyz[0] + xyz[1] + xyz[2];
    if !(sum.is_finite() && sum > MINIMUM_RADIUS) {
        return Err(HullError::LocusAndCenter);
    }
    let center = [xyz[0] / sum, xyz[1] / sum];
    if !(center[0].is_finite() && center[1].is_finite()) {
        return Err(HullError::LocusAndCenter);
    }
    Ok(center)
}

fn fourier_lowpass(
    source: &[f64; DIRECTIONS],
    #[cfg(feature = "test-support")] mut trace: Option<&mut fixtures::Trace>,
) -> [f64; DIRECTIONS] {
    let mut coefficients = [[0.0; 2]; DIRECTIONS];
    for (mode, output) in coefficients.iter_mut().enumerate() {
        let mut coefficient = [0.0; 2];
        for (sample, &radius) in source.iter().enumerate() {
            let angle = -TWO_PI * (mode * sample) as f64 / DIRECTIONS as f64;
            coefficient[0] += radius * angle.cos();
            coefficient[1] += radius * angle.sin();
        }
        #[cfg(feature = "test-support")]
        if let Some(trace) = trace.as_deref_mut() {
            trace.forward[mode] = coefficient;
        }
        let signed_mode = if mode <= DIRECTIONS / 2 {
            mode as i32
        } else {
            mode as i32 - DIRECTIONS as i32
        };
        let normalized_mode = f64::from(signed_mode) / 5.0;
        let weight = (-0.5 * normalized_mode * normalized_mode).exp();
        *output = [coefficient[0] * weight, coefficient[1] * weight];
    }
    #[cfg(feature = "test-support")]
    if let Some(trace) = trace.as_deref_mut() {
        trace.coefficients = coefficients;
    }
    let mut smoothed = [0.0; DIRECTIONS];
    for (sample, output) in smoothed.iter_mut().enumerate() {
        let mut value = [0.0; 2];
        for (mode, coefficient) in coefficients.iter().enumerate() {
            let angle = TWO_PI * (mode * sample) as f64 / DIRECTIONS as f64;
            let cosine = angle.cos();
            let sine = angle.sin();
            // Both components retain the native complex multiply/add schedule.
            value[0] += coefficient[0] * cosine - coefficient[1] * sine;
            value[1] += coefficient[0] * sine + coefficient[1] * cosine;
        }
        #[cfg(feature = "test-support")]
        if let Some(trace) = trace.as_deref_mut() {
            trace.inverse[sample] = value;
        }
        *output = value[0] / DIRECTIONS as f64;
    }
    smoothed
}

fn hull_hash(center: &[f32; 2], xy: &[[f32; 2]; 1025]) -> u64 {
    let mut identity = hash::FNV_OFFSET;
    hash::update_bytes(&mut identity, &1_u32.to_le_bytes());
    for sample in center.iter().chain(xy.iter().flatten()) {
        hash::update_bytes(&mut identity, &sample.to_bits().to_le_bytes());
    }
    identity
}

fn build_hull(
    input: HullInput<'_>,
    #[cfg(feature = "test-support")] mut trace: Option<&mut fixtures::Trace>,
) -> Result<Hull, HullError> {
    let polygon = locus(input)?;
    let center = d65_center(input)?;
    #[cfg(feature = "test-support")]
    if let Some(trace) = trace.as_deref_mut() {
        trace.locus_center = Some((polygon, center));
    }
    let mut reach = [0.0; DIRECTIONS];
    for (index, distance) in reach.iter_mut().enumerate() {
        let angle = TWO_PI * index as f64 / DIRECTIONS as f64;
        *distance = ray_polygon_distance(center, [angle.cos(), angle.sin()], &polygon);
        if !(distance.is_finite() && *distance > 0.0) {
            return Err(HullError::RayIntersections);
        }
    }
    #[cfg(feature = "test-support")]
    if let Some(trace) = trace.as_deref_mut() {
        trace.reach = Some(reach);
    }
    let mut smoothed = fourier_lowpass(
        &reach,
        #[cfg(feature = "test-support")]
        trace.as_deref_mut(),
    );
    #[cfg(feature = "test-support")]
    if let Some(trace) = trace.as_deref_mut() {
        trace.smoothed = Some(smoothed);
    }
    let mut scale = f64::INFINITY;
    for (index, radius) in smoothed.iter_mut().enumerate() {
        // Ordered native max/min retain their first operand on ties or NaN.
        if *radius < MINIMUM_RADIUS {
            *radius = MINIMUM_RADIUS;
        }
        let candidate = reach[index] / *radius;
        if candidate < scale {
            scale = candidate;
        }
    }
    #[cfg(feature = "test-support")]
    if let Some(trace) = trace {
        trace.floored = Some(smoothed);
        trace.scale = Some(scale);
    }
    if !(scale.is_finite() && scale > 0.0) {
        return Err(HullError::InscribedScale);
    }
    let mut xy = [[0.0; 2]; 1025];
    for (index, point) in xy[..DIRECTIONS].iter_mut().enumerate() {
        let angle = TWO_PI * index as f64 / DIRECTIONS as f64;
        let radius = smoothed[index] * scale;
        *point = [
            (center[0] + radius * angle.cos()) as f32,
            (center[1] + radius * angle.sin()) as f32,
        ];
    }
    xy[DIRECTIONS] = xy[0];
    let center = center.map(|value| value as f32);
    let hash = hull_hash(&center, &xy);
    if hash == 0 {
        return Err(HullError::Identity);
    }
    Ok(Hull { center, xy, hash })
}

pub fn build_input_hull(input: HullInput<'_>) -> Result<Hull, HullError> {
    build_hull(
        input,
        #[cfg(feature = "test-support")]
        None,
    )
}

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
fn ray_polygon_distance<T: Copy + Into<f64>, const N: usize>(
    center: [f64; 2],
    direction: [f64; 2],
    polygon: &[[T; 2]; N],
) -> f64 {
    let mut minimum = f64::INFINITY;
    for edge in polygon.windows(2) {
        let ax: f64 = edge[0][0].into();
        let ay: f64 = edge[0][1].into();
        let ex = edge[1][0].into() - ax;
        let ey = edge[1][1].into() - ay;
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

#[cfg(test)]
mod tests {
    #[test]
    fn identity_preserves_raw_ieee_words_and_closure() {
        let center = [-0.0, f32::from_bits(0x7fc0_1234)];
        let mut polygon = [[0.0; 2]; 1025];
        polygon[0] = [-0.0, f32::INFINITY];
        // Independent FNV byte-stream witness: LE version, center, every xy word.
        assert_eq!(super::hull_hash(&center, &polygon), 0x55eed413b5422a16);
        let original = super::hull_hash(&center, &polygon);
        polygon[0][0] = 0.0;
        assert_ne!(super::hull_hash(&center, &polygon), original);
        let open_polygon = super::hull_hash(&center, &polygon);
        polygon[1024] = polygon[0];
        assert_ne!(super::hull_hash(&center, &polygon), open_polygon);
    }
}
