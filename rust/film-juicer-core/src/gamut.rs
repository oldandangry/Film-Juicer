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

// Output construction shares the native fixed carrier until S4.E; cache policy
// and CUDA execution remain outside this safe mathematical producer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum OutputSpace {
    Srgb,
    DciP3,
    DisplayP3,
    AdobeRgb,
    Bt2020,
    ProPhotoRgb,
    Aces2065,
    DaVinciWideGamut,
    Rec709,
}
impl TryFrom<u32> for OutputSpace {
    type Error = ();
    fn try_from(tag: u32) -> Result<Self, Self::Error> {
        match tag {
            0 => Ok(Self::Srgb),
            1 => Ok(Self::DciP3),
            2 => Ok(Self::DisplayP3),
            3 => Ok(Self::AdobeRgb),
            4 => Ok(Self::Bt2020),
            5 => Ok(Self::ProPhotoRgb),
            6 => Ok(Self::Aces2065),
            7 => Ok(Self::DaVinciWideGamut),
            8 => Ok(Self::Rec709),
            _ => Err(()),
        }
    }
}
#[derive(Clone, Copy)]
pub struct TransformInput<'a> {
    pub space: OutputSpace,
    pub rgb_to_xyz: &'a [f32; 9],
    pub xyz_to_rgb: &'a [f32; 9],
    pub white_xyz: &'a [f32; 3],
    pub d65_white_xyz: &'a [f32; 3],
}
pub struct OutputTransform {
    native_rgb_to_d65_xyz: [f32; 9],
    d65_xyz_to_native_rgb: [f32; 9],
    space: OutputSpace,
    hash: u64,
    contract_hash: u64,
}
impl OutputTransform {
    pub fn native_rgb_to_d65_xyz(&self) -> &[f32; 9] {
        &self.native_rgb_to_d65_xyz
    }
    pub fn d65_xyz_to_native_rgb(&self) -> &[f32; 9] {
        &self.d65_xyz_to_native_rgb
    }
    pub fn space(&self) -> OutputSpace {
        self.space
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
    pub fn contract_hash(&self) -> u64 {
        self.contract_hash
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransformError {
    FiniteNativeWhiteCat16,
    Identity,
}
impl std::fmt::Display for TransformError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::FiniteNativeWhiteCat16 => "MalformedRequiredResource component=output_gamut_transform requirement=finite_native_white_cat16",
            Self::Identity => "ResourceDescriptorMismatch component=output_gamut_transform field=identity",
        })
    }
}
impl std::error::Error for TransformError {}

pub const OUTPUT_BOUNDARY_VALUE_COUNT: usize = 64 * 720;
const OUTPUT_LIGHTNESS_COUNT: usize = 64;
const OUTPUT_HUE_COUNT: usize = 720;
const OUTPUT_BISECTIONS: usize = 20;
const OUTPUT_MINIMUM_LIGHTNESS: f32 = 0.02;
const OUTPUT_MAXIMUM_LIGHTNESS: f32 = 1.0;
const OUTPUT_CUBE_SLACK: f32 = 1e-6;
const OUTPUT_MAXIMUM_CHROMA: f32 = 2.0;
const DERIVATIVE_TOLERANCE: f64 = 1e-14;

#[expect(
    clippy::excessive_precision,
    reason = "retain every authored f32 OkLab literal shared with CUDA"
)]
const OKLAB_XYZ_TO_LMS: [f32; 9] = [
    0.8189330101,
    0.3618667424,
    -0.1288597137,
    0.0329845436,
    0.9293118715,
    0.0361456387,
    0.0482003018,
    0.2643662691,
    0.6338517070,
];

#[expect(
    clippy::excessive_precision,
    reason = "retain every authored f32 OkLab literal shared with CUDA"
)]
const OKLAB_LMS_TO_XYZ: [f32; 9] = [
    1.2270138511,
    -0.5577999807,
    0.2812561490,
    -0.0405801784,
    1.1122568696,
    -0.0716766787,
    -0.0763812845,
    -0.4214819784,
    1.5861632204,
];

#[expect(
    clippy::excessive_precision,
    reason = "retain every authored f32 OkLab literal shared with CUDA"
)]
const OKLAB_LMS_ROOT_TO_LAB: [f32; 9] = [
    0.2104542553,
    0.7936177850,
    -0.0040720468,
    1.9779984951,
    -2.4285922050,
    0.4505937099,
    0.0259040371,
    0.7827717662,
    -0.8086757660,
];

#[expect(
    clippy::excessive_precision,
    reason = "retain every authored f32 OkLab literal shared with CUDA"
)]
const OKLAB_LAB_TO_LMS_ROOT: [f32; 9] = [
    1.0,
    0.3963377774,
    0.2158037573,
    1.0,
    -0.1055613458,
    -0.0638541728,
    1.0,
    -0.0894841775,
    -1.2914855480,
];

fn multiply_matrix(left: &[f64; 9], right: &[f64; 9]) -> [f64; 9] {
    let mut result = [0.0; 9];
    for row in 0..3 {
        for column in 0..3 {
            for inner in 0..3 {
                result[row * 3 + column] += left[row * 3 + inner] * right[inner * 3 + column];
            }
        }
    }
    result
}
fn output_contract_hash(transform_hash: u64) -> u64 {
    let mut identity = hash::FNV_OFFSET;
    hash::update_bytes(&mut identity, &1_u32.to_le_bytes());
    hash::update_bytes(&mut identity, &transform_hash.to_le_bytes());
    for value in [OUTPUT_LIGHTNESS_COUNT, OUTPUT_HUE_COUNT, OUTPUT_BISECTIONS] {
        hash::update_bytes(&mut identity, &(value as i32).to_le_bytes());
    }
    for value in [
        OUTPUT_MINIMUM_LIGHTNESS,
        OUTPUT_MAXIMUM_LIGHTNESS,
        OUTPUT_CUBE_SLACK,
        OUTPUT_MAXIMUM_CHROMA,
    ] {
        hash::update_bytes(&mut identity, &value.to_bits().to_le_bytes());
    }
    identity
}
pub fn output_transform(input: TransformInput<'_>) -> Result<OutputTransform, TransformError> {
    let native_white = input
        .white_xyz
        .map(|v| (f64::from(v) / f64::from(input.white_xyz[1])) as f32);
    let entry_adapt = crate::color::cat16_matrix(crate::color::Whites {
        source_xyz: native_white,
        destination_xyz: *input.d65_white_xyz,
    });
    let exit_adapt = crate::color::cat16_matrix(crate::color::Whites {
        source_xyz: *input.d65_white_xyz,
        destination_xyz: native_white,
    });
    let entry = multiply_matrix(
        &entry_adapt.map(f64::from),
        &input.rgb_to_xyz.map(f64::from),
    );
    let exit = multiply_matrix(&input.xyz_to_rgb.map(f64::from), &exit_adapt.map(f64::from));
    if !entry.iter().chain(&exit).all(|v| v.is_finite()) {
        return Err(TransformError::FiniteNativeWhiteCat16);
    }
    let entry = entry.map(|v| v as f32);
    let exit = exit.map(|v| v as f32);
    let mut identity = hash::FNV_OFFSET;
    hash::update_bytes(&mut identity, &1_u32.to_le_bytes());
    // The native enum occupies one byte, independently of the raw C u32 tag.
    hash::update_bytes(&mut identity, &[input.space as u8]);
    for sample in input
        .rgb_to_xyz
        .iter()
        .chain(input.xyz_to_rgb)
        .chain(input.white_xyz)
        .chain(&entry)
        .chain(&exit)
        .chain(&OKLAB_XYZ_TO_LMS)
        .chain(&OKLAB_LMS_TO_XYZ)
        .chain(&OKLAB_LMS_ROOT_TO_LAB)
        .chain(&OKLAB_LAB_TO_LMS_ROOT)
    {
        hash::update_bytes(&mut identity, &sample.to_bits().to_le_bytes());
    }
    if identity == 0 {
        return Err(TransformError::Identity);
    }
    Ok(OutputTransform {
        native_rgb_to_d65_xyz: entry,
        d65_xyz_to_native_rgb: exit,
        space: input.space,
        hash: identity,
        contract_hash: output_contract_hash(identity),
    })
}

/// Synchronous operands from a completed native transform. Correspondence of
/// its matrix and identity belongs to that immutable producer, not a new scan.
#[derive(Clone, Copy)]
pub struct BoundaryInput<'a> {
    pub space: OutputSpace,
    pub d65_xyz_to_native_rgb: &'a [f32; 9],
    pub transform_hash: u64,
}
pub struct OutputBoundary {
    cmax: Vec<f32>,
    transform_hash: u64,
    contract_hash: u64,
    hash: u64,
}
impl OutputBoundary {
    pub fn cmax(&self) -> &[f32] {
        &self.cmax
    }
    pub fn transform_hash(&self) -> u64 {
        self.transform_hash
    }
    pub fn contract_hash(&self) -> u64 {
        self.contract_hash
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoundaryError {
    Transform,
    C2Outside {
        space: OutputSpace,
        lightness_index: u32,
        hue_index: u32,
    },
    FiniteFirstExit,
    Identity,
    AllocationFailure,
}
impl std::fmt::Display for BoundaryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transform => formatter.write_str("ResourceDescriptorMismatch component=output_boundary_table field=transform"),
            Self::C2Outside { space, lightness_index, hue_index } => write!(formatter,"MalformedRequiredResource component=output_boundary_table requirement=c2_outside output_space={} lightness_index={lightness_index} hue_index={hue_index}", *space as u8),
            Self::FiniteFirstExit => formatter.write_str("MalformedRequiredResource component=output_boundary_table requirement=finite_first_exit"),
            Self::Identity => formatter.write_str("ResourceDescriptorMismatch component=output_boundary_table field=identity"),
            Self::AllocationFailure => formatter.write_str("output boundary allocation failed"),
        }
    }
}
impl std::error::Error for BoundaryError {}

fn evaluate_cubic(coefficient: &[f64; 4], chroma: f64) -> f64 {
    ((coefficient[3] * chroma + coefficient[2]) * chroma + coefficient[1]) * chroma + coefficient[0]
}
fn channel_inside_cube(value: f64) -> bool {
    let slack = f64::from(OUTPUT_CUBE_SLACK);
    value.is_finite() && value >= -slack && value <= 1.0 + slack
}
#[expect(
    clippy::neg_cmp_op_on_partial_ord,
    reason = "native discriminant rejection includes unordered values"
)]
fn append_derivative_roots(coefficient: &[f64; 4], partitions: &mut [f64; 4], count: &mut usize) {
    let a = 3.0 * coefficient[3];
    let b = 2.0 * coefficient[2];
    let c = coefficient[1];
    let mut append = |root: f64| {
        if root.is_finite() && root > 0.0 && root < f64::from(OUTPUT_MAXIMUM_CHROMA) {
            partitions[*count] = root;
            *count += 1;
        }
    };
    // std::max(initializer_list) retains the first operand on ties and NaN.
    let mut scale = a.abs();
    for value in [b.abs(), c.abs(), 1.0] {
        if scale < value {
            scale = value;
        }
    }
    if a.abs() <= DERIVATIVE_TOLERANCE * scale {
        if b.abs() > DERIVATIVE_TOLERANCE * scale {
            append(-c / b);
        }
        return;
    }
    let discriminant = b * b - 4.0 * a * c;
    if !(discriminant >= 0.0) {
        return;
    }
    let square_root = discriminant.sqrt();
    let q = -0.5 * (b + square_root.copysign(b));
    if q.abs() <= DERIVATIVE_TOLERANCE * scale {
        append(-b / (2.0 * a));
        return;
    }
    append(q / a);
    append(c / q);
}
fn first_channel_exit(coefficient: &[f64; 4]) -> f64 {
    if !channel_inside_cube(evaluate_cubic(coefficient, 0.0)) {
        return 0.0;
    }
    let mut partitions = [0.0, f64::from(OUTPUT_MAXIMUM_CHROMA), 0.0, 0.0];
    let mut count = 2;
    append_derivative_roots(coefficient, &mut partitions, &mut count);
    for index in 1..count {
        let value = partitions[index];
        let mut insertion = index;
        while insertion > 0 && value < partitions[insertion - 1] {
            partitions[insertion] = partitions[insertion - 1];
            insertion -= 1;
        }
        partitions[insertion] = value;
    }
    let mut retained = 1;
    for index in 1..count {
        if (partitions[retained - 1] - partitions[index]).abs() <= 1e-12 {
            continue;
        }
        partitions[retained] = partitions[index];
        retained += 1;
    }
    for partition in 1..retained {
        let mut lower = partitions[partition - 1];
        let mut upper = partitions[partition];
        if channel_inside_cube(evaluate_cubic(coefficient, upper)) {
            continue;
        }
        for _ in 0..OUTPUT_BISECTIONS {
            let middle = 0.5 * (lower + upper);
            if channel_inside_cube(evaluate_cubic(coefficient, middle)) {
                lower = middle;
            } else {
                upper = middle;
            }
        }
        return lower;
    }
    f64::INFINITY
}
fn boundary_cubics(
    lms_to_native_rgb: &[f64; 9],
    lightness: f64,
    direction: [f64; 2],
) -> [[f64; 4]; 3] {
    let mut u = [0.0; 3];
    let mut v = [0.0; 3];
    for row in 0..3 {
        u[row] = f64::from(OKLAB_LAB_TO_LMS_ROOT[row * 3]) * lightness;
        v[row] = f64::from(OKLAB_LAB_TO_LMS_ROOT[row * 3 + 1]) * direction[0]
            + f64::from(OKLAB_LAB_TO_LMS_ROOT[row * 3 + 2]) * direction[1];
    }
    let mut cubics = [[0.0; 4]; 3];
    for channel in 0..3 {
        for component in 0..3 {
            let weight = lms_to_native_rgb[channel * 3 + component];
            let component_u = u[component];
            let component_v = v[component];
            cubics[channel][0] += weight * component_u * component_u * component_u;
            cubics[channel][1] += weight * 3.0 * component_u * component_u * component_v;
            cubics[channel][2] += weight * 3.0 * component_u * component_v * component_v;
            cubics[channel][3] += weight * component_v * component_v * component_v;
        }
    }
    cubics
}
fn boundary_storage(count: usize) -> Result<Vec<f32>, BoundaryError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| BoundaryError::AllocationFailure)?;
    Ok(values)
}
pub fn build_output_boundary(input: BoundaryInput<'_>) -> Result<OutputBoundary, BoundaryError> {
    let contract_hash = output_contract_hash(input.transform_hash);
    if input.transform_hash == 0 || contract_hash == 0 {
        return Err(BoundaryError::Transform);
    }
    let lms_to_native = multiply_matrix(
        &input.d65_xyz_to_native_rgb.map(f64::from),
        &OKLAB_LMS_TO_XYZ.map(f64::from),
    );
    let directions: [[f64; 2]; OUTPUT_HUE_COUNT] = std::array::from_fn(|h| {
        let hue = -0.5 * TWO_PI + TWO_PI * h as f64 / OUTPUT_HUE_COUNT as f64;
        [hue.cos(), hue.sin()]
    });
    let mut cmax = boundary_storage(OUTPUT_BOUNDARY_VALUE_COUNT)?;
    for lightness_index in 0..OUTPUT_LIGHTNESS_COUNT {
        let lightness = f64::from(OUTPUT_MINIMUM_LIGHTNESS)
            + (f64::from(OUTPUT_MAXIMUM_LIGHTNESS) - f64::from(OUTPUT_MINIMUM_LIGHTNESS))
                * lightness_index as f64
                / (OUTPUT_LIGHTNESS_COUNT - 1) as f64;
        for (hue_index, &direction) in directions.iter().enumerate() {
            let cubics = boundary_cubics(&lms_to_native, lightness, direction);
            let mut endpoint_outside = false;
            let mut neutral_inside = true;
            let mut first_exit = f64::INFINITY;
            for cubic in &cubics {
                neutral_inside = neutral_inside && channel_inside_cube(evaluate_cubic(cubic, 0.0));
                endpoint_outside = endpoint_outside
                    || !channel_inside_cube(evaluate_cubic(
                        cubic,
                        f64::from(OUTPUT_MAXIMUM_CHROMA),
                    ));
                let exit = first_channel_exit(cubic);
                if exit < first_exit {
                    first_exit = exit;
                }
            }
            if !endpoint_outside {
                return Err(BoundaryError::C2Outside {
                    space: input.space,
                    lightness_index: lightness_index as u32,
                    hue_index: hue_index as u32,
                });
            }
            if !neutral_inside {
                first_exit = 0.0;
            }
            if !(first_exit.is_finite()
                && first_exit >= 0.0
                && first_exit <= f64::from(OUTPUT_MAXIMUM_CHROMA))
            {
                return Err(BoundaryError::FiniteFirstExit);
            }
            cmax.push(first_exit as f32);
        }
    }
    let mut identity = hash::FNV_OFFSET;
    hash::update_bytes(&mut identity, &contract_hash.to_le_bytes());
    for value in &cmax {
        hash::update_bytes(&mut identity, &value.to_bits().to_le_bytes());
    }
    if identity == 0 {
        return Err(BoundaryError::Identity);
    }
    Ok(OutputBoundary {
        cmax,
        transform_hash: input.transform_hash,
        contract_hash,
        hash: identity,
    })
}
