//! Fixed positional spectral tables and the distinct scanner/reference white.

use std::fmt;

use crate::hash;

#[derive(Clone, Copy)]
pub struct Observer<'a> {
    pub x: &'a [f32; 81],
    pub y: &'a [f32; 81],
    pub z: &'a [f32; 81],
}

pub struct Baseline<'a> {
    pub minimum: &'a [f32; 81],
    pub midpoint: Option<&'a [f32; 81]>,
}

pub struct Input<'a> {
    pub dyes_cmy: &'a [[f32; 3]; 81],
    pub observer: Observer<'a>,
    pub illuminant: &'a [f32; 81],
    pub baseline: Option<Baseline<'a>>,
    pub illuminant_hash: u64,
}

/// Complete numerical result; later film/publication admission remains separate.
pub struct Tables {
    lambda_nm: [f32; 81],
    illuminant: [f32; 81],
    observer_xyz: [[f32; 81]; 3],
    weighted_xyz: [[f32; 81]; 3],
    dyes_cmy: [[f32; 81]; 3],
    baseline_min: [f32; 81],
    baseline_mid: [f32; 81],
    inv_yn: f32,
    white_xyz: [f32; 3],
    has_baseline: bool,
    illuminant_hash: u64,
    hash: u64,
}
impl Tables {
    pub fn lambda_nm(&self) -> &[f32; 81] {
        &self.lambda_nm
    }
    pub fn illuminant(&self) -> &[f32; 81] {
        &self.illuminant
    }
    pub fn observer_xyz(&self) -> &[[f32; 81]; 3] {
        &self.observer_xyz
    }
    pub fn weighted_xyz(&self) -> &[[f32; 81]; 3] {
        &self.weighted_xyz
    }
    pub fn dyes_cmy(&self) -> &[[f32; 81]; 3] {
        &self.dyes_cmy
    }
    pub fn baseline_min(&self) -> &[f32; 81] {
        &self.baseline_min
    }
    pub fn baseline_mid(&self) -> &[f32; 81] {
        &self.baseline_mid
    }
    pub fn inv_yn(&self) -> f32 {
        self.inv_yn
    }
    pub fn white_xyz(&self) -> &[f32; 3] {
        &self.white_xyz
    }
    pub fn has_baseline(&self) -> bool {
        self.has_baseline
    }
    pub fn illuminant_hash(&self) -> u64 {
        self.illuminant_hash
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

fn digest(samples: &[f32]) -> u64 {
    let hashes = hash::f32s_with_nan_mask(samples);
    hash::u64s(&[hashes.values, hashes.nan_mask])
}

pub fn build_tables(input: Input<'_>) -> Tables {
    let lambda_nm = std::array::from_fn(|i| 380.0 + 5.0 * i as f32);
    let observer_xyz = [*input.observer.x, *input.observer.y, *input.observer.z];
    let weighted_xyz: [[f32; 81]; 3] = std::array::from_fn(|channel| {
        std::array::from_fn(|i| input.illuminant[i] * observer_xyz[channel][i])
    });
    let mut sums = [0.0_f64; 3];
    for ((&x, &y), &z) in weighted_xyz[0]
        .iter()
        .zip(&weighted_xyz[1])
        .zip(&weighted_xyz[2])
    {
        sums[0] += f64::from(x);
        sums[1] += f64::from(y);
        sums[2] += f64::from(z);
    }
    // This reciprocal is deliberately rounded to f32 before white scaling.
    let inv_yn = if sums[1] > 0.0 {
        1.0_f32 / sums[1] as f32
    } else {
        1.0
    };
    let white_xyz = sums.map(|sum| (sum * f64::from(inv_yn)) as f32);
    let dyes_cmy: [[f32; 81]; 3] =
        std::array::from_fn(|channel| std::array::from_fn(|i| input.dyes_cmy[i][channel]));
    let mut baseline_min = [0.0; 81];
    let mut baseline_mid = [0.0; 81];
    let has_baseline = input.baseline.is_some();
    if let Some(baseline) = input.baseline {
        baseline_min = *baseline.minimum;
        if let Some(midpoint) = baseline.midpoint {
            baseline_mid = *midpoint;
        }
    }
    let illuminant_hash = if input.illuminant_hash == 0 {
        digest(input.illuminant)
    } else {
        input.illuminant_hash
    };
    let fields = [
        illuminant_hash,
        digest(&lambda_nm),
        digest(&[5.0]),
        digest(&[inv_yn]),
        digest(&white_xyz),
        digest(&white_xyz),
        digest(&weighted_xyz[0]),
        digest(&weighted_xyz[1]),
        digest(&weighted_xyz[2]),
        digest(input.illuminant),
        digest(&observer_xyz[0]),
        digest(&observer_xyz[1]),
        digest(&observer_xyz[2]),
        digest(&dyes_cmy[0]),
        digest(&dyes_cmy[1]),
        digest(&dyes_cmy[2]),
        digest(&baseline_min),
        digest(&baseline_mid),
        digest(&[0.0]),
        hash::bytes(&[u8::from(has_baseline)]),
    ];
    Tables {
        lambda_nm,
        illuminant: *input.illuminant,
        observer_xyz,
        weighted_xyz,
        dyes_cmy,
        baseline_min,
        baseline_mid,
        inv_yn,
        white_xyz,
        has_baseline,
        illuminant_hash,
        hash: hash::u64s(&fields),
    }
}

pub struct White {
    normalization: f32,
    xyz: [f32; 3],
    xy: [f32; 2],
    hash: u64,
}
impl White {
    pub fn normalization(&self) -> f32 {
        self.normalization
    }
    pub fn xyz(&self) -> &[f32; 3] {
        &self.xyz
    }
    pub fn xy(&self) -> &[f32; 2] {
        &self.xy
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

#[derive(Debug)]
pub enum WhiteError {
    NonfiniteSample,
    InvalidLuminance(f64),
    InvalidSum,
    NonfiniteHashOperand,
    ZeroIdentity,
}
impl fmt::Display for WhiteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonfiniteSample => f.write_str("non-finite CMF/SPD sample"),
            Self::InvalidLuminance(sum) => write!(f, "invalid luminance sum (Yn={sum})"),
            Self::InvalidSum => f.write_str("invalid white sum"),
            Self::NonfiniteHashOperand => {
                f.write_str("non-finite white normalization hash operand")
            }
            Self::ZeroIdentity => f.write_str("zero white identity"),
        }
    }
}
impl std::error::Error for WhiteError {}

pub fn integrate_white(
    observer: Observer<'_>,
    illuminant: &[f32; 81],
) -> Result<White, WhiteError> {
    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let mut sum_z = 0.0_f64;
    for (i, &spd) in illuminant.iter().enumerate() {
        let x = observer.x[i];
        let y = observer.y[i];
        let z = observer.z[i];
        if !(spd.is_finite() && x.is_finite() && y.is_finite() && z.is_finite()) {
            return Err(WhiteError::NonfiniteSample);
        }
        sum_x += f64::from(spd) * f64::from(x);
        sum_y += f64::from(spd) * f64::from(y);
        sum_z += f64::from(spd) * f64::from(z);
    }
    if !(sum_y.is_finite() && sum_y > 0.0) {
        return Err(WhiteError::InvalidLuminance(sum_y));
    }
    let normalization = sum_y as f32;
    let inv_yn = 1.0 / sum_y;
    let xyz = [(sum_x * inv_yn) as f32, 1.0, (sum_z * inv_yn) as f32];
    let sum = sum_x + sum_y + sum_z;
    if !(sum.is_finite() && sum > 0.0) {
        return Err(WhiteError::InvalidSum);
    }
    let xy = [(sum_x / sum) as f32, (sum_y / sum) as f32];
    let mut samples = [0.0; 82];
    samples[..81].copy_from_slice(illuminant);
    samples[81] = normalization;
    let hash = hash::finite_f32s(&samples).map_err(|_| WhiteError::NonfiniteHashOperand)?;
    if hash == 0 {
        return Err(WhiteError::ZeroIdentity);
    }
    Ok(White {
        normalization,
        xyz,
        xy,
        hash,
    })
}

/// Weighted X/Y/Z planes. Clamp uses the native first-argument max semantics.
pub fn s_inverse(weights: [&[f32; 81]; 3]) -> [f32; 9] {
    let mut sums = [0.0_f64; 9];
    for ((&x, &y), &z) in weights[0].iter().zip(weights[1]).zip(weights[2]) {
        let components = [x, y, z];
        for (row, &component) in components.iter().enumerate() {
            let clamped = if component > 0.0 { component } else { 0.0 };
            for (col, &original) in components.iter().enumerate() {
                sums[row * 3 + col] += f64::from(clamped * original);
            }
        }
    }
    for sum in &mut sums {
        *sum *= 5.0;
    }
    let [xx, xy, xz, yx, yy, yz, zx, zy, zz] = sums;
    let det = xx * (yy * zz - yz * zy) - xy * (yx * zz - yz * zx) + xz * (yx * zy - yy * zx);
    if det.abs() < 1e-20 {
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    }
    let inv_det = 1.0 / det;
    [
        (yy * zz - yz * zy) * inv_det,
        (xz * zy - xy * zz) * inv_det,
        (xy * yz - xz * yy) * inv_det,
        (yz * zx - yx * zz) * inv_det,
        (xx * zz - xz * zx) * inv_det,
        (xz * yx - xx * yz) * inv_det,
        (yx * zy - yy * zx) * inv_det,
        (xy * zx - xx * zy) * inv_det,
        (xx * yy - xy * yx) * inv_det,
    ]
    .map(|x| x as f32)
}
