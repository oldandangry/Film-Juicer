//! Reference reconstruction and complete film TC preparation from borrowed spectra.

use std::fmt;

pub const SPECTRA_SAMPLE_COUNT: usize = 192 * 192 * 81;

pub struct ReferenceWhite {
    samples: [f32; 81],
}
impl ReferenceWhite {
    pub fn samples(&self) -> &[f32; 81] {
        &self.samples
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    InvalidBlur,
    InvalidWhite,
    NonfiniteReferenceWhite,
    InvalidKernel,
    ImpossibleKernelSize,
    AllocationFailure,
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let requirement = match self {
            Self::InvalidBlur => "finite_blur",
            Self::InvalidWhite => "finite_reference_white",
            Self::NonfiniteReferenceWhite => "finite_reconstructed_reference_white",
            Self::InvalidKernel => "finite_blur_kernel",
            Self::ImpossibleKernelSize => "representable_blur_kernel",
            Self::AllocationFailure => "blur_kernel_allocation",
        };
        write!(
            f,
            "MalformedRequiredResource component=hanatos_window requirement={requirement}"
        )
    }
}
impl std::error::Error for Error {}

fn projected_white(white: [f32; 3]) -> Result<[f32; 2], Error> {
    let sum = (white[0] + white[1]) + white[2];
    if !sum.is_finite() || sum <= 0.0 {
        return Err(Error::InvalidWhite);
    }
    let tx = white[0] / sum;
    let ty = white[1] / sum;
    let [c, m] = tri2quad(tx, ty);
    if c.is_finite() && m.is_finite() {
        Ok([c, m])
    } else {
        Err(Error::InvalidWhite)
    }
}

fn mitchell_weight(t: f32) -> f32 {
    const B: f32 = 1.0 / 3.0;
    const C: f32 = 1.0 / 3.0;
    let x = t.abs();
    if x < 1.0 {
        return (1.0 / 6.0)
            * ((12.0 - 9.0 * B - 6.0 * C) * x * x * x
                + (-18.0 + 12.0 * B + 6.0 * C) * x * x
                + (6.0 - 2.0 * B));
    }
    if x < 2.0 {
        return (1.0 / 6.0)
            * ((-B - 6.0 * C) * x * x * x
                + (6.0 * B + 30.0 * C) * x * x
                + (-12.0 * B - 48.0 * C) * x
                + (8.0 * B + 24.0 * C));
    }
    0.0
}

fn mitchell_coordinate(normalized: f32) -> Option<(i32, f32)> {
    let value = normalized.clamp(0.0, 1.0) * 191.0;
    if !value.is_finite() {
        return None;
    }
    if value >= 191.0 {
        return Some((190, 1.0));
    }
    let base = value.floor() as i32;
    Some((base, value - base as f32))
}

fn mitchell_reflect_index(index: i32) -> usize {
    // The only callers supply the one-fold stencil [-1,192].
    if index < 0 {
        (-index) as usize
    } else if index >= 192 {
        (382 - index) as usize
    } else {
        index as usize
    }
}

fn sample_reference(
    spectra: &[f32; SPECTRA_SAMPLE_COUNT],
    tc: [f32; 2],
) -> Result<[f32; 81], Error> {
    let (c_base, c_fraction) = mitchell_coordinate(tc[0]).expect("finite projected TC coordinates");
    let (m_base, m_fraction) = mitchell_coordinate(tc[1]).expect("finite projected TC coordinates");
    let c_weights = [
        mitchell_weight(c_fraction + 1.0),
        mitchell_weight(c_fraction),
        mitchell_weight(c_fraction - 1.0),
        mitchell_weight(c_fraction - 2.0),
    ];
    let m_weights = [
        mitchell_weight(m_fraction + 1.0),
        mitchell_weight(m_fraction),
        mitchell_weight(m_fraction - 1.0),
        mitchell_weight(m_fraction - 2.0),
    ];
    let mut samples = [0.0; 81];
    for (sample, out) in samples.iter_mut().enumerate() {
        let mut value = 0.0_f32;
        let mut weight_sum = 0.0_f32;
        for (dc, &c_weight) in c_weights.iter().enumerate() {
            let c = mitchell_reflect_index(c_base - 1 + dc as i32);
            for (dm, &m_weight) in m_weights.iter().enumerate() {
                let m = mitchell_reflect_index(m_base - 1 + dm as i32);
                let weight = c_weight * m_weight;
                weight_sum += weight;
                // Zero weights still consume the source value, including NaNs.
                value += weight * spectra[(c * 192 + m) * 81 + sample];
            }
        }
        value = if weight_sum != 0.0 {
            value / weight_sum
        } else {
            0.0
        };
        if !value.is_finite() {
            return Err(Error::NonfiniteReferenceWhite);
        }
        *out = value;
    }
    Ok(samples)
}

fn gaussian_kernel(sigma: f32) -> Result<Vec<f32>, Error> {
    let rounded = 4.0_f32 * sigma + 0.5_f32;
    if !rounded.is_finite() || rounded < 1.0 {
        return Err(Error::InvalidKernel);
    }
    if f64::from(rounded) >= i64::MAX as f64 {
        return Err(Error::ImpossibleKernelSize);
    }
    let radius = rounded as i64;
    let radius_usize = usize::try_from(radius).map_err(|_| Error::ImpossibleKernelSize)?;
    let count = radius_usize
        .checked_mul(2)
        .and_then(|n| n.checked_add(1))
        .ok_or(Error::ImpossibleKernelSize)?;
    let bytes = count
        .checked_mul(size_of::<f32>())
        .ok_or(Error::ImpossibleKernelSize)?;
    if bytes > isize::MAX as usize {
        return Err(Error::ImpossibleKernelSize);
    }
    let mut kernel = Vec::new();
    kernel
        .try_reserve_exact(count)
        .map_err(|_| Error::AllocationFailure)?;
    let sigma_squared = sigma * sigma;
    let mut sum = 0.0_f32;
    for index in 0..count {
        let offset = index as i128 - i128::from(radius);
        // Widen the integer square before its original f32 conversion.
        let squared = offset * offset;
        let weight = (-0.5_f32 * squared as f32 / sigma_squared).exp();
        kernel.push(weight);
        sum += weight;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(Error::InvalidKernel);
    }
    for weight in &mut kernel {
        *weight /= sum;
    }
    Ok(kernel)
}

fn gaussian_reflect_index(index: i128) -> usize {
    // Equivalent to repeated half-sample reflection; widened arithmetic also
    // covers offsets outside native int's domain without an unbounded fold loop.
    let index = index.rem_euclid(162);
    if index >= 81 {
        (161 - index) as usize
    } else {
        index as usize
    }
}

#[expect(
    clippy::neg_cmp_op_on_partial_ord,
    reason = "preserve native signed-zero blur branch after finite admission"
)]
pub fn reference_white(
    spectra: &[f32; SPECTRA_SAMPLE_COUNT],
    spectral_blur: f32,
    white_xyz: [f32; 3],
) -> Result<ReferenceWhite, Error> {
    if !spectral_blur.is_finite() || spectral_blur < 0.0 {
        return Err(Error::InvalidBlur);
    }
    let sampled = sample_reference(spectra, projected_white(white_xyz)?)?;
    if !(spectral_blur > 0.0) {
        return Ok(ReferenceWhite { samples: sampled });
    }
    // Sampling and all 81 finite checks precede kernel size/reservation work.
    let kernel = gaussian_kernel(spectral_blur)?;
    let radius = (kernel.len() / 2) as i128;
    let mut samples = [0.0; 81];
    for (sample, out) in samples.iter_mut().enumerate() {
        let mut value = 0.0_f32;
        for (index, &weight) in kernel.iter().enumerate() {
            let reflected = gaussian_reflect_index(sample as i128 + index as i128 - radius);
            value += weight * sampled[reflected];
        }
        if !value.is_finite() {
            return Err(Error::NonfiniteReferenceWhite);
        }
        *out = value;
    }
    Ok(ReferenceWhite { samples })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn widened_half_sample_reflection_matches_repeated_folds() {
        for index in -46_340_i64..=46_420 {
            let mut folded = index;
            while !(0..81).contains(&folded) {
                folded = if folded < 0 {
                    -folded - 1
                } else {
                    162 - folded - 1
                };
            }
            assert_eq!(gaussian_reflect_index(i128::from(index)), folded as usize);
        }
    }
    #[test]
    fn size_impossibility_and_real_reservation_failure_are_distinct() {
        assert_eq!(
            gaussian_kernel(f32::from_bits(0x5e00_0000)),
            Err(Error::ImpossibleKernelSize)
        );
        // A representable ~4 EiB layout exceeds supported userspace address
        // spaces. This actually calls try_reserve_exact; it is not fault metadata.
        assert_eq!(
            gaussian_kernel(f32::from_bits(0x5c00_0000)),
            Err(Error::AllocationFailure)
        );
    }
}

pub const TC_LUT_SAMPLE_COUNT: usize = 192 * 192 * 4;

pub(crate) fn tri2quad(tx: f32, ty: f32) -> [f32; 2] {
    let x = 1.0 - tx;
    let denominator = if x < 1e-10 { 1e-10 } else { x };
    [(x * x).clamp(0.0, 1.0), (ty / denominator).clamp(0.0, 1.0)]
}

pub enum Method<'a> {
    Hanatos {
        spectral_blur: f32,
        surface_rgb: Option<&'a [[f32; 15]; 3]>,
    },
    Arctic,
}
pub struct Input<'a> {
    pub spectra: &'a [f32; SPECTRA_SAMPLE_COUNT],
    pub sensitivity_rgb: &'a [[f32; 3]; 81],
    pub reference_illuminant: &'a [f32; 81],
    pub projection_white_xyz: [f32; 3],
    pub method: Method<'a>,
    pub input_compression_active: bool,
    pub input_hull: Option<crate::gamut::InputHull<'a>>,
}
pub struct FilmTcLut {
    samples: Vec<f32>,
}
impl FilmTcLut {
    pub fn samples(&self) -> &[f32; TC_LUT_SAMPLE_COUNT] {
        self.samples.as_slice().try_into().expect("complete TC LUT")
    }
    pub fn into_samples(self) -> Vec<f32> {
        self.samples
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TcError {
    InvalidReferenceIlluminant,
    InvalidBlur,
    InvalidKernel,
    ImpossibleKernelSize,
    AllocationFailure,
    InvalidArcticWhite,
    InvalidNeutralResponse,
    InvalidSurfaceWhite,
    NonfiniteIntegratedSample,
    NonfiniteBlurredSample,
    MissingInputHull,
    UnusableCompressedCoordinates,
    NonfiniteRemappedSample,
    UnusableSamplingCoordinates,
}
impl TcError {
    pub fn diagnostic(&self) -> &'static str {
        match self {
            Self::InvalidReferenceIlluminant => {
                "MalformedRequiredResource component=film_tc_lut requirement=finite_nonnegative_reference_illuminant"
            }
            Self::InvalidBlur => {
                "MalformedRequiredResource component=film_tc_lut method=hanatos2025 requirement=finite_blur"
            }
            Self::InvalidKernel => {
                "MalformedRequiredResource component=film_tc_lut method=hanatos2025 requirement=finite_blur_kernel"
            }
            Self::ImpossibleKernelSize => {
                "MalformedRequiredResource component=film_tc_lut requirement=representable_blur_kernel"
            }
            Self::AllocationFailure => "film_tc_lut allocation failure",
            Self::InvalidArcticWhite => {
                "MalformedRequiredResource component=film_tc_lut method=arctic2026beta04 requirement=finite_d65_projection_white"
            }
            Self::InvalidNeutralResponse => {
                "MalformedRequiredResource component=film_tc_lut method=arctic2026beta04 requirement=finite_positive_neutral_response"
            }
            Self::InvalidSurfaceWhite => {
                "MalformedRequiredResource component=film_tc_lut method=hanatos2025 requirement=finite_surface_reference_white"
            }
            Self::NonfiniteIntegratedSample => {
                "MalformedRequiredResource component=film_tc_lut requirement=finite_integrated_samples"
            }
            Self::NonfiniteBlurredSample => {
                "MalformedRequiredResource component=film_tc_lut method=hanatos2025 requirement=finite_blurred_spectra"
            }
            Self::MissingInputHull => {
                "MissingRequiredResource component=film_tc_lut requirement=input_compression_hull"
            }
            Self::UnusableCompressedCoordinates => {
                "MalformedRequiredResource component=input_gamut_remap requirement=finite_compressed_coordinates"
            }
            Self::NonfiniteRemappedSample => {
                "MalformedRequiredResource component=input_gamut_remap requirement=finite_remapped_tc_lut"
            }
            Self::UnusableSamplingCoordinates => {
                "MalformedRequiredResource component=film_tc_lut requirement=usable_sampling_coordinates"
            }
        }
    }
}
pub(crate) fn tc_scratch(count: usize) -> Result<Vec<f32>, TcError> {
    if count
        .checked_mul(size_of::<f32>())
        .is_none_or(|bytes| bytes > isize::MAX as usize)
    {
        return Err(TcError::ImpossibleKernelSize);
    }
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(count)
        .map_err(|_| TcError::AllocationFailure)?;
    samples.resize(count, 0.0);
    Ok(samples)
}
fn sample_mitchell(
    samples: &[f32],
    tc: [f32; 2],
    stride: usize,
    component: usize,
) -> Result<f32, TcError> {
    let (c_base, c_fraction) =
        mitchell_coordinate(tc[0]).ok_or(TcError::UnusableSamplingCoordinates)?;
    let (m_base, m_fraction) =
        mitchell_coordinate(tc[1]).ok_or(TcError::UnusableSamplingCoordinates)?;
    let cw = [
        mitchell_weight(c_fraction + 1.0),
        mitchell_weight(c_fraction),
        mitchell_weight(c_fraction - 1.0),
        mitchell_weight(c_fraction - 2.0),
    ];
    let mw = [
        mitchell_weight(m_fraction + 1.0),
        mitchell_weight(m_fraction),
        mitchell_weight(m_fraction - 1.0),
        mitchell_weight(m_fraction - 2.0),
    ];
    let mut value = 0.0_f32;
    let mut weight_sum = 0.0_f32;
    for (dc, &c_weight) in cw.iter().enumerate() {
        let c = mitchell_reflect_index(c_base - 1 + dc as i32);
        for (dm, &m_weight) in mw.iter().enumerate() {
            let m = mitchell_reflect_index(m_base - 1 + dm as i32);
            let weight = c_weight * m_weight;
            weight_sum += weight;
            value += weight * samples[(c * 192 + m) * stride + component];
        }
    }
    Ok(if weight_sum != 0.0 {
        value / weight_sum
    } else {
        0.0
    })
}
fn blur_spectra(source: &[f32; SPECTRA_SAMPLE_COUNT], sigma: f32) -> Result<Vec<f32>, TcError> {
    if !sigma.is_finite() || sigma < 0.0 {
        return Err(TcError::InvalidBlur);
    }
    let kernel = gaussian_kernel(sigma).map_err(|error| match error {
        Error::InvalidKernel => TcError::InvalidKernel,
        Error::ImpossibleKernelSize => TcError::ImpossibleKernelSize,
        Error::AllocationFailure => TcError::AllocationFailure,
        _ => unreachable!("kernel returns only arithmetic, size or reservation failures"),
    })?;
    let radius = (kernel.len() / 2) as i128;
    let mut output = tc_scratch(SPECTRA_SAMPLE_COUNT)?;
    for c in 0..192 {
        for m in 0..192 {
            let base = (c * 192 + m) * 81;
            for sample in 0..81 {
                let mut value = 0.0_f32;
                for (index, &weight) in kernel.iter().enumerate() {
                    let reflected = gaussian_reflect_index(sample as i128 + index as i128 - radius);
                    value += weight * source[base + reflected];
                }
                if !value.is_finite() {
                    return Err(TcError::NonfiniteBlurredSample);
                }
                output[base + sample] = value;
            }
        }
    }
    Ok(output)
}
fn eval_hanatos_surface(params: &[f32; 15], tc: [f32; 2], center: [f32; 2]) -> f32 {
    let x = tc[0] - center[0];
    let y = tc[1] - center[1];
    let x2 = x * x;
    let y2 = y * y;
    let x3 = x2 * x;
    let y3 = y2 * y;
    let raw = params[1] * x
        + params[2] * y
        + params[3] * x2
        + params[4] * y2
        + params[5] * x * y
        + params[6] * x3
        + params[7] * y3
        + params[8] * x2 * y
        + params[9] * x * y2
        + params[10] * x2 * x2
        + params[11] * y2 * y2
        + params[12] * x3 * y
        + params[13] * x2 * y2
        + params[14] * x * y3;
    let normalized = raw / 2.0;
    raw / (1.0 + normalized * normalized).sqrt()
}
pub fn build_tc_lut(input: Input<'_>) -> Result<FilmTcLut, TcError> {
    if !input
        .reference_illuminant
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0)
    {
        return Err(TcError::InvalidReferenceIlluminant);
    }
    let blurred = match input.method {
        Method::Hanatos {
            spectral_blur: sigma,
            ..
        } if sigma > 0.0 => Some(blur_spectra(input.spectra, sigma)?),
        _ => None,
    };
    let selected = blurred.as_deref().unwrap_or(input.spectra);
    let mut denominator = [1.0; 3];
    if matches!(input.method, Method::Arctic) {
        let tc =
            projected_white(input.projection_white_xyz).map_err(|_| TcError::InvalidArcticWhite)?;
        denominator = [0.0; 3];
        for sample in 0..81 {
            let reflectance = sample_mitchell(selected, tc, 81, sample)?;
            let relit = reflectance * input.reference_illuminant[sample];
            for (channel, out) in denominator.iter_mut().enumerate() {
                *out += relit * input.sensitivity_rgb[sample][channel];
            }
        }
        if !denominator.iter().all(|v| v.is_finite() && *v > 0.0) {
            return Err(TcError::InvalidNeutralResponse);
        }
    }
    let mut integrated = tc_scratch(TC_LUT_SAMPLE_COUNT)?;
    let correction = match input.method {
        Method::Hanatos {
            surface_rgb: Some(params),
            ..
        } => Some((
            params,
            projected_white(input.projection_white_xyz)
                .map_err(|_| TcError::InvalidSurfaceWhite)?,
        )),
        _ => None,
    };
    for c in 0..192 {
        let tc_c = c as f32 / 191.0;
        for m in 0..192 {
            let tc_m = m as f32 / 191.0;
            let mut raw = [0.0_f32; 3];
            let spectral_base = (c * 192 + m) * 81;
            for sample in 0..81 {
                let mut energy = selected[spectral_base + sample];
                if matches!(input.method, Method::Arctic) {
                    energy *= input.reference_illuminant[sample];
                }
                for (channel, out) in raw.iter_mut().enumerate() {
                    *out += energy * input.sensitivity_rgb[sample][channel];
                }
            }
            for channel in 0..3 {
                let mut value = raw[channel] / denominator[channel];
                if let Some((params, center)) = correction {
                    value *= eval_hanatos_surface(&params[channel], [tc_c, tc_m], center).exp2();
                }
                if !value.is_finite() {
                    return Err(TcError::NonfiniteIntegratedSample);
                }
                integrated[(c * 192 + m) * 4 + channel] = value;
            }
        }
    }
    if input.input_compression_active {
        integrated = crate::gamut::remap_tc_lut(
            integrated
                .as_slice()
                .try_into()
                .expect("complete pre-remap table"),
            input.input_hull.ok_or(TcError::MissingInputHull)?,
        )?;
    }
    Ok(FilmTcLut {
        samples: integrated,
    })
}
pub fn sample_tc_lut(
    samples: &[f32; TC_LUT_SAMPLE_COUNT],
    xyz: [f32; 3],
) -> Result<[f32; 3], TcError> {
    let clean = xyz.map(|v| if v.is_finite() { v } else { 0.0 });
    let brightness = (clean[0] + clean[1]) + clean[2];
    let denominator = if brightness < 1e-10 {
        1e-10
    } else {
        brightness
    };
    let tc = tri2quad(clean[0] / denominator, clean[1] / denominator);
    let mut out = [0.0; 3];
    for (channel, value) in out.iter_mut().enumerate() {
        let sampled = sample_mitchell(samples, tc, 4, channel)?;
        let result = brightness * sampled;
        *value = if result.is_finite() { result } else { 0.0 };
    }
    Ok(out)
}

impl std::fmt::Display for TcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.diagnostic())
    }
}
impl std::error::Error for TcError {}
