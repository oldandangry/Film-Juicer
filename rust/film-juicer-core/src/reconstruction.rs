//! Invocation-local Hanatos reference reconstruction from the immutable tensor.

use std::fmt;

pub const HANATOS_SAMPLE_COUNT: usize = 192 * 192 * 81;

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
    let x = 1.0 - tx;
    // std::max(a,b) preserves a on an unordered comparison.
    let denominator = if x < 1e-10 { 1e-10 } else { x };
    let c = (x * x).clamp(0.0, 1.0);
    let m = (ty / denominator).clamp(0.0, 1.0);
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

fn mitchell_coordinate(normalized: f32) -> (i32, f32) {
    let value = normalized.clamp(0.0, 1.0) * 191.0;
    if value >= 191.0 {
        return (190, 1.0);
    }
    let base = value.floor() as i32;
    (base, value - base as f32)
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
    spectra: &[f32; HANATOS_SAMPLE_COUNT],
    tc: [f32; 2],
) -> Result<[f32; 81], Error> {
    let (c_base, c_fraction) = mitchell_coordinate(tc[0]);
    let (m_base, m_fraction) = mitchell_coordinate(tc[1]);
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
    spectra: &[f32; HANATOS_SAMPLE_COUNT],
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
