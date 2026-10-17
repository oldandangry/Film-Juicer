//! Capture-film DIR chemistry and compensated development axes (RGB).

use std::fmt;
use std::mem::size_of;

use crate::hash;
use crate::profile::Polarity;
use super::Development;

pub struct Params {
    pub active: bool,
    pub amount: f32,
    pub inhibition_same_layer: f32,
    pub inhibition_interlayer: f32,
    pub diffusion_size_um: f32,
    pub diffusion_tail_um: f32,
    pub diffusion_tail_weight: f32,
    pub gamma_use_stock: bool,
    pub gamma_same_layer_rgb: [f32; 3],
    pub gamma_interlayer_r_to_gb: [f32; 2],
    pub gamma_interlayer_g_to_rb: [f32; 2],
    pub gamma_interlayer_b_to_rg: [f32; 2],
    pub langmuir_donor_k_rgb: [f32; 3],
    pub langmuir_receiver_k_rgb: [f32; 3],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mode {
    Inactive,
    NegativeDonorLangmuir,
    PositiveReceiverLangmuir,
}

#[derive(Debug)]
pub enum Error {
    Matrix,
    LogExposure { sample: usize },
    DensityMaximum,
    ReceiverK { channel: usize },
    DonorK { channel: usize },
    DonorLangmuir { channel: usize, sample: usize },
    ReceiverLangmuir { channel: usize, sample: usize },
    CompensatedAxis { channel: usize, sample: usize },
    Size,
    Allocation,
    ZeroIdentity,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Matrix => f.write_str("DIR matrix requires finite result"),
            Self::LogExposure { sample } => write!(
                f,
                "DIR log_exposure requires strictly_increasing finite span sample={sample}"
            ),
            Self::DensityMaximum => {
                f.write_str("DIR density maximum requires finite positive result")
            }
            Self::ReceiverK { channel } => write!(
                f,
                "DIR receiver knee requires finite positive coupled result channel={channel}"
            ),
            Self::DonorK { channel } => write!(
                f,
                "DIR donor knee requires finite positive result channel={channel}"
            ),
            Self::DonorLangmuir { channel, sample } => write!(
                f,
                "DIR donor Langmuir requires finite defined result channel={channel} sample={sample}"
            ),
            Self::ReceiverLangmuir { channel, sample } => write!(
                f,
                "DIR receiver Langmuir requires finite defined result channel={channel} sample={sample}"
            ),
            Self::CompensatedAxis { channel, sample } => write!(
                f,
                "DIR compensated axis requires finite f32 result channel={channel} sample={sample}"
            ),
            Self::Size => f.write_str("DIR axis exceeds addressable storage"),
            Self::Allocation => f.write_str("DIR axis reservation failed"),
            Self::ZeroIdentity => f.write_str("DIR identity is zero"),
        }
    }
}
impl std::error::Error for Error {}

pub struct Recipe {
    polarity: Option<Polarity>,
    mode: Mode,
    matrix_rgb: [[f32; 3]; 3],
    diffusion_size_um: f32,
    diffusion_tail_um: f32,
    diffusion_tail_weight: f32,
    density_max_rgb: [f32; 3],
    density_ref_rgb: [f32; 3],
    donor_k_rgb: [f32; 3],
    receiver_c_ref_rgb: [f32; 3],
    receiver_kr_rgb: [f32; 3],
    compensated_axes_rgb: [Vec<f32>; 3],
    compensated_axes_hash: u64,
    hash: u64,
}

fn langmuir(value: f64, knee: f64, reference: f64) -> Option<f64> {
    let denominator = knee + value;
    let result = value * (knee + reference) / denominator;
    (denominator != 0.0 && result.is_finite()).then_some(result)
}

impl Recipe {
    fn inactive(polarity: Option<Polarity>) -> Self {
        Self {
            polarity,
            mode: Mode::Inactive,
            matrix_rgb: [[0.0; 3]; 3],
            diffusion_size_um: 0.0,
            diffusion_tail_um: 0.0,
            diffusion_tail_weight: 0.0,
            density_max_rgb: [0.0; 3],
            density_ref_rgb: [0.0; 3],
            donor_k_rgb: [0.0; 3],
            receiver_c_ref_rgb: [0.0; 3],
            receiver_kr_rgb: [0.0; 3],
            compensated_axes_rgb: std::array::from_fn(|_| Vec::new()),
            compensated_axes_hash: 0,
            hash: 0,
        }
    }

    pub fn new(development: &Development, params: Params) -> Result<Self, Error> {
        let mut result = Self::inactive(Some(development.polarity()));
        if !(params.active && params.amount > 0.0) {
            return Ok(result);
        }
        result.diffusion_size_um = params.diffusion_size_um;
        if result.diffusion_size_um > 0.0 {
            result.diffusion_tail_weight = params.diffusion_tail_weight;
            if result.diffusion_tail_weight > 0.0 {
                result.diffusion_tail_um = params.diffusion_tail_um;
            }
        }
        let defaults = development.profile.processing_defaults();
        let (same, red, green, blue) = if params.gamma_use_stock {
            (
                defaults.gamma_samelayer_rgb,
                defaults.gamma_interlayer_r_to_gb,
                defaults.gamma_interlayer_g_to_rb,
                defaults.gamma_interlayer_b_to_rg,
            )
        } else {
            (
                params.gamma_same_layer_rgb,
                params.gamma_interlayer_r_to_gb,
                params.gamma_interlayer_g_to_rb,
                params.gamma_interlayer_b_to_rg,
            )
        };
        let mut unit = [[0.0; 3]; 3];
        for channel in 0..3 {
            unit[channel][channel] = same[channel] * params.inhibition_same_layer;
        }
        unit[0][1] = red[0] * params.inhibition_interlayer;
        unit[0][2] = red[1] * params.inhibition_interlayer;
        unit[1][0] = green[0] * params.inhibition_interlayer;
        unit[1][2] = green[1] * params.inhibition_interlayer;
        unit[2][0] = blue[0] * params.inhibition_interlayer;
        unit[2][1] = blue[1] * params.inhibition_interlayer;
        result.matrix_rgb = unit;
        for row in &mut result.matrix_rgb {
            for value in row {
                *value *= params.amount;
                if !value.is_finite() {
                    return Err(Error::Matrix);
                }
            }
        }
        if !result.matrix_rgb.iter().flatten().any(|v| *v != 0.0) {
            return Ok(Self::inactive(None));
        }
        let axis = development.log_exposure();
        // Development permits duplicate samples and ordered infinities. Active
        // DIR requires usable original spans at this later operation boundary.
        for sample in 1..axis.len() {
            let span = f64::from(axis[sample]) - f64::from(axis[sample - 1]);
            if !span.is_finite() || span <= 0.0 {
                return Err(Error::LogExposure { sample });
            }
        }
        result.density_max_rgb = [f32::NEG_INFINITY; 3];
        for row in development.normalized_curves_cmy() {
            for (channel, &value) in row.iter().enumerate() {
                if value.is_finite() && result.density_max_rgb[channel] < value {
                    result.density_max_rgb[channel] = value;
                }
            }
        }
        if result
            .density_max_rgb
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(Error::DensityMaximum);
        }
        result.density_ref_rgb = result.density_max_rgb.map(|v| 0.5 * v);
        let positive = development.polarity() == Polarity::Positive;
        result.mode = if positive {
            Mode::PositiveReceiverLangmuir
        } else {
            Mode::NegativeDonorLangmuir
        };
        if positive {
            for receiver in 0..3 {
                let mut reference = 0.0_f64;
                for (donor, row) in unit.iter().enumerate() {
                    reference +=
                        f64::from(result.density_ref_rgb[donor]) * f64::from(row[receiver]);
                }
                result.receiver_c_ref_rgb[receiver] = reference as f32;
                result.receiver_kr_rgb[receiver] = params.langmuir_receiver_k_rgb[receiver]
                    * (2.0 * result.receiver_c_ref_rgb[receiver]);
                if !result.receiver_c_ref_rgb[receiver].is_finite()
                    || !result.receiver_kr_rgb[receiver].is_finite()
                    || (result.receiver_c_ref_rgb[receiver] > 0.0
                        && result.receiver_kr_rgb[receiver] <= 0.0)
                {
                    return Err(Error::ReceiverK { channel: receiver });
                }
            }
        } else {
            for donor in 0..3 {
                result.donor_k_rgb[donor] =
                    params.langmuir_donor_k_rgb[donor] * result.density_max_rgb[donor];
                if !result.donor_k_rgb[donor].is_finite() || result.donor_k_rgb[donor] <= 0.0 {
                    return Err(Error::DonorK { channel: donor });
                }
            }
        }
        #[cfg(feature = "test-support")]
        match test_support::take_allocation() {
            1 => return Err(Error::Size),
            2 => return Err(Error::Allocation),
            _ => {}
        }
        let bytes = axis
            .len()
            .checked_mul(size_of::<f32>())
            .ok_or(Error::Size)?;
        if bytes > isize::MAX as usize {
            return Err(Error::Size);
        }
        for compensated in &mut result.compensated_axes_rgb {
            compensated
                .try_reserve_exact(axis.len())
                .map_err(|_| Error::Allocation)?;
            compensated.resize(axis.len(), 0.0);
        }
        for (sample, &exposure) in axis.iter().enumerate() {
            let mut donors = [0.0_f64; 3];
            for (donor, value) in donors.iter_mut().enumerate() {
                let density = f64::from(development.normalized_curves_cmy()[sample][donor]);
                *value = if positive {
                    f64::from(result.density_max_rgb[donor]) - density
                } else {
                    langmuir(
                        density,
                        f64::from(result.donor_k_rgb[donor]),
                        f64::from(result.density_ref_rgb[donor]),
                    )
                    .ok_or(Error::DonorLangmuir {
                        channel: donor,
                        sample,
                    })?
                };
            }
            for receiver in 0..3 {
                let mut correction = 0.0_f64;
                for (donor, &value) in donors.iter().enumerate() {
                    correction += value * f64::from(result.matrix_rgb[donor][receiver]);
                }
                if positive {
                    correction = if result.receiver_c_ref_rgb[receiver] == 0.0 {
                        0.0
                    } else {
                        langmuir(
                            correction,
                            f64::from(result.receiver_kr_rgb[receiver]),
                            f64::from(result.receiver_c_ref_rgb[receiver]),
                        )
                        .ok_or(Error::ReceiverLangmuir {
                            channel: receiver,
                            sample,
                        })?
                    };
                }
                let coordinate = f64::from(exposure) - correction;
                if !coordinate.is_finite()
                    || coordinate < -f64::from(f32::MAX)
                    || coordinate > f64::from(f32::MAX)
                {
                    return Err(Error::CompensatedAxis {
                        channel: receiver,
                        sample,
                    });
                }
                result.compensated_axes_rgb[receiver][sample] = coordinate as f32;
            }
        }
        let mut axes_hash = hash::FNV_OFFSET;
        for value in [1_u64, development.normalized_curves_hash()] {
            hash::update_bytes(&mut axes_hash, &value.to_le_bytes());
        }
        for compensated in &result.compensated_axes_rgb {
            let pair = hash::f32s_with_nan_mask(compensated);
            hash::update_bytes(
                &mut axes_hash,
                &hash::u64s(&[pair.values, pair.nan_mask]).to_le_bytes(),
            );
        }
        result.compensated_axes_hash = axes_hash;
        let mut identity = hash::FNV_OFFSET;
        hash::update_bytes(&mut identity, &3_u64.to_le_bytes());
        hash::update_bytes(
            &mut identity,
            &[u8::from(positive), if positive { 2 } else { 1 }],
        );
        for values in result.matrix_rgb.iter().chain([
            &result.density_max_rgb,
            &result.density_ref_rgb,
            &result.donor_k_rgb,
            &result.receiver_c_ref_rgb,
            &result.receiver_kr_rgb,
        ]) {
            for value in values {
                hash::update_bytes(&mut identity, &value.to_bits().to_le_bytes());
            }
        }
        hash::update_bytes(&mut identity, &axes_hash.to_le_bytes());
        if result.diffusion_size_um > 0.0 {
            hash::update_bytes(
                &mut identity,
                &result.diffusion_size_um.to_bits().to_le_bytes(),
            );
            hash::update_bytes(
                &mut identity,
                &result.diffusion_tail_weight.to_bits().to_le_bytes(),
            );
            if result.diffusion_tail_weight > 0.0 {
                hash::update_bytes(
                    &mut identity,
                    &result.diffusion_tail_um.to_bits().to_le_bytes(),
                );
            }
        }
        result.hash = identity;
        #[cfg(feature = "test-support")]
        if test_support::take_identity() {
            result.hash = 0;
        }
        if result.hash == 0 || result.compensated_axes_hash == 0 {
            return Err(Error::ZeroIdentity);
        }
        Ok(result)
    }

    pub fn active(&self) -> bool {
        self.mode != Mode::Inactive
    }
    pub fn polarity(&self) -> Option<Polarity> {
        self.polarity
    }
    pub fn mode(&self) -> Mode {
        self.mode
    }
    pub fn matrix_rgb(&self) -> &[[f32; 3]; 3] {
        &self.matrix_rgb
    }
    pub fn diffusion_size_um(&self) -> f32 {
        self.diffusion_size_um
    }
    pub fn diffusion_tail_um(&self) -> f32 {
        self.diffusion_tail_um
    }
    pub fn diffusion_tail_weight(&self) -> f32 {
        self.diffusion_tail_weight
    }
    pub fn density_max_rgb(&self) -> &[f32; 3] {
        &self.density_max_rgb
    }
    pub fn density_ref_rgb(&self) -> &[f32; 3] {
        &self.density_ref_rgb
    }
    pub fn donor_k_rgb(&self) -> &[f32; 3] {
        &self.donor_k_rgb
    }
    pub fn receiver_c_ref_rgb(&self) -> &[f32; 3] {
        &self.receiver_c_ref_rgb
    }
    pub fn receiver_kr_rgb(&self) -> &[f32; 3] {
        &self.receiver_kr_rgb
    }
    pub fn compensated_axes_rgb(&self) -> &[Vec<f32>; 3] {
        &self.compensated_axes_rgb
    }
    pub fn compensated_axes_hash(&self) -> u64 {
        self.compensated_axes_hash
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

#[cfg(feature = "test-support")]
pub mod test_support {
    use std::cell::Cell;
    thread_local! { static NEXT: Cell<u32> = const { Cell::new(0) }; }
    pub fn reject_next(fault: u32) {
        NEXT.set(fault);
    }
    pub(super) fn take_allocation() -> u32 {
        let fault = NEXT.get();
        if matches!(fault, 1 | 2) {
            NEXT.set(0);
            fault
        } else {
            0
        }
    }
    pub(super) fn take_identity() -> bool {
        if NEXT.get() == 3 {
            NEXT.set(0);
            true
        } else {
            false
        }
    }
}
