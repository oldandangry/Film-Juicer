//! Final film sensitivity, its identity and the distinct recipe Mallett scale.

use std::fmt;

use crate::hash;

pub struct BandPass {
    pub uv: [f32; 3],
    pub ir: [f32; 3],
}
pub struct Window<'a> {
    pub params: &'a [f32; 4],
    pub reference_white: Option<&'a [f32; 81]>,
}
pub enum Method<'a> {
    Hanatos { window: Option<Window<'a>> },
    Mallett,
    Arctic,
}
pub struct Input<'a> {
    pub linear_sensitivity_rgb: &'a [[f32; 3]; 81],
    pub reference_illuminant: &'a [f32; 81],
    pub band_pass: Option<BandPass>,
    pub method: Method<'a>,
}
pub struct Sensitivity {
    values_rgb: [[f32; 3]; 81],
    hash: u64,
    mallett_green_scale: f32,
}
impl Sensitivity {
    pub fn values_rgb(&self) -> &[[f32; 3]; 81] {
        &self.values_rgb
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
    pub fn mallett_green_scale(&self) -> f32 {
        self.mallett_green_scale
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HashFailure {
    NonfiniteOperand(usize),
    ZeroIdentity,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErrorKind {
    ReferenceIlluminant,
    BandPassResponse,
    MissingReference,
    WindowSample,
    WindowResponse,
    AdaptedSensitivity,
    Hash,
    MallettResponse,
    MallettScale,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Error {
    kind: ErrorKind,
    hash_failure: Option<HashFailure>,
}
impl Error {
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }
    pub fn hash_failure(&self) -> Option<HashFailure> {
        self.hash_failure
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MalformedRequiredProfileData phase=3B field=final_sensitivity requirement={:?}",
            self.kind
        )
    }
}
impl std::error::Error for Error {}

#[expect(
    clippy::neg_cmp_op_on_partial_ord,
    reason = "native amplitude guard also bypasses NaN"
)]
fn camera_filter_sample(wavelength: f32, filter: [f32; 3], uv: bool) -> f32 {
    let amplitude = filter[0].clamp(0.0, 1.0);
    if !(amplitude > 0.0) {
        return 1.0;
    }
    let mut width = filter[2];
    if !width.is_finite() || width.abs() < 1e-6 {
        width = if uv { 1e-6 } else { -1e-6 };
    }
    width = if uv { width.abs() } else { -width.abs() };
    let sigmoid = 0.5 * (libm::erff((wavelength - filter[1]) / width) + 1.0);
    1.0 - amplitude + amplitude * sigmoid
}

#[expect(
    clippy::excessive_precision,
    clippy::approx_constant,
    reason = "preserve the accepted native f32 sqrt(2) literal and narrowing"
)]
fn hanatos_window_sample(wavelength: f32, params: &[f32; 4]) -> f32 {
    const SQRT2: f32 = 1.4142135623730950488;
    let uv = 0.5 * (1.0 + libm::erff((wavelength - params[0]) / (params[1] * SQRT2)));
    let ir = 0.5 * (1.0 - libm::erff((wavelength - params[2]) / (params[3] * SQRT2)));
    uv * ir
}

fn finite_source(source: f32) -> f64 {
    if source.is_finite() && source > 0.0 {
        f64::from(source)
    } else {
        0.0
    }
}

pub fn prepare_sensitivity(input: Input<'_>) -> Result<Sensitivity, Error> {
    let failure = |kind| Error {
        kind,
        hash_failure: None,
    };
    let mut unfiltered = [0.0_f64; 3];
    let mut filtered = [0.0_f64; 3];
    let mut band_pass = [0.0_f32; 81];
    for (i, filter_out) in band_pass.iter_mut().enumerate() {
        let wavelength = 380.0 + 5.0 * i as f32;
        let filter = if let Some(filter) = &input.band_pass {
            camera_filter_sample(wavelength, filter.uv, true)
                * camera_filter_sample(wavelength, filter.ir, false)
        } else {
            1.0
        };
        *filter_out = if filter.is_finite() && filter > 0.0 {
            filter
        } else {
            0.0
        };
        let illuminant = input.reference_illuminant[i];
        if !illuminant.is_finite() || illuminant < 0.0 {
            return Err(failure(ErrorKind::ReferenceIlluminant));
        }
        for (channel, &sensitivity) in input.linear_sensitivity_rgb[i].iter().enumerate() {
            let finite = finite_source(sensitivity);
            unfiltered[channel] += finite * f64::from(illuminant);
            filtered[channel] += finite * f64::from(*filter_out) * f64::from(illuminant);
        }
    }
    let mut normalization = [0.0_f64; 3];
    for (channel, out) in normalization.iter_mut().enumerate() {
        if !(unfiltered[channel].is_finite()
            && unfiltered[channel] > 0.0
            && filtered[channel].is_finite()
            && filtered[channel] > 0.0)
        {
            return Err(failure(ErrorKind::BandPassResponse));
        }
        *out = filtered[channel] / unfiltered[channel];
    }
    let mut values = [[0.0_f32; 3]; 81];
    for (i, rgb) in values.iter_mut().enumerate() {
        for (channel, out) in rgb.iter_mut().enumerate() {
            let derived = finite_source(input.linear_sensitivity_rgb[i][channel])
                * f64::from(band_pass[i])
                / normalization[channel];
            *out = if derived.is_finite() && derived > 0.0 {
                derived as f32
            } else {
                0.0
            };
        }
    }
    if let Method::Hanatos {
        window: Some(window),
    } = &input.method
    {
        let reference = window
            .reference_white
            .ok_or_else(|| failure(ErrorKind::MissingReference))?;
        let mut response = [0.0_f64; 3];
        let mut windowed = [0.0_f64; 3];
        let mut samples = [0.0_f32; 81];
        for (i, sample_out) in samples.iter_mut().enumerate() {
            let sample = hanatos_window_sample(380.0 + 5.0 * i as f32, window.params);
            if !sample.is_finite() || sample < 0.0 {
                return Err(failure(ErrorKind::WindowSample));
            }
            *sample_out = sample;
            for (channel, &sensitivity) in values[i].iter().enumerate() {
                let weighted = f64::from(sensitivity) * f64::from(reference[i]);
                response[channel] += weighted;
                windowed[channel] += weighted * f64::from(sample);
            }
        }
        for (channel, out) in normalization.iter_mut().enumerate() {
            if !(response[channel].is_finite()
                && response[channel] > 0.0
                && windowed[channel].is_finite()
                && windowed[channel] > 0.0)
            {
                return Err(failure(ErrorKind::WindowResponse));
            }
            *out = windowed[channel] / response[channel];
        }
        for (i, rgb) in values.iter_mut().enumerate() {
            for (channel, out) in rgb.iter_mut().enumerate() {
                let adapted = f64::from(*out) * f64::from(samples[i]) / normalization[channel];
                if !adapted.is_finite() || adapted < 0.0 {
                    return Err(failure(ErrorKind::AdaptedSensitivity));
                }
                *out = adapted as f32;
            }
        }
    }
    let (hash, hash_failure) = match hash::finite_f32s(values.as_flattened()) {
        Ok(0) => (0, Some(HashFailure::ZeroIdentity)),
        Ok(hash) => (hash, None),
        Err(e) => (0, Some(HashFailure::NonfiniteOperand(e.index))),
    };
    let failure = |kind| Error { kind, hash_failure };
    let mut mallett_green_scale = 1.0;
    if matches!(input.method, Method::Mallett) {
        let mut green = 0.0_f64;
        for (i, rgb) in values.iter().enumerate() {
            green += 0.184 * f64::from(input.reference_illuminant[i]) * f64::from(rgb[1]);
        }
        if !green.is_finite() || green <= 0.0 {
            return Err(failure(ErrorKind::MallettResponse));
        }
        mallett_green_scale = (1.0 / green) as f32;
        if !mallett_green_scale.is_finite() || mallett_green_scale <= 0.0 {
            return Err(failure(ErrorKind::MallettScale));
        }
    }
    if hash_failure.is_some() {
        return Err(failure(ErrorKind::Hash));
    }
    Ok(Sensitivity {
        values_rgb: values,
        hash,
        mallett_green_scale,
    })
}

#[cfg(feature = "test-support")]
pub fn window_sample_for_test(wavelength: f32, params: &[f32; 4]) -> f32 {
    hanatos_window_sample(wavelength, params)
}
