//! Completed scanner black/white correction. Inputs are synchronous borrows.

use crate::hash;
use std::fmt;

#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub route: u8,
    pub black_correction: bool,
    pub white_correction: bool,
    pub black_level: f32,
    pub white_level: f32,
}

impl Params {
    pub fn direct_inputs_required(self) -> Result<bool, CorrectionError> {
        if !self.black_correction && !self.white_correction || self.route == 0 {
            Ok(false)
        } else if self.route == 2 {
            Ok(true)
        } else {
            Err(CorrectionError::DirectInputs)
        }
    }

    pub fn print_inputs_required(self) -> Result<bool, CorrectionError> {
        if !self.black_correction && !self.white_correction {
            Ok(false)
        } else if self.route == 1 || self.route == 3 {
            Ok(true)
        } else {
            // Includes revision 1's checked replacement of native byte exceptions.
            Err(CorrectionError::PrintInputs)
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ReferenceTables<'a> {
    pub eps_c: &'a [f32],
    pub eps_m: &'a [f32],
    pub eps_y: &'a [f32],
    pub weighted_y: &'a [f32],
    pub baseline_min: Option<&'a [f32]>,
    pub sample_count: i32,
    pub inv_yn: f32,
}

impl ReferenceTables<'_> {
    fn has_samples(self) -> bool {
        self.sample_count > 0
            && [self.eps_c, self.eps_m, self.eps_y, self.weighted_y]
                .iter()
                .all(|samples| samples.len() >= self.sample_count as usize)
            && self
                .baseline_min
                .is_none_or(|samples| samples.len() >= self.sample_count as usize)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DirectInput<'a> {
    pub tables: ReferenceTables<'a>,
    pub film_base_density: &'a [f32; 81],
    pub film_log_exposure: &'a [f32],
    pub film_authored_density_curves: &'a [[f32; 3]],
    pub minimum_density_cmy: [f32; 3],
    pub maximum_density_cmy: [f32; 3],
    pub inverse_span_cmy: [f32; 3],
    pub baseline_density_cmy: [f32; 3],
    pub reference_hash: u64,
    pub medium: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct PrintInput<'a> {
    pub tables: ReferenceTables<'a>,
    pub film_base_density: &'a [f32; 81],
    pub film_channel_density_cmy: &'a [[f32; 3]; 81],
    pub print_sensitivity_cmy: &'a [[f32; 3]; 81],
    pub filtered_main_illuminant: &'a [f32; 81],
    pub print_base_density: &'a [f32; 81],
    pub print_log_exposure: &'a [f32],
    pub print_density_curves: &'a [[f32; 3]],
    pub minimum_density_cmy: [f32; 3],
    pub inverse_span_cmy: [f32; 3],
    pub film_minimum_density_cmy: [f32; 3],
    pub film_maximum_density_cmy: [f32; 3],
    pub preflash_raw_cmy: [f32; 3],
    pub normalizer: f32,
    pub preflash_exposure: f32,
    pub medium: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceOutcome {
    Accepted,
    Midgray,
    Malformed,
    ZeroHash,
}

#[derive(Clone, Copy, Debug)]
pub struct Trace {
    outcome: TraceOutcome,
    // Reference black/white, target black/white, slope/offset, midgray, exposure.
    values: [f32; 8],
    hash: u64,
}

impl Trace {
    pub fn outcome(&self) -> TraceOutcome {
        self.outcome
    }
    pub fn values(&self) -> &[f32; 8] {
        &self.values
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

#[derive(Clone, Copy, Debug)]
pub enum CorrectionError {
    DirectInputs,
    DirectReference,
    DirectScan,
    DirectMidgray(Trace),
    DirectLookup,
    PrintInputs,
    PrintReferenceLookup,
    PrintScan,
    PrintMidgray(Trace),
    PrintLookup,
    Invalid(Trace),
    Identity(Trace),
}

impl CorrectionError {
    pub fn trace(&self) -> Option<&Trace> {
        match self {
            Self::DirectMidgray(trace)
            | Self::PrintMidgray(trace)
            | Self::Invalid(trace)
            | Self::Identity(trace) => Some(trace),
            _ => None,
        }
    }
}

impl fmt::Display for CorrectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::DirectInputs => "ResourceDescriptorMismatch phase=8B direct correction inputs",
            Self::DirectReference => {
                "ResourceDescriptorMismatch phase=8B direct synthetic reference"
            }
            Self::DirectScan => "MalformedRequiredProfileData phase=8B direct reference scan",
            Self::DirectMidgray(_) => {
                "MalformedRequiredProfileData phase=8B direct corrected midgray"
            }
            Self::DirectLookup => "ResourceDescriptorMismatch phase=8B direct density lookup",
            Self::PrintInputs => "ResourceDescriptorMismatch phase=8B print correction inputs",
            Self::PrintReferenceLookup => {
                "ResourceDescriptorMismatch phase=8B print reference density lookup"
            }
            Self::PrintScan => "MalformedRequiredProfileData phase=8B print reference scan",
            Self::PrintMidgray(_) => {
                "MalformedRequiredProfileData phase=8B print corrected midgray"
            }
            Self::PrintLookup => "ResourceDescriptorMismatch phase=8B print density lookup",
            Self::Invalid(_) => "MalformedRequiredProfileData phase=8B scanner correction",
            Self::Identity(_) => "",
        })
    }
}
impl std::error::Error for CorrectionError {}

#[derive(Clone, Copy, Debug)]
enum Storage {
    Inactive,
    Active {
        black: bool,
        white: bool,
        trace: Trace,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct Correction {
    route: u8,
    storage: Storage,
}

impl Correction {
    pub fn direct(params: Params, input: Option<DirectInput<'_>>) -> Result<Self, CorrectionError> {
        if !params.direct_inputs_required()? {
            return Ok(Self::inactive(params.route));
        }
        let input = input.ok_or(CorrectionError::DirectInputs)?;
        if !input.tables.has_samples() {
            return Err(CorrectionError::DirectInputs);
        }
        if input.reference_hash == 0 {
            return Err(CorrectionError::DirectReference);
        }
        let scan = |density| {
            reference_y(
                input.tables,
                input.medium,
                input.minimum_density_cmy,
                input.inverse_span_cmy,
                density,
            )
        };
        let black = scan(input.maximum_density_cmy).ok_or(CorrectionError::DirectScan)?;
        let white = scan([0.0; 3]).ok_or(CorrectionError::DirectScan)?;
        let coefficients = coefficients(params, black, white);
        let midgray = (0.184 - coefficients[3]) / coefficients[2];
        if !(midgray.is_finite() && midgray > 0.0) {
            return Err(CorrectionError::DirectMidgray(trace(
                TraceOutcome::Midgray,
                black,
                white,
                coefficients,
                midgray,
                f32::NAN,
            )));
        }
        let base = finite_mean(input.film_base_density);
        let density = base + finite_mean(&input.baseline_density_cmy);
        let corrected_density = density - midgray.log10() + 0.184_f32.log10();
        let lookup = |query| {
            interpolate(
                query,
                input.film_authored_density_curves.len(),
                input.film_log_exposure.len(),
                |i| -finite_mean(&input.film_authored_density_curves[i]),
                |i| input.film_log_exposure[i],
            )
        };
        let corrected = lookup(-(corrected_density - base));
        let baseline = lookup(-(density - base));
        let (Some(corrected), Some(baseline)) = (corrected, baseline) else {
            return Err(CorrectionError::DirectLookup);
        };
        let corrected_log_exposure = -corrected;
        let baseline_log_exposure = -baseline;
        Self::finish(
            params,
            black,
            white,
            10.0_f32.powf(baseline_log_exposure - corrected_log_exposure),
        )
    }

    pub fn print(params: Params, input: Option<PrintInput<'_>>) -> Result<Self, CorrectionError> {
        if !params.print_inputs_required()? {
            return Ok(Self::inactive(params.route));
        }
        let input = input.ok_or(CorrectionError::PrintInputs)?;
        if !input.tables.has_samples() || !input.normalizer.is_finite() {
            return Err(CorrectionError::PrintInputs);
        }
        let print_black = print_reference_density(input, input.film_minimum_density_cmy);
        let print_white = print_reference_density(input, input.film_maximum_density_cmy);
        let (Some(print_black), Some(print_white)) = (print_black, print_white) else {
            return Err(CorrectionError::PrintReferenceLookup);
        };
        let scan = |density| {
            reference_y(
                input.tables,
                input.medium,
                input.minimum_density_cmy,
                input.inverse_span_cmy,
                density,
            )
        };
        let black = scan(print_black).ok_or(CorrectionError::PrintScan)?;
        let white = scan(print_white).ok_or(CorrectionError::PrintScan)?;
        let coefficients = coefficients(params, black, white);
        let midgray = (0.184 - coefficients[3]) / coefficients[2];
        if !(midgray.is_finite() && midgray > 0.0) {
            return Err(CorrectionError::PrintMidgray(trace(
                TraceOutcome::Midgray,
                black,
                white,
                coefficients,
                midgray,
                f32::NAN,
            )));
        }
        let density = -0.184_f32.log10();
        let corrected_density = -midgray.log10();
        let base = finite_mean(input.print_base_density);
        let lookup = |query| {
            interpolate(
                query,
                input.print_density_curves.len(),
                input.print_log_exposure.len(),
                |i| finite_mean(&input.print_density_curves[i]),
                |i| input.print_log_exposure[i],
            )
        };
        let corrected = lookup(corrected_density - base);
        let baseline = lookup(density - base);
        let (Some(corrected), Some(baseline)) = (corrected, baseline) else {
            return Err(CorrectionError::PrintLookup);
        };
        Self::finish(params, black, white, 10.0_f32.powf(corrected - baseline))
    }

    fn inactive(route: u8) -> Self {
        Self {
            route,
            storage: Storage::Inactive,
        }
    }

    fn finish(
        params: Params,
        black: f32,
        white: f32,
        exposure: f32,
    ) -> Result<Self, CorrectionError> {
        let coefficients = coefficients(params, black, white);
        let midgray = if coefficients[2] != 0.0 {
            (0.184 - coefficients[3]) / coefficients[2]
        } else {
            f32::NAN
        };
        let mut trace = trace(
            TraceOutcome::Malformed,
            black,
            white,
            coefficients,
            midgray,
            exposure,
        );
        let valid_delta = white - black > 1.0e-6;
        if !black.is_finite()
            || !white.is_finite()
            || !valid_delta
            || !coefficients[2].is_finite()
            || !coefficients[3].is_finite()
            || !exposure.is_finite()
            || exposure <= 0.0
        {
            return Err(CorrectionError::Invalid(trace));
        }
        let mut bytes = [0_u8; 35];
        bytes[..3].copy_from_slice(&[
            params.route,
            u8::from(params.black_correction),
            u8::from(params.white_correction),
        ]);
        let values = [
            coefficients[0],
            coefficients[1],
            black,
            white,
            coefficients[2],
            coefficients[3],
            exposure,
        ];
        for (i, value) in values.iter().enumerate() {
            bytes[3 + 4 * i..7 + 4 * i].copy_from_slice(&value.to_bits().to_le_bytes());
        }
        bytes[31..].copy_from_slice(&1_u32.to_le_bytes());
        trace.hash = hash::bytes(&bytes);
        if trace.hash == 0 {
            trace.outcome = TraceOutcome::ZeroHash;
            return Err(CorrectionError::Identity(trace));
        }
        trace.outcome = TraceOutcome::Accepted;
        Ok(Self {
            route: params.route,
            storage: Storage::Active {
                black: params.black_correction,
                white: params.white_correction,
                trace,
            },
        })
    }

    pub fn route(&self) -> u8 {
        self.route
    }
    pub fn active(&self) -> bool {
        matches!(self.storage, Storage::Active { .. })
    }
    pub fn black_correction(&self) -> bool {
        matches!(self.storage, Storage::Active { black: true, .. })
    }
    pub fn white_correction(&self) -> bool {
        matches!(self.storage, Storage::Active { white: true, .. })
    }
    pub fn trace(&self) -> Option<&Trace> {
        match &self.storage {
            Storage::Inactive => None,
            Storage::Active { trace, .. } => Some(trace),
        }
    }
    fn value(&self, index: usize, inactive: f32) -> f32 {
        self.trace().map_or(inactive, |trace| trace.values[index])
    }
    pub fn target_black_linear(&self) -> f32 {
        self.value(2, 0.0)
    }
    pub fn target_white_linear(&self) -> f32 {
        self.value(3, 1.0)
    }
    pub fn reference_black_y(&self) -> f32 {
        self.value(0, 0.0)
    }
    pub fn reference_white_y(&self) -> f32 {
        self.value(1, 1.0)
    }
    pub fn xyz_slope(&self) -> f32 {
        self.value(4, 1.0)
    }
    pub fn xyz_offset(&self) -> f32 {
        self.value(5, 0.0)
    }
    pub fn exposure_scale(&self) -> f32 {
        self.value(7, 1.0)
    }
    pub fn schema_version(&self) -> u32 {
        1
    }
    pub fn hash(&self) -> u64 {
        self.trace().map_or(0, |trace| trace.hash)
    }
}

fn coefficients(params: Params, black: f32, white: f32) -> [f32; 4] {
    let target_black = if params.black_correction {
        decode_target(params.black_level)
    } else {
        black
    };
    let target_white = if params.white_correction {
        decode_target(params.white_level)
    } else {
        white
    };
    let slope = (target_white - target_black) / (white - black + 1.0e-10);
    let offset = target_black - slope * black;
    [target_black, target_white, slope, offset]
}

fn decode_target(level: f32) -> f32 {
    let level = level.clamp(0.0, 1.0);
    if level <= 0.04045 {
        level / 12.92
    } else {
        ((level + 0.055) / 1.055).powf(2.4)
    }
}

fn trace(
    outcome: TraceOutcome,
    black: f32,
    white: f32,
    coefficients: [f32; 4],
    midgray: f32,
    exposure: f32,
) -> Trace {
    Trace {
        outcome,
        values: [
            black,
            white,
            coefficients[0],
            coefficients[1],
            coefficients[2],
            coefficients[3],
            midgray,
            exposure,
        ],
        hash: 0,
    }
}

fn reference_y(
    tables: ReferenceTables<'_>,
    medium: u8,
    minimum: [f32; 3],
    inverse: [f32; 3],
    density: [f32; 3],
) -> Option<f32> {
    let mut denormalized = [0.0_f64; 3];
    for c in 0..3 {
        let range_min = if medium == 1 { minimum[c] } else { -minimum[c] };
        let normalized = if medium == 1 {
            (f64::from(density[c]) - f64::from(range_min)) * f64::from(inverse[c])
        } else {
            (f64::from(density[c]) + f64::from(range_min)) * f64::from(inverse[c])
        };
        // Preserve the native round trip, including zero/nonfinite inverse spans.
        denormalized[c] = if medium == 1 {
            normalized / f64::from(inverse[c]) + f64::from(range_min)
        } else {
            normalized / f64::from(inverse[c]) - f64::from(range_min)
        };
    }
    let mut y = 0.0_f64;
    for i in 0..tables.sample_count as usize {
        let base = tables
            .baseline_min
            .map_or(0.0, |samples| f64::from(samples[i]));
        let density = denormalized[0] * f64::from(tables.eps_c[i])
            + denormalized[1] * f64::from(tables.eps_m[i])
            + denormalized[2] * f64::from(tables.eps_y[i])
            + base;
        let transmitted = 10.0_f64.powf(-density);
        let ay = f64::from(tables.weighted_y[i]);
        if ay.is_finite() {
            let product = transmitted * ay;
            if !product.is_nan() {
                y += product;
            }
        }
    }
    y *= f64::from(tables.inv_yn);
    let log_y = (y.max(0.0) + 1.0e-10).log10();
    let y = 10.0_f64.powf(log_y);
    if y.is_finite() { Some(y as f32) } else { None }
}

fn finite_mean(samples: &[f32]) -> f32 {
    let mut sum = 0.0_f64;
    let mut count = 0_usize;
    for &sample in samples {
        if sample.is_finite() {
            sum += f64::from(sample);
            count += 1;
        }
    }
    if count > 0 {
        (sum / count as f64) as f32
    } else {
        0.0
    }
}

// Scanner's first inclusive bracket differs from synthetic-reference lookup.
fn interpolate(
    query: f32,
    x_len: usize,
    y_len: usize,
    x: impl Fn(usize) -> f32,
    y: impl Fn(usize) -> f32,
) -> Option<f32> {
    if query.is_nan() || x_len == 0 || x_len != y_len {
        return None;
    }
    let ascending = x_len < 2 || x(x_len - 1) >= x(0);
    let at = |i| if ascending { i } else { x_len - 1 - i };
    if query <= x(at(0)) {
        return Some(y(at(0)));
    }
    if query >= x(at(x_len - 1)) {
        return Some(y(at(y_len - 1)));
    }
    for i in 1..x_len {
        let (lo, hi) = (at(i - 1), at(i));
        if query >= x(lo) && query <= x(hi) {
            let span = x(hi) - x(lo);
            let t = if span > 0.0 {
                (query - x(lo)) / span
            } else {
                0.0
            };
            return Some(y(lo) + t * (y(hi) - y(lo)));
        }
    }
    None
}

fn sample_curve(query: f32, axis: &[f32], curves: &[[f32; 3]], channel: usize) -> Option<f32> {
    if channel >= 3 {
        return None;
    }
    interpolate(
        query,
        axis.len(),
        curves.len(),
        |i| axis[i],
        |i| curves[i][channel],
    )
}

fn print_reference_density(input: PrintInput<'_>, film_density: [f32; 3]) -> Option<[f32; 3]> {
    let mut raw = [0.0_f32; 3];
    for k in 0..81 {
        let density = input.film_base_density[k]
            + film_density[0] * input.film_channel_density_cmy[k][0]
            + film_density[1] * input.film_channel_density_cmy[k][1]
            + film_density[2] * input.film_channel_density_cmy[k][2];
        let transmitted = 10.0_f32.powf(-density) * input.filtered_main_illuminant[k];
        let light = if transmitted.is_nan() {
            0.0
        } else {
            transmitted
        };
        for (c, sum) in raw.iter_mut().enumerate() {
            let sensitivity = input.print_sensitivity_cmy[k][c];
            if sensitivity.is_finite() {
                *sum += light * sensitivity;
            }
        }
    }
    let mut density = [0.0; 3];
    for c in 0..3 {
        raw[c] = raw[c] * input.normalizer + input.preflash_raw_cmy[c] * input.preflash_exposure;
        density[c] = sample_curve(
            (raw[c].max(0.0) + 1.0e-10).log10(),
            input.print_log_exposure,
            input.print_density_curves,
            c,
        )?;
    }
    Some(density)
}

#[cfg(feature = "test-support")]
pub mod test_support {
    pub fn lookup(query: f32, axis: &[f32], values: &[f32]) -> Option<f32> {
        super::interpolate(query, axis.len(), values.len(), |i| axis[i], |i| values[i])
    }
    pub fn density_curve(
        query: f32,
        axis: &[f32],
        curves: &[[f32; 3]],
        channel: usize,
    ) -> Option<f32> {
        super::sample_curve(query, axis, curves, channel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[cfg(all(not(target_os = "windows"), debug_assertions))]
    const PRESET: &str = "linux-debug";
    #[cfg(all(not(target_os = "windows"), not(debug_assertions)))]
    const PRESET: &str = "linux-release";
    #[cfg(all(target_os = "windows", debug_assertions))]
    const PRESET: &str = "windows-clang-debug";
    #[cfg(all(target_os = "windows", not(debug_assertions)))]
    const PRESET: &str = "windows-clang-release";

    fn floats(value: &Value) -> Vec<f32> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|word| f32::from_bits(word.as_u64().unwrap() as u32))
            .collect()
    }
    fn fixed<const N: usize>(value: &Value) -> [f32; N] {
        floats(value).try_into().unwrap()
    }
    fn rows(value: &Value) -> Vec<[f32; 3]> {
        let flat = floats(value);
        let (rows, remainder) = flat.as_chunks::<3>();
        assert!(remainder.is_empty());
        rows.to_vec()
    }
    fn scalar(value: &Value) -> f32 {
        f32::from_bits(value.as_u64().unwrap() as u32)
    }
    fn reason(error: CorrectionError) -> u32 {
        match error {
            CorrectionError::DirectInputs => 2,
            CorrectionError::DirectReference => 3,
            CorrectionError::DirectScan => 4,
            CorrectionError::DirectMidgray(_) => 5,
            CorrectionError::DirectLookup => 6,
            CorrectionError::PrintInputs => 7,
            CorrectionError::PrintReferenceLookup => 8,
            CorrectionError::PrintScan => 9,
            CorrectionError::PrintMidgray(_) => 10,
            CorrectionError::PrintLookup => 11,
            CorrectionError::Invalid(_) => 12,
            CorrectionError::Identity(_) => 13,
        }
    }

    // The returned completed result outlives all decoded input storage here.
    fn construct(input: &Value, print: bool) -> Result<Correction, CorrectionError> {
        let flags = input["flags"].as_u64().unwrap();
        let levels = fixed::<2>(&input["levels"]);
        let params = Params {
            route: input["route"].as_u64().unwrap() as u8,
            black_correction: flags & 1 != 0,
            white_correction: flags & 2 != 0,
            black_level: levels[0],
            white_level: levels[1],
        };
        if input.get("film_base").is_none() {
            return if print {
                Correction::print(params, None)
            } else {
                Correction::direct(params, None)
            };
        }
        let table = &input["tables"];
        let c = floats(&table["c"]);
        let m = floats(&table["m"]);
        let y = floats(&table["y"]);
        let ay = floats(&table["ay"]);
        let baseline = floats(&table["baseline"]);
        let tables = ReferenceTables {
            eps_c: &c,
            eps_m: &m,
            eps_y: &y,
            weighted_y: &ay,
            baseline_min: table["has_baseline"]
                .as_bool()
                .unwrap()
                .then_some(baseline.as_slice()),
            sample_count: table["k"].as_i64().unwrap() as i32,
            inv_yn: scalar(&table["inv_yn"]),
        };
        let base = fixed(&input["film_base"]);
        if print {
            let dyes = rows(&input["film_dyes"]);
            let sensitivity = rows(&input["sensitivity"]);
            let print_base = fixed(&input["print_base"]);
            let axis = floats(&input["print_axis"]);
            let curves = rows(&input["print_curves"]);
            let light = fixed(&input["light"]);
            Correction::print(
                params,
                Some(PrintInput {
                    tables,
                    film_base_density: &base,
                    film_channel_density_cmy: dyes.as_slice().try_into().unwrap(),
                    print_sensitivity_cmy: sensitivity.as_slice().try_into().unwrap(),
                    filtered_main_illuminant: &light,
                    print_base_density: &print_base,
                    print_log_exposure: &axis,
                    print_density_curves: &curves,
                    minimum_density_cmy: fixed(&input["minimum"]),
                    inverse_span_cmy: fixed(&input["inverse"]),
                    film_minimum_density_cmy: fixed(&input["film_minimum"]),
                    film_maximum_density_cmy: fixed(&input["film_maximum"]),
                    preflash_raw_cmy: fixed(&input["preflash"]),
                    normalizer: scalar(&input["normalizer"]),
                    preflash_exposure: scalar(&input["preflash_exposure"]),
                    medium: input["medium"].as_u64().unwrap() as u8,
                }),
            )
        } else {
            let axis = floats(&input["film_axis"]);
            let curves = rows(&input["authored_curves"]);
            Correction::direct(
                params,
                Some(DirectInput {
                    tables,
                    film_base_density: &base,
                    film_log_exposure: &axis,
                    film_authored_density_curves: &curves,
                    minimum_density_cmy: fixed(&input["minimum"]),
                    maximum_density_cmy: fixed(&input["maximum"]),
                    inverse_span_cmy: fixed(&input["inverse"]),
                    baseline_density_cmy: fixed(&input["baseline_density"]),
                    reference_hash: input["reference_hash"].as_u64().unwrap(),
                    medium: input["medium"].as_u64().unwrap() as u8,
                }),
            )
        }
    }

    fn compare(result: Correction, expected: &Value, id: &str) {
        let observed = serde_json::json!({"route":result.route(), "active":result.active(),
            "black":result.black_correction(), "white":result.white_correction(), "schema":result.schema_version(), "hash":result.hash(),
            "words":[result.target_black_linear().to_bits(),result.target_white_linear().to_bits(),
                result.reference_black_y().to_bits(),result.reference_white_y().to_bits(),result.xyz_slope().to_bits(),
                result.xyz_offset().to_bits(),result.exposure_scale().to_bits()]});
        assert_eq!(&observed, expected, "{PRESET} {id}");
        assert_eq!(result.trace().is_some(), result.active());
    }

    #[test]
    fn exact_native_correction_and_actual_resource_inputs() {
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/scanner/correction-fixtures");
        let read = |path| {
            serde_json::from_str::<Value>(&std::fs::read_to_string(directory.join(path)).unwrap())
                .unwrap()
        };
        let manifest = read("manifest.json");
        let captured = read(manifest["presets"][PRESET]["native"].as_str().unwrap());
        let checked = read("checked-expectations.json");
        assert_eq!(captured["cases"].as_array().unwrap().len(), 150);
        assert_eq!(captured["products"].as_array().unwrap().len(), 48);
        for row in captured["cases"].as_array().unwrap() {
            let id = row["id"].as_str().unwrap();
            let normative = &checked["cases"][id];
            if normative["native_only"].as_bool().unwrap() {
                continue;
            }
            match construct(&row["input"], row["spec"]["print"].as_bool().unwrap()) {
                Ok(value) => {
                    assert_eq!(normative["reason"], 0, "{id}");
                    compare(value, &row["native"]["descriptor"], id);
                }
                Err(error) => {
                    assert_eq!(
                        u64::from(reason(error)),
                        normative["reason"].as_u64().unwrap(),
                        "{PRESET} {id}: {error}"
                    );
                    if normative["intentional_exception_replacement"]
                        .as_bool()
                        .unwrap()
                    {
                        assert_eq!(row["native"]["native_outcome"], "exception");
                        assert!(row["native"]["native_bool"].is_null());
                        assert_eq!(
                            error.to_string(),
                            checked["print_byte_diagnostic"].as_str().unwrap()
                        );
                    } else {
                        assert_eq!(
                            error.to_string(),
                            row["native"]["diagnostic"].as_str().unwrap(),
                            "{id}"
                        );
                    }
                }
            }
        }
        for row in captured["products"].as_array().unwrap() {
            let id = row["id"].as_str().unwrap();
            compare(
                construct(&row["input"], row["print"].as_bool().unwrap()).unwrap(),
                &row["native"]["descriptor"],
                id,
            );
        }
    }

    #[test]
    fn demand_precedes_missing_storage_for_every_byte_and_flag_pair() {
        for route in 0..=255 {
            for flags in 0..4 {
                let params = Params {
                    route,
                    black_correction: flags & 1 != 0,
                    white_correction: flags & 2 != 0,
                    black_level: f32::NAN,
                    white_level: f32::NAN,
                };
                for print in [false, true] {
                    let demand = if print {
                        params.print_inputs_required()
                    } else {
                        params.direct_inputs_required()
                    };
                    let result = if print {
                        Correction::print(params, None)
                    } else {
                        Correction::direct(params, None)
                    };
                    if flags == 0 || !print && route == 0 {
                        assert!(!demand.unwrap());
                        let result = result.unwrap();
                        assert_eq!(result.route(), route);
                        assert!(!result.active());
                        assert_eq!(result.hash(), 0);
                        assert_eq!(result.exposure_scale().to_bits(), 1.0_f32.to_bits());
                    } else {
                        if print && (route == 1 || route == 3) || !print && route == 2 {
                            assert!(demand.unwrap());
                        } else {
                            assert_eq!(reason(demand.unwrap_err()), if print { 7 } else { 2 });
                        }
                        assert_eq!(reason(result.unwrap_err()), if print { 7 } else { 2 });
                    }
                }
            }
        }
    }

    #[test]
    fn scanner_lookup_retains_first_bracket_and_present_nan() {
        let x = [0.0, 2.0, 2.0, 4.0];
        let y = [0.0, 4.0, 8.0, 12.0];
        assert_eq!(
            interpolate(2.0, x.len(), y.len(), |i| x[i], |i| y[i]),
            Some(4.0)
        );
        let x = [f32::NEG_INFINITY, 0.0, f32::INFINITY];
        let y = [1.0, 2.0, 3.0];
        assert!(
            interpolate(-1.0, x.len(), y.len(), |i| x[i], |i| y[i])
                .unwrap()
                .is_nan()
        );
        assert_eq!(finite_mean(&[f32::NAN, f32::INFINITY]), 0.0);
    }
}
