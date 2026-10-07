//! Fixed-axis illuminant construction. Samples are relative linear SPD in f32.

use std::fmt;

#[cfg(feature = "test-support")]
thread_local! {
    static SCRATCH_CAPACITIES: std::cell::Cell<([usize; 10], usize)> = const { std::cell::Cell::new(([0; 10], 0)) };
}
#[cfg(feature = "test-support")]
fn observe_scratch(bytes: usize) {
    if bytes == 0 {
        return;
    }
    let (mut capacities, count) = SCRATCH_CAPACITIES.get();
    capacities[count] = bytes;
    SCRATCH_CAPACITIES.set((capacities, count + 1));
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Coverage {
    pub warnings: u32,
    pub min_nm: f32,
    pub max_nm: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    Preparation,
    Capacity,
}

#[derive(Debug)]
pub struct Error {
    pub kind: ErrorKind,
    pub coverage: Coverage,
    detail: &'static str,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.detail)
    }
}
impl std::error::Error for Error {}

fn preparation(detail: &'static str) -> Error {
    Error {
        kind: ErrorKind::Preparation,
        coverage: Coverage::default(),
        detail,
    }
}
fn capacity() -> Error {
    Error {
        kind: ErrorKind::Capacity,
        coverage: Coverage::default(),
        detail: "illuminant scratch capacity",
    }
}
fn with_coverage<T>(result: Result<T, Error>, coverage: Coverage) -> Result<(T, Coverage), Error> {
    result.map(|value| (value, coverage)).map_err(|mut error| {
        error.coverage = coverage;
        error
    })
}

fn wavelength(index: usize) -> f32 {
    380.0 + 5.0 * index as f32
}
fn planck(wavelength_nm: f32, temperature_kelvin: f32) -> f32 {
    let lambda_m = f64::from(wavelength_nm) * 1e-9;
    let c = 2.99792458e8;
    let h = 6.62607015e-34;
    let k = 1.380649e-23;
    let c1 = 2.0 * h * c * c;
    let c2 = h * c / k;
    let denominator = (c2 / (lambda_m * f64::from(temperature_kelvin))).exp() - 1.0;
    let radiance = if denominator > 0.0 {
        c1 / (lambda_m.powf(5.0) * denominator)
    } else {
        0.0
    };
    radiance as f32
}
fn normalize(samples: &mut [f32]) {
    if samples.is_empty() {
        return;
    }
    let mut sum = 0.0_f64;
    for &sample in samples.iter() {
        sum += f64::from(sample);
    }
    let mean = sum / samples.len() as f64;
    if mean > 0.0 {
        for sample in samples {
            *sample = (f64::from(*sample) / mean) as f32;
        }
    }
}
fn on_axis(rows: &[[f32; 2]]) -> bool {
    if rows.len() != 81 {
        return false;
    }
    for (i, row) in rows.iter().enumerate() {
        // Synthetic NaN labels pass; source decoding rejects them separately.
        if (row[0] - wavelength(i)).abs() > 1e-3 {
            return false;
        }
    }
    true
}
fn coverage(rows: &[[f32; 2]]) -> Coverage {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for row in rows {
        if row[0].is_finite() {
            if row[0] < min {
                min = row[0];
            }
            if max < row[0] {
                max = row[0];
            }
        }
    }
    if !min.is_finite() || !max.is_finite() {
        return Coverage::default();
    }
    Coverage {
        warnings: u32::from(min > 370.0) | (u32::from(max < 790.0) << 1),
        min_nm: min,
        max_nm: max,
    }
}

pub fn from_samples(rows: &[[f32; 2]]) -> Result<[f32; 81], Error> {
    if !on_axis(rows) {
        return Err(preparation("illuminant CSV axis mismatch"));
    }
    let mut sum = 0.0_f64;
    for row in rows {
        if !row[1].is_finite() {
            return Err(preparation("illuminant CSV nonfinite sample"));
        }
        sum += f64::from(row[1]);
    }
    let mean = sum / rows.len() as f64;
    if !mean.is_finite() || (mean - 1.0).abs() > 1e-5 {
        return Err(preparation("illuminant CSV mean-power mismatch"));
    }
    Ok(std::array::from_fn(|i| rows[i][1]))
}
pub fn blackbody(temperature_kelvin: f32) -> [f32; 81] {
    let mut samples = std::array::from_fn(|i| planck(wavelength(i), temperature_kelvin));
    normalize(&mut samples);
    samples
}
pub fn equal_energy() -> [f32; 81] {
    [1.0; 81]
}

fn scratch(count: usize) -> Result<Vec<f64>, Error> {
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|_| capacity())?;
    values.resize(count, 0.0);
    #[cfg(feature = "test-support")]
    observe_scratch(values.capacity() * std::mem::size_of::<f64>());
    Ok(values)
}
fn resample(rows: &[[f32; 2]]) -> Result<[f32; 81], Error> {
    #[cfg(feature = "test-support")]
    SCRATCH_CAPACITIES.set(([0; 10], 0));
    if on_axis(rows) {
        return Ok(std::array::from_fn(|i| rows[i][1]));
    }
    let mut points = Vec::new();
    points
        .try_reserve_exact(rows.len())
        .map_err(|_| capacity())?;
    #[cfg(feature = "test-support")]
    observe_scratch(points.capacity() * std::mem::size_of::<(f32, f32, usize)>());
    for (i, row) in rows.iter().enumerate() {
        if row[0].is_finite() && row[1].is_finite() {
            points.push((row[0], row[1], i));
        }
    }
    points.sort_unstable_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .expect("finite wavelength")
            .then(a.2.cmp(&b.2))
    });
    points.dedup_by(|a, b| a.0 == b.0);
    let n = points.len();
    if n < 2 {
        return Err(preparation(
            "illuminant filter resample needs two finite unique points",
        ));
    }
    let mut slopes = scratch(n)?;
    if n == 2 {
        let interval = f64::from(points[1].0) - f64::from(points[0].0);
        let slope = (f64::from(points[1].1) - f64::from(points[0].1)) / interval;
        slopes[0] = slope;
        slopes[1] = slope;
    } else {
        let extended_count = n.checked_add(3).ok_or_else(capacity)?;
        let mut intervals = scratch(n - 1)?;
        for i in 0..n - 1 {
            intervals[i] = f64::from(points[i + 1].0) - f64::from(points[i].0);
        }
        let mut extended = scratch(extended_count)?;
        let mut defaults = scratch(n)?;
        let mut differences = scratch(extended_count - 1)?;
        let mut forward = scratch(n)?;
        let mut backward = scratch(n)?;
        let mut sums = scratch(n)?;
        for i in 0..n - 1 {
            extended[i + 2] = (f64::from(points[i + 1].1) - f64::from(points[i].1)) / intervals[i];
        }
        extended[1] = 2.0 * extended[2] - extended[3];
        extended[0] = 2.0 * extended[1] - extended[2];
        extended[extended_count - 2] =
            2.0 * extended[extended_count - 3] - extended[extended_count - 4];
        extended[extended_count - 1] =
            2.0 * extended[extended_count - 2] - extended[extended_count - 3];
        for i in 0..n {
            defaults[i] = 0.5 * (extended[i] + extended[i + 3]);
        }
        for i in 0..extended_count - 1 {
            differences[i] = (extended[i + 1] - extended[i]).abs();
        }
        let mut maximum = 0.0_f64;
        for i in 0..n {
            forward[i] = differences[i + 2];
            backward[i] = differences[i];
            sums[i] = forward[i] + backward[i];
            if maximum < sums[i] {
                maximum = sums[i];
            }
        }
        let cutoff = 1e-9 * maximum;
        for i in 0..n {
            let mut slope = defaults[i];
            if sums[i] > cutoff {
                let numerator = backward[i] * (extended[i + 2] - extended[i + 1]);
                slope = extended[i + 1] + numerator / sums[i];
            }
            slopes[i] = slope;
        }
    }
    let segments = n - 1;
    let mut coefficients = scratch(segments.checked_mul(4).ok_or_else(capacity)?)?;
    for i in 0..segments {
        let interval = f64::from(points[i + 1].0) - f64::from(points[i].0);
        let inverse = 1.0 / interval;
        let inverse_squared = inverse * inverse;
        let delta = (f64::from(points[i + 1].1) - f64::from(points[i].1)) * inverse;
        coefficients[i * 4] = f64::from(points[i].1);
        coefficients[i * 4 + 1] = slopes[i];
        coefficients[i * 4 + 2] = (3.0 * delta - 2.0 * slopes[i] - slopes[i + 1]) * inverse;
        coefficients[i * 4 + 3] = (slopes[i] + slopes[i + 1] - 2.0 * delta) * inverse_squared;
    }
    Ok(std::array::from_fn(|i| {
        let x = f64::from(wavelength(i));
        if x < f64::from(points[0].0) || x > f64::from(points[n - 1].0) {
            return f32::NAN;
        }
        let upper = points
            .partition_point(|point| f64::from(point.0) <= x)
            .clamp(1, n - 1);
        let segment = upper - 1;
        let offset = x - f64::from(points[segment].0);
        let c = &coefficients[segment * 4..segment * 4 + 4];
        let mut value = c[3];
        value = value * offset + c[2];
        value = value * offset + c[1];
        value = value * offset + c[0];
        value as f32
    }))
}

pub fn tungsten_kg3(rows: &[[f32; 2]]) -> Result<([f32; 81], Coverage), Error> {
    let coverage = coverage(rows);
    let result = (|| {
        if rows.is_empty()
            || coverage.warnings != 0
            || (coverage.min_nm == 0.0 && coverage.max_nm == 0.0)
        {
            return Err(preparation("KG3 filter coverage incomplete"));
        }
        let kg3 = resample(rows)?;
        let mut samples = std::array::from_fn(|i| planck(wavelength(i), 3400.0) * kg3[i]);
        normalize(&mut samples);
        Ok(samples)
    })();
    with_coverage(result, coverage)
}

/// Complete call-local preparation; source rows are never retained.
pub struct LensInput {
    blackbody: [f32; 81],
    kg3: [f32; 81],
}
pub fn prepare_lens(rows: &[[f32; 2]]) -> Result<(LensInput, Coverage), Error> {
    let blackbody = std::array::from_fn(|i| planck(wavelength(i), 3200.0));
    if rows.is_empty() {
        return Err(preparation("KG3 filter empty"));
    }
    let coverage = coverage(rows);
    with_coverage(
        resample(rows).map(|kg3| LensInput { blackbody, kg3 }),
        coverage,
    )
}
pub fn finish_lens(input: LensInput, rows: &[[f32; 2]]) -> Result<([f32; 81], Coverage), Error> {
    if rows.is_empty() {
        return Err(preparation("lens transmission empty"));
    }
    let coverage = coverage(rows);
    let result = resample(rows).map(|lens| {
        let mut samples = std::array::from_fn(|i| input.blackbody[i] * input.kg3[i] * lens[i]);
        normalize(&mut samples);
        samples
    });
    with_coverage(result, coverage)
}

#[cfg(feature = "test-support")]
pub mod test_support {
    pub fn scratch_capacities() -> ([usize; 10], usize) {
        super::SCRATCH_CAPACITIES.get()
    }
    pub fn planck_sample(wavelength_nm: f32, temperature_kelvin: f32) -> f32 {
        super::planck(wavelength_nm, temperature_kelvin)
    }
    pub fn normalize(samples: &mut [f32]) {
        super::normalize(samples);
    }
    pub fn resample(rows: &[[f32; 2]]) -> Result<[f32; 81], super::Error> {
        super::resample(rows)
    }
    pub fn capacity_failure() -> Result<(), super::Error> {
        super::SCRATCH_CAPACITIES.set(([0; 10], 0));
        super::scratch(usize::MAX).map(|_| ())
    }
}
