//! Complete host scanner PCHIP storage; no CUDA or retained input ownership.

use std::fmt;
use std::mem::size_of;

#[derive(Clone, Copy, Debug)]
pub struct LutTables<'a> {
    pub eps_c: &'a [f32],
    pub eps_m: &'a [f32],
    pub eps_y: &'a [f32],
    pub weighted_x: &'a [f32],
    pub weighted_y: &'a [f32],
    pub weighted_z: &'a [f32],
    pub baseline_min: Option<&'a [f32]>,
    pub sample_count: i32,
    pub inv_yn: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct LutInput<'a> {
    pub tables: LutTables<'a>,
    pub range_min_cmy: [f32; 3],
    pub inverse_span_cmy: [f32; 3],
    pub medium: u8,
    pub resolution: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LutError {
    Raw,
    Resolution,
    Size { buffer: u32 },
    Allocation { buffer: u32 },
    Nonfinite { index: u64, channel: u32 },
}
impl fmt::Display for LutError {
    fn fmt(&self, writer: &mut fmt::Formatter<'_>) -> fmt::Result {
        writer.write_str(match self {
            Self::Raw => "invalid scanner LUT arguments",
            Self::Resolution => "canonical scan LUT resolution must be at least two",
            Self::Size { .. } => "scanner LUT dimensions exceed addressable capacity",
            Self::Allocation { .. } => "scanner LUT host allocation failed",
            Self::Nonfinite { .. } => "canonical scan LUT build produced non-finite log2 XYZ",
        })
    }
}

#[derive(Debug)]
pub struct Lut {
    resolution: u32,
    log2_xyz: Vec<f32>,
    slope_c: Vec<f32>,
    slope_m: Vec<f32>,
    slope_y: Vec<f32>,
    cell_min: Vec<f32>,
    cell_max: Vec<f32>,
}

fn dimensions(resolution: u32) -> Result<[usize; 3], LutError> {
    if resolution < 2 {
        return Err(LutError::Resolution);
    }
    let r = resolution as usize;
    let cube = |side: usize, buffer| {
        side.checked_mul(side)
            .and_then(|n| n.checked_mul(side))
            .and_then(|n| n.checked_mul(3))
            .ok_or(LutError::Size { buffer })
    };
    let nodes = cube(r, 1)?;
    let capacity = |count: usize, bytes: usize, buffer| {
        count
            .checked_mul(bytes)
            .filter(|&n| n <= isize::MAX as usize)
            .map(|_| ())
            .ok_or(LutError::Size { buffer })
    };
    capacity(nodes, size_of::<f64>(), 1)?;
    for buffer in 2..=5 {
        capacity(nodes, size_of::<f32>(), buffer)?;
    }
    let cells = cube(r - 1, 6)?;
    for buffer in 6..=7 {
        capacity(cells, size_of::<f32>(), buffer)?;
    }
    capacity(r, size_of::<f64>(), 8)?;
    capacity(r, size_of::<f64>(), 9)?;
    capacity(r - 1, size_of::<f64>(), 10)?;
    Ok([r, nodes, cells])
}

fn reserve<T: Copy + Default>(
    count: usize,
    buffer: u32,
    rejected: u32,
) -> Result<Vec<T>, LutError> {
    let mut values = Vec::new();
    if rejected == buffer {
        return Err(LutError::Allocation { buffer });
    }
    values
        .try_reserve_exact(count)
        .map_err(|_| LutError::Allocation { buffer })?;
    values.resize(count, T::default());
    Ok(values)
}

fn slopes(values: &[f64], out: &mut [f64], deltas: &mut [f64]) {
    let n = values.len();
    for i in 0..n - 1 {
        deltas[i] = values[i + 1] - values[i];
    }
    if n == 2 {
        out.fill(deltas[0]);
        return;
    }
    let endpoint = |a: f64, b: f64| {
        let slope = 0.5 * (3.0 * a - b);
        if slope * a <= 0.0 {
            0.0
        } else if a * b < 0.0 && slope.abs() > (3.0 * a).abs() {
            3.0 * a
        } else {
            slope
        }
    };
    out[0] = endpoint(deltas[0], deltas[1]);
    for i in 1..n - 1 {
        let previous = deltas[i - 1];
        let next = deltas[i];
        out[i] = if previous == 0.0 || next == 0.0 || previous * next <= 0.0 {
            0.0
        } else {
            2.0 * previous * next / (previous + next)
        };
    }
    out[n - 1] = endpoint(deltas[n - 2], deltas[n - 3]);
}

fn log_xyz(input: LutInput<'_>, normalized: [f64; 3]) -> [f64; 3] {
    let tables = input.tables;
    if tables.sample_count <= 0 {
        return [f64::NAN; 3];
    }
    let density = std::array::from_fn::<_, 3, _>(|channel| {
        let scaled = normalized[channel] / f64::from(input.inverse_span_cmy[channel]);
        if input.medium == 0 {
            scaled - f64::from(input.range_min_cmy[channel])
        } else {
            scaled + f64::from(input.range_min_cmy[channel])
        }
    });
    let mut xyz = [0.0; 3];
    for sample in 0..tables.sample_count as usize {
        let baseline = tables.baseline_min.map_or(0.0, |v| f64::from(v[sample]));
        let spectral = density[0] * f64::from(tables.eps_c[sample])
            + density[1] * f64::from(tables.eps_m[sample])
            + density[2] * f64::from(tables.eps_y[sample])
            + baseline;
        let transmittance = 10.0_f64.powf(-spectral);
        for (channel, weights) in [tables.weighted_x, tables.weighted_y, tables.weighted_z]
            .iter()
            .enumerate()
        {
            let weight = f64::from(weights[sample]);
            if weight.is_finite() {
                let product = transmittance * weight;
                if !product.is_nan() {
                    xyz[channel] += product;
                }
            }
        }
    }
    let xyz = xyz.map(|sum| sum * f64::from(tables.inv_yn));
    xyz.map(|sum| {
        // fmax's NaN handling is required; either zero sign is erased by epsilon.
        (sum.max(0.0) + 1e-10).log10()
    })
}

impl Lut {
    /// Raw adapters use this same capacity authority before binding consumed spans.
    pub fn check_capacity(resolution: u32) -> Result<(), LutError> {
        dimensions(resolution).map(|_| ())
    }

    pub fn new(input: LutInput<'_>) -> Result<Self, LutError> {
        #[cfg(feature = "test-support")]
        let rejected = super::test_support::take_lut_reservation();
        #[cfg(not(feature = "test-support"))]
        let rejected = 0;
        let [r, count, cells] = dimensions(input.resolution)?;
        let t = input.tables;
        if t.sample_count > 0 {
            let k = t.sample_count as usize;
            if [
                t.eps_c,
                t.eps_m,
                t.eps_y,
                t.weighted_x,
                t.weighted_y,
                t.weighted_z,
            ]
            .iter()
            .any(|v| v.len() != k)
                || t.baseline_min.is_some_and(|v| v.len() != k)
            {
                return Err(LutError::Raw);
            }
        }
        let mut nodes = reserve::<f64>(count, 1, rejected)?;
        let mut log2_xyz = reserve::<f32>(count, 2, rejected)?;
        let mut slope_c = reserve::<f32>(count, 3, rejected)?;
        let mut slope_m = reserve::<f32>(count, 4, rejected)?;
        let mut slope_y = reserve::<f32>(count, 5, rejected)?;
        let mut cell_min = reserve::<f32>(cells, 6, rejected)?;
        let mut cell_max = reserve::<f32>(cells, 7, rejected)?;
        let index = |c, m, y, channel| (((c * r + m) * r + y) * 3) + channel;
        for c in 0..r {
            for m in 0..r {
                for y in 0..r {
                    let normalized = [c, m, y].map(|i| i as f64 / (r - 1) as f64);
                    for (channel, value) in log_xyz(input, normalized).into_iter().enumerate() {
                        if !value.is_finite() {
                            return Err(LutError::Nonfinite {
                                index: ((c * r + m) * r + y) as u64,
                                channel: channel as u32,
                            });
                        }
                        let i = index(c, m, y, channel);
                        nodes[i] = value * std::f64::consts::LOG2_10;
                        log2_xyz[i] = nodes[i] as f32;
                    }
                }
            }
        }
        let mut line = reserve::<f64>(r, 8, rejected)?;
        let mut line_slopes = reserve::<f64>(r, 9, rejected)?;
        let mut deltas = reserve::<f64>(r - 1, 10, rejected)?;
        for (axis, out) in [&mut slope_c, &mut slope_m, &mut slope_y]
            .into_iter()
            .enumerate()
        {
            for a in 0..r {
                for b in 0..r {
                    for channel in 0..3 {
                        let point = |i| match axis {
                            0 => index(i, a, b, channel),
                            1 => index(a, i, b, channel),
                            _ => index(a, b, i, channel),
                        };
                        for i in 0..r {
                            line[i] = nodes[point(i)];
                        }
                        slopes(&line, &mut line_slopes, &mut deltas);
                        for i in 0..r {
                            out[point(i)] = line_slopes[i] as f32;
                        }
                    }
                }
            }
        }
        let s = r - 1;
        for c in 0..s {
            for m in 0..s {
                for y in 0..s {
                    for channel in 0..3 {
                        let mut minimum = nodes[index(c, m, y, channel)];
                        let mut maximum = minimum;
                        for dc in 0..2 {
                            for dm in 0..2 {
                                for dy in 0..2 {
                                    let value = nodes[index(c + dc, m + dm, y + dy, channel)];
                                    if value < minimum {
                                        minimum = value;
                                    }
                                    if maximum < value {
                                        maximum = value;
                                    }
                                }
                            }
                        }
                        let i = (((c * s + m) * s + y) * 3) + channel;
                        cell_min[i] = minimum as f32;
                        cell_max[i] = maximum as f32;
                    }
                }
            }
        }
        Ok(Self {
            resolution: input.resolution,
            log2_xyz,
            slope_c,
            slope_m,
            slope_y,
            cell_min,
            cell_max,
        })
    }
    pub fn resolution(&self) -> u32 {
        self.resolution
    }
    pub fn log2_xyz(&self) -> &[f32] {
        &self.log2_xyz
    }
    pub fn slope_c(&self) -> &[f32] {
        &self.slope_c
    }
    pub fn slope_m(&self) -> &[f32] {
        &self.slope_m
    }
    pub fn slope_y(&self) -> &[f32] {
        &self.slope_y
    }
    pub fn cell_min(&self) -> &[f32] {
        &self.cell_min
    }
    pub fn cell_max(&self) -> &[f32] {
        &self.cell_max
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn fixture() -> Value {
        let preset = match (cfg!(windows), cfg!(debug_assertions)) {
            (false, true) => "linux-debug",
            (false, false) => "linux-release",
            (true, true) => "windows-clang-debug",
            (true, false) => "windows-clang-release",
        };
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/scanner/lut-fixtures");
        let read = |name: &str| -> Value {
            serde_json::from_slice(&std::fs::read(directory.join(name)).unwrap()).unwrap()
        };
        let manifest = read("manifest.json");
        let logical = manifest["presets"][preset]["native"].as_str().unwrap();
        let mut fixture = read(manifest["payloads"][logical].as_str().unwrap());
        let context = read(manifest["contexts"][logical].as_str().unwrap());
        for (key, value) in context.as_object().unwrap() {
            assert!(
                fixture.get(key).is_none(),
                "shared fixture context overlaps payload"
            );
            fixture[key] = value.clone();
        }
        assert_eq!(fixture["preset"], preset);
        fixture
    }
    fn floats(value: &Value) -> Vec<f32> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|v| f32::from_bits(v.as_u64().unwrap() as u32))
            .collect()
    }
    fn construct(v: &Value) -> Result<Lut, LutError> {
        let spectra = [
            "eps_c",
            "eps_m",
            "eps_y",
            "weighted_x",
            "weighted_y",
            "weighted_z",
            "baseline",
        ]
        .map(|name| floats(&v[name]));
        let minima = floats(&v["range_min_cmy"]);
        let inverse = floats(&v["inverse_span_cmy"]);
        Lut::new(LutInput {
            resolution: v["resolution"].as_u64().unwrap() as u32,
            medium: v["medium"].as_u64().unwrap() as u8,
            range_min_cmy: minima.try_into().unwrap(),
            inverse_span_cmy: inverse.try_into().unwrap(),
            tables: LutTables {
                eps_c: &spectra[0],
                eps_m: &spectra[1],
                eps_y: &spectra[2],
                weighted_x: &spectra[3],
                weighted_y: &spectra[4],
                weighted_z: &spectra[5],
                baseline_min: v["has_baseline"]
                    .as_bool()
                    .unwrap()
                    .then_some(spectra[6].as_slice()),
                sample_count: v["sample_count"].as_i64().unwrap() as i32,
                inv_yn: f32::from_bits(v["inv_yn"].as_u64().unwrap() as u32),
            },
        })
    }
    #[test]
    fn whole_native_lut_words_and_failures() {
        let captured = fixture();
        for row in captured["cases"]
            .as_array()
            .unwrap()
            .iter()
            .chain(captured["products"].as_array().unwrap())
        {
            let result = construct(&row["input"]);
            let expected = &row["native"];
            if expected["success"].as_bool().unwrap() {
                // construct returns after every input allocation has expired.
                let result = result.unwrap();
                assert_eq!(
                    result.resolution(),
                    expected["resolution"].as_u64().unwrap() as u32
                );
                for (name, actual) in [
                    "log2_xyz", "slope_c", "slope_m", "slope_y", "cell_min", "cell_max",
                ]
                .into_iter()
                .zip([
                    result.log2_xyz(),
                    result.slope_c(),
                    result.slope_m(),
                    result.slope_y(),
                    result.cell_min(),
                    result.cell_max(),
                ]) {
                    let words: Vec<_> = actual.iter().map(|v| v.to_bits()).collect();
                    let wanted: Vec<_> = expected[name]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as u32)
                        .collect();
                    assert_eq!(words, wanted, "{} {name}", row["id"]);
                }
            } else {
                let error = result.unwrap_err();
                assert_eq!(error.to_string(), expected["diagnostic"].as_str().unwrap());
                match error {
                    LutError::Nonfinite { index, channel } => {
                        assert_eq!(index, expected["failed_voxel"].as_u64().unwrap());
                        assert_eq!(channel, expected["failed_channel"].as_u64().unwrap() as u32);
                    }
                    LutError::Resolution => {
                        assert_eq!(expected["failed_voxel"].as_u64().unwrap(), u64::MAX)
                    }
                    _ => panic!("unexpected native failure {error:?}"),
                }
            }
        }
    }
    #[test]
    fn separate_native_slope_branches() {
        for row in fixture()["slopes"].as_array().unwrap() {
            let input: Vec<_> = row["input"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| f64::from_bits(v.as_u64().unwrap()))
                .collect();
            let mut output = vec![0.0; input.len()];
            let mut deltas = vec![0.0; input.len() - 1];
            slopes(&input, &mut output, &mut deltas);
            let words: Vec<_> = output.iter().map(|v| v.to_bits()).collect();
            let expected: Vec<_> = row["native"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect();
            assert_eq!(words, expected, "{}", row["id"]);
        }
    }
    #[test]
    fn capacity_checks_precede_allocation() {
        assert_eq!(Lut::check_capacity(0), Err(LutError::Resolution));
        assert_eq!(Lut::check_capacity(1), Err(LutError::Resolution));
        assert_eq!(
            Lut::check_capacity(u32::MAX),
            Err(LutError::Size { buffer: 1 })
        );
        assert_eq!(
            dimensions(128).unwrap(),
            [128, 3 * 128usize.pow(3), 3 * 127usize.pow(3)]
        );
    }
}
