//! Frozen Portra bank and exact-footprint tables. Adapted from Fotufilm's
//! FilmGrainTiles.swift, revision 87a3f46af093a9e6bf76acf21aca83d3b9e2c600.
use super::{AssetError, Assets};
use crate::film::density::portra::{
    CALIBRATED_MAX_FOOTPRINT, MAX_FOOTPRINT, MIN_FOOTPRINT, SINGLE_WRAP_MAX_FOOTPRINT,
};
use crate::hash;
use serde::Deserialize;
use std::{fmt, fs, path::Path, sync::Arc};

const LEVELS: usize = 17;
const STRIDE: usize = 257;
const TILE: usize = STRIDE * STRIDE;
const RECORD_WORDS: usize = 3 + LEVELS;
const RECORD_HASH: u64 = 0x0f35_2397_bf64_2fdb;
const SUMS_HASH: u64 = 0x76ab_1ed7_b762_2233;
const TABLE_VERSION: u64 = 1;

#[derive(Debug)]
pub enum PortraGrainError {
    Io(std::io::Error),
    Manifest(serde_json::Error),
    Invalid(&'static str),
    CachePoisoned,
}
impl fmt::Display for PortraGrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "Portra grain resource: {e}"),
            Self::Manifest(e) => write!(f, "Portra grain manifest: {e}"),
            Self::Invalid(reason) => write!(f, "invalid Portra grain: {reason}"),
            Self::CachePoisoned => write!(f, "Portra grain table cache poisoned"),
        }
    }
}
impl std::error::Error for PortraGrainError {}

#[derive(Debug, Clone, Copy)]
pub struct PortraGrainRecord {
    pub minimum_gross: f32,
    pub maximum_gross: f32,
    pub offset: f32,
    pub mean_light: [f32; LEVELS],
}
#[derive(Debug)]
pub struct PortraGrainBank {
    records: [PortraGrainRecord; 3],
    sums: Vec<f32>,
    identity: u64,
}
impl PortraGrainBank {
    pub fn records(&self) -> &[PortraGrainRecord; 3] {
        &self.records
    }
    pub fn sums(&self) -> &[f32] {
        &self.sums
    }
    pub fn identity(&self) -> u64 {
        self.identity
    }
}
#[derive(Debug)]
pub struct PortraGrainTable {
    pub footprint: f32,
    pub mean_density: [[f32; LEVELS]; 3],
    pub correlation: [[f32; LEVELS]; 3],
    bank_identity: u64,
    version: u64,
}
#[derive(Deserialize)]
struct Manifest {
    version: u32,
    stock: String,
    records: String,
    sums: String,
    records_sha256: String,
    sums_sha256: String,
}
fn words(path: &Path, count: usize) -> Result<Vec<f32>, PortraGrainError> {
    let bytes = fs::read(path).map_err(PortraGrainError::Io)?;
    if bytes.len() != count * 4 {
        return Err(PortraGrainError::Invalid("word count"));
    }
    let values: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(f32::from_le_bytes)
        .collect();
    if values.iter().any(|v| !v.is_finite()) {
        return Err(PortraGrainError::Invalid("nonfinite word"));
    }
    Ok(values)
}
fn load_bank(root: &Path) -> Result<PortraGrainBank, PortraGrainError> {
    let dir = root.join("grain/portra-400");
    let manifest: Manifest =
        serde_json::from_slice(&fs::read(dir.join("manifest.json")).map_err(PortraGrainError::Io)?)
            .map_err(PortraGrainError::Manifest)?;
    if manifest.version != 1
        || manifest.stock != "kodak_portra_400"
        || manifest.records != "records.f32"
        || manifest.sums != "sums.f32"
        || manifest.records_sha256
            != "4b2f7cc90bec83f5a63510beca5c224d78bd35ca79c64a785f8eaa7335bd3cdb"
        || manifest.sums_sha256
            != "b5dc55ffca1a7165dfe1316f269ffb9e270aad54575d3a675fa63825b628d1d5"
    {
        return Err(PortraGrainError::Invalid("manifest association or version"));
    }
    let raw = words(&dir.join("records.f32"), 3 * RECORD_WORDS)?;
    let sums = words(&dir.join("sums.f32"), 3 * LEVELS * TILE)?;
    let rh = hash::resource_f32s(&raw);
    let sh = hash::resource_f32s(&sums);
    if rh != RECORD_HASH || sh != SUMS_HASH {
        return Err(PortraGrainError::Invalid("bank identity"));
    }
    for tile in sums.as_chunks::<TILE>().0 {
        if tile[..STRIDE].iter().any(|&v| v != 0.0) || (0..STRIDE).any(|y| tile[y * STRIDE] != 0.0)
        {
            return Err(PortraGrainError::Invalid("SAT border"));
        }
    }
    let records = std::array::from_fn(|r| {
        let start = r * RECORD_WORDS;
        PortraGrainRecord {
            minimum_gross: raw[start],
            maximum_gross: raw[start + 1],
            offset: raw[start + 2],
            mean_light: std::array::from_fn(|k| raw[start + 3 + k]),
        }
    });
    if records
        .iter()
        .any(|r| r.maximum_gross - r.minimum_gross < 1e-6)
    {
        return Err(PortraGrainError::Invalid("density range"));
    }
    let identity = hash::u64s(&[rh, sh]);
    Ok(PortraGrainBank {
        records,
        sums,
        identity,
    })
}
impl Assets {
    pub fn portra_grain_bank(&self) -> Result<Arc<PortraGrainBank>, AssetError> {
        self.portra_grain
            .get_or_init(|| {
                load_bank(&self.resource_dir)
                    .map(Arc::new)
                    .map_err(Arc::new)
            })
            .as_ref()
            .map(Arc::clone)
            .map_err(|e| AssetError::PortraGrain(Arc::clone(e)))
    }
    pub fn portra_grain_table(
        &self,
        bank: &Arc<PortraGrainBank>,
        footprint: f32,
    ) -> Result<Arc<PortraGrainTable>, AssetError> {
        let fail = |e| AssetError::PortraGrain(Arc::new(e));
        if !(MIN_FOOTPRINT..=MAX_FOOTPRINT).contains(&footprint) {
            return Err(fail(PortraGrainError::Invalid("unsupported footprint")));
        }
        let request = {
            let cache = self
                .portra_table
                .lock()
                .map_err(|_| fail(PortraGrainError::CachePoisoned))?;
            if let Some(t) = cache.as_ref().filter(|t| {
                t.bank_identity == bank.identity
                    && t.footprint.to_bits() == footprint.to_bits()
                    && t.version == TABLE_VERSION
            }) {
                return Ok(Arc::clone(t));
            }
            self.portra_table_request
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                .wrapping_add(1)
        };
        let result = Arc::new(measure(bank, footprint).map_err(fail)?);
        let mut cache = self
            .portra_table
            .lock()
            .map_err(|_| fail(PortraGrainError::CachePoisoned))?;
        if let Some(t) = cache.as_ref().filter(|t| {
            t.bank_identity == bank.identity
                && t.footprint.to_bits() == footprint.to_bits()
                && t.version == TABLE_VERSION
        }) {
            return Ok(Arc::clone(t));
        }
        if self
            .portra_table_request
            .load(std::sync::atomic::Ordering::Relaxed)
            == request
        {
            *cache = Some(Arc::clone(&result));
        }
        Ok(result)
    }
}
fn pcg(input: u32) -> u32 {
    let state = input.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}
fn sum_at(s: &[f32], x: f64, y: f64) -> f64 {
    let ix = (x as usize).min(255);
    let iy = (y as usize).min(255);
    let ax = x - ix as f64;
    let ay = y - iy as f64;
    let a = f64::from(s[iy * STRIDE + ix]);
    let b = f64::from(s[iy * STRIDE + ix + 1]);
    let c = f64::from(s[(iy + 1) * STRIDE + ix]);
    let d = f64::from(s[(iy + 1) * STRIDE + ix + 1]);
    a + (b - a) * ax + (c - a) * ay + (d - b - c + a) * ax * ay
}
fn wrapped(s: &[f32], x: f64, y: f64) -> f64 {
    let px = if x >= 256.0 { x - 256.0 } else { x };
    let py = if y >= 256.0 { y - 256.0 } else { y };
    let mut sum = sum_at(s, px, py);
    if x >= 256.0 {
        sum += sum_at(s, 256.0, py);
    }
    if y >= 256.0 {
        sum += sum_at(s, px, 256.0);
    }
    if x >= 256.0 && y >= 256.0 {
        sum += sum_at(s, 256.0, 256.0);
    }
    sum
}
fn box_sum(s: &[f32], x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    wrapped(s, x1, y1) - wrapped(s, x0, y1) - wrapped(s, x1, y0) + wrapped(s, x0, y0)
}

fn periodic_prefix(s: &[f32], x: f64, y: f64) -> f64 {
    let nx = (x / 256.0).floor();
    let ny = (y / 256.0).floor();
    let tx = x - nx * 256.0;
    let ty = y - ny * 256.0;
    sum_at(s, tx, ty)
        + nx * sum_at(s, 256.0, ty)
        + ny * sum_at(s, tx, 256.0)
        + nx * ny * f64::from(s[TILE - 1])
}

fn measure(bank: &PortraGrainBank, footprint: f32) -> Result<PortraGrainTable, PortraGrainError> {
    let pitch = f64::from(footprint);
    let mut table = PortraGrainTable {
        footprint,
        mean_density: [[0.0; LEVELS]; 3],
        correlation: [[1.0; LEVELS]; 3],
        bank_identity: bank.identity,
        version: TABLE_VERSION,
    };
    for r in 0..3 {
        let mut fields = vec![vec![0.0_f64; 128 * 128]; LEVELS];
        let positions = (0..128 * 128).map(|i| {
            let x = ((i % 128 * 7919) % 4096) as f64 * pitch;
            let y = ((i / 128 * 104729) % 4096) as f64 * pitch;
            (x, y)
        });
        for k in (0..LEVELS).step_by(2) {
            let next = (k + 1).min(LEVELS - 1);
            for (i, (x, y)) in positions.clone().enumerate() {
                let bx = ((x + pitch / 2.0) / 128.0).floor() as u32;
                let by = ((y + pitch / 2.0) / 128.0).floor() as u32;
                let h =
                    pcg(bx
                        ^ pcg(
                            by ^ pcg(0x5a3f_1e1d ^ (200_u32 + r as u32).wrapping_mul(0x9e37_79b9))
                        ));
                let left = f64::from(bx) * 128.0;
                let top = f64::from(by) * 128.0;
                // Each footprint uses the same calibration positions and
                // centre-selected placement. Average its complete square.
                let (u0, u1, v0, v1) = (x - left, (x + pitch) - left, y - top, (y + pitch) - top);
                let area = ((u1 - u0) * (v1 - v0)).max(1e-6);
                let (mut p0, mut p1, mut q0, mut q1) = (u0, u1, v0, v1);
                let turn = (h >> 16) & 7;
                if turn & 1 != 0 {
                    (p0, p1) = (128.0 - u1, 128.0 - u0);
                }
                if turn & 2 != 0 {
                    (q0, q1) = (128.0 - v1, 128.0 - v0);
                }
                if turn & 4 != 0 {
                    (p0, p1, q0, q1) = (q0, q1, p0, p1);
                }
                let mut sx = 64.0 + f64::from(h % 256);
                let mut sy = 64.0 + f64::from((h >> 8) % 256);
                if footprint > CALIBRATED_MAX_FOOTPRINT {
                    // Move the lower corner into the first period.
                    sx -= (p0 + sx).div_euclid(256.0) * 256.0;
                    sy -= (q0 + sy).div_euclid(256.0) * 256.0;
                }
                for level in [k, next] {
                    let start = (r * LEVELS + level) * TILE;
                    let sums = &bank.sums[start..start + TILE];
                    let mean = bank.records[r].mean_light[level];
                    let light = if footprint > SINGLE_WRAP_MAX_FOOTPRINT {
                        let (x0, y0) = (p0 + sx, q0 + sy);
                        let (x1, y1) = (x0 + pitch, y0 + pitch);
                        let sum = periodic_prefix(sums, x1, y1)
                            - periodic_prefix(sums, x0, y1)
                            - periodic_prefix(sums, x1, y0)
                            + periodic_prefix(sums, x0, y0);
                        (f64::from(mean) + sum / (pitch * pitch)) as f32
                    } else {
                        let sum = box_sum(sums, p0 + sx, p1 + sx, q0 + sy, q1 + sy);
                        mean + (sum / area) as f32
                    };
                    fields[level][i] = -f64::from(light).max(1e-9).log10();
                }
            }
        }
        let means: Vec<f64> = fields
            .iter()
            .map(|v| {
                if footprint > SINGLE_WRAP_MAX_FOOTPRINT {
                    // Exact periods can give constant light. Center before
                    // summation so a constant field has exactly zero variance.
                    let first = v[0];
                    first + v.iter().map(|sample| sample - first).sum::<f64>() / v.len() as f64
                } else {
                    v.iter().sum::<f64>() / v.len() as f64
                }
            })
            .collect();
        for (target, mean) in table.mean_density[r].iter_mut().zip(&means) {
            *target = *mean as f32;
        }
        for k in 0..LEVELS - 1 {
            let (mut aa, mut bb, mut ab) = (0.0, 0.0, 0.0);
            for (&a, &b) in fields[k].iter().zip(&fields[k + 1]) {
                let a = a - means[k];
                let b = b - means[k + 1];
                aa += a * a;
                bb += b * b;
                ab += a * b;
            }
            table.correlation[r][k] = if aa > 0.0 && bb > 0.0 {
                (ab / (aa * bb).sqrt()).clamp(-1.0, 1.0) as f32
            } else {
                1.0
            };
        }
        if table.mean_density[r]
            .iter()
            .chain(&table.correlation[r])
            .any(|v| !v.is_finite())
            || table.correlation[r].iter().any(|&v| v <= -0.9997)
        {
            return Err(PortraGrainError::Invalid("computed normalization"));
        }
    }
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn frozen_tables_and_cache() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../Resources");
        let assets = Assets::new(root);
        let bank = assets.portra_grain_bank().expect("packaged Portra bank");
        let fixtures: [(f32, &[u8]); 5] = [
            (
                0.25,
                include_bytes!("../../../../tests/grain/portra-table-fixtures/footprint-0.25.f32"),
            ),
            (
                9.375,
                include_bytes!("../../../../tests/grain/portra-table-fixtures/footprint-9.375.f32"),
            ),
            (
                12.5,
                include_bytes!("../../../../tests/grain/portra-table-fixtures/footprint-12.5.f32"),
            ),
            (
                18.75,
                include_bytes!("../../../../tests/grain/portra-table-fixtures/footprint-18.75.f32"),
            ),
            (
                37.5,
                include_bytes!("../../../../tests/grain/portra-table-fixtures/footprint-37.5.f32"),
            ),
        ];
        for (footprint, bytes) in fixtures {
            let table = assets
                .portra_grain_table(&bank, footprint)
                .expect("footprint");
            let expected: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .copied()
                .map(f32::from_le_bytes)
                .collect();
            for r in 0..3 {
                for k in 0..LEVELS {
                    assert!(
                        (table.mean_density[r][k] - expected[r * 34 + k]).abs() <= 2e-6,
                        "mean {footprint} {r} {k}"
                    );
                    assert!(
                        (table.correlation[r][k] - expected[r * 34 + 17 + k]).abs() <= 2e-6,
                        "correlation {footprint} {r} {k}: {} != {}",
                        table.correlation[r][k],
                        expected[r * 34 + 17 + k]
                    );
                }
            }
            assert!(Arc::ptr_eq(
                &table,
                &assets.portra_grain_table(&bank, footprint).unwrap()
            ));
        }
        assert!(assets.portra_grain_table(&bank, f32::NAN).is_err());
        assert!(
            assets
                .portra_grain_table(&bank, MAX_FOOTPRINT.next_up())
                .is_err()
        );
    }
}
