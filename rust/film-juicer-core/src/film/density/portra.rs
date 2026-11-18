//! Film v1 allowance for the version-1 frozen Portra summed-area bank.

use crate::assets::{PortraGrainBank, PortraGrainTable};

pub(crate) const FILM_LONG_EDGE_UM: f64 = 36_000.0;
pub(crate) const MAX_RENDER_AXIS: i32 = 4097;

// Outward-rounded extrema of footprint times the single-placement OD delta
// from -log10(mean light). The bank certificate covers every footprint in
// [2, 37.5], enclosing the production minimum 36000 / (4097 * 4).
const MIN_SCALED_DELTA: f64 = -4.03;
const MAX_SCALED_DELTA: f64 = 4.20;
const MAX_SCALED_MEAN_SHIFT: f64 = 0.16;
const MIN_CORRELATION: f32 = 0.25;

/// Check the computed normalizers used in the enclosure at the exact width
/// rendered. The means and correlations are published together before this
/// check and remain immutable through projection.
pub(crate) fn verify_table(bank: &PortraGrainBank, table: &PortraGrainTable) -> bool {
    (2.0..=37.5).contains(&table.footprint)
        && bank.records().iter().enumerate().all(|(r, record)| {
            record
                .mean_light
                .iter()
                .zip(table.mean_density[r])
                .all(|(&light, mean)| {
                    (f64::from(table.footprint) * (f64::from(mean) + f64::from(light).log10()))
                        .abs()
                        <= MAX_SCALED_MEAN_SHIFT
                })
                && table.correlation[r][..16]
                    .iter()
                    .all(|&correlation| (MIN_CORRELATION..=1.0).contains(&correlation))
        })
}

pub(crate) fn allowances(gain: f32, size_um: f32) -> [f32; 2] {
    // In each dimension only one of the two half-block-shifted layings can
    // straddle a seam. The sum of their normalized weight gains is at most
    // max(sqrt(4) + 1, sqrt(2) + sqrt(2)) = 3. The adjacent-level denominator
    // is at least sqrt(1 + rho), and fractional time adds at most sqrt(2).
    // Round 3 * sqrt(2 / 1.25) up to 3.81 to include Float32 arithmetic.
    // gain / footprint = gain * size / physical pixel width.
    let scale = f64::from(gain) * f64::from(size_um) * f64::from(MAX_RENDER_AXIS)
        / FILM_LONG_EDGE_UM
        * 3.81;
    // Reduced chroma mixes transmission ratios with nonnegative weights.
    // Its exact OD delta remains between the pre-projection global extrema.
    // The additional 0.002 D covers the finite Float32 exp/log projection;
    // the worst certified CMY spread is less than 33 D.
    [
        (((MIN_SCALED_DELTA - MAX_SCALED_MEAN_SHIFT) * scale - 0.002) as f32).next_down(),
        (((MAX_SCALED_DELTA + MAX_SCALED_MEAN_SHIFT) * scale + 0.002) as f32).next_up(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::Assets;
    use std::path::Path;

    const TILE: usize = 256;
    const STRIDE: usize = TILE + 1;
    const PADDED_STRIDE: usize = STRIDE + 40;

    fn assets() -> Assets {
        Assets::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Resources"))
    }

    fn periodic_prefix(tile: &[f32]) -> (Vec<f64>, [f64; 2]) {
        let mut prefix = vec![0.0; PADDED_STRIDE * PADDED_STRIDE];
        let mut limits = [f64::INFINITY, f64::NEG_INFINITY];
        for row in 0..PADDED_STRIDE - 1 {
            let y = (row + TILE - 1) % TILE;
            let mut running = 0.0;
            for column in 0..PADDED_STRIDE - 1 {
                let x = (column + TILE - 1) % TILE;
                let top = y * STRIDE + x;
                let bottom = top + STRIDE;
                // Decode the represented texel deviation from the four
                // stored Float32 SAT words, with Float64 accumulation.
                let delta = f64::from(tile[bottom + 1])
                    - f64::from(tile[bottom])
                    - f64::from(tile[top + 1])
                    + f64::from(tile[top]);
                limits[0] = limits[0].min(delta);
                limits[1] = limits[1].max(delta);
                running += delta;
                prefix[(row + 1) * PADDED_STRIDE + column + 1] =
                    prefix[row * PADDED_STRIDE + column + 1] + running;
            }
        }
        (prefix, limits)
    }

    fn rectangle(prefix: &[f64], x: usize, y: usize, w: usize, h: usize) -> f64 {
        let first = y * PADDED_STRIDE + x;
        let last = first + h * PADDED_STRIDE;
        prefix[last + w] - prefix[last] - prefix[first + w] + prefix[first]
    }

    fn square_extrema(prefix: &[f64], width: f64) -> [f64; 2] {
        // A translated rectangle integral is bilinear on each cell between
        // crossings of either edge with an integer coordinate. Enumerate all
        // cell corners as an integral square plus the fractional strips on
        // either side of each axis, including their shared corner.
        let side = width.floor() as usize;
        let fraction = width - side as f64;
        let mut limits = [f64::INFINITY, f64::NEG_INFINITY];
        for y in 1..=TILE {
            for x in 1..=TILE {
                let center = rectangle(prefix, x, y, side, side);
                let vertical = [
                    rectangle(prefix, x - 1, y, 1, side),
                    rectangle(prefix, x + side, y, 1, side),
                ];
                let horizontal = [
                    rectangle(prefix, x, y - 1, side, 1),
                    rectangle(prefix, x, y + side, side, 1),
                ];
                for (x, horizontal_offset) in [x - 1, x + side].into_iter().zip(vertical) {
                    for (y, vertical_offset) in [y - 1, y + side].into_iter().zip(horizontal) {
                        let corner = rectangle(prefix, x, y, 1, 1);
                        let sum = center
                            + fraction * (horizontal_offset + vertical_offset)
                            + fraction * fraction * corner;
                        limits[0] = limits[0].min(sum);
                        limits[1] = limits[1].max(sum);
                    }
                }
            }
        }
        limits.map(|value| value / width.powi(2))
    }

    #[test]
    fn continuous_footprint_certificate() {
        let bank = assets().portra_grain_bank().unwrap();
        let levels = bank.records().iter().flat_map(|record| record.mean_light);
        let mut extrema = [0.0_f64; 2];
        for (tile, mean_light) in bank
            .sums()
            .as_chunks::<{ STRIDE * STRIDE }>()
            .0
            .iter()
            .zip(levels)
        {
            let (prefix, texels) = periodic_prefix(tile);
            let sat_max = tile
                .iter()
                .map(|value| f64::from(value.abs()))
                .fold(0.0, f64::max);
            let max_deviation = texels[0].abs().max(texels[1].abs());
            let mean = f64::from(mean_light);
            for quarter in 8..150 {
                let low = f64::from(quarter) / 4.0;
                let high = low + 0.25;
                // Float32 right-edge construction can move a side length
                // by an ulp even at the most distant guarded placement.
                let a = low - 1.0 / 64.0;
                let b = high + 1.0 / 64.0;
                let corners = square_extrema(&prefix, a);
                let covered = (a / b).powi(2);
                // Any rectangle with sides in [a, b] contains the a-square
                // at its lower corner. Bound the remaining average with the
                // decoded texel extrema. 640 * u * max|SAT| bounds the four
                // Float32 wrapped bilinear corner sums and their differences.
                // The width term covers the local/placed area discrepancy.
                let error = 640.0 * 2.0_f64.powi(-24) * sat_max / a.powi(2)
                    + 0.0005 * max_deviation / a
                    + 3e-7;
                let darkest = mean + covered * corners[0] + (1.0 - covered) * texels[0] - error;
                let lightest = mean + covered * corners[1] + (1.0 - covered) * texels[1] + error;
                assert!(
                    darkest > 1e-9 && lightest.is_finite(),
                    "{low} {darkest} {lightest}"
                );
                let lower = (-((lightest / mean).log10()) - 0.00005) * high;
                let upper = (-((darkest / mean).log10()) + 0.00005) * high;
                extrema[0] = extrema[0].min(lower);
                extrema[1] = extrema[1].max(upper);
            }
        }
        assert!(
            extrema[0] >= MIN_SCALED_DELTA && extrema[1] <= MAX_SCALED_DELTA,
            "{extrema:?}"
        );
        let [lower, upper] = allowances(16.0, 0.5);
        assert!(upper - lower + 2.3 < 33.0);
    }

    #[test]
    fn computed_normalizers_cover_preview_footprints() {
        let assets = assets();
        let bank = assets.portra_grain_bank().unwrap();
        for quarter in 8..=150 {
            let width = quarter as f32 / 4.0;
            let table = assets.portra_grain_table(&bank, width).unwrap();
            assert!(verify_table(&bank, &table), "footprint {width}");
        }
    }
}
