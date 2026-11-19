//! Film v1 allowance for the version-1 frozen Portra summed-area bank.

use crate::assets::{PortraGrainBank, PortraGrainTable};

#[cfg(feature = "grain-preview")]
pub(crate) const MIN_FOOTPRINT: f32 = 0.25;
// The 8 mm, Size 4 µm proxy of an 8192-pixel Full raster uses this cell pitch.
// Internal samples use the subtexel convex-light certificate below.
pub(crate) const MIN_PROXY_SAMPLE_FOOTPRINT: f32 = 8_000.0 / (8_192.0 * 4.0);
pub(crate) const CALIBRATED_MAX_FOOTPRINT: f32 = 37.5;
pub(crate) const SINGLE_WRAP_MAX_FOOTPRINT: f32 = 255.0;
// 120 mm at Size 0.5 µm on a one-pixel rendered long edge.
pub(crate) const MAX_FOOTPRINT: f32 = 240_000.0;

// Outward-rounded extrema of footprint times the single-placement OD delta
// from -log10(mean light), paired with each band's upper footprint.
// continuous_footprint_certificate recomputes these enclosures from the bank.
const SCALED_DELTAS: [(f32, [f64; 2]); 10] = [
    (2.0, [-0.85, 1.44]),
    (4.0, [-1.62, 2.45]),
    (6.0, [-2.25, 3.04]),
    (8.0, [-2.66, 3.29]),
    (10.0, [-2.84, 3.45]),
    (12.0, [-2.85, 3.47]),
    (16.0, [-3.08, 3.62]),
    (20.0, [-3.34, 3.86]),
    (28.0, [-3.57, 4.14]),
    (CALIBRATED_MAX_FOOTPRINT, [-4.03, 4.20]),
];
const MAX_SCALED_MEAN_SHIFT: f64 = 0.16;
const MIN_CORRELATION: f32 = 0.25;
// A sub-texel sample is a convex average of the represented texel lights.
// The bank certificate below verifies the extrema with the sampler's rounding.
const SUBTEXEL_DELTAS: [f64; 2] = [-0.424, 0.752];

pub(crate) fn allowances(
    bank: &PortraGrainBank,
    table: &PortraGrainTable,
    gain: f32,
) -> Option<[f32; 2]> {
    let width = f64::from(table.footprint);
    if !(MIN_PROXY_SAMPLE_FOOTPRINT..=MAX_FOOTPRINT).contains(&table.footprint) {
        return None;
    }
    // Subdivide the placed rectangle into equal pieces with widths inside the
    // certificate. The full transmission average is a convex combination of
    // their averages, so it obeys the same bounds after the monotone -log10.
    // Local box construction keeps the side error below the certificate's
    // 1/64 texel allowance. For multiple periods, the Float64 prefix counts
    // their contributions and the residuals within that same error budget.
    let divisions = (width / f64::from(CALIBRATED_MAX_FOOTPRINT)).ceil();
    let certified_width = width / divisions;
    let [lower, upper] = if width < 1.0 {
        SUBTEXEL_DELTAS.map(|delta| delta * width)
    } else {
        SCALED_DELTAS
            .iter()
            .find(|&&(high, _)| certified_width <= f64::from(high))?
            .1
    };
    let calibrated = width >= 1.0 && divisions == 1.0;
    let mut mean_shift = [0.0_f64; 2];
    let mut min_correlation = 1.0_f32;
    for (r, record) in bank.records().iter().enumerate() {
        for (&light, mean) in record.mean_light.iter().zip(table.mean_density[r]) {
            let shift = width * (f64::from(mean) + f64::from(light).log10());
            if calibrated && !(-MAX_SCALED_MEAN_SHIFT..=MAX_SCALED_MEAN_SHIFT).contains(&shift) {
                return None;
            }
            mean_shift[0] = mean_shift[0].min(shift);
            mean_shift[1] = mean_shift[1].max(shift);
        }
        for &correlation in &table.correlation[r][..16] {
            if calibrated && correlation < MIN_CORRELATION {
                return None;
            }
            min_correlation = min_correlation.min(correlation);
        }
    }

    // In each dimension only one of the two half-block-shifted layings can
    // straddle a seam. The sum of their normalized weight gains is at most
    // max(sqrt(4) + 1, sqrt(2) + sqrt(2)) = 3. The adjacent-level denominator
    // is at least sqrt(1 + rho), and fractional time adds at most sqrt(2).
    // Round the seam gain up to include Float32 arithmetic.
    let mut level_norm_squared = 1.0 + f64::from(min_correlation);
    if !calibrated {
        // Bound Float32 evaluation of kept * kLayings at the measured rho,
        // including negative correlations permitted by table construction
        // for widths outside the calibrated interval.
        level_norm_squared -= 8.0 * f64::from(f32::EPSILON);
    }
    let scale = f64::from(gain) / width * 3.012 * (2.0 / level_norm_squared).sqrt();
    // Reduced chroma mixes transmission ratios with nonnegative weights.
    // Its exact OD delta remains between the pre-projection global extrema.
    // The additional 0.002 D covers exp/log projection and adding the delta.
    Some([
        (((divisions * lower - mean_shift[1]) * scale - 0.002) as f32).next_down(),
        (((divisions * upper - mean_shift[0]) * scale + 0.002) as f32).next_up(),
    ])
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
            // The sub-texel sampler differences stored SAT words and
            // blends normalized overlaps in Float64, then rounds light
            // once to Float32 before the same logarithm.
            let subtexel_error = 32.0 * f64::EPSILON * sat_max
                + 0.5 * f64::from(f32::EPSILON) * (mean + max_deviation);
            let darkest = mean + texels[0] - subtexel_error;
            let lightest = mean + texels[1] + subtexel_error;
            assert!(darkest > 1e-9 && lightest.is_finite());
            assert!(-(lightest / mean).log10() - 0.00005 >= SUBTEXEL_DELTAS[0]);
            assert!(-(darkest / mean).log10() + 0.00005 <= SUBTEXEL_DELTAS[1]);
            for quarter in 4..150 {
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
                let &(_, [certified_lower, certified_upper]) = SCALED_DELTAS
                    .iter()
                    .find(|&&(end, _)| high <= f64::from(end))
                    .unwrap();
                assert!(
                    lower >= certified_lower && upper <= certified_upper,
                    "footprint {low} to {high}, mean {mean}: {lower} to {upper}"
                );
            }
        }
    }

    #[test]
    fn computed_normalizers_cover_preview_footprints() {
        let assets = assets();
        let bank = assets.portra_grain_bank().unwrap();
        let geometries = [302, 604, 1208, 1920, 2416, 3840, 4832, 7418];
        let sizes = [0.5, 1.0, 4.0];
        let widths = geometries.into_iter().flat_map(|long_edge| {
            sizes.map(|size| (36_000.0 / f64::from(long_edge) / size) as f32)
        });
        let film_formats = [
            8.0, 11.66, 19.125, 21.95, 24.89, 25.0, 35.0, 37.72, 48.5, 56.0, 70.0, 70.4, 120.0,
        ]
        .into_iter()
        .flat_map(|mm| sizes.map(|size| (f64::from(mm as f32) * 1000.0 / 4832.0 / size) as f32));
        for width in (1..=150)
            .map(|quarter| quarter as f32 / 4.0)
            .chain([
                MIN_PROXY_SAMPLE_FOOTPRINT,
                1.0_f32.next_down(),
                1.0_f32.next_up(),
                CALIBRATED_MAX_FOOTPRINT.next_up(),
                128.0,
                SINGLE_WRAP_MAX_FOOTPRINT,
                SINGLE_WRAP_MAX_FOOTPRINT.next_up(),
                256.0,
                512.0,
                (36_000.0 / 92.0) as f32,
                (240_000.0 / 92.0) as f32,
                MAX_FOOTPRINT,
            ])
            .chain(widths)
            .chain(film_formats)
        {
            let table = assets.portra_grain_table(&bank, width).unwrap();
            let [lower, upper] =
                allowances(&bank, &table, 16.0).unwrap_or_else(|| panic!("footprint {width}"));
            assert!(lower.is_finite() && upper.is_finite() && lower < 0.0 && upper > 0.0);
        }
    }
}
