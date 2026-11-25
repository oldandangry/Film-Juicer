//! Film v1 allowance for the version-1 frozen Portra summed-area bank.

use crate::assets::{PortraGrainBank, PortraGrainTable};

mod placement_extrema;
mod periodic_extrema;
#[cfg(test)]
mod calibration_interval;
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
const PERIODIC_ALLOWANCE_MIN_FOOTPRINT: f64 = 1536.0;
const PERIODIC_STRIP_LIMIT: f64 = 1476.0;
const PERIODIC_CORNER_LIMIT: f64 = 4720.0;

pub(super) fn density_range(
    bank: &PortraGrainBank,
    table: &PortraGrainTable,
    gain: f32,
    chroma_fraction: f32,
    clean_maximum: [f32; 3],
) -> Option<[[f32; 3]; 2]> {
    let band = if table.footprint < 1.0 {
        0
    } else {
        placement_extrema::BANDS[1..]
            .iter()
            .position(|&(end, _)| table.footprint <= end)?
            + 1
    };
    let (_, extrema) = placement_extrema::BANDS[band];
    let bright_deficits = placement_extrema::BRIGHT_PAIR_DEFICITS[band];
    let clean_light = clean_maximum.map(|density| 10.0_f64.powf(-f64::from(density)));
    let mut minimum = [0.0_f64; 3];
    let mut maximum = clean_maximum.map(f64::from);
    // A placement weight sum of at most 3, rounded up for Float32.
    // The sqrt(2) maximum temporal gain cancels the sqrt(2) from the layings.
    let gain = 3.012 * f64::from(gain);
    for (channel, record) in bank.records().iter().enumerate() {
        let range = f64::from(record.maximum_gross) - f64::from(record.minimum_gross);
        let offset = f64::from(record.minimum_gross) - f64::from(record.offset);
        let level_density = |level: usize| offset + range * (level as f64 / 16.0);
        let shift: [f64; 17] = std::array::from_fn(|level| {
            f64::from(table.mean_density[channel][level])
                + f64::from(record.mean_light[level]).log10()
        });
        for (level, &bright_deficit) in bright_deficits[channel].iter().enumerate() {
            // Enclose rounding of the Float32 level normalization.
            let rho = f64::from(table.correlation[channel][level]) - 8.0 * f64::from(f32::EPSILON);
            let [lower, upper] = extrema[channel];
            let light = |k| shift[k] - f64::from(lower[k]) / 1000.0;
            let dark = |k| f64::from(upper[k]) / 1000.0 - shift[k];
            let a = light(level);
            let b = light(level + 1);
            // The two levels use the same rectangle. Its joint light cap
            // bounds the sum of the two deficits from the separate bright
            // extremes. At least min(t, 1-t) of that loss survives the blend.
            let loss = f64::from(bright_deficit) / 1000.0;
            let brighter = gain
                * adjacent_extreme(a, b - loss, rho, [0.0, 0.5]).max(adjacent_extreme(
                    a - loss,
                    b,
                    rho,
                    [0.5, 1.0],
                ));
            let darker = gain * adjacent_extreme(dark(level), dark(level + 1), rho, [0.0, 1.0]);
            // Saturation includes the tails of the first and last bracket.
            // Widen the interior cuts for Float32 gross and level arithmetic.
            let low = if level == 0 {
                0.0
            } else {
                (level_density(level) - 0.0001).max(0.0)
            };
            let high = if level == 15 {
                f64::from(clean_maximum[channel])
            } else {
                (level_density(level + 1) + 0.0001).min(f64::from(clean_maximum[channel]))
            };
            if low <= high {
                minimum[channel] = minimum[channel].min(low - brighter);
                maximum[channel] = maximum[channel].max(high + darker);
            }
        }
    }

    // Output light is a convex Colour blend of its grained component and
    // the grained luminance times T_i / sum_j(w_j T_j), where T is clean
    // transmission. The ratio increases with T_i and decreases with each
    // other T_j. Use the extreme clean corner for this ratio, independently
    // of the extreme grained luminance.
    const WEIGHTS: [f64; 3] = [0.2126, 0.7152, 0.0722];
    let lightest = minimum.map(|density| 10.0_f64.powf(-density));
    let darkest = maximum.map(|density| 10.0_f64.powf(-density));
    let luminance = |light: [f64; 3]| {
        light
            .into_iter()
            .zip(WEIGHTS)
            .map(|(t, w)| t * w)
            .sum::<f64>()
    };
    let fraction = f64::from(chroma_fraction);
    let output: [[f32; 2]; 3] = std::array::from_fn(|i| {
        let clean_bright = luminance(std::array::from_fn(|j| {
            if i == j { 1.0 } else { clean_light[j] }
        }));
        let clean_dark = luminance(std::array::from_fn(|j| {
            if i == j { clean_light[i] } else { 1.0 }
        }));
        let bright = fraction * lightest[i] + (1.0 - fraction) * luminance(lightest) / clean_bright;
        let dark = fraction * darkest[i]
            + (1.0 - fraction) * luminance(darkest) * clean_light[i] / clean_dark;
        // Cover Float32 density addition, both projection paths and the
        // Float64 positive-weight proxy average and density reconstruction.
        [
            ((-bright.log10() - 0.002) as f32).next_down(),
            ((-dark.log10() + 0.002) as f32).next_up(),
        ]
    });
    Some([output.map(|range| range[0]), output.map(|range| range[1])])
}

fn adjacent_extreme(a: f64, b: f64, rho: f64, [low, high]: [f64; 2]) -> f64 {
    let a = a.max(0.0);
    let b = b.max(0.0);
    // Maximize ((1-t)a + tb) / sqrt((1-t)^2 + t^2 + 2rho t(1-t))
    // for t in [0,1]. The optimizing direction is the inverse of the
    // two-by-two correlation matrix times [a,b] if both entries are
    // positive. Otherwise the maximum is at an endpoint. Restrict that
    // maximum to the fraction interval used for this linear piece.
    let x = a - rho * b;
    let y = b - rho * a;
    let t = if x <= 0.0 {
        1.0
    } else if y <= 0.0 {
        0.0
    } else {
        y / (x + y)
    }
    .clamp(low, high);
    let s = 1.0 - t;
    (s * a + t * b) / (s * s + t * t + 2.0 * rho * s * t).sqrt()
}

pub(super) fn periodic_allowances(table: &PortraGrainTable, gain: f32) -> Option<[f32; 2]> {
    let width = f64::from(table.footprint);
    if table.footprint <= SINGLE_WRAP_MAX_FOOTPRINT {
        return None;
    }
    let scale = |correlation: f32| {
        let norm_squared = 1.0 + f64::from(correlation) - 8.0 * f64::from(f32::EPSILON);
        f64::from(gain) * 3.012 * (2.0 / norm_squared).sqrt()
    };

    let allowance = if width < PERIODIC_ALLOWANCE_MIN_FOOTPRINT {
        let periods = (width / 256.0).round();
        let remainder = width - periods * 256.0;
        let band = ((remainder.abs() / 8.0) as usize).min(periodic_extrema::BANDS.len() - 1);
        let area = width * width;
        let entries = &periodic_extrema::BANDS[band];
        let mut allowance = 0.0_f64;
        for (record, extrema) in table.correlation.iter().zip(entries.as_chunks::<17>().0) {
            let delta = extrema.map(|[strip_lo, strip_hi, corner_lo, corner_hi]| {
                let [lo, hi] = if remainder < 0.0 {
                    [-f64::from(strip_hi), -f64::from(strip_lo)]
                } else {
                    [f64::from(strip_lo), f64::from(strip_hi)]
                };
                let lower = (periods * lo + f64::from(corner_lo)) / area - 1e-7;
                let upper = (periods * hi + f64::from(corner_hi)) / area + 1e-7;
                ((1.0 + upper) / (1.0 + lower)).log10() + 0.0001
            });
            for (level, &correlation) in record[..16].iter().enumerate() {
                allowance = allowance.max(scale(correlation) * delta[level].max(delta[level + 1]));
            }
        }
        allowance
    } else {
        // Each frozen tile has a zero whole-period total. An aperture consists
        // of whole periods, at most width / 256 copies of the residual strips,
        // and one wrapped residual rectangle. These limits are relative to the
        // bank's mean light and apply equally to calibration and rendering.
        let error =
            PERIODIC_STRIP_LIMIT / (256.0 * width) + PERIODIC_CORNER_LIMIT / (width * width);
        // The measured OD mean lies between the signed placement extremes.
        // Charge their full separation, including the light and log roundings.
        let delta = ((1.0 + error) / (1.0 - error)).log10() + 0.0001;
        let correlation = table
            .correlation
            .iter()
            .flat_map(|record| &record[..16])
            .copied()
            .fold(1.0, f32::min);
        delta * scale(correlation)
    };
    let allowance = ((allowance + 0.002) as f32).next_up();
    Some([-allowance, allowance])
}

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
    use crate::{film, route::ScanRoute};
    use super::super::Bounds;
    use std::path::Path;

    const TILE: usize = 256;
    const STRIDE: usize = TILE + 1;
    const PADDED_STRIDE: usize = STRIDE + 40;

    fn assets() -> Assets {
        Assets::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Resources"))
    }

    fn development(assets: &Assets) -> film::Development {
        film::Development::new(
            assets.film("kodak_portra_400").unwrap(),
            film::Params {
                gamma_factor: 1.0,
                layers_required: false,
                requested_route: ScanRoute::NegativeDirect,
            },
        )
        .unwrap()
    }

    fn upper_pad(d: f32) -> f32 {
        // Across a span of at most 64 D the Float32 span and reciprocal
        // roundings can extend the reconstructed upper node by less than this.
        (d + 0.00002).next_up()
    }

    fn check_colour(
        development: &film::Development,
        widths: [f32; 2],
        at: impl Fn(f32) -> [[f32; 3]; 2],
    ) {
        // Each endpoint light is affine in Colour. Its density is monotone
        // in an axis-specific direction. Check the union over each interval
        // to cover mixed-sign spectral corners between the endpoints.
        let partitions = (0..32)
            .map(|n| n as f32 / 32.0)
            .chain((5..=24).map(|n| 1.0 - 2.0_f32.powi(-n)).skip(1))
            .chain([1.0]);
        let mut previous = at(0.0);
        for colour in partitions.skip(1) {
            let current = at(colour);
            let lo = std::array::from_fn(|k| previous[0][k].min(current[0][k]).next_down());
            let hi = std::array::from_fn(|k| upper_pad(previous[1][k].max(current[1][k])));
            Bounds::checked_film_v1(development, lo, hi).unwrap_or_else(|error| {
                panic!("widths {widths:?}, interval ending at Colour {colour}, box {lo:?} to {hi:?}: {error}")
            });
            previous = current;
        }
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

    fn rectangle_extrema(prefix: &[f64], width: f64, height: f64) -> [f64; 2] {
        // A translated rectangle integral is bilinear on each cell between
        // crossings of either edge with an integer coordinate. Enumerate all
        // cell corners as an integral rectangle plus the fractional strips on
        // either side of each axis, including their shared corner.
        let w = width.floor() as usize;
        let h = height.floor() as usize;
        let fx = width - w as f64;
        let fy = height - h as f64;
        let mut limits = [f64::INFINITY, f64::NEG_INFINITY];
        for y in 1..=TILE {
            for x in 1..=TILE {
                let center = rectangle(prefix, x, y, w, h);
                let vertical = [
                    rectangle(prefix, x - 1, y, 1, h),
                    rectangle(prefix, x + w, y, 1, h),
                ];
                let horizontal = [
                    rectangle(prefix, x, y - 1, w, 1),
                    rectangle(prefix, x, y + h, w, 1),
                ];
                for (x, vertical_strip) in [x - 1, x + w].into_iter().zip(vertical) {
                    for (y, horizontal_strip) in [y - 1, y + h].into_iter().zip(horizontal) {
                        let corner = rectangle(prefix, x, y, 1, 1);
                        let sum =
                            center + fx * vertical_strip + fy * horizontal_strip + fx * fy * corner;
                        limits[0] = limits[0].min(sum);
                        limits[1] = limits[1].max(sum);
                    }
                }
            }
        }
        limits.map(|value| value / (width * height))
    }

    fn band_extrema(prefix: &[f64], a: f64, b: f64) -> [f64; 2] {
        // With integral parts fixed, a translation corner's normalized
        // rectangle sum is (C + xU + yV + xyT) / ((w+x)(h+y)).
        // Holding either side fixed gives a fractional-linear function of
        // the other side. Its extrema occur on their interval endpoints,
        // with an additional cut at every integer side length.
        let ends = [a, b]
            .into_iter()
            .chain((a.ceil() as u32..=b.floor() as u32).map(f64::from));
        let mut limits = [f64::INFINITY, f64::NEG_INFINITY];
        for width in ends.clone() {
            for height in ends.clone() {
                let [lo, hi] = rectangle_extrema(prefix, width, height);
                limits = [limits[0].min(lo), limits[1].max(hi)];
            }
        }
        limits
    }

    fn certify_bright_pair(tiles: [&[f32]; 2], means: [f64; 2], record: usize, level: usize) {
        let unpacked = tiles.map(periodic_prefix);
        let sat_max = tiles.map(|tile| tile.iter().map(|v| f64::from(v.abs())).fold(0.0, f64::max));
        let max_deviation = unpacked
            .each_ref()
            .map(|(_, [lo, hi])| lo.abs().max(hi.abs()));
        let mut prefix = vec![0.0; PADDED_STRIDE * PADDED_STRIDE];
        for (band, &(end, extrema)) in placement_extrema::BANDS.iter().enumerate() {
            let loss = placement_extrema::BRIGHT_PAIR_DEFICITS[band][record][level];
            if loss == 0 {
                continue;
            }
            let cap = [0, 1].map(|k| {
                means[k] * 10.0_f64.powf(-f64::from(extrema[record][0][level + k]) / 1000.0)
            });
            for (i, value) in prefix.iter_mut().enumerate() {
                *value = unpacked[0].0[i] / cap[0] + unpacked[1].0[i] / cap[1];
            }
            let (maximum, error) = if band == 0 {
                let maximum = (1..=TILE)
                    .flat_map(|y| (1..=TILE).map(move |x| (x, y)))
                    .map(|(x, y)| rectangle(&prefix, x, y, 1, 1))
                    .fold(f64::NEG_INFINITY, f64::max);
                let error = [0, 1].map(|k| {
                    32.0 * f64::EPSILON * sat_max[k]
                        + 0.5 * f64::from(f32::EPSILON) * (means[k] + max_deviation[k])
                });
                (maximum, error)
            } else {
                let a = f64::from(placement_extrema::BANDS[band - 1].0) - 1.0 / 64.0;
                let b = f64::from(end) + 1.0 / 64.0;
                let error = [0, 1].map(|k| {
                    640.0 * 2.0_f64.powi(-24) * sat_max[k] / a.powi(2)
                        + 0.0005 * max_deviation[k] / a
                        + 3e-7
                });
                (band_extrema(&prefix, a, b)[1], error)
            };
            let sum = maximum + (means[0] + error[0]) / cap[0] + (means[1] + error[1]) / cap[1];
            // The geometric mean of the two light ratios is at most
            // their arithmetic mean, including the signed log error.
            let deficit = -2.0 * (sum / 2.0).log10() - 0.0001;
            assert!(
                deficit >= f64::from(loss) / 1000.0,
                "pair band ending at {end}, record {record}, level {level}: {deficit} below {loss} millidensity"
            );
        }
    }

    #[test]
    fn continuous_footprint_certificate() {
        let bank = assets().portra_grain_bank().unwrap();
        let levels = bank.records().iter().flat_map(|record| record.mean_light);
        for (index, (tile, mean_light)) in bank
            .sums()
            .as_chunks::<{ STRIDE * STRIDE }>()
            .0
            .iter()
            .zip(levels)
            .enumerate()
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
            for (band, &(end, extrema)) in placement_extrema::BANDS.iter().enumerate() {
                let [darkest, lightest] = if band == 0 {
                    [darkest, lightest]
                } else {
                    let a = f64::from(placement_extrema::BANDS[band - 1].0) - 1.0 / 64.0;
                    let b = f64::from(end) + 1.0 / 64.0;
                    let [lo, hi] = band_extrema(&prefix, a, b);
                    let error = 640.0 * 2.0_f64.powi(-24) * sat_max / a.powi(2)
                        + 0.0005 * max_deviation / a
                        + 3e-7;
                    [mean + lo - error, mean + hi + error]
                };
                assert!(darkest > 1e-9 && lightest.is_finite());
                let lower = -(lightest / mean).log10() - 0.00005;
                let upper = -(darkest / mean).log10() + 0.00005;
                let [lo, hi] = extrema[index / 17].map(|side| f64::from(side[index % 17]) / 1000.0);
                assert!(
                    lo <= lower && upper <= hi,
                    "placement band ending at {end}, tile {index}: {lower} to {upper} exceeds [{lo}, {hi}]"
                );
            }
            for quarter in 4..150 {
                let low = f64::from(quarter) / 4.0;
                let high = low + 0.25;
                // Float32 right-edge construction can move a side length
                // by an ulp even at the most distant guarded placement.
                let a = low - 1.0 / 64.0;
                let b = high + 1.0 / 64.0;
                let corners = rectangle_extrema(&prefix, a, a);
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
        let tiles = bank.sums().as_chunks::<{ STRIDE * STRIDE }>().0;
        for (r, record) in bank.records().iter().enumerate() {
            for k in 0..16 {
                certify_bright_pair(
                    [&tiles[17 * r + k], &tiles[17 * r + k + 1]],
                    [
                        f64::from(record.mean_light[k]),
                        f64::from(record.mean_light[k + 1]),
                    ],
                    r,
                    k,
                );
            }
        }
    }

    #[test]
    fn adjacent_extrema_enclose_fractional_levels() {
        for a in [0.0, 0.031, 0.5, 0.9] {
            for b in [0.0, 0.05, 0.7, 0.8] {
                for rho in [0.25, 0.53, 0.7, 0.99999] {
                    for [low, high] in [[0.0, 0.5], [0.5, 1.0], [0.0, 1.0]] {
                        let bound = adjacent_extreme(a, b, rho, [low, high]);
                        for j in 0..=128 {
                            let t = low + (high - low) * f64::from(j) / 128.0;
                            let value = ((1.0 - t) * a + t * b)
                                / (1.0 - 2.0 * t * (1.0 - t) * (1.0 - rho)).sqrt();
                            assert!(value <= bound + 1e-14);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn high_gain_boxes_cover_footprint_band_edges() {
        let assets = assets();
        let bank = assets.portra_grain_bank().unwrap();
        let development = development(&assets);
        let maximum = Bounds::film_v1_clean_maximum(&development);
        let review = 1.953_125_f32;
        let bright_probe = 1.117_187_5_f32;
        let widths =
            [
                MIN_PROXY_SAMPLE_FOOTPRINT,
                0.25,
                0.5,
                bright_probe.next_down(),
                bright_probe,
                bright_probe.next_up(),
                review.next_down(),
                review,
                review.next_up(),
                CALIBRATED_MAX_FOOTPRINT.next_down(),
                CALIBRATED_MAX_FOOTPRINT,
                CALIBRATED_MAX_FOOTPRINT.next_up(),
                SINGLE_WRAP_MAX_FOOTPRINT.next_down(),
                SINGLE_WRAP_MAX_FOOTPRINT,
                SINGLE_WRAP_MAX_FOOTPRINT.next_up(),
                256.0_f32.next_down(),
                256.0,
                256.0_f32.next_up(),
                MAX_FOOTPRINT,
            ]
            .into_iter()
            .chain([3840.0, 4832.0].into_iter().flat_map(|pixels| {
                [8.0_f32, 11.66].map(|mm| (f64::from(mm) * 2000.0 / pixels) as f32)
            }))
            .chain(
                placement_extrema::BANDS
                    .iter()
                    .flat_map(|&(end, _)| [end.next_down(), end, end.next_up()]),
            );
        for width in widths {
            let table = assets.portra_grain_table(&bank, width).unwrap();
            let [lower, upper] = allowances(&bank, &table, 16.0).unwrap();
            let original_hi = maximum.map(|d| upper_pad((d + upper).next_up()));
            if Bounds::checked_film_v1(&development, [lower; 3], original_hi).is_ok() {
                continue;
            }
            check_colour(&development, [width; 2], |colour| {
                density_range(&bank, &table, 16.0, colour, maximum).unwrap()
            });
        }
    }

    #[test]
    fn calibration_intervals_enclose_executed_words_and_signed_guards() {
        let assets = assets();
        let bank = assets.portra_grain_bank().unwrap();
        let development = development(&assets);
        let maximum = Bounds::film_v1_clean_maximum(&development);
        for (first, steps) in [
            (MIN_PROXY_SAMPLE_FOOTPRINT, 512),
            (f32::from_bits(1.0_f32.to_bits() - 128), 127),
            (1.0, 128),
            (1.117_187_5, 256),
            (f32::from_bits(1.5_f32.to_bits() - 256), 256),
            (1.5_f32.next_up(), 512),
            (1.953_125, 1024),
            (3.311_258_3, 512),
            (4.826_159, 256),
            (6.072_916_5, 256),
            (f32::from_bits(10.0_f32.to_bits() - 128), 128),
        ] {
            let last = f32::from_bits(first.to_bits() + steps);
            // The endpoint widths must select the same placement band.
            let band = |width| {
                if width < 1.0 {
                    0
                } else {
                    placement_extrema::BANDS[1..]
                        .iter()
                        .position(|&(end, _)| width <= end)
                        .unwrap()
                        + 1
                }
            };
            assert_eq!(band(first), band(last));
            let bounds = calibration_interval::normalize(&bank, first, last);
            for width in [first, f32::from_bits(first.to_bits() + steps / 2), last] {
                let actual = assets.portra_grain_table(&bank, width).unwrap();
                for (r, record) in bank.records().iter().enumerate() {
                    for k in 0..17 {
                        let [lo, hi] = bounds.mean.map(|side| side[r][k]);
                        assert!((lo..=hi).contains(&actual.mean_density[r][k]));
                        assert!(bounds.correlation[r][k] <= actual.correlation[r][k]);
                        if first >= 1.0 {
                            for mean in [lo, hi] {
                                let shift =
                                    f64::from(mean) + f64::from(record.mean_light[k]).log10();
                                assert!(shift.abs() * f64::from(last) <= MAX_SCALED_MEAN_SHIFT);
                            }
                            assert!(bounds.correlation[r][k] >= MIN_CORRELATION);
                        } else {
                            assert!(bounds.correlation[r][k] > -0.9997);
                        }
                    }
                }
            }

            let mut table = (*assets.portra_grain_table(&bank, first).unwrap()).clone();
            table.correlation = bounds.correlation;
            let [mut dark, mut bright] = [table.clone(), table];
            dark.mean_density = bounds.mean[0];
            bright.mean_density = bounds.mean[1];
            check_colour(&development, [first, last], |colour| {
                // A higher mean strengthens the bright allowance and a
                // lower mean strengthens the dark allowance. The lower
                // correlation strengthens both at any level fraction.
                [
                    density_range(&bank, &bright, 16.0, colour, maximum).unwrap()[0],
                    density_range(&bank, &dark, 16.0, colour, maximum).unwrap()[1],
                ]
            });
        }
    }

    #[test]
    fn retained_calibrated_guards_follow_from_the_normalization_limits() {
        let assets = assets();
        let development = development(&assets);
        let maximum = Bounds::film_v1_clean_maximum(&development);
        let mut start = 10.0;
        for &(end, [lower, upper]) in &SCALED_DELTAS[5..] {
            let scale = 16.0 * 3.012 * (2.0 / (1.0 + f64::from(MIN_CORRELATION))).sqrt() / start;
            let lo = (((lower - MAX_SCALED_MEAN_SHIFT) * scale - 0.002) as f32).next_down();
            let hi = (((upper + MAX_SCALED_MEAN_SHIFT) * scale + 0.002) as f32).next_up();
            Bounds::checked_film_v1(
                &development,
                [lo; 3],
                maximum.map(|d| upper_pad((d + hi).next_up())),
            )
            .unwrap_or_else(|error| panic!("widths {start} to {end}: {error}"));
            start = f64::from(end);
        }
    }

    #[test]
    fn whole_period_tables_clear_the_retained_guards() {
        let assets = assets();
        let bank = assets.portra_grain_bank().unwrap();
        let development = development(&assets);
        let mut table = (*assets.portra_grain_table(&bank, TILE as f32).unwrap()).clone();
        let tiles = bank.sums().as_chunks::<{ STRIDE * STRIDE }>().0;
        assert!(bank.sums().iter().all(|s| s.abs() < 400.0));
        for (r, record) in bank.records().iter().enumerate() {
            for (k, &mean) in record.mean_light.iter().enumerate() {
                let complete = f64::from(tiles[17 * r + k][STRIDE * STRIDE - 1]);
                let light = f64::from(mean) + complete / (TILE * TILE) as f64;
                // A square of n whole periods has integral n² times the
                // last SAT word for every placement. Each of the four
                // prefixes is bounded by 400*(n+1)² and uses exact integer
                // coordinates here. With n>=1 the Float64 arithmetic
                // error divided by n²*256², plus the light addition, is
                // below 1e-12. Both ends round to the same light word.
                let [lo, hi] = [(light - 1e-12) as f32, (light + 1e-12) as f32];
                assert_eq!(lo.to_bits(), hi.to_bits());
                assert_eq!((-f64::from(lo).log10()) as f32, table.mean_density[r][k]);
                assert_eq!(table.correlation[r][k], 1.0);
            }
        }
        // The wide calibration centers its reduction on the first sample,
        // so uniform light gives exactly zero variance at every whole period.
        for width in (TILE..=MAX_FOOTPRINT as usize).step_by(TILE) {
            table.footprint = width as f32;
            let allowance = allowances(&bank, &table, 16.0).unwrap();
            Bounds::film_v1(&development, allowance)
                .unwrap_or_else(|error| panic!("whole period {width}: {error}"));
        }
    }

    #[test]
    fn periodic_allowance_covers_the_frozen_bank_and_normalization_floor() {
        let assets = assets();
        let bank = assets.portra_grain_bank().unwrap();
        let development = development(&assets);
        let tiles = bank.sums().as_chunks::<{ STRIDE * STRIDE }>().0;
        let mut stripe_limit = 0.0_f64;
        let mut corner_limit = 0.0_f64;
        for (tile, mean) in tiles
            .iter()
            .zip(bank.records().iter().flat_map(|record| record.mean_light))
        {
            assert_eq!(tile[STRIDE * STRIDE - 1], 0.0);
            let range = |line: &[f32]| {
                let bounds = line.iter().fold([0.0_f64, 0.0], |[lo, hi], &word| {
                    [lo.min(f64::from(word)), hi.max(f64::from(word))]
                });
                bounds[1] - bounds[0]
            };
            let row = &tile[TILE * STRIDE..];
            let column: [f32; STRIDE] = std::array::from_fn(|row| tile[row * STRIDE + TILE]);
            let stripe = range(row) + range(&column);
            let prefix = tile
                .iter()
                .map(|&word| f64::from(word).abs())
                .fold(0.0, f64::max);
            // Each residual stripe is an edge-prefix difference. A
            // wrapped corner adds at most one difference per axis to
            // its four interior prefixes.
            stripe_limit = stripe_limit.max(stripe / f64::from(mean));
            corner_limit = corner_limit.max((4.0 * prefix + stripe) / f64::from(mean));
        }
        assert!(stripe_limit <= PERIODIC_STRIP_LIMIT);
        assert!(corner_limit <= PERIODIC_CORNER_LIMIT);

        let width = PERIODIC_ALLOWANCE_MIN_FOOTPRINT as f32;
        let measured = assets.portra_grain_table(&bank, width).unwrap();
        let retained =
            Bounds::film_v1(&development, allowances(&bank, &measured, 16.0).unwrap()).unwrap();
        let selected = Bounds::portra_film_v1(&development, &bank, &measured, 16.0, 1.0).unwrap();
        assert_eq!(retained.hash(), selected.hash());

        let mut worst = (*measured).clone();
        // The production table accepts correlations strictly above this
        // floor. This synthetic table only exercises the enclosure limit.
        worst.correlation = [[(-0.9997_f32).next_up(); 17]; 3];
        let original = Bounds::film_v1(&development, allowances(&bank, &worst, 16.0).unwrap());
        assert!(matches!(
            original,
            Err(film::density::Error::FilmProjectionSpan)
        ));
        let expected =
            Bounds::film_v1(&development, periodic_allowances(&worst, 16.0).unwrap()).unwrap();
        for colour in [0.0, 0.5, 1.0] {
            let fallback =
                Bounds::portra_film_v1(&development, &bank, &worst, 16.0, colour).unwrap();
            assert_eq!(fallback.hash(), expected.hash());
        }
        worst.footprint = SINGLE_WRAP_MAX_FOOTPRINT;
        assert!(periodic_allowances(&worst, 16.0).is_none());
    }

    #[test]
    fn retained_periodic_guards_cover_the_bounded_table_statistics() {
        let assets = assets();
        let development = development(&assets);
        let maximum = Bounds::film_v1_clean_maximum(&development);
        // The ratio w / ceil(w / 37.5) exceeds 28 for w > 255,
        // so only the final scaled band can contribute.
        let piece = 1.0 / (1.0 / 37.5 + 1.0 / 255.0);
        let scale = 16.0 * 3.012 * (2.0 / (0.25 - 8.0 * f64::from(f32::EPSILON))).sqrt();
        let lower = (((-4.03 / piece - 0.005) * scale - 0.002) as f32).next_down();
        let upper = (((4.20 / piece + 0.005) * scale + 0.002) as f32).next_up();
        Bounds::checked_film_v1(
            &development,
            [lower; 3],
            maximum.map(|d| upper_pad((d + upper).next_up())),
        )
        .unwrap();
        Bounds::film_v1(&development, [-20.5, 20.5]).unwrap();
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
