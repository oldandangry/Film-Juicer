//! Enclose represented calibration words between two footprint words.

use crate::assets::PortraGrainBank;

const SIDE: usize = 256;
const STRIDE: usize = SIDE + 1;
const LEVELS: usize = 17;
const GRID: usize = 128;
const SAMPLES: usize = GRID * GRID;

#[derive(Clone, Copy)]
struct Placement {
    block: [u32; 2],
    edges: [f64; 4],
}

impl Placement {
    fn at(width: f64, lattice: [usize; 2], seed: u32) -> Self {
        let corner = lattice.map(|i| i as f64 * width);
        // The Float32 width has at most 24 significant bits. These products
        // and translations are exact in Float64 for the calibration lattice.
        let block = corner.map(|x| ((x + width * 0.5) / 128.0).floor() as u32);
        let hash = pcg(block[0] ^ pcg(block[1] ^ seed));
        let mut axes = std::array::from_fn::<_, 2, _>(|axis| {
            let shift = f64::from(block[axis]) * 128.0;
            let [lo, hi] = [corner[axis] - shift, corner[axis] + width - shift];
            if hash & (1 << (16 + axis)) != 0 {
                [128.0 - hi, 128.0 - lo]
            } else {
                [lo, hi]
            }
        });
        if hash & (1 << 18) != 0 {
            axes.swap(0, 1);
        }
        for (axis, bounds) in axes.iter_mut().enumerate() {
            let offset = 64.0 + f64::from((hash >> (8 * axis)) & 255);
            *bounds = bounds.map(|x| x + offset);
        }
        Self {
            block,
            edges: [axes[0][0], axes[0][1], axes[1][0], axes[1][1]],
        }
    }

    fn cell(self) -> [usize; 4] {
        self.edges.map(|x| x as usize)
    }
}

fn pcg(input: u32) -> u32 {
    let state = input.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

fn prefix(tile: &[f32], x: f64, y: f64) -> f64 {
    let inside = |x: f64, y: f64| {
        let i = (x as usize).min(SIDE - 1);
        let j = (y as usize).min(SIDE - 1);
        let u = x - i as f64;
        let v = y - j as f64;
        let [a, b, c, d] = [0, 1, STRIDE, STRIDE + 1].map(|p| f64::from(tile[j * STRIDE + i + p]));
        a + (b - a) * u + (c - a) * v + (d - b - c + a) * u * v
    };
    let wrap_x = x >= SIDE as f64;
    let wrap_y = y >= SIDE as f64;
    let (x, y) = (
        x - if wrap_x { SIDE as f64 } else { 0.0 },
        y - if wrap_y { SIDE as f64 } else { 0.0 },
    );
    inside(x, y)
        + if wrap_x { inside(SIDE as f64, y) } else { 0.0 }
        + if wrap_y { inside(x, SIDE as f64) } else { 0.0 }
        + if wrap_x && wrap_y {
            inside(SIDE as f64, SIDE as f64)
        } else {
            0.0
        }
}

fn average(tile: &[f32], [x0, x1, y0, y1]: [f64; 4], width: f64) -> f64 {
    (prefix(tile, x1, y1) - prefix(tile, x0, y1) - prefix(tile, x1, y0) + prefix(tile, x0, y0))
        / (width * width)
}

fn light_residuals(
    tiles: &[[f32; STRIDE * STRIDE]],
    lattice: [usize; 2],
    seed: u32,
    first: u32,
    last: u32,
    limits: &mut [[f64; 2]; LEVELS],
) {
    let [left, right] = [first, last].map(|bits| f64::from(f32::from_bits(bits)));
    let a = Placement::at(left, lattice, seed);
    let b = Placement::at(right, lattice, seed);
    // Block indices are monotone. With a fixed placement the edges are
    // affine, so equal endpoint cells certify all intermediate cells.
    if a.block != b.block || a.cell() != b.cell() {
        let mid = first + (last - first) / 2;
        light_residuals(tiles, lattice, seed, first, mid, limits);
        light_residuals(tiles, lattice, seed, mid + 1, last, limits);
        return;
    }

    // Within these cells the four prefix differences are a quadratic
    // in width. Dividing by width squared makes the average quadratic
    // in inverse width. Its three Bernstein coefficients enclose it.
    let center = 2.0 * left * right / (left + right);
    let t = if first == last {
        0.0
    } else {
        (center - left) / (right - left)
    };
    let middle = std::array::from_fn(|i| a.edges[i] + t * (b.edges[i] - a.edges[i]));
    for (tile, [min, max]) in tiles.iter().zip(limits) {
        let lo = average(tile, a.edges, left);
        let hi = average(tile, b.edges, right);
        let mid = 2.0 * average(tile, middle, center) - 0.5 * (lo + hi);
        *min = min.min(lo.min(hi).min(mid));
        *max = max.max(lo.max(hi).max(mid));
    }
}

/// Lower and upper bounds on each mean word and a lower bound on each
/// adjacent correlation word. This proof uses the calibrated placement
/// regime, before the lower corners are moved into the first period.
pub(super) struct Normalization {
    pub mean: [[[f32; LEVELS]; 3]; 2],
    pub correlation: [[f32; LEVELS]; 3],
}

pub(super) fn normalize(bank: &PortraGrainBank, first: f32, last: f32) -> Normalization {
    assert!(super::MIN_PROXY_SAMPLE_FOOTPRINT <= first && first <= last);
    assert!(last <= super::CALIBRATED_MAX_FOOTPRINT);
    assert!(last <= 2.0 * first);
    // The harmonic point incurs at most three Float64 unit roundoffs
    // times the upper width. Corner slopes are at most 4096. Including
    // interpolation gives a coordinate error below 1.5e-12 * last +
    // 4e-13. Each wrapped prefix slope is at most 1600. Charge both
    // axes, four corners, and twice the midpoint error in the Bernstein
    // coefficient, plus the endpoint and division roundoffs.
    assert!(bank.sums().iter().all(|s| s.abs() < 400.0));
    let width = f64::from(first);
    let residual_pad = (2e-8 + 4e-8 * f64::from(last)) / (width * width);

    let mut result = Normalization {
        mean: [[[0.0; LEVELS]; 3]; 2],
        correlation: [[1.0; LEVELS]; 3],
    };
    let tiles = bank.sums().as_chunks::<{ STRIDE * STRIDE }>().0;
    for (r, record) in bank.records().iter().enumerate() {
        let seed = pcg(0x5a3f_1e1d ^ (200_u32 + r as u32).wrapping_mul(0x9e37_79b9));
        let mut fields = vec![[0.0_f64; LEVELS]; SAMPLES];
        let mut radius_squared = [0.0_f64; LEVELS];
        let mut means = [0.0_f64; LEVELS];
        let mut mean_radii = [0.0_f64; LEVELS];
        for (i, point) in fields.iter_mut().enumerate() {
            let lattice = [(i % GRID * 7919) % 4096, (i / GRID * 104729) % 4096];
            let mut residuals = [[f64::INFINITY, f64::NEG_INFINITY]; LEVELS];
            light_residuals(
                &tiles[r * LEVELS..(r + 1) * LEVELS],
                lattice,
                seed,
                first.to_bits(),
                last.to_bits(),
                &mut residuals,
            );
            for (k, [low, high]) in residuals.into_iter().enumerate() {
                // Cast the residual then add the stored light just as the
                // measured table does. Both round-to-nearest steps are monotone.
                let density = |residual: f64| {
                    -f64::from(record.mean_light[k] + residual as f32)
                        .max(1e-9)
                        .log10()
                };
                let lo = density(high + residual_pad) - 1e-12;
                let hi = density(low - residual_pad) + 1e-12;
                point[k] = (lo + hi) * 0.5;
                let radius = (hi - lo) * 0.5;
                radius_squared[k] += radius * radius;
                means[k] += point[k];
                mean_radii[k] += radius;
            }
        }

        let mut norm_squared = [0.0_f64; LEVELS];
        let mut dot = [0.0_f64; LEVELS - 1];
        for k in 0..LEVELS {
            means[k] /= SAMPLES as f64;
            mean_radii[k] = mean_radii[k] / SAMPLES as f64 + 1e-10;
            result.mean[0][r][k] = (means[k] - mean_radii[k]) as f32;
            result.mean[1][r][k] = (means[k] + mean_radii[k]) as f32;
        }
        for point in fields {
            let centered = std::array::from_fn::<_, LEVELS, _>(|k| point[k] - means[k]);
            for (k, &value) in centered.iter().enumerate() {
                norm_squared[k] += value * value;
            }
            for (product, pair) in dot.iter_mut().zip(centered.windows(2)) {
                *product += pair[0] * pair[1];
            }
        }

        // Centering is an orthogonal projection and cannot enlarge each
        // field error. A ball of radius e about a vector of length n
        // subtends asin(e/n). Add the two angular errors to the angle
        // between the centers. The pads bound the Float64 reductions.
        let norm = norm_squared.map(f64::sqrt);
        let angular = std::array::from_fn::<_, LEVELS, _>(|k| {
            let ratio = (radius_squared[k].sqrt() + 2e-8) / (norm[k] - 2e-8);
            if (0.0..1.0).contains(&ratio) {
                ratio.asin()
            } else {
                std::f64::consts::PI
            }
        });
        for (k, &product) in dot.iter().enumerate() {
            let bound = if norm[k].min(norm[k + 1]) > 2e-8 {
                let angle = (product / (norm[k] * norm[k + 1]) - 1e-10)
                    .clamp(-1.0, 1.0)
                    .acos();
                (angle + angular[k] + angular[k + 1])
                    .min(std::f64::consts::PI)
                    .cos()
                    - 1e-10
            } else {
                -1.0
            };
            result.correlation[r][k] = (bound as f32).next_down().max(-1.0);
        }
    }
    result
}
