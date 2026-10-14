//! Completed immutable output constants. CUDA owns CCTF evaluation and clipping.

use std::sync::OnceLock;

use crate::hash;
use super::OutputSpace;

#[derive(Clone, Copy)]
#[repr(u8)]
pub enum CctfKind {
    Linear,
    Gamma,
    Srgb,
    Bt2020,
    ProPhoto,
    DavinciIntermediate,
}

#[derive(Clone, Copy)]
pub struct Cctf {
    pub kind: CctfKind,
    pub gamma: f32,
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub linear_cutoff: f32,
}

pub struct OutputColor {
    rgb_to_xyz: [f32; 9],
    xyz_to_rgb: [f32; 9],
    white_xyz: [f32; 3],
    cctf: Cctf,
    dwg_to_output: [f32; 9],
    hash: u64,
}

impl OutputColor {
    pub fn rgb_to_xyz(&self) -> &[f32; 9] {
        &self.rgb_to_xyz
    }
    pub fn xyz_to_rgb(&self) -> &[f32; 9] {
        &self.xyz_to_rgb
    }
    pub fn white_xyz(&self) -> &[f32; 3] {
        &self.white_xyz
    }
    pub fn cctf(&self) -> Cctf {
        self.cctf
    }
    pub fn dwg_to_output(&self) -> &[f32; 9] {
        &self.dwg_to_output
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

struct Literal {
    base: [f32; 11],
    rgb_to_xyz: [f32; 9],
    xyz_to_rgb: [f32; 9],
    kind: CctfKind,
    coefficients: [f32; 6],
}

// Linear DWG source is distinct from the Intermediate output entry, including its identity.
#[expect(
    clippy::excessive_precision,
    reason = "Preserve exact authored native f32 literals"
)]
const SOURCE: Literal = Literal {
    base: [
        0.8000, 0.3130, 0.1682, 0.9877, 0.0790, -0.1155, 0.3127, 0.3290, 0.950455, 1.0, 1.089058,
    ],
    rgb_to_xyz: [
        0.70062239,
        0.14877482,
        0.10105872,
        0.27411851,
        0.87363190,
        -0.14775041,
        -0.09896291,
        -0.13789533,
        1.32591599,
    ],
    xyz_to_rgb: [
        1.51667204,
        -0.28147805,
        -0.14696363,
        -0.46491710,
        1.25142378,
        0.17488461,
        0.07578536,
        0.08076209,
        0.76034476,
    ],
    kind: CctfKind::Linear,
    coefficients: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
};

#[expect(
    clippy::excessive_precision,
    reason = "Preserve exact authored native f32 literals and fractions"
)]
const LITERALS: [Literal; 9] = [
    Literal {
        base: [
            0.6400, 0.3300, 0.3000, 0.6000, 0.1500, 0.0600, 0.3127, 0.3290, 0.95045593, 1.0,
            1.08905775,
        ],
        rgb_to_xyz: [
            0.41240000, 0.35760000, 0.18050000, 0.21260000, 0.71520000, 0.07220000, 0.01930000,
            0.11920000, 0.95050000,
        ],
        xyz_to_rgb: [
            3.24060000,
            -1.53720000,
            -0.49860000,
            -0.96890000,
            1.87580000,
            0.04150000,
            0.05570000,
            -0.20400000,
            1.05700000,
        ],
        kind: CctfKind::Srgb,
        coefficients: [2.4, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    Literal {
        base: [
            0.6800, 0.3200, 0.2650, 0.6900, 0.1500, 0.0600, 0.3140, 0.3510, 0.89458689, 1.0,
            0.95441595,
        ],
        rgb_to_xyz: [
            0.44516982,
            0.27713441,
            0.17228267,
            0.20949168,
            0.72159525,
            0.06891307,
            -0.00000000,
            0.04706056,
            0.90735539,
        ],
        xyz_to_rgb: [
            2.72539403,
            -1.01800301,
            -0.44016320,
            -0.79516803,
            1.68973205,
            0.02264719,
            0.04124189,
            -0.08763902,
            1.10092938,
        ],
        kind: CctfKind::Gamma,
        coefficients: [1.0 / 2.6, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    Literal {
        base: [
            0.6800, 0.3200, 0.2650, 0.6900, 0.1500, 0.0600, 0.3127, 0.3290, 0.95045593, 1.0,
            1.08905775,
        ],
        rgb_to_xyz: [
            0.48657095,
            0.26566769,
            0.19821729,
            0.22897456,
            0.69173852,
            0.07928691,
            -0.00000000,
            0.04511338,
            1.04394437,
        ],
        xyz_to_rgb: [
            2.49349691,
            -0.93138362,
            -0.40271078,
            -0.82948897,
            1.76266406,
            0.02362469,
            0.03584583,
            -0.07617239,
            0.95688452,
        ],
        kind: CctfKind::Srgb,
        coefficients: [2.4, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    Literal {
        base: [
            0.6400, 0.3300, 0.2100, 0.7100, 0.1500, 0.0600, 0.3127, 0.3290, 0.95045593, 1.0,
            1.08905775,
        ],
        rgb_to_xyz: [
            0.57667000, 0.18556000, 0.18823000, 0.29734000, 0.62736000, 0.07529000, 0.02703000,
            0.07069000, 0.99134000,
        ],
        xyz_to_rgb: [
            2.04159000,
            -0.56501000,
            -0.34473000,
            -0.96924000,
            1.87597000,
            0.04156000,
            0.01344000,
            -0.11836000,
            1.01517000,
        ],
        kind: CctfKind::Gamma,
        coefficients: [1.0 / 2.19921875, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    Literal {
        base: [
            0.7080, 0.2920, 0.1700, 0.7970, 0.1310, 0.0460, 0.3127, 0.3290, 0.95045593, 1.0,
            1.08905775,
        ],
        rgb_to_xyz: [
            0.63695805, 0.14461690, 0.16888098, 0.26270021, 0.67799807, 0.05930172, 0.00000000,
            0.02807269, 1.06098506,
        ],
        xyz_to_rgb: [
            1.71665119,
            -0.35567078,
            -0.25336628,
            -0.66668435,
            1.61648124,
            0.01576855,
            0.01763986,
            -0.04277061,
            0.94210312,
        ],
        kind: CctfKind::Bt2020,
        coefficients: [1.0, 1.09929681, 0.01805397, 0.0, 0.0, 0.0],
    },
    Literal {
        base: [
            0.7347, 0.2653, 0.1596, 0.8404, 0.0366, 0.0001, 0.34567, 0.35850, 0.96429568, 1.0,
            0.82510460,
        ],
        rgb_to_xyz: [
            0.79770000, 0.13520000, 0.03130000, 0.28800000, 0.71190000, 0.00010000, 0.00000000,
            0.00000000, 0.82490000,
        ],
        xyz_to_rgb: [
            1.34600000,
            -0.25560000,
            -0.05110000,
            -0.54460000,
            1.50820000,
            0.02050000,
            0.00000000,
            0.00000000,
            1.21230000,
        ],
        kind: CctfKind::ProPhoto,
        coefficients: [1.0 / 1.8, 0.0, 0.0, 0.0, 0.0, 0.001953125],
    },
    Literal {
        base: [
            0.7347, 0.2653, 0.0000, 1.0000, 0.0001, -0.0770, 0.32168, 0.33767, 0.95264607, 1.0,
            1.00882518,
        ],
        rgb_to_xyz: [
            0.95255240,
            0.00000000,
            0.00009368,
            0.34396645,
            0.72816610,
            -0.07213255,
            0.00000000,
            0.00000000,
            1.00882518,
        ],
        xyz_to_rgb: [
            1.04981102,
            0.00000000,
            -0.00009748,
            -0.49590302,
            1.37331305,
            0.09824004,
            0.00000000,
            0.00000000,
            0.99125202,
        ],
        kind: CctfKind::Linear,
        coefficients: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    Literal {
        base: [
            0.8000, 0.3130, 0.1682, 0.9877, 0.0790, -0.1155, 0.3127, 0.3290, 0.950455, 1.0,
            1.089058,
        ],
        rgb_to_xyz: [
            0.70062239,
            0.14877482,
            0.10105872,
            0.27411851,
            0.87363190,
            -0.14775041,
            -0.09896291,
            -0.13789533,
            1.32591599,
        ],
        xyz_to_rgb: [
            1.51667204,
            -0.28147805,
            -0.14696363,
            -0.46491710,
            1.25142378,
            0.17488461,
            0.07578536,
            0.08076209,
            0.76034476,
        ],
        kind: CctfKind::DavinciIntermediate,
        coefficients: [1.0, 0.0075, 7.0, 0.07329248, 10.44426855, 0.00262409],
    },
    Literal {
        base: [
            0.6400, 0.3300, 0.3000, 0.6000, 0.1500, 0.0600, 0.3127, 0.3290, 0.950455, 1.0, 1.089058,
        ],
        rgb_to_xyz: [
            0.4124564, 0.3575761, 0.1804375, 0.2126729, 0.7151522, 0.0721750, 0.0193339, 0.1191920,
            0.9503041,
        ],
        xyz_to_rgb: [
            3.2404542, -1.5371385, -0.4985314, -0.9692660, 1.8760108, 0.0415560, 0.0556434,
            -0.2040259, 1.0572252,
        ],
        kind: CctfKind::Gamma,
        coefficients: [1.0 / 2.4, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
];

fn subhashes(literal: &Literal) -> [u64; 4] {
    // A nonfinite component contributes zero; outer composition still runs.
    [
        &literal.base[..],
        &literal.rgb_to_xyz[..],
        &literal.xyz_to_rgb[..],
        &literal.coefficients[..],
    ]
    .map(|values| hash::finite_f32s(values).unwrap_or(0))
}

fn entry_hash(literal: &Literal) -> u64 {
    let [base, rgb, inverse, cctf] = subhashes(literal);
    hash::u64s(&[base, rgb, inverse, literal.kind as u64, cctf])
}

fn complete(literal: &Literal) -> OutputColor {
    let a = literal.xyz_to_rgb;
    let b = SOURCE.rgb_to_xyz;
    let mut dwg_to_output = [0.0; 9];
    for row in 0..3 {
        for column in 0..3 {
            // Native three-term f32 schedule; no zero accumulator, f64 or FMA.
            dwg_to_output[row * 3 + column] = (a[row * 3] * b[column]
                + a[row * 3 + 1] * b[3 + column])
                + a[row * 3 + 2] * b[6 + column];
        }
    }
    let [gamma, a, b, c, d, linear_cutoff] = literal.coefficients;
    OutputColor {
        rgb_to_xyz: literal.rgb_to_xyz,
        xyz_to_rgb: literal.xyz_to_rgb,
        white_xyz: [literal.base[8], literal.base[9], literal.base[10]],
        cctf: Cctf {
            kind: literal.kind,
            gamma,
            a,
            b,
            c,
            d,
            linear_cutoff,
        },
        dwg_to_output,
        hash: entry_hash(literal),
    }
}

static CATALOGUE: OnceLock<[OutputColor; 9]> = OnceLock::new();

pub fn output_color(space: OutputSpace) -> &'static OutputColor {
    &CATALOGUE.get_or_init(|| std::array::from_fn(|index| complete(&LITERALS[index])))
        [space as usize]
}

#[cfg(test)]
mod tests;
