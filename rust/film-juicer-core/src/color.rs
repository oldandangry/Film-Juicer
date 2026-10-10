//! CAT02/CAT16 host preparation. XYZ operands are unsanitized; whites are Y-relative.

#[derive(Clone, Copy)]
pub struct Whites {
    pub source_xyz: [f32; 3],
    pub destination_xyz: [f32; 3],
}

const CAT16: [f32; 9] = [
    0.401288, 0.650173, -0.051461, -0.250268, 1.204414, 0.045854, -0.002079, 0.048952, 0.953127,
];

// The accepted native initializer uses this f32 cofactor sequence. Keep every
// rounding point, including the reciprocal before multiplying each cofactor.
const fn cat16_inverse() -> [f32; 9] {
    let m = CAT16;
    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    let inv_det = 1.0 / det;
    [
        (m[4] * m[8] - m[5] * m[7]) * inv_det,
        (m[2] * m[7] - m[1] * m[8]) * inv_det,
        (m[1] * m[5] - m[2] * m[4]) * inv_det,
        (m[5] * m[6] - m[3] * m[8]) * inv_det,
        (m[0] * m[8] - m[2] * m[6]) * inv_det,
        (m[2] * m[3] - m[0] * m[5]) * inv_det,
        (m[3] * m[7] - m[4] * m[6]) * inv_det,
        (m[1] * m[6] - m[0] * m[7]) * inv_det,
        (m[0] * m[4] - m[1] * m[3]) * inv_det,
    ]
}
const CAT16_INVERSE: [f32; 9] = cat16_inverse();

fn multiply(matrix: [f32; 9], xyz: [f32; 3]) -> [f32; 3] {
    [
        matrix[0] * xyz[0] + matrix[1] * xyz[1] + matrix[2] * xyz[2],
        matrix[3] * xyz[0] + matrix[4] * xyz[1] + matrix[5] * xyz[2],
        matrix[6] * xyz[0] + matrix[7] * xyz[1] + matrix[8] * xyz[2],
    ]
}

fn normalize_white(mut xyz: [f32; 3]) -> [f32; 3] {
    for value in &mut xyz {
        // std::max(0.0f, value) selects the first operand on a zero tie.
        if !value.is_finite() || *value <= 0.0 {
            *value = 0.0;
        }
    }
    let y = if xyz[1] > 0.0 { xyz[1] } else { 1.0 };
    let inv_y = 1.0 / y;
    for value in &mut xyz {
        *value *= inv_y;
    }
    xyz[1] = 1.0;
    xyz
}

pub fn adapt_cat16(xyz: [f32; 3], whites: Whites) -> [f32; 3] {
    let source_lms = multiply(CAT16, normalize_white(whites.source_xyz));
    let destination_lms = multiply(CAT16, normalize_white(whites.destination_xyz));
    let mut lms = multiply(CAT16, xyz);
    for channel in 0..3 {
        let scale = if source_lms[channel] > 1e-6 {
            destination_lms[channel] / source_lms[channel]
        } else {
            1.0
        };
        lms[channel] *= scale;
    }
    multiply(CAT16_INVERSE, lms)
}

pub fn cat16_matrix(whites: Whites) -> [f32; 9] {
    let mut matrix = [0.0; 9];
    for column in 0..3 {
        let mut basis = [0.0; 3];
        basis[column] = 1.0;
        let adapted = adapt_cat16(basis, whites);
        for row in 0..3 {
            matrix[row * 3 + column] = adapted[row];
        }
    }
    matrix
}

const CAT02: [f32; 9] = [
    0.7328, 0.4296, -0.1624, -0.7036, 1.6975, 0.0061, 0.003, 0.0136, 0.9834,
];
const CAT02_INVERSE: [f32; 9] = [
    1.0961238, -0.278869, 0.1827452, 0.454369, 0.4735332, 0.0720978, -0.0096276, -0.005698,
    1.0153256,
];

pub fn adapt_cat02(xyz: [f32; 3], whites: Whites) -> [f32; 3] {
    let source_lms = multiply(CAT02, normalize_white(whites.source_xyz));
    let destination_lms = multiply(CAT02, normalize_white(whites.destination_xyz));
    let mut lms = multiply(CAT02, xyz);
    for channel in 0..3 {
        let scale = if source_lms[channel] > 1e-6 {
            destination_lms[channel] / source_lms[channel]
        } else {
            1.0
        };
        let value = lms[channel];
        lms[channel] = scale * value;
    }
    multiply(CAT02_INVERSE, lms)
}

pub fn cat02_matrix(whites: Whites) -> [f32; 9] {
    let mut matrix = [0.0; 9];
    for column in 0..3 {
        let mut basis = [0.0; 3];
        basis[column] = 1.0;
        let adapted = adapt_cat02(basis, whites);
        for row in 0..3 {
            matrix[row * 3 + column] = adapted[row];
        }
    }
    matrix
}

#[derive(Clone, Copy)]
pub enum InputSpace {
    DavinciWideGamut,
    Bt2020,
    Aces2065_1,
    SrgbRec709,
}

#[derive(Clone, Copy)]
pub struct InputMatrices {
    pub rgb_to_xyz: [f32; 9],
    pub nominal_white_xyz: [f32; 3],
    pub xyz_to_linear_srgb: [f32; 9],
    pub d65_white_xyz: [f32; 3],
}

#[derive(Clone, Copy)]
pub struct InputConversion {
    pub space: InputSpace,
    pub decode_cctf: bool,
    pub rgb_to_xyz: [f32; 9],
    pub xyz_adaptation: Option<[f32; 9]>,
}

pub struct ConvertedRgb {
    pub rgb: [f32; 3],
    pub xyz: [f32; 3],
}

#[expect(
    clippy::excessive_precision,
    reason = "Preserve the authored native f32 input matrix literals"
)]
const DWG_TO_XYZ: [f32; 9] = [
    0.70062239,
    0.14877482,
    0.10105872,
    0.27411851,
    0.87363190,
    -0.14775041,
    -0.09896291,
    -0.13789533,
    1.32591599,
];
#[expect(
    clippy::excessive_precision,
    reason = "The native DWG inverse is authored independently of its forward matrix"
)]
const XYZ_TO_DWG: [f32; 9] = [
    1.51667204,
    -0.28147805,
    -0.14696363,
    -0.46491710,
    1.25142378,
    0.17488461,
    0.07578536,
    0.08076209,
    0.76034476,
];
#[expect(
    clippy::excessive_precision,
    reason = "Preserve the authored native f32 input matrix literals"
)]
const BT2020_TO_XYZ: [f32; 9] = [
    0.63695806, 0.14461690, 0.16888097, 0.26270020, 0.67799807, 0.05930172, 0.00000000, 0.02807269,
    1.06098509,
];
#[expect(
    clippy::excessive_precision,
    reason = "Preserve the authored native f32 input matrix literals"
)]
const ACES_TO_XYZ: [f32; 9] = [
    0.95255238,
    0.00000000,
    0.00009368,
    0.34396645,
    0.72816610,
    -0.07213255,
    0.00000000,
    0.00000000,
    1.00882518,
];
const SRGB_TO_XYZ: [f32; 9] = [
    0.4124, 0.3576, 0.1805, 0.2126, 0.7152, 0.0722, 0.0193, 0.1192, 0.9505,
];
#[expect(
    clippy::excessive_precision,
    reason = "Input D65 is distinct from the process DWG white"
)]
const INPUT_D65: [f32; 3] = [0.95045590, 1.0, 1.08905780];
#[expect(
    clippy::excessive_precision,
    reason = "Preserve the authored native ACES nominal white"
)]
const INPUT_ACES_WHITE: [f32; 3] = [0.95264608, 1.0, 1.00882518];

fn inverse(m: [f32; 9]) -> [f32; 9] {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if !det.is_finite() || det.abs() <= 1e-6 {
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    }
    let inv_det = 1.0 / det;
    [
        (m[4] * m[8] - m[5] * m[7]) * inv_det,
        (m[2] * m[7] - m[1] * m[8]) * inv_det,
        (m[1] * m[5] - m[2] * m[4]) * inv_det,
        (m[5] * m[6] - m[3] * m[8]) * inv_det,
        (m[0] * m[8] - m[2] * m[6]) * inv_det,
        (m[2] * m[3] - m[0] * m[5]) * inv_det,
        (m[3] * m[7] - m[4] * m[6]) * inv_det,
        (m[1] * m[6] - m[0] * m[7]) * inv_det,
        (m[0] * m[4] - m[1] * m[3]) * inv_det,
    ]
}

pub fn input_matrices(space: InputSpace) -> InputMatrices {
    InputMatrices {
        rgb_to_xyz: match space {
            InputSpace::DavinciWideGamut => DWG_TO_XYZ,
            InputSpace::Bt2020 => BT2020_TO_XYZ,
            InputSpace::Aces2065_1 => ACES_TO_XYZ,
            InputSpace::SrgbRec709 => SRGB_TO_XYZ,
        },
        nominal_white_xyz: match space {
            InputSpace::Aces2065_1 => INPUT_ACES_WHITE,
            _ => INPUT_D65,
        },
        xyz_to_linear_srgb: inverse(SRGB_TO_XYZ),
        d65_white_xyz: INPUT_D65,
    }
}

fn finite_channel(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

#[expect(
    clippy::excessive_precision,
    reason = "Retain native f32 decoder constants, reciprocal multiplication and threshold branches"
)]
pub fn decode_input(space: InputSpace, decode_cctf: bool, rgb: [f32; 3]) -> [f32; 3] {
    rgb.map(|channel| {
        let value = finite_channel(channel);
        if !decode_cctf || matches!(space, InputSpace::DavinciWideGamut | InputSpace::Aces2065_1) {
            return value;
        }
        // Native max(0, value) selects positive zero on either zero tie.
        let x = if value <= 0.0 { 0.0 } else { value };
        if matches!(space, InputSpace::Bt2020) {
            const A: f32 = 1.09929681;
            const INV_A: f32 = 1.0 / A;
            if x < 0.0812428791 {
                x / 4.5
            } else {
                ((x + (A - 1.0)) * INV_A).powf(2.2222222222)
            }
        } else if x <= 0.04045 {
            x / 12.92
        } else {
            const INV_SCALE: f32 = 1.0 / 1.055;
            ((x + 0.055) * INV_SCALE).powf(2.4)
        }
    })
}

fn input_xyz(input: InputConversion, rgb: [f32; 3]) -> [f32; 3] {
    let xyz = multiply(
        input.rgb_to_xyz,
        decode_input(input.space, input.decode_cctf, rgb),
    );
    match input.xyz_adaptation {
        Some(matrix) => multiply(matrix, xyz),
        None => xyz,
    }
}

pub fn input_to_dwg(
    input: InputConversion,
    rgb: [f32; 3],
    clamp_nonnegative: bool,
) -> ConvertedRgb {
    let xyz = input_xyz(input, rgb);
    let mut converted = multiply(XYZ_TO_DWG, xyz);
    if clamp_nonnegative {
        for channel in &mut converted {
            *channel = if *channel > 0.0 { *channel } else { 0.0 };
        }
    }
    ConvertedRgb {
        rgb: converted.map(finite_channel),
        // Sanitation of this observation must not change the target's operand.
        xyz: xyz.map(|channel| {
            let value = finite_channel(channel);
            if clamp_nonnegative && value < 0.0 {
                0.0
            } else {
                value
            }
        }),
    }
}

pub fn input_to_linear_srgb(
    input: InputConversion,
    rgb: [f32; 3],
    xyz_to_linear_srgb: [f32; 9],
) -> ConvertedRgb {
    let xyz = input_xyz(input, rgb);
    ConvertedRgb {
        rgb: multiply(xyz_to_linear_srgb, xyz).map(finite_channel),
        xyz: xyz.map(finite_channel),
    }
}

pub fn linear_srgb_to_xyz(rgb: [f32; 3]) -> [f32; 3] {
    multiply(SRGB_TO_XYZ, rgb)
}

pub fn dwg_to_xyz(rgb: [f32; 3]) -> [f32; 3] {
    multiply(DWG_TO_XYZ, rgb)
}

pub fn project_linear_rgb_to_xyz(
    rgb: [f32; 3],
    rgb_to_xyz: [f32; 9],
    xyz_adapt: [f32; 9],
) -> [f32; 3] {
    multiply(xyz_adapt, multiply(rgb_to_xyz, rgb))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_matches_the_initialized_native_capture() {
        assert_eq!(
            CAT16_INVERSE.map(f32::to_bits),
            [
                0x3fee583e, 0xbf8170ca, 0x3e18c46a, 0x3ec669e1, 0x3f1f172e, 0xbc13079e, 0xbc81c607,
                0xbd0bc47d, 0x3f86653c,
            ]
        );
    }

    #[test]
    fn input_inverse_fallback_boundary() {
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let threshold = 1e-6_f32;
        for det in [
            0.0,
            threshold,
            -threshold,
            f32::from_bits(threshold.to_bits() - 1),
            f32::NAN,
            f32::INFINITY,
        ] {
            let mut m = identity;
            m[0] = det;
            assert_eq!(super::inverse(m), identity);
        }
        let mut m = identity;
        m[0] = f32::from_bits(threshold.to_bits() + 1);
        assert_ne!(super::inverse(m), identity);
    }
}
