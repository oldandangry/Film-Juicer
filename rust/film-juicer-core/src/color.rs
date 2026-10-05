//! CAT16 host preparation. XYZ operands are unsanitized; whites are Y-relative.

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
}
