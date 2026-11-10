use film_juicer_core::{color, exposure};
fn invalid() -> exposure::MallettInput<'static> {
    let basis = [[1.0; 3]; 81];
    let illuminant = [1.0; 81];
    let sensitivity = [[1.0; 3]; 81];
    exposure::MallettInput {
        color: color::InputConversion { space: color::InputSpace::SrgbRec709, decode_cctf: false, rgb_to_xyz: [1.0; 9], xyz_adaptation: None },
        xyz_to_linear_srgb: [1.0; 9],
        basis_rgb: &basis,
        illuminant: &illuminant,
        sensitivity_rgb: &sensitivity,
    }
}
