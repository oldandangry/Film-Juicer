use film_juicer_core::{scanner, color};
fn prepare(input: scanner::ColorInput) -> Result<scanner::Color, scanner::ColorError> {
    let value = scanner::Color::new(input)?;
    let _: &[f32; 9] = value.cat02(); let _: &[f32; 9] = value.xyz_to_rgb();
    let _: &[f32; 3] = value.illuminant_xyz(); let _: u8 = value.output_space();
    let _: bool = value.apply_cctf(); let _: bool = value.input_is_output_space();
    let _: color::Cctf = value.cctf(); let _: &[f32; 9] = value.dwg_to_output();
    let _: u64 = value.active_gamut_recipe_hash(); let _: u64 = value.hash();
    Ok(value)
}
