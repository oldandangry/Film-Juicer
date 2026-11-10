use film_juicer_core::scanner;
fn prepare() -> (scanner::Color, scanner::Color) {
    let result;
    {
        let mut input = scanner::ColorInput { medium: 255, output_space: 255, apply_cctf: false, illuminant_xyz: [1.0; 3], illuminant_hash: 1, active_gamut_recipe_hash: 0 };
        let first = scanner::Color::new(input).unwrap();
        input.output_space = 1; input.illuminant_xyz = [0.0; 3];
        let second = scanner::Color::new(input).unwrap();
        result = (first, second);
    }
    result
}
