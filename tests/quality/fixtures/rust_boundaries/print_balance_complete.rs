use film_juicer_core::print;
fn prepare(base: &[f32;81], dye: &[[f32;3];81], sensitivity: &[[f32;3];81], light: &[f32;81]) -> print::Balance {
    let balance = print::Balance::new(print::BalanceParams { baseline_density_cmy: [0.0;3], compensated_density_cmy: [1.0,0.0,0.0], reference_hash: 0, normalization_mode: 0 }, base, dye, sensitivity, light).unwrap();
    let _: &[f32;3] = balance.baseline_raw_rgb(); let _: &[f32;3] = balance.compensated_raw_rgb();
    let _: [f32;3] = [balance.factor_midgray(), balance.factor_midgray_comp(), balance.normalizer()];
    let _: [u64;2] = [balance.filtered_main_illuminant_hash(), balance.hash()];
    balance
}
