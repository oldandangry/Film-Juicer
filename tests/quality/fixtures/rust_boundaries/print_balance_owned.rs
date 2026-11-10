use film_juicer_core::print;
fn prepare() -> print::Balance {
    let balance;
    {
        let mut base = [0.0;81]; let mut dye = [[1.0,0.0,0.0];81];
        let mut sensitivity = [[1.0;3];81]; let mut light = [1.0;81];
        balance = print::Balance::new(print::BalanceParams { baseline_density_cmy: [0.0;3], compensated_density_cmy: [1.0,0.0,0.0], reference_hash: 0, normalization_mode: 0 }, &base, &dye, &sensitivity, &light).unwrap();
        base.fill(1.0); dye.fill([0.0;3]); sensitivity.fill([0.0;3]); light.fill(0.0);
    }
    balance
}
