use film_juicer_core::scanner;
fn prepare() -> (scanner::Lut,scanner::Lut) {
    let owners;
    {
        let mut eps = [0.25f32]; let mut weights = [1.0f32]; let mut baseline = [0.0f32];
        let input = scanner::LutInput { resolution: 2, medium: 0, range_min_cmy: [0.0;3], inverse_span_cmy: [1.0;3], tables: scanner::LutTables { eps_c: &eps, eps_m: &eps, eps_y: &eps, weighted_x: &weights, weighted_y: &weights, weighted_z: &weights, baseline_min: Some(&baseline), sample_count: 1, inv_yn: 1.0 } };
        owners = (scanner::Lut::new(input).unwrap(),scanner::Lut::new(input).unwrap());
        eps.fill(0.0);weights.fill(0.0);baseline.fill(10.0);
    }
    owners
}
