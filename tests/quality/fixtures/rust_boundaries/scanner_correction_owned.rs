use film_juicer_core::scanner;
fn prepare() -> (scanner::Correction, scanner::Correction) {
    let corrections;
    {
        let mut base = [0.0;81]; let mut dyes = [[1.0/3.0;3];81];
        let mut sensitivity = [[1.0/81.0;3];81]; let mut light = [1.0;81];
        let mut eps = [1.0/3.0]; let mut ay = [1.0]; let mut axis = [-2.0,0.0,2.0];
        let mut authored = [[3.0;3],[1.5;3],[0.0;3]]; let mut curves = [[0.0;3],[1.5;3],[3.0;3]];
        let params = scanner::Params { route: 2, black_correction: true, white_correction: true, black_level: 0.0, white_level: 1.0 };
        let tables = scanner::ReferenceTables { eps_c: &eps, eps_m: &eps, eps_y: &eps, weighted_y: &ay, baseline_min: None, sample_count: 1, inv_yn: 1.0 };
        let direct = scanner::Correction::direct(params, Some(scanner::DirectInput {
            tables, film_base_density: &base, film_log_exposure: &axis, film_authored_density_curves: &authored,
            minimum_density_cmy: [0.0;3], maximum_density_cmy: [3.0;3], inverse_span_cmy: [1.0/3.0;3], baseline_density_cmy: [1.0;3], reference_hash: 1, medium: 0,
        })).unwrap();
        let print = scanner::Correction::print(scanner::Params { route: 1, ..params }, Some(scanner::PrintInput {
            tables, film_base_density: &base, film_channel_density_cmy: &dyes, print_sensitivity_cmy: &sensitivity,
            filtered_main_illuminant: &light, print_base_density: &base, print_log_exposure: &axis, print_density_curves: &curves,
            minimum_density_cmy: [0.0;3], inverse_span_cmy: [1.0/3.0;3], film_minimum_density_cmy: [0.0;3], film_maximum_density_cmy: [3.0;3],
            preflash_raw_cmy: [0.0;3], normalizer: 1.0, preflash_exposure: 0.0, medium: 1,
        })).unwrap();
        corrections = (direct, print);
        base.fill(1.0); dyes.fill([0.0;3]); sensitivity.fill([0.0;3]); light.fill(0.0);
        eps.fill(0.0); ay.fill(0.0); axis.fill(0.0); authored.fill([0.0;3]); curves.fill([0.0;3]);
    }
    corrections
}
