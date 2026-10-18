use film_juicer_core::optics;
fn prepare(mut params: optics::Params) -> optics::Recipe {
    let recipe;
    {
        let first_sigma_um = [65.0; 3];
        let primary_amount = [0.08, 0.02, 0.0];
        params.scatter_halation.first_sigma_um = first_sigma_um;
        params.scatter_halation.primary_amount = primary_amount;
        recipe = optics::Recipe::new(params).unwrap();
    }
    recipe
}
