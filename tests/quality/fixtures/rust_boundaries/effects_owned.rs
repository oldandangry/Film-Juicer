use film_juicer_core::effects;
fn consume(mut params: effects::Params) -> effects::Recipe {
    let recipe = effects::Recipe::new(params).unwrap();
    params.film_dust_amount = 0.0;
    assert!(params.film_dust_amount == 0.0);
    recipe
}
