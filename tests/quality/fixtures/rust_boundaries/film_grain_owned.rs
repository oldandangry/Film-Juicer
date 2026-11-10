use film_juicer_core::film::grain;
fn consume(development: film_juicer_core::film::Development, params: grain::Params) -> grain::Recipe {
    let recipe = grain::Recipe::new(&development, params).unwrap();
    drop(development);
    recipe
}
