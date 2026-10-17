use film_juicer_core::film::grain;
fn consume(recipe: &grain::Recipe) {
    let _ = recipe.particle_density_min_cmy;
    let _ = recipe.axis_block_prefix_max_cmy;
}
