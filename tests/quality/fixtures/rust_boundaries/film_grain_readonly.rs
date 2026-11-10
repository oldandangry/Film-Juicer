use film_juicer_core::film::grain;
fn consume(recipe: &mut grain::Recipe) {
    recipe.particle_scale_cmy()[0] = 0.0;
    recipe.axis_block_prefix_max_cmy()[0][0] = 0.0;
}
