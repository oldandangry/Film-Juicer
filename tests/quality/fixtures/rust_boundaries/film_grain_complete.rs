use film_juicer_core::film::{Development, grain};
fn consume(development: &Development, params: grain::Params) -> grain::Recipe {
    let recipe = grain::Recipe::new(development, params).unwrap();
    let _ = recipe.active();
    let _ = recipe.sublayers_active();
    let _ = recipe.axis_finite_cmy();
    let _ = recipe.axis_block_prefix_max_cmy();
    let _ = recipe.particle_area_um2();
    let _ = recipe.particle_scale_cmy();
    let _ = recipe.particle_scale_layers();
    let _ = recipe.particle_density_min_cmy();
    let _ = recipe.uniformity_cmy();
    let _ = recipe.correlation_sigma_px();
    let _ = recipe.dye_cloud_blur_um();
    let _ = recipe.micro_structure();
    let _ = recipe.sublayer_count();
    let _ = recipe.amplitude();
    let _ = recipe.chroma_mix();
    let _ = recipe.chroma_shared_weight();
    let _ = recipe.chroma_independent_weight();
    let _ = recipe.fine_weight();
    let _ = recipe.mid_weight();
    let _ = recipe.coarse_weight();
    let _ = recipe.size_mix_scale();
    let _ = recipe.clump_temporal_mix();
    let _ = recipe.clump_morph_period_sec();
    let _ = recipe.debug_view();
    let _ = recipe.layers_hash();
    let _ = recipe.hash();
    recipe
}
