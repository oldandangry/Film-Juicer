use film_juicer_core::optics;
fn consume(params: optics::Params) -> optics::Recipe {
    let recipe = optics::Recipe::new(params).unwrap();
    for diffusion in [recipe.camera_diffusion(), recipe.enlarger_diffusion()] {
        let _ = diffusion.domain();
        let _ = diffusion.active();
        let _ = diffusion.family();
        let _ = diffusion.scatter_fraction();
        let _ = diffusion.group_weights();
        let _ = diffusion.group_lambda_um();
        let _ = diffusion.warmth();
        let _ = diffusion.spatial_scale();
        let _ = diffusion.resolved_hash();
        let _ = diffusion.hash();
    }
    let _ = recipe.camera_lens_blur().sigma_um();
    let _ = recipe.camera_lens_blur().hash();
    let halation = recipe.scatter_halation();
    let _ = halation.scatter_active();
    let _ = halation.back_reflection_active();
    let _ = halation.scatter_amount();
    let _ = halation.scatter_spatial_scale();
    let _ = halation.halation_spatial_scale();
    let _ = halation.first_sigma_um();
    let _ = halation.total_strength();
    let _ = halation.hash();
    let _ = recipe.hash();
    recipe
}
