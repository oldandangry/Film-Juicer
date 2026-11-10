use film_juicer_core::optics;
fn reject(params: optics::Params) {
    let recipe = optics::Recipe::new(params).unwrap();
    let _ = recipe.camera_diffusion;
    let _ = recipe.camera_lens_blur().sigma_um;
    let _ = recipe.scatter_halation().first_sigma_um;
    let _ = optics::Diffusion::inactive();
    let _ = optics::LensBlur::new(0.0);
    let _ = optics::Halation::new(params.scatter_halation);
}
