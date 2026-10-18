use film_juicer_core::optics;
fn reject(recipe: optics::Recipe, diffusion: optics::Diffusion, lens: optics::LensBlur, halation: optics::Halation) {
    *recipe.camera_diffusion() = diffusion;
    *recipe.camera_lens_blur() = lens;
    *recipe.scatter_halation() = halation;
}
