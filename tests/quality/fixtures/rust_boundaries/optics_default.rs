use film_juicer_core::optics;
fn reject() {
    let _ = optics::Recipe::default();
    let _ = optics::Diffusion::default();
    let _ = optics::LensBlur::default();
    let _ = optics::Halation::default();
}
