use film_juicer_core::optics;
fn reject(recipe: optics::Recipe, diffusion: optics::Diffusion, lens: optics::LensBlur, halation: optics::Halation) {
    let _ = optics::Recipe { ..recipe };
    let _ = optics::Diffusion { ..diffusion };
    let _ = optics::LensBlur { ..lens };
    let _ = optics::Halation { ..halation };
}
