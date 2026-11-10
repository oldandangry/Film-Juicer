use film_juicer_core::effects;
fn rejected() {
    let _ = effects::Recipe::default();
    let _ = effects::Dust::default();
    let _ = effects::Scratch::default();
}
