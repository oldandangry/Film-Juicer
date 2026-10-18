use film_juicer_core::effects;
fn rejected(recipe: &effects::Recipe, dust: &effects::Dust, scratch: &effects::Scratch) {
    let _ = recipe.film_dust;
    let _ = dust.cell_width_mm;
    let _ = scratch.length_min_mm;
    let _ = effects::Recipe::zero();
    let _ = effects::Dust::new(1.0, 0.80, 0.00025, 0.003);
    let _ = effects::Scratch::new(1.0, 0.080, 0.95, 0.00015, 0.0015);
}
