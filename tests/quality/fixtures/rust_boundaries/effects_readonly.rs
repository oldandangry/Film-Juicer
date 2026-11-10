use film_juicer_core::effects;
fn rejected(recipe: &effects::Recipe, dust: effects::Dust, scratch: effects::Scratch) {
    *recipe.film_dust() = dust;
    *recipe.gate_scratch() = scratch;
}
