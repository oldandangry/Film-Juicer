use film_juicer_core::effects;
fn rejected(recipe: effects::Recipe, dust: effects::Dust, scratch: effects::Scratch) {
    let _ = effects::Recipe { ..recipe };
    let _ = effects::Dust { ..dust };
    let _ = effects::Scratch { ..scratch };
}
