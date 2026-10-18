use film_juicer_core::film::grain;
fn consume(recipe: grain::Recipe) -> grain::Recipe { grain::Recipe { ..recipe } }
