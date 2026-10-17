use film_juicer_core::film::{dir::Recipe,density::{Minimum,Bounds}};
fn consume(recipe: Recipe, minimum: Minimum, bounds: Bounds) {
    let _ = Recipe { ..recipe }; let _ = Minimum { ..minimum }; let _ = Bounds { ..bounds };
}
