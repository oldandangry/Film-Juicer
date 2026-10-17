use film_juicer_core::film::{dir::Recipe,density::{Minimum,Bounds}};
fn consume(recipe: &Recipe, minimum: &Minimum, bounds: &Bounds) {
    let _ = recipe.compensated_axes_rgb; let _ = minimum.cmy; let _ = bounds.hash;
}
