use film_juicer_core::film::{dir::Recipe,density::{Minimum,Bounds}};
fn consume(recipe: &Recipe, minimum: &Minimum, bounds: &Bounds) {
    recipe.matrix_rgb()[0][0] = 0.0; minimum.cmy()[0] = 0.0; bounds.minimum_cmy()[0] = 0.0;
}
