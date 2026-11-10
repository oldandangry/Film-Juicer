use film_juicer_core::film::dir::boundary::Boundary;
fn early_drop(boundary: Boundary) {
    let weights = boundary.weights();
    drop(boundary);
    let _: f64 = weights[0];
}
