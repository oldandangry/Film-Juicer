use film_juicer_core::film::dir::boundary::Boundary;
fn readonly(boundary: Boundary) {
    boundary.weights()[0] = 0.0;
    boundary.horizontal().terminal()[0] = 0.0;
}
