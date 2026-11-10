use film_juicer_core::film::dir::boundary::{Axis, Boundary, Error, Input, Key};
fn private(input: Input, boundary: Boundary, axis: Axis, error: Error, key: Key) {
    let _ = input.coefficients;
    let _ = boundary.weights;
    let _ = axis.terminal;
    let _ = error.reason;
    let _ = key.0;
}
