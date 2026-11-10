use film_juicer_core::film::dir::boundary::{Boundary, Input};
fn borrow<'a>(input: Input) -> &'a [f64] {
    let boundary = Boundary::new(input).unwrap();
    boundary.weights()
}
