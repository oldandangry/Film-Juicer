use film_juicer_core::film::dir::boundary::{Axis, Boundary, Error, Input, Key};
fn defaults() {
    let _ = Input::default();
    let _ = Boundary::default();
    let _ = Axis::default();
    let _ = Error::default();
    let _ = Key::default();
}
