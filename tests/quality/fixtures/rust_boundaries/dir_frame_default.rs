use film_juicer_core::film::dir::frame::{Component, Error, FailureTrace, Plan};
fn incomplete() {
    let _ = Plan::default();
    let _ = Component::default();
    let _ = FailureTrace::default();
    let _ = Error::default();
}
