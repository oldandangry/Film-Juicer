use film_juicer_core::film::dir::frame::{Error, Plan};
fn consume(_: &Plan) {}
fn failed(error: Error) {
    consume(error.trace().unwrap());
}
