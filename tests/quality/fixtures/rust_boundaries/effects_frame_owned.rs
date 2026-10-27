use film_juicer_core::effects::frame::{Frame, Input, Error};
fn owned(input: Input) -> Result<Option<Frame>, Error> {
    let frame = Frame::new(input);
    let _ = input;
    frame
}
fn fixed_thread_owned() {
    fn assert_owned<T: Send + Sync + 'static>() {}
    assert_owned::<Frame>();
}
