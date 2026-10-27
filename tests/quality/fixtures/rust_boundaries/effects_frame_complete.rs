use film_juicer_core::effects::frame::{Frame, Input, Error, Origin};
fn complete(input: Input) -> Result<Option<Frame>, Error> {
    let result = Frame::new(input);
    if let Ok(Some(frame)) = &result {
        let _: &[Origin; 4] = frame.origins();
        let _: &[f32; 4] = frame.weave();
        let _: u64 = frame.hash();
        let _: bool = frame.requires_full_frame();
    }
    result
}

