use film_juicer_core::optics::diffusion_frame::{Sample, Components, FrameSet};
fn private(sample: Sample, components: Components, frame: FrameSet) {
    let _ = sample.fields;
    let _ = components.lambda_px;
    let _ = frame.hash;
    let _ = frame.camera;
}
