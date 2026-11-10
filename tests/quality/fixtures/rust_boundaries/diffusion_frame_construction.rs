use film_juicer_core::optics::diffusion_frame::{Sample, Components, FrameSet};
fn incomplete(sample: Sample, components: Components, frame: FrameSet) {
    let _ = Sample { ..sample };
    let _ = Components { ..components };
    let _ = FrameSet { ..frame };
}
