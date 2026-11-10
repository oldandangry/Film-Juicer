use film_juicer_core::optics::diffusion_frame::{SampleInput, SampleFields, Sample, Components, FrameInput, FrameSet, Error};
fn complete(sample: SampleInput, frame: FrameInput) -> Result<Option<FrameSet>, Error> {
    let sample = Sample::new(sample)?;
    let fields: SampleFields = (&sample).into();
    let components = Components::new(fields)?;
    let _: &[f64; 9] = components.lambda_px();
    let _: &[f64; 9] = components.red_weights();
    let _: &[f64; 9] = components.green_weights();
    let _: &[f64; 9] = components.blue_weights();
    let frame = FrameSet::new(frame)?;
    if let Some(f) = &frame {
        let _: [i32; 4] = f.full_frame();
        let _: u64 = f.hash();
        if let Some(stage) = f.camera().or(f.enlarger()) {
            let _: u32 = stage.stage();
            let _: f64 = stage.scatter_fraction();
            let _: SampleFields = stage.sample().fields();
        }
    }
    Ok(frame)
}
