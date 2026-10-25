use film_juicer_core::optics::{HalationFrame, HalationFrameInput, HalationFrameError};
fn complete(input: HalationFrameInput, pixel_size_um: f32) -> Result<Option<HalationFrame>, HalationFrameError> {
    let frame = HalationFrame::new(input, pixel_size_um)?;
    if let Some(value) = &frame {
        let _: u64 = value.recipe_hash();
        let _: f32 = value.scatter_amount();
        for channel in value.channels() {
            let _: &[f32; 19] = channel.core().fir_weights();
            let _: &[f32; 4] = channel.bounce()[2].yvv();
            let _: i32 = channel.tail()[2].radius();
        }
    }
    Ok(frame)
}
