use film_juicer_core::film::{self, reference};
fn prepare(development: &film::Development, reconstruction: Option<reference::Reconstruction<'_>>) -> Result<reference::Reference, reference::Error> {
    let value = reference::Reference::new(development, reference::Params { manual_exposure_ev: 0.0, auto_exposure_enabled: true }, reconstruction)?;
    let _: &[f32; 3] = value.baseline_raw_rgb(); let _: &[f32; 3] = value.compensated_raw_rgb();
    let _: &[f32; 3] = value.baseline_density_cmy(); let _: &[f32; 3] = value.compensated_density_cmy();
    let _: f32 = value.meter_scale(); let _: u64 = value.hash();
    Ok(value)
}
