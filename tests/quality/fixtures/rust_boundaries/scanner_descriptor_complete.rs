use film_juicer_core::{scanner, route};
fn prepare(input: scanner::LutDescriptorInput<'_>) -> Result<scanner::LutDescriptor, scanner::LutDescriptorError> {
    let descriptor = scanner::describe_lut(input)?;
    let _: route::ScanRoute = descriptor.route();
    let _: u8 = descriptor.medium(); let _: u8 = descriptor.capture_polarity();
    let _: u32 = descriptor.resolution();
    let _: u64 = descriptor.density_bounds_hash(); let _: u64 = descriptor.channel_density_hash();
    let _: u64 = descriptor.base_density_hash(); let _: u64 = descriptor.viewing_illuminant_hash();
    let _: u64 = descriptor.observer_hash(); let _: u64 = descriptor.hash();
    Ok(descriptor)
}
