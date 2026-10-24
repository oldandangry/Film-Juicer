use film_juicer_core::{scanner, route};
fn prepare() -> scanner::LutDescriptor {
    let result;
    {
        let mut channel = [[0.0; 3]; 81];
        let mut base = [0.0; 81];
        let mut viewing = vec![68, 53, 48];
        result = scanner::describe_lut(scanner::LutDescriptorInput::Direct {
            params: scanner::LutDescriptorParams { route: route::ScanRoute::NegativeDirect,
                capture_polarity: 255, bounds_medium: 0, density_bounds_hash: 1,
                viewing_illuminant: &viewing, observer_identity: b"observer", resolution: 0 },
            film: Some(scanner::LutSpectra { channel_density_cmy: &channel, base_density: &base }) }).unwrap();
        channel.fill([f32::NAN; 3]); base.fill(f32::NAN); viewing.clear();
    }
    result
}
