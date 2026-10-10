use film_juicer_core::{exposure::Sensitivity, reconstruction::ReferenceWhite};
fn invalid(sensitivity: &mut Sensitivity, white: &mut ReferenceWhite) {
    sensitivity.values_rgb[0][0] = 0.0;
    white.samples[0] = 0.0;
}
