use film_juicer_core::{exposure::Sensitivity, reconstruction::ReferenceWhite};
fn invalid(sensitivity: Sensitivity, white: ReferenceWhite) {
    let _ = Sensitivity { ..sensitivity };
    let _ = ReferenceWhite { ..white };
}
