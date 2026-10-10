use film_juicer_core::{exposure::Sensitivity, reconstruction::ReferenceWhite};
fn invalid() {
    let _ = Sensitivity::default();
    let _ = ReferenceWhite::default();
}
