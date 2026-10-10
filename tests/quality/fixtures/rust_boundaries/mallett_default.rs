use film_juicer_core::exposure::{MallettMidgray, MidgrayNormalization, ReferenceRaw, ReferenceSource};
fn invalid() {
    let _ = MallettMidgray::default();
    let _ = MidgrayNormalization::default();
    let _ = ReferenceSource::default();
    let _ = ReferenceRaw::default();
}
