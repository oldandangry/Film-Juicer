use film_juicer_core::exposure::{MallettMidgray, MidgrayNormalization, ReferenceRaw, ReferenceSource};
fn invalid(midgray: MallettMidgray, normalization: MidgrayNormalization, source: ReferenceSource, raw: ReferenceRaw) {
    let _ = MallettMidgray { ..midgray };
    let _ = MidgrayNormalization { ..normalization };
    let _ = ReferenceSource { ..source };
    let _ = ReferenceRaw { ..raw };
}
