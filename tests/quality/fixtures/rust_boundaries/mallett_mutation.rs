use film_juicer_core::exposure::{MallettMidgray, MidgrayNormalization, ReferenceRaw, ReferenceSource};
fn invalid(mut midgray: MallettMidgray, mut normalization: MidgrayNormalization, mut source: ReferenceSource, mut raw: ReferenceRaw) {
    midgray.raw_midgray_bgr[0] = 0.0;
    normalization.raw_green = 0.0;
    source.value = 0.0;
    raw.rgb[0] = 0.0;
}
