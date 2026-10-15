use film_juicer_core::exposure::{MallettMidgray, ReferenceRaw};
fn invalid(midgray: MallettMidgray, raw: ReferenceRaw) {
    midgray.raw_midgray_bgr()[0] = 0.0;
    midgray.midgray_dwg_rgb()[0] = 0.0;
    raw.rgb()[0] = 0.0;
}
