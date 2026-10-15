use film_juicer_core::{color, exposure};
fn complete() -> (exposure::MallettMidgray, exposure::MidgrayNormalization, exposure::ReferenceSource, exposure::ReferenceRaw) {
    let basis = [[1.0; 3]; 81];
    let illuminant = [1.0; 81];
    let sensitivity = [[1.0; 3]; 81];
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let midgray = exposure::mallett_midgray(exposure::MallettInput {
        color: color::InputConversion { space: color::InputSpace::SrgbRec709, decode_cctf: false, rgb_to_xyz: identity, xyz_adaptation: None },
        xyz_to_linear_srgb: identity, basis_rgb: &basis, illuminant: &illuminant, sensitivity_rgb: &sensitivity,
    });
    let source = exposure::reference_source(0.0).unwrap();
    let raw = exposure::mallett_reference_raw(exposure::ReferenceInput {
        basis_rgb: &basis, illuminant: &illuminant, sensitivity_rgb: &sensitivity,
        source: source.value(), green_scale: 1.0,
    }).unwrap();
    (midgray, exposure::tc_midgray(1.0), source, raw)
}
fn consume_after_sources_expire() {
    let (midgray, normalization, source, raw) = complete();
    let _ = (midgray.midgray_dwg_rgb(), midgray.raw_midgray_bgr(), midgray.normalization().scale(), normalization.raw_green(), source.value(), raw.rgb());
}
