use film_juicer_core::print;
fn prepare(base: &[f32;81], light: &[f32;81], sensitivity: &[[f32;3];81]) -> print::PreflashRaw {
    let raw = print::PreflashRaw::new(base, light, sensitivity).unwrap();
    let _: &[f32;3] = raw.cmy();
    raw
}
