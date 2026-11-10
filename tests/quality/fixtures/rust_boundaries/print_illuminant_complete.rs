use film_juicer_core::print;
fn consume(params: print::FilterParams, source: &[f32; 81]) -> print::Illuminant {
    let illuminant = print::Illuminant::new(params, source).unwrap();
    let _: &[f32; 81] = illuminant.samples();
    illuminant
}
