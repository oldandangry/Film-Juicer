use film_juicer_core::print;
fn prepare(params: print::FilterParams) -> print::Illuminant {
    let illuminant;
    {
        let mut source = [1.0; 81];
        illuminant = print::Illuminant::new(params, &source).unwrap();
        source.fill(0.0);
    }
    illuminant
}
