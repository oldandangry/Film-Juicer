use film_juicer_core::print;
fn reject(illuminant: print::Illuminant) {
    illuminant.samples()[0] = 0.0;
}
