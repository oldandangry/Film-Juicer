use film_juicer_core::scanner;
fn mutate(lut: scanner::Lut) { lut.log2_xyz()[0] = 0.0; }
