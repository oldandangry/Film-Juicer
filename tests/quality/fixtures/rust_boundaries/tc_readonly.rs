use film_juicer_core::reconstruction::FilmTcLut;
fn rejected(lut: FilmTcLut) { lut.samples()[0] = 1.0; }
