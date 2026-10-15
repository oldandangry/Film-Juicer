use film_juicer_core::reconstruction::FilmTcLut;
fn escaping(lut: FilmTcLut) -> &'static [f32] { lut.samples() }
fn early_release(lut: FilmTcLut) { let samples = lut.samples(); drop(lut); let _ = samples[0]; }
