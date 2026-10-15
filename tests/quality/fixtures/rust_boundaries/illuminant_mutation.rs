use film_juicer_core::illuminant::LensInput;
pub fn mutate_lens(mut input: LensInput) { input.kg3[0] = 1.0; }
