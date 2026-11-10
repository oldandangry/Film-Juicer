use film_juicer_core::illuminant::LensInput;
pub fn build_lens() -> LensInput { LensInput { blackbody: [1.0;81], kg3: [1.0;81] } }
