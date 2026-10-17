use film_juicer_core::color::{InputColor, InputSpace, input_matrices};
fn consume() { let _ = InputColor::from_matrices(InputSpace::DavinciWideGamut, false, input_matrices(InputSpace::DavinciWideGamut), [0.0;9], [0.0;3]); }
