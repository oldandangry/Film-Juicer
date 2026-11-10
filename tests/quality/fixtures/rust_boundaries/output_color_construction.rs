use film_juicer_core::color::{OutputColor, OutputSpace, output_color};
fn consume() { let _ = OutputColor { ..*output_color(OutputSpace::Srgb) }; }
