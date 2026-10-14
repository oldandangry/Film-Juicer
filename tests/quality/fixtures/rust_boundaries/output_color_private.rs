use film_juicer_core::color::{OutputSpace, output_color};
fn consume() { let color = output_color(OutputSpace::Srgb); let _ = color.hash; }
