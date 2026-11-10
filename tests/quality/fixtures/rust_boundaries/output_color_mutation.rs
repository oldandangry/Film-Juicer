use film_juicer_core::color::{OutputSpace, output_color};
fn consume() { let color = output_color(OutputSpace::Srgb); color.rgb_to_xyz()[0] = 1.0; }
