use film_juicer_core::color::InputColor;
fn consume(color: &mut InputColor) { color.rgb_to_xyz()[0] = 1.0; }
