use film_juicer_core::optics::HalationFrame;
fn mutate(frame: HalationFrame) { frame.channels()[0].core().fir_weights()[0] = 0.0; }
