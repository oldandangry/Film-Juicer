use film_juicer_core::optics::HalationFrame;
fn read(frame: HalationFrame) {
    let _ = frame.recipe_hash;
    let _ = frame.channels;
    let _ = frame.channels()[0].core().radius;
}
