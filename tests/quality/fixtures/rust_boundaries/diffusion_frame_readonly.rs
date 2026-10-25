use film_juicer_core::optics::diffusion_frame::Components;
fn immutable(components: &Components) {
    components.blue_weights()[8] = 0.0;
}
