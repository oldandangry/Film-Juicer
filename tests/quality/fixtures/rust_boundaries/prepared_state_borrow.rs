use film_juicer_core::recipe::RenderState;
fn detach(state: RenderState) -> &'static [f32; 9] { state.s_inverse() }
fn move_while_borrowed(state: RenderState) {
    let borrowed = state.s_inverse();
    drop(state);
    let _ = borrowed[0];
}
