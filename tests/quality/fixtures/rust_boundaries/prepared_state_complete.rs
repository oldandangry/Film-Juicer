use film_juicer_core::{assets::Assets, recipe::{self, Params, RenderState}};
fn prepare(assets: &Assets, params: &Params) -> Result<RenderState, recipe::PrepareError> {
    let state = recipe::prepare(assets, params)?;
    let _: &str = state.film_key();
    let _: &[f32; 9] = state.s_inverse();
    let _: &[f32; 81] = state.exposure_tables().illuminant();
    let _: u64 = state.scanner().lut().hash();
    match state.route() {
        recipe::PreparedRoute::Direct(direct) => { let _ = direct.correction().hash(); }
        recipe::PreparedRoute::Print(print) => { let _ = print.curves().totals(); }
    }
    fn transferable<T: Send + Sync>() {}
    transferable::<RenderState>();
    Ok(state)
}
