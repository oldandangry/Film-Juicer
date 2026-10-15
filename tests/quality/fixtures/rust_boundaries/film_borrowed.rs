use film_juicer_core::film::Development;
fn escape<'a>(development: Development) -> &'a [f32] { development.log_exposure() }
fn early_drop(development: Development) {
    let curves = development.normalized_curves_cmy();
    drop(development);
    let _ = curves[0][0];
}
