use film_juicer_core::film::{Development, dir, density};
fn consume(development: &Development, params: dir::Params) {
    let minimum = density::Minimum::new([0.07,0.08,0.12]).unwrap();
    let bounds = density::Bounds::new(development,&minimum).unwrap();
    let recipe = dir::Recipe::new(development,params).unwrap();
    let _ = (minimum.cmy(),bounds.route(),bounds.polarity(),bounds.source(),bounds.minimum_cmy(),bounds.maximum_cmy(),bounds.inv_span_cmy(),bounds.authored_minimum_cmy(),bounds.authored_maximum_cmy(),bounds.hash());
    let _ = (recipe.active(),recipe.polarity(),recipe.mode(),recipe.matrix_rgb(),recipe.diffusion_size_um(),recipe.diffusion_tail_um(),recipe.diffusion_tail_weight(),recipe.density_max_rgb(),recipe.density_ref_rgb(),recipe.donor_k_rgb(),recipe.receiver_c_ref_rgb(),recipe.receiver_kr_rgb(),recipe.compensated_axes_rgb(),recipe.compensated_axes_hash(),recipe.hash());
}
