use film_juicer_core::film::Development;
fn mutate(development: &Development) { development.gamma_rgb()[0] = 1.0; development.normalized_curves_cmy()[0][0] = 0.0; }
