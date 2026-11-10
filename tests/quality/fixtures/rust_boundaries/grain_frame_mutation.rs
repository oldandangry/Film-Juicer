use film_juicer_core::film::grain::frame::Plan;
fn mutate(plan:&Plan) { plan.density_max_cmy()[0] = 0.0; plan.density_min_layers()[0][0] = 0.0; }
