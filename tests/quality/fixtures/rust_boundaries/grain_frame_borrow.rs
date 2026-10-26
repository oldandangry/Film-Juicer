use film_juicer_core::film::grain::frame::Plan;
fn escape<'a>(plan:Plan) -> &'a [f32;3] { plan.density_max_cmy() }
fn drop_while_borrowed(plan:Plan) { let borrowed = plan.correlation(); drop(plan); let _ = borrowed[0].radius(); }
