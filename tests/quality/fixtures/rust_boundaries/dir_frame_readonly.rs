use film_juicer_core::film::dir::frame::Plan;
fn mutate(plan: Plan) {
    plan.components()[0].coefficients()[0] = 0.0;
}
