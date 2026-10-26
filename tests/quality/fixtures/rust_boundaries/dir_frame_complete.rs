use film_juicer_core::film::dir::frame::{Component, Error, Input, Operator, Plan};
use film_juicer_core::frame::Domain;
fn complete(input: Input) -> Result<Option<Plan>, Error> {
    let result = Plan::new(input);
    if let Ok(Some(plan)) = &result {
        let _: u64 = plan.recipe_hash();
        let _: Domain = plan.render_extent();
        let _: Domain = plan.full_frame_extent();
        let _: usize = plan.component_count();
        let components: &[Component; 4] = plan.components();
        let _: Operator = components[0].operator();
        let _: &[f64; 9] = components[0].coefficients();
    }
    if let Err(error) = &result {
        let _: usize = error.component();
        let _: &str = error.field().name();
        if let Some(trace) = error.trace() {
            let _: &[Component; 4] = trace.components();
            let _: usize = trace.component_count();
        }
    }
    result
}
