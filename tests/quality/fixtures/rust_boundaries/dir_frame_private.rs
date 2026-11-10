use film_juicer_core::film::dir::frame::{Component, Error, Plan};
fn private(plan: Plan, component: Component, error: Error) {
    let _ = plan.progress;
    let _ = component.sigma_px;
    let _ = error.trace;
}
