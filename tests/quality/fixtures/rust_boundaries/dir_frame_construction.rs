use film_juicer_core::film::dir::frame::{Component, FailureTrace, Plan};
fn incomplete(plan: Plan, component: Component, trace: FailureTrace) {
    let _ = Plan { ..plan };
    let _ = Component { ..component };
    let _ = FailureTrace { ..trace };
}
