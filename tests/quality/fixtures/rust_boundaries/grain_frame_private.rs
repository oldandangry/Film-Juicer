use film_juicer_core::film::grain::frame::{Plan,Gaussian,Error};
fn private(plan:Plan, gaussian:Gaussian, error:Error) { let _ = plan.frame0; let _ = gaussian.sigma_px; let _ = error.field; }
