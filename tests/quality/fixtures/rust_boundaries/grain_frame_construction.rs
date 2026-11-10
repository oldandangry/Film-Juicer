use film_juicer_core::film::grain::frame::{Plan,Gaussian,Error};
fn construct(plan:Plan, gaussian:Gaussian, error:Error) { let _ = Plan { ..plan }; let _ = Gaussian { ..gaussian }; let _ = Error { ..error }; }
