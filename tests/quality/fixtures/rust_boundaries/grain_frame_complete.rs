use film_juicer_core::film::grain::frame::{Input,Plan,Gaussian,Error};
fn complete(input: Input) -> Result<Option<Plan>,Error> {
    let curves = [[[1.0_f32];3];3];
    let result = Plan::new(input, |l,c| Ok(&curves[l][c]));
    if let Ok(Some(plan)) = &result {
        let _: &[[f32;3];3] = plan.density_min_layers();
        let gaussians: &[Gaussian;3] = plan.correlation();
        let _: f32 = gaussians[0].sigma_px();
        let _: i32 = gaussians[0].radius();
        let _: u64 = gaussians[0].hash();
        let _: &[[Gaussian;3];3] = plan.dye_cloud();
        let _: u64 = plan.seed_base();
    }
    result
}
