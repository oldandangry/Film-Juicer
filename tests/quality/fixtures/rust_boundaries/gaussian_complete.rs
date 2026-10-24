use film_juicer_core::optics::GaussianKernel;
fn complete() -> GaussianKernel {
    let kernel = GaussianKernel::new(2048, 512.0).unwrap();
    let _: u32 = kernel.radius();
    let _: f32 = kernel.sigma_px();
    let _: &[f32] = kernel.weights();
    kernel
}
