use film_juicer_core::optics::GaussianKernel;
fn borrow<'a>() -> &'a [f32] { let kernel=GaussianKernel::new(3,1.0).unwrap(); kernel.weights() }
