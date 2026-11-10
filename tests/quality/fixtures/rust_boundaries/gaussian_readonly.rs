use film_juicer_core::optics::GaussianKernel;
fn readonly() { let kernel=GaussianKernel::new(3,1.0).unwrap(); kernel.weights()[0] = 0.0; }
