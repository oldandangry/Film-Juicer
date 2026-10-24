use film_juicer_core::optics::GaussianKernel;
fn private() { let kernel=GaussianKernel::new(3,1.0).unwrap(); let _ = kernel.radius; let _ = kernel.weights; }
