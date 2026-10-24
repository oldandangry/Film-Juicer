use film_juicer_core::optics::GaussianKernel;
fn construction() { let kernel = GaussianKernel::new(3,1.0).unwrap(); let _ = GaussianKernel { ..kernel }; }
