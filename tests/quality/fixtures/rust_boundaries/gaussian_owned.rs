use film_juicer_core::optics::GaussianKernel;
fn owned() -> GaussianKernel { let kernel={ let radius=3; let sigma=1.0; GaussianKernel::new(radius,sigma).unwrap() }; let _later=GaussianKernel::new(75,18.75); kernel }
