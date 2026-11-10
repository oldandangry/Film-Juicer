use film_juicer_core::film::dir::boundary::{Axis, Boundary, Error, Input, Key};
fn complete(input: Input) -> Result<Boundary, Error> {
    let _: Key = input.key();
    let boundary = Boundary::new(input)?;
    let _: &[f64] = boundary.weights();
    let axis: &Axis = boundary.horizontal();
    let _: &[f64; 9] = axis.terminal();
    let _: (i32, i32) = (axis.length(), axis.terminal_size());
    let _: u64 = boundary.key().value();
    let _: (i32, i32) = (boundary.width(), boundary.height());
    Ok(boundary)
}
fn error(error: Error) {
    let _: (u32, u32, u32) = (error.reason() as u32, error.axis(), error.row());
    let _: (i32, f32, f64, f64) = (error.length(), error.sigma_px(), error.sum(), error.tolerance());
}
