use film_juicer_core::gamut::{OutputBoundary, OutputTransform};
fn escape<'a>(boundary: OutputBoundary) -> &'a [f32] { boundary.cmax() }
fn early_drop(boundary: OutputBoundary, transform: OutputTransform) {
    let table = boundary.cmax();
    drop(boundary);
    let _ = table[0];
    let matrix = transform.native_rgb_to_d65_xyz();
    drop(transform);
    let _ = matrix[0];
}
