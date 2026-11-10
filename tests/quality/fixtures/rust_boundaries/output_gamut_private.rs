use film_juicer_core::gamut::{OutputTransform, OutputBoundary};
fn mutate(transform: &mut OutputTransform, boundary: &mut OutputBoundary) {
    transform.hash = 1;
    boundary.cmax[0] = 0.0;
}
