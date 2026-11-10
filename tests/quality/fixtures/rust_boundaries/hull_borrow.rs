use film_juicer_core::gamut::{Hull, HullInput};
fn escaped() -> HullInput<'static> {
    let samples = [0.0; 81];
    HullInput { x: &samples, y: &samples, z: &samples, d65: &samples }
}
fn expired(hull: Hull) {
    let xy = hull.xy();
    drop(hull);
    let _ = xy[0];
}
