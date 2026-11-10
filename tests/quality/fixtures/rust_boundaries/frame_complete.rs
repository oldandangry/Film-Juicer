use film_juicer_core::frame::{Bounds, Domain, Geometry};
fn complete() -> Geometry {
    let domain = Domain::from_bounds(Bounds { x1: -2, y1: 3, x2: 7, y2: 3 }).unwrap();
    let geometry = Geometry::new(domain, 36.0);
    let _: Domain = geometry.domain();
    let _: f32 = geometry.pixel_size_um();
    geometry
}
