//! Frame geometry used to prepare the callback-local frame.

use film_juicer_core::frame::{Bounds, Domain, Geometry};

use crate::cuda::sys::FjExtent;

#[derive(Clone, Copy)]
pub(crate) struct FrameGeometry {
    pub(crate) full_frame: FjExtent,
    pub(crate) pixel_size_um: f32,
}

/// Convert OFX bounds to the frame domain.
pub(crate) fn checked_domain(bounds: Bounds) -> Result<Domain, ()> {
    Domain::from_bounds(bounds).map_err(|_| ())
}

pub(crate) fn checked_geometry(domain: Domain, film_format_long_edge_mm: f32) -> Geometry {
    Geometry::new(domain, film_format_long_edge_mm)
}
