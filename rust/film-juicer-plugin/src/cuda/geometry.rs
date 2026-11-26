//! Frame geometry projected into the native continuation.

use std::mem::{align_of, size_of};

use film_juicer_core::frame::{Bounds, Domain, Geometry};

use crate::cuda::sys::FjExtent;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjFrameGeometry {
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

const _: () = {
    assert!(size_of::<FjFrameGeometry>() == 20 && align_of::<FjFrameGeometry>() == 4);
    assert!(std::mem::offset_of!(FjFrameGeometry, full_frame) == 0);
    assert!(std::mem::offset_of!(FjFrameGeometry, pixel_size_um) == 16);
};
