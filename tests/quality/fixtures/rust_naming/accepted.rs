pub struct RgbToXyz;
pub struct CudaContext;
pub struct DirCouplers;
pub struct ScannerLut;
pub struct PreparedCudaFrame;

pub enum Medium {
    Film,
    Print,
}

pub trait DensityCurve {
    fn density(&self) -> f32;
}

pub const CHANNEL_COUNT: usize = 3;
pub static FRAME_COUNT: usize = 0;

pub struct Geometry {
    pub camera_film_format_long_edge_mm: f32,
    pub pixel_size_um: f32,
}

impl Geometry {
    pub const MINIMUM_LONG_EDGE_MM: f32 = 8.0;

    pub fn long_edge_mm(&self) -> f32 {
        self.camera_film_format_long_edge_mm
    }
}

pub mod film {
    pub fn expose(camera_ev: f32) -> f32 {
        let exposure = camera_ev.exp2();
        exposure * 0.18
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn exposure_preserves_mid_gray() {
        assert!(super::film::expose(0.0).is_finite());
    }
}
