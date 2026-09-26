pub struct bad_type;
pub trait bad_trait {}

pub enum CaptureMode {
    negative_direct,
}

pub mod badModule {}
pub fn renderFrame() {}

pub struct Frame {
    pub rowBytes: usize,
}

pub fn expose(cameraEV: f32) -> f32 {
    let filmDensity = cameraEV.exp2();
    filmDensity * 0.18
}

pub const kFilmFormat: f32 = 36.0;
pub static currentFrame: u32 = 0;

pub struct RGB;
pub struct RGBToXYZ;

pub enum Medium {
    CMY,
    PrintRGB,
}

struct CUDAContext;

pub fn context_size() -> usize {
    std::mem::size_of::<CUDAContext>()
}

pub struct Rgb;

impl Rgb {
    pub const kChannels: usize = 3;

    pub fn fromRGB() -> Self {
        Self
    }
}

#[allow(non_snake_case)]
pub fn unexplainedName() {}

#[expect(non_snake_case)]
pub fn unexplainedExpectedName() {}

#[expect(non_snake_case, reason = "deliberately stale exception")]
pub fn already_snake_case() {}

#[cfg(test)]
mod tests {
    #[test]
    fn testRenderFrame() {}
}
