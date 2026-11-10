// An exact foreign record spelling stays at the raw binding boundary.
#[expect(
    clippy::upper_case_acronyms,
    reason = "the raw OpenFX binding preserves the foreign OfxRGBAColourF name"
)]
#[repr(C)]
pub struct OfxRGBAColourF {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

// Handwritten Rust retains its own spelling through an explicit ABI mapping.
#[allow(
    unsafe_code,
    reason = "this isolated fixture declares the fixed OpenFX export name"
)]
#[unsafe(export_name = "OfxGetNumberOfPlugins")]
pub extern "C" fn plugin_count() -> i32 {
    1
}
