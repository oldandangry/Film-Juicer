//! Describe-only authored properties; host operations retain their own status rules.

#![forbid(unsafe_code)]

use super::host::{Callback, Status};

pub(super) fn describe(callback: &Callback<'_>) -> Result<(), Status> {
    let properties = callback.descriptor_properties()?;
    properties.set_string(c"OfxPropLabel", 0, c"Juicer")?;
    // OFXS setLabels suppresses only these two optional property failures.
    let _ = properties.set_string(c"OfxPropShortLabel", 0, c"Juicer");
    let _ = properties.set_string(c"OfxPropLongLabel", 0, c"Juicer");
    properties.set_string(c"OfxImageEffectPluginPropGrouping", 0, c"Negative-juice")?;
    let index = properties.dimension(c"OfxImageEffectPropSupportedContexts")?;
    properties.set_string(
        c"OfxImageEffectPropSupportedContexts",
        index,
        c"OfxImageEffectContextFilter",
    )?;
    let index = properties.dimension(c"OfxImageEffectPropSupportedPixelDepths")?;
    properties.set_string(
        c"OfxImageEffectPropSupportedPixelDepths",
        index,
        c"OfxBitDepthFloat",
    )?;
    properties.set_int(c"OfxImageEffectPluginPropSingleInstance", 0, 0)?;
    properties.set_int(c"OfxImageEffectPluginPropHostFrameThreading", 0, 0)?;
    properties.set_int(c"OfxImageEffectPropSupportsMultiResolution", 0, 1)?;
    properties.set_int(c"OfxImageEffectPropSupportsTiles", 0, 0)?;
    properties.set_string(
        c"OfxImageEffectPluginRenderThreadSafety",
        0,
        c"OfxImageEffectRenderUnsafe",
    )?;
    properties.set_string(c"OfxImageEffectPropCudaRenderSupported", 0, c"true")?;
    properties.set_string(c"OfxImageEffectPropCudaStreamSupported", 0, c"true")
}
