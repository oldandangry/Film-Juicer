//! Filter-context clips and parameter descriptors.

#![forbid(unsafe_code)]

use std::ffi::CStr;

use film_juicer_core::optics::{HALATION_MAX_AUTHORED, HALATION_MAX_BOUNCES, HalationShape};
use film_juicer_core::optics::highlight_boost;

use super::host::{Callback, Parameters, Properties, Status};

fn append_index(properties: &Properties<'_>, property: &CStr) -> Result<i32, Status> {
    let index = properties.dimension(property)?;
    if index < 0 {
        Err(Status::BAD_INDEX)
    } else {
        Ok(index)
    }
}

fn append(properties: &Properties<'_>, property: &CStr, value: &CStr) -> Result<(), Status> {
    let index = append_index(properties, property)?;
    properties.set_string(property, index, value)
}

fn option(properties: &Properties<'_>, value: &CStr) -> Result<(), Status> {
    append(properties, c"OfxParamPropChoiceOption", value)
}

fn catalog_option(properties: &Properties<'_>, key: &str, label: &str) -> Result<(), Status> {
    let index = append_index(properties, c"OfxParamPropChoiceOption")?;
    let enum_index = append_index(properties, c"OfxParamPropChoiceEnum")?;
    if index != enum_index {
        return Err(Status::FAILED);
    }
    properties.set_catalog_string(c"OfxParamPropChoiceEnum", index, key)?;
    properties.set_catalog_string(c"OfxParamPropChoiceOption", index, label)
}

fn string_option(properties: &Properties<'_>, key: &CStr, label: &CStr) -> Result<(), Status> {
    let index = append_index(properties, c"OfxParamPropChoiceOption")?;
    let enum_index = append_index(properties, c"OfxParamPropChoiceEnum")?;
    if index != enum_index {
        return Err(Status::FAILED);
    }
    properties.set_string(c"OfxParamPropChoiceEnum", index, key)?;
    properties.set_string(c"OfxParamPropChoiceOption", index, label)
}

fn describe_advanced_halation(parameters: &Parameters<'_>) -> Result<(), Status> {
    let defaults = HalationShape::default();
    let group = c"HalationAdvancedGroup";
    let p = parameters.define(c"OfxParamTypeGroup", group)?;
    p.set_string(c"OfxPropLabel", 0, c"Advanced Halation")?;
    let _ = p.set_int(c"OfxParamPropGroupOpen", 0, 0);
    p.set_string(c"OfxParamPropParent", 0, c"HalationGroup")?;

    for (name, label, hint, values, maximum, display_max, increment) in [
        (
            c"HalationScatterCoreSigmaUm",
            c"Scatter Core σ (µm)",
            c"Gaussian sigma of fine in-emulsion scatter for each RGB channel.",
            defaults.scatter_core_um,
            HALATION_MAX_AUTHORED,
            10.0,
            0.5,
        ),
        (
            c"HalationScatterTailScaleUm",
            c"Scatter Tail Scale (µm)",
            c"Exponential decay length of the broad in-emulsion scatter tail for each RGB channel.",
            defaults.scatter_tail_um,
            HALATION_MAX_AUTHORED,
            40.0,
            1.0,
        ),
        (
            c"HalationScatterTailWeight",
            c"Scatter Tail Weight",
            c"Fraction of scattered light in the broad tail for each RGB channel. The core receives the remaining fraction.",
            defaults.scatter_tail_weight,
            1.0,
            1.0,
            0.01,
        ),
    ] {
        let p = parameters.define(c"OfxParamTypeDouble3D", name)?;
        describe_halation_rgb(&p, label, hint, values, maximum, display_max, increment)?;
    }

    let p = parameters.define(c"OfxParamTypeChoice", c"HalationReflectionSource")?;
    p.set_string(c"OfxPropLabel", 0, c"Reflection Source")?;
    let _ = p.set_string(c"OfxParamPropHint", 0, c"Profile follows the selected stock. Custom uses saved absolute RGB strength and width, initially set to the schema defaults. Switching to Custom can change the result immediately.");
    option(&p, c"Profile")?;
    option(&p, c"Custom")?;
    p.set_int(c"OfxParamPropDefault", 0, 0)?;
    p.set_string(c"OfxParamPropParent", 0, group)?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;

    let p = parameters.define(c"OfxParamTypeDouble3D", c"HalationStrengthRgb")?;
    describe_halation_rgb(
        &p,
        c"Halation Strength",
        c"Custom total back-reflection strength relative to incident exposure for each RGB channel, before the Halation amount multiplier.",
        [0.05, 0.015, 0.0],
        HALATION_MAX_AUTHORED,
        0.5,
        0.005,
    )?;
    let p = parameters.define(c"OfxParamTypeDouble3D", c"HalationFirstBounceSigmaUm")?;
    describe_halation_rgb(
        &p,
        c"First-Bounce σ (µm)",
        c"Custom first-bounce Gaussian sigma for each RGB channel. Later bounces use widths proportional to the square root of the bounce number.",
        [65.0; 3],
        HALATION_MAX_AUTHORED,
        150.0,
        1.0,
    )?;

    let p = parameters.define(c"OfxParamTypeInteger", c"HalationBounceCount")?;
    p.set_string(c"OfxPropLabel", 0, c"Bounces")?;
    let _ = p.set_string(c"OfxParamPropHint", 0, c"Number of Gaussian bounces used for back-reflection. Advanced Gaussian widths can be at most 1024 pixels at the rendered frame size.");
    p.set_int(c"OfxParamPropDefault", 0, defaults.bounce_count)?;
    p.set_int(c"OfxParamPropMin", 0, 1)?;
    p.set_int(c"OfxParamPropMax", 0, HALATION_MAX_BOUNCES as i32)?;
    p.set_int(c"OfxParamPropDisplayMin", 0, 1)?;
    p.set_int(c"OfxParamPropDisplayMax", 0, HALATION_MAX_BOUNCES as i32)?;
    p.set_string(c"OfxParamPropParent", 0, group)?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;

    let p = parameters.define(c"OfxParamTypeDouble", c"HalationBounceDecay")?;
    p.set_string(c"OfxPropLabel", 0, c"Bounce Decay")?;
    let _ = p.set_string(c"OfxParamPropHint", 0, c"Amplitude ratio between consecutive back-reflection bounces. Zero uses only the first bounce. One weights all bounces equally.");
    p.set_double(c"OfxParamPropDefault", 0, defaults.bounce_decay)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 1.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 1.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.05)?;
    p.set_string(c"OfxParamPropParent", 0, group)?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;

    let p = parameters.define(c"OfxParamTypeBoolean", c"HalationRenormalize")?;
    p.set_string(c"OfxPropLabel", 0, c"Renormalize")?;
    let _ = p.set_string(c"OfxParamPropHint", 0, c"Preserve uniform exposure by dividing by one plus the reflection strength in each channel. Turn off for additive reflection.");
    p.set_int(c"OfxParamPropDefault", 0, i32::from(defaults.renormalize))?;
    p.set_string(c"OfxParamPropParent", 0, group)?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;

    let highlight_group = c"HalationHighlightGroup";
    let p = parameters.define(c"OfxParamTypeGroup", highlight_group)?;
    p.set_string(c"OfxPropLabel", 0, c"Highlight Reconstruction")?;
    p.set_string(c"OfxParamPropParent", 0, group)?;
    let boost = highlight_boost::Controls::default();
    for (name, label, hint, default, maximum, display_max, increment) in [
        (
            c"HalationHighlightBoostEv",
            c"Highlight Boost (EV)",
            c"Maximum highlight boost in stops. Uses the brightest film-linear sample in the rendered frame before camera diffusion.",
            boost.boost_ev,
            highlight_boost::MAX_EV,
            8.0,
            0.5,
        ),
        (
            c"HalationHighlightProtectionEv",
            c"Highlight Protection (EV)",
            c"Sets the onset of reconstruction in stops above a film-linear exposure of 0.184.",
            boost.protection_ev,
            highlight_boost::MAX_EV,
            10.0,
            0.5,
        ),
        (
            c"HalationHighlightBoostRange",
            c"Highlight Boost Range",
            c"Shape of the highlight response. Zero concentrates the boost near the peak. One spreads it more broadly above the onset.",
            boost.range,
            1.0,
            1.0,
            0.05,
        ),
    ] {
        let p = parameters.define(c"OfxParamTypeDouble", name)?;
        p.set_string(c"OfxPropLabel", 0, label)?;
        let _ = p.set_string(c"OfxParamPropHint", 0, hint);
        p.set_double(c"OfxParamPropDefault", 0, default)?;
        p.set_double(c"OfxParamPropMin", 0, 0.0)?;
        p.set_double(c"OfxParamPropMax", 0, maximum)?;
        p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
        p.set_double(c"OfxParamPropDisplayMax", 0, display_max)?;
        p.set_double(c"OfxParamPropIncrement", 0, increment)?;
        p.set_string(c"OfxParamPropParent", 0, highlight_group)?;
        p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    }
    Ok(())
}

fn describe_halation_rgb(
    p: &Properties<'_>,
    label: &CStr,
    hint: &CStr,
    values: [f64; 3],
    maximum: f64,
    display_max: f64,
    increment: f64,
) -> Result<(), Status> {
    p.set_string(c"OfxPropLabel", 0, label)?;
    let _ = p.set_string(c"OfxParamPropHint", 0, hint);
    for (index, value, dimension) in [
        (0, values[0], c"R"),
        (1, values[1], c"G"),
        (2, values[2], c"B"),
    ] {
        p.set_double(c"OfxParamPropDefault", index, value)?;
        p.set_double(c"OfxParamPropMin", index, 0.0)?;
        p.set_double(c"OfxParamPropMax", index, maximum)?;
        p.set_double(c"OfxParamPropDisplayMin", index, 0.0)?;
        p.set_double(c"OfxParamPropDisplayMax", index, display_max)?;
        p.set_string(c"OfxParamPropDimensionLabel", index, dimension)?;
    }
    p.set_double(c"OfxParamPropIncrement", 0, increment)?;
    p.set_string(c"OfxParamPropParent", 0, c"HalationAdvancedGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)
}

pub(super) fn describe(callback: &Callback<'_>, parameters: &Parameters<'_>) -> Result<(), Status> {
    let catalog = callback.catalog()?;
    let source = callback.define_clip(c"Source")?;
    append(
        &source,
        c"OfxImageEffectPropSupportedComponents",
        c"OfxImageComponentRGBA",
    )?;
    append(
        &source,
        c"OfxImageEffectPropSupportedComponents",
        c"OfxImageComponentRGB",
    )?;
    source.set_int(c"OfxImageEffectPropSupportsTiles", 0, 0)?;
    source.set_int(c"OfxImageClipPropOptional", 0, 0)?;
    let output = callback.define_clip(c"Output")?;
    append(
        &output,
        c"OfxImageEffectPropSupportedComponents",
        c"OfxImageComponentRGBA",
    )?;
    append(
        &output,
        c"OfxImageEffectPropSupportedComponents",
        c"OfxImageComponentRGB",
    )?;
    output.set_int(c"OfxImageEffectPropSupportsTiles", 0, 0)?;

    let p = parameters.define(c"OfxParamTypeGroup", c"InputGroup")?;
    p.set_string(c"OfxPropLabel", 0, c"Input")?;

    let p = parameters.define(c"OfxParamTypeDouble", c"Exposure")?;
    p.set_string(c"OfxPropLabel", 0, c"Exposure Compensation Ev")?;
    let _ = p.set_string(c"OfxParamPropHint", 0, c"Camera exposure compensation in EV stops. Per agx-emulsion parity: this is camera.exposure_compensation_ev, applied as 2^EV multiplication to film exposure. Positive values brighten the image.");
    p.set_double(c"OfxParamPropDefault", 0, 0.0)?;
    p.set_double(c"OfxParamPropMin", 0, -8.0)?;
    p.set_double(c"OfxParamPropMax", 0, 8.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, -4.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 4.0)?;
    p.set_string(c"OfxParamPropParent", 0, c"InputGroup")?;

    let p = parameters.define(c"OfxParamTypeChoice", c"CameraMeteringMethod")?;
    p.set_string(c"OfxPropLabel", 0, c"Camera metering")?;
    for label in [
        c"Center-weighted",
        c"Average",
        c"Median",
        c"Partial",
        c"Matrix",
        c"Multi-zone",
        c"Highlight-weighted",
    ] {
        option(&p, label)?;
    }
    p.set_int(c"OfxParamPropDefault", 0, 0)?;
    p.set_string(c"OfxParamPropParent", 0, c"InputGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;

    let p = parameters.define(c"OfxParamTypeBoolean", c"CameraAutoExposure")?;
    p.set_string(c"OfxPropLabel", 0, c"Camera auto exposure")?;
    let _ = p.set_string(c"OfxParamPropHint", 0, c"Enable the camera auto meter (agx-emulsion camera.auto_exposure). Scanner auto exposure remains independent.");
    p.set_int(c"OfxParamPropDefault", 0, 1)?;
    p.set_string(c"OfxParamPropParent", 0, c"InputGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;

    let p = parameters.define(c"OfxParamTypeChoice", c"CameraFilmFormatPreset")?;
    p.set_string(c"OfxPropLabel", 0, c"Camera film format preset")?;
    for label in [
        c"Custom",
        c"APS-C film (25 mm)",
        c"35 mm / 135 (36 mm)",
        c"645 / 6x4.5 (56 mm)",
        c"6x7 (70 mm)",
        c"4x5 (120 mm)",
        c"Super 16 (11.66 mm)",
        c"35mm Academy (21.95 mm)",
        c"Super 35 (24.89 mm)",
        c"VistaVision 8-perf (37.72 mm)",
        c"IMAX/65mm (70.4 mm)",
    ] {
        option(&p, label)?;
    }
    p.set_int(c"OfxParamPropDefault", 0, 0)?;
    let _ = p.set_string(c"OfxParamPropHint", 0, c"Choose a film format preset; the selected long edge is written to the editable control below.");
    p.set_string(c"OfxParamPropParent", 0, c"InputGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;

    let p = parameters.define(c"OfxParamTypeDouble", c"CameraFilmFormatMm")?;
    p.set_string(c"OfxPropLabel", 0, c"Camera film format (mm)")?;
    p.set_double(c"OfxParamPropDefault", 0, 35.0)?;
    p.set_double(c"OfxParamPropMin", 0, 8.0)?;
    p.set_double(c"OfxParamPropMax", 0, 120.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 8.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 120.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 1.0)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Longest capture dimension in millimeters; used for micrometer-to-pixel conversions.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"InputGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;

    let catalog = catalog.ok_or(Status::FATAL)?;
    let p = parameters.define(c"OfxParamTypeStrChoice", c"FilmProfileKey")?;
    p.set_string(c"OfxPropLabel", 0, c"Film stock")?;
    for entry in catalog.films() {
        catalog_option(&p, entry.key(), entry.label())?;
    }
    p.set_string(c"OfxParamPropDefault", 0, c"kodak_portra_400")?;
    p.set_string(c"OfxParamPropParent", 0, c"InputGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeChoice", c"SpectralUpsampling")?;
    p.set_string(c"OfxPropLabel", 0, c"Spectral upsampling")?;
    option(&p, c"Hanatos")?;
    option(&p, c"Mallett")?;
    option(&p, c"Arctic")?;
    let _ = p.set_string(c"OfxParamPropHint", 0, c"Choose the spectral reconstruction method used for film exposure. Hanatos uses the Hanatos 2025 LUT when available; Mallett uses the Mallett 2019 sRGB basis reconstruction; Arctic uses the Arctic 2026 beta04 reflectance LUT.");
    p.set_int(c"OfxParamPropDefault", 0, 0)?;
    p.set_string(c"OfxParamPropParent", 0, c"InputGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeChoice", c"InputColorSpace")?;
    p.set_string(c"OfxPropLabel", 0, c"Input color space")?;
    option(&p, c"DaVinci Wide Gamut")?;
    option(&p, c"ITU-R BT.2020")?;
    option(&p, c"ACES2065-1")?;
    option(&p, c"sRGB / Rec.709")?;
    p.set_int(c"OfxParamPropDefault", 0, 0)?;
    p.set_string(c"OfxParamPropParent", 0, c"InputGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeBoolean", c"InputCctfDecoding")?;
    p.set_string(c"OfxPropLabel", 0, c"Decode input CCTF")?;
    p.set_int(c"OfxParamPropDefault", 0, 0)?;
    p.set_string(c"OfxParamPropParent", 0, c"InputGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeGroup", c"InputAdvancedGroup")?;
    p.set_string(c"OfxPropLabel", 0, c"Advanced")?;
    let _ = p.set_int(c"OfxParamPropGroupOpen", 0, 0);
    p.set_string(c"OfxParamPropParent", 0, c"InputGroup")?;
    let p = parameters.define(c"OfxParamTypeBoolean", c"InputCompression")?;
    p.set_string(c"OfxPropLabel", 0, c"Input gamut compression")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Compress out-of-domain input chromaticities before TC spectral reconstruction.",
    );
    p.set_int(c"OfxParamPropDefault", 0, 1)?;
    p.set_string(c"OfxParamPropParent", 0, c"InputAdvancedGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeBoolean", c"Hanatos2025AdaptationWindow")?;
    p.set_string(c"OfxPropLabel", 0, c"hanatos2025 adaptation window")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Apply the Hanatos 2025 bandpass adaptation window when reconstructing spectra.",
    );
    p.set_int(c"OfxParamPropDefault", 0, 1)?;
    p.set_string(c"OfxParamPropParent", 0, c"InputAdvancedGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeBoolean", c"Hanatos2025AdaptationSurface")?;
    p.set_string(c"OfxPropLabel", 0, c"hanatos2025 adaptation surface")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Apply the Hanatos 2025 surface adaptation polynomial when reconstructing spectra.",
    );
    p.set_int(c"OfxParamPropDefault", 0, 0)?;
    p.set_string(c"OfxParamPropParent", 0, c"InputAdvancedGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeGroup", c"PrintGroup")?;
    p.set_string(c"OfxPropLabel", 0, c"Print")?;
    let p = parameters.define(c"OfxParamTypeStrChoice", c"PrintProfileKey")?;
    p.set_string(c"OfxPropLabel", 0, c"Print paper")?;
    for entry in catalog.prints() {
        catalog_option(&p, entry.key(), entry.label())?;
    }
    p.set_string(c"OfxParamPropDefault", 0, c"kodak_portra_endura")?;
    p.set_string(c"OfxParamPropParent", 0, c"PrintGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeStrChoice", c"ScanRoute")?;
    p.set_string(c"OfxPropLabel", 0, c"Scan route")?;
    string_option(&p, c"NegativeDirectScan", c"Negative direct scan")?;
    string_option(&p, c"NegativePrintScan", c"Negative print scan")?;
    string_option(&p, c"PositiveDirectScan", c"Positive direct scan")?;
    string_option(&p, c"PositivePrintScan", c"Positive print scan")?;
    p.set_string(c"OfxParamPropDefault", 0, c"NegativePrintScan")?;
    p.set_string(c"OfxParamPropParent", 0, c"PrintGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"PrintExposure")?;
    p.set_string(c"OfxPropLabel", 0, c"Print exposure")?;
    p.set_double(c"OfxParamPropDefault", 0, 1.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.1)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 10.0)?;
    p.set_string(c"OfxParamPropParent", 0, c"PrintGroup")?;
    let p = parameters.define(c"OfxParamTypeDouble", c"PrintPreflash")?;
    p.set_string(c"OfxPropLabel", 0, c"Print preflash")?;
    p.set_double(c"OfxParamPropDefault", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 1.0)?;
    p.set_string(c"OfxParamPropParent", 0, c"PrintGroup")?;
    let p = parameters.define(c"OfxParamTypeBoolean", c"PrintExposureCompensation")?;
    p.set_string(c"OfxPropLabel", 0, c"Print exposure compensation")?;
    p.set_int(c"OfxParamPropDefault", 0, 1)?;
    p.set_string(c"OfxParamPropParent", 0, c"PrintGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"EnlargerY")?;
    p.set_string(c"OfxPropLabel", 0, c"Enlarger Y offset (Kodak CC)")?;
    p.set_double(c"OfxParamPropDefault", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, -200.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 200.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 1.0)?;
    p.set_string(c"OfxParamPropParent", 0, c"PrintGroup")?;
    let p = parameters.define(c"OfxParamTypeDouble", c"EnlargerM")?;
    p.set_string(c"OfxPropLabel", 0, c"Enlarger M offset (Kodak CC)")?;
    p.set_double(c"OfxParamPropDefault", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, -200.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 200.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 1.0)?;
    p.set_string(c"OfxParamPropParent", 0, c"PrintGroup")?;
    let p = parameters.define(c"OfxParamTypeDouble", c"EnlargerC")?;
    p.set_string(c"OfxPropLabel", 0, c"Enlarger C offset (Kodak CC)")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"User cyan offset carried as FilmJuicerMainCFilterShift at the print recipe boundary.",
    );
    p.set_double(c"OfxParamPropDefault", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, -200.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 200.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 1.0)?;
    p.set_string(c"OfxParamPropParent", 0, c"PrintGroup")?;
    let p = parameters.define(c"OfxParamTypeGroup", c"TuningGroup")?;
    p.set_string(c"OfxPropLabel", 0, c"Tuning")?;
    let _ = p.set_int(c"OfxParamPropGroupOpen", 0, 0);
    let p = parameters.define(c"OfxParamTypeDouble", c"FilmGammaFactor")?;
    p.set_string(c"OfxPropLabel", 0, c"Film gamma factor")?;
    p.set_double(c"OfxParamPropDefault", 0, 1.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.05)?;
    p.set_double(c"OfxParamPropMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.5)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 2.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.05)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Adjust film-development contrast. Below 1 reduces contrast; above 1 increases contrast.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"TuningGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    p.set_int(c"OfxParamPropAnimates", 0, 0)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"PrintGammaFactor")?;
    p.set_string(c"OfxPropLabel", 0, c"Print gamma factor")?;
    p.set_double(c"OfxParamPropDefault", 0, 1.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.5)?;
    p.set_double(c"OfxParamPropMax", 0, 2.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.5)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 2.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.05)?;
    let _ = p.set_string(c"OfxParamPropHint", 0, c"Adjust print-medium development contrast. Below 1 reduces contrast; above 1 increases contrast.");
    p.set_string(c"OfxParamPropParent", 0, c"TuningGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    p.set_int(c"OfxParamPropAnimates", 0, 0)?;
    let p = parameters.define(c"OfxParamTypeGroup", c"Couplers")?;
    p.set_string(c"OfxPropLabel", 0, c"DIR couplers")?;
    let p = parameters.define(c"OfxParamTypeBoolean", c"CouplersActive")?;
    p.set_string(c"OfxPropLabel", 0, c"Active")?;
    p.set_int(c"OfxParamPropDefault", 0, 1)?;
    p.set_string(c"OfxParamPropParent", 0, c"Couplers")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"CouplersAmount")?;
    p.set_string(c"OfxPropLabel", 0, c"Amount")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Global multiplier on the DIR inhibition matrix.",
    );
    p.set_double(c"OfxParamPropDefault", 0, 1.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 2.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 2.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.05)?;
    p.set_string(c"OfxParamPropParent", 0, c"Couplers")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"CouplersInhibitionSameLayer")?;
    p.set_string(c"OfxPropLabel", 0, c"Inhibition_samelayer")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Multiplier on same-layer DIR inhibition.",
    );
    p.set_double(c"OfxParamPropDefault", 0, 1.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 2.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 2.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.05)?;
    p.set_string(c"OfxParamPropParent", 0, c"Couplers")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"CouplersInhibitionInterlayer")?;
    p.set_string(c"OfxPropLabel", 0, c"Inhibition_interlayer")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Multiplier on cross-layer DIR inhibition.",
    );
    p.set_double(c"OfxParamPropDefault", 0, 1.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 2.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 2.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.05)?;
    p.set_string(c"OfxParamPropParent", 0, c"Couplers")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"CouplersDiffusionSizeUm")?;
    p.set_string(c"OfxPropLabel", 0, c"Diffusion_size_um")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Sigma in micrometers for spatial diffusion of the DIR correction.",
    );
    p.set_double(c"OfxParamPropDefault", 0, 20.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 200.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 200.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 5.0)?;
    p.set_string(c"OfxParamPropParent", 0, c"Couplers")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeGroup", c"CouplersAdvancedGroup")?;
    p.set_string(c"OfxPropLabel", 0, c"Advanced")?;
    let _ = p.set_int(c"OfxParamPropGroupOpen", 0, 0);
    p.set_string(c"OfxParamPropParent", 0, c"Couplers")?;
    let p = parameters.define(c"OfxParamTypeDouble3D", c"CouplersLangmuirDonorKRgb")?;
    p.set_string(c"OfxPropLabel", 0, c"Langmuir donor K RGB")?;
    let _ = p.set_string(c"OfxParamPropHint", 0, c"Negative-film normalized donor knees in RGB order. Values above the display range may be entered numerically.");
    p.set_double(c"OfxParamPropDefault", 0, 1.0)?;
    p.set_double(c"OfxParamPropDefault", 1, 1.0)?;
    p.set_double(c"OfxParamPropDefault", 2, 1.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.1)?;
    p.set_double(c"OfxParamPropMin", 1, 0.1)?;
    p.set_double(c"OfxParamPropMin", 2, 0.1)?;
    p.set_double(c"OfxParamPropMax", 0, f64::from(f32::MAX))?;
    p.set_double(c"OfxParamPropMax", 1, f64::from(f32::MAX))?;
    p.set_double(c"OfxParamPropMax", 2, f64::from(f32::MAX))?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.1)?;
    p.set_double(c"OfxParamPropDisplayMin", 1, 0.1)?;
    p.set_double(c"OfxParamPropDisplayMin", 2, 0.1)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 10.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 1, 10.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 2, 10.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.1)?;
    p.set_string(c"OfxParamPropDimensionLabel", 0, c"R")?;
    p.set_string(c"OfxParamPropDimensionLabel", 1, c"G")?;
    p.set_string(c"OfxParamPropDimensionLabel", 2, c"B")?;
    p.set_string(c"OfxParamPropParent", 0, c"CouplersAdvancedGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble3D", c"CouplersLangmuirReceiverKRgb")?;
    p.set_string(c"OfxPropLabel", 0, c"Langmuir receiver K RGB")?;
    let _ = p.set_string(c"OfxParamPropHint", 0, c"Positive-film normalized receiver knees in RGB order. Values above the display range may be entered numerically.");
    p.set_double(c"OfxParamPropDefault", 0, 1.0)?;
    p.set_double(c"OfxParamPropDefault", 1, 1.0)?;
    p.set_double(c"OfxParamPropDefault", 2, 1.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.1)?;
    p.set_double(c"OfxParamPropMin", 1, 0.1)?;
    p.set_double(c"OfxParamPropMin", 2, 0.1)?;
    p.set_double(c"OfxParamPropMax", 0, f64::from(f32::MAX))?;
    p.set_double(c"OfxParamPropMax", 1, f64::from(f32::MAX))?;
    p.set_double(c"OfxParamPropMax", 2, f64::from(f32::MAX))?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.1)?;
    p.set_double(c"OfxParamPropDisplayMin", 1, 0.1)?;
    p.set_double(c"OfxParamPropDisplayMin", 2, 0.1)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 10.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 1, 10.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 2, 10.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.1)?;
    p.set_string(c"OfxParamPropDimensionLabel", 0, c"R")?;
    p.set_string(c"OfxParamPropDimensionLabel", 1, c"G")?;
    p.set_string(c"OfxParamPropDimensionLabel", 2, c"B")?;
    p.set_string(c"OfxParamPropParent", 0, c"CouplersAdvancedGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"CouplersDiffusionTailUm")?;
    p.set_string(c"OfxPropLabel", 0, c"Diffusion tail (um)")?;
    let _ = p.set_string(c"OfxParamPropHint", 0, c"Physical DIR tail sigma in micrometers; consumed only when spatial DIR and tail weight are active.");
    p.set_double(c"OfxParamPropDefault", 0, 200.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, f64::from(f32::MAX))?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 1000.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 10.0)?;
    p.set_string(c"OfxParamPropParent", 0, c"CouplersAdvancedGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"CouplersDiffusionTailWeight")?;
    p.set_string(c"OfxPropLabel", 0, c"Diffusion tail weight")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Tail mixture weight; consumed only while the spatial DIR core is active.",
    );
    p.set_double(c"OfxParamPropDefault", 0, 0.03)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 1.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 1.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.01)?;
    p.set_string(c"OfxParamPropParent", 0, c"CouplersAdvancedGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeBoolean", c"CouplersGammaUseStock")?;
    p.set_string(c"OfxPropLabel", 0, c"Gamma_use_stock")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Use the selected film profile's DIR gamma values.",
    );
    p.set_int(c"OfxParamPropDefault", 0, 1)?;
    p.set_string(c"OfxParamPropParent", 0, c"CouplersAdvancedGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble3D", c"CouplersGammaSameLayerRgb")?;
    p.set_string(c"OfxPropLabel", 0, c"Gamma_samelayer_rgb")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Same-layer DIR gamma in donor RGB order.",
    );
    p.set_double(c"OfxParamPropDefault", 0, 0.341)?;
    p.set_double(c"OfxParamPropDefault", 1, 0.324)?;
    p.set_double(c"OfxParamPropDefault", 2, 0.273)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMin", 1, 0.0)?;
    p.set_double(c"OfxParamPropMin", 2, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropMax", 1, 4.0)?;
    p.set_double(c"OfxParamPropMax", 2, 4.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 1, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 2, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 1.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 1, 1.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 2, 1.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.02)?;
    p.set_string(c"OfxParamPropDimensionLabel", 0, c"R")?;
    p.set_string(c"OfxParamPropDimensionLabel", 1, c"G")?;
    p.set_string(c"OfxParamPropDimensionLabel", 2, c"B")?;
    p.set_string(c"OfxParamPropParent", 0, c"CouplersAdvancedGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble2D", c"CouplersGammaInterlayerRToGb")?;
    p.set_string(c"OfxPropLabel", 0, c"Gamma_interlayer_r_to_gb")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"DIR inhibition from the R layer onto G and B.",
    );
    p.set_double(c"OfxParamPropDefault", 0, 0.355)?;
    p.set_double(c"OfxParamPropDefault", 1, 0.305)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMin", 1, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropMax", 1, 4.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 1, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 1.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 1, 1.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.02)?;
    p.set_string(c"OfxParamPropDimensionLabel", 0, c"G")?;
    p.set_string(c"OfxParamPropDimensionLabel", 1, c"B")?;
    p.set_string(c"OfxParamPropParent", 0, c"CouplersAdvancedGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble2D", c"CouplersGammaInterlayerGToRb")?;
    p.set_string(c"OfxPropLabel", 0, c"Gamma_interlayer_g_to_rb")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"DIR inhibition from the G layer onto R and B.",
    );
    p.set_double(c"OfxParamPropDefault", 0, 0.154)?;
    p.set_double(c"OfxParamPropDefault", 1, 0.358)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMin", 1, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropMax", 1, 4.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 1, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 1.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 1, 1.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.02)?;
    p.set_string(c"OfxParamPropDimensionLabel", 0, c"R")?;
    p.set_string(c"OfxParamPropDimensionLabel", 1, c"B")?;
    p.set_string(c"OfxParamPropParent", 0, c"CouplersAdvancedGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble2D", c"CouplersGammaInterlayerBToRg")?;
    p.set_string(c"OfxPropLabel", 0, c"Gamma_interlayer_b_to_rg")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"DIR inhibition from the B layer onto R and G.",
    );
    p.set_double(c"OfxParamPropDefault", 0, 0.171)?;
    p.set_double(c"OfxParamPropDefault", 1, 0.225)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMin", 1, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropMax", 1, 4.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 1, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 1.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 1, 1.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.02)?;
    p.set_string(c"OfxParamPropDimensionLabel", 0, c"R")?;
    p.set_string(c"OfxParamPropDimensionLabel", 1, c"G")?;
    p.set_string(c"OfxParamPropParent", 0, c"CouplersAdvancedGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeGroup", c"ScannerOptics")?;
    p.set_string(c"OfxPropLabel", 0, c"Scanner Optics")?;
    let p = parameters.define(c"OfxParamTypeGroup", c"ScannerMath")?;
    p.set_string(c"OfxPropLabel", 0, c"Scanner Math")?;
    let p = parameters.define(c"OfxParamTypeDouble", c"ScannerLensBlurSigmaPx")?;
    p.set_string(c"OfxPropLabel", 0, c"Scanner lens blur (px)")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Gaussian blur sigma in pixels for scanner optics.",
    );
    p.set_double(c"OfxParamPropDefault", 0, 0.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 10.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 3.0)?;
    p.set_string(c"OfxParamPropParent", 0, c"ScannerOptics")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble2D", c"ScannerUnsharpMask")?;
    p.set_string(c"OfxPropLabel", 0, c"Scanner unsharp mask")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Unsharp sigma (px) and amount applied after scanner blur.",
    );
    p.set_double(c"OfxParamPropDefault", 0, 0.7)?;
    p.set_double(c"OfxParamPropDefault", 1, 0.7)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMin", 1, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 5.0)?;
    p.set_double(c"OfxParamPropMax", 1, 3.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 1, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 5.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 1, 3.0)?;
    p.set_string(c"OfxParamPropDimensionLabel", 0, c"Sigma (px)")?;
    p.set_string(c"OfxParamPropDimensionLabel", 1, c"Amount")?;
    p.set_string(c"OfxParamPropParent", 0, c"ScannerOptics")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeBoolean", c"ScannerBlackCorrection")?;
    p.set_string(c"OfxPropLabel", 0, c"Scanner black correction")?;
    p.set_int(c"OfxParamPropDefault", 0, 0)?;
    p.set_string(c"OfxParamPropParent", 0, c"ScannerMath")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeBoolean", c"ScannerWhiteCorrection")?;
    p.set_string(c"OfxPropLabel", 0, c"Scanner white correction")?;
    p.set_int(c"OfxParamPropDefault", 0, 0)?;
    p.set_string(c"OfxParamPropParent", 0, c"ScannerMath")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"ScannerBlackLevel")?;
    p.set_string(c"OfxPropLabel", 0, c"Scanner black level")?;
    p.set_double(c"OfxParamPropDefault", 0, 0.01)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 1.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 0.1)?;
    p.set_string(c"OfxParamPropParent", 0, c"ScannerMath")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"ScannerWhiteLevel")?;
    p.set_string(c"OfxPropLabel", 0, c"Scanner white level")?;
    p.set_double(c"OfxParamPropDefault", 0, 0.98)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 1.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.8)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 1.0)?;
    p.set_string(c"OfxParamPropParent", 0, c"ScannerMath")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeGroup", c"HalationGroup")?;
    p.set_string(c"OfxPropLabel", 0, c"Halation")?;
    let _ = p.set_int(c"OfxParamPropGroupOpen", 0, 0);
    let p = parameters.define(c"OfxParamTypeBoolean", c"HalationActive")?;
    p.set_string(c"OfxPropLabel", 0, c"Add halation")?;
    p.set_int(c"OfxParamPropDefault", 0, 1)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Enable in-emulsion scatter and back-reflection in the capture film.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"HalationGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"HalationScatterAmount")?;
    p.set_string(c"OfxPropLabel", 0, c"Scatter amount")?;
    p.set_double(c"OfxParamPropDefault", 0, 1.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 2.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 2.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.05)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Scale in-emulsion scatter strength in the capture film.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"HalationGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"HalationScatterSpatialScale")?;
    p.set_string(c"OfxPropLabel", 0, c"Scatter spatial scale")?;
    p.set_double(c"OfxParamPropDefault", 0, 1.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 2.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 2.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.1)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Scale the in-emulsion scatter radius in the capture film.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"HalationGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"HalationAmount")?;
    p.set_string(c"OfxPropLabel", 0, c"Halation amount")?;
    p.set_double(c"OfxParamPropDefault", 0, 1.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 2.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 2.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.05)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Scale profile-derived back-reflection strength in the capture film.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"HalationGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"HalationSpatialScale")?;
    p.set_string(c"OfxPropLabel", 0, c"Halation spatial scale")?;
    p.set_double(c"OfxParamPropDefault", 0, 1.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 2.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 2.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.1)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Scale the profile-derived back-reflection radius in the capture film.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"HalationGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    describe_advanced_halation(parameters)?;
    let p = parameters.define(c"OfxParamTypeGroup", c"GrainGroup")?;
    p.set_string(c"OfxPropLabel", 0, c"Grain")?;
    let _ = p.set_int(c"OfxParamPropGroupOpen", 0, 0);
    let p = parameters.define(c"OfxParamTypeBoolean", c"GrainActive")?;
    p.set_string(c"OfxPropLabel", 0, c"Add grain")?;
    p.set_int(c"OfxParamPropDefault", 0, 0)?;
    let _ = p.set_string(c"OfxParamPropHint", 0, c"Add grain to the negative.");
    p.set_string(c"OfxParamPropParent", 0, c"GrainGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    #[cfg(feature = "grain-preview")]
    {
        let p = parameters.define(c"OfxParamTypeStrChoice", c"GrainModel")?;
        p.set_string(c"OfxPropLabel", 0, c"Model")?;
        string_option(&p, c"current", c"Current")?;
        string_option(&p, c"film_v1", c"Film v1")?;
        p.set_string(c"OfxParamPropDefault", 0, c"current")?;
        p.set_string(c"OfxParamPropParent", 0, c"GrainGroup")?;
        p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
        for (name, label, default, min, max, increment) in [
            (c"FilmGrainAmountEv", c"Amount (EV)", 0.0, -3.0, 3.0, 0.1),
            (c"FilmGrainTexelPitchUm", c"Size (µm)", 1.0, 0.5, 4.0, 0.01),
            (
                c"FilmGrainChromaFraction",
                c"Colour",
                film_juicer_core::film::grain::preview::DEFAULT_CHROMA,
                0.0,
                1.0,
                0.01,
            ),
        ] {
            let p = parameters.define(c"OfxParamTypeDouble", name)?;
            p.set_string(c"OfxPropLabel", 0, label)?;
            p.set_double(c"OfxParamPropDefault", 0, default)?;
            p.set_double(c"OfxParamPropMin", 0, min)?;
            p.set_double(c"OfxParamPropMax", 0, max)?;
            p.set_double(c"OfxParamPropDisplayMin", 0, min)?;
            p.set_double(c"OfxParamPropDisplayMax", 0, max)?;
            p.set_double(c"OfxParamPropIncrement", 0, increment)?;
            p.set_string(c"OfxParamPropParent", 0, c"GrainGroup")?;
            p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
            p.set_int(c"OfxParamPropSecret", 0, 1)?;
        }
    }
    let p = parameters.define(c"OfxParamTypeBoolean", c"GrainSublayersActive")?;
    p.set_string(c"OfxPropLabel", 0, c"Use sublayers")?;
    p.set_int(c"OfxParamPropDefault", 0, 1)?;
    let _ = p.set_string(c"OfxParamPropHint", 0, c"Enable sublayer grain simulation.");
    p.set_string(c"OfxParamPropParent", 0, c"GrainGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeChoice", c"GrainPreset")?;
    p.set_string(c"OfxPropLabel", 0, c"Grain Preset")?;
    option(&p, c"Fine")?;
    option(&p, c"Medium")?;
    option(&p, c"Coarse")?;
    p.set_int(c"OfxParamPropDefault", 0, 1)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Starting point for grain size/strength presets.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"GrainGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"GrainAmplitude")?;
    p.set_string(c"OfxPropLabel", 0, c"Grain Amount (EV)")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Grain amount in stops (EV); scales OD delta.",
    );
    p.set_double(c"OfxParamPropDefault", 0, -1.20)?;
    p.set_double(c"OfxParamPropMin", 0, -3.0)?;
    p.set_double(c"OfxParamPropMax", 0, 3.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, -3.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 3.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.1)?;
    p.set_string(c"OfxParamPropParent", 0, c"GrainGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"GrainBlur")?;
    p.set_string(c"OfxPropLabel", 0, c"Grain Size (px)")?;
    let _ = p.set_string(c"OfxParamPropHint", 0, c"Grain correlation size in pixels.");
    p.set_double(c"OfxParamPropDefault", 0, 0.50)?;
    p.set_double(c"OfxParamPropMin", 0, 0.20)?;
    p.set_double(c"OfxParamPropMax", 0, 2.00)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.20)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 2.00)?;
    p.set_string(c"OfxParamPropParent", 0, c"GrainGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"GrainSharpness")?;
    p.set_string(c"OfxPropLabel", 0, c"Grain Sharpness")?;
    let _ = p.set_string(c"OfxParamPropHint", 0, c"Controls grain edge softness.");
    p.set_double(c"OfxParamPropDefault", 0, 0.5)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 1.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 1.0)?;
    p.set_string(c"OfxParamPropParent", 0, c"GrainGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"GrainChroma")?;
    p.set_string(c"OfxPropLabel", 0, c"Grain Chroma")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"0 = achromatic, 1 = independent RGB grain.",
    );
    p.set_double(c"OfxParamPropDefault", 0, 0.3)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 1.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 1.0)?;
    p.set_string(c"OfxParamPropParent", 0, c"GrainGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"GrainTexture")?;
    p.set_string(c"OfxPropLabel", 0, c"Grain Texture")?;
    let _ = p.set_string(c"OfxParamPropHint", 0, c"Controls clumping and PSD tail.");
    p.set_double(c"OfxParamPropDefault", 0, 0.55)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 1.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 1.0)?;
    p.set_string(c"OfxParamPropParent", 0, c"GrainGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeGroup", c"GrainAdvancedGroup")?;
    p.set_string(c"OfxPropLabel", 0, c"Advanced")?;
    let _ = p.set_int(c"OfxParamPropGroupOpen", 0, 0);
    p.set_string(c"OfxParamPropParent", 0, c"GrainGroup")?;
    let p = parameters.define(c"OfxParamTypePushButton", c"GrainResetAdvanced")?;
    p.set_string(c"OfxPropLabel", 0, c"Reset Advanced")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Reset advanced grain controls to the base values.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"GrainAdvancedGroup")?;
    let p = parameters.define(c"OfxParamTypeDouble", c"GrainParticleAreaUm2")?;
    p.set_string(c"OfxPropLabel", 0, c"Particle area (um^2)")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Particle area in um^2; roughly 0.1 for ISO 100-200, 0.4 for ISO 400.",
    );
    p.set_double(c"OfxParamPropDefault", 0, 0.25)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 10.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 1.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.1)?;
    p.set_string(c"OfxParamPropParent", 0, c"GrainAdvancedGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"GrainParticleScaleMaster")?;
    p.set_string(c"OfxPropLabel", 0, c"Particle scale (M)")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Master control for particle scale; adjusts RGB values together.",
    );
    p.set_double(c"OfxParamPropDefault", 0, 1.48)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 10.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 3.0)?;
    p.set_string(c"OfxParamPropParent", 0, c"GrainAdvancedGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"GrainParticleScaleLayersMaster")?;
    p.set_string(c"OfxPropLabel", 0, c"Particle scale layers (M)")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Master control for sublayer particle scale; adjusts RGB values together.",
    );
    p.set_double(c"OfxParamPropDefault", 0, 1.922)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 10.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 4.0)?;
    p.set_string(c"OfxParamPropParent", 0, c"GrainAdvancedGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"GrainDensityMinMaster")?;
    p.set_string(c"OfxPropLabel", 0, c"Density min (M)")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Master control for minimum grain density; adjusts RGB values together.",
    );
    p.set_double(c"OfxParamPropDefault", 0, 0.08)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 1.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 0.2)?;
    p.set_string(c"OfxParamPropParent", 0, c"GrainAdvancedGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"GrainUniformityMaster")?;
    p.set_string(c"OfxPropLabel", 0, c"Uniformity (M)")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Master control for grain uniformity; adjusts RGB values together.",
    );
    p.set_double(c"OfxParamPropDefault", 0, 0.97)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 1.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.9)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 1.0)?;
    p.set_string(c"OfxParamPropParent", 0, c"GrainAdvancedGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"GrainBlurDyeCloudsUm")?;
    p.set_string(c"OfxPropLabel", 0, c"Dye cloud blur scale (px)")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Scale factor for dye cloud blur sigma in pixels.",
    );
    p.set_double(c"OfxParamPropDefault", 0, 1.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 10.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 3.0)?;
    p.set_string(c"OfxParamPropParent", 0, c"GrainAdvancedGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"GrainSizeMixWeight")?;
    p.set_string(c"OfxPropLabel", 0, c"Size mix weight")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Weight of the coarse grain population in the size mixture.",
    );
    p.set_double(c"OfxParamPropDefault", 0, 0.226)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 1.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 1.0)?;
    p.set_string(c"OfxParamPropParent", 0, c"GrainAdvancedGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"GrainSizeMixWeightMid")?;
    p.set_string(c"OfxPropLabel", 0, c"Size mix mid weight")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Weight of the mid grain population in the size mixture.",
    );
    p.set_double(c"OfxParamPropDefault", 0, 0.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 1.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 1.0)?;
    p.set_string(c"OfxParamPropParent", 0, c"GrainAdvancedGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"GrainSizeMixScale")?;
    p.set_string(c"OfxPropLabel", 0, c"Size mix scale")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Relative particle size scale for the coarse grain population.",
    );
    p.set_double(c"OfxParamPropDefault", 0, 19.0)?;
    p.set_double(c"OfxParamPropMin", 0, 1.0)?;
    p.set_double(c"OfxParamPropMax", 0, 50.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 1.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 50.0)?;
    p.set_string(c"OfxParamPropParent", 0, c"GrainAdvancedGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble2D", c"GrainMicroStructure")?;
    p.set_string(c"OfxPropLabel", 0, c"Micro-structure")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Micro-structure parameters: clump cell size (um) and clump sigma (x1e-3).",
    );
    p.set_double(c"OfxParamPropDefault", 0, 60.0)?;
    p.set_double(c"OfxParamPropDefault", 1, 170.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMin", 1, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 100.0)?;
    p.set_double(c"OfxParamPropMax", 1, 1000.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 1, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 100.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 1, 200.0)?;
    p.set_string(c"OfxParamPropDimensionLabel", 0, c"Cell (um)")?;
    p.set_string(c"OfxParamPropDimensionLabel", 1, c"Sigma (1e-3)")?;
    p.set_string(c"OfxParamPropParent", 0, c"GrainAdvancedGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeChoice", c"GrainDebugView")?;
    p.set_string(c"OfxPropLabel", 0, c"Grain debug view")?;
    option(&p, c"Off")?;
    option(&p, c"Delta mix")?;
    option(&p, c"Delta fine")?;
    option(&p, c"Delta coarse")?;
    option(&p, c"Delta fine raw")?;
    option(&p, c"Delta coarse raw")?;
    option(&p, c"Mean density")?;
    p.set_int(c"OfxParamPropDefault", 0, 0)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Debug view selector for grain delta fields (pre-scanner).",
    );
    p.set_string(c"OfxParamPropParent", 0, c"GrainAdvancedGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble3D", c"GrainParticleScale")?;
    p.set_string(c"OfxPropLabel", 0, c"Particle scale")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Scale of particle area for the RGB layers (multiplies particle area).",
    );
    p.set_double(c"OfxParamPropDefault", 0, 1.48)?;
    p.set_double(c"OfxParamPropDefault", 1, 1.48)?;
    p.set_double(c"OfxParamPropDefault", 2, 1.48)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMin", 1, 0.0)?;
    p.set_double(c"OfxParamPropMin", 2, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 10.0)?;
    p.set_double(c"OfxParamPropMax", 1, 10.0)?;
    p.set_double(c"OfxParamPropMax", 2, 10.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 1, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 2, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 3.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 1, 3.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 2, 3.0)?;
    p.set_string(c"OfxParamPropDimensionLabel", 0, c"R")?;
    p.set_string(c"OfxParamPropDimensionLabel", 1, c"G")?;
    p.set_string(c"OfxParamPropDimensionLabel", 2, c"B")?;
    p.set_int(c"OfxParamPropSecret", 0, 1)?;
    p.set_string(c"OfxParamPropParent", 0, c"GrainAdvancedGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble3D", c"GrainParticleScaleLayers")?;
    p.set_string(c"OfxPropLabel", 0, c"Particle scale layers")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Scale of particle area for sublayers in each color layer.",
    );
    p.set_double(c"OfxParamPropDefault", 0, 1.922)?;
    p.set_double(c"OfxParamPropDefault", 1, 1.922)?;
    p.set_double(c"OfxParamPropDefault", 2, 1.922)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMin", 1, 0.0)?;
    p.set_double(c"OfxParamPropMin", 2, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 10.0)?;
    p.set_double(c"OfxParamPropMax", 1, 10.0)?;
    p.set_double(c"OfxParamPropMax", 2, 10.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 1, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 2, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 1, 4.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 2, 4.0)?;
    p.set_string(c"OfxParamPropDimensionLabel", 0, c"R")?;
    p.set_string(c"OfxParamPropDimensionLabel", 1, c"G")?;
    p.set_string(c"OfxParamPropDimensionLabel", 2, c"B")?;
    p.set_int(c"OfxParamPropSecret", 0, 1)?;
    p.set_string(c"OfxParamPropParent", 0, c"GrainAdvancedGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble3D", c"GrainDensityMin")?;
    p.set_string(c"OfxPropLabel", 0, c"Density min")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Minimum grain density per layer (typ. 0.03-0.06).",
    );
    p.set_double(c"OfxParamPropDefault", 0, 0.08)?;
    p.set_double(c"OfxParamPropDefault", 1, 0.08)?;
    p.set_double(c"OfxParamPropDefault", 2, 0.08)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMin", 1, 0.0)?;
    p.set_double(c"OfxParamPropMin", 2, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 1.0)?;
    p.set_double(c"OfxParamPropMax", 1, 1.0)?;
    p.set_double(c"OfxParamPropMax", 2, 1.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 1, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 2, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 0.2)?;
    p.set_double(c"OfxParamPropDisplayMax", 1, 0.2)?;
    p.set_double(c"OfxParamPropDisplayMax", 2, 0.2)?;
    p.set_string(c"OfxParamPropDimensionLabel", 0, c"C")?;
    p.set_string(c"OfxParamPropDimensionLabel", 1, c"M")?;
    p.set_string(c"OfxParamPropDimensionLabel", 2, c"Y")?;
    p.set_int(c"OfxParamPropSecret", 0, 1)?;
    p.set_string(c"OfxParamPropParent", 0, c"GrainAdvancedGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble3D", c"GrainUniformity")?;
    p.set_string(c"OfxPropLabel", 0, c"Uniformity")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Uniformity of grain (typ. 0.94-0.98).",
    );
    p.set_double(c"OfxParamPropDefault", 0, 0.97)?;
    p.set_double(c"OfxParamPropDefault", 1, 0.97)?;
    p.set_double(c"OfxParamPropDefault", 2, 0.97)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMin", 1, 0.0)?;
    p.set_double(c"OfxParamPropMin", 2, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 1.0)?;
    p.set_double(c"OfxParamPropMax", 1, 1.0)?;
    p.set_double(c"OfxParamPropMax", 2, 1.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.9)?;
    p.set_double(c"OfxParamPropDisplayMin", 1, 0.9)?;
    p.set_double(c"OfxParamPropDisplayMin", 2, 0.9)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 1.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 1, 1.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 2, 1.0)?;
    p.set_string(c"OfxParamPropDimensionLabel", 0, c"C")?;
    p.set_string(c"OfxParamPropDimensionLabel", 1, c"M")?;
    p.set_string(c"OfxParamPropDimensionLabel", 2, c"Y")?;
    p.set_int(c"OfxParamPropSecret", 0, 1)?;
    p.set_string(c"OfxParamPropParent", 0, c"GrainAdvancedGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"GrainClumpTemporalMix")?;
    p.set_string(c"OfxPropLabel", 0, c"Clump temporal mix")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Modulates clump strength over time (0 = static, higher = more breathing).",
    );
    p.set_double(c"OfxParamPropDefault", 0, 0.30)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 0.30)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 0.30)?;
    p.set_string(c"OfxParamPropParent", 0, c"GrainAdvancedGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"GrainClumpMorphPeriodSec")?;
    p.set_string(c"OfxPropLabel", 0, c"Clump morph period (s)")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Seconds for clump field to evolve; longer is more stable.",
    );
    p.set_double(c"OfxParamPropDefault", 0, 8.0)?;
    p.set_double(c"OfxParamPropMin", 0, 5.0)?;
    p.set_double(c"OfxParamPropMax", 0, 60.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 5.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 60.0)?;
    p.set_string(c"OfxParamPropParent", 0, c"GrainAdvancedGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeGroup", c"EffectsGroup")?;
    p.set_string(c"OfxPropLabel", 0, c"Effects")?;
    let _ = p.set_int(c"OfxParamPropGroupOpen", 0, 0);
    let p = parameters.define(c"OfxParamTypeDouble", c"GateWeaveAmount")?;
    p.set_string(c"OfxPropLabel", 0, c"Gate weave")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Scales the global gate weave (film transport jitter).",
    );
    p.set_double(c"OfxParamPropDefault", 0, 0.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 10.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 10.0)?;
    p.set_string(c"OfxParamPropParent", 0, c"EffectsGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"FilmDustAmount")?;
    p.set_string(c"OfxPropLabel", 0, c"Film dust")?;
    let _ = p.set_string(c"OfxParamPropHint", 0, c"Dust carried by the moving film.");
    p.set_double(c"OfxParamPropDefault", 0, 0.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 10.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 10.0)?;
    p.set_string(c"OfxParamPropParent", 0, c"EffectsGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"GateDustAmount")?;
    p.set_string(c"OfxPropLabel", 0, c"Gate dust")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Dust that stays fixed while the picture moves through the gate.",
    );
    p.set_double(c"OfxParamPropDefault", 0, 0.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 10.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 10.0)?;
    p.set_string(c"OfxParamPropParent", 0, c"EffectsGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"FilmScratchAmount")?;
    p.set_string(c"OfxPropLabel", 0, c"Film scratches")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Light damage carried by the moving film.",
    );
    p.set_double(c"OfxParamPropDefault", 0, 0.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 10.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 10.0)?;
    p.set_string(c"OfxParamPropParent", 0, c"EffectsGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"GateScratchAmount")?;
    p.set_string(c"OfxPropLabel", 0, c"Gate scratches")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Fine persistent marks in the scan/gate path.",
    );
    p.set_double(c"OfxParamPropDefault", 0, 0.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 10.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 10.0)?;
    p.set_string(c"OfxParamPropParent", 0, c"EffectsGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeGroup", c"DiffusionGroup")?;
    p.set_string(c"OfxPropLabel", 0, c"Diffusion")?;
    let _ = p.set_int(c"OfxParamPropGroupOpen", 0, 0);
    let p = parameters.define(c"OfxParamTypeBoolean", c"CameraDiffusionEnabled")?;
    p.set_string(c"OfxPropLabel", 0, c"Camera Enabled")?;
    p.set_int(c"OfxParamPropDefault", 0, 0)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Apply diffusion on the camera stage before film exposure.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"DiffusionGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeChoice", c"CameraDiffusionFamily")?;
    p.set_string(c"OfxPropLabel", 0, c"Camera Family")?;
    option(&p, c"Glimmerglass")?;
    option(&p, c"Black Pro-Mist")?;
    option(&p, c"Pro-Mist")?;
    option(&p, c"CineBloom")?;
    p.set_int(c"OfxParamPropDefault", 0, 1)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"PSF family used on the camera stage before film exposure.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"DiffusionGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"CameraDiffusionStrength")?;
    p.set_string(c"OfxPropLabel", 0, c"Camera Strength")?;
    p.set_double(c"OfxParamPropDefault", 0, 0.5)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 2.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 2.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.125)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Commercial filter stop: 0, 1/8, 1/4, 1/2, 1, or 2; intermediate values are supported.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"DiffusionGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"CameraDiffusionSpatialScale")?;
    p.set_string(c"OfxPropLabel", 0, c"Camera Spatial Scale")?;
    p.set_double(c"OfxParamPropDefault", 0, 1.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, f64::MAX)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.1)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Multiplier on the camera-stage image-plane PSF widths.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"DiffusionGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"CameraDiffusionHaloWarmth")?;
    p.set_string(c"OfxPropLabel", 0, c"Camera Halo Warmth")?;
    p.set_double(c"OfxParamPropDefault", 0, 0.0)?;
    p.set_double(c"OfxParamPropMin", 0, -1.5)?;
    p.set_double(c"OfxParamPropMax", 0, 1.5)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, -1.5)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 1.5)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.05)?;
    let _ = p.set_string(c"OfxParamPropHint", 0, c"Additive camera-stage halo warmth offset. Positive warms the outer halo; negative inverts it.");
    p.set_string(c"OfxParamPropParent", 0, c"DiffusionGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"CameraDiffusionCoreIntensity")?;
    p.set_string(c"OfxPropLabel", 0, c"Camera Core Intensity")?;
    p.set_double(c"OfxParamPropDefault", 0, 1.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.05)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Multiplier on the camera-stage core weight. 1.0 uses the family default.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"DiffusionGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"CameraDiffusionCoreSize")?;
    p.set_string(c"OfxPropLabel", 0, c"Camera Core Size")?;
    p.set_double(c"OfxParamPropDefault", 0, 1.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.1)?;
    p.set_double(c"OfxParamPropMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.1)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.05)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Multiplier on the camera-stage core size. 1.0 uses the family default.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"DiffusionGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"CameraDiffusionHaloIntensity")?;
    p.set_string(c"OfxPropLabel", 0, c"Camera Halo Intensity")?;
    p.set_double(c"OfxParamPropDefault", 0, 1.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.05)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Multiplier on the camera-stage halo weight. 1.0 uses the family default.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"DiffusionGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"CameraDiffusionHaloSize")?;
    p.set_string(c"OfxPropLabel", 0, c"Camera Halo Size")?;
    p.set_double(c"OfxParamPropDefault", 0, 1.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.1)?;
    p.set_double(c"OfxParamPropMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.1)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.05)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Multiplier on the camera-stage halo size. 1.0 uses the family default.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"DiffusionGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"CameraDiffusionBloomIntensity")?;
    p.set_string(c"OfxPropLabel", 0, c"Camera Bloom Intensity")?;
    p.set_double(c"OfxParamPropDefault", 0, 1.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.05)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Multiplier on the camera-stage bloom weight. 1.0 uses the family default.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"DiffusionGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"CameraDiffusionBloomSize")?;
    p.set_string(c"OfxPropLabel", 0, c"Camera Bloom Size")?;
    p.set_double(c"OfxParamPropDefault", 0, 1.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.1)?;
    p.set_double(c"OfxParamPropMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.1)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.05)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Multiplier on the camera-stage bloom size. 1.0 uses the family default.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"DiffusionGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeBoolean", c"PrintDiffusionEnabled")?;
    p.set_string(c"OfxPropLabel", 0, c"Print Enabled")?;
    p.set_int(c"OfxParamPropDefault", 0, 0)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Apply diffusion on the print stage after enlarger exposure is formed.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"DiffusionGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeChoice", c"PrintDiffusionFamily")?;
    p.set_string(c"OfxPropLabel", 0, c"Print Family")?;
    option(&p, c"Glimmerglass")?;
    option(&p, c"Black Pro-Mist")?;
    option(&p, c"Pro-Mist")?;
    option(&p, c"CineBloom")?;
    p.set_int(c"OfxParamPropDefault", 0, 1)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"PSF family used on the print stage after enlarger exposure is formed.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"DiffusionGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"PrintDiffusionStrength")?;
    p.set_string(c"OfxPropLabel", 0, c"Print Strength")?;
    p.set_double(c"OfxParamPropDefault", 0, 0.5)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 2.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 2.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.125)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Commercial filter stop: 0, 1/8, 1/4, 1/2, 1, or 2; intermediate values are supported.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"DiffusionGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"PrintDiffusionSpatialScale")?;
    p.set_string(c"OfxPropLabel", 0, c"Print Spatial Scale")?;
    p.set_double(c"OfxParamPropDefault", 0, 1.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, f64::MAX)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.1)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Multiplier on the print-stage image-plane PSF widths.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"DiffusionGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"PrintDiffusionHaloWarmth")?;
    p.set_string(c"OfxPropLabel", 0, c"Print Halo Warmth")?;
    p.set_double(c"OfxParamPropDefault", 0, 0.0)?;
    p.set_double(c"OfxParamPropMin", 0, -1.5)?;
    p.set_double(c"OfxParamPropMax", 0, 1.5)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, -1.5)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 1.5)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.05)?;
    let _ = p.set_string(c"OfxParamPropHint", 0, c"Additive print-stage halo warmth offset. Positive warms the outer halo; negative inverts it.");
    p.set_string(c"OfxParamPropParent", 0, c"DiffusionGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"PrintDiffusionCoreIntensity")?;
    p.set_string(c"OfxPropLabel", 0, c"Print Core Intensity")?;
    p.set_double(c"OfxParamPropDefault", 0, 1.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.05)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Multiplier on the print-stage core weight. 1.0 uses the family default.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"DiffusionGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"PrintDiffusionCoreSize")?;
    p.set_string(c"OfxPropLabel", 0, c"Print Core Size")?;
    p.set_double(c"OfxParamPropDefault", 0, 1.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.1)?;
    p.set_double(c"OfxParamPropMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.1)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.05)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Multiplier on the print-stage core size. 1.0 uses the family default.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"DiffusionGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"PrintDiffusionHaloIntensity")?;
    p.set_string(c"OfxPropLabel", 0, c"Print Halo Intensity")?;
    p.set_double(c"OfxParamPropDefault", 0, 1.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.05)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Multiplier on the print-stage halo weight. 1.0 uses the family default.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"DiffusionGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"PrintDiffusionHaloSize")?;
    p.set_string(c"OfxPropLabel", 0, c"Print Halo Size")?;
    p.set_double(c"OfxParamPropDefault", 0, 1.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.1)?;
    p.set_double(c"OfxParamPropMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.1)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.05)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Multiplier on the print-stage halo size. 1.0 uses the family default.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"DiffusionGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"PrintDiffusionBloomIntensity")?;
    p.set_string(c"OfxPropLabel", 0, c"Print Bloom Intensity")?;
    p.set_double(c"OfxParamPropDefault", 0, 1.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.05)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Multiplier on the print-stage bloom weight. 1.0 uses the family default.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"DiffusionGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"PrintDiffusionBloomSize")?;
    p.set_string(c"OfxPropLabel", 0, c"Print Bloom Size")?;
    p.set_double(c"OfxParamPropDefault", 0, 1.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.1)?;
    p.set_double(c"OfxParamPropMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.1)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 4.0)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.05)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Multiplier on the print-stage bloom size. 1.0 uses the family default.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"DiffusionGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeGroup", c"GlareGroup")?;
    p.set_string(c"OfxPropLabel", 0, c"Glare")?;
    let _ = p.set_int(c"OfxParamPropGroupOpen", 0, 0);
    let p = parameters.define(c"OfxParamTypeBoolean", c"GlareActive")?;
    p.set_string(c"OfxPropLabel", 0, c"Add glare")?;
    p.set_int(c"OfxParamPropDefault", 0, 1)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Add glare to the print (scanner-stage stray light).",
    );
    p.set_string(c"OfxParamPropParent", 0, c"GlareGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"GlarePercent")?;
    p.set_string(c"OfxPropLabel", 0, c"Glare percent")?;
    p.set_double(c"OfxParamPropDefault", 0, 0.03)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 1.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 0.5)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.05)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Percentage of glare light (typ. 0.10-0.25). Value is in percent, not fraction.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"GlareGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"GlareRoughness")?;
    p.set_string(c"OfxPropLabel", 0, c"Glare roughness")?;
    p.set_double(c"OfxParamPropDefault", 0, 0.7)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 1.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 1.0)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Glare roughness (0-1). Stddev = roughness * percent.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"GlareGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"GlareBlurSigmaPx")?;
    p.set_string(c"OfxPropLabel", 0, c"Glare blur sigma (px)")?;
    p.set_double(c"OfxParamPropDefault", 0, 0.5)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 10.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 3.0)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Gaussian blur sigma in pixels applied to the glare field.",
    );
    p.set_string(c"OfxParamPropParent", 0, c"GlareGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"PrintShadowCompensationFactor")?;
    p.set_string(c"OfxPropLabel", 0, c"Compensation removal factor")?;
    p.set_double(c"OfxParamPropDefault", 0, 0.0)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 1.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 0.2)?;
    p.set_double(c"OfxParamPropIncrement", 0, 0.05)?;
    let _ = p.set_string(c"OfxParamPropHint", 0, c"Apply print shadow compensation to density curves. 0.2 = 20% underexposed shadows. Intended as alternative to stochastic glare (set GlarePercent=0).");
    p.set_string(c"OfxParamPropParent", 0, c"GlareGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"PrintShadowCompensationDensity")?;
    p.set_string(c"OfxPropLabel", 0, c"Compensation removal density")?;
    p.set_double(c"OfxParamPropDefault", 0, 1.2)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 3.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.8)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 2.0)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Density at which the compensation-removal transition is centered (typ. 1.0-1.5).",
    );
    p.set_string(c"OfxParamPropParent", 0, c"GlareGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeDouble", c"PrintShadowCompensationTransition")?;
    p.set_string(c"OfxPropLabel", 0, c"Compensation removal transition")?;
    p.set_double(c"OfxParamPropDefault", 0, 0.3)?;
    p.set_double(c"OfxParamPropMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropMax", 0, 2.0)?;
    p.set_double(c"OfxParamPropDisplayMin", 0, 0.0)?;
    p.set_double(c"OfxParamPropDisplayMax", 0, 0.8)?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Transition density range for compensation removal (typ. 0.1-0.5).",
    );
    p.set_string(c"OfxParamPropParent", 0, c"GlareGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeChoice", c"ReferenceIlluminant")?;
    p.set_string(c"OfxPropLabel", 0, c"Reference illuminant")?;
    option(&p, c"D65")?;
    option(&p, c"D55")?;
    option(&p, c"D50")?;
    option(&p, c"TH-KG3-L")?;
    option(&p, c"T")?;
    option(&p, c"K75P")?;
    option(&p, c"Equal energy")?;
    p.set_int(c"OfxParamPropDefault", 0, 0)?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeChoice", c"EnlargerIlluminant")?;
    p.set_string(c"OfxPropLabel", 0, c"Enlarger illuminant")?;
    option(&p, c"D65")?;
    option(&p, c"D55")?;
    option(&p, c"D50")?;
    option(&p, c"TH-KG3")?;
    option(&p, c"TH-KG3-L")?;
    option(&p, c"T")?;
    option(&p, c"K75P")?;
    option(&p, c"Equal energy")?;
    p.set_int(c"OfxParamPropDefault", 0, 3)?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeGroup", c"OutputEncodingGroup")?;
    p.set_string(c"OfxPropLabel", 0, c"Output encoding")?;
    let p = parameters.define(c"OfxParamTypeChoice", c"OutputColorSpace")?;
    p.set_string(c"OfxPropLabel", 0, c"Output color space")?;
    option(&p, c"sRGB")?;
    option(&p, c"DCI-P3")?;
    option(&p, c"Display P3")?;
    option(&p, c"Adobe RGB (1998)")?;
    option(&p, c"ITU-R BT.2020")?;
    option(&p, c"ProPhoto RGB")?;
    option(&p, c"ACES2065-1")?;
    option(&p, c"DaVinci Wide Gamut Intermediate")?;
    option(&p, c"Rec. 709")?;
    p.set_int(c"OfxParamPropDefault", 0, 0)?;
    p.set_string(c"OfxParamPropParent", 0, c"OutputEncodingGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeBoolean", c"OutputCctfEncoding")?;
    p.set_string(c"OfxPropLabel", 0, c"Apply output CCTF")?;
    let _ = p.set_string(c"OfxParamPropHint", 0, c"Disable to output linear RGB in the selected output color space. Output remains clipped to 0-1.");
    p.set_int(c"OfxParamPropDefault", 0, 1)?;
    p.set_string(c"OfxParamPropParent", 0, c"OutputEncodingGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    let p = parameters.define(c"OfxParamTypeBoolean", c"OutputGamutCompression")?;
    p.set_string(c"OfxPropLabel", 0, c"Output gamut compression")?;
    let _ = p.set_string(
        c"OfxParamPropHint",
        0,
        c"Compress destination-linear highlights and chroma before scanner optics.",
    );
    p.set_int(c"OfxParamPropDefault", 0, 1)?;
    p.set_string(c"OfxParamPropParent", 0, c"OutputEncodingGroup")?;
    p.set_int(c"OfxParamPropEvaluateOnChange", 0, 1)?;
    Ok(())
}
