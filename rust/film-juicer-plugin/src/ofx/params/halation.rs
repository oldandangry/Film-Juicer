//! Halation and highlight reconstruction descriptors.

use super::*;
use super::Property::*;

use film_juicer_core::optics::{
    HALATION_MAX_AUTHORED, HALATION_MAX_BOUNCES, HalationShape, highlight_boost,
};

pub(super) fn describe_advanced(parameters: &Parameters<'_>) -> Result<(), Status> {
    let defaults = HalationShape::default();
    let group = c"HalationAdvancedGroup";
    let p = parameters.define(ofx_str(sys::kOfxParamTypeGroup), group)?;
    p.set_string(ofx_str(sys::kOfxPropLabel), 0, c"Advanced Halation")?;
    let _ = p.set_int(ofx_str(sys::kOfxParamPropGroupOpen), 0, 0);
    p.set_string(ofx_str(sys::kOfxParamPropParent), 0, c"HalationGroup")?;

    for (name, label, hint, values, maximum, display_max, increment) in [
        (
            ParamId::HalationScatterCoreSigmaUm.name(),
            c"Scatter Core σ (µm)",
            c"Gaussian sigma of fine in-emulsion scatter for each RGB channel.",
            defaults.scatter_core_um,
            HALATION_MAX_AUTHORED,
            10.0,
            0.5,
        ),
        (
            ParamId::HalationScatterTailScaleUm.name(),
            c"Scatter Tail Scale (µm)",
            c"Exponential decay length of the broad in-emulsion scatter tail for each RGB channel.",
            defaults.scatter_tail_um,
            HALATION_MAX_AUTHORED,
            40.0,
            1.0,
        ),
        (
            ParamId::HalationScatterTailWeight.name(),
            c"Scatter Tail Weight",
            c"Fraction of scattered light in the broad tail for each RGB channel. The core receives the remaining fraction.",
            defaults.scatter_tail_weight,
            1.0,
            1.0,
            0.01,
        ),
    ] {
        let p = parameters.define(ofx_str(sys::kOfxParamTypeDouble3D), name)?;
        describe_halation_rgb(&p, label, hint, values, maximum, display_max, increment)?;
    }

    let p = parameters.define(
        ofx_str(sys::kOfxParamTypeChoice),
        ParamId::HalationReflectionSource.name(),
    )?;
    p.set_string(ofx_str(sys::kOfxPropLabel), 0, c"Reflection Source")?;
    let _ = p.set_string(ofx_str(sys::kOfxParamPropHint), 0, c"Profile follows the selected stock. Custom uses saved absolute RGB strength and width, initially set to the schema defaults. Switching to Custom can change the result immediately.");
    option(&p, c"Profile")?;
    option(&p, c"Custom")?;
    p.set_int(ofx_str(sys::kOfxParamPropDefault), 0, 0)?;
    p.set_string(ofx_str(sys::kOfxParamPropParent), 0, group)?;
    p.set_int(ofx_str(sys::kOfxParamPropEvaluateOnChange), 0, 1)?;

    let p = parameters.define(
        ofx_str(sys::kOfxParamTypeDouble3D),
        ParamId::HalationStrengthRgb.name(),
    )?;
    describe_halation_rgb(
        &p,
        c"Halation Strength",
        c"Custom total back-reflection strength relative to incident exposure for each RGB channel, before the Halation amount multiplier.",
        [0.05, 0.015, 0.0],
        HALATION_MAX_AUTHORED,
        0.5,
        0.005,
    )?;
    let p = parameters.define(
        ofx_str(sys::kOfxParamTypeDouble3D),
        ParamId::HalationFirstBounceSigmaUm.name(),
    )?;
    describe_halation_rgb(
        &p,
        c"First-Bounce σ (µm)",
        c"Custom first-bounce Gaussian sigma for each RGB channel. Later bounces use widths proportional to the square root of the bounce number.",
        [65.0; 3],
        HALATION_MAX_AUTHORED,
        150.0,
        1.0,
    )?;

    let p = parameters.define(
        ofx_str(sys::kOfxParamTypeInteger),
        ParamId::HalationBounceCount.name(),
    )?;
    p.set_string(ofx_str(sys::kOfxPropLabel), 0, c"Bounces")?;
    let _ = p.set_string(ofx_str(sys::kOfxParamPropHint), 0, c"Number of Gaussian bounces used for back-reflection. Advanced Gaussian widths can be at most 1024 pixels at the rendered frame size.");
    p.set_int(ofx_str(sys::kOfxParamPropDefault), 0, defaults.bounce_count)?;
    p.set_int(ofx_str(sys::kOfxParamPropMin), 0, 1)?;
    p.set_int(
        ofx_str(sys::kOfxParamPropMax),
        0,
        HALATION_MAX_BOUNCES as i32,
    )?;
    p.set_int(ofx_str(sys::kOfxParamPropDisplayMin), 0, 1)?;
    p.set_int(
        ofx_str(sys::kOfxParamPropDisplayMax),
        0,
        HALATION_MAX_BOUNCES as i32,
    )?;
    p.set_string(ofx_str(sys::kOfxParamPropParent), 0, group)?;
    p.set_int(ofx_str(sys::kOfxParamPropEvaluateOnChange), 0, 1)?;

    let p = parameters.define(
        ofx_str(sys::kOfxParamTypeDouble),
        ParamId::HalationBounceDecay.name(),
    )?;
    p.set_string(ofx_str(sys::kOfxPropLabel), 0, c"Bounce Decay")?;
    let _ = p.set_string(ofx_str(sys::kOfxParamPropHint), 0, c"Amplitude ratio between consecutive back-reflection bounces. Zero uses only the first bounce. One weights all bounces equally.");
    p.set_double(ofx_str(sys::kOfxParamPropDefault), 0, defaults.bounce_decay)?;
    p.set_double(ofx_str(sys::kOfxParamPropMin), 0, 0.0)?;
    p.set_double(ofx_str(sys::kOfxParamPropMax), 0, 1.0)?;
    p.set_double(ofx_str(sys::kOfxParamPropDisplayMin), 0, 0.0)?;
    p.set_double(ofx_str(sys::kOfxParamPropDisplayMax), 0, 1.0)?;
    p.set_double(ofx_str(sys::kOfxParamPropIncrement), 0, 0.05)?;
    p.set_string(ofx_str(sys::kOfxParamPropParent), 0, group)?;
    p.set_int(ofx_str(sys::kOfxParamPropEvaluateOnChange), 0, 1)?;

    let p = parameters.define(
        ofx_str(sys::kOfxParamTypeBoolean),
        ParamId::HalationRenormalize.name(),
    )?;
    p.set_string(ofx_str(sys::kOfxPropLabel), 0, c"Renormalize")?;
    let _ = p.set_string(ofx_str(sys::kOfxParamPropHint), 0, c"Preserve uniform exposure by dividing by one plus the reflection strength in each channel. Turn off for additive reflection.");
    p.set_int(
        ofx_str(sys::kOfxParamPropDefault),
        0,
        i32::from(defaults.renormalize),
    )?;
    p.set_string(ofx_str(sys::kOfxParamPropParent), 0, group)?;
    p.set_int(ofx_str(sys::kOfxParamPropEvaluateOnChange), 0, 1)?;

    let highlight_group = c"HalationHighlightGroup";
    let p = parameters.define(ofx_str(sys::kOfxParamTypeGroup), highlight_group)?;
    p.set_string(ofx_str(sys::kOfxPropLabel), 0, c"Highlight Reconstruction")?;
    p.set_string(ofx_str(sys::kOfxParamPropParent), 0, group)?;
    let boost = highlight_boost::Controls::default();
    for (name, label, hint, default, maximum, display_max, increment) in [
        (
            ParamId::HalationHighlightBoostEv.name(),
            c"Highlight Boost (EV)",
            c"Maximum highlight boost in stops. Uses the brightest film-linear sample in the rendered frame before camera diffusion.",
            boost.boost_ev,
            highlight_boost::MAX_EV,
            8.0,
            0.5,
        ),
        (
            ParamId::HalationHighlightProtectionEv.name(),
            c"Highlight Protection (EV)",
            c"Sets the onset of reconstruction in stops above a film-linear exposure of 0.184.",
            boost.protection_ev,
            highlight_boost::MAX_EV,
            10.0,
            0.5,
        ),
        (
            ParamId::HalationHighlightBoostRange.name(),
            c"Highlight Boost Range",
            c"Shape of the highlight response. Zero concentrates the boost near the peak. One spreads it more broadly above the onset.",
            boost.range,
            1.0,
            1.0,
            0.05,
        ),
    ] {
        let p = parameters.define(ofx_str(sys::kOfxParamTypeDouble), name)?;
        p.set_string(ofx_str(sys::kOfxPropLabel), 0, label)?;
        let _ = p.set_string(ofx_str(sys::kOfxParamPropHint), 0, hint);
        p.set_double(ofx_str(sys::kOfxParamPropDefault), 0, default)?;
        p.set_double(ofx_str(sys::kOfxParamPropMin), 0, 0.0)?;
        p.set_double(ofx_str(sys::kOfxParamPropMax), 0, maximum)?;
        p.set_double(ofx_str(sys::kOfxParamPropDisplayMin), 0, 0.0)?;
        p.set_double(ofx_str(sys::kOfxParamPropDisplayMax), 0, display_max)?;
        p.set_double(ofx_str(sys::kOfxParamPropIncrement), 0, increment)?;
        p.set_string(ofx_str(sys::kOfxParamPropParent), 0, highlight_group)?;
        p.set_int(ofx_str(sys::kOfxParamPropEvaluateOnChange), 0, 1)?;
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
    p.set_string(ofx_str(sys::kOfxPropLabel), 0, label)?;
    let _ = p.set_string(ofx_str(sys::kOfxParamPropHint), 0, hint);
    for (index, value, dimension) in [
        (0, values[0], c"R"),
        (1, values[1], c"G"),
        (2, values[2], c"B"),
    ] {
        p.set_double(ofx_str(sys::kOfxParamPropDefault), index, value)?;
        p.set_double(ofx_str(sys::kOfxParamPropMin), index, 0.0)?;
        p.set_double(ofx_str(sys::kOfxParamPropMax), index, maximum)?;
        p.set_double(ofx_str(sys::kOfxParamPropDisplayMin), index, 0.0)?;
        p.set_double(ofx_str(sys::kOfxParamPropDisplayMax), index, display_max)?;
        p.set_string(ofx_str(sys::kOfxParamPropDimensionLabel), index, dimension)?;
    }
    p.set_double(ofx_str(sys::kOfxParamPropIncrement), 0, increment)?;
    p.set_string(
        ofx_str(sys::kOfxParamPropParent),
        0,
        c"HalationAdvancedGroup",
    )?;
    p.set_int(ofx_str(sys::kOfxParamPropEvaluateOnChange), 0, 1)
}

pub(super) const HALATION: &[Descriptor] = &[
    Descriptor::group(c"HalationGroup", c"Halation", &[Closed]),
    Descriptor::parameter(
        ParamId::HalationActive,
        c"Add halation",
        &[
            Int(DEFAULT, 1),
            Hint(c"Enable in-emulsion scatter and back-reflection in the capture film."),
            Parent(c"HalationGroup"),
            Int(EVALUATE_ON_CHANGE, 1),
        ],
    ),
    Descriptor::parameter(
        ParamId::HalationScatterAmount,
        c"Scatter amount",
        &[
            Double(DEFAULT, &[1.0]),
            Double(MIN, &[0.0]),
            Double(MAX, &[2.0]),
            Double(DISPLAY_MIN, &[0.0]),
            Double(DISPLAY_MAX, &[2.0]),
            Double(INCREMENT, &[0.05]),
            Hint(c"Scale in-emulsion scatter strength in the capture film."),
            Parent(c"HalationGroup"),
            Int(EVALUATE_ON_CHANGE, 1),
        ],
    ),
    Descriptor::parameter(
        ParamId::HalationScatterSpatialScale,
        c"Scatter spatial scale",
        &[
            Double(DEFAULT, &[1.0]),
            Double(MIN, &[0.0]),
            Double(MAX, &[2.0]),
            Double(DISPLAY_MIN, &[0.0]),
            Double(DISPLAY_MAX, &[2.0]),
            Double(INCREMENT, &[0.1]),
            Hint(c"Scale the in-emulsion scatter radius in the capture film."),
            Parent(c"HalationGroup"),
            Int(EVALUATE_ON_CHANGE, 1),
        ],
    ),
    Descriptor::parameter(
        ParamId::HalationAmount,
        c"Halation amount",
        &[
            Double(DEFAULT, &[1.0]),
            Double(MIN, &[0.0]),
            Double(MAX, &[2.0]),
            Double(DISPLAY_MIN, &[0.0]),
            Double(DISPLAY_MAX, &[2.0]),
            Double(INCREMENT, &[0.05]),
            Hint(c"Scale profile-derived back-reflection strength in the capture film."),
            Parent(c"HalationGroup"),
            Int(EVALUATE_ON_CHANGE, 1),
        ],
    ),
    Descriptor::parameter(
        ParamId::HalationSpatialScale,
        c"Halation spatial scale",
        &[
            Double(DEFAULT, &[1.0]),
            Double(MIN, &[0.0]),
            Double(MAX, &[2.0]),
            Double(DISPLAY_MIN, &[0.0]),
            Double(DISPLAY_MAX, &[2.0]),
            Double(INCREMENT, &[0.1]),
            Hint(c"Scale the profile-derived back-reflection radius in the capture film."),
            Parent(c"HalationGroup"),
            Int(EVALUATE_ON_CHANGE, 1),
        ],
    ),
];
