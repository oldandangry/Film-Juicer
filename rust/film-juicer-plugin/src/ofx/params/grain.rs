//! Current and Film v1 grain descriptors.

use super::*;
use super::Property::*;

use film_juicer_core::film::grain::film_v1;

pub(super) fn describe_film(parameters: &Parameters<'_>) -> Result<(), Status> {
    for (name, label, default, range, increment) in [
        (
            ParamId::FilmGrainAmountEv,
            c"Amount (EV)",
            film_v1::DEFAULT_AMOUNT_EV,
            film_v1::AMOUNT_EV_RANGE,
            0.1,
        ),
        (
            ParamId::FilmGrainTexelPitchUm,
            c"Size (µm)",
            film_v1::DEFAULT_TEXEL_PITCH_UM,
            film_v1::TEXEL_PITCH_UM_RANGE,
            0.01,
        ),
        (
            ParamId::FilmGrainChromaFraction,
            c"Colour",
            film_v1::DEFAULT_CHROMA_FRACTION,
            film_v1::CHROMA_FRACTION_RANGE,
            0.01,
        ),
    ] {
        let p = parameters.define(name.kind().ofx(), name.name())?;
        p.set_string(ofx_str(sys::kOfxPropLabel), 0, label)?;
        let min = f64::from(*range.start());
        let max = f64::from(*range.end());
        p.set_double(ofx_str(sys::kOfxParamPropDefault), 0, f64::from(default))?;
        p.set_double(ofx_str(sys::kOfxParamPropMin), 0, min)?;
        p.set_double(ofx_str(sys::kOfxParamPropMax), 0, max)?;
        p.set_double(ofx_str(sys::kOfxParamPropDisplayMin), 0, min)?;
        p.set_double(ofx_str(sys::kOfxParamPropDisplayMax), 0, max)?;
        p.set_double(ofx_str(sys::kOfxParamPropIncrement), 0, increment)?;
        p.set_string(ofx_str(sys::kOfxParamPropParent), 0, c"GrainGroup")?;
        p.set_int(ofx_str(sys::kOfxParamPropEvaluateOnChange), 0, 1)?;
        p.set_int(ofx_str(sys::kOfxParamPropSecret), 0, 1)?;
    }
    Ok(())
}

pub(super) const GRAIN: &[Descriptor] = &[
    Descriptor::group(c"GrainGroup", c"Grain", &[Closed]),
    Descriptor::parameter(
        ParamId::GrainActive,
        c"Add grain",
        &[
            Int(DEFAULT, 0),
            Hint(c"Add grain to the negative."),
            Parent(c"GrainGroup"),
            Int(EVALUATE_ON_CHANGE, 1),
        ],
    ),
];

pub(super) const FILM_MODEL: &[Descriptor] = &[Descriptor::parameter(
    ParamId::GrainModel,
    c"Model",
    &[
        StringChoices(&[(c"current", c"Current"), (c"film_v1", c"Film v1")]),
        Text(DEFAULT, &[c"current"]),
        Parent(c"GrainGroup"),
        Int(EVALUATE_ON_CHANGE, 1),
    ],
)];

pub(super) const GRAIN_CONTROLS: &[Descriptor] = &[
    Descriptor::parameter(
        ParamId::GrainSublayersActive,
        c"Use sublayers",
        &[
            Int(DEFAULT, 1),
            Hint(c"Enable sublayer grain simulation."),
            Parent(c"GrainGroup"),
            Int(EVALUATE_ON_CHANGE, 1),
        ],
    ),
    Descriptor::parameter(
        ParamId::GrainPreset,
        c"Grain Preset",
        &[
            Choices(&[c"Fine", c"Medium", c"Coarse"]),
            Int(DEFAULT, 1),
            Hint(c"Starting point for grain size/strength presets."),
            Parent(c"GrainGroup"),
            Int(EVALUATE_ON_CHANGE, 1),
        ],
    ),
    Descriptor::parameter(
        ParamId::GrainAmplitude,
        c"Grain Amount (EV)",
        &[
            Hint(c"Grain amount in stops (EV); scales OD delta."),
            Double(DEFAULT, &[-1.20]),
            Double(MIN, &[-3.0]),
            Double(MAX, &[3.0]),
            Double(DISPLAY_MIN, &[-3.0]),
            Double(DISPLAY_MAX, &[3.0]),
            Double(INCREMENT, &[0.1]),
            Parent(c"GrainGroup"),
            Int(EVALUATE_ON_CHANGE, 1),
        ],
    ),
    Descriptor::parameter(
        ParamId::GrainBlur,
        c"Grain Size (px)",
        &[
            Hint(c"Grain correlation size in pixels."),
            Double(DEFAULT, &[0.50]),
            Double(MIN, &[0.20]),
            Double(MAX, &[2.00]),
            Double(DISPLAY_MIN, &[0.20]),
            Double(DISPLAY_MAX, &[2.00]),
            Parent(c"GrainGroup"),
            Int(EVALUATE_ON_CHANGE, 1),
        ],
    ),
    Descriptor::parameter(
        ParamId::GrainSharpness,
        c"Grain Sharpness",
        &[
            Hint(c"Controls grain edge softness."),
            Double(DEFAULT, &[0.5]),
            Double(MIN, &[0.0]),
            Double(MAX, &[1.0]),
            Double(DISPLAY_MIN, &[0.0]),
            Double(DISPLAY_MAX, &[1.0]),
            Parent(c"GrainGroup"),
            Int(EVALUATE_ON_CHANGE, 1),
        ],
    ),
    Descriptor::parameter(
        ParamId::GrainChroma,
        c"Grain Chroma",
        &[
            Hint(c"0 = achromatic, 1 = independent RGB grain."),
            Double(DEFAULT, &[0.3]),
            Double(MIN, &[0.0]),
            Double(MAX, &[1.0]),
            Double(DISPLAY_MIN, &[0.0]),
            Double(DISPLAY_MAX, &[1.0]),
            Parent(c"GrainGroup"),
            Int(EVALUATE_ON_CHANGE, 1),
        ],
    ),
    Descriptor::parameter(
        ParamId::GrainTexture,
        c"Grain Texture",
        &[
            Hint(c"Controls clumping and PSD tail."),
            Double(DEFAULT, &[0.55]),
            Double(MIN, &[0.0]),
            Double(MAX, &[1.0]),
            Double(DISPLAY_MIN, &[0.0]),
            Double(DISPLAY_MAX, &[1.0]),
            Parent(c"GrainGroup"),
            Int(EVALUATE_ON_CHANGE, 1),
        ],
    ),
    Descriptor::group(
        c"GrainAdvancedGroup",
        c"Advanced",
        &[Closed, Parent(c"GrainGroup")],
    ),
    Descriptor::parameter(
        ParamId::GrainResetAdvanced,
        c"Reset Advanced",
        &[
            Hint(c"Reset advanced grain controls to the base values."),
            Parent(c"GrainAdvancedGroup"),
        ],
    ),
    Descriptor::parameter(
        ParamId::GrainParticleAreaUm2,
        c"Particle area (um^2)",
        &[
            Hint(c"Particle area in um^2; roughly 0.1 for ISO 100-200, 0.4 for ISO 400."),
            Double(DEFAULT, &[0.25]),
            Double(MIN, &[0.0]),
            Double(MAX, &[10.0]),
            Double(DISPLAY_MIN, &[0.0]),
            Double(DISPLAY_MAX, &[1.0]),
            Double(INCREMENT, &[0.1]),
            Parent(c"GrainAdvancedGroup"),
            Int(EVALUATE_ON_CHANGE, 1),
        ],
    ),
    Descriptor::parameter(
        ParamId::GrainParticleScaleMaster,
        c"Particle scale (M)",
        &[
            Hint(c"Master control for particle scale; adjusts RGB values together."),
            Double(DEFAULT, &[1.48]),
            Double(MIN, &[0.0]),
            Double(MAX, &[10.0]),
            Double(DISPLAY_MIN, &[0.0]),
            Double(DISPLAY_MAX, &[3.0]),
            Parent(c"GrainAdvancedGroup"),
            Int(EVALUATE_ON_CHANGE, 1),
        ],
    ),
    Descriptor::parameter(
        ParamId::GrainParticleScaleLayersMaster,
        c"Particle scale layers (M)",
        &[
            Hint(c"Master control for sublayer particle scale; adjusts RGB values together."),
            Double(DEFAULT, &[1.922]),
            Double(MIN, &[0.0]),
            Double(MAX, &[10.0]),
            Double(DISPLAY_MIN, &[0.0]),
            Double(DISPLAY_MAX, &[4.0]),
            Parent(c"GrainAdvancedGroup"),
            Int(EVALUATE_ON_CHANGE, 1),
        ],
    ),
    Descriptor::parameter(
        ParamId::GrainDensityMinMaster,
        c"Density min (M)",
        &[
            Hint(c"Master control for minimum grain density; adjusts RGB values together."),
            Double(DEFAULT, &[0.08]),
            Double(MIN, &[0.0]),
            Double(MAX, &[1.0]),
            Double(DISPLAY_MIN, &[0.0]),
            Double(DISPLAY_MAX, &[0.2]),
            Parent(c"GrainAdvancedGroup"),
            Int(EVALUATE_ON_CHANGE, 1),
        ],
    ),
    Descriptor::parameter(
        ParamId::GrainUniformityMaster,
        c"Uniformity (M)",
        &[
            Hint(c"Master control for grain uniformity; adjusts RGB values together."),
            Double(DEFAULT, &[0.97]),
            Double(MIN, &[0.0]),
            Double(MAX, &[1.0]),
            Double(DISPLAY_MIN, &[0.9]),
            Double(DISPLAY_MAX, &[1.0]),
            Parent(c"GrainAdvancedGroup"),
            Int(EVALUATE_ON_CHANGE, 1),
        ],
    ),
    Descriptor::parameter(
        ParamId::GrainBlurDyeCloudsUm,
        c"Dye cloud blur scale (px)",
        &[
            Hint(c"Scale factor for dye cloud blur sigma in pixels."),
            Double(DEFAULT, &[1.0]),
            Double(MIN, &[0.0]),
            Double(MAX, &[10.0]),
            Double(DISPLAY_MIN, &[0.0]),
            Double(DISPLAY_MAX, &[3.0]),
            Parent(c"GrainAdvancedGroup"),
            Int(EVALUATE_ON_CHANGE, 1),
        ],
    ),
    Descriptor::parameter(
        ParamId::GrainSizeMixWeight,
        c"Size mix weight",
        &[
            Hint(c"Weight of the coarse grain population in the size mixture."),
            Double(DEFAULT, &[0.226]),
            Double(MIN, &[0.0]),
            Double(MAX, &[1.0]),
            Double(DISPLAY_MIN, &[0.0]),
            Double(DISPLAY_MAX, &[1.0]),
            Parent(c"GrainAdvancedGroup"),
            Int(EVALUATE_ON_CHANGE, 1),
        ],
    ),
    Descriptor::parameter(
        ParamId::GrainSizeMixWeightMid,
        c"Size mix mid weight",
        &[
            Hint(c"Weight of the mid grain population in the size mixture."),
            Double(DEFAULT, &[0.0]),
            Double(MIN, &[0.0]),
            Double(MAX, &[1.0]),
            Double(DISPLAY_MIN, &[0.0]),
            Double(DISPLAY_MAX, &[1.0]),
            Parent(c"GrainAdvancedGroup"),
            Int(EVALUATE_ON_CHANGE, 1),
        ],
    ),
    Descriptor::parameter(
        ParamId::GrainSizeMixScale,
        c"Size mix scale",
        &[
            Hint(c"Relative particle size scale for the coarse grain population."),
            Double(DEFAULT, &[19.0]),
            Double(MIN, &[1.0]),
            Double(MAX, &[50.0]),
            Double(DISPLAY_MIN, &[1.0]),
            Double(DISPLAY_MAX, &[50.0]),
            Parent(c"GrainAdvancedGroup"),
            Int(EVALUATE_ON_CHANGE, 1),
        ],
    ),
    Descriptor::parameter(
        ParamId::GrainMicroStructure,
        c"Micro-structure",
        &[
            Hint(c"Micro-structure parameters: clump cell size (um) and clump sigma (x1e-3)."),
            Double(DEFAULT, &[60.0, 170.0]),
            Double(MIN, &[0.0, 0.0]),
            Double(MAX, &[100.0, 1000.0]),
            Double(DISPLAY_MIN, &[0.0, 0.0]),
            Double(DISPLAY_MAX, &[100.0, 200.0]),
            Text(DIMENSION_LABEL, &[c"Cell (um)", c"Sigma (1e-3)"]),
            Parent(c"GrainAdvancedGroup"),
            Int(EVALUATE_ON_CHANGE, 1),
        ],
    ),
    Descriptor::parameter(
        ParamId::GrainDebugView,
        c"Grain debug view",
        &[
            Choices(&[
                c"Off",
                c"Delta mix",
                c"Delta fine",
                c"Delta coarse",
                c"Delta fine raw",
                c"Delta coarse raw",
                c"Mean density",
            ]),
            Int(DEFAULT, 0),
            Hint(c"Debug view selector for grain delta fields (pre-scanner)."),
            Parent(c"GrainAdvancedGroup"),
            Int(EVALUATE_ON_CHANGE, 1),
        ],
    ),
    Descriptor::parameter(
        ParamId::GrainParticleScale,
        c"Particle scale",
        &[
            Hint(c"Scale of particle area for the RGB layers (multiplies particle area)."),
            Double(DEFAULT, &[1.48, 1.48, 1.48]),
            Double(MIN, &[0.0, 0.0, 0.0]),
            Double(MAX, &[10.0, 10.0, 10.0]),
            Double(DISPLAY_MIN, &[0.0, 0.0, 0.0]),
            Double(DISPLAY_MAX, &[3.0, 3.0, 3.0]),
            Text(DIMENSION_LABEL, &[c"R", c"G", c"B"]),
            Int(SECRET, 1),
            Parent(c"GrainAdvancedGroup"),
            Int(EVALUATE_ON_CHANGE, 1),
        ],
    ),
    Descriptor::parameter(
        ParamId::GrainParticleScaleLayers,
        c"Particle scale layers",
        &[
            Hint(c"Scale of particle area for sublayers in each color layer."),
            Double(DEFAULT, &[1.922, 1.922, 1.922]),
            Double(MIN, &[0.0, 0.0, 0.0]),
            Double(MAX, &[10.0, 10.0, 10.0]),
            Double(DISPLAY_MIN, &[0.0, 0.0, 0.0]),
            Double(DISPLAY_MAX, &[4.0, 4.0, 4.0]),
            Text(DIMENSION_LABEL, &[c"R", c"G", c"B"]),
            Int(SECRET, 1),
            Parent(c"GrainAdvancedGroup"),
            Int(EVALUATE_ON_CHANGE, 1),
        ],
    ),
    Descriptor::parameter(
        ParamId::GrainDensityMin,
        c"Density min",
        &[
            Hint(c"Minimum grain density per layer (typ. 0.03-0.06)."),
            Double(DEFAULT, &[0.08, 0.08, 0.08]),
            Double(MIN, &[0.0, 0.0, 0.0]),
            Double(MAX, &[1.0, 1.0, 1.0]),
            Double(DISPLAY_MIN, &[0.0, 0.0, 0.0]),
            Double(DISPLAY_MAX, &[0.2, 0.2, 0.2]),
            Text(DIMENSION_LABEL, &[c"C", c"M", c"Y"]),
            Int(SECRET, 1),
            Parent(c"GrainAdvancedGroup"),
            Int(EVALUATE_ON_CHANGE, 1),
        ],
    ),
    Descriptor::parameter(
        ParamId::GrainUniformity,
        c"Uniformity",
        &[
            Hint(c"Uniformity of grain (typ. 0.94-0.98)."),
            Double(DEFAULT, &[0.97, 0.97, 0.97]),
            Double(MIN, &[0.0, 0.0, 0.0]),
            Double(MAX, &[1.0, 1.0, 1.0]),
            Double(DISPLAY_MIN, &[0.9, 0.9, 0.9]),
            Double(DISPLAY_MAX, &[1.0, 1.0, 1.0]),
            Text(DIMENSION_LABEL, &[c"C", c"M", c"Y"]),
            Int(SECRET, 1),
            Parent(c"GrainAdvancedGroup"),
            Int(EVALUATE_ON_CHANGE, 1),
        ],
    ),
    Descriptor::parameter(
        ParamId::GrainClumpTemporalMix,
        c"Clump temporal mix",
        &[
            Hint(c"Modulates clump strength over time (0 = static, higher = more breathing)."),
            Double(DEFAULT, &[0.30]),
            Double(MIN, &[0.0]),
            Double(MAX, &[0.30]),
            Double(DISPLAY_MIN, &[0.0]),
            Double(DISPLAY_MAX, &[0.30]),
            Parent(c"GrainAdvancedGroup"),
            Int(EVALUATE_ON_CHANGE, 1),
        ],
    ),
    Descriptor::parameter(
        ParamId::GrainClumpMorphPeriodSec,
        c"Clump morph period (s)",
        &[
            Hint(c"Seconds for clump field to evolve; longer is more stable."),
            Double(DEFAULT, &[8.0]),
            Double(MIN, &[5.0]),
            Double(MAX, &[60.0]),
            Double(DISPLAY_MIN, &[5.0]),
            Double(DISPLAY_MAX, &[60.0]),
            Parent(c"GrainAdvancedGroup"),
            Int(EVALUATE_ON_CHANGE, 1),
        ],
    ),
];
