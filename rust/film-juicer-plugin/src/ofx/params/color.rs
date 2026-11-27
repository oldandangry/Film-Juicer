//! Input, print, glare and output descriptors.

use super::*;
use super::Property::*;

pub(super) const INPUT: &[Descriptor] = &[
Descriptor::group(c"InputGroup", c"Input", &[
]),
Descriptor::parameter(ParamId::Exposure, c"Exposure Compensation Ev", &[
    Hint(c"Camera exposure compensation in EV stops. Per agx-emulsion parity: this is camera.exposure_compensation_ev, applied as 2^EV multiplication to film exposure. Positive values brighten the image."),
    Double(DEFAULT, &[0.0]),
    Double(MIN, &[-8.0]),
    Double(MAX, &[8.0]),
    Double(DISPLAY_MIN, &[-4.0]),
    Double(DISPLAY_MAX, &[4.0]),
    Parent(c"InputGroup"),
]),
Descriptor::parameter(ParamId::CameraMeteringMethod, c"Camera metering", &[
    Choices(&[c"Center-weighted", c"Average", c"Median", c"Partial", c"Matrix", c"Multi-zone", c"Highlight-weighted"]),
    Int(DEFAULT, 0),
    Parent(c"InputGroup"),
    Int(EVALUATE_ON_CHANGE, 1),
]),
Descriptor::parameter(ParamId::CameraAutoExposure, c"Camera auto exposure", &[
    Hint(c"Enable the camera auto meter (agx-emulsion camera.auto_exposure). Scanner auto exposure remains independent."),
    Int(DEFAULT, 1),
    Parent(c"InputGroup"),
    Int(EVALUATE_ON_CHANGE, 1),
]),
Descriptor::parameter(ParamId::CameraFilmFormatPreset, c"Camera film format preset", &[
    Choices(&[c"Custom", c"APS-C film (25 mm)", c"35 mm / 135 (36 mm)", c"645 / 6x4.5 (56 mm)", c"6x7 (70 mm)", c"4x5 (120 mm)", c"Super 16 (11.66 mm)", c"35mm Academy (21.95 mm)", c"Super 35 (24.89 mm)", c"VistaVision 8-perf (37.72 mm)", c"IMAX/65mm (70.4 mm)"]),
    Int(DEFAULT, 0),
    Hint(c"Choose a film format preset; the selected long edge is written to the editable control below."),
    Parent(c"InputGroup"),
    Int(EVALUATE_ON_CHANGE, 1),
]),
Descriptor::parameter(ParamId::CameraFilmFormatMm, c"Camera film format (mm)", &[
    Double(DEFAULT, &[35.0]),
    Double(MIN, &[8.0]),
    Double(MAX, &[120.0]),
    Double(DISPLAY_MIN, &[8.0]),
    Double(DISPLAY_MAX, &[120.0]),
    Double(INCREMENT, &[1.0]),
    Hint(c"Longest capture dimension in millimeters; used for micrometer-to-pixel conversions."),
    Parent(c"InputGroup"),
    Int(EVALUATE_ON_CHANGE, 1),
]),
];

pub(super) const INPUT_COLOR: &[Descriptor] = &[
Descriptor::parameter(ParamId::FilmProfileKey, c"Film stock", &[
    Profiles(Role::Film),
    Text(DEFAULT, &[c"kodak_portra_400"]),
    Parent(c"InputGroup"),
    Int(EVALUATE_ON_CHANGE, 1),
]),
Descriptor::parameter(ParamId::SpectralUpsampling, c"Spectral upsampling", &[
    Choices(&[c"Hanatos", c"Mallett", c"Arctic"]),
    Hint(c"Choose the spectral reconstruction method used for film exposure. Hanatos uses the Hanatos 2025 LUT when available; Mallett uses the Mallett 2019 sRGB basis reconstruction; Arctic uses the Arctic 2026 beta04 reflectance LUT."),
    Int(DEFAULT, 0),
    Parent(c"InputGroup"),
    Int(EVALUATE_ON_CHANGE, 1),
]),
Descriptor::parameter(ParamId::InputColorSpace, c"Input color space", &[
    Choices(&[c"DaVinci Wide Gamut", c"ITU-R BT.2020", c"ACES2065-1", c"sRGB / Rec.709"]),
    Int(DEFAULT, 0),
    Parent(c"InputGroup"),
    Int(EVALUATE_ON_CHANGE, 1),
]),
Descriptor::parameter(ParamId::InputCctfDecoding, c"Decode input CCTF", &[
    Int(DEFAULT, 0),
    Parent(c"InputGroup"),
    Int(EVALUATE_ON_CHANGE, 1),
]),
Descriptor::group(c"InputAdvancedGroup", c"Advanced", &[
    Closed,
    Parent(c"InputGroup"),
]),
Descriptor::parameter(ParamId::InputCompression, c"Input gamut compression", &[
    Hint(c"Compress out-of-domain input chromaticities before TC spectral reconstruction."),
    Int(DEFAULT, 1),
    Parent(c"InputAdvancedGroup"),
    Int(EVALUATE_ON_CHANGE, 1),
]),
Descriptor::parameter(ParamId::Hanatos2025AdaptationWindow, c"hanatos2025 adaptation window", &[
    Hint(c"Apply the Hanatos 2025 bandpass adaptation window when reconstructing spectra."),
    Int(DEFAULT, 1),
    Parent(c"InputAdvancedGroup"),
    Int(EVALUATE_ON_CHANGE, 1),
]),
Descriptor::parameter(ParamId::Hanatos2025AdaptationSurface, c"hanatos2025 adaptation surface", &[
    Hint(c"Apply the Hanatos 2025 surface adaptation polynomial when reconstructing spectra."),
    Int(DEFAULT, 0),
    Parent(c"InputAdvancedGroup"),
    Int(EVALUATE_ON_CHANGE, 1),
]),
];

pub(super) const PRINT: &[Descriptor] = &[
Descriptor::group(c"PrintGroup", c"Print", &[
]),
Descriptor::parameter(ParamId::PrintProfileKey, c"Print paper", &[
    Profiles(Role::Print),
    Text(DEFAULT, &[c"kodak_portra_endura"]),
    Parent(c"PrintGroup"),
    Int(EVALUATE_ON_CHANGE, 1),
]),
Descriptor::parameter(ParamId::ScanRoute, c"Scan route", &[
    StringChoices(&[(c"NegativeDirectScan", c"Negative direct scan"), (c"NegativePrintScan", c"Negative print scan"), (c"PositiveDirectScan", c"Positive direct scan"), (c"PositivePrintScan", c"Positive print scan")]),
    Text(DEFAULT, &[c"NegativePrintScan"]),
    Parent(c"PrintGroup"),
    Int(EVALUATE_ON_CHANGE, 1),
]),
Descriptor::parameter(ParamId::PrintExposure, c"Print exposure", &[
    Double(DEFAULT, &[1.0]),
    Double(DISPLAY_MIN, &[0.1]),
    Double(DISPLAY_MAX, &[10.0]),
    Parent(c"PrintGroup"),
]),
Descriptor::parameter(ParamId::PrintPreflash, c"Print preflash", &[
    Double(DEFAULT, &[0.0]),
    Double(DISPLAY_MIN, &[0.0]),
    Double(DISPLAY_MAX, &[1.0]),
    Parent(c"PrintGroup"),
]),
Descriptor::parameter(ParamId::PrintExposureCompensation, c"Print exposure compensation", &[
    Int(DEFAULT, 1),
    Parent(c"PrintGroup"),
    Int(EVALUATE_ON_CHANGE, 1),
]),
Descriptor::parameter(ParamId::EnlargerY, c"Enlarger Y offset (Kodak CC)", &[
    Double(DEFAULT, &[0.0]),
    Double(DISPLAY_MIN, &[-200.0]),
    Double(DISPLAY_MAX, &[200.0]),
    Double(INCREMENT, &[1.0]),
    Parent(c"PrintGroup"),
]),
Descriptor::parameter(ParamId::EnlargerM, c"Enlarger M offset (Kodak CC)", &[
    Double(DEFAULT, &[0.0]),
    Double(DISPLAY_MIN, &[-200.0]),
    Double(DISPLAY_MAX, &[200.0]),
    Double(INCREMENT, &[1.0]),
    Parent(c"PrintGroup"),
]),
Descriptor::parameter(ParamId::EnlargerC, c"Enlarger C offset (Kodak CC)", &[
    Hint(c"User cyan offset carried as FilmJuicerMainCFilterShift at the print recipe boundary."),
    Double(DEFAULT, &[0.0]),
    Double(DISPLAY_MIN, &[-200.0]),
    Double(DISPLAY_MAX, &[200.0]),
    Double(INCREMENT, &[1.0]),
    Parent(c"PrintGroup"),
]),
];

pub(super) const GLARE: &[Descriptor] = &[
Descriptor::group(c"GlareGroup", c"Glare", &[
    Closed,
]),
Descriptor::parameter(ParamId::GlareActive, c"Add glare", &[
    Int(DEFAULT, 1),
    Hint(c"Add glare to the print (scanner-stage stray light)."),
    Parent(c"GlareGroup"),
    Int(EVALUATE_ON_CHANGE, 1),
]),
Descriptor::parameter(ParamId::GlarePercent, c"Glare percent", &[
    Double(DEFAULT, &[0.03]),
    Double(MIN, &[0.0]),
    Double(MAX, &[1.0]),
    Double(DISPLAY_MIN, &[0.0]),
    Double(DISPLAY_MAX, &[0.5]),
    Double(INCREMENT, &[0.05]),
    Hint(c"Percentage of glare light (typ. 0.10-0.25). Value is in percent, not fraction."),
    Parent(c"GlareGroup"),
    Int(EVALUATE_ON_CHANGE, 1),
]),
Descriptor::parameter(ParamId::GlareRoughness, c"Glare roughness", &[
    Double(DEFAULT, &[0.7]),
    Double(MIN, &[0.0]),
    Double(MAX, &[1.0]),
    Double(DISPLAY_MIN, &[0.0]),
    Double(DISPLAY_MAX, &[1.0]),
    Hint(c"Glare roughness (0-1). Stddev = roughness * percent."),
    Parent(c"GlareGroup"),
    Int(EVALUATE_ON_CHANGE, 1),
]),
Descriptor::parameter(ParamId::GlareBlurSigmaPx, c"Glare blur sigma (px)", &[
    Double(DEFAULT, &[0.5]),
    Double(MIN, &[0.0]),
    Double(MAX, &[10.0]),
    Double(DISPLAY_MIN, &[0.0]),
    Double(DISPLAY_MAX, &[3.0]),
    Hint(c"Gaussian blur sigma in pixels applied to the glare field."),
    Parent(c"GlareGroup"),
    Int(EVALUATE_ON_CHANGE, 1),
]),
Descriptor::parameter(ParamId::PrintShadowCompensationFactor, c"Compensation removal factor", &[
    Double(DEFAULT, &[0.0]),
    Double(MIN, &[0.0]),
    Double(MAX, &[1.0]),
    Double(DISPLAY_MIN, &[0.0]),
    Double(DISPLAY_MAX, &[0.2]),
    Double(INCREMENT, &[0.05]),
    Hint(c"Apply print shadow compensation to density curves. 0.2 = 20% underexposed shadows. Intended as alternative to stochastic glare (set GlarePercent=0)."),
    Parent(c"GlareGroup"),
    Int(EVALUATE_ON_CHANGE, 1),
]),
Descriptor::parameter(ParamId::PrintShadowCompensationDensity, c"Compensation removal density", &[
    Double(DEFAULT, &[1.2]),
    Double(MIN, &[0.0]),
    Double(MAX, &[3.0]),
    Double(DISPLAY_MIN, &[0.8]),
    Double(DISPLAY_MAX, &[2.0]),
    Hint(c"Density at which the compensation-removal transition is centered (typ. 1.0-1.5)."),
    Parent(c"GlareGroup"),
    Int(EVALUATE_ON_CHANGE, 1),
]),
Descriptor::parameter(ParamId::PrintShadowCompensationTransition, c"Compensation removal transition", &[
    Double(DEFAULT, &[0.3]),
    Double(MIN, &[0.0]),
    Double(MAX, &[2.0]),
    Double(DISPLAY_MIN, &[0.0]),
    Double(DISPLAY_MAX, &[0.8]),
    Hint(c"Transition density range for compensation removal (typ. 0.1-0.5)."),
    Parent(c"GlareGroup"),
    Int(EVALUATE_ON_CHANGE, 1),
]),
Descriptor::parameter(ParamId::ReferenceIlluminant, c"Reference illuminant", &[
    Choices(&[c"D65", c"D55", c"D50", c"TH-KG3-L", c"T", c"K75P", c"Equal energy"]),
    Int(DEFAULT, 0),
    Int(EVALUATE_ON_CHANGE, 1),
]),
Descriptor::parameter(ParamId::EnlargerIlluminant, c"Enlarger illuminant", &[
    Choices(&[c"D65", c"D55", c"D50", c"TH-KG3", c"TH-KG3-L", c"T", c"K75P", c"Equal energy"]),
    Int(DEFAULT, 3),
    Int(EVALUATE_ON_CHANGE, 1),
]),
];

pub(super) const OUTPUT: &[Descriptor] = &[
Descriptor::group(c"OutputEncodingGroup", c"Output encoding", &[
]),
Descriptor::parameter(ParamId::OutputColorSpace, c"Output color space", &[
    Choices(&[c"sRGB", c"DCI-P3", c"Display P3", c"Adobe RGB (1998)", c"ITU-R BT.2020", c"ProPhoto RGB", c"ACES2065-1", c"DaVinci Wide Gamut Intermediate", c"Rec. 709"]),
    Int(DEFAULT, 0),
    Parent(c"OutputEncodingGroup"),
    Int(EVALUATE_ON_CHANGE, 1),
]),
Descriptor::parameter(ParamId::OutputCctfEncoding, c"Apply output CCTF", &[
    Hint(c"Disable to output linear RGB in the selected output color space. Output remains clipped to 0-1."),
    Int(DEFAULT, 1),
    Parent(c"OutputEncodingGroup"),
    Int(EVALUATE_ON_CHANGE, 1),
]),
Descriptor::parameter(ParamId::OutputGamutCompression, c"Output gamut compression", &[
    Hint(c"Compress destination-linear highlights and chroma before scanner optics."),
    Int(DEFAULT, 1),
    Parent(c"OutputEncodingGroup"),
    Int(EVALUATE_ON_CHANGE, 1),
]),
];
