//! Saved OFX IDs and constructor acquisition order for instance controls.

use std::ffi::CStr;
use super::{ofx_str, sys};

#[derive(Clone, Copy, PartialEq)]
pub(super) enum Kind {
    Double,
    Boolean,
    Choice,
    StringChoice,
    Integer,
    Double2D,
    Double3D,
    PushButton,
    Group,
}

impl Kind {
    pub(super) const fn ofx(self) -> &'static CStr {
        match self {
            Self::Double => ofx_str(sys::kOfxParamTypeDouble),
            Self::Boolean => ofx_str(sys::kOfxParamTypeBoolean),
            Self::Choice => ofx_str(sys::kOfxParamTypeChoice),
            Self::StringChoice => ofx_str(sys::kOfxParamTypeStrChoice),
            Self::Integer => ofx_str(sys::kOfxParamTypeInteger),
            Self::Double2D => ofx_str(sys::kOfxParamTypeDouble2D),
            Self::Double3D => ofx_str(sys::kOfxParamTypeDouble3D),
            Self::PushButton => ofx_str(sys::kOfxParamTypePushButton),
            Self::Group => ofx_str(sys::kOfxParamTypeGroup),
        }
    }
}

macro_rules! parameters {
    ($($id:ident: $kind:ident => $name:literal),+ $(,)?) => {
        #[derive(Clone, Copy)]
        pub(crate) enum ParamId { $($id),+ }

        impl ParamId {
            pub(super) const ALL: &[Self] = &[$(Self::$id),+];

            pub(super) const fn name(self) -> &'static CStr {
                match self { $(Self::$id => $name),+ }
            }

            pub(super) const fn kind(self) -> Kind {
                match self { $(Self::$id => Kind::$kind),+ }
            }
        }
    };
}

parameters! {
    Exposure: Double => c"Exposure",
    CameraAutoExposure: Boolean => c"CameraAutoExposure",
    CameraFilmFormatPreset: Choice => c"CameraFilmFormatPreset",
    CameraFilmFormatMm: Double => c"CameraFilmFormatMm",
    CameraMeteringMethod: Choice => c"CameraMeteringMethod",
    FilmProfileKey: StringChoice => c"FilmProfileKey",
    SpectralUpsampling: Choice => c"SpectralUpsampling",
    PrintProfileKey: StringChoice => c"PrintProfileKey",
    ReferenceIlluminant: Choice => c"ReferenceIlluminant",
    EnlargerIlluminant: Choice => c"EnlargerIlluminant",
    InputColorSpace: Choice => c"InputColorSpace",
    InputCctfDecoding: Boolean => c"InputCctfDecoding",
    InputCompression: Boolean => c"InputCompression",
    Hanatos2025AdaptationWindow: Boolean => c"Hanatos2025AdaptationWindow",
    Hanatos2025AdaptationSurface: Boolean => c"Hanatos2025AdaptationSurface",
    ScanRoute: StringChoice => c"ScanRoute",
    FilmGammaFactor: Double => c"FilmGammaFactor",
    PrintGammaFactor: Double => c"PrintGammaFactor",
    OutputColorSpace: Choice => c"OutputColorSpace",
    OutputCctfEncoding: Boolean => c"OutputCctfEncoding",
    OutputGamutCompression: Boolean => c"OutputGamutCompression",
    CouplersActive: Boolean => c"CouplersActive",
    CouplersAmount: Double => c"CouplersAmount",
    CouplersInhibitionSameLayer: Double => c"CouplersInhibitionSameLayer",
    CouplersInhibitionInterlayer: Double => c"CouplersInhibitionInterlayer",
    CouplersDiffusionSizeUm: Double => c"CouplersDiffusionSizeUm",
    CouplersLangmuirDonorKRgb: Double3D => c"CouplersLangmuirDonorKRgb",
    CouplersLangmuirReceiverKRgb: Double3D => c"CouplersLangmuirReceiverKRgb",
    CouplersDiffusionTailUm: Double => c"CouplersDiffusionTailUm",
    CouplersDiffusionTailWeight: Double => c"CouplersDiffusionTailWeight",
    CouplersGammaUseStock: Boolean => c"CouplersGammaUseStock",
    CouplersGammaSameLayerRgb: Double3D => c"CouplersGammaSameLayerRgb",
    CouplersGammaInterlayerRToGb: Double2D => c"CouplersGammaInterlayerRToGb",
    CouplersGammaInterlayerGToRb: Double2D => c"CouplersGammaInterlayerGToRb",
    CouplersGammaInterlayerBToRg: Double2D => c"CouplersGammaInterlayerBToRg",
    ScannerLensBlurSigmaPx: Double => c"ScannerLensBlurSigmaPx",
    ScannerUnsharpMask: Double2D => c"ScannerUnsharpMask",
    ScannerBlackCorrection: Boolean => c"ScannerBlackCorrection",
    ScannerWhiteCorrection: Boolean => c"ScannerWhiteCorrection",
    ScannerBlackLevel: Double => c"ScannerBlackLevel",
    ScannerWhiteLevel: Double => c"ScannerWhiteLevel",
    PrintExposure: Double => c"PrintExposure",
    PrintPreflash: Double => c"PrintPreflash",
    PrintExposureCompensation: Boolean => c"PrintExposureCompensation",
    EnlargerY: Double => c"EnlargerY",
    EnlargerM: Double => c"EnlargerM",
    EnlargerC: Double => c"EnlargerC",
    HalationActive: Boolean => c"HalationActive",
    HalationScatterAmount: Double => c"HalationScatterAmount",
    HalationScatterSpatialScale: Double => c"HalationScatterSpatialScale",
    HalationAmount: Double => c"HalationAmount",
    HalationSpatialScale: Double => c"HalationSpatialScale",
    HalationScatterCoreSigmaUm: Double3D => c"HalationScatterCoreSigmaUm",
    HalationScatterTailScaleUm: Double3D => c"HalationScatterTailScaleUm",
    HalationScatterTailWeight: Double3D => c"HalationScatterTailWeight",
    HalationReflectionSource: Choice => c"HalationReflectionSource",
    HalationStrengthRgb: Double3D => c"HalationStrengthRgb",
    HalationFirstBounceSigmaUm: Double3D => c"HalationFirstBounceSigmaUm",
    HalationBounceCount: Integer => c"HalationBounceCount",
    HalationBounceDecay: Double => c"HalationBounceDecay",
    HalationRenormalize: Boolean => c"HalationRenormalize",
    HalationHighlightBoostEv: Double => c"HalationHighlightBoostEv",
    HalationHighlightProtectionEv: Double => c"HalationHighlightProtectionEv",
    HalationHighlightBoostRange: Double => c"HalationHighlightBoostRange",
    GrainActive: Boolean => c"GrainActive",
    GrainModel: StringChoice => c"GrainModel",
    FilmGrainAmountEv: Double => c"FilmGrainAmountEv",
    FilmGrainTexelPitchUm: Double => c"FilmGrainTexelPitchUm",
    FilmGrainChromaFraction: Double => c"FilmGrainChromaFraction",
    GrainSublayersActive: Boolean => c"GrainSublayersActive",
    GrainPreset: Choice => c"GrainPreset",
    GrainParticleAreaUm2: Double => c"GrainParticleAreaUm2",
    GrainAmplitude: Double => c"GrainAmplitude",
    GrainSharpness: Double => c"GrainSharpness",
    GrainChroma: Double => c"GrainChroma",
    GrainTexture: Double => c"GrainTexture",
    GrainParticleScaleMaster: Double => c"GrainParticleScaleMaster",
    GrainParticleScaleLayersMaster: Double => c"GrainParticleScaleLayersMaster",
    GrainDensityMinMaster: Double => c"GrainDensityMinMaster",
    GrainUniformityMaster: Double => c"GrainUniformityMaster",
    GrainParticleScale: Double3D => c"GrainParticleScale",
    GrainParticleScaleLayers: Double3D => c"GrainParticleScaleLayers",
    GrainDensityMin: Double3D => c"GrainDensityMin",
    GrainUniformity: Double3D => c"GrainUniformity",
    GrainBlur: Double => c"GrainBlur",
    GrainBlurDyeCloudsUm: Double => c"GrainBlurDyeCloudsUm",
    GrainSizeMixWeight: Double => c"GrainSizeMixWeight",
    GrainSizeMixWeightMid: Double => c"GrainSizeMixWeightMid",
    GrainSizeMixScale: Double => c"GrainSizeMixScale",
    GrainClumpTemporalMix: Double => c"GrainClumpTemporalMix",
    GrainClumpMorphPeriodSec: Double => c"GrainClumpMorphPeriodSec",
    GrainDebugView: Choice => c"GrainDebugView",
    GrainMicroStructure: Double2D => c"GrainMicroStructure",
    GrainResetAdvanced: PushButton => c"GrainResetAdvanced",
    GateWeaveAmount: Double => c"GateWeaveAmount",
    FilmDustAmount: Double => c"FilmDustAmount",
    GateDustAmount: Double => c"GateDustAmount",
    FilmScratchAmount: Double => c"FilmScratchAmount",
    GateScratchAmount: Double => c"GateScratchAmount",
    CameraDiffusionEnabled: Boolean => c"CameraDiffusionEnabled",
    CameraDiffusionFamily: Choice => c"CameraDiffusionFamily",
    CameraDiffusionStrength: Double => c"CameraDiffusionStrength",
    CameraDiffusionSpatialScale: Double => c"CameraDiffusionSpatialScale",
    CameraDiffusionHaloWarmth: Double => c"CameraDiffusionHaloWarmth",
    CameraDiffusionCoreIntensity: Double => c"CameraDiffusionCoreIntensity",
    CameraDiffusionCoreSize: Double => c"CameraDiffusionCoreSize",
    CameraDiffusionHaloIntensity: Double => c"CameraDiffusionHaloIntensity",
    CameraDiffusionHaloSize: Double => c"CameraDiffusionHaloSize",
    CameraDiffusionBloomIntensity: Double => c"CameraDiffusionBloomIntensity",
    CameraDiffusionBloomSize: Double => c"CameraDiffusionBloomSize",
    PrintDiffusionEnabled: Boolean => c"PrintDiffusionEnabled",
    PrintDiffusionFamily: Choice => c"PrintDiffusionFamily",
    PrintDiffusionStrength: Double => c"PrintDiffusionStrength",
    PrintDiffusionSpatialScale: Double => c"PrintDiffusionSpatialScale",
    PrintDiffusionHaloWarmth: Double => c"PrintDiffusionHaloWarmth",
    PrintDiffusionCoreIntensity: Double => c"PrintDiffusionCoreIntensity",
    PrintDiffusionCoreSize: Double => c"PrintDiffusionCoreSize",
    PrintDiffusionHaloIntensity: Double => c"PrintDiffusionHaloIntensity",
    PrintDiffusionHaloSize: Double => c"PrintDiffusionHaloSize",
    PrintDiffusionBloomIntensity: Double => c"PrintDiffusionBloomIntensity",
    PrintDiffusionBloomSize: Double => c"PrintDiffusionBloomSize",
    GlareActive: Boolean => c"GlareActive",
    GlarePercent: Double => c"GlarePercent",
    GlareRoughness: Double => c"GlareRoughness",
    GlareBlurSigmaPx: Double => c"GlareBlurSigmaPx",
    PrintShadowCompensationFactor: Double => c"PrintShadowCompensationFactor",
    PrintShadowCompensationDensity: Double => c"PrintShadowCompensationDensity",
    PrintShadowCompensationTransition: Double => c"PrintShadowCompensationTransition",
}
