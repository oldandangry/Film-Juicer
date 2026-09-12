#pragma once

#include <array>

namespace JuicerParams {
    struct CameraFilmFormatPreset {
        const char* label = nullptr;
        double longEdgeMm = 0.0;
    };

    inline constexpr const char kCameraFilmFormatPreset[] = "CameraFilmFormatPreset";
    inline constexpr const char kCameraFilmFormatMm[] = "CameraFilmFormatMm";
    inline constexpr std::array<CameraFilmFormatPreset, 10> kCameraFilmFormatPresets{{{"APS-C film (25 mm)", 25.0},
                                                                                      {"35 mm / 135 (36 mm)", 36.0},
                                                                                      {"645 / 6x4.5 (56 mm)", 56.0},
                                                                                      {"6x7 (70 mm)", 70.0},
                                                                                      {"4x5 (120 mm)", 120.0},
                                                                                      {"Super 16 (11.66 mm)", 11.66},
                                                                                      {"35mm Academy (21.95 mm)", 21.95},
                                                                                      {"Super 35 (24.89 mm)", 24.89},
                                                                                      {"VistaVision 8-perf (37.72 mm)", 37.72},
                                                                                      {"IMAX/65mm (70.4 mm)", 70.4}}};
    inline constexpr const char kCameraMeteringMethod[] = "CameraMeteringMethod";
    inline constexpr const char kFilmProfileKey[] = "FilmProfileKey";
    inline constexpr const char kPrintProfileKey[] = "PrintProfileKey";
    inline constexpr const char kParamScanRoute[] = "ScanRoute";
    inline constexpr const char kTuningGroup[] = "TuningGroup";
    inline constexpr const char kFilmGammaFactor[] = "FilmGammaFactor";
    inline constexpr const char kPrintGammaFactor[] = "PrintGammaFactor";
    inline constexpr const char kInputColorSpace[] = "InputColorSpace";
    inline constexpr const char kInputCctfDecoding[] = "InputCctfDecoding";
    inline constexpr const char kHanatos2025AdaptationWindow[] = "Hanatos2025AdaptationWindow";
    inline constexpr const char kHanatos2025AdaptationSurface[] = "Hanatos2025AdaptationSurface";
    inline constexpr const char kDirCouplersGroup[] = "Couplers";
    inline constexpr const char kDirCouplersActive[] = "CouplersActive";
    inline constexpr const char kDirCouplersAmount[] = "CouplersAmount";
    inline constexpr const char kDirCouplersInhibitionSameLayer[] = "CouplersInhibitionSameLayer";
    inline constexpr const char kDirCouplersInhibitionInterlayer[] = "CouplersInhibitionInterlayer";
    inline constexpr const char kDirCouplersDiffusionSizeUm[] = "CouplersDiffusionSizeUm";
    inline constexpr const char kDirCouplersGammaUseStock[] = "CouplersGammaUseStock";
    inline constexpr const char kDirCouplersGammaSameLayerRgb[] = "CouplersGammaSameLayerRgb";
    inline constexpr const char kDirCouplersGammaInterlayerRToGb[] = "CouplersGammaInterlayerRToGb";
    inline constexpr const char kDirCouplersGammaInterlayerGToRb[] = "CouplersGammaInterlayerGToRb";
    inline constexpr const char kDirCouplersGammaInterlayerBToRg[] = "CouplersGammaInterlayerBToRg";
    inline constexpr const char kHalationActive[] = "HalationActive";
    inline constexpr const char kHalationScatterAmount[] = "HalationScatterAmount";
    inline constexpr const char kHalationScatterSpatialScale[] = "HalationScatterSpatialScale";
    inline constexpr const char kHalationAmount[] = "HalationAmount";
    inline constexpr const char kHalationSpatialScale[] = "HalationSpatialScale";
    inline constexpr const char kGrainActive[] = "GrainActive";
    inline constexpr const char kGrainSublayersActive[] = "GrainSublayersActive";
    inline constexpr const char kGrainParticleAreaUm2[] = "GrainParticleAreaUm2";
    inline constexpr const char kGrainAmplitude[] = "GrainAmplitude";
    inline constexpr const char kGrainSharpness[] = "GrainSharpness";
    inline constexpr const char kGrainChroma[] = "GrainChroma";
    inline constexpr const char kGrainTexture[] = "GrainTexture";
    inline constexpr const char kGrainPreset[] = "GrainPreset";
    inline constexpr const char kGrainResetAdvanced[] = "GrainResetAdvanced";
    inline constexpr const char kGrainParticleScaleMaster[] = "GrainParticleScaleMaster";
    inline constexpr const char kGrainParticleScaleLayersMaster[] = "GrainParticleScaleLayersMaster";
    inline constexpr const char kGrainDensityMinMaster[] = "GrainDensityMinMaster";
    inline constexpr const char kGrainUniformityMaster[] = "GrainUniformityMaster";
    inline constexpr const char kGrainParticleScale[] = "GrainParticleScale";
    inline constexpr const char kGrainParticleScaleLayers[] = "GrainParticleScaleLayers";
    inline constexpr const char kGrainDensityMin[] = "GrainDensityMin";
    inline constexpr const char kGrainUniformity[] = "GrainUniformity";
    inline constexpr const char kGrainBlur[] = "GrainBlur";
    inline constexpr const char kGrainBlurDyeCloudsUm[] = "GrainBlurDyeCloudsUm";
    inline constexpr const char kGrainSizeMixWeight[] = "GrainSizeMixWeight";
    inline constexpr const char kGrainSizeMixWeightMid[] = "GrainSizeMixWeightMid";
    inline constexpr const char kGrainSizeMixScale[] = "GrainSizeMixScale";
    inline constexpr const char kGrainClumpTemporalMix[] = "GrainClumpTemporalMix";
    inline constexpr const char kGrainClumpMorphPeriodSec[] = "GrainClumpMorphPeriodSec";
    inline constexpr const char kGrainDebugView[] = "GrainDebugView";
    inline constexpr const char kGrainMicroStructure[] = "GrainMicroStructure";
    inline constexpr const char kGateWeaveAmount[] = "GateWeaveAmount";
    inline constexpr const char kFilmDustAmount[] = "FilmDustAmount";
    inline constexpr const char kGateDustAmount[] = "GateDustAmount";
    inline constexpr const char kFilmScratchAmount[] = "FilmScratchAmount";
    inline constexpr const char kGateScratchAmount[] = "GateScratchAmount";
    inline constexpr const char kDiffusionGroup[] = "DiffusionGroup";
    inline constexpr const char kCameraDiffusionEnabled[] = "CameraDiffusionEnabled";
    inline constexpr const char kCameraDiffusionFamily[] = "CameraDiffusionFamily";
    inline constexpr const char kCameraDiffusionStrength[] = "CameraDiffusionStrength";
    inline constexpr const char kCameraDiffusionSpatialScale[] = "CameraDiffusionSpatialScale";
    inline constexpr const char kCameraDiffusionHaloWarmth[] = "CameraDiffusionHaloWarmth";
    inline constexpr const char kCameraDiffusionCoreIntensity[] = "CameraDiffusionCoreIntensity";
    inline constexpr const char kCameraDiffusionCoreSize[] = "CameraDiffusionCoreSize";
    inline constexpr const char kCameraDiffusionHaloIntensity[] = "CameraDiffusionHaloIntensity";
    inline constexpr const char kCameraDiffusionHaloSize[] = "CameraDiffusionHaloSize";
    inline constexpr const char kCameraDiffusionBloomIntensity[] = "CameraDiffusionBloomIntensity";
    inline constexpr const char kCameraDiffusionBloomSize[] = "CameraDiffusionBloomSize";
    inline constexpr const char kPrintDiffusionEnabled[] = "PrintDiffusionEnabled";
    inline constexpr const char kPrintDiffusionFamily[] = "PrintDiffusionFamily";
    inline constexpr const char kPrintDiffusionStrength[] = "PrintDiffusionStrength";
    inline constexpr const char kPrintDiffusionSpatialScale[] = "PrintDiffusionSpatialScale";
    inline constexpr const char kPrintDiffusionHaloWarmth[] = "PrintDiffusionHaloWarmth";
    inline constexpr const char kPrintDiffusionCoreIntensity[] = "PrintDiffusionCoreIntensity";
    inline constexpr const char kPrintDiffusionCoreSize[] = "PrintDiffusionCoreSize";
    inline constexpr const char kPrintDiffusionHaloIntensity[] = "PrintDiffusionHaloIntensity";
    inline constexpr const char kPrintDiffusionHaloSize[] = "PrintDiffusionHaloSize";
    inline constexpr const char kPrintDiffusionBloomIntensity[] = "PrintDiffusionBloomIntensity";
    inline constexpr const char kPrintDiffusionBloomSize[] = "PrintDiffusionBloomSize";
    inline constexpr const char kGlareActive[] = "GlareActive";
    inline constexpr const char kGlarePercent[] = "GlarePercent";
    inline constexpr const char kGlareRoughness[] = "GlareRoughness";
    inline constexpr const char kGlareBlurSigmaPx[] = "GlareBlurSigmaPx";
    inline constexpr const char kPrintShadowCompensationFactor[] = "PrintShadowCompensationFactor";
    inline constexpr const char kPrintShadowCompensationDensity[] = "PrintShadowCompensationDensity";
    inline constexpr const char kPrintShadowCompensationTransition[] = "PrintShadowCompensationTransition";
    inline constexpr const char kScannerLensBlurSigmaPx[] = "ScannerLensBlurSigmaPx";
    inline constexpr const char kScannerUnsharpMask[] = "ScannerUnsharpMask";
    inline constexpr const char kScannerBlackCorrection[] = "ScannerBlackCorrection";
    inline constexpr const char kScannerWhiteCorrection[] = "ScannerWhiteCorrection";
    inline constexpr const char kScannerBlackLevel[] = "ScannerBlackLevel";
    inline constexpr const char kScannerWhiteLevel[] = "ScannerWhiteLevel";
    inline constexpr const char kScannerUseLut[] = "ScannerUseLUT";
    inline constexpr const char kScannerLutResolution[] = "ScannerLutResolution";
} // namespace JuicerParams
