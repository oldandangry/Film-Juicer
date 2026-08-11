#pragma once

#include <memory>
#include <string>

#include "ofxsImageEffect.h"

struct InstanceState;
struct ParamSnapshot;
struct VisualGrainControls;

namespace Spektrafilm {
    struct DiffusionFilterAuthoredControls;
    using ::VisualGrainControls;
} // namespace Spektrafilm

namespace OFX {
    class Clip;
    class Image;
    class DoubleParam;
    class Double3DParam;
    class ChoiceParam;
    class StrChoiceParam;
    class Double2DParam;
    class IntParam;
    class BooleanParam;
    struct InstanceChangedArgs;
    struct RenderArguments;
} // namespace OFX

// Placeholder parameter names (Step 1)
// Per agx-emulsion parity: this is "camera.exposure_compensation_ev" (not just "exposure")
#define kParamExposure "Exposure" // UI: "Exposure Compensation Ev"
#define kParamCameraAutoExposure "CameraAutoExposure"
#define kParamContrast "Contrast" // unitless
#define kParamSpectralMode "SpectralUpsampling"
#define kParamReferenceIlluminant "ReferenceIlluminant"
#define kParamEnlargerIlluminant "EnlargerIlluminant"
#define kParamDichroicFilterSet "DichroicFilterSet"

// Output encoding parameters
#define kParamOutputColorSpace "OutputColorSpace"
#define kParamOutputCctfEncoding "OutputCctfEncoding"
#define kParamOutputLinearPassThrough "OutputLinearPassThrough"

class JuicerEffect : public OFX::ImageEffect {
public:
    explicit JuicerEffect(OfxImageEffectHandle handle);
    ~JuicerEffect() override;

    void render(const OFX::RenderArguments& args) override;
    void changedParam(const OFX::InstanceChangedArgs& args, const std::string& paramName) override;

private:
    struct ExposureParams {
        double sliderEV = 0.0;
        float sliderScale = 1.0f;
        bool cameraAutoEnabled = true;
        int meteringMethod = 0;
    };

    struct DiffusionUiParams {
        OFX::BooleanParam* enabled = nullptr;
        OFX::ChoiceParam* family = nullptr;
        OFX::DoubleParam* strength = nullptr;
        OFX::DoubleParam* spatialScale = nullptr;
        OFX::DoubleParam* haloWarmth = nullptr;
        OFX::DoubleParam* coreIntensity = nullptr;
        OFX::DoubleParam* coreSize = nullptr;
        OFX::DoubleParam* haloIntensity = nullptr;
        OFX::DoubleParam* haloSize = nullptr;
        OFX::DoubleParam* bloomIntensity = nullptr;
        OFX::DoubleParam* bloomSize = nullptr;
    };

    ExposureParams gatherExposureParams() const;
    Spektrafilm::VisualGrainControls gatherGrainUi() const;
    Spektrafilm::DiffusionFilterAuthoredControls gatherDiffusionUi(
        const DiffusionUiParams& params) const;
    void applyDirGammaProfileDefaults();
    void applyGrainPresetDefaults(int presetIndex);
    void resetGrainAdvancedControls();
    void updateGrainPresetLabel(bool custom);
    void updateGrainChromaEnabled();
    void updateDiffusionControlState();
    [[noreturn]] void throw_spektrafilm_phase1a_render_cutoff(const OFX::RenderArguments& args) const;
    ParamSnapshot snapshotParams() const;
    void onParamsPossiblyChanged(const char* changedNameOrNull);
    void initialize_pending_render_state();

    OFX::Clip* _src = nullptr;
    OFX::Clip* _dst = nullptr;

    // Cached params (wrappers)
    OFX::DoubleParam* _pExposure = nullptr;
    OFX::BooleanParam* _pCameraAutoExposure = nullptr;
    OFX::DoubleParam* _pCameraFilmFormat = nullptr;
    OFX::ChoiceParam* _pCameraMeteringMethod = nullptr;
    OFX::StrChoiceParam* _pFilmProfileKey = nullptr;
    OFX::ChoiceParam* _pSpectralMode = nullptr;
    OFX::StrChoiceParam* _pPrintProfileKey = nullptr;
    OFX::ChoiceParam* _pRefIll = nullptr;
    OFX::ChoiceParam* _pEnlIll = nullptr;
    OFX::ChoiceParam* _pEnlDichroicSet = nullptr;
    OFX::ChoiceParam* _pInputColorSpace = nullptr;
    OFX::BooleanParam* _pInputCctfDecoding = nullptr;
    OFX::BooleanParam* _pHanatos2025AdaptationWindow = nullptr;
    OFX::BooleanParam* _pHanatos2025AdaptationSurface = nullptr;
    OFX::StrChoiceParam* _pScanRoute = nullptr;
    OFX::ChoiceParam* _pOutputColorSpace = nullptr;
    OFX::BooleanParam* _pOutputCctfEncoding = nullptr;
    OFX::BooleanParam* _pOutputLinearPassThrough = nullptr;


    OFX::BooleanParam* _pCouplersActive = nullptr;
    OFX::DoubleParam* _pCouplersAmount = nullptr;
    OFX::DoubleParam* _pCouplersInhibitionSameLayer = nullptr;
    OFX::DoubleParam* _pCouplersInhibitionInterlayer = nullptr;
    OFX::DoubleParam* _pCouplersDiffusionSizeUm = nullptr;
    OFX::BooleanParam* _pCouplersGammaUseStock = nullptr;
    OFX::Double3DParam* _pCouplersGammaSameLayerRgb = nullptr;
    OFX::Double2DParam* _pCouplersGammaInterlayerRToGb = nullptr;
    OFX::Double2DParam* _pCouplersGammaInterlayerGToRb = nullptr;
    OFX::Double2DParam* _pCouplersGammaInterlayerBToRg = nullptr;

    // Scanner and print params
    OFX::DoubleParam* _pScannerLensBlur = nullptr;
    OFX::Double2DParam* _pScannerUnsharp = nullptr;
    OFX::BooleanParam* _pScannerBlackCorrection = nullptr;
    OFX::BooleanParam* _pScannerWhiteCorrection = nullptr;
    OFX::DoubleParam* _pScannerBlackLevel = nullptr;
    OFX::DoubleParam* _pScannerWhiteLevel = nullptr;
    OFX::BooleanParam* _pScannerUseLut = nullptr;
    OFX::IntParam* _pScannerLutResolution = nullptr;

    OFX::DoubleParam* _pPrintExposure = nullptr;
    OFX::DoubleParam* _pPrintPreflash = nullptr;
    OFX::BooleanParam* _pPrintExposureComp = nullptr;
    OFX::DoubleParam* _pEnlargerY = nullptr;
    OFX::DoubleParam* _pEnlargerM = nullptr;
    OFX::DoubleParam* _pEnlargerC = nullptr;

    OFX::BooleanParam* _pHalationActive = nullptr;

    OFX::BooleanParam* _pGrainActive = nullptr;
    OFX::BooleanParam* _pGrainSublayersActive = nullptr;
    OFX::ChoiceParam* _pGrainPreset = nullptr;
    OFX::DoubleParam* _pGrainParticleAreaUm2 = nullptr;
    OFX::DoubleParam* _pGrainAmplitude = nullptr;
    OFX::DoubleParam* _pGrainSharpness = nullptr;
    OFX::DoubleParam* _pGrainChroma = nullptr;
    OFX::DoubleParam* _pGrainTexture = nullptr;
    OFX::DoubleParam* _pGrainParticleScaleMaster = nullptr;
    OFX::DoubleParam* _pGrainParticleScaleLayersMaster = nullptr;
    OFX::DoubleParam* _pGrainDensityMinMaster = nullptr;
    OFX::DoubleParam* _pGrainUniformityMaster = nullptr;
    OFX::Double3DParam* _pGrainParticleScale = nullptr;
    OFX::Double3DParam* _pGrainParticleScaleLayers = nullptr;
    OFX::Double3DParam* _pGrainDensityMin = nullptr;
    OFX::Double3DParam* _pGrainUniformity = nullptr;
    OFX::DoubleParam* _pGrainBlur = nullptr;
    OFX::DoubleParam* _pGrainBlurDyeCloudsUm = nullptr;
    OFX::DoubleParam* _pGrainSizeMixWeight = nullptr;
    OFX::DoubleParam* _pGrainSizeMixWeightMid = nullptr;
    OFX::DoubleParam* _pGrainSizeMixScale = nullptr;
    OFX::DoubleParam* _pGrainClumpTemporalMix = nullptr;
    OFX::DoubleParam* _pGrainClumpMorphPeriodSec = nullptr;
    OFX::ChoiceParam* _pGrainDebugView = nullptr;
    OFX::Double2DParam* _pGrainMicroStructure = nullptr;
    OFX::PushButtonParam* _pGrainResetAdvanced = nullptr;
    bool _grainPresetCustom = false;
    std::string _grainPresetLabel = "Grain Preset";
    std::string _grainChromaHint = "0 = achromatic, 1 = independent RGB grain.";
    OFX::DoubleParam* _pGateWeaveAmount = nullptr;
    OFX::DoubleParam* _pFilmDustAmount = nullptr;
    OFX::DoubleParam* _pGateDustAmount = nullptr;
    OFX::DoubleParam* _pFilmScratchAmount = nullptr;
    OFX::DoubleParam* _pGateScratchAmount = nullptr;

    DiffusionUiParams _cameraDiffusionUi;
    DiffusionUiParams _printDiffusionUi;

    OFX::BooleanParam* _pGlareActive = nullptr;
    OFX::DoubleParam* _pGlarePercent = nullptr;
    OFX::DoubleParam* _pGlareRoughness = nullptr;
    OFX::DoubleParam* _pGlareBlurSigmaPx = nullptr;
    OFX::DoubleParam* _pGlareCompRemovalFactor = nullptr;
    OFX::DoubleParam* _pGlareCompRemovalDensity = nullptr;
    OFX::DoubleParam* _pGlareCompRemovalTransition = nullptr;
    OFX::DoubleParam* _pPrintDminFactor = nullptr;

    std::unique_ptr<InstanceState> _state;

    double _grainParticleScaleMasterLast = 0.0;
    double _grainParticleScaleLayersMasterLast = 0.0;
    double _grainDensityMinMasterLast = 0.0;
    double _grainUniformityMasterLast = 0.0;
};
