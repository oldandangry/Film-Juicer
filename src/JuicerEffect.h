#pragma once

#include <memory>
#include <string>

#include "ofxsImageEffect.h"

struct InstanceState;
struct ParamSnapshot;
struct WorkingState;

namespace Profiles {
    struct HalationMetadata;
    struct GrainMetadata;
    struct ProfileGlare;
} // namespace Profiles

namespace Scanner {
    struct Options;
    struct Settings;
} // namespace Scanner

namespace Print {
    struct Params;
    struct Runtime;
} // namespace Print

namespace OutputEncoding {
    struct Params;
}

#ifdef JUICER_ENABLE_COUPLERS
namespace Couplers {
    struct Runtime;
}
#endif

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
#define kParamEnlargerDichroicSet "EnlargerDichroicSet"

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

    struct AutoExposureResult {
        float exposureScale = 1.0f;
        double autoEV = 0.0;
        OfxRectI meterBounds{0, 0, 0, 0};
        bool meterBoundsValid = false;
    };

    struct WorkingStateInfo {
        std::shared_ptr<const WorkingState> workingState;
        const Print::Runtime* printRuntime = nullptr;
        bool workingStateReady = false;
        bool printRuntimeReady = false;
    };

    ExposureParams gatherExposureParams() const;
    Scanner::Options gatherScannerOptions() const;
    Scanner::Settings gatherScannerSettings() const;
    Print::Params gatherPrintParams() const;
    Profiles::HalationMetadata gatherHalationUi() const;
    Profiles::GrainMetadata gatherGrainUi() const;
    Profiles::ProfileGlare gatherGlareUi() const;
    OutputEncoding::Params gatherOutputEncodingParams() const;
    void applyHalationProfileDefaults();
    void applyGrainPresetDefaults(int presetIndex);
    void resetGrainAdvancedControls();
    void updateGrainPresetLabel(bool custom);
    void updateGrainChromaEnabled();
    [[noreturn]] void throw_spektrafilm_phase1a_render_cutoff(const OFX::RenderArguments& args) const;
    // SF_TEMP_BRIDGE_CPUProductRendererBlocked owner=Phase1A remove=Phase3:
    // retained declaration is unreachable from product render after the Phase 1A cutoff.
    AutoExposureResult computeAutoExposure(
        const OFX::RenderArguments& args,
        OFX::Image* srcImg,
        const OfxRectI& fullBounds,
        const ExposureParams& exposureParams) const;
#ifdef JUICER_ENABLE_COUPLERS
    Couplers::Runtime prepareCouplers(
        const OFX::RenderArguments& args,
        int fullWidth,
        int fullHeight,
        float pixelSizeUm) const;
#endif
    WorkingStateInfo prepareWorkingState() const;

    ParamSnapshot snapshotParams() const;
    void onParamsPossiblyChanged(const char* changedNameOrNull);
    void bootstrap_after_attach();
    void applyNeutralFilters(const ParamSnapshot& P, Print::Runtime& runtime);
    bool applyMetadataIlluminantDefaults(ParamSnapshot& P, const Print::Runtime& runtime);
#ifdef JUICER_ENABLE_COUPLERS
    void initializeCouplerParamsFromProfileIfNeeded(ParamSnapshot& P);
    void syncCouplerParamsFromProfileFollowMask(ParamSnapshot& P);
    void clearCouplerFollowStockForParam(const char* changedNameOrNull);
    void applyCouplerProfileDefaults(ParamSnapshot& P);
#endif

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
    OFX::StrChoiceParam* _pScanRoute = nullptr;
    OFX::ChoiceParam* _pOutputColorSpace = nullptr;
    OFX::BooleanParam* _pOutputCctfEncoding = nullptr;
    OFX::BooleanParam* _pOutputLinearPassThrough = nullptr;


#ifdef JUICER_ENABLE_COUPLERS
    OFX::BooleanParam* _pCouplersActive = nullptr;
    OFX::DoubleParam* _pCouplersAmount = nullptr;
    OFX::DoubleParam* _pCouplersAmountR = nullptr;
    OFX::DoubleParam* _pCouplersAmountG = nullptr;
    OFX::DoubleParam* _pCouplersAmountB = nullptr;
    OFX::DoubleParam* _pCouplersSigma = nullptr;
    OFX::DoubleParam* _pCouplersHigh = nullptr;
    OFX::DoubleParam* _pCouplersSpatialSigma = nullptr;
    OFX::IntParam* _pCouplersInitVersion = nullptr;
    OFX::IntParam* _pCouplersFollowMask = nullptr;
#endif

    // Scanner and print params
    OFX::DoubleParam* _pScannerLensBlur = nullptr;
    OFX::Double2DParam* _pScannerUnsharp = nullptr;
    OFX::BooleanParam* _pScannerUseLut = nullptr;
    OFX::IntParam* _pScannerLutResolution = nullptr;

    OFX::DoubleParam* _pPrintExposure = nullptr;
    OFX::DoubleParam* _pPrintPreflash = nullptr;
    OFX::BooleanParam* _pPrintExposureComp = nullptr;
    OFX::DoubleParam* _pEnlargerY = nullptr;
    OFX::DoubleParam* _pEnlargerM = nullptr;
    OFX::DoubleParam* _pEnlargerC = nullptr;

    OFX::BooleanParam* _pHalationActive = nullptr;
    OFX::DoubleParam* _pHalationStrengthMaster = nullptr;
    OFX::DoubleParam* _pHalationSizeUmMaster = nullptr;
    OFX::DoubleParam* _pHalationScatteringStrengthMaster = nullptr;
    OFX::DoubleParam* _pHalationScatteringSizeUmMaster = nullptr;
    OFX::PushButtonParam* _pHalationRevertToStock = nullptr;
    OFX::Double3DParam* _pHalationStrength = nullptr;
    OFX::Double3DParam* _pHalationSizeUm = nullptr;
    OFX::Double3DParam* _pHalationScatteringStrength = nullptr;
    OFX::Double3DParam* _pHalationScatteringSizeUm = nullptr;

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
    OFX::BooleanParam* _pGrainBreathingDebug = nullptr;
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

    OFX::BooleanParam* _pGlareActive = nullptr;
    OFX::DoubleParam* _pGlarePercent = nullptr;
    OFX::DoubleParam* _pGlareRoughness = nullptr;
    OFX::DoubleParam* _pGlareBlurSigmaPx = nullptr;
    OFX::DoubleParam* _pGlareCompRemovalFactor = nullptr;
    OFX::DoubleParam* _pGlareCompRemovalDensity = nullptr;
    OFX::DoubleParam* _pGlareCompRemovalTransition = nullptr;
    OFX::DoubleParam* _pPrintDminFactor = nullptr;

    std::unique_ptr<InstanceState> _state;

    double _halationStrengthMasterLast = 0.0;
    double _halationSizeUmMasterLast = 0.0;
    double _halationScatteringStrengthMasterLast = 0.0;
    double _halationScatteringSizeUmMasterLast = 0.0;
    double _grainParticleScaleMasterLast = 0.0;
    double _grainParticleScaleLayersMasterLast = 0.0;
    double _grainDensityMinMasterLast = 0.0;
    double _grainUniformityMasterLast = 0.0;
};
