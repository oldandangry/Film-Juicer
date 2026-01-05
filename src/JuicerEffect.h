#pragma once

#if defined(_MSC_VER)
// MSVC STL warns (and this project treats warnings as errors) about shared_ptr atomic free-functions
// being deprecated in C++20. We intentionally use them for C++17 compatibility.
#ifndef _SILENCE_CXX20_OLD_SHARED_PTR_ATOMIC_SUPPORT_DEPRECATION_WARNING
#define _SILENCE_CXX20_OLD_SHARED_PTR_ATOMIC_SUPPORT_DEPRECATION_WARNING
#endif
#endif

#include <memory>
#include <string>

#include "ofxsImageEffect.h"
#include "JuicerState.h"
#include "Scanner.h"

namespace Scanner {
    struct Options;
    struct Settings;
}

namespace Print {
    struct Params;
    struct Runtime;
}

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
    class Double2DParam;
    class IntParam;
    class BooleanParam;
    struct InstanceChangedArgs;
    struct RenderArguments;
} // namespace OFX

// Placeholder parameter names (Step 1)
// Per agx-emulsion parity: this is "camera.exposure_compensation_ev" (not just "exposure")
#define kParamExposure "Exposure"   // UI: "Exposure Compensation Ev"
#define kParamCameraAutoExposure "CameraAutoExposure"
#define kParamContrast "Contrast"   // unitless
#define kParamSpectralMode "SpectralUpsampling"
#define kParamReferenceIlluminant "ReferenceIlluminant"
#define kParamEnlargerIlluminant "EnlargerIlluminant"
#define kParamEnlargerDichroicSet "EnlargerDichroicSet"

// Film stock parameter
#define kParamFilmStock "FilmStock"

// Print paper parameter
#define kParamPrintPaper "PrintPaper"

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
    void applyNeutralFilters(const ParamSnapshot& P, bool resetFilterParams, bool ensureExposureComp);
    bool applyMetadataIlluminantDefaults(ParamSnapshot& P);
#ifdef JUICER_ENABLE_COUPLERS
    void applyCouplerProfileDefaults(ParamSnapshot& P);
#endif

    OFX::Clip* _src = nullptr;
    OFX::Clip* _dst = nullptr;

    // Cached params (wrappers)
    OFX::DoubleParam* _pExposure = nullptr;
    OFX::BooleanParam* _pCameraAutoExposure = nullptr;
    OFX::DoubleParam* _pCameraFilmFormat = nullptr;
    OFX::ChoiceParam* _pCameraMeteringMethod = nullptr;
    OFX::ChoiceParam* _pFilmStock = nullptr;
    OFX::ChoiceParam* _pSpectralMode = nullptr;
    OFX::ChoiceParam* _pPrintPaper = nullptr;
    OFX::ChoiceParam* _pRefIll = nullptr;
    OFX::ChoiceParam* _pEnlIll = nullptr;
    OFX::ChoiceParam* _pEnlDichroicSet = nullptr;
    OFX::ChoiceParam* _pInputColorSpace = nullptr;
    OFX::BooleanParam* _pInputCctfDecoding = nullptr;
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
#endif

    // Scanner and print params
    OFX::DoubleParam* _pScannerLensBlur = nullptr;
    OFX::Double2DParam* _pScannerUnsharp = nullptr;
    OFX::BooleanParam* _pScannerUseLut = nullptr;
    OFX::IntParam* _pScannerLutResolution = nullptr;

    OFX::BooleanParam* _pPrintBypass = nullptr;
    OFX::DoubleParam* _pPrintExposure = nullptr;
    OFX::DoubleParam* _pPrintPreflash = nullptr;
    OFX::BooleanParam* _pPrintExposureComp = nullptr;
    OFX::DoubleParam* _pEnlargerY = nullptr;
    OFX::DoubleParam* _pEnlargerM = nullptr;

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
    OFX::DoubleParam* _pGrainParticleAreaUm2 = nullptr;
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
