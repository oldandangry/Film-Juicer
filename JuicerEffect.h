#pragma once

#include <memory>
#include <string>

#include "ofxsImageEffect.h"
#include "JuicerState.h"

namespace Scanner {
    struct Params;
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
    class ChoiceParam;
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
#define kParamViewingIllum  "ViewingIlluminant"
#define kParamEnlargerIlluminant "EnlargerIlluminant"

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
    };

    struct AutoExposureResult {
        float exposureScale = 1.0f;
        double autoEV = 0.0;
    };

    struct WorkingStateInfo {
        WorkingState* activeWorkingState = nullptr;
        const WorkingState* workingState = nullptr;
        const Print::Runtime* printRuntime = nullptr;
        bool workingStateReady = false;
        bool printRuntimeReady = false;
    };

    ExposureParams gatherExposureParams() const;
    Scanner::Params gatherScannerParams() const;
    Print::Params gatherPrintParams() const;
    OutputEncoding::Params gatherOutputEncodingParams() const;
    AutoExposureResult computeAutoExposure(
        const OFX::RenderArguments& args,
        OFX::Image* srcImg,
        const OfxRectI& fullBounds,
        const Scanner::Params& scannerParams,
        const ExposureParams& exposureParams) const;
#ifdef JUICER_ENABLE_COUPLERS
    Couplers::Runtime prepareCouplers(
        const OFX::RenderArguments& args,
        int fullWidth,
        int fullHeight) const;
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
    OFX::ChoiceParam* _pFilmStock = nullptr;
    OFX::ChoiceParam* _pPrintPaper = nullptr;
    OFX::ChoiceParam* _pRefIll = nullptr;
    OFX::ChoiceParam* _pViewIll = nullptr;
    OFX::ChoiceParam* _pEnlIll = nullptr;
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
    OFX::BooleanParam* _pScanEnabled = nullptr;
    OFX::BooleanParam* _pScanAuto = nullptr;
    OFX::DoubleParam* _pScanTargetY = nullptr;
    OFX::DoubleParam* _pScanFilmLongEdge = nullptr;

    OFX::BooleanParam* _pPrintBypass = nullptr;
    OFX::DoubleParam* _pPrintExposure = nullptr;
    OFX::DoubleParam* _pPrintPreflash = nullptr;
    OFX::BooleanParam* _pPrintExposureComp = nullptr;
    OFX::DoubleParam* _pEnlargerY = nullptr;
    OFX::DoubleParam* _pEnlargerM = nullptr;

    std::unique_ptr<InstanceState> _state;

};
