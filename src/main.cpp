

// OpenFX 1.4 canonical headers (no non-spec feature macros)
#include "ofxCore.h"        // OfxHost, OfxPlugin, kOfxAction*
#include "ofxProperty.h"    // OfxPropertySuiteV1, property keys
#include "ofxParam.h"       // OfxParameterSuiteV1, kOfxTypeParameter
#include "ofxImageEffect.h" // OfxImageEffectSuiteV1, OfxRectI, OfxRectD, image effect props
#include "ofxPixels.h"      // Pixel/rect helpers used by some 1.4 distributions

#include <filesystem>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// Resolve OFX support library C++ wrappers (OpenFX 1.4 compliant)
#pragma warning(push)
#pragma warning(disable: 5040)
#include "ofxsCore.h"
#include "ofxsImageEffect.h"
#include "ofxsParam.h"
#include "ofxsProcessing.h"
#include "ofxsMemory.h"
#include "ofxsLog.h"
#pragma warning(pop)
// Note: These headers provide the factory macros, ImageEffect base, descriptors,
// Param wrappers, Clip/Image RAII, RenderArguments, and optional processors.


#ifndef kOfxActionInstanceChanged
#error "Missing kOfxActionInstanceChanged in ofxCore.h (OpenFX 1.4)."
#endif
#ifndef kOfxPropType
#error "Missing kOfxPropType in ofxCore.h (OpenFX 1.4)."
#endif
#ifndef kOfxTypeParameter
#error "Missing kOfxTypeParameter in ofxParam.h (OpenFX 1.4)."
#endif

#include "Couplers.h"
#include "JuicerEffect.h"
#include "JuicerState.h"
#include "OutputEncoding.h"
#include "SpectralProcessing.h"
#include "ColorTransforms.h"
#include "ParamNames.h"
#include "Print.h"
#include "ProfileJSONLoader.h"

namespace {

    static std::string computeDataDir()
    {
        namespace fs = std::filesystem;

#if defined(_WIN32)
        HMODULE module = nullptr;
        if (!GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&computeDataDir),
            &module)) {
            return std::string();
        }

        std::wstring buffer(MAX_PATH, L'\0');
        DWORD length = 0;
        for (;;) {
            SetLastError(ERROR_SUCCESS);
            length = GetModuleFileNameW(module, buffer.data(), static_cast<DWORD>(buffer.size()));
            if (length == 0) {
                return std::string();
            }
            if (length < buffer.size()) {
                buffer.resize(length);
                break;
            }
            if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
                buffer.resize(length);
                break;
            }
            buffer.resize(buffer.size() * 2);
        }

        fs::path modulePath(buffer);
        fs::path moduleDir = modulePath.parent_path();
        if (moduleDir.empty()) {
            return std::string();
        }
        fs::path contentsDir = moduleDir.parent_path();
        if (contentsDir.empty()) {
            return std::string();
        }

        fs::path resourcesDir = (contentsDir / "Resources").lexically_normal();
        resourcesDir.make_preferred();
        std::wstring native = resourcesDir.native();
        if (!native.empty() && native.back() != L'\\') {
            native.push_back(L'\\');
        }

        if (native.empty()) {
            return std::string();
        }

        int required = WideCharToMultiByte(CP_UTF8, 0, native.c_str(), static_cast<int>(native.size()), nullptr, 0, nullptr, nullptr);
        if (required <= 0) {
            return std::string();
        }

        std::string path(static_cast<size_t>(required), '\0');
        WideCharToMultiByte(CP_UTF8, 0, native.c_str(), static_cast<int>(native.size()), path.data(), required, nullptr, nullptr);
        return path;
#else
        Dl_info info{};
        if (dladdr(reinterpret_cast<const void*>(&computeDataDir), &info) == 0 || info.dli_fname == nullptr) {
            return std::string();
        }

        fs::path modulePath(info.dli_fname);
        fs::path moduleDir = modulePath.parent_path();
        if (moduleDir.empty()) {
            return std::string();
        }
        fs::path contentsDir = moduleDir.parent_path();
        if (contentsDir.empty()) {
            return std::string();
        }

        fs::path resourcesDir = (contentsDir / "Resources").lexically_normal();
        resourcesDir.make_preferred();
        std::string path = resourcesDir.u8string();
        if (!path.empty() && path.back() != '/') {
            path.push_back('/');
        }
        return path;
#endif
    }

} // namespace

// Plugin data directory (immutable)
const std::string gDataDir = computeDataDir();

// === Resolve support library factory (Step 1) ===
// The factory owns the plugin identity and wires descriptor/instance creation.

// === Resolve support library factory (Resolve pattern) ===
// Define a factory that derives from PluginFactoryHelper and wires ID/version.
#define kPluginIdentifier "com.juicer.Juicer"
#define kPluginVersionMajor 1
#define kPluginVersionMinor 0

class JuicerPluginFactory : public OFX::PluginFactoryHelper<JuicerPluginFactory> {
public:
    JuicerPluginFactory()
        : OFX::PluginFactoryHelper<JuicerPluginFactory>(kPluginIdentifier,
            kPluginVersionMajor,
            kPluginVersionMinor) {
    }

    void describe(OFX::ImageEffectDescriptor& desc) override;
    void describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum context) override;
    OFX::ImageEffect* createInstance(OfxImageEffectHandle handle, OFX::ContextEnum context) override;
};


void JuicerPluginFactory::describe(OFX::ImageEffectDescriptor& desc)
{
    // Label/group
    desc.setLabels("Juicer", "Juicer", "Juicer");
    desc.setPluginGrouping("Negative-juice");

    // Contexts
    desc.addSupportedContext(OFX::eContextFilter);

    // Pixel depths
    desc.addSupportedBitDepth(OFX::eBitDepthFloat);

    // Flags
    desc.setSingleInstance(false);
    desc.setHostFrameThreading(false);
    desc.setSupportsMultiResolution(true);
    desc.setSupportsTiles(false);
    desc.setRenderThreadSafety(OFX::eRenderFullySafe);

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    desc.setSupportsCudaRender(true);
    desc.setSupportsCudaStream(true);
#endif
}

void JuicerPluginFactory::describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum context)
{
    if (context != OFX::eContextFilter) return;

    // Clips
    {
        OFX::ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
        src->addSupportedComponent(OFX::ePixelComponentRGBA);
        src->addSupportedComponent(OFX::ePixelComponentRGB);
        src->setSupportsTiles(false);
        src->setOptional(false);
    }
    {
        OFX::ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
        dst->addSupportedComponent(OFX::ePixelComponentRGBA);
        dst->addSupportedComponent(OFX::ePixelComponentRGB);
        dst->setSupportsTiles(false);
    }

    // Parameters — mirror current define semantics (names, defaults, ranges)
    // Exposure Compensation (per agx-emulsion: camera.exposure_compensation_ev)
    {
        OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(kParamExposure);
        p->setLabel("Exposure Compensation Ev");
        p->setHint("Camera exposure compensation in EV stops. Per agx-emulsion parity: "
            "this is camera.exposure_compensation_ev, applied as 2^EV multiplication "
            "to film exposure. Positive values brighten the image.");
        p->setDefault(0.0);
        p->setRange(-8.0, 8.0);
        p->setDisplayRange(-4.0, 4.0);
    }
    {
        OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam(JuicerParams::kCameraMeteringMethod);
        p->setLabel("Camera metering");
        p->appendOption("Center-weighted");
        p->appendOption("Median");
        p->setDefault(0);
        p->setEvaluateOnChange(true);
    }
    {
        OFX::BooleanParamDescriptor* p = desc.defineBooleanParam(kParamCameraAutoExposure);
        p->setLabel("Camera auto exposure");
        p->setHint("Enable the camera auto meter (agx-emulsion camera.auto_exposure). Scanner auto exposure remains independent.");
        p->setDefault(true);
        p->setEvaluateOnChange(true);
    }
    {
        OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kCameraFilmFormatMm);
        p->setLabel("Camera film format (mm)");
        p->setDefault(35.0);
        p->setRange(5.0, 400.0);
        p->setDisplayRange(8.0, 70.0);
        p->setHint("Longest capture dimension in millimeters; used for micrometer-to-pixel conversions.");
        p->setEvaluateOnChange(true);
    }

    // Film stock (choice)
    {
        OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam(kParamFilmStock);
        p->setLabel("Film stock");
        const int stockCount = film_stock_option_count();
        for (int i = 0; i < stockCount; ++i) p->appendOption(film_stock_option_label(i));
        p->setDefault(0);
        p->setEvaluateOnChange(true);
    }
    // Spectral upsampling
    {
        OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam(kParamSpectralMode);
        p->setLabel("Spectral upsampling");
        p->appendOption("Hanatos");
        p->appendOption("Mallett");
        p->setHint("Choose the spectral reconstruction method used for film exposure. "
            "Hanatos uses the Hanatos 2025 LUT when available; "
            "Mallett uses the Mallett 2019 basis reconstruction (tables + S-inverse).");
        p->setDefault(0);
        p->setEvaluateOnChange(true);

    }
    // Input colour space and encoding
    {
        OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam(JuicerParams::kInputColorSpace);
        p->setLabel("Input color space");
        for (std::size_t i = 0; i < Spectral::kInputColorSpaceCount; ++i) {
            p->appendOption(Spectral::kInputColorSpaceLabels[i]);
        }
        p->setDefault(Spectral::inputColorSpaceToIndex(Spectral::InputColorSpace::DaVinciWideGamut));
        p->setEvaluateOnChange(true);
    }
    {
        OFX::BooleanParamDescriptor* p = desc.defineBooleanParam(JuicerParams::kInputCctfDecoding);
        p->setLabel("Decode input CCTF");
        p->setDefault(false);
        p->setEvaluateOnChange(true);
    }

#ifdef JUICER_ENABLE_COUPLERS
    // Couplers (DIR) parameters — wrapper descriptors matching Couplers::define_params
    {
        OFX::GroupParamDescriptor* grpCouplers = desc.defineGroupParam(Couplers::kParamCouplersGroup);
        if (grpCouplers) grpCouplers->setLabel("DIR couplers");

        {
            OFX::BooleanParamDescriptor* p = desc.defineBooleanParam(Couplers::kParamCouplersActive);
            p->setLabel("Active");
            p->setDefault(true);
            if (grpCouplers) p->setParent(*grpCouplers);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(Couplers::kParamCouplersAmount);
            p->setLabel("Couplers amount");
            p->setDefault(1.0);
            p->setRange(0.0, 2.0);
            p->setDisplayRange(0.0, 2.0);
            if (grpCouplers) p->setParent(*grpCouplers);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(Couplers::kParamCouplersAmountR);
            p->setLabel("Couplers ratio R");
            p->setDefault(1.0);
            p->setRange(0.0, 1.0);
            p->setDisplayRange(0.0, 1.0);
            if (grpCouplers) p->setParent(*grpCouplers);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(Couplers::kParamCouplersAmountG);
            p->setLabel("Couplers ratio G");
            p->setDefault(1.0);
            p->setRange(0.0, 1.0);
            p->setDisplayRange(0.0, 1.0);
            if (grpCouplers) p->setParent(*grpCouplers);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(Couplers::kParamCouplersAmountB);
            p->setLabel("Couplers ratio B");
            p->setDefault(1.0);
            p->setRange(0.0, 1.0);
            p->setDisplayRange(0.0, 1.0);
            if (grpCouplers) p->setParent(*grpCouplers);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(Couplers::kParamCouplersLayerSigma);
            p->setLabel("Layer diffusion");
            p->setDefault(2.0);
            p->setRange(0.0, 4.0);
            p->setDisplayRange(0.0, 4.0);
            if (grpCouplers) p->setParent(*grpCouplers);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(Couplers::kParamCouplersHighExpShift);
            p->setLabel("High exposure shift");
            p->setDefault(0.0);
            p->setRange(0.0, 1.0);
            p->setDisplayRange(0.0, 1.0);
            if (grpCouplers) p->setParent(*grpCouplers);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(Couplers::kParamCouplersSpatialSigma);
            p->setLabel("Couplers spatial diffusion (\xC2\xB5m)");
            p->setDefault(10.0);
            p->setRange(0.0, 50.0);
            p->setDisplayRange(0.0, 50.0);
            p->setHint("Micrometers of DIR spatial diffusion; scaled by the Camera film format parameter.");
            if (grpCouplers) p->setParent(*grpCouplers);
            p->setEvaluateOnChange(true);
        }
    }

#endif

    // Scanner optics and math
    OFX::GroupParamDescriptor* grpScannerOptics = desc.defineGroupParam("ScannerOptics");
    if (grpScannerOptics) grpScannerOptics->setLabel("Scanner Optics");
    OFX::GroupParamDescriptor* grpScannerMath = desc.defineGroupParam("ScannerMath");
    if (grpScannerMath) grpScannerMath->setLabel("Scanner Math");
    {
        OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kScannerLensBlurSigmaPx);
        p->setLabel("Scanner lens blur (px)");
        p->setHint("Gaussian blur sigma in pixels for scanner optics.");
        p->setDefault(0.55);
        p->setRange(0.0, 10.0);
        p->setDisplayRange(0.0, 3.0);
        if (grpScannerOptics) p->setParent(*grpScannerOptics);
        p->setEvaluateOnChange(true);
    }
    {
        OFX::Double2DParamDescriptor* p = desc.defineDouble2DParam(JuicerParams::kScannerUnsharpMask);
        p->setLabel("Scanner unsharp mask");
        p->setHint("Unsharp sigma (px) and amount applied after scanner blur.");
        p->setDefault(0.7, 1.0);
        p->setRange(0.0, 0.0, 5.0, 3.0);
        p->setDisplayRange(0.0, 0.0, 5.0, 3.0);
        p->setDimensionLabels("Sigma (px)", "Amount");
        if (grpScannerOptics) p->setParent(*grpScannerOptics);
        p->setEvaluateOnChange(true);
    }
    {
        OFX::BooleanParamDescriptor* p = desc.defineBooleanParam(JuicerParams::kScannerUseLut);
        p->setLabel("Scanner use LUT");
        p->setDefault(true);
        if (grpScannerMath) p->setParent(*grpScannerMath);
        p->setHint("Enable precomputed scanner spectral LUTs.");
        p->setEvaluateOnChange(true);
    }
    {
        OFX::IntParamDescriptor* p = desc.defineIntParam(JuicerParams::kScannerLutResolution);
        p->setLabel("Scanner LUT resolution");
        p->setDefault(17);
        p->setRange(17, 128);
        p->setDisplayRange(17, 128);
        if (grpScannerMath) p->setParent(*grpScannerMath);
        p->setHint("Cube resolution for scanner spectral LUTs.");
        p->setEvaluateOnChange(true);
    }

    // Print group
    {
        OFX::GroupParamDescriptor* grpPrint = nullptr;
        {
            grpPrint = desc.defineGroupParam("PrintGroup");
            grpPrint->setLabel("Print");
        }
        {
            OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam(kParamPrintPaper);
            p->setLabel("Print paper");
            const int paperCount = print_paper_option_count();
            for (int i = 0; i < paperCount; ++i) p->appendOption(print_paper_option_label(i));
            p->setDefault(0);
            if (grpPrint) p->setParent(*grpPrint);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::BooleanParamDescriptor* p = desc.defineBooleanParam("PrintBypass");
            p->setLabel("Bypass print");
            p->setDefault(true);
            if (grpPrint) p->setParent(*grpPrint);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("PrintExposure");
            p->setLabel("Print exposure");
            p->setDefault(1.0);
            p->setDisplayRange(0.1, 10.0);
            if (grpPrint) p->setParent(*grpPrint);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("PrintPreflash");
            p->setLabel("Print preflash");
            p->setDefault(0.0);
            p->setDisplayRange(0.0, 1.0);
            if (grpPrint) p->setParent(*grpPrint);
        }
        {
            OFX::BooleanParamDescriptor* p = desc.defineBooleanParam("PrintExposureCompensation");
            p->setLabel("Print exposure compensation");
            p->setDefault(true);
            if (grpPrint) p->setParent(*grpPrint);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam(kParamEnlargerDichroicSet);
            p->setLabel("Enlarger dichroics");
            p->setHint("Select the dichroic filter set used by the enlarger Y/M/C wheels. This also selects the corresponding neutral Y/M/C baseline database.");
            p->appendOption("Durst Digital Light");
            p->appendOption("Thorlabs");
            p->appendOption("Edmund Optics");
            p->setDefault(0);
            if (grpPrint) p->setParent(*grpPrint);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("EnlargerY");
            p->setLabel("Enlarger Y");
            p->setDefault(0.0);
            p->setDisplayRange(-Print::kEnlargerSteps, Print::kEnlargerSteps);
            p->setIncrement(1.0);
            if (grpPrint) p->setParent(*grpPrint);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("EnlargerM");
            p->setLabel("Enlarger M");
            p->setDefault(0.0);
            p->setDisplayRange(-Print::kEnlargerSteps, Print::kEnlargerSteps);
            p->setIncrement(1.0);
            if (grpPrint) p->setParent(*grpPrint);
        }
    }

    // Halation group
    {
        OFX::GroupParamDescriptor* grpHalation = desc.defineGroupParam("HalationGroup");
        if (grpHalation) {
            grpHalation->setLabel("Halation");
            grpHalation->setOpen(false);
        }
        {
            OFX::BooleanParamDescriptor* p = desc.defineBooleanParam(JuicerParams::kHalationActive);
            p->setLabel("Add halation");
            p->setDefault(false);
            p->setHint("Add halation to the negative (scattering in raw exposure).");
            if (grpHalation) p->setParent(*grpHalation);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kHalationScatteringStrengthMaster);
            p->setLabel("Scattering strength (M)");
            p->setHint("Master control for scattering strength; adjusts RGB values together.");
            p->setDefault((1.0 + 2.0 + 4.0) / 3.0);
            p->setRange(0.0, 100.0);
            p->setDisplayRange(0.0, 25.0);
            if (grpHalation) p->setParent(*grpHalation);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kHalationScatteringSizeUmMaster);
            p->setLabel("Scattering size (M)");
            p->setHint("Master control for scattering size; adjusts RGB values together.");
            p->setDefault((30.0 + 20.0 + 15.0) / 3.0);
            p->setRange(0.0, 1000.0);
            p->setDisplayRange(0.0, 500.0);
            if (grpHalation) p->setParent(*grpHalation);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kHalationStrengthMaster);
            p->setLabel("Halation strength (M)");
            p->setHint("Master control for halation strength; adjusts RGB values together.");
            p->setDefault((3.0 + 0.30 + 0.10) / 3.0);
            p->setRange(0.0, 100.0);
            p->setDisplayRange(0.0, 25.0);
            if (grpHalation) p->setParent(*grpHalation);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kHalationSizeUmMaster);
            p->setLabel("Halation size (M)");
            p->setHint("Master control for halation size; adjusts RGB values together.");
            p->setDefault(200.0);
            p->setRange(0.0, 1000.0);
            p->setDisplayRange(0.0, 1000.0);
            if (grpHalation) p->setParent(*grpHalation);
            p->setEvaluateOnChange(true);
        }
        OFX::GroupParamDescriptor* grpHalationAdvanced = desc.defineGroupParam("HalationAdvancedGroup");
        if (grpHalationAdvanced) {
            grpHalationAdvanced->setLabel("Advanced");
            grpHalationAdvanced->setOpen(false);
            if (grpHalation) grpHalationAdvanced->setParent(*grpHalation);
        }
        {
            OFX::PushButtonParamDescriptor* p = desc.definePushButtonParam(JuicerParams::kHalationRevertToStock);
            p->setLabel("Revert to stock defaults");
            p->setHint("Reset halation parameters to the current film stock defaults.");
            if (grpHalationAdvanced) p->setParent(*grpHalationAdvanced);
        }
        {
            OFX::Double3DParamDescriptor* p = desc.defineDouble3DParam(JuicerParams::kHalationScatteringStrength);
            p->setLabel("Scattering strength (%)");
            p->setHint("Fraction of scattered light (0-100, percentage) per channel.");
            p->setDefault(1.0, 2.0, 4.0);
            p->setRange(0.0, 0.0, 0.0, 100.0, 100.0, 100.0);
            p->setDisplayRange(0.0, 0.0, 0.0, 10.0, 10.0, 10.0);
            p->setDimensionLabels("R", "G", "B");
            if (grpHalationAdvanced) p->setParent(*grpHalationAdvanced);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::Double3DParamDescriptor* p = desc.defineDouble3DParam(JuicerParams::kHalationScatteringSizeUm);
            p->setLabel("Scattering size (\xC2\xB5m)");
            p->setHint("Sigma of the scattering blur in micrometers per channel.");
            p->setDefault(30.0, 20.0, 15.0);
            p->setRange(0.0, 0.0, 0.0, 1000.0, 1000.0, 1000.0);
            p->setDisplayRange(0.0, 0.0, 0.0, 300.0, 300.0, 300.0);
            p->setDimensionLabels("R", "G", "B");
            if (grpHalationAdvanced) p->setParent(*grpHalationAdvanced);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::Double3DParamDescriptor* p = desc.defineDouble3DParam(JuicerParams::kHalationStrength);
            p->setLabel("Halation strength (%)");
            p->setHint("Fraction of halation light (0-100, percentage) per channel.");
            p->setDefault(3.0, 0.30, 0.10);
            p->setRange(0.0, 0.0, 0.0, 100.0, 100.0, 100.0);
            p->setDisplayRange(0.0, 0.0, 0.0, 10.0, 10.0, 10.0);
            p->setDimensionLabels("R", "G", "B");
            if (grpHalationAdvanced) p->setParent(*grpHalationAdvanced);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::Double3DParamDescriptor* p = desc.defineDouble3DParam(JuicerParams::kHalationSizeUm);
            p->setLabel("Halation size (\xC2\xB5m)");
            p->setHint("Sigma of the halation blur in micrometers per channel.");
            p->setDefault(200.0, 200.0, 200.0);
            p->setRange(0.0, 0.0, 0.0, 1000.0, 1000.0, 1000.0);
            p->setDisplayRange(0.0, 0.0, 0.0, 400.0, 400.0, 400.0);
            p->setDimensionLabels("R", "G", "B");
            if (grpHalationAdvanced) p->setParent(*grpHalationAdvanced);
            p->setEvaluateOnChange(true);
        }
    }

    // Grain group
    {
        OFX::GroupParamDescriptor* grpGrain = desc.defineGroupParam("GrainGroup");
        if (grpGrain) {
            grpGrain->setLabel("Grain");
            grpGrain->setOpen(false);
        }
        {
            OFX::BooleanParamDescriptor* p = desc.defineBooleanParam(JuicerParams::kGrainActive);
            p->setLabel("Add grain");
            p->setDefault(false);
            p->setHint("Add grain to the negative.");
            if (grpGrain) p->setParent(*grpGrain);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::BooleanParamDescriptor* p = desc.defineBooleanParam(JuicerParams::kGrainSublayersActive);
            p->setLabel("Use sublayers");
            p->setDefault(true);
            p->setHint("Enable sublayer grain simulation.");
            if (grpGrain) p->setParent(*grpGrain);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam(JuicerParams::kGrainPreset);
            p->setLabel("Grain Preset");
            p->appendOption("Fine");
            p->appendOption("Medium");
            p->appendOption("Coarse");
            p->setDefault(1);
            p->setHint("Starting point for grain size/strength presets.");
            if (grpGrain) p->setParent(*grpGrain);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kGrainAmplitude);
            p->setLabel("Grain Amount (EV)");
            p->setHint("Grain amount in stops (EV); scales OD delta.");
            p->setDefault(-1.20);
            p->setRange(-3.0, 3.0);
            p->setDisplayRange(-3.0, 3.0);
            p->setIncrement(0.1);
            if (grpGrain) p->setParent(*grpGrain);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kGrainBlur);
            p->setLabel("Grain Size (px)");
            p->setHint("Grain correlation size in pixels.");
            p->setDefault(0.50);
            p->setRange(0.20, 2.00);
            p->setDisplayRange(0.20, 2.00);
            if (grpGrain) p->setParent(*grpGrain);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kGrainSharpness);
            p->setLabel("Grain Sharpness");
            p->setHint("Controls grain edge softness.");
            p->setDefault(0.5);
            p->setRange(0.0, 1.0);
            p->setDisplayRange(0.0, 1.0);
            if (grpGrain) p->setParent(*grpGrain);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kGrainChroma);
            p->setLabel("Grain Chroma");
            p->setHint("0 = achromatic, 1 = independent RGB grain.");
            p->setDefault(0.3);
            p->setRange(0.0, 1.0);
            p->setDisplayRange(0.0, 1.0);
            if (grpGrain) p->setParent(*grpGrain);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kGrainTexture);
            p->setLabel("Grain Texture");
            p->setHint("Controls clumping and PSD tail.");
            p->setDefault(0.55);
            p->setRange(0.0, 1.0);
            p->setDisplayRange(0.0, 1.0);
            if (grpGrain) p->setParent(*grpGrain);
            p->setEvaluateOnChange(true);
        }
        OFX::GroupParamDescriptor* grpGrainAdvanced = desc.defineGroupParam("GrainAdvancedGroup");
        if (grpGrainAdvanced) {
            grpGrainAdvanced->setLabel("Advanced");
            grpGrainAdvanced->setOpen(false);
            if (grpGrain) grpGrainAdvanced->setParent(*grpGrain);
        }
        {
            OFX::PushButtonParamDescriptor* p = desc.definePushButtonParam(JuicerParams::kGrainResetAdvanced);
            p->setLabel("Reset Advanced");
            p->setHint("Reset advanced grain controls to the base values.");
            if (grpGrainAdvanced) p->setParent(*grpGrainAdvanced);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kGrainParticleAreaUm2);
            p->setLabel("Particle area (um^2)");
            p->setHint("Particle area in um^2; roughly 0.1 for ISO 100-200, 0.4 for ISO 400.");
            p->setDefault(0.25);
            p->setRange(0.0, 10.0);
            p->setDisplayRange(0.0, 1.0);
            p->setIncrement(0.1);
            if (grpGrainAdvanced) p->setParent(*grpGrainAdvanced);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kGrainParticleScaleMaster);
            p->setLabel("Particle scale (M)");
            p->setHint("Master control for particle scale; adjusts RGB values together.");
            p->setDefault(1.48);
            p->setRange(0.0, 10.0);
            p->setDisplayRange(0.0, 3.0);
            if (grpGrainAdvanced) p->setParent(*grpGrainAdvanced);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kGrainParticleScaleLayersMaster);
            p->setLabel("Particle scale layers (M)");
            p->setHint("Master control for sublayer particle scale; adjusts RGB values together.");
            p->setDefault(1.922);
            p->setRange(0.0, 10.0);
            p->setDisplayRange(0.0, 4.0);
            if (grpGrainAdvanced) p->setParent(*grpGrainAdvanced);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kGrainDensityMinMaster);
            p->setLabel("Density min (M)");
            p->setHint("Master control for minimum grain density; adjusts RGB values together.");
            p->setDefault(0.08);
            p->setRange(0.0, 1.0);
            p->setDisplayRange(0.0, 0.2);
            if (grpGrainAdvanced) p->setParent(*grpGrainAdvanced);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kGrainUniformityMaster);
            p->setLabel("Uniformity (M)");
            p->setHint("Master control for grain uniformity; adjusts RGB values together.");
            p->setDefault(0.97);
            p->setRange(0.0, 1.0);
            p->setDisplayRange(0.9, 1.0);
            if (grpGrainAdvanced) p->setParent(*grpGrainAdvanced);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kGrainBlurDyeCloudsUm);
            p->setLabel("Dye cloud blur scale (px)");
            p->setHint("Scale factor for dye cloud blur sigma in pixels.");
            p->setDefault(1.0);
            p->setRange(0.0, 10.0);
            p->setDisplayRange(0.0, 3.0);
            if (grpGrainAdvanced) p->setParent(*grpGrainAdvanced);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kGrainSizeMixWeight);
            p->setLabel("Size mix weight");
            p->setHint("Weight of the coarse grain population in the size mixture.");
            p->setDefault(0.226);
            p->setRange(0.0, 1.0);
            p->setDisplayRange(0.0, 1.0);
            if (grpGrainAdvanced) p->setParent(*grpGrainAdvanced);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kGrainSizeMixWeightMid);
            p->setLabel("Size mix mid weight");
            p->setHint("Weight of the mid grain population in the size mixture.");
            p->setDefault(0.0);
            p->setRange(0.0, 1.0);
            p->setDisplayRange(0.0, 1.0);
            if (grpGrainAdvanced) p->setParent(*grpGrainAdvanced);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kGrainSizeMixScale);
            p->setLabel("Size mix scale");
            p->setHint("Relative particle size scale for the coarse grain population.");
            p->setDefault(19.0);
            p->setRange(1.0, 25.0);
            p->setDisplayRange(1.0, 25.0);
            if (grpGrainAdvanced) p->setParent(*grpGrainAdvanced);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::Double2DParamDescriptor* p = desc.defineDouble2DParam(JuicerParams::kGrainMicroStructure);
            p->setLabel("Micro-structure");
            p->setHint("Micro-structure parameters: clump cell size (um) and clump sigma (x1e-3).");
            p->setDefault(60.0, 170.0);
            p->setRange(0.0, 0.0, 100.0, 1000.0);
            p->setDisplayRange(0.0, 0.0, 100.0, 200.0);
            p->setDimensionLabels("Cell (um)", "Sigma (1e-3)");
            if (grpGrainAdvanced) p->setParent(*grpGrainAdvanced);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::BooleanParamDescriptor* p = desc.defineBooleanParam(JuicerParams::kGrainBreathingDebug);
            p->setLabel("Breathing debug");
            p->setHint("Debug view for the breathing field.");
            p->setDefault(false);
            if (grpGrainAdvanced) p->setParent(*grpGrainAdvanced);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam(JuicerParams::kGrainDebugView);
            p->setLabel("Grain debug view");
            p->appendOption("Off");
            p->appendOption("Delta mix");
            p->appendOption("Delta fine");
            p->appendOption("Delta coarse");
            p->appendOption("Delta fine raw");
            p->appendOption("Delta coarse raw");
            p->appendOption("Mean density");
            p->setDefault(0);
            p->setHint("Debug view selector for grain delta fields (pre-scanner).");
            if (grpGrainAdvanced) p->setParent(*grpGrainAdvanced);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::Double3DParamDescriptor* p = desc.defineDouble3DParam(JuicerParams::kGrainParticleScale);
            p->setLabel("Particle scale");
            p->setHint("Scale of particle area for the RGB layers (multiplies particle area).");
            p->setDefault(1.48, 1.48, 1.48);
            p->setRange(0.0, 0.0, 0.0, 10.0, 10.0, 10.0);
            p->setDisplayRange(0.0, 0.0, 0.0, 3.0, 3.0, 3.0);
            p->setDimensionLabels("R", "G", "B");
            p->setIsSecret(true);
            if (grpGrainAdvanced) p->setParent(*grpGrainAdvanced);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::Double3DParamDescriptor* p = desc.defineDouble3DParam(JuicerParams::kGrainParticleScaleLayers);
            p->setLabel("Particle scale layers");
            p->setHint("Scale of particle area for sublayers in each color layer.");
            p->setDefault(1.922, 1.922, 1.922);
            p->setRange(0.0, 0.0, 0.0, 10.0, 10.0, 10.0);
            p->setDisplayRange(0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
            p->setDimensionLabels("R", "G", "B");
            p->setIsSecret(true);
            if (grpGrainAdvanced) p->setParent(*grpGrainAdvanced);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::Double3DParamDescriptor* p = desc.defineDouble3DParam(JuicerParams::kGrainDensityMin);
            p->setLabel("Density min");
            p->setHint("Minimum grain density per layer (typ. 0.03-0.06).");
            p->setDefault(0.08, 0.08, 0.08);
            p->setRange(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
            p->setDisplayRange(0.0, 0.0, 0.0, 0.2, 0.2, 0.2);
            p->setDimensionLabels("C", "M", "Y");
            p->setIsSecret(true);
            if (grpGrainAdvanced) p->setParent(*grpGrainAdvanced);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::Double3DParamDescriptor* p = desc.defineDouble3DParam(JuicerParams::kGrainUniformity);
            p->setLabel("Uniformity");
            p->setHint("Uniformity of grain (typ. 0.94-0.98).");
            p->setDefault(0.97, 0.97, 0.97);
            p->setRange(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
            p->setDisplayRange(0.9, 0.9, 0.9, 1.0, 1.0, 1.0);
            p->setDimensionLabels("C", "M", "Y");
            p->setIsSecret(true);
            if (grpGrainAdvanced) p->setParent(*grpGrainAdvanced);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kGrainClumpTemporalMix);
            p->setLabel("Clump temporal mix");
            p->setHint("Modulates clump strength over time (0 = static, higher = more breathing).");
            p->setDefault(0.30);
            p->setRange(0.0, 0.30);
            p->setDisplayRange(0.0, 0.30);
            if (grpGrainAdvanced) p->setParent(*grpGrainAdvanced);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kGrainClumpMorphPeriodSec);
            p->setLabel("Clump morph period (s)");
            p->setHint("Seconds for clump field to evolve; longer is more stable.");
            p->setDefault(8.0);
            p->setRange(5.0, 60.0);
            p->setDisplayRange(5.0, 60.0);
            if (grpGrainAdvanced) p->setParent(*grpGrainAdvanced);
            p->setEvaluateOnChange(true);
        }
    }

    // Effects group
    {
        OFX::GroupParamDescriptor* grpEffects = desc.defineGroupParam("EffectsGroup");
        if (grpEffects) {
            grpEffects->setLabel("Effects");
            grpEffects->setOpen(false);
        }

        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kGateWeaveAmount);
            p->setLabel("Gate weave");
            p->setHint("Scales the global gate weave (film transport jitter).");
            p->setDefault(1.0);
            p->setRange(0.0, 10.0);
            p->setDisplayRange(0.0, 10.0);
            if (grpEffects) p->setParent(*grpEffects);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kFilmDustAmount);
            p->setLabel("Film dust");
            p->setHint("Amount of film-local dust specks (strip space).");
            p->setDefault(0.0);
            p->setRange(0.0, 10.0);
            p->setDisplayRange(0.0, 10.0);
            if (grpEffects) p->setParent(*grpEffects);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kGateDustAmount);
            p->setLabel("Gate dust");
            p->setHint("Amount of gate-local dust specks (sensor/gate space).");
            p->setDefault(0.0);
            p->setRange(0.0, 10.0);
            p->setDisplayRange(0.0, 10.0);
            if (grpEffects) p->setParent(*grpEffects);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kFilmScratchAmount);
            p->setLabel("Film scratches");
            p->setHint("Amount of film-local scratches (strip space).");
            p->setDefault(0.0);
            p->setRange(0.0, 10.0);
            p->setDisplayRange(0.0, 10.0);
            if (grpEffects) p->setParent(*grpEffects);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kGateScratchAmount);
            p->setLabel("Gate scratches");
            p->setHint("Amount of gate-local scratches (sensor/gate space).");
            p->setDefault(0.0);
            p->setRange(0.0, 10.0);
            p->setDisplayRange(0.0, 10.0);
            if (grpEffects) p->setParent(*grpEffects);
            p->setEvaluateOnChange(true);
        }
    }

    // Glare group
    {
        OFX::GroupParamDescriptor* grpGlare = desc.defineGroupParam("GlareGroup");
        if (grpGlare) {
            grpGlare->setLabel("Glare");
            grpGlare->setOpen(false);
        }

        {
            OFX::BooleanParamDescriptor* p = desc.defineBooleanParam(JuicerParams::kGlareActive);
            p->setLabel("Add glare");
            p->setDefault(true);
            p->setHint("Add glare to the print (scanner-stage stray light).");
            if (grpGlare) p->setParent(*grpGlare);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kGlarePercent);
            p->setLabel("Glare percent");
            p->setDefault(0.10);
            p->setRange(0.0, 1.0);
            p->setDisplayRange(0.0, 0.5);
            p->setIncrement(0.05);
            p->setHint("Percentage of glare light (typ. 0.10-0.25). Value is in percent, not fraction.");
            if (grpGlare) p->setParent(*grpGlare);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kGlareRoughness);
            p->setLabel("Glare roughness");
            p->setDefault(0.4);
            p->setRange(0.0, 1.0);
            p->setDisplayRange(0.0, 1.0);
            p->setHint("Glare roughness (0-1). Stddev = roughness * percent.");
            if (grpGlare) p->setParent(*grpGlare);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kGlareBlurSigmaPx);
            p->setLabel("Glare blur sigma (px)");
            p->setDefault(0.5);
            p->setRange(0.0, 10.0);
            p->setDisplayRange(0.0, 3.0);
            p->setHint("Gaussian blur sigma in pixels applied to the glare field.");
            if (grpGlare) p->setParent(*grpGlare);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kGlareCompensationRemovalFactor);
            p->setLabel("Compensation removal factor");
            p->setDefault(0.0);
            p->setRange(0.0, 1.0);
            p->setDisplayRange(0.0, 0.2);
            p->setIncrement(0.05);
            p->setHint("Remove viewing glare compensation from print curves. 0.2 = 20% underexposed shadows. Intended as alternative to stochastic glare (set GlarePercent=0).");
            if (grpGlare) p->setParent(*grpGlare);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kGlareCompensationRemovalDensity);
            p->setLabel("Compensation removal density");
            p->setDefault(1.2);
            p->setRange(0.0, 3.0);
            p->setDisplayRange(0.8, 2.0);
            p->setHint("Density at which the compensation-removal transition is centered (typ. 1.0-1.5).");
            if (grpGlare) p->setParent(*grpGlare);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kGlareCompensationRemovalTransition);
            p->setLabel("Compensation removal transition");
            p->setDefault(0.3);
            p->setRange(0.0, 2.0);
            p->setDisplayRange(0.0, 0.8);
            p->setHint("Transition density range for compensation removal (typ. 0.1-0.5).");
            if (grpGlare) p->setParent(*grpGlare);
            p->setEvaluateOnChange(true);
        }
    }

    // Special group
    {
        OFX::GroupParamDescriptor* grpSpecial = desc.defineGroupParam("SpecialGroup");
        if (grpSpecial) {
            grpSpecial->setLabel("Special");
            grpSpecial->setOpen(false);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kPrintDminFactor);
            p->setLabel("Print Dmin");
            p->setDefault(0.4);
            p->setRange(0.0, 1.0);
            p->setDisplayRange(0.0, 1.0);
            p->setIncrement(0.2);
            p->setHint("Minimum density factor of the print paper (0-1), make the white less white.");
            if (grpSpecial) p->setParent(*grpSpecial);
            p->setEvaluateOnChange(true);
        }
    }

    // Illuminants
    {
        OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam("ReferenceIlluminant");
        p->setLabel("Reference illuminant");
        p->appendOption("D65");
        p->appendOption("D55");
        p->appendOption("D50");
        p->appendOption("TH-KG3-L");
        p->appendOption("T");
        p->appendOption("K75P");
        p->appendOption("Equal energy");
        p->setDefault(0);
        p->setEvaluateOnChange(true);
    }
    {
        OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam("EnlargerIlluminant");
        p->setLabel("Enlarger illuminant");
        p->appendOption("D65");
        p->appendOption("D55");
        p->appendOption("D50");
        p->appendOption("TH-KG3-L");
        p->appendOption("T");
        p->appendOption("K75P");
        p->appendOption("Equal energy");
        p->setDefault(3);
        p->setEvaluateOnChange(true);
    }

    // Output encoding group
    {
        OFX::GroupParamDescriptor* grpOutput = desc.defineGroupParam("OutputEncodingGroup");
        if (grpOutput) grpOutput->setLabel("Output encoding");

        {
            OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam(kParamOutputColorSpace);
            p->setLabel("Output color space");
            for (std::size_t i = 0; i < OutputEncoding::kColorSpaceCount; ++i) {
                p->appendOption(OutputEncoding::kColorSpaceLabels[i]);
            }
            p->setDefault(OutputEncoding::toIndex(OutputEncoding::ColorSpace::sRGB));
            if (grpOutput) p->setParent(*grpOutput);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::BooleanParamDescriptor* p = desc.defineBooleanParam(kParamOutputCctfEncoding);
            p->setLabel("Apply output CCTF");
            p->setDefault(true);
            if (grpOutput) p->setParent(*grpOutput);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::BooleanParamDescriptor* p = desc.defineBooleanParam(kParamOutputLinearPassThrough);
            p->setLabel("Output linear pass-through");
            p->setDefault(false);
            if (grpOutput) p->setParent(*grpOutput);
            p->setEvaluateOnChange(true);
        }
    }
}

OFX::ImageEffect* JuicerPluginFactory::createInstance(OfxImageEffectHandle handle, OFX::ContextEnum /*context*/)
{
    // Create our effect instance (constructor attaches InstanceState + runs bootstrap).
    return new JuicerEffect(handle);
}

// Resolve support library entry: register our factory.
void OFX::Plugin::getPluginIDs(OFX::PluginFactoryArray& arr) {
    static JuicerPluginFactory factory;
    arr.push_back(&factory);
}
