// JuicerProcessing.h
#pragma once

#include "ofxsProcessing.h"
#include "ofxsImageEffect.h"
#include "SpectralData.h"
#include "PipelineTypes.h"
#include "Print.h"
#include "Scanner.h"
#include "Couplers.h"
#include "OutputColor.h"
#include "RenderFrameRequest.h"
#include <vector>
#include <cmath>
#include <cstdint>
#include <memory>

struct InstanceState;
struct WorkingState;

namespace JuicerProc {

    // Separable Gaussian kernel builder, with radius cap for safety.
    // Kept inline so test wrappers can use it without linking a specific TU.
    inline void buildGaussianKernel(float sigma, std::vector<float>& kernel) {
        kernel.clear();
        if (!(std::isfinite(sigma)) || sigma <= 0.0f) {
            kernel.push_back(1.0f);
            return;
        }

        const int radiusRaw = std::max(1, int(std::ceil(3.0f * sigma)));
        const int radius = std::min(radiusRaw, 75); // cap at 75 taps each side

        const size_t radiusSize = static_cast<size_t>(radius);
        kernel.resize(radiusSize * 2u + 1u);
        const float s2 = sigma * sigma * 2.0f;
        float wsum = 0.0f;
        size_t kernelIndex = 0;
        for (int i = -radius; i <= radius; ++i, ++kernelIndex) {
            const float iFloat = static_cast<float>(i);
            const float w = std::exp(-(iFloat * iFloat) / s2);
            kernel[kernelIndex] = w;
            wsum += w;
        }
        for (float& w : kernel)
            w /= wsum;
    }

} // namespace JuicerProc

// Full class declaration
class JuicerProcessor : public OFX::ImageProcessor {
public:
    explicit JuicerProcessor(OFX::ImageEffect& effect);

    using DirectFrameRequest = Spektrafilm::DirectFrameRequest;
    using PrintFrameRequest = Spektrafilm::PrintFrameRequest;
    using FrameRequest = Spektrafilm::FrameRequest;

    struct SourceDestinationImages {
        OFX::Image* src = nullptr;
        OFX::Image* dst = nullptr;
    };

    struct CameraAutoExposureSettings {
        bool enabled = false;
        int meteringMethod = 0;
        double sliderEV = 0.0;
    };

    struct SessionTokens {
        std::uint64_t sessionSeed = 1;
        std::uint64_t instanceToken = 1;
    };

    void setSrcDst(const SourceDestinationImages& images);
    void setDirectFrameRequest(const DirectFrameRequest& request);
    void setPrintFrameRequest(const PrintFrameRequest& request);
    void setFrameRequest(const FrameRequest& request);
    void setRenderWindowRect(const OfxRectI& rect);
    void setComponents(int n);
    void setScannerOptions(const Scanner::Options& o);
    void setScannerSettings(const Scanner::Settings& s);
    void setPrintParams(const Print::Params& p);
    void setHalationOverride(const Profiles::HalationMetadata& halation);
    void setPrintGlareOverride(const Profiles::ProfileGlare& glare);
    void setWorkingState(const WorkingState* ws, bool wsReady);
    void setPrintRuntime(const Print::Runtime* prt, bool printReady);
    void setExposure(float exposureScale);
    void setCameraAutoExposure(const CameraAutoExposureSettings& settings);
    void setAutoExposureMeterBounds(const OfxRectI& bounds, bool valid);
    void setOutputEncoding(const OutputEncoding::Params& p);
    void setInstanceState(InstanceState* s);
    void setSessionTokens(const SessionTokens& tokens);
    void setClipToken(std::uintptr_t token);
    void setFrameTime(double time);
    void setFrameRate(double frameRate);
    void setFrameBoundsVersion(std::uint32_t v);
    void setPixelSizeUm(float pixelSizeUm);
    void setRenderHints(bool interactiveRenderStatus, bool renderQualityDraft, bool sequentialRenderStatus);

    void process() override;
    void processImagesCUDA() override;

private:
    OFX::Image* _srcImg;
    int _nComponents;

    Scanner::Options _scannerOptions;
    Scanner::Settings _scannerSettings;
    Print::Params _printParams;
    Profiles::HalationMetadata _halationOverride{};
    bool _hasHalationOverride = false;
    Profiles::ProfileGlare _printGlareOverride{};
    bool _hasPrintGlareOverride = false;
    Couplers::Runtime _dirRT;

    const Print::Runtime* _prt;
    const WorkingState* _ws;
    std::shared_ptr<const WorkingState> _wsHold;
    std::shared_ptr<const RenderRecipe> _recipeHold;
    std::shared_ptr<const DirectRenderState> _directStateHold;
    std::shared_ptr<const PrintRenderState> _printStateHold;
    InstanceState* _instanceState = nullptr;
    bool _wsReady;
    bool _printReady;

    float _exposureScale;
    bool _cameraAutoEnabled = false;
    int _cameraMeteringMethod = 0;
    double _cameraSliderEV = 0.0;
    OfxRectI _autoExposureMeterBounds{0, 0, 0, 0};
    bool _autoExposureMeterBoundsValid = false;
    OutputEncoding::Params _outputEncoding;
    std::uint64_t _sessionSeed = 1;
    std::uint64_t _instanceToken = 1;
    std::uintptr_t _clipToken = 0;
    std::uint64_t _frameTimeHash = 0;
    std::int64_t _frameIndex = 0;
    double _timeFrames = 0.0;
    double _frameRate = 0.0;
    bool _renderInteractiveStatus = false;
    bool _renderQualityDraft = false;
    bool _renderSequentialStatus = false;

    std::uint32_t _frameBoundsVersion = 0;
    OfxRectI _fullFrameExtent{0, 0, 0, 0};
    std::optional<Spektrafilm::DiffusionFrameSetDescriptor> _diffusionFrameSetDescriptor;
    float _pixelSizeUm = 0.0f;
};
// Test-facing wrappers to access internal spatial utilities without changing production behavior.
namespace JuicerProcTest {

    // Build a separable Gaussian kernel and return it for assertions.
    // Why inline: ensures tests link even if production TU isn't linked.
    inline void makeGaussianKernel(float sigma, std::vector<float>& outKernel) {
        JuicerProc::buildGaussianKernel(sigma, outKernel);
    }

    // Curve sanity check (monotonic X, matching sizes), mirrors internal curve_ok().
    inline bool curveOk(const Spectral::Curve& c) {
        const size_t N = c.lambda_nm.size();
        if (N < 2 || c.linear.size() != N)
            return false;
        float prev = c.lambda_nm[0];
        if (!std::isfinite(prev))
            return false;
        for (size_t i = 1; i < N; ++i) {
            const float xi = c.lambda_nm[i];
            if (!std::isfinite(xi))
                return false;
            if (xi < prev)
                return false; // allow duplicates, never decreasing
            prev = xi;
        }
        return true;
    }

} // namespace JuicerProcTest
