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
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdint>

struct InstanceState;
struct WorkingState;

namespace JuicerProc {

    struct SpatialDIRWorkspace {
        std::vector<float> filmRaw_B, filmRaw_G, filmRaw_R;
        std::vector<float> corrY, corrM, corrC;
        std::vector<float> corrYBlur, corrMBlur, corrCBlur;
        std::vector<float> tmp;
    };

} // namespace JuicerProc

namespace SpatialDIR {

    struct Callbacks {
        void* user = nullptr;
        bool (*fetchRGB)(void* user, int xx, int yy, float rgb[3]) = nullptr;
        bool (*abortCheck)(void* user) = nullptr;
    };

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

        kernel.resize(size_t(2 * radius + 1));
        const float s2 = sigma * sigma * 2.0f;
        float wsum = 0.0f;
        for (int i = -radius; i <= radius; ++i) {
            const float w = std::exp(-(i * i) / s2);
            kernel[size_t(i + radius)] = w;
            wsum += w;
        }
        for (float& w : kernel) w /= wsum;
    }

    void buildSpatialDIRCorrections(
        int width,
        int height,
        const WorkingState& ws,
        const Couplers::Runtime& dirRT,
        float exposureScale,
        const Callbacks& callbacks,
        JuicerProc::SpatialDIRWorkspace& work,
        std::vector<float>& kernelCache);

} // namespace SpatialDIR

namespace JuicerProc {

    void copyNonFloatRect(OFX::Image* src, OFX::Image* dst);
    using DensityBuffer = Scanner::DensityBuffer;

    // Separable Gaussian kernel builder, with radius cap for safety. Kept inline so
    // tests can exercise it without linking the processing translation unit.
    inline void buildGaussianKernel(float sigma, std::vector<float>& kernel) {
        SpatialDIR::buildGaussianKernel(sigma, kernel);
    }

    using PrintPipelineScratch = Pipeline::PrintPipelineScratch;

    struct StageScratch {
        SpatialDIRWorkspace dirWorkspace;
        std::vector<float> gaussianKernel;
        std::vector<PrintPipelineScratch> printScratchPerWorker;
    };
}

// Full class declaration
class JuicerProcessor : public OFX::ImageProcessor {
public:
    explicit JuicerProcessor(OFX::ImageEffect& effect);

    void setSrcDst(OFX::Image* src, OFX::Image* dst);
    void setRenderWindowRect(const OfxRectI& rect);
    void setComponents(int n);
    void setScannerOptions(const Scanner::Options& o);
    void setScannerSettings(const Scanner::Settings& s);
    void setPrintParams(const Print::Params& p);
    void setHalationOverride(const Profiles::HalationMetadata& halation);
    void setGrainOverride(const Profiles::GrainMetadata& grain);
    void setPrintGlareOverride(const Profiles::ProfileGlare& glare);
    void setDirRuntime(const Couplers::Runtime& rt);
    void setWorkingState(const WorkingState* ws, bool wsReady);
    void setPrintRuntime(const Print::Runtime* prt, bool printReady);
    void setExposure(float exposureScale);
    void setCameraAutoExposure(bool enabled, int meteringMethod, double sliderEV);
    void setOutputEncoding(const OutputEncoding::Params& p);
    void setInstanceState(InstanceState* s);
    void setClipToken(std::uintptr_t token);
    void setGateWeaveAmount(double amount);
    void setFrameTime(double time);
    void setFrameRate(double frameRate);
    void setFrameBoundsVersion(std::uint32_t v);
    void setPixelSizeUm(float pixelSizeUm);
    void setRenderHints(bool interactiveRenderStatus, bool renderQualityDraft, bool sequentialRenderStatus);

    void process() override;
    void multiThreadProcessImages(OfxRectI procWindow) override;
    void processImagesCUDA() override;

private:
    struct RenderContext {
        OfxRectI window{};
        int width = 0;
        int height = 0;
        bool useSpatialDIR = false;
        bool printActive = false;
        float exposureScaleSafe = 1.0f;
        float kMidSpectral = 1.0f;
        float pixelSizeUm = 0.0f;
    };

    RenderContext prepareRenderContext() const;
    bool ensureDensityCapacity(int width, int height);
    void writeMediumDensities(const RenderContext& ctx, unsigned int threadCount);
    void renderScannerFromDensity(const RenderContext& ctx, unsigned int threadCount);

	    void processImpl();

    OFX::Image* _srcImg;
    int _nComponents;

    Scanner::Options _scannerOptions;
    Scanner::Settings _scannerSettings;
    Print::Params _printParams;
    Profiles::HalationMetadata _halationOverride{};
    bool _hasHalationOverride = false;
    Profiles::GrainMetadata _grainOverride{};
    bool _hasGrainOverride = false;
    Profiles::ProfileGlare _printGlareOverride{};
    bool _hasPrintGlareOverride = false;
    Couplers::Runtime _dirRT;

    const Print::Runtime* _prt;
    const WorkingState* _ws;
    InstanceState* _instanceState = nullptr;
    bool _wsReady;
    bool _printReady;

    float _exposureScale;
    bool _cameraAutoEnabled = false;
    int _cameraMeteringMethod = 0;
    double _cameraSliderEV = 0.0;
    OutputEncoding::Params _outputEncoding;
    std::uintptr_t _clipToken = 0;
    std::uint64_t _frameTimeHash = 0;
    std::int64_t _frameIndex = 0;
    double _timeFrames = 0.0;
    double _frameRate = 0.0;
    double _gateWeaveAmount = 1.0;
    bool _renderInteractiveStatus = false;
    bool _renderQualityDraft = false;
    bool _renderSequentialStatus = false;

    JuicerProc::StageScratch _scratch;
    JuicerProc::DensityBuffer _density;
    std::uint32_t _frameBoundsVersion = 0;
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
        if (N < 2 || c.linear.size() != N) return false;
        float prev = c.lambda_nm[0];
        if (!std::isfinite(prev)) return false;
        for (size_t i = 1; i < N; ++i) {
            const float xi = c.lambda_nm[i];
            if (!std::isfinite(xi)) return false;
            if (xi < prev) return false; // allow duplicates, never decreasing
            prev = xi;
        }
        return true;
    }

}
