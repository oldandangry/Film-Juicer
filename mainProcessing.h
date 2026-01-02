// JuicerProcessing.h
#pragma once

#include "ofxsProcessing.h"
#include "ofxsImageEffect.h"
#include "SpectralData.h"
#include "SpectralProcessing.h"
#include "FilmProcessing.h"
#include "ColorTransforms.h"
#include "WorkingState.h"
#include "PipelineTypes.h"
#include "Print.h"
#include "Scanner.h"
#include "Couplers.h"
#include "SpatialDIR.h"
#include "OutputEncoding.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>

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

struct InstanceState;

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
