// JuicerProcessing.h
#pragma once

#include "ofxsProcessing.h"
#include "ofxsImageEffect.h"
#include "SpectralData.h"
#include "SpectralProcessing.h"
#include "FilmProcessing.h"
#include "ColorTransforms.h"
#include "WorkingState.h"
#include "Print.h"
#include "Scanner.h"
#include "Couplers.h"
#include "OutputEncoding.h"
#include <vector>
#include <algorithm>
#include <cmath>

namespace JuicerProc {
    void copyNonFloatRect(OFX::Image* src, OFX::Image* dst);

    // Separable Gaussian kernel builder, with radius cap for safety. Kept inline so
    // tests can exercise it without linking the processing translation unit.
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

    struct SpatialDIRWorkspace {
        std::vector<float> logE_B, logE_G, logE_R;
        std::vector<float> corrY, corrM, corrC;
        std::vector<float> corrYBlur, corrMBlur, corrCBlur;
        std::vector<float> tmp;
    };

    struct PrintPipelineScratch {
        std::vector<float> Tneg, Ee_expose, Ee_filtered, Tprint, Ee_viewed;
        std::vector<float> Tpreflash, Ee_preflash;
    };
}

// Full class declaration
class JuicerProcessor : public OFX::ImageProcessor {
public:
    explicit JuicerProcessor(OFX::ImageEffect& effect);

    void setSrcDst(OFX::Image* src, OFX::Image* dst);
    void setRenderWindowRect(const OfxRectI& rect);
    void setComponents(int n);
    void setScannerParams(const Scanner::Params& p);
    void setPrintParams(const Print::Params& p);
    void setDirRuntime(const Couplers::Runtime& rt);
    void setWorkingState(const WorkingState* ws, bool wsReady);
    void setPrintRuntime(const Print::Runtime* prt, bool printReady);
    void setExposure(float exposureScale);
    void setOutputEncoding(const OutputEncoding::Params& p);

    void multiThreadProcessImages(OfxRectI procWindow) override;

private:
    struct RenderContext {
        OfxRectI window{};
        int tileWidth = 0;
        int tileHeight = 0;
        bool useSpatialDIR = false;
        float exposureScaleSafe = 1.0f;
        float kMidSpectral = 1.0f;
    };

    RenderContext prepareRenderContext(const OfxRectI& procWindow) const;
    void renderSpatialDIR(const RenderContext& ctx);
    void renderScalar(const RenderContext& ctx);

    OFX::Image* _srcImg;
    int _nComponents;

    Scanner::Params _scannerParams;
    Print::Params _printParams;
    Couplers::Runtime _dirRT;

    const Print::Runtime* _prt;
    const WorkingState* _ws;
    bool _wsReady;
    bool _printReady;

    float _exposureScale;
    OutputEncoding::Params _outputEncoding;

    std::vector<float> _gaussianKernel;
    JuicerProc::SpatialDIRWorkspace _dirWorkspace;
    JuicerProc::PrintPipelineScratch _printScratch;
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
