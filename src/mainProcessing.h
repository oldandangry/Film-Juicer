// JuicerProcessing.h
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "ofxsImageEffect.h"
#include "ofxsProcessing.h"

#include "RenderRecipe.h"
#include "SpectralData.h"

struct InstanceState;
struct DirectRenderState;
struct PrintRenderState;

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

    struct SourceDestinationImages {
        OFX::Image* src = nullptr;
        OFX::Image* dst = nullptr;
    };

    struct DirectFrameRequest {
        std::shared_ptr<const DirectRenderState> state;
        std::optional<Spektrafilm::DiffusionFrameSetDescriptor> diffusionFrameSet;
        int components = 0;
        OfxRectI renderWindow{0, 0, 0, 0};
        OfxRectI fullFrameExtent{0, 0, 0, 0};
        std::uint64_t sessionSeed = 1;
        std::uint64_t instanceToken = 1;
        std::uintptr_t clipToken = 0;
        double frameTime = 0.0;
        double frameRate = 0.0;
        float pixelSizeUm = 0.0f;
    };

    struct PrintFrameRequest {
        std::shared_ptr<const PrintRenderState> state;
        std::optional<Spektrafilm::DiffusionFrameSetDescriptor> diffusionFrameSet;
        int components = 0;
        OfxRectI renderWindow{0, 0, 0, 0};
        OfxRectI fullFrameExtent{0, 0, 0, 0};
        std::uint64_t sessionSeed = 1;
        std::uint64_t instanceToken = 1;
        std::uintptr_t clipToken = 0;
        double frameTime = 0.0;
        double frameRate = 0.0;
        float pixelSizeUm = 0.0f;
    };

    void setSrcDst(const SourceDestinationImages& images);
    void setDirectFrameRequest(const DirectFrameRequest& request);
    void setPrintFrameRequest(const PrintFrameRequest& request);
    void setInstanceState(InstanceState* s);

    void process() override;
    void processImagesCUDA() override;

private:
    OFX::Image* _srcImg = nullptr;
    int _nComponents = 0;
    std::shared_ptr<const DirectRenderState> _directStateHold;
    std::shared_ptr<const PrintRenderState> _printStateHold;
    InstanceState* _instanceState = nullptr;
    std::uint64_t _sessionSeed = 1;
    std::uint64_t _instanceToken = 1;
    std::uintptr_t _clipToken = 0;
    std::int64_t _frameIndex = 0;
    double _timeFrames = 0.0;
    double _frameRate = 0.0;
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
