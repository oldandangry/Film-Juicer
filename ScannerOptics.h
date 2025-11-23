// ScannerOptics.h
#pragma once

#include <functional>
#include <vector>

#include "ofxsImageEffect.h"
#include "Scanner.h"
#include "OutputEncoding.h"

namespace OFX {
    class Image;
}

namespace ScannerOptics {

    struct PlaneView {
        float* r = nullptr;
        float* g = nullptr;
        float* b = nullptr;
        float* a = nullptr;
        int width = 0;
        int height = 0;
        std::ptrdiff_t strideBytes = 0;
        int originX = 0;
        int originY = 0;
    };

    struct GlareCache {
        std::vector<float> amount;
        std::vector<float> tmp;
        Scanner::ScannerKey key{};
        std::uint64_t seedHash = 0;
        int width = 0;
        int height = 0;
        bool valid = false;
    };

    struct Runtime {
        Scanner::SpectralLutBuffer lut;
        std::vector<float> blurKernel;
        std::vector<float> unsharpKernel;
        float unsharpAmount = 0.0f;
        GlareCache glare;
        Scanner::ScannerKey key{};
    };

    struct RenderAbortHandle {
        std::function<bool()> shouldAbort;
        bool abortRequested() const { return shouldAbort && shouldAbort(); }
    };

    struct RenderContext {
        const Scanner::ScannerMediumRuntime* medium = nullptr;
        const Scanner::ScannerDensityBuffer* density = nullptr;
        Runtime* runtime = nullptr;
        OFX::Image* srcImage = nullptr;
        OFX::Image* dstImage = nullptr;
        const Scanner::ColorRuntime* color = nullptr;
        int nComponents = 0;
        OfxRectI bounds{};
        Scanner::Options options{};
        Scanner::Settings settings{};
        Scanner::ScannerRuntimeKey runtimeKey{};
        Scanner::ScannerKey scannerKey{};
        std::uint64_t seedBase = 0;
        bool hasBaseline = false;
        unsigned int threadCount = 1;
        RenderAbortHandle abort{};
    };

    Scanner::ColorRuntime build_color_runtime(
        const Scanner::ScannerMediumRuntime& medium,
        const OutputEncoding::Params& outputEncoding);

    void render_density_to_rgb(const RenderContext& ctx);

} // namespace ScannerOptics
