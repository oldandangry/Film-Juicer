// ScannerOptics.h
#pragma once

#include <functional>

#include "ofxsImageEffect.h"
#include "Scanner.h"
#include "OutputEncoding.h"

namespace OFX {
    class Image;
}

namespace JuicerProc {
    struct StageScratch;
}

namespace ScannerOptics {

    struct RenderAbortHandle {
        std::function<bool()> shouldAbort;
        bool abortRequested() const { return shouldAbort && shouldAbort(); }
    };

    struct RenderContext {
        const Scanner::ScannerMediumRuntime* medium = nullptr;
        const Scanner::ScannerDensityBuffer* density = nullptr;
        JuicerProc::StageScratch* scratch = nullptr;
        OFX::Image* srcImage = nullptr;
        OFX::Image* dstImage = nullptr;
        int nComponents = 0;
        OfxRectI bounds{};
        Scanner::Options options{};
        Scanner::Settings settings{};
        OutputEncoding::Params encoding{};
        Scanner::ScannerRuntimeKey runtimeKey{};
        Scanner::ScannerKey scannerKey{};
        bool hasBaseline = false;
        unsigned int threadCount = 1;
        RenderAbortHandle abort{};
    };

    void render_density_to_rgb(const RenderContext& ctx);

} // namespace ScannerOptics

