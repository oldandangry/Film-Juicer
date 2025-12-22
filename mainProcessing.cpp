// JuicerProcessing.cpp

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <string>
#include <atomic>
#include <sstream>
#include <mutex>

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
#include <cuda_runtime.h>
#endif

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__) && defined(JUICER_CUDA_SELF_CHECK) && (JUICER_CUDA_SELF_CHECK != 0)
#include "Cuda/JuicerCudaSelfCheck.h"
#endif

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
#include "Cuda/JuicerCudaResources.h"
#include "Cuda/JuicerCudaPayloads.h"
#include "GeneratedColorSpaces.h"
#endif

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
extern "C" cudaError_t juicer_cuda_negative_pipeline(
    const JuicerCuda::PipelineRunParams* hParams,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_build_spatial_dir(
    const JuicerCuda::PipelineRunParams* hParams,
    float* dCorrY,
    float* dCorrM,
    float* dCorrC,
    float* dTmp,
    const float* dKernel,
    int kernelRadius,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_negative_pipeline_optics(
    const JuicerCuda::PipelineRunParams* hParams,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    float* dScratchBlurred,
    const float* dLensBlurKernel,
    int lensBlurRadius,
    const float* dUnsharpKernel,
    int unsharpRadius,
    float unsharpAmount,
    int glareOriginX,
    int glareOriginY,
    std::uint64_t glareSeed,
    float glarePercent,
    float glareRoughness,
    const float* dGlareKernel,
    int glareRadius,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_print_pipeline(
    const JuicerCuda::PipelineRunParams* hParams,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_print_pipeline_optics(
    const JuicerCuda::PipelineRunParams* hParams,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    float* dScratchBlurred,
    const float* dLensBlurKernel,
    int lensBlurRadius,
    const float* dUnsharpKernel,
    int unsharpRadius,
    float unsharpAmount,
    int glareOriginX,
    int glareOriginY,
    std::uint64_t glareSeed,
    float glarePercent,
    float glareRoughness,
    const float* dGlareKernel,
    int glareRadius,
    void* cudaStreamOpaque);
#endif

// Resolve OFX support library C++ wrappers — suppress MSVC C5040 for dynamic exception specs
#pragma warning(push)
#pragma warning(disable: 5040)
#include "ofxsProcessing.h"
#include "ofxsImageEffect.h"
#pragma warning(pop)
#include "Logging.h"
#include "Hash.h"
#include "SpectralData.h"
#include "SpectralProcessing.h"
#include "FilmProcessing.h"
#include "ColorTransforms.h"
#include "WorkingState.h"
#include "Print.h"
#include "JuicerState.h"
#include "Scanner.h"
#include "OutputEncoding.h"
#include "ScannerOptics.h"
#include "Couplers.h"
#include "mainProcessing.h"
#include "PipelineRunner.h"

namespace JuicerProc {

    // Copied from main.cpp helper, unchanged behavior.
    void copyNonFloatRect(OFX::Image* src, OFX::Image* dst) {
        const OfxRectI bounds = src->getBounds();
        const OFX::PixelComponentEnum comps = src->getPixelComponents();
        const OFX::BitDepthEnum depth = src->getPixelDepth();

        int nComponents = 0;
        switch (comps) {
        case OFX::ePixelComponentRGBA: nComponents = 4; break;
        case OFX::ePixelComponentRGB:  nComponents = 3; break;
        case OFX::ePixelComponentAlpha:nComponents = 1; break;
        default: return;
        }
        int bytesPerComp = 0;
        switch (depth) {
        case OFX::eBitDepthUByte:  bytesPerComp = 1; break;
        case OFX::eBitDepthUShort: bytesPerComp = 2; break;
        case OFX::eBitDepthFloat:  bytesPerComp = 4; break;
        default: return;
        }
        const size_t bytesPerPixel = size_t(nComponents * bytesPerComp);
        for (int y = bounds.y1; y < bounds.y2; ++y) {
            for (int x = bounds.x1; x < bounds.x2; ++x) {
                const void* s = src->getPixelAddress(x, y);
                void* d = dst->getPixelAddress(x, y);
                if (!s || !d) continue;
                std::memcpy(d, s, bytesPerPixel);
            }
        }
    }

}

// --- Spatial DIR: defensive curve utilities (monotonic + robust interpolation) ---

static unsigned int compute_thread_count(int width, int height) {
    if (width <= 0 || height <= 0) {
        return 1u;
    }
    const unsigned int w = static_cast<unsigned int>(width);
    const unsigned int h = static_cast<unsigned int>(height);
    unsigned int nCPUs = (std::min(w, 4096u) * h) / 4096u;
    if (nCPUs == 0) {
        nCPUs = 1;
    }
    const unsigned int maxThreads = OFX::MultiThread::getNumCPUs();
    if (maxThreads > 0) {
        nCPUs = std::min(nCPUs, maxThreads);
    }
    return std::max(1u, nCPUs);
}

static std::uint64_t hash_scanner_settings(const Scanner::Settings& settings, const Scanner::Options& options) {
    const std::uint64_t lutHash = Hash::hash_bytes(&settings.useLut, sizeof(settings.useLut));
    const float fields[3] = {
        options.lensBlurSigmaPx,
        options.unsharpSigmaPx,
        options.unsharpAmount
    };
    const std::uint64_t optHash = Hash::hash_float_span(fields, 3);
    if (lutHash == 0 || optHash == 0) {
        JTRACE("HASH", "FATAL: invalid scanner settings for hashing");
        return 0;
    }
    const std::uint64_t combined[2] = { lutHash, optHash };
    return Hash::hash_bytes(combined, sizeof(combined));
}

static inline bool curve_ok(const Spectral::Curve& c) {
    const size_t N = c.lambda_nm.size();
    if (N < 2 || c.linear.size() != N) return false;
    float prev = c.lambda_nm[0];
    if (!std::isfinite(prev)) return false;
    for (size_t i = 1; i < N; ++i) {
        float xi = c.lambda_nm[i];
        if (!std::isfinite(xi)) return false;
        if (xi < prev) return false; // allow duplicates (xi == prev), but never decreasing
        prev = xi;
    }
    return true;
}


// JuicerProcessor method definitions matching JuicerProcessing.h

JuicerProcessor::JuicerProcessor(OFX::ImageEffect& effect)
    : OFX::ImageProcessor(effect)
    , _srcImg(nullptr)
    , _nComponents(0)
    , _scannerOptions{}
    , _scannerSettings{}
    , _printParams{}
    , _halationOverride{}
    , _hasHalationOverride(false)
    , _grainOverride{}
    , _hasGrainOverride(false)
    , _printGlareOverride{}
    , _hasPrintGlareOverride(false)
    , _dirRT{}
    , _prt(nullptr)
    , _ws(nullptr)
    , _wsReady(false)
    , _printReady(false)
    , _exposureScale(1.0f)
    , _outputEncoding{}
    , _scratch{}
    , _density{}
    , _frameBoundsVersion(0)
    , _pixelSizeUm(0.0f)
{
}

void JuicerProcessor::setSrcDst(OFX::Image* src, OFX::Image* dst) {
    _srcImg = src;
    setDstImg(dst);
}

void JuicerProcessor::setRenderWindowRect(const OfxRectI& rect) { setRenderWindow(rect); }
void JuicerProcessor::setComponents(int n) { _nComponents = n; }
void JuicerProcessor::setScannerOptions(const Scanner::Options& o) { _scannerOptions = o; }
void JuicerProcessor::setScannerSettings(const Scanner::Settings& s) { _scannerSettings = s; }
void JuicerProcessor::setPrintParams(const Print::Params& p) { _printParams = p; }
void JuicerProcessor::setHalationOverride(const Profiles::HalationMetadata& halation) {
    _halationOverride = halation;
    _hasHalationOverride = true;
}
void JuicerProcessor::setGrainOverride(const Profiles::GrainMetadata& grain) {
    _grainOverride = grain;
    _hasGrainOverride = true;
}
void JuicerProcessor::setPrintGlareOverride(const Profiles::ProfileGlare& glare) {
    _printGlareOverride = glare;
    _printGlareOverride.compensationRemovalFactor = 0.0f;
    _printGlareOverride.compensationRemovalDensity = 0.0f;
    _printGlareOverride.compensationRemovalTransition = 0.0f;
    _hasPrintGlareOverride = true;
}
void JuicerProcessor::setDirRuntime(const Couplers::Runtime& rt) { _dirRT = rt; }
void JuicerProcessor::setWorkingState(const WorkingState* ws, bool wsReady) {
    _ws = ws;
    _wsReady = wsReady;
    // Align DIR normalization constants to per-instance maxima if available
    if (_wsReady && _ws) {
        _dirRT.dMax[0] = (std::isfinite(_ws->dMax[0]) && _ws->dMax[0] > 1e-4f) ? _ws->dMax[0] : 1.0f;
        _dirRT.dMax[1] = (std::isfinite(_ws->dMax[1]) && _ws->dMax[1] > 1e-4f) ? _ws->dMax[1] : 1.0f;
        _dirRT.dMax[2] = (std::isfinite(_ws->dMax[2]) && _ws->dMax[2] > 1e-4f) ? _ws->dMax[2] : 1.0f;
    }
}
void JuicerProcessor::setPrintRuntime(const Print::Runtime* prt, bool printReady) { _prt = prt; _printReady = printReady; }
void JuicerProcessor::setExposure(float exposureScale) {
    _exposureScale = exposureScale;
}

void JuicerProcessor::setOutputEncoding(const OutputEncoding::Params& p) {
    _outputEncoding = p;
}

void JuicerProcessor::setInstanceState(InstanceState* s) {
    _instanceState = s;
}

void JuicerProcessor::setClipToken(std::uintptr_t token) {
    _clipToken = token;
}

void JuicerProcessor::setFrameTime(double time) {
    _frameTimeHash = Hash::hash_bytes(&time, sizeof(time));
}

void JuicerProcessor::setFrameBoundsVersion(std::uint32_t v) {
    _frameBoundsVersion = v;
}

void JuicerProcessor::setPixelSizeUm(float pixelSizeUm) {
    _pixelSizeUm = pixelSizeUm;
}

JuicerProcessor::RenderContext JuicerProcessor::prepareRenderContext() const {
    RenderContext ctx{};
    ctx.window = _renderWindow;
    ctx.width = _renderWindow.x2 - _renderWindow.x1;
    ctx.height = _renderWindow.y2 - _renderWindow.y1;
    ctx.exposureScaleSafe = (std::isfinite(_exposureScale) && _exposureScale > 0.0f)
        ? _exposureScale
        : 1.0f;
    ctx.useSpatialDIR = (_dirRT.active && std::isfinite(_dirRT.spatialSigmaPixels) &&
        _dirRT.spatialSigmaPixels > 0.0f && _nComponents >= 3 && _wsReady && _ws);
    ctx.printActive = (_wsReady && _ws && _printReady && _prt && !_printParams.bypass);

    ctx.kMidSpectral = 1.0f;
    if (ctx.printActive) {
        const float exposureCompScale = _printParams.exposureCompensationEnabled
            ? _printParams.exposureCompensationScale
            : 1.0f;
        ctx.kMidSpectral = Pipeline::PipelineRunner::compute_midgray_factor(
            *_ws,
            *_prt,
            _printParams,
            _dirRT,
            exposureCompScale);
    }

    ctx.pixelSizeUm = (std::isfinite(_pixelSizeUm) && _pixelSizeUm > 0.0f) ? _pixelSizeUm : 0.0f;
    return ctx;
}

bool JuicerProcessor::ensureDensityCapacity(int width, int height) {
    if (width <= 0 || height <= 0) {
        return false;
    }
    const size_t planeSize = size_t(width) * size_t(height);
    try {
        _density.c.resize(planeSize);
        _density.m.resize(planeSize);
        _density.y.resize(planeSize);
    }
    catch (...) {
        JTRACE("SCAN", "FATAL: failed to allocate density slab");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
    _density.width = width;
    _density.height = height;
    _density.originX = _renderWindow.x1;
    _density.originY = _renderWindow.y1;
    _density.stride = width;
    return true;
}

void JuicerProcessor::writeMediumDensities(const RenderContext& ctx, unsigned int threadCount) {
    if (!_ws || !_wsReady || ctx.width <= 0 || ctx.height <= 0) {
        return;
    }

    _density.medium = ctx.printActive
        ? Scanner::ScannerMedium::Print
        : Scanner::ScannerMedium::Negative;

    if (ctx.useSpatialDIR) {
        struct SpatialDIRUser {
            OFX::ImageEffect* effect = nullptr;
            OFX::Image* srcImg = nullptr;
            OfxRectI window{};
        };
        SpatialDIRUser user{};
        user.effect = &_effect;
        user.srcImg = _srcImg;
        user.window = ctx.window;

        SpatialDIR::Callbacks callbacks{};
        callbacks.user = &user;
        callbacks.fetchRGB = [](void* u, int xx, int yy, float rgb[3]) -> bool {
            auto* self = static_cast<SpatialDIRUser*>(u);
            const int x = self->window.x1 + xx;
            const int y = self->window.y1 + yy;
            const float* srcPix = reinterpret_cast<const float*>(self->srcImg->getPixelAddress(x, y));
            if (!srcPix) return false;
            rgb[0] = srcPix[0];
            rgb[1] = srcPix[1];
            rgb[2] = srcPix[2];
            return true;
            };
        callbacks.abortCheck = [](void* u) -> bool {
            auto* self = static_cast<SpatialDIRUser*>(u);
            return self->effect->abort();
            };

        SpatialDIR::buildSpatialDIRCorrections(
            ctx.width,
            ctx.height,
            *_ws,
            _dirRT,
            ctx.exposureScaleSafe,
            callbacks,
            _scratch.dirWorkspace,
            _scratch.gaussianKernel);
    }

    std::atomic<bool> abortFlag{ false };
    std::atomic<bool> failure{ false };
    const int width = ctx.width;
    const int height = ctx.height;
    const int originX = ctx.window.x1;
    const int originY = ctx.window.y1;

    const unsigned int nThreads = std::max(1u, threadCount);
    if (ctx.printActive) {
        _scratch.printScratchPerWorker.resize(nThreads);
    }

    Pipeline::PipelineRunnerConfig runnerCfg{};
    runnerCfg.enablePrint = ctx.printActive;
    const Pipeline::PipelineRunner runner(runnerCfg);

    struct DensityProcessor final : OFX::MultiThread::Processor {
        JuicerProcessor& self;
        const RenderContext& ctx;
        const Pipeline::PipelineRunner& runner;
        std::atomic<bool>& abortFlag;
        std::atomic<bool>& failure;
        const int width;
        const int height;
        const int originX;
        const int originY;

        DensityProcessor(
            JuicerProcessor& self_,
            const RenderContext& ctx_,
            const Pipeline::PipelineRunner& runner_,
            std::atomic<bool>& abortFlag_,
            std::atomic<bool>& failure_,
            int width_,
            int height_,
            int originX_,
            int originY_)
            : self(self_)
            , ctx(ctx_)
            , runner(runner_)
            , abortFlag(abortFlag_)
            , failure(failure_)
            , width(width_)
            , height(height_)
            , originX(originX_)
            , originY(originY_)
        {
        }

        void multiThreadFunction(unsigned int threadId, unsigned int nThreads) override {
            const int rowsPerThread = (height + int(nThreads) - 1) / int(nThreads);
            const int yStart = rowsPerThread * int(threadId);
            if (yStart >= height) {
                return;
            }
            const int yEnd = std::min(height, rowsPerThread * int(threadId + 1));

            JuicerProc::PrintPipelineScratch* printScratch = nullptr;
            if (ctx.printActive) {
                if (threadId < self._scratch.printScratchPerWorker.size()) {
                    printScratch = &self._scratch.printScratchPerWorker[threadId];
                }
            }

            for (int yOff = yStart; yOff < yEnd && !abortFlag.load(std::memory_order_relaxed); ++yOff) {
                if (self._effect.abort()) {
                    abortFlag.store(true, std::memory_order_relaxed);
                    break;
                }
                const int y = originY + yOff;
                const size_t rowOffset = size_t(yOff) * size_t(width);
                for (int xOff = 0; xOff < width; ++xOff) {
                    if (abortFlag.load(std::memory_order_relaxed)) {
                        break;
                    }
                    const int x = originX + xOff;
                    const size_t idx = rowOffset + size_t(xOff);

                    Pipeline::DensityPixelInputs pxIn{};
                    pxIn.exposureScale = ctx.exposureScaleSafe;
                    pxIn.dirRuntime = &self._dirRT;
                    pxIn.applyDirRuntime = true;

                    if (ctx.useSpatialDIR) {
                        pxIn.useFilmRawOverride = true;
                        pxIn.filmRawOverride.v[0] = self._scratch.dirWorkspace.filmRaw_B[idx];
                        pxIn.filmRawOverride.v[1] = self._scratch.dirWorkspace.filmRaw_G[idx];
                        pxIn.filmRawOverride.v[2] = self._scratch.dirWorkspace.filmRaw_R[idx];
                        pxIn.useSpatialDIR = true;
                        pxIn.spatialLogECorrectionsYMC[0] = self._scratch.dirWorkspace.corrYBlur[idx];
                        pxIn.spatialLogECorrectionsYMC[1] = self._scratch.dirWorkspace.corrMBlur[idx];
                        pxIn.spatialLogECorrectionsYMC[2] = self._scratch.dirWorkspace.corrCBlur[idx];
                    }
                    else {
                        const float* srcPix = reinterpret_cast<const float*>(self._srcImg->getPixelAddress(x, y));
                        if (!srcPix) {
                            self._density.c[idx] = 0.0f;
                            self._density.m[idx] = 0.0f;
                            self._density.y[idx] = 0.0f;
                            continue;
                        }
                        pxIn.rgb.v[0] = srcPix[0];
                        pxIn.rgb.v[1] = srcPix[1];
                        pxIn.rgb.v[2] = srcPix[2];
                    }

                    if (ctx.printActive) {
                        pxIn.printRuntime = self._prt;
                        pxIn.printParams = &self._printParams;
                        pxIn.midgrayFactor = ctx.kMidSpectral;
                        pxIn.printScratch = printScratch;
                    }

                    Pipeline::DensityPixelOutputs pxOut{};
                    if (!runner.run_density_pixel(*self._ws, pxIn, pxOut)) {
                        if (ctx.printActive) {
                            failure.store(true, std::memory_order_relaxed);
                            abortFlag.store(true, std::memory_order_relaxed);
                            break;
                        }
                        self._density.c[idx] = 0.0f;
                        self._density.m[idx] = 0.0f;
                        self._density.y[idx] = 0.0f;
                        continue;
                    }

                    if (ctx.printActive) {
                        if (pxOut.medium != Pipeline::DensityMedium::Print) {
                            failure.store(true, std::memory_order_relaxed);
                            abortFlag.store(true, std::memory_order_relaxed);
                            break;
                        }
                        self._density.c[idx] = pxOut.printDensity.v[0];
                        self._density.m[idx] = pxOut.printDensity.v[1];
                        self._density.y[idx] = pxOut.printDensity.v[2];
                    }
                    else {
                        self._density.c[idx] = pxOut.negativeDensity.v[0];
                        self._density.m[idx] = pxOut.negativeDensity.v[1];
                        self._density.y[idx] = pxOut.negativeDensity.v[2];
                    }
                }
            }
        }
    };

    DensityProcessor densityProcessor(
        *this,
        ctx,
        runner,
        abortFlag,
        failure,
        width,
        height,
        originX,
        originY);
    densityProcessor.multiThread(nThreads);

    if (failure.load(std::memory_order_relaxed)) {
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
    if (abortFlag.load(std::memory_order_relaxed)) {
        return;
    }
}


void JuicerProcessor::renderScannerFromDensity(const RenderContext& ctx, unsigned int threadCount) {
    if (!_ws || !_wsReady) {
        return;
    }

    // The scanner now consumes only the staged CMY density slab; legacy RGB entry points are removed.
    if (ctx.printActive) {
        if (!_ws->printScannerValid) {
            JTRACE("SCAN", "FATAL: print scanner runtime invalid");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
    }
    else {
        if (!_ws->negativeScannerValid) {
            JTRACE("SCAN", "FATAL: negative scanner runtime invalid");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
    }
    Scanner::ScannerMediumRuntime printMediumOverride{};
    const Scanner::ScannerMediumRuntime* mediumRuntime = ctx.printActive
        ? &_ws->printMediumRuntime
        : &_ws->negativeMediumRuntime;
    if (ctx.printActive) {
        printMediumOverride = _ws->printMediumRuntime;
        if (_hasPrintGlareOverride) {
            printMediumOverride.glare.active = _printGlareOverride.active;
            printMediumOverride.glare.percent = _printGlareOverride.percent;
            printMediumOverride.glare.roughness = _printGlareOverride.roughness;
            printMediumOverride.glare.blur = _printGlareOverride.blur;
            printMediumOverride.glare.compensationRemovalFactor = 0.0f;
            printMediumOverride.glare.compensationRemovalDensity = 0.0f;
            printMediumOverride.glare.compensationRemovalTransition = 0.0f;
        }
        const std::uint64_t glareHash = Scanner::hash_glare(printMediumOverride.glare);
        if (glareHash == 0) {
            JTRACE("HASH", "FATAL: failed to hash print glare override parameters");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        printMediumOverride.staticKey.glareHash = glareHash;
        mediumRuntime = &printMediumOverride;
    }
    if (!mediumRuntime) {
        JTRACE("SCAN", "FATAL: scanner medium runtime missing");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
    if (_density.medium != mediumRuntime->medium) {
        JTRACE("SCAN", "FATAL: density slab medium does not match selected scanner medium");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
    Scanner::ScannerStaticKey staticKey = mediumRuntime->staticKey;

    Scanner::ScannerRuntimeKey runtimeKey{};
    runtimeKey.settingsHash = hash_scanner_settings(_scannerSettings, _scannerOptions);
    runtimeKey.frameBoundsVersion = _frameBoundsVersion;
    if (runtimeKey.settingsHash == 0) {
        JTRACE("HASH", "FATAL: scanner runtime settings hash invalid");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
    Scanner::finalize_runtime_key(runtimeKey);

    const Spectral::SpectralTables* tables = mediumRuntime->tables;
    if (!tables || tables->K <= 0) {
        JTRACE("SCAN", "FATAL: scanner spectral tables unavailable");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
    if (tables->tablesHash != staticKey.tablesHash) {
        JTRACE("SCAN", "FATAL: scanner tables hash mismatch for medium");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
    if (mediumRuntime->range.digest == 0) {
        JTRACE("SCAN", "FATAL: scanner density range missing or invalid");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
    const std::uint64_t illumHash = tables->illuminantHash;
    if (illumHash != 0 && mediumRuntime->illuminant.hash != 0 && illumHash != mediumRuntime->illuminant.hash) {
        JTRACE("SCAN", "FATAL: scanner illuminant hash mismatch for medium");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    const Scanner::ColorRuntime* colorPtr = mediumRuntime->color;
    if (!colorPtr || colorPtr->hash == 0) {
        JTRACE("HASH", "FATAL: scanner color runtime missing or invalid");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
    if (staticKey.colorRuntimeHash != colorPtr->hash) {
        JTRACE("HASH", "FATAL: scanner static key color hash mismatch");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
    Scanner::finalize_static_key(staticKey);
    if (staticKey.hash == 0) {
        JTRACE("SCAN", "FATAL: scanner static key missing or invalid");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    Scanner::ScannerKey scannerKey{};
    scannerKey.staticKey = staticKey;
    scannerKey.runtimeKey = runtimeKey;
    Scanner::finalize_scanner_key(scannerKey);
    if (scannerKey.hash == 0) {
        JTRACE("HASH", "FATAL: scanner combined key invalid");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    ScannerOptics::Runtime* opticsRuntime = nullptr;
    if (_instanceState) {
        std::lock_guard<std::mutex> lock(_instanceState->m);
        if (_ws == &_instanceState->workA) {
            opticsRuntime = &_instanceState->scannerRuntimeA;
        }
        else if (_ws == &_instanceState->workB) {
            opticsRuntime = &_instanceState->scannerRuntimeB;
        }
    }
    static ScannerOptics::Runtime fallbackRuntime;
    if (!opticsRuntime) {
        opticsRuntime = &fallbackRuntime;
    }

    const std::uint64_t buildCounter = _ws ? _ws->buildCounter : 0;
    const std::uint64_t seedFields[3] = {
        static_cast<std::uint64_t>(_clipToken),
        _frameTimeHash,
        buildCounter
    };
    std::uint64_t seedBase = Hash::hash_bytes(seedFields, sizeof(seedFields));
    if (seedBase == 0) {
        seedBase = 1;
    }

    ScannerOptics::RenderContext optCtx{};
    optCtx.medium = mediumRuntime;
    optCtx.density = &_density;
    optCtx.runtime = opticsRuntime;
    optCtx.srcImage = _srcImg;
    optCtx.color = colorPtr;
    optCtx.nComponents = _nComponents;
    optCtx.bounds = ctx.window;
    optCtx.copyAlpha = (_nComponents == 4);
    optCtx.dstView.originX = ctx.window.x1;
    optCtx.dstView.originY = ctx.window.y1;
    optCtx.dstView.width = ctx.width;
    optCtx.dstView.height = ctx.height;
    optCtx.dstView.strideBytes = _dstImg ? _dstImg->getRowBytes() : 0;
    optCtx.dstView.r = (_dstImg)
        ? reinterpret_cast<float*>(_dstImg->getPixelAddress(ctx.window.x1, ctx.window.y1))
        : nullptr;
    optCtx.dstView.g = optCtx.dstView.r;
    optCtx.dstView.b = optCtx.dstView.r;
    optCtx.dstView.a = (_dstImg && _nComponents == 4) ? optCtx.dstView.r + 3 : nullptr;
    optCtx.options = _scannerOptions;
    optCtx.settings = _scannerSettings;
    optCtx.runtimeKey = runtimeKey;
    optCtx.scannerKey = scannerKey;
    optCtx.seedBase = seedBase;
    optCtx.hasBaseline = (_ws ? _ws->hasBaseline : false);
    optCtx.threadCount = std::max(1u, threadCount);
    optCtx.abort.shouldAbort = [this]() -> bool { return _effect.abort(); };

    ScannerOptics::render_density_to_rgb(optCtx);
}

void JuicerProcessor::processImpl() {
    if (!_srcImg || !_dstImg) return;

    if (_isEnabledOpenCLRender || _isEnabledCudaRender || _isEnabledMetalRender) {
        // CPU-only staging layer: GPU/device paths are intentionally disabled until parity lands.
        JTRACE("SCAN", "FATAL: GPU paths are unsupported in scanner staging");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    if (_nComponents < 1) {
        return;
    }

    const bool wsReady = _wsReady && _ws;
    if (!wsReady) {
        JTRACE("BUILD", "FATAL: working state unavailable; cannot render");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
    if (_nComponents < 3) {
        for (int y = _renderWindow.y1; y < _renderWindow.y2; ++y) {
            for (int x = _renderWindow.x1; x < _renderWindow.x2; ++x) {
                float* dstPix = reinterpret_cast<float*>(_dstImg->getPixelAddress(x, y));
                const float* srcPix = reinterpret_cast<const float*>(_srcImg->getPixelAddress(x, y));
                if (!dstPix || !srcPix) {
                    continue;
                }
                if (_nComponents == 1) {
                    dstPix[0] = srcPix[0];
                }
            }
        }
        return;
    }

#if defined(JUICER_SPD_DEBUG)
    Spectral::spd_probe_reset();
#endif

    RenderContext ctx = prepareRenderContext();
    if (ctx.width <= 0 || ctx.height <= 0) {
        return;
    }

    const unsigned int threadCount = compute_thread_count(ctx.width, ctx.height);
    ensureDensityCapacity(ctx.width, ctx.height);
    writeMediumDensities(ctx, threadCount);
    if (_effect.abort()) {
        return;
    }
    renderScannerFromDensity(ctx, threadCount);
}

void JuicerProcessor::process() {
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
    // CUDA-only mode: refuse CPU/OpenCL/Metal entry points.
    if (!_isEnabledCudaRender) {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
#endif
    if (_isEnabledOpenCLRender || _isEnabledCudaRender || _isEnabledMetalRender) {
        OFX::ImageProcessor::process();
        return;
    }
    processImpl();
}

void JuicerProcessor::multiThreadProcessImages(OfxRectI) {
    processImpl();
}

void JuicerProcessor::processImagesCUDA() {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
    OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
#else
    enum class RenderMode {
        NegativeOnly,
        Print
    };

    if (!_srcImg || !_dstImg) {
        return;
    }

    if (!(_nComponents == 1 || _nComponents == 3 || _nComponents == 4)) {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }

    const OfxRectI srcBounds = _srcImg->getBounds();
    const OfxRectI dstBounds = _dstImg->getBounds();
    const OfxRectI win = _renderWindow;
    const int width = win.x2 - win.x1;
    const int height = win.y2 - win.y1;
    if (width <= 0 || height <= 0) {
        return;
    }

    const int bytesPerPixel = _nComponents * static_cast<int>(sizeof(float));
    const std::ptrdiff_t srcRowBytes = _srcImg->getRowBytes();
    const std::ptrdiff_t dstRowBytes = _dstImg->getRowBytes();
    if (srcRowBytes <= 0 || dstRowBytes <= 0) {
        JTRACE("CUDA", "FATAL: invalid row bytes for CUDA copy");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    const std::ptrdiff_t xSrc = static_cast<std::ptrdiff_t>(win.x1 - srcBounds.x1);
    const std::ptrdiff_t ySrc = static_cast<std::ptrdiff_t>(win.y1 - srcBounds.y1);
    const std::ptrdiff_t xDst = static_cast<std::ptrdiff_t>(win.x1 - dstBounds.x1);
    const std::ptrdiff_t yDst = static_cast<std::ptrdiff_t>(win.y1 - dstBounds.y1);

    const std::ptrdiff_t widthBytes = static_cast<std::ptrdiff_t>(width) * static_cast<std::ptrdiff_t>(bytesPerPixel);
    if (xSrc < 0 || ySrc < 0 || xDst < 0 || yDst < 0 || widthBytes <= 0) {
        JTRACE("CUDA", "FATAL: CUDA render window out of bounds");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    const unsigned char* srcBase = static_cast<const unsigned char*>(_srcImg->getPixelData());
    unsigned char* dstBase = static_cast<unsigned char*>(_dstImg->getPixelData());
    if (!srcBase || !dstBase) {
        JTRACE("CUDA", "FATAL: missing device pointers for CUDA render");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    int deviceId = -1;
    {
        cudaPointerAttributes srcAttr{};
        cudaError_t attrErr = cudaPointerGetAttributes(&srcAttr, srcBase);
#if CUDART_VERSION >= 10000
        if (attrErr == cudaSuccess) {
            deviceId = srcAttr.device;
        }
#else
        if (attrErr == cudaSuccess) {
            deviceId = srcAttr.device;
        }
#endif

        cudaPointerAttributes dstAttr{};
        cudaError_t dstAttrErr = cudaPointerGetAttributes(&dstAttr, dstBase);
        if (dstAttrErr == cudaSuccess && deviceId >= 0 && dstAttr.device != deviceId) {
            JTRACE("CUDA", "FATAL: source/destination device mismatch");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }

        if (deviceId < 0) {
            int cur = -1;
            cudaError_t devErr = cudaGetDevice(&cur);
            if (devErr != cudaSuccess || cur < 0) {
                JTRACE("CUDA", "FATAL: failed to determine CUDA device for OFX pointers");
                throw OFX::Exception::Suite(kOfxStatErrFatal);
            }
            deviceId = cur;
        }

        cudaError_t setErr = cudaSetDevice(deviceId);
        if (setErr != cudaSuccess) {
            const char* msg = cudaGetErrorString(setErr);
            JTRACE("CUDA", std::string("FATAL: cudaSetDevice failed: ") + (msg ? msg : "(unknown)"));
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
    }

    const unsigned char* srcPtr = srcBase + ySrc * srcRowBytes + xSrc * bytesPerPixel;
    unsigned char* dstPtr = dstBase + yDst * dstRowBytes + xDst * bytesPerPixel;

#if defined(JUICER_TRACE_CUDA)
    JTRACE("CUDA", "processImagesCUDA");
#endif

    const bool wsReady = _wsReady && _ws;
    if (!wsReady) {
        JTRACE("CUDA", "FATAL: working state unavailable; cannot serve CUDA render");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
    if (!_instanceState) {
        JTRACE("CUDA", "FATAL: instance state missing; cannot serve CUDA render");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    JuicerCuda::Resources* cudaResources = nullptr;
    {
        std::lock_guard<std::mutex> lock(_instanceState->cudaMutex);
        auto& slot = _instanceState->cudaByDevice[deviceId];
        if (!slot) {
            slot.reset(JuicerCuda::create());
            if (!slot) {
                JTRACE("CUDA", "FATAL: failed to allocate CUDA resources");
                throw OFX::Exception::Suite(kOfxStatErrFatal);
            }
        }
        cudaResources = slot.get();
    }

    std::string uploadError;
    {
        std::lock_guard<std::mutex> lock(_instanceState->cudaMutex);
        if (!cudaResources) {
            JTRACE("CUDA", "FATAL: CUDA resources missing after allocation");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        if (!JuicerCuda::ensure_uploaded(*cudaResources, *_ws, _pCudaStream, uploadError)) {
            JTRACE("CUDA", std::string("CUDA WorkingState upload failed: ") + uploadError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
            throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
            throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
        }
#if JUICER_TRACE_PRINT_SWAP
        {
            std::lock_guard<std::mutex> resLock(cudaResources->m);
            const std::uint64_t build = _ws ? _ws->buildCounter : 0;
            std::string msg = std::string("cuda upload build=") + std::to_string(build)
                + " uploaded=" + std::to_string(cudaResources->uploadedBuildCounter)
                + " printIllumBuild=" + std::to_string(cudaResources->printIllumBuildCounter)
                + " printPreflashBuild=" + std::to_string(cudaResources->printPreflashBuildCounter);
            JTRACE("PRINTDBG", msg);
        }
#endif
    }

#if defined(JUICER_CUDA_VALIDATE_PRIMITIVES) && (JUICER_CUDA_VALIDATE_PRIMITIVES != 0)
    {
        std::string validateError;
        std::lock_guard<std::mutex> lock(_instanceState->cudaMutex);
        if (!cudaResources) {
            JTRACE("CUDA", "FATAL: CUDA resources missing for validation");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        if (!JuicerCuda::validate_density_primitives(*cudaResources, *_ws, _pCudaStream, validateError)) {
            JTRACE("CUDA", std::string("FATAL: CUDA primitive validation failed: ") + validateError);
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }

        if (_ws && _printReady && _prt && !_printParams.bypass) {
            float kMidSpectral = 1.0f;
            {
                const float exposureCompScale = _printParams.exposureCompensationEnabled
                    ? _printParams.exposureCompensationScale
                    : 1.0f;
                kMidSpectral = Pipeline::PipelineRunner::compute_midgray_factor(
                    *_ws,
                    *_prt,
                    _printParams,
                    _dirRT,
                    exposureCompScale);
                if (!std::isfinite(kMidSpectral) || !(kMidSpectral > 0.0f)) {
                    kMidSpectral = 1.0f;
                }
            }
            if (!JuicerCuda::validate_print_primitives(*cudaResources, *_ws, *_prt, _printParams, kMidSpectral, _pCudaStream, validateError)) {
                JTRACE("CUDA", std::string("FATAL: CUDA print validation failed: ") + validateError);
                throw OFX::Exception::Suite(kOfxStatErrFatal);
            }
        }

        JuicerCuda::record_use(*cudaResources, _pCudaStream);
    }
#endif

#if defined(JUICER_CUDA_SELF_CHECK) && (JUICER_CUDA_SELF_CHECK != 0)
    // Runtime CUDA self-check.
    // This is intentionally a host-runtime probe (not JUICER_TESTS), and is designed to be easy
    // to remove later: disable JUICER_CUDA_SELF_CHECK or delete Cuda/JuicerCudaSelfCheck.*.
    static std::once_flag sSelfCheckOnce;
    static bool sSelfCheckOk = true;
    static const char* sSelfCheckErr = nullptr;
    std::call_once(sSelfCheckOnce, [&]() {
        const bool ok = juicer_cuda_runtime_self_check(_pCudaStream, &sSelfCheckErr);
        sSelfCheckOk = ok;
        if (!ok) {
            JTRACE("CUDA", std::string("CUDA self-check failed; forcing CPU fallback. Error: ") + (sSelfCheckErr ? sSelfCheckErr : "(unknown)"));
        } else {
            JTRACE("CUDA", "CUDA self-check passed");
        }
    });
    if (!sSelfCheckOk) {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
#endif

    if (_nComponents == 1) {
        if (srcPtr == dstPtr && srcRowBytes == dstRowBytes) {
            return;
        }
        const cudaStream_t stream = reinterpret_cast<cudaStream_t>(_pCudaStream);
        const cudaError_t err = cudaMemcpy2DAsync(
            dstPtr,
            static_cast<size_t>(dstRowBytes),
            srcPtr,
            static_cast<size_t>(srcRowBytes),
            static_cast<size_t>(widthBytes),
            static_cast<size_t>(height),
            cudaMemcpyDeviceToDevice,
            stream);
        if (err != cudaSuccess) {
            const char* msg = cudaGetErrorString(err);
            JTRACE("CUDA", std::string("FATAL: cudaMemcpy2DAsync failed: ") + (msg ? msg : "(unknown)"));
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        JuicerCuda::record_use(*cudaResources, _pCudaStream);
        return;
    }

    const RenderMode renderMode = _printParams.bypass ? RenderMode::NegativeOnly : RenderMode::Print;

    // RenderMode::NegativeOnly (PrintBypass=true).
    if (renderMode == RenderMode::NegativeOnly) {
        const bool scannerUseLut = _scannerSettings.useLut;
        const float lensBlurSigmaPx = _scannerOptions.lensBlurSigmaPx;
        const float unsharpSigmaPx = _scannerOptions.unsharpSigmaPx;
        const float unsharpAmount = _scannerOptions.unsharpAmount;
        const bool glareActive = _ws && _ws->negativeMediumRuntime.glare.active && (_ws->negativeMediumRuntime.glare.percent > 0.0f);
        const bool useSpatialDIR = (_dirRT.active && std::isfinite(_dirRT.spatialSigmaPixels) && _dirRT.spatialSigmaPixels > 0.0f);

        if (!_ws->negativeScannerValid) {
            JTRACE("CUDA", "FATAL: negative scanner runtime invalid");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        if (_ws->negativeColorRuntime.hash == 0) {
            JTRACE("CUDA", "FATAL: negative scanner color runtime invalid");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        if (!_ws->negativeMediumRuntime.tables || _ws->negativeMediumRuntime.tables->K <= 0) {
            JTRACE("CUDA", "FATAL: negative scanner spectral tables unavailable");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }

        JuicerCuda::PipelineRunParams run{};
        run.src = srcPtr;
        run.srcRowBytes = static_cast<std::size_t>(srcRowBytes);
        run.dst = dstPtr;
        run.dstRowBytes = static_cast<std::size_t>(dstRowBytes);
        run.width = width;
        run.height = height;
        run.nComponents = _nComponents;

        // Film raw conversion payload
        run.filmRaw.inputColorSpaceIndex = Spectral::inputColorSpaceToIndex(_ws->filmRaw.inputColorSpace);
        run.filmRaw.applyCctfDecoding = _ws->filmRaw.applyCctfDecoding ? 1 : 0;
        run.filmRaw.applyInputChromaticAdapt = _ws->filmRaw.applyInputChromaticAdapt ? 1 : 0;
        run.filmRaw.spectralUpsamplingMode = static_cast<int>(_ws->filmRaw.spectralUpsamplingMode);
        for (int i = 0; i < 9; ++i) {
            run.filmRaw.inputRGBToXYZ[i] = _ws->filmRaw.inputRGBToXYZ.m[i];
            run.filmRaw.inputXYZAdapt[i] = _ws->filmRaw.inputXYZAdapt.m[i];
        }
        run.filmRaw.midgrayScale = _ws->filmRaw.midgrayScale;
        for (int i = 0; i < 3; ++i) {
            run.filmRaw.refIllumWhiteXYZ[i] = _ws->filmRaw.refIllumWhiteXYZ[i];
        }

        run.filmExpose.exposureScale = _exposureScale;
        run.filmDevelop.gammaFactorB = _ws->gammaFactorB;
        run.filmDevelop.gammaFactorG = _ws->gammaFactorG;
        run.filmDevelop.gammaFactorR = _ws->gammaFactorR;
        run.filmDevelop.dirPrecorrected = _ws->dirPrecorrected ? 1 : 0;

        // DIR runtime payload.
        run.filmDevelop.dir.active = _dirRT.active ? 1 : 0;
        run.filmDevelop.dir.highShift = _dirRT.highShift;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                run.filmDevelop.dir.M[r * 3 + c] = _dirRT.M[r][c];
            }
        }
        for (int i = 0; i < 3; ++i) {
            run.filmDevelop.dir.dMax[i] = _dirRT.dMax[i];
        }

        // Scan color payload + output encoding
        {
            const Scanner::ColorRuntime& color = _ws->negativeColorRuntime;
            for (int i = 0; i < 9; ++i) {
                run.scanStage.scanColor.cat02[i] = color.cat02[i];
                run.scanStage.scanColor.xyzToRgb[i] = color.xyzToRgb[i];
            }
            for (int i = 0; i < 3; ++i) {
                run.scanStage.scanColor.illuminantXYZ[i] = color.illuminantXYZ[i];
            }

            run.scanStage.scanColor.encoding.outputColorSpaceIndex = OutputEncoding::toIndex(color.encoding.colorSpace);
            run.scanStage.scanColor.encoding.applyCctfEncoding = color.encoding.applyCctfEncoding ? 1 : 0;
            run.scanStage.scanColor.encoding.preserveLinearRange = color.encoding.preserveLinearRange ? 1 : 0;
            run.scanStage.scanColor.encoding.inputIsOutputSpace = color.encoding.inputIsOutputSpace ? 1 : 0;

            const auto& outSpace = GeneratedColorSpaces::get(color.encoding.colorSpace);
            run.scanStage.scanColor.encoding.cctf.kind = static_cast<int>(outSpace.cctf.kind);
            run.scanStage.scanColor.encoding.cctf.gamma = outSpace.cctf.gamma;
            run.scanStage.scanColor.encoding.cctf.a = outSpace.cctf.a;
            run.scanStage.scanColor.encoding.cctf.b = outSpace.cctf.b;
            run.scanStage.scanColor.encoding.cctf.c = outSpace.cctf.c;
            run.scanStage.scanColor.encoding.cctf.d = outSpace.cctf.d;
            run.scanStage.scanColor.encoding.cctf.linearCutoff = outSpace.cctf.linearCutoff;

            const OutputEncoding::Matrix3x3 dwgToOutput = OutputEncoding::dwg_to_output_matrix(color.encoding.colorSpace);
            for (int i = 0; i < 9; ++i) {
                run.scanStage.scanColor.encoding.dwgToOutput[i] = dwgToOutput.m[i];
            }
        }

        const cudaStream_t stream = _pCudaStream ? reinterpret_cast<cudaStream_t>(_pCudaStream) : nullptr;

        // Device pointers from the per-instance cache + kernel launch + record_use must be
        // serialized to avoid rebuild races before lastUseEvent is recorded.
        {
            std::lock_guard<std::mutex> lock(_instanceState->cudaMutex);
            if (!cudaResources) {
                JTRACE("CUDA", "FATAL: CUDA resources missing for negative pipeline");
                throw OFX::Exception::Suite(kOfxStatErrFatal);
            }

            run.filmDevelop.densB = { cudaResources->densB.x, cudaResources->densB.y, cudaResources->densB.n };
            run.filmDevelop.densG = { cudaResources->densG.x, cudaResources->densG.y, cudaResources->densG.n };
            run.filmDevelop.densR = { cudaResources->densR.x, cudaResources->densR.y, cudaResources->densR.n };
            run.filmDevelop.dirDensB = { cudaResources->dirDensB.x, cudaResources->dirDensB.y, cudaResources->dirDensB.n };
            run.filmDevelop.dirDensG = { cudaResources->dirDensG.x, cudaResources->dirDensG.y, cudaResources->dirDensG.n };
            run.filmDevelop.dirDensR = { cudaResources->dirDensR.x, cudaResources->dirDensR.y, cudaResources->dirDensR.n };
            run.filmExpose.sensB = { cudaResources->sensB.x, cudaResources->sensB.y, cudaResources->sensB.n };
            run.filmExpose.sensG = { cudaResources->sensG.x, cudaResources->sensG.y, cudaResources->sensG.n };
            run.filmExpose.sensR = { cudaResources->sensR.x, cudaResources->sensR.y, cudaResources->sensR.n };

            run.filmExpose.tablesAx = cudaResources->tablesAx;
            run.filmExpose.tablesAy = cudaResources->tablesAy;
            run.filmExpose.tablesAz = cudaResources->tablesAz;
            run.filmExpose.tablesK = cudaResources->tablesK;
            for (int i = 0; i < 9; ++i) {
                run.filmExpose.spdSInv[i] = cudaResources->spdSInv[i];
            }

            run.filmExpose.hanatosLut = cudaResources->hanatosLut;
            run.filmExpose.hanatosN = cudaResources->hanatosN;
            run.filmExpose.hanatosLutIntegrated = cudaResources->hanatosLutIntegrated;
            run.filmExpose.hanatosNIntegrated = cudaResources->hanatosNIntegrated;

            run.scanStage.scannerUseLut = scannerUseLut ? 1 : 0;
            run.scanStage.scanLutLogXYZ = nullptr;
            run.scanStage.scanLutRes = 0;
            if (run.scanStage.scannerUseLut) {
                std::string lutError;
                if (!JuicerCuda::ensure_scan_lut(*cudaResources, *_ws, true, _pCudaStream, lutError)) {
                    JTRACE("CUDA", std::string("CUDA scan LUT upload failed: ") + lutError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                }
                run.scanStage.scanLutLogXYZ = cudaResources->scanNegativeLut.logXYZ;
                run.scanStage.scanLutRes = static_cast<int>(cudaResources->scanNegativeLut.res);
                if (!run.scanStage.scanLutLogXYZ || run.scanStage.scanLutRes <= 0) {
                    JTRACE("CUDA", "FATAL: scan LUT missing after successful upload");
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
                }
            }

            // Negative scan tables payload
            run.scanStage.scanTables.epsC = cudaResources->scanNegative.tables.epsC;
            run.scanStage.scanTables.epsM = cudaResources->scanNegative.tables.epsM;
            run.scanStage.scanTables.epsY = cudaResources->scanNegative.tables.epsY;
            run.scanStage.scanTables.Ax = cudaResources->scanNegative.tables.Ax;
            run.scanStage.scanTables.Ay = cudaResources->scanNegative.tables.Ay;
            run.scanStage.scanTables.Az = cudaResources->scanNegative.tables.Az;
            run.scanStage.scanTables.baseMin = cudaResources->scanNegative.tables.baseMin;
            run.scanStage.scanTables.K = cudaResources->scanNegative.tables.K;
            run.scanStage.scanTables.hasBaseline = cudaResources->scanNegative.tables.hasBaseline;
            run.scanStage.scanTables.invYn = cudaResources->scanNegative.tables.invYn;
            run.scanStage.scanTables.mediumIsNegative = cudaResources->scanNegative.mediumIsNegative;
            for (int i = 0; i < 3; ++i) {
                run.scanStage.scanTables.min_cmy[i] = cudaResources->scanNegative.min_cmy[i];
                run.scanStage.scanTables.inv_max_cmy[i] = cudaResources->scanNegative.inv_max_cmy[i];
            }

            std::string scanFlagError;
            if (!JuicerCuda::ensure_scan_error_flag(*cudaResources, _pCudaStream, scanFlagError)) {
                JTRACE("CUDA", std::string("CUDA scan error flag allocation failed: ") + scanFlagError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
            }
            run.scanStage.scanErrorFlag = cudaResources->scanErrorFlag;
            if (!run.scanStage.scanErrorFlag) {
                JTRACE("CUDA", "FATAL: scan error flag missing after allocation");
                throw OFX::Exception::Suite(kOfxStatErrFatal);
            }

            cudaEvent_t scanEvent = cudaResources->scanErrorEventOpaque
                ? reinterpret_cast<cudaEvent_t>(cudaResources->scanErrorEventOpaque)
                : nullptr;
            if (cudaResources->scanErrorPending && scanEvent && cudaResources->scanErrorHost) {
                cudaError_t pollErr = cudaEventQuery(scanEvent);
                if (pollErr == cudaErrorNotReady) {
                    pollErr = cudaEventSynchronize(scanEvent);
                }
                if (pollErr != cudaSuccess) {
                    const char* msg = cudaGetErrorString(pollErr);
                    JTRACE("CUDA", std::string("CUDA scan error event sync failed: ") + (msg ? msg : "(unknown)"));
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
                }
                cudaResources->scanErrorPending = 0;
                if (*cudaResources->scanErrorHost != 0) {
                    JTRACE("CUDA", "FATAL: previous scan produced non-finite RGB");
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
                }
            }

            cudaError_t flagErr = cudaMemsetAsync(run.scanStage.scanErrorFlag, 0, sizeof(int), stream);
            if (flagErr != cudaSuccess) {
                const char* msg = cudaGetErrorString(flagErr);
                JTRACE("CUDA", std::string("CUDA scan error flag memset failed: ") + (msg ? msg : "(unknown)"));
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
            }

            run.filmDevelop.spatialDir.active = useSpatialDIR ? 1 : 0;
            run.filmDevelop.spatialDir.corrY = nullptr;
            run.filmDevelop.spatialDir.corrM = nullptr;
            run.filmDevelop.spatialDir.corrC = nullptr;
            if (useSpatialDIR) {
                std::string dirError;
                if (!JuicerCuda::ensure_spatial_dir_scratch(*cudaResources, width, height, _pCudaStream, dirError)) {
                    JTRACE("CUDA", std::string("CUDA spatial DIR scratch allocation failed: ") + dirError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                }
                if (!JuicerCuda::ensure_spatial_dir_kernel(*cudaResources, cudaResources->spatialDirKernel, _dirRT.spatialSigmaPixels, _pCudaStream, dirError)) {
                    JTRACE("CUDA", std::string("CUDA spatial DIR kernel upload failed: ") + dirError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                }

                run.filmDevelop.spatialDir.corrY = cudaResources->spatialDirScratch.corrY;
                run.filmDevelop.spatialDir.corrM = cudaResources->spatialDirScratch.corrM;
                run.filmDevelop.spatialDir.corrC = cudaResources->spatialDirScratch.corrC;

                cudaError_t dirErr = juicer_cuda_build_spatial_dir(
                    &run,
                    cudaResources->spatialDirScratch.corrY,
                    cudaResources->spatialDirScratch.corrM,
                    cudaResources->spatialDirScratch.corrC,
                    cudaResources->spatialDirScratch.tmp,
                    cudaResources->spatialDirKernel.weights,
                    cudaResources->spatialDirKernel.radius,
                    _pCudaStream);
                if (dirErr != cudaSuccess) {
                    const char* msg = cudaGetErrorString(dirErr);
                    JTRACE("CUDA", std::string("FATAL: spatial DIR build failed: ") + (msg ? msg : "(unknown)"));
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
                }
            }

            const bool wantGlare = glareActive;
            float glarePercent = 0.0f;
            float glareRoughness = 0.0f;
            float glareBlurSigmaPx = 0.0f;
            std::uint64_t glareSeed = 0;
            if (wantGlare && _ws) {
                glarePercent = _ws->negativeMediumRuntime.glare.percent;
                glareRoughness = _ws->negativeMediumRuntime.glare.roughness;
                glareBlurSigmaPx = _ws->negativeMediumRuntime.glare.blur;

                const std::uint64_t buildCounter = _ws->buildCounter;
                const std::uint64_t seedBaseFields[3] = {
                    static_cast<std::uint64_t>(_clipToken),
                    _frameTimeHash,
                    buildCounter
                };
                std::uint64_t seedBase = Hash::hash_bytes(seedBaseFields, sizeof(seedBaseFields));
                if (seedBase == 0) {
                    seedBase = 1;
                }

                const std::uint64_t glareFields[4] = {
                    seedBase,
                    static_cast<std::uint64_t>(_frameBoundsVersion),
                    _ws->negativeMediumRuntime.staticKey.glareHash,
                    static_cast<std::uint64_t>(Scanner::ScannerMedium::Negative)
                };
                glareSeed = Hash::hash_bytes(glareFields, sizeof(glareFields));
            }

            const bool wantLensBlur = std::isfinite(lensBlurSigmaPx) && lensBlurSigmaPx > 0.0f;
            const bool wantUnsharp = std::isfinite(unsharpSigmaPx) && unsharpSigmaPx > 0.0f &&
                std::isfinite(unsharpAmount) && unsharpAmount != 0.0f;
            const bool wantOptics = wantLensBlur || wantUnsharp || wantGlare;

            cudaError_t err = cudaSuccess;
            if (!wantOptics) {
                err = juicer_cuda_negative_pipeline(&run, _pCudaStream);
            }
            else {
                std::string opticsError;
                if (!JuicerCuda::ensure_optics_scratch(*cudaResources, width, height, wantUnsharp, _pCudaStream, opticsError)) {
                    JTRACE("CUDA", std::string("CUDA optics scratch allocation failed: ") + opticsError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                }
                if (!JuicerCuda::ensure_gaussian_kernel(*cudaResources, cudaResources->scannerLensBlurKernel, lensBlurSigmaPx, _pCudaStream, opticsError)) {
                    JTRACE("CUDA", std::string("CUDA lens blur kernel upload failed: ") + opticsError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                }
                if (!JuicerCuda::ensure_gaussian_kernel(*cudaResources, cudaResources->scannerUnsharpKernel, unsharpSigmaPx, _pCudaStream, opticsError)) {
                    JTRACE("CUDA", std::string("CUDA unsharp kernel upload failed: ") + opticsError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                }
                if (!JuicerCuda::ensure_gaussian_kernel(*cudaResources, cudaResources->scannerGlareKernel, wantGlare ? glareBlurSigmaPx : 0.0f, _pCudaStream, opticsError)) {
                    JTRACE("CUDA", std::string("CUDA glare kernel upload failed: ") + opticsError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                }

                err = juicer_cuda_negative_pipeline_optics(
                    &run,
                    cudaResources->scannerScratch.rgbR,
                    cudaResources->scannerScratch.rgbG,
                    cudaResources->scannerScratch.rgbB,
                    cudaResources->scannerScratch.tmp,
                    cudaResources->scannerScratch.blurred,
                    cudaResources->scannerLensBlurKernel.weights,
                    cudaResources->scannerLensBlurKernel.radius,
                    cudaResources->scannerUnsharpKernel.weights,
                    cudaResources->scannerUnsharpKernel.radius,
                    unsharpAmount,
                    win.x1,
                    win.y1,
                    glareSeed,
                    glarePercent,
                    glareRoughness,
                    cudaResources->scannerGlareKernel.weights,
                    cudaResources->scannerGlareKernel.radius,
                    _pCudaStream);
            }
            if (err != cudaSuccess) {
                const char* msg = cudaGetErrorString(err);
                JTRACE("CUDA", std::string("FATAL: negative pipeline kernel launch failed: ") + (msg ? msg : "(unknown)"));
                throw OFX::Exception::Suite(kOfxStatErrFatal);
            }

            if (cudaResources->scanErrorHost && scanEvent) {
                flagErr = cudaMemcpyAsync(cudaResources->scanErrorHost, run.scanStage.scanErrorFlag, sizeof(int), cudaMemcpyDeviceToHost, stream);
                if (flagErr != cudaSuccess) {
                    const char* msg = cudaGetErrorString(flagErr);
                    JTRACE("CUDA", std::string("CUDA scan error flag readback failed: ") + (msg ? msg : "(unknown)"));
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                }
                cudaError_t evErr = cudaEventRecord(scanEvent, stream);
                if (evErr != cudaSuccess) {
                    const char* msg = cudaGetErrorString(evErr);
                    JTRACE("CUDA", std::string("CUDA scan error event record failed: ") + (msg ? msg : "(unknown)"));
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
                }
                cudaResources->scanErrorPending = 1;
                cudaError_t pollErr = cudaEventQuery(scanEvent);
                if (pollErr == cudaSuccess) {
                    cudaResources->scanErrorPending = 0;
                    if (*cudaResources->scanErrorHost != 0) {
                        JTRACE("CUDA", "FATAL: negative pipeline scan produced non-finite RGB");
                        throw OFX::Exception::Suite(kOfxStatErrFatal);
                    }
                } else if (pollErr != cudaErrorNotReady) {
                    const char* msg = cudaGetErrorString(pollErr);
                    JTRACE("CUDA", std::string("CUDA scan error event query failed: ") + (msg ? msg : "(unknown)"));
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
                }
            }
            else {
                int scanError = 0;
                flagErr = cudaMemcpyAsync(&scanError, run.scanStage.scanErrorFlag, sizeof(int), cudaMemcpyDeviceToHost, stream);
                if (flagErr != cudaSuccess) {
                    const char* msg = cudaGetErrorString(flagErr);
                    JTRACE("CUDA", std::string("CUDA scan error flag readback failed: ") + (msg ? msg : "(unknown)"));
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                }
                flagErr = cudaStreamSynchronize(stream);
                if (flagErr != cudaSuccess) {
                    const char* msg = cudaGetErrorString(flagErr);
                    JTRACE("CUDA", std::string("CUDA stream sync failed after negative pipeline: ") + (msg ? msg : "(unknown)"));
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
                }
                if (scanError != 0) {
                    JTRACE("CUDA", "FATAL: negative pipeline scan produced non-finite RGB");
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
                }
            }

            JuicerCuda::record_use(*cudaResources, _pCudaStream);
        }
        return;
    }

    // RenderMode::Print (PrintBypass=false).
    {
        if (!_printReady || !_prt) {
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
            throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
            throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
        }

        const bool useSpatialDIR = (_dirRT.active && std::isfinite(_dirRT.spatialSigmaPixels) && _dirRT.spatialSigmaPixels > 0.0f);

        if (!_ws->printScannerValid) {
            JTRACE("CUDA", "FATAL: print scanner runtime invalid");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        if (_ws->printColorRuntime.hash == 0) {
            JTRACE("CUDA", "FATAL: print scanner color runtime invalid");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        if (!_ws->printMediumRuntime.tables || _ws->printMediumRuntime.tables->K <= 0) {
            JTRACE("CUDA", "FATAL: print scanner spectral tables unavailable");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }

        // Print exposure compensation factor is computed on CPU (no image reads; safe for CUDA renders).
        float kMidSpectral = 1.0f;
        {
            const float exposureCompScale = _printParams.exposureCompensationEnabled
                ? _printParams.exposureCompensationScale
                : 1.0f;
            kMidSpectral = Pipeline::PipelineRunner::compute_midgray_factor(
                *_ws,
                *_prt,
                _printParams,
                _dirRT,
                exposureCompScale);
            if (!std::isfinite(kMidSpectral) || !(kMidSpectral > 0.0f)) {
                kMidSpectral = 1.0f;
            }
        }

        JuicerCuda::PipelineRunParams run{};
        run.src = srcPtr;
        run.srcRowBytes = static_cast<std::size_t>(srcRowBytes);
        run.dst = dstPtr;
        run.dstRowBytes = static_cast<std::size_t>(dstRowBytes);
        run.width = width;
        run.height = height;
        run.nComponents = _nComponents;

        // Film raw conversion payload
        run.filmRaw.inputColorSpaceIndex = Spectral::inputColorSpaceToIndex(_ws->filmRaw.inputColorSpace);
        run.filmRaw.applyCctfDecoding = _ws->filmRaw.applyCctfDecoding ? 1 : 0;
        run.filmRaw.applyInputChromaticAdapt = _ws->filmRaw.applyInputChromaticAdapt ? 1 : 0;
        run.filmRaw.spectralUpsamplingMode = static_cast<int>(_ws->filmRaw.spectralUpsamplingMode);
        for (int i = 0; i < 9; ++i) {
            run.filmRaw.inputRGBToXYZ[i] = _ws->filmRaw.inputRGBToXYZ.m[i];
            run.filmRaw.inputXYZAdapt[i] = _ws->filmRaw.inputXYZAdapt.m[i];
        }
        run.filmRaw.midgrayScale = _ws->filmRaw.midgrayScale;
        for (int i = 0; i < 3; ++i) {
            run.filmRaw.refIllumWhiteXYZ[i] = _ws->filmRaw.refIllumWhiteXYZ[i];
        }

        run.filmExpose.exposureScale = _exposureScale;
        run.filmDevelop.gammaFactorB = _ws->gammaFactorB;
        run.filmDevelop.gammaFactorG = _ws->gammaFactorG;
        run.filmDevelop.gammaFactorR = _ws->gammaFactorR;
        run.filmDevelop.dirPrecorrected = _ws->dirPrecorrected ? 1 : 0;

        // DIR runtime payload.
        run.filmDevelop.dir.active = _dirRT.active ? 1 : 0;
        run.filmDevelop.dir.highShift = _dirRT.highShift;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                run.filmDevelop.dir.M[r * 3 + c] = _dirRT.M[r][c];
            }
        }
        for (int i = 0; i < 3; ++i) {
            run.filmDevelop.dir.dMax[i] = _dirRT.dMax[i];
        }

        // Print scan color payload + output encoding (print medium).
        {
            const Scanner::ColorRuntime& color = _ws->printColorRuntime;
            for (int i = 0; i < 9; ++i) {
                run.scanStage.scanColor.cat02[i] = color.cat02[i];
                run.scanStage.scanColor.xyzToRgb[i] = color.xyzToRgb[i];
            }
            for (int i = 0; i < 3; ++i) {
                run.scanStage.scanColor.illuminantXYZ[i] = color.illuminantXYZ[i];
            }

            run.scanStage.scanColor.encoding.outputColorSpaceIndex = OutputEncoding::toIndex(color.encoding.colorSpace);
            run.scanStage.scanColor.encoding.applyCctfEncoding = color.encoding.applyCctfEncoding ? 1 : 0;
            run.scanStage.scanColor.encoding.preserveLinearRange = color.encoding.preserveLinearRange ? 1 : 0;
            run.scanStage.scanColor.encoding.inputIsOutputSpace = color.encoding.inputIsOutputSpace ? 1 : 0;

            const auto& outSpace = GeneratedColorSpaces::get(color.encoding.colorSpace);
            run.scanStage.scanColor.encoding.cctf.kind = static_cast<int>(outSpace.cctf.kind);
            run.scanStage.scanColor.encoding.cctf.gamma = outSpace.cctf.gamma;
            run.scanStage.scanColor.encoding.cctf.a = outSpace.cctf.a;
            run.scanStage.scanColor.encoding.cctf.b = outSpace.cctf.b;
            run.scanStage.scanColor.encoding.cctf.c = outSpace.cctf.c;
            run.scanStage.scanColor.encoding.cctf.d = outSpace.cctf.d;
            run.scanStage.scanColor.encoding.cctf.linearCutoff = outSpace.cctf.linearCutoff;

            const OutputEncoding::Matrix3x3 dwgToOutput = OutputEncoding::dwg_to_output_matrix(color.encoding.colorSpace);
            for (int i = 0; i < 9; ++i) {
                run.scanStage.scanColor.encoding.dwgToOutput[i] = dwgToOutput.m[i];
            }
        }

        const cudaStream_t stream = _pCudaStream ? reinterpret_cast<cudaStream_t>(_pCudaStream) : nullptr;

        // Device pointers from the per-instance cache + kernel launch + record_use must be
        // serialized to avoid rebuild races before lastUseEvent is recorded.
        {
            std::lock_guard<std::mutex> lock(_instanceState->cudaMutex);
            if (!cudaResources) {
                JTRACE("CUDA", "FATAL: CUDA resources missing for print pipeline");
                throw OFX::Exception::Suite(kOfxStatErrFatal);
            }

            // Ensure the print illuminant filtered is available for current print params.
            std::string illumError;
            if (!JuicerCuda::ensure_print_illuminant_filtered(*cudaResources, *_ws, *_prt, _printParams, _pCudaStream, illumError)) {
                JTRACE("CUDA", std::string("CUDA print illuminant upload failed: ") + illumError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
            }
#if JUICER_TRACE_PRINT_SWAP
            {
                std::lock_guard<std::mutex> resLock(cudaResources->m);
                const std::uintptr_t prtPtr = reinterpret_cast<std::uintptr_t>(_prt);
                const std::uintptr_t illumPtr = reinterpret_cast<std::uintptr_t>(cudaResources->printIllumRuntimePtr);
                const std::uintptr_t preflashPtr = reinterpret_cast<std::uintptr_t>(cudaResources->printPreflashRuntimePtr);
                std::string msg = std::string("cuda print payload build=") + std::to_string(_ws ? _ws->buildCounter : 0)
                    + " printRT=" + std::to_string(prtPtr)
                    + " neutralY/M/C=" + std::to_string(_prt ? _prt->neutralY : 0.0f)
                    + "/" + std::to_string(_prt ? _prt->neutralM : 0.0f)
                    + "/" + std::to_string(_prt ? _prt->neutralC : 0.0f)
                    + " yFilter=" + std::to_string(_printParams.yFilter)
                    + " mFilter=" + std::to_string(_printParams.mFilter)
                    + " illumPtr=" + std::to_string(illumPtr)
                    + " illumBuild=" + std::to_string(cudaResources->printIllumBuildCounter)
                    + " illumY/M/Csteps=" + std::to_string(cudaResources->printIllumYShiftSteps)
                    + "/" + std::to_string(cudaResources->printIllumMShiftSteps)
                    + "/" + std::to_string(cudaResources->printIllumCShiftSteps)
                    + " preflashValid=" + std::to_string(cudaResources->printPreflashValid ? 1 : 0)
                    + " preflashPtr=" + std::to_string(preflashPtr)
                    + " preflashBuild=" + std::to_string(cudaResources->printPreflashBuildCounter);
                JTRACE("PRINTDBG", msg);
            }
#endif

            // Film density curves + sensitivities + SPD reconstruction tables.
            run.filmDevelop.densB = { cudaResources->densB.x, cudaResources->densB.y, cudaResources->densB.n };
            run.filmDevelop.densG = { cudaResources->densG.x, cudaResources->densG.y, cudaResources->densG.n };
            run.filmDevelop.densR = { cudaResources->densR.x, cudaResources->densR.y, cudaResources->densR.n };
            run.filmDevelop.dirDensB = { cudaResources->dirDensB.x, cudaResources->dirDensB.y, cudaResources->dirDensB.n };
            run.filmDevelop.dirDensG = { cudaResources->dirDensG.x, cudaResources->dirDensG.y, cudaResources->dirDensG.n };
            run.filmDevelop.dirDensR = { cudaResources->dirDensR.x, cudaResources->dirDensR.y, cudaResources->dirDensR.n };
            run.filmExpose.sensB = { cudaResources->sensB.x, cudaResources->sensB.y, cudaResources->sensB.n };
            run.filmExpose.sensG = { cudaResources->sensG.x, cudaResources->sensG.y, cudaResources->sensG.n };
            run.filmExpose.sensR = { cudaResources->sensR.x, cudaResources->sensR.y, cudaResources->sensR.n };

            run.filmExpose.tablesAx = cudaResources->tablesAx;
            run.filmExpose.tablesAy = cudaResources->tablesAy;
            run.filmExpose.tablesAz = cudaResources->tablesAz;
            run.filmExpose.tablesK = cudaResources->tablesK;
            for (int i = 0; i < 9; ++i) {
                run.filmExpose.spdSInv[i] = cudaResources->spdSInv[i];
            }

            run.filmExpose.hanatosLut = cudaResources->hanatosLut;
            run.filmExpose.hanatosN = cudaResources->hanatosN;
            run.filmExpose.hanatosLutIntegrated = cudaResources->hanatosLutIntegrated;
            run.filmExpose.hanatosNIntegrated = cudaResources->hanatosNIntegrated;

            // Scan LUT selection (print medium).
            run.scanStage.scannerUseLut = _scannerSettings.useLut ? 1 : 0;
            run.scanStage.scanLutLogXYZ = nullptr;
            run.scanStage.scanLutRes = 0;
            if (run.scanStage.scannerUseLut) {
                std::string lutError;
                if (!JuicerCuda::ensure_scan_lut(*cudaResources, *_ws, false, _pCudaStream, lutError)) {
                    JTRACE("CUDA", std::string("CUDA print scan LUT upload failed: ") + lutError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                }
                run.scanStage.scanLutLogXYZ = cudaResources->scanPrintLut.logXYZ;
                run.scanStage.scanLutRes = static_cast<int>(cudaResources->scanPrintLut.res);
                if (!run.scanStage.scanLutLogXYZ || run.scanStage.scanLutRes <= 0) {
                    JTRACE("CUDA", "FATAL: print scan LUT missing after successful upload");
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
                }
            }

            // Print scan tables payload (for the scan stage).
            run.scanStage.scanTables.epsC = cudaResources->scanPrint.tables.epsC;
            run.scanStage.scanTables.epsM = cudaResources->scanPrint.tables.epsM;
            run.scanStage.scanTables.epsY = cudaResources->scanPrint.tables.epsY;
            run.scanStage.scanTables.Ax = cudaResources->scanPrint.tables.Ax;
            run.scanStage.scanTables.Ay = cudaResources->scanPrint.tables.Ay;
            run.scanStage.scanTables.Az = cudaResources->scanPrint.tables.Az;
            run.scanStage.scanTables.baseMin = cudaResources->scanPrint.tables.baseMin;
            run.scanStage.scanTables.K = cudaResources->scanPrint.tables.K;
            run.scanStage.scanTables.hasBaseline = cudaResources->scanPrint.tables.hasBaseline;
            run.scanStage.scanTables.invYn = cudaResources->scanPrint.tables.invYn;
            run.scanStage.scanTables.mediumIsNegative = cudaResources->scanPrint.mediumIsNegative;
            for (int i = 0; i < 3; ++i) {
                run.scanStage.scanTables.min_cmy[i] = cudaResources->scanPrint.min_cmy[i];
                run.scanStage.scanTables.inv_max_cmy[i] = cudaResources->scanPrint.inv_max_cmy[i];
            }

            std::string scanFlagError;
            if (!JuicerCuda::ensure_scan_error_flag(*cudaResources, _pCudaStream, scanFlagError)) {
                JTRACE("CUDA", std::string("CUDA scan error flag allocation failed: ") + scanFlagError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
            }
            run.scanStage.scanErrorFlag = cudaResources->scanErrorFlag;
            if (!run.scanStage.scanErrorFlag) {
                JTRACE("CUDA", "FATAL: scan error flag missing after allocation");
                throw OFX::Exception::Suite(kOfxStatErrFatal);
            }

            cudaEvent_t scanEvent = cudaResources->scanErrorEventOpaque
                ? reinterpret_cast<cudaEvent_t>(cudaResources->scanErrorEventOpaque)
                : nullptr;
            if (cudaResources->scanErrorPending && scanEvent && cudaResources->scanErrorHost) {
                cudaError_t pollErr = cudaEventQuery(scanEvent);
                if (pollErr == cudaErrorNotReady) {
                    pollErr = cudaEventSynchronize(scanEvent);
                }
                if (pollErr != cudaSuccess) {
                    const char* msg = cudaGetErrorString(pollErr);
                    JTRACE("CUDA", std::string("CUDA scan error event sync failed: ") + (msg ? msg : "(unknown)"));
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
                }
                cudaResources->scanErrorPending = 0;
                if (*cudaResources->scanErrorHost != 0) {
                    JTRACE("CUDA", "FATAL: previous scan produced non-finite RGB");
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
                }
            }

            cudaError_t flagErr = cudaMemsetAsync(run.scanStage.scanErrorFlag, 0, sizeof(int), stream);
            if (flagErr != cudaSuccess) {
                const char* msg = cudaGetErrorString(flagErr);
                JTRACE("CUDA", std::string("CUDA scan error flag memset failed: ") + (msg ? msg : "(unknown)"));
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
            }

            run.filmDevelop.spatialDir.active = useSpatialDIR ? 1 : 0;
            run.filmDevelop.spatialDir.corrY = nullptr;
            run.filmDevelop.spatialDir.corrM = nullptr;
            run.filmDevelop.spatialDir.corrC = nullptr;
            if (useSpatialDIR) {
                std::string dirError;
                if (!JuicerCuda::ensure_spatial_dir_scratch(*cudaResources, width, height, _pCudaStream, dirError)) {
                    JTRACE("CUDA", std::string("CUDA spatial DIR scratch allocation failed: ") + dirError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                }
                if (!JuicerCuda::ensure_spatial_dir_kernel(*cudaResources, cudaResources->spatialDirKernel, _dirRT.spatialSigmaPixels, _pCudaStream, dirError)) {
                    JTRACE("CUDA", std::string("CUDA spatial DIR kernel upload failed: ") + dirError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                }

                run.filmDevelop.spatialDir.corrY = cudaResources->spatialDirScratch.corrY;
                run.filmDevelop.spatialDir.corrM = cudaResources->spatialDirScratch.corrM;
                run.filmDevelop.spatialDir.corrC = cudaResources->spatialDirScratch.corrC;

                cudaError_t dirErr = juicer_cuda_build_spatial_dir(
                    &run,
                    cudaResources->spatialDirScratch.corrY,
                    cudaResources->spatialDirScratch.corrM,
                    cudaResources->spatialDirScratch.corrC,
                    cudaResources->spatialDirScratch.tmp,
                    cudaResources->spatialDirKernel.weights,
                    cudaResources->spatialDirKernel.radius,
                    _pCudaStream);
                if (dirErr != cudaSuccess) {
                    const char* msg = cudaGetErrorString(dirErr);
                    JTRACE("CUDA", std::string("FATAL: spatial DIR build failed: ") + (msg ? msg : "(unknown)"));
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
                }
            }

            // Print pipeline payloads.
            run.printExpose.active = 1;
            run.printExpose.negTables.epsC = cudaResources->scanNegative.tables.epsC;
            run.printExpose.negTables.epsM = cudaResources->scanNegative.tables.epsM;
            run.printExpose.negTables.epsY = cudaResources->scanNegative.tables.epsY;
            run.printExpose.negTables.Ax = cudaResources->scanNegative.tables.Ax;
            run.printExpose.negTables.Ay = cudaResources->scanNegative.tables.Ay;
            run.printExpose.negTables.Az = cudaResources->scanNegative.tables.Az;
            run.printExpose.negTables.baseMin = cudaResources->scanNegative.tables.baseMin;
            run.printExpose.negTables.K = cudaResources->scanNegative.tables.K;
            run.printExpose.negTables.hasBaseline = cudaResources->scanNegative.tables.hasBaseline;
            run.printExpose.negTables.invYn = cudaResources->scanNegative.tables.invYn;
            run.printExpose.negTables.mediumIsNegative = cudaResources->scanNegative.mediumIsNegative;
            for (int i = 0; i < 3; ++i) {
                run.printExpose.negTables.min_cmy[i] = cudaResources->scanNegative.min_cmy[i];
                run.printExpose.negTables.inv_max_cmy[i] = cudaResources->scanNegative.inv_max_cmy[i];
            }

            run.printExpose.printIllumFiltered = cudaResources->printIllumFiltered;
            run.printExpose.printIllumK = cudaResources->printIllumK;
            run.printExpose.printSensC = { cudaResources->printSensC.x, cudaResources->printSensC.y, cudaResources->printSensC.n };
            run.printExpose.printSensM = { cudaResources->printSensM.x, cudaResources->printSensM.y, cudaResources->printSensM.n };
            run.printExpose.printSensY = { cudaResources->printSensY.x, cudaResources->printSensY.y, cudaResources->printSensY.n };
            run.printDevelop.printDcC = { cudaResources->printDcC.x, cudaResources->printDcC.y, cudaResources->printDcC.n };
            run.printDevelop.printDcM = { cudaResources->printDcM.x, cudaResources->printDcM.y, cudaResources->printDcM.n };
            run.printDevelop.printDcY = { cudaResources->printDcY.x, cudaResources->printDcY.y, cudaResources->printDcY.n };
            run.printDevelop.printGammaC = cudaResources->printGammaC;
            run.printDevelop.printGammaM = cudaResources->printGammaM;
            run.printDevelop.printGammaY = cudaResources->printGammaY;
            run.printExpose.printExposure = _printParams.exposure;
            run.printExpose.printPreflashExposure = _printParams.preflashExposure;
            run.printExpose.printMidgrayFactor = kMidSpectral;
            for (int i = 0; i < 3; ++i) {
                run.printExpose.printPreflashRaw[i] = cudaResources->printPreflashRaw[i];
            }

            if (!run.printExpose.printIllumFiltered || run.printExpose.printIllumK <= 0 ||
                !run.printExpose.printSensC.y || !run.printExpose.printSensM.y || !run.printExpose.printSensY.y ||
                !run.printDevelop.printDcC.y || !run.printDevelop.printDcM.y || !run.printDevelop.printDcY.y) {
                JTRACE("CUDA", "CUDA print payloads missing; cannot render print pipeline");
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
            }

            // Scanner optics/glare for the print medium.
            Scanner::ScannerMediumRuntime printMedium = _ws->printMediumRuntime;
            if (_hasPrintGlareOverride) {
                printMedium.glare.active = _printGlareOverride.active;
                printMedium.glare.percent = _printGlareOverride.percent;
                printMedium.glare.roughness = _printGlareOverride.roughness;
                printMedium.glare.blur = _printGlareOverride.blur;
                printMedium.glare.compensationRemovalFactor = 0.0f;
                printMedium.glare.compensationRemovalDensity = 0.0f;
                printMedium.glare.compensationRemovalTransition = 0.0f;
                const std::uint64_t glareHash = Scanner::hash_glare(printMedium.glare);
                if (glareHash == 0) {
                    JTRACE("HASH", "FATAL: failed to hash print glare override parameters");
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
                }
                printMedium.staticKey.glareHash = glareHash;
            }

            const bool wantGlare = printMedium.glare.active && (printMedium.glare.percent > 0.0f);
            float glarePercent = 0.0f;
            float glareRoughness = 0.0f;
            float glareBlurSigmaPx = 0.0f;
            std::uint64_t glareSeed = 0;
            if (wantGlare) {
                glarePercent = printMedium.glare.percent;
                glareRoughness = printMedium.glare.roughness;
                glareBlurSigmaPx = printMedium.glare.blur;

                const std::uint64_t buildCounter = _ws->buildCounter;
                const std::uint64_t seedBaseFields[3] = {
                    static_cast<std::uint64_t>(_clipToken),
                    _frameTimeHash,
                    buildCounter
                };
                std::uint64_t seedBase = Hash::hash_bytes(seedBaseFields, sizeof(seedBaseFields));
                if (seedBase == 0) {
                    seedBase = 1;
                }

                const std::uint64_t glareFields[4] = {
                    seedBase,
                    static_cast<std::uint64_t>(_frameBoundsVersion),
                    printMedium.staticKey.glareHash,
                    static_cast<std::uint64_t>(Scanner::ScannerMedium::Print)
                };
                glareSeed = Hash::hash_bytes(glareFields, sizeof(glareFields));
            }

            const float lensBlurSigmaPx = _scannerOptions.lensBlurSigmaPx;
            const float unsharpSigmaPx = _scannerOptions.unsharpSigmaPx;
            const float unsharpAmount = _scannerOptions.unsharpAmount;

            const bool wantLensBlur = std::isfinite(lensBlurSigmaPx) && lensBlurSigmaPx > 0.0f;
            const bool wantUnsharp = std::isfinite(unsharpSigmaPx) && unsharpSigmaPx > 0.0f &&
                std::isfinite(unsharpAmount) && unsharpAmount != 0.0f;
            const bool wantOptics = wantLensBlur || wantUnsharp || wantGlare;

            cudaError_t err = cudaSuccess;
            if (!wantOptics) {
                err = juicer_cuda_print_pipeline(&run, _pCudaStream);
            }
            else {
                std::string opticsError;
                if (!JuicerCuda::ensure_optics_scratch(*cudaResources, width, height, wantUnsharp, _pCudaStream, opticsError)) {
                    JTRACE("CUDA", std::string("CUDA optics scratch allocation failed: ") + opticsError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                }
                if (!JuicerCuda::ensure_gaussian_kernel(*cudaResources, cudaResources->scannerLensBlurKernel, lensBlurSigmaPx, _pCudaStream, opticsError)) {
                    JTRACE("CUDA", std::string("CUDA lens blur kernel upload failed: ") + opticsError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                }
                if (!JuicerCuda::ensure_gaussian_kernel(*cudaResources, cudaResources->scannerUnsharpKernel, unsharpSigmaPx, _pCudaStream, opticsError)) {
                    JTRACE("CUDA", std::string("CUDA unsharp kernel upload failed: ") + opticsError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                }
                if (!JuicerCuda::ensure_gaussian_kernel(*cudaResources, cudaResources->scannerGlareKernel, wantGlare ? glareBlurSigmaPx : 0.0f, _pCudaStream, opticsError)) {
                    JTRACE("CUDA", std::string("CUDA glare kernel upload failed: ") + opticsError);
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                }

                err = juicer_cuda_print_pipeline_optics(
                    &run,
                    cudaResources->scannerScratch.rgbR,
                    cudaResources->scannerScratch.rgbG,
                    cudaResources->scannerScratch.rgbB,
                    cudaResources->scannerScratch.tmp,
                    cudaResources->scannerScratch.blurred,
                    cudaResources->scannerLensBlurKernel.weights,
                    cudaResources->scannerLensBlurKernel.radius,
                    cudaResources->scannerUnsharpKernel.weights,
                    cudaResources->scannerUnsharpKernel.radius,
                    unsharpAmount,
                    win.x1,
                    win.y1,
                    glareSeed,
                    glarePercent,
                    glareRoughness,
                    cudaResources->scannerGlareKernel.weights,
                    cudaResources->scannerGlareKernel.radius,
                    _pCudaStream);
            }
            if (err != cudaSuccess) {
                const char* msg = cudaGetErrorString(err);
                JTRACE("CUDA", std::string("FATAL: print pipeline kernel launch failed: ") + (msg ? msg : "(unknown)"));
                throw OFX::Exception::Suite(kOfxStatErrFatal);
            }

            if (cudaResources->scanErrorHost && scanEvent) {
                flagErr = cudaMemcpyAsync(cudaResources->scanErrorHost, run.scanStage.scanErrorFlag, sizeof(int), cudaMemcpyDeviceToHost, stream);
                if (flagErr != cudaSuccess) {
                    const char* msg = cudaGetErrorString(flagErr);
                    JTRACE("CUDA", std::string("CUDA scan error flag readback failed: ") + (msg ? msg : "(unknown)"));
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                }
                cudaError_t evErr = cudaEventRecord(scanEvent, stream);
                if (evErr != cudaSuccess) {
                    const char* msg = cudaGetErrorString(evErr);
                    JTRACE("CUDA", std::string("CUDA scan error event record failed: ") + (msg ? msg : "(unknown)"));
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
                }
                cudaResources->scanErrorPending = 1;
                cudaError_t pollErr = cudaEventQuery(scanEvent);
                if (pollErr == cudaSuccess) {
                    cudaResources->scanErrorPending = 0;
                    if (*cudaResources->scanErrorHost != 0) {
                        JTRACE("CUDA", "FATAL: print pipeline scan produced non-finite RGB");
                        throw OFX::Exception::Suite(kOfxStatErrFatal);
                    }
                }
                else if (pollErr != cudaErrorNotReady) {
                    const char* msg = cudaGetErrorString(pollErr);
                    JTRACE("CUDA", std::string("CUDA scan error event query failed: ") + (msg ? msg : "(unknown)"));
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
                }
            }
            else {
                int scanError = 0;
                flagErr = cudaMemcpyAsync(&scanError, run.scanStage.scanErrorFlag, sizeof(int), cudaMemcpyDeviceToHost, stream);
                if (flagErr != cudaSuccess) {
                    const char* msg = cudaGetErrorString(flagErr);
                    JTRACE("CUDA", std::string("CUDA scan error flag readback failed: ") + (msg ? msg : "(unknown)"));
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                }
                flagErr = cudaStreamSynchronize(stream);
                if (flagErr != cudaSuccess) {
                    const char* msg = cudaGetErrorString(flagErr);
                    JTRACE("CUDA", std::string("CUDA stream sync failed after print pipeline: ") + (msg ? msg : "(unknown)"));
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
                }
                if (scanError != 0) {
                    JTRACE("CUDA", "FATAL: print pipeline scan produced non-finite RGB");
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
                }
            }

            JuicerCuda::record_use(*cudaResources, _pCudaStream);
        }
        return;
    }
#endif
}
