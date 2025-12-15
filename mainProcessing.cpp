// JuicerProcessing.cpp

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <string>
#include <atomic>
#include <thread>
#include <sstream>
#include <mutex>

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
#include "ScannerOptics.h"
#include "Couplers.h"
#include "mainProcessing.h"
#include "ExposeFilmStage.h"
#include "DevelopFilmStage.h"
#include "ExposePrintStage.h"
#include "DevelopPrintStage.h"

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

    void blurChannelSeparable(const std::vector<float>& src, std::vector<float>& tmp, std::vector<float>& dst,
        int width, int height, const std::vector<float>& k) {
        tmp.assign(size_t(width * height), 0.0f);
        const int radius = int(k.size() / 2);
        const auto reflectIndex = [](int idx, int size) -> int {
            if (size <= 1) {
                return 0;
            }
            while (idx < 0 || idx >= size) {
                if (idx < 0) {
                    idx = -idx;
                }
                else {
                    idx = 2 * size - idx - 2;
                }
            }
            return idx;
            };
        for (int y = 0; y < height; ++y) {
            const float* srow = &src[size_t(y * width)];
            float* trow = &tmp[size_t(y * width)];
            for (int x = 0; x < width; ++x) {
                float acc = 0.0f;
                for (int j = -radius; j <= radius; ++j) {
                    const int xx = reflectIndex(x + j, width);
                    acc += srow[xx] * k[size_t(j + radius)];
                }
                trow[x] = acc;
            }
        }
        dst.assign(size_t(width * height), 0.0f);
        for (int x = 0; x < width; ++x) {
            for (int y = 0; y < height; ++y) {
                float acc = 0.0f;
                for (int j = -radius; j <= radius; ++j) {
                    const int yy = reflectIndex(y + j, height);
                    acc += tmp[size_t(yy * width + x)] * k[size_t(j + radius)];
                }
                dst[size_t(y * width + x)] = acc;
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


namespace {

    template <typename FetchRGB, typename AbortCheck>
	    void buildSpatialDIRCorrections(
	        int width, int height,
	        const WorkingState& ws,
	        const Couplers::Runtime& dirRT,
	        float exposureScale,
	        FetchRGB&& fetchRGB,
	        AbortCheck&& abortCheck,
	        JuicerProc::SpatialDIRWorkspace& work,
	        std::vector<float>& kernelCache)
	    {
	        const size_t total = size_t(width) * size_t(height);
	        work.filmRaw_B.assign(total, 0.0f);
	        work.filmRaw_G.assign(total, 0.0f);
	        work.filmRaw_R.assign(total, 0.0f);
	        work.corrY.assign(total, 0.0f);
	        work.corrM.assign(total, 0.0f);
	        work.corrC.assign(total, 0.0f);
	        work.corrYBlur.assign(total, 0.0f);
	        work.corrMBlur.assign(total, 0.0f);
	        work.corrCBlur.assign(total, 0.0f);
	        work.tmp.assign(total, 0.0f);

	        if (!dirRT.active) {
	            return;
	        }


        // Per agx-emulsion parity: use same sensitivities everywhere (no separate "before balance" state).
        // Profiles contain pre-balanced sensitivities; spatial DIR and mid-gray must match pixel render.
        const Spectral::Curve& sensB_forExposure = ws.sensB;
        const Spectral::Curve& sensG_forExposure = ws.sensG;
        const Spectral::Curve& sensR_forExposure = ws.sensR;

        // DEBUG: Log WorkingState sensitivity curves before use
        {
            const int idx_450 = 14, idx_520 = 28, idx_650 = 54;
            std::ostringstream oss;
            oss << "WS_SENS_PREUSE: B[450nm]=" << (sensB_forExposure.linear.size() > idx_450 ? sensB_forExposure.linear[idx_450] : -999.0f)
                << " G[450nm]=" << (sensG_forExposure.linear.size() > idx_450 ? sensG_forExposure.linear[idx_450] : -999.0f)
                << " R[450nm]=" << (sensR_forExposure.linear.size() > idx_450 ? sensR_forExposure.linear[idx_450] : -999.0f)
                << " | B[520nm]=" << (sensB_forExposure.linear.size() > idx_520 ? sensB_forExposure.linear[idx_520] : -999.0f)
                << " G[520nm]=" << (sensG_forExposure.linear.size() > idx_520 ? sensG_forExposure.linear[idx_520] : -999.0f)
                << " R[520nm]=" << (sensR_forExposure.linear.size() > idx_520 ? sensR_forExposure.linear[idx_520] : -999.0f)
                << " | B[650nm]=" << (sensB_forExposure.linear.size() > idx_650 ? sensB_forExposure.linear[idx_650] : -999.0f)
                << " G[650nm]=" << (sensG_forExposure.linear.size() > idx_650 ? sensG_forExposure.linear[idx_650] : -999.0f)
                << " R[650nm]=" << (sensR_forExposure.linear.size() > idx_650 ? sensR_forExposure.linear[idx_650] : -999.0f);
            JTRACE("SPECTRAL", oss.str());
        }

        // Pass A: sample exposures, convert to logE, compute DIR corrections per pixel.
        const int center_xx = width / 2;
        const int center_yy = height / 2;

	        for (int yy = 0; yy < height; ++yy) {
	            if (abortCheck()) break;
	            for (int xx = 0; xx < width; ++xx) {
	                const size_t idx = size_t(yy) * size_t(width) + size_t(xx);
	                float rgbIn[3] = { 0.0f, 0.0f, 0.0f };
	                if (!fetchRGB(xx, yy, rgbIn)) {
	                    work.filmRaw_B[idx] = work.filmRaw_G[idx] = work.filmRaw_R[idx] = 0.0f;
	                    work.corrY[idx] = work.corrM[idx] = work.corrC[idx] = 0.0f;
	                    continue;
	                }

                // SPD DEBUG: Log input RGB for center pixel
                if (xx == center_xx && yy == center_yy) {
                    std::ostringstream oss;
                    oss << "INPUT_RGB tile_pixel(" << xx << "," << yy << "): R=" << rgbIn[0] << " G=" << rgbIn[1] << " B=" << rgbIn[2];
                    JTRACE("SPECTRAL", oss.str());
                }

	                Pipeline::ExposeFilmInputs exposeIn{};
	                exposeIn.rgb.v[0] = rgbIn[0];
	                exposeIn.rgb.v[1] = rgbIn[1];
	                exposeIn.rgb.v[2] = rgbIn[2];
	                exposeIn.exposureScale = exposureScale;

	                Pipeline::ExposeFilmOutputs exposeOut{};
	                if (!Pipeline::ExposeFilmStage::run(ws, exposeIn, exposeOut)) {
	                    work.filmRaw_B[idx] = work.filmRaw_G[idx] = work.filmRaw_R[idx] = 0.0f;
	                    work.corrY[idx] = work.corrM[idx] = work.corrC[idx] = 0.0f;
	                    continue;
	                }

	                work.filmRaw_B[idx] = exposeOut.filmRaw.v[0];
	                work.filmRaw_G[idx] = exposeOut.filmRaw.v[1];
	                work.filmRaw_R[idx] = exposeOut.filmRaw.v[2];

	                // SPD DEBUG: Log film raw exposure (pre-log) for center pixel
	                if (xx == center_xx && yy == center_yy) {
	                    std::ostringstream oss;
	                    oss << "FILM_RAW tile_pixel(" << xx << "," << yy << "): B=" << exposeOut.filmRaw.v[0]
	                        << " G=" << exposeOut.filmRaw.v[1]
	                        << " R=" << exposeOut.filmRaw.v[2];
	                    JTRACE("SPECTRAL", oss.str());

                    // Log sensitivity curve values at key wavelengths
                    const int idx_450 = 14;  // (450-380)/5 = 14
                    const int idx_520 = 28;  // (520-380)/5 = 28
                    const int idx_650 = 54;  // (650-380)/5 = 54
                    if (!sensB_forExposure.linear.empty() && !sensG_forExposure.linear.empty() && !sensR_forExposure.linear.empty()) {
                        std::ostringstream oss1, oss2, oss3;
                        oss1 << "SENS_CURVES tile_pixel(" << xx << "," << yy << "): B[450nm]=" << sensB_forExposure.linear[idx_450]
                            << " G[450nm]=" << sensG_forExposure.linear[idx_450] << " R[450nm]=" << sensR_forExposure.linear[idx_450];
                        oss2 << "SENS_CURVES tile_pixel(" << xx << "," << yy << "): B[520nm]=" << sensB_forExposure.linear[idx_520]
                            << " G[520nm]=" << sensG_forExposure.linear[idx_520] << " R[520nm]=" << sensR_forExposure.linear[idx_520];
                        oss3 << "SENS_CURVES tile_pixel(" << xx << "," << yy << "): B[650nm]=" << sensB_forExposure.linear[idx_650]
                            << " G[650nm]=" << sensG_forExposure.linear[idx_650] << " R[650nm]=" << sensR_forExposure.linear[idx_650];
                        JTRACE("SPECTRAL", oss1.str());
                        JTRACE("SPECTRAL", oss2.str());
                        JTRACE("SPECTRAL", oss3.str());
	                    }
	                }

	                Pipeline::DevelopFilmInputs devIn{};
	                devIn.filmRaw = exposeOut.filmRaw;
	                devIn.dirRuntime = &dirRT;
	                devIn.applyDirRuntime = false; // Pass A wants pre-DIR densities for correction computation.
	                Pipeline::DevelopFilmOutputs devOut{};
	                if (!Pipeline::DevelopFilmStage::run(ws, devIn, devOut)) {
	                    work.corrY[idx] = work.corrM[idx] = work.corrC[idx] = 0.0f;
	                    continue;
	                }

	                const float leB = devOut.filmLogRaw.v[0];
	                const float leG = devOut.filmLogRaw.v[1];
	                const float leR = devOut.filmLogRaw.v[2];

	                // Convert CMY -> YMC to match Couplers::ApplyInputLogE contract.
	                const float D_Y = devOut.negativeDensity.v[2];
	                const float D_M = devOut.negativeDensity.v[1];
	                const float D_C = devOut.negativeDensity.v[0];

	                float aCorr[3];
	                Couplers::ApplyInputLogE io{ { leB, leG, leR }, { D_Y, D_M, D_C } };
	                Couplers::compute_logE_corrections(io, dirRT, aCorr);
	                for (float& v : aCorr) {
                    if (!std::isfinite(v)) v = 0.0f;
                }
                work.corrY[idx] = aCorr[0];
                work.corrM[idx] = aCorr[1];
                work.corrC[idx] = aCorr[2];
            }
        }

        // Blur corrections spatially (shared between preview + render path)
        kernelCache.clear();
        JuicerProc::buildGaussianKernel(dirRT.spatialSigmaPixels, kernelCache);
        JuicerProc::blurChannelSeparable(work.corrY, work.tmp, work.corrYBlur, width, height, kernelCache);
        JuicerProc::blurChannelSeparable(work.corrM, work.tmp, work.corrMBlur, width, height, kernelCache);
        JuicerProc::blurChannelSeparable(work.corrC, work.tmp, work.corrCBlur, width, height, kernelCache);

        auto scrubClamp = [](std::vector<float>& v) {
            for (float& t : v) {
                if (!std::isfinite(t)) t = 0.0f;
                if (t < -10.0f) t = -10.0f;
                if (t > 10.0f) t = 10.0f;
            }
            };
        scrubClamp(work.corrYBlur);
        scrubClamp(work.corrMBlur);
        scrubClamp(work.corrCBlur);
    }
}

// JuicerProcessor method definitions matching JuicerProcessing.h

JuicerProcessor::JuicerProcessor(OFX::ImageEffect& effect)
    : OFX::ImageProcessor(effect)
    , _srcImg(nullptr)
    , _nComponents(0)
    , _scannerOptions{}
    , _scannerSettings{}
    , _printParams{}
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
        ctx.kMidSpectral = Pipeline::ExposePrintStage::compute_midgray_factor(
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

void JuicerProcessor::writeNegativeDensities(const RenderContext& ctx, unsigned int threadCount) {
    if (!_ws || !_wsReady || ctx.width <= 0 || ctx.height <= 0) {
        return;
    }

    _density.medium = Scanner::ScannerMedium::Negative;

	    if (ctx.useSpatialDIR) {
        auto fetchRGB = [&](int xx, int yy, float rgb[3])->bool {
            const int x = ctx.window.x1 + xx;
            const int y = ctx.window.y1 + yy;
            const float* srcPix = reinterpret_cast<const float*>(_srcImg->getPixelAddress(x, y));
            if (!srcPix) return false;
            rgb[0] = srcPix[0];
            rgb[1] = srcPix[1];
            rgb[2] = srcPix[2];
            return true;
            };

        auto abortCheck = [&]() -> bool { return _effect.abort(); };

        buildSpatialDIRCorrections(
            ctx.width,
            ctx.height,
            *_ws,
            _dirRT,
            ctx.exposureScaleSafe,
            fetchRGB,
            abortCheck,
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
    const int rowsPerThread = (height + int(nThreads) - 1) / int(nThreads);
    std::vector<std::thread> threads;
    threads.reserve(nThreads);

    for (unsigned int t = 0; t < nThreads; ++t) {
        const int yStart = rowsPerThread * int(t);
        const int yEnd = std::min(height, rowsPerThread * int(t + 1));
        threads.emplace_back([&, yStart, yEnd, t]() {
            for (int yOff = yStart; yOff < yEnd && !abortFlag.load(std::memory_order_relaxed); ++yOff) {
                if (_effect.abort()) {
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
	                    if (ctx.useSpatialDIR) {
	                        Pipeline::DevelopFilmInputs devIn{};
	                        devIn.filmRaw.v[0] = _scratch.dirWorkspace.filmRaw_B[idx];
	                        devIn.filmRaw.v[1] = _scratch.dirWorkspace.filmRaw_G[idx];
	                        devIn.filmRaw.v[2] = _scratch.dirWorkspace.filmRaw_R[idx];
	                        devIn.dirRuntime = &_dirRT;
	                        devIn.useSpatialDIR = true;
	                        devIn.spatialLogECorrectionsYMC[0] = _scratch.dirWorkspace.corrYBlur[idx];
	                        devIn.spatialLogECorrectionsYMC[1] = _scratch.dirWorkspace.corrMBlur[idx];
	                        devIn.spatialLogECorrectionsYMC[2] = _scratch.dirWorkspace.corrCBlur[idx];

	                        Pipeline::DevelopFilmOutputs devOut{};
	                        if (!Pipeline::DevelopFilmStage::run(*_ws, devIn, devOut)) {
	                            failure.store(true, std::memory_order_relaxed);
	                            abortFlag.store(true, std::memory_order_relaxed);
	                            break;
	                        }

	                        _density.c[idx] = devOut.negativeDensity.v[0]; // C
	                        _density.m[idx] = devOut.negativeDensity.v[1]; // M
	                        _density.y[idx] = devOut.negativeDensity.v[2]; // Y
	                    }
	                    else {
	                        const float* srcPix = reinterpret_cast<const float*>(_srcImg->getPixelAddress(x, y));
	                        if (!srcPix) {
	                            _density.c[idx] = 0.0f;
                            _density.m[idx] = 0.0f;
                            _density.y[idx] = 0.0f;
                            continue;
                        }
	                        Pipeline::ExposeFilmInputs exposeIn{};
	                        exposeIn.rgb.v[0] = srcPix[0];
	                        exposeIn.rgb.v[1] = srcPix[1];
	                        exposeIn.rgb.v[2] = srcPix[2];
	                        exposeIn.exposureScale = ctx.exposureScaleSafe;

	                        Pipeline::ExposeFilmOutputs exposeOut{};
	                        if (!Pipeline::ExposeFilmStage::run(*_ws, exposeIn, exposeOut)) {
	                            _density.c[idx] = 0.0f;
	                            _density.m[idx] = 0.0f;
	                            _density.y[idx] = 0.0f;
	                            continue;
	                        }

	                        Pipeline::DevelopFilmInputs devIn{};
	                        devIn.filmRaw = exposeOut.filmRaw;
	                        devIn.dirRuntime = &_dirRT;
	                        devIn.applyDirRuntime = true;

	                        Pipeline::DevelopFilmOutputs devOut{};
	                        if (!Pipeline::DevelopFilmStage::run(*_ws, devIn, devOut)) {
	                            failure.store(true, std::memory_order_relaxed);
	                            abortFlag.store(true, std::memory_order_relaxed);
	                            break;
	                        }

	                        _density.c[idx] = devOut.negativeDensity.v[0]; // C
	                        _density.m[idx] = devOut.negativeDensity.v[1]; // M
	                        _density.y[idx] = devOut.negativeDensity.v[2]; // Y
	                    }
	                }
	            }
	            });
	    }

    for (std::thread& th : threads) {
        if (th.joinable()) th.join();
    }

    if (failure.load(std::memory_order_relaxed)) {
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
    if (abortFlag.load(std::memory_order_relaxed)) {
        return;
    }
}

void JuicerProcessor::convertNegativeToPrint(const RenderContext& ctx, unsigned int threadCount) {
    if (!ctx.printActive || !_ws || !_prt) {
        return;
    }

    const unsigned int nThreads = std::max(1u, threadCount);
    _scratch.printScratchPerWorker.resize(nThreads);
    std::atomic<bool> abortFlag{ false };
    std::atomic<bool> failure{ false };

    const int width = ctx.width;
    const int height = ctx.height;
    const int originX = ctx.window.x1;
    const int originY = ctx.window.y1;
    const int rowsPerThread = (height + int(nThreads) - 1) / int(nThreads);

    std::vector<std::thread> threads;
    threads.reserve(nThreads);

    for (unsigned int t = 0; t < nThreads; ++t) {
        const int yStart = rowsPerThread * int(t);
        const int yEnd = std::min(height, rowsPerThread * int(t + 1));
        threads.emplace_back([&, yStart, yEnd, t]() {
            JuicerProc::PrintPipelineScratch& scratch = _scratch.printScratchPerWorker[t];
            for (int yOff = yStart; yOff < yEnd && !abortFlag.load(std::memory_order_relaxed); ++yOff) {
                if (_effect.abort()) {
                    abortFlag.store(true, std::memory_order_relaxed);
                    break;
                }
                const size_t rowOffset = size_t(yOff) * size_t(width);
                for (int xOff = 0; xOff < width; ++xOff) {
                    if (abortFlag.load(std::memory_order_relaxed)) {
                        break;
                    }
                    const size_t idx = rowOffset + size_t(xOff);

                    Pipeline::ExposePrintInputs exposeIn{};
                    exposeIn.printRuntime = _prt;
                    exposeIn.printParams = &_printParams;
                    exposeIn.midgrayFactor = ctx.kMidSpectral;
                    exposeIn.negativeDensity.v[0] = _density.c[idx];
                    exposeIn.negativeDensity.v[1] = _density.m[idx];
                    exposeIn.negativeDensity.v[2] = _density.y[idx];

                    Pipeline::ExposePrintOutputs exposeOut{};
                    if (!Pipeline::ExposePrintStage::run(*_ws, exposeIn, exposeOut, scratch)) {
                        JTRACE("PRINT", "FATAL: failed to convert negative densities to print log raw");
                        abortFlag.store(true, std::memory_order_relaxed);
                        failure.store(true, std::memory_order_relaxed);
                        break;
                    }

                    Pipeline::DevelopPrintInputs devIn{};
                    devIn.printRuntime = _prt;
                    devIn.printLogRaw = exposeOut.printLogRaw;

                    Pipeline::DevelopPrintOutputs devOut{};
                    if (!Pipeline::DevelopPrintStage::run(devIn, devOut)) {
                        JTRACE("PRINT", "FATAL: failed to convert print log raw to densities");
                        abortFlag.store(true, std::memory_order_relaxed);
                        failure.store(true, std::memory_order_relaxed);
                        break;
                    }

                    _density.c[idx] = devOut.printDensity.v[0];
                    _density.m[idx] = devOut.printDensity.v[1];
                    _density.y[idx] = devOut.printDensity.v[2];
                }
            }
            });
    }

    for (std::thread& th : threads) {
        if (th.joinable()) th.join();
    }

    if (!failure.load(std::memory_order_relaxed) &&
        !abortFlag.load(std::memory_order_relaxed)) {
        _density.medium = Scanner::ScannerMedium::Print;
    }

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
    const Scanner::ScannerMediumRuntime* mediumRuntime = ctx.printActive
        ? &_ws->printMediumRuntime
        : &_ws->negativeMediumRuntime;
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
    writeNegativeDensities(ctx, threadCount);
    if (_effect.abort()) {
        return;
    }
    if (ctx.printActive) {
        convertNegativeToPrint(ctx, threadCount);
    }
    if (_effect.abort()) {
        return;
    }
    renderScannerFromDensity(ctx, threadCount);
}

void JuicerProcessor::process() {
    processImpl();
}

void JuicerProcessor::multiThreadProcessImages(OfxRectI) {
    processImpl();
}
