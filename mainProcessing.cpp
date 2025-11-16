// JuicerProcessing.cpp

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <string>

// Resolve OFX support library C++ wrappers — suppress MSVC C5040 for dynamic exception specs
#pragma warning(push)
#pragma warning(disable: 5040)
#include "ofxsProcessing.h"
#include "ofxsImageEffect.h"
#pragma warning(pop)
#include "Logging.h"
#include "SpectralData.h"
#include "SpectralProcessing.h"
#include "FilmProcessing.h"
#include "ColorTransforms.h"
#include "WorkingState.h"
#include "Print.h"
#include "Scanner.h"
#include "Couplers.h"
#include "mainProcessing.h"

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

    // Separable blur, unchanged.
    static void blurChannelSeparable(const std::vector<float>& src, std::vector<float>& tmp, std::vector<float>& dst,
        int width, int height, const std::vector<float>& k) {
        // Horizontal
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
        // Vertical
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

    static bool run_print_pipeline_from_dir_corrected_densities(
        const WorkingState& ws,
        const Print::Runtime& prt,
        const Print::Params& prm,
        const float D_cmy[3],
        float kMid_spectral,
        float rgbOut[3],
        JuicerProc::PrintPipelineScratch& scratch,
        int debug_x = -1,
        int debug_y = -1)
    {
        const int viewK = ws.tablesView.K;
        const int printK = ws.tablesPrint.K;
        const int shapeK = Spectral::gShape.K;
        if (viewK <= 0 || printK <= 0 || shapeK <= 0) {
            JTRACE("PRINT", std::string("print pipeline aborted: non-positive spectral length view=")
                + std::to_string(viewK)
                + " print=" + std::to_string(printK)
                + " gShape=" + std::to_string(shapeK));
            return false;
        }
        if (viewK != printK || viewK != shapeK) {
            JTRACE("PRINT", std::string("print pipeline aborted: spectral length mismatch view=")
                + std::to_string(viewK)
                + " print=" + std::to_string(printK)
                + " gShape=" + std::to_string(shapeK));
            return false;
        }

        const int K = shapeK;
        auto& Tneg = scratch.Tneg;
        auto& Ee_expose = scratch.Ee_expose;
        auto& Ee_filtered = scratch.Ee_filtered;
        auto& Tprint = scratch.Tprint;
        auto& Ee_viewed = scratch.Ee_viewed;
        auto& Tpreflash = scratch.Tpreflash;
        auto& Ee_preflash = scratch.Ee_preflash;

        Print::negative_T_from_dyes(ws, D_cmy, Tneg);
        if (int(Tneg.size()) < K) {
            Tneg.resize(size_t(K), 0.0f);
        }

        Ee_expose.resize(size_t(K));
        for (int i = 0; i < K; ++i) {
            const float Ee = (prt.illumEnlarger.linear.size() > size_t(i))
                ? prt.illumEnlarger.linear[i]
                : 1.0f;
            Ee_expose[i] = std::max(0.0f, Ee * Tneg[i]);
        }

        Ee_filtered.resize(size_t(K));
        const float yAmount = Print::compose_dichroic_amount(prt.neutralY, prm.yFilter);
        const float mAmount = Print::compose_dichroic_amount(prt.neutralM, prm.mFilter);
        const float cAmount = Print::compose_dichroic_amount(prt.neutralC, 0.0f);
        for (int i = 0; i < K; ++i) {
            const float fY = Print::blend_dichroic_filter_linear(
                (prt.filterY.linear.size() > size_t(i)) ? prt.filterY.linear[i] : 1.0f,
                yAmount);
            const float fM = Print::blend_dichroic_filter_linear(
                (prt.filterM.linear.size() > size_t(i)) ? prt.filterM.linear[i] : 1.0f,
                mAmount);
            const float fC = Print::blend_dichroic_filter_linear(
                (prt.filterC.linear.size() > size_t(i)) ? prt.filterC.linear[i] : 1.0f,
                cAmount);
            const float fTotal = fY * fM * fC;
            Ee_filtered[i] = std::max(0.0f, Ee_expose[i] * fTotal);
        }

        // SPD DEBUG: Log spectral data for center pixel to compare with Python reference
        if (debug_x >= 0 && debug_y >= 0) {
            // Key wavelengths: 450nm (blue), 520nm (green), 650nm (red)
            // Spectral shape: 380-780nm in 5nm steps, so index = (wavelength - 380) / 5
            const int idx_450 = (450 - 380) / 5;  // index 14
            const int idx_520 = (520 - 380) / 5;  // index 28
            const int idx_650 = (650 - 380) / 5;  // index 54

            std::ostringstream oss1, oss2, oss3, oss4;
            oss1 << "SPD_TRACE pixel(" << debug_x << "," << debug_y << "): Filter amounts: Y=" << yAmount << " M=" << mAmount << " C=" << cAmount;
            oss2 << "SPD_TRACE pixel(" << debug_x << "," << debug_y << "): Tneg[450nm]=" << Tneg[idx_450] << " [520nm]=" << Tneg[idx_520] << " [650nm]=" << Tneg[idx_650];
            oss3 << "SPD_TRACE pixel(" << debug_x << "," << debug_y << "): Ee_expose[450nm]=" << Ee_expose[idx_450] << " [520nm]=" << Ee_expose[idx_520] << " [650nm]=" << Ee_expose[idx_650];
            oss4 << "SPD_TRACE pixel(" << debug_x << "," << debug_y << "): Ee_filtered[450nm]=" << Ee_filtered[idx_450] << " [520nm]=" << Ee_filtered[idx_520] << " [650nm]=" << Ee_filtered[idx_650];
            JTRACE("SPECTRAL", oss1.str());
            JTRACE("SPECTRAL", oss2.str());
            JTRACE("SPECTRAL", oss3.str());
            JTRACE("SPECTRAL", oss4.str());
        }

        float raw[3];
        Print::raw_exposures_from_filtered_light(prt.profile, Ee_filtered, raw);

        const float expPrint = std::isfinite(prm.exposure)
            ? std::max(0.0f, prm.exposure)
            : 1.0f;
        const float rawScale = expPrint * kMid_spectral;
        raw[0] *= rawScale;
        raw[1] *= rawScale;
        raw[2] *= rawScale;

        if (std::isfinite(prm.preflashExposure) && prm.preflashExposure > 0.0f) {
            float rawPre[3];
            Print::compute_preflash_raw(prt, ws, Tpreflash, Ee_preflash, rawPre);
            raw[0] += rawPre[0] * prm.preflashExposure;
            raw[1] += rawPre[1] * prm.preflashExposure;
            raw[2] += rawPre[2] * prm.preflashExposure;
        }

        float D_print[3];
        Print::print_densities_from_Eprint(prt.profile, raw, D_print);

        // SPD DEBUG: Log print densities and raw exposures
        if (debug_x >= 0 && debug_y >= 0) {
            std::ostringstream oss1, oss2;
            oss1 << "PRINT_DENSITY pixel(" << debug_x << "," << debug_y << "): raw[B/G/R]=" << raw[0] << "/" << raw[1] << "/" << raw[2];
            oss2 << "PRINT_DENSITY pixel(" << debug_x << "," << debug_y << "): D_print[C/M/Y]=" << D_print[0] << "/" << D_print[1] << "/" << D_print[2];
            JTRACE("SPECTRAL", oss1.str());
            JTRACE("SPECTRAL", oss2.str());
        }

        Print::print_T_from_dyes(prt.profile, D_print, Tprint);
        if (int(Tprint.size()) < K) {
            Tprint.resize(size_t(K), 0.0f);
        }

        // SPD DEBUG: Log print transmittance SPD
        if (debug_x >= 0 && debug_y >= 0) {
            const int idx_450 = 14;
            const int idx_520 = 28;
            const int idx_650 = 54;
            std::ostringstream oss;
            oss << "PRINT_T pixel(" << debug_x << "," << debug_y << "): Tprint[450nm]=" << Tprint[idx_450] << " [520nm]=" << Tprint[idx_520] << " [650nm]=" << Tprint[idx_650];
            JTRACE("SPECTRAL", oss.str());
        }

        Ee_viewed.resize(size_t(K));
        for (int i = 0; i < K; ++i) {
            const float Ev = (prt.illumView.linear.size() > size_t(i))
                ? prt.illumView.linear[i]
                : 1.0f;
            Ee_viewed[i] = std::max(0.0f, Ev * Tprint[i]);
        }

        // SPD DEBUG: Log scanner viewing SPD
        if (debug_x >= 0 && debug_y >= 0) {
            const int idx_450 = 14;
            const int idx_520 = 28;
            const int idx_650 = 54;
            std::ostringstream oss;
            oss << "SCANNER_SPD pixel(" << debug_x << "," << debug_y << "): Ee_viewed[450nm]=" << Ee_viewed[idx_450] << " [520nm]=" << Ee_viewed[idx_520] << " [650nm]=" << Ee_viewed[idx_650];
            JTRACE("SPECTRAL", oss.str());
        }

        float XYZ[3];
        Spectral::Ee_to_XYZ_given_tables(ws.tablesPrint, Ee_viewed, XYZ);

        // SPD DEBUG: Log final XYZ and RGB output
        if (debug_x >= 0 && debug_y >= 0) {
            std::ostringstream oss;
            oss << "FINAL_OUTPUT pixel(" << debug_x << "," << debug_y << "): XYZ=" << XYZ[0] << "/" << XYZ[1] << "/" << XYZ[2];
            JTRACE("SPECTRAL", oss.str());
        }

        Spectral::XYZ_to_DWG_linear_adapted(ws.tablesPrint, XYZ, rgbOut);
        rgbOut[0] = std::max(0.0f, rgbOut[0]);
        rgbOut[1] = std::max(0.0f, rgbOut[1]);
        rgbOut[2] = std::max(0.0f, rgbOut[2]);
        return true;
    }

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
        work.logE_B.assign(total, 0.0f);
        work.logE_G.assign(total, 0.0f);
        work.logE_R.assign(total, 0.0f);
        work.corrY.assign(total, 0.0f);
        work.corrM.assign(total, 0.0f);
        work.corrC.assign(total, 0.0f);

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
                    work.logE_B[idx] = work.logE_G[idx] = work.logE_R[idx] = 0.0f;
                    work.corrY[idx] = work.corrM[idx] = work.corrC[idx] = 0.0f;
                    continue;
                }

                // SPD DEBUG: Log input RGB for center pixel
                if (xx == center_xx && yy == center_yy) {
                    std::ostringstream oss;
                    oss << "INPUT_RGB tile_pixel(" << xx << "," << yy << "): R=" << rgbIn[0] << " G=" << rgbIn[1] << " B=" << rgbIn[2];
                    JTRACE("SPECTRAL", oss.str());
                }

                float E[3];
                const Spectral::SpectralTables* tablesSPD =
                    (ws.spdReady && ws.tablesRef.K > 0) ? &ws.tablesRef : nullptr;
                const float sExp = (std::isfinite(exposureScale) ? std::max(0.0f, exposureScale) : 1.0f);
                Spectral::rgb_input_to_film_raw(
                    rgbIn, E, sExp,
                    ws.filmRaw,
                    tablesSPD,
                    (ws.spdReady ? ws.spdSInv : nullptr),
                    ws.spdReady,
                    sensB_forExposure,
                    sensG_forExposure,
                    sensR_forExposure);

                // SPD DEBUG: Log film raw exposure (pre-log) for center pixel
                if (xx == center_xx && yy == center_yy) {
                    std::ostringstream oss;
                    oss << "FILM_RAW tile_pixel(" << xx << "," << yy << "): B=" << E[0] << " G=" << E[1] << " R=" << E[2];
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

                float leB = std::log10(std::max(0.0f, E[0]) + 1e-10f);
                float leG = std::log10(std::max(0.0f, E[1]) + 1e-10f);
                float leR = std::log10(std::max(0.0f, E[2]) + 1e-10f);

                if (!std::isfinite(leB) && !ws.densB.lambda_nm.empty()) {
                    leB = ws.densB.lambda_nm.front();
                }
                if (!std::isfinite(leG) && !ws.densG.lambda_nm.empty()) {
                    leG = ws.densG.lambda_nm.front();
                }
                if (!std::isfinite(leR) && !ws.densR.lambda_nm.empty()) {
                    leR = ws.densR.lambda_nm.front();
                }

                work.logE_B[idx] = leB;
                work.logE_G[idx] = leG;
                work.logE_R[idx] = leR;

                float D_Y = Spectral::sample_density_at_logE(ws.densB, leB, ws.gammaFactorB);
                float D_M = Spectral::sample_density_at_logE(ws.densG, leG, ws.gammaFactorG);
                float D_C = Spectral::sample_density_at_logE(ws.densR, leR, ws.gammaFactorR);

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
    , _prt(nullptr)
    , _ws(nullptr)
    , _wsReady(false)
    , _printReady(false)
    , _exposureScale(1.0f)
    , _outputEncoding{}
{
}

void JuicerProcessor::setSrcDst(OFX::Image* src, OFX::Image* dst) {
    _srcImg = src;
    setDstImg(dst);
}

void JuicerProcessor::setRenderWindowRect(const OfxRectI& rect) { setRenderWindow(rect); }
void JuicerProcessor::setComponents(int n) { _nComponents = n; }
void JuicerProcessor::setScannerParams(const Scanner::Params& p) { _scannerParams = p; }
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

JuicerProcessor::RenderContext JuicerProcessor::prepareRenderContext(const OfxRectI& procWindow) const {
    RenderContext ctx;
    ctx.window = procWindow;
    ctx.tileWidth = procWindow.x2 - procWindow.x1;
    ctx.tileHeight = procWindow.y2 - procWindow.y1;
    ctx.exposureScaleSafe = (std::isfinite(_exposureScale) && _exposureScale > 0.0f)
        ? _exposureScale
        : 1.0f;
    ctx.useSpatialDIR = (_dirRT.active && std::isfinite(_dirRT.spatialSigmaPixels) &&
        _dirRT.spatialSigmaPixels > 0.0f && _nComponents >= 3 && _wsReady && _ws);

    ctx.kMidSpectral = 1.0f;
    if (_wsReady && _ws && _printReady && _prt && !_printParams.bypass) {
        const float exposureCompScale = _printParams.exposureCompensationEnabled
            ? _printParams.exposureCompensationScale
            : 1.0f;
        ctx.kMidSpectral = Print::compute_exposure_factor_midgray(
            *_ws,
            *_prt,
            _printParams,
            _dirRT,
            exposureCompScale);
    }

    return ctx;
}

void JuicerProcessor::renderSpatialDIR(const RenderContext& ctx) {
    const int tileW = ctx.tileWidth;
    const int tileH = ctx.tileHeight;

    auto& dirWorkspace = _dirWorkspace;

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
        tileW,
        tileH,
        *_ws,
        _dirRT,
        ctx.exposureScaleSafe,
        fetchRGB,
        abortCheck,
        dirWorkspace,
        _gaussianKernel);

    const bool runPrint = _printReady && _prt && !_printParams.bypass;

    for (int yy = 0; yy < tileH; ++yy) {
        if (_effect.abort()) break;
        for (int xx = 0; xx < tileW; ++xx) {
            const int x = ctx.window.x1 + xx;
            const int y = ctx.window.y1 + yy;

            float* dstPix = reinterpret_cast<float*>(_dstImg->getPixelAddress(x, y));
            const float* srcPix = reinterpret_cast<const float*>(_srcImg->getPixelAddress(x, y));
            if (!dstPix || !srcPix) {
                continue;
            }

            float rgbOut[3] = { srcPix[0], srcPix[1], srcPix[2] };

            const size_t idx = size_t(yy * tileW + xx);
            float leB2 = dirWorkspace.logE_B[idx] - dirWorkspace.corrYBlur[idx];
            float leG2 = dirWorkspace.logE_G[idx] - dirWorkspace.corrMBlur[idx];
            float leR2 = dirWorkspace.logE_R[idx] - dirWorkspace.corrCBlur[idx];

            if (!std::isfinite(leB2) && !_ws->densB.lambda_nm.empty()) {
                leB2 = _ws->densB.lambda_nm.front();
            }
            if (!std::isfinite(leG2) && !_ws->densG.lambda_nm.empty()) {
                leG2 = _ws->densG.lambda_nm.front();
            }
            if (!std::isfinite(leR2) && !_ws->densR.lambda_nm.empty()) {
                leR2 = _ws->densR.lambda_nm.front();
            }

            const Spectral::Curve& dirB = (_ws->dirPrecorrected ? _ws->dirDensB : _ws->densB);
            const Spectral::Curve& dirG = (_ws->dirPrecorrected ? _ws->dirDensG : _ws->densG);
            const Spectral::Curve& dirR = (_ws->dirPrecorrected ? _ws->dirDensR : _ws->densR);

            // SPD DEBUG: Log film exposure before density lookup (center pixel only)
            const int center_x = _renderWindow.x1 + (_renderWindow.x2 - _renderWindow.x1) / 2;
            const int center_y = _renderWindow.y1 + (_renderWindow.y2 - _renderWindow.y1) / 2;
            if (x == center_x && y == center_y) {
                std::ostringstream oss;
                oss << "FILM_EXPOSURE pixel(" << x << "," << y << "): logE[B/G/R]=" << leB2 << "/" << leG2 << "/" << leR2;
                JTRACE("SPECTRAL", oss.str());
            }

            float D_cmy[3];
            D_cmy[0] = Spectral::sample_density_at_logE(dirB, leB2, _ws->gammaFactorB);
            D_cmy[1] = Spectral::sample_density_at_logE(dirG, leG2, _ws->gammaFactorG);
            D_cmy[2] = Spectral::sample_density_at_logE(dirR, leR2, _ws->gammaFactorR);

            // SPD DEBUG: Log film densities after lookup (center pixel only)
            if (x == center_x && y == center_y) {
                std::ostringstream oss;
                oss << "FILM_DENSITY pixel(" << x << "," << y << "): D_cmy[C/M/Y]=" << D_cmy[0] << "/" << D_cmy[1] << "/" << D_cmy[2];
                JTRACE("SPECTRAL", oss.str());
            }

            Print::clamp_negative_densities_to_dmax(*_ws, _dirRT, D_cmy);

            if (runPrint) {
                // Pass debug coordinates for center pixel logging
                const int center_x = _renderWindow.x1 + (_renderWindow.x2 - _renderWindow.x1) / 2;
                const int center_y = _renderWindow.y1 + (_renderWindow.y2 - _renderWindow.y1) / 2;
                const int dbg_x = (x == center_x && y == center_y) ? x : -1;
                const int dbg_y = (x == center_x && y == center_y) ? y : -1;

                if (!run_print_pipeline_from_dir_corrected_densities(
                    *_ws,
                    *_prt,
                    _printParams,
                    D_cmy,
                    ctx.kMidSpectral,
                    rgbOut,
                    _printScratch,
                    dbg_x,
                    dbg_y))
                {
                    OutputEncoding::applyEncoding(_outputEncoding, rgbOut);
                    dstPix[0] = rgbOut[0];
                    dstPix[1] = rgbOut[1];
                    dstPix[2] = rgbOut[2];
                    if (_nComponents == 4) dstPix[3] = srcPix[3];
                    continue;
                }
            }
            else {
                const Spectral::SpectralTables* tables = nullptr;
                if (_ws->tablesScan.K > 0) {
                    tables = &_ws->tablesScan;
                }
                else if (_ws->tablesView.K > 0) {
                    tables = &_ws->tablesView;
                }

                if (tables) {
                    float XYZ[3] = { 0.0f, 0.0f, 0.0f };
                    const bool useBaseline = _ws->hasBaseline && tables->hasBaseline;
                    if (useBaseline) {
                        Spectral::dyes_to_XYZ_with_baseline_given_tables(*tables, D_cmy, XYZ);
                    }
                    else {
                        Spectral::dyes_to_XYZ_given_tables(*tables, D_cmy, XYZ);
                    }

                    Spectral::XYZ_to_DWG_linear_adapted(*tables, XYZ, rgbOut);
                    rgbOut[0] = std::max(0.0f, rgbOut[0]);
                    rgbOut[1] = std::max(0.0f, rgbOut[1]);
                    rgbOut[2] = std::max(0.0f, rgbOut[2]);
                }
            }

            OutputEncoding::applyEncoding(_outputEncoding, rgbOut);
            dstPix[0] = rgbOut[0];
            dstPix[1] = rgbOut[1];
            dstPix[2] = rgbOut[2];
            if (_nComponents == 4) dstPix[3] = srcPix[3];
        }
    }
}

void JuicerProcessor::renderScalar(const RenderContext& ctx) {
    const bool printActive = (_wsReady && _ws && _printReady && _prt && !_printParams.bypass);
    const float kMid_spectral = printActive ? ctx.kMidSpectral : 1.0f;

    for (int y = ctx.window.y1; y < ctx.window.y2; ++y) {
        if (_effect.abort()) break;

        if (_nComponents >= 3) {
            for (int x = ctx.window.x1; x < ctx.window.x2; ++x) {
                float* dstPix = reinterpret_cast<float*>(_dstImg->getPixelAddress(x, y));
                const float* srcPix = reinterpret_cast<const float*>(_srcImg->getPixelAddress(x, y));
                if (!dstPix || !srcPix) continue;

                float rgbIn[3] = { srcPix[0], srcPix[1], srcPix[2] };
                float rgbOut[3] = { rgbIn[0],  rgbIn[1],  rgbIn[2] };

                if (_wsReady && _ws) {
                    if (_printParams.bypass || !_printReady || !_prt) {
                        Scanner::simulate_scanner(
                            rgbIn, rgbOut,
                            _scannerParams, _dirRT, *_ws,
                            ctx.exposureScaleSafe);
                    }
                    else {
                        Print::simulate_print_pixel(
                            rgbIn, _printParams,
                            *_prt, _dirRT, *_ws,
                            ctx.exposureScaleSafe,
                            kMid_spectral,
                            rgbOut);
                    }
                }

                OutputEncoding::applyEncoding(_outputEncoding, rgbOut);
                dstPix[0] = rgbOut[0];
                dstPix[1] = rgbOut[1];
                dstPix[2] = rgbOut[2];
                if (_nComponents == 4) dstPix[3] = srcPix[3];
            }
        }
        else if (_nComponents == 1) {
            for (int x = ctx.window.x1; x < ctx.window.x2; ++x) {
                float* dstPix = reinterpret_cast<float*>(_dstImg->getPixelAddress(x, y));
                const float* srcPix = reinterpret_cast<const float*>(_srcImg->getPixelAddress(x, y));
                if (dstPix && srcPix) dstPix[0] = srcPix[0];
            }
        }
    }
}

void JuicerProcessor::multiThreadProcessImages(OfxRectI procWindow) {
    if (!_srcImg || !_dstImg) return;

#if defined(JUICER_SPD_DEBUG)
    Spectral::spd_probe_reset();
#endif

    auto curvesReady = [&]()->bool {
        if (!_ws) return false;
        return curve_ok(_ws->densB) && curve_ok(_ws->densG) && curve_ok(_ws->densR);
        };

    RenderContext ctx = prepareRenderContext(procWindow);

    if (ctx.useSpatialDIR && curvesReady()) {
        renderSpatialDIR(ctx);
    }
    else {
        renderScalar(ctx);
    }
}
