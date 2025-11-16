#include "JuicerEffect.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ColorTransforms.h"
#include "Couplers.h"
#include "Illuminants.h"
#include "IlluminantKeys.h"
#include "NeutralFilters.h"
#include "OutputEncoding.h"
#include "Print.h"
#include "ParamNames.h"
#include "Scanner.h"
#include "SpectralData.h"
#include "FilmProcessing.h"
#include "Logging.h"
#include "mainProcessing.h"

namespace {
    static std::once_flag gSpectralGlobalsOnce;

    inline bool nearly_equal_double(double a, double b) {
        const double diff = std::fabs(a - b);
        const double scale = std::max({ 1.0, std::fabs(a), std::fabs(b) });
        return diff <= scale * 1e-9;
    }

    static double build_center_weight_mask(int width, int height, double sigma, std::vector<double>& outMask) {
        outMask.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
        if (!(std::isfinite(sigma)) || sigma <= 0.0) {
            std::fill(outMask.begin(), outMask.end(), 0.0);
            return 0.0;
        }
        double sumMask = 0.0;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const double nx = static_cast<double>(x) / static_cast<double>(width) - 0.5;
                const double ny = static_cast<double>(y) / static_cast<double>(height) - 0.5;
                const double maxDim = static_cast<double>(std::max(width, height));
                const double invMax = (maxDim > 0.0) ? (1.0 / maxDim) : 0.0;
                const double normX = nx * static_cast<double>(width) * invMax;
                const double normY = ny * static_cast<double>(height) * invMax;
                const double r2 = normX * normX + normY * normY;
                const double w = std::exp(-r2 / (2.0 * sigma * sigma));
                outMask[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] = w;
                sumMask += w;
            }
        }
        return sumMask;
    }

    static double measure_center_weighted_Y_DWG_cached(
        OFX::Image* img,
        const OfxRectI& bounds,
        double sigma,
        InstanceState* state,
        double renderScaleX,
        double renderScaleY,
        std::uintptr_t clipToken,
        Spectral::InputColorSpace inputColorSpace,
        const Spectral::Mat3& rgbToXYZ,
        bool applyCctfDecoding) {

        if (!img) {
            return 0.0;
        }

        const int width = bounds.x2 - bounds.x1;
        const int height = bounds.y2 - bounds.y1;
        if (width <= 0 || height <= 0) {
            return 0.0;
        }

        auto accumulateY = [&](const std::vector<double>& mask, double* outSumMask) {
            double sumY = 0.0;
            double sumMask = 0.0;
            for (int yy = bounds.y1; yy < bounds.y2; ++yy) {
                const size_t rowOffset = static_cast<size_t>(yy - bounds.y1) * static_cast<size_t>(width);
                for (int xx = bounds.x1; xx < bounds.x2; ++xx) {
                    const float* pix = reinterpret_cast<const float*>(img->getPixelAddress(xx, yy));
                    if (!pix) {
                        continue;
                    }
                    const double w = mask[rowOffset + static_cast<size_t>(xx - bounds.x1)];
                    float linear[3];
                    Spectral::apply_input_cctf_decoding(inputColorSpace, applyCctfDecoding, pix, linear);
                    float XYZ[3];
                    rgbToXYZ.mul(linear, XYZ);
                    const double Y = static_cast<double>(XYZ[1]);
                    if (!std::isfinite(Y)) {
                        continue;
                    }
                    sumY += Y * w;
                    sumMask += w;
                }
            }
            if (outSumMask) {
                *outSumMask = sumMask;
            }
            if (sumMask <= 0.0) {
                return 0.0;
            }
            return sumY / sumMask;
            };

        if (!state) {
            std::vector<double> mask;
            build_center_weight_mask(width, height, sigma, mask);
            return accumulateY(mask, nullptr);
        }

        std::lock_guard<std::mutex> lock(state->autoExposureMutex);

        bool rebuildMask = !state->autoExposureMaskValid
            || state->autoExposureMaskWidth != width
            || state->autoExposureMaskHeight != height
            || !nearly_equal_double(state->autoExposureMaskSigma, sigma)
            || !nearly_equal_double(state->autoExposureMaskRenderScaleX, renderScaleX)
            || !nearly_equal_double(state->autoExposureMaskRenderScaleY, renderScaleY)
            || state->autoExposureMaskClipToken != clipToken;

        if (rebuildMask) {
            const double sumMask = build_center_weight_mask(width, height, sigma, state->autoExposureMaskWeights);
            state->autoExposureMaskWidth = width;
            state->autoExposureMaskHeight = height;
            state->autoExposureMaskSigma = sigma;
            state->autoExposureMaskSum = sumMask;
            state->autoExposureMaskRenderScaleX = renderScaleX;
            state->autoExposureMaskRenderScaleY = renderScaleY;
            state->autoExposureMaskClipToken = clipToken;
            state->autoExposureMaskValid = sumMask > 0.0;
        }

        if (!state->autoExposureMaskValid) {
            state->autoExposureMaskSum = 0.0;
            return 0.0;
        }

        double effectiveSumMask = 0.0;
        const double measuredY = accumulateY(state->autoExposureMaskWeights, &effectiveSumMask);
        state->autoExposureMaskSum = effectiveSumMask;
        return measuredY;
    }

    void init_spectral_globals_once() {
        try {
            Spectral::lock_shape_to_reference_axis();
            const auto cmf = Spectral::load_csv_triplets(data_dir_string("cie1931_2deg.csv"));
            if (!Spectral::cmf_triplets_match_reference_axis(cmf)) {
                JTRACE("INIT", "FATAL: CMF wavelengths do not match 380-780@5nm grid");
                throw std::runtime_error("CMF grid mismatch");
            }
            Spectral::set_cie_1931_2deg_cmf(cmf.xbar, cmf.ybar, cmf.zbar);
            Spectral::ensure_precomputed_up_to_date();
            Spectral::disable_hanatos_if_reference_mismatch();
        }
        catch (...) {
            // Leave globals as-is; instance guards will pass-through if shape is invalid.
        }

        // Hanatos LUT: load once and set availability flag atomically.
        try {
            const std::string lutPath = data_dir_string("luts", "spectral_upsampling", "irradiance_xy_tc.npy");
            Spectral::load_hanatos_spectra_lut(lutPath);
        }
        catch (...) {
            Spectral::set_hanatos_available(false);
        }

        // KG3 filter fallback: set once if not present; safe idempotently.
        std::vector<std::pair<float, float>> kg3_pairs_raw;
        try { kg3_pairs_raw = Spectral::load_csv_pairs(data_dir_string("filters", "heat_absorbing", "schott", "KG3.csv")); }
        catch (...) { kg3_pairs_raw.clear(); }
        if (kg3_pairs_raw.empty()) {
            kg3_pairs_raw = {
                { Spectral::gShape.lambdaMin, 1.0f },
                { Spectral::gShape.lambdaMax, 1.0f }
            };
        }
        Spectral::set_filter_KG3_from_pairs(kg3_pairs_raw);
    }


    static std::vector<std::string> enlarger_illuminant_keys_for_choice(int choice) {
        switch (choice) {
        case 0: return { "D65", "d65" };
        case 1: return { "D55", "d55" };
        case 2: return { "D50", "d50" };
        case 3: return { "TH-KG3-L", "th-kg3-l" };
        case 4: return { "T", "t", "Incandescent", "incandescent" };
        case 5: return { "K75P", "k75p", "Kinoton75P", "Kinoton 75P", "kinoton75p", "kinoton_75p" };
        case 6: return { "EqualEnergy", "equal_energy", "Equal energy" };
        default: break;
        }
        return {};
    }

    static int illuminant_choice_index_from_string(const std::string& value) {
        const std::string normalized = IlluminantKeys::normalize(value);
        if (normalized.empty()) {
            return -1;
        }
        constexpr int kIlluminantChoiceCount = 7;
        for (int choice = 0; choice < kIlluminantChoiceCount; ++choice) {
            const auto keys = enlarger_illuminant_keys_for_choice(choice);
            for (const std::string& key : keys) {
                if (IlluminantKeys::normalize(key) == normalized) {
                    return choice;
                }
            }
        }
        return -1;
    }

} // namespace

JuicerEffect::ExposureParams JuicerEffect::gatherExposureParams() const {
    ExposureParams params{};
    double exposureSliderEV = 0.0;
    if (_pExposure) {
        _pExposure->getValue(exposureSliderEV);
    }
    if (!std::isfinite(exposureSliderEV)) {
        exposureSliderEV = 0.0;
    }
    params.sliderEV = exposureSliderEV;
    params.sliderScale = static_cast<float>(std::pow(2.0, exposureSliderEV));
    if (!std::isfinite(params.sliderScale)) {
        params.sliderScale = 1.0f;
    }
    bool cameraAuto = true;
    if (_pCameraAutoExposure) {
        _pCameraAutoExposure->getValue(cameraAuto);
    }
    params.cameraAutoEnabled = cameraAuto;
    return params;
}

Scanner::Params JuicerEffect::gatherScannerParams() const {
    Scanner::Params params{};
    bool scanEnabled = false;
    bool scanAuto = true;
    double scanY = 0.18;
    double scanFilmMm = 36.0;
    if (_pScanEnabled) _pScanEnabled->getValue(scanEnabled);
    if (_pScanAuto) _pScanAuto->getValue(scanAuto);
    if (_pScanTargetY) _pScanTargetY->getValue(scanY);
    if (_pScanFilmLongEdge) _pScanFilmLongEdge->getValue(scanFilmMm);
    params.enabled = scanEnabled;
    params.autoExposure = scanAuto;
    params.targetY = static_cast<float>(scanY);
    params.filmLongEdgeMm =
        (std::isfinite(scanFilmMm) && scanFilmMm > 0.0)
        ? static_cast<float>(scanFilmMm)
        : 36.0f;
    return params;
}

Print::Params JuicerEffect::gatherPrintParams() const {
    Print::Params params{};
    bool bypass = true;
    double pexp = 1.0;
    double preflash = 0.0;
    double y = 0.0;
    double m = 0.0;
    if (_pPrintBypass) _pPrintBypass->getValue(bypass);
    if (_pPrintExposure) _pPrintExposure->getValue(pexp);
    if (_pPrintPreflash) _pPrintPreflash->getValue(preflash);
    if (_pEnlargerY) _pEnlargerY->getValue(y);
    if (_pEnlargerM) _pEnlargerM->getValue(m);
    auto clampShift = [](double v) -> double {
        if (!std::isfinite(v)) return 0.0;
        const double limit = static_cast<double>(Print::kEnlargerSteps);
        return std::clamp(v, -limit, limit);
        };
    params.bypass = bypass;
    params.exposure = static_cast<float>(pexp);
    params.preflashExposure = static_cast<float>(preflash);
    params.yFilter = static_cast<float>(clampShift(y));
    params.mFilter = static_cast<float>(clampShift(m));
    return params;
}

OutputEncoding::Params JuicerEffect::gatherOutputEncodingParams() const {
    OutputEncoding::Params params{};
    int csIndex = OutputEncoding::toIndex(OutputEncoding::ColorSpace::sRGB);
    if (_pOutputColorSpace) _pOutputColorSpace->getValue(csIndex);
    bool applyCctf = true;
    if (_pOutputCctfEncoding) _pOutputCctfEncoding->getValue(applyCctf);
    bool preserveLinear = false;
    if (_pOutputLinearPassThrough) _pOutputLinearPassThrough->getValue(preserveLinear);
    params.colorSpace = OutputEncoding::colorSpaceFromIndex(csIndex);
    params.applyCctfEncoding = applyCctf;
    params.preserveLinearRange = preserveLinear;
    return params;
}

JuicerEffect::AutoExposureResult JuicerEffect::computeAutoExposure(
    const OFX::RenderArguments& args,
    OFX::Image* srcImg,
    const OfxRectI& fullBounds,
    const Scanner::Params& scannerParams,
    const ExposureParams& exposureParams) const {

    AutoExposureResult result{};
    result.exposureScale = 1.0f;
    result.autoEV = 0.0;

    if (!srcImg) {
        result.exposureScale = static_cast<float>(std::pow(2.0, exposureParams.sliderEV));
        if (!std::isfinite(result.exposureScale)) {
            result.exposureScale = 1.0f;
        }
        return result;
    }

    auto rect_equal = [](const OfxRectI& a, const OfxRectI& b) {
        return a.x1 == b.x1 && a.y1 == b.y1 && a.x2 == b.x2 && a.y2 == b.y2;
        };

    OfxRectI meterBounds = fullBounds;
    if (_src) {
        try {
            const OfxRectD rod = _src->getRegionOfDefinition(args.time);
            const double rodWidth = rod.x2 - rod.x1;
            const double rodHeight = rod.y2 - rod.y1;
            if (std::isfinite(rodWidth) && rodWidth > 0.0 &&
                std::isfinite(rodHeight) && rodHeight > 0.0) {
                meterBounds.x1 = static_cast<int>(std::floor(rod.x1));
                meterBounds.y1 = static_cast<int>(std::floor(rod.y1));
                meterBounds.x2 = static_cast<int>(std::ceil(rod.x2));
                meterBounds.y2 = static_cast<int>(std::ceil(rod.y2));
            }
        }
        catch (...) {
            // Ignore failures; fall back to full bounds.
        }
    }

    const WorkingState* wsCur = (_state ? _state->activeWS.load(std::memory_order_acquire) : nullptr);
    const uint64_t wsBuildCounter = wsCur ? wsCur->buildCounter : 0;

    // Camera auto-exposure always meters against AgX's fixed 18.4% target (independent of scanner target tweaks).
    constexpr double kCameraMeterTargetY = 0.184;

    const double sigma = 0.2;
    const double renderScaleX = (std::isfinite(args.renderScale.x) && args.renderScale.x > 0.0)
        ? args.renderScale.x
        : 1.0;
    const double renderScaleY = (std::isfinite(args.renderScale.y) && args.renderScale.y > 0.0)
        ? args.renderScale.y
        : 1.0;
    const std::uintptr_t clipToken = reinterpret_cast<std::uintptr_t>(_src);

    int inputColorSpaceIndex = Spectral::inputColorSpaceToIndex(Spectral::InputColorSpace::DaVinciWideGamut);
    if (_pInputColorSpace) {
        _pInputColorSpace->getValue(inputColorSpaceIndex);
    }
    bool applyInputCctfDecoding = false;
    if (_pInputCctfDecoding) {
        _pInputCctfDecoding->getValue(applyInputCctfDecoding);
    }
    const Spectral::InputColorSpace inputColorSpace =
        Spectral::inputColorSpaceFromIndex(inputColorSpaceIndex);
    const Spectral::Mat3 inputRgbToXYZ = Spectral::matrix_input_rgb_to_xyz(inputColorSpace);

    double autoEV = 0.0;
    bool haveCachedAutoEV = false;
    const bool cameraAutoEnabled = exposureParams.cameraAutoEnabled;
    if (_state) {
        std::lock_guard<std::mutex> cacheLock(_state->autoExposureMutex);
        if (_state->autoExposureCacheValid &&
            _state->autoExposureCacheAutoEnabled == cameraAutoEnabled &&
            nearly_equal_double(_state->autoExposureCacheTime, args.time) &&
            _state->autoExposureCacheBuildCounter == wsBuildCounter &&
            rect_equal(_state->autoExposureCacheBounds, meterBounds) &&
            nearly_equal_double(_state->autoExposureCacheRenderScaleX, renderScaleX) &&
            nearly_equal_double(_state->autoExposureCacheRenderScaleY, renderScaleY) &&
            _state->autoExposureCacheClipToken == clipToken &&
            _state->autoExposureCacheInputColorSpaceIndex == inputColorSpaceIndex &&
            _state->autoExposureCacheApplyCctfDecoding == applyInputCctfDecoding) {
            autoEV = _state->autoExposureCacheEV;
            haveCachedAutoEV = true;
        }
    }

    if (!haveCachedAutoEV) {
        bool measurementValid = !cameraAutoEnabled;
        double evComp = 0.0;
        if (cameraAutoEnabled) {
            const double Yexp = measure_center_weighted_Y_DWG_cached(
                srcImg,
                meterBounds,
                sigma,
                _state.get(),
                renderScaleX,
                renderScaleY,
                clipToken,
                inputColorSpace,
                inputRgbToXYZ,
                applyInputCctfDecoding);
            if (Yexp > 0.0 && kCameraMeterTargetY > 0.0) {
                const double exposureRatio = Yexp / kCameraMeterTargetY;
                evComp = -std::log(exposureRatio) / std::log(2.0);
                measurementValid = std::isfinite(evComp);
            }
            if (!std::isfinite(evComp)) {
                evComp = 0.0;
                measurementValid = false;
            }
        }
        autoEV = evComp;

        if (_state) {
            std::lock_guard<std::mutex> cacheLock(_state->autoExposureMutex);
            _state->autoExposureCacheValid = measurementValid;
            _state->autoExposureCacheTime = args.time;
            _state->autoExposureCacheAutoEnabled = cameraAutoEnabled;
            _state->autoExposureCacheBuildCounter = wsBuildCounter;
            _state->autoExposureCacheBounds = meterBounds;
            _state->autoExposureCacheEV = autoEV;
            _state->autoExposureCacheRenderScaleX = renderScaleX;
            _state->autoExposureCacheRenderScaleY = renderScaleY;
            _state->autoExposureCacheClipToken = clipToken;
            _state->autoExposureCacheInputColorSpaceIndex = inputColorSpaceIndex;
            _state->autoExposureCacheApplyCctfDecoding = applyInputCctfDecoding;
            _state->autoExposureCanonicalBounds = meterBounds;
            _state->autoExposureCanonicalValid = true;
        }
    }

    const double sliderEV = exposureParams.sliderEV;
    const double totalEV = autoEV + sliderEV;
    result.autoEV = autoEV;
    result.exposureScale = static_cast<float>(std::pow(2.0, totalEV));
    if (!std::isfinite(result.exposureScale)) {
        result.exposureScale = 1.0f;
    }
    return result;
}

#ifdef JUICER_ENABLE_COUPLERS
Couplers::Runtime JuicerEffect::prepareCouplers(
    const OFX::RenderArguments& args,
    int fullWidth,
    int fullHeight) const {

    Couplers::Runtime dirRT{};
    if (!(_state && _state->baseLoaded)) {
        return dirRT;
    }

    const WorkingState* wsCur = _state->activeWS.load(std::memory_order_acquire);
    if (wsCur && wsCur->buildCounter > 0) {
        dirRT = wsCur->dirRT;
        for (int i = 0; i < 3; ++i) {
            float v = dirRT.dMax[i];
            if (!std::isfinite(v) || v <= 0.0f) v = 1.0f;
            if (v > 1000.0f) v = 1000.0f;
            dirRT.dMax[i] = v;
        }
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                float m = dirRT.M[r][c];
                if (!std::isfinite(m)) m = 0.0f;
                if (m < -10.0f) m = -10.0f;
                if (m > 10.0f)  m = 10.0f;
                dirRT.M[r][c] = m;
            }
        }
    }

    auto valid_dim = [](double v) -> bool {
        return std::isfinite(v) && v > 0.0;
        };

    double canonicalWidth = 0.0;
    double canonicalHeight = 0.0;

    const OfxPointD projectSize = getProjectSize();
    if (valid_dim(projectSize.x) && valid_dim(projectSize.y)) {
        canonicalWidth = projectSize.x;
        canonicalHeight = projectSize.y;
    }

    if (_src && (!valid_dim(canonicalWidth) || !valid_dim(canonicalHeight))) {
        try {
            const OfxRectD rod = _src->getRegionOfDefinition(args.time);
            const double rodWidth = rod.x2 - rod.x1;
            const double rodHeight = rod.y2 - rod.y1;
            if (valid_dim(rodWidth) && valid_dim(rodHeight)) {
                canonicalWidth = rodWidth;
                canonicalHeight = rodHeight;
            }
        }
        catch (...) {
            // Ignore failures; we'll fall back to image dimensions below.
        }
    }
    if (!valid_dim(canonicalWidth) || !valid_dim(canonicalHeight)) {
        if (_state) {
            const double cachedW = _state->spatialSigmaCanonicalWidth.load(std::memory_order_acquire);
            const double cachedH = _state->spatialSigmaCanonicalHeight.load(std::memory_order_acquire);
            if (valid_dim(cachedW) && valid_dim(cachedH)) {
                canonicalWidth = cachedW;
                canonicalHeight = cachedH;
            }
        }
    }
    if (!valid_dim(canonicalWidth)) canonicalWidth = static_cast<double>(fullWidth);
    if (!valid_dim(canonicalHeight)) canonicalHeight = static_cast<double>(fullHeight);

    double filmLongEdgeMm = 35.0;
    if (_pCameraFilmFormat) {
        double filmFormat = 35.0;
        _pCameraFilmFormat->getValue(filmFormat);
        if (std::isfinite(filmFormat) && filmFormat > 0.0) {
            filmLongEdgeMm = filmFormat;
        }
    }
    float canonicalSigmaPixels = 0.0f;
    const float sigmaMicrometers = dirRT.spatialSigmaMicrometers;
    if (sigmaMicrometers > 0.0f && valid_dim(canonicalWidth) && valid_dim(canonicalHeight)) {
        const auto nearly_equal_double_local = [](double a, double b) {
            const double diff = std::fabs(a - b);
            const double scale = std::max({ 1.0, std::fabs(a), std::fabs(b) });
            return diff <= scale * 1e-9;
            };
        const auto nearly_equal_float = [](float a, float b) {
            const float diff = std::fabs(a - b);
            const float scale = std::max({ 1.0f, std::fabs(a), std::fabs(b) });
            return diff <= scale * 1e-6f;
            };

        bool cacheHit = false;
        if (_state) {
            const bool cacheValid = _state->spatialSigmaCacheValid.load(std::memory_order_acquire);
            if (cacheValid) {
                const float cachedMic = _state->spatialSigmaMicrometers.load(std::memory_order_relaxed);
                const double cachedW = _state->spatialSigmaCanonicalWidth.load(std::memory_order_relaxed);
                const double cachedH = _state->spatialSigmaCanonicalHeight.load(std::memory_order_relaxed);
                const double cachedFilm = _state->spatialSigmaCameraFilmMm.load(std::memory_order_relaxed);
                const float cachedSigma = _state->spatialSigmaPixelsCanonical.load(std::memory_order_relaxed);
                if (nearly_equal_float(cachedMic, sigmaMicrometers) &&
                    nearly_equal_double_local(cachedW, canonicalWidth) &&
                    nearly_equal_double_local(cachedH, canonicalHeight) &&
                    nearly_equal_double_local(cachedFilm, filmLongEdgeMm)) {
                    canonicalSigmaPixels = cachedSigma;
                    cacheHit = true;
                }
            }
        }

        if (!cacheHit) {
            canonicalSigmaPixels = Couplers::spatial_sigma_pixels_from_micrometers(
                sigmaMicrometers,
                filmLongEdgeMm,
                canonicalWidth,
                canonicalHeight);
            if (_state) {
                _state->spatialSigmaCanonicalWidth.store(canonicalWidth, std::memory_order_release);
                _state->spatialSigmaCanonicalHeight.store(canonicalHeight, std::memory_order_release);
                _state->spatialSigmaCameraFilmMm.store(filmLongEdgeMm, std::memory_order_release);
                _state->spatialSigmaMicrometers.store(sigmaMicrometers, std::memory_order_release);
                _state->spatialSigmaPixelsCanonical.store(canonicalSigmaPixels, std::memory_order_release);
                _state->spatialSigmaCacheValid.store(true, std::memory_order_release);
            }
        }

        double scaleX = (std::isfinite(args.renderScale.x) && args.renderScale.x > 0.0)
            ? args.renderScale.x
            : 1.0;
        double scaleY = (std::isfinite(args.renderScale.y) && args.renderScale.y > 0.0)
            ? args.renderScale.y
            : 1.0;
        double renderScaleFactor = std::max(scaleX, scaleY);
        if (!(std::isfinite(renderScaleFactor)) || renderScaleFactor <= 0.0) {
            renderScaleFactor = 1.0;
        }
        float scaledSigma = canonicalSigmaPixels * static_cast<float>(renderScaleFactor);
        if (!std::isfinite(scaledSigma) || scaledSigma < 0.0f) scaledSigma = 0.0f;
        if (scaledSigma > 25.0f) scaledSigma = 25.0f;
        dirRT.spatialSigmaPixels = scaledSigma;
    }
    else {
        dirRT.spatialSigmaPixels = 0.0f;
    }

    return dirRT;
}
#endif

JuicerEffect::WorkingStateInfo JuicerEffect::prepareWorkingState() const {
    WorkingStateInfo info{};
    if (!(_state && _state->baseLoaded)) {
        return info;
    }

    info.activeWorkingState = _state->activeWS.load(std::memory_order_acquire);
    info.workingState = info.activeWorkingState;

    const WorkingState* ws = info.workingState;
    if (ws && ws->buildCounter > 0 && ws->printRT) {
        info.printRuntime = ws->printRT.get();
    }

    const bool baselineReady = (!ws || !ws->hasBaseline) ||
        (ws->baseMin.linear.size() == static_cast<size_t>(Spectral::gShape.K));

    info.workingStateReady = (ws && ws->buildCounter > 0 &&
        ws->tablesView.K == Spectral::gShape.K &&
        ws->tablesView.epsY.size() == static_cast<size_t>(Spectral::gShape.K) &&
        ws->tablesView.epsM.size() == static_cast<size_t>(Spectral::gShape.K) &&
        ws->tablesView.epsC.size() == static_cast<size_t>(Spectral::gShape.K) &&
        baselineReady &&
        !ws->densB.lambda_nm.empty() && !ws->densB.linear.empty() &&
        !ws->densG.lambda_nm.empty() && !ws->densG.linear.empty() &&
        !ws->densR.lambda_nm.empty() && !ws->densR.linear.empty());

    const Print::Runtime* prt = info.printRuntime;
    info.printRuntimeReady =
        (prt != nullptr) &&
        Print::profile_is_valid(prt->profile) &&
        prt->illumView.linear.size() == static_cast<size_t>(Spectral::gShape.K) &&
        prt->illumEnlarger.linear.size() == static_cast<size_t>(Spectral::gShape.K) &&
        (ws && ws->tablesPrint.K == Spectral::gShape.K) &&
        info.workingStateReady;

    return info;
}

namespace JuicerRegistry {
    static std::mutex MTX;
    static std::unordered_map<OfxImageEffectHandle, InstanceState*> MAP;

    inline void set(OfxImageEffectHandle h, InstanceState* s) {
        std::lock_guard<std::mutex> lk(MTX);
        MAP[h] = s;
    }
    inline InstanceState* get(OfxImageEffectHandle h) {
        std::lock_guard<std::mutex> lk(MTX);
        auto it = MAP.find(h);
        return (it == MAP.end()) ? nullptr : it->second;
    }
    inline void erase(OfxImageEffectHandle h) {
        std::lock_guard<std::mutex> lk(MTX);
        MAP.erase(h);
    }
} // namespace JuicerRegistry

JuicerEffect::JuicerEffect(OfxImageEffectHandle handle)
    : OFX::ImageEffect(handle)
{
    // Cache clips (wrappers) for Step 2; safe even if render still uses legacy path.
    try {
        _src = fetchClip(kOfxImageEffectSimpleSourceClipName); // "Source"
        _dst = fetchClip(kOfxImageEffectOutputClipName);       // "Output"
    }
    catch (...) {
        _src = nullptr;
        _dst = nullptr;
    }

    // Cache parameter handles (wrappers)
    try {
        _pExposure = fetchDoubleParam(kParamExposure);
        _pCameraAutoExposure = fetchBooleanParam(kParamCameraAutoExposure);
        _pCameraFilmFormat = fetchDoubleParam(JuicerParams::kCameraFilmFormatMm);
        _pFilmStock = fetchChoiceParam(kParamFilmStock);
        _pPrintPaper = fetchChoiceParam(kParamPrintPaper);
        _pRefIll = fetchChoiceParam("ReferenceIlluminant");
        _pViewIll = fetchChoiceParam("ViewingIlluminant");
        _pEnlIll = fetchChoiceParam("EnlargerIlluminant");
        _pInputColorSpace = fetchChoiceParam(JuicerParams::kInputColorSpace);
        _pInputCctfDecoding = fetchBooleanParam(JuicerParams::kInputCctfDecoding);
        _pOutputColorSpace = fetchChoiceParam(kParamOutputColorSpace);
        _pOutputCctfEncoding = fetchBooleanParam(kParamOutputCctfEncoding);
        _pOutputLinearPassThrough = fetchBooleanParam(kParamOutputLinearPassThrough);


#ifdef JUICER_ENABLE_COUPLERS
        _pCouplersActive = fetchBooleanParam(Couplers::kParamCouplersActive);
        _pCouplersAmount = fetchDoubleParam(Couplers::kParamCouplersAmount);
        _pCouplersAmountR = fetchDoubleParam(Couplers::kParamCouplersAmountR);
        _pCouplersAmountG = fetchDoubleParam(Couplers::kParamCouplersAmountG);
        _pCouplersAmountB = fetchDoubleParam(Couplers::kParamCouplersAmountB);
        _pCouplersSigma = fetchDoubleParam(Couplers::kParamCouplersLayerSigma);
        _pCouplersHigh = fetchDoubleParam(Couplers::kParamCouplersHighExpShift);
        _pCouplersSpatialSigma = fetchDoubleParam(Couplers::kParamCouplersSpatialSigma);
#endif

        _pScanEnabled = fetchBooleanParam("ScannerEnabled");
        _pScanAuto = fetchBooleanParam("ScannerAutoExposure");
        _pScanTargetY = fetchDoubleParam("ScannerTargetY");
        _pScanFilmLongEdge = fetchDoubleParam(JuicerParams::kScannerFilmLongEdgeMm);

        _pPrintBypass = fetchBooleanParam("PrintBypass");
        _pPrintExposure = fetchDoubleParam("PrintExposure");
        _pPrintPreflash = fetchDoubleParam("PrintPreflash");
        _pPrintExposureComp = fetchBooleanParam("PrintExposureCompensation");
        _pEnlargerY = fetchDoubleParam("EnlargerY");
        _pEnlargerM = fetchDoubleParam("EnlargerM");
    }
    catch (...) {
        // Safe: any missing param will remain nullptr and defaults are used in snapshot/usage paths.
    }

    // Own per-instance state
    _state = std::make_unique<InstanceState>();
    _state->dataDir = ensure_trailing_separator(data_dir_string());
    _state->activeWS.store(&_state->workA, std::memory_order_release);
    _state->activeBuildCounter = 0;

    // Optional compatibility registry
    JuicerRegistry::set(handle, _state.get());

    // Defer heavy bootstrap until first param change
}

JuicerEffect::~JuicerEffect() {
    // Mirror destroyInstance() guards without touching C suites.
    JuicerRegistry::erase(this->getHandle());
    _state.reset();
}

void JuicerEffect::render(const OFX::RenderArguments& args) {
    // Fetch images via wrappers
    std::unique_ptr<OFX::Image> srcImg(_src ? _src->fetchImage(args.time) : nullptr);
    std::unique_ptr<OFX::Image> dstImg(_dst ? _dst->fetchImage(args.time) : nullptr);
    if (!srcImg || !dstImg) return;

    // Components and depth
    const OFX::PixelComponentEnum comps = srcImg->getPixelComponents();
    const OFX::BitDepthEnum depth = srcImg->getPixelDepth();
    if (depth != OFX::eBitDepthFloat) {
        JuicerProc::copyNonFloatRect(srcImg.get(), dstImg.get());
        return;
    }

    const int nComponents =
        (comps == OFX::ePixelComponentRGBA) ? 4 :
        (comps == OFX::ePixelComponentRGB) ? 3 :
        (comps == OFX::ePixelComponentAlpha) ? 1 : 0;

    if (nComponents == 0) {
        JuicerProc::copyNonFloatRect(srcImg.get(), dstImg.get());
        return;
    }

    const OfxRectI fullBounds = srcImg->getBounds();

    // ROI: args.renderWindow if provided; otherwise use image bounds
    OfxRectI roi = args.renderWindow;
    if (roi.x1 == roi.x2 && roi.y1 == roi.y2) {
        roi = fullBounds;
    }
    const int width = roi.x2 - roi.x1;
    const int height = roi.y2 - roi.y1;
    if (width <= 0 || height <= 0) return;
    const int fullWidth = fullBounds.x2 - fullBounds.x1;
    const int fullHeight = fullBounds.y2 - fullBounds.y1;

    // Ensure bootstrap has run before we rely on parameter state
    if (_state && !_state->baseLoaded) {
        if (!_state->inBootstrap) {
            struct SuppressGuard {
                InstanceState* state;
                bool previous;
                explicit SuppressGuard(InstanceState* s)
                    : state(s), previous(s ? s->suppressParamEvents : false) {
                    if (state) {
                        state->suppressParamEvents = true;
                    }
                }
                ~SuppressGuard() {
                    if (state) {
                        state->suppressParamEvents = previous;
                    }
                }
            } guard(_state.get());
            bootstrap_after_attach();
        }
    }
    const ExposureParams exposureParams = gatherExposureParams();
    Scanner::Params scannerParams = gatherScannerParams();
    Print::Params printParams = gatherPrintParams();
    OutputEncoding::Params outputEncodingParams = gatherOutputEncodingParams();

    const AutoExposureResult autoExposure = computeAutoExposure(
        args,
        srcImg.get(),
        fullBounds,
        scannerParams,
        exposureParams);

#ifdef JUICER_ENABLE_COUPLERS
    Couplers::Runtime dirRT = prepareCouplers(args, fullWidth, fullHeight);
#else
    Couplers::Runtime dirRT{};
#endif

    WorkingStateInfo wsInfo = prepareWorkingState();
    WorkingState* activeWorkingState = wsInfo.activeWorkingState;
    const WorkingState* ws = wsInfo.workingState;
    const Print::Runtime* prt = wsInfo.printRuntime;
    const bool wsReady = wsInfo.workingStateReady;
    const bool printReady = wsInfo.printRuntimeReady;
    if (!printReady) {
        if (prt && !Print::profile_is_valid(prt->profile)) {
            JTRACE("PRINT", "profile invalid: missing print sensitivities or curves; bypassing print path");
        }
        else if (prt) {
            JTRACE("PRINT", "print runtime invalid: illuminants not pinned or working tables not ready; bypassing print path");
        }
    }

    // --- Print exposure compensation via spectral mid-gray probe (agx parity) ---
    {
        bool printComp = false;
        if (_pPrintExposureComp) { bool pc = false; _pPrintExposureComp->getValue(pc); printComp = pc; }

        printParams.exposureCompensationEnabled = printComp;
        printParams.exposureCompensationScale = printComp ? exposureParams.sliderScale : 1.0f;
    }

    // Tile-based multithreaded processing via OFX::ImageProcessor
    JuicerProcessor proc(*this);
    proc.setSrcDst(srcImg.get(), dstImg.get());
    proc.setComponents(nComponents);
    proc.setScannerParams(scannerParams);
    proc.setPrintParams(printParams);
    proc.setDirRuntime(dirRT);
    proc.setWorkingState(ws, wsReady);
    proc.setPrintRuntime(prt, printReady);
    float filmExposureScale = autoExposure.exposureScale;
    // Per agx-emulsion parity: autoExposure.exposureScale already encodes 2^(autoEV + sliderEV).
    if (!std::isfinite(filmExposureScale) || filmExposureScale <= 0.0f) {
        filmExposureScale = 1.0f;
    }
    proc.setExposure(filmExposureScale);
    proc.setOutputEncoding(outputEncodingParams);
    proc.setRenderWindowRect(roi);
    proc.setGPURenderArgs(args);

    struct RenderGuard {
        InstanceState* state;
        WorkingState* ws;
        RenderGuard(InstanceState* s, WorkingState* w) : state(s), ws(w) {
            if (state) {
                state->rendersInFlight.fetch_add(1, std::memory_order_acq_rel);
                state->renderWS.store(w, std::memory_order_release);
            }
        }
        ~RenderGuard() {
            if (state) {
                const int prev = state->rendersInFlight.fetch_sub(1, std::memory_order_acq_rel);
                if (prev <= 1) {
                    WorkingState* expected = ws;
                    state->renderWS.compare_exchange_strong(
                        expected, nullptr, std::memory_order_acq_rel, std::memory_order_acquire);
                }
                state->renderCv.notify_all();
            }
        }
    } guard(_state.get(), activeWorkingState);

    // Dispatch to support library's threaded/tiled CPU path
    proc.process();
}

void JuicerEffect::changedParam(const OFX::InstanceChangedArgs&, const std::string& paramName) {
    // Suppress recursion while we are programmatically setting params
    if (_state && _state->suppressParamEvents) {
        JTRACE("BUILD", std::string("changedParam suppressed for '") + paramName + "'");
        return;
    }
    if (_state && _state->inBootstrap) {
        JTRACE("BUILD", std::string("changedParam ignored during bootstrap for '") + paramName + "'");
        return;
    }
    if (_state && paramName == kParamCameraAutoExposure) {
        std::lock_guard<std::mutex> cacheLock(_state->autoExposureMutex);
        _state->autoExposureCacheValid = false;
    }
    onParamsPossiblyChanged(paramName.c_str());
}

ParamSnapshot JuicerEffect::snapshotParams() const {
    ParamSnapshot P;
    if (_pFilmStock)      _pFilmStock->getValue(P.filmStockIndex);
    if (_pPrintPaper)     _pPrintPaper->getValue(P.printPaperIndex);
    if (_pRefIll)         _pRefIll->getValue(P.refIll);
    if (_pViewIll)        _pViewIll->getValue(P.viewIll);
    if (_pEnlIll)         _pEnlIll->getValue(P.enlIll);
    if (_pInputColorSpace) _pInputColorSpace->getValue(P.inputColorSpace);
    if (_pInputCctfDecoding) { bool v = false; _pInputCctfDecoding->getValue(v); P.inputCctfDecoding = v ? 1 : 0; }
#ifdef JUICER_ENABLE_COUPLERS
    if (_pCouplersActive) { bool v = true; _pCouplersActive->getValue(v); P.couplersActive = v ? 1 : 0; }
    if (_pCouplersAmount)  _pCouplersAmount->getValue(P.couplersAmount);
    if (_pCouplersAmountR) _pCouplersAmountR->getValue(P.ratioR);
    if (_pCouplersAmountG) _pCouplersAmountG->getValue(P.ratioG);
    if (_pCouplersAmountB) _pCouplersAmountB->getValue(P.ratioB);
    if (_pCouplersSigma)      _pCouplersSigma->getValue(P.sigma);
    if (_pCouplersHigh)       _pCouplersHigh->getValue(P.high);
    double spatial = 0.0;
    if (_pCouplersSpatialSigma) _pCouplersSpatialSigma->getValue(spatial);
    P.spatialSigmaMicrometers = spatial;
#endif
    return P;
}

void JuicerEffect::bootstrap_after_attach() {
    // Initialize Spectral globals exactly once per process.
    std::call_once(gSpectralGlobalsOnce, init_spectral_globals_once);
    JTRACE("BUILD", "spectral globals ensured once; proceeding to profile and film stock load");
    // Suppress re-entrant param events during bootstrap
    _state->inBootstrap = true;
    _state->suppressParamEvents = true;

    _state->printRT = Print::Runtime{};
    ParamSnapshot P = snapshotParams();

    // Load selected print paper profile
    const std::string printDir = print_dir_for_index(P.printPaperIndex);
    const char* paperKey = print_paper_json_key_for_index(P.printPaperIndex);
    const std::string printProfileJson = paperKey
        ? data_dir_string("profiles", std::string(paperKey) + ".json")
        : std::string();
    Print::load_profile_from_dir(printDir, _state->printRT.profile, printProfileJson, &_state->printRT);
    _state->printRT.hasMidNeutralDensity = _state->printRT.profile.hasMidNeutralDensity;
    _state->printRT.midNeutralDensity = std::move(_state->printRT.profile.midNeutralDensity);
    _state->printRT.hasMidNeutralLogE = _state->printRT.profile.hasMidNeutralLogE;
    _state->printRT.midNeutralLogE = std::move(_state->printRT.profile.midNeutralLogE);

    // Load film stock before applying metadata-driven illuminant defaults
    _state->baseLoaded = load_film_stock_into_base(P.filmStockIndex, *_state);

    // Apply metadata-driven illuminant defaults and rebuild runtime illuminants
    applyMetadataIlluminantDefaults(P);
    Print::build_illuminant_from_choice(P.enlIll, _state->printRT, _state->dataDir, /*forEnlarger*/true);
    Print::build_illuminant_from_choice(P.viewIll, _state->printRT, _state->dataDir, /*forEnlarger*/false);

    // Load dichroic filters (Durst Digital Light by default)
    const std::string durstDir = ensure_trailing_separator(
        data_dir_string("filters", "dichroics", "durst_digital_light"));
    try {
        Print::load_dichroic_filters_from_csvs(durstDir, _state->printRT);
    }
    catch (const std::exception& ex) {
        // Identity fallback is already handled in loader via 1.0 curves
        JTRACE("PRINT", std::string("dichroic load failed at '") + durstDir + "' (" + ex.what() + "); using identity filters");
    }
    catch (...) {
        JTRACE("PRINT", std::string("dichroic load failed at '") + durstDir + "' (unknown error); using identity filters");
    }

    applyNeutralFilters(P, /*resetFilterParams*/true, /*ensureExposureComp*/true);


    if (_state->baseLoaded) {
#ifdef JUICER_ENABLE_COUPLERS
        applyCouplerProfileDefaults(P);
#endif
        rebuild_working_state(this->getHandle(), *_state, P);
        _state->lastParams = P;
        _state->lastHash = hash_params(P);
    }
    else {
        JTRACE("STOCK", "bootstrap: failed to load film stock; deferring rebuild");
    }

    // Re-enable changedParam handling now that bootstrap is complete
    _state->suppressParamEvents = false;
    _state->inBootstrap = false;
}

void JuicerEffect::applyNeutralFilters(const ParamSnapshot& P, bool resetFilterParams, bool ensureExposureComp) {
    if (!_state) {
        return;
    }

    const char* paperKey = print_paper_json_key_for_index(P.printPaperIndex);
    const char* negativeKey = negative_json_key_for_stock_index(P.filmStockIndex);
    const std::vector<std::string> illumKeys = enlarger_illuminant_keys_for_choice(P.enlIll);

    auto join_illum_keys = [&]() -> std::string {
        std::string combined;
        for (const std::string& key : illumKeys) {
            if (!combined.empty()) {
                combined += ",";
            }
            combined += key;
        }
        if (combined.empty()) {
            combined = "<none>";
        }
        return combined;
        };

    if (!(paperKey && negativeKey && !illumKeys.empty())) {
        const std::string paperStr = paperKey ? paperKey : "<unset>";
        const std::string negStr = negativeKey ? negativeKey : "<unset>";
        JTRACE("PRINT", "Neutral filter lookup prerequisites missing: paper="
            + paperStr + " negative=" + negStr + " illum_choices=" + join_illum_keys());
        throw std::runtime_error("Neutral filter metadata incomplete for current selection");
    }

    float neutralY = Print::kDefaultNeutralY;
    float neutralM = Print::kDefaultNeutralM;
    float neutralC = Print::kDefaultNeutralC;
    bool loaded = false;

    const std::string jsonPath = data_dir_string("profiles", "enlarger_neutral_ymc_filters.json");
    std::tuple<float, float, float> ymc{};
    for (const std::string& illumKey : illumKeys) {
        if (illumKey.empty()) {
            continue;
        }
        if (load_enlarger_neutral_filters(jsonPath, paperKey, illumKey, negativeKey, ymc)) {
            neutralY = std::clamp(std::get<0>(ymc), 0.0f, 1.0f);
            neutralM = std::clamp(std::get<1>(ymc), 0.0f, 1.0f);
            neutralC = std::clamp(std::get<2>(ymc), 0.0f, 1.0f);
            loaded = true;
            JTRACE("PRINT", "Neutral filters loaded for " + std::string(illumKey)
                + " Y/M/C=" + std::to_string(neutralY) + "/" + std::to_string(neutralM)
                + "/" + std::to_string(neutralC));
            break;
        }
    }

    if (!loaded) {
        const std::string paperStr = paperKey ? paperKey : "<unset>";
        const std::string negStr = negativeKey ? negativeKey : "<unset>";
        JTRACE("PRINT", "Neutral filters missing for paper=" + paperStr
            + " illuminant_keys=" + join_illum_keys()
            + " negative=" + negStr + "; aborting print path");
        throw std::runtime_error("Neutral filter database entry not found");
    }

    _state->printRT.neutralY = neutralY;
    _state->printRT.neutralM = neutralM;
    _state->printRT.neutralC = neutralC;

    if (resetFilterParams) {
        const bool wasSuppressed = _state->suppressParamEvents;
        _state->suppressParamEvents = true;
        if (_pEnlargerY) _pEnlargerY->setValue(0.0);
        if (_pEnlargerM) _pEnlargerM->setValue(0.0);
        _state->suppressParamEvents = wasSuppressed;
    }

    if (ensureExposureComp && _pPrintExposureComp) {
        bool exposureToggle = false;
        _pPrintExposureComp->getValue(exposureToggle);
        if (!exposureToggle) {
            const bool wasSuppressed = _state->suppressParamEvents;
            _state->suppressParamEvents = true;
            _pPrintExposureComp->setValue(true);
            _state->suppressParamEvents = wasSuppressed;
        }
    }
}

bool JuicerEffect::applyMetadataIlluminantDefaults(ParamSnapshot& P) {
    if (!_state) {
        return false;
    }

    bool changed = false;

    const std::string& filmRef = !_state->filmReferenceIlluminant.empty()
        ? _state->filmReferenceIlluminant
        : _state->base.referenceIlluminant;
    const std::string& filmView = !_state->filmViewingIlluminant.empty()
        ? _state->filmViewingIlluminant
        : _state->base.viewingIlluminant;
    const std::string& printRef = _state->printRT.referenceIlluminant;
    const std::string& printView = _state->printRT.viewingIlluminant;

    std::string refSource = !filmRef.empty() ? filmRef : (!printRef.empty() ? printRef : printView);
    std::string enlSource = !printRef.empty() ? printRef : (!filmRef.empty() ? filmRef : printView);
    std::string viewSource = !printView.empty() ? printView
        : (!filmView.empty() ? filmView
            : (!refSource.empty() ? refSource : enlSource));

    const bool wasSuppressed = _state->suppressParamEvents;
    _state->suppressParamEvents = true;

    auto tryApply = [&](OFX::ChoiceParam* param, int& currentIndex, bool overrideFlag, const std::string& source) {
        if (overrideFlag || !param) {
            return;
        }
        const int mapped = illuminant_choice_index_from_string(source);
        if (mapped < 0 || currentIndex == mapped) {
            return;
        }
        param->setValue(mapped);
        currentIndex = mapped;
        changed = true;
        };

    tryApply(_pRefIll, P.refIll, _state->illuminantOverride.reference, refSource);
    tryApply(_pEnlIll, P.enlIll, _state->illuminantOverride.enlarger, enlSource);
    tryApply(_pViewIll, P.viewIll, _state->illuminantOverride.viewing, viewSource);

    _state->suppressParamEvents = wasSuppressed;

    if (changed) {
        P = snapshotParams();
    }

    return changed;
}

#ifdef JUICER_ENABLE_COUPLERS
void JuicerEffect::applyCouplerProfileDefaults(ParamSnapshot& P) {
    if (!_state) {
        return;
    }

    const Profiles::DirCouplersProfile& dirCfg = _state->base.dirCouplers;
    if (!dirCfg.hasData) {
        return;
    }

    auto sanitize_range = [](float value, double fallback, double lo, double hi) -> double {
        double v = static_cast<double>(value);
        if (!std::isfinite(v)) {
            return fallback;
        }
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        return v;
        };

    const bool wasSuppressed = _state->suppressParamEvents;
    _state->suppressParamEvents = true;

    if (!_state->couplerDirty.active) {
        const bool active = dirCfg.active;
        if (_pCouplersActive) {
            _pCouplersActive->setValue(active);
        }
        P.couplersActive = active ? 1 : 0;
    }

    if (!_state->couplerDirty.amount) {
        const double amount = sanitize_range(dirCfg.amount, P.couplersAmount, 0.0, 2.0);
        if (_pCouplersAmount) {
            _pCouplersAmount->setValue(amount);
        }
        P.couplersAmount = amount;
    }

    if (!_state->couplerDirty.ratioB) {
        const double ratioB = sanitize_range(dirCfg.ratioRGB[0], P.ratioB, 0.0, 1.0);
        if (_pCouplersAmountB) {
            _pCouplersAmountB->setValue(ratioB);
        }
        P.ratioB = ratioB;
    }

    if (!_state->couplerDirty.ratioG) {
        const double ratioG = sanitize_range(dirCfg.ratioRGB[1], P.ratioG, 0.0, 1.0);
        if (_pCouplersAmountG) {
            _pCouplersAmountG->setValue(ratioG);
        }
        P.ratioG = ratioG;
    }

    if (!_state->couplerDirty.ratioR) {
        const double ratioR = sanitize_range(dirCfg.ratioRGB[2], P.ratioR, 0.0, 1.0);
        if (_pCouplersAmountR) {
            _pCouplersAmountR->setValue(ratioR);
        }
        P.ratioR = ratioR;
    }

    if (!_state->couplerDirty.sigma) {
        const double sigma = sanitize_range(dirCfg.diffusionInterlayer, P.sigma, 0.0, 3.0);
        if (_pCouplersSigma) {
            _pCouplersSigma->setValue(sigma);
        }
        P.sigma = sigma;
    }

    if (!_state->couplerDirty.high) {
        const double high = sanitize_range(dirCfg.highExposureShift, P.high, 0.0, 1.0);
        if (_pCouplersHigh) {
            _pCouplersHigh->setValue(high);
        }
        P.high = high;
    }

    if (!_state->couplerDirty.spatialSigma) {
        const float profileSpatialSigma = _state->couplerProfileSpatialSigmaValid
            ? static_cast<float>(_state->couplerProfileSpatialSigmaMicrometers)
            : dirCfg.diffusionSizeUm;
        const double spatial = sanitize_range(profileSpatialSigma, P.spatialSigmaMicrometers, 0.0, 50.0);
        if (_pCouplersSpatialSigma) {
            _pCouplersSpatialSigma->setValue(spatial);
        }
        P.spatialSigmaMicrometers = spatial;
    }

    _state->suppressParamEvents = wasSuppressed;
}
#endif

void JuicerEffect::onParamsPossiblyChanged(const char* changedNameOrNull) {
    if (!_state) return;
    // Suppress re-entrant param handling while programmatic changes are in flight
    if (_state->suppressParamEvents) {
        JTRACE("BUILD", "onParamsPossiblyChanged suppressed");
        return;
    }

    // If bootstrap hasn’t run yet, run it once now
    if (_state->lastHash == 0 && !_state->baseLoaded) {
        bootstrap_after_attach();
    }

    ParamSnapshot P = snapshotParams();
#ifdef JUICER_ENABLE_COUPLERS
    if (changedNameOrNull) {
        if (std::strcmp(changedNameOrNull, Couplers::kParamCouplersActive) == 0) {
            _state->couplerDirty.active = true;
        }
        else if (std::strcmp(changedNameOrNull, Couplers::kParamCouplersAmount) == 0) {
            _state->couplerDirty.amount = true;
        }
        else if (std::strcmp(changedNameOrNull, Couplers::kParamCouplersAmountB) == 0) {
            _state->couplerDirty.ratioB = true;
        }
        else if (std::strcmp(changedNameOrNull, Couplers::kParamCouplersAmountG) == 0) {
            _state->couplerDirty.ratioG = true;
        }
        else if (std::strcmp(changedNameOrNull, Couplers::kParamCouplersAmountR) == 0) {
            _state->couplerDirty.ratioR = true;
        }
        else if (std::strcmp(changedNameOrNull, Couplers::kParamCouplersLayerSigma) == 0) {
            _state->couplerDirty.sigma = true;
        }
        else if (std::strcmp(changedNameOrNull, Couplers::kParamCouplersHighExpShift) == 0) {
            _state->couplerDirty.high = true;
        }
        else if (std::strcmp(changedNameOrNull, Couplers::kParamCouplersSpatialSigma) == 0) {
            _state->couplerDirty.spatialSigma = true;
        }
    }
#endif

    // Track user overrides for illuminant choices
    if (changedNameOrNull) {
        if (std::strcmp(changedNameOrNull, kParamReferenceIlluminant) == 0) {
            _state->illuminantOverride.reference = true;
        }
        else if (std::strcmp(changedNameOrNull, kParamEnlargerIlluminant) == 0) {
            _state->illuminantOverride.enlarger = true;
        }
        else if (std::strcmp(changedNameOrNull, kParamViewingIllum) == 0) {
            _state->illuminantOverride.viewing = true;
        }
    }

    bool printReloaded = false;
    if (changedNameOrNull && std::strcmp(changedNameOrNull, kParamPrintPaper) == 0) {
        const std::string printDir = print_dir_for_index(P.printPaperIndex);
        const char* paperKey = print_paper_json_key_for_index(P.printPaperIndex);
        const std::string printProfileJson = paperKey
            ? data_dir_string("profiles", std::string(paperKey) + ".json")
            : std::string();
        Print::load_profile_from_dir(printDir, _state->printRT.profile, printProfileJson, &_state->printRT);
        _state->printRT.hasMidNeutralDensity = _state->printRT.profile.hasMidNeutralDensity;
        _state->printRT.midNeutralDensity = std::move(_state->printRT.profile.midNeutralDensity);
        _state->printRT.hasMidNeutralLogE = _state->printRT.profile.hasMidNeutralLogE;
        _state->printRT.midNeutralLogE = std::move(_state->printRT.profile.midNeutralLogE);

        // Reload dichroic filters (keep vendor default; can be parameterized later)
        const std::string durstDirReload = ensure_trailing_separator(
            data_dir_string("filters", "dichroics", "durst_digital_light"));
        try {
            Print::load_dichroic_filters_from_csvs(durstDirReload, _state->printRT);
        }
        catch (const std::exception& ex) {
            JTRACE("PRINT", std::string("dichroic reload failed at '") + durstDirReload + "' (" + ex.what() + "); identity filters remain active");
        }
        catch (...) {
            JTRACE("PRINT", std::string("dichroic reload failed at '") + durstDirReload + "' (unknown error); identity filters remain active");
        }

        printReloaded = true;
    }

    bool filmReloaded = false;
    if (changedNameOrNull && std::strcmp(changedNameOrNull, kParamFilmStock) == 0) {
        _state->baseLoaded = load_film_stock_into_base(P.filmStockIndex, *_state);
        filmReloaded = _state->baseLoaded;
    }
    else if (P.filmStockIndex != _state->lastParams.filmStockIndex) {
        _state->baseLoaded = load_film_stock_into_base(P.filmStockIndex, *_state);
        filmReloaded = _state->baseLoaded;
    }

    if (printReloaded || filmReloaded) {
        applyMetadataIlluminantDefaults(P);
        Print::build_illuminant_from_choice(P.enlIll, _state->printRT, _state->dataDir, /*forEnlarger*/true);
        Print::build_illuminant_from_choice(P.viewIll, _state->printRT, _state->dataDir, /*forEnlarger*/false);
    }

    bool neutralApplied = false;
    if (printReloaded) {
        applyNeutralFilters(P, /*resetFilterParams*/true, /*ensureExposureComp*/false);
        neutralApplied = true;
    }
    if (filmReloaded && !neutralApplied) {
        applyNeutralFilters(P, /*resetFilterParams*/true, /*ensureExposureComp*/false);
        neutralApplied = true;
    }

    if (printReloaded && _state->baseLoaded) {
        // Force working state snapshot to carry updated printRT profile into render
        rebuild_working_state(this->getHandle(), *_state, P);
        _state->lastParams = P;
        _state->lastHash = hash_params(P);
    }

    if (changedNameOrNull && std::strcmp(changedNameOrNull, kParamEnlargerIlluminant) == 0) {
        applyNeutralFilters(P, /*resetFilterParams*/true, /*ensureExposureComp*/false);
    }

    // Rebuild if any effective param changed
    if (_state->baseLoaded) {
#ifdef JUICER_ENABLE_COUPLERS
        applyCouplerProfileDefaults(P);
#endif
    }

    const uint64_t h = hash_params(P);
    if (_state->baseLoaded && h != _state->lastHash) {
        rebuild_working_state(this->getHandle(), *_state, P);
        _state->lastParams = P;
        _state->lastHash = h;
    }

#ifdef JUICER_ENABLE_COUPLERS
    if (changedNameOrNull) {
        using namespace Couplers;
        const bool isCouplerParam =
            std::strcmp(changedNameOrNull, kParamCouplersActive) == 0 ||
            std::strcmp(changedNameOrNull, kParamCouplersAmount) == 0 ||
            std::strcmp(changedNameOrNull, kParamCouplersAmountR) == 0 ||
            std::strcmp(changedNameOrNull, kParamCouplersAmountG) == 0 ||
            std::strcmp(changedNameOrNull, kParamCouplersAmountB) == 0 ||
            std::strcmp(changedNameOrNull, kParamCouplersLayerSigma) == 0 ||
            std::strcmp(changedNameOrNull, kParamCouplersHighExpShift) == 0 ||
            std::strcmp(changedNameOrNull, kParamCouplersSpatialSigma) == 0;

        if (isCouplerParam) {
            // Avoid cascading rebuild loops: mark mixing dirty only if hash actually changes.
            ParamSnapshot Pnew = snapshotParams();
            const uint64_t hnew = hash_params(Pnew);
            if (hnew != _state->lastHash) {
                Couplers::on_param_changed(changedNameOrNull);
            }
        }
    }
#endif
}
