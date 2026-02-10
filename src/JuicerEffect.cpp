#include "JuicerEffect.h"

#include <algorithm>
#include <cstddef>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <atomic>
#include <limits>
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
#include "Hash.h"
#include "mainProcessing.h"

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
#include "Cuda/JuicerCudaAutoExposure.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#endif

namespace {
    static std::once_flag gSpectralGlobalsOnce;

    enum class MeteringMethod : int {
        CenterWeighted = 0,
        Median = 1
    };

    const char* dichroic_dir_name_for_choice(int choice) {
        switch (choice) {
        case 1: return "thorlabs";
        case 2: return "edmund_optics";
        case 0:
        default: return "durst_digital_light";
        }
    }

    const char* enlarger_neutral_filters_json_for_choice(int choice) {
        switch (choice) {
        case 1: return "enlarger_neutral_ymc_filters_thorlabs.json";
        case 2: return "enlarger_neutral_ymc_filters_edmund.json";
        case 0:
        default: return "enlarger_neutral_ymc_filters.json";
        }
    }

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

        const size_t expectedMaskSize = static_cast<size_t>(width) * static_cast<size_t>(height);
        auto needsMaskRebuild = [&](const InstanceState& s) -> bool {
            const std::shared_ptr<const std::vector<double>>& weights = s.autoExposureMaskWeights;
            return !s.autoExposureMaskValid
                || s.autoExposureMaskWidth != width
                || s.autoExposureMaskHeight != height
                || !nearly_equal_double(s.autoExposureMaskSigma, sigma)
                || !nearly_equal_double(s.autoExposureMaskRenderScaleX, renderScaleX)
                || !nearly_equal_double(s.autoExposureMaskRenderScaleY, renderScaleY)
                || s.autoExposureMaskClipToken != clipToken
                || !weights
                || weights->size() != expectedMaskSize;
            };

        std::shared_ptr<const std::vector<double>> maskSnapshot;
        bool maskValid = false;
        bool rebuildMask = false;
        {
            std::lock_guard<std::mutex> lock(state->autoExposureMutex);
            rebuildMask = needsMaskRebuild(*state);
            if (!rebuildMask) {
                maskSnapshot = state->autoExposureMaskWeights;
                maskValid = state->autoExposureMaskValid && static_cast<bool>(maskSnapshot);
                if (!maskValid) {
                    state->autoExposureMaskSum = 0.0;
                }
            }
        }

        if (rebuildMask) {
            auto rebuiltMask = std::make_shared<std::vector<double>>();
            const double sumMask = build_center_weight_mask(width, height, sigma, *rebuiltMask);
            const bool rebuiltValid = sumMask > 0.0;

            std::lock_guard<std::mutex> lock(state->autoExposureMutex);
            if (needsMaskRebuild(*state)) {
                state->autoExposureMaskWidth = width;
                state->autoExposureMaskHeight = height;
                state->autoExposureMaskSigma = sigma;
                state->autoExposureMaskRenderScaleX = renderScaleX;
                state->autoExposureMaskRenderScaleY = renderScaleY;
                state->autoExposureMaskClipToken = clipToken;
                state->autoExposureMaskWeights = rebuiltMask;
                state->autoExposureMaskValid = rebuiltValid;
                state->autoExposureMaskSum = rebuiltValid ? sumMask : 0.0;
            }
            maskSnapshot = state->autoExposureMaskWeights;
            maskValid = state->autoExposureMaskValid && static_cast<bool>(maskSnapshot);
            if (!maskValid) {
                state->autoExposureMaskSum = 0.0;
            }
        }

        if (!maskValid || !maskSnapshot || maskSnapshot->size() != expectedMaskSize) {
            return 0.0;
        }

        double effectiveSumMask = 0.0;
        const double measuredY = accumulateY(*maskSnapshot, &effectiveSumMask);
        {
            std::lock_guard<std::mutex> lock(state->autoExposureMutex);
            if (state->autoExposureMaskWeights == maskSnapshot) {
                state->autoExposureMaskSum = effectiveSumMask;
            }
        }
        return measuredY;
    }

    static double measure_median_Y_DWG(
        OFX::Image* img,
        const OfxRectI& bounds,
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

        const size_t total = static_cast<size_t>(width) * static_cast<size_t>(height);
        std::vector<float> values;
        values.reserve(total);

        for (int yy = bounds.y1; yy < bounds.y2; ++yy) {
            for (int xx = bounds.x1; xx < bounds.x2; ++xx) {
                const float* pix = reinterpret_cast<const float*>(img->getPixelAddress(xx, yy));
                if (!pix) {
                    continue;
                }
                float linear[3];
                Spectral::apply_input_cctf_decoding(inputColorSpace, applyCctfDecoding, pix, linear);
                float XYZ[3];
                rgbToXYZ.mul(linear, XYZ);
                float Y = XYZ[1];
                if (!std::isfinite(Y)) {
                    continue;
                }
                if (Y < 0.0f) {
                    Y = 0.0f;
                }
                values.push_back(Y);
            }
        }

        if (values.empty()) {
            return 0.0;
        }

        const size_t n = values.size();
        const size_t mid = n / 2;
        auto midIt = values.begin() + static_cast<std::ptrdiff_t>(mid);
        std::nth_element(values.begin(), midIt, values.end());
        const float high = *midIt;
        if ((n & 1U) != 0U) {
            return static_cast<double>(high);
        }

        auto lowIt = values.begin() + static_cast<std::ptrdiff_t>(mid - 1);
        std::nth_element(values.begin(), lowIt, values.end());
        const float low = *lowIt;
        return static_cast<double>((low + high) * 0.5f);
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

        // Mallett 2019 basis: load once for sRGB basis reconstruction.
        try {
            const std::string basisPath = data_dir_string("luts", "spectral_upsampling", "mallett2019_basis.npy");
            Spectral::load_mallett2019_basis_npy(basisPath);
        }
        catch (...) {
            Spectral::set_mallett_available(false);
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
    int meteringMethod = 0;
    if (_pCameraMeteringMethod) {
        _pCameraMeteringMethod->getValue(meteringMethod);
    }
    params.meteringMethod = meteringMethod;
    return params;
}

Scanner::Options JuicerEffect::gatherScannerOptions() const {
    Scanner::Options opts{};
    double blurSigma = opts.lensBlurSigmaPx;
    if (_pScannerLensBlur) {
        _pScannerLensBlur->getValue(blurSigma);
    }
    if (!std::isfinite(blurSigma)) {
        blurSigma = 0.55;
    }
    blurSigma = std::clamp(blurSigma, 0.0, 10.0);
    opts.lensBlurSigmaPx = static_cast<float>(blurSigma);

    double unsharpSigma = opts.unsharpSigmaPx;
    double unsharpAmount = opts.unsharpAmount;
    if (_pScannerUnsharp) {
        _pScannerUnsharp->getValue(unsharpSigma, unsharpAmount);
    }
    if (!std::isfinite(unsharpSigma)) {
        unsharpSigma = 0.7;
    }
    if (!std::isfinite(unsharpAmount)) {
        unsharpAmount = 1.0;
    }
    unsharpSigma = std::clamp(unsharpSigma, 0.0, 5.0);
    unsharpAmount = std::clamp(unsharpAmount, 0.0, 3.0);
    opts.unsharpSigmaPx = static_cast<float>(unsharpSigma);
    opts.unsharpAmount = static_cast<float>(unsharpAmount);
    return opts;
}

Scanner::Settings JuicerEffect::gatherScannerSettings() const {
    Scanner::Settings settings{};
    bool useLut = settings.useLut;
    if (_pScannerUseLut) {
        _pScannerUseLut->getValue(useLut);
    }
    settings.useLut = useLut;
    int lutRes = static_cast<int>(settings.lutResolution);
    if (_pScannerLutResolution) {
        _pScannerLutResolution->getValue(lutRes);
    }
    if (lutRes < 17 || lutRes > 128) {
        lutRes = 17;
    }
    settings.lutResolution = static_cast<std::uint32_t>(lutRes);
    return settings;
}

Print::Params JuicerEffect::gatherPrintParams() const {
    Print::Params params{};
    bool bypass = true;
    double pexp = 1.0;
    double preflash = 0.0;
    double y = 0.0;
    double m = 0.0;
    double c = 0.0;
    if (_pPrintBypass) _pPrintBypass->getValue(bypass);
    if (_pPrintExposure) _pPrintExposure->getValue(pexp);
    if (_pPrintPreflash) _pPrintPreflash->getValue(preflash);
    if (_pEnlargerY) _pEnlargerY->getValue(y);
    if (_pEnlargerM) _pEnlargerM->getValue(m);
    if (_pEnlargerC) _pEnlargerC->getValue(c);
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
    params.cFilter = static_cast<float>(clampShift(c));
    return params;
}

Profiles::HalationMetadata JuicerEffect::gatherHalationUi() const {
    Profiles::HalationMetadata halation{};

    bool active = false;
    if (_pHalationActive) {
        _pHalationActive->getValue(active);
    }
    halation.active = active;

    auto sanitize = [](double value, double fallback, double minValue, double maxValue) -> double {
        if (!std::isfinite(value)) return fallback;
        return std::clamp(value, minValue, maxValue);
    };
    auto read3 = [&](OFX::Double3DParam* param, const std::array<double, 3>& defaults, double minValue, double maxValue) {
        std::array<double, 3> values = defaults;
        if (param) {
            param->getValue(values[0], values[1], values[2]);
        }
        for (int i = 0; i < 3; ++i) {
            values[i] = sanitize(values[i], defaults[i], minValue, maxValue);
        }
        return values;
    };

    const std::array<double, 3> strengthPercent = read3(_pHalationStrength, { {3.0, 0.30, 0.10} }, 0.0, 100.0);
    const std::array<double, 3> sizeUm = read3(_pHalationSizeUm, { {200.0, 200.0, 200.0} }, 0.0, 1000.0);
    const std::array<double, 3> scatterStrengthPercent = read3(_pHalationScatteringStrength, { {1.0, 2.0, 4.0} }, 0.0, 100.0);
    const std::array<double, 3> scatterSizeUm = read3(_pHalationScatteringSizeUm, { {30.0, 20.0, 15.0} }, 0.0, 1000.0);

    for (int i = 0; i < 3; ++i) {
        halation.strength[i] = static_cast<float>(strengthPercent[i] * 0.01);
        halation.sizeUm[i] = static_cast<float>(sizeUm[i]);
        halation.scatteringStrength[i] = static_cast<float>(scatterStrengthPercent[i] * 0.01);
        halation.scatteringSizeUm[i] = static_cast<float>(scatterSizeUm[i]);
    }

    return halation;
}

void JuicerEffect::applyHalationProfileDefaults() {
    if (!_state) {
        return;
    }
    if (!_state->baseLoaded) {
        return;
    }

    const Profiles::HalationMetadata& halationCfg = _state->base.halation;

    auto sanitize_range = [](double value, double fallback, double lo, double hi) -> double {
        if (!std::isfinite(value)) {
            value = fallback;
        }
        return std::clamp(value, lo, hi);
    };

    double strengthR = 0.0, strengthG = 0.0, strengthB = 0.0;
    double sizeR = 0.0, sizeG = 0.0, sizeB = 0.0;
    double scatterStrengthR = 0.0, scatterStrengthG = 0.0, scatterStrengthB = 0.0;
    double scatterSizeR = 0.0, scatterSizeG = 0.0, scatterSizeB = 0.0;

    if (_pHalationStrength) {
        _pHalationStrength->getValue(strengthR, strengthG, strengthB);
    }
    if (_pHalationSizeUm) {
        _pHalationSizeUm->getValue(sizeR, sizeG, sizeB);
    }
    if (_pHalationScatteringStrength) {
        _pHalationScatteringStrength->getValue(scatterStrengthR, scatterStrengthG, scatterStrengthB);
    }
    if (_pHalationScatteringSizeUm) {
        _pHalationScatteringSizeUm->getValue(scatterSizeR, scatterSizeG, scatterSizeB);
    }

    const double strengthPctR = sanitize_range(static_cast<double>(halationCfg.strength[0]) * 100.0, strengthR, 0.0, 100.0);
    const double strengthPctG = sanitize_range(static_cast<double>(halationCfg.strength[1]) * 100.0, strengthG, 0.0, 100.0);
    const double strengthPctB = sanitize_range(static_cast<double>(halationCfg.strength[2]) * 100.0, strengthB, 0.0, 100.0);
    const double sizeUmR = sanitize_range(static_cast<double>(halationCfg.sizeUm[0]), sizeR, 0.0, 1000.0);
    const double sizeUmG = sanitize_range(static_cast<double>(halationCfg.sizeUm[1]), sizeG, 0.0, 1000.0);
    const double sizeUmB = sanitize_range(static_cast<double>(halationCfg.sizeUm[2]), sizeB, 0.0, 1000.0);
    const double scatterStrengthPctR = sanitize_range(static_cast<double>(halationCfg.scatteringStrength[0]) * 100.0, scatterStrengthR, 0.0, 100.0);
    const double scatterStrengthPctG = sanitize_range(static_cast<double>(halationCfg.scatteringStrength[1]) * 100.0, scatterStrengthG, 0.0, 100.0);
    const double scatterStrengthPctB = sanitize_range(static_cast<double>(halationCfg.scatteringStrength[2]) * 100.0, scatterStrengthB, 0.0, 100.0);
    const double scatterSizeUmR = sanitize_range(static_cast<double>(halationCfg.scatteringSizeUm[0]), scatterSizeR, 0.0, 1000.0);
    const double scatterSizeUmG = sanitize_range(static_cast<double>(halationCfg.scatteringSizeUm[1]), scatterSizeG, 0.0, 1000.0);
    const double scatterSizeUmB = sanitize_range(static_cast<double>(halationCfg.scatteringSizeUm[2]), scatterSizeB, 0.0, 1000.0);

    const double strengthMaster = (strengthPctR + strengthPctG + strengthPctB) / 3.0;
    const double sizeMaster = (sizeUmR + sizeUmG + sizeUmB) / 3.0;
    const double scatterStrengthMaster = (scatterStrengthPctR + scatterStrengthPctG + scatterStrengthPctB) / 3.0;
    const double scatterSizeMaster = (scatterSizeUmR + scatterSizeUmG + scatterSizeUmB) / 3.0;

    const bool wasSuppressed = _state->suppressParamEvents;
    _state->suppressParamEvents = true;

    if (_pHalationStrength) {
        _pHalationStrength->setValue(strengthPctR, strengthPctG, strengthPctB);
    }
    if (_pHalationSizeUm) {
        _pHalationSizeUm->setValue(sizeUmR, sizeUmG, sizeUmB);
    }
    if (_pHalationScatteringStrength) {
        _pHalationScatteringStrength->setValue(scatterStrengthPctR, scatterStrengthPctG, scatterStrengthPctB);
    }
    if (_pHalationScatteringSizeUm) {
        _pHalationScatteringSizeUm->setValue(scatterSizeUmR, scatterSizeUmG, scatterSizeUmB);
    }
    if (_pHalationStrengthMaster) {
        _pHalationStrengthMaster->setValue(strengthMaster);
    }
    if (_pHalationSizeUmMaster) {
        _pHalationSizeUmMaster->setValue(sizeMaster);
    }
    if (_pHalationScatteringStrengthMaster) {
        _pHalationScatteringStrengthMaster->setValue(scatterStrengthMaster);
    }
    if (_pHalationScatteringSizeUmMaster) {
        _pHalationScatteringSizeUmMaster->setValue(scatterSizeMaster);
    }

    _state->suppressParamEvents = wasSuppressed;

    _halationStrengthMasterLast = strengthMaster;
    _halationSizeUmMasterLast = sizeMaster;
    _halationScatteringStrengthMasterLast = scatterStrengthMaster;
    _halationScatteringSizeUmMasterLast = scatterSizeMaster;
}

namespace {
    struct GrainPresetDefaults {
        double amountEV = -1.20;
        double sizePx = 0.50;
        double sharpness = 0.5;
        double chroma = 0.3;
        double texture = 0.55;
        double particleAreaUm2 = 0.25;
        double sizeMixScale = 19.0;
        double densityMinMaster = 0.08;
        double uniformityMaster = 0.97;
        double sizeMixWeight = std::numeric_limits<double>::quiet_NaN();
        double microCell = std::numeric_limits<double>::quiet_NaN();
        double microSigma = std::numeric_limits<double>::quiet_NaN();
        double particleScaleMaster = 1.48;
        double particleScaleLayersMaster = 1.922;
        bool sublayersActive = true;
    };

    static GrainPresetDefaults grain_preset_defaults(int presetIndex) {
        GrainPresetDefaults d;
        switch (presetIndex) {
        case 0: // Fine
            d.amountEV = -1.396;
            d.sizePx = 0.615;
            d.sharpness = 0.50;
            d.chroma = 0.00;
            d.texture = 0.63;
            d.particleAreaUm2 = 0.25;
            d.sizeMixScale = 31.3;
            d.densityMinMaster = 0.08;
            d.uniformityMaster = 0.97;
            d.sizeMixWeight = 0.119;
            d.microCell = 60.0;
            d.microSigma = 181.2;
            d.particleScaleMaster = 1.48;
            d.particleScaleLayersMaster = 1.99;
            d.sublayersActive = true;
            break;
        case 2: // Coarse
            d.amountEV = -0.57;
            d.sizePx = 0.56;
            d.sharpness = 0.50;
            d.chroma = 0.50;
            d.texture = 0.35;
            d.particleAreaUm2 = 0.33;
            d.sizeMixScale = 16.0;
            d.densityMinMaster = 0.09;
            d.uniformityMaster = 0.96;
            d.sublayersActive = true;
            break;
        case 1: // Medium
        default:
            break;
        }
        return d;
    }

    inline double grain_lerp(double a, double b, double t) {
        return a + (b - a) * t;
    }

    static std::array<double, 3> default_particle_scale_ratio() {
        return { {0.8, 1.0, 2.0} };
    }

    static std::array<double, 3> default_particle_scale_layers_ratio() {
        return { {2.5, 1.0, 0.5} };
    }

    static std::array<double, 3> default_density_min_ratio() {
        return { {0.07, 0.08, 0.12} };
    }

    static std::array<double, 3> default_uniformity_ratio() {
        return { {0.97, 0.97, 0.99} };
    }

    static void normalize_ratio(std::array<double, 3>& values) {
        double sum = 0.0;
        for (double v : values) {
            if (!std::isfinite(v) || !(v > 0.0)) {
                values = { {1.0, 1.0, 1.0} };
                return;
            }
            sum += v;
        }
        const double mean = sum / 3.0;
        if (!std::isfinite(mean) || !(mean > 0.0)) {
            values = { {1.0, 1.0, 1.0} };
            return;
        }
        for (double& v : values) {
            v /= mean;
        }
    }
}

Profiles::GrainMetadata JuicerEffect::gatherGrainUi() const {
    Profiles::GrainMetadata grain{};

    bool active = false;
    if (_pGrainActive) {
        _pGrainActive->getValue(active);
    }
    grain.active = active;

    int presetIndex = 1;
    if (_pGrainPreset) {
        _pGrainPreset->getValue(presetIndex);
    }
    presetIndex = std::clamp(presetIndex, 0, 2);
    const GrainPresetDefaults preset = grain_preset_defaults(presetIndex);

    bool sublayers = preset.sublayersActive;
    if (_pGrainSublayersActive) {
        _pGrainSublayersActive->getValue(sublayers);
    }
    grain.sublayersActive = sublayers;

    auto sanitize = [](double value, double fallback, double minValue, double maxValue) -> double {
        if (!std::isfinite(value)) return fallback;
        return std::clamp(value, minValue, maxValue);
    };
    auto read3 = [&](OFX::Double3DParam* param, const std::array<double, 3>& defaults, double minValue, double maxValue) {
        std::array<double, 3> values = defaults;
        if (param) {
            param->getValue(values[0], values[1], values[2]);
        }
        for (int i = 0; i < 3; ++i) {
            values[i] = sanitize(values[i], defaults[i], minValue, maxValue);
        }
        return values;
    };
    auto read2 = [&](OFX::Double2DParam* param, const std::array<double, 2>& defaults, double minValue, double maxValue) {
        std::array<double, 2> values = defaults;
        if (param) {
            param->getValue(values[0], values[1]);
        }
        for (int i = 0; i < 2; ++i) {
            values[i] = sanitize(values[i], defaults[i], minValue, maxValue);
        }
        return values;
    };

    double amountEV = preset.amountEV;
    if (_pGrainAmplitude) {
        _pGrainAmplitude->getValue(amountEV);
    }
    amountEV = sanitize(amountEV, preset.amountEV, -3.0, 3.0);
    double amplitude = std::pow(2.0, amountEV);
    if (!std::isfinite(amplitude) || amplitude < 0.0) {
        amplitude = 1.0;
    }
    grain.amplitude = static_cast<float>(amplitude);

    double sizePx = preset.sizePx;
    if (_pGrainBlur) {
        _pGrainBlur->getValue(sizePx);
    }
    sizePx = sanitize(sizePx, preset.sizePx, 0.20, 2.00);
    grain.blur = static_cast<float>(sizePx);

    double sharpness = preset.sharpness;
    if (_pGrainSharpness) {
        _pGrainSharpness->getValue(sharpness);
    }
    sharpness = sanitize(sharpness, preset.sharpness, 0.0, 1.0);

    double chroma = preset.chroma;
    if (_pGrainChroma) {
        _pGrainChroma->getValue(chroma);
    }
    chroma = sanitize(chroma, preset.chroma, 0.0, 1.0);

    double texture = preset.texture;
    if (_pGrainTexture) {
        _pGrainTexture->getValue(texture);
    }
    texture = sanitize(texture, preset.texture, 0.0, 1.0);

    double blurDyeCloudsBase = grain_lerp(1.40, 0.60, sharpness);
    double sizeMixWeightBase = std::isfinite(preset.sizeMixWeight)
        ? preset.sizeMixWeight
        : grain_lerp(0.072, 0.38, texture);
    double microCellBase = std::isfinite(preset.microCell)
        ? preset.microCell
        : grain_lerp(50.0, 70.0, texture);
    double microSigmaBase = std::isfinite(preset.microSigma)
        ? preset.microSigma
        : grain_lerp(140.0, 200.0, texture);

    double particleArea = preset.particleAreaUm2;
    if (_pGrainParticleAreaUm2) {
        _pGrainParticleAreaUm2->getValue(particleArea);
    }
    particleArea = sanitize(particleArea, preset.particleAreaUm2, 0.0, 10.0);
    grain.agxParticleAreaUm2 = static_cast<float>(particleArea);

    double sizeMixScale = preset.sizeMixScale;
    if (_pGrainSizeMixScale) {
        _pGrainSizeMixScale->getValue(sizeMixScale);
    }
    sizeMixScale = sanitize(sizeMixScale, preset.sizeMixScale, 1.0, 50.0);
    grain.sizeMixScale = static_cast<float>(sizeMixScale);

    double sizeMixWeight = sizeMixWeightBase;
    if (_pGrainSizeMixWeight) {
        _pGrainSizeMixWeight->getValue(sizeMixWeight);
    }
    sizeMixWeight = sanitize(sizeMixWeight, sizeMixWeightBase, 0.0, 1.0);
    grain.sizeMixWeight = static_cast<float>(sizeMixWeight);

    double sizeMixWeightMid = 0.0;
    if (_pGrainSizeMixWeightMid) {
        _pGrainSizeMixWeightMid->getValue(sizeMixWeightMid);
    }
    sizeMixWeightMid = sanitize(sizeMixWeightMid, 0.0, 0.0, 1.0);
    grain.sizeMixWeightMid = static_cast<float>(sizeMixWeightMid);

    double blurDyeClouds = blurDyeCloudsBase;
    if (_pGrainBlurDyeCloudsUm) {
        _pGrainBlurDyeCloudsUm->getValue(blurDyeClouds);
    }
    blurDyeClouds = sanitize(blurDyeClouds, blurDyeCloudsBase, 0.0, 10.0);
    grain.blurDyeCloudsUm = static_cast<float>(blurDyeClouds);

    chroma = sanitize(chroma, chroma, 0.0, 1.0);
    grain.chroma = static_cast<float>(chroma);
    grain.chromaSharedWeight = static_cast<float>(std::sqrt(std::max(0.0, 1.0 - chroma)));
    grain.chromaIndWeight = static_cast<float>(std::sqrt(std::max(0.0, chroma)));
    const std::array<double, 3> defaultParticleScale = { {
        preset.particleScaleMaster,
        preset.particleScaleMaster,
        preset.particleScaleMaster
    } };
    const std::array<double, 3> defaultParticleScaleLayers = { {
        preset.particleScaleLayersMaster,
        preset.particleScaleLayersMaster,
        preset.particleScaleLayersMaster
    } };
    const std::array<double, 3> defaultDensityMin = { {
        preset.densityMinMaster,
        preset.densityMinMaster,
        preset.densityMinMaster
    } };
    const std::array<double, 3> defaultUniformity = { {
        preset.uniformityMaster,
        preset.uniformityMaster,
        preset.uniformityMaster
    } };
    const std::array<double, 3> particleScale = read3(_pGrainParticleScale, defaultParticleScale, 0.0, 10.0);
    const std::array<double, 3> particleScaleLayers = read3(_pGrainParticleScaleLayers, defaultParticleScaleLayers, 0.0, 10.0);
    const std::array<double, 3> densityMin = read3(_pGrainDensityMin, defaultDensityMin, 0.0, 1.0);
    const std::array<double, 3> uniformity = read3(_pGrainUniformity, defaultUniformity, 0.0, 1.0);
    for (int i = 0; i < 3; ++i) {
        grain.agxParticleScale[i] = static_cast<float>(particleScale[i]);
        grain.agxParticleScaleLayers[i] = static_cast<float>(particleScaleLayers[i]);
        grain.densityMin[i] = static_cast<float>(densityMin[i]);
        grain.uniformity[i] = static_cast<float>(uniformity[i]);
    }

    double clumpTemporalMix = 0.30;
    if (_pGrainClumpTemporalMix) {
        _pGrainClumpTemporalMix->getValue(clumpTemporalMix);
    }
    clumpTemporalMix = sanitize(clumpTemporalMix, 0.30, 0.0, 0.30);
    grain.clumpTemporalMix = static_cast<float>(clumpTemporalMix);

    double clumpMorphPeriodSec = 8.0;
    if (_pGrainClumpMorphPeriodSec) {
        _pGrainClumpMorphPeriodSec->getValue(clumpMorphPeriodSec);
    }
    clumpMorphPeriodSec = sanitize(clumpMorphPeriodSec, 8.0, 5.0, 60.0);
    grain.clumpMorphPeriodSec = static_cast<float>(clumpMorphPeriodSec);

    std::array<double, 2> microStructure = { {microCellBase, microSigmaBase} };
    microStructure = read2(_pGrainMicroStructure, microStructure, 0.0, 1000.0);
    grain.microStructure[0] = static_cast<float>(microStructure[0]);
    grain.microStructure[1] = static_cast<float>(microStructure[1]);

    bool breathingDebug = false;
    if (_pGrainBreathingDebug) {
        _pGrainBreathingDebug->getValue(breathingDebug);
    }
    grain.breathingDebug = breathingDebug;

    int debugView = 0;
    if (_pGrainDebugView) {
        _pGrainDebugView->getValue(debugView);
    }
    grain.debugView = std::clamp(debugView, 0, 6);

    double filmDust = 0.0;
    if (_pFilmDustAmount) {
        _pFilmDustAmount->getValue(filmDust);
    }
    filmDust = sanitize(filmDust, 0.0, 0.0, 10.0);
    grain.filmDustAmount = static_cast<float>(filmDust);

    double gateDust = 0.0;
    if (_pGateDustAmount) {
        _pGateDustAmount->getValue(gateDust);
    }
    gateDust = sanitize(gateDust, 0.0, 0.0, 10.0);
    grain.gateDustAmount = static_cast<float>(gateDust);

    double filmScratch = 0.0;
    if (_pFilmScratchAmount) {
        _pFilmScratchAmount->getValue(filmScratch);
    }
    filmScratch = sanitize(filmScratch, 0.0, 0.0, 10.0);
    grain.filmScratchAmount = static_cast<float>(filmScratch);

    double gateScratch = 0.0;
    if (_pGateScratchAmount) {
        _pGateScratchAmount->getValue(gateScratch);
    }
    gateScratch = sanitize(gateScratch, 0.0, 0.0, 10.0);
    grain.gateScratchAmount = static_cast<float>(gateScratch);

    grain.nSubLayers = 1;
    return grain;
}

void JuicerEffect::applyGrainPresetDefaults(int presetIndex) {
    const GrainPresetDefaults preset = grain_preset_defaults(presetIndex);
    const bool hadState = (_state != nullptr);
    const bool wasSuppressed = hadState ? _state->suppressParamEvents : false;
    if (hadState) {
        _state->suppressParamEvents = true;
    }

    const double blurDyeClouds = grain_lerp(1.40, 0.60, preset.sharpness);
    const double sizeMixWeight = std::isfinite(preset.sizeMixWeight)
        ? preset.sizeMixWeight
        : grain_lerp(0.072, 0.38, preset.texture);
    const double microCell = std::isfinite(preset.microCell)
        ? preset.microCell
        : grain_lerp(50.0, 70.0, preset.texture);
    const double microSigma = std::isfinite(preset.microSigma)
        ? preset.microSigma
        : grain_lerp(140.0, 200.0, preset.texture);

    std::array<double, 3> scaleRatio = default_particle_scale_ratio();
    std::array<double, 3> scaleLayersRatio = default_particle_scale_layers_ratio();
    std::array<double, 3> densityMinRatio = default_density_min_ratio();
    std::array<double, 3> uniformityRatio = default_uniformity_ratio();
    normalize_ratio(scaleRatio);
    normalize_ratio(scaleLayersRatio);
    normalize_ratio(densityMinRatio);
    normalize_ratio(uniformityRatio);

    if (_pGrainAmplitude) {
        _pGrainAmplitude->setValue(preset.amountEV);
    }
    if (_pGrainBlur) {
        _pGrainBlur->setValue(preset.sizePx);
    }
    if (_pGrainSharpness) {
        _pGrainSharpness->setValue(preset.sharpness);
    }
    if (_pGrainChroma) {
        _pGrainChroma->setValue(preset.chroma);
    }
    if (_pGrainTexture) {
        _pGrainTexture->setValue(preset.texture);
    }
    if (_pGrainSublayersActive) {
        _pGrainSublayersActive->setValue(preset.sublayersActive);
    }

    if (_pGrainParticleAreaUm2) {
        _pGrainParticleAreaUm2->setValue(preset.particleAreaUm2);
    }
    if (_pGrainParticleScaleMaster) {
        _pGrainParticleScaleMaster->setValue(preset.particleScaleMaster);
        _grainParticleScaleMasterLast = preset.particleScaleMaster;
    }
    if (_pGrainParticleScale) {
        _pGrainParticleScale->setValue(
            std::clamp(preset.particleScaleMaster * scaleRatio[0], 0.0, 10.0),
            std::clamp(preset.particleScaleMaster * scaleRatio[1], 0.0, 10.0),
            std::clamp(preset.particleScaleMaster * scaleRatio[2], 0.0, 10.0));
    }
    if (_pGrainParticleScaleLayersMaster) {
        _pGrainParticleScaleLayersMaster->setValue(preset.particleScaleLayersMaster);
        _grainParticleScaleLayersMasterLast = preset.particleScaleLayersMaster;
    }
    if (_pGrainParticleScaleLayers) {
        _pGrainParticleScaleLayers->setValue(
            std::clamp(preset.particleScaleLayersMaster * scaleLayersRatio[0], 0.0, 10.0),
            std::clamp(preset.particleScaleLayersMaster * scaleLayersRatio[1], 0.0, 10.0),
            std::clamp(preset.particleScaleLayersMaster * scaleLayersRatio[2], 0.0, 10.0));
    }
    if (_pGrainDensityMinMaster) {
        _pGrainDensityMinMaster->setValue(preset.densityMinMaster);
        _grainDensityMinMasterLast = preset.densityMinMaster;
    }
    if (_pGrainDensityMin) {
        _pGrainDensityMin->setValue(
            std::clamp(preset.densityMinMaster * densityMinRatio[0], 0.0, 1.0),
            std::clamp(preset.densityMinMaster * densityMinRatio[1], 0.0, 1.0),
            std::clamp(preset.densityMinMaster * densityMinRatio[2], 0.0, 1.0));
    }
    if (_pGrainUniformityMaster) {
        _pGrainUniformityMaster->setValue(preset.uniformityMaster);
        _grainUniformityMasterLast = preset.uniformityMaster;
    }
    if (_pGrainUniformity) {
        _pGrainUniformity->setValue(
            std::clamp(preset.uniformityMaster * uniformityRatio[0], 0.0, 1.0),
            std::clamp(preset.uniformityMaster * uniformityRatio[1], 0.0, 1.0),
            std::clamp(preset.uniformityMaster * uniformityRatio[2], 0.0, 1.0));
    }
    if (_pGrainBlurDyeCloudsUm) {
        _pGrainBlurDyeCloudsUm->setValue(std::clamp(blurDyeClouds, 0.0, 10.0));
    }
    if (_pGrainSizeMixWeight) {
        _pGrainSizeMixWeight->setValue(std::clamp(sizeMixWeight, 0.0, 1.0));
    }
    if (_pGrainSizeMixWeightMid) {
        _pGrainSizeMixWeightMid->setValue(0.0);
    }
    if (_pGrainSizeMixScale) {
        _pGrainSizeMixScale->setValue(std::clamp(preset.sizeMixScale, 1.0, 50.0));
    }
    if (_pGrainMicroStructure) {
        _pGrainMicroStructure->setValue(
            std::clamp(microCell, 0.0, 1000.0),
            std::clamp(microSigma, 0.0, 1000.0));
    }
    if (_pGrainClumpTemporalMix) {
        _pGrainClumpTemporalMix->setValue(0.30);
    }
    if (_pGrainClumpMorphPeriodSec) {
        _pGrainClumpMorphPeriodSec->setValue(8.0);
    }

    if (hadState) {
        _state->suppressParamEvents = wasSuppressed;
    }
    updateGrainPresetLabel(false);
    updateGrainChromaEnabled();
}

void JuicerEffect::resetGrainAdvancedControls() {
    int presetIndex = 1;
    if (_pGrainPreset) {
        _pGrainPreset->getValue(presetIndex);
    }
    presetIndex = std::clamp(presetIndex, 0, 2);
    const GrainPresetDefaults preset = grain_preset_defaults(presetIndex);

    auto sanitize = [](double value, double fallback, double minValue, double maxValue) -> double {
        if (!std::isfinite(value)) return fallback;
        return std::clamp(value, minValue, maxValue);
    };

    double sharpness = preset.sharpness;
    if (_pGrainSharpness) {
        _pGrainSharpness->getValue(sharpness);
    }
    sharpness = sanitize(sharpness, preset.sharpness, 0.0, 1.0);

    double texture = preset.texture;
    if (_pGrainTexture) {
        _pGrainTexture->getValue(texture);
    }
    texture = sanitize(texture, preset.texture, 0.0, 1.0);

    const double blurDyeClouds = grain_lerp(1.40, 0.60, sharpness);
    const double sizeMixWeight = std::isfinite(preset.sizeMixWeight)
        ? preset.sizeMixWeight
        : grain_lerp(0.072, 0.38, texture);
    const double microCell = std::isfinite(preset.microCell)
        ? preset.microCell
        : grain_lerp(50.0, 70.0, texture);
    const double microSigma = std::isfinite(preset.microSigma)
        ? preset.microSigma
        : grain_lerp(140.0, 200.0, texture);

    const double particleArea = preset.particleAreaUm2;
    const double sizeMixScale = preset.sizeMixScale;
    const double densityMinMaster = preset.densityMinMaster;
    const double uniformityMaster = preset.uniformityMaster;
    const double particleScaleMaster = preset.particleScaleMaster;
    const double particleScaleLayersMaster = preset.particleScaleLayersMaster;
    const double clumpTemporalMix = 0.30;
    const double clumpMorphPeriodSec = 8.0;

    std::array<double, 3> scaleRatio = default_particle_scale_ratio();
    std::array<double, 3> scaleLayersRatio = default_particle_scale_layers_ratio();
    std::array<double, 3> densityMinRatio = default_density_min_ratio();
    std::array<double, 3> uniformityRatio = default_uniformity_ratio();
    normalize_ratio(scaleRatio);
    normalize_ratio(scaleLayersRatio);
    normalize_ratio(densityMinRatio);
    normalize_ratio(uniformityRatio);

    const bool hadState = (_state != nullptr);
    const bool wasSuppressed = hadState ? _state->suppressParamEvents : false;
    if (hadState) {
        _state->suppressParamEvents = true;
    }

    if (_pGrainParticleAreaUm2) {
        _pGrainParticleAreaUm2->setValue(particleArea);
    }
    if (_pGrainParticleScaleMaster) {
        _pGrainParticleScaleMaster->setValue(particleScaleMaster);
        _grainParticleScaleMasterLast = particleScaleMaster;
    }
    if (_pGrainParticleScale) {
        _pGrainParticleScale->setValue(
            particleScaleMaster * scaleRatio[0],
            particleScaleMaster * scaleRatio[1],
            particleScaleMaster * scaleRatio[2]);
    }
    if (_pGrainParticleScaleLayersMaster) {
        _pGrainParticleScaleLayersMaster->setValue(particleScaleLayersMaster);
        _grainParticleScaleLayersMasterLast = particleScaleLayersMaster;
    }
    if (_pGrainParticleScaleLayers) {
        _pGrainParticleScaleLayers->setValue(
            particleScaleLayersMaster * scaleLayersRatio[0],
            particleScaleLayersMaster * scaleLayersRatio[1],
            particleScaleLayersMaster * scaleLayersRatio[2]);
    }
    if (_pGrainDensityMinMaster) {
        _pGrainDensityMinMaster->setValue(densityMinMaster);
        _grainDensityMinMasterLast = densityMinMaster;
    }
    if (_pGrainDensityMin) {
        _pGrainDensityMin->setValue(
            densityMinMaster * densityMinRatio[0],
            densityMinMaster * densityMinRatio[1],
            densityMinMaster * densityMinRatio[2]);
    }
    if (_pGrainUniformityMaster) {
        _pGrainUniformityMaster->setValue(uniformityMaster);
        _grainUniformityMasterLast = uniformityMaster;
    }
    if (_pGrainUniformity) {
        _pGrainUniformity->setValue(
            uniformityMaster * uniformityRatio[0],
            uniformityMaster * uniformityRatio[1],
            uniformityMaster * uniformityRatio[2]);
    }
    if (_pGrainBlurDyeCloudsUm) {
        _pGrainBlurDyeCloudsUm->setValue(blurDyeClouds);
    }
    if (_pGrainSizeMixWeight) {
        _pGrainSizeMixWeight->setValue(sizeMixWeight);
    }
    if (_pGrainSizeMixWeightMid) {
        _pGrainSizeMixWeightMid->setValue(0.0);
    }
    if (_pGrainSizeMixScale) {
        _pGrainSizeMixScale->setValue(sizeMixScale);
    }
    if (_pGrainMicroStructure) {
        _pGrainMicroStructure->setValue(microCell, microSigma);
    }
    if (_pGrainClumpTemporalMix) {
        _pGrainClumpTemporalMix->setValue(clumpTemporalMix);
    }
    if (_pGrainClumpMorphPeriodSec) {
        _pGrainClumpMorphPeriodSec->setValue(clumpMorphPeriodSec);
    }

    if (hadState) {
        _state->suppressParamEvents = wasSuppressed;
    }
    updateGrainChromaEnabled();
}

void JuicerEffect::updateGrainPresetLabel(bool custom) {
    _grainPresetCustom = custom;
    if (_pGrainPreset) {
        const std::string label = custom
            ? (_grainPresetLabel + " (Custom)")
            : _grainPresetLabel;
        _pGrainPreset->setLabel(label);
    }
}

void JuicerEffect::updateGrainChromaEnabled() {
    const bool perChannelDirty = false;
    if (_pGrainChroma) {
        _pGrainChroma->setEnabled(!perChannelDirty);
        if (perChannelDirty) {
            _pGrainChroma->setHint("Chroma disabled when per-channel overrides are active.");
        }
        else {
            _pGrainChroma->setHint(_grainChromaHint);
        }
    }
}

Profiles::ProfileGlare JuicerEffect::gatherGlareUi() const {
    Profiles::ProfileGlare glare{};

    bool active = true;
    if (_pGlareActive) {
        _pGlareActive->getValue(active);
    }
    glare.active = active;

    double percent = 0.10;
    if (_pGlarePercent) {
        _pGlarePercent->getValue(percent);
    }
    if (!std::isfinite(percent)) {
        percent = 0.10;
    }
    percent = std::clamp(percent, 0.0, 1.0);
    glare.percent = static_cast<float>(percent);

    double roughness = 0.4;
    if (_pGlareRoughness) {
        _pGlareRoughness->getValue(roughness);
    }
    if (!std::isfinite(roughness)) {
        roughness = 0.4;
    }
    roughness = std::clamp(roughness, 0.0, 1.0);
    glare.roughness = static_cast<float>(roughness);

    double blur = 0.5;
    if (_pGlareBlurSigmaPx) {
        _pGlareBlurSigmaPx->getValue(blur);
    }
    if (!std::isfinite(blur)) {
        blur = 0.5;
    }
    blur = std::clamp(blur, 0.0, 10.0);
    glare.blur = static_cast<float>(blur);

    double factor = 0.0;
    if (_pGlareCompRemovalFactor) {
        _pGlareCompRemovalFactor->getValue(factor);
    }
    if (!std::isfinite(factor)) {
        factor = 0.0;
    }
    factor = std::clamp(factor, 0.0, 1.0);
    glare.compensationRemovalFactor = static_cast<float>(factor);

    double density = 1.2;
    if (_pGlareCompRemovalDensity) {
        _pGlareCompRemovalDensity->getValue(density);
    }
    if (!std::isfinite(density)) {
        density = 1.2;
    }
    density = std::clamp(density, 0.0, 3.0);
    glare.compensationRemovalDensity = static_cast<float>(density);

    double transition = 0.3;
    if (_pGlareCompRemovalTransition) {
        _pGlareCompRemovalTransition->getValue(transition);
    }
    if (!std::isfinite(transition)) {
        transition = 0.3;
    }
    transition = std::clamp(transition, 0.0, 2.0);
    glare.compensationRemovalTransition = static_cast<float>(transition);

    return glare;
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

    if (_state) {
        std::lock_guard<std::mutex> cacheLock(_state->autoExposureMutex);
        _state->autoExposureCanonicalBounds = meterBounds;
        _state->autoExposureCanonicalValid = true;
    }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    // CUDA path: metering + exposure scale are computed and applied entirely on the GPU to avoid
    // forcing a stream synchronization just to read back Y/EV on the CPU.
    if (exposureParams.cameraAutoEnabled && args.isEnabledCudaRender) {
        result.autoEV = 0.0;
        result.exposureScale = 1.0f;
        return result;
    }
#endif

    const std::shared_ptr<const WorkingState> wsCur = (_state ? JuicerAtomic::load_shared_ptr(&_state->activeWorkingState) : nullptr);
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
    const int meteringMethod = exposureParams.meteringMethod;
    if (_state) {
        std::lock_guard<std::mutex> cacheLock(_state->autoExposureMutex);
        if (_state->autoExposureCacheValid &&
            _state->autoExposureCacheIsCudaRender == args.isEnabledCudaRender &&
            _state->autoExposureCacheAutoEnabled == cameraAutoEnabled &&
            _state->autoExposureCacheMeteringMethod == meteringMethod &&
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
            double Yexp = 0.0;
            bool haveY = false;
            if (args.isEnabledCudaRender) {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
                const OfxRectI imgBounds = srcImg->getBounds();
                const OFX::PixelComponentEnum comps = srcImg->getPixelComponents();
                const int nComponents =
                    (comps == OFX::ePixelComponentRGBA) ? 4 :
                    (comps == OFX::ePixelComponentRGB) ? 3 :
                    (comps == OFX::ePixelComponentAlpha) ? 1 : 0;
                if (!(nComponents == 3 || nComponents == 4)) {
                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
                }
                const std::ptrdiff_t rowBytes = srcImg->getRowBytes();
                if (rowBytes <= 0) {
                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
                }
                const void* srcDevice = srcImg->getPixelData();
                if (!srcDevice) {
                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
                }

                const char* errMsg = nullptr;
                const int rc = (meteringMethod == static_cast<int>(MeteringMethod::Median))
                    ? juicer_cuda_measure_median_Y(
                        srcDevice,
                        static_cast<std::size_t>(rowBytes),
                        imgBounds.x1,
                        imgBounds.y1,
                        imgBounds.x2,
                        imgBounds.y2,
                        meterBounds.x1,
                        meterBounds.y1,
                        meterBounds.x2,
                        meterBounds.y2,
                        nComponents,
                        inputColorSpaceIndex,
                        applyInputCctfDecoding ? 1 : 0,
                        inputRgbToXYZ.m,
                        &Yexp,
                        args.pCudaStream,
                        &errMsg)
                    : juicer_cuda_measure_center_weighted_Y(
                        srcDevice,
                        static_cast<std::size_t>(rowBytes),
                        imgBounds.x1,
                        imgBounds.y1,
                        imgBounds.x2,
                        imgBounds.y2,
                        meterBounds.x1,
                        meterBounds.y1,
                        meterBounds.x2,
                        meterBounds.y2,
                        nComponents,
                        inputColorSpaceIndex,
                        applyInputCctfDecoding ? 1 : 0,
                        inputRgbToXYZ.m,
                        &Yexp,
                        args.pCudaStream,
                        &errMsg);
                if (rc == 0) {
                    haveY = true;
                } else {
                    JTRACE("CUDA", std::string("CUDA auto-exposure metering failed: ") + (errMsg ? errMsg : "(unknown)"));
#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
#else
                    throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
                }
#else
                throw OFX::Exception::Suite(kOfxStatErrUnsupported);
#endif
            } else {
                if (meteringMethod == static_cast<int>(MeteringMethod::Median)) {
                    Yexp = measure_median_Y_DWG(
                        srcImg,
                        meterBounds,
                        inputColorSpace,
                        inputRgbToXYZ,
                        applyInputCctfDecoding);
                } else {
                    Yexp = measure_center_weighted_Y_DWG_cached(
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
                }
                haveY = true;
            }

            if (haveY) {
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
        }
        autoEV = evComp;

        if (_state) {
            std::lock_guard<std::mutex> cacheLock(_state->autoExposureMutex);
            _state->autoExposureCacheValid = measurementValid;
            _state->autoExposureCacheIsCudaRender = args.isEnabledCudaRender;
            _state->autoExposureCacheTime = args.time;
            _state->autoExposureCacheAutoEnabled = cameraAutoEnabled;
            _state->autoExposureCacheMeteringMethod = meteringMethod;
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
    int fullHeight,
    float pixelSizeUm) const {

    Couplers::Runtime dirRT{};
    if (!(_state && _state->baseLoaded)) {
        return dirRT;
    }

    const std::shared_ptr<const WorkingState> wsCur = JuicerAtomic::load_shared_ptr(&_state->activeWorkingState);
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

    auto valid_positive = [](double v) -> bool { return std::isfinite(v) && v > 0.0; };

    float sigmaPixels = 0.0f;
    const float sigmaMicrometers = dirRT.spatialSigmaMicrometers;
    if (sigmaMicrometers > 0.0f && valid_positive(pixelSizeUm)) {
        sigmaPixels = sigmaMicrometers / pixelSizeUm;
        if (!std::isfinite(sigmaPixels) || sigmaPixels < 0.0f) {
            sigmaPixels = 0.0f;
        }
    }
    else {
        // Fallback to legacy geometry if pixelSizeUm was not available
        double filmLongEdgeMm = 35.0;
        if (_pCameraFilmFormat) {
            double filmFormat = 35.0;
            _pCameraFilmFormat->getValue(filmFormat);
            if (std::isfinite(filmFormat) && filmFormat > 0.0) {
                filmLongEdgeMm = filmFormat;
            }
        }

        const double widthPx = static_cast<double>(fullWidth);
        const double heightPx = static_cast<double>(fullHeight);
        const double longEdgePx = std::max(widthPx, heightPx);

        if (sigmaMicrometers > 0.0f && valid_positive(longEdgePx) && valid_positive(filmLongEdgeMm)) {
            sigmaPixels = Couplers::spatial_sigma_pixels_from_micrometers(
                sigmaMicrometers,
                filmLongEdgeMm,
                widthPx,
                heightPx);
            if (!std::isfinite(sigmaPixels) || sigmaPixels < 0.0f) {
                sigmaPixels = 0.0f;
            }
        }
    }
    dirRT.spatialSigmaPixels = sigmaPixels;

    auto dir_has_effect = [](const Couplers::Runtime& rt) -> bool {
        if (!rt.active) {
            return false;
        }
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                const float v = rt.M[r][c];
                if (!std::isfinite(v)) {
                    continue;
                }
                if (v != 0.0f) {
                    return true;
                }
            }
        }
        return false;
    };

    if (!dir_has_effect(dirRT)) {
        // Provably zero-effect DIR (e.g. amount==0). Treat as inactive so we can skip the
        // extra DIR sampling path and any spatial-DIR build work without changing results.
        dirRT.active = false;
    }

    return dirRT;
}
#endif

JuicerEffect::WorkingStateInfo JuicerEffect::prepareWorkingState() const {
    WorkingStateInfo info{};
    if (!(_state && _state->baseLoaded)) {
        return info;
    }

    info.workingState = JuicerAtomic::load_shared_ptr(&_state->activeWorkingState);

    const WorkingState* ws = info.workingState.get();
    if (ws && ws->buildCounter > 0 && ws->printRT) {
        info.printRuntime = ws->printRT.get();
    }

    const bool baselineReady = (!ws || !ws->hasBaseline) ||
        (ws->baseMin.linear.size() == static_cast<size_t>(Spectral::gShape.K));

    const bool negativeScannerReady = ws && ws->negativeScannerValid &&
        ws->negativeMediumRuntime.staticKey.hash != 0 &&
        ws->negativeMediumRuntime.range.digest != 0 &&
        ws->negativeMediumRuntime.tables &&
        ws->negativeMediumRuntime.tables->K == Spectral::gShape.K;

    info.workingStateReady = (ws && ws->buildCounter > 0 &&
        ws->tablesView.K == Spectral::gShape.K &&
        ws->tablesView.epsY.size() == static_cast<size_t>(Spectral::gShape.K) &&
        ws->tablesView.epsM.size() == static_cast<size_t>(Spectral::gShape.K) &&
        ws->tablesView.epsC.size() == static_cast<size_t>(Spectral::gShape.K) &&
        baselineReady &&
        !ws->densB.lambda_nm.empty() && !ws->densB.linear.empty() &&
        !ws->densG.lambda_nm.empty() && !ws->densG.linear.empty() &&
        !ws->densR.lambda_nm.empty() && !ws->densR.linear.empty() &&
        negativeScannerReady);

    const Print::Runtime* prt = info.printRuntime;
    const bool printScannerReady = ws && ws->printScannerValid &&
        ws->printMediumRuntime.staticKey.hash != 0 &&
        ws->printMediumRuntime.range.digest != 0 &&
        ws->printMediumRuntime.tables &&
        ws->printMediumRuntime.tables->K == Spectral::gShape.K;

    info.printRuntimeReady =
        (prt != nullptr) &&
        Print::profile_is_valid(prt->profile) &&
        prt->illumView.linear.size() == static_cast<size_t>(Spectral::gShape.K) &&
        prt->illumEnlarger.linear.size() == static_cast<size_t>(Spectral::gShape.K) &&
        (ws && ws->tablesPrint.K == Spectral::gShape.K) &&
        info.workingStateReady &&
        printScannerReady;

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
        _pCameraMeteringMethod = fetchChoiceParam(JuicerParams::kCameraMeteringMethod);
        _pFilmStock = fetchChoiceParam(kParamFilmStock);
        _pSpectralMode = fetchChoiceParam(kParamSpectralMode);
        _pPrintPaper = fetchChoiceParam(kParamPrintPaper);
        _pRefIll = fetchChoiceParam("ReferenceIlluminant");
        _pEnlIll = fetchChoiceParam("EnlargerIlluminant");
        _pEnlDichroicSet = fetchChoiceParam(kParamEnlargerDichroicSet);
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

        _pScannerLensBlur = fetchDoubleParam(JuicerParams::kScannerLensBlurSigmaPx);
        _pScannerUnsharp = fetchDouble2DParam(JuicerParams::kScannerUnsharpMask);
        _pScannerUseLut = fetchBooleanParam(JuicerParams::kScannerUseLut);
        _pScannerLutResolution = fetchIntParam(JuicerParams::kScannerLutResolution);

        _pPrintBypass = fetchBooleanParam("PrintBypass");
        _pPrintExposure = fetchDoubleParam("PrintExposure");
        _pPrintPreflash = fetchDoubleParam("PrintPreflash");
        _pPrintExposureComp = fetchBooleanParam("PrintExposureCompensation");
        _pEnlargerY = fetchDoubleParam("EnlargerY");
        _pEnlargerM = fetchDoubleParam("EnlargerM");
        _pEnlargerC = fetchDoubleParam("EnlargerC");

        _pHalationActive = fetchBooleanParam(JuicerParams::kHalationActive);
        _pHalationStrengthMaster = fetchDoubleParam(JuicerParams::kHalationStrengthMaster);
        _pHalationSizeUmMaster = fetchDoubleParam(JuicerParams::kHalationSizeUmMaster);
        _pHalationScatteringStrengthMaster = fetchDoubleParam(JuicerParams::kHalationScatteringStrengthMaster);
        _pHalationScatteringSizeUmMaster = fetchDoubleParam(JuicerParams::kHalationScatteringSizeUmMaster);
        _pHalationRevertToStock = fetchPushButtonParam(JuicerParams::kHalationRevertToStock);
        _pHalationStrength = fetchDouble3DParam(JuicerParams::kHalationStrength);
        _pHalationSizeUm = fetchDouble3DParam(JuicerParams::kHalationSizeUm);
        _pHalationScatteringStrength = fetchDouble3DParam(JuicerParams::kHalationScatteringStrength);
        _pHalationScatteringSizeUm = fetchDouble3DParam(JuicerParams::kHalationScatteringSizeUm);

        _pGrainActive = fetchBooleanParam(JuicerParams::kGrainActive);
        _pGrainSublayersActive = fetchBooleanParam(JuicerParams::kGrainSublayersActive);
        _pGrainPreset = fetchChoiceParam(JuicerParams::kGrainPreset);
        _pGrainParticleAreaUm2 = fetchDoubleParam(JuicerParams::kGrainParticleAreaUm2);
        _pGrainAmplitude = fetchDoubleParam(JuicerParams::kGrainAmplitude);
        _pGrainSharpness = fetchDoubleParam(JuicerParams::kGrainSharpness);
        _pGrainChroma = fetchDoubleParam(JuicerParams::kGrainChroma);
        _pGrainTexture = fetchDoubleParam(JuicerParams::kGrainTexture);
        _pGrainParticleScaleMaster = fetchDoubleParam(JuicerParams::kGrainParticleScaleMaster);
        _pGrainParticleScaleLayersMaster = fetchDoubleParam(JuicerParams::kGrainParticleScaleLayersMaster);
        _pGrainDensityMinMaster = fetchDoubleParam(JuicerParams::kGrainDensityMinMaster);
        _pGrainUniformityMaster = fetchDoubleParam(JuicerParams::kGrainUniformityMaster);
        _pGrainParticleScale = fetchDouble3DParam(JuicerParams::kGrainParticleScale);
        _pGrainParticleScaleLayers = fetchDouble3DParam(JuicerParams::kGrainParticleScaleLayers);
        _pGrainDensityMin = fetchDouble3DParam(JuicerParams::kGrainDensityMin);
        _pGrainUniformity = fetchDouble3DParam(JuicerParams::kGrainUniformity);
        _pGrainBlur = fetchDoubleParam(JuicerParams::kGrainBlur);
        _pGrainBlurDyeCloudsUm = fetchDoubleParam(JuicerParams::kGrainBlurDyeCloudsUm);
        _pGrainSizeMixWeight = fetchDoubleParam(JuicerParams::kGrainSizeMixWeight);
        _pGrainSizeMixWeightMid = fetchDoubleParam(JuicerParams::kGrainSizeMixWeightMid);
        _pGrainSizeMixScale = fetchDoubleParam(JuicerParams::kGrainSizeMixScale);
        _pGrainClumpTemporalMix = fetchDoubleParam(JuicerParams::kGrainClumpTemporalMix);
        _pGrainClumpMorphPeriodSec = fetchDoubleParam(JuicerParams::kGrainClumpMorphPeriodSec);
        _pGrainBreathingDebug = fetchBooleanParam(JuicerParams::kGrainBreathingDebug);
        _pGrainDebugView = fetchChoiceParam(JuicerParams::kGrainDebugView);
        _pGrainMicroStructure = fetchDouble2DParam(JuicerParams::kGrainMicroStructure);
        _pGrainResetAdvanced = fetchPushButtonParam(JuicerParams::kGrainResetAdvanced);
        _pGateWeaveAmount = fetchDoubleParam(JuicerParams::kGateWeaveAmount);
        _pFilmDustAmount = fetchDoubleParam(JuicerParams::kFilmDustAmount);
        _pGateDustAmount = fetchDoubleParam(JuicerParams::kGateDustAmount);
        _pFilmScratchAmount = fetchDoubleParam(JuicerParams::kFilmScratchAmount);
        _pGateScratchAmount = fetchDoubleParam(JuicerParams::kGateScratchAmount);

        _pGlareActive = fetchBooleanParam(JuicerParams::kGlareActive);
        _pGlarePercent = fetchDoubleParam(JuicerParams::kGlarePercent);
        _pGlareRoughness = fetchDoubleParam(JuicerParams::kGlareRoughness);
        _pGlareBlurSigmaPx = fetchDoubleParam(JuicerParams::kGlareBlurSigmaPx);
        _pGlareCompRemovalFactor = fetchDoubleParam(JuicerParams::kGlareCompensationRemovalFactor);
        _pGlareCompRemovalDensity = fetchDoubleParam(JuicerParams::kGlareCompensationRemovalDensity);
        _pGlareCompRemovalTransition = fetchDoubleParam(JuicerParams::kGlareCompensationRemovalTransition);
        _pPrintDminFactor = fetchDoubleParam(JuicerParams::kPrintDminFactor);
    }
    catch (...) {
        // Safe: any missing param will remain nullptr and defaults are used in snapshot/usage paths.
    }

    if (_pGrainPreset) {
        std::string label;
        _pGrainPreset->getLabel(label);
        if (!label.empty()) {
            _grainPresetLabel = label;
        }
    }
    if (_pGrainChroma) {
        const std::string hint = _pGrainChroma->getHint();
        if (!hint.empty()) {
            _grainChromaHint = hint;
        }
    }

    auto initMasterCache = [](OFX::DoubleParam* param, double& outValue) {
        if (!param) {
            outValue = std::numeric_limits<double>::quiet_NaN();
            return;
        }
        double v = 0.0;
        param->getValue(v);
        outValue = v;
    };
    initMasterCache(_pHalationStrengthMaster, _halationStrengthMasterLast);
    initMasterCache(_pHalationSizeUmMaster, _halationSizeUmMasterLast);
    initMasterCache(_pHalationScatteringStrengthMaster, _halationScatteringStrengthMasterLast);
    initMasterCache(_pHalationScatteringSizeUmMaster, _halationScatteringSizeUmMasterLast);
    initMasterCache(_pGrainParticleScaleMaster, _grainParticleScaleMasterLast);
    initMasterCache(_pGrainParticleScaleLayersMaster, _grainParticleScaleLayersMasterLast);
    initMasterCache(_pGrainDensityMinMaster, _grainDensityMinMasterLast);
    initMasterCache(_pGrainUniformityMaster, _grainUniformityMasterLast);

    // Own per-instance state
    _state = std::make_unique<InstanceState>();
    _state->dataDir = ensure_trailing_separator(data_dir_string());
    JuicerAtomic::store_shared_ptr(&_state->activeWorkingState, std::shared_ptr<const WorkingState>{});
    _state->activeBuildCounter = 0;
    {
        const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        const std::uint64_t seedFields[2] = {
            static_cast<std::uint64_t>(now),
            static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(this))
        };
        std::uint64_t seed = Hash::hash_bytes(seedFields, sizeof(seedFields));
        if (seed == 0) {
            seed = 1;
        }
        _state->sessionSeed = seed;
        _state->instanceToken = seed;
    }

    // Optional compatibility registry
    JuicerRegistry::set(handle, _state.get());

    // Defer heavy bootstrap until first param change
}

JuicerEffect::~JuicerEffect() {
    // Mirror destroyInstance() guards without touching C suites.
    JuicerRegistry::erase(this->getHandle());
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    if (_state) {
        std::vector<JuicerCuda::ResourceManager::DeviceContextKey> keys;
        {
            std::lock_guard<std::mutex> lock(_state->cudaMutex);
            keys.reserve(_state->cudaByDevice.size());
            for (const auto& entry : _state->cudaByDevice) {
                keys.push_back(entry.first);
            }
        }
        for (const auto& key : keys) {
            std::string retireError;
            const bool retireOk = JuicerCuda::ResourceManager::command_retire_context_idle(key, retireError);
            if (!retireOk || !retireError.empty()) {
                const std::uintptr_t contextBits = reinterpret_cast<std::uintptr_t>(key.contextOpaque);
                std::string msg = std::string("teardown_retire_idle_failed device_id=")
                    + std::to_string(key.deviceId)
                    + " context=" + std::to_string(contextBits)
                    + " accepted=" + std::to_string(retireOk ? 1 : 0);
                if (!retireError.empty()) {
                    msg += " error=" + retireError;
                }
                JTRACE("MSLCY", msg);
            }
        }
    }
#endif
    _state.reset();
}

void JuicerEffect::render(const OFX::RenderArguments& args) {
    // Fetch images via wrappers
    std::unique_ptr<OFX::Image> srcImg(_src ? _src->fetchImage(args.time) : nullptr);
    std::unique_ptr<OFX::Image> dstImg(_dst ? _dst->fetchImage(args.time) : nullptr);
    if (!srcImg || !dstImg) return;

#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
    // CUDA-only mode: reject CPU/OpenCL/Metal renders. During development this stays disabled so
    // we can fall back to the CPU pipeline while CUDA parity is still in progress.
    if (!args.isEnabledCudaRender) {
        JTRACE("CUDA", "JUICER_CUDA_ONLY: rejecting non-CUDA render request");
        throw OFX::Exception::Suite(kOfxStatErrUnsupported);
    }
#endif

    // Components and depth
    const OFX::PixelComponentEnum comps = srcImg->getPixelComponents();
    const OFX::BitDepthEnum depth = srcImg->getPixelDepth();

    const int nComponents =
        (comps == OFX::ePixelComponentRGBA) ? 4 :
        (comps == OFX::ePixelComponentRGB) ? 3 :
        (comps == OFX::ePixelComponentAlpha) ? 1 : 0;

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    if (args.isEnabledCudaRender) {
        // CUDA renders use device pointers; avoid CPU pixel reads (auto-exposure, non-float copies, etc.).
        if (depth != OFX::eBitDepthFloat || nComponents == 0) {
            throw OFX::Exception::Suite(kOfxStatErrUnsupported);
        }
    }
#else
    if (args.isEnabledCudaRender) {
        throw OFX::Exception::Suite(kOfxStatErrUnsupported);
    }
#endif

    if (depth != OFX::eBitDepthFloat) {
        JuicerProc::copyNonFloatRect(srcImg.get(), dstImg.get());
        return;
    }

    if (nComponents == 0) {
        JuicerProc::copyNonFloatRect(srcImg.get(), dstImg.get());
        return;
    }

    const OfxRectI fullBounds = srcImg->getBounds();
    if (_state) {
        std::lock_guard<std::mutex> lock(_state->m);
        const OfxRectI prev = _state->cachedFrameBounds;
        const bool changed = prev.x1 != fullBounds.x1 || prev.y1 != fullBounds.y1 ||
            prev.x2 != fullBounds.x2 || prev.y2 != fullBounds.y2;
        if (changed) {
            _state->cachedFrameBounds = fullBounds;
            std::uint32_t next = _state->frameBoundsVersion.load(std::memory_order_relaxed);
            next = (next == std::numeric_limits<std::uint32_t>::max()) ? next : (next + 1U);
            if (next == 0) {
                next = 1;
            }
            _state->frameBoundsVersion.store(next, std::memory_order_release);
        }
    }

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
    const bool fullFrame = (roi.x1 == fullBounds.x1 && roi.y1 == fullBounds.y1 &&
        roi.x2 == fullBounds.x2 && roi.y2 == fullBounds.y2);
    if (!fullFrame) {
        JTRACE("RENDER", "FATAL: render window must match full frame; tiles/ROIs are unsupported");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    double filmFormatMm = 35.0;
    if (_pCameraFilmFormat) {
        double filmFormat = 35.0;
        _pCameraFilmFormat->getValue(filmFormat);
        if (std::isfinite(filmFormat) && filmFormat > 0.0) {
            filmFormatMm = filmFormat;
        }
    }
    const double longEdgePx = static_cast<double>(std::max(fullWidth, fullHeight));
    float pixelSizeUm = 0.0f;
    if (std::isfinite(filmFormatMm) && filmFormatMm > 0.0 && longEdgePx > 0.0) {
        pixelSizeUm = static_cast<float>((filmFormatMm * 1000.0) / longEdgePx);
    }

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

    // Coalesce parameter-driven WorkingState rebuilds on the render thread to keep UI callbacks fast.
    if (_state && _state->baseLoaded) {
        ParamSnapshot pendingParams{};
        std::uint64_t pendingFullHash = 0;
        std::uint64_t pendingCoreHash = 0;
        std::uint64_t pendingDirHash = 0;
        {
            std::lock_guard<std::mutex> lock(_state->pending.m);
            pendingParams = _state->pending.params;
            pendingFullHash = _state->pending.fullHash;
            pendingCoreHash = _state->pending.coreHash;
            pendingDirHash = _state->pending.dirHash;
        }

        const std::shared_ptr<const WorkingState> wsCur = JuicerAtomic::load_shared_ptr(&_state->activeWorkingState);
        const std::uint64_t builtFullHash = wsCur ? wsCur->fullHash : 0;
        if (pendingFullHash != 0 && pendingFullHash != builtFullHash) {
            const std::uint64_t builtCoreHash = wsCur ? wsCur->coreHash : 0;
            const std::uint64_t builtDirHash = wsCur ? wsCur->dirHash : 0;
            const bool dirOnly = (pendingCoreHash != 0) && (builtCoreHash != 0) &&
                (pendingCoreHash == builtCoreHash) &&
                (pendingDirHash != 0) && (pendingDirHash != builtDirHash);
            if (dirOnly) {
                rebuild_working_state_couplers_only(this->getHandle(), *_state, pendingParams);
            }
            else {
                rebuild_working_state(this->getHandle(), *_state, pendingParams);
            }
        }
    }
    const ExposureParams exposureParams = gatherExposureParams();
    const Scanner::Options scannerOptions = gatherScannerOptions();
    const Scanner::Settings scannerSettings = gatherScannerSettings();
    Print::Params printParams = gatherPrintParams();
    const Profiles::HalationMetadata halationUi = gatherHalationUi();
    const Profiles::GrainMetadata grainUi = gatherGrainUi();
    const Profiles::ProfileGlare glareUi = gatherGlareUi();
    double gateWeaveAmount = 1.0;
    if (_pGateWeaveAmount) {
        _pGateWeaveAmount->getValue(gateWeaveAmount);
    }
    OutputEncoding::Params outputEncodingParams = gatherOutputEncodingParams();

    const AutoExposureResult autoExposure = computeAutoExposure(
        args,
        srcImg.get(),
        fullBounds,
        exposureParams);

#ifdef JUICER_ENABLE_COUPLERS
    Couplers::Runtime dirRT = prepareCouplers(args, fullWidth, fullHeight, pixelSizeUm);
#else
    Couplers::Runtime dirRT{};
#endif

    WorkingStateInfo wsInfo = prepareWorkingState();
    std::shared_ptr<const WorkingState> wsHold = wsInfo.workingState;
    const WorkingState* ws = wsHold.get();
    const Print::Runtime* prt = wsInfo.printRuntime;
    const bool wsReady = wsInfo.workingStateReady;
    const bool printReady = wsInfo.printRuntimeReady;
    if (JTRACE_ENABLED(3)) {
        ParamSnapshot Pdbg = snapshotParams();
        const char* paperKey = print_paper_json_key_for_index(Pdbg.printPaperIndex);
        const char* filmKey = negative_json_key_for_stock_index(Pdbg.filmStockIndex);
        const std::uintptr_t prtPtr = reinterpret_cast<std::uintptr_t>(prt);
        const std::uint64_t buildCounter = ws ? ws->buildCounter : 0;
        const float neutralY = prt ? prt->neutralY : 0.0f;
        const float neutralM = prt ? prt->neutralM : 0.0f;
        const float neutralC = prt ? prt->neutralC : 0.0f;
        std::string msg = std::string("render print state build=") + std::to_string(buildCounter)
            + " paper=" + std::string(paperKey ? paperKey : "<null>")
            + " film=" + std::string(filmKey ? filmKey : "<null>")
            + " printRT=" + std::to_string(prtPtr)
            + " neutralY/M/C=" + std::to_string(neutralY) + "/" + std::to_string(neutralM) + "/" + std::to_string(neutralC)
            + " yFilter=" + std::to_string(printParams.yFilter)
            + " mFilter=" + std::to_string(printParams.mFilter)
            + " cFilter=" + std::to_string(printParams.cFilter)
            + " bypass=" + std::to_string(printParams.bypass ? 1 : 0);
        JTRACE_VERBOSE("PRINTDBG", msg);
    }
    if (!wsReady) {
        JTRACE("BUILD", "FATAL: working state not ready; aborting render");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    if (!printParams.bypass && !printReady) {
        JTRACE("PRINT", "FATAL: print runtime not ready while print path requested");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
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
    proc.setScannerOptions(scannerOptions);
    proc.setScannerSettings(scannerSettings);
    proc.setPrintParams(printParams);
    proc.setHalationOverride(halationUi);
    proc.setGrainOverride(grainUi);
    proc.setGateWeaveAmount(gateWeaveAmount);
    proc.setPrintGlareOverride(glareUi);
    proc.setDirRuntime(dirRT);
    proc.setWorkingState(ws, wsReady);
    proc.setPrintRuntime(prt, printReady);
    proc.setInstanceState(_state.get());
    const std::uint32_t frameVersion = _state
        ? _state->frameBoundsVersion.load(std::memory_order_acquire)
        : 0;
    proc.setFrameBoundsVersion(frameVersion);
    proc.setPixelSizeUm(pixelSizeUm);
    float filmExposureScale = autoExposure.exposureScale;
    // Per agx-emulsion parity: autoExposure.exposureScale already encodes 2^(autoEV + sliderEV).
    if (!std::isfinite(filmExposureScale) || filmExposureScale <= 0.0f) {
        filmExposureScale = 1.0f;
    }
    proc.setExposure(filmExposureScale);
    proc.setCameraAutoExposure(exposureParams.cameraAutoEnabled, exposureParams.meteringMethod, exposureParams.sliderEV);
    proc.setOutputEncoding(outputEncodingParams);
    const std::uintptr_t renderClipToken = reinterpret_cast<std::uintptr_t>(_src);
    proc.setClipToken(renderClipToken);
    proc.setFrameRate(getFrameRate());
    proc.setFrameTime(args.time);
    proc.setRenderWindowRect(roi);
    proc.setGPURenderArgs(args);
    proc.setRenderHints(args.interactiveRenderStatus, args.renderQualityDraft, args.sequentialRenderStatus);

    // Dispatch to support library's threaded/tiled CPU path
    proc.process();
}

void JuicerEffect::changedParam(const OFX::InstanceChangedArgs& args, const std::string& paramName) {
    // Suppress recursion while we are programmatically setting params
    if (_state && _state->suppressParamEvents) {
        JTRACE("BUILD", std::string("changedParam suppressed for '") + paramName + "'");
        return;
    }
    if (_state && _state->inBootstrap) {
        JTRACE("BUILD", std::string("changedParam ignored during bootstrap for '") + paramName + "'");
        return;
    }
    if (_state && (paramName == kParamCameraAutoExposure || paramName == JuicerParams::kCameraMeteringMethod)) {
        std::lock_guard<std::mutex> cacheLock(_state->autoExposureMutex);
        _state->autoExposureCacheValid = false;
    }
    if (paramName == JuicerParams::kHalationRevertToStock) {
        applyHalationProfileDefaults();
        onParamsPossiblyChanged(paramName.c_str());
        return;
    }

    const bool userEdit = (args.reason == OFX::eChangeUserEdit);
    if (paramName == JuicerParams::kGrainPreset && userEdit) {
        int presetIndex = 1;
        if (_pGrainPreset) {
            _pGrainPreset->getValue(presetIndex);
        }
        presetIndex = std::clamp(presetIndex, 0, 2);
        applyGrainPresetDefaults(presetIndex);
        onParamsPossiblyChanged(paramName.c_str());
        return;
    }
    if (paramName == JuicerParams::kGrainResetAdvanced && userEdit) {
        resetGrainAdvancedControls();
        onParamsPossiblyChanged(paramName.c_str());
        return;
    }

    auto sanitize_scalar = [](double value, double fallback, double lo, double hi) -> double {
        if (!std::isfinite(value)) value = fallback;
        return std::clamp(value, lo, hi);
    };

    auto apply_master_delta = [&](OFX::DoubleParam* masterParam,
        OFX::Double3DParam* advParam,
        double& masterCache,
        double lo,
        double hi) {
        if (!masterParam || !advParam || !_state) {
            return;
        }
        double master = 0.0;
        masterParam->getValue(master);
        if (!std::isfinite(master)) {
            return;
        }
        master = std::clamp(master, lo, hi);
        double prev = masterCache;
        if (!std::isfinite(prev)) {
            prev = master;
        }
        const double delta = master - prev;
        double r = 0.0, g = 0.0, b = 0.0;
        advParam->getValue(r, g, b);
        r = sanitize_scalar(r, master, lo, hi);
        g = sanitize_scalar(g, master, lo, hi);
        b = sanitize_scalar(b, master, lo, hi);
        if (delta != 0.0) {
            r = std::clamp(r + delta, lo, hi);
            g = std::clamp(g + delta, lo, hi);
            b = std::clamp(b + delta, lo, hi);
            const bool wasSuppressed = _state->suppressParamEvents;
            _state->suppressParamEvents = true;
            advParam->setValue(r, g, b);
            _state->suppressParamEvents = wasSuppressed;
        }
        masterCache = master;
    };

    auto apply_ratio_master = [&](OFX::DoubleParam* masterParam,
        OFX::Double3DParam* advParam,
        const std::array<double, 3>& fallbackRatio,
        double& masterCache,
        double lo,
        double hi) {
        if (!masterParam || !advParam || !_state) {
            return;
        }
        double master = 0.0;
        masterParam->getValue(master);
        if (!std::isfinite(master)) {
            return;
        }
        master = std::clamp(master, lo, hi);

        double r = 0.0, g = 0.0, b = 0.0;
        advParam->getValue(r, g, b);
        r = sanitize_scalar(r, master, lo, hi);
        g = sanitize_scalar(g, master, lo, hi);
        b = sanitize_scalar(b, master, lo, hi);

        std::array<double, 3> ratio = fallbackRatio;
        const double mean = (r + g + b) / 3.0;
        if (std::isfinite(mean) && mean > 0.0) {
            ratio[0] = r / mean;
            ratio[1] = g / mean;
            ratio[2] = b / mean;
        }

        r = std::clamp(master * ratio[0], lo, hi);
        g = std::clamp(master * ratio[1], lo, hi);
        b = std::clamp(master * ratio[2], lo, hi);
        const bool wasSuppressed = _state->suppressParamEvents;
        _state->suppressParamEvents = true;
        advParam->setValue(r, g, b);
        _state->suppressParamEvents = wasSuppressed;
        masterCache = master;
    };

    if (userEdit) {
        if (paramName == JuicerParams::kGrainAmplitude ||
            paramName == JuicerParams::kGrainBlur ||
            paramName == JuicerParams::kGrainSharpness ||
            paramName == JuicerParams::kGrainChroma ||
            paramName == JuicerParams::kGrainTexture ||
            paramName == JuicerParams::kGrainSublayersActive ||
            paramName == JuicerParams::kGrainParticleAreaUm2 ||
            paramName == JuicerParams::kGrainParticleScaleMaster ||
            paramName == JuicerParams::kGrainParticleScaleLayersMaster ||
            paramName == JuicerParams::kGrainDensityMinMaster ||
            paramName == JuicerParams::kGrainUniformityMaster ||
            paramName == JuicerParams::kGrainParticleScale ||
            paramName == JuicerParams::kGrainParticleScaleLayers ||
            paramName == JuicerParams::kGrainDensityMin ||
            paramName == JuicerParams::kGrainUniformity ||
            paramName == JuicerParams::kGrainBlurDyeCloudsUm ||
            paramName == JuicerParams::kGrainSizeMixWeight ||
            paramName == JuicerParams::kGrainSizeMixWeightMid ||
            paramName == JuicerParams::kGrainSizeMixScale ||
            paramName == JuicerParams::kGrainMicroStructure ||
            paramName == JuicerParams::kGrainClumpTemporalMix ||
            paramName == JuicerParams::kGrainClumpMorphPeriodSec) {
            updateGrainPresetLabel(true);
        }
    }

    updateGrainChromaEnabled();

    if (paramName == JuicerParams::kHalationStrengthMaster) {
        apply_master_delta(_pHalationStrengthMaster, _pHalationStrength, _halationStrengthMasterLast, 0.0, 100.0);
    }
    else if (paramName == JuicerParams::kHalationSizeUmMaster) {
        apply_master_delta(_pHalationSizeUmMaster, _pHalationSizeUm, _halationSizeUmMasterLast, 0.0, 1000.0);
    }
    else if (paramName == JuicerParams::kHalationScatteringStrengthMaster) {
        apply_master_delta(_pHalationScatteringStrengthMaster, _pHalationScatteringStrength, _halationScatteringStrengthMasterLast, 0.0, 100.0);
    }
    else if (paramName == JuicerParams::kHalationScatteringSizeUmMaster) {
        apply_master_delta(_pHalationScatteringSizeUmMaster, _pHalationScatteringSizeUm, _halationScatteringSizeUmMasterLast, 0.0, 1000.0);
    }
    else if (userEdit && (paramName == JuicerParams::kGrainParticleScaleMaster ||
        paramName == JuicerParams::kGrainParticleScaleLayersMaster ||
        paramName == JuicerParams::kGrainDensityMinMaster ||
        paramName == JuicerParams::kGrainUniformityMaster)) {
        std::array<double, 3> scaleRatio = default_particle_scale_ratio();
        std::array<double, 3> scaleLayersRatio = default_particle_scale_layers_ratio();
        std::array<double, 3> densityMinRatio = default_density_min_ratio();
        std::array<double, 3> uniformityRatio = default_uniformity_ratio();
        normalize_ratio(scaleRatio);
        normalize_ratio(scaleLayersRatio);
        normalize_ratio(densityMinRatio);
        normalize_ratio(uniformityRatio);

        if (paramName == JuicerParams::kGrainParticleScaleMaster) {
            apply_ratio_master(_pGrainParticleScaleMaster, _pGrainParticleScale, scaleRatio, _grainParticleScaleMasterLast, 0.0, 10.0);
        }
        else if (paramName == JuicerParams::kGrainParticleScaleLayersMaster) {
            apply_ratio_master(_pGrainParticleScaleLayersMaster, _pGrainParticleScaleLayers, scaleLayersRatio, _grainParticleScaleLayersMasterLast, 0.0, 10.0);
        }
        else if (paramName == JuicerParams::kGrainDensityMinMaster) {
            apply_ratio_master(_pGrainDensityMinMaster, _pGrainDensityMin, densityMinRatio, _grainDensityMinMasterLast, 0.0, 1.0);
        }
        else if (paramName == JuicerParams::kGrainUniformityMaster) {
            apply_ratio_master(_pGrainUniformityMaster, _pGrainUniformity, uniformityRatio, _grainUniformityMasterLast, 0.0, 1.0);
        }
    }

    if (userEdit && paramName == JuicerParams::kGrainSharpness && _pGrainSharpness && _pGrainBlurDyeCloudsUm && _state) {
        double sharpness = 0.5;
        _pGrainSharpness->getValue(sharpness);
        if (std::isfinite(sharpness)) {
            sharpness = std::clamp(sharpness, 0.0, 1.0);
            const double blurDyeClouds = std::clamp(grain_lerp(1.40, 0.60, sharpness), 0.0, 10.0);
            const bool wasSuppressed = _state->suppressParamEvents;
            _state->suppressParamEvents = true;
            _pGrainBlurDyeCloudsUm->setValue(blurDyeClouds);
            _state->suppressParamEvents = wasSuppressed;
        }
    }
    if (userEdit && paramName == JuicerParams::kGrainTexture && _pGrainTexture && _pGrainSizeMixWeight && _pGrainMicroStructure && _state) {
        double texture = 0.55;
        _pGrainTexture->getValue(texture);
        if (std::isfinite(texture)) {
            texture = std::clamp(texture, 0.0, 1.0);
            const double sizeMixWeight = std::clamp(grain_lerp(0.072, 0.38, texture), 0.0, 1.0);
            const double microCell = std::clamp(grain_lerp(50.0, 70.0, texture), 0.0, 1000.0);
            const double microSigma = std::clamp(grain_lerp(140.0, 200.0, texture), 0.0, 1000.0);
            const bool wasSuppressed = _state->suppressParamEvents;
            _state->suppressParamEvents = true;
            _pGrainSizeMixWeight->setValue(sizeMixWeight);
            _pGrainMicroStructure->setValue(microCell, microSigma);
            _state->suppressParamEvents = wasSuppressed;
        }
    }
    onParamsPossiblyChanged(paramName.c_str());
}

ParamSnapshot JuicerEffect::snapshotParams() const {
    ParamSnapshot P;
    if (_pFilmStock)      _pFilmStock->getValue(P.filmStockIndex);
    if (_pPrintPaper)     _pPrintPaper->getValue(P.printPaperIndex);
    if (_pSpectralMode)   _pSpectralMode->getValue(P.spectralUpsamplingMode);
    if (_pRefIll)         _pRefIll->getValue(P.refIll);
    if (_pEnlIll)         _pEnlIll->getValue(P.enlIll);
    if (_pEnlDichroicSet) _pEnlDichroicSet->getValue(P.enlDichroicSet);
    if (_pGlareCompRemovalFactor) {
        double v = P.glareCompRemovalFactor;
        _pGlareCompRemovalFactor->getValue(v);
        if (!std::isfinite(v)) v = 0.0;
        P.glareCompRemovalFactor = std::clamp(v, 0.0, 1.0);
    }
    if (_pGlareCompRemovalDensity) {
        double v = P.glareCompRemovalDensity;
        _pGlareCompRemovalDensity->getValue(v);
        if (!std::isfinite(v)) v = 1.2;
        P.glareCompRemovalDensity = std::clamp(v, 0.0, 3.0);
    }
    if (_pGlareCompRemovalTransition) {
        double v = P.glareCompRemovalTransition;
        _pGlareCompRemovalTransition->getValue(v);
        if (!std::isfinite(v)) v = 0.3;
        P.glareCompRemovalTransition = std::clamp(v, 0.0, 2.0);
    }
    if (_pPrintDminFactor) {
        double v = P.printDminFactor;
        _pPrintDminFactor->getValue(v);
        if (!std::isfinite(v)) v = 0.4;
        P.printDminFactor = std::clamp(v, 0.0, 1.0);
    }
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
    if (_pScannerLensBlur) _pScannerLensBlur->getValue(P.scannerLensBlurSigmaPx);
    if (_pScannerUnsharp) {
        double sigma = P.scannerUnsharpMask[0];
        double amount = P.scannerUnsharpMask[1];
        _pScannerUnsharp->getValue(sigma, amount);
        P.scannerUnsharpMask = { sigma, amount };
    }
    if (_pScannerUseLut) { bool v = true; _pScannerUseLut->getValue(v); P.scannerUseLut = v ? 1 : 0; }
    if (_pScannerLutResolution) _pScannerLutResolution->getValue(P.scannerLutResolution);
    if (_pOutputColorSpace) _pOutputColorSpace->getValue(P.outputColorSpace);
    if (_pOutputCctfEncoding) { bool v = true; _pOutputCctfEncoding->getValue(v); P.outputCctfEncoding = v ? 1 : 0; }
    if (_pOutputLinearPassThrough) { bool v = false; _pOutputLinearPassThrough->getValue(v); P.outputLinearPassThrough = v ? 1 : 0; }
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
    _state->printRT.midNeutralDensity = _state->printRT.profile.midNeutralDensity;
    _state->printRT.hasMidNeutralLogE = _state->printRT.profile.hasMidNeutralLogE;
    _state->printRT.midNeutralLogE = _state->printRT.profile.midNeutralLogE;

    // Load film stock before applying metadata-driven illuminant defaults
    _state->baseLoaded = load_film_stock_into_base(P.filmStockIndex, *_state);
    if (_state->baseLoaded) {
        applyHalationProfileDefaults();
    }

    // Apply metadata-driven illuminant defaults and rebuild runtime illuminants
    applyMetadataIlluminantDefaults(P);
    Print::build_illuminant_from_choice(P.enlIll, _state->printRT, _state->dataDir, /*forEnlarger*/true);

    // Load dichroic filters (set selection controls which vendor curves are used).
    const std::string dichroicDir = ensure_trailing_separator(
        data_dir_string("filters", "dichroics", dichroic_dir_name_for_choice(P.enlDichroicSet)));
    try {
        Print::load_dichroic_filters_from_csvs(dichroicDir, _state->printRT);
    }
    catch (const std::exception& ex) {
        // Identity fallback is already handled in loader via 1.0 curves
        JTRACE("PRINT", std::string("dichroic load failed at '") + dichroicDir + "' (" + ex.what() + "); using identity filters");
    }
    catch (...) {
        JTRACE("PRINT", std::string("dichroic load failed at '") + dichroicDir + "' (unknown error); using identity filters");
    }

    applyNeutralFilters(P);

    if (_state->baseLoaded) {
#ifdef JUICER_ENABLE_COUPLERS
        applyCouplerProfileDefaults(P);
#endif
        rebuild_working_state(this->getHandle(), *_state, P);
    }
    else {
        JTRACE("STOCK", "bootstrap: failed to load film stock; deferring rebuild");
    }

    // Re-enable changedParam handling now that bootstrap is complete
    _state->suppressParamEvents = false;
    _state->inBootstrap = false;
}

void JuicerEffect::applyNeutralFilters(const ParamSnapshot& P) {
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

    const std::string jsonPathPrimary = data_dir_string("profiles", enlarger_neutral_filters_json_for_choice(P.enlDichroicSet));
    const std::string jsonPathFallback = data_dir_string("profiles", "enlarger_neutral_ymc_filters.json");
    std::tuple<float, float, float> ymc{};
    for (const std::string& illumKey : illumKeys) {
        if (illumKey.empty()) {
            continue;
        }
        if (load_enlarger_neutral_filters(jsonPathPrimary, paperKey, illumKey, negativeKey, ymc, NeutralFilterThreadClass::Control) ||
            (jsonPathPrimary != jsonPathFallback &&
                load_enlarger_neutral_filters(jsonPathFallback, paperKey, illumKey, negativeKey, ymc, NeutralFilterThreadClass::Control))) {
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
    // Preserve user-entered enlarger offsets and exposure toggle; neutral baselines update independently.

}

bool JuicerEffect::applyMetadataIlluminantDefaults(ParamSnapshot& P) {
    if (!_state) {
        return false;
    }

    bool changed = false;

    const std::string& filmRef = !_state->filmReferenceIlluminant.empty()
        ? _state->filmReferenceIlluminant
        : _state->base.referenceIlluminant;
    const std::string& printRef = _state->printRT.referenceIlluminant;
    const std::string& printView = _state->printRT.viewingIlluminant;

    std::string refSource = !filmRef.empty() ? filmRef : (!printRef.empty() ? printRef : printView);
    std::string enlSource = !printRef.empty() ? printRef : (!filmRef.empty() ? filmRef : printView);

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

    if (!_state->couplerDirty.active.load(std::memory_order_acquire)) {
        const bool active = dirCfg.active;
        if (_pCouplersActive) {
            _pCouplersActive->setValue(active);
        }
        P.couplersActive = active ? 1 : 0;
    }

    if (!_state->couplerDirty.amount.load(std::memory_order_acquire)) {
        const double amount = sanitize_range(dirCfg.amount, P.couplersAmount, 0.0, 2.0);
        if (_pCouplersAmount) {
            _pCouplersAmount->setValue(amount);
        }
        P.couplersAmount = amount;
    }

    if (!_state->couplerDirty.ratioB.load(std::memory_order_acquire)) {
        const double ratioB = sanitize_range(dirCfg.ratioRGB[0], P.ratioB, 0.0, 1.0);
        if (_pCouplersAmountB) {
            _pCouplersAmountB->setValue(ratioB);
        }
        P.ratioB = ratioB;
    }

    if (!_state->couplerDirty.ratioG.load(std::memory_order_acquire)) {
        const double ratioG = sanitize_range(dirCfg.ratioRGB[1], P.ratioG, 0.0, 1.0);
        if (_pCouplersAmountG) {
            _pCouplersAmountG->setValue(ratioG);
        }
        P.ratioG = ratioG;
    }

    if (!_state->couplerDirty.ratioR.load(std::memory_order_acquire)) {
        const double ratioR = sanitize_range(dirCfg.ratioRGB[2], P.ratioR, 0.0, 1.0);
        if (_pCouplersAmountR) {
            _pCouplersAmountR->setValue(ratioR);
        }
        P.ratioR = ratioR;
    }

    if (!_state->couplerDirty.sigma.load(std::memory_order_acquire)) {
        const double sigma = sanitize_range(dirCfg.diffusionInterlayer, P.sigma, 0.0, 4.0);
        if (_pCouplersSigma) {
            _pCouplersSigma->setValue(sigma);
        }
        P.sigma = sigma;
    }

    if (!_state->couplerDirty.high.load(std::memory_order_acquire)) {
        const double high = sanitize_range(dirCfg.highExposureShift, P.high, 0.0, 1.0);
        if (_pCouplersHigh) {
            _pCouplersHigh->setValue(high);
        }
        P.high = high;
    }

    if (!_state->couplerDirty.spatialSigma.load(std::memory_order_acquire)) {
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
    if (!_state->baseLoaded) {
        bootstrap_after_attach();
    }

    ParamSnapshot P = snapshotParams();
    if (JTRACE_ENABLED(3)) {
        const char* paperKey = print_paper_json_key_for_index(P.printPaperIndex);
        const char* filmKey = negative_json_key_for_stock_index(P.filmStockIndex);
        const std::shared_ptr<const WorkingState> wsDbg = JuicerAtomic::load_shared_ptr(&_state->activeWorkingState);
        const std::uint64_t activeBuild = wsDbg ? wsDbg->buildCounter : 0;
        const std::uint64_t lastHash = _state->lastHash.load(std::memory_order_acquire);
        std::string msg = std::string("params change name=") + (changedNameOrNull ? changedNameOrNull : "<null>")
            + " printIndex=" + std::to_string(P.printPaperIndex)
            + " printKey=" + std::string(paperKey ? paperKey : "<null>")
            + " filmIndex=" + std::to_string(P.filmStockIndex)
            + " filmKey=" + std::string(filmKey ? filmKey : "<null>")
            + " activeBuild=" + std::to_string(activeBuild)
            + " lastHash=" + std::to_string(lastHash);
        JTRACE_VERBOSE("PRINTDBG", msg);
    }
#ifdef JUICER_ENABLE_COUPLERS
    if (changedNameOrNull) {
        if (std::strcmp(changedNameOrNull, Couplers::kParamCouplersActive) == 0) {
            _state->couplerDirty.active.store(true, std::memory_order_release);
        }
        else if (std::strcmp(changedNameOrNull, Couplers::kParamCouplersAmount) == 0) {
            _state->couplerDirty.amount.store(true, std::memory_order_release);
        }
        else if (std::strcmp(changedNameOrNull, Couplers::kParamCouplersAmountB) == 0) {
            _state->couplerDirty.ratioB.store(true, std::memory_order_release);
        }
        else if (std::strcmp(changedNameOrNull, Couplers::kParamCouplersAmountG) == 0) {
            _state->couplerDirty.ratioG.store(true, std::memory_order_release);
        }
        else if (std::strcmp(changedNameOrNull, Couplers::kParamCouplersAmountR) == 0) {
            _state->couplerDirty.ratioR.store(true, std::memory_order_release);
        }
        else if (std::strcmp(changedNameOrNull, Couplers::kParamCouplersLayerSigma) == 0) {
            _state->couplerDirty.sigma.store(true, std::memory_order_release);
        }
        else if (std::strcmp(changedNameOrNull, Couplers::kParamCouplersHighExpShift) == 0) {
            _state->couplerDirty.high.store(true, std::memory_order_release);
        }
        else if (std::strcmp(changedNameOrNull, Couplers::kParamCouplersSpatialSigma) == 0) {
            _state->couplerDirty.spatialSigma.store(true, std::memory_order_release);
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
        if (JTRACE_ENABLED(3)) {
            std::string msg = std::string("print reload key=") + std::string(paperKey ? paperKey : "<null>")
                + " dir=" + printDir
                + " json=" + printProfileJson
                + " ref=" + _state->printRT.referenceIlluminant
                + " view=" + _state->printRT.viewingIlluminant;
            JTRACE_VERBOSE("PRINTDBG", msg);
        }

        // Reload dichroic filters (vendor selection controls which curves are used).
        const std::string dichroicDirReload = ensure_trailing_separator(
            data_dir_string("filters", "dichroics", dichroic_dir_name_for_choice(P.enlDichroicSet)));
        try {
            Print::load_dichroic_filters_from_csvs(dichroicDirReload, _state->printRT);
        }
        catch (const std::exception& ex) {
            JTRACE("PRINT", std::string("dichroic reload failed at '") + dichroicDirReload + "' (" + ex.what() + "); identity filters remain active");
        }
        catch (...) {
            JTRACE("PRINT", std::string("dichroic reload failed at '") + dichroicDirReload + "' (unknown error); identity filters remain active");
        }

        printReloaded = true;
    }

    bool dichroicReloaded = false;
    if (changedNameOrNull && std::strcmp(changedNameOrNull, kParamEnlargerDichroicSet) == 0) {
        const std::string dichroicDirReload = ensure_trailing_separator(
            data_dir_string("filters", "dichroics", dichroic_dir_name_for_choice(P.enlDichroicSet)));
        try {
            Print::load_dichroic_filters_from_csvs(dichroicDirReload, _state->printRT);
            dichroicReloaded = true;
        }
        catch (const std::exception& ex) {
            JTRACE("PRINT", std::string("dichroic reload failed at '") + dichroicDirReload + "' (" + ex.what() + "); identity filters remain active");
        }
        catch (...) {
            JTRACE("PRINT", std::string("dichroic reload failed at '") + dichroicDirReload + "' (unknown error); identity filters remain active");
        }
    }

    bool filmReloaded = false;
    if (changedNameOrNull && std::strcmp(changedNameOrNull, kParamFilmStock) == 0) {
        _state->baseLoaded = load_film_stock_into_base(P.filmStockIndex, *_state);
        filmReloaded = _state->baseLoaded;
    }
    if (filmReloaded) {
        applyHalationProfileDefaults();
    }

    if (printReloaded || filmReloaded) {
        applyMetadataIlluminantDefaults(P);
        Print::build_illuminant_from_choice(P.enlIll, _state->printRT, _state->dataDir, /*forEnlarger*/true);
    }

    bool neutralApplied = false;
    if (printReloaded || dichroicReloaded) {
        applyNeutralFilters(P);
        neutralApplied = true;
        if (JTRACE_ENABLED(3)) {
            const char* paperKey = print_paper_json_key_for_index(P.printPaperIndex);
            const char* filmKey = negative_json_key_for_stock_index(P.filmStockIndex);
            std::string msg = std::string("neutral filters applied (print/dichroic) paper=") + std::string(paperKey ? paperKey : "<null>")
                + " film=" + std::string(filmKey ? filmKey : "<null>")
                + " Y/M/C=" + std::to_string(_state->printRT.neutralY)
                + "/" + std::to_string(_state->printRT.neutralM)
                + "/" + std::to_string(_state->printRT.neutralC);
            JTRACE_VERBOSE("PRINTDBG", msg);
        }
    }
    if (filmReloaded && !neutralApplied) {
        applyNeutralFilters(P);
        neutralApplied = true;
        if (JTRACE_ENABLED(3)) {
            const char* paperKey = print_paper_json_key_for_index(P.printPaperIndex);
            const char* filmKey = negative_json_key_for_stock_index(P.filmStockIndex);
            std::string msg = std::string("neutral filters applied (film) paper=") + std::string(paperKey ? paperKey : "<null>")
                + " film=" + std::string(filmKey ? filmKey : "<null>")
                + " Y/M/C=" + std::to_string(_state->printRT.neutralY)
                + "/" + std::to_string(_state->printRT.neutralM)
                + "/" + std::to_string(_state->printRT.neutralC);
            JTRACE_VERBOSE("PRINTDBG", msg);
        }
    }

    if (changedNameOrNull && std::strcmp(changedNameOrNull, kParamEnlargerIlluminant) == 0) {
        applyNeutralFilters(P);
    }

    // Rebuild if any effective param changed
    if (_state->baseLoaded) {
#ifdef JUICER_ENABLE_COUPLERS
        applyCouplerProfileDefaults(P);
#endif
    }

    const std::uint64_t fullHash = hash_params(P);
    const std::uint64_t coreHash = hash_params_core(P);
    const std::uint64_t dirHash = hash_params_dir(P);
    {
        std::lock_guard<std::mutex> lock(_state->pending.m);
        _state->pending.params = P;
        _state->pending.fullHash = fullHash;
        _state->pending.coreHash = coreHash;
        _state->pending.dirHash = dirHash;
        std::uint64_t next = _state->pending.seq + 1;
        _state->pending.seq = (next == 0) ? 1 : next;
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
            Couplers::on_param_changed(changedNameOrNull);
        }
    }
#endif
}
