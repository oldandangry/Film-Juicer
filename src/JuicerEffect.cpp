#include "JuicerEffect.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <atomic>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "JuicerState.h"
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
#include "Logging.h"
#include "Hash.h"
#include "mainProcessing.h"
#include "WorkingState.h"

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
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

    std::string profile_json_path_for_key_or_empty(const char* jsonKey) {
        if (!jsonKey) {
            return {};
        }
        std::string profileName(jsonKey);
        profileName += ".json";
        return data_dir_string("profiles", profileName);
    }

    void trace_dichroic_load_failure(
        const char* operation,
        const std::string& dichroicDir,
        const char* detail,
        const char* fallbackState) {

        if (!JTRACE_ENABLED(1)) {
            return;
        }
        const char* errorDetail = detail ? detail : "unknown error";
        std::string msg;
        msg.reserve(160 + dichroicDir.size());
        msg = operation ? operation : "dichroic load failed";
        msg += " at '";
        msg += dichroicDir;
        msg += "' (";
        msg += errorDetail;
        msg += "); ";
        msg += (fallbackState ? fallbackState : "using identity filters");
        JTRACE("PRINT", msg);
    }

    inline bool nearly_equal_double(double a, double b) {
        const double diff = std::fabs(a - b);
        const double scale = std::max({ 1.0, std::fabs(a), std::fabs(b) });
        return diff <= scale * 1e-9;
    }

    inline bool is_finite(float value) {
        return std::isfinite(value);
    }

    inline bool is_finite(double value) {
        return std::isfinite(value);
    }

    inline double sanitize_finite_clamped(double value, double fallback, double minValue, double maxValue) {
        if (!is_finite(value)) return fallback;
        return std::clamp(value, minValue, maxValue);
    }

    inline double sanitize_finite_or(double value, double fallback) {
        return is_finite(value) ? value : fallback;
    }

    inline double sanitize_positive_finite_or(double value, double fallback) {
        return (is_finite(value) && value > 0.0) ? value : fallback;
    }

    inline bool is_positive_finite(double value) {
        return is_finite(value) && value > 0.0;
    }

    inline float sanitize_nonnegative_finite_or(float value, float fallback) {
        return (is_finite(value) && value >= 0.0f) ? value : fallback;
    }

    inline double read_sanitized_double(
        OFX::DoubleParam* param,
        double fallback,
        double minValue,
        double maxValue)
    {
        double value = fallback;
        if (param) {
            param->getValue(value);
        }
        return sanitize_finite_clamped(value, fallback, minValue, maxValue);
    }

    inline std::array<double, 3> read_sanitized_double3(
        OFX::Double3DParam* param,
        const std::array<double, 3>& defaults,
        double minValue,
        double maxValue)
    {
        std::array<double, 3> values = defaults;
        if (param) {
            param->getValue(values[0], values[1], values[2]);
        }
        double* valueIt = values.data();
        const double* defaultIt = defaults.data();
        for (int i = 0; i < 3; ++i, ++valueIt, ++defaultIt) {
            *valueIt = sanitize_finite_clamped(*valueIt, *defaultIt, minValue, maxValue);
        }
        return values;
    }

    inline std::array<double, 2> read_sanitized_double2(
        OFX::Double2DParam* param,
        const std::array<double, 2>& defaults,
        double minValue,
        double maxValue)
    {
        std::array<double, 2> values = defaults;
        if (param) {
            param->getValue(values[0], values[1]);
        }
        double* valueIt = values.data();
        const double* defaultIt = defaults.data();
        for (int i = 0; i < 2; ++i, ++valueIt, ++defaultIt) {
            *valueIt = sanitize_finite_clamped(*valueIt, *defaultIt, minValue, maxValue);
        }
        return values;
    }

    inline int pixel_component_count(OFX::PixelComponentEnum comps) {
        switch (comps) {
        case OFX::ePixelComponentRGBA: return 4;
        case OFX::ePixelComponentRGB: return 3;
        case OFX::ePixelComponentAlpha: return 1;
        default: return 0;
        }
    }

    inline bool span_x_within_bounds(int xStart, int xEnd, const OfxRectI& bounds) {
        return xStart >= bounds.x1 && xEnd <= bounds.x2;
    }

    inline bool row_has_full_coverage(const OfxRectI& bounds, int xStart, int xEnd, int y) {
        return span_x_within_bounds(xStart, xEnd, bounds) &&
            y >= bounds.y1 && y < bounds.y2;
    }

    template <typename T>
    inline T* row_ptr_if_fully_covered(
        OFX::Image* image,
        const OfxRectI& bounds,
        int xStart,
        int xEnd,
        int y) {
        if (!row_has_full_coverage(bounds, xStart, xEnd, y)) {
            return nullptr;
        }
        return reinterpret_cast<T*>(image->getPixelAddress(xStart, y));
    }

    inline const float* row_start_if_covered(
        OFX::Image* image,
        const OfxRectI& srcBounds,
        const OfxRectI& meterBounds,
        int y) {
        return row_ptr_if_fully_covered<const float>(
            image,
            srcBounds,
            meterBounds.x1,
            meterBounds.x2,
            y);
    }

    template <typename T>
    inline T* pixel_ptr(OFX::Image* image, int x, int y) {
        return reinterpret_cast<T*>(image->getPixelAddress(x, y));
    }

    inline void decode_input_pixel_linear(
        const float* pix,
        bool singleComponent,
        Spectral::InputColorSpace inputColorSpace,
        bool applyCctfDecoding,
        float linear[3]) {
        if (singleComponent) {
            const float gray = pix[0];
            const float grayRgb[3] = { gray, gray, gray };
            Spectral::apply_input_cctf_decoding(inputColorSpace, applyCctfDecoding, grayRgb, linear);
            return;
        }
        Spectral::apply_input_cctf_decoding(inputColorSpace, applyCctfDecoding, pix, linear);
    }

    inline void set_optional_sum_mask(double* outSumMask, double value) {
        if (outSumMask) {
            *outSumMask = value;
        }
    }

    inline double weighted_mean_or_zero(double sumY, double sumMask) {
        return (sumMask > 0.0) ? (sumY / sumMask) : 0.0;
    }

    inline float clamp_nonnegative(float value) {
        return (value < 0.0f) ? 0.0f : value;
    }

    inline double gaussian_weight(double normX, double normY, double invSigmaDenom) {
        const double r2 = normX * normX + normY * normY;
        return std::exp(-r2 * invSigmaDenom);
    }

    inline float finite_exp2_scale(double ev) {
        const float scale = static_cast<float>(std::exp2(ev));
        return is_finite(scale) ? scale : 1.0f;
    }

    inline void sanitize_dir_matrix(float matrix[3][3]) {
        float* valueIt = &matrix[0][0];
        const float* const valueEnd = valueIt + 9;
        for (; valueIt < valueEnd; ++valueIt) {
            float value = *valueIt;
            if (!is_finite(value)) value = 0.0f;
            if (value < -10.0f) value = -10.0f;
            if (value > 10.0f) value = 10.0f;
            *valueIt = value;
        }
    }

    inline bool has_nonzero_finite_dir_matrix(const float matrix[3][3]) {
        const float* valueIt = &matrix[0][0];
        const float* const valueEnd = valueIt + 9;
        for (; valueIt < valueEnd; ++valueIt) {
            const float value = *valueIt;
            if (is_finite(value) && value != 0.0f) {
                return true;
            }
        }
        return false;
    }

    constexpr std::uint64_t kAutoExposureMaskCacheMaxBytes = 96ull * 1024ull * 1024ull;
    constexpr std::size_t kAutoExposureMedianScratchMaxSamples =
        static_cast<std::size_t>(kAutoExposureMaskCacheMaxBytes / sizeof(float));
    static std::atomic<std::uint64_t> gAutoExposureMaskCacheResidentBytes{ 0 };

    std::vector<float>& auto_exposure_median_scratch() {
        thread_local std::vector<float> scratch;
        return scratch;
    }

    inline std::uint64_t mask_bytes_for_dimensions(int width, int height) {
        if (width <= 0 || height <= 0) {
            return 0;
        }
        const std::uint64_t w = static_cast<std::uint64_t>(width);
        const std::uint64_t h = static_cast<std::uint64_t>(height);
        if (w > (std::numeric_limits<std::uint64_t>::max() / h)) {
            return 0;
        }
        const std::uint64_t samples = w * h;
        const std::uint64_t sampleBytes = static_cast<std::uint64_t>(sizeof(double));
        if (samples > (std::numeric_limits<std::uint64_t>::max() / sampleBytes)) {
            return 0;
        }
        return samples * sampleBytes;
    }

    inline void update_auto_exposure_mask_resident_bytes(
        std::uint64_t previousBytes,
        std::uint64_t nextBytes) {

        if (nextBytes > previousBytes) {
            gAutoExposureMaskCacheResidentBytes.fetch_add(nextBytes - previousBytes, std::memory_order_relaxed);
        }
        else if (previousBytes > nextBytes) {
            const std::uint64_t delta = previousBytes - nextBytes;
            std::uint64_t observed = gAutoExposureMaskCacheResidentBytes.load(std::memory_order_relaxed);
            while (true) {
                const std::uint64_t updated = (observed > delta) ? (observed - delta) : 0;
                if (gAutoExposureMaskCacheResidentBytes.compare_exchange_weak(
                    observed,
                    updated,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed)) {
                    break;
                }
            }
        }
    }

    inline void trace_auto_exposure_mask_cache_event(
        const InstanceState* state,
        const char* event,
        int width,
        int height,
        std::uint64_t requestedBytes,
        std::uint64_t cachedBytes,
        std::uint64_t previousCachedBytes,
        const char* reason) {

        if (!JTRACE_ENABLED(2)) {
            return;
        }
        const std::uint64_t residentBytes =
            gAutoExposureMaskCacheResidentBytes.load(std::memory_order_relaxed);
        std::string msg;
        msg.reserve(256);
        msg = "event=";
        msg += (event ? event : "unknown");
        msg += " instance_token=";
        msg += std::to_string(state ? state->instanceToken : 0);
        msg += " width=";
        msg += std::to_string(width);
        msg += " height=";
        msg += std::to_string(height);
        msg += " requested_bytes=";
        msg += std::to_string(requestedBytes);
        msg += " previous_cached_bytes=";
        msg += std::to_string(previousCachedBytes);
        msg += " cached_bytes=";
        msg += std::to_string(cachedBytes);
        msg += " resident_bytes=";
        msg += std::to_string(residentBytes);
        msg += " cap_bytes=";
        msg += std::to_string(kAutoExposureMaskCacheMaxBytes);
        if (reason && reason[0] != '\0') {
            msg += " reason=";
            msg += reason;
        }
        JTRACE_LEVEL(2, "MSAEM", msg);
    }

    static double build_center_weight_mask(int width, int height, double sigma, std::vector<double>& outMask) {
        outMask.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
        if (!is_finite(sigma) || sigma <= 0.0) {
            std::fill(outMask.begin(), outMask.end(), 0.0);
            return 0.0;
        }
        const double maxDim = static_cast<double>(std::max(width, height));
        const double invMax = (maxDim > 0.0) ? (1.0 / maxDim) : 0.0;
        const double invWidth = (width > 0) ? (1.0 / static_cast<double>(width)) : 0.0;
        const double invHeight = (height > 0) ? (1.0 / static_cast<double>(height)) : 0.0;
        const double scaleX = static_cast<double>(width) * invMax;
        const double scaleY = static_cast<double>(height) * invMax;
        const double sigmaDenom = 2.0 * sigma * sigma;
        const double invSigmaDenom = 1.0 / sigmaDenom;

        double sumMask = 0.0;
        for (int y = 0; y < height; ++y) {
            double* row = outMask.data() + static_cast<size_t>(y) * static_cast<size_t>(width);
            double* rowIt = row;
            const double ny = static_cast<double>(y) * invHeight - 0.5;
            const double normY = ny * scaleY;
            for (int x = 0; x < width; ++x) {
                const double nx = static_cast<double>(x) * invWidth - 0.5;
                const double normX = nx * scaleX;
                const double w = gaussian_weight(normX, normY, invSigmaDenom);
                *rowIt++ = w;
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
        const OfxRectI srcBounds = img->getBounds();
        const int nComponents = pixel_component_count(img->getPixelComponents());
        if (nComponents <= 0) {
            return 0.0;
        }
        const std::size_t pixelStride = static_cast<std::size_t>(nComponents);
        const bool singleComponent = (nComponents == 1);
        const int xStart = bounds.x1;
        const int xEnd = bounds.x2;
        auto accumulate_weighted_Y = [&](const float* pix, double weight, double& sumY, double& sumMask) {
            float linear[3];
            decode_input_pixel_linear(
                pix,
                singleComponent,
                inputColorSpace,
                applyCctfDecoding,
                linear);
            float XYZ[3];
            rgbToXYZ.mul(linear, XYZ);
            const double Y = static_cast<double>(XYZ[1]);
            if (!is_finite(Y)) {
                return;
            }
            sumY += Y * weight;
            sumMask += weight;
            };
        auto accumulate_weighted_pixel_if_present = [&](
            int x,
            int y,
            double weight,
            double& sumY,
            double& sumMask) {
            const float* pix = pixel_ptr<const float>(img, x, y);
            if (!pix) {
                return;
            }
            accumulate_weighted_Y(pix, weight, sumY, sumMask);
            };

        auto accumulateYFromMask = [&](const std::vector<double>& mask, double* outSumMask) {
            double sumY = 0.0;
            double sumMask = 0.0;
            for (int yy = bounds.y1; yy < bounds.y2; ++yy) {
                const size_t rowOffset = static_cast<size_t>(yy - bounds.y1) * static_cast<size_t>(width);
                const double* maskRow = mask.data() + rowOffset;
                const float* rowPix = row_start_if_covered(img, srcBounds, bounds, yy);
                const double* maskIt = maskRow;
                if (rowPix) {
                    const float* rowPixIt = rowPix;
                    for (int xOff = 0; xOff < width; ++xOff) {
                        const float* pix = rowPixIt;
                        rowPixIt += pixelStride;
                        const double w = *maskIt++;
                        accumulate_weighted_Y(pix, w, sumY, sumMask);
                    }
                    continue;
                }
                for (int xx = xStart; xx < xEnd; ++xx) {
                    const double w = *maskIt++;
                    accumulate_weighted_pixel_if_present(xx, yy, w, sumY, sumMask);
                }
            }
            set_optional_sum_mask(outSumMask, sumMask);
            return weighted_mean_or_zero(sumY, sumMask);
        };

        auto accumulateYUncached = [&](double* outSumMask) {
            if (!is_finite(sigma) || sigma <= 0.0) {
                set_optional_sum_mask(outSumMask, 0.0);
                return 0.0;
            }

            const double maxDim = static_cast<double>(std::max(width, height));
            const double invMax = (maxDim > 0.0) ? (1.0 / maxDim) : 0.0;
            const double invWidth = 1.0 / static_cast<double>(width);
            const double invHeight = 1.0 / static_cast<double>(height);
            const double scaleX = static_cast<double>(width) * invMax;
            const double scaleY = static_cast<double>(height) * invMax;
            const double sigmaDenom = 2.0 * sigma * sigma;
            if (!is_finite(sigmaDenom) || sigmaDenom <= 0.0) {
                set_optional_sum_mask(outSumMask, 0.0);
                return 0.0;
            }
            const double invSigmaDenom = 1.0 / sigmaDenom;

            double sumY = 0.0;
            double sumMask = 0.0;
            for (int yy = bounds.y1; yy < bounds.y2; ++yy) {
                const int localY = yy - bounds.y1;
                const double ny = static_cast<double>(localY) * invHeight - 0.5;
                const double normY = ny * scaleY;
                const float* rowPix = row_start_if_covered(img, srcBounds, bounds, yy);
                auto accumulate_sigma_weighted_pixel_if_present = [&](
                    int x,
                    double nxValue,
                    double& sumYRef,
                    double& sumMaskRef) {
                    const float* pix = pixel_ptr<const float>(img, x, yy);
                    if (!pix) {
                        return;
                    }
                    const double normX = nxValue * scaleX;
                    const double w = gaussian_weight(normX, normY, invSigmaDenom);
                    accumulate_weighted_Y(pix, w, sumYRef, sumMaskRef);
                    };
                double nx = -0.5;
                if (rowPix) {
                    const float* rowPixIt = rowPix;
                    for (int xOff = 0; xOff < width; ++xOff) {
                        const float* pix = rowPixIt;
                        rowPixIt += pixelStride;
                        const double normX = nx * scaleX;
                        const double w = gaussian_weight(normX, normY, invSigmaDenom);
                        accumulate_weighted_Y(pix, w, sumY, sumMask);
                        nx += invWidth;
                    }
                    continue;
                }
                for (int xx = xStart; xx < xEnd; ++xx) {
                    accumulate_sigma_weighted_pixel_if_present(xx, nx, sumY, sumMask);
                    nx += invWidth;
                }
            }
            set_optional_sum_mask(outSumMask, sumMask);
            return weighted_mean_or_zero(sumY, sumMask);
        };

        if (!state) {
            return accumulateYUncached(nullptr);
        }

        const std::uint64_t requestedMaskBytes = mask_bytes_for_dimensions(width, height);
        const bool cacheEligible = (requestedMaskBytes > 0) && (requestedMaskBytes <= kAutoExposureMaskCacheMaxBytes);

        if (!cacheEligible) {
            bool emitBypassTrace = false;
            std::uint64_t previousCachedBytes = 0;
            {
                std::lock_guard<std::mutex> lock(state->autoExposureMutex);
                previousCachedBytes = state->autoExposureMaskCachedBytes;
                const bool wasBypass = state->autoExposureMaskPolicyBypass;
                const std::uint64_t previousRequestedBytes = state->autoExposureMaskLastRequestedBytes;

                if (previousCachedBytes > 0) {
                    update_auto_exposure_mask_resident_bytes(previousCachedBytes, 0);
                }
                state->autoExposureMaskWeights.reset();
                state->autoExposureMaskCachedBytes = 0;
                state->autoExposureMaskValid = false;
                state->autoExposureMaskWidth = width;
                state->autoExposureMaskHeight = height;
                state->autoExposureMaskSigma = sigma;
                state->autoExposureMaskRenderScaleX = renderScaleX;
                state->autoExposureMaskRenderScaleY = renderScaleY;
                state->autoExposureMaskClipToken = clipToken;
                state->autoExposureMaskSum = 0.0;
                state->autoExposureMaskPolicyBypass = true;
                state->autoExposureMaskLastRequestedBytes = requestedMaskBytes;

                emitBypassTrace = (previousCachedBytes > 0)
                    || !wasBypass
                    || (previousRequestedBytes != requestedMaskBytes);
            }

            if (emitBypassTrace) {
                trace_auto_exposure_mask_cache_event(
                    state,
                    "cache_bypass",
                    width,
                    height,
                    requestedMaskBytes,
                    0,
                    previousCachedBytes,
                    (requestedMaskBytes == 0) ? "overflow_or_invalid" : "over_cap");
            }

            double effectiveSumMask = 0.0;
            const double measuredY = accumulateYUncached(&effectiveSumMask);
            {
                std::lock_guard<std::mutex> lock(state->autoExposureMutex);
                if (state->autoExposureMaskPolicyBypass &&
                    state->autoExposureMaskLastRequestedBytes == requestedMaskBytes) {
                    state->autoExposureMaskSum = effectiveSumMask;
                }
            }
            return measuredY;
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
            bool emitStoreTrace = false;
            std::uint64_t previousCachedBytes = 0;
            {
                std::lock_guard<std::mutex> lock(state->autoExposureMutex);
                if (needsMaskRebuild(*state)) {
                    previousCachedBytes = state->autoExposureMaskCachedBytes;
                    update_auto_exposure_mask_resident_bytes(previousCachedBytes, requestedMaskBytes);
                    state->autoExposureMaskWidth = width;
                    state->autoExposureMaskHeight = height;
                    state->autoExposureMaskSigma = sigma;
                    state->autoExposureMaskRenderScaleX = renderScaleX;
                    state->autoExposureMaskRenderScaleY = renderScaleY;
                    state->autoExposureMaskClipToken = clipToken;
                    state->autoExposureMaskWeights = rebuiltMask;
                    state->autoExposureMaskCachedBytes = requestedMaskBytes;
                    state->autoExposureMaskValid = rebuiltValid;
                    state->autoExposureMaskSum = rebuiltValid ? sumMask : 0.0;
                    emitStoreTrace = (previousCachedBytes != requestedMaskBytes) || state->autoExposureMaskPolicyBypass;
                    state->autoExposureMaskPolicyBypass = false;
                    state->autoExposureMaskLastRequestedBytes = requestedMaskBytes;
                }
                maskSnapshot = state->autoExposureMaskWeights;
                maskValid = state->autoExposureMaskValid && static_cast<bool>(maskSnapshot);
                if (!maskValid) {
                    state->autoExposureMaskSum = 0.0;
                }
            }

            if (emitStoreTrace) {
                trace_auto_exposure_mask_cache_event(
                    state,
                    "cache_store",
                    width,
                    height,
                    requestedMaskBytes,
                    requestedMaskBytes,
                    previousCachedBytes,
                    rebuiltValid ? "rebuilt" : "rebuilt_invalid");
            }
        }

        if (!maskValid || !maskSnapshot || maskSnapshot->size() != expectedMaskSize) {
            return 0.0;
        }

        double effectiveSumMask = 0.0;
        const double measuredY = accumulateYFromMask(*maskSnapshot, &effectiveSumMask);
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
        const OfxRectI srcBounds = img->getBounds();

        const size_t total = static_cast<size_t>(width) * static_cast<size_t>(height);
        const int nComponents = pixel_component_count(img->getPixelComponents());
        if (nComponents <= 0) {
            return 0.0;
        }
        const std::size_t pixelStride = static_cast<std::size_t>(nComponents);
        const bool singleComponent = (nComponents == 1);
        std::vector<float>* valuesPtr = nullptr;
        std::vector<float> localValues;
        if (total <= kAutoExposureMedianScratchMaxSamples) {
            std::vector<float>& scratch = auto_exposure_median_scratch();
            scratch.clear();
            scratch.reserve(total);
            valuesPtr = &scratch;
        }
        else {
            localValues.reserve(total);
            valuesPtr = &localValues;
        }
        std::vector<float>& values = *valuesPtr;
        auto append_finite_luma = [&](const float* pix) {
            float linear[3];
            decode_input_pixel_linear(
                pix,
                singleComponent,
                inputColorSpace,
                applyCctfDecoding,
                linear);
            float XYZ[3];
            rgbToXYZ.mul(linear, XYZ);
            float Y = XYZ[1];
            if (!is_finite(Y)) {
                return;
            }
            values.emplace_back(clamp_nonnegative(Y));
            };
        auto append_luma_if_pixel_present = [&](int x, int y) {
            const float* pix = pixel_ptr<const float>(img, x, y);
            if (!pix) {
                return;
            }
            append_finite_luma(pix);
            };

        const int xStart = bounds.x1;
        const int xEnd = bounds.x2;
        for (int yy = bounds.y1; yy < bounds.y2; ++yy) {
            const float* rowPix = row_start_if_covered(img, srcBounds, bounds, yy);
            if (rowPix) {
                const float* rowPixIt = rowPix;
                for (int xOff = 0; xOff < width; ++xOff) {
                    const float* pix = rowPixIt;
                    rowPixIt += pixelStride;
                    append_finite_luma(pix);
                }
                continue;
            }
            int x = xStart;
            for (int xOff = 0; xOff < width; ++xOff, ++x) {
                append_luma_if_pixel_present(x, yy);
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

        const float low = *std::max_element(values.begin(), midIt);
        return static_cast<double>((low + high) * 0.5f);
    }

    void init_spectral_globals_once() {
        Spectral::SpectralMutationScope mutationScope(
            Spectral::SpectralMutationStage::Bootstrap,
            "init_spectral_globals_once");
        (void)mutationScope;

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
            const std::string* keyData = keys.data();
            const size_t keyCount = keys.size();
            for (size_t i = 0; i < keyCount; ++i, ++keyData) {
                if (IlluminantKeys::normalize(*keyData) == normalized) {
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
    exposureSliderEV = sanitize_finite_or(exposureSliderEV, 0.0);
    params.sliderEV = exposureSliderEV;
    params.sliderScale = finite_exp2_scale(exposureSliderEV);
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
    blurSigma = sanitize_finite_or(blurSigma, 0.55);
    blurSigma = std::clamp(blurSigma, 0.0, 10.0);
    opts.lensBlurSigmaPx = static_cast<float>(blurSigma);

    double unsharpSigma = opts.unsharpSigmaPx;
    double unsharpAmount = opts.unsharpAmount;
    if (_pScannerUnsharp) {
        _pScannerUnsharp->getValue(unsharpSigma, unsharpAmount);
    }
    unsharpSigma = sanitize_finite_or(unsharpSigma, 0.7);
    unsharpAmount = sanitize_finite_or(unsharpAmount, 1.0);
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
        if (!is_finite(v)) return 0.0;
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

    const std::array<double, 3> strengthPercent =
        read_sanitized_double3(_pHalationStrength, { {3.0, 0.30, 0.10} }, 0.0, 100.0);
    const std::array<double, 3> sizeUm =
        read_sanitized_double3(_pHalationSizeUm, { {200.0, 200.0, 200.0} }, 0.0, 1000.0);
    const std::array<double, 3> scatterStrengthPercent =
        read_sanitized_double3(_pHalationScatteringStrength, { {1.0, 2.0, 4.0} }, 0.0, 100.0);
    const std::array<double, 3> scatterSizeUm =
        read_sanitized_double3(_pHalationScatteringSizeUm, { {30.0, 20.0, 15.0} }, 0.0, 1000.0);

    float* strengthIt = halation.strength.data();
    float* sizeIt = halation.sizeUm.data();
    float* scatterStrengthIt = halation.scatteringStrength.data();
    float* scatterSizeIt = halation.scatteringSizeUm.data();
    const double* strengthSrc = strengthPercent.data();
    const double* sizeSrc = sizeUm.data();
    const double* scatterStrengthSrc = scatterStrengthPercent.data();
    const double* scatterSizeSrc = scatterSizeUm.data();
    for (int i = 0; i < 3; ++i,
         ++strengthIt, ++sizeIt, ++scatterStrengthIt, ++scatterSizeIt,
         ++strengthSrc, ++sizeSrc, ++scatterStrengthSrc, ++scatterSizeSrc) {
        *strengthIt = static_cast<float>((*strengthSrc) * 0.01);
        *sizeIt = static_cast<float>(*sizeSrc);
        *scatterStrengthIt = static_cast<float>((*scatterStrengthSrc) * 0.01);
        *scatterSizeIt = static_cast<float>(*scatterSizeSrc);
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

    const double strengthPctR = sanitize_finite_clamped(static_cast<double>(halationCfg.strength[0]) * 100.0, strengthR, 0.0, 100.0);
    const double strengthPctG = sanitize_finite_clamped(static_cast<double>(halationCfg.strength[1]) * 100.0, strengthG, 0.0, 100.0);
    const double strengthPctB = sanitize_finite_clamped(static_cast<double>(halationCfg.strength[2]) * 100.0, strengthB, 0.0, 100.0);
    const double sizeUmR = sanitize_finite_clamped(static_cast<double>(halationCfg.sizeUm[0]), sizeR, 0.0, 1000.0);
    const double sizeUmG = sanitize_finite_clamped(static_cast<double>(halationCfg.sizeUm[1]), sizeG, 0.0, 1000.0);
    const double sizeUmB = sanitize_finite_clamped(static_cast<double>(halationCfg.sizeUm[2]), sizeB, 0.0, 1000.0);
    const double scatterStrengthPctR = sanitize_finite_clamped(static_cast<double>(halationCfg.scatteringStrength[0]) * 100.0, scatterStrengthR, 0.0, 100.0);
    const double scatterStrengthPctG = sanitize_finite_clamped(static_cast<double>(halationCfg.scatteringStrength[1]) * 100.0, scatterStrengthG, 0.0, 100.0);
    const double scatterStrengthPctB = sanitize_finite_clamped(static_cast<double>(halationCfg.scatteringStrength[2]) * 100.0, scatterStrengthB, 0.0, 100.0);
    const double scatterSizeUmR = sanitize_finite_clamped(static_cast<double>(halationCfg.scatteringSizeUm[0]), scatterSizeR, 0.0, 1000.0);
    const double scatterSizeUmG = sanitize_finite_clamped(static_cast<double>(halationCfg.scatteringSizeUm[1]), scatterSizeG, 0.0, 1000.0);
    const double scatterSizeUmB = sanitize_finite_clamped(static_cast<double>(halationCfg.scatteringSizeUm[2]), scatterSizeB, 0.0, 1000.0);

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
        double* data = values.data();
        double sum = 0.0;
        const double* const dataEnd = data + values.size();
        for (; data < dataEnd; ++data) {
            const double v = *data;
            if (!is_positive_finite(v)) {
                values = { {1.0, 1.0, 1.0} };
                return;
            }
            sum += v;
        }
        const double mean = sum / 3.0;
        if (!is_positive_finite(mean)) {
            values = { {1.0, 1.0, 1.0} };
            return;
        }
        double* outData = values.data();
        const double* const outEnd = outData + values.size();
        for (; outData < outEnd; ++outData) {
            *outData /= mean;
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

    double amountEV = read_sanitized_double(_pGrainAmplitude, preset.amountEV, -3.0, 3.0);
    const double amplitude = std::exp2(amountEV);
    grain.amplitude = static_cast<float>(amplitude);

    double sizePx = read_sanitized_double(_pGrainBlur, preset.sizePx, 0.20, 2.00);
    grain.blur = static_cast<float>(sizePx);

    double sharpness = read_sanitized_double(_pGrainSharpness, preset.sharpness, 0.0, 1.0);

    double chroma = read_sanitized_double(_pGrainChroma, preset.chroma, 0.0, 1.0);

    double texture = read_sanitized_double(_pGrainTexture, preset.texture, 0.0, 1.0);

    double blurDyeCloudsBase = grain_lerp(1.40, 0.60, sharpness);
    double sizeMixWeightBase = sanitize_finite_or(preset.sizeMixWeight, grain_lerp(0.072, 0.38, texture));
    double microCellBase = sanitize_finite_or(preset.microCell, grain_lerp(50.0, 70.0, texture));
    double microSigmaBase = sanitize_finite_or(preset.microSigma, grain_lerp(140.0, 200.0, texture));

    double particleArea = read_sanitized_double(_pGrainParticleAreaUm2, preset.particleAreaUm2, 0.0, 10.0);
    grain.agxParticleAreaUm2 = static_cast<float>(particleArea);

    double sizeMixScale = read_sanitized_double(_pGrainSizeMixScale, preset.sizeMixScale, 1.0, 50.0);
    grain.sizeMixScale = static_cast<float>(sizeMixScale);

    double sizeMixWeight = read_sanitized_double(_pGrainSizeMixWeight, sizeMixWeightBase, 0.0, 1.0);
    grain.sizeMixWeight = static_cast<float>(sizeMixWeight);

    double sizeMixWeightMid = read_sanitized_double(_pGrainSizeMixWeightMid, 0.0, 0.0, 1.0);
    grain.sizeMixWeightMid = static_cast<float>(sizeMixWeightMid);

    double blurDyeClouds = read_sanitized_double(_pGrainBlurDyeCloudsUm, blurDyeCloudsBase, 0.0, 10.0);
    grain.blurDyeCloudsUm = static_cast<float>(blurDyeClouds);

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
    const std::array<double, 3> particleScale =
        read_sanitized_double3(_pGrainParticleScale, defaultParticleScale, 0.0, 10.0);
    const std::array<double, 3> particleScaleLayers =
        read_sanitized_double3(_pGrainParticleScaleLayers, defaultParticleScaleLayers, 0.0, 10.0);
    const std::array<double, 3> densityMin =
        read_sanitized_double3(_pGrainDensityMin, defaultDensityMin, 0.0, 1.0);
    const std::array<double, 3> uniformity =
        read_sanitized_double3(_pGrainUniformity, defaultUniformity, 0.0, 1.0);
    float* particleScaleDst = grain.agxParticleScale.data();
    float* particleScaleLayerDst = grain.agxParticleScaleLayers.data();
    float* densityMinDst = grain.densityMin.data();
    float* uniformityDst = grain.uniformity.data();
    const double* particleScaleSrc = particleScale.data();
    const double* particleScaleLayerSrc = particleScaleLayers.data();
    const double* densityMinSrc = densityMin.data();
    const double* uniformitySrc = uniformity.data();
    for (int i = 0; i < 3; ++i,
         ++particleScaleDst, ++particleScaleLayerDst, ++densityMinDst, ++uniformityDst,
         ++particleScaleSrc, ++particleScaleLayerSrc, ++densityMinSrc, ++uniformitySrc) {
        *particleScaleDst = static_cast<float>(*particleScaleSrc);
        *particleScaleLayerDst = static_cast<float>(*particleScaleLayerSrc);
        *densityMinDst = static_cast<float>(*densityMinSrc);
        *uniformityDst = static_cast<float>(*uniformitySrc);
    }

    double clumpTemporalMix = read_sanitized_double(_pGrainClumpTemporalMix, 0.30, 0.0, 0.30);
    grain.clumpTemporalMix = static_cast<float>(clumpTemporalMix);

    double clumpMorphPeriodSec = read_sanitized_double(_pGrainClumpMorphPeriodSec, 8.0, 5.0, 60.0);
    grain.clumpMorphPeriodSec = static_cast<float>(clumpMorphPeriodSec);

    std::array<double, 2> microStructure = { {microCellBase, microSigmaBase} };
    microStructure = read_sanitized_double2(_pGrainMicroStructure, microStructure, 0.0, 1000.0);
    float* microDst = grain.microStructure.data();
    const double* microSrc = microStructure.data();
    for (int i = 0; i < 2; ++i, ++microDst, ++microSrc) {
        *microDst = static_cast<float>(*microSrc);
    }

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

    double filmDust = read_sanitized_double(_pFilmDustAmount, 0.0, 0.0, 10.0);
    grain.filmDustAmount = static_cast<float>(filmDust);

    double gateDust = read_sanitized_double(_pGateDustAmount, 0.0, 0.0, 10.0);
    grain.gateDustAmount = static_cast<float>(gateDust);

    double filmScratch = read_sanitized_double(_pFilmScratchAmount, 0.0, 0.0, 10.0);
    grain.filmScratchAmount = static_cast<float>(filmScratch);

    double gateScratch = read_sanitized_double(_pGateScratchAmount, 0.0, 0.0, 10.0);
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
    const double sizeMixWeight = sanitize_finite_or(preset.sizeMixWeight, grain_lerp(0.072, 0.38, preset.texture));
    const double microCell = sanitize_finite_or(preset.microCell, grain_lerp(50.0, 70.0, preset.texture));
    const double microSigma = sanitize_finite_or(preset.microSigma, grain_lerp(140.0, 200.0, preset.texture));

    std::array<double, 3> scaleRatio = default_particle_scale_ratio();
    std::array<double, 3> scaleLayersRatio = default_particle_scale_layers_ratio();
    std::array<double, 3> densityMinRatio = default_density_min_ratio();
    std::array<double, 3> uniformityRatio = default_uniformity_ratio();
    normalize_ratio(scaleRatio);
    normalize_ratio(scaleLayersRatio);
    normalize_ratio(densityMinRatio);
    normalize_ratio(uniformityRatio);

    auto set_scaled_triplet = [](OFX::Double3DParam* param,
                                 double master,
                                 const std::array<double, 3>& ratio,
                                 double lo,
                                 double hi) {
        if (!param) {
            return;
        }
        param->setValue(
            std::clamp(master * ratio[0], lo, hi),
            std::clamp(master * ratio[1], lo, hi),
            std::clamp(master * ratio[2], lo, hi));
    };

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
    set_scaled_triplet(_pGrainParticleScale, preset.particleScaleMaster, scaleRatio, 0.0, 10.0);
    if (_pGrainParticleScaleLayersMaster) {
        _pGrainParticleScaleLayersMaster->setValue(preset.particleScaleLayersMaster);
        _grainParticleScaleLayersMasterLast = preset.particleScaleLayersMaster;
    }
    set_scaled_triplet(_pGrainParticleScaleLayers, preset.particleScaleLayersMaster, scaleLayersRatio, 0.0, 10.0);
    if (_pGrainDensityMinMaster) {
        _pGrainDensityMinMaster->setValue(preset.densityMinMaster);
        _grainDensityMinMasterLast = preset.densityMinMaster;
    }
    set_scaled_triplet(_pGrainDensityMin, preset.densityMinMaster, densityMinRatio, 0.0, 1.0);
    if (_pGrainUniformityMaster) {
        _pGrainUniformityMaster->setValue(preset.uniformityMaster);
        _grainUniformityMasterLast = preset.uniformityMaster;
    }
    set_scaled_triplet(_pGrainUniformity, preset.uniformityMaster, uniformityRatio, 0.0, 1.0);
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

    const double sharpness = read_sanitized_double(_pGrainSharpness, preset.sharpness, 0.0, 1.0);

    const double texture = read_sanitized_double(_pGrainTexture, preset.texture, 0.0, 1.0);

    const double blurDyeClouds = grain_lerp(1.40, 0.60, sharpness);
    const double sizeMixWeight = sanitize_finite_or(preset.sizeMixWeight, grain_lerp(0.072, 0.38, texture));
    const double microCell = sanitize_finite_or(preset.microCell, grain_lerp(50.0, 70.0, texture));
    const double microSigma = sanitize_finite_or(preset.microSigma, grain_lerp(140.0, 200.0, texture));

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

    double percent = read_sanitized_double(_pGlarePercent, 0.10, 0.0, 1.0);
    glare.percent = static_cast<float>(percent);

    double roughness = read_sanitized_double(_pGlareRoughness, 0.4, 0.0, 1.0);
    glare.roughness = static_cast<float>(roughness);

    double blur = read_sanitized_double(_pGlareBlurSigmaPx, 0.5, 0.0, 10.0);
    glare.blur = static_cast<float>(blur);

    double factor = read_sanitized_double(_pGlareCompRemovalFactor, 0.0, 0.0, 1.0);
    glare.compensationRemovalFactor = static_cast<float>(factor);

    double density = read_sanitized_double(_pGlareCompRemovalDensity, 1.2, 0.0, 3.0);
    glare.compensationRemovalDensity = static_cast<float>(density);

    double transition = read_sanitized_double(_pGlareCompRemovalTransition, 0.3, 0.0, 2.0);
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
        result.exposureScale = finite_exp2_scale(exposureParams.sliderEV);
        return result;
    }
    if (!exposureParams.cameraAutoEnabled) {
        result.exposureScale = finite_exp2_scale(exposureParams.sliderEV);
        return result;
    }

    InstanceState* state = _state.get();
    const bool isCudaRender = args.isEnabledCudaRender;

    auto rect_equal = [](const OfxRectI& a, const OfxRectI& b) {
        return a.x1 == b.x1 && a.y1 == b.y1 && a.x2 == b.x2 && a.y2 == b.y2;
        };

    OfxRectI meterBounds = fullBounds;
    if (_src) {
        try {
            const OfxRectD rod = _src->getRegionOfDefinition(args.time);
            const double rodWidth = rod.x2 - rod.x1;
            const double rodHeight = rod.y2 - rod.y1;
            const bool hasRodDimensions =
                sanitize_positive_finite_or(rodWidth, 0.0) > 0.0 &&
                sanitize_positive_finite_or(rodHeight, 0.0) > 0.0;
            if (hasRodDimensions) {
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

    if (state) {
        std::lock_guard<std::mutex> cacheLock(state->autoExposureMutex);
        state->autoExposureCanonicalBounds = meterBounds;
        state->autoExposureCanonicalValid = true;
    }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    // CUDA path: metering + exposure scale are computed and applied entirely on the GPU to avoid
    // forcing a stream synchronization just to read back Y/EV on the CPU.
    if (isCudaRender) {
        result.autoEV = 0.0;
        result.exposureScale = 1.0f;
        return result;
    }
#endif

    const std::shared_ptr<const WorkingState> wsCur = (state ? JuicerAtomic::load_shared_ptr(&state->activeWorkingState) : nullptr);
    const uint64_t wsBuildCounter = wsCur ? wsCur->buildCounter : 0;

    // Camera auto-exposure always meters against AgX's fixed 18.4% target (independent of scanner target tweaks).
    constexpr double kCameraMeterTargetY = 0.184;
    constexpr double kInvLn2 = 1.44269504088896340736;

    const double sigma = 0.2;
    const double renderScaleX = sanitize_positive_finite_or(args.renderScale.x, 1.0);
    const double renderScaleY = sanitize_positive_finite_or(args.renderScale.y, 1.0);
    const std::uintptr_t clipToken = reinterpret_cast<std::uintptr_t>(_src);

    int inputColorSpaceIndex = Spectral::inputColorSpaceToIndex(Spectral::InputColorSpace::DaVinciWideGamut);
    if (_pInputColorSpace) {
        _pInputColorSpace->getValue(inputColorSpaceIndex);
    }
    bool applyInputCctfDecoding = false;
    if (_pInputCctfDecoding) {
        _pInputCctfDecoding->getValue(applyInputCctfDecoding);
    }

    double autoEV = 0.0;
    bool haveCachedAutoEV = false;
    const int meteringMethod = exposureParams.meteringMethod;
    if (state) {
        std::lock_guard<std::mutex> cacheLock(state->autoExposureMutex);
        if (state->autoExposureCacheValid &&
            state->autoExposureCacheIsCudaRender == isCudaRender &&
            state->autoExposureCacheAutoEnabled &&
            state->autoExposureCacheMeteringMethod == meteringMethod &&
            nearly_equal_double(state->autoExposureCacheTime, args.time) &&
            state->autoExposureCacheBuildCounter == wsBuildCounter &&
            rect_equal(state->autoExposureCacheBounds, meterBounds) &&
            nearly_equal_double(state->autoExposureCacheRenderScaleX, renderScaleX) &&
            nearly_equal_double(state->autoExposureCacheRenderScaleY, renderScaleY) &&
            state->autoExposureCacheClipToken == clipToken &&
            state->autoExposureCacheInputColorSpaceIndex == inputColorSpaceIndex &&
            state->autoExposureCacheApplyCctfDecoding == applyInputCctfDecoding) {
            autoEV = state->autoExposureCacheEV;
            haveCachedAutoEV = true;
        }
    }

    if (!haveCachedAutoEV) {
        bool measurementValid = false;
        double evComp = 0.0;
        double Yexp = 0.0;
        const Spectral::InputColorSpace inputColorSpace =
            Spectral::inputColorSpaceFromIndex(inputColorSpaceIndex);
        const Spectral::Mat3 inputRgbToXYZ = Spectral::matrix_input_rgb_to_xyz(inputColorSpace);
        if (meteringMethod == static_cast<int>(MeteringMethod::Median)) {
            Yexp = measure_median_Y_DWG(
                srcImg,
                meterBounds,
                inputColorSpace,
                inputRgbToXYZ,
                applyInputCctfDecoding);
        }
        else {
            Yexp = measure_center_weighted_Y_DWG_cached(
                srcImg,
                meterBounds,
                sigma,
                state,
                renderScaleX,
                renderScaleY,
                clipToken,
                inputColorSpace,
                inputRgbToXYZ,
                applyInputCctfDecoding);
        }
        const bool canComputeEv = (Yexp > 0.0 && kCameraMeterTargetY > 0.0);
        if (canComputeEv) {
            const double exposureRatio = Yexp / kCameraMeterTargetY;
            evComp = -std::log(exposureRatio) * kInvLn2;
        }
        if (!is_finite(evComp)) {
            evComp = 0.0;
            measurementValid = false;
        }
        else {
            measurementValid = canComputeEv;
        }
        autoEV = evComp;

        if (state) {
            std::lock_guard<std::mutex> cacheLock(state->autoExposureMutex);
            state->autoExposureCacheValid = measurementValid;
            state->autoExposureCacheIsCudaRender = isCudaRender;
            state->autoExposureCacheTime = args.time;
            state->autoExposureCacheAutoEnabled = true;
            state->autoExposureCacheMeteringMethod = meteringMethod;
            state->autoExposureCacheBuildCounter = wsBuildCounter;
            state->autoExposureCacheBounds = meterBounds;
            state->autoExposureCacheEV = autoEV;
            state->autoExposureCacheRenderScaleX = renderScaleX;
            state->autoExposureCacheRenderScaleY = renderScaleY;
            state->autoExposureCacheClipToken = clipToken;
            state->autoExposureCacheInputColorSpaceIndex = inputColorSpaceIndex;
            state->autoExposureCacheApplyCctfDecoding = applyInputCctfDecoding;
        }
    }

    const double sliderEV = exposureParams.sliderEV;
    const double totalEV = autoEV + sliderEV;
    result.autoEV = autoEV;
    result.exposureScale = finite_exp2_scale(totalEV);
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
        float* dMaxIt = dirRT.dMax;
        for (int i = 0; i < 3; ++i, ++dMaxIt) {
            float v = static_cast<float>(sanitize_positive_finite_or(*dMaxIt, 1.0));
            if (v > 1000.0f) v = 1000.0f;
            *dMaxIt = v;
        }
        sanitize_dir_matrix(dirRT.M);
    }

    float sigmaPixels = 0.0f;
    const float sigmaMicrometers = dirRT.spatialSigmaMicrometers;
    const bool hasPixelSize = sanitize_positive_finite_or(pixelSizeUm, 0.0) > 0.0;
    if (sigmaMicrometers > 0.0f && hasPixelSize) {
        sigmaPixels = sigmaMicrometers / pixelSizeUm;
        sigmaPixels = sanitize_nonnegative_finite_or(sigmaPixels, 0.0f);
    }
    else {
        // Fallback to legacy geometry if pixelSizeUm was not available
        double filmLongEdgeMm = 35.0;
        if (_pCameraFilmFormat) {
            double filmFormat = 35.0;
            _pCameraFilmFormat->getValue(filmFormat);
            filmLongEdgeMm = sanitize_positive_finite_or(filmFormat, filmLongEdgeMm);
        }

        const double widthPx = static_cast<double>(fullWidth);
        const double heightPx = static_cast<double>(fullHeight);
        const double longEdgePx = std::max(widthPx, heightPx);

        if (sigmaMicrometers > 0.0f && longEdgePx > 0.0 && filmLongEdgeMm > 0.0) {
            sigmaPixels = Couplers::spatial_sigma_pixels_from_micrometers(
                sigmaMicrometers,
                filmLongEdgeMm,
                widthPx,
                heightPx);
            sigmaPixels = sanitize_nonnegative_finite_or(sigmaPixels, 0.0f);
        }
    }
    dirRT.spatialSigmaPixels = sigmaPixels;

    auto dir_has_effect = [](const Couplers::Runtime& rt) -> bool {
        if (!rt.active) {
            return false;
        }
        return has_nonzero_finite_dir_matrix(rt.M);
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

    // Defer heavy bootstrap until first param change
}

JuicerEffect::~JuicerEffect() {
    std::uint64_t releasedMaskBytes = 0;
    if (_state) {
        std::lock_guard<std::mutex> lock(_state->autoExposureMutex);
        releasedMaskBytes = _state->autoExposureMaskCachedBytes;
        _state->autoExposureMaskWeights.reset();
        _state->autoExposureMaskCachedBytes = 0;
        _state->autoExposureMaskValid = false;
        _state->autoExposureMaskSum = 0.0;
    }
    if (releasedMaskBytes > 0) {
        update_auto_exposure_mask_resident_bytes(releasedMaskBytes, 0);
        trace_auto_exposure_mask_cache_event(
            _state.get(),
            "cache_release",
            0,
            0,
            0,
            0,
            releasedMaskBytes,
            "instance_destroy");
    }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    if (_state) {
        const bool traceInfo = JTRACE_ENABLED(1);
        std::vector<JuicerCuda::ResourceManager::DeviceContextKey> keys;
        {
            std::lock_guard<std::mutex> lock(_state->cudaMutex);
            keys.reserve(_state->cudaByDevice.size());
            for (const auto& entry : _state->cudaByDevice) {
                keys.emplace_back(entry.first);
            }
        }
        const JuicerCuda::ResourceManager::DeviceContextKey* keyData = keys.data();
        const size_t keyCount = keys.size();
        for (size_t i = 0; i < keyCount; ++i, ++keyData) {
            const auto& key = *keyData;
            std::string retireError;
            const bool retireOk = JuicerCuda::ResourceManager::command_retire_context_idle(key, retireError);
            if (!retireOk || !retireError.empty()) {
                if (traceInfo) {
                    const std::uintptr_t contextBits = reinterpret_cast<std::uintptr_t>(key.contextOpaque);
                    std::string msg;
                    msg.reserve(192);
                    msg = "teardown_retire_idle_failed device_id=";
                    msg += std::to_string(key.deviceId);
                    msg += " context=";
                    msg += std::to_string(contextBits);
                    msg += " accepted=";
                    msg += std::to_string(retireOk ? 1 : 0);
                    if (!retireError.empty()) {
                        msg += " error=";
                        msg += retireError;
                    }
                    JTRACE("MSLCY", msg);
                }
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

    const int nComponents = pixel_component_count(comps);
    const bool traceVerbose = JTRACE_ENABLED(3);

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
        filmFormatMm = sanitize_positive_finite_or(filmFormat, filmFormatMm);
    }
    const double longEdgePx = static_cast<double>(std::max(fullWidth, fullHeight));
    float pixelSizeUm = 0.0f;
    if (filmFormatMm > 0.0 && longEdgePx > 0.0) {
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
    if (traceVerbose) {
        ParamSnapshot Pdbg = snapshotParams();
        const char* paperKey = print_paper_json_key_for_index(Pdbg.printPaperIndex);
        const char* filmKey = negative_json_key_for_stock_index(Pdbg.filmStockIndex);
        const std::uintptr_t prtPtr = reinterpret_cast<std::uintptr_t>(prt);
        const std::uint64_t buildCounter = ws ? ws->buildCounter : 0;
        const float neutralY = prt ? prt->neutralY : 0.0f;
        const float neutralM = prt ? prt->neutralM : 0.0f;
        const float neutralC = prt ? prt->neutralC : 0.0f;
        const char* paperLabel = paperKey ? paperKey : "<null>";
        const char* filmLabel = filmKey ? filmKey : "<null>";
        std::string msg;
        msg.reserve(256);
        msg = "render print state build=";
        msg += std::to_string(buildCounter);
        msg += " paper=";
        msg += paperLabel;
        msg += " film=";
        msg += filmLabel;
        msg += " printRT=";
        msg += std::to_string(prtPtr);
        msg += " neutralY/M/C=";
        msg += std::to_string(neutralY);
        msg += "/";
        msg += std::to_string(neutralM);
        msg += "/";
        msg += std::to_string(neutralC);
        msg += " yFilter=";
        msg += std::to_string(printParams.yFilter);
        msg += " mFilter=";
        msg += std::to_string(printParams.mFilter);
        msg += " cFilter=";
        msg += std::to_string(printParams.cFilter);
        msg += " bypass=";
        msg += std::to_string(printParams.bypass ? 1 : 0);
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
    // Per agx-emulsion parity: autoExposure.exposureScale already encodes 2^(autoEV + sliderEV).
    float filmExposureScale = static_cast<float>(sanitize_positive_finite_or(autoExposure.exposureScale, 1.0));
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
    const bool traceInfo = JTRACE_ENABLED(1);
    auto trace_changed_param_gate = [&](const char* prefix) {
        if (!traceInfo) {
            return;
        }
        std::string msg;
        msg.reserve((prefix ? std::strlen(prefix) : 0u) + paramName.size() + 1);
        msg = prefix;
        msg += paramName;
        msg.push_back('\'');
        JTRACE("BUILD", msg);
    };

    // Suppress recursion while we are programmatically setting params
    if (_state && _state->suppressParamEvents) {
        trace_changed_param_gate("changedParam suppressed for '");
        return;
    }
    if (_state && _state->inBootstrap) {
        trace_changed_param_gate("changedParam ignored during bootstrap for '");
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

    auto has_master_triplet_params = [this](OFX::DoubleParam* masterParam, OFX::Double3DParam* advParam) -> bool {
        return masterParam && advParam && _state;
    };

    auto read_master_value = [&](OFX::DoubleParam* masterParam, double lo, double hi, double& master) -> bool {
        if (!masterParam) {
            return false;
        }
        masterParam->getValue(master);
        if (!is_finite(master)) {
            return false;
        }
        master = std::clamp(master, lo, hi);
        return true;
    };

    auto apply_master_delta = [&](OFX::DoubleParam* masterParam,
        OFX::Double3DParam* advParam,
        double& masterCache,
        double lo,
        double hi) {
        if (!has_master_triplet_params(masterParam, advParam)) {
            return;
        }
        double master = 0.0;
        if (!read_master_value(masterParam, lo, hi, master)) {
            return;
        }
        double prev = masterCache;
        if (!is_finite(prev)) {
            prev = master;
        }
        const double delta = master - prev;
        std::array<double, 3> values{ {0.0, 0.0, 0.0} };
        advParam->getValue(values[0], values[1], values[2]);
        for (double& value : values) {
            value = sanitize_finite_clamped(value, master, lo, hi);
        }
        if (delta != 0.0) {
            for (double& value : values) {
                value = std::clamp(value + delta, lo, hi);
            }
            const bool wasSuppressed = _state->suppressParamEvents;
            _state->suppressParamEvents = true;
            advParam->setValue(values[0], values[1], values[2]);
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
        if (!has_master_triplet_params(masterParam, advParam)) {
            return;
        }
        double master = 0.0;
        if (!read_master_value(masterParam, lo, hi, master)) {
            return;
        }

        std::array<double, 3> values{ {0.0, 0.0, 0.0} };
        advParam->getValue(values[0], values[1], values[2]);
        for (double& value : values) {
            value = sanitize_finite_clamped(value, master, lo, hi);
        }

        std::array<double, 3> ratio = fallbackRatio;
        const double mean = (values[0] + values[1] + values[2]) / 3.0;
        if (is_positive_finite(mean)) {
            double* ratioIt = ratio.data();
            const double* valueIt = values.data();
            for (int i = 0; i < 3; ++i, ++ratioIt, ++valueIt) {
                *ratioIt = *valueIt / mean;
            }
        }

        double* valueIt = values.data();
        const double* ratioIt = ratio.data();
        for (int i = 0; i < 3; ++i, ++valueIt, ++ratioIt) {
            *valueIt = std::clamp(master * *ratioIt, lo, hi);
        }
        const bool wasSuppressed = _state->suppressParamEvents;
        _state->suppressParamEvents = true;
        advParam->setValue(values[0], values[1], values[2]);
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
        if (is_finite(sharpness)) {
            sharpness = sanitize_finite_clamped(sharpness, 0.5, 0.0, 1.0);
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
        if (is_finite(texture)) {
            texture = sanitize_finite_clamped(texture, 0.55, 0.0, 1.0);
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
    auto read_bool_as_int = [](auto* param, bool fallback) -> int {
        bool value = fallback;
        if (param) {
            param->getValue(value);
        }
        return value ? 1 : 0;
    };

    if (_pFilmStock)      _pFilmStock->getValue(P.filmStockIndex);
    if (_pPrintPaper)     _pPrintPaper->getValue(P.printPaperIndex);
    if (_pSpectralMode)   _pSpectralMode->getValue(P.spectralUpsamplingMode);
    if (_pRefIll)         _pRefIll->getValue(P.refIll);
    if (_pEnlIll)         _pEnlIll->getValue(P.enlIll);
    if (_pEnlDichroicSet) _pEnlDichroicSet->getValue(P.enlDichroicSet);
    if (_pGlareCompRemovalFactor) {
        P.glareCompRemovalFactor =
            read_sanitized_double(_pGlareCompRemovalFactor, P.glareCompRemovalFactor, 0.0, 1.0);
    }
    if (_pGlareCompRemovalDensity) {
        P.glareCompRemovalDensity =
            read_sanitized_double(_pGlareCompRemovalDensity, P.glareCompRemovalDensity, 0.0, 3.0);
    }
    if (_pGlareCompRemovalTransition) {
        P.glareCompRemovalTransition =
            read_sanitized_double(_pGlareCompRemovalTransition, P.glareCompRemovalTransition, 0.0, 2.0);
    }
    if (_pPrintDminFactor) {
        P.printDminFactor =
            read_sanitized_double(_pPrintDminFactor, P.printDminFactor, 0.0, 1.0);
    }
    if (_pInputColorSpace) _pInputColorSpace->getValue(P.inputColorSpace);
    P.inputCctfDecoding = read_bool_as_int(_pInputCctfDecoding, false);
#ifdef JUICER_ENABLE_COUPLERS
    P.couplersActive = read_bool_as_int(_pCouplersActive, true);
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
    P.scannerUseLut = read_bool_as_int(_pScannerUseLut, true);
    if (_pScannerLutResolution) _pScannerLutResolution->getValue(P.scannerLutResolution);
    if (_pOutputColorSpace) _pOutputColorSpace->getValue(P.outputColorSpace);
    P.outputCctfEncoding = read_bool_as_int(_pOutputCctfEncoding, true);
    P.outputLinearPassThrough = read_bool_as_int(_pOutputLinearPassThrough, false);
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
    const std::string printProfileJson = profile_json_path_for_key_or_empty(paperKey);
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
        // Identity fallback is already handled in loader via 1.0 curves.
        trace_dichroic_load_failure("dichroic load failed", dichroicDir, ex.what(), "using identity filters");
    }
    catch (...) {
        trace_dichroic_load_failure("dichroic load failed", dichroicDir, nullptr, "using identity filters");
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
    const bool traceInfo = JTRACE_ENABLED(1);

    const char* paperKey = print_paper_json_key_for_index(P.printPaperIndex);
    const char* negativeKey = negative_json_key_for_stock_index(P.filmStockIndex);
    const std::vector<std::string> illumKeys = enlarger_illuminant_keys_for_choice(P.enlIll);

    auto join_illum_keys = [&]() -> std::string {
        std::string combined;
        size_t reserveHint = 0;
        const std::string* keyData = illumKeys.data();
        const size_t keyCount = illumKeys.size();
        for (size_t i = 0; i < keyCount; ++i, ++keyData) {
            reserveHint += keyData->size() + 1;
        }
        combined.reserve(reserveHint);
        keyData = illumKeys.data();
        for (size_t i = 0; i < keyCount; ++i, ++keyData) {
            if (!combined.empty()) {
                combined += ",";
            }
            combined += *keyData;
        }
        if (combined.empty()) {
            combined = "<none>";
        }
        return combined;
        };

    if (!(paperKey && negativeKey && !illumKeys.empty())) {
        if (traceInfo) {
            const std::string paperStr = paperKey ? paperKey : "<unset>";
            const std::string negStr = negativeKey ? negativeKey : "<unset>";
            JTRACE("PRINT", "Neutral filter lookup prerequisites missing: paper="
                + paperStr + " negative=" + negStr + " illum_choices=" + join_illum_keys());
        }
        throw std::runtime_error("Neutral filter metadata incomplete for current selection");
    }

    float neutralY = Print::kDefaultNeutralY;
    float neutralM = Print::kDefaultNeutralM;
    float neutralC = Print::kDefaultNeutralC;
    bool loaded = false;

    const std::string jsonPathPrimary = data_dir_string("profiles", enlarger_neutral_filters_json_for_choice(P.enlDichroicSet));
    const std::string jsonPathFallback = data_dir_string("profiles", "enlarger_neutral_ymc_filters.json");
    std::tuple<float, float, float> ymc{};
    std::string selectedDbVersionHash;
    const std::string* illumKeyData = illumKeys.data();
    const size_t illumKeyCount = illumKeys.size();
    for (size_t i = 0; i < illumKeyCount; ++i, ++illumKeyData) {
        const std::string& illumKey = *illumKeyData;
        if (illumKey.empty()) {
            continue;
        }
        if (load_enlarger_neutral_filters(jsonPathPrimary, paperKey, illumKey, negativeKey, ymc, NeutralFilterThreadClass::Control, &selectedDbVersionHash) ||
            (jsonPathPrimary != jsonPathFallback &&
                load_enlarger_neutral_filters(jsonPathFallback, paperKey, illumKey, negativeKey, ymc, NeutralFilterThreadClass::Control, &selectedDbVersionHash))) {
            neutralY = std::clamp(std::get<0>(ymc), 0.0f, 1.0f);
            neutralM = std::clamp(std::get<1>(ymc), 0.0f, 1.0f);
            neutralC = std::clamp(std::get<2>(ymc), 0.0f, 1.0f);
            loaded = true;
            if (traceInfo) {
                std::string msg;
                msg.reserve(192);
                msg = "Neutral filters loaded for ";
                msg += illumKey;
                msg += " Y/M/C=";
                msg += std::to_string(neutralY);
                msg += "/";
                msg += std::to_string(neutralM);
                msg += "/";
                msg += std::to_string(neutralC);
                msg += " db_version_hash=";
                msg += selectedDbVersionHash.empty() ? "none" : selectedDbVersionHash;
                JTRACE("PRINT", msg);
            }
            break;
        }
    }

    if (!loaded) {
        if (traceInfo) {
            const std::string paperStr = paperKey ? paperKey : "<unset>";
            const std::string negStr = negativeKey ? negativeKey : "<unset>";
            JTRACE("PRINT", "Neutral filters missing for paper=" + paperStr
                + " illuminant_keys=" + join_illum_keys()
                + " negative=" + negStr + "; aborting print path");
        }
        throw std::runtime_error("Neutral filter database entry not found");
    }

    std::uint64_t neutralFilterHash = Print::kDefaultNeutralFilterHash;
    if (!selectedDbVersionHash.empty()) {
        neutralFilterHash = Hash::hash_bytes(selectedDbVersionHash.data(), selectedDbVersionHash.size());
        if (neutralFilterHash == 0) {
            neutralFilterHash = Print::kDefaultNeutralFilterHash;
        }
    }

    _state->printRT.neutralY = neutralY;
    _state->printRT.neutralM = neutralM;
    _state->printRT.neutralC = neutralC;
    _state->printRT.neutralFilterHash = neutralFilterHash;
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

    const bool wasSuppressed = _state->suppressParamEvents;
    _state->suppressParamEvents = true;

    auto apply_clean_double = [&](bool dirty,
                                  double source,
                                  double fallback,
                                  double lo,
                                  double hi,
                                  OFX::DoubleParam* param,
                                  double& target) {
        if (dirty) {
            return;
        }
        const double value = sanitize_finite_clamped(source, fallback, lo, hi);
        if (param) {
            param->setValue(value);
        }
        target = value;
    };

    if (!_state->couplerDirty.active.load(std::memory_order_acquire)) {
        const bool active = dirCfg.active;
        if (_pCouplersActive) {
            _pCouplersActive->setValue(active);
        }
        P.couplersActive = active ? 1 : 0;
    }

    apply_clean_double(
        _state->couplerDirty.amount.load(std::memory_order_acquire),
        static_cast<double>(dirCfg.amount),
        P.couplersAmount,
        0.0,
        2.0,
        _pCouplersAmount,
        P.couplersAmount);

    apply_clean_double(
        _state->couplerDirty.ratioB.load(std::memory_order_acquire),
        static_cast<double>(dirCfg.ratioRGB[0]),
        P.ratioB,
        0.0,
        1.0,
        _pCouplersAmountB,
        P.ratioB);

    apply_clean_double(
        _state->couplerDirty.ratioG.load(std::memory_order_acquire),
        static_cast<double>(dirCfg.ratioRGB[1]),
        P.ratioG,
        0.0,
        1.0,
        _pCouplersAmountG,
        P.ratioG);

    apply_clean_double(
        _state->couplerDirty.ratioR.load(std::memory_order_acquire),
        static_cast<double>(dirCfg.ratioRGB[2]),
        P.ratioR,
        0.0,
        1.0,
        _pCouplersAmountR,
        P.ratioR);

    apply_clean_double(
        _state->couplerDirty.sigma.load(std::memory_order_acquire),
        static_cast<double>(dirCfg.diffusionInterlayer),
        P.sigma,
        0.0,
        4.0,
        _pCouplersSigma,
        P.sigma);

    apply_clean_double(
        _state->couplerDirty.high.load(std::memory_order_acquire),
        static_cast<double>(dirCfg.highExposureShift),
        P.high,
        0.0,
        1.0,
        _pCouplersHigh,
        P.high);

    const float profileSpatialSigma = _state->couplerProfileSpatialSigmaValid
        ? static_cast<float>(_state->couplerProfileSpatialSigmaMicrometers)
        : dirCfg.diffusionSizeUm;
    apply_clean_double(
        _state->couplerDirty.spatialSigma.load(std::memory_order_acquire),
        static_cast<double>(profileSpatialSigma),
        P.spatialSigmaMicrometers,
        0.0,
        50.0,
        _pCouplersSpatialSigma,
        P.spatialSigmaMicrometers);

    _state->suppressParamEvents = wasSuppressed;
}
#endif

void JuicerEffect::onParamsPossiblyChanged(const char* changedNameOrNull) {
    if (!_state) return;
    const bool traceVerbose = JTRACE_ENABLED(3);
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
    if (traceVerbose) {
        const char* paperKey = print_paper_json_key_for_index(P.printPaperIndex);
        const char* filmKey = negative_json_key_for_stock_index(P.filmStockIndex);
        const std::shared_ptr<const WorkingState> wsDbg = JuicerAtomic::load_shared_ptr(&_state->activeWorkingState);
        const std::uint64_t activeBuild = wsDbg ? wsDbg->buildCounter : 0;
        const std::uint64_t lastHash = _state->lastHash.load(std::memory_order_acquire);
        const char* paperLabel = paperKey ? paperKey : "<null>";
        const char* filmLabel = filmKey ? filmKey : "<null>";
        std::string msg;
        msg.reserve(224);
        msg = "params change name=";
        msg += (changedNameOrNull ? changedNameOrNull : "<null>");
        msg += " printIndex=";
        msg += std::to_string(P.printPaperIndex);
        msg += " printKey=";
        msg += paperLabel;
        msg += " filmIndex=";
        msg += std::to_string(P.filmStockIndex);
        msg += " filmKey=";
        msg += filmLabel;
        msg += " activeBuild=";
        msg += std::to_string(activeBuild);
        msg += " lastHash=";
        msg += std::to_string(lastHash);
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

    auto reload_dichroic_filters = [&]() -> bool {
        const std::string dichroicDirReload = ensure_trailing_separator(
            data_dir_string("filters", "dichroics", dichroic_dir_name_for_choice(P.enlDichroicSet)));
        try {
            Print::load_dichroic_filters_from_csvs(dichroicDirReload, _state->printRT);
            return true;
        }
        catch (const std::exception& ex) {
            trace_dichroic_load_failure(
                "dichroic reload failed",
                dichroicDirReload,
                ex.what(),
                "identity filters remain active");
        }
        catch (...) {
            trace_dichroic_load_failure(
                "dichroic reload failed",
                dichroicDirReload,
                nullptr,
                "identity filters remain active");
        }
        return false;
    };

    auto trace_neutral_filters_applied = [&](const char* reloadSource) {
        if (!traceVerbose) {
            return;
        }
        const char* paperKey = print_paper_json_key_for_index(P.printPaperIndex);
        const char* filmKey = negative_json_key_for_stock_index(P.filmStockIndex);
        const char* paperLabel = paperKey ? paperKey : "<null>";
        const char* filmLabel = filmKey ? filmKey : "<null>";
        std::string msg;
        msg.reserve(192);
        msg = "neutral filters applied (";
        msg += (reloadSource ? reloadSource : "unspecified");
        msg += ") paper=";
        msg += paperLabel;
        msg += " film=";
        msg += filmLabel;
        msg += " Y/M/C=";
        msg += std::to_string(_state->printRT.neutralY);
        msg += "/";
        msg += std::to_string(_state->printRT.neutralM);
        msg += "/";
        msg += std::to_string(_state->printRT.neutralC);
        JTRACE_VERBOSE("PRINTDBG", msg);
    };

    bool printReloaded = false;
    if (changedNameOrNull && std::strcmp(changedNameOrNull, kParamPrintPaper) == 0) {
        const std::string printDir = print_dir_for_index(P.printPaperIndex);
        const char* paperKey = print_paper_json_key_for_index(P.printPaperIndex);
        const std::string printProfileJson = profile_json_path_for_key_or_empty(paperKey);
        Print::load_profile_from_dir(printDir, _state->printRT.profile, printProfileJson, &_state->printRT);
        _state->printRT.hasMidNeutralDensity = _state->printRT.profile.hasMidNeutralDensity;
        _state->printRT.midNeutralDensity = std::move(_state->printRT.profile.midNeutralDensity);
        _state->printRT.hasMidNeutralLogE = _state->printRT.profile.hasMidNeutralLogE;
        _state->printRT.midNeutralLogE = std::move(_state->printRT.profile.midNeutralLogE);
        if (traceVerbose) {
            const char* paperLabel = paperKey ? paperKey : "<null>";
            std::string msg;
            msg.reserve(256);
            msg = "print reload key=";
            msg += paperLabel;
            msg += " dir=";
            msg += printDir;
            msg += " json=";
            msg += printProfileJson;
            msg += " ref=";
            msg += _state->printRT.referenceIlluminant;
            msg += " view=";
            msg += _state->printRT.viewingIlluminant;
            JTRACE_VERBOSE("PRINTDBG", msg);
        }

        // Reload dichroic filters (vendor selection controls which curves are used).
        (void)reload_dichroic_filters();

        printReloaded = true;
    }

    bool dichroicReloaded = false;
    if (changedNameOrNull && std::strcmp(changedNameOrNull, kParamEnlargerDichroicSet) == 0) {
        dichroicReloaded = reload_dichroic_filters();
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
        trace_neutral_filters_applied("print/dichroic");
    }
    if (filmReloaded && !neutralApplied) {
        applyNeutralFilters(P);
        neutralApplied = true;
        trace_neutral_filters_applied("film");
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
