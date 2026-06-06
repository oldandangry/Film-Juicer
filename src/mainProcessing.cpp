// JuicerProcessing.cpp

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <cctype>
#include <string>
#include <atomic>
#include <mutex>
#include <limits>
#include <chrono>
#include <thread>

#include "GaussianSciPy.h"

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
#include <cuda_runtime.h>
#include <cuda.h>
#if defined(_WIN32)
#include <windows.h>
#endif
#endif

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
#include "Cuda/JuicerCudaDirectFilmPayloads.h"
#include "Cuda/JuicerCudaResources.h"
#include "Cuda/JuicerCudaPayloads.h"
#include "Cuda/ResourceManager/JuicerCudaResourceCore.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#endif

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
extern "C" cudaError_t juicer_cuda_negative_pipeline(
    const JuicerCuda::PipelineRunParams* hParams,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_build_gate_defect_mask(
    const JuicerCuda::PipelineRunParams* hParams,
    float* dGateMask,
    int gateWidth,
    int gateHeight,
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
    float* dAux,
    float* dGrainTmp,
    float* dGrainTmpShared,
    float* dGrainTmpMid,
    float* dGrainTmpCoarse,
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
    float* dAux,
    float* dGrainTmp,
    float* dGrainTmpShared,
    float* dGrainTmpMid,
    float* dGrainTmpCoarse,
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
#pragma warning(disable : 5040)
#include "ofxsProcessing.h"
#include "ofxsImageEffect.h"
#pragma warning(pop)
#include "Logging.h"
#include "Hash.h"
#include "SpectralData.h"
#include "ColorTransforms.h"
#include "Print.h"
#include "ProcessRoot.h"
#include "JuicerState.h"
#include "Scanner.h"
#include "OutputColor.h"
#include "Couplers.h"
#include "mainProcessing.h"

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__) && defined(JUICER_CUDA_SELF_CHECK) && (JUICER_CUDA_SELF_CHECK != 0)
bool juicer_cuda_runtime_self_check(void* cudaStreamOpaque, const char** outError);
#endif

namespace JuicerProcScanner {

    struct ScannerPreflightResult {
        const Scanner::ScannerMediumRuntime* mediumRuntime = nullptr;
        const Scanner::ColorRuntime* colorRuntime = nullptr;
        Scanner::ScannerStaticKey staticKey{};
    };

    struct ScannerKeyBundle {
        Scanner::ScannerRuntimeKey runtimeKey{};
        Scanner::ScannerKey scannerKey{};
    };

    struct ScannerMediumRuntimeBinding {
        const Scanner::ScannerMediumRuntime* selectedRuntime = nullptr;
        Scanner::ScannerMediumRuntime printOverride{};
        bool usesPrintOverride = false;
        bool valid = false;
        const char* label = "scanner";

        const Scanner::ScannerMediumRuntime* runtime() const {
            return usesPrintOverride ? &printOverride : selectedRuntime;
        }
    };

    std::uint64_t hash_scanner_runtime_lane(
        const WorkingState* ws,
        const Scanner::Settings& settings,
        const Scanner::Options& options,
        std::uint32_t frameBoundsVersion);

    ScannerMediumRuntimeBinding bind_scanner_medium_runtime(
        const WorkingState& ws,
        bool printActive,
        bool hasPrintGlareOverride,
        const Profiles::ProfileGlare* printGlareOverride,
        bool forcePrintGlareHash);

    ScannerPreflightResult validate_scanner_preflight_or_throw(
        bool runtimeValid,
        const char* mediumLabel,
        const Scanner::ScannerMediumRuntime* mediumRuntime,
        bool traceInfoEnabled,
        bool traceVerboseEnabled,
        const char* path,
        const char* fatalTag);

    ScannerKeyBundle make_scanner_key_bundle_or_throw(
        const Scanner::ScannerStaticKey& staticKey,
        const Scanner::Settings& settings,
        const Scanner::Options& options,
        std::uint32_t frameBoundsVersion);

} // namespace JuicerProcScanner

namespace {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    constexpr std::uint64_t kSeedPassGrain = 1;
#endif
    constexpr std::uint64_t kSeedPassGlare = 2;
    inline float sanitize_nonnegative_or(float value, float fallback);
    inline float sanitize_unit_or(float value, float fallback);
    inline bool is_finite(float value);
    inline bool is_finite(double value);

    inline int pixel_component_count(OFX::PixelComponentEnum comps) {
        switch (comps) {
            case OFX::ePixelComponentRGBA:
                return 4;
            case OFX::ePixelComponentRGB:
                return 3;
            case OFX::ePixelComponentAlpha:
                return 1;
            default:
                return 0;
        }
    }

    inline int bytes_per_component(OFX::BitDepthEnum depth) {
        switch (depth) {
            case OFX::eBitDepthUByte:
                return 1;
            case OFX::eBitDepthUShort:
                return 2;
            case OFX::eBitDepthFloat:
                return 4;
            default:
                return 0;
        }
    }

    inline bool span_x_within_bounds(int xStart, int xEnd, const OfxRectI& bounds) {
        return xStart >= bounds.x1 && xEnd <= bounds.x2;
    }

    inline bool row_has_full_coverage(const OfxRectI& bounds, int xStart, int xEnd, int y) {
        return span_x_within_bounds(xStart, xEnd, bounds) &&
               y >= bounds.y1 && y < bounds.y2;
    }

    inline int offset_from_start(int start, int offset) {
        return start + offset;
    }

    inline std::size_t linear_row_offset(int rowOffset, int rowWidth) {
        return static_cast<std::size_t>(rowOffset) * static_cast<std::size_t>(rowWidth);
    }

    inline Scanner::ScannerMedium scanner_medium_from_print_active(bool printActive) {
        return printActive
                   ? Scanner::ScannerMedium::Print
                   : Scanner::ScannerMedium::Negative;
    }

    inline bool print_pipeline_active(
        bool wsAvailable,
        const WorkingState* ws,
        bool printReady,
        const Print::Runtime* prt,
        bool printBypass) {
        return wsAvailable && ws && printReady && prt && !printBypass;
    }

    inline const char* scan_stage_label_from_negative_medium(bool negativeMedium) {
        return negativeMedium ? "scan" : "print scan";
    }

    template <typename T>
    inline const T* ptr_if_enabled(bool enabled, const T* ptr) {
        return enabled ? ptr : nullptr;
    }

    template <typename T>
    inline const T* negative_or_print_ptr(
        bool negativeMedium,
        const T& negativeValue,
        const T& printValue) {
        return negativeMedium ? &negativeValue : &printValue;
    }

    inline int row_bytes_or_zero(const OFX::Image* image) {
        return image ? image->getRowBytes() : 0;
    }

    inline float* float_pixel_ptr_or_null(OFX::Image* image, int x, int y) {
        return image ? reinterpret_cast<float*>(image->getPixelAddress(x, y)) : nullptr;
    }

    inline float* alpha_channel_ptr_if_rgba(float* rgbPtr, int nComponents) {
        return (rgbPtr && nComponents == 4) ? (rgbPtr + 3) : nullptr;
    }

    inline bool has_baseline_or_false(const WorkingState* ws) {
        return ws ? ws->hasBaseline : false;
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

    inline const float* read_rgb_row_if_fully_covered(
        OFX::Image* image,
        const OfxRectI& bounds,
        int xStart,
        int xEnd,
        int y) {
        return row_ptr_if_fully_covered<const float>(image, bounds, xStart, xEnd, y);
    }

    inline const std::uint8_t* read_u8_row_if_fully_covered(
        OFX::Image* image,
        const OfxRectI& bounds,
        int xStart,
        int xEnd,
        int y) {
        return row_ptr_if_fully_covered<const std::uint8_t>(image, bounds, xStart, xEnd, y);
    }

    inline std::uint8_t* write_u8_row_if_fully_covered(
        OFX::Image* image,
        const OfxRectI& bounds,
        int xStart,
        int xEnd,
        int y) {
        return row_ptr_if_fully_covered<std::uint8_t>(image, bounds, xStart, xEnd, y);
    }

    inline const float* read_float_row_if_fully_covered(
        OFX::Image* image,
        const OfxRectI& bounds,
        int xStart,
        int xEnd,
        int y) {
        return read_rgb_row_if_fully_covered(image, bounds, xStart, xEnd, y);
    }

    inline float* write_float_row_if_fully_covered(
        OFX::Image* image,
        const OfxRectI& bounds,
        int xStart,
        int xEnd,
        int y) {
        return row_ptr_if_fully_covered<float>(image, bounds, xStart, xEnd, y);
    }

    template <typename T>
    inline T* pixel_ptr(OFX::Image* image, int x, int y) {
        return reinterpret_cast<T*>(image->getPixelAddress(x, y));
    }

    struct CopyRowSpan {
        int xStart = 0;
        int xEnd = 0;
        int y = 0;
    };

    struct CopyRowBytesRequest {
        OFX::Image* src = nullptr;
        OFX::Image* dst = nullptr;
        CopyRowSpan span{};
        size_t bytesPerPixel = 0;
        const std::uint8_t* srcRow = nullptr;
        std::uint8_t* dstRow = nullptr;
    };

    struct CopyRowFloatScalarRequest {
        OFX::Image* src = nullptr;
        OFX::Image* dst = nullptr;
        CopyRowSpan span{};
        const float* srcRow = nullptr;
        float* dstRow = nullptr;
    };

    inline void copy_row_bytes_with_fallback(const CopyRowBytesRequest& request) {
        OFX::Image* const src = request.src;
        OFX::Image* const dst = request.dst;
        const int xStart = request.span.xStart;
        const int xEnd = request.span.xEnd;
        const int y = request.span.y;
        const size_t bytesPerPixel = request.bytesPerPixel;
        const std::uint8_t* const srcRow = request.srcRow;
        std::uint8_t* const dstRow = request.dstRow;
        const int width = xEnd - xStart;
        if (width <= 0) {
            return;
        }
        if (srcRow && dstRow) {
            const size_t rowBytes = static_cast<size_t>(width) * bytesPerPixel;
            std::memcpy(dstRow, srcRow, rowBytes);
            return;
        }
        if (srcRow) {
            const std::uint8_t* srcPixIt = srcRow;
            for (int xOff = 0; xOff < width; ++xOff, srcPixIt += bytesPerPixel) {
                const int x = offset_from_start(xStart, xOff);
                std::uint8_t* dstPix = pixel_ptr<std::uint8_t>(dst, x, y);
                if (!dstPix) {
                    continue;
                }
                std::memcpy(dstPix, srcPixIt, bytesPerPixel);
            }
            return;
        }
        if (dstRow) {
            std::uint8_t* dstPixIt = dstRow;
            for (int xOff = 0; xOff < width; ++xOff, dstPixIt += bytesPerPixel) {
                const int x = offset_from_start(xStart, xOff);
                const std::uint8_t* srcPix = pixel_ptr<const std::uint8_t>(src, x, y);
                if (!srcPix) {
                    continue;
                }
                std::memcpy(dstPixIt, srcPix, bytesPerPixel);
            }
            return;
        }
        for (int xOff = 0; xOff < width; ++xOff) {
            const int x = offset_from_start(xStart, xOff);
            const std::uint8_t* srcPix = pixel_ptr<const std::uint8_t>(src, x, y);
            std::uint8_t* dstPix = pixel_ptr<std::uint8_t>(dst, x, y);
            if (!srcPix || !dstPix) {
                continue;
            }
            std::memcpy(dstPix, srcPix, bytesPerPixel);
        }
    }

    inline void copy_row_float_scalar_with_fallback(const CopyRowFloatScalarRequest& request) {
        OFX::Image* const src = request.src;
        OFX::Image* const dst = request.dst;
        const int xStart = request.span.xStart;
        const int xEnd = request.span.xEnd;
        const int y = request.span.y;
        const float* const srcRow = request.srcRow;
        float* const dstRow = request.dstRow;
        const int width = xEnd - xStart;
        if (width <= 0) {
            return;
        }
        if (srcRow && dstRow) {
            const size_t rowBytes = static_cast<size_t>(width) * sizeof(float);
            std::memcpy(dstRow, srcRow, rowBytes);
            return;
        }
        if (dstRow) {
            float* dstPixIt = dstRow;
            for (int xOff = 0; xOff < width; ++xOff, ++dstPixIt) {
                const int x = offset_from_start(xStart, xOff);
                const float* srcPix = pixel_ptr<const float>(src, x, y);
                if (!srcPix) {
                    continue;
                }
                *dstPixIt = *srcPix;
            }
            return;
        }
        if (srcRow) {
            const float* srcPixIt = srcRow;
            for (int xOff = 0; xOff < width; ++xOff, ++srcPixIt) {
                const int x = offset_from_start(xStart, xOff);
                float* dstPix = pixel_ptr<float>(dst, x, y);
                if (!dstPix) {
                    continue;
                }
                *dstPix = *srcPixIt;
            }
            return;
        }
        for (int xOff = 0; xOff < width; ++xOff) {
            const int x = offset_from_start(xStart, xOff);
            float* dstPix = pixel_ptr<float>(dst, x, y);
            const float* srcPix = pixel_ptr<const float>(src, x, y);
            if (!dstPix || !srcPix) {
                continue;
            }
            *dstPix = *srcPix;
        }
    }

    inline void copy_float3(float dst[3], const float src[3]) {
        std::memcpy(dst, src, 3u * sizeof(float));
    }

    inline const char* nonempty_cstr_or(const char* value, const char* fallback) {
        return (value && value[0] != '\0') ? value : fallback;
    }

    inline const char* detail_or_unknown(const char* detail) {
        return nonempty_cstr_or(detail, "(unknown)");
    }

    inline const char* cstr_or_null_if_empty(const std::string& value) {
        return value.empty() ? nullptr : value.c_str();
    }

    inline int bool_to_i32(bool value) {
        return value ? 1 : 0;
    }

    inline void clear_glare_compensation_fields(Profiles::ProfileGlare& glare) {
        glare.compensationRemovalFactor = 0.0f;
        glare.compensationRemovalDensity = 0.0f;
        glare.compensationRemovalTransition = 0.0f;
    }

    inline const char* scanner_medium_label_from_print_active(bool printActive) {
        return printActive ? "print" : "negative";
    }

    inline std::uint64_t bool_to_u64(bool value) {
        return value ? 1ull : 0ull;
    }

    inline void apply_glare_override_fields(
        Profiles::ProfileGlare& dstGlare,
        const Profiles::ProfileGlare& srcGlare) {
        dstGlare.active = srcGlare.active;
        dstGlare.percent = srcGlare.percent;
        dstGlare.roughness = srcGlare.roughness;
        dstGlare.blur = srcGlare.blur;
        clear_glare_compensation_fields(dstGlare);
    }

    inline bool assign_glare_hash(Scanner::ScannerMediumRuntime& mediumRuntime) {
        const std::uint64_t glareHash = Scanner::hash_glare(mediumRuntime.glare);
        if (glareHash == 0) {
            return false;
        }
        mediumRuntime.staticKey.glareHash = glareHash;
        return true;
    }

    std::uint64_t hash_scanner_settings(
        const Scanner::Settings& settings,
        const Scanner::Options& options) {
        const std::uint64_t lutHash = Hash::hash_bytes(&settings.useLut, sizeof(settings.useLut));
        const float fields[3] = {
            options.lensBlurSigmaPx,
            options.unsharpSigmaPx,
            options.unsharpAmount};
        const std::uint64_t optHash = Hash::hash_float_span(fields, 3);
        if (lutHash == 0 || optHash == 0) {
            JTRACE("HASH", "FATAL: invalid scanner settings for hashing");
            return 0;
        }
        const std::uint64_t combined[2] = {lutHash, optHash};
        return Hash::hash_bytes(combined, sizeof(combined));
    }

    const char* scanner_medium_label_or_default(const char* mediumLabel) {
        return nonempty_cstr_or(mediumLabel, "scanner");
    }

    struct ScannerMediumSelection {
        const Scanner::ScannerMediumRuntime* runtime = nullptr;
        bool valid = false;
        const char* label = "scanner";
    };

    ScannerMediumSelection select_scanner_medium(
        const WorkingState& ws,
        bool printActive) {
        ScannerMediumSelection selection{};
        selection.label = scanner_medium_label_from_print_active(printActive);
        if (printActive) {
            selection.runtime = &ws.printMediumRuntime;
            selection.valid = ws.printScannerValid;
            return selection;
        }
        selection.runtime = &ws.negativeMediumRuntime;
        selection.valid = ws.negativeScannerValid;
        return selection;
    }

    void trace_scanner_preflight_fail_if_enabled(
        bool traceInfoEnabled,
        const char* path,
        const char* mediumLabel,
        const std::string& error,
        const char* fatalTag) {
        if (!traceInfoEnabled) {
            return;
        }
        const char* label = scanner_medium_label_or_default(mediumLabel);
        std::string msg;
        msg.reserve(96 + error.size());
        msg = "path=";
        msg += nonempty_cstr_or(path, "unspecified");
        msg += " result=fail medium=";
        msg += label;
        msg += " reason=";
        msg += error;
        JTRACE("MSSKV", msg);
        std::string fatalMsg;
        fatalMsg.reserve(8 + error.size());
        fatalMsg = "FATAL: ";
        fatalMsg += error;
        JTRACE(nonempty_cstr_or(fatalTag, "SCAN"), fatalMsg);
    }

    void trace_scanner_preflight_ok_if_enabled(
        bool traceVerboseEnabled,
        const char* path,
        const char* mediumLabel,
        std::uint64_t staticKeyHash) {
        if (!traceVerboseEnabled) {
            return;
        }
        const char* label = scanner_medium_label_or_default(mediumLabel);
        std::string msg;
        msg.reserve(96);
        msg = "path=";
        msg += nonempty_cstr_or(path, "unspecified");
        msg += " result=ok medium=";
        msg += label;
        msg += " static_key_hash=";
        msg += std::to_string(staticKeyHash);
        JTRACE_VERBOSE("MSSKV", msg);
    }

    void assign_glare_hash_or_throw(Scanner::ScannerMediumRuntime& mediumRuntime) {
        if (assign_glare_hash(mediumRuntime)) {
            Scanner::finalize_static_key(mediumRuntime.staticKey);
            return;
        }
        JTRACE("HASH", "FATAL: failed to hash print glare override parameters");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    bool validate_scanner_preflight_runtime(
        bool runtimeValid,
        const char* mediumLabel,
        const Scanner::ScannerMediumRuntime* mediumRuntime,
        JuicerProcScanner::ScannerPreflightResult& out,
        std::string& outError) {
        out = JuicerProcScanner::ScannerPreflightResult{};
        outError.clear();

        const char* label = scanner_medium_label_or_default(mediumLabel);
        auto set_error = [&](const char* suffix) {
            outError = label;
            outError += suffix;
        };
        if (!runtimeValid) {
            set_error(" scanner runtime invalid");
            return false;
        }
        if (!mediumRuntime) {
            set_error(" scanner medium runtime missing");
            return false;
        }

        const Spectral::SpectralTables* tables = mediumRuntime->tables;
        if (!tables || tables->K <= 0) {
            set_error(" scanner spectral tables unavailable");
            return false;
        }

        Scanner::ScannerStaticKey staticKey = mediumRuntime->staticKey;
        if (tables->tablesHash != staticKey.tablesHash) {
            set_error(" scanner tables hash mismatch for medium");
            return false;
        }
        if (mediumRuntime->range.digest == 0) {
            set_error(" scanner density range missing or invalid");
            return false;
        }
        if (mediumRuntime->range.digest != staticKey.densityRangeHash) {
            set_error(" scanner density range hash mismatch for medium");
            return false;
        }

        const std::uint64_t illumHash = tables->illuminantHash;
        if (illumHash != 0 && mediumRuntime->illuminant.hash != 0 && illumHash != mediumRuntime->illuminant.hash) {
            set_error(" scanner illuminant hash mismatch for medium");
            return false;
        }

        const Scanner::ColorRuntime* colorPtr = mediumRuntime->color;
        if (!colorPtr || colorPtr->hash == 0) {
            set_error(" scanner color runtime missing or invalid");
            return false;
        }
        if (staticKey.colorRuntimeHash != colorPtr->hash) {
            set_error(" scanner static key color hash mismatch");
            return false;
        }

        const std::uint64_t expectedGlareHash = Scanner::hash_glare(mediumRuntime->glare);
        if (expectedGlareHash == 0) {
            set_error(" scanner glare hash missing or invalid");
            return false;
        }
        if (staticKey.glareHash != expectedGlareHash) {
            set_error(" scanner static key glare hash mismatch");
            return false;
        }

        const std::uint64_t storedStaticHash = staticKey.hash;
        Scanner::finalize_static_key(staticKey);
        if (staticKey.hash == 0) {
            set_error(" scanner static key missing or invalid");
            return false;
        }
        if (storedStaticHash != staticKey.hash) {
            set_error(" scanner static key hash mismatch");
            return false;
        }

        out.mediumRuntime = mediumRuntime;
        out.colorRuntime = colorPtr;
        out.staticKey = staticKey;
        return true;
    }

    inline void trace_cuda_fatal_prefixed_if(
        bool traceEnabled,
        const char* prefix,
        const char* detail = nullptr) {
        if (!traceEnabled) {
            return;
        }
        const char* safePrefix = nonempty_cstr_or(prefix, "CUDA operation failed");
        const bool hasDetail = detail && detail[0] != '\0';
        std::string traceMsg;
        traceMsg.reserve(
            8 + std::strlen(safePrefix) + (hasDetail ? (2 + std::strlen(detail)) : 0));
        traceMsg = "FATAL: ";
        traceMsg += safePrefix;
        if (hasDetail) {
            traceMsg += ": ";
            traceMsg += detail;
        }
        JTRACE("CUDA", traceMsg);
    }

    inline bool read_rgb_pixel_if_present(
        OFX::Image* src,
        int x,
        int y,
        float rgb[3]) {
        const float* srcPix = pixel_ptr<const float>(src, x, y);
        if (!srcPix) {
            return false;
        }
        copy_float3(rgb, srcPix);
        return true;
    }

    inline void load_float3_from_planar(
        float dst[3],
        const float*& c0,
        const float*& c1,
        const float*& c2) {
        dst[0] = *c0++;
        dst[1] = *c1++;
        dst[2] = *c2++;
    }

    inline void load_spatial_dir_density_inputs(
        Pipeline::DensityPixelInputs& pxIn,
        const float*& filmRawB,
        const float*& filmRawG,
        const float*& filmRawR,
        const float*& corrY,
        const float*& corrM,
        const float*& corrC) {
        load_float3_from_planar(pxIn.filmRawOverride.v, filmRawB, filmRawG, filmRawR);
        load_float3_from_planar(pxIn.spatialLogECorrectionsYMC, corrY, corrM, corrC);
    }

    inline void copy_float9(float dst[9], const float src[9]) {
        std::memcpy(dst, src, 9u * sizeof(float));
    }

    inline void copy_float3x3(float dst[3][3], const float src[3][3]) {
        std::memcpy(dst, src, 9u * sizeof(float));
    }

    inline void sanitize_nonnegative_triplet(float values[3]) {
        for (float* valueIt = values; valueIt != values + 3; ++valueIt) {
            *valueIt = sanitize_nonnegative_or(*valueIt, 0.0f);
        }
    }

    inline void sanitize_unit_triplet(float values[3]) {
        for (float* valueIt = values; valueIt != values + 3; ++valueIt) {
            *valueIt = sanitize_unit_or(*valueIt, 0.0f);
        }
    }

    inline void zero_float2(float values[2]) {
        float* valueIt = values;
        for (int i = 0; i < 2; ++i, ++valueIt) {
            *valueIt = 0.0f;
        }
    }

    inline void copy_float2(float dst[2], const float src[2]) {
        std::memcpy(dst, src, 2u * sizeof(float));
    }

    inline void store_density_triplet(
        float* densityC,
        float* densityM,
        float* densityY,
        size_t idx,
        const float values[3]) {
        densityC[idx] = values[0];
        densityM[idx] = values[1];
        densityY[idx] = values[2];
    }

    inline void store_zero_density_triplet(
        float* densityC,
        float* densityM,
        float* densityY,
        size_t idx) {
        static constexpr float kZeroDensity[3] = {0.0f, 0.0f, 0.0f};
        store_density_triplet(densityC, densityM, densityY, idx, kZeroDensity);
    }

    inline bool should_abort_relaxed(const std::atomic<bool>& abortFlag) {
        return abortFlag.load(std::memory_order_relaxed);
    }

    inline void mark_abort(std::atomic<bool>& abortFlag) {
        abortFlag.store(true, std::memory_order_relaxed);
    }

    inline void mark_failure_and_abort(std::atomic<bool>& failure, std::atomic<bool>& abortFlag) {
        failure.store(true, std::memory_order_relaxed);
        mark_abort(abortFlag);
    }

    inline float sanitize_amount_0_10(float value) {
        return is_finite(value) ? std::clamp(value, 0.0f, 10.0f) : 0.0f;
    }

    inline float divide_or_zero_if_positive(float numerator, float denominator) {
        return (denominator > 0.0f) ? (numerator / denominator) : 0.0f;
    }

    inline double reciprocal_or_zero(double denominator) {
        return (denominator != 0.0) ? (1.0 / denominator) : 0.0;
    }

    inline float reciprocal_sqrt_or_one(float value, float minValue) {
        return (value > minValue) ? (1.0f / std::sqrt(value)) : 1.0f;
    }

    inline float float_if_enabled(bool enabled, float enabledValue, float fallbackValue) {
        return enabled ? enabledValue : fallbackValue;
    }

    inline float sigma_if_enabled(bool enabled, float sigma) {
        return float_if_enabled(enabled, sigma, 0.0f);
    }

    inline bool grain_blur_enabled(bool wantSublayers, float sigmaPx) {
        return wantSublayers ? (sigmaPx > 0.0f) : (sigmaPx > 0.4f);
    }

    inline float sanitize_nonnegative_or(float value, float fallback) {
        if (!is_finite(value)) {
            return fallback;
        }
        return std::max(0.0f, value);
    }

    inline float sanitize_unit_or(float value, float fallback) {
        if (!is_finite(value)) {
            return fallback;
        }
        return std::clamp(value, 0.0f, 1.0f);
    }

    inline float finite_or_zero(float value) {
        return is_finite(value) ? value : 0.0f;
    }

    inline double finite_or(double value, double fallback) {
        return is_finite(value) ? value : fallback;
    }

    inline double sanitize_finite_clamped_or(double value, double fallback, double lo, double hi) {
        return is_finite(value) ? std::clamp(value, lo, hi) : fallback;
    }

    inline void divide_triplet(float dst[3], const float src[3], float denominator) {
        float* dstIt = dst;
        const float* srcIt = src;
        for (int i = 0; i < 3; ++i, ++dstIt, ++srcIt) {
            *dstIt = *srcIt / denominator;
        }
    }

    inline bool any_positive_triplet(const float values[3]) {
        const float* valueIt = values;
        const float* const valueEnd = values + 3;
        for (; valueIt < valueEnd; ++valueIt) {
            if (*valueIt > 0.0f) {
                return true;
            }
        }
        return false;
    }

    inline bool is_gpu_render_requested(bool openclEnabled, bool cudaEnabled, bool metalEnabled) {
        return openclEnabled || cudaEnabled || metalEnabled;
    }

    inline bool has_grain_defects(
        float filmDustAmount,
        float gateDustAmount,
        float filmScratchAmount,
        float gateScratchAmount) {
        return (filmDustAmount > 0.0f) || (gateDustAmount > 0.0f) ||
               (filmScratchAmount > 0.0f) || (gateScratchAmount > 0.0f);
    }

    inline bool needs_gate_mask_for_defects(float gateDustAmount, float gateScratchAmount) {
        return (gateDustAmount > 0.0f) || (gateScratchAmount > 0.0f);
    }

    inline bool needs_blurred_optics_scratch(bool wantGlareBlur, bool wantGrainBlur, bool wantGrainSublayers) {
        return wantGlareBlur || wantGrainBlur || wantGrainSublayers;
    }

    inline bool needs_independent_grain_path(int debugView, float chromaMix) {
        return (debugView == 0 || debugView == 1) &&
               (is_finite(chromaMix) && chromaMix < 0.999f);
    }

    inline bool is_positive_finite(float value) {
        return is_finite(value) && value > 0.0f;
    }

    inline bool is_positive_finite(double value) {
        return is_finite(value) && value > 0.0;
    }

    inline float positive_finite_or(float value, float fallback) {
        return is_positive_finite(value) ? value : fallback;
    }

    inline double positive_finite_or(double value, double fallback) {
        return is_positive_finite(value) ? value : fallback;
    }

    inline bool is_nonzero_finite(float value) {
        return is_finite(value) && value != 0.0f;
    }

    inline bool is_finite(float value) {
        return std::isfinite(value);
    }

    inline bool is_finite(double value) {
        return std::isfinite(value);
    }

    inline bool wants_unsharp(float sigmaPx, float amount) {
        return is_positive_finite(sigmaPx) && is_nonzero_finite(amount);
    }

    inline bool needs_grain_shared(bool wantGrain, int debugView, float chromaMix) {
        return wantGrain && needs_independent_grain_path(debugView, chromaMix);
    }

    inline bool wants_optics_stage(
        bool wantLensBlur,
        bool wantUnsharp,
        bool wantGlare,
        bool wantHalation,
        bool wantGrain,
        bool wantWeave,
        bool wantDefects) {
        return wantLensBlur || wantUnsharp || wantGlare || wantHalation || wantGrain || wantWeave || wantDefects;
    }

    struct OpticsScratchNeeds {
        bool blurred = false;
        bool aux = false;
        bool grain = false;
    };

    inline OpticsScratchNeeds build_optics_scratch_needs(
        bool wantGlareBlur,
        bool wantGrainBlur,
        bool wantGrainSublayers,
        bool wantGrainMix) {
        OpticsScratchNeeds needs{};
        needs.blurred = needs_blurred_optics_scratch(wantGlareBlur, wantGrainBlur, wantGrainSublayers);
        needs.aux = wantGrainSublayers;
        needs.grain = wantGrainMix;
        return needs;
    }

    inline bool spatial_dir_enabled(const Couplers::Runtime& dirRT) {
        return dirRT.active && is_positive_finite(dirRT.spatialSigmaPixels);
    }

    inline float sanitize_dir_dmax_value(float value) {
        if (!is_finite(value) || value <= 1e-4f) {
            return 1.0f;
        }
        return value;
    }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    using CuCtxGetCurrentFn = CUresult(CUDAAPI*)(CUcontext*);

    struct CudaDriverDispatch {
        CuCtxGetCurrentFn cuCtxGetCurrent = nullptr;
        const char* loadError = nullptr;
    };

    const CudaDriverDispatch& cuda_driver_dispatch() {
        static CudaDriverDispatch dispatch{};
        static std::once_flag once;
        std::call_once(once, []() {
#if defined(_WIN32)
            HMODULE module = GetModuleHandleA("nvcuda.dll");
            if (!module) {
                module = LoadLibraryA("nvcuda.dll");
            }
            if (!module) {
                dispatch.loadError = "nvcuda.dll not available";
                return;
            }
            dispatch.cuCtxGetCurrent =
                reinterpret_cast<CuCtxGetCurrentFn>(GetProcAddress(module, "cuCtxGetCurrent"));
            if (!dispatch.cuCtxGetCurrent) {
                dispatch.loadError = "cuCtxGetCurrent symbol not found";
                return;
            }
#else
            dispatch.loadError = "dynamic cuCtxGetCurrent loader unsupported on this platform";
            return;
#endif
        });
        return dispatch;
    }

    bool query_current_cuda_context(void*& outContextOpaque, std::string& outError) {
        outContextOpaque = nullptr;
        outError.clear();

        const CudaDriverDispatch& dispatch = cuda_driver_dispatch();
        if (!dispatch.cuCtxGetCurrent) {
            outError = nonempty_cstr_or(dispatch.loadError, "driver dispatch unavailable");
            return false;
        }

        CUcontext currentContext = nullptr;
        const CUresult ctxResult = dispatch.cuCtxGetCurrent(&currentContext);
        if (ctxResult != CUDA_SUCCESS) {
            outError = "cuCtxGetCurrent failed (code=";
            outError += std::to_string(static_cast<int>(ctxResult));
            outError += ")";
            return false;
        }
        if (!currentContext) {
            outError = "current CUDA context is null";
            return false;
        }

        outContextOpaque = reinterpret_cast<void*>(currentContext);
        return true;
    }

    std::string ascii_lower_copy(const std::string& value) {
        std::string out = value;
        std::transform(out.begin(), out.end(), out.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        return out;
    }

    bool text_has_context_loss_marker(const std::string& text) {
        if (text.empty()) {
            return false;
        }
        const std::string lower = ascii_lower_copy(text);
        return lower.find("context is destroyed") != std::string::npos ||
               lower.find("context destroyed") != std::string::npos ||
               lower.find("cudaerrorcontextisdestroyed") != std::string::npos ||
               lower.find("device lost") != std::string::npos ||
               lower.find("driver shutting down") != std::string::npos ||
               lower.find("context reset") != std::string::npos ||
               lower.find("device unavailable") != std::string::npos ||
               lower.find("cudaerrordeviceuninitialized") != std::string::npos;
    }

    bool is_cuda_context_loss_signal(cudaError_t error, const std::string& detail) {
        if (text_has_context_loss_marker(detail)) {
            return true;
        }
        if (error == cudaSuccess || error == cudaErrorNotReady) {
            return false;
        }
        const char* errorName = cudaGetErrorName(error);
        if (errorName && text_has_context_loss_marker(errorName)) {
            return true;
        }
        const char* errorText = cudaGetErrorString(error);
        if (errorText && text_has_context_loss_marker(errorText)) {
            return true;
        }
        return false;
    }

    void recover_context_loss_state(
        InstanceState* instanceState,
        const JuicerCuda::ResourceManager::DeviceContextKey& key,
        const char* stage,
        cudaError_t error,
        const std::string& detail) {
        if (!instanceState) {
            return;
        }
        if (!is_cuda_context_loss_signal(error, detail)) {
            return;
        }

        std::string retireError;
#if JUICER_DIAGNOSTICS_COMPILED
        const char* stageName = nonempty_cstr_or(stage, "unknown_stage");
        const bool traceInfo = JTRACE_ENABLED(1);
        const bool retireAccepted = JuicerProcess::root().retire_reset_context(
            key.deviceId,
            key.contextOpaque,
            retireError);
#else
        JuicerProcess::root().retire_reset_context(
            key.deviceId,
            key.contextOpaque,
            retireError);
#endif

#if JUICER_DIAGNOSTICS_COMPILED
        bool latchCleared = false;
#endif
        {
            std::lock_guard<std::mutex> lock(instanceState->submissionSnapshotLatchMutex);
            if (instanceState->submissionSnapshotLatchValid &&
                instanceState->submissionSnapshotLatch.deviceContextKey == key) {
                instanceState->submissionSnapshotLatch = JuicerCuda::ResourceManager::SubmissionSnapshot{};
                instanceState->submissionSnapshotLatchValid = false;
#if JUICER_DIAGNOSTICS_COMPILED
                latchCleared = true;
#endif
            }
        }

#if JUICER_DIAGNOSTICS_COMPILED
        if (traceInfo) {
            const std::uintptr_t contextBits = reinterpret_cast<std::uintptr_t>(key.contextOpaque);
            std::string msg;
            msg.reserve(192);
            msg = "stage=";
            msg += stageName;
            msg += " device_id=";
            msg += std::to_string(key.deviceId);
            msg += " context=";
            msg += std::to_string(contextBits);
            msg += " error_code=";
            msg += std::to_string(static_cast<int>(error));
            msg += " retire_accepted=";
            msg += std::to_string(bool_to_i32(retireAccepted));
            msg += " latch_cleared=";
            msg += std::to_string(bool_to_i32(latchCleared));
            if (!retireError.empty()) {
                msg += " retire_error=";
                msg += retireError;
            }
            JTRACE("MSLCY", msg);
        }
#endif
    }
#endif

    std::int64_t frame_index_from_time(double time) {
        return static_cast<std::int64_t>(std::floor(finite_or(time, 0.0)));
    }

    std::uint64_t session_seed_or_default(std::uint64_t sessionSeed) {
        return (sessionSeed != 0) ? sessionSeed : 1;
    }

    std::uint64_t instance_token_or_session_seed(std::uint64_t instanceToken, std::uint64_t sessionSeed) {
        return (instanceToken != 0) ? instanceToken : session_seed_or_default(sessionSeed);
    }

#if JUICER_DIAGNOSTICS_COMPILED
    const char* runtime_lease_outcome_label(bool waitedForLease) {
        return waitedForLease ? "wait_acquired" : "acquired";
    }

    const char* submission_snapshot_action_label(bool reusingSnapshotLatch) {
        return reusingSnapshotLatch ? "reuse" : "new";
    }
#endif

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__) &&                                    \
    ((defined(JUICER_CUDA_VALIDATE_PRIMITIVES) && (JUICER_CUDA_VALIDATE_PRIMITIVES != 0)) || \
     (defined(JUICER_CUDA_SELF_CHECK) && (JUICER_CUDA_SELF_CHECK != 0)))
    struct DiagnosticsHookPolicy {
        bool diagnosticsMode = false;
        bool validatePrimitives = false;
        bool runtimeSelfCheck = false;
    };

    inline const char* diagnostics_mode_label(bool modeEnabled) {
        return modeEnabled ? "diagnostics" : "serving";
    }

    bool parse_env_toggle(const char* name, bool fallback) {
        return JuicerLogging::parse_env_int(name, bool_to_i32(fallback)) != 0;
    }

    const DiagnosticsHookPolicy& diagnostics_hook_policy() {
        static const DiagnosticsHookPolicy policy = []() {
            DiagnosticsHookPolicy out{};
            out.diagnosticsMode = parse_env_toggle("JUICER_DIAGNOSTICS_MODE", false);
            out.validatePrimitives = parse_env_toggle("JUICER_DIAGNOSTICS_VALIDATE", true);
            out.runtimeSelfCheck = parse_env_toggle("JUICER_DIAGNOSTICS_SELF_CHECK", true);
            return out;
        }();
        return policy;
    }

#if defined(JUICER_CUDA_VALIDATE_PRIMITIVES) && (JUICER_CUDA_VALIDATE_PRIMITIVES != 0)
    void trace_validation_hook_state_once(
        bool compiled,
        bool modeEnabled,
        bool toggleEnabled,
        bool verboseEnabled,
        bool active) {
        static std::once_flag once;
        std::call_once(once, [&]() {
            if (!JTRACE_ENABLED(2)) {
                return;
            }
            const char* reason = "active";
            if (!compiled) {
                reason = "compile_disabled";
            } else if (!modeEnabled) {
                reason = "diagnostics_mode_disabled";
            } else if (!toggleEnabled) {
                reason = "validation_toggle_disabled";
            } else if (!verboseEnabled) {
                reason = "diagnostics_level_below_verbose";
            }
            std::string msg;
            msg.reserve(160);
            msg = "event=diagnostics_hook";
            msg += " hook=validation";
            msg += " mode=";
            msg += diagnostics_mode_label(modeEnabled);
            msg += " compiled=";
            msg += std::to_string(bool_to_i32(compiled));
            msg += " toggle_enabled=";
            msg += std::to_string(bool_to_i32(toggleEnabled));
            msg += " verbose_enabled=";
            msg += std::to_string(bool_to_i32(verboseEnabled));
            msg += " active=";
            msg += std::to_string(bool_to_i32(active));
            msg += " reason=";
            msg += reason;
            JTRACE("MSDBG", msg);
        });
    }
#endif

#if defined(JUICER_CUDA_SELF_CHECK) && (JUICER_CUDA_SELF_CHECK != 0)
    void trace_self_check_hook_state_once(
        bool compiled,
        bool modeEnabled,
        bool toggleEnabled,
        bool active) {
        static std::once_flag once;
        std::call_once(once, [&]() {
            if (!JTRACE_ENABLED(2)) {
                return;
            }
            const char* reason = "active";
            if (!compiled) {
                reason = "compile_disabled";
            } else if (!modeEnabled) {
                reason = "diagnostics_mode_disabled";
            } else if (!toggleEnabled) {
                reason = "self_check_toggle_disabled";
            }
            std::string msg;
            msg.reserve(144);
            msg = "event=diagnostics_hook";
            msg += " hook=self_check";
            msg += " mode=";
            msg += diagnostics_mode_label(modeEnabled);
            msg += " compiled=";
            msg += std::to_string(bool_to_i32(compiled));
            msg += " toggle_enabled=";
            msg += std::to_string(bool_to_i32(toggleEnabled));
            msg += " active=";
            msg += std::to_string(bool_to_i32(active));
            msg += " reason=";
            msg += reason;
            JTRACE("MSDBG", msg);
        });
    }
#endif
#endif

    std::uint64_t make_seed_base(std::uintptr_t clipToken,
                                 std::int64_t frameIndex,
                                 std::uint64_t sessionSeed,
                                 std::uint64_t passId) {
        const std::uint64_t fields[4] = {
            static_cast<std::uint64_t>(clipToken),
            static_cast<std::uint64_t>(frameIndex),
            sessionSeed,
            passId};
        std::uint64_t h = Hash::hash_bytes(fields, sizeof(fields));
        if (h == 0) {
            h = 1;
        }
        return h;
    }

    std::uint64_t seed_base_for_pass(
        std::uint64_t sessionSeed,
        std::uintptr_t clipToken,
        std::int64_t frameIndex,
        std::uint64_t passId) {
        return make_seed_base(
            clipToken,
            frameIndex,
            session_seed_or_default(sessionSeed),
            passId);
    }

} // namespace

namespace JuicerProcScanner {

    std::uint64_t hash_scanner_runtime_lane(
        const WorkingState* ws,
        const Scanner::Settings& settings,
        const Scanner::Options& options,
        std::uint32_t frameBoundsVersion) {
        if (!ws) {
            return 0;
        }
        const std::uint64_t settingsHash = hash_scanner_settings(settings, options);
        if (settingsHash == 0) {
            return 0;
        }

        std::uint64_t negStaticHash = 0;
        if (ws->negativeScannerValid) {
            Scanner::ScannerStaticKey negStaticKey = ws->negativeStaticKey;
            Scanner::finalize_static_key(negStaticKey);
            negStaticHash = negStaticKey.hash;
        }

        const bool printValid = ws->printScannerValid;
        std::uint64_t printStaticHash = 0;
        if (printValid) {
            Scanner::ScannerStaticKey printStaticKey = ws->printStaticKey;
            Scanner::finalize_static_key(printStaticKey);
            printStaticHash = printStaticKey.hash;
        }

        const std::uint64_t fields[] = {
            settingsHash,
            static_cast<std::uint64_t>(frameBoundsVersion),
            bool_to_u64(ws->negativeScannerValid),
            negStaticHash,
            bool_to_u64(printValid),
            printStaticHash};
        return Hash::hash_bytes(fields, sizeof(fields));
    }

    ScannerMediumRuntimeBinding bind_scanner_medium_runtime(
        const WorkingState& ws,
        bool printActive,
        bool hasPrintGlareOverride,
        const Profiles::ProfileGlare* printGlareOverride,
        bool forcePrintGlareHash) {
        ScannerMediumRuntimeBinding binding{};
        const ScannerMediumSelection selection = select_scanner_medium(ws, printActive);
        binding.selectedRuntime = selection.runtime;
        binding.valid = selection.valid;
        binding.label = selection.label;
        binding.usesPrintOverride = false;
        if (!printActive) {
            return binding;
        }

        binding.printOverride = ws.printMediumRuntime;
        if (hasPrintGlareOverride && printGlareOverride) {
            apply_glare_override_fields(binding.printOverride.glare, *printGlareOverride);
        }
        if (forcePrintGlareHash || hasPrintGlareOverride) {
            assign_glare_hash_or_throw(binding.printOverride);
        }
        binding.usesPrintOverride = true;
        return binding;
    }

    ScannerPreflightResult validate_scanner_preflight_or_throw(
        bool runtimeValid,
        const char* mediumLabel,
        const Scanner::ScannerMediumRuntime* mediumRuntime,
        bool traceInfoEnabled,
        bool traceVerboseEnabled,
        const char* path,
        const char* fatalTag) {
        ScannerPreflightResult scannerPreflight{};
        std::string scannerPreflightError;
        if (!validate_scanner_preflight_runtime(
                runtimeValid,
                mediumLabel,
                mediumRuntime,
                scannerPreflight,
                scannerPreflightError)) {
            trace_scanner_preflight_fail_if_enabled(
                traceInfoEnabled,
                path,
                mediumLabel,
                scannerPreflightError,
                fatalTag);
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        trace_scanner_preflight_ok_if_enabled(
            traceVerboseEnabled,
            path,
            mediumLabel,
            scannerPreflight.staticKey.hash);
        return scannerPreflight;
    }

    ScannerKeyBundle make_scanner_key_bundle_or_throw(
        const Scanner::ScannerStaticKey& staticKey,
        const Scanner::Settings& settings,
        const Scanner::Options& options,
        std::uint32_t frameBoundsVersion) {
        ScannerKeyBundle bundle{};
        bundle.runtimeKey.settingsHash = hash_scanner_settings(settings, options);
        bundle.runtimeKey.frameBoundsVersion = frameBoundsVersion;
        if (bundle.runtimeKey.settingsHash == 0) {
            JTRACE("HASH", "FATAL: scanner runtime settings hash invalid");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        Scanner::finalize_runtime_key(bundle.runtimeKey);

        bundle.scannerKey.staticKey = staticKey;
        bundle.scannerKey.runtimeKey = bundle.runtimeKey;
        Scanner::finalize_scanner_key(bundle.scannerKey);
        if (bundle.scannerKey.hash == 0) {
            JTRACE("HASH", "FATAL: scanner combined key invalid");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        return bundle;
    }

} // namespace JuicerProcScanner

namespace {

    JuicerCuda::AutoExposurePreviewDescriptor make_auto_exposure_preview_descriptor(
        const OfxRectI& sourceBounds,
        const OfxRectI& meterBounds,
        Spektrafilm::AutoExposureMethod method) {
        JuicerCuda::AutoExposurePreviewDescriptor descriptor{};
        descriptor.sourceX1 = sourceBounds.x1;
        descriptor.sourceY1 = sourceBounds.y1;
        descriptor.sourceX2 = sourceBounds.x2;
        descriptor.sourceY2 = sourceBounds.y2;
        descriptor.meterX1 = meterBounds.x1;
        descriptor.meterY1 = meterBounds.y1;
        descriptor.meterX2 = meterBounds.x2;
        descriptor.meterY2 = meterBounds.y2;
        descriptor.method = method;
        descriptor.sampling = JuicerCuda::AutoExposurePreviewDescriptor::Sampling::NearestNeighbor;

        const int width = std::max(0, meterBounds.x2 - meterBounds.x1);
        const int height = std::max(0, meterBounds.y2 - meterBounds.y1);
        const int longEdge = std::max(width, height);
        descriptor.previewWidth = width;
        descriptor.previewHeight = height;
        if (longEdge > JuicerCuda::AutoExposurePreviewDescriptor::kMaxLongEdge) {
            if (width >= height) {
                descriptor.previewWidth = JuicerCuda::AutoExposurePreviewDescriptor::kMaxLongEdge;
                descriptor.previewHeight = std::max(
                    1,
                    (height * JuicerCuda::AutoExposurePreviewDescriptor::kMaxLongEdge + width / 2) /
                        width);
            } else {
                descriptor.previewHeight = JuicerCuda::AutoExposurePreviewDescriptor::kMaxLongEdge;
                descriptor.previewWidth = std::max(
                    1,
                    (width * JuicerCuda::AutoExposurePreviewDescriptor::kMaxLongEdge + height / 2) /
                        height);
            }
        }
        descriptor.hash = Hash::kFnvOffset;
        Hash::hash_bytes_update(descriptor.hash, &descriptor.sourceX1, sizeof(descriptor.sourceX1));
        Hash::hash_bytes_update(descriptor.hash, &descriptor.sourceY1, sizeof(descriptor.sourceY1));
        Hash::hash_bytes_update(descriptor.hash, &descriptor.sourceX2, sizeof(descriptor.sourceX2));
        Hash::hash_bytes_update(descriptor.hash, &descriptor.sourceY2, sizeof(descriptor.sourceY2));
        Hash::hash_bytes_update(descriptor.hash, &descriptor.meterX1, sizeof(descriptor.meterX1));
        Hash::hash_bytes_update(descriptor.hash, &descriptor.meterY1, sizeof(descriptor.meterY1));
        Hash::hash_bytes_update(descriptor.hash, &descriptor.meterX2, sizeof(descriptor.meterX2));
        Hash::hash_bytes_update(descriptor.hash, &descriptor.meterY2, sizeof(descriptor.meterY2));
        Hash::hash_bytes_update(descriptor.hash, &descriptor.previewWidth, sizeof(descriptor.previewWidth));
        Hash::hash_bytes_update(descriptor.hash, &descriptor.previewHeight, sizeof(descriptor.previewHeight));
        Hash::hash_bytes_update(descriptor.hash, &descriptor.method, sizeof(descriptor.method));
        Hash::hash_bytes_update(descriptor.hash, &descriptor.sampling, sizeof(descriptor.sampling));
        if (descriptor.hash == 0) {
            descriptor.hash = 1;
        }
        return descriptor;
    }

    float compute_print_midgray_factor(
        const WorkingState& ws,
        const Print::Runtime& prt,
        const Print::Params& printParams,
        const Couplers::Runtime& dirRT) {
        const float exposureCompScale = printParams.exposureCompensationEnabled
                                            ? printParams.exposureCompensationScale
                                            : 1.0f;

        float kMid = Pipeline::PipelineRunner::compute_midgray_factor(
            ws,
            prt,
            printParams,
            dirRT,
            exposureCompScale);
        kMid = positive_finite_or(kMid, 1.0f);

        return kMid;
    }

    int stbn_frame_index(std::int64_t frameIndex, int frames, std::uint64_t sessionSeed) {
        if (frames <= 0) {
            return 0;
        }
        const std::int64_t phase = static_cast<std::int64_t>(sessionSeed % static_cast<std::uint64_t>(frames));
        std::int64_t t = frameIndex + phase;
        int f = static_cast<int>(t % frames);
        if (f < 0) {
            f += frames;
        }
        return f;
    }

    int stbn_offset(std::uint64_t sessionSeed, int dim, std::uint64_t salt) {
        if (dim <= 0) {
            return 0;
        }
        const std::uint64_t fields[2] = {sessionSeed, salt};
        const std::uint64_t h = Hash::hash_bytes(fields, sizeof(fields));
        return static_cast<int>(h % static_cast<std::uint64_t>(dim));
    }

    constexpr std::uint64_t kSeedPassWeave = 3;

    double hash_to_unit(std::uint64_t h) {
        // Convert to [0,1) using the top 53 bits (double mantissa).
        constexpr double kInv = 1.0 / 9007199254740992.0; // 2^53
        return static_cast<double>(h >> 11) * kInv;
    }

    double phase_from_seed(std::uint64_t sessionSeed, std::uint64_t passId, int axis, int component) {
        const std::uint64_t fields[4] = {
            sessionSeed,
            passId,
            static_cast<std::uint64_t>(axis),
            static_cast<std::uint64_t>(component)};
        std::uint64_t h = Hash::hash_bytes(fields, sizeof(fields));
        if (h == 0) {
            h = 1;
        }
        constexpr double kTwoPi = 6.28318530717958647692;
        return hash_to_unit(h) * kTwoPi;
    }

    double sin_sum(const double* freqs, int count, std::uint64_t sessionSeed, std::uint64_t passId, int axis, int componentOffset, double timeSeconds) {
        constexpr double kTwoPi = 6.28318530717958647692;
        double sum = 0.0;
        const double* freqIt = freqs;
        for (int i = 0; i < count; ++i, ++freqIt) {
            const double phase = phase_from_seed(sessionSeed, passId, axis, componentOffset + i);
            sum += std::sin(kTwoPi * (*freqIt) * timeSeconds + phase);
        }
        return sum;
    }

    struct GateWeaveSignal {
        float dxPx = 0.0f;
        float dyPx = 0.0f;
        float cosRot = 1.0f;
        float sinRot = 0.0f;
    };

    GateWeaveSignal compute_gate_weave(
        std::uint64_t sessionSeed,
        double timeSeconds,
        double translateRmsUm,
        double rotateRmsDeg,
        double pixelSizeUm,
        double amount) {
        GateWeaveSignal out{};
        if (!(amount > 0.0) || !is_positive_finite(pixelSizeUm)) {
            return out;
        }

        const double translateRms = translateRmsUm * amount;
        const double rotateRms = rotateRmsDeg * amount;
        if (!(translateRms > 0.0 || rotateRms > 0.0)) {
            return out;
        }

        constexpr double driftFreqs[] = {0.15, 0.35, 0.80};
        constexpr double jitterFreqs[] = {6.0, 12.0};
        constexpr int driftCount = static_cast<int>(sizeof(driftFreqs) / sizeof(driftFreqs[0]));
        constexpr int jitterCount = static_cast<int>(sizeof(jitterFreqs) / sizeof(jitterFreqs[0]));
        const double driftNorm = 1.0 / std::sqrt(0.5 * static_cast<double>(driftCount));
        const double jitterNorm = 1.0 / std::sqrt(0.5 * static_cast<double>(jitterCount));
        const double driftWeight = 0.85;
        const double jitterWeight = 0.15;
        const double weightNorm = 1.0 / std::sqrt(driftWeight * driftWeight + jitterWeight * jitterWeight);

        for (int axis = 0; axis < 2; ++axis) {
            const double drift = sin_sum(driftFreqs, driftCount, sessionSeed, kSeedPassWeave, axis, 0, timeSeconds) * driftNorm;
            const double jitter = sin_sum(jitterFreqs, jitterCount, sessionSeed, kSeedPassWeave, axis, 10, timeSeconds) * jitterNorm;
            const double composite = (driftWeight * drift + jitterWeight * jitter) * weightNorm;
            const double deltaUm = composite * translateRms;
            const double deltaPx = deltaUm / pixelSizeUm;
            if (axis == 0) {
                out.dxPx = static_cast<float>(deltaPx);
            } else {
                out.dyPx = static_cast<float>(deltaPx);
            }
        }

        if (rotateRms > 0.0) {
            const double rotSignal = sin_sum(driftFreqs, driftCount, sessionSeed, kSeedPassWeave, 2, 0, timeSeconds) * driftNorm;
            const double rotDeg = rotSignal * rotateRms;
            const double rotRad = rotDeg * (3.14159265358979323846 / 180.0);
            out.cosRot = static_cast<float>(std::cos(rotRad));
            out.sinRot = static_cast<float>(std::sin(rotRad));
        }
        return out;
    }
} // namespace

namespace JuicerProc {

    // SF_TEMP_BRIDGE_CPUProductRendererBlocked owner=Phase1A remove=Phase3/Phase4:
    // retained CPU copy/read helpers are unreachable from product render after JuicerEffect::render cutoff.
    // Copied from main.cpp helper, unchanged behavior.
    void copyNonFloatRect(OFX::Image* src, OFX::Image* dst) {
        if (!src || !dst) {
            return;
        }
        const OfxRectI srcBounds = src->getBounds();
        const OfxRectI dstBounds = dst->getBounds();
        const int xStart = srcBounds.x1;
        const int xEnd = srcBounds.x2;
        const int yStart = srcBounds.y1;
        const int yEnd = srcBounds.y2;
        const OFX::PixelComponentEnum comps = src->getPixelComponents();
        const OFX::BitDepthEnum depth = src->getPixelDepth();

        const int nComponents = pixel_component_count(comps);
        if (nComponents <= 0) {
            return;
        }
        const int bytesPerComp = bytes_per_component(depth);
        if (bytesPerComp <= 0) {
            return;
        }
        const size_t bytesPerPixel = static_cast<size_t>(nComponents) * static_cast<size_t>(bytesPerComp);
        const int width = xEnd - xStart;
        if (width <= 0) {
            return;
        }
        if (yStart >= yEnd) {
            return;
        }
        for (int y = yStart; y < yEnd; ++y) {
            const std::uint8_t* sRow = read_u8_row_if_fully_covered(
                src, srcBounds, xStart, xEnd, y);
            std::uint8_t* dRow = write_u8_row_if_fully_covered(
                dst, dstBounds, xStart, xEnd, y);
            CopyRowBytesRequest copyRequest{};
            copyRequest.src = src;
            copyRequest.dst = dst;
            copyRequest.span = CopyRowSpan{xStart, xEnd, y};
            copyRequest.bytesPerPixel = bytesPerPixel;
            copyRequest.srcRow = sRow;
            copyRequest.dstRow = dRow;
            copy_row_bytes_with_fallback(copyRequest);
        }
    }

} // namespace JuicerProc

// --- Spatial DIR: defensive curve utilities (monotonic + robust interpolation) ---

inline unsigned int positive_u32_or(unsigned int value, unsigned int fallback);
inline int positive_i32_or(int value, int fallback);

static unsigned int compute_thread_count(int width, int height) {
    if (width <= 0 || height <= 0) {
        return 1u;
    }
    const unsigned int w = static_cast<unsigned int>(width);
    const unsigned int h = static_cast<unsigned int>(height);
    const std::uint64_t scaledPixels =
        static_cast<std::uint64_t>(std::min(w, 4096u)) * static_cast<std::uint64_t>(h);
    unsigned int nCPUs = static_cast<unsigned int>(scaledPixels / 4096u);
    if (nCPUs == 0) {
        nCPUs = 1;
    }
    static const unsigned int maxThreads = []() -> unsigned int {
        const unsigned int value = OFX::MultiThread::getNumCPUs();
        return positive_u32_or(value, 1u);
    }();
    nCPUs = std::min(nCPUs, maxThreads);
    return std::max(1u, nCPUs);
}

inline unsigned int positive_u32_or(unsigned int value, unsigned int fallback) {
    return (value > 0) ? value : fallback;
}

inline int positive_i32_or(int value, int fallback) {
    return (value > 0) ? value : fallback;
}

inline std::uint64_t upload_core_hash_or_core_hash(const WorkingState& ws) {
    return (ws.uploadCoreHash != 0) ? ws.uploadCoreHash : ws.coreHash;
}

struct GateMaskHashInputs {
    std::uint64_t sessionSeed = 1;
    int originX = 0;
    int originY = 0;
    int width = 0;
    int height = 0;
    float pixelSizeUm = 0.0f;
    float gateDustAmount = 0.0f;
    float gateScratchAmount = 0.0f;
};

static std::uint64_t make_gate_mask_hash(const GateMaskHashInputs& inputs) {
    const std::uint64_t originXBits = static_cast<std::uint64_t>(inputs.originX);
    const std::uint64_t originYBits = static_cast<std::uint64_t>(inputs.originY);
    const std::uint64_t widthBits = static_cast<std::uint64_t>(inputs.width);
    const std::uint64_t heightBits = static_cast<std::uint64_t>(inputs.height);
    std::uint64_t h = Hash::kFnvOffset;
    Hash::hash_bytes_update(h, &inputs.sessionSeed, sizeof(inputs.sessionSeed));
    Hash::hash_bytes_update(h, &originXBits, sizeof(originXBits));
    Hash::hash_bytes_update(h, &originYBits, sizeof(originYBits));
    Hash::hash_bytes_update(h, &widthBits, sizeof(widthBits));
    Hash::hash_bytes_update(h, &heightBits, sizeof(heightBits));
    Hash::hash_bytes_update(h, &inputs.pixelSizeUm, sizeof(inputs.pixelSizeUm));
    Hash::hash_bytes_update(h, &inputs.gateDustAmount, sizeof(inputs.gateDustAmount));
    Hash::hash_bytes_update(h, &inputs.gateScratchAmount, sizeof(inputs.gateScratchAmount));
    if (h == 0) {
        h = 1;
    }
    return h;
}

bool curve_ok(const Spectral::Curve& c) {
    const size_t N = c.lambda_nm.size();
    if (N < 2 || c.linear.size() != N)
        return false;
    const float* lambdaData = c.lambda_nm.data();
    float prev = lambdaData[0];
    if (!is_finite(prev))
        return false;
    for (size_t i = 1; i < N; ++i) {
        const float xi = lambdaData[i];
        if (!is_finite(xi))
            return false;
        if (xi < prev)
            return false; // allow duplicates (xi == prev), but never decreasing
        prev = xi;
    }
    return true;
}


// JuicerProcessor method definitions matching JuicerProcessing.h

JuicerProcessor::JuicerProcessor(OFX::ImageEffect& effect)
    : OFX::ImageProcessor(effect), _srcImg(nullptr), _nComponents(0), _scannerOptions{}, _scannerSettings{}, _printParams{}, _halationOverride{}, _hasHalationOverride(false), _grainOverride{}, _hasGrainOverride(false), _printGlareOverride{}, _hasPrintGlareOverride(false), _dirRT{}, _prt(nullptr), _ws(nullptr), _wsReady(false), _printReady(false), _exposureScale(1.0f), _outputEncoding{}, _scratch{}, _density{}, _frameBoundsVersion(0), _pixelSizeUm(0.0f) {
}

void JuicerProcessor::setSrcDst(const SourceDestinationImages& images) {
    _srcImg = images.src;
    setDstImg(images.dst);
}

void JuicerProcessor::setFrameRequest(const FrameRequest& request) {
    // SF_TEMP_BRIDGE_FrameRequestSideChannelCopy owner=Phase3C direct-launch/Phase4 print audit:
    // allowed=JuicerEffect::render adapter; hash_owner=none; output_impact=blocked legacy path;
    // direct film-raw/develop and auto-exposure policy now reads request.recipe through
    // _recipeHold; remove remaining scanner/print/optics/grain copies in their owning phases.
    setRenderWindowRect(request.renderWindow);
    setComponents(request.components);
    _scannerOptions = request.scannerOptions;
    _scannerSettings = request.scannerSettings;
    _printParams = request.printParams;
    _halationOverride = request.halationOverride;
    _hasHalationOverride = request.hasHalationOverride;
    _grainOverride = request.grainOverride;
    _hasGrainOverride = request.hasGrainOverride;
    _printGlareOverride = request.printGlareOverride;
    if (request.hasPrintGlareOverride) {
        clear_glare_compensation_fields(_printGlareOverride);
    }
    _hasPrintGlareOverride = request.hasPrintGlareOverride;
    _dirRT = request.dirRuntime;

    _recipeHold = request.recipe;
    _wsHold = request.workingState;
    setWorkingState(_wsHold.get(), request.workingStateReady);
    setPrintRuntime(
        request.printRuntime ? request.printRuntime : ((_ws && _ws->printRT) ? _ws->printRT.get() : nullptr),
        request.printRuntimeReady);
    setExposure(request.exposureScale);
    CameraAutoExposureSettings autoExposureSettings{};
    autoExposureSettings.enabled = request.cameraAutoEnabled;
    autoExposureSettings.meteringMethod = request.cameraMeteringMethod;
    autoExposureSettings.sliderEV = request.cameraSliderEV;
    setCameraAutoExposure(autoExposureSettings);
    setAutoExposureMeterBounds(
        request.autoExposureMeterBounds,
        request.autoExposureMeterBoundsValid);
    _outputEncoding = request.outputEncoding;
    setSessionTokens(SessionTokens{request.sessionSeed, request.instanceToken});
    setClipToken(request.clipToken);
    setGateWeaveAmount(request.gateWeaveAmount);
    setFrameTime(request.frameTime);
    setFrameRate(request.frameRate);
    setFrameBoundsVersion(request.frameBoundsVersion);
    setPixelSizeUm(request.pixelSizeUm);
    setRenderHints(
        request.interactiveRenderStatus,
        request.renderQualityDraft,
        request.sequentialRenderStatus);
}

void JuicerProcessor::setRenderWindowRect(const OfxRectI& rect) {
    setRenderWindow(rect);
}
void JuicerProcessor::setComponents(int n) {
    _nComponents = n;
}
void JuicerProcessor::setScannerOptions(const Scanner::Options& o) {
    _scannerOptions = o;
}
void JuicerProcessor::setScannerSettings(const Scanner::Settings& s) {
    _scannerSettings = s;
}
void JuicerProcessor::setPrintParams(const Print::Params& p) {
    _printParams = p;
}
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
    clear_glare_compensation_fields(_printGlareOverride);
    _hasPrintGlareOverride = true;
}
void JuicerProcessor::setDirRuntime(const Couplers::Runtime& rt) {
    _dirRT = rt;
}
void JuicerProcessor::setWorkingState(const WorkingState* ws, bool wsReady) {
    if (_wsHold.get() != ws) {
        _wsHold.reset();
    }
    _ws = ws;
    _wsReady = wsReady;
    // Align DIR normalization constants to per-instance maxima if available
    if (_wsReady && _ws) {
        const float* srcDMax = _ws->dMax;
        float* dstDMax = _dirRT.dMax;
        for (int i = 0; i < 3; ++i, ++srcDMax, ++dstDMax) {
            *dstDMax = sanitize_dir_dmax_value(*srcDMax);
        }
    }
}
void JuicerProcessor::setPrintRuntime(const Print::Runtime* prt, bool printReady) {
    _prt = prt;
    _printReady = printReady;
}
void JuicerProcessor::setExposure(float exposureScale) {
    _exposureScale = positive_finite_or(exposureScale, 1.0f);
}

void JuicerProcessor::setCameraAutoExposure(const CameraAutoExposureSettings& settings) {
    _cameraAutoEnabled = settings.enabled;
    _cameraMeteringMethod = settings.meteringMethod;
    _cameraSliderEV = settings.sliderEV;
}

void JuicerProcessor::setAutoExposureMeterBounds(const OfxRectI& bounds, bool valid) {
    _autoExposureMeterBounds = bounds;
    _autoExposureMeterBoundsValid = valid;
}

void JuicerProcessor::setOutputEncoding(const OutputEncoding::Params& p) {
    _outputEncoding = p;
}

void JuicerProcessor::setInstanceState(InstanceState* s) {
    _instanceState = s;
}

void JuicerProcessor::setSessionTokens(const SessionTokens& tokens) {
    _sessionSeed = session_seed_or_default(tokens.sessionSeed);
    _instanceToken = instance_token_or_session_seed(tokens.instanceToken, _sessionSeed);
}

void JuicerProcessor::setClipToken(std::uintptr_t token) {
    _clipToken = token;
}

void JuicerProcessor::setGateWeaveAmount(double amount) {
    _gateWeaveAmount = finite_or(amount, _gateWeaveAmount);
}

void JuicerProcessor::setFrameTime(double time) {
    _timeFrames = finite_or(time, 0.0);
    _frameIndex = frame_index_from_time(_timeFrames);
    _frameTimeHash = Hash::hash_bytes(&_frameIndex, sizeof(_frameIndex));
    if (_frameTimeHash == 0) {
        _frameTimeHash = 1;
    }
}

void JuicerProcessor::setFrameRate(double frameRate) {
    _frameRate = positive_finite_or(frameRate, 0.0);
}

void JuicerProcessor::setFrameBoundsVersion(std::uint32_t v) {
    _frameBoundsVersion = v;
}

void JuicerProcessor::setPixelSizeUm(float pixelSizeUm) {
    _pixelSizeUm = sanitize_nonnegative_or(pixelSizeUm, 0.0f);
}

void JuicerProcessor::setRenderHints(bool interactiveRenderStatus, bool renderQualityDraft, bool sequentialRenderStatus) {
    _renderInteractiveStatus = interactiveRenderStatus;
    _renderQualityDraft = renderQualityDraft;
    _renderSequentialStatus = sequentialRenderStatus;
}

JuicerProcessor::RenderContext JuicerProcessor::prepareRenderContext() const {
    RenderContext ctx{};
    ctx.window = _renderWindow;
    ctx.width = _renderWindow.x2 - _renderWindow.x1;
    ctx.height = _renderWindow.y2 - _renderWindow.y1;
    ctx.exposureScaleSafe = positive_finite_or(_exposureScale, 1.0f);
    ctx.useSpatialDIR = (spatial_dir_enabled(_dirRT) &&
                         _nComponents >= 3 && _wsReady && _ws);
    ctx.printActive = print_pipeline_active(
        _wsReady,
        _ws,
        _printReady,
        _prt,
        _printParams.bypass);

    ctx.kMidSpectral = 1.0f;
    if (ctx.printActive) {
        ctx.kMidSpectral = compute_print_midgray_factor(
            *_ws,
            *_prt,
            _printParams,
            _dirRT);
    }

    ctx.pixelSizeUm = positive_finite_or(_pixelSizeUm, 0.0f);
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
    } catch (...) {
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

    _density.medium = scanner_medium_from_print_active(ctx.printActive);

    if (ctx.useSpatialDIR) {
        struct SpatialDIRUser {
            OFX::ImageEffect* effect = nullptr;
            OFX::Image* srcImg = nullptr;
            OfxRectI window{};
            OfxRectI srcBounds{};
            int windowWidth = 0;
            int windowHeight = 0;
            int srcComponents = 0;
            size_t srcStride = 0;
            bool canReadRowRgb = false;
            int cachedY = std::numeric_limits<int>::min();
            const float* cachedRow = nullptr;
        };
        SpatialDIRUser user{};
        user.effect = &_effect;
        user.srcImg = _srcImg;
        user.window = ctx.window;
        user.srcBounds = _srcImg->getBounds();
        user.windowWidth = std::max(0, ctx.window.x2 - ctx.window.x1);
        user.windowHeight = std::max(0, ctx.window.y2 - ctx.window.y1);
        user.srcComponents = _nComponents;
        user.srcStride = static_cast<size_t>(std::max(_nComponents, 0));
        user.canReadRowRgb = (_nComponents >= 3);

        SpatialDIR::Callbacks callbacks{};
        callbacks.user = &user;
        callbacks.fetchRGB = [](void* u, int xx, int yy, float rgb[3]) -> bool {
            auto* self = static_cast<SpatialDIRUser*>(u);
            if (xx < 0 || yy < 0 || xx >= self->windowWidth || yy >= self->windowHeight) {
                return false;
            }
            const int y = offset_from_start(self->window.y1, yy);
            if (self->cachedY != y) {
                self->cachedY = y;
                self->cachedRow = read_rgb_row_if_fully_covered(
                    self->srcImg,
                    self->srcBounds,
                    self->window.x1,
                    self->window.x2,
                    y);
            }
            if (self->cachedRow && self->canReadRowRgb) {
                const size_t xOffset = static_cast<size_t>(xx);
                const float* srcPix = self->cachedRow + xOffset * self->srcStride;
                copy_float3(rgb, srcPix);
                return true;
            }
            const int x = offset_from_start(self->window.x1, xx);
            return read_rgb_pixel_if_present(self->srcImg, x, y, rgb);
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

    std::atomic<bool> abortFlag{false};
    std::atomic<bool> failure{false};
    const int width = ctx.width;
    const int height = ctx.height;
    const int originX = ctx.window.x1;
    const int originY = ctx.window.y1;

    const unsigned int nThreads = std::max(1u, threadCount);
    if (ctx.printActive && _scratch.printScratchPerWorker.size() != nThreads) {
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
        const int originXEnd;
        const int originY;
        const OfxRectI srcBounds;
        const size_t srcStride;

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
            : self(self_), ctx(ctx_), runner(runner_), abortFlag(abortFlag_), failure(failure_), width(width_), height(height_), originX(originX_), originXEnd(originX_ + width_), originY(originY_), srcBounds(self_._srcImg->getBounds()), srcStride(static_cast<size_t>(self_._nComponents)) {
        }

        void multiThreadFunction(unsigned int threadId, unsigned int nThreads) override {
            const int rowsPerThread = (height + int(nThreads) - 1) / int(nThreads);
            const int yStart = rowsPerThread * int(threadId);
            if (yStart >= height) {
                return;
            }
            const int yEnd = std::min(height, yStart + rowsPerThread);
            const bool printActive = ctx.printActive;
            const bool useSpatialDIR = ctx.useSpatialDIR;

            JuicerProc::PrintPipelineScratch* printScratch = nullptr;
            if (printActive && threadId < self._scratch.printScratchPerWorker.size()) {
                printScratch = &self._scratch.printScratchPerWorker[threadId];
            }

            const auto& dirWorkspace = self._scratch.dirWorkspace;
            const float* filmRawB = ptr_if_enabled(useSpatialDIR, dirWorkspace.filmRaw_B.data());
            const float* filmRawG = ptr_if_enabled(useSpatialDIR, dirWorkspace.filmRaw_G.data());
            const float* filmRawR = ptr_if_enabled(useSpatialDIR, dirWorkspace.filmRaw_R.data());
            const float* corrY = ptr_if_enabled(useSpatialDIR, dirWorkspace.corrYBlur.data());
            const float* corrM = ptr_if_enabled(useSpatialDIR, dirWorkspace.corrMBlur.data());
            const float* corrC = ptr_if_enabled(useSpatialDIR, dirWorkspace.corrCBlur.data());

            Pipeline::DensityPixelInputs pxIn{};
            pxIn.exposureScale = ctx.exposureScaleSafe;
            pxIn.dirRuntime = &self._dirRT;
            pxIn.applyDirRuntime = true;
            if (printActive) {
                pxIn.printRuntime = self._prt;
                pxIn.printParams = &self._printParams;
                pxIn.midgrayFactor = ctx.kMidSpectral;
                pxIn.printScratch = printScratch;
            }
            if (useSpatialDIR) {
                pxIn.useFilmRawOverride = true;
                pxIn.useSpatialDIR = true;
            }
            Pipeline::DensityPixelOutputs pxOut{};
            const WorkingState& wsRef = *self._ws;
            float* densityC = self._density.c.data();
            float* densityM = self._density.m.data();
            float* densityY = self._density.y.data();
            OFX::Image* srcImg = self._srcImg;
            const size_t srcStride = this->srcStride;

            for (int yOff = yStart; yOff < yEnd && !should_abort_relaxed(abortFlag); ++yOff) {
                if (self._effect.abort()) {
                    mark_abort(abortFlag);
                    break;
                }
                const int y = offset_from_start(originY, yOff);
                const std::size_t rowOffset = linear_row_offset(yOff, width);
                if (useSpatialDIR) {
                    size_t idx = rowOffset;
                    const float* filmRawBIt = filmRawB + rowOffset;
                    const float* filmRawGIt = filmRawG + rowOffset;
                    const float* filmRawRIt = filmRawR + rowOffset;
                    const float* corrYIt = corrY + rowOffset;
                    const float* corrMIt = corrM + rowOffset;
                    const float* corrCIt = corrC + rowOffset;
                    for (int xOff = 0; xOff < width; ++xOff, ++idx) {
                        if (should_abort_relaxed(abortFlag)) {
                            break;
                        }
                        load_spatial_dir_density_inputs(
                            pxIn,
                            filmRawBIt,
                            filmRawGIt,
                            filmRawRIt,
                            corrYIt,
                            corrMIt,
                            corrCIt);
                        if (!runner.run_density_pixel(wsRef, pxIn, pxOut)) {
                            if (printActive) {
                                mark_failure_and_abort(failure, abortFlag);
                                break;
                            }
                            store_zero_density_triplet(densityC, densityM, densityY, idx);
                            continue;
                        }
                        if (printActive) {
                            if (pxOut.medium != Pipeline::DensityMedium::Print) {
                                mark_failure_and_abort(failure, abortFlag);
                                break;
                            }
                            store_density_triplet(densityC, densityM, densityY, idx, pxOut.printDensity.v);
                            continue;
                        }
                        store_density_triplet(densityC, densityM, densityY, idx, pxOut.negativeDensity.v);
                    }
                    continue;
                }

                const float* srcRow = read_rgb_row_if_fully_covered(
                    srcImg,
                    srcBounds,
                    originX,
                    originXEnd,
                    y);
                if (srcRow) {
                    const float* srcPixIt = srcRow;
                    size_t idx = rowOffset;
                    for (int xOff = 0; xOff < width; ++xOff, ++idx) {
                        if (should_abort_relaxed(abortFlag)) {
                            break;
                        }
                        copy_float3(pxIn.rgb.v, srcPixIt);
                        if (!runner.run_density_pixel(wsRef, pxIn, pxOut)) {
                            if (printActive) {
                                mark_failure_and_abort(failure, abortFlag);
                                break;
                            }
                            store_zero_density_triplet(densityC, densityM, densityY, idx);
                            srcPixIt += srcStride;
                            continue;
                        }
                        if (printActive) {
                            if (pxOut.medium != Pipeline::DensityMedium::Print) {
                                mark_failure_and_abort(failure, abortFlag);
                                break;
                            }
                            store_density_triplet(densityC, densityM, densityY, idx, pxOut.printDensity.v);
                            srcPixIt += srcStride;
                            continue;
                        }
                        store_density_triplet(densityC, densityM, densityY, idx, pxOut.negativeDensity.v);
                        srcPixIt += srcStride;
                    }
                    continue;
                }

                size_t idx = rowOffset;
                for (int xOff = 0; xOff < width; ++xOff, ++idx) {
                    if (should_abort_relaxed(abortFlag)) {
                        break;
                    }
                    const int x = offset_from_start(originX, xOff);
                    if (!read_rgb_pixel_if_present(srcImg, x, y, pxIn.rgb.v)) {
                        store_zero_density_triplet(densityC, densityM, densityY, idx);
                        continue;
                    }
                    if (!runner.run_density_pixel(wsRef, pxIn, pxOut)) {
                        if (printActive) {
                            mark_failure_and_abort(failure, abortFlag);
                            break;
                        }
                        store_zero_density_triplet(densityC, densityM, densityY, idx);
                        continue;
                    }
                    if (printActive) {
                        if (pxOut.medium != Pipeline::DensityMedium::Print) {
                            mark_failure_and_abort(failure, abortFlag);
                            break;
                        }
                        store_density_triplet(densityC, densityM, densityY, idx, pxOut.printDensity.v);
                        continue;
                    }
                    store_density_triplet(densityC, densityM, densityY, idx, pxOut.negativeDensity.v);
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
    if (should_abort_relaxed(abortFlag)) {
        return;
    }
}


void JuicerProcessor::renderScannerFromDensity(const RenderContext& ctx, unsigned int threadCount) {
    if (!_ws || !_wsReady) {
        return;
    }
    const bool traceInfo = JTRACE_ENABLED(1);
    const bool traceVerbose = JTRACE_ENABLED(3);
    const auto should_abort_effect = [this]() -> bool {
        return _effect.abort();
    };

    // The scanner now consumes only the staged CMY density slab; legacy RGB entry points are removed.
    const JuicerProcScanner::ScannerMediumRuntimeBinding scannerBinding = JuicerProcScanner::bind_scanner_medium_runtime(
        *_ws,
        ctx.printActive,
        _hasPrintGlareOverride,
        &_printGlareOverride,
        /*forcePrintGlareHash*/ true);
    const Scanner::ScannerMediumRuntime* mediumRuntime = scannerBinding.runtime();
    const bool scannerRuntimeValid = scannerBinding.valid;
    const char* cpuMediumLabel = scannerBinding.label;
    JuicerProcScanner::ScannerPreflightResult scannerPreflight = JuicerProcScanner::validate_scanner_preflight_or_throw(
        scannerRuntimeValid,
        cpuMediumLabel,
        mediumRuntime,
        traceInfo,
        traceVerbose,
        "cpu",
        "SCAN");

    mediumRuntime = scannerPreflight.mediumRuntime;
    const Scanner::ColorRuntime* colorPtr = scannerPreflight.colorRuntime;
    Scanner::ScannerStaticKey staticKey = scannerPreflight.staticKey;

    if (_density.medium != scannerPreflight.mediumRuntime->medium) {
        JTRACE("SCAN", "FATAL: density slab medium does not match selected scanner medium");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }

    const JuicerProcScanner::ScannerKeyBundle scannerKeys = JuicerProcScanner::make_scanner_key_bundle_or_throw(
        staticKey,
        _scannerSettings,
        _scannerOptions,
        _frameBoundsVersion);
    const Scanner::ScannerRuntimeKey& runtimeKey = scannerKeys.runtimeKey;
    const Scanner::ScannerKey& scannerKey = scannerKeys.scannerKey;

    struct ScannerRuntimeLease {
        InstanceState* state = nullptr;
        ScannerOptics::Runtime* runtime = nullptr;
        std::atomic<bool>* inUse = nullptr;
        const char* slotName = "none";

        explicit ScannerRuntimeLease(InstanceState* s) : state(s) {}

        ScannerOptics::Runtime* acquire() {
            if (!state) {
                slotName = "none";
                return nullptr;
            }
            bool expected = false;
            if (state->scannerRuntimeAInUse.compare_exchange_strong(
                    expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
                inUse = &state->scannerRuntimeAInUse;
                runtime = &state->scannerRuntimeA;
                slotName = "A";
                return runtime;
            }
            expected = false;
            if (state->scannerRuntimeBInUse.compare_exchange_strong(
                    expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
                inUse = &state->scannerRuntimeBInUse;
                runtime = &state->scannerRuntimeB;
                slotName = "B";
                return runtime;
            }
            slotName = "none";
            return nullptr;
        }

        ~ScannerRuntimeLease() {
            if (inUse) {
                inUse->store(false, std::memory_order_release);
            }
        }
    };

    ScannerRuntimeLease runtimeLease(_instanceState);
    ScannerOptics::Runtime* opticsRuntime = runtimeLease.acquire();
    constexpr std::uint32_t kScannerRuntimeLeaseMaxWaitUs = 16000u;
    constexpr std::uint32_t kScannerRuntimeLeasePollSleepUs = 50u;
#if JUICER_DIAGNOSTICS_COMPILED
    bool waitedForLease = false;
#endif
    const auto leaseWaitStart = std::chrono::steady_clock::now();
    const auto leaseDeadline = leaseWaitStart + std::chrono::microseconds(kScannerRuntimeLeaseMaxWaitUs);
    while (!opticsRuntime) {
#if JUICER_DIAGNOSTICS_COMPILED
        waitedForLease = true;
#endif
        if (should_abort_effect()) {
            JTRACE_VERBOSE("MSSRL", "event=runtime_lease outcome=abort");
            return;
        }
        if (std::chrono::steady_clock::now() >= leaseDeadline) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(kScannerRuntimeLeasePollSleepUs));
        opticsRuntime = runtimeLease.acquire();
    }
#if JUICER_DIAGNOSTICS_COMPILED
    const std::uint64_t leaseWaitUs = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                                     std::chrono::steady_clock::now() - leaseWaitStart)
                                                                     .count());
#endif
    if (!opticsRuntime) {
#if JUICER_DIAGNOSTICS_COMPILED
        if (traceInfo) {
            std::string msg;
            msg.reserve(128);
            msg = "event=runtime_lease outcome=timeout";
            msg += " wait_us=";
            msg += std::to_string(leaseWaitUs);
            msg += " wait_budget_us=";
            msg += std::to_string(static_cast<unsigned long long>(kScannerRuntimeLeaseMaxWaitUs));
            JTRACE("MSSRL", msg);
        }
#endif
        JTRACE("SCAN", "FATAL: scanner runtime lease unavailable after bounded wait");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
#if JUICER_DIAGNOSTICS_COMPILED
    if (traceVerbose) {
        std::string msg;
        msg.reserve(96);
        msg = "event=runtime_lease outcome=";
        msg += runtime_lease_outcome_label(waitedForLease);
        msg += " wait_us=";
        msg += std::to_string(leaseWaitUs);
        msg += " slot=";
        msg += runtimeLease.slotName;
        JTRACE_VERBOSE("MSSRL", msg);
    }
#endif

    const std::uint64_t seedBase = seed_base_for_pass(
        _sessionSeed,
        _clipToken,
        _frameIndex,
        kSeedPassGlare);

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
    optCtx.dstView.strideBytes = row_bytes_or_zero(_dstImg);
    optCtx.dstView.r = float_pixel_ptr_or_null(_dstImg, ctx.window.x1, ctx.window.y1);
    optCtx.dstView.g = optCtx.dstView.r;
    optCtx.dstView.b = optCtx.dstView.r;
    optCtx.dstView.a = alpha_channel_ptr_if_rgba(optCtx.dstView.r, _nComponents);
    optCtx.options = _scannerOptions;
    optCtx.settings = _scannerSettings;
    optCtx.runtimeKey = runtimeKey;
    optCtx.scannerKey = scannerKey;
    optCtx.seedBase = seedBase;
    optCtx.hasBaseline = has_baseline_or_false(_ws);
    const unsigned int scannerThreadCount = std::max(1u, threadCount);
    optCtx.threadCount = scannerThreadCount;
    optCtx.abort.shouldAbort = should_abort_effect;

    ScannerOptics::render_density_to_rgb(optCtx);
}


void JuicerProcessor::processImpl() {
    // SF_TEMP_BRIDGE_CPUProductRendererBlocked owner=Phase1A remove=Phase3/Phase4:
    // old CPU renderer is retained only as unreachable bridge debt; JuicerEffect::render()
    // blocks product output first with SpektrafilmPixelPipelineNotImplementedForPhase1A.
    JTRACE("SPEKTRAFILM", "FATAL: SpektrafilmPixelPipelineNotImplementedForPhase1A at JuicerProcessor::processImpl");
    throw OFX::Exception::Suite(kOfxStatErrFatal);

    if (!_srcImg || !_dstImg)
        return;
    const auto should_abort_effect = [this]() -> bool {
        return _effect.abort();
    };

    if (is_gpu_render_requested(_isEnabledOpenCLRender, _isEnabledCudaRender, _isEnabledMetalRender)) {
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
    const int xStart = _renderWindow.x1;
    const int xEnd = _renderWindow.x2;
    const int yStart = _renderWindow.y1;
    const int yEnd = _renderWindow.y2;

    if (_nComponents < 3) {
        if (_nComponents != 1) {
            return;
        }
        const int width = xEnd - xStart;
        if (width <= 0) {
            return;
        }
        const OfxRectI srcBounds = _srcImg->getBounds();
        const OfxRectI dstBounds = _dstImg->getBounds();
        for (int y = yStart; y < yEnd; ++y) {
            float* dstRow = write_float_row_if_fully_covered(
                _dstImg, dstBounds, xStart, xEnd, y);
            const float* srcRow = read_float_row_if_fully_covered(
                _srcImg, srcBounds, xStart, xEnd, y);
            CopyRowFloatScalarRequest copyRequest{};
            copyRequest.src = _srcImg;
            copyRequest.dst = _dstImg;
            copyRequest.span = CopyRowSpan{xStart, xEnd, y};
            copyRequest.srcRow = srcRow;
            copyRequest.dstRow = dstRow;
            copy_row_float_scalar_with_fallback(copyRequest);
        }
        return;
    }

#if defined(JUICER_SPD_DEBUG) && (JUICER_SPD_DEBUG != 0)
    Spectral::spd_probe_reset();
#endif

    RenderContext ctx = prepareRenderContext();
    if (ctx.width <= 0 || ctx.height <= 0) {
        return;
    }

    const unsigned int threadCount = compute_thread_count(ctx.width, ctx.height);
    ensureDensityCapacity(ctx.width, ctx.height);
    writeMediumDensities(ctx, threadCount);
    if (should_abort_effect()) {
        return;
    }
    renderScannerFromDensity(ctx, threadCount);
}

void JuicerProcessor::process() {
    // SF_TEMP_BRIDGE_CPUProductRendererBlocked owner=Phase1A remove=Phase3/Phase4:
    // defensive lower-level cutoff; product renders should already fail in JuicerEffect::render().
    JTRACE("SPEKTRAFILM", "FATAL: SpektrafilmPixelPipelineNotImplementedForPhase1A at JuicerProcessor::process");
    OFX::throwSuiteStatusException(kOfxStatErrFatal);

#if defined(JUICER_CUDA_ONLY) && (JUICER_CUDA_ONLY != 0)
    // CUDA-only mode: refuse CPU/OpenCL/Metal entry points.
    if (!_isEnabledCudaRender) {
        JTRACE("CUDA", "FATAL: JUICER_CUDA_ONLY rejected non-CUDA render request");
        OFX::throwSuiteStatusException(kOfxStatErrFatal);
    }
#endif
    if (is_gpu_render_requested(_isEnabledOpenCLRender, _isEnabledCudaRender, _isEnabledMetalRender)) {
        OFX::ImageProcessor::process();
        return;
    }
    processImpl();
}

void JuicerProcessor::multiThreadProcessImages(OfxRectI) {
    processImpl();
}

void JuicerProcessor::processImagesCUDA() {
    // SF_TEMP_BRIDGE_CPUProductRendererBlocked owner=Phase1A remove=Phase3/Phase4:
    // defensive lower-level CUDA cutoff; product renders should already fail in JuicerEffect::render().
    JTRACE("SPEKTRAFILM", "FATAL: SpektrafilmPixelPipelineNotImplementedForPhase1A at JuicerProcessor::processImagesCUDA");
    OFX::throwSuiteStatusException(kOfxStatErrFatal);

#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
    JTRACE("CUDA", "FATAL: CUDA render requested but the CUDA backend is unavailable in this build");
    OFX::throwSuiteStatusException(kOfxStatErrFatal);
#else
    enum class RenderMode {
        NegativeOnly,
        Print
    };

    auto render_mode_from_print_bypass = [](bool bypass) -> RenderMode {
        return bypass ? RenderMode::NegativeOnly : RenderMode::Print;
    };

    auto cuda_stream_or_null = [](void* rawStream) -> cudaStream_t {
        return rawStream ? reinterpret_cast<cudaStream_t>(rawStream) : nullptr;
    };

    auto gate_weave_debug_scale_or_one = [](double debugScalePx) -> float {
        return static_cast<float>((debugScalePx > 1e-6) ? debugScalePx : 1.0);
    };

    if (!_srcImg || !_dstImg) {
        return;
    }

    if (!(_nComponents == 1 || _nComponents == 3 || _nComponents == 4)) {
        std::string msg = "FATAL: CUDA render requested with unsupported component count=";
        msg += std::to_string(_nComponents);
        JTRACE("CUDA", msg);
        OFX::throwSuiteStatusException(kOfxStatErrFatal);
    }

    const OfxRectI srcBounds = _srcImg->getBounds();
    const OfxRectI dstBounds = _dstImg->getBounds();
    const OfxRectI win = _renderWindow;
    const int width = win.x2 - win.x1;
    const int height = win.y2 - win.y1;
    if (width <= 0 || height <= 0) {
        return;
    }
    const bool traceInfo = JTRACE_ENABLED(1);
    const bool traceVerbose = JTRACE_ENABLED(3);
    const auto should_abort_effect = [this]() -> bool {
        return _effect.abort();
    };

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
    void* contextOpaque = nullptr;
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
            const char* msg = detail_or_unknown(cudaGetErrorString(setErr));
            trace_cuda_fatal_prefixed_if(
                traceInfo,
                "cudaSetDevice failed",
                msg);
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }

        std::string contextError;
        if (!query_current_cuda_context(contextOpaque, contextError)) {
            trace_cuda_fatal_prefixed_if(
                traceInfo,
                "failed to capture CUDA context identity",
                cstr_or_null_if_empty(contextError));
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
    }

    if (should_abort_effect()) {
        return;
    }

    const unsigned char* srcPtr = srcBase + ySrc * srcRowBytes + xSrc * bytesPerPixel;
    unsigned char* dstPtr = dstBase + yDst * dstRowBytes + xDst * bytesPerPixel;

    JTRACE_VERBOSE("CUDA", "processImagesCUDA");

    const bool wsReady = _wsReady && _ws;
    if (!wsReady) {
        JTRACE("CUDA", "FATAL: working state unavailable; cannot serve CUDA render");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
    if (!_instanceState) {
        JTRACE("CUDA", "FATAL: instance state missing; cannot serve CUDA render");
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    }
    const bool printActiveForRender = print_pipeline_active(
        wsReady,
        _ws,
        _printReady,
        _prt,
        _printParams.bypass);
    const float printMidgrayFactor = printActiveForRender
                                         ? compute_print_midgray_factor(*_ws, *_prt, _printParams, _dirRT)
                                         : 1.0f;

    JuicerCuda::ResourceManager::DeviceContextKey deviceContextKey{};
    deviceContextKey.deviceId = deviceId;
    deviceContextKey.contextOpaque = contextOpaque;

    struct PendingContextLossRecovery {
        bool pending = false;
        cudaError_t error = cudaSuccess;
        const char* stage = nullptr;
        std::string detail;
    } pendingContextLossRecovery{};

    auto mark_context_loss_recovery = [&](const char* stage, cudaError_t error, const std::string& detail) {
        if (pendingContextLossRecovery.pending) {
            return;
        }
        if (!is_cuda_context_loss_signal(error, detail)) {
            return;
        }
        pendingContextLossRecovery.pending = true;
        pendingContextLossRecovery.error = error;
        pendingContextLossRecovery.stage = stage;
        pendingContextLossRecovery.detail = detail;
    };

    auto run_pending_context_loss_recovery = [&]() {
        if (!pendingContextLossRecovery.pending) {
            return;
        }
        recover_context_loss_state(
            _instanceState,
            deviceContextKey,
            pendingContextLossRecovery.stage,
            pendingContextLossRecovery.error,
            pendingContextLossRecovery.detail);
        pendingContextLossRecovery = PendingContextLossRecovery{};
    };

    auto run_pending_context_loss_recovery_noexcept = [&]() noexcept {
        try {
            run_pending_context_loss_recovery();
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    };

    struct ContextLossRecoveryScope {
        decltype(run_pending_context_loss_recovery_noexcept)* onExit = nullptr;
        ~ContextLossRecoveryScope() noexcept {
            if (onExit) {
                (*onExit)();
            }
        }
    } contextLossRecoveryScope{&run_pending_context_loss_recovery_noexcept};

    auto record_cuda_use = [&](JuicerProcess::Root::PreparedCudaFrame& frame) {
        frame.record_use(_pCudaStream);
    };

    auto abort_cuda_path_if_requested = [&](JuicerProcess::Root::PreparedCudaFrame& frame) -> bool {
        if (!should_abort_effect()) {
            return false;
        }
        record_cuda_use(frame);
        return true;
    };

    auto throw_cuda_policy_fatal = [&](const char* failurePrefix = nullptr,
                                       const char* detail = nullptr) {
        trace_cuda_fatal_prefixed_if(
            traceInfo,
            nonempty_cstr_or(failurePrefix, "CUDA render cannot continue"),
            detail);
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    };

    auto throw_cuda_stage_fatal = [&](const char* stageTag,
                                      const char* failurePrefix,
                                      cudaError_t errorCode) {
        const char* errorMsg = cudaGetErrorString(errorCode);
        const char* detail = detail_or_unknown(errorMsg);
        mark_context_loss_recovery(nonempty_cstr_or(stageTag, "cuda_stage"), errorCode, detail);
        throw_cuda_policy_fatal(nonempty_cstr_or(failurePrefix, "CUDA stage failed"), detail);
    };

    auto trace_and_throw_cuda_policy_fatal = [&](const char* prefix, const char* detail) {
        throw_cuda_policy_fatal(prefix, detail);
    };

    auto mark_context_and_throw_cuda_policy_fatal = [&](const char* stageTag,
                                                        const char* prefix,
                                                        const std::string& detail) {
        mark_context_loss_recovery(
            nonempty_cstr_or(stageTag, "cuda_stage"),
            cudaErrorUnknown,
            detail);
        trace_and_throw_cuda_policy_fatal(prefix, cstr_or_null_if_empty(detail));
    };

    auto throw_submission_fatal = [&](const char* stageTag,
                                      const char* failurePrefix,
                                      const std::string& error) {
        mark_context_loss_recovery(
            nonempty_cstr_or(stageTag, "submission_stage"),
            cudaErrorUnknown,
            error);
        trace_cuda_fatal_prefixed_if(
            traceInfo,
            nonempty_cstr_or(failurePrefix, "submission stage failed"),
            cstr_or_null_if_empty(error));
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    };

    auto copy_scan_tables_payload = [&](auto& dstTables,
                                        auto& dstMediumIsNegative,
                                        float* dstMinCmy,
                                        float* dstInvMaxCmy,
                                        const JuicerCuda::Resources::DeviceScanMedium& scanMedium) {
        dstTables.epsC = scanMedium.tables.epsC;
        dstTables.epsM = scanMedium.tables.epsM;
        dstTables.epsY = scanMedium.tables.epsY;
        dstTables.Ax = scanMedium.tables.Ax;
        dstTables.Ay = scanMedium.tables.Ay;
        dstTables.Az = scanMedium.tables.Az;
        dstTables.baseMin = scanMedium.tables.baseMin;
        dstTables.K = scanMedium.tables.K;
        dstTables.hasBaseline = scanMedium.tables.hasBaseline;
        dstTables.invYn = scanMedium.tables.invYn;
        dstMediumIsNegative = scanMedium.mediumIsNegative;
        copy_float3(dstMinCmy, scanMedium.min_cmy);
        copy_float3(dstInvMaxCmy, scanMedium.inv_max_cmy);
    };

    auto make_cuda_prefixed_failure = [&](const char* label, const char* suffix) {
        std::string msg = "CUDA ";
        msg += nonempty_cstr_or(label, "operation");
        msg += nonempty_cstr_or(suffix, " failed");
        return msg;
    };

    if (should_abort_effect()) {
        return;
    }

    const RenderRecipe* directRecipe =
        (_recipeHold && !Spektrafilm::scan_route_is_print(_recipeHold->profileRoute.scanRoute))
            ? _recipeHold.get()
            : nullptr;
    const FilmRawRecipe* directFilmRaw = directRecipe ? &directRecipe->filmRaw : nullptr;
    const bool cameraAutoEnabled =
        directFilmRaw ? directFilmRaw->autoExposureEnabled : _cameraAutoEnabled;
    const Spektrafilm::AutoExposureMethod cameraMeteringMethod =
        directFilmRaw
            ? directFilmRaw->autoExposureMethod
            : static_cast<Spektrafilm::AutoExposureMethod>(
                  std::clamp(_cameraMeteringMethod, 0, 6));
    OfxRectI meterBounds = srcBounds;
    if (!directFilmRaw && cameraAutoEnabled && _autoExposureMeterBoundsValid) {
        meterBounds = _autoExposureMeterBounds;
    }
    auto clamp_rect = [](OfxRectI r, const OfxRectI& bounds) {
        r.x1 = std::clamp(r.x1, bounds.x1, bounds.x2);
        r.x2 = std::clamp(r.x2, bounds.x1, bounds.x2);
        r.y1 = std::clamp(r.y1, bounds.y1, bounds.y2);
        r.y2 = std::clamp(r.y2, bounds.y1, bounds.y2);
        if (r.x2 < r.x1) {
            const int tmp = r.x1;
            r.x1 = r.x2;
            r.x2 = tmp;
        }
        if (r.y2 < r.y1) {
            const int tmp = r.y1;
            r.y1 = r.y2;
            r.y2 = tmp;
        }
        return r;
    };
    meterBounds = clamp_rect(meterBounds, srcBounds);
    if ((meterBounds.x2 - meterBounds.x1) <= 0 || (meterBounds.y2 - meterBounds.y1) <= 0) {
        meterBounds = srcBounds;
    }
    const JuicerCuda::AutoExposurePreviewDescriptor autoExposureDescriptor =
        make_auto_exposure_preview_descriptor(srcBounds, meterBounds, cameraMeteringMethod);
    JuicerProcess::Root::AutoExposureBufferRequest autoExposureBufferRequest{};
    autoExposureBufferRequest.enabled =
        cameraAutoEnabled &&
        (_nComponents == 3 || _nComponents == 4) &&
        autoExposureDescriptor.previewWidth > 0 &&
        autoExposureDescriptor.previewHeight > 0;
    autoExposureBufferRequest.descriptor = autoExposureDescriptor;
    autoExposureBufferRequest.reusableKeyHash = autoExposureDescriptor.hash;

    JuicerCuda::ResourceManager::SubmissionSnapshot snapshot{};
    {
        snapshot.instanceToken.value = instance_token_or_session_seed(_instanceToken, _sessionSeed);
        snapshot.frameToken.value = static_cast<std::uint64_t>(_frameIndex);
        snapshot.deviceContextKey = deviceContextKey;
        const std::uint64_t uploadCoreHash = upload_core_hash_or_core_hash(*_ws);
        const std::uint64_t scannerRuntimeHash = JuicerProcScanner::hash_scanner_runtime_lane(
            _ws,
            _scannerSettings,
            _scannerOptions,
            _frameBoundsVersion);
        snapshot.keyDigests =
            JuicerCuda::ResourceManager::make_key_digests(
                uploadCoreHash,
                _ws->dirHash,
                scannerRuntimeHash,
                autoExposureDescriptor.hash);
        snapshot.keySchemaVersion = JuicerCuda::ResourceManager::kSubmissionKeySchemaVersion;
        snapshot.traceSchemaVersion = JuicerCuda::ResourceManager::kTraceSchemaVersion;
#if JUICER_DIAGNOSTICS_COMPILED
        bool reusingSnapshotLatch = false;
#endif
        {
            std::lock_guard<std::mutex> latchLock(_instanceState->submissionSnapshotLatchMutex);
            const auto& latched = _instanceState->submissionSnapshotLatch;
            const bool digestsMatch =
                latched.keyDigests.uploadCoreHash == snapshot.keyDigests.uploadCoreHash &&
                latched.keyDigests.dirHash == snapshot.keyDigests.dirHash &&
                latched.keyDigests.scannerHash == snapshot.keyDigests.scannerHash &&
                latched.keyDigests.autoExposureHash == snapshot.keyDigests.autoExposureHash;
            if (_instanceState->submissionSnapshotLatchValid &&
                latched.instanceToken.value == snapshot.instanceToken.value &&
                latched.frameToken.value == snapshot.frameToken.value &&
                latched.deviceContextKey == snapshot.deviceContextKey &&
                latched.keySchemaVersion == snapshot.keySchemaVersion &&
                latched.traceSchemaVersion == snapshot.traceSchemaVersion &&
                digestsMatch &&
                latched.snapshotId != 0) {
                snapshot = latched;
#if JUICER_DIAGNOSTICS_COMPILED
                reusingSnapshotLatch = true;
#endif
            } else {
                std::uint64_t nextSnapshotId =
                    _instanceState->submissionSnapshotIdNext.fetch_add(1, std::memory_order_relaxed);
                if (nextSnapshotId == 0) {
                    nextSnapshotId = _instanceState->submissionSnapshotIdNext.fetch_add(1, std::memory_order_relaxed);
                }
                snapshot.snapshotId = nextSnapshotId;
                _instanceState->submissionSnapshotLatch = snapshot;
                _instanceState->submissionSnapshotLatchValid = true;
            }
        }
#if JUICER_DIAGNOSTICS_COMPILED
        if (traceVerbose) {
            std::string msg;
            msg.reserve(128);
            msg = "path=cuda action=";
            msg += submission_snapshot_action_label(reusingSnapshotLatch);
            msg += " frame_token=";
            msg += std::to_string(snapshot.frameToken.value);
            msg += " snapshot_id=";
            msg += std::to_string(snapshot.snapshotId);
            msg += " instance_token=";
            msg += std::to_string(snapshot.instanceToken.value);
            JTRACE_VERBOSE("MSSNP", msg);
        }
#endif
    }

    std::string prepareFrameError;
    JuicerProcess::Root::PreparedCudaFrame preparedFrame =
        JuicerProcess::root().prepare_cuda_frame(
            deviceContextKey,
            snapshot,
            *_ws,
            autoExposureBufferRequest,
            _pCudaStream,
            prepareFrameError);
    if (!preparedFrame.active()) {
        throw_submission_fatal(
            preparedFrame.failure_stage_tag(),
            preparedFrame.failure_prefix(),
            prepareFrameError);
    }

    if (traceVerbose) {
        const JuicerProcess::Root::PreparedCudaFrame::UploadTraceView uploadTrace =
            preparedFrame.upload_trace_view();
        std::string msg;
        msg.reserve(192);
        msg = "cuda upload build=";
        msg += std::to_string(_ws->buildCounter);
        msg += " uploaded=";
        msg += std::to_string(uploadTrace.uploadedBuildCounter);
        msg += " printIllumBuild=";
        msg += std::to_string(uploadTrace.printIllumBuildCounter);
        msg += " printPreflashKey=";
        msg += std::to_string(uploadTrace.printPreflashKeyHash);
        JTRACE_VERBOSE("PRINTDBG", msg);
    }

    if (abort_cuda_path_if_requested(preparedFrame)) {
        return;
    }

#if defined(JUICER_CUDA_VALIDATE_PRIMITIVES) && (JUICER_CUDA_VALIDATE_PRIMITIVES != 0)
    {
        const DiagnosticsHookPolicy& diagnosticsPolicy = diagnostics_hook_policy();
        const bool validationHookActive =
            diagnosticsPolicy.diagnosticsMode &&
            diagnosticsPolicy.validatePrimitives &&
            traceVerbose;
        trace_validation_hook_state_once(
            true,
            diagnosticsPolicy.diagnosticsMode,
            diagnosticsPolicy.validatePrimitives,
            traceVerbose,
            validationHookActive);
        if (validationHookActive) {
            std::string validateError;
            if (!preparedFrame.validate_density_primitives(*_ws, _pCudaStream, validateError)) {
                std::string msg;
                msg.reserve(48 + validateError.size());
                msg = "FATAL: CUDA primitive validation failed: ";
                msg += validateError;
                JTRACE("CUDA", msg);
                throw OFX::Exception::Suite(kOfxStatErrFatal);
            }

            if (printActiveForRender) {
                if (!preparedFrame.validate_print_primitives(
                        *_ws,
                        *_prt,
                        _printParams,
                        printMidgrayFactor,
                        _pCudaStream,
                        validateError)) {
                    std::string msg;
                    msg.reserve(44 + validateError.size());
                    msg = "FATAL: CUDA print validation failed: ";
                    msg += validateError;
                    JTRACE("CUDA", msg);
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
                }
            }

            record_cuda_use(preparedFrame);
        }
    }
#endif

#if defined(JUICER_CUDA_SELF_CHECK) && (JUICER_CUDA_SELF_CHECK != 0)
    {
        const DiagnosticsHookPolicy& diagnosticsPolicy = diagnostics_hook_policy();
        const bool selfCheckHookActive =
            diagnosticsPolicy.diagnosticsMode &&
            diagnosticsPolicy.runtimeSelfCheck;
        trace_self_check_hook_state_once(
            true,
            diagnosticsPolicy.diagnosticsMode,
            diagnosticsPolicy.runtimeSelfCheck,
            selfCheckHookActive);
        if (selfCheckHookActive) {
            // Runtime CUDA self-check.
            // This is intentionally a host-runtime probe, not part of the deprecated test-hook path,
            // and is designed to be easy to remove later: disable JUICER_CUDA_SELF_CHECK or delete
            // Cuda/JuicerCudaSelfCheck.*.
            static std::once_flag sSelfCheckOnce;
            static bool sSelfCheckOk = true;
            static const char* sSelfCheckErr = nullptr;
            std::call_once(sSelfCheckOnce, [&]() {
                const bool ok = juicer_cuda_runtime_self_check(_pCudaStream, &sSelfCheckErr);
                sSelfCheckOk = ok;
                if (!ok) {
                    if (traceInfo) {
                        std::string msg;
                        msg.reserve(96);
                        msg = "CUDA self-check failed; aborting CUDA render. Error: ";
                        msg += detail_or_unknown(sSelfCheckErr);
                        JTRACE("CUDA", msg);
                    }
                } else {
                    JTRACE("CUDA", "CUDA self-check passed");
                }
            });
            if (!sSelfCheckOk) {
                OFX::throwSuiteStatusException(kOfxStatErrFatal);
            }
        }
    }
#endif

    if (_nComponents == 1) {
        if (srcPtr == dstPtr && srcRowBytes == dstRowBytes) {
            return;
        }
        const cudaStream_t stream = cuda_stream_or_null(_pCudaStream);
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
            throw_cuda_stage_fatal("copy_alpha_memcpy2d", "cudaMemcpy2DAsync failed", err);
        }
        record_cuda_use(preparedFrame);
        return;
    }

    const RenderMode renderMode = render_mode_from_print_bypass(_printParams.bypass);

    auto setup_camera_auto_exposure = [&](JuicerCuda::PipelineRunParams& run) {
        if (!cameraAutoEnabled) {
            return;
        }
        if (should_abort_effect()) {
            return;
        }
        if (!(run.nComponents == 3 || run.nComponents == 4)) {
            return;
        }

        if (autoExposureDescriptor.previewWidth <= 0 || autoExposureDescriptor.previewHeight <= 0) {
            return;
        }

        auto throw_auto_exposure_mode_fatal = [&](const char* prefix, const char* detail) {
            trace_and_throw_cuda_policy_fatal(
                nonempty_cstr_or(prefix, "CUDA auto-exposure failed"),
                detail_or_unknown(detail));
        };

        const JuicerProcess::Root::PreparedCudaFrame::AutoExposureBufferView autoExposureBuffers =
            preparedFrame.auto_exposure_buffers();
        if (!autoExposureBuffers.active) {
            throw_auto_exposure_mode_fatal(
                "CUDA auto-exposure buffers missing",
                "auto-exposure buffers missing after allocation");
        }

        const JuicerCudaAutoExposureScratch scratch = autoExposureBuffers.scratch;
        const JuicerCudaAutoExposureDeviceState state = autoExposureBuffers.deviceState;

        std::uint64_t meterStateKey = Hash::kFnvOffset;
        const double timeFrames = finite_or(_timeFrames, 0.0);
        Hash::hash_bytes_update(meterStateKey, &timeFrames, sizeof(timeFrames));
        Hash::hash_bytes_update(meterStateKey, &_clipToken, sizeof(_clipToken));
        Hash::hash_bytes_update(
            meterStateKey,
            &autoExposureDescriptor.hash,
            sizeof(autoExposureDescriptor.hash));
        Hash::hash_bytes_update(meterStateKey, &srcRowBytes, sizeof(srcRowBytes));
        Hash::hash_bytes_update(meterStateKey, &run.nComponents, sizeof(run.nComponents));
        Hash::hash_bytes_update(meterStateKey, &run.filmRaw.inputColorSpaceIndex, sizeof(run.filmRaw.inputColorSpaceIndex));
        Hash::hash_bytes_update(meterStateKey, &run.filmRaw.applyCctfDecoding, sizeof(run.filmRaw.applyCctfDecoding));
        Hash::hash_bytes_update(meterStateKey, &run.filmRaw.inputRGBToXYZ, sizeof(run.filmRaw.inputRGBToXYZ));
        Hash::hash_bytes_update(meterStateKey, &cameraMeteringMethod, sizeof(cameraMeteringMethod));
        if (meterStateKey == 0) {
            meterStateKey = 1;
        }

        const bool needMeter = (autoExposureBuffers.keyHash != meterStateKey);
        const char* errMsg = nullptr;
        if (needMeter) {
            if (cameraMeteringMethod == Spektrafilm::AutoExposureMethod::CenterWeighted) {
                if (!scratch.weightsX || !scratch.weightsY ||
                    autoExposureBuffers.weightsWidth != autoExposureDescriptor.previewWidth ||
                    autoExposureBuffers.weightsHeight != autoExposureDescriptor.previewHeight) {
                    const int rcW = juicer_cuda_auto_exposure_build_center_weight_tables(
                        autoExposureDescriptor.previewWidth,
                        autoExposureDescriptor.previewHeight,
                        scratch.weightsX,
                        scratch.weightsY,
                        _pCudaStream,
                        &errMsg);
                    if (rcW != 0) {
                        throw_auto_exposure_mode_fatal(
                            "CUDA auto-exposure weight build failed",
                            errMsg);
                    }
                    preparedFrame.mark_auto_exposure_weights_built(
                        JuicerProcess::Root::PreparedCudaFrame::AutoExposureWeightsExtent{
                            autoExposureDescriptor.previewWidth,
                            autoExposureDescriptor.previewHeight});
                }
            }

            const int rc = juicer_cuda_auto_exposure_meter_to_device(
                srcBase,
                static_cast<std::size_t>(srcRowBytes),
                autoExposureDescriptor,
                run.nComponents,
                run.filmRaw.inputColorSpaceIndex,
                run.filmRaw.applyCctfDecoding,
                run.filmRaw.inputRGBToXYZ,
                scratch,
                state,
                _pCudaStream,
                &errMsg);
            if (rc != 0) {
                throw_auto_exposure_mode_fatal(
                    "CUDA auto-exposure metering failed",
                    errMsg);
            }
            preparedFrame.mark_auto_exposure_metered(
                JuicerProcess::Root::PreparedCudaFrame::AutoExposureMeteredResult{
                    meterStateKey});
        }

        run.filmExpose.exposureScaleDevice = autoExposureBuffers.deviceState.exposureScale;
    };

    struct GrainSetupResult {
        bool wantGrain = false;
        bool wantGrainSublayers = false;
        bool wantGrainBlur = false;
        bool wantGrainMix = false;
        float grainBlurSigmaPx = 0.0f;
        float grainBlurSigmaMidPx = 0.0f;
        float grainBlurSigmaCoarsePx = 0.0f;
        float grainDyeSigmaPx[3][3] = {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
    };

    struct HalationSetupResult {
        float strengthBGR[3] = {0.0f, 0.0f, 0.0f};
        float scatterStrengthBGR[3] = {0.0f, 0.0f, 0.0f};
        float sigmaPx[3] = {0.0f, 0.0f, 0.0f};
        float scatterSigmaPx[3] = {0.0f, 0.0f, 0.0f};
        bool wantHalation = false;
    };

    auto setup_halation_payload = [&](const Profiles::HalationMetadata& halationUi) -> HalationSetupResult {
        HalationSetupResult result{};

        const float strengthBGR[3] = {
            halationUi.strength[2],
            halationUi.strength[1],
            halationUi.strength[0]};
        const float scatterStrengthBGR[3] = {
            halationUi.scatteringStrength[2],
            halationUi.scatteringStrength[1],
            halationUi.scatteringStrength[0]};
        const float sizeBGR[3] = {
            halationUi.sizeUm[2],
            halationUi.sizeUm[1],
            halationUi.sizeUm[0]};
        const float scatterSizeBGR[3] = {
            halationUi.scatteringSizeUm[2],
            halationUi.scatteringSizeUm[1],
            halationUi.scatteringSizeUm[0]};

        copy_float3(result.strengthBGR, strengthBGR);
        copy_float3(result.scatterStrengthBGR, scatterStrengthBGR);
        copy_float3(result.sigmaPx, sizeBGR);
        copy_float3(result.scatterSigmaPx, scatterSizeBGR);
        sanitize_nonnegative_triplet(result.strengthBGR);
        sanitize_nonnegative_triplet(result.scatterStrengthBGR);
        sanitize_nonnegative_triplet(result.sigmaPx);
        sanitize_nonnegative_triplet(result.scatterSigmaPx);
        const bool hasPixelSize = is_positive_finite(_pixelSizeUm);

        if (hasPixelSize) {
            divide_triplet(result.sigmaPx, result.sigmaPx, _pixelSizeUm);
            divide_triplet(result.scatterSigmaPx, result.scatterSigmaPx, _pixelSizeUm);
        }

        result.wantHalation = halationUi.active &&
                              (any_positive_triplet(result.strengthBGR) || any_positive_triplet(result.scatterStrengthBGR)) &&
                              hasPixelSize;
        return result;
    };

    auto setup_grain_payload = [&](JuicerCuda::PipelineRunParams& run,
                                   const Profiles::GrainMetadata& grainUi,
                                   bool includeDefects) -> GrainSetupResult {
        GrainSetupResult result{};

        auto nanmax_vector = [](const std::vector<float>& values, float& outMax) -> bool {
            double m = -std::numeric_limits<double>::infinity();
            bool found = false;
            const float* data = values.data();
            const float* const dataEnd = data + values.size();
            for (; data < dataEnd; ++data) {
                const float v = *data;
                if (is_finite(v)) {
                    m = std::max(m, static_cast<double>(v));
                    found = true;
                }
            }
            if (!found || !is_finite(m)) {
                return false;
            }
            outMax = static_cast<float>(m);
            return is_finite(outMax);
        };
        auto nanmax_curve = [&](const Spectral::Curve& curve, float& outMax) -> bool {
            return nanmax_vector(curve.linear, outMax);
        };

        bool wantGrain = grainUi.active && is_positive_finite(_pixelSizeUm);
        bool wantGrainSublayers = false;
        bool wantGrainBlur = false;
        bool wantGrainMix = false;
        float grainBlurSigmaPx = 0.0f;
        float grainBlurSigmaMidPx = 0.0f;
        float grainBlurSigmaCoarsePx = 0.0f;
        float grainDyeSigmaPx[3][3] = {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};

        {
            const JuicerProcess::Root::PreparedCudaFrame::GrainStaticAssets grainStaticAssets =
                preparedFrame.grain_static_assets();
            const std::uint64_t sessionSeed = session_seed_or_default(_sessionSeed);
            const double fps = positive_finite_or(_frameRate, 24.0);
            const double timeFrames = finite_or(_timeFrames, static_cast<double>(_frameIndex));
            const double alphaFrames = timeFrames - static_cast<double>(_frameIndex);
            const float timeAlpha = static_cast<float>(std::clamp(alphaFrames, 0.0, 1.0));
            const double timeSeconds = timeFrames / fps;
            const double weaveAmount = sanitize_finite_clamped_or(_gateWeaveAmount, 0.0, 0.0, 10.0);
            const bool hasPixelSize = is_positive_finite(_pixelSizeUm);
            const GateWeaveSignal weave = compute_gate_weave(
                sessionSeed,
                timeSeconds,
                6.0,
                0.005,
                static_cast<double>(_pixelSizeUm),
                weaveAmount);
            const double debugScalePx = hasPixelSize
                                            ? (4.0 * 6.0 * weaveAmount / static_cast<double>(_pixelSizeUm))
                                            : 1.0;
            const int breathingPeriodFrames = std::max(1, static_cast<int>(std::llround(fps * 2.5)));
            const double clumpPeriodSec = sanitize_finite_clamped_or(
                static_cast<double>(grainUi.clumpMorphPeriodSec),
                25.0,
                5.0,
                60.0);
            const int clumpMorphPeriodFrames = std::max(1, static_cast<int>(std::llround(fps * clumpPeriodSec)));
            const double longEdgePx = static_cast<double>(std::max(width, height));
            const double filmFormatMm = (hasPixelSize && longEdgePx > 0.0)
                                            ? (static_cast<double>(_pixelSizeUm) * longEdgePx / 1000.0)
                                            : 0.0;
            const double pitchMm = is_positive_finite(filmFormatMm)
                                       ? (filmFormatMm * static_cast<double>(height) / longEdgePx)
                                       : 0.0;
            const int pitchPx = (hasPixelSize && is_positive_finite(pitchMm))
                                    ? static_cast<int>(std::llround(pitchMm * 1000.0 / static_cast<double>(_pixelSizeUm)))
                                    : height;
            const double filmScale = positive_finite_or(filmFormatMm, 10.0) / 10.0;
            run.grain.seedBase = seed_base_for_pass(
                _sessionSeed,
                _clipToken,
                _frameIndex,
                kSeedPassGrain);
            run.grain.seedBaseNext = seed_base_for_pass(
                _sessionSeed,
                _clipToken,
                _frameIndex + 1,
                kSeedPassGrain);
            run.grain.frameIndex = _frameIndex;
            run.grain.stbnSessionSeed = sessionSeed;
            run.grain.clipToken = static_cast<std::uint64_t>(_clipToken);
            run.grain.timeAlpha = timeAlpha;
            run.gateWeave.active = bool_to_i32(weaveAmount > 0.0 && hasPixelSize);
            run.gateWeave.dxPx = weave.dxPx;
            run.gateWeave.dyPx = weave.dyPx;
            run.gateWeave.cosRot = weave.cosRot;
            run.gateWeave.sinRot = weave.sinRot;
            run.gateWeave.debugScalePx = gate_weave_debug_scale_or_one(debugScalePx);
            run.grain.pitchPx = pitchPx;
            run.grain.breathingPeriodFrames = breathingPeriodFrames;
            run.grain.breathingAmplitude = 0.01902219f;
            run.grain.breathingCellUmSmall = static_cast<float>(2500.0 * filmScale);
            run.grain.breathingCellUmLarge = static_cast<float>(5000.0 * filmScale);
            run.grain.breathingMix = 0.30f;
            run.grain.breathingDriftUmPerFrame = 1.0f;
            run.grain.sizeMixWeight = 0.30f;
            run.grain.sizeMixScale = 3.0f;
            run.grain.clumpTemporalMix = static_cast<float>(
                sanitize_finite_clamped_or(static_cast<double>(grainUi.clumpTemporalMix), 0.0, 0.0, 0.30));
            run.grain.clumpMorphPeriodFrames = clumpMorphPeriodFrames;
            run.grain.wangCellMm = 2.0f;
            if (grainStaticAssets.stbn &&
                grainStaticAssets.stbnWidth > 0 && grainStaticAssets.stbnHeight > 0 && grainStaticAssets.stbnFrames > 0) {
                run.grain.stbn = grainStaticAssets.stbn;
                run.grain.stbnWidth = grainStaticAssets.stbnWidth;
                run.grain.stbnHeight = grainStaticAssets.stbnHeight;
                run.grain.stbnFrames = grainStaticAssets.stbnFrames;
                run.grain.stbnOffsetX = stbn_offset(sessionSeed, run.grain.stbnWidth, 0xA5u);
                run.grain.stbnOffsetY = stbn_offset(sessionSeed, run.grain.stbnHeight, 0x5Au);
                run.grain.stbnFrame = stbn_frame_index(_frameIndex, run.grain.stbnFrames, sessionSeed);
            }
            if (grainStaticAssets.wangTiles && grainStaticAssets.wangLut &&
                grainStaticAssets.wangWidth > 0 && grainStaticAssets.wangHeight > 0 &&
                grainStaticAssets.wangCount > 0 && grainStaticAssets.wangColors > 0) {
                run.grain.wangTiles = grainStaticAssets.wangTiles;
                run.grain.wangLut = grainStaticAssets.wangLut;
                run.grain.wangWidth = grainStaticAssets.wangWidth;
                run.grain.wangHeight = grainStaticAssets.wangHeight;
                run.grain.wangCount = grainStaticAssets.wangCount;
                run.grain.wangColors = grainStaticAssets.wangColors;
            }
        }
        if (wantGrain) {
            float densityMin[3];
            float uniformity[3];
            copy_float3(densityMin, grainUi.densityMin.data());
            copy_float3(uniformity, grainUi.uniformity.data());
            sanitize_nonnegative_triplet(densityMin);
            sanitize_unit_triplet(uniformity);

            float maxC = 0.0f;
            float maxM = 0.0f;
            float maxY = 0.0f;
            const bool maxOk =
                nanmax_curve(_ws->densR, maxC) &&
                nanmax_curve(_ws->densG, maxM) &&
                nanmax_curve(_ws->densB, maxY);
            if (!maxOk) {
                wantGrain = false;
            }

            const float pixelAreaUm2 = _pixelSizeUm * _pixelSizeUm;
            if (!is_positive_finite(pixelAreaUm2)) {
                wantGrain = false;
            }

            const int nSubLayers = positive_i32_or(grainUi.nSubLayers, 1);
            run.grain.nSubLayers = nSubLayers;
            run.grain.originX = win.x1;
            run.grain.originY = win.y1;
            run.grain.pixelSizeUm = _pixelSizeUm;
            run.grain.blurSigmaPx = sanitize_nonnegative_or(grainUi.blur, 0.0f);
            run.grain.blurDyeCloudsUm = sanitize_nonnegative_or(grainUi.blurDyeCloudsUm, 0.0f);
            run.grain.sizeMixWeight = sanitize_unit_or(grainUi.sizeMixWeight, 0.0f);
            run.grain.sizeMixWeightMid = sanitize_unit_or(grainUi.sizeMixWeightMid, 0.0f);
            run.grain.sizeMixScale = std::max(1.0f, sanitize_nonnegative_or(grainUi.sizeMixScale, 1.0f));
            run.grain.breathingDebug = bool_to_i32(grainUi.breathingDebug);
            run.grain.debugView = std::clamp(grainUi.debugView, 0, 6);
            run.grain.amplitude = sanitize_nonnegative_or(grainUi.amplitude, 1.0f);
            const float chromaMix = sanitize_unit_or(grainUi.chroma, 1.0f);
            run.grain.chromaMix = chromaMix;
            run.grain.chromaSharedWeight = std::sqrt(std::max(0.0f, 1.0f - chromaMix));
            run.grain.chromaIndWeight = std::sqrt(std::max(0.0f, chromaMix));
            copy_float2(run.grain.microStructure, grainUi.microStructure.data());
            if (includeDefects) {
                run.grain.filmDustAmount = sanitize_amount_0_10(grainUi.filmDustAmount);
                run.grain.gateDustAmount = sanitize_amount_0_10(grainUi.gateDustAmount);
                run.grain.filmScratchAmount = sanitize_amount_0_10(grainUi.filmScratchAmount);
                run.grain.gateScratchAmount = sanitize_amount_0_10(grainUi.gateScratchAmount);
            }
            copy_float3(run.grain.densityMin, densityMin);
            copy_float3(run.grain.uniformity, uniformity);

            bool paramsOk = wantGrain;
            float blurAreaRatio = 1.0f;
            float blurRatioSum = 0.0f;
            int blurRatioCount = 0;
            if (paramsOk) {
                constexpr float kDefaultParticleAreaUm2 = 0.335f;
                constexpr float kDefaultParticleScale[3] = {1.10f, 1.27f, 2.08f};
                const float densityMaxCurves[3] = {maxC, maxM, maxY};
                const float* densityMaxCurveIt = densityMaxCurves;
                const float* densityMinIt = densityMin;
                const float* grainScaleIt = grainUi.agxParticleScale.data();
                const float* defaultScaleIt = kDefaultParticleScale;
                float* densityMaxOutIt = run.grain.densityMax;
                float* nParticlesOutIt = run.grain.nParticles;
                float* odParticleOutIt = run.grain.odParticle;
                for (int i = 0; i < 3; ++i,
                         ++densityMaxCurveIt,
                         ++densityMinIt,
                         ++grainScaleIt,
                         ++defaultScaleIt,
                         ++densityMaxOutIt,
                         ++nParticlesOutIt,
                         ++odParticleOutIt) {
                    const float densityMax = *densityMaxCurveIt + *densityMinIt;
                    const float particleArea = grainUi.agxParticleAreaUm2 * (*grainScaleIt);
                    if (!is_positive_finite(particleArea)) {
                        paramsOk = false;
                        break;
                    }
                    const float particleAreaRef = kDefaultParticleAreaUm2 * (*defaultScaleIt);
                    if (is_positive_finite(particleAreaRef)) {
                        blurRatioSum += particleArea / particleAreaRef;
                        blurRatioCount += 1;
                    }
                    float nParticles = pixelAreaUm2 / particleArea;
                    if (nSubLayers > 1) {
                        nParticles /= static_cast<float>(nSubLayers);
                    }
                    if (!is_positive_finite(nParticles)) {
                        paramsOk = false;
                        break;
                    }
                    const float odParticle = densityMax / nParticles;
                    *densityMaxOutIt = densityMax;
                    *nParticlesOutIt = nParticles;
                    *odParticleOutIt = finite_or_zero(odParticle);
                }
            }
            if (!paramsOk) {
                wantGrain = false;
            }

            if (wantGrain) {
                if (blurRatioCount > 0) {
                    blurAreaRatio = blurRatioSum / static_cast<float>(blurRatioCount);
                }
                grainBlurSigmaPx = run.grain.blurSigmaPx;
                if (is_positive_finite(grainBlurSigmaPx)) {
                    grainBlurSigmaPx *= std::sqrt(std::max(blurAreaRatio, 0.0f));
                }

                if (!is_positive_finite(run.grain.microStructure[1])) {
                    zero_float2(run.grain.microStructure);
                }

                const JuicerProcess::Root::PreparedCudaFrame::DurableBundleView durableBundle =
                    preparedFrame.durable_bundle();
                if (grainUi.sublayersActive && _ws->hasDensityCurvesLayers && durableBundle.film.hasDensityCurvesLayers) {
                    float densityMaxLayers[3][3] = {{0.0f, 0.0f, 0.0f},
                                                    {0.0f, 0.0f, 0.0f},
                                                    {0.0f, 0.0f, 0.0f}};
                    bool layersOk = true;
                    for (int layer = 0; layer < 3; ++layer) {
                        for (int ch = 0; ch < 3; ++ch) {
                            if (!nanmax_vector(_ws->densityCurvesLayers[layer][ch], densityMaxLayers[layer][ch])) {
                                layersOk = false;
                            }
                        }
                    }

                    if (layersOk) {
                        for (int ch = 0; ch < 3; ++ch) {
                            float total = 0.0f;
                            for (int layer = 0; layer < 3; ++layer) {
                                total += densityMaxLayers[layer][ch];
                            }
                            if (!is_positive_finite(total)) {
                                layersOk = false;
                                break;
                            }

                            for (int layer = 0; layer < 3; ++layer) {
                                const float fraction = densityMaxLayers[layer][ch] / total;
                                const float minLayer = fraction * densityMin[ch];
                                const float maxLayer = densityMaxLayers[layer][ch] + minLayer;
                                const float particleAreaLayer = grainUi.agxParticleAreaUm2 * grainUi.agxParticleScale[ch] * grainUi.agxParticleScaleLayers[layer];
                                if (!is_positive_finite(particleAreaLayer)) {
                                    layersOk = false;
                                    break;
                                }
                                const float nParticlesLayer = pixelAreaUm2 * fraction / particleAreaLayer;
                                const float odParticle = divide_or_zero_if_positive(maxLayer, nParticlesLayer);
                                run.grain.densityMinLayers[layer][ch] = minLayer;
                                run.grain.densityMaxLayers[layer][ch] = maxLayer;
                                run.grain.nParticlesLayers[layer][ch] = finite_or_zero(nParticlesLayer);
                                run.grain.odParticleLayers[layer][ch] = finite_or_zero(odParticle);
                                run.grain.densityCurvesLayers[layer][ch] = durableBundle.film.densityCurvesLayers[layer][ch];
                                const float dyeSigma = run.grain.blurDyeCloudsUm * std::sqrt(std::max(0.0f, run.grain.odParticleLayers[layer][ch]));
                                grainDyeSigmaPx[layer][ch] = finite_or_zero(dyeSigma);
                            }
                            if (!layersOk) {
                                break;
                            }
                        }
                    }
                    wantGrainSublayers = layersOk;
                }
            }
            if (is_positive_finite(grainBlurSigmaPx)) {
                wantGrainBlur = grain_blur_enabled(wantGrainSublayers, grainBlurSigmaPx);
            }
        }
        run.grain.active = bool_to_i32(wantGrain);
        run.grain.sublayersActive = bool_to_i32(wantGrainSublayers);

        // Debug view scaling: stable linear mapping for signed delta fields.
        {
            float densityMaxAvg = (run.grain.densityMax[0] + run.grain.densityMax[1] + run.grain.densityMax[2]) * (1.0f / 3.0f);
            if (!is_positive_finite(densityMaxAvg)) {
                densityMaxAvg = 1.0f;
            }
            run.grain.debugScale = 0.25f / std::max(1e-6f, densityMaxAvg);
        }

        // Phase 3: three-scale mix configuration (fine + mid + coarse).
        {
            float wC = sanitize_unit_or(run.grain.sizeMixWeight, 0.0f);
            float wM = sanitize_unit_or(run.grain.sizeMixWeightMid, 0.0f);
            float wF = 1.0f - wM - wC;
            if (wF < 0.0f) {
                wF = 0.0f;
            }
            float wSum = wF + wM + wC;
            if (wSum > 0.0f) {
                const float invSum = 1.0f / wSum;
                wF *= invSum;
                wM *= invSum;
                wC *= invSum;
            } else {
                wF = 1.0f;
                wM = 0.0f;
                wC = 0.0f;
            }
            run.grain.sizeMixWeight = wC;
            run.grain.sizeMixWeightMid = wM;

            const float scale = std::max(1.0f, sanitize_nonnegative_or(run.grain.sizeMixScale, 1.0f));
            const bool canMix = wantGrain && wantGrainBlur && is_positive_finite(grainBlurSigmaPx) && (scale > 1.0f);
            wantGrainMix = canMix && ((wM > 0.0f) || (wC > 0.0f));

            if (wantGrainMix) {
                const float sigmaF = grainBlurSigmaPx;
                const float sigmaCRaw = sigmaF * std::sqrt(scale);
                grainBlurSigmaCoarsePx = std::max(sigmaF, std::min(sigmaCRaw, sigmaF * 4.0f));
                if (!is_positive_finite(grainBlurSigmaCoarsePx)) {
                    wantGrainMix = false;
                    grainBlurSigmaCoarsePx = 0.0f;
                }
                if (wantGrainMix) {
                    grainBlurSigmaMidPx = std::sqrt(std::max(0.0f, sigmaF * grainBlurSigmaCoarsePx));
                    if (!is_positive_finite(grainBlurSigmaMidPx)) {
                        wantGrainMix = false;
                        grainBlurSigmaMidPx = 0.0f;
                    }
                }
            }
            if (!wantGrainMix) {
                run.grain.sizeMixGain = 1.0f;
                run.grain.sizeMixWeight = 0.0f;
                run.grain.sizeMixWeightMid = 0.0f;
                grainBlurSigmaMidPx = 0.0f;
                grainBlurSigmaCoarsePx = 0.0f;
            } else {
                auto kernel_energy_2d = [](float sigma) -> float {
                    if (!is_positive_finite(sigma)) {
                        return 1.0f;
                    }
                    const int radiusRaw = JuicerGaussian::scipy_gaussian_radius(sigma, 4.0f);
                    const int radius = std::min(radiusRaw, 75);
                    if (radius <= 0) {
                        return 1.0f;
                    }
                    constexpr int kMaxRadius = 75;
                    const double s2 = static_cast<double>(sigma) * static_cast<double>(sigma) * 2.0;
                    double wsum = 0.0;
                    double w[(kMaxRadius * 2) + 1] = {};
                    for (int i = -radius; i <= radius; ++i) {
                        const double iDouble = static_cast<double>(i);
                        const double wi = std::exp(-(iDouble * iDouble) / s2);
                        w[i + radius] = wi;
                        wsum += wi;
                    }
                    const double invW = reciprocal_or_zero(wsum);
                    double sumSq = 0.0;
                    for (int i = -radius; i <= radius; ++i) {
                        const double wn = w[i + radius] * invW;
                        sumSq += wn * wn;
                    }
                    const double e1 = std::max(0.0, sumSq);
                    const double e2 = e1 * e1;
                    return static_cast<float>(std::max(1e-12, e2));
                };

                const float sigmaF = grainBlurSigmaPx;
                const float sigmaM = grainBlurSigmaMidPx;
                const float sigmaC = grainBlurSigmaCoarsePx;
                const float eF = kernel_energy_2d(sigmaF);
                const float eM = kernel_energy_2d(sigmaM);
                const float eC = kernel_energy_2d(sigmaC);
                const float midScale = std::sqrt(scale);
                const float rM = midScale * (eM / std::max(1e-12f, eF));
                const float rC = scale * (eC / std::max(1e-12f, eF));
                const float denom = wF * wF + wM * wM * rM + wC * wC * rC;
                run.grain.sizeMixGain = reciprocal_sqrt_or_one(denom, 1e-12f);
            }
        }

        result.wantGrain = wantGrain;
        result.wantGrainSublayers = wantGrainSublayers;
        result.wantGrainBlur = wantGrainBlur;
        result.wantGrainMix = wantGrainMix;
        result.grainBlurSigmaPx = grainBlurSigmaPx;
        result.grainBlurSigmaMidPx = grainBlurSigmaMidPx;
        result.grainBlurSigmaCoarsePx = grainBlurSigmaCoarsePx;
        copy_float3x3(result.grainDyeSigmaPx, grainDyeSigmaPx);

        return result;
    };

    auto resolve_halation_metadata = [&]() -> Profiles::HalationMetadata {
        return _hasHalationOverride ? _halationOverride : Profiles::HalationMetadata{};
    };

    auto resolve_grain_metadata = [&]() -> Profiles::GrainMetadata {
        return _hasGrainOverride ? _grainOverride : _ws->grain;
    };

    struct OpticsFeatureSetup {
        HalationSetupResult halation{};
        GrainSetupResult grain{};
        bool needGrainShared = false;
    };

    struct GrainOpticsState {
        bool wantGrain = false;
        bool wantGrainSublayers = false;
        bool wantGrainBlur = false;
        bool wantGrainMix = false;
        float grainBlurSigmaPx = 0.0f;
        float grainBlurSigmaMidPx = 0.0f;
        bool needGrainShared = false;
    };

    auto setup_optics_feature_payloads = [&](JuicerCuda::PipelineRunParams& run,
                                             bool includeDefects) -> OpticsFeatureSetup {
        OpticsFeatureSetup setup{};
        setup.halation = setup_halation_payload(resolve_halation_metadata());
        setup.grain = setup_grain_payload(run, resolve_grain_metadata(), includeDefects);
        setup.needGrainShared = needs_grain_shared(
            setup.grain.wantGrain,
            run.grain.debugView,
            run.grain.chromaMix);
        return setup;
    };

    auto resolve_grain_optics_state = [&](const OpticsFeatureSetup& featureSetup) -> GrainOpticsState {
        GrainOpticsState state{};
        const GrainSetupResult& grainSetup = featureSetup.grain;
        state.wantGrain = grainSetup.wantGrain;
        state.wantGrainSublayers = grainSetup.wantGrainSublayers;
        state.wantGrainBlur = grainSetup.wantGrainBlur;
        state.wantGrainMix = grainSetup.wantGrainMix;
        state.grainBlurSigmaPx = grainSetup.grainBlurSigmaPx;
        state.grainBlurSigmaMidPx = grainSetup.grainBlurSigmaMidPx;
        state.needGrainShared = featureSetup.needGrainShared;
        return state;
    };

    auto build_optics_scratch_needs_for_stage = [&](bool wantGlareBlur,
                                                    const GrainOpticsState& grainState) -> OpticsScratchNeeds {
        return build_optics_scratch_needs(
            wantGlareBlur,
            grainState.wantGrainBlur,
            grainState.wantGrainSublayers,
            grainState.wantGrainMix);
    };

    auto throw_prepared_frame_failure = [&](const std::string& detail) {
        if (preparedFrame.failure_marks_context_loss()) {
            mark_context_and_throw_cuda_policy_fatal(
                preparedFrame.failure_stage_tag(),
                preparedFrame.failure_prefix(),
                detail);
        }
        trace_and_throw_cuda_policy_fatal(
            preparedFrame.failure_prefix(),
            cstr_or_null_if_empty(detail));
    };

    auto launch_base_pipeline_graph = [&](int renderModeKey,
                                          JuicerCuda::PipelineRunParams& run) -> cudaError_t {
        std::string graphError;
        int graphErrCode = static_cast<int>(cudaErrorUnknown);
        if (!preparedFrame.launch_base_pipeline_graph(
                run,
                renderModeKey,
                _pCudaStream,
                graphErrCode,
                graphError)) {
            throw_prepared_frame_failure(graphError);
        }
        return static_cast<cudaError_t>(graphErrCode);
    };

    auto prepare_scan_error_stage = [&](JuicerCuda::PipelineRunParams& run,
                                        cudaStream_t stream) -> bool {
        std::string scanErrorError;
        if (!preparedFrame.prepare_scan_error_stage(
                run.scanStage.scanErrorFlag,
                stream,
                scanErrorError)) {
            throw_prepared_frame_failure(scanErrorError);
        }
        return true;
    };

    auto finalize_scan_error_stage = [&](const JuicerCuda::PipelineRunParams& run,
                                         cudaStream_t stream) {
        std::string scanErrorError;
        if (!preparedFrame.finalize_scan_error_stage(
                run.scanStage.scanErrorFlag,
                stream,
                scanErrorError)) {
            throw_prepared_frame_failure(scanErrorError);
        }
    };

    auto finalize_cuda_pipeline_tail_or_abort = [&](const JuicerCuda::PipelineRunParams& run,
                                                    cudaStream_t pipelineStream) -> bool {
        if (abort_cuda_path_if_requested(preparedFrame)) {
            return false;
        }
        finalize_scan_error_stage(run, pipelineStream);
        record_cuda_use(preparedFrame);
        return true;
    };

    auto commit_submission_or_throw = [&]() {
        std::string commitError;
        if (!preparedFrame.finish(_pCudaStream, commitError)) {
            throw_submission_fatal("prepared_frame_finish", "prepared frame finish failed", commitError);
        }
    };

    struct ScanStageMediumSelection {
        const char* scanLabel = "scan";
        const JuicerCuda::Resources::DeviceSpectralLut* scanLut = nullptr;
        const JuicerCuda::Resources::DeviceScanMedium* scanMedium = nullptr;
    };

    auto select_scan_stage_medium = [&](bool negativeMedium) -> ScanStageMediumSelection {
        const JuicerProcess::Root::PreparedCudaFrame::DurableBundleView durableBundle =
            preparedFrame.durable_bundle();
        ScanStageMediumSelection selection{};
        selection.scanLabel = scan_stage_label_from_negative_medium(negativeMedium);
        selection.scanLut = durableBundle.scan.lut(negativeMedium);
        selection.scanMedium = durableBundle.scan.medium(negativeMedium);
        return selection;
    };

    auto setup_scan_stage_resources = [&](JuicerCuda::PipelineRunParams& run,
                                          const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& workspace,
                                          cudaStream_t stream,
                                          bool negativeMedium) -> bool {
        const ScanStageMediumSelection selection = select_scan_stage_medium(negativeMedium);
        const char* scanLabel = selection.scanLabel;
        run.scanStage.scannerUseLut = bool_to_i32(_scannerSettings.useLut);
        run.scanStage.scanLutLog2XYZ = nullptr;
        run.scanStage.scanLutRes = 0;
        if (run.scanStage.scannerUseLut) {
            std::string lutError;
            if (!preparedFrame.prepare_scan_lut(
                    *_ws,
                    negativeMedium,
                    workspace,
                    _pCudaStream,
                    lutError)) {
                throw_prepared_frame_failure(lutError);
            }
            const JuicerCuda::Resources::DeviceSpectralLut& scanLut = *selection.scanLut;
            run.scanStage.scanLutLog2XYZ = scanLut.log2XYZ;
            run.scanStage.scanLutRes = static_cast<int>(scanLut.res);
            if (!run.scanStage.scanLutLog2XYZ || run.scanStage.scanLutRes <= 0) {
                std::string missingLutPrefix = scanLabel;
                missingLutPrefix += " LUT missing after successful upload";
                trace_cuda_fatal_prefixed_if(traceInfo, missingLutPrefix.c_str());
                throw OFX::Exception::Suite(kOfxStatErrFatal);
            }
        }

        const JuicerCuda::Resources::DeviceScanMedium& scanMedium = *selection.scanMedium;
        copy_scan_tables_payload(
            run.scanStage.scanTables,
            run.scanStage.scanTables.mediumIsNegative,
            run.scanStage.scanTables.min_cmy,
            run.scanStage.scanTables.inv_max_cmy,
            scanMedium);

        return prepare_scan_error_stage(run, stream);
    };

    auto setup_spatial_dir_stage = [&](JuicerCuda::PipelineRunParams& run,
                                       const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& workspace,
                                       int frameWidth,
                                       int frameHeight,
                                       bool useSpatialDir) -> bool {
        run.filmDevelop.spatialDir.active = bool_to_i32(useSpatialDir);
        run.filmDevelop.spatialDir.corrY = nullptr;
        run.filmDevelop.spatialDir.corrM = nullptr;
        run.filmDevelop.spatialDir.corrC = nullptr;
        if (!useSpatialDir) {
            return true;
        }
        if (abort_cuda_path_if_requested(preparedFrame)) {
            return false;
        }

        std::string dirError;
        if (!preparedFrame.prepare_spatial_dir_scratch(
                workspace,
                _pCudaStream,
                dirError)) {
            throw_prepared_frame_failure(dirError);
        }
        if (!preparedFrame.prepare_spatial_dir_kernel(
                _dirRT.spatialSigmaPixels,
                _pCudaStream,
                dirError)) {
            throw_prepared_frame_failure(dirError);
        }

        const JuicerProcess::Root::PreparedCudaFrame::SpatialDirScratchView spatialDirScratch =
            preparedFrame.spatial_dir_scratch(workspace);
        if (!spatialDirScratch.active) {
            trace_and_throw_cuda_policy_fatal(
                "CUDA spatial DIR scratch missing",
                "spatial DIR scratch missing after allocation");
        }

        run.filmDevelop.spatialDir.corrY = spatialDirScratch.corrY;
        run.filmDevelop.spatialDir.corrM = spatialDirScratch.corrM;
        run.filmDevelop.spatialDir.corrC = spatialDirScratch.corrC;

        const JuicerProcess::Root::PreparedCudaFrame::OpticsKernelView opticsKernels =
            preparedFrame.optics_kernels();
        const cudaError_t dirErr = juicer_cuda_build_spatial_dir(
            &run,
            spatialDirScratch.corrY,
            spatialDirScratch.corrM,
            spatialDirScratch.corrC,
            spatialDirScratch.tmp,
            opticsKernels.spatialDir.weights,
            opticsKernels.spatialDir.radius,
            _pCudaStream);
        if (dirErr != cudaSuccess) {
            throw_cuda_stage_fatal("build_spatial_dir", "spatial DIR build failed", dirErr);
        }
        return true;
    };

    auto validate_cuda_scanner_preflight_or_throw = [&](bool scannerRuntimeValid,
                                                        const char* mediumLabel,
                                                        const Scanner::ScannerMediumRuntime* mediumRuntime)
        -> JuicerProcScanner::ScannerPreflightResult {
        return JuicerProcScanner::validate_scanner_preflight_or_throw(
            scannerRuntimeValid,
            mediumLabel,
            mediumRuntime,
            traceInfo,
            traceVerbose,
            "cuda",
            "CUDA");
    };

    auto ensure_print_illuminant_filtered_or_throw =
        [&](const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& workspace) {
            std::string illumError;
            if (preparedFrame.prepare_print_illuminant_filtered(
                    *_ws,
                    *_prt,
                    _printParams,
                    workspace,
                    _pCudaStream,
                    illumError)) {
                return;
            }
            throw_prepared_frame_failure(illumError);
        };

    auto trace_print_payload_verbose = [&]() {
        if (!traceVerbose) {
            return;
        }
        const JuicerProcess::Root::PreparedCudaFrame::UploadTraceView uploadTrace =
            preparedFrame.upload_trace_view();
        if (!uploadTrace.active) {
            return;
        }
        const std::uint64_t uploadCoreHash = upload_core_hash_or_core_hash(*_ws);
        std::string msg;
        msg.reserve(384);
        msg = "cuda print payload build=";
        msg += std::to_string(_ws->buildCounter);
        msg += " uploadCoreHash=";
        msg += std::to_string(uploadCoreHash);
        msg += " neutralY/M/C=";
        msg += std::to_string(_prt->neutralY);
        msg += "/";
        msg += std::to_string(_prt->neutralM);
        msg += "/";
        msg += std::to_string(_prt->neutralC);
        msg += " yFilter=";
        msg += std::to_string(_printParams.yFilter);
        msg += " mFilter=";
        msg += std::to_string(_printParams.mFilter);
        msg += " cFilter=";
        msg += std::to_string(_printParams.cFilter);
        msg += " illumBuild=";
        msg += std::to_string(uploadTrace.printIllumBuildCounter);
        msg += " illumCoreHash=";
        msg += std::to_string(uploadTrace.printIllumCoreHash);
        msg += " illumNeutralHash=";
        msg += std::to_string(uploadTrace.printIllumNeutralFilterHash);
        msg += " illumY/M/Csteps=";
        msg += std::to_string(uploadTrace.printIllumYShiftSteps);
        msg += "/";
        msg += std::to_string(uploadTrace.printIllumMShiftSteps);
        msg += "/";
        msg += std::to_string(uploadTrace.printIllumCShiftSteps);
        msg += " preflashValid=";
        msg += std::to_string(bool_to_i32(uploadTrace.printPreflashValid));
        msg += " preflashKey=";
        msg += std::to_string(uploadTrace.printPreflashKeyHash);
        JTRACE_VERBOSE("PRINTDBG", msg);
    };

    auto ensure_optics_scratch_or_throw = [&](const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& workspace,
                                              std::string& opticsError) {
        if (preparedFrame.prepare_optics_scratch(
                workspace,
                _pCudaStream,
                opticsError)) {
            return;
        }
        throw_prepared_frame_failure(opticsError);
    };

    auto throw_cuda_gate_mask_build_failure = [&](const char* stageTag, cudaError_t gateErr) {
        throw_cuda_stage_fatal(stageTag, "gate defect mask build failed", gateErr);
    };

    auto scanner_optics_scratch_or_throw =
        [&](const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& workspace)
        -> JuicerProcess::Root::PreparedCudaFrame::ScannerOpticsScratchView {
        const JuicerProcess::Root::PreparedCudaFrame::ScannerOpticsScratchView scratch =
            preparedFrame.scanner_optics_scratch(workspace);
        if (!scratch.active) {
            trace_and_throw_cuda_policy_fatal(
                "CUDA scanner optics scratch missing",
                "scanner optics scratch missing after allocation");
        }
        return scratch;
    };

    auto setup_gate_mask_if_needed = [&](JuicerCuda::PipelineRunParams& run,
                                         const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& workspace,
                                         bool needGateMask,
                                         const char* stageTag) -> bool {
        run.grain.gateMask = nullptr;
        run.grain.gateMaskWidth = 0;
        run.grain.gateMaskHeight = 0;
        const JuicerProcess::Root::PreparedCudaFrame::ScannerOpticsScratchView opticsScratch =
            scanner_optics_scratch_or_throw(workspace);
        if (!(needGateMask && opticsScratch.hasGateMask)) {
            return true;
        }

        GateMaskHashInputs gateMaskInputs{};
        gateMaskInputs.sessionSeed = run.grain.stbnSessionSeed;
        gateMaskInputs.originX = run.grain.originX;
        gateMaskInputs.originY = run.grain.originY;
        gateMaskInputs.width = width;
        gateMaskInputs.height = height;
        gateMaskInputs.pixelSizeUm = run.grain.pixelSizeUm;
        gateMaskInputs.gateDustAmount = run.grain.gateDustAmount;
        gateMaskInputs.gateScratchAmount = run.grain.gateScratchAmount;
        const std::uint64_t gateHash = make_gate_mask_hash(gateMaskInputs);
        if (gateHash != opticsScratch.gateMaskHash) {
            if (abort_cuda_path_if_requested(preparedFrame)) {
                return false;
            }
            cudaError_t gateErr = juicer_cuda_build_gate_defect_mask(
                &run,
                opticsScratch.gateMask,
                opticsScratch.gateMaskWidth,
                opticsScratch.gateMaskHeight,
                _pCudaStream);
            if (gateErr != cudaSuccess) {
                throw_cuda_gate_mask_build_failure(stageTag, gateErr);
            }
            preparedFrame.mark_gate_mask_built(gateHash);
        }

        run.grain.gateMask = opticsScratch.gateMask;
        run.grain.gateMaskWidth = opticsScratch.gateMaskWidth;
        run.grain.gateMaskHeight = opticsScratch.gateMaskHeight;
        return true;
    };

    auto throw_if_kernel_prepare_failed = [&](bool prepared,
                                              const char* kernelLabel,
                                              std::string& opticsError) {
        if (prepared) {
            return;
        }
        std::string prefix = make_cuda_prefixed_failure(nonempty_cstr_or(kernelLabel, "gaussian"),
                                                        " kernel upload failed");
        trace_and_throw_cuda_policy_fatal(prefix.c_str(), cstr_or_null_if_empty(opticsError));
    };

    auto ensure_grain_dye_kernel_or_throw = [&](int layer,
                                                int channel,
                                                float sigma,
                                                std::string& opticsError) {
        throw_if_kernel_prepare_failed(
            preparedFrame.prepare_grain_dye_kernel(
                layer,
                channel,
                sigma,
                _pCudaStream,
                opticsError),
            "grain dye-cloud",
            opticsError);
    };

    auto throw_pipeline_launch_failure = [&](const char* stageTag,
                                             const char* failurePrefix,
                                             cudaError_t pipelineErr) {
        throw_cuda_stage_fatal(
            stageTag,
            failurePrefix,
            pipelineErr);
    };

    auto throw_if_pipeline_launch_failed = [&](cudaError_t pipelineErr,
                                               const char* stageTag,
                                               const char* failurePrefix) {
        if (pipelineErr == cudaSuccess) {
            return;
        }
        throw_pipeline_launch_failure(stageTag, failurePrefix, pipelineErr);
    };

    auto reset_grain_kernel_slots = [&](JuicerCuda::PipelineRunParams& run) {
        run.grainKernels.blurKernel = nullptr;
        run.grainKernels.blurRadius = 0;
        run.grainKernels.blurKernelMid = nullptr;
        run.grainKernels.blurRadiusMid = 0;
        run.grainKernels.blurKernelCoarse = nullptr;
        run.grainKernels.blurRadiusCoarse = 0;
        std::fill_n(&run.grainKernels.dyeKernel[0][0], 9, nullptr);
        std::fill_n(&run.grainKernels.dyeRadius[0][0], 9, 0);
    };

    auto reset_halation_kernel_slots = [&](JuicerCuda::PipelineRunParams& run,
                                           bool wantHalation,
                                           const float* halationStrengthBGR,
                                           const float* halationScatterStrengthBGR) {
        run.halation.active = bool_to_i32(wantHalation);
        float* strengthIt = run.halation.strength;
        float* scatterStrengthIt = run.halation.scatteringStrength;
        const float* srcStrengthIt = halationStrengthBGR;
        const float* srcScatterStrengthIt = halationScatterStrengthBGR;
        const float* const srcStrengthEnd = srcStrengthIt + 3;
        const float fillValue = 0.0f;
        for (; srcStrengthIt != srcStrengthEnd;
             ++strengthIt, ++scatterStrengthIt, ++srcStrengthIt, ++srcScatterStrengthIt) {
            *strengthIt = float_if_enabled(wantHalation, *srcStrengthIt, fillValue);
            *scatterStrengthIt = float_if_enabled(wantHalation, *srcScatterStrengthIt, fillValue);
        }
        std::fill_n(run.halationKernels.halationKernel, 3, nullptr);
        std::fill_n(run.halationKernels.halationRadius, 3, 0);
        std::fill_n(run.halationKernels.scatteringKernel, 3, nullptr);
        std::fill_n(run.halationKernels.scatteringRadius, 3, 0);
    };

    auto bind_grain_dye_kernels_or_throw = [&](JuicerCuda::PipelineRunParams& run,
                                               const GrainSetupResult& grainSetup,
                                               std::string& opticsError) {
        for (int layer = 0; layer < 3; ++layer) {
            for (int ch = 0; ch < 3; ++ch) {
                const float sigma = grainSetup.grainDyeSigmaPx[layer][ch];
                ensure_grain_dye_kernel_or_throw(
                    layer,
                    ch,
                    sigma,
                    opticsError);
            }
        }

        const JuicerProcess::Root::PreparedCudaFrame::OpticsKernelView opticsKernels =
            preparedFrame.optics_kernels();
        for (int layer = 0; layer < 3; ++layer) {
            for (int ch = 0; ch < 3; ++ch) {
                const float sigma = grainSetup.grainDyeSigmaPx[layer][ch];
                if (sigma > 0.0f) {
                    run.grainKernels.dyeKernel[layer][ch] = opticsKernels.grainDye[layer][ch].weights;
                    run.grainKernels.dyeRadius[layer][ch] = opticsKernels.grainDye[layer][ch].radius;
                }
            }
        }
    };

    auto bind_halation_kernels_or_throw = [&](JuicerCuda::PipelineRunParams& run,
                                              const float* halationStrengthBGR,
                                              const float* halationSigmaPx,
                                              const float* halationScatterStrengthBGR,
                                              const float* halationScatterSigmaPx,
                                              std::string& opticsError) {
        const float* strengthIt = halationStrengthBGR;
        const float* sigmaIt = halationSigmaPx;
        const float* scatterStrengthIt = halationScatterStrengthBGR;
        const float* scatterSigmaIt = halationScatterSigmaPx;
        const float* const strengthEnd = strengthIt + 3;
        int i = 0;
        for (; strengthIt != strengthEnd; ++strengthIt, ++sigmaIt, ++scatterStrengthIt, ++scatterSigmaIt, ++i) {
            if (*strengthIt > 0.0f && *sigmaIt > 0.0f) {
                throw_if_kernel_prepare_failed(
                    preparedFrame.prepare_halation_kernel(
                        i,
                        *sigmaIt,
                        _pCudaStream,
                        opticsError),
                    "halation",
                    opticsError);
                const JuicerProcess::Root::PreparedCudaFrame::OpticsKernelView opticsKernels =
                    preparedFrame.optics_kernels();
                run.halationKernels.halationKernel[i] = opticsKernels.halation[i].weights;
                run.halationKernels.halationRadius[i] = opticsKernels.halation[i].radius;
            }
            if (*scatterStrengthIt > 0.0f && *scatterSigmaIt > 0.0f) {
                throw_if_kernel_prepare_failed(
                    preparedFrame.prepare_halation_scatter_kernel(
                        i,
                        *scatterSigmaIt,
                        _pCudaStream,
                        opticsError),
                    "halation scatter",
                    opticsError);
                const JuicerProcess::Root::PreparedCudaFrame::OpticsKernelView opticsKernels =
                    preparedFrame.optics_kernels();
                run.halationKernels.scatteringKernel[i] = opticsKernels.halationScatter[i].weights;
                run.halationKernels.scatteringRadius[i] = opticsKernels.halationScatter[i].radius;
            }
        }
    };

    auto bind_optics_kernels_or_throw = [&](JuicerCuda::PipelineRunParams& run,
                                            bool wantGlare,
                                            float glareBlurSigmaPx,
                                            float lensBlurSigmaPx,
                                            float unsharpSigmaPx,
                                            bool wantGrain,
                                            bool wantGrainBlur,
                                            bool wantGrainMix,
                                            bool wantGrainSublayers,
                                            float grainBlurSigmaPx,
                                            float grainBlurSigmaMidPx,
                                            const GrainSetupResult& grainSetup,
                                            bool wantHalation,
                                            const float* halationStrengthBGR,
                                            const float* halationSigmaPx,
                                            const float* halationScatterStrengthBGR,
                                            const float* halationScatterSigmaPx,
                                            std::string& opticsError) {
        throw_if_kernel_prepare_failed(
            preparedFrame.prepare_scanner_lens_blur_kernel(lensBlurSigmaPx, _pCudaStream, opticsError),
            "lens blur",
            opticsError);
        throw_if_kernel_prepare_failed(
            preparedFrame.prepare_scanner_unsharp_kernel(unsharpSigmaPx, _pCudaStream, opticsError),
            "unsharp",
            opticsError);
        throw_if_kernel_prepare_failed(
            preparedFrame.prepare_scanner_glare_kernel(
                sigma_if_enabled(wantGlare, glareBlurSigmaPx),
                _pCudaStream,
                opticsError),
            "glare",
            opticsError);

        reset_grain_kernel_slots(run);
        if (wantGrain) {
            throw_if_kernel_prepare_failed(
                preparedFrame.prepare_grain_blur_kernel(
                    sigma_if_enabled(wantGrainBlur, grainBlurSigmaPx),
                    _pCudaStream,
                    opticsError),
                "grain blur",
                opticsError);
            const JuicerProcess::Root::PreparedCudaFrame::OpticsKernelView opticsKernels =
                preparedFrame.optics_kernels();
            if (wantGrainBlur) {
                run.grainKernels.blurKernel = opticsKernels.grainBlur.weights;
                run.grainKernels.blurRadius = opticsKernels.grainBlur.radius;
            }
            if (wantGrainMix) {
                throw_if_kernel_prepare_failed(
                    preparedFrame.prepare_grain_blur_mid_kernel(
                        grainBlurSigmaMidPx,
                        _pCudaStream,
                        opticsError),
                    "grain mid blur",
                    opticsError);
                throw_if_kernel_prepare_failed(
                    preparedFrame.prepare_grain_blur_coarse_kernel(
                        grainSetup.grainBlurSigmaCoarsePx,
                        _pCudaStream,
                        opticsError),
                    "grain coarse blur",
                    opticsError);
                const JuicerProcess::Root::PreparedCudaFrame::OpticsKernelView mixedOpticsKernels =
                    preparedFrame.optics_kernels();
                run.grainKernels.blurKernelMid = mixedOpticsKernels.grainBlurMid.weights;
                run.grainKernels.blurRadiusMid = mixedOpticsKernels.grainBlurMid.radius;
                run.grainKernels.blurKernelCoarse = mixedOpticsKernels.grainBlurCoarse.weights;
                run.grainKernels.blurRadiusCoarse = mixedOpticsKernels.grainBlurCoarse.radius;
            }

            if (wantGrainSublayers) {
                bind_grain_dye_kernels_or_throw(run, grainSetup, opticsError);
            }
        }

        reset_halation_kernel_slots(run, wantHalation, halationStrengthBGR, halationScatterStrengthBGR);
        if (wantHalation) {
            bind_halation_kernels_or_throw(
                run,
                halationStrengthBGR,
                halationSigmaPx,
                halationScatterStrengthBGR,
                halationScatterSigmaPx,
                opticsError);
        }
    };

    struct GlareSetupResult {
        bool wantGlare = false;
        float percent = 0.0f;
        float roughness = 0.0f;
        float blurSigmaPx = 0.0f;
        std::uint64_t seed = 0;
    };

    struct OpticsIntent {
        bool wantLensBlur = false;
        bool wantUnsharp = false;
        bool wantGlareBlur = false;
        bool needGateMask = false;
        bool wantOptics = false;
    };

    struct PipelineLaunchResult {
        bool shouldReturnEarly = false;
        cudaError_t error = cudaSuccess;
    };

    struct ScannerOpticsParams {
        float lensBlurSigmaPx = 0.0f;
        float unsharpSigmaPx = 0.0f;
        float unsharpAmount = 0.0f;
    };

    auto setup_glare_payload = [&](const Scanner::ScannerMediumRuntime& medium,
                                   Scanner::ScannerMedium mediumType) -> GlareSetupResult {
        GlareSetupResult result{};
        result.wantGlare = medium.glare.active && (medium.glare.percent > 0.0f);
        if (!result.wantGlare) {
            return result;
        }

        result.percent = medium.glare.percent;
        result.roughness = medium.glare.roughness;
        result.blurSigmaPx = medium.glare.blur;

        const std::uint64_t seedBase = seed_base_for_pass(
            _sessionSeed,
            _clipToken,
            _frameIndex,
            kSeedPassGlare);
        const std::uint64_t glareFields[4] = {
            seedBase,
            static_cast<std::uint64_t>(_frameBoundsVersion),
            medium.staticKey.glareHash,
            static_cast<std::uint64_t>(mediumType)};
        result.seed = Hash::hash_bytes(glareFields, sizeof(glareFields));
        return result;
    };

    auto build_optics_intent = [&](const JuicerCuda::PipelineRunParams& run,
                                   float lensBlurSigmaPx,
                                   float unsharpSigmaPx,
                                   float unsharpAmount,
                                   bool wantGlare,
                                   float glareBlurSigmaPx,
                                   bool wantHalation,
                                   bool wantGrain) -> OpticsIntent {
        OpticsIntent intent{};
        intent.wantLensBlur = is_positive_finite(lensBlurSigmaPx);
        intent.wantUnsharp = wants_unsharp(unsharpSigmaPx, unsharpAmount);
        intent.wantGlareBlur = wantGlare && is_positive_finite(glareBlurSigmaPx);
        intent.needGateMask = needs_gate_mask_for_defects(
            run.grain.gateDustAmount,
            run.grain.gateScratchAmount);
        const bool wantWeave = (run.gateWeave.active != 0);
        const bool wantDefects = has_grain_defects(
            run.grain.filmDustAmount,
            run.grain.gateDustAmount,
            run.grain.filmScratchAmount,
            run.grain.gateScratchAmount);
        intent.wantOptics = wants_optics_stage(
            intent.wantLensBlur,
            intent.wantUnsharp,
            wantGlare,
            wantHalation,
            wantGrain,
            wantWeave,
            wantDefects);
        return intent;
    };

    auto launch_pipeline_with_optional_optics = [&](JuicerCuda::PipelineRunParams& run,
                                                    const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& workspace,
                                                    const ScannerOpticsParams& scannerOptics,
                                                    const OpticsIntent& opticsIntent,
                                                    const GrainOpticsState& grainState,
                                                    const GrainSetupResult& grainSetup,
                                                    const HalationSetupResult& halationSetup,
                                                    bool wantGlare,
                                                    float glareBlurSigmaPx,
                                                    std::uint64_t glareSeed,
                                                    float glarePercent,
                                                    float glareRoughness,
                                                    const char* gateStageTag,
                                                    auto&& launchOpticsKernel) -> PipelineLaunchResult {
        PipelineLaunchResult result{};
        if (!opticsIntent.wantOptics) {
            result.error = launch_base_pipeline_graph(
                static_cast<int>(renderMode),
                run);
            return result;
        }

        std::string opticsError;
        ensure_optics_scratch_or_throw(
            workspace,
            opticsError);
        if (!setup_gate_mask_if_needed(
                run,
                workspace,
                opticsIntent.needGateMask,
                gateStageTag)) {
            result.shouldReturnEarly = true;
            return result;
        }
        bind_optics_kernels_or_throw(
            run,
            wantGlare,
            glareBlurSigmaPx,
            scannerOptics.lensBlurSigmaPx,
            scannerOptics.unsharpSigmaPx,
            grainState.wantGrain,
            grainState.wantGrainBlur,
            grainState.wantGrainMix,
            grainState.wantGrainSublayers,
            grainState.grainBlurSigmaPx,
            grainState.grainBlurSigmaMidPx,
            grainSetup,
            halationSetup.wantHalation,
            halationSetup.strengthBGR,
            halationSetup.sigmaPx,
            halationSetup.scatterStrengthBGR,
            halationSetup.scatterSigmaPx,
            opticsError);
        JuicerCuda::LaunchGraphCounters::record_kernel_launch();
        result.error = launchOpticsKernel(glareSeed, glarePercent, glareRoughness);
        return result;
    };

    auto resolve_scanner_optics_params = [&]() -> ScannerOpticsParams {
        ScannerOpticsParams params{};
        params.lensBlurSigmaPx = _scannerOptions.lensBlurSigmaPx;
        params.unsharpSigmaPx = _scannerOptions.unsharpSigmaPx;
        params.unsharpAmount = _scannerOptions.unsharpAmount;
        return params;
    };

    auto ensure_prepared_frame_active_or_throw = [&](const char* stageLabel) {
        if (preparedFrame.active()) {
            return;
        }
        std::string prefix = "CUDA prepared frame inactive for ";
        prefix += nonempty_cstr_or(stageLabel, "pipeline");
        trace_cuda_fatal_prefixed_if(true, prefix.c_str());
        throw OFX::Exception::Suite(kOfxStatErrFatal);
    };

    auto initialize_pipeline_run = [&](JuicerCuda::PipelineRunParams& run) {
        run.src = srcPtr;
        run.srcRowBytes = static_cast<std::size_t>(srcRowBytes);
        run.dst = dstPtr;
        run.dstRowBytes = static_cast<std::size_t>(dstRowBytes);
        run.width = width;
        run.height = height;
        run.nComponents = _nComponents;
    };

    auto populate_scan_color_payload = [&](JuicerCuda::PipelineRunParams& run,
                                           const Scanner::ColorRuntime& color) {
        copy_float9(run.scanStage.scanColor.cat02, color.cat02);
        copy_float9(run.scanStage.scanColor.xyzToRgb, color.xyzToRgb);
        copy_float3(run.scanStage.scanColor.illuminantXYZ, color.illuminantXYZ);

        run.scanStage.scanColor.encoding.outputColorSpaceIndex = OutputEncoding::toIndex(color.encoding.colorSpace);
        run.scanStage.scanColor.encoding.applyCctfEncoding = bool_to_i32(color.encoding.applyCctfEncoding);
        run.scanStage.scanColor.encoding.preserveLinearRange = bool_to_i32(color.encoding.preserveLinearRange);
        run.scanStage.scanColor.encoding.inputIsOutputSpace = bool_to_i32(color.encoding.inputIsOutputSpace);

        const auto& outSpace = GeneratedColorSpaces::get(color.encoding.colorSpace);
        run.scanStage.scanColor.encoding.cctf.kind = static_cast<int>(outSpace.cctf.kind);
        run.scanStage.scanColor.encoding.cctf.gamma = outSpace.cctf.gamma;
        run.scanStage.scanColor.encoding.cctf.a = outSpace.cctf.a;
        run.scanStage.scanColor.encoding.cctf.b = outSpace.cctf.b;
        run.scanStage.scanColor.encoding.cctf.c = outSpace.cctf.c;
        run.scanStage.scanColor.encoding.cctf.d = outSpace.cctf.d;
        run.scanStage.scanColor.encoding.cctf.linearCutoff = outSpace.cctf.linearCutoff;

        const OutputEncoding::Matrix3x3 dwgToOutput = OutputEncoding::dwg_to_output_matrix(color.encoding.colorSpace);
        copy_float9(run.scanStage.scanColor.encoding.dwgToOutput, dwgToOutput.m);
    };

    auto populate_common_pipeline_payload = [&](JuicerCuda::PipelineRunParams& run,
                                                const JuicerProcScanner::ScannerPreflightResult& scannerPreflight) {
        run.filmRaw.inputColorSpaceIndex = Spectral::inputColorSpaceToIndex(_ws->filmRaw.inputColorSpace);
        run.filmRaw.applyCctfDecoding = bool_to_i32(_ws->filmRaw.applyCctfDecoding);
        run.filmRaw.applyInputChromaticAdapt = bool_to_i32(_ws->filmRaw.applyInputChromaticAdapt);
        run.filmRaw.spectralUpsamplingMode = static_cast<int>(_ws->filmRaw.spectralUpsamplingMode);
        copy_float9(run.filmRaw.inputRGBToXYZ, _ws->filmRaw.inputRGBToXYZ.m);
        copy_float9(run.filmRaw.inputXYZAdapt, _ws->filmRaw.inputXYZAdapt.m);
        run.filmRaw.mallettGreenMidgrayScale =
            directFilmRaw ? directFilmRaw->mallettGreenMidgrayScale : _ws->filmRaw.midgrayScale;
        copy_float3(run.filmRaw.refIllumWhiteXYZ, _ws->filmRaw.refIllumWhiteXYZ);

        const double manualExposureEv =
            directFilmRaw
                ? static_cast<double>(directFilmRaw->manualExposureCompensationEv)
                : _cameraSliderEV;
        const double manualExposureScale = std::exp2(manualExposureEv);
        run.filmExpose.manualExposureScale =
            std::isfinite(manualExposureScale) && manualExposureScale > 0.0
                ? static_cast<float>(manualExposureScale)
                : 1.0f;
        if (directFilmRaw) {
            run.filmRaw.inputColorSpaceIndex = directFilmRaw->inputColorSpace;
            run.filmRaw.applyCctfDecoding = bool_to_i32(directFilmRaw->inputCctfDecoding);
            run.filmRaw.spectralUpsamplingMode =
                directFilmRaw->rgbToRawMethod == Spektrafilm::RgbToRawMethod::Mallett2019 ? 1 : 0;
            run.filmExpose.highlightBoost.boostEv = directFilmRaw->highlightBoost.boostEv;
            run.filmExpose.highlightBoost.boostRange = directFilmRaw->highlightBoost.boostRange;
            run.filmExpose.highlightBoost.protectEv = directFilmRaw->highlightBoost.protectEv;
        }
        run.filmDevelop.gammaFactorB = _ws->gammaFactorB;
        run.filmDevelop.gammaFactorG = _ws->gammaFactorG;
        run.filmDevelop.gammaFactorR = _ws->gammaFactorR;
        run.filmDevelop.dirPrecorrected = bool_to_i32(_ws->dirPrecorrected);

        run.filmDevelop.dir.active = bool_to_i32(_dirRT.active);
        run.filmDevelop.dir.highShift = _dirRT.highShift;
        copy_float9(run.filmDevelop.dir.M, &_dirRT.M[0][0]);
        copy_float3(run.filmDevelop.dir.dMax, _dirRT.dMax);

        populate_scan_color_payload(run, *scannerPreflight.colorRuntime);
    };

    auto populate_film_runtime_payload = [&](JuicerCuda::PipelineRunParams& run) {
        const JuicerProcess::Root::PreparedCudaFrame::DurableBundleView durableBundle =
            preparedFrame.durable_bundle();
        const auto& film = durableBundle.film;
        run.filmDevelop.densB = {film.densB->x, film.densB->y, film.densB->n, film.densB->domainBegin, film.densB->domainEnd};
        run.filmDevelop.densG = {film.densG->x, film.densG->y, film.densG->n, film.densG->domainBegin, film.densG->domainEnd};
        run.filmDevelop.densR = {film.densR->x, film.densR->y, film.densR->n, film.densR->domainBegin, film.densR->domainEnd};
        run.filmDevelop.dirDensB = {film.dirDensB->x, film.dirDensB->y, film.dirDensB->n, film.dirDensB->domainBegin, film.dirDensB->domainEnd};
        run.filmDevelop.dirDensG = {film.dirDensG->x, film.dirDensG->y, film.dirDensG->n, film.dirDensG->domainBegin, film.dirDensG->domainEnd};
        run.filmDevelop.dirDensR = {film.dirDensR->x, film.dirDensR->y, film.dirDensR->n, film.dirDensR->domainBegin, film.dirDensR->domainEnd};
        run.filmExpose.sensB = {film.sensB->x, film.sensB->y, film.sensB->n, film.sensB->domainBegin, film.sensB->domainEnd};
        run.filmExpose.sensG = {film.sensG->x, film.sensG->y, film.sensG->n, film.sensG->domainBegin, film.sensG->domainEnd};
        run.filmExpose.sensR = {film.sensR->x, film.sensR->y, film.sensR->n, film.sensR->domainBegin, film.sensR->domainEnd};

        run.filmExpose.tablesAx = film.tablesAx;
        run.filmExpose.tablesAy = film.tablesAy;
        run.filmExpose.tablesAz = film.tablesAz;
        run.filmExpose.tablesIllum = film.tablesIllum;
        run.filmExpose.tablesK = film.tablesK;
        copy_float9(run.filmExpose.spdSInv, film.spdSInv);

        run.filmExpose.hanatosLut = film.hanatosLut;
        run.filmExpose.hanatosN = film.hanatosN;
        run.filmExpose.hanatosLutIntegrated = film.hanatosLutIntegrated;
        run.filmExpose.hanatosNIntegrated = film.hanatosNIntegrated;
        run.filmExpose.mallettBasis = film.mallettBasis;
        run.filmExpose.mallettBasisK = film.mallettBasisK;
    };

    const ScannerOpticsParams scannerOptics = resolve_scanner_optics_params();
    const cudaStream_t stream = cuda_stream_or_null(_pCudaStream);
    const bool useSpatialDIR = spatial_dir_enabled(_dirRT);

    struct OpticsLaunchInputs {
        GlareSetupResult glare{};
        HalationSetupResult halation{};
        GrainSetupResult grain{};
        GrainOpticsState grainState{};
        OpticsIntent intent{};
        OpticsScratchNeeds scratchNeeds{};
    };

    auto build_optics_launch_inputs = [&](JuicerCuda::PipelineRunParams& run,
                                          const Scanner::ScannerMediumRuntime& mediumRuntime,
                                          Scanner::ScannerMedium mediumType,
                                          bool includeDefects) -> OpticsLaunchInputs {
        OpticsLaunchInputs inputs{};
        inputs.glare = setup_glare_payload(mediumRuntime, mediumType);
        const OpticsFeatureSetup featureSetup = setup_optics_feature_payloads(run, includeDefects);
        inputs.halation = featureSetup.halation;
        inputs.grain = featureSetup.grain;
        inputs.grainState = resolve_grain_optics_state(featureSetup);
        inputs.intent = build_optics_intent(
            run,
            scannerOptics.lensBlurSigmaPx,
            scannerOptics.unsharpSigmaPx,
            scannerOptics.unsharpAmount,
            inputs.glare.wantGlare,
            inputs.glare.blurSigmaPx,
            inputs.halation.wantHalation,
            inputs.grainState.wantGrain);
        if (inputs.intent.wantOptics) {
            inputs.scratchNeeds = build_optics_scratch_needs_for_stage(
                inputs.intent.wantGlareBlur,
                inputs.grainState);
        }
        return inputs;
    };

    auto build_medium_workspace_request = [&](const OpticsLaunchInputs& opticsInputs)
        -> JuicerProcess::Root::PreparedCudaFrame::WorkspaceRequest {
        JuicerProcess::Root::PreparedCudaFrame::WorkspaceRequest request{};
        request.needOptics = opticsInputs.intent.wantOptics;
        request.needSpatialDir = useSpatialDIR;
        request.requestedWidth = width;
        request.requestedHeight = height;
        request.needBlurred = opticsInputs.scratchNeeds.blurred;
        request.needAux = opticsInputs.scratchNeeds.aux;
        request.needGrainTriplet = opticsInputs.scratchNeeds.grain;
        request.needGrainShared = opticsInputs.grainState.needGrainShared;
        request.needGateMask = opticsInputs.intent.needGateMask;
        return request;
    };

    auto prepare_common_cuda_pipeline_stages = [&](JuicerCuda::PipelineRunParams& run,
                                                   const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& workspace,
                                                   bool useSpatialDIR,
                                                   bool negativeMedium) -> bool {
        setup_camera_auto_exposure(run);
        populate_film_runtime_payload(run);
        if (!setup_scan_stage_resources(run, workspace, stream, negativeMedium)) {
            return false;
        }
        return setup_spatial_dir_stage(run, workspace, width, height, useSpatialDIR);
    };

    auto initialize_medium_pipeline_run = [&](JuicerCuda::PipelineRunParams& run,
                                              const JuicerProcScanner::ScannerPreflightResult& scannerPreflight) {
        initialize_pipeline_run(run);
        populate_common_pipeline_payload(run, scannerPreflight);
    };

    auto prepare_common_cuda_pipeline_stages_for_medium = [&](JuicerCuda::PipelineRunParams& run,
                                                              const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& workspace,
                                                              bool negativeMedium) -> bool {
        return prepare_common_cuda_pipeline_stages(
            run,
            workspace,
            useSpatialDIR,
            negativeMedium);
    };

    auto launch_negative_optics_kernel = [&](JuicerCuda::PipelineRunParams& run,
                                             const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& workspace,
                                             std::uint64_t glareSeedValue,
                                             float glarePercentValue,
                                             float glareRoughnessValue) -> cudaError_t {
        const JuicerProcess::Root::PreparedCudaFrame::OpticsKernelView opticsKernels =
            preparedFrame.optics_kernels();
        const JuicerProcess::Root::PreparedCudaFrame::ScannerOpticsScratchView opticsScratch =
            scanner_optics_scratch_or_throw(workspace);
        return juicer_cuda_negative_pipeline_optics(
            &run,
            opticsScratch.rgbR,
            opticsScratch.rgbG,
            opticsScratch.rgbB,
            opticsScratch.tmp,
            opticsScratch.blurred,
            opticsScratch.aux,
            opticsScratch.grainTmp,
            opticsScratch.grainTmpShared,
            opticsScratch.grainTmpMid,
            opticsScratch.grainTmpCoarse,
            opticsKernels.scannerLensBlur.weights,
            opticsKernels.scannerLensBlur.radius,
            opticsKernels.scannerUnsharp.weights,
            opticsKernels.scannerUnsharp.radius,
            scannerOptics.unsharpAmount,
            win.x1,
            win.y1,
            glareSeedValue,
            glarePercentValue,
            glareRoughnessValue,
            opticsKernels.scannerGlare.weights,
            opticsKernels.scannerGlare.radius,
            _pCudaStream);
    };

    auto launch_print_optics_kernel = [&](JuicerCuda::PipelineRunParams& run,
                                          const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& workspace,
                                          std::uint64_t glareSeedValue,
                                          float glarePercentValue,
                                          float glareRoughnessValue) -> cudaError_t {
        const JuicerProcess::Root::PreparedCudaFrame::OpticsKernelView opticsKernels =
            preparedFrame.optics_kernels();
        const JuicerProcess::Root::PreparedCudaFrame::ScannerOpticsScratchView opticsScratch =
            scanner_optics_scratch_or_throw(workspace);
        return juicer_cuda_print_pipeline_optics(
            &run,
            opticsScratch.rgbR,
            opticsScratch.rgbG,
            opticsScratch.rgbB,
            opticsScratch.tmp,
            opticsScratch.blurred,
            opticsScratch.aux,
            opticsScratch.grainTmp,
            opticsScratch.grainTmpShared,
            opticsScratch.grainTmpMid,
            opticsScratch.grainTmpCoarse,
            opticsKernels.scannerLensBlur.weights,
            opticsKernels.scannerLensBlur.radius,
            opticsKernels.scannerUnsharp.weights,
            opticsKernels.scannerUnsharp.radius,
            scannerOptics.unsharpAmount,
            win.x1,
            win.y1,
            glareSeedValue,
            glarePercentValue,
            glareRoughnessValue,
            opticsKernels.scannerGlare.weights,
            opticsKernels.scannerGlare.radius,
            _pCudaStream);
    };

    auto populate_print_pipeline_payload = [&](JuicerCuda::PipelineRunParams& run,
                                               float midgrayFactor) {
        const JuicerProcess::Root::PreparedCudaFrame::DurableBundleView durableBundle =
            preparedFrame.durable_bundle();
        const auto& scan = durableBundle.scan;
        const auto& print = durableBundle.print;
        run.printExpose.active = 1;
        copy_scan_tables_payload(
            run.printExpose.negTables,
            run.printExpose.negTables.mediumIsNegative,
            run.printExpose.negTables.min_cmy,
            run.printExpose.negTables.inv_max_cmy,
            *scan.negativeMedium);

        run.printExpose.printIllumFiltered = print.printIllumFiltered;
        run.printExpose.printIllumK = print.printIllumK;
        run.printExpose.printSensC = {print.printSensC->x, print.printSensC->y, print.printSensC->n, print.printSensC->domainBegin, print.printSensC->domainEnd};
        run.printExpose.printSensM = {print.printSensM->x, print.printSensM->y, print.printSensM->n, print.printSensM->domainBegin, print.printSensM->domainEnd};
        run.printExpose.printSensY = {print.printSensY->x, print.printSensY->y, print.printSensY->n, print.printSensY->domainBegin, print.printSensY->domainEnd};
        run.printDevelop.printDcC = {print.printDcC->x, print.printDcC->y, print.printDcC->n, print.printDcC->domainBegin, print.printDcC->domainEnd};
        run.printDevelop.printDcM = {print.printDcM->x, print.printDcM->y, print.printDcM->n, print.printDcM->domainBegin, print.printDcM->domainEnd};
        run.printDevelop.printDcY = {print.printDcY->x, print.printDcY->y, print.printDcY->n, print.printDcY->domainBegin, print.printDcY->domainEnd};
        run.printDevelop.printGammaC = print.printGammaC;
        run.printDevelop.printGammaM = print.printGammaM;
        run.printDevelop.printGammaY = print.printGammaY;
        run.printExpose.printExposure = _printParams.exposure;
        run.printExpose.printPreflashExposure = _printParams.preflashExposure;
        run.printExpose.printMidgrayFactor = midgrayFactor;
        copy_float3(run.printExpose.printPreflashRaw, print.printPreflashRaw);
    };

    auto print_pipeline_payloads_ready = [&](const JuicerCuda::PipelineRunParams& run) -> bool {
        return run.printExpose.printIllumFiltered && run.printExpose.printIllumK > 0 &&
               run.printExpose.printSensC.y && run.printExpose.printSensM.y && run.printExpose.printSensY.y &&
               run.printDevelop.printDcC.y && run.printDevelop.printDcM.y && run.printDevelop.printDcY.y;
    };

    auto throw_if_print_payloads_missing = [&](const JuicerCuda::PipelineRunParams& run) {
        if (print_pipeline_payloads_ready(run)) {
            return;
        }
        throw_cuda_policy_fatal("CUDA print payloads missing; cannot render print pipeline");
    };

    auto begin_medium_pipeline_or_abort = [&](const char* stageLabel) -> bool {
        ensure_prepared_frame_active_or_throw(stageLabel);
        return !abort_cuda_path_if_requested(preparedFrame);
    };

    auto checkpoint_medium_scratch_phase_or_throw = [&](const char* stageTag,
                                                        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& workspace) {
        if (!workspace.has_any_family()) {
            return;
        }

        std::string scratchPhaseError;
        if (!preparedFrame.checkpoint_scratch_phase(
                workspace,
                stageTag,
                scratchPhaseError)) {
            throw_prepared_frame_failure(scratchPhaseError);
        }
    };

    auto ensure_current_medium_uploaded_or_throw =
        [&](bool negativeMedium,
            const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& workspace) {
            std::string currentMediumUploadError;
            if (preparedFrame.prepare_current_medium(
                    *_ws,
                    negativeMedium,
                    workspace,
                    _pCudaStream,
                    currentMediumUploadError)) {
                return;
            }
            throw_prepared_frame_failure(currentMediumUploadError);
        };

    auto validate_negative_scanner_preflight_or_throw = [&]() -> JuicerProcScanner::ScannerPreflightResult {
        const JuicerProcScanner::ScannerMediumRuntimeBinding scannerMedium = JuicerProcScanner::bind_scanner_medium_runtime(
            *_ws,
            /*printActive*/ false,
            /*hasPrintGlareOverride*/ false,
            nullptr,
            /*forcePrintGlareHash*/ false);
        return validate_cuda_scanner_preflight_or_throw(
            scannerMedium.valid,
            scannerMedium.label,
            scannerMedium.runtime());
    };

    auto finalize_pipeline_launch_or_abort = [&](const PipelineLaunchResult& launchResult,
                                                 JuicerCuda::PipelineRunParams& run,
                                                 const char* launchStageTag,
                                                 const char* launchFailurePrefix) -> bool {
        if (launchResult.shouldReturnEarly) {
            return false;
        }
        throw_if_pipeline_launch_failed(
            launchResult.error,
            launchStageTag,
            launchFailurePrefix);
        return finalize_cuda_pipeline_tail_or_abort(
            run,
            stream);
    };

    auto launch_and_finalize_medium_pipeline_or_abort = [&](JuicerCuda::PipelineRunParams& run,
                                                            const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& workspace,
                                                            const OpticsLaunchInputs& opticsInputs,
                                                            const char* gateStageTag,
                                                            const char* launchStageTag,
                                                            const char* launchFailurePrefix,
                                                            auto&& launchOpticsKernel) -> bool {
        if (abort_cuda_path_if_requested(preparedFrame)) {
            return false;
        }
        const PipelineLaunchResult launchResult = launch_pipeline_with_optional_optics(
            run,
            workspace,
            scannerOptics,
            opticsInputs.intent,
            opticsInputs.grainState,
            opticsInputs.grain,
            opticsInputs.halation,
            opticsInputs.glare.wantGlare,
            opticsInputs.glare.blurSigmaPx,
            opticsInputs.glare.seed,
            opticsInputs.glare.percent,
            opticsInputs.glare.roughness,
            gateStageTag,
            [&](std::uint64_t glareSeedValue, float glarePercentValue, float glareRoughnessValue) {
                return launchOpticsKernel(
                    run,
                    workspace,
                    glareSeedValue,
                    glarePercentValue,
                    glareRoughnessValue);
            });
        return finalize_pipeline_launch_or_abort(
            launchResult,
            run,
            launchStageTag,
            launchFailurePrefix);
    };

    auto run_medium_optics_pipeline_or_abort = [&](JuicerCuda::PipelineRunParams& run,
                                                   const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& workspace,
                                                   const OpticsLaunchInputs& opticsInputs,
                                                   const char* gateStageTag,
                                                   const char* launchStageTag,
                                                   const char* launchFailurePrefix,
                                                   auto&& launchOpticsKernel) -> bool {
        return launch_and_finalize_medium_pipeline_or_abort(
            run,
            workspace,
            opticsInputs,
            gateStageTag,
            launchStageTag,
            launchFailurePrefix,
            launchOpticsKernel);
    };

    // RenderMode::NegativeOnly (PrintBypass=true).
    if (renderMode == RenderMode::NegativeOnly) {
        JuicerProcScanner::ScannerPreflightResult scannerPreflight =
            validate_negative_scanner_preflight_or_throw();
        const Scanner::ScannerMediumRuntime& negativeMediumRuntime = *scannerPreflight.mediumRuntime;

        JuicerCuda::PipelineRunParams run{};
        initialize_medium_pipeline_run(run, scannerPreflight);
        const OpticsLaunchInputs opticsInputs = build_optics_launch_inputs(
            run,
            negativeMediumRuntime,
            Scanner::ScannerMedium::Negative,
            true);
        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceRequest workspaceRequest =
            build_medium_workspace_request(opticsInputs);
        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker workspace =
            preparedFrame.bind_workspace_request(workspaceRequest);

        {
            if (!begin_medium_pipeline_or_abort("negative pipeline")) {
                return;
            }
            checkpoint_medium_scratch_phase_or_throw(
                "command_checkpoint_negative_medium_scratch_phase",
                workspace);
            ensure_current_medium_uploaded_or_throw(true, workspace);

            if (!prepare_common_cuda_pipeline_stages_for_medium(
                    run,
                    workspace,
                    true)) {
                return;
            }

            if (!run_medium_optics_pipeline_or_abort(
                    run,
                    workspace,
                    opticsInputs,
                    "build_gate_mask_negative",
                    "negative_pipeline_kernel_launch",
                    "negative pipeline kernel launch failed",
                    launch_negative_optics_kernel)) {
                return;
            }
        }
        commit_submission_or_throw();
        JuicerCuda::LaunchGraphCounters::record_frame_completed();
        return;
    }

    // RenderMode::Print (PrintBypass=false).
    {
        if (!_printReady || !_prt) {
            throw_cuda_policy_fatal("CUDA print pipeline prerequisites unavailable");
        }

        const JuicerProcScanner::ScannerMediumRuntimeBinding printScannerBinding = JuicerProcScanner::bind_scanner_medium_runtime(
            *_ws,
            /*printActive*/ true,
            _hasPrintGlareOverride,
            &_printGlareOverride,
            /*forcePrintGlareHash*/ false);
        JuicerProcScanner::ScannerPreflightResult scannerPreflight = validate_cuda_scanner_preflight_or_throw(
            printScannerBinding.valid,
            printScannerBinding.label,
            printScannerBinding.runtime());
        const Scanner::ScannerMediumRuntime& printMediumRuntime = *scannerPreflight.mediumRuntime;

        // Print exposure compensation factor was captured on CPU before CUDA pipeline setup.
        const float kMidSpectral = printMidgrayFactor;

        JuicerCuda::PipelineRunParams run{};
        initialize_medium_pipeline_run(run, scannerPreflight);
        const OpticsLaunchInputs opticsInputs = build_optics_launch_inputs(
            run,
            printMediumRuntime,
            Scanner::ScannerMedium::Print,
            false);
        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceRequest workspaceRequest =
            build_medium_workspace_request(opticsInputs);
        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker workspace =
            preparedFrame.bind_workspace_request(workspaceRequest);

        {
            if (!begin_medium_pipeline_or_abort("print pipeline")) {
                return;
            }

            checkpoint_medium_scratch_phase_or_throw(
                "command_checkpoint_print_medium_scratch_phase",
                workspace);
            ensure_current_medium_uploaded_or_throw(false, workspace);

            // Ensure the print illuminant filtered is available for current print params.
            ensure_print_illuminant_filtered_or_throw(workspace);
            trace_print_payload_verbose();

            if (!prepare_common_cuda_pipeline_stages_for_medium(
                    run,
                    workspace,
                    false)) {
                return;
            }

            // Print pipeline payloads.
            populate_print_pipeline_payload(run, kMidSpectral);
            throw_if_print_payloads_missing(run);

            if (!run_medium_optics_pipeline_or_abort(
                    run,
                    workspace,
                    opticsInputs,
                    "build_gate_mask_print",
                    "print_pipeline_kernel_launch",
                    "print pipeline kernel launch failed",
                    launch_print_optics_kernel)) {
                return;
            }
        }
        commit_submission_or_throw();
        JuicerCuda::LaunchGraphCounters::record_frame_completed();
        return;
    }
#endif
}
