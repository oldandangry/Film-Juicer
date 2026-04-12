// Cuda/JuicerCudaResources.cpp
//
// WorkingState uploads + primitive validation hooks.
//
#include "Cuda/JuicerCudaResources.h"
#include "Cuda/JuicerCudaPayloads.h"
#include "Cuda/ResourceManager/JuicerCudaResourceCore.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"

#include "FilmProcessing.h"
#include "ColorTransforms.h"
#include "PipelineTypes.h"
#include "SpectralProcessing.h"
#include "Print.h"
#include "JuicerState.h"
#include "ProcessRoot.h"

#include "GaussianSciPy.h"

#include "Logging.h"
#include "Hash.h"
#include "nlohmann/json.hpp"

extern const std::string gDataDir;

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
#include <cuda_runtime.h>
#include <cuda.h>
#if defined(_WIN32)
#include <windows.h>
#endif
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <atomic>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace JuicerCuda {
namespace Precompute {

bool build_scan_lut_cpu(const Scanner::ScannerMediumRuntime& medium, std::uint32_t res, std::vector<double>& out, std::string& outError) {
    // Match the GPU hot path: store log2(XYZ) so device code can use exp2() instead of pow(10, ...).
    // Note: Scanner::spectral_to_log_xyz returns log10(XYZ) (with epsilon).
    constexpr double kLog2_10 = 3.32192809488736234787;

    const size_t sRes = static_cast<size_t>(res);
    const size_t voxels = sRes * sRes * sRes;
    const size_t count = voxels * 3u;
    out.clear();
    out.resize(count);

    for (std::uint32_t z = 0; z < res; ++z) {
        const double nz = (res > 1u) ? static_cast<double>(z) / static_cast<double>(res - 1u) : 0.0;
        for (std::uint32_t y = 0; y < res; ++y) {
            const double ny = (res > 1u) ? static_cast<double>(y) / static_cast<double>(res - 1u) : 0.0;
            for (std::uint32_t x = 0; x < res; ++x) {
                const double nx = (res > 1u) ? static_cast<double>(x) / static_cast<double>(res - 1u) : 0.0;
                const size_t idx = (static_cast<size_t>(z) * sRes + static_cast<size_t>(y)) * sRes + static_cast<size_t>(x);
                const size_t base = idx * 3u;
                const double D_norm[3] = { nx, ny, nz };
                double logXYZ[3] = { 0.0, 0.0, 0.0 };
                Scanner::spectral_to_log_xyz(medium, D_norm, logXYZ);
                if (!std::isfinite(logXYZ[0]) || !std::isfinite(logXYZ[1]) || !std::isfinite(logXYZ[2])) {
                    outError = "scan LUT build produced non-finite logXYZ";
                    return false;
                }
                out[base + 0] = logXYZ[0];
                out[base + 1] = logXYZ[1];
                out[base + 2] = logXYZ[2];

                out[base + 0] *= kLog2_10;
                out[base + 1] *= kLog2_10;
                out[base + 2] *= kLog2_10;
            }
        }
    }

    return true;
}

void build_hanatos_integrated_lut_cpu(const Spectral::SpectralContext& ctx, const WorkingState& ws, std::vector<float>& out) {
    const int N = ctx.hanSpectra.size;
    const int K = ctx.hanSpectra.numSamples;
    out.clear();
    out.resize(static_cast<size_t>(N) * static_cast<size_t>(N) * 4u, 0.0f);

    const float* lut = ctx.hanSpectra.data.data();
    const float* sB = ws.sensB.linear.data();
    const float* sG = ws.sensG.linear.data();
    const float* sR = ws.sensR.linear.data();
    const size_t stride = static_cast<size_t>(K);
    for (int x = 0; x < N; ++x) {
        for (int y = 0; y < N; ++y) {
            double accB = 0.0;
            double accG = 0.0;
            double accR = 0.0;
            const size_t base = (static_cast<size_t>(x) * static_cast<size_t>(N) + static_cast<size_t>(y)) * stride;
            for (int k = 0; k < K; ++k) {
                const float raw = lut[base + static_cast<size_t>(k)];
                if (!std::isfinite(raw)) {
                    continue;
                }
                const float e = (raw > 0.0f) ? raw : 0.0f;
                if (!std::isfinite(e)) {
                    continue;
                }
                const double e64 = static_cast<double>(e);
                const float sb = sB[k];
                const float sg = sG[k];
                const float sr = sR[k];
                if (std::isfinite(sb)) accB += e64 * static_cast<double>(sb);
                if (std::isfinite(sg)) accG += e64 * static_cast<double>(sg);
                if (std::isfinite(sr)) accR += e64 * static_cast<double>(sr);
            }

            const size_t outBase = (static_cast<size_t>(x) * static_cast<size_t>(N) + static_cast<size_t>(y)) * 4u;
            out[outBase + 0] = static_cast<float>(accR);
            out[outBase + 1] = static_cast<float>(accG);
            out[outBase + 2] = static_cast<float>(accB);
            out[outBase + 3] = 0.0f;
        }
    }
}

bool build_print_preflash_raw(const WorkingState& ws, const Print::Runtime& prt, float outRaw[3], int& outShapeK) {
    return Pipeline::compute_preflash_raw(ws, prt, outRaw, outShapeK);
}
}
}

namespace JuicerCuda {

bool validate_resource_owner_locked(Resources& resources, std::string& outError, bool bindIfUnset) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
    (void)resources;
    (void)bindIfUnset;
    outError = "CUDA is not enabled";
    return false;
#else
    int cur = -1;
    const cudaError_t devErr = cudaGetDevice(&cur);
    if (devErr != cudaSuccess || cur < 0) {
        outError = std::string("cudaGetDevice failed: ")
            + (cudaGetErrorString(devErr) ? cudaGetErrorString(devErr) : "(unknown)");
        return false;
    }
    if (bindIfUnset && resources.deviceId < 0) {
        resources.deviceId = cur;
    }
    if (resources.deviceId != cur) {
        outError = "CUDA device mismatch for cached resources";
        return false;
    }
    return true;
#endif
}

static bool enqueue_host_to_device_copy(
    const char* stage,
    const char* label,
    void* dst,
    const void* src,
    std::size_t bytes,
    void* cudaStreamOpaque,
    std::string& outError);

static void free_stbn(Resources& resources) noexcept;
static void free_wang(Resources& resources) noexcept;
static void free_scan_error_flag(Resources& resources) noexcept;
static void free_tables(Resources& resources) noexcept;
static bool retire_tables_locked(Resources& resources, void* cudaStreamOpaque, const char* label, std::string& outError);
static void free_scan_medium(Resources::DeviceScanMedium& m) noexcept;
static bool retire_scan_medium_locked(Resources& resources, Resources::DeviceScanMedium& m, void* cudaStreamOpaque, const char* label, std::string& outError);
static void free_scan_lut(Resources::DeviceSpectralLut& lut) noexcept;
static void free_gaussian_kernel(Resources::DeviceGaussianKernel& k) noexcept;
static bool is_async_device_ptr_tracked_locked(const Resources& resources, const void* ptr) noexcept;
static void untrack_async_device_ptr_locked(Resources& resources, void* ptr) noexcept;
static void free_auto_exposure(Resources& resources) noexcept;
static void free_optics_scratch(Resources& resources, Resources::DeviceOpticsScratch& s, void* cudaStreamOpaque) noexcept;
static void free_spatial_dir_scratch(Resources& resources, Resources::DeviceSpatialDirScratch& s, void* cudaStreamOpaque) noexcept;
static void free_shared_tmp_plane(Resources& resources) noexcept;

} // namespace JuicerCuda

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__) && defined(JUICER_CUDA_VALIDATE_PRIMITIVES) && (JUICER_CUDA_VALIDATE_PRIMITIVES != 0)
extern "C" cudaError_t juicer_cuda_probe_density_curve(
    const float* dX,
    const float* dY,
    int n,
    float gammaFactor,
    const float* hLogE,
    int m,
    float* hOut,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_probe_density_curve_sanitize_inf(
    const float* dX,
    const float* dY,
    int n,
    float gammaFactor,
    const float* hLogE,
    int m,
    float* hOut,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_probe_film_log_raw(
    const float filmRaw3[3],
    float outLogRaw3[3],
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_probe_hanatos_layer_exposures(
    const float rgbDWG[3],
    const float* dHanatosLut,
    int hanatosN,
    const float refIllumWhiteXYZ[3],
    const float* dSensB,
    const float* dSensG,
    const float* dSensR,
    int K,
    float exposureScale,
    int applyDeltaLambda,
    float* outE3,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_probe_convert_input_to_DWG(
    const float rgbIn[3],
    int inputColorSpaceIndex,
    int applyCctfDecoding,
    int applyInputChromaticAdapt,
    const float inputRGBToXYZ9[9],
    const float inputXYZAdapt9[9],
    float outRgbDWG[3],
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_probe_tables_layer_exposures(
    const float rgbDWG[3],
    const float S_inv9[9],
    const float refIllumWhiteXYZ[3],
    const float* dAx,
    const float* dAy,
    const float* dAz,
    int K,
    const float* dSensB,
    const float* dSensG,
    const float* dSensR,
    float exposureScale,
    int applyDeltaLambda,
    float* outE3,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_probe_scan_spectral_to_log_xyz(
    const double D_norm3[3],
    int mediumIsNegative,
    const float min_cmy[3],
    const float inv_max_cmy[3],
    const float* dEpsC,
    const float* dEpsM,
    const float* dEpsY,
    const float* dAx,
    const float* dAy,
    const float* dAz,
    const float* dBaseMin,
    int K,
    int hasBaseline,
    float invYn,
    double outLogXYZ3[3],
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_probe_clamp_logE_to_curve_domain(
    const float* dX,
    int n,
    float logE,
    float* outLogE,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_probe_print_pipeline(
    const float* hNegCmy,
    int count,
    const JuicerCuda::PipelineRunParams* hParams,
    float* hOutPrintCmy,
    void* cudaStreamOpaque);
#endif


namespace JuicerCuda {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    struct DeferredDestroyEntry {
        Resources* resources = nullptr;
        int ownerDeviceId = -1;
        void* ownerContextOpaque = nullptr;
    };

    static std::mutex& deferred_destroy_mutex() {
        static std::mutex mutex;
        return mutex;
    }

    static std::vector<DeferredDestroyEntry>& deferred_destroy_queue() {
        static std::vector<DeferredDestroyEntry> queue;
        return queue;
    }

#if defined(_WIN32)
    using CuCtxGetCurrentFn = CUresult(CUDAAPI*)(CUcontext*);

    struct CudaDriverDispatch {
        CuCtxGetCurrentFn cuCtxGetCurrent = nullptr;
        const char* loadError = nullptr;
    };

    static const CudaDriverDispatch& cuda_driver_dispatch_for_teardown() {
        static CudaDriverDispatch dispatch{};
        static std::once_flag once;
        std::call_once(once, []() {
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
            }
        });
        return dispatch;
    }
#endif

    static bool query_current_cuda_context(void*& outContextOpaque, std::string& outError) {
        outContextOpaque = nullptr;
        outError.clear();
#if defined(_WIN32)
        const CudaDriverDispatch& dispatch = cuda_driver_dispatch_for_teardown();
        if (!dispatch.cuCtxGetCurrent) {
            outError = dispatch.loadError ? dispatch.loadError : "driver dispatch unavailable";
            return false;
        }
        CUcontext currentContext = nullptr;
        const CUresult result = dispatch.cuCtxGetCurrent(&currentContext);
        if (result != CUDA_SUCCESS) {
            outError = std::string("cuCtxGetCurrent failed (code=")
                + std::to_string(static_cast<int>(result)) + ")";
            return false;
        }
        if (!currentContext) {
            outError = "current CUDA context is null";
            return false;
        }
        outContextOpaque = reinterpret_cast<void*>(currentContext);
        return true;
#else
        outError = "cuCtxGetCurrent loader unsupported on this platform";
        return false;
#endif
    }

    static bool query_current_cuda_device(int& outDeviceId, std::string& outError) {
        outDeviceId = -1;
        outError.clear();
        const cudaError_t err = cudaGetDevice(&outDeviceId);
        if (err != cudaSuccess || outDeviceId < 0) {
            outError = std::string("cudaGetDevice failed: ")
                + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            outDeviceId = -1;
            return false;
        }
        return true;
    }

    static bool owner_matches_current(
        int ownerDeviceId,
        void* ownerContextOpaque,
        int currentDeviceId,
        bool currentDeviceValid,
        void* currentContextOpaque,
        bool currentContextValid,
        bool& outDeviceMatch,
        bool& outContextMatch) {
        outDeviceMatch = (ownerDeviceId < 0) ||
            (currentDeviceValid && currentDeviceId == ownerDeviceId);
        outContextMatch = (ownerContextOpaque == nullptr) ||
            (currentContextValid && currentContextOpaque == ownerContextOpaque);
        return outDeviceMatch && outContextMatch;
    }

    static void trace_teardown_event(
        const char* stage,
        const char* outcome,
        int ownerDeviceId,
        void* ownerContextOpaque,
        int currentDeviceId,
        void* currentContextOpaque,
        bool deviceMatch,
        bool contextMatch,
        bool switchAttempted,
        bool switchSucceeded,
        bool managerRetireAttempted,
        bool managerRetireAccepted,
        std::size_t deferredQueueDepth,
        const std::string& detail) {
        if (!JTRACE_ENABLED(2)) {
            return;
        }
        const std::uintptr_t ownerContextBits =
            reinterpret_cast<std::uintptr_t>(ownerContextOpaque);
        const std::uintptr_t currentContextBits =
            reinterpret_cast<std::uintptr_t>(currentContextOpaque);
        std::ostringstream oss;
        oss << "stage=" << (stage ? stage : "unknown")
            << " outcome=" << (outcome ? outcome : "unknown")
            << " owner_device=" << ownerDeviceId
            << " current_device=" << currentDeviceId
            << " owner_context=" << ownerContextBits
            << " current_context=" << currentContextBits
            << " device_match=" << (deviceMatch ? 1 : 0)
            << " context_match=" << (contextMatch ? 1 : 0)
            << " switch_attempted=" << (switchAttempted ? 1 : 0)
            << " switch_succeeded=" << (switchSucceeded ? 1 : 0)
            << " retire_attempted=" << (managerRetireAttempted ? 1 : 0)
            << " retire_accepted=" << (managerRetireAccepted ? 1 : 0)
            << " deferred_queue_depth=" << static_cast<unsigned long long>(deferredQueueDepth);
        if (!detail.empty()) {
            oss << " detail=" << detail;
        }
        JTRACE("MSTDN", oss.str());
    }

    static void reap_deferred_destroy_queue(const char* stage) {
        std::vector<DeferredDestroyEntry> readyEntries;
        std::size_t remainingDepth = 0;
        int currentDeviceId = -1;
        std::string deviceError;
        const bool currentDeviceValid = query_current_cuda_device(currentDeviceId, deviceError);
        void* currentContextOpaque = nullptr;
        std::string contextError;
        const bool currentContextValid = query_current_cuda_context(currentContextOpaque, contextError);

        {
            std::lock_guard<std::mutex> lock(deferred_destroy_mutex());
            auto& queue = deferred_destroy_queue();
            if (queue.empty()) {
                return;
            }
            for (std::size_t i = 0; i < queue.size();) {
                bool deviceMatch = false;
                bool contextMatch = false;
                if (owner_matches_current(
                        queue[i].ownerDeviceId,
                        queue[i].ownerContextOpaque,
                        currentDeviceId,
                        currentDeviceValid,
                        currentContextOpaque,
                        currentContextValid,
                        deviceMatch,
                        contextMatch)) {
                    readyEntries.push_back(queue[i]);
                    queue[i] = queue.back();
                    queue.pop_back();
                    continue;
                }
                ++i;
            }
            remainingDepth = queue.size();
        }

        for (const DeferredDestroyEntry& entry : readyEntries) {
            bool deviceMatch = false;
            bool contextMatch = false;
            (void)owner_matches_current(
                entry.ownerDeviceId,
                entry.ownerContextOpaque,
                currentDeviceId,
                currentDeviceValid,
                currentContextOpaque,
                currentContextValid,
                deviceMatch,
                contextMatch);
            std::string detail;
            if (!deviceError.empty()) {
                detail = deviceError;
            }
            if (!contextError.empty()) {
                if (!detail.empty()) {
                    detail += "; ";
                }
                detail += contextError;
            }
            trace_teardown_event(
                stage,
                "deferred_reap",
                entry.ownerDeviceId,
                entry.ownerContextOpaque,
                currentDeviceId,
                currentContextOpaque,
                deviceMatch,
                contextMatch,
                false,
                false,
                false,
                false,
                remainingDepth,
                detail);
            delete entry.resources;
        }
    }
#endif


    static bool add_u64_saturating(std::uint64_t lhs, std::uint64_t rhs, std::uint64_t& out) noexcept {
        if (lhs > (std::numeric_limits<std::uint64_t>::max() - rhs)) {
            out = std::numeric_limits<std::uint64_t>::max();
            return false;
        }
        out = lhs + rhs;
        return true;
    }

    static std::uint64_t bytes_for_count_u64(
        std::size_t count,
        std::size_t elementBytes,
        bool& overflow) noexcept {
        if (overflow) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        if (count == 0 || elementBytes == 0) {
            return 0;
        }

        const std::uint64_t count64 = static_cast<std::uint64_t>(count);
        const std::uint64_t elementBytes64 = static_cast<std::uint64_t>(elementBytes);
        if (count64 > (std::numeric_limits<std::uint64_t>::max() / elementBytes64)) {
            overflow = true;
            return std::numeric_limits<std::uint64_t>::max();
        }
        return count64 * elementBytes64;
    }

    static void add_residency_bytes(
        std::uint64_t bytes,
        std::uint64_t& total,
        bool& overflow) noexcept {
        std::uint64_t next = 0;
        if (!add_u64_saturating(total, bytes, next)) {
            overflow = true;
        }
        total = next;
    }

    static void refresh_scratch_residency_state_locked(Resources& resources) noexcept {
        Resources::ScratchResidencyState next{};
        next.retainedGeneration = std::max<std::uint64_t>(1ull, resources.scratchResidency.retainedGeneration);

        bool overflow = false;
        const std::uint64_t opticsPlaneBytes =
            bytes_for_count_u64(resources.scannerScratch.capacityElements, sizeof(float), overflow);
        const std::uint64_t spatialDirPlaneBytes =
            bytes_for_count_u64(resources.spatialDirScratch.capacityElements, sizeof(float), overflow);
        const std::uint64_t gateMaskBytes =
            bytes_for_count_u64(resources.scannerScratch.gateMaskCapacityElements, sizeof(float), overflow);
        const std::uint64_t sharedTmpBytes =
            bytes_for_count_u64(resources.sharedTmpCapacityElements, sizeof(float), overflow);
        if (overflow) {
            next.overflow = true;
        }

        auto add_candidate_plane = [&](ResourceManager::ScratchPolicyCandidate candidate, bool live, std::uint64_t bytes) {
            if (!live || bytes == 0) {
                return;
            }
            const std::size_t index = ResourceManager::scratch_policy_candidate_index(candidate);
            add_residency_bytes(bytes, next.candidateLiveBytes[index], next.overflow);
        };

        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::OpticsBase, resources.scannerScratch.rgbR != nullptr, opticsPlaneBytes);
        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::OpticsBase, resources.scannerScratch.rgbG != nullptr, opticsPlaneBytes);
        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::OpticsBase, resources.scannerScratch.rgbB != nullptr, opticsPlaneBytes);
        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::OpticsBlurred, resources.scannerScratch.blurred != nullptr, opticsPlaneBytes);
        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::OpticsAux, resources.scannerScratch.aux != nullptr, opticsPlaneBytes);
        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::OpticsGrainTriplet, resources.scannerScratch.grainTmp != nullptr, opticsPlaneBytes);
        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::OpticsGrainTriplet, resources.scannerScratch.grainTmpMid != nullptr, opticsPlaneBytes);
        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::OpticsGrainTriplet, resources.scannerScratch.grainTmpCoarse != nullptr, opticsPlaneBytes);
        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::OpticsGrainShared, resources.scannerScratch.grainTmpShared != nullptr, opticsPlaneBytes);
        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::OpticsGateMask, resources.scannerScratch.gateMask != nullptr, gateMaskBytes);
        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::SpatialDirBase, resources.spatialDirScratch.corrY != nullptr, spatialDirPlaneBytes);
        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::SpatialDirBase, resources.spatialDirScratch.corrM != nullptr, spatialDirPlaneBytes);
        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::SpatialDirBase, resources.spatialDirScratch.corrC != nullptr, spatialDirPlaneBytes);

        if (resources.sharedTmpPlane) {
            next.helperSharedBytes = sharedTmpBytes;
        }

        auto assign_non_policy_bytes = [&](ResourceManager::ScratchHelperNonPolicyAllocation allocation, bool live, std::uint64_t bytes) {
            if (!live || bytes == 0) {
                return;
            }
            const std::size_t index = ResourceManager::scratch_helper_non_policy_index(allocation);
            next.helperNonPolicyBytes[index] = bytes;
            add_residency_bytes(bytes, next.helperNonPolicyTotalBytes, next.overflow);
        };

        assign_non_policy_bytes(ResourceManager::ScratchHelperNonPolicyAllocation::ScanErrorFlag, resources.scanErrorFlag != nullptr, sizeof(int));
        assign_non_policy_bytes(ResourceManager::ScratchHelperNonPolicyAllocation::ScanErrorHost, resources.scanErrorHost != nullptr, sizeof(int));
        assign_non_policy_bytes(ResourceManager::ScratchHelperNonPolicyAllocation::AutoExposureExposureScale, resources.autoExposureExposureScale != nullptr, sizeof(float));
        assign_non_policy_bytes(ResourceManager::ScratchHelperNonPolicyAllocation::AutoExposureAutoEV, resources.autoExposureAutoEV != nullptr, sizeof(double));
        assign_non_policy_bytes(ResourceManager::ScratchHelperNonPolicyAllocation::AutoExposureValid, resources.autoExposureValid != nullptr, sizeof(int));
        assign_non_policy_bytes(ResourceManager::ScratchHelperNonPolicyAllocation::AutoExposureMaxYBits, resources.autoExposureScratch.maxYBits != nullptr, sizeof(unsigned int));
        assign_non_policy_bytes(
            ResourceManager::ScratchHelperNonPolicyAllocation::AutoExposureHistogram,
            resources.autoExposureScratch.histogram != nullptr,
            static_cast<std::uint64_t>(2048u) * sizeof(unsigned int));
        assign_non_policy_bytes(
            ResourceManager::ScratchHelperNonPolicyAllocation::AutoExposureWeightsX,
            resources.autoExposureScratch.weightsX != nullptr,
            bytes_for_count_u64(static_cast<std::size_t>(std::max(0, resources.autoExposureScratch.weightsXCapacity)), sizeof(float), next.overflow));
        assign_non_policy_bytes(
            ResourceManager::ScratchHelperNonPolicyAllocation::AutoExposureWeightsY,
            resources.autoExposureScratch.weightsY != nullptr,
            bytes_for_count_u64(static_cast<std::size_t>(std::max(0, resources.autoExposureScratch.weightsYCapacity)), sizeof(float), next.overflow));
        assign_non_policy_bytes(
            ResourceManager::ScratchHelperNonPolicyAllocation::AutoExposurePartialsA,
            resources.autoExposureScratch.partialsA != nullptr,
            bytes_for_count_u64(
                static_cast<std::size_t>(std::max(0, resources.autoExposureScratch.partialCapacity)),
                sizeof(JuicerCudaAutoExposurePartial),
                next.overflow));
        assign_non_policy_bytes(
            ResourceManager::ScratchHelperNonPolicyAllocation::AutoExposurePartialsB,
            resources.autoExposureScratch.partialsB != nullptr,
            bytes_for_count_u64(
                static_cast<std::size_t>(std::max(0, resources.autoExposureScratch.partialCapacity)),
                sizeof(JuicerCudaAutoExposurePartial),
                next.overflow));

        for (std::uint64_t bytes : next.candidateLiveBytes) {
            add_residency_bytes(bytes, next.policyLiveRetainedBytes, next.overflow);
        }
        add_residency_bytes(next.helperSharedBytes, next.policyLiveRetainedBytes, next.overflow);
        next.totalLiveRetainedBytes = next.policyLiveRetainedBytes;
        add_residency_bytes(next.helperNonPolicyTotalBytes, next.totalLiveRetainedBytes, next.overflow);

        const bool changed =
            next.candidateLiveBytes != resources.scratchResidency.candidateLiveBytes ||
            next.helperNonPolicyBytes != resources.scratchResidency.helperNonPolicyBytes ||
            next.helperSharedBytes != resources.scratchResidency.helperSharedBytes ||
            next.helperNonPolicyTotalBytes != resources.scratchResidency.helperNonPolicyTotalBytes ||
            next.policyLiveRetainedBytes != resources.scratchResidency.policyLiveRetainedBytes ||
            next.totalLiveRetainedBytes != resources.scratchResidency.totalLiveRetainedBytes ||
            next.overflow != resources.scratchResidency.overflow;
        if (changed) {
            next.retainedGeneration =
                (resources.scratchResidency.retainedGeneration == std::numeric_limits<std::uint64_t>::max())
                ? std::numeric_limits<std::uint64_t>::max()
                : std::max<std::uint64_t>(1ull, resources.scratchResidency.retainedGeneration + 1ull);
        }

        resources.scratchResidency = next;
    }

    static void reap_retire_queue_locked(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        for (size_t i = 0; i < resources.retireQueue.size();) {
            Resources::RetireEntry& e = resources.retireQueue[i];
            cudaEvent_t ev = reinterpret_cast<cudaEvent_t>(e.doneEventOpaque);
            if (!ev) {
                // No fence: best-effort free immediately.
                if (e.kind == Resources::RetireKind::DeviceFree && e.ptr) {
                    cudaFree(e.ptr);
                }
                else if (e.kind == Resources::RetireKind::DeviceFreeAsync && e.ptr) {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
                    const cudaError_t asyncErr = cudaFreeAsync(e.ptr, nullptr);
                    if (asyncErr != cudaSuccess) {
                        cudaFree(e.ptr);
                    }
#else
                    cudaFree(e.ptr);
#endif
                }
                else if (e.kind == Resources::RetireKind::HostPinnedFree && e.ptr) {
                    cudaFreeHost(e.ptr);
                }
                else if (e.kind == Resources::RetireKind::EventDestroy && e.ptr) {
                    cudaEventDestroy(reinterpret_cast<cudaEvent_t>(e.ptr));
                }
                if (resources.retireBytes >= e.bytes) {
                    resources.retireBytes -= e.bytes;
                }
                if (e.scratchTier && resources.retireScratchBytes >= e.bytes) {
                    resources.retireScratchBytes -= e.bytes;
                }
                resources.retireQueue[i] = resources.retireQueue.back();
                resources.retireQueue.pop_back();
                continue;
            }

            const cudaError_t q = cudaEventQuery(ev);
            if (q == cudaSuccess) {
                if (e.kind == Resources::RetireKind::DeviceFree && e.ptr) {
                    cudaFree(e.ptr);
                }
                else if (e.kind == Resources::RetireKind::DeviceFreeAsync && e.ptr) {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
                    const cudaError_t asyncErr = cudaFreeAsync(e.ptr, nullptr);
                    if (asyncErr != cudaSuccess) {
                        cudaFree(e.ptr);
                    }
#else
                    cudaFree(e.ptr);
#endif
                }
                else if (e.kind == Resources::RetireKind::HostPinnedFree && e.ptr) {
                    cudaFreeHost(e.ptr);
                }
                else if (e.kind == Resources::RetireKind::EventDestroy && e.ptr) {
                    cudaEventDestroy(reinterpret_cast<cudaEvent_t>(e.ptr));
                }

                // Return the fence to the pool.
                resources.retireEventPoolOpaque.push_back(e.doneEventOpaque);
                if (resources.retireBytes >= e.bytes) {
                    resources.retireBytes -= e.bytes;
                }
                if (e.scratchTier && resources.retireScratchBytes >= e.bytes) {
                    resources.retireScratchBytes -= e.bytes;
                }
                resources.retireQueue[i] = resources.retireQueue.back();
                resources.retireQueue.pop_back();
                continue;
            }
            if (q == cudaErrorNotReady) {
                ++i;
                continue;
            }

            // On unexpected CUDA errors, keep the entry so we don't free too early.
            ++i;
        }
#else
        (void)resources;
#endif
    }

    static void drain_retire_queue_blocking(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        for (Resources::RetireEntry& e : resources.retireQueue) {
            cudaEvent_t ev = reinterpret_cast<cudaEvent_t>(e.doneEventOpaque);
            if (ev) {
                (void)cudaEventSynchronize(ev);
            }
            if (e.kind == Resources::RetireKind::DeviceFree && e.ptr) {
                cudaFree(e.ptr);
            }
            else if (e.kind == Resources::RetireKind::DeviceFreeAsync && e.ptr) {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
                const cudaError_t asyncErr = cudaFreeAsync(e.ptr, nullptr);
                if (asyncErr == cudaSuccess) {
                    (void)cudaStreamSynchronize(nullptr);
                }
                else {
                    cudaFree(e.ptr);
                }
#else
                cudaFree(e.ptr);
#endif
            }
            else if (e.kind == Resources::RetireKind::HostPinnedFree && e.ptr) {
                cudaFreeHost(e.ptr);
            }
            else if (e.kind == Resources::RetireKind::EventDestroy && e.ptr) {
                cudaEventDestroy(reinterpret_cast<cudaEvent_t>(e.ptr));
            }
            if (ev) {
                resources.retireEventPoolOpaque.push_back(e.doneEventOpaque);
            }
        }
        resources.retireQueue.clear();
        resources.retireBytes = 0;
        resources.retireScratchBytes = 0;

        // Destroy pooled events.
        for (void* p : resources.retireEventPoolOpaque) {
            cudaEvent_t ev = reinterpret_cast<cudaEvent_t>(p);
            if (ev) {
                cudaEventDestroy(ev);
            }
        }
        resources.retireEventPoolOpaque.clear();
#else
        (void)resources;
#endif
    }

    static bool acquire_retire_event_locked(Resources& resources, void*& outEventOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)outEventOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        if (!resources.retireEventPoolOpaque.empty()) {
            outEventOpaque = resources.retireEventPoolOpaque.back();
            resources.retireEventPoolOpaque.pop_back();
            return true;
        }
        cudaEvent_t ev = nullptr;
        const cudaError_t err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        if (err != cudaSuccess || !ev) {
            outError = std::string("cudaEventCreateWithFlags failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            outEventOpaque = nullptr;
            return false;
        }
        outEventOpaque = reinterpret_cast<void*>(ev);
        return true;
#endif
    }

    static bool record_retire_fence_locked(Resources& resources, void* retireEventOpaque, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)retireEventOpaque;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        cudaEvent_t retireEv = reinterpret_cast<cudaEvent_t>(retireEventOpaque);
        if (!retireEv) {
            outError = "retire fence event missing";
            return false;
        }
        const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
        if (resources.lastUseEventOpaque) {
            const cudaEvent_t lastUseEv = reinterpret_cast<cudaEvent_t>(resources.lastUseEventOpaque);
            const cudaError_t waitErr = cudaStreamWaitEvent(stream, lastUseEv, 0);
            if (waitErr != cudaSuccess) {
                outError = std::string("cudaStreamWaitEvent before ") + label + " retire failed: " +
                    (cudaGetErrorString(waitErr) ? cudaGetErrorString(waitErr) : "(unknown)");
                return false;
            }
        }
        const cudaError_t recErr = cudaEventRecord(retireEv, stream);
        if (recErr != cudaSuccess) {
            outError = std::string("cudaEventRecord for ") + label + " retire failed: " +
                (cudaGetErrorString(recErr) ? cudaGetErrorString(recErr) : "(unknown)");
            return false;
        }
        return true;
#endif
    }

    static bool retire_ptr_locked(Resources& resources, void* ptr, std::size_t bytes, Resources::RetireKind kind, void* cudaStreamOpaque, const char* label, std::string& outError, bool scratchTier = false) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)ptr;
        (void)bytes;
        (void)kind;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (!ptr) {
            return true;
        }

        bool asyncTracked = false;
        Resources::RetireKind effectiveKind = kind;
        if (kind == Resources::RetireKind::DeviceFree) {
            asyncTracked = is_async_device_ptr_tracked_locked(resources, ptr);
            if (asyncTracked) {
                effectiveKind = Resources::RetireKind::DeviceFreeAsync;
            }
        }

        reap_retire_queue_locked(resources);

        void* retireEventOpaque = nullptr;
        if (!acquire_retire_event_locked(resources, retireEventOpaque, outError)) {
            if (outError.empty()) {
                outError = std::string("retire fence acquisition failed for ") + (label ? label : "resource");
            }
            return false;
        }

        if (!record_retire_fence_locked(resources, retireEventOpaque, cudaStreamOpaque, label, outError)) {
            // If we can't record the retire fence, fail closed instead of falling back to a blocking sync.
            cudaEventDestroy(reinterpret_cast<cudaEvent_t>(retireEventOpaque));
            if (outError.empty()) {
                outError = std::string("retire fence record failed for ") + (label ? label : "resource");
            }
            return false;
        }

        Resources::RetireEntry e{};
        e.ptr = ptr;
        e.bytes = bytes;
        e.kind = effectiveKind;
        e.scratchTier = scratchTier;
        e.doneEventOpaque = retireEventOpaque;
        resources.retireQueue.push_back(e);
        resources.retireBytes += bytes;
        if (scratchTier) {
            resources.retireScratchBytes += bytes;
        }
        if (effectiveKind == Resources::RetireKind::DeviceFreeAsync && asyncTracked) {
            untrack_async_device_ptr_locked(resources, ptr);
        }
        return true;
#endif
    }

    static void free_curve(DeviceCurve& c) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (c.x) {
            cudaFree(c.x);
            c.x = nullptr;
        }
        if (c.y) {
            cudaFree(c.y);
            c.y = nullptr;
        }
#endif
        c.n = 0;
        c.domainBegin = 0;
        c.domainEnd = 0;
    }

    static bool retire_curve_locked(Resources& resources, DeviceCurve& c, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)c;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        const size_t bytes = static_cast<size_t>(std::max(0, c.n)) * sizeof(float);
        if (c.x) {
            if (!retire_ptr_locked(resources, c.x, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                return false;
            }
            c.x = nullptr;
        }
        if (c.y) {
            if (!retire_ptr_locked(resources, c.y, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                return false;
            }
            c.y = nullptr;
        }
        c.n = 0;
        c.domainBegin = 0;
        c.domainEnd = 0;
        return true;
#endif
    }

    static void free_density_layers(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        for (int layer = 0; layer < 3; ++layer) {
            for (int ch = 0; ch < 3; ++ch) {
                if (resources.densityCurvesLayers[layer][ch]) {
                    cudaFree(resources.densityCurvesLayers[layer][ch]);
                    resources.densityCurvesLayers[layer][ch] = nullptr;
                }
            }
        }
#endif
        for (int ch = 0; ch < 3; ++ch) {
            resources.densityCurvesLayersChannelN[ch] = 0;
        }
        resources.hasDensityCurvesLayers = 0;
    }

    static bool retire_density_layers_locked(Resources& resources, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        for (int ch = 0; ch < 3; ++ch) {
            const size_t bytes =
                static_cast<size_t>(std::max(0, resources.densityCurvesLayersChannelN[ch])) *
                sizeof(float);
            for (int layer = 0; layer < 3; ++layer) {
                if (resources.densityCurvesLayers[layer][ch]) {
                    if (!retire_ptr_locked(resources, resources.densityCurvesLayers[layer][ch], bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                        return false;
                    }
                    resources.densityCurvesLayers[layer][ch] = nullptr;
                }
            }
            resources.densityCurvesLayersChannelN[ch] = 0;
        }
        resources.hasDensityCurvesLayers = 0;
        return true;
#endif
    }

    static void free_hanatos(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (resources.hanatosLut) {
            cudaFree(resources.hanatosLut);
            resources.hanatosLut = nullptr;
        }
#endif
        resources.hanatosN = 0;
    }

    static void free_hanatos_integrated(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (resources.hanatosLutIntegrated) {
            cudaFree(resources.hanatosLutIntegrated);
            resources.hanatosLutIntegrated = nullptr;
        }
#endif
        resources.hanatosNIntegrated = 0;
        resources.hanatosIntegratedBuildCounter = 0;
    }

    static void free_mallett_basis(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (resources.mallettBasis) {
            cudaFree(resources.mallettBasis);
            resources.mallettBasis = nullptr;
        }
#endif
        resources.mallettBasisK = 0;
    }


    static void free_print_payloads(Resources& resources) noexcept {
        free_curve(resources.printDcC);
        free_curve(resources.printDcM);
        free_curve(resources.printDcY);
        free_curve(resources.printSensC);
        free_curve(resources.printSensM);
        free_curve(resources.printSensY);

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (resources.printIllumFiltered) {
            cudaFree(resources.printIllumFiltered);
            resources.printIllumFiltered = nullptr;
        }
#endif
        resources.printIllumK = 0;
        resources.printIllumYShiftSteps = 0.0f;
        resources.printIllumMShiftSteps = 0.0f;
        resources.printIllumCShiftSteps = 0.0f;
        resources.printIllumNeutralFilterHash = 0;
        resources.printIllumShapeK = 0;
        resources.printIllumBuildCounter = 0;
        resources.printIllumCoreHash = 0;

        resources.printGammaC = 1.0f;
        resources.printGammaM = 1.0f;
        resources.printGammaY = 1.0f;

        resources.printPreflashRaw[0] = resources.printPreflashRaw[1] = resources.printPreflashRaw[2] = 0.0f;
        resources.printPreflashValid = false;
        resources.printPreflashBuildCounter = 0;
        resources.printPreflashShapeK = 0;
    }

    static bool retire_print_payloads_locked(Resources& resources, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (!retire_curve_locked(resources, resources.printDcC, cudaStreamOpaque, label, outError)) return false;
        if (!retire_curve_locked(resources, resources.printDcM, cudaStreamOpaque, label, outError)) return false;
        if (!retire_curve_locked(resources, resources.printDcY, cudaStreamOpaque, label, outError)) return false;
        if (!retire_curve_locked(resources, resources.printSensC, cudaStreamOpaque, label, outError)) return false;
        if (!retire_curve_locked(resources, resources.printSensM, cudaStreamOpaque, label, outError)) return false;
        if (!retire_curve_locked(resources, resources.printSensY, cudaStreamOpaque, label, outError)) return false;

        if (resources.printIllumFiltered) {
            const size_t bytes = static_cast<size_t>(std::max(0, resources.printIllumK)) * sizeof(float);
            if (!retire_ptr_locked(resources, resources.printIllumFiltered, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                return false;
            }
            resources.printIllumFiltered = nullptr;
        }

        resources.printIllumK = 0;
        resources.printIllumYShiftSteps = 0.0f;
        resources.printIllumMShiftSteps = 0.0f;
        resources.printIllumCShiftSteps = 0.0f;
        resources.printIllumNeutralFilterHash = 0;
        resources.printIllumShapeK = 0;
        resources.printIllumBuildCounter = 0;
        resources.printIllumCoreHash = 0;

        resources.printGammaC = 1.0f;
        resources.printGammaM = 1.0f;
        resources.printGammaY = 1.0f;

        resources.printPreflashRaw[0] = resources.printPreflashRaw[1] = resources.printPreflashRaw[2] = 0.0f;
        resources.printPreflashValid = false;
        resources.printPreflashBuildCounter = 0;
        resources.printPreflashShapeK = 0;
        return true;
#endif
    }

    static bool alloc_and_upload_array(float*& dst, const float* src, int n, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)dst;
        (void)src;
        (void)n;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (!src || n <= 0) {
            outError = std::string(label) + " array is empty";
            return false;
        }
        const size_t bytes = static_cast<size_t>(n) * sizeof(float);
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&dst), bytes);
        if (err != cudaSuccess) {
            outError = std::string("cudaMalloc(") + label + ") failed: " + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            return false;
        }
        if (!enqueue_host_to_device_copy(
                "alloc_and_upload_array",
                label,
                dst,
                src,
                bytes,
                cudaStreamOpaque,
                outError)) {
            cudaFree(dst);
            dst = nullptr;
            return false;
        }
        return true;
#endif
    }

    static bool relock_resources_after_offlock_upload(
        Resources& resources,
        std::unique_lock<std::mutex>* resourcesLock,
        std::string& outError) {
        if (!resourcesLock) {
            return true;
        }
        resourcesLock->lock();
        reap_retire_queue_locked(resources);
        return validate_resource_owner_locked(resources, outError, false);
    }

    static bool wait_for_last_use_event_snapshot(
        void* lastUseEventOpaque,
        void* cudaStreamOpaque,
        const char* label,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)lastUseEventOpaque;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (!lastUseEventOpaque) {
            return true;
        }
        const cudaStream_t stream = cudaStreamOpaque
            ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque)
            : nullptr;
        const cudaEvent_t lastUseEv = reinterpret_cast<cudaEvent_t>(lastUseEventOpaque);
        const cudaError_t waitErr = cudaStreamWaitEvent(stream, lastUseEv, 0);
        if (waitErr != cudaSuccess) {
            outError = std::string("cudaStreamWaitEvent before ")
                + (label ? label : "resource")
                + " update failed: "
                + (cudaGetErrorString(waitErr) ? cudaGetErrorString(waitErr) : "(unknown)");
            return false;
        }
        return true;
#endif
    }

    static bool alloc_and_upload_bytes(
        void*& dst,
        const void* src,
        std::size_t bytes,
        void* cudaStreamOpaque,
        const char* label,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)dst;
        (void)src;
        (void)bytes;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (!src || bytes == 0) {
            outError = std::string(label ? label : "buffer") + " buffer is empty";
            return false;
        }

        const cudaError_t err = cudaMalloc(&dst, bytes);
        if (err != cudaSuccess) {
            outError = std::string("cudaMalloc(")
                + (label ? label : "buffer")
                + ") failed: "
                + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            return false;
        }

        if (!enqueue_host_to_device_copy(
                "alloc_and_upload_bytes",
                label,
                dst,
                src,
                bytes,
                cudaStreamOpaque,
                outError)) {
            cudaFree(dst);
            dst = nullptr;
            return false;
        }

        return true;
#endif
    }

    static bool alloc_and_upload_bytes_locked(
        Resources& resources,
        void*& dst,
        const void* src,
        std::size_t bytes,
        void* cudaStreamOpaque,
        std::unique_lock<std::mutex>* resourcesLock,
        const char* label,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)dst;
        (void)src;
        (void)bytes;
        (void)cudaStreamOpaque;
        (void)resourcesLock;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (resourcesLock) {
            resourcesLock->unlock();
        }
        const bool allocOk =
            alloc_and_upload_bytes(dst, src, bytes, cudaStreamOpaque, label, outError);
        if (!relock_resources_after_offlock_upload(resources, resourcesLock, outError)) {
            if (dst) {
                cudaFree(dst);
                dst = nullptr;
            }
            return false;
        }
        return allocOk;
#endif
    }

    static bool upload_array_locked(
        Resources& resources,
        float*& dst,
        int currentN,
        const float* src,
        int n,
        void* cudaStreamOpaque,
        std::unique_lock<std::mutex>* resourcesLock,
        const char* label,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)dst;
        (void)currentN;
        (void)src;
        (void)n;
        (void)cudaStreamOpaque;
        (void)resourcesLock;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (!src || n <= 0) {
            outError = std::string(label ? label : "array") + " array is empty";
            return false;
        }

        const size_t bytes = static_cast<size_t>(n) * sizeof(float);
        if (dst && currentN == n) {
            const bool waitOk = wait_for_last_use_event_snapshot(
                resources.lastUseEventOpaque,
                cudaStreamOpaque,
                label,
                outError);
            const bool copyOk = waitOk && enqueue_host_to_device_copy(
                "upload_array_locked",
                label,
                    dst,
                    src,
                    bytes,
                    cudaStreamOpaque,
                    outError);
            return copyOk;
        }

        float* tmp = nullptr;
        if (resourcesLock) {
            resourcesLock->unlock();
        }
        const bool allocOk =
            alloc_and_upload_array(tmp, src, n, cudaStreamOpaque, label, outError);
        if (!relock_resources_after_offlock_upload(resources, resourcesLock, outError)) {
            if (tmp) {
                cudaFree(tmp);
            }
            return false;
        }
        if (!allocOk) {
            return false;
        }

        const size_t oldBytes = static_cast<size_t>(std::max(0, currentN)) * sizeof(float);
        if (dst) {
            if (!retire_ptr_locked(resources, dst, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                cudaFree(tmp);
                return false;
            }
        }

        dst = tmp;
        return true;
#endif
    }

    static bool alloc_and_upload_curve(DeviceCurve& dst, const Spectral::Curve& src, void* cudaStreamOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)dst;
        (void)src;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        if (src.lambda_nm.empty() || src.linear.empty() || src.lambda_nm.size() != src.linear.size()) {
            outError = "curve has no samples or mismatched arrays";
            return false;
        }

        const int n = static_cast<int>(src.lambda_nm.size());
        if (n <= 0) {
            outError = "curve sample count invalid";
            return false;
        }

        int domainBegin = 0;
        while (domainBegin < n && !std::isfinite(src.lambda_nm[static_cast<size_t>(domainBegin)])) {
            ++domainBegin;
        }
        int domainEnd = n - 1;
        while (domainEnd > domainBegin && !std::isfinite(src.lambda_nm[static_cast<size_t>(domainEnd)])) {
            --domainEnd;
        }
        dst.domainBegin = domainBegin;
        dst.domainEnd = domainEnd;

        const size_t bytes = static_cast<size_t>(n) * sizeof(float);
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&dst.x), bytes);
        if (err != cudaSuccess) {
            outError = std::string("cudaMalloc(curve.x) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            free_curve(dst);
            return false;
        }
        err = cudaMalloc(reinterpret_cast<void**>(&dst.y), bytes);
        if (err != cudaSuccess) {
            outError = std::string("cudaMalloc(curve.y) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            free_curve(dst);
            return false;
        }

        if (!enqueue_host_to_device_copy(
                "alloc_and_upload_curve",
                "curve.x",
                dst.x,
                src.lambda_nm.data(),
                bytes,
                cudaStreamOpaque,
                outError)) {
            outError = std::string("curve.x upload failed: ") + outError;
            free_curve(dst);
            return false;
        }
        if (!enqueue_host_to_device_copy(
                "alloc_and_upload_curve",
                "curve.y",
                dst.y,
                src.linear.data(),
                bytes,
                cudaStreamOpaque,
                outError)) {
            outError = std::string("curve.y upload failed: ") + outError;
            free_curve(dst);
            return false;
        }

        dst.n = n;
        return true;
#endif
    }

    static bool upload_curve_locked(
        Resources& resources,
        DeviceCurve& dst,
        const Spectral::Curve& src,
        void* cudaStreamOpaque,
        std::unique_lock<std::mutex>* resourcesLock,
        const char* label,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)dst;
        (void)src;
        (void)cudaStreamOpaque;
        (void)resourcesLock;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (src.lambda_nm.empty() || src.linear.empty() || src.lambda_nm.size() != src.linear.size()) {
            outError = std::string(label) + ": curve has no samples or mismatched arrays";
            return false;
        }
        const int n = static_cast<int>(src.lambda_nm.size());
        if (n <= 0) {
            outError = std::string(label) + ": curve sample count invalid";
            return false;
        }

        const size_t bytes = static_cast<size_t>(n) * sizeof(float);

        // If the allocation matches, update in place to avoid alloc/free churn (common during slider scrubs).
        if (dst.x && dst.y && dst.n == n) {
            const char* baseLabel = label ? label : "curve";
            const std::string labelX = std::string(baseLabel) + ".x";
            int domainBegin = 0;
            while (domainBegin < n && !std::isfinite(src.lambda_nm[static_cast<size_t>(domainBegin)])) {
                ++domainBegin;
            }
            int domainEnd = n - 1;
            while (domainEnd > domainBegin && !std::isfinite(src.lambda_nm[static_cast<size_t>(domainEnd)])) {
                --domainEnd;
            }

            const bool waitOk = wait_for_last_use_event_snapshot(
                resources.lastUseEventOpaque,
                cudaStreamOpaque,
                baseLabel,
                outError);
            const bool copyXOk = waitOk && enqueue_host_to_device_copy(
                    "upload_curve_locked",
                    labelX.c_str(),
                    dst.x,
                    src.lambda_nm.data(),
                    bytes,
                    cudaStreamOpaque,
                    outError);
            const std::string labelY = std::string(baseLabel) + ".y";
            const bool copyYOk = copyXOk && enqueue_host_to_device_copy(
                    "upload_curve_locked",
                    labelY.c_str(),
                    dst.y,
                    src.linear.data(),
                    bytes,
                    cudaStreamOpaque,
                    outError);
            if (!copyXOk) {
                outError = std::string(baseLabel) + ".x upload failed: " + outError;
                return false;
            }
            if (!copyYOk) {
                outError = std::string(baseLabel) + ".y upload failed: " + outError;
                return false;
            }

            dst.domainBegin = domainBegin;
            dst.domainEnd = domainEnd;
            dst.n = n;
            return true;
        }

        DeviceCurve tmp{};
        if (resourcesLock) {
            resourcesLock->unlock();
        }
        const bool allocOk =
            alloc_and_upload_curve(tmp, src, cudaStreamOpaque, outError);
        if (!relock_resources_after_offlock_upload(resources, resourcesLock, outError)) {
            free_curve(tmp);
            return false;
        }
        if (!allocOk) {
            outError = std::string(label) + ": " + outError;
            return false;
        }

        const size_t oldBytes = static_cast<size_t>(std::max(0, dst.n)) * sizeof(float);
        if (dst.x) {
            if (!retire_ptr_locked(resources, dst.x, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                free_curve(tmp);
                return false;
            }
            dst.x = nullptr;
        }
        if (dst.y) {
            if (!retire_ptr_locked(resources, dst.y, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                free_curve(tmp);
                return false;
            }
            dst.y = nullptr;
        }

        dst = tmp;
        return true;
#endif
    }

    static bool alloc_and_upload_spectral_samples(DeviceCurve& dst, const std::vector<float>& src, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)dst;
        (void)src;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (src.empty()) {
            outError = std::string(label) + " array is empty";
            return false;
        }
        const int n = static_cast<int>(src.size());
        if (n <= 0) {
            outError = std::string(label) + " sample count invalid";
            return false;
        }
        const size_t bytes = static_cast<size_t>(n) * sizeof(float);
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&dst.y), bytes);
        if (err != cudaSuccess) {
            outError = std::string("cudaMalloc(") + label + ") failed: " + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            free_curve(dst);
            return false;
        }
        if (!enqueue_host_to_device_copy(
                "alloc_and_upload_spectral_samples",
                label,
                dst.y,
                src.data(),
                bytes,
                cudaStreamOpaque,
                outError)) {
            free_curve(dst);
            return false;
        }
        dst.n = n;
        dst.domainBegin = 0;
        dst.domainEnd = n - 1;
        return true;
#endif
    }

    static bool upload_spectral_samples_locked(
        Resources& resources,
        DeviceCurve& dst,
        const std::vector<float>& src,
        void* cudaStreamOpaque,
        std::unique_lock<std::mutex>* resourcesLock,
        const char* label,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)dst;
        (void)src;
        (void)cudaStreamOpaque;
        (void)resourcesLock;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (src.empty()) {
            outError = std::string(label ? label : "spectral samples") + " array is empty";
            return false;
        }

        const int n = static_cast<int>(src.size());
        if (n <= 0) {
            outError = std::string(label ? label : "spectral samples") + " sample count invalid";
            return false;
        }

        const size_t bytes = static_cast<size_t>(n) * sizeof(float);
        if (dst.y && dst.n == n) {
            const bool waitOk = wait_for_last_use_event_snapshot(
                resources.lastUseEventOpaque,
                cudaStreamOpaque,
                label,
                outError);
            const bool copyOk = waitOk && enqueue_host_to_device_copy(
                    "upload_spectral_samples_locked",
                    label,
                    dst.y,
                    src.data(),
                    bytes,
                    cudaStreamOpaque,
                    outError);
            if (!copyOk) {
                return false;
            }
            dst.n = n;
            dst.domainBegin = 0;
            dst.domainEnd = n - 1;
            return true;
        }

        DeviceCurve tmp{};
        if (resourcesLock) {
            resourcesLock->unlock();
        }
        const bool allocOk =
            alloc_and_upload_spectral_samples(tmp, src, cudaStreamOpaque, label, outError);
        if (!relock_resources_after_offlock_upload(resources, resourcesLock, outError)) {
            free_curve(tmp);
            return false;
        }
        if (!allocOk) {
            return false;
        }

        if (!retire_curve_locked(resources, dst, cudaStreamOpaque, label, outError)) {
            free_curve(tmp);
            return false;
        }

        dst = tmp;
        return true;
#endif
    }

    Resources::~Resources() {
        drain_retire_queue_blocking(*this);
        free_curve(densB);
        free_curve(densG);
        free_curve(densR);
        free_density_layers(*this);
        free_curve(dirDensB);
        free_curve(dirDensG);
        free_curve(dirDensR);
        free_curve(sensB);
        free_curve(sensG);
        free_curve(sensR);
        free_tables(*this);
        free_scan_medium(scanNegative);
        free_scan_medium(scanPrint);
        free_scan_lut(scanNegativeLut);
        free_scan_lut(scanPrintLut);
        free_gaussian_kernel(scannerLensBlurKernel);
        free_gaussian_kernel(scannerUnsharpKernel);
        free_gaussian_kernel(scannerGlareKernel);
        free_gaussian_kernel(grainBlurKernel);
        free_gaussian_kernel(grainBlurKernelMid);
        free_gaussian_kernel(grainBlurKernelCoarse);
        for (int layer = 0; layer < 3; ++layer) {
            for (int ch = 0; ch < 3; ++ch) {
                free_gaussian_kernel(grainDyeKernel[layer][ch]);
            }
        }
        for (int i = 0; i < 3; ++i) {
            free_gaussian_kernel(halationKernel[i]);
            free_gaussian_kernel(halationScatterKernel[i]);
        }
        free_optics_scratch(*this, scannerScratch, nullptr);
        free_gaussian_kernel(spatialDirKernel);
        free_spatial_dir_scratch(*this, spatialDirScratch, nullptr);
        free_shared_tmp_plane(*this);
        free_stbn(*this);
        free_wang(*this);
        free_print_payloads(*this);
        free_hanatos(*this);
        free_hanatos_integrated(*this);
        free_mallett_basis(*this);
        free_scan_error_flag(*this);
        free_auto_exposure(*this);
        asyncDeviceAllocPointers.clear();
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (lastUseEventOpaque) {
            cudaEvent_t ev = reinterpret_cast<cudaEvent_t>(lastUseEventOpaque);
            cudaEventDestroy(ev);
            lastUseEventOpaque = nullptr;
        }
#endif
    }

    Resources* create() noexcept {
        try {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
            reap_deferred_destroy_queue("create");
#endif
            Resources* r = new Resources();
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
            int dev = -1;
            if (cudaGetDevice(&dev) == cudaSuccess) {
                r->deviceId = dev;
            }
            void* contextOpaque = nullptr;
            std::string contextError;
            if (query_current_cuda_context(contextOpaque, contextError)) {
                r->ownerContextOpaque = contextOpaque;
            }
#endif
            return r;
        }
        catch (...) {
            return nullptr;
        }
    }

    void destroy(Resources* resources) noexcept {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        delete resources;
#else
        if (!resources) {
            return;
        }

        reap_deferred_destroy_queue("destroy_pre");

        const int ownerDeviceId = resources->deviceId;
        void* ownerContextOpaque = resources->ownerContextOpaque;
        int currentDeviceId = -1;
        std::string currentDeviceError;
        const bool currentDeviceValid = query_current_cuda_device(currentDeviceId, currentDeviceError);
        void* currentContextOpaque = nullptr;
        std::string currentContextError;
        const bool currentContextValid = query_current_cuda_context(currentContextOpaque, currentContextError);

        bool deviceMatch = false;
        bool contextMatch = false;
        bool ownerMatch = owner_matches_current(
            ownerDeviceId,
            ownerContextOpaque,
            currentDeviceId,
            currentDeviceValid,
            currentContextOpaque,
            currentContextValid,
            deviceMatch,
            contextMatch);

        bool switchAttempted = false;
        bool switchSucceeded = false;
        int restoreDeviceId = currentDeviceId;
        bool restoreDevice = false;

        if (!ownerMatch && ownerDeviceId >= 0 &&
            (!currentDeviceValid || currentDeviceId != ownerDeviceId)) {
            switchAttempted = true;
            const cudaError_t setErr = cudaSetDevice(ownerDeviceId);
            if (setErr == cudaSuccess) {
                switchSucceeded = true;
                restoreDevice = currentDeviceValid && currentDeviceId != ownerDeviceId;

                int switchedDeviceId = -1;
                std::string switchedDeviceError;
                const bool switchedDeviceValid = query_current_cuda_device(switchedDeviceId, switchedDeviceError);

                void* switchedContextOpaque = nullptr;
                std::string switchedContextError;
                const bool switchedContextValid =
                    query_current_cuda_context(switchedContextOpaque, switchedContextError);

                ownerMatch = owner_matches_current(
                    ownerDeviceId,
                    ownerContextOpaque,
                    switchedDeviceId,
                    switchedDeviceValid,
                    switchedContextOpaque,
                    switchedContextValid,
                    deviceMatch,
                    contextMatch);

                if (switchedDeviceValid) {
                    currentDeviceId = switchedDeviceId;
                }
                if (switchedContextValid) {
                    currentContextOpaque = switchedContextOpaque;
                }
                if (!switchedDeviceError.empty()) {
                    currentDeviceError = switchedDeviceError;
                }
                if (!switchedContextError.empty()) {
                    currentContextError = switchedContextError;
                }
            }
            else {
                currentContextError = std::string("cudaSetDevice failed: ")
                    + (cudaGetErrorString(setErr) ? cudaGetErrorString(setErr) : "(unknown)");
            }
        }

        if (ownerMatch) {
            std::size_t deferredQueueDepth = 0;
            {
                std::lock_guard<std::mutex> lock(deferred_destroy_mutex());
                deferredQueueDepth = deferred_destroy_queue().size();
            }
            std::string detail;
            if (!currentDeviceError.empty()) {
                detail = currentDeviceError;
            }
            if (!currentContextError.empty()) {
                if (!detail.empty()) {
                    detail += "; ";
                }
                detail += currentContextError;
            }
            trace_teardown_event(
                "destroy",
                "direct_free",
                ownerDeviceId,
                ownerContextOpaque,
                currentDeviceId,
                currentContextOpaque,
                deviceMatch,
                contextMatch,
                switchAttempted,
                switchSucceeded,
                false,
                false,
                deferredQueueDepth,
                detail);
            delete resources;

            if (restoreDevice && restoreDeviceId >= 0 && restoreDeviceId != ownerDeviceId) {
                (void)cudaSetDevice(restoreDeviceId);
            }
            reap_deferred_destroy_queue("destroy_post");
            return;
        }

        if (restoreDevice && restoreDeviceId >= 0 && restoreDeviceId != ownerDeviceId) {
            (void)cudaSetDevice(restoreDeviceId);
        }

        bool managerRetireAttempted = false;
        bool managerRetireAccepted = false;
        std::string managerRetireError;
        if (ownerDeviceId >= 0 && ownerContextOpaque) {
            managerRetireAttempted = true;
            managerRetireAccepted =
                JuicerProcess::root().retire_idle_context(ownerDeviceId, ownerContextOpaque, managerRetireError);
        }

        std::size_t deferredQueueDepth = 0;
        {
            std::lock_guard<std::mutex> lock(deferred_destroy_mutex());
            auto& queue = deferred_destroy_queue();
            DeferredDestroyEntry entry{};
            entry.resources = resources;
            entry.ownerDeviceId = ownerDeviceId;
            entry.ownerContextOpaque = ownerContextOpaque;
            queue.push_back(entry);
            deferredQueueDepth = queue.size();
        }

        std::string detail;
        if (!currentDeviceError.empty()) {
            detail = currentDeviceError;
        }
        if (!currentContextError.empty()) {
            if (!detail.empty()) {
                detail += "; ";
            }
            detail += currentContextError;
        }
        if (!managerRetireError.empty()) {
            if (!detail.empty()) {
                detail += "; ";
            }
            detail += managerRetireError;
        }
        trace_teardown_event(
            "destroy",
            "deferred_enqueue",
            ownerDeviceId,
            ownerContextOpaque,
            currentDeviceId,
            currentContextOpaque,
            deviceMatch,
            contextMatch,
            switchAttempted,
            switchSucceeded,
            managerRetireAttempted,
            managerRetireAccepted,
            deferredQueueDepth,
            detail);
#endif
    }

    bool reap_retired_allocations(Resources& resources, std::size_t& reclaimedBytes, std::string& outError) {
        reclaimedBytes = 0;
        outError.clear();
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        outError = "CUDA is not enabled";
        return false;
#else
        std::lock_guard<std::mutex> lock(resources.m);
        if (!validate_resource_owner_locked(resources, outError, true)) {
            return false;
        }
        const std::size_t beforeBytes = resources.retireBytes;
        reap_retire_queue_locked(resources);
        const std::size_t afterBytes = resources.retireBytes;
        reclaimedBytes = (beforeBytes >= afterBytes) ? (beforeBytes - afterBytes) : 0;
        return true;
#endif
    }

    void snapshot_scratch_stage1_state(
        Resources& resources,
        ResourceManager::ScratchStage1DecisionState& outState) noexcept {
        outState = ResourceManager::ScratchStage1DecisionState{};
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        std::lock_guard<std::mutex> lock(resources.m);
        refresh_scratch_residency_state_locked(resources);
        outState.policyLiveRetainedBytes = resources.scratchResidency.policyLiveRetainedBytes;
        outState.retainedGeneration = resources.scratchResidency.retainedGeneration;
#else
        (void)resources;
#endif
    }

    void snapshot_scratch_residency_view(
        Resources& resources,
        ResourceManager::ScratchResidencyView& outView) noexcept {
        outView = ResourceManager::ScratchResidencyView{};
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        std::lock_guard<std::mutex> lock(resources.m);
        refresh_scratch_residency_state_locked(resources);
        for (std::size_t i = 0; i < ResourceManager::kScratchPolicyCandidateCount; ++i) {
            outView.candidates[i].candidate = ResourceManager::kScratchPolicyCandidateOrder[i];
            outView.candidates[i].liveRetainedBytes = resources.scratchResidency.candidateLiveBytes[i];
        }
        outView.helperNonPolicyBytes = resources.scratchResidency.helperNonPolicyBytes;
        outView.helperSharedBytes = resources.scratchResidency.helperSharedBytes;
        outView.helperNonPolicyTotalBytes = resources.scratchResidency.helperNonPolicyTotalBytes;
        outView.policyLiveRetainedBytes = resources.scratchResidency.policyLiveRetainedBytes;
        outView.totalLiveRetainedBytes = resources.scratchResidency.totalLiveRetainedBytes;
        outView.retirePendingScratchBytes = static_cast<std::uint64_t>(
            std::min<std::size_t>(
                resources.retireScratchBytes,
                static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())));
        outView.retainedGeneration = resources.scratchResidency.retainedGeneration;
        outView.overflow = resources.scratchResidency.overflow;
#else
        (void)resources;
#endif
    }

    void record_use(Resources& resources, void* cudaStreamOpaque) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
        std::lock_guard<std::mutex> lock(resources.m);
        reap_retire_queue_locked(resources);
        if (!resources.lastUseEventOpaque) {
            cudaEvent_t ev = nullptr;
            const cudaError_t err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
            if (err != cudaSuccess || !ev) {
                return;
            }
            resources.lastUseEventOpaque = reinterpret_cast<void*>(ev);
        }
        cudaEvent_t ev = reinterpret_cast<cudaEvent_t>(resources.lastUseEventOpaque);
        (void)cudaEventRecord(ev, stream);
#else
        (void)resources;
        (void)cudaStreamOpaque;
#endif
    }



    // Split implementation sections (single-TU include model to preserve exact behavior while
    // reducing monolithic file size and keeping ownership boundaries explicit).
    #include "Cuda/JuicerCudaResourcesServing.cpp"
// Cuda/JuicerCudaResourcesScratch.cpp
//
// Included by JuicerCudaResources.cpp (single-TU split).
    enum class SharedGaussianKind : std::uint8_t {
        Standard = 0,
        Halation = 1,
        SpatialDir = 2
    };

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)

    struct SharedGaussianKey {
        int deviceId = -1;
        void* contextOpaque = nullptr;
        SharedGaussianKind kind = SharedGaussianKind::Standard;
        int radius = 0;
        std::uint32_t sigmaBits = 0;

        bool operator==(const SharedGaussianKey& other) const noexcept {
            return deviceId == other.deviceId &&
                contextOpaque == other.contextOpaque &&
                kind == other.kind &&
                radius == other.radius &&
                sigmaBits == other.sigmaBits;
        }
    };

    struct SharedGaussianKeyHash {
        std::size_t operator()(const SharedGaussianKey& key) const noexcept {
            const std::size_t hDevice = std::hash<int>{}(key.deviceId);
            const std::size_t hContext = std::hash<std::uintptr_t>{}(
                reinterpret_cast<std::uintptr_t>(key.contextOpaque));
            const std::size_t hKind = std::hash<std::uint8_t>{}(
                static_cast<std::uint8_t>(key.kind));
            const std::size_t hRadius = std::hash<int>{}(key.radius);
            const std::size_t hSigma = std::hash<std::uint32_t>{}(key.sigmaBits);
            std::size_t h = hDevice;
            h ^= hContext + 0x9e3779b9u + (h << 6u) + (h >> 2u);
            h ^= hKind + 0x9e3779b9u + (h << 6u) + (h >> 2u);
            h ^= hRadius + 0x9e3779b9u + (h << 6u) + (h >> 2u);
            h ^= hSigma + 0x9e3779b9u + (h << 6u) + (h >> 2u);
            return h;
        }
    };

    struct SharedGaussianEntry {
        float* weights = nullptr;
        int radius = 0;
        float sigma = 0.0f;
        int capacity = 0;
        std::uint64_t id = 0;
    };

    struct SharedGaussianCacheState {
        std::mutex mutex;
        std::unordered_map<SharedGaussianKey, SharedGaussianEntry, SharedGaussianKeyHash> byKey;
    };

    static SharedGaussianCacheState& shared_gaussian_cache_state() {
        static SharedGaussianCacheState state;
        return state;
    }

    static std::uint32_t gaussian_sigma_bits(float sigma) noexcept {
        float canonical = sigma;
        if (canonical == 0.0f) {
            canonical = 0.0f;
        }
        std::uint32_t bits = 0;
        std::memcpy(&bits, &canonical, sizeof(bits));
        return bits;
    }

    static std::uint64_t make_shared_gaussian_id(const SharedGaussianKey& key) noexcept {
        std::uint64_t h = Hash::kFnvOffset;
        Hash::hash_bytes_update(h, &key.deviceId, sizeof(key.deviceId));
        const std::uintptr_t contextBits =
            reinterpret_cast<std::uintptr_t>(key.contextOpaque);
        Hash::hash_bytes_update(h, &contextBits, sizeof(contextBits));
        const std::uint8_t kindValue = static_cast<std::uint8_t>(key.kind);
        Hash::hash_bytes_update(h, &kindValue, sizeof(kindValue));
        Hash::hash_bytes_update(h, &key.radius, sizeof(key.radius));
        Hash::hash_bytes_update(h, &key.sigmaBits, sizeof(key.sigmaBits));
        return h;
    }

    static SharedGaussianKey make_shared_gaussian_key(
        const Resources& resources,
        SharedGaussianKind kind,
        int radius,
        float sigma) noexcept {
        SharedGaussianKey key{};
        key.deviceId = resources.deviceId;
        key.contextOpaque = resources.ownerContextOpaque;
        key.kind = kind;
        key.radius = radius;
        key.sigmaBits = gaussian_sigma_bits(sigma);
        return key;
    }

    static void build_gaussian_weights_cpu(
        int radius,
        float sigma,
        std::vector<float>& outWeights) {
        outWeights.clear();
        if (radius <= 0 || !std::isfinite(sigma) || sigma <= 0.0f) {
            return;
        }
        outWeights.resize(static_cast<std::size_t>(2 * radius + 1));
        const double s2 = static_cast<double>(sigma) * static_cast<double>(sigma) * 2.0;
        double wsum = 0.0;
        for (int i = -radius; i <= radius; ++i) {
            const double w = std::exp(-(static_cast<double>(i * i)) / s2);
            outWeights[static_cast<std::size_t>(i + radius)] = static_cast<float>(w);
            wsum += w;
        }
        const double invW = (wsum != 0.0) ? (1.0 / wsum) : 0.0;
        for (float& w : outWeights) {
            w = static_cast<float>(static_cast<double>(w) * invW);
        }
    }

    static bool acquire_shared_gaussian_entry(
        const SharedGaussianKey& key,
        int radius,
        float sigma,
        const std::vector<float>& cpuWeights,
        SharedGaussianEntry& outEntry,
        std::string& outError) {
        outError.clear();
        SharedGaussianCacheState& cache = shared_gaussian_cache_state();
        {
            std::lock_guard<std::mutex> lock(cache.mutex);
            const auto it = cache.byKey.find(key);
            if (it != cache.byKey.end()) {
                outEntry = it->second;
                return true;
            }
        }

        float* dWeights = nullptr;
        const std::size_t bytes = cpuWeights.size() * sizeof(float);
        const cudaError_t allocErr = cudaMalloc(reinterpret_cast<void**>(&dWeights), bytes);
        if (allocErr != cudaSuccess || !dWeights) {
            outError = std::string("cudaMalloc(shared gaussian kernel) failed: ")
                + (cudaGetErrorString(allocErr) ? cudaGetErrorString(allocErr) : "(unknown)");
            dWeights = nullptr;
            return false;
        }
        const cudaError_t copyErr = cudaMemcpy(
            dWeights,
            cpuWeights.data(),
            bytes,
            cudaMemcpyHostToDevice);
        if (copyErr != cudaSuccess) {
            outError = std::string("cudaMemcpy(shared gaussian kernel) failed: ")
                + (cudaGetErrorString(copyErr) ? cudaGetErrorString(copyErr) : "(unknown)");
            cudaFree(dWeights);
            return false;
        }

        std::lock_guard<std::mutex> lock(cache.mutex);
        const auto existingIt = cache.byKey.find(key);
        if (existingIt != cache.byKey.end()) {
            cudaFree(dWeights);
            outEntry = existingIt->second;
            return true;
        }

        SharedGaussianEntry entry{};
        entry.weights = dWeights;
        entry.radius = radius;
        entry.sigma = sigma;
        entry.capacity = static_cast<int>(cpuWeights.size());
        entry.id = make_shared_gaussian_id(key);
        cache.byKey.emplace(key, entry);
        outEntry = entry;
        return true;
    }

    static bool shared_gaussian_entry_matches_cache(
        const SharedGaussianKey& key,
        const Resources::DeviceGaussianKernel& kernel) {
        if (kernel.sharedKernelId == 0 || !kernel.weights) {
            return false;
        }
        SharedGaussianCacheState& cache = shared_gaussian_cache_state();
        std::lock_guard<std::mutex> lock(cache.mutex);
        const auto it = cache.byKey.find(key);
        if (it == cache.byKey.end()) {
            return false;
        }
        return it->second.id == kernel.sharedKernelId &&
            it->second.weights == kernel.weights;
    }

    static const char* to_cstr(ResourceManager::AllocatorBackendMode mode) noexcept {
        switch (mode) {
        case ResourceManager::AllocatorBackendMode::Legacy:
            return "legacy";
        case ResourceManager::AllocatorBackendMode::AsyncPool:
            return "async_pool";
        case ResourceManager::AllocatorBackendMode::Slab:
            return "slab";
        default:
            return "unknown";
        }
    }

    static ResourceManager::AllocatorBackendMode scratch_allocator_backend_mode_locked(
        const Resources& resources) noexcept {
        ResourceManager::DeviceContextKey key{};
        key.deviceId = resources.deviceId;
        key.contextOpaque = resources.ownerContextOpaque;
        return ResourceManager::query_allocator_backend_mode(key);
    }

    static bool is_async_device_ptr_tracked_locked(const Resources& resources, const void* ptr) noexcept {
        return ptr &&
            resources.asyncDeviceAllocPointers.find(const_cast<void*>(ptr)) !=
                resources.asyncDeviceAllocPointers.end();
    }

    static void track_async_device_ptr_locked(Resources& resources, void* ptr, bool asyncAllocated) noexcept {
        if (!ptr) {
            return;
        }
        if (asyncAllocated) {
            resources.asyncDeviceAllocPointers.insert(ptr);
        }
        else {
            resources.asyncDeviceAllocPointers.erase(ptr);
        }
    }

    static void untrack_async_device_ptr_locked(Resources& resources, void* ptr) noexcept {
        if (!ptr) {
            return;
        }
        resources.asyncDeviceAllocPointers.erase(ptr);
    }

    static cudaError_t device_free_async_compat(void* ptr, void* cudaStreamOpaque) noexcept {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
        const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
        return cudaFreeAsync(ptr, stream);
#else
        (void)cudaStreamOpaque;
        return cudaErrorNotSupported;
#endif
    }

    static void free_tracked_device_ptr_locked(
        Resources& resources,
        void*& ptr,
        void* cudaStreamOpaque) noexcept {
        if (!ptr) {
            return;
        }
        const bool asyncTracked = is_async_device_ptr_tracked_locked(resources, ptr);
        if (asyncTracked) {
            const cudaError_t asyncErr = device_free_async_compat(ptr, cudaStreamOpaque);
            if (asyncErr != cudaSuccess) {
                (void)cudaFree(ptr);
            }
            else if (!cudaStreamOpaque) {
                (void)cudaStreamSynchronize(nullptr);
            }
        }
        else {
            (void)cudaFree(ptr);
        }
        untrack_async_device_ptr_locked(resources, ptr);
        ptr = nullptr;
    }

    template <typename T>
    static void free_tracked_device_ptr_locked(
        Resources& resources,
        T*& ptr,
        void* cudaStreamOpaque) noexcept {
        void* raw = reinterpret_cast<void*>(ptr);
        free_tracked_device_ptr_locked(resources, raw, cudaStreamOpaque);
        ptr = reinterpret_cast<T*>(raw);
    }

    static bool allocate_scratch_device_ptr_locked(
        Resources& resources,
        void*& outPtr,
        std::size_t bytes,
        void* cudaStreamOpaque,
        const char* label,
        std::string& outError) {
        outPtr = nullptr;
        if (bytes == 0) {
            outError = std::string(label ? label : "scratch") + " bytes invalid";
            return false;
        }

        const ResourceManager::AllocatorBackendMode backendMode =
            scratch_allocator_backend_mode_locked(resources);
        const bool preferAsync = (backendMode == ResourceManager::AllocatorBackendMode::AsyncPool);

        cudaError_t asyncErr = cudaSuccess;
        if (preferAsync) {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
            const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
            asyncErr = cudaMallocAsync(reinterpret_cast<void**>(&outPtr), bytes, stream);
            if (asyncErr == cudaSuccess && outPtr) {
                track_async_device_ptr_locked(resources, outPtr, true);
                return true;
            }
            outPtr = nullptr;
#else
            asyncErr = cudaErrorNotSupported;
#endif
        }

        const cudaError_t allocErr = cudaMalloc(reinterpret_cast<void**>(&outPtr), bytes);
        if (allocErr == cudaSuccess && outPtr) {
            track_async_device_ptr_locked(resources, outPtr, false);
            if (preferAsync && JTRACE_ENABLED(2)) {
                std::ostringstream oss;
                oss << "event=alloc_fallback"
                    << " path=scratch"
                    << " label=" << (label ? label : "scratch")
                    << " backend_requested=" << to_cstr(backendMode)
                    << " async_error=" << (cudaGetErrorString(asyncErr) ? cudaGetErrorString(asyncErr) : "(unknown)");
                JTRACE("MSALC", oss.str());
            }
            return true;
        }

        if (preferAsync) {
            outError = std::string("scratch alloc failed (async+legacy) [")
                + (label ? label : "scratch")
                + "]: async="
                + (cudaGetErrorString(asyncErr) ? cudaGetErrorString(asyncErr) : "(unknown)")
                + ", legacy="
                + (cudaGetErrorString(allocErr) ? cudaGetErrorString(allocErr) : "(unknown)");
        }
        else {
            outError = std::string("cudaMalloc(") + (label ? label : "scratch") + ") failed: "
                + (cudaGetErrorString(allocErr) ? cudaGetErrorString(allocErr) : "(unknown)");
        }
        outPtr = nullptr;
        return false;
    }

    template <typename T>
    static bool allocate_scratch_device_ptr_locked(
        Resources& resources,
        T*& outPtr,
        std::size_t bytes,
        void* cudaStreamOpaque,
        const char* label,
        std::string& outError) {
        void* raw = nullptr;
        const bool ok = allocate_scratch_device_ptr_locked(
            resources,
            raw,
            bytes,
            cudaStreamOpaque,
            label,
            outError);
        outPtr = reinterpret_cast<T*>(raw);
        return ok;
    }
#endif

    static void free_auto_exposure(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (resources.autoExposureScratch.partialsA) {
            cudaFree(resources.autoExposureScratch.partialsA);
            resources.autoExposureScratch.partialsA = nullptr;
        }
        if (resources.autoExposureScratch.partialsB) {
            cudaFree(resources.autoExposureScratch.partialsB);
            resources.autoExposureScratch.partialsB = nullptr;
        }
        if (resources.autoExposureScratch.maxYBits) {
            cudaFree(resources.autoExposureScratch.maxYBits);
            resources.autoExposureScratch.maxYBits = nullptr;
        }
        if (resources.autoExposureScratch.histogram) {
            cudaFree(resources.autoExposureScratch.histogram);
            resources.autoExposureScratch.histogram = nullptr;
        }
        if (resources.autoExposureScratch.weightsX) {
            cudaFree(resources.autoExposureScratch.weightsX);
            resources.autoExposureScratch.weightsX = nullptr;
        }
        if (resources.autoExposureScratch.weightsY) {
            cudaFree(resources.autoExposureScratch.weightsY);
            resources.autoExposureScratch.weightsY = nullptr;
        }
        if (resources.autoExposureExposureScale) {
            cudaFree(resources.autoExposureExposureScale);
            resources.autoExposureExposureScale = nullptr;
        }
        if (resources.autoExposureAutoEV) {
            cudaFree(resources.autoExposureAutoEV);
            resources.autoExposureAutoEV = nullptr;
        }
        if (resources.autoExposureValid) {
            cudaFree(resources.autoExposureValid);
            resources.autoExposureValid = nullptr;
        }
#endif
        resources.autoExposureScratch.partialCapacity = 0;
        resources.autoExposureScratch.weightsXCapacity = 0;
        resources.autoExposureScratch.weightsYCapacity = 0;
        resources.autoExposureScratch.weightsWidth = 0;
        resources.autoExposureScratch.weightsHeight = 0;
        resources.autoExposureKeyHash = 0;
        resources.autoExposureSliderEV = std::numeric_limits<double>::quiet_NaN();
    }

    static void free_tables(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (resources.tablesAx) {
            cudaFree(resources.tablesAx);
            resources.tablesAx = nullptr;
        }
        if (resources.tablesAy) {
            cudaFree(resources.tablesAy);
            resources.tablesAy = nullptr;
        }
        if (resources.tablesAz) {
            cudaFree(resources.tablesAz);
            resources.tablesAz = nullptr;
        }
        if (resources.tablesIllum) {
            cudaFree(resources.tablesIllum);
            resources.tablesIllum = nullptr;
        }
#endif
        resources.tablesK = 0;
    }

    static bool retire_tables_locked(Resources& resources, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        const size_t bytes = static_cast<size_t>(std::max(0, resources.tablesK)) * sizeof(float);
        if (resources.tablesAx) {
            if (!retire_ptr_locked(resources, resources.tablesAx, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                return false;
            }
            resources.tablesAx = nullptr;
        }
        if (resources.tablesAy) {
            if (!retire_ptr_locked(resources, resources.tablesAy, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                return false;
            }
            resources.tablesAy = nullptr;
        }
        if (resources.tablesAz) {
            if (!retire_ptr_locked(resources, resources.tablesAz, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                return false;
            }
            resources.tablesAz = nullptr;
        }
        if (resources.tablesIllum) {
            if (!retire_ptr_locked(resources, resources.tablesIllum, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                return false;
            }
            resources.tablesIllum = nullptr;
        }
        resources.tablesK = 0;
        return true;
#endif
    }

    static void free_spectral_tables(Resources::DeviceSpectralTables& t) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (t.epsC) { cudaFree(t.epsC); t.epsC = nullptr; }
        if (t.epsM) { cudaFree(t.epsM); t.epsM = nullptr; }
        if (t.epsY) { cudaFree(t.epsY); t.epsY = nullptr; }
        if (t.Ax) { cudaFree(t.Ax); t.Ax = nullptr; }
        if (t.Ay) { cudaFree(t.Ay); t.Ay = nullptr; }
        if (t.Az) { cudaFree(t.Az); t.Az = nullptr; }
        if (t.baseMin) { cudaFree(t.baseMin); t.baseMin = nullptr; }
#endif
        t.K = 0;
        t.hasBaseline = 0;
        t.invYn = 1.0f;
    }

    static bool retire_spectral_tables_locked(Resources& resources, Resources::DeviceSpectralTables& t, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)t;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        const size_t bytes = static_cast<size_t>(std::max(0, t.K)) * sizeof(float);
        if (t.epsC) {
            if (!retire_ptr_locked(resources, t.epsC, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) return false;
            t.epsC = nullptr;
        }
        if (t.epsM) {
            if (!retire_ptr_locked(resources, t.epsM, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) return false;
            t.epsM = nullptr;
        }
        if (t.epsY) {
            if (!retire_ptr_locked(resources, t.epsY, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) return false;
            t.epsY = nullptr;
        }
        if (t.Ax) {
            if (!retire_ptr_locked(resources, t.Ax, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) return false;
            t.Ax = nullptr;
        }
        if (t.Ay) {
            if (!retire_ptr_locked(resources, t.Ay, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) return false;
            t.Ay = nullptr;
        }
        if (t.Az) {
            if (!retire_ptr_locked(resources, t.Az, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) return false;
            t.Az = nullptr;
        }
        if (t.baseMin) {
            if (!retire_ptr_locked(resources, t.baseMin, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) return false;
            t.baseMin = nullptr;
        }
        t.K = 0;
        t.hasBaseline = 0;
        t.invYn = 1.0f;
        return true;
#endif
    }

    static void free_scan_medium(Resources::DeviceScanMedium& m) noexcept {
        free_spectral_tables(m.tables);
        m.mediumIsNegative = 1;
        m.min_cmy[0] = m.min_cmy[1] = m.min_cmy[2] = 0.0f;
        m.inv_max_cmy[0] = m.inv_max_cmy[1] = m.inv_max_cmy[2] = 1.0f;
    }

    static bool retire_scan_medium_locked(Resources& resources, Resources::DeviceScanMedium& m, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)m;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (!retire_spectral_tables_locked(resources, m.tables, cudaStreamOpaque, label, outError)) {
            return false;
        }
        m.mediumIsNegative = 1;
        m.min_cmy[0] = m.min_cmy[1] = m.min_cmy[2] = 0.0f;
        m.inv_max_cmy[0] = m.inv_max_cmy[1] = m.inv_max_cmy[2] = 1.0f;
        return true;
#endif
    }

    static void free_scan_lut(Resources::DeviceSpectralLut& lut) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (lut.log2XYZ) {
            cudaFree(lut.log2XYZ);
            lut.log2XYZ = nullptr;
        }
#endif
        lut.res = 0;
        lut.hash = 0;
    }

    static void free_gaussian_kernel(Resources::DeviceGaussianKernel& k) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (k.weights) {
            if (k.sharedKernelId == 0) {
                cudaFree(k.weights);
            }
            k.weights = nullptr;
        }
#endif
        k.radius = 0;
        k.sigma = 0.0f;
        k.capacity = 0;
        k.sharedKernelId = 0;
    }

    static void free_optics_scratch(Resources& resources, Resources::DeviceOpticsScratch& s, void* cudaStreamOpaque = nullptr) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        free_tracked_device_ptr_locked(resources, s.rgbR, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.rgbG, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.rgbB, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.blurred, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.aux, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.grainTmp, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.grainTmpShared, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.grainTmpMid, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.grainTmpCoarse, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.gateMask, cudaStreamOpaque);
#endif
        s.tmp = nullptr;
        s.width = 0;
        s.height = 0;
        s.capacityElements = 0;
        s.gateWidth = 0;
        s.gateHeight = 0;
        s.gateMaskCapacityElements = 0;
        s.gateMaskHash = 0;
    }

    static bool retire_optics_scratch_locked(Resources& resources, Resources::DeviceOpticsScratch& s, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)s;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        const size_t planeBytes = s.capacityElements * sizeof(float);
        if (s.rgbR) {
            if (!retire_ptr_locked(resources, s.rgbR, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true)) return false;
            s.rgbR = nullptr;
        }
        if (s.rgbG) {
            if (!retire_ptr_locked(resources, s.rgbG, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true)) return false;
            s.rgbG = nullptr;
        }
        if (s.rgbB) {
            if (!retire_ptr_locked(resources, s.rgbB, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true)) return false;
            s.rgbB = nullptr;
        }
        if (s.blurred) {
            if (!retire_ptr_locked(resources, s.blurred, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true)) return false;
            s.blurred = nullptr;
        }
        if (s.aux) {
            if (!retire_ptr_locked(resources, s.aux, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true)) return false;
            s.aux = nullptr;
        }
        if (s.grainTmp) {
            if (!retire_ptr_locked(resources, s.grainTmp, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true)) return false;
            s.grainTmp = nullptr;
        }
        if (s.grainTmpShared) {
            if (!retire_ptr_locked(resources, s.grainTmpShared, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true)) return false;
            s.grainTmpShared = nullptr;
        }
        if (s.grainTmpMid) {
            if (!retire_ptr_locked(resources, s.grainTmpMid, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true)) return false;
            s.grainTmpMid = nullptr;
        }
        if (s.grainTmpCoarse) {
            if (!retire_ptr_locked(resources, s.grainTmpCoarse, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true)) return false;
            s.grainTmpCoarse = nullptr;
        }

        const size_t gateBytes = s.gateMaskCapacityElements * sizeof(float);
        if (s.gateMask) {
            if (!retire_ptr_locked(resources, s.gateMask, gateBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true)) return false;
            s.gateMask = nullptr;
        }

        s.tmp = nullptr;
        s.width = 0;
        s.height = 0;
        s.capacityElements = 0;
        s.gateWidth = 0;
        s.gateHeight = 0;
        s.gateMaskCapacityElements = 0;
        s.gateMaskHash = 0;
        return true;
#endif
    }

    static void free_spatial_dir_scratch(Resources& resources, Resources::DeviceSpatialDirScratch& s, void* cudaStreamOpaque = nullptr) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        free_tracked_device_ptr_locked(resources, s.corrY, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.corrM, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.corrC, cudaStreamOpaque);
#endif
        s.tmp = nullptr;
        s.width = 0;
        s.height = 0;
        s.capacityElements = 0;
    }

    static bool retire_spatial_dir_scratch_locked(Resources& resources, Resources::DeviceSpatialDirScratch& s, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)s;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        const size_t bytes = s.capacityElements * sizeof(float);
        if (s.corrY) {
            if (!retire_ptr_locked(resources, s.corrY, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true)) return false;
            s.corrY = nullptr;
        }
        if (s.corrM) {
            if (!retire_ptr_locked(resources, s.corrM, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true)) return false;
            s.corrM = nullptr;
        }
        if (s.corrC) {
            if (!retire_ptr_locked(resources, s.corrC, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true)) return false;
            s.corrC = nullptr;
        }
        s.tmp = nullptr;
        s.width = 0;
        s.height = 0;
        s.capacityElements = 0;
        return true;
#endif
    }

    static void free_shared_tmp_plane(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        free_tracked_device_ptr_locked(
            resources,
            resources.sharedTmpPlane,
            nullptr);
#endif
        resources.sharedTmpPlane = nullptr;
        resources.sharedTmpWidth = 0;
        resources.sharedTmpHeight = 0;
        resources.sharedTmpCapacityElements = 0;
    }

    static bool optics_base_live_locked(const Resources& resources) noexcept {
        const auto& scratch = resources.scannerScratch;
        return scratch.rgbR || scratch.rgbG || scratch.rgbB;
    }

    static bool optics_any_live_locked(const Resources::DeviceOpticsScratch& scratch) noexcept {
        return scratch.rgbR || scratch.rgbG || scratch.rgbB ||
            scratch.blurred || scratch.aux ||
            scratch.grainTmp || scratch.grainTmpShared ||
            scratch.grainTmpMid || scratch.grainTmpCoarse ||
            scratch.gateMask;
    }

    static bool spatial_dir_base_live_locked(const Resources& resources) noexcept {
        const auto& scratch = resources.spatialDirScratch;
        return scratch.corrY || scratch.corrM || scratch.corrC;
    }

    static bool retire_shared_tmp_plane_locked(
        Resources& resources,
        void* cudaStreamOpaque,
        const char* label,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (!resources.sharedTmpPlane) {
            return true;
        }

        const std::size_t bytes = resources.sharedTmpCapacityElements * sizeof(float);
        if (!retire_ptr_locked(
                resources,
                resources.sharedTmpPlane,
                bytes,
                Resources::RetireKind::DeviceFree,
                cudaStreamOpaque,
                label ? label : "shared tmp plane",
                outError,
                true)) {
            return false;
        }
        resources.sharedTmpPlane = nullptr;
        resources.sharedTmpWidth = 0;
        resources.sharedTmpHeight = 0;
        resources.sharedTmpCapacityElements = 0;
        resources.scannerScratch.tmp = nullptr;
        resources.spatialDirScratch.tmp = nullptr;
        return true;
#endif
    }

    static bool retire_orphaned_shared_tmp_plane_locked(
        Resources& resources,
        void* cudaStreamOpaque,
        const char* label,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (!resources.sharedTmpPlane) {
            return true;
        }
        if (optics_base_live_locked(resources) || spatial_dir_base_live_locked(resources)) {
            return true;
        }
        return retire_shared_tmp_plane_locked(
            resources,
            cudaStreamOpaque,
            label ? label : "scratch normalization shared tmp plane",
            outError);
#endif
    }

    static bool retire_optics_gate_mask_candidate_locked(
        Resources& resources,
        void* cudaStreamOpaque,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        Resources::DeviceOpticsScratch& scratch = resources.scannerScratch;
        if (!scratch.gateMask) {
            return true;
        }

        const std::size_t bytes = scratch.gateMaskCapacityElements * sizeof(float);
        if (!retire_ptr_locked(
                resources,
                scratch.gateMask,
                bytes,
                Resources::RetireKind::DeviceFree,
                cudaStreamOpaque,
                "scratch normalization optics gate mask",
                outError,
                true)) {
            return false;
        }
        scratch.gateMask = nullptr;
        scratch.gateWidth = 0;
        scratch.gateHeight = 0;
        scratch.gateMaskCapacityElements = 0;
        scratch.gateMaskHash = 0;
        return true;
#endif
    }

    static bool retire_optics_grain_shared_candidate_locked(
        Resources& resources,
        void* cudaStreamOpaque,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        Resources::DeviceOpticsScratch& scratch = resources.scannerScratch;
        if (!scratch.grainTmpShared) {
            return true;
        }

        const std::size_t bytes = scratch.capacityElements * sizeof(float);
        if (!retire_ptr_locked(
                resources,
                scratch.grainTmpShared,
                bytes,
                Resources::RetireKind::DeviceFree,
                cudaStreamOpaque,
                "scratch normalization optics grain shared",
                outError,
                true)) {
            return false;
        }
        scratch.grainTmpShared = nullptr;
        return true;
#endif
    }

    static bool retire_optics_grain_triplet_candidate_locked(
        Resources& resources,
        void* cudaStreamOpaque,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        Resources::DeviceOpticsScratch& scratch = resources.scannerScratch;
        const std::size_t bytes = scratch.capacityElements * sizeof(float);
        if (scratch.grainTmp) {
            if (!retire_ptr_locked(
                    resources,
                    scratch.grainTmp,
                    bytes,
                    Resources::RetireKind::DeviceFree,
                    cudaStreamOpaque,
                    "scratch normalization optics grain triplet",
                    outError,
                    true)) {
                return false;
            }
            scratch.grainTmp = nullptr;
        }
        if (scratch.grainTmpMid) {
            if (!retire_ptr_locked(
                    resources,
                    scratch.grainTmpMid,
                    bytes,
                    Resources::RetireKind::DeviceFree,
                    cudaStreamOpaque,
                    "scratch normalization optics grain triplet",
                    outError,
                    true)) {
                return false;
            }
            scratch.grainTmpMid = nullptr;
        }
        if (scratch.grainTmpCoarse) {
            if (!retire_ptr_locked(
                    resources,
                    scratch.grainTmpCoarse,
                    bytes,
                    Resources::RetireKind::DeviceFree,
                    cudaStreamOpaque,
                    "scratch normalization optics grain triplet",
                    outError,
                    true)) {
                return false;
            }
            scratch.grainTmpCoarse = nullptr;
        }
        return true;
#endif
    }

    static bool retire_optics_aux_candidate_locked(
        Resources& resources,
        void* cudaStreamOpaque,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        Resources::DeviceOpticsScratch& scratch = resources.scannerScratch;
        if (!scratch.aux) {
            return true;
        }

        const std::size_t bytes = scratch.capacityElements * sizeof(float);
        if (!retire_ptr_locked(
                resources,
                scratch.aux,
                bytes,
                Resources::RetireKind::DeviceFree,
                cudaStreamOpaque,
                "scratch normalization optics aux",
                outError,
                true)) {
            return false;
        }
        scratch.aux = nullptr;
        return true;
#endif
    }

    static bool retire_optics_blurred_candidate_locked(
        Resources& resources,
        void* cudaStreamOpaque,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        Resources::DeviceOpticsScratch& scratch = resources.scannerScratch;
        if (!scratch.blurred) {
            return true;
        }

        const std::size_t bytes = scratch.capacityElements * sizeof(float);
        if (!retire_ptr_locked(
                resources,
                scratch.blurred,
                bytes,
                Resources::RetireKind::DeviceFree,
                cudaStreamOpaque,
                "scratch normalization optics blurred",
                outError,
                true)) {
            return false;
        }
        scratch.blurred = nullptr;
        return true;
#endif
    }

    static bool retire_spatial_dir_base_candidate_locked(
        Resources& resources,
        void* cudaStreamOpaque,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        Resources::DeviceSpatialDirScratch& scratch = resources.spatialDirScratch;
        const std::size_t bytes = scratch.capacityElements * sizeof(float);
        if (scratch.corrY) {
            if (!retire_ptr_locked(
                    resources,
                    scratch.corrY,
                    bytes,
                    Resources::RetireKind::DeviceFree,
                    cudaStreamOpaque,
                    "scratch normalization spatial dir base",
                    outError,
                    true)) {
                return false;
            }
            scratch.corrY = nullptr;
        }
        if (scratch.corrM) {
            if (!retire_ptr_locked(
                    resources,
                    scratch.corrM,
                    bytes,
                    Resources::RetireKind::DeviceFree,
                    cudaStreamOpaque,
                    "scratch normalization spatial dir base",
                    outError,
                    true)) {
                return false;
            }
            scratch.corrM = nullptr;
        }
        if (scratch.corrC) {
            if (!retire_ptr_locked(
                    resources,
                    scratch.corrC,
                    bytes,
                    Resources::RetireKind::DeviceFree,
                    cudaStreamOpaque,
                    "scratch normalization spatial dir base",
                    outError,
                    true)) {
                return false;
            }
            scratch.corrC = nullptr;
        }
        scratch.tmp = nullptr;
        scratch.width = 0;
        scratch.height = 0;
        scratch.capacityElements = 0;
        return retire_orphaned_shared_tmp_plane_locked(
            resources,
            cudaStreamOpaque,
            "scratch normalization spatial dir shared tmp",
            outError);
#endif
    }

    static bool retire_optics_base_candidate_locked(
        Resources& resources,
        void* cudaStreamOpaque,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        Resources::DeviceOpticsScratch& scratch = resources.scannerScratch;
        const std::size_t bytes = scratch.capacityElements * sizeof(float);
        if (scratch.rgbR) {
            if (!retire_ptr_locked(
                    resources,
                    scratch.rgbR,
                    bytes,
                    Resources::RetireKind::DeviceFree,
                    cudaStreamOpaque,
                    "scratch normalization optics base",
                    outError,
                    true)) {
                return false;
            }
            scratch.rgbR = nullptr;
        }
        if (scratch.rgbG) {
            if (!retire_ptr_locked(
                    resources,
                    scratch.rgbG,
                    bytes,
                    Resources::RetireKind::DeviceFree,
                    cudaStreamOpaque,
                    "scratch normalization optics base",
                    outError,
                    true)) {
                return false;
            }
            scratch.rgbG = nullptr;
        }
        if (scratch.rgbB) {
            if (!retire_ptr_locked(
                    resources,
                    scratch.rgbB,
                    bytes,
                    Resources::RetireKind::DeviceFree,
                    cudaStreamOpaque,
                    "scratch normalization optics base",
                    outError,
                    true)) {
                return false;
            }
            scratch.rgbB = nullptr;
        }

        if (!optics_any_live_locked(scratch)) {
            scratch.tmp = nullptr;
            scratch.width = 0;
            scratch.height = 0;
            scratch.capacityElements = 0;
            scratch.gateWidth = 0;
            scratch.gateHeight = 0;
            scratch.gateMaskCapacityElements = 0;
            scratch.gateMaskHash = 0;
        }
        return retire_orphaned_shared_tmp_plane_locked(
            resources,
            cudaStreamOpaque,
            "scratch normalization optics shared tmp",
            outError);
#endif
    }

    bool retire_scratch_policy_candidate(
        Resources& resources,
        ResourceManager::ScratchPolicyCandidate candidate,
        void* cudaStreamOpaque,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)candidate;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        std::lock_guard<std::mutex> lock(resources.m);
        reap_retire_queue_locked(resources);
        if (!validate_resource_owner_locked(resources, outError, true)) {
            return false;
        }

        switch (candidate) {
        case ResourceManager::ScratchPolicyCandidate::OpticsGateMask:
            return retire_optics_gate_mask_candidate_locked(resources, cudaStreamOpaque, outError);
        case ResourceManager::ScratchPolicyCandidate::OpticsGrainShared:
            return retire_optics_grain_shared_candidate_locked(resources, cudaStreamOpaque, outError);
        case ResourceManager::ScratchPolicyCandidate::OpticsGrainTriplet:
            return retire_optics_grain_triplet_candidate_locked(resources, cudaStreamOpaque, outError);
        case ResourceManager::ScratchPolicyCandidate::OpticsAux:
            return retire_optics_aux_candidate_locked(resources, cudaStreamOpaque, outError);
        case ResourceManager::ScratchPolicyCandidate::OpticsBlurred:
            return retire_optics_blurred_candidate_locked(resources, cudaStreamOpaque, outError);
        case ResourceManager::ScratchPolicyCandidate::SpatialDirBase:
            return retire_spatial_dir_base_candidate_locked(resources, cudaStreamOpaque, outError);
        case ResourceManager::ScratchPolicyCandidate::OpticsBase:
            return retire_optics_base_candidate_locked(resources, cudaStreamOpaque, outError);
        default:
            outError = "scratch normalization candidate is invalid";
            return false;
        }
#endif
    }

    bool retire_orphaned_shared_tmp_plane(
        Resources& resources,
        void* cudaStreamOpaque,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        std::lock_guard<std::mutex> lock(resources.m);
        reap_retire_queue_locked(resources);
        if (!validate_resource_owner_locked(resources, outError, true)) {
            return false;
        }
        return retire_orphaned_shared_tmp_plane_locked(
            resources,
            cudaStreamOpaque,
            "scratch normalization orphaned shared tmp",
            outError);
#endif
    }

    bool ensure_auto_exposure_buffers(Resources& resources, int meterWidth, int meterHeight, void* cudaStreamOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)meterWidth;
        (void)meterHeight;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        if (meterWidth <= 0 || meterHeight <= 0) {
            outError = "auto-exposure meter dimensions invalid";
            return false;
        }

        const int blockX = 16;
        const int blockY = 16;
        const int gridX = (meterWidth + blockX - 1) / blockX;
        const int gridY = (meterHeight + blockY - 1) / blockY;
        const int neededPartials = gridX * gridY;
        if (neededPartials <= 0) {
            outError = "auto-exposure partial count invalid";
            return false;
        }

        std::lock_guard<std::mutex> lock(resources.m);
        if (!validate_resource_owner_locked(resources, outError, true)) {
            return false;
        }

        if (!resources.autoExposureExposureScale) {
            const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.autoExposureExposureScale), sizeof(float));
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(auto-exposure scale) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                free_auto_exposure(resources);
                return false;
            }
        }
        if (!resources.autoExposureAutoEV) {
            const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.autoExposureAutoEV), sizeof(double));
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(auto-exposure autoEV) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                free_auto_exposure(resources);
                return false;
            }
        }
        if (!resources.autoExposureValid) {
            const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.autoExposureValid), sizeof(int));
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(auto-exposure valid) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                free_auto_exposure(resources);
                return false;
            }
        }

        if (!resources.autoExposureScratch.maxYBits) {
            const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.autoExposureScratch.maxYBits), sizeof(unsigned int));
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(auto-exposure maxYBits) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                free_auto_exposure(resources);
                return false;
            }
        }
        if (!resources.autoExposureScratch.histogram) {
            const size_t bytes = sizeof(unsigned int) * 2048u;
            const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.autoExposureScratch.histogram), bytes);
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(auto-exposure histogram) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                free_auto_exposure(resources);
                return false;
            }
        }

        const bool needWeightsX = resources.autoExposureScratch.weightsXCapacity < meterWidth || !resources.autoExposureScratch.weightsX;
        const bool needWeightsY = resources.autoExposureScratch.weightsYCapacity < meterHeight || !resources.autoExposureScratch.weightsY;
        if (needWeightsX || needWeightsY) {
            if (needWeightsX) {
                if (resources.autoExposureScratch.weightsX) {
                    const size_t oldBytes = static_cast<size_t>(std::max(0, resources.autoExposureScratch.weightsXCapacity)) * sizeof(float);
                    if (!retire_ptr_locked(resources, resources.autoExposureScratch.weightsX, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "auto-exposure weightsX", outError, true)) {
                        return false;
                    }
                    resources.autoExposureScratch.weightsX = nullptr;
                }
                const size_t bytes = static_cast<size_t>(meterWidth) * sizeof(float);
                const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.autoExposureScratch.weightsX), bytes);
                if (err != cudaSuccess) {
                    outError = std::string("cudaMalloc(auto-exposure weightsX) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                    free_auto_exposure(resources);
                    return false;
                }
                resources.autoExposureScratch.weightsXCapacity = meterWidth;
            }
            if (needWeightsY) {
                if (resources.autoExposureScratch.weightsY) {
                    const size_t oldBytes = static_cast<size_t>(std::max(0, resources.autoExposureScratch.weightsYCapacity)) * sizeof(float);
                    if (!retire_ptr_locked(resources, resources.autoExposureScratch.weightsY, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "auto-exposure weightsY", outError, true)) {
                        return false;
                    }
                    resources.autoExposureScratch.weightsY = nullptr;
                }
                const size_t bytes = static_cast<size_t>(meterHeight) * sizeof(float);
                const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.autoExposureScratch.weightsY), bytes);
                if (err != cudaSuccess) {
                    outError = std::string("cudaMalloc(auto-exposure weightsY) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                    free_auto_exposure(resources);
                    return false;
                }
                resources.autoExposureScratch.weightsYCapacity = meterHeight;
            }

            resources.autoExposureScratch.weightsWidth = 0;
            resources.autoExposureScratch.weightsHeight = 0;
        }

        if (resources.autoExposureScratch.partialCapacity < neededPartials) {
            if (resources.autoExposureScratch.partialsA || resources.autoExposureScratch.partialsB) {
                const size_t oldBytes = static_cast<size_t>(std::max(0, resources.autoExposureScratch.partialCapacity)) * sizeof(JuicerCudaAutoExposurePartial);
                if (resources.autoExposureScratch.partialsA) {
                    if (!retire_ptr_locked(resources, resources.autoExposureScratch.partialsA, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "auto-exposure partialsA", outError, true)) {
                        return false;
                    }
                    resources.autoExposureScratch.partialsA = nullptr;
                }
                if (resources.autoExposureScratch.partialsB) {
                    if (!retire_ptr_locked(resources, resources.autoExposureScratch.partialsB, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "auto-exposure partialsB", outError, true)) {
                        return false;
                    }
                    resources.autoExposureScratch.partialsB = nullptr;
                }
                resources.autoExposureScratch.partialCapacity = 0;
            }

            const size_t bytes = static_cast<size_t>(neededPartials) * sizeof(JuicerCudaAutoExposurePartial);
            cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.autoExposureScratch.partialsA), bytes);
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(auto-exposure partialsA) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                free_auto_exposure(resources);
                return false;
            }
            err = cudaMalloc(reinterpret_cast<void**>(&resources.autoExposureScratch.partialsB), bytes);
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(auto-exposure partialsB) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                free_auto_exposure(resources);
                return false;
            }

            resources.autoExposureScratch.partialCapacity = neededPartials;
            resources.autoExposureKeyHash = 0;
            resources.autoExposureSliderEV = std::numeric_limits<double>::quiet_NaN();
        }

        return true;
#endif
    }

    static bool ensure_shared_tmp_plane_locked(Resources& resources, int width, int height, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)width;
        (void)height;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (width <= 0 || height <= 0) {
            outError = std::string(label ? label : "shared tmp") + " dimensions invalid";
            return false;
        }

        const size_t requiredElements = static_cast<size_t>(width) * static_cast<size_t>(height);
        if (resources.sharedTmpPlane &&
            resources.sharedTmpCapacityElements >= requiredElements) {
            resources.sharedTmpWidth = width;
            resources.sharedTmpHeight = height;
            return true;
        }

        if (resources.sharedTmpPlane) {
            const size_t oldBytes = resources.sharedTmpCapacityElements * sizeof(float);
            if (!retire_ptr_locked(resources, resources.sharedTmpPlane, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label ? label : "shared tmp", outError, true)) {
                return false;
            }
            resources.sharedTmpPlane = nullptr;
        }
        resources.sharedTmpWidth = 0;
        resources.sharedTmpHeight = 0;
        resources.sharedTmpCapacityElements = 0;

        const size_t bytes = requiredElements * sizeof(float);
        if (!allocate_scratch_device_ptr_locked(
                resources,
                resources.sharedTmpPlane,
                bytes,
                cudaStreamOpaque,
                label ? label : "shared tmp",
                outError)) {
            resources.sharedTmpPlane = nullptr;
            return false;
        }

        resources.sharedTmpWidth = width;
        resources.sharedTmpHeight = height;
        resources.sharedTmpCapacityElements = requiredElements;
        return true;
#endif
    }

    bool ensure_optics_scratch(Resources& resources, int width, int height, bool needBlurredScratch, bool needAuxScratch, bool needGrainScratch, bool needGrainSharedScratch, bool needGateMask, void* cudaStreamOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)width;
        (void)height;
        (void)needBlurredScratch;
        (void)needAuxScratch;
        (void)needGrainScratch;
        (void)needGateMask;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        if (width <= 0 || height <= 0) {
            outError = "optics scratch dimensions invalid";
            return false;
        }

        std::lock_guard<std::mutex> lock(resources.m);
        if (!validate_resource_owner_locked(resources, outError, true)) {
            return false;
        }

        const size_t requiredElements = static_cast<size_t>(width) * static_cast<size_t>(height);
        const bool dimsMatch = (resources.scannerScratch.width == width && resources.scannerScratch.height == height);
        const bool capacityMatch = resources.scannerScratch.capacityElements >= requiredElements;
        const bool haveBase = resources.scannerScratch.rgbR && resources.scannerScratch.rgbG && resources.scannerScratch.rgbB;

        if (!capacityMatch || !haveBase) {
            if (resources.scannerScratch.rgbR || resources.scannerScratch.rgbG || resources.scannerScratch.rgbB ||
                resources.scannerScratch.blurred || resources.scannerScratch.aux || resources.scannerScratch.grainTmp ||
                resources.scannerScratch.grainTmpShared || resources.scannerScratch.grainTmpMid ||
                resources.scannerScratch.grainTmpCoarse || resources.scannerScratch.gateMask) {
                if (!retire_optics_scratch_locked(resources, resources.scannerScratch, cudaStreamOpaque, "optics scratch", outError)) {
                    return false;
                }
            }
            else {
                free_optics_scratch(resources, resources.scannerScratch, cudaStreamOpaque);
            }

            const size_t bytes = requiredElements * sizeof(float);
            if (!allocate_scratch_device_ptr_locked(
                    resources,
                    resources.scannerScratch.rgbR,
                    bytes,
                    cudaStreamOpaque,
                    "scannerScratch.rgbR",
                    outError)) {
                free_optics_scratch(resources, resources.scannerScratch, cudaStreamOpaque);
                return false;
            }
            if (!allocate_scratch_device_ptr_locked(
                    resources,
                    resources.scannerScratch.rgbG,
                    bytes,
                    cudaStreamOpaque,
                    "scannerScratch.rgbG",
                    outError)) {
                free_optics_scratch(resources, resources.scannerScratch, cudaStreamOpaque);
                return false;
            }
            if (!allocate_scratch_device_ptr_locked(
                    resources,
                    resources.scannerScratch.rgbB,
                    bytes,
                    cudaStreamOpaque,
                    "scannerScratch.rgbB",
                    outError)) {
                free_optics_scratch(resources, resources.scannerScratch, cudaStreamOpaque);
                return false;
            }

            resources.scannerScratch.capacityElements = requiredElements;
        }
        resources.scannerScratch.width = width;
        resources.scannerScratch.height = height;
        if (!dimsMatch) {
            resources.scannerScratch.gateMaskHash = 0;
        }

        if (!ensure_shared_tmp_plane_locked(resources, width, height, cudaStreamOpaque, "shared tmp plane", outError)) {
            free_optics_scratch(resources, resources.scannerScratch, cudaStreamOpaque);
            return false;
        }
        resources.scannerScratch.tmp = resources.sharedTmpPlane;
        const size_t planeBytes = resources.scannerScratch.capacityElements * sizeof(float);

        if (needBlurredScratch) {
            if (!resources.scannerScratch.blurred) {
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.blurred,
                        planeBytes,
                        cudaStreamOpaque,
                        "scannerScratch.blurred",
                        outError)) {
                    return false;
                }
            }
        }
        else {
            if (resources.scannerScratch.blurred) {
                if (!retire_ptr_locked(resources, resources.scannerScratch.blurred, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "unsharp scratch", outError, true)) {
                    return false;
                }
                resources.scannerScratch.blurred = nullptr;
            }
        }

        if (needAuxScratch) {
            if (!resources.scannerScratch.aux) {
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.aux,
                        planeBytes,
                        cudaStreamOpaque,
                        "scannerScratch.aux",
                        outError)) {
                    return false;
                }
            }
        }
        else {
            if (resources.scannerScratch.aux) {
                if (!retire_ptr_locked(resources, resources.scannerScratch.aux, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "grain scratch", outError, true)) {
                    return false;
                }
                resources.scannerScratch.aux = nullptr;
            }
        }

        if (needGrainScratch) {
            if (!resources.scannerScratch.grainTmp) {
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.grainTmp,
                        planeBytes,
                        cudaStreamOpaque,
                        "scannerScratch.grainTmp",
                        outError)) {
                    return false;
                }
            }
            if (!resources.scannerScratch.grainTmpMid) {
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.grainTmpMid,
                        planeBytes,
                        cudaStreamOpaque,
                        "scannerScratch.grainTmpMid",
                        outError)) {
                    return false;
                }
            }
            if (!resources.scannerScratch.grainTmpCoarse) {
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.grainTmpCoarse,
                        planeBytes,
                        cudaStreamOpaque,
                        "scannerScratch.grainTmpCoarse",
                        outError)) {
                    return false;
                }
            }
        }
        else {
            if (resources.scannerScratch.grainTmp) {
                if (!retire_ptr_locked(resources, resources.scannerScratch.grainTmp, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "grain mix scratch", outError, true)) {
                    return false;
                }
                resources.scannerScratch.grainTmp = nullptr;
            }
            if (resources.scannerScratch.grainTmpMid) {
                if (!retire_ptr_locked(resources, resources.scannerScratch.grainTmpMid, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "grain mix mid scratch", outError, true)) {
                    return false;
                }
                resources.scannerScratch.grainTmpMid = nullptr;
            }
            if (resources.scannerScratch.grainTmpCoarse) {
                if (!retire_ptr_locked(resources, resources.scannerScratch.grainTmpCoarse, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "grain mix coarse scratch", outError, true)) {
                    return false;
                }
                resources.scannerScratch.grainTmpCoarse = nullptr;
            }
        }

        if (needGrainSharedScratch) {
            if (!resources.scannerScratch.grainTmpShared) {
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.grainTmpShared,
                        planeBytes,
                        cudaStreamOpaque,
                        "scannerScratch.grainTmpShared",
                        outError)) {
                    return false;
                }
            }
        }
        else {
            if (resources.scannerScratch.grainTmpShared) {
                if (!retire_ptr_locked(resources, resources.scannerScratch.grainTmpShared, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "grain shared scratch", outError, true)) {
                    return false;
                }
                resources.scannerScratch.grainTmpShared = nullptr;
            }
        }

        const int gateWidth = (width + 1) / 2;
        const int gateHeight = (height + 1) / 2;
        if (needGateMask) {
            const size_t requiredGateElements = static_cast<size_t>(gateWidth) * static_cast<size_t>(gateHeight);
            const bool gateDimsMatch = (resources.scannerScratch.gateWidth == gateWidth &&
                resources.scannerScratch.gateHeight == gateHeight);
            const bool gateCapacityMatch =
                resources.scannerScratch.gateMaskCapacityElements >= requiredGateElements;
            if (!resources.scannerScratch.gateMask || !gateCapacityMatch) {
                if (resources.scannerScratch.gateMask) {
                    const size_t oldBytes =
                        resources.scannerScratch.gateMaskCapacityElements * sizeof(float);
                    if (!retire_ptr_locked(resources, resources.scannerScratch.gateMask, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "gate defect mask", outError, true)) {
                        return false;
                    }
                    resources.scannerScratch.gateMask = nullptr;
                }
                const size_t bytes = requiredGateElements * sizeof(float);
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.gateMask,
                        bytes,
                        cudaStreamOpaque,
                        "scannerScratch.gateMask",
                        outError)) {
                    return false;
                }
                resources.scannerScratch.gateMaskCapacityElements = requiredGateElements;
            }
            if (!gateDimsMatch) {
                resources.scannerScratch.gateMaskHash = 0;
            }
            resources.scannerScratch.gateWidth = gateWidth;
            resources.scannerScratch.gateHeight = gateHeight;
        }
        else if (resources.scannerScratch.gateMask) {
            const size_t oldBytes =
                resources.scannerScratch.gateMaskCapacityElements * sizeof(float);
            if (!retire_ptr_locked(resources, resources.scannerScratch.gateMask, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "gate defect mask", outError, true)) {
                return false;
            }
            resources.scannerScratch.gateMask = nullptr;
            resources.scannerScratch.gateWidth = 0;
            resources.scannerScratch.gateHeight = 0;
            resources.scannerScratch.gateMaskCapacityElements = 0;
            resources.scannerScratch.gateMaskHash = 0;
        }

        return true;
#endif
    }

    bool ensure_spatial_dir_scratch(Resources& resources, int width, int height, void* cudaStreamOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)width;
        (void)height;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        if (width <= 0 || height <= 0) {
            outError = "spatial DIR scratch dimensions invalid";
            return false;
        }

        std::lock_guard<std::mutex> lock(resources.m);
        if (!validate_resource_owner_locked(resources, outError, true)) {
            return false;
        }

        Resources::DeviceSpatialDirScratch& scratch = resources.spatialDirScratch;
        const size_t requiredElements = static_cast<size_t>(width) * static_cast<size_t>(height);
        const bool dimsMatch = (scratch.width == width && scratch.height == height);
        const bool capacityMatch = scratch.capacityElements >= requiredElements;
        const bool haveBase = scratch.corrY && scratch.corrM && scratch.corrC;
        if (dimsMatch && capacityMatch && haveBase) {
            if (!ensure_shared_tmp_plane_locked(resources, width, height, cudaStreamOpaque, "shared tmp plane", outError)) {
                return false;
            }
            scratch.tmp = resources.sharedTmpPlane;
            return true;
        }

        if (scratch.corrY || scratch.corrM || scratch.corrC) {
            if (!capacityMatch || !haveBase) {
                if (!retire_spatial_dir_scratch_locked(resources, scratch, cudaStreamOpaque, "spatial DIR scratch", outError)) {
                    return false;
                }
            }
        }

        if (capacityMatch && haveBase) {
            scratch.width = width;
            scratch.height = height;
            if (!ensure_shared_tmp_plane_locked(resources, width, height, cudaStreamOpaque, "shared tmp plane", outError)) {
                return false;
            }
            scratch.tmp = resources.sharedTmpPlane;
            return true;
        }

        const size_t bytes = requiredElements * sizeof(float);
        if (!allocate_scratch_device_ptr_locked(
                resources,
                scratch.corrY,
                bytes,
                cudaStreamOpaque,
                "spatial DIR corrY",
                outError)) {
            free_spatial_dir_scratch(resources, scratch, cudaStreamOpaque);
            return false;
        }
        if (!allocate_scratch_device_ptr_locked(
                resources,
                scratch.corrM,
                bytes,
                cudaStreamOpaque,
                "spatial DIR corrM",
                outError)) {
            free_spatial_dir_scratch(resources, scratch, cudaStreamOpaque);
            return false;
        }
        if (!allocate_scratch_device_ptr_locked(
                resources,
                scratch.corrC,
                bytes,
                cudaStreamOpaque,
                "spatial DIR corrC",
                outError)) {
            free_spatial_dir_scratch(resources, scratch, cudaStreamOpaque);
            return false;
        }
        if (!ensure_shared_tmp_plane_locked(resources, width, height, cudaStreamOpaque, "shared tmp plane", outError)) {
            free_spatial_dir_scratch(resources, scratch, cudaStreamOpaque);
            return false;
        }
        scratch.tmp = resources.sharedTmpPlane;

        scratch.width = width;
        scratch.height = height;
        scratch.capacityElements = requiredElements;
        return true;
#endif
    }

    static bool retire_or_clear_gaussian_kernel_locked(
        Resources& resources,
        Resources::DeviceGaussianKernel& kernel,
        void* cudaStreamOpaque,
        const char* label,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)kernel;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (kernel.weights) {
            if (kernel.sharedKernelId == 0) {
                const std::size_t bytes =
                    static_cast<std::size_t>(std::max(0, kernel.capacity)) * sizeof(float);
                if (!retire_ptr_locked(
                        resources,
                        kernel.weights,
                        bytes,
                        Resources::RetireKind::DeviceFree,
                        cudaStreamOpaque,
                        label,
                        outError)) {
                    return false;
                }
            }
            kernel.weights = nullptr;
        }
        kernel.radius = 0;
        kernel.sigma = 0.0f;
        kernel.capacity = 0;
        kernel.sharedKernelId = 0;
        return true;
#endif
    }

    static bool ensure_shared_gaussian_kernel(
        Resources& resources,
        Resources::DeviceGaussianKernel& kernel,
        float sigma,
        int radius,
        SharedGaussianKind kind,
        void* cudaStreamOpaque,
        const char* label,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)kernel;
        (void)sigma;
        (void)radius;
        (void)kind;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        const bool sigmaOk = std::isfinite(sigma) && sigma > 0.0f;
        if (!sigmaOk || radius <= 0) {
            std::lock_guard<std::mutex> lock(resources.m);
            reap_retire_queue_locked(resources);
            if (!validate_resource_owner_locked(resources, outError, true)) {
                return false;
            }
            return retire_or_clear_gaussian_kernel_locked(
                resources,
                kernel,
                cudaStreamOpaque,
                label,
                outError);
        }

        SharedGaussianKey key{};
        std::uint64_t kernelId = 0;
        {
            std::lock_guard<std::mutex> lock(resources.m);
            reap_retire_queue_locked(resources);
            if (!validate_resource_owner_locked(resources, outError, true)) {
                return false;
            }
            key = make_shared_gaussian_key(resources, kind, radius, sigma);
            kernelId = make_shared_gaussian_id(key);
            const bool same =
                kernel.weights &&
                kernel.sharedKernelId == kernelId &&
                kernel.radius == radius &&
                std::fabs(kernel.sigma - sigma) <= 1e-6f &&
                shared_gaussian_entry_matches_cache(key, kernel);
            if (same) {
                return true;
            }
        }

        std::vector<float> cpuWeights;
        build_gaussian_weights_cpu(radius, sigma, cpuWeights);
        if (cpuWeights.empty()) {
            outError = std::string(label ? label : "gaussian kernel")
                + " weights build failed";
            return false;
        }

        SharedGaussianEntry sharedEntry{};
        if (!acquire_shared_gaussian_entry(
                key,
                radius,
                sigma,
                cpuWeights,
                sharedEntry,
                outError)) {
            return false;
        }

        std::lock_guard<std::mutex> lock(resources.m);
        reap_retire_queue_locked(resources);
        if (!validate_resource_owner_locked(resources, outError, true)) {
            return false;
        }

        key = make_shared_gaussian_key(resources, kind, radius, sigma);
        kernelId = make_shared_gaussian_id(key);
        if (kernelId != sharedEntry.id) {
            if (!acquire_shared_gaussian_entry(
                    key,
                    radius,
                    sigma,
                    cpuWeights,
                    sharedEntry,
                    outError)) {
                return false;
            }
            kernelId = sharedEntry.id;
        }

        const bool same =
            kernel.weights &&
            kernel.sharedKernelId == kernelId &&
            kernel.radius == radius &&
            std::fabs(kernel.sigma - sigma) <= 1e-6f &&
            shared_gaussian_entry_matches_cache(key, kernel);
        if (same) {
            return true;
        }

        if (!retire_or_clear_gaussian_kernel_locked(
                resources,
                kernel,
                cudaStreamOpaque,
                label,
                outError)) {
            return false;
        }

        kernel.weights = sharedEntry.weights;
        kernel.radius = sharedEntry.radius;
        kernel.sigma = sharedEntry.sigma;
        kernel.capacity = 0;
        kernel.sharedKernelId = sharedEntry.id;
        return true;
#endif
    }

    bool ensure_spatial_dir_kernel(Resources& resources, Resources::DeviceGaussianKernel& kernel, float sigma, void* cudaStreamOpaque, std::string& outError) {
        constexpr int kMaxRadius = 75;
        const bool sigmaOk = std::isfinite(sigma) && sigma > 0.0f;
        const int radiusRaw = sigmaOk
            ? std::max(1, static_cast<int>(std::ceil(3.0f * sigma)))
            : 0;
        const int radius = std::min(radiusRaw, kMaxRadius);
        return ensure_shared_gaussian_kernel(
            resources,
            kernel,
            sigma,
            radius,
            SharedGaussianKind::SpatialDir,
            cudaStreamOpaque,
            "spatial DIR kernel",
            outError);
    }

    bool ensure_gaussian_kernel(Resources& resources, Resources::DeviceGaussianKernel& kernel, float sigma, void* cudaStreamOpaque, std::string& outError) {
        constexpr int kMaxRadius = 75;
        const bool sigmaOk = std::isfinite(sigma) && sigma > 0.0f;
        const int radiusRaw = sigmaOk
            ? JuicerGaussian::scipy_gaussian_radius(sigma, 4.0f)
            : 0;
        const int radius = std::min(radiusRaw, kMaxRadius);
        return ensure_shared_gaussian_kernel(
            resources,
            kernel,
            sigma,
            radius,
            SharedGaussianKind::Standard,
            cudaStreamOpaque,
            "gaussian kernel",
            outError);
    }

    bool ensure_halation_kernel(Resources& resources, Resources::DeviceGaussianKernel& kernel, float sigma, void* cudaStreamOpaque, std::string& outError) {
        constexpr int kMaxRadius = 75;
        const bool sigmaOk = std::isfinite(sigma) && sigma > 0.0f;
        const int radiusRaw = sigmaOk
            ? JuicerGaussian::scipy_gaussian_radius(sigma, 7.0f)
            : 0;
        const int radius = std::min(radiusRaw, kMaxRadius);
        return ensure_shared_gaussian_kernel(
            resources,
            kernel,
            sigma,
            radius,
            SharedGaussianKind::Halation,
            cudaStreamOpaque,
            "halation kernel",
            outError);
    }

    void purge_shared_gaussian_kernels_for_context(int deviceId, void* contextOpaque) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (deviceId < 0 || contextOpaque == nullptr) {
            return;
        }

        std::vector<float*> toFree;
        {
            SharedGaussianCacheState& cache = shared_gaussian_cache_state();
            std::lock_guard<std::mutex> lock(cache.mutex);
            for (auto it = cache.byKey.begin(); it != cache.byKey.end();) {
                if (it->first.deviceId == deviceId &&
                    it->first.contextOpaque == contextOpaque) {
                    if (it->second.weights) {
                        toFree.push_back(it->second.weights);
                    }
                    it = cache.byKey.erase(it);
                    continue;
                }
                ++it;
            }
        }

        if (toFree.empty()) {
            return;
        }

        int previousDevice = -1;
        const cudaError_t prevErr = cudaGetDevice(&previousDevice);
        const bool havePreviousDevice = (prevErr == cudaSuccess && previousDevice >= 0);
        const bool needRestore = havePreviousDevice && previousDevice != deviceId;
        (void)cudaSetDevice(deviceId);

        for (float* ptr : toFree) {
            if (ptr) {
                (void)cudaFree(ptr);
            }
        }

        if (needRestore) {
            (void)cudaSetDevice(previousDevice);
        }
#else
        (void)deviceId;
        (void)contextOpaque;
#endif
    }
// Cuda/JuicerCudaResourcesValidation.cpp
//
// Included by JuicerCudaResources.cpp (single-TU split).
    bool validate_density_primitives(Resources& resources, const WorkingState& ws, void* cudaStreamOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)ws;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
#if defined(JUICER_CUDA_VALIDATE_PRIMITIVES) && (JUICER_CUDA_VALIDATE_PRIMITIVES != 0)
        std::lock_guard<std::mutex> lock(resources.m);
        if (resources.validatedBuildCounter == ws.buildCounter && ws.buildCounter != 0) {
            return true;
        }

        // Lightweight parity probe: sample density curves at a few canonical logE values and compare
        // against the CPU implementation (FilmProcessing.h). This is a development-only check and is
        // expected to be compiled out in shipping builds.
        static constexpr float kLogE[5] = { -3.0f, -1.0f, 0.0f, 2.0f, 4.0f };
        float outB[5] = {};
        float outG[5] = {};
        float outR[5] = {};

        cudaError_t err = ::juicer_cuda_probe_density_curve(resources.densB.x, resources.densB.y, resources.densB.n,
            ws.gammaFactorB, kLogE, 5, outB, cudaStreamOpaque);
        if (err != cudaSuccess) {
            outError = std::string("probe densB failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            return false;
        }
        err = ::juicer_cuda_probe_density_curve(resources.densG.x, resources.densG.y, resources.densG.n,
            ws.gammaFactorG, kLogE, 5, outG, cudaStreamOpaque);
        if (err != cudaSuccess) {
            outError = std::string("probe densG failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            return false;
        }
        err = ::juicer_cuda_probe_density_curve(resources.densR.x, resources.densR.y, resources.densR.n,
            ws.gammaFactorR, kLogE, 5, outR, cudaStreamOpaque);
        if (err != cudaSuccess) {
            outError = std::string("probe densR failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            return false;
        }

        float maxAbs = 0.0f;
        for (int i = 0; i < 5; ++i) {
            const float cpuB = Spectral::sample_density_at_logE(ws.densB, kLogE[i], ws.gammaFactorB);
            const float cpuG = Spectral::sample_density_at_logE(ws.densG, kLogE[i], ws.gammaFactorG);
            const float cpuR = Spectral::sample_density_at_logE(ws.densR, kLogE[i], ws.gammaFactorR);

            if (!std::isfinite(cpuB) || !std::isfinite(cpuG) || !std::isfinite(cpuR) ||
                !std::isfinite(outB[i]) || !std::isfinite(outG[i]) || !std::isfinite(outR[i])) {
                outError = "density probe produced non-finite values";
                return false;
            }

            maxAbs = std::max(maxAbs, std::fabs(outB[i] - cpuB));
            maxAbs = std::max(maxAbs, std::fabs(outG[i] - cpuG));
            maxAbs = std::max(maxAbs, std::fabs(outR[i] - cpuR));
        }

        // This primitive is expected to match exactly (same math, float-only) within a tiny epsilon.
        if (!(maxAbs <= 1e-6f)) {
            outError = "density probe mismatch: maxAbs=" + std::to_string(maxAbs);
            return false;
        }

        // Validate +/-inf sanitization before sampling (DevelopFilmStage parity).
        {
            const float kLogEInf[3] = {
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                0.0f
            };
            float outBInf[3] = {};
            float outGInf[3] = {};
            float outRInf[3] = {};

            auto cpu_sanitize_inf = [](float logE, const Spectral::Curve& c) -> float {
                if (std::isfinite(logE) || std::isnan(logE)) {
                    return logE;
                }
                float xmin = 0.0f, xmax = 0.0f;
                const size_t n = c.lambda_nm.size();
                if (n == 0 || c.linear.size() != n) {
                    return logE;
                }
                size_t begin = 0;
                while (begin < n && !std::isfinite(c.lambda_nm[begin])) {
                    ++begin;
                }
                if (begin == n) {
                    return logE;
                }
                size_t end = n - 1;
                while (end > begin && !std::isfinite(c.lambda_nm[end])) {
                    --end;
                }
                xmin = c.lambda_nm[begin];
                xmax = c.lambda_nm[end];
                if (!std::isfinite(xmin) || !std::isfinite(xmax) || !(xmax >= xmin)) {
                    return logE;
                }
                return (logE > 0.0f) ? xmax : xmin;
            };

            cudaError_t err = ::juicer_cuda_probe_density_curve_sanitize_inf(resources.densB.x, resources.densB.y, resources.densB.n,
                ws.gammaFactorB, kLogEInf, 3, outBInf, cudaStreamOpaque);
            if (err != cudaSuccess) {
                outError = std::string("probe densB sanitize_inf failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                return false;
            }
            err = ::juicer_cuda_probe_density_curve_sanitize_inf(resources.densG.x, resources.densG.y, resources.densG.n,
                ws.gammaFactorG, kLogEInf, 3, outGInf, cudaStreamOpaque);
            if (err != cudaSuccess) {
                outError = std::string("probe densG sanitize_inf failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                return false;
            }
            err = ::juicer_cuda_probe_density_curve_sanitize_inf(resources.densR.x, resources.densR.y, resources.densR.n,
                ws.gammaFactorR, kLogEInf, 3, outRInf, cudaStreamOpaque);
            if (err != cudaSuccess) {
                outError = std::string("probe densR sanitize_inf failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                return false;
            }

            float maxDiff = 0.0f;
            for (int i = 0; i < 3; ++i) {
                const float cpuB = Spectral::sample_density_at_logE(ws.densB, cpu_sanitize_inf(kLogEInf[i], ws.densB), ws.gammaFactorB);
                const float cpuG = Spectral::sample_density_at_logE(ws.densG, cpu_sanitize_inf(kLogEInf[i], ws.densG), ws.gammaFactorG);
                const float cpuR = Spectral::sample_density_at_logE(ws.densR, cpu_sanitize_inf(kLogEInf[i], ws.densR), ws.gammaFactorR);
                maxDiff = std::max(maxDiff, std::fabs(outBInf[i] - cpuB));
                maxDiff = std::max(maxDiff, std::fabs(outGInf[i] - cpuG));
                maxDiff = std::max(maxDiff, std::fabs(outRInf[i] - cpuR));
            }
            if (!(maxDiff <= 1e-6f)) {
                outError = "sanitize_inf_logE mismatch: maxAbs=" + std::to_string(maxDiff);
                return false;
            }
        }

        // Validate film log-raw computation parity (DevelopFilmStage::compute_log_raw).
        {
            const float samples[][3] = {
                { 0.184f, 0.184f, 0.184f },
                { -0.5f, 0.0f, 2.0f },
                { std::numeric_limits<float>::quiet_NaN(), 1.0f, std::numeric_limits<float>::infinity() }
            };

            for (const auto& filmRaw : samples) {
                float gpuLogRaw[3] = { 0.0f, 0.0f, 0.0f };
                const cudaError_t err = ::juicer_cuda_probe_film_log_raw(filmRaw, gpuLogRaw, cudaStreamOpaque);
                if (err != cudaSuccess) {
                    outError = std::string("film log-raw probe failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                    return false;
                }

                constexpr float kEps = 1e-10f;
                float cpuLogRaw[3] = {
                    std::log10(std::fmax(filmRaw[0], 0.0f) + kEps),
                    std::log10(std::fmax(filmRaw[1], 0.0f) + kEps),
                    std::log10(std::fmax(filmRaw[2], 0.0f) + kEps)
                };

                float maxDiff = 0.0f;
                for (int c = 0; c < 3; ++c) {
                    if (std::isfinite(cpuLogRaw[c]) && std::isfinite(gpuLogRaw[c])) {
                        maxDiff = std::max(maxDiff, std::fabs(cpuLogRaw[c] - gpuLogRaw[c]));
                    }
                    else if (std::isnan(cpuLogRaw[c]) != std::isnan(gpuLogRaw[c])) {
                        outError = "film log-raw NaN mismatch";
                        return false;
                    }
                    else if (std::isinf(cpuLogRaw[c]) != std::isinf(gpuLogRaw[c])) {
                        outError = "film log-raw inf mismatch";
                        return false;
                    }
                }
                if (!(maxDiff <= 1e-6f)) {
                    outError = "film log-raw mismatch: maxAbs=" + std::to_string(maxDiff);
                    return false;
                }
            }
        }

        // Validate input RGB -> DWG conversion against the CPU implementation.
        {
            const int inputColorSpaceIndex = Spectral::inputColorSpaceToIndex(ws.filmRaw.inputColorSpace);
            const int applyDecode = ws.filmRaw.applyCctfDecoding ? 1 : 0;
            const int applyAdapt = ws.filmRaw.applyInputChromaticAdapt ? 1 : 0;

            const float samplesIn[][3] = {
                { 0.184f, 0.184f, 0.184f },
                { 1.2f, -0.1f, 0.5f },   // includes negative input channel (must match CPU sanitize-only behavior)
                { 0.0f, 0.5f, 2.0f }
            };

            for (const auto& rgbIn : samplesIn) {
                float gpuDWG[3] = { 0.0f, 0.0f, 0.0f };
                const cudaError_t err = ::juicer_cuda_probe_convert_input_to_DWG(
                    rgbIn,
                    inputColorSpaceIndex,
                    applyDecode,
                    applyAdapt,
                    ws.filmRaw.inputRGBToXYZ.m,
                    ws.filmRaw.inputXYZAdapt.m,
                    gpuDWG,
                    cudaStreamOpaque);
                if (err != cudaSuccess) {
                    outError = std::string("convert_input_rgb_to_DWG probe failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                    return false;
                }

                float cpuDWG[3] = { 0.0f, 0.0f, 0.0f };
                Spectral::convert_input_rgb_to_DWG(ws.filmRaw, rgbIn, cpuDWG);

                float maxDiff = 0.0f;
                for (int c = 0; c < 3; ++c) {
                    maxDiff = std::max(maxDiff, std::fabs(gpuDWG[c] - cpuDWG[c]));
                }
                if (!(maxDiff <= 2e-5f)) {
                    outError = "convert_input_rgb_to_DWG mismatch: maxAbs=" + std::to_string(maxDiff);
                    return false;
                }
            }
        }

        // Validate DIR "clamp corrected logE to curve domain" behavior (Couplers::apply_runtime_logE_with_curves clamp_to).
        {
            auto cpu_clamp_to = [](float le, const Spectral::Curve& c) -> float {
                if (c.lambda_nm.empty()) return le;
                const float xmin = c.lambda_nm.front();
                const float xmax = c.lambda_nm.back();
                if (!std::isfinite(le)) return xmin;
                return std::min(std::max(le, xmin), xmax);
            };

            const float samples[] = {
                -1000.0f,
                1000.0f,
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::quiet_NaN(),
                0.0f
            };

            for (float le : samples) {
                float outB = 0.0f, outG = 0.0f, outR = 0.0f;
                cudaError_t err = ::juicer_cuda_probe_clamp_logE_to_curve_domain(resources.densB.x, resources.densB.n, le, &outB, cudaStreamOpaque);
                if (err != cudaSuccess) { outError = std::string("clamp_to densB probe failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)"); return false; }
                err = ::juicer_cuda_probe_clamp_logE_to_curve_domain(resources.densG.x, resources.densG.n, le, &outG, cudaStreamOpaque);
                if (err != cudaSuccess) { outError = std::string("clamp_to densG probe failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)"); return false; }
                err = ::juicer_cuda_probe_clamp_logE_to_curve_domain(resources.densR.x, resources.densR.n, le, &outR, cudaStreamOpaque);
                if (err != cudaSuccess) { outError = std::string("clamp_to densR probe failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)"); return false; }

                const float cpuB = cpu_clamp_to(le, ws.densB);
                const float cpuG = cpu_clamp_to(le, ws.densG);
                const float cpuR = cpu_clamp_to(le, ws.densR);

                auto eq = [](float a, float b) -> bool {
                    if (std::isnan(a) && std::isnan(b)) return true;
                    return a == b;
                };

                if (!eq(outB, cpuB) || !eq(outG, cpuG) || !eq(outR, cpuR)) {
                    outError = "DIR clamp_to mismatch";
                    return false;
                }
            }
        }

        // Validate scan-stage spectral_to_log_xyz primitive (negative medium) against CPU.
        if (resources.scanNegative.tables.K == Spectral::gShape.K &&
            resources.scanNegative.tables.epsC &&
            resources.scanNegative.tables.Ax)
        {
            const double samples[][3] = {
                { 0.0, 0.0, 0.0 },
                { 0.5, 0.5, 0.5 },
                { 1.0, 1.0, 1.0 },
                { 0.1, 0.7, 0.3 }
            };

            for (const auto& D_norm : samples) {
                double gpuLogXYZ[3] = { 0.0, 0.0, 0.0 };
                const cudaError_t err = ::juicer_cuda_probe_scan_spectral_to_log_xyz(
                    D_norm,
                    resources.scanNegative.mediumIsNegative,
                    resources.scanNegative.min_cmy,
                    resources.scanNegative.inv_max_cmy,
                    resources.scanNegative.tables.epsC,
                    resources.scanNegative.tables.epsM,
                    resources.scanNegative.tables.epsY,
                    resources.scanNegative.tables.Ax,
                    resources.scanNegative.tables.Ay,
                    resources.scanNegative.tables.Az,
                    resources.scanNegative.tables.baseMin,
                    resources.scanNegative.tables.K,
                    resources.scanNegative.tables.hasBaseline,
                    resources.scanNegative.tables.invYn,
                    gpuLogXYZ,
                    cudaStreamOpaque);
                if (err != cudaSuccess) {
                    outError = std::string("scan logXYZ probe failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                    return false;
                }

                double cpuLogXYZ[3] = { 0.0, 0.0, 0.0 };
                Scanner::spectral_to_log_xyz(ws.negativeMediumRuntime, D_norm, cpuLogXYZ);

                double maxDiff = 0.0;
                for (int c = 0; c < 3; ++c) {
                    if (!std::isfinite(cpuLogXYZ[c]) || !std::isfinite(gpuLogXYZ[c])) {
                        outError = "scan logXYZ produced non-finite values";
                        return false;
                    }
                    maxDiff = std::max(maxDiff, std::fabs(cpuLogXYZ[c] - gpuLogXYZ[c]));
                }
                if (!(maxDiff <= 1e-4)) {
                    outError = "scan logXYZ mismatch: maxAbs=" + std::to_string(maxDiff);
                    return false;
                }
            }
        }

        // Validate scan-stage spectral_to_log_xyz primitive (print medium) against CPU when available.
        if (ws.printMediumRuntime.tables &&
            resources.scanPrint.tables.K == Spectral::gShape.K &&
            resources.scanPrint.tables.epsC &&
            resources.scanPrint.tables.Ax)
        {
            const double samples[][3] = {
                { 0.0, 0.0, 0.0 },
                { 0.5, 0.5, 0.5 },
                { 1.0, 1.0, 1.0 },
                { 0.1, 0.7, 0.3 }
            };

            for (const auto& D_norm : samples) {
                double gpuLogXYZ[3] = { 0.0, 0.0, 0.0 };
                const cudaError_t err = ::juicer_cuda_probe_scan_spectral_to_log_xyz(
                    D_norm,
                    resources.scanPrint.mediumIsNegative,
                    resources.scanPrint.min_cmy,
                    resources.scanPrint.inv_max_cmy,
                    resources.scanPrint.tables.epsC,
                    resources.scanPrint.tables.epsM,
                    resources.scanPrint.tables.epsY,
                    resources.scanPrint.tables.Ax,
                    resources.scanPrint.tables.Ay,
                    resources.scanPrint.tables.Az,
                    resources.scanPrint.tables.baseMin,
                    resources.scanPrint.tables.K,
                    resources.scanPrint.tables.hasBaseline,
                    resources.scanPrint.tables.invYn,
                    gpuLogXYZ,
                    cudaStreamOpaque);
                if (err != cudaSuccess) {
                    outError = std::string("scan logXYZ (print) probe failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                    return false;
                }

                double cpuLogXYZ[3] = { 0.0, 0.0, 0.0 };
                Scanner::spectral_to_log_xyz(ws.printMediumRuntime, D_norm, cpuLogXYZ);

                double maxDiff = 0.0;
                for (int c = 0; c < 3; ++c) {
                    if (!std::isfinite(cpuLogXYZ[c]) || !std::isfinite(gpuLogXYZ[c])) {
                        outError = "scan logXYZ (print) produced non-finite values";
                        return false;
                    }
                    maxDiff = std::max(maxDiff, std::fabs(cpuLogXYZ[c] - gpuLogXYZ[c]));
                }
                if (!(maxDiff <= 1e-4)) {
                    outError = "scan logXYZ (print) mismatch: maxAbs=" + std::to_string(maxDiff);
                    return false;
                }
            }
        }

        // Validate table-based (S_inv) exposure primitive against CPU (forced non-Hanatos path).
        if (ws.spdReady && resources.tablesAx && resources.tablesAy && resources.tablesAz && resources.tablesK == Spectral::gShape.K) {
            const float samples[][3] = {
                { 0.184f, 0.184f, 0.184f }, // mid-gray
                { 0.9f, 0.1f, 0.1f },       // red-ish
                { 0.05f, 0.2f, 0.9f }       // blue-ish
            };

            for (const auto& rgbDWG : samples) {
                float gpuE[3] = { 0.0f, 0.0f, 0.0f };
                const cudaError_t err = ::juicer_cuda_probe_tables_layer_exposures(
                    rgbDWG,
                    resources.spdSInv,
                    resources.refIllumWhiteXYZ,
                    resources.tablesAx,
                    resources.tablesAy,
                    resources.tablesAz,
                    resources.tablesK,
                    resources.sensB.y,
                    resources.sensG.y,
                    resources.sensR.y,
                    1.0f,
                    1,
                    gpuE,
                    cudaStreamOpaque);
                if (err != cudaSuccess) {
                    outError = std::string("tables exposure probe failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                    return false;
                }

                std::vector<float> Ee;
                Spectral::reconstruct_Ee_from_DWG_RGB_with_tables(rgbDWG, ws.tablesRef, ws.spdSInv, Ee);
                float cpuE[3] = { 0.0f, 0.0f, 0.0f };
                Spectral::layerExposures_from_sceneSPD_with_curves(Ee, ws.sensB, ws.sensG, ws.sensR, cpuE, 1.0f, true);

                float maxDiff = 0.0f;
                for (int c = 0; c < 3; ++c) {
                    maxDiff = std::max(maxDiff, std::fabs(gpuE[c] - cpuE[c]));
                }
                if (!(maxDiff <= 2e-4f)) {
                    outError = "tables exposure mismatch: maxAbs=" + std::to_string(maxDiff);
                    return false;
                }
            }
        }

        // If Hanatos is available and uploaded, validate the SPD→exposure integration primitive on a
        // few representative DWG RGB samples.
        if (resources.hanatosLut && resources.hanatosN > 0) {
            const float refWhite[3] = {
                ws.tablesRef.refIllumWhiteXYZ[0],
                ws.tablesRef.refIllumWhiteXYZ[1],
                ws.tablesRef.refIllumWhiteXYZ[2]
            };
            const float samples[][3] = {
                { 0.184f, 0.184f, 0.184f }, // mid-gray
                { 0.9f, 0.1f, 0.1f },       // red-ish
                { 0.05f, 0.2f, 0.9f }       // blue-ish
            };

            for (const auto& rgbDWG : samples) {
                float gpuE[3] = { 0.0f, 0.0f, 0.0f };
                const cudaError_t err = ::juicer_cuda_probe_hanatos_layer_exposures(
                    rgbDWG,
                    resources.hanatosLut,
                    resources.hanatosN,
                    refWhite,
                    resources.sensB.y,
                    resources.sensG.y,
                    resources.sensR.y,
                    Spectral::kNumSamples,
                    1.0f,
                    0,
                    gpuE,
                    cudaStreamOpaque);
                if (err != cudaSuccess) {
                    outError = std::string("Hanatos exposure probe failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                    return false;
                }

                // CPU reference: match Spectral::rgbDWG_to_layerExposures_from_tables_with_curves
                // semantics for Hanatos path (applyDeltaLambda = false).
                std::vector<float> Ee;
                Spectral::reconstruct_Ee_from_DWG_RGB_hanatos(rgbDWG, Ee, refWhite);
                float cpuE[3] = { 0.0f, 0.0f, 0.0f };
                Spectral::layerExposures_from_sceneSPD_with_curves(Ee, ws.sensB, ws.sensG, ws.sensR, cpuE, 1.0f, false);

                float maxDiff = 0.0f;
                for (int c = 0; c < 3; ++c) {
                    maxDiff = std::max(maxDiff, std::fabs(gpuE[c] - cpuE[c]));
                }
                if (!(maxDiff <= 2e-4f)) {
                    outError = "Hanatos exposure mismatch: maxAbs=" + std::to_string(maxDiff);
                    return false;
                }
            }
        }

        resources.validatedBuildCounter = ws.buildCounter;
        return true;
#else
        (void)resources;
        (void)ws;
        (void)cudaStreamOpaque;
        (void)outError;
        return true;
#endif
#endif
    }

    bool validate_print_primitives(
        Resources& resources,
        const WorkingState& ws,
        const Print::Runtime& prt,
        const Print::Params& prm,
        float midgrayFactor,
        void* cudaStreamOpaque,
        std::string& outError)
    {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)ws;
        (void)prt;
        (void)prm;
        (void)midgrayFactor;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
#if defined(JUICER_CUDA_VALIDATE_PRIMITIVES) && (JUICER_CUDA_VALIDATE_PRIMITIVES != 0)
        // Cache: avoid re-running the (expensive) CPU-vs-GPU probe every frame. This is for debug-only
        // validation; correctness is still enforced when the key changes.
        std::uint64_t paramsHash = Hash::kFnvOffset;
        auto hash_u32 = [&](std::uint32_t v) {
            Hash::hash_bytes_update(paramsHash, &v, sizeof(v));
        };
        auto hash_u64 = [&](std::uint64_t v) {
            Hash::hash_bytes_update(paramsHash, &v, sizeof(v));
        };
        auto hash_f32 = [&](float v) {
            float vv = v;
            if (vv == 0.0f) {
                vv = 0.0f; // canonicalize -0.0f to +0.0f
            }
            std::uint32_t bits = 0;
            std::memcpy(&bits, &vv, sizeof(bits));
            hash_u32(bits);
        };
        auto hash_bool = [&](bool v) {
            const std::uint32_t b = v ? 1U : 0U;
            hash_u32(b);
        };

        hash_u64(ws.buildCounter);
        hash_f32(prm.exposure);
        hash_f32(prm.preflashExposure);
        hash_f32(prm.yFilter);
        hash_f32(prm.mFilter);
        hash_f32(prm.cFilter);
        hash_u64((prt.neutralFilterHash != 0) ? prt.neutralFilterHash : Print::kDefaultNeutralFilterHash);
        hash_bool(prm.exposureCompensationEnabled);
        hash_f32(prm.exposureCompensationScale);
        hash_f32(midgrayFactor);

        {
            std::lock_guard<std::mutex> lock(resources.m);
            if (resources.validatedPrintBuildCounter == ws.buildCounter &&
                resources.validatedPrintParamsHash == paramsHash) {
                outError.clear();
                return true;
            }
        }

        // Ensure current print illuminant (filtered by print params) is available on device.
        {
            std::string illumError;
            if (!ensure_print_illuminant_filtered(resources, ws, prt, prm, cudaStreamOpaque, illumError)) {
                outError = std::string("ensure_print_illuminant_filtered failed: ") + illumError;
                return false;
            }
        }

        static constexpr float kNegCmySamples[][3] = {
            { 0.0f, 0.0f, 0.0f },
            { 0.2f, 0.3f, 0.4f },
            { 0.5f, 0.5f, 0.5f },
            { 1.0f, 0.8f, 0.6f },
            { 2.0f, 1.5f, 1.0f },
            { 3.0f, 3.0f, 3.0f }
        };
        constexpr int kCount = static_cast<int>(sizeof(kNegCmySamples) / sizeof(kNegCmySamples[0]));

        const float kMid = (std::isfinite(midgrayFactor) && midgrayFactor > 0.0f) ? midgrayFactor : 1.0f;

        float cpuOut[kCount * 3] = {};
        Pipeline::PrintPipelineScratch scratch{};
        for (int i = 0; i < kCount; ++i) {
            Pipeline::ExposePrintInputs in{};
            in.printRuntime = &prt;
            in.printParams = &prm;
            in.negativeDensity.v[0] = kNegCmySamples[i][0];
            in.negativeDensity.v[1] = kNegCmySamples[i][1];
            in.negativeDensity.v[2] = kNegCmySamples[i][2];
            in.midgrayFactor = kMid;

            Pipeline::ExposePrintOutputs ex{};
            if (!Pipeline::ExposePrintStage::run(ws, in, ex, scratch)) {
                outError = "CPU ExposePrintStage::run failed";
                return false;
            }

            Pipeline::DevelopPrintInputs din{};
            din.printRuntime = &prt;
            din.printLogRaw = ex.printLogRaw;
            Pipeline::DevelopPrintOutputs dout{};
            if (!Pipeline::DevelopPrintStage::run(din, dout)) {
                outError = "CPU DevelopPrintStage::run failed";
                return false;
            }

            cpuOut[i * 3 + 0] = dout.printDensity.v[0];
            cpuOut[i * 3 + 1] = dout.printDensity.v[1];
            cpuOut[i * 3 + 2] = dout.printDensity.v[2];
        }

        JuicerCuda::PipelineRunParams run{};
        run.printExpose.active = 1;
        run.printExpose.printExposure = prm.exposure;
        run.printExpose.printPreflashExposure = prm.preflashExposure;
        run.printExpose.printMidgrayFactor = kMid;

        float inNeg[kCount * 3] = {};
        for (int i = 0; i < kCount; ++i) {
            inNeg[i * 3 + 0] = kNegCmySamples[i][0];
            inNeg[i * 3 + 1] = kNegCmySamples[i][1];
            inNeg[i * 3 + 2] = kNegCmySamples[i][2];
        }
        float gpuOut[kCount * 3] = {};

        {
            std::lock_guard<std::mutex> lock(resources.m);
            run.printExpose.negTables.epsC = resources.scanNegative.tables.epsC;
            run.printExpose.negTables.epsM = resources.scanNegative.tables.epsM;
            run.printExpose.negTables.epsY = resources.scanNegative.tables.epsY;
            run.printExpose.negTables.Ax = resources.scanNegative.tables.Ax;
            run.printExpose.negTables.Ay = resources.scanNegative.tables.Ay;
            run.printExpose.negTables.Az = resources.scanNegative.tables.Az;
            run.printExpose.negTables.baseMin = resources.scanNegative.tables.baseMin;
            run.printExpose.negTables.K = resources.scanNegative.tables.K;
            run.printExpose.negTables.hasBaseline = resources.scanNegative.tables.hasBaseline;
            run.printExpose.negTables.invYn = resources.scanNegative.tables.invYn;
            run.printExpose.negTables.mediumIsNegative = resources.scanNegative.mediumIsNegative;
            for (int j = 0; j < 3; ++j) {
                run.printExpose.negTables.min_cmy[j] = resources.scanNegative.min_cmy[j];
                run.printExpose.negTables.inv_max_cmy[j] = resources.scanNegative.inv_max_cmy[j];
            }

            run.printExpose.printIllumFiltered = resources.printIllumFiltered;
            run.printExpose.printIllumK = resources.printIllumK;
            run.printExpose.printSensC = { resources.printSensC.x, resources.printSensC.y, resources.printSensC.n, resources.printSensC.domainBegin, resources.printSensC.domainEnd };
            run.printExpose.printSensM = { resources.printSensM.x, resources.printSensM.y, resources.printSensM.n, resources.printSensM.domainBegin, resources.printSensM.domainEnd };
            run.printExpose.printSensY = { resources.printSensY.x, resources.printSensY.y, resources.printSensY.n, resources.printSensY.domainBegin, resources.printSensY.domainEnd };
            run.printDevelop.printDcC = { resources.printDcC.x, resources.printDcC.y, resources.printDcC.n, resources.printDcC.domainBegin, resources.printDcC.domainEnd };
            run.printDevelop.printDcM = { resources.printDcM.x, resources.printDcM.y, resources.printDcM.n, resources.printDcM.domainBegin, resources.printDcM.domainEnd };
            run.printDevelop.printDcY = { resources.printDcY.x, resources.printDcY.y, resources.printDcY.n, resources.printDcY.domainBegin, resources.printDcY.domainEnd };
            run.printDevelop.printGammaC = resources.printGammaC;
            run.printDevelop.printGammaM = resources.printGammaM;
            run.printDevelop.printGammaY = resources.printGammaY;
            for (int j = 0; j < 3; ++j) {
                run.printExpose.printPreflashRaw[j] = resources.printPreflashRaw[j];
            }
        }

        const cudaError_t probeErr = ::juicer_cuda_probe_print_pipeline(inNeg, kCount, &run, gpuOut, cudaStreamOpaque);
        if (probeErr != cudaSuccess) {
            outError = std::string("GPU print pipeline probe failed: ") + (cudaGetErrorString(probeErr) ? cudaGetErrorString(probeErr) : "(unknown)");
            return false;
        }

        float maxAbs = 0.0f;
        for (int i = 0; i < kCount; ++i) {
            const float* cpu = cpuOut + i * 3;
            const float* gpu = gpuOut + i * 3;
            bool mismatch = false;
            for (int c = 0; c < 3; ++c) {
                const float a = cpu[c];
                const float b = gpu[c];
                if (std::isfinite(a) && std::isfinite(b)) {
                    maxAbs = std::max(maxAbs, std::fabs(a - b));
                } else if (!(std::isnan(a) && std::isnan(b))) {
                    mismatch = true;
                }
            }
            if (mismatch) {
                std::ostringstream oss;
                oss << "CUDA print validation non-finite mismatch"
                    << " sample=" << i
                    << " neg=[" << kNegCmySamples[i][0] << "," << kNegCmySamples[i][1] << "," << kNegCmySamples[i][2] << "]"
                    << " cpu=[" << cpu[0] << "," << cpu[1] << "," << cpu[2] << "]"
                    << " gpu=[" << gpu[0] << "," << gpu[1] << "," << gpu[2] << "]"
                    << " printExposure=" << run.printExpose.printExposure
                    << " preflash=" << run.printExpose.printPreflashExposure
                    << " midgray=" << run.printExpose.printMidgrayFactor
                    << " illumK=" << run.printExpose.printIllumK
                    << " illumPtr=" << (run.printExpose.printIllumFiltered ? 1 : 0)
                    << " sensC.n=" << run.printExpose.printSensC.n
                    << " sensM.n=" << run.printExpose.printSensM.n
                    << " sensY.n=" << run.printExpose.printSensY.n
                    << " dcC.n=" << run.printDevelop.printDcC.n
                    << " dcM.n=" << run.printDevelop.printDcM.n
                    << " dcY.n=" << run.printDevelop.printDcY.n;
                JTRACE("CUDA", oss.str());
                outError = "print pipeline probe produced non-finite mismatch";
                return false;
            }
        }

        if (!(maxAbs <= 1e-3f)) {
            outError = "print pipeline mismatch: maxAbs=" + std::to_string(maxAbs);
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(resources.m);
            resources.validatedPrintBuildCounter = ws.buildCounter;
            resources.validatedPrintParamsHash = paramsHash;
        }

        return true;
#else
        (void)resources;
        (void)ws;
        (void)prt;
        (void)prm;
        (void)midgrayFactor;
        (void)cudaStreamOpaque;
        (void)outError;
        return true;
#endif
#endif
    }

} // namespace JuicerCuda
