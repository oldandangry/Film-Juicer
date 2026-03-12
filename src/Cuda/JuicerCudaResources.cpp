// Cuda/JuicerCudaResources.cpp
//
// WorkingState uploads + primitive validation hooks.
//
#include "Cuda/JuicerCudaResources.h"
#include "Cuda/JuicerCudaResourcesInternal.h"
#include "Cuda/JuicerCudaPayloads.h"
#include "Cuda/JuicerCudaValidationProbes.h"
#include "Cuda/ResourceManager/JuicerCudaResourceConfig.h"
#include "Cuda/ResourceManager/JuicerCudaResourceKeys.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#include "Cuda/ResourceManager/JuicerCudaResourceState.h"
#include "Cuda/ResourceManager/JuicerCudaResourceTelemetry.h"

#include "FilmProcessing.h"
#include "ColorTransforms.h"
#include "SpectralProcessing.h"
#include "WorkingState.h"
#include "ScanStage.h"
#include "Print.h"
#include "ExposePrintStage.h"
#include "DevelopPrintStage.h"

#include "GaussianSciPy.h"

#include "Logging.h"
#include "Hash.h"
#include "SpectralContext.h"
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
    bool build_scan_lut_cpu(const Scanner::ScannerMediumRuntime& medium, std::uint32_t res, std::vector<double>& out, std::string& outError);
    void build_hanatos_integrated_lut_cpu(const Spectral::SpectralContext& ctx, const WorkingState& ws, std::vector<float>& out);
    bool build_print_preflash_raw(const WorkingState& ws, const Print::Runtime& prt, float outRaw[3], int& outShapeK);
}
}


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

    static bool retire_ptr_locked(Resources& resources, void* ptr, std::size_t bytes, Resources::RetireKind kind, void* cudaStreamOpaque, const char* label, std::string& outError) {
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
        e.doneEventOpaque = retireEventOpaque;
        resources.retireQueue.push_back(e);
        resources.retireBytes += bytes;
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
        resources.densityCurvesLayersN = 0;
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
        const size_t bytes = static_cast<size_t>(std::max(0, resources.densityCurvesLayersN)) * sizeof(float);
        for (int layer = 0; layer < 3; ++layer) {
            for (int ch = 0; ch < 3; ++ch) {
                if (resources.densityCurvesLayers[layer][ch]) {
                    if (!retire_ptr_locked(resources, resources.densityCurvesLayers[layer][ch], bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                        return false;
                    }
                    resources.densityCurvesLayers[layer][ch] = nullptr;
                }
            }
        }
        resources.densityCurvesLayersN = 0;
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

    static bool upload_curve_locked(Resources& resources, DeviceCurve& dst, const Spectral::Curve& src, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)dst;
        (void)src;
        (void)cudaStreamOpaque;
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

        const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
        const size_t bytes = static_cast<size_t>(n) * sizeof(float);

        // If the allocation matches, update in place to avoid alloc/free churn (common during slider scrubs).
        if (dst.x && dst.y && dst.n == n) {
            if (resources.lastUseEventOpaque) {
                const cudaEvent_t lastUseEv = reinterpret_cast<cudaEvent_t>(resources.lastUseEventOpaque);
                const cudaError_t waitErr = cudaStreamWaitEvent(stream, lastUseEv, 0);
                if (waitErr != cudaSuccess) {
                    outError = std::string("cudaStreamWaitEvent before ") + label + " update failed: " +
                        (cudaGetErrorString(waitErr) ? cudaGetErrorString(waitErr) : "(unknown)");
                    return false;
                }
            }

            const char* baseLabel = label ? label : "curve";
            const std::string labelX = std::string(baseLabel) + ".x";
            if (!enqueue_host_to_device_copy(
                    "upload_curve_locked",
                    labelX.c_str(),
                    dst.x,
                    src.lambda_nm.data(),
                    bytes,
                    cudaStreamOpaque,
                    outError)) {
                outError = std::string(baseLabel) + ".x upload failed: " + outError;
                return false;
            }
            const std::string labelY = std::string(baseLabel) + ".y";
            if (!enqueue_host_to_device_copy(
                    "upload_curve_locked",
                    labelY.c_str(),
                    dst.y,
                    src.linear.data(),
                    bytes,
                    cudaStreamOpaque,
                    outError)) {
                outError = std::string(baseLabel) + ".y upload failed: " + outError;
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
            dst.n = n;
            return true;
        }

        DeviceCurve tmp{};
        if (!alloc_and_upload_curve(tmp, src, cudaStreamOpaque, outError)) {
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
            ResourceManager::DeviceContextKey ownerKey{};
            ownerKey.deviceId = ownerDeviceId;
            ownerKey.contextOpaque = ownerContextOpaque;
            managerRetireAccepted =
                ResourceManager::command_retire_context_idle(ownerKey, managerRetireError);
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
    #include "Cuda/JuicerCudaResourcesScratch.cpp"
    #include "Cuda/JuicerCudaResourcesValidation.cpp"

} // namespace JuicerCuda
