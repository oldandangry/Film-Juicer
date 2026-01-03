// Cuda/JuicerCudaResources.cpp
//
// WorkingState uploads + primitive validation hooks.
//
#include "Cuda/JuicerCudaResources.h"
#include "Cuda/JuicerCudaPayloads.h"

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
#endif

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <atomic>
#include <sstream>
#include <vector>

namespace JuicerCuda {
namespace Precompute {
    bool build_scan_lut_cpu(const Scanner::ScannerMediumRuntime& medium, std::uint32_t res, std::vector<double>& out, std::string& outError);
    void build_hanatos_integrated_lut_cpu(const Spectral::SpectralContext& ctx, const WorkingState& ws, std::vector<float>& out);
    bool build_print_preflash_raw(const WorkingState& ws, const Print::Runtime& prt, float outRaw[3], int& outShapeK);
}
}

// Implemented in Cuda/JuicerCudaValidation.cu
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

// Implemented in Cuda/JuicerCudaValidation.cu
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

// Implemented in Cuda/JuicerCudaValidation.cu
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

        reap_retire_queue_locked(resources);

        void* retireEventOpaque = nullptr;
        if (!acquire_retire_event_locked(resources, retireEventOpaque, outError)) {
            // Fallback: block and free immediately. This should be rare.
            const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
            const cudaError_t syncErr = cudaStreamSynchronize(stream);
            if (syncErr != cudaSuccess) {
                outError = std::string("cudaStreamSynchronize fallback before ") + label + " free failed: " +
                    (cudaGetErrorString(syncErr) ? cudaGetErrorString(syncErr) : "(unknown)");
                return false;
            }
            if (kind == Resources::RetireKind::DeviceFree) {
                cudaFree(ptr);
            }
            else if (kind == Resources::RetireKind::HostPinnedFree) {
                cudaFreeHost(ptr);
            }
            else if (kind == Resources::RetireKind::EventDestroy) {
                cudaEventDestroy(reinterpret_cast<cudaEvent_t>(ptr));
            }
            return true;
        }

        if (!record_retire_fence_locked(resources, retireEventOpaque, cudaStreamOpaque, label, outError)) {
            // If we can't record the retire fence, destroy the event and fall back to a blocking free.
            cudaEventDestroy(reinterpret_cast<cudaEvent_t>(retireEventOpaque));
            retireEventOpaque = nullptr;

            const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
            const cudaError_t syncErr = cudaStreamSynchronize(stream);
            if (syncErr != cudaSuccess) {
                outError = std::string("cudaStreamSynchronize fallback before ") + label + " free failed: " +
                    (cudaGetErrorString(syncErr) ? cudaGetErrorString(syncErr) : "(unknown)");
                return false;
            }
            if (kind == Resources::RetireKind::DeviceFree) {
                cudaFree(ptr);
            }
            else if (kind == Resources::RetireKind::HostPinnedFree) {
                cudaFreeHost(ptr);
            }
            else if (kind == Resources::RetireKind::EventDestroy) {
                cudaEventDestroy(reinterpret_cast<cudaEvent_t>(ptr));
            }
            return true;
        }

        Resources::RetireEntry e{};
        e.ptr = ptr;
        e.bytes = bytes;
        e.kind = kind;
        e.doneEventOpaque = retireEventOpaque;
        resources.retireQueue.push_back(e);
        resources.retireBytes += bytes;
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

    struct StbnCpuCache {
        std::vector<std::uint8_t> data;
        int width = 512;
        int height = 512;
        int frames = 256;
        bool loaded = false;
        bool valid = false;
    };

    struct WangCpuCache {
        std::vector<std::uint8_t> tiles;
        std::vector<std::uint8_t> lut;
        int width = 0;
        int height = 0;
        int count = 0;
        int colors = 0;
        bool loaded = false;
        bool valid = false;
    };

    static StbnCpuCache& stbn_cache() {
        static StbnCpuCache cache;
        return cache;
    }

    static WangCpuCache& wang_cache() {
        static WangCpuCache cache;
        return cache;
    }

    static std::atomic<bool> gStbnWarned{ false };
    static std::atomic<bool> gWangWarned{ false };

    static bool load_stbn_cpu(StbnCpuCache& cache, std::string& outError) {
        if (cache.loaded) {
            return cache.valid;
        }
        cache.loaded = true;
        cache.valid = false;

        if (gDataDir.empty()) {
            outError = "STBN load failed: data directory missing";
            return false;
        }

        std::filesystem::path path = std::filesystem::path(gDataDir) / "Noise" / "stbn_scalar_512x512x256_u8.bin";
        path.make_preferred();

        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) {
            outError = std::string("STBN load failed: cannot open ") + path.string();
            return false;
        }

        const std::streamsize size = file.tellg();
        if (size <= 0) {
            outError = std::string("STBN load failed: empty file ") + path.string();
            return false;
        }

        const std::size_t expected = static_cast<std::size_t>(cache.width) *
            static_cast<std::size_t>(cache.height) *
            static_cast<std::size_t>(cache.frames);
        if (static_cast<std::size_t>(size) != expected) {
            outError = std::string("STBN load failed: unexpected size for ") + path.string();
            return false;
        }

        cache.data.resize(expected);
        file.seekg(0, std::ios::beg);
        if (!file.read(reinterpret_cast<char*>(cache.data.data()), size)) {
            outError = std::string("STBN load failed: read error for ") + path.string();
            cache.data.clear();
            return false;
        }

        cache.valid = true;
        return true;
    }

    static std::size_t wang_lut_index(int l, int r, int t, int b, int colors) {
        const std::size_t c = static_cast<std::size_t>(colors);
        return (((static_cast<std::size_t>(l) * c + static_cast<std::size_t>(r)) * c +
                  static_cast<std::size_t>(t)) * c +
                static_cast<std::size_t>(b));
    }

    static bool load_wang_cpu(WangCpuCache& cache, std::string& outError) {
        if (cache.loaded) {
            return cache.valid;
        }
        cache.loaded = true;
        cache.valid = false;

        if (gDataDir.empty()) {
            outError = "Wang tiles load failed: data directory missing";
            return false;
        }

        std::filesystem::path base = std::filesystem::path(gDataDir) / "Noise" / "Wang";
        base.make_preferred();
        std::filesystem::path binPath = base / "wang_tiles_256x256x16_u8.bin";
        std::filesystem::path jsonPath = base / "tiles.json";

        if (!std::filesystem::exists(binPath) || !std::filesystem::exists(jsonPath)) {
            outError = std::string("Wang tiles load failed: missing assets under ") + base.string();
            return false;
        }

        std::ifstream jf(jsonPath);
        if (!jf) {
            outError = std::string("Wang tiles load failed: cannot open ") + jsonPath.string();
            return false;
        }

        nlohmann::json root;
        try {
            jf >> root;
        } catch (const std::exception& e) {
            outError = std::string("Wang tiles load failed: invalid JSON ") + e.what();
            return false;
        }

        if (!root.contains("resolution") || !root.contains("tiles") || !root.contains("colors") || !root.contains("mapping")) {
            outError = "Wang tiles load failed: tiles.json missing required fields";
            return false;
        }

        cache.width = root.value("resolution", 0);
        cache.height = cache.width;
        cache.count = root.value("tiles", 0);
        cache.colors = root.value("colors", 0);
        if (cache.width <= 0 || cache.height <= 0 || cache.count <= 0 || cache.colors <= 0) {
            outError = "Wang tiles load failed: invalid metadata in tiles.json";
            return false;
        }

        const std::size_t lutSize = static_cast<std::size_t>(cache.colors) *
            static_cast<std::size_t>(cache.colors) *
            static_cast<std::size_t>(cache.colors) *
            static_cast<std::size_t>(cache.colors);
        cache.lut.assign(lutSize, 0);

        const auto& mapping = root["mapping"];
        if (!mapping.is_array()) {
            outError = "Wang tiles load failed: mapping is not an array";
            return false;
        }

        for (const auto& entry : mapping) {
            if (!entry.contains("index") || !entry.contains("labels")) {
                continue;
            }
            const int idx = entry.value("index", 0);
            const auto& labels = entry["labels"];
            const int l = labels.value("L", 0);
            const int r = labels.value("R", 0);
            const int t = labels.value("T", 0);
            const int b = labels.value("B", 0);
            if (l < 0 || r < 0 || t < 0 || b < 0 ||
                l >= cache.colors || r >= cache.colors || t >= cache.colors || b >= cache.colors) {
                continue;
            }
            const std::size_t lutIndex = wang_lut_index(l, r, t, b, cache.colors);
            if (lutIndex < cache.lut.size() && idx >= 0 && idx < cache.count) {
                cache.lut[lutIndex] = static_cast<std::uint8_t>(idx);
            }
        }

        std::ifstream bin(binPath, std::ios::binary | std::ios::ate);
        if (!bin) {
            outError = std::string("Wang tiles load failed: cannot open ") + binPath.string();
            return false;
        }
        const std::streamsize size = bin.tellg();
        if (size <= 0) {
            outError = std::string("Wang tiles load failed: empty file ") + binPath.string();
            return false;
        }
        const std::size_t expected = static_cast<std::size_t>(cache.width) *
            static_cast<std::size_t>(cache.height) *
            static_cast<std::size_t>(cache.count);
        if (static_cast<std::size_t>(size) != expected) {
            outError = std::string("Wang tiles load failed: unexpected size for ") + binPath.string();
            return false;
        }
        cache.tiles.resize(expected);
        bin.seekg(0, std::ios::beg);
        if (!bin.read(reinterpret_cast<char*>(cache.tiles.data()), size)) {
            outError = std::string("Wang tiles load failed: read error for ") + binPath.string();
            cache.tiles.clear();
            return false;
        }

        cache.valid = true;
        return true;
    }

    static void free_stbn(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (resources.stbnData) {
            cudaFree(resources.stbnData);
            resources.stbnData = nullptr;
        }
#endif
        resources.stbnWidth = 0;
        resources.stbnHeight = 0;
        resources.stbnFrames = 0;
    }

    static void free_wang(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (resources.wangTilesData) {
            cudaFree(resources.wangTilesData);
            resources.wangTilesData = nullptr;
        }
        if (resources.wangLutData) {
            cudaFree(resources.wangLutData);
            resources.wangLutData = nullptr;
        }
#endif
        resources.wangWidth = 0;
        resources.wangHeight = 0;
        resources.wangCount = 0;
        resources.wangColors = 0;
    }

    static void free_scan_error_flag(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (resources.scanErrorFlag) {
            cudaFree(resources.scanErrorFlag);
            resources.scanErrorFlag = nullptr;
        }
        if (resources.scanErrorHost) {
            cudaFreeHost(resources.scanErrorHost);
            resources.scanErrorHost = nullptr;
        }
        if (resources.scanErrorEventOpaque) {
            cudaEvent_t ev = reinterpret_cast<cudaEvent_t>(resources.scanErrorEventOpaque);
            cudaEventDestroy(ev);
            resources.scanErrorEventOpaque = nullptr;
        }
#endif
        resources.scanErrorPending = 0;
    }

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
            cudaFree(k.weights);
            k.weights = nullptr;
        }
#endif
        k.radius = 0;
        k.sigma = 0.0f;
        k.capacity = 0;
    }

    static void free_optics_scratch(Resources::DeviceOpticsScratch& s) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (s.rgbR) { cudaFree(s.rgbR); s.rgbR = nullptr; }
        if (s.rgbG) { cudaFree(s.rgbG); s.rgbG = nullptr; }
        if (s.rgbB) { cudaFree(s.rgbB); s.rgbB = nullptr; }
        if (s.blurred) { cudaFree(s.blurred); s.blurred = nullptr; }
        if (s.aux) { cudaFree(s.aux); s.aux = nullptr; }
        if (s.grainTmp) { cudaFree(s.grainTmp); s.grainTmp = nullptr; }
        if (s.gateMask) { cudaFree(s.gateMask); s.gateMask = nullptr; }
#endif
        s.tmp = nullptr;
        s.width = 0;
        s.height = 0;
        s.gateWidth = 0;
        s.gateHeight = 0;
        s.gateMaskHash = 0;
    }

    static void free_spatial_dir_scratch(Resources::DeviceSpatialDirScratch& s) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (s.corrY) { cudaFree(s.corrY); s.corrY = nullptr; }
        if (s.corrM) { cudaFree(s.corrM); s.corrM = nullptr; }
        if (s.corrC) { cudaFree(s.corrC); s.corrC = nullptr; }
#endif
        s.tmp = nullptr;
        s.width = 0;
        s.height = 0;
    }

    static void free_shared_tmp_plane(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (resources.sharedTmpPlane) {
            cudaFree(resources.sharedTmpPlane);
            resources.sharedTmpPlane = nullptr;
        }
#endif
        resources.sharedTmpWidth = 0;
        resources.sharedTmpHeight = 0;
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
        resources.printIllumShapeK = 0;
        resources.printIllumBuildCounter = 0;
        resources.printIllumCoreHash = 0;
        resources.printIllumRuntimePtr = nullptr;

        resources.printGammaC = 1.0f;
        resources.printGammaM = 1.0f;
        resources.printGammaY = 1.0f;

        resources.printPreflashRaw[0] = resources.printPreflashRaw[1] = resources.printPreflashRaw[2] = 0.0f;
        resources.printPreflashValid = false;
        resources.printPreflashBuildCounter = 0;
        resources.printPreflashRuntimePtr = nullptr;
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
        resources.printIllumShapeK = 0;
        resources.printIllumBuildCounter = 0;
        resources.printIllumCoreHash = 0;
        resources.printIllumRuntimePtr = nullptr;

        resources.printGammaC = 1.0f;
        resources.printGammaM = 1.0f;
        resources.printGammaY = 1.0f;

        resources.printPreflashRaw[0] = resources.printPreflashRaw[1] = resources.printPreflashRaw[2] = 0.0f;
        resources.printPreflashValid = false;
        resources.printPreflashBuildCounter = 0;
        resources.printPreflashRuntimePtr = nullptr;
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
        const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
        err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            outError = std::string("cudaMemcpyAsync(") + label + ") failed: " + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
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

        const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
        err = cudaMemcpyAsync(dst.x, src.lambda_nm.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            outError = std::string("cudaMemcpyAsync(curve.x) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            free_curve(dst);
            return false;
        }
        err = cudaMemcpyAsync(dst.y, src.linear.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            outError = std::string("cudaMemcpyAsync(curve.y) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
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

            cudaError_t err = cudaMemcpyAsync(dst.x, src.lambda_nm.data(), bytes, cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess) {
                outError = std::string("cudaMemcpyAsync(") + label + ".x failed: " + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                return false;
            }
            err = cudaMemcpyAsync(dst.y, src.linear.data(), bytes, cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess) {
                outError = std::string("cudaMemcpyAsync(") + label + ".y failed: " + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
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
        const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
        err = cudaMemcpyAsync(dst.y, src.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            outError = std::string("cudaMemcpyAsync(") + label + ") failed: " + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
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
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        for (auto& entry : baseGraphs) {
            if (entry.execOpaque) {
                cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(entry.execOpaque));
                entry.execOpaque = nullptr;
            }
            if (entry.graphOpaque) {
                cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(entry.graphOpaque));
                entry.graphOpaque = nullptr;
            }
            entry.kernelNodeOpaque = nullptr;
            entry.kernelFuncOpaque = nullptr;
            entry.gridX = 0;
            entry.gridY = 0;
            entry.gridZ = 0;
            entry.blockX = 0;
            entry.blockY = 0;
            entry.blockZ = 0;
            entry.sharedMemBytes = 0;
        }
        baseGraphs.clear();
        baseGraphTick = 0;
#endif

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
        free_optics_scratch(scannerScratch);
        free_gaussian_kernel(spatialDirKernel);
        free_spatial_dir_scratch(spatialDirScratch);
        free_shared_tmp_plane(*this);
        free_stbn(*this);
        free_wang(*this);
        free_print_payloads(*this);
        free_hanatos(*this);
        free_hanatos_integrated(*this);
        free_scan_error_flag(*this);
        free_auto_exposure(*this);
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
            Resources* r = new Resources();
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
            int dev = -1;
            if (cudaGetDevice(&dev) == cudaSuccess) {
                r->deviceId = dev;
            }
#endif
            return r;
        }
        catch (...) {
            return nullptr;
        }
    }

    void destroy(Resources* resources) noexcept {
        delete resources;
    }

    static bool sync_before_rebuild(Resources& resources, void* cudaStreamOpaque, const char* label, std::string& outError);

    bool ensure_uploaded(Resources& resources, const WorkingState& ws, void* cudaStreamOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)ws;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        std::unique_lock<std::mutex> lock(resources.m);
        reap_retire_queue_locked(resources);
        {
            int cur = -1;
            const cudaError_t devErr = cudaGetDevice(&cur);
            if (devErr != cudaSuccess || cur < 0) {
                outError = std::string("cudaGetDevice failed: ") + (cudaGetErrorString(devErr) ? cudaGetErrorString(devErr) : "(unknown)");
                return false;
            }
            if (resources.deviceId < 0) {
                resources.deviceId = cur;
            }
            if (resources.deviceId != cur) {
                outError = "CUDA device mismatch for cached resources";
                return false;
            }
        }

        if (!resources.stbnData) {
            std::string stbnError;
            StbnCpuCache& cache = stbn_cache();
            if (load_stbn_cpu(cache, stbnError)) {
                const std::size_t bytes = cache.data.size();
                if (bytes > 0) {
                    const cudaError_t allocErr = cudaMalloc(reinterpret_cast<void**>(&resources.stbnData), bytes);
                    if (allocErr == cudaSuccess) {
                        const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
                        const cudaError_t copyErr = cudaMemcpyAsync(resources.stbnData, cache.data.data(), bytes, cudaMemcpyHostToDevice, stream);
                        if (copyErr != cudaSuccess) {
                            stbnError = std::string("cudaMemcpyAsync(STBN) failed: ") + (cudaGetErrorString(copyErr) ? cudaGetErrorString(copyErr) : "(unknown)");
                            free_stbn(resources);
                        } else {
                            resources.stbnWidth = cache.width;
                            resources.stbnHeight = cache.height;
                            resources.stbnFrames = cache.frames;
                        }
                    } else {
                        stbnError = std::string("cudaMalloc(STBN) failed: ") + (cudaGetErrorString(allocErr) ? cudaGetErrorString(allocErr) : "(unknown)");
                        free_stbn(resources);
                    }
                }
            }
            if (!stbnError.empty() && !gStbnWarned.exchange(true)) {
                JTRACE("CUDA", stbnError);
            }
        }

        if (!resources.wangTilesData || !resources.wangLutData) {
            if (resources.wangTilesData || resources.wangLutData) {
                free_wang(resources);
            }
            std::string wangError;
            WangCpuCache& cache = wang_cache();
            if (load_wang_cpu(cache, wangError)) {
                const std::size_t tileBytes = cache.tiles.size();
                const std::size_t lutBytes = cache.lut.size();
                if (tileBytes > 0 && lutBytes > 0) {
                    const cudaError_t allocTiles = cudaMalloc(reinterpret_cast<void**>(&resources.wangTilesData), tileBytes);
                    const cudaError_t allocLut = cudaMalloc(reinterpret_cast<void**>(&resources.wangLutData), lutBytes);
                    if (allocTiles == cudaSuccess && allocLut == cudaSuccess) {
                        const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
                        const cudaError_t copyTiles = cudaMemcpyAsync(resources.wangTilesData, cache.tiles.data(), tileBytes, cudaMemcpyHostToDevice, stream);
                        const cudaError_t copyLut = cudaMemcpyAsync(resources.wangLutData, cache.lut.data(), lutBytes, cudaMemcpyHostToDevice, stream);
                        if (copyTiles != cudaSuccess || copyLut != cudaSuccess) {
                            wangError = std::string("cudaMemcpyAsync(Wang) failed: ") +
                                (cudaGetErrorString(copyTiles != cudaSuccess ? copyTiles : copyLut) ? cudaGetErrorString(copyTiles != cudaSuccess ? copyTiles : copyLut) : "(unknown)");
                            free_wang(resources);
                        } else {
                            resources.wangWidth = cache.width;
                            resources.wangHeight = cache.height;
                            resources.wangCount = cache.count;
                            resources.wangColors = cache.colors;
                        }
                    } else {
                        wangError = std::string("cudaMalloc(Wang) failed: ") +
                            (cudaGetErrorString(allocTiles != cudaSuccess ? allocTiles : allocLut) ? cudaGetErrorString(allocTiles != cudaSuccess ? allocTiles : allocLut) : "(unknown)");
                        free_wang(resources);
                    }
                }
            }
            if (!wangError.empty() && !gWangWarned.exchange(true)) {
                JTRACE("CUDA", wangError);
            }
        }

        if (ws.buildCounter == 0) {
            outError = "WorkingState buildCounter is 0";
            return false;
        }

        const std::uint64_t wsCoreHash = ws.coreHash;
        const std::uint64_t wsDirHash = ws.dirHash;
        if (wsCoreHash == 0 || wsDirHash == 0) {
            outError = "WorkingState hash is 0";
            return false;
        }

        const bool coreUpToDate = (resources.uploadedCoreHash != 0) && (resources.uploadedCoreHash == wsCoreHash);
        const bool dirUpToDate = (resources.uploadedDirHash != 0) && (resources.uploadedDirHash == wsDirHash);

        if (coreUpToDate && dirUpToDate) {
            resources.uploadedBuildCounter = ws.buildCounter;
            return true;
        }

        // DIR-only update: avoid a full WorkingState re-upload when only the DIR pre-corrected
        // density curves changed (slider interaction).
        if (coreUpToDate && !dirUpToDate) {
            if (!upload_curve_locked(resources, resources.dirDensB, ws.dirDensB, cudaStreamOpaque, "dirDensB", outError)) return false;
            if (!upload_curve_locked(resources, resources.dirDensG, ws.dirDensG, cudaStreamOpaque, "dirDensG", outError)) return false;
            if (!upload_curve_locked(resources, resources.dirDensR, ws.dirDensR, cudaStreamOpaque, "dirDensR", outError)) return false;

            resources.uploadedDirHash = wsDirHash;
            resources.uploadedBuildCounter = ws.buildCounter;
            return true;
        }

        // Core rebuild required: retire and replace device pointers without blocking sync.
        if (!retire_curve_locked(resources, resources.densB, cudaStreamOpaque, "densB", outError)) return false;
        if (!retire_curve_locked(resources, resources.densG, cudaStreamOpaque, "densG", outError)) return false;
        if (!retire_curve_locked(resources, resources.densR, cudaStreamOpaque, "densR", outError)) return false;
        if (!retire_density_layers_locked(resources, cudaStreamOpaque, "densityCurvesLayers", outError)) return false;
        if (!retire_curve_locked(resources, resources.dirDensB, cudaStreamOpaque, "dirDensB", outError)) return false;
        if (!retire_curve_locked(resources, resources.dirDensG, cudaStreamOpaque, "dirDensG", outError)) return false;
        if (!retire_curve_locked(resources, resources.dirDensR, cudaStreamOpaque, "dirDensR", outError)) return false;
        if (!retire_curve_locked(resources, resources.sensB, cudaStreamOpaque, "sensB", outError)) return false;
        if (!retire_curve_locked(resources, resources.sensG, cudaStreamOpaque, "sensG", outError)) return false;
        if (!retire_curve_locked(resources, resources.sensR, cudaStreamOpaque, "sensR", outError)) return false;
        if (!retire_tables_locked(resources, cudaStreamOpaque, "tables", outError)) return false;
        if (!retire_scan_medium_locked(resources, resources.scanNegative, cudaStreamOpaque, "scanNegative", outError)) return false;
        if (!retire_scan_medium_locked(resources, resources.scanPrint, cudaStreamOpaque, "scanPrint", outError)) return false;
        if (!retire_print_payloads_locked(resources, cudaStreamOpaque, "print payloads", outError)) return false;

        resources.validatedBuildCounter = 0;
        resources.uploadedBuildCounter = 0;
        resources.uploadedCoreHash = 0;
        resources.uploadedDirHash = 0;

        if (!alloc_and_upload_curve(resources.densB, ws.densB, cudaStreamOpaque, outError)) return false;
        if (!alloc_and_upload_curve(resources.densG, ws.densG, cudaStreamOpaque, outError)) return false;
        if (!alloc_and_upload_curve(resources.densR, ws.densR, cudaStreamOpaque, outError)) return false;

        {
            const int nR = static_cast<int>(ws.densR.linear.size());
            const int nG = static_cast<int>(ws.densG.linear.size());
            const int nB = static_cast<int>(ws.densB.linear.size());
            bool wantLayers = ws.hasDensityCurvesLayers;
            bool sizesOk = wantLayers && (nR > 0 && nG > 0 && nB > 0);
            if (sizesOk) {
                for (int layer = 0; layer < 3; ++layer) {
                    sizesOk = sizesOk && (static_cast<int>(ws.densityCurvesLayers[layer][0].size()) == nR);
                    sizesOk = sizesOk && (static_cast<int>(ws.densityCurvesLayers[layer][1].size()) == nG);
                    sizesOk = sizesOk && (static_cast<int>(ws.densityCurvesLayers[layer][2].size()) == nB);
                }
            }
            const bool sameN = (nR == nG && nR == nB);

            if (!sizesOk) {
                free_density_layers(resources);
            }
            else if (!resources.hasDensityCurvesLayers ||
                resources.densityCurvesLayersN != nR ||
                !resources.densityCurvesLayers[0][0]) {
                free_density_layers(resources);
                std::string layersError;
                for (int layer = 0; layer < 3; ++layer) {
                    for (int ch = 0; ch < 3; ++ch) {
                        const int n = (ch == 0) ? nR : (ch == 1 ? nG : nB);
                        if (!alloc_and_upload_array(resources.densityCurvesLayers[layer][ch],
                            ws.densityCurvesLayers[layer][ch].data(),
                            n,
                            cudaStreamOpaque,
                            "grain density layer",
                            layersError))
                        {
                            outError = std::string("upload grain density layers failed: ") + layersError;
                            free_density_layers(resources);
                            return false;
                        }
                    }
                }
                resources.densityCurvesLayersN = sameN ? nR : 0;
                resources.hasDensityCurvesLayers = 1;
            }
        }

        if (!alloc_and_upload_curve(resources.dirDensB, ws.dirDensB, cudaStreamOpaque, outError)) return false;
        if (!alloc_and_upload_curve(resources.dirDensG, ws.dirDensG, cudaStreamOpaque, outError)) return false;
        if (!alloc_and_upload_curve(resources.dirDensR, ws.dirDensR, cudaStreamOpaque, outError)) return false;

        if (!alloc_and_upload_curve(resources.sensB, ws.sensB, cudaStreamOpaque, outError)) return false;
        if (!alloc_and_upload_curve(resources.sensG, ws.sensG, cudaStreamOpaque, outError)) return false;
        if (!alloc_and_upload_curve(resources.sensR, ws.sensR, cudaStreamOpaque, outError)) return false;

        // Upload per-instance Mallett basis tables (Ax/Ay/Az) and keep a host-side copy of S_inv + ref white.
        {
            const int K = ws.tablesRef.K;
            const bool want =
                ws.spdReady &&
                K == Spectral::gShape.K &&
                static_cast<int>(ws.tablesRef.Ax.size()) == K &&
                static_cast<int>(ws.tablesRef.Ay.size()) == K &&
                static_cast<int>(ws.tablesRef.Az.size()) == K;

            if (!want) {
                free_tables(resources);
            } else if (resources.tablesK != K || !resources.tablesAx || !resources.tablesAy || !resources.tablesAz) {
                free_tables(resources);
                if (!alloc_and_upload_array(resources.tablesAx, ws.tablesRef.Ax.data(), K, cudaStreamOpaque, "tablesAx", outError)) { free_tables(resources); return false; }
                if (!alloc_and_upload_array(resources.tablesAy, ws.tablesRef.Ay.data(), K, cudaStreamOpaque, "tablesAy", outError)) { free_tables(resources); return false; }
                if (!alloc_and_upload_array(resources.tablesAz, ws.tablesRef.Az.data(), K, cudaStreamOpaque, "tablesAz", outError)) { free_tables(resources); return false; }
                resources.tablesK = K;
            }

            for (int i = 0; i < 9; ++i) resources.spdSInv[i] = ws.spdSInv[i];
            if (ws.spdReady && ws.tablesRef.K > 0) {
                for (int i = 0; i < 3; ++i) resources.refIllumWhiteXYZ[i] = ws.tablesRef.refIllumWhiteXYZ[i];
            } else {
                for (int i = 0; i < 3; ++i) resources.refIllumWhiteXYZ[i] = ws.filmRaw.refIllumWhiteXYZ[i];
            }
        }

        auto upload_scan_medium = [&](Resources::DeviceScanMedium& dst, const Scanner::ScannerMediumRuntime& medium, std::string& outErrorLocal) -> bool {
            const Spectral::SpectralTables* t = medium.tables;
            if (!t || t->K != Spectral::gShape.K) {
                free_scan_medium(dst);
                return true;
            }
            const int K = t->K;
            const bool arraysOk =
                static_cast<int>(t->epsC.size()) == K &&
                static_cast<int>(t->epsM.size()) == K &&
                static_cast<int>(t->epsY.size()) == K &&
                static_cast<int>(t->Ax.size()) == K &&
                static_cast<int>(t->Ay.size()) == K &&
                static_cast<int>(t->Az.size()) == K &&
                (!t->hasBaseline || static_cast<int>(t->baseMin.size()) == K);
            if (!arraysOk) {
                outErrorLocal = "scan spectral tables missing required arrays";
                return false;
            }

            // Rebuild if size mismatches or not allocated yet.
            if (dst.tables.K != K || !dst.tables.epsC || !dst.tables.Ax) {
                free_scan_medium(dst);
                if (!alloc_and_upload_array(dst.tables.epsC, t->epsC.data(), K, cudaStreamOpaque, "scan.epsC", outErrorLocal)) { free_scan_medium(dst); return false; }
                if (!alloc_and_upload_array(dst.tables.epsM, t->epsM.data(), K, cudaStreamOpaque, "scan.epsM", outErrorLocal)) { free_scan_medium(dst); return false; }
                if (!alloc_and_upload_array(dst.tables.epsY, t->epsY.data(), K, cudaStreamOpaque, "scan.epsY", outErrorLocal)) { free_scan_medium(dst); return false; }
                if (!alloc_and_upload_array(dst.tables.Ax, t->Ax.data(), K, cudaStreamOpaque, "scan.Ax", outErrorLocal)) { free_scan_medium(dst); return false; }
                if (!alloc_and_upload_array(dst.tables.Ay, t->Ay.data(), K, cudaStreamOpaque, "scan.Ay", outErrorLocal)) { free_scan_medium(dst); return false; }
                if (!alloc_and_upload_array(dst.tables.Az, t->Az.data(), K, cudaStreamOpaque, "scan.Az", outErrorLocal)) { free_scan_medium(dst); return false; }
                if (t->hasBaseline) {
                    if (!alloc_and_upload_array(dst.tables.baseMin, t->baseMin.data(), K, cudaStreamOpaque, "scan.baseMin", outErrorLocal)) { free_scan_medium(dst); return false; }
                }

                dst.tables.K = K;
            }

            // Baseline can toggle without changing K; keep device pointer in sync.
            if (t->hasBaseline) {
                if (!dst.tables.baseMin) {
                    if (!alloc_and_upload_array(dst.tables.baseMin, t->baseMin.data(), K, cudaStreamOpaque, "scan.baseMin", outErrorLocal)) { free_scan_medium(dst); return false; }
                }
            }
            else {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
                if (dst.tables.baseMin) {
                    cudaFree(dst.tables.baseMin);
                    dst.tables.baseMin = nullptr;
                }
#endif
            }

            dst.tables.hasBaseline = t->hasBaseline ? 1 : 0;
            dst.tables.invYn = (std::isfinite(t->invYn) && t->invYn > 0.0f) ? t->invYn : 1.0f;

            dst.mediumIsNegative = (medium.medium == Scanner::ScannerMedium::Negative) ? 1 : 0;
            for (int i = 0; i < 3; ++i) {
                dst.min_cmy[i] = medium.range.min_cmy[i];
                dst.inv_max_cmy[i] = (std::isfinite(medium.range.inv_max_cmy[i]) && medium.range.inv_max_cmy[i] > 0.0f)
                    ? medium.range.inv_max_cmy[i]
                    : 1.0f;
            }
            return true;
        };

        {
            std::string scanError;
            if (!upload_scan_medium(resources.scanNegative, ws.negativeMediumRuntime, scanError)) {
                outError = std::string("upload scan negative failed: ") + scanError;
                return false;
            }
            if (!upload_scan_medium(resources.scanPrint, ws.printMediumRuntime, scanError)) {
                outError = std::string("upload scan print failed: ") + scanError;
                return false;
            }
        }

        // Print pipeline: upload payloads when a valid print runtime is present.
        {
            const Print::Runtime* prt = ws.printRT.get();
            if (!prt || !Print::profile_is_valid(prt->profile)) {
                free_print_payloads(resources);
            }
            else {
                const Print::Profile& p = prt->profile;

                // Upload print density curves (logE->D) for C/M/Y.
                auto ensure_print_curve = [&](DeviceCurve& dst, const Spectral::Curve& src, const char* label, std::string& err) -> bool {
                    const int n = static_cast<int>(src.lambda_nm.size());
                    const bool want = n > 1 && src.linear.size() == src.lambda_nm.size();
                    if (!want) {
                        free_curve(dst);
                        return true;
                    }
                    if (dst.n != n || !dst.x || !dst.y) {
                        free_curve(dst);
                        if (!alloc_and_upload_curve(dst, src, cudaStreamOpaque, err)) {
                            err = std::string(label) + ": " + err;
                            return false;
                        }
                    }
                    return true;
                };

                std::string printErr;
                if (!ensure_print_curve(resources.printDcC, p.dcC, "print dcC", printErr)) { outError = printErr; return false; }
                if (!ensure_print_curve(resources.printDcM, p.dcM, "print dcM", printErr)) { outError = printErr; return false; }
                if (!ensure_print_curve(resources.printDcY, p.dcY, "print dcY", printErr)) { outError = printErr; return false; }

                // Upload print paper sensitivities (linear domain, pinned to shape).
                const int K = Spectral::gShape.K;
                const bool sensOk =
                    K > 0 &&
                    static_cast<int>(p.sensC_log.linear.size()) == K &&
                    static_cast<int>(p.sensM_log.linear.size()) == K &&
                    static_cast<int>(p.sensY_log.linear.size()) == K;
                if (!sensOk) {
                    free_curve(resources.printSensC);
                    free_curve(resources.printSensM);
                    free_curve(resources.printSensY);
                }
                else {
                    if (!resources.printSensC.y || resources.printSensC.n != K) {
                        free_curve(resources.printSensC);
                        if (!alloc_and_upload_spectral_samples(resources.printSensC, p.sensC_log.linear, cudaStreamOpaque, "print sensC", outError)) { return false; }
                    }
                    if (!resources.printSensM.y || resources.printSensM.n != K) {
                        free_curve(resources.printSensM);
                        if (!alloc_and_upload_spectral_samples(resources.printSensM, p.sensM_log.linear, cudaStreamOpaque, "print sensM", outError)) { return false; }
                    }
                    if (!resources.printSensY.y || resources.printSensY.n != K) {
                        free_curve(resources.printSensY);
                        if (!alloc_and_upload_spectral_samples(resources.printSensY, p.sensY_log.linear, cudaStreamOpaque, "print sensY", outError)) { return false; }
                    }
                }

                auto gamma_safe = [](float v) -> float {
                    return (std::isfinite(v) && v > 0.0f) ? v : 1.0f;
                };
                resources.printGammaC = gamma_safe(p.gammaFactor[0]);
                resources.printGammaM = gamma_safe(p.gammaFactor[1]);
                resources.printGammaY = gamma_safe(p.gammaFactor[2]);

                // Preflash raw is computed for (y=m=c=0, Dneg=0) and cached per WorkingState/runtime.
                if (!resources.printPreflashValid ||
                    resources.printPreflashBuildCounter != ws.buildCounter ||
                    resources.printPreflashRuntimePtr != prt ||
                    resources.printPreflashShapeK != Spectral::gShape.K) {
                    float preflashRaw[3] = { 0.0f, 0.0f, 0.0f };
                    int shapeK = 0;
                    const bool ok = Precompute::build_print_preflash_raw(ws, *prt, preflashRaw, shapeK);
                    if (!ok) {
                        resources.printPreflashRaw[0] = resources.printPreflashRaw[1] = resources.printPreflashRaw[2] = 0.0f;
                        resources.printPreflashValid = false;
                        resources.printPreflashBuildCounter = ws.buildCounter;
                        resources.printPreflashRuntimePtr = nullptr;
                        resources.printPreflashShapeK = 0;
                    }
                    else {
                        resources.printPreflashRaw[0] = preflashRaw[0];
                        resources.printPreflashRaw[1] = preflashRaw[1];
                        resources.printPreflashRaw[2] = preflashRaw[2];
                        resources.printPreflashValid = true;
                        resources.printPreflashBuildCounter = ws.buildCounter;
                        resources.printPreflashRuntimePtr = prt;
                        resources.printPreflashShapeK = shapeK;
                    }
                }
            }
        }

        // Upload Hanatos LUT if available (uploaded once and reused across WorkingState rebuilds).
        {
            Spectral::SpectralContext& ctx = Spectral::context();
            const bool hanatosAvailable = ctx.hanatosAvailable.load(std::memory_order_acquire);
            const int N = ctx.hanSpectra.size;
            const int K = ctx.hanSpectra.numSamples;
            const bool want = hanatosAvailable && N > 0 && K == Spectral::kNumSamples && !ctx.hanSpectra.data.empty();
            if (!want) {
                // If Hanatos becomes unavailable (asset missing/mismatch), drop the device copy.
                if (resources.hanatosLut) {
                    const size_t count = static_cast<size_t>(resources.hanatosN) * static_cast<size_t>(resources.hanatosN) * static_cast<size_t>(Spectral::kNumSamples);
                    const size_t bytes = count * sizeof(float);
                    if (!retire_ptr_locked(resources, resources.hanatosLut, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "Hanatos LUT", outError)) {
                        return false;
                    }
                    resources.hanatosLut = nullptr;
                    resources.hanatosN = 0;
                }
                if (resources.hanatosLutIntegrated) {
                    const size_t count = static_cast<size_t>(resources.hanatosNIntegrated) * static_cast<size_t>(resources.hanatosNIntegrated) * 4u;
                    const size_t bytes = count * sizeof(float);
                    if (!retire_ptr_locked(resources, resources.hanatosLutIntegrated, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "Hanatos integrated LUT", outError)) {
                        return false;
                    }
                    resources.hanatosLutIntegrated = nullptr;
                    resources.hanatosNIntegrated = 0;
                    resources.hanatosIntegratedBuildCounter = 0;
                }
            } else {
                if (!resources.hanatosLut || resources.hanatosN != N) {
                    const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
                    if (resources.hanatosLut) {
                        const size_t count = static_cast<size_t>(resources.hanatosN) * static_cast<size_t>(resources.hanatosN) * static_cast<size_t>(Spectral::kNumSamples);
                        const size_t bytes = count * sizeof(float);
                        if (!retire_ptr_locked(resources, resources.hanatosLut, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "Hanatos LUT", outError)) {
                            return false;
                        }
                        resources.hanatosLut = nullptr;
                        resources.hanatosN = 0;
                    }

                    const size_t count = static_cast<size_t>(N) * static_cast<size_t>(N) * static_cast<size_t>(K);
                    const size_t bytes = count * sizeof(float);
                    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.hanatosLut), bytes);
                    if (err != cudaSuccess) {
                        outError = std::string("cudaMalloc(Hanatos LUT) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                        free_hanatos(resources);
                        return false;
                    }
                    err = cudaMemcpyAsync(resources.hanatosLut, ctx.hanSpectra.data.data(), bytes, cudaMemcpyHostToDevice, stream);
                    if (err != cudaSuccess) {
                        outError = std::string("cudaMemcpyAsync(Hanatos LUT) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                        free_hanatos(resources);
                        return false;
                    }
                    resources.hanatosN = N;
                }
            }

            // Build + upload the Hanatos LUT preintegrated with per-instance sensitivities.
            const bool sensOk =
                static_cast<int>(ws.sensB.linear.size()) == K &&
                static_cast<int>(ws.sensG.linear.size()) == K &&
                static_cast<int>(ws.sensR.linear.size()) == K;
            const bool wantIntegrated = want && sensOk;
            if (!wantIntegrated) {
                if (resources.hanatosLutIntegrated) {
                    const size_t count = static_cast<size_t>(resources.hanatosNIntegrated) * static_cast<size_t>(resources.hanatosNIntegrated) * 4u;
                    const size_t bytes = count * sizeof(float);
                    if (!retire_ptr_locked(resources, resources.hanatosLutIntegrated, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "Hanatos integrated LUT", outError)) {
                        return false;
                    }
                    resources.hanatosLutIntegrated = nullptr;
                    resources.hanatosNIntegrated = 0;
                    resources.hanatosIntegratedBuildCounter = 0;
                }
            }
            else {
                const bool needAlloc = (!resources.hanatosLutIntegrated || resources.hanatosNIntegrated != N);
                const bool needUpload = needAlloc || resources.hanatosIntegratedBuildCounter != ws.buildCounter;
                if (needUpload) {
                    // Build CPU LUT outside the resources lock.
                    lock.unlock();
                    std::vector<float> cpu;
                    Precompute::build_hanatos_integrated_lut_cpu(ctx, ws, cpu);
                    lock.lock();
                    reap_retire_queue_locked(resources);

                    const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
                    const bool stillNeedAlloc = (!resources.hanatosLutIntegrated || resources.hanatosNIntegrated != N);
                    const bool stillNeedUpload = stillNeedAlloc || resources.hanatosIntegratedBuildCounter != ws.buildCounter;
                    if (stillNeedUpload) {
                        const size_t bytes = cpu.size() * sizeof(float);
                        float* dLut = nullptr;
                        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&dLut), bytes);
                        if (err != cudaSuccess) {
                            outError = std::string("cudaMalloc(Hanatos integrated LUT) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                            return false;
                        }
                        err = cudaMemcpyAsync(dLut, cpu.data(), bytes, cudaMemcpyHostToDevice, stream);
                        if (err != cudaSuccess) {
                            outError = std::string("cudaMemcpyAsync(Hanatos integrated LUT) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                            cudaFree(dLut);
                            return false;
                        }

                        if (resources.hanatosLutIntegrated) {
                            const size_t count = static_cast<size_t>(resources.hanatosNIntegrated) * static_cast<size_t>(resources.hanatosNIntegrated) * 4u;
                            const size_t oldBytes = count * sizeof(float);
                            if (!retire_ptr_locked(resources, resources.hanatosLutIntegrated, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "Hanatos integrated LUT", outError)) {
                                cudaFree(dLut);
                                return false;
                            }
                        }

                        resources.hanatosLutIntegrated = dLut;
                        resources.hanatosNIntegrated = N;
                        resources.hanatosIntegratedBuildCounter = ws.buildCounter;
                    }
                }
            }
        }

        resources.uploadedCoreHash = wsCoreHash;
        resources.uploadedDirHash = wsDirHash;
        resources.uploadedBuildCounter = ws.buildCounter;
        return true;
#endif
    }

    static bool sync_before_rebuild(Resources& resources, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (resources.lastUseEventOpaque) {
            cudaEvent_t ev = reinterpret_cast<cudaEvent_t>(resources.lastUseEventOpaque);
            const cudaError_t evErr = cudaEventSynchronize(ev);
            if (evErr != cudaSuccess) {
                outError = std::string("cudaEventSynchronize before ") + label + " rebuild failed: " + (cudaGetErrorString(evErr) ? cudaGetErrorString(evErr) : "(unknown)");
                return false;
            }
        }
        else {
            const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
            const cudaError_t syncErr = cudaStreamSynchronize(stream);
            if (syncErr != cudaSuccess) {
                outError = std::string("cudaStreamSynchronize before ") + label + " rebuild failed: " + (cudaGetErrorString(syncErr) ? cudaGetErrorString(syncErr) : "(unknown)");
                return false;
            }
        }
        return true;
#endif
    }

    bool ensure_scan_lut(Resources& resources, const WorkingState& ws, bool negativeMedium, void* cudaStreamOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)ws;
        (void)negativeMedium;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        const Scanner::ScannerMediumRuntime& medium = negativeMedium ? ws.negativeMediumRuntime : ws.printMediumRuntime;
        const Scanner::ScannerStaticKey& staticKey = negativeMedium ? ws.negativeStaticKey : ws.printStaticKey;

        if (!medium.tables || medium.tables->K <= 0) {
            outError = "scan LUT build requested but medium tables are unavailable";
            return false;
        }

        const std::uint32_t res = std::clamp(staticKey.lutResolution, 17u, 128u);
        const std::uint64_t expectedHash = Hash::hash_bytes(&staticKey.hash, sizeof(staticKey.hash));
        if (expectedHash == 0) {
            outError = "scan LUT staticKey hash invalid";
            return false;
        }

        Resources::DeviceSpectralLut* dst = negativeMedium ? &resources.scanNegativeLut : &resources.scanPrintLut;
        {
            std::lock_guard<std::mutex> lock(resources.m);
            int cur = -1;
            const cudaError_t devErr = cudaGetDevice(&cur);
            if (devErr != cudaSuccess || cur < 0) {
                outError = std::string("cudaGetDevice failed: ") + (cudaGetErrorString(devErr) ? cudaGetErrorString(devErr) : "(unknown)");
                return false;
            }
            if (resources.deviceId < 0) {
                resources.deviceId = cur;
            }
            if (resources.deviceId != cur) {
                outError = "CUDA device mismatch for cached resources";
                return false;
            }
            if (dst->log2XYZ && dst->res == res && dst->hash == expectedHash) {
                return true;
            }
        }

        // Build CPU LUT first (can overlap with any in-flight GPU work) before we synchronize to
        // safely retire the previous device buffer.
        std::vector<double> cpu;
        if (!Precompute::build_scan_lut_cpu(medium, res, cpu, outError)) {
            return false;
        }

        const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
        {
            std::lock_guard<std::mutex> lock(resources.m);
            int cur = -1;
            const cudaError_t devErr = cudaGetDevice(&cur);
            if (devErr != cudaSuccess || cur < 0) {
                outError = std::string("cudaGetDevice failed: ") + (cudaGetErrorString(devErr) ? cudaGetErrorString(devErr) : "(unknown)");
                return false;
            }
            if (resources.deviceId != cur) {
                outError = "CUDA device mismatch for cached resources";
                return false;
            }
            if (dst->log2XYZ && dst->res == res && dst->hash == expectedHash) {
                return true;
            }

            if (dst->log2XYZ) {
                // Retire the previous LUT without blocking the CPU.
                const size_t count = static_cast<size_t>(dst->res) * static_cast<size_t>(dst->res) * static_cast<size_t>(dst->res) * 3u;
                const size_t bytes = count * sizeof(double);
                if (!retire_ptr_locked(resources, dst->log2XYZ, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "scan LUT", outError)) {
                    return false;
                }
                dst->log2XYZ = nullptr;
                dst->res = 0;
                dst->hash = 0;
            }

            double* dLut = nullptr;
            const size_t bytes = cpu.size() * sizeof(double);
            cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&dLut), bytes);
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(scan LUT) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                free_scan_lut(*dst);
                return false;
            }
            err = cudaMemcpyAsync(dLut, cpu.data(), bytes, cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess) {
                outError = std::string("cudaMemcpyAsync(scan LUT) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                cudaFree(dLut);
                free_scan_lut(*dst);
                return false;
            }

            dst->log2XYZ = dLut;
            dst->res = res;
            dst->hash = expectedHash;
            return true;
        }
#endif
    }

    bool ensure_scan_error_flag(Resources& resources, void* cudaStreamOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        std::lock_guard<std::mutex> lock(resources.m);
        {
            int cur = -1;
            const cudaError_t devErr = cudaGetDevice(&cur);
            if (devErr != cudaSuccess || cur < 0) {
                outError = std::string("cudaGetDevice failed: ") + (cudaGetErrorString(devErr) ? cudaGetErrorString(devErr) : "(unknown)");
                return false;
            }
            if (resources.deviceId < 0) {
                resources.deviceId = cur;
            }
            if (resources.deviceId != cur) {
                outError = "CUDA device mismatch for cached resources";
                return false;
            }
        }

        if (!resources.scanErrorFlag) {
            cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.scanErrorFlag), sizeof(int));
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(scan error flag) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                resources.scanErrorFlag = nullptr;
                return false;
            }
        }
        if (!resources.scanErrorHost) {
            cudaError_t err = cudaMallocHost(reinterpret_cast<void**>(&resources.scanErrorHost), sizeof(int));
            if (err != cudaSuccess) {
                resources.scanErrorHost = nullptr;
            }
        }
        if (!resources.scanErrorEventOpaque) {
            cudaEvent_t ev = nullptr;
            cudaError_t err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
            if (err == cudaSuccess && ev) {
                resources.scanErrorEventOpaque = reinterpret_cast<void*>(ev);
            }
        }
        return true;
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
        {
            int cur = -1;
            const cudaError_t devErr = cudaGetDevice(&cur);
            if (devErr != cudaSuccess || cur < 0) {
                outError = std::string("cudaGetDevice failed: ") + (cudaGetErrorString(devErr) ? cudaGetErrorString(devErr) : "(unknown)");
                return false;
            }
            if (resources.deviceId < 0) {
                resources.deviceId = cur;
            }
            if (resources.deviceId != cur) {
                outError = "CUDA device mismatch for cached resources";
                return false;
            }
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
            const bool needSync = (needWeightsX && resources.autoExposureScratch.weightsX) ||
                (needWeightsY && resources.autoExposureScratch.weightsY);
            if (needSync) {
                if (!sync_before_rebuild(resources, cudaStreamOpaque, "auto-exposure weights", outError)) {
                    return false;
                }
            }
            if (needWeightsX) {
                if (resources.autoExposureScratch.weightsX) {
                    cudaFree(resources.autoExposureScratch.weightsX);
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
                    cudaFree(resources.autoExposureScratch.weightsY);
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
                if (!sync_before_rebuild(resources, cudaStreamOpaque, "auto-exposure", outError)) {
                    return false;
                }
                if (resources.autoExposureScratch.partialsA) {
                    cudaFree(resources.autoExposureScratch.partialsA);
                    resources.autoExposureScratch.partialsA = nullptr;
                }
                if (resources.autoExposureScratch.partialsB) {
                    cudaFree(resources.autoExposureScratch.partialsB);
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

        if (resources.sharedTmpPlane &&
            resources.sharedTmpWidth == width &&
            resources.sharedTmpHeight == height) {
            return true;
        }

        if (resources.sharedTmpPlane) {
            if (!sync_before_rebuild(resources, cudaStreamOpaque, label ? label : "shared tmp", outError)) {
                return false;
            }
            cudaFree(resources.sharedTmpPlane);
            resources.sharedTmpPlane = nullptr;
        }
        resources.sharedTmpWidth = 0;
        resources.sharedTmpHeight = 0;

        const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
        const size_t bytes = n * sizeof(float);
        const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.sharedTmpPlane), bytes);
        if (err != cudaSuccess) {
            outError = std::string("cudaMalloc(") + (label ? label : "shared tmp") + ") failed: " +
                (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            resources.sharedTmpPlane = nullptr;
            return false;
        }

        resources.sharedTmpWidth = width;
        resources.sharedTmpHeight = height;
        return true;
#endif
    }

    bool ensure_optics_scratch(Resources& resources, int width, int height, bool needBlurredScratch, bool needAuxScratch, bool needGrainScratch, bool needGateMask, void* cudaStreamOpaque, std::string& outError) {
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
        {
            int cur = -1;
            const cudaError_t devErr = cudaGetDevice(&cur);
            if (devErr != cudaSuccess || cur < 0) {
                outError = std::string("cudaGetDevice failed: ") + (cudaGetErrorString(devErr) ? cudaGetErrorString(devErr) : "(unknown)");
                return false;
            }
            if (resources.deviceId < 0) {
                resources.deviceId = cur;
            }
            if (resources.deviceId != cur) {
                outError = "CUDA device mismatch for cached resources";
                return false;
            }
        }

        const bool dimsMatch = (resources.scannerScratch.width == width && resources.scannerScratch.height == height);
        const bool haveBase = resources.scannerScratch.rgbR && resources.scannerScratch.rgbG && resources.scannerScratch.rgbB;

        if (!dimsMatch || !haveBase) {
            if (resources.scannerScratch.rgbR || resources.scannerScratch.rgbG || resources.scannerScratch.rgbB ||
                resources.scannerScratch.blurred || resources.scannerScratch.aux || resources.scannerScratch.grainTmp ||
                resources.scannerScratch.gateMask) {
                if (!sync_before_rebuild(resources, cudaStreamOpaque, "optics scratch", outError)) {
                    return false;
                }
            }
            free_optics_scratch(resources.scannerScratch);

            const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
            const size_t bytes = n * sizeof(float);
            cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.scannerScratch.rgbR), bytes);
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(scannerScratch.rgbR) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                free_optics_scratch(resources.scannerScratch);
                return false;
            }
            err = cudaMalloc(reinterpret_cast<void**>(&resources.scannerScratch.rgbG), bytes);
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(scannerScratch.rgbG) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                free_optics_scratch(resources.scannerScratch);
                return false;
            }
            err = cudaMalloc(reinterpret_cast<void**>(&resources.scannerScratch.rgbB), bytes);
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(scannerScratch.rgbB) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                free_optics_scratch(resources.scannerScratch);
                return false;
            }

            resources.scannerScratch.width = width;
            resources.scannerScratch.height = height;
        }

        if (!ensure_shared_tmp_plane_locked(resources, width, height, cudaStreamOpaque, "shared tmp plane", outError)) {
            free_optics_scratch(resources.scannerScratch);
            return false;
        }
        resources.scannerScratch.tmp = resources.sharedTmpPlane;

        if (needBlurredScratch) {
            if (!resources.scannerScratch.blurred) {
                if (!sync_before_rebuild(resources, cudaStreamOpaque, "unsharp scratch", outError)) {
                    return false;
                }
                const size_t n = static_cast<size_t>(resources.scannerScratch.width) * static_cast<size_t>(resources.scannerScratch.height);
                const size_t bytes = n * sizeof(float);
                const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.scannerScratch.blurred), bytes);
                if (err != cudaSuccess) {
                    outError = std::string("cudaMalloc(scannerScratch.blurred) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                    return false;
                }
            }
        }
        else {
            if (resources.scannerScratch.blurred) {
                if (!sync_before_rebuild(resources, cudaStreamOpaque, "unsharp scratch free", outError)) {
                    return false;
                }
                cudaFree(resources.scannerScratch.blurred);
                resources.scannerScratch.blurred = nullptr;
            }
        }

        if (needAuxScratch) {
            if (!resources.scannerScratch.aux) {
                if (!sync_before_rebuild(resources, cudaStreamOpaque, "grain scratch", outError)) {
                    return false;
                }
                const size_t n = static_cast<size_t>(resources.scannerScratch.width) * static_cast<size_t>(resources.scannerScratch.height);
                const size_t bytes = n * sizeof(float);
                const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.scannerScratch.aux), bytes);
                if (err != cudaSuccess) {
                    outError = std::string("cudaMalloc(scannerScratch.aux) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                    return false;
                }
            }
        }
        else {
            if (resources.scannerScratch.aux) {
                if (!sync_before_rebuild(resources, cudaStreamOpaque, "grain scratch free", outError)) {
                    return false;
                }
                cudaFree(resources.scannerScratch.aux);
                resources.scannerScratch.aux = nullptr;
            }
        }

        if (needGrainScratch) {
            if (!resources.scannerScratch.grainTmp) {
                if (!sync_before_rebuild(resources, cudaStreamOpaque, "grain mix scratch", outError)) {
                    return false;
                }
                const size_t n = static_cast<size_t>(resources.scannerScratch.width) * static_cast<size_t>(resources.scannerScratch.height);
                const size_t bytes = n * sizeof(float);
                const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.scannerScratch.grainTmp), bytes);
                if (err != cudaSuccess) {
                    outError = std::string("cudaMalloc(scannerScratch.grainTmp) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                    return false;
                }
            }
        }
        else {
            if (resources.scannerScratch.grainTmp) {
                if (!sync_before_rebuild(resources, cudaStreamOpaque, "grain mix scratch free", outError)) {
                    return false;
                }
                cudaFree(resources.scannerScratch.grainTmp);
                resources.scannerScratch.grainTmp = nullptr;
            }
        }

        const int gateWidth = (width + 1) / 2;
        const int gateHeight = (height + 1) / 2;
        if (needGateMask) {
            const bool gateDimsMatch = (resources.scannerScratch.gateWidth == gateWidth &&
                resources.scannerScratch.gateHeight == gateHeight);
            if (!resources.scannerScratch.gateMask || !gateDimsMatch) {
                if (!sync_before_rebuild(resources, cudaStreamOpaque, "gate defect mask", outError)) {
                    return false;
                }
                if (resources.scannerScratch.gateMask) {
                    cudaFree(resources.scannerScratch.gateMask);
                    resources.scannerScratch.gateMask = nullptr;
                }
                const size_t n = static_cast<size_t>(gateWidth) * static_cast<size_t>(gateHeight);
                const size_t bytes = n * sizeof(float);
                const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.scannerScratch.gateMask), bytes);
                if (err != cudaSuccess) {
                    outError = std::string("cudaMalloc(scannerScratch.gateMask) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                    return false;
                }
                resources.scannerScratch.gateWidth = gateWidth;
                resources.scannerScratch.gateHeight = gateHeight;
                resources.scannerScratch.gateMaskHash = 0;
            }
        }
        else if (resources.scannerScratch.gateMask) {
            if (!sync_before_rebuild(resources, cudaStreamOpaque, "gate defect mask free", outError)) {
                return false;
            }
            cudaFree(resources.scannerScratch.gateMask);
            resources.scannerScratch.gateMask = nullptr;
            resources.scannerScratch.gateWidth = 0;
            resources.scannerScratch.gateHeight = 0;
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
        {
            int cur = -1;
            const cudaError_t devErr = cudaGetDevice(&cur);
            if (devErr != cudaSuccess || cur < 0) {
                outError = std::string("cudaGetDevice failed: ") + (cudaGetErrorString(devErr) ? cudaGetErrorString(devErr) : "(unknown)");
                return false;
            }
            if (resources.deviceId < 0) {
                resources.deviceId = cur;
            }
            if (resources.deviceId != cur) {
                outError = "CUDA device mismatch for cached resources";
                return false;
            }
        }

        Resources::DeviceSpatialDirScratch& scratch = resources.spatialDirScratch;
        const bool haveBase = (scratch.width == width && scratch.height == height && scratch.corrY && scratch.corrM && scratch.corrC);
        if (haveBase) {
            if (!ensure_shared_tmp_plane_locked(resources, width, height, cudaStreamOpaque, "shared tmp plane", outError)) {
                return false;
            }
            scratch.tmp = resources.sharedTmpPlane;
            return true;
        }

        if (scratch.corrY || scratch.corrM || scratch.corrC) {
            if (!sync_before_rebuild(resources, cudaStreamOpaque, "spatial DIR scratch", outError)) {
                return false;
            }
            free_spatial_dir_scratch(scratch);
        }

        const size_t total = static_cast<size_t>(width) * static_cast<size_t>(height);
        const size_t bytes = total * sizeof(float);
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&scratch.corrY), bytes);
        if (err != cudaSuccess) {
            outError = std::string("cudaMalloc(spatial DIR corrY) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            free_spatial_dir_scratch(scratch);
            return false;
        }
        err = cudaMalloc(reinterpret_cast<void**>(&scratch.corrM), bytes);
        if (err != cudaSuccess) {
            outError = std::string("cudaMalloc(spatial DIR corrM) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            free_spatial_dir_scratch(scratch);
            return false;
        }
        err = cudaMalloc(reinterpret_cast<void**>(&scratch.corrC), bytes);
        if (err != cudaSuccess) {
            outError = std::string("cudaMalloc(spatial DIR corrC) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            free_spatial_dir_scratch(scratch);
            return false;
        }
        if (!ensure_shared_tmp_plane_locked(resources, width, height, cudaStreamOpaque, "shared tmp plane", outError)) {
            free_spatial_dir_scratch(scratch);
            return false;
        }
        scratch.tmp = resources.sharedTmpPlane;

        scratch.width = width;
        scratch.height = height;
        return true;
#endif
    }

    bool ensure_spatial_dir_kernel(Resources& resources, Resources::DeviceGaussianKernel& kernel, float sigma, void* cudaStreamOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)kernel;
        (void)sigma;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        constexpr int kMaxRadius = 75;
        constexpr int kMaxCount = 2 * kMaxRadius + 1;

        const bool sigmaOk = (std::isfinite(sigma) && sigma > 0.0f);
        const int radiusRaw = sigmaOk ? std::max(1, static_cast<int>(std::ceil(3.0f * sigma))) : 0;
        const int radius = std::min(radiusRaw, kMaxRadius);
        const bool wantDisable = (!sigmaOk || radius <= 0);

        {
            std::lock_guard<std::mutex> lock(resources.m);
            reap_retire_queue_locked(resources);
            int cur = -1;
            const cudaError_t devErr = cudaGetDevice(&cur);
            if (devErr != cudaSuccess || cur < 0) {
                outError = std::string("cudaGetDevice failed: ") + (cudaGetErrorString(devErr) ? cudaGetErrorString(devErr) : "(unknown)");
                return false;
            }
            if (resources.deviceId < 0) {
                resources.deviceId = cur;
            }
            if (resources.deviceId != cur) {
                outError = "CUDA device mismatch for cached resources";
                return false;
            }

            if (wantDisable) {
                if (kernel.weights) {
                    const size_t bytes = static_cast<size_t>(std::max(0, kernel.capacity)) * sizeof(float);
                    if (!retire_ptr_locked(resources, kernel.weights, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "spatial DIR kernel", outError)) {
                        return false;
                    }
                    kernel.weights = nullptr;
                }
                kernel.radius = 0;
                kernel.sigma = 0.0f;
                kernel.capacity = 0;
                return true;
            }

            const bool same = (kernel.weights && kernel.radius == radius && std::fabs(kernel.sigma - sigma) <= 1e-6f);
            if (same) {
                return true;
            }
        }

        std::vector<float> cpu;
        cpu.resize(static_cast<size_t>(2 * radius + 1));
        const double s2 = static_cast<double>(sigma) * static_cast<double>(sigma) * 2.0;
        double wsum = 0.0;
        for (int i = -radius; i <= radius; ++i) {
            const double w = std::exp(-(static_cast<double>(i * i)) / s2);
            cpu[static_cast<size_t>(i + radius)] = static_cast<float>(w);
            wsum += w;
        }
        const double invW = (wsum != 0.0) ? (1.0 / wsum) : 0.0;
        for (float& w : cpu) {
            w = static_cast<float>(static_cast<double>(w) * invW);
        }

        std::lock_guard<std::mutex> lock(resources.m);
        reap_retire_queue_locked(resources);
        {
            int cur = -1;
            const cudaError_t devErr = cudaGetDevice(&cur);
            if (devErr != cudaSuccess || cur < 0) {
                outError = std::string("cudaGetDevice failed: ") + (cudaGetErrorString(devErr) ? cudaGetErrorString(devErr) : "(unknown)");
                return false;
            }
            if (resources.deviceId < 0) {
                resources.deviceId = cur;
            }
            if (resources.deviceId != cur) {
                outError = "CUDA device mismatch for cached resources";
                return false;
            }
        }

        const bool same = (kernel.weights && kernel.radius == radius && std::fabs(kernel.sigma - sigma) <= 1e-6f);
        if (same) {
            return true;
        }

        const bool overwriting = (kernel.weights != nullptr);
        if (kernel.weights && kernel.capacity < kMaxCount) {
            const size_t bytes = static_cast<size_t>(std::max(0, kernel.capacity)) * sizeof(float);
            if (!retire_ptr_locked(resources, kernel.weights, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "spatial DIR kernel resize", outError)) {
                return false;
            }
            kernel.weights = nullptr;
            kernel.capacity = 0;
        }
        if (!kernel.weights) {
            cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&kernel.weights), static_cast<size_t>(kMaxCount) * sizeof(float));
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(spatial DIR kernel) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                kernel.weights = nullptr;
                kernel.capacity = 0;
                return false;
            }
            kernel.capacity = kMaxCount;
        }

        const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
        if (overwriting && resources.lastUseEventOpaque) {
            const cudaEvent_t lastUseEv = reinterpret_cast<cudaEvent_t>(resources.lastUseEventOpaque);
            const cudaError_t waitErr = cudaStreamWaitEvent(stream, lastUseEv, 0);
            if (waitErr != cudaSuccess) {
                outError = std::string("cudaStreamWaitEvent before spatial DIR kernel update failed: ") + (cudaGetErrorString(waitErr) ? cudaGetErrorString(waitErr) : "(unknown)");
                return false;
            }
        }

        const size_t bytes = cpu.size() * sizeof(float);
        cudaError_t err = cudaMemcpyAsync(kernel.weights, cpu.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            outError = std::string("cudaMemcpyAsync(spatial DIR kernel) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            return false;
        }

        kernel.radius = radius;
        kernel.sigma = sigma;
        return true;
#endif
    }

    bool ensure_gaussian_kernel(Resources& resources, Resources::DeviceGaussianKernel& kernel, float sigma, void* cudaStreamOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)kernel;
        (void)sigma;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        constexpr int kMaxRadius = 75;
        constexpr int kMaxCount = 2 * kMaxRadius + 1;

        const bool sigmaOk = (std::isfinite(sigma) && sigma > 0.0f);
        const int radiusRaw = sigmaOk ? JuicerGaussian::scipy_gaussian_radius(sigma, 4.0f) : 0;
        const int radius = std::min(radiusRaw, kMaxRadius);
        const bool wantDisable = (!sigmaOk || radius <= 0);

        {
            std::lock_guard<std::mutex> lock(resources.m);
            reap_retire_queue_locked(resources);
            int cur = -1;
            const cudaError_t devErr = cudaGetDevice(&cur);
            if (devErr != cudaSuccess || cur < 0) {
                outError = std::string("cudaGetDevice failed: ") + (cudaGetErrorString(devErr) ? cudaGetErrorString(devErr) : "(unknown)");
                return false;
            }
            if (resources.deviceId < 0) {
                resources.deviceId = cur;
            }
            if (resources.deviceId != cur) {
                outError = "CUDA device mismatch for cached resources";
                return false;
            }

            if (wantDisable) {
                if (kernel.weights) {
                    const size_t bytes = static_cast<size_t>(std::max(0, kernel.capacity)) * sizeof(float);
                    if (!retire_ptr_locked(resources, kernel.weights, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "gaussian kernel", outError)) {
                        return false;
                    }
                    kernel.weights = nullptr;
                }
                kernel.radius = 0;
                kernel.sigma = 0.0f;
                kernel.capacity = 0;
                return true;
            }

            const bool same = (kernel.weights && kernel.radius == radius && std::fabs(kernel.sigma - sigma) <= 1e-6f);
            if (same) {
                return true;
            }
        }

        std::vector<float> cpu;
        cpu.resize(static_cast<size_t>(2 * radius + 1));
        const double s2 = static_cast<double>(sigma) * static_cast<double>(sigma) * 2.0;
        double wsum = 0.0;
        for (int i = -radius; i <= radius; ++i) {
            const double w = std::exp(-(static_cast<double>(i * i)) / s2);
            cpu[static_cast<size_t>(i + radius)] = static_cast<float>(w);
            wsum += w;
        }
        const double invW = (wsum != 0.0) ? (1.0 / wsum) : 0.0;
        for (float& w : cpu) {
            w = static_cast<float>(static_cast<double>(w) * invW);
        }

        std::lock_guard<std::mutex> lock(resources.m);
        reap_retire_queue_locked(resources);
        {
            int cur = -1;
            const cudaError_t devErr = cudaGetDevice(&cur);
            if (devErr != cudaSuccess || cur < 0) {
                outError = std::string("cudaGetDevice failed: ") + (cudaGetErrorString(devErr) ? cudaGetErrorString(devErr) : "(unknown)");
                return false;
            }
            if (resources.deviceId < 0) {
                resources.deviceId = cur;
            }
            if (resources.deviceId != cur) {
                outError = "CUDA device mismatch for cached resources";
                return false;
            }
        }

        const bool same = (kernel.weights && kernel.radius == radius && std::fabs(kernel.sigma - sigma) <= 1e-6f);
        if (same) {
            return true;
        }

        // Fixed-capacity allocation: avoid alloc/free churn on animated sigma.
        const bool overwriting = (kernel.weights != nullptr);
        if (kernel.weights && kernel.capacity < kMaxCount) {
            const size_t bytes = static_cast<size_t>(std::max(0, kernel.capacity)) * sizeof(float);
            if (!retire_ptr_locked(resources, kernel.weights, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "gaussian kernel resize", outError)) {
                return false;
            }
            kernel.weights = nullptr;
            kernel.capacity = 0;
        }
        if (!kernel.weights) {
            cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&kernel.weights), static_cast<size_t>(kMaxCount) * sizeof(float));
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(gaussian kernel) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                kernel.weights = nullptr;
                kernel.capacity = 0;
                return false;
            }
            kernel.capacity = kMaxCount;
        }

        const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
        if (overwriting && resources.lastUseEventOpaque) {
            const cudaEvent_t lastUseEv = reinterpret_cast<cudaEvent_t>(resources.lastUseEventOpaque);
            const cudaError_t waitErr = cudaStreamWaitEvent(stream, lastUseEv, 0);
            if (waitErr != cudaSuccess) {
                outError = std::string("cudaStreamWaitEvent before gaussian kernel update failed: ") + (cudaGetErrorString(waitErr) ? cudaGetErrorString(waitErr) : "(unknown)");
                return false;
            }
        }

        const size_t bytes = cpu.size() * sizeof(float);
        cudaError_t err = cudaMemcpyAsync(kernel.weights, cpu.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            outError = std::string("cudaMemcpyAsync(gaussian kernel) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            return false;
        }

        kernel.radius = radius;
        kernel.sigma = sigma;
        return true;
#endif
    }

    bool ensure_halation_kernel(Resources& resources, Resources::DeviceGaussianKernel& kernel, float sigma, void* cudaStreamOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)kernel;
        (void)sigma;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        constexpr int kMaxRadius = 75;
        constexpr int kMaxCount = 2 * kMaxRadius + 1;

        const bool sigmaOk = (std::isfinite(sigma) && sigma > 0.0f);
        const int radiusRaw = sigmaOk ? JuicerGaussian::scipy_gaussian_radius(sigma, 7.0f) : 0;
        const int radius = std::min(radiusRaw, kMaxRadius);
        const bool wantDisable = (!sigmaOk || radius <= 0);

        {
            std::lock_guard<std::mutex> lock(resources.m);
            reap_retire_queue_locked(resources);
            int cur = -1;
            const cudaError_t devErr = cudaGetDevice(&cur);
            if (devErr != cudaSuccess || cur < 0) {
                outError = std::string("cudaGetDevice failed: ") + (cudaGetErrorString(devErr) ? cudaGetErrorString(devErr) : "(unknown)");
                return false;
            }
            if (resources.deviceId < 0) {
                resources.deviceId = cur;
            }
            if (resources.deviceId != cur) {
                outError = "CUDA device mismatch for cached resources";
                return false;
            }

            if (wantDisable) {
                if (kernel.weights) {
                    const size_t bytes = static_cast<size_t>(std::max(0, kernel.capacity)) * sizeof(float);
                    if (!retire_ptr_locked(resources, kernel.weights, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "halation kernel", outError)) {
                        return false;
                    }
                    kernel.weights = nullptr;
                }
                kernel.radius = 0;
                kernel.sigma = 0.0f;
                kernel.capacity = 0;
                return true;
            }

            const bool same = (kernel.weights && kernel.radius == radius && std::fabs(kernel.sigma - sigma) <= 1e-6f);
            if (same) {
                return true;
            }
        }

        std::vector<float> cpu;
        cpu.resize(static_cast<size_t>(2 * radius + 1));
        const double s2 = static_cast<double>(sigma) * static_cast<double>(sigma) * 2.0;
        double wsum = 0.0;
        for (int i = -radius; i <= radius; ++i) {
            const double w = std::exp(-(static_cast<double>(i * i)) / s2);
            cpu[static_cast<size_t>(i + radius)] = static_cast<float>(w);
            wsum += w;
        }
        const double invW = (wsum != 0.0) ? (1.0 / wsum) : 0.0;
        for (float& w : cpu) {
            w = static_cast<float>(static_cast<double>(w) * invW);
        }

        std::lock_guard<std::mutex> lock(resources.m);
        reap_retire_queue_locked(resources);
        {
            int cur = -1;
            const cudaError_t devErr = cudaGetDevice(&cur);
            if (devErr != cudaSuccess || cur < 0) {
                outError = std::string("cudaGetDevice failed: ") + (cudaGetErrorString(devErr) ? cudaGetErrorString(devErr) : "(unknown)");
                return false;
            }
            if (resources.deviceId < 0) {
                resources.deviceId = cur;
            }
            if (resources.deviceId != cur) {
                outError = "CUDA device mismatch for cached resources";
                return false;
            }
        }

        const bool same = (kernel.weights && kernel.radius == radius && std::fabs(kernel.sigma - sigma) <= 1e-6f);
        if (same) {
            return true;
        }

        const bool overwriting = (kernel.weights != nullptr);
        if (kernel.weights && kernel.capacity < kMaxCount) {
            const size_t bytes = static_cast<size_t>(std::max(0, kernel.capacity)) * sizeof(float);
            if (!retire_ptr_locked(resources, kernel.weights, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "halation kernel resize", outError)) {
                return false;
            }
            kernel.weights = nullptr;
            kernel.capacity = 0;
        }
        if (!kernel.weights) {
            cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&kernel.weights), static_cast<size_t>(kMaxCount) * sizeof(float));
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(halation kernel) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                kernel.weights = nullptr;
                kernel.capacity = 0;
                return false;
            }
            kernel.capacity = kMaxCount;
        }

        const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
        if (overwriting && resources.lastUseEventOpaque) {
            const cudaEvent_t lastUseEv = reinterpret_cast<cudaEvent_t>(resources.lastUseEventOpaque);
            const cudaError_t waitErr = cudaStreamWaitEvent(stream, lastUseEv, 0);
            if (waitErr != cudaSuccess) {
                outError = std::string("cudaStreamWaitEvent before halation kernel update failed: ") + (cudaGetErrorString(waitErr) ? cudaGetErrorString(waitErr) : "(unknown)");
                return false;
            }
        }

        const size_t bytes = cpu.size() * sizeof(float);
        cudaError_t err = cudaMemcpyAsync(kernel.weights, cpu.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            outError = std::string("cudaMemcpyAsync(halation kernel) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            return false;
        }

        kernel.radius = radius;
        kernel.sigma = sigma;
        return true;
#endif
    }

    bool ensure_print_illuminant_filtered(
        Resources& resources,
        const WorkingState& ws,
        const Print::Runtime& prt,
        const Print::Params& prm,
        void* cudaStreamOpaque,
        std::string& outError)
    {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)ws;
        (void)prt;
        (void)prm;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        std::lock_guard<std::mutex> lock(resources.m);
        {
            int cur = -1;
            const cudaError_t devErr = cudaGetDevice(&cur);
            if (devErr != cudaSuccess || cur < 0) {
                outError = std::string("cudaGetDevice failed: ") + (cudaGetErrorString(devErr) ? cudaGetErrorString(devErr) : "(unknown)");
                return false;
            }
            if (resources.deviceId < 0) {
                resources.deviceId = cur;
            }
            if (resources.deviceId != cur) {
                outError = "CUDA device mismatch for cached resources";
                return false;
            }
        }

        const int K = Spectral::gShape.K;
        if (K <= 0) {
            outError = "spectral shape invalid";
            return false;
        }

        // Normalize filter step keys for cache parity with ExposePrintStage.
        const float yKey = std::isfinite(prm.yFilter) ? prm.yFilter : 0.0f;
        const float mKey = std::isfinite(prm.mFilter) ? prm.mFilter : 0.0f;
        const float cKey = 0.0f;

        const bool cached =
            resources.printIllumFiltered &&
            resources.printIllumK == K &&
            resources.printIllumShapeK == K &&
            resources.printIllumCoreHash == ws.coreHash &&
            resources.printIllumRuntimePtr == &prt &&
            resources.printIllumYShiftSteps == yKey &&
            resources.printIllumMShiftSteps == mKey &&
            resources.printIllumCShiftSteps == cKey;
        if (cached) {
            return true;
        }

        auto blend = [](float curveVal, float normalizedAmount) -> float {
            const float a = std::isfinite(normalizedAmount) ? normalizedAmount : 0.0f;
            return 1.0f - (1.0f - curveVal) * a;
        };
        auto compose_amount = [](float neutralAmount, float deltaSteps) -> float {
            const float neutral = std::isfinite(neutralAmount)
                ? std::clamp(neutralAmount, 0.0f, 1.0f)
                : 0.0f;
            float ds = std::isfinite(deltaSteps) ? deltaSteps : 0.0f;
            ds = std::clamp(ds, -Print::kEnlargerSteps, Print::kEnlargerSteps);
            const float totalSteps = neutral * Print::kEnlargerSteps + ds;
            return totalSteps / Print::kEnlargerSteps;
        };

        const float yAmount = compose_amount(prt.neutralY, yKey);
        const float mAmount = compose_amount(prt.neutralM, mKey);
        const float cAmount = compose_amount(prt.neutralC, cKey);

        std::vector<float> cpu;
        cpu.resize(static_cast<size_t>(K));
        for (int i = 0; i < K; ++i) {
            const float Ee = (prt.illumEnlarger.linear.size() > static_cast<size_t>(i))
                ? prt.illumEnlarger.linear[static_cast<size_t>(i)]
                : 1.0f;
            const float fY = blend(
                (prt.filterY.linear.size() > static_cast<size_t>(i)) ? prt.filterY.linear[static_cast<size_t>(i)] : 1.0f,
                yAmount);
            const float fM = blend(
                (prt.filterM.linear.size() > static_cast<size_t>(i)) ? prt.filterM.linear[static_cast<size_t>(i)] : 1.0f,
                mAmount);
            const float fC = blend(
                (prt.filterC.linear.size() > static_cast<size_t>(i)) ? prt.filterC.linear[static_cast<size_t>(i)] : 1.0f,
                cAmount);
            cpu[static_cast<size_t>(i)] = Ee * (fY * fM * fC);
        }

        const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

        if (resources.printIllumFiltered) {
            if (!sync_before_rebuild(resources, cudaStreamOpaque, "print illuminant filtered", outError)) {
                return false;
            }
            cudaFree(resources.printIllumFiltered);
            resources.printIllumFiltered = nullptr;
            resources.printIllumK = 0;
        }

        float* dIllum = nullptr;
        const size_t bytes = static_cast<size_t>(K) * sizeof(float);
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&dIllum), bytes);
        if (err != cudaSuccess) {
            outError = std::string("cudaMalloc(print illuminant filtered) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            return false;
        }
        err = cudaMemcpyAsync(dIllum, cpu.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            outError = std::string("cudaMemcpyAsync(print illuminant filtered) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            cudaFree(dIllum);
            return false;
        }

        resources.printIllumFiltered = dIllum;
        resources.printIllumK = K;
        resources.printIllumYShiftSteps = yKey;
        resources.printIllumMShiftSteps = mKey;
        resources.printIllumCShiftSteps = cKey;
        resources.printIllumShapeK = K;
        resources.printIllumBuildCounter = ws.buildCounter;
        resources.printIllumCoreHash = ws.coreHash;
        resources.printIllumRuntimePtr = &prt;
        return true;
#endif
    }

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
                Pipeline::ScanStage::spectral_to_log_xyz(ws.negativeMediumRuntime, D_norm, cpuLogXYZ);

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
                Pipeline::ScanStage::spectral_to_log_xyz(ws.printMediumRuntime, D_norm, cpuLogXYZ);

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

        // Validate Mallett (tables + S_inv) exposure primitive against CPU (forced non-Hanatos path).
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

} // namespace JuicerCuda
