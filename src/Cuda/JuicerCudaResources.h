// Cuda/JuicerCudaResources.h
//
// Per-instance CUDA resource cache keyed by WorkingState.{coreHash,dirHash}.
//
// This module intentionally owns only GPU-side mirrors of CPU WorkingState data (curves/tables/etc).
// The render path remains responsible for gating unsupported features (e.g. auto-exposure) until
// they are ported to CUDA.
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#include "Cuda/JuicerCudaAutoExposure.h"

struct WorkingState;
namespace Print {
    struct Runtime;
    struct Params;
}

namespace JuicerCuda {

    struct DeviceCurve {
        float* x = nullptr;
        float* y = nullptr;
        int n = 0;
        // Inclusive finite-domain indices into x[]/y[] (precomputed on CPU).
        int domainBegin = 0;
        int domainEnd = 0;
    };

    struct Resources {
        std::mutex m;
        // Serializes per-frame submission (ensure_* + kernel launches + record_use) to keep lastUseEvent
        // ordering correct across streams without blocking the CPU.
        std::mutex submitMutex;
        int deviceId = -1;
        std::uint64_t uploadedBuildCounter = 0;
        std::uint64_t uploadedCoreHash = 0;
        std::uint64_t uploadedDirHash = 0;
        std::uint64_t validatedBuildCounter = 0;
        std::uint64_t validatedPrintBuildCounter = 0;
        std::uint64_t validatedPrintParamsHash = 0;

        // Opaque CUDA event (cudaEvent_t) recorded on the stream after enqueuing work that
        // uses this resource set. Used to safely retire/rebuild buffers across streams.
        void* lastUseEventOpaque = nullptr;

        enum class RetireKind : int {
            DeviceFree = 0,
            HostPinnedFree = 1,
            EventDestroy = 2
        };

        struct RetireEntry {
            void* ptr = nullptr;
            std::size_t bytes = 0;
            RetireKind kind = RetireKind::DeviceFree;
            void* doneEventOpaque = nullptr; // cudaEvent_t recorded once for this entry.
        };

        // Deferred frees to avoid blocking synchronize/free in hot paths.
        std::vector<RetireEntry> retireQueue;
        std::vector<void*> retireEventPoolOpaque; // cudaEvent_t pool (cudaEventDisableTiming)
        std::size_t retireBytes = 0;

        DeviceCurve densB;
        DeviceCurve densG;
        DeviceCurve densR;
        float* densityCurvesLayers[3][3] = { {nullptr, nullptr, nullptr},
                                             {nullptr, nullptr, nullptr},
                                             {nullptr, nullptr, nullptr} };
        int densityCurvesLayersN = 0;
        int hasDensityCurvesLayers = 0;

        DeviceCurve dirDensB;
        DeviceCurve dirDensG;
        DeviceCurve dirDensR;

        DeviceCurve sensB;
        DeviceCurve sensG;
        DeviceCurve sensR;

        // SPD reconstruction (Mallett 2019 basis via per-instance tables + S_inv).
        float* tablesAx = nullptr;
        float* tablesAy = nullptr;
        float* tablesAz = nullptr;
        int tablesK = 0;
        float spdSInv[9] = { 1,0,0, 0,1,0, 0,0,1 };
        float refIllumWhiteXYZ[3] = { 0.950455f, 1.0f, 1.089058f };

        struct DeviceSpectralTables {
            float* epsC = nullptr;
            float* epsM = nullptr;
            float* epsY = nullptr;
            float* Ax = nullptr;
            float* Ay = nullptr;
            float* Az = nullptr;
            float* baseMin = nullptr;
            int K = 0;
            int hasBaseline = 0;
            float invYn = 1.0f;
        };

        struct DeviceScanMedium {
            DeviceSpectralTables tables;
            int mediumIsNegative = 1;
            float min_cmy[3] = { 0.0f, 0.0f, 0.0f };
            float inv_max_cmy[3] = { 1.0f, 1.0f, 1.0f };
        };

        DeviceScanMedium scanNegative;
        DeviceScanMedium scanPrint;

        struct DeviceSpectralLut {
            double* log2XYZ = nullptr; // layout: ((z*res + y)*res + x) * 3 + c
            std::uint32_t res = 0;
            std::uint64_t hash = 0;
        };

        DeviceSpectralLut scanNegativeLut;
        DeviceSpectralLut scanPrintLut;

        struct DeviceGaussianKernel {
            float* weights = nullptr;
            int radius = 0;
            float sigma = 0.0f;
            int capacity = 0;
        };

        // Shared single-plane W×H float scratch used as a blur/unsharp intermediate.
        // Spatial DIR and scanner optics reuse this to reduce peak VRAM.
        float* sharedTmpPlane = nullptr;
        int sharedTmpWidth = 0;
        int sharedTmpHeight = 0;

    struct DeviceOpticsScratch {
        float* rgbR = nullptr;
        float* rgbG = nullptr;
        float* rgbB = nullptr;
        float* tmp = nullptr;
        float* blurred = nullptr;
        float* aux = nullptr;
        float* grainTmp = nullptr;
        float* grainTmpMid = nullptr;
        float* grainTmpCoarse = nullptr;
        float* gateMask = nullptr;
        int width = 0;
        int height = 0;
        int gateWidth = 0;
        int gateHeight = 0;
        std::uint64_t gateMaskHash = 0;
    };

        struct DeviceSpatialDirScratch {
            float* corrY = nullptr;
            float* corrM = nullptr;
            float* corrC = nullptr;
            float* tmp = nullptr;
            int width = 0;
            int height = 0;
        };

        DeviceGaussianKernel scannerLensBlurKernel;
        DeviceGaussianKernel scannerUnsharpKernel;
        DeviceGaussianKernel scannerGlareKernel;
        DeviceGaussianKernel grainBlurKernel;
        DeviceGaussianKernel grainBlurKernelMid;
        DeviceGaussianKernel grainBlurKernelCoarse;
        DeviceGaussianKernel grainDyeKernel[3][3];
        DeviceGaussianKernel halationKernel[3];
        DeviceGaussianKernel halationScatterKernel[3];
        DeviceOpticsScratch scannerScratch;
        DeviceGaussianKernel spatialDirKernel;
        DeviceSpatialDirScratch spatialDirScratch;

        std::uint8_t* stbnData = nullptr;
        int stbnWidth = 0;
        int stbnHeight = 0;
        int stbnFrames = 0;
        std::uint8_t* wangTilesData = nullptr;
        std::uint8_t* wangLutData = nullptr;
        int wangWidth = 0;
        int wangHeight = 0;
        int wangCount = 0;
        int wangColors = 0;

        // Print pipeline (PrintBypass=false) payloads.
        DeviceCurve printDcC;
        DeviceCurve printDcM;
        DeviceCurve printDcY;

        DeviceCurve printSensC;
        DeviceCurve printSensM;
        DeviceCurve printSensY;

        float printGammaC = 1.0f;
        float printGammaM = 1.0f;
        float printGammaY = 1.0f;

        float printPreflashRaw[3] = { 0.0f, 0.0f, 0.0f };
        bool printPreflashValid = false;
        std::uint64_t printPreflashBuildCounter = 0;
        const void* printPreflashRuntimePtr = nullptr;
        int printPreflashShapeK = 0;

        // Cached enlarger illuminant filtered by dichroic Y/M/C for the current print params.
        float* printIllumFiltered = nullptr;
        int printIllumK = 0;
        float printIllumYShiftSteps = 0.0f;
        float printIllumMShiftSteps = 0.0f;
        float printIllumCShiftSteps = 0.0f;
        int printIllumShapeK = 0;
        std::uint64_t printIllumBuildCounter = 0;
        std::uint64_t printIllumCoreHash = 0;
        const void* printIllumRuntimePtr = nullptr;

        // Hanatos LUT (process-global on CPU, uploaded on demand).
        // Layout matches NpySpectraLUT: ((x*N + y) * K + k), K=81.
        float* hanatosLut = nullptr;
        int hanatosN = 0;

        // Hanatos LUT preintegrated with per-instance sensitivities.
        // Layout: ((x*N + y) * 4 + c), c=0..2 (RGB), c=3 unused/padding.
        float* hanatosLutIntegrated = nullptr;
        int hanatosNIntegrated = 0;
        std::uint64_t hanatosIntegratedBuildCounter = 0;

        // Device-side flag for scan-stage non-finite detection (set by kernels).
        int* scanErrorFlag = nullptr;
        int* scanErrorHost = nullptr;
        void* scanErrorEventOpaque = nullptr;
        int scanErrorPending = 0;

        struct DeviceAutoExposureScratch {
            JuicerCudaAutoExposurePartial* partialsA = nullptr;
            JuicerCudaAutoExposurePartial* partialsB = nullptr;
            int partialCapacity = 0;
            unsigned int* maxYBits = nullptr;
            unsigned int* histogram = nullptr;
            float* weightsX = nullptr;
            float* weightsY = nullptr;
            int weightsXCapacity = 0;
            int weightsYCapacity = 0;
            int weightsWidth = 0;
            int weightsHeight = 0;
        };

        DeviceAutoExposureScratch autoExposureScratch;
        float* autoExposureExposureScale = nullptr;
        double* autoExposureAutoEV = nullptr;
        int* autoExposureValid = nullptr;
        std::uint64_t autoExposureKeyHash = 0;
        double autoExposureSliderEV = std::numeric_limits<double>::quiet_NaN();

        // CUDA Graph caches (Tier 1): base path graphs to reduce CPU submission overhead.
        // Stored as opaque CUDA handles to keep headers CUDA-free.
        struct BaseGraphKey {
            int width = 0;
            int height = 0;
            int nComponents = 0;
            int renderMode = 0; // 0=NegativeOnly, 1=Print
        };
        struct BaseGraphEntry {
            BaseGraphKey key{};
            void* graphOpaque = nullptr; // cudaGraph_t
            void* execOpaque = nullptr; // cudaGraphExec_t
            void* kernelNodeOpaque = nullptr; // cudaGraphNode_t (single kernel node)
            void* kernelFuncOpaque = nullptr; // cudaKernelNodeParams::func
            unsigned int gridX = 0;
            unsigned int gridY = 0;
            unsigned int gridZ = 0;
            unsigned int blockX = 0;
            unsigned int blockY = 0;
            unsigned int blockZ = 0;
            unsigned int sharedMemBytes = 0;
            std::uint64_t lastUseTick = 0;
        };
        std::vector<BaseGraphEntry> baseGraphs;
        std::uint64_t baseGraphTick = 0;

        Resources() = default;
        Resources(const Resources&) = delete;
        Resources& operator=(const Resources&) = delete;

        ~Resources();
    };

    Resources* create() noexcept;
    void destroy(Resources* resources) noexcept;

    // Uploads WorkingState curves when buildCounter changes; returns false on failure with outError filled.
    bool ensure_uploaded(Resources& resources, const WorkingState& ws, void* cudaStreamOpaque, std::string& outError);

    // Builds + uploads scan-stage log2(XYZ) LUT on demand (Mitchell cubic sampler parity with CPU).
    bool ensure_scan_lut(Resources& resources, const WorkingState& ws, bool negativeMedium, void* cudaStreamOpaque, std::string& outError);

    // Ensures the scan error flag device buffer is allocated.
    bool ensure_scan_error_flag(Resources& resources, void* cudaStreamOpaque, std::string& outError);

    // Allocates scratch + state buffers used by CUDA auto-exposure metering.
    bool ensure_auto_exposure_buffers(Resources& resources, int meterWidth, int meterHeight, void* cudaStreamOpaque, std::string& outError);

    // Allocates scratch buffers used by scanner optics (blur/unsharp/glare/grain) when active.
    bool ensure_optics_scratch(Resources& resources, int width, int height, bool needBlurredScratch, bool needAuxScratch, bool needGrainScratch, bool needGateMask, void* cudaStreamOpaque, std::string& outError);

    // Allocates scratch buffers used by spatial DIR diffusion (corr planes + temp).
    bool ensure_spatial_dir_scratch(Resources& resources, int width, int height, void* cudaStreamOpaque, std::string& outError);

    // Builds and uploads spatial DIR Gaussian kernel (truncate=3.0, radius cap=75).
    bool ensure_spatial_dir_kernel(Resources& resources, Resources::DeviceGaussianKernel& kernel, float sigma, void* cudaStreamOpaque, std::string& outError);

    // Builds and uploads a SciPy-compatible Gaussian kernel (truncate=4.0, radius clamp=75) for optics.
    bool ensure_gaussian_kernel(Resources& resources, Resources::DeviceGaussianKernel& kernel, float sigma, void* cudaStreamOpaque, std::string& outError);

    // Builds and uploads a SciPy-compatible Gaussian kernel (truncate=7.0, radius clamp=75) for halation.
    bool ensure_halation_kernel(Resources& resources, Resources::DeviceGaussianKernel& kernel, float sigma, void* cudaStreamOpaque, std::string& outError);

    // Print pipeline: builds + uploads the enlarger illuminant filtered by the current print params (Y/M/C filters).
    bool ensure_print_illuminant_filtered(
        Resources& resources,
        const WorkingState& ws,
        const Print::Runtime& prt,
        const Print::Params& prm,
        void* cudaStreamOpaque,
        std::string& outError);

    // Optional debug validation of primitives (kept here to avoid a separate JUICER_TESTS harness).
    bool validate_density_primitives(Resources& resources, const WorkingState& ws, void* cudaStreamOpaque, std::string& outError);
    bool validate_print_primitives(
        Resources& resources,
        const WorkingState& ws,
        const Print::Runtime& prt,
        const Print::Params& prm,
        float midgrayFactor,
        void* cudaStreamOpaque,
        std::string& outError);

    // Records a "last use" event on the given stream to allow safe rebuilds without global sync.
    void record_use(Resources& resources, void* cudaStreamOpaque) noexcept;

} // namespace JuicerCuda
