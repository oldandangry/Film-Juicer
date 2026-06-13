// Cuda/JuicerCudaResources.h
//
// Per-context CUDA resource storage and helper payloads served through ProcessRoot and
// ResourceManager preparation commands. Durable residency is owned by Root/context slots;
// frame-local mutable work is accessed through PreparedCudaFrame leases.
//
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "Cuda/ResourceManager/JuicerCudaResourceCore.h"
#include "RenderRecipe.h"

struct WorkingState;
namespace Spectral {
    struct FilmRawConfig;
    struct SpectralTables;
} // namespace Spectral
namespace Scanner {
    struct ColorRuntime;
    struct ScannerSpectralLutDescriptor;
} // namespace Scanner
namespace Print {
    struct Runtime;
    struct Params;
} // namespace Print

struct JuicerCudaAutoExposurePartial {
    static constexpr int kLaneCount = 4;
    double sumY[kLaneCount];
    double sumW[kLaneCount];
};

// Device-side scratch buffers for metering. All pointers are CUDA device pointers.
struct JuicerCudaAutoExposureScratch {
    JuicerCudaAutoExposurePartial* partialsA = nullptr;
    JuicerCudaAutoExposurePartial* partialsB = nullptr;
    int partialCapacity = 0;
    unsigned int* maxYBits = nullptr;
    unsigned int* histogram = nullptr; // 2048 bins
    float* weightsX = nullptr; // meterWidth floats
    float* weightsY = nullptr; // meterHeight floats
};

// Device-side outputs for metering. All pointers are CUDA device pointers.
struct JuicerCudaAutoExposureDeviceState {
    float* exposureScale = nullptr; // float scalar
    double* autoEV = nullptr;       // double scalar
    int* valid = nullptr;           // int scalar (0/1)
};

namespace JuicerCuda {

    struct AutoExposurePreviewDescriptor {
        enum class Sampling : std::uint8_t {
            NearestNeighbor = 0
        };

        static constexpr int kMaxLongEdge = 256;

        int sourceX1 = 0;
        int sourceY1 = 0;
        int sourceX2 = 0;
        int sourceY2 = 0;
        int meterX1 = 0;
        int meterY1 = 0;
        int meterX2 = 0;
        int meterY2 = 0;
        int previewWidth = 0;
        int previewHeight = 0;
        Spektrafilm::AutoExposureMethod method = Spektrafilm::AutoExposureMethod::CenterWeighted;
        Sampling sampling = Sampling::NearestNeighbor;
        std::uint64_t hash = 0;
    };

} // namespace JuicerCuda

// Returns 0 on success, non-zero on failure; on failure outErrorMsg points to a stable message.
extern "C" int juicer_cuda_measure_center_weighted_Y(
    const void* srcDeviceBase,
    std::size_t srcRowBytes,
    int srcBoundsX1,
    int srcBoundsY1,
    int srcBoundsX2,
    int srcBoundsY2,
    int meterX1,
    int meterY1,
    int meterX2,
    int meterY2,
    int nComponents,
    int inputColorSpaceIndex,
    int applyCctfDecoding,
    const float* rgbToXYZ9,
    double* outY,
    void* cudaStreamOpaque,
    const char** outErrorMsg);

extern "C" int juicer_cuda_measure_median_Y(
    const void* srcDeviceBase,
    std::size_t srcRowBytes,
    int srcBoundsX1,
    int srcBoundsY1,
    int srcBoundsX2,
    int srcBoundsY2,
    int meterX1,
    int meterY1,
    int meterX2,
    int meterY2,
    int nComponents,
    int inputColorSpaceIndex,
    int applyCctfDecoding,
    const float* rgbToXYZ9,
    double* outY,
    void* cudaStreamOpaque,
    const char** outErrorMsg);

// Enqueues auto-exposure metering and writes autoEV/exposureScale to device outputs. This
// function does not synchronize; it only enqueues work on the given stream.
// Returns 0 on success, non-zero on failure; on failure outErrorMsg points to a stable message.
extern "C" int juicer_cuda_auto_exposure_meter_to_device(
    const void* srcDeviceBase,
    std::size_t srcRowBytes,
    JuicerCuda::AutoExposurePreviewDescriptor descriptor,
    int nComponents,
    int inputColorSpaceIndex,
    int applyCctfDecoding,
    const float* rgbToXYZ9,
    JuicerCudaAutoExposureScratch scratch,
    JuicerCudaAutoExposureDeviceState outState,
    void* cudaStreamOpaque,
    const char** outErrorMsg);

// Builds separable center-weighted metering weights (wX/wY) on the GPU. This function does not
// synchronize; it only enqueues work on the given stream.
extern "C" int juicer_cuda_auto_exposure_build_center_weight_tables(
    int meterWidth,
    int meterHeight,
    float* weightsX,
    float* weightsY,
    void* cudaStreamOpaque,
    const char** outErrorMsg);

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
        // Serializes multi-step serving mutations (`ensure_uploaded`, `ensure_scan_lut`,
        // `ensure_print_illuminant_filtered`) so `m` can remain a short-lived leaf lock for
        // individual resource-state access/update phases. Always take this mutex before `m`.
        std::mutex servingUpdateMutex;
        // Leaf lock for per-device CUDA resource state. Do not hold InstanceState locks or
        // resource-manager admission/cache bookkeeping locks while taking this mutex, and do not
        // sleep or wait on external work while it is held. Multi-phase serving updates should use
        // `servingUpdateMutex` to serialize the transaction and take `m` only around leaf work.
        std::mutex m;
        int deviceId = -1;
        // CUcontext identity captured from the render slot key; used by teardown safety checks.
        void* ownerContextOpaque = nullptr;
        std::uint64_t uploadedBuildCounter = 0;
        std::uint64_t uploadedCoreHash = 0;
        std::uint64_t uploadedDirHash = 0;
        std::uint64_t validatedBuildCounter = 0;
        std::uint64_t validatedPrintBuildCounter = 0;
        std::uint64_t validatedPrintParamsHash = 0;
        std::uint64_t directFinalSensitivityHash = 0;
        std::uint64_t directDensityCurvesHash = 0;
        std::uint64_t directDensityBoundsHash = 0;
        std::uint64_t directScannerDescriptorHash = 0;
        Spektrafilm::RgbToRawMethod directSelectedMethod = Spektrafilm::RgbToRawMethod::Hanatos2025;
        std::uint64_t directUploadCounter = 0;

        struct PendingFrameUseEvent {
            void* eventOpaque = nullptr;
        };

        // Submitted prepared-frame use events retained until reuse-gating waits can observe
        // completion. These events are never recorded again after insertion.
        std::vector<PendingFrameUseEvent> pendingFrameUseEvents;

        enum class RetireKind : int {
            DeviceFree = 0,
            HostPinnedFree = 1,
            EventDestroy = 2,
            DeviceFreeAsync = 3
        };

        struct RetireEntry {
            void* ptr = nullptr;
            std::size_t bytes = 0;
            RetireKind kind = RetireKind::DeviceFree;
            bool scratchTier = false;
            void* doneEventOpaque = nullptr; // cudaEvent_t recorded once for this entry.
        };

        // Deferred frees to avoid blocking synchronize/free in hot paths.
        std::vector<RetireEntry> retireQueue;
        std::vector<void*> retireEventPoolOpaque; // cudaEvent_t pool (cudaEventDisableTiming)
        std::size_t retireBytes = 0;
        std::size_t retireScratchBytes = 0;
        // Tracks pointers allocated with cudaMallocAsync so free/retire uses cudaFreeAsync.
        std::unordered_set<void*> asyncDeviceAllocPointers;

        struct ScratchResidencyState {
            std::array<std::uint64_t, ResourceManager::kScratchPolicyCandidateCount> candidateLiveBytes{};
            std::array<std::uint64_t, ResourceManager::kScratchHelperNonPolicyAllocationCount> helperNonPolicyBytes{};
            std::uint64_t helperSharedBytes = 0;
            std::uint64_t helperNonPolicyTotalBytes = 0;
            std::uint64_t policyLiveRetainedBytes = 0;
            std::uint64_t totalLiveRetainedBytes = 0;
            std::uint64_t retainedGeneration = 1;
            bool overflow = false;
        };

        DeviceCurve densB;
        DeviceCurve densG;
        DeviceCurve densR;
        float* densityCurvesLayers[3][3] = { {nullptr, nullptr, nullptr},
                                             {nullptr, nullptr, nullptr},
                                             {nullptr, nullptr, nullptr} };
        int densityCurvesLayersChannelN[3] = {0, 0, 0};
        int hasDensityCurvesLayers = 0;

        DeviceCurve dirDensB;
        DeviceCurve dirDensG;
        DeviceCurve dirDensR;

        DeviceCurve sensB;
        DeviceCurve sensG;
        DeviceCurve sensR;

        // SPD reconstruction (reference illuminant tables; Mallett basis uses illum + sensitivities).
        float* tablesAx = nullptr;
        float* tablesAy = nullptr;
        float* tablesAz = nullptr;
        float* tablesIllum = nullptr;
        int tablesK = 0;
        float spdSInv[9] = { 1,0,0, 0,1,0, 0,0,1 };
        float refIllumWhiteXYZ[3] = { 0.950455f, 1.0f, 1.089058f };

        float* mallettBasis = nullptr;
        int mallettBasisK = 0;

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
            double* log10XYZ = nullptr; // canonical layout: ((C*res + M)*res + Y) * 3 + XYZ
            double* slopeC = nullptr;
            double* slopeM = nullptr;
            double* slopeY = nullptr;
            double* cellMin = nullptr; // layout: ((C*(res-1) + M)*(res-1) + Y) * 3 + XYZ
            double* cellMax = nullptr;
            std::uint32_t res = 0;
            std::uint64_t hash = 0;

            bool canonical_ready() const noexcept {
                return log10XYZ && slopeC && slopeM && slopeY && cellMin && cellMax && res >= 2u;
            }
        };

        DeviceSpectralLut scanNegativeLut;
        DeviceSpectralLut scanPrintLut;
        bool privateLutFallbackNegativeActive = false;
        bool privateLutFallbackPrintActive = false;
        std::uint64_t privateLutFallbackNegativeHash = 0;
        std::uint64_t privateLutFallbackPrintHash = 0;

        struct DeviceGaussianKernel {
            float* weights = nullptr;
            int radius = 0;
            float sigma = 0.0f;
            int capacity = 0;
            // Non-zero when this kernel points to the process-shared immutable Gaussian cache.
            std::uint64_t sharedKernelId = 0;
        };

        // Shared single-plane W×H float scratch used as a blur/unsharp intermediate.
        // Spatial DIR and scanner optics reuse this to reduce peak VRAM.
        float* sharedTmpPlane = nullptr;
        int sharedTmpWidth = 0;
        int sharedTmpHeight = 0;
        std::size_t sharedTmpCapacityElements = 0;

        struct DeviceOpticsScratch {
            float* rgbR = nullptr;
            float* rgbG = nullptr;
            float* rgbB = nullptr;
            float* tmp = nullptr;
            float* blurred = nullptr;
            float* aux = nullptr;
            float* grainTmp = nullptr;
            float* grainTmpShared = nullptr;
            float* grainTmpMid = nullptr;
            float* grainTmpCoarse = nullptr;
            float* gateMask = nullptr;
            int width = 0;
            int height = 0;
            std::size_t capacityElements = 0;
            int gateWidth = 0;
            int gateHeight = 0;
            std::size_t gateMaskCapacityElements = 0;
            std::uint64_t gateMaskHash = 0;
        };

        struct DeviceSpatialDirScratch {
            float* corrY = nullptr;
            float* corrM = nullptr;
            float* corrC = nullptr;
            float* tmp = nullptr;
            int width = 0;
            int height = 0;
            std::size_t capacityElements = 0;
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
        std::uint64_t retainedScratchLeaseGeneration = 0;

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
        std::uint64_t printPreflashKeyHash = 0;
        int printPreflashShapeK = 0;

        // Cached enlarger illuminant filtered by dichroic Y/M/C for the current print params.
        float* printIllumFiltered = nullptr;
        int printIllumK = 0;
        float printIllumYShiftSteps = 0.0f;
        float printIllumMShiftSteps = 0.0f;
        float printIllumCShiftSteps = 0.0f;
        std::uint64_t printIllumNeutralFilterHash = 0;
        int printIllumShapeK = 0;
        std::uint64_t printIllumBuildCounter = 0;
        std::uint64_t printIllumCoreHash = 0;

        // Hanatos LUT (process-global on CPU, uploaded on demand).
        // Layout matches NpySpectraLUT: ((x*N + y) * K + k), K=81.
        float* hanatosLut = nullptr;
        int hanatosN = 0;

        // Hanatos LUT preintegrated with recipe sensitivities.
        // Layout: ((x*N + y) * 4 + c), c=0..2 (RGB), c=3 unused/padding.
        float* hanatosLutIntegrated = nullptr;
        int hanatosNIntegrated = 0;
        std::uint64_t hanatosIntegratedKeyHash = 0;

        struct PendingScanErrorReadback {
            int* host = nullptr;
            void* eventOpaque = nullptr;
        };

        // Submitted readbacks retained after a prepared frame releases its exclusive scan-error
        // stage. These entries are not reusable workspace.
        std::vector<PendingScanErrorReadback> pendingScanErrorReadbacks;

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
        ScratchResidencyState scratchResidency{};

        Resources() = default;
        Resources(const Resources&) = delete;
        Resources& operator=(const Resources&) = delete;

        ~Resources() noexcept;
    };

    Resources* create() noexcept;
    void destroy(Resources* resources) noexcept;

    // Narrow context-static serving helper used by the process-owned Root grain slots.
    bool ensure_grain_static_assets_uploaded(Resources& resources, void* cudaStreamOpaque, std::string& outError);

    struct DirectResourcePreparation {
        const RenderRecipe* recipe = nullptr;
        const Spectral::SpectralTables* exposureTables = nullptr;
        const float* spdSInv = nullptr;
        const Spectral::FilmRawConfig* filmRawConfig = nullptr;
        const Spectral::SpectralTables* scannerTables = nullptr;
        const Scanner::ColorRuntime* scannerColor = nullptr;
        const Scanner::ScannerSpectralLutDescriptor* scannerLutDescriptor = nullptr;
    };

    // Descriptor-driven direct-route preparation. This is called only behind Root's prepared
    // frame boundary and intentionally has no WorkingState or static-noise input.
    bool prepare_direct_resources(
        Resources& resources,
        const DirectResourcePreparation& request,
        void* cudaStreamOpaque,
        std::string& outError);

    // Runtime serving acquisition/rebuild calls are intentionally manager-only via
    // ResourceManager::command_* wrappers.

    // Optional debug validation of primitives, kept in the production CUDA validation surface.
    bool validate_density_primitives(Resources& resources, const WorkingState& ws, void* cudaStreamOpaque, std::string& outError);
    bool validate_print_primitives(
        Resources& resources,
        const WorkingState& ws,
        const Print::Runtime& prt,
        const Print::Params& prm,
        float midgrayFactor,
        void* cudaStreamOpaque,
        std::string& outError);

    // Reaps deferred retire entries that are ready and returns reclaimed bytes.
    bool reap_retired_allocations(Resources& resources, std::size_t& reclaimedBytes, std::string& outError);

    // Manager-only scratch telemetry surfaces use the helper-owned retained-scratch view built here.
    void snapshot_scratch_stage1_state(
        Resources& resources,
        ResourceManager::ScratchStage1DecisionState& outState) noexcept;
    void snapshot_scratch_residency_view(
        Resources& resources,
        ResourceManager::ScratchResidencyView& outView) noexcept;
    bool retire_scratch_policy_candidate(
        Resources& resources,
        ResourceManager::ScratchPolicyCandidate candidate,
        void* cudaStreamOpaque,
        std::string& outError);
    bool retire_orphaned_shared_tmp_plane(
        Resources& resources,
        void* cudaStreamOpaque,
        std::string& outError);
    bool try_acquire_retained_frame_scratch_lease(
        Resources& resources,
        std::uint64_t leaseGeneration,
        void* cudaStreamOpaque,
        bool& outAcquired,
        std::string& outError);
    bool release_retained_frame_scratch_lease(
        Resources& resources,
        std::uint64_t leaseGeneration,
        std::string& outError);
    bool retire_frame_scratch_allocation(
        Resources& resources,
        void* ptr,
        std::size_t bytes,
        void* cudaStreamOpaque,
        const char* label,
        std::string& outError);
    bool retain_scan_error_readback(
        Resources& resources,
        int*& host,
        void*& eventOpaque,
        std::string& outError);
    bool poll_scan_error_readbacks(
        Resources& resources,
        void* cudaStreamOpaque,
        bool& outDetected,
        std::string& outError);

    bool retain_frame_use_event(
        Resources& resources,
        void*& eventOpaque,
        std::string& outError);

    // Purges process-shared Gaussian kernels for one device/context key.
    void purge_shared_gaussian_kernels_for_context(int deviceId, void* contextOpaque) noexcept;

    // Purges process-shared pinned upload staging blocks for one device/context key.
    void purge_pinned_upload_staging_for_context(int deviceId, void* contextOpaque) noexcept;

    // Purges process-shared host asset caches once no live CUDA managers remain.
    void purge_host_asset_caches_if_registry_idle(const char* stage) noexcept;

} // namespace JuicerCuda
