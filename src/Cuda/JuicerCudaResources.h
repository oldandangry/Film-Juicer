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
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "Cuda/JuicerCudaDeviceLedger.h"
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
#include "Cuda/Diffusion/JuicerCudaDiffusionResources.h"
#endif
#include "Cuda/JuicerCudaPayloads.h"
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
namespace JuicerAssets {
    class Library;
    struct StaticNoisePayloadSet;
} // namespace JuicerAssets

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
    float* weightsX = nullptr;         // meterWidth floats
    float* weightsY = nullptr;         // meterHeight floats
};

// Device-side outputs for metering. All pointers are CUDA device pointers.
struct JuicerCudaAutoExposureDeviceState {
    float* exposureScale = nullptr; // float scalar
};

namespace JuicerCuda {

    // Phase 4B focused print-preparation contract:
    // - producer: build_print_resource_descriptors() from RenderRecipe::print and selected
    //   validated profiles; consumer: prepare_print_resources() and pack_print_cuda_payloads().
    // - update frequency: descriptor miss only; descriptor hits perform no asset lookup,
    //   spectral derivation, profile packing, allocation, or upload.
    // - layouts/units: canonical 81-sample host/device spectra, capture-film spectral density,
    //   density curves on authored logE, and C/M/Y channel order. Filter values are Kodak CC units.
    // - ownership/lifetime: immutable host derivations are temporary; device arrays and scalar
    //   results are exact-context/epoch Resources residency exposed through PreparedCudaFrame.
    // - schema/hash: schema v1 includes selected profile tokens, process-owned source illuminant
    //   asset version, and Phase 4A recipe/resource identities; excludes file discovery, CUDA
    //   pointers, frame tokens, and launch policy.
    // - preparation point: Root's Phase 4B print prelaunch prepared-frame overload.
    // - disabled behavior: disabled preflash has zero descriptor identity and no derivation/upload.
    struct PrintProfileTablesDescriptor {
        static constexpr std::uint32_t kSchemaVersion = 1u;

        std::uint64_t printProfileAssetVersionToken = 0;
        std::uint64_t densityCurvesHash = 0;
        std::uint64_t sensitivitiesHash = 0;
        std::uint32_t densitySampleCount = 0;
        std::uint32_t spectralSampleCount = 0;
        std::uint64_t hash = 0;
    };

    struct PrintFilmDensityTablesDescriptor {
        static constexpr std::uint32_t kSchemaVersion = 1u;

        std::uint64_t filmProfileAssetVersionToken = 0;
        std::uint64_t densityTablesHash = 0;
        std::uint32_t spectralSampleCount = 0;
        std::uint64_t hash = 0;
    };

    struct FilteredPrintIlluminantDescriptor {
        static constexpr std::uint32_t kSchemaVersion = 1u;

        std::uint64_t sourceIlluminantAssetVersionToken = 0;
        std::uint64_t printIlluminantHash = 0;
        std::uint64_t dichroicResourceHash = 0;
        CmyCcTriplet cmyCc{};
        bool preflash = false;
        std::uint64_t hash = 0;
    };

    struct PrintPreflashRawDescriptor {
        static constexpr std::uint32_t kSchemaVersion = 1u;

        std::uint64_t filmProfileAssetVersionToken = 0;
        std::uint64_t printProfileTablesHash = 0;
        std::uint64_t filteredPreflashIlluminantHash = 0;
        std::uint64_t hash = 0;
    };

    struct PrintBalanceDescriptor {
        static constexpr std::uint32_t kSchemaVersion = 1u;

        std::uint64_t filmProfileAssetVersionToken = 0;
        std::uint64_t filmReferenceIlluminantAssetVersionToken = 0;
        std::uint64_t filmRawRecipeHash = 0;
        std::uint64_t filmDevelopRecipeHash = 0;
        std::uint64_t printProfileTablesHash = 0;
        std::uint64_t filteredMainIlluminantHash = 0;
        Spektrafilm::PrintNormalizationMode normalizationMode =
            Spektrafilm::PrintNormalizationMode::None;
        float cameraExposureCompensationEv = 0.0f;
        std::uint64_t hash = 0;
    };

    struct PrintResourceDescriptors {
        PrintProfileTablesDescriptor profileTables{};
        PrintFilmDensityTablesDescriptor filmDensityTables{};
        FilteredPrintIlluminantDescriptor mainIlluminant{};
        FilteredPrintIlluminantDescriptor preflashIlluminant{};
        PrintPreflashRawDescriptor preflashRaw{};
        PrintBalanceDescriptor balance{};
        bool preflashActive = false;
        std::uint64_t hash = 0;
    };

    struct PrintResourcePreparation {
        const RenderRecipe* recipe = nullptr;
        JuicerAssets::Library* assets = nullptr;
    };

    struct PrintPreparedView {
        ScanTablesPayload filmDensityTables{};
        DeviceCurveView printSensC{};
        DeviceCurveView printSensM{};
        DeviceCurveView printSensY{};
        DeviceCurveView printDcC{};
        DeviceCurveView printDcM{};
        DeviceCurveView printDcY{};
        const float* mainIlluminant = nullptr;
        const float* mainIlluminantHost = nullptr;
        const float* preflashIlluminant = nullptr;
        int spectralSampleCount = 0;
        float preflashRawCmy[3] = {0.0f, 0.0f, 0.0f};
        float normalizer = 1.0f;
        std::uint64_t filmDensityTablesHash = 0;
        std::uint64_t profileTablesHash = 0;
        std::uint64_t mainIlluminantHash = 0;
        std::uint64_t preflashIlluminantHash = 0;
        std::uint64_t preflashRawHash = 0;
        std::uint64_t balanceHash = 0;
        std::uint64_t preparationHash = 0;
        bool preflashActive = false;
        bool active = false;
    };

    struct PrintCudaPayloadPack {
        PrintExposePayload expose{};
        PrintDevelopPayload develop{};
        float printRawScale = 1.0f;
    };

    struct AutoExposurePreviewDescriptor {
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
        std::uint64_t hash = 0;
    };

    struct AutoExposureSourceFormat {
        int componentCount = 0;
        int inputColorSpaceIndex = 0;
        int applyCctfDecoding = 0;
    };

} // namespace JuicerCuda

// Enqueues auto-exposure metering and writes the exposure scale to device output. This
// function does not synchronize; it only enqueues work on the given stream.
// Returns 0 on success, non-zero on failure; on failure outErrorMsg points to a stable message.
extern "C" int juicer_cuda_auto_exposure_meter_to_device(
    const void* srcDeviceBase,
    std::size_t srcRowBytes,
    JuicerCuda::AutoExposurePreviewDescriptor descriptor,
    JuicerCuda::AutoExposureSourceFormat sourceFormat,
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
        static constexpr std::uint32_t kAllocationOwnershipSchemaVersion = 1u;

        struct PendingFrameUseEvent {
            void* eventOpaque = nullptr;
        };

        enum class RetireKind : std::uint8_t {
            DeviceFree = 0,
            HostPinnedFree = 1,
            EventDestroy = 2
        };

        struct RetireEntry {
            void* ptr = nullptr;
            std::size_t bytes = 0;
            RetireKind kind = RetireKind::DeviceFree;
            bool scratchTier = false;
            void* doneEventOpaque = nullptr; // cudaEvent_t recorded once for this entry.
            DeviceByteReservation deviceReservation;

            RetireEntry() noexcept = default;
            RetireEntry(const RetireEntry&) = delete;
            RetireEntry& operator=(const RetireEntry&) = delete;
            RetireEntry(RetireEntry&&) noexcept = default;
            RetireEntry& operator=(RetireEntry&&) noexcept = default;
        };

        struct DeviceSpectralTables {
            float* epsC = nullptr;
            float* epsM = nullptr;
            float* epsY = nullptr;
            float* Ax = nullptr;
            float* Ay = nullptr;
            float* Az = nullptr;
            float* baseDensityMin = nullptr;
            int K = 0;
            int hasBaseline = 0;
            float invYn = 1.0f;
        };

        struct DeviceScanMedium {
            DeviceSpectralTables tables;
            int mediumIsNegative = 1;
            float min_cmy[3] = {0.0f, 0.0f, 0.0f};
            float inv_max_cmy[3] = {1.0f, 1.0f, 1.0f};
        };

        struct DeviceSpectralLut {
            float* log2PchipXYZ = nullptr; // canonical layout: ((C*res + M)*res + Y) * 3 + XYZ
            float* slopeC = nullptr;
            float* slopeM = nullptr;
            float* slopeY = nullptr;
            float* cellMin = nullptr; // layout: ((C*(res-1) + M)*(res-1) + Y) * 3 + XYZ
            float* cellMax = nullptr;
            std::uint32_t res = 0;
            std::uint64_t hash = 0;

            bool canonical_ready() const noexcept {
                return log2PchipXYZ && slopeC && slopeM && slopeY && cellMin && cellMax && res >= 2u;
            }
        };

        struct DeviceGaussianKernel {
            float* weights = nullptr;
            int radius = 0;
            float sigma = 0.0f;
            int capacity = 0;
            // Non-zero when this kernel points to the process-shared immutable Gaussian cache.
            std::uint64_t sharedKernelId = 0;
        };

        struct DeviceOpticsScratch {
            float* rgbR = nullptr;
            float* rgbG = nullptr;
            float* rgbB = nullptr;
            float* tmp = nullptr;
            float* blurred = nullptr;
            float* aux = nullptr;
            float* grainTmp = nullptr;
            float* grainTmpShared = nullptr;
            GrainFrameUniforms* grainFrameUniforms = nullptr;
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
            float* rawCorrectionY = nullptr;
            float* rawCorrectionM = nullptr;
            float* rawCorrectionC = nullptr;
            float* filteredCorrectionY = nullptr;
            float* filteredCorrectionM = nullptr;
            float* filteredCorrectionC = nullptr;
            float* filterTemp = nullptr;
            float* filterTempM = nullptr;
            float* filterTempC = nullptr;
            float* logRawB = nullptr;
            float* logRawG = nullptr;
            float* logRawR = nullptr;
            int width = 0;
            int height = 0;
            std::size_t capacityElements = 0;
        };

        struct PendingScanErrorReadback {
            int* host = nullptr;
            void* eventOpaque = nullptr;
        };

        // Eight-byte-aligned identity, residency, pointer, and container state is grouped first
        // so the per-context resource object does not pay repeated alignment gaps between its
        // narrower counters and flags.
        ResourceManager::DeviceContextKey ownerContextKey{};
        std::uint64_t contextEpoch = 0;
        std::shared_ptr<DeviceAllocationLedger> deviceLedger;
        std::map<void*, DeviceByteReservation> deviceAllocationRecords;
        std::map<void*, DeviceByteReservation>
            contextLossOnlyDeviceAllocationRecords;
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        Diffusion::DiffusionContextResources diffusion;
#endif
        std::uint64_t filmFinalSensitivityHash = 0;
        std::uint64_t filmDensityCurvesHash = 0;
        std::uint64_t filmDensityLayersHash = 0;
        std::uint64_t filmDirHash = 0;
        std::uint64_t routeDensityBoundsHash = 0;
        std::uint64_t routeScannerDescriptorHash = 0;
        std::uint64_t focusedPreparationCounter = 0;
        std::size_t retireBytes = 0;
        std::size_t retireScratchBytes = 0;

        // SPD reconstruction (reference illuminant tables; Mallett basis uses illum + sensitivities).
        float* tablesAx = nullptr;
        float* tablesAy = nullptr;
        float* tablesAz = nullptr;
        float* tablesIllum = nullptr;
        float* mallettBasis = nullptr;

        // Shared single-plane W×H float scratch used as a blur/unsharp intermediate.
        // Spatial DIR and scanner optics reuse this to reduce peak VRAM.
        float* sharedTmpPlane = nullptr;
        std::size_t sharedTmpCapacityElements = 0;
        std::uint64_t retainedScratchLeaseGeneration = 0;

        std::uint8_t* stbnData = nullptr;
        std::uint8_t* wangTilesData = nullptr;
        std::uint8_t* wangLutData = nullptr;
        std::uint64_t grainStaticAssetVersion = 0;

        std::uint64_t printPreflashKeyHash = 0;
        float* printIllumFiltered = nullptr;
        float* printPreflashIllumFiltered = nullptr;
        std::uint64_t printFilmDensityTablesDescriptorHash = 0;
        std::uint64_t printProfileTablesDescriptorHash = 0;
        std::uint64_t printMainIlluminantDescriptorHash = 0;
        std::uint64_t printPreflashIlluminantDescriptorHash = 0;
        std::uint64_t printPreflashRawDescriptorHash = 0;
        std::uint64_t printBalanceDescriptorHash = 0;
        std::uint64_t printPreparationDescriptorHash = 0;
        std::uint64_t printPreparationCounter = 0;

        // Hanatos LUTs are process-global on CPU and uploaded on demand.
        float* hanatosLut = nullptr;
        float* hanatosLutIntegrated = nullptr;
        std::uint64_t hanatosIntegratedKeyHash = 0;

        // Submitted prepared-frame use events retained until reuse-gating waits can observe
        // completion. These events are never recorded again after insertion.
        std::vector<PendingFrameUseEvent> pendingFrameUseEvents;
        // Deferred frees to avoid blocking synchronize/free in hot paths.
        std::vector<RetireEntry> retireQueue;
        // Completed untimed cudaEvent_t handles available for later completion fences.
        std::vector<void*> completionEventPoolOpaque;
        // Submitted readbacks retained after a prepared frame releases its exclusive scan-error
        // stage. These entries are not reusable workspace.
        std::vector<PendingScanErrorReadback> pendingScanErrorReadbacks;

        DeviceCurve densB;
        DeviceCurve densG;
        DeviceCurve densR;
        DeviceCurve dirDensB;
        DeviceCurve dirDensG;
        DeviceCurve dirDensR;
        DeviceCurve sensB;
        DeviceCurve sensG;
        DeviceCurve sensR;

        DeviceGaussianKernel scannerLensBlurKernel;
        DeviceGaussianKernel scannerUnsharpKernel;
        DeviceGaussianKernel scannerGlareKernel;
        DeviceGaussianKernel grainBlurKernel;
        DeviceGaussianKernel grainBlurKernelMid;
        DeviceGaussianKernel grainBlurKernelCoarse;

        // Print-route payloads.
        DeviceCurve printDcC;
        DeviceCurve printDcM;
        DeviceCurve printDcY;
        DeviceCurve printSensC;
        DeviceCurve printSensM;
        DeviceCurve printSensY;

        DeviceSpectralLut scanNegativeLut;
        DeviceSpectralLut scanPrintLut;
        float* densityCurvesLayers[3][3] = {{nullptr, nullptr, nullptr},
                                            {nullptr, nullptr, nullptr},
                                            {nullptr, nullptr, nullptr}};

        // Focused print resources carry the accepted preparation descriptor identities.
        DeviceSpectralTables printFilmDensityTables;

        // Serializes multi-step serving mutations. Always take this mutex before `m`.
        std::mutex servingUpdateMutex;
        // Serializes only the pointer-to-accounting-token ownership table.
        std::mutex deviceAllocationRecordsMutex;
        // Leaf lock for per-device CUDA resource state. Do not wait on external work while held.
        std::mutex m;

        DeviceScanMedium scanNegative;
        DeviceScanMedium scanPrint;
        DeviceOpticsScratch scannerScratch;
        DeviceSpatialDirScratch spatialDirScratch;
        std::array<DeviceGaussianKernel, 4> spatialDirKernels{};
        DeviceGaussianKernel grainDyeKernel[3][3];

        // Four-byte scalar and fixed-array state follows the aligned resource owners.
        int deviceId = -1;
        int hasDensityCurvesLayers = 0;
        int tablesK = 0;
        int mallettBasisK = 0;
        int sharedTmpWidth = 0;
        int sharedTmpHeight = 0;
        int stbnWidth = 0;
        int stbnHeight = 0;
        int stbnFrames = 0;
        int wangWidth = 0;
        int wangHeight = 0;
        int wangCount = 0;
        int wangColors = 0;
        float printGammaC = 1.0f;
        float printGammaM = 1.0f;
        float printGammaY = 1.0f;
        int printPreflashShapeK = 0;
        int printIllumK = 0;
        int printPreflashIllumK = 0;
        float printBalanceFactorMidgray = 1.0f;
        float printBalanceFactorMidgrayComp = 1.0f;
        float printBalanceNormalizer = 1.0f;
        int hanatosN = 0;
        int hanatosNIntegrated = 0;
        int densityCurvesLayersChannelN[3] = {0, 0, 0};
        float refIllumWhiteXYZ[3] = {0.950455f, 1.0f, 1.089058f};
        float printPreflashRaw[3] = {0.0f, 0.0f, 0.0f};
        float spdSInv[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        std::array<float, 81> printIllumFilteredHost{};
        std::array<float, 81> printPreflashIllumFilteredHost{};

        Spektrafilm::RgbToRawMethod filmRgbToRawMethod = Spektrafilm::RgbToRawMethod::Hanatos2025;
        bool printPreflashValid = false;
        bool printIllumFilteredHostValid = false;
        bool printPreflashIllumFilteredHostValid = false;
        bool contextInvalidatedByProvenLoss = false;

        Resources(
            const ResourceManager::DeviceContextKey& contextKey,
            std::uint64_t epoch,
            std::shared_ptr<DeviceAllocationLedger> ledger);
        Resources(const Resources&) = delete;
        Resources& operator=(const Resources&) = delete;

        ~Resources() noexcept;
    };

    Resources* create(
        const ResourceManager::DeviceContextKey& contextKey,
        std::uint64_t contextEpoch,
        std::uint32_t allocationOwnershipSchemaVersion,
        std::shared_ptr<DeviceAllocationLedger> deviceLedger,
        std::string& outError) noexcept;
    void destroy(Resources* resources) noexcept;
    bool drain_for_context_retire(
        Resources& resources,
        std::string& outError) noexcept;
    void invalidate_resources_after_proven_context_loss(
        Resources& resources) noexcept;

    // Narrow context-static serving helper used by the process-owned Root grain slots.
    bool ensure_grain_static_assets_uploaded(
        Resources& resources,
        const JuicerAssets::StaticNoisePayloadSet& payloads,
        std::uint64_t expectedAssetVersion,
        void* cudaStreamOpaque,
        std::string& outError);

    struct FocusedRouteResourcePreparation {
        const RenderRecipe* recipe = nullptr;
        const Spectral::SpectralTables* exposureTables = nullptr;
        const float* spdSInv = nullptr;
        const Spectral::FilmRawConfig* filmRawConfig = nullptr;
        const Spectral::SpectralTables* scannerTables = nullptr;
        const Scanner::ColorRuntime* scannerColor = nullptr;
        const Scanner::ScannerSpectralLutDescriptor* scannerLutDescriptor = nullptr;
    };

    // Descriptor-driven film and selected scan-route preparation. This is called only behind
    // Root's prepared-frame boundary and intentionally has no WorkingState or static-noise input.
    bool prepare_focused_route_resources(
        Resources& resources,
        const FocusedRouteResourcePreparation& request,
        void* cudaStreamOpaque,
        std::string& outError);

    bool build_print_resource_descriptors(
        const RenderRecipe& recipe,
        PrintResourceDescriptors& out,
        std::string& diagnostic);

    bool prepare_print_resources(
        Resources& resources,
        const PrintResourcePreparation& request,
        void* cudaStreamOpaque,
        std::string& outError);

    bool pack_print_cuda_payloads(
        const PrintRecipe& recipe,
        const PrintPreparedView& prepared,
        float routeCorrectionScale,
        PrintCudaPayloadPack& out,
        std::string& diagnostic);

    // Resource acquisition/rebuild calls with admission or retry policy remain
    // manager-owned. Prepared-frame Gaussian serving calls enter here after
    // their active frame or workspace lease has already been validated.
    bool ensure_spatial_dir_kernel(
        Resources& resources,
        Resources::DeviceGaussianKernel& kernel,
        float sigma,
        void* cudaStreamOpaque,
        std::string& outError);
    bool ensure_gaussian_kernel(
        Resources& resources,
        Resources::DeviceGaussianKernel& kernel,
        float sigma,
        void* cudaStreamOpaque,
        std::string& outError);

    // Reaps deferred retire entries that are ready and returns reclaimed bytes.
    bool reap_retired_allocations(Resources& resources, std::size_t& reclaimedBytes, std::string& outError);

    bool acquire_retained_frame_scratch_lease(
        Resources& resources,
        std::uint64_t leaseGeneration,
        void* cudaStreamOpaque,
        std::string& outError);
    bool release_retained_frame_scratch_lease(
        Resources& resources,
        std::uint64_t leaseGeneration,
        std::string& outError);

    struct SpatialDirCachedLogRawStageStats {
        std::size_t pendingScratchBytesBefore = 0;
        std::size_t cachedLogRawAllocatedBytes = 0;
    };

    struct LargeScratchTransitionReclaimStats {
        std::size_t pendingScratchBytesBefore = 0;
        std::size_t opticsRetiredBytes = 0;
        std::size_t spatialDirRetiredBytes = 0;
        std::size_t sharedTmpRetiredBytes = 0;
        std::size_t reclaimedBytes = 0;
    };

    struct PostFrameScratchShedStats {
        std::size_t pendingScratchBytesBefore = 0;
        std::size_t opticsRetiredBytes = 0;
        std::size_t spatialDirRetiredBytes = 0;
        std::size_t sharedTmpRetiredBytes = 0;
        std::size_t reclaimedBytes = 0;
    };

    bool ensure_retained_spatial_dir_cached_log_raw_stage(
        Resources& resources,
        std::uint64_t leaseGeneration,
        const ResourceManager::ScratchRequestDescriptor& scratchRequest,
        void* cudaStreamOpaque,
        SpatialDirCachedLogRawStageStats& outStats,
        std::string& outError);
    bool reclaim_large_scratch_transition(
        Resources& resources,
        const ResourceManager::ScratchRequestDescriptor& scratchRequest,
        void* cudaStreamOpaque,
        LargeScratchTransitionReclaimStats& outStats,
        std::string& outError);
    bool shed_retained_scratch_after_frame(
        Resources& resources,
        void* cudaStreamOpaque,
        PostFrameScratchShedStats& outStats,
        std::string& outError);
    bool retire_frame_scratch_allocation(
        Resources& resources,
        void* ptr,
        std::size_t bytes,
        DeviceByteReservation&& reservation,
        void* cudaStreamOpaque,
        const char* label,
        std::string& outError);
    bool adopt_failed_frame_allocation_record(
        Resources& resources,
        std::map<void*, DeviceByteReservation>::node_type& allocationRecord,
        bool completionCertain,
        std::string& outError) noexcept;
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

    bool record_frame_use_event(
        Resources& resources,
        void* cudaStreamOpaque,
        const char* label,
        std::string& outError);

    // Purges process-shared Gaussian kernels for one device/context key.
    bool purge_shared_gaussian_kernels_for_context(
        int deviceId,
        void* contextOpaque,
        std::uint64_t contextEpoch,
        std::string& outError) noexcept;
    void invalidate_shared_gaussian_kernels_after_proven_context_loss(
        int deviceId,
        void* contextOpaque,
        std::uint64_t contextEpoch) noexcept;

    enum class PinnedUploadPurgeDisposition : std::uint8_t {
        NormalRetire = 0,
        ProvenContextLoss
    };

    // Purges process-shared pinned upload staging blocks for one device/context key.
    void purge_pinned_upload_staging_for_context(
        int deviceId,
        void* contextOpaque,
        PinnedUploadPurgeDisposition disposition) noexcept;

} // namespace JuicerCuda
