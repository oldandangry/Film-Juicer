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
#include <condition_variable>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "Cuda/JuicerCudaFailure.h"
#include "Cuda/JuicerCudaFilmGrain.h"

#include "Cuda/JuicerCudaDeviceLedger.h"
#include "Cuda/Diffusion/JuicerCudaDiffusionResources.h"
#include "Cuda/JuicerCudaPayloads.h"
#include "Cuda/ResourceManager/JuicerCudaResourceCore.h"
#include "RenderValues.h"

struct FjHostPreparation;
struct FjFilmGrain;

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

    struct FocusedRouteResourceInput;
    struct PrintResourceInput;
    struct StaticNoiseInput;

    // Rust-derived device identities for one exact-context Print owner.
    struct PrintResourceDescriptors {
        std::uint64_t filmDensityTablesHash = 0;
        std::uint64_t profileTablesHash = 0;
        std::uint64_t mainIlluminantHash = 0;
        std::uint64_t preflashIlluminantHash = 0;
        CmyCcTriplet mainCmyCc{};
        bool preflashActive = false;
    };

    struct PrintPreparedView {
        FilmDensityTablesPayload filmDensityTables{};
        DeviceCurveView printSensC{};
        DeviceCurveView printSensM{};
        DeviceCurveView printSensY{};
        DeviceCurveView printDcC{};
        DeviceCurveView printDcM{};
        DeviceCurveView printDcY{};
        const float* mainIlluminant = nullptr;
        const float* preflashIlluminant = nullptr;
        int spectralSampleCount = 0;
        float preflashRawCmy[3] = {0.0f, 0.0f, 0.0f};
        float normalizer = 1.0f;
        std::uint64_t filmDensityTablesHash = 0;
        std::uint64_t profileTablesHash = 0;
        std::uint64_t mainIlluminantHash = 0;
        std::uint64_t preflashIlluminantHash = 0;
        bool preflashActive = false;
        bool active = false;
    };

    struct PrintCudaPayloadPack {
        PrintExposePayload expose{};
        PrintDevelopPayload develop{};
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
        FilmRawPayload filmRaw{};
        FilmReconstructionPayload reconstruction{};
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

    struct ScanErrorReadbackIdentity {
        std::uint64_t dirRecipeHash = 0;
        std::uint64_t dirDescriptorHash = 0;
        std::uint64_t contextEpoch = 0;
        std::string profileKey;
        int scanRoute = 0;
        int capturePolarity = 0;
        std::uint64_t submissionId = 0;
        std::uint64_t frameToken = 0;
        std::uint64_t scannerHash = 0;
        std::uint64_t scannedProfileAssetVersion = 0;
        std::string paperProfileKey;
    };

    struct Resources {
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
            bool requiresContextDrain = false;
            DeviceByteReservation deviceReservation;

            RetireEntry() noexcept = default;
            RetireEntry(const RetireEntry&) = delete;
            RetireEntry& operator=(const RetireEntry&) = delete;
            RetireEntry(RetireEntry&&) noexcept = default;
            RetireEntry& operator=(RetireEntry&&) noexcept = default;
        };

        struct DeviceFilmDensityTables {
            float* epsC = nullptr;
            float* epsM = nullptr;
            float* epsY = nullptr;
            float* baseDensityMin = nullptr;
            int K = 0;
        };

        struct DeviceGaussianKernel {
            float* weights = nullptr;
            int radius = 0;
            float sigma = 0.0f;
        };

        struct DeviceSpatialDirBoundary {
            double* weights = nullptr;
            std::size_t bytes = 0;
            int horizontalLength = 0;
            int verticalLength = 0;
            int horizontalTerminalSize = 0;
            int verticalTerminalSize = 0;
            std::array<double, 9> horizontalTerminalMatrix{};
            std::array<double, 9> verticalTerminalMatrix{};
            std::uint64_t identity = 0;
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
            float* gateTransmittance = nullptr;
            float* filmDustTransmittance = nullptr;
            int width = 0;
            int height = 0;
            std::size_t capacityElements = 0;
            int gateWidth = 0;
            int gateHeight = 0;
            std::size_t gateTransmittanceCapacityElements = 0;
            std::size_t filmDustCapacityElements = 0;
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
            ScanErrorReadbackIdentity identity{};
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
        // Owns immutable Gaussian weights for this exact context/epoch. Named
        // Gaussian slots below are non-owning views into this cache.
        std::map<std::uint64_t, float*> gaussianKernelCache;
        Diffusion::DiffusionContextResources diffusion;
        std::uint64_t filmFinalSensitivityHash = 0;
        std::uint64_t filmDensityCurvesHash = 0;
        std::uint64_t filmDensityLayersHash = 0;
        std::uint64_t filmDirHash = 0;
        std::uint64_t outputGamutTableHash = 0;
        std::uint64_t outputGamutRecipeHash = 0;
        std::size_t retireBytes = 0;
        std::size_t retireScratchBytes = 0;

        // SPD reconstruction (reference illuminant tables; Mallett basis uses illum + sensitivities).
        float* tablesIllum = nullptr;
        float* mallettBasis = nullptr;

        // Shared single-plane WÃƒâ€”H float scratch used as a blur/unsharp intermediate.
        // Spatial DIR and scanner optics reuse this to reduce peak VRAM.
        float* sharedTmpPlane = nullptr;
        void* unfencedFrameUseStreamOpaque = nullptr;
        std::size_t sharedTmpCapacityElements = 0;
        std::uint64_t retainedScratchLeaseGeneration = 0;

        std::uint8_t* stbnData = nullptr;
        std::uint8_t* wangTilesData = nullptr;
        std::uint8_t* wangLutData = nullptr;

        float* printIllumFiltered = nullptr;
        float* printPreflashIllumFiltered = nullptr;
        std::uint64_t printFilmDensityTablesDescriptorHash = 0;
        std::uint64_t printProfileTablesDescriptorHash = 0;
        std::uint64_t printMainIlluminantDescriptorHash = 0;
        std::uint64_t printPreflashIlluminantDescriptorHash = 0;

        // The selected TC method owns one integrated 192x192x4 table.
        float* filmTcLut = nullptr;
        std::uint64_t filmTcLutKeyHash = 0;

        // The active output space owns one 64x720 Cmax table for this exact
        // context/epoch. Prepared views borrow this pointer.
        float* outputGamutCmax = nullptr;
        struct FilmGrainSlot {
            FilmGrain::FilmQuad* bank = nullptr;
            FjStatus failure{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
            bool loading = true;
            std::condition_variable ready;
        };
        std::map<std::uint64_t, std::shared_ptr<FilmGrainSlot>> filmGrainBanks;

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

        float* densityCurvesLayers[3][3] = {{nullptr, nullptr, nullptr},
                                            {nullptr, nullptr, nullptr},
                                            {nullptr, nullptr, nullptr}};

        // Focused print resources carry the accepted preparation descriptor identities.
        DeviceFilmDensityTables printFilmDensityTables;

        // Serializes multi-step serving mutations. Always take this mutex before `m`.
        std::mutex servingUpdateMutex;
        // Serializes only the pointer-to-accounting-token ownership table.
        std::mutex deviceAllocationRecordsMutex;
        // Leaf lock for per-device CUDA resource state. Do not wait on external work while held.
        std::mutex m;

        DeviceOpticsScratch scannerScratch;
        DeviceSpatialDirScratch spatialDirScratch;
        std::array<DeviceGaussianKernel, 4> spatialDirKernels{};
        std::array<DeviceSpatialDirBoundary, 4> spatialDirBoundaries{};
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
        int printIllumK = 0;
        int printPreflashIllumK = 0;
        int filmTcLutExtent = 0;
        int densityCurvesLayersChannelN[3] = {0, 0, 0};

        Spektrafilm::RgbToRawMethod filmRgbToRawMethod = Spektrafilm::RgbToRawMethod::Hanatos2025;
        bool frameUseFenceQuarantined = false;
        bool contextInvalidatedByProvenLoss = false;

        Resources(
            const ResourceManager::DeviceContextKey& contextKey,
            std::uint64_t epoch,
            std::shared_ptr<DeviceAllocationLedger> ledger);
        Resources(const Resources&) = delete;
        Resources& operator=(const Resources&) = delete;

        ~Resources() noexcept;

    private:
        friend struct DeferredDestroyQueue;
        // The existing deferred owner links already allocated Resources; queue
        // transfer and exceptional restoration must not allocate storage.
        Resources* deferredDestroyNext = nullptr;
    };

    Resources* create(
        const ResourceManager::DeviceContextKey& contextKey,
        std::uint64_t contextEpoch,
        std::shared_ptr<DeviceAllocationLedger> deviceLedger,
        Failure& outError) noexcept;
    void destroy(Resources* resources) noexcept;
    void invalidate_deferred_resources_after_proven_context_loss(
        const ResourceManager::DeviceContextKey& key);
    bool drain_deferred_resources(
        const ResourceManager::DeviceContextKey& key,
        Failure& outError);
#if defined(JUICER_DEFERRED_DESTROY_TEST_HOOK)
    namespace DeferredDestroyTest {
        void before_drain(Resources& resources);
    } // namespace DeferredDestroyTest
#endif

    bool drain_for_context_retire(
        Resources& resources,
        Failure& outError) noexcept;
    void invalidate_resources_after_proven_context_loss(
        Resources& resources) noexcept;

    // Narrow context-static serving helper used by the process-owned Root grain slots.

    bool ensure_film_grain_uploaded(Resources& resources, const FjFilmGrain& input, const FilmGrain::FilmQuad*& out, Failure& error);

    bool ensure_grain_static_assets_uploaded(
        Resources& resources,
        const StaticNoiseInput& input,
        void* cudaStreamOpaque,
        Failure& outError);

    // Prepare the frame's film and optional output gamut device resources.
    bool prepare_focused_route_resources(
        Resources& resources,
        const FocusedRouteResourceInput& input,
        void* cudaStreamOpaque,
        Failure& outError);

    bool prepare_print_resources(
        Resources& resources,
        const PrintResourceInput& input,
        void* cudaStreamOpaque,
        Failure& outError);

    bool pack_print_cuda_payloads(
        float printExposure,
        float preflashExposure,
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
        int radius,
        float sigma,
        Failure& outError,
        const FjHostPreparation* hostPreparation = nullptr);
    bool clear_spatial_dir_kernel_binding(
        Resources& resources,
        Resources::DeviceGaussianKernel& kernel,
        Failure& outError);
    bool ensure_spatial_dir_boundary(
        Resources& resources,
        Resources::DeviceSpatialDirBoundary& boundary,
        const Spektrafilm::DirGaussianComponentPlan& component,
        int width,
        int height,
        void* cudaStreamOpaque,
        Failure& outError);
    bool clear_spatial_dir_boundary_binding(
        Resources& resources,
        Resources::DeviceSpatialDirBoundary& boundary,
        void* cudaStreamOpaque,
        Failure& outError);
    bool ensure_gaussian_kernel(
        Resources& resources,
        Resources::DeviceGaussianKernel& kernel,
        float sigma,
        int radius,
        Failure& outError,
        const FjHostPreparation* hostPreparation = nullptr);

    // Reaps deferred retire entries that are ready and returns reclaimed bytes.
    bool reap_retired_allocations(Resources& resources, std::size_t& reclaimedBytes, Failure& outError);

    bool acquire_retained_frame_scratch_lease(
        Resources& resources,
        std::uint64_t leaseGeneration,
        void* cudaStreamOpaque,
        Failure& outError);
    bool release_retained_frame_scratch_lease(
        Resources& resources,
        std::uint64_t leaseGeneration,
        Failure& outError);

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
        Failure& outError);
    bool reclaim_large_scratch_transition(
        Resources& resources,
        const ResourceManager::ScratchRequestDescriptor& scratchRequest,
        void* cudaStreamOpaque,
        LargeScratchTransitionReclaimStats& outStats,
        Failure& outError);
    bool shed_retained_scratch_after_frame(
        Resources& resources,
        void* cudaStreamOpaque,
        PostFrameScratchShedStats& outStats,
        Failure& outError);
    bool retire_frame_scratch_allocation(
        Resources& resources,
        void* ptr,
        std::size_t bytes,
        DeviceByteReservation&& reservation,
        void* cudaStreamOpaque,
        const char* label,
        Failure& outError);
    bool adopt_failed_frame_allocation_record(
        Resources& resources,
        std::map<void*, DeviceByteReservation>::node_type& allocationRecord,
        bool completionCertain,
        Failure& outError) noexcept;
    struct ScanErrorReadbackResult {
        int status = 0;
        ScanErrorReadbackIdentity identity;
    };

    bool retain_scan_error_readback(
        Resources& resources,
        int*& host,
        void*& eventOpaque,
        ScanErrorReadbackIdentity& identity,
        Failure& outError);
    bool poll_scan_error_readbacks(
        Resources& resources,
        void* cudaStreamOpaque,
        ScanErrorReadbackResult& outResult,
        Failure& outError);

    bool record_frame_use_event(
        Resources& resources,
        void* cudaStreamOpaque,
        const char* label,
        Failure& outError);

    bool register_frame_status_staging(Resources& resources, void* host, std::size_t bytes, Failure& outError);
    bool retire_frame_status_staging(
        Resources& resources,
        void* host,
        cudaEvent_t completionEvent,
        bool requiresContextDrain,
        Failure& outError);
    void quarantine_unfenced_frame_use(
        Resources& resources,
        void* cudaStreamOpaque) noexcept;

    enum class PinnedUploadPurgeDisposition : std::uint8_t {
        NormalRetire = 0,
        ProvenContextLoss
    };

    // Purges process-shared pinned upload staging blocks for one device/context key.
    void purge_pinned_upload_staging_for_context(
        int deviceId,
        void* contextOpaque,
        PinnedUploadPurgeDisposition disposition) noexcept;

#if defined(JUICER_GAUSSIAN_TEST_HOOK)
    namespace GaussianKernelTest {
        std::int32_t copy(Resources&, std::uint64_t key, void* destination, const void* source, std::size_t bytes) noexcept;
        void before_insert(Resources&, std::uint64_t key) noexcept;
        bool after_upload(Resources&, std::uint64_t key, float* candidate) noexcept;
        void observed_free(Resources&, std::uint64_t key, void* pointer, bool success) noexcept;
    } // namespace GaussianKernelTest
#endif

#if defined(JUICER_PINNED_UPLOAD_TEST_HOOK) || defined(JUICER_CUDA_RENDER_TEST_HOOK)
    namespace PinnedUploadTest {
        struct TransferCount {
            std::size_t calls = 0;
            std::size_t bytes = 0;
            bool operator==(const TransferCount&) const = default;
        };
        struct UploadObservation {
            TransferCount print{};
            std::array<TransferCount, 4> filmDensity{};
            TransferCount mainIlluminant{};
            TransferCount preflashIlluminant{};
        };
    } // namespace PinnedUploadTest
#endif

#if defined(JUICER_PINNED_UPLOAD_TEST_HOOK)
    namespace PinnedUploadTest {
        enum class Operation : std::uint8_t {
            Allocate,
            CreateEvent,
            Copy,
            RecordEvent,
            Synchronize
        };
        std::int32_t injected_error(Operation operation) noexcept;
        void before_diagnostic();
        void after_overwrite_staging(Resources&) noexcept;

        struct Block {
            void* pointer = nullptr;
            void* event = nullptr;
            std::size_t capacity = 0;
            std::uint64_t id = 0;
        };
        struct Snapshot {
            std::size_t available = 0;
            std::size_t reserved = 0;
            std::size_t inFlight = 0;
            std::size_t quarantined = 0;
            std::size_t totalBytes = 0;
            std::vector<Block> blocks;
        };
        bool upload(Resources& resources, void* destination, const void* source, std::size_t bytes, void* stream, Failure& failure);
        Snapshot snapshot(const ResourceManager::DeviceContextKey& key);
        void poll(const ResourceManager::DeviceContextKey& key);
        void reset_upload_observation() noexcept;
        UploadObservation finish_upload_observation() noexcept;
        void observe_staged_upload(const char* label, std::size_t bytes) noexcept;
    } // namespace PinnedUploadTest
#endif

} // namespace JuicerCuda
