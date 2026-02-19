// Cuda/ResourceManager/JuicerCudaResourceManager.cpp

#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"

#include "Cuda/JuicerCudaResources.h"
#include "Cuda/ResourceManager/JuicerCudaManagerRegistry.h"
#include "Cuda/ResourceManager/JuicerCudaResourceConfig.h"
#include "Cuda/ResourceManager/JuicerCudaResourceKeys.h"
#include "Cuda/ResourceManager/JuicerCudaResourcePolicy.h"
#include "Cuda/ResourceManager/JuicerCudaResourceState.h"
#include "Cuda/ResourceManager/JuicerCudaResourceTelemetry.h"
#include "Print.h"
#include "WorkingState.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
#include <cuda_runtime.h>
#endif

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
extern "C" cudaError_t juicer_cuda_negative_pipeline(
    const JuicerCuda::PipelineRunParams* hParams,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_print_pipeline(
    const JuicerCuda::PipelineRunParams* hParams,
    void* cudaStreamOpaque);
#endif

namespace JuicerCuda {

// Internal resource-acquire helpers are intentionally consumed only by command wrappers
// in this module; they are not part of the public JuicerCudaResources API surface.
bool ensure_uploaded(Resources& resources, const WorkingState& ws, void* cudaStreamOpaque, std::string& outError);
bool ensure_scan_lut(Resources& resources, const WorkingState& ws, bool negativeMedium, void* cudaStreamOpaque, std::string& outError);
bool ensure_scan_error_flag(Resources& resources, void* cudaStreamOpaque, std::string& outError);
bool ensure_auto_exposure_buffers(Resources& resources, int meterWidth, int meterHeight, void* cudaStreamOpaque, std::string& outError);
bool ensure_optics_scratch(Resources& resources, int width, int height, bool needBlurredScratch, bool needAuxScratch, bool needGrainScratch, bool needGrainSharedScratch, bool needGateMask, void* cudaStreamOpaque, std::string& outError);
bool ensure_spatial_dir_scratch(Resources& resources, int width, int height, void* cudaStreamOpaque, std::string& outError);
bool ensure_spatial_dir_kernel(Resources& resources, Resources::DeviceGaussianKernel& kernel, float sigma, void* cudaStreamOpaque, std::string& outError);
bool ensure_gaussian_kernel(Resources& resources, Resources::DeviceGaussianKernel& kernel, float sigma, void* cudaStreamOpaque, std::string& outError);
bool ensure_halation_kernel(Resources& resources, Resources::DeviceGaussianKernel& kernel, float sigma, void* cudaStreamOpaque, std::string& outError);
bool ensure_print_illuminant_filtered(
    Resources& resources,
    const WorkingState& ws,
    const Print::Runtime& prt,
    const Print::Params& prm,
    void* cudaStreamOpaque,
    std::string& outError);

namespace ResourceManager {

namespace {

struct ShadowHistoryKey {
    std::uint64_t instanceToken = 0;
    DeviceContextKey deviceContextKey{};

    bool operator==(const ShadowHistoryKey& other) const noexcept {
        return instanceToken == other.instanceToken &&
            deviceContextKey == other.deviceContextKey;
    }
};

struct ShadowHistoryKeyHasher {
    std::size_t operator()(const ShadowHistoryKey& key) const noexcept {
        const std::size_t hInstance = std::hash<std::uint64_t>{}(key.instanceToken);
        const std::size_t hContext = DeviceContextKeyHash{}(key.deviceContextKey);
        return hInstance ^ (hContext + 0x9e3779b9u + (hInstance << 6u) + (hInstance >> 2u));
    }
};

struct ShadowHistoryEntry {
    bool valid = false;
    KeyDigests digests{};
    std::uint32_t keySchemaVersion = 1;
    std::uint64_t snapshotId = 0;
};

struct ShadowHistoryState {
    std::mutex mutex;
    std::unordered_map<ShadowHistoryKey, ShadowHistoryEntry, ShadowHistoryKeyHasher> bySubmissionKey;
};

ShadowHistoryState& shadow_history_state() noexcept {
    static ShadowHistoryState state{};
    return state;
}

struct AutoExposureOwnershipEntry {
    bool valid = false;
    std::uint64_t keyHash = 0;
    int meterWidth = 0;
    int meterHeight = 0;
    std::uint32_t keySchemaVersion = 1;
};

struct AutoExposureOwnershipState {
    std::mutex mutex;
    std::unordered_map<ShadowHistoryKey, AutoExposureOwnershipEntry, ShadowHistoryKeyHasher> bySubmissionKey;
};

AutoExposureOwnershipState& auto_exposure_ownership_state() noexcept {
    static AutoExposureOwnershipState state{};
    return state;
}

struct FrameSnapshotKey {
    std::uint64_t instanceToken = 0;
    DeviceContextKey deviceContextKey{};

    bool operator==(const FrameSnapshotKey& other) const noexcept {
        return instanceToken == other.instanceToken &&
            deviceContextKey == other.deviceContextKey;
    }
};

struct FrameSnapshotKeyHasher {
    std::size_t operator()(const FrameSnapshotKey& key) const noexcept {
        const std::size_t hInstance = std::hash<std::uint64_t>{}(key.instanceToken);
        const std::size_t hContext = DeviceContextKeyHash{}(key.deviceContextKey);
        return hInstance ^ (hContext + 0x9e3779b9u + (hInstance << 6u) + (hInstance >> 2u));
    }
};

struct FrameSnapshotEntry {
    bool valid = false;
    std::uint64_t frameToken = 0;
    KeyDigests digests{};
    std::uint32_t keySchemaVersion = 1;
    std::uint64_t snapshotId = 0;
};

struct FrameSnapshotState {
    std::mutex mutex;
    std::unordered_map<FrameSnapshotKey, FrameSnapshotEntry, FrameSnapshotKeyHasher> bySubmissionKey;
};

FrameSnapshotState& frame_snapshot_state() noexcept {
    static FrameSnapshotState state{};
    return state;
}

enum class ScratchWorkClass : std::uint8_t {
    Optics = 0,
    SpatialDir = 1
};

enum class PressureLane : std::uint8_t {
    Builder = 0,
    Upload = 1
};

enum class BuilderReservationTier : std::uint8_t {
    Scratch = 0,
    Lut = 1,
    Graph = 2
};

enum class TierCircuitState : std::uint8_t {
    Closed = 0,
    Open = 1,
    HalfOpen = 2
};

struct AllocatorBackendContextEntry {
    bool valid = false;
    bool traced = false;
    AllocatorBackendPreference requested = AllocatorBackendPreference::Auto;
    AllocatorBackendMode candidate = AllocatorBackendMode::Legacy;
    AllocatorBackendMode active = AllocatorBackendMode::Legacy;
    bool asyncPoolSupported = false;
    bool slabSupported = false;
    bool fallbackCapability = false;
    bool fallbackScaffold = false;
    bool mempoolPolicyValid = false;
    bool mempoolPolicyApplied = false;
    std::uint64_t mempoolReleaseThresholdBytes = 0;
    PressureState mempoolPolicyPressureState = PressureState::Normal;
    const char* candidateReason = "unknown";
    const char* activeReason = "unknown";
};

struct AllocatorBackendState {
    std::mutex mutex;
    std::unordered_map<DeviceContextKey, AllocatorBackendContextEntry, DeviceContextKeyHash> byContext;
};

AllocatorBackendState& allocator_backend_state() noexcept {
    static AllocatorBackendState state{};
    return state;
}

AllocatorBackendContextEntry& allocator_backend_get_or_init_locked(
    AllocatorBackendState& state,
    const DeviceContextKey& key,
    const ResourceManagerConfigEffective& cfg) noexcept;

bool allocator_backend_try_get_active_mode(
    const DeviceContextKey& key,
    AllocatorBackendMode& outMode) noexcept {
    AllocatorBackendState& state = allocator_backend_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    const auto it = state.byContext.find(key);
    if (it == state.byContext.end() || !it->second.valid) {
        return false;
    }
    outMode = it->second.active;
    return true;
}

const char* to_cstr(ScratchWorkClass workClass) noexcept {
    switch (workClass) {
    case ScratchWorkClass::Optics:
        return "optics";
    case ScratchWorkClass::SpatialDir:
        return "spatial_dir";
    default:
        return "unknown";
    }
}

const char* to_cstr(PressureLane lane) noexcept {
    switch (lane) {
    case PressureLane::Builder:
        return "builder";
    case PressureLane::Upload:
        return "upload";
    default:
        return "unknown";
    }
}

const char* to_cstr(BuilderReservationTier tier) noexcept {
    switch (tier) {
    case BuilderReservationTier::Scratch:
        return "scratch";
    case BuilderReservationTier::Lut:
        return "lut";
    case BuilderReservationTier::Graph:
        return "graph";
    default:
        return "unknown";
    }
}

const char* to_cstr(ResourceTier tier) noexcept {
    switch (tier) {
    case ResourceTier::Immutable:
        return "immutable";
    case ResourceTier::Lut:
        return "lut";
    case ResourceTier::Scratch:
        return "scratch";
    case ResourceTier::Graph:
        return "graph";
    default:
        return "unknown";
    }
}

const char* to_cstr(TierCircuitState state) noexcept {
    switch (state) {
    case TierCircuitState::Closed:
        return "closed";
    case TierCircuitState::Open:
        return "open";
    case TierCircuitState::HalfOpen:
        return "half_open";
    default:
        return "unknown";
    }
}

const char* to_cstr(AllocatorBackendPreference value) noexcept {
    switch (value) {
    case AllocatorBackendPreference::Legacy:
        return "legacy";
    case AllocatorBackendPreference::AsyncPool:
        return "async_pool";
    case AllocatorBackendPreference::Slab:
        return "slab";
    case AllocatorBackendPreference::Auto:
        return "auto";
    default:
        return "unknown";
    }
}

const char* to_cstr(AllocatorBackendMode value) noexcept {
    switch (value) {
    case AllocatorBackendMode::Legacy:
        return "legacy";
    case AllocatorBackendMode::AsyncPool:
        return "async_pool";
    case AllocatorBackendMode::Slab:
        return "slab";
    default:
        return "unknown";
    }
}

struct ScratchBucketKey {
    std::uint32_t widthBucket = 0;
    std::uint32_t heightBucket = 0;
    std::uint16_t bucketStepPx = 0;
    ScratchWorkClass workClass = ScratchWorkClass::Optics;
    bool largeFrame = false;

    bool operator==(const ScratchBucketKey& other) const noexcept {
        return widthBucket == other.widthBucket &&
            heightBucket == other.heightBucket &&
            bucketStepPx == other.bucketStepPx &&
            workClass == other.workClass &&
            largeFrame == other.largeFrame;
    }
};

struct ScratchBucketKeyHash {
    std::size_t operator()(const ScratchBucketKey& key) const noexcept {
        const std::size_t hW = std::hash<std::uint32_t>{}(key.widthBucket);
        const std::size_t hH = std::hash<std::uint32_t>{}(key.heightBucket);
        const std::size_t hStep = std::hash<std::uint16_t>{}(key.bucketStepPx);
        const std::size_t hWork = std::hash<std::uint8_t>{}(static_cast<std::uint8_t>(key.workClass));
        const std::size_t hLarge = std::hash<bool>{}(key.largeFrame);
        return (((hW ^ (hH + 0x9e3779b9u + (hW << 6u) + (hW >> 2u)))
            ^ (hStep + 0x9e3779b9u + (hW << 6u) + (hW >> 2u)))
            ^ (hWork + 0x9e3779b9u + (hW << 6u) + (hW >> 2u)))
            ^ (hLarge + 0x9e3779b9u + (hW << 6u) + (hW >> 2u));
    }
};

struct ScratchBucketEntry {
    std::size_t inFlightBytes = 0;
    std::uint32_t inFlightSets = 0;
    std::uint64_t attemptCount = 0;
    std::uint64_t exhaustedCount = 0;
    std::uint64_t allocGrowthEvents = 0;
    std::uint64_t reuseEvents = 0;
    bool starvationLatched = false;
};

struct ScratchQuarantineEntry {
    ScratchBucketKey key{};
    std::size_t bytes = 0;
    std::uint64_t touchedMs = 0;
    std::uint64_t sequence = 0;
};

struct ScratchContextState {
    std::unordered_map<ScratchBucketKey, ScratchBucketEntry, ScratchBucketKeyHash> buckets;
    std::vector<ScratchQuarantineEntry> largeFrameQuarantine;
    std::size_t largeFrameQuarantineBytes = 0;
    std::size_t totalInFlightBytes = 0;
    std::uint64_t nextQuarantineSequence = 1;
};

struct ScratchPolicyState {
    std::mutex mutex;
    std::unordered_map<DeviceContextKey, ScratchContextState, DeviceContextKeyHash> byContext;
    std::uint64_t totalInFlightBytes = 0;
};

ScratchPolicyState& scratch_policy_state() noexcept {
    static ScratchPolicyState state{};
    return state;
}

struct UploadReservationContextState {
    std::uint64_t inFlightBytes = 0;
    struct FairnessEntry {
        std::uint32_t sharedTokens = 0;
        std::uint32_t criticalTokens = 0;
        std::uint64_t lastRefillMs = 0;
    };
    std::unordered_map<std::uint64_t, FairnessEntry> fairnessByInstance;
};

struct UploadReservationState {
    std::mutex mutex;
    std::unordered_map<DeviceContextKey, UploadReservationContextState, DeviceContextKeyHash> byContext;
    std::uint64_t totalInFlightBytes = 0;
};

UploadReservationState& upload_reservation_state() noexcept {
    static UploadReservationState state{};
    return state;
}

struct BuilderReservationContextState {
    std::uint64_t inFlightScratchBytes = 0;
    std::uint64_t inFlightLutBytes = 0;
    std::uint64_t inFlightGraphBytes = 0;
    struct FairnessEntry {
        std::uint32_t sharedTokens = 0;
        std::uint32_t criticalTokens = 0;
        std::uint64_t lastRefillMs = 0;
    };
    std::unordered_map<std::uint64_t, FairnessEntry> fairnessByInstance;
};

struct BuilderReservationState {
    std::mutex mutex;
    std::unordered_map<DeviceContextKey, BuilderReservationContextState, DeviceContextKeyHash> byContext;
    std::uint64_t totalScratchInFlightBytes = 0;
    std::uint64_t totalLutInFlightBytes = 0;
    std::uint64_t totalGraphInFlightBytes = 0;
};

BuilderReservationState& builder_reservation_state() noexcept {
    static BuilderReservationState state{};
    return state;
}

struct BurstDebtEntry {
    std::uint32_t debtPct = 0;
    std::uint64_t lastUpdateMs = 0;
};

struct PressureContextState {
    bool valid = false;
    std::uint64_t lastSampleMs = 0;
    PressureDecision lastDecision{};
    PressureState lastState = PressureState::Normal;
    std::uint64_t lastStateChangeMs = 0;
    std::uint64_t transitionWindowStartMs = 0;
    std::uint32_t transitionsInWindow = 0;
    bool reserveCrossed = false;
    bool effectiveReserveValid = false;
    std::uint64_t effectiveReserveBytes = 0;
    bool opportunisticFrozen = false;
    bool burstActive = false;
    bool burstCapHitLatched = false;
    std::uint64_t burstWindowStartMs = 0;
    std::uint64_t burstPeakOverTargetBytes = 0;
    std::unordered_map<std::uint64_t, BurstDebtEntry> burstDebtByInstance;
    bool headroomSourceValid = false;
    HeadroomSource lastHeadroomSource = HeadroomSource::FreeVramOnly;
};

struct PressurePolicyState {
    std::mutex mutex;
    std::unordered_map<DeviceContextKey, PressureContextState, DeviceContextKeyHash> byContext;
};

PressurePolicyState& pressure_policy_state() noexcept {
    static PressurePolicyState state{};
    return state;
}

struct TierCircuitTierState {
    TierCircuitState state = TierCircuitState::Closed;
    std::uint64_t windowStartMs = 0;
    std::uint32_t windowErrors = 0;
    std::uint64_t openedAtMs = 0;
    bool probeInFlight = false;
};

struct TierCircuitContextState {
    std::array<TierCircuitTierState, 4> tiers{};
};

struct TierCircuitPolicyState {
    std::mutex mutex;
    std::unordered_map<DeviceContextKey, TierCircuitContextState, DeviceContextKeyHash> byContext;
};

TierCircuitPolicyState& tier_circuit_policy_state() noexcept {
    static TierCircuitPolicyState state{};
    return state;
}

struct AdmissionChurnSnapshot {
    bool enabled = false;
    bool active = false;
    std::uint32_t windowMs = 0;
    std::uint32_t enterOneHitRatePct = 0;
    std::uint32_t exitOneHitRatePct = 0;
    std::uint32_t probationHitBonus = 0;
    std::uint32_t keepHotMs = 0;
    std::uint32_t readmitCooldownMs = 0;
    std::uint32_t ghostHitsForReadmit = 0;
    std::uint32_t oneHitRatePct = 0;
    std::uint32_t uniqueKeys = 0;
    std::uint32_t windowSamples = 0;
};

struct LargeEntryReadmitDecision {
    bool enabled = false;
    bool candidate = false;
    bool criticalCurrentFrame = false;
    bool hadHistory = false;
    bool inCooldown = false;
    bool blocked = false;
    bool ghostBypass = false;
    std::uint64_t ageMs = 0;
    std::uint32_t cooldownMs = 0;
    std::uint32_t ghostHitsRequired = 0;
    std::uint32_t observedGhostHits = 0;
    const char* reason = "disabled";
};

struct AdmissionChurnContextState {
    std::uint64_t windowStartMs = 0;
    std::uint64_t windowSamples = 0;
    std::uint64_t windowOneHitSamples = 0;
    std::unordered_map<std::uint64_t, std::uint32_t> windowDigestHits;
    bool active = false;
};

struct AdmissionChurnPolicyState {
    std::mutex mutex;
    std::unordered_map<DeviceContextKey, AdmissionChurnContextState, DeviceContextKeyHash> byContext;
};

AdmissionChurnPolicyState& admission_churn_policy_state() noexcept {
    static AdmissionChurnPolicyState state{};
    return state;
}

struct OptionalHeuristicTraceContextState {
    bool keepHotTraced = false;
    bool burstDebtTraced = false;
    bool supersededBuilderCancelTraced = false;
};

struct OptionalHeuristicTraceState {
    std::mutex mutex;
    std::unordered_map<DeviceContextKey, OptionalHeuristicTraceContextState, DeviceContextKeyHash> byContext;
};

OptionalHeuristicTraceState& optional_heuristic_trace_state() noexcept {
    static OptionalHeuristicTraceState state{};
    return state;
}

struct ManagerMemorySnapshot {
    std::uint64_t activeBytes = 0;
    std::uint64_t reclaimableBytes = 0;
    std::uint64_t retirePendingBytes = 0;
    std::uint64_t transientNonManagerBytes = 0;
    bool overflow = false;
};

struct TierBudgetSnapshot {
    std::array<std::uint64_t, 4> activeBytes{};
    std::array<std::uint64_t, 4> reclaimableBytes{};
    std::array<std::uint64_t, 4> targetBytes{};
    std::array<std::uint64_t, 4> overTargetBytes{};
    std::uint64_t totalActiveBytes = 0;
    std::uint64_t totalReclaimableBytes = 0;
    bool anyOverTarget = false;
    ResourceTier dominantOverTargetTier = ResourceTier::Immutable;
    std::uint64_t dominantOverTargetBytes = 0;
    bool overflow = false;
};

struct HeadroomTelemetry {
    std::uint64_t effectiveHeadroomBytes = 0;
    std::uint64_t driverFreeBytes = 0;
    std::uint64_t allocatorPoolReservedBytes = 0;
    std::uint64_t allocatorPoolUsedBytes = 0;
    HeadroomSource source = HeadroomSource::FreeVramOnly;
    bool poolTelemetryAvailable = false;
};

struct PressureCheckpoint {
    PressureInput input{};
    PressureDecision decision{};
    PressureState previousState = PressureState::Normal;
    PressureState desiredState = PressureState::Normal;
    bool sampled = false;
    bool transition = false;
    bool transitionDeferredByDwell = false;
    bool transitionDeferredByRate = false;
    bool reserveCrossing = false;
    bool reserveCrossedNow = false;
    bool freezeTransitionEnter = false;
    bool freezeTransitionExit = false;
    std::uint64_t reserveTargetBytes = 0;
    std::uint64_t reserveBeforeBytes = 0;
    bool reserveUpdated = false;
    std::uint32_t pollIntervalMs = 0;
    ManagerMemorySnapshot memory{};
};

struct ActiveBurstDecision {
    bool considered = false;
    bool active = false;
    bool allowed = false;
    bool entered = false;
    bool exited = false;
    bool capHit = false;
    std::uint64_t overTargetBytes = 0;
    std::uint64_t capBytes = 0;
    std::uint64_t elapsedMs = 0;
};

struct BurstDebtRuntimeDecision {
    bool enabled = false;
    bool sampled = false;
    bool burstConsumed = false;
    bool throttled = false;
    std::uint32_t debtBeforePct = 0;
    std::uint32_t debtAfterPct = 0;
    std::uint32_t debtIncrementPct = 0;
    const char* reason = "disabled";
};

struct ScratchPolicyClaim {
    DeviceContextKey contextKey{};
    ScratchBucketKey bucketKey{};
    std::size_t bytes = 0;
    bool acquired = false;
};

struct ScratchPolicySnapshot {
    std::size_t inFlightBytes = 0;
    std::uint32_t inFlightSets = 0;
    std::size_t quarantineBytes = 0;
    std::uint32_t quarantineEntries = 0;
    std::uint64_t bucketAttempts = 0;
    std::uint64_t bucketExhausted = 0;
    std::uint64_t bucketAllocGrowth = 0;
    std::uint64_t bucketReuse = 0;
    bool starvationLatched = false;
    ScratchBucketKey bucketKey{};
};

struct ReservationAttemptInfo {
    ReservationDecision decision{};
    std::uint64_t bytesInFlight = 0;
    std::uint64_t capBytes = 0;
    std::uint64_t thresholdBytes = 0;
    std::uint64_t instanceToken = 0;
    std::uint32_t sharedTokens = 0;
    std::uint32_t criticalTokens = 0;
    bool considered = false;
};

struct UploadReservationClaim {
    DeviceContextKey contextKey{};
    std::uint64_t bytes = 0;
    bool acquired = false;
};

struct BuilderReservationClaim {
    DeviceContextKey contextKey{};
    BuilderReservationTier tier = BuilderReservationTier::Scratch;
    std::uint64_t bytes = 0;
    bool acquired = false;
};

struct TierCircuitAttempt {
    DeviceContextKey contextKey{};
    ResourceTier tier = ResourceTier::Immutable;
    bool started = false;
    bool probe = false;
};

constexpr std::uint32_t kMaxTempScratchSets = 1;
constexpr std::size_t kMaxTempScratchBytes = static_cast<std::size_t>(1024ull * 1024ull * 1024ull);
constexpr int kScratchWaitStepMs = 1;
constexpr int kScratchWaitMaxMs = 4;
constexpr int kScratchWaitMaxMediumMs = 16;
constexpr int kScratchWaitMaxLargeMs = 64;
constexpr std::size_t kScratchWaitMediumRequestBytes = static_cast<std::size_t>(256ull * 1024ull * 1024ull);
constexpr std::size_t kScratchWaitLargeRequestBytes = static_cast<std::size_t>(768ull * 1024ull * 1024ull);
constexpr int kScratchBucketStepBasePx = 64;
constexpr int kScratchBucketStepLargePx = 128;
constexpr int kScratchBucketStepXLargePx = 256;
constexpr int kScratchBucketMinPx = 128;
constexpr std::uint64_t kLargeFrameThresholdPixels = static_cast<std::uint64_t>(7680ull * 4320ull);
constexpr std::size_t kLargeFrameQuarantineMaxBytes = static_cast<std::size_t>(1024ull * 1024ull * 1024ull);
constexpr std::size_t kLargeFrameQuarantineMaxEntries = 2;
constexpr std::uint64_t kLargeFrameQuarantineDecayMs = 2000;
constexpr std::uint64_t kGraphLargeEntryDecayMs = 2000;
constexpr std::uint64_t kGraphLargeEntryThresholdDefaultBytes = 128ull * 1024ull * 1024ull;
constexpr std::uint64_t kGraphLargeEntryQuarantineMaxBytesDefault = 512ull * 1024ull * 1024ull;
constexpr std::uint32_t kGraphLargeEntryQuarantineMaxEntriesDefault = 2;
constexpr const char* kScratchExhaustedPrefix = "scratch_exhausted:";
constexpr const char* kReservationDeferredPrefix = "reservation_deferred:";
constexpr std::uint64_t kTransientReservationCapDefaultBytes = 512ull * 1024ull * 1024ull;
constexpr std::uint64_t kTransientReservationThresholdDefaultBytes = 64ull * 1024ull * 1024ull;
constexpr std::uint64_t kScratchBuilderReservationCapDefaultBytes = 256ull * 1024ull * 1024ull;
constexpr std::uint64_t kLutBuilderReservationCapDefaultBytes = 128ull * 1024ull * 1024ull;
constexpr std::uint64_t kGraphBuilderReservationCapDefaultBytes = 128ull * 1024ull * 1024ull;
constexpr std::uint64_t kBuilderReservationThresholdDefaultBytes = 16ull * 1024ull * 1024ull;
constexpr std::uint64_t kUploadReservationCapDefaultBytes = 256ull * 1024ull * 1024ull;
constexpr std::uint64_t kUploadReservationThresholdDefaultBytes = 16ull * 1024ull * 1024ull;
constexpr std::uint64_t kStbnUploadDefaultBytes = 512ull * 512ull * 256ull;
constexpr std::uint64_t kWangTilesUploadDefaultBytes = 256ull * 256ull * 16ull;
constexpr std::uint64_t kWangLutUploadDefaultBytes = 8ull * 8ull * 8ull * 8ull;
constexpr std::uint64_t kBytesPerMiB = 1024ull * 1024ull;
constexpr std::uint64_t kBasisPointsDenominator = 10000ull;
constexpr int kBuilderReservationWaitStepMs = 1;
constexpr int kBuilderReservationWaitMaxMs = 8;
constexpr std::uint64_t kBuilderFairnessTickMs = 4;
constexpr int kUploadReservationWaitStepMs = 1;
constexpr int kUploadReservationWaitMaxMs = 8;
constexpr std::uint64_t kUploadFairnessTickMs = 4;

bool ascii_iequals(std::string_view lhs, std::string_view rhs) noexcept {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        const unsigned char a = static_cast<unsigned char>(lhs[i]);
        const unsigned char b = static_cast<unsigned char>(rhs[i]);
        if (std::tolower(a) != std::tolower(b)) {
            return false;
        }
    }
    return true;
}

std::string_view trim_ascii_ws(std::string_view value) noexcept {
    std::size_t begin = 0;
    while (begin < value.size()) {
        const unsigned char c = static_cast<unsigned char>(value[begin]);
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
            break;
        }
        ++begin;
    }
    std::size_t end = value.size();
    while (end > begin) {
        const unsigned char c = static_cast<unsigned char>(value[end - 1]);
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
            break;
        }
        --end;
    }
    return value.substr(begin, end - begin);
}

bool parse_env_bool(const char* name, bool& outValue) noexcept {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) {
        return false;
    }
    const std::string_view text = trim_ascii_ws(raw);
    if (text.empty()) {
        return false;
    }
    if (text == "1" ||
        ascii_iequals(text, "true") ||
        ascii_iequals(text, "yes") ||
        ascii_iequals(text, "on")) {
        outValue = true;
        return true;
    }
    if (text == "0" ||
        ascii_iequals(text, "false") ||
        ascii_iequals(text, "no") ||
        ascii_iequals(text, "off")) {
        outValue = false;
        return true;
    }
    return false;
}

bool parse_env_u32(const char* name, std::uint32_t& outValue) noexcept {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(raw, &end, 10);
    if (errno != 0 || end == raw || (end && *end != '\0')) {
        return false;
    }
    if (parsed > static_cast<unsigned long long>(std::numeric_limits<std::uint32_t>::max())) {
        return false;
    }
    outValue = static_cast<std::uint32_t>(parsed);
    return true;
}

struct FiveXEnvOverrides {
    ResourceManagerConfigRaw raw{};
    bool anyOverride = false;
    bool profileEnabled = false;
    bool profileDisabled = false;
};

void apply_5x_profile_enabled(ResourceManagerConfigRaw& raw) noexcept {
    raw.keepHotMs = 300;
    raw.admissionChurnWindowMs = 1500;
    raw.admissionChurnEnterOneHitRatePct = 65;
    raw.admissionChurnExitOneHitRatePct = 50;
    raw.admissionChurnProbationHitBonus = 1;
    raw.largeEntryReadmitCooldownMs = 1200;
    raw.largeEntryGhostHitsForReadmit = 1;
    raw.burstDebtHalfLifeMs = 3000;
    raw.maxBurstDebtPct = 50;
    raw.cancelSupersededBuilders = true;
}

void apply_5x_profile_disabled(ResourceManagerConfigRaw& raw) noexcept {
    raw.keepHotMs = 0;
    raw.admissionChurnWindowMs = 0;
    raw.admissionChurnEnterOneHitRatePct = 75;
    raw.admissionChurnExitOneHitRatePct = 60;
    raw.admissionChurnProbationHitBonus = 0;
    raw.largeEntryReadmitCooldownMs = 0;
    raw.largeEntryGhostHitsForReadmit = 0;
    raw.burstDebtHalfLifeMs = 0;
    raw.maxBurstDebtPct = 100;
    raw.cancelSupersededBuilders = false;
}

FiveXEnvOverrides load_5x_env_overrides() noexcept {
    FiveXEnvOverrides out{};
    out.raw = ResourceManagerConfigRaw{};

    bool value = false;
    if (parse_env_bool("JUICER_5X_ENABLE", value)) {
        out.profileEnabled = value;
        if (!value) {
            out.profileDisabled = true;
        }
        out.anyOverride = true;
    }
    if (parse_env_bool("JUICER_5X_DISABLE", value)) {
        if (value) {
            out.profileDisabled = true;
            out.anyOverride = true;
        }
    }

    if (out.profileEnabled && !out.profileDisabled) {
        apply_5x_profile_enabled(out.raw);
    }
    else if (out.profileDisabled) {
        apply_5x_profile_disabled(out.raw);
    }

    auto parse_u32_override = [&](const char* envName, std::uint32_t& target) {
        std::uint32_t parsed = 0;
        if (parse_env_u32(envName, parsed)) {
            target = parsed;
            out.anyOverride = true;
        }
    };
    auto parse_bool_override = [&](const char* envName, bool& target) {
        bool parsed = false;
        if (parse_env_bool(envName, parsed)) {
            target = parsed;
            out.anyOverride = true;
        }
    };

    parse_u32_override("JUICER_5X_KEEP_HOT_MS", out.raw.keepHotMs);
    parse_u32_override("JUICER_5X_ADMISSION_CHURN_WINDOW_MS", out.raw.admissionChurnWindowMs);
    parse_u32_override(
        "JUICER_5X_ADMISSION_CHURN_ENTER_ONE_HIT_RATE_PCT",
        out.raw.admissionChurnEnterOneHitRatePct);
    parse_u32_override(
        "JUICER_5X_ADMISSION_CHURN_EXIT_ONE_HIT_RATE_PCT",
        out.raw.admissionChurnExitOneHitRatePct);
    parse_u32_override(
        "JUICER_5X_ADMISSION_CHURN_PROBATION_HIT_BONUS",
        out.raw.admissionChurnProbationHitBonus);
    parse_u32_override(
        "JUICER_5X_LARGE_ENTRY_READMIT_COOLDOWN_MS",
        out.raw.largeEntryReadmitCooldownMs);
    parse_u32_override(
        "JUICER_5X_LARGE_ENTRY_GHOST_HITS_FOR_READMIT",
        out.raw.largeEntryGhostHitsForReadmit);
    parse_u32_override("JUICER_5X_BURST_DEBT_HALF_LIFE_MS", out.raw.burstDebtHalfLifeMs);
    parse_u32_override("JUICER_5X_MAX_BURST_DEBT_PCT", out.raw.maxBurstDebtPct);
    parse_bool_override("JUICER_5X_CANCEL_SUPERSEDED_BUILDERS", out.raw.cancelSupersededBuilders);

    return out;
}

void trace_5x_env_overrides(const FiveXEnvOverrides& overrides, const ResourceManagerConfigEffective& cfg) {
    if (!overrides.anyOverride || !JTRACE_ENABLED(2)) {
        return;
    }
    const std::string msg = std::string("event=5x_env_overrides")
        + " profile_enabled=" + std::to_string(overrides.profileEnabled ? 1 : 0)
        + " profile_disabled=" + std::to_string(overrides.profileDisabled ? 1 : 0)
        + " keep_hot_ms=" + std::to_string(static_cast<unsigned long long>(cfg.keepHotMs))
        + " admission_churn_window_ms=" + std::to_string(static_cast<unsigned long long>(cfg.admissionChurnWindowMs))
        + " admission_churn_enter_one_hit_rate_pct=" + std::to_string(
            static_cast<unsigned long long>(cfg.admissionChurnEnterOneHitRatePct))
        + " admission_churn_exit_one_hit_rate_pct=" + std::to_string(
            static_cast<unsigned long long>(cfg.admissionChurnExitOneHitRatePct))
        + " admission_churn_probation_hit_bonus=" + std::to_string(
            static_cast<unsigned long long>(cfg.admissionChurnProbationHitBonus))
        + " large_entry_readmit_cooldown_ms=" + std::to_string(
            static_cast<unsigned long long>(cfg.largeEntryReadmitCooldownMs))
        + " large_entry_ghost_hits_for_readmit=" + std::to_string(
            static_cast<unsigned long long>(cfg.largeEntryGhostHitsForReadmit))
        + " burst_debt_half_life_ms=" + std::to_string(
            static_cast<unsigned long long>(cfg.burstDebtHalfLifeMs))
        + " max_burst_debt_pct=" + std::to_string(static_cast<unsigned long long>(cfg.maxBurstDebtPct))
        + " cancel_superseded_builders=" + std::to_string(cfg.cancelSupersededBuilders ? 1 : 0);
    JTRACE("MSCFG", msg);
}

const ResourceManagerConfigEffective& manager_effective_config() noexcept {
    static const ResourceManagerConfigEffective cfg = []() {
        const FiveXEnvOverrides overrides = load_5x_env_overrides();
        const ResourceManagerConfigEffective effective = sanitize_config(overrides.raw);
        trace_5x_env_overrides(overrides, effective);
        return effective;
    }();
    return cfg;
}

AllocatorBackendPreference sanitize_allocator_backend_preference(std::uint32_t value) noexcept {
    switch (value) {
    case 0u:
        return AllocatorBackendPreference::Legacy;
    case 1u:
        return AllocatorBackendPreference::AsyncPool;
    case 2u:
        return AllocatorBackendPreference::Slab;
    case 3u:
        return AllocatorBackendPreference::Auto;
    default:
        return AllocatorBackendPreference::Auto;
    }
}

bool probe_async_pool_support_for_device(int deviceId) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__) && defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
    if (deviceId < 0) {
        return false;
    }
    int memoryPoolsSupported = 0;
    const cudaError_t err = cudaDeviceGetAttribute(
        &memoryPoolsSupported,
        cudaDevAttrMemoryPoolsSupported,
        deviceId);
    return err == cudaSuccess && memoryPoolsSupported != 0;
#else
    (void)deviceId;
    return false;
#endif
}

AllocatorBackendContextEntry compute_allocator_backend_context_entry(
    const DeviceContextKey& key,
    const ResourceManagerConfigEffective& cfg) noexcept {
    AllocatorBackendContextEntry out{};
    out.valid = true;
    out.requested = sanitize_allocator_backend_preference(cfg.allocatorBackendPreference);
    out.asyncPoolSupported = probe_async_pool_support_for_device(key.deviceId);
    out.slabSupported = false;
    out.candidate = AllocatorBackendMode::Legacy;
    out.active = AllocatorBackendMode::Legacy;
    out.fallbackCapability = false;
    out.fallbackScaffold = false;
    out.candidateReason = "legacy_default";
    out.activeReason = "legacy_active";

    switch (out.requested) {
    case AllocatorBackendPreference::Legacy:
        out.candidate = AllocatorBackendMode::Legacy;
        out.candidateReason = "requested_legacy";
        break;
    case AllocatorBackendPreference::AsyncPool:
        if (out.asyncPoolSupported) {
            out.candidate = AllocatorBackendMode::AsyncPool;
            out.candidateReason = "requested_async_supported";
        }
        else {
            out.candidate = AllocatorBackendMode::Legacy;
            out.fallbackCapability = true;
            out.candidateReason = "requested_async_unsupported";
        }
        break;
    case AllocatorBackendPreference::Slab:
        if (out.slabSupported) {
            out.candidate = AllocatorBackendMode::Slab;
            out.candidateReason = "requested_slab_supported";
        }
        else {
            out.candidate = AllocatorBackendMode::Legacy;
            out.fallbackCapability = true;
            out.candidateReason = "requested_slab_unsupported";
        }
        break;
    case AllocatorBackendPreference::Auto:
        if (out.asyncPoolSupported) {
            out.candidate = AllocatorBackendMode::AsyncPool;
            out.candidateReason = "auto_select_async";
        }
        else if (out.slabSupported) {
            out.candidate = AllocatorBackendMode::Slab;
            out.candidateReason = "auto_select_slab";
        }
        else {
            out.candidate = AllocatorBackendMode::Legacy;
            out.candidateReason = "auto_no_optional_supported";
        }
        break;
    default:
        out.candidate = AllocatorBackendMode::Legacy;
        out.candidateReason = "unknown_preference_fallback";
        break;
    }

    if (out.candidate == AllocatorBackendMode::AsyncPool) {
        out.active = AllocatorBackendMode::AsyncPool;
        out.activeReason = "active_async_pool";
    }
    else if (out.candidate == AllocatorBackendMode::Slab) {
        // Slab backend is not wired yet; retain deterministic legacy fallback.
        out.fallbackScaffold = true;
        out.active = AllocatorBackendMode::Legacy;
        out.activeReason = "slab_scaffold_fallback_legacy";
    }
    else if (out.fallbackCapability) {
        out.activeReason = "capability_fallback_legacy";
    }
    else {
        out.activeReason = out.candidateReason;
    }
    return out;
}

AllocatorBackendContextEntry& allocator_backend_get_or_init_locked(
    AllocatorBackendState& state,
    const DeviceContextKey& key,
    const ResourceManagerConfigEffective& cfg) noexcept {
    AllocatorBackendContextEntry& entry = state.byContext[key];
    if (!entry.valid) {
        entry = compute_allocator_backend_context_entry(key, cfg);
    }
    return entry;
}

void trace_allocator_backend_mode_once(
    const SubmissionTransaction& transaction,
    const AllocatorBackendContextEntry& entry) {
    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=backend_mode")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " requested=" + to_cstr(entry.requested)
        + " candidate=" + to_cstr(entry.candidate)
        + " active=" + to_cstr(entry.active)
        + " async_pool_supported=" + std::to_string(entry.asyncPoolSupported ? 1 : 0)
        + " slab_supported=" + std::to_string(entry.slabSupported ? 1 : 0)
        + " fallback_capability=" + std::to_string(entry.fallbackCapability ? 1 : 0)
        + " fallback_scaffold=" + std::to_string(entry.fallbackScaffold ? 1 : 0)
        + " candidate_reason=" + (entry.candidateReason ? entry.candidateReason : "unknown")
        + " active_reason=" + (entry.activeReason ? entry.activeReason : "unknown");
    JTRACE("MSALC", msg);
}

void ensure_allocator_backend_mode_initialized(
    const SubmissionTransaction& transaction,
    const ResourceManagerConfigEffective& cfg) {
    AllocatorBackendContextEntry traceEntry{};
    bool emitTrace = false;
    {
        AllocatorBackendState& state = allocator_backend_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        AllocatorBackendContextEntry& entry = allocator_backend_get_or_init_locked(
            state,
            transaction.snapshot.deviceContextKey,
            cfg);
        if (!entry.traced) {
            entry.traced = true;
            traceEntry = entry;
            emitTrace = true;
        }
    }
    if (emitTrace) {
        trace_allocator_backend_mode_once(transaction, traceEntry);
    }
}

std::uint64_t async_mempool_release_threshold_bytes_for_state(
    const ResourceManagerConfigEffective& cfg,
    PressureState pressureState) noexcept {
    const std::uint64_t baseMb = static_cast<std::uint64_t>(cfg.asyncMempoolReleaseThresholdMB);
    const std::uint64_t maxMbBeforeOverflow = std::numeric_limits<std::uint64_t>::max() / kBytesPerMiB;
    const std::uint64_t baseBytes = (baseMb > maxMbBeforeOverflow)
        ? std::numeric_limits<std::uint64_t>::max()
        : (baseMb * kBytesPerMiB);
    switch (pressureState) {
    case PressureState::Emergency:
        return 0;
    case PressureState::Critical:
        return baseBytes / 4ull;
    case PressureState::Constrained:
        return baseBytes / 2ull;
    case PressureState::Normal:
    default:
        return baseBytes;
    }
}

bool set_async_mempool_release_threshold(
    const DeviceContextKey& key,
    std::uint64_t thresholdBytes,
    std::string& outError) noexcept {
    outError.clear();
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__) && defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
    if (key.deviceId < 0) {
        outError = "invalid device id";
        return false;
    }

    int previousDevice = -1;
    const cudaError_t queryErr = cudaGetDevice(&previousDevice);
    const bool havePrevious = (queryErr == cudaSuccess && previousDevice >= 0);
    bool switchedDevice = false;

    if (!havePrevious || previousDevice != key.deviceId) {
        const cudaError_t setErr = cudaSetDevice(key.deviceId);
        if (setErr != cudaSuccess) {
            outError = std::string("cudaSetDevice failed: ")
                + (cudaGetErrorString(setErr) ? cudaGetErrorString(setErr) : "(unknown)");
            return false;
        }
        switchedDevice = havePrevious && previousDevice != key.deviceId;
    }

    cudaMemPool_t pool = nullptr;
    const cudaError_t poolErr = cudaDeviceGetDefaultMemPool(&pool, key.deviceId);
    if (poolErr != cudaSuccess || pool == nullptr) {
        if (switchedDevice) {
            (void)cudaSetDevice(previousDevice);
        }
        outError = std::string("cudaDeviceGetDefaultMemPool failed: ")
            + (cudaGetErrorString(poolErr) ? cudaGetErrorString(poolErr) : "(unknown)");
        return false;
    }

    std::size_t thresholdValue = (thresholdBytes >= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        ? std::numeric_limits<std::size_t>::max()
        : static_cast<std::size_t>(thresholdBytes);
    const cudaError_t setAttrErr = cudaMemPoolSetAttribute(
        pool,
        cudaMemPoolAttrReleaseThreshold,
        &thresholdValue);

    if (switchedDevice) {
        (void)cudaSetDevice(previousDevice);
    }

    if (setAttrErr != cudaSuccess) {
        outError = std::string("cudaMemPoolSetAttribute(release_threshold) failed: ")
            + (cudaGetErrorString(setAttrErr) ? cudaGetErrorString(setAttrErr) : "(unknown)");
        return false;
    }
    return true;
#else
    (void)key;
    (void)thresholdBytes;
    outError = "async mempool release threshold unsupported";
    return false;
#endif
}

void trace_async_mempool_release_policy(
    const SubmissionTransaction& transaction,
    PressureState pressureState,
    std::uint64_t thresholdBytes,
    bool applied,
    bool changed,
    const char* reason,
    const char* detail) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }
    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=mempool_release_policy")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " pressure_state=" + to_cstr(pressureState)
        + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(thresholdBytes))
        + " applied=" + std::to_string(applied ? 1 : 0)
        + " changed=" + std::to_string(changed ? 1 : 0)
        + " reason=" + (reason ? reason : "unspecified")
        + " detail=" + (detail ? detail : "none");
    JTRACE("MSALC", msg);
}

void maybe_apply_async_mempool_release_policy(
    const SubmissionTransaction& transaction,
    PressureState pressureState,
    const ResourceManagerConfigEffective& cfg,
    const char* reason) {
    const DeviceContextKey key = transaction.snapshot.deviceContextKey;
    const std::uint64_t thresholdBytes =
        async_mempool_release_threshold_bytes_for_state(cfg, pressureState);

    bool changed = false;
    {
        AllocatorBackendState& state = allocator_backend_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        AllocatorBackendContextEntry& entry =
            allocator_backend_get_or_init_locked(state, key, cfg);
        if (entry.active != AllocatorBackendMode::AsyncPool) {
            return;
        }
        changed =
            !entry.mempoolPolicyValid ||
            entry.mempoolReleaseThresholdBytes != thresholdBytes ||
            entry.mempoolPolicyPressureState != pressureState;
        if (!changed) {
            return;
        }
    }

    std::string applyError;
    const bool applied = set_async_mempool_release_threshold(
        key,
        thresholdBytes,
        applyError);

    {
        AllocatorBackendState& state = allocator_backend_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        AllocatorBackendContextEntry& entry =
            allocator_backend_get_or_init_locked(state, key, cfg);
        entry.mempoolPolicyValid = true;
        entry.mempoolPolicyApplied = applied;
        entry.mempoolReleaseThresholdBytes = thresholdBytes;
        entry.mempoolPolicyPressureState = pressureState;
    }

    trace_async_mempool_release_policy(
        transaction,
        pressureState,
        thresholdBytes,
        applied,
        changed,
        reason,
        applied ? "set_ok" : applyError.c_str());
}

void allocator_backend_retire_context(const DeviceContextKey& key) noexcept {
    AllocatorBackendState& state = allocator_backend_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.byContext.erase(key);
}

constexpr std::size_t tier_circuit_index(ResourceTier tier) noexcept {
    switch (tier) {
    case ResourceTier::Immutable:
        return 0u;
    case ResourceTier::Lut:
        return 1u;
    case ResourceTier::Scratch:
        return 2u;
    case ResourceTier::Graph:
        return 3u;
    default:
        return 0u;
    }
}

inline bool tier_circuit_blocks_admission(ResourceTier tier) noexcept {
    return tier == ResourceTier::Graph;
}

inline bool tier_circuit_policy_enabled(const ResourceManagerConfigEffective& cfg) noexcept {
    return cfg.tierErrorWindowMs > 0 &&
        cfg.tierErrorThreshold > 0 &&
        cfg.tierCircuitOpenMs > 0;
}

bool contains_ascii_case_insensitive(const std::string& haystack, const char* needle) noexcept {
    if (!needle || !*needle) {
        return true;
    }
    if (haystack.empty()) {
        return false;
    }
    const std::size_t needleLen = std::char_traits<char>::length(needle);
    if (needleLen == 0 || needleLen > haystack.size()) {
        return false;
    }
    for (std::size_t i = 0; i + needleLen <= haystack.size(); ++i) {
        bool match = true;
        for (std::size_t j = 0; j < needleLen; ++j) {
            const unsigned char a = static_cast<unsigned char>(haystack[i + j]);
            const unsigned char b = static_cast<unsigned char>(needle[j]);
            if (std::tolower(a) != std::tolower(b)) {
                match = false;
                break;
            }
        }
        if (match) {
            return true;
        }
    }
    return false;
}

bool is_allocator_oom_error(const std::string& error) noexcept {
    if (error.empty()) {
        return false;
    }
    return contains_ascii_case_insensitive(error, "out of memory") ||
        contains_ascii_case_insensitive(error, "memory allocation");
}

bool tier_circuit_should_count_failure(const std::string& error) noexcept {
    if (error.empty()) {
        return true;
    }
    if (error.rfind(kScratchExhaustedPrefix, 0) == 0) {
        return false;
    }
    if (error.rfind(kReservationDeferredPrefix, 0) == 0) {
        return false;
    }
    if (error.rfind("pressure_shed_noncritical:", 0) == 0) {
        return false;
    }
    if (error.rfind("pressure_copy_compute_guard:", 0) == 0) {
        return false;
    }
    return true;
}

inline std::uint64_t monotonic_time_ms() noexcept {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

inline std::uint64_t area_pixels_for_extent(int width, int height) noexcept {
    if (width <= 0 || height <= 0) {
        return 0;
    }
    const std::uint64_t w = static_cast<std::uint64_t>(width);
    const std::uint64_t h = static_cast<std::uint64_t>(height);
    if (w != 0 && h > (std::numeric_limits<std::uint64_t>::max() / w)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return w * h;
}

inline bool is_large_frame_extent(int width, int height) noexcept {
    const std::uint64_t area = area_pixels_for_extent(width, height);
    const int maxDim = std::max(width, height);
    return maxDim >= 7680 || area >= kLargeFrameThresholdPixels;
}

inline int adaptive_bucket_step_px(int width, int height) noexcept {
    const int maxDim = std::max(width, height);
    const std::uint64_t area = area_pixels_for_extent(width, height);
    if (maxDim >= 8192 || area >= static_cast<std::uint64_t>(8192ull * 4320ull)) {
        return kScratchBucketStepXLargePx;
    }
    if (maxDim >= 4096 || area >= static_cast<std::uint64_t>(4096ull * 2160ull)) {
        return kScratchBucketStepLargePx;
    }
    return kScratchBucketStepBasePx;
}

inline std::uint32_t round_up_bucket_dim(int value, int step) noexcept {
    const int safeStep = std::max(1, step);
    const int clampedValue = std::max(kScratchBucketMinPx, value);
    const int rounded = ((clampedValue + safeStep - 1) / safeStep) * safeStep;
    return static_cast<std::uint32_t>(std::max(kScratchBucketMinPx, rounded));
}

inline ScratchBucketKey make_scratch_bucket_key(
    int width,
    int height,
    ScratchWorkClass workClass) noexcept {
    const int step = adaptive_bucket_step_px(width, height);
    ScratchBucketKey key{};
    key.widthBucket = round_up_bucket_dim(width, step);
    key.heightBucket = round_up_bucket_dim(height, step);
    key.bucketStepPx = static_cast<std::uint16_t>(std::max(1, step));
    key.workClass = workClass;
    key.largeFrame = is_large_frame_extent(width, height);
    return key;
}

inline std::size_t effective_temp_scratch_bytes_cap(std::size_t requestBytes) noexcept {
    return std::max<std::size_t>(kMaxTempScratchBytes, requestBytes);
}

inline int scratch_wait_budget_ms(const ScratchBucketKey& key, std::size_t requestBytes) noexcept {
    if (requestBytes == 0) {
        return 0;
    }
    if (key.largeFrame || requestBytes >= kScratchWaitLargeRequestBytes) {
        return kScratchWaitMaxLargeMs;
    }
    if (requestBytes >= kScratchWaitMediumRequestBytes) {
        return kScratchWaitMaxMediumMs;
    }
    return kScratchWaitMaxMs;
}

inline std::size_t plane_bytes_for_extent(int width, int height) noexcept {
    if (width <= 0 || height <= 0) {
        return 0;
    }
    const std::size_t w = static_cast<std::size_t>(width);
    const std::size_t h = static_cast<std::size_t>(height);
    if (h > (std::numeric_limits<std::size_t>::max() / w)) {
        return std::numeric_limits<std::size_t>::max();
    }
    const std::size_t n = w * h;
    if (n > (std::numeric_limits<std::size_t>::max() / sizeof(float))) {
        return std::numeric_limits<std::size_t>::max();
    }
    return n * sizeof(float);
}

inline bool add_bytes_checked(std::size_t value, std::size_t add, std::size_t& out) noexcept {
    if (add == 0) {
        out = value;
        return true;
    }
    if (value > (std::numeric_limits<std::size_t>::max() - add)) {
        return false;
    }
    out = value + add;
    return true;
}

inline bool add_u64_checked(std::uint64_t value, std::uint64_t add, std::uint64_t& out) noexcept {
    if (add == 0) {
        out = value;
        return true;
    }
    if (value > (std::numeric_limits<std::uint64_t>::max() - add)) {
        return false;
    }
    out = value + add;
    return true;
}

inline bool mul_u64_checked(std::uint64_t a, std::uint64_t b, std::uint64_t& out) noexcept {
    if (a == 0 || b == 0) {
        out = 0;
        return true;
    }
    if (a > (std::numeric_limits<std::uint64_t>::max() / b)) {
        return false;
    }
    out = a * b;
    return true;
}

inline std::uint64_t& tier_bytes_at(
    std::array<std::uint64_t, 4>& values,
    ResourceTier tier) noexcept {
    return values[tier_circuit_index(tier)];
}

inline const std::uint64_t& tier_bytes_at(
    const std::array<std::uint64_t, 4>& values,
    ResourceTier tier) noexcept {
    return values[tier_circuit_index(tier)];
}

std::uint64_t tier_target_basis_points(
    const ResourceManagerConfigEffective& cfg,
    ResourceTier tier) noexcept {
    switch (tier) {
    case ResourceTier::Immutable:
        return cfg.tierTargetImmutableBp;
    case ResourceTier::Lut:
        return cfg.tierTargetLutBp;
    case ResourceTier::Scratch:
        return cfg.tierTargetScratchBp;
    case ResourceTier::Graph:
        return cfg.tierTargetGraphBp;
    default:
        return 0;
    }
}

// Graph-cache helpers are defined in split manager command sections included later in this TU.
std::uint64_t estimate_graph_cache_active_bytes_for_context(const DeviceContextKey& key) noexcept;
std::uint64_t evict_noncritical_graph_entries_for_context(const DeviceContextKey& key) noexcept;

std::uint64_t tier_target_bytes(
    const ResourceManagerConfigEffective& cfg,
    ResourceTier tier) noexcept {
    if (cfg.managerSoftTargetBytes == 0) {
        return 0;
    }
    const std::uint64_t bp = std::min<std::uint64_t>(
        tier_target_basis_points(cfg, tier),
        kBasisPointsDenominator);
    std::uint64_t weighted = 0;
    if (!mul_u64_checked(cfg.managerSoftTargetBytes, bp, weighted)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return weighted / kBasisPointsDenominator;
}

inline std::size_t saturating_u64_to_size_t(std::uint64_t value) noexcept {
    const std::uint64_t maxSizeT = static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
    if (value >= maxSizeT) {
        return std::numeric_limits<std::size_t>::max();
    }
    return static_cast<std::size_t>(value);
}

inline std::uint64_t non_negative_u64(int value) noexcept {
    return value > 0 ? static_cast<std::uint64_t>(value) : 0ull;
}

void add_snapshot_bytes(ManagerMemorySnapshot& snapshot, std::uint64_t bytes) noexcept {
    std::uint64_t next = 0;
    if (!add_u64_checked(snapshot.activeBytes, bytes, next)) {
        snapshot.activeBytes = std::numeric_limits<std::uint64_t>::max();
        snapshot.overflow = true;
        return;
    }
    snapshot.activeBytes = next;
}

std::uint64_t bytes_for_count_u64(std::uint64_t count, std::size_t elementBytes, bool& overflow) noexcept {
    std::uint64_t bytes = 0;
    if (!mul_u64_checked(count, static_cast<std::uint64_t>(elementBytes), bytes)) {
        overflow = true;
        return std::numeric_limits<std::uint64_t>::max();
    }
    return bytes;
}

std::uint64_t bytes_for_plane_extent_u64(int width, int height, bool& overflow) noexcept {
    const std::uint64_t w = non_negative_u64(width);
    const std::uint64_t h = non_negative_u64(height);
    std::uint64_t count = 0;
    if (!mul_u64_checked(w, h, count)) {
        overflow = true;
        return std::numeric_limits<std::uint64_t>::max();
    }
    return bytes_for_count_u64(count, sizeof(float), overflow);
}

void add_curve_bytes(const JuicerCuda::DeviceCurve& curve, ManagerMemorySnapshot& snapshot) noexcept {
    const std::uint64_t n = non_negative_u64(curve.n);
    if (n == 0) {
        return;
    }
    bool overflow = false;
    const std::uint64_t bytes = bytes_for_count_u64(n, sizeof(float), overflow);
    if (overflow) {
        snapshot.overflow = true;
    }
    if (curve.x) {
        add_snapshot_bytes(snapshot, bytes);
    }
    if (curve.y) {
        add_snapshot_bytes(snapshot, bytes);
    }
}

void add_spectral_tables_bytes(
    const JuicerCuda::Resources::DeviceSpectralTables& tables,
    ManagerMemorySnapshot& snapshot) noexcept {
    const std::uint64_t k = non_negative_u64(tables.K);
    if (k == 0) {
        return;
    }
    bool overflow = false;
    const std::uint64_t bytes = bytes_for_count_u64(k, sizeof(float), overflow);
    if (overflow) {
        snapshot.overflow = true;
    }
    if (tables.epsC) add_snapshot_bytes(snapshot, bytes);
    if (tables.epsM) add_snapshot_bytes(snapshot, bytes);
    if (tables.epsY) add_snapshot_bytes(snapshot, bytes);
    if (tables.Ax) add_snapshot_bytes(snapshot, bytes);
    if (tables.Ay) add_snapshot_bytes(snapshot, bytes);
    if (tables.Az) add_snapshot_bytes(snapshot, bytes);
    if (tables.baseMin) add_snapshot_bytes(snapshot, bytes);
}

void add_scan_medium_bytes(
    const JuicerCuda::Resources::DeviceScanMedium& medium,
    ManagerMemorySnapshot& snapshot) noexcept {
    add_spectral_tables_bytes(medium.tables, snapshot);
}

void add_scan_lut_bytes(
    const JuicerCuda::Resources::DeviceSpectralLut& lut,
    ManagerMemorySnapshot& snapshot) noexcept {
    if (!lut.log2XYZ || lut.res == 0u) {
        return;
    }
    const std::uint64_t res = static_cast<std::uint64_t>(lut.res);
    std::uint64_t count = 0;
    if (!mul_u64_checked(res, res, count) || !mul_u64_checked(count, res, count) || !mul_u64_checked(count, 3ull, count)) {
        snapshot.overflow = true;
        add_snapshot_bytes(snapshot, std::numeric_limits<std::uint64_t>::max());
        return;
    }
    bool overflow = false;
    const std::uint64_t bytes = bytes_for_count_u64(count, sizeof(double), overflow);
    if (overflow) {
        snapshot.overflow = true;
    }
    add_snapshot_bytes(snapshot, bytes);
}

void add_kernel_bytes(
    const JuicerCuda::Resources::DeviceGaussianKernel& kernel,
    ManagerMemorySnapshot& snapshot) noexcept {
    if (!kernel.weights || kernel.capacity <= 0) {
        return;
    }
    const std::uint64_t count = non_negative_u64(kernel.capacity);
    bool overflow = false;
    const std::uint64_t bytes = bytes_for_count_u64(count, sizeof(float), overflow);
    if (overflow) {
        snapshot.overflow = true;
    }
    add_snapshot_bytes(snapshot, bytes);
}

void add_optics_scratch_bytes(
    const JuicerCuda::Resources::DeviceOpticsScratch& scratch,
    ManagerMemorySnapshot& snapshot) noexcept {
    bool overflow = false;
    const std::uint64_t planeBytes = bytes_for_plane_extent_u64(scratch.width, scratch.height, overflow);
    if (overflow) {
        snapshot.overflow = true;
    }
    if (scratch.rgbR) add_snapshot_bytes(snapshot, planeBytes);
    if (scratch.rgbG) add_snapshot_bytes(snapshot, planeBytes);
    if (scratch.rgbB) add_snapshot_bytes(snapshot, planeBytes);
    if (scratch.blurred) add_snapshot_bytes(snapshot, planeBytes);
    if (scratch.aux) add_snapshot_bytes(snapshot, planeBytes);
    if (scratch.grainTmp) add_snapshot_bytes(snapshot, planeBytes);
    if (scratch.grainTmpShared) add_snapshot_bytes(snapshot, planeBytes);
    if (scratch.grainTmpMid) add_snapshot_bytes(snapshot, planeBytes);
    if (scratch.grainTmpCoarse) add_snapshot_bytes(snapshot, planeBytes);

    overflow = false;
    const std::uint64_t gateBytes = bytes_for_plane_extent_u64(scratch.gateWidth, scratch.gateHeight, overflow);
    if (overflow) {
        snapshot.overflow = true;
    }
    if (scratch.gateMask) {
        add_snapshot_bytes(snapshot, gateBytes);
    }
}

void add_spatial_dir_scratch_bytes(
    const JuicerCuda::Resources::DeviceSpatialDirScratch& scratch,
    ManagerMemorySnapshot& snapshot) noexcept {
    bool overflow = false;
    const std::uint64_t planeBytes = bytes_for_plane_extent_u64(scratch.width, scratch.height, overflow);
    if (overflow) {
        snapshot.overflow = true;
    }
    if (scratch.corrY) add_snapshot_bytes(snapshot, planeBytes);
    if (scratch.corrM) add_snapshot_bytes(snapshot, planeBytes);
    if (scratch.corrC) add_snapshot_bytes(snapshot, planeBytes);
}

void add_resources_active_bytes_locked(
    const JuicerCuda::Resources& resources,
    ManagerMemorySnapshot& snapshot) noexcept {
    add_curve_bytes(resources.densB, snapshot);
    add_curve_bytes(resources.densG, snapshot);
    add_curve_bytes(resources.densR, snapshot);
    add_curve_bytes(resources.dirDensB, snapshot);
    add_curve_bytes(resources.dirDensG, snapshot);
    add_curve_bytes(resources.dirDensR, snapshot);
    add_curve_bytes(resources.sensB, snapshot);
    add_curve_bytes(resources.sensG, snapshot);
    add_curve_bytes(resources.sensR, snapshot);

    const std::uint64_t layerN = non_negative_u64(resources.densityCurvesLayersN);
    if (layerN > 0) {
        bool overflow = false;
        const std::uint64_t layerBytes = bytes_for_count_u64(layerN, sizeof(float), overflow);
        if (overflow) {
            snapshot.overflow = true;
        }
        for (int layer = 0; layer < 3; ++layer) {
            for (int ch = 0; ch < 3; ++ch) {
                if (resources.densityCurvesLayers[layer][ch]) {
                    add_snapshot_bytes(snapshot, layerBytes);
                }
            }
        }
    }

    {
        const std::uint64_t k = non_negative_u64(resources.tablesK);
        if (k > 0) {
            bool overflow = false;
            const std::uint64_t bytes = bytes_for_count_u64(k, sizeof(float), overflow);
            if (overflow) {
                snapshot.overflow = true;
            }
            if (resources.tablesAx) add_snapshot_bytes(snapshot, bytes);
            if (resources.tablesAy) add_snapshot_bytes(snapshot, bytes);
            if (resources.tablesAz) add_snapshot_bytes(snapshot, bytes);
            if (resources.tablesIllum) add_snapshot_bytes(snapshot, bytes);
        }
    }

    if (resources.mallettBasis && resources.mallettBasisK > 0) {
        std::uint64_t count = non_negative_u64(resources.mallettBasisK);
        if (!mul_u64_checked(count, 3ull, count)) {
            snapshot.overflow = true;
            count = std::numeric_limits<std::uint64_t>::max();
        }
        bool overflow = false;
        const std::uint64_t bytes = bytes_for_count_u64(count, sizeof(float), overflow);
        if (overflow) {
            snapshot.overflow = true;
        }
        add_snapshot_bytes(snapshot, bytes);
    }

    add_scan_medium_bytes(resources.scanNegative, snapshot);
    add_scan_medium_bytes(resources.scanPrint, snapshot);
    add_scan_lut_bytes(resources.scanNegativeLut, snapshot);
    add_scan_lut_bytes(resources.scanPrintLut, snapshot);

    {
        bool overflow = false;
        const std::uint64_t sharedTmpBytes = bytes_for_plane_extent_u64(resources.sharedTmpWidth, resources.sharedTmpHeight, overflow);
        if (overflow) {
            snapshot.overflow = true;
        }
        if (resources.sharedTmpPlane) {
            add_snapshot_bytes(snapshot, sharedTmpBytes);
        }
    }

    add_optics_scratch_bytes(resources.scannerScratch, snapshot);
    add_spatial_dir_scratch_bytes(resources.spatialDirScratch, snapshot);

    add_kernel_bytes(resources.scannerLensBlurKernel, snapshot);
    add_kernel_bytes(resources.scannerUnsharpKernel, snapshot);
    add_kernel_bytes(resources.scannerGlareKernel, snapshot);
    add_kernel_bytes(resources.grainBlurKernel, snapshot);
    add_kernel_bytes(resources.grainBlurKernelMid, snapshot);
    add_kernel_bytes(resources.grainBlurKernelCoarse, snapshot);
    add_kernel_bytes(resources.spatialDirKernel, snapshot);
    for (int layer = 0; layer < 3; ++layer) {
        for (int ch = 0; ch < 3; ++ch) {
            add_kernel_bytes(resources.grainDyeKernel[layer][ch], snapshot);
        }
    }
    for (int ch = 0; ch < 3; ++ch) {
        add_kernel_bytes(resources.halationKernel[ch], snapshot);
        add_kernel_bytes(resources.halationScatterKernel[ch], snapshot);
    }

    if (resources.stbnData) {
        std::uint64_t count = non_negative_u64(resources.stbnWidth);
        std::uint64_t tmp = 0;
        if (!mul_u64_checked(count, non_negative_u64(resources.stbnHeight), tmp) ||
            !mul_u64_checked(tmp, non_negative_u64(resources.stbnFrames), tmp)) {
            snapshot.overflow = true;
            tmp = std::numeric_limits<std::uint64_t>::max();
        }
        add_snapshot_bytes(snapshot, tmp);
    }
    if (resources.wangTilesData) {
        std::uint64_t count = non_negative_u64(resources.wangWidth);
        std::uint64_t tmp = 0;
        if (!mul_u64_checked(count, non_negative_u64(resources.wangHeight), tmp) ||
            !mul_u64_checked(tmp, non_negative_u64(resources.wangCount), tmp)) {
            snapshot.overflow = true;
            tmp = std::numeric_limits<std::uint64_t>::max();
        }
        add_snapshot_bytes(snapshot, tmp);
    }
    if (resources.wangLutData) {
        std::uint64_t count = non_negative_u64(resources.wangColors);
        std::uint64_t tmp = count;
        if (!mul_u64_checked(tmp, count, tmp) ||
            !mul_u64_checked(tmp, count, tmp) ||
            !mul_u64_checked(tmp, count, tmp)) {
            snapshot.overflow = true;
            tmp = std::numeric_limits<std::uint64_t>::max();
        }
        add_snapshot_bytes(snapshot, tmp);
    }

    add_curve_bytes(resources.printDcC, snapshot);
    add_curve_bytes(resources.printDcM, snapshot);
    add_curve_bytes(resources.printDcY, snapshot);
    add_curve_bytes(resources.printSensC, snapshot);
    add_curve_bytes(resources.printSensM, snapshot);
    add_curve_bytes(resources.printSensY, snapshot);

    if (resources.printIllumFiltered && resources.printIllumK > 0) {
        bool overflow = false;
        const std::uint64_t bytes = bytes_for_count_u64(non_negative_u64(resources.printIllumK), sizeof(float), overflow);
        if (overflow) {
            snapshot.overflow = true;
        }
        add_snapshot_bytes(snapshot, bytes);
    }

    if (resources.hanatosLut && resources.hanatosN > 0) {
        std::uint64_t n = non_negative_u64(resources.hanatosN);
        std::uint64_t count = 0;
        if (!mul_u64_checked(n, n, count) ||
            !mul_u64_checked(count, static_cast<std::uint64_t>(81u), count)) {
            snapshot.overflow = true;
            count = std::numeric_limits<std::uint64_t>::max();
        }
        bool overflow = false;
        const std::uint64_t bytes = bytes_for_count_u64(count, sizeof(float), overflow);
        if (overflow) {
            snapshot.overflow = true;
        }
        add_snapshot_bytes(snapshot, bytes);
    }
    if (resources.hanatosLutIntegrated && resources.hanatosNIntegrated > 0) {
        std::uint64_t n = non_negative_u64(resources.hanatosNIntegrated);
        std::uint64_t count = 0;
        if (!mul_u64_checked(n, n, count) || !mul_u64_checked(count, 4ull, count)) {
            snapshot.overflow = true;
            count = std::numeric_limits<std::uint64_t>::max();
        }
        bool overflow = false;
        const std::uint64_t bytes = bytes_for_count_u64(count, sizeof(float), overflow);
        if (overflow) {
            snapshot.overflow = true;
        }
        add_snapshot_bytes(snapshot, bytes);
    }

    if (resources.scanErrorFlag) {
        add_snapshot_bytes(snapshot, sizeof(int));
    }
    if (resources.scanErrorHost) {
        add_snapshot_bytes(snapshot, sizeof(int));
    }

    if (resources.autoExposureExposureScale) add_snapshot_bytes(snapshot, sizeof(float));
    if (resources.autoExposureAutoEV) add_snapshot_bytes(snapshot, sizeof(double));
    if (resources.autoExposureValid) add_snapshot_bytes(snapshot, sizeof(int));

    if (resources.autoExposureScratch.maxYBits) {
        add_snapshot_bytes(snapshot, sizeof(unsigned int));
    }
    if (resources.autoExposureScratch.histogram) {
        add_snapshot_bytes(snapshot, static_cast<std::uint64_t>(2048u) * sizeof(unsigned int));
    }
    if (resources.autoExposureScratch.weightsX && resources.autoExposureScratch.weightsXCapacity > 0) {
        bool overflow = false;
        const std::uint64_t bytes = bytes_for_count_u64(
            non_negative_u64(resources.autoExposureScratch.weightsXCapacity),
            sizeof(float),
            overflow);
        if (overflow) {
            snapshot.overflow = true;
        }
        add_snapshot_bytes(snapshot, bytes);
    }
    if (resources.autoExposureScratch.weightsY && resources.autoExposureScratch.weightsYCapacity > 0) {
        bool overflow = false;
        const std::uint64_t bytes = bytes_for_count_u64(
            non_negative_u64(resources.autoExposureScratch.weightsYCapacity),
            sizeof(float),
            overflow);
        if (overflow) {
            snapshot.overflow = true;
        }
        add_snapshot_bytes(snapshot, bytes);
    }
    if (resources.autoExposureScratch.partialsA && resources.autoExposureScratch.partialCapacity > 0) {
        bool overflow = false;
        const std::uint64_t bytes = bytes_for_count_u64(
            non_negative_u64(resources.autoExposureScratch.partialCapacity),
            sizeof(JuicerCudaAutoExposurePartial),
            overflow);
        if (overflow) {
            snapshot.overflow = true;
        }
        add_snapshot_bytes(snapshot, bytes);
    }
    if (resources.autoExposureScratch.partialsB && resources.autoExposureScratch.partialCapacity > 0) {
        bool overflow = false;
        const std::uint64_t bytes = bytes_for_count_u64(
            non_negative_u64(resources.autoExposureScratch.partialCapacity),
            sizeof(JuicerCudaAutoExposurePartial),
            overflow);
        if (overflow) {
            snapshot.overflow = true;
        }
        add_snapshot_bytes(snapshot, bytes);
    }
}

void add_scratch_tier_bytes_locked(
    const JuicerCuda::Resources& resources,
    ManagerMemorySnapshot& scratchSnapshot) noexcept {
    bool overflow = false;
    const std::uint64_t sharedTmpBytes =
        bytes_for_plane_extent_u64(resources.sharedTmpWidth, resources.sharedTmpHeight, overflow);
    if (overflow) {
        scratchSnapshot.overflow = true;
    }
    if (resources.sharedTmpPlane) {
        add_snapshot_bytes(scratchSnapshot, sharedTmpBytes);
    }

    add_optics_scratch_bytes(resources.scannerScratch, scratchSnapshot);
    add_spatial_dir_scratch_bytes(resources.spatialDirScratch, scratchSnapshot);

    if (resources.scanErrorFlag) {
        add_snapshot_bytes(scratchSnapshot, sizeof(int));
    }
    if (resources.scanErrorHost) {
        add_snapshot_bytes(scratchSnapshot, sizeof(int));
    }

    if (resources.autoExposureExposureScale) add_snapshot_bytes(scratchSnapshot, sizeof(float));
    if (resources.autoExposureAutoEV) add_snapshot_bytes(scratchSnapshot, sizeof(double));
    if (resources.autoExposureValid) add_snapshot_bytes(scratchSnapshot, sizeof(int));

    if (resources.autoExposureScratch.maxYBits) {
        add_snapshot_bytes(scratchSnapshot, sizeof(unsigned int));
    }
    if (resources.autoExposureScratch.histogram) {
        add_snapshot_bytes(scratchSnapshot, static_cast<std::uint64_t>(2048u) * sizeof(unsigned int));
    }
    if (resources.autoExposureScratch.weightsX && resources.autoExposureScratch.weightsXCapacity > 0) {
        bool localOverflow = false;
        const std::uint64_t bytes = bytes_for_count_u64(
            non_negative_u64(resources.autoExposureScratch.weightsXCapacity),
            sizeof(float),
            localOverflow);
        if (localOverflow) {
            scratchSnapshot.overflow = true;
        }
        add_snapshot_bytes(scratchSnapshot, bytes);
    }
    if (resources.autoExposureScratch.weightsY && resources.autoExposureScratch.weightsYCapacity > 0) {
        bool localOverflow = false;
        const std::uint64_t bytes = bytes_for_count_u64(
            non_negative_u64(resources.autoExposureScratch.weightsYCapacity),
            sizeof(float),
            localOverflow);
        if (localOverflow) {
            scratchSnapshot.overflow = true;
        }
        add_snapshot_bytes(scratchSnapshot, bytes);
    }
    if (resources.autoExposureScratch.partialsA && resources.autoExposureScratch.partialCapacity > 0) {
        bool localOverflow = false;
        const std::uint64_t bytes = bytes_for_count_u64(
            non_negative_u64(resources.autoExposureScratch.partialCapacity),
            sizeof(JuicerCudaAutoExposurePartial),
            localOverflow);
        if (localOverflow) {
            scratchSnapshot.overflow = true;
        }
        add_snapshot_bytes(scratchSnapshot, bytes);
    }
    if (resources.autoExposureScratch.partialsB && resources.autoExposureScratch.partialCapacity > 0) {
        bool localOverflow = false;
        const std::uint64_t bytes = bytes_for_count_u64(
            non_negative_u64(resources.autoExposureScratch.partialCapacity),
            sizeof(JuicerCudaAutoExposurePartial),
            localOverflow);
        if (localOverflow) {
            scratchSnapshot.overflow = true;
        }
        add_snapshot_bytes(scratchSnapshot, bytes);
    }
}

void fill_tier_budget_snapshot(
    const SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const ResourceManagerConfigEffective& cfg,
    const ManagerMemorySnapshot& managerMemory,
    TierBudgetSnapshot& out) noexcept {
    out = TierBudgetSnapshot{};

    ManagerMemorySnapshot lutSnapshot{};
    ManagerMemorySnapshot scratchSnapshot{};
    {
        std::lock_guard<std::mutex> lock(resources.m);
        add_scan_lut_bytes(resources.scanNegativeLut, lutSnapshot);
        add_scan_lut_bytes(resources.scanPrintLut, lutSnapshot);
        add_scratch_tier_bytes_locked(resources, scratchSnapshot);
    }

    const std::uint64_t graphActiveBytes =
        estimate_graph_cache_active_bytes_for_context(transaction.snapshot.deviceContextKey);
    const std::uint64_t managerActiveBytesNoGraph =
        (managerMemory.activeBytes >= graphActiveBytes)
        ? (managerMemory.activeBytes - graphActiveBytes)
        : 0;
    const std::uint64_t lutActiveBytes = std::min(lutSnapshot.activeBytes, managerActiveBytesNoGraph);
    const std::uint64_t remainingAfterLut = managerActiveBytesNoGraph - lutActiveBytes;
    const std::uint64_t scratchActiveBytes = std::min(scratchSnapshot.activeBytes, remainingAfterLut);
    const std::uint64_t immutableActiveBytes = remainingAfterLut - scratchActiveBytes;

    tier_bytes_at(out.activeBytes, ResourceTier::Immutable) = immutableActiveBytes;
    tier_bytes_at(out.activeBytes, ResourceTier::Lut) = lutActiveBytes;
    tier_bytes_at(out.activeBytes, ResourceTier::Scratch) = scratchActiveBytes;
    tier_bytes_at(out.activeBytes, ResourceTier::Graph) = graphActiveBytes;

    const std::uint64_t immutableReclaimableBytes =
        (managerMemory.reclaimableBytes >= graphActiveBytes)
        ? (managerMemory.reclaimableBytes - graphActiveBytes)
        : 0;
    tier_bytes_at(out.reclaimableBytes, ResourceTier::Immutable) = immutableReclaimableBytes;
    tier_bytes_at(out.reclaimableBytes, ResourceTier::Lut) = 0;
    tier_bytes_at(out.reclaimableBytes, ResourceTier::Scratch) = 0;
    tier_bytes_at(out.reclaimableBytes, ResourceTier::Graph) = graphActiveBytes;

    out.totalActiveBytes = managerActiveBytesNoGraph;
    if (!add_u64_checked(out.totalActiveBytes, graphActiveBytes, out.totalActiveBytes)) {
        out.totalActiveBytes = std::numeric_limits<std::uint64_t>::max();
        out.overflow = true;
    }
    out.totalReclaimableBytes = immutableReclaimableBytes;
    if (!add_u64_checked(
            out.totalReclaimableBytes,
            tier_bytes_at(out.reclaimableBytes, ResourceTier::Graph),
            out.totalReclaimableBytes)) {
        out.totalReclaimableBytes = std::numeric_limits<std::uint64_t>::max();
        out.overflow = true;
    }

    constexpr ResourceTier kTierOrder[4] = {
        ResourceTier::Immutable,
        ResourceTier::Lut,
        ResourceTier::Scratch,
        ResourceTier::Graph
    };
    for (ResourceTier tier : kTierOrder) {
        const std::uint64_t targetBytes = tier_target_bytes(cfg, tier);
        tier_bytes_at(out.targetBytes, tier) = targetBytes;
        const std::uint64_t activeBytes = tier_bytes_at(out.activeBytes, tier);
        if (activeBytes > targetBytes) {
            const std::uint64_t overBytes = activeBytes - targetBytes;
            tier_bytes_at(out.overTargetBytes, tier) = overBytes;
            out.anyOverTarget = true;
            if (overBytes > out.dominantOverTargetBytes) {
                out.dominantOverTargetBytes = overBytes;
                out.dominantOverTargetTier = tier;
            }
        }
    }
}

ManagerMemorySnapshot snapshot_manager_memory(JuicerCuda::Resources& resources) noexcept {
    ManagerMemorySnapshot snapshot{};
    {
        std::lock_guard<std::mutex> lock(resources.m);
        add_resources_active_bytes_locked(resources, snapshot);
        snapshot.retirePendingBytes = static_cast<std::uint64_t>(
            std::min<std::size_t>(
                resources.retireBytes,
                static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())));
    }
    snapshot.reclaimableBytes = snapshot.retirePendingBytes;
    snapshot.transientNonManagerBytes =
        global_state().transientNonManagerBytes.load(std::memory_order_relaxed);
    return snapshot;
}

void publish_manager_memory_snapshot(const ManagerMemorySnapshot& snapshot) noexcept {
    ResourceManagerState& state = global_state();
    state.managerActiveBytes.store(snapshot.activeBytes, std::memory_order_relaxed);
    state.managerReclaimableBytes.store(snapshot.reclaimableBytes, std::memory_order_relaxed);
    state.managerRetirePendingBytes.store(snapshot.retirePendingBytes, std::memory_order_relaxed);
    state.transientNonManagerBytes.store(snapshot.transientNonManagerBytes, std::memory_order_relaxed);
}

inline bool pressure_policy_enabled(const ResourceManagerConfigEffective& cfg) noexcept {
    return cfg.managerSoftTargetBytes > 0;
}

inline bool should_collect_manager_memory_snapshots(const ResourceManagerConfigEffective& cfg) noexcept {
    return pressure_policy_enabled(cfg) || JTRACE_ENABLED(3);
}

std::uint64_t transient_reservation_cap_bytes(const ResourceManagerConfigEffective& cfg) noexcept {
    if (cfg.managerSoftTargetBytes == 0) {
        return kTransientReservationCapDefaultBytes;
    }
    const std::uint64_t quarterTarget = cfg.managerSoftTargetBytes / 4ull;
    return std::max<std::uint64_t>(
        kTransientReservationThresholdDefaultBytes,
        std::min<std::uint64_t>(kTransientReservationCapDefaultBytes, quarterTarget));
}

std::uint64_t transient_reservation_threshold_bytes(const ResourceManagerConfigEffective& cfg) noexcept {
    const std::uint64_t capBytes = transient_reservation_cap_bytes(cfg);
    if (capBytes == 0) {
        return 0;
    }
    return std::min<std::uint64_t>(kTransientReservationThresholdDefaultBytes, capBytes);
}

std::uint64_t builder_reservation_cap_bytes(
    const ResourceManagerConfigEffective& cfg,
    BuilderReservationTier tier) noexcept {
    std::uint32_t configCapMb = 0;
    std::uint64_t defaultCapBytes = kScratchBuilderReservationCapDefaultBytes;
    switch (tier) {
    case BuilderReservationTier::Scratch:
        configCapMb = cfg.scratchBuilderBytesInFlightLimitMB;
        defaultCapBytes = kScratchBuilderReservationCapDefaultBytes;
        break;
    case BuilderReservationTier::Lut:
        configCapMb = cfg.lutBuilderBytesInFlightLimitMB;
        defaultCapBytes = kLutBuilderReservationCapDefaultBytes;
        break;
    case BuilderReservationTier::Graph:
        configCapMb = cfg.graphBuilderBytesInFlightLimitMB;
        defaultCapBytes = kGraphBuilderReservationCapDefaultBytes;
        break;
    default:
        configCapMb = cfg.scratchBuilderBytesInFlightLimitMB;
        defaultCapBytes = kScratchBuilderReservationCapDefaultBytes;
        break;
    }

    std::uint64_t capBytes = 0;
    if (!mul_u64_checked(static_cast<std::uint64_t>(configCapMb), kBytesPerMiB, capBytes)) {
        capBytes = std::numeric_limits<std::uint64_t>::max();
    }
    if (capBytes == 0) {
        capBytes = defaultCapBytes;
    }
    return std::max<std::uint64_t>(kBuilderReservationThresholdDefaultBytes, capBytes);
}

std::uint64_t builder_reservation_threshold_bytes(
    const ResourceManagerConfigEffective& cfg,
    BuilderReservationTier tier) noexcept {
    const std::uint64_t capBytes = builder_reservation_cap_bytes(cfg, tier);
    if (capBytes == 0) {
        return 0;
    }
    return std::min<std::uint64_t>(kBuilderReservationThresholdDefaultBytes, capBytes);
}

std::uint64_t& builder_context_bytes_for_tier(
    BuilderReservationContextState& contextState,
    BuilderReservationTier tier) noexcept {
    switch (tier) {
    case BuilderReservationTier::Scratch:
        return contextState.inFlightScratchBytes;
    case BuilderReservationTier::Lut:
        return contextState.inFlightLutBytes;
    case BuilderReservationTier::Graph:
        return contextState.inFlightGraphBytes;
    default:
        return contextState.inFlightScratchBytes;
    }
}

std::uint64_t& builder_total_bytes_for_tier(
    BuilderReservationState& state,
    BuilderReservationTier tier) noexcept {
    switch (tier) {
    case BuilderReservationTier::Scratch:
        return state.totalScratchInFlightBytes;
    case BuilderReservationTier::Lut:
        return state.totalLutInFlightBytes;
    case BuilderReservationTier::Graph:
        return state.totalGraphInFlightBytes;
    default:
        return state.totalScratchInFlightBytes;
    }
}

std::atomic<std::uint64_t>& builder_global_gauge_for_tier(
    ResourceManagerState& state,
    BuilderReservationTier tier) noexcept {
    switch (tier) {
    case BuilderReservationTier::Scratch:
        return state.scratchBuilderBytesInFlight;
    case BuilderReservationTier::Lut:
        return state.lutBuilderBytesInFlight;
    case BuilderReservationTier::Graph:
        return state.graphBuilderBytesInFlight;
    default:
        return state.scratchBuilderBytesInFlight;
    }
}

bool builder_context_has_inflight(const BuilderReservationContextState& contextState) noexcept {
    return contextState.inFlightScratchBytes > 0 ||
        contextState.inFlightLutBytes > 0 ||
        contextState.inFlightGraphBytes > 0;
}

std::uint64_t upload_reservation_cap_bytes(const ResourceManagerConfigEffective& cfg) noexcept {
    std::uint64_t configCapBytes = 0;
    if (!mul_u64_checked(
            static_cast<std::uint64_t>(cfg.uploadBytesInFlightLimitMB),
            kBytesPerMiB,
            configCapBytes)) {
        configCapBytes = std::numeric_limits<std::uint64_t>::max();
    }
    if (configCapBytes == 0) {
        configCapBytes = kUploadReservationCapDefaultBytes;
    }
    if (cfg.managerSoftTargetBytes == 0) {
        return std::max<std::uint64_t>(kUploadReservationThresholdDefaultBytes, configCapBytes);
    }
    const std::uint64_t quarterTarget = cfg.managerSoftTargetBytes / 4ull;
    const std::uint64_t boundedCap = std::min<std::uint64_t>(configCapBytes, quarterTarget);
    return std::max<std::uint64_t>(kUploadReservationThresholdDefaultBytes, boundedCap);
}

std::uint64_t upload_reservation_threshold_bytes(const ResourceManagerConfigEffective& cfg) noexcept {
    const std::uint64_t capBytes = upload_reservation_cap_bytes(cfg);
    if (capBytes == 0) {
        return 0;
    }
    return std::min<std::uint64_t>(kUploadReservationThresholdDefaultBytes, capBytes);
}

void add_estimate_bytes_u64(std::uint64_t bytes, std::uint64_t& total, bool& overflow) noexcept {
    if (overflow) {
        return;
    }
    std::uint64_t next = 0;
    if (!add_u64_checked(total, bytes, next)) {
        total = std::numeric_limits<std::uint64_t>::max();
        overflow = true;
        return;
    }
    total = next;
}

template <typename T>
void add_vector_upload_estimate_bytes(
    const std::vector<T>& values,
    std::uint64_t& total,
    bool& overflow) noexcept {
    if (values.empty() || overflow) {
        return;
    }
    std::uint64_t bytes = 0;
    if (!mul_u64_checked(
            static_cast<std::uint64_t>(values.size()),
            static_cast<std::uint64_t>(sizeof(T)),
            bytes)) {
        total = std::numeric_limits<std::uint64_t>::max();
        overflow = true;
        return;
    }
    add_estimate_bytes_u64(bytes, total, overflow);
}

void add_curve_upload_estimate_bytes(
    const Spectral::Curve& curve,
    std::uint64_t& total,
    bool& overflow) noexcept {
    const std::size_t n = std::min(curve.lambda_nm.size(), curve.linear.size());
    if (n == 0 || overflow) {
        return;
    }
    std::uint64_t bytes = 0;
    if (!mul_u64_checked(static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(2u * sizeof(float)), bytes)) {
        total = std::numeric_limits<std::uint64_t>::max();
        overflow = true;
        return;
    }
    add_estimate_bytes_u64(bytes, total, overflow);
}

void add_scan_medium_upload_estimate_bytes(
    const Scanner::ScannerMediumRuntime& medium,
    std::uint64_t& total,
    bool& overflow) noexcept {
    const Spectral::SpectralTables* t = medium.tables;
    if (!t) {
        return;
    }
    add_vector_upload_estimate_bytes(t->epsC, total, overflow);
    add_vector_upload_estimate_bytes(t->epsM, total, overflow);
    add_vector_upload_estimate_bytes(t->epsY, total, overflow);
    add_vector_upload_estimate_bytes(t->Ax, total, overflow);
    add_vector_upload_estimate_bytes(t->Ay, total, overflow);
    add_vector_upload_estimate_bytes(t->Az, total, overflow);
    if (t->hasBaseline) {
        add_vector_upload_estimate_bytes(t->baseMin, total, overflow);
    }
}

std::uint64_t estimate_upload_core_request_bytes(
    JuicerCuda::Resources& resources,
    const WorkingState& ws) noexcept {
    const std::uint64_t wsCoreHash =
        (ws.uploadCoreHash != 0) ? ws.uploadCoreHash : ws.coreHash;
    const std::uint64_t wsDirHash = ws.dirHash;
    if (wsCoreHash == 0 || wsDirHash == 0) {
        return kUploadReservationThresholdDefaultBytes;
    }

    bool coreUpToDate = false;
    bool dirUpToDate = false;
    bool needStbnUpload = false;
    bool needWangUpload = false;
    {
        std::lock_guard<std::mutex> lock(resources.m);
        coreUpToDate = (resources.uploadedCoreHash != 0) && (resources.uploadedCoreHash == wsCoreHash);
        dirUpToDate = (resources.uploadedDirHash != 0) && (resources.uploadedDirHash == wsDirHash);
        needStbnUpload = (resources.stbnData == nullptr);
        needWangUpload = (resources.wangTilesData == nullptr) || (resources.wangLutData == nullptr);
    }

    std::uint64_t estimateBytes = 0;
    bool overflow = false;
    if (needStbnUpload) {
        add_estimate_bytes_u64(kStbnUploadDefaultBytes, estimateBytes, overflow);
    }
    if (needWangUpload) {
        add_estimate_bytes_u64(kWangTilesUploadDefaultBytes, estimateBytes, overflow);
        add_estimate_bytes_u64(kWangLutUploadDefaultBytes, estimateBytes, overflow);
    }

    if (coreUpToDate && dirUpToDate) {
        return estimateBytes;
    }

    if (coreUpToDate && !dirUpToDate) {
        add_curve_upload_estimate_bytes(ws.dirDensB, estimateBytes, overflow);
        add_curve_upload_estimate_bytes(ws.dirDensG, estimateBytes, overflow);
        add_curve_upload_estimate_bytes(ws.dirDensR, estimateBytes, overflow);
        return estimateBytes;
    }

    add_curve_upload_estimate_bytes(ws.densB, estimateBytes, overflow);
    add_curve_upload_estimate_bytes(ws.densG, estimateBytes, overflow);
    add_curve_upload_estimate_bytes(ws.densR, estimateBytes, overflow);
    add_curve_upload_estimate_bytes(ws.dirDensB, estimateBytes, overflow);
    add_curve_upload_estimate_bytes(ws.dirDensG, estimateBytes, overflow);
    add_curve_upload_estimate_bytes(ws.dirDensR, estimateBytes, overflow);
    add_curve_upload_estimate_bytes(ws.sensB, estimateBytes, overflow);
    add_curve_upload_estimate_bytes(ws.sensG, estimateBytes, overflow);
    add_curve_upload_estimate_bytes(ws.sensR, estimateBytes, overflow);
    add_vector_upload_estimate_bytes(ws.tablesRef.Ax, estimateBytes, overflow);
    add_vector_upload_estimate_bytes(ws.tablesRef.Ay, estimateBytes, overflow);
    add_vector_upload_estimate_bytes(ws.tablesRef.Az, estimateBytes, overflow);
    add_vector_upload_estimate_bytes(ws.tablesRef.illum, estimateBytes, overflow);

    for (int layer = 0; layer < 3; ++layer) {
        for (int ch = 0; ch < 3; ++ch) {
            add_vector_upload_estimate_bytes(ws.densityCurvesLayers[layer][ch], estimateBytes, overflow);
        }
    }

    add_scan_medium_upload_estimate_bytes(ws.negativeMediumRuntime, estimateBytes, overflow);
    add_scan_medium_upload_estimate_bytes(ws.printMediumRuntime, estimateBytes, overflow);

    if (ws.printRT && Print::profile_is_valid(ws.printRT->profile)) {
        const Print::Profile& p = ws.printRT->profile;
        add_curve_upload_estimate_bytes(p.dcC, estimateBytes, overflow);
        add_curve_upload_estimate_bytes(p.dcM, estimateBytes, overflow);
        add_curve_upload_estimate_bytes(p.dcY, estimateBytes, overflow);
        add_vector_upload_estimate_bytes(p.sensC_log.linear, estimateBytes, overflow);
        add_vector_upload_estimate_bytes(p.sensM_log.linear, estimateBytes, overflow);
        add_vector_upload_estimate_bytes(p.sensY_log.linear, estimateBytes, overflow);
    }

    // Keep a deterministic floor so large rebuilds always enter upload reservation admission.
    if (estimateBytes < kUploadReservationThresholdDefaultBytes) {
        estimateBytes = kUploadReservationThresholdDefaultBytes;
    }
    return estimateBytes;
}

void refill_upload_fairness_tokens(
    UploadReservationContextState::FairnessEntry& entry,
    const ResourceManagerConfigEffective& cfg,
    std::uint64_t nowMs) noexcept {
    const std::uint32_t sharedPerTick = std::max<std::uint32_t>(1u, cfg.uploadFairnessTokensPerTick);
    const std::uint32_t criticalPerTick = std::min<std::uint32_t>(
        std::max<std::uint32_t>(1u, cfg.criticalUploadReservedTokens),
        sharedPerTick);

    if (entry.lastRefillMs == 0) {
        entry.sharedTokens = sharedPerTick;
        entry.criticalTokens = criticalPerTick;
        entry.lastRefillMs = nowMs;
        return;
    }

    const std::uint64_t elapsedMs = (nowMs > entry.lastRefillMs) ? (nowMs - entry.lastRefillMs) : 0;
    if (elapsedMs < kUploadFairnessTickMs) {
        return;
    }

    entry.sharedTokens = sharedPerTick;
    entry.criticalTokens = criticalPerTick;
    entry.lastRefillMs = nowMs;
}

void refill_builder_fairness_tokens(
    BuilderReservationContextState::FairnessEntry& entry,
    const ResourceManagerConfigEffective& cfg,
    std::uint64_t nowMs) noexcept {
    const std::uint32_t sharedPerTick = std::max<std::uint32_t>(1u, cfg.builderFairnessTokensPerTick);
    const std::uint32_t criticalPerTick = std::min<std::uint32_t>(
        std::max<std::uint32_t>(1u, cfg.criticalBuilderReservedTokens),
        sharedPerTick);

    if (entry.lastRefillMs == 0) {
        entry.sharedTokens = sharedPerTick;
        entry.criticalTokens = criticalPerTick;
        entry.lastRefillMs = nowMs;
        return;
    }

    const std::uint64_t elapsedMs = (nowMs > entry.lastRefillMs) ? (nowMs - entry.lastRefillMs) : 0;
    if (elapsedMs < kBuilderFairnessTickMs) {
        return;
    }

    entry.sharedTokens = sharedPerTick;
    entry.criticalTokens = criticalPerTick;
    entry.lastRefillMs = nowMs;
}

std::uint64_t estimate_scan_lut_upload_bytes(
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    bool negativeMedium) noexcept {
    const Scanner::ScannerStaticKey& staticKey = negativeMedium ? ws.negativeStaticKey : ws.printStaticKey;
    const std::uint32_t res =
        ResourceManager::normalize_scan_lut_resolution(staticKey.lutResolution);
    std::uint64_t voxelCount = 0;
    if (!mul_u64_checked(static_cast<std::uint64_t>(res), static_cast<std::uint64_t>(res), voxelCount)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    if (!mul_u64_checked(voxelCount, static_cast<std::uint64_t>(res), voxelCount)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    std::uint64_t values = 0;
    if (!mul_u64_checked(voxelCount, 3ull, values)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    std::uint64_t bytes = 0;
    if (!mul_u64_checked(values, static_cast<std::uint64_t>(sizeof(double)), bytes)) {
        return std::numeric_limits<std::uint64_t>::max();
    }

    const Scanner::ScannerMediumRuntime& medium = negativeMedium ? ws.negativeMediumRuntime : ws.printMediumRuntime;
    if (!medium.tables || medium.tables->tablesHash == 0 || medium.range.digest == 0) {
        return 0;
    }
    const std::uint64_t expectedHash = ResourceManager::make_scan_lut_key_digest(
        static_cast<std::uint32_t>(medium.medium),
        medium.tables->tablesHash,
        medium.range.digest,
        res);
    if (expectedHash == 0) {
        return 0;
    }

    bool cached = false;
    {
        std::lock_guard<std::mutex> lock(resources.m);
        const JuicerCuda::Resources::DeviceSpectralLut& dst =
            negativeMedium ? resources.scanNegativeLut : resources.scanPrintLut;
        cached = dst.log2XYZ && dst.res == res && dst.hash == expectedHash;
    }
    return cached ? 0 : bytes;
}

std::uint64_t estimate_print_illuminant_upload_bytes(
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    const Print::Runtime& prt,
    const Print::Params& params) noexcept {
    const int k = Spectral::gShape.K;
    if (k <= 0) {
        return 0;
    }
    const std::uint64_t wsCoreHash =
        (ws.uploadCoreHash != 0) ? ws.uploadCoreHash : ws.coreHash;
    if (wsCoreHash == 0) {
        return 0;
    }
    const float yKey = std::isfinite(params.yFilter) ? params.yFilter : 0.0f;
    const float mKey = std::isfinite(params.mFilter) ? params.mFilter : 0.0f;
    const float cKey = std::isfinite(params.cFilter) ? params.cFilter : 0.0f;
    const std::uint64_t neutralFilterHash =
        (prt.neutralFilterHash != 0) ? prt.neutralFilterHash : Print::kDefaultNeutralFilterHash;

    bool cached = false;
    {
        std::lock_guard<std::mutex> lock(resources.m);
        cached =
            resources.printIllumFiltered &&
            resources.printIllumK == k &&
            resources.printIllumShapeK == k &&
            resources.printIllumCoreHash == wsCoreHash &&
            resources.printIllumYShiftSteps == yKey &&
            resources.printIllumMShiftSteps == mKey &&
            resources.printIllumCShiftSteps == cKey &&
            resources.printIllumNeutralFilterHash == neutralFilterHash;
    }
    if (cached) {
        return 0;
    }
    std::uint64_t bytes = 0;
    if (!mul_u64_checked(
            static_cast<std::uint64_t>(k),
            static_cast<std::uint64_t>(sizeof(float)),
            bytes)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return bytes;
}


// Split implementation sections (single-TU include model to preserve exact behavior while
// reducing monolithic file size and keeping ownership boundaries explicit).
#include "Cuda/ResourceManager/JuicerCudaResourceManagerAdmission.cpp"
#include "Cuda/ResourceManager/JuicerCudaResourceManagerSubmission.cpp"
#include "Cuda/ResourceManager/JuicerCudaResourceManagerCommands.cpp"

bool query_submission_active(const SubmissionTransaction& transaction) noexcept {
    QueryReadOnlyGuard queryGuard("query_submission_active");
    (void)queryGuard;
    return transaction.active;
}

AllocatorBackendMode query_allocator_backend_mode(const DeviceContextKey& key) noexcept {
    QueryReadOnlyGuard queryGuard("query_allocator_backend_mode", &key);
    (void)queryGuard;
    if (key.deviceId < 0) {
        return AllocatorBackendMode::Legacy;
    }
    AllocatorBackendMode activeMode = AllocatorBackendMode::Legacy;
    if (allocator_backend_try_get_active_mode(key, activeMode)) {
        return activeMode;
    }

    // Query APIs are read-only: derive an uncached mode when command path has not initialized state yet.
    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    const AllocatorBackendContextEntry derived =
        compute_allocator_backend_context_entry(key, cfg);
    return derived.active;
}

bool begin_submission(
    SubmissionTransaction& outTransaction,
    const SubmissionSnapshot& snapshot,
    std::string& outError) {
    outError.clear();
    MetadataMutationGuard mutationGuard("begin_submission", &snapshot.deviceContextKey);
    if (!mutationGuard.ok()) {
        outError = "metadata mutation guard rejected begin_submission";
        return false;
    }

    if (outTransaction.active) {
        outError = "submission transaction already active";
        return false;
    }

    ResourceManagerState& state = global_state();
    outTransaction.transactionId = state.nextTransactionId.fetch_add(1, std::memory_order_relaxed);
    if (outTransaction.transactionId == 0) {
        outTransaction.transactionId = state.nextTransactionId.fetch_add(1, std::memory_order_relaxed);
    }

    outTransaction.snapshot = snapshot;
    const std::uint32_t requestedTraceSchemaVersion = outTransaction.snapshot.traceSchemaVersion;
    if (!trace_schema_matches_contract(requestedTraceSchemaVersion)) {
        outError = "trace schema mismatch";
        telemetry_record_trace_schema_mismatch();
        telemetry_trace_schema_mismatch(
            outTransaction.transactionId,
            outTransaction.snapshot.snapshotId,
            requestedTraceSchemaVersion);
        return false;
    }

    std::uint64_t registryGeneration = state.registryGeneration.load(std::memory_order_relaxed);
    std::uint64_t contextEpoch = state.contextEpoch.load(std::memory_order_relaxed);
    if (registryGeneration == 0) {
        registryGeneration = 1;
    }
    if (contextEpoch == 0) {
        contextEpoch = 1;
    }
    outTransaction.snapshot.registryGeneration = registryGeneration;
    outTransaction.snapshot.contextEpoch = contextEpoch;
    outTransaction.snapshot.keySchemaVersion = std::max<std::uint32_t>(1u, outTransaction.snapshot.keySchemaVersion);
    outTransaction.snapshot.traceSchemaVersion = sanitize_trace_schema_version(outTransaction.snapshot.traceSchemaVersion);
    outTransaction.snapshot.keyDigests = normalize_key_digests(outTransaction.snapshot.keyDigests);

    std::uint64_t leaseGeneration = state.nextLeaseGeneration.fetch_add(1, std::memory_order_relaxed);
    if (leaseGeneration == 0) {
        leaseGeneration = state.nextLeaseGeneration.fetch_add(1, std::memory_order_relaxed);
    }
    outTransaction.leaseGeneration = leaseGeneration;
    outTransaction.active = true;
    outTransaction.committed = false;

    (void)registry_get_or_create(outTransaction.snapshot.deviceContextKey);
    if (!validate_lifecycle_for_stage(outTransaction, "begin", false, &outError)) {
        outTransaction.active = false;
        outTransaction.committed = false;
        return false;
    }
    if (!registry_note_submission_begin(outTransaction.snapshot.deviceContextKey)) {
        outError = "registry submission-begin tracking rejected";
        outTransaction.active = false;
        outTransaction.committed = false;
        return false;
    }
    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    ensure_allocator_backend_mode_initialized(outTransaction, cfg);
    state_note_latest_snapshot(outTransaction.snapshot);
    maybe_apply_async_mempool_release_policy(
        outTransaction,
        PressureState::Normal,
        cfg,
        "begin_submission");
    trace_optional_heuristic_surfaces_once(
        outTransaction,
        cfg,
        "begin_submission");
    telemetry_trace_schema_announcement(
        outTransaction.transactionId,
        outTransaction.snapshot.snapshotId,
        outTransaction.snapshot.traceSchemaVersion);
    telemetry_record_begin_submission();
    return true;
}

bool acquire_plan(
    SubmissionTransaction& transaction,
    std::string& outError) {
    outError.clear();
    MetadataMutationGuard mutationGuard("acquire_plan", &transaction.snapshot.deviceContextKey);
    if (!mutationGuard.ok()) {
        outError = "metadata mutation guard rejected acquire_plan";
        return false;
    }
    const std::uint64_t acquireId = telemetry_next_acquire_attempt_id();

    if (!validate_lifecycle_for_stage(transaction, "acquire", false, &outError)) {
        const ResourcePlan errorPlan = make_uniform_resource_plan(AcquireStatus::Error);
        telemetry_record_acquire_status(AcquireStatus::Error);
        for (ResourceKind kind : kResourceKindOrder) {
            telemetry_record_acquire_status_for_kind(kind, AcquireStatus::Error);
        }
        telemetry_trace_acquire(
            acquireId,
            transaction.transactionId,
            transaction.snapshot.snapshotId,
            transaction.snapshot.traceSchemaVersion,
            AcquireStatus::Error,
            errorPlan,
            false);
        return false;
    }

    if (!validate_stale_tuple_for_stage(
            transaction,
            "acquire",
            LeaseObservationMode::ActiveOnly,
            "stale transaction in acquire path (reason=",
            &outError,
            false,
            nullptr)) {
        const ResourcePlan errorPlan = make_uniform_resource_plan(AcquireStatus::Error);
        telemetry_record_acquire_status(AcquireStatus::Error);
        for (ResourceKind kind : kResourceKindOrder) {
            telemetry_record_acquire_status_for_kind(kind, AcquireStatus::Error);
        }
        telemetry_trace_acquire(
            acquireId,
            transaction.transactionId,
            transaction.snapshot.snapshotId,
            transaction.snapshot.traceSchemaVersion,
            AcquireStatus::Error,
            errorPlan,
            false);
        return false;
    }

    if (!transaction.active) {
        const ResourcePlan errorPlan = make_uniform_resource_plan(AcquireStatus::Error);
        outError = "submission transaction is not active";
        telemetry_record_acquire_status(AcquireStatus::Error);
        for (ResourceKind kind : kResourceKindOrder) {
            telemetry_record_acquire_status_for_kind(kind, AcquireStatus::Error);
        }
        telemetry_trace_acquire(
            acquireId,
            transaction.transactionId,
            transaction.snapshot.snapshotId,
            transaction.snapshot.traceSchemaVersion,
            AcquireStatus::Error,
            errorPlan,
            false);
        return false;
    }

    SubmissionSnapshot& snapshot = transaction.snapshot;
    if (!trace_schema_matches_contract(snapshot.traceSchemaVersion)) {
        const ResourcePlan errorPlan = make_uniform_resource_plan(AcquireStatus::Error);
        outError = "trace schema mismatch";
        telemetry_record_trace_schema_mismatch();
        telemetry_trace_schema_mismatch(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion);
        telemetry_record_acquire_status(AcquireStatus::Error);
        for (ResourceKind kind : kResourceKindOrder) {
            telemetry_record_acquire_status_for_kind(kind, AcquireStatus::Error);
        }
        telemetry_trace_acquire(
            acquireId,
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            AcquireStatus::Error,
            errorPlan,
            false);
        return false;
    }

    if (snapshot.keySchemaVersion == 0) {
        snapshot.keySchemaVersion = 1;
        telemetry_record_module_boundary_violation();
        telemetry_trace_module_boundary_violation(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            "key_schema_version_zero_sanitized");
    }

    const KeyDigests rawDigests = snapshot.keyDigests;
    snapshot.keyDigests = normalize_key_digests(snapshot.keyDigests);
    if (rawDigests.uploadCoreHash != snapshot.keyDigests.uploadCoreHash ||
        rawDigests.dirHash != snapshot.keyDigests.dirHash ||
        rawDigests.scannerHash != snapshot.keyDigests.scannerHash ||
        rawDigests.autoExposureHash != snapshot.keyDigests.autoExposureHash) {
        telemetry_trace_key_normalization(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            rawDigests,
            snapshot.keyDigests);
    }

    std::uint64_t expectedSnapshotIdForFrame = 0;
    bool frameSnapshotMismatch = false;
    {
        const FrameSnapshotKey frameKey{ snapshot.instanceToken.value, snapshot.deviceContextKey };
        FrameSnapshotState& state = frame_snapshot_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        FrameSnapshotEntry& entry = state.bySubmissionKey[frameKey];
        const bool sameFrameToken = entry.valid && entry.frameToken == snapshot.frameToken.value;
        const bool sameSchema = entry.valid && entry.keySchemaVersion == snapshot.keySchemaVersion;
        const bool sameDigests = entry.valid && key_digests_equal(entry.digests, snapshot.keyDigests);
        if (sameFrameToken && sameSchema && sameDigests) {
            if (entry.snapshotId == 0) {
                entry.snapshotId = snapshot.snapshotId;
            }
            else if (entry.snapshotId != snapshot.snapshotId) {
                frameSnapshotMismatch = true;
                expectedSnapshotIdForFrame = entry.snapshotId;
            }
        }
        else {
            entry.valid = true;
            entry.frameToken = snapshot.frameToken.value;
            entry.digests = snapshot.keyDigests;
            entry.keySchemaVersion = snapshot.keySchemaVersion;
            entry.snapshotId = snapshot.snapshotId;
        }
    }
    if (frameSnapshotMismatch) {
        telemetry_record_frame_snapshot_mismatch();
        telemetry_trace_frame_snapshot_mismatch(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            snapshot.frameToken.value,
            expectedSnapshotIdForFrame,
            snapshot.snapshotId);
    }

    const ShadowHistoryKey key{ snapshot.instanceToken.value, snapshot.deviceContextKey };
    ShadowHistoryEntry previous{};
    bool hasPrevious = false;
    {
        ShadowHistoryState& state = shadow_history_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        auto it = state.bySubmissionKey.find(key);
        if (it != state.bySubmissionKey.end() && it->second.valid) {
            previous = it->second;
            hasPrevious = true;
        }
    }

    ShadowKeyDelta delta{};
    delta.hasPrevious = hasPrevious;
    delta.keySchemaChanged = !hasPrevious || (previous.keySchemaVersion != snapshot.keySchemaVersion);
    delta.uploadCoreChanged = !hasPrevious || (previous.digests.uploadCoreHash != snapshot.keyDigests.uploadCoreHash);
    delta.dirChanged = !hasPrevious || (previous.digests.dirHash != snapshot.keyDigests.dirHash);
    delta.scannerChanged = !hasPrevious || (previous.digests.scannerHash != snapshot.keyDigests.scannerHash);
    delta.autoExposureChanged = !hasPrevious || (previous.digests.autoExposureHash != snapshot.keyDigests.autoExposureHash);

    const ResourcePlan plan = build_shadow_resource_plan(delta);

    if (!validate_resource_kind_onboarding_contract(outError)) {
        const ResourcePlan errorPlan = make_uniform_resource_plan(AcquireStatus::Error);
        telemetry_record_module_boundary_violation();
        telemetry_trace_module_boundary_violation(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            "resource_kind_onboarding_contract_invalid");
        telemetry_record_acquire_status(AcquireStatus::Error);
        for (ResourceKind kind : kResourceKindOrder) {
            telemetry_record_acquire_status_for_kind(kind, AcquireStatus::Error);
        }
        telemetry_trace_acquire(
            acquireId,
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            AcquireStatus::Error,
            errorPlan,
            hasPrevious);
        return false;
    }

    for (ResourceKind kind : kResourceKindOrder) {
        const ResourcePlanEntry& entry = resource_plan_entry(plan, kind);
        const ResourceKindContractEntry& contract = resource_kind_contract_entry(kind);
        const bool laneChanged = shadow_key_changed_for_kind(delta, kind);
        const std::uint64_t previousHash = hasPrevious ? key_digest_for_kind(previous.digests, kind) : 0;
        const std::uint64_t currentHash = key_digest_for_kind(snapshot.keyDigests, kind);

        if (entry.invalidated) {
            telemetry_trace_invalidation(
                transaction.transactionId,
                snapshot.snapshotId,
                snapshot.traceSchemaVersion,
                contract.invalidationLane,
                delta.keySchemaChanged ? "key_schema_changed" : (laneChanged ? "lane_hash_changed" : "policy_invalidated"),
                previousHash,
                currentHash);
        }

        if (delta.keySchemaChanged || laneChanged) {
            telemetry_trace_dag_edge(
                transaction.transactionId,
                snapshot.snapshotId,
                snapshot.traceSchemaVersion,
                contract.invalidationLane,
                contract.resourceNode,
                true,
                "allowed_lane_invalidation");
        }

        telemetry_record_acquire_status_for_kind(kind, entry.acquire.status);
    }

    bool forbiddenEdgeDetected = false;
    const ResourceKindContractEntry& scannerContract = resource_kind_contract_entry(ResourceKind::Scanner);
    const ResourceKindContractEntry& dirContract = resource_kind_contract_entry(ResourceKind::Dir);
    const ResourceKindContractEntry& uploadContract = resource_kind_contract_entry(ResourceKind::UploadCore);
    if (delta.scannerChanged &&
        !delta.uploadCoreChanged &&
        !delta.keySchemaChanged &&
        resource_plan_entry(plan, ResourceKind::UploadCore).invalidated) {
        forbiddenEdgeDetected = true;
        telemetry_record_forbidden_invalidation_edge();
        telemetry_trace_dag_edge(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            scannerContract.invalidationLane,
            uploadContract.resourceNode,
            false,
            "forbidden_edge_scanner_to_upload");
    }
    if (delta.scannerChanged &&
        !delta.dirChanged &&
        !delta.keySchemaChanged &&
        resource_plan_entry(plan, ResourceKind::Dir).invalidated) {
        forbiddenEdgeDetected = true;
        telemetry_record_forbidden_invalidation_edge();
        telemetry_trace_dag_edge(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            scannerContract.invalidationLane,
            dirContract.resourceNode,
            false,
            "forbidden_edge_scanner_to_dir");
    }
    if (delta.dirChanged &&
        !delta.uploadCoreChanged &&
        !delta.keySchemaChanged &&
        resource_plan_entry(plan, ResourceKind::UploadCore).invalidated) {
        forbiddenEdgeDetected = true;
        telemetry_record_forbidden_invalidation_edge();
        telemetry_trace_dag_edge(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            dirContract.invalidationLane,
            uploadContract.resourceNode,
            false,
            "forbidden_edge_dir_to_upload");
    }

    if (forbiddenEdgeDetected) {
        telemetry_record_module_boundary_violation();
        telemetry_trace_module_boundary_violation(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            "forbidden_invalidation_edge_detected");
    }

    {
        ShadowHistoryState& state = shadow_history_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        ShadowHistoryEntry& entry = state.bySubmissionKey[key];
        entry.valid = true;
        entry.digests = snapshot.keyDigests;
        entry.keySchemaVersion = snapshot.keySchemaVersion;
        entry.snapshotId = snapshot.snapshotId;
    }

    const AcquireStatus finalStatus = combine_status(plan);

    telemetry_record_acquire_status(finalStatus);
    telemetry_trace_acquire(
        acquireId,
        transaction.transactionId,
        snapshot.snapshotId,
        snapshot.traceSchemaVersion,
        finalStatus,
        plan,
        hasPrevious);
    telemetry_record_acquire_plan();
    return true;
}

bool commit_submission(
    SubmissionTransaction& transaction,
    void* cudaStreamOpaque,
    std::string& outError) {
    (void)cudaStreamOpaque;
    outError.clear();
    MetadataMutationGuard mutationGuard("commit_submission", &transaction.snapshot.deviceContextKey);
    if (!mutationGuard.ok()) {
        outError = "metadata mutation guard rejected commit_submission";
        return false;
    }
    if (!validate_lifecycle_for_stage(transaction, "commit", false, &outError)) {
        return false;
    }
    if (!validate_stale_tuple_for_stage(
            transaction,
            "commit",
            LeaseObservationMode::ActiveOnly,
            "stale transaction in commit path (reason=",
            &outError,
            false,
            nullptr)) {
        return false;
    }
    if (!transaction.active) {
        outError = "submission transaction is not active";
        return false;
    }
    (void)registry_note_submission_end(transaction.snapshot.deviceContextKey);
    transaction.committed = true;
    transaction.active = false;
    telemetry_record_commit_submission();
    return true;
}


void rollback_submission(
    SubmissionTransaction& transaction,
    const char* reason) noexcept {
    (void)reason;
    MetadataMutationGuard mutationGuard("rollback_submission", &transaction.snapshot.deviceContextKey);
    if (!mutationGuard.ok()) {
        return;
    }
    (void)validate_lifecycle_for_stage(transaction, "release", true, nullptr);
    (void)validate_stale_tuple_for_stage(
        transaction,
        "release",
        LeaseObservationMode::Always,
        nullptr,
        nullptr,
        false,
        nullptr);
    if (!transaction.active) {
        return;
    }
    (void)registry_note_submission_end(transaction.snapshot.deviceContextKey);
    transaction.committed = false;
    transaction.active = false;
    telemetry_record_rollback_submission();
}

} // namespace ResourceManager
} // namespace JuicerCuda
