// Cuda/ResourceManager/JuicerCudaResourceManager.cpp

#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"

#include "Cuda/JuicerCudaLaunchGraphCounters.h"
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

#include "Cuda/ResourceManager/JuicerCudaResourceManagerInternal.h"

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

const char* trace_or_unknown(const char* value) noexcept {
    return value ? value : "unknown";
}

const char* trace_or_unspecified(const char* value) noexcept {
    return value ? value : "unspecified";
}

const char* trace_or(const char* value, const char* fallback) noexcept {
    return value ? value : fallback;
}

const char* trace_or_non_empty(const char* value, const char* fallback) noexcept {
    return (value && value[0] != '\0') ? value : fallback;
}

const char* failure_reason_class(const char* token) noexcept {
    const std::string_view value = trace_or_non_empty(token, "");
    if (value.empty()) {
        return nullptr;
    }

    if (value == "auto_no_optional_supported" ||
        value == "capability_fallback_legacy" ||
        value == "requested_async_unsupported" ||
        value == "requested_slab_unsupported") {
        return "capability_unavailable";
    }

    if (value == "canonical_normalization" ||
        value == "invalid_expected_hash" ||
        value == "invalid_scan_lut_key" ||
        value == "missing_instance_token" ||
        value == "mixed_snapshot_id_for_frame" ||
        value == "trace_schema_mismatch") {
        return "validation_or_safety";
    }

    if (value == "context_query_failed" ||
        value == "device_query_failed" ||
        value == "event_create_failed" ||
        value == "lifecycle_stage_rejected" ||
        value == "lifecycle_state_not_allowed" ||
        value == "missing_registry_entry" ||
        value == "private_fallback_failed" ||
        value == "staged_copy_failed" ||
        value == "sync_fallback_required" ||
        value == "unknown" ||
        value == "active_async_pool" ||
        value == "auto_select_async" ||
        value == "requested_async_supported" ||
        value == "unspecified") {
        return "orchestration_failure";
    }

    if (value == "already_active" ||
        value == "cap_disabled" ||
        value == "disabled" ||
        value == "empty_request" ||
        value == "legacy_active" ||
        value == "legacy_default" ||
        value == "not_candidate" ||
        value == "not_superseded" ||
        value == "per_instance_cap_reached" ||
        value == "per_medium_cap_zero" ||
        value == "private_fallback_denied" ||
        value == "requested_legacy" ||
        value == "requested_slab_supported" ||
        value == "slab_scaffold_fallback_legacy" ||
        value == "auto_select_slab" ||
        value == "unknown_preference_fallback" ||
        value == "zero_request") {
        return "policy_denied";
    }

    if (value == "accrue_burst_consumed" ||
        value == "admit_new" ||
        value == "below_debt_threshold" ||
        value == "below_threshold" ||
        value == "builder_reservation_reject" ||
        value == "cancel_superseded_noncritical" ||
        value == "cap_exceeded" ||
        value == "cooldown_blocked" ||
        value == "cooldown_expired" ||
        value == "critical_bypass" ||
        value == "critical_last_resort" ||
        value == "critical_no_burst_consumption" ||
        value == "critical_preserve" ||
        value == "fairness_tokens_exhausted" ||
        value == "ghost_hit_bypass" ||
        value == "granted" ||
        value == "host_alloc_failed" ||
        value == "no_history" ||
        value == "normal" ||
        value == "pressure_gate_reject" ||
        value == "pressure_pre_upload_reclaim_failed" ||
        value == "private_fallback_admit" ||
        value == "private_fallback_served" ||
        value == "probation_admit" ||
        value == "probation_critical_override" ||
        value == "probation_defer" ||
        value == "slot_reuse" ||
        value == "throttle_max_debt" ||
        value == "tier_circuit_blocked" ||
        value == "too_large_critical_override" ||
        value == "too_large_noncritical" ||
        value == "upload_reservation_reject" ||
        value == "wait_budget_reached") {
        return "resource_contention";
    }

    return nullptr;
}

const char* trace_reason_class_or_invalid(const char* token) noexcept {
    const char* normalizedClass = failure_reason_class(token);
    return normalizedClass ? normalizedClass : "invalid_unmapped";
}

std::string trace_event_prefix(
    const char* eventName,
    const SubmissionTransaction& transaction,
    const char* commandName) {
    return std::string("event=") + trace_or_unknown(eventName)
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + trace_or_unknown(commandName);
}

std::string trace_event_identity_prefix(
    const char* eventName,
    const SubmissionTransaction& transaction) {
    return std::string("event=") + trace_or_unknown(eventName)
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion);
}

std::string trace_device_context_fields(const SubmissionTransaction& transaction) {
    const auto& contextKey = transaction.snapshot.deviceContextKey;
    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(contextKey.contextOpaque);
    return std::string(" device_id=") + std::to_string(contextKey.deviceId)
        + " context=" + std::to_string(contextBits);
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


struct OptionalHeuristicTraceContextState {
    bool keepHotTraced = false;
    bool burstDebtTraced = false;
    bool supersededBuilderCancelTraced = false;
};

struct OptionalHeuristicTraceState {
    std::mutex mutex;
    std::unordered_map<DeviceContextKey, OptionalHeuristicTraceContextState, DeviceContextKeyHash> byContext;
};


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
constexpr const char* kPressureShedNonCriticalPrefix = "pressure_shed_noncritical:";
constexpr const char* kPressureCopyComputeGuardPrefix = "pressure_copy_compute_guard:";
constexpr const char* kBurstDebtThrottleNonCriticalPrefix = "burst_debt_throttle_noncritical:";
constexpr const char* kTierCircuitOpenPrefix = "tier_circuit_open:";
constexpr const char* kTierCircuitHalfOpenBusyPrefix = "tier_circuit_half_open_busy:";
constexpr const char* kPrivateLutFallbackFailedPrefix = "private_lut_fallback_failed:";
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


inline bool should_collect_manager_memory_snapshots(const ResourceManagerConfigEffective& cfg) noexcept {
    return cfg.managerSoftTargetBytes > 0 || JTRACE_ENABLED(3);
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
} // namespace

AdmissionChurnPolicyState& admission_churn_policy_state() noexcept;
OptionalHeuristicTraceState& optional_heuristic_trace_state() noexcept;

void trace_scratch_policy_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const char* result,
    std::size_t requestBytes,
    const ScratchPolicySnapshot& snapshot,
    int waitMs) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uint64_t churnDenom = snapshot.bucketAllocGrowth + snapshot.bucketReuse;
    const std::uint64_t churnRatioMilli = (churnDenom == 0)
        ? 0
        : (snapshot.bucketAllocGrowth * 1000ull) / churnDenom;
    const std::size_t effectiveTempCap = effective_temp_scratch_bytes_cap(requestBytes);

    const std::string msg = trace_event_prefix("scratch_policy", transaction, commandName)
        + " result=" + trace_or_unknown(result)
        + " work_class=" + to_cstr(snapshot.bucketKey.workClass)
        + trace_device_context_fields(transaction)
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " in_flight_bytes=" + std::to_string(static_cast<unsigned long long>(snapshot.inFlightBytes))
        + " in_flight_sets=" + std::to_string(snapshot.inFlightSets)
        + " bucket_w=" + std::to_string(snapshot.bucketKey.widthBucket)
        + " bucket_h=" + std::to_string(snapshot.bucketKey.heightBucket)
        + " bucket_step_px=" + std::to_string(snapshot.bucketKey.bucketStepPx)
        + " large_frame=" + std::to_string(snapshot.bucketKey.largeFrame ? 1 : 0)
        + " bucket_attempts=" + std::to_string(snapshot.bucketAttempts)
        + " bucket_exhausted=" + std::to_string(snapshot.bucketExhausted)
        + " bucket_alloc_growth=" + std::to_string(snapshot.bucketAllocGrowth)
        + " bucket_reuse=" + std::to_string(snapshot.bucketReuse)
        + " bucket_starvation_latched=" + std::to_string(snapshot.starvationLatched ? 1 : 0)
        + " scratch_churn_ratio_milli=" + std::to_string(churnRatioMilli)
        + " quarantine_bytes=" + std::to_string(static_cast<unsigned long long>(snapshot.quarantineBytes))
        + " quarantine_entries=" + std::to_string(snapshot.quarantineEntries)
        + " wait_ms=" + std::to_string(waitMs)
        + " max_temp_sets=" + std::to_string(kMaxTempScratchSets)
        + " max_temp_bytes=" + std::to_string(static_cast<unsigned long long>(kMaxTempScratchBytes))
        + " effective_max_temp_bytes=" + std::to_string(static_cast<unsigned long long>(effectiveTempCap))
        + " quarantine_max_bytes=" + std::to_string(static_cast<unsigned long long>(kLargeFrameQuarantineMaxBytes))
        + " quarantine_max_entries=" + std::to_string(static_cast<unsigned long long>(kLargeFrameQuarantineMaxEntries));
    JTRACE("MSACQ", msg);
}

void trace_transient_reservation_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::size_t requestBytes,
    std::uint64_t bytesInFlight,
    std::uint64_t capBytes,
    std::uint64_t thresholdBytes,
    const ReservationDecision& decision,
    bool criticalCurrentFrame,
    int waitMs,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("transient_reservation", transaction, commandName)
        + " kind=" + to_cstr(ReservationKind::TransientNonManager)
        + trace_device_context_fields(transaction)
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " bytes_in_flight=" + std::to_string(static_cast<unsigned long long>(bytesInFlight))
        + " cap_bytes=" + std::to_string(static_cast<unsigned long long>(capBytes))
        + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(thresholdBytes))
        + " granted=" + std::to_string(decision.granted ? 1 : 0)
        + " should_wait=" + std::to_string(decision.shouldWait ? 1 : 0)
        + " wait_ms=" + std::to_string(waitMs)
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + " decision_reason=" + trace_or_unspecified(decision.reason)
        + " decision_reason_class=" + trace_reason_class_or_invalid(decision.reason)
        + " reason=" + trace_or_unspecified(reason)
        + " reason_class=" + trace_reason_class_or_invalid(reason);
    JTRACE("MSTRS", msg);
}

void trace_upload_reservation_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::uint64_t requestBytes,
    std::uint64_t bytesInFlight,
    std::uint64_t capBytes,
    std::uint64_t thresholdBytes,
    const ReservationDecision& decision,
    bool criticalCurrentFrame,
    std::uint64_t instanceToken,
    std::uint32_t sharedTokens,
    std::uint32_t criticalTokens,
    int waitMs,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("upload_reservation", transaction, commandName)
        + " kind=" + to_cstr(ReservationKind::UploadCopy)
        + " instance_token=" + std::to_string(static_cast<unsigned long long>(instanceToken))
        + trace_device_context_fields(transaction)
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " bytes_in_flight=" + std::to_string(static_cast<unsigned long long>(bytesInFlight))
        + " cap_bytes=" + std::to_string(static_cast<unsigned long long>(capBytes))
        + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(thresholdBytes))
        + " shared_tokens=" + std::to_string(static_cast<unsigned long long>(sharedTokens))
        + " critical_tokens=" + std::to_string(static_cast<unsigned long long>(criticalTokens))
        + " granted=" + std::to_string(decision.granted ? 1 : 0)
        + " should_wait=" + std::to_string(decision.shouldWait ? 1 : 0)
        + " wait_ms=" + std::to_string(waitMs)
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + " decision_reason=" + trace_or_unspecified(decision.reason)
        + " decision_reason_class=" + trace_reason_class_or_invalid(decision.reason)
        + " reason=" + trace_or_unspecified(reason)
        + " reason_class=" + trace_reason_class_or_invalid(reason);
    JTRACE("MSUPL", msg);
}

void trace_builder_reservation_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    BuilderReservationTier tier,
    std::uint64_t requestBytes,
    std::uint64_t bytesInFlight,
    std::uint64_t capBytes,
    std::uint64_t thresholdBytes,
    const ReservationDecision& decision,
    bool criticalCurrentFrame,
    std::uint64_t instanceToken,
    std::uint32_t sharedTokens,
    std::uint32_t criticalTokens,
    int waitMs,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("builder_reservation", transaction, commandName)
        + " kind=" + to_cstr(ReservationKind::BuilderWork)
        + " tier=" + to_cstr(tier)
        + " instance_token=" + std::to_string(static_cast<unsigned long long>(instanceToken))
        + trace_device_context_fields(transaction)
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " bytes_in_flight=" + std::to_string(static_cast<unsigned long long>(bytesInFlight))
        + " cap_bytes=" + std::to_string(static_cast<unsigned long long>(capBytes))
        + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(thresholdBytes))
        + " shared_tokens=" + std::to_string(static_cast<unsigned long long>(sharedTokens))
        + " critical_tokens=" + std::to_string(static_cast<unsigned long long>(criticalTokens))
        + " granted=" + std::to_string(decision.granted ? 1 : 0)
        + " should_wait=" + std::to_string(decision.shouldWait ? 1 : 0)
        + " wait_ms=" + std::to_string(waitMs)
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + " decision_reason=" + trace_or_unspecified(decision.reason)
        + " decision_reason_class=" + trace_reason_class_or_invalid(decision.reason)
        + " reason=" + trace_or_unspecified(reason)
        + " reason_class=" + trace_reason_class_or_invalid(reason);
    JTRACE("MSBPR", msg);
}

void trace_tier_circuit_event(
    const SubmissionTransaction& transaction,
    const char* commandName,
    ResourceTier tier,
    TierCircuitState previousState,
    TierCircuitState state,
    bool transition,
    bool allowed,
    bool probe,
    std::uint32_t windowErrors,
    std::uint32_t threshold,
    std::uint32_t windowMs,
    std::uint32_t openMs,
    std::uint64_t openRemainingMs,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("tier_circuit", transaction, commandName)
        + " tier=" + to_cstr(tier)
        + " prev_state=" + to_cstr(previousState)
        + " state=" + to_cstr(state)
        + " transition=" + std::to_string(transition ? 1 : 0)
        + " allowed=" + std::to_string(allowed ? 1 : 0)
        + " probe=" + std::to_string(probe ? 1 : 0)
        + " window_errors=" + std::to_string(windowErrors)
        + " threshold=" + std::to_string(threshold)
        + " window_ms=" + std::to_string(windowMs)
        + " open_ms=" + std::to_string(openMs)
        + " open_remaining_ms=" + std::to_string(static_cast<unsigned long long>(openRemainingMs))
        + trace_device_context_fields(transaction)
        + " reason=" + trace_or_unspecified(reason);
    JTRACE("MSCB", msg);
}

void trace_tier_budget_event(
    const SubmissionTransaction& transaction,
    const char* commandName,
    PressureLane lane,
    PressureState pressureState,
    const TierBudgetSnapshot& tierBudget,
    std::size_t requestBytes,
    bool criticalCurrentFrame,
    bool requestReclaimPass,
    std::uint64_t scratchTrimmedEntries,
    std::uint64_t graphEvictedEntries,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("tier_budget", transaction, commandName)
        + " lane=" + to_cstr(lane)
        + " state=" + to_cstr(pressureState)
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " request_reclaim_pass=" + std::to_string(requestReclaimPass ? 1 : 0)
        + " any_over_target=" + std::to_string(tierBudget.anyOverTarget ? 1 : 0)
        + " dominant_tier=" + to_cstr(tierBudget.dominantOverTargetTier)
        + " dominant_over_target_bytes=" + std::to_string(
            static_cast<unsigned long long>(tierBudget.dominantOverTargetBytes))
        + " immutable_active_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.activeBytes, ResourceTier::Immutable)))
        + " immutable_target_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.targetBytes, ResourceTier::Immutable)))
        + " immutable_over_target_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.overTargetBytes, ResourceTier::Immutable)))
        + " immutable_reclaimable_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.reclaimableBytes, ResourceTier::Immutable)))
        + " lut_active_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.activeBytes, ResourceTier::Lut)))
        + " lut_target_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.targetBytes, ResourceTier::Lut)))
        + " lut_over_target_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.overTargetBytes, ResourceTier::Lut)))
        + " lut_reclaimable_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.reclaimableBytes, ResourceTier::Lut)))
        + " scratch_active_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.activeBytes, ResourceTier::Scratch)))
        + " scratch_target_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.targetBytes, ResourceTier::Scratch)))
        + " scratch_over_target_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.overTargetBytes, ResourceTier::Scratch)))
        + " scratch_reclaimable_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.reclaimableBytes, ResourceTier::Scratch)))
        + " graph_active_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.activeBytes, ResourceTier::Graph)))
        + " graph_target_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.targetBytes, ResourceTier::Graph)))
        + " graph_over_target_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.overTargetBytes, ResourceTier::Graph)))
        + " graph_reclaimable_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.reclaimableBytes, ResourceTier::Graph)))
        + " total_active_bytes=" + std::to_string(static_cast<unsigned long long>(tierBudget.totalActiveBytes))
        + " total_reclaimable_bytes=" + std::to_string(
            static_cast<unsigned long long>(tierBudget.totalReclaimableBytes))
        + " scratch_trimmed_entries=" + std::to_string(static_cast<unsigned long long>(scratchTrimmedEntries))
        + " graph_evicted_entries=" + std::to_string(static_cast<unsigned long long>(graphEvictedEntries))
        + trace_device_context_fields(transaction)
        + " reason=" + trace_or_unspecified(reason);
    JTRACE("MSTGT", msg);
}

void trace_effective_reserve_event(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::uint64_t reserveBaseBytes,
    std::uint64_t reserveBeforeBytes,
    std::uint64_t reserveTargetBytes,
    std::uint64_t reserveAfterBytes,
    std::uint64_t transientNonManagerBytes,
    bool updated,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    const std::string msg = trace_event_prefix("effective_reserve", transaction, commandName)
        + " reserve_base_bytes=" + std::to_string(static_cast<unsigned long long>(reserveBaseBytes))
        + " reserve_before_bytes=" + std::to_string(static_cast<unsigned long long>(reserveBeforeBytes))
        + " reserve_target_bytes=" + std::to_string(static_cast<unsigned long long>(reserveTargetBytes))
        + " reserve_after_bytes=" + std::to_string(static_cast<unsigned long long>(reserveAfterBytes))
        + " transient_non_manager_bytes=" + std::to_string(
            static_cast<unsigned long long>(transientNonManagerBytes))
        + " reserve_safety_margin_bytes=" + std::to_string(
            static_cast<unsigned long long>(cfg.reserveSafetyMarginBytes))
        + " reserve_step_up_bytes=" + std::to_string(
            static_cast<unsigned long long>(cfg.reserveAdaptUpStepBytes))
        + " reserve_step_down_bytes=" + std::to_string(
            static_cast<unsigned long long>(cfg.reserveAdaptDownStepBytes))
        + " updated=" + std::to_string(updated ? 1 : 0)
        + trace_device_context_fields(transaction)
        + " reason=" + trace_or_unspecified(reason);
    JTRACE("MSRSV", msg);
}

void trace_opportunistic_freeze_event(
    const SubmissionTransaction& transaction,
    const char* commandName,
    PressureState pressureState,
    std::uint64_t effectiveHeadroomBytes,
    std::uint64_t effectiveReserveBytes,
    bool frozen,
    bool allowed,
    bool criticalCurrentFrame,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("opportunistic_freeze", transaction, commandName)
        + " state=" + to_cstr(pressureState)
        + " effective_headroom_bytes=" + std::to_string(
            static_cast<unsigned long long>(effectiveHeadroomBytes))
        + " effective_reserve_bytes=" + std::to_string(
            static_cast<unsigned long long>(effectiveReserveBytes))
        + " frozen=" + std::to_string(frozen ? 1 : 0)
        + " allowed=" + std::to_string(allowed ? 1 : 0)
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + trace_device_context_fields(transaction)
        + " reason=" + trace_or_unspecified(reason);
    JTRACE("MSFRZ", msg);
}

void trace_active_burst_event(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const ActiveBurstDecision& burst,
    bool criticalCurrentFrame,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("active_burst", transaction, commandName)
        + " considered=" + std::to_string(burst.considered ? 1 : 0)
        + " active=" + std::to_string(burst.active ? 1 : 0)
        + " allowed=" + std::to_string(burst.allowed ? 1 : 0)
        + " entered=" + std::to_string(burst.entered ? 1 : 0)
        + " exited=" + std::to_string(burst.exited ? 1 : 0)
        + " cap_hit=" + std::to_string(burst.capHit ? 1 : 0)
        + " over_target_bytes=" + std::to_string(static_cast<unsigned long long>(burst.overTargetBytes))
        + " cap_bytes=" + std::to_string(static_cast<unsigned long long>(burst.capBytes))
        + " elapsed_ms=" + std::to_string(static_cast<unsigned long long>(burst.elapsedMs))
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + trace_device_context_fields(transaction)
        + " reason=" + trace_or_unspecified(reason);
    JTRACE("MSBURST", msg);
}

void trace_budget_reclaim_retry(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::uint32_t attempt,
    std::size_t reclaimedBytes,
    bool success,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("reclaim_retry", transaction, commandName)
        + " attempt=" + std::to_string(static_cast<unsigned long long>(attempt))
        + " reclaimed_bytes=" + std::to_string(static_cast<unsigned long long>(reclaimedBytes))
        + trace_device_context_fields(transaction)
        + " success=" + std::to_string(success ? 1 : 0)
        + " reason=" + trace_or_unspecified(reason);
    JTRACE("MSEVICT", msg);
}

void trace_pressure_checkpoint(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const PressureCheckpoint& checkpoint,
    std::size_t pendingGrowthBytes,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("pressure_checkpoint", transaction, commandName)
        + " state=" + to_cstr(checkpoint.decision.state)
        + " prev_state=" + to_cstr(checkpoint.previousState)
        + " desired_state=" + to_cstr(checkpoint.desiredState)
        + " transition=" + std::to_string(checkpoint.transition ? 1 : 0)
        + " transition_deferred_dwell=" + std::to_string(checkpoint.transitionDeferredByDwell ? 1 : 0)
        + " transition_deferred_rate=" + std::to_string(checkpoint.transitionDeferredByRate ? 1 : 0)
        + " reserve_crossing=" + std::to_string(checkpoint.reserveCrossing ? 1 : 0)
        + " reserve_crossed=" + std::to_string(checkpoint.reserveCrossedNow ? 1 : 0)
        + " sampled=" + std::to_string(checkpoint.sampled ? 1 : 0)
        + " poll_interval_ms=" + std::to_string(checkpoint.pollIntervalMs)
        + trace_device_context_fields(transaction)
        + " soft_target_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.input.softTargetBytes))
        + " reserve_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.input.reserveBytes))
        + " manager_resident_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.input.managerResidentBytes))
        + " retire_pending_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.input.retirePendingBytes))
        + " transient_non_manager_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.input.transientNonManagerBytes))
        + " pressure_total_bytes=" + std::to_string(static_cast<unsigned long long>(pressure_total_bytes(checkpoint.input)))
        + " active_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.memory.activeBytes))
        + " reclaimable_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.memory.reclaimableBytes))
        + " pending_growth_bytes=" + std::to_string(static_cast<unsigned long long>(pendingGrowthBytes))
        + " allow_opportunistic=" + std::to_string(checkpoint.decision.allowOpportunistic ? 1 : 0)
        + " freeze_opportunistic=" + std::to_string(checkpoint.decision.freezeOpportunistic ? 1 : 0)
        + " request_reclaim_pass=" + std::to_string(checkpoint.decision.requestReclaimPass ? 1 : 0)
        + " should_shed_non_critical=" + std::to_string(checkpoint.decision.shouldShedNonCritical ? 1 : 0)
        + " reserve_before_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.reserveBeforeBytes))
        + " reserve_target_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.reserveTargetBytes))
        + " reserve_updated=" + std::to_string(checkpoint.reserveUpdated ? 1 : 0)
        + " effective_reserve_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.decision.effectiveReserveBytes))
        + " effective_headroom_bytes=" + std::to_string(
            static_cast<unsigned long long>(checkpoint.decision.effectiveHeadroomBytes))
        + " headroom_source=" + to_cstr(checkpoint.decision.headroomSource)
        + " driver_free_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.input.driverFreeBytes))
        + " allocator_pool_reserved_bytes=" + std::to_string(
            static_cast<unsigned long long>(checkpoint.input.allocatorPoolReservedBytes))
        + " allocator_pool_used_bytes=" + std::to_string(
            static_cast<unsigned long long>(checkpoint.input.allocatorPoolUsedBytes))
        + " reason=" + trace_or_unspecified(reason);
    JTRACE("MSPRS", msg);
}

void trace_headroom_sample(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const PressureCheckpoint& checkpoint,
    std::size_t requestBytes,
    bool sourceSwitch,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("headroom", transaction, commandName)
        + trace_device_context_fields(transaction)
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " effective_headroom_bytes=" + std::to_string(
            static_cast<unsigned long long>(checkpoint.input.effectiveHeadroomBytes))
        + " headroom_source=" + to_cstr(checkpoint.input.headroomSource)
        + " driver_free_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.input.driverFreeBytes))
        + " allocator_pool_reserved_bytes=" + std::to_string(
            static_cast<unsigned long long>(checkpoint.input.allocatorPoolReservedBytes))
        + " allocator_pool_used_bytes=" + std::to_string(
            static_cast<unsigned long long>(checkpoint.input.allocatorPoolUsedBytes))
        + " source_switch=" + std::to_string(sourceSwitch ? 1 : 0)
        + " reason=" + trace_or_unspecified(reason);
    JTRACE("MSHDR", msg);
}

void trace_transient_non_manager_sample(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const PressureCheckpoint& checkpoint,
    std::size_t pendingGrowthBytes,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("transient_non_manager_sample", transaction, commandName)
        + trace_device_context_fields(transaction)
        + " transient_non_manager_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.input.transientNonManagerBytes))
        + " pending_growth_bytes=" + std::to_string(static_cast<unsigned long long>(pendingGrowthBytes))
        + " pressure_total_bytes=" + std::to_string(static_cast<unsigned long long>(pressure_total_bytes(checkpoint.input)))
        + " reason=" + trace_or_unspecified(reason);
    JTRACE("MSTRN", msg);
}

void trace_emergency_shed_action(
    const SubmissionTransaction& transaction,
    const char* commandName,
    PressureLane lane,
    PressureState state,
    std::size_t requestBytes,
    bool criticalCurrentFrame,
    bool allowed,
    std::uint64_t uploadBytesInFlight,
    std::uint64_t uploadCapBytes,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("emergency_shed", transaction, commandName)
        + " lane=" + to_cstr(lane)
        + " state=" + to_cstr(state)
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + " allowed=" + std::to_string(allowed ? 1 : 0)
        + " upload_bytes_in_flight=" + std::to_string(static_cast<unsigned long long>(uploadBytesInFlight))
        + " upload_cap_bytes=" + std::to_string(static_cast<unsigned long long>(uploadCapBytes))
        + trace_device_context_fields(transaction)
        + " reason=" + trace_or_unspecified(reason);
    JTRACE("MSEMS", msg);
}

void trace_lane_wait_event(
    const SubmissionTransaction& transaction,
    const char* commandName,
    PressureLane lane,
    bool criticalCurrentFrame,
    int waitMs,
    const char* outcome,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("lane_wait", transaction, commandName)
        + " lane=" + to_cstr(lane)
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + " wait_ms=" + std::to_string(waitMs)
        + " outcome=" + trace_or_unknown(outcome)
        + trace_device_context_fields(transaction)
        + " reason=" + trace_or_unspecified(reason);
    JTRACE("MSFAIR", msg);
}

void trace_copy_compute_guard(
    const SubmissionTransaction& transaction,
    const char* commandName,
    PressureState state,
    std::uint64_t uploadBytesInFlight,
    std::uint64_t uploadCapBytes,
    bool criticalCurrentFrame,
    bool allowed,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("copy_compute_guard", transaction, commandName)
        + " state=" + to_cstr(state)
        + " upload_bytes_in_flight=" + std::to_string(static_cast<unsigned long long>(uploadBytesInFlight))
        + " upload_cap_bytes=" + std::to_string(static_cast<unsigned long long>(uploadCapBytes))
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + " allowed=" + std::to_string(allowed ? 1 : 0)
        + trace_device_context_fields(transaction)
        + " reason=" + trace_or_unspecified(reason);
    JTRACE("MSCOPY", msg);
}

AdmissionChurnSnapshot sample_admission_churn_state(
    const SubmissionTransaction& transaction,
    const ResourceManagerConfigEffective& cfg,
    std::uint64_t entryDigest,
    std::uint32_t observedProbationHits) {
    AdmissionChurnSnapshot out{};
    out.enabled = (cfg.admissionChurnWindowMs > 0);
    out.windowMs = cfg.admissionChurnWindowMs;
    out.enterOneHitRatePct = cfg.admissionChurnEnterOneHitRatePct;
    out.exitOneHitRatePct = cfg.admissionChurnExitOneHitRatePct;
    out.probationHitBonus = cfg.admissionChurnProbationHitBonus;
    out.keepHotMs = cfg.keepHotMs;
    out.readmitCooldownMs = cfg.largeEntryReadmitCooldownMs;
    out.ghostHitsForReadmit = cfg.largeEntryGhostHitsForReadmit;
    if (!out.enabled) {
        return out;
    }

    const std::uint64_t nowMs = monotonic_time_ms();
    AdmissionChurnPolicyState& state = admission_churn_policy_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    AdmissionChurnContextState& contextState = state.byContext[transaction.snapshot.deviceContextKey];

    const bool windowReset =
        (contextState.windowStartMs == 0) ||
        ((nowMs > contextState.windowStartMs) &&
         ((nowMs - contextState.windowStartMs) >= static_cast<std::uint64_t>(cfg.admissionChurnWindowMs)));
    if (windowReset) {
        if (contextState.active) {
            contextState.active = false;
            telemetry_counter_add(global_state().admissionChurnExitEvents, 1);
        }
        contextState.windowStartMs = nowMs;
        contextState.windowSamples = 0;
        contextState.windowOneHitSamples = 0;
        contextState.windowDigestHits.clear();
    }

    contextState.windowSamples += 1;
    if (observedProbationHits == 0) {
        contextState.windowOneHitSamples += 1;
    }
    std::uint32_t& digestHits = contextState.windowDigestHits[entryDigest];
    if (digestHits < std::numeric_limits<std::uint32_t>::max()) {
        ++digestHits;
    }

    const std::uint64_t oneHitRatePctU64 = (contextState.windowSamples == 0)
        ? 0
        : ((contextState.windowOneHitSamples * 100ull) / contextState.windowSamples);
    const std::uint32_t oneHitRatePct = static_cast<std::uint32_t>(std::min<std::uint64_t>(oneHitRatePctU64, 100ull));

    bool nextActive = contextState.active;
    if (!nextActive) {
        nextActive = oneHitRatePct >= cfg.admissionChurnEnterOneHitRatePct;
    }
    else {
        nextActive = oneHitRatePct > cfg.admissionChurnExitOneHitRatePct;
    }
    if (!contextState.active && nextActive) {
        telemetry_counter_add(global_state().admissionChurnEnterEvents, 1);
    }
    else if (contextState.active && !nextActive) {
        telemetry_counter_add(global_state().admissionChurnExitEvents, 1);
    }
    contextState.active = nextActive;

    out.active = contextState.active;
    out.oneHitRatePct = oneHitRatePct;
    out.uniqueKeys = static_cast<std::uint32_t>(std::min<std::size_t>(
        contextState.windowDigestHits.size(),
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())));
    out.windowSamples = static_cast<std::uint32_t>(std::min<std::uint64_t>(
        contextState.windowSamples,
        static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())));
    telemetry_counter_add(global_state().admissionChurnSampleEvents, 1);
    return out;
}

void trace_keep_hot_surface(
    const SubmissionTransaction& transaction,
    const ResourceManagerConfigEffective& cfg,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }
    const std::string msg = trace_event_identity_prefix("keep_hot_surface", transaction)
        + trace_device_context_fields(transaction)
        + " enabled=" + std::to_string(cfg.keepHotMs > 0 ? 1 : 0)
        + " keep_hot_ms=" + std::to_string(static_cast<unsigned long long>(cfg.keepHotMs))
        + " reason=" + trace_or_unspecified(reason)
        + " reason_class=" + trace_reason_class_or_invalid(reason);
    JTRACE("MSHOT", msg);
    telemetry_counter_add(global_state().keepHotSurfaceTraceEvents, 1);
}

void trace_keep_hot_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::uint64_t keepHotMs,
    std::uint64_t bypassEvents,
    std::uint64_t forcedEvictEvents,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("keep_hot_decision", transaction, commandName)
        + " keep_hot_ms=" + std::to_string(static_cast<unsigned long long>(keepHotMs))
        + " bypass_events=" + std::to_string(static_cast<unsigned long long>(bypassEvents))
        + " forced_evict_events=" + std::to_string(static_cast<unsigned long long>(forcedEvictEvents))
        + trace_device_context_fields(transaction)
        + " reason=" + trace_or_unspecified(reason)
        + " reason_class=" + trace_reason_class_or_invalid(reason);
    JTRACE("MSHOT", msg);
}

void trace_burst_debt_surface(
    const SubmissionTransaction& transaction,
    const ResourceManagerConfigEffective& cfg,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }
    const bool enabled = (cfg.burstDebtHalfLifeMs > 0) && (cfg.maxBurstDebtPct < 100);
    const std::string msg = trace_event_identity_prefix("burst_debt_surface", transaction)
        + trace_device_context_fields(transaction)
        + " enabled=" + std::to_string(enabled ? 1 : 0)
        + " burst_debt_half_life_ms=" + std::to_string(static_cast<unsigned long long>(cfg.burstDebtHalfLifeMs))
        + " max_burst_debt_pct=" + std::to_string(static_cast<unsigned long long>(cfg.maxBurstDebtPct))
        + " reason=" + trace_or_unspecified(reason)
        + " reason_class=" + trace_reason_class_or_invalid(reason);
    JTRACE("MSBDE", msg);
    telemetry_counter_add(global_state().burstDebtSurfaceTraceEvents, 1);
}

void trace_burst_debt_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    PressureLane lane,
    PressureState pressureState,
    bool criticalCurrentFrame,
    std::size_t requestBytes,
    const ResourceManagerConfigEffective& cfg,
    const BurstDebtRuntimeDecision& decision) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("burst_debt_decision", transaction, commandName)
        + " lane=" + to_cstr(lane)
        + " pressure_state=" + to_cstr(pressureState)
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + " enabled=" + std::to_string(decision.enabled ? 1 : 0)
        + " sampled=" + std::to_string(decision.sampled ? 1 : 0)
        + " burst_consumed=" + std::to_string(decision.burstConsumed ? 1 : 0)
        + " throttled=" + std::to_string(decision.throttled ? 1 : 0)
        + " debt_before_pct=" + std::to_string(decision.debtBeforePct)
        + " debt_after_pct=" + std::to_string(decision.debtAfterPct)
        + " debt_increment_pct=" + std::to_string(decision.debtIncrementPct)
        + " burst_debt_half_life_ms=" + std::to_string(
            static_cast<unsigned long long>(cfg.burstDebtHalfLifeMs))
        + " max_burst_debt_pct=" + std::to_string(
            static_cast<unsigned long long>(cfg.maxBurstDebtPct))
        + " instance_token=" + std::to_string(
            static_cast<unsigned long long>(transaction.snapshot.instanceToken.value))
        + trace_device_context_fields(transaction)
        + " reason=" + trace_or_unspecified(decision.reason)
        + " reason_class=" + trace_reason_class_or_invalid(decision.reason);
    JTRACE("MSBDE", msg);
}

void trace_superseded_builder_cancel_surface(
    const SubmissionTransaction& transaction,
    const ResourceManagerConfigEffective& cfg,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }
    const std::string msg = trace_event_identity_prefix("superseded_builder_cancel_surface", transaction)
        + trace_device_context_fields(transaction)
        + " enabled=" + std::to_string(cfg.cancelSupersededBuilders ? 1 : 0)
        + " reason=" + trace_or_unspecified(reason)
        + " reason_class=" + trace_reason_class_or_invalid(reason);
    JTRACE("MSCNL", msg);
    telemetry_counter_add(global_state().supersededBuilderCancelSurfaceTraceEvents, 1);
}

void trace_superseded_builder_cancel_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    bool criticalCurrentFrame,
    std::uint64_t requestBytes,
    std::uint64_t latestSnapshotId,
    const SupersededBuilderCancelInput& input,
    const SupersededBuilderCancelDecision& decision) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }
    const std::uint64_t savedBytes = decision.cancel ? requestBytes : 0;
    const std::string msg = trace_event_prefix("superseded_builder_cancel_decision", transaction, commandName)
        + " enabled=" + std::to_string(input.enabled ? 1 : 0)
        + " superseded=" + std::to_string(input.superseded ? 1 : 0)
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + " cancel=" + std::to_string(decision.cancel ? 1 : 0)
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " saved_bytes=" + std::to_string(static_cast<unsigned long long>(savedBytes))
        + " latest_snapshot_id=" + std::to_string(static_cast<unsigned long long>(latestSnapshotId))
        + " instance_token=" + std::to_string(
            static_cast<unsigned long long>(transaction.snapshot.instanceToken.value))
        + trace_device_context_fields(transaction)
        + " reason=" + trace_or_unspecified(decision.reason)
        + " reason_class=" + trace_reason_class_or_invalid(decision.reason);
    JTRACE("MSCNL", msg);
}

void trace_optional_heuristic_surfaces_once(
    const SubmissionTransaction& transaction,
    const ResourceManagerConfigEffective& cfg,
    const char* reason) {
    bool traceKeepHot = false;
    bool traceBurstDebt = false;
    bool traceCancel = false;
    {
        OptionalHeuristicTraceState& traceState = optional_heuristic_trace_state();
        std::lock_guard<std::mutex> lock(traceState.mutex);
        OptionalHeuristicTraceContextState& entry =
            traceState.byContext[transaction.snapshot.deviceContextKey];
        if (!entry.keepHotTraced) {
            entry.keepHotTraced = true;
            traceKeepHot = true;
        }
        if (!entry.burstDebtTraced) {
            entry.burstDebtTraced = true;
            traceBurstDebt = true;
        }
        if (!entry.supersededBuilderCancelTraced) {
            entry.supersededBuilderCancelTraced = true;
            traceCancel = true;
        }
    }
    if (traceKeepHot) {
        trace_keep_hot_surface(transaction, cfg, reason);
    }
    if (traceBurstDebt) {
        trace_burst_debt_surface(transaction, cfg, reason);
    }
    if (traceCancel) {
        trace_superseded_builder_cancel_surface(transaction, cfg, reason);
    }
}

std::uint32_t effective_probation_hits_required(
    std::uint32_t baseRequired,
    const AdmissionChurnSnapshot& churnSnapshot) noexcept {
    if (!churnSnapshot.enabled || !churnSnapshot.active || churnSnapshot.probationHitBonus == 0) {
        return baseRequired;
    }
    const std::uint64_t expanded =
        static_cast<std::uint64_t>(baseRequired) + static_cast<std::uint64_t>(churnSnapshot.probationHitBonus);
    const std::uint64_t capped = std::min<std::uint64_t>(
        expanded,
        static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()));
    return static_cast<std::uint32_t>(capped);
}

std::uint32_t next_probation_hits(std::uint32_t observed) noexcept {
    return (observed < std::numeric_limits<std::uint32_t>::max())
        ? (observed + 1u)
        : std::numeric_limits<std::uint32_t>::max();
}

void trace_large_entry_readmit_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::uint64_t entryDigest,
    std::uint64_t requestBytes,
    std::uint64_t thresholdBytes,
    const LargeEntryReadmitDecision& decision) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("large_entry_readmit", transaction, commandName)
        + " entry_digest=" + std::to_string(static_cast<unsigned long long>(entryDigest))
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(thresholdBytes))
        + " enabled=" + std::to_string(decision.enabled ? 1 : 0)
        + " candidate=" + std::to_string(decision.candidate ? 1 : 0)
        + " critical_current_frame=" + std::to_string(decision.criticalCurrentFrame ? 1 : 0)
        + " had_history=" + std::to_string(decision.hadHistory ? 1 : 0)
        + " in_cooldown=" + std::to_string(decision.inCooldown ? 1 : 0)
        + " blocked=" + std::to_string(decision.blocked ? 1 : 0)
        + " ghost_bypass=" + std::to_string(decision.ghostBypass ? 1 : 0)
        + " age_ms=" + std::to_string(static_cast<unsigned long long>(decision.ageMs))
        + " cooldown_ms=" + std::to_string(static_cast<unsigned long long>(decision.cooldownMs))
        + " ghost_hits_required=" + std::to_string(decision.ghostHitsRequired)
        + " observed_ghost_hits=" + std::to_string(decision.observedGhostHits)
        + trace_device_context_fields(transaction)
        + " reason=" + trace_or_unspecified(decision.reason)
        + " reason_class=" + trace_reason_class_or_invalid(decision.reason);
    JTRACE("MSTHR", msg);
}

void trace_cache_admission_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const CacheAdmissionDecision& decision,
    const AdmissionChurnSnapshot& churnSnapshot,
    bool criticalCurrentFrame,
    std::uint64_t requestBytes,
    std::uint32_t observedProbationHits,
    std::uint64_t entryDigest,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("cache_admission", transaction, commandName)
        + " class=" + to_cstr(decision.admissionClass)
        + " allow_durable=" + std::to_string(decision.allowDurableAdmission ? 1 : 0)
        + " probation_applied=" + std::to_string(decision.probationApplied ? 1 : 0)
        + " probation_hits_required=" + std::to_string(decision.probationHitsRequired)
        + " observed_probation_hits=" + std::to_string(observedProbationHits)
        + " churn_enabled=" + std::to_string(churnSnapshot.enabled ? 1 : 0)
        + " churn_active=" + std::to_string(churnSnapshot.active ? 1 : 0)
        + " churn_window_ms=" + std::to_string(churnSnapshot.windowMs)
        + " churn_window_samples=" + std::to_string(churnSnapshot.windowSamples)
        + " churn_one_hit_rate_pct=" + std::to_string(churnSnapshot.oneHitRatePct)
        + " churn_unique_keys=" + std::to_string(churnSnapshot.uniqueKeys)
        + " churn_enter_one_hit_rate_pct=" + std::to_string(churnSnapshot.enterOneHitRatePct)
        + " churn_exit_one_hit_rate_pct=" + std::to_string(churnSnapshot.exitOneHitRatePct)
        + " churn_probation_hit_bonus=" + std::to_string(churnSnapshot.probationHitBonus)
        + " keep_hot_ms=" + std::to_string(churnSnapshot.keepHotMs)
        + " readmit_cooldown_ms=" + std::to_string(churnSnapshot.readmitCooldownMs)
        + " ghost_hits_for_readmit=" + std::to_string(churnSnapshot.ghostHitsForReadmit)
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " max_durable_bytes=" + std::to_string(static_cast<unsigned long long>(decision.maxDurableBytes))
        + " entry_digest=" + std::to_string(static_cast<unsigned long long>(entryDigest))
        + trace_device_context_fields(transaction)
        + " decision_reason=" + trace_or_unspecified(decision.reason)
        + " decision_reason_class=" + trace_reason_class_or_invalid(decision.reason)
        + " reason=" + trace_or_unspecified(reason)
        + " reason_class=" + trace_reason_class_or_invalid(reason);
    JTRACE("MSADM", msg);
}

void trace_probation_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::uint64_t entryDigest,
    std::uint32_t observedProbationHits,
    std::uint32_t requiredProbationHits,
    bool admitted,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("probation", transaction, commandName)
        + " entry_digest=" + std::to_string(static_cast<unsigned long long>(entryDigest))
        + " observed_probation_hits=" + std::to_string(observedProbationHits)
        + " required_probation_hits=" + std::to_string(requiredProbationHits)
        + " admitted=" + std::to_string(admitted ? 1 : 0)
        + trace_device_context_fields(transaction)
        + " reason=" + trace_or_unspecified(reason)
        + " reason_class=" + trace_reason_class_or_invalid(reason);
    JTRACE("MSPRB", msg);
}

void trace_graph_large_entry_quarantine(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::uint64_t thresholdBytes,
    std::uint64_t capBytes,
    std::uint32_t capEntries,
    std::uint64_t residentBytes,
    std::uint32_t residentEntries,
    std::uint64_t decayEvictedEntries,
    std::uint64_t capTrimEvictedEntries,
    bool capHit,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("graph_large_quarantine", transaction, commandName)
        + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(thresholdBytes))
        + " cap_bytes=" + std::to_string(static_cast<unsigned long long>(capBytes))
        + " cap_entries=" + std::to_string(capEntries)
        + " resident_bytes=" + std::to_string(static_cast<unsigned long long>(residentBytes))
        + " resident_entries=" + std::to_string(residentEntries)
        + " decay_evicted_entries=" + std::to_string(static_cast<unsigned long long>(decayEvictedEntries))
        + " cap_trim_evicted_entries=" + std::to_string(
            static_cast<unsigned long long>(capTrimEvictedEntries))
        + " cap_hit=" + std::to_string(capHit ? 1 : 0)
        + trace_device_context_fields(transaction)
        + " reason=" + trace_or_unspecified(reason);
    JTRACE("MSADM", msg);
}

void trace_reap_pass(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::size_t reclaimedBytes,
    bool success,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("reap_pass", transaction, commandName)
        + " reclaimed_bytes=" + std::to_string(static_cast<unsigned long long>(reclaimedBytes))
        + " success=" + std::to_string(success ? 1 : 0)
        + trace_device_context_fields(transaction)
        + " reason=" + trace_or_unspecified(reason);
    JTRACE("MSREAP", msg);
}

void trace_fragmentation_recovery(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::uint32_t attempt,
    std::size_t requestBytes,
    std::size_t reapedBytes,
    std::uint64_t quarantineTrimmedEntries,
    std::uint64_t graphEvictedEntries,
    bool success,
    const char* stage,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("fragmentation_recovery", transaction, commandName)
        + " attempt=" + std::to_string(static_cast<unsigned long long>(attempt))
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " reaped_bytes=" + std::to_string(static_cast<unsigned long long>(reapedBytes))
        + " quarantine_trimmed_entries=" + std::to_string(
            static_cast<unsigned long long>(quarantineTrimmedEntries))
        + " graph_evicted_entries=" + std::to_string(static_cast<unsigned long long>(graphEvictedEntries))
        + " success=" + std::to_string(success ? 1 : 0)
        + " stage=" + trace_or_unknown(stage)
        + trace_device_context_fields(transaction)
        + " reason=" + trace_or_unspecified(reason);
    JTRACE("MSFRAG", msg);
}

// Split implementation sections (single-TU include model to preserve exact behavior while
// reducing monolithic file size and keeping ownership boundaries explicit).
#include "Cuda/ResourceManager/JuicerCudaResourceManagerAdmission.cpp"
#include "Cuda/ResourceManager/JuicerCudaResourceManagerSubmission.cpp"
#include "Cuda/ResourceManager/JuicerCudaResourceManagerCommands.cpp"

} // namespace ResourceManager
} // namespace JuicerCuda
