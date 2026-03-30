// Cuda/ResourceManager/JuicerCudaResourceManager.h
//
// Shared resource-manager API for lifecycle, submissions, state tracking, and telemetry.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <string>

#include "Cuda/JuicerCudaPayloads.h"
#include "Cuda/JuicerCudaResources.h"
#include "Cuda/ResourceManager/JuicerCudaResourceCore.h"

struct WorkingState;
namespace Print {
    struct Runtime;
    struct Params;
}

namespace JuicerCuda {
namespace ResourceManager {

// Lifecycle state for the per-context manager registry.
enum class RegistryRetireReason : std::uint8_t {
    Unknown = 0,
    Idle = 1,
    ContextReset = 2
};

enum class ContextLifecycleState : std::uint8_t {
    Unbound = 0,
    Binding = 1,
    Active = 2,
    Freezing = 3,
    Draining = 4,
    Rebinding = 5,
    Retired = 6
};

enum class LifecycleStageDecision : std::uint8_t {
    Allowed = 0,
    MissingRegistryEntry = 1,
    StateNotAllowed = 2,
    TimedOut = 3
};

struct RegistryHandle {
    std::uint64_t value = 0;
};

struct LifecycleStageValidation {
    LifecycleStageDecision decision = LifecycleStageDecision::MissingRegistryEntry;
    ContextLifecycleState observedState = ContextLifecycleState::Unbound;
    std::uint64_t observedStateAgeMs = 0;
    bool escalated = false;
};

const char* to_cstr(ContextLifecycleState state) noexcept;
const char* to_cstr(LifecycleStageDecision decision) noexcept;

// Registry lookups and lifecycle transitions for device-context managers.
RegistryHandle registry_get_or_create(const DeviceContextKey& key) noexcept;
bool registry_get(const DeviceContextKey& key, RegistryHandle& outHandle) noexcept;
bool registry_get_lifecycle_state(const DeviceContextKey& key, ContextLifecycleState& outState) noexcept;
bool registry_validate_lifecycle_stage(
    const DeviceContextKey& key,
    bool allowNonActiveRelease,
    LifecycleStageValidation& outValidation) noexcept;
bool registry_note_submission_begin(const DeviceContextKey& key) noexcept;
bool registry_note_submission_end(const DeviceContextKey& key) noexcept;
bool registry_transition_lifecycle_state(
    const DeviceContextKey& key,
    ContextLifecycleState expectedState,
    ContextLifecycleState desiredState,
    const char* reason) noexcept;
bool registry_freeze_drain_bump_resume(
    const DeviceContextKey& key,
    const char* reason) noexcept;
void registry_retire(
    RegistryHandle handle,
    RegistryRetireReason reason,
    const DeviceContextKey* managerKey = nullptr) noexcept;

bool query_submission_active(const SubmissionTransaction& transaction) noexcept;

// Read-only query surface; these helpers must not mutate manager state.
AllocatorBackendMode query_allocator_backend_mode(const DeviceContextKey& key) noexcept;

// Submission planning and execution entry points.
bool begin_submission(
    SubmissionTransaction& outTransaction,
    const SubmissionSnapshot& snapshot,
    std::string& outError);

bool acquire_plan(
    SubmissionTransaction& transaction,
    std::string& outError);

bool commit_submission(
    SubmissionTransaction& transaction,
    void* cudaStreamOpaque,
    std::string& outError);

bool command_freeze_drain_bump_resume(
    const DeviceContextKey& key,
    const char* reason,
    std::string& outError);

bool command_retire_context_reset(
    const DeviceContextKey& key,
    std::string& outError);

bool command_retire_context_idle(
    const DeviceContextKey& key,
    std::string& outError);

bool command_ensure_uploaded(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    void* cudaStreamOpaque,
    std::string& outError);

bool command_ensure_current_medium_uploaded(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    bool negativeMedium,
    const ScratchRequestDescriptor& scratchRequest,
    void* cudaStreamOpaque,
    std::string& outError);

bool command_ensure_scan_lut(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    bool negativeMedium,
    const ScratchRequestDescriptor& scratchRequest,
    void* cudaStreamOpaque,
    std::string& outError);

bool command_ensure_scan_error_flag(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    void* cudaStreamOpaque,
    std::string& outError);

bool command_checkpoint_scratch_phase(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const ScratchRequestDescriptor& scratchRequest,
    const char* commandName,
    std::string& outError);

bool command_ensure_print_illuminant_filtered(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    const Print::Runtime& prt,
    const Print::Params& params,
    const ScratchRequestDescriptor& scratchRequest,
    void* cudaStreamOpaque,
    std::string& outError);

bool command_ensure_optics_scratch(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const ScratchRequestDescriptor& scratchRequest,
    void* cudaStreamOpaque,
    std::string& outError);

bool command_ensure_spatial_dir_scratch(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const ScratchRequestDescriptor& scratchRequest,
    void* cudaStreamOpaque,
    std::string& outError);

bool command_ensure_spatial_dir_kernel(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    JuicerCuda::Resources::DeviceGaussianKernel& kernel,
    float sigma,
    void* cudaStreamOpaque,
    std::string& outError);

bool command_ensure_gaussian_kernel(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    JuicerCuda::Resources::DeviceGaussianKernel& kernel,
    float sigma,
    void* cudaStreamOpaque,
    std::string& outError);

bool command_ensure_halation_kernel(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    JuicerCuda::Resources::DeviceGaussianKernel& kernel,
    float sigma,
    void* cudaStreamOpaque,
    std::string& outError);

bool command_ensure_auto_exposure_buffers(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    int meterWidth,
    int meterHeight,
    std::uint64_t autoExposureKeyHash,
    void* cudaStreamOpaque,
    std::string& outError);

bool command_launch_base_pipeline_graph(
    SubmissionTransaction& transaction,
    JuicerCuda::PipelineRunParams& run,
    int renderModeKey,
    void* cudaStreamOpaque,
    int& outCudaErrorCode,
    std::string& outError);

bool error_is_scratch_exhausted(const std::string& error) noexcept;

void rollback_submission(
    SubmissionTransaction& transaction,
    const char* reason) noexcept;

// Process-wide counters and sequencing state shared across RM registry and submission paths.
struct ResourceKindAcquireCounters {
    std::atomic<std::uint64_t> hit{ 0 };
    std::atomic<std::uint64_t> miss{ 0 };
    std::atomic<std::uint64_t> busy{ 0 };
    std::atomic<std::uint64_t> exhausted{ 0 };
    std::atomic<std::uint64_t> error{ 0 };
};

struct ResourceManagerState {
    std::atomic<std::uint64_t> nextTransactionId{ 1 };
    std::atomic<std::uint64_t> nextAcquireAttemptId{ 1 };
    std::atomic<std::uint64_t> nextLeaseGeneration{ 1 };
    std::atomic<std::uint64_t> nextMetadataMutationSequence{ 1 };
    std::atomic<std::uint64_t> registryGeneration{ 1 };
    std::atomic<std::uint64_t> contextEpoch{ 1 };
    std::atomic<std::uint64_t> beginSubmissionCalls{ 0 };
    std::atomic<std::uint64_t> acquirePlanCalls{ 0 };
    std::atomic<std::uint64_t> commitSubmissionCalls{ 0 };
    std::atomic<std::uint64_t> rollbackSubmissionCalls{ 0 };
    std::atomic<std::uint64_t> acquireStatusHit{ 0 };
    std::atomic<std::uint64_t> acquireStatusMiss{ 0 };
    std::atomic<std::uint64_t> acquireStatusBusy{ 0 };
    std::atomic<std::uint64_t> acquireStatusExhausted{ 0 };
    std::atomic<std::uint64_t> acquireStatusError{ 0 };
    std::atomic<std::uint64_t> traceSchemaMismatchEvents{ 0 };
    std::atomic<std::uint64_t> forbiddenInvalidationEdges{ 0 };
    std::atomic<std::uint64_t> moduleBoundaryViolations{ 0 };
    std::atomic<std::uint64_t> queryMutationViolationEvents{ 0 };
    std::atomic<std::uint64_t> frameSnapshotMismatchEvents{ 0 };
    std::atomic<std::uint64_t> staleTupleHardRejects{ 0 };
    std::atomic<std::uint64_t> lifecycleTransitionCalls{ 0 };
    std::atomic<std::uint64_t> lifecycleTransitionRejects{ 0 };
    std::atomic<std::uint64_t> lifecycleStageRejects{ 0 };
    std::atomic<std::uint64_t> lifecycleBarrierCalls{ 0 };
    std::atomic<std::uint64_t> lifecycleBarrierRejects{ 0 };
    std::atomic<std::uint64_t> lifecycleTimeoutEvents{ 0 };
    std::atomic<std::uint64_t> metadataMutationBeginCalls{ 0 };
    std::atomic<std::uint64_t> metadataMutationEndCalls{ 0 };
    std::atomic<std::uint64_t> metadataMutationRejects{ 0 };
    std::atomic<std::uint64_t> metadataMutationOrderViolations{ 0 };
    std::atomic<std::uint64_t> metadataMutationQueueEnqueueCalls{ 0 };
    std::atomic<std::uint64_t> metadataMutationQueueDequeueCalls{ 0 };
    std::atomic<std::uint64_t> metadataMutationQueueWaitEvents{ 0 };
    std::atomic<std::uint64_t> metadataMutationQueueBackpressureEvents{ 0 };
    std::atomic<std::uint64_t> metadataMutationQueueRejects{ 0 };
    std::atomic<std::uint64_t> metadataMutationQueueMaxDepth{ 0 };
    std::atomic<std::uint64_t> scratchPolicyWaitEvents{ 0 };
    std::atomic<std::uint64_t> scratchPolicyExhaustedEvents{ 0 };
    std::atomic<std::uint64_t> scratchBucketAcquireAttempts{ 0 };
    std::atomic<std::uint64_t> scratchBucketExhaustedEvents{ 0 };
    std::atomic<std::uint64_t> scratchBucketStarvationEvents{ 0 };
    std::atomic<std::uint64_t> scratchAllocGrowthEvents{ 0 };
    std::atomic<std::uint64_t> scratchReuseEvents{ 0 };
    std::atomic<std::uint64_t> scratchLargeQuarantineTrimEvents{ 0 };
    std::atomic<std::uint64_t> scratchLargeQuarantineDecayEvents{ 0 };
    std::atomic<std::uint64_t> budgetReclaimRetryAttempts{ 0 };
    std::atomic<std::uint64_t> budgetReclaimRetrySuccess{ 0 };
    std::atomic<std::uint64_t> budgetAllocatorOomEvents{ 0 };
    std::atomic<std::uint64_t> fragmentationRecoveryAttempts{ 0 };
    std::atomic<std::uint64_t> fragmentationRecoverySuccess{ 0 };
    std::atomic<std::uint64_t> fragmentationRecoveryFailures{ 0 };
    std::atomic<std::uint64_t> fragmentationRecoveryGraphEvictedEntries{ 0 };
    std::atomic<std::uint64_t> pressureStateTransitions{ 0 };
    std::atomic<std::uint64_t> pressureTransitionDwellDefers{ 0 };
    std::atomic<std::uint64_t> pressureTransitionRateDefers{ 0 };
    std::atomic<std::uint64_t> reserveCrossingEvents{ 0 };
    std::atomic<std::uint64_t> reserveAdaptationEvents{ 0 };
    std::atomic<std::uint64_t> opportunisticFreezeEnterEvents{ 0 };
    std::atomic<std::uint64_t> opportunisticFreezeExitEvents{ 0 };
    std::atomic<std::uint64_t> opportunisticFreezeDenyEvents{ 0 };
    std::atomic<std::uint64_t> activeBurstEnterEvents{ 0 };
    std::atomic<std::uint64_t> activeBurstExitEvents{ 0 };
    std::atomic<std::uint64_t> activeBurstCapHitEvents{ 0 };
    std::atomic<std::uint64_t> headroomSourceSwitches{ 0 };
    std::atomic<std::uint64_t> allocFailAboveHeadroomEvents{ 0 };
    std::atomic<std::uint64_t> retireReapPasses{ 0 };
    std::atomic<std::uint64_t> retireReapBytes{ 0 };
    std::atomic<std::uint64_t> registryLiveManagers{ 0 };
    std::atomic<std::uint64_t> registryReapEvents{ 0 };
    std::atomic<std::uint64_t> managerActiveBytes{ 0 };
    std::atomic<std::uint64_t> managerReclaimableBytes{ 0 };
    std::atomic<std::uint64_t> managerRetirePendingBytes{ 0 };
    std::atomic<std::uint64_t> hostAssetCacheBytes{ 0 };
    std::atomic<std::uint64_t> hostAssetCacheTrimEvents{ 0 };
    std::atomic<std::uint64_t> hostAssetCacheTrimBytes{ 0 };
    std::atomic<std::uint64_t> hostAssetCacheCapHits{ 0 };
    std::atomic<std::uint64_t> pinnedStagingBytes{ 0 };
    std::atomic<std::uint64_t> pinnedStagingCapHits{ 0 };
    std::atomic<std::uint64_t> pinnedStagingFallbackEvents{ 0 };
    std::atomic<std::uint64_t> pinnedStagingTrimEvents{ 0 };
    std::atomic<std::uint64_t> pinnedStagingTrimBytes{ 0 };
    std::atomic<std::uint64_t> transientNonManagerBytes{ 0 };
    std::atomic<std::uint64_t> allocatorEffectiveHeadroomBytes{ 0 };
    std::atomic<std::uint64_t> allocatorPoolReservedBytes{ 0 };
    std::atomic<std::uint64_t> allocatorPoolUsedBytes{ 0 };
    std::atomic<std::uint64_t> transientReservationRequests{ 0 };
    std::atomic<std::uint64_t> transientReservationGranted{ 0 };
    std::atomic<std::uint64_t> transientReservationDeferred{ 0 };
    std::atomic<std::uint64_t> transientReservationDenied{ 0 };
    std::atomic<std::uint64_t> builderReservationRequests{ 0 };
    std::atomic<std::uint64_t> builderReservationGranted{ 0 };
    std::atomic<std::uint64_t> builderReservationDeferred{ 0 };
    std::atomic<std::uint64_t> builderReservationDenied{ 0 };
    std::atomic<std::uint64_t> builderFairnessTokenDeferred{ 0 };
    std::atomic<std::uint64_t> builderReservationBypass{ 0 };
    std::atomic<std::uint64_t> builderFairnessTokenBypass{ 0 };
    std::atomic<std::uint64_t> builderFairnessWaitEvents{ 0 };
    std::atomic<std::uint64_t> scratchBuilderBytesInFlight{ 0 };
    std::atomic<std::uint64_t> lutBuilderBytesInFlight{ 0 };
    std::atomic<std::uint64_t> graphBuilderBytesInFlight{ 0 };
    std::atomic<std::uint64_t> uploadReservationRequests{ 0 };
    std::atomic<std::uint64_t> uploadReservationGranted{ 0 };
    std::atomic<std::uint64_t> uploadReservationDeferred{ 0 };
    std::atomic<std::uint64_t> uploadReservationDenied{ 0 };
    std::atomic<std::uint64_t> uploadReservationBypass{ 0 };
    std::atomic<std::uint64_t> uploadFairnessTokenDeferred{ 0 };
    std::atomic<std::uint64_t> uploadFairnessTokenBypass{ 0 };
    std::atomic<std::uint64_t> uploadFairnessWaitEvents{ 0 };
    std::atomic<std::uint64_t> criticalBuilderWaitEvents{ 0 };
    std::atomic<std::uint64_t> criticalBuilderWaitTotalMs{ 0 };
    std::atomic<std::uint64_t> criticalUploadWaitEvents{ 0 };
    std::atomic<std::uint64_t> criticalUploadWaitTotalMs{ 0 };
    std::atomic<std::uint64_t> criticalLaneStarvationEvents{ 0 };
    std::atomic<std::uint64_t> uploadEmergencyShedDenials{ 0 };
    std::atomic<std::uint64_t> builderEmergencyShedDenials{ 0 };
    std::atomic<std::uint64_t> copyComputeGuardShedEvents{ 0 };
    std::atomic<std::uint64_t> cacheAdmissionTooLargeEvents{ 0 };
    std::atomic<std::uint64_t> cacheAdmissionProbationDeferredEvents{ 0 };
    std::atomic<std::uint64_t> cacheAdmissionProbationAdmitEvents{ 0 };
    std::atomic<std::uint64_t> cacheAdmissionCriticalOverrideEvents{ 0 };
    std::atomic<std::uint64_t> largeEntryReadmitBlockedEvents{ 0 };
    std::atomic<std::uint64_t> largeEntryReadmitGhostBypassEvents{ 0 };
    std::atomic<std::uint64_t> admissionChurnSampleEvents{ 0 };
    std::atomic<std::uint64_t> admissionChurnEnterEvents{ 0 };
    std::atomic<std::uint64_t> admissionChurnExitEvents{ 0 };
    std::atomic<std::uint64_t> keepHotSurfaceTraceEvents{ 0 };
    std::atomic<std::uint64_t> keepHotBypassEvents{ 0 };
    std::atomic<std::uint64_t> keepHotForcedEvictEvents{ 0 };
    std::atomic<std::uint64_t> burstDebtSurfaceTraceEvents{ 0 };
    std::atomic<std::uint64_t> burstDebtSampleEvents{ 0 };
    std::atomic<std::uint64_t> burstDebtAccrualEvents{ 0 };
    std::atomic<std::uint64_t> burstDebtThrottleEvents{ 0 };
    std::atomic<std::uint64_t> supersededBuilderCancelSurfaceTraceEvents{ 0 };
    std::atomic<std::uint64_t> supersededBuilderCancelEvents{ 0 };
    std::atomic<std::uint64_t> supersededBuilderCancelSavedBytes{ 0 };
    std::atomic<std::uint64_t> graphNonResidentServeEvents{ 0 };
    std::atomic<std::uint64_t> graphLargeEntryDecayEvents{ 0 };
    std::atomic<std::uint64_t> graphLargeEntryTrimEvents{ 0 };
    std::atomic<std::uint64_t> graphLargeEntryCapHits{ 0 };
    std::atomic<std::uint64_t> graphLargeEntryResidentBytes{ 0 };
    std::atomic<std::uint64_t> tierCircuitOpenEvents{ 0 };
    std::atomic<std::uint64_t> tierCircuitHalfOpenEvents{ 0 };
    std::atomic<std::uint64_t> tierCircuitCloseEvents{ 0 };
    std::atomic<std::uint64_t> tierCircuitBlockedEvents{ 0 };
    std::atomic<std::uint64_t> uploadBytesInFlight{ 0 };
    std::array<ResourceKindAcquireCounters, kResourceKindCount> acquireStatusByKind{};
};

ResourceManagerState& global_state() noexcept;
void state_record_acquire_status_for_kind(ResourceKind kind, AcquireStatus status) noexcept;
void state_note_latest_snapshot(const SubmissionSnapshot& snapshot) noexcept;
bool state_snapshot_is_superseded(
    const SubmissionSnapshot& snapshot,
    std::uint64_t* outLatestSnapshotId = nullptr) noexcept;
void state_clear_latest_snapshot_for_context(const DeviceContextKey& key) noexcept;

enum class LeaseObservationMode : std::uint8_t {
    ActiveOnly = 0,
    Always = 1
};

StaleInput state_build_stale_input(
    const SubmissionTransaction& transaction,
    LeaseObservationMode leaseObservationMode = LeaseObservationMode::ActiveOnly) noexcept;

struct QueryReadOnlySnapshot {
    bool threadMutationActive = false;
    std::uint32_t threadMutationDepth = 0;
    std::uint64_t threadMutationTicket = 0;
    std::uint64_t threadMutationBeginCount = 0;
};

// Captures thread-local mutation state around a read-only RM query.
class QueryReadOnlyGuard {
public:
    explicit QueryReadOnlyGuard(const char* queryName, const DeviceContextKey* key = nullptr) noexcept;
    ~QueryReadOnlyGuard() noexcept;

private:
    const char* _queryName = nullptr;
    const DeviceContextKey* _key = nullptr;
    QueryReadOnlySnapshot _before{};
};

struct MetadataMutationScope {
    std::uint64_t sequence = 0;
    std::uint64_t queueTicket = 0;
    DeviceContextKey managerKey{};
    bool hasManagerKey = false;
    bool active = false;
};

bool metadata_mutation_begin(
    const char* stage,
    MetadataMutationScope& outScope,
    const DeviceContextKey* managerKey = nullptr) noexcept;
void metadata_mutation_end(MetadataMutationScope& scope, const char* stage) noexcept;
bool metadata_mutation_thread_active() noexcept;
std::uint32_t metadata_mutation_thread_depth() noexcept;
std::uint64_t metadata_mutation_thread_ticket() noexcept;
std::uint64_t metadata_mutation_thread_begin_count() noexcept;

// Serializes multi-step metadata edits on a manager key and releases automatically on scope exit.
class MetadataMutationGuard {
public:
    explicit MetadataMutationGuard(const char* stage, const DeviceContextKey* managerKey = nullptr) noexcept;
    ~MetadataMutationGuard() noexcept;

    bool ok() const noexcept { return _scope.active; }
    std::uint64_t sequence() const noexcept { return _scope.sequence; }

private:
    const char* _stage = nullptr;
    bool _hasManagerKey = false;
    DeviceContextKey _managerKey{};
    MetadataMutationScope _scope{};
};

// Lightweight counter helpers used throughout the RM implementation.
#ifndef JUICER_RM_TELEMETRY_COUNTERS_COMPILED
#define JUICER_RM_TELEMETRY_COUNTERS_COMPILED 0
#endif

inline void telemetry_counter_add(
    std::atomic<std::uint64_t>& counter,
    std::uint64_t delta = 1) noexcept {
#if JUICER_RM_TELEMETRY_COUNTERS_COMPILED
    if (delta == 0) {
        return;
    }
    counter.fetch_add(delta, std::memory_order_relaxed);
#else
    (void)counter;
    (void)delta;
#endif
}

inline void telemetry_counter_note_max(
    std::atomic<std::uint64_t>& counter,
    std::uint64_t value) noexcept {
#if JUICER_RM_TELEMETRY_COUNTERS_COMPILED
    std::uint64_t observed = counter.load(std::memory_order_relaxed);
    while (value > observed) {
        if (counter.compare_exchange_weak(
                observed,
                value,
                std::memory_order_relaxed,
                std::memory_order_relaxed)) {
            break;
        }
    }
#else
    (void)counter;
    (void)value;
#endif
}

inline void telemetry_counter_subtract_saturating(
    std::atomic<std::uint64_t>& counter,
    std::uint64_t delta) noexcept {
#if JUICER_RM_TELEMETRY_COUNTERS_COMPILED
    if (delta == 0) {
        return;
    }
    std::uint64_t current = counter.load(std::memory_order_relaxed);
    while (true) {
        const std::uint64_t next = (current >= delta) ? (current - delta) : 0;
        if (counter.compare_exchange_weak(
                current,
                next,
                std::memory_order_relaxed,
                std::memory_order_relaxed)) {
            return;
        }
    }
#else
    (void)counter;
    (void)delta;
#endif
}

// Counter and trace hooks implemented by the RM runtime.
void telemetry_record_begin_submission() noexcept;
void telemetry_record_acquire_plan() noexcept;
void telemetry_record_commit_submission() noexcept;
void telemetry_record_rollback_submission() noexcept;
void telemetry_record_acquire_status(AcquireStatus status) noexcept;
void telemetry_record_acquire_status_for_kind(ResourceKind kind, AcquireStatus status) noexcept;
void telemetry_record_trace_schema_mismatch() noexcept;
void telemetry_record_forbidden_invalidation_edge() noexcept;
void telemetry_record_module_boundary_violation() noexcept;
void telemetry_record_query_mutation_violation() noexcept;
void telemetry_record_frame_snapshot_mismatch() noexcept;
void telemetry_record_stale_tuple_hard_reject() noexcept;
void telemetry_record_metadata_mutation_begin() noexcept;
void telemetry_record_metadata_mutation_end() noexcept;
void telemetry_record_metadata_mutation_reject() noexcept;
void telemetry_record_metadata_mutation_order_violation() noexcept;
void telemetry_record_metadata_queue_enqueue() noexcept;
void telemetry_record_metadata_queue_dequeue() noexcept;
void telemetry_record_metadata_queue_wait() noexcept;
void telemetry_record_metadata_queue_backpressure() noexcept;
void telemetry_record_metadata_queue_reject() noexcept;
void telemetry_note_metadata_queue_depth(std::uint64_t depth) noexcept;
std::uint64_t telemetry_next_acquire_attempt_id() noexcept;

void telemetry_trace_schema_announcement(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion) noexcept;

void telemetry_trace_schema_mismatch(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t observedTraceSchemaVersion) noexcept;

void telemetry_trace_key_normalization(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    const KeyDigests& before,
    const KeyDigests& after) noexcept;

void telemetry_trace_invalidation(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    const char* lane,
    const char* reason,
    std::uint64_t previousHash,
    std::uint64_t currentHash) noexcept;

void telemetry_trace_dag_edge(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    const char* fromNode,
    const char* toNode,
    bool allowed,
    const char* reason) noexcept;

void telemetry_trace_module_boundary_violation(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    const char* reason) noexcept;

void telemetry_trace_query_mutation_violation(
    const char* queryName,
    const DeviceContextKey* key,
    std::uint32_t beforeThreadMutationDepth,
    std::uint32_t afterThreadMutationDepth,
    std::uint64_t beforeThreadMutationTicket,
    std::uint64_t afterThreadMutationTicket,
    std::uint64_t beforeThreadMutationBeginCount,
    std::uint64_t afterThreadMutationBeginCount,
    const char* reason) noexcept;

void telemetry_trace_frame_snapshot_mismatch(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    std::uint64_t frameToken,
    std::uint64_t expectedSnapshotId,
    std::uint64_t observedSnapshotId) noexcept;

void telemetry_trace_stale_decision(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    const char* stage,
    const StaleInput& input,
    const StaleDecision& decision) noexcept;

void telemetry_trace_metadata_mutation(
    const char* phase,
    const char* stage,
    std::uint64_t sequence,
    bool accepted,
    std::uint64_t expectedSequence,
    const char* reason) noexcept;

void telemetry_trace_metadata_queue(
    const char* eventName,
    const char* stage,
    std::uint64_t ticket,
    std::uint64_t depth,
    std::uint64_t waitedMs,
    bool accepted,
    const char* reason) noexcept;

void telemetry_trace_acquire(
    std::uint64_t acquireId,
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    AcquireStatus finalStatus,
    const ResourcePlan& plan,
    bool hadPreviousSnapshot) noexcept;

void telemetry_trace_auto_exposure_ownership(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    const char* mode,
    const char* eventName,
    bool hit,
    std::uint64_t keyHash,
    int meterWidth,
    int meterHeight,
    bool hadPrevious,
    const char* reason) noexcept;

} // namespace ResourceManager
} // namespace JuicerCuda
