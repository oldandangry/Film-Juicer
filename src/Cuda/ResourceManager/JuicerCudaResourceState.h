// Cuda/ResourceManager/JuicerCudaResourceState.h
//
// Phase-0 state scaffolding.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>

#include "Cuda/ResourceManager/JuicerCudaResourcePolicy.h"
#include "Cuda/ResourceManager/JuicerCudaResourceTypes.h"

namespace JuicerCuda {
namespace ResourceManager {

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
    std::atomic<std::uint64_t> frameSnapshotMismatchEvents{ 0 };
    std::atomic<std::uint64_t> staleTupleHardRejects{ 0 };
    std::atomic<std::uint64_t> lifecycleTransitionCalls{ 0 };
    std::atomic<std::uint64_t> lifecycleTransitionRejects{ 0 };
    std::atomic<std::uint64_t> lifecycleStageRejects{ 0 };
    std::atomic<std::uint64_t> lifecycleBarrierCalls{ 0 };
    std::atomic<std::uint64_t> lifecycleBarrierRejects{ 0 };
    std::atomic<std::uint64_t> metadataMutationBeginCalls{ 0 };
    std::atomic<std::uint64_t> metadataMutationEndCalls{ 0 };
    std::atomic<std::uint64_t> metadataMutationRejects{ 0 };
    std::atomic<std::uint64_t> metadataMutationOrderViolations{ 0 };
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
    std::atomic<std::uint64_t> fragmentationRecoveryQuarantineTrimmedEntries{ 0 };
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
    std::atomic<std::uint64_t> burstDebtSurfaceTraceEvents{ 0 };
    std::atomic<std::uint64_t> supersededBuilderCancelSurfaceTraceEvents{ 0 };
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

struct MetadataMutationScope {
    std::uint64_t sequence = 0;
    bool active = false;
};

bool metadata_mutation_begin(const char* stage, MetadataMutationScope& outScope) noexcept;
void metadata_mutation_end(MetadataMutationScope& scope, const char* stage) noexcept;

class MetadataMutationGuard {
public:
    explicit MetadataMutationGuard(const char* stage) noexcept;
    ~MetadataMutationGuard() noexcept;

    bool ok() const noexcept { return _scope.active; }
    std::uint64_t sequence() const noexcept { return _scope.sequence; }

private:
    const char* _stage = nullptr;
    MetadataMutationScope _scope{};
};

} // namespace ResourceManager
} // namespace JuicerCuda
