// Cuda/ResourceManager/JuicerCudaResourceTelemetry.cpp

#include "Cuda/ResourceManager/JuicerCudaResourceTelemetry.h"
#include "Cuda/ResourceManager/JuicerCudaResourceState.h"

#include <string>

#include "Logging.h"

namespace JuicerCuda {
namespace ResourceManager {

void telemetry_record_begin_submission() noexcept {
    global_state().beginSubmissionCalls.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_acquire_plan() noexcept {
    global_state().acquirePlanCalls.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_commit_submission() noexcept {
    global_state().commitSubmissionCalls.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_rollback_submission() noexcept {
    global_state().rollbackSubmissionCalls.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_acquire_status(AcquireStatus status) noexcept {
    ResourceManagerState& state = global_state();
    switch (status) {
    case AcquireStatus::Hit:
        state.acquireStatusHit.fetch_add(1, std::memory_order_relaxed);
        break;
    case AcquireStatus::Miss:
        state.acquireStatusMiss.fetch_add(1, std::memory_order_relaxed);
        break;
    case AcquireStatus::Busy:
        state.acquireStatusBusy.fetch_add(1, std::memory_order_relaxed);
        break;
    case AcquireStatus::Exhausted:
        state.acquireStatusExhausted.fetch_add(1, std::memory_order_relaxed);
        break;
    case AcquireStatus::Error:
    default:
        state.acquireStatusError.fetch_add(1, std::memory_order_relaxed);
        break;
    }
}

void telemetry_record_acquire_status_for_kind(ResourceKind kind, AcquireStatus status) noexcept {
    state_record_acquire_status_for_kind(kind, status);
}

void telemetry_record_trace_schema_mismatch() noexcept {
    global_state().traceSchemaMismatchEvents.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_forbidden_invalidation_edge() noexcept {
    global_state().forbiddenInvalidationEdges.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_module_boundary_violation() noexcept {
    global_state().moduleBoundaryViolations.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_query_mutation_violation() noexcept {
    global_state().queryMutationViolationEvents.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_frame_snapshot_mismatch() noexcept {
    global_state().frameSnapshotMismatchEvents.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_stale_tuple_hard_reject() noexcept {
    global_state().staleTupleHardRejects.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_metadata_mutation_begin() noexcept {
    global_state().metadataMutationBeginCalls.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_metadata_mutation_end() noexcept {
    global_state().metadataMutationEndCalls.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_metadata_mutation_reject() noexcept {
    global_state().metadataMutationRejects.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_metadata_mutation_order_violation() noexcept {
    global_state().metadataMutationOrderViolations.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_metadata_queue_enqueue() noexcept {
    global_state().metadataMutationQueueEnqueueCalls.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_metadata_queue_dequeue() noexcept {
    global_state().metadataMutationQueueDequeueCalls.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_metadata_queue_wait() noexcept {
    global_state().metadataMutationQueueWaitEvents.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_metadata_queue_backpressure() noexcept {
    global_state().metadataMutationQueueBackpressureEvents.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_metadata_queue_reject() noexcept {
    global_state().metadataMutationQueueRejects.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_note_metadata_queue_depth(std::uint64_t depth) noexcept {
    std::atomic<std::uint64_t>& gauge = global_state().metadataMutationQueueMaxDepth;
    std::uint64_t observed = gauge.load(std::memory_order_relaxed);
    while (depth > observed) {
        if (gauge.compare_exchange_weak(observed, depth, std::memory_order_relaxed)) {
            break;
        }
    }
}

std::uint64_t telemetry_next_acquire_attempt_id() noexcept {
    ResourceManagerState& state = global_state();
    std::uint64_t id = state.nextAcquireAttemptId.fetch_add(1, std::memory_order_relaxed);
    if (id == 0) {
        id = state.nextAcquireAttemptId.fetch_add(1, std::memory_order_relaxed);
    }
    return id;
}

void telemetry_trace_schema_announcement(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion) noexcept {
    const std::string msg =
        std::string("transaction_id=") + std::to_string(transactionId) +
        " snapshot_id=" + std::to_string(snapshotId) +
        " trace_schema=" + std::to_string(traceSchemaVersion) +
        " expected_schema=" + std::to_string(kTraceSchemaVersion);
    JTRACE("MSTRC", msg);
}

void telemetry_trace_schema_mismatch(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t observedTraceSchemaVersion) noexcept {
    const std::string msg =
        std::string("transaction_id=") + std::to_string(transactionId) +
        " snapshot_id=" + std::to_string(snapshotId) +
        " observed_schema=" + std::to_string(observedTraceSchemaVersion) +
        " expected_schema=" + std::to_string(kTraceSchemaVersion) +
        " reason=trace_schema_mismatch";
    JTRACE("MSTRC", msg);
}

void telemetry_trace_key_normalization(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    const KeyDigests& before,
    const KeyDigests& after) noexcept {
    const std::string msg =
        std::string("transaction_id=") + std::to_string(transactionId) +
        " snapshot_id=" + std::to_string(snapshotId) +
        " trace_schema=" + std::to_string(traceSchemaVersion) +
        " upload_before=" + std::to_string(before.uploadCoreHash) +
        " upload_after=" + std::to_string(after.uploadCoreHash) +
        " dir_before=" + std::to_string(before.dirHash) +
        " dir_after=" + std::to_string(after.dirHash) +
        " scanner_before=" + std::to_string(before.scannerHash) +
        " scanner_after=" + std::to_string(after.scannerHash) +
        " auto_exposure_before=" + std::to_string(before.autoExposureHash) +
        " auto_exposure_after=" + std::to_string(after.autoExposureHash) +
        " reason=canonical_normalization";
    JTRACE("MSNORM", msg);
}

void telemetry_trace_invalidation(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    const char* lane,
    const char* reason,
    std::uint64_t previousHash,
    std::uint64_t currentHash) noexcept {
    const std::string msg =
        std::string("transaction_id=") + std::to_string(transactionId) +
        " snapshot_id=" + std::to_string(snapshotId) +
        " trace_schema=" + std::to_string(traceSchemaVersion) +
        " lane=" + (lane ? lane : "unknown") +
        " previous_hash=" + std::to_string(previousHash) +
        " current_hash=" + std::to_string(currentHash) +
        " reason=" + (reason ? reason : "unknown");
    JTRACE("MSINV", msg);
}

void telemetry_trace_dag_edge(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    const char* fromNode,
    const char* toNode,
    bool allowed,
    const char* reason) noexcept {
    const std::string msg =
        std::string("transaction_id=") + std::to_string(transactionId) +
        " snapshot_id=" + std::to_string(snapshotId) +
        " trace_schema=" + std::to_string(traceSchemaVersion) +
        " from=" + (fromNode ? fromNode : "unknown") +
        " to=" + (toNode ? toNode : "unknown") +
        " allowed=" + std::to_string(allowed ? 1 : 0) +
        " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSDAG", msg);
}

void telemetry_trace_module_boundary_violation(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    const char* reason) noexcept {
    const std::string msg =
        std::string("transaction_id=") + std::to_string(transactionId) +
        " snapshot_id=" + std::to_string(snapshotId) +
        " trace_schema=" + std::to_string(traceSchemaVersion) +
        " reason=" + (reason ? reason : "unknown");
    JTRACE("MSCMD", msg);
}

void telemetry_trace_query_mutation_violation(
    const char* queryName,
    const DeviceContextKey* key,
    std::uint32_t beforeThreadMutationDepth,
    std::uint32_t afterThreadMutationDepth,
    std::uint64_t beforeThreadMutationTicket,
    std::uint64_t afterThreadMutationTicket,
    std::uint64_t beforeThreadMutationBeginCount,
    std::uint64_t afterThreadMutationBeginCount,
    const char* reason) noexcept {
    const int deviceId = key ? key->deviceId : -1;
    const std::uintptr_t contextBits = key
        ? reinterpret_cast<std::uintptr_t>(key->contextOpaque)
        : 0;
    const std::string msg =
        std::string("event=query_mutation_violation") +
        " query=" + (queryName ? queryName : "unknown") +
        " reason=" + (reason ? reason : "unknown") +
        " device_id=" + std::to_string(deviceId) +
        " context=" + std::to_string(contextBits) +
        " before_thread_mutation_depth=" + std::to_string(beforeThreadMutationDepth) +
        " after_thread_mutation_depth=" + std::to_string(afterThreadMutationDepth) +
        " before_thread_mutation_ticket=" + std::to_string(beforeThreadMutationTicket) +
        " after_thread_mutation_ticket=" + std::to_string(afterThreadMutationTicket) +
        " before_thread_mutation_begin_count=" + std::to_string(beforeThreadMutationBeginCount) +
        " after_thread_mutation_begin_count=" + std::to_string(afterThreadMutationBeginCount);
    JTRACE("MSCMD", msg);
}

void telemetry_trace_frame_snapshot_mismatch(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    std::uint64_t frameToken,
    std::uint64_t expectedSnapshotId,
    std::uint64_t observedSnapshotId) noexcept {
    const std::string msg =
        std::string("transaction_id=") + std::to_string(transactionId) +
        " snapshot_id=" + std::to_string(snapshotId) +
        " trace_schema=" + std::to_string(traceSchemaVersion) +
        " frame_token=" + std::to_string(frameToken) +
        " expected_snapshot_id=" + std::to_string(expectedSnapshotId) +
        " observed_snapshot_id=" + std::to_string(observedSnapshotId) +
        " reason=mixed_snapshot_id_for_frame";
    JTRACE("MSSNP", msg);
}

void telemetry_trace_stale_decision(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    const char* stage,
    const StaleInput& input,
    const StaleDecision& decision) noexcept {
    const std::string msg =
        std::string("transaction_id=") + std::to_string(transactionId) +
        " snapshot_id=" + std::to_string(snapshotId) +
        " trace_schema=" + std::to_string(traceSchemaVersion) +
        " stage=" + (stage ? stage : "unknown") +
        " expected_registry_generation=" + std::to_string(input.expectedRegistryGeneration) +
        " observed_registry_generation=" + std::to_string(input.observedRegistryGeneration) +
        " expected_context_epoch=" + std::to_string(input.expectedContextEpoch) +
        " observed_context_epoch=" + std::to_string(input.observedContextEpoch) +
        " expected_lease_generation=" + std::to_string(input.expectedLeaseGeneration) +
        " observed_lease_generation=" + std::to_string(input.observedLeaseGeneration) +
        " key_schema_mismatch=" + std::to_string(input.keySchemaMismatch ? 1 : 0) +
        " hard_stale=" + std::to_string(decision.hardStale ? 1 : 0) +
        " hard_miss=" + std::to_string(decision.hardMiss ? 1 : 0) +
        " reason=" + to_cstr(decision.reason);
    JTRACE("MSSTL", msg);
}

void telemetry_trace_metadata_mutation(
    const char* phase,
    const char* stage,
    std::uint64_t sequence,
    bool accepted,
    std::uint64_t expectedSequence,
    const char* reason) noexcept {
    const std::string msg =
        std::string("phase=") + (phase ? phase : "unknown") +
        " stage=" + (stage ? stage : "unknown") +
        " sequence=" + std::to_string(sequence) +
        " expected_sequence=" + std::to_string(expectedSequence) +
        " accepted=" + std::to_string(accepted ? 1 : 0) +
        " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSMUT", msg);
}

void telemetry_trace_metadata_queue(
    const char* eventName,
    const char* stage,
    std::uint64_t ticket,
    std::uint64_t depth,
    std::uint64_t waitedMs,
    bool accepted,
    const char* reason) noexcept {
    const std::string msg =
        std::string("event=") + (eventName ? eventName : "unknown") +
        " stage=" + (stage ? stage : "unknown") +
        " ticket=" + std::to_string(ticket) +
        " depth=" + std::to_string(depth) +
        " waited_ms=" + std::to_string(waitedMs) +
        " accepted=" + std::to_string(accepted ? 1 : 0) +
        " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSMQ", msg);
}

void telemetry_trace_acquire(
    std::uint64_t acquireId,
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    AcquireStatus finalStatus,
    const ResourcePlan& plan,
    bool hadPreviousSnapshot) noexcept {
    std::string msg =
        std::string("acquire_id=") + std::to_string(acquireId) +
        " transaction_id=" + std::to_string(transactionId) +
        " snapshot_id=" + std::to_string(snapshotId) +
        " trace_schema=" + std::to_string(traceSchemaVersion) +
        " final_status=" + to_cstr(finalStatus) +
        " had_previous=" + std::to_string(hadPreviousSnapshot ? 1 : 0);
    for (ResourceKind kind : kResourceKindOrder) {
        const ResourcePlanEntry& entry = resource_plan_entry(plan, kind);
        const ResourceKindContractEntry& contract = resource_kind_contract_entry(kind);
        msg += std::string(" ") + contract.acquireStatusField + "=" + to_cstr(entry.acquire.status);
        msg += std::string(" ") + contract.invalidationLane + "_invalidated=" +
            std::to_string(entry.invalidated ? 1 : 0);
    }
    JTRACE("MSACQ", msg);
}

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
    const char* reason) noexcept {
    const std::string msg =
        std::string("transaction_id=") + std::to_string(transactionId) +
        " snapshot_id=" + std::to_string(snapshotId) +
        " trace_schema=" + std::to_string(traceSchemaVersion) +
        " mode=" + (mode ? mode : "unknown") +
        " event=" + (eventName ? eventName : "unknown") +
        " hit=" + std::to_string(hit ? 1 : 0) +
        " key_hash=" + std::to_string(keyHash) +
        " meter_w=" + std::to_string(meterWidth) +
        " meter_h=" + std::to_string(meterHeight) +
        " had_previous=" + std::to_string(hadPrevious ? 1 : 0) +
        " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSAEX", msg);
}

} // namespace ResourceManager
} // namespace JuicerCuda
