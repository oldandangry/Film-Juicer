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

void telemetry_record_trace_schema_mismatch() noexcept {
    global_state().traceSchemaMismatchEvents.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_forbidden_invalidation_edge() noexcept {
    global_state().forbiddenInvalidationEdges.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_module_boundary_violation() noexcept {
    global_state().moduleBoundaryViolations.fetch_add(1, std::memory_order_relaxed);
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

void telemetry_trace_acquire(
    std::uint64_t acquireId,
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    AcquireStatus finalStatus,
    AcquireStatus uploadStatus,
    AcquireStatus dirStatus,
    AcquireStatus scannerStatus,
    bool hadPreviousSnapshot) noexcept {
    const std::string msg =
        std::string("acquire_id=") + std::to_string(acquireId) +
        " transaction_id=" + std::to_string(transactionId) +
        " snapshot_id=" + std::to_string(snapshotId) +
        " trace_schema=" + std::to_string(traceSchemaVersion) +
        " final_status=" + to_cstr(finalStatus) +
        " upload_status=" + to_cstr(uploadStatus) +
        " dir_status=" + to_cstr(dirStatus) +
        " scanner_status=" + to_cstr(scannerStatus) +
        " had_previous=" + std::to_string(hadPreviousSnapshot ? 1 : 0);
    JTRACE("MSACQ", msg);
}

} // namespace ResourceManager
} // namespace JuicerCuda
