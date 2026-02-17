// Cuda/ResourceManager/JuicerCudaResourceTelemetry.h
//
// Phase-0 telemetry scaffolding for fixed counters and trace schema contract.
#pragma once

#include <cstdint>

#include "Cuda/ResourceManager/JuicerCudaResourcePolicy.h"
#include "Cuda/ResourceManager/JuicerCudaResourceTypes.h"

namespace JuicerCuda {
namespace ResourceManager {

void telemetry_record_begin_submission() noexcept;
void telemetry_record_acquire_plan() noexcept;
void telemetry_record_commit_submission() noexcept;
void telemetry_record_rollback_submission() noexcept;
void telemetry_record_acquire_status(AcquireStatus status) noexcept;
void telemetry_record_acquire_status_for_kind(ResourceKind kind, AcquireStatus status) noexcept;
void telemetry_record_trace_schema_mismatch() noexcept;
void telemetry_record_forbidden_invalidation_edge() noexcept;
void telemetry_record_module_boundary_violation() noexcept;
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
