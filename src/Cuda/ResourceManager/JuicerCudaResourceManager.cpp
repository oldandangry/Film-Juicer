// Cuda/ResourceManager/JuicerCudaResourceManager.cpp

#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"

#include "Cuda/ResourceManager/JuicerCudaManagerRegistry.h"
#include "Cuda/ResourceManager/JuicerCudaResourceKeys.h"
#include "Cuda/ResourceManager/JuicerCudaResourcePolicy.h"
#include "Cuda/ResourceManager/JuicerCudaResourceState.h"
#include "Cuda/ResourceManager/JuicerCudaResourceTelemetry.h"

#include <algorithm>
#include <mutex>
#include <unordered_map>

namespace JuicerCuda {
namespace ResourceManager {

namespace {

struct ShadowHistoryKey {
    std::uint64_t instanceToken = 0;
    int deviceId = -1;

    bool operator==(const ShadowHistoryKey& other) const noexcept {
        return instanceToken == other.instanceToken && deviceId == other.deviceId;
    }
};

struct ShadowHistoryKeyHasher {
    std::size_t operator()(const ShadowHistoryKey& key) const noexcept {
        const std::uint64_t mixed =
            key.instanceToken ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.deviceId)) << 1u);
        return static_cast<std::size_t>(mixed);
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

AcquireStatus combine_status(
    AcquireStatus upload,
    AcquireStatus dir,
    AcquireStatus scanner) noexcept {
    if (upload == AcquireStatus::Error || dir == AcquireStatus::Error || scanner == AcquireStatus::Error) {
        return AcquireStatus::Error;
    }
    if (upload == AcquireStatus::Exhausted || dir == AcquireStatus::Exhausted || scanner == AcquireStatus::Exhausted) {
        return AcquireStatus::Exhausted;
    }
    if (upload == AcquireStatus::Busy || dir == AcquireStatus::Busy || scanner == AcquireStatus::Busy) {
        return AcquireStatus::Busy;
    }
    if (upload == AcquireStatus::Miss || dir == AcquireStatus::Miss || scanner == AcquireStatus::Miss) {
        return AcquireStatus::Miss;
    }
    return AcquireStatus::Hit;
}

} // namespace

bool begin_submission(
    SubmissionTransaction& outTransaction,
    const SubmissionSnapshot& snapshot,
    std::string& outError) {
    outError.clear();

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
    outTransaction.snapshot.keySchemaVersion = std::max<std::uint32_t>(1u, outTransaction.snapshot.keySchemaVersion);
    outTransaction.snapshot.traceSchemaVersion = std::max<std::uint32_t>(1u, outTransaction.snapshot.traceSchemaVersion);
    outTransaction.snapshot.keyDigests = normalize_key_digests(outTransaction.snapshot.keyDigests);

    outTransaction.active = true;
    outTransaction.committed = false;

    (void)registry_get_or_create(snapshot.deviceContextKey);
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
    const std::uint64_t acquireId = telemetry_next_acquire_attempt_id();

    if (!transaction.active) {
        outError = "submission transaction is not active";
        telemetry_record_acquire_status(AcquireStatus::Error);
        telemetry_trace_acquire(
            acquireId,
            transaction.transactionId,
            transaction.snapshot.snapshotId,
            transaction.snapshot.traceSchemaVersion,
            AcquireStatus::Error,
            AcquireStatus::Error,
            AcquireStatus::Error,
            AcquireStatus::Error,
            false);
        return false;
    }

    SubmissionSnapshot& snapshot = transaction.snapshot;
    if (snapshot.traceSchemaVersion != kTraceSchemaVersion) {
        outError = "trace schema mismatch";
        telemetry_record_trace_schema_mismatch();
        telemetry_trace_schema_mismatch(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion);
        telemetry_record_acquire_status(AcquireStatus::Error);
        telemetry_trace_acquire(
            acquireId,
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            AcquireStatus::Error,
            AcquireStatus::Error,
            AcquireStatus::Error,
            AcquireStatus::Error,
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
        rawDigests.scannerHash != snapshot.keyDigests.scannerHash) {
        telemetry_trace_key_normalization(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            rawDigests,
            snapshot.keyDigests);
    }

    const ShadowHistoryKey key{ snapshot.instanceToken.value, snapshot.deviceContextKey.deviceId };
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

    const ResourcePlan plan = build_shadow_resource_plan(delta);

    const std::uint64_t prevUpload = hasPrevious ? previous.digests.uploadCoreHash : 0;
    const std::uint64_t prevDir = hasPrevious ? previous.digests.dirHash : 0;
    const std::uint64_t prevScanner = hasPrevious ? previous.digests.scannerHash : 0;

    if (plan.uploadCore.invalidated) {
        telemetry_trace_invalidation(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            "UploadCoreKey",
            delta.keySchemaChanged ? "key_schema_changed" : "upload_hash_changed",
            prevUpload,
            snapshot.keyDigests.uploadCoreHash);
    }
    if (plan.dir.invalidated) {
        telemetry_trace_invalidation(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            "DirKey",
            delta.keySchemaChanged ? "key_schema_changed" : "dir_hash_changed",
            prevDir,
            snapshot.keyDigests.dirHash);
    }
    if (plan.scanner.invalidated) {
        telemetry_trace_invalidation(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            "ScannerColorKey",
            delta.keySchemaChanged ? "key_schema_changed" : "scanner_hash_changed",
            prevScanner,
            snapshot.keyDigests.scannerHash);
    }

    if (delta.keySchemaChanged || delta.uploadCoreChanged) {
        telemetry_trace_dag_edge(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            "UploadCoreKey",
            "UploadCoreResources",
            true,
            "allowed_lane_invalidation");
    }
    if (delta.keySchemaChanged || delta.dirChanged) {
        telemetry_trace_dag_edge(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            "DirKey",
            "DirResources",
            true,
            "allowed_lane_invalidation");
    }
    if (delta.keySchemaChanged || delta.scannerChanged) {
        telemetry_trace_dag_edge(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            "ScannerColorKey",
            "ScannerColorResources",
            true,
            "allowed_lane_invalidation");
    }

    bool forbiddenEdgeDetected = false;
    if (delta.scannerChanged && !delta.uploadCoreChanged && !delta.keySchemaChanged && plan.uploadCore.invalidated) {
        forbiddenEdgeDetected = true;
        telemetry_record_forbidden_invalidation_edge();
        telemetry_trace_dag_edge(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            "ScannerColorKey",
            "UploadCoreResources",
            false,
            "forbidden_edge_scanner_to_upload");
    }
    if (delta.scannerChanged && !delta.dirChanged && !delta.keySchemaChanged && plan.dir.invalidated) {
        forbiddenEdgeDetected = true;
        telemetry_record_forbidden_invalidation_edge();
        telemetry_trace_dag_edge(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            "ScannerColorKey",
            "DirResources",
            false,
            "forbidden_edge_scanner_to_dir");
    }
    if (delta.dirChanged && !delta.uploadCoreChanged && !delta.keySchemaChanged && plan.uploadCore.invalidated) {
        forbiddenEdgeDetected = true;
        telemetry_record_forbidden_invalidation_edge();
        telemetry_trace_dag_edge(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            "DirKey",
            "UploadCoreResources",
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

    const AcquireStatus finalStatus = combine_status(
        plan.uploadCore.acquire.status,
        plan.dir.acquire.status,
        plan.scanner.acquire.status);

    telemetry_record_acquire_status(finalStatus);
    telemetry_trace_acquire(
        acquireId,
        transaction.transactionId,
        snapshot.snapshotId,
        snapshot.traceSchemaVersion,
        finalStatus,
        plan.uploadCore.acquire.status,
        plan.dir.acquire.status,
        plan.scanner.acquire.status,
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
    if (!transaction.active) {
        outError = "submission transaction is not active";
        return false;
    }
    transaction.committed = true;
    transaction.active = false;
    telemetry_record_commit_submission();
    return true;
}

void rollback_submission(
    SubmissionTransaction& transaction,
    const char* reason) noexcept {
    (void)reason;
    if (!transaction.active) {
        return;
    }
    transaction.committed = false;
    transaction.active = false;
    telemetry_record_rollback_submission();
}

} // namespace ResourceManager
} // namespace JuicerCuda
