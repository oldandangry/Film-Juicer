// Cuda/ResourceManager/JuicerCudaResourceManager.cpp

#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"

#include "Cuda/JuicerCudaResources.h"
#include "Cuda/ResourceManager/JuicerCudaManagerRegistry.h"
#include "Cuda/ResourceManager/JuicerCudaResourceKeys.h"
#include "Cuda/ResourceManager/JuicerCudaResourcePolicy.h"
#include "Cuda/ResourceManager/JuicerCudaResourceState.h"
#include "Cuda/ResourceManager/JuicerCudaResourceTelemetry.h"
#include "Print.h"
#include "WorkingState.h"

#include <algorithm>
#include <mutex>
#include <unordered_map>

namespace JuicerCuda {
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

bool key_digests_equal(const KeyDigests& lhs, const KeyDigests& rhs) noexcept {
    return lhs.uploadCoreHash == rhs.uploadCoreHash &&
        lhs.dirHash == rhs.dirHash &&
        lhs.scannerHash == rhs.scannerHash;
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

void trace_lifecycle_stage_decision(
    const SubmissionTransaction& transaction,
    ContextLifecycleState observedState,
    const char* stage,
    bool accepted,
    const char* reason) {
    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("transaction_id=") + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " stage=" + (stage ? stage : "unknown")
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " observed_state=" + to_cstr(observedState)
        + " accepted=" + std::to_string(accepted ? 1 : 0)
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSLCY", msg);
}

bool lifecycle_state_allowed_for_stage(ContextLifecycleState state, bool allowNonActiveRelease) {
    if (allowNonActiveRelease) {
        return state != ContextLifecycleState::Unbound;
    }
    return state == ContextLifecycleState::Active;
}

bool validate_lifecycle_for_stage(const SubmissionTransaction& transaction,
                                  const char* stage,
                                  bool allowNonActiveRelease,
                                  std::string* outError) {
    ContextLifecycleState lifecycleState = ContextLifecycleState::Unbound;
    if (!registry_get_lifecycle_state(transaction.snapshot.deviceContextKey, lifecycleState)) {
        global_state().lifecycleStageRejects.fetch_add(1, std::memory_order_relaxed);
        trace_lifecycle_stage_decision(transaction, lifecycleState, stage, false, "missing_registry_entry");
        if (outError) {
            *outError = "missing registry entry for lifecycle validation";
        }
        return false;
    }
    if (!lifecycle_state_allowed_for_stage(lifecycleState, allowNonActiveRelease)) {
        global_state().lifecycleStageRejects.fetch_add(1, std::memory_order_relaxed);
        trace_lifecycle_stage_decision(transaction, lifecycleState, stage, false, "lifecycle_state_not_allowed");
        if (outError) {
            *outError = std::string("lifecycle state not allowed for stage (state=")
                + to_cstr(lifecycleState) + ")";
        }
        return false;
    }
    if (JTRACE_ENABLED(3)) {
        trace_lifecycle_stage_decision(transaction, lifecycleState, stage, true, "stage_allowed");
    }
    return true;
}

bool ensure_active_for_command(
    const SubmissionTransaction& transaction,
    std::string& outError,
    const char* commandName) {
    if (!validate_lifecycle_for_stage(transaction, commandName ? commandName : "command", false, &outError)) {
        telemetry_record_module_boundary_violation();
        telemetry_trace_module_boundary_violation(
            transaction.transactionId,
            transaction.snapshot.snapshotId,
            transaction.snapshot.traceSchemaVersion,
            commandName ? commandName : "command_requires_active_submission");
        return false;
    }

    ResourceManagerState& state = global_state();
    StaleInput staleInput{};
    staleInput.expectedRegistryGeneration = transaction.snapshot.registryGeneration;
    staleInput.observedRegistryGeneration = state.registryGeneration.load(std::memory_order_relaxed);
    staleInput.expectedContextEpoch = transaction.snapshot.contextEpoch;
    staleInput.observedContextEpoch = state.contextEpoch.load(std::memory_order_relaxed);
    staleInput.expectedLeaseGeneration = transaction.leaseGeneration;
    staleInput.observedLeaseGeneration = transaction.active ? transaction.leaseGeneration : 0;
    staleInput.keySchemaMismatch = (transaction.snapshot.keySchemaVersion == 0);

    const StaleDecision staleDecision = classify_stale_path(staleInput);
    telemetry_trace_stale_decision(
        transaction.transactionId,
        transaction.snapshot.snapshotId,
        transaction.snapshot.traceSchemaVersion,
        commandName ? commandName : "command",
        staleInput,
        staleDecision);
    if (!staleDecision.hardStale && !staleDecision.hardMiss) {
        return true;
    }
    telemetry_record_stale_tuple_hard_reject();
    outError = std::string("stale transaction in command path (reason=") +
        to_cstr(staleDecision.reason) + ")";
    telemetry_record_module_boundary_violation();
    telemetry_trace_module_boundary_violation(
        transaction.transactionId,
        transaction.snapshot.snapshotId,
        transaction.snapshot.traceSchemaVersion,
        commandName ? commandName : "command_requires_active_submission");
    return false;
}

} // namespace

bool query_submission_active(const SubmissionTransaction& transaction) noexcept {
    return transaction.active;
}

bool begin_submission(
    SubmissionTransaction& outTransaction,
    const SubmissionSnapshot& snapshot,
    std::string& outError) {
    outError.clear();
    MetadataMutationGuard mutationGuard("begin_submission");
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
    outTransaction.snapshot.traceSchemaVersion = std::max<std::uint32_t>(1u, outTransaction.snapshot.traceSchemaVersion);
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
    MetadataMutationGuard mutationGuard("acquire_plan");
    if (!mutationGuard.ok()) {
        outError = "metadata mutation guard rejected acquire_plan";
        return false;
    }
    const std::uint64_t acquireId = telemetry_next_acquire_attempt_id();

    if (!validate_lifecycle_for_stage(transaction, "acquire", false, &outError)) {
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

    {
        ResourceManagerState& state = global_state();
        StaleInput staleInput{};
        staleInput.expectedRegistryGeneration = transaction.snapshot.registryGeneration;
        staleInput.observedRegistryGeneration = state.registryGeneration.load(std::memory_order_relaxed);
        staleInput.expectedContextEpoch = transaction.snapshot.contextEpoch;
        staleInput.observedContextEpoch = state.contextEpoch.load(std::memory_order_relaxed);
        staleInput.expectedLeaseGeneration = transaction.leaseGeneration;
        staleInput.observedLeaseGeneration = transaction.active ? transaction.leaseGeneration : 0;
        staleInput.keySchemaMismatch = (transaction.snapshot.keySchemaVersion == 0);
        const StaleDecision staleDecision = classify_stale_path(staleInput);
        telemetry_trace_stale_decision(
            transaction.transactionId,
            transaction.snapshot.snapshotId,
            transaction.snapshot.traceSchemaVersion,
            "acquire",
            staleInput,
            staleDecision);
        if (staleDecision.hardStale || staleDecision.hardMiss) {
            telemetry_record_stale_tuple_hard_reject();
            outError = std::string("stale transaction in acquire path (reason=") +
                to_cstr(staleDecision.reason) + ")";
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
    }

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
    MetadataMutationGuard mutationGuard("commit_submission");
    if (!mutationGuard.ok()) {
        outError = "metadata mutation guard rejected commit_submission";
        return false;
    }
    if (!validate_lifecycle_for_stage(transaction, "commit", false, &outError)) {
        return false;
    }
    {
        ResourceManagerState& state = global_state();
        StaleInput staleInput{};
        staleInput.expectedRegistryGeneration = transaction.snapshot.registryGeneration;
        staleInput.observedRegistryGeneration = state.registryGeneration.load(std::memory_order_relaxed);
        staleInput.expectedContextEpoch = transaction.snapshot.contextEpoch;
        staleInput.observedContextEpoch = state.contextEpoch.load(std::memory_order_relaxed);
        staleInput.expectedLeaseGeneration = transaction.leaseGeneration;
        staleInput.observedLeaseGeneration = transaction.active ? transaction.leaseGeneration : 0;
        staleInput.keySchemaMismatch = (transaction.snapshot.keySchemaVersion == 0);
        const StaleDecision staleDecision = classify_stale_path(staleInput);
        telemetry_trace_stale_decision(
            transaction.transactionId,
            transaction.snapshot.snapshotId,
            transaction.snapshot.traceSchemaVersion,
            "commit",
            staleInput,
            staleDecision);
        if (staleDecision.hardStale || staleDecision.hardMiss) {
            telemetry_record_stale_tuple_hard_reject();
            outError = std::string("stale transaction in commit path (reason=") +
                to_cstr(staleDecision.reason) + ")";
            return false;
        }
    }
    if (!transaction.active) {
        outError = "submission transaction is not active";
        return false;
    }
    transaction.committed = true;
    transaction.active = false;
    telemetry_record_commit_submission();
    return true;
}

bool command_freeze_drain_bump_resume(
    const DeviceContextKey& key,
    const char* reason,
    std::string& outError) {
    outError.clear();
    MetadataMutationGuard mutationGuard("command_freeze_drain_bump_resume");
    if (!mutationGuard.ok()) {
        outError = "metadata mutation guard rejected command_freeze_drain_bump_resume";
        return false;
    }
    if (!registry_freeze_drain_bump_resume(key, reason)) {
        outError = "freeze-drain-bump-resume barrier rejected";
        return false;
    }
    return true;
}

namespace {
bool command_retire_context_with_reason(
    const DeviceContextKey& key,
    RegistryRetireReason reason,
    const char* commandName,
    std::string& outError) {
    outError.clear();
    MetadataMutationGuard mutationGuard(commandName ? commandName : "command_retire_context");
    if (!mutationGuard.ok()) {
        outError = std::string("metadata mutation guard rejected ")
            + (commandName ? commandName : "command_retire_context");
        return false;
    }

    RegistryHandle handle{};
    if (!registry_get(key, handle) || handle.value == 0) {
        return true;
    }

    registry_retire(handle, reason);
    return true;
}
} // namespace

bool command_retire_context_reset(
    const DeviceContextKey& key,
    std::string& outError) {
    return command_retire_context_with_reason(
        key,
        RegistryRetireReason::ContextReset,
        "command_retire_context_reset",
        outError);
}

bool command_retire_context_idle(
    const DeviceContextKey& key,
    std::string& outError) {
    return command_retire_context_with_reason(
        key,
        RegistryRetireReason::Idle,
        "command_retire_context_idle",
        outError);
}

bool command_ensure_uploaded(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_uploaded")) {
        return false;
    }
    return JuicerCuda::ensure_uploaded(resources, ws, cudaStreamOpaque, outError);
}

bool command_ensure_scan_lut(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    bool negativeMedium,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_scan_lut")) {
        return false;
    }
    return JuicerCuda::ensure_scan_lut(resources, ws, negativeMedium, cudaStreamOpaque, outError);
}

bool command_ensure_scan_error_flag(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_scan_error_flag")) {
        return false;
    }
    return JuicerCuda::ensure_scan_error_flag(resources, cudaStreamOpaque, outError);
}

bool command_ensure_print_illuminant_filtered(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    const Print::Runtime& prt,
    const Print::Params& params,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_print_illuminant_filtered")) {
        return false;
    }
    return JuicerCuda::ensure_print_illuminant_filtered(resources, ws, prt, params, cudaStreamOpaque, outError);
}

bool command_ensure_optics_scratch(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    int width,
    int height,
    bool needBlurredScratch,
    bool needAuxScratch,
    bool needGrainScratch,
    bool needGrainSharedScratch,
    bool needGateMask,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_optics_scratch")) {
        return false;
    }
    return JuicerCuda::ensure_optics_scratch(
        resources,
        width,
        height,
        needBlurredScratch,
        needAuxScratch,
        needGrainScratch,
        needGrainSharedScratch,
        needGateMask,
        cudaStreamOpaque,
        outError);
}

bool command_ensure_spatial_dir_scratch(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    int width,
    int height,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_spatial_dir_scratch")) {
        return false;
    }
    return JuicerCuda::ensure_spatial_dir_scratch(resources, width, height, cudaStreamOpaque, outError);
}

bool command_ensure_spatial_dir_kernel(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    JuicerCuda::Resources::DeviceGaussianKernel& kernel,
    float sigma,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_spatial_dir_kernel")) {
        return false;
    }
    return JuicerCuda::ensure_spatial_dir_kernel(resources, kernel, sigma, cudaStreamOpaque, outError);
}

bool command_ensure_gaussian_kernel(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    JuicerCuda::Resources::DeviceGaussianKernel& kernel,
    float sigma,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_gaussian_kernel")) {
        return false;
    }
    return JuicerCuda::ensure_gaussian_kernel(resources, kernel, sigma, cudaStreamOpaque, outError);
}

bool command_ensure_halation_kernel(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    JuicerCuda::Resources::DeviceGaussianKernel& kernel,
    float sigma,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_halation_kernel")) {
        return false;
    }
    return JuicerCuda::ensure_halation_kernel(resources, kernel, sigma, cudaStreamOpaque, outError);
}

void rollback_submission(
    SubmissionTransaction& transaction,
    const char* reason) noexcept {
    (void)reason;
    MetadataMutationGuard mutationGuard("rollback_submission");
    if (!mutationGuard.ok()) {
        return;
    }
    (void)validate_lifecycle_for_stage(transaction, "release", true, nullptr);
    ResourceManagerState& state = global_state();
    StaleInput staleInput{};
    staleInput.expectedRegistryGeneration = transaction.snapshot.registryGeneration;
    staleInput.observedRegistryGeneration = state.registryGeneration.load(std::memory_order_relaxed);
    staleInput.expectedContextEpoch = transaction.snapshot.contextEpoch;
    staleInput.observedContextEpoch = state.contextEpoch.load(std::memory_order_relaxed);
    staleInput.expectedLeaseGeneration = transaction.leaseGeneration;
    staleInput.observedLeaseGeneration = transaction.leaseGeneration;
    staleInput.keySchemaMismatch = (transaction.snapshot.keySchemaVersion == 0);
    const StaleDecision staleDecision = classify_stale_path(staleInput);
    telemetry_trace_stale_decision(
        transaction.transactionId,
        transaction.snapshot.snapshotId,
        transaction.snapshot.traceSchemaVersion,
        "release",
        staleInput,
        staleDecision);
    if (staleDecision.hardStale || staleDecision.hardMiss) {
        telemetry_record_stale_tuple_hard_reject();
    }
    if (!transaction.active) {
        return;
    }
    transaction.committed = false;
    transaction.active = false;
    telemetry_record_rollback_submission();
}

} // namespace ResourceManager
} // namespace JuicerCuda
