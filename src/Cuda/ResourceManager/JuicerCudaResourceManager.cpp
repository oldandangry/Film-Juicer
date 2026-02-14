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
#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>
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

bool key_digests_equal(const KeyDigests& lhs, const KeyDigests& rhs) noexcept {
    return lhs.uploadCoreHash == rhs.uploadCoreHash &&
        lhs.dirHash == rhs.dirHash &&
        lhs.scannerHash == rhs.scannerHash &&
        lhs.autoExposureHash == rhs.autoExposureHash;
}

ResourcePlan make_uniform_resource_plan(AcquireStatus status) noexcept {
    ResourcePlan plan{};
    for (ResourceKind kind : kResourceKindOrder) {
        ResourcePlanEntry& entry = resource_plan_entry(plan, kind);
        entry.acquire.status = status;
        entry.acquire.shouldBuild = (status == AcquireStatus::Miss);
        entry.invalidated = false;
    }
    return plan;
}

AcquireStatus combine_status(const ResourcePlan& plan) noexcept {
    bool sawMiss = false;
    bool sawBusy = false;
    bool sawExhausted = false;
    for (ResourceKind kind : kResourceKindOrder) {
        const AcquireStatus status = resource_plan_entry(plan, kind).acquire.status;
        if (status == AcquireStatus::Error) {
            return AcquireStatus::Error;
        }
        if (status == AcquireStatus::Exhausted) {
            sawExhausted = true;
        }
        else if (status == AcquireStatus::Busy) {
            sawBusy = true;
        }
        else if (status == AcquireStatus::Miss) {
            sawMiss = true;
        }
    }
    if (sawExhausted) {
        return AcquireStatus::Exhausted;
    }
    if (sawBusy) {
        return AcquireStatus::Busy;
    }
    if (sawMiss) {
        return AcquireStatus::Miss;
    }
    return AcquireStatus::Hit;
}

bool validate_resource_kind_onboarding_contract(std::string& outError) noexcept {
    if (!resource_kind_contract_is_valid()) {
        outError = "resource kind onboarding contract invalid";
        return false;
    }
    for (ResourceKind kind : kResourceKindOrder) {
        const ResourceKindContractEntry& entry = resource_kind_contract_entry(kind);
        if (entry.kind != kind ||
            entry.keyField == nullptr ||
            entry.invalidationLane == nullptr ||
            entry.resourceNode == nullptr ||
            entry.acquireStatusField == nullptr ||
            entry.telemetryTag == nullptr) {
            outError = "resource kind onboarding entry missing required fields";
            return false;
        }
    }
    return true;
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

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
struct BaseGraphKey {
    int width = 0;
    int height = 0;
    int nComponents = 0;
    int renderMode = 0;
};

struct BaseGraphEntry {
    BaseGraphKey key{};
    void* graphOpaque = nullptr;
    void* execOpaque = nullptr;
    void* kernelNodeOpaque = nullptr;
    void* kernelFuncOpaque = nullptr;
    unsigned int gridX = 0;
    unsigned int gridY = 0;
    unsigned int gridZ = 0;
    unsigned int blockX = 0;
    unsigned int blockY = 0;
    unsigned int blockZ = 0;
    unsigned int sharedMemBytes = 0;
    std::uint64_t lastUseTick = 0;
};

struct BaseGraphBucketState {
    std::mutex mutex;
    std::uint64_t contextEpoch = 0;
    std::uint64_t useTick = 0;
    std::vector<BaseGraphEntry> entries;
};

struct BaseGraphCacheState {
    std::mutex mutex;
    std::unordered_map<DeviceContextKey, std::shared_ptr<BaseGraphBucketState>, DeviceContextKeyHash> byContext;
};

BaseGraphCacheState& base_graph_cache_state() noexcept {
    static BaseGraphCacheState state{};
    return state;
}

using BasePipelineLaunchFn = cudaError_t(*)(const JuicerCuda::PipelineRunParams*, void*);

BasePipelineLaunchFn base_pipeline_launch_fn_for_mode(int renderModeKey) noexcept {
    switch (renderModeKey) {
    case 0:
        return juicer_cuda_negative_pipeline;
    case 1:
        return juicer_cuda_print_pipeline;
    default:
        return nullptr;
    }
}

bool base_graph_key_equal(const BaseGraphKey& a, const BaseGraphKey& b) noexcept {
    return a.width == b.width &&
        a.height == b.height &&
        a.nComponents == b.nComponents &&
        a.renderMode == b.renderMode;
}

void destroy_base_graph_entry(BaseGraphEntry& entry) noexcept {
    if (entry.execOpaque) {
        cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(entry.execOpaque));
        entry.execOpaque = nullptr;
    }
    if (entry.graphOpaque) {
        cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(entry.graphOpaque));
        entry.graphOpaque = nullptr;
    }
    entry.kernelNodeOpaque = nullptr;
    entry.kernelFuncOpaque = nullptr;
    entry.gridX = 0;
    entry.gridY = 0;
    entry.gridZ = 0;
    entry.blockX = 0;
    entry.blockY = 0;
    entry.blockZ = 0;
    entry.sharedMemBytes = 0;
    entry.lastUseTick = 0;
}

std::shared_ptr<BaseGraphBucketState> get_or_create_base_graph_bucket(
    const DeviceContextKey& key) noexcept {
    BaseGraphCacheState& state = base_graph_cache_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.byContext.find(key);
    if (it != state.byContext.end() && it->second) {
        return it->second;
    }
    auto bucket = std::make_shared<BaseGraphBucketState>();
    state.byContext[key] = bucket;
    return bucket;
}

void clear_base_graph_bucket(BaseGraphBucketState& bucket) noexcept {
    for (auto& entry : bucket.entries) {
        destroy_base_graph_entry(entry);
    }
    bucket.entries.clear();
    bucket.useTick = 0;
}

void retire_base_graph_cache_for_context(const DeviceContextKey& key) noexcept {
    BaseGraphCacheState& state = base_graph_cache_state();
    std::shared_ptr<BaseGraphBucketState> bucket;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        auto it = state.byContext.find(key);
        if (it == state.byContext.end()) {
            return;
        }
        bucket = it->second;
        state.byContext.erase(it);
    }
    if (bucket) {
        std::lock_guard<std::mutex> lock(bucket->mutex);
        clear_base_graph_bucket(*bucket);
    }
}

BaseGraphEntry* find_base_graph_entry(
    BaseGraphBucketState& bucket,
    const BaseGraphKey& key) noexcept {
    for (auto& entry : bucket.entries) {
        if (base_graph_key_equal(entry.key, key) &&
            entry.execOpaque &&
            entry.graphOpaque &&
            entry.kernelNodeOpaque) {
            return &entry;
        }
    }
    return nullptr;
}

BaseGraphEntry* build_base_graph_entry(
    BaseGraphBucketState& bucket,
    const BaseGraphKey& key,
    BasePipelineLaunchFn launchFn,
    JuicerCuda::PipelineRunParams& run,
    cudaStream_t stream) noexcept {
    if (!launchFn) {
        return nullptr;
    }

    constexpr std::size_t kBaseGraphCap = 4;
    if (bucket.entries.size() >= kBaseGraphCap && !bucket.entries.empty()) {
        std::size_t victim = 0;
        std::uint64_t bestTick = bucket.entries[0].lastUseTick;
        for (std::size_t i = 1; i < bucket.entries.size(); ++i) {
            if (bucket.entries[i].lastUseTick < bestTick) {
                bestTick = bucket.entries[i].lastUseTick;
                victim = i;
            }
        }
        destroy_base_graph_entry(bucket.entries[victim]);
        bucket.entries.erase(bucket.entries.begin() + static_cast<std::ptrdiff_t>(victim));
    }

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaGraphNode_t kernelNode = nullptr;

    cudaError_t capErr = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    if (capErr != cudaSuccess) {
        return nullptr;
    }

    cudaError_t launchErr = launchFn(&run, reinterpret_cast<void*>(stream));
    if (launchErr != cudaSuccess) {
        cudaGraph_t abortGraph = nullptr;
        cudaStreamEndCapture(stream, &abortGraph);
        if (abortGraph) {
            cudaGraphDestroy(abortGraph);
        }
        return nullptr;
    }

    capErr = cudaStreamEndCapture(stream, &graph);
    if (capErr != cudaSuccess || !graph) {
        if (graph) {
            cudaGraphDestroy(graph);
        }
        return nullptr;
    }

    cudaError_t instErr = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    if (instErr != cudaSuccess || !exec) {
        cudaGraphDestroy(graph);
        return nullptr;
    }

    std::size_t nodeCount = 0;
    cudaError_t nodeErr = cudaGraphGetNodes(graph, nullptr, &nodeCount);
    if (nodeErr == cudaSuccess && nodeCount > 0) {
        std::vector<cudaGraphNode_t> nodes(nodeCount);
        nodeErr = cudaGraphGetNodes(graph, nodes.data(), &nodeCount);
        if (nodeErr == cudaSuccess) {
            for (cudaGraphNode_t node : nodes) {
                cudaGraphNodeType nodeType = cudaGraphNodeTypeEmpty;
                if (cudaGraphNodeGetType(node, &nodeType) == cudaSuccess && nodeType == cudaGraphNodeTypeKernel) {
                    kernelNode = node;
                    break;
                }
            }
        }
    }

    if (!kernelNode) {
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        return nullptr;
    }

    cudaKernelNodeParams baseParams{};
    if (cudaGraphKernelNodeGetParams(kernelNode, &baseParams) != cudaSuccess || !baseParams.func) {
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        return nullptr;
    }

    BaseGraphEntry entry{};
    entry.key = key;
    entry.graphOpaque = reinterpret_cast<void*>(graph);
    entry.execOpaque = reinterpret_cast<void*>(exec);
    entry.kernelNodeOpaque = reinterpret_cast<void*>(kernelNode);
    entry.kernelFuncOpaque = baseParams.func;
    entry.gridX = baseParams.gridDim.x;
    entry.gridY = baseParams.gridDim.y;
    entry.gridZ = baseParams.gridDim.z;
    entry.blockX = baseParams.blockDim.x;
    entry.blockY = baseParams.blockDim.y;
    entry.blockZ = baseParams.blockDim.z;
    entry.sharedMemBytes = baseParams.sharedMemBytes;
    entry.lastUseTick = bucket.useTick;
    bucket.entries.push_back(entry);
    return &bucket.entries.back();
}
#endif

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
            const ResourcePlan errorPlan = make_uniform_resource_plan(AcquireStatus::Error);
            telemetry_record_stale_tuple_hard_reject();
            outError = std::string("stale transaction in acquire path (reason=") +
                to_cstr(staleDecision.reason) + ")";
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

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    retire_base_graph_cache_for_context(key);
#endif

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

bool command_ensure_auto_exposure_buffers(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    int meterWidth,
    int meterHeight,
    std::uint64_t autoExposureKeyHash,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_auto_exposure_buffers")) {
        return false;
    }
    const std::uint64_t normalizedKeyHash = normalize_key_u64(autoExposureKeyHash);

    const ShadowHistoryKey ownershipKey{
        transaction.snapshot.instanceToken.value,
        transaction.snapshot.deviceContextKey
    };

    bool hadPrevious = false;
    bool metadataHit = false;
    {
        AutoExposureOwnershipState& ownershipState = auto_exposure_ownership_state();
        std::lock_guard<std::mutex> lock(ownershipState.mutex);
        auto it = ownershipState.bySubmissionKey.find(ownershipKey);
        if (it != ownershipState.bySubmissionKey.end() && it->second.valid) {
            hadPrevious = true;
            const AutoExposureOwnershipEntry& previous = it->second;
            metadataHit =
                previous.keySchemaVersion == transaction.snapshot.keySchemaVersion &&
                previous.keyHash == normalizedKeyHash &&
                previous.meterWidth == meterWidth &&
                previous.meterHeight == meterHeight;
        }
    }

    telemetry_trace_auto_exposure_ownership(
        transaction.transactionId,
        transaction.snapshot.snapshotId,
        transaction.snapshot.traceSchemaVersion,
        "ManagerOnly",
        "acquire",
        metadataHit,
        normalizedKeyHash,
        meterWidth,
        meterHeight,
        hadPrevious,
        metadataHit ? "metadata_hit" : "metadata_miss");

    if (!JuicerCuda::ensure_auto_exposure_buffers(resources, meterWidth, meterHeight, cudaStreamOpaque, outError)) {
        telemetry_trace_auto_exposure_ownership(
            transaction.transactionId,
            transaction.snapshot.snapshotId,
            transaction.snapshot.traceSchemaVersion,
            "ManagerOnly",
            "ensure_fail",
            false,
            normalizedKeyHash,
            meterWidth,
            meterHeight,
            hadPrevious,
            outError.empty() ? "ensure_failed" : outError.c_str());
        return false;
    }

    {
        AutoExposureOwnershipState& ownershipState = auto_exposure_ownership_state();
        std::lock_guard<std::mutex> lock(ownershipState.mutex);
        AutoExposureOwnershipEntry& entry = ownershipState.bySubmissionKey[ownershipKey];
        entry.valid = true;
        entry.keyHash = normalizedKeyHash;
        entry.meterWidth = meterWidth;
        entry.meterHeight = meterHeight;
        entry.keySchemaVersion = transaction.snapshot.keySchemaVersion;
    }

    telemetry_trace_auto_exposure_ownership(
        transaction.transactionId,
        transaction.snapshot.snapshotId,
        transaction.snapshot.traceSchemaVersion,
        "ManagerOnly",
        "publish",
        metadataHit,
        normalizedKeyHash,
        meterWidth,
        meterHeight,
        hadPrevious,
        metadataHit ? "reuse" : "refresh");
    return true;
}

bool command_launch_base_pipeline_graph(
    SubmissionTransaction& transaction,
    JuicerCuda::PipelineRunParams& run,
    int renderModeKey,
    void* cudaStreamOpaque,
    int& outCudaErrorCode,
    std::string& outError) {
    outError.clear();

    if (!ensure_active_for_command(transaction, outError, "command_launch_base_pipeline_graph")) {
        return false;
    }

#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
    (void)run;
    (void)renderModeKey;
    (void)cudaStreamOpaque;
    outCudaErrorCode = 0;
    outError = "CUDA is not enabled";
    return false;
#else
    outCudaErrorCode = static_cast<int>(cudaErrorUnknown);
    BasePipelineLaunchFn launchFn = base_pipeline_launch_fn_for_mode(renderModeKey);
    if (!launchFn) {
        outCudaErrorCode = static_cast<int>(cudaErrorInvalidValue);
        return true;
    }

    const cudaStream_t stream = cudaStreamOpaque
        ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque)
        : nullptr;

    BaseGraphKey key{};
    key.width = run.width;
    key.height = run.height;
    key.nComponents = run.nComponents;
    key.renderMode = renderModeKey;

    std::shared_ptr<BaseGraphBucketState> bucketPtr =
        get_or_create_base_graph_bucket(transaction.snapshot.deviceContextKey);
    if (!bucketPtr) {
        outCudaErrorCode = static_cast<int>(cudaErrorUnknown);
        outError = "base graph bucket unavailable";
        return false;
    }
    std::lock_guard<std::mutex> bucketLock(bucketPtr->mutex);
    BaseGraphBucketState& bucket = *bucketPtr;
    std::uint64_t activeEpoch = transaction.snapshot.contextEpoch;
    if (activeEpoch == 0) {
        activeEpoch = 1;
    }
    if (bucket.contextEpoch != activeEpoch) {
        clear_base_graph_bucket(bucket);
        bucket.contextEpoch = activeEpoch;
    }

    bucket.useTick++;
    if (bucket.useTick == 0) {
        bucket.useTick = 1;
    }
    const std::uint64_t useTick = bucket.useTick;

    BaseGraphEntry* found = find_base_graph_entry(bucket, key);
    if (!found) {
        found = build_base_graph_entry(bucket, key, launchFn, run, stream);
    }

    if (!found || !found->execOpaque || !found->kernelNodeOpaque) {
        outCudaErrorCode = static_cast<int>(launchFn(&run, reinterpret_cast<void*>(stream)));
        return true;
    }

    found->lastUseTick = useTick;

    cudaGraphExec_t exec = reinterpret_cast<cudaGraphExec_t>(found->execOpaque);
    cudaGraphNode_t node = reinterpret_cast<cudaGraphNode_t>(found->kernelNodeOpaque);

    cudaKernelNodeParams nodeParams{};
    nodeParams.func = found->kernelFuncOpaque;
    nodeParams.gridDim = dim3(found->gridX, found->gridY, found->gridZ);
    nodeParams.blockDim = dim3(found->blockX, found->blockY, found->blockZ);
    nodeParams.sharedMemBytes = found->sharedMemBytes;
    void* kernelArgs[] = { &run };
    nodeParams.kernelParams = kernelArgs;
    nodeParams.extra = nullptr;
    cudaError_t setErr = cudaGraphExecKernelNodeSetParams(exec, node, &nodeParams);
    if (setErr != cudaSuccess) {
        destroy_base_graph_entry(*found);
        outCudaErrorCode = static_cast<int>(launchFn(&run, reinterpret_cast<void*>(stream)));
        return true;
    }

    cudaError_t runErr = cudaGraphLaunch(exec, stream);
    if (runErr == cudaSuccess) {
        runErr = cudaGetLastError();
    }
    if (runErr != cudaSuccess) {
        destroy_base_graph_entry(*found);
        outCudaErrorCode = static_cast<int>(launchFn(&run, reinterpret_cast<void*>(stream)));
        return true;
    }

    outCudaErrorCode = static_cast<int>(cudaSuccess);
    return true;
#endif
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
