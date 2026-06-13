// Cuda/ResourceManager/JuicerCudaResourceManager.cpp

#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"

#include "Cuda/JuicerCudaResources.h"
#include "Cuda/ResourceManager/JuicerCudaResourceCore.h"
#include "Hash.h"
#include "Logging.h"
#include "Print.h"
#include "JuicerState.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

namespace LaunchGraphCounters {

bool compiled_enabled() noexcept {
#if JUICER_LAUNCH_GRAPH_COUNTERS_COMPILED
    return true;
#else
    return false;
#endif
}

#if JUICER_LAUNCH_GRAPH_COUNTERS_COMPILED
namespace {

struct RuntimeConfig {
    bool runtimeEnabled = false;
    std::filesystem::path snapshotPath{};
};

int parse_env_int(const char* name, int fallback) noexcept {
    if (!name) {
        return fallback;
    }
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return fallback;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value) {
        return fallback;
    }
    return static_cast<int>(parsed);
}

const RuntimeConfig& runtime_config() noexcept {
    static const RuntimeConfig config = []() {
        RuntimeConfig out{};
        out.runtimeEnabled = parse_env_int("JUICER_LAUNCH_GRAPH_COUNTERS", 0) != 0;
        const char* snapshotPathEnv = std::getenv("JUICER_LAUNCH_GRAPH_COUNTERS_PATH");
        if (out.runtimeEnabled && snapshotPathEnv && *snapshotPathEnv) {
            try {
                out.snapshotPath = std::filesystem::path(snapshotPathEnv);
                out.snapshotPath.make_preferred();
            }
            catch (...) {
                out.snapshotPath.clear();
            }
        }
        return out;
    }();
    return config;
}

std::atomic<std::uint64_t>& frames_rendered_counter() noexcept {
    static std::atomic<std::uint64_t> counter{ 0 };
    return counter;
}

std::atomic<std::uint64_t>& kernel_launch_total_counter() noexcept {
    static std::atomic<std::uint64_t> counter{ 0 };
    return counter;
}

std::atomic<std::uint64_t>& graph_eligible_submission_total_counter() noexcept {
    static std::atomic<std::uint64_t> counter{ 0 };
    return counter;
}

std::atomic<std::uint64_t>& graph_replay_hit_total_counter() noexcept {
    static std::atomic<std::uint64_t> counter{ 0 };
    return counter;
}

std::mutex& snapshot_write_mutex() noexcept {
    static std::mutex m;
    return m;
}

std::once_flag& snapshot_dir_once_flag() noexcept {
    static std::once_flag flag;
    return flag;
}

void ensure_snapshot_dir_exists_if_needed(const std::filesystem::path& path) noexcept {
    if (path.empty() || !path.has_parent_path()) {
        return;
    }
    std::call_once(snapshot_dir_once_flag(), [&]() {
        try {
            std::error_code ec;
            std::filesystem::create_directories(path.parent_path(), ec);
        }
        catch (...) {
        }
    });
}

void flush_snapshot_file() noexcept {
    const RuntimeConfig& config = runtime_config();
    if (!config.runtimeEnabled || config.snapshotPath.empty()) {
        return;
    }

    try {
        // Keep the snapshot write best-effort so telemetry never interferes with rendering.
        ensure_snapshot_dir_exists_if_needed(config.snapshotPath);
        const Snapshot current = snapshot();

        std::lock_guard<std::mutex> lock(snapshot_write_mutex());
        std::ofstream out(config.snapshotPath, std::ios::out | std::ios::trunc);
        if (!out.is_open()) {
            return;
        }
        out.setf(std::ios::fixed, std::ios::floatfield);
        out.precision(6);
        out << "format_version=1\n";
        out << "compiled=1\n";
        out << "runtime_enabled=1\n";
        out << "frames_rendered=" << current.framesRendered << "\n";
        out << "kernel_launch_total=" << current.kernelLaunchTotal << "\n";
        out << "graph_eligible_submission_total=" << current.graphEligibleSubmissionTotal << "\n";
        out << "graph_replay_hit_total=" << current.graphReplayHitTotal << "\n";
        out << "kernel_launches_per_frame=" << kernel_launches_per_frame(current) << "\n";
        out << "cuda_graph_replay_hit_rate=" << cuda_graph_replay_hit_rate(current) << "\n";
    }
    catch (...) {
    }
}

} // namespace
#endif

Snapshot snapshot() noexcept {
#if JUICER_LAUNCH_GRAPH_COUNTERS_COMPILED
    Snapshot out{};
    out.framesRendered = frames_rendered_counter().load(std::memory_order_relaxed);
    out.kernelLaunchTotal = kernel_launch_total_counter().load(std::memory_order_relaxed);
    out.graphEligibleSubmissionTotal =
        graph_eligible_submission_total_counter().load(std::memory_order_relaxed);
    out.graphReplayHitTotal =
        graph_replay_hit_total_counter().load(std::memory_order_relaxed);
    return out;
#else
    return {};
#endif
}

double kernel_launches_per_frame(const Snapshot& snapshot) noexcept {
    if (snapshot.framesRendered == 0) {
        return 0.0;
    }
    return static_cast<double>(snapshot.kernelLaunchTotal) /
        static_cast<double>(snapshot.framesRendered);
}

double cuda_graph_replay_hit_rate(const Snapshot& snapshot) noexcept {
    if (snapshot.graphEligibleSubmissionTotal == 0) {
        return 0.0;
    }
    return (100.0 * static_cast<double>(snapshot.graphReplayHitTotal)) /
        static_cast<double>(snapshot.graphEligibleSubmissionTotal);
}

bool runtime_enabled() noexcept {
#if JUICER_LAUNCH_GRAPH_COUNTERS_COMPILED
    return runtime_config().runtimeEnabled;
#else
    return false;
#endif
}

void record_kernel_launch(std::uint64_t delta) noexcept {
#if JUICER_LAUNCH_GRAPH_COUNTERS_COMPILED
    if (delta == 0 || !runtime_enabled()) {
        return;
    }
    kernel_launch_total_counter().fetch_add(delta, std::memory_order_relaxed);
#else
    (void)delta;
#endif
}

void record_graph_eligible_submission(std::uint64_t delta) noexcept {
#if JUICER_LAUNCH_GRAPH_COUNTERS_COMPILED
    if (delta == 0 || !runtime_enabled()) {
        return;
    }
    graph_eligible_submission_total_counter().fetch_add(delta, std::memory_order_relaxed);
#else
    (void)delta;
#endif
}

void record_graph_replay_hit(std::uint64_t delta) noexcept {
#if JUICER_LAUNCH_GRAPH_COUNTERS_COMPILED
    if (delta == 0 || !runtime_enabled()) {
        return;
    }
    graph_replay_hit_total_counter().fetch_add(delta, std::memory_order_relaxed);
#else
    (void)delta;
#endif
}

void record_frame_completed(std::uint64_t delta) noexcept {
#if JUICER_LAUNCH_GRAPH_COUNTERS_COMPILED
    if (delta == 0 || !runtime_enabled()) {
        return;
    }
    frames_rendered_counter().fetch_add(delta, std::memory_order_relaxed);
    flush_snapshot_file();
#else
    (void)delta;
#endif
}

} // namespace LaunchGraphCounters

// Internal resource-acquire helpers are intentionally consumed only by command wrappers
// in this module; they are not part of the public JuicerCudaResources API surface.
bool ensure_uploaded(
    Resources& resources,
    const WorkingState& ws,
    bool includeCurrentMediumUploads,
    bool negativeMedium,
    void* cudaStreamOpaque,
    std::string& outError);
bool ensure_scan_lut(Resources& resources, const WorkingState& ws, bool negativeMedium, void* cudaStreamOpaque, std::string& outError);
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
bool validate_resource_owner_locked(Resources& resources, std::string& outError, bool bindIfUnset);

namespace ResourceManager {

// Cross-split helper declarations stay local to the owning TU rather than
// living behind a one-consumer internal header.
std::uint64_t estimate_graph_cache_active_bytes_for_context(const DeviceContextKey& key) noexcept;
std::uint64_t evict_noncritical_graph_entries_for_context(const DeviceContextKey& key) noexcept;
std::uint64_t tier_target_bytes(
    const ResourceManagerConfigEffective& cfg,
    const ResolvedPressurePolicy& policy,
    ResourceTier tier) noexcept;
std::uint64_t pressure_total_bytes(const PressureInput& input) noexcept;
bool pressure_policy_enabled(const ResolvedPressurePolicy& policy) noexcept;

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
    std::uint32_t keySchemaVersion = kSubmissionKeySchemaVersion;
    std::uint64_t snapshotId = 0;
};

struct ShadowHistoryState {
    std::mutex mutex;
    std::unordered_map<ShadowHistoryKey, ShadowHistoryEntry, ShadowHistoryKeyHasher> bySubmissionKey;
};

ShadowHistoryState& shadow_history_state() {
    static ShadowHistoryState state{};
    return state;
}

struct AutoExposureOwnershipEntry {
    bool valid = false;
    std::uint64_t keyHash = 0;
    int meterWidth = 0;
    int meterHeight = 0;
    std::uint32_t keySchemaVersion = kSubmissionKeySchemaVersion;
};

struct AutoExposureOwnershipState {
    std::mutex mutex;
    std::unordered_map<ShadowHistoryKey, AutoExposureOwnershipEntry, ShadowHistoryKeyHasher> bySubmissionKey;
};

AutoExposureOwnershipState& auto_exposure_ownership_state() {
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
    std::uint32_t keySchemaVersion = kSubmissionKeySchemaVersion;
    std::uint64_t snapshotId = 0;
};

struct FrameSnapshotState {
    std::mutex mutex;
    std::unordered_map<FrameSnapshotKey, FrameSnapshotEntry, FrameSnapshotKeyHasher> bySubmissionKey;
};

FrameSnapshotState& frame_snapshot_state() {
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

enum class ScratchCheckpointInvocation : std::uint8_t {
    OuterPhaseCheckpoint = 0,
    TierPretrim = 1
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

#if JUICER_DIAGNOSTICS_COMPILED
const char* to_cstr(ScratchCheckpointInvocation invocation) noexcept {
    switch (invocation) {
    case ScratchCheckpointInvocation::OuterPhaseCheckpoint:
        return "outer_phase_checkpoint";
    case ScratchCheckpointInvocation::TierPretrim:
        return "tier_pretrim";
    default:
        return "unknown";
    }
}
#endif

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

#if JUICER_DIAGNOSTICS_COMPILED
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
#endif

const char* trace_or(const char* value, const char* fallback) noexcept {
    return value ? value : fallback;
}

const char* trace_or_unknown(const char* value) noexcept {
    return value ? value : "unknown";
}

const char* trace_or_unspecified(const char* value) noexcept {
    return value ? value : "unspecified";
}

const char* trace_or_non_empty(const char* value, const char* fallback) noexcept {
    return (value && value[0] != '\0') ? value : fallback;
}

#if JUICER_DIAGNOSTICS_COMPILED
const char* failure_reason_class(const char* token) noexcept {
    const std::string_view value = trace_or_non_empty(token, "");
    if (value.empty()) {
        return nullptr;
    }

    if (value == "auto_no_optional_supported" ||
        value == "capability_fallback_legacy" ||
        value == "requested_async_unsupported") {
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
        value == "fragmentation_recovery_reap_failed" ||
        value == "lifecycle_stage_rejected" ||
        value == "lifecycle_state_not_allowed" ||
        value == "missing_registry_entry" ||
        value == "non_allocator_error" ||
        value == "private_fallback_failed" ||
        value == "reap_failed" ||
        value == "reap_retry_failed" ||
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
        value == "requested_slab_disallowed" ||
        value == "unknown_preference_fallback" ||
        value == "zero_request") {
        return "policy_denied";
    }

    if (value == "accrue_burst_consumed" ||
        value == "allocation_retry_success" ||
        value == "admit_new" ||
        value == "allocator_oom_final" ||
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
        value == "fragmentation_recovery_reap" ||
        value == "fragmentation_recovery_reap_no_progress" ||
        value == "ghost_hit_bypass" ||
        value == "granted" ||
        value == "host_alloc_failed" ||
        value == "no_history" ||
        value == "normal" ||
        value == "pressure_gate_reject" ||
        value == "pressure_pre_growth_reclaim_failed" ||
        value == "pressure_pre_upload_reclaim_failed" ||
        value == "private_fallback_admit" ||
        value == "private_fallback_served" ||
        value == "probation_admit" ||
        value == "probation_critical_override" ||
        value == "probation_defer" ||
        value == "reap_no_progress" ||
        value == "retry_after_reap" ||
        value == "retry_once" ||
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

std::string trace_reason_class_field_if_known(
    const char* fieldName,
    const char* token) {
    const char* normalizedClass = failure_reason_class(token);
    if (!normalizedClass) {
        return {};
    }
    return std::string(" ") + trace_or_non_empty(fieldName, "reason_class") + "=" + normalizedClass;
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
#endif

template <typename Action>
bool with_explicit_cuda_device(int targetDevice, Action&& action) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    if (targetDevice < 0) {
        return false;
    }

    int previousDevice = -1;
    if (cudaGetDevice(&previousDevice) != cudaSuccess) {
        return false;
    }
    if (previousDevice != targetDevice &&
        cudaSetDevice(targetDevice) != cudaSuccess) {
        return false;
    }

    action();

    if (previousDevice != targetDevice &&
        cudaSetDevice(previousDevice) != cudaSuccess) {
        return false;
    }
    return true;
#else
    (void)targetDevice;
    (void)action;
    return false;
#endif
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
    std::condition_variable cv;
    std::uint64_t stateVersion = 0;
    std::unordered_map<DeviceContextKey, ScratchContextState, DeviceContextKeyHash> byContext;
    std::uint64_t totalInFlightBytes = 0;
};

struct ScratchNormalizationNoShedCache {
    bool valid = false;
    std::uint64_t requestDescriptorGeneration = 0;
    std::uint64_t retainedScratchGeneration = 0;
    std::uint64_t scratchTargetBytes = 0;
};

struct ScratchNormalizationContextState {
    ScratchNormalizationNoShedCache noShedCache{};
};

struct ScratchNormalizationState {
    std::mutex mutex;
    std::unordered_map<DeviceContextKey, ScratchNormalizationContextState, DeviceContextKeyHash> byContext;
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
    std::condition_variable cv;
    std::uint64_t stateVersion = 0;
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
    std::condition_variable cv;
    std::uint64_t stateVersion = 0;
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
    int sampledDeviceId = -1;
    bool sampleSuccess = false;
    bool poolTelemetryAvailable = false;
};

struct PressureCheckpoint {
    PressureInput input{};
    HeadroomTelemetry headroom{};
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

struct ScratchEligibilityResult {
    std::array<bool, kScratchPolicyCandidateCount> eligibleNow{};
    std::uint64_t reclaimableLiveBytes = 0;
    bool sharedTmpEligibleNow = false;
    bool anyEligible = false;
};

struct ScratchCheckpointObservation {
    bool requestActive = false;
    bool stage1OverTarget = false;
    bool stage2Evaluated = false;
    bool stage2SkippedByNoShedCache = false;
    bool overTargetButNotReducible = false;
    bool sheddingAttempted = false;
    bool sheddingProgressed = false;
    bool sheddingPartialFailure = false;
    bool sheddingTargetReached = false;
    bool sheddingOrphanedSharedTmpRetired = false;
    std::uint64_t requestDescriptorGeneration = 0;
    std::uint64_t retainedScratchGeneration = 0;
    std::uint64_t scratchTargetBytes = 0;
    std::uint64_t hysteresisBytes = 0;
    std::uint64_t policyLiveRetainedBytes = 0;
    std::uint64_t reclaimableLiveBytes = 0;
    std::uint32_t shedRetiredActionCount = 0;
    std::uint64_t shedRetiredLiveBytes = 0;
    ScratchResidencyView residency{};
    ScratchEligibilityResult eligibility{};
};

struct ScratchSheddingResult {
    bool attempted = false;
    bool progressed = false;
    bool partialFailure = false;
    bool orphanedSharedTmpRetired = false;
    bool targetReached = false;
    std::uint32_t retiredActionCount = 0;
    std::uint64_t retiredLiveBytes = 0;
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
    std::uint64_t stateVersion = 0;
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

bool parse_env_u64(const char* name, std::uint64_t& outValue) noexcept {
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
    outValue = static_cast<std::uint64_t>(parsed);
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
    auto parse_u64_override = [&](const char* envName, std::uint64_t& target) {
        std::uint64_t parsed = 0;
        if (parse_env_u64(envName, parsed)) {
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

    parse_u64_override("JUICER_MANAGER_SOFT_TARGET_BYTES", out.raw.managerSoftTargetBytes);
    parse_u64_override("JUICER_5X_MANAGER_SOFT_TARGET_BYTES", out.raw.managerSoftTargetBytes);
    parse_u64_override("JUICER_MANAGER_RESERVE_BYTES", out.raw.managerReserveBytes);
    parse_u64_override("JUICER_5X_MANAGER_RESERVE_BYTES", out.raw.managerReserveBytes);
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
#if JUICER_DIAGNOSTICS_COMPILED
    if (!overrides.anyOverride || !JTRACE_ENABLED(2)) {
        return;
    }
    const std::string msg = std::string("event=5x_env_overrides")
        + " profile_enabled=" + std::to_string(overrides.profileEnabled ? 1 : 0)
        + " profile_disabled=" + std::to_string(overrides.profileDisabled ? 1 : 0)
        + " manager_soft_target_bytes=" + std::to_string(
            static_cast<unsigned long long>(cfg.managerSoftTargetBytes))
        + " manager_reserve_bytes=" + std::to_string(
            static_cast<unsigned long long>(cfg.managerReserveBytes))
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
#endif
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
    if (lut.res == 0u) {
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
    if (lut.log2XYZ)
        add_snapshot_bytes(snapshot, bytes);
    if (lut.log10XYZ)
        add_snapshot_bytes(snapshot, bytes);
    if (lut.slopeC)
        add_snapshot_bytes(snapshot, bytes);
    if (lut.slopeM)
        add_snapshot_bytes(snapshot, bytes);
    if (lut.slopeY)
        add_snapshot_bytes(snapshot, bytes);

    if (lut.res < 2u || (!lut.cellMin && !lut.cellMax)) {
        return;
    }
    const std::uint64_t cellRes = res - 1u;
    std::uint64_t cellCount = 0;
    if (!mul_u64_checked(cellRes, cellRes, cellCount) ||
        !mul_u64_checked(cellCount, cellRes, cellCount) ||
        !mul_u64_checked(cellCount, 3ull, cellCount)) {
        snapshot.overflow = true;
        add_snapshot_bytes(snapshot, std::numeric_limits<std::uint64_t>::max());
        return;
    }
    bool cellOverflow = false;
    const std::uint64_t cellBytes = bytes_for_count_u64(cellCount, sizeof(double), cellOverflow);
    if (cellOverflow) {
        snapshot.overflow = true;
    }
    if (lut.cellMin)
        add_snapshot_bytes(snapshot, cellBytes);
    if (lut.cellMax)
        add_snapshot_bytes(snapshot, cellBytes);
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
    const std::uint64_t planeBytes =
        (scratch.capacityElements >
         (std::numeric_limits<std::uint64_t>::max() / sizeof(float)))
            ? (overflow = true, 0ull)
            : static_cast<std::uint64_t>(scratch.capacityElements) * sizeof(float);
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
    const std::uint64_t gateBytes =
        (scratch.gateMaskCapacityElements >
         (std::numeric_limits<std::uint64_t>::max() / sizeof(float)))
            ? (overflow = true, 0ull)
            : static_cast<std::uint64_t>(scratch.gateMaskCapacityElements) * sizeof(float);
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
    const std::uint64_t planeBytes =
        (scratch.capacityElements >
         (std::numeric_limits<std::uint64_t>::max() / sizeof(float)))
            ? (overflow = true, 0ull)
            : static_cast<std::uint64_t>(scratch.capacityElements) * sizeof(float);
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

    for (int ch = 0; ch < 3; ++ch) {
        const std::uint64_t layerN = non_negative_u64(resources.densityCurvesLayersChannelN[ch]);
        if (layerN == 0) {
            continue;
        }
        bool overflow = false;
        const std::uint64_t layerBytes = bytes_for_count_u64(layerN, sizeof(float), overflow);
        if (overflow) {
            snapshot.overflow = true;
        }
        for (int layer = 0; layer < 3; ++layer) {
            if (resources.densityCurvesLayers[layer][ch]) {
                add_snapshot_bytes(snapshot, layerBytes);
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
        const std::uint64_t sharedTmpBytes =
            (resources.sharedTmpCapacityElements >
             (std::numeric_limits<std::uint64_t>::max() / sizeof(float)))
                ? (overflow = true, 0ull)
                : static_cast<std::uint64_t>(resources.sharedTmpCapacityElements) * sizeof(float);
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

    add_snapshot_bytes(
        snapshot,
        static_cast<std::uint64_t>(resources.pendingScanErrorReadbacks.size()) * sizeof(int));

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
        (resources.sharedTmpCapacityElements >
         (std::numeric_limits<std::uint64_t>::max() / sizeof(float)))
            ? (overflow = true, 0ull)
            : static_cast<std::uint64_t>(resources.sharedTmpCapacityElements) * sizeof(float);
    if (overflow) {
        scratchSnapshot.overflow = true;
    }
    if (resources.sharedTmpPlane) {
        add_snapshot_bytes(scratchSnapshot, sharedTmpBytes);
    }

    add_optics_scratch_bytes(resources.scannerScratch, scratchSnapshot);
    add_spatial_dir_scratch_bytes(resources.spatialDirScratch, scratchSnapshot);

    add_snapshot_bytes(
        scratchSnapshot,
        static_cast<std::uint64_t>(resources.pendingScanErrorReadbacks.size()) * sizeof(int));

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
        const std::uint64_t targetBytes =
            tier_target_bytes(cfg, transaction.resolvedPressurePolicy, tier);
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


inline bool should_collect_manager_memory_snapshots(const ResolvedPressurePolicy& policy) noexcept {
    return pressure_policy_enabled(policy) || JTRACE_ENABLED(3);
}

#if JUICER_DIAGNOSTICS_COMPILED
std::uint32_t trace_policy_match_u32(
    const SubmissionTransaction& transaction,
    std::uint64_t softTargetBytes,
    std::uint64_t reserveBytes) noexcept {
    const ResolvedPressurePolicy& policy = transaction.resolvedPressurePolicy;
    return (softTargetBytes == policy.softTargetBytes && reserveBytes == policy.reserveBytes) ? 1u : 0u;
}

std::string trace_resolved_policy_fields(
    const SubmissionTransaction& transaction,
    std::uint64_t softTargetBytes,
    std::uint64_t reserveBytes) {
    const ResolvedPressurePolicy& policy = transaction.resolvedPressurePolicy;
    return std::string(" policy_source=") + to_cstr(policy.policySource)
        + " policy_device_id=" + std::to_string(policy.policyDeviceId)
        + " soft_target_bytes=" + std::to_string(static_cast<unsigned long long>(softTargetBytes))
        + " reserve_bytes=" + std::to_string(static_cast<unsigned long long>(reserveBytes))
        + " reader_matches_submission_policy=" + std::to_string(
            trace_policy_match_u32(transaction, softTargetBytes, reserveBytes));
}

std::string trace_resolved_policy_fields(const SubmissionTransaction& transaction) {
    return trace_resolved_policy_fields(
        transaction,
        transaction.resolvedPressurePolicy.softTargetBytes,
        transaction.resolvedPressurePolicy.reserveBytes);
}

std::string trace_headroom_identity_fields(
    const SubmissionTransaction& transaction,
    const HeadroomTelemetry& headroom) {
    const int submissionDeviceId = transaction.snapshot.deviceContextKey.deviceId;
    const bool sampleMatchesSubmissionDevice =
        headroom.sampleSuccess && headroom.sampledDeviceId == submissionDeviceId;
    return std::string(" submission_device_id=") + std::to_string(submissionDeviceId)
        + " sampled_device_id=" + std::to_string(headroom.sampledDeviceId)
        + " sample_matches_submission_device=" + std::to_string(sampleMatchesSubmissionDevice ? 1u : 0u)
        + " sample_success=" + std::to_string(headroom.sampleSuccess ? 1u : 0u);
}
#endif


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

AdmissionChurnPolicyState& admission_churn_policy_state();
OptionalHeuristicTraceState& optional_heuristic_trace_state();

void trace_scratch_policy_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const char* result,
    std::size_t requestBytes,
    const ScratchPolicySnapshot& snapshot,
    int waitMs) {
#if JUICER_DIAGNOSTICS_COMPILED
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
#endif
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
#if JUICER_DIAGNOSTICS_COMPILED
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("transient_reservation", transaction, commandName)
        + " kind=" + to_cstr(ReservationKind::TransientNonManager)
        + trace_resolved_policy_fields(transaction)
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
        + trace_reason_class_field_if_known("reason_class", reason);
    JTRACE("MSTRS", msg);
#endif
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
#if JUICER_DIAGNOSTICS_COMPILED
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("upload_reservation", transaction, commandName)
        + " kind=" + to_cstr(ReservationKind::UploadCopy)
        + " instance_token=" + std::to_string(static_cast<unsigned long long>(instanceToken))
        + trace_resolved_policy_fields(transaction)
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
        + trace_reason_class_field_if_known("reason_class", reason);
    JTRACE("MSUPL", msg);
#endif
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
#if JUICER_DIAGNOSTICS_COMPILED
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
        + trace_reason_class_field_if_known("reason_class", reason);
    JTRACE("MSBPR", msg);
#endif
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
#if JUICER_DIAGNOSTICS_COMPILED
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
#endif
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
    std::uint64_t scratchNormalizedActions,
    std::uint64_t graphEvictedEntries,
    const char* reason) {
#if JUICER_DIAGNOSTICS_COMPILED
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("tier_budget", transaction, commandName)
        + " lane=" + to_cstr(lane)
        + " state=" + to_cstr(pressureState)
        + trace_resolved_policy_fields(transaction)
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
        + " scratch_normalized_actions=" + std::to_string(static_cast<unsigned long long>(scratchNormalizedActions))
        + " graph_evicted_entries=" + std::to_string(static_cast<unsigned long long>(graphEvictedEntries))
        + trace_device_context_fields(transaction)
        + " reason=" + trace_or_unspecified(reason);
    JTRACE("MSTGT", msg);
#endif
}

void trace_scratch_request_descriptor(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const ScratchRequestDescriptor& descriptor,
    const char* reason) {
#if JUICER_DIAGNOSTICS_COMPILED
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("scratch_request", transaction, commandName)
        + " request_generation=" + std::to_string(static_cast<unsigned long long>(descriptor.generation))
        + " need_optics=" + std::to_string(descriptor.needOptics ? 1 : 0)
        + " need_spatial_dir=" + std::to_string(descriptor.needSpatialDir ? 1 : 0)
        + " requested_width=" + std::to_string(descriptor.requestedWidth)
        + " requested_height=" + std::to_string(descriptor.requestedHeight)
        + " need_blurred=" + std::to_string(descriptor.needBlurred ? 1 : 0)
        + " need_aux=" + std::to_string(descriptor.needAux ? 1 : 0)
        + " need_grain_triplet=" + std::to_string(descriptor.needGrainTriplet ? 1 : 0)
        + " need_grain_shared=" + std::to_string(descriptor.needGrainShared ? 1 : 0)
        + " need_gate_mask=" + std::to_string(descriptor.needGateMask ? 1 : 0)
        + trace_device_context_fields(transaction)
        + " reason=" + trace_or_unspecified(reason);
    JTRACE("MSSRQ", msg);
#endif
}

void trace_scratch_checkpoint_event(
    const SubmissionTransaction& transaction,
    const char* commandName,
    ScratchCheckpointInvocation invocation,
    const ScratchCheckpointObservation& observation,
    const char* reason) {
#if JUICER_DIAGNOSTICS_COMPILED
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const auto bytes_for_candidate = [&](ScratchPolicyCandidate candidate) -> std::uint64_t {
        const std::size_t index = scratch_policy_candidate_index(candidate);
        if (index >= observation.residency.candidates.size()) {
            return 0;
        }
        return observation.residency.candidates[index].liveRetainedBytes;
    };

    const auto helper_non_policy_bytes = [&](ScratchHelperNonPolicyAllocation allocation) -> std::uint64_t {
        const std::size_t index = scratch_helper_non_policy_index(allocation);
        if (index >= observation.residency.helperNonPolicyBytes.size()) {
            return 0;
        }
        return observation.residency.helperNonPolicyBytes[index];
    };

    const auto eligible_now = [&](ScratchPolicyCandidate candidate) -> std::uint32_t {
        const std::size_t index = scratch_policy_candidate_index(candidate);
        if (index >= observation.eligibility.eligibleNow.size()) {
            return 0u;
        }
        return observation.eligibility.eligibleNow[index] ? 1u : 0u;
    };

    const ResolvedPressurePolicy& policy = transaction.resolvedPressurePolicy;
    const std::string msg = trace_event_prefix("scratch_checkpoint", transaction, commandName)
        + " invocation=" + to_cstr(invocation)
        + " request_active=" + std::to_string(observation.requestActive ? 1 : 0)
        + " pressure_policy_enabled=" + std::to_string(pressure_policy_enabled(policy) ? 1 : 0)
        + " policy_source=" + std::string(to_cstr(policy.policySource))
        + " policy_device_id=" + std::to_string(policy.policyDeviceId)
        + " soft_target_bytes=" + std::to_string(
            static_cast<unsigned long long>(policy.softTargetBytes))
        + " reserve_bytes=" + std::to_string(
            static_cast<unsigned long long>(policy.reserveBytes))
        + " reader_matches_submission_policy=1"
        + " stage1_over_target=" + std::to_string(observation.stage1OverTarget ? 1 : 0)
        + " stage2_evaluated=" + std::to_string(observation.stage2Evaluated ? 1 : 0)
        + " stage2_skipped_no_shed_cache=" + std::to_string(observation.stage2SkippedByNoShedCache ? 1 : 0)
        + " over_target_not_reducible=" + std::to_string(observation.overTargetButNotReducible ? 1 : 0)
        + " shedding_attempted=" + std::to_string(observation.sheddingAttempted ? 1 : 0)
        + " shedding_progressed=" + std::to_string(observation.sheddingProgressed ? 1 : 0)
        + " shedding_partial_failure=" + std::to_string(observation.sheddingPartialFailure ? 1 : 0)
        + " shedding_target_reached=" + std::to_string(observation.sheddingTargetReached ? 1 : 0)
        + " shedding_orphaned_shared_tmp_retired=" + std::to_string(
            observation.sheddingOrphanedSharedTmpRetired ? 1 : 0)
        + " request_generation=" + std::to_string(
            static_cast<unsigned long long>(observation.requestDescriptorGeneration))
        + " retained_generation=" + std::to_string(
            static_cast<unsigned long long>(observation.retainedScratchGeneration))
        + " scratch_target_bytes=" + std::to_string(
            static_cast<unsigned long long>(observation.scratchTargetBytes))
        + " hysteresis_bytes=" + std::to_string(
            static_cast<unsigned long long>(observation.hysteresisBytes))
        + " policy_live_retained_bytes=" + std::to_string(
            static_cast<unsigned long long>(observation.policyLiveRetainedBytes))
        + " reclaimable_live_bytes=" + std::to_string(
            static_cast<unsigned long long>(observation.reclaimableLiveBytes))
        + " shed_retired_action_count=" + std::to_string(
            static_cast<unsigned long long>(observation.shedRetiredActionCount))
        + " shed_retired_live_bytes=" + std::to_string(
            static_cast<unsigned long long>(observation.shedRetiredLiveBytes))
        + " total_live_retained_bytes=" + std::to_string(
            static_cast<unsigned long long>(observation.residency.totalLiveRetainedBytes))
        + " retire_pending_scratch_bytes=" + std::to_string(
            static_cast<unsigned long long>(observation.residency.retirePendingScratchBytes))
        + " helper_shared_bytes=" + std::to_string(
            static_cast<unsigned long long>(observation.residency.helperSharedBytes))
        + " helper_non_policy_bytes=" + std::to_string(
            static_cast<unsigned long long>(observation.residency.helperNonPolicyTotalBytes))
        + " optics_base_bytes=" + std::to_string(
            static_cast<unsigned long long>(bytes_for_candidate(ScratchPolicyCandidate::OpticsBase)))
        + " optics_blurred_bytes=" + std::to_string(
            static_cast<unsigned long long>(bytes_for_candidate(ScratchPolicyCandidate::OpticsBlurred)))
        + " optics_aux_bytes=" + std::to_string(
            static_cast<unsigned long long>(bytes_for_candidate(ScratchPolicyCandidate::OpticsAux)))
        + " optics_grain_triplet_bytes=" + std::to_string(
            static_cast<unsigned long long>(bytes_for_candidate(ScratchPolicyCandidate::OpticsGrainTriplet)))
        + " optics_grain_shared_bytes=" + std::to_string(
            static_cast<unsigned long long>(bytes_for_candidate(ScratchPolicyCandidate::OpticsGrainShared)))
        + " optics_gate_mask_bytes=" + std::to_string(
            static_cast<unsigned long long>(bytes_for_candidate(ScratchPolicyCandidate::OpticsGateMask)))
        + " spatial_dir_base_bytes=" + std::to_string(
            static_cast<unsigned long long>(bytes_for_candidate(ScratchPolicyCandidate::SpatialDirBase)))
        + " optics_base_eligible_now=" + std::to_string(eligible_now(ScratchPolicyCandidate::OpticsBase))
        + " optics_blurred_eligible_now=" + std::to_string(eligible_now(ScratchPolicyCandidate::OpticsBlurred))
        + " optics_aux_eligible_now=" + std::to_string(eligible_now(ScratchPolicyCandidate::OpticsAux))
        + " optics_grain_triplet_eligible_now=" + std::to_string(eligible_now(ScratchPolicyCandidate::OpticsGrainTriplet))
        + " optics_grain_shared_eligible_now=" + std::to_string(eligible_now(ScratchPolicyCandidate::OpticsGrainShared))
        + " optics_gate_mask_eligible_now=" + std::to_string(eligible_now(ScratchPolicyCandidate::OpticsGateMask))
        + " spatial_dir_base_eligible_now=" + std::to_string(eligible_now(ScratchPolicyCandidate::SpatialDirBase))
        + " shared_tmp_eligible_now=" + std::to_string(observation.eligibility.sharedTmpEligibleNow ? 1 : 0)
        + " scan_error_flag_bytes=" + std::to_string(
            static_cast<unsigned long long>(helper_non_policy_bytes(ScratchHelperNonPolicyAllocation::ScanErrorFlag)))
        + " scan_error_host_bytes=" + std::to_string(
            static_cast<unsigned long long>(helper_non_policy_bytes(ScratchHelperNonPolicyAllocation::ScanErrorHost)))
        + " auto_exposure_scale_bytes=" + std::to_string(
            static_cast<unsigned long long>(helper_non_policy_bytes(ScratchHelperNonPolicyAllocation::AutoExposureExposureScale)))
        + " auto_exposure_ev_bytes=" + std::to_string(
            static_cast<unsigned long long>(helper_non_policy_bytes(ScratchHelperNonPolicyAllocation::AutoExposureAutoEV)))
        + " auto_exposure_valid_bytes=" + std::to_string(
            static_cast<unsigned long long>(helper_non_policy_bytes(ScratchHelperNonPolicyAllocation::AutoExposureValid)))
        + " auto_exposure_max_y_bits_bytes=" + std::to_string(
            static_cast<unsigned long long>(helper_non_policy_bytes(ScratchHelperNonPolicyAllocation::AutoExposureMaxYBits)))
        + " auto_exposure_histogram_bytes=" + std::to_string(
            static_cast<unsigned long long>(helper_non_policy_bytes(ScratchHelperNonPolicyAllocation::AutoExposureHistogram)))
        + " auto_exposure_weights_x_bytes=" + std::to_string(
            static_cast<unsigned long long>(helper_non_policy_bytes(ScratchHelperNonPolicyAllocation::AutoExposureWeightsX)))
        + " auto_exposure_weights_y_bytes=" + std::to_string(
            static_cast<unsigned long long>(helper_non_policy_bytes(ScratchHelperNonPolicyAllocation::AutoExposureWeightsY)))
        + " auto_exposure_partials_a_bytes=" + std::to_string(
            static_cast<unsigned long long>(helper_non_policy_bytes(ScratchHelperNonPolicyAllocation::AutoExposurePartialsA)))
        + " auto_exposure_partials_b_bytes=" + std::to_string(
            static_cast<unsigned long long>(helper_non_policy_bytes(ScratchHelperNonPolicyAllocation::AutoExposurePartialsB)))
        + trace_device_context_fields(transaction)
        + " reason=" + trace_or_unspecified(reason);
    JTRACE("MSSCP", msg);
#endif
}

void trace_pressure_gate_reader(
    const SubmissionTransaction& transaction,
    const char* commandName,
    PressureLane lane,
    std::size_t requestBytes,
    bool criticalCurrentFrame,
    const ScratchRequestDescriptor* scratchRequest,
    const char* path,
    const char* reason) {
#if JUICER_DIAGNOSTICS_COMPILED
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("pressure_gate_reader", transaction, commandName)
        + " lane=" + to_cstr(lane)
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + " pressure_policy_enabled=" + std::to_string(
            pressure_policy_enabled(transaction.resolvedPressurePolicy) ? 1 : 0)
        + trace_resolved_policy_fields(transaction)
        + " request_active=" + std::to_string(scratchRequest ? 1 : 0)
        + " request_generation=" + std::to_string(
            static_cast<unsigned long long>(scratchRequest ? scratchRequest->generation : 0))
        + " path=" + trace_or_unspecified(path)
        + " reason=" + trace_or_unspecified(reason);
    JTRACE("MSPGR", msg);
#endif
}

void trace_scratch_shedding_event(
    const SubmissionTransaction& transaction,
    const char* commandName,
    ScratchCheckpointInvocation invocation,
    const char* actionName,
    std::uint64_t candidateLiveBytesBefore,
    std::uint64_t candidateLiveBytesAfter,
    std::uint64_t policyLiveBytesBefore,
    std::uint64_t policyLiveBytesAfter,
    bool orphanedSharedTmpRetired,
    bool partialFailure,
    const char* reason) {
#if JUICER_DIAGNOSTICS_COMPILED
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("scratch_shedding", transaction, commandName)
        + " invocation=" + to_cstr(invocation)
        + " action=" + trace_or_unspecified(actionName)
        + " candidate_live_bytes_before=" + std::to_string(
            static_cast<unsigned long long>(candidateLiveBytesBefore))
        + " candidate_live_bytes_after=" + std::to_string(
            static_cast<unsigned long long>(candidateLiveBytesAfter))
        + " policy_live_retained_bytes_before=" + std::to_string(
            static_cast<unsigned long long>(policyLiveBytesBefore))
        + " policy_live_retained_bytes_after=" + std::to_string(
            static_cast<unsigned long long>(policyLiveBytesAfter))
        + " orphaned_shared_tmp_retired=" + std::to_string(orphanedSharedTmpRetired ? 1 : 0)
        + " partial_failure=" + std::to_string(partialFailure ? 1 : 0)
        + trace_device_context_fields(transaction)
        + " reason=" + trace_or_unspecified(reason);
    JTRACE("MSSSH", msg);
#endif
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
#if JUICER_DIAGNOSTICS_COMPILED
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    const std::string msg = trace_event_prefix("effective_reserve", transaction, commandName)
        + trace_resolved_policy_fields(
            transaction,
            transaction.resolvedPressurePolicy.softTargetBytes,
            reserveBaseBytes)
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
#endif
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
#if JUICER_DIAGNOSTICS_COMPILED
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
#endif
}

void trace_active_burst_event(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const ActiveBurstDecision& burst,
    bool criticalCurrentFrame,
    const char* reason) {
#if JUICER_DIAGNOSTICS_COMPILED
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
#endif
}

void trace_budget_reclaim_retry(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::uint32_t attempt,
    std::size_t reclaimedBytes,
    bool success,
    const char* reason) {
#if JUICER_DIAGNOSTICS_COMPILED
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
#endif
}

void trace_pressure_checkpoint(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const PressureCheckpoint& checkpoint,
    std::size_t pendingGrowthBytes,
    const char* reason) {
#if JUICER_DIAGNOSTICS_COMPILED
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
        + trace_resolved_policy_fields(
            transaction,
            checkpoint.input.softTargetBytes,
            checkpoint.input.reserveBytes)
        + trace_headroom_identity_fields(transaction, checkpoint.headroom)
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
#endif
}

void trace_headroom_sample(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const PressureCheckpoint& checkpoint,
    std::size_t requestBytes,
    bool sourceSwitch,
    const char* reason) {
#if JUICER_DIAGNOSTICS_COMPILED
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("headroom", transaction, commandName)
        + trace_device_context_fields(transaction)
        + trace_resolved_policy_fields(
            transaction,
            checkpoint.input.softTargetBytes,
            checkpoint.input.reserveBytes)
        + trace_headroom_identity_fields(transaction, checkpoint.headroom)
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
#endif
}

void trace_transient_non_manager_sample(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const PressureCheckpoint& checkpoint,
    std::size_t pendingGrowthBytes,
    const char* reason) {
#if JUICER_DIAGNOSTICS_COMPILED
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("transient_non_manager_sample", transaction, commandName)
        + trace_device_context_fields(transaction)
        + trace_resolved_policy_fields(
            transaction,
            checkpoint.input.softTargetBytes,
            checkpoint.input.reserveBytes)
        + " transient_non_manager_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.input.transientNonManagerBytes))
        + " pending_growth_bytes=" + std::to_string(static_cast<unsigned long long>(pendingGrowthBytes))
        + " pressure_total_bytes=" + std::to_string(static_cast<unsigned long long>(pressure_total_bytes(checkpoint.input)))
        + " reason=" + trace_or_unspecified(reason);
    JTRACE("MSTRN", msg);
#endif
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
#if JUICER_DIAGNOSTICS_COMPILED
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
#endif
}

void trace_lane_wait_event(
    const SubmissionTransaction& transaction,
    const char* commandName,
    PressureLane lane,
    bool criticalCurrentFrame,
    int waitMs,
    const char* outcome,
    const char* reason) {
#if JUICER_DIAGNOSTICS_COMPILED
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
#endif
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
#if JUICER_DIAGNOSTICS_COMPILED
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
#endif
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
#if JUICER_DIAGNOSTICS_COMPILED
    if (!JTRACE_ENABLED(2)) {
        return;
    }
    const std::string msg = trace_event_identity_prefix("keep_hot_surface", transaction)
        + trace_device_context_fields(transaction)
        + " enabled=" + std::to_string(cfg.keepHotMs > 0 ? 1 : 0)
        + " keep_hot_ms=" + std::to_string(static_cast<unsigned long long>(cfg.keepHotMs))
        + " reason=" + trace_or_unspecified(reason)
        + trace_reason_class_field_if_known("reason_class", reason);
    JTRACE("MSHOT", msg);
    telemetry_counter_add(global_state().keepHotSurfaceTraceEvents, 1);
#endif
}

void trace_keep_hot_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::uint64_t keepHotMs,
    std::uint64_t bypassEvents,
    std::uint64_t forcedEvictEvents,
    const char* reason) {
#if JUICER_DIAGNOSTICS_COMPILED
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("keep_hot_decision", transaction, commandName)
        + " keep_hot_ms=" + std::to_string(static_cast<unsigned long long>(keepHotMs))
        + " bypass_events=" + std::to_string(static_cast<unsigned long long>(bypassEvents))
        + " forced_evict_events=" + std::to_string(static_cast<unsigned long long>(forcedEvictEvents))
        + trace_device_context_fields(transaction)
        + " reason=" + trace_or_unspecified(reason)
        + trace_reason_class_field_if_known("reason_class", reason);
    JTRACE("MSHOT", msg);
#endif
}

void trace_burst_debt_surface(
    const SubmissionTransaction& transaction,
    const ResourceManagerConfigEffective& cfg,
    const char* reason) {
#if JUICER_DIAGNOSTICS_COMPILED
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
        + trace_reason_class_field_if_known("reason_class", reason);
    JTRACE("MSBDE", msg);
    telemetry_counter_add(global_state().burstDebtSurfaceTraceEvents, 1);
#endif
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
#if JUICER_DIAGNOSTICS_COMPILED
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
#endif
}

void trace_superseded_builder_cancel_surface(
    const SubmissionTransaction& transaction,
    const ResourceManagerConfigEffective& cfg,
    const char* reason) {
#if JUICER_DIAGNOSTICS_COMPILED
    if (!JTRACE_ENABLED(2)) {
        return;
    }
    const std::string msg = trace_event_identity_prefix("superseded_builder_cancel_surface", transaction)
        + trace_device_context_fields(transaction)
        + " enabled=" + std::to_string(cfg.cancelSupersededBuilders ? 1 : 0)
        + " reason=" + trace_or_unspecified(reason)
        + trace_reason_class_field_if_known("reason_class", reason);
    JTRACE("MSCNL", msg);
    telemetry_counter_add(global_state().supersededBuilderCancelSurfaceTraceEvents, 1);
#endif
}

void trace_superseded_builder_cancel_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    bool criticalCurrentFrame,
    std::uint64_t requestBytes,
    std::uint64_t latestSnapshotId,
    const SupersededBuilderCancelInput& input,
    const SupersededBuilderCancelDecision& decision) {
#if JUICER_DIAGNOSTICS_COMPILED
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
#endif
}

void trace_optional_heuristic_surfaces_once(
    const SubmissionTransaction& transaction,
    const ResourceManagerConfigEffective& cfg,
    const char* reason) {
#if JUICER_DIAGNOSTICS_COMPILED
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
#endif
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
#if JUICER_DIAGNOSTICS_COMPILED
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
#endif
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
#if JUICER_DIAGNOSTICS_COMPILED
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
        + trace_reason_class_field_if_known("reason_class", reason);
    JTRACE("MSADM", msg);
#endif
}

void trace_probation_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::uint64_t entryDigest,
    std::uint32_t observedProbationHits,
    std::uint32_t requiredProbationHits,
    bool admitted,
    const char* reason) {
#if JUICER_DIAGNOSTICS_COMPILED
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
#endif
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
#if JUICER_DIAGNOSTICS_COMPILED
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
#endif
}

void trace_reap_pass(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::size_t reclaimedBytes,
    bool success,
    const char* reason) {
#if JUICER_DIAGNOSTICS_COMPILED
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("reap_pass", transaction, commandName)
        + " reclaimed_bytes=" + std::to_string(static_cast<unsigned long long>(reclaimedBytes))
        + " success=" + std::to_string(success ? 1 : 0)
        + trace_device_context_fields(transaction)
        + " reason=" + trace_or_unspecified(reason)
        + trace_reason_class_field_if_known("reason_class", reason);
    JTRACE("MSREAP", msg);
#endif
}

void trace_fragmentation_recovery(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::uint32_t attempt,
    std::size_t requestBytes,
    std::size_t reapedBytes,
    std::uint64_t graphEvictedEntries,
    bool success,
    const char* stage,
    const char* reason) {
#if JUICER_DIAGNOSTICS_COMPILED
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::string msg = trace_event_prefix("fragmentation_recovery", transaction, commandName)
        + " attempt=" + std::to_string(static_cast<unsigned long long>(attempt))
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " reaped_bytes=" + std::to_string(static_cast<unsigned long long>(reapedBytes))
        + " graph_evicted_entries=" + std::to_string(static_cast<unsigned long long>(graphEvictedEntries))
        + " success=" + std::to_string(success ? 1 : 0)
        + " stage=" + trace_or_unknown(stage)
        + trace_device_context_fields(transaction)
        + " reason=" + trace_or_unspecified(reason)
        + trace_reason_class_field_if_known("reason_class", reason);
    JTRACE("MSFRAG", msg);
#endif
}

namespace {

AcquireDecision make_decision(AcquireStatus status) noexcept {
    AcquireDecision out{};
    out.status = status;
    out.shouldBuild = (status == AcquireStatus::Miss);
    return out;
}

} // namespace

AcquireDecision classify_shadow_acquire(bool hasPrevious, bool invalidated) noexcept {
    if (!hasPrevious) {
        return make_decision(AcquireStatus::Miss);
    }
    if (invalidated) {
        return make_decision(AcquireStatus::Miss);
    }
    return make_decision(AcquireStatus::Hit);
}

ResourcePlan build_shadow_resource_plan(const ShadowKeyDelta& delta) noexcept {
    ResourcePlan plan{};
    for (ResourceKind kind : kResourceKindOrder) {
        ResourcePlanEntry& entry = resource_plan_entry(plan, kind);
        entry.invalidated = delta.keySchemaChanged || shadow_key_changed_for_kind(delta, kind);
        entry.acquire = classify_shadow_acquire(delta.hasPrevious, entry.invalidated);
    }
    return plan;
}

bool shadow_key_changed_for_kind(const ShadowKeyDelta& delta, ResourceKind kind) noexcept {
    switch (kind) {
    case ResourceKind::UploadCore:
        return delta.uploadCoreChanged;
    case ResourceKind::Dir:
        return delta.dirChanged;
    case ResourceKind::Scanner:
        return delta.scannerChanged;
    case ResourceKind::AutoExposure:
        return delta.autoExposureChanged;
    default:
        return true;
    }
}

ResourcePlanEntry& resource_plan_entry(ResourcePlan& plan, ResourceKind kind) noexcept {
    switch (kind) {
    case ResourceKind::UploadCore:
        return plan.uploadCore;
    case ResourceKind::Dir:
        return plan.dir;
    case ResourceKind::Scanner:
        return plan.scanner;
    case ResourceKind::AutoExposure:
        return plan.autoExposure;
    default:
        return plan.uploadCore;
    }
}

const ResourcePlanEntry& resource_plan_entry(const ResourcePlan& plan, ResourceKind kind) noexcept {
    switch (kind) {
    case ResourceKind::UploadCore:
        return plan.uploadCore;
    case ResourceKind::Dir:
        return plan.dir;
    case ResourceKind::Scanner:
        return plan.scanner;
    case ResourceKind::AutoExposure:
        return plan.autoExposure;
    default:
        return plan.uploadCore;
    }
}

const char* to_cstr(AcquireStatus status) noexcept {
    switch (status) {
    case AcquireStatus::Hit:
        return "Hit";
    case AcquireStatus::Miss:
        return "Miss";
    case AcquireStatus::Busy:
        return "Busy";
    case AcquireStatus::Exhausted:
        return "Exhausted";
    case AcquireStatus::Error:
        return "Error";
    default:
        return "Unknown";
    }
}

StaleDecision classify_stale_path(const StaleInput& input) noexcept {
    StaleDecision out{};
    if (input.expectedRegistryGeneration != input.observedRegistryGeneration) {
        out.hardStale = true;
        out.reason = StaleReason::RegistryGenerationMismatch;
        return out;
    }
    if (input.expectedContextEpoch != input.observedContextEpoch) {
        out.hardStale = true;
        out.reason = StaleReason::ContextEpochMismatch;
        return out;
    }
    if (input.expectedLeaseGeneration != input.observedLeaseGeneration) {
        out.hardStale = true;
        out.reason = StaleReason::LeaseGenerationMismatch;
        return out;
    }
    if (input.keySchemaMismatch) {
        out.hardMiss = true;
        out.reason = StaleReason::KeySchemaMismatch;
        return out;
    }
    return out;
}

const char* to_cstr(StaleReason reason) noexcept {
    switch (reason) {
    case StaleReason::None:
        return "None";
    case StaleReason::RegistryGenerationMismatch:
        return "RegistryGenerationMismatch";
    case StaleReason::ContextEpochMismatch:
        return "ContextEpochMismatch";
    case StaleReason::LeaseGenerationMismatch:
        return "LeaseGenerationMismatch";
    case StaleReason::KeySchemaMismatch:
        return "KeySchemaMismatch";
    default:
        return "Unknown";
    }
}

const char* to_cstr(PressureState state) noexcept {
    switch (state) {
    case PressureState::Normal:
        return "Normal";
    case PressureState::Constrained:
        return "Constrained";
    case PressureState::Critical:
        return "Critical";
    case PressureState::Emergency:
        return "Emergency";
    default:
        return "Unknown";
    }
}

const char* to_cstr(ReservationKind kind) noexcept {
    switch (kind) {
    case ReservationKind::TransientNonManager:
        return "TransientNonManager";
    case ReservationKind::UploadCopy:
        return "UploadCopy";
    case ReservationKind::BuilderWork:
        return "BuilderWork";
    default:
        return "Unknown";
    }
}

const char* to_cstr(HeadroomSource source) noexcept {
    switch (source) {
    case HeadroomSource::FreeVramOnly:
        return "free_vram_only";
    case HeadroomSource::AllocatorPool:
        return "allocator_pool";
    default:
        return "unknown";
    }
}

const char* to_cstr(CacheAdmissionClass value) noexcept {
    switch (value) {
    case CacheAdmissionClass::Normal:
        return "Normal";
    case CacheAdmissionClass::Probation:
        return "Probation";
    case CacheAdmissionClass::TooLargeToCache:
        return "TooLargeToCache";
    default:
        return "Unknown";
    }
}

int pressure_state_rank(PressureState state) noexcept {
    switch (state) {
    case PressureState::Normal:
        return 0;
    case PressureState::Constrained:
        return 1;
    case PressureState::Critical:
        return 2;
    case PressureState::Emergency:
        return 3;
    default:
        return 0;
    }
}

PressureState max_pressure_state(PressureState a, PressureState b) noexcept {
    return (pressure_state_rank(a) >= pressure_state_rank(b)) ? a : b;
}

PressureDecision classify_pressure(const PressureInput& input) noexcept {
    PressureDecision out{};
    const std::uint64_t effectiveReserveBytes =
        (input.effectiveReserveBytes > 0) ? input.effectiveReserveBytes : input.reserveBytes;
    out.effectiveReserveBytes = effectiveReserveBytes;
    out.effectiveHeadroomBytes = input.effectiveHeadroomBytes;
    out.headroomSource = input.headroomSource;

    if (input.softTargetBytes == 0) {
        out.state = PressureState::Normal;
        return out;
    }

    const std::uint64_t pressureBytes =
        input.managerResidentBytes + input.retirePendingBytes + input.transientNonManagerBytes;

    const std::uint64_t constrainedThreshold = input.softTargetBytes;
    const std::uint64_t criticalThreshold = input.softTargetBytes + (effectiveReserveBytes / 2u);
    const std::uint64_t emergencyThreshold = input.softTargetBytes + effectiveReserveBytes;

    PressureState budgetState = PressureState::Normal;
    if (pressureBytes >= emergencyThreshold && emergencyThreshold > 0) {
        budgetState = PressureState::Emergency;
    }
    else if (pressureBytes >= criticalThreshold && criticalThreshold > 0) {
        budgetState = PressureState::Critical;
    }
    else if (pressureBytes >= constrainedThreshold) {
        budgetState = PressureState::Constrained;
    }

    PressureState headroomState = PressureState::Normal;
    if (effectiveReserveBytes > 0) {
        const std::uint64_t emergencyHeadroomThreshold = effectiveReserveBytes / 2u;
        const std::uint64_t criticalHeadroomThreshold = effectiveReserveBytes;
        const std::uint64_t constrainedHeadroomThreshold =
            effectiveReserveBytes + (effectiveReserveBytes / 2u);
        if (input.effectiveHeadroomBytes <= emergencyHeadroomThreshold) {
            headroomState = PressureState::Emergency;
        }
        else if (input.effectiveHeadroomBytes <= criticalHeadroomThreshold) {
            headroomState = PressureState::Critical;
        }
        else if (input.effectiveHeadroomBytes <= constrainedHeadroomThreshold) {
            headroomState = PressureState::Constrained;
        }
    }

    out.state = max_pressure_state(budgetState, headroomState);
    switch (out.state) {
    case PressureState::Emergency:
        out.allowOpportunistic = false;
        out.requestReclaimPass = true;
        out.shouldShedNonCritical = true;
        break;
    case PressureState::Critical:
        out.allowOpportunistic = false;
        out.requestReclaimPass = true;
        out.shouldShedNonCritical = false;
        break;
    case PressureState::Constrained:
        out.allowOpportunistic = true;
        out.requestReclaimPass = true;
        out.shouldShedNonCritical = false;
        break;
    case PressureState::Normal:
    default:
        break;
    }

    const bool belowEffectiveReserve = (effectiveReserveBytes > 0) &&
        (input.effectiveHeadroomBytes <= effectiveReserveBytes);
    if (input.freezeOpportunisticBelowReserve && belowEffectiveReserve) {
        out.allowOpportunistic = false;
        out.freezeOpportunistic = true;
        out.requestReclaimPass = true;
        if (pressure_state_rank(out.state) < pressure_state_rank(PressureState::Constrained)) {
            out.state = PressureState::Constrained;
        }
    }
    return out;
}

ReservationDecision classify_reservation(const ReservationInput& input) noexcept {
    ReservationDecision out{};
    out.granted = true;
    out.reason = "granted";

    if (input.capBytes == 0) {
        out.granted = true;
        out.reason = "cap_disabled";
        return out;
    }

    const std::uint64_t nextBytes =
        (input.requestBytes > (std::numeric_limits<std::uint64_t>::max() - input.bytesInFlight))
        ? std::numeric_limits<std::uint64_t>::max()
        : input.bytesInFlight + input.requestBytes;
    if (nextBytes <= input.capBytes) {
        return out;
    }

    if (input.criticalCurrentFrame) {
        out.granted = true;
        out.reason = "critical_last_resort";
        return out;
    }

    out.granted = false;
    out.shouldWait = true;
    out.waitMs = 1;
    out.reason = "cap_exceeded";
    return out;
}

CacheAdmissionDecision classify_cache_admission(const CacheAdmissionInput& input) noexcept {
    CacheAdmissionDecision out{};
    out.reason = "normal";

    if (input.requestBytes == 0) {
        out.reason = "empty_request";
        return out;
    }

    const std::uint64_t hardMaxBytes = input.maxCacheableEntryBytes;
    std::uint64_t pctMaxBytes = 0;
    if (input.cacheTargetBytes > 0 && input.maxCacheableEntryPctOfTarget > 0) {
        const std::uint64_t pct = static_cast<std::uint64_t>(input.maxCacheableEntryPctOfTarget);
        const std::uint64_t maxU64 = std::numeric_limits<std::uint64_t>::max();
        const std::uint64_t numerator =
            (input.cacheTargetBytes > (maxU64 / pct))
            ? maxU64
            : (input.cacheTargetBytes * pct);
        pctMaxBytes = numerator / 100ull;
    }

    std::uint64_t maxDurableBytes = std::numeric_limits<std::uint64_t>::max();
    if (hardMaxBytes > 0 && pctMaxBytes > 0) {
        maxDurableBytes = std::min<std::uint64_t>(hardMaxBytes, pctMaxBytes);
    }
    else if (hardMaxBytes > 0) {
        maxDurableBytes = hardMaxBytes;
    }
    else if (pctMaxBytes > 0) {
        maxDurableBytes = pctMaxBytes;
    }
    out.maxDurableBytes = maxDurableBytes;

    const bool tooLarge =
        (maxDurableBytes != std::numeric_limits<std::uint64_t>::max()) &&
        (input.requestBytes > maxDurableBytes);
    if (tooLarge) {
        out.admissionClass = CacheAdmissionClass::TooLargeToCache;
        if (input.criticalCurrentFrame) {
            out.allowDurableAdmission = true;
            out.reason = "too_large_critical_override";
        }
        else {
            out.allowDurableAdmission = false;
            out.reason = "too_large_noncritical";
        }
        return out;
    }

    const std::uint64_t probationThreshold = input.largeEntryProbationThresholdBytes;
    if (probationThreshold > 0 && input.requestBytes > probationThreshold) {
        out.admissionClass = CacheAdmissionClass::Probation;
        out.probationApplied = true;
        const std::uint32_t requiredHits = std::max<std::uint32_t>(1u, input.largeEntryProbationHitsRequired);
        out.probationHitsRequired = requiredHits;
        const std::uint32_t observed =
            (input.observedProbationHits < std::numeric_limits<std::uint32_t>::max())
            ? (input.observedProbationHits + 1u)
            : std::numeric_limits<std::uint32_t>::max();

        if (input.criticalCurrentFrame) {
            out.allowDurableAdmission = true;
            out.reason = "probation_critical_override";
            return out;
        }

        if (observed < requiredHits) {
            out.allowDurableAdmission = false;
            out.reason = "probation_defer";
            return out;
        }

        out.allowDurableAdmission = true;
        out.reason = "probation_admit";
        return out;
    }

    return out;
}

BurstDebtDecision classify_burst_debt(const BurstDebtInput& input) noexcept {
    BurstDebtDecision out{};
    out.enabled = (input.burstDebtHalfLifeMs > 0) && (input.maxBurstDebtPct < 100);
    if (!out.enabled) {
        out.reason = "disabled";
        return out;
    }

    if (input.criticalCurrentFrame && input.burstConsumed) {
        out.accrueDebt = true;
        std::uint32_t incrementPct = 4;
        if (input.burstCapBytes > 0 && input.burstOverTargetBytes > 0) {
            const std::uint64_t ratioPctU64 = std::min<std::uint64_t>(
                100ull,
                (input.burstOverTargetBytes * 100ull) / input.burstCapBytes);
            const std::uint32_t ratioPct = static_cast<std::uint32_t>(ratioPctU64);
            incrementPct = std::max<std::uint32_t>(1u, 1u + (ratioPct / 10u));
        }
        out.debtIncrementPct = std::min<std::uint32_t>(25u, incrementPct);
        out.reason = "accrue_burst_consumed";
        return out;
    }

    if (!input.criticalCurrentFrame &&
        input.currentDebtPct >= input.maxBurstDebtPct) {
        out.throttleOpportunistic = true;
        out.reason = "throttle_max_debt";
        return out;
    }

    if (input.criticalCurrentFrame) {
        out.reason = "critical_no_burst_consumption";
    }
    else {
        out.reason = "below_debt_threshold";
    }
    return out;
}

SupersededBuilderCancelDecision classify_superseded_builder_cancel(
    const SupersededBuilderCancelInput& input) noexcept {
    SupersededBuilderCancelDecision out{};
    if (!input.enabled) {
        out.reason = "disabled";
        return out;
    }
    if (input.criticalCurrentFrame) {
        out.reason = "critical_preserve";
        return out;
    }
    if (!input.superseded) {
        out.reason = "not_superseded";
        return out;
    }
    out.cancel = true;
    out.reason = "cancel_superseded_noncritical";
    return out;
}

AcquireDecision default_acquire_decision() noexcept {
    return AcquireDecision{};
}

PressureDecision default_pressure_decision() noexcept {
    return PressureDecision{};
}

ReservationDecision default_reservation_decision() noexcept {
    return ReservationDecision{};
}

CacheAdmissionDecision default_cache_admission_decision() noexcept {
    return CacheAdmissionDecision{};
}


// Former RM foundation implementation now owned by the RM TU.
ResourceManagerConfigEffective sanitize_config(const ResourceManagerConfigRaw& raw) {
    constexpr std::uint64_t kBasisPointsDenom = 10000ull;
    constexpr std::uint32_t kMinLiveManagers = 8u;
    constexpr std::uint32_t kMaxLiveManagers = 32u;
    constexpr std::uint32_t kMinIdleReapMs = 2000u;
    constexpr std::uint32_t kMaxIdleReapMs = 5000u;
    constexpr std::uint32_t kMinPressureSampleMs = 10u;
    constexpr std::uint32_t kMaxPressureSampleMs = 5000u;
    constexpr std::uint64_t kMiB = 1024ull * 1024ull;
    constexpr std::uint64_t kMinReserveSafetyMarginBytes = 32ull * kMiB;
    constexpr std::uint64_t kMaxReserveSafetyMarginBytes = 1024ull * kMiB;
    constexpr std::uint64_t kMinReserveAdaptStepBytes = 8ull * kMiB;
    constexpr std::uint64_t kMaxReserveAdaptStepBytes = 512ull * kMiB;
    constexpr std::uint64_t kMinMaxActiveBurstBytes = 32ull * kMiB;
    constexpr std::uint64_t kMaxMaxActiveBurstBytes = 1024ull * kMiB;
    constexpr std::uint32_t kMinMaxActiveBurstPctOfTarget = 5u;
    constexpr std::uint32_t kMaxMaxActiveBurstPctOfTarget = 50u;
    constexpr std::uint32_t kMinMaxActiveBurstMs = 50u;
    constexpr std::uint32_t kMaxMaxActiveBurstMs = 5000u;
    constexpr std::uint32_t kMinPressurePollIntervalMs = 10u;
    constexpr std::uint32_t kMaxPressurePollIntervalMs = 5000u;
    constexpr std::uint32_t kMinPressureStateMinDwellMs = 50u;
    constexpr std::uint32_t kMaxPressureStateMinDwellMs = 2000u;
    constexpr std::uint32_t kMinPressureStateMaxTransitionsPerMin = 1u;
    constexpr std::uint32_t kMaxPressureStateMaxTransitionsPerMin = 120u;
    constexpr std::uint32_t kMaxReclaimRetryAttempts = 3u;
    constexpr std::uint32_t kMinTierErrorWindowMs = 100u;
    constexpr std::uint32_t kMaxTierErrorWindowMs = 10000u;
    constexpr std::uint32_t kMinTierErrorThreshold = 1u;
    constexpr std::uint32_t kMaxTierErrorThreshold = 16u;
    constexpr std::uint32_t kMinTierCircuitOpenMs = 100u;
    constexpr std::uint32_t kMaxTierCircuitOpenMs = 10000u;
    constexpr std::uint32_t kMinHostAssetIdleTrimMs = 1000u;
    constexpr std::uint32_t kMaxHostAssetIdleTrimMs = 60000u;
    constexpr std::uint32_t kMinPinnedStagingIdleTrimMs = 500u;
    constexpr std::uint32_t kMaxPinnedStagingIdleTrimMs = 60000u;
    constexpr std::uint32_t kMinScratchBuilderBytesInFlightLimitMB = 128u;
    constexpr std::uint32_t kMaxScratchBuilderBytesInFlightLimitMB = 512u;
    constexpr std::uint32_t kMinLutBuilderBytesInFlightLimitMB = 64u;
    constexpr std::uint32_t kMaxLutBuilderBytesInFlightLimitMB = 256u;
    constexpr std::uint32_t kMinGraphBuilderBytesInFlightLimitMB = 64u;
    constexpr std::uint32_t kMaxGraphBuilderBytesInFlightLimitMB = 256u;
    constexpr std::uint32_t kMinBuilderFairnessTokensPerTick = 1u;
    constexpr std::uint32_t kMaxBuilderFairnessTokensPerTick = 2u;
    constexpr std::uint32_t kMinCriticalBuilderReservedTokens = 1u;
    constexpr std::uint32_t kMinUploadBytesInFlightLimitMB = 128u;
    constexpr std::uint32_t kMaxUploadBytesInFlightLimitMB = 512u;
    constexpr std::uint32_t kMinUploadFairnessTokensPerTick = 1u;
    constexpr std::uint32_t kMaxUploadFairnessTokensPerTick = 2u;
    constexpr std::uint32_t kMinCriticalUploadReservedTokens = 1u;
    constexpr std::uint64_t kMinMaxCacheableEntryBytes = 32ull * kMiB;
    constexpr std::uint64_t kMaxMaxCacheableEntryBytes = 1024ull * kMiB;
    constexpr std::uint32_t kMinMaxCacheableEntryPctOfTarget = 5u;
    constexpr std::uint32_t kMaxMaxCacheableEntryPctOfTarget = 50u;
    constexpr std::uint64_t kMinGraphLargeEntryThresholdBytes = 16ull * kMiB;
    constexpr std::uint64_t kMaxGraphLargeEntryThresholdBytes = 2048ull * kMiB;
    constexpr std::uint64_t kMinGraphLargeEntryQuarantineMaxBytes = 32ull * kMiB;
    constexpr std::uint64_t kMaxGraphLargeEntryQuarantineMaxBytes = 2048ull * kMiB;
    constexpr std::uint32_t kMinGraphLargeEntryQuarantineMaxEntries = 1u;
    constexpr std::uint32_t kMaxGraphLargeEntryQuarantineMaxEntries = 16u;
    constexpr std::uint64_t kMinLargeEntryProbationThresholdBytes = 16ull * kMiB;
    constexpr std::uint32_t kMinLargeEntryProbationHitsRequired = 1u;
    constexpr std::uint32_t kMaxLargeEntryProbationHitsRequired = 4u;
    constexpr std::uint32_t kMinKeepHotMs = 0u;
    constexpr std::uint32_t kMaxKeepHotMs = 5000u;
    constexpr std::uint32_t kMinAdmissionChurnWindowMs = 0u;
    constexpr std::uint32_t kMaxAdmissionChurnWindowMs = 60000u;
    constexpr std::uint32_t kMinAdmissionChurnOneHitRatePct = 1u;
    constexpr std::uint32_t kMaxAdmissionChurnOneHitRatePct = 100u;
    constexpr std::uint32_t kMinAdmissionChurnProbationHitBonus = 0u;
    constexpr std::uint32_t kMaxAdmissionChurnProbationHitBonus = 4u;
    constexpr std::uint32_t kMinLargeEntryReadmitCooldownMs = 0u;
    constexpr std::uint32_t kMaxLargeEntryReadmitCooldownMs = 60000u;
    constexpr std::uint32_t kMinLargeEntryGhostHitsForReadmit = 0u;
    constexpr std::uint32_t kMaxLargeEntryGhostHitsForReadmit = 8u;
    constexpr std::uint32_t kMinBurstDebtHalfLifeMs = 0u;
    constexpr std::uint32_t kMaxBurstDebtHalfLifeMs = 60000u;
    constexpr std::uint32_t kMinMaxBurstDebtPct = 1u;
    constexpr std::uint32_t kMaxMaxBurstDebtPct = 100u;
    constexpr std::uint64_t kMinHostAssetCacheMaxBytes = 64ull * kMiB;
    constexpr std::uint64_t kMaxHostAssetCacheMaxBytes = 1024ull * kMiB;
    constexpr std::uint64_t kMinHostAssetTrimBatchBytes = 8ull * kMiB;
    constexpr std::uint32_t kMinAllocatorBackendPreference = 0u;
    constexpr std::uint32_t kMaxAllocatorBackendPreference = 3u;
    constexpr std::uint32_t kMinAsyncMempoolReleaseThresholdMB = 0u;
    constexpr std::uint32_t kMaxAsyncMempoolReleaseThresholdMB = 4096u;
    constexpr std::uint32_t kMinPrivateLutFallbackPerMediumCap = 0u;
    constexpr std::uint32_t kMaxPrivateLutFallbackPerMediumCap = 1u;
    constexpr std::uint32_t kMinPrivateLutFallbackPerInstanceCap = 0u;
    constexpr std::uint32_t kMaxPrivateLutFallbackPerInstanceCap = 2u;
    constexpr std::uint64_t kMinPinnedStagingMaxBytes = 8ull * kMiB;
    constexpr std::uint64_t kMaxPinnedStagingMaxBytes = 2048ull * kMiB;
    constexpr std::uint64_t kMinPinnedStagingTrimBatchBytes = 1ull * kMiB;

    ResourceManagerConfigEffective out{};
    out.keySchemaVersion = sanitize_submission_key_schema_version(raw.keySchemaVersion);
    out.traceSchemaVersion = sanitize_trace_schema_version(raw.traceSchemaVersion);
    out.allowShadowMode = raw.allowShadowMode;
    out.maxLiveManagersPerProcess = std::clamp(
        raw.maxLiveManagersPerProcess,
        kMinLiveManagers,
        kMaxLiveManagers);
    out.managerIdleReapMs = std::clamp(
        raw.managerIdleReapMs,
        kMinIdleReapMs,
        kMaxIdleReapMs);
    out.managerSoftTargetBytes = raw.managerSoftTargetBytes;
    out.managerReserveBytes = std::min(raw.managerReserveBytes, out.managerSoftTargetBytes);
    out.reserveSafetyMarginBytes = std::clamp(
        raw.reserveSafetyMarginBytes,
        kMinReserveSafetyMarginBytes,
        kMaxReserveSafetyMarginBytes);
    out.reserveAdaptUpStepBytes = std::clamp(
        raw.reserveAdaptUpStepBytes,
        kMinReserveAdaptStepBytes,
        kMaxReserveAdaptStepBytes);
    out.reserveAdaptDownStepBytes = std::clamp(
        raw.reserveAdaptDownStepBytes,
        kMinReserveAdaptStepBytes,
        kMaxReserveAdaptStepBytes);
    out.freezeOpportunisticBelowReserve = raw.freezeOpportunisticBelowReserve;
    out.allowActiveFrameBurst = raw.allowActiveFrameBurst;
    out.maxActiveBurstBytes = std::clamp(
        raw.maxActiveBurstBytes,
        kMinMaxActiveBurstBytes,
        kMaxMaxActiveBurstBytes);
    out.maxActiveBurstPctOfTarget = std::clamp(
        raw.maxActiveBurstPctOfTarget,
        kMinMaxActiveBurstPctOfTarget,
        kMaxMaxActiveBurstPctOfTarget);
    out.maxActiveBurstMs = std::clamp(
        raw.maxActiveBurstMs,
        kMinMaxActiveBurstMs,
        kMaxMaxActiveBurstMs);
    out.pressureSampleIntervalMs = std::clamp(
        raw.pressureSampleIntervalMs,
        kMinPressureSampleMs,
        kMaxPressureSampleMs);
    out.pressurePollIntervalMs = std::clamp(
        raw.pressurePollIntervalMs,
        kMinPressurePollIntervalMs,
        kMaxPressurePollIntervalMs);
    out.pressurePollIntervalMsNormal = std::clamp(
        raw.pressurePollIntervalMsNormal,
        kMinPressurePollIntervalMs,
        kMaxPressurePollIntervalMs);
    out.pressurePollIntervalMsCritical = std::clamp(
        raw.pressurePollIntervalMsCritical,
        kMinPressurePollIntervalMs,
        kMaxPressurePollIntervalMs);
    out.pressurePollIntervalMsCritical = std::min(
        out.pressurePollIntervalMsCritical,
        out.pressurePollIntervalMsNormal);
    out.pressureStateMinDwellMs = std::clamp(
        raw.pressureStateMinDwellMs,
        kMinPressureStateMinDwellMs,
        kMaxPressureStateMinDwellMs);
    out.pressureStateMaxTransitionsPerMin = std::clamp(
        raw.pressureStateMaxTransitionsPerMin,
        kMinPressureStateMaxTransitionsPerMin,
        kMaxPressureStateMaxTransitionsPerMin);
    out.reclaimRetryMaxAttempts = std::min(raw.reclaimRetryMaxAttempts, kMaxReclaimRetryAttempts);
    out.tierErrorWindowMs = std::clamp(
        raw.tierErrorWindowMs,
        kMinTierErrorWindowMs,
        kMaxTierErrorWindowMs);
    out.tierErrorThreshold = std::clamp(
        raw.tierErrorThreshold,
        kMinTierErrorThreshold,
        kMaxTierErrorThreshold);
    out.tierCircuitOpenMs = std::clamp(
        raw.tierCircuitOpenMs,
        kMinTierCircuitOpenMs,
        kMaxTierCircuitOpenMs);
    out.fragmentationRecoveryEnabled = raw.fragmentationRecoveryEnabled;
    out.hostAssetCacheMaxBytes = std::clamp(
        raw.hostAssetCacheMaxBytes,
        kMinHostAssetCacheMaxBytes,
        kMaxHostAssetCacheMaxBytes);
    out.hostAssetIdleTrimMs = std::clamp(
        raw.hostAssetIdleTrimMs,
        kMinHostAssetIdleTrimMs,
        kMaxHostAssetIdleTrimMs);
    out.hostAssetTrimBatchBytes = std::clamp(
        raw.hostAssetTrimBatchBytes,
        kMinHostAssetTrimBatchBytes,
        out.hostAssetCacheMaxBytes);
    out.allocatorBackendPreference = std::clamp(
        raw.allocatorBackendPreference,
        kMinAllocatorBackendPreference,
        kMaxAllocatorBackendPreference);
    out.asyncMempoolReleaseThresholdMB = std::clamp(
        raw.asyncMempoolReleaseThresholdMB,
        kMinAsyncMempoolReleaseThresholdMB,
        kMaxAsyncMempoolReleaseThresholdMB);
    out.privateLutFallbackPerMediumCap = std::clamp(
        raw.privateLutFallbackPerMediumCap,
        kMinPrivateLutFallbackPerMediumCap,
        kMaxPrivateLutFallbackPerMediumCap);
    out.privateLutFallbackPerInstanceCap = std::clamp(
        raw.privateLutFallbackPerInstanceCap,
        kMinPrivateLutFallbackPerInstanceCap,
        kMaxPrivateLutFallbackPerInstanceCap);
    out.privateLutFallbackPerInstanceCap = std::max(
        out.privateLutFallbackPerInstanceCap,
        out.privateLutFallbackPerMediumCap);
    out.pinnedUploadStagingMaxBytes = std::clamp(
        raw.pinnedUploadStagingMaxBytes,
        kMinPinnedStagingMaxBytes,
        kMaxPinnedStagingMaxBytes);
    out.pinnedUploadStagingIdleTrimMs = std::clamp(
        raw.pinnedUploadStagingIdleTrimMs,
        kMinPinnedStagingIdleTrimMs,
        kMaxPinnedStagingIdleTrimMs);
    out.pinnedUploadStagingTrimBatchBytes = std::clamp(
        raw.pinnedUploadStagingTrimBatchBytes,
        kMinPinnedStagingTrimBatchBytes,
        out.pinnedUploadStagingMaxBytes);
    out.scratchBuilderBytesInFlightLimitMB = std::clamp(
        raw.scratchBuilderBytesInFlightLimitMB,
        kMinScratchBuilderBytesInFlightLimitMB,
        kMaxScratchBuilderBytesInFlightLimitMB);
    out.lutBuilderBytesInFlightLimitMB = std::clamp(
        raw.lutBuilderBytesInFlightLimitMB,
        kMinLutBuilderBytesInFlightLimitMB,
        kMaxLutBuilderBytesInFlightLimitMB);
    out.graphBuilderBytesInFlightLimitMB = std::clamp(
        raw.graphBuilderBytesInFlightLimitMB,
        kMinGraphBuilderBytesInFlightLimitMB,
        kMaxGraphBuilderBytesInFlightLimitMB);
    out.builderFairnessTokensPerTick = std::clamp(
        raw.builderFairnessTokensPerTick,
        kMinBuilderFairnessTokensPerTick,
        kMaxBuilderFairnessTokensPerTick);
    out.criticalBuilderReservedTokens = std::clamp(
        raw.criticalBuilderReservedTokens,
        kMinCriticalBuilderReservedTokens,
        out.builderFairnessTokensPerTick);
    out.uploadBytesInFlightLimitMB = std::clamp(
        raw.uploadBytesInFlightLimitMB,
        kMinUploadBytesInFlightLimitMB,
        kMaxUploadBytesInFlightLimitMB);
    out.uploadFairnessTokensPerTick = std::clamp(
        raw.uploadFairnessTokensPerTick,
        kMinUploadFairnessTokensPerTick,
        kMaxUploadFairnessTokensPerTick);
    out.criticalUploadReservedTokens = std::clamp(
        raw.criticalUploadReservedTokens,
        kMinCriticalUploadReservedTokens,
        out.uploadFairnessTokensPerTick);
    out.maxCacheableEntryBytes = std::clamp(
        raw.maxCacheableEntryBytes,
        kMinMaxCacheableEntryBytes,
        kMaxMaxCacheableEntryBytes);
    out.maxCacheableEntryPctOfTarget = std::clamp(
        raw.maxCacheableEntryPctOfTarget,
        kMinMaxCacheableEntryPctOfTarget,
        kMaxMaxCacheableEntryPctOfTarget);
    out.graphLargeEntryThresholdBytes = std::clamp(
        raw.graphLargeEntryThresholdBytes,
        kMinGraphLargeEntryThresholdBytes,
        kMaxGraphLargeEntryThresholdBytes);
    out.graphLargeEntryQuarantineMaxBytes = std::clamp(
        raw.graphLargeEntryQuarantineMaxBytes,
        kMinGraphLargeEntryQuarantineMaxBytes,
        kMaxGraphLargeEntryQuarantineMaxBytes);
    out.graphLargeEntryQuarantineMaxEntries = std::clamp(
        raw.graphLargeEntryQuarantineMaxEntries,
        kMinGraphLargeEntryQuarantineMaxEntries,
        kMaxGraphLargeEntryQuarantineMaxEntries);
    out.graphLargeEntryQuarantineMaxBytes = std::max<std::uint64_t>(
        out.graphLargeEntryQuarantineMaxBytes,
        out.graphLargeEntryThresholdBytes);
    out.largeEntryProbationThresholdBytes = std::clamp(
        raw.largeEntryProbationThresholdBytes,
        kMinLargeEntryProbationThresholdBytes,
        out.maxCacheableEntryBytes);
    out.largeEntryProbationHitsRequired = std::clamp(
        raw.largeEntryProbationHitsRequired,
        kMinLargeEntryProbationHitsRequired,
        kMaxLargeEntryProbationHitsRequired);
    out.keepHotMs = std::clamp(
        raw.keepHotMs,
        kMinKeepHotMs,
        kMaxKeepHotMs);
    out.admissionChurnWindowMs = std::clamp(
        raw.admissionChurnWindowMs,
        kMinAdmissionChurnWindowMs,
        kMaxAdmissionChurnWindowMs);
    out.admissionChurnEnterOneHitRatePct = std::clamp(
        raw.admissionChurnEnterOneHitRatePct,
        kMinAdmissionChurnOneHitRatePct,
        kMaxAdmissionChurnOneHitRatePct);
    out.admissionChurnExitOneHitRatePct = std::clamp(
        raw.admissionChurnExitOneHitRatePct,
        kMinAdmissionChurnOneHitRatePct,
        out.admissionChurnEnterOneHitRatePct);
    out.admissionChurnProbationHitBonus = std::clamp(
        raw.admissionChurnProbationHitBonus,
        kMinAdmissionChurnProbationHitBonus,
        kMaxAdmissionChurnProbationHitBonus);
    out.largeEntryReadmitCooldownMs = std::clamp(
        raw.largeEntryReadmitCooldownMs,
        kMinLargeEntryReadmitCooldownMs,
        kMaxLargeEntryReadmitCooldownMs);
    out.largeEntryGhostHitsForReadmit = std::clamp(
        raw.largeEntryGhostHitsForReadmit,
        kMinLargeEntryGhostHitsForReadmit,
        kMaxLargeEntryGhostHitsForReadmit);
    out.burstDebtHalfLifeMs = std::clamp(
        raw.burstDebtHalfLifeMs,
        kMinBurstDebtHalfLifeMs,
        kMaxBurstDebtHalfLifeMs);
    out.maxBurstDebtPct = std::clamp(
        raw.maxBurstDebtPct,
        kMinMaxBurstDebtPct,
        kMaxMaxBurstDebtPct);
    out.cancelSupersededBuilders = raw.cancelSupersededBuilders;

    std::uint64_t immutableBp = std::min<std::uint64_t>(raw.tierTargetImmutableBp, kBasisPointsDenom);
    std::uint64_t lutBp = std::min<std::uint64_t>(raw.tierTargetLutBp, kBasisPointsDenom);
    std::uint64_t scratchBp = std::min<std::uint64_t>(raw.tierTargetScratchBp, kBasisPointsDenom);
    std::uint64_t graphBp = std::min<std::uint64_t>(raw.tierTargetGraphBp, kBasisPointsDenom);
    std::uint64_t totalBp = immutableBp + lutBp + scratchBp + graphBp;
    if (totalBp == 0) {
        immutableBp = 2500;
        lutBp = 2500;
        scratchBp = 3000;
        graphBp = 2000;
        totalBp = immutableBp + lutBp + scratchBp + graphBp;
    }
    if (totalBp > kBasisPointsDenom) {
        immutableBp = (immutableBp * kBasisPointsDenom) / totalBp;
        lutBp = (lutBp * kBasisPointsDenom) / totalBp;
        scratchBp = (scratchBp * kBasisPointsDenom) / totalBp;
        graphBp = kBasisPointsDenom - (immutableBp + lutBp + scratchBp);
    }

    out.tierTargetImmutableBp = immutableBp;
    out.tierTargetLutBp = lutBp;
    out.tierTargetScratchBp = scratchBp;
    out.tierTargetGraphBp = graphBp;
    return out;
}



std::uint64_t normalize_key_u64(std::uint64_t value) noexcept {
    if (value == 0) {
        return 1;
    }
    return value;
}

std::uint64_t normalize_key_float(double value, double scale) noexcept {
    if (!std::isfinite(value) || !std::isfinite(scale) || scale <= 0.0) {
        return 1;
    }
    const double scaled = value * scale;
    if (!std::isfinite(scaled)) {
        return 1;
    }
    const long long quantized = std::llround(scaled);
    const std::uint64_t raw = static_cast<std::uint64_t>(quantized);
    return normalize_key_u64(raw);
}

std::uint32_t normalize_scan_lut_resolution(std::uint32_t value) noexcept {
    return std::clamp(value, kScanLutResolutionMin, kScanLutResolutionMax);
}

std::uint64_t make_scan_lut_key_digest(
    std::uint32_t medium,
    std::uint64_t tablesHash,
    std::uint64_t densityRangeHash,
    std::uint32_t lutResolution,
    std::uint32_t lutFormatVersion,
    std::uint32_t keySchemaVersion) noexcept {
    if (medium > 1u ||
        tablesHash == 0 ||
        densityRangeHash == 0 ||
        lutFormatVersion == 0 ||
        keySchemaVersion == 0) {
        return 0;
    }

    const std::uint64_t fields[] = {
        static_cast<std::uint64_t>(keySchemaVersion),
        static_cast<std::uint64_t>(medium),
        tablesHash,
        densityRangeHash,
        static_cast<std::uint64_t>(normalize_scan_lut_resolution(lutResolution)),
        static_cast<std::uint64_t>(lutFormatVersion)
    };
    return Hash::hash_bytes(fields, sizeof(fields));
}

KeyDigests make_key_digests(
    std::uint64_t uploadCoreHash,
    std::uint64_t dirHash,
    std::uint64_t scannerHash,
    std::uint64_t autoExposureHash) noexcept {
    KeyDigests digests{};
    digests.uploadCoreHash = normalize_key_u64(uploadCoreHash);
    digests.dirHash = normalize_key_u64(dirHash);
    digests.scannerHash = normalize_key_u64(scannerHash);
    digests.autoExposureHash = normalize_key_u64(autoExposureHash);
    return digests;
}

std::uint64_t key_digest_for_kind(const KeyDigests& digests, ResourceKind kind) noexcept {
    switch (kind) {
    case ResourceKind::UploadCore:
        return digests.uploadCoreHash;
    case ResourceKind::Dir:
        return digests.dirHash;
    case ResourceKind::Scanner:
        return digests.scannerHash;
    case ResourceKind::AutoExposure:
        return digests.autoExposureHash;
    default:
        return 1;
    }
}

void set_key_digest_for_kind(KeyDigests& digests, ResourceKind kind, std::uint64_t hashValue) noexcept {
    const std::uint64_t normalized = normalize_key_u64(hashValue);
    switch (kind) {
    case ResourceKind::UploadCore:
        digests.uploadCoreHash = normalized;
        return;
    case ResourceKind::Dir:
        digests.dirHash = normalized;
        return;
    case ResourceKind::Scanner:
        digests.scannerHash = normalized;
        return;
    case ResourceKind::AutoExposure:
        digests.autoExposureHash = normalized;
        return;
    default:
        return;
    }
}

KeyDigests normalize_key_digests(const KeyDigests& digests) noexcept {
    KeyDigests out = digests;
    for (ResourceKind kind : kResourceKindOrder) {
        set_key_digest_for_kind(out, kind, key_digest_for_kind(out, kind));
    }
    return out;
}



namespace {

struct LatestSnapshotKey {
    std::uint64_t instanceToken = 0;
    DeviceContextKey deviceContextKey{};

    bool operator==(const LatestSnapshotKey& other) const noexcept {
        return instanceToken == other.instanceToken &&
            deviceContextKey == other.deviceContextKey;
    }
};

struct LatestSnapshotKeyHasher {
    std::size_t operator()(const LatestSnapshotKey& key) const noexcept {
        const std::size_t hInstance = std::hash<std::uint64_t>{}(key.instanceToken);
        const std::size_t hContext = DeviceContextKeyHash{}(key.deviceContextKey);
        return hInstance ^ (hContext + 0x9e3779b9u + (hInstance << 6u) + (hInstance >> 2u));
    }
};

struct LatestSnapshotState {
    std::mutex mutex;
    std::unordered_map<LatestSnapshotKey, std::uint64_t, LatestSnapshotKeyHasher> bySubmissionKey;
};

LatestSnapshotState& latest_snapshot_state() {
    static LatestSnapshotState state{};
    return state;
}

QueryReadOnlySnapshot take_query_read_only_snapshot() noexcept {
    QueryReadOnlySnapshot snapshot{};
    snapshot.threadMutationActive = metadata_mutation_thread_active();
    snapshot.threadMutationDepth = metadata_mutation_thread_depth();
    snapshot.threadMutationTicket = metadata_mutation_thread_ticket();
    snapshot.threadMutationBeginCount = metadata_mutation_thread_begin_count();
    return snapshot;
}

bool query_read_only_snapshot_equal(
    const QueryReadOnlySnapshot& lhs,
    const QueryReadOnlySnapshot& rhs) noexcept {
    return lhs.threadMutationActive == rhs.threadMutationActive &&
        lhs.threadMutationDepth == rhs.threadMutationDepth &&
        lhs.threadMutationTicket == rhs.threadMutationTicket &&
        lhs.threadMutationBeginCount == rhs.threadMutationBeginCount;
}

const char* query_mutation_reason(
    const QueryReadOnlySnapshot& before,
    const QueryReadOnlySnapshot& after) noexcept {
    if (!before.threadMutationActive && after.threadMutationActive) {
        return "thread_entered_metadata_mutation";
    }
    if (before.threadMutationDepth != after.threadMutationDepth) {
        return "thread_mutation_depth_changed";
    }
    if (before.threadMutationTicket != after.threadMutationTicket) {
        return "thread_mutation_ticket_changed";
    }
    if (before.threadMutationBeginCount != after.threadMutationBeginCount) {
        return "thread_mutation_begin_count_changed";
    }
    return "unknown";
}

} // namespace

ResourceManagerState& global_state() noexcept {
    static ResourceManagerState state{};
    return state;
}

std::uint64_t foundation_observed_lease_generation(
    bool observeLease,
    std::uint64_t leaseGeneration) noexcept;

StaleInput state_build_stale_input(
    const SubmissionTransaction& transaction,
    LeaseObservationMode leaseObservationMode) noexcept {
    StaleInput staleInput{};
    staleInput.expectedRegistryGeneration = transaction.snapshot.registryGeneration;
    staleInput.expectedContextEpoch = transaction.snapshot.contextEpoch;
    RegistrySnapshotGenerations observedGenerations{};
    if (registry_get_snapshot_generations(
            transaction.snapshot.deviceContextKey,
            observedGenerations)) {
        staleInput.observedRegistryGeneration = observedGenerations.registryGeneration;
        staleInput.observedContextEpoch = observedGenerations.contextEpoch;
    }
    staleInput.expectedLeaseGeneration = transaction.leaseGeneration;
    const bool observeLease = (leaseObservationMode == LeaseObservationMode::Always) || transaction.active;
    staleInput.observedLeaseGeneration =
        foundation_observed_lease_generation(observeLease, transaction.leaseGeneration);
    staleInput.keySchemaMismatch =
        !submission_key_schema_matches_contract(transaction.snapshot.keySchemaVersion);
    return staleInput;
}

QueryReadOnlyGuard::QueryReadOnlyGuard(
    const char* queryName,
    const DeviceContextKey* key) noexcept
    : _queryName(queryName)
    , _key(key)
    , _before(take_query_read_only_snapshot()) {
}

QueryReadOnlyGuard::~QueryReadOnlyGuard() noexcept {
    const QueryReadOnlySnapshot after = take_query_read_only_snapshot();
    if (query_read_only_snapshot_equal(_before, after)) {
        return;
    }

    const char* reason = query_mutation_reason(_before, after);
    telemetry_record_query_mutation_violation();
    telemetry_record_module_boundary_violation();
    telemetry_trace_query_mutation_violation(
        _queryName,
        _key,
        _before.threadMutationDepth,
        after.threadMutationDepth,
        _before.threadMutationTicket,
        after.threadMutationTicket,
        _before.threadMutationBeginCount,
        after.threadMutationBeginCount,
        reason);
}

void state_record_acquire_status_for_kind(ResourceKind kind, AcquireStatus status) noexcept {
    ResourceManagerState& state = global_state();
    const std::size_t kindIndex = resource_kind_index(kind);
    if (kindIndex >= kResourceKindCount) {
        telemetry_record_module_boundary_violation();
        return;
    }
    ResourceKindAcquireCounters& counters = state.acquireStatusByKind[kindIndex];
    switch (status) {
    case AcquireStatus::Hit:
        telemetry_counter_add(counters.hit, 1);
        break;
    case AcquireStatus::Miss:
        telemetry_counter_add(counters.miss, 1);
        break;
    case AcquireStatus::Busy:
        telemetry_counter_add(counters.busy, 1);
        break;
    case AcquireStatus::Exhausted:
        telemetry_counter_add(counters.exhausted, 1);
        break;
    case AcquireStatus::Error:
    default:
        telemetry_counter_add(counters.error, 1);
        break;
    }
}

void state_note_latest_snapshot(const SubmissionSnapshot& snapshot) noexcept {
    try {
        const std::uint64_t instanceToken = snapshot.instanceToken.value;
        if (instanceToken == 0 || snapshot.snapshotId == 0) {
            return;
        }

        const LatestSnapshotKey key{ instanceToken, snapshot.deviceContextKey };
        LatestSnapshotState& state = latest_snapshot_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        std::uint64_t& latest = state.bySubmissionKey[key];
        if (snapshot.snapshotId > latest) {
            latest = snapshot.snapshotId;
        }
    }
    catch (...) {
        JuicerLogging::discard_current_exception();
    }
}

bool state_snapshot_is_superseded(
    const SubmissionSnapshot& snapshot,
    std::uint64_t* outLatestSnapshotId) noexcept {
    if (outLatestSnapshotId) {
        *outLatestSnapshotId = 0;
    }
    try {

        const std::uint64_t instanceToken = snapshot.instanceToken.value;
        if (instanceToken == 0 || snapshot.snapshotId == 0) {
            return false;
        }

        const LatestSnapshotKey key{ instanceToken, snapshot.deviceContextKey };
        LatestSnapshotState& state = latest_snapshot_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        auto it = state.bySubmissionKey.find(key);
        if (it == state.bySubmissionKey.end()) {
            return false;
        }
        if (outLatestSnapshotId) {
            *outLatestSnapshotId = it->second;
        }
        return it->second > snapshot.snapshotId;
    }
    catch (...) {
        JuicerLogging::discard_current_exception();
        if (outLatestSnapshotId) {
            *outLatestSnapshotId = 0;
        }
        return false;
    }
}

void state_clear_latest_snapshot_for_context(const DeviceContextKey& key) noexcept {
    try {
        LatestSnapshotState& state = latest_snapshot_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        for (auto it = state.bySubmissionKey.begin(); it != state.bySubmissionKey.end();) {
            if (it->first.deviceContextKey == key) {
                it = state.bySubmissionKey.erase(it);
            }
            else {
                ++it;
            }
        }
    }
    catch (...) {
        JuicerLogging::discard_current_exception();
    }
}



void telemetry_record_begin_submission() noexcept {
    telemetry_counter_add(global_state().beginSubmissionCalls, 1);
}

void telemetry_record_acquire_plan() noexcept {
    telemetry_counter_add(global_state().acquirePlanCalls, 1);
}

void telemetry_record_commit_submission() noexcept {
    telemetry_counter_add(global_state().commitSubmissionCalls, 1);
}

void telemetry_record_rollback_submission() noexcept {
    telemetry_counter_add(global_state().rollbackSubmissionCalls, 1);
}

void telemetry_record_acquire_status(AcquireStatus status) noexcept {
    ResourceManagerState& state = global_state();
    switch (status) {
    case AcquireStatus::Hit:
        telemetry_counter_add(state.acquireStatusHit, 1);
        break;
    case AcquireStatus::Miss:
        telemetry_counter_add(state.acquireStatusMiss, 1);
        break;
    case AcquireStatus::Busy:
        telemetry_counter_add(state.acquireStatusBusy, 1);
        break;
    case AcquireStatus::Exhausted:
        telemetry_counter_add(state.acquireStatusExhausted, 1);
        break;
    case AcquireStatus::Error:
    default:
        telemetry_counter_add(state.acquireStatusError, 1);
        break;
    }
}

void telemetry_record_acquire_status_for_kind(ResourceKind kind, AcquireStatus status) noexcept {
    state_record_acquire_status_for_kind(kind, status);
}

void telemetry_record_trace_schema_mismatch() noexcept {
    telemetry_counter_add(global_state().traceSchemaMismatchEvents, 1);
}

void telemetry_record_forbidden_invalidation_edge() noexcept {
    telemetry_counter_add(global_state().forbiddenInvalidationEdges, 1);
}

void telemetry_record_module_boundary_violation() noexcept {
    telemetry_counter_add(global_state().moduleBoundaryViolations, 1);
}

void telemetry_record_query_mutation_violation() noexcept {
    telemetry_counter_add(global_state().queryMutationViolationEvents, 1);
}

void telemetry_record_frame_snapshot_mismatch() noexcept {
    telemetry_counter_add(global_state().frameSnapshotMismatchEvents, 1);
}

void telemetry_record_stale_tuple_hard_reject() noexcept {
    telemetry_counter_add(global_state().staleTupleHardRejects, 1);
}

void telemetry_record_metadata_mutation_begin() noexcept {
    telemetry_counter_add(global_state().metadataMutationBeginCalls, 1);
}

void telemetry_record_metadata_mutation_end() noexcept {
    telemetry_counter_add(global_state().metadataMutationEndCalls, 1);
}

void telemetry_record_metadata_mutation_reject() noexcept {
    telemetry_counter_add(global_state().metadataMutationRejects, 1);
}

void telemetry_record_metadata_mutation_order_violation() noexcept {
    telemetry_counter_add(global_state().metadataMutationOrderViolations, 1);
}

void telemetry_record_metadata_queue_enqueue() noexcept {
    telemetry_counter_add(global_state().metadataMutationQueueEnqueueCalls, 1);
}

void telemetry_record_metadata_queue_dequeue() noexcept {
    telemetry_counter_add(global_state().metadataMutationQueueDequeueCalls, 1);
}

void telemetry_record_metadata_queue_wait() noexcept {
    telemetry_counter_add(global_state().metadataMutationQueueWaitEvents, 1);
}

void telemetry_record_metadata_queue_backpressure() noexcept {
    telemetry_counter_add(global_state().metadataMutationQueueBackpressureEvents, 1);
}

void telemetry_record_metadata_queue_reject() noexcept {
    telemetry_counter_add(global_state().metadataMutationQueueRejects, 1);
}

void telemetry_note_metadata_queue_depth(std::uint64_t depth) noexcept {
    telemetry_counter_note_max(global_state().metadataMutationQueueMaxDepth, depth);
}

std::uint64_t telemetry_next_acquire_attempt_id() noexcept {
    ResourceManagerState& state = global_state();
    std::uint64_t id = state.nextAcquireAttemptId.fetch_add(1, std::memory_order_relaxed);
    if (id == 0) {
        id = state.nextAcquireAttemptId.fetch_add(1, std::memory_order_relaxed);
    }
    return id;
}

constexpr const char* kTraceTokenUnknown = "unknown";
#if JUICER_DIAGNOSTICS_COMPILED
constexpr const char* kTraceTokenUnspecified = "unspecified";
#endif

const char* trace_token_or(const char* value, const char* fallback) noexcept {
    if (value) {
        return value;
    }
    return fallback;
}

std::uint32_t foundation_bool_u32(bool value) noexcept {
    if (value) {
        return 1u;
    }
    return 0u;
}

std::uint64_t foundation_observed_lease_generation(
    bool observeLease,
    std::uint64_t leaseGeneration) noexcept {
    if (observeLease) {
        return leaseGeneration;
    }
    return 0;
}

std::string telemetry_trace_txn_snapshot_prefix(
    std::uint64_t transactionId,
    std::uint64_t snapshotId) {
    return std::string("transaction_id=") + std::to_string(transactionId)
        + " snapshot_id=" + std::to_string(snapshotId);
}

std::string telemetry_trace_txn_snapshot_schema_prefix(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion) {
    return telemetry_trace_txn_snapshot_prefix(transactionId, snapshotId)
        + " trace_schema=" + std::to_string(traceSchemaVersion);
}

std::string telemetry_trace_device_context_fields(const DeviceContextKey* key) {
    int deviceId = -1;
    std::uintptr_t contextBits = 0;
    if (key) {
        deviceId = key->deviceId;
        contextBits = reinterpret_cast<std::uintptr_t>(key->contextOpaque);
    }
    return std::string(" device_id=") + std::to_string(deviceId)
        + " context=" + std::to_string(contextBits);
}

std::string telemetry_trace_event_prefix(const char* eventName) {
    return std::string("event=") + trace_token_or(eventName, kTraceTokenUnknown);
}

std::string telemetry_trace_event_stage_prefix(const char* eventName, const char* stage) {
    return telemetry_trace_event_prefix(eventName)
        + " stage=" + trace_token_or(stage, kTraceTokenUnknown);
}

std::string telemetry_trace_phase_stage_prefix(const char* phase, const char* stage) {
    return std::string("phase=") + trace_token_or(phase, kTraceTokenUnknown)
        + " stage=" + trace_token_or(stage, kTraceTokenUnknown);
}

void telemetry_trace_schema_announcement(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion) noexcept {
#if JUICER_DIAGNOSTICS_COMPILED
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    const std::string msg =
        telemetry_trace_txn_snapshot_schema_prefix(transactionId, snapshotId, traceSchemaVersion) +
        " expected_schema=" + std::to_string(kTraceSchemaVersion);
    JTRACE("MSTRC", msg);
#endif
}

void telemetry_trace_schema_mismatch(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t observedTraceSchemaVersion) noexcept {
#if JUICER_DIAGNOSTICS_COMPILED
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    const std::string msg =
        telemetry_trace_txn_snapshot_prefix(transactionId, snapshotId) +
        " observed_schema=" + std::to_string(observedTraceSchemaVersion) +
        " expected_schema=" + std::to_string(kTraceSchemaVersion) +
        " reason=trace_schema_mismatch";
    JTRACE("MSTRC", msg);
#endif
}

void telemetry_trace_key_normalization(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    const KeyDigests& before,
    const KeyDigests& after) noexcept {
#if JUICER_DIAGNOSTICS_COMPILED
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    const std::string msg =
        telemetry_trace_txn_snapshot_schema_prefix(transactionId, snapshotId, traceSchemaVersion) +
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
#endif
}

void telemetry_trace_invalidation(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    const char* lane,
    const char* reason,
    std::uint64_t previousHash,
    std::uint64_t currentHash) noexcept {
#if JUICER_DIAGNOSTICS_COMPILED
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    const std::string msg =
        telemetry_trace_txn_snapshot_schema_prefix(transactionId, snapshotId, traceSchemaVersion) +
        " lane=" + trace_token_or(lane, kTraceTokenUnknown) +
        " previous_hash=" + std::to_string(previousHash) +
        " current_hash=" + std::to_string(currentHash) +
        " reason=" + trace_token_or(reason, kTraceTokenUnknown);
    JTRACE("MSINV", msg);
#endif
}

void telemetry_trace_dag_edge(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    const char* fromNode,
    const char* toNode,
    bool allowed,
    const char* reason) noexcept {
#if JUICER_DIAGNOSTICS_COMPILED
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    const std::string msg =
        telemetry_trace_txn_snapshot_schema_prefix(transactionId, snapshotId, traceSchemaVersion) +
        " from=" + trace_token_or(fromNode, kTraceTokenUnknown) +
        " to=" + trace_token_or(toNode, kTraceTokenUnknown) +
        " allowed=" + std::to_string(foundation_bool_u32(allowed)) +
        " reason=" + trace_token_or(reason, kTraceTokenUnspecified);
    JTRACE("MSDAG", msg);
#endif
}

void telemetry_trace_module_boundary_violation(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    const char* reason) noexcept {
#if JUICER_DIAGNOSTICS_COMPILED
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    const std::string msg =
        telemetry_trace_txn_snapshot_schema_prefix(transactionId, snapshotId, traceSchemaVersion) +
        " reason=" + trace_token_or(reason, kTraceTokenUnknown);
    JTRACE("MSCMD", msg);
#endif
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
#if JUICER_DIAGNOSTICS_COMPILED
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    const std::string msg =
        telemetry_trace_event_prefix("query_mutation_violation") +
        " query=" + trace_token_or(queryName, kTraceTokenUnknown) +
        " reason=" + trace_token_or(reason, kTraceTokenUnknown) +
        telemetry_trace_device_context_fields(key) +
        " before_thread_mutation_depth=" + std::to_string(beforeThreadMutationDepth) +
        " after_thread_mutation_depth=" + std::to_string(afterThreadMutationDepth) +
        " before_thread_mutation_ticket=" + std::to_string(beforeThreadMutationTicket) +
        " after_thread_mutation_ticket=" + std::to_string(afterThreadMutationTicket) +
        " before_thread_mutation_begin_count=" + std::to_string(beforeThreadMutationBeginCount) +
        " after_thread_mutation_begin_count=" + std::to_string(afterThreadMutationBeginCount);
    JTRACE("MSCMD", msg);
#endif
}

void telemetry_trace_frame_snapshot_mismatch(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    std::uint64_t frameToken,
    std::uint64_t expectedSnapshotId,
    std::uint64_t observedSnapshotId) noexcept {
#if JUICER_DIAGNOSTICS_COMPILED
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    const std::string msg =
        telemetry_trace_txn_snapshot_schema_prefix(transactionId, snapshotId, traceSchemaVersion) +
        " frame_token=" + std::to_string(frameToken) +
        " expected_snapshot_id=" + std::to_string(expectedSnapshotId) +
        " observed_snapshot_id=" + std::to_string(observedSnapshotId) +
        " reason=mixed_snapshot_id_for_frame";
    JTRACE("MSSNP", msg);
#endif
}

void telemetry_trace_stale_decision(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    const char* stage,
    const StaleInput& input,
    const StaleDecision& decision) noexcept {
#if JUICER_DIAGNOSTICS_COMPILED
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    const std::string msg =
        telemetry_trace_txn_snapshot_schema_prefix(transactionId, snapshotId, traceSchemaVersion) +
        " stage=" + trace_token_or(stage, kTraceTokenUnknown) +
        " expected_registry_generation=" + std::to_string(input.expectedRegistryGeneration) +
        " observed_registry_generation=" + std::to_string(input.observedRegistryGeneration) +
        " expected_context_epoch=" + std::to_string(input.expectedContextEpoch) +
        " observed_context_epoch=" + std::to_string(input.observedContextEpoch) +
        " expected_lease_generation=" + std::to_string(input.expectedLeaseGeneration) +
        " observed_lease_generation=" + std::to_string(input.observedLeaseGeneration) +
        " key_schema_mismatch=" + std::to_string(foundation_bool_u32(input.keySchemaMismatch)) +
        " hard_stale=" + std::to_string(foundation_bool_u32(decision.hardStale)) +
        " hard_miss=" + std::to_string(foundation_bool_u32(decision.hardMiss)) +
        " reason=" + to_cstr(decision.reason);
    JTRACE("MSSTL", msg);
#endif
}

void telemetry_trace_metadata_mutation(
    const char* phase,
    const char* stage,
    std::uint64_t sequence,
    bool accepted,
    std::uint64_t expectedSequence,
    const char* reason) noexcept {
#if JUICER_DIAGNOSTICS_COMPILED
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    const std::string msg =
        telemetry_trace_phase_stage_prefix(phase, stage) +
        " sequence=" + std::to_string(sequence) +
        " expected_sequence=" + std::to_string(expectedSequence) +
        " accepted=" + std::to_string(foundation_bool_u32(accepted)) +
        " reason=" + trace_token_or(reason, kTraceTokenUnspecified);
    JTRACE("MSMUT", msg);
#endif
}

void telemetry_trace_metadata_queue(
    const char* eventName,
    const char* stage,
    std::uint64_t ticket,
    std::uint64_t depth,
    std::uint64_t waitedMs,
    bool accepted,
    const char* reason) noexcept {
#if JUICER_DIAGNOSTICS_COMPILED
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    const std::string msg =
        telemetry_trace_event_stage_prefix(eventName, stage) +
        " ticket=" + std::to_string(ticket) +
        " depth=" + std::to_string(depth) +
        " waited_ms=" + std::to_string(waitedMs) +
        " accepted=" + std::to_string(foundation_bool_u32(accepted)) +
        " reason=" + trace_token_or(reason, kTraceTokenUnspecified);
    JTRACE("MSMQ", msg);
#endif
}

void telemetry_trace_acquire(
    std::uint64_t acquireId,
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    AcquireStatus finalStatus,
    const ResourcePlan& plan,
    bool hadPreviousSnapshot) noexcept {
#if JUICER_DIAGNOSTICS_COMPILED
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    std::string msg;
    msg.reserve(192u + (kResourceKindOrder.size() * 96u));
    msg =
        std::string("acquire_id=") + std::to_string(acquireId) +
        " transaction_id=" + std::to_string(transactionId) +
        " snapshot_id=" + std::to_string(snapshotId) +
        " trace_schema=" + std::to_string(traceSchemaVersion) +
        " final_status=" + to_cstr(finalStatus) +
        " had_previous=" + std::to_string(foundation_bool_u32(hadPreviousSnapshot));
    for (ResourceKind kind : kResourceKindOrder) {
        const ResourcePlanEntry& entry = resource_plan_entry(plan, kind);
        const ResourceKindContractEntry& contract = resource_kind_contract_entry(kind);
        msg += std::string(" ") + contract.acquireStatusField + "=" + to_cstr(entry.acquire.status);
        msg += std::string(" ") + contract.invalidationLane + "_invalidated=" +
            std::to_string(foundation_bool_u32(entry.invalidated));
    }
    JTRACE("MSACQ", msg);
#endif
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
#if JUICER_DIAGNOSTICS_COMPILED
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    const std::string msg =
        telemetry_trace_txn_snapshot_schema_prefix(transactionId, snapshotId, traceSchemaVersion) +
        " mode=" + trace_token_or(mode, kTraceTokenUnknown) +
        " event=" + trace_token_or(eventName, kTraceTokenUnknown) +
        " hit=" + std::to_string(foundation_bool_u32(hit)) +
        " key_hash=" + std::to_string(keyHash) +
        " meter_w=" + std::to_string(meterWidth) +
        " meter_h=" + std::to_string(meterHeight) +
        " had_previous=" + std::to_string(foundation_bool_u32(hadPrevious)) +
        " reason=" + trace_token_or(reason, kTraceTokenUnspecified);
    JTRACE("MSAEX", msg);
#endif
}


// Split implementation sections (single-TU include model to preserve exact behavior while
// reducing monolithic file size and keeping ownership boundaries explicit).
#include "Cuda/ResourceManager/JuicerCudaResourceManagerAdmission.inc"
#include "Cuda/ResourceManager/JuicerCudaResourceManagerSubmission.inc"
#include "Cuda/ResourceManager/JuicerCudaResourceManagerCommands.inc"

} // namespace ResourceManager
} // namespace JuicerCuda
