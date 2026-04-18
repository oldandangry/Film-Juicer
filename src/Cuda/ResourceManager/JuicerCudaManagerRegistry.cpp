// Cuda/ResourceManager/JuicerCudaManagerRegistry.cpp

#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#include "Cuda/ResourceManager/JuicerCudaResourceCore.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "Logging.h"

namespace JuicerCuda {
namespace ResourceManager {
namespace {

struct RegistryEntry {
    RegistryHandle handle{};
    ContextLifecycleState lifecycleState = ContextLifecycleState::Unbound;
    std::uint64_t lifecycleSinceMs = 0;
    std::uint64_t createOrder = 0;
    std::uint64_t lastTouchedMs = 0;
    std::uint64_t activeSubmissionCount = 0;
};

struct RegistryState {
    std::mutex mutex;
    std::atomic<std::uint64_t> nextHandle{ 1 };
    std::atomic<std::uint64_t> nextCreateOrder{ 1 };
    std::uint64_t lastIdleReapScanMs = 0;
    std::unordered_map<DeviceContextKey, RegistryEntry, DeviceContextKeyHash> byDeviceContext;
    std::unordered_map<std::uint64_t, DeviceContextKey> keyByHandle;
};

struct ReapCandidate {
    DeviceContextKey key{};
    RegistryEntry entry{};
    std::uint64_t idleMs = 0;
};

constexpr std::uint64_t kMetadataMutationQueueDepthLimit = 64ull;
thread_local std::uint32_t gMutationThreadDepth = 0;
thread_local std::uint64_t gMutationThreadTicket = 0;
thread_local std::uint64_t gMutationThreadBeginCount = 0;

struct MetadataMutationLane {
    std::mutex mutex;
    std::condition_variable cv;
    std::uint64_t nextTicket = 1;
    std::uint64_t servingTicket = 1;
    std::uint64_t activeTicket = 0;
    std::thread::id ownerThread{};
    std::uint32_t ownerDepth = 0;
};

struct MetadataMutationLaneDirectory {
    std::mutex mutex;
    std::shared_ptr<MetadataMutationLane> fallbackLane = std::make_shared<MetadataMutationLane>();
    std::unordered_map<DeviceContextKey, std::shared_ptr<MetadataMutationLane>, DeviceContextKeyHash> byManagerKey;
};

struct LaneAcquireResult {
    std::uint64_t ticket = 0;
    std::uint64_t waitedMs = 0;
    bool observedWait = false;
    bool observedBackpressure = false;
    bool reentrant = false;
};

MetadataMutationLaneDirectory& mutation_lane_directory() {
    static MetadataMutationLaneDirectory directory{};
    return directory;
}

std::mutex& mutation_sequence_mutex() {
    static std::mutex m;
    return m;
}

std::uint64_t& mutation_last_issued_sequence() {
    static std::uint64_t sequence = 0;
    return sequence;
}

inline std::uint64_t lane_depth_nolock(const MetadataMutationLane& lane) noexcept {
    if (lane.nextTicket < lane.servingTicket) {
        return 0;
    }
    return lane.nextTicket - lane.servingTicket;
}

inline std::uint64_t elapsed_ms(const std::chrono::steady_clock::time_point& start) noexcept {
    const auto delta = std::chrono::steady_clock::now() - start;
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(delta).count());
}

const char* registry_trace_or_unknown(const char* value) noexcept {
    if (value && value[0] != '\0') {
        return value;
    }
    return "unknown";
}

const char* registry_trace_or_unspecified(const char* value) noexcept {
    if (value && value[0] != '\0') {
        return value;
    }
    return "unspecified";
}

const char* registry_trace_or(const char* value, const char* fallback) noexcept {
    if (value) {
        return value;
    }
    return fallback;
}

const char* registry_bool_reason(bool value, const char* whenTrue, const char* whenFalse) noexcept {
    if (value) {
        return whenTrue;
    }
    return whenFalse;
}

std::uint32_t registry_bool_u32(bool value) noexcept {
    if (value) {
        return 1u;
    }
    return 0u;
}

const char* registry_mutation_begin_reason(bool orderOk, bool reentrant) noexcept {
    if (!orderOk) {
        return "sequence_order_violation";
    }
    if (reentrant) {
        return "ok_reentrant";
    }
    return "ok";
}

const char* registry_entry_lifecycle_name(const RegistryEntry* entry) noexcept {
    if (entry) {
        return to_cstr(entry->lifecycleState);
    }
    return "Missing";
}

int registry_device_id_or_default(const DeviceContextKey* key) noexcept {
    if (key) {
        return key->deviceId;
    }
    return -1;
}

std::uintptr_t registry_context_bits_or_zero(const DeviceContextKey* key) noexcept {
    if (key) {
        return reinterpret_cast<std::uintptr_t>(key->contextOpaque);
    }
    return 0;
}

std::string registry_trace_event_prefix(const char* eventName) {
    return std::string("event=") + registry_trace_or_unknown(eventName);
}

std::string registry_trace_device_context_fields(const DeviceContextKey* key) {
    const int deviceId = registry_device_id_or_default(key);
    const std::uintptr_t contextBits = registry_context_bits_or_zero(key);
    return std::string(" device_id=") + std::to_string(deviceId)
        + " context=" + std::to_string(contextBits);
}

std::string registry_trace_handle_prefix(RegistryHandle handle, const DeviceContextKey& key) {
    return std::string("handle=") + std::to_string(handle.value)
        + registry_trace_device_context_fields(&key);
}

const char* registry_retire_reason_name(RegistryRetireReason reason) noexcept {
    switch (reason) {
    case RegistryRetireReason::ContextReset:
        return "context_reset";
    case RegistryRetireReason::Idle:
        return "idle";
    default:
        return "unknown";
    }
}

const char* registry_lifecycle_timeout_reason(ContextLifecycleState observedState) noexcept {
    if (observedState == ContextLifecycleState::Draining) {
        return "drain_timeout";
    }
    return "rebind_timeout";
}

std::shared_ptr<MetadataMutationLane> resolve_mutation_lane(const DeviceContextKey* managerKey) {
    MetadataMutationLaneDirectory& directory = mutation_lane_directory();
    if (!managerKey || managerKey->deviceId < 0) {
        return directory.fallbackLane;
    }

    std::lock_guard<std::mutex> lock(directory.mutex);
    auto it = directory.byManagerKey.find(*managerKey);
    if (it != directory.byManagerKey.end()) {
        return it->second;
    }

    std::shared_ptr<MetadataMutationLane> lane = std::make_shared<MetadataMutationLane>();
    directory.byManagerKey.emplace(*managerKey, lane);
    return lane;
}

void reset_scope(MetadataMutationScope& scope) noexcept {
    scope.sequence = 0;
    scope.queueTicket = 0;
    scope.managerKey = DeviceContextKey{};
    scope.hasManagerKey = false;
    scope.active = false;
}

void reset_thread_mutation_context() noexcept {
    gMutationThreadDepth = 0;
    gMutationThreadTicket = 0;
}

DeviceContextKey registry_manager_key_or_default(
    bool hasManagerKey,
    const DeviceContextKey* managerKey) noexcept {
    if (hasManagerKey && managerKey) {
        return *managerKey;
    }
    return DeviceContextKey{};
}

const DeviceContextKey* registry_manager_key_ptr_or_null(
    bool hasManagerKey,
    const DeviceContextKey* managerKey) noexcept {
    if (hasManagerKey) {
        return managerKey;
    }
    return nullptr;
}

void trace_mutation_reject(
    const char* action,
    const char* stage,
    std::uint64_t sequence,
    std::uint64_t expectedSequence,
    std::uint64_t queueTicket,
    std::uint64_t queueDepth,
    std::uint64_t waitedMs,
    bool accepted,
    const char* reason) noexcept {
    telemetry_record_metadata_mutation_reject();
    telemetry_record_metadata_queue_reject();
    telemetry_trace_metadata_queue(
        "reject",
        stage,
        queueTicket,
        queueDepth,
        waitedMs,
        accepted,
        reason);
    telemetry_trace_metadata_mutation(
        action,
        stage,
        sequence,
        false,
        expectedSequence,
        reason);
}

void note_queue_wait_if_needed(bool& observedWait) noexcept {
    if (!observedWait) {
        observedWait = true;
        telemetry_record_metadata_queue_wait();
    }
}

bool acquire_lane_ticket(
    MetadataMutationLane& lane,
    const std::thread::id& currentThread,
    const char* stage,
    LaneAcquireResult& out) {
    std::unique_lock<std::mutex> lock(lane.mutex);
    if (lane.ownerDepth > 0 && lane.ownerThread == currentThread) {
        out.reentrant = true;
        ++lane.ownerDepth;
        out.ticket = lane.activeTicket;
        if (out.ticket == 0) {
            out.ticket = lane.servingTicket;
            lane.activeTicket = out.ticket;
        }
        telemetry_trace_metadata_queue(
            "reenter",
            stage,
            out.ticket,
            lane_depth_nolock(lane),
            0,
            true,
            "owner_reentrant");
        return true;
    }

    while (lane_depth_nolock(lane) >= kMetadataMutationQueueDepthLimit) {
        if (!out.observedBackpressure) {
            out.observedBackpressure = true;
            telemetry_record_metadata_queue_backpressure();
            telemetry_trace_metadata_queue(
                "backpressure",
                stage,
                0,
                lane_depth_nolock(lane),
                out.waitedMs,
                true,
                "depth_limit");
        }
        note_queue_wait_if_needed(out.observedWait);
        // Event-driven wait avoids the prior 1 ms polling loop on queue pressure paths.
        const auto waitStart = std::chrono::steady_clock::now();
        lane.cv.wait(lock, [&lane]() {
            return lane_depth_nolock(lane) < kMetadataMutationQueueDepthLimit;
        });
        out.waitedMs += elapsed_ms(waitStart);
    }

    out.ticket = lane.nextTicket++;
    if (out.ticket == 0) {
        trace_mutation_reject(
            "begin",
            stage,
            0,
            0,
            0,
            lane_depth_nolock(lane),
            out.waitedMs,
            false,
            "ticket_overflow");
        return false;
    }

    telemetry_record_metadata_queue_enqueue();
    const std::uint64_t depthAfterEnqueue = lane_depth_nolock(lane);
    telemetry_note_metadata_queue_depth(depthAfterEnqueue);
    telemetry_trace_metadata_queue(
        "enqueue",
        stage,
        out.ticket,
        depthAfterEnqueue,
        out.waitedMs,
        true,
        registry_bool_reason(out.observedBackpressure, "accepted_after_backpressure", "accepted"));

    while (out.ticket != lane.servingTicket || lane.ownerDepth != 0) {
        note_queue_wait_if_needed(out.observedWait);
        const auto waitStart = std::chrono::steady_clock::now();
        lane.cv.wait(lock, [&lane, &out]() {
            return out.ticket == lane.servingTicket && lane.ownerDepth == 0;
        });
        out.waitedMs += elapsed_ms(waitStart);
    }

    lane.ownerThread = currentThread;
    lane.ownerDepth = 1;
    lane.activeTicket = out.ticket;
    telemetry_record_metadata_queue_dequeue();
    telemetry_trace_metadata_queue(
        "dequeue",
        stage,
        out.ticket,
        lane_depth_nolock(lane),
        out.waitedMs,
        true,
        registry_bool_reason(out.observedWait, "turn_wait", "immediate"));
    return true;
}

RegistryState& registry_state() {
    static RegistryState state{};
    return state;
}

const ResourceManagerConfigEffective& registry_policy_config() noexcept {
    static const ResourceManagerConfigEffective cfg = sanitize_config(ResourceManagerConfigRaw{});
    return cfg;
}

inline std::uint64_t monotonic_time_ms() noexcept {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

inline std::uint64_t saturating_elapsed_ms(std::uint64_t nowMs, std::uint64_t thenMs) noexcept {
    if (nowMs >= thenMs) {
        return nowMs - thenMs;
    }
    return 0ull;
}

void publish_registry_live_count(std::size_t count) noexcept {
    global_state().registryLiveManagers.store(static_cast<std::uint64_t>(count), std::memory_order_relaxed);
}

inline std::uint64_t next_nonzero_counter(std::atomic<std::uint64_t>& counter) noexcept {
    std::uint64_t value = counter.fetch_add(1, std::memory_order_relaxed);
    if (value == 0) {
        value = counter.fetch_add(1, std::memory_order_relaxed);
    }
    return value;
}

std::uint64_t registry_entry_handle_value_or_zero(const RegistryEntry* entry) noexcept {
    if (entry) {
        return entry->handle.value;
    }
    return 0;
}

std::uint64_t registry_entry_active_submissions_or_zero(const RegistryEntry* entry) noexcept {
    if (entry) {
        return entry->activeSubmissionCount;
    }
    return 0;
}

void trace_registry_event(const DeviceContextKey* key,
                          const RegistryEntry* entry,
                          const char* eventName,
                          bool accepted,
                          const char* reason,
                          std::uint64_t idleMs,
                          std::uint64_t liveManagers,
                          std::uint64_t maxLiveManagers,
                          std::uint64_t reapEvents) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }
    const std::uint64_t handleValue = registry_entry_handle_value_or_zero(entry);
    const char* lifecycle = registry_entry_lifecycle_name(entry);
    const std::uint64_t activeSubmissions = registry_entry_active_submissions_or_zero(entry);
    const std::string msg =
        registry_trace_event_prefix(eventName) +
        " accepted=" + std::to_string(registry_bool_u32(accepted)) +
        " reason=" + registry_trace_or_unspecified(reason) +
        " handle=" + std::to_string(handleValue) +
        registry_trace_device_context_fields(key) +
        " lifecycle=" + lifecycle +
        " active_submissions=" + std::to_string(activeSubmissions) +
        " idle_ms=" + std::to_string(idleMs) +
        " live_managers=" + std::to_string(liveManagers) +
        " max_live_managers=" + std::to_string(maxLiveManagers) +
        " reap_events=" + std::to_string(reapEvents);
    JTRACE_LEVEL(2, "MSREG", msg);
}

void trace_registry_event_current(
    const DeviceContextKey* key,
    const RegistryEntry* entry,
    const char* eventName,
    bool accepted,
    const char* reason,
    std::uint64_t idleMs) noexcept {
    ResourceManagerState& rmState = global_state();
    trace_registry_event(
        key,
        entry,
        eventName,
        accepted,
        reason,
        idleMs,
        rmState.registryLiveManagers.load(std::memory_order_relaxed),
        static_cast<std::uint64_t>(registry_policy_config().maxLiveManagersPerProcess),
        rmState.registryReapEvents.load(std::memory_order_relaxed));
}

void trace_registry_missing_entry(const DeviceContextKey& key, const char* eventName) noexcept {
    trace_registry_event_current(
        &key,
        nullptr,
        eventName,
        false,
        "missing_registry_entry",
        0);
}

void trace_reap_skip_event(const char* reason) noexcept {
    trace_registry_event_current(
        nullptr,
        nullptr,
        "reap_skip",
        false,
        reason,
        0);
}

bool is_legal_transition(ContextLifecycleState from, ContextLifecycleState to) noexcept {
    if (from == to) {
        return true;
    }
    switch (from) {
    case ContextLifecycleState::Unbound:
        return to == ContextLifecycleState::Binding || to == ContextLifecycleState::Retired;
    case ContextLifecycleState::Binding:
        return to == ContextLifecycleState::Active || to == ContextLifecycleState::Retired;
    case ContextLifecycleState::Active:
        return to == ContextLifecycleState::Freezing || to == ContextLifecycleState::Retired;
    case ContextLifecycleState::Freezing:
        return to == ContextLifecycleState::Draining ||
            to == ContextLifecycleState::Active ||
            to == ContextLifecycleState::Retired;
    case ContextLifecycleState::Draining:
        return to == ContextLifecycleState::Rebinding || to == ContextLifecycleState::Retired;
    case ContextLifecycleState::Rebinding:
        return to == ContextLifecycleState::Active || to == ContextLifecycleState::Retired;
    case ContextLifecycleState::Retired:
    default:
        return false;
    }
}

void trace_lifecycle_transition(const DeviceContextKey& key,
                                RegistryHandle handle,
                                ContextLifecycleState from,
                                ContextLifecycleState to,
                                bool accepted,
                                const char* reason) {
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    const std::string msg = registry_trace_handle_prefix(handle, key)
        + " from=" + to_cstr(from)
        + " to=" + to_cstr(to)
        + " accepted=" + std::to_string(registry_bool_u32(accepted))
        + " reason=" + registry_trace_or_unspecified(reason);
    JTRACE("MSLCY", msg);
}

void trace_lifecycle_bump(const DeviceContextKey& key,
                          RegistryHandle handle,
                          std::uint64_t previousRegistryGeneration,
                          std::uint64_t newRegistryGeneration,
                          std::uint64_t previousContextEpoch,
                          std::uint64_t newContextEpoch,
                          bool accepted,
                          const char* reason) {
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    const std::string msg = registry_trace_handle_prefix(handle, key)
        + " action=bump"
        + " accepted=" + std::to_string(registry_bool_u32(accepted))
        + " prev_registry_generation=" + std::to_string(previousRegistryGeneration)
        + " new_registry_generation=" + std::to_string(newRegistryGeneration)
        + " prev_context_epoch=" + std::to_string(previousContextEpoch)
        + " new_context_epoch=" + std::to_string(newContextEpoch)
        + " reason=" + registry_trace_or_unspecified(reason);
    JTRACE("MSLCY", msg);
}

constexpr std::uint64_t kLifecycleTimeoutMinMs = 2000ull;
constexpr std::uint64_t kRebindingTimeoutMinMs = 1000ull;

bool lifecycle_state_allowed_for_stage(ContextLifecycleState state, bool allowNonActiveRelease) noexcept {
    if (allowNonActiveRelease) {
        return state != ContextLifecycleState::Unbound;
    }
    return state == ContextLifecycleState::Active;
}

std::uint64_t lifecycle_timeout_ms_for_state(
    ContextLifecycleState state,
    const ResourceManagerConfigEffective& cfg) noexcept {
    const std::uint64_t baseMs = std::max<std::uint64_t>(
        static_cast<std::uint64_t>(cfg.managerIdleReapMs),
        kLifecycleTimeoutMinMs);
    switch (state) {
    case ContextLifecycleState::Draining:
        return baseMs;
    case ContextLifecycleState::Rebinding:
        return std::max<std::uint64_t>(baseMs / 2ull, kRebindingTimeoutMinMs);
    default:
        return 0ull;
    }
}

void bump_registry_epoch_locked(
    const DeviceContextKey& key,
    RegistryHandle handle,
    bool accepted,
    const char* reason) noexcept {
    ResourceManagerState& rmState = global_state();
    const std::uint64_t prevRegistryGeneration =
        rmState.registryGeneration.fetch_add(1, std::memory_order_relaxed);
    std::uint64_t newRegistryGeneration = prevRegistryGeneration + 1;
    if (newRegistryGeneration == 0) {
        newRegistryGeneration = 1;
        rmState.registryGeneration.store(newRegistryGeneration, std::memory_order_relaxed);
    }
    const std::uint64_t prevContextEpoch =
        rmState.contextEpoch.fetch_add(1, std::memory_order_relaxed);
    std::uint64_t newContextEpoch = prevContextEpoch + 1;
    if (newContextEpoch == 0) {
        newContextEpoch = 1;
        rmState.contextEpoch.store(newContextEpoch, std::memory_order_relaxed);
    }
    trace_lifecycle_bump(
        key,
        handle,
        prevRegistryGeneration,
        newRegistryGeneration,
        prevContextEpoch,
        newContextEpoch,
        accepted,
        reason);
}

void trace_lifecycle_timeout(
    const DeviceContextKey& key,
    RegistryHandle handle,
    ContextLifecycleState observedState,
    std::uint64_t stateAgeMs,
    std::uint64_t timeoutMs,
    bool escalated,
    const char* reason) noexcept {
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    const std::string msg = registry_trace_handle_prefix(handle, key)
        + " action=watchdog_timeout"
        + " observed_state=" + to_cstr(observedState)
        + " state_age_ms=" + std::to_string(stateAgeMs)
        + " timeout_ms=" + std::to_string(timeoutMs)
        + " escalated=" + std::to_string(registry_bool_u32(escalated))
        + " reason=" + registry_trace_or_unspecified(reason);
    JTRACE("MSLCY", msg);
}

bool transition_entry_locked(const DeviceContextKey& key,
                             RegistryEntry& entry,
                             ContextLifecycleState expectedState,
                             ContextLifecycleState desiredState,
                             const char* reason) {
    ResourceManagerState& rmState = global_state();
    telemetry_counter_add(rmState.lifecycleTransitionCalls, 1);
    const ContextLifecycleState observed = entry.lifecycleState;
    const bool expectedOk = (observed == expectedState);
    const bool legal = expectedOk && is_legal_transition(observed, desiredState);
    if (!legal) {
        telemetry_counter_add(rmState.lifecycleTransitionRejects, 1);
        trace_lifecycle_transition(key, entry.handle, observed, desiredState, false, reason);
        return false;
    }
    entry.lifecycleState = desiredState;
    entry.lifecycleSinceMs = monotonic_time_ms();
    trace_lifecycle_transition(key, entry.handle, observed, desiredState, true, reason);
    return true;
}

bool transition_entry_to_retired_locked(const DeviceContextKey& key,
                                        RegistryEntry& entry,
                                        const char* reason) {
    if (entry.lifecycleState == ContextLifecycleState::Retired) {
        return true;
    }
    const ContextLifecycleState observed = entry.lifecycleState;
    return transition_entry_locked(
        key,
        entry,
        observed,
        ContextLifecycleState::Retired,
        reason);
}

ContextLifecycleState desired_retire_state(RegistryRetireReason reason) noexcept {
    (void)reason;
    return ContextLifecycleState::Retired;
}

bool erase_registry_entry_locked(RegistryState& state,
                                 const DeviceContextKey& key,
                                 const RegistryEntry& entry,
                                 const char* eventName,
                                 const char* reason,
                                 bool countReapEvent,
                                 std::uint64_t idleMs) noexcept {
    const std::uint64_t handleValue = entry.handle.value;
    auto mapIt = state.byDeviceContext.find(key);
    if (mapIt == state.byDeviceContext.end()) {
        return false;
    }
    state.byDeviceContext.erase(mapIt);
    if (handleValue != 0) {
        state.keyByHandle.erase(handleValue);
    }

    ResourceManagerState& rmState = global_state();
    publish_registry_live_count(state.byDeviceContext.size());
    if (countReapEvent) {
        telemetry_counter_add(rmState.registryReapEvents, 1);
    }
    trace_registry_event_current(
        &key,
        &entry,
        eventName,
        true,
        reason,
        idleMs);
    return true;
}

void collect_reap_candidates_locked(
    const RegistryState& state,
    const DeviceContextKey* protectKey,
    std::uint64_t nowMs,
    std::uint64_t idleReapMs,
    std::vector<ReapCandidate>& outCandidates) {
    outCandidates.clear();
    outCandidates.reserve(state.byDeviceContext.size());
    for (const auto& kv : state.byDeviceContext) {
        const DeviceContextKey& candidateKey = kv.first;
        const RegistryEntry& entry = kv.second;
        if (protectKey && candidateKey == *protectKey) {
            continue;
        }
        if (entry.activeSubmissionCount != 0) {
            continue;
        }
        if (entry.lifecycleState != ContextLifecycleState::Active &&
            entry.lifecycleState != ContextLifecycleState::Retired) {
            continue;
        }

        const std::uint64_t idleMs = saturating_elapsed_ms(nowMs, entry.lastTouchedMs);
        const bool eligible = (entry.lifecycleState == ContextLifecycleState::Retired) ||
            (idleMs >= idleReapMs);
        if (!eligible) {
            continue;
        }

        ReapCandidate candidate{};
        candidate.key = candidateKey;
        candidate.entry = entry;
        candidate.idleMs = idleMs;
        outCandidates.push_back(candidate);
    }
}

void sort_reap_candidates(std::vector<ReapCandidate>& candidates) {
    std::sort(candidates.begin(), candidates.end(), [](const ReapCandidate& lhs, const ReapCandidate& rhs) {
        if (lhs.entry.lifecycleState != rhs.entry.lifecycleState) {
            if (lhs.entry.lifecycleState == ContextLifecycleState::Retired) {
                return true;
            }
            if (rhs.entry.lifecycleState == ContextLifecycleState::Retired) {
                return false;
            }
        }
        if (lhs.idleMs != rhs.idleMs) {
            return lhs.idleMs > rhs.idleMs;
        }
        if (lhs.entry.createOrder != rhs.entry.createOrder) {
            return lhs.entry.createOrder < rhs.entry.createOrder;
        }
        return lhs.entry.handle.value < rhs.entry.handle.value;
    });
}

std::size_t compute_reap_count(std::size_t candidateCount, std::uint64_t overflow) noexcept {
    if (overflow == 0) {
        return candidateCount;
    }
    const std::size_t overflowCount = static_cast<std::size_t>(
        std::min<std::uint64_t>(overflow, static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())));
    return std::min(candidateCount, overflowCount);
}

std::uint64_t registry_overflow_count_or_zero(
    std::uint64_t liveManagersBefore,
    std::uint64_t maxLive) noexcept {
    if (liveManagersBefore > maxLive) {
        return liveManagersBefore - maxLive;
    }
    return 0ull;
}

void maybe_reap_idle_locked(RegistryState& state, const DeviceContextKey* protectKey) noexcept {
    const ResourceManagerConfigEffective& cfg = registry_policy_config();
    if (cfg.maxLiveManagersPerProcess == 0 || cfg.managerIdleReapMs == 0) {
        return;
    }
    if (state.byDeviceContext.empty()) {
        publish_registry_live_count(0);
        return;
    }

    const std::uint64_t nowMs = monotonic_time_ms();
    const std::uint64_t elapsedMs = saturating_elapsed_ms(nowMs, state.lastIdleReapScanMs);
    const bool cadenceDue = (state.lastIdleReapScanMs == 0) ||
        (elapsedMs >= static_cast<std::uint64_t>(cfg.managerIdleReapMs));
    const std::uint64_t liveManagersBefore = static_cast<std::uint64_t>(state.byDeviceContext.size());
    publish_registry_live_count(state.byDeviceContext.size());
    const std::uint64_t maxLive = static_cast<std::uint64_t>(cfg.maxLiveManagersPerProcess);
    const std::uint64_t overflow = registry_overflow_count_or_zero(liveManagersBefore, maxLive);
    if (!cadenceDue && overflow == 0) {
        return;
    }
    state.lastIdleReapScanMs = nowMs;

    std::vector<ReapCandidate> candidates;
    collect_reap_candidates_locked(
        state,
        protectKey,
        nowMs,
        static_cast<std::uint64_t>(cfg.managerIdleReapMs),
        candidates);

    if (candidates.empty()) {
        if (overflow > 0) {
            trace_reap_skip_event("overflow_no_eligible_idle_manager");
        }
        return;
    }

    sort_reap_candidates(candidates);
    const std::size_t reapCount = compute_reap_count(candidates.size(), overflow);

    for (std::size_t i = 0; i < reapCount; ++i) {
        const ReapCandidate& candidate = candidates[i];
        const char* reason = registry_bool_reason(overflow > 0, "max_live_oldest_idle", "idle_timeout");
        (void)erase_registry_entry_locked(
            state,
            candidate.key,
            candidate.entry,
            "reap",
            reason,
            true,
            candidate.idleMs);
    }

    if (overflow > static_cast<std::uint64_t>(reapCount)) {
        trace_reap_skip_event("overflow_remaining_after_safe_reap");
    }
}

bool run_freeze_drain_bump_resume_locked(const DeviceContextKey& key,
                                         RegistryEntry& entry,
                                         const char* reason) {
    ResourceManagerState& rmState = global_state();
    telemetry_counter_add(rmState.lifecycleBarrierCalls, 1);

    bool ok = true;
    ok = ok && transition_entry_locked(
        key, entry, ContextLifecycleState::Active, ContextLifecycleState::Freezing, "barrier_freeze");
    ok = ok && transition_entry_locked(
        key, entry, ContextLifecycleState::Freezing, ContextLifecycleState::Draining, "barrier_drain");

    bump_registry_epoch_locked(
        key,
        entry.handle,
        ok,
        registry_trace_or(reason, "barrier_bump"));

    ok = ok && transition_entry_locked(
        key, entry, ContextLifecycleState::Draining, ContextLifecycleState::Rebinding, "barrier_rebind");
    ok = ok && transition_entry_locked(
        key, entry, ContextLifecycleState::Rebinding, ContextLifecycleState::Active, "barrier_resume");

    if (!ok) {
        telemetry_counter_add(rmState.lifecycleBarrierRejects, 1);
    }
    return ok;
}

} // namespace

bool metadata_mutation_begin(
    const char* stage,
    MetadataMutationScope& outScope,
    const DeviceContextKey* managerKey) noexcept {
    // Lock-order invariant:
    // - Resolve lane (directory mutex), then acquire/release lane mutex for ticketing.
    // - Sequence mutex is acquired only after lane mutex is released.
    // - Do not hold registry_state().mutex while touching lane/directory/sequence mutexes.
    if (outScope.active) {
        trace_mutation_reject(
            "begin",
            stage,
            outScope.sequence,
            outScope.sequence,
            outScope.queueTicket,
            0,
            0,
            false,
            "scope_already_active");
        return false;
    }

    std::shared_ptr<MetadataMutationLane> lane = resolve_mutation_lane(managerKey);
    if (!lane) {
        trace_mutation_reject(
            "begin",
            stage,
            0,
            0,
            0,
            0,
            0,
            false,
            "missing_lane");
        return false;
    }

    LaneAcquireResult acquire{};
    if (!acquire_lane_ticket(*lane, std::this_thread::get_id(), stage, acquire)) {
        return false;
    }

    telemetry_record_metadata_mutation_begin();
    std::uint64_t sequence = 0;
    std::uint64_t expectedSequence = 0;
    bool orderOk = true;
    {
        std::lock_guard<std::mutex> lock(mutation_sequence_mutex());
        ResourceManagerState& state = global_state();
        sequence = state.nextMetadataMutationSequence.fetch_add(1, std::memory_order_relaxed);
        if (sequence == 0) {
            sequence = state.nextMetadataMutationSequence.fetch_add(1, std::memory_order_relaxed);
        }
        expectedSequence = mutation_last_issued_sequence() + 1;
        if (expectedSequence == 0) {
            expectedSequence = 1;
        }
        orderOk = (sequence == expectedSequence);
        if (!orderOk) {
            telemetry_record_metadata_mutation_order_violation();
        }
        mutation_last_issued_sequence() = sequence;
    }

    outScope.sequence = sequence;
    outScope.queueTicket = acquire.ticket;
    outScope.hasManagerKey = (managerKey && managerKey->deviceId >= 0);
    outScope.managerKey = registry_manager_key_or_default(outScope.hasManagerKey, managerKey);
    outScope.active = true;
    gMutationThreadDepth += 1;
    gMutationThreadTicket = acquire.ticket;
    gMutationThreadBeginCount += 1;

    telemetry_trace_metadata_mutation(
        "begin",
        stage,
        sequence,
        true,
        expectedSequence,
        registry_mutation_begin_reason(orderOk, acquire.reentrant));
    return true;
}

void metadata_mutation_end(MetadataMutationScope& scope, const char* stage) noexcept {
    if (!scope.active) {
        trace_mutation_reject(
            "end",
            stage,
            scope.sequence,
            scope.sequence,
            scope.queueTicket,
            0,
            0,
            false,
            "scope_not_active");
        return;
    }

    std::shared_ptr<MetadataMutationLane> lane =
        resolve_mutation_lane(registry_manager_key_ptr_or_null(scope.hasManagerKey, &scope.managerKey));
    if (!lane) {
        trace_mutation_reject(
            "end",
            stage,
            scope.sequence,
            scope.sequence,
            scope.queueTicket,
            0,
            0,
            false,
            "missing_lane");
        reset_thread_mutation_context();
        reset_scope(scope);
        return;
    }

    bool releaseTopLevel = false;
    std::uint64_t depthAfterRelease = 0;
    const std::thread::id currentThread = std::this_thread::get_id();
    {
        std::lock_guard<std::mutex> lock(lane->mutex);
        if (lane->ownerDepth == 0 || lane->ownerThread != currentThread) {
            trace_mutation_reject(
                "end",
                stage,
                scope.sequence,
                scope.sequence,
                scope.queueTicket,
                lane_depth_nolock(*lane),
                0,
                false,
                "owner_mismatch");
            reset_thread_mutation_context();
            reset_scope(scope);
            return;
        }

        telemetry_record_metadata_mutation_end();
        --lane->ownerDepth;
        if (lane->ownerDepth == 0) {
            if (lane->servingTicket != scope.queueTicket) {
                telemetry_record_metadata_mutation_order_violation();
                telemetry_trace_metadata_queue(
                    "order_violation",
                    stage,
                    scope.queueTicket,
                    lane_depth_nolock(*lane),
                    0,
                    false,
                    "release_ticket_mismatch");
            }
            if (lane->servingTicket < lane->nextTicket) {
                ++lane->servingTicket;
            }
            lane->ownerThread = std::thread::id{};
            lane->activeTicket = 0;
            releaseTopLevel = true;
        }
        depthAfterRelease = lane_depth_nolock(*lane);
    }

    if (releaseTopLevel) {
        lane->cv.notify_all();
        telemetry_trace_metadata_queue(
            "release",
            stage,
            scope.queueTicket,
            depthAfterRelease,
            0,
            true,
            "ok");
    }
    else {
        telemetry_trace_metadata_queue(
            "release_nested",
            stage,
            scope.queueTicket,
            depthAfterRelease,
            0,
            true,
            "owner_reentrant");
    }

    telemetry_trace_metadata_mutation(
        "end",
        stage,
        scope.sequence,
        true,
        scope.sequence,
        "ok");
    if (gMutationThreadDepth > 0) {
        --gMutationThreadDepth;
    }
    if (gMutationThreadDepth == 0) {
        gMutationThreadTicket = 0;
    }
    reset_scope(scope);
}

bool metadata_mutation_thread_active() noexcept {
    return gMutationThreadDepth > 0;
}

std::uint32_t metadata_mutation_thread_depth() noexcept {
    return gMutationThreadDepth;
}

std::uint64_t metadata_mutation_thread_ticket() noexcept {
    return gMutationThreadTicket;
}

std::uint64_t metadata_mutation_thread_begin_count() noexcept {
    return gMutationThreadBeginCount;
}

MetadataMutationGuard::MetadataMutationGuard(
    const char* stage,
    const DeviceContextKey* managerKey) noexcept
    : _stage(stage) {
    if (managerKey) {
        _hasManagerKey = true;
        _managerKey = *managerKey;
    }
    const DeviceContextKey* key = registry_manager_key_ptr_or_null(_hasManagerKey, &_managerKey);
    (void)metadata_mutation_begin(_stage, _scope, key);
}

MetadataMutationGuard::~MetadataMutationGuard() noexcept {
    metadata_mutation_end(_scope, _stage);
}

const char* to_cstr(ContextLifecycleState state) noexcept {
    switch (state) {
    case ContextLifecycleState::Unbound:
        return "Unbound";
    case ContextLifecycleState::Binding:
        return "Binding";
    case ContextLifecycleState::Active:
        return "Active";
    case ContextLifecycleState::Freezing:
        return "Freezing";
    case ContextLifecycleState::Draining:
        return "Draining";
    case ContextLifecycleState::Rebinding:
        return "Rebinding";
    case ContextLifecycleState::Retired:
        return "Retired";
    default:
        return "Unknown";
    }
}

const char* to_cstr(LifecycleStageDecision decision) noexcept {
    switch (decision) {
    case LifecycleStageDecision::Allowed:
        return "Allowed";
    case LifecycleStageDecision::MissingRegistryEntry:
        return "MissingRegistryEntry";
    case LifecycleStageDecision::StateNotAllowed:
        return "StateNotAllowed";
    case LifecycleStageDecision::TimedOut:
        return "TimedOut";
    default:
        return "Unknown";
    }
}

RegistryHandle registry_get_or_create(const DeviceContextKey& key) noexcept {
    MetadataMutationGuard mutationGuard("registry_get_or_create", &key);
    if (!mutationGuard.ok()) {
        return RegistryHandle{};
    }
    RegistryState& state = registry_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    maybe_reap_idle_locked(state, &key);

    const std::uint64_t nowMs = monotonic_time_ms();
    auto it = state.byDeviceContext.find(key);
    if (it != state.byDeviceContext.end()) {
        if (it->second.lifecycleState == ContextLifecycleState::Retired) {
            RegistryEntry retiredEntry = it->second;
            (void)erase_registry_entry_locked(
                state,
                key,
                retiredEntry,
                "recreate",
                "retired_recreate",
                false,
                0);
        }
        else {
            it->second.lastTouchedMs = nowMs;
            publish_registry_live_count(state.byDeviceContext.size());
            return it->second.handle;
        }
    }

    RegistryEntry entry{};
    entry.handle.value = next_nonzero_counter(state.nextHandle);
    entry.lifecycleState = ContextLifecycleState::Unbound;
    entry.createOrder = next_nonzero_counter(state.nextCreateOrder);
    entry.lastTouchedMs = nowMs;
    entry.lifecycleSinceMs = nowMs;
    entry.activeSubmissionCount = 0;

    auto inserted = state.byDeviceContext.emplace(key, entry);
    RegistryEntry& insertedEntry = inserted.first->second;
    state.keyByHandle[insertedEntry.handle.value] = key;
    const bool createBound = transition_entry_locked(
        key,
        insertedEntry,
        ContextLifecycleState::Unbound,
        ContextLifecycleState::Binding,
        "create_bind");
    if (!createBound) {
        (void)transition_entry_to_retired_locked(
            key,
            insertedEntry,
            "create_bind_retire");
    }
    else if (!transition_entry_locked(
            key,
            insertedEntry,
            ContextLifecycleState::Binding,
            ContextLifecycleState::Active,
            "create_activate")) {
        (void)transition_entry_to_retired_locked(
            key,
            insertedEntry,
            "create_activate_retire");
    }

    publish_registry_live_count(state.byDeviceContext.size());
    trace_registry_event_current(
        &key,
        &insertedEntry,
        "create",
        true,
        "create_or_reuse",
        0);

    maybe_reap_idle_locked(state, &key);
    return insertedEntry.handle;
}

bool registry_get_lifecycle_state(const DeviceContextKey& key, ContextLifecycleState& outState) noexcept {
    RegistryState& state = registry_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.byDeviceContext.find(key);
    if (it == state.byDeviceContext.end()) {
        outState = ContextLifecycleState::Unbound;
        return false;
    }
    outState = it->second.lifecycleState;
    return true;
}

void registry_snapshot_context_keys(std::vector<DeviceContextKey>& outKeys) {
    outKeys.clear();
    RegistryState& state = registry_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    outKeys.reserve(state.byDeviceContext.size());
    for (const auto& entry : state.byDeviceContext) {
        outKeys.push_back(entry.first);
    }
}

bool registry_validate_lifecycle_stage(
    const DeviceContextKey& key,
    bool allowNonActiveRelease,
    LifecycleStageValidation& outValidation) noexcept {
    outValidation = LifecycleStageValidation{};
    RegistryState& state = registry_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.byDeviceContext.find(key);
    if (it == state.byDeviceContext.end()) {
        outValidation.decision = LifecycleStageDecision::MissingRegistryEntry;
        outValidation.observedState = ContextLifecycleState::Unbound;
        outValidation.observedStateAgeMs = 0;
        outValidation.escalated = false;
        return false;
    }

    RegistryEntry& entry = it->second;
    const std::uint64_t nowMs = monotonic_time_ms();
    if (entry.lifecycleSinceMs == 0) {
        entry.lifecycleSinceMs = nowMs;
    }

    const ContextLifecycleState observedState = entry.lifecycleState;
    const std::uint64_t stateAgeMs = saturating_elapsed_ms(nowMs, entry.lifecycleSinceMs);
    outValidation.observedState = observedState;
    outValidation.observedStateAgeMs = stateAgeMs;

    const ResourceManagerConfigEffective& cfg = registry_policy_config();
    const std::uint64_t timeoutMs = lifecycle_timeout_ms_for_state(observedState, cfg);
    if (timeoutMs > 0 && stateAgeMs >= timeoutMs) {
        ResourceManagerState& rmState = global_state();
        telemetry_counter_add(rmState.lifecycleTimeoutEvents, 1);
        const RegistryHandle observedHandle = entry.handle;
        const char* timeoutReason = registry_lifecycle_timeout_reason(observedState);

        bool escalated = false;
        if (entry.activeSubmissionCount == 0) {
            bump_registry_epoch_locked(key, observedHandle, true, "watchdog_timeout_bump");
            if (transition_entry_locked(
                    key,
                    entry,
                    observedState,
                    ContextLifecycleState::Retired,
                    "watchdog_timeout_retire")) {
                entry.lastTouchedMs = nowMs;
                RegistryEntry retiredEntry = entry;
                (void)erase_registry_entry_locked(
                    state,
                    key,
                    retiredEntry,
                    "lifecycle_timeout",
                    "watchdog_timeout_retire",
                    false,
                    0);
                escalated = true;
            }
        }

        trace_lifecycle_timeout(
            key,
            observedHandle,
            observedState,
            stateAgeMs,
            timeoutMs,
            escalated,
            timeoutReason);
        outValidation.decision = LifecycleStageDecision::TimedOut;
        outValidation.escalated = escalated;
        return false;
    }

    if (!lifecycle_state_allowed_for_stage(observedState, allowNonActiveRelease)) {
        outValidation.decision = LifecycleStageDecision::StateNotAllowed;
        outValidation.escalated = false;
        return false;
    }

    outValidation.decision = LifecycleStageDecision::Allowed;
    outValidation.escalated = false;
    return true;
}

bool registry_note_submission_begin(const DeviceContextKey& key) noexcept {
    RegistryState& state = registry_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.byDeviceContext.find(key);
    if (it == state.byDeviceContext.end()) {
        trace_registry_missing_entry(key, "submission_begin");
        return false;
    }

    RegistryEntry& entry = it->second;
    if (entry.activeSubmissionCount < std::numeric_limits<std::uint64_t>::max()) {
        ++entry.activeSubmissionCount;
    }
    entry.lastTouchedMs = monotonic_time_ms();
    return true;
}

bool registry_note_submission_end(const DeviceContextKey& key) noexcept {
    RegistryState& state = registry_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.byDeviceContext.find(key);
    if (it == state.byDeviceContext.end()) {
        trace_registry_missing_entry(key, "submission_end");
        return false;
    }

    RegistryEntry& entry = it->second;
    if (entry.activeSubmissionCount == 0) {
        trace_registry_event_current(
            &key,
            &entry,
            "submission_end",
            false,
            "active_submission_underflow",
            0);
        return false;
    }
    --entry.activeSubmissionCount;
    entry.lastTouchedMs = monotonic_time_ms();
    maybe_reap_idle_locked(state, &key);
    return true;
}

bool registry_transition_lifecycle_state(
    const DeviceContextKey& key,
    ContextLifecycleState expectedState,
    ContextLifecycleState desiredState,
    const char* reason) noexcept {
    MetadataMutationGuard mutationGuard("registry_transition_lifecycle_state", &key);
    if (!mutationGuard.ok()) {
        return false;
    }
    RegistryState& state = registry_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.byDeviceContext.find(key);
    if (it == state.byDeviceContext.end()) {
        telemetry_counter_add(global_state().lifecycleTransitionCalls, 1);
        telemetry_counter_add(global_state().lifecycleTransitionRejects, 1);
        const RegistryHandle missingHandle{};
        trace_lifecycle_transition(key, missingHandle, ContextLifecycleState::Unbound, desiredState, false, reason);
        return false;
    }
    const bool ok = transition_entry_locked(key, it->second, expectedState, desiredState, reason);
    if (ok) {
        it->second.lastTouchedMs = monotonic_time_ms();
        maybe_reap_idle_locked(state, &key);
    }
    return ok;
}

bool registry_freeze_drain_bump_resume(
    const DeviceContextKey& key,
    const char* reason) noexcept {
    MetadataMutationGuard mutationGuard("registry_freeze_drain_bump_resume", &key);
    if (!mutationGuard.ok()) {
        return false;
    }
    RegistryState& state = registry_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.byDeviceContext.find(key);
    if (it == state.byDeviceContext.end()) {
        ResourceManagerState& rmState = global_state();
        telemetry_counter_add(rmState.lifecycleBarrierCalls, 1);
        telemetry_counter_add(rmState.lifecycleBarrierRejects, 1);
        const RegistryHandle missingHandle{};
        trace_lifecycle_transition(
            key,
            missingHandle,
            ContextLifecycleState::Unbound,
            ContextLifecycleState::Freezing,
            false,
            registry_trace_or(reason, "barrier_missing_entry"));
        return false;
    }
    const bool ok = run_freeze_drain_bump_resume_locked(key, it->second, reason);
    if (ok) {
        it->second.lastTouchedMs = monotonic_time_ms();
    }
    return ok;
}

bool registry_get(const DeviceContextKey& key, RegistryHandle& outHandle) noexcept {
    RegistryState& state = registry_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.byDeviceContext.find(key);
    if (it == state.byDeviceContext.end()) {
        outHandle = RegistryHandle{};
        return false;
    }
    outHandle = it->second.handle;
    return true;
}

bool registry_retire(
    RegistryHandle handle,
    RegistryRetireReason reason,
    const DeviceContextKey* managerKey) noexcept {
    MetadataMutationGuard mutationGuard("registry_retire", managerKey);
    if (!mutationGuard.ok()) {
        return false;
    }
    if (handle.value == 0) {
        return true;
    }
    RegistryState& state = registry_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto keyIt = state.keyByHandle.find(handle.value);
    if (keyIt == state.keyByHandle.end()) {
        return true;
    }
    auto entryIt = state.byDeviceContext.find(keyIt->second);
    if (entryIt == state.byDeviceContext.end()) {
        state.keyByHandle.erase(keyIt);
        publish_registry_live_count(state.byDeviceContext.size());
        return true;
    }
    RegistryEntry& entry = entryIt->second;
    const DeviceContextKey deviceKey = keyIt->second;
    const std::uint64_t nowMs = monotonic_time_ms();
    entry.lastTouchedMs = nowMs;
    if (reason == RegistryRetireReason::Idle && entry.activeSubmissionCount != 0) {
        trace_registry_event_current(
            &deviceKey,
            &entry,
            "retire",
            false,
            "active_submissions",
            0);
        return false;
    }
    if (reason == RegistryRetireReason::ContextReset) {
        const bool barrierOk = run_freeze_drain_bump_resume_locked(
            deviceKey,
            entry,
            "context_reset_barrier");
        if (!barrierOk) {
            trace_lifecycle_transition(
                deviceKey,
                entry.handle,
                entry.lifecycleState,
                entry.lifecycleState,
                false,
                "context_reset_barrier_reject");
        }
    }
    ContextLifecycleState desired = desired_retire_state(reason);
    if (desired != ContextLifecycleState::Retired) {
        telemetry_counter_add(global_state().lifecycleTransitionCalls, 1);
        telemetry_counter_add(global_state().lifecycleTransitionRejects, 1);
        trace_lifecycle_transition(deviceKey, entry.handle, entry.lifecycleState, desired, false, "retire_unsupported");
        return false;
    }
    if (!transition_entry_to_retired_locked(deviceKey, entry, "retire")) {
        return false;
    }
    RegistryEntry removed = entry;
    (void)erase_registry_entry_locked(
        state,
        deviceKey,
        removed,
        "retire",
        registry_retire_reason_name(reason),
        false,
        0);
    return true;
}

} // namespace ResourceManager
} // namespace JuicerCuda
