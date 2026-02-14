// Cuda/ResourceManager/JuicerCudaManagerRegistry.cpp

#include "Cuda/ResourceManager/JuicerCudaManagerRegistry.h"
#include "Cuda/ResourceManager/JuicerCudaResourceConfig.h"
#include "Cuda/ResourceManager/JuicerCudaResourceState.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "Logging.h"

namespace JuicerCuda {
namespace ResourceManager {
namespace {

struct RegistryEntry {
    RegistryHandle handle{};
    ContextLifecycleState lifecycleState = ContextLifecycleState::Unbound;
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
    return (nowMs >= thenMs) ? (nowMs - thenMs) : 0ull;
}

void publish_registry_live_count(std::size_t count) noexcept {
    global_state().registryLiveManagers.store(static_cast<std::uint64_t>(count), std::memory_order_relaxed);
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
    const int deviceId = key ? key->deviceId : -1;
    const std::uintptr_t contextBits = key ? reinterpret_cast<std::uintptr_t>(key->contextOpaque) : 0;
    const std::uint64_t handleValue = entry ? entry->handle.value : 0;
    const char* lifecycle = entry ? to_cstr(entry->lifecycleState) : "Missing";
    const std::uint64_t activeSubmissions = entry ? entry->activeSubmissionCount : 0;
    const std::string msg =
        std::string("event=") + (eventName ? eventName : "unknown") +
        " accepted=" + std::to_string(accepted ? 1 : 0) +
        " reason=" + (reason ? reason : "unspecified") +
        " handle=" + std::to_string(handleValue) +
        " device_id=" + std::to_string(deviceId) +
        " context=" + std::to_string(contextBits) +
        " lifecycle=" + lifecycle +
        " active_submissions=" + std::to_string(activeSubmissions) +
        " idle_ms=" + std::to_string(idleMs) +
        " live_managers=" + std::to_string(liveManagers) +
        " max_live_managers=" + std::to_string(maxLiveManagers) +
        " reap_events=" + std::to_string(reapEvents);
    JTRACE_LEVEL(2, "MSREG", msg);
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
    const std::uintptr_t contextBits = reinterpret_cast<std::uintptr_t>(key.contextOpaque);
    const std::string msg = std::string("handle=") + std::to_string(handle.value)
        + " device_id=" + std::to_string(key.deviceId)
        + " context=" + std::to_string(contextBits)
        + " from=" + to_cstr(from)
        + " to=" + to_cstr(to)
        + " accepted=" + std::to_string(accepted ? 1 : 0)
        + " reason=" + (reason ? reason : "unspecified");
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
    const std::uintptr_t contextBits = reinterpret_cast<std::uintptr_t>(key.contextOpaque);
    const std::string msg = std::string("handle=") + std::to_string(handle.value)
        + " device_id=" + std::to_string(key.deviceId)
        + " context=" + std::to_string(contextBits)
        + " action=bump"
        + " accepted=" + std::to_string(accepted ? 1 : 0)
        + " prev_registry_generation=" + std::to_string(previousRegistryGeneration)
        + " new_registry_generation=" + std::to_string(newRegistryGeneration)
        + " prev_context_epoch=" + std::to_string(previousContextEpoch)
        + " new_context_epoch=" + std::to_string(newContextEpoch)
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSLCY", msg);
}

bool transition_entry_locked(const DeviceContextKey& key,
                             RegistryEntry& entry,
                             ContextLifecycleState expectedState,
                             ContextLifecycleState desiredState,
                             const char* reason) {
    ResourceManagerState& rmState = global_state();
    rmState.lifecycleTransitionCalls.fetch_add(1, std::memory_order_relaxed);
    const ContextLifecycleState observed = entry.lifecycleState;
    const bool expectedOk = (observed == expectedState);
    const bool legal = expectedOk && is_legal_transition(observed, desiredState);
    if (!legal) {
        rmState.lifecycleTransitionRejects.fetch_add(1, std::memory_order_relaxed);
        trace_lifecycle_transition(key, entry.handle, observed, desiredState, false, reason);
        return false;
    }
    entry.lifecycleState = desiredState;
    trace_lifecycle_transition(key, entry.handle, observed, desiredState, true, reason);
    return true;
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
        rmState.registryReapEvents.fetch_add(1, std::memory_order_relaxed);
    }
    const std::uint64_t liveManagers = rmState.registryLiveManagers.load(std::memory_order_relaxed);
    const std::uint64_t reapEvents = rmState.registryReapEvents.load(std::memory_order_relaxed);
    const ResourceManagerConfigEffective& cfg = registry_policy_config();
    trace_registry_event(
        &key,
        &entry,
        eventName,
        true,
        reason,
        idleMs,
        liveManagers,
        static_cast<std::uint64_t>(cfg.maxLiveManagersPerProcess),
        reapEvents);
    return true;
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
    const std::uint64_t overflow = (liveManagersBefore > maxLive) ? (liveManagersBefore - maxLive) : 0ull;
    if (!cadenceDue && overflow == 0) {
        return;
    }
    state.lastIdleReapScanMs = nowMs;

    struct ReapCandidate {
        DeviceContextKey key{};
        RegistryEntry entry{};
        std::uint64_t idleMs = 0;
    };

    std::vector<ReapCandidate> candidates;
    candidates.reserve(state.byDeviceContext.size());
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
            (idleMs >= static_cast<std::uint64_t>(cfg.managerIdleReapMs));
        if (!eligible) {
            continue;
        }

        ReapCandidate candidate{};
        candidate.key = candidateKey;
        candidate.entry = entry;
        candidate.idleMs = idleMs;
        candidates.push_back(candidate);
    }

    if (candidates.empty()) {
        if (overflow > 0) {
            ResourceManagerState& rmState = global_state();
            trace_registry_event(
                nullptr,
                nullptr,
                "reap_skip",
                false,
                "overflow_no_eligible_idle_manager",
                0,
                rmState.registryLiveManagers.load(std::memory_order_relaxed),
                maxLive,
                rmState.registryReapEvents.load(std::memory_order_relaxed));
        }
        return;
    }

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

    std::size_t reapCount = candidates.size();
    if (overflow > 0) {
        const std::size_t overflowCount = static_cast<std::size_t>(
            std::min<std::uint64_t>(overflow, static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())));
        reapCount = std::min(reapCount, overflowCount);
    }

    for (std::size_t i = 0; i < reapCount; ++i) {
        const ReapCandidate& candidate = candidates[i];
        const char* reason = (overflow > 0) ? "max_live_oldest_idle" : "idle_timeout";
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
        ResourceManagerState& rmState = global_state();
        trace_registry_event(
            nullptr,
            nullptr,
            "reap_skip",
            false,
            "overflow_remaining_after_safe_reap",
            0,
            rmState.registryLiveManagers.load(std::memory_order_relaxed),
            maxLive,
            rmState.registryReapEvents.load(std::memory_order_relaxed));
    }
}

bool run_freeze_drain_bump_resume_locked(const DeviceContextKey& key,
                                         RegistryEntry& entry,
                                         const char* reason) {
    ResourceManagerState& rmState = global_state();
    rmState.lifecycleBarrierCalls.fetch_add(1, std::memory_order_relaxed);

    bool ok = true;
    ok = ok && transition_entry_locked(
        key, entry, ContextLifecycleState::Active, ContextLifecycleState::Freezing, "barrier_freeze");
    ok = ok && transition_entry_locked(
        key, entry, ContextLifecycleState::Freezing, ContextLifecycleState::Draining, "barrier_drain");

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
        entry.handle,
        prevRegistryGeneration,
        newRegistryGeneration,
        prevContextEpoch,
        newContextEpoch,
        ok,
        reason ? reason : "barrier_bump");

    ok = ok && transition_entry_locked(
        key, entry, ContextLifecycleState::Draining, ContextLifecycleState::Rebinding, "barrier_rebind");
    ok = ok && transition_entry_locked(
        key, entry, ContextLifecycleState::Rebinding, ContextLifecycleState::Active, "barrier_resume");

    if (!ok) {
        rmState.lifecycleBarrierRejects.fetch_add(1, std::memory_order_relaxed);
    }
    return ok;
}

} // namespace

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

RegistryHandle registry_get_or_create(const DeviceContextKey& key) noexcept {
    MetadataMutationGuard mutationGuard("registry_get_or_create");
    if (!mutationGuard.ok()) {
        return RegistryHandle{};
    }
    RegistryState& state = registry_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    maybe_reap_idle_locked(state, &key);

    const std::uint64_t nowMs = monotonic_time_ms();
    auto it = state.byDeviceContext.find(key);
    if (it != state.byDeviceContext.end()) {
        it->second.lastTouchedMs = nowMs;
        publish_registry_live_count(state.byDeviceContext.size());
        return it->second.handle;
    }

    RegistryEntry entry{};
    entry.handle.value = state.nextHandle.fetch_add(1, std::memory_order_relaxed);
    if (entry.handle.value == 0) {
        entry.handle.value = state.nextHandle.fetch_add(1, std::memory_order_relaxed);
    }
    entry.lifecycleState = ContextLifecycleState::Unbound;
    entry.createOrder = state.nextCreateOrder.fetch_add(1, std::memory_order_relaxed);
    if (entry.createOrder == 0) {
        entry.createOrder = state.nextCreateOrder.fetch_add(1, std::memory_order_relaxed);
    }
    entry.lastTouchedMs = nowMs;
    entry.activeSubmissionCount = 0;

    auto inserted = state.byDeviceContext.emplace(key, entry);
    RegistryEntry& insertedEntry = inserted.first->second;
    state.keyByHandle[insertedEntry.handle.value] = key;
    if (!transition_entry_locked(key, insertedEntry,
            ContextLifecycleState::Unbound,
            ContextLifecycleState::Binding,
            "create_bind")) {
        insertedEntry.lifecycleState = ContextLifecycleState::Retired;
    }
    if (!transition_entry_locked(key, insertedEntry,
            ContextLifecycleState::Binding,
            ContextLifecycleState::Active,
            "create_activate")) {
        insertedEntry.lifecycleState = ContextLifecycleState::Retired;
    }

    publish_registry_live_count(state.byDeviceContext.size());
    trace_registry_event(
        &key,
        &insertedEntry,
        "create",
        true,
        "create_or_reuse",
        0,
        global_state().registryLiveManagers.load(std::memory_order_relaxed),
        static_cast<std::uint64_t>(registry_policy_config().maxLiveManagersPerProcess),
        global_state().registryReapEvents.load(std::memory_order_relaxed));

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

bool registry_note_submission_begin(const DeviceContextKey& key) noexcept {
    RegistryState& state = registry_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.byDeviceContext.find(key);
    if (it == state.byDeviceContext.end()) {
        trace_registry_event(
            &key,
            nullptr,
            "submission_begin",
            false,
            "missing_registry_entry",
            0,
            global_state().registryLiveManagers.load(std::memory_order_relaxed),
            static_cast<std::uint64_t>(registry_policy_config().maxLiveManagersPerProcess),
            global_state().registryReapEvents.load(std::memory_order_relaxed));
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
        trace_registry_event(
            &key,
            nullptr,
            "submission_end",
            false,
            "missing_registry_entry",
            0,
            global_state().registryLiveManagers.load(std::memory_order_relaxed),
            static_cast<std::uint64_t>(registry_policy_config().maxLiveManagersPerProcess),
            global_state().registryReapEvents.load(std::memory_order_relaxed));
        return false;
    }

    RegistryEntry& entry = it->second;
    if (entry.activeSubmissionCount == 0) {
        trace_registry_event(
            &key,
            &entry,
            "submission_end",
            false,
            "active_submission_underflow",
            0,
            global_state().registryLiveManagers.load(std::memory_order_relaxed),
            static_cast<std::uint64_t>(registry_policy_config().maxLiveManagersPerProcess),
            global_state().registryReapEvents.load(std::memory_order_relaxed));
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
    MetadataMutationGuard mutationGuard("registry_transition_lifecycle_state");
    if (!mutationGuard.ok()) {
        return false;
    }
    RegistryState& state = registry_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.byDeviceContext.find(key);
    if (it == state.byDeviceContext.end()) {
        global_state().lifecycleTransitionCalls.fetch_add(1, std::memory_order_relaxed);
        global_state().lifecycleTransitionRejects.fetch_add(1, std::memory_order_relaxed);
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
    MetadataMutationGuard mutationGuard("registry_freeze_drain_bump_resume");
    if (!mutationGuard.ok()) {
        return false;
    }
    RegistryState& state = registry_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.byDeviceContext.find(key);
    if (it == state.byDeviceContext.end()) {
        ResourceManagerState& rmState = global_state();
        rmState.lifecycleBarrierCalls.fetch_add(1, std::memory_order_relaxed);
        rmState.lifecycleBarrierRejects.fetch_add(1, std::memory_order_relaxed);
        const RegistryHandle missingHandle{};
        trace_lifecycle_transition(
            key,
            missingHandle,
            ContextLifecycleState::Unbound,
            ContextLifecycleState::Freezing,
            false,
            reason ? reason : "barrier_missing_entry");
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

void registry_retire(RegistryHandle handle, RegistryRetireReason reason) noexcept {
    MetadataMutationGuard mutationGuard("registry_retire");
    if (!mutationGuard.ok()) {
        return;
    }
    if (handle.value == 0) {
        return;
    }
    RegistryState& state = registry_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto keyIt = state.keyByHandle.find(handle.value);
    if (keyIt == state.keyByHandle.end()) {
        return;
    }
    auto entryIt = state.byDeviceContext.find(keyIt->second);
    if (entryIt == state.byDeviceContext.end()) {
        state.keyByHandle.erase(keyIt);
        publish_registry_live_count(state.byDeviceContext.size());
        return;
    }
    RegistryEntry& entry = entryIt->second;
    const DeviceContextKey deviceKey = keyIt->second;
    const std::uint64_t nowMs = monotonic_time_ms();
    entry.lastTouchedMs = nowMs;
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
    if (entry.lifecycleState != ContextLifecycleState::Retired) {
        if (!is_legal_transition(entry.lifecycleState, desired)) {
            global_state().lifecycleTransitionCalls.fetch_add(1, std::memory_order_relaxed);
            global_state().lifecycleTransitionRejects.fetch_add(1, std::memory_order_relaxed);
            trace_lifecycle_transition(deviceKey, entry.handle, entry.lifecycleState, desired, false, "retire_illegal");
            return;
        }
        global_state().lifecycleTransitionCalls.fetch_add(1, std::memory_order_relaxed);
        trace_lifecycle_transition(deviceKey, entry.handle, entry.lifecycleState, desired, true, "retire");
        entry.lifecycleState = ContextLifecycleState::Retired;
    }
    RegistryEntry removed = entry;
    const char* retireReason = (reason == RegistryRetireReason::ContextReset)
        ? "context_reset"
        : ((reason == RegistryRetireReason::Idle) ? "idle" : "unknown");
    (void)erase_registry_entry_locked(
        state,
        deviceKey,
        removed,
        "retire",
        retireReason,
        false,
        0);
}

} // namespace ResourceManager
} // namespace JuicerCuda
