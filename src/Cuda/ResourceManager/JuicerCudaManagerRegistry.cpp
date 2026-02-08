// Cuda/ResourceManager/JuicerCudaManagerRegistry.cpp

#include "Cuda/ResourceManager/JuicerCudaManagerRegistry.h"
#include "Cuda/ResourceManager/JuicerCudaResourceState.h"

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>

#include "Logging.h"

namespace JuicerCuda {
namespace ResourceManager {
namespace {

struct RegistryEntry {
    RegistryHandle handle{};
    ContextLifecycleState lifecycleState = ContextLifecycleState::Unbound;
};

struct RegistryState {
    std::mutex mutex;
    std::atomic<std::uint64_t> nextHandle{ 1 };
    std::unordered_map<DeviceContextKey, RegistryEntry, DeviceContextKeyHash> byDeviceContext;
    std::unordered_map<std::uint64_t, DeviceContextKey> keyByHandle;
};

RegistryState& registry_state() {
    static RegistryState state{};
    return state;
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
    RegistryState& state = registry_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.byDeviceContext.find(key);
    if (it != state.byDeviceContext.end()) {
        return it->second.handle;
    }
    RegistryEntry entry{};
    entry.handle.value = state.nextHandle.fetch_add(1, std::memory_order_relaxed);
    if (entry.handle.value == 0) {
        entry.handle.value = state.nextHandle.fetch_add(1, std::memory_order_relaxed);
    }
    entry.lifecycleState = ContextLifecycleState::Unbound;

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

bool registry_transition_lifecycle_state(
    const DeviceContextKey& key,
    ContextLifecycleState expectedState,
    ContextLifecycleState desiredState,
    const char* reason) noexcept {
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
    return transition_entry_locked(key, it->second, expectedState, desiredState, reason);
}

bool registry_freeze_drain_bump_resume(
    const DeviceContextKey& key,
    const char* reason) noexcept {
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
    return run_freeze_drain_bump_resume_locked(key, it->second, reason);
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
        return;
    }
    RegistryEntry& entry = entryIt->second;
    if (reason == RegistryRetireReason::ContextReset) {
        const bool barrierOk = run_freeze_drain_bump_resume_locked(
            keyIt->second,
            entry,
            "context_reset_barrier");
        if (!barrierOk) {
            trace_lifecycle_transition(
                keyIt->second,
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
            trace_lifecycle_transition(keyIt->second, entry.handle, entry.lifecycleState, desired, false, "retire_illegal");
            return;
        }
        global_state().lifecycleTransitionCalls.fetch_add(1, std::memory_order_relaxed);
        trace_lifecycle_transition(keyIt->second, entry.handle, entry.lifecycleState, desired, true, "retire");
        entry.lifecycleState = ContextLifecycleState::Retired;
    }
    state.byDeviceContext.erase(entryIt);
    state.keyByHandle.erase(keyIt);
}

} // namespace ResourceManager
} // namespace JuicerCuda
