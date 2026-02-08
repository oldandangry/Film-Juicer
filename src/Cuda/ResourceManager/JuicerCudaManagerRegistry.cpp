// Cuda/ResourceManager/JuicerCudaManagerRegistry.cpp

#include "Cuda/ResourceManager/JuicerCudaManagerRegistry.h"

#include <atomic>
#include <mutex>
#include <unordered_map>

namespace JuicerCuda {
namespace ResourceManager {
namespace {

struct RegistryState {
    std::mutex mutex;
    std::atomic<std::uint64_t> nextHandle{ 1 };
    std::unordered_map<DeviceContextKey, RegistryHandle, DeviceContextKeyHash> byDeviceContext;
};

RegistryState& registry_state() {
    static RegistryState state{};
    return state;
}

} // namespace

RegistryHandle registry_get_or_create(const DeviceContextKey& key) noexcept {
    RegistryState& state = registry_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.byDeviceContext.find(key);
    if (it != state.byDeviceContext.end()) {
        return it->second;
    }
    RegistryHandle handle{};
    handle.value = state.nextHandle.fetch_add(1, std::memory_order_relaxed);
    if (handle.value == 0) {
        handle.value = state.nextHandle.fetch_add(1, std::memory_order_relaxed);
    }
    state.byDeviceContext.emplace(key, handle);
    return handle;
}

bool registry_get(const DeviceContextKey& key, RegistryHandle& outHandle) noexcept {
    RegistryState& state = registry_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.byDeviceContext.find(key);
    if (it == state.byDeviceContext.end()) {
        outHandle = RegistryHandle{};
        return false;
    }
    outHandle = it->second;
    return true;
}

void registry_retire(RegistryHandle handle, RegistryRetireReason reason) noexcept {
    (void)reason;
    if (handle.value == 0) {
        return;
    }
    // Phase-0 scaffolding: no active retirement behavior yet.
}

} // namespace ResourceManager
} // namespace JuicerCuda
