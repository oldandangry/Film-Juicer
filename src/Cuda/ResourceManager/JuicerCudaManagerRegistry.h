// Cuda/ResourceManager/JuicerCudaManagerRegistry.h
//
// Phase-0 registry scaffolding.
#pragma once

#include <cstdint>

#include "Cuda/ResourceManager/JuicerCudaResourceTypes.h"

namespace JuicerCuda {
namespace ResourceManager {

enum class RegistryRetireReason : std::uint8_t {
    Unknown = 0,
    Idle = 1,
    ContextReset = 2
};

enum class ContextLifecycleState : std::uint8_t {
    Unbound = 0,
    Binding = 1,
    Active = 2,
    Freezing = 3,
    Draining = 4,
    Rebinding = 5,
    Retired = 6
};

struct RegistryHandle {
    std::uint64_t value = 0;
};

const char* to_cstr(ContextLifecycleState state) noexcept;

RegistryHandle registry_get_or_create(const DeviceContextKey& key) noexcept;
bool registry_get(const DeviceContextKey& key, RegistryHandle& outHandle) noexcept;
bool registry_get_lifecycle_state(const DeviceContextKey& key, ContextLifecycleState& outState) noexcept;
bool registry_note_submission_begin(const DeviceContextKey& key) noexcept;
bool registry_note_submission_end(const DeviceContextKey& key) noexcept;
bool registry_transition_lifecycle_state(
    const DeviceContextKey& key,
    ContextLifecycleState expectedState,
    ContextLifecycleState desiredState,
    const char* reason) noexcept;
bool registry_freeze_drain_bump_resume(
    const DeviceContextKey& key,
    const char* reason) noexcept;
void registry_retire(
    RegistryHandle handle,
    RegistryRetireReason reason,
    const DeviceContextKey* managerKey = nullptr) noexcept;

} // namespace ResourceManager
} // namespace JuicerCuda
