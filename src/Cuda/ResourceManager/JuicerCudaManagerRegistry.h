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

struct RegistryHandle {
    std::uint64_t value = 0;
};

RegistryHandle registry_get_or_create(const DeviceContextKey& key) noexcept;
bool registry_get(const DeviceContextKey& key, RegistryHandle& outHandle) noexcept;
void registry_retire(RegistryHandle handle, RegistryRetireReason reason) noexcept;

} // namespace ResourceManager
} // namespace JuicerCuda

