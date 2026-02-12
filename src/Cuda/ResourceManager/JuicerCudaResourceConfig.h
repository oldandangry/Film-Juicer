// Cuda/ResourceManager/JuicerCudaResourceConfig.h
//
// Phase-0 config scaffolding for ResourceManager.
#pragma once

#include <cstdint>

#include "Cuda/ResourceManager/JuicerCudaResourceTypes.h"

namespace JuicerCuda {
namespace ResourceManager {

struct ResourceManagerConfigRaw {
    std::uint32_t keySchemaVersion = 1;
    std::uint32_t traceSchemaVersion = kTraceSchemaVersion;
    bool allowShadowMode = true;
};

struct ResourceManagerConfigEffective {
    std::uint32_t keySchemaVersion = 1;
    std::uint32_t traceSchemaVersion = kTraceSchemaVersion;
    bool allowShadowMode = true;
};

ResourceManagerConfigEffective sanitize_config(const ResourceManagerConfigRaw& raw);

} // namespace ResourceManager
} // namespace JuicerCuda
