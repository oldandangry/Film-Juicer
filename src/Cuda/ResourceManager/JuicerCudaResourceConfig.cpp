// Cuda/ResourceManager/JuicerCudaResourceConfig.cpp

#include "Cuda/ResourceManager/JuicerCudaResourceConfig.h"

#include <algorithm>

namespace JuicerCuda {
namespace ResourceManager {

ResourceManagerConfigEffective sanitize_config(const ResourceManagerConfigRaw& raw) {
    ResourceManagerConfigEffective out{};
    out.keySchemaVersion = std::max<std::uint32_t>(1u, raw.keySchemaVersion);
    out.traceSchemaVersion = sanitize_trace_schema_version(raw.traceSchemaVersion);
    out.allowShadowMode = raw.allowShadowMode;
    return out;
}

} // namespace ResourceManager
} // namespace JuicerCuda
