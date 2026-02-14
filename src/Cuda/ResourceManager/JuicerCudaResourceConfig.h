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
    std::uint32_t maxLiveManagersPerProcess = 16;
    std::uint32_t managerIdleReapMs = 3000;
    std::uint64_t managerSoftTargetBytes = 0;
    std::uint64_t managerReserveBytes = 0;
    std::uint32_t pressureSampleIntervalMs = 250;
    std::uint32_t reclaimRetryMaxAttempts = 1;
    std::uint32_t uploadBytesInFlightLimitMB = 256;
    std::uint32_t uploadFairnessTokensPerTick = 1;
    std::uint32_t criticalUploadReservedTokens = 1;
    std::uint64_t tierTargetImmutableBp = 2500;
    std::uint64_t tierTargetLutBp = 2500;
    std::uint64_t tierTargetScratchBp = 3000;
    std::uint64_t tierTargetGraphBp = 2000;
};

struct ResourceManagerConfigEffective {
    std::uint32_t keySchemaVersion = 1;
    std::uint32_t traceSchemaVersion = kTraceSchemaVersion;
    bool allowShadowMode = true;
    std::uint32_t maxLiveManagersPerProcess = 16;
    std::uint32_t managerIdleReapMs = 3000;
    std::uint64_t managerSoftTargetBytes = 0;
    std::uint64_t managerReserveBytes = 0;
    std::uint32_t pressureSampleIntervalMs = 250;
    std::uint32_t reclaimRetryMaxAttempts = 1;
    std::uint32_t uploadBytesInFlightLimitMB = 256;
    std::uint32_t uploadFairnessTokensPerTick = 1;
    std::uint32_t criticalUploadReservedTokens = 1;
    std::uint64_t tierTargetImmutableBp = 2500;
    std::uint64_t tierTargetLutBp = 2500;
    std::uint64_t tierTargetScratchBp = 3000;
    std::uint64_t tierTargetGraphBp = 2000;
};

ResourceManagerConfigEffective sanitize_config(const ResourceManagerConfigRaw& raw);

} // namespace ResourceManager
} // namespace JuicerCuda
