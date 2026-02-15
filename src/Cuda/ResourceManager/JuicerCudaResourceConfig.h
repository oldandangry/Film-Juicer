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
    std::uint64_t reserveSafetyMarginBytes = 256ull * 1024ull * 1024ull;
    std::uint64_t reserveAdaptUpStepBytes = 128ull * 1024ull * 1024ull;
    std::uint64_t reserveAdaptDownStepBytes = 64ull * 1024ull * 1024ull;
    bool freezeOpportunisticBelowReserve = true;
    bool allowActiveFrameBurst = true;
    std::uint64_t maxActiveBurstBytes = 512ull * 1024ull * 1024ull;
    std::uint32_t maxActiveBurstPctOfTarget = 20;
    std::uint32_t maxActiveBurstMs = 250;
    std::uint32_t pressureSampleIntervalMs = 250;
    std::uint32_t pressurePollIntervalMs = 250;
    std::uint32_t pressurePollIntervalMsNormal = 250;
    std::uint32_t pressurePollIntervalMsCritical = 100;
    std::uint32_t pressureStateMinDwellMs = 250;
    std::uint32_t pressureStateMaxTransitionsPerMin = 12;
    std::uint32_t reclaimRetryMaxAttempts = 1;
    std::uint32_t tierErrorWindowMs = 1000;
    std::uint32_t tierErrorThreshold = 3;
    std::uint32_t tierCircuitOpenMs = 2000;
    bool fragmentationRecoveryEnabled = true;
    std::uint64_t hostAssetCacheMaxBytes = 256ull * 1024ull * 1024ull;
    std::uint32_t hostAssetIdleTrimMs = 5000;
    std::uint64_t hostAssetTrimBatchBytes = 64ull * 1024ull * 1024ull;
    std::uint32_t privateLutFallbackPerMediumCap = 1;
    std::uint32_t privateLutFallbackPerInstanceCap = 2;
    std::uint64_t pinnedUploadStagingMaxBytes = 128ull * 1024ull * 1024ull;
    std::uint32_t pinnedUploadStagingIdleTrimMs = 5000;
    std::uint64_t pinnedUploadStagingTrimBatchBytes = 32ull * 1024ull * 1024ull;
    std::uint32_t scratchBuilderBytesInFlightLimitMB = 256;
    std::uint32_t lutBuilderBytesInFlightLimitMB = 128;
    std::uint32_t graphBuilderBytesInFlightLimitMB = 128;
    std::uint32_t builderFairnessTokensPerTick = 1;
    std::uint32_t criticalBuilderReservedTokens = 1;
    std::uint32_t uploadBytesInFlightLimitMB = 256;
    std::uint32_t uploadFairnessTokensPerTick = 1;
    std::uint32_t criticalUploadReservedTokens = 1;
    std::uint64_t maxCacheableEntryBytes = 256ull * 1024ull * 1024ull;
    std::uint32_t maxCacheableEntryPctOfTarget = 20;
    std::uint64_t graphLargeEntryThresholdBytes = 128ull * 1024ull * 1024ull;
    std::uint64_t graphLargeEntryQuarantineMaxBytes = 512ull * 1024ull * 1024ull;
    std::uint32_t graphLargeEntryQuarantineMaxEntries = 2;
    std::uint64_t largeEntryProbationThresholdBytes = 128ull * 1024ull * 1024ull;
    std::uint32_t largeEntryProbationHitsRequired = 2;
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
    std::uint64_t reserveSafetyMarginBytes = 256ull * 1024ull * 1024ull;
    std::uint64_t reserveAdaptUpStepBytes = 128ull * 1024ull * 1024ull;
    std::uint64_t reserveAdaptDownStepBytes = 64ull * 1024ull * 1024ull;
    bool freezeOpportunisticBelowReserve = true;
    bool allowActiveFrameBurst = true;
    std::uint64_t maxActiveBurstBytes = 512ull * 1024ull * 1024ull;
    std::uint32_t maxActiveBurstPctOfTarget = 20;
    std::uint32_t maxActiveBurstMs = 250;
    std::uint32_t pressureSampleIntervalMs = 250;
    std::uint32_t pressurePollIntervalMs = 250;
    std::uint32_t pressurePollIntervalMsNormal = 250;
    std::uint32_t pressurePollIntervalMsCritical = 100;
    std::uint32_t pressureStateMinDwellMs = 250;
    std::uint32_t pressureStateMaxTransitionsPerMin = 12;
    std::uint32_t reclaimRetryMaxAttempts = 1;
    std::uint32_t tierErrorWindowMs = 1000;
    std::uint32_t tierErrorThreshold = 3;
    std::uint32_t tierCircuitOpenMs = 2000;
    bool fragmentationRecoveryEnabled = true;
    std::uint64_t hostAssetCacheMaxBytes = 256ull * 1024ull * 1024ull;
    std::uint32_t hostAssetIdleTrimMs = 5000;
    std::uint64_t hostAssetTrimBatchBytes = 64ull * 1024ull * 1024ull;
    std::uint32_t privateLutFallbackPerMediumCap = 1;
    std::uint32_t privateLutFallbackPerInstanceCap = 2;
    std::uint64_t pinnedUploadStagingMaxBytes = 128ull * 1024ull * 1024ull;
    std::uint32_t pinnedUploadStagingIdleTrimMs = 5000;
    std::uint64_t pinnedUploadStagingTrimBatchBytes = 32ull * 1024ull * 1024ull;
    std::uint32_t scratchBuilderBytesInFlightLimitMB = 256;
    std::uint32_t lutBuilderBytesInFlightLimitMB = 128;
    std::uint32_t graphBuilderBytesInFlightLimitMB = 128;
    std::uint32_t builderFairnessTokensPerTick = 1;
    std::uint32_t criticalBuilderReservedTokens = 1;
    std::uint32_t uploadBytesInFlightLimitMB = 256;
    std::uint32_t uploadFairnessTokensPerTick = 1;
    std::uint32_t criticalUploadReservedTokens = 1;
    std::uint64_t maxCacheableEntryBytes = 256ull * 1024ull * 1024ull;
    std::uint32_t maxCacheableEntryPctOfTarget = 20;
    std::uint64_t graphLargeEntryThresholdBytes = 128ull * 1024ull * 1024ull;
    std::uint64_t graphLargeEntryQuarantineMaxBytes = 512ull * 1024ull * 1024ull;
    std::uint32_t graphLargeEntryQuarantineMaxEntries = 2;
    std::uint64_t largeEntryProbationThresholdBytes = 128ull * 1024ull * 1024ull;
    std::uint32_t largeEntryProbationHitsRequired = 2;
    std::uint64_t tierTargetImmutableBp = 2500;
    std::uint64_t tierTargetLutBp = 2500;
    std::uint64_t tierTargetScratchBp = 3000;
    std::uint64_t tierTargetGraphBp = 2000;
};

ResourceManagerConfigEffective sanitize_config(const ResourceManagerConfigRaw& raw);

} // namespace ResourceManager
} // namespace JuicerCuda
