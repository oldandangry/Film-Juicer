// Cuda/ResourceManager/JuicerCudaResourceConfig.cpp

#include "Cuda/ResourceManager/JuicerCudaResourceConfig.h"

#include <algorithm>

namespace JuicerCuda {
namespace ResourceManager {

ResourceManagerConfigEffective sanitize_config(const ResourceManagerConfigRaw& raw) {
    constexpr std::uint64_t kBasisPointsDenom = 10000ull;
    constexpr std::uint32_t kMinLiveManagers = 8u;
    constexpr std::uint32_t kMaxLiveManagers = 32u;
    constexpr std::uint32_t kMinIdleReapMs = 2000u;
    constexpr std::uint32_t kMaxIdleReapMs = 5000u;
    constexpr std::uint32_t kMinPressureSampleMs = 10u;
    constexpr std::uint32_t kMaxPressureSampleMs = 5000u;
    constexpr std::uint64_t kMiB = 1024ull * 1024ull;
    constexpr std::uint64_t kMinReserveSafetyMarginBytes = 32ull * kMiB;
    constexpr std::uint64_t kMaxReserveSafetyMarginBytes = 1024ull * kMiB;
    constexpr std::uint64_t kMinReserveAdaptStepBytes = 8ull * kMiB;
    constexpr std::uint64_t kMaxReserveAdaptStepBytes = 512ull * kMiB;
    constexpr std::uint64_t kMinMaxActiveBurstBytes = 32ull * kMiB;
    constexpr std::uint64_t kMaxMaxActiveBurstBytes = 1024ull * kMiB;
    constexpr std::uint32_t kMinMaxActiveBurstPctOfTarget = 5u;
    constexpr std::uint32_t kMaxMaxActiveBurstPctOfTarget = 50u;
    constexpr std::uint32_t kMinMaxActiveBurstMs = 50u;
    constexpr std::uint32_t kMaxMaxActiveBurstMs = 5000u;
    constexpr std::uint32_t kMinPressurePollIntervalMs = 10u;
    constexpr std::uint32_t kMaxPressurePollIntervalMs = 5000u;
    constexpr std::uint32_t kMinPressureStateMinDwellMs = 50u;
    constexpr std::uint32_t kMaxPressureStateMinDwellMs = 2000u;
    constexpr std::uint32_t kMinPressureStateMaxTransitionsPerMin = 1u;
    constexpr std::uint32_t kMaxPressureStateMaxTransitionsPerMin = 120u;
    constexpr std::uint32_t kMaxReclaimRetryAttempts = 3u;
    constexpr std::uint32_t kMinTierErrorWindowMs = 100u;
    constexpr std::uint32_t kMaxTierErrorWindowMs = 10000u;
    constexpr std::uint32_t kMinTierErrorThreshold = 1u;
    constexpr std::uint32_t kMaxTierErrorThreshold = 16u;
    constexpr std::uint32_t kMinTierCircuitOpenMs = 100u;
    constexpr std::uint32_t kMaxTierCircuitOpenMs = 10000u;
    constexpr std::uint32_t kMinHostAssetIdleTrimMs = 1000u;
    constexpr std::uint32_t kMaxHostAssetIdleTrimMs = 60000u;
    constexpr std::uint32_t kMinScratchBuilderBytesInFlightLimitMB = 128u;
    constexpr std::uint32_t kMaxScratchBuilderBytesInFlightLimitMB = 512u;
    constexpr std::uint32_t kMinLutBuilderBytesInFlightLimitMB = 64u;
    constexpr std::uint32_t kMaxLutBuilderBytesInFlightLimitMB = 256u;
    constexpr std::uint32_t kMinGraphBuilderBytesInFlightLimitMB = 64u;
    constexpr std::uint32_t kMaxGraphBuilderBytesInFlightLimitMB = 256u;
    constexpr std::uint32_t kMinBuilderFairnessTokensPerTick = 1u;
    constexpr std::uint32_t kMaxBuilderFairnessTokensPerTick = 2u;
    constexpr std::uint32_t kMinCriticalBuilderReservedTokens = 1u;
    constexpr std::uint32_t kMinUploadBytesInFlightLimitMB = 128u;
    constexpr std::uint32_t kMaxUploadBytesInFlightLimitMB = 512u;
    constexpr std::uint32_t kMinUploadFairnessTokensPerTick = 1u;
    constexpr std::uint32_t kMaxUploadFairnessTokensPerTick = 2u;
    constexpr std::uint32_t kMinCriticalUploadReservedTokens = 1u;
    constexpr std::uint64_t kMinMaxCacheableEntryBytes = 32ull * kMiB;
    constexpr std::uint64_t kMaxMaxCacheableEntryBytes = 1024ull * kMiB;
    constexpr std::uint32_t kMinMaxCacheableEntryPctOfTarget = 5u;
    constexpr std::uint32_t kMaxMaxCacheableEntryPctOfTarget = 50u;
    constexpr std::uint64_t kMinLargeEntryProbationThresholdBytes = 16ull * kMiB;
    constexpr std::uint32_t kMinLargeEntryProbationHitsRequired = 1u;
    constexpr std::uint32_t kMaxLargeEntryProbationHitsRequired = 4u;
    constexpr std::uint64_t kMinHostAssetCacheMaxBytes = 64ull * kMiB;
    constexpr std::uint64_t kMaxHostAssetCacheMaxBytes = 1024ull * kMiB;
    constexpr std::uint64_t kMinHostAssetTrimBatchBytes = 8ull * kMiB;

    ResourceManagerConfigEffective out{};
    out.keySchemaVersion = std::max<std::uint32_t>(1u, raw.keySchemaVersion);
    out.traceSchemaVersion = sanitize_trace_schema_version(raw.traceSchemaVersion);
    out.allowShadowMode = raw.allowShadowMode;
    out.maxLiveManagersPerProcess = std::clamp(
        raw.maxLiveManagersPerProcess,
        kMinLiveManagers,
        kMaxLiveManagers);
    out.managerIdleReapMs = std::clamp(
        raw.managerIdleReapMs,
        kMinIdleReapMs,
        kMaxIdleReapMs);
    out.managerSoftTargetBytes = raw.managerSoftTargetBytes;
    out.managerReserveBytes = std::min(raw.managerReserveBytes, out.managerSoftTargetBytes);
    out.reserveSafetyMarginBytes = std::clamp(
        raw.reserveSafetyMarginBytes,
        kMinReserveSafetyMarginBytes,
        kMaxReserveSafetyMarginBytes);
    out.reserveAdaptUpStepBytes = std::clamp(
        raw.reserveAdaptUpStepBytes,
        kMinReserveAdaptStepBytes,
        kMaxReserveAdaptStepBytes);
    out.reserveAdaptDownStepBytes = std::clamp(
        raw.reserveAdaptDownStepBytes,
        kMinReserveAdaptStepBytes,
        kMaxReserveAdaptStepBytes);
    out.freezeOpportunisticBelowReserve = raw.freezeOpportunisticBelowReserve;
    out.allowActiveFrameBurst = raw.allowActiveFrameBurst;
    out.maxActiveBurstBytes = std::clamp(
        raw.maxActiveBurstBytes,
        kMinMaxActiveBurstBytes,
        kMaxMaxActiveBurstBytes);
    out.maxActiveBurstPctOfTarget = std::clamp(
        raw.maxActiveBurstPctOfTarget,
        kMinMaxActiveBurstPctOfTarget,
        kMaxMaxActiveBurstPctOfTarget);
    out.maxActiveBurstMs = std::clamp(
        raw.maxActiveBurstMs,
        kMinMaxActiveBurstMs,
        kMaxMaxActiveBurstMs);
    out.pressureSampleIntervalMs = std::clamp(
        raw.pressureSampleIntervalMs,
        kMinPressureSampleMs,
        kMaxPressureSampleMs);
    out.pressurePollIntervalMs = std::clamp(
        raw.pressurePollIntervalMs,
        kMinPressurePollIntervalMs,
        kMaxPressurePollIntervalMs);
    out.pressurePollIntervalMsNormal = std::clamp(
        raw.pressurePollIntervalMsNormal,
        kMinPressurePollIntervalMs,
        kMaxPressurePollIntervalMs);
    out.pressurePollIntervalMsCritical = std::clamp(
        raw.pressurePollIntervalMsCritical,
        kMinPressurePollIntervalMs,
        kMaxPressurePollIntervalMs);
    out.pressurePollIntervalMsCritical = std::min(
        out.pressurePollIntervalMsCritical,
        out.pressurePollIntervalMsNormal);
    out.pressureStateMinDwellMs = std::clamp(
        raw.pressureStateMinDwellMs,
        kMinPressureStateMinDwellMs,
        kMaxPressureStateMinDwellMs);
    out.pressureStateMaxTransitionsPerMin = std::clamp(
        raw.pressureStateMaxTransitionsPerMin,
        kMinPressureStateMaxTransitionsPerMin,
        kMaxPressureStateMaxTransitionsPerMin);
    out.reclaimRetryMaxAttempts = std::min(raw.reclaimRetryMaxAttempts, kMaxReclaimRetryAttempts);
    out.tierErrorWindowMs = std::clamp(
        raw.tierErrorWindowMs,
        kMinTierErrorWindowMs,
        kMaxTierErrorWindowMs);
    out.tierErrorThreshold = std::clamp(
        raw.tierErrorThreshold,
        kMinTierErrorThreshold,
        kMaxTierErrorThreshold);
    out.tierCircuitOpenMs = std::clamp(
        raw.tierCircuitOpenMs,
        kMinTierCircuitOpenMs,
        kMaxTierCircuitOpenMs);
    out.fragmentationRecoveryEnabled = raw.fragmentationRecoveryEnabled;
    out.hostAssetCacheMaxBytes = std::clamp(
        raw.hostAssetCacheMaxBytes,
        kMinHostAssetCacheMaxBytes,
        kMaxHostAssetCacheMaxBytes);
    out.hostAssetIdleTrimMs = std::clamp(
        raw.hostAssetIdleTrimMs,
        kMinHostAssetIdleTrimMs,
        kMaxHostAssetIdleTrimMs);
    out.hostAssetTrimBatchBytes = std::clamp(
        raw.hostAssetTrimBatchBytes,
        kMinHostAssetTrimBatchBytes,
        out.hostAssetCacheMaxBytes);
    out.scratchBuilderBytesInFlightLimitMB = std::clamp(
        raw.scratchBuilderBytesInFlightLimitMB,
        kMinScratchBuilderBytesInFlightLimitMB,
        kMaxScratchBuilderBytesInFlightLimitMB);
    out.lutBuilderBytesInFlightLimitMB = std::clamp(
        raw.lutBuilderBytesInFlightLimitMB,
        kMinLutBuilderBytesInFlightLimitMB,
        kMaxLutBuilderBytesInFlightLimitMB);
    out.graphBuilderBytesInFlightLimitMB = std::clamp(
        raw.graphBuilderBytesInFlightLimitMB,
        kMinGraphBuilderBytesInFlightLimitMB,
        kMaxGraphBuilderBytesInFlightLimitMB);
    out.builderFairnessTokensPerTick = std::clamp(
        raw.builderFairnessTokensPerTick,
        kMinBuilderFairnessTokensPerTick,
        kMaxBuilderFairnessTokensPerTick);
    out.criticalBuilderReservedTokens = std::clamp(
        raw.criticalBuilderReservedTokens,
        kMinCriticalBuilderReservedTokens,
        out.builderFairnessTokensPerTick);
    out.uploadBytesInFlightLimitMB = std::clamp(
        raw.uploadBytesInFlightLimitMB,
        kMinUploadBytesInFlightLimitMB,
        kMaxUploadBytesInFlightLimitMB);
    out.uploadFairnessTokensPerTick = std::clamp(
        raw.uploadFairnessTokensPerTick,
        kMinUploadFairnessTokensPerTick,
        kMaxUploadFairnessTokensPerTick);
    out.criticalUploadReservedTokens = std::clamp(
        raw.criticalUploadReservedTokens,
        kMinCriticalUploadReservedTokens,
        out.uploadFairnessTokensPerTick);
    out.maxCacheableEntryBytes = std::clamp(
        raw.maxCacheableEntryBytes,
        kMinMaxCacheableEntryBytes,
        kMaxMaxCacheableEntryBytes);
    out.maxCacheableEntryPctOfTarget = std::clamp(
        raw.maxCacheableEntryPctOfTarget,
        kMinMaxCacheableEntryPctOfTarget,
        kMaxMaxCacheableEntryPctOfTarget);
    out.largeEntryProbationThresholdBytes = std::clamp(
        raw.largeEntryProbationThresholdBytes,
        kMinLargeEntryProbationThresholdBytes,
        out.maxCacheableEntryBytes);
    out.largeEntryProbationHitsRequired = std::clamp(
        raw.largeEntryProbationHitsRequired,
        kMinLargeEntryProbationHitsRequired,
        kMaxLargeEntryProbationHitsRequired);

    std::uint64_t immutableBp = std::min<std::uint64_t>(raw.tierTargetImmutableBp, kBasisPointsDenom);
    std::uint64_t lutBp = std::min<std::uint64_t>(raw.tierTargetLutBp, kBasisPointsDenom);
    std::uint64_t scratchBp = std::min<std::uint64_t>(raw.tierTargetScratchBp, kBasisPointsDenom);
    std::uint64_t graphBp = std::min<std::uint64_t>(raw.tierTargetGraphBp, kBasisPointsDenom);
    std::uint64_t totalBp = immutableBp + lutBp + scratchBp + graphBp;
    if (totalBp == 0) {
        immutableBp = 2500;
        lutBp = 2500;
        scratchBp = 3000;
        graphBp = 2000;
        totalBp = immutableBp + lutBp + scratchBp + graphBp;
    }
    if (totalBp > kBasisPointsDenom) {
        immutableBp = (immutableBp * kBasisPointsDenom) / totalBp;
        lutBp = (lutBp * kBasisPointsDenom) / totalBp;
        scratchBp = (scratchBp * kBasisPointsDenom) / totalBp;
        graphBp = kBasisPointsDenom - (immutableBp + lutBp + scratchBp);
    }

    out.tierTargetImmutableBp = immutableBp;
    out.tierTargetLutBp = lutBp;
    out.tierTargetScratchBp = scratchBp;
    out.tierTargetGraphBp = graphBp;
    return out;
}

} // namespace ResourceManager
} // namespace JuicerCuda
