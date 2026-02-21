// Cuda/ResourceManager/JuicerCudaResourceFoundation.cpp
//
// Consolidated foundation ownership for config/key/state/telemetry modules.

#include "Cuda/ResourceManager/JuicerCudaResourceConfig.h"
#include "Cuda/ResourceManager/JuicerCudaResourceKeys.h"
#include "Cuda/ResourceManager/JuicerCudaResourceState.h"
#include "Cuda/ResourceManager/JuicerCudaResourceTelemetry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

#include "Hash.h"
#include "Logging.h"

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
    constexpr std::uint32_t kMinPinnedStagingIdleTrimMs = 500u;
    constexpr std::uint32_t kMaxPinnedStagingIdleTrimMs = 60000u;
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
    constexpr std::uint64_t kMinGraphLargeEntryThresholdBytes = 16ull * kMiB;
    constexpr std::uint64_t kMaxGraphLargeEntryThresholdBytes = 2048ull * kMiB;
    constexpr std::uint64_t kMinGraphLargeEntryQuarantineMaxBytes = 32ull * kMiB;
    constexpr std::uint64_t kMaxGraphLargeEntryQuarantineMaxBytes = 2048ull * kMiB;
    constexpr std::uint32_t kMinGraphLargeEntryQuarantineMaxEntries = 1u;
    constexpr std::uint32_t kMaxGraphLargeEntryQuarantineMaxEntries = 16u;
    constexpr std::uint64_t kMinLargeEntryProbationThresholdBytes = 16ull * kMiB;
    constexpr std::uint32_t kMinLargeEntryProbationHitsRequired = 1u;
    constexpr std::uint32_t kMaxLargeEntryProbationHitsRequired = 4u;
    constexpr std::uint32_t kMinKeepHotMs = 0u;
    constexpr std::uint32_t kMaxKeepHotMs = 5000u;
    constexpr std::uint32_t kMinAdmissionChurnWindowMs = 0u;
    constexpr std::uint32_t kMaxAdmissionChurnWindowMs = 60000u;
    constexpr std::uint32_t kMinAdmissionChurnOneHitRatePct = 1u;
    constexpr std::uint32_t kMaxAdmissionChurnOneHitRatePct = 100u;
    constexpr std::uint32_t kMinAdmissionChurnProbationHitBonus = 0u;
    constexpr std::uint32_t kMaxAdmissionChurnProbationHitBonus = 4u;
    constexpr std::uint32_t kMinLargeEntryReadmitCooldownMs = 0u;
    constexpr std::uint32_t kMaxLargeEntryReadmitCooldownMs = 60000u;
    constexpr std::uint32_t kMinLargeEntryGhostHitsForReadmit = 0u;
    constexpr std::uint32_t kMaxLargeEntryGhostHitsForReadmit = 8u;
    constexpr std::uint32_t kMinBurstDebtHalfLifeMs = 0u;
    constexpr std::uint32_t kMaxBurstDebtHalfLifeMs = 60000u;
    constexpr std::uint32_t kMinMaxBurstDebtPct = 1u;
    constexpr std::uint32_t kMaxMaxBurstDebtPct = 100u;
    constexpr std::uint64_t kMinHostAssetCacheMaxBytes = 64ull * kMiB;
    constexpr std::uint64_t kMaxHostAssetCacheMaxBytes = 1024ull * kMiB;
    constexpr std::uint64_t kMinHostAssetTrimBatchBytes = 8ull * kMiB;
    constexpr std::uint32_t kMinAllocatorBackendPreference = 0u;
    constexpr std::uint32_t kMaxAllocatorBackendPreference = 3u;
    constexpr std::uint32_t kMinAsyncMempoolReleaseThresholdMB = 0u;
    constexpr std::uint32_t kMaxAsyncMempoolReleaseThresholdMB = 4096u;
    constexpr std::uint32_t kMinPrivateLutFallbackPerMediumCap = 0u;
    constexpr std::uint32_t kMaxPrivateLutFallbackPerMediumCap = 1u;
    constexpr std::uint32_t kMinPrivateLutFallbackPerInstanceCap = 0u;
    constexpr std::uint32_t kMaxPrivateLutFallbackPerInstanceCap = 2u;
    constexpr std::uint64_t kMinPinnedStagingMaxBytes = 8ull * kMiB;
    constexpr std::uint64_t kMaxPinnedStagingMaxBytes = 2048ull * kMiB;
    constexpr std::uint64_t kMinPinnedStagingTrimBatchBytes = 1ull * kMiB;

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
    out.allocatorBackendPreference = std::clamp(
        raw.allocatorBackendPreference,
        kMinAllocatorBackendPreference,
        kMaxAllocatorBackendPreference);
    out.asyncMempoolReleaseThresholdMB = std::clamp(
        raw.asyncMempoolReleaseThresholdMB,
        kMinAsyncMempoolReleaseThresholdMB,
        kMaxAsyncMempoolReleaseThresholdMB);
    out.privateLutFallbackPerMediumCap = std::clamp(
        raw.privateLutFallbackPerMediumCap,
        kMinPrivateLutFallbackPerMediumCap,
        kMaxPrivateLutFallbackPerMediumCap);
    out.privateLutFallbackPerInstanceCap = std::clamp(
        raw.privateLutFallbackPerInstanceCap,
        kMinPrivateLutFallbackPerInstanceCap,
        kMaxPrivateLutFallbackPerInstanceCap);
    out.privateLutFallbackPerInstanceCap = std::max(
        out.privateLutFallbackPerInstanceCap,
        out.privateLutFallbackPerMediumCap);
    out.pinnedUploadStagingMaxBytes = std::clamp(
        raw.pinnedUploadStagingMaxBytes,
        kMinPinnedStagingMaxBytes,
        kMaxPinnedStagingMaxBytes);
    out.pinnedUploadStagingIdleTrimMs = std::clamp(
        raw.pinnedUploadStagingIdleTrimMs,
        kMinPinnedStagingIdleTrimMs,
        kMaxPinnedStagingIdleTrimMs);
    out.pinnedUploadStagingTrimBatchBytes = std::clamp(
        raw.pinnedUploadStagingTrimBatchBytes,
        kMinPinnedStagingTrimBatchBytes,
        out.pinnedUploadStagingMaxBytes);
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
    out.graphLargeEntryThresholdBytes = std::clamp(
        raw.graphLargeEntryThresholdBytes,
        kMinGraphLargeEntryThresholdBytes,
        kMaxGraphLargeEntryThresholdBytes);
    out.graphLargeEntryQuarantineMaxBytes = std::clamp(
        raw.graphLargeEntryQuarantineMaxBytes,
        kMinGraphLargeEntryQuarantineMaxBytes,
        kMaxGraphLargeEntryQuarantineMaxBytes);
    out.graphLargeEntryQuarantineMaxEntries = std::clamp(
        raw.graphLargeEntryQuarantineMaxEntries,
        kMinGraphLargeEntryQuarantineMaxEntries,
        kMaxGraphLargeEntryQuarantineMaxEntries);
    out.graphLargeEntryQuarantineMaxBytes = std::max<std::uint64_t>(
        out.graphLargeEntryQuarantineMaxBytes,
        out.graphLargeEntryThresholdBytes);
    out.largeEntryProbationThresholdBytes = std::clamp(
        raw.largeEntryProbationThresholdBytes,
        kMinLargeEntryProbationThresholdBytes,
        out.maxCacheableEntryBytes);
    out.largeEntryProbationHitsRequired = std::clamp(
        raw.largeEntryProbationHitsRequired,
        kMinLargeEntryProbationHitsRequired,
        kMaxLargeEntryProbationHitsRequired);
    out.keepHotMs = std::clamp(
        raw.keepHotMs,
        kMinKeepHotMs,
        kMaxKeepHotMs);
    out.admissionChurnWindowMs = std::clamp(
        raw.admissionChurnWindowMs,
        kMinAdmissionChurnWindowMs,
        kMaxAdmissionChurnWindowMs);
    out.admissionChurnEnterOneHitRatePct = std::clamp(
        raw.admissionChurnEnterOneHitRatePct,
        kMinAdmissionChurnOneHitRatePct,
        kMaxAdmissionChurnOneHitRatePct);
    out.admissionChurnExitOneHitRatePct = std::clamp(
        raw.admissionChurnExitOneHitRatePct,
        kMinAdmissionChurnOneHitRatePct,
        out.admissionChurnEnterOneHitRatePct);
    out.admissionChurnProbationHitBonus = std::clamp(
        raw.admissionChurnProbationHitBonus,
        kMinAdmissionChurnProbationHitBonus,
        kMaxAdmissionChurnProbationHitBonus);
    out.largeEntryReadmitCooldownMs = std::clamp(
        raw.largeEntryReadmitCooldownMs,
        kMinLargeEntryReadmitCooldownMs,
        kMaxLargeEntryReadmitCooldownMs);
    out.largeEntryGhostHitsForReadmit = std::clamp(
        raw.largeEntryGhostHitsForReadmit,
        kMinLargeEntryGhostHitsForReadmit,
        kMaxLargeEntryGhostHitsForReadmit);
    out.burstDebtHalfLifeMs = std::clamp(
        raw.burstDebtHalfLifeMs,
        kMinBurstDebtHalfLifeMs,
        kMaxBurstDebtHalfLifeMs);
    out.maxBurstDebtPct = std::clamp(
        raw.maxBurstDebtPct,
        kMinMaxBurstDebtPct,
        kMaxMaxBurstDebtPct);
    out.cancelSupersededBuilders = raw.cancelSupersededBuilders;

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



std::uint64_t normalize_key_u64(std::uint64_t value) noexcept {
    if (value == 0) {
        return 1;
    }
    return value;
}

std::uint64_t normalize_key_float(double value, double scale) noexcept {
    if (!std::isfinite(value) || !std::isfinite(scale) || scale <= 0.0) {
        return 1;
    }
    const double scaled = value * scale;
    if (!std::isfinite(scaled)) {
        return 1;
    }
    const long long quantized = static_cast<long long>(std::llround(scaled));
    const std::uint64_t raw = static_cast<std::uint64_t>(quantized);
    return normalize_key_u64(raw);
}

std::uint32_t normalize_scan_lut_resolution(std::uint32_t value) noexcept {
    return std::clamp(value, kScanLutResolutionMin, kScanLutResolutionMax);
}

std::uint64_t make_scan_lut_key_digest(
    std::uint32_t medium,
    std::uint64_t tablesHash,
    std::uint64_t densityRangeHash,
    std::uint32_t lutResolution,
    std::uint32_t lutFormatVersion,
    std::uint32_t keySchemaVersion) noexcept {
    if (medium > 1u ||
        tablesHash == 0 ||
        densityRangeHash == 0 ||
        lutFormatVersion == 0 ||
        keySchemaVersion == 0) {
        return 0;
    }

    const std::uint64_t fields[] = {
        static_cast<std::uint64_t>(keySchemaVersion),
        static_cast<std::uint64_t>(medium),
        tablesHash,
        densityRangeHash,
        static_cast<std::uint64_t>(normalize_scan_lut_resolution(lutResolution)),
        static_cast<std::uint64_t>(lutFormatVersion)
    };
    return Hash::hash_bytes(fields, sizeof(fields));
}

KeyDigests make_key_digests(
    std::uint64_t uploadCoreHash,
    std::uint64_t dirHash,
    std::uint64_t scannerHash,
    std::uint64_t autoExposureHash) noexcept {
    KeyDigests digests{};
    digests.uploadCoreHash = normalize_key_u64(uploadCoreHash);
    digests.dirHash = normalize_key_u64(dirHash);
    digests.scannerHash = normalize_key_u64(scannerHash);
    digests.autoExposureHash = normalize_key_u64(autoExposureHash);
    return digests;
}

std::uint64_t key_digest_for_kind(const KeyDigests& digests, ResourceKind kind) noexcept {
    switch (kind) {
    case ResourceKind::UploadCore:
        return digests.uploadCoreHash;
    case ResourceKind::Dir:
        return digests.dirHash;
    case ResourceKind::Scanner:
        return digests.scannerHash;
    case ResourceKind::AutoExposure:
        return digests.autoExposureHash;
    default:
        return 1;
    }
}

void set_key_digest_for_kind(KeyDigests& digests, ResourceKind kind, std::uint64_t hashValue) noexcept {
    const std::uint64_t normalized = normalize_key_u64(hashValue);
    switch (kind) {
    case ResourceKind::UploadCore:
        digests.uploadCoreHash = normalized;
        return;
    case ResourceKind::Dir:
        digests.dirHash = normalized;
        return;
    case ResourceKind::Scanner:
        digests.scannerHash = normalized;
        return;
    case ResourceKind::AutoExposure:
        digests.autoExposureHash = normalized;
        return;
    default:
        return;
    }
}

KeyDigests normalize_key_digests(const KeyDigests& digests) noexcept {
    KeyDigests out = digests;
    for (ResourceKind kind : kResourceKindOrder) {
        set_key_digest_for_kind(out, kind, key_digest_for_kind(out, kind));
    }
    return out;
}



namespace {

struct LatestSnapshotKey {
    std::uint64_t instanceToken = 0;
    DeviceContextKey deviceContextKey{};

    bool operator==(const LatestSnapshotKey& other) const noexcept {
        return instanceToken == other.instanceToken &&
            deviceContextKey == other.deviceContextKey;
    }
};

struct LatestSnapshotKeyHasher {
    std::size_t operator()(const LatestSnapshotKey& key) const noexcept {
        const std::size_t hInstance = std::hash<std::uint64_t>{}(key.instanceToken);
        const std::size_t hContext = DeviceContextKeyHash{}(key.deviceContextKey);
        return hInstance ^ (hContext + 0x9e3779b9u + (hInstance << 6u) + (hInstance >> 2u));
    }
};

struct LatestSnapshotState {
    std::mutex mutex;
    std::unordered_map<LatestSnapshotKey, std::uint64_t, LatestSnapshotKeyHasher> bySubmissionKey;
};

LatestSnapshotState& latest_snapshot_state() {
    static LatestSnapshotState state{};
    return state;
}

QueryReadOnlySnapshot take_query_read_only_snapshot() noexcept {
    QueryReadOnlySnapshot snapshot{};
    snapshot.threadMutationActive = metadata_mutation_thread_active();
    snapshot.threadMutationDepth = metadata_mutation_thread_depth();
    snapshot.threadMutationTicket = metadata_mutation_thread_ticket();
    snapshot.threadMutationBeginCount = metadata_mutation_thread_begin_count();
    return snapshot;
}

bool query_read_only_snapshot_equal(
    const QueryReadOnlySnapshot& lhs,
    const QueryReadOnlySnapshot& rhs) noexcept {
    return lhs.threadMutationActive == rhs.threadMutationActive &&
        lhs.threadMutationDepth == rhs.threadMutationDepth &&
        lhs.threadMutationTicket == rhs.threadMutationTicket &&
        lhs.threadMutationBeginCount == rhs.threadMutationBeginCount;
}

const char* query_mutation_reason(
    const QueryReadOnlySnapshot& before,
    const QueryReadOnlySnapshot& after) noexcept {
    if (!before.threadMutationActive && after.threadMutationActive) {
        return "thread_entered_metadata_mutation";
    }
    if (before.threadMutationDepth != after.threadMutationDepth) {
        return "thread_mutation_depth_changed";
    }
    if (before.threadMutationTicket != after.threadMutationTicket) {
        return "thread_mutation_ticket_changed";
    }
    if (before.threadMutationBeginCount != after.threadMutationBeginCount) {
        return "thread_mutation_begin_count_changed";
    }
    return "unknown";
}

} // namespace

ResourceManagerState& global_state() noexcept {
    static ResourceManagerState state{};
    return state;
}

StaleInput state_build_stale_input(
    const SubmissionTransaction& transaction,
    LeaseObservationMode leaseObservationMode) noexcept {
    StaleInput staleInput{};
    ResourceManagerState& state = global_state();
    staleInput.expectedRegistryGeneration = transaction.snapshot.registryGeneration;
    staleInput.observedRegistryGeneration = state.registryGeneration.load(std::memory_order_relaxed);
    staleInput.expectedContextEpoch = transaction.snapshot.contextEpoch;
    staleInput.observedContextEpoch = state.contextEpoch.load(std::memory_order_relaxed);
    staleInput.expectedLeaseGeneration = transaction.leaseGeneration;
    const bool observeLease = (leaseObservationMode == LeaseObservationMode::Always) || transaction.active;
    staleInput.observedLeaseGeneration = observeLease ? transaction.leaseGeneration : 0;
    staleInput.keySchemaMismatch = (transaction.snapshot.keySchemaVersion == 0);
    return staleInput;
}

QueryReadOnlyGuard::QueryReadOnlyGuard(
    const char* queryName,
    const DeviceContextKey* key) noexcept
    : _queryName(queryName)
    , _key(key)
    , _before(take_query_read_only_snapshot()) {
}

QueryReadOnlyGuard::~QueryReadOnlyGuard() noexcept {
    const QueryReadOnlySnapshot after = take_query_read_only_snapshot();
    if (query_read_only_snapshot_equal(_before, after)) {
        return;
    }

    const char* reason = query_mutation_reason(_before, after);
    telemetry_record_query_mutation_violation();
    telemetry_record_module_boundary_violation();
    telemetry_trace_query_mutation_violation(
        _queryName,
        _key,
        _before.threadMutationDepth,
        after.threadMutationDepth,
        _before.threadMutationTicket,
        after.threadMutationTicket,
        _before.threadMutationBeginCount,
        after.threadMutationBeginCount,
        reason);
}

void state_record_acquire_status_for_kind(ResourceKind kind, AcquireStatus status) noexcept {
    ResourceManagerState& state = global_state();
    const std::size_t kindIndex = resource_kind_index(kind);
    if (kindIndex >= kResourceKindCount) {
        telemetry_record_module_boundary_violation();
        return;
    }
    ResourceKindAcquireCounters& counters = state.acquireStatusByKind[kindIndex];
    switch (status) {
    case AcquireStatus::Hit:
        counters.hit.fetch_add(1, std::memory_order_relaxed);
        break;
    case AcquireStatus::Miss:
        counters.miss.fetch_add(1, std::memory_order_relaxed);
        break;
    case AcquireStatus::Busy:
        counters.busy.fetch_add(1, std::memory_order_relaxed);
        break;
    case AcquireStatus::Exhausted:
        counters.exhausted.fetch_add(1, std::memory_order_relaxed);
        break;
    case AcquireStatus::Error:
    default:
        counters.error.fetch_add(1, std::memory_order_relaxed);
        break;
    }
}

void state_note_latest_snapshot(const SubmissionSnapshot& snapshot) noexcept {
    const std::uint64_t instanceToken = snapshot.instanceToken.value;
    if (instanceToken == 0 || snapshot.snapshotId == 0) {
        return;
    }

    const LatestSnapshotKey key{ instanceToken, snapshot.deviceContextKey };
    LatestSnapshotState& state = latest_snapshot_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    std::uint64_t& latest = state.bySubmissionKey[key];
    if (snapshot.snapshotId > latest) {
        latest = snapshot.snapshotId;
    }
}

bool state_snapshot_is_superseded(
    const SubmissionSnapshot& snapshot,
    std::uint64_t* outLatestSnapshotId) noexcept {
    if (outLatestSnapshotId) {
        *outLatestSnapshotId = 0;
    }

    const std::uint64_t instanceToken = snapshot.instanceToken.value;
    if (instanceToken == 0 || snapshot.snapshotId == 0) {
        return false;
    }

    const LatestSnapshotKey key{ instanceToken, snapshot.deviceContextKey };
    LatestSnapshotState& state = latest_snapshot_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.bySubmissionKey.find(key);
    if (it == state.bySubmissionKey.end()) {
        return false;
    }
    if (outLatestSnapshotId) {
        *outLatestSnapshotId = it->second;
    }
    return it->second > snapshot.snapshotId;
}

void state_clear_latest_snapshot_for_context(const DeviceContextKey& key) noexcept {
    LatestSnapshotState& state = latest_snapshot_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    for (auto it = state.bySubmissionKey.begin(); it != state.bySubmissionKey.end();) {
        if (it->first.deviceContextKey == key) {
            it = state.bySubmissionKey.erase(it);
        }
        else {
            ++it;
        }
    }
}



void telemetry_record_begin_submission() noexcept {
    global_state().beginSubmissionCalls.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_acquire_plan() noexcept {
    global_state().acquirePlanCalls.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_commit_submission() noexcept {
    global_state().commitSubmissionCalls.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_rollback_submission() noexcept {
    global_state().rollbackSubmissionCalls.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_acquire_status(AcquireStatus status) noexcept {
    ResourceManagerState& state = global_state();
    switch (status) {
    case AcquireStatus::Hit:
        state.acquireStatusHit.fetch_add(1, std::memory_order_relaxed);
        break;
    case AcquireStatus::Miss:
        state.acquireStatusMiss.fetch_add(1, std::memory_order_relaxed);
        break;
    case AcquireStatus::Busy:
        state.acquireStatusBusy.fetch_add(1, std::memory_order_relaxed);
        break;
    case AcquireStatus::Exhausted:
        state.acquireStatusExhausted.fetch_add(1, std::memory_order_relaxed);
        break;
    case AcquireStatus::Error:
    default:
        state.acquireStatusError.fetch_add(1, std::memory_order_relaxed);
        break;
    }
}

void telemetry_record_acquire_status_for_kind(ResourceKind kind, AcquireStatus status) noexcept {
    state_record_acquire_status_for_kind(kind, status);
}

void telemetry_record_trace_schema_mismatch() noexcept {
    global_state().traceSchemaMismatchEvents.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_forbidden_invalidation_edge() noexcept {
    global_state().forbiddenInvalidationEdges.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_module_boundary_violation() noexcept {
    global_state().moduleBoundaryViolations.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_query_mutation_violation() noexcept {
    global_state().queryMutationViolationEvents.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_frame_snapshot_mismatch() noexcept {
    global_state().frameSnapshotMismatchEvents.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_stale_tuple_hard_reject() noexcept {
    global_state().staleTupleHardRejects.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_metadata_mutation_begin() noexcept {
    global_state().metadataMutationBeginCalls.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_metadata_mutation_end() noexcept {
    global_state().metadataMutationEndCalls.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_metadata_mutation_reject() noexcept {
    global_state().metadataMutationRejects.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_metadata_mutation_order_violation() noexcept {
    global_state().metadataMutationOrderViolations.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_metadata_queue_enqueue() noexcept {
    global_state().metadataMutationQueueEnqueueCalls.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_metadata_queue_dequeue() noexcept {
    global_state().metadataMutationQueueDequeueCalls.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_metadata_queue_wait() noexcept {
    global_state().metadataMutationQueueWaitEvents.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_metadata_queue_backpressure() noexcept {
    global_state().metadataMutationQueueBackpressureEvents.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_record_metadata_queue_reject() noexcept {
    global_state().metadataMutationQueueRejects.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_note_metadata_queue_depth(std::uint64_t depth) noexcept {
    std::atomic<std::uint64_t>& gauge = global_state().metadataMutationQueueMaxDepth;
    std::uint64_t observed = gauge.load(std::memory_order_relaxed);
    while (depth > observed) {
        if (gauge.compare_exchange_weak(observed, depth, std::memory_order_relaxed)) {
            break;
        }
    }
}

std::uint64_t telemetry_next_acquire_attempt_id() noexcept {
    ResourceManagerState& state = global_state();
    std::uint64_t id = state.nextAcquireAttemptId.fetch_add(1, std::memory_order_relaxed);
    if (id == 0) {
        id = state.nextAcquireAttemptId.fetch_add(1, std::memory_order_relaxed);
    }
    return id;
}

constexpr const char* kTraceTokenUnknown = "unknown";
constexpr const char* kTraceTokenUnspecified = "unspecified";

const char* trace_token_or(const char* value, const char* fallback) noexcept {
    return value ? value : fallback;
}

void telemetry_trace_schema_announcement(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion) noexcept {
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    const std::string msg =
        std::string("transaction_id=") + std::to_string(transactionId) +
        " snapshot_id=" + std::to_string(snapshotId) +
        " trace_schema=" + std::to_string(traceSchemaVersion) +
        " expected_schema=" + std::to_string(kTraceSchemaVersion);
    JTRACE("MSTRC", msg);
}

void telemetry_trace_schema_mismatch(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t observedTraceSchemaVersion) noexcept {
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    const std::string msg =
        std::string("transaction_id=") + std::to_string(transactionId) +
        " snapshot_id=" + std::to_string(snapshotId) +
        " observed_schema=" + std::to_string(observedTraceSchemaVersion) +
        " expected_schema=" + std::to_string(kTraceSchemaVersion) +
        " reason=trace_schema_mismatch";
    JTRACE("MSTRC", msg);
}

void telemetry_trace_key_normalization(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    const KeyDigests& before,
    const KeyDigests& after) noexcept {
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    const std::string msg =
        std::string("transaction_id=") + std::to_string(transactionId) +
        " snapshot_id=" + std::to_string(snapshotId) +
        " trace_schema=" + std::to_string(traceSchemaVersion) +
        " upload_before=" + std::to_string(before.uploadCoreHash) +
        " upload_after=" + std::to_string(after.uploadCoreHash) +
        " dir_before=" + std::to_string(before.dirHash) +
        " dir_after=" + std::to_string(after.dirHash) +
        " scanner_before=" + std::to_string(before.scannerHash) +
        " scanner_after=" + std::to_string(after.scannerHash) +
        " auto_exposure_before=" + std::to_string(before.autoExposureHash) +
        " auto_exposure_after=" + std::to_string(after.autoExposureHash) +
        " reason=canonical_normalization";
    JTRACE("MSNORM", msg);
}

void telemetry_trace_invalidation(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    const char* lane,
    const char* reason,
    std::uint64_t previousHash,
    std::uint64_t currentHash) noexcept {
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    const std::string msg =
        std::string("transaction_id=") + std::to_string(transactionId) +
        " snapshot_id=" + std::to_string(snapshotId) +
        " trace_schema=" + std::to_string(traceSchemaVersion) +
        " lane=" + trace_token_or(lane, kTraceTokenUnknown) +
        " previous_hash=" + std::to_string(previousHash) +
        " current_hash=" + std::to_string(currentHash) +
        " reason=" + trace_token_or(reason, kTraceTokenUnknown);
    JTRACE("MSINV", msg);
}

void telemetry_trace_dag_edge(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    const char* fromNode,
    const char* toNode,
    bool allowed,
    const char* reason) noexcept {
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    const std::string msg =
        std::string("transaction_id=") + std::to_string(transactionId) +
        " snapshot_id=" + std::to_string(snapshotId) +
        " trace_schema=" + std::to_string(traceSchemaVersion) +
        " from=" + trace_token_or(fromNode, kTraceTokenUnknown) +
        " to=" + trace_token_or(toNode, kTraceTokenUnknown) +
        " allowed=" + std::to_string(allowed ? 1 : 0) +
        " reason=" + trace_token_or(reason, kTraceTokenUnspecified);
    JTRACE("MSDAG", msg);
}

void telemetry_trace_module_boundary_violation(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    const char* reason) noexcept {
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    const std::string msg =
        std::string("transaction_id=") + std::to_string(transactionId) +
        " snapshot_id=" + std::to_string(snapshotId) +
        " trace_schema=" + std::to_string(traceSchemaVersion) +
        " reason=" + trace_token_or(reason, kTraceTokenUnknown);
    JTRACE("MSCMD", msg);
}

void telemetry_trace_query_mutation_violation(
    const char* queryName,
    const DeviceContextKey* key,
    std::uint32_t beforeThreadMutationDepth,
    std::uint32_t afterThreadMutationDepth,
    std::uint64_t beforeThreadMutationTicket,
    std::uint64_t afterThreadMutationTicket,
    std::uint64_t beforeThreadMutationBeginCount,
    std::uint64_t afterThreadMutationBeginCount,
    const char* reason) noexcept {
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    const int deviceId = key ? key->deviceId : -1;
    const std::uintptr_t contextBits = key
        ? reinterpret_cast<std::uintptr_t>(key->contextOpaque)
        : 0;
    const std::string msg =
        std::string("event=query_mutation_violation") +
        " query=" + trace_token_or(queryName, kTraceTokenUnknown) +
        " reason=" + trace_token_or(reason, kTraceTokenUnknown) +
        " device_id=" + std::to_string(deviceId) +
        " context=" + std::to_string(contextBits) +
        " before_thread_mutation_depth=" + std::to_string(beforeThreadMutationDepth) +
        " after_thread_mutation_depth=" + std::to_string(afterThreadMutationDepth) +
        " before_thread_mutation_ticket=" + std::to_string(beforeThreadMutationTicket) +
        " after_thread_mutation_ticket=" + std::to_string(afterThreadMutationTicket) +
        " before_thread_mutation_begin_count=" + std::to_string(beforeThreadMutationBeginCount) +
        " after_thread_mutation_begin_count=" + std::to_string(afterThreadMutationBeginCount);
    JTRACE("MSCMD", msg);
}

void telemetry_trace_frame_snapshot_mismatch(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    std::uint64_t frameToken,
    std::uint64_t expectedSnapshotId,
    std::uint64_t observedSnapshotId) noexcept {
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    const std::string msg =
        std::string("transaction_id=") + std::to_string(transactionId) +
        " snapshot_id=" + std::to_string(snapshotId) +
        " trace_schema=" + std::to_string(traceSchemaVersion) +
        " frame_token=" + std::to_string(frameToken) +
        " expected_snapshot_id=" + std::to_string(expectedSnapshotId) +
        " observed_snapshot_id=" + std::to_string(observedSnapshotId) +
        " reason=mixed_snapshot_id_for_frame";
    JTRACE("MSSNP", msg);
}

void telemetry_trace_stale_decision(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    const char* stage,
    const StaleInput& input,
    const StaleDecision& decision) noexcept {
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    const std::string msg =
        std::string("transaction_id=") + std::to_string(transactionId) +
        " snapshot_id=" + std::to_string(snapshotId) +
        " trace_schema=" + std::to_string(traceSchemaVersion) +
        " stage=" + trace_token_or(stage, kTraceTokenUnknown) +
        " expected_registry_generation=" + std::to_string(input.expectedRegistryGeneration) +
        " observed_registry_generation=" + std::to_string(input.observedRegistryGeneration) +
        " expected_context_epoch=" + std::to_string(input.expectedContextEpoch) +
        " observed_context_epoch=" + std::to_string(input.observedContextEpoch) +
        " expected_lease_generation=" + std::to_string(input.expectedLeaseGeneration) +
        " observed_lease_generation=" + std::to_string(input.observedLeaseGeneration) +
        " key_schema_mismatch=" + std::to_string(input.keySchemaMismatch ? 1 : 0) +
        " hard_stale=" + std::to_string(decision.hardStale ? 1 : 0) +
        " hard_miss=" + std::to_string(decision.hardMiss ? 1 : 0) +
        " reason=" + to_cstr(decision.reason);
    JTRACE("MSSTL", msg);
}

void telemetry_trace_metadata_mutation(
    const char* phase,
    const char* stage,
    std::uint64_t sequence,
    bool accepted,
    std::uint64_t expectedSequence,
    const char* reason) noexcept {
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    const std::string msg =
        std::string("phase=") + trace_token_or(phase, kTraceTokenUnknown) +
        " stage=" + trace_token_or(stage, kTraceTokenUnknown) +
        " sequence=" + std::to_string(sequence) +
        " expected_sequence=" + std::to_string(expectedSequence) +
        " accepted=" + std::to_string(accepted ? 1 : 0) +
        " reason=" + trace_token_or(reason, kTraceTokenUnspecified);
    JTRACE("MSMUT", msg);
}

void telemetry_trace_metadata_queue(
    const char* eventName,
    const char* stage,
    std::uint64_t ticket,
    std::uint64_t depth,
    std::uint64_t waitedMs,
    bool accepted,
    const char* reason) noexcept {
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    const std::string msg =
        std::string("event=") + trace_token_or(eventName, kTraceTokenUnknown) +
        " stage=" + trace_token_or(stage, kTraceTokenUnknown) +
        " ticket=" + std::to_string(ticket) +
        " depth=" + std::to_string(depth) +
        " waited_ms=" + std::to_string(waitedMs) +
        " accepted=" + std::to_string(accepted ? 1 : 0) +
        " reason=" + trace_token_or(reason, kTraceTokenUnspecified);
    JTRACE("MSMQ", msg);
}

void telemetry_trace_acquire(
    std::uint64_t acquireId,
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    AcquireStatus finalStatus,
    const ResourcePlan& plan,
    bool hadPreviousSnapshot) noexcept {
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    std::string msg;
    msg.reserve(192u + (kResourceKindOrder.size() * 96u));
    msg =
        std::string("acquire_id=") + std::to_string(acquireId) +
        " transaction_id=" + std::to_string(transactionId) +
        " snapshot_id=" + std::to_string(snapshotId) +
        " trace_schema=" + std::to_string(traceSchemaVersion) +
        " final_status=" + to_cstr(finalStatus) +
        " had_previous=" + std::to_string(hadPreviousSnapshot ? 1 : 0);
    for (ResourceKind kind : kResourceKindOrder) {
        const ResourcePlanEntry& entry = resource_plan_entry(plan, kind);
        const ResourceKindContractEntry& contract = resource_kind_contract_entry(kind);
        msg += std::string(" ") + contract.acquireStatusField + "=" + to_cstr(entry.acquire.status);
        msg += std::string(" ") + contract.invalidationLane + "_invalidated=" +
            std::to_string(entry.invalidated ? 1 : 0);
    }
    JTRACE("MSACQ", msg);
}

void telemetry_trace_auto_exposure_ownership(
    std::uint64_t transactionId,
    std::uint64_t snapshotId,
    std::uint32_t traceSchemaVersion,
    const char* mode,
    const char* eventName,
    bool hit,
    std::uint64_t keyHash,
    int meterWidth,
    int meterHeight,
    bool hadPrevious,
    const char* reason) noexcept {
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    const std::string msg =
        std::string("transaction_id=") + std::to_string(transactionId) +
        " snapshot_id=" + std::to_string(snapshotId) +
        " trace_schema=" + std::to_string(traceSchemaVersion) +
        " mode=" + trace_token_or(mode, kTraceTokenUnknown) +
        " event=" + trace_token_or(eventName, kTraceTokenUnknown) +
        " hit=" + std::to_string(hit ? 1 : 0) +
        " key_hash=" + std::to_string(keyHash) +
        " meter_w=" + std::to_string(meterWidth) +
        " meter_h=" + std::to_string(meterHeight) +
        " had_previous=" + std::to_string(hadPrevious ? 1 : 0) +
        " reason=" + trace_token_or(reason, kTraceTokenUnspecified);
    JTRACE("MSAEX", msg);
}


} // namespace ResourceManager
} // namespace JuicerCuda
