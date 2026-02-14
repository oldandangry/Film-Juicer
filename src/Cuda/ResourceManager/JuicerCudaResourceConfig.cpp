// Cuda/ResourceManager/JuicerCudaResourceConfig.cpp

#include "Cuda/ResourceManager/JuicerCudaResourceConfig.h"

#include <algorithm>

namespace JuicerCuda {
namespace ResourceManager {

ResourceManagerConfigEffective sanitize_config(const ResourceManagerConfigRaw& raw) {
    constexpr std::uint64_t kBasisPointsDenom = 10000ull;
    constexpr std::uint32_t kMinPressureSampleMs = 10u;
    constexpr std::uint32_t kMaxPressureSampleMs = 5000u;
    constexpr std::uint32_t kMaxReclaimRetryAttempts = 3u;

    ResourceManagerConfigEffective out{};
    out.keySchemaVersion = std::max<std::uint32_t>(1u, raw.keySchemaVersion);
    out.traceSchemaVersion = sanitize_trace_schema_version(raw.traceSchemaVersion);
    out.allowShadowMode = raw.allowShadowMode;
    out.managerSoftTargetBytes = raw.managerSoftTargetBytes;
    out.managerReserveBytes = std::min(raw.managerReserveBytes, out.managerSoftTargetBytes);
    out.pressureSampleIntervalMs = std::clamp(
        raw.pressureSampleIntervalMs,
        kMinPressureSampleMs,
        kMaxPressureSampleMs);
    out.reclaimRetryMaxAttempts = std::min(raw.reclaimRetryMaxAttempts, kMaxReclaimRetryAttempts);

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
