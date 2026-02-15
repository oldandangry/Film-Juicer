// Cuda/ResourceManager/JuicerCudaResourcePolicy.cpp

#include "Cuda/ResourceManager/JuicerCudaResourcePolicy.h"

#include <algorithm>
#include <limits>

namespace JuicerCuda {
namespace ResourceManager {

namespace {

AcquireDecision make_decision(AcquireStatus status) noexcept {
    AcquireDecision out{};
    out.status = status;
    out.shouldBuild = (status == AcquireStatus::Miss);
    return out;
}

} // namespace

AcquireDecision classify_shadow_acquire(bool hasPrevious, bool invalidated) noexcept {
    if (!hasPrevious) {
        return make_decision(AcquireStatus::Miss);
    }
    if (invalidated) {
        return make_decision(AcquireStatus::Miss);
    }
    return make_decision(AcquireStatus::Hit);
}

ResourcePlan build_shadow_resource_plan(const ShadowKeyDelta& delta) noexcept {
    ResourcePlan plan{};
    for (ResourceKind kind : kResourceKindOrder) {
        ResourcePlanEntry& entry = resource_plan_entry(plan, kind);
        entry.invalidated = delta.keySchemaChanged || shadow_key_changed_for_kind(delta, kind);
        entry.acquire = classify_shadow_acquire(delta.hasPrevious, entry.invalidated);
    }
    return plan;
}

bool shadow_key_changed_for_kind(const ShadowKeyDelta& delta, ResourceKind kind) noexcept {
    switch (kind) {
    case ResourceKind::UploadCore:
        return delta.uploadCoreChanged;
    case ResourceKind::Dir:
        return delta.dirChanged;
    case ResourceKind::Scanner:
        return delta.scannerChanged;
    case ResourceKind::AutoExposure:
        return delta.autoExposureChanged;
    default:
        return true;
    }
}

ResourcePlanEntry& resource_plan_entry(ResourcePlan& plan, ResourceKind kind) noexcept {
    switch (kind) {
    case ResourceKind::UploadCore:
        return plan.uploadCore;
    case ResourceKind::Dir:
        return plan.dir;
    case ResourceKind::Scanner:
        return plan.scanner;
    case ResourceKind::AutoExposure:
        return plan.autoExposure;
    default:
        return plan.uploadCore;
    }
}

const ResourcePlanEntry& resource_plan_entry(const ResourcePlan& plan, ResourceKind kind) noexcept {
    switch (kind) {
    case ResourceKind::UploadCore:
        return plan.uploadCore;
    case ResourceKind::Dir:
        return plan.dir;
    case ResourceKind::Scanner:
        return plan.scanner;
    case ResourceKind::AutoExposure:
        return plan.autoExposure;
    default:
        return plan.uploadCore;
    }
}

const char* to_cstr(AcquireStatus status) noexcept {
    switch (status) {
    case AcquireStatus::Hit:
        return "Hit";
    case AcquireStatus::Miss:
        return "Miss";
    case AcquireStatus::Busy:
        return "Busy";
    case AcquireStatus::Exhausted:
        return "Exhausted";
    case AcquireStatus::Error:
        return "Error";
    default:
        return "Unknown";
    }
}

StaleDecision classify_stale_path(const StaleInput& input) noexcept {
    StaleDecision out{};
    if (input.expectedRegistryGeneration != input.observedRegistryGeneration) {
        out.hardStale = true;
        out.reason = StaleReason::RegistryGenerationMismatch;
        return out;
    }
    if (input.expectedContextEpoch != input.observedContextEpoch) {
        out.hardStale = true;
        out.reason = StaleReason::ContextEpochMismatch;
        return out;
    }
    if (input.expectedLeaseGeneration != input.observedLeaseGeneration) {
        out.hardStale = true;
        out.reason = StaleReason::LeaseGenerationMismatch;
        return out;
    }
    if (input.keySchemaMismatch) {
        out.hardMiss = true;
        out.reason = StaleReason::KeySchemaMismatch;
        return out;
    }
    return out;
}

const char* to_cstr(StaleReason reason) noexcept {
    switch (reason) {
    case StaleReason::None:
        return "None";
    case StaleReason::RegistryGenerationMismatch:
        return "RegistryGenerationMismatch";
    case StaleReason::ContextEpochMismatch:
        return "ContextEpochMismatch";
    case StaleReason::LeaseGenerationMismatch:
        return "LeaseGenerationMismatch";
    case StaleReason::KeySchemaMismatch:
        return "KeySchemaMismatch";
    default:
        return "Unknown";
    }
}

const char* to_cstr(PressureState state) noexcept {
    switch (state) {
    case PressureState::Normal:
        return "Normal";
    case PressureState::Constrained:
        return "Constrained";
    case PressureState::Critical:
        return "Critical";
    case PressureState::Emergency:
        return "Emergency";
    default:
        return "Unknown";
    }
}

const char* to_cstr(ReservationKind kind) noexcept {
    switch (kind) {
    case ReservationKind::TransientNonManager:
        return "TransientNonManager";
    case ReservationKind::UploadCopy:
        return "UploadCopy";
    case ReservationKind::BuilderWork:
        return "BuilderWork";
    default:
        return "Unknown";
    }
}

const char* to_cstr(HeadroomSource source) noexcept {
    switch (source) {
    case HeadroomSource::FreeVramOnly:
        return "free_vram_only";
    case HeadroomSource::AllocatorPool:
        return "allocator_pool";
    default:
        return "unknown";
    }
}

const char* to_cstr(CacheAdmissionClass value) noexcept {
    switch (value) {
    case CacheAdmissionClass::Normal:
        return "Normal";
    case CacheAdmissionClass::Probation:
        return "Probation";
    case CacheAdmissionClass::TooLargeToCache:
        return "TooLargeToCache";
    default:
        return "Unknown";
    }
}

int pressure_state_rank(PressureState state) noexcept {
    switch (state) {
    case PressureState::Normal:
        return 0;
    case PressureState::Constrained:
        return 1;
    case PressureState::Critical:
        return 2;
    case PressureState::Emergency:
        return 3;
    default:
        return 0;
    }
}

PressureState max_pressure_state(PressureState a, PressureState b) noexcept {
    return (pressure_state_rank(a) >= pressure_state_rank(b)) ? a : b;
}

PressureDecision classify_pressure(const PressureInput& input) noexcept {
    PressureDecision out{};
    const std::uint64_t effectiveReserveBytes =
        (input.effectiveReserveBytes > 0) ? input.effectiveReserveBytes : input.reserveBytes;
    out.effectiveReserveBytes = effectiveReserveBytes;
    out.effectiveHeadroomBytes = input.effectiveHeadroomBytes;
    out.headroomSource = input.headroomSource;

    if (input.softTargetBytes == 0) {
        out.state = PressureState::Normal;
        return out;
    }

    const std::uint64_t pressureBytes =
        input.managerResidentBytes + input.retirePendingBytes + input.transientNonManagerBytes;

    const std::uint64_t constrainedThreshold = input.softTargetBytes;
    const std::uint64_t criticalThreshold = input.softTargetBytes + (effectiveReserveBytes / 2u);
    const std::uint64_t emergencyThreshold = input.softTargetBytes + effectiveReserveBytes;

    PressureState budgetState = PressureState::Normal;
    if (pressureBytes >= emergencyThreshold && emergencyThreshold > 0) {
        budgetState = PressureState::Emergency;
    }
    else if (pressureBytes >= criticalThreshold && criticalThreshold > 0) {
        budgetState = PressureState::Critical;
    }
    else if (pressureBytes >= constrainedThreshold) {
        budgetState = PressureState::Constrained;
    }

    PressureState headroomState = PressureState::Normal;
    if (effectiveReserveBytes > 0) {
        const std::uint64_t emergencyHeadroomThreshold = effectiveReserveBytes / 2u;
        const std::uint64_t criticalHeadroomThreshold = effectiveReserveBytes;
        const std::uint64_t constrainedHeadroomThreshold =
            effectiveReserveBytes + (effectiveReserveBytes / 2u);
        if (input.effectiveHeadroomBytes <= emergencyHeadroomThreshold) {
            headroomState = PressureState::Emergency;
        }
        else if (input.effectiveHeadroomBytes <= criticalHeadroomThreshold) {
            headroomState = PressureState::Critical;
        }
        else if (input.effectiveHeadroomBytes <= constrainedHeadroomThreshold) {
            headroomState = PressureState::Constrained;
        }
    }

    out.state = max_pressure_state(budgetState, headroomState);
    switch (out.state) {
    case PressureState::Emergency:
        out.allowOpportunistic = false;
        out.requestReclaimPass = true;
        out.shouldShedNonCritical = true;
        break;
    case PressureState::Critical:
        out.allowOpportunistic = false;
        out.requestReclaimPass = true;
        out.shouldShedNonCritical = false;
        break;
    case PressureState::Constrained:
        out.allowOpportunistic = true;
        out.requestReclaimPass = true;
        out.shouldShedNonCritical = false;
        break;
    case PressureState::Normal:
    default:
        break;
    }

    const bool belowEffectiveReserve = (effectiveReserveBytes > 0) &&
        (input.effectiveHeadroomBytes <= effectiveReserveBytes);
    if (input.freezeOpportunisticBelowReserve && belowEffectiveReserve) {
        out.allowOpportunistic = false;
        out.freezeOpportunistic = true;
        out.requestReclaimPass = true;
        if (pressure_state_rank(out.state) < pressure_state_rank(PressureState::Constrained)) {
            out.state = PressureState::Constrained;
        }
    }
    return out;
}

ReservationDecision classify_reservation(const ReservationInput& input) noexcept {
    ReservationDecision out{};
    out.granted = true;
    out.reason = "granted";

    if (input.capBytes == 0) {
        out.granted = true;
        out.reason = "cap_disabled";
        return out;
    }

    std::uint64_t nextBytes = input.bytesInFlight;
    if (input.requestBytes > (std::numeric_limits<std::uint64_t>::max() - input.bytesInFlight)) {
        nextBytes = std::numeric_limits<std::uint64_t>::max();
    }
    else {
        nextBytes = input.bytesInFlight + input.requestBytes;
    }
    if (nextBytes <= input.capBytes) {
        return out;
    }

    if (input.criticalCurrentFrame) {
        out.granted = true;
        out.reason = "critical_last_resort";
        return out;
    }

    out.granted = false;
    out.shouldWait = true;
    out.waitMs = 1;
    out.reason = "cap_exceeded";
    return out;
}

CacheAdmissionDecision classify_cache_admission(const CacheAdmissionInput& input) noexcept {
    CacheAdmissionDecision out{};
    out.reason = "normal";

    if (input.requestBytes == 0) {
        out.reason = "empty_request";
        return out;
    }

    const std::uint64_t hardMaxBytes = input.maxCacheableEntryBytes;
    std::uint64_t pctMaxBytes = 0;
    if (input.cacheTargetBytes > 0 && input.maxCacheableEntryPctOfTarget > 0) {
        const std::uint64_t pct = static_cast<std::uint64_t>(input.maxCacheableEntryPctOfTarget);
        const std::uint64_t maxU64 = std::numeric_limits<std::uint64_t>::max();
        const std::uint64_t numerator =
            (input.cacheTargetBytes > (maxU64 / pct))
            ? maxU64
            : (input.cacheTargetBytes * pct);
        pctMaxBytes = numerator / 100ull;
    }

    std::uint64_t maxDurableBytes = std::numeric_limits<std::uint64_t>::max();
    if (hardMaxBytes > 0 && pctMaxBytes > 0) {
        maxDurableBytes = std::min<std::uint64_t>(hardMaxBytes, pctMaxBytes);
    }
    else if (hardMaxBytes > 0) {
        maxDurableBytes = hardMaxBytes;
    }
    else if (pctMaxBytes > 0) {
        maxDurableBytes = pctMaxBytes;
    }
    out.maxDurableBytes = maxDurableBytes;

    const bool tooLarge =
        (maxDurableBytes != std::numeric_limits<std::uint64_t>::max()) &&
        (input.requestBytes > maxDurableBytes);
    if (tooLarge) {
        out.admissionClass = CacheAdmissionClass::TooLargeToCache;
        if (input.criticalCurrentFrame) {
            out.allowDurableAdmission = true;
            out.reason = "too_large_critical_override";
        }
        else {
            out.allowDurableAdmission = false;
            out.reason = "too_large_noncritical";
        }
        return out;
    }

    const std::uint64_t probationThreshold = input.largeEntryProbationThresholdBytes;
    if (probationThreshold > 0 && input.requestBytes > probationThreshold) {
        out.admissionClass = CacheAdmissionClass::Probation;
        out.probationApplied = true;
        const std::uint32_t requiredHits = std::max<std::uint32_t>(1u, input.largeEntryProbationHitsRequired);
        out.probationHitsRequired = requiredHits;
        const std::uint32_t observed =
            (input.observedProbationHits < std::numeric_limits<std::uint32_t>::max())
            ? (input.observedProbationHits + 1u)
            : std::numeric_limits<std::uint32_t>::max();

        if (input.criticalCurrentFrame) {
            out.allowDurableAdmission = true;
            out.reason = "probation_critical_override";
            return out;
        }

        if (observed < requiredHits) {
            out.allowDurableAdmission = false;
            out.reason = "probation_defer";
            return out;
        }

        out.allowDurableAdmission = true;
        out.reason = "probation_admit";
        return out;
    }

    return out;
}

BurstDebtDecision classify_burst_debt(const BurstDebtInput& input) noexcept {
    BurstDebtDecision out{};
    out.enabled = (input.burstDebtHalfLifeMs > 0) && (input.maxBurstDebtPct < 100);
    if (!out.enabled) {
        out.reason = "disabled";
        return out;
    }

    if (input.criticalCurrentFrame && input.burstConsumed) {
        out.accrueDebt = true;
        std::uint32_t incrementPct = 4;
        if (input.burstCapBytes > 0 && input.burstOverTargetBytes > 0) {
            const std::uint64_t ratioPctU64 = std::min<std::uint64_t>(
                100ull,
                (input.burstOverTargetBytes * 100ull) / input.burstCapBytes);
            const std::uint32_t ratioPct = static_cast<std::uint32_t>(ratioPctU64);
            incrementPct = std::max<std::uint32_t>(1u, 1u + (ratioPct / 10u));
        }
        out.debtIncrementPct = std::min<std::uint32_t>(25u, incrementPct);
        out.reason = "accrue_burst_consumed";
        return out;
    }

    if (!input.criticalCurrentFrame &&
        input.currentDebtPct >= input.maxBurstDebtPct) {
        out.throttleOpportunistic = true;
        out.reason = "throttle_max_debt";
        return out;
    }

    out.reason = input.criticalCurrentFrame
        ? "critical_no_burst_consumption"
        : "below_debt_threshold";
    return out;
}

AcquireDecision default_acquire_decision() noexcept {
    return AcquireDecision{};
}

PressureDecision default_pressure_decision() noexcept {
    return PressureDecision{};
}

ReservationDecision default_reservation_decision() noexcept {
    return ReservationDecision{};
}

CacheAdmissionDecision default_cache_admission_decision() noexcept {
    return CacheAdmissionDecision{};
}

} // namespace ResourceManager
} // namespace JuicerCuda
