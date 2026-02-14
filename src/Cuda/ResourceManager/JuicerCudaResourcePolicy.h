// Cuda/ResourceManager/JuicerCudaResourcePolicy.h
//
// Phase-0 policy scaffolding.
#pragma once

#include <cstdint>

#include "Cuda/ResourceManager/JuicerCudaResourceTypes.h"

namespace JuicerCuda {
namespace ResourceManager {

enum class AcquireStatus : std::uint8_t {
    Hit = 0,
    Miss = 1,
    Busy = 2,
    Exhausted = 3,
    Error = 4
};

enum class StaleReason : std::uint8_t {
    None = 0,
    RegistryGenerationMismatch = 1,
    ContextEpochMismatch = 2,
    LeaseGenerationMismatch = 3,
    KeySchemaMismatch = 4
};

enum class PressureState : std::uint8_t {
    Normal = 0,
    Constrained = 1,
    Critical = 2,
    Emergency = 3
};

enum class ReservationKind : std::uint8_t {
    TransientNonManager = 0,
    UploadCopy = 1,
    BuilderWork = 2
};

enum class HeadroomSource : std::uint8_t {
    FreeVramOnly = 0,
    AllocatorPool = 1
};

enum class CacheAdmissionClass : std::uint8_t {
    Normal = 0,
    Probation = 1,
    TooLargeToCache = 2
};

struct AcquireDecision {
    AcquireStatus status = AcquireStatus::Miss;
    bool shouldBuild = true;
};

struct StaleInput {
    std::uint64_t expectedRegistryGeneration = 0;
    std::uint64_t observedRegistryGeneration = 0;
    std::uint64_t expectedContextEpoch = 0;
    std::uint64_t observedContextEpoch = 0;
    std::uint64_t expectedLeaseGeneration = 0;
    std::uint64_t observedLeaseGeneration = 0;
    bool keySchemaMismatch = false;
};

struct StaleDecision {
    bool hardStale = false;
    bool hardMiss = false;
    StaleReason reason = StaleReason::None;
};

struct ShadowKeyDelta {
    bool hasPrevious = false;
    bool keySchemaChanged = false;
    bool uploadCoreChanged = false;
    bool dirChanged = false;
    bool scannerChanged = false;
    bool autoExposureChanged = false;
};

struct ResourcePlanEntry {
    AcquireDecision acquire{};
    bool invalidated = false;
};

struct ResourcePlan {
    ResourcePlanEntry uploadCore{};
    ResourcePlanEntry dir{};
    ResourcePlanEntry scanner{};
    ResourcePlanEntry autoExposure{};
};

struct PressureInput {
    std::uint64_t softTargetBytes = 0;
    std::uint64_t reserveBytes = 0;
    std::uint64_t effectiveReserveBytes = 0;
    bool freezeOpportunisticBelowReserve = true;
    std::uint64_t managerResidentBytes = 0;
    std::uint64_t retirePendingBytes = 0;
    std::uint64_t transientNonManagerBytes = 0;
    std::uint64_t effectiveHeadroomBytes = 0;
    std::uint64_t driverFreeBytes = 0;
    std::uint64_t allocatorPoolReservedBytes = 0;
    std::uint64_t allocatorPoolUsedBytes = 0;
    HeadroomSource headroomSource = HeadroomSource::FreeVramOnly;
};

struct PressureDecision {
    PressureState state = PressureState::Normal;
    bool allowOpportunistic = true;
    bool freezeOpportunistic = false;
    bool requestReclaimPass = false;
    bool shouldShedNonCritical = false;
    std::uint64_t effectiveReserveBytes = 0;
    std::uint64_t effectiveHeadroomBytes = 0;
    HeadroomSource headroomSource = HeadroomSource::FreeVramOnly;
};

struct ReservationInput {
    ReservationKind kind = ReservationKind::TransientNonManager;
    std::uint64_t requestBytes = 0;
    std::uint64_t bytesInFlight = 0;
    std::uint64_t capBytes = 0;
    bool criticalCurrentFrame = false;
};

struct ReservationDecision {
    bool granted = true;
    bool shouldWait = false;
    std::uint32_t waitMs = 0;
    const char* reason = "granted";
};

struct CacheAdmissionInput {
    std::uint64_t requestBytes = 0;
    std::uint64_t cacheTargetBytes = 0;
    std::uint64_t maxCacheableEntryBytes = 0;
    std::uint32_t maxCacheableEntryPctOfTarget = 0;
    std::uint64_t largeEntryProbationThresholdBytes = 0;
    std::uint32_t largeEntryProbationHitsRequired = 2;
    std::uint32_t observedProbationHits = 0;
    bool criticalCurrentFrame = false;
};

struct CacheAdmissionDecision {
    CacheAdmissionClass admissionClass = CacheAdmissionClass::Normal;
    bool allowDurableAdmission = true;
    bool probationApplied = false;
    std::uint32_t probationHitsRequired = 0;
    std::uint64_t maxDurableBytes = 0;
    const char* reason = "normal";
};

AcquireDecision classify_shadow_acquire(bool hasPrevious, bool invalidated) noexcept;
ResourcePlan build_shadow_resource_plan(const ShadowKeyDelta& delta) noexcept;
bool shadow_key_changed_for_kind(const ShadowKeyDelta& delta, ResourceKind kind) noexcept;
ResourcePlanEntry& resource_plan_entry(ResourcePlan& plan, ResourceKind kind) noexcept;
const ResourcePlanEntry& resource_plan_entry(const ResourcePlan& plan, ResourceKind kind) noexcept;
const char* to_cstr(AcquireStatus status) noexcept;
StaleDecision classify_stale_path(const StaleInput& input) noexcept;
const char* to_cstr(StaleReason reason) noexcept;
const char* to_cstr(PressureState state) noexcept;
const char* to_cstr(ReservationKind kind) noexcept;
const char* to_cstr(HeadroomSource source) noexcept;
const char* to_cstr(CacheAdmissionClass value) noexcept;
PressureDecision classify_pressure(const PressureInput& input) noexcept;
ReservationDecision classify_reservation(const ReservationInput& input) noexcept;
CacheAdmissionDecision classify_cache_admission(const CacheAdmissionInput& input) noexcept;

AcquireDecision default_acquire_decision() noexcept;
PressureDecision default_pressure_decision() noexcept;
ReservationDecision default_reservation_decision() noexcept;
CacheAdmissionDecision default_cache_admission_decision() noexcept;

} // namespace ResourceManager
} // namespace JuicerCuda
