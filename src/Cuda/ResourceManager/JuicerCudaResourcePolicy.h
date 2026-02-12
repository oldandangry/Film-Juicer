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

struct PressureDecision {
    bool allowOpportunistic = true;
};

struct ReservationDecision {
    bool granted = true;
};

AcquireDecision classify_shadow_acquire(bool hasPrevious, bool invalidated) noexcept;
ResourcePlan build_shadow_resource_plan(const ShadowKeyDelta& delta) noexcept;
bool shadow_key_changed_for_kind(const ShadowKeyDelta& delta, ResourceKind kind) noexcept;
ResourcePlanEntry& resource_plan_entry(ResourcePlan& plan, ResourceKind kind) noexcept;
const ResourcePlanEntry& resource_plan_entry(const ResourcePlan& plan, ResourceKind kind) noexcept;
const char* to_cstr(AcquireStatus status) noexcept;
StaleDecision classify_stale_path(const StaleInput& input) noexcept;
const char* to_cstr(StaleReason reason) noexcept;

AcquireDecision default_acquire_decision() noexcept;
PressureDecision default_pressure_decision() noexcept;
ReservationDecision default_reservation_decision() noexcept;

} // namespace ResourceManager
} // namespace JuicerCuda
