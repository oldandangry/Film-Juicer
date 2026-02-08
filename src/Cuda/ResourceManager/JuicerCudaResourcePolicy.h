// Cuda/ResourceManager/JuicerCudaResourcePolicy.h
//
// Phase-0 policy scaffolding.
#pragma once

#include <cstdint>

namespace JuicerCuda {
namespace ResourceManager {

enum class AcquireStatus : std::uint8_t {
    Hit = 0,
    Miss = 1,
    Busy = 2,
    Exhausted = 3,
    Error = 4
};

struct AcquireDecision {
    AcquireStatus status = AcquireStatus::Miss;
    bool shouldBuild = true;
};

struct ShadowKeyDelta {
    bool hasPrevious = false;
    bool keySchemaChanged = false;
    bool uploadCoreChanged = false;
    bool dirChanged = false;
    bool scannerChanged = false;
};

struct ResourcePlanEntry {
    AcquireDecision acquire{};
    bool invalidated = false;
};

struct ResourcePlan {
    ResourcePlanEntry uploadCore{};
    ResourcePlanEntry dir{};
    ResourcePlanEntry scanner{};
};

struct PressureDecision {
    bool allowOpportunistic = true;
};

struct ReservationDecision {
    bool granted = true;
};

AcquireDecision classify_shadow_acquire(bool hasPrevious, bool invalidated) noexcept;
ResourcePlan build_shadow_resource_plan(const ShadowKeyDelta& delta) noexcept;
const char* to_cstr(AcquireStatus status) noexcept;

AcquireDecision default_acquire_decision() noexcept;
PressureDecision default_pressure_decision() noexcept;
ReservationDecision default_reservation_decision() noexcept;

} // namespace ResourceManager
} // namespace JuicerCuda
