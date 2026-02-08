// Cuda/ResourceManager/JuicerCudaResourcePolicy.cpp

#include "Cuda/ResourceManager/JuicerCudaResourcePolicy.h"

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
    const bool uploadInvalidated = delta.keySchemaChanged || delta.uploadCoreChanged;
    const bool dirInvalidated = delta.keySchemaChanged || delta.dirChanged;
    const bool scannerInvalidated = delta.keySchemaChanged || delta.scannerChanged;

    plan.uploadCore.invalidated = uploadInvalidated;
    plan.dir.invalidated = dirInvalidated;
    plan.scanner.invalidated = scannerInvalidated;

    plan.uploadCore.acquire = classify_shadow_acquire(delta.hasPrevious, uploadInvalidated);
    plan.dir.acquire = classify_shadow_acquire(delta.hasPrevious, dirInvalidated);
    plan.scanner.acquire = classify_shadow_acquire(delta.hasPrevious, scannerInvalidated);
    return plan;
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

AcquireDecision default_acquire_decision() noexcept {
    return AcquireDecision{};
}

PressureDecision default_pressure_decision() noexcept {
    return PressureDecision{};
}

ReservationDecision default_reservation_decision() noexcept {
    return ReservationDecision{};
}

} // namespace ResourceManager
} // namespace JuicerCuda
