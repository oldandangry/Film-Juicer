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
