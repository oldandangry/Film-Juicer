// Cuda/ResourceManager/JuicerCudaResourceTelemetry.cpp

#include "Cuda/ResourceManager/JuicerCudaResourceTelemetry.h"
#include "Cuda/ResourceManager/JuicerCudaResourceState.h"

namespace JuicerCuda {
namespace ResourceManager {

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

} // namespace ResourceManager
} // namespace JuicerCuda

