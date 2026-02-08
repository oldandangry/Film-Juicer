// Cuda/ResourceManager/JuicerCudaResourceTelemetry.h
//
// Phase-0 telemetry scaffolding for fixed counters and trace schema contract.
#pragma once

#include <cstdint>

namespace JuicerCuda {
namespace ResourceManager {

constexpr std::uint32_t kTraceSchemaVersion = 1;

void telemetry_record_begin_submission() noexcept;
void telemetry_record_acquire_plan() noexcept;
void telemetry_record_commit_submission() noexcept;
void telemetry_record_rollback_submission() noexcept;

} // namespace ResourceManager
} // namespace JuicerCuda

