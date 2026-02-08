// Cuda/ResourceManager/JuicerCudaResourceManager.h
//
// Phase-0 submission transaction scaffolding.
#pragma once

#include <string>

#include "Cuda/ResourceManager/JuicerCudaResourceTypes.h"

struct WorkingState;
namespace Print {
    struct Runtime;
    struct Params;
}

namespace JuicerCuda {
struct Resources;
namespace ResourceManager {

bool query_submission_active(const SubmissionTransaction& transaction) noexcept;

bool begin_submission(
    SubmissionTransaction& outTransaction,
    const SubmissionSnapshot& snapshot,
    std::string& outError);

bool acquire_plan(
    SubmissionTransaction& transaction,
    std::string& outError);

bool commit_submission(
    SubmissionTransaction& transaction,
    void* cudaStreamOpaque,
    std::string& outError);

bool command_freeze_drain_bump_resume(
    const DeviceContextKey& key,
    const char* reason,
    std::string& outError);

bool command_ensure_uploaded(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    void* cudaStreamOpaque,
    std::string& outError);

bool command_ensure_scan_lut(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    bool negativeMedium,
    void* cudaStreamOpaque,
    std::string& outError);

bool command_ensure_scan_error_flag(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    void* cudaStreamOpaque,
    std::string& outError);

bool command_ensure_print_illuminant_filtered(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    const Print::Runtime& prt,
    const Print::Params& params,
    void* cudaStreamOpaque,
    std::string& outError);

void rollback_submission(
    SubmissionTransaction& transaction,
    const char* reason) noexcept;

} // namespace ResourceManager
} // namespace JuicerCuda
