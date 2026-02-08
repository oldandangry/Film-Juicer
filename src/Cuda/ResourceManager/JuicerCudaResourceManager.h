// Cuda/ResourceManager/JuicerCudaResourceManager.h
//
// Phase-0 submission transaction scaffolding.
#pragma once

#include <string>

#include "Cuda/ResourceManager/JuicerCudaResourceTypes.h"

namespace JuicerCuda {
namespace ResourceManager {

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

void rollback_submission(
    SubmissionTransaction& transaction,
    const char* reason) noexcept;

} // namespace ResourceManager
} // namespace JuicerCuda

