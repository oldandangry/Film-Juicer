#pragma once

#include <string>

#include "juicer_cuda_api.h"
#include "Cuda/JuicerCudaExecutor.h"

struct FjHostPreparation;

namespace JuicerCuda {

    // The route and both pointer/count representations must already be validated.
    std::uint64_t spectral_binding_digest(const FjPreparedHostData& source);

    bool execute_prepared_host_data(
        JuicerProcess::Root& root,
        const FjPreparedHostData& source,
        const ExecutionFrame& frame,
        ResourceManager::SubmissionSnapshot& snapshot,
        PendingContextLossRecovery& recovery,
        FjAbortCallback abortCallback,
        std::string& diagnostic,
        const FjHostPreparation* hostPreparation = nullptr);

} // namespace JuicerCuda
