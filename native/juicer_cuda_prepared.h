#pragma once

#include <string>

#include "juicer_cuda_api.h"
#include "Cuda/JuicerCudaExecutor.h"

namespace JuicerCuda {

    bool execute_prepared_host_data(
        const FjPreparedHostData& source,
        const ExecutionFrame& frame,
        ResourceManager::SubmissionSnapshot& snapshot,
        PendingContextLossRecovery& recovery,
        const DirFailureMessage& dirFailureMessage,
        std::string& diagnostic);

} // namespace JuicerCuda
