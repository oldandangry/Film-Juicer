#pragma once

#include <cstddef>
#include <optional>
#include <cuda_runtime_api.h>

namespace JuicerTest {
    struct SynchronizationFault {
        std::size_t call;
        cudaError_t code;
    };

    struct Calls {
        std::size_t copies = 0;
        std::size_t copyBytes = 0;
        cudaError_t nextCopy = cudaSuccess;
        cudaError_t nextRecord = cudaSuccess;
        cudaError_t nextSynchronization = cudaSuccess;
        std::optional<SynchronizationFault> scheduledSynchronization;
        std::size_t synchronizations = 0;
        std::size_t eventRecords = 0;
        cudaEvent_t lastEvent = nullptr;
        void* lastStaging = nullptr;
    };
    extern thread_local Calls calls;
} // namespace JuicerTest
