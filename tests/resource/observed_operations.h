#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <cuda_runtime_api.h>

struct FjCuda;
struct FjCudaContext;
struct FjSubmission;

namespace JuicerTest {
    struct SynchronizationFault {
        std::size_t call;
        cudaError_t code;
    };

    struct Calls {
        std::function<void(FjCuda*, const FjCudaContext&, const FjSubmission&)> beforeRenderAdmission;
        std::size_t copies = 0;
        std::size_t copyBytes = 0;
        cudaError_t nextCopy = cudaSuccess;
        void (*beforeCopy)(cudaStream_t, void*) = nullptr;
        void* beforeCopyUser = nullptr;
        bool nextScanDirFailure = false;
        std::size_t scanErrorReadbacks = 0;
        cudaError_t nextRecord = cudaSuccess;
        cudaError_t nextSynchronization = cudaSuccess;
        void (*beforeSynchronization)(void*) = nullptr;
        void* beforeSynchronizationUser = nullptr;
        std::optional<SynchronizationFault> scheduledSynchronization;
        std::size_t synchronizations = 0;
        std::size_t eventRecords = 0;
        cudaEvent_t lastEvent = nullptr;
        void* lastStaging = nullptr;
    };
    extern thread_local Calls calls;
} // namespace JuicerTest
