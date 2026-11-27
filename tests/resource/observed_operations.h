#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <cuda_runtime_api.h>
#include <cufft.h>

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
        std::size_t uploadAllocations = 0;
        std::size_t copies = 0;
        std::size_t copyBytes = 0;
        cudaError_t nextCreateUploadEvent = cudaSuccess;
        cudaError_t nextFreeHost = cudaSuccess;
        cudaError_t nextDestroyEvent = cudaSuccess;
        std::size_t successfulHostFrees = 0;
        std::size_t successfulEventDestroys = 0;
        cudaError_t nextCopy = cudaSuccess;
        void (*beforeCopy)(cudaStream_t, void*) = nullptr;
        void* beforeCopyUser = nullptr;
        cufftResult nextSpectrumTransform = CUFFT_SUCCESS;
        bool nextScanDirFailure = false;
        std::size_t scanErrorReadbacks = 0;
        cudaError_t nextRecord = cudaSuccess;
        cudaError_t nextSynchronization = cudaSuccess;
        void (*beforeSynchronization)(void*) = nullptr;
        void* beforeSynchronizationUser = nullptr;
        std::optional<SynchronizationFault> scheduledSynchronization;
        cudaStream_t failedStream = nullptr;
        cudaError_t streamWaitFault = cudaSuccess;
        cudaError_t nextContextSynchronization = cudaSuccess;
        std::size_t contextSynchronizations = 0;
        void (*beforeContextSynchronization)(void*) = nullptr;
        void* beforeContextSynchronizationUser = nullptr;
        std::array<cudaError_t, 2> deviceFreeFaults{};
        std::size_t deviceFreeFaultIndex = 0;
        void (*beforeLastDeviceFreeFault)() noexcept = nullptr;
        std::size_t synchronizations = 0;
        std::size_t eventRecords = 0;
        cudaEvent_t lastEvent = nullptr;
        void* lastStaging = nullptr;
    };
    extern thread_local Calls calls;
} // namespace JuicerTest
