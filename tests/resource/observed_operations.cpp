#include "observed_operations.h"

#include <utility>
#include "Cuda/JuicerCudaResourceOperations.h"

namespace JuicerTest {
    thread_local Calls calls;
} // namespace JuicerTest

namespace {
    void CUDART_CB write_dir_failure(void* host) {
        constexpr int kDirError = 1 << 8;
        constexpr int kDirChannelMask = 0b010 << 9;
        *static_cast<int*>(host) = kDirError | kDirChannelMask;
    }
} // namespace

namespace JuicerCuda::ResourceOperations {
    using JuicerTest::calls;

    void before_render_admission(FjCuda* owner, const FjCudaContext* context, const FjSubmission* submission) {
        if (calls.beforeRenderAdmission && owner && context && submission) {
            calls.beforeRenderAdmission(owner, *context, *submission);
        }
    }

    cudaError_t allocate_upload(void** destination, std::size_t bytes) {
        return cudaMallocHost(destination, bytes);
    }

    cudaError_t create_upload_event(cudaEvent_t* event) {
        return cudaEventCreateWithFlags(event, cudaEventDisableTiming);
    }

    cudaError_t copy_upload(void* destination, const void* source, std::size_t bytes, cudaStream_t stream) {
        const auto fault = std::exchange(calls.nextCopy, cudaSuccess);
        if (fault != cudaSuccess) {
            return fault;
        }
        if (const auto before = std::exchange(calls.beforeCopy, nullptr)) {
            before(stream, std::exchange(calls.beforeCopyUser, nullptr));
        }
        const auto result = cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream);
        if (result == cudaSuccess) {
            ++calls.copies;
            calls.copyBytes += bytes;
            calls.lastStaging = const_cast<void*>(source);
        }
        return result;
    }

    cudaError_t read_scan_error(int* host, const int* device, cudaStream_t stream) {
        // The production event and pinned lease still fence the injected readback.
        const auto result = std::exchange(calls.nextScanDirFailure, false)
                                ? cudaLaunchHostFunc(stream, &write_dir_failure, host)
                                : cudaMemcpyAsync(host, device, sizeof(*host), cudaMemcpyDeviceToHost, stream);
        if (result == cudaSuccess) {
            ++calls.scanErrorReadbacks;
        }
        return result;
    }

    cudaError_t record_event(cudaEvent_t event, cudaStream_t stream) {
        const auto fault = std::exchange(calls.nextRecord, cudaSuccess);
        if (fault != cudaSuccess) {
            return fault;
        }
        const auto result = cudaEventRecord(event, stream);
        if (result == cudaSuccess) {
            ++calls.eventRecords;
            calls.lastEvent = event;
        }
        return result;
    }

    cudaError_t synchronize_stream(cudaStream_t stream) {
        ++calls.synchronizations;
        if (const auto before = std::exchange(calls.beforeSynchronization, nullptr)) {
            before(std::exchange(calls.beforeSynchronizationUser, nullptr));
        }
        if (calls.scheduledSynchronization && calls.scheduledSynchronization->call == calls.synchronizations) {
            const auto code = calls.scheduledSynchronization->code;
            calls.scheduledSynchronization.reset();
            return code;
        }
        const auto fault = std::exchange(calls.nextSynchronization, cudaSuccess);
        return fault == cudaSuccess ? cudaStreamSynchronize(stream) : fault;
    }

    cudaError_t free_device(void* pointer) {
        return cudaFree(pointer);
    }
} // namespace JuicerCuda::ResourceOperations
