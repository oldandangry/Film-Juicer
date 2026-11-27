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
        ++calls.uploadAllocations;
        return cudaMallocHost(destination, bytes);
    }

    cudaError_t create_upload_event(cudaEvent_t* event) {
        const auto result = cudaEventCreateWithFlags(event, cudaEventDisableTiming);
        const auto fault = std::exchange(calls.nextCreateUploadEvent, cudaSuccess);
        return result == cudaSuccess && fault != cudaSuccess ? fault : result;
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

    cudaError_t query_event(cudaEvent_t event) noexcept {
        return cudaEventQuery(event);
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
        if (fault != cudaSuccess) {
            return fault;
        }
        if (calls.streamWaitFault != cudaSuccess && stream == calls.failedStream) {
            return calls.streamWaitFault;
        }
        return cudaStreamSynchronize(stream);
    }

    cudaError_t synchronize_context() noexcept {
        ++calls.contextSynchronizations;
        if (const auto before = std::exchange(calls.beforeContextSynchronization, nullptr)) {
            before(std::exchange(calls.beforeContextSynchronizationUser, nullptr));
        }
        const auto fault = std::exchange(calls.nextContextSynchronization, cudaSuccess);
        return fault == cudaSuccess ? cudaDeviceSynchronize() : fault;
    }

    cudaError_t free_device(void* pointer) {
        if (calls.deviceFreeFaultIndex < calls.deviceFreeFaults.size()) {
            const auto fault = calls.deviceFreeFaults[calls.deviceFreeFaultIndex++];
            if (fault != cudaSuccess) {
                if (calls.deviceFreeFaultIndex == calls.deviceFreeFaults.size() && calls.beforeLastDeviceFreeFault) {
                    std::exchange(calls.beforeLastDeviceFreeFault, nullptr)();
                }
                return fault;
            }
        }
        return cudaFree(pointer);
    }

    cudaError_t free_host(void* pointer) noexcept {
        const auto fault = std::exchange(calls.nextFreeHost, cudaSuccess);
        const auto result = fault == cudaSuccess ? cudaFreeHost(pointer) : fault;
        calls.successfulHostFrees += result == cudaSuccess;
        return result;
    }

    cudaError_t destroy_event(cudaEvent_t event) noexcept {
        const auto fault = std::exchange(calls.nextDestroyEvent, cudaSuccess);
        const auto result = fault == cudaSuccess ? cudaEventDestroy(event) : fault;
        calls.successfulEventDestroys += result == cudaSuccess;
        return result;
    }

    cufftResult destroy_plan(cufftHandle plan) noexcept {
        return cufftDestroy(plan);
    }

    cufftResult execute_spectrum_r2c(cufftHandle plan, cufftReal* input, cufftComplex* output) noexcept {
        const auto fault = std::exchange(calls.nextSpectrumTransform, CUFFT_SUCCESS);
        return fault == CUFFT_SUCCESS ? cufftExecR2C(plan, input, output) : fault;
    }
} // namespace JuicerCuda::ResourceOperations
