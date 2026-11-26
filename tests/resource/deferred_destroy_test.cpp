#include <array>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda.h>
#include <cuda_runtime.h>

#include "Cuda/JuicerCudaResourceOperations.h"
#include "Cuda/JuicerCudaResources.h"

namespace {
    bool failFree = false;
    unsigned freeAttempts = 0;

    void require(bool condition, const char* message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }

    JuicerCuda::Resources* allocate(const JuicerCuda::ResourceManager::DeviceContextKey& key,
                                    const std::shared_ptr<JuicerCuda::DeviceAllocationLedger>& ledger) {
        auto resources = std::make_unique<JuicerCuda::Resources>(key, 1, ledger);
        JuicerCuda::Failure failure;
        JuicerCuda::DeviceByteReservation reservation;
        require(ledger->reserve({key, 1, 4096}, reservation, failure), "reserve 4 KiB");
        void* allocation = nullptr;
        require(cudaMalloc(&allocation, 4096) == cudaSuccess, "allocate 4 KiB");
        std::string diagnostic;
        require(reservation.commit(4096, diagnostic), "commit 4 KiB");
        resources->deviceAllocationRecords.emplace(allocation, std::move(reservation));
        return resources.release();
    }

    void check() {
        require(cuInit(0) == CUDA_SUCCESS, "initialize CUDA driver");
        CUdevice device{};
        require(cuDeviceGet(&device, 0) == CUDA_SUCCESS, "select device");
        std::array<CUcontext, 3> contexts{};
        for (auto& context : contexts) {
            require(cuCtxCreate(&context, nullptr, 0, device) == CUDA_SUCCESS, "create fixture context");
        }
        const JuicerCuda::ResourceManager::DeviceContextKey key{0, contexts[0]};
        const JuicerCuda::ResourceManager::DeviceContextKey unrelated{0, contexts[1]};
        std::string diagnostic;
        auto ledger = JuicerCuda::DeviceAllocationLedger::create({0, 65536}, diagnostic);
        require(ledger && ledger->bind_or_validate_cap(65536, diagnostic), "bind ledger cap");
        require(cuCtxSetCurrent(contexts[0]) == CUDA_SUCCESS, "make owner current");
        std::array<JuicerCuda::Resources*, 2> resources{allocate(key, ledger), allocate(key, ledger)};
        require(cuCtxSetCurrent(contexts[1]) == CUDA_SUCCESS, "make unrelated owner current");
        auto* const other = allocate(unrelated, ledger);
        require(cuCtxSetCurrent(contexts[2]) == CUDA_SUCCESS, "make foreign context current");
        for (auto* resource : resources) {
            JuicerCuda::destroy(resource);
        }
        JuicerCuda::destroy(other);
        require(ledger->snapshot().chargedBytes == 12288 && ledger->snapshot().recordCount == 3, "deferred accounting before drain");
        require(cuCtxSetCurrent(contexts[0]) == CUDA_SUCCESS, "restore exact owner");
        // A runtime call fails after the exact-context batch has been detached.
        failFree = true;
        JuicerCuda::Failure failure;
        require(!JuicerCuda::drain_deferred_resources(key, 1, failure), "faulted drain succeeded");
        require(failure.status.category == FJ_STATUS_CUDA_FAILURE &&
                    failure.status.api == FJ_API_CUDA_RUNTIME &&
                    failure.status.native_code == cudaErrorInvalidValue,
                "drain lost the typed CUDA free failure");
        require(ledger->snapshot().chargedBytes == 12288 && ledger->snapshot().recordCount == 3,
                "failure discarded the batch or its ledger charges");
        require(freeAttempts == 1, "failed drain retried or freed a later entry");
        require(cuCtxSetCurrent(contexts[0]) == CUDA_SUCCESS, "restore owner after fault");
        require(JuicerCuda::drain_deferred_resources(key, 1, failure), "explicit drain after clearing fault");
        std::printf("after retry charged=%llu records=%llu attempts=%u\n",
                    static_cast<unsigned long long>(ledger->snapshot().chargedBytes),
                    static_cast<unsigned long long>(ledger->snapshot().recordCount),
                    freeAttempts);
        require(ledger->snapshot().chargedBytes == 4096 && ledger->snapshot().recordCount == 1, "failed or unprocessed owner lost, duplicated or unrelated owner freed");
        require(ledger->record_count_for_context(unrelated, 1) == 1, "unrelated exact-context entry changed");
        require(JuicerCuda::drain_deferred_resources(key, 1, failure), "empty drain failed");
        require(cuCtxSetCurrent(contexts[1]) == CUDA_SUCCESS, "restore unrelated owner");
        require(JuicerCuda::drain_deferred_resources(unrelated, 1, failure), "unrelated deferred drain");
        require(ledger->snapshot().chargedBytes == 0 && ledger->snapshot().recordCount == 0, "deferred ownership did not drain to zero");
        require(freeAttempts == 4u, "deferred entries duplicated or reprocessed");
        for (auto* context : contexts) {
            require(cuCtxDestroy(context) == CUDA_SUCCESS, "destroy fixture-owned context");
        }
    }
} // namespace

namespace JuicerCuda::ResourceOperations {
    void before_render_admission(FjCuda*, const FjCudaContext*, const FjSubmission*) {}

    cudaError_t allocate_upload(void** destination, std::size_t bytes) {
        return cudaMallocHost(destination, bytes);
    }
    cudaError_t create_upload_event(cudaEvent_t* event) {
        return cudaEventCreateWithFlags(event, cudaEventDisableTiming);
    }
    cudaError_t copy_upload(void* destination, const void* source, std::size_t bytes, cudaStream_t stream) {
        return cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream);
    }
    cudaError_t read_scan_error(int* host, const int* device, cudaStream_t stream) {
        return cudaMemcpyAsync(host, device, sizeof(*host), cudaMemcpyDeviceToHost, stream);
    }
    cudaError_t record_event(cudaEvent_t event, cudaStream_t stream) {
        return cudaEventRecord(event, stream);
    }
    cudaError_t synchronize_stream(cudaStream_t stream) {
        return cudaStreamSynchronize(stream);
    }

    cudaError_t free_device(void* pointer) {
        ++freeAttempts;
        return std::exchange(failFree, false) ? cudaErrorInvalidValue : cudaFree(pointer);
    }
} // namespace JuicerCuda::ResourceOperations

int main() {
    try {
        check();
        std::puts("PASS deferred resource ownership");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL deferred resource ownership: %s\n", error.what());
        return 1;
    }
}
