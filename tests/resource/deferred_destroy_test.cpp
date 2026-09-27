#include <array>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <cuda.h>
#include <cuda_runtime.h>

#include "Cuda/JuicerCudaResources.h"

namespace {
    bool failDrain = false;
    bool rejectAllocation = false;
    CUcontext mismatchContext = nullptr;
    unsigned drainAttempts = 0;
    bool failAllocation = false;
    const std::system_error servingLockFailure{std::make_error_code(std::errc::invalid_argument)};

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

    void check(const std::string& mode) {
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
        const bool control = mode == "control";
        failDrain = !control;
        failAllocation = mode == "allocation" || mode == "storage";
        mismatchContext = mode == "ordinary" ? contexts[2] : nullptr;
        rejectAllocation = mode == "storage";
        // The hook throws at the actual drain/serving-lock boundary after queue
        // extraction. Either selected entry may be first; both must survive an
        // early failure. The hook fails the first attempt, independent of order.
        JuicerCuda::Failure failure;
        bool drained = false;
        try {
            drained = JuicerCuda::drain_deferred_resources(key, failure);
        } catch (const std::exception& error) {
            std::printf("drain escaped: %s\n", error.what());
        }
        rejectAllocation = false;
        std::printf("after failure charged=%llu records=%llu attempts=%u\n",
                    static_cast<unsigned long long>(ledger->snapshot().chargedBytes),
                    static_cast<unsigned long long>(ledger->snapshot().recordCount),
                    drainAttempts);
        if (!control) {
            require(!drained, "faulted drain succeeded");
            require(ledger->snapshot().chargedBytes == 12288 && ledger->snapshot().recordCount == 3, "failure discarded queued entries");
            const auto expected = mismatchContext ? FJ_STATUS_PREPARATION_FAILURE
                                                  : (failAllocation ? FJ_STATUS_ALLOCATION_FAILURE : FJ_STATUS_INTERNAL_FAILURE);
            require(failure.status.category == expected && failure.status.api == FJ_API_NONE && failure.status.native_code == 0,
                    "drain exception lost its typed origin");
            require(drainAttempts == 1, "failed drain retried or processed later entries");
        } else {
            require(drained, "unfaulted control drain failed");
        }
        failDrain = false;
        mismatchContext = nullptr;
        require(cuCtxSetCurrent(contexts[0]) == CUDA_SUCCESS, "restore owner after fault");
        require(JuicerCuda::drain_deferred_resources(key, failure), "explicit drain after clearing fault");
        std::printf("after retry charged=%llu records=%llu attempts=%u\n",
                    static_cast<unsigned long long>(ledger->snapshot().chargedBytes),
                    static_cast<unsigned long long>(ledger->snapshot().recordCount),
                    drainAttempts);
        require(ledger->snapshot().chargedBytes == 4096 && ledger->snapshot().recordCount == 1, "failed or unprocessed owner lost, duplicated or unrelated owner freed");
        require(ledger->record_count_for_context(unrelated, 1) == 1, "unrelated exact-context entry changed");
        require(JuicerCuda::drain_deferred_resources(key, failure), "empty drain failed");
        require(cuCtxSetCurrent(contexts[1]) == CUDA_SUCCESS, "restore unrelated owner");
        require(JuicerCuda::drain_deferred_resources(unrelated, failure), "unrelated deferred drain");
        require(ledger->snapshot().chargedBytes == 0 && ledger->snapshot().recordCount == 0, "deferred ownership did not drain to zero");
        require(drainAttempts == (control ? 3u : 4u), "deferred entries duplicated or reprocessed");
        for (auto* context : contexts) {
            require(cuCtxDestroy(context) == CUDA_SUCCESS, "destroy fixture-owned context");
        }
    }
} // namespace

namespace JuicerCuda::DeferredDestroyTest {
    void before_drain(Resources&) {
        ++drainAttempts;
        if (failDrain) {
            if (mismatchContext) {
                require(cuCtxSetCurrent(mismatchContext) == CUDA_SUCCESS, "inject exact-context mismatch");
                return;
            }
            if (failAllocation) {
                throw std::bad_alloc{};
            }
            throw std::system_error{servingLockFailure};
        }
    }
} // namespace JuicerCuda::DeferredDestroyTest

// Reject allocation while extracting/draining/restoring the deferred batch.
// CUDA allocations and fixture construction precede this bounded interval.
void* operator new(std::size_t size) {
    if (rejectAllocation) {
        throw std::bad_alloc{};
    }
    if (void* value = std::malloc(size ? size : 1)) {
        return value;
    }
    throw std::bad_alloc{};
}
void operator delete(void* block) noexcept {
    std::free(block);
}
void* operator new[](std::size_t size) {
    return ::operator new(size);
}
void operator delete[](void* block) noexcept {
    ::operator delete(block);
}
#if defined(__cpp_sized_deallocation)
void operator delete(void* block, std::size_t) noexcept {
    ::operator delete(block);
}
void operator delete[](void* block, std::size_t) noexcept {
    ::operator delete(block);
}
#endif

int main(int argc, char** argv) {
    try {
        check(argc > 1 ? argv[1] : "lock");
        std::puts("PASS deferred resource ownership");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL deferred resource ownership: %s\n", error.what());
        return 1;
    }
}
