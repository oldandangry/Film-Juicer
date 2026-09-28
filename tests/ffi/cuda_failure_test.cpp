#include <algorithm>
#include <array>
#include <cstdint>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <gtest/gtest.h>

#include "Cuda/Diffusion/JuicerCudaDiffusion.h"
#include "Cuda/JuicerCudaDeviceLedger.h"
#include "Cuda/JuicerCudaExecutor.h"
#include "Cuda/JuicerCudaFailure.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"

namespace {
    using JuicerCuda::Failure;

    void expect_status(FjStatus actual, FjStatus expected) {
        EXPECT_EQ(actual.category, expected.category);
        EXPECT_EQ(actual.api, expected.api);
        EXPECT_EQ(actual.native_code, expected.native_code);
    }

    TEST(CudaFailure, RuntimeCodesSelectRecoveryWithoutText) {
        for (const auto code : {cudaErrorContextIsDestroyed, cudaErrorDeviceUninitialized, cudaErrorCudartUnloading}) {
            const Failure failure{JuicerCuda::runtime_failure_status(code), "renamed"};
            expect_status(failure.status, {FJ_STATUS_CONTEXT_LOSS, FJ_API_CUDA_RUNTIME, code});
            EXPECT_TRUE(JuicerCuda::context_loss(failure));
        }
        for (const auto code : {cudaErrorInvalidValue, cudaErrorNotReady, cudaErrorIllegalAddress, cudaErrorDevicesUnavailable, cudaErrorUnknown}) {
            const Failure failure{JuicerCuda::runtime_failure_status(code), "context is destroyed device lost driver shutting down"};
            expect_status(failure.status, {FJ_STATUS_CUDA_FAILURE, FJ_API_CUDA_RUNTIME, code});
            EXPECT_FALSE(JuicerCuda::context_loss(failure));
        }
    }

    TEST(CudaFailure, DriverCodesRetainApiAndRecovery) {
        for (const auto code : {CUDA_ERROR_CONTEXT_IS_DESTROYED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_DEINITIALIZED}) {
            const Failure failure{JuicerCuda::driver_failure_status(code), "renamed"};
            expect_status(failure.status, {FJ_STATUS_CONTEXT_LOSS, FJ_API_CUDA_DRIVER, code});
            EXPECT_TRUE(JuicerCuda::context_loss(failure));
        }
        for (const auto code : {CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEVICE_UNAVAILABLE, CUDA_ERROR_UNKNOWN}) {
            const Failure failure{JuicerCuda::driver_failure_status(code), "device lost"};
            expect_status(failure.status, {FJ_STATUS_CUDA_FAILURE, FJ_API_CUDA_DRIVER, code});
            EXPECT_FALSE(JuicerCuda::context_loss(failure));
        }
    }

    TEST(CudaFailure, MissingNativeCodeRemainsPreparation) {
        Failure failure;
        failure.diagnostic = "cudaErrorContextIsDestroyed out of memory";
        expect_status(failure.status, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0});
        EXPECT_FALSE(JuicerCuda::context_loss(failure));
        EXPECT_FALSE(JuicerCuda::allocation_capacity_exhausted(failure));
        expect_status(JuicerCuda::runtime_failure_status(0), failure.status);
        expect_status(JuicerCuda::driver_failure_status(0), failure.status);
        expect_status(JuicerCuda::Diffusion::launch_failure_status({JuicerCuda::Diffusion::FailureApi::Validation, -1, "binding"}), failure.status);
    }

    TEST(CudaFailure, CodeLessAdmissionFailureReplacesEarlierStatus) {
        JuicerCuda::ResourceManager::SubmissionTransaction transaction;
        JuicerCuda::ResourceManager::SubmissionSnapshot snapshot;
        Failure failure{JuicerCuda::runtime_failure_status(cudaErrorContextIsDestroyed), "device lost"};
        EXPECT_FALSE(JuicerCuda::ResourceManager::begin_submission(transaction, snapshot, 0, failure));
        expect_status(failure.status, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0});
        EXPECT_FALSE(failure.diagnostic.empty());
    }

    TEST(CudaFailure, DiffusionLaunchRetainsRuntimeAndCufftCodes) {
        expect_status(JuicerCuda::Diffusion::launch_failure_status({JuicerCuda::Diffusion::FailureApi::Cuda, cudaErrorContextIsDestroyed, "launch"}),
                      {FJ_STATUS_CONTEXT_LOSS, FJ_API_CUDA_RUNTIME, cudaErrorContextIsDestroyed});
        expect_status(JuicerCuda::Diffusion::launch_failure_status({JuicerCuda::Diffusion::FailureApi::Cufft, CUFFT_ALLOC_FAILED, "plan"}),
                      {FJ_STATUS_CUFFT_FAILURE, FJ_API_CUFFT, CUFFT_ALLOC_FAILED});
    }

    TEST(CudaFailure, ExceptionAndTruncationPreserveStatus) {
        const Failure original{JuicerCuda::driver_failure_status(-901), "diagnostic that exceeds the buffer"};
        try {
            throw JuicerCuda::ExecutionFailure{original};
        } catch (const JuicerCuda::ExecutionFailure& exception) {
            for (const std::size_t capacity : {0u, 1u, 5u, 64u}) {
                std::array<char, 64> bytes{};
                FjErrorBuffer buffer{bytes.data(), capacity, 99};
                expect_status(JuicerCuda::write_status(exception.failure.status, exception.failure.diagnostic, &buffer), original.status);
                EXPECT_EQ(exception.failure.diagnostic, original.diagnostic);
                EXPECT_EQ(buffer.length, capacity ? std::min(capacity - 1, original.diagnostic.size()) : 0);
                if (capacity) {
                    EXPECT_EQ(bytes[buffer.length], '\0');
                    EXPECT_EQ(std::string(bytes.data(), buffer.length), original.diagnostic.substr(0, buffer.length));
                }
            }
            expect_status(JuicerCuda::write_status(exception.failure.status, exception.failure.diagnostic, nullptr), original.status);
        }
    }

    TEST(CudaFailure, LedgerCapacityOriginIsDistinctFromValidation) {
        std::string diagnostic;
        auto ledger = JuicerCuda::DeviceAllocationLedger::create({0, 1024}, diagnostic);
        ASSERT_NE(ledger, nullptr) << diagnostic;
        ASSERT_TRUE(ledger->bind_or_validate_cap(512, diagnostic)) << diagnostic;
        int contextToken = 0;
        JuicerCuda::DeviceByteReservation reservation;
        Failure failure;
        ASSERT_FALSE(ledger->reserve({{0, &contextToken}, 1, 512}, reservation, failure));
        expect_status(failure.status, {FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0});
        EXPECT_EQ(failure.diagnostic, "device_cap_exceeded");
        failure.diagnostic = "different wording";
        EXPECT_TRUE(JuicerCuda::allocation_capacity_exhausted(failure));
        ASSERT_FALSE(ledger->reserve({{0, &contextToken}, 1, 0}, reservation, failure));
        expect_status(failure.status, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0});
        failure.diagnostic = "device_cap_exceeded memory allocation out of memory";
        EXPECT_FALSE(JuicerCuda::allocation_capacity_exhausted(failure));
    }

    struct AllocationScenario {
        Failure first;
        Failure retry;
        Failure reclaimFailure;
        std::size_t reclaimedBytes = 64;
        int attempts = 0;
        int reclaims = 0;
        bool retrySucceeds = true;
        bool reclaimSucceeds = true;
    };
    AllocationScenario scenario;

    bool allocate(Failure& failure) {
        ++scenario.attempts;
        if (scenario.attempts == 1) {
            failure = scenario.first;
            return false;
        }
        failure = scenario.retry;
        return scenario.retrySucceeds;
    }

    bool reclaim(std::size_t& bytes, Failure& failure) {
        ++scenario.reclaims;
        bytes = scenario.reclaimedBytes;
        failure = scenario.reclaimFailure;
        return scenario.reclaimSucceeds;
    }

    TEST(AllocationRetry, CapacityRetriesOnceRegardlessOfDiagnostic) {
        for (const auto* wording : {"out of memory", "device_cap_exceeded", "unrelated wording", ""}) {
            scenario = {};
            scenario.first = {JuicerCuda::runtime_failure_status(cudaErrorMemoryAllocation), wording};
            Failure failure;
            EXPECT_TRUE(JuicerCuda::ResourceManager::test_allocation_retry(allocate, reclaim, failure));
            EXPECT_EQ(scenario.attempts, 2);
            EXPECT_EQ(scenario.reclaims, 1);
            EXPECT_TRUE(failure.diagnostic.empty());
        }
    }

    TEST(AllocationRetry, IdenticalTextCannotEnableRetryForOtherCategories) {
        for (const auto status : {FjStatus{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0},
                                  JuicerCuda::runtime_failure_status(cudaErrorInvalidValue),
                                  JuicerCuda::driver_failure_status(CUDA_ERROR_CONTEXT_IS_DESTROYED),
                                  JuicerCuda::cufft_failure_status(CUFFT_ALLOC_FAILED)}) {
            scenario = {};
            scenario.first = {status, "out of memory memory allocation device_cap_exceeded"};
            Failure failure;
            EXPECT_FALSE(JuicerCuda::ResourceManager::test_allocation_retry(allocate, reclaim, failure));
            EXPECT_EQ(scenario.attempts, 1);
            EXPECT_EQ(scenario.reclaims, 0);
            expect_status(failure.status, status);
            EXPECT_EQ(failure.diagnostic, scenario.first.diagnostic);
        }
    }

    TEST(AllocationRetry, NoReclaimedBytesStopsAfterFirstAttempt) {
        scenario = {};
        scenario.first = {{FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0}, "initial capacity diagnostic"};
        scenario.reclaimedBytes = 0;
        Failure failure;
        EXPECT_FALSE(JuicerCuda::ResourceManager::test_allocation_retry(allocate, reclaim, failure));
        EXPECT_EQ(scenario.attempts, 1);
        EXPECT_EQ(scenario.reclaims, 1);
        expect_status(failure.status, scenario.first.status);
        EXPECT_EQ(failure.diagnostic, scenario.first.diagnostic);
    }

    TEST(AllocationRetry, ReclaimFailurePreservesOriginalDiagnosticAndTerminalCode) {
        scenario = {};
        scenario.first = {JuicerCuda::runtime_failure_status(cudaErrorMemoryAllocation), "first"};
        scenario.reclaimSucceeds = false;
        scenario.reclaimFailure = {JuicerCuda::driver_failure_status(CUDA_ERROR_CONTEXT_IS_DESTROYED), "reclaim"};
        Failure failure;
        EXPECT_FALSE(JuicerCuda::ResourceManager::test_allocation_retry(allocate, reclaim, failure));
        EXPECT_EQ(scenario.attempts, 1);
        EXPECT_EQ(scenario.reclaims, 1);
        expect_status(failure.status, scenario.reclaimFailure.status);
        EXPECT_EQ(failure.diagnostic, "first | reclaim_failed: reclaim");
        EXPECT_TRUE(JuicerCuda::context_loss(failure));
    }

    TEST(AllocationRetry, FailedRetryPreservesOriginalDiagnosticAndNeverRetriesAgain) {
        scenario = {};
        scenario.first = {JuicerCuda::runtime_failure_status(cudaErrorMemoryAllocation), "first"};
        scenario.retrySucceeds = false;
        scenario.retry = {JuicerCuda::runtime_failure_status(cudaErrorContextIsDestroyed), "second"};
        Failure failure;
        EXPECT_FALSE(JuicerCuda::ResourceManager::test_allocation_retry(allocate, reclaim, failure));
        EXPECT_EQ(scenario.attempts, 2);
        EXPECT_EQ(scenario.reclaims, 1);
        expect_status(failure.status, scenario.retry.status);
        EXPECT_EQ(failure.diagnostic, "first | allocation_retry_failed: second");
        EXPECT_TRUE(JuicerCuda::context_loss(failure));
    }
} // namespace
