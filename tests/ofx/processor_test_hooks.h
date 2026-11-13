#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

#include <cuda_runtime.h>

#include "Cuda/JuicerCudaExecutor.h"

#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
namespace JuicerCuda::ExecutorTest {
    extern thread_local std::optional<WorkspaceHandoff> workspaceHandoff;
    enum class Event : std::uint8_t {
        Injected,
        Message,
        Classified,
        FrameAborted,
        RecoveryStarted,
        RecoveryEnded,
        FatalMapped
    };
    enum class Delivery : std::uint8_t {
        Success,
        Failure,
        Throw
    };
    struct FailureObservation {
        std::array<Event, 8> events{};
        std::size_t count = 0;
        const char* diagnostic = nullptr;
        FjStatus status{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        const char* stage = nullptr;
        Delivery delivery = Delivery::Success;
        JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult scanResult = JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult::Failed;
        bool overflow = false;
        bool failureStageMatches = false;
        bool failureStatusMatches = false;
        bool nativeRecoveryPending = false;
        bool adapterRecoveryPending = false;
        bool messageMatches = false;
        bool armed = true;
        bool diffusionComponentsPanic = false;
        FjStatus classifiedStatus{};
        bool failureDiagnosticMatches = false;
        bool retirementObserved = false;
        bool retirementAccepted = false;
        FjStatus invalidationStatus{};
        bool invalidationCleared = false;
        bool completionObserved = false;
        FjRenderOutcome completionOutcome{};
        bool completionFatal = false;
        bool completionDiagnosticMatches = false;

        void record(Event event) noexcept;
    };
    extern thread_local FailureObservation* observation;

    struct ResourceFailures {
        std::optional<Failure> stbn;
        std::optional<Failure> wang;
        cudaError_t eventCreate = cudaSuccess;
        cudaError_t eventRecord = cudaSuccess;
        unsigned lutPosition = 0;
        unsigned lutAttempts = 0;
        int stbnAttempts = 0;
        int wangAttempts = 0;
        int injections = 0;

        void record_injection() noexcept;
    };
    extern thread_local ResourceFailures* resourceFailures;
    extern thread_local void (*recoveryStartHook)(void*) noexcept;
    extern thread_local void* recoveryStartHookUser;
} // namespace JuicerCuda::ExecutorTest
#endif
