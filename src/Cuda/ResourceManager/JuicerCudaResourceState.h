// Cuda/ResourceManager/JuicerCudaResourceState.h
//
// Phase-0 state scaffolding.
#pragma once

#include <atomic>
#include <cstdint>

namespace JuicerCuda {
namespace ResourceManager {

struct ResourceManagerState {
    std::atomic<std::uint64_t> nextTransactionId{ 1 };
    std::atomic<std::uint64_t> nextAcquireAttemptId{ 1 };
    std::atomic<std::uint64_t> beginSubmissionCalls{ 0 };
    std::atomic<std::uint64_t> acquirePlanCalls{ 0 };
    std::atomic<std::uint64_t> commitSubmissionCalls{ 0 };
    std::atomic<std::uint64_t> rollbackSubmissionCalls{ 0 };
    std::atomic<std::uint64_t> acquireStatusHit{ 0 };
    std::atomic<std::uint64_t> acquireStatusMiss{ 0 };
    std::atomic<std::uint64_t> acquireStatusBusy{ 0 };
    std::atomic<std::uint64_t> acquireStatusExhausted{ 0 };
    std::atomic<std::uint64_t> acquireStatusError{ 0 };
    std::atomic<std::uint64_t> traceSchemaMismatchEvents{ 0 };
    std::atomic<std::uint64_t> forbiddenInvalidationEdges{ 0 };
    std::atomic<std::uint64_t> moduleBoundaryViolations{ 0 };
};

ResourceManagerState& global_state() noexcept;

} // namespace ResourceManager
} // namespace JuicerCuda
