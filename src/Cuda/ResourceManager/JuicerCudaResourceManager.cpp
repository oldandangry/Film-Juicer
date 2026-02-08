// Cuda/ResourceManager/JuicerCudaResourceManager.cpp

#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"

#include "Cuda/ResourceManager/JuicerCudaManagerRegistry.h"
#include "Cuda/ResourceManager/JuicerCudaResourceState.h"
#include "Cuda/ResourceManager/JuicerCudaResourceTelemetry.h"

namespace JuicerCuda {
namespace ResourceManager {

bool begin_submission(
    SubmissionTransaction& outTransaction,
    const SubmissionSnapshot& snapshot,
    std::string& outError) {
    outError.clear();

    if (outTransaction.active) {
        outError = "submission transaction already active";
        return false;
    }

    ResourceManagerState& state = global_state();
    outTransaction.transactionId = state.nextTransactionId.fetch_add(1, std::memory_order_relaxed);
    if (outTransaction.transactionId == 0) {
        outTransaction.transactionId = state.nextTransactionId.fetch_add(1, std::memory_order_relaxed);
    }

    outTransaction.snapshot = snapshot;
    outTransaction.active = true;
    outTransaction.committed = false;

    (void)registry_get_or_create(snapshot.deviceContextKey);
    telemetry_record_begin_submission();
    return true;
}

bool acquire_plan(
    SubmissionTransaction& transaction,
    std::string& outError) {
    outError.clear();
    if (!transaction.active) {
        outError = "submission transaction is not active";
        return false;
    }
    telemetry_record_acquire_plan();
    return true;
}

bool commit_submission(
    SubmissionTransaction& transaction,
    void* cudaStreamOpaque,
    std::string& outError) {
    (void)cudaStreamOpaque;
    outError.clear();
    if (!transaction.active) {
        outError = "submission transaction is not active";
        return false;
    }
    transaction.committed = true;
    transaction.active = false;
    telemetry_record_commit_submission();
    return true;
}

void rollback_submission(
    SubmissionTransaction& transaction,
    const char* reason) noexcept {
    (void)reason;
    if (!transaction.active) {
        return;
    }
    transaction.committed = false;
    transaction.active = false;
    telemetry_record_rollback_submission();
}

} // namespace ResourceManager
} // namespace JuicerCuda

