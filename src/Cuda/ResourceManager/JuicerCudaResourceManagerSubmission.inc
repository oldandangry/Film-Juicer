// Cuda/ResourceManager/JuicerCudaResourceManagerSubmission.inc
//
// Included by JuicerCudaResourceManager.cpp (single-TU split).
void finalize_submission_transaction(
    SubmissionTransaction& transaction,
    bool committed) noexcept {
    transaction.committed = committed;
    transaction.active = false;
}

bool ensure_submission_active(
    const SubmissionTransaction& transaction,
    std::string& outError) {
    if (transaction.active) {
        return true;
    }
    outError = "submission transaction is not active";
    return false;
}

bool finalize_submission_end_or_trace(
    const SubmissionTransaction& transaction,
    const char* stage,
    std::string* outError = nullptr) {
    if (registry_note_submission_end(transaction.snapshot.deviceContextKey)) {
        return true;
    }
    telemetry_record_module_boundary_violation();
    telemetry_trace_module_boundary_violation(
        TelemetryModuleBoundaryViolationTrace{
            .context = TelemetryTraceContext{
                .transactionId = transaction.transactionId,
                .snapshotId = transaction.snapshot.snapshotId},
            .reason = trace_or(stage, "registry_submission_end_rejected")});
    if (outError) {
        *outError = "registry submission-end tracking rejected";
    }
    return false;
}

bool validate_resolved_memory_budget(
    const ResolvedMemoryBudget& budget,
    std::string& outError) {
    outError.clear();
    if (budget.deviceId < 0) {
        outError = "submission memory budget device id is invalid";
        return false;
    }
    if (budget.deviceBudgetBytes <= 1) {
        outError = "submission memory budget must be greater than one byte";
        return false;
    }
    if (budget.allocationCapBytes == 0) {
        outError = "submission allocation cap must be non-zero";
        return false;
    }
    if (budget.allocationCapBytes > budget.deviceBudgetBytes) {
        outError = "submission allocation cap exceeds device budget";
        return false;
    }
    return true;
}

bool resolve_submission_memory_budget(
    const SubmissionSnapshot& snapshot,
    std::uint64_t deviceBudgetBytes,
    ResolvedMemoryBudget& outBudget,
    std::string& outError) {
    outBudget = ResolvedMemoryBudget{};
    outBudget.deviceId = snapshot.deviceContextKey.deviceId;
    outBudget.deviceBudgetBytes = deviceBudgetBytes;

    constexpr std::uint64_t kDefaultDeviceHeadroomBytes =
        256ull * 1024ull * 1024ull;
    const std::uint64_t maximumHeadroom =
        deviceBudgetBytes > 0 ? deviceBudgetBytes - 1 : 0;
    const std::uint64_t configuredHeadroom =
        std::min(kDefaultDeviceHeadroomBytes, maximumHeadroom);
    outBudget.allocationCapBytes = deviceBudgetBytes - configuredHeadroom;
    return validate_resolved_memory_budget(outBudget, outError);
}

bool begin_submission(
    SubmissionTransaction& outTransaction,
    const SubmissionSnapshot& snapshot,
    std::uint64_t deviceBudgetBytes,
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
    outTransaction.snapshot.keyDigests = normalize_key_digests(outTransaction.snapshot.keyDigests);

    if (!resolve_submission_memory_budget(
            outTransaction.snapshot,
            deviceBudgetBytes,
            outTransaction.resolvedMemoryBudget,
            outError)) {
        return false;
    }

    RegistryContextSnapshot contextSnapshot{};
    if (!registry_begin_submission(
            outTransaction.snapshot.deviceContextKey,
            contextSnapshot)) {
        outError = "registry submission admission failed";
        return false;
    }
    outTransaction.snapshot.contextEpoch = contextSnapshot.contextEpoch;

    std::uint64_t leaseGeneration = state.nextLeaseGeneration.fetch_add(1, std::memory_order_relaxed);
    if (leaseGeneration == 0) {
        leaseGeneration = state.nextLeaseGeneration.fetch_add(1, std::memory_order_relaxed);
    }
    outTransaction.leaseGeneration = leaseGeneration;
    outTransaction.active = true;
    outTransaction.committed = false;

    telemetry_record_begin_submission();
    return true;
}

bool commit_submission(
    SubmissionTransaction& transaction,
    void* cudaStreamOpaque,
    std::string& outError) {
    (void)cudaStreamOpaque;
    outError.clear();
    if (!ensure_submission_active(transaction, outError)) {
        return false;
    }
    if (!finalize_submission_end_or_trace(transaction, "registry_submission_end_rejected_commit", &outError)) {
        finalize_submission_transaction(transaction, false);
        return false;
    }
    finalize_submission_transaction(transaction, true);
    telemetry_record_commit_submission();
    return true;
}


void rollback_submission(
    SubmissionTransaction& transaction,
    const char* reason) noexcept {
    try {
        (void)reason;
        if (!transaction.active) {
            return;
        }
        (void)finalize_submission_end_or_trace(transaction, "registry_submission_end_rejected_rollback");
        finalize_submission_transaction(transaction, false);
        telemetry_record_rollback_submission();
    } catch (...) {
        JuicerLogging::discard_current_exception();
    }
}
