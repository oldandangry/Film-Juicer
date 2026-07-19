// Cuda/ResourceManager/JuicerCudaResourceManagerCommands.inc
//
// Included by JuicerCudaResourceManager.cpp (single-TU split).
bool ensure_active_for_command(
    const SubmissionTransaction& transaction,
    std::string& outError,
    const char* commandName) {
    const char* stage = trace_or(commandName, "command");
    const char* rejectionStage = trace_or(commandName, "command_requires_active_submission");
    if (!validate_lifecycle_for_stage(transaction, stage, false, &outError)) {
        telemetry_record_module_boundary_violation();
        telemetry_trace_module_boundary_violation(
            TelemetryModuleBoundaryViolationTrace{
                .context = TelemetryTraceContext{
                    .transactionId = transaction.transactionId,
                    .snapshotId = transaction.snapshot.snapshotId,
                    .traceSchemaVersion = transaction.snapshot.traceSchemaVersion},
                .reason = rejectionStage});
        return false;
    }

    return validate_stale_tuple_for_stage(
        transaction,
        stage,
        LeaseObservationMode::ActiveOnly,
        "stale transaction in command path (reason=",
        &outError,
        true,
        rejectionStage);
}

const char* reclaim_retry_reason(std::size_t reclaimedBytes) noexcept {
    if (reclaimedBytes > 0) {
        return "retry_after_reap";
    }
    return "reap_no_progress";
}

const char* bool_reason(bool value, const char* whenTrue, const char* whenFalse) noexcept {
    if (value) {
        return whenTrue;
    }
    return whenFalse;
}

std::uint32_t bool_u32(bool value) noexcept {
    if (value) {
        return 1u;
    }
    return 0u;
}

bool admission_retry_error_is_effective_headroom_denial(const std::string& error) noexcept {
    return error.rfind(kScratchExhaustedPrefix, 0) == 0 &&
           error.find("reason=insufficient_effective_headroom") != std::string::npos;
}

bool admission_retry_reclaim_progressed(
    const JuicerCuda::AdmissionRetryOpticsReclaimStats& stats) noexcept {
    return stats.opticsRetiredBytes > 0 ||
           stats.sharedTmpRetiredBytes > 0 ||
           stats.reclaimedBytes > 0;
}

bool admission_retry_optics_reclaim_is_allowed(
    ScratchWorkClass workClass,
    const ScratchRequestDescriptor& scratchRequest,
    const std::string& admissionError) noexcept {
    return workClass == ScratchWorkClass::SpatialDir &&
           scratchRequest.needSpatialDir &&
           scratchRequest.needOptics &&
           admission_retry_error_is_effective_headroom_denial(admissionError);
}

void trace_admission_retry_optics_reclaim(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const JuicerCuda::AdmissionRetryOpticsReclaimStats& stats,
    bool ok,
    bool retryAllowed,
    const char* reason,
    const std::string& error) {
#if JUICER_DIAGNOSTICS_COMPILED
    if ((ok && !retryAllowed) ? !JTRACE_ENABLED(2) : !JTRACE_ENABLED(1)) {
        return;
    }
    std::string msg = trace_event_prefix(
        "admission_retry_optics_reclaim",
        transaction,
        commandName);
    msg += " ok=";
    msg += ok ? "1" : "0";
    msg += " retry_allowed=";
    msg += retryAllowed ? "1" : "0";
    msg += " pending_scratch_bytes_before=";
    msg += std::to_string(static_cast<unsigned long long>(stats.pendingScratchBytesBefore));
    msg += " optics_retired_bytes=";
    msg += std::to_string(static_cast<unsigned long long>(stats.opticsRetiredBytes));
    msg += " shared_tmp_retired_bytes=";
    msg += std::to_string(static_cast<unsigned long long>(stats.sharedTmpRetiredBytes));
    msg += " reclaimed_bytes=";
    msg += std::to_string(static_cast<unsigned long long>(stats.reclaimedBytes));
    if (!error.empty()) {
        msg += " error=";
        msg += error;
    }
    msg += " reason=";
    msg += trace_or_unspecified(reason);
    JTRACE("MSADM", msg);
#else
    (void)transaction;
    (void)commandName;
    (void)stats;
    (void)ok;
    (void)retryAllowed;
    (void)reason;
    (void)error;
#endif
}

bool try_reclaim_retained_optics_for_admission_retry(
    const SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const char* commandName,
    ScratchWorkClass workClass,
    const ScratchRequestDescriptor& scratchRequest,
    const std::string& admissionError,
    void* cudaStreamOpaque,
    bool captureMemorySnapshots,
    std::string& outError) {
    outError = admissionError;
    if (!admission_retry_optics_reclaim_is_allowed(
            workClass,
            scratchRequest,
            admissionError)) {
        return false;
    }

    JuicerCuda::AdmissionRetryOpticsReclaimStats stats{};
    std::string reclaimError;
    const bool ok = JuicerCuda::reclaim_retained_optics_for_admission_retry(
        resources,
        transaction.leaseGeneration,
        scratchRequest,
        cudaStreamOpaque,
        stats,
        reclaimError);
    const bool progressed = ok && admission_retry_reclaim_progressed(stats);
    trace_admission_retry_optics_reclaim(
        transaction,
        commandName,
        stats,
        ok,
        progressed,
        progressed ? "retry_after_optics_reclaim" : (ok ? "no_reclaim_progress" : "reclaim_failed"),
        reclaimError);
    if (!ok) {
        outError = admissionError;
        if (!reclaimError.empty()) {
            outError += " | admission_retry_optics_reclaim_failed: " + reclaimError;
        }
        return false;
    }
    if (!progressed) {
        return false;
    }
    if (captureMemorySnapshots) {
        publish_manager_memory_snapshot(snapshot_manager_memory(resources));
    }
    outError.clear();
    return true;
}

void registry_snapshot_context_keys(std::vector<DeviceContextKey>& outKeys);

const char* commands_error_or_cstr(const std::string& error, const char* fallback) noexcept {
    if (error.empty()) {
        return fallback;
    }
    return error.c_str();
}

std::string commands_error_or_message(const std::string& error, const char* fallback) {
    if (error.empty()) {
        return std::string(fallback);
    }
    return error;
}

std::string commands_scratch_exhausted_error(
    const char* commandName,
    const char* reason,
    const std::string& detail) {
    std::string message = std::string(kScratchExhaustedPrefix) +
                          " command=" + trace_or_unknown(commandName) +
                          " reason=" + trace_or_unspecified(reason);
    if (!detail.empty()) {
        message += " detail=" + detail;
    }
    return message;
}

bool resolve_optional_active_scratch_request(
    const ScratchRequestDescriptor& scratchRequest,
    const char* commandName,
    const ScratchRequestDescriptor*& outActiveScratchRequest,
    std::string& outError) {
    outActiveScratchRequest = nullptr;
    if (!scratchRequest.has_any_family()) {
        if (scratchRequest.needBlurred ||
            scratchRequest.aliasScannerRgbFromSpatialDirFiltered ||
            scratchRequest.needAux ||
            scratchRequest.needSharedTmp ||
            scratchRequest.needGrainFrameUniforms ||
            scratchRequest.needGrainLayerWork ||
            scratchRequest.needGrainShared ||
            scratchRequest.needGateMask) {
            outError = std::string(trace_or_non_empty(commandName, "scratch_request")) + ": invalid scratch request descriptor";
            return false;
        }
        return true;
    }

    if (!validate_scratch_request_descriptor_for_manager(
            scratchRequest,
            commandName,
            outError)) {
        return false;
    }

    outActiveScratchRequest = &scratchRequest;
    return true;
}

bool validate_scratch_request_for_family(
    const ScratchRequestDescriptor& scratchRequest,
    const char* commandName,
    ScratchWorkClass workClass,
    std::string& outError) {
    if (!validate_scratch_request_descriptor_for_manager(
            scratchRequest,
            commandName,
            outError)) {
        return false;
    }

    if (workClass == ScratchWorkClass::Optics && !scratchRequest.needOptics) {
        outError = std::string(trace_or_non_empty(commandName, "command")) + ": optics scratch request missing needOptics";
        return false;
    }
    if (workClass == ScratchWorkClass::SpatialDir && !scratchRequest.needSpatialDir) {
        outError = std::string(trace_or_non_empty(commandName, "command")) + ": spatial DIR scratch request missing needSpatialDir";
        return false;
    }
    return true;
}

void complete_tier_circuit_attempt(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const TierCircuitAttempt& attempt,
    bool success,
    const char* successReason,
    const std::string& failureError);

struct UploadWorkEstimate {
    std::uint64_t growthBytes = 0;
    std::uint64_t reservationBytes = 0;
};

template <typename Action>
bool execute_upload_immutable_command(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const char* commandName,
    std::uint64_t pressureRequestBytes,
    std::uint64_t reservationRequestBytes,
    bool criticalRequest,
    const ScratchRequestDescriptor* scratchRequest,
    const char* overflowMessage,
    Action&& action,
    std::string& outError) {
    if (pressureRequestBytes == std::numeric_limits<std::uint64_t>::max() ||
        reservationRequestBytes == std::numeric_limits<std::uint64_t>::max()) {
        outError = trace_or_non_empty(overflowMessage, "upload reservation request byte estimation overflow");
        return false;
    }

    const bool captureMemorySnapshots =
        should_collect_manager_memory_snapshots(transaction.resolvedPressurePolicy);
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);

    const char* stageName = trace_or_non_empty(commandName, "command");
    bool requestPreReclaim = false;
    if (!enforce_pressure_gate(
            transaction,
            resources,
            stageName,
            PressureLane::Upload,
            saturating_u64_to_size_t(pressureRequestBytes),
            criticalRequest,
            scratchRequest,
            requestPreReclaim,
            outError)) {
        return false;
    }
    if (requestPreReclaim) {
        std::string reclaimError;
        if (!run_reap_pass_for_pressure(
                transaction,
                resources,
                stageName,
                "pressure_pre_upload",
                reclaimError)) {
            outError =
                commands_error_or_message(reclaimError, "pressure pre-upload reclaim failed");
            return false;
        }
    }

    UploadReservationClaim uploadClaim{};
    if (!acquire_upload_reservation_with_wait(
            transaction,
            stageName,
            reservationRequestBytes,
            criticalRequest,
            uploadClaim,
            outError)) {
        return false;
    }
    UploadReservationGuard uploadGuard(uploadClaim);
    TierCircuitAttempt circuitAttempt{};
    std::string circuitError;
    if (!tier_circuit_begin_attempt(
            transaction,
            stageName,
            ResourceTier::Immutable,
            false,
            circuitAttempt,
            circuitError)) {
        outError = circuitError;
        return false;
    }

    const bool ok = action(outError);
    complete_tier_circuit_attempt(
        transaction,
        stageName,
        circuitAttempt,
        ok,
        "ensure_success",
        outError);
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    return ok;
}

template <typename Action>
bool execute_snapshot_wrapped_command(
    const SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    Action&& action,
    std::string& outError) {
    const bool captureMemorySnapshots =
        should_collect_manager_memory_snapshots(transaction.resolvedPressurePolicy);
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    const bool ok = action(outError);
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    return ok;
}

template <typename Action, typename PolicyFailureFallback>
bool execute_upload_lut_command(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const char* commandName,
    std::uint64_t pressureRequestBytes,
    std::uint64_t reservationRequestBytes,
    bool criticalRequest,
    const ScratchRequestDescriptor* scratchRequest,
    const char* overflowMessage,
    bool captureMemorySnapshots,
    bool publishInitialSnapshot,
    Action&& action,
    PolicyFailureFallback&& policyFailureFallback,
    std::string& outError) {
    if (pressureRequestBytes == std::numeric_limits<std::uint64_t>::max() ||
        reservationRequestBytes == std::numeric_limits<std::uint64_t>::max()) {
        outError = trace_or_non_empty(overflowMessage, "upload reservation request byte estimation overflow");
        return false;
    }

    if (publishInitialSnapshot) {
        maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    }

    const char* stageName = trace_or_non_empty(commandName, "command");
    auto try_policy_fallback = [&](const char* triggerReason) -> bool {
        if (!policyFailureFallback(triggerReason)) {
            return false;
        }
        outError.clear();
        return true;
    };

    bool requestPreReclaim = false;
    if (!enforce_pressure_gate(
            transaction,
            resources,
            stageName,
            PressureLane::Upload,
            saturating_u64_to_size_t(pressureRequestBytes),
            criticalRequest,
            scratchRequest,
            requestPreReclaim,
            outError)) {
        return try_policy_fallback("pressure_gate_reject");
    }
    if (requestPreReclaim) {
        std::string reclaimError;
        if (!run_reap_pass_for_pressure(
                transaction,
                resources,
                stageName,
                "pressure_pre_upload",
                reclaimError)) {
            outError =
                commands_error_or_message(reclaimError, "pressure pre-upload reclaim failed");
            return try_policy_fallback("pressure_pre_upload_reclaim_failed");
        }
    }

    UploadReservationClaim uploadClaim{};
    if (!acquire_upload_reservation_with_wait(
            transaction,
            stageName,
            reservationRequestBytes,
            criticalRequest,
            uploadClaim,
            outError)) {
        return try_policy_fallback("upload_reservation_reject");
    }
    UploadReservationGuard uploadGuard(uploadClaim);

    BuilderReservationClaim builderClaim{};
    if (!acquire_builder_reservation_with_wait(
            transaction,
            stageName,
            BuilderReservationTier::Lut,
            reservationRequestBytes,
            criticalRequest,
            builderClaim,
            outError)) {
        return try_policy_fallback("builder_reservation_reject");
    }
    BuilderReservationGuard builderGuard(builderClaim);

    TierCircuitAttempt circuitAttempt{};
    std::string circuitError;
    if (!tier_circuit_begin_attempt(
            transaction,
            stageName,
            ResourceTier::Lut,
            false,
            circuitAttempt,
            circuitError)) {
        outError = circuitError;
        return try_policy_fallback("tier_circuit_blocked");
    }

    const bool ok = action(outError);
    complete_tier_circuit_attempt(
        transaction,
        stageName,
        circuitAttempt,
        ok,
        "ensure_success",
        outError);
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    return ok;
}

template <typename Action>
bool execute_scratch_growth_command(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const char* commandName,
    std::size_t growthBytes,
    ScratchWorkClass workClass,
    const ScratchRequestDescriptor& scratchRequest,
    void* cudaStreamOpaque,
    bool captureMemorySnapshots,
    Action&& action,
    std::string& outError) {
    const ResourceManagerConfigEffective& cfg = manager_effective_config();

    bool requestPreReclaim = false;
    if (!enforce_pressure_gate(
            transaction,
            resources,
            commandName,
            PressureLane::Builder,
            growthBytes,
            true,
            &scratchRequest,
            requestPreReclaim,
            outError)) {
        return false;
    }
    if (requestPreReclaim) {
        std::string reclaimError;
        if (!run_reap_pass_for_pressure(
                transaction,
                resources,
                commandName,
                "pressure_pre_growth",
                reclaimError)) {
            outError = commands_error_or_message(reclaimError, "pressure pre-growth reclaim failed");
            return false;
        }
    }
    if (!admit_active_frame_vram_growth(
            transaction,
            resources,
            commandName,
            growthBytes,
            &scratchRequest,
            true,
            ActiveFrameAdmissionHeadroom::DriverFreeOnly,
            outError)) {
        const std::string admissionError = outError;
        if (!try_reclaim_retained_optics_for_admission_retry(
                transaction,
                resources,
                commandName,
                workClass,
                scratchRequest,
                admissionError,
                cudaStreamOpaque,
                captureMemorySnapshots,
                outError)) {
            return false;
        }
        if (!admit_active_frame_vram_growth(
                transaction,
                resources,
                commandName,
                growthBytes,
                &scratchRequest,
                true,
                ActiveFrameAdmissionHeadroom::DriverFreeOnly,
                outError)) {
            return false;
        }
    }

    BuilderReservationClaim builderClaim{};
    if (!acquire_builder_reservation_with_wait(
            transaction,
            commandName,
            BuilderReservationTier::Scratch,
            static_cast<std::uint64_t>(growthBytes),
            true,
            builderClaim,
            outError)) {
        return false;
    }
    BuilderReservationGuard builderGuard(builderClaim);

    ScratchPolicyClaim scratchClaim{};
    const ScratchBucketKey scratchBucketKey = make_scratch_bucket_key(
        scratchRequest.requestedWidth,
        scratchRequest.requestedHeight,
        workClass);
    if (!acquire_scratch_policy_claim_with_wait(
            transaction,
            commandName,
            scratchBucketKey,
            growthBytes,
            true,
            scratchClaim,
            outError)) {
        return false;
    }
    ScratchPolicyGuard scratchGuard(scratchClaim);

    ResourceManagerState& managerState = global_state();
    std::uint32_t attempts = 0;
    bool fragmentationRecoveryTriggered = false;
    bool fragmentationRecoveryPendingOutcome = false;
    auto finalizeFragmentationOutcome = [&](bool success, const char* reason) {
        if (!fragmentationRecoveryPendingOutcome) {
            return;
        }
        if (success) {
            telemetry_counter_add(managerState.fragmentationRecoverySuccess, 1);
        } else {
            telemetry_counter_add(managerState.fragmentationRecoveryFailures, 1);
        }
        trace_fragmentation_recovery(
            transaction,
            FragmentationRecoveryTrace{
                .commandName = commandName,
                .stage = "outcome",
                .reason = reason,
                .requestBytes = growthBytes,
                .attempt = attempts,
                .success = success});
        fragmentationRecoveryPendingOutcome = false;
    };

    while (true) {
        outError.clear();
        TierCircuitAttempt circuitAttempt{};
        std::string circuitError;
        if (!tier_circuit_begin_attempt(
                transaction,
                commandName,
                ResourceTier::Scratch,
                false,
                circuitAttempt,
                circuitError)) {
            outError = circuitError;
            finalizeFragmentationOutcome(false, "tier_circuit_blocked");
            return false;
        }
        maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
        if (action(outError)) {
            complete_tier_circuit_attempt(
                transaction,
                commandName,
                circuitAttempt,
                true,
                "ensure_success",
                outError);
            maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
            if (attempts > 0) {
                telemetry_counter_add(managerState.budgetReclaimRetrySuccess, 1);
            }
            finalizeFragmentationOutcome(true, "allocation_retry_success");
            return true;
        }
        complete_tier_circuit_attempt(
            transaction,
            commandName,
            circuitAttempt,
            false,
            "ensure_success",
            outError);
        maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);

        const bool allocatorOom = is_allocator_oom_error(outError);
        if (!allocatorOom) {
            finalizeFragmentationOutcome(false, "non_allocator_error");
            return false;
        }
        if (attempts >= cfg.reclaimRetryMaxAttempts) {
            if (cfg.fragmentationRecoveryEnabled && !fragmentationRecoveryTriggered) {
                std::string recoveryError;
                if (!run_fragmentation_recovery_once(
                        transaction,
                        resources,
                        commandName,
                        growthBytes,
                        attempts,
                        captureMemorySnapshots,
                        recoveryError)) {
                    if (!recoveryError.empty()) {
                        outError += " | fragmentation_recovery_failed: " + recoveryError;
                    }
                    record_allocator_oom_headroom_observation(
                        transaction,
                        commandName,
                        growthBytes);
                    telemetry_counter_add(managerState.budgetAllocatorOomEvents, 1);
                    return false;
                }
                fragmentationRecoveryTriggered = true;
                fragmentationRecoveryPendingOutcome = true;
                continue;
            }
            record_allocator_oom_headroom_observation(
                transaction,
                commandName,
                growthBytes);
            telemetry_counter_add(managerState.budgetAllocatorOomEvents, 1);
            finalizeFragmentationOutcome(false, "allocator_oom_final");
            return false;
        }

        ++attempts;
        telemetry_counter_add(managerState.budgetReclaimRetryAttempts, 1);
        std::size_t reclaimedBytes = 0;
        std::string reclaimError;
        if (!JuicerCuda::reap_retired_allocations(resources, reclaimedBytes, reclaimError)) {
            trace_reap_pass(
                transaction,
                commandName,
                reclaimedBytes,
                false,
                commands_error_or_cstr(reclaimError, "reap_failed"));
            trace_budget_reclaim_retry(
                transaction,
                BudgetReclaimRetryTrace{
                    .commandName = commandName,
                    .reason = commands_error_or_cstr(reclaimError, "reap_failed"),
                    .reclaimedBytes = reclaimedBytes,
                    .attempt = attempts});
            if (!reclaimError.empty()) {
                outError += " | reclaim_retry_failed: " + reclaimError;
            }
            telemetry_counter_add(managerState.budgetAllocatorOomEvents, 1);
            finalizeFragmentationOutcome(false, "reap_retry_failed");
            return false;
        }

        if (reclaimedBytes > 0) {
            telemetry_counter_add(managerState.retireReapPasses, 1);
            telemetry_counter_add(managerState.retireReapBytes, reclaimedBytes);
        }
        trace_reap_pass(
            transaction,
            commandName,
            reclaimedBytes,
            true,
            reclaim_retry_reason(reclaimedBytes));
        trace_budget_reclaim_retry(
            transaction,
            BudgetReclaimRetryTrace{
                .commandName = commandName,
                .reason = reclaim_retry_reason(reclaimedBytes),
                .reclaimedBytes = reclaimedBytes,
                .attempt = attempts,
                .success = true});
        maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);

        if (reclaimedBytes == 0) {
            if (cfg.fragmentationRecoveryEnabled && !fragmentationRecoveryTriggered) {
                std::string recoveryError;
                if (!run_fragmentation_recovery_once(
                        transaction,
                        resources,
                        commandName,
                        growthBytes,
                        attempts,
                        captureMemorySnapshots,
                        recoveryError)) {
                    if (!recoveryError.empty()) {
                        outError += " | fragmentation_recovery_failed: " + recoveryError;
                    }
                    record_allocator_oom_headroom_observation(
                        transaction,
                        commandName,
                        growthBytes);
                    telemetry_counter_add(managerState.budgetAllocatorOomEvents, 1);
                    return false;
                }
                fragmentationRecoveryTriggered = true;
                fragmentationRecoveryPendingOutcome = true;
                continue;
            }
            record_allocator_oom_headroom_observation(
                transaction,
                commandName,
                growthBytes);
            telemetry_counter_add(managerState.budgetAllocatorOomEvents, 1);
            finalizeFragmentationOutcome(false, "reap_no_progress");
            return false;
        }
    }
}

void complete_tier_circuit_attempt(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const TierCircuitAttempt& attempt,
    bool success,
    const char* successReason,
    const std::string& failureError) {
    const char* stageName = trace_or_non_empty(commandName, "command");
    if (success) {
        tier_circuit_record_outcome(
            transaction,
            stageName,
            attempt,
            true,
            trace_or(successReason, "ensure_success"));
        return;
    }
    if (tier_circuit_should_count_failure(failureError)) {
        tier_circuit_record_outcome(
            transaction,
            stageName,
            attempt,
            false,
            failureError.c_str());
        return;
    }
    tier_circuit_cancel_attempt(
        transaction,
        stageName,
        attempt,
        "ignored_policy_failure");
}

struct UploadEstimateBytes {
    std::uint64_t value = 0;
};

struct UploadEstimateAccumulator {
    std::uint64_t* total = nullptr;
    bool* overflow = nullptr;
};

void add_estimate_bytes_u64(UploadEstimateBytes bytes, UploadEstimateAccumulator accumulator) noexcept {
    if (!accumulator.total || !accumulator.overflow) {
        return;
    }
    std::uint64_t& total = *accumulator.total;
    bool& overflow = *accumulator.overflow;
    if (overflow) {
        return;
    }
    std::uint64_t next = 0;
    if (!add_u64_checked(total, bytes.value, next)) {
        total = std::numeric_limits<std::uint64_t>::max();
        overflow = true;
        return;
    }
    total = next;
}

void add_count_upload_estimate_bytes(
    std::uint64_t count,
    std::size_t elementSize,
    std::uint64_t& total,
    bool& overflow) noexcept {
    if (overflow || count == 0 || elementSize == 0) {
        return;
    }
    std::uint64_t bytes = 0;
    if (!mul_u64_checked(count, static_cast<std::uint64_t>(elementSize), bytes)) {
        total = std::numeric_limits<std::uint64_t>::max();
        overflow = true;
        return;
    }
    add_estimate_bytes_u64(UploadEstimateBytes{bytes}, UploadEstimateAccumulator{&total, &overflow});
}

std::uint64_t commands_preferred_upload_core_hash(const WorkingState& ws) noexcept {
    if (ws.uploadCoreHash != 0) {
        return ws.uploadCoreHash;
    }
    return ws.coreHash;
}

float commands_sanitize_filter_shift_step(float value) noexcept {
    if (std::isfinite(value)) {
        return value;
    }
    return 0.0f;
}

std::uint64_t commands_neutral_filter_hash_or_default(const Print::Runtime& prt) noexcept {
    if (prt.neutralFilterHash != 0) {
        return prt.neutralFilterHash;
    }
    return Print::kNeutralCalibrationHashSeed;
}

const Scanner::ScannerStaticKey& commands_select_scanner_static_key(
    const WorkingState& ws,
    bool negativeMedium) noexcept {
    if (negativeMedium) {
        return ws.negativeStaticKey;
    }
    return ws.printStaticKey;
}

const Scanner::ScannerMediumRuntime& commands_select_scanner_medium_runtime(
    const WorkingState& ws,
    bool negativeMedium) noexcept {
    if (negativeMedium) {
        return ws.negativeMediumRuntime;
    }
    return ws.printMediumRuntime;
}

const JuicerCuda::Resources::DeviceSpectralLut& commands_select_scan_lut_slot(
    JuicerCuda::Resources& resources,
    bool negativeMedium) noexcept {
    if (negativeMedium) {
        return resources.scanNegativeLut;
    }
    return resources.scanPrintLut;
}

std::uint64_t commands_elapsed_ms_since(std::uint64_t nowMs, std::uint64_t earlierMs) noexcept {
    if (nowMs > earlierMs) {
        return nowMs - earlierMs;
    }
    return 0;
}

UploadWorkEstimate commands_uncached_upload_work(
    bool cached,
    UploadWorkEstimate uncachedWork) noexcept {
    if (!cached) {
        return uncachedWork;
    }
    return {};
}

bool commands_allow_durable_admission(
    const CacheAdmissionDecision& decision,
    bool churnProbationAllowDurable) noexcept {
    if (decision.probationApplied) {
        return churnProbationAllowDurable;
    }
    return decision.allowDurableAdmission;
}

template <typename T>
void add_vector_upload_estimate_bytes(
    const std::vector<T>& values,
    std::uint64_t& total,
    bool& overflow) noexcept {
    if (values.empty() || overflow) {
        return;
    }
    std::uint64_t bytes = 0;
    if (!mul_u64_checked(
            static_cast<std::uint64_t>(values.size()),
            static_cast<std::uint64_t>(sizeof(T)),
            bytes)) {
        total = std::numeric_limits<std::uint64_t>::max();
        overflow = true;
        return;
    }
    add_estimate_bytes_u64(UploadEstimateBytes{bytes}, UploadEstimateAccumulator{&total, &overflow});
}

void add_curve_upload_estimate_bytes(
    const Spectral::Curve& curve,
    std::uint64_t& total,
    bool& overflow) noexcept {
    const std::size_t n = std::min(curve.lambda_nm.size(), curve.linear.size());
    if (n == 0 || overflow) {
        return;
    }
    std::uint64_t bytes = 0;
    if (!mul_u64_checked(static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(2u * sizeof(float)), bytes)) {
        total = std::numeric_limits<std::uint64_t>::max();
        overflow = true;
        return;
    }
    add_estimate_bytes_u64(UploadEstimateBytes{bytes}, UploadEstimateAccumulator{&total, &overflow});
}

void add_scan_medium_upload_estimate_bytes(
    const Scanner::ScannerMediumRuntime& medium,
    std::uint64_t& total,
    bool& overflow) noexcept {
    const Spectral::SpectralTables* t = medium.tables;
    if (!t) {
        return;
    }
    add_vector_upload_estimate_bytes(t->epsC, total, overflow);
    add_vector_upload_estimate_bytes(t->epsM, total, overflow);
    add_vector_upload_estimate_bytes(t->epsY, total, overflow);
    add_vector_upload_estimate_bytes(t->Ax, total, overflow);
    add_vector_upload_estimate_bytes(t->Ay, total, overflow);
    add_vector_upload_estimate_bytes(t->Az, total, overflow);
    if (t->hasBaseline) {
        add_vector_upload_estimate_bytes(t->baseDensityMin, total, overflow);
    }
}

void add_curve_growth_estimate_bytes(
    const JuicerCuda::DeviceCurve& dst,
    const Spectral::Curve& src,
    std::uint64_t& total,
    bool& overflow) noexcept {
    if (src.lambda_nm.empty() || src.linear.empty() || src.lambda_nm.size() != src.linear.size()) {
        return;
    }
    const int n = static_cast<int>(src.lambda_nm.size());
    if (n <= 0) {
        return;
    }
    if (dst.x && dst.y && dst.n == n) {
        return;
    }
    std::uint64_t bytes = 0;
    if (!mul_u64_checked(static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(2u * sizeof(float)), bytes)) {
        total = std::numeric_limits<std::uint64_t>::max();
        overflow = true;
        return;
    }
    add_estimate_bytes_u64(UploadEstimateBytes{bytes}, UploadEstimateAccumulator{&total, &overflow});
}

void add_array_growth_estimate_bytes(
    const float* dst,
    int currentN,
    int requestedN,
    std::uint64_t& total,
    bool& overflow) noexcept {
    if (requestedN <= 0) {
        return;
    }
    if (dst && currentN == requestedN) {
        return;
    }
    add_count_upload_estimate_bytes(
        static_cast<std::uint64_t>(requestedN),
        sizeof(float),
        total,
        overflow);
}

void add_spectral_sample_growth_estimate_bytes(
    const JuicerCuda::DeviceCurve& dst,
    const std::vector<float>& src,
    std::uint64_t& total,
    bool& overflow) noexcept {
    if (src.empty()) {
        return;
    }
    add_array_growth_estimate_bytes(
        dst.y,
        dst.n,
        static_cast<int>(src.size()),
        total,
        overflow);
}

void add_density_layers_growth_estimate_bytes(
    const JuicerCuda::Resources& resources,
    const WorkingState& ws,
    std::uint64_t& total,
    bool& overflow) noexcept {
    const int nR = static_cast<int>(ws.densR.linear.size());
    const int nG = static_cast<int>(ws.densG.linear.size());
    const int nB = static_cast<int>(ws.densB.linear.size());
    const int layerChannelN[3] = {nR, nG, nB};
    bool sizesOk = ws.hasDensityCurvesLayers && (nR > 0 && nG > 0 && nB > 0);
    if (sizesOk) {
        for (int layer = 0; layer < 3; ++layer) {
            sizesOk = sizesOk && (static_cast<int>(ws.densityCurvesLayers[layer][0].size()) == nR);
            sizesOk = sizesOk && (static_cast<int>(ws.densityCurvesLayers[layer][1].size()) == nG);
            sizesOk = sizesOk && (static_cast<int>(ws.densityCurvesLayers[layer][2].size()) == nB);
        }
    }
    if (!sizesOk) {
        return;
    }

    bool canReuse = resources.hasDensityCurvesLayers != 0;
    if (canReuse) {
        for (int ch = 0; ch < 3; ++ch) {
            canReuse = canReuse && (resources.densityCurvesLayersChannelN[ch] == layerChannelN[ch]);
        }
        for (int layer = 0; layer < 3 && canReuse; ++layer) {
            for (int ch = 0; ch < 3; ++ch) {
                canReuse = canReuse && (resources.densityCurvesLayers[layer][ch] != nullptr);
            }
        }
    }
    if (canReuse) {
        return;
    }

    for (int layer = 0; layer < 3; ++layer) {
        for (int ch = 0; ch < 3; ++ch) {
            add_count_upload_estimate_bytes(
                static_cast<std::uint64_t>(layerChannelN[ch]),
                sizeof(float),
                total,
                overflow);
        }
    }
}

void add_tables_growth_estimate_bytes(
    const JuicerCuda::Resources& resources,
    const WorkingState& ws,
    std::uint64_t& total,
    bool& overflow) noexcept {
    const int K = ws.tablesRef.K;
    const bool want =
        ws.spdReady &&
        K == Spectral::gShape.K &&
        static_cast<int>(ws.tablesRef.Ax.size()) == K &&
        static_cast<int>(ws.tablesRef.Ay.size()) == K &&
        static_cast<int>(ws.tablesRef.Az.size()) == K &&
        static_cast<int>(ws.tablesRef.illum.size()) == K;
    if (!want) {
        return;
    }
    const bool canReuse =
        resources.tablesK == K &&
        resources.tablesAx &&
        resources.tablesAy &&
        resources.tablesAz &&
        resources.tablesIllum;
    if (canReuse) {
        return;
    }

    add_count_upload_estimate_bytes(static_cast<std::uint64_t>(K), sizeof(float), total, overflow);
    add_count_upload_estimate_bytes(static_cast<std::uint64_t>(K), sizeof(float), total, overflow);
    add_count_upload_estimate_bytes(static_cast<std::uint64_t>(K), sizeof(float), total, overflow);
    add_count_upload_estimate_bytes(static_cast<std::uint64_t>(K), sizeof(float), total, overflow);
}

void add_scan_medium_growth_estimate_bytes(
    const JuicerCuda::Resources::DeviceScanMedium& dst,
    const Scanner::ScannerMediumRuntime& medium,
    std::uint64_t& total,
    bool& overflow) noexcept {
    const Spectral::SpectralTables* t = medium.tables;
    if (!t || t->K != Spectral::gShape.K) {
        return;
    }
    const int K = t->K;
    const bool arraysOk =
        static_cast<int>(t->epsC.size()) == K &&
        static_cast<int>(t->epsM.size()) == K &&
        static_cast<int>(t->epsY.size()) == K &&
        static_cast<int>(t->Ax.size()) == K &&
        static_cast<int>(t->Ay.size()) == K &&
        static_cast<int>(t->Az.size()) == K &&
        (!t->hasBaseline || static_cast<int>(t->baseDensityMin.size()) == K);
    if (!arraysOk) {
        return;
    }

    const bool haveCoreArrays =
        dst.tables.epsC &&
        dst.tables.epsM &&
        dst.tables.epsY &&
        dst.tables.Ax &&
        dst.tables.Ay &&
        dst.tables.Az;
    const bool canReuseCore = (dst.tables.K == K) && haveCoreArrays;
    if (!canReuseCore) {
        for (int i = 0; i < 6; ++i) {
            add_count_upload_estimate_bytes(static_cast<std::uint64_t>(K), sizeof(float), total, overflow);
        }
    }

    if (t->hasBaseline && !(dst.tables.baseDensityMin && dst.tables.K == K)) {
        add_count_upload_estimate_bytes(static_cast<std::uint64_t>(K), sizeof(float), total, overflow);
    }
}

void add_print_payload_growth_estimate_bytes(
    const JuicerCuda::Resources& resources,
    const WorkingState& ws,
    std::uint64_t& total,
    bool& overflow) noexcept {
    const Print::Runtime* prt = ws.printRT.get();
    if (!prt || !Print::profile_is_valid(prt->profile)) {
        return;
    }
    const Print::Profile& p = prt->profile;

    auto add_print_curve_growth = [&](const JuicerCuda::DeviceCurve& dst, const Spectral::Curve& src) {
        const int n = static_cast<int>(src.lambda_nm.size());
        const bool want = n > 1 && src.linear.size() == src.lambda_nm.size();
        if (!want) {
            return;
        }
        add_curve_growth_estimate_bytes(dst, src, total, overflow);
    };

    add_print_curve_growth(resources.printDcC, p.dcC);
    add_print_curve_growth(resources.printDcM, p.dcM);
    add_print_curve_growth(resources.printDcY, p.dcY);

    const int K = Spectral::gShape.K;
    const bool sensOk =
        K > 0 &&
        static_cast<int>(p.sensC_log.linear.size()) == K &&
        static_cast<int>(p.sensM_log.linear.size()) == K &&
        static_cast<int>(p.sensY_log.linear.size()) == K;
    if (!sensOk) {
        return;
    }

    add_spectral_sample_growth_estimate_bytes(resources.printSensC, p.sensC_log.linear, total, overflow);
    add_spectral_sample_growth_estimate_bytes(resources.printSensM, p.sensM_log.linear, total, overflow);
    add_spectral_sample_growth_estimate_bytes(resources.printSensY, p.sensY_log.linear, total, overflow);
}

UploadWorkEstimate estimate_upload_core_request_bytes(
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    bool includeCurrentMediumUploads,
    bool negativeMedium) noexcept {
    UploadWorkEstimate estimate{};
    const std::uint64_t wsCoreHash = commands_preferred_upload_core_hash(ws);
    const std::uint64_t wsDirHash = ws.dirHash;
    if (wsCoreHash == 0 || wsDirHash == 0) {
        estimate.reservationBytes = kUploadReservationThresholdDefaultBytes;
        return estimate;
    }

    bool coreUpToDate = false;
    bool dirUpToDate = false;
    bool needHanatosUpload = false;
    bool needHanatosRetire = false;
    bool needHanatosIntegratedUpload = false;
    bool needHanatosIntegratedRetire = false;
    bool needMallettUpload = false;
    bool needMallettRetire = false;
    bool needAuxiliaryPackageUpdate = false;
    std::uint64_t hanatosUploadCount = 0;
    std::uint64_t hanatosIntegratedUploadCount = 0;
    std::uint64_t mallettUploadCount = 0;
    Spectral::SpectralContext& ctx = Spectral::context();
    const bool hanatosAvailable = ctx.hanatosAvailable.load(std::memory_order_acquire);
    const int hanatosN = ctx.hanSpectra.size;
    const int hanatosK = ctx.hanSpectra.numSamples;
    const bool wantHanatos =
        hanatosAvailable &&
        hanatosN > 0 &&
        hanatosK == Spectral::kNumSamples &&
        !ctx.hanSpectra.data.empty();
    const bool wantHanatosIntegrated =
        wantHanatos &&
        static_cast<int>(ws.sensB.linear.size()) == hanatosK &&
        static_cast<int>(ws.sensG.linear.size()) == hanatosK &&
        static_cast<int>(ws.sensR.linear.size()) == hanatosK;
    const bool mallettAvailable = ctx.mallettAvailable.load(std::memory_order_acquire);
    const int mallettK = ctx.mallettBasis.rows;
    const int mallettCols = ctx.mallettBasis.cols;
    const bool wantMallett =
        mallettAvailable &&
        mallettK == Spectral::kNumSamples &&
        mallettCols == 3 &&
        !ctx.mallettBasis.data.empty();
    bool overflow = false;
    {
        std::lock_guard<std::mutex> lock(resources.m);
        coreUpToDate = (resources.uploadedCoreHash != 0) && (resources.uploadedCoreHash == wsCoreHash);
        dirUpToDate = (resources.uploadedDirHash != 0) && (resources.uploadedDirHash == wsDirHash);
        needHanatosUpload = wantHanatos &&
                            (!resources.hanatosLut || resources.hanatosN != hanatosN);
        needHanatosRetire = !wantHanatos && resources.hanatosLut;
        needHanatosIntegratedUpload = wantHanatosIntegrated &&
                                      ((!resources.hanatosLutIntegrated || resources.hanatosNIntegrated != hanatosN) ||
                                       (resources.hanatosIntegratedKeyHash != wsCoreHash));
        needHanatosIntegratedRetire = !wantHanatosIntegrated && resources.hanatosLutIntegrated;
        needMallettUpload = wantMallett &&
                            (!resources.mallettBasis || resources.mallettBasisK != mallettK);
        needMallettRetire = !wantMallett && resources.mallettBasis;
        needAuxiliaryPackageUpdate =
            needHanatosUpload ||
            needHanatosRetire ||
            needHanatosIntegratedUpload ||
            needHanatosIntegratedRetire ||
            needMallettUpload ||
            needMallettRetire;
        if (coreUpToDate && dirUpToDate && !includeCurrentMediumUploads && !needAuxiliaryPackageUpdate) {
            if (overflow) {
                estimate.growthBytes = std::numeric_limits<std::uint64_t>::max();
                estimate.reservationBytes = std::numeric_limits<std::uint64_t>::max();
            }
            return estimate;
        }

        if (coreUpToDate && !dirUpToDate) {
            add_curve_growth_estimate_bytes(resources.dirDensB, ws.dirDensB, estimate.growthBytes, overflow);
            add_curve_growth_estimate_bytes(resources.dirDensG, ws.dirDensG, estimate.growthBytes, overflow);
            add_curve_growth_estimate_bytes(resources.dirDensR, ws.dirDensR, estimate.growthBytes, overflow);
        } else {
            add_curve_growth_estimate_bytes(resources.densB, ws.densB, estimate.growthBytes, overflow);
            add_curve_growth_estimate_bytes(resources.densG, ws.densG, estimate.growthBytes, overflow);
            add_curve_growth_estimate_bytes(resources.densR, ws.densR, estimate.growthBytes, overflow);
            add_density_layers_growth_estimate_bytes(resources, ws, estimate.growthBytes, overflow);
            add_curve_growth_estimate_bytes(resources.dirDensB, ws.dirDensB, estimate.growthBytes, overflow);
            add_curve_growth_estimate_bytes(resources.dirDensG, ws.dirDensG, estimate.growthBytes, overflow);
            add_curve_growth_estimate_bytes(resources.dirDensR, ws.dirDensR, estimate.growthBytes, overflow);
            add_curve_growth_estimate_bytes(resources.sensB, ws.sensB, estimate.growthBytes, overflow);
            add_curve_growth_estimate_bytes(resources.sensG, ws.sensG, estimate.growthBytes, overflow);
            add_curve_growth_estimate_bytes(resources.sensR, ws.sensR, estimate.growthBytes, overflow);
            add_tables_growth_estimate_bytes(resources, ws, estimate.growthBytes, overflow);
        }

        if (includeCurrentMediumUploads) {
            add_scan_medium_growth_estimate_bytes(
                resources.scanNegative,
                ws.negativeMediumRuntime,
                estimate.growthBytes,
                overflow);
            if (!negativeMedium) {
                add_scan_medium_growth_estimate_bytes(
                    resources.scanPrint,
                    ws.printMediumRuntime,
                    estimate.growthBytes,
                    overflow);
                add_print_payload_growth_estimate_bytes(
                    resources,
                    ws,
                    estimate.growthBytes,
                    overflow);
            }
        }

        if (wantHanatos) {
            std::uint64_t count = static_cast<std::uint64_t>(hanatosN);
            if (!mul_u64_checked(count, static_cast<std::uint64_t>(hanatosN), count) ||
                !mul_u64_checked(count, static_cast<std::uint64_t>(hanatosK), count)) {
                estimate.growthBytes = std::numeric_limits<std::uint64_t>::max();
                estimate.reservationBytes = std::numeric_limits<std::uint64_t>::max();
                return estimate;
            }
            hanatosUploadCount = count;
        }
        if (wantHanatosIntegrated) {
            std::uint64_t count = static_cast<std::uint64_t>(hanatosN);
            if (!mul_u64_checked(count, static_cast<std::uint64_t>(hanatosN), count) ||
                !mul_u64_checked(count, 4ull, count)) {
                estimate.growthBytes = std::numeric_limits<std::uint64_t>::max();
                estimate.reservationBytes = std::numeric_limits<std::uint64_t>::max();
                return estimate;
            }
            hanatosIntegratedUploadCount = count;
        }
        if (wantMallett) {
            std::uint64_t count = static_cast<std::uint64_t>(mallettK);
            if (!mul_u64_checked(count, 3ull, count)) {
                estimate.growthBytes = std::numeric_limits<std::uint64_t>::max();
                estimate.reservationBytes = std::numeric_limits<std::uint64_t>::max();
                return estimate;
            }
            mallettUploadCount = count;
        }

        if (needHanatosUpload) {
            add_count_upload_estimate_bytes(hanatosUploadCount, sizeof(float), estimate.growthBytes, overflow);
        }
        if (needHanatosIntegratedUpload &&
            (!resources.hanatosLutIntegrated || resources.hanatosNIntegrated != hanatosN)) {
            add_count_upload_estimate_bytes(
                hanatosIntegratedUploadCount,
                sizeof(float),
                estimate.growthBytes,
                overflow);
        }
        if (needMallettUpload) {
            add_count_upload_estimate_bytes(mallettUploadCount, sizeof(float), estimate.growthBytes, overflow);
        }
    }

    if (overflow) {
        estimate.growthBytes = std::numeric_limits<std::uint64_t>::max();
        estimate.reservationBytes = std::numeric_limits<std::uint64_t>::max();
        return estimate;
    }

    if (coreUpToDate && !dirUpToDate) {
        add_curve_upload_estimate_bytes(ws.dirDensB, estimate.reservationBytes, overflow);
        add_curve_upload_estimate_bytes(ws.dirDensG, estimate.reservationBytes, overflow);
        add_curve_upload_estimate_bytes(ws.dirDensR, estimate.reservationBytes, overflow);
    } else if (!coreUpToDate) {
        add_curve_upload_estimate_bytes(ws.densB, estimate.reservationBytes, overflow);
        add_curve_upload_estimate_bytes(ws.densG, estimate.reservationBytes, overflow);
        add_curve_upload_estimate_bytes(ws.densR, estimate.reservationBytes, overflow);
        add_curve_upload_estimate_bytes(ws.dirDensB, estimate.reservationBytes, overflow);
        add_curve_upload_estimate_bytes(ws.dirDensG, estimate.reservationBytes, overflow);
        add_curve_upload_estimate_bytes(ws.dirDensR, estimate.reservationBytes, overflow);
        add_curve_upload_estimate_bytes(ws.sensB, estimate.reservationBytes, overflow);
        add_curve_upload_estimate_bytes(ws.sensG, estimate.reservationBytes, overflow);
        add_curve_upload_estimate_bytes(ws.sensR, estimate.reservationBytes, overflow);
        add_vector_upload_estimate_bytes(ws.tablesRef.Ax, estimate.reservationBytes, overflow);
        add_vector_upload_estimate_bytes(ws.tablesRef.Ay, estimate.reservationBytes, overflow);
        add_vector_upload_estimate_bytes(ws.tablesRef.Az, estimate.reservationBytes, overflow);
        add_vector_upload_estimate_bytes(ws.tablesRef.illum, estimate.reservationBytes, overflow);

        for (int layer = 0; layer < 3; ++layer) {
            for (int ch = 0; ch < 3; ++ch) {
                add_vector_upload_estimate_bytes(ws.densityCurvesLayers[layer][ch], estimate.reservationBytes, overflow);
            }
        }

        if (needHanatosUpload) {
            add_count_upload_estimate_bytes(hanatosUploadCount, sizeof(float), estimate.reservationBytes, overflow);
        }
        if (needHanatosIntegratedUpload) {
            add_count_upload_estimate_bytes(hanatosIntegratedUploadCount, sizeof(float), estimate.reservationBytes, overflow);
        }
        if (needMallettUpload) {
            add_count_upload_estimate_bytes(mallettUploadCount, sizeof(float), estimate.reservationBytes, overflow);
        }
    }

    if (coreUpToDate) {
        if (needHanatosUpload) {
            add_count_upload_estimate_bytes(hanatosUploadCount, sizeof(float), estimate.reservationBytes, overflow);
        }
        if (needHanatosIntegratedUpload) {
            add_count_upload_estimate_bytes(hanatosIntegratedUploadCount, sizeof(float), estimate.reservationBytes, overflow);
        }
        if (needMallettUpload) {
            add_count_upload_estimate_bytes(mallettUploadCount, sizeof(float), estimate.reservationBytes, overflow);
        }
    }

    if (includeCurrentMediumUploads) {
        add_scan_medium_upload_estimate_bytes(
            ws.negativeMediumRuntime,
            estimate.reservationBytes,
            overflow);
        if (!negativeMedium) {
            add_scan_medium_upload_estimate_bytes(
                ws.printMediumRuntime,
                estimate.reservationBytes,
                overflow);
            if (ws.printRT && Print::profile_is_valid(ws.printRT->profile)) {
                const Print::Profile& p = ws.printRT->profile;
                add_curve_upload_estimate_bytes(p.dcC, estimate.reservationBytes, overflow);
                add_curve_upload_estimate_bytes(p.dcM, estimate.reservationBytes, overflow);
                add_curve_upload_estimate_bytes(p.dcY, estimate.reservationBytes, overflow);
                add_vector_upload_estimate_bytes(p.sensC_log.linear, estimate.reservationBytes, overflow);
                add_vector_upload_estimate_bytes(p.sensM_log.linear, estimate.reservationBytes, overflow);
                add_vector_upload_estimate_bytes(p.sensY_log.linear, estimate.reservationBytes, overflow);
            }
        }
    }

    if (overflow) {
        estimate.growthBytes = std::numeric_limits<std::uint64_t>::max();
        estimate.reservationBytes = std::numeric_limits<std::uint64_t>::max();
        return estimate;
    }

    // Keep a deterministic floor so rebuilds and retire-only auxiliary updates enter upload admission.
    if ((!coreUpToDate || needAuxiliaryPackageUpdate) &&
        estimate.reservationBytes < kUploadReservationThresholdDefaultBytes) {
        estimate.reservationBytes = kUploadReservationThresholdDefaultBytes;
    }
    return estimate;
}

UploadWorkEstimate estimate_scan_lut_upload_bytes(
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    bool negativeMedium) noexcept {
    const Scanner::ScannerStaticKey& staticKey =
        commands_select_scanner_static_key(ws, negativeMedium);
    const std::uint32_t res =
        ResourceManager::normalize_scan_lut_resolution(staticKey.lutResolution);
    std::uint64_t voxelCount = 0;
    if (!mul_u64_checked(static_cast<std::uint64_t>(res), static_cast<std::uint64_t>(res), voxelCount)) {
        return {std::numeric_limits<std::uint64_t>::max(), std::numeric_limits<std::uint64_t>::max()};
    }
    if (!mul_u64_checked(voxelCount, static_cast<std::uint64_t>(res), voxelCount)) {
        return {std::numeric_limits<std::uint64_t>::max(), std::numeric_limits<std::uint64_t>::max()};
    }
    std::uint64_t values = 0;
    if (!mul_u64_checked(voxelCount, 3ull, values)) {
        return {std::numeric_limits<std::uint64_t>::max(), std::numeric_limits<std::uint64_t>::max()};
    }
    std::uint64_t voxelBytes = 0;
    if (!mul_u64_checked(values, static_cast<std::uint64_t>(sizeof(float)), voxelBytes)) {
        return {std::numeric_limits<std::uint64_t>::max(), std::numeric_limits<std::uint64_t>::max()};
    }
    std::uint64_t bytes = 0;
    if (!mul_u64_checked(voxelBytes, 4ull, bytes)) {
        return {std::numeric_limits<std::uint64_t>::max(), std::numeric_limits<std::uint64_t>::max()};
    }
    if (res >= 2u) {
        std::uint64_t cellCount = 0;
        const std::uint64_t cellRes = static_cast<std::uint64_t>(res - 1u);
        if (!mul_u64_checked(cellRes, cellRes, cellCount) ||
            !mul_u64_checked(cellCount, cellRes, cellCount) ||
            !mul_u64_checked(cellCount, 3ull, cellCount)) {
            return {std::numeric_limits<std::uint64_t>::max(), std::numeric_limits<std::uint64_t>::max()};
        }
        std::uint64_t cellBytes = 0;
        if (!mul_u64_checked(cellCount, static_cast<std::uint64_t>(sizeof(float)), cellBytes) ||
            !mul_u64_checked(cellBytes, 2ull, cellBytes)) {
            return {std::numeric_limits<std::uint64_t>::max(), std::numeric_limits<std::uint64_t>::max()};
        }
        if (std::numeric_limits<std::uint64_t>::max() - bytes < cellBytes) {
            return {std::numeric_limits<std::uint64_t>::max(), std::numeric_limits<std::uint64_t>::max()};
        }
        bytes += cellBytes;
    }

    const Scanner::ScannerMediumRuntime& medium =
        commands_select_scanner_medium_runtime(ws, negativeMedium);
    if (!medium.tables || medium.tables->tablesHash == 0 || medium.range.digest == 0) {
        return {};
    }
    const std::uint64_t expectedHash = ResourceManager::make_scan_lut_key_digest(
        static_cast<std::uint32_t>(medium.medium),
        medium.tables->tablesHash,
        medium.range.digest,
        res);
    if (expectedHash == 0) {
        return {};
    }

    bool cached = false;
    bool canOverwriteInPlace = false;
    {
        std::lock_guard<std::mutex> lock(resources.m);
        const JuicerCuda::Resources::DeviceSpectralLut& dst =
            commands_select_scan_lut_slot(resources, negativeMedium);
        cached = dst.canonical_ready() && dst.res == res && dst.hash == expectedHash;
        canOverwriteInPlace = dst.canonical_ready() && dst.res == res;
    }
    UploadWorkEstimate uncachedWork{};
    uncachedWork.growthBytes = canOverwriteInPlace ? 0 : bytes;
    uncachedWork.reservationBytes = bytes;
    return commands_uncached_upload_work(cached, uncachedWork);
}

bool command_scan_lut_is_cached(
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    bool negativeMedium) noexcept {
    const Scanner::ScannerStaticKey& staticKey =
        commands_select_scanner_static_key(ws, negativeMedium);
    const std::uint32_t res =
        ResourceManager::normalize_scan_lut_resolution(staticKey.lutResolution);
    const Scanner::ScannerMediumRuntime& medium =
        commands_select_scanner_medium_runtime(ws, negativeMedium);
    if (!medium.tables || medium.tables->tablesHash == 0 || medium.range.digest == 0) {
        return false;
    }
    const std::uint64_t expectedHash = ResourceManager::make_scan_lut_key_digest(
        static_cast<std::uint32_t>(medium.medium),
        medium.tables->tablesHash,
        medium.range.digest,
        res);
    if (expectedHash == 0) {
        return false;
    }

    std::lock_guard<std::mutex> lock(resources.m);
    const JuicerCuda::Resources::DeviceSpectralLut& dst =
        commands_select_scan_lut_slot(resources, negativeMedium);
    return dst.canonical_ready() && dst.res == res && dst.hash == expectedHash;
}

UploadWorkEstimate estimate_print_illuminant_upload_bytes(
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    const Print::Runtime& prt,
    const Print::Params& params) noexcept {
    const int k = Spectral::gShape.K;
    if (k <= 0) {
        return {};
    }
    const std::uint64_t wsCoreHash = commands_preferred_upload_core_hash(ws);
    if (wsCoreHash == 0) {
        return {};
    }
    const float yKey = commands_sanitize_filter_shift_step(params.yFilter);
    const float mKey = commands_sanitize_filter_shift_step(params.mFilter);
    const float cKey = commands_sanitize_filter_shift_step(params.cFilter);
    const std::uint64_t neutralFilterHash = commands_neutral_filter_hash_or_default(prt);

    bool cached = false;
    bool canOverwriteInPlace = false;
    {
        std::lock_guard<std::mutex> lock(resources.m);
        cached =
            resources.printIllumFiltered &&
            resources.printIllumK == k &&
            resources.printIllumShapeK == k &&
            resources.printIllumCoreHash == wsCoreHash &&
            resources.printIllumYShiftSteps == yKey &&
            resources.printIllumMShiftSteps == mKey &&
            resources.printIllumCShiftSteps == cKey &&
            resources.printIllumNeutralFilterHash == neutralFilterHash;
        canOverwriteInPlace =
            resources.printIllumFiltered &&
            resources.printIllumK == k;
    }
    if (cached) {
        return {};
    }
    std::uint64_t bytes = 0;
    if (!mul_u64_checked(
            static_cast<std::uint64_t>(k),
            static_cast<std::uint64_t>(sizeof(float)),
            bytes)) {
        return {std::numeric_limits<std::uint64_t>::max(), std::numeric_limits<std::uint64_t>::max()};
    }
    return {canOverwriteInPlace ? 0 : bytes, bytes};
}

bool command_print_illuminant_filtered_is_cached(
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    const Print::Runtime& prt,
    const Print::Params& params) noexcept {
    const int k = Spectral::gShape.K;
    if (k <= 0) {
        return false;
    }
    const std::uint64_t wsCoreHash = commands_preferred_upload_core_hash(ws);
    if (wsCoreHash == 0) {
        return false;
    }
    const float yKey = commands_sanitize_filter_shift_step(params.yFilter);
    const float mKey = commands_sanitize_filter_shift_step(params.mFilter);
    const float cKey = commands_sanitize_filter_shift_step(params.cFilter);
    const std::uint64_t neutralFilterHash = commands_neutral_filter_hash_or_default(prt);

    std::lock_guard<std::mutex> lock(resources.m);
    return resources.printIllumFiltered &&
           resources.printIllumK == k &&
           resources.printIllumShapeK == k &&
           resources.printIllumCoreHash == wsCoreHash &&
           resources.printIllumYShiftSteps == yKey &&
           resources.printIllumMShiftSteps == mKey &&
           resources.printIllumCShiftSteps == cKey &&
           resources.printIllumNeutralFilterHash == neutralFilterHash;
}

bool command_freeze_drain_bump_resume(
    const DeviceContextKey& key,
    const char* reason,
    std::string& outError) {
    outError.clear();
    MetadataMutationGuard mutationGuard("command_freeze_drain_bump_resume", &key);
    if (!mutationGuard.ok()) {
        outError = "metadata mutation guard rejected command_freeze_drain_bump_resume";
        return false;
    }
    if (!registry_freeze_drain_bump_resume(key, reason)) {
        outError = "freeze-drain-bump-resume barrier rejected";
        return false;
    }
    return true;
}

bool command_begin_context_owner_retire(
    const DeviceContextKey& key,
    RegistrySnapshotGenerations& outGenerations,
    std::string& outError) {
    outGenerations = RegistrySnapshotGenerations{};
    outError.clear();
    if (!registry_begin_owner_retire(key, outGenerations)) {
        outError = "context owner-retire freeze/drain rejected";
        return false;
    }
    if (outGenerations.contextEpoch == 0 ||
        outGenerations.registryGeneration == 0) {
        outError = "context owner-retire generation identity invalid";
        return false;
    }
    return true;
}

namespace {
    bool command_retire_context_with_reason(
        const DeviceContextKey& key,
        RegistryRetireReason reason,
        const char* commandName,
        std::string& outError) {
        outError.clear();
        const char* stageName = trace_or(commandName, "command_retire_context");
        MetadataMutationGuard mutationGuard(stageName, &key);
        if (!mutationGuard.ok()) {
            outError = std::string("metadata mutation guard rejected ") + stageName;
            return false;
        }

        RegistryHandle handle{};
        const bool hasRegistryEntry = registry_get(key, handle) && handle.value != 0;
        if (hasRegistryEntry && !registry_retire(handle, reason, &key)) {
            outError = std::string(stageName) + " registry retire rejected";
            return false;
        }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        JuicerCuda::purge_pinned_upload_staging_for_context(
            key.deviceId,
            key.contextOpaque);
#endif
        tier_circuit_retire_context(key);
        pressure_policy_retire_context(key);
        scratch_normalization_retire_context(key);
        admission_churn_retire_context(key);
        optional_heuristic_trace_retire_context(key);
        state_clear_latest_snapshot_for_context(key);
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        JuicerCuda::purge_host_asset_caches_if_registry_idle(stageName);
#endif
        return true;
    }
} // namespace

bool command_retire_context_reset(
    const DeviceContextKey& key,
    std::string& outError) {
    return command_retire_context_with_reason(
        key,
        RegistryRetireReason::ContextReset,
        "command_retire_context_reset",
        outError);
}

bool command_retire_context_idle(
    const DeviceContextKey& key,
    std::string& outError) {
    return command_retire_context_with_reason(
        key,
        RegistryRetireReason::Idle,
        "command_retire_context_idle",
        outError);
}

bool command_retire_all_contexts_idle(std::string& outError) {
    outError.clear();
    std::vector<DeviceContextKey> keys;
    try {
        registry_snapshot_context_keys(keys);
    } catch (...) {
        outError = "context key snapshot failed";
        return false;
    }

    bool ok = true;
    for (const DeviceContextKey& key : keys) {
        std::string retireError;
        if (!command_retire_context_idle(key, retireError)) {
            ok = false;
            if (outError.empty()) {
                outError = retireError.empty() ? "context retire failed" : retireError;
            }
        }
    }
    try {
        keys.clear();
        registry_snapshot_context_keys(keys);
    } catch (...) {
        if (outError.empty()) {
            outError = "context key verification failed";
        }
        return false;
    }
    if (!keys.empty()) {
        ok = false;
        if (outError.empty()) {
            outError = std::string("context retire incomplete; live_contexts=") +
                       std::to_string(static_cast<unsigned long long>(keys.size()));
        }
    }
    return ok;
}

namespace {
    bool command_ensure_scan_lut_internal(
        SubmissionTransaction& transaction,
        JuicerCuda::Resources& resources,
        const WorkingState& ws,
        bool negativeMedium,
        bool criticalRequest,
        const ScratchRequestDescriptor* scratchRequest,
        const char* commandName,
        void* cudaStreamOpaque,
        std::string& outError);
} // namespace

bool command_ensure_uploaded(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_uploaded")) {
        return false;
    }
    const UploadWorkEstimate uploadEstimate =
        estimate_upload_core_request_bytes(resources, ws, false, false);
    if (uploadEstimate.reservationBytes == 0) {
        return true;
    }
    return execute_upload_immutable_command(
        transaction,
        resources,
        "command_ensure_uploaded",
        uploadEstimate.growthBytes,
        uploadEstimate.reservationBytes,
        true,
        nullptr,
        "upload reservation request byte estimation overflow (core)",
        [&](std::string& actionError) {
            return JuicerCuda::ensure_uploaded(
                resources,
                ws,
                false,
                false,
                cudaStreamOpaque,
                actionError);
        },
        outError);
}

bool command_ensure_current_medium_uploaded(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    bool negativeMedium,
    const ScratchRequestDescriptor& scratchRequest,
    void* cudaStreamOpaque,
    std::string& outError) {
    const char* stageName = negativeMedium
                                ? "command_ensure_current_medium_uploaded_negative"
                                : "current_medium_upload_print";
    if (!ensure_active_for_command(transaction, outError, stageName)) {
        return false;
    }
    const ScratchRequestDescriptor* activeScratchRequest = nullptr;
    if (!resolve_optional_active_scratch_request(
            scratchRequest,
            stageName,
            activeScratchRequest,
            outError)) {
        return false;
    }
    (void)activeScratchRequest;

    const UploadWorkEstimate uploadEstimate =
        estimate_upload_core_request_bytes(resources, ws, true, negativeMedium);
    if (uploadEstimate.reservationBytes == 0) {
        return true;
    }
    return execute_upload_immutable_command(
        transaction,
        resources,
        stageName,
        uploadEstimate.growthBytes,
        uploadEstimate.reservationBytes,
        true,
        activeScratchRequest,
        "upload reservation request byte estimation overflow (current medium)",
        [&](std::string& actionError) {
            return JuicerCuda::ensure_uploaded(
                resources,
                ws,
                true,
                negativeMedium,
                cudaStreamOpaque,
                actionError);
        },
        outError);
}

namespace {
    bool command_ensure_scan_lut_internal(
        SubmissionTransaction& transaction,
        JuicerCuda::Resources& resources,
        const WorkingState& ws,
        bool negativeMedium,
        bool criticalRequest,
        const ScratchRequestDescriptor* scratchRequest,
        const char* commandName,
        void* cudaStreamOpaque,
        std::string& outError) {
        const char* stageName = trace_or_non_empty(commandName, "command_ensure_scan_lut");
        if (!ensure_active_for_command(transaction, outError, stageName)) {
            return false;
        }
        const ScratchRequestDescriptor* activeScratchRequest = nullptr;
        if (scratchRequest &&
            !resolve_optional_active_scratch_request(
                *scratchRequest,
                stageName,
                activeScratchRequest,
                outError)) {
            return false;
        }
        const UploadWorkEstimate uploadEstimate =
            estimate_scan_lut_upload_bytes(resources, ws, negativeMedium);
        if (uploadEstimate.growthBytes == std::numeric_limits<std::uint64_t>::max() ||
            uploadEstimate.reservationBytes == std::numeric_limits<std::uint64_t>::max()) {
            outError = "upload reservation request byte estimation overflow (scan LUT)";
            return false;
        }
        const bool captureMemorySnapshots =
            should_collect_manager_memory_snapshots(transaction.resolvedPressurePolicy);
        maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
        if (command_scan_lut_is_cached(resources, ws, negativeMedium)) {
            return true;
        }
        return execute_upload_lut_command(
            transaction,
            resources,
            stageName,
            uploadEstimate.growthBytes,
            uploadEstimate.reservationBytes,
            criticalRequest,
            activeScratchRequest,
            "upload reservation request byte estimation overflow (scan LUT)",
            captureMemorySnapshots,
            false,
            [&](std::string& actionError) {
                return JuicerCuda::ensure_scan_lut(
                    resources,
                    ws,
                    negativeMedium,
                    cudaStreamOpaque,
                    actionError);
            },
            [&](const char* triggerReason) {
                (void)triggerReason;
                return false;
            },
            outError);
    }
} // namespace

bool command_ensure_scan_lut(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    bool negativeMedium,
    const ScratchRequestDescriptor& scratchRequest,
    void* cudaStreamOpaque,
    std::string& outError) {
    return command_ensure_scan_lut_internal(
        transaction,
        resources,
        ws,
        negativeMedium,
        true,
        &scratchRequest,
        "command_ensure_scan_lut",
        cudaStreamOpaque,
        outError);
}

bool command_checkpoint_scratch_phase(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const ScratchRequestDescriptor& scratchRequest,
    const char* commandName,
    std::string& outError) {
    const char* stageName = trace_or_non_empty(commandName, "command_checkpoint_scratch_phase");
    if (!ensure_active_for_command(transaction, outError, stageName)) {
        return false;
    }
    if (!validate_scratch_request_descriptor_for_manager(
            scratchRequest,
            stageName,
            outError)) {
        return false;
    }

    trace_scratch_request_descriptor(transaction, stageName, scratchRequest, "phase_declared");
    ScratchCheckpointObservation observation{};
    return run_canonical_scratch_checkpoint(
        transaction,
        resources,
        stageName,
        &scratchRequest,
        ScratchCheckpointInvocation::OuterPhaseCheckpoint,
        observation,
        outError);
}

bool command_checkpoint_large_scratch_transition(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const ScratchRequestDescriptor& scratchRequest,
    void* cudaStreamOpaque,
    const char* commandName,
    std::string& outError) {
    const char* stageName =
        trace_or_non_empty(commandName, "command_checkpoint_large_scratch_transition");
    if (!ensure_active_for_command(transaction, outError, stageName)) {
        return false;
    }
    if (!validate_scratch_request_descriptor_for_manager(
            scratchRequest,
            stageName,
            outError)) {
        return false;
    }

    trace_scratch_request_descriptor(transaction, stageName, scratchRequest, "large_transition_declared");
    return run_large_scratch_transition_checkpoint(
        transaction,
        resources,
        stageName,
        scratchRequest,
        cudaStreamOpaque,
        outError);
}

bool command_admit_frame_scratch_overflow(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const ScratchRequestDescriptor& scratchRequest,
    std::string& outError) {
    const char* stageName = "command_admit_frame_scratch_overflow";
    if (!ensure_active_for_command(transaction, outError, stageName)) {
        return false;
    }
    if (!validate_scratch_request_descriptor_for_manager(
            scratchRequest,
            stageName,
            outError)) {
        return false;
    }

    const std::size_t allocationBytes =
        estimate_scratch_request_allocation_bytes(scratchRequest);
    if (allocationBytes == std::numeric_limits<std::size_t>::max()) {
        outError = "frame scratch overflow byte estimation overflow";
        return false;
    }
    return admit_active_frame_vram_growth(
        transaction,
        resources,
        stageName,
        allocationBytes,
        &scratchRequest,
        false,
        ActiveFrameAdmissionHeadroom::DriverFreeOnly,
        outError);
}

bool command_admit_spatial_dir_cached_log_raw_overflow(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const ScratchRequestDescriptor& scratchRequest,
    std::string& outError) {
    const char* stageName = "command_admit_spatial_dir_cached_log_raw_overflow";
    if (!ensure_active_for_command(transaction, outError, stageName)) {
        return false;
    }
    if (!validate_scratch_request_for_family(
            scratchRequest,
            stageName,
            ScratchWorkClass::SpatialDir,
            outError)) {
        return false;
    }
    if (scratchRequest.spatialDirTargetPlaneRoles.cachedLogRawPlanes != 3) {
        outError = "spatial DIR cached log raw overflow requested without Tier2 target roles";
        return false;
    }

    const std::size_t planeBytes =
        plane_bytes_for_extent(scratchRequest.requestedWidth, scratchRequest.requestedHeight);
    if (planeBytes == 0 || planeBytes == std::numeric_limits<std::size_t>::max()) {
        outError = "spatial DIR cached log raw overflow byte estimation overflow";
        return false;
    }
    std::size_t allocationBytes = 0;
    for (int plane = 0; plane < 3; ++plane) {
        std::size_t next = 0;
        if (!add_bytes_checked(allocationBytes, planeBytes, next)) {
            outError = "spatial DIR cached log raw overflow byte estimation overflow";
            return false;
        }
        allocationBytes = next;
    }

    return admit_active_frame_vram_growth(
        transaction,
        resources,
        stageName,
        allocationBytes,
        &scratchRequest,
        false,
        ActiveFrameAdmissionHeadroom::DriverFreeOnly,
        outError);
}

bool command_release_spatial_dir_scratch_stage(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    void* cudaStreamOpaque,
    JuicerCuda::SpatialDirStageReleaseStats& outStats,
    std::string& outError) {
    const char* stageName = "command_release_spatial_dir_scratch_stage";
    outStats = JuicerCuda::SpatialDirStageReleaseStats{};
    if (!ensure_active_for_command(transaction, outError, stageName)) {
        return false;
    }
    const bool ok = JuicerCuda::release_retained_spatial_dir_scratch_stage(
        resources,
        transaction.leaseGeneration,
        cudaStreamOpaque,
        outStats,
        outError);
#if JUICER_DIAGNOSTICS_COMPILED
    if (JTRACE_ENABLED(1)) {
        std::string msg = trace_event_prefix(
            "spatial_dir_stage_release",
            transaction,
            stageName);
        msg += " scratch_source=retained";
        msg += " ok=";
        msg += ok ? "1" : "0";
        msg += " retired_bytes=";
        msg += std::to_string(static_cast<unsigned long long>(outStats.retiredBytes));
        msg += " reclaimed_bytes=";
        msg += std::to_string(static_cast<unsigned long long>(outStats.reclaimedBytes));
        if (!outError.empty()) {
            msg += " error=";
            msg += outError;
        }
        JTRACE("MSADM", msg);
    }
#endif
    return ok;
}

bool command_release_spatial_dir_build_scratch_stage(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    void* cudaStreamOpaque,
    JuicerCuda::SpatialDirBuildScratchReleaseStats& outStats,
    std::string& outError) {
    const char* stageName = "command_release_spatial_dir_build_scratch_stage";
    outStats = JuicerCuda::SpatialDirBuildScratchReleaseStats{};
    if (!ensure_active_for_command(transaction, outError, stageName)) {
        return false;
    }
    const bool ok = JuicerCuda::release_retained_spatial_dir_build_scratch_stage(
        resources,
        transaction.leaseGeneration,
        cudaStreamOpaque,
        outStats,
        outError);
#if JUICER_DIAGNOSTICS_COMPILED
    if (JTRACE_ENABLED(1)) {
        std::string msg = trace_event_prefix(
            "spatial_dir_build_scratch_release",
            transaction,
            stageName);
        msg += " scratch_source=retained";
        msg += " ok=";
        msg += ok ? "1" : "0";
        msg += " pending_scratch_bytes_before=";
        msg += std::to_string(static_cast<unsigned long long>(outStats.pendingScratchBytesBefore));
        msg += " raw_correction_retired_bytes=";
        msg += std::to_string(static_cast<unsigned long long>(outStats.rawCorrectionRetiredBytes));
        msg += " filter_temp_retired_bytes=";
        msg += std::to_string(static_cast<unsigned long long>(outStats.filterTempRetiredBytes));
        msg += " shared_tmp_retired_bytes=";
        msg += std::to_string(static_cast<unsigned long long>(outStats.sharedTmpRetiredBytes));
        msg += " reclaimed_bytes=";
        msg += std::to_string(static_cast<unsigned long long>(outStats.reclaimedBytes));
        if (!outError.empty()) {
            msg += " error=";
            msg += outError;
        }
        JTRACE("MSADM", msg);
    }
#endif
    return ok;
}

bool command_release_spatial_dir_cached_log_raw_stage(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    void* cudaStreamOpaque,
    JuicerCuda::SpatialDirCachedLogRawReleaseStats& outStats,
    std::string& outError) {
    const char* stageName = "command_release_spatial_dir_cached_log_raw_stage";
    outStats = JuicerCuda::SpatialDirCachedLogRawReleaseStats{};
    if (!ensure_active_for_command(transaction, outError, stageName)) {
        return false;
    }
    const bool ok = JuicerCuda::release_retained_spatial_dir_cached_log_raw_stage(
        resources,
        transaction.leaseGeneration,
        cudaStreamOpaque,
        outStats,
        outError);
#if JUICER_DIAGNOSTICS_COMPILED
    if (JTRACE_ENABLED(1)) {
        std::string msg = trace_event_prefix(
            "spatial_dir_cached_log_raw_release",
            transaction,
            stageName);
        msg += " scratch_source=retained";
        msg += " ok=";
        msg += ok ? "1" : "0";
        msg += " pending_scratch_bytes_before=";
        msg += std::to_string(static_cast<unsigned long long>(outStats.pendingScratchBytesBefore));
        msg += " cached_log_raw_retired_bytes=";
        msg += std::to_string(static_cast<unsigned long long>(outStats.cachedLogRawRetiredBytes));
        msg += " reclaimed_bytes=";
        msg += std::to_string(static_cast<unsigned long long>(outStats.reclaimedBytes));
        if (!outError.empty()) {
            msg += " error=";
            msg += outError;
        }
        JTRACE("MSADM", msg);
    }
#endif
    return ok;
}

bool command_ensure_spatial_dir_cached_log_raw_stage(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const ScratchRequestDescriptor& scratchRequest,
    void* cudaStreamOpaque,
    JuicerCuda::SpatialDirCachedLogRawStageStats& outStats,
    std::string& outError) {
    const char* stageName = "command_ensure_spatial_dir_cached_log_raw_stage";
    outStats = JuicerCuda::SpatialDirCachedLogRawStageStats{};
    if (!ensure_active_for_command(transaction, outError, stageName)) {
        return false;
    }
    if (!validate_scratch_request_for_family(
            scratchRequest,
            stageName,
            ScratchWorkClass::SpatialDir,
            outError)) {
        return false;
    }
    if (scratchRequest.spatialDirTargetPlaneRoles.cachedLogRawPlanes != 3) {
        outError = "spatial DIR cached log raw stage requested without Tier2 target roles";
        return false;
    }

    const bool captureMemorySnapshots =
        should_collect_manager_memory_snapshots(transaction.resolvedPressurePolicy);
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    const std::size_t growthBytes = estimate_spatial_dir_cached_log_raw_stage_growth_bytes(
        resources,
        scratchRequest.requestedWidth,
        scratchRequest.requestedHeight);
    if (growthBytes == std::numeric_limits<std::size_t>::max()) {
        outError = "spatial DIR cached log raw stage growth byte estimation overflow";
        return false;
    }

    auto action = [&](std::string& actionError) {
        return JuicerCuda::ensure_retained_spatial_dir_cached_log_raw_stage(
            resources,
            transaction.leaseGeneration,
            scratchRequest,
            cudaStreamOpaque,
            outStats,
            actionError);
    };

    const bool ok = growthBytes == 0
                        ? action(outError)
                        : execute_scratch_growth_command(
                              transaction,
                              resources,
                              stageName,
                              growthBytes,
                              ScratchWorkClass::SpatialDir,
                              scratchRequest,
                              cudaStreamOpaque,
                              captureMemorySnapshots,
                              action,
                              outError);
#if JUICER_DIAGNOSTICS_COMPILED
    if (JTRACE_ENABLED(1)) {
        std::string msg = trace_event_prefix(
            "spatial_dir_cached_log_raw_stage",
            transaction,
            stageName);
        msg += " scratch_source=retained";
        msg += " ok=";
        msg += ok ? "1" : "0";
        msg += " growth_bytes=";
        msg += std::to_string(static_cast<unsigned long long>(growthBytes));
        msg += " pending_scratch_bytes_before=";
        msg += std::to_string(static_cast<unsigned long long>(outStats.pendingScratchBytesBefore));
        msg += " cached_log_raw_allocated_bytes=";
        msg += std::to_string(static_cast<unsigned long long>(outStats.cachedLogRawAllocatedBytes));
        if (!outError.empty()) {
            msg += " error=";
            msg += outError;
        }
        JTRACE("MSADM", msg);
    }
#endif
    return ok;
}

std::size_t post_frame_spatial_dir_build_bytes(
    const ScratchRequestDescriptor& scratchRequest) noexcept {
    if (!scratchRequest.needSpatialDir) {
        return 0;
    }
    const std::size_t planeBytes =
        plane_bytes_for_extent(scratchRequest.requestedWidth, scratchRequest.requestedHeight);
    if (planeBytes == 0 || planeBytes == std::numeric_limits<std::size_t>::max()) {
        return planeBytes;
    }
    const int planes = scratchRequest.spatialDirPlaneRoles.total_float_planes();
    if (planes <= 0) {
        return 0;
    }
    const std::size_t planeCount = static_cast<std::size_t>(planes);
    if (planeBytes > (std::numeric_limits<std::size_t>::max() / planeCount)) {
        return std::numeric_limits<std::size_t>::max();
    }
    return planeBytes * planeCount;
}

std::uint64_t post_frame_effective_reserve_bytes(
    const PressureCheckpoint& checkpoint) noexcept {
    if (checkpoint.input.effectiveReserveBytes > 0) {
        return checkpoint.input.effectiveReserveBytes;
    }
    if (checkpoint.decision.effectiveReserveBytes > 0) {
        return checkpoint.decision.effectiveReserveBytes;
    }
    return checkpoint.input.reserveBytes;
}

std::uint64_t post_frame_driver_free_floor_bytes(
    const PressureCheckpoint& checkpoint) noexcept {
    constexpr std::uint64_t kDriverFreeFloorBytes = 512ull * 1024ull * 1024ull;
    const std::uint64_t reserveBytes = post_frame_effective_reserve_bytes(checkpoint);
    if (reserveBytes > (std::numeric_limits<std::uint64_t>::max() / 2u)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return std::max(kDriverFreeFloorBytes, reserveBytes * 2u);
}

bool post_frame_large_spatial_dir_shed_required(
    const ScratchRequestDescriptor& scratchRequest,
    const PressureCheckpoint& checkpoint,
    std::size_t requestBuildBytes) noexcept {
    if (!scratchRequest.needSpatialDir ||
        !is_large_frame_extent(scratchRequest.requestedWidth, scratchRequest.requestedHeight) ||
        !checkpoint.headroom.sampleSuccess) {
        return false;
    }
    if (requestBuildBytes == std::numeric_limits<std::size_t>::max()) {
        return true;
    }
    const bool driverFreeLow =
        checkpoint.input.driverFreeBytes <= post_frame_driver_free_floor_bytes(checkpoint);
    if (requestBuildBytes == 0) {
        return driverFreeLow;
    }

    const std::uint64_t reserveBytes = post_frame_effective_reserve_bytes(checkpoint);
    std::uint64_t requiredBytes = 0;
    if (!add_u64_checked(
            static_cast<std::uint64_t>(requestBuildBytes),
            reserveBytes,
            requiredBytes)) {
        return true;
    }

    return driverFreeLow || checkpoint.input.effectiveHeadroomBytes < requiredBytes;
}

void trace_post_frame_scratch_shed(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const PressureCheckpoint& checkpoint,
    const ScratchRequestDescriptor& scratchRequest,
    std::size_t requestBuildBytes,
    bool requestLowHeadroom,
    const JuicerCuda::PostFrameScratchShedStats& stats,
    bool ok,
    bool shedAttempted,
    const char* reason,
    const std::string& error) {
#if JUICER_DIAGNOSTICS_COMPILED
    if ((ok && !shedAttempted) ? !JTRACE_ENABLED(2) : !JTRACE_ENABLED(1)) {
        return;
    }
    const bool driverFreeLow =
        checkpoint.headroom.sampleSuccess &&
        checkpoint.input.driverFreeBytes <= post_frame_driver_free_floor_bytes(checkpoint);
    const bool buildBytesZeroAfterStageRelease =
        scratchRequest.needSpatialDir &&
        is_large_frame_extent(scratchRequest.requestedWidth, scratchRequest.requestedHeight) &&
        requestBuildBytes == 0;
    std::string msg =
        trace_event_prefix("post_frame_scratch_shed", transaction, commandName) +
        " ok=" + std::to_string(ok ? 1 : 0) +
        " action=" + std::string(shedAttempted ? "shed" : "keep") +
        " state=" + to_cstr(checkpoint.decision.state) +
        " sample_success=" + std::to_string(checkpoint.headroom.sampleSuccess ? 1 : 0) +
        " request_width=" + std::to_string(scratchRequest.requestedWidth) +
        " request_height=" + std::to_string(scratchRequest.requestedHeight) +
        " request_large_frame=" +
        std::to_string(
            is_large_frame_extent(scratchRequest.requestedWidth, scratchRequest.requestedHeight) ? 1 : 0) +
        " request_spatial_dir_build_bytes=" +
        std::to_string(static_cast<unsigned long long>(requestBuildBytes)) +
        " request_low_headroom=" + std::to_string(requestLowHeadroom ? 1 : 0) +
        " driver_free_low=" + std::to_string(driverFreeLow ? 1 : 0) +
        " build_bytes_zero_after_stage_release=" +
        std::to_string(buildBytesZeroAfterStageRelease ? 1 : 0) +
        " driver_free_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.input.driverFreeBytes)) +
        " effective_headroom_bytes=" +
        std::to_string(static_cast<unsigned long long>(checkpoint.input.effectiveHeadroomBytes)) +
        " headroom_source=" + to_cstr(checkpoint.input.headroomSource) +
        " manager_active_bytes=" +
        std::to_string(static_cast<unsigned long long>(checkpoint.memory.activeBytes)) +
        " retire_pending_bytes=" +
        std::to_string(static_cast<unsigned long long>(checkpoint.memory.retirePendingBytes)) +
        " pending_scratch_bytes_before=" +
        std::to_string(static_cast<unsigned long long>(stats.pendingScratchBytesBefore)) +
        " optics_retired_bytes=" +
        std::to_string(static_cast<unsigned long long>(stats.opticsRetiredBytes)) +
        " spatial_dir_retired_bytes=" +
        std::to_string(static_cast<unsigned long long>(stats.spatialDirRetiredBytes)) +
        " shared_tmp_retired_bytes=" +
        std::to_string(static_cast<unsigned long long>(stats.sharedTmpRetiredBytes)) +
        " reclaimed_bytes=" +
        std::to_string(static_cast<unsigned long long>(stats.reclaimedBytes)) +
        trace_device_context_fields(transaction) +
        trace_resolved_policy_fields(transaction) +
        " reason=" + trace_or_unspecified(reason);
    if (!error.empty()) {
        msg += " error=";
        msg += error;
    }
    JTRACE("MSPFS", msg);
#else
    (void)transaction;
    (void)commandName;
    (void)checkpoint;
    (void)scratchRequest;
    (void)requestBuildBytes;
    (void)requestLowHeadroom;
    (void)stats;
    (void)ok;
    (void)shedAttempted;
    (void)reason;
    (void)error;
#endif
}

bool command_shed_post_frame_scratch(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const ScratchRequestDescriptor& scratchRequest,
    void* cudaStreamOpaque,
    const char* commandName,
    std::string& outError) {
    const char* stageName =
        trace_or_non_empty(commandName, "command_shed_post_frame_scratch");
    JuicerCuda::PostFrameScratchShedStats stats{};
    PressureCheckpoint checkpoint{};
    outError.clear();
    if (!ensure_active_for_command(transaction, outError, stageName)) {
        return false;
    }
    const ScratchRequestDescriptor* activeScratchRequest = nullptr;
    if (!resolve_optional_active_scratch_request(
            scratchRequest,
            stageName,
            activeScratchRequest,
            outError)) {
        return false;
    }
    (void)activeScratchRequest;
    const std::size_t requestBuildBytes =
        post_frame_spatial_dir_build_bytes(scratchRequest);
    bool requestLowHeadroom = false;
    if (!pressure_policy_enabled(transaction.resolvedPressurePolicy)) {
        trace_post_frame_scratch_shed(
            transaction,
            stageName,
            checkpoint,
            scratchRequest,
            requestBuildBytes,
            requestLowHeadroom,
            stats,
            true,
            false,
            "pressure_policy_disabled",
            outError);
        return true;
    }

    checkpoint = evaluate_pressure_checkpoint(transaction, resources, 0, true, stageName);
    requestLowHeadroom = post_frame_large_spatial_dir_shed_required(
        scratchRequest,
        checkpoint,
        requestBuildBytes);

    if (!checkpoint.headroom.sampleSuccess) {
        trace_post_frame_scratch_shed(
            transaction,
            stageName,
            checkpoint,
            scratchRequest,
            requestBuildBytes,
            requestLowHeadroom,
            stats,
            true,
            false,
            "headroom_sample_failed",
            outError);
        return true;
    }
    if (checkpoint.decision.state == PressureState::Normal) {
        if (!requestLowHeadroom) {
            trace_post_frame_scratch_shed(
                transaction,
                stageName,
                checkpoint,
                scratchRequest,
                requestBuildBytes,
                requestLowHeadroom,
                stats,
                true,
                false,
                "pressure_normal",
                outError);
            return true;
        }
    }

    const bool ok = JuicerCuda::shed_retained_scratch_after_frame(
        resources,
        cudaStreamOpaque,
        stats,
        outError);
    publish_manager_memory_snapshot(snapshot_manager_memory(resources));
    const char* reason = nullptr;
    if (!ok) {
        reason = requestLowHeadroom ? "large_frame_low_headroom_shed_failed"
                                    : "pressure_shed_failed";
    } else {
        reason = requestLowHeadroom ? "large_frame_low_headroom" : "pressure_shed";
    }
    trace_post_frame_scratch_shed(
        transaction,
        stageName,
        checkpoint,
        scratchRequest,
        requestBuildBytes,
        requestLowHeadroom,
        stats,
        ok,
        true,
        reason,
        outError);
    return ok;
}

bool command_ensure_print_illuminant_filtered(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    const Print::Runtime& prt,
    const Print::Params& params,
    const ScratchRequestDescriptor& scratchRequest,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_print_illuminant_filtered")) {
        return false;
    }
    const ScratchRequestDescriptor* activeScratchRequest = nullptr;
    if (!resolve_optional_active_scratch_request(
            scratchRequest,
            "command_ensure_print_illuminant_filtered",
            activeScratchRequest,
            outError)) {
        return false;
    }
    const UploadWorkEstimate uploadEstimate =
        estimate_print_illuminant_upload_bytes(resources, ws, prt, params);
    if (command_print_illuminant_filtered_is_cached(resources, ws, prt, params)) {
        return true;
    }
    return execute_upload_immutable_command(
        transaction,
        resources,
        "command_ensure_print_illuminant_filtered",
        uploadEstimate.growthBytes,
        uploadEstimate.reservationBytes,
        true,
        activeScratchRequest,
        "upload reservation request byte estimation overflow (print illuminant)",
        [&](std::string& actionError) {
            return JuicerCuda::ensure_print_illuminant_filtered(
                resources,
                ws,
                prt,
                params,
                cudaStreamOpaque,
                actionError);
        },
        outError);
}

bool command_ensure_optics_scratch(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const ScratchRequestDescriptor& scratchRequest,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_optics_scratch")) {
        return false;
    }
    if (!validate_scratch_request_for_family(
            scratchRequest,
            "command_ensure_optics_scratch",
            ScratchWorkClass::Optics,
            outError)) {
        return false;
    }
    const bool captureMemorySnapshots =
        should_collect_manager_memory_snapshots(transaction.resolvedPressurePolicy);
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    const std::size_t growthBytes = estimate_optics_growth_bytes(
        resources,
        scratchRequest.requestedWidth,
        scratchRequest.requestedHeight,
        scratchRequest.needBlurred,
        scratchRequest.aliasScannerRgbFromSpatialDirFiltered,
        scratchRequest.needAux,
        scratchRequest.needSharedTmp,
        scratchRequest.needGrainFrameUniforms,
        scratchRequest.needGrainLayerWork,
        scratchRequest.needGrainShared,
        scratchRequest.needGateMask);
    if (growthBytes == std::numeric_limits<std::size_t>::max()) {
        outError = "scratch growth byte estimation overflow";
        return false;
    }
    if (growthBytes == 0 && !scratchRequest.aliasScannerRgbFromSpatialDirFiltered) {
        return true;
    }
    if (growthBytes == 0) {
        return JuicerCuda::ensure_optics_scratch(
            resources,
            scratchRequest,
            cudaStreamOpaque,
            outError);
    }
    return execute_scratch_growth_command(
        transaction,
        resources,
        "command_ensure_optics_scratch",
        growthBytes,
        ScratchWorkClass::Optics,
        scratchRequest,
        cudaStreamOpaque,
        captureMemorySnapshots,
        [&](std::string& actionError) {
            return JuicerCuda::ensure_optics_scratch(
                resources,
                scratchRequest,
                cudaStreamOpaque,
                actionError);
        },
        outError);
}

bool command_ensure_spatial_dir_scratch(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const ScratchRequestDescriptor& scratchRequest,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_spatial_dir_scratch")) {
        return false;
    }
    if (!validate_scratch_request_for_family(
            scratchRequest,
            "command_ensure_spatial_dir_scratch",
            ScratchWorkClass::SpatialDir,
            outError)) {
        return false;
    }
    const bool captureMemorySnapshots =
        should_collect_manager_memory_snapshots(transaction.resolvedPressurePolicy);
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    const std::size_t growthBytes = estimate_spatial_dir_growth_bytes(
        resources,
        scratchRequest.requestedWidth,
        scratchRequest.requestedHeight,
        scratchRequest.spatialDirScratchTier,
        scratchRequest.spatialDirPlaneRoles);
    if (growthBytes == std::numeric_limits<std::size_t>::max()) {
        outError = "spatial dir scratch growth byte estimation overflow";
        return false;
    }
    if (growthBytes == 0) {
        return JuicerCuda::ensure_spatial_dir_scratch(
            resources,
            scratchRequest,
            cudaStreamOpaque,
            outError);
    }
    const bool preferredOk = execute_scratch_growth_command(
        transaction,
        resources,
        "command_ensure_spatial_dir_scratch",
        growthBytes,
        ScratchWorkClass::SpatialDir,
        scratchRequest,
        cudaStreamOpaque,
        captureMemorySnapshots,
        [&](std::string& actionError) {
            return JuicerCuda::ensure_spatial_dir_scratch(
                resources,
                scratchRequest,
                cudaStreamOpaque,
                actionError);
        },
        outError);
    return preferredOk;
}

bool command_ensure_spatial_dir_kernel(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    JuicerCuda::Resources::DeviceGaussianKernel& kernel,
    float sigma,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_spatial_dir_kernel")) {
        return false;
    }
    return execute_snapshot_wrapped_command(
        transaction,
        resources,
        [&](std::string& actionError) {
            return JuicerCuda::ensure_spatial_dir_kernel(
                resources,
                kernel,
                sigma,
                cudaStreamOpaque,
                actionError);
        },
        outError);
}

bool command_ensure_gaussian_kernel(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    JuicerCuda::Resources::DeviceGaussianKernel& kernel,
    float sigma,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_gaussian_kernel")) {
        return false;
    }
    return execute_snapshot_wrapped_command(
        transaction,
        resources,
        [&](std::string& actionError) {
            return JuicerCuda::ensure_gaussian_kernel(
                resources,
                kernel,
                sigma,
                cudaStreamOpaque,
                actionError);
        },
        outError);
}

bool command_ensure_halation_kernel(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    JuicerCuda::Resources::DeviceGaussianKernel& kernel,
    float sigma,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_halation_kernel")) {
        return false;
    }
    return execute_snapshot_wrapped_command(
        transaction,
        resources,
        [&](std::string& actionError) {
            return JuicerCuda::ensure_halation_kernel(
                resources,
                kernel,
                sigma,
                cudaStreamOpaque,
                actionError);
        },
        outError);
}


bool error_is_scratch_exhausted(const std::string& error) noexcept {
    return error.rfind(kScratchExhaustedPrefix, 0) == 0 ||
           error.rfind(kReservationDeferredPrefix, 0) == 0;
}
