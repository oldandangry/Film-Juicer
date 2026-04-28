// Cuda/ResourceManager/JuicerCudaResourceManagerCommands.cpp
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
            transaction.transactionId,
            transaction.snapshot.snapshotId,
            transaction.snapshot.traceSchemaVersion,
            rejectionStage);
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

bool resolve_optional_active_scratch_request(
    const ScratchRequestDescriptor& scratchRequest,
    const char* commandName,
    const ScratchRequestDescriptor*& outActiveScratchRequest,
    std::string& outError) {
    outActiveScratchRequest = nullptr;
    if (!scratchRequest.has_any_family()) {
        if (scratchRequest.needBlurred ||
            scratchRequest.needAux ||
            scratchRequest.needGrainTriplet ||
            scratchRequest.needGrainShared ||
            scratchRequest.needGateMask) {
            outError = std::string(trace_or_non_empty(commandName, "scratch_request"))
                + ": invalid scratch request descriptor";
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
        outError = std::string(trace_or_non_empty(commandName, "command"))
            + ": optics scratch request missing needOptics";
        return false;
    }
    if (workClass == ScratchWorkClass::SpatialDir && !scratchRequest.needSpatialDir) {
        outError = std::string(trace_or_non_empty(commandName, "command"))
            + ": spatial DIR scratch request missing needSpatialDir";
        return false;
    }
    return true;
}

struct AutoExposureOwnershipObservation {
    ShadowHistoryKey ownershipKey{};
    bool hadPrevious = false;
    bool metadataHit = false;
};

AutoExposureOwnershipObservation observe_auto_exposure_ownership(
    const SubmissionTransaction& transaction,
    std::uint64_t normalizedKeyHash,
    int meterWidth,
    int meterHeight) {
    AutoExposureOwnershipObservation observation{};
    observation.ownershipKey = ShadowHistoryKey{
        transaction.snapshot.instanceToken.value,
        transaction.snapshot.deviceContextKey};

    AutoExposureOwnershipState& ownershipState = auto_exposure_ownership_state();
    std::lock_guard<std::mutex> lock(ownershipState.mutex);
    auto it = ownershipState.bySubmissionKey.find(observation.ownershipKey);
    if (it == ownershipState.bySubmissionKey.end() || !it->second.valid) {
        return observation;
    }

    observation.hadPrevious = true;
    const AutoExposureOwnershipEntry& previous = it->second;
    observation.metadataHit =
        previous.keySchemaVersion == transaction.snapshot.keySchemaVersion &&
        previous.keyHash == normalizedKeyHash &&
        previous.meterWidth == meterWidth &&
        previous.meterHeight == meterHeight;
    return observation;
}

void publish_auto_exposure_ownership(
    const SubmissionTransaction& transaction,
    const ShadowHistoryKey& ownershipKey,
    std::uint64_t normalizedKeyHash,
    int meterWidth,
    int meterHeight) {
    AutoExposureOwnershipState& ownershipState = auto_exposure_ownership_state();
    std::lock_guard<std::mutex> lock(ownershipState.mutex);
    AutoExposureOwnershipEntry& entry = ownershipState.bySubmissionKey[ownershipKey];
    entry.valid = true;
    entry.keyHash = normalizedKeyHash;
    entry.meterWidth = meterWidth;
    entry.meterHeight = meterHeight;
    entry.keySchemaVersion = transaction.snapshot.keySchemaVersion;
}

void trace_auto_exposure_ownership_event(
    const SubmissionTransaction& transaction,
    const AutoExposureOwnershipObservation& observation,
    std::uint64_t normalizedKeyHash,
    int meterWidth,
    int meterHeight,
    const char* action,
    const char* reason) {
    telemetry_trace_auto_exposure_ownership(
        transaction.transactionId,
        transaction.snapshot.snapshotId,
        transaction.snapshot.traceSchemaVersion,
        "ManagerOnly",
        action,
        observation.metadataHit,
        normalizedKeyHash,
        meterWidth,
        meterHeight,
        observation.hadPrevious,
        reason);
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
    UploadReservationGuard uploadGuard(std::move(uploadClaim));
    TierCircuitAttempt circuitAttempt{};
    std::string circuitError;
    if (!tier_circuit_begin_attempt(
            transaction,
            stageName,
            ResourceTier::Immutable,
            tier_circuit_blocks_admission(ResourceTier::Immutable),
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

bool acquire_graph_builder_reservation(
    const SubmissionTransaction& transaction,
    ResourceManagerState& managerState,
    std::uint64_t requestBytes,
    bool criticalCurrentFrame,
    BuilderReservationClaim& builderClaim,
    ReservationAttemptInfo& builderReservation) {
    telemetry_counter_add(managerState.builderReservationRequests, 1);
    if (!try_acquire_builder_reservation_claim(
            transaction,
            BuilderReservationTier::Graph,
            requestBytes,
            criticalCurrentFrame,
            builderClaim,
            builderReservation)) {
        const ReservationDecision& decision = builderReservation.decision;
        const bool fairnessDeferred =
            (decision.reason && std::string_view(decision.reason) == "fairness_tokens_exhausted");
        if (decision.shouldWait) {
            telemetry_counter_add(managerState.builderReservationDeferred, 1);
            if (fairnessDeferred) {
                telemetry_counter_add(managerState.builderFairnessTokenDeferred, 1);
            }
            trace_builder_reservation_decision(
                transaction,
                "command_launch_base_pipeline_graph",
                BuilderReservationTier::Graph,
                requestBytes,
                builderReservation.bytesInFlight,
                builderReservation.capBytes,
                builderReservation.thresholdBytes,
                decision,
                criticalCurrentFrame,
                builderReservation.instanceToken,
                builderReservation.sharedTokens,
                builderReservation.criticalTokens,
                0,
                "deferred_nonresident");
        } else {
            telemetry_counter_add(managerState.builderReservationDenied, 1);
            trace_builder_reservation_decision(
                transaction,
                "command_launch_base_pipeline_graph",
                BuilderReservationTier::Graph,
                requestBytes,
                builderReservation.bytesInFlight,
                builderReservation.capBytes,
                builderReservation.thresholdBytes,
                decision,
                criticalCurrentFrame,
                builderReservation.instanceToken,
                builderReservation.sharedTokens,
                builderReservation.criticalTokens,
                0,
                "denied_nonresident");
        }
        return false;
    }

    telemetry_counter_add(managerState.builderReservationGranted, 1);
    if (JTRACE_ENABLED(3)) {
        trace_builder_reservation_decision(
            transaction,
            "command_launch_base_pipeline_graph",
            BuilderReservationTier::Graph,
            requestBytes,
            builderReservation.bytesInFlight,
            builderReservation.capBytes,
            builderReservation.thresholdBytes,
            builderReservation.decision,
            criticalCurrentFrame,
            builderReservation.instanceToken,
            builderReservation.sharedTokens,
            builderReservation.criticalTokens,
            0,
            "admitted");
    }
    return true;
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
    UploadReservationGuard uploadGuard(std::move(uploadClaim));

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
    BuilderReservationGuard builderGuard(std::move(builderClaim));

    TierCircuitAttempt circuitAttempt{};
    std::string circuitError;
    if (!tier_circuit_begin_attempt(
            transaction,
            stageName,
            ResourceTier::Lut,
            tier_circuit_blocks_admission(ResourceTier::Lut),
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
    BuilderReservationGuard builderGuard(std::move(builderClaim));

    ScratchPolicyClaim scratchClaim{};
    if (!acquire_scratch_policy_claim_with_wait(
            transaction,
            commandName,
            workClass,
            scratchRequest.requestedWidth,
            scratchRequest.requestedHeight,
            growthBytes,
            true,
            scratchClaim,
            outError)) {
        return false;
    }
    ScratchPolicyGuard scratchGuard(std::move(scratchClaim));

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
            commandName,
            attempts,
            growthBytes,
            0,
            0,
            success,
            "outcome",
            reason);
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
                tier_circuit_blocks_admission(ResourceTier::Scratch),
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
                commandName,
                attempts,
                reclaimedBytes,
                false,
                commands_error_or_cstr(reclaimError, "reap_failed"));
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
            commandName,
            attempts,
            reclaimedBytes,
            true,
            reclaim_retry_reason(reclaimedBytes));
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

void add_estimate_bytes_u64(std::uint64_t bytes, std::uint64_t& total, bool& overflow) noexcept {
    if (overflow) {
        return;
    }
    std::uint64_t next = 0;
    if (!add_u64_checked(total, bytes, next)) {
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
    add_estimate_bytes_u64(bytes, total, overflow);
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
    return Print::kDefaultNeutralFilterHash;
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

bool& commands_select_private_lut_fallback_active(
    JuicerCuda::Resources& resources,
    bool negativeMedium) noexcept {
    if (negativeMedium) {
        return resources.privateLutFallbackNegativeActive;
    }
    return resources.privateLutFallbackPrintActive;
}

std::uint64_t& commands_select_private_lut_fallback_hash(
    JuicerCuda::Resources& resources,
    bool negativeMedium) noexcept {
    if (negativeMedium) {
        return resources.privateLutFallbackNegativeHash;
    }
    return resources.privateLutFallbackPrintHash;
}

void commands_clear_private_lut_fallback_locked(
    JuicerCuda::Resources& resources,
    bool negativeMedium) noexcept {
    commands_select_private_lut_fallback_active(resources, negativeMedium) = false;
    commands_select_private_lut_fallback_hash(resources, negativeMedium) = 0;
}

void commands_sync_private_lut_fallback_locked(
    JuicerCuda::Resources& resources,
    bool negativeMedium) noexcept {
    bool& active = commands_select_private_lut_fallback_active(resources, negativeMedium);
    std::uint64_t& hash = commands_select_private_lut_fallback_hash(resources, negativeMedium);
    const JuicerCuda::Resources::DeviceSpectralLut& lut =
        commands_select_scan_lut_slot(resources, negativeMedium);
    if (!active) {
        hash = 0;
        return;
    }
    if (!lut.log2XYZ || lut.hash == 0 || lut.hash != hash) {
        active = false;
        hash = 0;
    }
}

std::uint64_t commands_elapsed_ms_since(std::uint64_t nowMs, std::uint64_t earlierMs) noexcept {
    if (nowMs > earlierMs) {
        return nowMs - earlierMs;
    }
    return 0;
}

UploadWorkEstimate commands_uncached_upload_work(
    bool cached,
    std::uint64_t growthBytes,
    std::uint64_t reservationBytes) noexcept {
    UploadWorkEstimate estimate{};
    if (!cached) {
        estimate.growthBytes = growthBytes;
        estimate.reservationBytes = reservationBytes;
    }
    return estimate;
}

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
cudaStream_t commands_cuda_stream_or_null(void* cudaStreamOpaque) noexcept {
    if (!cudaStreamOpaque) {
        return nullptr;
    }
    return reinterpret_cast<cudaStream_t>(cudaStreamOpaque);
}
#endif

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
    add_estimate_bytes_u64(bytes, total, overflow);
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
    add_estimate_bytes_u64(bytes, total, overflow);
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
        add_vector_upload_estimate_bytes(t->baseMin, total, overflow);
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
    add_estimate_bytes_u64(bytes, total, overflow);
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
        (!t->hasBaseline || static_cast<int>(t->baseMin.size()) == K);
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

    if (t->hasBaseline && !(dst.tables.baseMin && dst.tables.K == K)) {
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
    std::uint64_t bytes = 0;
    if (!mul_u64_checked(values, static_cast<std::uint64_t>(sizeof(double)), bytes)) {
        return {std::numeric_limits<std::uint64_t>::max(), std::numeric_limits<std::uint64_t>::max()};
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
        cached = dst.log2XYZ && dst.res == res && dst.hash == expectedHash;
        canOverwriteInPlace = dst.log2XYZ && dst.res == res;
    }
    return commands_uncached_upload_work(
        cached,
        canOverwriteInPlace ? 0 : bytes,
        bytes);
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
    commands_sync_private_lut_fallback_locked(resources, negativeMedium);
    const JuicerCuda::Resources::DeviceSpectralLut& dst =
        commands_select_scan_lut_slot(resources, negativeMedium);
    return dst.log2XYZ && dst.res == res && dst.hash == expectedHash;
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

bool command_auto_exposure_buffers_ready(
    JuicerCuda::Resources& resources,
    int meterWidth,
    int meterHeight,
    std::string& outError) {
    if (meterWidth <= 0 || meterHeight <= 0) {
        outError = "auto-exposure meter dimensions invalid";
        return false;
    }

    const int blockX = 16;
    const int blockY = 16;
    const int gridX = (meterWidth + blockX - 1) / blockX;
    const int gridY = (meterHeight + blockY - 1) / blockY;
    const int neededPartials = gridX * gridY;
    if (neededPartials <= 0) {
        outError = "auto-exposure partial count invalid";
        return false;
    }

    std::lock_guard<std::mutex> lock(resources.m);
    if (!validate_resource_owner_locked(resources, outError, true)) {
        return false;
    }

    return resources.autoExposureExposureScale &&
        resources.autoExposureAutoEV &&
        resources.autoExposureValid &&
        resources.autoExposureScratch.maxYBits &&
        resources.autoExposureScratch.histogram &&
        resources.autoExposureScratch.weightsX &&
        resources.autoExposureScratch.weightsY &&
        resources.autoExposureScratch.partialsA &&
        resources.autoExposureScratch.partialsB &&
        resources.autoExposureScratch.weightsXCapacity >= meterWidth &&
        resources.autoExposureScratch.weightsYCapacity >= meterHeight &&
        resources.autoExposureScratch.partialCapacity >= neededPartials;
}

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
struct BaseGraphKey {
    int width = 0;
    int height = 0;
    int nComponents = 0;
    int renderMode = 0;
};

struct BaseGraphEntry {
    BaseGraphKey key{};
    void* graphOpaque = nullptr;
    void* execOpaque = nullptr;
    void* kernelNodeOpaque = nullptr;
    void* kernelFuncOpaque = nullptr;
    unsigned int gridX = 0;
    unsigned int gridY = 0;
    unsigned int gridZ = 0;
    unsigned int blockX = 0;
    unsigned int blockY = 0;
    unsigned int blockZ = 0;
    unsigned int sharedMemBytes = 0;
    std::uint64_t lastUseTick = 0;
    std::uint64_t lastUseMs = 0;
};

struct GraphLargeEntryReadmitState {
    std::uint64_t lastEvictedMs = 0;
    std::uint32_t ghostHits = 0;
};

struct BaseGraphBucketState {
    std::mutex mutex;
    std::uint64_t contextEpoch = 0;
    std::uint64_t useTick = 0;
    std::vector<BaseGraphEntry> entries;
    std::unordered_map<std::uint64_t, std::uint32_t> probationHitsByDigest;
    std::unordered_map<std::uint64_t, GraphLargeEntryReadmitState> largeEntryReadmitByDigest;
    std::uint64_t largeEntryResidentBytes = 0;
    std::uint32_t largeEntryResidentEntries = 0;
};

struct BaseGraphCacheState {
    std::mutex mutex;
    std::unordered_map<DeviceContextKey, std::shared_ptr<BaseGraphBucketState>, DeviceContextKeyHash> byContext;
};

BaseGraphCacheState& base_graph_cache_state() noexcept {
    static BaseGraphCacheState state{};
    return state;
}

using BasePipelineLaunchFn = cudaError_t(*)(const JuicerCuda::PipelineRunParams*, void*);

BasePipelineLaunchFn base_pipeline_launch_fn_for_mode(int renderModeKey) noexcept {
    switch (renderModeKey) {
    case 0:
        return juicer_cuda_negative_pipeline;
    case 1:
        return juicer_cuda_print_pipeline;
    default:
        return nullptr;
    }
}

bool base_graph_key_equal(const BaseGraphKey& a, const BaseGraphKey& b) noexcept {
    return a.width == b.width &&
        a.height == b.height &&
        a.nComponents == b.nComponents &&
        a.renderMode == b.renderMode;
}

std::uint64_t base_graph_key_digest(const BaseGraphKey& key) noexcept {
    std::uint64_t digest = 1469598103934665603ull;
    auto fold = [&digest](std::uint64_t value) noexcept {
        digest ^= value + 0x9e3779b97f4a7c15ull + (digest << 6u) + (digest >> 2u);
    };
    fold(static_cast<std::uint64_t>(std::max(0, key.width)));
    fold(static_cast<std::uint64_t>(std::max(0, key.height)));
    fold(static_cast<std::uint64_t>(std::max(0, key.nComponents)));
    fold(static_cast<std::uint64_t>(std::max(0, key.renderMode)));
    return digest;
}

inline bool base_graph_entry_live(const BaseGraphEntry& entry) noexcept {
    return entry.execOpaque != nullptr &&
        entry.graphOpaque != nullptr &&
        entry.kernelNodeOpaque != nullptr;
}

void add_graph_large_entry_resident_bytes_global(std::uint64_t bytes) noexcept {
    if (bytes == 0) {
        return;
    }
    telemetry_counter_add(global_state().graphLargeEntryResidentBytes, bytes);
}

void sub_graph_large_entry_resident_bytes_global(std::uint64_t bytes) noexcept {
    telemetry_counter_subtract_saturating(global_state().graphLargeEntryResidentBytes, bytes);
}

void publish_graph_large_entry_resident_snapshot_locked(
    BaseGraphBucketState& bucket,
    std::uint64_t residentBytes,
    std::uint32_t residentEntries) noexcept {
    if (residentBytes > bucket.largeEntryResidentBytes) {
        add_graph_large_entry_resident_bytes_global(residentBytes - bucket.largeEntryResidentBytes);
    }
    else if (bucket.largeEntryResidentBytes > residentBytes) {
        sub_graph_large_entry_resident_bytes_global(bucket.largeEntryResidentBytes - residentBytes);
    }
    bucket.largeEntryResidentBytes = residentBytes;
    bucket.largeEntryResidentEntries = residentEntries;
}

std::uint64_t estimate_base_graph_request_bytes(const BaseGraphKey& key) noexcept {
    const std::uint64_t w = static_cast<std::uint64_t>(std::max(0, key.width));
    const std::uint64_t h = static_cast<std::uint64_t>(std::max(0, key.height));
    const std::uint64_t n = static_cast<std::uint64_t>(std::max(0, key.nComponents));
    std::uint64_t pixels = 0;
    if (!mul_u64_checked(w, h, pixels)) {
        return std::numeric_limits<std::uint64_t>::max();
    }

    constexpr std::uint64_t kBaseGraphMetadataBytes = 2ull * 1024ull * 1024ull;
    constexpr std::uint64_t kPerMegapixelMetadataBytes = 64ull * 1024ull;
    constexpr std::uint64_t kPerComponentMetadataBytes = 128ull * 1024ull;
    constexpr std::uint64_t kMegapixelDivisor = 1024ull * 1024ull;

    std::uint64_t bytes = kBaseGraphMetadataBytes;
    std::uint64_t mpBytes = 0;
    if (!mul_u64_checked(pixels / kMegapixelDivisor, kPerMegapixelMetadataBytes, mpBytes)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    std::uint64_t nextBytes = 0;
    if (!add_u64_checked(bytes, mpBytes, nextBytes)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    bytes = nextBytes;

    std::uint64_t componentBytes = 0;
    if (!mul_u64_checked(n, kPerComponentMetadataBytes, componentBytes)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    if (!add_u64_checked(bytes, componentBytes, nextBytes)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return nextBytes;
}

BaseGraphEntry* find_base_graph_entry(
    BaseGraphBucketState& bucket,
    const BaseGraphKey& key) noexcept;

BaseGraphEntry* build_base_graph_entry(
    BaseGraphBucketState& bucket,
    const BaseGraphKey& key,
    BasePipelineLaunchFn launchFn,
    JuicerCuda::PipelineRunParams& run,
    cudaStream_t stream,
    std::uint64_t largeEntryThresholdBytes,
    std::uint32_t keepHotMs,
    std::uint64_t& outKeepHotBypassEvents,
    std::uint64_t& outKeepHotForcedEvictEvents) noexcept;

bool graph_entry_in_keep_hot_window(
    const BaseGraphEntry& entry,
    std::uint32_t keepHotMs,
    std::uint64_t nowMs) noexcept {
    if (keepHotMs == 0 || entry.lastUseMs == 0) {
        return false;
    }
    const std::uint64_t ageMs = commands_elapsed_ms_since(nowMs, entry.lastUseMs);
    return ageMs < static_cast<std::uint64_t>(keepHotMs);
}

void note_large_entry_evicted_for_readmit_locked(
    BaseGraphBucketState& bucket,
    const BaseGraphEntry& entry,
    std::uint64_t thresholdBytes,
    std::uint64_t nowMs) noexcept {
    if (thresholdBytes == 0 || !base_graph_entry_live(entry)) {
        return;
    }
    const std::uint64_t entryBytes = estimate_base_graph_request_bytes(entry.key);
    if (entryBytes < thresholdBytes) {
        return;
    }
    const std::uint64_t digest = base_graph_key_digest(entry.key);
    GraphLargeEntryReadmitState& state = bucket.largeEntryReadmitByDigest[digest];
    state.lastEvictedMs = nowMs;
    state.ghostHits = 0;
}

void destroy_base_graph_entry(BaseGraphEntry& entry) noexcept {
    if (entry.execOpaque) {
        cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(entry.execOpaque));
        entry.execOpaque = nullptr;
    }
    if (entry.graphOpaque) {
        cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(entry.graphOpaque));
        entry.graphOpaque = nullptr;
    }
    entry.kernelNodeOpaque = nullptr;
    entry.kernelFuncOpaque = nullptr;
    entry.gridX = 0;
    entry.gridY = 0;
    entry.gridZ = 0;
    entry.blockX = 0;
    entry.blockY = 0;
    entry.blockZ = 0;
    entry.sharedMemBytes = 0;
    entry.lastUseTick = 0;
    entry.lastUseMs = 0;
}

std::shared_ptr<BaseGraphBucketState> get_or_create_base_graph_bucket(
    const DeviceContextKey& key) noexcept {
    BaseGraphCacheState& state = base_graph_cache_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.byContext.find(key);
    if (it != state.byContext.end() && it->second) {
        return it->second;
    }
    auto bucket = std::make_shared<BaseGraphBucketState>();
    state.byContext[key] = bucket;
    return bucket;
}

void clear_base_graph_bucket(BaseGraphBucketState& bucket) noexcept {
    publish_graph_large_entry_resident_snapshot_locked(bucket, 0, 0);
    for (auto& entry : bucket.entries) {
        destroy_base_graph_entry(entry);
    }
    bucket.entries.clear();
    bucket.probationHitsByDigest.clear();
    bucket.largeEntryReadmitByDigest.clear();
    bucket.useTick = 0;
}

bool evaluate_graph_large_entry_readmit(
    const SubmissionTransaction& transaction,
    BaseGraphBucketState& bucket,
    std::uint64_t keyDigest,
    std::uint64_t requestBytes,
    std::uint64_t graphLargeThresholdBytes,
    bool criticalCurrentFrame,
    const ResourceManagerConfigEffective& cfg,
    ResourceManagerState& managerState,
    std::uint32_t& outObservedProbationHits) {
    outObservedProbationHits = 0;
    auto probationIt = bucket.probationHitsByDigest.find(keyDigest);
    if (probationIt != bucket.probationHitsByDigest.end()) {
        outObservedProbationHits = probationIt->second;
    }

    LargeEntryReadmitDecision readmitDecision{};
    readmitDecision.enabled =
        (cfg.largeEntryReadmitCooldownMs > 0) || (cfg.largeEntryGhostHitsForReadmit > 0);
    readmitDecision.candidate =
        (graphLargeThresholdBytes > 0) && (requestBytes >= graphLargeThresholdBytes);
    readmitDecision.criticalCurrentFrame = criticalCurrentFrame;
    readmitDecision.cooldownMs = cfg.largeEntryReadmitCooldownMs;
    readmitDecision.ghostHitsRequired = cfg.largeEntryGhostHitsForReadmit;
    if (readmitDecision.enabled) {
        readmitDecision.reason = "not_candidate";
    } else {
        readmitDecision.reason = "disabled";
    }
    if (readmitDecision.enabled && readmitDecision.candidate && !criticalCurrentFrame) {
        auto readmitIt = bucket.largeEntryReadmitByDigest.find(keyDigest);
        if (readmitIt != bucket.largeEntryReadmitByDigest.end()) {
            readmitDecision.hadHistory = true;
            GraphLargeEntryReadmitState& readmitState = readmitIt->second;
            const std::uint64_t nowMs = monotonic_time_ms();
            readmitDecision.ageMs = commands_elapsed_ms_since(nowMs, readmitState.lastEvictedMs);
            readmitDecision.inCooldown =
                (cfg.largeEntryReadmitCooldownMs > 0) &&
                (readmitDecision.ageMs < static_cast<std::uint64_t>(cfg.largeEntryReadmitCooldownMs));

            if (readmitDecision.inCooldown) {
                if (readmitState.ghostHits < std::numeric_limits<std::uint32_t>::max()) {
                    ++readmitState.ghostHits;
                }
                readmitDecision.observedGhostHits = readmitState.ghostHits;
                if (cfg.largeEntryGhostHitsForReadmit > 0 &&
                    readmitState.ghostHits >= cfg.largeEntryGhostHitsForReadmit) {
                    readmitDecision.ghostBypass = true;
                    readmitDecision.reason = "ghost_hit_bypass";
                    bucket.largeEntryReadmitByDigest.erase(readmitIt);
                } else {
                    readmitDecision.blocked = true;
                    readmitDecision.reason = "cooldown_blocked";
                }
            } else {
                readmitDecision.reason = "cooldown_expired";
                bucket.largeEntryReadmitByDigest.erase(readmitIt);
            }
        } else {
            readmitDecision.reason = "no_history";
        }
    } else if (readmitDecision.enabled && readmitDecision.candidate && criticalCurrentFrame) {
        readmitDecision.reason = "critical_bypass";
    }

    trace_large_entry_readmit_decision(
        transaction,
        "command_launch_base_pipeline_graph",
        keyDigest,
        requestBytes,
        graphLargeThresholdBytes,
        readmitDecision);
    if (readmitDecision.ghostBypass) {
        telemetry_counter_add(managerState.largeEntryReadmitGhostBypassEvents, 1);
    }
    if (readmitDecision.blocked) {
        telemetry_counter_add(managerState.largeEntryReadmitBlockedEvents, 1);
        return false;
    }
    return true;
}

bool evaluate_graph_cache_admission(
    const SubmissionTransaction& transaction,
    BaseGraphBucketState& bucket,
    std::uint64_t keyDigest,
    std::uint64_t requestBytes,
    std::uint32_t observedProbationHits,
    bool criticalCurrentFrame,
    const ResourceManagerConfigEffective& cfg,
    ResourceManagerState& managerState) {
    CacheAdmissionInput admissionInput{};
    admissionInput.requestBytes = requestBytes;
    admissionInput.cacheTargetBytes = cfg.managerSoftTargetBytes;
    admissionInput.maxCacheableEntryBytes = cfg.maxCacheableEntryBytes;
    admissionInput.maxCacheableEntryPctOfTarget = cfg.maxCacheableEntryPctOfTarget;
    admissionInput.largeEntryProbationThresholdBytes = cfg.largeEntryProbationThresholdBytes;
    admissionInput.largeEntryProbationHitsRequired = cfg.largeEntryProbationHitsRequired;
    admissionInput.observedProbationHits = observedProbationHits;
    admissionInput.criticalCurrentFrame = criticalCurrentFrame;
    const CacheAdmissionDecision admissionDecision = classify_cache_admission(admissionInput);
    const AdmissionChurnSnapshot churnSnapshot = sample_admission_churn_state(
        transaction,
        cfg,
        keyDigest,
        observedProbationHits);
    const std::uint32_t churnProbationHitsRequired = effective_probation_hits_required(
        admissionDecision.probationHitsRequired,
        churnSnapshot);
    const bool churnProbationAllowDurable =
        !admissionDecision.probationApplied ||
        (next_probation_hits(observedProbationHits) >= churnProbationHitsRequired);

    trace_cache_admission_decision(
        transaction,
        "command_launch_base_pipeline_graph",
        admissionDecision,
        churnSnapshot,
        criticalCurrentFrame,
        requestBytes,
        observedProbationHits,
        keyDigest,
        "graph_miss");
    if (admissionDecision.admissionClass == CacheAdmissionClass::TooLargeToCache) {
        telemetry_counter_add(managerState.cacheAdmissionTooLargeEvents, 1);
    }
    if (admissionDecision.probationApplied) {
        const std::uint32_t nextObservedHits = next_probation_hits(observedProbationHits);
        trace_probation_decision(
            transaction,
            "command_launch_base_pipeline_graph",
            keyDigest,
            nextObservedHits,
            churnProbationHitsRequired,
            churnProbationAllowDurable,
            admissionDecision.reason);
        if (churnProbationAllowDurable) {
            telemetry_counter_add(managerState.cacheAdmissionProbationAdmitEvents, 1);
        } else {
            telemetry_counter_add(managerState.cacheAdmissionProbationDeferredEvents, 1);
        }
    }
    if (admissionDecision.reason &&
        std::string_view(admissionDecision.reason).find("critical_override") != std::string_view::npos) {
        telemetry_counter_add(managerState.cacheAdmissionCriticalOverrideEvents, 1);
    }

    const bool allowDurableAdmission =
        commands_allow_durable_admission(admissionDecision, churnProbationAllowDurable);
    if (allowDurableAdmission) {
        return true;
    }

    if (admissionDecision.probationApplied) {
        std::uint32_t& probationHits = bucket.probationHitsByDigest[keyDigest];
        if (probationHits < std::numeric_limits<std::uint32_t>::max()) {
            ++probationHits;
        }
    } else {
        bucket.probationHitsByDigest.erase(keyDigest);
    }
    return false;
}

void clear_graph_admission_state(
    BaseGraphBucketState& bucket,
    std::uint64_t keyDigest) noexcept {
    bucket.probationHitsByDigest.erase(keyDigest);
    bucket.largeEntryReadmitByDigest.erase(keyDigest);
}

template <typename PolicyReplay>
BaseGraphEntry* build_admitted_graph_entry(
    const SubmissionTransaction& transaction,
    BaseGraphBucketState& bucket,
    const BaseGraphKey& key,
    std::uint64_t keyDigest,
    BasePipelineLaunchFn launchFn,
    JuicerCuda::PipelineRunParams& run,
    cudaStream_t stream,
    std::uint64_t graphLargeThresholdBytes,
    std::uint32_t keepHotMs,
    ResourceManagerState& managerState,
    PolicyReplay&& replayPolicy) {
    clear_graph_admission_state(bucket, keyDigest);

    std::uint64_t keepHotBypassEvents = 0;
    std::uint64_t keepHotForcedEvictEvents = 0;
    BaseGraphEntry* found = build_base_graph_entry(
        bucket,
        key,
        launchFn,
        run,
        stream,
        graphLargeThresholdBytes,
        keepHotMs,
        keepHotBypassEvents,
        keepHotForcedEvictEvents);
    if (keepHotBypassEvents > 0) {
        telemetry_counter_add(managerState.keepHotBypassEvents, keepHotBypassEvents);
    }
    if (keepHotForcedEvictEvents > 0) {
        telemetry_counter_add(managerState.keepHotForcedEvictEvents, keepHotForcedEvictEvents);
    }
    if (keepHotBypassEvents > 0 || keepHotForcedEvictEvents > 0) {
        trace_keep_hot_decision(
            transaction,
            "command_launch_base_pipeline_graph",
            keepHotMs,
            keepHotBypassEvents,
            keepHotForcedEvictEvents,
            "base_graph_cap");
    }
    if (!found) {
        return nullptr;
    }

    replayPolicy("post_build");
    return find_base_graph_entry(bucket, key);
}

template <typename PolicyReplay>
const char* launch_selected_graph_entry(
    BaseGraphEntry& entry,
    JuicerCuda::PipelineRunParams& run,
    cudaStream_t stream,
    std::uint64_t useTick,
    bool reusedResidentGraph,
    PolicyReplay&& replayPolicy,
    int& outCudaErrorCode) {
    entry.lastUseTick = useTick;
    entry.lastUseMs = monotonic_time_ms();

    cudaGraphExec_t exec = reinterpret_cast<cudaGraphExec_t>(entry.execOpaque);
    cudaGraphNode_t node = reinterpret_cast<cudaGraphNode_t>(entry.kernelNodeOpaque);
    cudaKernelNodeParams nodeParams{};
    nodeParams.func = entry.kernelFuncOpaque;
    nodeParams.gridDim = dim3(entry.gridX, entry.gridY, entry.gridZ);
    nodeParams.blockDim = dim3(entry.blockX, entry.blockY, entry.blockZ);
    nodeParams.sharedMemBytes = entry.sharedMemBytes;
    void* kernelArgs[] = {&run};
    nodeParams.kernelParams = kernelArgs;
    nodeParams.extra = nullptr;
    cudaError_t setErr = cudaGraphExecKernelNodeSetParams(exec, node, &nodeParams);
    if (setErr != cudaSuccess) {
        destroy_base_graph_entry(entry);
        replayPolicy("kernel_param_update_failed");
        return "kernel_param_update_failed";
    }

    JuicerCuda::LaunchGraphCounters::record_graph_eligible_submission();
    JuicerCuda::LaunchGraphCounters::record_kernel_launch();
    cudaError_t runErr = cudaGraphLaunch(exec, stream);
    if (runErr == cudaSuccess) {
        runErr = cudaGetLastError();
    }
    if (runErr != cudaSuccess) {
        destroy_base_graph_entry(entry);
        replayPolicy("graph_launch_failed");
        return "graph_launch_failed";
    }

    if (reusedResidentGraph) {
        JuicerCuda::LaunchGraphCounters::record_graph_replay_hit();
    }
    outCudaErrorCode = static_cast<int>(cudaSuccess);
    return nullptr;
}

template <typename DirectLaunch>
bool launch_base_pipeline_direct_fallback(
    ResourceManagerState& managerState,
    bool countNonResidentServe,
    DirectLaunch&& directLaunch,
    int& outCudaErrorCode) {
    if (countNonResidentServe) {
        telemetry_counter_add(managerState.graphNonResidentServeEvents, 1);
    }
    outCudaErrorCode = directLaunch();
    return true;
}

template <typename DirectLaunch>
bool cancel_graph_attempt_and_launch_direct_fallback(
    const SubmissionTransaction& transaction,
    TierCircuitAttempt& graphCircuitAttempt,
    bool& graphCircuitAttemptActive,
    const char* cancelReason,
    ResourceManagerState& managerState,
    bool countNonResidentServe,
    DirectLaunch&& directLaunch,
    int& outCudaErrorCode) {
    if (graphCircuitAttemptActive) {
        tier_circuit_cancel_attempt(
            transaction,
            "command_launch_base_pipeline_graph",
            graphCircuitAttempt,
            cancelReason);
        graphCircuitAttemptActive = false;
    }
    return launch_base_pipeline_direct_fallback(
        managerState,
        countNonResidentServe,
        std::forward<DirectLaunch>(directLaunch),
        outCudaErrorCode);
}

template <typename DirectLaunch>
bool fail_graph_attempt_and_launch_direct_fallback(
    const SubmissionTransaction& transaction,
    TierCircuitAttempt& graphCircuitAttempt,
    bool& graphCircuitAttemptActive,
    const char* failureReason,
    ResourceManagerState& managerState,
    bool countNonResidentServe,
    DirectLaunch&& directLaunch,
    int& outCudaErrorCode) {
    if (graphCircuitAttemptActive) {
        tier_circuit_record_outcome(
            transaction,
            "command_launch_base_pipeline_graph",
            graphCircuitAttempt,
            false,
            failureReason);
        graphCircuitAttemptActive = false;
    }
    return launch_base_pipeline_direct_fallback(
        managerState,
        countNonResidentServe,
        std::forward<DirectLaunch>(directLaunch),
        outCudaErrorCode);
}

void complete_graph_attempt_success(
    const SubmissionTransaction& transaction,
    TierCircuitAttempt& graphCircuitAttempt,
    bool& graphCircuitAttemptActive) {
    if (!graphCircuitAttemptActive) {
        return;
    }
    tier_circuit_record_outcome(
        transaction,
        "command_launch_base_pipeline_graph",
        graphCircuitAttempt,
        true,
        "durable_build_success");
    graphCircuitAttemptActive = false;
}

std::uint64_t prepare_base_graph_bucket_for_submission(
    BaseGraphBucketState& bucket,
    std::uint64_t contextEpoch) noexcept {
    std::uint64_t activeEpoch = contextEpoch;
    if (activeEpoch == 0) {
        activeEpoch = 1;
    }
    if (bucket.contextEpoch != activeEpoch) {
        clear_base_graph_bucket(bucket);
        bucket.contextEpoch = activeEpoch;
    }

    bucket.useTick++;
    if (bucket.useTick == 0) {
        bucket.useTick = 1;
    }
    return bucket.useTick;
}

template <typename PolicyReplay>
BaseGraphEntry* resolve_resident_base_graph_entry(
    BaseGraphBucketState& bucket,
    const BaseGraphKey& key,
    PolicyReplay&& replayPolicy,
    bool& outReusedResidentGraph) {
    BaseGraphEntry* found = find_base_graph_entry(bucket, key);
    const bool trivialResidentGraphHit = (found != nullptr) && (bucket.entries.size() == 1);
    if (!trivialResidentGraphHit) {
        replayPolicy("pre_admission");
        found = find_base_graph_entry(bucket, key);
    }
    outReusedResidentGraph = (found != nullptr);
    return found;
}

std::uint64_t estimate_base_graph_bucket_active_bytes_locked(
    const BaseGraphBucketState& bucket) noexcept {
    std::uint64_t totalBytes = 0;
    for (const BaseGraphEntry& entry : bucket.entries) {
        if (!base_graph_entry_live(entry)) {
            continue;
        }
        const std::uint64_t entryBytes = estimate_base_graph_request_bytes(entry.key);
        std::uint64_t nextBytes = 0;
        if (!add_u64_checked(totalBytes, entryBytes, nextBytes)) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        totalBytes = nextBytes;
    }
    return totalBytes;
}

std::uint64_t estimate_base_graph_bucket_large_entry_bytes_locked(
    const BaseGraphBucketState& bucket,
    std::uint64_t thresholdBytes,
    std::uint32_t& outEntries) noexcept {
    outEntries = 0;
    if (thresholdBytes == 0) {
        return 0;
    }

    std::uint64_t totalBytes = 0;
    for (const BaseGraphEntry& entry : bucket.entries) {
        if (!base_graph_entry_live(entry)) {
            continue;
        }
        const std::uint64_t entryBytes = estimate_base_graph_request_bytes(entry.key);
        if (entryBytes < thresholdBytes) {
            continue;
        }
        std::uint64_t nextBytes = 0;
        if (!add_u64_checked(totalBytes, entryBytes, nextBytes)) {
            totalBytes = std::numeric_limits<std::uint64_t>::max();
        }
        else {
            totalBytes = nextBytes;
        }
        if (outEntries < std::numeric_limits<std::uint32_t>::max()) {
            ++outEntries;
        }
    }
    return totalBytes;
}

std::uint64_t trim_graph_large_entry_decay_and_caps_locked(
    BaseGraphBucketState& bucket,
    std::uint64_t thresholdBytes,
    std::uint64_t capBytes,
    std::uint32_t capEntries,
    std::uint32_t keepHotMs,
    std::uint64_t nowMs,
    std::uint64_t& outDecayEvictedEntries,
    std::uint64_t& outCapTrimEvictedEntries,
    std::uint64_t& outKeepHotBypassEvents,
    std::uint64_t& outKeepHotForcedEvictEvents,
    bool& outCapHit,
    std::uint32_t& outResidentEntries) noexcept {
    outDecayEvictedEntries = 0;
    outCapTrimEvictedEntries = 0;
    outKeepHotBypassEvents = 0;
    outKeepHotForcedEvictEvents = 0;
    outCapHit = false;
    outResidentEntries = 0;

    if (thresholdBytes == 0) {
        publish_graph_large_entry_resident_snapshot_locked(bucket, 0, 0);
        return 0;
    }

    std::size_t index = 0;
    while (index < bucket.entries.size()) {
        BaseGraphEntry& entry = bucket.entries[index];
        if (!base_graph_entry_live(entry)) {
            ++index;
            continue;
        }
        const std::uint64_t entryBytes = estimate_base_graph_request_bytes(entry.key);
        if (entryBytes < thresholdBytes) {
            ++index;
            continue;
        }
        const std::uint64_t ageMs = commands_elapsed_ms_since(nowMs, entry.lastUseMs);
        if (ageMs < kGraphLargeEntryDecayMs) {
            ++index;
            continue;
        }
        note_large_entry_evicted_for_readmit_locked(
            bucket,
            entry,
            thresholdBytes,
            nowMs);
        destroy_base_graph_entry(entry);
        bucket.entries.erase(bucket.entries.begin() + static_cast<std::ptrdiff_t>(index));
        ++outDecayEvictedEntries;
    }

    auto recompute_large_snapshot = [&](std::size_t* outOldestIndex,
                                        std::size_t* outOldestNonHotIndex,
                                        bool* outHasKeepHotCandidates) noexcept -> std::uint64_t {
        if (outOldestIndex) {
            *outOldestIndex = std::numeric_limits<std::size_t>::max();
        }
        if (outOldestNonHotIndex) {
            *outOldestNonHotIndex = std::numeric_limits<std::size_t>::max();
        }
        if (outHasKeepHotCandidates) {
            *outHasKeepHotCandidates = false;
        }
        outResidentEntries = 0;
        std::uint64_t residentBytes = 0;
        std::uint64_t oldestUseMs = std::numeric_limits<std::uint64_t>::max();
        std::uint64_t oldestUseTick = std::numeric_limits<std::uint64_t>::max();
        std::uint64_t oldestNonHotUseMs = std::numeric_limits<std::uint64_t>::max();
        std::uint64_t oldestNonHotUseTick = std::numeric_limits<std::uint64_t>::max();

        for (std::size_t i = 0; i < bucket.entries.size(); ++i) {
            const BaseGraphEntry& entry = bucket.entries[i];
            if (!base_graph_entry_live(entry)) {
                continue;
            }
            const std::uint64_t entryBytes = estimate_base_graph_request_bytes(entry.key);
            if (entryBytes < thresholdBytes) {
                continue;
            }

            std::uint64_t nextBytes = 0;
            if (!add_u64_checked(residentBytes, entryBytes, nextBytes)) {
                residentBytes = std::numeric_limits<std::uint64_t>::max();
            }
            else {
                residentBytes = nextBytes;
            }

            if (outResidentEntries < std::numeric_limits<std::uint32_t>::max()) {
                ++outResidentEntries;
            }

            const bool inKeepHotWindow = graph_entry_in_keep_hot_window(entry, keepHotMs, nowMs);
            if (inKeepHotWindow && outHasKeepHotCandidates) {
                *outHasKeepHotCandidates = true;
            }

            const bool older = (entry.lastUseMs < oldestUseMs) ||
                ((entry.lastUseMs == oldestUseMs) && (entry.lastUseTick < oldestUseTick));
            if (older) {
                oldestUseMs = entry.lastUseMs;
                oldestUseTick = entry.lastUseTick;
                if (outOldestIndex) {
                    *outOldestIndex = i;
                }
            }

            if (!inKeepHotWindow) {
                const bool olderNonHot = (entry.lastUseMs < oldestNonHotUseMs) ||
                    ((entry.lastUseMs == oldestNonHotUseMs) && (entry.lastUseTick < oldestNonHotUseTick));
                if (olderNonHot) {
                    oldestNonHotUseMs = entry.lastUseMs;
                    oldestNonHotUseTick = entry.lastUseTick;
                    if (outOldestNonHotIndex) {
                        *outOldestNonHotIndex = i;
                    }
                }
            }
        }
        return residentBytes;
    };

    std::size_t oldestIndex = std::numeric_limits<std::size_t>::max();
    std::size_t oldestNonHotIndex = std::numeric_limits<std::size_t>::max();
    bool hasKeepHotCandidates = false;
    std::uint64_t residentBytes = recompute_large_snapshot(
        &oldestIndex,
        &oldestNonHotIndex,
        &hasKeepHotCandidates);
    const std::uint32_t allowedEntries = std::max<std::uint32_t>(1u, capEntries);
    while (oldestIndex != std::numeric_limits<std::size_t>::max() &&
           (outResidentEntries > allowedEntries || residentBytes > capBytes)) {
        outCapHit = true;
        std::size_t victimIndex = oldestIndex;
        if (keepHotMs > 0 && hasKeepHotCandidates) {
            if (oldestNonHotIndex != std::numeric_limits<std::size_t>::max()) {
                victimIndex = oldestNonHotIndex;
                if (victimIndex != oldestIndex) {
                    ++outKeepHotBypassEvents;
                }
            }
            else {
                ++outKeepHotForcedEvictEvents;
            }
        }
        BaseGraphEntry& victim = bucket.entries[victimIndex];
        note_large_entry_evicted_for_readmit_locked(
            bucket,
            victim,
            thresholdBytes,
            nowMs);
        destroy_base_graph_entry(victim);
        bucket.entries.erase(bucket.entries.begin() + static_cast<std::ptrdiff_t>(victimIndex));
        ++outCapTrimEvictedEntries;
        residentBytes = recompute_large_snapshot(
            &oldestIndex,
            &oldestNonHotIndex,
            &hasKeepHotCandidates);
    }

    publish_graph_large_entry_resident_snapshot_locked(bucket, residentBytes, outResidentEntries);
    return residentBytes;
}

void replay_graph_large_entry_policy(
    const SubmissionTransaction& transaction,
    BaseGraphBucketState& bucket,
    std::uint64_t graphLargeThresholdBytes,
    std::uint64_t graphLargeCapBytes,
    std::uint32_t graphLargeCapEntries,
    std::uint32_t keepHotMs,
    ResourceManagerState& managerState,
    const char* reason) {
    std::uint64_t decayEvictedEntries = 0;
    std::uint64_t capTrimEvictedEntries = 0;
    std::uint64_t keepHotBypassEvents = 0;
    std::uint64_t keepHotForcedEvictEvents = 0;
    bool capHit = false;
    std::uint32_t residentEntries = 0;
    const std::uint64_t residentBytes = trim_graph_large_entry_decay_and_caps_locked(
        bucket,
        graphLargeThresholdBytes,
        graphLargeCapBytes,
        graphLargeCapEntries,
        keepHotMs,
        monotonic_time_ms(),
        decayEvictedEntries,
        capTrimEvictedEntries,
        keepHotBypassEvents,
        keepHotForcedEvictEvents,
        capHit,
        residentEntries);
    if (decayEvictedEntries > 0) {
        telemetry_counter_add(managerState.graphLargeEntryDecayEvents, decayEvictedEntries);
    }
    if (capTrimEvictedEntries > 0) {
        telemetry_counter_add(managerState.graphLargeEntryTrimEvents, capTrimEvictedEntries);
    }
    if (capHit) {
        telemetry_counter_add(managerState.graphLargeEntryCapHits, 1);
    }
    if (keepHotBypassEvents > 0) {
        telemetry_counter_add(managerState.keepHotBypassEvents, keepHotBypassEvents);
    }
    if (keepHotForcedEvictEvents > 0) {
        telemetry_counter_add(managerState.keepHotForcedEvictEvents, keepHotForcedEvictEvents);
    }
    if (keepHotBypassEvents > 0 || keepHotForcedEvictEvents > 0) {
        trace_keep_hot_decision(
            transaction,
            "command_launch_base_pipeline_graph",
            keepHotMs,
            keepHotBypassEvents,
            keepHotForcedEvictEvents,
            trace_or(reason, "policy"));
    }
    if (decayEvictedEntries > 0 || capTrimEvictedEntries > 0 || capHit) {
        trace_graph_large_entry_quarantine(
            transaction,
            "command_launch_base_pipeline_graph",
            graphLargeThresholdBytes,
            graphLargeCapBytes,
            graphLargeCapEntries,
            residentBytes,
            residentEntries,
            decayEvictedEntries,
            capTrimEvictedEntries,
            capHit,
            trace_or(reason, "policy"));
    }
}

std::uint64_t estimate_graph_cache_active_bytes_for_context(const DeviceContextKey& key) noexcept {
    BaseGraphCacheState& state = base_graph_cache_state();
    std::shared_ptr<BaseGraphBucketState> bucket;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        auto it = state.byContext.find(key);
        if (it == state.byContext.end()) {
            return 0;
        }
        bucket = it->second;
    }
    if (!bucket) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(bucket->mutex);
    return estimate_base_graph_bucket_active_bytes_locked(*bucket);
}

std::uint64_t evict_noncritical_graph_entries_for_context(const DeviceContextKey& key) noexcept {
    BaseGraphCacheState& state = base_graph_cache_state();
    std::shared_ptr<BaseGraphBucketState> bucket;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        auto it = state.byContext.find(key);
        if (it == state.byContext.end()) {
            return 0;
        }
        bucket = it->second;
    }
    if (!bucket) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(bucket->mutex);
    const std::uint64_t evictedEntries = static_cast<std::uint64_t>(bucket->entries.size());
    if (evictedEntries == 0) {
        return 0;
    }
    clear_base_graph_bucket(*bucket);
    return evictedEntries;
}

void retire_base_graph_cache_for_context(const DeviceContextKey& key) noexcept {
    BaseGraphCacheState& state = base_graph_cache_state();
    std::shared_ptr<BaseGraphBucketState> bucket;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        auto it = state.byContext.find(key);
        if (it == state.byContext.end()) {
            return;
        }
        bucket = it->second;
        state.byContext.erase(it);
    }
    if (bucket) {
        std::lock_guard<std::mutex> lock(bucket->mutex);
        clear_base_graph_bucket(*bucket);
    }
}

BaseGraphEntry* find_base_graph_entry(
    BaseGraphBucketState& bucket,
    const BaseGraphKey& key) noexcept {
    for (auto& entry : bucket.entries) {
        if (base_graph_key_equal(entry.key, key) && base_graph_entry_live(entry)) {
            return &entry;
        }
    }
    return nullptr;
}

BaseGraphEntry* build_base_graph_entry(
    BaseGraphBucketState& bucket,
    const BaseGraphKey& key,
    BasePipelineLaunchFn launchFn,
    JuicerCuda::PipelineRunParams& run,
    cudaStream_t stream,
    std::uint64_t largeEntryThresholdBytes,
    std::uint32_t keepHotMs,
    std::uint64_t& outKeepHotBypassEvents,
    std::uint64_t& outKeepHotForcedEvictEvents) noexcept {
    outKeepHotBypassEvents = 0;
    outKeepHotForcedEvictEvents = 0;
    if (!launchFn) {
        return nullptr;
    }

    constexpr std::size_t kBaseGraphCap = 4;
    if (bucket.entries.size() >= kBaseGraphCap && !bucket.entries.empty()) {
        const std::uint64_t nowMs = monotonic_time_ms();
        std::size_t oldestAny = 0;
        std::uint64_t bestTick = bucket.entries[0].lastUseTick;
        for (std::size_t i = 1; i < bucket.entries.size(); ++i) {
            if (bucket.entries[i].lastUseTick < bestTick) {
                bestTick = bucket.entries[i].lastUseTick;
                oldestAny = i;
            }
        }

        std::size_t victim = oldestAny;
        if (keepHotMs > 0) {
            std::size_t oldestNonHot = std::numeric_limits<std::size_t>::max();
            std::uint64_t oldestNonHotTick = std::numeric_limits<std::uint64_t>::max();
            bool hasKeepHotCandidates = false;
            for (std::size_t i = 0; i < bucket.entries.size(); ++i) {
                const BaseGraphEntry& entry = bucket.entries[i];
                const bool inKeepHotWindow = graph_entry_in_keep_hot_window(entry, keepHotMs, nowMs);
                if (inKeepHotWindow) {
                    hasKeepHotCandidates = true;
                    continue;
                }
                if (oldestNonHot == std::numeric_limits<std::size_t>::max() ||
                    entry.lastUseTick < oldestNonHotTick) {
                    oldestNonHotTick = entry.lastUseTick;
                    oldestNonHot = i;
                }
            }

            if (hasKeepHotCandidates) {
                if (oldestNonHot != std::numeric_limits<std::size_t>::max()) {
                    victim = oldestNonHot;
                    if (victim != oldestAny) {
                        ++outKeepHotBypassEvents;
                    }
                }
                else {
                    ++outKeepHotForcedEvictEvents;
                }
            }
        }

        note_large_entry_evicted_for_readmit_locked(
            bucket,
            bucket.entries[victim],
            largeEntryThresholdBytes,
            nowMs);
        destroy_base_graph_entry(bucket.entries[victim]);
        bucket.entries.erase(bucket.entries.begin() + static_cast<std::ptrdiff_t>(victim));
    }

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaGraphNode_t kernelNode = nullptr;

    cudaError_t capErr = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    if (capErr != cudaSuccess) {
        return nullptr;
    }

    cudaError_t launchErr = launchFn(&run, reinterpret_cast<void*>(stream));
    if (launchErr != cudaSuccess) {
        cudaGraph_t abortGraph = nullptr;
        cudaStreamEndCapture(stream, &abortGraph);
        if (abortGraph) {
            cudaGraphDestroy(abortGraph);
        }
        return nullptr;
    }

    capErr = cudaStreamEndCapture(stream, &graph);
    if (capErr != cudaSuccess || !graph) {
        if (graph) {
            cudaGraphDestroy(graph);
        }
        return nullptr;
    }

    cudaError_t instErr = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    if (instErr != cudaSuccess || !exec) {
        cudaGraphDestroy(graph);
        return nullptr;
    }

    std::size_t nodeCount = 0;
    cudaError_t nodeErr = cudaGraphGetNodes(graph, nullptr, &nodeCount);
    if (nodeErr == cudaSuccess && nodeCount > 0) {
        std::vector<cudaGraphNode_t> nodes(nodeCount);
        nodeErr = cudaGraphGetNodes(graph, nodes.data(), &nodeCount);
        if (nodeErr == cudaSuccess) {
            for (cudaGraphNode_t node : nodes) {
                cudaGraphNodeType nodeType = cudaGraphNodeTypeEmpty;
                if (cudaGraphNodeGetType(node, &nodeType) == cudaSuccess && nodeType == cudaGraphNodeTypeKernel) {
                    kernelNode = node;
                    break;
                }
            }
        }
    }

    if (!kernelNode) {
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        return nullptr;
    }

    cudaKernelNodeParams baseParams{};
    if (cudaGraphKernelNodeGetParams(kernelNode, &baseParams) != cudaSuccess || !baseParams.func) {
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        return nullptr;
    }

    BaseGraphEntry entry{};
    entry.key = key;
    entry.graphOpaque = reinterpret_cast<void*>(graph);
    entry.execOpaque = reinterpret_cast<void*>(exec);
    entry.kernelNodeOpaque = reinterpret_cast<void*>(kernelNode);
    entry.kernelFuncOpaque = baseParams.func;
    entry.gridX = baseParams.gridDim.x;
    entry.gridY = baseParams.gridDim.y;
    entry.gridZ = baseParams.gridDim.z;
    entry.blockX = baseParams.blockDim.x;
    entry.blockY = baseParams.blockDim.y;
    entry.blockZ = baseParams.blockDim.z;
    entry.sharedMemBytes = baseParams.sharedMemBytes;
    entry.lastUseTick = bucket.useTick;
    entry.lastUseMs = monotonic_time_ms();
    bucket.entries.push_back(entry);
    return &bucket.entries.back();
}
#else
std::uint64_t estimate_graph_cache_active_bytes_for_context(const DeviceContextKey&) noexcept {
    return 0;
}

std::uint64_t evict_noncritical_graph_entries_for_context(const DeviceContextKey&) noexcept {
    return 0;
}
#endif


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
        JuicerCuda::purge_shared_gaussian_kernels_for_context(
            key.deviceId,
            key.contextOpaque);
        JuicerCuda::purge_pinned_upload_staging_for_context(
            key.deviceId,
            key.contextOpaque);
        retire_base_graph_cache_for_context(key);
#endif
        tier_circuit_retire_context(key);
        pressure_policy_retire_context(key);
        scratch_normalization_retire_context(key);
        admission_churn_retire_context(key);
        optional_heuristic_trace_retire_context(key);
        allocator_backend_retire_context(key);
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
const char* scan_lut_medium_name(bool negativeMedium) noexcept {
    return bool_reason(negativeMedium, "negative", "print");
}

bool compute_expected_scan_lut_hash(
    const WorkingState& ws,
    bool negativeMedium,
    std::uint64_t& outExpectedHash) noexcept {
    outExpectedHash = 0;
    const Scanner::ScannerMediumRuntime* medium = &ws.printMediumRuntime;
    const Scanner::ScannerStaticKey* staticKey = &ws.printStaticKey;
    Scanner::ScannerMedium expectedMedium = Scanner::ScannerMedium::Print;
    if (negativeMedium) {
        medium = &ws.negativeMediumRuntime;
        staticKey = &ws.negativeStaticKey;
        expectedMedium = Scanner::ScannerMedium::Negative;
    }

    if (!medium->tables || medium->tables->K <= 0) {
        return false;
    }
    if (medium->medium != expectedMedium || staticKey->medium != expectedMedium) {
        return false;
    }
    if (medium->tables->tablesHash == 0 || medium->range.digest == 0) {
        return false;
    }
    if (staticKey->tablesHash != medium->tables->tablesHash ||
        staticKey->densityRangeHash != medium->range.digest) {
        return false;
    }
    const std::uint32_t res =
        ResourceManager::normalize_scan_lut_resolution(staticKey->lutResolution);
    outExpectedHash = ResourceManager::make_scan_lut_key_digest(
        static_cast<std::uint32_t>(medium->medium),
        medium->tables->tablesHash,
        medium->range.digest,
        res);
    return outExpectedHash != 0;
}

struct PrivateLutFallbackDecision {
    bool allowed = false;
    bool alreadyActive = false;
    std::uint32_t activeCount = 0;
    std::uint32_t perMediumCap = 0;
    std::uint32_t perInstanceCap = 0;
    const char* reason = "unspecified";
};

PrivateLutFallbackDecision evaluate_private_lut_fallback(
    JuicerCuda::Resources& resources,
    bool negativeMedium,
    std::uint64_t expectedHash,
    const ResourceManagerConfigEffective& cfg) {
    PrivateLutFallbackDecision decision{};
    decision.perMediumCap = cfg.privateLutFallbackPerMediumCap;
    decision.perInstanceCap = cfg.privateLutFallbackPerInstanceCap;

    if (expectedHash == 0) {
        decision.reason = "invalid_expected_hash";
        return decision;
    }

    std::lock_guard<std::mutex> lock(resources.m);
    commands_sync_private_lut_fallback_locked(resources, true);
    commands_sync_private_lut_fallback_locked(resources, false);

    decision.activeCount =
        bool_u32(resources.privateLutFallbackNegativeActive) +
        bool_u32(resources.privateLutFallbackPrintActive);

    bool* slotActive = &resources.privateLutFallbackPrintActive;
    std::uint64_t* slotHash = &resources.privateLutFallbackPrintHash;
    const JuicerCuda::Resources::DeviceSpectralLut* slotLut = &resources.scanPrintLut;
    if (negativeMedium) {
        slotActive = &resources.privateLutFallbackNegativeActive;
        slotHash = &resources.privateLutFallbackNegativeHash;
        slotLut = &resources.scanNegativeLut;
    }

    if (*slotActive && slotLut->log2XYZ && slotLut->hash == expectedHash && *slotHash == expectedHash) {
        decision.allowed = true;
        decision.alreadyActive = true;
        decision.reason = "already_active";
        return decision;
    }

    if (decision.perMediumCap == 0 || decision.perInstanceCap == 0) {
        decision.reason = "disabled";
        return decision;
    }
    if (decision.perMediumCap < 1u) {
        decision.reason = "per_medium_cap_zero";
        return decision;
    }
    if (!*slotActive && decision.activeCount >= decision.perInstanceCap) {
        decision.reason = "per_instance_cap_reached";
        return decision;
    }

    decision.allowed = true;
    if (*slotActive) {
        decision.reason = "slot_reuse";
    }
    else {
        decision.reason = "admit_new";
    }
    return decision;
}

void mark_private_lut_fallback_active(
    JuicerCuda::Resources& resources,
    bool negativeMedium,
    std::uint64_t expectedHash) {
    if (expectedHash == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(resources.m);
    if (negativeMedium) {
        resources.privateLutFallbackNegativeActive = true;
        resources.privateLutFallbackNegativeHash = expectedHash;
    }
    else {
        resources.privateLutFallbackPrintActive = true;
        resources.privateLutFallbackPrintHash = expectedHash;
    }
}

void trace_private_lut_fallback(
    const char* stage,
    const char* eventName,
    bool negativeMedium,
    const PrivateLutFallbackDecision& decision,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }
    std::string msg =
        std::string("stage=") + trace_or_unknown(stage) +
        " event=" + trace_or_unknown(eventName) +
        " medium=" + scan_lut_medium_name(negativeMedium) +
        " allowed=" + std::to_string(bool_u32(decision.allowed)) +
        " already_active=" + std::to_string(bool_u32(decision.alreadyActive)) +
        " active_count=" + std::to_string(decision.activeCount) +
        " per_medium_cap=" + std::to_string(decision.perMediumCap) +
        " per_instance_cap=" + std::to_string(decision.perInstanceCap) +
        " reason=" + trace_or(reason, decision.reason) +
        trace_reason_class_field_if_known("reason_class", trace_or(reason, decision.reason));
    JTRACE("MSLUT", msg);
}

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
}

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
        : "command_ensure_current_medium_uploaded_print";
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
    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    auto try_private_fallback = [&](const char* triggerReason) -> bool {
        if (!criticalRequest) {
            return false;
        }
        trace_pressure_gate_reader(
            transaction,
            stageName,
            PressureLane::Upload,
            saturating_u64_to_size_t(uploadEstimate.growthBytes),
            criticalRequest,
            activeScratchRequest,
            "private_fallback",
            trace_or(triggerReason, "private_fallback"));

        std::uint64_t expectedHash = 0;
        if (!compute_expected_scan_lut_hash(ws, negativeMedium, expectedHash)) {
            PrivateLutFallbackDecision invalidDecision{};
            invalidDecision.reason = "invalid_scan_lut_key";
            trace_private_lut_fallback(
                stageName,
                "private_fallback_denied",
                negativeMedium,
                invalidDecision,
                trace_or(triggerReason, "invalid_scan_lut_key"));
            return false;
        }

        const PrivateLutFallbackDecision decision =
            evaluate_private_lut_fallback(resources, negativeMedium, expectedHash, cfg);
        if (!decision.allowed) {
            trace_private_lut_fallback(
                stageName,
                "private_fallback_denied",
                negativeMedium,
                decision,
                trace_or(triggerReason, decision.reason));
            return false;
        }

        trace_private_lut_fallback(
            stageName,
            "private_fallback_admit",
            negativeMedium,
            decision,
            trace_or(triggerReason, decision.reason));

        std::string fallbackError;
        const bool fallbackOk = JuicerCuda::ensure_scan_lut(
            resources,
            ws,
            negativeMedium,
            cudaStreamOpaque,
            fallbackError);
        if (!fallbackOk) {
            trace_private_lut_fallback(
                stageName,
                "private_fallback_failed",
                negativeMedium,
                decision,
                commands_error_or_cstr(fallbackError, "ensure_failed"));
            if (!fallbackError.empty()) {
                if (!outError.empty()) {
                    outError += " | ";
                }
                outError += std::string(kPrivateLutFallbackFailedPrefix) + " " + fallbackError;
            }
            return false;
        }

        mark_private_lut_fallback_active(resources, negativeMedium, expectedHash);
        trace_private_lut_fallback(
            stageName,
            "private_fallback_served",
            negativeMedium,
            decision,
            trace_or(triggerReason, "served"));
        maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
        return true;
    };
    const bool ok = execute_upload_lut_command(
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
            return try_private_fallback(triggerReason);
        },
        outError);
    if (ok) {
        std::lock_guard<std::mutex> lock(resources.m);
        commands_clear_private_lut_fallback_locked(resources, negativeMedium);
    }
    return ok;
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
        scratchRequest.needAux,
        scratchRequest.needGrainTriplet,
        scratchRequest.needGrainShared,
        scratchRequest.needGateMask);
    if (growthBytes == std::numeric_limits<std::size_t>::max()) {
        outError = "scratch growth byte estimation overflow";
        return false;
    }
    if (growthBytes == 0) {
        return true;
    }
    return execute_scratch_growth_command(
        transaction,
        resources,
        "command_ensure_optics_scratch",
        growthBytes,
        ScratchWorkClass::Optics,
        scratchRequest,
        captureMemorySnapshots,
        [&](std::string& actionError) {
            return JuicerCuda::ensure_optics_scratch(
                resources,
                scratchRequest.requestedWidth,
                scratchRequest.requestedHeight,
                scratchRequest.needBlurred,
                scratchRequest.needAux,
                scratchRequest.needGrainTriplet,
                scratchRequest.needGrainShared,
                scratchRequest.needGateMask,
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
        scratchRequest.requestedHeight);
    if (growthBytes == std::numeric_limits<std::size_t>::max()) {
        outError = "spatial dir scratch growth byte estimation overflow";
        return false;
    }
    if (growthBytes == 0) {
        return true;
    }
    return execute_scratch_growth_command(
        transaction,
        resources,
        "command_ensure_spatial_dir_scratch",
        growthBytes,
        ScratchWorkClass::SpatialDir,
        scratchRequest,
        captureMemorySnapshots,
        [&](std::string& actionError) {
            return JuicerCuda::ensure_spatial_dir_scratch(
                resources,
                scratchRequest.requestedWidth,
                scratchRequest.requestedHeight,
                cudaStreamOpaque,
                actionError);
        },
        outError);
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

bool command_ensure_auto_exposure_buffers(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    int meterWidth,
    int meterHeight,
    std::uint64_t autoExposureKeyHash,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_auto_exposure_buffers")) {
        return false;
    }
    const bool captureMemorySnapshots =
        should_collect_manager_memory_snapshots(transaction.resolvedPressurePolicy);
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    const std::uint64_t normalizedKeyHash = normalize_key_u64(autoExposureKeyHash);
    const AutoExposureOwnershipObservation ownershipObservation =
        observe_auto_exposure_ownership(transaction, normalizedKeyHash, meterWidth, meterHeight);
    AutoExposureOwnershipObservation failedOwnershipObservation = ownershipObservation;
    failedOwnershipObservation.metadataHit = false;

    trace_auto_exposure_ownership_event(
        transaction,
        ownershipObservation,
        normalizedKeyHash,
        meterWidth,
        meterHeight,
        "acquire",
        bool_reason(ownershipObservation.metadataHit, "metadata_hit", "metadata_miss"));

    if (command_auto_exposure_buffers_ready(resources, meterWidth, meterHeight, outError)) {
        publish_auto_exposure_ownership(
            transaction,
            ownershipObservation.ownershipKey,
            normalizedKeyHash,
            meterWidth,
            meterHeight);
        trace_auto_exposure_ownership_event(
            transaction,
            ownershipObservation,
            normalizedKeyHash,
            meterWidth,
            meterHeight,
            "publish",
            bool_reason(ownershipObservation.metadataHit, "reuse", "refresh"));
        return true;
    }
    if (!outError.empty()) {
        maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
        trace_auto_exposure_ownership_event(
            transaction,
            failedOwnershipObservation,
            normalizedKeyHash,
            meterWidth,
            meterHeight,
            "ensure_fail",
            commands_error_or_cstr(outError, "ensure_failed"));
        return false;
    }
    outError.clear();

    if (!JuicerCuda::ensure_auto_exposure_buffers(resources, meterWidth, meterHeight, cudaStreamOpaque, outError)) {
        maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
        trace_auto_exposure_ownership_event(
            transaction,
            failedOwnershipObservation,
            normalizedKeyHash,
            meterWidth,
            meterHeight,
            "ensure_fail",
            commands_error_or_cstr(outError, "ensure_failed"));
        return false;
    }
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);

    publish_auto_exposure_ownership(
        transaction,
        ownershipObservation.ownershipKey,
        normalizedKeyHash,
        meterWidth,
        meterHeight);
    trace_auto_exposure_ownership_event(
        transaction,
        ownershipObservation,
        normalizedKeyHash,
        meterWidth,
        meterHeight,
        "publish",
        bool_reason(ownershipObservation.metadataHit, "reuse", "refresh"));
    return true;
}

bool command_launch_base_pipeline_graph(
    SubmissionTransaction& transaction,
    JuicerCuda::PipelineRunParams& run,
    int renderModeKey,
    void* cudaStreamOpaque,
    int& outCudaErrorCode,
    std::string& outError) {
    outError.clear();

    if (!ensure_active_for_command(transaction, outError, "command_launch_base_pipeline_graph")) {
        return false;
    }

#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
    (void)run;
    (void)renderModeKey;
    (void)cudaStreamOpaque;
    outCudaErrorCode = 0;
    outError = "CUDA is not enabled";
    return false;
#else
    outCudaErrorCode = static_cast<int>(cudaErrorUnknown);
    BasePipelineLaunchFn launchFn = base_pipeline_launch_fn_for_mode(renderModeKey);
    if (!launchFn) {
        outCudaErrorCode = static_cast<int>(cudaErrorInvalidValue);
        return true;
    }

    const cudaStream_t stream = commands_cuda_stream_or_null(cudaStreamOpaque);
    auto launch_base_pipeline_direct = [&]() -> int {
        JuicerCuda::LaunchGraphCounters::record_kernel_launch();
        return static_cast<int>(launchFn(&run, reinterpret_cast<void*>(stream)));
    };

    BaseGraphKey key{};
    key.width = run.width;
    key.height = run.height;
    key.nComponents = run.nComponents;
    key.renderMode = renderModeKey;
    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    constexpr bool kGraphAdmissionCriticalCurrentFrame = false;
    ResourceManagerState& managerState = global_state();
    const std::uint64_t graphLargeThresholdBytes = graph_large_entry_threshold_bytes(cfg);
    const std::uint64_t graphLargeCapBytes = graph_large_entry_quarantine_cap_bytes(cfg);
    const std::uint32_t graphLargeCapEntries = graph_large_entry_quarantine_cap_entries(cfg);

    std::shared_ptr<BaseGraphBucketState> bucketPtr =
        get_or_create_base_graph_bucket(transaction.snapshot.deviceContextKey);
    if (!bucketPtr) {
        outCudaErrorCode = static_cast<int>(cudaErrorUnknown);
        outError = "base graph bucket unavailable";
        return false;
    }
    std::lock_guard<std::mutex> bucketLock(bucketPtr->mutex);
    BaseGraphBucketState& bucket = *bucketPtr;
    const std::uint64_t useTick = prepare_base_graph_bucket_for_submission(
        bucket,
        transaction.snapshot.contextEpoch);
    auto applyGraphLargeEntryPolicy = [&](const char* reason) {
        replay_graph_large_entry_policy(
            transaction,
            bucket,
            graphLargeThresholdBytes,
            graphLargeCapBytes,
            graphLargeCapEntries,
            cfg.keepHotMs,
            managerState,
            reason);
    };
    bool reusedResidentGraph = false;
    BaseGraphEntry* found = resolve_resident_base_graph_entry(
        bucket,
        key,
        applyGraphLargeEntryPolicy,
        reusedResidentGraph);
    const std::uint64_t keyDigest = base_graph_key_digest(key);
    TierCircuitAttempt graphCircuitAttempt{};
    bool graphCircuitAttemptActive = false;
    auto launch_graph_direct = [&](bool countNonResidentServe) -> bool {
        return launch_base_pipeline_direct_fallback(
            managerState,
            countNonResidentServe,
            launch_base_pipeline_direct,
            outCudaErrorCode);
    };
    auto cancel_graph_attempt_and_launch_direct =
        [&](const char* cancelReason, bool countNonResidentServe) -> bool {
        return cancel_graph_attempt_and_launch_direct_fallback(
            transaction,
            graphCircuitAttempt,
            graphCircuitAttemptActive,
            cancelReason,
            managerState,
            countNonResidentServe,
            launch_base_pipeline_direct,
            outCudaErrorCode);
    };
    auto fail_graph_attempt_and_launch_direct =
        [&](const char* failureReason, bool countNonResidentServe) -> bool {
        return fail_graph_attempt_and_launch_direct_fallback(
            transaction,
            graphCircuitAttempt,
            graphCircuitAttemptActive,
            failureReason,
            managerState,
            countNonResidentServe,
            launch_base_pipeline_direct,
            outCudaErrorCode);
    };
    if (!found) {
        const std::uint64_t requestBytes = estimate_base_graph_request_bytes(key);
        std::uint64_t supersededLatestSnapshotId = 0;
        if (requestBytes > 0 &&
            should_cancel_superseded_noncritical_builder(
                transaction,
                "command_launch_base_pipeline_graph",
                kGraphAdmissionCriticalCurrentFrame,
                requestBytes,
                supersededLatestSnapshotId)) {
            return launch_graph_direct(true);
        }
        std::uint32_t observedProbationHits = 0;
        if (!evaluate_graph_large_entry_readmit(
                transaction,
                bucket,
                keyDigest,
                requestBytes,
                graphLargeThresholdBytes,
                kGraphAdmissionCriticalCurrentFrame,
                cfg,
                managerState,
                observedProbationHits)) {
            return launch_graph_direct(true);
        }

        if (!evaluate_graph_cache_admission(
                transaction,
                bucket,
                keyDigest,
                requestBytes,
                observedProbationHits,
                kGraphAdmissionCriticalCurrentFrame,
                cfg,
                managerState)) {
            return launch_graph_direct(true);
        }

        std::string circuitError;
        if (!tier_circuit_begin_attempt(
                transaction,
                "command_launch_base_pipeline_graph",
                ResourceTier::Graph,
                tier_circuit_blocks_admission(ResourceTier::Graph),
                graphCircuitAttempt,
                circuitError)) {
            return launch_graph_direct(true);
        }
        graphCircuitAttemptActive = true;

        BuilderReservationClaim builderClaim{};
        ReservationAttemptInfo builderReservation{};
        if (!acquire_graph_builder_reservation(
                transaction,
                managerState,
                requestBytes,
                kGraphAdmissionCriticalCurrentFrame,
                builderClaim,
                builderReservation)) {
            return cancel_graph_attempt_and_launch_direct("nonresident_builder_gate", true);
        }
        BuilderReservationGuard builderGuard(std::move(builderClaim));

        found = build_admitted_graph_entry(
            transaction,
            bucket,
            key,
            keyDigest,
            launchFn,
            run,
            stream,
            graphLargeThresholdBytes,
            cfg.keepHotMs,
            managerState,
            applyGraphLargeEntryPolicy);
    } else {
        clear_graph_admission_state(bucket, keyDigest);
    }

    if (!found || !found->execOpaque || !found->kernelNodeOpaque) {
        return fail_graph_attempt_and_launch_direct("durable_build_failed", false);
    }

    if (const char* graphLaunchFailure = launch_selected_graph_entry(
            *found,
            run,
            stream,
            useTick,
            reusedResidentGraph,
            applyGraphLargeEntryPolicy,
            outCudaErrorCode)) {
        return fail_graph_attempt_and_launch_direct(graphLaunchFailure, false);
    }
    complete_graph_attempt_success(
        transaction,
        graphCircuitAttempt,
        graphCircuitAttemptActive);
    outCudaErrorCode = static_cast<int>(cudaSuccess);
    return true;
#endif
}

bool error_is_scratch_exhausted(const std::string& error) noexcept {
    return error.rfind(kScratchExhaustedPrefix, 0) == 0 ||
        error.rfind(kReservationDeferredPrefix, 0) == 0;
}
