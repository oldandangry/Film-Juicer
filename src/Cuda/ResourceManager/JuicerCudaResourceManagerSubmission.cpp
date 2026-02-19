// Cuda/ResourceManager/JuicerCudaResourceManagerSubmission.cpp
//
// Included by JuicerCudaResourceManager.cpp (single-TU split).
bool try_acquire_scratch_policy_claim(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const DeviceContextKey& contextKey,
    const ScratchBucketKey& bucketKey,
    std::size_t requestBytes,
    bool criticalCurrentFrame,
    ScratchPolicyClaim& outClaim,
    ScratchPolicySnapshot& outSnapshot,
    ReservationAttemptInfo& outReservation) noexcept {
    outReservation = ReservationAttemptInfo{};
    ScratchPolicyState& state = scratch_policy_state();
    std::lock_guard<std::mutex> lock(state.mutex);

    ScratchContextState& contextState = state.byContext[contextKey];
    const std::uint64_t nowMs = monotonic_time_ms();
    trim_large_frame_quarantine_decay_locked(contextState, nowMs);

    ScratchBucketEntry& bucketEntry = contextState.buckets[bucketKey];
    bucketEntry.attemptCount += 1;
    global_state().scratchBucketAcquireAttempts.fetch_add(1, std::memory_order_relaxed);

    if (requestBytes == 0) {
        bucketEntry.reuseEvents += 1;
        global_state().scratchReuseEvents.fetch_add(1, std::memory_order_relaxed);
        global_state().transientNonManagerBytes.store(state.totalInFlightBytes, std::memory_order_relaxed);
        snapshot_bucket_state_locked(contextState, bucketEntry, bucketKey, outSnapshot);
        return true;
    }

    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    outReservation.considered = true;
    outReservation.bytesInFlight = state.totalInFlightBytes;
    outReservation.capBytes = transient_reservation_cap_bytes(cfg);
    outReservation.thresholdBytes = transient_reservation_threshold_bytes(cfg);

    ReservationDecision reservationDecision{};
    if (static_cast<std::uint64_t>(requestBytes) < outReservation.thresholdBytes) {
        reservationDecision.granted = true;
        reservationDecision.reason = "below_threshold";
    }
    else {
        ReservationInput reservationInput{};
        reservationInput.kind = ReservationKind::TransientNonManager;
        reservationInput.requestBytes = static_cast<std::uint64_t>(requestBytes);
        reservationInput.bytesInFlight = state.totalInFlightBytes;
        reservationInput.capBytes = outReservation.capBytes;
        reservationInput.criticalCurrentFrame = criticalCurrentFrame;
        reservationDecision = classify_reservation(reservationInput);
    }
    outReservation.decision = reservationDecision;
    global_state().transientReservationRequests.fetch_add(1, std::memory_order_relaxed);
    if (!reservationDecision.granted) {
        if (reservationDecision.shouldWait) {
            global_state().transientReservationDeferred.fetch_add(1, std::memory_order_relaxed);
        }
        else {
            global_state().transientReservationDenied.fetch_add(1, std::memory_order_relaxed);
        }
        trace_transient_reservation_decision(
            transaction,
            commandName,
            requestBytes,
            outReservation.bytesInFlight,
            outReservation.capBytes,
            outReservation.thresholdBytes,
            reservationDecision,
            criticalCurrentFrame,
            static_cast<int>(reservationDecision.waitMs),
            reservationDecision.shouldWait ? "deferred" : "denied");
        snapshot_bucket_state_locked(contextState, bucketEntry, bucketKey, outSnapshot);
        return false;
    }

    const std::size_t effectiveCapBytes = effective_temp_scratch_bytes_cap(requestBytes);
    const bool setsOk = bucketEntry.inFlightSets < kMaxTempScratchSets;
    const bool bytesOk = requestBytes <=
        (effectiveCapBytes - std::min(bucketEntry.inFlightBytes, effectiveCapBytes));
    if (!setsOk || !bytesOk) {
        bucketEntry.exhaustedCount += 1;
        global_state().scratchBucketExhaustedEvents.fetch_add(1, std::memory_order_relaxed);

        const bool starvationNow =
            (bucketEntry.attemptCount >= 8) &&
            (bucketEntry.exhaustedCount * 4 >= bucketEntry.attemptCount);
        if (starvationNow && !bucketEntry.starvationLatched) {
            bucketEntry.starvationLatched = true;
            global_state().scratchBucketStarvationEvents.fetch_add(1, std::memory_order_relaxed);
        }
        else if (!starvationNow) {
            bucketEntry.starvationLatched = false;
        }

        snapshot_bucket_state_locked(contextState, bucketEntry, bucketKey, outSnapshot);
        return false;
    }

    bucketEntry.inFlightSets += 1;
    bucketEntry.inFlightBytes += requestBytes;
    bucketEntry.allocGrowthEvents += 1;
    {
        std::size_t nextContextBytes = 0;
        if (add_bytes_checked(contextState.totalInFlightBytes, requestBytes, nextContextBytes)) {
            contextState.totalInFlightBytes = nextContextBytes;
        }
        else {
            contextState.totalInFlightBytes = std::numeric_limits<std::size_t>::max();
        }
    }
    {
        std::uint64_t nextGlobalBytes = 0;
        if (add_u64_checked(state.totalInFlightBytes, static_cast<std::uint64_t>(requestBytes), nextGlobalBytes)) {
            state.totalInFlightBytes = nextGlobalBytes;
        }
        else {
            state.totalInFlightBytes = std::numeric_limits<std::uint64_t>::max();
        }
    }
    global_state().scratchAllocGrowthEvents.fetch_add(1, std::memory_order_relaxed);
    bucketEntry.starvationLatched = false;
    global_state().transientNonManagerBytes.store(state.totalInFlightBytes, std::memory_order_relaxed);
    global_state().transientReservationGranted.fetch_add(1, std::memory_order_relaxed);
    if (reservationDecision.reason && std::string_view(reservationDecision.reason) == "critical_last_resort") {
        trace_transient_reservation_decision(
            transaction,
            commandName,
            requestBytes,
            outReservation.bytesInFlight,
            outReservation.capBytes,
            outReservation.thresholdBytes,
            reservationDecision,
            criticalCurrentFrame,
            0,
            "critical_last_resort");
    }

    outClaim.contextKey = contextKey;
    outClaim.bucketKey = bucketKey;
    outClaim.bytes = requestBytes;
    outClaim.acquired = true;

    snapshot_bucket_state_locked(contextState, bucketEntry, bucketKey, outSnapshot);
    return true;
}

bool acquire_scratch_policy_claim_with_wait(
    const SubmissionTransaction& transaction,
    const char* commandName,
    ScratchWorkClass workClass,
    int width,
    int height,
    std::size_t requestBytes,
    bool criticalCurrentFrame,
    ScratchPolicyClaim& outClaim,
    std::string& outError) {
    outClaim = ScratchPolicyClaim{};
    outError.clear();
    ResourceManagerState& state = global_state();

    const ScratchBucketKey bucketKey = make_scratch_bucket_key(width, height, workClass);
    const int waitBudgetMs = scratch_wait_budget_ms(bucketKey, requestBytes);
    ScratchPolicySnapshot snapshot{};
    ReservationAttemptInfo reservation{};
    int waitedMs = 0;
    while (true) {
        if (try_acquire_scratch_policy_claim(
                transaction,
                commandName,
                transaction.snapshot.deviceContextKey,
                bucketKey,
                requestBytes,
                criticalCurrentFrame,
                outClaim,
                snapshot,
                reservation)) {
            if (waitedMs > 0) {
                state.scratchPolicyWaitEvents.fetch_add(1, std::memory_order_relaxed);
                if (criticalCurrentFrame) {
                    const std::uint64_t waitedMsU64 =
                        static_cast<std::uint64_t>(std::max(waitedMs, 0));
                    state.criticalBuilderWaitEvents.fetch_add(1, std::memory_order_relaxed);
                    state.criticalBuilderWaitTotalMs.fetch_add(waitedMsU64, std::memory_order_relaxed);
                    trace_lane_wait_event(
                        transaction,
                        commandName,
                        PressureLane::Builder,
                        criticalCurrentFrame,
                        waitedMs,
                        "admit_after_wait",
                        "scratch_policy_wait");
                }
                trace_scratch_policy_decision(
                    transaction,
                    commandName,
                    "admit_after_wait",
                    requestBytes,
                    snapshot,
                    waitedMs);
            }
            else if (requestBytes == 0) {
                trace_scratch_policy_decision(
                    transaction,
                    commandName,
                    "reuse_hit",
                    requestBytes,
                    snapshot,
                    waitedMs);
            }
            return true;
        }

        if (waitedMs >= waitBudgetMs) {
            state.scratchPolicyExhaustedEvents.fetch_add(1, std::memory_order_relaxed);
            if (criticalCurrentFrame) {
                const std::uint64_t waitedMsU64 =
                    static_cast<std::uint64_t>(std::max(waitedMs, 0));
                state.criticalBuilderWaitEvents.fetch_add(1, std::memory_order_relaxed);
                state.criticalBuilderWaitTotalMs.fetch_add(waitedMsU64, std::memory_order_relaxed);
                state.criticalLaneStarvationEvents.fetch_add(1, std::memory_order_relaxed);
                trace_lane_wait_event(
                    transaction,
                    commandName,
                    PressureLane::Builder,
                    criticalCurrentFrame,
                    waitedMs,
                    "starved",
                    (reservation.considered && reservation.decision.reason)
                        ? reservation.decision.reason
                        : "scratch_exhausted");
            }
            trace_scratch_policy_decision(
                transaction,
                commandName,
                "exhausted",
                requestBytes,
                snapshot,
                waitedMs);
            if (reservation.considered && !reservation.decision.granted) {
                outError = std::string(kReservationDeferredPrefix)
                    + " command=" + (commandName ? commandName : "unknown")
                    + " work_class=" + to_cstr(workClass)
                    + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
                    + " in_flight_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.bytesInFlight))
                    + " cap_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.capBytes))
                    + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.thresholdBytes))
                    + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
                    + " wait_ms=" + std::to_string(waitedMs)
                    + " wait_budget_ms=" + std::to_string(waitBudgetMs)
                    + " reason=" + (reservation.decision.reason ? reservation.decision.reason : "unspecified");
            }
            else {
                outError = std::string(kScratchExhaustedPrefix)
                    + " command=" + (commandName ? commandName : "unknown")
                    + " work_class=" + to_cstr(workClass)
                    + " bucket_w=" + std::to_string(bucketKey.widthBucket)
                    + " bucket_h=" + std::to_string(bucketKey.heightBucket)
                    + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
                    + " in_flight_bytes=" + std::to_string(static_cast<unsigned long long>(snapshot.inFlightBytes))
                    + " in_flight_sets=" + std::to_string(snapshot.inFlightSets)
                    + " bucket_exhausted=" + std::to_string(snapshot.bucketExhausted)
                    + " bucket_attempts=" + std::to_string(snapshot.bucketAttempts)
                    + " wait_ms=" + std::to_string(waitedMs)
                    + " wait_budget_ms=" + std::to_string(waitBudgetMs);
            }
            return false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(kScratchWaitStepMs));
        waitedMs += kScratchWaitStepMs;
    }
}

bool try_acquire_builder_reservation_claim(
    const SubmissionTransaction& transaction,
    BuilderReservationTier tier,
    std::uint64_t requestBytes,
    bool criticalCurrentFrame,
    BuilderReservationClaim& outClaim,
    ReservationAttemptInfo& outReservation) noexcept {
    outClaim = BuilderReservationClaim{};
    outReservation = ReservationAttemptInfo{};

    if (requestBytes == 0) {
        return true;
    }

    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    BuilderReservationState& state = builder_reservation_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    outReservation.instanceToken = transaction.snapshot.instanceToken.value;

    outReservation.considered = true;
    std::uint64_t& tierTotalBytes = builder_total_bytes_for_tier(state, tier);
    outReservation.bytesInFlight = tierTotalBytes;
    outReservation.capBytes = builder_reservation_cap_bytes(cfg, tier);
    outReservation.thresholdBytes = builder_reservation_threshold_bytes(cfg, tier);

    ReservationDecision reservationDecision{};
    if (requestBytes < outReservation.thresholdBytes) {
        reservationDecision.granted = true;
        reservationDecision.reason = "below_threshold";
    }
    else {
        ReservationInput reservationInput{};
        reservationInput.kind = ReservationKind::BuilderWork;
        reservationInput.requestBytes = requestBytes;
        reservationInput.bytesInFlight = tierTotalBytes;
        reservationInput.capBytes = outReservation.capBytes;
        reservationInput.criticalCurrentFrame = criticalCurrentFrame;
        reservationDecision = classify_reservation(reservationInput);
    }
    outReservation.decision = reservationDecision;

    if (!reservationDecision.granted) {
        return false;
    }

    BuilderReservationContextState& contextState =
        state.byContext[transaction.snapshot.deviceContextKey];
    BuilderReservationContextState::FairnessEntry& fairness =
        contextState.fairnessByInstance[outReservation.instanceToken];
    refill_builder_fairness_tokens(fairness, cfg, monotonic_time_ms());

    bool fairnessGranted = false;
    if (criticalCurrentFrame && fairness.criticalTokens > 0) {
        --fairness.criticalTokens;
        fairnessGranted = true;
    }
    else if (fairness.sharedTokens > 0) {
        --fairness.sharedTokens;
        fairnessGranted = true;
    }
    outReservation.sharedTokens = fairness.sharedTokens;
    outReservation.criticalTokens = fairness.criticalTokens;
    if (!fairnessGranted) {
        ReservationDecision fairnessDecision{};
        fairnessDecision.granted = false;
        fairnessDecision.shouldWait = true;
        fairnessDecision.waitMs = static_cast<std::uint32_t>(kBuilderReservationWaitStepMs);
        fairnessDecision.reason = "fairness_tokens_exhausted";
        outReservation.decision = fairnessDecision;
        return false;
    }

    std::uint64_t& contextTierBytes = builder_context_bytes_for_tier(contextState, tier);
    std::uint64_t nextContextBytes = 0;
    if (add_u64_checked(contextTierBytes, requestBytes, nextContextBytes)) {
        contextTierBytes = nextContextBytes;
    }
    else {
        contextTierBytes = std::numeric_limits<std::uint64_t>::max();
    }
    std::uint64_t nextTotalBytes = 0;
    if (add_u64_checked(tierTotalBytes, requestBytes, nextTotalBytes)) {
        tierTotalBytes = nextTotalBytes;
    }
    else {
        tierTotalBytes = std::numeric_limits<std::uint64_t>::max();
    }
    builder_global_gauge_for_tier(global_state(), tier).store(tierTotalBytes, std::memory_order_relaxed);
    outClaim.contextKey = transaction.snapshot.deviceContextKey;
    outClaim.tier = tier;
    outClaim.bytes = requestBytes;
    outClaim.acquired = true;
    return true;
}

bool acquire_builder_reservation_with_wait(
    const SubmissionTransaction& transaction,
    const char* commandName,
    BuilderReservationTier tier,
    std::uint64_t requestBytes,
    bool criticalCurrentFrame,
    BuilderReservationClaim& outClaim,
    std::string& outError) {
    outClaim = BuilderReservationClaim{};
    outError.clear();

    if (requestBytes == 0) {
        return true;
    }

    ResourceManagerState& state = global_state();
    state.builderReservationRequests.fetch_add(1, std::memory_order_relaxed);

    ReservationAttemptInfo reservation{};
    int waitedMs = 0;
    while (true) {
        if (try_acquire_builder_reservation_claim(
                transaction,
                tier,
                requestBytes,
                criticalCurrentFrame,
                outClaim,
                reservation)) {
            state.builderReservationGranted.fetch_add(1, std::memory_order_relaxed);
            if (waitedMs > 0) {
                state.builderFairnessWaitEvents.fetch_add(1, std::memory_order_relaxed);
                if (criticalCurrentFrame) {
                    const std::uint64_t waitedMsU64 =
                        static_cast<std::uint64_t>(std::max(waitedMs, 0));
                    state.criticalBuilderWaitEvents.fetch_add(1, std::memory_order_relaxed);
                    state.criticalBuilderWaitTotalMs.fetch_add(waitedMsU64, std::memory_order_relaxed);
                    trace_lane_wait_event(
                        transaction,
                        commandName,
                        PressureLane::Builder,
                        criticalCurrentFrame,
                        waitedMs,
                        "admit_after_wait",
                        reservation.decision.reason ? reservation.decision.reason : "waited");
                }
                trace_builder_reservation_decision(
                    transaction,
                    commandName,
                    tier,
                    requestBytes,
                    reservation.bytesInFlight,
                    reservation.capBytes,
                    reservation.thresholdBytes,
                    reservation.decision,
                    criticalCurrentFrame,
                    reservation.instanceToken,
                    reservation.sharedTokens,
                    reservation.criticalTokens,
                    waitedMs,
                    "admit_after_wait");
            }
            else if (reservation.decision.reason &&
                     std::string_view(reservation.decision.reason) == "critical_last_resort") {
                trace_builder_reservation_decision(
                    transaction,
                    commandName,
                    tier,
                    requestBytes,
                    reservation.bytesInFlight,
                    reservation.capBytes,
                    reservation.thresholdBytes,
                    reservation.decision,
                    criticalCurrentFrame,
                    reservation.instanceToken,
                    reservation.sharedTokens,
                    reservation.criticalTokens,
                    waitedMs,
                    "critical_last_resort");
            }
            else if (JTRACE_ENABLED(3)) {
                trace_builder_reservation_decision(
                    transaction,
                    commandName,
                    tier,
                    requestBytes,
                    reservation.bytesInFlight,
                    reservation.capBytes,
                    reservation.thresholdBytes,
                    reservation.decision,
                    criticalCurrentFrame,
                    reservation.instanceToken,
                    reservation.sharedTokens,
                    reservation.criticalTokens,
                    waitedMs,
                    "admitted");
            }
            return true;
        }

        const ReservationDecision& decision = reservation.decision;
        const bool fairnessDeferred =
            (decision.reason && std::string_view(decision.reason) == "fairness_tokens_exhausted");
        if (fairnessDeferred) {
            state.builderFairnessTokenDeferred.fetch_add(1, std::memory_order_relaxed);
        }
        if (!decision.shouldWait) {
            state.builderReservationDenied.fetch_add(1, std::memory_order_relaxed);
            if (criticalCurrentFrame) {
                const std::uint64_t waitedMsU64 =
                    static_cast<std::uint64_t>(std::max(waitedMs, 0));
                state.criticalBuilderWaitEvents.fetch_add(1, std::memory_order_relaxed);
                state.criticalBuilderWaitTotalMs.fetch_add(waitedMsU64, std::memory_order_relaxed);
                state.criticalLaneStarvationEvents.fetch_add(1, std::memory_order_relaxed);
                trace_lane_wait_event(
                    transaction,
                    commandName,
                    PressureLane::Builder,
                    criticalCurrentFrame,
                    waitedMs,
                    "denied",
                    decision.reason ? decision.reason : "reservation_denied");
            }
            trace_builder_reservation_decision(
                transaction,
                commandName,
                tier,
                requestBytes,
                reservation.bytesInFlight,
                reservation.capBytes,
                reservation.thresholdBytes,
                decision,
                criticalCurrentFrame,
                reservation.instanceToken,
                reservation.sharedTokens,
                reservation.criticalTokens,
                waitedMs,
                "denied");
            outError = std::string(kReservationDeferredPrefix)
                + " command=" + (commandName ? commandName : "unknown")
                + " kind=" + to_cstr(ReservationKind::BuilderWork)
                + " tier=" + to_cstr(tier)
                + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
                + " in_flight_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.bytesInFlight))
                + " cap_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.capBytes))
                + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.thresholdBytes))
                + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
                + " wait_ms=" + std::to_string(waitedMs)
                + " wait_budget_ms=" + std::to_string(kBuilderReservationWaitMaxMs)
                + " reason=" + (decision.reason ? decision.reason : "unspecified");
            return false;
        }

        if (waitedMs >= kBuilderReservationWaitMaxMs) {
            state.builderReservationDeferred.fetch_add(1, std::memory_order_relaxed);
            if (criticalCurrentFrame) {
                state.builderReservationBypass.fetch_add(1, std::memory_order_relaxed);
                if (fairnessDeferred) {
                    state.builderFairnessTokenBypass.fetch_add(1, std::memory_order_relaxed);
                }
                const std::uint64_t waitedMsU64 =
                    static_cast<std::uint64_t>(std::max(waitedMs, 0));
                state.criticalBuilderWaitEvents.fetch_add(1, std::memory_order_relaxed);
                state.criticalBuilderWaitTotalMs.fetch_add(waitedMsU64, std::memory_order_relaxed);
                state.criticalLaneStarvationEvents.fetch_add(1, std::memory_order_relaxed);
                trace_lane_wait_event(
                    transaction,
                    commandName,
                    PressureLane::Builder,
                    criticalCurrentFrame,
                    waitedMs,
                    fairnessDeferred ? "bypass_after_starvation" : "deferred_bypass",
                    decision.reason ? decision.reason : "wait_budget_reached");
                trace_builder_reservation_decision(
                    transaction,
                    commandName,
                    tier,
                    requestBytes,
                    reservation.bytesInFlight,
                    reservation.capBytes,
                    reservation.thresholdBytes,
                    decision,
                    criticalCurrentFrame,
                    reservation.instanceToken,
                    reservation.sharedTokens,
                    reservation.criticalTokens,
                    waitedMs,
                    "deferred_bypass_wait_budget");
                return true;
            }
            trace_builder_reservation_decision(
                transaction,
                commandName,
                tier,
                requestBytes,
                reservation.bytesInFlight,
                reservation.capBytes,
                reservation.thresholdBytes,
                decision,
                criticalCurrentFrame,
                reservation.instanceToken,
                reservation.sharedTokens,
                reservation.criticalTokens,
                waitedMs,
                "deferred_wait_budget");
            outError = std::string(kReservationDeferredPrefix)
                + " command=" + (commandName ? commandName : "unknown")
                + " kind=" + to_cstr(ReservationKind::BuilderWork)
                + " tier=" + to_cstr(tier)
                + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
                + " in_flight_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.bytesInFlight))
                + " cap_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.capBytes))
                + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.thresholdBytes))
                + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
                + " wait_ms=" + std::to_string(waitedMs)
                + " wait_budget_ms=" + std::to_string(kBuilderReservationWaitMaxMs)
                + " reason=" + (decision.reason ? decision.reason : "wait_budget_reached");
            return false;
        }

        const int sleepMs = std::max<int>(
            1,
            std::min<int>(
                static_cast<int>(decision.waitMs > 0 ? decision.waitMs : 1),
                kBuilderReservationWaitStepMs));
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
        waitedMs += sleepMs;
    }
}

bool try_acquire_upload_reservation_claim(
    const SubmissionTransaction& transaction,
    std::uint64_t requestBytes,
    bool criticalCurrentFrame,
    UploadReservationClaim& outClaim,
    ReservationAttemptInfo& outReservation) noexcept {
    outClaim = UploadReservationClaim{};
    outReservation = ReservationAttemptInfo{};

    if (requestBytes == 0) {
        return true;
    }

    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    UploadReservationState& state = upload_reservation_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    outReservation.instanceToken = transaction.snapshot.instanceToken.value;

    outReservation.considered = true;
    outReservation.bytesInFlight = state.totalInFlightBytes;
    outReservation.capBytes = upload_reservation_cap_bytes(cfg);
    outReservation.thresholdBytes = upload_reservation_threshold_bytes(cfg);

    ReservationDecision reservationDecision{};
    if (requestBytes < outReservation.thresholdBytes) {
        reservationDecision.granted = true;
        reservationDecision.reason = "below_threshold";
    }
    else {
        ReservationInput reservationInput{};
        reservationInput.kind = ReservationKind::UploadCopy;
        reservationInput.requestBytes = requestBytes;
        reservationInput.bytesInFlight = state.totalInFlightBytes;
        reservationInput.capBytes = outReservation.capBytes;
        reservationInput.criticalCurrentFrame = criticalCurrentFrame;
        reservationDecision = classify_reservation(reservationInput);
    }
    outReservation.decision = reservationDecision;

    if (!reservationDecision.granted) {
        return false;
    }

    UploadReservationContextState& contextState =
        state.byContext[transaction.snapshot.deviceContextKey];
    UploadReservationContextState::FairnessEntry& fairness =
        contextState.fairnessByInstance[outReservation.instanceToken];
    refill_upload_fairness_tokens(fairness, cfg, monotonic_time_ms());

    bool fairnessGranted = false;
    if (criticalCurrentFrame && fairness.criticalTokens > 0) {
        --fairness.criticalTokens;
        fairnessGranted = true;
    }
    else if (fairness.sharedTokens > 0) {
        --fairness.sharedTokens;
        fairnessGranted = true;
    }
    outReservation.sharedTokens = fairness.sharedTokens;
    outReservation.criticalTokens = fairness.criticalTokens;
    if (!fairnessGranted) {
        ReservationDecision fairnessDecision{};
        fairnessDecision.granted = false;
        fairnessDecision.shouldWait = true;
        fairnessDecision.waitMs = static_cast<std::uint32_t>(kUploadReservationWaitStepMs);
        fairnessDecision.reason = "fairness_tokens_exhausted";
        outReservation.decision = fairnessDecision;
        return false;
    }

    std::uint64_t nextContextBytes = 0;
    if (add_u64_checked(contextState.inFlightBytes, requestBytes, nextContextBytes)) {
        contextState.inFlightBytes = nextContextBytes;
    }
    else {
        contextState.inFlightBytes = std::numeric_limits<std::uint64_t>::max();
    }
    std::uint64_t nextTotalBytes = 0;
    if (add_u64_checked(state.totalInFlightBytes, requestBytes, nextTotalBytes)) {
        state.totalInFlightBytes = nextTotalBytes;
    }
    else {
        state.totalInFlightBytes = std::numeric_limits<std::uint64_t>::max();
    }
    global_state().uploadBytesInFlight.store(state.totalInFlightBytes, std::memory_order_relaxed);
    outClaim.contextKey = transaction.snapshot.deviceContextKey;
    outClaim.bytes = requestBytes;
    outClaim.acquired = true;
    return true;
}

bool acquire_upload_reservation_with_wait(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::uint64_t requestBytes,
    bool criticalCurrentFrame,
    UploadReservationClaim& outClaim,
    std::string& outError) {
    outClaim = UploadReservationClaim{};
    outError.clear();

    if (requestBytes == 0) {
        return true;
    }

    ResourceManagerState& state = global_state();
    state.uploadReservationRequests.fetch_add(1, std::memory_order_relaxed);

    ReservationAttemptInfo reservation{};
    int waitedMs = 0;
    while (true) {
        if (try_acquire_upload_reservation_claim(
                transaction,
                requestBytes,
                criticalCurrentFrame,
                outClaim,
                reservation)) {
            state.uploadReservationGranted.fetch_add(1, std::memory_order_relaxed);
            if (waitedMs > 0) {
                state.uploadFairnessWaitEvents.fetch_add(1, std::memory_order_relaxed);
                if (criticalCurrentFrame) {
                    const std::uint64_t waitedMsU64 =
                        static_cast<std::uint64_t>(std::max(waitedMs, 0));
                    state.criticalUploadWaitEvents.fetch_add(1, std::memory_order_relaxed);
                    state.criticalUploadWaitTotalMs.fetch_add(waitedMsU64, std::memory_order_relaxed);
                    trace_lane_wait_event(
                        transaction,
                        commandName,
                        PressureLane::Upload,
                        criticalCurrentFrame,
                        waitedMs,
                        "admit_after_wait",
                        reservation.decision.reason ? reservation.decision.reason : "waited");
                }
                trace_upload_reservation_decision(
                    transaction,
                    commandName,
                    requestBytes,
                    reservation.bytesInFlight,
                    reservation.capBytes,
                    reservation.thresholdBytes,
                    reservation.decision,
                    criticalCurrentFrame,
                    reservation.instanceToken,
                    reservation.sharedTokens,
                    reservation.criticalTokens,
                    waitedMs,
                    "admit_after_wait");
            }
            else if (reservation.decision.reason &&
                     std::string_view(reservation.decision.reason) == "critical_last_resort") {
                trace_upload_reservation_decision(
                    transaction,
                    commandName,
                    requestBytes,
                    reservation.bytesInFlight,
                    reservation.capBytes,
                    reservation.thresholdBytes,
                    reservation.decision,
                    criticalCurrentFrame,
                    reservation.instanceToken,
                    reservation.sharedTokens,
                    reservation.criticalTokens,
                    waitedMs,
                    "critical_last_resort");
            }
            return true;
        }

        const ReservationDecision& decision = reservation.decision;
        const bool fairnessDeferred =
            (decision.reason && std::string_view(decision.reason) == "fairness_tokens_exhausted");
        if (fairnessDeferred) {
            state.uploadFairnessTokenDeferred.fetch_add(1, std::memory_order_relaxed);
        }
        if (!decision.shouldWait) {
            state.uploadReservationDenied.fetch_add(1, std::memory_order_relaxed);
            if (criticalCurrentFrame) {
                const std::uint64_t waitedMsU64 =
                    static_cast<std::uint64_t>(std::max(waitedMs, 0));
                state.criticalUploadWaitEvents.fetch_add(1, std::memory_order_relaxed);
                state.criticalUploadWaitTotalMs.fetch_add(waitedMsU64, std::memory_order_relaxed);
                state.criticalLaneStarvationEvents.fetch_add(1, std::memory_order_relaxed);
                trace_lane_wait_event(
                    transaction,
                    commandName,
                    PressureLane::Upload,
                    criticalCurrentFrame,
                    waitedMs,
                    "denied",
                    decision.reason ? decision.reason : "reservation_denied");
            }
            trace_upload_reservation_decision(
                transaction,
                commandName,
                requestBytes,
                reservation.bytesInFlight,
                reservation.capBytes,
                reservation.thresholdBytes,
                decision,
                criticalCurrentFrame,
                reservation.instanceToken,
                reservation.sharedTokens,
                reservation.criticalTokens,
                waitedMs,
                "denied");
            outError = std::string(kReservationDeferredPrefix)
                + " command=" + (commandName ? commandName : "unknown")
                + " kind=" + to_cstr(ReservationKind::UploadCopy)
                + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
                + " in_flight_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.bytesInFlight))
                + " cap_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.capBytes))
                + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.thresholdBytes))
                + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
                + " wait_ms=" + std::to_string(waitedMs)
                + " wait_budget_ms=" + std::to_string(kUploadReservationWaitMaxMs)
                + " reason=" + (decision.reason ? decision.reason : "unspecified");
            return false;
        }

        if (waitedMs >= kUploadReservationWaitMaxMs) {
            state.uploadReservationDeferred.fetch_add(1, std::memory_order_relaxed);
            state.uploadReservationBypass.fetch_add(1, std::memory_order_relaxed);
            if (fairnessDeferred) {
                state.uploadFairnessTokenBypass.fetch_add(1, std::memory_order_relaxed);
            }
            if (criticalCurrentFrame) {
                const std::uint64_t waitedMsU64 =
                    static_cast<std::uint64_t>(std::max(waitedMs, 0));
                state.criticalUploadWaitEvents.fetch_add(1, std::memory_order_relaxed);
                state.criticalUploadWaitTotalMs.fetch_add(waitedMsU64, std::memory_order_relaxed);
                if (fairnessDeferred) {
                    state.criticalLaneStarvationEvents.fetch_add(1, std::memory_order_relaxed);
                }
                trace_lane_wait_event(
                    transaction,
                    commandName,
                    PressureLane::Upload,
                    criticalCurrentFrame,
                    waitedMs,
                    fairnessDeferred ? "bypass_after_starvation" : "deferred_bypass",
                    decision.reason ? decision.reason : "wait_budget_reached");
            }
            trace_upload_reservation_decision(
                transaction,
                commandName,
                requestBytes,
                reservation.bytesInFlight,
                reservation.capBytes,
                reservation.thresholdBytes,
                decision,
                criticalCurrentFrame,
                reservation.instanceToken,
                reservation.sharedTokens,
                reservation.criticalTokens,
                waitedMs,
                "deferred_bypass_wait_budget");
            return true;
        }

        const int sleepMs = std::max<int>(
            1,
            std::min<int>(
                static_cast<int>(decision.waitMs > 0 ? decision.waitMs : 1),
                kUploadReservationWaitStepMs));
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
        waitedMs += sleepMs;
    }
}

bool key_digests_equal(const KeyDigests& lhs, const KeyDigests& rhs) noexcept {
    return lhs.uploadCoreHash == rhs.uploadCoreHash &&
        lhs.dirHash == rhs.dirHash &&
        lhs.scannerHash == rhs.scannerHash &&
        lhs.autoExposureHash == rhs.autoExposureHash;
}

ResourcePlan make_uniform_resource_plan(AcquireStatus status) noexcept {
    ResourcePlan plan{};
    for (ResourceKind kind : kResourceKindOrder) {
        ResourcePlanEntry& entry = resource_plan_entry(plan, kind);
        entry.acquire.status = status;
        entry.acquire.shouldBuild = (status == AcquireStatus::Miss);
        entry.invalidated = false;
    }
    return plan;
}

AcquireStatus combine_status(const ResourcePlan& plan) noexcept {
    bool sawMiss = false;
    bool sawBusy = false;
    bool sawExhausted = false;
    for (ResourceKind kind : kResourceKindOrder) {
        const AcquireStatus status = resource_plan_entry(plan, kind).acquire.status;
        if (status == AcquireStatus::Error) {
            return AcquireStatus::Error;
        }
        if (status == AcquireStatus::Exhausted) {
            sawExhausted = true;
        }
        else if (status == AcquireStatus::Busy) {
            sawBusy = true;
        }
        else if (status == AcquireStatus::Miss) {
            sawMiss = true;
        }
    }
    if (sawExhausted) {
        return AcquireStatus::Exhausted;
    }
    if (sawBusy) {
        return AcquireStatus::Busy;
    }
    if (sawMiss) {
        return AcquireStatus::Miss;
    }
    return AcquireStatus::Hit;
}

bool validate_resource_kind_onboarding_contract(std::string& outError) noexcept {
    if (!resource_kind_contract_is_valid()) {
        outError = "resource kind onboarding contract invalid";
        return false;
    }
    for (ResourceKind kind : kResourceKindOrder) {
        const ResourceKindContractEntry& entry = resource_kind_contract_entry(kind);
        if (entry.kind != kind ||
            entry.keyField == nullptr ||
            entry.invalidationLane == nullptr ||
            entry.resourceNode == nullptr ||
            entry.acquireStatusField == nullptr ||
            entry.telemetryTag == nullptr) {
            outError = "resource kind onboarding entry missing required fields";
            return false;
        }
    }
    return true;
}

void trace_lifecycle_stage_decision(
    const SubmissionTransaction& transaction,
    ContextLifecycleState observedState,
    LifecycleStageDecision decision,
    std::uint64_t stateAgeMs,
    const char* stage,
    bool accepted,
    const char* reason) {
    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("transaction_id=") + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " stage=" + (stage ? stage : "unknown")
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " observed_state=" + to_cstr(observedState)
        + " decision=" + to_cstr(decision)
        + " state_age_ms=" + std::to_string(static_cast<unsigned long long>(stateAgeMs))
        + " accepted=" + std::to_string(accepted ? 1 : 0)
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSLCY", msg);
}

bool validate_lifecycle_for_stage(const SubmissionTransaction& transaction,
                                  const char* stage,
                                  bool allowNonActiveRelease,
                                  std::string* outError) {
    LifecycleStageValidation validation{};
    if (!registry_validate_lifecycle_stage(
            transaction.snapshot.deviceContextKey,
            allowNonActiveRelease,
            validation)) {
        global_state().lifecycleStageRejects.fetch_add(1, std::memory_order_relaxed);
        const char* reason = "lifecycle_stage_rejected";
        switch (validation.decision) {
        case LifecycleStageDecision::MissingRegistryEntry:
            reason = "missing_registry_entry";
            if (outError) {
                *outError = "missing registry entry for lifecycle validation";
            }
            break;
        case LifecycleStageDecision::StateNotAllowed:
            reason = "lifecycle_state_not_allowed";
            if (outError) {
                *outError = std::string("lifecycle state not allowed for stage (state=")
                    + to_cstr(validation.observedState) + ")";
            }
            break;
        case LifecycleStageDecision::TimedOut:
            reason = validation.escalated
                ? "lifecycle_state_timeout_escalated"
                : "lifecycle_state_timeout_not_escalated";
            if (outError) {
                *outError = std::string("lifecycle watchdog timeout for stage (state=")
                    + to_cstr(validation.observedState)
                    + ", age_ms="
                    + std::to_string(static_cast<unsigned long long>(validation.observedStateAgeMs))
                    + ")";
            }
            break;
        case LifecycleStageDecision::Allowed:
        default:
            break;
        }
        trace_lifecycle_stage_decision(
            transaction,
            validation.observedState,
            validation.decision,
            validation.observedStateAgeMs,
            stage,
            false,
            reason);
        if (outError) {
            if (outError->empty()) {
                *outError = "lifecycle stage rejected";
            }
        }
        return false;
    }
    if (JTRACE_ENABLED(3)) {
        trace_lifecycle_stage_decision(
            transaction,
            validation.observedState,
            validation.decision,
            validation.observedStateAgeMs,
            stage,
            true,
            "stage_allowed");
    }
    return true;
}

bool validate_stale_tuple_for_stage(
    const SubmissionTransaction& transaction,
    const char* stage,
    LeaseObservationMode leaseObservationMode,
    const char* errorPrefix,
    std::string* outError,
    bool recordModuleBoundaryViolation,
    const char* moduleBoundaryReason) {
    const StaleInput staleInput = state_build_stale_input(transaction, leaseObservationMode);
    const StaleDecision staleDecision = classify_stale_path(staleInput);
    telemetry_trace_stale_decision(
        transaction.transactionId,
        transaction.snapshot.snapshotId,
        transaction.snapshot.traceSchemaVersion,
        stage,
        staleInput,
        staleDecision);
    if (!staleDecision.hardStale && !staleDecision.hardMiss) {
        return true;
    }

    telemetry_record_stale_tuple_hard_reject();
    if (outError && errorPrefix) {
        *outError = std::string(errorPrefix) + to_cstr(staleDecision.reason) + ")";
    }
    if (recordModuleBoundaryViolation) {
        telemetry_record_module_boundary_violation();
        telemetry_trace_module_boundary_violation(
            transaction.transactionId,
            transaction.snapshot.snapshotId,
            transaction.snapshot.traceSchemaVersion,
            moduleBoundaryReason ? moduleBoundaryReason : "stale_tuple_reject");
    }
    return false;
}
