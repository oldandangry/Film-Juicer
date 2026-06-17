// Cuda/ResourceManager/JuicerCudaResourceManagerAdmission.inc
//
// Included by JuicerCudaResourceManager.cpp (single-TU split).
void maybe_apply_async_mempool_release_policy(
    const SubmissionTransaction& transaction,
    PressureState pressureState,
    const ResourceManagerConfigEffective& cfg,
    const char* reason);
void trim_large_frame_quarantine_decay_locked(
    ScratchContextState& contextState,
    std::uint64_t nowMs) noexcept;
void snapshot_bucket_state_locked(
    const ScratchContextState& contextState,
    const ScratchBucketEntry& bucketEntry,
    const ScratchBucketKey& bucketKey,
    ScratchPolicySnapshot& outSnapshot) noexcept;

ScratchPolicyState& scratch_policy_state() {
    static ScratchPolicyState state{};
    return state;
}

ScratchNormalizationState& scratch_normalization_state() {
    static ScratchNormalizationState state{};
    return state;
}

UploadReservationState& upload_reservation_state() {
    static UploadReservationState state{};
    return state;
}

BuilderReservationState& builder_reservation_state() {
    static BuilderReservationState state{};
    return state;
}

void bump_reservation_state_version(std::uint64_t& version) noexcept {
    version = (version == std::numeric_limits<std::uint64_t>::max()) ? 1ull : (version + 1ull);
}

void wait_for_scratch_policy_state_change_or_timeout(
    std::uint64_t observedVersion,
    int waitMs) noexcept {
    try {
        if (waitMs <= 0) {
            return;
        }
        ScratchPolicyState& state = scratch_policy_state();
        std::unique_lock<std::mutex> lock(state.mutex);
        if (state.stateVersion != observedVersion) {
            return;
        }
        state.cv.wait_for(
            lock,
            std::chrono::milliseconds(waitMs),
            [&state, observedVersion]() {
                return state.stateVersion != observedVersion;
            });
    } catch (...) {
        JuicerLogging::discard_current_exception();
    }
}

void wait_for_builder_reservation_state_change_or_timeout(
    std::uint64_t observedVersion,
    int waitMs) noexcept {
    try {
        if (waitMs <= 0) {
            return;
        }
        BuilderReservationState& state = builder_reservation_state();
        std::unique_lock<std::mutex> lock(state.mutex);
        if (state.stateVersion != observedVersion) {
            return;
        }
        state.cv.wait_for(
            lock,
            std::chrono::milliseconds(waitMs),
            [&state, observedVersion]() {
                return state.stateVersion != observedVersion;
            });
    } catch (...) {
        JuicerLogging::discard_current_exception();
    }
}

void wait_for_upload_reservation_state_change_or_timeout(
    std::uint64_t observedVersion,
    int waitMs) noexcept {
    try {
        if (waitMs <= 0) {
            return;
        }
        UploadReservationState& state = upload_reservation_state();
        std::unique_lock<std::mutex> lock(state.mutex);
        if (state.stateVersion != observedVersion) {
            return;
        }
        state.cv.wait_for(
            lock,
            std::chrono::milliseconds(waitMs),
            [&state, observedVersion]() {
                return state.stateVersion != observedVersion;
            });
    } catch (...) {
        JuicerLogging::discard_current_exception();
    }
}

PressurePolicyState& pressure_policy_state() {
    static PressurePolicyState state{};
    return state;
}

TierCircuitPolicyState& tier_circuit_policy_state() {
    static TierCircuitPolicyState state{};
    return state;
}

AdmissionChurnPolicyState& admission_churn_policy_state() {
    static AdmissionChurnPolicyState state{};
    return state;
}

OptionalHeuristicTraceState& optional_heuristic_trace_state() {
    static OptionalHeuristicTraceState state{};
    return state;
}

bool contains_ascii_case_insensitive(const std::string& haystack, const char* needle) noexcept {
    if (!needle || !*needle) {
        return true;
    }
    if (haystack.empty()) {
        return false;
    }
    const std::size_t needleLen = std::char_traits<char>::length(needle);
    if (needleLen == 0 || needleLen > haystack.size()) {
        return false;
    }
    for (std::size_t i = 0; i + needleLen <= haystack.size(); ++i) {
        bool match = true;
        for (std::size_t j = 0; j < needleLen; ++j) {
            const unsigned char a = static_cast<unsigned char>(haystack[i + j]);
            const unsigned char b = static_cast<unsigned char>(needle[j]);
            if (std::tolower(a) != std::tolower(b)) {
                match = false;
                break;
            }
        }
        if (match) {
            return true;
        }
    }
    return false;
}

bool is_allocator_oom_error(const std::string& error) noexcept {
    if (error.empty()) {
        return false;
    }
    return contains_ascii_case_insensitive(error, "out of memory") ||
           contains_ascii_case_insensitive(error, "memory allocation");
}

bool tier_circuit_should_count_failure(const std::string& error) noexcept {
    if (error.empty()) {
        return true;
    }
    if (error.rfind(kScratchExhaustedPrefix, 0) == 0) {
        return false;
    }
    if (error.rfind(kReservationDeferredPrefix, 0) == 0) {
        return false;
    }
    if (error.rfind(kPressureShedNonCriticalPrefix, 0) == 0) {
        return false;
    }
    if (error.rfind(kPressureCopyComputeGuardPrefix, 0) == 0) {
        return false;
    }
    return true;
}

std::string make_pressure_shed_noncritical_error(
    PressureLane lane,
    PressureState state) {
    return std::string(kPressureShedNonCriticalPrefix) + " lane=" + to_cstr(lane) + " state=" + to_cstr(state);
}

std::string make_burst_debt_throttle_noncritical_error(
    PressureLane lane,
    std::uint64_t debtPct,
    std::uint64_t maxDebtPct) {
    return std::string(kBurstDebtThrottleNonCriticalPrefix) + " lane=" + to_cstr(lane) + " debt_pct=" + std::to_string(static_cast<unsigned long long>(debtPct)) + " max_debt_pct=" + std::to_string(static_cast<unsigned long long>(maxDebtPct));
}

const char* reservation_wait_outcome_reason(bool shouldWait) noexcept {
    if (shouldWait) {
        return "deferred";
    }
    return "denied";
}

const char* fairness_bypass_reason(bool fairnessDeferred) noexcept {
    if (fairnessDeferred) {
        return "bypass_after_starvation";
    }
    return "deferred_bypass";
}

const char* sampled_cache_reason(bool sampled) noexcept {
    if (sampled) {
        return "sampled_checkpoint";
    }
    return "cached_checkpoint";
}

const char* headroom_trace_reason(bool sourceSwitch, bool sampled) noexcept {
    if (sourceSwitch) {
        return "headroom_source_switch";
    }
    return sampled_cache_reason(sampled);
}

const char* reserve_update_reason(bool reserveUpdated) noexcept {
    if (reserveUpdated) {
        return "reserve_update";
    }
    return "reserve_sample";
}

const char* freeze_transition_reason(bool entered, bool exited) noexcept {
    if (entered) {
        return "freeze_enter";
    }
    if (exited) {
        return "freeze_exit";
    }
    return "freeze_sample";
}

const char* headroom_alloc_fail_reason(bool aboveHeadroom) noexcept {
    if (aboveHeadroom) {
        return "alloc_fail_above_headroom";
    }
    return "alloc_fail_below_headroom";
}

const char* burst_transition_reason(bool capHit, bool entered, bool exited) noexcept {
    if (capHit) {
        return "burst_cap_hit";
    }
    if (entered) {
        return "burst_enter";
    }
    if (exited) {
        return "burst_exit";
    }
    return "burst_sample";
}

const char* tier_pretrim_reason(bool didTierPretrim) noexcept {
    if (didTierPretrim) {
        return "tier_target_pretrim";
    }
    return "tier_target_observation";
}

const char* emergency_progress_reason(bool criticalCurrentFrame) noexcept {
    if (criticalCurrentFrame) {
        return "allow_critical_progress";
    }
    return "allow_noncritical_progress";
}

std::uint64_t admission_elapsed_ms_since(std::uint64_t nowMs, std::uint64_t startMs) noexcept {
    if (nowMs > startMs) {
        return nowMs - startMs;
    }
    return 0;
}

int admission_elapsed_ms_since_clamped(std::uint64_t startMs) noexcept {
    if (startMs == 0) {
        return 0;
    }
    const std::uint64_t elapsedMs =
        admission_elapsed_ms_since(monotonic_time_ms(), startMs);
    const std::uint64_t maxInt =
        static_cast<std::uint64_t>(std::numeric_limits<int>::max());
    return static_cast<int>(std::min<std::uint64_t>(elapsedMs, maxInt));
}

std::uint32_t admission_bool_u32(bool value) noexcept {
    if (value) {
        return 1u;
    }
    return 0u;
}

int admission_wait_ms_or_one(std::uint64_t waitMs) noexcept {
    if (waitMs > 0) {
        return static_cast<int>(waitMs);
    }
    return 1;
}

const char* admission_reservation_reason_or(
    const ReservationAttemptInfo& reservation,
    const char* fallback) noexcept {
    if (reservation.considered && reservation.decision.reason) {
        return reservation.decision.reason;
    }
    return fallback;
}

std::uint64_t admission_u64_or_default(std::uint64_t value, std::uint64_t fallback) noexcept {
    if (value > 0) {
        return value;
    }
    return fallback;
}

std::uint32_t admission_u32_or_default(std::uint32_t value, std::uint32_t fallback) noexcept {
    if (value > 0) {
        return value;
    }
    return fallback;
}

PressureState admission_pressure_state_or_default(
    const PressureContextState& contextState,
    PressureState fallback) noexcept {
    if (contextState.valid) {
        return contextState.lastState;
    }
    return fallback;
}

std::uint64_t admission_effective_reserve_or_default(
    const PressureContextState& contextState,
    std::uint64_t fallback) noexcept {
    if (contextState.effectiveReserveValid) {
        return contextState.effectiveReserveBytes;
    }
    return fallback;
}

PressureState admission_previous_state_or_decision(
    const PressureContextState& contextState,
    PressureState decisionState) noexcept {
    if (contextState.valid) {
        return contextState.lastState;
    }
    return decisionState;
}

std::uint64_t admission_checkpoint_headroom_or_zero(
    bool checkpointValid,
    const PressureCheckpoint& checkpoint) noexcept {
    if (checkpointValid) {
        return checkpoint.input.effectiveHeadroomBytes;
    }
    return 0;
}

std::uint64_t admission_checkpoint_reserve_or_zero(
    bool checkpointValid,
    const PressureCheckpoint& checkpoint) noexcept {
    if (checkpointValid) {
        return checkpoint.decision.effectiveReserveBytes;
    }
    return 0;
}

const char* admission_upload_saturation_reason(bool criticalCurrentFrame) noexcept {
    if (criticalCurrentFrame) {
        return "critical_upload_allowed_despite_saturation";
    }
    return "upload_allowed_below_shed_threshold";
}

std::uint64_t admission_open_remaining_ms(std::uint64_t elapsedMs, std::uint32_t openMs) noexcept {
    const std::uint64_t openMsU64 = static_cast<std::uint64_t>(openMs);
    if (elapsedMs >= openMsU64) {
        return 0;
    }
    return openMsU64 - elapsedMs;
}

std::uint64_t admission_state_age_or_max(
    std::uint64_t nowMs,
    std::uint64_t lastStateChangeMs) noexcept {
    if (lastStateChangeMs == 0 || nowMs < lastStateChangeMs) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return nowMs - lastStateChangeMs;
}

std::string admission_error_or_message(const std::string& error, const char* fallback) {
    if (error.empty()) {
        return std::string(fallback);
    }
    return error;
}

std::uint64_t tier_target_basis_points(
    const ResourceManagerConfigEffective& cfg,
    ResourceTier tier) noexcept {
    switch (tier) {
        case ResourceTier::Immutable:
            return cfg.tierTargetImmutableBp;
        case ResourceTier::Lut:
            return cfg.tierTargetLutBp;
        case ResourceTier::Scratch:
            return cfg.tierTargetScratchBp;
        case ResourceTier::Graph:
            return cfg.tierTargetGraphBp;
        default:
            return 0;
    }
}

std::uint64_t tier_target_bytes(
    const ResourceManagerConfigEffective& cfg,
    const ResolvedPressurePolicy& policy,
    ResourceTier tier) noexcept {
    if (policy.softTargetBytes == 0) {
        return 0;
    }
    const std::uint64_t bp = std::min<std::uint64_t>(
        tier_target_basis_points(cfg, tier),
        kBasisPointsDenominator);
    std::uint64_t weighted = 0;
    if (!mul_u64_checked(policy.softTargetBytes, bp, weighted)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return weighted / kBasisPointsDenominator;
}

inline bool pressure_policy_enabled(const ResolvedPressurePolicy& policy) noexcept {
    return policy.softTargetBytes > 0;
}

bool validate_scratch_request_descriptor_for_manager(
    const ScratchRequestDescriptor& descriptor,
    const char* commandName,
    std::string& outError) noexcept {
    if (!scratch_request_descriptor_is_valid(descriptor)) {
        outError = std::string(trace_or_non_empty(commandName, "scratch_request")) + ": invalid scratch request descriptor";
        return false;
    }
    return true;
}

std::uint64_t scratch_candidate_live_bytes(
    const ScratchResidencyView& view,
    ScratchPolicyCandidate candidate) noexcept {
    const std::size_t index = scratch_policy_candidate_index(candidate);
    if (index >= view.candidates.size()) {
        return 0;
    }
    return view.candidates[index].liveRetainedBytes;
}

bool scratch_candidate_required_by_request(
    ScratchPolicyCandidate candidate,
    const ScratchRequestDescriptor& descriptor) noexcept {
    switch (candidate) {
        case ScratchPolicyCandidate::OpticsBase:
            return descriptor.needOptics;
        case ScratchPolicyCandidate::OpticsBlurred:
            return descriptor.needOptics && descriptor.needBlurred;
        case ScratchPolicyCandidate::OpticsAux:
            return descriptor.needOptics && descriptor.needAux;
        case ScratchPolicyCandidate::OpticsGrainTriplet:
            return descriptor.needOptics && descriptor.needGrainTriplet;
        case ScratchPolicyCandidate::OpticsGrainShared:
            return descriptor.needOptics && descriptor.needGrainShared;
        case ScratchPolicyCandidate::OpticsGateMask:
            return descriptor.needOptics && descriptor.needGateMask;
        case ScratchPolicyCandidate::SpatialDirBase:
            return descriptor.needSpatialDir;
        default:
            return false;
    }
}

ScratchEligibilityResult evaluate_scratch_eligibility(
    const ScratchResidencyView& view,
    const ScratchRequestDescriptor& descriptor) noexcept {
    ScratchEligibilityResult result{};
    for (ScratchPolicyCandidate candidate : kScratchPolicyCandidateOrder) {
        const std::size_t index = scratch_policy_candidate_index(candidate);
        if (index >= result.eligibleNow.size()) {
            continue;
        }
        const std::uint64_t liveBytes = scratch_candidate_live_bytes(view, candidate);
        const bool eligible = (liveBytes > 0) && !scratch_candidate_required_by_request(candidate, descriptor);
        result.eligibleNow[index] = eligible;
        if (eligible) {
            result.anyEligible = true;
            std::uint64_t next = 0;
            if (!add_u64_checked(result.reclaimableLiveBytes, liveBytes, next)) {
                result.reclaimableLiveBytes = std::numeric_limits<std::uint64_t>::max();
            } else {
                result.reclaimableLiveBytes = next;
            }
        }
    }

    if (view.helperSharedBytes > 0) {
        const bool opticsBaseLive =
            scratch_candidate_live_bytes(view, ScratchPolicyCandidate::OpticsBase) > 0;
        const bool spatialDirBaseLive =
            scratch_candidate_live_bytes(view, ScratchPolicyCandidate::SpatialDirBase) > 0;
        const bool opticsBaseEligible =
            result.eligibleNow[scratch_policy_candidate_index(ScratchPolicyCandidate::OpticsBase)];
        const bool spatialDirBaseEligible =
            result.eligibleNow[scratch_policy_candidate_index(ScratchPolicyCandidate::SpatialDirBase)];
        const bool sharedEligible =
            (!opticsBaseLive || opticsBaseEligible) &&
            (!spatialDirBaseLive || spatialDirBaseEligible);
        if (sharedEligible) {
            result.sharedTmpEligibleNow = true;
            result.anyEligible = true;
            std::uint64_t next = 0;
            if (!add_u64_checked(result.reclaimableLiveBytes, view.helperSharedBytes, next)) {
                result.reclaimableLiveBytes = std::numeric_limits<std::uint64_t>::max();
            } else {
                result.reclaimableLiveBytes = next;
            }
        }
    }

    return result;
}

bool scratch_no_shed_cache_matches(
    const ScratchNormalizationContextState& contextState,
    std::uint64_t requestDescriptorGeneration,
    std::uint64_t retainedScratchGeneration,
    std::uint64_t scratchTargetBytes) noexcept {
    return contextState.noShedCache.valid &&
           contextState.noShedCache.requestDescriptorGeneration == requestDescriptorGeneration &&
           contextState.noShedCache.retainedScratchGeneration == retainedScratchGeneration &&
           contextState.noShedCache.scratchTargetBytes == scratchTargetBytes;
}

std::uint64_t scratch_stage1_threshold_bytes(std::uint64_t scratchTargetBytes) noexcept {
    std::uint64_t threshold = 0;
    if (!add_u64_checked(
            scratchTargetBytes,
            kScratchNormalizationHysteresisBytes,
            threshold)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return threshold;
}

void refresh_scratch_checkpoint_stage2_observation(
    JuicerCuda::Resources& resources,
    const ScratchRequestDescriptor& scratchRequest,
    ScratchCheckpointObservation& outObservation) noexcept {
    JuicerCuda::snapshot_scratch_residency_view(resources, outObservation.residency);
    outObservation.stage2Evaluated = true;
    outObservation.stage2SkippedByNoShedCache = false;
    outObservation.policyLiveRetainedBytes = outObservation.residency.policyLiveRetainedBytes;
    outObservation.retainedScratchGeneration = outObservation.residency.retainedGeneration;
    outObservation.eligibility =
        evaluate_scratch_eligibility(outObservation.residency, scratchRequest);
    outObservation.reclaimableLiveBytes = outObservation.eligibility.reclaimableLiveBytes;

    const std::uint64_t stage1Threshold =
        scratch_stage1_threshold_bytes(outObservation.scratchTargetBytes);
    outObservation.stage1OverTarget =
        outObservation.policyLiveRetainedBytes > stage1Threshold;
    outObservation.overTargetButNotReducible =
        outObservation.stage1OverTarget &&
        (outObservation.reclaimableLiveBytes == 0);
}

bool execute_scratch_shedding_primitive(
    const SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const char* commandName,
    const ScratchRequestDescriptor& scratchRequest,
    ScratchCheckpointInvocation invocation,
    ScratchCheckpointObservation& ioObservation,
    ScratchSheddingResult& outResult,
    std::string& outError) {
    outResult = ScratchSheddingResult{};
    outError.clear();

    if (!ioObservation.requestActive || !ioObservation.stage2Evaluated) {
        return true;
    }

    const std::uint64_t stage1Threshold =
        scratch_stage1_threshold_bytes(ioObservation.scratchTargetBytes);
    if (ioObservation.policyLiveRetainedBytes <= stage1Threshold) {
        outResult.targetReached = true;
        return true;
    }

    auto record_progress = [&](std::uint64_t beforeBytes, std::uint64_t afterBytes) {
        if (afterBytes < beforeBytes) {
            outResult.progressed = true;
            const std::uint64_t retiredBytes = beforeBytes - afterBytes;
            std::uint64_t nextRetiredBytes = 0;
            if (!add_u64_checked(outResult.retiredLiveBytes, retiredBytes, nextRetiredBytes)) {
                outResult.retiredLiveBytes = std::numeric_limits<std::uint64_t>::max();
            } else {
                outResult.retiredLiveBytes = nextRetiredBytes;
            }
        }
    };

    for (ScratchPolicyCandidate candidate : kScratchPolicySheddingOrder) {
        if (ioObservation.policyLiveRetainedBytes <= stage1Threshold) {
            outResult.targetReached = true;
            return true;
        }

        const std::size_t index = scratch_policy_candidate_index(candidate);
        if (index >= ioObservation.eligibility.eligibleNow.size() ||
            !ioObservation.eligibility.eligibleNow[index]) {
            continue;
        }

        const std::uint64_t candidateBytesBefore =
            scratch_candidate_live_bytes(ioObservation.residency, candidate);
        if (candidateBytesBefore == 0) {
            continue;
        }

        const std::uint64_t policyBytesBefore = ioObservation.policyLiveRetainedBytes;
        const std::uint64_t sharedBytesBefore = ioObservation.residency.helperSharedBytes;
        outResult.attempted = true;

        std::string retireError;
        const bool retired = JuicerCuda::retire_scratch_policy_candidate(
            resources,
            candidate,
            nullptr,
            retireError);

        refresh_scratch_checkpoint_stage2_observation(
            resources,
            scratchRequest,
            ioObservation);

        const std::uint64_t candidateBytesAfter =
            scratch_candidate_live_bytes(ioObservation.residency, candidate);
        const bool orphanedSharedTmpRetired =
            ioObservation.residency.helperSharedBytes < sharedBytesBefore;
        const bool actionProgressed =
            ioObservation.policyLiveRetainedBytes < policyBytesBefore;

        record_progress(policyBytesBefore, ioObservation.policyLiveRetainedBytes);
        if (actionProgressed && outResult.retiredActionCount < std::numeric_limits<std::uint32_t>::max()) {
            ++outResult.retiredActionCount;
        }

        trace_scratch_shedding_event(
            transaction,
            ScratchSheddingEventTrace{
                .commandName = commandName,
                .actionName = to_cstr(candidate),
                .reason = retired ? "candidate_retired" : retireError.c_str(),
                .candidateLiveBytesBefore = candidateBytesBefore,
                .candidateLiveBytesAfter = candidateBytesAfter,
                .policyLiveBytesBefore = policyBytesBefore,
                .policyLiveBytesAfter = ioObservation.policyLiveRetainedBytes,
                .invocation = invocation,
                .orphanedSharedTmpRetired = orphanedSharedTmpRetired,
                .partialFailure = !retired});

        if (!retired) {
            outResult.partialFailure = true;
            outResult.orphanedSharedTmpRetired =
                outResult.orphanedSharedTmpRetired || orphanedSharedTmpRetired;
            outError = retireError;
            outResult.targetReached =
                ioObservation.policyLiveRetainedBytes <= stage1Threshold;
            return true;
        }

        outResult.orphanedSharedTmpRetired =
            outResult.orphanedSharedTmpRetired || orphanedSharedTmpRetired;
    }

    if (ioObservation.policyLiveRetainedBytes > stage1Threshold &&
        ioObservation.eligibility.sharedTmpEligibleNow) {
        const std::uint64_t policyBytesBefore = ioObservation.policyLiveRetainedBytes;
        const std::uint64_t sharedBytesBefore = ioObservation.residency.helperSharedBytes;
        outResult.attempted = true;

        std::string retireError;
        const bool retired = JuicerCuda::retire_orphaned_shared_tmp_plane(
            resources,
            nullptr,
            retireError);

        refresh_scratch_checkpoint_stage2_observation(
            resources,
            scratchRequest,
            ioObservation);

        const std::uint64_t sharedBytesAfter = ioObservation.residency.helperSharedBytes;
        const bool orphanedSharedTmpRetired = sharedBytesAfter < sharedBytesBefore;
        const bool actionProgressed =
            ioObservation.policyLiveRetainedBytes < policyBytesBefore;
        record_progress(policyBytesBefore, ioObservation.policyLiveRetainedBytes);
        outResult.orphanedSharedTmpRetired =
            outResult.orphanedSharedTmpRetired || orphanedSharedTmpRetired;
        if (actionProgressed && outResult.retiredActionCount < std::numeric_limits<std::uint32_t>::max()) {
            ++outResult.retiredActionCount;
        }

        trace_scratch_shedding_event(
            transaction,
            ScratchSheddingEventTrace{
                .commandName = commandName,
                .actionName = "HelperSharedTmp",
                .reason = retired
                              ? (orphanedSharedTmpRetired
                                     ? "orphaned_shared_tmp_retired"
                                     : "shared_tmp_not_orphaned")
                              : retireError.c_str(),
                .candidateLiveBytesBefore = sharedBytesBefore,
                .candidateLiveBytesAfter = sharedBytesAfter,
                .policyLiveBytesBefore = policyBytesBefore,
                .policyLiveBytesAfter = ioObservation.policyLiveRetainedBytes,
                .invocation = invocation,
                .orphanedSharedTmpRetired = orphanedSharedTmpRetired,
                .partialFailure = !retired});

        if (!retired) {
            outResult.partialFailure = true;
            outError = retireError;
            outResult.targetReached =
                ioObservation.policyLiveRetainedBytes <= stage1Threshold;
            return true;
        }
    }

    outResult.targetReached =
        ioObservation.policyLiveRetainedBytes <= stage1Threshold;
    return true;
}

bool run_canonical_scratch_checkpoint(
    const SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const char* commandName,
    const ScratchRequestDescriptor* scratchRequest,
    ScratchCheckpointInvocation invocation,
    ScratchCheckpointObservation& outObservation,
    std::string& outError) {
    outObservation = ScratchCheckpointObservation{};
    outError.clear();

    if (!scratchRequest) {
        trace_scratch_checkpoint_event(
            transaction,
            commandName,
            invocation,
            outObservation,
            "no_active_phase_request");
        return true;
    }

    if (!validate_scratch_request_descriptor_for_manager(*scratchRequest, commandName, outError)) {
        return false;
    }

    outObservation.requestActive = true;
    outObservation.requestDescriptorGeneration = scratchRequest->generation;
    outObservation.hysteresisBytes = kScratchNormalizationHysteresisBytes;

    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    if (!pressure_policy_enabled(transaction.resolvedPressurePolicy)) {
        trace_scratch_checkpoint_event(
            transaction,
            commandName,
            invocation,
            outObservation,
            "pressure_policy_disabled");
        return true;
    }
    outObservation.scratchTargetBytes =
        tier_target_bytes(cfg, transaction.resolvedPressurePolicy, ResourceTier::Scratch);

    ScratchStage1DecisionState stage1State{};
    JuicerCuda::snapshot_scratch_stage1_state(resources, stage1State);
    outObservation.policyLiveRetainedBytes = stage1State.policyLiveRetainedBytes;
    outObservation.retainedScratchGeneration = stage1State.retainedGeneration;

    const std::uint64_t stage1Threshold =
        scratch_stage1_threshold_bytes(outObservation.scratchTargetBytes);
    outObservation.stage1OverTarget =
        outObservation.policyLiveRetainedBytes > stage1Threshold;
    if (!outObservation.stage1OverTarget) {
        trace_scratch_checkpoint_event(
            transaction,
            commandName,
            invocation,
            outObservation,
            "below_stage1_threshold");
        return true;
    }

    {
        ScratchNormalizationState& state = scratch_normalization_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        ScratchNormalizationContextState& contextState =
            state.byContext[transaction.snapshot.deviceContextKey];
        if (scratch_no_shed_cache_matches(
                contextState,
                outObservation.requestDescriptorGeneration,
                outObservation.retainedScratchGeneration,
                outObservation.scratchTargetBytes)) {
            outObservation.stage2SkippedByNoShedCache = true;
            outObservation.overTargetButNotReducible = true;
            trace_scratch_checkpoint_event(
                transaction,
                commandName,
                invocation,
                outObservation,
                "no_shed_cache_hit");
            return true;
        }
    }

    refresh_scratch_checkpoint_stage2_observation(
        resources,
        *scratchRequest,
        outObservation);

    ScratchSheddingResult sheddingResult{};
    std::string sheddingError;
    if (!outObservation.overTargetButNotReducible) {
        (void)execute_scratch_shedding_primitive(
            transaction,
            resources,
            commandName,
            *scratchRequest,
            invocation,
            outObservation,
            sheddingResult,
            sheddingError);
        outObservation.sheddingAttempted = sheddingResult.attempted;
        outObservation.sheddingProgressed = sheddingResult.progressed;
        outObservation.sheddingPartialFailure = sheddingResult.partialFailure;
        outObservation.sheddingTargetReached = sheddingResult.targetReached;
        outObservation.sheddingOrphanedSharedTmpRetired =
            sheddingResult.orphanedSharedTmpRetired;
        outObservation.shedRetiredActionCount = sheddingResult.retiredActionCount;
        outObservation.shedRetiredLiveBytes = sheddingResult.retiredLiveBytes;
        if (sheddingResult.progressed || sheddingResult.partialFailure) {
            publish_manager_memory_snapshot(snapshot_manager_memory(resources));
        }
    }

    {
        ScratchNormalizationState& state = scratch_normalization_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        ScratchNormalizationContextState& contextState =
            state.byContext[transaction.snapshot.deviceContextKey];
        if (!outObservation.sheddingPartialFailure && outObservation.overTargetButNotReducible) {
            contextState.noShedCache.valid = true;
            contextState.noShedCache.requestDescriptorGeneration =
                outObservation.requestDescriptorGeneration;
            contextState.noShedCache.retainedScratchGeneration =
                outObservation.retainedScratchGeneration;
            contextState.noShedCache.scratchTargetBytes =
                outObservation.scratchTargetBytes;
        } else {
            contextState.noShedCache = ScratchNormalizationNoShedCache{};
        }
    }

    trace_scratch_checkpoint_event(
        transaction,
        commandName,
        invocation,
        outObservation,
        outObservation.sheddingPartialFailure
            ? "scratch_shedding_partial_failure"
            : (outObservation.sheddingProgressed
                   ? "live_retained_scratch_retired"
                   : (outObservation.overTargetButNotReducible
                          ? "over_target_not_reducible_for_request"
                          : "eligible_live_retained_scratch_observed")));
    outError.clear();
    return true;
}

std::uint64_t transient_reservation_cap_bytes(
    const ResourceManagerConfigEffective& cfg,
    const ResolvedPressurePolicy& policy) noexcept {
    if (policy.softTargetBytes == 0) {
        return kTransientReservationCapDefaultBytes;
    }
    const std::uint64_t quarterTarget = policy.softTargetBytes / 4ull;
    return std::max<std::uint64_t>(
        kTransientReservationThresholdDefaultBytes,
        std::min<std::uint64_t>(kTransientReservationCapDefaultBytes, quarterTarget));
}

std::uint64_t transient_reservation_threshold_bytes(
    const ResourceManagerConfigEffective& cfg,
    const ResolvedPressurePolicy& policy) noexcept {
    const std::uint64_t capBytes = transient_reservation_cap_bytes(cfg, policy);
    if (capBytes == 0) {
        return 0;
    }
    return std::min<std::uint64_t>(kTransientReservationThresholdDefaultBytes, capBytes);
}

std::uint64_t builder_reservation_cap_bytes(
    const ResourceManagerConfigEffective& cfg,
    BuilderReservationTier tier) noexcept {
    std::uint32_t configCapMb = 0;
    std::uint64_t defaultCapBytes = kScratchBuilderReservationCapDefaultBytes;
    switch (tier) {
        case BuilderReservationTier::Scratch:
            configCapMb = cfg.scratchBuilderBytesInFlightLimitMB;
            defaultCapBytes = kScratchBuilderReservationCapDefaultBytes;
            break;
        case BuilderReservationTier::Lut:
            configCapMb = cfg.lutBuilderBytesInFlightLimitMB;
            defaultCapBytes = kLutBuilderReservationCapDefaultBytes;
            break;
        case BuilderReservationTier::Graph:
            configCapMb = cfg.graphBuilderBytesInFlightLimitMB;
            defaultCapBytes = kGraphBuilderReservationCapDefaultBytes;
            break;
        default:
            configCapMb = cfg.scratchBuilderBytesInFlightLimitMB;
            defaultCapBytes = kScratchBuilderReservationCapDefaultBytes;
            break;
    }

    std::uint64_t capBytes = 0;
    if (!mul_u64_checked(static_cast<std::uint64_t>(configCapMb), kBytesPerMiB, capBytes)) {
        capBytes = std::numeric_limits<std::uint64_t>::max();
    }
    if (capBytes == 0) {
        capBytes = defaultCapBytes;
    }
    return std::max<std::uint64_t>(kBuilderReservationThresholdDefaultBytes, capBytes);
}

std::uint64_t builder_reservation_threshold_bytes(
    const ResourceManagerConfigEffective& cfg,
    BuilderReservationTier tier) noexcept {
    const std::uint64_t capBytes = builder_reservation_cap_bytes(cfg, tier);
    if (capBytes == 0) {
        return 0;
    }
    return std::min<std::uint64_t>(kBuilderReservationThresholdDefaultBytes, capBytes);
}

bool builder_context_has_inflight(const BuilderReservationContextState& contextState) noexcept {
    return contextState.inFlightScratchBytes > 0 ||
           contextState.inFlightLutBytes > 0 ||
           contextState.inFlightGraphBytes > 0;
}

std::uint64_t upload_reservation_cap_bytes(
    const ResourceManagerConfigEffective& cfg,
    const ResolvedPressurePolicy& policy) noexcept {
    std::uint64_t configCapBytes = 0;
    if (!mul_u64_checked(
            static_cast<std::uint64_t>(cfg.uploadBytesInFlightLimitMB),
            kBytesPerMiB,
            configCapBytes)) {
        configCapBytes = std::numeric_limits<std::uint64_t>::max();
    }
    if (configCapBytes == 0) {
        configCapBytes = kUploadReservationCapDefaultBytes;
    }
    if (policy.softTargetBytes == 0) {
        return std::max<std::uint64_t>(kUploadReservationThresholdDefaultBytes, configCapBytes);
    }
    const std::uint64_t quarterTarget = policy.softTargetBytes / 4ull;
    const std::uint64_t boundedCap = std::min<std::uint64_t>(configCapBytes, quarterTarget);
    return std::max<std::uint64_t>(kUploadReservationThresholdDefaultBytes, boundedCap);
}

std::uint64_t upload_reservation_threshold_bytes(
    const ResourceManagerConfigEffective& cfg,
    const ResolvedPressurePolicy& policy) noexcept {
    const std::uint64_t capBytes = upload_reservation_cap_bytes(cfg, policy);
    if (capBytes == 0) {
        return 0;
    }
    return std::min<std::uint64_t>(kUploadReservationThresholdDefaultBytes, capBytes);
}


void refill_upload_fairness_tokens(
    UploadReservationContextState::FairnessEntry& entry,
    const ResourceManagerConfigEffective& cfg,
    std::uint64_t nowMs) noexcept {
    const std::uint32_t sharedPerTick = std::max<std::uint32_t>(1u, cfg.uploadFairnessTokensPerTick);
    const std::uint32_t criticalPerTick = std::min<std::uint32_t>(
        std::max<std::uint32_t>(1u, cfg.criticalUploadReservedTokens),
        sharedPerTick);

    if (entry.lastRefillMs == 0) {
        entry.sharedTokens = sharedPerTick;
        entry.criticalTokens = criticalPerTick;
        entry.lastRefillMs = nowMs;
        return;
    }

    const std::uint64_t elapsedMs = admission_elapsed_ms_since(nowMs, entry.lastRefillMs);
    if (elapsedMs < kUploadFairnessTickMs) {
        return;
    }

    entry.sharedTokens = sharedPerTick;
    entry.criticalTokens = criticalPerTick;
    entry.lastRefillMs = nowMs;
}

void refill_builder_fairness_tokens(
    BuilderReservationContextState::FairnessEntry& entry,
    const ResourceManagerConfigEffective& cfg,
    std::uint64_t nowMs) noexcept {
    const std::uint32_t sharedPerTick = std::max<std::uint32_t>(1u, cfg.builderFairnessTokensPerTick);
    const std::uint32_t criticalPerTick = std::min<std::uint32_t>(
        std::max<std::uint32_t>(1u, cfg.criticalBuilderReservedTokens),
        sharedPerTick);

    if (entry.lastRefillMs == 0) {
        entry.sharedTokens = sharedPerTick;
        entry.criticalTokens = criticalPerTick;
        entry.lastRefillMs = nowMs;
        return;
    }

    const std::uint64_t elapsedMs = admission_elapsed_ms_since(nowMs, entry.lastRefillMs);
    if (elapsedMs < kBuilderFairnessTickMs) {
        return;
    }

    entry.sharedTokens = sharedPerTick;
    entry.criticalTokens = criticalPerTick;
    entry.lastRefillMs = nowMs;
}


bool try_acquire_scratch_policy_claim_impl(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const DeviceContextKey& contextKey,
    const ScratchBucketKey& bucketKey,
    std::size_t requestBytes,
    bool criticalCurrentFrame,
    ScratchPolicyClaim& outClaim,
    ScratchPolicySnapshot& outSnapshot,
    ReservationAttemptInfo& outReservation) {
    outReservation = ReservationAttemptInfo{};
    ScratchPolicyState& state = scratch_policy_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    outReservation.stateVersion = state.stateVersion;

    ScratchContextState& contextState = state.byContext[contextKey];
    const std::uint64_t nowMs = monotonic_time_ms();
    trim_large_frame_quarantine_decay_locked(contextState, nowMs);

    ScratchBucketEntry& bucketEntry = contextState.buckets[bucketKey];
    bucketEntry.attemptCount += 1;
    telemetry_counter_add(global_state().scratchBucketAcquireAttempts, 1);

    if (requestBytes == 0) {
        bucketEntry.reuseEvents += 1;
        telemetry_counter_add(global_state().scratchReuseEvents, 1);
        global_state().transientNonManagerBytes.store(state.totalInFlightBytes, std::memory_order_relaxed);
        snapshot_bucket_state_locked(contextState, bucketEntry, bucketKey, outSnapshot);
        return true;
    }

    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    const ResolvedPressurePolicy& policy = transaction.resolvedPressurePolicy;
    outReservation.considered = true;
    outReservation.bytesInFlight = state.totalInFlightBytes;
    outReservation.capBytes = transient_reservation_cap_bytes(cfg, policy);
    outReservation.thresholdBytes = transient_reservation_threshold_bytes(cfg, policy);

    ReservationDecision reservationDecision{};
    if (static_cast<std::uint64_t>(requestBytes) < outReservation.thresholdBytes) {
        reservationDecision.granted = true;
        reservationDecision.reason = "below_threshold";
    } else {
        ReservationInput reservationInput{};
        reservationInput.kind = ReservationKind::TransientNonManager;
        reservationInput.requestBytes = static_cast<std::uint64_t>(requestBytes);
        reservationInput.bytesInFlight = state.totalInFlightBytes;
        reservationInput.capBytes = outReservation.capBytes;
        reservationInput.criticalCurrentFrame = criticalCurrentFrame;
        reservationDecision = classify_reservation(reservationInput);
    }
    outReservation.decision = reservationDecision;
    telemetry_counter_add(global_state().transientReservationRequests, 1);
    if (!reservationDecision.granted) {
        if (reservationDecision.shouldWait) {
            telemetry_counter_add(global_state().transientReservationDeferred, 1);
        } else {
            telemetry_counter_add(global_state().transientReservationDenied, 1);
        }
        trace_transient_reservation_decision(
            transaction,
            ReservationDecisionTrace{
                .commandName = commandName,
                .reason = reservation_wait_outcome_reason(reservationDecision.shouldWait),
                .decision = &reservationDecision,
                .requestBytes = static_cast<std::uint64_t>(requestBytes),
                .bytesInFlight = outReservation.bytesInFlight,
                .capBytes = outReservation.capBytes,
                .thresholdBytes = outReservation.thresholdBytes,
                .waitMs = static_cast<int>(reservationDecision.waitMs),
                .criticalCurrentFrame = criticalCurrentFrame});
        snapshot_bucket_state_locked(contextState, bucketEntry, bucketKey, outSnapshot);
        return false;
    }

    const std::size_t effectiveCapBytes = effective_temp_scratch_bytes_cap(requestBytes);
    const bool setsOk = bucketEntry.inFlightSets < kMaxTempScratchSets;
    const bool bytesOk = requestBytes <=
                         (effectiveCapBytes - std::min(bucketEntry.inFlightBytes, effectiveCapBytes));
    if (!setsOk || !bytesOk) {
        bucketEntry.exhaustedCount += 1;
        telemetry_counter_add(global_state().scratchBucketExhaustedEvents, 1);

        const bool starvationNow =
            (bucketEntry.attemptCount >= 8) &&
            (bucketEntry.exhaustedCount * 4 >= bucketEntry.attemptCount);
        if (starvationNow && !bucketEntry.starvationLatched) {
            bucketEntry.starvationLatched = true;
            telemetry_counter_add(global_state().scratchBucketStarvationEvents, 1);
        } else if (!starvationNow) {
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
        } else {
            contextState.totalInFlightBytes = std::numeric_limits<std::size_t>::max();
        }
    }
    {
        std::uint64_t nextGlobalBytes = 0;
        if (add_u64_checked(state.totalInFlightBytes, static_cast<std::uint64_t>(requestBytes), nextGlobalBytes)) {
            state.totalInFlightBytes = nextGlobalBytes;
        } else {
            state.totalInFlightBytes = std::numeric_limits<std::uint64_t>::max();
        }
    }
    telemetry_counter_add(global_state().scratchAllocGrowthEvents, 1);
    bucketEntry.starvationLatched = false;
    global_state().transientNonManagerBytes.store(state.totalInFlightBytes, std::memory_order_relaxed);
    telemetry_counter_add(global_state().transientReservationGranted, 1);

    outClaim.contextKey = contextKey;
    outClaim.bucketKey = bucketKey;
    outClaim.bytes = requestBytes;
    outClaim.acquired = true;

    snapshot_bucket_state_locked(contextState, bucketEntry, bucketKey, outSnapshot);
    return true;
}

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
    try {
        return try_acquire_scratch_policy_claim_impl(
            transaction,
            commandName,
            contextKey,
            bucketKey,
            requestBytes,
            criticalCurrentFrame,
            outClaim,
            outSnapshot,
            outReservation);
    } catch (...) {
        JuicerLogging::discard_current_exception();
        outClaim = ScratchPolicyClaim{};
        outSnapshot = ScratchPolicySnapshot{};
        outReservation = ReservationAttemptInfo{};
        return false;
    }
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
    std::uint64_t waitStartMs = 0;
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
            if (reservation.considered) {
                trace_transient_reservation_decision(
                    transaction,
                    ReservationDecisionTrace{
                        .commandName = commandName,
                        .reason = waitedMs > 0
                                      ? "admit_after_wait"
                                      : trace_or(reservation.decision.reason, "granted"),
                        .decision = &reservation.decision,
                        .requestBytes = static_cast<std::uint64_t>(requestBytes),
                        .bytesInFlight = reservation.bytesInFlight,
                        .capBytes = reservation.capBytes,
                        .thresholdBytes = reservation.thresholdBytes,
                        .waitMs = waitedMs,
                        .criticalCurrentFrame = criticalCurrentFrame});
            }
            if (waitedMs > 0) {
                telemetry_counter_add(state.scratchPolicyWaitEvents, 1);
                if (criticalCurrentFrame) {
                    const std::uint64_t waitedMsU64 =
                        static_cast<std::uint64_t>(std::max(waitedMs, 0));
                    telemetry_counter_add(state.criticalBuilderWaitEvents, 1);
                    telemetry_counter_add(state.criticalBuilderWaitTotalMs, waitedMsU64);
                    trace_lane_wait_event(
                        transaction,
                        LaneWaitEventTrace{
                            .commandName = commandName,
                            .outcome = "admit_after_wait",
                            .reason = "scratch_policy_wait",
                            .lane = PressureLane::Builder,
                            .waitMs = waitedMs,
                            .criticalCurrentFrame = criticalCurrentFrame});
                }
                trace_scratch_policy_decision(
                    transaction,
                    ScratchPolicyDecisionTrace{
                        .commandName = commandName,
                        .result = "admit_after_wait",
                        .snapshot = &snapshot,
                        .requestBytes = requestBytes,
                        .waitMs = waitedMs});
            } else if (requestBytes == 0) {
                trace_scratch_policy_decision(
                    transaction,
                    ScratchPolicyDecisionTrace{
                        .commandName = commandName,
                        .result = "reuse_hit",
                        .snapshot = &snapshot,
                        .requestBytes = requestBytes,
                        .waitMs = waitedMs});
            }
            return true;
        }

        if (waitedMs >= waitBudgetMs) {
            telemetry_counter_add(state.scratchPolicyExhaustedEvents, 1);
            if (criticalCurrentFrame) {
                const std::uint64_t waitedMsU64 =
                    static_cast<std::uint64_t>(std::max(waitedMs, 0));
                telemetry_counter_add(state.criticalBuilderWaitEvents, 1);
                telemetry_counter_add(state.criticalBuilderWaitTotalMs, waitedMsU64);
                telemetry_counter_add(state.criticalLaneStarvationEvents, 1);
                trace_lane_wait_event(
                    transaction,
                    LaneWaitEventTrace{
                        .commandName = commandName,
                        .outcome = "starved",
                        .reason = admission_reservation_reason_or(reservation, "scratch_exhausted"),
                        .lane = PressureLane::Builder,
                        .waitMs = waitedMs,
                        .criticalCurrentFrame = criticalCurrentFrame});
            }
            trace_scratch_policy_decision(
                transaction,
                ScratchPolicyDecisionTrace{
                    .commandName = commandName,
                    .result = "exhausted",
                    .snapshot = &snapshot,
                    .requestBytes = requestBytes,
                    .waitMs = waitedMs});
            if (reservation.considered && !reservation.decision.granted) {
                outError = std::string(kReservationDeferredPrefix) + " command=" + trace_or_unknown(commandName) + " work_class=" + to_cstr(workClass) + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes)) + " in_flight_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.bytesInFlight)) + " cap_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.capBytes)) + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.thresholdBytes)) + " critical_current_frame=" + std::to_string(admission_bool_u32(criticalCurrentFrame)) + " wait_ms=" + std::to_string(waitedMs) + " wait_budget_ms=" + std::to_string(waitBudgetMs) + " reason=" + trace_or_unspecified(reservation.decision.reason);
            } else {
                outError = std::string(kScratchExhaustedPrefix) + " command=" + trace_or_unknown(commandName) + " work_class=" + to_cstr(workClass) + " bucket_w=" + std::to_string(bucketKey.widthBucket) + " bucket_h=" + std::to_string(bucketKey.heightBucket) + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes)) + " in_flight_bytes=" + std::to_string(static_cast<unsigned long long>(snapshot.inFlightBytes)) + " in_flight_sets=" + std::to_string(snapshot.inFlightSets) + " bucket_exhausted=" + std::to_string(snapshot.bucketExhausted) + " bucket_attempts=" + std::to_string(snapshot.bucketAttempts) + " wait_ms=" + std::to_string(waitedMs) + " wait_budget_ms=" + std::to_string(waitBudgetMs);
            }
            return false;
        }

        if (waitStartMs == 0) {
            waitStartMs = monotonic_time_ms();
        }
        wait_for_scratch_policy_state_change_or_timeout(
            reservation.stateVersion,
            kScratchWaitStepMs);
        waitedMs = admission_elapsed_ms_since_clamped(waitStartMs);
    }
}

bool try_acquire_builder_reservation_claim_impl(
    const SubmissionTransaction& transaction,
    BuilderReservationTier tier,
    std::uint64_t requestBytes,
    bool criticalCurrentFrame,
    BuilderReservationClaim& outClaim,
    ReservationAttemptInfo& outReservation) {
    outClaim = BuilderReservationClaim{};
    outReservation = ReservationAttemptInfo{};

    if (requestBytes == 0) {
        return true;
    }

    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    BuilderReservationState& state = builder_reservation_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    outReservation.instanceToken = transaction.snapshot.instanceToken.value;
    outReservation.stateVersion = state.stateVersion;

    outReservation.considered = true;
    std::uint64_t& tierTotalBytes = builder_total_bytes_for_tier(state, tier);
    outReservation.bytesInFlight = tierTotalBytes;
    outReservation.capBytes = builder_reservation_cap_bytes(cfg, tier);
    outReservation.thresholdBytes = builder_reservation_threshold_bytes(cfg, tier);

    ReservationDecision reservationDecision{};
    if (requestBytes < outReservation.thresholdBytes) {
        reservationDecision.granted = true;
        reservationDecision.reason = "below_threshold";
    } else {
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
    } else if (fairness.sharedTokens > 0) {
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
    } else {
        contextTierBytes = std::numeric_limits<std::uint64_t>::max();
    }
    std::uint64_t nextTotalBytes = 0;
    if (add_u64_checked(tierTotalBytes, requestBytes, nextTotalBytes)) {
        tierTotalBytes = nextTotalBytes;
    } else {
        tierTotalBytes = std::numeric_limits<std::uint64_t>::max();
    }
    builder_global_gauge_for_tier(global_state(), tier).store(tierTotalBytes, std::memory_order_relaxed);
    outClaim.contextKey = transaction.snapshot.deviceContextKey;
    outClaim.tier = tier;
    outClaim.bytes = requestBytes;
    outClaim.acquired = true;
    return true;
}

bool try_acquire_builder_reservation_claim(
    const SubmissionTransaction& transaction,
    BuilderReservationTier tier,
    std::uint64_t requestBytes,
    bool criticalCurrentFrame,
    BuilderReservationClaim& outClaim,
    ReservationAttemptInfo& outReservation) noexcept {
    try {
        return try_acquire_builder_reservation_claim_impl(
            transaction,
            tier,
            requestBytes,
            criticalCurrentFrame,
            outClaim,
            outReservation);
    } catch (...) {
        JuicerLogging::discard_current_exception();
        outClaim = BuilderReservationClaim{};
        outReservation = ReservationAttemptInfo{};
        return false;
    }
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
    telemetry_counter_add(state.builderReservationRequests, 1);

    ReservationAttemptInfo reservation{};
    int waitedMs = 0;
    std::uint64_t waitStartMs = 0;
    while (true) {
        if (try_acquire_builder_reservation_claim(
                transaction,
                tier,
                requestBytes,
                criticalCurrentFrame,
                outClaim,
                reservation)) {
            telemetry_counter_add(state.builderReservationGranted, 1);
            if (waitedMs > 0) {
                telemetry_counter_add(state.builderFairnessWaitEvents, 1);
                if (criticalCurrentFrame) {
                    const std::uint64_t waitedMsU64 =
                        static_cast<std::uint64_t>(std::max(waitedMs, 0));
                    telemetry_counter_add(state.criticalBuilderWaitEvents, 1);
                    telemetry_counter_add(state.criticalBuilderWaitTotalMs, waitedMsU64);
                    trace_lane_wait_event(
                        transaction,
                        LaneWaitEventTrace{
                            .commandName = commandName,
                            .outcome = "admit_after_wait",
                            .reason = trace_or(reservation.decision.reason, "waited"),
                            .lane = PressureLane::Builder,
                            .waitMs = waitedMs,
                            .criticalCurrentFrame = criticalCurrentFrame});
                }
                trace_builder_reservation_decision(
                    transaction,
                    tier,
                    ReservationDecisionTrace{
                        .commandName = commandName,
                        .reason = "admit_after_wait",
                        .decision = &reservation.decision,
                        .requestBytes = requestBytes,
                        .bytesInFlight = reservation.bytesInFlight,
                        .capBytes = reservation.capBytes,
                        .thresholdBytes = reservation.thresholdBytes,
                        .instanceToken = reservation.instanceToken,
                        .sharedTokens = reservation.sharedTokens,
                        .criticalTokens = reservation.criticalTokens,
                        .waitMs = waitedMs,
                        .criticalCurrentFrame = criticalCurrentFrame});
            } else if (reservation.decision.reason &&
                       std::string_view(reservation.decision.reason) == "critical_last_resort") {
                trace_builder_reservation_decision(
                    transaction,
                    tier,
                    ReservationDecisionTrace{
                        .commandName = commandName,
                        .reason = "critical_last_resort",
                        .decision = &reservation.decision,
                        .requestBytes = requestBytes,
                        .bytesInFlight = reservation.bytesInFlight,
                        .capBytes = reservation.capBytes,
                        .thresholdBytes = reservation.thresholdBytes,
                        .instanceToken = reservation.instanceToken,
                        .sharedTokens = reservation.sharedTokens,
                        .criticalTokens = reservation.criticalTokens,
                        .waitMs = waitedMs,
                        .criticalCurrentFrame = criticalCurrentFrame});
            } else if (JTRACE_ENABLED(3)) {
                trace_builder_reservation_decision(
                    transaction,
                    tier,
                    ReservationDecisionTrace{
                        .commandName = commandName,
                        .reason = "admitted",
                        .decision = &reservation.decision,
                        .requestBytes = requestBytes,
                        .bytesInFlight = reservation.bytesInFlight,
                        .capBytes = reservation.capBytes,
                        .thresholdBytes = reservation.thresholdBytes,
                        .instanceToken = reservation.instanceToken,
                        .sharedTokens = reservation.sharedTokens,
                        .criticalTokens = reservation.criticalTokens,
                        .waitMs = waitedMs,
                        .criticalCurrentFrame = criticalCurrentFrame});
            }
            return true;
        }

        const ReservationDecision& decision = reservation.decision;
        const bool fairnessDeferred =
            (decision.reason && std::string_view(decision.reason) == "fairness_tokens_exhausted");
        if (fairnessDeferred) {
            telemetry_counter_add(state.builderFairnessTokenDeferred, 1);
        }
        if (!decision.shouldWait) {
            telemetry_counter_add(state.builderReservationDenied, 1);
            if (criticalCurrentFrame) {
                const std::uint64_t waitedMsU64 =
                    static_cast<std::uint64_t>(std::max(waitedMs, 0));
                telemetry_counter_add(state.criticalBuilderWaitEvents, 1);
                telemetry_counter_add(state.criticalBuilderWaitTotalMs, waitedMsU64);
                telemetry_counter_add(state.criticalLaneStarvationEvents, 1);
                trace_lane_wait_event(
                    transaction,
                    LaneWaitEventTrace{
                        .commandName = commandName,
                        .outcome = "denied",
                        .reason = trace_or(decision.reason, "reservation_denied"),
                        .lane = PressureLane::Builder,
                        .waitMs = waitedMs,
                        .criticalCurrentFrame = criticalCurrentFrame});
            }
            trace_builder_reservation_decision(
                transaction,
                tier,
                ReservationDecisionTrace{
                    .commandName = commandName,
                    .reason = "denied",
                    .decision = &decision,
                    .requestBytes = requestBytes,
                    .bytesInFlight = reservation.bytesInFlight,
                    .capBytes = reservation.capBytes,
                    .thresholdBytes = reservation.thresholdBytes,
                    .instanceToken = reservation.instanceToken,
                    .sharedTokens = reservation.sharedTokens,
                    .criticalTokens = reservation.criticalTokens,
                    .waitMs = waitedMs,
                    .criticalCurrentFrame = criticalCurrentFrame});
            outError = std::string(kReservationDeferredPrefix) + " command=" + trace_or_unknown(commandName) + " kind=" + to_cstr(ReservationKind::BuilderWork) + " tier=" + to_cstr(tier) + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes)) + " in_flight_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.bytesInFlight)) + " cap_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.capBytes)) + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.thresholdBytes)) + " critical_current_frame=" + std::to_string(admission_bool_u32(criticalCurrentFrame)) + " wait_ms=" + std::to_string(waitedMs) + " wait_budget_ms=" + std::to_string(kBuilderReservationWaitMaxMs) + " reason=" + trace_or_unspecified(decision.reason);
            return false;
        }

        if (waitedMs >= kBuilderReservationWaitMaxMs) {
            telemetry_counter_add(state.builderReservationDeferred, 1);
            if (criticalCurrentFrame) {
                telemetry_counter_add(state.builderReservationBypass, 1);
                if (fairnessDeferred) {
                    telemetry_counter_add(state.builderFairnessTokenBypass, 1);
                }
                const std::uint64_t waitedMsU64 =
                    static_cast<std::uint64_t>(std::max(waitedMs, 0));
                telemetry_counter_add(state.criticalBuilderWaitEvents, 1);
                telemetry_counter_add(state.criticalBuilderWaitTotalMs, waitedMsU64);
                telemetry_counter_add(state.criticalLaneStarvationEvents, 1);
                trace_lane_wait_event(
                    transaction,
                    LaneWaitEventTrace{
                        .commandName = commandName,
                        .outcome = fairness_bypass_reason(fairnessDeferred),
                        .reason = trace_or(decision.reason, "wait_budget_reached"),
                        .lane = PressureLane::Builder,
                        .waitMs = waitedMs,
                        .criticalCurrentFrame = criticalCurrentFrame});
                trace_builder_reservation_decision(
                    transaction,
                    tier,
                    ReservationDecisionTrace{
                        .commandName = commandName,
                        .reason = "deferred_bypass_wait_budget",
                        .decision = &decision,
                        .requestBytes = requestBytes,
                        .bytesInFlight = reservation.bytesInFlight,
                        .capBytes = reservation.capBytes,
                        .thresholdBytes = reservation.thresholdBytes,
                        .instanceToken = reservation.instanceToken,
                        .sharedTokens = reservation.sharedTokens,
                        .criticalTokens = reservation.criticalTokens,
                        .waitMs = waitedMs,
                        .criticalCurrentFrame = criticalCurrentFrame});
                return true;
            }
            trace_builder_reservation_decision(
                transaction,
                tier,
                ReservationDecisionTrace{
                    .commandName = commandName,
                    .reason = "deferred_wait_budget",
                    .decision = &decision,
                    .requestBytes = requestBytes,
                    .bytesInFlight = reservation.bytesInFlight,
                    .capBytes = reservation.capBytes,
                    .thresholdBytes = reservation.thresholdBytes,
                    .instanceToken = reservation.instanceToken,
                    .sharedTokens = reservation.sharedTokens,
                    .criticalTokens = reservation.criticalTokens,
                    .waitMs = waitedMs,
                    .criticalCurrentFrame = criticalCurrentFrame});
            outError = std::string(kReservationDeferredPrefix) + " command=" + trace_or_unknown(commandName) + " kind=" + to_cstr(ReservationKind::BuilderWork) + " tier=" + to_cstr(tier) + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes)) + " in_flight_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.bytesInFlight)) + " cap_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.capBytes)) + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.thresholdBytes)) + " critical_current_frame=" + std::to_string(admission_bool_u32(criticalCurrentFrame)) + " wait_ms=" + std::to_string(waitedMs) + " wait_budget_ms=" + std::to_string(kBuilderReservationWaitMaxMs) + " reason=" + trace_or(decision.reason, "wait_budget_reached");
            return false;
        }

        const int sleepMs = std::max<int>(
            1,
            std::min<int>(
                admission_wait_ms_or_one(decision.waitMs),
                kBuilderReservationWaitStepMs));
        if (waitStartMs == 0) {
            waitStartMs = monotonic_time_ms();
        }
        wait_for_builder_reservation_state_change_or_timeout(
            reservation.stateVersion,
            sleepMs);
        waitedMs = admission_elapsed_ms_since_clamped(waitStartMs);
    }
}

bool try_acquire_upload_reservation_claim_impl(
    const SubmissionTransaction& transaction,
    std::uint64_t requestBytes,
    bool criticalCurrentFrame,
    UploadReservationClaim& outClaim,
    ReservationAttemptInfo& outReservation) {
    outClaim = UploadReservationClaim{};
    outReservation = ReservationAttemptInfo{};

    if (requestBytes == 0) {
        return true;
    }

    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    const ResolvedPressurePolicy& policy = transaction.resolvedPressurePolicy;
    UploadReservationState& state = upload_reservation_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    outReservation.instanceToken = transaction.snapshot.instanceToken.value;
    outReservation.stateVersion = state.stateVersion;

    outReservation.considered = true;
    outReservation.bytesInFlight = state.totalInFlightBytes;
    outReservation.capBytes = upload_reservation_cap_bytes(cfg, policy);
    outReservation.thresholdBytes = upload_reservation_threshold_bytes(cfg, policy);

    ReservationDecision reservationDecision{};
    if (requestBytes < outReservation.thresholdBytes) {
        reservationDecision.granted = true;
        reservationDecision.reason = "below_threshold";
    } else {
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
    } else if (fairness.sharedTokens > 0) {
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
    } else {
        contextState.inFlightBytes = std::numeric_limits<std::uint64_t>::max();
    }
    std::uint64_t nextTotalBytes = 0;
    if (add_u64_checked(state.totalInFlightBytes, requestBytes, nextTotalBytes)) {
        state.totalInFlightBytes = nextTotalBytes;
    } else {
        state.totalInFlightBytes = std::numeric_limits<std::uint64_t>::max();
    }
    global_state().uploadBytesInFlight.store(state.totalInFlightBytes, std::memory_order_relaxed);
    outClaim.contextKey = transaction.snapshot.deviceContextKey;
    outClaim.bytes = requestBytes;
    outClaim.acquired = true;
    return true;
}

bool try_acquire_upload_reservation_claim(
    const SubmissionTransaction& transaction,
    std::uint64_t requestBytes,
    bool criticalCurrentFrame,
    UploadReservationClaim& outClaim,
    ReservationAttemptInfo& outReservation) noexcept {
    try {
        return try_acquire_upload_reservation_claim_impl(
            transaction,
            requestBytes,
            criticalCurrentFrame,
            outClaim,
            outReservation);
    } catch (...) {
        JuicerLogging::discard_current_exception();
        outClaim = UploadReservationClaim{};
        outReservation = ReservationAttemptInfo{};
        return false;
    }
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
    telemetry_counter_add(state.uploadReservationRequests, 1);

    ReservationAttemptInfo reservation{};
    int waitedMs = 0;
    std::uint64_t waitStartMs = 0;
    while (true) {
        if (try_acquire_upload_reservation_claim(
                transaction,
                requestBytes,
                criticalCurrentFrame,
                outClaim,
                reservation)) {
            telemetry_counter_add(state.uploadReservationGranted, 1);
            if (reservation.considered) {
                trace_upload_reservation_decision(
                    transaction,
                    ReservationDecisionTrace{
                        .commandName = commandName,
                        .reason = waitedMs > 0
                                      ? "admit_after_wait"
                                      : trace_or(reservation.decision.reason, "granted"),
                        .decision = &reservation.decision,
                        .requestBytes = requestBytes,
                        .bytesInFlight = reservation.bytesInFlight,
                        .capBytes = reservation.capBytes,
                        .thresholdBytes = reservation.thresholdBytes,
                        .instanceToken = reservation.instanceToken,
                        .sharedTokens = reservation.sharedTokens,
                        .criticalTokens = reservation.criticalTokens,
                        .waitMs = waitedMs,
                        .criticalCurrentFrame = criticalCurrentFrame});
            }
            if (waitedMs > 0) {
                telemetry_counter_add(state.uploadFairnessWaitEvents, 1);
                if (criticalCurrentFrame) {
                    const std::uint64_t waitedMsU64 =
                        static_cast<std::uint64_t>(std::max(waitedMs, 0));
                    telemetry_counter_add(state.criticalUploadWaitEvents, 1);
                    telemetry_counter_add(state.criticalUploadWaitTotalMs, waitedMsU64);
                    trace_lane_wait_event(
                        transaction,
                        LaneWaitEventTrace{
                            .commandName = commandName,
                            .outcome = "admit_after_wait",
                            .reason = trace_or(reservation.decision.reason, "waited"),
                            .lane = PressureLane::Upload,
                            .waitMs = waitedMs,
                            .criticalCurrentFrame = criticalCurrentFrame});
                }
            }
            return true;
        }

        const ReservationDecision& decision = reservation.decision;
        const bool fairnessDeferred =
            (decision.reason && std::string_view(decision.reason) == "fairness_tokens_exhausted");
        if (fairnessDeferred) {
            telemetry_counter_add(state.uploadFairnessTokenDeferred, 1);
        }
        if (!decision.shouldWait) {
            telemetry_counter_add(state.uploadReservationDenied, 1);
            if (criticalCurrentFrame) {
                const std::uint64_t waitedMsU64 =
                    static_cast<std::uint64_t>(std::max(waitedMs, 0));
                telemetry_counter_add(state.criticalUploadWaitEvents, 1);
                telemetry_counter_add(state.criticalUploadWaitTotalMs, waitedMsU64);
                telemetry_counter_add(state.criticalLaneStarvationEvents, 1);
                trace_lane_wait_event(
                    transaction,
                    LaneWaitEventTrace{
                        .commandName = commandName,
                        .outcome = "denied",
                        .reason = trace_or(decision.reason, "reservation_denied"),
                        .lane = PressureLane::Upload,
                        .waitMs = waitedMs,
                        .criticalCurrentFrame = criticalCurrentFrame});
            }
            trace_upload_reservation_decision(
                transaction,
                ReservationDecisionTrace{
                    .commandName = commandName,
                    .reason = "denied",
                    .decision = &decision,
                    .requestBytes = requestBytes,
                    .bytesInFlight = reservation.bytesInFlight,
                    .capBytes = reservation.capBytes,
                    .thresholdBytes = reservation.thresholdBytes,
                    .instanceToken = reservation.instanceToken,
                    .sharedTokens = reservation.sharedTokens,
                    .criticalTokens = reservation.criticalTokens,
                    .waitMs = waitedMs,
                    .criticalCurrentFrame = criticalCurrentFrame});
            outError = std::string(kReservationDeferredPrefix) + " command=" + trace_or_unknown(commandName) + " kind=" + to_cstr(ReservationKind::UploadCopy) + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes)) + " in_flight_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.bytesInFlight)) + " cap_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.capBytes)) + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.thresholdBytes)) + " critical_current_frame=" + std::to_string(admission_bool_u32(criticalCurrentFrame)) + " wait_ms=" + std::to_string(waitedMs) + " wait_budget_ms=" + std::to_string(kUploadReservationWaitMaxMs) + " reason=" + trace_or_unspecified(decision.reason);
            return false;
        }

        if (waitedMs >= kUploadReservationWaitMaxMs) {
            telemetry_counter_add(state.uploadReservationDeferred, 1);
            telemetry_counter_add(state.uploadReservationBypass, 1);
            if (fairnessDeferred) {
                telemetry_counter_add(state.uploadFairnessTokenBypass, 1);
            }
            if (criticalCurrentFrame) {
                const std::uint64_t waitedMsU64 =
                    static_cast<std::uint64_t>(std::max(waitedMs, 0));
                telemetry_counter_add(state.criticalUploadWaitEvents, 1);
                telemetry_counter_add(state.criticalUploadWaitTotalMs, waitedMsU64);
                if (fairnessDeferred) {
                    telemetry_counter_add(state.criticalLaneStarvationEvents, 1);
                }
                trace_lane_wait_event(
                    transaction,
                    LaneWaitEventTrace{
                        .commandName = commandName,
                        .outcome = fairness_bypass_reason(fairnessDeferred),
                        .reason = trace_or(decision.reason, "wait_budget_reached"),
                        .lane = PressureLane::Upload,
                        .waitMs = waitedMs,
                        .criticalCurrentFrame = criticalCurrentFrame});
            }
            trace_upload_reservation_decision(
                transaction,
                ReservationDecisionTrace{
                    .commandName = commandName,
                    .reason = "deferred_bypass_wait_budget",
                    .decision = &decision,
                    .requestBytes = requestBytes,
                    .bytesInFlight = reservation.bytesInFlight,
                    .capBytes = reservation.capBytes,
                    .thresholdBytes = reservation.thresholdBytes,
                    .instanceToken = reservation.instanceToken,
                    .sharedTokens = reservation.sharedTokens,
                    .criticalTokens = reservation.criticalTokens,
                    .waitMs = waitedMs,
                    .criticalCurrentFrame = criticalCurrentFrame});
            return true;
        }

        const int sleepMs = std::max<int>(
            1,
            std::min<int>(
                admission_wait_ms_or_one(decision.waitMs),
                kUploadReservationWaitStepMs));
        if (waitStartMs == 0) {
            waitStartMs = monotonic_time_ms();
        }
        wait_for_upload_reservation_state_change_or_timeout(
            reservation.stateVersion,
            sleepMs);
        waitedMs = admission_elapsed_ms_since_clamped(waitStartMs);
    }
}


void maybe_publish_manager_memory_snapshot(
    JuicerCuda::Resources& resources,
    bool enabled) noexcept {
    if (!enabled) {
        return;
    }
    publish_manager_memory_snapshot(snapshot_manager_memory(resources));
}

HeadroomTelemetry sample_headroom_telemetry(const SubmissionTransaction& transaction) noexcept {
    HeadroomTelemetry out{};
    const int targetDevice = transaction.snapshot.deviceContextKey.deviceId;
    if (targetDevice < 0) {
        return out;
    }

    const bool deviceContextReady = with_explicit_cuda_device(
        targetDevice,
        [&]() noexcept {
            out.sampledDeviceId = targetDevice;
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
            std::size_t freeBytes = 0;
            std::size_t totalBytes = 0;
            if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess) {
                return;
            }
            out.sampleSuccess = true;
            out.driverFreeBytes = static_cast<std::uint64_t>(
                std::min<std::size_t>(
                    freeBytes,
                    static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())));
            out.effectiveHeadroomBytes = out.driverFreeBytes;
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
            cudaMemPool_t defaultPool = nullptr;
            if (cudaDeviceGetDefaultMemPool(&defaultPool, targetDevice) == cudaSuccess &&
                defaultPool != nullptr) {
                std::size_t poolReservedBytes = 0;
                std::size_t poolUsedBytes = 0;
                const cudaError_t reservedErr = cudaMemPoolGetAttribute(
                    defaultPool,
                    cudaMemPoolAttrReservedMemCurrent,
                    &poolReservedBytes);
                const cudaError_t usedErr = cudaMemPoolGetAttribute(
                    defaultPool,
                    cudaMemPoolAttrUsedMemCurrent,
                    &poolUsedBytes);
                if (reservedErr == cudaSuccess && usedErr == cudaSuccess) {
                    out.poolTelemetryAvailable = true;
                    out.source = HeadroomSource::AllocatorPool;
                    out.allocatorPoolReservedBytes = static_cast<std::uint64_t>(
                        std::min<std::size_t>(
                            poolReservedBytes,
                            static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())));
                    out.allocatorPoolUsedBytes = static_cast<std::uint64_t>(
                        std::min<std::size_t>(
                            poolUsedBytes,
                            static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())));
                    if (out.allocatorPoolUsedBytes > out.allocatorPoolReservedBytes) {
                        out.allocatorPoolUsedBytes = out.allocatorPoolReservedBytes;
                    }

                    const std::uint64_t poolFreeBytes =
                        out.allocatorPoolReservedBytes - out.allocatorPoolUsedBytes;
                    std::uint64_t combinedHeadroom = out.driverFreeBytes;
                    if (!add_u64_checked(combinedHeadroom, poolFreeBytes, combinedHeadroom)) {
                        combinedHeadroom = std::numeric_limits<std::uint64_t>::max();
                    }
                    out.effectiveHeadroomBytes = combinedHeadroom;
                }
            }
#endif
#else
            (void)targetDevice;
#endif
        });
    if (!deviceContextReady) {
        return HeadroomTelemetry{};
    }
    return out;
}

std::uint64_t pressure_total_bytes(const PressureInput& input) noexcept {
    std::uint64_t total = input.managerResidentBytes;
    std::uint64_t next = 0;
    if (!add_u64_checked(total, input.retirePendingBytes, next)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    total = next;
    if (!add_u64_checked(total, input.transientNonManagerBytes, next)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return next;
}

bool reserve_crossed(const PressureInput& input) noexcept {
    if (input.softTargetBytes == 0) {
        return false;
    }
    return pressure_total_bytes(input) > input.softTargetBytes;
}

std::uint64_t compute_effective_reserve_target_bytes(
    const ResourceManagerConfigEffective& cfg,
    const ResolvedPressurePolicy& policy,
    std::uint64_t transientBytes) noexcept {
    std::uint64_t targetBytes = policy.reserveBytes;
    std::uint64_t transientBoundBytes = transientBytes;
    if (!add_u64_checked(transientBoundBytes, cfg.reserveSafetyMarginBytes, transientBoundBytes)) {
        transientBoundBytes = std::numeric_limits<std::uint64_t>::max();
    }
    targetBytes = std::max<std::uint64_t>(targetBytes, transientBoundBytes);
    if (policy.softTargetBytes > 0) {
        targetBytes = std::min<std::uint64_t>(targetBytes, policy.softTargetBytes);
    }
    return targetBytes;
}

std::uint64_t step_effective_reserve_bytes(
    const ResourceManagerConfigEffective& cfg,
    std::uint64_t currentBytes,
    std::uint64_t targetBytes) noexcept {
    if (currentBytes == targetBytes) {
        return currentBytes;
    }

    const std::uint64_t stepUp = std::max<std::uint64_t>(1u, cfg.reserveAdaptUpStepBytes);
    const std::uint64_t stepDown = std::max<std::uint64_t>(1u, cfg.reserveAdaptDownStepBytes);
    if (targetBytes > currentBytes) {
        const std::uint64_t delta = targetBytes - currentBytes;
        const std::uint64_t step = std::min<std::uint64_t>(stepUp, delta);
        std::uint64_t next = currentBytes;
        if (!add_u64_checked(next, step, next)) {
            next = targetBytes;
        }
        return next;
    }

    const std::uint64_t delta = currentBytes - targetBytes;
    const std::uint64_t step = std::min<std::uint64_t>(stepDown, delta);
    return currentBytes - step;
}

std::uint64_t active_burst_cap_bytes(
    const ResourceManagerConfigEffective& cfg,
    std::uint64_t softTargetBytes) noexcept {
    std::uint64_t pctCapBytes = 0;
    if (softTargetBytes > 0 && cfg.maxActiveBurstPctOfTarget > 0) {
        std::uint64_t weighted = 0;
        if (!mul_u64_checked(
                softTargetBytes,
                static_cast<std::uint64_t>(cfg.maxActiveBurstPctOfTarget),
                weighted)) {
            weighted = std::numeric_limits<std::uint64_t>::max();
        }
        pctCapBytes = weighted / 100ull;
    }

    if (cfg.maxActiveBurstBytes == 0) {
        return pctCapBytes;
    }
    if (pctCapBytes == 0) {
        return cfg.maxActiveBurstBytes;
    }
    return std::min<std::uint64_t>(cfg.maxActiveBurstBytes, pctCapBytes);
}

std::uint32_t decay_burst_debt_pct(
    std::uint32_t currentDebtPct,
    std::uint64_t elapsedMs,
    std::uint32_t halfLifeMs) noexcept {
    if (currentDebtPct == 0 || elapsedMs == 0 || halfLifeMs == 0) {
        return currentDebtPct;
    }

    const double exponent =
        -static_cast<double>(elapsedMs) / static_cast<double>(halfLifeMs);
    const double decayed = static_cast<double>(currentDebtPct) * std::exp2(exponent);
    if (!std::isfinite(decayed) || decayed <= 0.0) {
        return 0;
    }

    const long long rounded = std::llround(decayed);
    if (rounded <= 0) {
        return 0;
    }
    if (rounded >= 100) {
        return 100;
    }
    return static_cast<std::uint32_t>(rounded);
}

std::uint64_t graph_large_entry_threshold_bytes(const ResourceManagerConfigEffective& cfg) noexcept {
    std::uint64_t thresholdBytes = std::max<std::uint64_t>(
        1ull,
        admission_u64_or_default(cfg.graphLargeEntryThresholdBytes, kGraphLargeEntryThresholdDefaultBytes));
    if (cfg.managerSoftTargetBytes > 0) {
        const std::uint64_t pctThresholdBytes = cfg.managerSoftTargetBytes / 10ull;
        if (pctThresholdBytes > 0) {
            thresholdBytes = std::min<std::uint64_t>(thresholdBytes, pctThresholdBytes);
        }
    }
    return std::max<std::uint64_t>(1ull, thresholdBytes);
}

std::uint64_t graph_large_entry_quarantine_cap_bytes(const ResourceManagerConfigEffective& cfg) noexcept {
    std::uint64_t capBytes = admission_u64_or_default(
        cfg.graphLargeEntryQuarantineMaxBytes,
        kGraphLargeEntryQuarantineMaxBytesDefault);
    if (cfg.managerSoftTargetBytes > 0) {
        std::uint64_t weighted = 0;
        if (mul_u64_checked(cfg.managerSoftTargetBytes, 15ull, weighted)) {
            const std::uint64_t pctCapBytes = weighted / 100ull;
            if (pctCapBytes > 0) {
                capBytes = std::min<std::uint64_t>(capBytes, pctCapBytes);
            }
        }
    }
    return capBytes;
}

std::uint32_t graph_large_entry_quarantine_cap_entries(const ResourceManagerConfigEffective& cfg) noexcept {
    return std::max<std::uint32_t>(
        1u,
        admission_u32_or_default(
            cfg.graphLargeEntryQuarantineMaxEntries,
            kGraphLargeEntryQuarantineMaxEntriesDefault));
}

std::uint32_t pressure_poll_interval_ms_for_state(
    const ResourceManagerConfigEffective& cfg,
    PressureState state) noexcept {
    const std::uint32_t normal = std::max<std::uint32_t>(1u, cfg.pressurePollIntervalMsNormal);
    const std::uint32_t critical = std::max<std::uint32_t>(
        1u,
        std::min<std::uint32_t>(cfg.pressurePollIntervalMsCritical, normal));
    switch (state) {
        case PressureState::Normal:
            return normal;
        case PressureState::Constrained:
            return std::max<std::uint32_t>(critical, (normal + critical) / 2u);
        case PressureState::Critical:
        case PressureState::Emergency:
            return critical;
        default:
            return std::max<std::uint32_t>(1u, cfg.pressurePollIntervalMs);
    }
}

std::size_t estimate_optics_growth_bytes(
    JuicerCuda::Resources& resources,
    int width,
    int height,
    bool needBlurredScratch,
    bool needAuxScratch,
    bool needGrainScratch,
    bool needGrainSharedScratch,
    bool needGateMask) noexcept {
    const std::size_t planeBytes = plane_bytes_for_extent(width, height);
    if (planeBytes == 0 || planeBytes == std::numeric_limits<std::size_t>::max()) {
        return planeBytes;
    }

    std::lock_guard<std::mutex> lock(resources.m);
    const auto& scratch = resources.scannerScratch;
    const std::size_t requiredElements = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    const bool capacityMatch = scratch.capacityElements >= requiredElements;
    const bool haveBase = scratch.rgbR && scratch.rgbG && scratch.rgbB;
    const bool fullRebuild = !capacityMatch || !haveBase;

    std::size_t estimate = 0;
    auto addBytes = [&](std::size_t bytes, std::size_t multiplier = 1) {
        for (std::size_t i = 0; i < multiplier; ++i) {
            std::size_t next = 0;
            if (!add_bytes_checked(estimate, bytes, next)) {
                estimate = std::numeric_limits<std::size_t>::max();
                return;
            }
            estimate = next;
        }
    };
    auto addPlane = [&](std::size_t multiplier = 1) {
        addBytes(planeBytes, multiplier);
    };

    if (fullRebuild) {
        addPlane(3); // rgbR/rgbG/rgbB
    }

    const bool sharedTmpMatch = resources.sharedTmpPlane &&
                                resources.sharedTmpCapacityElements >= requiredElements;
    if (!sharedTmpMatch) {
        addPlane(); // shared tmp plane
    }

    if (needBlurredScratch && (fullRebuild || !scratch.blurred)) {
        addPlane();
    }
    if (needAuxScratch && (fullRebuild || !scratch.aux)) {
        addPlane();
    }
    if (needGrainScratch) {
        if (fullRebuild || !scratch.grainTmp) {
            addPlane();
        }
        if (fullRebuild || !scratch.grainTmpMid) {
            addPlane();
        }
        if (fullRebuild || !scratch.grainTmpCoarse) {
            addPlane();
        }
    }
    if (needGrainSharedScratch && (fullRebuild || !scratch.grainTmpShared)) {
        addPlane();
    }
    if (needGateMask) {
        const int gateWidth = (width + 1) / 2;
        const int gateHeight = (height + 1) / 2;
        const std::size_t gateBytes = plane_bytes_for_extent(gateWidth, gateHeight);
        if (gateBytes == 0 || gateBytes == std::numeric_limits<std::size_t>::max()) {
            return gateBytes;
        }
        const std::size_t requiredGateElements =
            static_cast<std::size_t>(gateWidth) * static_cast<std::size_t>(gateHeight);
        const bool gateCapacityMatch =
            scratch.gateMaskCapacityElements >= requiredGateElements;
        if (fullRebuild || !scratch.gateMask || !gateCapacityMatch) {
            addBytes(gateBytes);
        }
    }

    return estimate;
}

std::size_t estimate_spatial_dir_growth_bytes(
    JuicerCuda::Resources& resources,
    int width,
    int height) noexcept {
    const std::size_t planeBytes = plane_bytes_for_extent(width, height);
    if (planeBytes == 0 || planeBytes == std::numeric_limits<std::size_t>::max()) {
        return planeBytes;
    }

    std::lock_guard<std::mutex> lock(resources.m);
    const auto& scratch = resources.spatialDirScratch;
    const std::size_t requiredElements = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    const bool haveBase = scratch.corrY && scratch.corrM && scratch.corrC &&
                          scratch.mixY && scratch.mixM && scratch.mixC;
    const bool capacityMatch = scratch.capacityElements >= requiredElements;
    const bool sharedTmpMatch = resources.sharedTmpPlane &&
                                resources.sharedTmpCapacityElements >= requiredElements;

    std::size_t estimate = 0;
    if (!haveBase || !capacityMatch) {
        std::size_t next = 0;
        if (!add_bytes_checked(estimate, planeBytes, next)) {
            return std::numeric_limits<std::size_t>::max();
        }
        estimate = next;
        if (!add_bytes_checked(estimate, planeBytes, next)) {
            return std::numeric_limits<std::size_t>::max();
        }
        estimate = next;
        if (!add_bytes_checked(estimate, planeBytes, next)) {
            return std::numeric_limits<std::size_t>::max();
        }
        estimate = next;
        if (!add_bytes_checked(estimate, planeBytes, next)) {
            return std::numeric_limits<std::size_t>::max();
        }
        estimate = next;
        if (!add_bytes_checked(estimate, planeBytes, next)) {
            return std::numeric_limits<std::size_t>::max();
        }
        estimate = next;
        if (!add_bytes_checked(estimate, planeBytes, next)) {
            return std::numeric_limits<std::size_t>::max();
        }
        estimate = next;
    }
    if (!sharedTmpMatch) {
        std::size_t next = 0;
        if (!add_bytes_checked(estimate, planeBytes, next)) {
            return std::numeric_limits<std::size_t>::max();
        }
        estimate = next;
    }
    return estimate;
}

void trim_large_frame_quarantine_decay_locked(
    ScratchContextState& contextState,
    std::uint64_t nowMs) noexcept {
    std::size_t index = 0;
    std::uint64_t removedCount = 0;
    while (index < contextState.largeFrameQuarantine.size()) {
        const ScratchQuarantineEntry& entry = contextState.largeFrameQuarantine[index];
        const std::uint64_t ageMs = admission_elapsed_ms_since(nowMs, entry.touchedMs);
        if (ageMs < kLargeFrameQuarantineDecayMs) {
            ++index;
            continue;
        }
        if (contextState.largeFrameQuarantineBytes >= entry.bytes) {
            contextState.largeFrameQuarantineBytes -= entry.bytes;
        } else {
            contextState.largeFrameQuarantineBytes = 0;
        }
        contextState.largeFrameQuarantine.erase(
            contextState.largeFrameQuarantine.begin() + static_cast<std::ptrdiff_t>(index));
        ++removedCount;
    }
    if (removedCount > 0) {
        telemetry_counter_add(global_state().scratchLargeQuarantineDecayEvents, removedCount);
    }
}

void trim_large_frame_quarantine_caps_locked(
    ScratchContextState& contextState) noexcept {
    std::uint64_t trimmedCount = 0;
    while ((contextState.largeFrameQuarantineBytes > kLargeFrameQuarantineMaxBytes) ||
           (contextState.largeFrameQuarantine.size() > kLargeFrameQuarantineMaxEntries)) {
        if (contextState.largeFrameQuarantine.empty()) {
            contextState.largeFrameQuarantineBytes = 0;
            break;
        }

        std::size_t victimIndex = 0;
        ScratchQuarantineEntry victim = contextState.largeFrameQuarantine[0];
        for (std::size_t i = 1; i < contextState.largeFrameQuarantine.size(); ++i) {
            const ScratchQuarantineEntry& candidate = contextState.largeFrameQuarantine[i];
            const bool older = (candidate.touchedMs < victim.touchedMs) ||
                               ((candidate.touchedMs == victim.touchedMs) && (candidate.sequence < victim.sequence));
            if (older) {
                victim = candidate;
                victimIndex = i;
            }
        }

        if (contextState.largeFrameQuarantineBytes >= victim.bytes) {
            contextState.largeFrameQuarantineBytes -= victim.bytes;
        } else {
            contextState.largeFrameQuarantineBytes = 0;
        }
        contextState.largeFrameQuarantine.erase(
            contextState.largeFrameQuarantine.begin() + static_cast<std::ptrdiff_t>(victimIndex));
        ++trimmedCount;
    }

    if (trimmedCount > 0) {
        telemetry_counter_add(global_state().scratchLargeQuarantineTrimEvents, trimmedCount);
    }
}

void snapshot_bucket_state_locked(
    const ScratchContextState& contextState,
    const ScratchBucketEntry& bucketEntry,
    const ScratchBucketKey& bucketKey,
    ScratchPolicySnapshot& outSnapshot) noexcept {
    outSnapshot.inFlightBytes = bucketEntry.inFlightBytes;
    outSnapshot.inFlightSets = bucketEntry.inFlightSets;
    outSnapshot.quarantineBytes = contextState.largeFrameQuarantineBytes;
    outSnapshot.quarantineEntries = static_cast<std::uint32_t>(contextState.largeFrameQuarantine.size());
    outSnapshot.bucketAttempts = bucketEntry.attemptCount;
    outSnapshot.bucketExhausted = bucketEntry.exhaustedCount;
    outSnapshot.bucketAllocGrowth = bucketEntry.allocGrowthEvents;
    outSnapshot.bucketReuse = bucketEntry.reuseEvents;
    outSnapshot.starvationLatched = bucketEntry.starvationLatched;
    outSnapshot.bucketKey = bucketKey;
}

void release_scratch_policy_claim_impl(ScratchPolicyClaim& claim) {
    if (!claim.acquired) {
        return;
    }

    ScratchPolicyState& state = scratch_policy_state();
    bool notifyWaiters = false;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        auto contextIt = state.byContext.find(claim.contextKey);
        if (contextIt == state.byContext.end()) {
            claim = ScratchPolicyClaim{};
            return;
        }

        ScratchContextState& contextState = contextIt->second;
        auto bucketIt = contextState.buckets.find(claim.bucketKey);
        std::size_t releasedBytes = 0;
        if (bucketIt != contextState.buckets.end()) {
            ScratchBucketEntry& bucketEntry = bucketIt->second;
            if (bucketEntry.inFlightSets > 0) {
                --bucketEntry.inFlightSets;
            }
            if (bucketEntry.inFlightBytes >= claim.bytes) {
                bucketEntry.inFlightBytes -= claim.bytes;
                releasedBytes = claim.bytes;
            } else {
                releasedBytes = bucketEntry.inFlightBytes;
                bucketEntry.inFlightBytes = 0;
            }
        }
        if (contextState.totalInFlightBytes >= releasedBytes) {
            contextState.totalInFlightBytes -= releasedBytes;
        } else {
            contextState.totalInFlightBytes = 0;
        }
        if (state.totalInFlightBytes >= static_cast<std::uint64_t>(releasedBytes)) {
            state.totalInFlightBytes -= static_cast<std::uint64_t>(releasedBytes);
        } else {
            state.totalInFlightBytes = 0;
        }
        global_state().transientNonManagerBytes.store(state.totalInFlightBytes, std::memory_order_relaxed);

        const std::uint64_t nowMs = monotonic_time_ms();
        trim_large_frame_quarantine_decay_locked(contextState, nowMs);

        if (claim.bucketKey.largeFrame && claim.bytes > 0) {
            ScratchQuarantineEntry entry{};
            entry.key = claim.bucketKey;
            entry.bytes = claim.bytes;
            entry.touchedMs = nowMs;
            entry.sequence = contextState.nextQuarantineSequence++;
            contextState.largeFrameQuarantine.push_back(entry);
            contextState.largeFrameQuarantineBytes += claim.bytes;
            trim_large_frame_quarantine_caps_locked(contextState);
        }

        bump_reservation_state_version(state.stateVersion);
        claim = ScratchPolicyClaim{};
        notifyWaiters = true;
    }
    if (notifyWaiters) {
        state.cv.notify_all();
    }
}

void release_scratch_policy_claim(ScratchPolicyClaim& claim) noexcept {
    try {
        release_scratch_policy_claim_impl(claim);
    } catch (...) {
        JuicerLogging::discard_current_exception();
        claim = ScratchPolicyClaim{};
    }
}

class ScratchPolicyGuard {
public:
    explicit ScratchPolicyGuard(ScratchPolicyClaim claim) noexcept
        : _claim(claim) {
    }

    ~ScratchPolicyGuard() noexcept {
        release_scratch_policy_claim(_claim);
    }

    ScratchPolicyGuard(const ScratchPolicyGuard&) = delete;
    ScratchPolicyGuard& operator=(const ScratchPolicyGuard&) = delete;

private:
    ScratchPolicyClaim _claim{};
};

void release_builder_reservation_claim_impl(BuilderReservationClaim& claim) {
    if (!claim.acquired) {
        return;
    }

    BuilderReservationState& state = builder_reservation_state();
    ResourceManagerState& managerState = global_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);

        std::uint64_t& tierTotalBytes = builder_total_bytes_for_tier(state, claim.tier);
        auto contextIt = state.byContext.find(claim.contextKey);
        if (contextIt == state.byContext.end()) {
            if (tierTotalBytes >= claim.bytes) {
                tierTotalBytes -= claim.bytes;
            } else {
                tierTotalBytes = 0;
            }
            builder_global_gauge_for_tier(managerState, claim.tier).store(tierTotalBytes, std::memory_order_relaxed);
            bump_reservation_state_version(state.stateVersion);
            claim = BuilderReservationClaim{};
        } else {
            BuilderReservationContextState& contextState = contextIt->second;
            std::uint64_t& contextTierBytes = builder_context_bytes_for_tier(contextState, claim.tier);
            if (contextTierBytes >= claim.bytes) {
                contextTierBytes -= claim.bytes;
            } else {
                contextTierBytes = 0;
            }
            if (tierTotalBytes >= claim.bytes) {
                tierTotalBytes -= claim.bytes;
            } else {
                tierTotalBytes = 0;
            }

            if (!builder_context_has_inflight(contextState)) {
                state.byContext.erase(contextIt);
            }
            builder_global_gauge_for_tier(managerState, claim.tier).store(tierTotalBytes, std::memory_order_relaxed);
            bump_reservation_state_version(state.stateVersion);
            claim = BuilderReservationClaim{};
        }
    }
    state.cv.notify_all();
}

void release_builder_reservation_claim(BuilderReservationClaim& claim) noexcept {
    try {
        release_builder_reservation_claim_impl(claim);
    } catch (...) {
        JuicerLogging::discard_current_exception();
        claim = BuilderReservationClaim{};
    }
}

class BuilderReservationGuard {
public:
    explicit BuilderReservationGuard(BuilderReservationClaim claim) noexcept
        : _claim(claim) {
    }

    ~BuilderReservationGuard() noexcept {
        release_builder_reservation_claim(_claim);
    }

    BuilderReservationGuard(const BuilderReservationGuard&) = delete;
    BuilderReservationGuard& operator=(const BuilderReservationGuard&) = delete;

private:
    BuilderReservationClaim _claim{};
};

void release_upload_reservation_claim_impl(UploadReservationClaim& claim) {
    if (!claim.acquired) {
        return;
    }

    UploadReservationState& state = upload_reservation_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        auto contextIt = state.byContext.find(claim.contextKey);
        if (contextIt == state.byContext.end()) {
            if (state.totalInFlightBytes >= claim.bytes) {
                state.totalInFlightBytes -= claim.bytes;
            } else {
                state.totalInFlightBytes = 0;
            }
            global_state().uploadBytesInFlight.store(state.totalInFlightBytes, std::memory_order_relaxed);
            bump_reservation_state_version(state.stateVersion);
            claim = UploadReservationClaim{};
        } else {
            UploadReservationContextState& contextState = contextIt->second;
            if (contextState.inFlightBytes >= claim.bytes) {
                contextState.inFlightBytes -= claim.bytes;
            } else {
                contextState.inFlightBytes = 0;
            }
            if (state.totalInFlightBytes >= claim.bytes) {
                state.totalInFlightBytes -= claim.bytes;
            } else {
                state.totalInFlightBytes = 0;
            }
            if (contextState.inFlightBytes == 0) {
                state.byContext.erase(contextIt);
            }
            global_state().uploadBytesInFlight.store(state.totalInFlightBytes, std::memory_order_relaxed);
            bump_reservation_state_version(state.stateVersion);
            claim = UploadReservationClaim{};
        }
    }
    state.cv.notify_all();
}

void release_upload_reservation_claim(UploadReservationClaim& claim) noexcept {
    try {
        release_upload_reservation_claim_impl(claim);
    } catch (...) {
        JuicerLogging::discard_current_exception();
        claim = UploadReservationClaim{};
    }
}

class UploadReservationGuard {
public:
    explicit UploadReservationGuard(UploadReservationClaim claim) noexcept
        : _claim(claim) {
    }

    ~UploadReservationGuard() noexcept {
        release_upload_reservation_claim(_claim);
    }

    UploadReservationGuard(const UploadReservationGuard&) = delete;
    UploadReservationGuard& operator=(const UploadReservationGuard&) = delete;

private:
    UploadReservationClaim _claim{};
};


PressureCheckpoint evaluate_pressure_checkpoint(
    const SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    std::size_t pendingGrowthBytes,
    bool forceSample,
    const char* commandName) {
    PressureCheckpoint checkpoint{};
    checkpoint.memory = snapshot_manager_memory(resources);

    const std::uint64_t graphActiveBytes =
        estimate_graph_cache_active_bytes_for_context(transaction.snapshot.deviceContextKey);
    std::uint64_t nextActiveBytes = 0;
    if (add_u64_checked(checkpoint.memory.activeBytes, graphActiveBytes, nextActiveBytes)) {
        checkpoint.memory.activeBytes = nextActiveBytes;
    } else {
        checkpoint.memory.activeBytes = std::numeric_limits<std::uint64_t>::max();
        checkpoint.memory.overflow = true;
    }
    std::uint64_t nextReclaimableBytes = 0;
    if (add_u64_checked(checkpoint.memory.reclaimableBytes, graphActiveBytes, nextReclaimableBytes)) {
        checkpoint.memory.reclaimableBytes = nextReclaimableBytes;
    } else {
        checkpoint.memory.reclaimableBytes = std::numeric_limits<std::uint64_t>::max();
        checkpoint.memory.overflow = true;
    }

    publish_manager_memory_snapshot(checkpoint.memory);

    const HeadroomTelemetry headroom = sample_headroom_telemetry(transaction);
    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    const ResolvedPressurePolicy& policy = transaction.resolvedPressurePolicy;
    ResourceManagerState& managerState = global_state();
    checkpoint.headroom = headroom;
    checkpoint.input.softTargetBytes = policy.softTargetBytes;
    checkpoint.input.reserveBytes = policy.reserveBytes;
    checkpoint.input.retirePendingBytes = checkpoint.memory.retirePendingBytes;
    checkpoint.input.transientNonManagerBytes = checkpoint.memory.transientNonManagerBytes;
    checkpoint.input.effectiveHeadroomBytes = headroom.effectiveHeadroomBytes;
    checkpoint.input.driverFreeBytes = headroom.driverFreeBytes;
    checkpoint.input.allocatorPoolReservedBytes = headroom.allocatorPoolReservedBytes;
    checkpoint.input.allocatorPoolUsedBytes = headroom.allocatorPoolUsedBytes;
    checkpoint.input.headroomSource = headroom.source;

    managerState.allocatorEffectiveHeadroomBytes.store(
        checkpoint.input.effectiveHeadroomBytes,
        std::memory_order_relaxed);
    managerState.allocatorPoolReservedBytes.store(
        checkpoint.input.allocatorPoolReservedBytes,
        std::memory_order_relaxed);
    managerState.allocatorPoolUsedBytes.store(
        checkpoint.input.allocatorPoolUsedBytes,
        std::memory_order_relaxed);

    std::uint64_t predictedResident = checkpoint.memory.activeBytes;
    std::uint64_t nextResident = 0;
    if (add_u64_checked(predictedResident, static_cast<std::uint64_t>(pendingGrowthBytes), nextResident)) {
        predictedResident = nextResident;
    } else {
        predictedResident = std::numeric_limits<std::uint64_t>::max();
        checkpoint.memory.overflow = true;
    }
    checkpoint.input.managerResidentBytes = predictedResident;
    checkpoint.input.freezeOpportunisticBelowReserve = cfg.freezeOpportunisticBelowReserve;
    bool headroomSourceSwitch = false;

    const std::uint64_t nowMs = monotonic_time_ms();
    PressurePolicyState& policyState = pressure_policy_state();
    {
        std::lock_guard<std::mutex> lock(policyState.mutex);
        PressureContextState& contextState = policyState.byContext[transaction.snapshot.deviceContextKey];
        const PressureState cadenceState =
            admission_pressure_state_or_default(contextState, PressureState::Normal);
        checkpoint.pollIntervalMs = pressure_poll_interval_ms_for_state(cfg, cadenceState);

        bool sampleDue = forceSample || !contextState.valid;
        if (!sampleDue && contextState.headroomSourceValid &&
            contextState.lastHeadroomSource != checkpoint.input.headroomSource) {
            sampleDue = true;
            headroomSourceSwitch = true;
        }
        if (!sampleDue) {
            const std::uint64_t elapsedMs =
                admission_elapsed_ms_since(nowMs, contextState.lastSampleMs);
            sampleDue = elapsedMs >= static_cast<std::uint64_t>(checkpoint.pollIntervalMs);
        }

        checkpoint.reserveBeforeBytes =
            admission_effective_reserve_or_default(contextState, policy.reserveBytes);
        checkpoint.reserveTargetBytes = compute_effective_reserve_target_bytes(
            cfg,
            policy,
            checkpoint.input.transientNonManagerBytes);
        if (sampleDue || !contextState.effectiveReserveValid) {
            std::uint64_t nextReserve = step_effective_reserve_bytes(
                cfg,
                checkpoint.reserveBeforeBytes,
                checkpoint.reserveTargetBytes);
            if (policy.softTargetBytes > 0) {
                nextReserve = std::min<std::uint64_t>(nextReserve, policy.softTargetBytes);
            }
            checkpoint.reserveUpdated = nextReserve != checkpoint.reserveBeforeBytes;
            contextState.effectiveReserveBytes = nextReserve;
            contextState.effectiveReserveValid = true;
        }

        checkpoint.input.effectiveReserveBytes =
            admission_effective_reserve_or_default(contextState, policy.reserveBytes);
        const PressureDecision computedDecision = classify_pressure(checkpoint.input);
        checkpoint.decision = computedDecision;
        checkpoint.desiredState = computedDecision.state;
        checkpoint.reserveCrossedNow = reserve_crossed(checkpoint.input);

        if (sampleDue) {
            checkpoint.sampled = true;
            checkpoint.previousState =
                admission_previous_state_or_decision(contextState, computedDecision.state);
            checkpoint.desiredState = computedDecision.state;

            if (contextState.transitionWindowStartMs == 0 ||
                nowMs < contextState.transitionWindowStartMs ||
                (nowMs - contextState.transitionWindowStartMs) >= 60000ull) {
                contextState.transitionWindowStartMs = nowMs;
                contextState.transitionsInWindow = 0;
            }

            PressureDecision effectiveDecision = computedDecision;
            const bool transitionRequested =
                contextState.valid && (contextState.lastState != computedDecision.state);
            bool transitionAllowed = true;
            if (transitionRequested) {
                const bool escalation =
                    pressure_state_rank(computedDecision.state) >
                    pressure_state_rank(contextState.lastState);
                if (!escalation) {
                    const std::uint64_t stateAgeMs =
                        admission_state_age_or_max(nowMs, contextState.lastStateChangeMs);
                    const bool dwellOk =
                        stateAgeMs >= static_cast<std::uint64_t>(cfg.pressureStateMinDwellMs);
                    const bool rateOk =
                        contextState.transitionsInWindow < cfg.pressureStateMaxTransitionsPerMin;
                    checkpoint.transitionDeferredByDwell = !dwellOk;
                    checkpoint.transitionDeferredByRate = !rateOk;
                    transitionAllowed = dwellOk && rateOk;
                }
            }

            checkpoint.transition = transitionRequested && transitionAllowed;
            if (transitionRequested && !transitionAllowed) {
                effectiveDecision = computedDecision;
                effectiveDecision.state = contextState.lastState;
                checkpoint.decision = effectiveDecision;
                if (checkpoint.transitionDeferredByDwell) {
                    telemetry_counter_add(managerState.pressureTransitionDwellDefers, 1);
                }
                if (checkpoint.transitionDeferredByRate) {
                    telemetry_counter_add(managerState.pressureTransitionRateDefers, 1);
                }
            } else {
                checkpoint.decision = effectiveDecision;
            }

            checkpoint.reserveCrossing = contextState.valid &&
                                         (contextState.reserveCrossed != checkpoint.reserveCrossedNow);
            if (contextState.headroomSourceValid &&
                contextState.lastHeadroomSource != checkpoint.input.headroomSource) {
                headroomSourceSwitch = true;
            }
            const bool freezeNow = checkpoint.decision.freezeOpportunistic;
            checkpoint.freezeTransitionEnter = !contextState.opportunisticFrozen && freezeNow;
            checkpoint.freezeTransitionExit = contextState.opportunisticFrozen && !freezeNow;
            contextState.valid = true;
            contextState.lastSampleMs = nowMs;
            contextState.lastDecision = checkpoint.decision;
            contextState.lastState = checkpoint.decision.state;
            contextState.opportunisticFrozen = freezeNow;
            if (!contextState.lastStateChangeMs) {
                contextState.lastStateChangeMs = nowMs;
            }
            if (checkpoint.transition) {
                contextState.lastStateChangeMs = nowMs;
                if (contextState.transitionsInWindow < std::numeric_limits<std::uint32_t>::max()) {
                    ++contextState.transitionsInWindow;
                }
            }
            contextState.reserveCrossed = checkpoint.reserveCrossedNow;
            contextState.headroomSourceValid = true;
            contextState.lastHeadroomSource = checkpoint.input.headroomSource;

            if (checkpoint.transition) {
                telemetry_counter_add(managerState.pressureStateTransitions, 1);
            }
            if (checkpoint.reserveCrossing) {
                telemetry_counter_add(managerState.reserveCrossingEvents, 1);
            }
            if (checkpoint.reserveUpdated) {
                telemetry_counter_add(managerState.reserveAdaptationEvents, 1);
            }
            if (checkpoint.freezeTransitionEnter) {
                telemetry_counter_add(managerState.opportunisticFreezeEnterEvents, 1);
            }
            if (checkpoint.freezeTransitionExit) {
                telemetry_counter_add(managerState.opportunisticFreezeExitEvents, 1);
            }
            if (headroomSourceSwitch) {
                telemetry_counter_add(managerState.headroomSourceSwitches, 1);
            }
        } else if (contextState.valid) {
            checkpoint.decision = contextState.lastDecision;
            checkpoint.previousState = contextState.lastState;
            checkpoint.desiredState = computedDecision.state;
            checkpoint.reserveCrossedNow = contextState.reserveCrossed;
            checkpoint.input.effectiveReserveBytes =
                admission_effective_reserve_or_default(contextState, policy.reserveBytes);
        }
    }

    const char* checkpointReason =
        headroom_trace_reason(headroomSourceSwitch, checkpoint.sampled);
    trace_pressure_checkpoint(
        transaction,
        commandName,
        checkpoint,
        pendingGrowthBytes,
        checkpointReason);
    trace_transient_non_manager_sample(
        transaction,
        commandName,
        checkpoint,
        pendingGrowthBytes,
        checkpointReason);
    trace_headroom_sample(
        transaction,
        commandName,
        checkpoint,
        pendingGrowthBytes,
        headroomSourceSwitch,
        checkpointReason);
    trace_effective_reserve_event(
        transaction,
        EffectiveReserveEventTrace{
            .commandName = commandName,
            .reason = checkpointReason,
            .reserveBaseBytes = checkpoint.input.reserveBytes,
            .reserveBeforeBytes = checkpoint.reserveBeforeBytes,
            .reserveTargetBytes = checkpoint.reserveTargetBytes,
            .reserveAfterBytes = checkpoint.input.effectiveReserveBytes,
            .transientNonManagerBytes = checkpoint.input.transientNonManagerBytes,
            .updated = checkpoint.reserveUpdated});

    if (checkpoint.freezeTransitionEnter || checkpoint.freezeTransitionExit || JTRACE_ENABLED(3)) {
        const bool freezeActive = checkpoint.decision.freezeOpportunistic;
        trace_opportunistic_freeze_event(
            transaction,
            OpportunisticFreezeEventTrace{
                .commandName = commandName,
                .reason = freeze_transition_reason(
                    checkpoint.freezeTransitionEnter,
                    checkpoint.freezeTransitionExit),
                .effectiveHeadroomBytes = checkpoint.input.effectiveHeadroomBytes,
                .effectiveReserveBytes = checkpoint.decision.effectiveReserveBytes,
                .pressureState = checkpoint.decision.state,
                .frozen = freezeActive,
                .allowed = !freezeActive,
                .criticalCurrentFrame = false});
    }

    return checkpoint;
}

void record_allocator_oom_headroom_observation(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::size_t requestBytes) {
    const HeadroomTelemetry headroom = sample_headroom_telemetry(transaction);
    ResourceManagerState& managerState = global_state();
    managerState.allocatorEffectiveHeadroomBytes.store(headroom.effectiveHeadroomBytes, std::memory_order_relaxed);
    managerState.allocatorPoolReservedBytes.store(headroom.allocatorPoolReservedBytes, std::memory_order_relaxed);
    managerState.allocatorPoolUsedBytes.store(headroom.allocatorPoolUsedBytes, std::memory_order_relaxed);

    const bool aboveHeadroom =
        (requestBytes > 0) && (headroom.effectiveHeadroomBytes >= static_cast<std::uint64_t>(requestBytes));
    if (aboveHeadroom) {
        telemetry_counter_add(managerState.allocFailAboveHeadroomEvents, 1);
    }

    PressureCheckpoint checkpoint{};
    checkpoint.headroom = headroom;
    checkpoint.input.softTargetBytes = transaction.resolvedPressurePolicy.softTargetBytes;
    checkpoint.input.reserveBytes = transaction.resolvedPressurePolicy.reserveBytes;
    checkpoint.input.effectiveHeadroomBytes = headroom.effectiveHeadroomBytes;
    checkpoint.input.driverFreeBytes = headroom.driverFreeBytes;
    checkpoint.input.allocatorPoolReservedBytes = headroom.allocatorPoolReservedBytes;
    checkpoint.input.allocatorPoolUsedBytes = headroom.allocatorPoolUsedBytes;
    checkpoint.input.headroomSource = headroom.source;
    trace_headroom_sample(
        transaction,
        commandName,
        checkpoint,
        requestBytes,
        false,
        headroom_alloc_fail_reason(aboveHeadroom));
}

bool run_reap_pass_for_pressure(
    const SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const char* commandName,
    const char* reason,
    std::string& outError) {
    std::size_t reclaimedBytes = 0;
    std::string reclaimError;
    if (!JuicerCuda::reap_retired_allocations(resources, reclaimedBytes, reclaimError)) {
        outError = admission_error_or_message(reclaimError, "reap pass failed");
        trace_reap_pass(
            transaction,
            commandName,
            reclaimedBytes,
            false,
            outError.c_str());
        return false;
    }

    if (reclaimedBytes > 0) {
        telemetry_counter_add(global_state().retireReapPasses, 1);
        telemetry_counter_add(global_state().retireReapBytes, reclaimedBytes);
    }
    trace_reap_pass(
        transaction,
        commandName,
        reclaimedBytes,
        true,
        reason);
    publish_manager_memory_snapshot(snapshot_manager_memory(resources));
    return true;
}

void run_tier_target_pretrim_impl(
    const SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const char* commandName,
    PressureLane lane,
    PressureState pressureState,
    const TierBudgetSnapshot& tierBudget,
    const ScratchRequestDescriptor* scratchRequest,
    bool& outDidWork,
    std::uint64_t& outScratchNormalizedActions,
    std::uint64_t& outGraphEvictedEntries) {
    outDidWork = false;
    outScratchNormalizedActions = 0;
    outGraphEvictedEntries = 0;

    const bool constrainedOrWorse =
        pressureState == PressureState::Constrained ||
        pressureState == PressureState::Critical ||
        pressureState == PressureState::Emergency;
    const bool allowProactiveTrim = constrainedOrWorse || lane == PressureLane::Builder;

    const std::uint64_t graphOverTarget =
        tier_bytes_at(tierBudget.overTargetBytes, ResourceTier::Graph);
    const std::uint64_t scratchOverTarget =
        tier_bytes_at(tierBudget.overTargetBytes, ResourceTier::Scratch);

    if (graphOverTarget > 0 && allowProactiveTrim) {
        outGraphEvictedEntries =
            evict_noncritical_graph_entries_for_context(transaction.snapshot.deviceContextKey);
    }
    if (scratchOverTarget > 0 && allowProactiveTrim) {
        ScratchCheckpointObservation scratchObservation{};
        std::string scratchCheckpointError;
        if (!run_canonical_scratch_checkpoint(
                transaction,
                resources,
                commandName,
                scratchRequest,
                ScratchCheckpointInvocation::TierPretrim,
                scratchObservation,
                scratchCheckpointError)) {
            scratchObservation.requestActive = (scratchRequest != nullptr);
            scratchObservation.requestDescriptorGeneration =
                scratchRequest ? scratchRequest->generation : 0;
            scratchObservation.hysteresisBytes = kScratchNormalizationHysteresisBytes;
            trace_scratch_checkpoint_event(
                transaction,
                commandName,
                ScratchCheckpointInvocation::TierPretrim,
                scratchObservation,
                "checkpoint_invocation_failed");
        }
        outScratchNormalizedActions = static_cast<std::uint64_t>(
            scratchObservation.shedRetiredActionCount);
    }

    outDidWork = (outGraphEvictedEntries > 0) || (outScratchNormalizedActions > 0);
    if (outDidWork && JTRACE_ENABLED(3)) {
        trace_tier_budget_event(
            transaction,
            TierBudgetEventTrace{
                .commandName = commandName,
                .reason = "tier_target_pretrim",
                .tierBudget = &tierBudget,
                .scratchNormalizedActions = outScratchNormalizedActions,
                .graphEvictedEntries = outGraphEvictedEntries,
                .lane = lane,
                .pressureState = pressureState});
    }
}

void run_tier_target_pretrim(
    const SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const char* commandName,
    PressureLane lane,
    PressureState pressureState,
    const TierBudgetSnapshot& tierBudget,
    const ScratchRequestDescriptor* scratchRequest,
    bool& outDidWork,
    std::uint64_t& outScratchNormalizedActions,
    std::uint64_t& outGraphEvictedEntries) noexcept {
    try {
        run_tier_target_pretrim_impl(
            transaction,
            resources,
            commandName,
            lane,
            pressureState,
            tierBudget,
            scratchRequest,
            outDidWork,
            outScratchNormalizedActions,
            outGraphEvictedEntries);
    } catch (...) {
        JuicerLogging::discard_current_exception();
        outDidWork = false;
        outScratchNormalizedActions = 0;
        outGraphEvictedEntries = 0;
    }
}

void evaluate_active_burst_window_impl(
    const SubmissionTransaction& transaction,
    const ResourceManagerConfigEffective& cfg,
    const PressureCheckpoint& checkpoint,
    std::size_t requestBytes,
    bool criticalCurrentFrame,
    ActiveBurstDecision& outDecision) {
    outDecision = ActiveBurstDecision{};

    const std::uint64_t softTargetBytes = checkpoint.input.softTargetBytes;
    if (requestBytes == 0 || softTargetBytes == 0) {
        return;
    }

    const std::uint64_t residentBytes = checkpoint.input.managerResidentBytes;
    if (residentBytes <= softTargetBytes) {
        PressurePolicyState& policyState = pressure_policy_state();
        std::lock_guard<std::mutex> lock(policyState.mutex);
        auto contextIt = policyState.byContext.find(transaction.snapshot.deviceContextKey);
        if (contextIt != policyState.byContext.end()) {
            PressureContextState& contextState = contextIt->second;
            if (contextState.burstActive) {
                contextState.burstActive = false;
                contextState.burstCapHitLatched = false;
                contextState.burstWindowStartMs = 0;
                contextState.burstPeakOverTargetBytes = 0;
                outDecision.exited = true;
            }
        }
        return;
    }

    outDecision.considered = true;
    outDecision.overTargetBytes = residentBytes - softTargetBytes;
    outDecision.capBytes = active_burst_cap_bytes(cfg, softTargetBytes);

    PressurePolicyState& policyState = pressure_policy_state();
    const std::uint64_t nowMs = monotonic_time_ms();
    std::lock_guard<std::mutex> lock(policyState.mutex);
    PressureContextState& contextState = policyState.byContext[transaction.snapshot.deviceContextKey];

    if (!cfg.allowActiveFrameBurst || !criticalCurrentFrame) {
        if (contextState.burstActive) {
            contextState.burstActive = false;
            contextState.burstCapHitLatched = false;
            contextState.burstWindowStartMs = 0;
            contextState.burstPeakOverTargetBytes = 0;
            outDecision.exited = true;
        }
        return;
    }

    if (!contextState.burstActive) {
        contextState.burstActive = true;
        contextState.burstCapHitLatched = false;
        contextState.burstWindowStartMs = nowMs;
        contextState.burstPeakOverTargetBytes = outDecision.overTargetBytes;
        outDecision.entered = true;
    } else {
        contextState.burstPeakOverTargetBytes = std::max<std::uint64_t>(
            contextState.burstPeakOverTargetBytes,
            outDecision.overTargetBytes);
    }

    const std::uint64_t elapsedMs =
        admission_elapsed_ms_since(nowMs, contextState.burstWindowStartMs);
    outDecision.elapsedMs = elapsedMs;
    outDecision.active = contextState.burstActive;
    outDecision.allowed = true;
    if (outDecision.capBytes == 0) {
        outDecision.allowed = false;
    }
    if (cfg.maxActiveBurstMs == 0 || elapsedMs > static_cast<std::uint64_t>(cfg.maxActiveBurstMs)) {
        outDecision.allowed = false;
    }
    if (contextState.burstPeakOverTargetBytes > outDecision.capBytes) {
        outDecision.allowed = false;
    }
    if (!outDecision.allowed && !contextState.burstCapHitLatched) {
        contextState.burstCapHitLatched = true;
        outDecision.capHit = true;
    }
}

void evaluate_active_burst_window(
    const SubmissionTransaction& transaction,
    const ResourceManagerConfigEffective& cfg,
    const PressureCheckpoint& checkpoint,
    std::size_t requestBytes,
    bool criticalCurrentFrame,
    ActiveBurstDecision& outDecision) noexcept {
    try {
        evaluate_active_burst_window_impl(
            transaction,
            cfg,
            checkpoint,
            requestBytes,
            criticalCurrentFrame,
            outDecision);
    } catch (...) {
        JuicerLogging::discard_current_exception();
        outDecision = ActiveBurstDecision{};
    }
}

BurstDebtRuntimeDecision evaluate_burst_debt_runtime(
    const SubmissionTransaction& transaction,
    const ResourceManagerConfigEffective& cfg,
    const ActiveBurstDecision& burstDecision,
    bool criticalCurrentFrame,
    std::size_t requestBytes) {
    BurstDebtRuntimeDecision out{};
    if (requestBytes == 0) {
        out.reason = "zero_request";
        return out;
    }

    out.enabled = (cfg.burstDebtHalfLifeMs > 0) && (cfg.maxBurstDebtPct < 100);
    if (!out.enabled) {
        out.reason = "disabled";
        return out;
    }
    out.sampled = true;
    telemetry_counter_add(global_state().burstDebtSampleEvents, 1);

    const std::uint64_t instanceToken = transaction.snapshot.instanceToken.value;
    if (instanceToken == 0) {
        out.reason = "missing_instance_token";
        return out;
    }

    const std::uint64_t nowMs = monotonic_time_ms();
    out.burstConsumed = criticalCurrentFrame &&
                        burstDecision.considered &&
                        burstDecision.allowed;

    PressurePolicyState& policyState = pressure_policy_state();
    std::lock_guard<std::mutex> lock(policyState.mutex);
    PressureContextState& contextState = policyState.byContext[transaction.snapshot.deviceContextKey];
    BurstDebtEntry& debtEntry = contextState.burstDebtByInstance[instanceToken];

    if (debtEntry.lastUpdateMs > 0) {
        const std::uint64_t elapsedMs =
            admission_elapsed_ms_since(nowMs, debtEntry.lastUpdateMs);
        debtEntry.debtPct = decay_burst_debt_pct(
            debtEntry.debtPct,
            elapsedMs,
            cfg.burstDebtHalfLifeMs);
    }
    debtEntry.lastUpdateMs = nowMs;
    out.debtBeforePct = debtEntry.debtPct;

    BurstDebtInput debtInput{};
    debtInput.burstDebtHalfLifeMs = cfg.burstDebtHalfLifeMs;
    debtInput.maxBurstDebtPct = cfg.maxBurstDebtPct;
    debtInput.currentDebtPct = debtEntry.debtPct;
    debtInput.criticalCurrentFrame = criticalCurrentFrame;
    debtInput.burstConsumed = out.burstConsumed;
    debtInput.burstOverTargetBytes = burstDecision.overTargetBytes;
    debtInput.burstCapBytes = burstDecision.capBytes;
    const BurstDebtDecision debtDecision = classify_burst_debt(debtInput);
    out.reason = debtDecision.reason;

    if (debtDecision.accrueDebt && debtDecision.debtIncrementPct > 0) {
        const std::uint64_t expandedDebt =
            static_cast<std::uint64_t>(debtEntry.debtPct) +
            static_cast<std::uint64_t>(debtDecision.debtIncrementPct);
        debtEntry.debtPct = static_cast<std::uint32_t>(std::min<std::uint64_t>(100ull, expandedDebt));
        out.debtIncrementPct = debtDecision.debtIncrementPct;
        telemetry_counter_add(global_state().burstDebtAccrualEvents, 1);
    }
    if (debtDecision.throttleOpportunistic) {
        out.throttled = true;
        telemetry_counter_add(global_state().burstDebtThrottleEvents, 1);
    }
    out.debtAfterPct = debtEntry.debtPct;

    if (debtEntry.debtPct == 0 && !out.burstConsumed && !out.throttled) {
        contextState.burstDebtByInstance.erase(instanceToken);
    }
    return out;
}

std::string make_superseded_builder_cancel_error(
    const char* commandName,
    std::uint64_t latestSnapshotId) {
    std::string out = std::string(kReservationDeferredPrefix) + "superseded_builder_cancel";
    if (commandName && *commandName) {
        out += ":command=";
        out += commandName;
    }
    if (latestSnapshotId > 0) {
        out += ":latest_snapshot_id=" + std::to_string(static_cast<unsigned long long>(latestSnapshotId));
    }
    return out;
}

bool should_cancel_superseded_noncritical_builder(
    const SubmissionTransaction& transaction,
    const char* commandName,
    bool criticalCurrentFrame,
    std::uint64_t requestBytes,
    std::uint64_t& outLatestSnapshotId) {
    outLatestSnapshotId = 0;
    const ResourceManagerConfigEffective& cfg = manager_effective_config();

    SupersededBuilderCancelInput input{};
    input.enabled = cfg.cancelSupersededBuilders;
    input.criticalCurrentFrame = criticalCurrentFrame;
    input.superseded = state_snapshot_is_superseded(transaction.snapshot, &outLatestSnapshotId);
    const SupersededBuilderCancelDecision decision =
        classify_superseded_builder_cancel(input);

    if (input.enabled || decision.cancel || JTRACE_ENABLED(3)) {
        trace_superseded_builder_cancel_decision(
            transaction,
            SupersededBuilderCancelTrace{
                .commandName = commandName,
                .input = &input,
                .decision = &decision,
                .requestBytes = requestBytes,
                .latestSnapshotId = outLatestSnapshotId,
                .criticalCurrentFrame = criticalCurrentFrame});
    }

    if (!decision.cancel) {
        return false;
    }

    ResourceManagerState& state = global_state();
    telemetry_counter_add(state.supersededBuilderCancelEvents, 1);
    if (requestBytes > 0) {
        telemetry_counter_add(state.supersededBuilderCancelSavedBytes, requestBytes);
    }
    return true;
}

bool enforce_pressure_gate(
    const SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const char* commandName,
    PressureLane lane,
    std::size_t requestBytes,
    bool criticalCurrentFrame,
    const ScratchRequestDescriptor* scratchRequest,
    bool& outRequestReclaimPass,
    std::string& outError) {
    outRequestReclaimPass = false;
    outError.clear();
    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    trace_pressure_gate_reader(
        transaction,
        PressureGateReaderTrace{
            .commandName = commandName,
            .path = "normal",
            .reason = "reader_observed",
            .scratchRequest = scratchRequest,
            .lane = lane,
            .requestBytes = requestBytes,
            .criticalCurrentFrame = criticalCurrentFrame});
    const std::uint64_t uploadCapBytes =
        upload_reservation_cap_bytes(cfg, transaction.resolvedPressurePolicy);
    const std::uint64_t uploadBytesInFlight =
        global_state().uploadBytesInFlight.load(std::memory_order_relaxed);
    const bool uploadCapEnabled = (uploadCapBytes > 0);
    const bool uploadCapSaturated = uploadCapEnabled && (uploadBytesInFlight >= uploadCapBytes);
    const bool nonCritical = !criticalCurrentFrame;
    std::uint64_t supersededLatestSnapshotId = 0;
    if (requestBytes > 0 &&
        should_cancel_superseded_noncritical_builder(
            transaction,
            commandName,
            criticalCurrentFrame,
            static_cast<std::uint64_t>(requestBytes),
            supersededLatestSnapshotId)) {
        outError = make_superseded_builder_cancel_error(commandName, supersededLatestSnapshotId);
        return false;
    }
    PressureState pressureState = PressureState::Normal;
    TierBudgetSnapshot tierBudget{};
    bool tierBudgetValid = false;
    const bool pressureEnabled = pressure_policy_enabled(transaction.resolvedPressurePolicy);
    PressureCheckpoint checkpoint{};
    bool checkpointValid = false;
    bool freezeBelowReserve = false;
    ActiveBurstDecision burstDecision{};
    BurstDebtRuntimeDecision burstDebtDecision{};
    if (pressureEnabled) {
        checkpoint = evaluate_pressure_checkpoint(
            transaction,
            resources,
            requestBytes,
            requestBytes > 0,
            commandName);
        checkpointValid = true;
        outRequestReclaimPass = checkpoint.decision.requestReclaimPass && (requestBytes > 0);
        pressureState = checkpoint.decision.state;
        freezeBelowReserve = checkpoint.decision.freezeOpportunistic;
        maybe_apply_async_mempool_release_policy(
            transaction,
            pressureState,
            cfg,
            "pressure_checkpoint");

        evaluate_active_burst_window(
            transaction,
            cfg,
            checkpoint,
            requestBytes,
            criticalCurrentFrame,
            burstDecision);
        if (burstDecision.entered) {
            telemetry_counter_add(global_state().activeBurstEnterEvents, 1);
        }
        if (burstDecision.exited) {
            telemetry_counter_add(global_state().activeBurstExitEvents, 1);
        }
        if (burstDecision.capHit) {
            telemetry_counter_add(global_state().activeBurstCapHitEvents, 1);
        }
        if (burstDecision.considered && !burstDecision.allowed && requestBytes > 0) {
            outRequestReclaimPass = true;
        }
        if (burstDecision.capHit && requestBytes > 0) {
            outRequestReclaimPass = true;
        }
        if (burstDecision.entered || burstDecision.exited || burstDecision.capHit || JTRACE_ENABLED(3)) {
            trace_active_burst_event(
                transaction,
                commandName,
                burstDecision,
                criticalCurrentFrame,
                burst_transition_reason(
                    burstDecision.capHit,
                    burstDecision.entered,
                    burstDecision.exited));
        }
        if (checkpoint.freezeTransitionEnter || checkpoint.freezeTransitionExit || JTRACE_ENABLED(3)) {
            trace_opportunistic_freeze_event(
                transaction,
                OpportunisticFreezeEventTrace{
                    .commandName = commandName,
                    .reason = freeze_transition_reason(
                        checkpoint.freezeTransitionEnter,
                        checkpoint.freezeTransitionExit),
                    .effectiveHeadroomBytes = checkpoint.input.effectiveHeadroomBytes,
                    .effectiveReserveBytes = checkpoint.decision.effectiveReserveBytes,
                    .pressureState = checkpoint.decision.state,
                    .frozen = freezeBelowReserve,
                    .allowed = !freezeBelowReserve,
                    .criticalCurrentFrame = criticalCurrentFrame});
        }

        burstDebtDecision = evaluate_burst_debt_runtime(
            transaction,
            cfg,
            burstDecision,
            criticalCurrentFrame,
            requestBytes);
        if (burstDebtDecision.sampled || JTRACE_ENABLED(3)) {
            trace_burst_debt_decision(
                transaction,
                commandName,
                lane,
                pressureState,
                criticalCurrentFrame,
                requestBytes,
                cfg,
                burstDebtDecision);
        }

        fill_tier_budget_snapshot(
            transaction,
            resources,
            cfg,
            checkpoint.memory,
            tierBudget);
        tierBudgetValid = true;
        if (requestBytes > 0 && tierBudget.anyOverTarget) {
            outRequestReclaimPass = true;
        }

        std::uint64_t scratchNormalizedActions = 0;
        std::uint64_t graphEvictedEntries = 0;
        bool didTierPretrim = false;
        run_tier_target_pretrim(
            transaction,
            resources,
            commandName,
            lane,
            pressureState,
            tierBudget,
            scratchRequest,
            didTierPretrim,
            scratchNormalizedActions,
            graphEvictedEntries);
        if (didTierPretrim) {
            outRequestReclaimPass = outRequestReclaimPass || (requestBytes > 0);
        }

        trace_tier_budget_event(
            transaction,
            TierBudgetEventTrace{
                .commandName = commandName,
                .reason = tier_pretrim_reason(didTierPretrim),
                .tierBudget = &tierBudget,
                .scratchNormalizedActions = scratchNormalizedActions,
                .graphEvictedEntries = graphEvictedEntries,
                .requestBytes = requestBytes,
                .lane = lane,
                .pressureState = pressureState,
                .criticalCurrentFrame = criticalCurrentFrame,
                .requestReclaimPass = outRequestReclaimPass});
    } else if (JTRACE_ENABLED(3)) {
        trace_tier_budget_event(
            transaction,
            TierBudgetEventTrace{
                .commandName = commandName,
                .reason = "tier_budget_policy_disabled",
                .tierBudget = &tierBudget,
                .requestBytes = requestBytes,
                .lane = lane,
                .pressureState = pressureState,
                .criticalCurrentFrame = criticalCurrentFrame,
                .requestReclaimPass = outRequestReclaimPass});
    }

    if (nonCritical && pressureEnabled && freezeBelowReserve && requestBytes > 0) {
        telemetry_counter_add(global_state().opportunisticFreezeDenyEvents, 1);
        trace_opportunistic_freeze_event(
            transaction,
            OpportunisticFreezeEventTrace{
                .commandName = commandName,
                .reason = "below_effective_reserve_noncritical",
                .effectiveHeadroomBytes = admission_checkpoint_headroom_or_zero(checkpointValid, checkpoint),
                .effectiveReserveBytes = admission_checkpoint_reserve_or_zero(checkpointValid, checkpoint),
                .pressureState = pressureState,
                .frozen = true,
                .allowed = false,
                .criticalCurrentFrame = criticalCurrentFrame});
        outError = make_pressure_shed_noncritical_error(lane, pressureState);
        return false;
    }

    if (nonCritical &&
        burstDebtDecision.enabled &&
        burstDebtDecision.throttled &&
        requestBytes > 0) {
        outError = make_burst_debt_throttle_noncritical_error(
            lane,
            burstDebtDecision.debtAfterPct,
            cfg.maxBurstDebtPct);
        return false;
    }

    if (nonCritical &&
        (pressureState == PressureState::Critical ||
         pressureState == PressureState::Emergency)) {
        const char* denyReason = "deny_non_critical_growth";
        if (tierBudgetValid && tierBudget.anyOverTarget) {
            denyReason = "deny_non_critical_growth_tier_over_target";
        }
        if (pressureState == PressureState::Emergency) {
            if (lane == PressureLane::Upload) {
                telemetry_counter_add(global_state().uploadEmergencyShedDenials, 1);
                denyReason = "emergency_shed_upload_precedence";
            } else {
                telemetry_counter_add(global_state().builderEmergencyShedDenials, 1);
                denyReason = "emergency_shed_builder_precedence";
            }
        }
        trace_emergency_shed_action(
            transaction,
            EmergencyShedActionTrace{
                .commandName = commandName,
                .reason = denyReason,
                .uploadBytesInFlight = uploadBytesInFlight,
                .uploadCapBytes = uploadCapBytes,
                .requestBytes = requestBytes,
                .lane = lane,
                .state = pressureState,
                .criticalCurrentFrame = criticalCurrentFrame});
        outError = make_pressure_shed_noncritical_error(lane, pressureState);
        return false;
    }

    if (lane == PressureLane::Upload && nonCritical && uploadCapSaturated) {
        telemetry_counter_add(global_state().copyComputeGuardShedEvents, 1);
        trace_copy_compute_guard(
            transaction,
            CopyComputeGuardTrace{
                .commandName = commandName,
                .reason = "upload_cap_saturated_noncritical",
                .uploadBytesInFlight = uploadBytesInFlight,
                .uploadCapBytes = uploadCapBytes,
                .state = pressureState,
                .criticalCurrentFrame = criticalCurrentFrame});
        outError = std::string(kPressureCopyComputeGuardPrefix) + " upload_cap_saturated_noncritical";
        return false;
    }

    if (pressureState == PressureState::Emergency) {
        trace_emergency_shed_action(
            transaction,
            EmergencyShedActionTrace{
                .commandName = commandName,
                .reason = emergency_progress_reason(criticalCurrentFrame),
                .uploadBytesInFlight = uploadBytesInFlight,
                .uploadCapBytes = uploadCapBytes,
                .requestBytes = requestBytes,
                .lane = lane,
                .state = pressureState,
                .criticalCurrentFrame = criticalCurrentFrame,
                .allowed = true});
    }

    if (lane == PressureLane::Upload && uploadCapSaturated) {
        trace_copy_compute_guard(
            transaction,
            CopyComputeGuardTrace{
                .commandName = commandName,
                .reason = admission_upload_saturation_reason(criticalCurrentFrame),
                .uploadBytesInFlight = uploadBytesInFlight,
                .uploadCapBytes = uploadCapBytes,
                .state = pressureState,
                .criticalCurrentFrame = criticalCurrentFrame,
                .allowed = true});
    }
    return true;
}

bool tier_circuit_begin_attempt(
    const SubmissionTransaction& transaction,
    const char* commandName,
    ResourceTier tier,
    bool blocksAdmission,
    TierCircuitAttempt& outAttempt,
    std::string& outError) {
    outAttempt = TierCircuitAttempt{};
    outError.clear();

    outAttempt.contextKey = transaction.snapshot.deviceContextKey;
    outAttempt.tier = tier;
    outAttempt.started = true;

    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    if (!tier_circuit_policy_enabled(cfg)) {
        return true;
    }

    const std::uint32_t threshold = std::max<std::uint32_t>(1u, cfg.tierErrorThreshold);
    const std::uint32_t windowMs = std::max<std::uint32_t>(1u, cfg.tierErrorWindowMs);
    const std::uint32_t openMs = std::max<std::uint32_t>(1u, cfg.tierCircuitOpenMs);
    const std::uint64_t nowMs = monotonic_time_ms();

    TierCircuitPolicyState& state = tier_circuit_policy_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    TierCircuitContextState& contextState = state.byContext[transaction.snapshot.deviceContextKey];
    TierCircuitTierState& tierState = contextState.tiers[tier_circuit_index(tier)];

    if (tierState.state == TierCircuitState::Open) {
        const std::uint64_t elapsedMs = admission_elapsed_ms_since(nowMs, tierState.openedAtMs);
        if (elapsedMs >= static_cast<std::uint64_t>(openMs)) {
            const TierCircuitState previousState = tierState.state;
            tierState.state = TierCircuitState::HalfOpen;
            tierState.probeInFlight = false;
            telemetry_counter_add(global_state().tierCircuitHalfOpenEvents, 1);
            trace_tier_circuit_event(
                transaction,
                TierCircuitEventTrace{
                    .commandName = commandName,
                    .reason = "open_elapsed_half_open",
                    .previousState = previousState,
                    .state = tierState.state,
                    .tier = tier,
                    .windowErrors = tierState.windowErrors,
                    .threshold = threshold,
                    .windowMs = windowMs,
                    .openMs = openMs,
                    .transition = true,
                    .allowed = true});
        }
    }

    if (tierState.state == TierCircuitState::Open && blocksAdmission) {
        const std::uint64_t elapsedMs = admission_elapsed_ms_since(nowMs, tierState.openedAtMs);
        const std::uint64_t openRemainingMs = admission_open_remaining_ms(elapsedMs, openMs);
        telemetry_counter_add(global_state().tierCircuitBlockedEvents, 1);
        trace_tier_circuit_event(
            transaction,
            TierCircuitEventTrace{
                .commandName = commandName,
                .reason = "open_blocked",
                .previousState = tierState.state,
                .state = tierState.state,
                .tier = tier,
                .openRemainingMs = openRemainingMs,
                .windowErrors = tierState.windowErrors,
                .threshold = threshold,
                .windowMs = windowMs,
                .openMs = openMs});
        outError = std::string(kTierCircuitOpenPrefix) + " tier=" + to_cstr(tier);
        return false;
    }

    if (tierState.state == TierCircuitState::HalfOpen) {
        if (tierState.probeInFlight && blocksAdmission) {
            telemetry_counter_add(global_state().tierCircuitBlockedEvents, 1);
            trace_tier_circuit_event(
                transaction,
                TierCircuitEventTrace{
                    .commandName = commandName,
                    .reason = "half_open_probe_in_flight",
                    .previousState = tierState.state,
                    .state = tierState.state,
                    .tier = tier,
                    .windowErrors = tierState.windowErrors,
                    .threshold = threshold,
                    .windowMs = windowMs,
                    .openMs = openMs});
            outError = std::string(kTierCircuitHalfOpenBusyPrefix) + " tier=" + to_cstr(tier);
            return false;
        }
        if (!tierState.probeInFlight) {
            tierState.probeInFlight = true;
            outAttempt.probe = true;
            trace_tier_circuit_event(
                transaction,
                TierCircuitEventTrace{
                    .commandName = commandName,
                    .reason = "half_open_probe_start",
                    .previousState = tierState.state,
                    .state = tierState.state,
                    .tier = tier,
                    .windowErrors = tierState.windowErrors,
                    .threshold = threshold,
                    .windowMs = windowMs,
                    .openMs = openMs,
                    .allowed = true,
                    .probe = true});
        }
    }

    return true;
}

void tier_circuit_cancel_attempt(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const TierCircuitAttempt& attempt,
    const char* reason) {
    if (!attempt.started) {
        return;
    }
    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    if (!tier_circuit_policy_enabled(cfg)) {
        return;
    }

    const std::uint32_t threshold = std::max<std::uint32_t>(1u, cfg.tierErrorThreshold);
    const std::uint32_t windowMs = std::max<std::uint32_t>(1u, cfg.tierErrorWindowMs);
    const std::uint32_t openMs = std::max<std::uint32_t>(1u, cfg.tierCircuitOpenMs);

    TierCircuitPolicyState& state = tier_circuit_policy_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto contextIt = state.byContext.find(attempt.contextKey);
    if (contextIt == state.byContext.end()) {
        return;
    }
    TierCircuitTierState& tierState = contextIt->second.tiers[tier_circuit_index(attempt.tier)];
    if (tierState.state == TierCircuitState::HalfOpen && tierState.probeInFlight) {
        tierState.probeInFlight = false;
        trace_tier_circuit_event(
            transaction,
            TierCircuitEventTrace{
                .commandName = commandName,
                .reason = trace_or(reason, "attempt_cancelled"),
                .previousState = tierState.state,
                .state = tierState.state,
                .tier = attempt.tier,
                .windowErrors = tierState.windowErrors,
                .threshold = threshold,
                .windowMs = windowMs,
                .openMs = openMs,
                .allowed = true});
    }
}

void tier_circuit_record_outcome(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const TierCircuitAttempt& attempt,
    bool success,
    const char* reason) {
    if (!attempt.started) {
        return;
    }

    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    if (!tier_circuit_policy_enabled(cfg)) {
        return;
    }

    const std::uint32_t threshold = std::max<std::uint32_t>(1u, cfg.tierErrorThreshold);
    const std::uint32_t windowMs = std::max<std::uint32_t>(1u, cfg.tierErrorWindowMs);
    const std::uint32_t openMs = std::max<std::uint32_t>(1u, cfg.tierCircuitOpenMs);
    const std::uint64_t nowMs = monotonic_time_ms();

    TierCircuitPolicyState& state = tier_circuit_policy_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto contextIt = state.byContext.find(attempt.contextKey);
    if (contextIt == state.byContext.end()) {
        return;
    }

    TierCircuitTierState& tierState = contextIt->second.tiers[tier_circuit_index(attempt.tier)];
    auto reset_window_if_stale = [&]() {
        if (tierState.windowStartMs == 0 ||
            nowMs < tierState.windowStartMs ||
            (nowMs - tierState.windowStartMs) >= static_cast<std::uint64_t>(windowMs)) {
            tierState.windowStartMs = nowMs;
            tierState.windowErrors = 0;
        }
    };

    if (success) {
        if (tierState.state == TierCircuitState::HalfOpen) {
            const TierCircuitState previousState = tierState.state;
            tierState.state = TierCircuitState::Closed;
            tierState.windowStartMs = 0;
            tierState.windowErrors = 0;
            tierState.openedAtMs = 0;
            tierState.probeInFlight = false;
            telemetry_counter_add(global_state().tierCircuitCloseEvents, 1);
            trace_tier_circuit_event(
                transaction,
                TierCircuitEventTrace{
                    .commandName = commandName,
                    .reason = trace_or(reason, "half_open_probe_success"),
                    .previousState = previousState,
                    .state = tierState.state,
                    .tier = attempt.tier,
                    .windowErrors = tierState.windowErrors,
                    .threshold = threshold,
                    .windowMs = windowMs,
                    .openMs = openMs,
                    .transition = true,
                    .allowed = true,
                    .probe = attempt.probe});
            return;
        }

        if (tierState.state == TierCircuitState::Closed &&
            tierState.windowStartMs > 0 &&
            (nowMs > tierState.windowStartMs) &&
            (nowMs - tierState.windowStartMs) >= static_cast<std::uint64_t>(windowMs)) {
            tierState.windowStartMs = 0;
            tierState.windowErrors = 0;
        }
        return;
    }

    if (tierState.state == TierCircuitState::HalfOpen) {
        const TierCircuitState previousState = tierState.state;
        tierState.state = TierCircuitState::Open;
        tierState.windowStartMs = nowMs;
        tierState.windowErrors = threshold;
        tierState.openedAtMs = nowMs;
        tierState.probeInFlight = false;
        telemetry_counter_add(global_state().tierCircuitOpenEvents, 1);
        trace_tier_circuit_event(
            transaction,
            TierCircuitEventTrace{
                .commandName = commandName,
                .reason = trace_or(reason, "half_open_probe_failed"),
                .previousState = previousState,
                .state = tierState.state,
                .tier = attempt.tier,
                .windowErrors = tierState.windowErrors,
                .threshold = threshold,
                .windowMs = windowMs,
                .openMs = openMs,
                .transition = true,
                .allowed = true,
                .probe = attempt.probe});
        return;
    }

    if (tierState.state == TierCircuitState::Closed) {
        reset_window_if_stale();
        if (tierState.windowErrors < std::numeric_limits<std::uint32_t>::max()) {
            ++tierState.windowErrors;
        }
        if (tierState.windowErrors >= threshold) {
            const TierCircuitState previousState = tierState.state;
            tierState.state = TierCircuitState::Open;
            tierState.openedAtMs = nowMs;
            tierState.probeInFlight = false;
            telemetry_counter_add(global_state().tierCircuitOpenEvents, 1);
            trace_tier_circuit_event(
                transaction,
                TierCircuitEventTrace{
                    .commandName = commandName,
                    .reason = trace_or(reason, "error_threshold_reached"),
                    .previousState = previousState,
                    .state = tierState.state,
                    .tier = attempt.tier,
                    .windowErrors = tierState.windowErrors,
                    .threshold = threshold,
                    .windowMs = windowMs,
                    .openMs = openMs,
                    .transition = true,
                    .allowed = true,
                    .probe = attempt.probe});
        }
        return;
    }

    if (tierState.state == TierCircuitState::Open) {
        reset_window_if_stale();
        if (tierState.windowErrors < std::numeric_limits<std::uint32_t>::max()) {
            ++tierState.windowErrors;
        }
        tierState.openedAtMs = nowMs;
    }
}

void tier_circuit_retire_context(const DeviceContextKey& key) noexcept {
    try {
        TierCircuitPolicyState& state = tier_circuit_policy_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.byContext.erase(key);
    } catch (...) {
        JuicerLogging::discard_current_exception();
    }
}

void pressure_policy_retire_context(const DeviceContextKey& key) noexcept {
    try {
        PressurePolicyState& state = pressure_policy_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.byContext.erase(key);
    } catch (...) {
        JuicerLogging::discard_current_exception();
    }
}

void scratch_normalization_retire_context(const DeviceContextKey& key) noexcept {
    try {
        ScratchNormalizationState& state = scratch_normalization_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.byContext.erase(key);
    } catch (...) {
        JuicerLogging::discard_current_exception();
    }
}

void admission_churn_retire_context(const DeviceContextKey& key) noexcept {
    try {
        AdmissionChurnPolicyState& state = admission_churn_policy_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.byContext.erase(key);
    } catch (...) {
        JuicerLogging::discard_current_exception();
    }
}

void optional_heuristic_trace_retire_context(const DeviceContextKey& key) noexcept {
    try {
        OptionalHeuristicTraceState& state = optional_heuristic_trace_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.byContext.erase(key);
    } catch (...) {
        JuicerLogging::discard_current_exception();
    }
}
