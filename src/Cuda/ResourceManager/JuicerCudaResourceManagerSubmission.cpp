// Cuda/ResourceManager/JuicerCudaResourceManagerSubmission.cpp
//
// Included by JuicerCudaResourceManager.cpp (single-TU split).
struct AllocatorBackendContextEntry {
    bool valid = false;
    bool traced = false;
    AllocatorBackendPreference requested = AllocatorBackendPreference::Auto;
    AllocatorBackendMode candidate = AllocatorBackendMode::Legacy;
    AllocatorBackendMode active = AllocatorBackendMode::Legacy;
    bool asyncPoolSupported = false;
    bool fallbackCapability = false;
    bool mempoolPolicyValid = false;
    bool mempoolPolicyApplied = false;
    std::uint64_t mempoolReleaseThresholdBytes = 0;
    PressureState mempoolPolicyPressureState = PressureState::Normal;
    const char* candidateReason = "unknown";
    const char* activeReason = "unknown";
    const char* policyDecision = "legacy_transitional";
    const char* policyRationale = "unknown";
    const char* deprecationAction = "none";
    const char* removalCriteria = "none";
};

struct AllocatorBackendState {
    std::mutex mutex;
    std::unordered_map<DeviceContextKey, AllocatorBackendContextEntry, DeviceContextKeyHash> byContext;
};

AllocatorBackendState& allocator_backend_state() noexcept {
    static AllocatorBackendState state{};
    return state;
}

AllocatorBackendContextEntry& allocator_backend_get_or_init_locked(
    AllocatorBackendState& state,
    const DeviceContextKey& key,
    const ResourceManagerConfigEffective& cfg) noexcept;

bool allocator_backend_try_get_active_mode(
    const DeviceContextKey& key,
    AllocatorBackendMode& outMode) noexcept {
    AllocatorBackendState& state = allocator_backend_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    const auto it = state.byContext.find(key);
    if (it == state.byContext.end() || !it->second.valid) {
        return false;
    }
    outMode = it->second.active;
    return true;
}

std::uint32_t submission_bool_u32(bool value) noexcept {
    if (value) {
        return 1u;
    }
    return 0u;
}

AllocatorBackendPreference sanitize_allocator_backend_preference(std::uint32_t value) noexcept {
    switch (value) {
    case 0u:
        return AllocatorBackendPreference::Legacy;
    case 1u:
        return AllocatorBackendPreference::AsyncPool;
    case 2u:
        return AllocatorBackendPreference::Slab;
    case 3u:
    default:
        return AllocatorBackendPreference::Auto;
    }
}

bool probe_async_pool_support_for_device(int deviceId) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__) && defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
    if (deviceId < 0) {
        return false;
    }
    int memoryPoolsSupported = 0;
    const cudaError_t err = cudaDeviceGetAttribute(
        &memoryPoolsSupported,
        cudaDevAttrMemoryPoolsSupported,
        deviceId);
    return err == cudaSuccess && memoryPoolsSupported != 0;
#else
    (void)deviceId;
    return false;
#endif
}

const char* allocator_backend_policy_decision(AllocatorBackendMode activeMode) noexcept {
    return activeMode == AllocatorBackendMode::AsyncPool
               ? "async_pool_preferred"
               : "legacy_transitional";
}

void finalize_allocator_backend_policy_metadata(AllocatorBackendContextEntry& entry) noexcept {
    entry.policyDecision = allocator_backend_policy_decision(entry.active);
    entry.policyRationale = entry.activeReason;
    entry.deprecationAction = "none";
    entry.removalCriteria = "none";

    if (entry.active == AllocatorBackendMode::AsyncPool) {
        entry.policyRationale = "async_pool_active";
        return;
    }

    if (entry.requested == AllocatorBackendPreference::Slab) {
        entry.policyRationale = "slab_not_in_active_plan";
        entry.deprecationAction = "remove_slab_preference_or_reapprove_scope";
        entry.removalCriteria = "slab_requires_new_committed_item_before_activation";
        return;
    }

    if (entry.fallbackCapability) {
        entry.policyRationale = "async_pool_capability_unavailable";
        entry.deprecationAction = "promote_async_pool_when_capability_available";
        entry.removalCriteria = "remove_legacy_transitional_once_locked_cuda_targets_support_async_pool";
        return;
    }

    if (entry.requested == AllocatorBackendPreference::Legacy) {
        entry.policyRationale = "legacy_preference_explicit";
        entry.deprecationAction = "migrate_config_off_legacy_preference";
        entry.removalCriteria = "disallow_explicit_legacy_preference_before_p3_closeout_or_reapprove_in_new_scope";
        return;
    }

    if (std::string_view(entry.candidateReason) == "auto_no_optional_supported") {
        entry.policyRationale = "no_optional_backend_supported";
        entry.deprecationAction = "promote_async_pool_when_capability_available";
        entry.removalCriteria = "remove_legacy_transitional_once_locked_cuda_targets_support_async_pool";
        return;
    }

    entry.deprecationAction = "review_legacy_transitional_path";
    entry.removalCriteria = "legacy_transitional_requires_explicit_reapproval_before_p3_closeout";
}

AllocatorBackendContextEntry compute_allocator_backend_context_entry(
    const DeviceContextKey& key,
    const ResourceManagerConfigEffective& cfg) noexcept {
    AllocatorBackendContextEntry out{};
    out.valid = true;
    out.requested = sanitize_allocator_backend_preference(cfg.allocatorBackendPreference);
    out.asyncPoolSupported = probe_async_pool_support_for_device(key.deviceId);
    out.candidate = AllocatorBackendMode::Legacy;
    out.active = AllocatorBackendMode::Legacy;
    out.fallbackCapability = false;
    out.candidateReason = "legacy_default";
    out.activeReason = "legacy_active";

    switch (out.requested) {
    case AllocatorBackendPreference::Legacy:
        out.candidate = AllocatorBackendMode::Legacy;
        out.candidateReason = "requested_legacy";
        break;
    case AllocatorBackendPreference::AsyncPool:
        if (out.asyncPoolSupported) {
            out.candidate = AllocatorBackendMode::AsyncPool;
            out.candidateReason = "requested_async_supported";
        }
        else {
            out.candidate = AllocatorBackendMode::Legacy;
            out.fallbackCapability = true;
            out.candidateReason = "requested_async_unsupported";
        }
        break;
    case AllocatorBackendPreference::Slab:
        out.candidate = AllocatorBackendMode::Legacy;
        out.candidateReason = "requested_slab_disallowed";
        break;
    case AllocatorBackendPreference::Auto:
        if (out.asyncPoolSupported) {
            out.candidate = AllocatorBackendMode::AsyncPool;
            out.candidateReason = "auto_select_async";
        }
        else {
            out.candidate = AllocatorBackendMode::Legacy;
            out.candidateReason = "auto_no_optional_supported";
        }
        break;
    default:
        out.candidate = AllocatorBackendMode::Legacy;
        out.candidateReason = "unknown_preference_fallback";
        break;
    }

    if (out.candidate == AllocatorBackendMode::AsyncPool) {
        out.active = AllocatorBackendMode::AsyncPool;
        out.activeReason = "active_async_pool";
    }
    else if (out.fallbackCapability) {
        out.activeReason = "capability_fallback_legacy";
    } else if (out.requested == AllocatorBackendPreference::Slab) {
        out.activeReason = "requested_slab_disallowed";
    } else {
        out.activeReason = out.candidateReason;
    }
    finalize_allocator_backend_policy_metadata(out);
    return out;
}

AllocatorBackendContextEntry& allocator_backend_get_or_init_locked(
    AllocatorBackendState& state,
    const DeviceContextKey& key,
    const ResourceManagerConfigEffective& cfg) noexcept {
    AllocatorBackendContextEntry& entry = state.byContext[key];
    if (!entry.valid) {
        entry = compute_allocator_backend_context_entry(key, cfg);
    }
    return entry;
}

void trace_allocator_backend_mode_once(
    const SubmissionTransaction& transaction,
    const AllocatorBackendContextEntry& entry) {
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    const std::string msg = trace_event_identity_prefix("backend_mode", transaction) + trace_device_context_fields(transaction) + " requested=" + to_cstr(entry.requested) + " candidate=" + to_cstr(entry.candidate) + " active=" + to_cstr(entry.active) + " async_pool_supported=" + std::to_string(submission_bool_u32(entry.asyncPoolSupported)) + " fallback_capability=" + std::to_string(submission_bool_u32(entry.fallbackCapability)) + " candidate_reason=" + trace_or_unknown(entry.candidateReason) + " candidate_reason_class=" + trace_reason_class_or_invalid(entry.candidateReason) + " active_reason=" + trace_or_unknown(entry.activeReason) + " active_reason_class=" + trace_reason_class_or_invalid(entry.activeReason) + " policy_decision=" + trace_or_unknown(entry.policyDecision) + " policy_rationale=" + trace_or_unknown(entry.policyRationale) + " deprecation_action=" + trace_or_unknown(entry.deprecationAction) + " removal_criteria=" + trace_or_unknown(entry.removalCriteria);
    JTRACE("MSALC", msg);
}

void ensure_allocator_backend_mode_initialized(
    const SubmissionTransaction& transaction,
    const ResourceManagerConfigEffective& cfg) {
    AllocatorBackendContextEntry traceEntry{};
    bool emitTrace = false;
    {
        AllocatorBackendState& state = allocator_backend_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        AllocatorBackendContextEntry& entry = allocator_backend_get_or_init_locked(
            state,
            transaction.snapshot.deviceContextKey,
            cfg);
        if (!entry.traced) {
            entry.traced = true;
            traceEntry = entry;
            emitTrace = true;
        }
    }
    if (emitTrace) {
        trace_allocator_backend_mode_once(transaction, traceEntry);
    }
}

std::uint64_t submission_saturating_mib_to_bytes(std::uint64_t mebibytes) noexcept {
    const std::uint64_t maxMbBeforeOverflow = std::numeric_limits<std::uint64_t>::max() / kBytesPerMiB;
    if (mebibytes > maxMbBeforeOverflow) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return mebibytes * kBytesPerMiB;
}

std::size_t submission_saturating_u64_to_size_t(std::uint64_t value) noexcept {
    const std::uint64_t sizeTMax = static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
    if (value >= sizeTMax) {
        return std::numeric_limits<std::size_t>::max();
    }
    return static_cast<std::size_t>(value);
}

const char* submission_apply_policy_detail(const std::string& applyError, bool applied) noexcept {
    if (applied) {
        return "set_ok";
    }
    return applyError.c_str();
}

const char* submission_lifecycle_timeout_reason(bool escalated) noexcept {
    if (escalated) {
        return "lifecycle_state_timeout_escalated";
    }
    return "lifecycle_state_timeout_not_escalated";
}

std::uint64_t async_mempool_release_threshold_bytes_for_state(
    const ResourceManagerConfigEffective& cfg,
    PressureState pressureState) noexcept {
    const std::uint64_t baseMb = static_cast<std::uint64_t>(cfg.asyncMempoolReleaseThresholdMB);
    const std::uint64_t baseBytes = submission_saturating_mib_to_bytes(baseMb);
    switch (pressureState) {
    case PressureState::Emergency:
        return 0;
    case PressureState::Critical:
        return baseBytes / 4ull;
    case PressureState::Constrained:
        return baseBytes / 2ull;
    case PressureState::Normal:
    default:
        return baseBytes;
    }
}

bool set_async_mempool_release_threshold(
    const DeviceContextKey& key,
    std::uint64_t thresholdBytes,
    std::string& outError) noexcept {
    outError.clear();
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__) && defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
    auto cuda_error_or_unknown = [](cudaError_t err) noexcept {
        return trace_or(cudaGetErrorString(err), "(unknown)");
    };
    if (key.deviceId < 0) {
        outError = "invalid device id";
        return false;
    }

    int previousDevice = -1;
    const cudaError_t queryErr = cudaGetDevice(&previousDevice);
    const bool havePrevious = (queryErr == cudaSuccess && previousDevice >= 0);
    bool switchedDevice = false;

    if (!havePrevious || previousDevice != key.deviceId) {
        const cudaError_t setErr = cudaSetDevice(key.deviceId);
        if (setErr != cudaSuccess) {
            outError = std::string("cudaSetDevice failed: ") + cuda_error_or_unknown(setErr);
            return false;
        }
        switchedDevice = havePrevious && previousDevice != key.deviceId;
    }

    cudaMemPool_t pool = nullptr;
    const cudaError_t poolErr = cudaDeviceGetDefaultMemPool(&pool, key.deviceId);
    if (poolErr != cudaSuccess || pool == nullptr) {
        if (switchedDevice) {
            (void)cudaSetDevice(previousDevice);
        }
        outError = std::string("cudaDeviceGetDefaultMemPool failed: ") + cuda_error_or_unknown(poolErr);
        return false;
    }

    std::size_t thresholdValue = submission_saturating_u64_to_size_t(thresholdBytes);
    const cudaError_t setAttrErr = cudaMemPoolSetAttribute(
        pool,
        cudaMemPoolAttrReleaseThreshold,
        &thresholdValue);

    if (switchedDevice) {
        (void)cudaSetDevice(previousDevice);
    }

    if (setAttrErr != cudaSuccess) {
        outError = std::string("cudaMemPoolSetAttribute(release_threshold) failed: ")
            + cuda_error_or_unknown(setAttrErr);
        return false;
    }
    return true;
#else
    (void)key;
    (void)thresholdBytes;
    outError = "async mempool release threshold unsupported";
    return false;
#endif
}

void trace_async_mempool_release_policy(
    const SubmissionTransaction& transaction,
    PressureState pressureState,
    std::uint64_t thresholdBytes,
    bool applied,
    bool changed,
    const char* reason,
    const char* detail) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }
    const std::string msg = trace_event_identity_prefix("mempool_release_policy", transaction)
        + trace_device_context_fields(transaction)
        + " pressure_state=" + to_cstr(pressureState)
        + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(thresholdBytes))
        + " applied=" + std::to_string(submission_bool_u32(applied))
        + " changed=" + std::to_string(submission_bool_u32(changed))
        + " reason=" + trace_or_unspecified(reason)
        + " detail=" + trace_or(detail, "none");
    JTRACE("MSALC", msg);
}

void maybe_apply_async_mempool_release_policy(
    const SubmissionTransaction& transaction,
    PressureState pressureState,
    const ResourceManagerConfigEffective& cfg,
    const char* reason) {
    const DeviceContextKey key = transaction.snapshot.deviceContextKey;
    const std::uint64_t thresholdBytes =
        async_mempool_release_threshold_bytes_for_state(cfg, pressureState);

    bool changed = false;
    {
        AllocatorBackendState& state = allocator_backend_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        AllocatorBackendContextEntry& entry =
            allocator_backend_get_or_init_locked(state, key, cfg);
        if (entry.active != AllocatorBackendMode::AsyncPool) {
            return;
        }
        changed =
            !entry.mempoolPolicyValid ||
            entry.mempoolReleaseThresholdBytes != thresholdBytes ||
            entry.mempoolPolicyPressureState != pressureState;
        if (!changed) {
            return;
        }
    }

    std::string applyError;
    const bool applied = set_async_mempool_release_threshold(
        key,
        thresholdBytes,
        applyError);

    {
        AllocatorBackendState& state = allocator_backend_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        AllocatorBackendContextEntry& entry =
            allocator_backend_get_or_init_locked(state, key, cfg);
        entry.mempoolPolicyValid = true;
        entry.mempoolPolicyApplied = applied;
        entry.mempoolReleaseThresholdBytes = thresholdBytes;
        entry.mempoolPolicyPressureState = pressureState;
    }

    trace_async_mempool_release_policy(
        transaction,
        pressureState,
        thresholdBytes,
        applied,
        changed,
        reason,
        submission_apply_policy_detail(applyError, applied));
}


void allocator_backend_retire_context(const DeviceContextKey& key) noexcept {
    AllocatorBackendState& state = allocator_backend_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.byContext.erase(key);
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

void finalize_submission_transaction(
    SubmissionTransaction& transaction,
    bool committed) noexcept {
    transaction.committed = committed;
    transaction.active = false;
}

void publish_committed_shadow_history(const SubmissionSnapshot& snapshot) noexcept {
    const ShadowHistoryKey key{ snapshot.instanceToken.value, snapshot.deviceContextKey };
    ShadowHistoryState& state = shadow_history_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    ShadowHistoryEntry& entry = state.bySubmissionKey[key];
    entry.valid = true;
    entry.digests = snapshot.keyDigests;
    entry.keySchemaVersion = snapshot.keySchemaVersion;
    entry.snapshotId = snapshot.snapshotId;
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

void record_uniform_acquire_status(AcquireStatus status) noexcept {
    telemetry_record_acquire_status(status);
    for (ResourceKind kind : kResourceKindOrder) {
        telemetry_record_acquire_status_for_kind(kind, status);
    }
}

bool trace_uniform_acquire_error(
    const SubmissionTransaction& transaction,
    std::uint64_t acquireId,
    bool hasPrevious) {
    const ResourcePlan errorPlan = make_uniform_resource_plan(AcquireStatus::Error);
    record_uniform_acquire_status(AcquireStatus::Error);
    telemetry_trace_acquire(
        acquireId,
        transaction.transactionId,
        transaction.snapshot.snapshotId,
        transaction.snapshot.traceSchemaVersion,
        AcquireStatus::Error,
        errorPlan,
        hasPrevious);
    return false;
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
    if (!JTRACE_ENABLED(1)) {
        return;
    }
    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("transaction_id=") + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " stage=" + trace_or_unknown(stage)
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " observed_state=" + to_cstr(observedState)
        + " decision=" + to_cstr(decision)
        + " state_age_ms=" + std::to_string(static_cast<unsigned long long>(stateAgeMs))
        + " accepted=" + std::to_string(submission_bool_u32(accepted))
        + " reason=" + trace_or_unspecified(reason);
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
        telemetry_counter_add(global_state().lifecycleStageRejects, 1);
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
            reason = submission_lifecycle_timeout_reason(validation.escalated);
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
            trace_or(moduleBoundaryReason, "stale_tuple_reject"));
    }
    return false;
}

const char* fragmentation_reap_outcome_reason(std::size_t reapedBytes) noexcept {
    if (reapedBytes > 0) {
        return "fragmentation_recovery_reap";
    }
    return "fragmentation_recovery_reap_no_progress";
}

const char* invalidation_reason(bool keySchemaChanged, bool laneChanged) noexcept {
    if (keySchemaChanged) {
        return "key_schema_changed";
    }
    if (laneChanged) {
        return "lane_hash_changed";
    }
    return "policy_invalidated";
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
        transaction.transactionId,
        transaction.snapshot.snapshotId,
        transaction.snapshot.traceSchemaVersion,
        trace_or(stage, "registry_submission_end_rejected"));
    if (outError) {
        *outError = "registry submission-end tracking rejected";
    }
    return false;
}

std::uint64_t submission_previous_digest_for_kind(
    bool hasPrevious,
    const ShadowHistoryEntry& previous,
    ResourceKind kind) noexcept {
    if (hasPrevious) {
        return key_digest_for_kind(previous.digests, kind);
    }
    return 0;
}

const char* submission_error_or_cstr(const std::string& error, const char* fallback) noexcept {
    if (error.empty()) {
        return fallback;
    }
    return error.c_str();
}

bool run_fragmentation_recovery_once(
    const SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const char* commandName,
    std::size_t requestBytes,
    std::uint32_t attempt,
    bool captureMemorySnapshots,
    std::string& outError) {
    outError.clear();
    ResourceManagerState& managerState = global_state();
    telemetry_counter_add(managerState.fragmentationRecoveryAttempts, 1);

    std::size_t reapedBytes = 0;
    std::string reapError;
    if (!JuicerCuda::reap_retired_allocations(resources, reapedBytes, reapError)) {
        trace_reap_pass(
            transaction,
            commandName,
            reapedBytes,
            false,
            submission_error_or_cstr(reapError, "fragmentation_recovery_reap_failed"));
        trace_fragmentation_recovery(
            transaction,
            commandName,
            attempt,
            requestBytes,
            reapedBytes,
            0,
            false,
            "attempt",
            submission_error_or_cstr(reapError, "reap_failed"));
        telemetry_counter_add(managerState.fragmentationRecoveryFailures, 1);
        outError = submission_error_or_cstr(reapError, "fragmentation recovery reap failed");
        return false;
    }

    if (reapedBytes > 0) {
        telemetry_counter_add(managerState.retireReapPasses, 1);
        telemetry_counter_add(managerState.retireReapBytes, reapedBytes);
    }
    trace_reap_pass(
        transaction,
        commandName,
        reapedBytes,
        true,
        fragmentation_reap_outcome_reason(reapedBytes));

    const std::uint64_t graphEvictedEntries =
        evict_noncritical_graph_entries_for_context(transaction.snapshot.deviceContextKey);

    if (graphEvictedEntries > 0) {
        telemetry_counter_add(managerState.fragmentationRecoveryGraphEvictedEntries, graphEvictedEntries);
    }

    if (captureMemorySnapshots) {
        maybe_publish_manager_memory_snapshot(resources, true);
    }

    trace_fragmentation_recovery(
        transaction,
        commandName,
        attempt,
        requestBytes,
        reapedBytes,
        graphEvictedEntries,
        true,
        "attempt",
        "retry_once");
    return true;
}

bool query_submission_active(const SubmissionTransaction& transaction) noexcept {
    QueryReadOnlyGuard queryGuard("query_submission_active");
    (void)queryGuard;
    return transaction.active;
}

AllocatorBackendMode query_allocator_backend_mode(const DeviceContextKey& key) noexcept {
    QueryReadOnlyGuard queryGuard("query_allocator_backend_mode", &key);
    (void)queryGuard;
    if (key.deviceId < 0) {
        return AllocatorBackendMode::Legacy;
    }
    AllocatorBackendMode activeMode = AllocatorBackendMode::Legacy;
    if (allocator_backend_try_get_active_mode(key, activeMode)) {
        return activeMode;
    }

    // Query APIs are read-only: derive an uncached mode when command path has not initialized state yet.
    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    const AllocatorBackendContextEntry derived =
        compute_allocator_backend_context_entry(key, cfg);
    return derived.active;
}

bool query_submission_device_budget_bytes(
    const DeviceContextKey& key,
    std::uint64_t& outDeviceBudgetBytes,
    std::string& outError) {
    outDeviceBudgetBytes = 0;
    outError.clear();

    if (key.deviceId < 0) {
        outError = "submission pressure policy requires a valid device id";
        return false;
    }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    bool queryOk = false;
    const bool deviceSwitchOk = with_explicit_cuda_device(
        key.deviceId,
        [&]() noexcept {
            std::size_t freeBytes = 0;
            std::size_t totalBytes = 0;
            if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess) {
                return;
            }
            outDeviceBudgetBytes = static_cast<std::uint64_t>(
                std::min<std::size_t>(
                    totalBytes,
                    static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())));
            queryOk = true;
        });
    if (!deviceSwitchOk) {
        outError = "submission pressure policy device-budget query failed";
        return false;
    }
    if (!queryOk) {
        outError = "submission pressure policy cudaMemGetInfo query failed";
        return false;
    }
    return true;
#else
    (void)key;
    outError = "submission pressure policy requires CUDA device-budget support";
    return false;
#endif
}

bool validate_resolved_pressure_policy(
    const ResolvedPressurePolicy& policy,
    std::string& outError) {
    outError.clear();
    if (policy.policyDeviceId < 0) {
        outError = "submission pressure policy device id is invalid";
        return false;
    }
    if (policy.deviceBudgetBytes <= 1) {
        outError = "submission pressure policy device budget must be greater than one byte";
        return false;
    }
    if (policy.softTargetBytes == 0) {
        outError = "submission pressure policy soft target must be non-zero";
        return false;
    }
    if (policy.reserveBytes == 0) {
        outError = "submission pressure policy reserve must be non-zero";
        return false;
    }
    if (policy.reserveBytes > policy.softTargetBytes) {
        outError = "submission pressure policy reserve exceeds soft target";
        return false;
    }
    if (policy.softTargetBytes > policy.deviceBudgetBytes) {
        outError = "submission pressure policy soft target exceeds device budget";
        return false;
    }
    return true;
}

bool resolve_submission_pressure_policy(
    const SubmissionSnapshot& snapshot,
    const ResourceManagerConfigEffective& cfg,
    ResolvedPressurePolicy& outPolicy,
    std::string& outError) {
    outPolicy = ResolvedPressurePolicy{};
    outPolicy.policyDeviceId = snapshot.deviceContextKey.deviceId;

    if (!query_submission_device_budget_bytes(
            snapshot.deviceContextKey,
            outPolicy.deviceBudgetBytes,
            outError)) {
        return false;
    }

    const bool softOverride = cfg.managerSoftTargetBytes > 0;
    const bool reserveOverride = cfg.managerReserveBytes > 0;
    const std::uint64_t maxDefaultReserveBytes =
        (outPolicy.deviceBudgetBytes > 0) ? (outPolicy.deviceBudgetBytes - 1) : 0;
    const std::uint64_t defaultReserveBytes =
        std::min<std::uint64_t>(cfg.reserveSafetyMarginBytes, maxDefaultReserveBytes);

    if (softOverride && reserveOverride) {
        outPolicy.policySource = ResolvedPressurePolicySource::SoftAndReserveOverride;
        outPolicy.softTargetBytes = cfg.managerSoftTargetBytes;
        outPolicy.reserveBytes = cfg.managerReserveBytes;
    }
    else if (softOverride) {
        outPolicy.policySource = ResolvedPressurePolicySource::SoftOverrideOnly;
        outPolicy.softTargetBytes = cfg.managerSoftTargetBytes;
        outPolicy.reserveBytes = std::min<std::uint64_t>(defaultReserveBytes, outPolicy.softTargetBytes);
    }
    else if (reserveOverride) {
        outPolicy.policySource = ResolvedPressurePolicySource::ReserveOverrideOnly;
        outPolicy.reserveBytes = cfg.managerReserveBytes;
        const std::uint64_t remainingBudget =
            (outPolicy.deviceBudgetBytes > outPolicy.reserveBytes)
                ? (outPolicy.deviceBudgetBytes - outPolicy.reserveBytes)
                : 0;
        outPolicy.softTargetBytes = std::max<std::uint64_t>(outPolicy.reserveBytes, remainingBudget);
    }
    else {
        outPolicy.policySource = ResolvedPressurePolicySource::DerivedDefault;
        outPolicy.reserveBytes = defaultReserveBytes;
        outPolicy.softTargetBytes = outPolicy.deviceBudgetBytes - outPolicy.reserveBytes;
    }

    return validate_resolved_pressure_policy(outPolicy, outError);
}

bool begin_submission(
    SubmissionTransaction& outTransaction,
    const SubmissionSnapshot& snapshot,
    std::string& outError) {
    outError.clear();
    MetadataMutationGuard mutationGuard("begin_submission", &snapshot.deviceContextKey);
    if (!mutationGuard.ok()) {
        outError = "metadata mutation guard rejected begin_submission";
        return false;
    }

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
    const std::uint32_t requestedTraceSchemaVersion = outTransaction.snapshot.traceSchemaVersion;
    if (!trace_schema_matches_contract(requestedTraceSchemaVersion)) {
        outError = "trace schema mismatch";
        telemetry_record_trace_schema_mismatch();
        telemetry_trace_schema_mismatch(
            outTransaction.transactionId,
            outTransaction.snapshot.snapshotId,
            requestedTraceSchemaVersion);
        return false;
    }

    outTransaction.snapshot.keySchemaVersion =
        sanitize_submission_key_schema_version(outTransaction.snapshot.keySchemaVersion);
    outTransaction.snapshot.traceSchemaVersion = sanitize_trace_schema_version(outTransaction.snapshot.traceSchemaVersion);
    outTransaction.snapshot.keyDigests = normalize_key_digests(outTransaction.snapshot.keyDigests);

    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    if (!resolve_submission_pressure_policy(
            outTransaction.snapshot,
            cfg,
            outTransaction.resolvedPressurePolicy,
            outError)) {
        return false;
    }

    const RegistryHandle registryHandle = registry_get_or_create(outTransaction.snapshot.deviceContextKey);
    if (registryHandle.value == 0) {
        outError = "registry get/create failed";
        return false;
    }

    std::uint64_t registryGeneration = 0;
    std::uint64_t contextEpoch = 0;
    if (!registry_get_snapshot_generations(
            outTransaction.snapshot.deviceContextKey,
            registryGeneration,
            contextEpoch)) {
        outError = "registry epoch snapshot unavailable";
        return false;
    }
    outTransaction.snapshot.registryGeneration = registryGeneration;
    outTransaction.snapshot.contextEpoch = contextEpoch;

    std::uint64_t leaseGeneration = state.nextLeaseGeneration.fetch_add(1, std::memory_order_relaxed);
    if (leaseGeneration == 0) {
        leaseGeneration = state.nextLeaseGeneration.fetch_add(1, std::memory_order_relaxed);
    }
    outTransaction.leaseGeneration = leaseGeneration;
    outTransaction.active = true;
    outTransaction.committed = false;

    if (!validate_lifecycle_for_stage(outTransaction, "begin", false, &outError)) {
        finalize_submission_transaction(outTransaction, false);
        return false;
    }
    if (!registry_note_submission_begin(outTransaction.snapshot.deviceContextKey)) {
        outError = "registry submission-begin tracking rejected";
        finalize_submission_transaction(outTransaction, false);
        return false;
    }
    ensure_allocator_backend_mode_initialized(outTransaction, cfg);
    state_note_latest_snapshot(outTransaction.snapshot);
    maybe_apply_async_mempool_release_policy(
        outTransaction,
        PressureState::Normal,
        cfg,
        "begin_submission");
    trace_optional_heuristic_surfaces_once(
        outTransaction,
        cfg,
        "begin_submission");
    telemetry_trace_schema_announcement(
        outTransaction.transactionId,
        outTransaction.snapshot.snapshotId,
        outTransaction.snapshot.traceSchemaVersion);
    telemetry_record_begin_submission();
    return true;
}

bool acquire_plan(
    SubmissionTransaction& transaction,
    std::string& outError) {
    outError.clear();
    MetadataMutationGuard mutationGuard("acquire_plan", &transaction.snapshot.deviceContextKey);
    if (!mutationGuard.ok()) {
        outError = "metadata mutation guard rejected acquire_plan";
        return false;
    }
    const std::uint64_t acquireId = telemetry_next_acquire_attempt_id();

    if (!validate_lifecycle_for_stage(transaction, "acquire", false, &outError)) {
        return trace_uniform_acquire_error(transaction, acquireId, false);
    }

    if (!validate_stale_tuple_for_stage(
            transaction,
            "acquire",
            LeaseObservationMode::ActiveOnly,
            "stale transaction in acquire path (reason=",
            &outError,
            false,
            nullptr)) {
        return trace_uniform_acquire_error(transaction, acquireId, false);
    }

    if (!ensure_submission_active(transaction, outError)) {
        return trace_uniform_acquire_error(transaction, acquireId, false);
    }

    SubmissionSnapshot& snapshot = transaction.snapshot;
    if (!trace_schema_matches_contract(snapshot.traceSchemaVersion)) {
        outError = "trace schema mismatch";
        telemetry_record_trace_schema_mismatch();
        telemetry_trace_schema_mismatch(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion);
        return trace_uniform_acquire_error(transaction, acquireId, false);
    }

    if (!submission_key_schema_matches_contract(snapshot.keySchemaVersion)) {
        const std::uint32_t normalizedKeySchemaVersion =
            sanitize_submission_key_schema_version(snapshot.keySchemaVersion);
        telemetry_record_module_boundary_violation();
        telemetry_trace_module_boundary_violation(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            "key_schema_mismatch");
        if (normalizedKeySchemaVersion != snapshot.keySchemaVersion) {
            snapshot.keySchemaVersion = normalizedKeySchemaVersion;
        }
        outError = "key schema mismatch";
        return trace_uniform_acquire_error(transaction, acquireId, false);
    }

    const KeyDigests rawDigests = snapshot.keyDigests;
    snapshot.keyDigests = normalize_key_digests(snapshot.keyDigests);
    if (rawDigests.uploadCoreHash != snapshot.keyDigests.uploadCoreHash ||
        rawDigests.dirHash != snapshot.keyDigests.dirHash ||
        rawDigests.scannerHash != snapshot.keyDigests.scannerHash ||
        rawDigests.autoExposureHash != snapshot.keyDigests.autoExposureHash) {
        telemetry_trace_key_normalization(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            rawDigests,
            snapshot.keyDigests);
    }

    std::uint64_t expectedSnapshotIdForFrame = 0;
    bool frameSnapshotMismatch = false;
    {
        const FrameSnapshotKey frameKey{ snapshot.instanceToken.value, snapshot.deviceContextKey };
        FrameSnapshotState& state = frame_snapshot_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        FrameSnapshotEntry& entry = state.bySubmissionKey[frameKey];
        const bool sameFrameToken = entry.valid && entry.frameToken == snapshot.frameToken.value;
        const bool sameSchema = entry.valid && entry.keySchemaVersion == snapshot.keySchemaVersion;
        const bool sameDigests = entry.valid && key_digests_equal(entry.digests, snapshot.keyDigests);
        if (sameFrameToken && sameSchema && sameDigests) {
            if (entry.snapshotId == 0) {
                entry.snapshotId = snapshot.snapshotId;
            }
            else if (entry.snapshotId != snapshot.snapshotId) {
                frameSnapshotMismatch = true;
                expectedSnapshotIdForFrame = entry.snapshotId;
            }
        }
        else {
            entry.valid = true;
            entry.frameToken = snapshot.frameToken.value;
            entry.digests = snapshot.keyDigests;
            entry.keySchemaVersion = snapshot.keySchemaVersion;
            entry.snapshotId = snapshot.snapshotId;
        }
    }
    if (frameSnapshotMismatch) {
        telemetry_record_frame_snapshot_mismatch();
        telemetry_trace_frame_snapshot_mismatch(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            snapshot.frameToken.value,
            expectedSnapshotIdForFrame,
            snapshot.snapshotId);
    }

    const ShadowHistoryKey key{ snapshot.instanceToken.value, snapshot.deviceContextKey };
    ShadowHistoryEntry previous{};
    bool hasPrevious = false;
    {
        ShadowHistoryState& state = shadow_history_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        auto it = state.bySubmissionKey.find(key);
        if (it != state.bySubmissionKey.end() && it->second.valid) {
            previous = it->second;
            hasPrevious = true;
        }
    }

    ShadowKeyDelta delta{};
    delta.hasPrevious = hasPrevious;
    delta.keySchemaChanged = !hasPrevious || (previous.keySchemaVersion != snapshot.keySchemaVersion);
    delta.uploadCoreChanged = !hasPrevious || (previous.digests.uploadCoreHash != snapshot.keyDigests.uploadCoreHash);
    delta.dirChanged = !hasPrevious || (previous.digests.dirHash != snapshot.keyDigests.dirHash);
    delta.scannerChanged = !hasPrevious || (previous.digests.scannerHash != snapshot.keyDigests.scannerHash);
    delta.autoExposureChanged = !hasPrevious || (previous.digests.autoExposureHash != snapshot.keyDigests.autoExposureHash);

    const ResourcePlan plan = build_shadow_resource_plan(delta);

    if (!validate_resource_kind_onboarding_contract(outError)) {
        telemetry_record_module_boundary_violation();
        telemetry_trace_module_boundary_violation(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            "resource_kind_onboarding_contract_invalid");
        return trace_uniform_acquire_error(transaction, acquireId, hasPrevious);
    }

    for (ResourceKind kind : kResourceKindOrder) {
        const ResourcePlanEntry& entry = resource_plan_entry(plan, kind);
        const ResourceKindContractEntry& contract = resource_kind_contract_entry(kind);
        const bool laneChanged = shadow_key_changed_for_kind(delta, kind);
        const std::uint64_t previousHash =
            submission_previous_digest_for_kind(hasPrevious, previous, kind);
        const std::uint64_t currentHash = key_digest_for_kind(snapshot.keyDigests, kind);

        if (entry.invalidated) {
            telemetry_trace_invalidation(
                transaction.transactionId,
                snapshot.snapshotId,
                snapshot.traceSchemaVersion,
                contract.invalidationLane,
                invalidation_reason(delta.keySchemaChanged, laneChanged),
                previousHash,
                currentHash);
        }

        if (delta.keySchemaChanged || laneChanged) {
            telemetry_trace_dag_edge(
                transaction.transactionId,
                snapshot.snapshotId,
                snapshot.traceSchemaVersion,
                contract.invalidationLane,
                contract.resourceNode,
                true,
                "allowed_lane_invalidation");
        }

        telemetry_record_acquire_status_for_kind(kind, entry.acquire.status);
    }

    bool forbiddenEdgeDetected = false;
    const ResourceKindContractEntry& scannerContract = resource_kind_contract_entry(ResourceKind::Scanner);
    const ResourceKindContractEntry& dirContract = resource_kind_contract_entry(ResourceKind::Dir);
    const ResourceKindContractEntry& uploadContract = resource_kind_contract_entry(ResourceKind::UploadCore);
    if (delta.scannerChanged &&
        !delta.uploadCoreChanged &&
        !delta.keySchemaChanged &&
        resource_plan_entry(plan, ResourceKind::UploadCore).invalidated) {
        forbiddenEdgeDetected = true;
        telemetry_record_forbidden_invalidation_edge();
        telemetry_trace_dag_edge(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            scannerContract.invalidationLane,
            uploadContract.resourceNode,
            false,
            "forbidden_edge_scanner_to_upload");
    }
    if (delta.scannerChanged &&
        !delta.dirChanged &&
        !delta.keySchemaChanged &&
        resource_plan_entry(plan, ResourceKind::Dir).invalidated) {
        forbiddenEdgeDetected = true;
        telemetry_record_forbidden_invalidation_edge();
        telemetry_trace_dag_edge(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            scannerContract.invalidationLane,
            dirContract.resourceNode,
            false,
            "forbidden_edge_scanner_to_dir");
    }
    if (delta.dirChanged &&
        !delta.uploadCoreChanged &&
        !delta.keySchemaChanged &&
        resource_plan_entry(plan, ResourceKind::UploadCore).invalidated) {
        forbiddenEdgeDetected = true;
        telemetry_record_forbidden_invalidation_edge();
        telemetry_trace_dag_edge(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            dirContract.invalidationLane,
            uploadContract.resourceNode,
            false,
            "forbidden_edge_dir_to_upload");
    }

    if (forbiddenEdgeDetected) {
        telemetry_record_module_boundary_violation();
        telemetry_trace_module_boundary_violation(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            "forbidden_invalidation_edge_detected");
    }

    const AcquireStatus finalStatus = combine_status(plan);

    telemetry_record_acquire_status(finalStatus);
    telemetry_trace_acquire(
        acquireId,
        transaction.transactionId,
        snapshot.snapshotId,
        snapshot.traceSchemaVersion,
        finalStatus,
        plan,
        hasPrevious);
    telemetry_record_acquire_plan();
    return true;
}

bool commit_submission(
    SubmissionTransaction& transaction,
    void* cudaStreamOpaque,
    std::string& outError) {
    (void)cudaStreamOpaque;
    outError.clear();
    MetadataMutationGuard mutationGuard("commit_submission", &transaction.snapshot.deviceContextKey);
    if (!mutationGuard.ok()) {
        outError = "metadata mutation guard rejected commit_submission";
        return false;
    }
    if (!validate_lifecycle_for_stage(transaction, "commit", false, &outError)) {
        return false;
    }
    if (!validate_stale_tuple_for_stage(
            transaction,
            "commit",
            LeaseObservationMode::ActiveOnly,
            "stale transaction in commit path (reason=",
            &outError,
            false,
            nullptr)) {
        return false;
    }
    if (!ensure_submission_active(transaction, outError)) {
        return false;
    }
    if (!finalize_submission_end_or_trace(transaction, "registry_submission_end_rejected_commit", &outError)) {
        finalize_submission_transaction(transaction, false);
        return false;
    }
    publish_committed_shadow_history(transaction.snapshot);
    finalize_submission_transaction(transaction, true);
    telemetry_record_commit_submission();
    return true;
}


void rollback_submission(
    SubmissionTransaction& transaction,
    const char* reason) noexcept {
    (void)reason;
    MetadataMutationGuard mutationGuard("rollback_submission", &transaction.snapshot.deviceContextKey);
    if (!mutationGuard.ok()) {
        return;
    }
    (void)validate_lifecycle_for_stage(transaction, "release", true, nullptr);
    (void)validate_stale_tuple_for_stage(
        transaction,
        "release",
        LeaseObservationMode::Always,
        nullptr,
        nullptr,
        false,
        nullptr);
    if (!transaction.active) {
        return;
    }
    (void)finalize_submission_end_or_trace(transaction, "registry_submission_end_rejected_rollback");
    finalize_submission_transaction(transaction, false);
    telemetry_record_rollback_submission();
}
