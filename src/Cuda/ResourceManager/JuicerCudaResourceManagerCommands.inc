// Cuda/ResourceManager/JuicerCudaResourceManagerCommands.inc
//
// Included by JuicerCudaResourceManager.cpp (single-TU split).
bool ensure_active_for_command(
    const SubmissionTransaction& transaction,
    std::string& outError,
    const char* commandName) {
    const char* rejectionStage = trace_or(commandName, "command_requires_active_submission");
    if (!ensure_submission_active(transaction, outError)) {
        telemetry_record_module_boundary_violation();
        telemetry_trace_module_boundary_violation(
            TelemetryModuleBoundaryViolationTrace{
                .context = TelemetryTraceContext{
                    .transactionId = transaction.transactionId,
                    .snapshotId = transaction.snapshot.snapshotId},
                .reason = rejectionStage});
        return false;
    }

    return true;
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

const char* commands_error_or_cstr(const std::string& error, const char* fallback) noexcept {
    if (error.empty()) {
        return fallback;
    }
    return error.c_str();
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

bool is_allocation_capacity_error(const std::string& error) noexcept {
    if (error.empty()) {
        return false;
    }
    return contains_ascii_case_insensitive(error, "out of memory") ||
           contains_ascii_case_insensitive(error, "memory allocation") ||
           contains_ascii_case_insensitive(error, "device_cap_exceeded");
}

bool validate_scratch_request_descriptor_for_manager(
    const ScratchRequestDescriptor& descriptor,
    const char* commandName,
    std::string& outError) noexcept {
    if (!scratch_request_descriptor_is_valid(descriptor)) {
        try {
            outError = std::string(trace_or_non_empty(commandName, "scratch_request")) +
                       ": invalid scratch request descriptor";
        } catch (...) {
            JuicerLogging::discard_current_exception();
            outError.clear();
        }
        return false;
    }
    return true;
}

bool run_large_scratch_transition_checkpoint(
    const SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const char* commandName,
    const ScratchRequestDescriptor& scratchRequest,
    void* cudaStreamOpaque,
    std::string& outError) {
    outError.clear();

    JuicerCuda::LargeScratchTransitionReclaimStats reclaimStats{};
    if (!JuicerCuda::reclaim_large_scratch_transition(
            resources,
            scratchRequest,
            cudaStreamOpaque,
            reclaimStats,
            outError)) {
        trace_large_scratch_transition_checkpoint(
            transaction,
            commandName,
            scratchRequest,
            reclaimStats,
            "large_scratch_transition_checkpoint_failed");
        return false;
    }

    trace_large_scratch_transition_checkpoint(
        transaction,
        commandName,
        scratchRequest,
        reclaimStats,
        "large_scratch_transition_checkpoint");
    return true;
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

bool validate_optics_scratch_request(
    const ScratchRequestDescriptor& scratchRequest,
    const char* commandName,
    std::string& outError) {
    if (!validate_scratch_request_descriptor_for_manager(
            scratchRequest,
            commandName,
            outError)) {
        return false;
    }

    if (!scratchRequest.needOptics) {
        outError = std::string(trace_or_non_empty(commandName, "command")) + ": optics scratch request missing needOptics";
        return false;
    }
    return true;
}

bool validate_spatial_dir_scratch_request(
    const ScratchRequestDescriptor& scratchRequest,
    const char* commandName,
    std::string& outError) {
    if (!validate_scratch_request_descriptor_for_manager(
            scratchRequest,
            commandName,
            outError)) {
        return false;
    }

    if (!scratchRequest.needSpatialDir) {
        outError = std::string(trace_or_non_empty(commandName, "command")) + ": spatial DIR scratch request missing needSpatialDir";
        return false;
    }
    return true;
}

template <typename Action>
bool execute_allocation_with_reclaim_retry(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const char* commandName,
    Action&& action,
    std::string& outError) {
    outError.clear();
    if (action(outError)) {
        return true;
    }
    if (!is_allocation_capacity_error(outError)) {
        return false;
    }

    std::size_t reclaimedBytes = 0;
    std::string reclaimError;
    if (!JuicerCuda::reap_retired_allocations(
            resources,
            reclaimedBytes,
            reclaimError)) {
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
                .attempt = 1});
        if (!reclaimError.empty()) {
            outError += " | reclaim_retry_failed: " + reclaimError;
        }
        return false;
    }

    trace_reap_pass(
        transaction,
        commandName,
        reclaimedBytes,
        true,
        reclaim_retry_reason(reclaimedBytes));

    outError.clear();
    const bool retrySucceeded = action(outError);
    trace_budget_reclaim_retry(
        transaction,
        BudgetReclaimRetryTrace{
            .commandName = commandName,
            .reason = retrySucceeded ? "allocation_retry_success"
                                     : "allocation_retry_failed",
            .reclaimedBytes = reclaimedBytes,
            .attempt = 1,
            .success = retrySucceeded});
    return retrySucceeded;
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
    if (!validate_spatial_dir_scratch_request(
            scratchRequest,
            stageName,
            outError)) {
        return false;
    }
    if (scratchRequest.spatialDirTargetPlaneRoles.cachedLogRawPlanes != 3) {
        outError = "spatial DIR cached log raw stage requested without Tier2 target roles";
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

    const bool ok = execute_allocation_with_reclaim_retry(
        transaction,
        resources,
        stageName,
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

void trace_post_frame_scratch_shed(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const ScratchRequestDescriptor& scratchRequest,
    std::size_t requestBuildBytes,
    const JuicerCuda::PostFrameScratchShedStats& stats,
    bool ok,
    bool shedAttempted,
    const char* reason,
    const std::string& error) {
#if JUICER_DIAGNOSTICS_COMPILED
    if ((ok && !shedAttempted) ? !JTRACE_ENABLED(2) : !JTRACE_ENABLED(1)) {
        return;
    }
    std::string msg =
        trace_event_prefix("post_frame_scratch_shed", transaction, commandName) +
        " ok=" + std::to_string(ok ? 1 : 0) +
        " action=" + std::string(shedAttempted ? "shed" : "keep") +
        " request_width=" + std::to_string(scratchRequest.requestedWidth) +
        " request_height=" + std::to_string(scratchRequest.requestedHeight) +
        " request_large_frame=" +
        std::to_string(
            is_large_frame_extent(
                scratchRequest.requestedWidth,
                scratchRequest.requestedHeight)
                ? 1
                : 0) +
        " request_spatial_dir_build_bytes=" +
        std::to_string(static_cast<unsigned long long>(requestBuildBytes)) +
        " pending_scratch_bytes_before=" +
        std::to_string(static_cast<unsigned long long>(
            stats.pendingScratchBytesBefore)) +
        " optics_retired_bytes=" +
        std::to_string(static_cast<unsigned long long>(stats.opticsRetiredBytes)) +
        " spatial_dir_retired_bytes=" +
        std::to_string(static_cast<unsigned long long>(
            stats.spatialDirRetiredBytes)) +
        " shared_tmp_retired_bytes=" +
        std::to_string(static_cast<unsigned long long>(
            stats.sharedTmpRetiredBytes)) +
        " reclaimed_bytes=" +
        std::to_string(static_cast<unsigned long long>(stats.reclaimedBytes)) +
        trace_device_context_fields(transaction) +
        " reason=" + trace_or_unspecified(reason);
    if (!error.empty()) {
        msg += " error=";
        msg += error;
    }
    JTRACE("MSPFS", msg);
#else
    (void)transaction;
    (void)commandName;
    (void)scratchRequest;
    (void)requestBuildBytes;
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
    const bool shedLargeSpatialDir =
        scratchRequest.needSpatialDir &&
        is_large_frame_extent(
            scratchRequest.requestedWidth,
            scratchRequest.requestedHeight);
    if (!shedLargeSpatialDir) {
        trace_post_frame_scratch_shed(
            transaction,
            stageName,
            scratchRequest,
            requestBuildBytes,
            stats,
            true,
            false,
            "ordinary_frame_retained",
            outError);
        return true;
    }

    const bool ok = JuicerCuda::shed_retained_scratch_after_frame(
        resources,
        cudaStreamOpaque,
        stats,
        outError);
    trace_post_frame_scratch_shed(
        transaction,
        stageName,
        scratchRequest,
        requestBuildBytes,
        stats,
        ok,
        true,
        ok ? "large_frame_frame_scoped" : "large_frame_release_failed",
        outError);
    return ok;
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
    if (!validate_optics_scratch_request(
            scratchRequest,
            "command_ensure_optics_scratch",
            outError)) {
        return false;
    }
    return execute_allocation_with_reclaim_retry(
        transaction,
        resources,
        "command_ensure_optics_scratch",
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
    if (!validate_spatial_dir_scratch_request(
            scratchRequest,
            "command_ensure_spatial_dir_scratch",
            outError)) {
        return false;
    }
    return execute_allocation_with_reclaim_retry(
        transaction,
        resources,
        "command_ensure_spatial_dir_scratch",
        [&](std::string& actionError) {
            return JuicerCuda::ensure_spatial_dir_scratch(
                resources,
                scratchRequest,
                cudaStreamOpaque,
                actionError);
        },
        outError);
}

bool error_is_allocation_capacity_exhausted(const std::string& error) noexcept {
    return is_allocation_capacity_error(error);
}
