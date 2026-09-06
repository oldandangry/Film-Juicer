// Cuda/ResourceManager/JuicerCudaResourceManager.cpp

#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"

#include "Cuda/JuicerCudaResources.h"
#include "Cuda/ResourceManager/JuicerCudaResourceCore.h"
#include "Logging.h"

#include <cctype>
#include <cstddef>
#include <limits>
#include <string>

namespace JuicerCuda {

    // Scratch acquire helpers remain consumed only by live command wrappers.
    bool ensure_optics_scratch(
        Resources& resources,
        const ResourceManager::ScratchRequestDescriptor& request,
        std::uint64_t expectedLease,
        void* cudaStreamOpaque,
        std::string& outError);
    bool ensure_spatial_dir_scratch(
        Resources& resources,
        const ResourceManager::ScratchRequestDescriptor& request,
        void* cudaStreamOpaque,
        std::string& outError);
    bool validate_resource_owner_locked(Resources& resources, std::string& outError, bool bindIfUnset);

    namespace ResourceManager {

        namespace {

            const char* trace_or(const char* value, const char* fallback) noexcept {
                return value ? value : fallback;
            }

            const char* trace_or_non_empty(const char* value, const char* fallback) noexcept {
                return (value && value[0] != '\0') ? value : fallback;
            }

#if JUICER_DIAGNOSTICS_COMPILED
            const char* trace_or_unknown(const char* value) noexcept {
                return value ? value : "unknown";
            }

            const char* trace_or_unspecified(const char* value) noexcept {
                return value ? value : "unspecified";
            }

            std::string trace_event_prefix(
                const char* eventName,
                const SubmissionTransaction& transaction,
                const char* commandName) {
                return std::string("event=") + trace_or_unknown(eventName) + " transaction_id=" + std::to_string(transaction.transactionId) + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId) + " command=" + trace_or_unknown(commandName);
            }

            std::string trace_device_context_fields(const SubmissionTransaction& transaction) {
                const auto& contextKey = transaction.snapshot.deviceContextKey;
                const std::uintptr_t contextBits =
                    reinterpret_cast<std::uintptr_t>(contextKey.contextOpaque);
                return std::string(" device_id=") + std::to_string(contextKey.deviceId) + " context=" + std::to_string(contextBits);
            }
#endif

            constexpr std::uint64_t kLargeFrameThresholdPixels = static_cast<std::uint64_t>(7680ull * 4320ull);

            inline std::uint64_t area_pixels_for_extent(int width, int height) noexcept {
                if (width <= 0 || height <= 0) {
                    return 0;
                }
                const std::uint64_t w = static_cast<std::uint64_t>(width);
                const std::uint64_t h = static_cast<std::uint64_t>(height);
                if (w != 0 && h > (std::numeric_limits<std::uint64_t>::max() / w)) {
                    return std::numeric_limits<std::uint64_t>::max();
                }
                return w * h;
            }

            inline bool is_large_frame_extent(int width, int height) noexcept {
                const std::uint64_t area = area_pixels_for_extent(width, height);
                const int maxDim = std::max(width, height);
                return maxDim >= 7680 || area >= kLargeFrameThresholdPixels;
            }

        } // namespace

        void trace_large_scratch_transition_checkpoint(
            const SubmissionTransaction& transaction,
            const char* commandName,
            const ScratchRequestDescriptor& scratchRequest,
            const JuicerCuda::LargeScratchTransitionReclaimStats& stats,
            bool ok,
            const std::string& error) {
#if JUICER_DIAGNOSTICS_COMPILED
            const bool hasResourceActivity =
                stats.pendingScratchBytesBefore > 0 ||
                stats.opticsRetiredBytes > 0 ||
                stats.spatialDirRetiredBytes > 0 ||
                stats.sharedTmpRetiredBytes > 0 ||
                stats.reclaimedBytes > 0;
            const bool largeRequest = is_large_frame_extent(
                scratchRequest.requestedWidth,
                scratchRequest.requestedHeight);
            if (ok && !largeRequest && !hasResourceActivity) {
                return;
            }
            if (ok ? !JTRACE_ENABLED(2) : !JTRACE_ENABLED(1)) {
                return;
            }

            std::string msg =
                trace_event_prefix("large_scratch_transition_checkpoint", transaction, commandName) +
                " request_generation=" +
                std::to_string(static_cast<unsigned long long>(scratchRequest.generation)) +
                " requested_width=" + std::to_string(scratchRequest.requestedWidth) +
                " requested_height=" + std::to_string(scratchRequest.requestedHeight) +
                " need_optics=" + std::to_string(scratchRequest.needOptics ? 1 : 0) +
                " need_spatial_dir=" + std::to_string(scratchRequest.needSpatialDir ? 1 : 0) +
                " alias_scanner_rgb_from_spatial_dir_filtered=" +
                std::to_string(scratchRequest.aliasScannerRgbFromSpatialDirFiltered ? 1 : 0) +
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
                " ok=" + std::to_string(ok ? 1 : 0) +
                trace_device_context_fields(transaction);
            if (!error.empty()) {
                msg += " error=";
                msg += error;
            }
            JTRACE("MSLTC", msg);
#else
            (void)transaction;
            (void)commandName;
            (void)scratchRequest;
            (void)stats;
            (void)ok;
            (void)error;
#endif
        }
        void trace_allocation_retry(
            const SubmissionTransaction& transaction,
            const char* commandName,
            std::size_t reclaimedBytes,
            bool success,
            const std::string& error) {
#if JUICER_DIAGNOSTICS_COMPILED
            if (!JTRACE_ENABLED(2)) {
                return;
            }

            std::string msg =
                trace_event_prefix("reclaim_retry", transaction, commandName) +
                " reclaimed_bytes=" +
                std::to_string(static_cast<unsigned long long>(reclaimedBytes)) +
                trace_device_context_fields(transaction) +
                " success=" + std::to_string(success ? 1 : 0);
            if (!error.empty()) {
                msg += " error=";
                msg += error;
            }
            JTRACE("MSEVICT", msg);
#endif
        }

        void trace_reap_pass(
            const SubmissionTransaction& transaction,
            const char* commandName,
            std::size_t reclaimedBytes,
            bool success,
            const char* reason) {
#if JUICER_DIAGNOSTICS_COMPILED
            if (!JTRACE_ENABLED(2)) {
                return;
            }

            const std::string msg = trace_event_prefix("reap_pass", transaction, commandName) + " reclaimed_bytes=" + std::to_string(static_cast<unsigned long long>(reclaimedBytes)) + " success=" + std::to_string(success ? 1 : 0) + trace_device_context_fields(transaction) + " reason=" + trace_or_unspecified(reason) + " reason_class=" + (success ? "resource_contention" : "orchestration_failure");
            JTRACE("MSREAP", msg);
#endif
        }

        std::uint64_t normalize_key_u64(std::uint64_t value) noexcept {
            if (value == 0) {
                return 1;
            }
            return value;
        }

        KeyDigests make_key_digests(
            std::uint64_t uploadCoreHash,
            std::uint64_t dirHash,
            std::uint64_t scannerHash,
            std::uint64_t autoExposureHash) noexcept {
            KeyDigests digests{};
            digests.uploadCoreHash = normalize_key_u64(uploadCoreHash);
            digests.dirHash = normalize_key_u64(dirHash);
            digests.scannerHash = normalize_key_u64(scannerHash);
            digests.autoExposureHash = normalize_key_u64(autoExposureHash);
            return digests;
        }

        KeyDigests normalize_key_digests(const KeyDigests& digests) noexcept {
            KeyDigests out = digests;
            out.uploadCoreHash = normalize_key_u64(out.uploadCoreHash);
            out.dirHash = normalize_key_u64(out.dirHash);
            out.scannerHash = normalize_key_u64(out.scannerHash);
            out.autoExposureHash = normalize_key_u64(out.autoExposureHash);
            return out;
        }
        ResourceManagerState& global_state() noexcept {
            static ResourceManagerState state{};
            return state;
        }

        void trace_module_boundary_violation(
            const SubmissionTransaction& transaction,
            const char* reason) noexcept {
#if JUICER_DIAGNOSTICS_COMPILED
            try {
                if (!JTRACE_ENABLED(1)) {
                    return;
                }
                const std::string msg =
                    std::string("transaction_id=") +
                    std::to_string(transaction.transactionId) +
                    " snapshot_id=" +
                    std::to_string(transaction.snapshot.snapshotId) +
                    " reason=" +
                    trace_or(reason, "unknown");
                JTRACE("MSCMD", msg);
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
#else
            (void)transaction;
            (void)reason;
#endif
        }

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
            trace_module_boundary_violation(
                transaction,
                trace_or(stage, "registry_submission_end_rejected"));
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
            return true;
        }


        void rollback_submission(SubmissionTransaction& transaction) noexcept {
            try {
                if (!transaction.active) {
                    return;
                }
                (void)finalize_submission_end_or_trace(transaction, "registry_submission_end_rejected_rollback");
                finalize_submission_transaction(transaction, false);
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
        }

        bool ensure_active_for_command(
            const SubmissionTransaction& transaction,
            std::string& outError,
            const char* commandName) {
            const char* rejectionStage = trace_or(commandName, "command_requires_active_submission");
            if (!ensure_submission_active(transaction, outError)) {
                trace_module_boundary_violation(transaction, rejectionStage);
                return false;
            }

            return true;
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
            const bool ok = JuicerCuda::reclaim_large_scratch_transition(
                resources,
                scratchRequest,
                cudaStreamOpaque,
                reclaimStats,
                outError);
            trace_large_scratch_transition_checkpoint(
                transaction,
                commandName,
                scratchRequest,
                reclaimStats,
                ok,
                outError);
            return ok;
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
                    scratchRequest.needGateTransmittance || scratchRequest.needFilmDustTransmittance) {
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

            const std::string firstAllocationError = outError;
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
                outError = firstAllocationError;
                if (!reclaimError.empty()) {
                    outError += " | reclaim_failed: " + reclaimError;
                }
                return false;
            }

            if (reclaimedBytes == 0) {
                trace_reap_pass(
                    transaction,
                    commandName,
                    0,
                    true,
                    "reap_no_progress");
                outError = firstAllocationError;
                return false;
            }

            std::string retryError;
            const bool retrySucceeded = action(retryError);
            trace_allocation_retry(
                transaction,
                commandName,
                reclaimedBytes,
                retrySucceeded,
                retryError);
            if (retrySucceeded) {
                outError.clear();
                return true;
            }

            outError = firstAllocationError;
            if (!retryError.empty()) {
                outError += " | allocation_retry_failed: " + retryError;
            }
            return false;
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

        void trace_post_frame_scratch_shed(
            const SubmissionTransaction& transaction,
            const char* commandName,
            const ScratchRequestDescriptor& scratchRequest,
            const JuicerCuda::PostFrameScratchShedStats& stats,
            bool ok,
            const std::string& error) {
#if JUICER_DIAGNOSTICS_COMPILED
            if (!JTRACE_ENABLED(1)) {
                return;
            }
            std::string msg =
                trace_event_prefix("post_frame_scratch_shed", transaction, commandName) +
                " ok=" + std::to_string(ok ? 1 : 0) +
                " request_width=" + std::to_string(scratchRequest.requestedWidth) +
                " request_height=" + std::to_string(scratchRequest.requestedHeight) +
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
                trace_device_context_fields(transaction);
            if (!error.empty()) {
                msg += " error=";
                msg += error;
            }
            JTRACE("MSPFS", msg);
#else
            (void)transaction;
            (void)commandName;
            (void)scratchRequest;
            (void)stats;
            (void)ok;
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

            const bool shedLargeSpatialDir =
                scratchRequest.needSpatialDir &&
                is_large_frame_extent(
                    scratchRequest.requestedWidth,
                    scratchRequest.requestedHeight);
            if (!shedLargeSpatialDir) {
                return true;
            }

            JuicerCuda::PostFrameScratchShedStats stats{};
            const bool ok = JuicerCuda::shed_retained_scratch_after_frame(
                resources,
                cudaStreamOpaque,
                stats,
                outError);
            trace_post_frame_scratch_shed(
                transaction,
                stageName,
                scratchRequest,
                stats,
                ok,
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
                        transaction.leaseGeneration,
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

    } // namespace ResourceManager
} // namespace JuicerCuda
