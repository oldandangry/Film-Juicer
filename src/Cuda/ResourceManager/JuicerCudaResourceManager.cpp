// Cuda/ResourceManager/JuicerCudaResourceManager.cpp

#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"

#include "Cuda/JuicerCudaResources.h"
#include "Cuda/ResourceManager/JuicerCudaResourceCore.h"
#include "Hash.h"
#include "Logging.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
#include <cuda_runtime.h>
#endif

namespace JuicerCuda {

    // Scratch acquire helpers remain consumed only by live command wrappers.
    bool ensure_optics_scratch(
        Resources& resources,
        const ResourceManager::ScratchRequestDescriptor& request,
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

            template <typename TraceAction>
            void run_telemetry_trace_noexcept(TraceAction&& action) noexcept {
                try {
                    action();
                } catch (...) {
                    JuicerLogging::discard_current_exception();
                }
            }

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

            std::string trace_event_identity_prefix(
                const char* eventName,
                const SubmissionTransaction& transaction) {
                return std::string("event=") + trace_or_unknown(eventName) + " transaction_id=" + std::to_string(transaction.transactionId) + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId);
            }

            std::string trace_device_context_fields(const SubmissionTransaction& transaction) {
                const auto& contextKey = transaction.snapshot.deviceContextKey;
                const std::uintptr_t contextBits =
                    reinterpret_cast<std::uintptr_t>(contextKey.contextOpaque);
                return std::string(" device_id=") + std::to_string(contextKey.deviceId) + " context=" + std::to_string(contextBits);
            }
#endif

            template <typename Action>
            bool with_explicit_cuda_device(int targetDevice, Action&& action) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
                if (targetDevice < 0) {
                    return false;
                }

                int previousDevice = -1;
                if (cudaGetDevice(&previousDevice) != cudaSuccess) {
                    return false;
                }
                if (previousDevice != targetDevice &&
                    cudaSetDevice(targetDevice) != cudaSuccess) {
                    return false;
                }

                action();

                if (previousDevice != targetDevice &&
                    cudaSetDevice(previousDevice) != cudaSuccess) {
                    return false;
                }
                return true;
#else
                (void)targetDevice;
                (void)action;
                return false;
#endif
            }

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

            inline std::size_t plane_bytes_for_extent(int width, int height) noexcept {
                if (width <= 0 || height <= 0) {
                    return 0;
                }
                const std::size_t w = static_cast<std::size_t>(width);
                const std::size_t h = static_cast<std::size_t>(height);
                if (h > (std::numeric_limits<std::size_t>::max() / w)) {
                    return std::numeric_limits<std::size_t>::max();
                }
                const std::size_t n = w * h;
                if (n > (std::numeric_limits<std::size_t>::max() / sizeof(float))) {
                    return std::numeric_limits<std::size_t>::max();
                }
                return n * sizeof(float);
            }

        } // namespace


        void trace_scratch_request_descriptor(
            const SubmissionTransaction& transaction,
            const char* commandName,
            const ScratchRequestDescriptor& descriptor,
            const char* reason) {
#if JUICER_DIAGNOSTICS_COMPILED
            if (!JTRACE_ENABLED(2)) {
                return;
            }

            const Spektrafilm::DirScratchPlaneRoles& roles = descriptor.spatialDirPlaneRoles;
            const Spektrafilm::DirScratchPlaneRoles& targetRoles = descriptor.spatialDirTargetPlaneRoles;
            const std::string msg = trace_event_prefix("scratch_request", transaction, commandName) + " request_generation=" + std::to_string(static_cast<unsigned long long>(descriptor.generation)) + " need_optics=" + std::to_string(descriptor.needOptics ? 1 : 0) + " need_spatial_dir=" + std::to_string(descriptor.needSpatialDir ? 1 : 0) + " spatial_dir_descriptor_hash=" + std::to_string(static_cast<unsigned long long>(descriptor.spatialDirDescriptorHash)) + " spatial_dir_scratch_tier=" + Spektrafilm::to_cstr(descriptor.spatialDirScratchTier) + " spatial_dir_target_scratch_tier=" + Spektrafilm::to_cstr(descriptor.spatialDirTargetScratchTier) + " raw_correction_planes=" + std::to_string(roles.rawCorrectionPlanes) + " filtered_correction_planes=" + std::to_string(roles.filteredCorrectionPlanes) + " filter_temp_planes=" + std::to_string(roles.filterTempPlanes) + " cached_log_raw_planes=" + std::to_string(roles.cachedLogRawPlanes) + " target_raw_correction_planes=" + std::to_string(targetRoles.rawCorrectionPlanes) + " target_filtered_correction_planes=" + std::to_string(targetRoles.filteredCorrectionPlanes) + " target_filter_temp_planes=" + std::to_string(targetRoles.filterTempPlanes) + " target_cached_log_raw_planes=" + std::to_string(targetRoles.cachedLogRawPlanes) + " requested_width=" + std::to_string(descriptor.requestedWidth) + " requested_height=" + std::to_string(descriptor.requestedHeight) + " need_blurred=" + std::to_string(descriptor.needBlurred ? 1 : 0) + " alias_scanner_rgb_from_spatial_dir_filtered=" + std::to_string(descriptor.aliasScannerRgbFromSpatialDirFiltered ? 1 : 0) + " need_aux=" + std::to_string(descriptor.needAux ? 1 : 0) + " need_shared_tmp=" + std::to_string(descriptor.needSharedTmp ? 1 : 0) + " need_grain_frame_uniforms=" + std::to_string(descriptor.needGrainFrameUniforms ? 1 : 0) + " need_grain_layer_work=" + std::to_string(descriptor.needGrainLayerWork ? 1 : 0) + " need_grain_shared=" + std::to_string(descriptor.needGrainShared ? 1 : 0) + " need_gate_mask=" + std::to_string(descriptor.needGateMask ? 1 : 0) + trace_device_context_fields(transaction) + " reason=" + trace_or_unspecified(reason);
            JTRACE("MSSRQ", msg);
#endif
        }


        void trace_large_scratch_transition_checkpoint(
            const SubmissionTransaction& transaction,
            const char* commandName,
            const ScratchRequestDescriptor& scratchRequest,
            const JuicerCuda::LargeScratchTransitionReclaimStats& stats,
            const char* reason) {
#if JUICER_DIAGNOSTICS_COMPILED
            if (!JTRACE_ENABLED(2)) {
                return;
            }

            const std::string msg =
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
                trace_device_context_fields(transaction) +
                " reason=" + trace_or_unspecified(reason);
            JTRACE("MSLTC", msg);
#else
            (void)transaction;
            (void)commandName;
            (void)scratchRequest;
            (void)stats;
            (void)reason;
#endif
        }


        struct BudgetReclaimRetryTrace {
            const char* commandName = nullptr;
            const char* reason = nullptr;
            std::size_t reclaimedBytes = 0;
            std::uint32_t attempt = 0;
            bool success = false;
        };

        void trace_budget_reclaim_retry(
            const SubmissionTransaction& transaction,
            const BudgetReclaimRetryTrace& trace) {
#if JUICER_DIAGNOSTICS_COMPILED
            if (!JTRACE_ENABLED(2)) {
                return;
            }

            const std::string msg = trace_event_prefix("reclaim_retry", transaction, trace.commandName) + " attempt=" + std::to_string(static_cast<unsigned long long>(trace.attempt)) + " reclaimed_bytes=" + std::to_string(static_cast<unsigned long long>(trace.reclaimedBytes)) + trace_device_context_fields(transaction) + " success=" + std::to_string(trace.success ? 1 : 0) + " reason=" + trace_or_unspecified(trace.reason);
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

        // Former RM foundation implementation now owned by the RM TU.
        ResourceManagerConfigEffective sanitize_config(const ResourceManagerConfigRaw& raw) noexcept {
            constexpr std::uint64_t kMiB = 1024ull * 1024ull;
            constexpr std::uint32_t kMinHostAssetIdleTrimMs = 1000u;
            constexpr std::uint32_t kMaxHostAssetIdleTrimMs = 60000u;
            constexpr std::uint32_t kMinPinnedStagingIdleTrimMs = 500u;
            constexpr std::uint32_t kMaxPinnedStagingIdleTrimMs = 60000u;
            constexpr std::uint64_t kMinHostAssetCacheMaxBytes = 64ull * kMiB;
            constexpr std::uint64_t kMaxHostAssetCacheMaxBytes = 1024ull * kMiB;
            constexpr std::uint64_t kMinHostAssetTrimBatchBytes = 8ull * kMiB;
            constexpr std::uint64_t kMinPinnedStagingMaxBytes = 8ull * kMiB;
            constexpr std::uint64_t kMaxPinnedStagingMaxBytes = 2048ull * kMiB;
            constexpr std::uint64_t kMinPinnedStagingTrimBatchBytes = 1ull * kMiB;

            ResourceManagerConfigEffective out{};
            out.hostAssetCacheMaxBytes = std::clamp(
                raw.hostAssetCacheMaxBytes,
                kMinHostAssetCacheMaxBytes,
                kMaxHostAssetCacheMaxBytes);
            out.hostAssetIdleTrimMs = std::clamp(
                raw.hostAssetIdleTrimMs,
                kMinHostAssetIdleTrimMs,
                kMaxHostAssetIdleTrimMs);
            out.hostAssetTrimBatchBytes = std::clamp(
                raw.hostAssetTrimBatchBytes,
                kMinHostAssetTrimBatchBytes,
                out.hostAssetCacheMaxBytes);
            out.pinnedUploadStagingMaxBytes = std::clamp(
                raw.pinnedUploadStagingMaxBytes,
                kMinPinnedStagingMaxBytes,
                kMaxPinnedStagingMaxBytes);
            out.pinnedUploadStagingIdleTrimMs = std::clamp(
                raw.pinnedUploadStagingIdleTrimMs,
                kMinPinnedStagingIdleTrimMs,
                kMaxPinnedStagingIdleTrimMs);
            out.pinnedUploadStagingTrimBatchBytes = std::clamp(
                raw.pinnedUploadStagingTrimBatchBytes,
                kMinPinnedStagingTrimBatchBytes,
                out.pinnedUploadStagingMaxBytes);
            return out;
        }


        std::uint64_t normalize_key_u64(std::uint64_t value) noexcept {
            if (value == 0) {
                return 1;
            }
            return value;
        }

        std::uint32_t normalize_scan_lut_resolution(std::uint32_t value) noexcept {
            return std::clamp(value, kScanLutResolutionMin, kScanLutResolutionMax);
        }

        std::uint64_t make_scan_lut_key_digest(
            std::uint32_t medium,
            std::uint64_t tablesHash,
            std::uint64_t densityRangeHash,
            std::uint32_t lutResolution,
            std::uint32_t lutFormatVersion,
            std::uint32_t keySchemaVersion) noexcept {
            if (medium > 1u ||
                tablesHash == 0 ||
                densityRangeHash == 0 ||
                lutFormatVersion == 0 ||
                keySchemaVersion == 0) {
                return 0;
            }

            const std::uint64_t fields[] = {
                static_cast<std::uint64_t>(keySchemaVersion),
                static_cast<std::uint64_t>(medium),
                tablesHash,
                densityRangeHash,
                static_cast<std::uint64_t>(normalize_scan_lut_resolution(lutResolution)),
                static_cast<std::uint64_t>(lutFormatVersion)};
            std::uint64_t digest = Hash::kFnvOffset;
            const auto* bytes = reinterpret_cast<const std::uint8_t*>(fields);
            for (std::size_t index = 0; index < sizeof(fields); ++index) {
                digest ^= static_cast<std::uint64_t>(bytes[index]);
                digest *= Hash::kFnvPrime;
            }
            return digest;
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

        void telemetry_record_begin_submission() noexcept {
            telemetry_counter_add(global_state().beginSubmissionCalls, 1);
        }

        void telemetry_record_commit_submission() noexcept {
            telemetry_counter_add(global_state().commitSubmissionCalls, 1);
        }

        void telemetry_record_rollback_submission() noexcept {
            telemetry_counter_add(global_state().rollbackSubmissionCalls, 1);
        }

        void telemetry_record_module_boundary_violation() noexcept {
            telemetry_counter_add(global_state().moduleBoundaryViolations, 1);
        }

        constexpr const char* kTraceTokenUnknown = "unknown";
        const char* trace_token_or(const char* value, const char* fallback) noexcept {
            if (value) {
                return value;
            }
            return fallback;
        }

        struct TelemetryTraceContext {
            std::uint64_t transactionId = 0;
            std::uint64_t snapshotId = 0;
        };

        std::string telemetry_trace_txn_snapshot_prefix(const TelemetryTraceContext& context) {
            return std::string("transaction_id=") + std::to_string(context.transactionId) + " snapshot_id=" + std::to_string(context.snapshotId);
        }

        std::string telemetry_trace_device_context_fields(const DeviceContextKey* key) {
            int deviceId = -1;
            std::uintptr_t contextBits = 0;
            if (key) {
                deviceId = key->deviceId;
                contextBits = reinterpret_cast<std::uintptr_t>(key->contextOpaque);
            }
            return std::string(" device_id=") + std::to_string(deviceId) + " context=" + std::to_string(contextBits);
        }

        std::string telemetry_trace_event_prefix(const char* eventName) {
            return std::string("event=") + trace_token_or(eventName, kTraceTokenUnknown);
        }

        struct TelemetryModuleBoundaryViolationTrace {
            TelemetryTraceContext context{};
            const char* reason = nullptr;
        };

        void telemetry_trace_module_boundary_violation(const TelemetryModuleBoundaryViolationTrace& trace) noexcept {
#if JUICER_DIAGNOSTICS_COMPILED
            run_telemetry_trace_noexcept([&]() {
                if (!JTRACE_ENABLED(1)) {
                    return;
                }
                const std::string msg =
                    telemetry_trace_txn_snapshot_prefix(trace.context) +
                    " reason=" + trace_token_or(trace.reason, kTraceTokenUnknown);
                JTRACE("MSCMD", msg);
            });
#endif
        }

// Split implementation sections (single-TU include model to preserve exact behavior while
// reducing monolithic file size and keeping ownership boundaries explicit).
#include "Cuda/ResourceManager/JuicerCudaResourceManagerSubmission.inc"
#include "Cuda/ResourceManager/JuicerCudaResourceManagerCommands.inc"

    } // namespace ResourceManager
} // namespace JuicerCuda
