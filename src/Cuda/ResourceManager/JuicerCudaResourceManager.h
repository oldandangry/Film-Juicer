// Cuda/ResourceManager/JuicerCudaResourceManager.h
//
// Shared resource-manager API for lifecycle, submissions, state tracking, and telemetry.
#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "Cuda/JuicerCudaPayloads.h"
#include "Cuda/JuicerCudaResources.h"
#include "Cuda/ResourceManager/JuicerCudaResourceCore.h"

namespace JuicerCuda {

    namespace ResourceManager {

        struct RegistryContextSnapshot {
            std::uint64_t contextEpoch = 0;
        };

        // Exact-context admission and explicit two-phase retirement.
        bool registry_begin_submission(
            const DeviceContextKey& key,
            RegistryContextSnapshot& outSnapshot) noexcept;
        bool registry_note_submission_end(const DeviceContextKey& key) noexcept;
        bool registry_begin_owner_retire(
            const DeviceContextKey& key,
            RegistryContextSnapshot& outSnapshot) noexcept;
        bool registry_retire(const DeviceContextKey& key) noexcept;
        void registry_snapshot_context_keys(std::vector<DeviceContextKey>& outKeys);

        // Submission execution entry points.
        bool begin_submission(
            SubmissionTransaction& outTransaction,
            const SubmissionSnapshot& snapshot,
            std::uint64_t deviceBudgetBytes,
            std::string& outError);

        bool commit_submission(
            SubmissionTransaction& transaction,
            void* cudaStreamOpaque,
            std::string& outError);

        bool command_checkpoint_large_scratch_transition(
            SubmissionTransaction& transaction,
            JuicerCuda::Resources& resources,
            const ScratchRequestDescriptor& scratchRequest,
            void* cudaStreamOpaque,
            const char* commandName,
            std::string& outError);

        bool command_ensure_spatial_dir_cached_log_raw_stage(
            SubmissionTransaction& transaction,
            JuicerCuda::Resources& resources,
            const ScratchRequestDescriptor& scratchRequest,
            void* cudaStreamOpaque,
            JuicerCuda::SpatialDirCachedLogRawStageStats& outStats,
            std::string& outError);

        bool command_shed_post_frame_scratch(
            SubmissionTransaction& transaction,
            JuicerCuda::Resources& resources,
            const ScratchRequestDescriptor& scratchRequest,
            void* cudaStreamOpaque,
            const char* commandName,
            std::string& outError);

        bool command_ensure_optics_scratch(
            SubmissionTransaction& transaction,
            JuicerCuda::Resources& resources,
            const ScratchRequestDescriptor& scratchRequest,
            void* cudaStreamOpaque,
            std::string& outError);

        bool command_ensure_spatial_dir_scratch(
            SubmissionTransaction& transaction,
            JuicerCuda::Resources& resources,
            const ScratchRequestDescriptor& scratchRequest,
            void* cudaStreamOpaque,
            std::string& outError);

        bool error_is_allocation_capacity_exhausted(const std::string& error) noexcept;

        void rollback_submission(
            SubmissionTransaction& transaction,
            const char* reason) noexcept;

        // Process-wide counters and sequencing state shared across RM registry and submission paths.
        struct ResourceManagerState {
            std::atomic<std::uint64_t> nextTransactionId{1};
            std::atomic<std::uint64_t> nextLeaseGeneration{1};
            std::atomic<std::uint64_t> contextEpoch{1};
            std::atomic<std::uint64_t> beginSubmissionCalls{0};
            std::atomic<std::uint64_t> commitSubmissionCalls{0};
            std::atomic<std::uint64_t> rollbackSubmissionCalls{0};
            std::atomic<std::uint64_t> moduleBoundaryViolations{0};
            std::atomic<std::uint64_t> registryLiveManagers{0};
            std::atomic<std::uint64_t> hostAssetCacheBytes{0};
            std::atomic<std::uint64_t> hostAssetCacheTrimEvents{0};
            std::atomic<std::uint64_t> hostAssetCacheTrimBytes{0};
            std::atomic<std::uint64_t> hostAssetCacheCapHits{0};
            std::atomic<std::uint64_t> pinnedStagingBytes{0};
            std::atomic<std::uint64_t> pinnedStagingCapHits{0};
            std::atomic<std::uint64_t> pinnedStagingFallbackEvents{0};
            std::atomic<std::uint64_t> pinnedStagingTrimEvents{0};
            std::atomic<std::uint64_t> pinnedStagingTrimBytes{0};
        };

        ResourceManagerState& global_state() noexcept;

// Lightweight counter helpers used throughout the RM implementation.
#ifndef JUICER_RM_TELEMETRY_COUNTERS_COMPILED
#define JUICER_RM_TELEMETRY_COUNTERS_COMPILED 0
#endif

        inline void telemetry_counter_add(
            std::atomic<std::uint64_t>& counter,
            std::uint64_t delta = 1) noexcept {
#if JUICER_RM_TELEMETRY_COUNTERS_COMPILED
            if (delta == 0) {
                return;
            }
            counter.fetch_add(delta, std::memory_order_relaxed);
#else
            (void)counter;
            (void)delta;
#endif
        }

        inline void telemetry_counter_note_max(
            std::atomic<std::uint64_t>& counter,
            std::uint64_t value) noexcept {
#if JUICER_RM_TELEMETRY_COUNTERS_COMPILED
            std::uint64_t observed = counter.load(std::memory_order_relaxed);
            while (value > observed) {
                if (counter.compare_exchange_weak(
                        observed,
                        value,
                        std::memory_order_relaxed,
                        std::memory_order_relaxed)) {
                    break;
                }
            }
#else
            (void)counter;
            (void)value;
#endif
        }

        inline void telemetry_counter_subtract_saturating(
            std::atomic<std::uint64_t>& counter,
            std::uint64_t delta) noexcept {
#if JUICER_RM_TELEMETRY_COUNTERS_COMPILED
            if (delta == 0) {
                return;
            }
            std::uint64_t current = counter.load(std::memory_order_relaxed);
            while (true) {
                const std::uint64_t next = (current >= delta) ? (current - delta) : 0;
                if (counter.compare_exchange_weak(
                        current,
                        next,
                        std::memory_order_relaxed,
                        std::memory_order_relaxed)) {
                    return;
                }
            }
#else
            (void)counter;
            (void)delta;
#endif
        }

        // Counter and trace hooks implemented by the RM runtime.
        void telemetry_record_begin_submission() noexcept;
        void telemetry_record_commit_submission() noexcept;
        void telemetry_record_rollback_submission() noexcept;
        void telemetry_record_module_boundary_violation() noexcept;

    } // namespace ResourceManager
} // namespace JuicerCuda
