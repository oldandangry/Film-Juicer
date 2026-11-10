// Cuda/ResourceManager/JuicerCudaResourceManager.h
//
// Shared resource-manager API for lifecycle, submissions, state tracking, and telemetry.
#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "Cuda/JuicerCudaFailure.h"

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
            Failure& outError);

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
            Failure& outError);

        bool command_ensure_spatial_dir_cached_log_raw_stage(
            SubmissionTransaction& transaction,
            JuicerCuda::Resources& resources,
            const ScratchRequestDescriptor& scratchRequest,
            void* cudaStreamOpaque,
            JuicerCuda::SpatialDirCachedLogRawStageStats& outStats,
            Failure& outError);

        bool command_shed_post_frame_scratch(
            SubmissionTransaction& transaction,
            JuicerCuda::Resources& resources,
            const ScratchRequestDescriptor& scratchRequest,
            void* cudaStreamOpaque,
            const char* commandName,
            Failure& outError);

        bool command_ensure_optics_scratch(
            SubmissionTransaction& transaction,
            JuicerCuda::Resources& resources,
            const ScratchRequestDescriptor& scratchRequest,
            void* cudaStreamOpaque,
            Failure& outError);

        bool command_ensure_spatial_dir_scratch(
            SubmissionTransaction& transaction,
            JuicerCuda::Resources& resources,
            const ScratchRequestDescriptor& scratchRequest,
            void* cudaStreamOpaque,
            Failure& outError);


#if defined(JUICER_ALLOCATION_RETRY_TEST_HOOK)
        bool test_allocation_retry(
            bool (*action)(Failure&),
            bool (*reclaim)(std::size_t&, Failure&),
            Failure& failure);
#endif

        void rollback_submission(SubmissionTransaction& transaction) noexcept;

        // Process-wide sequencing and operational state shared across RM paths.
        struct ResourceManagerState {
            std::atomic<std::uint64_t> nextTransactionId{1};
            std::atomic<std::uint64_t> nextLeaseGeneration{1};
            std::atomic<std::uint64_t> contextEpoch{1};
            std::atomic<std::uint64_t> pinnedStagingBytes{0};
        };

        ResourceManagerState& global_state() noexcept;

    } // namespace ResourceManager
} // namespace JuicerCuda
