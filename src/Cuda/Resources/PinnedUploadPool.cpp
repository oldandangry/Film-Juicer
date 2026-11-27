#include "Cuda/Resources/ResourceInternals.h"

#include "Cuda/JuicerCudaResourceOperations.h"
#include "Cuda/JuicerCudaHostViews.h"
#include "Cuda/JuicerCudaDriver.h"
#include "Cuda/JuicerCudaPayloads.h"
#include "Cuda/ResourceManager/JuicerCudaResourceCore.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"

#include "SpectralData.h"
#include "GamutCompression.h"
#include "Scanner.h"
#include "JuicerDirBoundary.h"
#include "juicer_host_preparation.h"

#include "Logging.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <new>
#include <numeric>
#include <atomic>
#include <span>
#include <sstream>
#include <exception>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace JuicerCuda {

    struct PinnedUploadContextKeyHash {
        std::size_t operator()(const PinnedUploadContextKey& key) const noexcept {
            const std::size_t hDevice = std::hash<int>{}(key.deviceId);
            const std::size_t hContext = std::hash<std::uintptr_t>{}(
                reinterpret_cast<std::uintptr_t>(key.contextOpaque));
            const auto context = hDevice ^ (hContext + 0x9e3779b9u + (hDevice << 6u) + (hDevice >> 2u));
            return context ^ std::hash<std::uint64_t>{}(key.epoch);
        }
    };

    constexpr std::size_t kPinnedUploadStagingMaxBytes =
        128ull * 1024ull * 1024ull;

    enum class PinnedUploadBlockState : std::uint8_t {
        Available = 0,
        Reserved,
        InFlight,
        Quarantined,
        Unpublished
    };

    struct PinnedUploadBlock {
        std::uint64_t id = 0;
        PinnedBuffer buffer;
        std::size_t capacity = 0;
        std::uint64_t lastTouchedMs = 0;
        RetiredEvent completion;
        PinnedUploadBlockState state = PinnedUploadBlockState::Available;
    };

    struct PinnedUploadPool {
        std::vector<PinnedUploadBlock> blocks;
        std::uint64_t nextBlockId = 1;
        std::size_t totalBytes = 0;
    };

    struct PinnedUploadStagingPolicyState {
        // Serializes reserve/trim/allocation operations. The state mutex below is
        // never held across CUDA calls.
        std::mutex reservationMutex;
        std::mutex mutex;
        std::unordered_map<PinnedUploadContextKey, PinnedUploadPool, PinnedUploadContextKeyHash> pools;
        std::size_t totalBytesAllContexts = 0;
    };

    static PinnedUploadStagingPolicyState& pinned_upload_staging_policy_state() {
        static PinnedUploadStagingPolicyState state;
        return state;
    }


    static std::uint64_t pinned_upload_now_ms() noexcept {
        using Clock = std::chrono::steady_clock;
        const auto now = Clock::now().time_since_epoch();
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
    }

    static void publish_pinned_upload_staging_bytes(std::size_t bytes) {
        ResourceManager::global_state().pinnedStagingBytes.store(
            static_cast<std::uint64_t>(bytes), std::memory_order_relaxed);
    }

    static std::size_t published_pinned_upload_staging_bytes() noexcept {
        return static_cast<std::size_t>(
            ResourceManager::global_state().pinnedStagingBytes.load(
                std::memory_order_relaxed));
    }

    static void trace_pinned_staging_event(
        const char* stage,
        const char* eventName,
        const char* reason,
        const PinnedUploadContextKey& key,
        std::size_t bytes,
        std::size_t totalBytes) noexcept {
        if (!JTRACE_ENABLED(2)) {
            return;
        }
        try {
            std::ostringstream oss;
            oss << "stage=" << (stage ? stage : "unknown")
                << " event=" << (eventName ? eventName : "unknown")
                << " reason=" << (reason ? reason : "none")
                << " device_id=" << key.deviceId
                << " context=" << reinterpret_cast<std::uintptr_t>(key.contextOpaque)
                << " bytes=" << bytes
                << " cap_bytes=" << kPinnedUploadStagingMaxBytes
                << " total_bytes=" << totalBytes;
            JTRACE("MSPIN", oss.str());
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }

    static PinnedUploadBlock* find_pinned_upload_block_locked(
        PinnedUploadStagingPolicyState& policyState,
        const PinnedUploadContextKey& key,
        std::uint64_t blockId) {
        const auto poolIt = policyState.pools.find(key);
        if (poolIt == policyState.pools.end()) {
            return nullptr;
        }
        for (PinnedUploadBlock& block : poolIt->second.blocks) {
            if (block.id == blockId) {
                return &block;
            }
        }
        return nullptr;
    }

    struct PinnedUploadCompletionProbe {
        PinnedUploadContextKey key{};
        std::uint64_t blockId = 0;
        void* doneEventOpaque = nullptr;
    };

    static void refresh_pinned_block_completions(
        PinnedUploadStagingPolicyState& policyState,
        const PinnedUploadContextKey* contextFilter) {
        std::vector<PinnedUploadCompletionProbe> probes;
        {
            std::lock_guard<std::mutex> lock(policyState.mutex);
            for (auto& poolEntry : policyState.pools) {
                if (contextFilter && !(poolEntry.first == *contextFilter)) {
                    continue;
                }
                probes.reserve(probes.size() + poolEntry.second.blocks.size());
                for (PinnedUploadBlock& block : poolEntry.second.blocks) {
                    if (block.state != PinnedUploadBlockState::InFlight) {
                        continue;
                    }
                    if (!block.completion.get()) {
                        block.state = PinnedUploadBlockState::Quarantined;
                        continue;
                    }
                    probes.push_back(PinnedUploadCompletionProbe{
                        poolEntry.first,
                        block.id,
                        block.completion.get()});
                }
            }
        }

        for (const PinnedUploadCompletionProbe& probe : probes) {
            const cudaError_t queryErr = cudaEventQuery(
                reinterpret_cast<cudaEvent_t>(probe.doneEventOpaque));
            PinnedUploadBlockState nextState = PinnedUploadBlockState::Quarantined;
            if (queryErr == cudaSuccess) {
                nextState = PinnedUploadBlockState::Available;
            } else if (queryErr == cudaErrorNotReady) {
                nextState = PinnedUploadBlockState::InFlight;
            }

            std::size_t blockBytes = 0;
            bool transitioned = false;
            {
                std::lock_guard<std::mutex> lock(policyState.mutex);
                PinnedUploadBlock* block =
                    find_pinned_upload_block_locked(
                        policyState,
                        probe.key,
                        probe.blockId);
                if (block && block->state == PinnedUploadBlockState::InFlight) {
                    block->state = nextState;
                    blockBytes = block->capacity;
                    if (nextState == PinnedUploadBlockState::Available) {
                        block->lastTouchedMs = pinned_upload_now_ms();
                    }
                    transitioned = true;
                }
            }

            if (transitioned && nextState == PinnedUploadBlockState::Quarantined) {
                const char* reason = cudaGetErrorString(queryErr);
                trace_pinned_staging_event(
                    "completion_query",
                    "quarantine",
                    reason ? reason : "event_query_failed",
                    probe.key,
                    blockBytes,
                    published_pinned_upload_staging_bytes());
            }
        }
    }

    static int pick_reusable_pinned_block_locked(
        PinnedUploadPool& pool,
        std::size_t requiredBytes) {
        int bestIndex = -1;
        std::size_t bestCapacity = std::numeric_limits<std::size_t>::max();
        for (std::size_t i = 0; i < pool.blocks.size(); ++i) {
            const PinnedUploadBlock& block = pool.blocks[i];
            if (block.state != PinnedUploadBlockState::Available || !block.buffer.get() ||
                !block.completion.get() || block.capacity < requiredBytes) {
                continue;
            }
            if (block.capacity < bestCapacity) {
                bestCapacity = block.capacity;
                bestIndex = static_cast<int>(i);
            }
        }
        return bestIndex;
    }

    static bool pinned_upload_pool_is_empty_locked(const PinnedUploadPool& pool) {
        return pool.blocks.empty() && pool.totalBytes == 0;
    }

    struct DetachedPinnedUploadBlock {
        PinnedUploadContextKey key{};
        PinnedUploadBlock block{};
    };

    static std::vector<DetachedPinnedUploadBlock> collect_pressure_trim_blocks_locked(
        PinnedUploadStagingPolicyState& policyState,
        std::size_t requiredBytes) {
        struct Candidate {
            PinnedUploadContextKey key;
            std::uint64_t blockId;
            std::uint64_t lastTouchedMs;
        };
        std::vector<Candidate> candidates;
        for (const auto& [key, pool] : policyState.pools) {
            for (const auto& block : pool.blocks) {
                if (block.state == PinnedUploadBlockState::Available && block.buffer) {
                    candidates.push_back({key, block.id, block.lastTouchedMs});
                }
            }
        }
        std::ranges::sort(candidates, {}, &Candidate::lastTouchedMs);
        std::vector<DetachedPinnedUploadBlock> detached;
        detached.reserve(candidates.size());
        std::size_t trimmedBytes = 0;
        for (const auto& candidate : candidates) {
            if (trimmedBytes >= requiredBytes) {
                break;
            }
            auto& pool = policyState.pools.at(candidate.key);
            const auto block = std::ranges::find(pool.blocks, candidate.blockId, &PinnedUploadBlock::id);
            trimmedBytes += block->capacity;
            pool.totalBytes -= block->capacity;
            detached.push_back({candidate.key, std::move(*block)});
            pool.blocks.erase(block);
        }
        return detached;
    }

    static std::size_t destroy_detached_pinned_upload_blocks(
        std::vector<DetachedPinnedUploadBlock>& blocks,
        const PinnedUploadContextKey& current,
        const char* stage,
        std::size_t totalBytes) noexcept {
        std::size_t freedBytes = 0;
        for (auto& [key, block] : blocks) {
            if (const auto status = block.buffer.free_after_completion(); status != cudaSuccess) {
                block.state = PinnedUploadBlockState::Quarantined;
                trace_pinned_staging_event(stage, "quarantine", cudaGetErrorString(status), key, block.capacity, totalBytes);
                continue;
            }
            freedBytes += block.capacity;
            trace_pinned_staging_event(stage, "trim", "cap_pressure", key, block.capacity, totalBytes);
            block.capacity = 0;
            // The host allocation is reusable capacity after completion. Its
            // context-local event stays with the original epoch for destruction.
            if (key == current) {
                (void)block.completion.destroy();
            }
        }
        return freedBytes;
    }

    static bool reserve_reusable_pinned_upload_block_locked(
        PinnedUploadStagingPolicyState& policyState,
        const PinnedUploadContextKey& key,
        std::size_t bytes,
        PinnedUploadReservation& outReservation) {
        const auto poolIt = policyState.pools.find(key);
        if (poolIt == policyState.pools.end()) {
            return false;
        }
        const int blockIndex =
            pick_reusable_pinned_block_locked(poolIt->second, bytes);
        if (blockIndex < 0) {
            return false;
        }

        PinnedUploadBlock& block =
            poolIt->second.blocks[static_cast<std::size_t>(blockIndex)];
        block.state = PinnedUploadBlockState::Reserved;
        block.lastTouchedMs = pinned_upload_now_ms();
        outReservation.staged = true;
        outReservation.key = key;
        outReservation.blockId = block.id;
        outReservation.stagingPtr = block.buffer.get();
        outReservation.doneEventOpaque = block.completion.get();
        return true;
    }

    static PinnedUploadReservation reserve_pinned_upload_block(
        const PinnedUploadContextKey& key,
        std::size_t bytes,
        const char* stage,
        bool& capExceeded,
        Failure& outError) {
        PinnedUploadReservation result{};
        if (bytes == 0) {
            return result;
        }

        result.key = key;
        if (key.deviceId < 0 || !key.contextOpaque) {
            outError.diagnostic = "pinned staging invalid_resource_owner";
            return result;
        }
        if (bytes > kPinnedUploadStagingMaxBytes) {
            capExceeded = true;
            outError.diagnostic = "pinned staging cap_exceeded";
            return result;
        }

        PinnedUploadStagingPolicyState& policyState = pinned_upload_staging_policy_state();
        std::lock_guard<std::mutex> reservationLock(
            policyState.reservationMutex);
        refresh_pinned_block_completions(policyState, &result.key);

        bool capPressure = false;
        {
            std::lock_guard<std::mutex> lock(policyState.mutex);
            if (reserve_reusable_pinned_upload_block_locked(
                    policyState,
                    result.key,
                    bytes,
                    result)) {
                return result;
            }

            if (policyState.totalBytesAllContexts >
                kPinnedUploadStagingMaxBytes - bytes) {
                capPressure = true;
            }
        }

        std::vector<DetachedPinnedUploadBlock> detachedBlocks;
        if (capPressure) {
            refresh_pinned_block_completions(policyState, nullptr);

            std::size_t totalBytesAfterTrim = 0;
            std::size_t requiredTrimBytes = 0;
            {
                std::lock_guard<std::mutex> lock(policyState.mutex);
                if (reserve_reusable_pinned_upload_block_locked(
                        policyState,
                        result.key,
                        bytes,
                        result)) {
                    return result;
                }

                requiredTrimBytes =
                    (policyState.totalBytesAllContexts >
                     kPinnedUploadStagingMaxBytes - bytes)
                        ? policyState.totalBytesAllContexts -
                              (kPinnedUploadStagingMaxBytes - bytes)
                        : 0;
                if (requiredTrimBytes > 0) {
                    detachedBlocks = collect_pressure_trim_blocks_locked(
                        policyState,
                        requiredTrimBytes);
                }
                totalBytesAfterTrim = policyState.totalBytesAllContexts;
            }

            const std::size_t freedBytes = destroy_detached_pinned_upload_blocks(
                detachedBlocks,
                result.key,
                stage,
                totalBytesAfterTrim);
            {
                std::lock_guard<std::mutex> lock(policyState.mutex);
                for (auto& [key, block] : detachedBlocks) {
                    if (!block.buffer && !block.completion) {
                        continue;
                    }
                    auto& pool = policyState.pools.at(key);
                    pool.totalBytes += block.capacity;
                    pool.blocks.push_back(std::move(block));
                }
                policyState.totalBytesAllContexts =
                    (policyState.totalBytesAllContexts >= freedBytes)
                        ? policyState.totalBytesAllContexts - freedBytes
                        : 0;
                for (auto it = policyState.pools.begin();
                     it != policyState.pools.end();) {
                    if (pinned_upload_pool_is_empty_locked(it->second)) {
                        it = policyState.pools.erase(it);
                    } else {
                        ++it;
                    }
                }
                totalBytesAfterTrim = policyState.totalBytesAllContexts;
                publish_pinned_upload_staging_bytes(totalBytesAfterTrim);
            }
            trace_pinned_staging_event(
                stage,
                "cap_hit",
                detachedBlocks.empty() ? "capacity_rechecked" : "pressure_trim",
                result.key,
                bytes,
                totalBytesAfterTrim);
            if (totalBytesAfterTrim > kPinnedUploadStagingMaxBytes - bytes) {
                capExceeded = true;
                outError.diagnostic = "pinned staging cap_exceeded";
                return result;
            }
        }

        PinnedUploadPool* pool = nullptr;
        PinnedUploadBlock* block = nullptr;
        std::size_t totalBytes = 0;
        try {
            std::lock_guard<std::mutex> lock(policyState.mutex);
            pool = &policyState.pools[result.key];
            block = &pool->blocks.emplace_back();
            block->id = pool->nextBlockId++;
            block->capacity = bytes;
            block->state = PinnedUploadBlockState::Unpublished;
            pool->totalBytes += bytes;
            policyState.totalBytesAllContexts += bytes;
            totalBytes = policyState.totalBytesAllContexts;
            publish_pinned_upload_staging_bytes(totalBytes);
        } catch (...) {
            JuicerLogging::discard_current_exception();
            set_failure(outError, outError.status, "pinned staging pool_insert_failed");
            return result;
        }

        // The reservation lock keeps the slot in place. Unpublished entries have
        // no borrowers and retain both handles and capacity during rollback.
        const auto discard = [&] {
            const auto eventStatus = block->completion.destroy();
            const auto hostStatus = block->buffer.free_after_completion();
            record_cleanup_context_loss(outError, runtime_failure_status(eventStatus));
            record_cleanup_context_loss(outError, runtime_failure_status(hostStatus));
            if (hostStatus == cudaSuccess) {
                std::lock_guard<std::mutex> lock(policyState.mutex);
                pool->totalBytes -= block->capacity;
                policyState.totalBytesAllContexts -= block->capacity;
                block->capacity = 0;
                publish_pinned_upload_staging_bytes(policyState.totalBytesAllContexts);
                if (!block->completion) {
                    pool->blocks.pop_back();
                }
            }
        };

        void* pinnedPtr = nullptr;
        const auto allocErr = ResourceOperations::allocate_upload(&pinnedPtr, bytes);
        block->buffer = PinnedBuffer(pinnedPtr);
        if (allocErr != cudaSuccess || !pinnedPtr) {
            // Host staging exhaustion must not select device reclaim.
            FjStatus status = runtime_failure_status(allocErr);
            if (allocErr == cudaErrorMemoryAllocation) {
                status.category = FJ_STATUS_CUDA_FAILURE;
            }
            set_failure(outError, status, "cudaMallocHost(pinned staging) failed");
            discard();
            return result;
        }

        cudaEvent_t doneEvent = nullptr;
        const auto eventErr = ResourceOperations::create_upload_event(&doneEvent);
        block->completion = RetiredEvent(doneEvent);
        if (eventErr != cudaSuccess || !doneEvent) {
            FjStatus status = runtime_failure_status(eventErr);
            if (eventErr == cudaErrorMemoryAllocation) {
                status.category = FJ_STATUS_CUDA_FAILURE;
            }
            set_failure(outError, status, "cudaEventCreateWithFlags(pinned staging) failed");
            discard();
            return result;
        }

        {
            std::lock_guard<std::mutex> lock(policyState.mutex);
            block->lastTouchedMs = pinned_upload_now_ms();
            block->state = PinnedUploadBlockState::Reserved;
            result.blockId = block->id;
            result.stagingPtr = pinnedPtr;
            result.doneEventOpaque = doneEvent;
            result.staged = true;
        }

        trace_pinned_staging_event(
            stage,
            "alloc",
            "new_block",
            result.key,
            bytes,
            totalBytes);
        return result;
    }

    static bool transition_pinned_upload_reservation(
        const PinnedUploadReservation& reservation,
        PinnedUploadBlockState nextState) {
        if (!reservation.staged || reservation.blockId == 0) {
            return false;
        }

        PinnedUploadStagingPolicyState& policyState = pinned_upload_staging_policy_state();
        std::lock_guard<std::mutex> lock(policyState.mutex);
        PinnedUploadBlock* block = find_pinned_upload_block_locked(
            policyState,
            reservation.key,
            reservation.blockId);
        if (!block || block->buffer.get() != reservation.stagingPtr ||
            block->completion.get() != reservation.doneEventOpaque ||
            block->state != PinnedUploadBlockState::Reserved) {
            return false;
        }

        block->state = nextState;
        if (nextState == PinnedUploadBlockState::Available) {
            block->lastTouchedMs = pinned_upload_now_ms();
        }
        return true;
    }

    static bool return_pinned_upload_reservation(
        const PinnedUploadReservation& reservation) {
        return transition_pinned_upload_reservation(
            reservation,
            PinnedUploadBlockState::Available);
    }

    StagedHostUpload::~StagedHostUpload() noexcept {
        if (reservation.staged) {
            try {
                (void)return_pinned_upload_reservation(reservation);
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
        }
    }

    enum class PinnedUploadCompletionResult : std::uint8_t {
        InFlight = 0,
        CompletedAfterExceptionalSync,
        Quarantined
    };

    static PinnedUploadCompletionResult complete_enqueued_pinned_upload(
        const PinnedUploadReservation& reservation,
        cudaStream_t stream,
        const char* stage,
        Resources* overwriteOwner,
        Failure& outError) {
        const auto retain_unfenced_destination = [&] {
            if (overwriteOwner) {
                // The caller owns resources.m and the destination is still published.
                overwriteOwner->frameUseFenceQuarantined = true;
            }
        };
        outError = {};
        const cudaError_t recordErr = ResourceOperations::record_event(
            reinterpret_cast<cudaEvent_t>(reservation.doneEventOpaque), stream);
        if (recordErr == cudaSuccess) {
            if (transition_pinned_upload_reservation(
                    reservation,
                    PinnedUploadBlockState::InFlight)) {
                return PinnedUploadCompletionResult::InFlight;
            }
            retain_unfenced_destination();
            outError.diagnostic = "pinned staging reservation ownership was lost";
            return PinnedUploadCompletionResult::Quarantined;
        }

        // Resolve completion and reservation ownership before allocating diagnostics.
        const cudaError_t syncErr = ResourceOperations::synchronize_stream(stream);
        if (syncErr == cudaSuccess) {
            if (!transition_pinned_upload_reservation(
                    reservation,
                    PinnedUploadBlockState::Available)) {
                outError.diagnostic = "pinned staging reservation ownership was lost";
                return PinnedUploadCompletionResult::Quarantined;
            }
            trace_pinned_staging_event(
                stage,
                "event_record_failed",
                "stream_sync_completed",
                reservation.key,
                0,
                published_pinned_upload_staging_bytes());
            return PinnedUploadCompletionResult::CompletedAfterExceptionalSync;
        }

        retain_unfenced_destination();
        const bool quarantined = transition_pinned_upload_reservation(
            reservation,
            PinnedUploadBlockState::Quarantined);
        const char* syncReason = cudaGetErrorString(syncErr);
        const char* recordReason = cudaGetErrorString(recordErr);
        std::array<char, 640> diagnostic{};
        std::snprintf(diagnostic.data(), diagnostic.size(), "cudaEventRecord(pinned staging) failed: %s | cudaStreamSynchronize(pinned staging) failed: %s%s", recordReason ? recordReason : "(unknown)", syncReason ? syncReason : "(unknown)", quarantined ? "" : " | pinned staging reservation ownership was lost");
        set_failure(outError, runtime_failure_status(recordErr), diagnostic.data());
        record_cleanup_context_loss(outError, runtime_failure_status(syncErr));
        trace_pinned_staging_event(
            stage,
            "quarantine",
            "event_record_and_stream_sync_failed",
            reservation.key,
            0,
            published_pinned_upload_staging_bytes());
        return PinnedUploadCompletionResult::Quarantined;
    }

    bool stage_host_to_device_copy(
        const PinnedUploadContextKey& key,
        const HostToDeviceCopyRequest& request,
        const void* src,
        std::size_t bytes,
        StagedHostUpload& staged,
        Failure& outError) noexcept try {
        outError = {};
        staged.capExceeded = false;
        staged.reservation = reserve_pinned_upload_block(key, bytes, request.stage, staged.capExceeded, outError);
        if (!staged.reservation.staged) {
            outError.diagnostic = std::string(request.label ? request.label : "upload") + ": " + outError.diagnostic;
            return false;
        }

        // The caller's storage expires at return, including local split/generated tables.
        std::memcpy(staged.reservation.stagingPtr, src, bytes);
        return true;
    } catch (...) {
        JuicerLogging::discard_current_exception();
        outError.diagnostic.clear();
        return false;
    }

    bool enqueue_staged_host_to_device_copy(
        Resources& resources,
        const HostToDeviceCopyRequest& request,
        void* dst,
        std::size_t bytes,
        void* cudaStreamOpaque,
        StagedHostUpload& staged,
        bool overwrite,
        Failure& outError) noexcept try {
        const cudaStream_t stream = reinterpret_cast<cudaStream_t>(cudaStreamOpaque);
        const PinnedUploadReservation reservation = std::exchange(staged.reservation, {});
        const cudaError_t copyErr = ResourceOperations::copy_upload(dst, reservation.stagingPtr, bytes, stream);
        if (copyErr != cudaSuccess) {
            const bool returned = return_pinned_upload_reservation(reservation);
            outError.status = runtime_failure_status(copyErr);
            outError.diagnostic = std::string("cudaMemcpyAsync(") + (request.label ? request.label : "upload") + " pinned staging) failed: " + cudaGetErrorString(copyErr);
            if (!returned) {
                outError.diagnostic += " | failed to return pinned staging reservation";
            }
            return false;
        }

        const PinnedUploadCompletionResult completion =
            complete_enqueued_pinned_upload(reservation, stream, request.stage, overwrite ? &resources : nullptr, outError);
        if (completion == PinnedUploadCompletionResult::Quarantined) {
            return false;
        }
        if (completion == PinnedUploadCompletionResult::CompletedAfterExceptionalSync) {
            outError = {};
        }
#if JUICER_DIAGNOSTICS_COMPILED
        if (JTRACE_ENABLED(3) && request.label) {
            try {
                JTRACE_VERBOSE("CUDA_UPLOAD", std::string("label=") + request.label + " bytes=" + std::to_string(bytes));
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
        }
#endif
        return true;
    } catch (...) {
        // Pool/diagnostic allocation failure must still reach the caller's device rollback.
        // Any enqueued copy has reached its terminal reservation state before formatting.
        JuicerLogging::discard_current_exception();
        outError.diagnostic.clear();
        return false;
    }

    bool enqueue_host_to_device_copy(
        Resources& resources,
        const HostToDeviceCopyRequest& request,
        void* dst,
        const void* src,
        std::size_t bytes,
        void* cudaStreamOpaque,
        Failure& outError) noexcept try {
        outError = {};
        if (bytes == 0) {
            return true;
        }
        if (!dst || !src) {
            outError.diagnostic = std::string(request.label ? request.label : "copy") + " upload args invalid";
            return false;
        }
        const PinnedUploadContextKey key = resources.deviceId == resources.ownerContextKey.deviceId
                                               ? PinnedUploadContextKey{resources.deviceId, resources.ownerContextKey.contextOpaque, resources.contextEpoch}
                                               : PinnedUploadContextKey{};
        StagedHostUpload staged;
        return stage_host_to_device_copy(key, request, src, bytes, staged, outError) &&
               enqueue_staged_host_to_device_copy(resources, request, dst, bytes, cudaStreamOpaque, staged, false, outError);
    } catch (...) {
        JuicerLogging::discard_current_exception();
        outError.diagnostic.clear();
        return false;
    }

    bool purge_pinned_upload_staging_for_context(
        int deviceId,
        void* contextOpaque,
        std::uint64_t epoch,
        PinnedUploadPurgeDisposition disposition,
        Failure& failure) noexcept try {
        const PinnedUploadContextKey key{deviceId, contextOpaque, epoch};
        auto& state = pinned_upload_staging_policy_state();
        std::lock_guard<std::mutex> reservationLock(state.reservationMutex);
        PinnedUploadPool* pool = nullptr;
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            const auto found = state.pools.find(key);
            if (found == state.pools.end()) {
                return true;
            }
            pool = &found->second;
        }

        void* current = nullptr;
        int driverCode = 0;
        std::string diagnostic;
        const bool currentOwner = query_current_cuda_context(current, diagnostic, &driverCode) && current == contextOpaque;
        const bool provenLoss = disposition == PinnedUploadPurgeDisposition::ProvenContextLoss;
        if (!currentOwner && !provenLoss) {
            set_failure(failure, driver_failure_status(driverCode), "pinned upload retirement requires its producing CUDA context");
            return false;
        }
        bool drainRequired = false;
        if (currentOwner) {
            std::lock_guard<std::mutex> lock(state.mutex);
            drainRequired = std::ranges::any_of(pool->blocks, [](const auto& block) {
                return block.state == PinnedUploadBlockState::Reserved ||
                       block.state == PinnedUploadBlockState::Quarantined;
            });
        }
        if (drainRequired) {
            const auto status = cudaDeviceSynchronize();
            if (status != cudaSuccess && !(provenLoss && runtime_failure_status(status).category == FJ_STATUS_CONTEXT_LOSS)) {
                set_failure(failure, runtime_failure_status(status), "pinned upload context drain failed");
                return false;
            }
        }

        // The retiring registry epoch can no longer admit uploads. The
        // reservation lock prevents other epochs from trimming these blocks.
        for (auto& block : pool->blocks) {
            if (!provenLoss && !drainRequired && block.state == PinnedUploadBlockState::InFlight) {
                const auto status = cudaEventSynchronize(block.completion.get());
                if (status != cudaSuccess) {
                    set_failure(failure, runtime_failure_status(status), "pinned upload completion wait failed");
                    return false;
                }
            }
            if (provenLoss) {
                (void)block.completion.release();
            } else if (const auto status = block.completion.destroy(); status != cudaSuccess) {
                set_failure(failure, runtime_failure_status(status), "pinned upload completion event destruction failed");
                return false;
            }
            block.state = PinnedUploadBlockState::Available;
            if (const auto status = block.buffer.free_after_completion(); status != cudaSuccess) {
                set_failure(failure, runtime_failure_status(status), "pinned upload allocation free failed");
                return false;
            }
            {
                std::lock_guard<std::mutex> lock(state.mutex);
                pool->totalBytes -= block.capacity;
                state.totalBytesAllContexts -= block.capacity;
                block.capacity = 0;
                publish_pinned_upload_staging_bytes(state.totalBytesAllContexts);
            }
        }
        std::lock_guard<std::mutex> lock(state.mutex);
        state.pools.erase(key);
        return true;
    } catch (...) {
        set_failure(failure, {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "pinned upload retirement bookkeeping failed");
        return false;
    }

} // namespace JuicerCuda
