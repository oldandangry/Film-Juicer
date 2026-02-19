// Cuda/JuicerCudaResourcesServing.cpp
//
// Included by JuicerCudaResources.cpp (single-TU split).
    enum class HostCacheLoadState : int {
        Uninitialized = 0,
        Loading = 1,
        Ready = 2,
        Failed = 3
    };

    struct StbnCpuCache {
        std::vector<std::uint8_t> data;
        int width = 512;
        int height = 512;
        int frames = 256;
        std::uint32_t activeUsers = 0;
        std::uint64_t lastTouchedMs = 0;
        bool loaded = false;
        bool valid = false;
        HostCacheLoadState state = HostCacheLoadState::Uninitialized;
        std::string failureReason;
        std::mutex mutex;
        std::condition_variable cv;
    };

    struct WangCpuCache {
        std::vector<std::uint8_t> tiles;
        std::vector<std::uint8_t> lut;
        int width = 0;
        int height = 0;
        int count = 0;
        int colors = 0;
        std::uint32_t activeUsers = 0;
        std::uint64_t lastTouchedMs = 0;
        bool loaded = false;
        bool valid = false;
        HostCacheLoadState state = HostCacheLoadState::Uninitialized;
        std::string failureReason;
        std::mutex mutex;
        std::condition_variable cv;
    };

    static StbnCpuCache& stbn_cache() {
        static StbnCpuCache cache;
        return cache;
    }

    static WangCpuCache& wang_cache() {
        static WangCpuCache cache;
        return cache;
    }

    static std::atomic<bool> gStbnWarned{ false };
    static std::atomic<bool> gWangWarned{ false };

    enum class HostAssetCacheId : int {
        None = 0,
        Stbn = 1,
        Wang = 2
    };

    struct HostAssetCachePolicyState {
        std::mutex mutex;
        std::uint64_t nextTrimSequence = 1;
    };

    static HostAssetCachePolicyState& host_asset_cache_policy_state() {
        static HostAssetCachePolicyState state;
        return state;
    }

    static const ResourceManager::ResourceManagerConfigEffective& host_asset_cache_config() {
        static const ResourceManager::ResourceManagerConfigEffective cfg =
            ResourceManager::sanitize_config(ResourceManager::ResourceManagerConfigRaw{});
        return cfg;
    }

    static std::uint64_t host_asset_now_ms() {
        using Clock = std::chrono::steady_clock;
        const auto now = Clock::now().time_since_epoch();
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
    }

    static std::size_t stbn_cache_bytes_locked(const StbnCpuCache& cache) {
        return cache.data.size();
    }

    static std::size_t wang_cache_bytes_locked(const WangCpuCache& cache) {
        return cache.tiles.size() + cache.lut.size();
    }

    static void publish_host_asset_cache_bytes(std::size_t bytes) {
        ResourceManager::global_state().hostAssetCacheBytes.store(
            static_cast<std::uint64_t>(bytes), std::memory_order_relaxed);
    }

    static bool host_cache_trim_eligible(const StbnCpuCache& cache) {
        return cache.state == HostCacheLoadState::Ready &&
            cache.activeUsers == 0 &&
            !cache.data.empty();
    }

    static bool host_cache_trim_eligible(const WangCpuCache& cache) {
        return cache.state == HostCacheLoadState::Ready &&
            cache.activeUsers == 0 &&
            (!cache.tiles.empty() || !cache.lut.empty());
    }

    static HostAssetCacheId pick_oldest_host_cache_candidate_locked(
        const StbnCpuCache& stbn,
        const WangCpuCache& wang) {
        const bool stbnEligible = host_cache_trim_eligible(stbn);
        const bool wangEligible = host_cache_trim_eligible(wang);
        if (!stbnEligible && !wangEligible) {
            return HostAssetCacheId::None;
        }
        if (stbnEligible && !wangEligible) {
            return HostAssetCacheId::Stbn;
        }
        if (!stbnEligible && wangEligible) {
            return HostAssetCacheId::Wang;
        }
        if (stbn.lastTouchedMs == wang.lastTouchedMs) {
            return HostAssetCacheId::Stbn;
        }
        return (stbn.lastTouchedMs < wang.lastTouchedMs) ?
            HostAssetCacheId::Stbn :
            HostAssetCacheId::Wang;
    }

    static std::size_t trim_host_cache_locked(
        StbnCpuCache& stbn,
        WangCpuCache& wang,
        HostAssetCacheId which,
        std::uint64_t nowMs) {
        if (which == HostAssetCacheId::Stbn) {
            if (!host_cache_trim_eligible(stbn)) {
                return 0;
            }
            const std::size_t bytes = stbn.data.size();
            stbn.data.clear();
            stbn.data.shrink_to_fit();
            stbn.loaded = false;
            stbn.valid = false;
            stbn.state = HostCacheLoadState::Uninitialized;
            stbn.failureReason.clear();
            stbn.lastTouchedMs = nowMs;
            return bytes;
        }
        if (which == HostAssetCacheId::Wang) {
            if (!host_cache_trim_eligible(wang)) {
                return 0;
            }
            const std::size_t bytes = wang.tiles.size() + wang.lut.size();
            wang.tiles.clear();
            wang.tiles.shrink_to_fit();
            wang.lut.clear();
            wang.lut.shrink_to_fit();
            wang.width = 0;
            wang.height = 0;
            wang.count = 0;
            wang.colors = 0;
            wang.loaded = false;
            wang.valid = false;
            wang.state = HostCacheLoadState::Uninitialized;
            wang.failureReason.clear();
            wang.lastTouchedMs = nowMs;
            return bytes;
        }
        return 0;
    }

    static bool is_idle_trim_candidate(
        const StbnCpuCache& stbn,
        const WangCpuCache& wang,
        HostAssetCacheId which,
        std::uint64_t nowMs,
        std::uint32_t idleTrimMs) {
        if (which == HostAssetCacheId::Stbn) {
            if (!host_cache_trim_eligible(stbn)) {
                return false;
            }
            const std::uint64_t ageMs = (nowMs >= stbn.lastTouchedMs) ? (nowMs - stbn.lastTouchedMs) : 0;
            return ageMs >= static_cast<std::uint64_t>(idleTrimMs);
        }
        if (which == HostAssetCacheId::Wang) {
            if (!host_cache_trim_eligible(wang)) {
                return false;
            }
            const std::uint64_t ageMs = (nowMs >= wang.lastTouchedMs) ? (nowMs - wang.lastTouchedMs) : 0;
            return ageMs >= static_cast<std::uint64_t>(idleTrimMs);
        }
        return false;
    }

    static const char* host_asset_cache_name(HostAssetCacheId which) {
        switch (which) {
        case HostAssetCacheId::Stbn:
            return "stbn";
        case HostAssetCacheId::Wang:
            return "wang";
        default:
            return "none";
        }
    }

    static void trace_host_asset_trim(
        const char* stage,
        const char* reason,
        HostAssetCacheId which,
        std::size_t trimmedBytes,
        std::size_t totalBytes,
        std::uint64_t sequence) {
        if (!JTRACE_ENABLED(2)) {
            return;
        }
        std::ostringstream oss;
        oss << "stage=" << (stage ? stage : "unknown")
            << " reason=" << (reason ? reason : "unknown")
            << " cache=" << host_asset_cache_name(which)
            << " trimmed_bytes=" << trimmedBytes
            << " total_bytes=" << totalBytes
            << " trim_sequence=" << sequence;
        JTRACE("MSHST", oss.str());
    }

    static void trace_host_asset_cap(
        const char* stage,
        std::size_t totalBytes,
        std::size_t capBytes,
        std::size_t trimBudgetBytes,
        std::size_t trimmedBytes) {
        if (!JTRACE_ENABLED(2)) {
            return;
        }
        std::ostringstream oss;
        oss << "stage=" << (stage ? stage : "unknown")
            << " total_bytes=" << totalBytes
            << " cap_bytes=" << capBytes
            << " trim_budget_bytes=" << trimBudgetBytes
            << " trimmed_bytes=" << trimmedBytes;
        JTRACE("MSHCP", oss.str());
    }

    static void trace_host_asset_event(
        const char* cacheName,
        const char* op,
        std::size_t bytes,
        const char* detail = nullptr) {
        if (!JTRACE_ENABLED(2)) {
            return;
        }
        std::ostringstream oss;
        oss << "cache=" << (cacheName ? cacheName : "unknown")
            << " op=" << (op ? op : "unknown")
            << " bytes=" << bytes;
        if (detail && *detail) {
            oss << " detail=" << detail;
        }
        JTRACE("MSHST", oss.str());
    }

    static void enforce_host_asset_cache_policy(const char* stage) {
        const ResourceManager::ResourceManagerConfigEffective& cfg = host_asset_cache_config();
        const std::uint64_t nowMs = host_asset_now_ms();
        const std::size_t capBytes = static_cast<std::size_t>(cfg.hostAssetCacheMaxBytes);
        const std::size_t trimBatchBytes = static_cast<std::size_t>(cfg.hostAssetTrimBatchBytes);

        HostAssetCachePolicyState& policy = host_asset_cache_policy_state();
        std::lock_guard<std::mutex> policyLock(policy.mutex);
        StbnCpuCache& stbn = stbn_cache();
        WangCpuCache& wang = wang_cache();
        std::scoped_lock<std::mutex, std::mutex> cachesLock(stbn.mutex, wang.mutex);

        std::size_t totalBytes = stbn_cache_bytes_locked(stbn) + wang_cache_bytes_locked(wang);
        publish_host_asset_cache_bytes(totalBytes);

        std::size_t trimmedBytesTotal = 0;
        const std::uint64_t trimSequence = policy.nextTrimSequence++;

        HostAssetCacheId idleCandidate = pick_oldest_host_cache_candidate_locked(stbn, wang);
        while (idleCandidate != HostAssetCacheId::None &&
               trimmedBytesTotal < trimBatchBytes &&
               is_idle_trim_candidate(stbn, wang, idleCandidate, nowMs, cfg.hostAssetIdleTrimMs)) {
            const std::size_t trimmed = trim_host_cache_locked(stbn, wang, idleCandidate, nowMs);
            if (trimmed == 0) {
                break;
            }
            trimmedBytesTotal += trimmed;
            totalBytes = stbn_cache_bytes_locked(stbn) + wang_cache_bytes_locked(wang);
            trace_host_asset_trim(stage, "idle_trim", idleCandidate, trimmed, totalBytes, trimSequence);
            idleCandidate = pick_oldest_host_cache_candidate_locked(stbn, wang);
        }

        if (totalBytes > capBytes) {
            ResourceManager::global_state().hostAssetCacheCapHits.fetch_add(1, std::memory_order_relaxed);
            std::size_t capTrimmed = 0;
            for (;;) {
                if (totalBytes <= capBytes) {
                    break;
                }
                if (trimmedBytesTotal >= trimBatchBytes && capTrimmed > 0) {
                    break;
                }
                const HostAssetCacheId candidate = pick_oldest_host_cache_candidate_locked(stbn, wang);
                if (candidate == HostAssetCacheId::None) {
                    break;
                }
                const std::size_t trimmed = trim_host_cache_locked(stbn, wang, candidate, nowMs);
                if (trimmed == 0) {
                    break;
                }
                trimmedBytesTotal += trimmed;
                capTrimmed += trimmed;
                totalBytes = stbn_cache_bytes_locked(stbn) + wang_cache_bytes_locked(wang);
                trace_host_asset_trim(stage, "cap_trim", candidate, trimmed, totalBytes, trimSequence);
            }
            trace_host_asset_cap(stage, totalBytes, capBytes, trimBatchBytes, capTrimmed);
        }

        if (trimmedBytesTotal > 0) {
            ResourceManager::ResourceManagerState& managerState = ResourceManager::global_state();
            managerState.hostAssetCacheTrimEvents.fetch_add(1, std::memory_order_relaxed);
            managerState.hostAssetCacheTrimBytes.fetch_add(
                static_cast<std::uint64_t>(trimmedBytesTotal), std::memory_order_relaxed);
        }

        publish_host_asset_cache_bytes(stbn_cache_bytes_locked(stbn) + wang_cache_bytes_locked(wang));
    }

    #if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)

    struct PinnedUploadContextKey {
        int deviceId = -1;
        void* contextOpaque = nullptr;

        bool operator==(const PinnedUploadContextKey& other) const noexcept {
            return deviceId == other.deviceId &&
                contextOpaque == other.contextOpaque;
        }
    };

    struct PinnedUploadContextKeyHash {
        std::size_t operator()(const PinnedUploadContextKey& key) const noexcept {
            const std::size_t hDevice = std::hash<int>{}(key.deviceId);
            const std::size_t hContext = std::hash<std::uintptr_t>{}(
                reinterpret_cast<std::uintptr_t>(key.contextOpaque));
            return hDevice ^ (hContext + 0x9e3779b9u + (hDevice << 6u) + (hDevice >> 2u));
        }
    };

    struct PinnedUploadBlock {
        std::uint64_t id = 0;
        void* ptr = nullptr;
        std::size_t capacity = 0;
        std::uint64_t lastTouchedMs = 0;
        void* doneEventOpaque = nullptr;
        bool inFlight = false;
        bool reserved = false;
    };

    struct PinnedUploadPool {
        std::vector<PinnedUploadBlock> blocks;
        std::uint64_t nextBlockId = 1;
        std::uint64_t nextTrimSequence = 1;
        std::size_t totalBytes = 0;
    };

    struct PinnedUploadStagingPolicyState {
        std::mutex mutex;
        std::unordered_map<PinnedUploadContextKey, PinnedUploadPool, PinnedUploadContextKeyHash> pools;
        std::size_t totalBytesAllContexts = 0;
    };

    static PinnedUploadStagingPolicyState& pinned_upload_staging_policy_state() {
        static PinnedUploadStagingPolicyState state;
        return state;
    }

    static const ResourceManager::ResourceManagerConfigEffective& pinned_upload_staging_config() {
        static const ResourceManager::ResourceManagerConfigEffective cfg =
            ResourceManager::sanitize_config(ResourceManager::ResourceManagerConfigRaw{});
        return cfg;
    }

    static void publish_pinned_upload_staging_bytes(std::size_t bytes) {
        ResourceManager::global_state().pinnedStagingBytes.store(
            static_cast<std::uint64_t>(bytes), std::memory_order_relaxed);
    }

    static void trace_pinned_staging_event(
        const char* stage,
        const char* eventName,
        const char* reason,
        const PinnedUploadContextKey& key,
        std::size_t bytes,
        std::size_t capBytes,
        std::size_t totalBytes,
        std::size_t trimBatchBytes,
        std::uint64_t sequence) {
        if (!JTRACE_ENABLED(2)) {
            return;
        }
        std::ostringstream oss;
        oss << "stage=" << (stage ? stage : "unknown")
            << " event=" << (eventName ? eventName : "unknown")
            << " reason=" << (reason ? reason : "none")
            << " device_id=" << key.deviceId
            << " context=" << reinterpret_cast<std::uintptr_t>(key.contextOpaque)
            << " bytes=" << bytes
            << " cap_bytes=" << capBytes
            << " total_bytes=" << totalBytes
            << " trim_batch_bytes=" << trimBatchBytes
            << " sequence=" << sequence;
        JTRACE("MSPIN", oss.str());
    }

    static void refresh_pinned_block_completion_locked(
        PinnedUploadBlock& block,
        std::uint64_t nowMs) {
        if (!block.inFlight || !block.doneEventOpaque) {
            return;
        }
        cudaEvent_t doneEvent = reinterpret_cast<cudaEvent_t>(block.doneEventOpaque);
        const cudaError_t queryErr = cudaEventQuery(doneEvent);
        if (queryErr == cudaSuccess) {
            block.inFlight = false;
            block.lastTouchedMs = nowMs;
            return;
        }
        if (queryErr != cudaErrorNotReady) {
            block.inFlight = false;
            block.lastTouchedMs = nowMs;
        }
    }

    static int pick_reusable_pinned_block_locked(
        PinnedUploadPool& pool,
        std::size_t requiredBytes,
        std::uint64_t nowMs) {
        int bestIndex = -1;
        std::size_t bestCapacity = std::numeric_limits<std::size_t>::max();
        for (std::size_t i = 0; i < pool.blocks.size(); ++i) {
            PinnedUploadBlock& block = pool.blocks[i];
            refresh_pinned_block_completion_locked(block, nowMs);
            if (block.reserved || block.inFlight || !block.ptr || block.capacity < requiredBytes) {
                continue;
            }
            if (block.capacity < bestCapacity) {
                bestCapacity = block.capacity;
                bestIndex = static_cast<int>(i);
            }
        }
        return bestIndex;
    }

    static int pick_trim_candidate_pinned_block_locked(
        PinnedUploadPool& pool,
        std::uint64_t nowMs,
        std::uint32_t idleTrimMs,
        bool idleOnly) {
        int candidateIndex = -1;
        std::uint64_t oldestTouched = 0;
        for (std::size_t i = 0; i < pool.blocks.size(); ++i) {
            PinnedUploadBlock& block = pool.blocks[i];
            refresh_pinned_block_completion_locked(block, nowMs);
            if (block.reserved || block.inFlight || !block.ptr) {
                continue;
            }
            if (idleOnly) {
                const std::uint64_t ageMs = (nowMs >= block.lastTouchedMs) ? (nowMs - block.lastTouchedMs) : 0;
                if (ageMs < static_cast<std::uint64_t>(idleTrimMs)) {
                    continue;
                }
            }
            if (candidateIndex < 0 || block.lastTouchedMs < oldestTouched) {
                candidateIndex = static_cast<int>(i);
                oldestTouched = block.lastTouchedMs;
            }
        }
        return candidateIndex;
    }

    static std::size_t trim_pinned_upload_pool_locked(
        PinnedUploadStagingPolicyState& policyState,
        PinnedUploadPool& pool,
        const PinnedUploadContextKey& key,
        const char* stage,
        const char* reason,
        std::size_t trimBudgetBytes,
        std::uint32_t idleTrimMs,
        bool idleOnly,
        std::size_t capBytes) {
        std::size_t trimmedBytes = 0;
        const std::uint64_t nowMs = host_asset_now_ms();
        const std::size_t trimBatchBytes = static_cast<std::size_t>(
            pinned_upload_staging_config().pinnedUploadStagingTrimBatchBytes);
        const std::uint64_t trimSequence = pool.nextTrimSequence++;

        while (trimmedBytes < trimBudgetBytes) {
            const int candidate = pick_trim_candidate_pinned_block_locked(
                pool, nowMs, idleTrimMs, idleOnly);
            if (candidate < 0) {
                break;
            }

            PinnedUploadBlock block = std::move(pool.blocks[static_cast<std::size_t>(candidate)]);
            pool.blocks.erase(pool.blocks.begin() + candidate);
            if (!block.ptr) {
                continue;
            }

            const std::size_t blockBytes = block.capacity;
            if (block.doneEventOpaque) {
                cudaEvent_t doneEvent = reinterpret_cast<cudaEvent_t>(block.doneEventOpaque);
                (void)cudaEventDestroy(doneEvent);
            }
            (void)cudaFreeHost(block.ptr);

            if (pool.totalBytes >= blockBytes) {
                pool.totalBytes -= blockBytes;
            }
            else {
                pool.totalBytes = 0;
            }
            if (policyState.totalBytesAllContexts >= blockBytes) {
                policyState.totalBytesAllContexts -= blockBytes;
            }
            else {
                policyState.totalBytesAllContexts = 0;
            }

            trimmedBytes += blockBytes;
            trace_pinned_staging_event(
                stage,
                "trim",
                reason,
                key,
                blockBytes,
                capBytes,
                policyState.totalBytesAllContexts,
                trimBatchBytes,
                trimSequence);
        }

        if (trimmedBytes > 0) {
            ResourceManager::ResourceManagerState& managerState = ResourceManager::global_state();
            managerState.pinnedStagingTrimEvents.fetch_add(1, std::memory_order_relaxed);
            managerState.pinnedStagingTrimBytes.fetch_add(
                static_cast<std::uint64_t>(trimmedBytes),
                std::memory_order_relaxed);
            publish_pinned_upload_staging_bytes(policyState.totalBytesAllContexts);
        }

        return trimmedBytes;
    }

    struct PinnedUploadReservation {
        bool staged = false;
        PinnedUploadContextKey key{};
        std::uint64_t blockId = 0;
        void* stagingPtr = nullptr;
        std::size_t capBytes = 0;
        std::size_t trimBatchBytes = 0;
        std::string fallbackReason;
    };

    static PinnedUploadReservation reserve_pinned_upload_block(
        std::size_t bytes,
        const char* stage) {
        PinnedUploadReservation result{};
        if (bytes == 0) {
            return result;
        }

        const ResourceManager::ResourceManagerConfigEffective& cfg = pinned_upload_staging_config();
        result.capBytes = static_cast<std::size_t>(cfg.pinnedUploadStagingMaxBytes);
        result.trimBatchBytes = static_cast<std::size_t>(cfg.pinnedUploadStagingTrimBatchBytes);

        int deviceId = -1;
        std::string deviceError;
        if (!query_current_cuda_device(deviceId, deviceError)) {
            result.fallbackReason = "device_query_failed";
            return result;
        }
        void* contextOpaque = nullptr;
        std::string contextError;
        if (!query_current_cuda_context(contextOpaque, contextError) || !contextOpaque) {
            result.fallbackReason = "context_query_failed";
            return result;
        }
        result.key.deviceId = deviceId;
        result.key.contextOpaque = contextOpaque;

        PinnedUploadStagingPolicyState& policyState = pinned_upload_staging_policy_state();
        std::lock_guard<std::mutex> lock(policyState.mutex);
        PinnedUploadPool& pool = policyState.pools[result.key];
        const std::uint64_t nowMs = host_asset_now_ms();

        (void)trim_pinned_upload_pool_locked(
            policyState,
            pool,
            result.key,
            stage,
            "idle_trim",
            result.trimBatchBytes,
            cfg.pinnedUploadStagingIdleTrimMs,
            true,
            result.capBytes);

        int blockIndex = pick_reusable_pinned_block_locked(pool, bytes, nowMs);
        if (blockIndex < 0) {
            if (pool.totalBytes + bytes > result.capBytes) {
                ResourceManager::global_state().pinnedStagingCapHits.fetch_add(1, std::memory_order_relaxed);
                trace_pinned_staging_event(
                    stage,
                    "cap_hit",
                    "cap_before_alloc",
                    result.key,
                    bytes,
                    result.capBytes,
                    policyState.totalBytesAllContexts,
                    result.trimBatchBytes,
                    pool.nextTrimSequence);
                (void)trim_pinned_upload_pool_locked(
                    policyState,
                    pool,
                    result.key,
                    stage,
                    "cap_trim",
                    result.trimBatchBytes,
                    cfg.pinnedUploadStagingIdleTrimMs,
                    false,
                    result.capBytes);
            }

            if (pool.totalBytes + bytes > result.capBytes) {
                result.fallbackReason = "cap_exceeded";
                return result;
            }

            void* pinnedPtr = nullptr;
            const cudaError_t allocErr = cudaMallocHost(&pinnedPtr, bytes);
            if (allocErr != cudaSuccess || !pinnedPtr) {
                result.fallbackReason = "host_alloc_failed";
                return result;
            }

            cudaEvent_t doneEvent = nullptr;
            const cudaError_t eventErr = cudaEventCreateWithFlags(&doneEvent, cudaEventDisableTiming);
            if (eventErr != cudaSuccess || !doneEvent) {
                (void)cudaFreeHost(pinnedPtr);
                result.fallbackReason = "event_create_failed";
                return result;
            }

            PinnedUploadBlock block{};
            block.id = pool.nextBlockId++;
            block.ptr = pinnedPtr;
            block.capacity = bytes;
            block.lastTouchedMs = nowMs;
            block.doneEventOpaque = reinterpret_cast<void*>(doneEvent);
            block.inFlight = false;
            block.reserved = true;
            pool.blocks.push_back(block);
            pool.totalBytes += bytes;
            policyState.totalBytesAllContexts += bytes;

            publish_pinned_upload_staging_bytes(policyState.totalBytesAllContexts);
            trace_pinned_staging_event(
                stage,
                "alloc",
                "new_block",
                result.key,
                bytes,
                result.capBytes,
                policyState.totalBytesAllContexts,
                result.trimBatchBytes,
                pool.nextTrimSequence);

            result.staged = true;
            result.blockId = block.id;
            result.stagingPtr = block.ptr;
            return result;
        }

        PinnedUploadBlock& block = pool.blocks[static_cast<std::size_t>(blockIndex)];
        block.reserved = true;
        block.lastTouchedMs = nowMs;
        result.staged = true;
        result.blockId = block.id;
        result.stagingPtr = block.ptr;
        return result;
    }

    static bool release_pinned_upload_reservation(
        const PinnedUploadReservation& reservation,
        bool copyEnqueued,
        cudaStream_t stream,
        const char* stage,
        std::string& outRecordError) {
        outRecordError.clear();
        if (!reservation.staged || reservation.blockId == 0) {
            return false;
        }

        PinnedUploadStagingPolicyState& policyState = pinned_upload_staging_policy_state();
        std::lock_guard<std::mutex> lock(policyState.mutex);
        const auto poolIt = policyState.pools.find(reservation.key);
        if (poolIt == policyState.pools.end()) {
            return false;
        }
        PinnedUploadPool& pool = poolIt->second;
        for (PinnedUploadBlock& block : pool.blocks) {
            if (block.id != reservation.blockId) {
                continue;
            }
            block.reserved = false;
            block.lastTouchedMs = host_asset_now_ms();
            if (!copyEnqueued) {
                block.inFlight = false;
                return false;
            }
            cudaEvent_t doneEvent = reinterpret_cast<cudaEvent_t>(block.doneEventOpaque);
            const cudaError_t recordErr = cudaEventRecord(doneEvent, stream);
            if (recordErr == cudaSuccess) {
                block.inFlight = true;
                return false;
            }

            outRecordError = std::string("cudaEventRecord(pinned staging) failed: ")
                + (cudaGetErrorString(recordErr) ? cudaGetErrorString(recordErr) : "(unknown)");
            block.inFlight = false;
            trace_pinned_staging_event(
                stage,
                "event_record_failed",
                "sync_fallback_required",
                reservation.key,
                block.capacity,
                reservation.capBytes,
                policyState.totalBytesAllContexts,
                reservation.trimBatchBytes,
                pool.nextTrimSequence);
            return true;
        }
        return false;
    }

    static bool enqueue_host_to_device_copy(
        const char* stage,
        const char* label,
        void* dst,
        const void* src,
        std::size_t bytes,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        if (bytes == 0) {
            return true;
        }
        if (!dst || !src) {
            outError = std::string(label ? label : "copy") + " upload args invalid";
            return false;
        }

        const cudaStream_t stream = cudaStreamOpaque
            ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque)
            : nullptr;

        PinnedUploadReservation reservation = reserve_pinned_upload_block(bytes, stage);
        bool stagedCopyAttempted = false;
        if (reservation.staged && reservation.stagingPtr) {
            stagedCopyAttempted = true;
            std::memcpy(reservation.stagingPtr, src, bytes);
            const cudaError_t stagedErr = cudaMemcpyAsync(
                dst,
                reservation.stagingPtr,
                bytes,
                cudaMemcpyHostToDevice,
                stream);
            const bool copyEnqueued = (stagedErr == cudaSuccess);
            std::string recordError;
            const bool syncRequired = release_pinned_upload_reservation(
                reservation,
                copyEnqueued,
                stream,
                stage,
                recordError);

            if (copyEnqueued) {
                if (syncRequired) {
                    const cudaError_t syncErr = cudaStreamSynchronize(stream);
                    if (syncErr != cudaSuccess) {
                        outError = std::string("cudaStreamSynchronize(")
                            + (label ? label : "upload")
                            + ") failed after pinned staging fallback: "
                            + (cudaGetErrorString(syncErr) ? cudaGetErrorString(syncErr) : "(unknown)");
                        return false;
                    }
                    ResourceManager::global_state().pinnedStagingFallbackEvents.fetch_add(
                        1, std::memory_order_relaxed);
                    trace_pinned_staging_event(
                        stage,
                        "fallback",
                        recordError.empty() ? "event_record_failed" : recordError.c_str(),
                        reservation.key,
                        bytes,
                        reservation.capBytes,
                        ResourceManager::global_state().pinnedStagingBytes.load(std::memory_order_relaxed),
                        reservation.trimBatchBytes,
                        0);
                }
                return true;
            }
        }

        const cudaError_t err = cudaMemcpyAsync(
            dst,
            src,
            bytes,
            cudaMemcpyHostToDevice,
            stream);
        if (err != cudaSuccess) {
            outError = std::string("cudaMemcpyAsync(")
                + (label ? label : "upload")
                + ") failed: "
                + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            return false;
        }

        if (stagedCopyAttempted || !reservation.fallbackReason.empty()) {
            ResourceManager::global_state().pinnedStagingFallbackEvents.fetch_add(
                1, std::memory_order_relaxed);
            trace_pinned_staging_event(
                stage,
                "fallback",
                reservation.fallbackReason.empty() ? "staged_copy_failed" : reservation.fallbackReason.c_str(),
                reservation.key,
                bytes,
                reservation.capBytes,
                ResourceManager::global_state().pinnedStagingBytes.load(std::memory_order_relaxed),
                reservation.trimBatchBytes,
                0);
        }
        return true;
    }

    #endif // defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)

    struct StbnCpuView {
        const std::uint8_t* data = nullptr;
        std::size_t bytes = 0;
        int width = 0;
        int height = 0;
        int frames = 0;
    };

    struct WangCpuView {
        const std::uint8_t* tiles = nullptr;
        const std::uint8_t* lut = nullptr;
        std::size_t tileBytes = 0;
        std::size_t lutBytes = 0;
        int width = 0;
        int height = 0;
        int count = 0;
        int colors = 0;
    };

    static bool acquire_stbn_cpu_view(StbnCpuCache& cache, StbnCpuView& outView, std::string& outError) {
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (cache.state != HostCacheLoadState::Ready || cache.data.empty()) {
            outError = "STBN cache is not ready";
            return false;
        }
        cache.activeUsers += 1;
        cache.lastTouchedMs = host_asset_now_ms();
        outView.data = cache.data.data();
        outView.bytes = cache.data.size();
        outView.width = cache.width;
        outView.height = cache.height;
        outView.frames = cache.frames;
        return true;
    }

    static void release_stbn_cpu_view(StbnCpuCache& cache) {
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (cache.activeUsers > 0) {
            cache.activeUsers -= 1;
        }
        cache.lastTouchedMs = host_asset_now_ms();
    }

    static bool acquire_wang_cpu_view(WangCpuCache& cache, WangCpuView& outView, std::string& outError) {
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (cache.state != HostCacheLoadState::Ready || cache.tiles.empty() || cache.lut.empty()) {
            outError = "Wang cache is not ready";
            return false;
        }
        cache.activeUsers += 1;
        cache.lastTouchedMs = host_asset_now_ms();
        outView.tiles = cache.tiles.data();
        outView.lut = cache.lut.data();
        outView.tileBytes = cache.tiles.size();
        outView.lutBytes = cache.lut.size();
        outView.width = cache.width;
        outView.height = cache.height;
        outView.count = cache.count;
        outView.colors = cache.colors;
        return true;
    }

    static void release_wang_cpu_view(WangCpuCache& cache) {
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (cache.activeUsers > 0) {
            cache.activeUsers -= 1;
        }
        cache.lastTouchedMs = host_asset_now_ms();
    }

    static bool load_stbn_cpu_uncached(const StbnCpuCache& cache, std::vector<std::uint8_t>& outData, std::string& outError) {
        if (gDataDir.empty()) {
            outError = "STBN load failed: data directory missing";
            return false;
        }

        std::filesystem::path path = std::filesystem::path(gDataDir) / "Noise" / "stbn_scalar_512x512x256_u8.bin";
        path.make_preferred();

        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) {
            outError = std::string("STBN load failed: cannot open ") + path.string();
            return false;
        }

        const std::streamsize size = file.tellg();
        if (size <= 0) {
            outError = std::string("STBN load failed: empty file ") + path.string();
            return false;
        }

        const std::size_t expected = static_cast<std::size_t>(cache.width) *
            static_cast<std::size_t>(cache.height) *
            static_cast<std::size_t>(cache.frames);
        if (static_cast<std::size_t>(size) != expected) {
            outError = std::string("STBN load failed: unexpected size for ") + path.string();
            return false;
        }

        outData.resize(expected);
        file.seekg(0, std::ios::beg);
        if (!file.read(reinterpret_cast<char*>(outData.data()), size)) {
            outError = std::string("STBN load failed: read error for ") + path.string();
            outData.clear();
            return false;
        }
        return true;
    }

    static std::size_t wang_lut_index(int l, int r, int t, int b, int colors) {
        const std::size_t c = static_cast<std::size_t>(colors);
        return (((static_cast<std::size_t>(l) * c + static_cast<std::size_t>(r)) * c +
                  static_cast<std::size_t>(t)) * c +
                static_cast<std::size_t>(b));
    }

    struct WangCpuLoadedData {
        std::vector<std::uint8_t> tiles;
        std::vector<std::uint8_t> lut;
        int width = 0;
        int height = 0;
        int count = 0;
        int colors = 0;
    };

    static bool load_wang_cpu_uncached(WangCpuLoadedData& outData, std::string& outError) {
        if (gDataDir.empty()) {
            outError = "Wang tiles load failed: data directory missing";
            return false;
        }

        std::filesystem::path base = std::filesystem::path(gDataDir) / "Noise" / "Wang";
        base.make_preferred();
        std::filesystem::path binPath = base / "wang_tiles_256x256x16_u8.bin";
        std::filesystem::path jsonPath = base / "tiles.json";

        if (!std::filesystem::exists(binPath) || !std::filesystem::exists(jsonPath)) {
            outError = std::string("Wang tiles load failed: missing assets under ") + base.string();
            return false;
        }

        std::ifstream jf(jsonPath);
        if (!jf) {
            outError = std::string("Wang tiles load failed: cannot open ") + jsonPath.string();
            return false;
        }

        nlohmann::json root;
        try {
            jf >> root;
        } catch (const std::exception& e) {
            outError = std::string("Wang tiles load failed: invalid JSON ") + e.what();
            return false;
        }

        if (!root.contains("resolution") || !root.contains("tiles") || !root.contains("colors") || !root.contains("mapping")) {
            outError = "Wang tiles load failed: tiles.json missing required fields";
            return false;
        }

        const int width = root.value("resolution", 0);
        const int height = width;
        const int count = root.value("tiles", 0);
        const int colors = root.value("colors", 0);
        if (width <= 0 || height <= 0 || count <= 0 || colors <= 0) {
            outError = "Wang tiles load failed: invalid metadata in tiles.json";
            return false;
        }

        const std::size_t lutSize = static_cast<std::size_t>(colors) *
            static_cast<std::size_t>(colors) *
            static_cast<std::size_t>(colors) *
            static_cast<std::size_t>(colors);
        outData.lut.assign(lutSize, 0);

        const auto& mapping = root["mapping"];
        if (!mapping.is_array()) {
            outError = "Wang tiles load failed: mapping is not an array";
            return false;
        }

        for (const auto& entry : mapping) {
            if (!entry.contains("index") || !entry.contains("labels")) {
                continue;
            }
            const int idx = entry.value("index", 0);
            const auto& labels = entry["labels"];
            const int l = labels.value("L", 0);
            const int r = labels.value("R", 0);
            const int t = labels.value("T", 0);
            const int b = labels.value("B", 0);
            if (l < 0 || r < 0 || t < 0 || b < 0 ||
                l >= colors || r >= colors || t >= colors || b >= colors) {
                continue;
            }
            const std::size_t lutIndex = wang_lut_index(l, r, t, b, colors);
            if (lutIndex < outData.lut.size() && idx >= 0 && idx < count) {
                outData.lut[lutIndex] = static_cast<std::uint8_t>(idx);
            }
        }

        std::ifstream bin(binPath, std::ios::binary | std::ios::ate);
        if (!bin) {
            outError = std::string("Wang tiles load failed: cannot open ") + binPath.string();
            return false;
        }
        const std::streamsize size = bin.tellg();
        if (size <= 0) {
            outError = std::string("Wang tiles load failed: empty file ") + binPath.string();
            return false;
        }
        const std::size_t expected = static_cast<std::size_t>(width) *
            static_cast<std::size_t>(height) *
            static_cast<std::size_t>(count);
        if (static_cast<std::size_t>(size) != expected) {
            outError = std::string("Wang tiles load failed: unexpected size for ") + binPath.string();
            return false;
        }
        outData.tiles.resize(expected);
        bin.seekg(0, std::ios::beg);
        if (!bin.read(reinterpret_cast<char*>(outData.tiles.data()), size)) {
            outError = std::string("Wang tiles load failed: read error for ") + binPath.string();
            outData.tiles.clear();
            return false;
        }

        outData.width = width;
        outData.height = height;
        outData.count = count;
        outData.colors = colors;
        return true;
    }

    static bool load_stbn_cpu(StbnCpuCache& cache, std::string& outError) {
        outError.clear();
        {
            std::unique_lock<std::mutex> lock(cache.mutex);
            for (;;) {
                if (cache.state == HostCacheLoadState::Ready) {
                    cache.loaded = true;
                    cache.valid = true;
                    cache.lastTouchedMs = host_asset_now_ms();
                    trace_host_asset_event("stbn", "hit", cache.data.size());
                    return true;
                }
                if (cache.state == HostCacheLoadState::Failed) {
                    cache.loaded = true;
                    cache.valid = false;
                    outError = cache.failureReason;
                    trace_host_asset_event("stbn", "failed_cached", 0, cache.failureReason.c_str());
                    return false;
                }
                if (cache.state == HostCacheLoadState::Loading) {
                    cache.cv.wait(lock);
                    continue;
                }
                cache.state = HostCacheLoadState::Loading;
                break;
            }
        }

        std::vector<std::uint8_t> loadedData;
        std::string loadError;
        const bool ok = load_stbn_cpu_uncached(cache, loadedData, loadError);
        {
            std::lock_guard<std::mutex> lock(cache.mutex);
            cache.loaded = true;
            cache.valid = ok;
            if (ok) {
                cache.data = std::move(loadedData);
                cache.failureReason.clear();
                cache.state = HostCacheLoadState::Ready;
                cache.lastTouchedMs = host_asset_now_ms();
                trace_host_asset_event("stbn", "load", cache.data.size());
            }
            else {
                cache.data.clear();
                cache.failureReason = loadError.empty() ? "STBN load failed: unknown error" : loadError;
                cache.state = HostCacheLoadState::Failed;
                outError = cache.failureReason;
                trace_host_asset_event("stbn", "failed_load", 0, cache.failureReason.c_str());
            }
        }
        cache.cv.notify_all();
        return ok;
    }

    static bool load_wang_cpu(WangCpuCache& cache, std::string& outError) {
        outError.clear();
        {
            std::unique_lock<std::mutex> lock(cache.mutex);
            for (;;) {
                if (cache.state == HostCacheLoadState::Ready) {
                    cache.loaded = true;
                    cache.valid = true;
                    cache.lastTouchedMs = host_asset_now_ms();
                    trace_host_asset_event("wang", "hit", cache.tiles.size() + cache.lut.size());
                    return true;
                }
                if (cache.state == HostCacheLoadState::Failed) {
                    cache.loaded = true;
                    cache.valid = false;
                    outError = cache.failureReason;
                    trace_host_asset_event("wang", "failed_cached", 0, cache.failureReason.c_str());
                    return false;
                }
                if (cache.state == HostCacheLoadState::Loading) {
                    cache.cv.wait(lock);
                    continue;
                }
                cache.state = HostCacheLoadState::Loading;
                break;
            }
        }

        WangCpuLoadedData loadedData;
        std::string loadError;
        const bool ok = load_wang_cpu_uncached(loadedData, loadError);
        {
            std::lock_guard<std::mutex> lock(cache.mutex);
            cache.loaded = true;
            cache.valid = ok;
            if (ok) {
                cache.tiles = std::move(loadedData.tiles);
                cache.lut = std::move(loadedData.lut);
                cache.width = loadedData.width;
                cache.height = loadedData.height;
                cache.count = loadedData.count;
                cache.colors = loadedData.colors;
                cache.failureReason.clear();
                cache.state = HostCacheLoadState::Ready;
                cache.lastTouchedMs = host_asset_now_ms();
                trace_host_asset_event("wang", "load", cache.tiles.size() + cache.lut.size());
            }
            else {
                cache.tiles.clear();
                cache.lut.clear();
                cache.width = 0;
                cache.height = 0;
                cache.count = 0;
                cache.colors = 0;
                cache.failureReason = loadError.empty() ? "Wang tiles load failed: unknown error" : loadError;
                cache.state = HostCacheLoadState::Failed;
                outError = cache.failureReason;
                trace_host_asset_event("wang", "failed_load", 0, cache.failureReason.c_str());
            }
        }
        cache.cv.notify_all();
        return ok;
    }

    static void free_stbn(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (resources.stbnData) {
            cudaFree(resources.stbnData);
            resources.stbnData = nullptr;
        }
#endif
        resources.stbnWidth = 0;
        resources.stbnHeight = 0;
        resources.stbnFrames = 0;
    }

    static void free_wang(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (resources.wangTilesData) {
            cudaFree(resources.wangTilesData);
            resources.wangTilesData = nullptr;
        }
        if (resources.wangLutData) {
            cudaFree(resources.wangLutData);
            resources.wangLutData = nullptr;
        }
#endif
        resources.wangWidth = 0;
        resources.wangHeight = 0;
        resources.wangCount = 0;
        resources.wangColors = 0;
    }

    static void free_scan_error_flag(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (resources.scanErrorFlag) {
            cudaFree(resources.scanErrorFlag);
            resources.scanErrorFlag = nullptr;
        }
        if (resources.scanErrorHost) {
            cudaFreeHost(resources.scanErrorHost);
            resources.scanErrorHost = nullptr;
        }
        if (resources.scanErrorEventOpaque) {
            cudaEvent_t ev = reinterpret_cast<cudaEvent_t>(resources.scanErrorEventOpaque);
            cudaEventDestroy(ev);
            resources.scanErrorEventOpaque = nullptr;
        }
#endif
        resources.scanErrorPending = 0;
    }


    bool ensure_uploaded(Resources& resources, const WorkingState& ws, void* cudaStreamOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)ws;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        std::unique_lock<std::mutex> lock(resources.m);
        reap_retire_queue_locked(resources);
        if (!validate_resource_owner_locked(resources, outError, true)) {
            return false;
        }
        enforce_host_asset_cache_policy("ensure_uploaded_pre");

        if (!resources.stbnData) {
            std::string stbnError;
            StbnCpuCache& cache = stbn_cache();
            if (load_stbn_cpu(cache, stbnError)) {
                StbnCpuView view{};
                if (acquire_stbn_cpu_view(cache, view, stbnError)) {
                    struct StbnViewGuard {
                        StbnCpuCache* cache = nullptr;
                        ~StbnViewGuard() {
                            if (cache) {
                                release_stbn_cpu_view(*cache);
                            }
                        }
                    } guard{ &cache };
                    if (view.bytes > 0) {
                        const cudaError_t allocErr = cudaMalloc(reinterpret_cast<void**>(&resources.stbnData), view.bytes);
                        if (allocErr == cudaSuccess) {
                            if (!enqueue_host_to_device_copy(
                                    "ensure_uploaded",
                                    "STBN",
                                    resources.stbnData,
                                    view.data,
                                    view.bytes,
                                    cudaStreamOpaque,
                                    stbnError)) {
                                free_stbn(resources);
                            }
                            else {
                                resources.stbnWidth = view.width;
                                resources.stbnHeight = view.height;
                                resources.stbnFrames = view.frames;
                            }
                        }
                        else {
                            stbnError = std::string("cudaMalloc(STBN) failed: ") + (cudaGetErrorString(allocErr) ? cudaGetErrorString(allocErr) : "(unknown)");
                            free_stbn(resources);
                        }
                    }
                }
            }
            if (!stbnError.empty() && !gStbnWarned.exchange(true)) {
                JTRACE("CUDA", stbnError);
            }
        }

        if (!resources.wangTilesData || !resources.wangLutData) {
            if (resources.wangTilesData || resources.wangLutData) {
                free_wang(resources);
            }
            std::string wangError;
            WangCpuCache& cache = wang_cache();
            if (load_wang_cpu(cache, wangError)) {
                WangCpuView view{};
                if (acquire_wang_cpu_view(cache, view, wangError)) {
                    struct WangViewGuard {
                        WangCpuCache* cache = nullptr;
                        ~WangViewGuard() {
                            if (cache) {
                                release_wang_cpu_view(*cache);
                            }
                        }
                    } guard{ &cache };
                    if (view.tileBytes > 0 && view.lutBytes > 0) {
                        const cudaError_t allocTiles = cudaMalloc(reinterpret_cast<void**>(&resources.wangTilesData), view.tileBytes);
                        const cudaError_t allocLut = cudaMalloc(reinterpret_cast<void**>(&resources.wangLutData), view.lutBytes);
                        if (allocTiles == cudaSuccess && allocLut == cudaSuccess) {
                            const bool tilesOk = enqueue_host_to_device_copy(
                                "ensure_uploaded",
                                "Wang.tiles",
                                resources.wangTilesData,
                                view.tiles,
                                view.tileBytes,
                                cudaStreamOpaque,
                                wangError);
                            const bool lutOk = tilesOk && enqueue_host_to_device_copy(
                                "ensure_uploaded",
                                "Wang.lut",
                                resources.wangLutData,
                                view.lut,
                                view.lutBytes,
                                cudaStreamOpaque,
                                wangError);
                            if (!tilesOk || !lutOk) {
                                free_wang(resources);
                            }
                            else {
                                resources.wangWidth = view.width;
                                resources.wangHeight = view.height;
                                resources.wangCount = view.count;
                                resources.wangColors = view.colors;
                            }
                        }
                        else {
                            wangError = std::string("cudaMalloc(Wang) failed: ") +
                                (cudaGetErrorString(allocTiles != cudaSuccess ? allocTiles : allocLut) ? cudaGetErrorString(allocTiles != cudaSuccess ? allocTiles : allocLut) : "(unknown)");
                            free_wang(resources);
                        }
                    }
                }
            }
            if (!wangError.empty() && !gWangWarned.exchange(true)) {
                JTRACE("CUDA", wangError);
            }
        }
        enforce_host_asset_cache_policy("ensure_uploaded_post");

        if (ws.buildCounter == 0) {
            outError = "WorkingState buildCounter is 0";
            return false;
        }

        const std::uint64_t wsCoreHash =
            (ws.uploadCoreHash != 0) ? ws.uploadCoreHash : ws.coreHash;
        const std::uint64_t wsDirHash = ws.dirHash;
        if (wsCoreHash == 0 || wsDirHash == 0) {
            outError = "WorkingState hash is 0";
            return false;
        }

        const bool coreUpToDate = (resources.uploadedCoreHash != 0) && (resources.uploadedCoreHash == wsCoreHash);
        const bool dirUpToDate = (resources.uploadedDirHash != 0) && (resources.uploadedDirHash == wsDirHash);

        if (coreUpToDate && dirUpToDate) {
            resources.uploadedBuildCounter = ws.buildCounter;
            return true;
        }

        // DIR-only update: avoid a full WorkingState re-upload when only the DIR pre-corrected
        // density curves changed (slider interaction).
        if (coreUpToDate && !dirUpToDate) {
            if (!upload_curve_locked(resources, resources.dirDensB, ws.dirDensB, cudaStreamOpaque, "dirDensB", outError)) return false;
            if (!upload_curve_locked(resources, resources.dirDensG, ws.dirDensG, cudaStreamOpaque, "dirDensG", outError)) return false;
            if (!upload_curve_locked(resources, resources.dirDensR, ws.dirDensR, cudaStreamOpaque, "dirDensR", outError)) return false;

            resources.uploadedDirHash = wsDirHash;
            resources.uploadedBuildCounter = ws.buildCounter;
            return true;
        }

        // Core rebuild required: retire and replace device pointers without blocking sync.
        if (!retire_curve_locked(resources, resources.densB, cudaStreamOpaque, "densB", outError)) return false;
        if (!retire_curve_locked(resources, resources.densG, cudaStreamOpaque, "densG", outError)) return false;
        if (!retire_curve_locked(resources, resources.densR, cudaStreamOpaque, "densR", outError)) return false;
        if (!retire_density_layers_locked(resources, cudaStreamOpaque, "densityCurvesLayers", outError)) return false;
        if (!retire_curve_locked(resources, resources.dirDensB, cudaStreamOpaque, "dirDensB", outError)) return false;
        if (!retire_curve_locked(resources, resources.dirDensG, cudaStreamOpaque, "dirDensG", outError)) return false;
        if (!retire_curve_locked(resources, resources.dirDensR, cudaStreamOpaque, "dirDensR", outError)) return false;
        if (!retire_curve_locked(resources, resources.sensB, cudaStreamOpaque, "sensB", outError)) return false;
        if (!retire_curve_locked(resources, resources.sensG, cudaStreamOpaque, "sensG", outError)) return false;
        if (!retire_curve_locked(resources, resources.sensR, cudaStreamOpaque, "sensR", outError)) return false;
        if (!retire_tables_locked(resources, cudaStreamOpaque, "tables", outError)) return false;
        if (!retire_scan_medium_locked(resources, resources.scanNegative, cudaStreamOpaque, "scanNegative", outError)) return false;
        if (!retire_scan_medium_locked(resources, resources.scanPrint, cudaStreamOpaque, "scanPrint", outError)) return false;
        if (!retire_print_payloads_locked(resources, cudaStreamOpaque, "print payloads", outError)) return false;

        resources.validatedBuildCounter = 0;
        resources.uploadedBuildCounter = 0;
        resources.uploadedCoreHash = 0;
        resources.uploadedDirHash = 0;

        if (!alloc_and_upload_curve(resources.densB, ws.densB, cudaStreamOpaque, outError)) return false;
        if (!alloc_and_upload_curve(resources.densG, ws.densG, cudaStreamOpaque, outError)) return false;
        if (!alloc_and_upload_curve(resources.densR, ws.densR, cudaStreamOpaque, outError)) return false;

        {
            const int nR = static_cast<int>(ws.densR.linear.size());
            const int nG = static_cast<int>(ws.densG.linear.size());
            const int nB = static_cast<int>(ws.densB.linear.size());
            bool wantLayers = ws.hasDensityCurvesLayers;
            bool sizesOk = wantLayers && (nR > 0 && nG > 0 && nB > 0);
            if (sizesOk) {
                for (int layer = 0; layer < 3; ++layer) {
                    sizesOk = sizesOk && (static_cast<int>(ws.densityCurvesLayers[layer][0].size()) == nR);
                    sizesOk = sizesOk && (static_cast<int>(ws.densityCurvesLayers[layer][1].size()) == nG);
                    sizesOk = sizesOk && (static_cast<int>(ws.densityCurvesLayers[layer][2].size()) == nB);
                }
            }
            const bool sameN = (nR == nG && nR == nB);

            if (!sizesOk) {
                free_density_layers(resources);
            }
            else if (!resources.hasDensityCurvesLayers ||
                resources.densityCurvesLayersN != nR ||
                !resources.densityCurvesLayers[0][0]) {
                free_density_layers(resources);
                std::string layersError;
                for (int layer = 0; layer < 3; ++layer) {
                    for (int ch = 0; ch < 3; ++ch) {
                        const int n = (ch == 0) ? nR : (ch == 1 ? nG : nB);
                        if (!alloc_and_upload_array(resources.densityCurvesLayers[layer][ch],
                            ws.densityCurvesLayers[layer][ch].data(),
                            n,
                            cudaStreamOpaque,
                            "grain density layer",
                            layersError))
                        {
                            outError = std::string("upload grain density layers failed: ") + layersError;
                            free_density_layers(resources);
                            return false;
                        }
                    }
                }
                resources.densityCurvesLayersN = sameN ? nR : 0;
                resources.hasDensityCurvesLayers = 1;
            }
        }

        if (!alloc_and_upload_curve(resources.dirDensB, ws.dirDensB, cudaStreamOpaque, outError)) return false;
        if (!alloc_and_upload_curve(resources.dirDensG, ws.dirDensG, cudaStreamOpaque, outError)) return false;
        if (!alloc_and_upload_curve(resources.dirDensR, ws.dirDensR, cudaStreamOpaque, outError)) return false;

        if (!alloc_and_upload_curve(resources.sensB, ws.sensB, cudaStreamOpaque, outError)) return false;
        if (!alloc_and_upload_curve(resources.sensG, ws.sensG, cudaStreamOpaque, outError)) return false;
        if (!alloc_and_upload_curve(resources.sensR, ws.sensR, cudaStreamOpaque, outError)) return false;

        // Upload per-instance reference illuminant tables (Ax/Ay/Az + illum) and keep a host-side copy of S_inv + ref white.
        {
            const int K = ws.tablesRef.K;
            const bool want =
                ws.spdReady &&
                K == Spectral::gShape.K &&
                static_cast<int>(ws.tablesRef.Ax.size()) == K &&
                static_cast<int>(ws.tablesRef.Ay.size()) == K &&
                static_cast<int>(ws.tablesRef.Az.size()) == K &&
                static_cast<int>(ws.tablesRef.illum.size()) == K;

            if (!want) {
                free_tables(resources);
            } else if (resources.tablesK != K || !resources.tablesAx || !resources.tablesAy || !resources.tablesAz || !resources.tablesIllum) {
                free_tables(resources);
                if (!alloc_and_upload_array(resources.tablesAx, ws.tablesRef.Ax.data(), K, cudaStreamOpaque, "tablesAx", outError)) { free_tables(resources); return false; }
                if (!alloc_and_upload_array(resources.tablesAy, ws.tablesRef.Ay.data(), K, cudaStreamOpaque, "tablesAy", outError)) { free_tables(resources); return false; }
                if (!alloc_and_upload_array(resources.tablesAz, ws.tablesRef.Az.data(), K, cudaStreamOpaque, "tablesAz", outError)) { free_tables(resources); return false; }
                if (!alloc_and_upload_array(resources.tablesIllum, ws.tablesRef.illum.data(), K, cudaStreamOpaque, "tablesIllum", outError)) { free_tables(resources); return false; }
                resources.tablesK = K;
            }

            for (int i = 0; i < 9; ++i) resources.spdSInv[i] = ws.spdSInv[i];
            if (ws.spdReady && ws.tablesRef.K > 0) {
                for (int i = 0; i < 3; ++i) resources.refIllumWhiteXYZ[i] = ws.tablesRef.refIllumWhiteXYZ[i];
            } else {
                for (int i = 0; i < 3; ++i) resources.refIllumWhiteXYZ[i] = ws.filmRaw.refIllumWhiteXYZ[i];
            }
        }

        auto upload_scan_medium = [&](Resources::DeviceScanMedium& dst, const Scanner::ScannerMediumRuntime& medium, std::string& outErrorLocal) -> bool {
            const Spectral::SpectralTables* t = medium.tables;
            if (!t || t->K != Spectral::gShape.K) {
                free_scan_medium(dst);
                return true;
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
                outErrorLocal = "scan spectral tables missing required arrays";
                return false;
            }

            // Rebuild if size mismatches or not allocated yet.
            if (dst.tables.K != K || !dst.tables.epsC || !dst.tables.Ax) {
                free_scan_medium(dst);
                if (!alloc_and_upload_array(dst.tables.epsC, t->epsC.data(), K, cudaStreamOpaque, "scan.epsC", outErrorLocal)) { free_scan_medium(dst); return false; }
                if (!alloc_and_upload_array(dst.tables.epsM, t->epsM.data(), K, cudaStreamOpaque, "scan.epsM", outErrorLocal)) { free_scan_medium(dst); return false; }
                if (!alloc_and_upload_array(dst.tables.epsY, t->epsY.data(), K, cudaStreamOpaque, "scan.epsY", outErrorLocal)) { free_scan_medium(dst); return false; }
                if (!alloc_and_upload_array(dst.tables.Ax, t->Ax.data(), K, cudaStreamOpaque, "scan.Ax", outErrorLocal)) { free_scan_medium(dst); return false; }
                if (!alloc_and_upload_array(dst.tables.Ay, t->Ay.data(), K, cudaStreamOpaque, "scan.Ay", outErrorLocal)) { free_scan_medium(dst); return false; }
                if (!alloc_and_upload_array(dst.tables.Az, t->Az.data(), K, cudaStreamOpaque, "scan.Az", outErrorLocal)) { free_scan_medium(dst); return false; }
                if (t->hasBaseline) {
                    if (!alloc_and_upload_array(dst.tables.baseMin, t->baseMin.data(), K, cudaStreamOpaque, "scan.baseMin", outErrorLocal)) { free_scan_medium(dst); return false; }
                }

                dst.tables.K = K;
            }

            // Baseline can toggle without changing K; keep device pointer in sync.
            if (t->hasBaseline) {
                if (!dst.tables.baseMin) {
                    if (!alloc_and_upload_array(dst.tables.baseMin, t->baseMin.data(), K, cudaStreamOpaque, "scan.baseMin", outErrorLocal)) { free_scan_medium(dst); return false; }
                }
            }
            else {
                if (dst.tables.baseMin) {
                    const size_t bytes = static_cast<size_t>(std::max(0, K)) * sizeof(float);
                    if (!retire_ptr_locked(resources, dst.tables.baseMin, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "scan.baseMin", outErrorLocal)) {
                        return false;
                    }
                    dst.tables.baseMin = nullptr;
                }
            }

            dst.tables.hasBaseline = t->hasBaseline ? 1 : 0;
            dst.tables.invYn = (std::isfinite(t->invYn) && t->invYn > 0.0f) ? t->invYn : 1.0f;

            dst.mediumIsNegative = (medium.medium == Scanner::ScannerMedium::Negative) ? 1 : 0;
            for (int i = 0; i < 3; ++i) {
                dst.min_cmy[i] = medium.range.min_cmy[i];
                dst.inv_max_cmy[i] = (std::isfinite(medium.range.inv_max_cmy[i]) && medium.range.inv_max_cmy[i] > 0.0f)
                    ? medium.range.inv_max_cmy[i]
                    : 1.0f;
            }
            return true;
        };

        {
            std::string scanError;
            if (!upload_scan_medium(resources.scanNegative, ws.negativeMediumRuntime, scanError)) {
                outError = std::string("upload scan negative failed: ") + scanError;
                return false;
            }
            if (!upload_scan_medium(resources.scanPrint, ws.printMediumRuntime, scanError)) {
                outError = std::string("upload scan print failed: ") + scanError;
                return false;
            }
        }

        // Print pipeline: upload payloads when a valid print runtime is present.
        {
            const Print::Runtime* prt = ws.printRT.get();
            if (!prt || !Print::profile_is_valid(prt->profile)) {
                free_print_payloads(resources);
            }
            else {
                const Print::Profile& p = prt->profile;

                // Upload print density curves (logE->D) for C/M/Y.
                auto ensure_print_curve = [&](DeviceCurve& dst, const Spectral::Curve& src, const char* label, std::string& err) -> bool {
                    const int n = static_cast<int>(src.lambda_nm.size());
                    const bool want = n > 1 && src.linear.size() == src.lambda_nm.size();
                    if (!want) {
                        free_curve(dst);
                        return true;
                    }
                    if (dst.n != n || !dst.x || !dst.y) {
                        free_curve(dst);
                        if (!alloc_and_upload_curve(dst, src, cudaStreamOpaque, err)) {
                            err = std::string(label) + ": " + err;
                            return false;
                        }
                    }
                    return true;
                };

                std::string printErr;
                if (!ensure_print_curve(resources.printDcC, p.dcC, "print dcC", printErr)) { outError = printErr; return false; }
                if (!ensure_print_curve(resources.printDcM, p.dcM, "print dcM", printErr)) { outError = printErr; return false; }
                if (!ensure_print_curve(resources.printDcY, p.dcY, "print dcY", printErr)) { outError = printErr; return false; }

                // Upload print paper sensitivities (linear domain, pinned to shape).
                const int K = Spectral::gShape.K;
                const bool sensOk =
                    K > 0 &&
                    static_cast<int>(p.sensC_log.linear.size()) == K &&
                    static_cast<int>(p.sensM_log.linear.size()) == K &&
                    static_cast<int>(p.sensY_log.linear.size()) == K;
                if (!sensOk) {
                    free_curve(resources.printSensC);
                    free_curve(resources.printSensM);
                    free_curve(resources.printSensY);
                }
                else {
                    if (!resources.printSensC.y || resources.printSensC.n != K) {
                        free_curve(resources.printSensC);
                        if (!alloc_and_upload_spectral_samples(resources.printSensC, p.sensC_log.linear, cudaStreamOpaque, "print sensC", outError)) { return false; }
                    }
                    if (!resources.printSensM.y || resources.printSensM.n != K) {
                        free_curve(resources.printSensM);
                        if (!alloc_and_upload_spectral_samples(resources.printSensM, p.sensM_log.linear, cudaStreamOpaque, "print sensM", outError)) { return false; }
                    }
                    if (!resources.printSensY.y || resources.printSensY.n != K) {
                        free_curve(resources.printSensY);
                        if (!alloc_and_upload_spectral_samples(resources.printSensY, p.sensY_log.linear, cudaStreamOpaque, "print sensY", outError)) { return false; }
                    }
                }

                auto gamma_safe = [](float v) -> float {
                    return (std::isfinite(v) && v > 0.0f) ? v : 1.0f;
                };
                resources.printGammaC = gamma_safe(p.gammaFactor[0]);
                resources.printGammaM = gamma_safe(p.gammaFactor[1]);
                resources.printGammaY = gamma_safe(p.gammaFactor[2]);

                // Preflash raw is computed for (y=m=c=0, Dneg=0) and cached per WorkingState build.
                if (!resources.printPreflashValid ||
                    resources.printPreflashBuildCounter != ws.buildCounter ||
                    resources.printPreflashShapeK != Spectral::gShape.K) {
                    float preflashRaw[3] = { 0.0f, 0.0f, 0.0f };
                    int shapeK = 0;
                    const bool ok = Precompute::build_print_preflash_raw(ws, *prt, preflashRaw, shapeK);
                    if (!ok) {
                        resources.printPreflashRaw[0] = resources.printPreflashRaw[1] = resources.printPreflashRaw[2] = 0.0f;
                        resources.printPreflashValid = false;
                        resources.printPreflashBuildCounter = ws.buildCounter;
                        resources.printPreflashShapeK = 0;
                    }
                    else {
                        resources.printPreflashRaw[0] = preflashRaw[0];
                        resources.printPreflashRaw[1] = preflashRaw[1];
                        resources.printPreflashRaw[2] = preflashRaw[2];
                        resources.printPreflashValid = true;
                        resources.printPreflashBuildCounter = ws.buildCounter;
                        resources.printPreflashShapeK = shapeK;
                    }
                }
            }
        }

        // Upload Hanatos LUT if available (uploaded once and reused across WorkingState rebuilds).
        {
            Spectral::SpectralContext& ctx = Spectral::context();
            const bool hanatosAvailable = ctx.hanatosAvailable.load(std::memory_order_acquire);
            const int N = ctx.hanSpectra.size;
            const int K = ctx.hanSpectra.numSamples;
            const bool want = hanatosAvailable && N > 0 && K == Spectral::kNumSamples && !ctx.hanSpectra.data.empty();
            if (!want) {
                // If Hanatos becomes unavailable (asset missing/mismatch), drop the device copy.
                if (resources.hanatosLut) {
                    const size_t count = static_cast<size_t>(resources.hanatosN) * static_cast<size_t>(resources.hanatosN) * static_cast<size_t>(Spectral::kNumSamples);
                    const size_t bytes = count * sizeof(float);
                    if (!retire_ptr_locked(resources, resources.hanatosLut, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "Hanatos LUT", outError)) {
                        return false;
                    }
                    resources.hanatosLut = nullptr;
                    resources.hanatosN = 0;
                }
                if (resources.hanatosLutIntegrated) {
                    const size_t count = static_cast<size_t>(resources.hanatosNIntegrated) * static_cast<size_t>(resources.hanatosNIntegrated) * 4u;
                    const size_t bytes = count * sizeof(float);
                    if (!retire_ptr_locked(resources, resources.hanatosLutIntegrated, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "Hanatos integrated LUT", outError)) {
                        return false;
                    }
                    resources.hanatosLutIntegrated = nullptr;
                    resources.hanatosNIntegrated = 0;
                    resources.hanatosIntegratedBuildCounter = 0;
                }
            } else {
                if (!resources.hanatosLut || resources.hanatosN != N) {
                    if (resources.hanatosLut) {
                        const size_t count = static_cast<size_t>(resources.hanatosN) * static_cast<size_t>(resources.hanatosN) * static_cast<size_t>(Spectral::kNumSamples);
                        const size_t bytes = count * sizeof(float);
                        if (!retire_ptr_locked(resources, resources.hanatosLut, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "Hanatos LUT", outError)) {
                            return false;
                        }
                        resources.hanatosLut = nullptr;
                        resources.hanatosN = 0;
                    }

                    const size_t count = static_cast<size_t>(N) * static_cast<size_t>(N) * static_cast<size_t>(K);
                    const size_t bytes = count * sizeof(float);
                    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.hanatosLut), bytes);
                    if (err != cudaSuccess) {
                        outError = std::string("cudaMalloc(Hanatos LUT) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                        free_hanatos(resources);
                        return false;
                    }
                    if (!enqueue_host_to_device_copy(
                            "ensure_uploaded",
                            "Hanatos LUT",
                            resources.hanatosLut,
                            ctx.hanSpectra.data.data(),
                            bytes,
                            cudaStreamOpaque,
                            outError)) {
                        free_hanatos(resources);
                        return false;
                    }
                    resources.hanatosN = N;
                }
            }

            // Build + upload the Hanatos LUT preintegrated with per-instance sensitivities.
            const bool sensOk =
                static_cast<int>(ws.sensB.linear.size()) == K &&
                static_cast<int>(ws.sensG.linear.size()) == K &&
                static_cast<int>(ws.sensR.linear.size()) == K;
            const bool wantIntegrated = want && sensOk;
            if (!wantIntegrated) {
                if (resources.hanatosLutIntegrated) {
                    const size_t count = static_cast<size_t>(resources.hanatosNIntegrated) * static_cast<size_t>(resources.hanatosNIntegrated) * 4u;
                    const size_t bytes = count * sizeof(float);
                    if (!retire_ptr_locked(resources, resources.hanatosLutIntegrated, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "Hanatos integrated LUT", outError)) {
                        return false;
                    }
                    resources.hanatosLutIntegrated = nullptr;
                    resources.hanatosNIntegrated = 0;
                    resources.hanatosIntegratedBuildCounter = 0;
                }
            }
            else {
                const bool needAlloc = (!resources.hanatosLutIntegrated || resources.hanatosNIntegrated != N);
                const bool needUpload = needAlloc || resources.hanatosIntegratedBuildCounter != ws.buildCounter;
                if (needUpload) {
                    // Build CPU LUT outside the resources lock.
                    lock.unlock();
                    std::vector<float> cpu;
                    Precompute::build_hanatos_integrated_lut_cpu(ctx, ws, cpu);
                    lock.lock();
                    reap_retire_queue_locked(resources);

                    const bool stillNeedAlloc = (!resources.hanatosLutIntegrated || resources.hanatosNIntegrated != N);
                    const bool stillNeedUpload = stillNeedAlloc || resources.hanatosIntegratedBuildCounter != ws.buildCounter;
                    if (stillNeedUpload) {
                        const size_t bytes = cpu.size() * sizeof(float);
                        float* dLut = nullptr;
                        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&dLut), bytes);
                        if (err != cudaSuccess) {
                            outError = std::string("cudaMalloc(Hanatos integrated LUT) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                            return false;
                        }
                        if (!enqueue_host_to_device_copy(
                                "ensure_uploaded",
                                "Hanatos integrated LUT",
                                dLut,
                                cpu.data(),
                                bytes,
                                cudaStreamOpaque,
                                outError)) {
                            cudaFree(dLut);
                            return false;
                        }

                        if (resources.hanatosLutIntegrated) {
                            const size_t count = static_cast<size_t>(resources.hanatosNIntegrated) * static_cast<size_t>(resources.hanatosNIntegrated) * 4u;
                            const size_t oldBytes = count * sizeof(float);
                            if (!retire_ptr_locked(resources, resources.hanatosLutIntegrated, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "Hanatos integrated LUT", outError)) {
                                cudaFree(dLut);
                                return false;
                            }
                        }

                        resources.hanatosLutIntegrated = dLut;
                        resources.hanatosNIntegrated = N;
                        resources.hanatosIntegratedBuildCounter = ws.buildCounter;
                    }
                }
            }
        }

        // Upload Mallett 2019 basis if available (uploaded once and reused across WorkingState rebuilds).
        {
            Spectral::SpectralContext& ctx = Spectral::context();
            const bool mallettAvailable = ctx.mallettAvailable.load(std::memory_order_acquire);
            const int K = ctx.mallettBasis.rows;
            const int cols = ctx.mallettBasis.cols;
            const bool want =
                mallettAvailable &&
                K == Spectral::kNumSamples &&
                cols == 3 &&
                !ctx.mallettBasis.data.empty();

            if (!want) {
                if (resources.mallettBasis) {
                    const size_t count = static_cast<size_t>(resources.mallettBasisK) * 3u;
                    const size_t bytes = count * sizeof(float);
                    if (!retire_ptr_locked(resources, resources.mallettBasis, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "Mallett basis", outError)) {
                        return false;
                    }
                    resources.mallettBasis = nullptr;
                    resources.mallettBasisK = 0;
                }
            }
            else if (!resources.mallettBasis || resources.mallettBasisK != K) {
                if (resources.mallettBasis) {
                    const size_t count = static_cast<size_t>(resources.mallettBasisK) * 3u;
                    const size_t bytes = count * sizeof(float);
                    if (!retire_ptr_locked(resources, resources.mallettBasis, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "Mallett basis", outError)) {
                        return false;
                    }
                    resources.mallettBasis = nullptr;
                    resources.mallettBasisK = 0;
                }
                const size_t count = static_cast<size_t>(K) * 3u;
                const size_t bytes = count * sizeof(float);
                cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.mallettBasis), bytes);
                if (err != cudaSuccess) {
                    outError = std::string("cudaMalloc(Mallett basis) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                    free_mallett_basis(resources);
                    return false;
                }
                if (!enqueue_host_to_device_copy(
                        "ensure_uploaded",
                        "Mallett basis",
                        resources.mallettBasis,
                        ctx.mallettBasis.data.data(),
                        bytes,
                        cudaStreamOpaque,
                        outError)) {
                    free_mallett_basis(resources);
                    return false;
                }
                resources.mallettBasisK = K;
            }
        }

        resources.uploadedCoreHash = wsCoreHash;
        resources.uploadedDirHash = wsDirHash;
        resources.uploadedBuildCounter = ws.buildCounter;
        return true;
#endif
    }

    bool ensure_scan_lut(Resources& resources, const WorkingState& ws, bool negativeMedium, void* cudaStreamOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)ws;
        (void)negativeMedium;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        const Scanner::ScannerMediumRuntime& medium = negativeMedium ? ws.negativeMediumRuntime : ws.printMediumRuntime;
        const Scanner::ScannerStaticKey& staticKey = negativeMedium ? ws.negativeStaticKey : ws.printStaticKey;
        const Scanner::ScannerMedium expectedMedium = negativeMedium
            ? Scanner::ScannerMedium::Negative
            : Scanner::ScannerMedium::Print;

        if (!medium.tables || medium.tables->K <= 0) {
            outError = "scan LUT build requested but medium tables are unavailable";
            return false;
        }
        if (medium.medium != expectedMedium || staticKey.medium != expectedMedium) {
            outError = "scan LUT build requested with mismatched medium identity";
            return false;
        }

        if (medium.tables->tablesHash == 0) {
            outError = "scan LUT build requested but medium tables hash is invalid";
            return false;
        }
        if (medium.range.digest == 0) {
            outError = "scan LUT build requested but medium density range hash is invalid";
            return false;
        }
        if (staticKey.tablesHash != medium.tables->tablesHash) {
            outError = "scan LUT build requested but static key tables hash mismatches medium tables hash";
            return false;
        }
        if (staticKey.densityRangeHash != medium.range.digest) {
            outError = "scan LUT build requested but static key density range hash mismatches medium density range hash";
            return false;
        }
        const std::uint32_t res =
            ResourceManager::normalize_scan_lut_resolution(staticKey.lutResolution);
        const std::uint64_t expectedHash = ResourceManager::make_scan_lut_key_digest(
            static_cast<std::uint32_t>(medium.medium),
            medium.tables->tablesHash,
            medium.range.digest,
            res);
        if (expectedHash == 0) {
            outError = "scan LUT key hash invalid";
            return false;
        }

        Resources::DeviceSpectralLut* dst = negativeMedium ? &resources.scanNegativeLut : &resources.scanPrintLut;
        {
            std::lock_guard<std::mutex> lock(resources.m);
            if (!validate_resource_owner_locked(resources, outError, true)) {
                return false;
            }
            if (dst->log2XYZ && dst->res == res && dst->hash == expectedHash) {
                return true;
            }
        }

        // Build CPU LUT first (can overlap with any in-flight GPU work) before we synchronize to
        // safely retire the previous device buffer.
        std::vector<double> cpu;
        if (!Precompute::build_scan_lut_cpu(medium, res, cpu, outError)) {
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(resources.m);
            if (!validate_resource_owner_locked(resources, outError, false)) {
                return false;
            }
            if (dst->log2XYZ && dst->res == res && dst->hash == expectedHash) {
                return true;
            }

            if (dst->log2XYZ) {
                // Retire the previous LUT without blocking the CPU.
                const size_t count = static_cast<size_t>(dst->res) * static_cast<size_t>(dst->res) * static_cast<size_t>(dst->res) * 3u;
                const size_t bytes = count * sizeof(double);
                if (!retire_ptr_locked(resources, dst->log2XYZ, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "scan LUT", outError)) {
                    return false;
                }
                dst->log2XYZ = nullptr;
                dst->res = 0;
                dst->hash = 0;
            }

            double* dLut = nullptr;
            const size_t bytes = cpu.size() * sizeof(double);
            cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&dLut), bytes);
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(scan LUT) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                free_scan_lut(*dst);
                return false;
            }
            if (!enqueue_host_to_device_copy(
                    "ensure_scan_lut",
                    "scan LUT",
                    dLut,
                    cpu.data(),
                    bytes,
                    cudaStreamOpaque,
                    outError)) {
                cudaFree(dLut);
                free_scan_lut(*dst);
                return false;
            }

            dst->log2XYZ = dLut;
            dst->res = res;
            dst->hash = expectedHash;
            return true;
        }
#endif
    }

    bool ensure_scan_error_flag(Resources& resources, void* cudaStreamOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        std::lock_guard<std::mutex> lock(resources.m);
        if (!validate_resource_owner_locked(resources, outError, true)) {
            return false;
        }

        if (!resources.scanErrorFlag) {
            cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.scanErrorFlag), sizeof(int));
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(scan error flag) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                resources.scanErrorFlag = nullptr;
                return false;
            }
        }
        if (!resources.scanErrorHost) {
            cudaError_t err = cudaMallocHost(reinterpret_cast<void**>(&resources.scanErrorHost), sizeof(int));
            if (err != cudaSuccess) {
                resources.scanErrorHost = nullptr;
            }
        }
        if (!resources.scanErrorEventOpaque) {
            cudaEvent_t ev = nullptr;
            cudaError_t err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
            if (err == cudaSuccess && ev) {
                resources.scanErrorEventOpaque = reinterpret_cast<void*>(ev);
            }
        }
        return true;
#endif
    }

    bool ensure_print_illuminant_filtered(
        Resources& resources,
        const WorkingState& ws,
        const Print::Runtime& prt,
        const Print::Params& prm,
        void* cudaStreamOpaque,
        std::string& outError)
    {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)ws;
        (void)prt;
        (void)prm;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        const int K = Spectral::gShape.K;
        if (K <= 0) {
            outError = "spectral shape invalid";
            return false;
        }
        const std::uint64_t wsCoreHash =
            (ws.uploadCoreHash != 0) ? ws.uploadCoreHash : ws.coreHash;
        if (wsCoreHash == 0) {
            outError = "print illuminant filtered requested with invalid upload-core hash";
            return false;
        }

        // Normalize filter step keys for cache parity with ExposePrintStage.
        const float yKey = std::isfinite(prm.yFilter) ? prm.yFilter : 0.0f;
        const float mKey = std::isfinite(prm.mFilter) ? prm.mFilter : 0.0f;
        const float cKey = std::isfinite(prm.cFilter) ? prm.cFilter : 0.0f;

        auto blend = [](float curveVal, float normalizedAmount) -> float {
            const float a = std::isfinite(normalizedAmount) ? normalizedAmount : 0.0f;
            return 1.0f - (1.0f - curveVal) * a;
        };
        auto compose_amount = [](float neutralAmount, float deltaSteps) -> float {
            const float neutral = std::isfinite(neutralAmount)
                ? std::clamp(neutralAmount, 0.0f, 1.0f)
                : 0.0f;
            float ds = std::isfinite(deltaSteps) ? deltaSteps : 0.0f;
            ds = std::clamp(ds, -Print::kEnlargerSteps, Print::kEnlargerSteps);
            const float totalSteps = neutral * Print::kEnlargerSteps + ds;
            return totalSteps / Print::kEnlargerSteps;
        };

        const float yAmount = compose_amount(prt.neutralY, yKey);
        const float mAmount = compose_amount(prt.neutralM, mKey);
        const float cAmount = compose_amount(prt.neutralC, cKey);
        const std::uint64_t neutralFilterHash =
            (prt.neutralFilterHash != 0) ? prt.neutralFilterHash : Print::kDefaultNeutralFilterHash;

        const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

        {
            std::lock_guard<std::mutex> lock(resources.m);
            reap_retire_queue_locked(resources);
            if (!validate_resource_owner_locked(resources, outError, true)) {
                return false;
            }

            const bool cached =
                resources.printIllumFiltered &&
                resources.printIllumK == K &&
                resources.printIllumShapeK == K &&
                resources.printIllumCoreHash == wsCoreHash &&
                resources.printIllumYShiftSteps == yKey &&
                resources.printIllumMShiftSteps == mKey &&
                resources.printIllumCShiftSteps == cKey &&
                resources.printIllumNeutralFilterHash == neutralFilterHash;
            if (cached) {
                return true;
            }
        }

        std::vector<float> cpu;
        cpu.resize(static_cast<size_t>(K));
        for (int i = 0; i < K; ++i) {
            const float Ee = (prt.illumEnlarger.linear.size() > static_cast<size_t>(i))
                ? prt.illumEnlarger.linear[static_cast<size_t>(i)]
                : 1.0f;
            const float fY = blend(
                (prt.filterY.linear.size() > static_cast<size_t>(i)) ? prt.filterY.linear[static_cast<size_t>(i)] : 1.0f,
                yAmount);
            const float fM = blend(
                (prt.filterM.linear.size() > static_cast<size_t>(i)) ? prt.filterM.linear[static_cast<size_t>(i)] : 1.0f,
                mAmount);
            const float fC = blend(
                (prt.filterC.linear.size() > static_cast<size_t>(i)) ? prt.filterC.linear[static_cast<size_t>(i)] : 1.0f,
                cAmount);
            cpu[static_cast<size_t>(i)] = Ee * (fY * fM * fC);
        }

        std::lock_guard<std::mutex> lock(resources.m);
        reap_retire_queue_locked(resources);
        if (!validate_resource_owner_locked(resources, outError, true)) {
            return false;
        }

        const bool cached =
            resources.printIllumFiltered &&
            resources.printIllumK == K &&
            resources.printIllumShapeK == K &&
            resources.printIllumCoreHash == wsCoreHash &&
            resources.printIllumYShiftSteps == yKey &&
            resources.printIllumMShiftSteps == mKey &&
            resources.printIllumCShiftSteps == cKey &&
            resources.printIllumNeutralFilterHash == neutralFilterHash;
        if (cached) {
            return true;
        }

        const bool overwriting = (resources.printIllumFiltered != nullptr) && (resources.printIllumK == K);
        if (resources.printIllumFiltered && resources.printIllumK != K) {
            const size_t oldBytes = static_cast<size_t>(std::max(0, resources.printIllumK)) * sizeof(float);
            if (!retire_ptr_locked(resources, resources.printIllumFiltered, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "print illuminant filtered resize", outError)) {
                return false;
            }
            resources.printIllumFiltered = nullptr;
            resources.printIllumK = 0;
        }
        if (!resources.printIllumFiltered) {
            const size_t bytes = static_cast<size_t>(K) * sizeof(float);
            cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.printIllumFiltered), bytes);
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(print illuminant filtered) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                resources.printIllumFiltered = nullptr;
                resources.printIllumK = 0;
                return false;
            }
            resources.printIllumK = K;
        }

        if (overwriting && resources.lastUseEventOpaque) {
            const cudaEvent_t lastUseEv = reinterpret_cast<cudaEvent_t>(resources.lastUseEventOpaque);
            const cudaError_t waitErr = cudaStreamWaitEvent(stream, lastUseEv, 0);
            if (waitErr != cudaSuccess) {
                outError = std::string("cudaStreamWaitEvent before print illuminant filtered update failed: ") + (cudaGetErrorString(waitErr) ? cudaGetErrorString(waitErr) : "(unknown)");
                return false;
            }
        }

        const size_t bytes = static_cast<size_t>(K) * sizeof(float);
        if (!enqueue_host_to_device_copy(
                "ensure_print_illuminant_filtered",
                "print illuminant filtered",
                resources.printIllumFiltered,
                cpu.data(),
                bytes,
                cudaStreamOpaque,
                outError)) {
            return false;
        }

        resources.printIllumYShiftSteps = yKey;
        resources.printIllumMShiftSteps = mKey;
        resources.printIllumCShiftSteps = cKey;
        resources.printIllumNeutralFilterHash = neutralFilterHash;
        resources.printIllumShapeK = K;
        resources.printIllumBuildCounter = ws.buildCounter;
        resources.printIllumCoreHash = wsCoreHash;
        return true;
#endif
    }

    void purge_pinned_upload_staging_for_context(int deviceId, void* contextOpaque) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (deviceId < 0 || contextOpaque == nullptr) {
            return;
        }

        const PinnedUploadContextKey key{ deviceId, contextOpaque };
        std::vector<PinnedUploadBlock> blocksToFree;
        {
            PinnedUploadStagingPolicyState& policyState = pinned_upload_staging_policy_state();
            std::lock_guard<std::mutex> lock(policyState.mutex);
            const auto it = policyState.pools.find(key);
            if (it == policyState.pools.end()) {
                return;
            }
            const std::uint64_t nowMs = host_asset_now_ms();
            PinnedUploadPool retained{};
            retained.nextBlockId = it->second.nextBlockId;
            retained.nextTrimSequence = it->second.nextTrimSequence;
            for (PinnedUploadBlock& block : it->second.blocks) {
                refresh_pinned_block_completion_locked(block, nowMs);
                if (block.reserved || block.inFlight || !block.ptr) {
                    retained.blocks.push_back(block);
                    retained.totalBytes += block.capacity;
                    continue;
                }
                blocksToFree.push_back(block);
            }

            const std::size_t oldBytes = it->second.totalBytes;
            const std::size_t retainedBytes = retained.totalBytes;
            if (oldBytes >= retainedBytes) {
                const std::size_t reclaimed = oldBytes - retainedBytes;
                if (policyState.totalBytesAllContexts >= reclaimed) {
                    policyState.totalBytesAllContexts -= reclaimed;
                }
                else {
                    policyState.totalBytesAllContexts = 0;
                }
            }

            if (retained.blocks.empty()) {
                policyState.pools.erase(it);
            }
            else {
                it->second = std::move(retained);
            }
            publish_pinned_upload_staging_bytes(policyState.totalBytesAllContexts);
        }

        if (blocksToFree.empty()) {
            return;
        }

        int previousDevice = -1;
        const cudaError_t prevErr = cudaGetDevice(&previousDevice);
        const bool havePreviousDevice = (prevErr == cudaSuccess && previousDevice >= 0);
        const bool needRestore = havePreviousDevice && previousDevice != deviceId;
        (void)cudaSetDevice(deviceId);
        for (const PinnedUploadBlock& block : blocksToFree) {
            if (block.doneEventOpaque) {
                cudaEvent_t doneEvent = reinterpret_cast<cudaEvent_t>(block.doneEventOpaque);
                (void)cudaEventDestroy(doneEvent);
            }
            if (block.ptr) {
                (void)cudaFreeHost(block.ptr);
            }
        }
        if (needRestore) {
            (void)cudaSetDevice(previousDevice);
        }
#else
        (void)deviceId;
        (void)contextOpaque;
#endif
    }
