// Cuda/JuicerCudaResourcesServing.inc
//
// Included by JuicerCudaResources.cpp (single-TU split).
enum class HostCacheLoadState : std::uint8_t {
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
    std::uint64_t version = 0;
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
    std::uint64_t version = 0;
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

static std::atomic<bool> gStbnWarned{false};
static std::atomic<bool> gWangWarned{false};

enum class HostAssetCacheId : std::uint8_t {
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

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
static const ResourceManager::ResourceManagerConfigEffective& host_asset_cache_config() {
    static const ResourceManager::ResourceManagerConfigEffective cfg =
        ResourceManager::sanitize_config(ResourceManager::ResourceManagerConfigRaw{});
    return cfg;
}
#endif

static std::uint64_t host_asset_now_ms() {
    using Clock = std::chrono::steady_clock;
    const auto now = Clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
}

template <typename T>
static std::size_t retained_vector_bytes_locked(const std::vector<T>& values) {
    return values.capacity() * sizeof(T);
}

template <typename T>
static void release_vector_storage(std::vector<T>& values) {
    std::vector<T> empty;
    values.swap(empty);
}

static std::size_t stbn_cache_bytes_locked(const StbnCpuCache& cache) {
    return retained_vector_bytes_locked(cache.data);
}

static std::size_t wang_cache_bytes_locked(const WangCpuCache& cache) {
    return retained_vector_bytes_locked(cache.tiles) +
           retained_vector_bytes_locked(cache.lut);
}

static void publish_host_asset_cache_bytes(std::size_t bytes) {
    ResourceManager::global_state().hostAssetCacheBytes.store(
        static_cast<std::uint64_t>(bytes), std::memory_order_relaxed);
}

static bool host_cache_trim_eligible(const StbnCpuCache& cache) {
    return cache.state != HostCacheLoadState::Loading &&
           cache.activeUsers == 0 &&
           stbn_cache_bytes_locked(cache) > 0;
}

static bool host_cache_trim_eligible(const WangCpuCache& cache) {
    return cache.state != HostCacheLoadState::Loading &&
           cache.activeUsers == 0 &&
           wang_cache_bytes_locked(cache) > 0;
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
    return (stbn.lastTouchedMs < wang.lastTouchedMs) ? HostAssetCacheId::Stbn : HostAssetCacheId::Wang;
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
        const std::size_t bytes = stbn_cache_bytes_locked(stbn);
        release_vector_storage(stbn.data);
        stbn.loaded = false;
        stbn.valid = false;
        stbn.state = HostCacheLoadState::Uninitialized;
        stbn.version = 0;
        stbn.failureReason.clear();
        stbn.lastTouchedMs = nowMs;
        return bytes;
    }
    if (which == HostAssetCacheId::Wang) {
        if (!host_cache_trim_eligible(wang)) {
            return 0;
        }
        const std::size_t bytes = wang_cache_bytes_locked(wang);
        release_vector_storage(wang.tiles);
        release_vector_storage(wang.lut);
        wang.width = 0;
        wang.height = 0;
        wang.count = 0;
        wang.colors = 0;
        wang.version = 0;
        wang.loaded = false;
        wang.valid = false;
        wang.state = HostCacheLoadState::Uninitialized;
        wang.failureReason.clear();
        wang.lastTouchedMs = nowMs;
        return bytes;
    }
    return 0;
}

struct HostCacheIdleTrimQuery {
    std::uint64_t nowMs = 0;
    std::uint32_t idleTrimMs = 0;
};

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
static bool is_idle_trim_candidate(
    const StbnCpuCache& stbn,
    const WangCpuCache& wang,
    HostAssetCacheId which,
    const HostCacheIdleTrimQuery& query) {
    if (which == HostAssetCacheId::Stbn) {
        if (!host_cache_trim_eligible(stbn)) {
            return false;
        }
        const std::uint64_t ageMs = (query.nowMs >= stbn.lastTouchedMs) ? (query.nowMs - stbn.lastTouchedMs) : 0;
        return ageMs >= static_cast<std::uint64_t>(query.idleTrimMs);
    }
    if (which == HostAssetCacheId::Wang) {
        if (!host_cache_trim_eligible(wang)) {
            return false;
        }
        const std::uint64_t ageMs = (query.nowMs >= wang.lastTouchedMs) ? (query.nowMs - wang.lastTouchedMs) : 0;
        return ageMs >= static_cast<std::uint64_t>(query.idleTrimMs);
    }
    return false;
}
#endif

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

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
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
#endif

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
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
#endif

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
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
           is_idle_trim_candidate(stbn, wang, idleCandidate, HostCacheIdleTrimQuery{nowMs, cfg.hostAssetIdleTrimMs})) {
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
        ResourceManager::telemetry_counter_add(ResourceManager::global_state().hostAssetCacheCapHits, 1);
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
        ResourceManager::telemetry_counter_add(managerState.hostAssetCacheTrimEvents, 1);
        ResourceManager::telemetry_counter_add(managerState.hostAssetCacheTrimBytes, static_cast<std::uint64_t>(trimmedBytesTotal));
    }

    publish_host_asset_cache_bytes(stbn_cache_bytes_locked(stbn) + wang_cache_bytes_locked(wang));
}
#endif

void purge_host_asset_caches_if_registry_idle(const char* stage) noexcept {
    try {
        ResourceManager::ResourceManagerState& managerState = ResourceManager::global_state();
        if (managerState.registryLiveManagers.load(std::memory_order_relaxed) != 0) {
            return;
        }

        HostAssetCachePolicyState& policy = host_asset_cache_policy_state();
        std::lock_guard<std::mutex> policyLock(policy.mutex);
        if (managerState.registryLiveManagers.load(std::memory_order_relaxed) != 0) {
            return;
        }

        const std::uint64_t nowMs = host_asset_now_ms();
        StbnCpuCache& stbn = stbn_cache();
        WangCpuCache& wang = wang_cache();
        std::scoped_lock<std::mutex, std::mutex> cachesLock(stbn.mutex, wang.mutex);

        std::size_t totalBytes = stbn_cache_bytes_locked(stbn) + wang_cache_bytes_locked(wang);
        publish_host_asset_cache_bytes(totalBytes);
        if (totalBytes == 0) {
            return;
        }

        std::size_t trimmedBytesTotal = 0;
        const std::uint64_t trimSequence = policy.nextTrimSequence++;
        for (;;) {
            const HostAssetCacheId candidate = pick_oldest_host_cache_candidate_locked(stbn, wang);
            if (candidate == HostAssetCacheId::None) {
                break;
            }
            const std::size_t trimmed = trim_host_cache_locked(stbn, wang, candidate, nowMs);
            if (trimmed == 0) {
                break;
            }
            trimmedBytesTotal += trimmed;
            totalBytes = stbn_cache_bytes_locked(stbn) + wang_cache_bytes_locked(wang);
            trace_host_asset_trim(stage, "registry_idle_purge", candidate, trimmed, totalBytes, trimSequence);
        }

        if (trimmedBytesTotal > 0) {
            ResourceManager::telemetry_counter_add(managerState.hostAssetCacheTrimEvents, 1);
            ResourceManager::telemetry_counter_add(managerState.hostAssetCacheTrimBytes, static_cast<std::uint64_t>(trimmedBytesTotal));
        }

        publish_host_asset_cache_bytes(stbn_cache_bytes_locked(stbn) + wang_cache_bytes_locked(wang));
    } catch (...) {
        JuicerLogging::discard_current_exception();
    }
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

struct PinnedUploadReuseQuery {
    std::size_t requiredBytes = 0;
    std::uint64_t nowMs = 0;
};

static int pick_reusable_pinned_block_locked(
    PinnedUploadPool& pool,
    const PinnedUploadReuseQuery& query) {
    int bestIndex = -1;
    std::size_t bestCapacity = std::numeric_limits<std::size_t>::max();
    for (std::size_t i = 0; i < pool.blocks.size(); ++i) {
        PinnedUploadBlock& block = pool.blocks[i];
        refresh_pinned_block_completion_locked(block, query.nowMs);
        if (block.reserved || block.inFlight || !block.ptr || block.capacity < query.requiredBytes) {
            continue;
        }
        if (block.capacity < bestCapacity) {
            bestCapacity = block.capacity;
            bestIndex = static_cast<int>(i);
        }
    }
    return bestIndex;
}

struct PinnedUploadTrimSearch {
    std::uint64_t nowMs = 0;
    std::uint32_t idleTrimMs = 0;
    bool idleOnly = false;
};

static int pick_trim_candidate_pinned_block_locked(
    PinnedUploadPool& pool,
    const PinnedUploadTrimSearch& search) {
    int candidateIndex = -1;
    std::uint64_t oldestTouched = 0;
    for (std::size_t i = 0; i < pool.blocks.size(); ++i) {
        PinnedUploadBlock& block = pool.blocks[i];
        refresh_pinned_block_completion_locked(block, search.nowMs);
        if (block.reserved || block.inFlight || !block.ptr) {
            continue;
        }
        if (search.idleOnly) {
            const std::uint64_t ageMs = (search.nowMs >= block.lastTouchedMs) ? (search.nowMs - block.lastTouchedMs) : 0;
            if (ageMs < static_cast<std::uint64_t>(search.idleTrimMs)) {
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

static bool pinned_upload_pool_is_empty_locked(const PinnedUploadPool& pool) {
    return pool.blocks.empty() && pool.totalBytes == 0;
}

struct PinnedUploadTrimTarget {
    bool found = false;
    PinnedUploadContextKey key{};
    int blockIndex = -1;
    std::uint64_t lastTouchedMs = 0;
};

static PinnedUploadTrimTarget pick_trim_candidate_across_pools_locked(
    PinnedUploadStagingPolicyState& policyState,
    std::uint64_t nowMs,
    std::uint32_t idleTrimMs,
    bool idleOnly) {
    PinnedUploadTrimTarget target{};
    for (auto& poolEntry : policyState.pools) {
        const int candidate = pick_trim_candidate_pinned_block_locked(
            poolEntry.second,
            PinnedUploadTrimSearch{nowMs, idleTrimMs, idleOnly});
        if (candidate < 0) {
            continue;
        }

        const PinnedUploadBlock& block =
            poolEntry.second.blocks[static_cast<std::size_t>(candidate)];
        if (!target.found || block.lastTouchedMs < target.lastTouchedMs) {
            target.found = true;
            target.key = poolEntry.first;
            target.blockIndex = candidate;
            target.lastTouchedMs = block.lastTouchedMs;
        }
    }
    return target;
}

static std::size_t trim_one_pinned_upload_block_locked(
    PinnedUploadStagingPolicyState& policyState,
    const PinnedUploadContextKey& key,
    int blockIndex,
    const char* stage,
    const char* reason,
    std::size_t capBytes) {
    auto poolIt = policyState.pools.find(key);
    if (poolIt == policyState.pools.end()) {
        return 0;
    }

    PinnedUploadPool& pool = poolIt->second;
    if (blockIndex < 0 ||
        static_cast<std::size_t>(blockIndex) >= pool.blocks.size()) {
        return 0;
    }

    const std::size_t trimBatchBytes = static_cast<std::size_t>(
        pinned_upload_staging_config().pinnedUploadStagingTrimBatchBytes);
    const std::uint64_t trimSequence = pool.nextTrimSequence++;
    PinnedUploadBlock block = pool.blocks[static_cast<std::size_t>(blockIndex)];
    pool.blocks.erase(pool.blocks.begin() + blockIndex);
    if (!block.ptr) {
        if (pinned_upload_pool_is_empty_locked(pool)) {
            policyState.pools.erase(poolIt);
        }
        return 0;
    }

    const std::size_t blockBytes = block.capacity;
    if (block.doneEventOpaque) {
        cudaEvent_t doneEvent = reinterpret_cast<cudaEvent_t>(block.doneEventOpaque);
        (void)cudaEventDestroy(doneEvent);
    }
    (void)cudaFreeHost(block.ptr);

    if (pool.totalBytes >= blockBytes) {
        pool.totalBytes -= blockBytes;
    } else {
        pool.totalBytes = 0;
    }
    if (policyState.totalBytesAllContexts >= blockBytes) {
        policyState.totalBytesAllContexts -= blockBytes;
    } else {
        policyState.totalBytesAllContexts = 0;
    }

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

    if (pinned_upload_pool_is_empty_locked(pool)) {
        policyState.pools.erase(poolIt);
    }

    return blockBytes;
}

struct PinnedUploadTrimPolicy {
    std::size_t trimBudgetBytes = 0;
    std::uint32_t idleTrimMs = 0;
    bool idleOnly = false;
    std::size_t capBytes = 0;
};

static std::size_t trim_pinned_upload_pools_locked(
    PinnedUploadStagingPolicyState& policyState,
    const char* stage,
    const char* reason,
    const PinnedUploadTrimPolicy& trimPolicy) {
    std::size_t trimmedBytes = 0;
    const std::uint64_t nowMs = host_asset_now_ms();

    while (trimmedBytes < trimPolicy.trimBudgetBytes) {
        const PinnedUploadTrimTarget target =
            pick_trim_candidate_across_pools_locked(
                policyState,
                nowMs,
                trimPolicy.idleTrimMs,
                trimPolicy.idleOnly);
        if (!target.found) {
            break;
        }

        const std::size_t trimmed = trim_one_pinned_upload_block_locked(
            policyState,
            target.key,
            target.blockIndex,
            stage,
            reason,
            trimPolicy.capBytes);
        if (trimmed == 0) {
            break;
        }

        trimmedBytes += trimmed;
    }

    if (trimmedBytes > 0) {
        ResourceManager::ResourceManagerState& managerState = ResourceManager::global_state();
        ResourceManager::telemetry_counter_add(managerState.pinnedStagingTrimEvents, 1);
        ResourceManager::telemetry_counter_add(managerState.pinnedStagingTrimBytes, static_cast<std::uint64_t>(trimmedBytes));
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
    const std::uint64_t nowMs = host_asset_now_ms();

    (void)trim_pinned_upload_pools_locked(
        policyState,
        stage,
        "idle_trim",
        PinnedUploadTrimPolicy{
            result.trimBatchBytes,
            cfg.pinnedUploadStagingIdleTrimMs,
            true,
            result.capBytes});

    auto poolIt = policyState.pools.find(result.key);
    int blockIndex = -1;
    if (poolIt != policyState.pools.end()) {
        blockIndex = pick_reusable_pinned_block_locked(
            poolIt->second,
            PinnedUploadReuseQuery{bytes, nowMs});
    }
    if (blockIndex < 0) {
        if (bytes > result.capBytes) {
            result.fallbackReason = "cap_exceeded";
            return result;
        }

        std::size_t requiredTrimBytes = 0;
        if (policyState.totalBytesAllContexts > result.capBytes - bytes) {
            requiredTrimBytes =
                policyState.totalBytesAllContexts - (result.capBytes - bytes);
        }
        if (requiredTrimBytes > 0) {
            ResourceManager::telemetry_counter_add(ResourceManager::global_state().pinnedStagingCapHits, 1);
            trace_pinned_staging_event(
                stage,
                "cap_hit",
                "cap_before_alloc",
                result.key,
                bytes,
                result.capBytes,
                policyState.totalBytesAllContexts,
                result.trimBatchBytes,
                0);
            const std::size_t capTrimBudgetBytes =
                std::max(result.trimBatchBytes, requiredTrimBytes);
            (void)trim_pinned_upload_pools_locked(
                policyState,
                stage,
                "cap_trim",
                PinnedUploadTrimPolicy{
                    capTrimBudgetBytes,
                    cfg.pinnedUploadStagingIdleTrimMs,
                    false,
                    result.capBytes});
        }

        if (policyState.totalBytesAllContexts > result.capBytes - bytes) {
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

        PinnedUploadPool& pool = policyState.pools[result.key];
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

    PinnedUploadBlock& block =
        poolIt->second.blocks[static_cast<std::size_t>(blockIndex)];
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

        outRecordError = std::string("cudaEventRecord(pinned staging) failed: ") + (cudaGetErrorString(recordErr) ? cudaGetErrorString(recordErr) : "(unknown)");
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
    const HostToDeviceCopyRequest& request,
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
        outError = std::string(request.label ? request.label : "copy") + " upload args invalid";
        return false;
    }

    const cudaStream_t stream = cudaStreamOpaque
                                    ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque)
                                    : nullptr;

    PinnedUploadReservation reservation = reserve_pinned_upload_block(bytes, request.stage);
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
            request.stage,
            recordError);

        if (copyEnqueued) {
            if (syncRequired) {
                const cudaError_t syncErr = cudaStreamSynchronize(stream);
                if (syncErr != cudaSuccess) {
                    outError = std::string("cudaStreamSynchronize(") + (request.label ? request.label : "upload") + ") failed after pinned staging fallback: " + (cudaGetErrorString(syncErr) ? cudaGetErrorString(syncErr) : "(unknown)");
                    return false;
                }
                ResourceManager::telemetry_counter_add(ResourceManager::global_state().pinnedStagingFallbackEvents, 1);
                trace_pinned_staging_event(
                    request.stage,
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
        outError = std::string("cudaMemcpyAsync(") + (request.label ? request.label : "upload") + ") failed: " + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
        return false;
    }

    if (stagedCopyAttempted || !reservation.fallbackReason.empty()) {
        ResourceManager::telemetry_counter_add(ResourceManager::global_state().pinnedStagingFallbackEvents, 1);
        trace_pinned_staging_event(
            request.stage,
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

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
struct StbnCpuView {
    const std::uint8_t* data = nullptr;
    std::size_t bytes = 0;
    int width = 0;
    int height = 0;
    int frames = 0;
    std::uint64_t version = 0;
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
    std::uint64_t version = 0;
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
    outView.version = cache.version;
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
    outView.version = cache.version;
    return true;
}

static void release_wang_cpu_view(WangCpuCache& cache) {
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (cache.activeUsers > 0) {
        cache.activeUsers -= 1;
    }
    cache.lastTouchedMs = host_asset_now_ms();
}

struct StbnCpuLoadedData {
    std::vector<std::uint8_t> data;
    int width = 0;
    int height = 0;
    int frames = 0;
    std::uint64_t version = 0;
};

static bool load_stbn_cpu_from_assets(StbnCpuLoadedData& outData, std::string& outError) {
    const auto payloads = JuicerProcess::root().assets().static_noise_payloads();
    if (!payloads || !payloads->stbn.valid) {
        outError = payloads ? payloads->stbn.error : "STBN load failed: asset payload unavailable";
        return false;
    }

    outData.data = payloads->stbn.data;
    outData.width = payloads->stbn.width;
    outData.height = payloads->stbn.height;
    outData.frames = payloads->stbn.frames;
    outData.version = payloads->stbn.version;
    return true;
}

struct WangCpuLoadedData {
    std::vector<std::uint8_t> tiles;
    std::vector<std::uint8_t> lut;
    int width = 0;
    int height = 0;
    int count = 0;
    int colors = 0;
    std::uint64_t version = 0;
};

static bool load_wang_cpu_from_assets(WangCpuLoadedData& outData, std::string& outError) {
    const auto payloads = JuicerProcess::root().assets().static_noise_payloads();
    if (!payloads || !payloads->wang.valid) {
        outError = payloads ? payloads->wang.error : "Wang tiles load failed: asset payload unavailable";
        return false;
    }

    outData.tiles = payloads->wang.tiles;
    outData.lut = payloads->wang.lut;
    outData.width = payloads->wang.width;
    outData.height = payloads->wang.height;
    outData.count = payloads->wang.count;
    outData.colors = payloads->wang.colors;
    outData.version = payloads->wang.version;
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

    StbnCpuLoadedData loadedData;
    std::string loadError;
    const bool ok = load_stbn_cpu_from_assets(loadedData, loadError);
    {
        std::lock_guard<std::mutex> lock(cache.mutex);
        cache.loaded = true;
        cache.valid = ok;
        if (ok) {
            cache.data = std::move(loadedData.data);
            cache.width = loadedData.width;
            cache.height = loadedData.height;
            cache.frames = loadedData.frames;
            cache.version = loadedData.version;
            cache.failureReason.clear();
            cache.state = HostCacheLoadState::Ready;
            cache.lastTouchedMs = host_asset_now_ms();
            trace_host_asset_event("stbn", "load", cache.data.size());
        } else {
            release_vector_storage(cache.data);
            cache.failureReason = loadError.empty() ? "STBN load failed: unknown error" : loadError;
            cache.state = HostCacheLoadState::Failed;
            cache.lastTouchedMs = host_asset_now_ms();
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
    const bool ok = load_wang_cpu_from_assets(loadedData, loadError);
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
            cache.version = loadedData.version;
            cache.failureReason.clear();
            cache.state = HostCacheLoadState::Ready;
            cache.lastTouchedMs = host_asset_now_ms();
            trace_host_asset_event("wang", "load", cache.tiles.size() + cache.lut.size());
        } else {
            release_vector_storage(cache.tiles);
            release_vector_storage(cache.lut);
            cache.width = 0;
            cache.height = 0;
            cache.count = 0;
            cache.colors = 0;
            cache.failureReason = loadError.empty() ? "Wang tiles load failed: unknown error" : loadError;
            cache.state = HostCacheLoadState::Failed;
            cache.lastTouchedMs = host_asset_now_ms();
            outError = cache.failureReason;
            trace_host_asset_event("wang", "failed_load", 0, cache.failureReason.c_str());
        }
    }
    cache.cv.notify_all();
    return ok;
}
#endif

static void free_stbn(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    if (resources.stbnData) {
        free_owned_device_noexcept(resources, resources.stbnData);
        resources.stbnData = nullptr;
    }
#endif
    resources.stbnWidth = 0;
    resources.stbnHeight = 0;
    resources.stbnFrames = 0;
    resources.grainStaticAssetVersion = 0;
}

static void free_wang(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    if (resources.wangTilesData) {
        free_owned_device_noexcept(resources, resources.wangTilesData);
        resources.wangTilesData = nullptr;
    }
    if (resources.wangLutData) {
        free_owned_device_noexcept(resources, resources.wangLutData);
        resources.wangLutData = nullptr;
    }
#endif
    resources.wangWidth = 0;
    resources.wangHeight = 0;
    resources.wangCount = 0;
    resources.wangColors = 0;
    resources.grainStaticAssetVersion = 0;
}

static void release_scan_error_readback_entry(
    Resources::PendingScanErrorReadback& entry,
    bool waitForEvent) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    cudaEvent_t ev = entry.eventOpaque
                         ? reinterpret_cast<cudaEvent_t>(entry.eventOpaque)
                         : nullptr;
    if (waitForEvent && ev) {
        (void)cudaEventSynchronize(ev);
    }
    if (entry.host) {
        cudaFreeHost(entry.host);
    }
    if (ev) {
        cudaEventDestroy(ev);
    }
#else
    (void)waitForEvent;
#endif
    entry = Resources::PendingScanErrorReadback{};
}

static void free_scan_error_readbacks(Resources& resources) noexcept {
    for (Resources::PendingScanErrorReadback& entry : resources.pendingScanErrorReadbacks) {
        release_scan_error_readback_entry(entry, true);
    }
    resources.pendingScanErrorReadbacks.clear();
}


bool retain_scan_error_readback(
    Resources& resources,
    int*& host,
    void*& eventOpaque,
    std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
    (void)resources;
    (void)host;
    (void)eventOpaque;
    outError = "CUDA is not enabled";
    return false;
#else
    outError.clear();
    if (!host || !eventOpaque) {
        outError = "scan error readback staging missing";
        return false;
    }

    std::lock_guard<std::mutex> lock(resources.m);
    if (!validate_resource_owner_locked(resources, outError, true)) {
        return false;
    }

    try {
        Resources::PendingScanErrorReadback entry{};
        entry.host = host;
        entry.eventOpaque = eventOpaque;
        resources.pendingScanErrorReadbacks.push_back(entry);
    } catch (...) {
        outError = "scan error readback retention failed";
        return false;
    }

    host = nullptr;
    eventOpaque = nullptr;
    return true;
#endif
}

bool poll_scan_error_readbacks(
    Resources& resources,
    void* cudaStreamOpaque,
    bool& outDetected,
    std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
    (void)resources;
    (void)cudaStreamOpaque;
    outDetected = false;
    outError = "CUDA is not enabled";
    return false;
#else
    outDetected = false;
    outError.clear();

    std::lock_guard<std::mutex> lock(resources.m);
    if (!validate_resource_owner_locked(resources, outError, true)) {
        return false;
    }

    std::vector<Resources::PendingScanErrorReadback>& pending =
        resources.pendingScanErrorReadbacks;
    for (std::size_t i = 0; i < pending.size();) {
        Resources::PendingScanErrorReadback& entry = pending[i];
        cudaEvent_t ev = entry.eventOpaque
                             ? reinterpret_cast<cudaEvent_t>(entry.eventOpaque)
                             : nullptr;
        if (!entry.host || !ev) {
            release_scan_error_readback_entry(entry, false);
            pending.erase(pending.begin() + static_cast<std::ptrdiff_t>(i));
            continue;
        }

        const cudaError_t pollErr = cudaEventQuery(ev);
        if (pollErr == cudaSuccess) {
            if (*entry.host != 0) {
                outDetected = true;
            }
            release_scan_error_readback_entry(entry, false);
            pending.erase(pending.begin() + static_cast<std::ptrdiff_t>(i));
            if (outDetected) {
                return true;
            }
            continue;
        }
        if (pollErr == cudaErrorNotReady) {
            cudaStream_t stream = cudaStreamOpaque
                                      ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque)
                                      : nullptr;
            const cudaError_t waitErr = cudaStreamWaitEvent(stream, ev, 0);
            if (waitErr != cudaSuccess) {
                outError = std::string("CUDA scan error stream wait failed: ") +
                           (cudaGetErrorString(waitErr) ? cudaGetErrorString(waitErr) : "(unknown)");
                return false;
            }
            ++i;
            continue;
        }

        outError = std::string("CUDA scan error event query failed: ") +
                   (cudaGetErrorString(pollErr) ? cudaGetErrorString(pollErr) : "(unknown)");
        return false;
    }
    return true;
#endif
}


bool ensure_grain_static_assets_uploaded(
    Resources& resources,
    std::uint64_t expectedAssetVersion,
    void* cudaStreamOpaque,
    std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
    (void)resources;
    (void)expectedAssetVersion;
    (void)cudaStreamOpaque;
    outError = "CUDA is not enabled";
    return false;
#else
    std::lock_guard<std::mutex> servingUpdateLock(resources.servingUpdateMutex);
    std::unique_lock<std::mutex> lock(resources.m);
    reap_retire_queue_locked(resources);
    if (!validate_resource_owner_locked(resources, outError, true)) {
        return false;
    }

    if (expectedAssetVersion == 0) {
        outError = "grain static asset version is invalid";
        return false;
    }

    const bool needStbnUpload = !resources.stbnData;
    const bool needWangUpload = !resources.wangTilesData || !resources.wangLutData;
    if (!needStbnUpload && !needWangUpload) {
        if (resources.grainStaticAssetVersion != expectedAssetVersion ||
            resources.stbnWidth <= 0 || resources.stbnHeight <= 0 ||
            resources.stbnFrames <= 0 || resources.wangWidth <= 0 ||
            resources.wangHeight <= 0 || resources.wangCount <= 0 ||
            resources.wangColors <= 0) {
            outError = "grain static resource identity mismatch";
            return false;
        }
        return true;
    }
    lock.unlock();

    enforce_host_asset_cache_policy("ensure_grain_static_pre");

    StbnCpuCache* stbnCache = nullptr;
    StbnCpuView stbnView{};
    std::string stbnError;
    if (needStbnUpload) {
        StbnCpuCache& cache = stbn_cache();
        if (load_stbn_cpu(cache, stbnError) &&
            acquire_stbn_cpu_view(cache, stbnView, stbnError)) {
            stbnCache = &cache;
        }
    }

    WangCpuCache* wangCache = nullptr;
    WangCpuView wangView{};
    std::string wangError;
    if (needWangUpload) {
        WangCpuCache& cache = wang_cache();
        if (load_wang_cpu(cache, wangError) &&
            acquire_wang_cpu_view(cache, wangView, wangError)) {
            wangCache = &cache;
        }
    }

    if ((stbnCache && stbnView.version != expectedAssetVersion) ||
        (wangCache && wangView.version != expectedAssetVersion)) {
        if (stbnCache) {
            release_stbn_cpu_view(*stbnCache);
        }
        if (wangCache) {
            release_wang_cpu_view(*wangCache);
        }
        outError = "grain static host asset identity mismatch";
        return false;
    }

    std::string resourceOwnerError;
    lock.lock();
    reap_retire_queue_locked(resources);
    if (!validate_resource_owner_locked(resources, resourceOwnerError, true)) {
        lock.unlock();
        if (stbnCache) {
            release_stbn_cpu_view(*stbnCache);
        }
        if (wangCache) {
            release_wang_cpu_view(*wangCache);
        }
        outError = resourceOwnerError;
        return false;
    }

    if (!resources.stbnData && stbnCache) {
        if (stbnView.bytes > 0) {
            void* stbnData = nullptr;
            if (!alloc_and_upload_bytes_locked(
                    resources,
                    stbnData,
                    stbnView.data,
                    stbnView.bytes,
                    cudaStreamOpaque,
                    &lock,
                    "STBN",
                    stbnError)) {
                free_stbn(resources);
            } else if (!resources.stbnData) {
                resources.stbnData = reinterpret_cast<std::uint8_t*>(stbnData);
                resources.stbnWidth = stbnView.width;
                resources.stbnHeight = stbnView.height;
                resources.stbnFrames = stbnView.frames;
            } else if (stbnData) {
                free_owned_device_noexcept(resources, stbnData);
            }
        }
    }

    if (!resources.wangTilesData || !resources.wangLutData) {
        if (resources.wangTilesData || resources.wangLutData) {
            free_wang(resources);
        }
        if (wangCache && wangView.tileBytes > 0 && wangView.lutBytes > 0) {
            void* wangTilesData = nullptr;
            void* wangLutData = nullptr;
            const bool tilesOk = alloc_and_upload_bytes_locked(
                resources,
                wangTilesData,
                wangView.tiles,
                wangView.tileBytes,
                cudaStreamOpaque,
                &lock,
                "Wang.tiles",
                wangError);
            const bool lutOk = tilesOk && alloc_and_upload_bytes_locked(
                                              resources,
                                              wangLutData,
                                              wangView.lut,
                                              wangView.lutBytes,
                                              cudaStreamOpaque,
                                              &lock,
                                              "Wang.lut",
                                              wangError);
            if (!tilesOk || !lutOk) {
                if (wangTilesData) {
                    free_owned_device_noexcept(resources, wangTilesData);
                }
                if (wangLutData) {
                    free_owned_device_noexcept(resources, wangLutData);
                }
                free_wang(resources);
            } else if (!resources.wangTilesData && !resources.wangLutData) {
                resources.wangTilesData = reinterpret_cast<std::uint8_t*>(wangTilesData);
                resources.wangLutData = reinterpret_cast<std::uint8_t*>(wangLutData);
                resources.wangWidth = wangView.width;
                resources.wangHeight = wangView.height;
                resources.wangCount = wangView.count;
                resources.wangColors = wangView.colors;
            } else {
                if (wangTilesData) {
                    free_owned_device_noexcept(resources, wangTilesData);
                }
                if (wangLutData) {
                    free_owned_device_noexcept(resources, wangLutData);
                }
            }
        }
    }

    lock.unlock();
    if (stbnCache) {
        release_stbn_cpu_view(*stbnCache);
    }
    if (wangCache) {
        release_wang_cpu_view(*wangCache);
    }
    enforce_host_asset_cache_policy("ensure_grain_static_post");
    if (!stbnError.empty() && !gStbnWarned.exchange(true)) {
        JTRACE("CUDA", stbnError);
    }
    if (!wangError.empty() && !gWangWarned.exchange(true)) {
        JTRACE("CUDA", wangError);
    }
    lock.lock();
    reap_retire_queue_locked(resources);
    if (!validate_resource_owner_locked(resources, outError, true)) {
        return false;
    }
    const bool ready =
        resources.stbnData && resources.stbnWidth > 0 &&
        resources.stbnHeight > 0 && resources.stbnFrames > 0 &&
        resources.wangTilesData && resources.wangLutData &&
        resources.wangWidth > 0 && resources.wangHeight > 0 &&
        resources.wangCount > 0 && resources.wangColors > 0;
    if (!ready) {
        outError = !stbnError.empty() ? stbnError : (!wangError.empty() ? wangError : "grain static resource upload incomplete");
        return false;
    }
    resources.grainStaticAssetVersion = expectedAssetVersion;
    return true;
#endif
}

void purge_pinned_upload_staging_for_context(int deviceId, void* contextOpaque) noexcept {
    try {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (deviceId < 0 || contextOpaque == nullptr) {
            return;
        }

        const PinnedUploadContextKey key{deviceId, contextOpaque};
        std::vector<PinnedUploadBlock> blocksToFree;
        const bool registryIdle =
            ResourceManager::global_state().registryLiveManagers.load(std::memory_order_relaxed) == 0;
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
            retained.blocks.reserve(it->second.blocks.size());
            blocksToFree.reserve(it->second.blocks.size());
            for (PinnedUploadBlock& block : it->second.blocks) {
                refresh_pinned_block_completion_locked(block, nowMs);
                if (block.reserved || !block.ptr) {
                    retained.blocks.push_back(block);
                    retained.totalBytes += block.capacity;
                    continue;
                }
                if (block.inFlight && !registryIdle) {
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
                } else {
                    policyState.totalBytesAllContexts = 0;
                }
            }

            if (retained.blocks.empty()) {
                policyState.pools.erase(it);
            } else {
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
            if (block.inFlight && block.doneEventOpaque) {
                cudaEvent_t doneEvent = reinterpret_cast<cudaEvent_t>(block.doneEventOpaque);
                (void)cudaEventSynchronize(doneEvent);
            }
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
    } catch (...) {
        JuicerLogging::discard_current_exception();
    }
}
