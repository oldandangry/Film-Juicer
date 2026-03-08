// Cuda/ResourceManager/JuicerCudaResourceManagerCommands.cpp
//
// Included by JuicerCudaResourceManager.cpp (single-TU split).
bool ensure_active_for_command(
    const SubmissionTransaction& transaction,
    std::string& outError,
    const char* commandName) {
    const char* stage = trace_or(commandName, "command");
    const char* rejectionStage = trace_or(commandName, "command_requires_active_submission");
    if (!validate_lifecycle_for_stage(transaction, stage, false, &outError)) {
        telemetry_record_module_boundary_violation();
        telemetry_trace_module_boundary_violation(
            transaction.transactionId,
            transaction.snapshot.snapshotId,
            transaction.snapshot.traceSchemaVersion,
            rejectionStage);
        return false;
    }

    return validate_stale_tuple_for_stage(
        transaction,
        stage,
        LeaseObservationMode::ActiveOnly,
        "stale transaction in command path (reason=",
        &outError,
        true,
        rejectionStage);
}

const char* prewarm_result_reason(const std::string& prewarmError) noexcept {
    if (prewarmError.empty()) {
        return "ok";
    }
    return "skip";
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

std::string commands_error_or_message(const std::string& error, const char* fallback) {
    if (error.empty()) {
        return std::string(fallback);
    }
    return error;
}

void complete_tier_circuit_attempt(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const TierCircuitAttempt& attempt,
    bool success,
    const char* successReason,
    const std::string& failureError) {
    const char* stageName = trace_or_non_empty(commandName, "command");
    if (success) {
        tier_circuit_record_outcome(
            transaction,
            stageName,
            attempt,
            true,
            trace_or(successReason, "ensure_success"));
        return;
    }
    if (tier_circuit_should_count_failure(failureError)) {
        tier_circuit_record_outcome(
            transaction,
            stageName,
            attempt,
            false,
            failureError.c_str());
        return;
    }
    tier_circuit_cancel_attempt(
        transaction,
        stageName,
        attempt,
        "ignored_policy_failure");
}

void add_estimate_bytes_u64(std::uint64_t bytes, std::uint64_t& total, bool& overflow) noexcept {
    if (overflow) {
        return;
    }
    std::uint64_t next = 0;
    if (!add_u64_checked(total, bytes, next)) {
        total = std::numeric_limits<std::uint64_t>::max();
        overflow = true;
        return;
    }
    total = next;
}

std::uint64_t commands_preferred_upload_core_hash(const WorkingState& ws) noexcept {
    if (ws.uploadCoreHash != 0) {
        return ws.uploadCoreHash;
    }
    return ws.coreHash;
}

float commands_sanitize_filter_shift_step(float value) noexcept {
    if (std::isfinite(value)) {
        return value;
    }
    return 0.0f;
}

std::uint64_t commands_neutral_filter_hash_or_default(const Print::Runtime& prt) noexcept {
    if (prt.neutralFilterHash != 0) {
        return prt.neutralFilterHash;
    }
    return Print::kDefaultNeutralFilterHash;
}

const Scanner::ScannerStaticKey& commands_select_scanner_static_key(
    const WorkingState& ws,
    bool negativeMedium) noexcept {
    if (negativeMedium) {
        return ws.negativeStaticKey;
    }
    return ws.printStaticKey;
}

const Scanner::ScannerMediumRuntime& commands_select_scanner_medium_runtime(
    const WorkingState& ws,
    bool negativeMedium) noexcept {
    if (negativeMedium) {
        return ws.negativeMediumRuntime;
    }
    return ws.printMediumRuntime;
}

const JuicerCuda::Resources::DeviceSpectralLut& commands_select_scan_lut_slot(
    JuicerCuda::Resources& resources,
    bool negativeMedium) noexcept {
    if (negativeMedium) {
        return resources.scanNegativeLut;
    }
    return resources.scanPrintLut;
}

std::uint64_t commands_elapsed_ms_since(std::uint64_t nowMs, std::uint64_t earlierMs) noexcept {
    if (nowMs > earlierMs) {
        return nowMs - earlierMs;
    }
    return 0;
}

std::uint64_t commands_uncached_bytes(bool cached, std::uint64_t bytes) noexcept {
    if (cached) {
        return 0;
    }
    return bytes;
}

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
cudaStream_t commands_cuda_stream_or_null(void* cudaStreamOpaque) noexcept {
    if (!cudaStreamOpaque) {
        return nullptr;
    }
    return reinterpret_cast<cudaStream_t>(cudaStreamOpaque);
}
#endif

bool commands_allow_durable_admission(
    const CacheAdmissionDecision& decision,
    bool churnProbationAllowDurable) noexcept {
    if (decision.probationApplied) {
        return churnProbationAllowDurable;
    }
    return decision.allowDurableAdmission;
}

template <typename T>
void add_vector_upload_estimate_bytes(
    const std::vector<T>& values,
    std::uint64_t& total,
    bool& overflow) noexcept {
    if (values.empty() || overflow) {
        return;
    }
    std::uint64_t bytes = 0;
    if (!mul_u64_checked(
            static_cast<std::uint64_t>(values.size()),
            static_cast<std::uint64_t>(sizeof(T)),
            bytes)) {
        total = std::numeric_limits<std::uint64_t>::max();
        overflow = true;
        return;
    }
    add_estimate_bytes_u64(bytes, total, overflow);
}

void add_curve_upload_estimate_bytes(
    const Spectral::Curve& curve,
    std::uint64_t& total,
    bool& overflow) noexcept {
    const std::size_t n = std::min(curve.lambda_nm.size(), curve.linear.size());
    if (n == 0 || overflow) {
        return;
    }
    std::uint64_t bytes = 0;
    if (!mul_u64_checked(static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(2u * sizeof(float)), bytes)) {
        total = std::numeric_limits<std::uint64_t>::max();
        overflow = true;
        return;
    }
    add_estimate_bytes_u64(bytes, total, overflow);
}

void add_scan_medium_upload_estimate_bytes(
    const Scanner::ScannerMediumRuntime& medium,
    std::uint64_t& total,
    bool& overflow) noexcept {
    const Spectral::SpectralTables* t = medium.tables;
    if (!t) {
        return;
    }
    add_vector_upload_estimate_bytes(t->epsC, total, overflow);
    add_vector_upload_estimate_bytes(t->epsM, total, overflow);
    add_vector_upload_estimate_bytes(t->epsY, total, overflow);
    add_vector_upload_estimate_bytes(t->Ax, total, overflow);
    add_vector_upload_estimate_bytes(t->Ay, total, overflow);
    add_vector_upload_estimate_bytes(t->Az, total, overflow);
    if (t->hasBaseline) {
        add_vector_upload_estimate_bytes(t->baseMin, total, overflow);
    }
}

std::uint64_t estimate_upload_core_request_bytes(
    JuicerCuda::Resources& resources,
    const WorkingState& ws) noexcept {
    const std::uint64_t wsCoreHash = commands_preferred_upload_core_hash(ws);
    const std::uint64_t wsDirHash = ws.dirHash;
    if (wsCoreHash == 0 || wsDirHash == 0) {
        return kUploadReservationThresholdDefaultBytes;
    }

    bool coreUpToDate = false;
    bool dirUpToDate = false;
    bool needStbnUpload = false;
    bool needWangUpload = false;
    {
        std::lock_guard<std::mutex> lock(resources.m);
        coreUpToDate = (resources.uploadedCoreHash != 0) && (resources.uploadedCoreHash == wsCoreHash);
        dirUpToDate = (resources.uploadedDirHash != 0) && (resources.uploadedDirHash == wsDirHash);
        needStbnUpload = (resources.stbnData == nullptr);
        needWangUpload = (resources.wangTilesData == nullptr) || (resources.wangLutData == nullptr);
    }

    std::uint64_t estimateBytes = 0;
    bool overflow = false;
    if (needStbnUpload) {
        add_estimate_bytes_u64(kStbnUploadDefaultBytes, estimateBytes, overflow);
    }
    if (needWangUpload) {
        add_estimate_bytes_u64(kWangTilesUploadDefaultBytes, estimateBytes, overflow);
        add_estimate_bytes_u64(kWangLutUploadDefaultBytes, estimateBytes, overflow);
    }

    if (coreUpToDate && dirUpToDate) {
        return estimateBytes;
    }

    if (coreUpToDate && !dirUpToDate) {
        add_curve_upload_estimate_bytes(ws.dirDensB, estimateBytes, overflow);
        add_curve_upload_estimate_bytes(ws.dirDensG, estimateBytes, overflow);
        add_curve_upload_estimate_bytes(ws.dirDensR, estimateBytes, overflow);
        return estimateBytes;
    }

    add_curve_upload_estimate_bytes(ws.densB, estimateBytes, overflow);
    add_curve_upload_estimate_bytes(ws.densG, estimateBytes, overflow);
    add_curve_upload_estimate_bytes(ws.densR, estimateBytes, overflow);
    add_curve_upload_estimate_bytes(ws.dirDensB, estimateBytes, overflow);
    add_curve_upload_estimate_bytes(ws.dirDensG, estimateBytes, overflow);
    add_curve_upload_estimate_bytes(ws.dirDensR, estimateBytes, overflow);
    add_curve_upload_estimate_bytes(ws.sensB, estimateBytes, overflow);
    add_curve_upload_estimate_bytes(ws.sensG, estimateBytes, overflow);
    add_curve_upload_estimate_bytes(ws.sensR, estimateBytes, overflow);
    add_vector_upload_estimate_bytes(ws.tablesRef.Ax, estimateBytes, overflow);
    add_vector_upload_estimate_bytes(ws.tablesRef.Ay, estimateBytes, overflow);
    add_vector_upload_estimate_bytes(ws.tablesRef.Az, estimateBytes, overflow);
    add_vector_upload_estimate_bytes(ws.tablesRef.illum, estimateBytes, overflow);

    for (int layer = 0; layer < 3; ++layer) {
        for (int ch = 0; ch < 3; ++ch) {
            add_vector_upload_estimate_bytes(ws.densityCurvesLayers[layer][ch], estimateBytes, overflow);
        }
    }

    add_scan_medium_upload_estimate_bytes(ws.negativeMediumRuntime, estimateBytes, overflow);
    add_scan_medium_upload_estimate_bytes(ws.printMediumRuntime, estimateBytes, overflow);

    if (ws.printRT && Print::profile_is_valid(ws.printRT->profile)) {
        const Print::Profile& p = ws.printRT->profile;
        add_curve_upload_estimate_bytes(p.dcC, estimateBytes, overflow);
        add_curve_upload_estimate_bytes(p.dcM, estimateBytes, overflow);
        add_curve_upload_estimate_bytes(p.dcY, estimateBytes, overflow);
        add_vector_upload_estimate_bytes(p.sensC_log.linear, estimateBytes, overflow);
        add_vector_upload_estimate_bytes(p.sensM_log.linear, estimateBytes, overflow);
        add_vector_upload_estimate_bytes(p.sensY_log.linear, estimateBytes, overflow);
    }

    // Keep a deterministic floor so large rebuilds always enter upload reservation admission.
    if (estimateBytes < kUploadReservationThresholdDefaultBytes) {
        estimateBytes = kUploadReservationThresholdDefaultBytes;
    }
    return estimateBytes;
}

std::uint64_t estimate_scan_lut_upload_bytes(
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    bool negativeMedium) noexcept {
    const Scanner::ScannerStaticKey& staticKey =
        commands_select_scanner_static_key(ws, negativeMedium);
    const std::uint32_t res =
        ResourceManager::normalize_scan_lut_resolution(staticKey.lutResolution);
    std::uint64_t voxelCount = 0;
    if (!mul_u64_checked(static_cast<std::uint64_t>(res), static_cast<std::uint64_t>(res), voxelCount)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    if (!mul_u64_checked(voxelCount, static_cast<std::uint64_t>(res), voxelCount)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    std::uint64_t values = 0;
    if (!mul_u64_checked(voxelCount, 3ull, values)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    std::uint64_t bytes = 0;
    if (!mul_u64_checked(values, static_cast<std::uint64_t>(sizeof(double)), bytes)) {
        return std::numeric_limits<std::uint64_t>::max();
    }

    const Scanner::ScannerMediumRuntime& medium =
        commands_select_scanner_medium_runtime(ws, negativeMedium);
    if (!medium.tables || medium.tables->tablesHash == 0 || medium.range.digest == 0) {
        return 0;
    }
    const std::uint64_t expectedHash = ResourceManager::make_scan_lut_key_digest(
        static_cast<std::uint32_t>(medium.medium),
        medium.tables->tablesHash,
        medium.range.digest,
        res);
    if (expectedHash == 0) {
        return 0;
    }

    bool cached = false;
    {
        std::lock_guard<std::mutex> lock(resources.m);
        const JuicerCuda::Resources::DeviceSpectralLut& dst =
            commands_select_scan_lut_slot(resources, negativeMedium);
        cached = dst.log2XYZ && dst.res == res && dst.hash == expectedHash;
    }
    return commands_uncached_bytes(cached, bytes);
}

std::uint64_t estimate_print_illuminant_upload_bytes(
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    const Print::Runtime& prt,
    const Print::Params& params) noexcept {
    const int k = Spectral::gShape.K;
    if (k <= 0) {
        return 0;
    }
    const std::uint64_t wsCoreHash = commands_preferred_upload_core_hash(ws);
    if (wsCoreHash == 0) {
        return 0;
    }
    const float yKey = commands_sanitize_filter_shift_step(params.yFilter);
    const float mKey = commands_sanitize_filter_shift_step(params.mFilter);
    const float cKey = commands_sanitize_filter_shift_step(params.cFilter);
    const std::uint64_t neutralFilterHash = commands_neutral_filter_hash_or_default(prt);

    bool cached = false;
    {
        std::lock_guard<std::mutex> lock(resources.m);
        cached =
            resources.printIllumFiltered &&
            resources.printIllumK == k &&
            resources.printIllumShapeK == k &&
            resources.printIllumCoreHash == wsCoreHash &&
            resources.printIllumYShiftSteps == yKey &&
            resources.printIllumMShiftSteps == mKey &&
            resources.printIllumCShiftSteps == cKey &&
            resources.printIllumNeutralFilterHash == neutralFilterHash;
    }
    if (cached) {
        return 0;
    }
    std::uint64_t bytes = 0;
    if (!mul_u64_checked(
            static_cast<std::uint64_t>(k),
            static_cast<std::uint64_t>(sizeof(float)),
            bytes)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return bytes;
}

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
struct BaseGraphKey {
    int width = 0;
    int height = 0;
    int nComponents = 0;
    int renderMode = 0;
};

struct BaseGraphEntry {
    BaseGraphKey key{};
    void* graphOpaque = nullptr;
    void* execOpaque = nullptr;
    void* kernelNodeOpaque = nullptr;
    void* kernelFuncOpaque = nullptr;
    unsigned int gridX = 0;
    unsigned int gridY = 0;
    unsigned int gridZ = 0;
    unsigned int blockX = 0;
    unsigned int blockY = 0;
    unsigned int blockZ = 0;
    unsigned int sharedMemBytes = 0;
    std::uint64_t lastUseTick = 0;
    std::uint64_t lastUseMs = 0;
};

struct GraphLargeEntryReadmitState {
    std::uint64_t lastEvictedMs = 0;
    std::uint32_t ghostHits = 0;
};

struct BaseGraphBucketState {
    std::mutex mutex;
    std::uint64_t contextEpoch = 0;
    std::uint64_t useTick = 0;
    std::vector<BaseGraphEntry> entries;
    std::unordered_map<std::uint64_t, std::uint32_t> probationHitsByDigest;
    std::unordered_map<std::uint64_t, GraphLargeEntryReadmitState> largeEntryReadmitByDigest;
    std::uint64_t largeEntryResidentBytes = 0;
    std::uint32_t largeEntryResidentEntries = 0;
};

struct BaseGraphCacheState {
    std::mutex mutex;
    std::unordered_map<DeviceContextKey, std::shared_ptr<BaseGraphBucketState>, DeviceContextKeyHash> byContext;
};

BaseGraphCacheState& base_graph_cache_state() noexcept {
    static BaseGraphCacheState state{};
    return state;
}

using BasePipelineLaunchFn = cudaError_t(*)(const JuicerCuda::PipelineRunParams*, void*);

BasePipelineLaunchFn base_pipeline_launch_fn_for_mode(int renderModeKey) noexcept {
    switch (renderModeKey) {
    case 0:
        return juicer_cuda_negative_pipeline;
    case 1:
        return juicer_cuda_print_pipeline;
    default:
        return nullptr;
    }
}

bool base_graph_key_equal(const BaseGraphKey& a, const BaseGraphKey& b) noexcept {
    return a.width == b.width &&
        a.height == b.height &&
        a.nComponents == b.nComponents &&
        a.renderMode == b.renderMode;
}

std::uint64_t base_graph_key_digest(const BaseGraphKey& key) noexcept {
    std::uint64_t digest = 1469598103934665603ull;
    auto fold = [&digest](std::uint64_t value) noexcept {
        digest ^= value + 0x9e3779b97f4a7c15ull + (digest << 6u) + (digest >> 2u);
    };
    fold(static_cast<std::uint64_t>(std::max(0, key.width)));
    fold(static_cast<std::uint64_t>(std::max(0, key.height)));
    fold(static_cast<std::uint64_t>(std::max(0, key.nComponents)));
    fold(static_cast<std::uint64_t>(std::max(0, key.renderMode)));
    return digest;
}

inline bool base_graph_entry_live(const BaseGraphEntry& entry) noexcept {
    return entry.execOpaque != nullptr &&
        entry.graphOpaque != nullptr &&
        entry.kernelNodeOpaque != nullptr;
}

void add_graph_large_entry_resident_bytes_global(std::uint64_t bytes) noexcept {
    if (bytes == 0) {
        return;
    }
    telemetry_counter_add(global_state().graphLargeEntryResidentBytes, bytes);
}

void sub_graph_large_entry_resident_bytes_global(std::uint64_t bytes) noexcept {
    telemetry_counter_subtract_saturating(global_state().graphLargeEntryResidentBytes, bytes);
}

void publish_graph_large_entry_resident_snapshot_locked(
    BaseGraphBucketState& bucket,
    std::uint64_t residentBytes,
    std::uint32_t residentEntries) noexcept {
    if (residentBytes > bucket.largeEntryResidentBytes) {
        add_graph_large_entry_resident_bytes_global(residentBytes - bucket.largeEntryResidentBytes);
    }
    else if (bucket.largeEntryResidentBytes > residentBytes) {
        sub_graph_large_entry_resident_bytes_global(bucket.largeEntryResidentBytes - residentBytes);
    }
    bucket.largeEntryResidentBytes = residentBytes;
    bucket.largeEntryResidentEntries = residentEntries;
}

std::uint64_t estimate_base_graph_request_bytes(const BaseGraphKey& key) noexcept {
    const std::uint64_t w = static_cast<std::uint64_t>(std::max(0, key.width));
    const std::uint64_t h = static_cast<std::uint64_t>(std::max(0, key.height));
    const std::uint64_t n = static_cast<std::uint64_t>(std::max(0, key.nComponents));
    std::uint64_t pixels = 0;
    if (!mul_u64_checked(w, h, pixels)) {
        return std::numeric_limits<std::uint64_t>::max();
    }

    constexpr std::uint64_t kBaseGraphMetadataBytes = 2ull * 1024ull * 1024ull;
    constexpr std::uint64_t kPerMegapixelMetadataBytes = 64ull * 1024ull;
    constexpr std::uint64_t kPerComponentMetadataBytes = 128ull * 1024ull;
    constexpr std::uint64_t kMegapixelDivisor = 1024ull * 1024ull;

    std::uint64_t bytes = kBaseGraphMetadataBytes;
    std::uint64_t mpBytes = 0;
    if (!mul_u64_checked(pixels / kMegapixelDivisor, kPerMegapixelMetadataBytes, mpBytes)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    std::uint64_t nextBytes = 0;
    if (!add_u64_checked(bytes, mpBytes, nextBytes)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    bytes = nextBytes;

    std::uint64_t componentBytes = 0;
    if (!mul_u64_checked(n, kPerComponentMetadataBytes, componentBytes)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    if (!add_u64_checked(bytes, componentBytes, nextBytes)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return nextBytes;
}

bool graph_entry_in_keep_hot_window(
    const BaseGraphEntry& entry,
    std::uint32_t keepHotMs,
    std::uint64_t nowMs) noexcept {
    if (keepHotMs == 0 || entry.lastUseMs == 0) {
        return false;
    }
    const std::uint64_t ageMs = commands_elapsed_ms_since(nowMs, entry.lastUseMs);
    return ageMs < static_cast<std::uint64_t>(keepHotMs);
}

void note_large_entry_evicted_for_readmit_locked(
    BaseGraphBucketState& bucket,
    const BaseGraphEntry& entry,
    std::uint64_t thresholdBytes,
    std::uint64_t nowMs) noexcept {
    if (thresholdBytes == 0 || !base_graph_entry_live(entry)) {
        return;
    }
    const std::uint64_t entryBytes = estimate_base_graph_request_bytes(entry.key);
    if (entryBytes < thresholdBytes) {
        return;
    }
    const std::uint64_t digest = base_graph_key_digest(entry.key);
    GraphLargeEntryReadmitState& state = bucket.largeEntryReadmitByDigest[digest];
    state.lastEvictedMs = nowMs;
    state.ghostHits = 0;
}

void destroy_base_graph_entry(BaseGraphEntry& entry) noexcept {
    if (entry.execOpaque) {
        cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(entry.execOpaque));
        entry.execOpaque = nullptr;
    }
    if (entry.graphOpaque) {
        cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(entry.graphOpaque));
        entry.graphOpaque = nullptr;
    }
    entry.kernelNodeOpaque = nullptr;
    entry.kernelFuncOpaque = nullptr;
    entry.gridX = 0;
    entry.gridY = 0;
    entry.gridZ = 0;
    entry.blockX = 0;
    entry.blockY = 0;
    entry.blockZ = 0;
    entry.sharedMemBytes = 0;
    entry.lastUseTick = 0;
    entry.lastUseMs = 0;
}

std::shared_ptr<BaseGraphBucketState> get_or_create_base_graph_bucket(
    const DeviceContextKey& key) noexcept {
    BaseGraphCacheState& state = base_graph_cache_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.byContext.find(key);
    if (it != state.byContext.end() && it->second) {
        return it->second;
    }
    auto bucket = std::make_shared<BaseGraphBucketState>();
    state.byContext[key] = bucket;
    return bucket;
}

void clear_base_graph_bucket(BaseGraphBucketState& bucket) noexcept {
    publish_graph_large_entry_resident_snapshot_locked(bucket, 0, 0);
    for (auto& entry : bucket.entries) {
        destroy_base_graph_entry(entry);
    }
    bucket.entries.clear();
    bucket.probationHitsByDigest.clear();
    bucket.largeEntryReadmitByDigest.clear();
    bucket.useTick = 0;
}

std::uint64_t estimate_base_graph_bucket_active_bytes_locked(
    const BaseGraphBucketState& bucket) noexcept {
    std::uint64_t totalBytes = 0;
    for (const BaseGraphEntry& entry : bucket.entries) {
        if (!base_graph_entry_live(entry)) {
            continue;
        }
        const std::uint64_t entryBytes = estimate_base_graph_request_bytes(entry.key);
        std::uint64_t nextBytes = 0;
        if (!add_u64_checked(totalBytes, entryBytes, nextBytes)) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        totalBytes = nextBytes;
    }
    return totalBytes;
}

std::uint64_t estimate_base_graph_bucket_large_entry_bytes_locked(
    const BaseGraphBucketState& bucket,
    std::uint64_t thresholdBytes,
    std::uint32_t& outEntries) noexcept {
    outEntries = 0;
    if (thresholdBytes == 0) {
        return 0;
    }

    std::uint64_t totalBytes = 0;
    for (const BaseGraphEntry& entry : bucket.entries) {
        if (!base_graph_entry_live(entry)) {
            continue;
        }
        const std::uint64_t entryBytes = estimate_base_graph_request_bytes(entry.key);
        if (entryBytes < thresholdBytes) {
            continue;
        }
        std::uint64_t nextBytes = 0;
        if (!add_u64_checked(totalBytes, entryBytes, nextBytes)) {
            totalBytes = std::numeric_limits<std::uint64_t>::max();
        }
        else {
            totalBytes = nextBytes;
        }
        if (outEntries < std::numeric_limits<std::uint32_t>::max()) {
            ++outEntries;
        }
    }
    return totalBytes;
}

std::uint64_t trim_graph_large_entry_decay_and_caps_locked(
    BaseGraphBucketState& bucket,
    std::uint64_t thresholdBytes,
    std::uint64_t capBytes,
    std::uint32_t capEntries,
    std::uint32_t keepHotMs,
    std::uint64_t nowMs,
    std::uint64_t& outDecayEvictedEntries,
    std::uint64_t& outCapTrimEvictedEntries,
    std::uint64_t& outKeepHotBypassEvents,
    std::uint64_t& outKeepHotForcedEvictEvents,
    bool& outCapHit,
    std::uint32_t& outResidentEntries) noexcept {
    outDecayEvictedEntries = 0;
    outCapTrimEvictedEntries = 0;
    outKeepHotBypassEvents = 0;
    outKeepHotForcedEvictEvents = 0;
    outCapHit = false;
    outResidentEntries = 0;

    if (thresholdBytes == 0) {
        publish_graph_large_entry_resident_snapshot_locked(bucket, 0, 0);
        return 0;
    }

    std::size_t index = 0;
    while (index < bucket.entries.size()) {
        BaseGraphEntry& entry = bucket.entries[index];
        if (!base_graph_entry_live(entry)) {
            ++index;
            continue;
        }
        const std::uint64_t entryBytes = estimate_base_graph_request_bytes(entry.key);
        if (entryBytes < thresholdBytes) {
            ++index;
            continue;
        }
        const std::uint64_t ageMs = commands_elapsed_ms_since(nowMs, entry.lastUseMs);
        if (ageMs < kGraphLargeEntryDecayMs) {
            ++index;
            continue;
        }
        note_large_entry_evicted_for_readmit_locked(
            bucket,
            entry,
            thresholdBytes,
            nowMs);
        destroy_base_graph_entry(entry);
        bucket.entries.erase(bucket.entries.begin() + static_cast<std::ptrdiff_t>(index));
        ++outDecayEvictedEntries;
    }

    auto recompute_large_snapshot = [&](std::size_t* outOldestIndex,
                                        std::size_t* outOldestNonHotIndex,
                                        bool* outHasKeepHotCandidates) noexcept -> std::uint64_t {
        if (outOldestIndex) {
            *outOldestIndex = std::numeric_limits<std::size_t>::max();
        }
        if (outOldestNonHotIndex) {
            *outOldestNonHotIndex = std::numeric_limits<std::size_t>::max();
        }
        if (outHasKeepHotCandidates) {
            *outHasKeepHotCandidates = false;
        }
        outResidentEntries = 0;
        std::uint64_t residentBytes = 0;
        std::uint64_t oldestUseMs = std::numeric_limits<std::uint64_t>::max();
        std::uint64_t oldestUseTick = std::numeric_limits<std::uint64_t>::max();
        std::uint64_t oldestNonHotUseMs = std::numeric_limits<std::uint64_t>::max();
        std::uint64_t oldestNonHotUseTick = std::numeric_limits<std::uint64_t>::max();

        for (std::size_t i = 0; i < bucket.entries.size(); ++i) {
            const BaseGraphEntry& entry = bucket.entries[i];
            if (!base_graph_entry_live(entry)) {
                continue;
            }
            const std::uint64_t entryBytes = estimate_base_graph_request_bytes(entry.key);
            if (entryBytes < thresholdBytes) {
                continue;
            }

            std::uint64_t nextBytes = 0;
            if (!add_u64_checked(residentBytes, entryBytes, nextBytes)) {
                residentBytes = std::numeric_limits<std::uint64_t>::max();
            }
            else {
                residentBytes = nextBytes;
            }

            if (outResidentEntries < std::numeric_limits<std::uint32_t>::max()) {
                ++outResidentEntries;
            }

            const bool inKeepHotWindow = graph_entry_in_keep_hot_window(entry, keepHotMs, nowMs);
            if (inKeepHotWindow && outHasKeepHotCandidates) {
                *outHasKeepHotCandidates = true;
            }

            const bool older = (entry.lastUseMs < oldestUseMs) ||
                ((entry.lastUseMs == oldestUseMs) && (entry.lastUseTick < oldestUseTick));
            if (older) {
                oldestUseMs = entry.lastUseMs;
                oldestUseTick = entry.lastUseTick;
                if (outOldestIndex) {
                    *outOldestIndex = i;
                }
            }

            if (!inKeepHotWindow) {
                const bool olderNonHot = (entry.lastUseMs < oldestNonHotUseMs) ||
                    ((entry.lastUseMs == oldestNonHotUseMs) && (entry.lastUseTick < oldestNonHotUseTick));
                if (olderNonHot) {
                    oldestNonHotUseMs = entry.lastUseMs;
                    oldestNonHotUseTick = entry.lastUseTick;
                    if (outOldestNonHotIndex) {
                        *outOldestNonHotIndex = i;
                    }
                }
            }
        }
        return residentBytes;
    };

    std::size_t oldestIndex = std::numeric_limits<std::size_t>::max();
    std::size_t oldestNonHotIndex = std::numeric_limits<std::size_t>::max();
    bool hasKeepHotCandidates = false;
    std::uint64_t residentBytes = recompute_large_snapshot(
        &oldestIndex,
        &oldestNonHotIndex,
        &hasKeepHotCandidates);
    const std::uint32_t allowedEntries = std::max<std::uint32_t>(1u, capEntries);
    while (oldestIndex != std::numeric_limits<std::size_t>::max() &&
           (outResidentEntries > allowedEntries || residentBytes > capBytes)) {
        outCapHit = true;
        std::size_t victimIndex = oldestIndex;
        if (keepHotMs > 0 && hasKeepHotCandidates) {
            if (oldestNonHotIndex != std::numeric_limits<std::size_t>::max()) {
                victimIndex = oldestNonHotIndex;
                if (victimIndex != oldestIndex) {
                    ++outKeepHotBypassEvents;
                }
            }
            else {
                ++outKeepHotForcedEvictEvents;
            }
        }
        BaseGraphEntry& victim = bucket.entries[victimIndex];
        note_large_entry_evicted_for_readmit_locked(
            bucket,
            victim,
            thresholdBytes,
            nowMs);
        destroy_base_graph_entry(victim);
        bucket.entries.erase(bucket.entries.begin() + static_cast<std::ptrdiff_t>(victimIndex));
        ++outCapTrimEvictedEntries;
        residentBytes = recompute_large_snapshot(
            &oldestIndex,
            &oldestNonHotIndex,
            &hasKeepHotCandidates);
    }

    publish_graph_large_entry_resident_snapshot_locked(bucket, residentBytes, outResidentEntries);
    return residentBytes;
}

std::uint64_t estimate_graph_cache_active_bytes_for_context(const DeviceContextKey& key) noexcept {
    BaseGraphCacheState& state = base_graph_cache_state();
    std::shared_ptr<BaseGraphBucketState> bucket;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        auto it = state.byContext.find(key);
        if (it == state.byContext.end()) {
            return 0;
        }
        bucket = it->second;
    }
    if (!bucket) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(bucket->mutex);
    return estimate_base_graph_bucket_active_bytes_locked(*bucket);
}

std::uint64_t evict_noncritical_graph_entries_for_context(const DeviceContextKey& key) noexcept {
    BaseGraphCacheState& state = base_graph_cache_state();
    std::shared_ptr<BaseGraphBucketState> bucket;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        auto it = state.byContext.find(key);
        if (it == state.byContext.end()) {
            return 0;
        }
        bucket = it->second;
    }
    if (!bucket) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(bucket->mutex);
    const std::uint64_t evictedEntries = static_cast<std::uint64_t>(bucket->entries.size());
    if (evictedEntries == 0) {
        return 0;
    }
    clear_base_graph_bucket(*bucket);
    return evictedEntries;
}

void retire_base_graph_cache_for_context(const DeviceContextKey& key) noexcept {
    BaseGraphCacheState& state = base_graph_cache_state();
    std::shared_ptr<BaseGraphBucketState> bucket;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        auto it = state.byContext.find(key);
        if (it == state.byContext.end()) {
            return;
        }
        bucket = it->second;
        state.byContext.erase(it);
    }
    if (bucket) {
        std::lock_guard<std::mutex> lock(bucket->mutex);
        clear_base_graph_bucket(*bucket);
    }
}

BaseGraphEntry* find_base_graph_entry(
    BaseGraphBucketState& bucket,
    const BaseGraphKey& key) noexcept {
    for (auto& entry : bucket.entries) {
        if (base_graph_key_equal(entry.key, key) && base_graph_entry_live(entry)) {
            return &entry;
        }
    }
    return nullptr;
}

BaseGraphEntry* build_base_graph_entry(
    BaseGraphBucketState& bucket,
    const BaseGraphKey& key,
    BasePipelineLaunchFn launchFn,
    JuicerCuda::PipelineRunParams& run,
    cudaStream_t stream,
    std::uint64_t largeEntryThresholdBytes,
    std::uint32_t keepHotMs,
    std::uint64_t& outKeepHotBypassEvents,
    std::uint64_t& outKeepHotForcedEvictEvents) noexcept {
    outKeepHotBypassEvents = 0;
    outKeepHotForcedEvictEvents = 0;
    if (!launchFn) {
        return nullptr;
    }

    constexpr std::size_t kBaseGraphCap = 4;
    if (bucket.entries.size() >= kBaseGraphCap && !bucket.entries.empty()) {
        const std::uint64_t nowMs = monotonic_time_ms();
        std::size_t oldestAny = 0;
        std::uint64_t bestTick = bucket.entries[0].lastUseTick;
        for (std::size_t i = 1; i < bucket.entries.size(); ++i) {
            if (bucket.entries[i].lastUseTick < bestTick) {
                bestTick = bucket.entries[i].lastUseTick;
                oldestAny = i;
            }
        }

        std::size_t victim = oldestAny;
        if (keepHotMs > 0) {
            std::size_t oldestNonHot = std::numeric_limits<std::size_t>::max();
            std::uint64_t oldestNonHotTick = std::numeric_limits<std::uint64_t>::max();
            bool hasKeepHotCandidates = false;
            for (std::size_t i = 0; i < bucket.entries.size(); ++i) {
                const BaseGraphEntry& entry = bucket.entries[i];
                const bool inKeepHotWindow = graph_entry_in_keep_hot_window(entry, keepHotMs, nowMs);
                if (inKeepHotWindow) {
                    hasKeepHotCandidates = true;
                    continue;
                }
                if (oldestNonHot == std::numeric_limits<std::size_t>::max() ||
                    entry.lastUseTick < oldestNonHotTick) {
                    oldestNonHotTick = entry.lastUseTick;
                    oldestNonHot = i;
                }
            }

            if (hasKeepHotCandidates) {
                if (oldestNonHot != std::numeric_limits<std::size_t>::max()) {
                    victim = oldestNonHot;
                    if (victim != oldestAny) {
                        ++outKeepHotBypassEvents;
                    }
                }
                else {
                    ++outKeepHotForcedEvictEvents;
                }
            }
        }

        note_large_entry_evicted_for_readmit_locked(
            bucket,
            bucket.entries[victim],
            largeEntryThresholdBytes,
            nowMs);
        destroy_base_graph_entry(bucket.entries[victim]);
        bucket.entries.erase(bucket.entries.begin() + static_cast<std::ptrdiff_t>(victim));
    }

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaGraphNode_t kernelNode = nullptr;

    cudaError_t capErr = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    if (capErr != cudaSuccess) {
        return nullptr;
    }

    cudaError_t launchErr = launchFn(&run, reinterpret_cast<void*>(stream));
    if (launchErr != cudaSuccess) {
        cudaGraph_t abortGraph = nullptr;
        cudaStreamEndCapture(stream, &abortGraph);
        if (abortGraph) {
            cudaGraphDestroy(abortGraph);
        }
        return nullptr;
    }

    capErr = cudaStreamEndCapture(stream, &graph);
    if (capErr != cudaSuccess || !graph) {
        if (graph) {
            cudaGraphDestroy(graph);
        }
        return nullptr;
    }

    cudaError_t instErr = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    if (instErr != cudaSuccess || !exec) {
        cudaGraphDestroy(graph);
        return nullptr;
    }

    std::size_t nodeCount = 0;
    cudaError_t nodeErr = cudaGraphGetNodes(graph, nullptr, &nodeCount);
    if (nodeErr == cudaSuccess && nodeCount > 0) {
        std::vector<cudaGraphNode_t> nodes(nodeCount);
        nodeErr = cudaGraphGetNodes(graph, nodes.data(), &nodeCount);
        if (nodeErr == cudaSuccess) {
            for (cudaGraphNode_t node : nodes) {
                cudaGraphNodeType nodeType = cudaGraphNodeTypeEmpty;
                if (cudaGraphNodeGetType(node, &nodeType) == cudaSuccess && nodeType == cudaGraphNodeTypeKernel) {
                    kernelNode = node;
                    break;
                }
            }
        }
    }

    if (!kernelNode) {
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        return nullptr;
    }

    cudaKernelNodeParams baseParams{};
    if (cudaGraphKernelNodeGetParams(kernelNode, &baseParams) != cudaSuccess || !baseParams.func) {
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        return nullptr;
    }

    BaseGraphEntry entry{};
    entry.key = key;
    entry.graphOpaque = reinterpret_cast<void*>(graph);
    entry.execOpaque = reinterpret_cast<void*>(exec);
    entry.kernelNodeOpaque = reinterpret_cast<void*>(kernelNode);
    entry.kernelFuncOpaque = baseParams.func;
    entry.gridX = baseParams.gridDim.x;
    entry.gridY = baseParams.gridDim.y;
    entry.gridZ = baseParams.gridDim.z;
    entry.blockX = baseParams.blockDim.x;
    entry.blockY = baseParams.blockDim.y;
    entry.blockZ = baseParams.blockDim.z;
    entry.sharedMemBytes = baseParams.sharedMemBytes;
    entry.lastUseTick = bucket.useTick;
    entry.lastUseMs = monotonic_time_ms();
    bucket.entries.push_back(entry);
    return &bucket.entries.back();
}
#else
std::uint64_t estimate_graph_cache_active_bytes_for_context(const DeviceContextKey&) noexcept {
    return 0;
}

std::uint64_t evict_noncritical_graph_entries_for_context(const DeviceContextKey&) noexcept {
    return 0;
}
#endif


bool command_freeze_drain_bump_resume(
    const DeviceContextKey& key,
    const char* reason,
    std::string& outError) {
    outError.clear();
    MetadataMutationGuard mutationGuard("command_freeze_drain_bump_resume", &key);
    if (!mutationGuard.ok()) {
        outError = "metadata mutation guard rejected command_freeze_drain_bump_resume";
        return false;
    }
    if (!registry_freeze_drain_bump_resume(key, reason)) {
        outError = "freeze-drain-bump-resume barrier rejected";
        return false;
    }
    return true;
}

namespace {
bool command_retire_context_with_reason(
    const DeviceContextKey& key,
    RegistryRetireReason reason,
    const char* commandName,
    std::string& outError) {
    outError.clear();
    const char* stageName = trace_or(commandName, "command_retire_context");
    MetadataMutationGuard mutationGuard(stageName, &key);
    if (!mutationGuard.ok()) {
        outError = std::string("metadata mutation guard rejected ") + stageName;
        return false;
    }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    JuicerCuda::purge_shared_gaussian_kernels_for_context(
        key.deviceId,
        key.contextOpaque);
    JuicerCuda::purge_pinned_upload_staging_for_context(
        key.deviceId,
        key.contextOpaque);
    retire_base_graph_cache_for_context(key);
#endif
    tier_circuit_retire_context(key);
    pressure_policy_retire_context(key);
    admission_churn_retire_context(key);
    optional_heuristic_trace_retire_context(key);
    allocator_backend_retire_context(key);
    state_clear_latest_snapshot_for_context(key);

    RegistryHandle handle{};
    if (!registry_get(key, handle) || handle.value == 0) {
        return true;
    }

    registry_retire(handle, reason, &key);
    return true;
}
} // namespace

bool command_retire_context_reset(
    const DeviceContextKey& key,
    std::string& outError) {
    return command_retire_context_with_reason(
        key,
        RegistryRetireReason::ContextReset,
        "command_retire_context_reset",
        outError);
}

bool command_retire_context_idle(
    const DeviceContextKey& key,
    std::string& outError) {
    return command_retire_context_with_reason(
        key,
        RegistryRetireReason::Idle,
        "command_retire_context_idle",
        outError);
}

namespace {
const char* scan_lut_medium_name(bool negativeMedium) noexcept {
    return bool_reason(negativeMedium, "negative", "print");
}

bool compute_expected_scan_lut_hash(
    const WorkingState& ws,
    bool negativeMedium,
    std::uint64_t& outExpectedHash) noexcept {
    outExpectedHash = 0;
    const Scanner::ScannerMediumRuntime* medium = &ws.printMediumRuntime;
    const Scanner::ScannerStaticKey* staticKey = &ws.printStaticKey;
    Scanner::ScannerMedium expectedMedium = Scanner::ScannerMedium::Print;
    if (negativeMedium) {
        medium = &ws.negativeMediumRuntime;
        staticKey = &ws.negativeStaticKey;
        expectedMedium = Scanner::ScannerMedium::Negative;
    }

    if (!medium->tables || medium->tables->K <= 0) {
        return false;
    }
    if (medium->medium != expectedMedium || staticKey->medium != expectedMedium) {
        return false;
    }
    if (medium->tables->tablesHash == 0 || medium->range.digest == 0) {
        return false;
    }
    if (staticKey->tablesHash != medium->tables->tablesHash ||
        staticKey->densityRangeHash != medium->range.digest) {
        return false;
    }
    const std::uint32_t res =
        ResourceManager::normalize_scan_lut_resolution(staticKey->lutResolution);
    outExpectedHash = ResourceManager::make_scan_lut_key_digest(
        static_cast<std::uint32_t>(medium->medium),
        medium->tables->tablesHash,
        medium->range.digest,
        res);
    return outExpectedHash != 0;
}

struct PrivateLutFallbackDecision {
    bool allowed = false;
    bool alreadyActive = false;
    std::uint32_t activeCount = 0;
    std::uint32_t perMediumCap = 0;
    std::uint32_t perInstanceCap = 0;
    const char* reason = "unspecified";
};

PrivateLutFallbackDecision evaluate_private_lut_fallback(
    JuicerCuda::Resources& resources,
    bool negativeMedium,
    std::uint64_t expectedHash,
    const ResourceManagerConfigEffective& cfg) {
    PrivateLutFallbackDecision decision{};
    decision.perMediumCap = cfg.privateLutFallbackPerMediumCap;
    decision.perInstanceCap = cfg.privateLutFallbackPerInstanceCap;

    if (expectedHash == 0) {
        decision.reason = "invalid_expected_hash";
        return decision;
    }

    std::lock_guard<std::mutex> lock(resources.m);
    auto clear_stale = [](bool& active, std::uint64_t& hash, const JuicerCuda::Resources::DeviceSpectralLut& lut) {
        if (!lut.log2XYZ) {
            active = false;
            hash = 0;
        }
    };
    clear_stale(
        resources.privateLutFallbackNegativeActive,
        resources.privateLutFallbackNegativeHash,
        resources.scanNegativeLut);
    clear_stale(
        resources.privateLutFallbackPrintActive,
        resources.privateLutFallbackPrintHash,
        resources.scanPrintLut);

    decision.activeCount =
        bool_u32(resources.privateLutFallbackNegativeActive) +
        bool_u32(resources.privateLutFallbackPrintActive);

    bool* slotActive = &resources.privateLutFallbackPrintActive;
    std::uint64_t* slotHash = &resources.privateLutFallbackPrintHash;
    const JuicerCuda::Resources::DeviceSpectralLut* slotLut = &resources.scanPrintLut;
    if (negativeMedium) {
        slotActive = &resources.privateLutFallbackNegativeActive;
        slotHash = &resources.privateLutFallbackNegativeHash;
        slotLut = &resources.scanNegativeLut;
    }

    if (*slotActive && slotLut->log2XYZ && *slotHash == expectedHash) {
        decision.allowed = true;
        decision.alreadyActive = true;
        decision.reason = "already_active";
        return decision;
    }

    if (decision.perMediumCap == 0 || decision.perInstanceCap == 0) {
        decision.reason = "disabled";
        return decision;
    }
    if (decision.perMediumCap < 1u) {
        decision.reason = "per_medium_cap_zero";
        return decision;
    }
    if (!*slotActive && decision.activeCount >= decision.perInstanceCap) {
        decision.reason = "per_instance_cap_reached";
        return decision;
    }

    decision.allowed = true;
    if (*slotActive) {
        decision.reason = "slot_reuse";
    }
    else {
        decision.reason = "admit_new";
    }
    return decision;
}

void mark_private_lut_fallback_active(
    JuicerCuda::Resources& resources,
    bool negativeMedium,
    std::uint64_t expectedHash) {
    if (expectedHash == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(resources.m);
    if (negativeMedium) {
        resources.privateLutFallbackNegativeActive = true;
        resources.privateLutFallbackNegativeHash = expectedHash;
    }
    else {
        resources.privateLutFallbackPrintActive = true;
        resources.privateLutFallbackPrintHash = expectedHash;
    }
}

void trace_private_lut_fallback(
    const char* stage,
    const char* eventName,
    bool negativeMedium,
    const PrivateLutFallbackDecision& decision,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }
    std::string msg =
        std::string("stage=") + trace_or_unknown(stage) +
        " event=" + trace_or_unknown(eventName) +
        " medium=" + scan_lut_medium_name(negativeMedium) +
        " allowed=" + std::to_string(bool_u32(decision.allowed)) +
        " already_active=" + std::to_string(bool_u32(decision.alreadyActive)) +
        " active_count=" + std::to_string(decision.activeCount) +
        " per_medium_cap=" + std::to_string(decision.perMediumCap) +
        " per_instance_cap=" + std::to_string(decision.perInstanceCap) +
        " reason=" + trace_or(reason, decision.reason) +
        " reason_class=" + trace_reason_class_or_invalid(trace_or(reason, decision.reason));
    JTRACE("MSLUT", msg);
}

bool command_ensure_scan_lut_internal(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    bool negativeMedium,
    bool criticalRequest,
    const char* commandName,
    void* cudaStreamOpaque,
    std::string& outError);
}

bool command_ensure_uploaded(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    bool allowLutPrewarm,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_uploaded")) {
        return false;
    }
    const std::uint64_t wsCoreHash = commands_preferred_upload_core_hash(ws);
    bool coreUploadStale = true;
    if (wsCoreHash != 0) {
        std::lock_guard<std::mutex> lock(resources.m);
        coreUploadStale = resources.uploadedCoreHash != wsCoreHash;
    }
    const std::uint64_t uploadRequestBytes =
        estimate_upload_core_request_bytes(resources, ws);
    if (uploadRequestBytes == std::numeric_limits<std::uint64_t>::max()) {
        outError = "upload reservation request byte estimation overflow (core)";
        return false;
    }
    const bool captureMemorySnapshots = should_collect_manager_memory_snapshots(manager_effective_config());
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    bool requestPreReclaim = false;
    if (!enforce_pressure_gate(
            transaction,
            resources,
            "command_ensure_uploaded",
            PressureLane::Upload,
            saturating_u64_to_size_t(uploadRequestBytes),
            true,
            requestPreReclaim,
            outError)) {
        return false;
    }
    if (requestPreReclaim) {
        std::string reclaimError;
        if (!run_reap_pass_for_pressure(
                transaction,
                resources,
                "command_ensure_uploaded",
                "pressure_pre_upload",
                reclaimError)) {
            outError = commands_error_or_message(reclaimError, "pressure pre-upload reclaim failed");
            return false;
        }
    }
    bool ok = false;
    {
        UploadReservationClaim uploadClaim{};
        if (!acquire_upload_reservation_with_wait(
                transaction,
                "command_ensure_uploaded",
                uploadRequestBytes,
                true,
                uploadClaim,
                outError)) {
            return false;
        }
        UploadReservationGuard uploadGuard(std::move(uploadClaim));
        TierCircuitAttempt circuitAttempt{};
        std::string circuitError;
        if (!tier_circuit_begin_attempt(
                transaction,
                "command_ensure_uploaded",
                ResourceTier::Immutable,
                tier_circuit_blocks_admission(ResourceTier::Immutable),
                circuitAttempt,
                circuitError)) {
            outError = circuitError;
            return false;
        }
        ok = JuicerCuda::ensure_uploaded(resources, ws, cudaStreamOpaque, outError);
        complete_tier_circuit_attempt(
            transaction,
            "command_ensure_uploaded",
            circuitAttempt,
            ok,
            "ensure_success",
            outError);
    }

    if (ok && allowLutPrewarm && coreUploadStale) {
        std::string prewarmError;
        (void)command_ensure_scan_lut_internal(
            transaction,
            resources,
            ws,
            true,
            false,
            "command_prewarm_scan_lut_negative",
            cudaStreamOpaque,
            prewarmError);
        if (JTRACE_ENABLED(3)) {
            std::string msg = std::string("stage=prewarm medium=negative result=")
                + prewarm_result_reason(prewarmError);
            if (!prewarmError.empty()) {
                msg += " detail=" + prewarmError;
            }
            JTRACE_VERBOSE("MSLUT", msg);
        }

        prewarmError.clear();
        (void)command_ensure_scan_lut_internal(
            transaction,
            resources,
            ws,
            false,
            false,
            "command_prewarm_scan_lut_print",
            cudaStreamOpaque,
            prewarmError);
        if (JTRACE_ENABLED(3)) {
            std::string msg = std::string("stage=prewarm medium=print result=")
                + prewarm_result_reason(prewarmError);
            if (!prewarmError.empty()) {
                msg += " detail=" + prewarmError;
            }
            JTRACE_VERBOSE("MSLUT", msg);
        }
    }

    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    return ok;
}

namespace {
bool command_ensure_scan_lut_internal(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    bool negativeMedium,
    bool criticalRequest,
    const char* commandName,
    void* cudaStreamOpaque,
    std::string& outError) {
    const char* stageName = trace_or_non_empty(commandName, "command_ensure_scan_lut");
    if (!ensure_active_for_command(transaction, outError, stageName)) {
        return false;
    }
    const std::uint64_t uploadRequestBytes =
        estimate_scan_lut_upload_bytes(resources, ws, negativeMedium);
    if (uploadRequestBytes == std::numeric_limits<std::uint64_t>::max()) {
        outError = "upload reservation request byte estimation overflow (scan LUT)";
        return false;
    }
    const bool captureMemorySnapshots = should_collect_manager_memory_snapshots(manager_effective_config());
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    auto try_private_fallback = [&](const char* triggerReason) -> bool {
        if (!criticalRequest) {
            return false;
        }

        std::uint64_t expectedHash = 0;
        if (!compute_expected_scan_lut_hash(ws, negativeMedium, expectedHash)) {
            PrivateLutFallbackDecision invalidDecision{};
            invalidDecision.reason = "invalid_scan_lut_key";
            trace_private_lut_fallback(
                stageName,
                "private_fallback_denied",
                negativeMedium,
                invalidDecision,
                trace_or(triggerReason, "invalid_scan_lut_key"));
            return false;
        }

        const PrivateLutFallbackDecision decision =
            evaluate_private_lut_fallback(resources, negativeMedium, expectedHash, cfg);
        if (!decision.allowed) {
            trace_private_lut_fallback(
                stageName,
                "private_fallback_denied",
                negativeMedium,
                decision,
                trace_or(triggerReason, decision.reason));
            return false;
        }

        trace_private_lut_fallback(
            stageName,
            "private_fallback_admit",
            negativeMedium,
            decision,
            trace_or(triggerReason, decision.reason));

        std::string fallbackError;
        const bool fallbackOk = JuicerCuda::ensure_scan_lut(
            resources,
            ws,
            negativeMedium,
            cudaStreamOpaque,
            fallbackError);
        if (!fallbackOk) {
            trace_private_lut_fallback(
                stageName,
                "private_fallback_failed",
                negativeMedium,
                decision,
                commands_error_or_cstr(fallbackError, "ensure_failed"));
            if (!fallbackError.empty()) {
                if (!outError.empty()) {
                    outError += " | ";
                }
                outError += std::string(kPrivateLutFallbackFailedPrefix) + " " + fallbackError;
            }
            return false;
        }

        mark_private_lut_fallback_active(resources, negativeMedium, expectedHash);
        trace_private_lut_fallback(
            stageName,
            "private_fallback_served",
            negativeMedium,
            decision,
            trace_or(triggerReason, "served"));
        maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
        return true;
    };
    bool requestPreReclaim = false;
    if (!enforce_pressure_gate(
            transaction,
            resources,
            stageName,
            PressureLane::Upload,
            saturating_u64_to_size_t(uploadRequestBytes),
            criticalRequest,
            requestPreReclaim,
            outError)) {
        if (try_private_fallback("pressure_gate_reject")) {
            outError.clear();
            return true;
        }
        return false;
    }
    if (requestPreReclaim) {
        std::string reclaimError;
        if (!run_reap_pass_for_pressure(
                transaction,
                resources,
                stageName,
                "pressure_pre_upload",
                reclaimError)) {
            outError = commands_error_or_message(reclaimError, "pressure pre-upload reclaim failed");
            if (try_private_fallback("pressure_pre_upload_reclaim_failed")) {
                outError.clear();
                return true;
            }
            return false;
        }
    }
    UploadReservationClaim uploadClaim{};
    if (!acquire_upload_reservation_with_wait(
            transaction,
            stageName,
            uploadRequestBytes,
            criticalRequest,
            uploadClaim,
            outError)) {
        if (try_private_fallback("upload_reservation_reject")) {
            outError.clear();
            return true;
        }
        return false;
    }
    UploadReservationGuard uploadGuard(std::move(uploadClaim));
    BuilderReservationClaim builderClaim{};
    if (!acquire_builder_reservation_with_wait(
            transaction,
            stageName,
            BuilderReservationTier::Lut,
            uploadRequestBytes,
            criticalRequest,
            builderClaim,
            outError)) {
        if (try_private_fallback("builder_reservation_reject")) {
            outError.clear();
            return true;
        }
        return false;
    }
    BuilderReservationGuard builderGuard(std::move(builderClaim));
    TierCircuitAttempt circuitAttempt{};
    std::string circuitError;
    if (!tier_circuit_begin_attempt(
            transaction,
            stageName,
            ResourceTier::Lut,
            tier_circuit_blocks_admission(ResourceTier::Lut),
            circuitAttempt,
            circuitError)) {
        outError = circuitError;
        if (try_private_fallback("tier_circuit_blocked")) {
            outError.clear();
            return true;
        }
        return false;
    }
    const bool ok = JuicerCuda::ensure_scan_lut(resources, ws, negativeMedium, cudaStreamOpaque, outError);
    complete_tier_circuit_attempt(
        transaction,
        stageName,
        circuitAttempt,
        ok,
        "ensure_success",
        outError);
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    return ok;
}
} // namespace

bool command_ensure_scan_lut(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    bool negativeMedium,
    void* cudaStreamOpaque,
    std::string& outError) {
    return command_ensure_scan_lut_internal(
        transaction,
        resources,
        ws,
        negativeMedium,
        true,
        "command_ensure_scan_lut",
        cudaStreamOpaque,
        outError);
}

bool command_ensure_scan_error_flag(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_scan_error_flag")) {
        return false;
    }
    const bool captureMemorySnapshots = should_collect_manager_memory_snapshots(manager_effective_config());
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    const bool ok = JuicerCuda::ensure_scan_error_flag(resources, cudaStreamOpaque, outError);
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    return ok;
}

bool command_ensure_print_illuminant_filtered(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    const Print::Runtime& prt,
    const Print::Params& params,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_print_illuminant_filtered")) {
        return false;
    }
    const std::uint64_t uploadRequestBytes =
        estimate_print_illuminant_upload_bytes(resources, ws, prt, params);
    if (uploadRequestBytes == std::numeric_limits<std::uint64_t>::max()) {
        outError = "upload reservation request byte estimation overflow (print illuminant)";
        return false;
    }
    const bool captureMemorySnapshots = should_collect_manager_memory_snapshots(manager_effective_config());
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    bool requestPreReclaim = false;
    if (!enforce_pressure_gate(
            transaction,
            resources,
            "command_ensure_print_illuminant_filtered",
            PressureLane::Upload,
            saturating_u64_to_size_t(uploadRequestBytes),
            true,
            requestPreReclaim,
            outError)) {
        return false;
    }
    if (requestPreReclaim) {
        std::string reclaimError;
        if (!run_reap_pass_for_pressure(
                transaction,
                resources,
                "command_ensure_print_illuminant_filtered",
                "pressure_pre_upload",
                reclaimError)) {
            outError = commands_error_or_message(reclaimError, "pressure pre-upload reclaim failed");
            return false;
        }
    }
    UploadReservationClaim uploadClaim{};
    if (!acquire_upload_reservation_with_wait(
            transaction,
            "command_ensure_print_illuminant_filtered",
            uploadRequestBytes,
            true,
            uploadClaim,
            outError)) {
        return false;
    }
    UploadReservationGuard uploadGuard(std::move(uploadClaim));
    TierCircuitAttempt circuitAttempt{};
    std::string circuitError;
    if (!tier_circuit_begin_attempt(
            transaction,
            "command_ensure_print_illuminant_filtered",
            ResourceTier::Immutable,
            tier_circuit_blocks_admission(ResourceTier::Immutable),
            circuitAttempt,
            circuitError)) {
        outError = circuitError;
        return false;
    }
    const bool ok =
        JuicerCuda::ensure_print_illuminant_filtered(resources, ws, prt, params, cudaStreamOpaque, outError);
    complete_tier_circuit_attempt(
        transaction,
        "command_ensure_print_illuminant_filtered",
        circuitAttempt,
        ok,
        "ensure_success",
        outError);
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    return ok;
}

bool command_ensure_optics_scratch(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    int width,
    int height,
    bool needBlurredScratch,
    bool needAuxScratch,
    bool needGrainScratch,
    bool needGrainSharedScratch,
    bool needGateMask,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_optics_scratch")) {
        return false;
    }
    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    const bool captureMemorySnapshots = should_collect_manager_memory_snapshots(cfg);
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    const std::size_t growthBytes = estimate_optics_growth_bytes(
        resources,
        width,
        height,
        needBlurredScratch,
        needAuxScratch,
        needGrainScratch,
        needGrainSharedScratch,
        needGateMask);
    if (growthBytes == std::numeric_limits<std::size_t>::max()) {
        outError = "scratch growth byte estimation overflow";
        return false;
    }

    bool requestPreReclaim = false;
    if (!enforce_pressure_gate(
            transaction,
            resources,
            "command_ensure_optics_scratch",
            PressureLane::Builder,
            growthBytes,
            true,
            requestPreReclaim,
            outError)) {
        return false;
    }
    if (requestPreReclaim) {
        std::string reclaimError;
        if (!run_reap_pass_for_pressure(
                transaction,
                resources,
                "command_ensure_optics_scratch",
                "pressure_pre_growth",
                reclaimError)) {
            outError = commands_error_or_message(reclaimError, "pressure pre-growth reclaim failed");
            return false;
        }
    }

    BuilderReservationClaim builderClaim{};
    if (!acquire_builder_reservation_with_wait(
            transaction,
            "command_ensure_optics_scratch",
            BuilderReservationTier::Scratch,
            static_cast<std::uint64_t>(growthBytes),
            true,
            builderClaim,
            outError)) {
        return false;
    }
    BuilderReservationGuard builderGuard(std::move(builderClaim));

    ScratchPolicyClaim scratchClaim{};
    if (!acquire_scratch_policy_claim_with_wait(
            transaction,
            "command_ensure_optics_scratch",
            ScratchWorkClass::Optics,
            width,
            height,
            growthBytes,
            true,
            scratchClaim,
            outError)) {
        return false;
    }
    ScratchPolicyGuard scratchGuard(std::move(scratchClaim));

    ResourceManagerState& managerState = global_state();
    std::uint32_t attempts = 0;
    bool fragmentationRecoveryTriggered = false;
    bool fragmentationRecoveryPendingOutcome = false;
    auto finalizeFragmentationOutcome = [&](bool success, const char* reason) {
        if (!fragmentationRecoveryPendingOutcome) {
            return;
        }
        if (success) {
            telemetry_counter_add(managerState.fragmentationRecoverySuccess, 1);
        }
        else {
            telemetry_counter_add(managerState.fragmentationRecoveryFailures, 1);
        }
        trace_fragmentation_recovery(
            transaction,
            "command_ensure_optics_scratch",
            attempts,
            growthBytes,
            0,
            0,
            0,
            success,
            "outcome",
            reason);
        fragmentationRecoveryPendingOutcome = false;
    };

    while (true) {
        outError.clear();
        TierCircuitAttempt circuitAttempt{};
        std::string circuitError;
        if (!tier_circuit_begin_attempt(
                transaction,
                "command_ensure_optics_scratch",
                ResourceTier::Scratch,
                tier_circuit_blocks_admission(ResourceTier::Scratch),
                circuitAttempt,
                circuitError)) {
            outError = circuitError;
            finalizeFragmentationOutcome(false, "tier_circuit_blocked");
            return false;
        }
        maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
        if (JuicerCuda::ensure_optics_scratch(
                resources,
                width,
                height,
                needBlurredScratch,
                needAuxScratch,
                needGrainScratch,
                needGrainSharedScratch,
                needGateMask,
                cudaStreamOpaque,
                outError)) {
            complete_tier_circuit_attempt(
                transaction,
                "command_ensure_optics_scratch",
                circuitAttempt,
                true,
                "ensure_success",
                outError);
            maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
            if (attempts > 0) {
                telemetry_counter_add(managerState.budgetReclaimRetrySuccess, 1);
            }
            finalizeFragmentationOutcome(true, "allocation_retry_success");
            return true;
        }
        complete_tier_circuit_attempt(
            transaction,
            "command_ensure_optics_scratch",
            circuitAttempt,
            false,
            "ensure_success",
            outError);
        maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);

        const bool allocatorOom = is_allocator_oom_error(outError);
        if (!allocatorOom) {
            finalizeFragmentationOutcome(false, "non_allocator_error");
            return false;
        }
        if (attempts >= cfg.reclaimRetryMaxAttempts) {
            if (cfg.fragmentationRecoveryEnabled && !fragmentationRecoveryTriggered) {
                std::string recoveryError;
                if (!run_fragmentation_recovery_once(
                        transaction,
                        resources,
                        "command_ensure_optics_scratch",
                        growthBytes,
                        attempts,
                        captureMemorySnapshots,
                        recoveryError)) {
                    if (!recoveryError.empty()) {
                        outError += " | fragmentation_recovery_failed: " + recoveryError;
                    }
                    record_allocator_oom_headroom_observation(
                        transaction,
                        "command_ensure_optics_scratch",
                        growthBytes);
                    telemetry_counter_add(managerState.budgetAllocatorOomEvents, 1);
                    return false;
                }
                fragmentationRecoveryTriggered = true;
                fragmentationRecoveryPendingOutcome = true;
                continue;
            }
            record_allocator_oom_headroom_observation(
                transaction,
                "command_ensure_optics_scratch",
                growthBytes);
            telemetry_counter_add(managerState.budgetAllocatorOomEvents, 1);
            finalizeFragmentationOutcome(false, "allocator_oom_final");
            return false;
        }

        ++attempts;
        telemetry_counter_add(managerState.budgetReclaimRetryAttempts, 1);
        std::size_t reclaimedBytes = 0;
        std::string reclaimError;
        if (!JuicerCuda::reap_retired_allocations(resources, reclaimedBytes, reclaimError)) {
            trace_reap_pass(
                transaction,
                "command_ensure_optics_scratch",
                reclaimedBytes,
                false,
                commands_error_or_cstr(reclaimError, "reap_failed"));
            trace_budget_reclaim_retry(
                transaction,
                "command_ensure_optics_scratch",
                attempts,
                reclaimedBytes,
                false,
                commands_error_or_cstr(reclaimError, "reap_failed"));
            if (!reclaimError.empty()) {
                outError += " | reclaim_retry_failed: " + reclaimError;
            }
            telemetry_counter_add(managerState.budgetAllocatorOomEvents, 1);
            finalizeFragmentationOutcome(false, "reap_retry_failed");
            return false;
        }

        if (reclaimedBytes > 0) {
            telemetry_counter_add(managerState.retireReapPasses, 1);
            telemetry_counter_add(managerState.retireReapBytes, reclaimedBytes);
        }
        trace_reap_pass(
            transaction,
            "command_ensure_optics_scratch",
            reclaimedBytes,
            true,
            reclaim_retry_reason(reclaimedBytes));
        trace_budget_reclaim_retry(
            transaction,
            "command_ensure_optics_scratch",
            attempts,
            reclaimedBytes,
            true,
            reclaim_retry_reason(reclaimedBytes));
        maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);

        if (reclaimedBytes == 0) {
            if (cfg.fragmentationRecoveryEnabled && !fragmentationRecoveryTriggered) {
                std::string recoveryError;
                if (!run_fragmentation_recovery_once(
                        transaction,
                        resources,
                        "command_ensure_optics_scratch",
                        growthBytes,
                        attempts,
                        captureMemorySnapshots,
                        recoveryError)) {
                    if (!recoveryError.empty()) {
                        outError += " | fragmentation_recovery_failed: " + recoveryError;
                    }
                    record_allocator_oom_headroom_observation(
                        transaction,
                        "command_ensure_optics_scratch",
                        growthBytes);
                    telemetry_counter_add(managerState.budgetAllocatorOomEvents, 1);
                    return false;
                }
                fragmentationRecoveryTriggered = true;
                fragmentationRecoveryPendingOutcome = true;
                continue;
            }
            record_allocator_oom_headroom_observation(
                transaction,
                "command_ensure_optics_scratch",
                growthBytes);
            telemetry_counter_add(managerState.budgetAllocatorOomEvents, 1);
            finalizeFragmentationOutcome(false, "reap_no_progress");
            return false;
        }
    }
}

bool command_ensure_spatial_dir_scratch(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    int width,
    int height,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_spatial_dir_scratch")) {
        return false;
    }
    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    const bool captureMemorySnapshots = should_collect_manager_memory_snapshots(cfg);
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    const std::size_t growthBytes = estimate_spatial_dir_growth_bytes(resources, width, height);
    if (growthBytes == std::numeric_limits<std::size_t>::max()) {
        outError = "spatial dir scratch growth byte estimation overflow";
        return false;
    }

    bool requestPreReclaim = false;
    if (!enforce_pressure_gate(
            transaction,
            resources,
            "command_ensure_spatial_dir_scratch",
            PressureLane::Builder,
            growthBytes,
            true,
            requestPreReclaim,
            outError)) {
        return false;
    }
    if (requestPreReclaim) {
        std::string reclaimError;
        if (!run_reap_pass_for_pressure(
                transaction,
                resources,
                "command_ensure_spatial_dir_scratch",
                "pressure_pre_growth",
                reclaimError)) {
            outError = commands_error_or_message(reclaimError, "pressure pre-growth reclaim failed");
            return false;
        }
    }

    BuilderReservationClaim builderClaim{};
    if (!acquire_builder_reservation_with_wait(
            transaction,
            "command_ensure_spatial_dir_scratch",
            BuilderReservationTier::Scratch,
            static_cast<std::uint64_t>(growthBytes),
            true,
            builderClaim,
            outError)) {
        return false;
    }
    BuilderReservationGuard builderGuard(std::move(builderClaim));

    ScratchPolicyClaim scratchClaim{};
    if (!acquire_scratch_policy_claim_with_wait(
            transaction,
            "command_ensure_spatial_dir_scratch",
            ScratchWorkClass::SpatialDir,
            width,
            height,
            growthBytes,
            true,
            scratchClaim,
            outError)) {
        return false;
    }
    ScratchPolicyGuard scratchGuard(std::move(scratchClaim));

    ResourceManagerState& managerState = global_state();
    std::uint32_t attempts = 0;
    bool fragmentationRecoveryTriggered = false;
    bool fragmentationRecoveryPendingOutcome = false;
    auto finalizeFragmentationOutcome = [&](bool success, const char* reason) {
        if (!fragmentationRecoveryPendingOutcome) {
            return;
        }
        if (success) {
            telemetry_counter_add(managerState.fragmentationRecoverySuccess, 1);
        }
        else {
            telemetry_counter_add(managerState.fragmentationRecoveryFailures, 1);
        }
        trace_fragmentation_recovery(
            transaction,
            "command_ensure_spatial_dir_scratch",
            attempts,
            growthBytes,
            0,
            0,
            0,
            success,
            "outcome",
            reason);
        fragmentationRecoveryPendingOutcome = false;
    };

    while (true) {
        outError.clear();
        TierCircuitAttempt circuitAttempt{};
        std::string circuitError;
        if (!tier_circuit_begin_attempt(
                transaction,
                "command_ensure_spatial_dir_scratch",
                ResourceTier::Scratch,
                tier_circuit_blocks_admission(ResourceTier::Scratch),
                circuitAttempt,
                circuitError)) {
            outError = circuitError;
            finalizeFragmentationOutcome(false, "tier_circuit_blocked");
            return false;
        }
        maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
        if (JuicerCuda::ensure_spatial_dir_scratch(resources, width, height, cudaStreamOpaque, outError)) {
            complete_tier_circuit_attempt(
                transaction,
                "command_ensure_spatial_dir_scratch",
                circuitAttempt,
                true,
                "ensure_success",
                outError);
            maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
            if (attempts > 0) {
                telemetry_counter_add(managerState.budgetReclaimRetrySuccess, 1);
            }
            finalizeFragmentationOutcome(true, "allocation_retry_success");
            return true;
        }
        complete_tier_circuit_attempt(
            transaction,
            "command_ensure_spatial_dir_scratch",
            circuitAttempt,
            false,
            "ensure_success",
            outError);
        maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);

        const bool allocatorOom = is_allocator_oom_error(outError);
        if (!allocatorOom) {
            finalizeFragmentationOutcome(false, "non_allocator_error");
            return false;
        }
        if (attempts >= cfg.reclaimRetryMaxAttempts) {
            if (cfg.fragmentationRecoveryEnabled && !fragmentationRecoveryTriggered) {
                std::string recoveryError;
                if (!run_fragmentation_recovery_once(
                        transaction,
                        resources,
                        "command_ensure_spatial_dir_scratch",
                        growthBytes,
                        attempts,
                        captureMemorySnapshots,
                        recoveryError)) {
                    if (!recoveryError.empty()) {
                        outError += " | fragmentation_recovery_failed: " + recoveryError;
                    }
                    record_allocator_oom_headroom_observation(
                        transaction,
                        "command_ensure_spatial_dir_scratch",
                        growthBytes);
                    telemetry_counter_add(managerState.budgetAllocatorOomEvents, 1);
                    return false;
                }
                fragmentationRecoveryTriggered = true;
                fragmentationRecoveryPendingOutcome = true;
                continue;
            }
            record_allocator_oom_headroom_observation(
                transaction,
                "command_ensure_spatial_dir_scratch",
                growthBytes);
            telemetry_counter_add(managerState.budgetAllocatorOomEvents, 1);
            finalizeFragmentationOutcome(false, "allocator_oom_final");
            return false;
        }

        ++attempts;
        telemetry_counter_add(managerState.budgetReclaimRetryAttempts, 1);
        std::size_t reclaimedBytes = 0;
        std::string reclaimError;
        if (!JuicerCuda::reap_retired_allocations(resources, reclaimedBytes, reclaimError)) {
            trace_reap_pass(
                transaction,
                "command_ensure_spatial_dir_scratch",
                reclaimedBytes,
                false,
                commands_error_or_cstr(reclaimError, "reap_failed"));
            trace_budget_reclaim_retry(
                transaction,
                "command_ensure_spatial_dir_scratch",
                attempts,
                reclaimedBytes,
                false,
                commands_error_or_cstr(reclaimError, "reap_failed"));
            if (!reclaimError.empty()) {
                outError += " | reclaim_retry_failed: " + reclaimError;
            }
            telemetry_counter_add(managerState.budgetAllocatorOomEvents, 1);
            finalizeFragmentationOutcome(false, "reap_retry_failed");
            return false;
        }

        if (reclaimedBytes > 0) {
            telemetry_counter_add(managerState.retireReapPasses, 1);
            telemetry_counter_add(managerState.retireReapBytes, reclaimedBytes);
        }
        trace_reap_pass(
            transaction,
            "command_ensure_spatial_dir_scratch",
            reclaimedBytes,
            true,
            reclaim_retry_reason(reclaimedBytes));
        trace_budget_reclaim_retry(
            transaction,
            "command_ensure_spatial_dir_scratch",
            attempts,
            reclaimedBytes,
            true,
            reclaim_retry_reason(reclaimedBytes));
        maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);

        if (reclaimedBytes == 0) {
            if (cfg.fragmentationRecoveryEnabled && !fragmentationRecoveryTriggered) {
                std::string recoveryError;
                if (!run_fragmentation_recovery_once(
                        transaction,
                        resources,
                        "command_ensure_spatial_dir_scratch",
                        growthBytes,
                        attempts,
                        captureMemorySnapshots,
                        recoveryError)) {
                    if (!recoveryError.empty()) {
                        outError += " | fragmentation_recovery_failed: " + recoveryError;
                    }
                    record_allocator_oom_headroom_observation(
                        transaction,
                        "command_ensure_spatial_dir_scratch",
                        growthBytes);
                    telemetry_counter_add(managerState.budgetAllocatorOomEvents, 1);
                    return false;
                }
                fragmentationRecoveryTriggered = true;
                fragmentationRecoveryPendingOutcome = true;
                continue;
            }
            record_allocator_oom_headroom_observation(
                transaction,
                "command_ensure_spatial_dir_scratch",
                growthBytes);
            telemetry_counter_add(managerState.budgetAllocatorOomEvents, 1);
            finalizeFragmentationOutcome(false, "reap_no_progress");
            return false;
        }
    }
}

bool command_ensure_spatial_dir_kernel(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    JuicerCuda::Resources::DeviceGaussianKernel& kernel,
    float sigma,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_spatial_dir_kernel")) {
        return false;
    }
    const bool captureMemorySnapshots = should_collect_manager_memory_snapshots(manager_effective_config());
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    const bool ok = JuicerCuda::ensure_spatial_dir_kernel(resources, kernel, sigma, cudaStreamOpaque, outError);
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    return ok;
}

bool command_ensure_gaussian_kernel(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    JuicerCuda::Resources::DeviceGaussianKernel& kernel,
    float sigma,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_gaussian_kernel")) {
        return false;
    }
    const bool captureMemorySnapshots = should_collect_manager_memory_snapshots(manager_effective_config());
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    const bool ok = JuicerCuda::ensure_gaussian_kernel(resources, kernel, sigma, cudaStreamOpaque, outError);
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    return ok;
}

bool command_ensure_halation_kernel(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    JuicerCuda::Resources::DeviceGaussianKernel& kernel,
    float sigma,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_halation_kernel")) {
        return false;
    }
    const bool captureMemorySnapshots = should_collect_manager_memory_snapshots(manager_effective_config());
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    const bool ok = JuicerCuda::ensure_halation_kernel(resources, kernel, sigma, cudaStreamOpaque, outError);
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    return ok;
}

bool command_ensure_auto_exposure_buffers(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    int meterWidth,
    int meterHeight,
    std::uint64_t autoExposureKeyHash,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_auto_exposure_buffers")) {
        return false;
    }
    const bool captureMemorySnapshots = should_collect_manager_memory_snapshots(manager_effective_config());
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    const std::uint64_t normalizedKeyHash = normalize_key_u64(autoExposureKeyHash);

    const ShadowHistoryKey ownershipKey{
        transaction.snapshot.instanceToken.value,
        transaction.snapshot.deviceContextKey
    };

    bool hadPrevious = false;
    bool metadataHit = false;
    {
        AutoExposureOwnershipState& ownershipState = auto_exposure_ownership_state();
        std::lock_guard<std::mutex> lock(ownershipState.mutex);
        auto it = ownershipState.bySubmissionKey.find(ownershipKey);
        if (it != ownershipState.bySubmissionKey.end() && it->second.valid) {
            hadPrevious = true;
            const AutoExposureOwnershipEntry& previous = it->second;
            metadataHit =
                previous.keySchemaVersion == transaction.snapshot.keySchemaVersion &&
                previous.keyHash == normalizedKeyHash &&
                previous.meterWidth == meterWidth &&
                previous.meterHeight == meterHeight;
        }
    }

    telemetry_trace_auto_exposure_ownership(
        transaction.transactionId,
        transaction.snapshot.snapshotId,
        transaction.snapshot.traceSchemaVersion,
        "ManagerOnly",
        "acquire",
        metadataHit,
        normalizedKeyHash,
        meterWidth,
        meterHeight,
        hadPrevious,
        bool_reason(metadataHit, "metadata_hit", "metadata_miss"));

    if (!JuicerCuda::ensure_auto_exposure_buffers(resources, meterWidth, meterHeight, cudaStreamOpaque, outError)) {
        maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
        telemetry_trace_auto_exposure_ownership(
            transaction.transactionId,
            transaction.snapshot.snapshotId,
            transaction.snapshot.traceSchemaVersion,
            "ManagerOnly",
            "ensure_fail",
            false,
            normalizedKeyHash,
            meterWidth,
            meterHeight,
            hadPrevious,
            commands_error_or_cstr(outError, "ensure_failed"));
        return false;
    }
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);

    {
        AutoExposureOwnershipState& ownershipState = auto_exposure_ownership_state();
        std::lock_guard<std::mutex> lock(ownershipState.mutex);
        AutoExposureOwnershipEntry& entry = ownershipState.bySubmissionKey[ownershipKey];
        entry.valid = true;
        entry.keyHash = normalizedKeyHash;
        entry.meterWidth = meterWidth;
        entry.meterHeight = meterHeight;
        entry.keySchemaVersion = transaction.snapshot.keySchemaVersion;
    }

    telemetry_trace_auto_exposure_ownership(
        transaction.transactionId,
        transaction.snapshot.snapshotId,
        transaction.snapshot.traceSchemaVersion,
        "ManagerOnly",
        "publish",
        metadataHit,
        normalizedKeyHash,
        meterWidth,
        meterHeight,
        hadPrevious,
        bool_reason(metadataHit, "reuse", "refresh"));
    return true;
}

bool command_launch_base_pipeline_graph(
    SubmissionTransaction& transaction,
    JuicerCuda::PipelineRunParams& run,
    int renderModeKey,
    void* cudaStreamOpaque,
    int& outCudaErrorCode,
    std::string& outError) {
    outError.clear();

    if (!ensure_active_for_command(transaction, outError, "command_launch_base_pipeline_graph")) {
        return false;
    }

#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
    (void)run;
    (void)renderModeKey;
    (void)cudaStreamOpaque;
    outCudaErrorCode = 0;
    outError = "CUDA is not enabled";
    return false;
#else
    outCudaErrorCode = static_cast<int>(cudaErrorUnknown);
    BasePipelineLaunchFn launchFn = base_pipeline_launch_fn_for_mode(renderModeKey);
    if (!launchFn) {
        outCudaErrorCode = static_cast<int>(cudaErrorInvalidValue);
        return true;
    }

    const cudaStream_t stream = commands_cuda_stream_or_null(cudaStreamOpaque);
    LaunchGraphCounters::record_graph_eligible_submission();
    auto launch_base_pipeline_direct = [&]() -> int {
        LaunchGraphCounters::record_kernel_launch();
        return static_cast<int>(launchFn(&run, reinterpret_cast<void*>(stream)));
    };

    BaseGraphKey key{};
    key.width = run.width;
    key.height = run.height;
    key.nComponents = run.nComponents;
    key.renderMode = renderModeKey;
    const std::uint64_t keyDigest = base_graph_key_digest(key);
    const std::uint64_t requestBytes = estimate_base_graph_request_bytes(key);
    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    constexpr bool kGraphAdmissionCriticalCurrentFrame = false;
    ResourceManagerState& managerState = global_state();
    std::uint64_t supersededLatestSnapshotId = 0;
    if (requestBytes > 0 &&
        should_cancel_superseded_noncritical_builder(
            transaction,
            "command_launch_base_pipeline_graph",
            kGraphAdmissionCriticalCurrentFrame,
            requestBytes,
            supersededLatestSnapshotId)) {
        telemetry_counter_add(managerState.graphNonResidentServeEvents, 1);
        outCudaErrorCode = launch_base_pipeline_direct();
        return true;
    }
    const std::uint64_t graphLargeThresholdBytes = graph_large_entry_threshold_bytes(cfg);
    const std::uint64_t graphLargeCapBytes = graph_large_entry_quarantine_cap_bytes(cfg);
    const std::uint32_t graphLargeCapEntries = graph_large_entry_quarantine_cap_entries(cfg);

    std::shared_ptr<BaseGraphBucketState> bucketPtr =
        get_or_create_base_graph_bucket(transaction.snapshot.deviceContextKey);
    if (!bucketPtr) {
        outCudaErrorCode = static_cast<int>(cudaErrorUnknown);
        outError = "base graph bucket unavailable";
        return false;
    }
    std::lock_guard<std::mutex> bucketLock(bucketPtr->mutex);
    BaseGraphBucketState& bucket = *bucketPtr;
    std::uint64_t activeEpoch = transaction.snapshot.contextEpoch;
    if (activeEpoch == 0) {
        activeEpoch = 1;
    }
    if (bucket.contextEpoch != activeEpoch) {
        clear_base_graph_bucket(bucket);
        bucket.contextEpoch = activeEpoch;
    }

    bucket.useTick++;
    if (bucket.useTick == 0) {
        bucket.useTick = 1;
    }
    const std::uint64_t useTick = bucket.useTick;
    auto applyGraphLargeEntryPolicy = [&](const char* reason) {
        std::uint64_t decayEvictedEntries = 0;
        std::uint64_t capTrimEvictedEntries = 0;
        std::uint64_t keepHotBypassEvents = 0;
        std::uint64_t keepHotForcedEvictEvents = 0;
        bool capHit = false;
        std::uint32_t residentEntries = 0;
        const std::uint64_t residentBytes = trim_graph_large_entry_decay_and_caps_locked(
            bucket,
            graphLargeThresholdBytes,
            graphLargeCapBytes,
            graphLargeCapEntries,
            cfg.keepHotMs,
            monotonic_time_ms(),
            decayEvictedEntries,
            capTrimEvictedEntries,
            keepHotBypassEvents,
            keepHotForcedEvictEvents,
            capHit,
            residentEntries);
        if (decayEvictedEntries > 0) {
            telemetry_counter_add(managerState.graphLargeEntryDecayEvents, decayEvictedEntries);
        }
        if (capTrimEvictedEntries > 0) {
            telemetry_counter_add(managerState.graphLargeEntryTrimEvents, capTrimEvictedEntries);
        }
        if (capHit) {
            telemetry_counter_add(managerState.graphLargeEntryCapHits, 1);
        }
        if (keepHotBypassEvents > 0) {
            telemetry_counter_add(managerState.keepHotBypassEvents, keepHotBypassEvents);
        }
        if (keepHotForcedEvictEvents > 0) {
            telemetry_counter_add(managerState.keepHotForcedEvictEvents, keepHotForcedEvictEvents);
        }
        if (keepHotBypassEvents > 0 || keepHotForcedEvictEvents > 0) {
            trace_keep_hot_decision(
                transaction,
                "command_launch_base_pipeline_graph",
                cfg.keepHotMs,
                keepHotBypassEvents,
                keepHotForcedEvictEvents,
                trace_or(reason, "policy"));
        }
        if (decayEvictedEntries > 0 || capTrimEvictedEntries > 0 || capHit) {
            trace_graph_large_entry_quarantine(
                transaction,
                "command_launch_base_pipeline_graph",
                graphLargeThresholdBytes,
                graphLargeCapBytes,
                graphLargeCapEntries,
                residentBytes,
                residentEntries,
                decayEvictedEntries,
                capTrimEvictedEntries,
                capHit,
                trace_or(reason, "policy"));
        }
    };
    applyGraphLargeEntryPolicy("pre_admission");

    BaseGraphEntry* found = find_base_graph_entry(bucket, key);
    const bool reusedResidentGraph = (found != nullptr);
    TierCircuitAttempt graphCircuitAttempt{};
    bool graphCircuitAttemptActive = false;
    if (!found) {
        std::uint32_t observedProbationHits = 0;
        auto probationIt = bucket.probationHitsByDigest.find(keyDigest);
        if (probationIt != bucket.probationHitsByDigest.end()) {
            observedProbationHits = probationIt->second;
        }
        LargeEntryReadmitDecision readmitDecision{};
        readmitDecision.enabled =
            (cfg.largeEntryReadmitCooldownMs > 0) || (cfg.largeEntryGhostHitsForReadmit > 0);
        readmitDecision.candidate =
            (graphLargeThresholdBytes > 0) && (requestBytes >= graphLargeThresholdBytes);
        readmitDecision.criticalCurrentFrame = kGraphAdmissionCriticalCurrentFrame;
        readmitDecision.cooldownMs = cfg.largeEntryReadmitCooldownMs;
        readmitDecision.ghostHitsRequired = cfg.largeEntryGhostHitsForReadmit;
        if (readmitDecision.enabled) {
            readmitDecision.reason = "not_candidate";
        }
        else {
            readmitDecision.reason = "disabled";
        }
        if (readmitDecision.enabled && readmitDecision.candidate && !kGraphAdmissionCriticalCurrentFrame) {
            auto readmitIt = bucket.largeEntryReadmitByDigest.find(keyDigest);
            if (readmitIt != bucket.largeEntryReadmitByDigest.end()) {
                readmitDecision.hadHistory = true;
                GraphLargeEntryReadmitState& readmitState = readmitIt->second;
                const std::uint64_t nowMs = monotonic_time_ms();
                readmitDecision.ageMs = commands_elapsed_ms_since(nowMs, readmitState.lastEvictedMs);
                readmitDecision.inCooldown =
                    (cfg.largeEntryReadmitCooldownMs > 0) &&
                    (readmitDecision.ageMs < static_cast<std::uint64_t>(cfg.largeEntryReadmitCooldownMs));

                if (readmitDecision.inCooldown) {
                    if (readmitState.ghostHits < std::numeric_limits<std::uint32_t>::max()) {
                        ++readmitState.ghostHits;
                    }
                    readmitDecision.observedGhostHits = readmitState.ghostHits;
                    if (cfg.largeEntryGhostHitsForReadmit > 0 &&
                        readmitState.ghostHits >= cfg.largeEntryGhostHitsForReadmit) {
                        readmitDecision.ghostBypass = true;
                        readmitDecision.reason = "ghost_hit_bypass";
                        bucket.largeEntryReadmitByDigest.erase(readmitIt);
                    }
                    else {
                        readmitDecision.blocked = true;
                        readmitDecision.reason = "cooldown_blocked";
                    }
                }
                else {
                    readmitDecision.reason = "cooldown_expired";
                    bucket.largeEntryReadmitByDigest.erase(readmitIt);
                }
            }
            else {
                readmitDecision.reason = "no_history";
            }
        }
        else if (readmitDecision.enabled && readmitDecision.candidate && kGraphAdmissionCriticalCurrentFrame) {
            readmitDecision.reason = "critical_bypass";
        }
        trace_large_entry_readmit_decision(
            transaction,
            "command_launch_base_pipeline_graph",
            keyDigest,
            requestBytes,
            graphLargeThresholdBytes,
            readmitDecision);
        if (readmitDecision.ghostBypass) {
            telemetry_counter_add(managerState.largeEntryReadmitGhostBypassEvents, 1);
        }
        if (readmitDecision.blocked) {
            telemetry_counter_add(managerState.largeEntryReadmitBlockedEvents, 1);
            telemetry_counter_add(managerState.graphNonResidentServeEvents, 1);
            outCudaErrorCode = launch_base_pipeline_direct();
            return true;
        }

        CacheAdmissionInput admissionInput{};
        admissionInput.requestBytes = requestBytes;
        admissionInput.cacheTargetBytes = cfg.managerSoftTargetBytes;
        admissionInput.maxCacheableEntryBytes = cfg.maxCacheableEntryBytes;
        admissionInput.maxCacheableEntryPctOfTarget = cfg.maxCacheableEntryPctOfTarget;
        admissionInput.largeEntryProbationThresholdBytes = cfg.largeEntryProbationThresholdBytes;
        admissionInput.largeEntryProbationHitsRequired = cfg.largeEntryProbationHitsRequired;
        admissionInput.observedProbationHits = observedProbationHits;
        admissionInput.criticalCurrentFrame = kGraphAdmissionCriticalCurrentFrame;
        const CacheAdmissionDecision admissionDecision = classify_cache_admission(admissionInput);
        const AdmissionChurnSnapshot churnSnapshot = sample_admission_churn_state(
            transaction,
            cfg,
            keyDigest,
            observedProbationHits);
        const std::uint32_t churnProbationHitsRequired = effective_probation_hits_required(
            admissionDecision.probationHitsRequired,
            churnSnapshot);
        const bool churnProbationAllowDurable =
            !admissionDecision.probationApplied ||
            (next_probation_hits(observedProbationHits) >= churnProbationHitsRequired);
        trace_cache_admission_decision(
            transaction,
            "command_launch_base_pipeline_graph",
            admissionDecision,
            churnSnapshot,
            kGraphAdmissionCriticalCurrentFrame,
            requestBytes,
            observedProbationHits,
            keyDigest,
            "graph_miss");
        if (admissionDecision.admissionClass == CacheAdmissionClass::TooLargeToCache) {
            telemetry_counter_add(managerState.cacheAdmissionTooLargeEvents, 1);
        }
        if (admissionDecision.probationApplied) {
            const std::uint32_t nextObservedHits = next_probation_hits(observedProbationHits);
            trace_probation_decision(
                transaction,
                "command_launch_base_pipeline_graph",
                keyDigest,
                nextObservedHits,
                churnProbationHitsRequired,
                churnProbationAllowDurable,
                admissionDecision.reason);
            if (churnProbationAllowDurable) {
                telemetry_counter_add(managerState.cacheAdmissionProbationAdmitEvents, 1);
            }
            else {
                telemetry_counter_add(managerState.cacheAdmissionProbationDeferredEvents, 1);
            }
        }
        if (admissionDecision.reason &&
            std::string_view(admissionDecision.reason).find("critical_override") != std::string_view::npos) {
            telemetry_counter_add(managerState.cacheAdmissionCriticalOverrideEvents, 1);
        }

        const bool allowDurableAdmission =
            commands_allow_durable_admission(admissionDecision, churnProbationAllowDurable);
        if (!allowDurableAdmission) {
            if (admissionDecision.probationApplied) {
                std::uint32_t& probationHits = bucket.probationHitsByDigest[keyDigest];
                if (probationHits < std::numeric_limits<std::uint32_t>::max()) {
                    ++probationHits;
                }
            }
            else {
                bucket.probationHitsByDigest.erase(keyDigest);
            }
            telemetry_counter_add(managerState.graphNonResidentServeEvents, 1);
            outCudaErrorCode = launch_base_pipeline_direct();
            return true;
        }

        std::string circuitError;
        if (!tier_circuit_begin_attempt(
                transaction,
                "command_launch_base_pipeline_graph",
                ResourceTier::Graph,
                tier_circuit_blocks_admission(ResourceTier::Graph),
                graphCircuitAttempt,
                circuitError)) {
            telemetry_counter_add(managerState.graphNonResidentServeEvents, 1);
            outCudaErrorCode = launch_base_pipeline_direct();
            return true;
        }
        graphCircuitAttemptActive = true;

        BuilderReservationClaim builderClaim{};
        ReservationAttemptInfo builderReservation{};
        telemetry_counter_add(managerState.builderReservationRequests, 1);
        if (!try_acquire_builder_reservation_claim(
                transaction,
                BuilderReservationTier::Graph,
                requestBytes,
                kGraphAdmissionCriticalCurrentFrame,
                builderClaim,
                builderReservation)) {
            const ReservationDecision& decision = builderReservation.decision;
            const bool fairnessDeferred =
                (decision.reason && std::string_view(decision.reason) == "fairness_tokens_exhausted");
            if (decision.shouldWait) {
                telemetry_counter_add(managerState.builderReservationDeferred, 1);
                if (fairnessDeferred) {
                    telemetry_counter_add(managerState.builderFairnessTokenDeferred, 1);
                }
                trace_builder_reservation_decision(
                    transaction,
                    "command_launch_base_pipeline_graph",
                    BuilderReservationTier::Graph,
                    requestBytes,
                    builderReservation.bytesInFlight,
                    builderReservation.capBytes,
                    builderReservation.thresholdBytes,
                    decision,
                    kGraphAdmissionCriticalCurrentFrame,
                    builderReservation.instanceToken,
                    builderReservation.sharedTokens,
                    builderReservation.criticalTokens,
                    0,
                    "deferred_nonresident");
            }
            else {
                telemetry_counter_add(managerState.builderReservationDenied, 1);
                trace_builder_reservation_decision(
                    transaction,
                    "command_launch_base_pipeline_graph",
                    BuilderReservationTier::Graph,
                    requestBytes,
                    builderReservation.bytesInFlight,
                    builderReservation.capBytes,
                    builderReservation.thresholdBytes,
                    decision,
                    kGraphAdmissionCriticalCurrentFrame,
                    builderReservation.instanceToken,
                    builderReservation.sharedTokens,
                    builderReservation.criticalTokens,
                    0,
                    "denied_nonresident");
            }
            if (graphCircuitAttemptActive) {
                tier_circuit_cancel_attempt(
                    transaction,
                    "command_launch_base_pipeline_graph",
                    graphCircuitAttempt,
                    "nonresident_builder_gate");
                graphCircuitAttemptActive = false;
            }
            telemetry_counter_add(managerState.graphNonResidentServeEvents, 1);
            outCudaErrorCode = launch_base_pipeline_direct();
            return true;
        }
        telemetry_counter_add(managerState.builderReservationGranted, 1);
        if (JTRACE_ENABLED(3)) {
            trace_builder_reservation_decision(
                transaction,
                "command_launch_base_pipeline_graph",
                BuilderReservationTier::Graph,
                requestBytes,
                builderReservation.bytesInFlight,
                builderReservation.capBytes,
                builderReservation.thresholdBytes,
                builderReservation.decision,
                kGraphAdmissionCriticalCurrentFrame,
                builderReservation.instanceToken,
                builderReservation.sharedTokens,
                builderReservation.criticalTokens,
                0,
                "admitted");
        }
        BuilderReservationGuard builderGuard(std::move(builderClaim));

        bucket.probationHitsByDigest.erase(keyDigest);
        bucket.largeEntryReadmitByDigest.erase(keyDigest);
        std::uint64_t keepHotBypassEvents = 0;
        std::uint64_t keepHotForcedEvictEvents = 0;
        found = build_base_graph_entry(
            bucket,
            key,
            launchFn,
            run,
            stream,
            graphLargeThresholdBytes,
            cfg.keepHotMs,
            keepHotBypassEvents,
            keepHotForcedEvictEvents);
        if (keepHotBypassEvents > 0) {
            telemetry_counter_add(managerState.keepHotBypassEvents, keepHotBypassEvents);
        }
        if (keepHotForcedEvictEvents > 0) {
            telemetry_counter_add(managerState.keepHotForcedEvictEvents, keepHotForcedEvictEvents);
        }
        if (keepHotBypassEvents > 0 || keepHotForcedEvictEvents > 0) {
            trace_keep_hot_decision(
                transaction,
                "command_launch_base_pipeline_graph",
                cfg.keepHotMs,
                keepHotBypassEvents,
                keepHotForcedEvictEvents,
                "base_graph_cap");
        }
        if (found) {
            applyGraphLargeEntryPolicy("post_build");
            found = find_base_graph_entry(bucket, key);
        }
    }
    else {
        bucket.probationHitsByDigest.erase(keyDigest);
        bucket.largeEntryReadmitByDigest.erase(keyDigest);
    }

    if (!found || !found->execOpaque || !found->kernelNodeOpaque) {
        if (graphCircuitAttemptActive) {
            tier_circuit_record_outcome(
                transaction,
                "command_launch_base_pipeline_graph",
                graphCircuitAttempt,
                false,
                "durable_build_failed");
            graphCircuitAttemptActive = false;
        }
        outCudaErrorCode = launch_base_pipeline_direct();
        return true;
    }

    found->lastUseTick = useTick;
    found->lastUseMs = monotonic_time_ms();

    cudaGraphExec_t exec = reinterpret_cast<cudaGraphExec_t>(found->execOpaque);
    cudaGraphNode_t node = reinterpret_cast<cudaGraphNode_t>(found->kernelNodeOpaque);

    cudaKernelNodeParams nodeParams{};
    nodeParams.func = found->kernelFuncOpaque;
    nodeParams.gridDim = dim3(found->gridX, found->gridY, found->gridZ);
    nodeParams.blockDim = dim3(found->blockX, found->blockY, found->blockZ);
    nodeParams.sharedMemBytes = found->sharedMemBytes;
    void* kernelArgs[] = { &run };
    nodeParams.kernelParams = kernelArgs;
    nodeParams.extra = nullptr;
    cudaError_t setErr = cudaGraphExecKernelNodeSetParams(exec, node, &nodeParams);
    if (setErr != cudaSuccess) {
        destroy_base_graph_entry(*found);
        applyGraphLargeEntryPolicy("kernel_param_update_failed");
        if (graphCircuitAttemptActive) {
            tier_circuit_record_outcome(
                transaction,
                "command_launch_base_pipeline_graph",
                graphCircuitAttempt,
                false,
                "kernel_param_update_failed");
            graphCircuitAttemptActive = false;
        }
        outCudaErrorCode = launch_base_pipeline_direct();
        return true;
    }

    LaunchGraphCounters::record_kernel_launch();
    cudaError_t runErr = cudaGraphLaunch(exec, stream);
    if (runErr == cudaSuccess) {
        runErr = cudaGetLastError();
    }
    if (runErr != cudaSuccess) {
        destroy_base_graph_entry(*found);
        applyGraphLargeEntryPolicy("graph_launch_failed");
        if (graphCircuitAttemptActive) {
            tier_circuit_record_outcome(
                transaction,
                "command_launch_base_pipeline_graph",
                graphCircuitAttempt,
                false,
                "graph_launch_failed");
            graphCircuitAttemptActive = false;
        }
        outCudaErrorCode = launch_base_pipeline_direct();
        return true;
    }

    if (reusedResidentGraph) {
        LaunchGraphCounters::record_graph_replay_hit();
    }
    if (graphCircuitAttemptActive) {
        tier_circuit_record_outcome(
            transaction,
            "command_launch_base_pipeline_graph",
            graphCircuitAttempt,
            true,
            "durable_build_success");
        graphCircuitAttemptActive = false;
    }
    outCudaErrorCode = static_cast<int>(cudaSuccess);
    return true;
#endif
}

bool error_is_scratch_exhausted(const std::string& error) noexcept {
    return error.rfind(kScratchExhaustedPrefix, 0) == 0 ||
        error.rfind(kReservationDeferredPrefix, 0) == 0;
}
