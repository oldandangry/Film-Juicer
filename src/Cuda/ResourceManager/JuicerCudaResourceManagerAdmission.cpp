// Cuda/ResourceManager/JuicerCudaResourceManagerAdmission.cpp
//
// Included by JuicerCudaResourceManager.cpp (single-TU split).
std::uint64_t estimate_graph_cache_active_bytes_for_context(const DeviceContextKey& key) noexcept;
std::uint64_t evict_noncritical_graph_entries_for_context(const DeviceContextKey& key) noexcept;
void maybe_apply_async_mempool_release_policy(
    const SubmissionTransaction& transaction,
    PressureState pressureState,
    const ResourceManagerConfigEffective& cfg,
    const char* reason);
void trim_large_frame_quarantine_decay_locked(
    ScratchContextState& contextState,
    std::uint64_t nowMs) noexcept;
void snapshot_bucket_state_locked(
    const ScratchContextState& contextState,
    const ScratchBucketEntry& bucketEntry,
    const ScratchBucketKey& bucketKey,
    ScratchPolicySnapshot& outSnapshot) noexcept;
void trace_scratch_policy_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const char* result,
    std::size_t requestBytes,
    const ScratchPolicySnapshot& snapshot,
    int waitMs);
void trace_transient_reservation_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::size_t requestBytes,
    std::uint64_t bytesInFlight,
    std::uint64_t capBytes,
    std::uint64_t thresholdBytes,
    const ReservationDecision& decision,
    bool criticalCurrentFrame,
    int waitMs,
    const char* reason);
void trace_upload_reservation_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::uint64_t requestBytes,
    std::uint64_t bytesInFlight,
    std::uint64_t capBytes,
    std::uint64_t thresholdBytes,
    const ReservationDecision& decision,
    bool criticalCurrentFrame,
    std::uint64_t instanceToken,
    std::uint32_t sharedTokens,
    std::uint32_t criticalTokens,
    int waitMs,
    const char* reason);
void trace_builder_reservation_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    BuilderReservationTier tier,
    std::uint64_t requestBytes,
    std::uint64_t bytesInFlight,
    std::uint64_t capBytes,
    std::uint64_t thresholdBytes,
    const ReservationDecision& decision,
    bool criticalCurrentFrame,
    std::uint64_t instanceToken,
    std::uint32_t sharedTokens,
    std::uint32_t criticalTokens,
    int waitMs,
    const char* reason);
void trace_lane_wait_event(
    const SubmissionTransaction& transaction,
    const char* commandName,
    PressureLane lane,
    bool criticalCurrentFrame,
    int waitMs,
    const char* outcome,
    const char* reason);

ScratchPolicyState& scratch_policy_state() noexcept {
    static ScratchPolicyState state{};
    return state;
}

UploadReservationState& upload_reservation_state() noexcept {
    static UploadReservationState state{};
    return state;
}

BuilderReservationState& builder_reservation_state() noexcept {
    static BuilderReservationState state{};
    return state;
}

PressurePolicyState& pressure_policy_state() noexcept {
    static PressurePolicyState state{};
    return state;
}

TierCircuitPolicyState& tier_circuit_policy_state() noexcept {
    static TierCircuitPolicyState state{};
    return state;
}

AdmissionChurnPolicyState& admission_churn_policy_state() noexcept {
    static AdmissionChurnPolicyState state{};
    return state;
}

OptionalHeuristicTraceState& optional_heuristic_trace_state() noexcept {
    static OptionalHeuristicTraceState state{};
    return state;
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

bool is_allocator_oom_error(const std::string& error) noexcept {
    if (error.empty()) {
        return false;
    }
    return contains_ascii_case_insensitive(error, "out of memory") ||
        contains_ascii_case_insensitive(error, "memory allocation");
}

bool tier_circuit_should_count_failure(const std::string& error) noexcept {
    if (error.empty()) {
        return true;
    }
    if (error.rfind(kScratchExhaustedPrefix, 0) == 0) {
        return false;
    }
    if (error.rfind(kReservationDeferredPrefix, 0) == 0) {
        return false;
    }
    if (error.rfind("pressure_shed_noncritical:", 0) == 0) {
        return false;
    }
    if (error.rfind("pressure_copy_compute_guard:", 0) == 0) {
        return false;
    }
    return true;
}

std::uint64_t tier_target_basis_points(
    const ResourceManagerConfigEffective& cfg,
    ResourceTier tier) noexcept {
    switch (tier) {
    case ResourceTier::Immutable:
        return cfg.tierTargetImmutableBp;
    case ResourceTier::Lut:
        return cfg.tierTargetLutBp;
    case ResourceTier::Scratch:
        return cfg.tierTargetScratchBp;
    case ResourceTier::Graph:
        return cfg.tierTargetGraphBp;
    default:
        return 0;
    }
}

std::uint64_t tier_target_bytes(
    const ResourceManagerConfigEffective& cfg,
    ResourceTier tier) noexcept {
    if (cfg.managerSoftTargetBytes == 0) {
        return 0;
    }
    const std::uint64_t bp = std::min<std::uint64_t>(
        tier_target_basis_points(cfg, tier),
        kBasisPointsDenominator);
    std::uint64_t weighted = 0;
    if (!mul_u64_checked(cfg.managerSoftTargetBytes, bp, weighted)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return weighted / kBasisPointsDenominator;
}

inline bool pressure_policy_enabled(const ResourceManagerConfigEffective& cfg) noexcept {
    return cfg.managerSoftTargetBytes > 0;
}

std::uint64_t transient_reservation_cap_bytes(const ResourceManagerConfigEffective& cfg) noexcept {
    if (cfg.managerSoftTargetBytes == 0) {
        return kTransientReservationCapDefaultBytes;
    }
    const std::uint64_t quarterTarget = cfg.managerSoftTargetBytes / 4ull;
    return std::max<std::uint64_t>(
        kTransientReservationThresholdDefaultBytes,
        std::min<std::uint64_t>(kTransientReservationCapDefaultBytes, quarterTarget));
}

std::uint64_t transient_reservation_threshold_bytes(const ResourceManagerConfigEffective& cfg) noexcept {
    const std::uint64_t capBytes = transient_reservation_cap_bytes(cfg);
    if (capBytes == 0) {
        return 0;
    }
    return std::min<std::uint64_t>(kTransientReservationThresholdDefaultBytes, capBytes);
}

std::uint64_t builder_reservation_cap_bytes(
    const ResourceManagerConfigEffective& cfg,
    BuilderReservationTier tier) noexcept {
    std::uint32_t configCapMb = 0;
    std::uint64_t defaultCapBytes = kScratchBuilderReservationCapDefaultBytes;
    switch (tier) {
    case BuilderReservationTier::Scratch:
        configCapMb = cfg.scratchBuilderBytesInFlightLimitMB;
        defaultCapBytes = kScratchBuilderReservationCapDefaultBytes;
        break;
    case BuilderReservationTier::Lut:
        configCapMb = cfg.lutBuilderBytesInFlightLimitMB;
        defaultCapBytes = kLutBuilderReservationCapDefaultBytes;
        break;
    case BuilderReservationTier::Graph:
        configCapMb = cfg.graphBuilderBytesInFlightLimitMB;
        defaultCapBytes = kGraphBuilderReservationCapDefaultBytes;
        break;
    default:
        configCapMb = cfg.scratchBuilderBytesInFlightLimitMB;
        defaultCapBytes = kScratchBuilderReservationCapDefaultBytes;
        break;
    }

    std::uint64_t capBytes = 0;
    if (!mul_u64_checked(static_cast<std::uint64_t>(configCapMb), kBytesPerMiB, capBytes)) {
        capBytes = std::numeric_limits<std::uint64_t>::max();
    }
    if (capBytes == 0) {
        capBytes = defaultCapBytes;
    }
    return std::max<std::uint64_t>(kBuilderReservationThresholdDefaultBytes, capBytes);
}

std::uint64_t builder_reservation_threshold_bytes(
    const ResourceManagerConfigEffective& cfg,
    BuilderReservationTier tier) noexcept {
    const std::uint64_t capBytes = builder_reservation_cap_bytes(cfg, tier);
    if (capBytes == 0) {
        return 0;
    }
    return std::min<std::uint64_t>(kBuilderReservationThresholdDefaultBytes, capBytes);
}

bool builder_context_has_inflight(const BuilderReservationContextState& contextState) noexcept {
    return contextState.inFlightScratchBytes > 0 ||
        contextState.inFlightLutBytes > 0 ||
        contextState.inFlightGraphBytes > 0;
}

std::uint64_t upload_reservation_cap_bytes(const ResourceManagerConfigEffective& cfg) noexcept {
    std::uint64_t configCapBytes = 0;
    if (!mul_u64_checked(
            static_cast<std::uint64_t>(cfg.uploadBytesInFlightLimitMB),
            kBytesPerMiB,
            configCapBytes)) {
        configCapBytes = std::numeric_limits<std::uint64_t>::max();
    }
    if (configCapBytes == 0) {
        configCapBytes = kUploadReservationCapDefaultBytes;
    }
    if (cfg.managerSoftTargetBytes == 0) {
        return std::max<std::uint64_t>(kUploadReservationThresholdDefaultBytes, configCapBytes);
    }
    const std::uint64_t quarterTarget = cfg.managerSoftTargetBytes / 4ull;
    const std::uint64_t boundedCap = std::min<std::uint64_t>(configCapBytes, quarterTarget);
    return std::max<std::uint64_t>(kUploadReservationThresholdDefaultBytes, boundedCap);
}

std::uint64_t upload_reservation_threshold_bytes(const ResourceManagerConfigEffective& cfg) noexcept {
    const std::uint64_t capBytes = upload_reservation_cap_bytes(cfg);
    if (capBytes == 0) {
        return 0;
    }
    return std::min<std::uint64_t>(kUploadReservationThresholdDefaultBytes, capBytes);
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
    const std::uint64_t wsCoreHash =
        (ws.uploadCoreHash != 0) ? ws.uploadCoreHash : ws.coreHash;
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

void refill_upload_fairness_tokens(
    UploadReservationContextState::FairnessEntry& entry,
    const ResourceManagerConfigEffective& cfg,
    std::uint64_t nowMs) noexcept {
    const std::uint32_t sharedPerTick = std::max<std::uint32_t>(1u, cfg.uploadFairnessTokensPerTick);
    const std::uint32_t criticalPerTick = std::min<std::uint32_t>(
        std::max<std::uint32_t>(1u, cfg.criticalUploadReservedTokens),
        sharedPerTick);

    if (entry.lastRefillMs == 0) {
        entry.sharedTokens = sharedPerTick;
        entry.criticalTokens = criticalPerTick;
        entry.lastRefillMs = nowMs;
        return;
    }

    const std::uint64_t elapsedMs = (nowMs > entry.lastRefillMs) ? (nowMs - entry.lastRefillMs) : 0;
    if (elapsedMs < kUploadFairnessTickMs) {
        return;
    }

    entry.sharedTokens = sharedPerTick;
    entry.criticalTokens = criticalPerTick;
    entry.lastRefillMs = nowMs;
}

void refill_builder_fairness_tokens(
    BuilderReservationContextState::FairnessEntry& entry,
    const ResourceManagerConfigEffective& cfg,
    std::uint64_t nowMs) noexcept {
    const std::uint32_t sharedPerTick = std::max<std::uint32_t>(1u, cfg.builderFairnessTokensPerTick);
    const std::uint32_t criticalPerTick = std::min<std::uint32_t>(
        std::max<std::uint32_t>(1u, cfg.criticalBuilderReservedTokens),
        sharedPerTick);

    if (entry.lastRefillMs == 0) {
        entry.sharedTokens = sharedPerTick;
        entry.criticalTokens = criticalPerTick;
        entry.lastRefillMs = nowMs;
        return;
    }

    const std::uint64_t elapsedMs = (nowMs > entry.lastRefillMs) ? (nowMs - entry.lastRefillMs) : 0;
    if (elapsedMs < kBuilderFairnessTickMs) {
        return;
    }

    entry.sharedTokens = sharedPerTick;
    entry.criticalTokens = criticalPerTick;
    entry.lastRefillMs = nowMs;
}

std::uint64_t estimate_scan_lut_upload_bytes(
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    bool negativeMedium) noexcept {
    const Scanner::ScannerStaticKey& staticKey = negativeMedium ? ws.negativeStaticKey : ws.printStaticKey;
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

    const Scanner::ScannerMediumRuntime& medium = negativeMedium ? ws.negativeMediumRuntime : ws.printMediumRuntime;
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
            negativeMedium ? resources.scanNegativeLut : resources.scanPrintLut;
        cached = dst.log2XYZ && dst.res == res && dst.hash == expectedHash;
    }
    return cached ? 0 : bytes;
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
    const std::uint64_t wsCoreHash =
        (ws.uploadCoreHash != 0) ? ws.uploadCoreHash : ws.coreHash;
    if (wsCoreHash == 0) {
        return 0;
    }
    const float yKey = std::isfinite(params.yFilter) ? params.yFilter : 0.0f;
    const float mKey = std::isfinite(params.mFilter) ? params.mFilter : 0.0f;
    const float cKey = std::isfinite(params.cFilter) ? params.cFilter : 0.0f;
    const std::uint64_t neutralFilterHash =
        (prt.neutralFilterHash != 0) ? prt.neutralFilterHash : Print::kDefaultNeutralFilterHash;

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

bool try_acquire_scratch_policy_claim(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const DeviceContextKey& contextKey,
    const ScratchBucketKey& bucketKey,
    std::size_t requestBytes,
    bool criticalCurrentFrame,
    ScratchPolicyClaim& outClaim,
    ScratchPolicySnapshot& outSnapshot,
    ReservationAttemptInfo& outReservation) noexcept {
    outReservation = ReservationAttemptInfo{};
    ScratchPolicyState& state = scratch_policy_state();
    std::lock_guard<std::mutex> lock(state.mutex);

    ScratchContextState& contextState = state.byContext[contextKey];
    const std::uint64_t nowMs = monotonic_time_ms();
    trim_large_frame_quarantine_decay_locked(contextState, nowMs);

    ScratchBucketEntry& bucketEntry = contextState.buckets[bucketKey];
    bucketEntry.attemptCount += 1;
    global_state().scratchBucketAcquireAttempts.fetch_add(1, std::memory_order_relaxed);

    if (requestBytes == 0) {
        bucketEntry.reuseEvents += 1;
        global_state().scratchReuseEvents.fetch_add(1, std::memory_order_relaxed);
        global_state().transientNonManagerBytes.store(state.totalInFlightBytes, std::memory_order_relaxed);
        snapshot_bucket_state_locked(contextState, bucketEntry, bucketKey, outSnapshot);
        return true;
    }

    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    outReservation.considered = true;
    outReservation.bytesInFlight = state.totalInFlightBytes;
    outReservation.capBytes = transient_reservation_cap_bytes(cfg);
    outReservation.thresholdBytes = transient_reservation_threshold_bytes(cfg);

    ReservationDecision reservationDecision{};
    if (static_cast<std::uint64_t>(requestBytes) < outReservation.thresholdBytes) {
        reservationDecision.granted = true;
        reservationDecision.reason = "below_threshold";
    }
    else {
        ReservationInput reservationInput{};
        reservationInput.kind = ReservationKind::TransientNonManager;
        reservationInput.requestBytes = static_cast<std::uint64_t>(requestBytes);
        reservationInput.bytesInFlight = state.totalInFlightBytes;
        reservationInput.capBytes = outReservation.capBytes;
        reservationInput.criticalCurrentFrame = criticalCurrentFrame;
        reservationDecision = classify_reservation(reservationInput);
    }
    outReservation.decision = reservationDecision;
    global_state().transientReservationRequests.fetch_add(1, std::memory_order_relaxed);
    if (!reservationDecision.granted) {
        if (reservationDecision.shouldWait) {
            global_state().transientReservationDeferred.fetch_add(1, std::memory_order_relaxed);
        }
        else {
            global_state().transientReservationDenied.fetch_add(1, std::memory_order_relaxed);
        }
        trace_transient_reservation_decision(
            transaction,
            commandName,
            requestBytes,
            outReservation.bytesInFlight,
            outReservation.capBytes,
            outReservation.thresholdBytes,
            reservationDecision,
            criticalCurrentFrame,
            static_cast<int>(reservationDecision.waitMs),
            reservationDecision.shouldWait ? "deferred" : "denied");
        snapshot_bucket_state_locked(contextState, bucketEntry, bucketKey, outSnapshot);
        return false;
    }

    const std::size_t effectiveCapBytes = effective_temp_scratch_bytes_cap(requestBytes);
    const bool setsOk = bucketEntry.inFlightSets < kMaxTempScratchSets;
    const bool bytesOk = requestBytes <=
        (effectiveCapBytes - std::min(bucketEntry.inFlightBytes, effectiveCapBytes));
    if (!setsOk || !bytesOk) {
        bucketEntry.exhaustedCount += 1;
        global_state().scratchBucketExhaustedEvents.fetch_add(1, std::memory_order_relaxed);

        const bool starvationNow =
            (bucketEntry.attemptCount >= 8) &&
            (bucketEntry.exhaustedCount * 4 >= bucketEntry.attemptCount);
        if (starvationNow && !bucketEntry.starvationLatched) {
            bucketEntry.starvationLatched = true;
            global_state().scratchBucketStarvationEvents.fetch_add(1, std::memory_order_relaxed);
        }
        else if (!starvationNow) {
            bucketEntry.starvationLatched = false;
        }

        snapshot_bucket_state_locked(contextState, bucketEntry, bucketKey, outSnapshot);
        return false;
    }

    bucketEntry.inFlightSets += 1;
    bucketEntry.inFlightBytes += requestBytes;
    bucketEntry.allocGrowthEvents += 1;
    {
        std::size_t nextContextBytes = 0;
        if (add_bytes_checked(contextState.totalInFlightBytes, requestBytes, nextContextBytes)) {
            contextState.totalInFlightBytes = nextContextBytes;
        }
        else {
            contextState.totalInFlightBytes = std::numeric_limits<std::size_t>::max();
        }
    }
    {
        std::uint64_t nextGlobalBytes = 0;
        if (add_u64_checked(state.totalInFlightBytes, static_cast<std::uint64_t>(requestBytes), nextGlobalBytes)) {
            state.totalInFlightBytes = nextGlobalBytes;
        }
        else {
            state.totalInFlightBytes = std::numeric_limits<std::uint64_t>::max();
        }
    }
    global_state().scratchAllocGrowthEvents.fetch_add(1, std::memory_order_relaxed);
    bucketEntry.starvationLatched = false;
    global_state().transientNonManagerBytes.store(state.totalInFlightBytes, std::memory_order_relaxed);
    global_state().transientReservationGranted.fetch_add(1, std::memory_order_relaxed);
    if (reservationDecision.reason && std::string_view(reservationDecision.reason) == "critical_last_resort") {
        trace_transient_reservation_decision(
            transaction,
            commandName,
            requestBytes,
            outReservation.bytesInFlight,
            outReservation.capBytes,
            outReservation.thresholdBytes,
            reservationDecision,
            criticalCurrentFrame,
            0,
            "critical_last_resort");
    }

    outClaim.contextKey = contextKey;
    outClaim.bucketKey = bucketKey;
    outClaim.bytes = requestBytes;
    outClaim.acquired = true;

    snapshot_bucket_state_locked(contextState, bucketEntry, bucketKey, outSnapshot);
    return true;
}

bool acquire_scratch_policy_claim_with_wait(
    const SubmissionTransaction& transaction,
    const char* commandName,
    ScratchWorkClass workClass,
    int width,
    int height,
    std::size_t requestBytes,
    bool criticalCurrentFrame,
    ScratchPolicyClaim& outClaim,
    std::string& outError) {
    outClaim = ScratchPolicyClaim{};
    outError.clear();
    ResourceManagerState& state = global_state();

    const ScratchBucketKey bucketKey = make_scratch_bucket_key(width, height, workClass);
    const int waitBudgetMs = scratch_wait_budget_ms(bucketKey, requestBytes);
    ScratchPolicySnapshot snapshot{};
    ReservationAttemptInfo reservation{};
    int waitedMs = 0;
    while (true) {
        if (try_acquire_scratch_policy_claim(
                transaction,
                commandName,
                transaction.snapshot.deviceContextKey,
                bucketKey,
                requestBytes,
                criticalCurrentFrame,
                outClaim,
                snapshot,
                reservation)) {
            if (waitedMs > 0) {
                state.scratchPolicyWaitEvents.fetch_add(1, std::memory_order_relaxed);
                if (criticalCurrentFrame) {
                    const std::uint64_t waitedMsU64 =
                        static_cast<std::uint64_t>(std::max(waitedMs, 0));
                    state.criticalBuilderWaitEvents.fetch_add(1, std::memory_order_relaxed);
                    state.criticalBuilderWaitTotalMs.fetch_add(waitedMsU64, std::memory_order_relaxed);
                    trace_lane_wait_event(
                        transaction,
                        commandName,
                        PressureLane::Builder,
                        criticalCurrentFrame,
                        waitedMs,
                        "admit_after_wait",
                        "scratch_policy_wait");
                }
                trace_scratch_policy_decision(
                    transaction,
                    commandName,
                    "admit_after_wait",
                    requestBytes,
                    snapshot,
                    waitedMs);
            }
            else if (requestBytes == 0) {
                trace_scratch_policy_decision(
                    transaction,
                    commandName,
                    "reuse_hit",
                    requestBytes,
                    snapshot,
                    waitedMs);
            }
            return true;
        }

        if (waitedMs >= waitBudgetMs) {
            state.scratchPolicyExhaustedEvents.fetch_add(1, std::memory_order_relaxed);
            if (criticalCurrentFrame) {
                const std::uint64_t waitedMsU64 =
                    static_cast<std::uint64_t>(std::max(waitedMs, 0));
                state.criticalBuilderWaitEvents.fetch_add(1, std::memory_order_relaxed);
                state.criticalBuilderWaitTotalMs.fetch_add(waitedMsU64, std::memory_order_relaxed);
                state.criticalLaneStarvationEvents.fetch_add(1, std::memory_order_relaxed);
                trace_lane_wait_event(
                    transaction,
                    commandName,
                    PressureLane::Builder,
                    criticalCurrentFrame,
                    waitedMs,
                    "starved",
                    (reservation.considered && reservation.decision.reason)
                        ? reservation.decision.reason
                        : "scratch_exhausted");
            }
            trace_scratch_policy_decision(
                transaction,
                commandName,
                "exhausted",
                requestBytes,
                snapshot,
                waitedMs);
            if (reservation.considered && !reservation.decision.granted) {
                outError = std::string(kReservationDeferredPrefix)
                    + " command=" + (commandName ? commandName : "unknown")
                    + " work_class=" + to_cstr(workClass)
                    + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
                    + " in_flight_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.bytesInFlight))
                    + " cap_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.capBytes))
                    + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.thresholdBytes))
                    + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
                    + " wait_ms=" + std::to_string(waitedMs)
                    + " wait_budget_ms=" + std::to_string(waitBudgetMs)
                    + " reason=" + (reservation.decision.reason ? reservation.decision.reason : "unspecified");
            }
            else {
                outError = std::string(kScratchExhaustedPrefix)
                    + " command=" + (commandName ? commandName : "unknown")
                    + " work_class=" + to_cstr(workClass)
                    + " bucket_w=" + std::to_string(bucketKey.widthBucket)
                    + " bucket_h=" + std::to_string(bucketKey.heightBucket)
                    + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
                    + " in_flight_bytes=" + std::to_string(static_cast<unsigned long long>(snapshot.inFlightBytes))
                    + " in_flight_sets=" + std::to_string(snapshot.inFlightSets)
                    + " bucket_exhausted=" + std::to_string(snapshot.bucketExhausted)
                    + " bucket_attempts=" + std::to_string(snapshot.bucketAttempts)
                    + " wait_ms=" + std::to_string(waitedMs)
                    + " wait_budget_ms=" + std::to_string(waitBudgetMs);
            }
            return false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(kScratchWaitStepMs));
        waitedMs += kScratchWaitStepMs;
    }
}

bool try_acquire_builder_reservation_claim(
    const SubmissionTransaction& transaction,
    BuilderReservationTier tier,
    std::uint64_t requestBytes,
    bool criticalCurrentFrame,
    BuilderReservationClaim& outClaim,
    ReservationAttemptInfo& outReservation) noexcept {
    outClaim = BuilderReservationClaim{};
    outReservation = ReservationAttemptInfo{};

    if (requestBytes == 0) {
        return true;
    }

    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    BuilderReservationState& state = builder_reservation_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    outReservation.instanceToken = transaction.snapshot.instanceToken.value;

    outReservation.considered = true;
    std::uint64_t& tierTotalBytes = builder_total_bytes_for_tier(state, tier);
    outReservation.bytesInFlight = tierTotalBytes;
    outReservation.capBytes = builder_reservation_cap_bytes(cfg, tier);
    outReservation.thresholdBytes = builder_reservation_threshold_bytes(cfg, tier);

    ReservationDecision reservationDecision{};
    if (requestBytes < outReservation.thresholdBytes) {
        reservationDecision.granted = true;
        reservationDecision.reason = "below_threshold";
    }
    else {
        ReservationInput reservationInput{};
        reservationInput.kind = ReservationKind::BuilderWork;
        reservationInput.requestBytes = requestBytes;
        reservationInput.bytesInFlight = tierTotalBytes;
        reservationInput.capBytes = outReservation.capBytes;
        reservationInput.criticalCurrentFrame = criticalCurrentFrame;
        reservationDecision = classify_reservation(reservationInput);
    }
    outReservation.decision = reservationDecision;

    if (!reservationDecision.granted) {
        return false;
    }

    BuilderReservationContextState& contextState =
        state.byContext[transaction.snapshot.deviceContextKey];
    BuilderReservationContextState::FairnessEntry& fairness =
        contextState.fairnessByInstance[outReservation.instanceToken];
    refill_builder_fairness_tokens(fairness, cfg, monotonic_time_ms());

    bool fairnessGranted = false;
    if (criticalCurrentFrame && fairness.criticalTokens > 0) {
        --fairness.criticalTokens;
        fairnessGranted = true;
    }
    else if (fairness.sharedTokens > 0) {
        --fairness.sharedTokens;
        fairnessGranted = true;
    }
    outReservation.sharedTokens = fairness.sharedTokens;
    outReservation.criticalTokens = fairness.criticalTokens;
    if (!fairnessGranted) {
        ReservationDecision fairnessDecision{};
        fairnessDecision.granted = false;
        fairnessDecision.shouldWait = true;
        fairnessDecision.waitMs = static_cast<std::uint32_t>(kBuilderReservationWaitStepMs);
        fairnessDecision.reason = "fairness_tokens_exhausted";
        outReservation.decision = fairnessDecision;
        return false;
    }

    std::uint64_t& contextTierBytes = builder_context_bytes_for_tier(contextState, tier);
    std::uint64_t nextContextBytes = 0;
    if (add_u64_checked(contextTierBytes, requestBytes, nextContextBytes)) {
        contextTierBytes = nextContextBytes;
    }
    else {
        contextTierBytes = std::numeric_limits<std::uint64_t>::max();
    }
    std::uint64_t nextTotalBytes = 0;
    if (add_u64_checked(tierTotalBytes, requestBytes, nextTotalBytes)) {
        tierTotalBytes = nextTotalBytes;
    }
    else {
        tierTotalBytes = std::numeric_limits<std::uint64_t>::max();
    }
    builder_global_gauge_for_tier(global_state(), tier).store(tierTotalBytes, std::memory_order_relaxed);
    outClaim.contextKey = transaction.snapshot.deviceContextKey;
    outClaim.tier = tier;
    outClaim.bytes = requestBytes;
    outClaim.acquired = true;
    return true;
}

bool acquire_builder_reservation_with_wait(
    const SubmissionTransaction& transaction,
    const char* commandName,
    BuilderReservationTier tier,
    std::uint64_t requestBytes,
    bool criticalCurrentFrame,
    BuilderReservationClaim& outClaim,
    std::string& outError) {
    outClaim = BuilderReservationClaim{};
    outError.clear();

    if (requestBytes == 0) {
        return true;
    }

    ResourceManagerState& state = global_state();
    state.builderReservationRequests.fetch_add(1, std::memory_order_relaxed);

    ReservationAttemptInfo reservation{};
    int waitedMs = 0;
    while (true) {
        if (try_acquire_builder_reservation_claim(
                transaction,
                tier,
                requestBytes,
                criticalCurrentFrame,
                outClaim,
                reservation)) {
            state.builderReservationGranted.fetch_add(1, std::memory_order_relaxed);
            if (waitedMs > 0) {
                state.builderFairnessWaitEvents.fetch_add(1, std::memory_order_relaxed);
                if (criticalCurrentFrame) {
                    const std::uint64_t waitedMsU64 =
                        static_cast<std::uint64_t>(std::max(waitedMs, 0));
                    state.criticalBuilderWaitEvents.fetch_add(1, std::memory_order_relaxed);
                    state.criticalBuilderWaitTotalMs.fetch_add(waitedMsU64, std::memory_order_relaxed);
                    trace_lane_wait_event(
                        transaction,
                        commandName,
                        PressureLane::Builder,
                        criticalCurrentFrame,
                        waitedMs,
                        "admit_after_wait",
                        reservation.decision.reason ? reservation.decision.reason : "waited");
                }
                trace_builder_reservation_decision(
                    transaction,
                    commandName,
                    tier,
                    requestBytes,
                    reservation.bytesInFlight,
                    reservation.capBytes,
                    reservation.thresholdBytes,
                    reservation.decision,
                    criticalCurrentFrame,
                    reservation.instanceToken,
                    reservation.sharedTokens,
                    reservation.criticalTokens,
                    waitedMs,
                    "admit_after_wait");
            }
            else if (reservation.decision.reason &&
                     std::string_view(reservation.decision.reason) == "critical_last_resort") {
                trace_builder_reservation_decision(
                    transaction,
                    commandName,
                    tier,
                    requestBytes,
                    reservation.bytesInFlight,
                    reservation.capBytes,
                    reservation.thresholdBytes,
                    reservation.decision,
                    criticalCurrentFrame,
                    reservation.instanceToken,
                    reservation.sharedTokens,
                    reservation.criticalTokens,
                    waitedMs,
                    "critical_last_resort");
            }
            else if (JTRACE_ENABLED(3)) {
                trace_builder_reservation_decision(
                    transaction,
                    commandName,
                    tier,
                    requestBytes,
                    reservation.bytesInFlight,
                    reservation.capBytes,
                    reservation.thresholdBytes,
                    reservation.decision,
                    criticalCurrentFrame,
                    reservation.instanceToken,
                    reservation.sharedTokens,
                    reservation.criticalTokens,
                    waitedMs,
                    "admitted");
            }
            return true;
        }

        const ReservationDecision& decision = reservation.decision;
        const bool fairnessDeferred =
            (decision.reason && std::string_view(decision.reason) == "fairness_tokens_exhausted");
        if (fairnessDeferred) {
            state.builderFairnessTokenDeferred.fetch_add(1, std::memory_order_relaxed);
        }
        if (!decision.shouldWait) {
            state.builderReservationDenied.fetch_add(1, std::memory_order_relaxed);
            if (criticalCurrentFrame) {
                const std::uint64_t waitedMsU64 =
                    static_cast<std::uint64_t>(std::max(waitedMs, 0));
                state.criticalBuilderWaitEvents.fetch_add(1, std::memory_order_relaxed);
                state.criticalBuilderWaitTotalMs.fetch_add(waitedMsU64, std::memory_order_relaxed);
                state.criticalLaneStarvationEvents.fetch_add(1, std::memory_order_relaxed);
                trace_lane_wait_event(
                    transaction,
                    commandName,
                    PressureLane::Builder,
                    criticalCurrentFrame,
                    waitedMs,
                    "denied",
                    decision.reason ? decision.reason : "reservation_denied");
            }
            trace_builder_reservation_decision(
                transaction,
                commandName,
                tier,
                requestBytes,
                reservation.bytesInFlight,
                reservation.capBytes,
                reservation.thresholdBytes,
                decision,
                criticalCurrentFrame,
                reservation.instanceToken,
                reservation.sharedTokens,
                reservation.criticalTokens,
                waitedMs,
                "denied");
            outError = std::string(kReservationDeferredPrefix)
                + " command=" + (commandName ? commandName : "unknown")
                + " kind=" + to_cstr(ReservationKind::BuilderWork)
                + " tier=" + to_cstr(tier)
                + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
                + " in_flight_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.bytesInFlight))
                + " cap_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.capBytes))
                + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.thresholdBytes))
                + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
                + " wait_ms=" + std::to_string(waitedMs)
                + " wait_budget_ms=" + std::to_string(kBuilderReservationWaitMaxMs)
                + " reason=" + (decision.reason ? decision.reason : "unspecified");
            return false;
        }

        if (waitedMs >= kBuilderReservationWaitMaxMs) {
            state.builderReservationDeferred.fetch_add(1, std::memory_order_relaxed);
            if (criticalCurrentFrame) {
                state.builderReservationBypass.fetch_add(1, std::memory_order_relaxed);
                if (fairnessDeferred) {
                    state.builderFairnessTokenBypass.fetch_add(1, std::memory_order_relaxed);
                }
                const std::uint64_t waitedMsU64 =
                    static_cast<std::uint64_t>(std::max(waitedMs, 0));
                state.criticalBuilderWaitEvents.fetch_add(1, std::memory_order_relaxed);
                state.criticalBuilderWaitTotalMs.fetch_add(waitedMsU64, std::memory_order_relaxed);
                state.criticalLaneStarvationEvents.fetch_add(1, std::memory_order_relaxed);
                trace_lane_wait_event(
                    transaction,
                    commandName,
                    PressureLane::Builder,
                    criticalCurrentFrame,
                    waitedMs,
                    fairnessDeferred ? "bypass_after_starvation" : "deferred_bypass",
                    decision.reason ? decision.reason : "wait_budget_reached");
                trace_builder_reservation_decision(
                    transaction,
                    commandName,
                    tier,
                    requestBytes,
                    reservation.bytesInFlight,
                    reservation.capBytes,
                    reservation.thresholdBytes,
                    decision,
                    criticalCurrentFrame,
                    reservation.instanceToken,
                    reservation.sharedTokens,
                    reservation.criticalTokens,
                    waitedMs,
                    "deferred_bypass_wait_budget");
                return true;
            }
            trace_builder_reservation_decision(
                transaction,
                commandName,
                tier,
                requestBytes,
                reservation.bytesInFlight,
                reservation.capBytes,
                reservation.thresholdBytes,
                decision,
                criticalCurrentFrame,
                reservation.instanceToken,
                reservation.sharedTokens,
                reservation.criticalTokens,
                waitedMs,
                "deferred_wait_budget");
            outError = std::string(kReservationDeferredPrefix)
                + " command=" + (commandName ? commandName : "unknown")
                + " kind=" + to_cstr(ReservationKind::BuilderWork)
                + " tier=" + to_cstr(tier)
                + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
                + " in_flight_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.bytesInFlight))
                + " cap_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.capBytes))
                + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.thresholdBytes))
                + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
                + " wait_ms=" + std::to_string(waitedMs)
                + " wait_budget_ms=" + std::to_string(kBuilderReservationWaitMaxMs)
                + " reason=" + (decision.reason ? decision.reason : "wait_budget_reached");
            return false;
        }

        const int sleepMs = std::max<int>(
            1,
            std::min<int>(
                static_cast<int>(decision.waitMs > 0 ? decision.waitMs : 1),
                kBuilderReservationWaitStepMs));
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
        waitedMs += sleepMs;
    }
}

bool try_acquire_upload_reservation_claim(
    const SubmissionTransaction& transaction,
    std::uint64_t requestBytes,
    bool criticalCurrentFrame,
    UploadReservationClaim& outClaim,
    ReservationAttemptInfo& outReservation) noexcept {
    outClaim = UploadReservationClaim{};
    outReservation = ReservationAttemptInfo{};

    if (requestBytes == 0) {
        return true;
    }

    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    UploadReservationState& state = upload_reservation_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    outReservation.instanceToken = transaction.snapshot.instanceToken.value;

    outReservation.considered = true;
    outReservation.bytesInFlight = state.totalInFlightBytes;
    outReservation.capBytes = upload_reservation_cap_bytes(cfg);
    outReservation.thresholdBytes = upload_reservation_threshold_bytes(cfg);

    ReservationDecision reservationDecision{};
    if (requestBytes < outReservation.thresholdBytes) {
        reservationDecision.granted = true;
        reservationDecision.reason = "below_threshold";
    }
    else {
        ReservationInput reservationInput{};
        reservationInput.kind = ReservationKind::UploadCopy;
        reservationInput.requestBytes = requestBytes;
        reservationInput.bytesInFlight = state.totalInFlightBytes;
        reservationInput.capBytes = outReservation.capBytes;
        reservationInput.criticalCurrentFrame = criticalCurrentFrame;
        reservationDecision = classify_reservation(reservationInput);
    }
    outReservation.decision = reservationDecision;

    if (!reservationDecision.granted) {
        return false;
    }

    UploadReservationContextState& contextState =
        state.byContext[transaction.snapshot.deviceContextKey];
    UploadReservationContextState::FairnessEntry& fairness =
        contextState.fairnessByInstance[outReservation.instanceToken];
    refill_upload_fairness_tokens(fairness, cfg, monotonic_time_ms());

    bool fairnessGranted = false;
    if (criticalCurrentFrame && fairness.criticalTokens > 0) {
        --fairness.criticalTokens;
        fairnessGranted = true;
    }
    else if (fairness.sharedTokens > 0) {
        --fairness.sharedTokens;
        fairnessGranted = true;
    }
    outReservation.sharedTokens = fairness.sharedTokens;
    outReservation.criticalTokens = fairness.criticalTokens;
    if (!fairnessGranted) {
        ReservationDecision fairnessDecision{};
        fairnessDecision.granted = false;
        fairnessDecision.shouldWait = true;
        fairnessDecision.waitMs = static_cast<std::uint32_t>(kUploadReservationWaitStepMs);
        fairnessDecision.reason = "fairness_tokens_exhausted";
        outReservation.decision = fairnessDecision;
        return false;
    }

    std::uint64_t nextContextBytes = 0;
    if (add_u64_checked(contextState.inFlightBytes, requestBytes, nextContextBytes)) {
        contextState.inFlightBytes = nextContextBytes;
    }
    else {
        contextState.inFlightBytes = std::numeric_limits<std::uint64_t>::max();
    }
    std::uint64_t nextTotalBytes = 0;
    if (add_u64_checked(state.totalInFlightBytes, requestBytes, nextTotalBytes)) {
        state.totalInFlightBytes = nextTotalBytes;
    }
    else {
        state.totalInFlightBytes = std::numeric_limits<std::uint64_t>::max();
    }
    global_state().uploadBytesInFlight.store(state.totalInFlightBytes, std::memory_order_relaxed);
    outClaim.contextKey = transaction.snapshot.deviceContextKey;
    outClaim.bytes = requestBytes;
    outClaim.acquired = true;
    return true;
}

bool acquire_upload_reservation_with_wait(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::uint64_t requestBytes,
    bool criticalCurrentFrame,
    UploadReservationClaim& outClaim,
    std::string& outError) {
    outClaim = UploadReservationClaim{};
    outError.clear();

    if (requestBytes == 0) {
        return true;
    }

    ResourceManagerState& state = global_state();
    state.uploadReservationRequests.fetch_add(1, std::memory_order_relaxed);

    ReservationAttemptInfo reservation{};
    int waitedMs = 0;
    while (true) {
        if (try_acquire_upload_reservation_claim(
                transaction,
                requestBytes,
                criticalCurrentFrame,
                outClaim,
                reservation)) {
            state.uploadReservationGranted.fetch_add(1, std::memory_order_relaxed);
            if (waitedMs > 0) {
                state.uploadFairnessWaitEvents.fetch_add(1, std::memory_order_relaxed);
                if (criticalCurrentFrame) {
                    const std::uint64_t waitedMsU64 =
                        static_cast<std::uint64_t>(std::max(waitedMs, 0));
                    state.criticalUploadWaitEvents.fetch_add(1, std::memory_order_relaxed);
                    state.criticalUploadWaitTotalMs.fetch_add(waitedMsU64, std::memory_order_relaxed);
                    trace_lane_wait_event(
                        transaction,
                        commandName,
                        PressureLane::Upload,
                        criticalCurrentFrame,
                        waitedMs,
                        "admit_after_wait",
                        reservation.decision.reason ? reservation.decision.reason : "waited");
                }
                trace_upload_reservation_decision(
                    transaction,
                    commandName,
                    requestBytes,
                    reservation.bytesInFlight,
                    reservation.capBytes,
                    reservation.thresholdBytes,
                    reservation.decision,
                    criticalCurrentFrame,
                    reservation.instanceToken,
                    reservation.sharedTokens,
                    reservation.criticalTokens,
                    waitedMs,
                    "admit_after_wait");
            }
            else if (reservation.decision.reason &&
                     std::string_view(reservation.decision.reason) == "critical_last_resort") {
                trace_upload_reservation_decision(
                    transaction,
                    commandName,
                    requestBytes,
                    reservation.bytesInFlight,
                    reservation.capBytes,
                    reservation.thresholdBytes,
                    reservation.decision,
                    criticalCurrentFrame,
                    reservation.instanceToken,
                    reservation.sharedTokens,
                    reservation.criticalTokens,
                    waitedMs,
                    "critical_last_resort");
            }
            return true;
        }

        const ReservationDecision& decision = reservation.decision;
        const bool fairnessDeferred =
            (decision.reason && std::string_view(decision.reason) == "fairness_tokens_exhausted");
        if (fairnessDeferred) {
            state.uploadFairnessTokenDeferred.fetch_add(1, std::memory_order_relaxed);
        }
        if (!decision.shouldWait) {
            state.uploadReservationDenied.fetch_add(1, std::memory_order_relaxed);
            if (criticalCurrentFrame) {
                const std::uint64_t waitedMsU64 =
                    static_cast<std::uint64_t>(std::max(waitedMs, 0));
                state.criticalUploadWaitEvents.fetch_add(1, std::memory_order_relaxed);
                state.criticalUploadWaitTotalMs.fetch_add(waitedMsU64, std::memory_order_relaxed);
                state.criticalLaneStarvationEvents.fetch_add(1, std::memory_order_relaxed);
                trace_lane_wait_event(
                    transaction,
                    commandName,
                    PressureLane::Upload,
                    criticalCurrentFrame,
                    waitedMs,
                    "denied",
                    decision.reason ? decision.reason : "reservation_denied");
            }
            trace_upload_reservation_decision(
                transaction,
                commandName,
                requestBytes,
                reservation.bytesInFlight,
                reservation.capBytes,
                reservation.thresholdBytes,
                decision,
                criticalCurrentFrame,
                reservation.instanceToken,
                reservation.sharedTokens,
                reservation.criticalTokens,
                waitedMs,
                "denied");
            outError = std::string(kReservationDeferredPrefix)
                + " command=" + (commandName ? commandName : "unknown")
                + " kind=" + to_cstr(ReservationKind::UploadCopy)
                + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
                + " in_flight_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.bytesInFlight))
                + " cap_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.capBytes))
                + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.thresholdBytes))
                + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
                + " wait_ms=" + std::to_string(waitedMs)
                + " wait_budget_ms=" + std::to_string(kUploadReservationWaitMaxMs)
                + " reason=" + (decision.reason ? decision.reason : "unspecified");
            return false;
        }

        if (waitedMs >= kUploadReservationWaitMaxMs) {
            state.uploadReservationDeferred.fetch_add(1, std::memory_order_relaxed);
            state.uploadReservationBypass.fetch_add(1, std::memory_order_relaxed);
            if (fairnessDeferred) {
                state.uploadFairnessTokenBypass.fetch_add(1, std::memory_order_relaxed);
            }
            if (criticalCurrentFrame) {
                const std::uint64_t waitedMsU64 =
                    static_cast<std::uint64_t>(std::max(waitedMs, 0));
                state.criticalUploadWaitEvents.fetch_add(1, std::memory_order_relaxed);
                state.criticalUploadWaitTotalMs.fetch_add(waitedMsU64, std::memory_order_relaxed);
                if (fairnessDeferred) {
                    state.criticalLaneStarvationEvents.fetch_add(1, std::memory_order_relaxed);
                }
                trace_lane_wait_event(
                    transaction,
                    commandName,
                    PressureLane::Upload,
                    criticalCurrentFrame,
                    waitedMs,
                    fairnessDeferred ? "bypass_after_starvation" : "deferred_bypass",
                    decision.reason ? decision.reason : "wait_budget_reached");
            }
            trace_upload_reservation_decision(
                transaction,
                commandName,
                requestBytes,
                reservation.bytesInFlight,
                reservation.capBytes,
                reservation.thresholdBytes,
                decision,
                criticalCurrentFrame,
                reservation.instanceToken,
                reservation.sharedTokens,
                reservation.criticalTokens,
                waitedMs,
                "deferred_bypass_wait_budget");
            return true;
        }

        const int sleepMs = std::max<int>(
            1,
            std::min<int>(
                static_cast<int>(decision.waitMs > 0 ? decision.waitMs : 1),
                kUploadReservationWaitStepMs));
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
        waitedMs += sleepMs;
    }
}


void maybe_publish_manager_memory_snapshot(
    JuicerCuda::Resources& resources,
    bool enabled) noexcept {
    if (!enabled) {
        return;
    }
    publish_manager_memory_snapshot(snapshot_manager_memory(resources));
}

HeadroomTelemetry sample_headroom_telemetry() noexcept {
    HeadroomTelemetry out{};
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    std::size_t freeBytes = 0;
    std::size_t totalBytes = 0;
    if (cudaMemGetInfo(&freeBytes, &totalBytes) == cudaSuccess) {
        (void)totalBytes;
        out.driverFreeBytes = static_cast<std::uint64_t>(
            std::min<std::size_t>(freeBytes, static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())));
        out.effectiveHeadroomBytes = out.driverFreeBytes;
    }
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
    int currentDevice = 0;
    if (cudaGetDevice(&currentDevice) == cudaSuccess) {
        cudaMemPool_t defaultPool = nullptr;
        if (cudaDeviceGetDefaultMemPool(&defaultPool, currentDevice) == cudaSuccess && defaultPool != nullptr) {
            std::size_t poolReservedBytes = 0;
            std::size_t poolUsedBytes = 0;
            const cudaError_t reservedErr = cudaMemPoolGetAttribute(
                defaultPool,
                cudaMemPoolAttrReservedMemCurrent,
                &poolReservedBytes);
            const cudaError_t usedErr = cudaMemPoolGetAttribute(
                defaultPool,
                cudaMemPoolAttrUsedMemCurrent,
                &poolUsedBytes);
            if (reservedErr == cudaSuccess && usedErr == cudaSuccess) {
                out.poolTelemetryAvailable = true;
                out.source = HeadroomSource::AllocatorPool;
                out.allocatorPoolReservedBytes = static_cast<std::uint64_t>(
                    std::min<std::size_t>(
                        poolReservedBytes,
                        static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())));
                out.allocatorPoolUsedBytes = static_cast<std::uint64_t>(
                    std::min<std::size_t>(
                        poolUsedBytes,
                        static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())));
                if (out.allocatorPoolUsedBytes > out.allocatorPoolReservedBytes) {
                    out.allocatorPoolUsedBytes = out.allocatorPoolReservedBytes;
                }

                const std::uint64_t poolFreeBytes =
                    out.allocatorPoolReservedBytes - out.allocatorPoolUsedBytes;
                std::uint64_t combinedHeadroom = out.driverFreeBytes;
                if (!add_u64_checked(combinedHeadroom, poolFreeBytes, combinedHeadroom)) {
                    combinedHeadroom = std::numeric_limits<std::uint64_t>::max();
                }
                out.effectiveHeadroomBytes = combinedHeadroom;
            }
        }
    }
#endif
#endif
    return out;
}

std::uint64_t pressure_total_bytes(const PressureInput& input) noexcept {
    std::uint64_t total = input.managerResidentBytes;
    std::uint64_t next = 0;
    if (!add_u64_checked(total, input.retirePendingBytes, next)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    total = next;
    if (!add_u64_checked(total, input.transientNonManagerBytes, next)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return next;
}

bool reserve_crossed(const PressureInput& input) noexcept {
    if (input.softTargetBytes == 0) {
        return false;
    }
    return pressure_total_bytes(input) > input.softTargetBytes;
}

std::uint64_t compute_effective_reserve_target_bytes(
    const ResourceManagerConfigEffective& cfg,
    std::uint64_t transientBytes) noexcept {
    std::uint64_t targetBytes = cfg.managerReserveBytes;
    std::uint64_t transientBoundBytes = transientBytes;
    if (!add_u64_checked(transientBoundBytes, cfg.reserveSafetyMarginBytes, transientBoundBytes)) {
        transientBoundBytes = std::numeric_limits<std::uint64_t>::max();
    }
    targetBytes = std::max<std::uint64_t>(targetBytes, transientBoundBytes);
    if (cfg.managerSoftTargetBytes > 0) {
        targetBytes = std::min<std::uint64_t>(targetBytes, cfg.managerSoftTargetBytes);
    }
    return targetBytes;
}

std::uint64_t step_effective_reserve_bytes(
    const ResourceManagerConfigEffective& cfg,
    std::uint64_t currentBytes,
    std::uint64_t targetBytes) noexcept {
    if (currentBytes == targetBytes) {
        return currentBytes;
    }

    const std::uint64_t stepUp = std::max<std::uint64_t>(1u, cfg.reserveAdaptUpStepBytes);
    const std::uint64_t stepDown = std::max<std::uint64_t>(1u, cfg.reserveAdaptDownStepBytes);
    if (targetBytes > currentBytes) {
        const std::uint64_t delta = targetBytes - currentBytes;
        const std::uint64_t step = std::min<std::uint64_t>(stepUp, delta);
        std::uint64_t next = currentBytes;
        if (!add_u64_checked(next, step, next)) {
            next = targetBytes;
        }
        return next;
    }

    const std::uint64_t delta = currentBytes - targetBytes;
    const std::uint64_t step = std::min<std::uint64_t>(stepDown, delta);
    return currentBytes - step;
}

std::uint64_t active_burst_cap_bytes(
    const ResourceManagerConfigEffective& cfg,
    std::uint64_t softTargetBytes) noexcept {
    std::uint64_t pctCapBytes = 0;
    if (softTargetBytes > 0 && cfg.maxActiveBurstPctOfTarget > 0) {
        std::uint64_t weighted = 0;
        if (!mul_u64_checked(
                softTargetBytes,
                static_cast<std::uint64_t>(cfg.maxActiveBurstPctOfTarget),
                weighted)) {
            weighted = std::numeric_limits<std::uint64_t>::max();
        }
        pctCapBytes = weighted / 100ull;
    }

    if (cfg.maxActiveBurstBytes == 0) {
        return pctCapBytes;
    }
    if (pctCapBytes == 0) {
        return cfg.maxActiveBurstBytes;
    }
    return std::min<std::uint64_t>(cfg.maxActiveBurstBytes, pctCapBytes);
}

std::uint32_t decay_burst_debt_pct(
    std::uint32_t currentDebtPct,
    std::uint64_t elapsedMs,
    std::uint32_t halfLifeMs) noexcept {
    if (currentDebtPct == 0 || elapsedMs == 0 || halfLifeMs == 0) {
        return currentDebtPct;
    }

    const double exponent =
        -static_cast<double>(elapsedMs) / static_cast<double>(halfLifeMs);
    const double decayed = static_cast<double>(currentDebtPct) * std::exp2(exponent);
    if (!std::isfinite(decayed) || decayed <= 0.0) {
        return 0;
    }

    const long long rounded = std::llround(decayed);
    if (rounded <= 0) {
        return 0;
    }
    if (rounded >= 100) {
        return 100;
    }
    return static_cast<std::uint32_t>(rounded);
}

std::uint64_t graph_large_entry_threshold_bytes(const ResourceManagerConfigEffective& cfg) noexcept {
    std::uint64_t thresholdBytes = std::max<std::uint64_t>(
        1ull,
        cfg.graphLargeEntryThresholdBytes > 0
            ? cfg.graphLargeEntryThresholdBytes
            : kGraphLargeEntryThresholdDefaultBytes);
    if (cfg.managerSoftTargetBytes > 0) {
        const std::uint64_t pctThresholdBytes = cfg.managerSoftTargetBytes / 10ull;
        if (pctThresholdBytes > 0) {
            thresholdBytes = std::min<std::uint64_t>(thresholdBytes, pctThresholdBytes);
        }
    }
    return std::max<std::uint64_t>(1ull, thresholdBytes);
}

std::uint64_t graph_large_entry_quarantine_cap_bytes(const ResourceManagerConfigEffective& cfg) noexcept {
    std::uint64_t capBytes = cfg.graphLargeEntryQuarantineMaxBytes > 0
        ? cfg.graphLargeEntryQuarantineMaxBytes
        : kGraphLargeEntryQuarantineMaxBytesDefault;
    if (cfg.managerSoftTargetBytes > 0) {
        std::uint64_t weighted = 0;
        if (mul_u64_checked(cfg.managerSoftTargetBytes, 15ull, weighted)) {
            const std::uint64_t pctCapBytes = weighted / 100ull;
            if (pctCapBytes > 0) {
                capBytes = std::min<std::uint64_t>(capBytes, pctCapBytes);
            }
        }
    }
    return capBytes;
}

std::uint32_t graph_large_entry_quarantine_cap_entries(const ResourceManagerConfigEffective& cfg) noexcept {
    return std::max<std::uint32_t>(
        1u,
        cfg.graphLargeEntryQuarantineMaxEntries > 0
            ? cfg.graphLargeEntryQuarantineMaxEntries
            : kGraphLargeEntryQuarantineMaxEntriesDefault);
}

std::uint32_t pressure_poll_interval_ms_for_state(
    const ResourceManagerConfigEffective& cfg,
    PressureState state) noexcept {
    const std::uint32_t fallback = std::max<std::uint32_t>(1u, cfg.pressurePollIntervalMs);
    const std::uint32_t normal = std::max<std::uint32_t>(1u, cfg.pressurePollIntervalMsNormal);
    const std::uint32_t critical = std::max<std::uint32_t>(
        1u,
        std::min<std::uint32_t>(cfg.pressurePollIntervalMsCritical, normal));
    switch (state) {
    case PressureState::Normal:
        return normal;
    case PressureState::Constrained:
        return std::max<std::uint32_t>(critical, (normal + critical) / 2u);
    case PressureState::Critical:
    case PressureState::Emergency:
        return critical;
    default:
        return fallback;
    }
}

std::size_t estimate_optics_growth_bytes(
    JuicerCuda::Resources& resources,
    int width,
    int height,
    bool needBlurredScratch,
    bool needAuxScratch,
    bool needGrainScratch,
    bool needGrainSharedScratch,
    bool needGateMask) noexcept {
    const std::size_t planeBytes = plane_bytes_for_extent(width, height);
    if (planeBytes == 0 || planeBytes == std::numeric_limits<std::size_t>::max()) {
        return planeBytes;
    }

    std::lock_guard<std::mutex> lock(resources.m);
    const auto& scratch = resources.scannerScratch;
    const bool dimsMatch = (scratch.width == width && scratch.height == height);
    const bool haveBase = scratch.rgbR && scratch.rgbG && scratch.rgbB;
    const bool fullRebuild = !dimsMatch || !haveBase;

    std::size_t estimate = 0;
    auto addPlane = [&](std::size_t multiplier = 1) {
        for (std::size_t i = 0; i < multiplier; ++i) {
            std::size_t next = 0;
            if (!add_bytes_checked(estimate, planeBytes, next)) {
                estimate = std::numeric_limits<std::size_t>::max();
                return;
            }
            estimate = next;
        }
    };

    if (fullRebuild) {
        addPlane(3); // rgbR/rgbG/rgbB
    }

    const bool sharedTmpMatch = resources.sharedTmpPlane &&
        resources.sharedTmpWidth == width &&
        resources.sharedTmpHeight == height;
    if (!sharedTmpMatch) {
        addPlane(); // shared tmp plane
    }

    if (needBlurredScratch && (fullRebuild || !scratch.blurred)) {
        addPlane();
    }
    if (needAuxScratch && (fullRebuild || !scratch.aux)) {
        addPlane();
    }
    if (needGrainScratch) {
        if (fullRebuild || !scratch.grainTmp) {
            addPlane();
        }
        if (fullRebuild || !scratch.grainTmpMid) {
            addPlane();
        }
        if (fullRebuild || !scratch.grainTmpCoarse) {
            addPlane();
        }
    }
    if (needGrainSharedScratch && (fullRebuild || !scratch.grainTmpShared)) {
        addPlane();
    }
    if (needGateMask) {
        const bool gateDimsMatch = (scratch.gateWidth == width && scratch.gateHeight == height);
        if (fullRebuild || !scratch.gateMask || !gateDimsMatch) {
            addPlane();
        }
    }

    return estimate;
}

std::size_t estimate_spatial_dir_growth_bytes(
    JuicerCuda::Resources& resources,
    int width,
    int height) noexcept {
    const std::size_t planeBytes = plane_bytes_for_extent(width, height);
    if (planeBytes == 0 || planeBytes == std::numeric_limits<std::size_t>::max()) {
        return planeBytes;
    }

    std::lock_guard<std::mutex> lock(resources.m);
    const auto& scratch = resources.spatialDirScratch;
    const bool haveBase = (scratch.width == width && scratch.height == height && scratch.corrY && scratch.corrM && scratch.corrC);
    const bool sharedTmpMatch = resources.sharedTmpPlane &&
        resources.sharedTmpWidth == width &&
        resources.sharedTmpHeight == height;

    std::size_t estimate = 0;
    if (!haveBase) {
        std::size_t next = 0;
        if (!add_bytes_checked(estimate, planeBytes, next)) {
            return std::numeric_limits<std::size_t>::max();
        }
        estimate = next;
        if (!add_bytes_checked(estimate, planeBytes, next)) {
            return std::numeric_limits<std::size_t>::max();
        }
        estimate = next;
        if (!add_bytes_checked(estimate, planeBytes, next)) {
            return std::numeric_limits<std::size_t>::max();
        }
        estimate = next;
    }
    if (!sharedTmpMatch) {
        std::size_t next = 0;
        if (!add_bytes_checked(estimate, planeBytes, next)) {
            return std::numeric_limits<std::size_t>::max();
        }
        estimate = next;
    }
    return estimate;
}

void trim_large_frame_quarantine_decay_locked(
    ScratchContextState& contextState,
    std::uint64_t nowMs) noexcept {
    std::size_t index = 0;
    std::uint64_t removedCount = 0;
    while (index < contextState.largeFrameQuarantine.size()) {
        const ScratchQuarantineEntry& entry = contextState.largeFrameQuarantine[index];
        const std::uint64_t ageMs = (nowMs > entry.touchedMs) ? (nowMs - entry.touchedMs) : 0;
        if (ageMs < kLargeFrameQuarantineDecayMs) {
            ++index;
            continue;
        }
        if (contextState.largeFrameQuarantineBytes >= entry.bytes) {
            contextState.largeFrameQuarantineBytes -= entry.bytes;
        }
        else {
            contextState.largeFrameQuarantineBytes = 0;
        }
        contextState.largeFrameQuarantine.erase(
            contextState.largeFrameQuarantine.begin() + static_cast<std::ptrdiff_t>(index));
        ++removedCount;
    }
    if (removedCount > 0) {
        global_state().scratchLargeQuarantineDecayEvents.fetch_add(removedCount, std::memory_order_relaxed);
    }
}

void trim_large_frame_quarantine_caps_locked(
    ScratchContextState& contextState) noexcept {
    std::uint64_t trimmedCount = 0;
    while ((contextState.largeFrameQuarantineBytes > kLargeFrameQuarantineMaxBytes) ||
           (contextState.largeFrameQuarantine.size() > kLargeFrameQuarantineMaxEntries)) {
        if (contextState.largeFrameQuarantine.empty()) {
            contextState.largeFrameQuarantineBytes = 0;
            break;
        }

        std::size_t victimIndex = 0;
        ScratchQuarantineEntry victim = contextState.largeFrameQuarantine[0];
        for (std::size_t i = 1; i < contextState.largeFrameQuarantine.size(); ++i) {
            const ScratchQuarantineEntry& candidate = contextState.largeFrameQuarantine[i];
            const bool older = (candidate.touchedMs < victim.touchedMs) ||
                ((candidate.touchedMs == victim.touchedMs) && (candidate.sequence < victim.sequence));
            if (older) {
                victim = candidate;
                victimIndex = i;
            }
        }

        if (contextState.largeFrameQuarantineBytes >= victim.bytes) {
            contextState.largeFrameQuarantineBytes -= victim.bytes;
        }
        else {
            contextState.largeFrameQuarantineBytes = 0;
        }
        contextState.largeFrameQuarantine.erase(
            contextState.largeFrameQuarantine.begin() + static_cast<std::ptrdiff_t>(victimIndex));
        ++trimmedCount;
    }

    if (trimmedCount > 0) {
        global_state().scratchLargeQuarantineTrimEvents.fetch_add(trimmedCount, std::memory_order_relaxed);
    }
}

std::uint64_t trim_large_frame_quarantine_for_context(
    const DeviceContextKey& contextKey) noexcept {
    ScratchPolicyState& state = scratch_policy_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto contextIt = state.byContext.find(contextKey);
    if (contextIt == state.byContext.end()) {
        return 0;
    }
    ScratchContextState& contextState = contextIt->second;
    const std::uint64_t removedEntries = static_cast<std::uint64_t>(contextState.largeFrameQuarantine.size());
    if (removedEntries == 0) {
        return 0;
    }
    contextState.largeFrameQuarantine.clear();
    contextState.largeFrameQuarantineBytes = 0;
    return removedEntries;
}

void snapshot_bucket_state_locked(
    const ScratchContextState& contextState,
    const ScratchBucketEntry& bucketEntry,
    const ScratchBucketKey& bucketKey,
    ScratchPolicySnapshot& outSnapshot) noexcept {
    outSnapshot.inFlightBytes = bucketEntry.inFlightBytes;
    outSnapshot.inFlightSets = bucketEntry.inFlightSets;
    outSnapshot.quarantineBytes = contextState.largeFrameQuarantineBytes;
    outSnapshot.quarantineEntries = static_cast<std::uint32_t>(contextState.largeFrameQuarantine.size());
    outSnapshot.bucketAttempts = bucketEntry.attemptCount;
    outSnapshot.bucketExhausted = bucketEntry.exhaustedCount;
    outSnapshot.bucketAllocGrowth = bucketEntry.allocGrowthEvents;
    outSnapshot.bucketReuse = bucketEntry.reuseEvents;
    outSnapshot.starvationLatched = bucketEntry.starvationLatched;
    outSnapshot.bucketKey = bucketKey;
}

void release_scratch_policy_claim(ScratchPolicyClaim& claim) noexcept {
    if (!claim.acquired) {
        return;
    }

    ScratchPolicyState& state = scratch_policy_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto contextIt = state.byContext.find(claim.contextKey);
    if (contextIt == state.byContext.end()) {
        claim = ScratchPolicyClaim{};
        return;
    }

    ScratchContextState& contextState = contextIt->second;
    auto bucketIt = contextState.buckets.find(claim.bucketKey);
    std::size_t releasedBytes = 0;
    if (bucketIt != contextState.buckets.end()) {
        ScratchBucketEntry& bucketEntry = bucketIt->second;
        if (bucketEntry.inFlightSets > 0) {
            --bucketEntry.inFlightSets;
        }
        if (bucketEntry.inFlightBytes >= claim.bytes) {
            bucketEntry.inFlightBytes -= claim.bytes;
            releasedBytes = claim.bytes;
        }
        else {
            releasedBytes = bucketEntry.inFlightBytes;
            bucketEntry.inFlightBytes = 0;
        }
    }
    if (contextState.totalInFlightBytes >= releasedBytes) {
        contextState.totalInFlightBytes -= releasedBytes;
    }
    else {
        contextState.totalInFlightBytes = 0;
    }
    if (state.totalInFlightBytes >= static_cast<std::uint64_t>(releasedBytes)) {
        state.totalInFlightBytes -= static_cast<std::uint64_t>(releasedBytes);
    }
    else {
        state.totalInFlightBytes = 0;
    }
    global_state().transientNonManagerBytes.store(state.totalInFlightBytes, std::memory_order_relaxed);

    const std::uint64_t nowMs = monotonic_time_ms();
    trim_large_frame_quarantine_decay_locked(contextState, nowMs);

    if (claim.bucketKey.largeFrame && claim.bytes > 0) {
        ScratchQuarantineEntry entry{};
        entry.key = claim.bucketKey;
        entry.bytes = claim.bytes;
        entry.touchedMs = nowMs;
        entry.sequence = contextState.nextQuarantineSequence++;
        contextState.largeFrameQuarantine.push_back(entry);
        contextState.largeFrameQuarantineBytes += claim.bytes;
        trim_large_frame_quarantine_caps_locked(contextState);
    }

    claim = ScratchPolicyClaim{};
}

class ScratchPolicyGuard {
public:
    explicit ScratchPolicyGuard(ScratchPolicyClaim&& claim) noexcept
        : _claim(std::move(claim)) {
    }

    ~ScratchPolicyGuard() noexcept {
        release_scratch_policy_claim(_claim);
    }

    ScratchPolicyGuard(const ScratchPolicyGuard&) = delete;
    ScratchPolicyGuard& operator=(const ScratchPolicyGuard&) = delete;

private:
    ScratchPolicyClaim _claim{};
};

void release_builder_reservation_claim(BuilderReservationClaim& claim) noexcept {
    if (!claim.acquired) {
        return;
    }

    BuilderReservationState& state = builder_reservation_state();
    ResourceManagerState& managerState = global_state();
    std::lock_guard<std::mutex> lock(state.mutex);

    std::uint64_t& tierTotalBytes = builder_total_bytes_for_tier(state, claim.tier);
    auto contextIt = state.byContext.find(claim.contextKey);
    if (contextIt == state.byContext.end()) {
        if (tierTotalBytes >= claim.bytes) {
            tierTotalBytes -= claim.bytes;
        }
        else {
            tierTotalBytes = 0;
        }
        builder_global_gauge_for_tier(managerState, claim.tier).store(tierTotalBytes, std::memory_order_relaxed);
        claim = BuilderReservationClaim{};
        return;
    }

    BuilderReservationContextState& contextState = contextIt->second;
    std::uint64_t& contextTierBytes = builder_context_bytes_for_tier(contextState, claim.tier);
    if (contextTierBytes >= claim.bytes) {
        contextTierBytes -= claim.bytes;
    }
    else {
        contextTierBytes = 0;
    }
    if (tierTotalBytes >= claim.bytes) {
        tierTotalBytes -= claim.bytes;
    }
    else {
        tierTotalBytes = 0;
    }

    if (!builder_context_has_inflight(contextState)) {
        state.byContext.erase(contextIt);
    }
    builder_global_gauge_for_tier(managerState, claim.tier).store(tierTotalBytes, std::memory_order_relaxed);
    claim = BuilderReservationClaim{};
}

class BuilderReservationGuard {
public:
    explicit BuilderReservationGuard(BuilderReservationClaim&& claim) noexcept
        : _claim(std::move(claim)) {
    }

    ~BuilderReservationGuard() noexcept {
        release_builder_reservation_claim(_claim);
    }

    BuilderReservationGuard(const BuilderReservationGuard&) = delete;
    BuilderReservationGuard& operator=(const BuilderReservationGuard&) = delete;

private:
    BuilderReservationClaim _claim{};
};

void release_upload_reservation_claim(UploadReservationClaim& claim) noexcept {
    if (!claim.acquired) {
        return;
    }

    UploadReservationState& state = upload_reservation_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto contextIt = state.byContext.find(claim.contextKey);
    if (contextIt == state.byContext.end()) {
        if (state.totalInFlightBytes >= claim.bytes) {
            state.totalInFlightBytes -= claim.bytes;
        }
        else {
            state.totalInFlightBytes = 0;
        }
        claim = UploadReservationClaim{};
        global_state().uploadBytesInFlight.store(state.totalInFlightBytes, std::memory_order_relaxed);
        return;
    }

    UploadReservationContextState& contextState = contextIt->second;
    if (contextState.inFlightBytes >= claim.bytes) {
        contextState.inFlightBytes -= claim.bytes;
    }
    else {
        contextState.inFlightBytes = 0;
    }
    if (state.totalInFlightBytes >= claim.bytes) {
        state.totalInFlightBytes -= claim.bytes;
    }
    else {
        state.totalInFlightBytes = 0;
    }
    if (contextState.inFlightBytes == 0) {
        state.byContext.erase(contextIt);
    }
    global_state().uploadBytesInFlight.store(state.totalInFlightBytes, std::memory_order_relaxed);
    claim = UploadReservationClaim{};
}

class UploadReservationGuard {
public:
    explicit UploadReservationGuard(UploadReservationClaim&& claim) noexcept
        : _claim(std::move(claim)) {
    }

    ~UploadReservationGuard() noexcept {
        release_upload_reservation_claim(_claim);
    }

    UploadReservationGuard(const UploadReservationGuard&) = delete;
    UploadReservationGuard& operator=(const UploadReservationGuard&) = delete;

private:
    UploadReservationClaim _claim{};
};

void trace_scratch_policy_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const char* result,
    std::size_t requestBytes,
    const ScratchPolicySnapshot& snapshot,
    int waitMs) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::uint64_t churnDenom = snapshot.bucketAllocGrowth + snapshot.bucketReuse;
    const std::uint64_t churnRatioMilli = (churnDenom == 0)
        ? 0
        : (snapshot.bucketAllocGrowth * 1000ull) / churnDenom;
    const std::size_t effectiveTempCap = effective_temp_scratch_bytes_cap(requestBytes);

    const std::string msg = std::string("event=scratch_policy")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " result=" + (result ? result : "unknown")
        + " work_class=" + to_cstr(snapshot.bucketKey.workClass)
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " in_flight_bytes=" + std::to_string(static_cast<unsigned long long>(snapshot.inFlightBytes))
        + " in_flight_sets=" + std::to_string(snapshot.inFlightSets)
        + " bucket_w=" + std::to_string(snapshot.bucketKey.widthBucket)
        + " bucket_h=" + std::to_string(snapshot.bucketKey.heightBucket)
        + " bucket_step_px=" + std::to_string(snapshot.bucketKey.bucketStepPx)
        + " large_frame=" + std::to_string(snapshot.bucketKey.largeFrame ? 1 : 0)
        + " bucket_attempts=" + std::to_string(snapshot.bucketAttempts)
        + " bucket_exhausted=" + std::to_string(snapshot.bucketExhausted)
        + " bucket_alloc_growth=" + std::to_string(snapshot.bucketAllocGrowth)
        + " bucket_reuse=" + std::to_string(snapshot.bucketReuse)
        + " bucket_starvation_latched=" + std::to_string(snapshot.starvationLatched ? 1 : 0)
        + " scratch_churn_ratio_milli=" + std::to_string(churnRatioMilli)
        + " quarantine_bytes=" + std::to_string(static_cast<unsigned long long>(snapshot.quarantineBytes))
        + " quarantine_entries=" + std::to_string(snapshot.quarantineEntries)
        + " wait_ms=" + std::to_string(waitMs)
        + " max_temp_sets=" + std::to_string(kMaxTempScratchSets)
        + " max_temp_bytes=" + std::to_string(static_cast<unsigned long long>(kMaxTempScratchBytes))
        + " effective_max_temp_bytes=" + std::to_string(static_cast<unsigned long long>(effectiveTempCap))
        + " quarantine_max_bytes=" + std::to_string(static_cast<unsigned long long>(kLargeFrameQuarantineMaxBytes))
        + " quarantine_max_entries=" + std::to_string(static_cast<unsigned long long>(kLargeFrameQuarantineMaxEntries));
    JTRACE("MSACQ", msg);
}

void trace_transient_reservation_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::size_t requestBytes,
    std::uint64_t bytesInFlight,
    std::uint64_t capBytes,
    std::uint64_t thresholdBytes,
    const ReservationDecision& decision,
    bool criticalCurrentFrame,
    int waitMs,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=transient_reservation")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " kind=" + to_cstr(ReservationKind::TransientNonManager)
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " bytes_in_flight=" + std::to_string(static_cast<unsigned long long>(bytesInFlight))
        + " cap_bytes=" + std::to_string(static_cast<unsigned long long>(capBytes))
        + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(thresholdBytes))
        + " granted=" + std::to_string(decision.granted ? 1 : 0)
        + " should_wait=" + std::to_string(decision.shouldWait ? 1 : 0)
        + " wait_ms=" + std::to_string(waitMs)
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + " decision_reason=" + (decision.reason ? decision.reason : "unspecified")
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSTRS", msg);
}

void trace_upload_reservation_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::uint64_t requestBytes,
    std::uint64_t bytesInFlight,
    std::uint64_t capBytes,
    std::uint64_t thresholdBytes,
    const ReservationDecision& decision,
    bool criticalCurrentFrame,
    std::uint64_t instanceToken,
    std::uint32_t sharedTokens,
    std::uint32_t criticalTokens,
    int waitMs,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=upload_reservation")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " kind=" + to_cstr(ReservationKind::UploadCopy)
        + " instance_token=" + std::to_string(static_cast<unsigned long long>(instanceToken))
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " bytes_in_flight=" + std::to_string(static_cast<unsigned long long>(bytesInFlight))
        + " cap_bytes=" + std::to_string(static_cast<unsigned long long>(capBytes))
        + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(thresholdBytes))
        + " shared_tokens=" + std::to_string(static_cast<unsigned long long>(sharedTokens))
        + " critical_tokens=" + std::to_string(static_cast<unsigned long long>(criticalTokens))
        + " granted=" + std::to_string(decision.granted ? 1 : 0)
        + " should_wait=" + std::to_string(decision.shouldWait ? 1 : 0)
        + " wait_ms=" + std::to_string(waitMs)
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + " decision_reason=" + (decision.reason ? decision.reason : "unspecified")
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSUPL", msg);
}

void trace_builder_reservation_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    BuilderReservationTier tier,
    std::uint64_t requestBytes,
    std::uint64_t bytesInFlight,
    std::uint64_t capBytes,
    std::uint64_t thresholdBytes,
    const ReservationDecision& decision,
    bool criticalCurrentFrame,
    std::uint64_t instanceToken,
    std::uint32_t sharedTokens,
    std::uint32_t criticalTokens,
    int waitMs,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=builder_reservation")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " kind=" + to_cstr(ReservationKind::BuilderWork)
        + " tier=" + to_cstr(tier)
        + " instance_token=" + std::to_string(static_cast<unsigned long long>(instanceToken))
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " bytes_in_flight=" + std::to_string(static_cast<unsigned long long>(bytesInFlight))
        + " cap_bytes=" + std::to_string(static_cast<unsigned long long>(capBytes))
        + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(thresholdBytes))
        + " shared_tokens=" + std::to_string(static_cast<unsigned long long>(sharedTokens))
        + " critical_tokens=" + std::to_string(static_cast<unsigned long long>(criticalTokens))
        + " granted=" + std::to_string(decision.granted ? 1 : 0)
        + " should_wait=" + std::to_string(decision.shouldWait ? 1 : 0)
        + " wait_ms=" + std::to_string(waitMs)
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + " decision_reason=" + (decision.reason ? decision.reason : "unspecified")
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSBPR", msg);
}

void trace_tier_circuit_event(
    const SubmissionTransaction& transaction,
    const char* commandName,
    ResourceTier tier,
    TierCircuitState previousState,
    TierCircuitState state,
    bool transition,
    bool allowed,
    bool probe,
    std::uint32_t windowErrors,
    std::uint32_t threshold,
    std::uint32_t windowMs,
    std::uint32_t openMs,
    std::uint64_t openRemainingMs,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=tier_circuit")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " tier=" + to_cstr(tier)
        + " prev_state=" + to_cstr(previousState)
        + " state=" + to_cstr(state)
        + " transition=" + std::to_string(transition ? 1 : 0)
        + " allowed=" + std::to_string(allowed ? 1 : 0)
        + " probe=" + std::to_string(probe ? 1 : 0)
        + " window_errors=" + std::to_string(windowErrors)
        + " threshold=" + std::to_string(threshold)
        + " window_ms=" + std::to_string(windowMs)
        + " open_ms=" + std::to_string(openMs)
        + " open_remaining_ms=" + std::to_string(static_cast<unsigned long long>(openRemainingMs))
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSCB", msg);
}

void trace_tier_budget_event(
    const SubmissionTransaction& transaction,
    const char* commandName,
    PressureLane lane,
    PressureState pressureState,
    const TierBudgetSnapshot& tierBudget,
    std::size_t requestBytes,
    bool criticalCurrentFrame,
    bool requestReclaimPass,
    std::uint64_t scratchTrimmedEntries,
    std::uint64_t graphEvictedEntries,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=tier_budget")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " lane=" + to_cstr(lane)
        + " state=" + to_cstr(pressureState)
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " request_reclaim_pass=" + std::to_string(requestReclaimPass ? 1 : 0)
        + " any_over_target=" + std::to_string(tierBudget.anyOverTarget ? 1 : 0)
        + " dominant_tier=" + to_cstr(tierBudget.dominantOverTargetTier)
        + " dominant_over_target_bytes=" + std::to_string(
            static_cast<unsigned long long>(tierBudget.dominantOverTargetBytes))
        + " immutable_active_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.activeBytes, ResourceTier::Immutable)))
        + " immutable_target_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.targetBytes, ResourceTier::Immutable)))
        + " immutable_over_target_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.overTargetBytes, ResourceTier::Immutable)))
        + " immutable_reclaimable_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.reclaimableBytes, ResourceTier::Immutable)))
        + " lut_active_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.activeBytes, ResourceTier::Lut)))
        + " lut_target_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.targetBytes, ResourceTier::Lut)))
        + " lut_over_target_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.overTargetBytes, ResourceTier::Lut)))
        + " lut_reclaimable_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.reclaimableBytes, ResourceTier::Lut)))
        + " scratch_active_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.activeBytes, ResourceTier::Scratch)))
        + " scratch_target_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.targetBytes, ResourceTier::Scratch)))
        + " scratch_over_target_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.overTargetBytes, ResourceTier::Scratch)))
        + " scratch_reclaimable_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.reclaimableBytes, ResourceTier::Scratch)))
        + " graph_active_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.activeBytes, ResourceTier::Graph)))
        + " graph_target_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.targetBytes, ResourceTier::Graph)))
        + " graph_over_target_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.overTargetBytes, ResourceTier::Graph)))
        + " graph_reclaimable_bytes=" + std::to_string(
            static_cast<unsigned long long>(tier_bytes_at(tierBudget.reclaimableBytes, ResourceTier::Graph)))
        + " total_active_bytes=" + std::to_string(static_cast<unsigned long long>(tierBudget.totalActiveBytes))
        + " total_reclaimable_bytes=" + std::to_string(
            static_cast<unsigned long long>(tierBudget.totalReclaimableBytes))
        + " scratch_trimmed_entries=" + std::to_string(static_cast<unsigned long long>(scratchTrimmedEntries))
        + " graph_evicted_entries=" + std::to_string(static_cast<unsigned long long>(graphEvictedEntries))
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSTGT", msg);
}

void trace_effective_reserve_event(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::uint64_t reserveBaseBytes,
    std::uint64_t reserveBeforeBytes,
    std::uint64_t reserveTargetBytes,
    std::uint64_t reserveAfterBytes,
    std::uint64_t transientNonManagerBytes,
    bool updated,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    const std::string msg = std::string("event=effective_reserve")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " reserve_base_bytes=" + std::to_string(static_cast<unsigned long long>(reserveBaseBytes))
        + " reserve_before_bytes=" + std::to_string(static_cast<unsigned long long>(reserveBeforeBytes))
        + " reserve_target_bytes=" + std::to_string(static_cast<unsigned long long>(reserveTargetBytes))
        + " reserve_after_bytes=" + std::to_string(static_cast<unsigned long long>(reserveAfterBytes))
        + " transient_non_manager_bytes=" + std::to_string(
            static_cast<unsigned long long>(transientNonManagerBytes))
        + " reserve_safety_margin_bytes=" + std::to_string(
            static_cast<unsigned long long>(cfg.reserveSafetyMarginBytes))
        + " reserve_step_up_bytes=" + std::to_string(
            static_cast<unsigned long long>(cfg.reserveAdaptUpStepBytes))
        + " reserve_step_down_bytes=" + std::to_string(
            static_cast<unsigned long long>(cfg.reserveAdaptDownStepBytes))
        + " updated=" + std::to_string(updated ? 1 : 0)
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSRSV", msg);
}

void trace_opportunistic_freeze_event(
    const SubmissionTransaction& transaction,
    const char* commandName,
    PressureState pressureState,
    std::uint64_t effectiveHeadroomBytes,
    std::uint64_t effectiveReserveBytes,
    bool frozen,
    bool allowed,
    bool criticalCurrentFrame,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=opportunistic_freeze")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " state=" + to_cstr(pressureState)
        + " effective_headroom_bytes=" + std::to_string(
            static_cast<unsigned long long>(effectiveHeadroomBytes))
        + " effective_reserve_bytes=" + std::to_string(
            static_cast<unsigned long long>(effectiveReserveBytes))
        + " frozen=" + std::to_string(frozen ? 1 : 0)
        + " allowed=" + std::to_string(allowed ? 1 : 0)
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSFRZ", msg);
}

void trace_active_burst_event(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const ActiveBurstDecision& burst,
    bool criticalCurrentFrame,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=active_burst")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " considered=" + std::to_string(burst.considered ? 1 : 0)
        + " active=" + std::to_string(burst.active ? 1 : 0)
        + " allowed=" + std::to_string(burst.allowed ? 1 : 0)
        + " entered=" + std::to_string(burst.entered ? 1 : 0)
        + " exited=" + std::to_string(burst.exited ? 1 : 0)
        + " cap_hit=" + std::to_string(burst.capHit ? 1 : 0)
        + " over_target_bytes=" + std::to_string(static_cast<unsigned long long>(burst.overTargetBytes))
        + " cap_bytes=" + std::to_string(static_cast<unsigned long long>(burst.capBytes))
        + " elapsed_ms=" + std::to_string(static_cast<unsigned long long>(burst.elapsedMs))
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSBURST", msg);
}

void trace_budget_reclaim_retry(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::uint32_t attempt,
    std::size_t reclaimedBytes,
    bool success,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=reclaim_retry")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " attempt=" + std::to_string(static_cast<unsigned long long>(attempt))
        + " reclaimed_bytes=" + std::to_string(static_cast<unsigned long long>(reclaimedBytes))
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " success=" + std::to_string(success ? 1 : 0)
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSEVICT", msg);
}

void trace_pressure_checkpoint(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const PressureCheckpoint& checkpoint,
    std::size_t pendingGrowthBytes,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=pressure_checkpoint")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " state=" + to_cstr(checkpoint.decision.state)
        + " prev_state=" + to_cstr(checkpoint.previousState)
        + " desired_state=" + to_cstr(checkpoint.desiredState)
        + " transition=" + std::to_string(checkpoint.transition ? 1 : 0)
        + " transition_deferred_dwell=" + std::to_string(checkpoint.transitionDeferredByDwell ? 1 : 0)
        + " transition_deferred_rate=" + std::to_string(checkpoint.transitionDeferredByRate ? 1 : 0)
        + " reserve_crossing=" + std::to_string(checkpoint.reserveCrossing ? 1 : 0)
        + " reserve_crossed=" + std::to_string(checkpoint.reserveCrossedNow ? 1 : 0)
        + " sampled=" + std::to_string(checkpoint.sampled ? 1 : 0)
        + " poll_interval_ms=" + std::to_string(checkpoint.pollIntervalMs)
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " soft_target_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.input.softTargetBytes))
        + " reserve_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.input.reserveBytes))
        + " manager_resident_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.input.managerResidentBytes))
        + " retire_pending_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.input.retirePendingBytes))
        + " transient_non_manager_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.input.transientNonManagerBytes))
        + " pressure_total_bytes=" + std::to_string(static_cast<unsigned long long>(pressure_total_bytes(checkpoint.input)))
        + " active_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.memory.activeBytes))
        + " reclaimable_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.memory.reclaimableBytes))
        + " pending_growth_bytes=" + std::to_string(static_cast<unsigned long long>(pendingGrowthBytes))
        + " allow_opportunistic=" + std::to_string(checkpoint.decision.allowOpportunistic ? 1 : 0)
        + " freeze_opportunistic=" + std::to_string(checkpoint.decision.freezeOpportunistic ? 1 : 0)
        + " request_reclaim_pass=" + std::to_string(checkpoint.decision.requestReclaimPass ? 1 : 0)
        + " should_shed_non_critical=" + std::to_string(checkpoint.decision.shouldShedNonCritical ? 1 : 0)
        + " reserve_before_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.reserveBeforeBytes))
        + " reserve_target_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.reserveTargetBytes))
        + " reserve_updated=" + std::to_string(checkpoint.reserveUpdated ? 1 : 0)
        + " effective_reserve_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.decision.effectiveReserveBytes))
        + " effective_headroom_bytes=" + std::to_string(
            static_cast<unsigned long long>(checkpoint.decision.effectiveHeadroomBytes))
        + " headroom_source=" + to_cstr(checkpoint.decision.headroomSource)
        + " driver_free_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.input.driverFreeBytes))
        + " allocator_pool_reserved_bytes=" + std::to_string(
            static_cast<unsigned long long>(checkpoint.input.allocatorPoolReservedBytes))
        + " allocator_pool_used_bytes=" + std::to_string(
            static_cast<unsigned long long>(checkpoint.input.allocatorPoolUsedBytes))
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSPRS", msg);
}

void trace_headroom_sample(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const PressureCheckpoint& checkpoint,
    std::size_t requestBytes,
    bool sourceSwitch,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=headroom")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " effective_headroom_bytes=" + std::to_string(
            static_cast<unsigned long long>(checkpoint.input.effectiveHeadroomBytes))
        + " headroom_source=" + to_cstr(checkpoint.input.headroomSource)
        + " driver_free_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.input.driverFreeBytes))
        + " allocator_pool_reserved_bytes=" + std::to_string(
            static_cast<unsigned long long>(checkpoint.input.allocatorPoolReservedBytes))
        + " allocator_pool_used_bytes=" + std::to_string(
            static_cast<unsigned long long>(checkpoint.input.allocatorPoolUsedBytes))
        + " source_switch=" + std::to_string(sourceSwitch ? 1 : 0)
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSHDR", msg);
}

void trace_transient_non_manager_sample(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const PressureCheckpoint& checkpoint,
    std::size_t pendingGrowthBytes,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=transient_non_manager_sample")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " transient_non_manager_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.input.transientNonManagerBytes))
        + " pending_growth_bytes=" + std::to_string(static_cast<unsigned long long>(pendingGrowthBytes))
        + " pressure_total_bytes=" + std::to_string(static_cast<unsigned long long>(pressure_total_bytes(checkpoint.input)))
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSTRN", msg);
}

void trace_emergency_shed_action(
    const SubmissionTransaction& transaction,
    const char* commandName,
    PressureLane lane,
    PressureState state,
    std::size_t requestBytes,
    bool criticalCurrentFrame,
    bool allowed,
    std::uint64_t uploadBytesInFlight,
    std::uint64_t uploadCapBytes,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=emergency_shed")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " lane=" + to_cstr(lane)
        + " state=" + to_cstr(state)
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + " allowed=" + std::to_string(allowed ? 1 : 0)
        + " upload_bytes_in_flight=" + std::to_string(static_cast<unsigned long long>(uploadBytesInFlight))
        + " upload_cap_bytes=" + std::to_string(static_cast<unsigned long long>(uploadCapBytes))
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSEMS", msg);
}

void trace_lane_wait_event(
    const SubmissionTransaction& transaction,
    const char* commandName,
    PressureLane lane,
    bool criticalCurrentFrame,
    int waitMs,
    const char* outcome,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=lane_wait")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " lane=" + to_cstr(lane)
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + " wait_ms=" + std::to_string(waitMs)
        + " outcome=" + (outcome ? outcome : "unknown")
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSFAIR", msg);
}

void trace_copy_compute_guard(
    const SubmissionTransaction& transaction,
    const char* commandName,
    PressureState state,
    std::uint64_t uploadBytesInFlight,
    std::uint64_t uploadCapBytes,
    bool criticalCurrentFrame,
    bool allowed,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=copy_compute_guard")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " state=" + to_cstr(state)
        + " upload_bytes_in_flight=" + std::to_string(static_cast<unsigned long long>(uploadBytesInFlight))
        + " upload_cap_bytes=" + std::to_string(static_cast<unsigned long long>(uploadCapBytes))
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + " allowed=" + std::to_string(allowed ? 1 : 0)
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSCOPY", msg);
}

AdmissionChurnSnapshot sample_admission_churn_state(
    const SubmissionTransaction& transaction,
    const ResourceManagerConfigEffective& cfg,
    std::uint64_t entryDigest,
    std::uint32_t observedProbationHits) {
    AdmissionChurnSnapshot out{};
    out.enabled = (cfg.admissionChurnWindowMs > 0);
    out.windowMs = cfg.admissionChurnWindowMs;
    out.enterOneHitRatePct = cfg.admissionChurnEnterOneHitRatePct;
    out.exitOneHitRatePct = cfg.admissionChurnExitOneHitRatePct;
    out.probationHitBonus = cfg.admissionChurnProbationHitBonus;
    out.keepHotMs = cfg.keepHotMs;
    out.readmitCooldownMs = cfg.largeEntryReadmitCooldownMs;
    out.ghostHitsForReadmit = cfg.largeEntryGhostHitsForReadmit;
    if (!out.enabled) {
        return out;
    }

    const std::uint64_t nowMs = monotonic_time_ms();
    AdmissionChurnPolicyState& state = admission_churn_policy_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    AdmissionChurnContextState& contextState = state.byContext[transaction.snapshot.deviceContextKey];

    const bool windowReset =
        (contextState.windowStartMs == 0) ||
        ((nowMs > contextState.windowStartMs) &&
         ((nowMs - contextState.windowStartMs) >= static_cast<std::uint64_t>(cfg.admissionChurnWindowMs)));
    if (windowReset) {
        if (contextState.active) {
            contextState.active = false;
            global_state().admissionChurnExitEvents.fetch_add(1, std::memory_order_relaxed);
        }
        contextState.windowStartMs = nowMs;
        contextState.windowSamples = 0;
        contextState.windowOneHitSamples = 0;
        contextState.windowDigestHits.clear();
    }

    contextState.windowSamples += 1;
    if (observedProbationHits == 0) {
        contextState.windowOneHitSamples += 1;
    }
    std::uint32_t& digestHits = contextState.windowDigestHits[entryDigest];
    if (digestHits < std::numeric_limits<std::uint32_t>::max()) {
        ++digestHits;
    }

    const std::uint64_t oneHitRatePctU64 = (contextState.windowSamples == 0)
        ? 0
        : ((contextState.windowOneHitSamples * 100ull) / contextState.windowSamples);
    const std::uint32_t oneHitRatePct = static_cast<std::uint32_t>(std::min<std::uint64_t>(oneHitRatePctU64, 100ull));

    bool nextActive = contextState.active;
    if (!nextActive) {
        nextActive = oneHitRatePct >= cfg.admissionChurnEnterOneHitRatePct;
    }
    else {
        nextActive = oneHitRatePct > cfg.admissionChurnExitOneHitRatePct;
    }
    if (!contextState.active && nextActive) {
        global_state().admissionChurnEnterEvents.fetch_add(1, std::memory_order_relaxed);
    }
    else if (contextState.active && !nextActive) {
        global_state().admissionChurnExitEvents.fetch_add(1, std::memory_order_relaxed);
    }
    contextState.active = nextActive;

    out.active = contextState.active;
    out.oneHitRatePct = oneHitRatePct;
    out.uniqueKeys = static_cast<std::uint32_t>(std::min<std::size_t>(
        contextState.windowDigestHits.size(),
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())));
    out.windowSamples = static_cast<std::uint32_t>(std::min<std::uint64_t>(
        contextState.windowSamples,
        static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())));
    global_state().admissionChurnSampleEvents.fetch_add(1, std::memory_order_relaxed);
    return out;
}

void trace_keep_hot_surface(
    const SubmissionTransaction& transaction,
    const ResourceManagerConfigEffective& cfg,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }
    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=keep_hot_surface")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " enabled=" + std::to_string(cfg.keepHotMs > 0 ? 1 : 0)
        + " keep_hot_ms=" + std::to_string(static_cast<unsigned long long>(cfg.keepHotMs))
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSHOT", msg);
    global_state().keepHotSurfaceTraceEvents.fetch_add(1, std::memory_order_relaxed);
}

void trace_keep_hot_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::uint64_t keepHotMs,
    std::uint64_t bypassEvents,
    std::uint64_t forcedEvictEvents,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=keep_hot_decision")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " keep_hot_ms=" + std::to_string(static_cast<unsigned long long>(keepHotMs))
        + " bypass_events=" + std::to_string(static_cast<unsigned long long>(bypassEvents))
        + " forced_evict_events=" + std::to_string(static_cast<unsigned long long>(forcedEvictEvents))
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSHOT", msg);
}

void trace_burst_debt_surface(
    const SubmissionTransaction& transaction,
    const ResourceManagerConfigEffective& cfg,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }
    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const bool enabled = (cfg.burstDebtHalfLifeMs > 0) && (cfg.maxBurstDebtPct < 100);
    const std::string msg = std::string("event=burst_debt_surface")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " enabled=" + std::to_string(enabled ? 1 : 0)
        + " burst_debt_half_life_ms=" + std::to_string(static_cast<unsigned long long>(cfg.burstDebtHalfLifeMs))
        + " max_burst_debt_pct=" + std::to_string(static_cast<unsigned long long>(cfg.maxBurstDebtPct))
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSBDE", msg);
    global_state().burstDebtSurfaceTraceEvents.fetch_add(1, std::memory_order_relaxed);
}

void trace_burst_debt_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    PressureLane lane,
    PressureState pressureState,
    bool criticalCurrentFrame,
    std::size_t requestBytes,
    const ResourceManagerConfigEffective& cfg,
    const BurstDebtRuntimeDecision& decision) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=burst_debt_decision")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " lane=" + to_cstr(lane)
        + " pressure_state=" + to_cstr(pressureState)
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + " enabled=" + std::to_string(decision.enabled ? 1 : 0)
        + " sampled=" + std::to_string(decision.sampled ? 1 : 0)
        + " burst_consumed=" + std::to_string(decision.burstConsumed ? 1 : 0)
        + " throttled=" + std::to_string(decision.throttled ? 1 : 0)
        + " debt_before_pct=" + std::to_string(decision.debtBeforePct)
        + " debt_after_pct=" + std::to_string(decision.debtAfterPct)
        + " debt_increment_pct=" + std::to_string(decision.debtIncrementPct)
        + " burst_debt_half_life_ms=" + std::to_string(
            static_cast<unsigned long long>(cfg.burstDebtHalfLifeMs))
        + " max_burst_debt_pct=" + std::to_string(
            static_cast<unsigned long long>(cfg.maxBurstDebtPct))
        + " instance_token=" + std::to_string(
            static_cast<unsigned long long>(transaction.snapshot.instanceToken.value))
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " reason=" + (decision.reason ? decision.reason : "unspecified");
    JTRACE("MSBDE", msg);
}

void trace_superseded_builder_cancel_surface(
    const SubmissionTransaction& transaction,
    const ResourceManagerConfigEffective& cfg,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }
    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=superseded_builder_cancel_surface")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " enabled=" + std::to_string(cfg.cancelSupersededBuilders ? 1 : 0)
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSCNL", msg);
    global_state().supersededBuilderCancelSurfaceTraceEvents.fetch_add(1, std::memory_order_relaxed);
}

void trace_superseded_builder_cancel_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    bool criticalCurrentFrame,
    std::uint64_t requestBytes,
    std::uint64_t latestSnapshotId,
    const SupersededBuilderCancelInput& input,
    const SupersededBuilderCancelDecision& decision) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }
    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::uint64_t savedBytes = decision.cancel ? requestBytes : 0;
    const std::string msg = std::string("event=superseded_builder_cancel_decision")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " enabled=" + std::to_string(input.enabled ? 1 : 0)
        + " superseded=" + std::to_string(input.superseded ? 1 : 0)
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + " cancel=" + std::to_string(decision.cancel ? 1 : 0)
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " saved_bytes=" + std::to_string(static_cast<unsigned long long>(savedBytes))
        + " latest_snapshot_id=" + std::to_string(static_cast<unsigned long long>(latestSnapshotId))
        + " instance_token=" + std::to_string(
            static_cast<unsigned long long>(transaction.snapshot.instanceToken.value))
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " reason=" + (decision.reason ? decision.reason : "unspecified");
    JTRACE("MSCNL", msg);
}

void trace_optional_heuristic_surfaces_once(
    const SubmissionTransaction& transaction,
    const ResourceManagerConfigEffective& cfg,
    const char* reason) {
    bool traceKeepHot = false;
    bool traceBurstDebt = false;
    bool traceCancel = false;
    {
        OptionalHeuristicTraceState& traceState = optional_heuristic_trace_state();
        std::lock_guard<std::mutex> lock(traceState.mutex);
        OptionalHeuristicTraceContextState& entry =
            traceState.byContext[transaction.snapshot.deviceContextKey];
        if (!entry.keepHotTraced) {
            entry.keepHotTraced = true;
            traceKeepHot = true;
        }
        if (!entry.burstDebtTraced) {
            entry.burstDebtTraced = true;
            traceBurstDebt = true;
        }
        if (!entry.supersededBuilderCancelTraced) {
            entry.supersededBuilderCancelTraced = true;
            traceCancel = true;
        }
    }
    if (traceKeepHot) {
        trace_keep_hot_surface(transaction, cfg, reason);
    }
    if (traceBurstDebt) {
        trace_burst_debt_surface(transaction, cfg, reason);
    }
    if (traceCancel) {
        trace_superseded_builder_cancel_surface(transaction, cfg, reason);
    }
}

std::uint32_t effective_probation_hits_required(
    std::uint32_t baseRequired,
    const AdmissionChurnSnapshot& churnSnapshot) noexcept {
    if (!churnSnapshot.enabled || !churnSnapshot.active || churnSnapshot.probationHitBonus == 0) {
        return baseRequired;
    }
    const std::uint64_t expanded =
        static_cast<std::uint64_t>(baseRequired) + static_cast<std::uint64_t>(churnSnapshot.probationHitBonus);
    const std::uint64_t capped = std::min<std::uint64_t>(
        expanded,
        static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()));
    return static_cast<std::uint32_t>(capped);
}

std::uint32_t next_probation_hits(std::uint32_t observed) noexcept {
    return (observed < std::numeric_limits<std::uint32_t>::max())
        ? (observed + 1u)
        : std::numeric_limits<std::uint32_t>::max();
}

void trace_large_entry_readmit_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::uint64_t entryDigest,
    std::uint64_t requestBytes,
    std::uint64_t thresholdBytes,
    const LargeEntryReadmitDecision& decision) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=large_entry_readmit")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " entry_digest=" + std::to_string(static_cast<unsigned long long>(entryDigest))
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(thresholdBytes))
        + " enabled=" + std::to_string(decision.enabled ? 1 : 0)
        + " candidate=" + std::to_string(decision.candidate ? 1 : 0)
        + " critical_current_frame=" + std::to_string(decision.criticalCurrentFrame ? 1 : 0)
        + " had_history=" + std::to_string(decision.hadHistory ? 1 : 0)
        + " in_cooldown=" + std::to_string(decision.inCooldown ? 1 : 0)
        + " blocked=" + std::to_string(decision.blocked ? 1 : 0)
        + " ghost_bypass=" + std::to_string(decision.ghostBypass ? 1 : 0)
        + " age_ms=" + std::to_string(static_cast<unsigned long long>(decision.ageMs))
        + " cooldown_ms=" + std::to_string(static_cast<unsigned long long>(decision.cooldownMs))
        + " ghost_hits_required=" + std::to_string(decision.ghostHitsRequired)
        + " observed_ghost_hits=" + std::to_string(decision.observedGhostHits)
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " reason=" + (decision.reason ? decision.reason : "unspecified");
    JTRACE("MSTHR", msg);
}

void trace_cache_admission_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const CacheAdmissionDecision& decision,
    const AdmissionChurnSnapshot& churnSnapshot,
    bool criticalCurrentFrame,
    std::uint64_t requestBytes,
    std::uint32_t observedProbationHits,
    std::uint64_t entryDigest,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=cache_admission")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " class=" + to_cstr(decision.admissionClass)
        + " allow_durable=" + std::to_string(decision.allowDurableAdmission ? 1 : 0)
        + " probation_applied=" + std::to_string(decision.probationApplied ? 1 : 0)
        + " probation_hits_required=" + std::to_string(decision.probationHitsRequired)
        + " observed_probation_hits=" + std::to_string(observedProbationHits)
        + " churn_enabled=" + std::to_string(churnSnapshot.enabled ? 1 : 0)
        + " churn_active=" + std::to_string(churnSnapshot.active ? 1 : 0)
        + " churn_window_ms=" + std::to_string(churnSnapshot.windowMs)
        + " churn_window_samples=" + std::to_string(churnSnapshot.windowSamples)
        + " churn_one_hit_rate_pct=" + std::to_string(churnSnapshot.oneHitRatePct)
        + " churn_unique_keys=" + std::to_string(churnSnapshot.uniqueKeys)
        + " churn_enter_one_hit_rate_pct=" + std::to_string(churnSnapshot.enterOneHitRatePct)
        + " churn_exit_one_hit_rate_pct=" + std::to_string(churnSnapshot.exitOneHitRatePct)
        + " churn_probation_hit_bonus=" + std::to_string(churnSnapshot.probationHitBonus)
        + " keep_hot_ms=" + std::to_string(churnSnapshot.keepHotMs)
        + " readmit_cooldown_ms=" + std::to_string(churnSnapshot.readmitCooldownMs)
        + " ghost_hits_for_readmit=" + std::to_string(churnSnapshot.ghostHitsForReadmit)
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " max_durable_bytes=" + std::to_string(static_cast<unsigned long long>(decision.maxDurableBytes))
        + " entry_digest=" + std::to_string(static_cast<unsigned long long>(entryDigest))
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " decision_reason=" + (decision.reason ? decision.reason : "unspecified")
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSADM", msg);
}

void trace_probation_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::uint64_t entryDigest,
    std::uint32_t observedProbationHits,
    std::uint32_t requiredProbationHits,
    bool admitted,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=probation")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " entry_digest=" + std::to_string(static_cast<unsigned long long>(entryDigest))
        + " observed_probation_hits=" + std::to_string(observedProbationHits)
        + " required_probation_hits=" + std::to_string(requiredProbationHits)
        + " admitted=" + std::to_string(admitted ? 1 : 0)
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSPRB", msg);
}

void trace_graph_large_entry_quarantine(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::uint64_t thresholdBytes,
    std::uint64_t capBytes,
    std::uint32_t capEntries,
    std::uint64_t residentBytes,
    std::uint32_t residentEntries,
    std::uint64_t decayEvictedEntries,
    std::uint64_t capTrimEvictedEntries,
    bool capHit,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=graph_large_quarantine")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(thresholdBytes))
        + " cap_bytes=" + std::to_string(static_cast<unsigned long long>(capBytes))
        + " cap_entries=" + std::to_string(capEntries)
        + " resident_bytes=" + std::to_string(static_cast<unsigned long long>(residentBytes))
        + " resident_entries=" + std::to_string(residentEntries)
        + " decay_evicted_entries=" + std::to_string(static_cast<unsigned long long>(decayEvictedEntries))
        + " cap_trim_evicted_entries=" + std::to_string(
            static_cast<unsigned long long>(capTrimEvictedEntries))
        + " cap_hit=" + std::to_string(capHit ? 1 : 0)
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSADM", msg);
}

void trace_reap_pass(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::size_t reclaimedBytes,
    bool success,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=reap_pass")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " reclaimed_bytes=" + std::to_string(static_cast<unsigned long long>(reclaimedBytes))
        + " success=" + std::to_string(success ? 1 : 0)
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSREAP", msg);
}

void trace_fragmentation_recovery(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::uint32_t attempt,
    std::size_t requestBytes,
    std::size_t reapedBytes,
    std::uint64_t quarantineTrimmedEntries,
    std::uint64_t graphEvictedEntries,
    bool success,
    const char* stage,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=fragmentation_recovery")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " attempt=" + std::to_string(static_cast<unsigned long long>(attempt))
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " reaped_bytes=" + std::to_string(static_cast<unsigned long long>(reapedBytes))
        + " quarantine_trimmed_entries=" + std::to_string(
            static_cast<unsigned long long>(quarantineTrimmedEntries))
        + " graph_evicted_entries=" + std::to_string(static_cast<unsigned long long>(graphEvictedEntries))
        + " success=" + std::to_string(success ? 1 : 0)
        + " stage=" + (stage ? stage : "unknown")
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSFRAG", msg);
}

PressureCheckpoint evaluate_pressure_checkpoint(
    const SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    std::size_t pendingGrowthBytes,
    bool forceSample,
    const char* commandName) {
    PressureCheckpoint checkpoint{};
    checkpoint.memory = snapshot_manager_memory(resources);

    const std::uint64_t graphActiveBytes =
        estimate_graph_cache_active_bytes_for_context(transaction.snapshot.deviceContextKey);
    std::uint64_t nextActiveBytes = 0;
    if (add_u64_checked(checkpoint.memory.activeBytes, graphActiveBytes, nextActiveBytes)) {
        checkpoint.memory.activeBytes = nextActiveBytes;
    }
    else {
        checkpoint.memory.activeBytes = std::numeric_limits<std::uint64_t>::max();
        checkpoint.memory.overflow = true;
    }
    std::uint64_t nextReclaimableBytes = 0;
    if (add_u64_checked(checkpoint.memory.reclaimableBytes, graphActiveBytes, nextReclaimableBytes)) {
        checkpoint.memory.reclaimableBytes = nextReclaimableBytes;
    }
    else {
        checkpoint.memory.reclaimableBytes = std::numeric_limits<std::uint64_t>::max();
        checkpoint.memory.overflow = true;
    }

    publish_manager_memory_snapshot(checkpoint.memory);

    const HeadroomTelemetry headroom = sample_headroom_telemetry();
    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    ResourceManagerState& managerState = global_state();
    checkpoint.input.softTargetBytes = cfg.managerSoftTargetBytes;
    checkpoint.input.reserveBytes = cfg.managerReserveBytes;
    checkpoint.input.retirePendingBytes = checkpoint.memory.retirePendingBytes;
    checkpoint.input.transientNonManagerBytes = checkpoint.memory.transientNonManagerBytes;
    checkpoint.input.effectiveHeadroomBytes = headroom.effectiveHeadroomBytes;
    checkpoint.input.driverFreeBytes = headroom.driverFreeBytes;
    checkpoint.input.allocatorPoolReservedBytes = headroom.allocatorPoolReservedBytes;
    checkpoint.input.allocatorPoolUsedBytes = headroom.allocatorPoolUsedBytes;
    checkpoint.input.headroomSource = headroom.source;

    managerState.allocatorEffectiveHeadroomBytes.store(
        checkpoint.input.effectiveHeadroomBytes,
        std::memory_order_relaxed);
    managerState.allocatorPoolReservedBytes.store(
        checkpoint.input.allocatorPoolReservedBytes,
        std::memory_order_relaxed);
    managerState.allocatorPoolUsedBytes.store(
        checkpoint.input.allocatorPoolUsedBytes,
        std::memory_order_relaxed);

    std::uint64_t predictedResident = checkpoint.memory.activeBytes;
    std::uint64_t nextResident = 0;
    if (add_u64_checked(predictedResident, static_cast<std::uint64_t>(pendingGrowthBytes), nextResident)) {
        predictedResident = nextResident;
    }
    else {
        predictedResident = std::numeric_limits<std::uint64_t>::max();
        checkpoint.memory.overflow = true;
    }
    checkpoint.input.managerResidentBytes = predictedResident;
    checkpoint.input.freezeOpportunisticBelowReserve = cfg.freezeOpportunisticBelowReserve;
    bool headroomSourceSwitch = false;

    const std::uint64_t nowMs = monotonic_time_ms();
    PressurePolicyState& policyState = pressure_policy_state();
    {
        std::lock_guard<std::mutex> lock(policyState.mutex);
        PressureContextState& contextState = policyState.byContext[transaction.snapshot.deviceContextKey];
        const PressureState cadenceState = contextState.valid
            ? contextState.lastState
            : PressureState::Normal;
        checkpoint.pollIntervalMs = pressure_poll_interval_ms_for_state(cfg, cadenceState);

        bool sampleDue = forceSample || !contextState.valid;
        if (!sampleDue && contextState.headroomSourceValid &&
            contextState.lastHeadroomSource != checkpoint.input.headroomSource) {
            sampleDue = true;
            headroomSourceSwitch = true;
        }
        if (!sampleDue) {
            const std::uint64_t elapsedMs =
                (nowMs > contextState.lastSampleMs) ? (nowMs - contextState.lastSampleMs) : 0;
            sampleDue = elapsedMs >= static_cast<std::uint64_t>(checkpoint.pollIntervalMs);
        }

        checkpoint.reserveBeforeBytes = contextState.effectiveReserveValid
            ? contextState.effectiveReserveBytes
            : cfg.managerReserveBytes;
        checkpoint.reserveTargetBytes = compute_effective_reserve_target_bytes(
            cfg,
            checkpoint.input.transientNonManagerBytes);
        if (sampleDue || !contextState.effectiveReserveValid) {
            std::uint64_t nextReserve = step_effective_reserve_bytes(
                cfg,
                checkpoint.reserveBeforeBytes,
                checkpoint.reserveTargetBytes);
            if (cfg.managerSoftTargetBytes > 0) {
                nextReserve = std::min<std::uint64_t>(nextReserve, cfg.managerSoftTargetBytes);
            }
            checkpoint.reserveUpdated = nextReserve != checkpoint.reserveBeforeBytes;
            contextState.effectiveReserveBytes = nextReserve;
            contextState.effectiveReserveValid = true;
        }

        checkpoint.input.effectiveReserveBytes = contextState.effectiveReserveValid
            ? contextState.effectiveReserveBytes
            : cfg.managerReserveBytes;
        const PressureDecision computedDecision = classify_pressure(checkpoint.input);
        checkpoint.decision = computedDecision;
        checkpoint.desiredState = computedDecision.state;
        checkpoint.reserveCrossedNow = reserve_crossed(checkpoint.input);

        if (sampleDue) {
            checkpoint.sampled = true;
            checkpoint.previousState = contextState.valid ? contextState.lastState : computedDecision.state;
            checkpoint.desiredState = computedDecision.state;

            if (contextState.transitionWindowStartMs == 0 ||
                nowMs < contextState.transitionWindowStartMs ||
                (nowMs - contextState.transitionWindowStartMs) >= 60000ull) {
                contextState.transitionWindowStartMs = nowMs;
                contextState.transitionsInWindow = 0;
            }

            PressureDecision effectiveDecision = computedDecision;
            const bool transitionRequested =
                contextState.valid && (contextState.lastState != computedDecision.state);
            bool transitionAllowed = true;
            if (transitionRequested) {
                const bool escalation =
                    pressure_state_rank(computedDecision.state) >
                    pressure_state_rank(contextState.lastState);
                if (!escalation) {
                    const std::uint64_t stateAgeMs = (contextState.lastStateChangeMs == 0 || nowMs < contextState.lastStateChangeMs)
                        ? std::numeric_limits<std::uint64_t>::max()
                        : (nowMs - contextState.lastStateChangeMs);
                    const bool dwellOk =
                        stateAgeMs >= static_cast<std::uint64_t>(cfg.pressureStateMinDwellMs);
                    const bool rateOk =
                        contextState.transitionsInWindow < cfg.pressureStateMaxTransitionsPerMin;
                    checkpoint.transitionDeferredByDwell = !dwellOk;
                    checkpoint.transitionDeferredByRate = !rateOk;
                    transitionAllowed = dwellOk && rateOk;
                }
            }

            checkpoint.transition = transitionRequested && transitionAllowed;
            if (transitionRequested && !transitionAllowed) {
                effectiveDecision = computedDecision;
                effectiveDecision.state = contextState.lastState;
                checkpoint.decision = effectiveDecision;
                if (checkpoint.transitionDeferredByDwell) {
                    managerState.pressureTransitionDwellDefers.fetch_add(1, std::memory_order_relaxed);
                }
                if (checkpoint.transitionDeferredByRate) {
                    managerState.pressureTransitionRateDefers.fetch_add(1, std::memory_order_relaxed);
                }
            }
            else {
                checkpoint.decision = effectiveDecision;
            }

            checkpoint.reserveCrossing = contextState.valid &&
                (contextState.reserveCrossed != checkpoint.reserveCrossedNow);
            if (contextState.headroomSourceValid &&
                contextState.lastHeadroomSource != checkpoint.input.headroomSource) {
                headroomSourceSwitch = true;
            }
            const bool freezeNow = checkpoint.decision.freezeOpportunistic;
            checkpoint.freezeTransitionEnter = !contextState.opportunisticFrozen && freezeNow;
            checkpoint.freezeTransitionExit = contextState.opportunisticFrozen && !freezeNow;
            contextState.valid = true;
            contextState.lastSampleMs = nowMs;
            contextState.lastDecision = checkpoint.decision;
            contextState.lastState = checkpoint.decision.state;
            contextState.opportunisticFrozen = freezeNow;
            if (!contextState.lastStateChangeMs) {
                contextState.lastStateChangeMs = nowMs;
            }
            if (checkpoint.transition) {
                contextState.lastStateChangeMs = nowMs;
                if (contextState.transitionsInWindow < std::numeric_limits<std::uint32_t>::max()) {
                    ++contextState.transitionsInWindow;
                }
            }
            contextState.reserveCrossed = checkpoint.reserveCrossedNow;
            contextState.headroomSourceValid = true;
            contextState.lastHeadroomSource = checkpoint.input.headroomSource;

            if (checkpoint.transition) {
                managerState.pressureStateTransitions.fetch_add(1, std::memory_order_relaxed);
            }
            if (checkpoint.reserveCrossing) {
                managerState.reserveCrossingEvents.fetch_add(1, std::memory_order_relaxed);
            }
            if (checkpoint.reserveUpdated) {
                managerState.reserveAdaptationEvents.fetch_add(1, std::memory_order_relaxed);
            }
            if (checkpoint.freezeTransitionEnter) {
                managerState.opportunisticFreezeEnterEvents.fetch_add(1, std::memory_order_relaxed);
            }
            if (checkpoint.freezeTransitionExit) {
                managerState.opportunisticFreezeExitEvents.fetch_add(1, std::memory_order_relaxed);
            }
            if (headroomSourceSwitch) {
                managerState.headroomSourceSwitches.fetch_add(1, std::memory_order_relaxed);
            }
        }
        else if (contextState.valid) {
            checkpoint.decision = contextState.lastDecision;
            checkpoint.previousState = contextState.lastState;
            checkpoint.desiredState = computedDecision.state;
            checkpoint.reserveCrossedNow = contextState.reserveCrossed;
            checkpoint.input.effectiveReserveBytes = contextState.effectiveReserveValid
                ? contextState.effectiveReserveBytes
                : cfg.managerReserveBytes;
        }
    }

    if (checkpoint.sampled ||
        checkpoint.transition ||
        checkpoint.reserveCrossing ||
        checkpoint.reserveUpdated ||
        checkpoint.freezeTransitionEnter ||
        checkpoint.freezeTransitionExit ||
        headroomSourceSwitch) {
        const char* pressureReason = "sample";
        if (checkpoint.transitionDeferredByRate) {
            pressureReason = "state_transition_rate_deferred";
        }
        else if (checkpoint.transitionDeferredByDwell) {
            pressureReason = "state_transition_dwell_deferred";
        }
        else if (checkpoint.transition) {
            pressureReason = "state_transition";
        }
        else if (checkpoint.freezeTransitionEnter) {
            pressureReason = "freeze_enter";
        }
        else if (checkpoint.freezeTransitionExit) {
            pressureReason = "freeze_exit";
        }
        else if (checkpoint.reserveUpdated) {
            pressureReason = "reserve_updated";
        }
        else if (checkpoint.reserveCrossing) {
            pressureReason = "reserve_crossing";
        }
        trace_pressure_checkpoint(
            transaction,
            commandName,
            checkpoint,
            pendingGrowthBytes,
            pressureReason);
        trace_transient_non_manager_sample(
            transaction,
            commandName,
            checkpoint,
            pendingGrowthBytes,
            checkpoint.sampled ? "sampled" : "cached");
        trace_headroom_sample(
            transaction,
            commandName,
            checkpoint,
            pendingGrowthBytes,
            headroomSourceSwitch,
            headroomSourceSwitch ? "source_switch" : (checkpoint.sampled ? "sampled" : "cached"));

        if (checkpoint.sampled || checkpoint.reserveUpdated || JTRACE_ENABLED(3)) {
            trace_effective_reserve_event(
                transaction,
                commandName,
                checkpoint.input.reserveBytes,
                checkpoint.reserveBeforeBytes,
                checkpoint.reserveTargetBytes,
                checkpoint.input.effectiveReserveBytes,
                checkpoint.input.transientNonManagerBytes,
                checkpoint.reserveUpdated,
                checkpoint.reserveUpdated ? "reserve_update" : "reserve_sample");
        }
        if (checkpoint.freezeTransitionEnter || checkpoint.freezeTransitionExit || JTRACE_ENABLED(3)) {
            const bool freezeActive = checkpoint.decision.freezeOpportunistic;
            trace_opportunistic_freeze_event(
                transaction,
                commandName,
                checkpoint.decision.state,
                checkpoint.input.effectiveHeadroomBytes,
                checkpoint.decision.effectiveReserveBytes,
                freezeActive,
                !freezeActive,
                false,
                checkpoint.freezeTransitionEnter
                    ? "freeze_enter"
                    : (checkpoint.freezeTransitionExit ? "freeze_exit" : "freeze_sample"));
        }
    }

    return checkpoint;
}

void record_allocator_oom_headroom_observation(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::size_t requestBytes) {
    const HeadroomTelemetry headroom = sample_headroom_telemetry();
    ResourceManagerState& managerState = global_state();
    managerState.allocatorEffectiveHeadroomBytes.store(headroom.effectiveHeadroomBytes, std::memory_order_relaxed);
    managerState.allocatorPoolReservedBytes.store(headroom.allocatorPoolReservedBytes, std::memory_order_relaxed);
    managerState.allocatorPoolUsedBytes.store(headroom.allocatorPoolUsedBytes, std::memory_order_relaxed);

    const bool aboveHeadroom =
        (requestBytes > 0) && (headroom.effectiveHeadroomBytes >= static_cast<std::uint64_t>(requestBytes));
    if (aboveHeadroom) {
        managerState.allocFailAboveHeadroomEvents.fetch_add(1, std::memory_order_relaxed);
    }

    PressureCheckpoint checkpoint{};
    checkpoint.input.effectiveHeadroomBytes = headroom.effectiveHeadroomBytes;
    checkpoint.input.driverFreeBytes = headroom.driverFreeBytes;
    checkpoint.input.allocatorPoolReservedBytes = headroom.allocatorPoolReservedBytes;
    checkpoint.input.allocatorPoolUsedBytes = headroom.allocatorPoolUsedBytes;
    checkpoint.input.headroomSource = headroom.source;
    trace_headroom_sample(
        transaction,
        commandName,
        checkpoint,
        requestBytes,
        false,
        aboveHeadroom ? "alloc_fail_above_headroom" : "alloc_fail_below_headroom");
}

bool run_reap_pass_for_pressure(
    const SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const char* commandName,
    const char* reason,
    std::string& outError) {
    std::size_t reclaimedBytes = 0;
    std::string reclaimError;
    if (!JuicerCuda::reap_retired_allocations(resources, reclaimedBytes, reclaimError)) {
        outError = reclaimError.empty() ? "reap pass failed" : reclaimError;
        trace_reap_pass(
            transaction,
            commandName,
            reclaimedBytes,
            false,
            outError.c_str());
        return false;
    }

    if (reclaimedBytes > 0) {
        global_state().retireReapPasses.fetch_add(1, std::memory_order_relaxed);
        global_state().retireReapBytes.fetch_add(reclaimedBytes, std::memory_order_relaxed);
    }
    trace_reap_pass(
        transaction,
        commandName,
        reclaimedBytes,
        true,
        reason);
    publish_manager_memory_snapshot(snapshot_manager_memory(resources));
    return true;
}

void run_tier_target_pretrim(
    const SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const char* commandName,
    PressureLane lane,
    PressureState pressureState,
    const TierBudgetSnapshot& tierBudget,
    bool& outDidWork,
    std::uint64_t& outScratchTrimmedEntries,
    std::uint64_t& outGraphEvictedEntries) noexcept {
    (void)resources;
    outDidWork = false;
    outScratchTrimmedEntries = 0;
    outGraphEvictedEntries = 0;

    const bool constrainedOrWorse =
        pressureState == PressureState::Constrained ||
        pressureState == PressureState::Critical ||
        pressureState == PressureState::Emergency;
    const bool allowProactiveTrim = constrainedOrWorse || lane == PressureLane::Builder;

    const std::uint64_t graphOverTarget =
        tier_bytes_at(tierBudget.overTargetBytes, ResourceTier::Graph);
    const std::uint64_t scratchOverTarget =
        tier_bytes_at(tierBudget.overTargetBytes, ResourceTier::Scratch);

    if (graphOverTarget > 0 && allowProactiveTrim) {
        outGraphEvictedEntries =
            evict_noncritical_graph_entries_for_context(transaction.snapshot.deviceContextKey);
    }
    if (scratchOverTarget > 0 && allowProactiveTrim) {
        outScratchTrimmedEntries =
            trim_large_frame_quarantine_for_context(transaction.snapshot.deviceContextKey);
    }

    outDidWork = (outGraphEvictedEntries > 0) || (outScratchTrimmedEntries > 0);
    if (outDidWork && JTRACE_ENABLED(3)) {
        trace_tier_budget_event(
            transaction,
            commandName,
            lane,
            pressureState,
            tierBudget,
            0,
            false,
            false,
            outScratchTrimmedEntries,
            outGraphEvictedEntries,
            "tier_target_pretrim");
    }
}

void evaluate_active_burst_window(
    const SubmissionTransaction& transaction,
    const ResourceManagerConfigEffective& cfg,
    const PressureCheckpoint& checkpoint,
    std::size_t requestBytes,
    bool criticalCurrentFrame,
    ActiveBurstDecision& outDecision) noexcept {
    outDecision = ActiveBurstDecision{};

    const std::uint64_t softTargetBytes = checkpoint.input.softTargetBytes;
    if (requestBytes == 0 || softTargetBytes == 0) {
        return;
    }

    const std::uint64_t residentBytes = checkpoint.input.managerResidentBytes;
    if (residentBytes <= softTargetBytes) {
        PressurePolicyState& policyState = pressure_policy_state();
        std::lock_guard<std::mutex> lock(policyState.mutex);
        auto contextIt = policyState.byContext.find(transaction.snapshot.deviceContextKey);
        if (contextIt != policyState.byContext.end()) {
            PressureContextState& contextState = contextIt->second;
            if (contextState.burstActive) {
                contextState.burstActive = false;
                contextState.burstCapHitLatched = false;
                contextState.burstWindowStartMs = 0;
                contextState.burstPeakOverTargetBytes = 0;
                outDecision.exited = true;
            }
        }
        return;
    }

    outDecision.considered = true;
    outDecision.overTargetBytes = residentBytes - softTargetBytes;
    outDecision.capBytes = active_burst_cap_bytes(cfg, softTargetBytes);

    PressurePolicyState& policyState = pressure_policy_state();
    const std::uint64_t nowMs = monotonic_time_ms();
    std::lock_guard<std::mutex> lock(policyState.mutex);
    PressureContextState& contextState = policyState.byContext[transaction.snapshot.deviceContextKey];

    if (!cfg.allowActiveFrameBurst || !criticalCurrentFrame) {
        if (contextState.burstActive) {
            contextState.burstActive = false;
            contextState.burstCapHitLatched = false;
            contextState.burstWindowStartMs = 0;
            contextState.burstPeakOverTargetBytes = 0;
            outDecision.exited = true;
        }
        return;
    }

    if (!contextState.burstActive) {
        contextState.burstActive = true;
        contextState.burstCapHitLatched = false;
        contextState.burstWindowStartMs = nowMs;
        contextState.burstPeakOverTargetBytes = outDecision.overTargetBytes;
        outDecision.entered = true;
    }
    else {
        contextState.burstPeakOverTargetBytes = std::max<std::uint64_t>(
            contextState.burstPeakOverTargetBytes,
            outDecision.overTargetBytes);
    }

    const std::uint64_t elapsedMs = (nowMs > contextState.burstWindowStartMs)
        ? (nowMs - contextState.burstWindowStartMs)
        : 0;
    outDecision.elapsedMs = elapsedMs;
    outDecision.active = contextState.burstActive;
    outDecision.allowed = true;
    if (outDecision.capBytes == 0) {
        outDecision.allowed = false;
    }
    if (cfg.maxActiveBurstMs == 0 || elapsedMs > static_cast<std::uint64_t>(cfg.maxActiveBurstMs)) {
        outDecision.allowed = false;
    }
    if (contextState.burstPeakOverTargetBytes > outDecision.capBytes) {
        outDecision.allowed = false;
    }
    if (!outDecision.allowed && !contextState.burstCapHitLatched) {
        contextState.burstCapHitLatched = true;
        outDecision.capHit = true;
    }
}

BurstDebtRuntimeDecision evaluate_burst_debt_runtime(
    const SubmissionTransaction& transaction,
    const ResourceManagerConfigEffective& cfg,
    const ActiveBurstDecision& burstDecision,
    bool criticalCurrentFrame,
    std::size_t requestBytes) {
    BurstDebtRuntimeDecision out{};
    if (requestBytes == 0) {
        out.reason = "zero_request";
        return out;
    }

    out.enabled = (cfg.burstDebtHalfLifeMs > 0) && (cfg.maxBurstDebtPct < 100);
    if (!out.enabled) {
        out.reason = "disabled";
        return out;
    }
    out.sampled = true;
    global_state().burstDebtSampleEvents.fetch_add(1, std::memory_order_relaxed);

    const std::uint64_t instanceToken = transaction.snapshot.instanceToken.value;
    if (instanceToken == 0) {
        out.reason = "missing_instance_token";
        return out;
    }

    const std::uint64_t nowMs = monotonic_time_ms();
    out.burstConsumed = criticalCurrentFrame &&
        burstDecision.considered &&
        burstDecision.allowed;

    PressurePolicyState& policyState = pressure_policy_state();
    std::lock_guard<std::mutex> lock(policyState.mutex);
    PressureContextState& contextState = policyState.byContext[transaction.snapshot.deviceContextKey];
    BurstDebtEntry& debtEntry = contextState.burstDebtByInstance[instanceToken];

    if (debtEntry.lastUpdateMs > 0) {
        const std::uint64_t elapsedMs = (nowMs > debtEntry.lastUpdateMs)
            ? (nowMs - debtEntry.lastUpdateMs)
            : 0;
        debtEntry.debtPct = decay_burst_debt_pct(
            debtEntry.debtPct,
            elapsedMs,
            cfg.burstDebtHalfLifeMs);
    }
    debtEntry.lastUpdateMs = nowMs;
    out.debtBeforePct = debtEntry.debtPct;

    BurstDebtInput debtInput{};
    debtInput.burstDebtHalfLifeMs = cfg.burstDebtHalfLifeMs;
    debtInput.maxBurstDebtPct = cfg.maxBurstDebtPct;
    debtInput.currentDebtPct = debtEntry.debtPct;
    debtInput.criticalCurrentFrame = criticalCurrentFrame;
    debtInput.burstConsumed = out.burstConsumed;
    debtInput.burstOverTargetBytes = burstDecision.overTargetBytes;
    debtInput.burstCapBytes = burstDecision.capBytes;
    const BurstDebtDecision debtDecision = classify_burst_debt(debtInput);
    out.reason = debtDecision.reason;

    if (debtDecision.accrueDebt && debtDecision.debtIncrementPct > 0) {
        const std::uint64_t expandedDebt =
            static_cast<std::uint64_t>(debtEntry.debtPct) +
            static_cast<std::uint64_t>(debtDecision.debtIncrementPct);
        debtEntry.debtPct = static_cast<std::uint32_t>(std::min<std::uint64_t>(100ull, expandedDebt));
        out.debtIncrementPct = debtDecision.debtIncrementPct;
        global_state().burstDebtAccrualEvents.fetch_add(1, std::memory_order_relaxed);
    }
    if (debtDecision.throttleOpportunistic) {
        out.throttled = true;
        global_state().burstDebtThrottleEvents.fetch_add(1, std::memory_order_relaxed);
    }
    out.debtAfterPct = debtEntry.debtPct;

    if (debtEntry.debtPct == 0 && !out.burstConsumed && !out.throttled) {
        contextState.burstDebtByInstance.erase(instanceToken);
    }
    return out;
}

std::string make_superseded_builder_cancel_error(
    const char* commandName,
    std::uint64_t latestSnapshotId) {
    std::string out = std::string(kReservationDeferredPrefix) + "superseded_builder_cancel";
    if (commandName && *commandName) {
        out += ":command=";
        out += commandName;
    }
    if (latestSnapshotId > 0) {
        out += ":latest_snapshot_id=" + std::to_string(static_cast<unsigned long long>(latestSnapshotId));
    }
    return out;
}

bool should_cancel_superseded_noncritical_builder(
    const SubmissionTransaction& transaction,
    const char* commandName,
    bool criticalCurrentFrame,
    std::uint64_t requestBytes,
    std::uint64_t& outLatestSnapshotId) {
    outLatestSnapshotId = 0;
    const ResourceManagerConfigEffective& cfg = manager_effective_config();

    SupersededBuilderCancelInput input{};
    input.enabled = cfg.cancelSupersededBuilders;
    input.criticalCurrentFrame = criticalCurrentFrame;
    input.superseded = state_snapshot_is_superseded(transaction.snapshot, &outLatestSnapshotId);
    const SupersededBuilderCancelDecision decision =
        classify_superseded_builder_cancel(input);

    if (input.enabled || decision.cancel || JTRACE_ENABLED(3)) {
        trace_superseded_builder_cancel_decision(
            transaction,
            commandName,
            criticalCurrentFrame,
            requestBytes,
            outLatestSnapshotId,
            input,
            decision);
    }

    if (!decision.cancel) {
        return false;
    }

    ResourceManagerState& state = global_state();
    state.supersededBuilderCancelEvents.fetch_add(1, std::memory_order_relaxed);
    if (requestBytes > 0) {
        state.supersededBuilderCancelSavedBytes.fetch_add(requestBytes, std::memory_order_relaxed);
    }
    return true;
}

bool enforce_pressure_gate(
    const SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const char* commandName,
    PressureLane lane,
    std::size_t requestBytes,
    bool criticalCurrentFrame,
    bool& outRequestReclaimPass,
    std::string& outError) {
    outRequestReclaimPass = false;
    outError.clear();
    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    const std::uint64_t uploadCapBytes = upload_reservation_cap_bytes(cfg);
    const std::uint64_t uploadBytesInFlight =
        global_state().uploadBytesInFlight.load(std::memory_order_relaxed);
    const bool uploadCapEnabled = (uploadCapBytes > 0);
    const bool uploadCapSaturated = uploadCapEnabled && (uploadBytesInFlight >= uploadCapBytes);
    const bool nonCritical = !criticalCurrentFrame;
    std::uint64_t supersededLatestSnapshotId = 0;
    if (requestBytes > 0 &&
        should_cancel_superseded_noncritical_builder(
            transaction,
            commandName,
            criticalCurrentFrame,
            static_cast<std::uint64_t>(requestBytes),
            supersededLatestSnapshotId)) {
        outError = make_superseded_builder_cancel_error(commandName, supersededLatestSnapshotId);
        return false;
    }
    PressureState pressureState = PressureState::Normal;
    TierBudgetSnapshot tierBudget{};
    bool tierBudgetValid = false;
    const bool pressureEnabled = pressure_policy_enabled(cfg);
    PressureCheckpoint checkpoint{};
    bool checkpointValid = false;
    bool freezeBelowReserve = false;
    ActiveBurstDecision burstDecision{};
    BurstDebtRuntimeDecision burstDebtDecision{};
    if (pressureEnabled) {
        checkpoint = evaluate_pressure_checkpoint(
            transaction,
            resources,
            requestBytes,
            requestBytes > 0,
            commandName);
        checkpointValid = true;
        outRequestReclaimPass = checkpoint.decision.requestReclaimPass && (requestBytes > 0);
        pressureState = checkpoint.decision.state;
        freezeBelowReserve = checkpoint.decision.freezeOpportunistic;
        maybe_apply_async_mempool_release_policy(
            transaction,
            pressureState,
            cfg,
            "pressure_checkpoint");

        evaluate_active_burst_window(
            transaction,
            cfg,
            checkpoint,
            requestBytes,
            criticalCurrentFrame,
            burstDecision);
        if (burstDecision.entered) {
            global_state().activeBurstEnterEvents.fetch_add(1, std::memory_order_relaxed);
        }
        if (burstDecision.exited) {
            global_state().activeBurstExitEvents.fetch_add(1, std::memory_order_relaxed);
        }
        if (burstDecision.capHit) {
            global_state().activeBurstCapHitEvents.fetch_add(1, std::memory_order_relaxed);
        }
        if (burstDecision.considered && !burstDecision.allowed && requestBytes > 0) {
            outRequestReclaimPass = true;
        }
        if (burstDecision.capHit && requestBytes > 0) {
            outRequestReclaimPass = true;
        }
        if (burstDecision.entered || burstDecision.exited || burstDecision.capHit || JTRACE_ENABLED(3)) {
            trace_active_burst_event(
                transaction,
                commandName,
                burstDecision,
                criticalCurrentFrame,
                burstDecision.capHit
                    ? "burst_cap_hit"
                    : (burstDecision.entered
                        ? "burst_enter"
                        : (burstDecision.exited ? "burst_exit" : "burst_sample")));
        }
        if (checkpoint.freezeTransitionEnter || checkpoint.freezeTransitionExit || JTRACE_ENABLED(3)) {
            trace_opportunistic_freeze_event(
                transaction,
                commandName,
                checkpoint.decision.state,
                checkpoint.input.effectiveHeadroomBytes,
                checkpoint.decision.effectiveReserveBytes,
                freezeBelowReserve,
                !freezeBelowReserve,
                criticalCurrentFrame,
                checkpoint.freezeTransitionEnter
                    ? "freeze_enter"
                    : (checkpoint.freezeTransitionExit ? "freeze_exit" : "freeze_sample"));
        }

        burstDebtDecision = evaluate_burst_debt_runtime(
            transaction,
            cfg,
            burstDecision,
            criticalCurrentFrame,
            requestBytes);
        if (burstDebtDecision.sampled || JTRACE_ENABLED(3)) {
            trace_burst_debt_decision(
                transaction,
                commandName,
                lane,
                pressureState,
                criticalCurrentFrame,
                requestBytes,
                cfg,
                burstDebtDecision);
        }

        fill_tier_budget_snapshot(
            transaction,
            resources,
            cfg,
            checkpoint.memory,
            tierBudget);
        tierBudgetValid = true;
        if (requestBytes > 0 && tierBudget.anyOverTarget) {
            outRequestReclaimPass = true;
        }

        std::uint64_t scratchTrimmedEntries = 0;
        std::uint64_t graphEvictedEntries = 0;
        bool didTierPretrim = false;
        run_tier_target_pretrim(
            transaction,
            resources,
            commandName,
            lane,
            pressureState,
            tierBudget,
            didTierPretrim,
            scratchTrimmedEntries,
            graphEvictedEntries);
        if (didTierPretrim) {
            outRequestReclaimPass = outRequestReclaimPass || (requestBytes > 0);
        }

        if (tierBudget.anyOverTarget || didTierPretrim || JTRACE_ENABLED(3)) {
            trace_tier_budget_event(
                transaction,
                commandName,
                lane,
                pressureState,
                tierBudget,
                requestBytes,
                criticalCurrentFrame,
                outRequestReclaimPass,
                scratchTrimmedEntries,
                graphEvictedEntries,
                didTierPretrim ? "tier_target_pretrim" : "tier_target_observation");
        }
    }
    else if (JTRACE_ENABLED(3)) {
        trace_tier_budget_event(
            transaction,
            commandName,
            lane,
            pressureState,
            tierBudget,
            requestBytes,
            criticalCurrentFrame,
            outRequestReclaimPass,
            0,
            0,
            "tier_budget_policy_disabled");
    }

    if (nonCritical && pressureEnabled && freezeBelowReserve && requestBytes > 0) {
        global_state().opportunisticFreezeDenyEvents.fetch_add(1, std::memory_order_relaxed);
        trace_opportunistic_freeze_event(
            transaction,
            commandName,
            pressureState,
            checkpointValid ? checkpoint.input.effectiveHeadroomBytes : 0,
            checkpointValid ? checkpoint.decision.effectiveReserveBytes : 0,
            true,
            false,
            criticalCurrentFrame,
            "below_effective_reserve_noncritical");
        outError = std::string("pressure_shed_noncritical: lane=")
            + to_cstr(lane)
            + " state="
            + to_cstr(pressureState);
        return false;
    }

    if (nonCritical &&
        burstDebtDecision.enabled &&
        burstDebtDecision.throttled &&
        requestBytes > 0) {
        outError = std::string("burst_debt_throttle_noncritical: lane=")
            + to_cstr(lane)
            + " debt_pct=" + std::to_string(static_cast<unsigned long long>(burstDebtDecision.debtAfterPct))
            + " max_debt_pct=" + std::to_string(static_cast<unsigned long long>(cfg.maxBurstDebtPct));
        return false;
    }

    if (nonCritical &&
        (pressureState == PressureState::Critical ||
         pressureState == PressureState::Emergency)) {
        const char* denyReason = "deny_non_critical_growth";
        if (tierBudgetValid && tierBudget.anyOverTarget) {
            denyReason = "deny_non_critical_growth_tier_over_target";
        }
        if (pressureState == PressureState::Emergency) {
            if (lane == PressureLane::Upload) {
                global_state().uploadEmergencyShedDenials.fetch_add(1, std::memory_order_relaxed);
                denyReason = "emergency_shed_upload_precedence";
            }
            else {
                global_state().builderEmergencyShedDenials.fetch_add(1, std::memory_order_relaxed);
                denyReason = "emergency_shed_builder_precedence";
            }
        }
        trace_emergency_shed_action(
            transaction,
            commandName,
            lane,
            pressureState,
            requestBytes,
            criticalCurrentFrame,
            false,
            uploadBytesInFlight,
            uploadCapBytes,
            denyReason);
        outError = std::string("pressure_shed_noncritical: lane=")
            + to_cstr(lane)
            + " state="
            + to_cstr(pressureState);
        return false;
    }

    if (lane == PressureLane::Upload && nonCritical && uploadCapSaturated) {
        global_state().copyComputeGuardShedEvents.fetch_add(1, std::memory_order_relaxed);
        trace_copy_compute_guard(
            transaction,
            commandName,
            pressureState,
            uploadBytesInFlight,
            uploadCapBytes,
            criticalCurrentFrame,
            false,
            "upload_cap_saturated_noncritical");
        outError = "pressure_copy_compute_guard: upload_cap_saturated_noncritical";
        return false;
    }

    if (pressureState == PressureState::Emergency) {
        trace_emergency_shed_action(
            transaction,
            commandName,
            lane,
            pressureState,
            requestBytes,
            criticalCurrentFrame,
            true,
            uploadBytesInFlight,
            uploadCapBytes,
            criticalCurrentFrame ? "allow_critical_progress" : "allow_noncritical_progress");
    }

    if (lane == PressureLane::Upload && uploadCapSaturated) {
        trace_copy_compute_guard(
            transaction,
            commandName,
            pressureState,
            uploadBytesInFlight,
            uploadCapBytes,
            criticalCurrentFrame,
            true,
            criticalCurrentFrame
                ? "critical_upload_allowed_despite_saturation"
                : "upload_allowed_below_shed_threshold");
    }
    return true;
}

bool tier_circuit_begin_attempt(
    const SubmissionTransaction& transaction,
    const char* commandName,
    ResourceTier tier,
    bool blocksAdmission,
    TierCircuitAttempt& outAttempt,
    std::string& outError) {
    outAttempt = TierCircuitAttempt{};
    outError.clear();

    outAttempt.contextKey = transaction.snapshot.deviceContextKey;
    outAttempt.tier = tier;
    outAttempt.started = true;

    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    if (!tier_circuit_policy_enabled(cfg)) {
        return true;
    }

    const std::uint32_t threshold = std::max<std::uint32_t>(1u, cfg.tierErrorThreshold);
    const std::uint32_t windowMs = std::max<std::uint32_t>(1u, cfg.tierErrorWindowMs);
    const std::uint32_t openMs = std::max<std::uint32_t>(1u, cfg.tierCircuitOpenMs);
    const std::uint64_t nowMs = monotonic_time_ms();

    TierCircuitPolicyState& state = tier_circuit_policy_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    TierCircuitContextState& contextState = state.byContext[transaction.snapshot.deviceContextKey];
    TierCircuitTierState& tierState = contextState.tiers[tier_circuit_index(tier)];

    if (tierState.state == TierCircuitState::Open) {
        const std::uint64_t elapsedMs = (nowMs > tierState.openedAtMs) ? (nowMs - tierState.openedAtMs) : 0;
        if (elapsedMs >= static_cast<std::uint64_t>(openMs)) {
            const TierCircuitState previousState = tierState.state;
            tierState.state = TierCircuitState::HalfOpen;
            tierState.probeInFlight = false;
            global_state().tierCircuitHalfOpenEvents.fetch_add(1, std::memory_order_relaxed);
            trace_tier_circuit_event(
                transaction,
                commandName,
                tier,
                previousState,
                tierState.state,
                true,
                true,
                false,
                tierState.windowErrors,
                threshold,
                windowMs,
                openMs,
                0,
                "open_elapsed_half_open");
        }
    }

    if (tierState.state == TierCircuitState::Open && blocksAdmission) {
        const std::uint64_t elapsedMs = (nowMs > tierState.openedAtMs) ? (nowMs - tierState.openedAtMs) : 0;
        const std::uint64_t openRemainingMs = (elapsedMs >= static_cast<std::uint64_t>(openMs))
            ? 0
            : (static_cast<std::uint64_t>(openMs) - elapsedMs);
        global_state().tierCircuitBlockedEvents.fetch_add(1, std::memory_order_relaxed);
        trace_tier_circuit_event(
            transaction,
            commandName,
            tier,
            tierState.state,
            tierState.state,
            false,
            false,
            false,
            tierState.windowErrors,
            threshold,
            windowMs,
            openMs,
            openRemainingMs,
            "open_blocked");
        outError = std::string("tier_circuit_open: tier=") + to_cstr(tier);
        return false;
    }

    if (tierState.state == TierCircuitState::HalfOpen) {
        if (tierState.probeInFlight && blocksAdmission) {
            global_state().tierCircuitBlockedEvents.fetch_add(1, std::memory_order_relaxed);
            trace_tier_circuit_event(
                transaction,
                commandName,
                tier,
                tierState.state,
                tierState.state,
                false,
                false,
                false,
                tierState.windowErrors,
                threshold,
                windowMs,
                openMs,
                0,
                "half_open_probe_in_flight");
            outError = std::string("tier_circuit_half_open_busy: tier=") + to_cstr(tier);
            return false;
        }
        if (!tierState.probeInFlight) {
            tierState.probeInFlight = true;
            outAttempt.probe = true;
            trace_tier_circuit_event(
                transaction,
                commandName,
                tier,
                tierState.state,
                tierState.state,
                false,
                true,
                true,
                tierState.windowErrors,
                threshold,
                windowMs,
                openMs,
                0,
                "half_open_probe_start");
        }
    }

    return true;
}

void tier_circuit_cancel_attempt(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const TierCircuitAttempt& attempt,
    const char* reason) {
    if (!attempt.started) {
        return;
    }
    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    if (!tier_circuit_policy_enabled(cfg)) {
        return;
    }

    const std::uint32_t threshold = std::max<std::uint32_t>(1u, cfg.tierErrorThreshold);
    const std::uint32_t windowMs = std::max<std::uint32_t>(1u, cfg.tierErrorWindowMs);
    const std::uint32_t openMs = std::max<std::uint32_t>(1u, cfg.tierCircuitOpenMs);

    TierCircuitPolicyState& state = tier_circuit_policy_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto contextIt = state.byContext.find(attempt.contextKey);
    if (contextIt == state.byContext.end()) {
        return;
    }
    TierCircuitTierState& tierState = contextIt->second.tiers[tier_circuit_index(attempt.tier)];
    if (tierState.state == TierCircuitState::HalfOpen && tierState.probeInFlight) {
        tierState.probeInFlight = false;
        trace_tier_circuit_event(
            transaction,
            commandName,
            attempt.tier,
            tierState.state,
            tierState.state,
            false,
            true,
            false,
            tierState.windowErrors,
            threshold,
            windowMs,
            openMs,
            0,
            reason ? reason : "attempt_cancelled");
    }
}

void tier_circuit_record_outcome(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const TierCircuitAttempt& attempt,
    bool success,
    const char* reason) {
    if (!attempt.started) {
        return;
    }

    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    if (!tier_circuit_policy_enabled(cfg)) {
        return;
    }

    const std::uint32_t threshold = std::max<std::uint32_t>(1u, cfg.tierErrorThreshold);
    const std::uint32_t windowMs = std::max<std::uint32_t>(1u, cfg.tierErrorWindowMs);
    const std::uint32_t openMs = std::max<std::uint32_t>(1u, cfg.tierCircuitOpenMs);
    const std::uint64_t nowMs = monotonic_time_ms();

    TierCircuitPolicyState& state = tier_circuit_policy_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto contextIt = state.byContext.find(attempt.contextKey);
    if (contextIt == state.byContext.end()) {
        return;
    }

    TierCircuitTierState& tierState = contextIt->second.tiers[tier_circuit_index(attempt.tier)];
    auto reset_window_if_stale = [&]() {
        if (tierState.windowStartMs == 0 ||
            nowMs < tierState.windowStartMs ||
            (nowMs - tierState.windowStartMs) >= static_cast<std::uint64_t>(windowMs)) {
            tierState.windowStartMs = nowMs;
            tierState.windowErrors = 0;
        }
    };

    if (success) {
        if (tierState.state == TierCircuitState::HalfOpen) {
            const TierCircuitState previousState = tierState.state;
            tierState.state = TierCircuitState::Closed;
            tierState.windowStartMs = 0;
            tierState.windowErrors = 0;
            tierState.openedAtMs = 0;
            tierState.probeInFlight = false;
            global_state().tierCircuitCloseEvents.fetch_add(1, std::memory_order_relaxed);
            trace_tier_circuit_event(
                transaction,
                commandName,
                attempt.tier,
                previousState,
                tierState.state,
                true,
                true,
                attempt.probe,
                tierState.windowErrors,
                threshold,
                windowMs,
                openMs,
                0,
                reason ? reason : "half_open_probe_success");
            return;
        }

        if (tierState.state == TierCircuitState::Closed &&
            tierState.windowStartMs > 0 &&
            (nowMs > tierState.windowStartMs) &&
            (nowMs - tierState.windowStartMs) >= static_cast<std::uint64_t>(windowMs)) {
            tierState.windowStartMs = 0;
            tierState.windowErrors = 0;
        }
        return;
    }

    if (tierState.state == TierCircuitState::HalfOpen) {
        const TierCircuitState previousState = tierState.state;
        tierState.state = TierCircuitState::Open;
        tierState.windowStartMs = nowMs;
        tierState.windowErrors = threshold;
        tierState.openedAtMs = nowMs;
        tierState.probeInFlight = false;
        global_state().tierCircuitOpenEvents.fetch_add(1, std::memory_order_relaxed);
        trace_tier_circuit_event(
            transaction,
            commandName,
            attempt.tier,
            previousState,
            tierState.state,
            true,
            true,
            attempt.probe,
            tierState.windowErrors,
            threshold,
            windowMs,
            openMs,
            0,
            reason ? reason : "half_open_probe_failed");
        return;
    }

    if (tierState.state == TierCircuitState::Closed) {
        reset_window_if_stale();
        if (tierState.windowErrors < std::numeric_limits<std::uint32_t>::max()) {
            ++tierState.windowErrors;
        }
        if (tierState.windowErrors >= threshold) {
            const TierCircuitState previousState = tierState.state;
            tierState.state = TierCircuitState::Open;
            tierState.openedAtMs = nowMs;
            tierState.probeInFlight = false;
            global_state().tierCircuitOpenEvents.fetch_add(1, std::memory_order_relaxed);
            trace_tier_circuit_event(
                transaction,
                commandName,
                attempt.tier,
                previousState,
                tierState.state,
                true,
                true,
                attempt.probe,
                tierState.windowErrors,
                threshold,
                windowMs,
                openMs,
                0,
                reason ? reason : "error_threshold_reached");
        }
        return;
    }

    if (tierState.state == TierCircuitState::Open) {
        reset_window_if_stale();
        if (tierState.windowErrors < std::numeric_limits<std::uint32_t>::max()) {
            ++tierState.windowErrors;
        }
        tierState.openedAtMs = nowMs;
    }
}

void tier_circuit_retire_context(const DeviceContextKey& key) noexcept {
    TierCircuitPolicyState& state = tier_circuit_policy_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.byContext.erase(key);
}

void pressure_policy_retire_context(const DeviceContextKey& key) noexcept {
    PressurePolicyState& state = pressure_policy_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.byContext.erase(key);
}

void admission_churn_retire_context(const DeviceContextKey& key) noexcept {
    AdmissionChurnPolicyState& state = admission_churn_policy_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.byContext.erase(key);
}

void optional_heuristic_trace_retire_context(const DeviceContextKey& key) noexcept {
    OptionalHeuristicTraceState& state = optional_heuristic_trace_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.byContext.erase(key);
}
