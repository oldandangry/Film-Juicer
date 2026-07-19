// Cuda/ResourceManager/JuicerCudaManagerRegistry.cpp

#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#include "Cuda/ResourceManager/JuicerCudaResourceCore.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "Logging.h"

namespace JuicerCuda {
    namespace ResourceManager {
        namespace {

            template <typename TraceAction>
            void run_registry_trace_noexcept(TraceAction&& action) noexcept {
                try {
                    action();
                } catch (...) {
                    JuicerLogging::discard_current_exception();
                }
            }

            struct RegistryEntry {
                RegistryHandle handle{};
                ContextLifecycleState lifecycleState = ContextLifecycleState::Unbound;
                std::uint64_t lifecycleSinceMs = 0;
                std::uint64_t createOrder = 0;
                std::uint64_t lastTouchedMs = 0;
                std::uint64_t activeSubmissionCount = 0;
                std::uint64_t registryGeneration = 1;
                std::uint64_t contextEpoch = 1;
            };

            struct RegistryState {
                std::mutex mutex;
                std::atomic<std::uint64_t> nextHandle{1};
                std::atomic<std::uint64_t> nextCreateOrder{1};
                std::uint64_t lastIdleReapScanMs = 0;
                std::unordered_map<DeviceContextKey, RegistryEntry, DeviceContextKeyHash> byDeviceContext;
                std::unordered_map<std::uint64_t, DeviceContextKey> keyByHandle;
            };

            struct ReapCandidate {
                DeviceContextKey key{};
                RegistryEntry entry{};
                std::uint64_t idleMs = 0;
            };

            constexpr std::uint64_t kMetadataMutationQueueDepthLimit = 64ull;
            thread_local std::uint32_t gMutationThreadDepth = 0;
            thread_local std::uint64_t gMutationThreadTicket = 0;
            thread_local std::uint64_t gMutationThreadBeginCount = 0;

            struct MetadataMutationLane {
                std::mutex mutex;
                std::condition_variable cv;
                std::uint64_t nextTicket = 1;
                std::uint64_t servingTicket = 1;
                std::uint64_t activeTicket = 0;
                std::thread::id ownerThread;
                std::uint32_t ownerDepth = 0;
            };

            struct MetadataMutationLaneDirectory {
                std::mutex mutex;
                std::shared_ptr<MetadataMutationLane> fallbackLane = std::make_shared<MetadataMutationLane>();
                std::unordered_map<DeviceContextKey, std::shared_ptr<MetadataMutationLane>, DeviceContextKeyHash> byManagerKey;
            };

            struct LaneAcquireResult {
                std::uint64_t ticket = 0;
                std::uint64_t waitedMs = 0;
                bool observedWait = false;
                bool observedBackpressure = false;
                bool reentrant = false;
            };

            MetadataMutationLaneDirectory& mutation_lane_directory() {
                static MetadataMutationLaneDirectory directory{};
                return directory;
            }

            std::mutex& mutation_sequence_mutex() {
                static std::mutex mutex;
                return mutex;
            }

            std::uint64_t& mutation_last_issued_sequence() {
                static std::uint64_t sequence = 0;
                return sequence;
            }

            inline std::uint64_t lane_depth_nolock(const MetadataMutationLane& lane) noexcept {
                if (lane.nextTicket < lane.servingTicket) {
                    return 0;
                }
                return lane.nextTicket - lane.servingTicket;
            }

            inline std::uint64_t elapsed_ms(const std::chrono::steady_clock::time_point& start) noexcept {
                const auto delta = std::chrono::steady_clock::now() - start;
                return static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(delta).count());
            }

#if JUICER_DIAGNOSTICS_COMPILED
            const char* registry_trace_or_unknown(const char* value) noexcept {
                if (value && value[0] != '\0') {
                    return value;
                }
                return "unknown";
            }

            const char* registry_trace_or_unspecified(const char* value) noexcept {
                if (value && value[0] != '\0') {
                    return value;
                }
                return "unspecified";
            }
#endif

            const char* registry_trace_or(const char* value, const char* fallback) noexcept {
                if (value) {
                    return value;
                }
                return fallback;
            }

            const char* registry_bool_reason(bool value, const char* whenTrue, const char* whenFalse) noexcept {
                if (value) {
                    return whenTrue;
                }
                return whenFalse;
            }

#if JUICER_DIAGNOSTICS_COMPILED
            std::uint32_t registry_bool_u32(bool value) noexcept {
                if (value) {
                    return 1u;
                }
                return 0u;
            }
#endif

            const char* registry_mutation_begin_reason(bool orderOk, bool reentrant) noexcept {
                if (!orderOk) {
                    return "sequence_order_violation";
                }
                if (reentrant) {
                    return "ok_reentrant";
                }
                return "ok";
            }

#if JUICER_DIAGNOSTICS_COMPILED
            const char* registry_entry_lifecycle_name(const RegistryEntry* entry) noexcept {
                if (entry) {
                    return to_cstr(entry->lifecycleState);
                }
                return "Missing";
            }

            int registry_device_id_or_default(const DeviceContextKey* key) noexcept {
                if (key) {
                    return key->deviceId;
                }
                return -1;
            }

            std::uintptr_t registry_context_bits_or_zero(const DeviceContextKey* key) noexcept {
                if (key) {
                    return reinterpret_cast<std::uintptr_t>(key->contextOpaque);
                }
                return 0;
            }

            std::string registry_trace_event_prefix(const char* eventName) {
                return std::string("event=") + registry_trace_or_unknown(eventName);
            }

            std::string registry_trace_device_context_fields(const DeviceContextKey* key) {
                const int deviceId = registry_device_id_or_default(key);
                const std::uintptr_t contextBits = registry_context_bits_or_zero(key);
                return std::string(" device_id=") + std::to_string(deviceId) + " context=" + std::to_string(contextBits);
            }

            std::string registry_trace_handle_prefix(RegistryHandle handle, const DeviceContextKey& key) {
                return std::string("handle=") + std::to_string(handle.value) + registry_trace_device_context_fields(&key);
            }
#endif

            const char* registry_retire_reason_name(RegistryRetireReason reason) noexcept {
                switch (reason) {
                    case RegistryRetireReason::ContextReset:
                        return "context_reset";
                    case RegistryRetireReason::Idle:
                        return "idle";
                    default:
                        return "unknown";
                }
            }

            const char* registry_lifecycle_timeout_reason(ContextLifecycleState observedState) noexcept {
                if (observedState == ContextLifecycleState::Draining) {
                    return "drain_timeout";
                }
                return "rebind_timeout";
            }

            std::shared_ptr<MetadataMutationLane> resolve_mutation_lane(const DeviceContextKey* managerKey) {
                MetadataMutationLaneDirectory& directory = mutation_lane_directory();
                if (!managerKey || managerKey->deviceId < 0) {
                    return directory.fallbackLane;
                }

                std::lock_guard<std::mutex> lock(directory.mutex);
                auto it = directory.byManagerKey.find(*managerKey);
                if (it != directory.byManagerKey.end()) {
                    return it->second;
                }

                std::shared_ptr<MetadataMutationLane> lane = std::make_shared<MetadataMutationLane>();
                directory.byManagerKey.emplace(*managerKey, lane);
                return lane;
            }

            std::shared_ptr<MetadataMutationLane> find_mutation_lane_noexcept(const DeviceContextKey* managerKey) noexcept {
                try {
                    MetadataMutationLaneDirectory& directory = mutation_lane_directory();
                    if (!managerKey || managerKey->deviceId < 0) {
                        return directory.fallbackLane;
                    }

                    std::lock_guard<std::mutex> lock(directory.mutex);
                    auto it = directory.byManagerKey.find(*managerKey);
                    if (it == directory.byManagerKey.end()) {
                        return nullptr;
                    }
                    return it->second;
                } catch (...) {
                    JuicerLogging::discard_current_exception();
                    return nullptr;
                }
            }

            void reset_scope(MetadataMutationScope& scope) noexcept {
                scope.sequence = 0;
                scope.queueTicket = 0;
                scope.managerKey = DeviceContextKey{};
                scope.hasManagerKey = false;
                scope.active = false;
            }

            void reset_thread_mutation_context() noexcept {
                gMutationThreadDepth = 0;
                gMutationThreadTicket = 0;
            }

            DeviceContextKey registry_manager_key_or_default(
                bool hasManagerKey,
                const DeviceContextKey* managerKey) noexcept {
                if (hasManagerKey && managerKey) {
                    return *managerKey;
                }
                return DeviceContextKey{};
            }

            const DeviceContextKey* registry_manager_key_ptr_or_null(
                bool hasManagerKey,
                const DeviceContextKey* managerKey) noexcept {
                if (hasManagerKey) {
                    return managerKey;
                }
                return nullptr;
            }

            struct MutationRejectTrace {
                const char* action = nullptr;
                const char* stage = nullptr;
                const char* reason = nullptr;
                std::uint64_t sequence = 0;
                std::uint64_t expectedSequence = 0;
                std::uint64_t queueTicket = 0;
                std::uint64_t queueDepth = 0;
                std::uint64_t waitedMs = 0;
                bool accepted = false;
            };

            void trace_mutation_reject(const MutationRejectTrace& trace) noexcept {
                telemetry_record_metadata_mutation_reject();
                telemetry_record_metadata_queue_reject();
                telemetry_trace_metadata_queue(
                    TelemetryMetadataQueueTrace{
                        .eventName = "reject",
                        .stage = trace.stage,
                        .reason = trace.reason,
                        .ticket = trace.queueTicket,
                        .depth = trace.queueDepth,
                        .waitedMs = trace.waitedMs,
                        .accepted = trace.accepted});
                telemetry_trace_metadata_mutation(
                    TelemetryMetadataMutationTrace{
                        .phase = trace.action,
                        .stage = trace.stage,
                        .reason = trace.reason,
                        .sequence = trace.sequence,
                        .expectedSequence = trace.expectedSequence});
            }

            void note_queue_wait_if_needed(bool& observedWait) noexcept {
                if (!observedWait) {
                    observedWait = true;
                    telemetry_record_metadata_queue_wait();
                }
            }

            bool acquire_lane_ticket(
                MetadataMutationLane& lane,
                const std::thread::id& currentThread,
                const char* stage,
                LaneAcquireResult& out) {
                std::unique_lock<std::mutex> lock(lane.mutex);
                if (lane.ownerDepth > 0 && lane.ownerThread == currentThread) {
                    out.reentrant = true;
                    ++lane.ownerDepth;
                    out.ticket = lane.activeTicket;
                    if (out.ticket == 0) {
                        out.ticket = lane.servingTicket;
                        lane.activeTicket = out.ticket;
                    }
                    telemetry_trace_metadata_queue(
                        TelemetryMetadataQueueTrace{
                            .eventName = "reenter",
                            .stage = stage,
                            .reason = "owner_reentrant",
                            .ticket = out.ticket,
                            .depth = lane_depth_nolock(lane),
                            .accepted = true});
                    return true;
                }

                while (lane_depth_nolock(lane) >= kMetadataMutationQueueDepthLimit) {
                    if (!out.observedBackpressure) {
                        out.observedBackpressure = true;
                        telemetry_record_metadata_queue_backpressure();
                        telemetry_trace_metadata_queue(
                            TelemetryMetadataQueueTrace{
                                .eventName = "backpressure",
                                .stage = stage,
                                .reason = "depth_limit",
                                .depth = lane_depth_nolock(lane),
                                .waitedMs = out.waitedMs,
                                .accepted = true});
                    }
                    note_queue_wait_if_needed(out.observedWait);
                    // Event-driven wait avoids the prior 1 ms polling loop on queue pressure paths.
                    const auto waitStart = std::chrono::steady_clock::now();
                    lane.cv.wait(lock, [&lane]() {
                        return lane_depth_nolock(lane) < kMetadataMutationQueueDepthLimit;
                    });
                    out.waitedMs += elapsed_ms(waitStart);
                }

                out.ticket = lane.nextTicket++;
                if (out.ticket == 0) {
                    trace_mutation_reject(
                        MutationRejectTrace{
                            .action = "begin",
                            .stage = stage,
                            .reason = "ticket_overflow",
                            .queueDepth = lane_depth_nolock(lane),
                            .waitedMs = out.waitedMs});
                    return false;
                }

                telemetry_record_metadata_queue_enqueue();
                const std::uint64_t depthAfterEnqueue = lane_depth_nolock(lane);
                telemetry_note_metadata_queue_depth(depthAfterEnqueue);
                telemetry_trace_metadata_queue(
                    TelemetryMetadataQueueTrace{
                        .eventName = "enqueue",
                        .stage = stage,
                        .reason = registry_bool_reason(out.observedBackpressure, "accepted_after_backpressure", "accepted"),
                        .ticket = out.ticket,
                        .depth = depthAfterEnqueue,
                        .waitedMs = out.waitedMs,
                        .accepted = true});

                while (out.ticket != lane.servingTicket || lane.ownerDepth != 0) {
                    note_queue_wait_if_needed(out.observedWait);
                    const auto waitStart = std::chrono::steady_clock::now();
                    lane.cv.wait(lock, [&lane, &out]() {
                        return out.ticket == lane.servingTicket && lane.ownerDepth == 0;
                    });
                    out.waitedMs += elapsed_ms(waitStart);
                }

                lane.ownerThread = currentThread;
                lane.ownerDepth = 1;
                lane.activeTicket = out.ticket;
                telemetry_record_metadata_queue_dequeue();
                telemetry_trace_metadata_queue(
                    TelemetryMetadataQueueTrace{
                        .eventName = "dequeue",
                        .stage = stage,
                        .reason = registry_bool_reason(out.observedWait, "turn_wait", "immediate"),
                        .ticket = out.ticket,
                        .depth = lane_depth_nolock(lane),
                        .waitedMs = out.waitedMs,
                        .accepted = true});
                return true;
            }

            RegistryState& registry_state() {
                static RegistryState state{};
                return state;
            }

            const ResourceManagerConfigEffective& registry_policy_config() noexcept {
                static const ResourceManagerConfigEffective cfg = sanitize_config(ResourceManagerConfigRaw{});
                return cfg;
            }

            inline std::uint64_t monotonic_time_ms() noexcept {
                return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                                      std::chrono::steady_clock::now().time_since_epoch())
                                                      .count());
            }

            inline std::uint64_t saturating_elapsed_ms(std::uint64_t nowMs, std::uint64_t thenMs) noexcept {
                if (nowMs >= thenMs) {
                    return nowMs - thenMs;
                }
                return 0ull;
            }

            void publish_registry_live_count(std::size_t count) noexcept {
                global_state().registryLiveManagers.store(static_cast<std::uint64_t>(count), std::memory_order_relaxed);
            }

            inline std::uint64_t next_nonzero_counter(std::atomic<std::uint64_t>& counter) noexcept {
                std::uint64_t value = counter.fetch_add(1, std::memory_order_relaxed);
                if (value == 0) {
                    value = counter.fetch_add(1, std::memory_order_relaxed);
                }
                return value;
            }

            inline void assign_entry_generations(RegistryEntry& entry) noexcept {
                ResourceManagerState& rmState = global_state();
                entry.registryGeneration = next_nonzero_counter(rmState.registryGeneration);
                entry.contextEpoch = next_nonzero_counter(rmState.contextEpoch);
            }

#if JUICER_DIAGNOSTICS_COMPILED
            std::uint64_t registry_entry_handle_value_or_zero(const RegistryEntry* entry) noexcept {
                if (entry) {
                    return entry->handle.value;
                }
                return 0;
            }

            std::uint64_t registry_entry_active_submissions_or_zero(const RegistryEntry* entry) noexcept {
                if (entry) {
                    return entry->activeSubmissionCount;
                }
                return 0;
            }
#endif

            struct RegistryEventTrace {
                const DeviceContextKey* key = nullptr;
                const RegistryEntry* entry = nullptr;
                const char* eventName = nullptr;
                const char* reason = nullptr;
                std::uint64_t idleMs = 0;
                std::uint64_t liveManagers = 0;
                std::uint64_t maxLiveManagers = 0;
                std::uint64_t reapEvents = 0;
                bool accepted = false;
            };

            void trace_registry_event(const RegistryEventTrace& trace) {
#if JUICER_DIAGNOSTICS_COMPILED
                if (!JTRACE_ENABLED(2)) {
                    return;
                }
                const std::uint64_t handleValue = registry_entry_handle_value_or_zero(trace.entry);
                const char* lifecycle = registry_entry_lifecycle_name(trace.entry);
                const std::uint64_t activeSubmissions = registry_entry_active_submissions_or_zero(trace.entry);
                const std::string msg =
                    registry_trace_event_prefix(trace.eventName) +
                    " accepted=" + std::to_string(registry_bool_u32(trace.accepted)) +
                    " reason=" + registry_trace_or_unspecified(trace.reason) +
                    " handle=" + std::to_string(handleValue) +
                    registry_trace_device_context_fields(trace.key) +
                    " lifecycle=" + lifecycle +
                    " active_submissions=" + std::to_string(activeSubmissions) +
                    " idle_ms=" + std::to_string(trace.idleMs) +
                    " live_managers=" + std::to_string(trace.liveManagers) +
                    " max_live_managers=" + std::to_string(trace.maxLiveManagers) +
                    " reap_events=" + std::to_string(trace.reapEvents);
                JTRACE_LEVEL(2, "MSREG", msg);
#endif
            }

            void trace_registry_event_current(
                const DeviceContextKey* key,
                const RegistryEntry* entry,
                const char* eventName,
                bool accepted,
                const char* reason,
                std::uint64_t idleMs) noexcept {
                ResourceManagerState& rmState = global_state();
                run_registry_trace_noexcept([&]() {
                    trace_registry_event(
                        RegistryEventTrace{
                            .key = key,
                            .entry = entry,
                            .eventName = eventName,
                            .reason = reason,
                            .idleMs = idleMs,
                            .liveManagers = rmState.registryLiveManagers.load(std::memory_order_relaxed),
                            .maxLiveManagers = static_cast<std::uint64_t>(registry_policy_config().maxLiveManagersPerProcess),
                            .reapEvents = rmState.registryReapEvents.load(std::memory_order_relaxed),
                            .accepted = accepted});
                });
            }

            void trace_registry_missing_entry(const DeviceContextKey& key, const char* eventName) noexcept {
                trace_registry_event_current(
                    &key,
                    nullptr,
                    eventName,
                    false,
                    "missing_registry_entry",
                    0);
            }

            void trace_reap_skip_event(const char* reason) noexcept {
                trace_registry_event_current(
                    nullptr,
                    nullptr,
                    "reap_skip",
                    false,
                    reason,
                    0);
            }

            bool is_legal_transition(ContextLifecycleState from, ContextLifecycleState to) noexcept {
                if (from == to) {
                    return true;
                }
                switch (from) {
                    case ContextLifecycleState::Unbound:
                        return to == ContextLifecycleState::Binding || to == ContextLifecycleState::Retired;
                    case ContextLifecycleState::Binding:
                        return to == ContextLifecycleState::Active || to == ContextLifecycleState::Retired;
                    case ContextLifecycleState::Active:
                        return to == ContextLifecycleState::Freezing || to == ContextLifecycleState::Retired;
                    case ContextLifecycleState::Freezing:
                        return to == ContextLifecycleState::Draining ||
                               to == ContextLifecycleState::Active ||
                               to == ContextLifecycleState::Retired;
                    case ContextLifecycleState::Draining:
                        return to == ContextLifecycleState::Rebinding || to == ContextLifecycleState::Retired;
                    case ContextLifecycleState::Rebinding:
                        return to == ContextLifecycleState::Active || to == ContextLifecycleState::Retired;
                    case ContextLifecycleState::Retired:
                    default:
                        return false;
                }
            }

            struct LifecycleTransitionTrace {
                const DeviceContextKey* key = nullptr;
                const char* reason = nullptr;
                RegistryHandle handle{};
                ContextLifecycleState from = ContextLifecycleState::Unbound;
                ContextLifecycleState to = ContextLifecycleState::Unbound;
                bool accepted = false;
            };

            void trace_lifecycle_transition(const LifecycleTransitionTrace& trace) {
#if JUICER_DIAGNOSTICS_COMPILED
                if (!JTRACE_ENABLED(1)) {
                    return;
                }
                const DeviceContextKey key = trace.key ? *trace.key : DeviceContextKey{};
                const std::string msg = registry_trace_handle_prefix(trace.handle, key) + " from=" + to_cstr(trace.from) + " to=" + to_cstr(trace.to) + " accepted=" + std::to_string(registry_bool_u32(trace.accepted)) + " reason=" + registry_trace_or_unspecified(trace.reason);
                JTRACE("MSLCY", msg);
#endif
            }

            struct LifecycleBumpTrace {
                const DeviceContextKey* key = nullptr;
                const char* reason = nullptr;
                RegistryHandle handle{};
                std::uint64_t previousRegistryGeneration = 0;
                std::uint64_t newRegistryGeneration = 0;
                std::uint64_t previousContextEpoch = 0;
                std::uint64_t newContextEpoch = 0;
                bool accepted = false;
            };

            void trace_lifecycle_bump(const LifecycleBumpTrace& trace) {
#if JUICER_DIAGNOSTICS_COMPILED
                if (!JTRACE_ENABLED(1)) {
                    return;
                }
                const DeviceContextKey key = trace.key ? *trace.key : DeviceContextKey{};
                const std::string msg = registry_trace_handle_prefix(trace.handle, key) + " action=bump" + " accepted=" + std::to_string(registry_bool_u32(trace.accepted)) + " prev_registry_generation=" + std::to_string(trace.previousRegistryGeneration) + " new_registry_generation=" + std::to_string(trace.newRegistryGeneration) + " prev_context_epoch=" + std::to_string(trace.previousContextEpoch) + " new_context_epoch=" + std::to_string(trace.newContextEpoch) + " reason=" + registry_trace_or_unspecified(trace.reason);
                JTRACE("MSLCY", msg);
#endif
            }

            constexpr std::uint64_t kLifecycleTimeoutMinMs = 2000ull;
            constexpr std::uint64_t kRebindingTimeoutMinMs = 1000ull;

            bool lifecycle_state_allowed_for_stage(ContextLifecycleState state, bool allowNonActiveRelease) noexcept {
                if (allowNonActiveRelease) {
                    return state != ContextLifecycleState::Unbound;
                }
                return state == ContextLifecycleState::Active;
            }

            std::uint64_t lifecycle_timeout_ms_for_state(
                ContextLifecycleState state,
                const ResourceManagerConfigEffective& cfg) noexcept {
                const std::uint64_t baseMs = std::max<std::uint64_t>(
                    static_cast<std::uint64_t>(cfg.managerIdleReapMs),
                    kLifecycleTimeoutMinMs);
                switch (state) {
                    case ContextLifecycleState::Draining:
                        return baseMs;
                    case ContextLifecycleState::Rebinding:
                        return std::max<std::uint64_t>(baseMs / 2ull, kRebindingTimeoutMinMs);
                    default:
                        return 0ull;
                }
            }

            void bump_registry_epoch_locked(
                const DeviceContextKey& key,
                RegistryEntry& entry,
                bool accepted,
                const char* reason) noexcept {
                const std::uint64_t prevRegistryGeneration = entry.registryGeneration;
                const std::uint64_t prevContextEpoch = entry.contextEpoch;
                assign_entry_generations(entry);
                run_registry_trace_noexcept([&]() {
                    trace_lifecycle_bump(
                        LifecycleBumpTrace{
                            .key = &key,
                            .reason = reason,
                            .handle = entry.handle,
                            .previousRegistryGeneration = prevRegistryGeneration,
                            .newRegistryGeneration = entry.registryGeneration,
                            .previousContextEpoch = prevContextEpoch,
                            .newContextEpoch = entry.contextEpoch,
                            .accepted = accepted});
                });
            }

            struct LifecycleTimeoutTrace {
                const DeviceContextKey* key = nullptr;
                const char* reason = nullptr;
                RegistryHandle handle{};
                ContextLifecycleState observedState = ContextLifecycleState::Unbound;
                std::uint64_t stateAgeMs = 0;
                std::uint64_t timeoutMs = 0;
                bool escalated = false;
            };

            void trace_lifecycle_timeout(const LifecycleTimeoutTrace& trace) noexcept {
#if JUICER_DIAGNOSTICS_COMPILED
                run_registry_trace_noexcept([&]() {
                    if (!JTRACE_ENABLED(1)) {
                        return;
                    }
                    const DeviceContextKey key = trace.key ? *trace.key : DeviceContextKey{};
                    const std::string msg = registry_trace_handle_prefix(trace.handle, key) + " action=watchdog_timeout" + " observed_state=" + to_cstr(trace.observedState) + " state_age_ms=" + std::to_string(trace.stateAgeMs) + " timeout_ms=" + std::to_string(trace.timeoutMs) + " escalated=" + std::to_string(registry_bool_u32(trace.escalated)) + " reason=" + registry_trace_or_unspecified(trace.reason);
                    JTRACE("MSLCY", msg);
                });
#endif
            }

            struct LifecycleTransitionRequest {
                ContextLifecycleState expectedState = ContextLifecycleState::Unbound;
                ContextLifecycleState desiredState = ContextLifecycleState::Unbound;
            };

            bool transition_entry_locked(const DeviceContextKey& key,
                                         RegistryEntry& entry,
                                         const LifecycleTransitionRequest& request,
                                         const char* reason) {
                ResourceManagerState& rmState = global_state();
                telemetry_counter_add(rmState.lifecycleTransitionCalls, 1);
                const ContextLifecycleState observed = entry.lifecycleState;
                const bool expectedOk = (observed == request.expectedState);
                const bool legal = expectedOk && is_legal_transition(observed, request.desiredState);
                if (!legal) {
                    telemetry_counter_add(rmState.lifecycleTransitionRejects, 1);
                    trace_lifecycle_transition(
                        LifecycleTransitionTrace{
                            .key = &key,
                            .reason = reason,
                            .handle = entry.handle,
                            .from = observed,
                            .to = request.desiredState});
                    return false;
                }
                entry.lifecycleState = request.desiredState;
                entry.lifecycleSinceMs = monotonic_time_ms();
                trace_lifecycle_transition(
                    LifecycleTransitionTrace{
                        .key = &key,
                        .reason = reason,
                        .handle = entry.handle,
                        .from = observed,
                        .to = request.desiredState,
                        .accepted = true});
                return true;
            }

            bool transition_entry_to_retired_locked(const DeviceContextKey& key,
                                                    RegistryEntry& entry,
                                                    const char* reason) {
                if (entry.lifecycleState == ContextLifecycleState::Retired) {
                    return true;
                }
                const ContextLifecycleState observed = entry.lifecycleState;
                return transition_entry_locked(
                    key,
                    entry,
                    LifecycleTransitionRequest{
                        .expectedState = observed,
                        .desiredState = ContextLifecycleState::Retired},
                    reason);
            }

            ContextLifecycleState desired_retire_state(RegistryRetireReason reason) noexcept {
                (void)reason;
                return ContextLifecycleState::Retired;
            }

            bool erase_registry_entry_locked(RegistryState& state,
                                             const DeviceContextKey& key,
                                             const RegistryEntry& entry,
                                             const char* eventName,
                                             const char* reason,
                                             bool countReapEvent,
                                             std::uint64_t idleMs) noexcept {
                const std::uint64_t handleValue = entry.handle.value;
                auto mapIt = state.byDeviceContext.find(key);
                if (mapIt == state.byDeviceContext.end()) {
                    return false;
                }
                state.byDeviceContext.erase(mapIt);
                if (handleValue != 0) {
                    state.keyByHandle.erase(handleValue);
                }

                ResourceManagerState& rmState = global_state();
                publish_registry_live_count(state.byDeviceContext.size());
                if (countReapEvent) {
                    telemetry_counter_add(rmState.registryReapEvents, 1);
                }
                trace_registry_event_current(
                    &key,
                    &entry,
                    eventName,
                    true,
                    reason,
                    idleMs);
                return true;
            }

            struct RegistryReapQuery {
                std::uint64_t nowMs = 0;
                std::uint64_t idleReapMs = 0;
            };

            void collect_reap_candidates_locked(
                const RegistryState& state,
                const DeviceContextKey* protectKey,
                const RegistryReapQuery& query,
                std::vector<ReapCandidate>& outCandidates) {
                outCandidates.clear();
                outCandidates.reserve(state.byDeviceContext.size());
                for (const auto& kv : state.byDeviceContext) {
                    const DeviceContextKey& candidateKey = kv.first;
                    const RegistryEntry& entry = kv.second;
                    if (protectKey && candidateKey == *protectKey) {
                        continue;
                    }
                    if (entry.activeSubmissionCount != 0) {
                        continue;
                    }
                    if (entry.lifecycleState != ContextLifecycleState::Active &&
                        entry.lifecycleState != ContextLifecycleState::Retired) {
                        continue;
                    }

                    const std::uint64_t idleMs = saturating_elapsed_ms(query.nowMs, entry.lastTouchedMs);
                    const bool eligible = (entry.lifecycleState == ContextLifecycleState::Retired) ||
                                          (idleMs >= query.idleReapMs);
                    if (!eligible) {
                        continue;
                    }

                    ReapCandidate candidate{};
                    candidate.key = candidateKey;
                    candidate.entry = entry;
                    candidate.idleMs = idleMs;
                    outCandidates.push_back(candidate);
                }
            }

            void sort_reap_candidates(std::vector<ReapCandidate>& candidates) {
                std::sort(candidates.begin(), candidates.end(), [](const ReapCandidate& lhs, const ReapCandidate& rhs) {
                    if (lhs.entry.lifecycleState != rhs.entry.lifecycleState) {
                        if (lhs.entry.lifecycleState == ContextLifecycleState::Retired) {
                            return true;
                        }
                        if (rhs.entry.lifecycleState == ContextLifecycleState::Retired) {
                            return false;
                        }
                    }
                    if (lhs.idleMs != rhs.idleMs) {
                        return lhs.idleMs > rhs.idleMs;
                    }
                    if (lhs.entry.createOrder != rhs.entry.createOrder) {
                        return lhs.entry.createOrder < rhs.entry.createOrder;
                    }
                    return lhs.entry.handle.value < rhs.entry.handle.value;
                });
            }

            std::size_t compute_reap_count(std::size_t candidateCount, std::uint64_t overflow) noexcept {
                if (overflow == 0) {
                    return candidateCount;
                }
                const std::size_t overflowCount = static_cast<std::size_t>(
                    std::min<std::uint64_t>(overflow, static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())));
                return std::min(candidateCount, overflowCount);
            }

            std::uint64_t registry_overflow_count_or_zero(
                std::uint64_t liveManagersBefore,
                std::uint64_t maxLive) noexcept {
                if (liveManagersBefore > maxLive) {
                    return liveManagersBefore - maxLive;
                }
                return 0ull;
            }

            void maybe_reap_idle_locked(RegistryState& state, const DeviceContextKey* protectKey) noexcept {
                try {
                    const ResourceManagerConfigEffective& cfg = registry_policy_config();
                    if (cfg.maxLiveManagersPerProcess == 0 || cfg.managerIdleReapMs == 0) {
                        return;
                    }
                    if (state.byDeviceContext.empty()) {
                        publish_registry_live_count(0);
                        return;
                    }

                    const std::uint64_t nowMs = monotonic_time_ms();
                    const std::uint64_t elapsedMs = saturating_elapsed_ms(nowMs, state.lastIdleReapScanMs);
                    const bool cadenceDue = (state.lastIdleReapScanMs == 0) ||
                                            (elapsedMs >= static_cast<std::uint64_t>(cfg.managerIdleReapMs));
                    const std::uint64_t liveManagersBefore = static_cast<std::uint64_t>(state.byDeviceContext.size());
                    publish_registry_live_count(state.byDeviceContext.size());
                    const std::uint64_t maxLive = static_cast<std::uint64_t>(cfg.maxLiveManagersPerProcess);
                    const std::uint64_t overflow = registry_overflow_count_or_zero(liveManagersBefore, maxLive);
                    if (!cadenceDue && overflow == 0) {
                        return;
                    }
                    state.lastIdleReapScanMs = nowMs;

                    std::vector<ReapCandidate> candidates;
                    collect_reap_candidates_locked(
                        state,
                        protectKey,
                        RegistryReapQuery{
                            .nowMs = nowMs,
                            .idleReapMs = static_cast<std::uint64_t>(cfg.managerIdleReapMs)},
                        candidates);

                    if (candidates.empty()) {
                        if (overflow > 0) {
                            trace_reap_skip_event("overflow_no_eligible_idle_manager");
                        }
                        return;
                    }

                    sort_reap_candidates(candidates);
                    const std::size_t reapCount = compute_reap_count(candidates.size(), overflow);

                    for (std::size_t i = 0; i < reapCount; ++i) {
                        const ReapCandidate& candidate = candidates[i];
                        const char* reason = registry_bool_reason(overflow > 0, "max_live_oldest_idle", "idle_timeout");
                        (void)erase_registry_entry_locked(
                            state,
                            candidate.key,
                            candidate.entry,
                            "reap",
                            reason,
                            true,
                            candidate.idleMs);
                    }

                    if (overflow > static_cast<std::uint64_t>(reapCount)) {
                        trace_reap_skip_event("overflow_remaining_after_safe_reap");
                    }
                } catch (...) {
                    JuicerLogging::discard_current_exception();
                }
            }

            bool run_freeze_drain_bump_resume_locked(const DeviceContextKey& key,
                                                     RegistryEntry& entry,
                                                     const char* reason) {
                ResourceManagerState& rmState = global_state();
                telemetry_counter_add(rmState.lifecycleBarrierCalls, 1);

                bool ok = true;
                ok = ok && transition_entry_locked(
                               key,
                               entry,
                               LifecycleTransitionRequest{
                                   .expectedState = ContextLifecycleState::Active,
                                   .desiredState = ContextLifecycleState::Freezing},
                               "barrier_freeze");
                ok = ok && transition_entry_locked(
                               key,
                               entry,
                               LifecycleTransitionRequest{
                                   .expectedState = ContextLifecycleState::Freezing,
                                   .desiredState = ContextLifecycleState::Draining},
                               "barrier_drain");

                bump_registry_epoch_locked(
                    key,
                    entry,
                    ok,
                    registry_trace_or(reason, "barrier_bump"));

                ok = ok && transition_entry_locked(
                               key,
                               entry,
                               LifecycleTransitionRequest{
                                   .expectedState = ContextLifecycleState::Draining,
                                   .desiredState = ContextLifecycleState::Rebinding},
                               "barrier_rebind");
                ok = ok && transition_entry_locked(
                               key,
                               entry,
                               LifecycleTransitionRequest{
                                   .expectedState = ContextLifecycleState::Rebinding,
                                   .desiredState = ContextLifecycleState::Active},
                               "barrier_resume");

                if (!ok) {
                    telemetry_counter_add(rmState.lifecycleBarrierRejects, 1);
                }
                return ok;
            }

        } // namespace

        bool metadata_mutation_begin(
            const char* stage,
            MetadataMutationScope& outScope,
            const DeviceContextKey* managerKey) noexcept {
            // Lock-order invariant:
            // - Resolve lane (directory mutex), then acquire/release lane mutex for ticketing.
            // - Sequence mutex is acquired only after lane mutex is released.
            // - Do not hold registry_state().mutex while touching lane/directory/sequence mutexes.
            try {
                if (outScope.active) {
                    trace_mutation_reject(
                        MutationRejectTrace{
                            .action = "begin",
                            .stage = stage,
                            .reason = "scope_already_active",
                            .sequence = outScope.sequence,
                            .expectedSequence = outScope.sequence,
                            .queueTicket = outScope.queueTicket});
                    return false;
                }

                std::shared_ptr<MetadataMutationLane> lane = resolve_mutation_lane(managerKey);
                if (!lane) {
                    trace_mutation_reject(
                        MutationRejectTrace{
                            .action = "begin",
                            .stage = stage,
                            .reason = "missing_lane"});
                    return false;
                }

                LaneAcquireResult acquire{};
                if (!acquire_lane_ticket(*lane, std::this_thread::get_id(), stage, acquire)) {
                    return false;
                }
                outScope.queueTicket = acquire.ticket;
                outScope.hasManagerKey = (managerKey && managerKey->deviceId >= 0);
                outScope.managerKey = registry_manager_key_or_default(outScope.hasManagerKey, managerKey);
                outScope.active = true;

                telemetry_record_metadata_mutation_begin();
                std::uint64_t sequence = 0;
                std::uint64_t expectedSequence = 0;
                bool orderOk = true;
                {
                    std::lock_guard<std::mutex> lock(mutation_sequence_mutex());
                    ResourceManagerState& state = global_state();
                    sequence = state.nextMetadataMutationSequence.fetch_add(1, std::memory_order_relaxed);
                    if (sequence == 0) {
                        sequence = state.nextMetadataMutationSequence.fetch_add(1, std::memory_order_relaxed);
                    }
                    expectedSequence = mutation_last_issued_sequence() + 1;
                    if (expectedSequence == 0) {
                        expectedSequence = 1;
                    }
                    orderOk = (sequence == expectedSequence);
                    if (!orderOk) {
                        telemetry_record_metadata_mutation_order_violation();
                    }
                    mutation_last_issued_sequence() = sequence;
                }

                outScope.sequence = sequence;
                gMutationThreadDepth += 1;
                gMutationThreadTicket = acquire.ticket;
                gMutationThreadBeginCount += 1;

                telemetry_trace_metadata_mutation(
                    TelemetryMetadataMutationTrace{
                        .phase = "begin",
                        .stage = stage,
                        .reason = registry_mutation_begin_reason(orderOk, acquire.reentrant),
                        .sequence = sequence,
                        .expectedSequence = expectedSequence,
                        .accepted = true});
                return true;
            } catch (...) {
                JuicerLogging::discard_current_exception();
                if (outScope.active) {
                    metadata_mutation_end(outScope, stage);
                } else {
                    reset_scope(outScope);
                }
                return false;
            }
        }

        void metadata_mutation_end(MetadataMutationScope& scope, const char* stage) noexcept {
            if (!scope.active) {
                trace_mutation_reject(
                    MutationRejectTrace{
                        .action = "end",
                        .stage = stage,
                        .reason = "scope_not_active",
                        .sequence = scope.sequence,
                        .expectedSequence = scope.sequence,
                        .queueTicket = scope.queueTicket});
                return;
            }

            std::shared_ptr<MetadataMutationLane> lane =
                find_mutation_lane_noexcept(registry_manager_key_ptr_or_null(scope.hasManagerKey, &scope.managerKey));
            if (!lane) {
                trace_mutation_reject(
                    MutationRejectTrace{
                        .action = "end",
                        .stage = stage,
                        .reason = "missing_lane",
                        .sequence = scope.sequence,
                        .expectedSequence = scope.sequence,
                        .queueTicket = scope.queueTicket});
                reset_thread_mutation_context();
                reset_scope(scope);
                return;
            }

            bool releaseTopLevel = false;
            std::uint64_t depthAfterRelease = 0;
            const std::thread::id currentThread = std::this_thread::get_id();
            {
                std::lock_guard<std::mutex> lock(lane->mutex);
                if (lane->ownerDepth == 0 || lane->ownerThread != currentThread) {
                    trace_mutation_reject(
                        MutationRejectTrace{
                            .action = "end",
                            .stage = stage,
                            .reason = "owner_mismatch",
                            .sequence = scope.sequence,
                            .expectedSequence = scope.sequence,
                            .queueTicket = scope.queueTicket,
                            .queueDepth = lane_depth_nolock(*lane)});
                    reset_thread_mutation_context();
                    reset_scope(scope);
                    return;
                }

                telemetry_record_metadata_mutation_end();
                --lane->ownerDepth;
                if (lane->ownerDepth == 0) {
                    if (lane->servingTicket != scope.queueTicket) {
                        telemetry_record_metadata_mutation_order_violation();
                        telemetry_trace_metadata_queue(
                            TelemetryMetadataQueueTrace{
                                .eventName = "order_violation",
                                .stage = stage,
                                .reason = "release_ticket_mismatch",
                                .ticket = scope.queueTicket,
                                .depth = lane_depth_nolock(*lane)});
                    }
                    if (lane->servingTicket < lane->nextTicket) {
                        ++lane->servingTicket;
                    }
                    lane->ownerThread = std::thread::id{};
                    lane->activeTicket = 0;
                    releaseTopLevel = true;
                }
                depthAfterRelease = lane_depth_nolock(*lane);
            }

            if (releaseTopLevel) {
                lane->cv.notify_all();
                telemetry_trace_metadata_queue(
                    TelemetryMetadataQueueTrace{
                        .eventName = "release",
                        .stage = stage,
                        .reason = "ok",
                        .ticket = scope.queueTicket,
                        .depth = depthAfterRelease,
                        .accepted = true});
            } else {
                telemetry_trace_metadata_queue(
                    TelemetryMetadataQueueTrace{
                        .eventName = "release_nested",
                        .stage = stage,
                        .reason = "owner_reentrant",
                        .ticket = scope.queueTicket,
                        .depth = depthAfterRelease,
                        .accepted = true});
            }

            telemetry_trace_metadata_mutation(
                TelemetryMetadataMutationTrace{
                    .phase = "end",
                    .stage = stage,
                    .reason = "ok",
                    .sequence = scope.sequence,
                    .expectedSequence = scope.sequence,
                    .accepted = true});
            if (gMutationThreadDepth > 0) {
                --gMutationThreadDepth;
            }
            if (gMutationThreadDepth == 0) {
                gMutationThreadTicket = 0;
            }
            reset_scope(scope);
        }

        bool metadata_mutation_thread_active() noexcept {
            return gMutationThreadDepth > 0;
        }

        std::uint32_t metadata_mutation_thread_depth() noexcept {
            return gMutationThreadDepth;
        }

        std::uint64_t metadata_mutation_thread_ticket() noexcept {
            return gMutationThreadTicket;
        }

        std::uint64_t metadata_mutation_thread_begin_count() noexcept {
            return gMutationThreadBeginCount;
        }

        MetadataMutationGuard::MetadataMutationGuard(
            const char* stage,
            const DeviceContextKey* managerKey) noexcept
            : _stage(stage) {
            if (managerKey) {
                _hasManagerKey = true;
                _managerKey = *managerKey;
            }
            const DeviceContextKey* key = registry_manager_key_ptr_or_null(_hasManagerKey, &_managerKey);
            (void)metadata_mutation_begin(_stage, _scope, key);
        }

        MetadataMutationGuard::~MetadataMutationGuard() noexcept {
            metadata_mutation_end(_scope, _stage);
        }

        const char* to_cstr(ContextLifecycleState state) noexcept {
            switch (state) {
                case ContextLifecycleState::Unbound:
                    return "Unbound";
                case ContextLifecycleState::Binding:
                    return "Binding";
                case ContextLifecycleState::Active:
                    return "Active";
                case ContextLifecycleState::Freezing:
                    return "Freezing";
                case ContextLifecycleState::Draining:
                    return "Draining";
                case ContextLifecycleState::Rebinding:
                    return "Rebinding";
                case ContextLifecycleState::Retired:
                    return "Retired";
                default:
                    return "Unknown";
            }
        }

        const char* to_cstr(LifecycleStageDecision decision) noexcept {
            switch (decision) {
                case LifecycleStageDecision::Allowed:
                    return "Allowed";
                case LifecycleStageDecision::MissingRegistryEntry:
                    return "MissingRegistryEntry";
                case LifecycleStageDecision::StateNotAllowed:
                    return "StateNotAllowed";
                case LifecycleStageDecision::TimedOut:
                    return "TimedOut";
                default:
                    return "Unknown";
            }
        }

        RegistryHandle registry_get_or_create(const DeviceContextKey& key) noexcept {
            try {
                MetadataMutationGuard mutationGuard("registry_get_or_create", &key);
                if (!mutationGuard.ok()) {
                    return RegistryHandle{};
                }
                RegistryState& state = registry_state();
                std::lock_guard<std::mutex> lock(state.mutex);
                maybe_reap_idle_locked(state, &key);

                const std::uint64_t nowMs = monotonic_time_ms();
                auto it = state.byDeviceContext.find(key);
                if (it != state.byDeviceContext.end()) {
                    if (it->second.lifecycleState == ContextLifecycleState::Retired) {
                        RegistryEntry retiredEntry = it->second;
                        (void)erase_registry_entry_locked(
                            state,
                            key,
                            retiredEntry,
                            "recreate",
                            "retired_recreate",
                            false,
                            0);
                    } else {
                        it->second.lastTouchedMs = nowMs;
                        publish_registry_live_count(state.byDeviceContext.size());
                        return it->second.handle;
                    }
                }

                RegistryEntry entry{};
                entry.handle.value = next_nonzero_counter(state.nextHandle);
                entry.lifecycleState = ContextLifecycleState::Unbound;
                entry.createOrder = next_nonzero_counter(state.nextCreateOrder);
                entry.lastTouchedMs = nowMs;
                entry.lifecycleSinceMs = nowMs;
                entry.activeSubmissionCount = 0;
                assign_entry_generations(entry);

                auto inserted = state.byDeviceContext.emplace(key, entry);
                RegistryEntry& insertedEntry = inserted.first->second;
                state.keyByHandle[insertedEntry.handle.value] = key;
                const bool createBound = transition_entry_locked(
                    key,
                    insertedEntry,
                    LifecycleTransitionRequest{
                        .expectedState = ContextLifecycleState::Unbound,
                        .desiredState = ContextLifecycleState::Binding},
                    "create_bind");
                if (!createBound) {
                    (void)transition_entry_to_retired_locked(
                        key,
                        insertedEntry,
                        "create_bind_retire");
                } else if (!transition_entry_locked(
                               key,
                               insertedEntry,
                               LifecycleTransitionRequest{
                                   .expectedState = ContextLifecycleState::Binding,
                                   .desiredState = ContextLifecycleState::Active},
                               "create_activate")) {
                    (void)transition_entry_to_retired_locked(
                        key,
                        insertedEntry,
                        "create_activate_retire");
                }

                publish_registry_live_count(state.byDeviceContext.size());
                trace_registry_event_current(
                    &key,
                    &insertedEntry,
                    "create",
                    true,
                    "create_or_reuse",
                    0);

                maybe_reap_idle_locked(state, &key);
                return insertedEntry.handle;
            } catch (...) {
                JuicerLogging::discard_current_exception();
                return RegistryHandle{};
            }
        }

        bool registry_get_snapshot_generations(
            const DeviceContextKey& key,
            RegistrySnapshotGenerations& outGenerations) noexcept {
            try {
                RegistryState& state = registry_state();
                std::lock_guard<std::mutex> lock(state.mutex);
                auto it = state.byDeviceContext.find(key);
                if (it == state.byDeviceContext.end() ||
                    it->second.lifecycleState == ContextLifecycleState::Retired) {
                    outGenerations = RegistrySnapshotGenerations{};
                    return false;
                }
                outGenerations.registryGeneration = it->second.registryGeneration;
                outGenerations.contextEpoch = it->second.contextEpoch;
                return true;
            } catch (...) {
                JuicerLogging::discard_current_exception();
                outGenerations = RegistrySnapshotGenerations{};
                return false;
            }
        }

        void registry_snapshot_context_keys(std::vector<DeviceContextKey>& outKeys) {
            outKeys.clear();
            RegistryState& state = registry_state();
            std::lock_guard<std::mutex> lock(state.mutex);
            outKeys.reserve(state.byDeviceContext.size());
            for (const auto& entry : state.byDeviceContext) {
                outKeys.push_back(entry.first);
            }
        }

        bool registry_validate_lifecycle_stage(
            const DeviceContextKey& key,
            bool allowNonActiveRelease,
            LifecycleStageValidation& outValidation) noexcept {
            outValidation = LifecycleStageValidation{};
            try {
                RegistryState& state = registry_state();
                std::lock_guard<std::mutex> lock(state.mutex);
                auto it = state.byDeviceContext.find(key);
                if (it == state.byDeviceContext.end()) {
                    outValidation.decision = LifecycleStageDecision::MissingRegistryEntry;
                    outValidation.observedState = ContextLifecycleState::Unbound;
                    outValidation.observedStateAgeMs = 0;
                    outValidation.escalated = false;
                    return false;
                }

                RegistryEntry& entry = it->second;
                const std::uint64_t nowMs = monotonic_time_ms();
                if (entry.lifecycleSinceMs == 0) {
                    entry.lifecycleSinceMs = nowMs;
                }

                const ContextLifecycleState observedState = entry.lifecycleState;
                const std::uint64_t stateAgeMs = saturating_elapsed_ms(nowMs, entry.lifecycleSinceMs);
                outValidation.observedState = observedState;
                outValidation.observedStateAgeMs = stateAgeMs;

                const ResourceManagerConfigEffective& cfg = registry_policy_config();
                const std::uint64_t timeoutMs = lifecycle_timeout_ms_for_state(observedState, cfg);
                if (timeoutMs > 0 && stateAgeMs >= timeoutMs) {
                    ResourceManagerState& rmState = global_state();
                    telemetry_counter_add(rmState.lifecycleTimeoutEvents, 1);
                    const RegistryHandle observedHandle = entry.handle;
                    const char* timeoutReason = registry_lifecycle_timeout_reason(observedState);

                    bool escalated = false;
                    if (entry.activeSubmissionCount == 0) {
                        bump_registry_epoch_locked(key, entry, true, "watchdog_timeout_bump");
                        if (transition_entry_locked(
                                key,
                                entry,
                                LifecycleTransitionRequest{
                                    .expectedState = observedState,
                                    .desiredState = ContextLifecycleState::Retired},
                                "watchdog_timeout_retire")) {
                            entry.lastTouchedMs = nowMs;
                            RegistryEntry retiredEntry = entry;
                            (void)erase_registry_entry_locked(
                                state,
                                key,
                                retiredEntry,
                                "lifecycle_timeout",
                                "watchdog_timeout_retire",
                                false,
                                0);
                            escalated = true;
                        }
                    }

                    trace_lifecycle_timeout(
                        LifecycleTimeoutTrace{
                            .key = &key,
                            .reason = timeoutReason,
                            .handle = observedHandle,
                            .observedState = observedState,
                            .stateAgeMs = stateAgeMs,
                            .timeoutMs = timeoutMs,
                            .escalated = escalated});
                    outValidation.decision = LifecycleStageDecision::TimedOut;
                    outValidation.escalated = escalated;
                    return false;
                }

                if (!lifecycle_state_allowed_for_stage(observedState, allowNonActiveRelease)) {
                    outValidation.decision = LifecycleStageDecision::StateNotAllowed;
                    outValidation.escalated = false;
                    return false;
                }

                outValidation.decision = LifecycleStageDecision::Allowed;
                outValidation.escalated = false;
                return true;
            } catch (...) {
                JuicerLogging::discard_current_exception();
                outValidation = LifecycleStageValidation{};
                outValidation.decision = LifecycleStageDecision::MissingRegistryEntry;
                outValidation.observedState = ContextLifecycleState::Unbound;
                return false;
            }
        }

        bool registry_note_submission_begin(const DeviceContextKey& key) noexcept {
            try {
                RegistryState& state = registry_state();
                std::lock_guard<std::mutex> lock(state.mutex);
                auto it = state.byDeviceContext.find(key);
                if (it == state.byDeviceContext.end()) {
                    trace_registry_missing_entry(key, "submission_begin");
                    return false;
                }

                RegistryEntry& entry = it->second;
                if (entry.activeSubmissionCount < std::numeric_limits<std::uint64_t>::max()) {
                    ++entry.activeSubmissionCount;
                }
                entry.lastTouchedMs = monotonic_time_ms();
                return true;
            } catch (...) {
                JuicerLogging::discard_current_exception();
                return false;
            }
        }

        bool registry_note_submission_end(const DeviceContextKey& key) noexcept {
            try {
                RegistryState& state = registry_state();
                std::lock_guard<std::mutex> lock(state.mutex);
                auto it = state.byDeviceContext.find(key);
                if (it == state.byDeviceContext.end()) {
                    trace_registry_missing_entry(key, "submission_end");
                    return false;
                }

                RegistryEntry& entry = it->second;
                if (entry.activeSubmissionCount == 0) {
                    trace_registry_event_current(
                        &key,
                        &entry,
                        "submission_end",
                        false,
                        "active_submission_underflow",
                        0);
                    return false;
                }
                --entry.activeSubmissionCount;
                entry.lastTouchedMs = monotonic_time_ms();
                maybe_reap_idle_locked(state, &key);
                return true;
            } catch (...) {
                JuicerLogging::discard_current_exception();
                return false;
            }
        }

        bool registry_transition_lifecycle_state(
            const DeviceContextKey& key,
            ContextLifecycleState expectedState,
            ContextLifecycleState desiredState,
            const char* reason) noexcept {
            try {
                MetadataMutationGuard mutationGuard("registry_transition_lifecycle_state", &key);
                if (!mutationGuard.ok()) {
                    return false;
                }
                RegistryState& state = registry_state();
                std::lock_guard<std::mutex> lock(state.mutex);
                auto it = state.byDeviceContext.find(key);
                if (it == state.byDeviceContext.end()) {
                    telemetry_counter_add(global_state().lifecycleTransitionCalls, 1);
                    telemetry_counter_add(global_state().lifecycleTransitionRejects, 1);
                    const RegistryHandle missingHandle{};
                    trace_lifecycle_transition(
                        LifecycleTransitionTrace{
                            .key = &key,
                            .reason = reason,
                            .handle = missingHandle,
                            .from = ContextLifecycleState::Unbound,
                            .to = desiredState});
                    return false;
                }
                const bool ok = transition_entry_locked(
                    key,
                    it->second,
                    LifecycleTransitionRequest{
                        .expectedState = expectedState,
                        .desiredState = desiredState},
                    reason);
                if (ok) {
                    it->second.lastTouchedMs = monotonic_time_ms();
                    maybe_reap_idle_locked(state, &key);
                }
                return ok;
            } catch (...) {
                JuicerLogging::discard_current_exception();
                return false;
            }
        }

        bool registry_freeze_drain_bump_resume(
            const DeviceContextKey& key,
            const char* reason) noexcept {
            try {
                MetadataMutationGuard mutationGuard("registry_freeze_drain_bump_resume", &key);
                if (!mutationGuard.ok()) {
                    return false;
                }
                RegistryState& state = registry_state();
                std::lock_guard<std::mutex> lock(state.mutex);
                auto it = state.byDeviceContext.find(key);
                if (it == state.byDeviceContext.end()) {
                    ResourceManagerState& rmState = global_state();
                    telemetry_counter_add(rmState.lifecycleBarrierCalls, 1);
                    telemetry_counter_add(rmState.lifecycleBarrierRejects, 1);
                    const RegistryHandle missingHandle{};
                    trace_lifecycle_transition(
                        LifecycleTransitionTrace{
                            .key = &key,
                            .reason = registry_trace_or(reason, "barrier_missing_entry"),
                            .handle = missingHandle,
                            .from = ContextLifecycleState::Unbound,
                            .to = ContextLifecycleState::Freezing});
                    return false;
                }
                const bool ok = run_freeze_drain_bump_resume_locked(key, it->second, reason);
                if (ok) {
                    it->second.lastTouchedMs = monotonic_time_ms();
                }
                return ok;
            } catch (...) {
                JuicerLogging::discard_current_exception();
                return false;
            }
        }

        bool registry_begin_owner_retire(
            const DeviceContextKey& key,
            RegistrySnapshotGenerations& outGenerations) noexcept {
            outGenerations = RegistrySnapshotGenerations{};
            try {
                MetadataMutationGuard mutationGuard(
                    "registry_begin_owner_retire",
                    &key);
                if (!mutationGuard.ok()) {
                    return false;
                }
                RegistryState& state = registry_state();
                std::lock_guard<std::mutex> lock(state.mutex);
                const auto it = state.byDeviceContext.find(key);
                if (it == state.byDeviceContext.end()) {
                    return false;
                }
                RegistryEntry& entry = it->second;
                if (entry.activeSubmissionCount != 0) {
                    trace_registry_event_current(
                        &key,
                        &entry,
                        "owner_retire_begin",
                        false,
                        "active_submissions",
                        0);
                    return false;
                }
                outGenerations.registryGeneration = entry.registryGeneration;
                outGenerations.contextEpoch = entry.contextEpoch;
                if (entry.lifecycleState == ContextLifecycleState::Draining) {
                    return true;
                }
                if (!transition_entry_locked(
                        key,
                        entry,
                        LifecycleTransitionRequest{
                            .expectedState = ContextLifecycleState::Active,
                            .desiredState = ContextLifecycleState::Freezing},
                        "owner_retire_freeze") ||
                    !transition_entry_locked(
                        key,
                        entry,
                        LifecycleTransitionRequest{
                            .expectedState = ContextLifecycleState::Freezing,
                            .desiredState = ContextLifecycleState::Draining},
                        "owner_retire_drain")) {
                    return false;
                }
                entry.lastTouchedMs = monotonic_time_ms();
                return true;
            } catch (...) {
                JuicerLogging::discard_current_exception();
                outGenerations = RegistrySnapshotGenerations{};
                return false;
            }
        }

        bool registry_get(const DeviceContextKey& key, RegistryHandle& outHandle) noexcept {
            try {
                RegistryState& state = registry_state();
                std::lock_guard<std::mutex> lock(state.mutex);
                auto it = state.byDeviceContext.find(key);
                if (it == state.byDeviceContext.end()) {
                    outHandle = RegistryHandle{};
                    return false;
                }
                outHandle = it->second.handle;
                return true;
            } catch (...) {
                JuicerLogging::discard_current_exception();
                outHandle = RegistryHandle{};
                return false;
            }
        }

        bool registry_retire(
            RegistryHandle handle,
            RegistryRetireReason reason,
            const DeviceContextKey* managerKey) noexcept {
            try {
                MetadataMutationGuard mutationGuard("registry_retire", managerKey);
                if (!mutationGuard.ok()) {
                    return false;
                }
                if (handle.value == 0) {
                    return true;
                }
                RegistryState& state = registry_state();
                std::lock_guard<std::mutex> lock(state.mutex);
                auto keyIt = state.keyByHandle.find(handle.value);
                if (keyIt == state.keyByHandle.end()) {
                    return true;
                }
                auto entryIt = state.byDeviceContext.find(keyIt->second);
                if (entryIt == state.byDeviceContext.end()) {
                    state.keyByHandle.erase(keyIt);
                    publish_registry_live_count(state.byDeviceContext.size());
                    return true;
                }
                RegistryEntry& entry = entryIt->second;
                const DeviceContextKey deviceKey = keyIt->second;
                const std::uint64_t nowMs = monotonic_time_ms();
                entry.lastTouchedMs = nowMs;
                const bool requiresNoActiveSubmissions =
                    reason == RegistryRetireReason::Idle ||
                    reason == RegistryRetireReason::ContextReset;
                if (requiresNoActiveSubmissions && entry.activeSubmissionCount != 0) {
                    trace_registry_event_current(
                        &deviceKey,
                        &entry,
                        "retire",
                        false,
                        "active_submissions",
                        0);
                    return false;
                }
                if (reason == RegistryRetireReason::ContextReset &&
                    entry.lifecycleState != ContextLifecycleState::Draining) {
                    const bool barrierOk = run_freeze_drain_bump_resume_locked(
                        deviceKey,
                        entry,
                        "context_reset_barrier");
                    if (!barrierOk) {
                        trace_lifecycle_transition(
                            LifecycleTransitionTrace{
                                .key = &deviceKey,
                                .reason = "context_reset_barrier_reject",
                                .handle = entry.handle,
                                .from = entry.lifecycleState,
                                .to = entry.lifecycleState});
                    }
                }
                ContextLifecycleState desired = desired_retire_state(reason);
                if (desired != ContextLifecycleState::Retired) {
                    telemetry_counter_add(global_state().lifecycleTransitionCalls, 1);
                    telemetry_counter_add(global_state().lifecycleTransitionRejects, 1);
                    trace_lifecycle_transition(
                        LifecycleTransitionTrace{
                            .key = &deviceKey,
                            .reason = "retire_unsupported",
                            .handle = entry.handle,
                            .from = entry.lifecycleState,
                            .to = desired});
                    return false;
                }
                if (!transition_entry_to_retired_locked(deviceKey, entry, "retire")) {
                    return false;
                }
                RegistryEntry removed = entry;
                (void)erase_registry_entry_locked(
                    state,
                    deviceKey,
                    removed,
                    "retire",
                    registry_retire_reason_name(reason),
                    false,
                    0);
                return true;
            } catch (...) {
                JuicerLogging::discard_current_exception();
                return false;
            }
        }

    } // namespace ResourceManager
} // namespace JuicerCuda
