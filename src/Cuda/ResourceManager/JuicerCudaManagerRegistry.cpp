// Cuda/ResourceManager/JuicerCudaManagerRegistry.cpp

#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#include "Cuda/ResourceManager/JuicerCudaResourceCore.h"

#include <atomic>
#include <limits>
#include <mutex>
#include <string>
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
                std::uint64_t activeSubmissionCount = 0;
                std::uint64_t contextEpoch = 1;
                bool retiring = false;
            };

            struct RegistryState {
                std::mutex mutex;
                std::unordered_map<DeviceContextKey, RegistryEntry, DeviceContextKeyHash> byDeviceContext;
            };

            RegistryState& registry_state() {
                static RegistryState state{};
                return state;
            }

            std::uint64_t next_nonzero_epoch() noexcept {
                std::atomic<std::uint64_t>& counter = global_state().contextEpoch;
                std::uint64_t epoch = counter.fetch_add(1, std::memory_order_relaxed);
                if (epoch == 0) {
                    epoch = counter.fetch_add(1, std::memory_order_relaxed);
                }
                return epoch;
            }

            void publish_registry_live_count(std::size_t count) noexcept {
                global_state().registryLiveManagers.store(
                    static_cast<std::uint64_t>(count),
                    std::memory_order_relaxed);
            }

#if JUICER_DIAGNOSTICS_COMPILED
            const char* trace_or_unknown(const char* value) noexcept {
                if (value && value[0] != '\0') {
                    return value;
                }
                return "unknown";
            }

            std::string trace_device_context_fields(const DeviceContextKey* key) {
                const int deviceId = key ? key->deviceId : -1;
                const std::uintptr_t contextBits = key
                                                       ? reinterpret_cast<std::uintptr_t>(key->contextOpaque)
                                                       : 0;
                return std::string(" device_id=") + std::to_string(deviceId) +
                       " context=" + std::to_string(contextBits);
            }
#endif

            struct RegistryEventTrace {
                const DeviceContextKey* key = nullptr;
                const RegistryEntry* entry = nullptr;
                const char* eventName = nullptr;
                const char* reason = nullptr;
                std::uint64_t liveManagers = 0;
                bool accepted = false;
            };

            void trace_registry_event(const RegistryEventTrace& trace) {
#if JUICER_DIAGNOSTICS_COMPILED
                if (!JTRACE_ENABLED(2)) {
                    return;
                }
                const std::uint64_t contextEpoch = trace.entry ? trace.entry->contextEpoch : 0;
                const std::uint64_t activeSubmissions = trace.entry ? trace.entry->activeSubmissionCount : 0;
                const bool retiring = trace.entry && trace.entry->retiring;
                const std::string msg =
                    std::string("event=") + trace_or_unknown(trace.eventName) +
                    " accepted=" + std::to_string(trace.accepted ? 1 : 0) +
                    " reason=" + trace_or_unknown(trace.reason) +
                    trace_device_context_fields(trace.key) +
                    " context_epoch=" + std::to_string(contextEpoch) +
                    " active_submissions=" + std::to_string(activeSubmissions) +
                    " retiring=" + std::to_string(retiring ? 1 : 0) +
                    " live_managers=" + std::to_string(trace.liveManagers);
                JTRACE_LEVEL(2, "MSREG", msg);
#else
                (void)trace;
#endif
            }

            void trace_registry_event_current(
                const DeviceContextKey* key,
                const RegistryEntry* entry,
                const char* eventName,
                bool accepted,
                const char* reason) noexcept {
                const std::uint64_t liveManagers =
                    global_state().registryLiveManagers.load(std::memory_order_relaxed);
                run_registry_trace_noexcept([&]() {
                    trace_registry_event(
                        RegistryEventTrace{
                            .key = key,
                            .entry = entry,
                            .eventName = eventName,
                            .reason = reason,
                            .liveManagers = liveManagers,
                            .accepted = accepted});
                });
            }

            void trace_missing_entry(const DeviceContextKey& key, const char* eventName) noexcept {
                trace_registry_event_current(
                    &key,
                    nullptr,
                    eventName,
                    false,
                    "missing_registry_entry");
            }

            bool erase_registry_entry_locked(
                RegistryState& state,
                const DeviceContextKey& key) noexcept {
                const auto it = state.byDeviceContext.find(key);
                if (it == state.byDeviceContext.end()) {
                    return false;
                }
                const RegistryEntry removed = it->second;
                state.byDeviceContext.erase(it);
                publish_registry_live_count(state.byDeviceContext.size());
                trace_registry_event_current(
                    &key,
                    &removed,
                    "retire",
                    true,
                    "explicit_owner_retire");
                return true;
            }

        } // namespace

        bool registry_begin_submission(
            const DeviceContextKey& key,
            RegistryContextSnapshot& outSnapshot) noexcept {
            outSnapshot = RegistryContextSnapshot{};
            try {
                RegistryState& state = registry_state();
                std::lock_guard<std::mutex> lock(state.mutex);
                auto [it, inserted] = state.byDeviceContext.try_emplace(key);
                RegistryEntry& entry = it->second;
                if (inserted) {
                    entry.contextEpoch = next_nonzero_epoch();
                    publish_registry_live_count(state.byDeviceContext.size());
                    trace_registry_event_current(
                        &key,
                        &entry,
                        "create",
                        true,
                        "create_for_submission");
                }

                if (entry.retiring) {
                    trace_registry_event_current(
                        &key,
                        &entry,
                        "submission_begin",
                        false,
                        "context_retiring");
                    return false;
                }
                if (entry.activeSubmissionCount == std::numeric_limits<std::uint64_t>::max()) {
                    trace_registry_event_current(
                        &key,
                        &entry,
                        "submission_begin",
                        false,
                        "active_submission_count_overflow");
                    return false;
                }

                ++entry.activeSubmissionCount;
                outSnapshot.contextEpoch = entry.contextEpoch;
                return true;
            } catch (...) {
                JuicerLogging::discard_current_exception();
                outSnapshot = RegistryContextSnapshot{};
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

        bool registry_note_submission_end(const DeviceContextKey& key) noexcept {
            try {
                RegistryState& state = registry_state();
                std::lock_guard<std::mutex> lock(state.mutex);
                const auto it = state.byDeviceContext.find(key);
                if (it == state.byDeviceContext.end()) {
                    trace_missing_entry(key, "submission_end");
                    return false;
                }
                RegistryEntry& entry = it->second;
                if (entry.activeSubmissionCount == 0) {
                    trace_registry_event_current(
                        &key,
                        &entry,
                        "submission_end",
                        false,
                        "active_submission_underflow");
                    return false;
                }
                --entry.activeSubmissionCount;
                return true;
            } catch (...) {
                JuicerLogging::discard_current_exception();
                return false;
            }
        }

        bool registry_begin_owner_retire(
            const DeviceContextKey& key,
            RegistryContextSnapshot& outSnapshot) noexcept {
            outSnapshot = RegistryContextSnapshot{};
            try {
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
                        "active_submissions");
                    return false;
                }
                if (!entry.retiring) {
                    entry.retiring = true;
                    trace_registry_event_current(
                        &key,
                        &entry,
                        "owner_retire_begin",
                        true,
                        "context_retiring");
                }
                outSnapshot.contextEpoch = entry.contextEpoch;
                return true;
            } catch (...) {
                JuicerLogging::discard_current_exception();
                outSnapshot = RegistryContextSnapshot{};
                return false;
            }
        }

        bool registry_retire(const DeviceContextKey& key) noexcept {
            try {
                RegistryState& state = registry_state();
                std::lock_guard<std::mutex> lock(state.mutex);
                const auto it = state.byDeviceContext.find(key);
                if (it == state.byDeviceContext.end()) {
                    return true;
                }
                RegistryEntry& entry = it->second;
                if (entry.activeSubmissionCount != 0) {
                    trace_registry_event_current(
                        &key,
                        &entry,
                        "retire",
                        false,
                        "active_submissions");
                    return false;
                }
                if (!entry.retiring) {
                    trace_registry_event_current(
                        &key,
                        &entry,
                        "retire",
                        false,
                        "owner_retire_not_started");
                    return false;
                }
                return erase_registry_entry_locked(state, key);
            } catch (...) {
                JuicerLogging::discard_current_exception();
                return false;
            }
        }

    } // namespace ResourceManager
} // namespace JuicerCuda
