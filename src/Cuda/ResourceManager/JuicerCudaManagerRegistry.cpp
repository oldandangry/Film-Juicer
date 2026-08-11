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

#if JUICER_DIAGNOSTICS_COMPILED
            const char* trace_or_unknown(const char* value) noexcept {
                if (value && value[0] != '\0') {
                    return value;
                }
                return "unknown";
            }

            std::string trace_device_context_fields(const DeviceContextKey& key) {
                const std::uintptr_t contextBits =
                    reinterpret_cast<std::uintptr_t>(key.contextOpaque);
                return std::string(" device_id=") + std::to_string(key.deviceId) +
                       " context=" + std::to_string(contextBits);
            }
#endif

            struct RegistryTrace {
                DeviceContextKey key{};
                RegistryEntry entry{};
                const char* eventName = nullptr;
                const char* reason = nullptr;
                std::uint64_t liveManagers = 0;
                bool hasEntry = false;
                bool accepted = false;
            };

            RegistryTrace snapshot_registry_trace(
                const RegistryState& state,
                const DeviceContextKey& key,
                const RegistryEntry* entry,
                const char* eventName,
                bool accepted,
                const char* reason,
                bool traceEnabled) noexcept {
                if (!traceEnabled) {
                    return RegistryTrace{};
                }
                RegistryTrace trace{
                    .key = key,
                    .eventName = eventName,
                    .reason = reason,
                    .liveManagers = static_cast<std::uint64_t>(state.byDeviceContext.size()),
                    .hasEntry = entry != nullptr,
                    .accepted = accepted};
                if (entry) {
                    trace.entry = *entry;
                }
                return trace;
            }

            void trace_registry_event(const RegistryTrace& trace) noexcept {
#if JUICER_DIAGNOSTICS_COMPILED
                if (!trace.eventName || !JTRACE_ENABLED(2)) {
                    return;
                }
                try {
                    const std::uint64_t contextEpoch =
                        trace.hasEntry ? trace.entry.contextEpoch : 0;
                    const std::uint64_t activeSubmissions =
                        trace.hasEntry ? trace.entry.activeSubmissionCount : 0;
                    const bool retiring = trace.hasEntry && trace.entry.retiring;
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
                } catch (...) {
                    JuicerLogging::discard_current_exception();
                }
#else
                (void)trace;
#endif
            }

        } // namespace

        bool registry_begin_submission(
            const DeviceContextKey& key,
            RegistryContextSnapshot& outSnapshot) noexcept {
            outSnapshot = RegistryContextSnapshot{};
            try {
                const bool traceEnabled = JTRACE_ENABLED(2);
                RegistryTrace trace{};
                bool accepted = false;
                RegistryState& state = registry_state();
                {
                    std::lock_guard<std::mutex> lock(state.mutex);
                    auto [it, inserted] = state.byDeviceContext.try_emplace(key);
                    RegistryEntry& entry = it->second;
                    if (inserted) {
                        entry.contextEpoch = next_nonzero_epoch();
                        trace = snapshot_registry_trace(
                            state,
                            key,
                            &entry,
                            "create",
                            true,
                            "create_for_submission",
                            traceEnabled);
                    }

                    if (entry.retiring) {
                        trace = snapshot_registry_trace(
                            state,
                            key,
                            &entry,
                            "submission_begin",
                            false,
                            "context_retiring",
                            traceEnabled);
                    } else if (entry.activeSubmissionCount ==
                               std::numeric_limits<std::uint64_t>::max()) {
                        trace = snapshot_registry_trace(
                            state,
                            key,
                            &entry,
                            "submission_begin",
                            false,
                            "active_submission_count_overflow",
                            traceEnabled);
                    } else {
                        ++entry.activeSubmissionCount;
                        outSnapshot.contextEpoch = entry.contextEpoch;
                        accepted = true;
                    }
                }
                trace_registry_event(trace);
                return accepted;
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
                const bool traceEnabled = JTRACE_ENABLED(2);
                RegistryTrace trace{};
                bool accepted = false;
                RegistryState& state = registry_state();
                {
                    std::lock_guard<std::mutex> lock(state.mutex);
                    const auto it = state.byDeviceContext.find(key);
                    if (it == state.byDeviceContext.end()) {
                        trace = snapshot_registry_trace(
                            state,
                            key,
                            nullptr,
                            "submission_end",
                            false,
                            "missing_registry_entry",
                            traceEnabled);
                    } else if (it->second.activeSubmissionCount == 0) {
                        trace = snapshot_registry_trace(
                            state,
                            key,
                            &it->second,
                            "submission_end",
                            false,
                            "active_submission_underflow",
                            traceEnabled);
                    } else {
                        --it->second.activeSubmissionCount;
                        accepted = true;
                    }
                }
                trace_registry_event(trace);
                return accepted;
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
                const bool traceEnabled = JTRACE_ENABLED(2);
                RegistryTrace trace{};
                bool accepted = false;
                RegistryState& state = registry_state();
                {
                    std::lock_guard<std::mutex> lock(state.mutex);
                    const auto it = state.byDeviceContext.find(key);
                    if (it != state.byDeviceContext.end()) {
                        RegistryEntry& entry = it->second;
                        if (entry.activeSubmissionCount != 0) {
                            trace = snapshot_registry_trace(
                                state,
                                key,
                                &entry,
                                "owner_retire_begin",
                                false,
                                "active_submissions",
                                traceEnabled);
                        } else {
                            if (!entry.retiring) {
                                entry.retiring = true;
                                trace = snapshot_registry_trace(
                                    state,
                                    key,
                                    &entry,
                                    "owner_retire_begin",
                                    true,
                                    "context_retiring",
                                    traceEnabled);
                            }
                            outSnapshot.contextEpoch = entry.contextEpoch;
                            accepted = true;
                        }
                    }
                }
                trace_registry_event(trace);
                return accepted;
            } catch (...) {
                JuicerLogging::discard_current_exception();
                outSnapshot = RegistryContextSnapshot{};
                return false;
            }
        }

        bool registry_retire(const DeviceContextKey& key) noexcept {
            try {
                const bool traceEnabled = JTRACE_ENABLED(2);
                RegistryTrace trace{};
                bool accepted = false;
                RegistryState& state = registry_state();
                {
                    std::lock_guard<std::mutex> lock(state.mutex);
                    const auto it = state.byDeviceContext.find(key);
                    if (it == state.byDeviceContext.end()) {
                        accepted = true;
                    } else if (it->second.activeSubmissionCount != 0) {
                        trace = snapshot_registry_trace(
                            state,
                            key,
                            &it->second,
                            "retire",
                            false,
                            "active_submissions",
                            traceEnabled);
                    } else if (!it->second.retiring) {
                        trace = snapshot_registry_trace(
                            state,
                            key,
                            &it->second,
                            "retire",
                            false,
                            "owner_retire_not_started",
                            traceEnabled);
                    } else {
                        const RegistryEntry removedEntry = it->second;
                        state.byDeviceContext.erase(it);
                        trace = snapshot_registry_trace(
                            state,
                            key,
                            &removedEntry,
                            "retire",
                            true,
                            "explicit_owner_retire",
                            traceEnabled);
                        accepted = true;
                    }
                }
                trace_registry_event(trace);
                return accepted;
            } catch (...) {
                JuicerLogging::discard_current_exception();
                return false;
            }
        }

    } // namespace ResourceManager
} // namespace JuicerCuda
