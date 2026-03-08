// Cuda/JuicerCudaLaunchGraphCounters.h
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>

namespace JuicerCuda {
namespace LaunchGraphCounters {

#ifndef JUICER_LAUNCH_GRAPH_COUNTERS_COMPILED
#define JUICER_LAUNCH_GRAPH_COUNTERS_COMPILED 0
#endif

struct Snapshot {
    std::uint64_t framesRendered = 0;
    std::uint64_t kernelLaunchTotal = 0;
    std::uint64_t graphEligibleSubmissionTotal = 0;
    std::uint64_t graphReplayHitTotal = 0;
};

inline bool compiled_enabled() noexcept {
#if JUICER_LAUNCH_GRAPH_COUNTERS_COMPILED
    return true;
#else
    return false;
#endif
}

#if JUICER_LAUNCH_GRAPH_COUNTERS_COMPILED
namespace detail {

struct RuntimeConfig {
    bool runtimeEnabled = false;
    std::filesystem::path snapshotPath{};
};

inline int parse_env_int(const char* name, int fallback) noexcept {
    if (!name) {
        return fallback;
    }
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return fallback;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value) {
        return fallback;
    }
    return static_cast<int>(parsed);
}

inline const RuntimeConfig& runtime_config() noexcept {
    static const RuntimeConfig config = []() {
        RuntimeConfig out{};
        out.runtimeEnabled = parse_env_int("JUICER_LAUNCH_GRAPH_COUNTERS", 0) != 0;
        const char* snapshotPathEnv = std::getenv("JUICER_LAUNCH_GRAPH_COUNTERS_PATH");
        if (out.runtimeEnabled && snapshotPathEnv && *snapshotPathEnv) {
            try {
                out.snapshotPath = std::filesystem::path(snapshotPathEnv);
                out.snapshotPath.make_preferred();
            }
            catch (...) {
                out.snapshotPath.clear();
            }
        }
        return out;
    }();
    return config;
}

inline std::atomic<std::uint64_t>& frames_rendered_counter() noexcept {
    static std::atomic<std::uint64_t> counter{ 0 };
    return counter;
}

inline std::atomic<std::uint64_t>& kernel_launch_total_counter() noexcept {
    static std::atomic<std::uint64_t> counter{ 0 };
    return counter;
}

inline std::atomic<std::uint64_t>& graph_eligible_submission_total_counter() noexcept {
    static std::atomic<std::uint64_t> counter{ 0 };
    return counter;
}

inline std::atomic<std::uint64_t>& graph_replay_hit_total_counter() noexcept {
    static std::atomic<std::uint64_t> counter{ 0 };
    return counter;
}

inline std::mutex& snapshot_write_mutex() noexcept {
    static std::mutex m;
    return m;
}

inline std::once_flag& snapshot_dir_once_flag() noexcept {
    static std::once_flag flag;
    return flag;
}

inline Snapshot snapshot() noexcept {
    Snapshot out{};
    out.framesRendered = frames_rendered_counter().load(std::memory_order_relaxed);
    out.kernelLaunchTotal = kernel_launch_total_counter().load(std::memory_order_relaxed);
    out.graphEligibleSubmissionTotal =
        graph_eligible_submission_total_counter().load(std::memory_order_relaxed);
    out.graphReplayHitTotal =
        graph_replay_hit_total_counter().load(std::memory_order_relaxed);
    return out;
}

inline double kernel_launches_per_frame(const Snapshot& snapshot) noexcept {
    if (snapshot.framesRendered == 0) {
        return 0.0;
    }
    return static_cast<double>(snapshot.kernelLaunchTotal) /
        static_cast<double>(snapshot.framesRendered);
}

inline double cuda_graph_replay_hit_rate(const Snapshot& snapshot) noexcept {
    if (snapshot.graphEligibleSubmissionTotal == 0) {
        return 0.0;
    }
    return (100.0 * static_cast<double>(snapshot.graphReplayHitTotal)) /
        static_cast<double>(snapshot.graphEligibleSubmissionTotal);
}

inline void ensure_snapshot_dir_exists_if_needed(const std::filesystem::path& path) noexcept {
    if (path.empty() || !path.has_parent_path()) {
        return;
    }
    std::call_once(snapshot_dir_once_flag(), [&]() {
        try {
            std::error_code ec;
            std::filesystem::create_directories(path.parent_path(), ec);
        }
        catch (...) {
        }
    });
}

inline void flush_snapshot_file() noexcept {
    const RuntimeConfig& config = runtime_config();
    if (!config.runtimeEnabled || config.snapshotPath.empty()) {
        return;
    }
    try {
        ensure_snapshot_dir_exists_if_needed(config.snapshotPath);
        const Snapshot current = snapshot();

        std::lock_guard<std::mutex> lock(snapshot_write_mutex());
        std::ofstream out(config.snapshotPath, std::ios::out | std::ios::trunc);
        if (!out.is_open()) {
            return;
        }
        out.setf(std::ios::fixed, std::ios::floatfield);
        out.precision(6);
        out << "format_version=1\n";
        out << "compiled=1\n";
        out << "runtime_enabled=1\n";
        out << "frames_rendered=" << current.framesRendered << "\n";
        out << "kernel_launch_total=" << current.kernelLaunchTotal << "\n";
        out << "graph_eligible_submission_total=" << current.graphEligibleSubmissionTotal << "\n";
        out << "graph_replay_hit_total=" << current.graphReplayHitTotal << "\n";
        out << "kernel_launches_per_frame=" << kernel_launches_per_frame(current) << "\n";
        out << "cuda_graph_replay_hit_rate=" << cuda_graph_replay_hit_rate(current) << "\n";
    }
    catch (...) {
    }
}

} // namespace detail
#endif

inline bool runtime_enabled() noexcept {
#if JUICER_LAUNCH_GRAPH_COUNTERS_COMPILED
    return detail::runtime_config().runtimeEnabled;
#else
    return false;
#endif
}

inline void record_kernel_launch(std::uint64_t delta = 1) noexcept {
#if JUICER_LAUNCH_GRAPH_COUNTERS_COMPILED
    if (delta == 0 || !runtime_enabled()) {
        return;
    }
    detail::kernel_launch_total_counter().fetch_add(delta, std::memory_order_relaxed);
#else
    (void)delta;
#endif
}

inline void record_graph_eligible_submission(std::uint64_t delta = 1) noexcept {
#if JUICER_LAUNCH_GRAPH_COUNTERS_COMPILED
    if (delta == 0 || !runtime_enabled()) {
        return;
    }
    detail::graph_eligible_submission_total_counter().fetch_add(delta, std::memory_order_relaxed);
#else
    (void)delta;
#endif
}

inline void record_graph_replay_hit(std::uint64_t delta = 1) noexcept {
#if JUICER_LAUNCH_GRAPH_COUNTERS_COMPILED
    if (delta == 0 || !runtime_enabled()) {
        return;
    }
    detail::graph_replay_hit_total_counter().fetch_add(delta, std::memory_order_relaxed);
#else
    (void)delta;
#endif
}

inline void record_frame_completed(std::uint64_t delta = 1) noexcept {
#if JUICER_LAUNCH_GRAPH_COUNTERS_COMPILED
    if (delta == 0 || !runtime_enabled()) {
        return;
    }
    detail::frames_rendered_counter().fetch_add(delta, std::memory_order_relaxed);
    detail::flush_snapshot_file();
#else
    (void)delta;
#endif
}

} // namespace LaunchGraphCounters
} // namespace JuicerCuda
