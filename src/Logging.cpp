#include "Logging.h"

#if JUICER_DIAGNOSTICS_COMPILED

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <string_view>
#include <utility>

namespace JuicerLogging {
    namespace {

        int clamp_level(int level) {
            if (level < 0)
                return 0;
            if (level > 3)
                return 3;
            return level;
        }
        int parse_env_int(const char* name, int fallback) {
            if (!name)
                return fallback;
            const char* v = std::getenv(name);
            if (!v || !*v)
                return fallback;
            char* end = nullptr;
            long parsed = std::strtol(v, &end, 10);
            if (end == v)
                return fallback;
            if (parsed < static_cast<long>(std::numeric_limits<int>::min()))
                return fallback;
            if (parsed > static_cast<long>(std::numeric_limits<int>::max()))
                return fallback;
            return static_cast<int>(parsed);
        }
        std::atomic<int>& diagnostics_level_storage() {
            static std::atomic<int> level{-1};
            return level;
        }
        std::once_flag& init_once_flag() {
            static std::once_flag flag;
            return flag;
        }
        std::optional<std::filesystem::path>& configured_path_storage() {
            static std::optional<std::filesystem::path> path;
            return path;
        }
        std::filesystem::path default_log_path() {
            namespace fs = std::filesystem;
            try {
                fs::path base = fs::temp_directory_path();
                base /= "juicer_trace.txt";
                base.make_preferred();
                return base;
            } catch (...) {
                return fs::path();
            }
        }
        std::filesystem::path configured_log_path() {
            const auto& cfg = configured_path_storage();
            if (cfg && !cfg->empty()) {
                return *cfg;
            }
            return default_log_path();
        }
        void init_from_environment_once() {
            std::call_once(init_once_flag(), []() {
                // Default: OFF
                int level = clamp_level(parse_env_int("JUICER_DIAGNOSTICS", 0));
                diagnostics_level_storage().store(level, std::memory_order_release);
                const char* pathEnv = std::getenv("JUICER_DIAGNOSTICS_PATH");
                if (pathEnv && *pathEnv) {
                    try {
                        std::filesystem::path p(pathEnv);
                        if (!p.empty()) {
                            p.make_preferred();
                            configured_path_storage() = std::move(p);
                        }
                    } catch (...) {
                        discard_current_exception();
                        configured_path_storage().reset();
                    }
                }
            });
        }
        int diagnostics_level() {
            int level = diagnostics_level_storage().load(std::memory_order_acquire);
            if (level >= 0) {
                return level;
            }
            init_from_environment_once();
            level = diagnostics_level_storage().load(std::memory_order_acquire);
            return (level >= 0) ? level : 0;
        }
        std::uint64_t max_log_bytes() {
            // Keep logs bounded even if verbose diagnostics are accidentally enabled.
            return 64ull * 1024ull * 1024ull;
        }
        int max_rotated_files() {
            return 3;
        }
        struct Sink {
            std::mutex m;
            std::ofstream stream;
            std::filesystem::path path;
            std::uint64_t sizeBytes = 0;
            std::uint64_t droppedLines = 0;
            unsigned int linesSinceFlush = 0;
            bool initialized = false;
        };
        Sink& sink() {
            static Sink s;
            return s;
        }
        std::uint64_t file_size_or_zero(const std::filesystem::path& path) {
            std::error_code ec;
            const auto size = std::filesystem::file_size(path, ec);
            return ec ? 0 : static_cast<std::uint64_t>(size);
        }
        std::filesystem::path rotated_path(const std::filesystem::path& base, int index) {
            if (index <= 0) {
                return base;
            }
            std::filesystem::path p = base;
            p += "." + std::to_string(index);
            return p;
        }
        void rotate_logs_locked(Sink& s) {
            if (s.path.empty()) {
                return;
            }
            s.stream.close();
            const int keep = std::max(0, max_rotated_files());
            if (keep > 0) {
                std::error_code ec;
                {
                    std::filesystem::path oldest = rotated_path(s.path, keep);
                    if (std::filesystem::exists(oldest, ec)) {
                        std::filesystem::remove(oldest, ec);
                    }
                }
                for (int i = keep - 1; i >= 1; --i) {
                    std::filesystem::path src = rotated_path(s.path, i);
                    std::filesystem::path dst = rotated_path(s.path, i + 1);
                    if (std::filesystem::exists(dst, ec)) {
                        std::filesystem::remove(dst, ec);
                    }
                    if (std::filesystem::exists(src, ec)) {
                        std::filesystem::rename(src, dst, ec);
                    }
                }
                {
                    std::filesystem::path dst = rotated_path(s.path, 1);
                    if (std::filesystem::exists(dst, ec)) {
                        std::filesystem::remove(dst, ec);
                    }
                    if (std::filesystem::exists(s.path, ec)) {
                        std::filesystem::rename(s.path, dst, ec);
                    }
                }
            } else {
                std::error_code ec;
                if (std::filesystem::exists(s.path, ec)) {
                    std::filesystem::remove(s.path, ec);
                }
            }
            s.sizeBytes = 0;
            s.linesSinceFlush = 0;
            s.initialized = false;
        }
        void ensure_sink_locked(Sink& s) {
            if (s.initialized) {
                return;
            }
            s.initialized = true;
            const std::filesystem::path path = configured_log_path();
            if (path.empty()) {
                return;
            }
            s.path = path;
            std::error_code ec;
            std::filesystem::create_directories(path.parent_path(), ec);
            const std::uint64_t existingBytes = file_size_or_zero(path);
            if (existingBytes >= max_log_bytes()) {
                rotate_logs_locked(s);
                s.path = path;
            }
            s.sizeBytes = file_size_or_zero(path);
            s.stream.open(path, std::ios::out | std::ios::app | std::ios::binary);
            if (!s.stream.is_open()) {
                return;
            }
            // One-line header on first enable to make logs self-identifying.
            const auto now = std::chrono::system_clock::now();
            const auto secs = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
            std::string header = "INIT | diagnostics enabled level=" + std::to_string(diagnostics_level()) + " time_s=" + std::to_string(secs);
            s.stream << header << '\n';
            s.sizeBytes += static_cast<std::uint64_t>(header.size()) + 1ull;
            ++s.linesSinceFlush;
            s.stream.flush();
        }
        void write_unchecked_locked(Sink& s, const char* tag, std::string_view msg) {
            if (!s.stream.is_open()) {
                return;
            }
            const std::uint64_t lineBytes =
                static_cast<std::uint64_t>(std::char_traits<char>::length(tag)) +
                3ull +
                static_cast<std::uint64_t>(msg.size()) +
                1ull;
            if (s.sizeBytes + lineBytes >= max_log_bytes()) {
                rotate_logs_locked(s);
                ensure_sink_locked(s);
            }
            if (!s.stream.is_open()) {
                ++s.droppedLines;
                return;
            }
            s.stream << tag << " | " << msg << '\n';
            s.sizeBytes += lineBytes;
            ++s.linesSinceFlush;
            // Avoid flush-per-line; keep interactivity by periodic flushing only.
            if (s.linesSinceFlush >= 64) {
                s.stream.flush();
                s.linesSinceFlush = 0;
            }
        }

    } // namespace

    bool enabled(int minLevel) {
        return diagnostics_level() >= minLevel;
    }

    void write(const char* tag, const std::string& msg) {
        if (!tag) {
            return;
        }
        // Gate here as well for safety; macros should prevent calls when disabled.
        if (!enabled(1)) {
            return;
        }
        const int level = diagnostics_level();
        if (level == 1) {
            // Best-effort "errors only" filter at level 1 without requiring call site changes.
            // Convention in this codebase: error logs typically contain "FATAL:".
            if (msg.find("FATAL") == std::string::npos && msg.find("ERROR") == std::string::npos) {
                return;
            }
        }
        Sink& s = sink();
        std::lock_guard<std::mutex> lock(s.m);
        ensure_sink_locked(s);
        write_unchecked_locked(s, tag, msg);
    }

    Scope::Scope(const char* t, std::string n) : tag(t), name(std::move(n)) {
        active = enabled(2);
        if (active) {
            write(tag, "BEGIN " + name);
        }
    }

    Scope::~Scope() noexcept {
        try {
            if (active) {
                write(tag, "END   " + name);
            }
        } catch (...) {
            discard_current_exception();
        }
    }

} // namespace JuicerLogging

#endif
