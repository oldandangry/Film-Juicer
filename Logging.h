// Logging.h
#pragma once

#include <atomic>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <filesystem>

namespace JuicerLogging {

    inline std::mutex& sink_mutex() {
        static std::mutex m;
        return m;
    }

    inline std::optional<std::filesystem::path>& configured_path() {
        static std::optional<std::filesystem::path> path;
        return path;
    }

    inline std::filesystem::path default_log_path() {
        namespace fs = std::filesystem;
        try {
            fs::path base = fs::temp_directory_path();
            base /= "juicer_trace.txt";
            base.make_preferred();
            return base;
        }
        catch (...) {
            return fs::path();
        }
    }

    inline std::filesystem::path current_log_path() {
        const auto& cfg = configured_path();
        if (cfg && !cfg->empty()) {
            return *cfg;
        }
        return default_log_path();
    }

    inline void set_log_path(const std::filesystem::path& path) {
        std::lock_guard<std::mutex> lock(sink_mutex());
        auto& cfg = configured_path();
        cfg = path;
    }

    inline bool& sink_disabled() {
        static bool disabled = false;
        return disabled;
    }

    inline void enable_logging(bool enabled) {
        std::lock_guard<std::mutex> lock(sink_mutex());
        sink_disabled() = !enabled;
    }

    inline std::ofstream& sink_stream() {
        static std::ofstream stream;
        return stream;
    }

    inline bool& sink_initialized() {
        static bool initialized = false;
        return initialized;
    }

    inline void ensure_sink_locked() {
        if (sink_initialized()) {
            return;
        }
        sink_initialized() = true;
        std::ofstream& stream = sink_stream();
        if (stream.is_open()) {
            return;
        }
        std::filesystem::path path = current_log_path();
        if (path.empty()) {
            sink_disabled() = true;
            return;
        }
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
        stream.open(path, std::ios::out | std::ios::app);
        if (!stream.is_open()) {
            sink_disabled() = true;
        }
    }

    inline void write(const char* tag, const std::string& msg) {
        if (!tag) {
            return;
        }
        std::lock_guard<std::mutex> lock(sink_mutex());
        if (sink_disabled()) {
            return;
        }
        ensure_sink_locked();
        std::ofstream& stream = sink_stream();
        if (!stream.is_open()) {
            return;
        }
        stream << tag << " | " << msg << '\n';
        stream.flush();
    }

    struct Scope {
        const char* tag;
        std::string name;
        Scope(const char* t, std::string n) : tag(t), name(std::move(n)) {
            write(tag, "BEGIN " + name);
        }
        ~Scope() {
            write(tag, "END   " + name);
        }
    };

} // namespace JuicerLogging

#define JTRACE(tag, msg) ::JuicerLogging::write((tag), (msg))
#define JTRACE_SCOPE(tag, name) ::JuicerLogging::Scope _juicer_scope_guard_((tag), (name))

// TEMP DEBUG: set to 0 or remove when print swap diagnostics are no longer needed.
#ifndef JUICER_TRACE_PRINT_SWAP
#define JUICER_TRACE_PRINT_SWAP 1
#endif
