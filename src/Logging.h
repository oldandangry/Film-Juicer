#pragma once

#include <exception>

#ifndef JUICER_DIAGNOSTICS_COMPILED
#define JUICER_DIAGNOSTICS_COMPILED 1
#endif

#if JUICER_DIAGNOSTICS_COMPILED
#include <string>
#endif

namespace JuicerLogging {

    inline void discard_current_exception() noexcept {
        (void)std::current_exception();
    }

#if JUICER_DIAGNOSTICS_COMPILED
    // Runtime levels: 0 = off, 1 = errors, 2 = state changes, 3 = verbose.
    bool enabled(int minLevel);
    void write(const char* tag, const std::string& msg);

    struct Scope {
        const char* tag;
        std::string name;
        bool active = false;

        Scope(const char* t, std::string n);
        ~Scope() noexcept;
    };
#endif

} // namespace JuicerLogging

#if JUICER_DIAGNOSTICS_COMPILED
#define JTRACE_ENABLED(level) (::JuicerLogging::enabled((level)))
#define JTRACE_LEVEL(level, tag, msg)             \
    do {                                          \
        if (::JuicerLogging::enabled((level))) {  \
            ::JuicerLogging::write((tag), (msg)); \
        }                                         \
    } while (0)
#define JTRACE(tag, msg) JTRACE_LEVEL(1, (tag), (msg))
#define JTRACE_VERBOSE(tag, msg) JTRACE_LEVEL(3, (tag), (msg))
#define JTRACE_SCOPE(tag, name) ::JuicerLogging::Scope _juicer_scope_guard_((tag), (name))
#else
#define JTRACE_ENABLED(level) (false)
#define JTRACE_LEVEL(level, tag, msg) ((void)0)
#define JTRACE(tag, msg) ((void)0)
#define JTRACE_VERBOSE(tag, msg) ((void)0)
#define JTRACE_SCOPE(tag, name) ((void)0)
#endif
