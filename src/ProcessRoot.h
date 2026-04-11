#pragma once

#include <mutex>

struct InstanceState;

namespace JuicerProcess {

    using BootstrapFn = void (*)();

    class Root {
    public:
        static Root& instance() noexcept;

        Root(const Root&) = delete;
        Root& operator=(const Root&) = delete;

        void ensure_bootstrap(BootstrapFn callback);
        void retire_idle_contexts(InstanceState& state) noexcept;

    private:
        Root() = default;

        std::once_flag _bootstrapOnce;
    };

    Root& root() noexcept;

} // namespace JuicerProcess
