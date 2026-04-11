#pragma once

#include <mutex>

#include "ResourceAssetLibrary.h"

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
        JuicerAssets::Library& assets() noexcept;

    private:
        Root() = default;

        std::once_flag _bootstrapOnce;
        JuicerAssets::Library _assets;
    };

    Root& root() noexcept;

} // namespace JuicerProcess
