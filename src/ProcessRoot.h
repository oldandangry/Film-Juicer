#pragma once

#include <mutex>
#include <string>

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
        bool retire_reset_context(int deviceId, void* contextOpaque, std::string& outError) noexcept;
        JuicerAssets::Library& assets() noexcept;

    private:
        Root() = default;

        bool retire_idle_context(int deviceId, void* contextOpaque, std::string& outError) noexcept;

        std::once_flag _bootstrapOnce;
        JuicerAssets::Library _assets;
    };

    Root& root() noexcept;

} // namespace JuicerProcess
