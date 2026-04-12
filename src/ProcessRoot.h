#pragma once

#include <mutex>
#include <string>

#include "ResourceAssetLibrary.h"

struct InstanceState;

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
namespace JuicerCuda {
    struct Resources;
}
#endif

namespace JuicerProcess {

    using BootstrapFn = void (*)();

    class Root {
    public:
        static Root& instance() noexcept;

        Root(const Root&) = delete;
        Root& operator=(const Root&) = delete;

        void ensure_bootstrap(BootstrapFn callback);
        void retire_idle_contexts(InstanceState& state) noexcept;
        bool retire_idle_context(int deviceId, void* contextOpaque, std::string& outError) noexcept;
        bool retire_reset_context(int deviceId, void* contextOpaque, std::string& outError) noexcept;
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        void destroy_cuda_resources(JuicerCuda::Resources* resources) noexcept;
#endif
        JuicerAssets::Library& assets() noexcept;

    private:
        Root() = default;

        std::once_flag _bootstrapOnce;
        JuicerAssets::Library _assets;
    };

    Root& root() noexcept;

} // namespace JuicerProcess
