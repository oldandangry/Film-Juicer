#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include "ResourceAssetLibrary.h"

struct InstanceState;

namespace WorkingStateSharing {
    struct AcquireCoreSharedResult;
    struct WorkingStateCorePayload;
} // namespace WorkingStateSharing

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
namespace JuicerCuda {
    struct Resources;
}
#endif

namespace JuicerProcess {

    class Root {
    public:
        static Root& instance() noexcept;

        Root(const Root&) = delete;
        Root& operator=(const Root&) = delete;

        void ensure_bootstrap();
        void shutdown() noexcept;
        void retire_idle_contexts(InstanceState& state) noexcept;
        bool retire_idle_context(int deviceId, void* contextOpaque, std::string& outError) noexcept;
        bool retire_reset_context(int deviceId, void* contextOpaque, std::string& outError) noexcept;
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        void destroy_cuda_resources(JuicerCuda::Resources* resources) noexcept;
#endif
        JuicerAssets::Library& assets() noexcept;
        WorkingStateSharing::AcquireCoreSharedResult acquire_working_state_core(
            std::uint64_t keyHash,
            std::shared_ptr<const WorkingStateSharing::WorkingStateCorePayload> insertPayload = nullptr);

    private:
        Root() = default;

        void release_working_state_cores() noexcept;

        std::once_flag _bootstrapOnce;
        JuicerAssets::Library _assets;
    };

    Root& root() noexcept;

} // namespace JuicerProcess
