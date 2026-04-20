#pragma once

#include <cstdint>
#include <condition_variable>
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
    namespace ResourceManager {
        struct DeviceContextKey;
        struct SubmissionSnapshot;
        struct SubmissionTransaction;
    } // namespace ResourceManager
} // namespace JuicerCuda
#endif

namespace JuicerProcess {

    class Root {
    public:
        class FramePreparationToken final {
        public:
            FramePreparationToken() noexcept = default;
            ~FramePreparationToken();

            FramePreparationToken(const FramePreparationToken&) = delete;
            FramePreparationToken& operator=(const FramePreparationToken&) = delete;

            FramePreparationToken(FramePreparationToken&& other) noexcept;
            FramePreparationToken& operator=(FramePreparationToken&& other) noexcept;

            bool active() const noexcept;

        private:
            friend class Root;

            explicit FramePreparationToken(Root* root) noexcept;
            void reset() noexcept;

            Root* _root = nullptr;
        };

        static Root& instance() noexcept;

        Root(const Root&) = delete;
        Root& operator=(const Root&) = delete;

        const std::string& data_dir() const noexcept;
        void ensure_bootstrap();
        void shutdown() noexcept;
        FramePreparationToken begin_frame_preparation() noexcept;
        void retire_idle_contexts(InstanceState& state) noexcept;
        bool retire_idle_context(int deviceId, void* contextOpaque, std::string& outError) noexcept;
        bool retire_reset_context(int deviceId, void* contextOpaque, std::string& outError) noexcept;
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        class PreparedCudaFrame final {
        public:
            PreparedCudaFrame() noexcept = default;
            ~PreparedCudaFrame();

            PreparedCudaFrame(const PreparedCudaFrame&) = delete;
            PreparedCudaFrame& operator=(const PreparedCudaFrame&) = delete;

            PreparedCudaFrame(PreparedCudaFrame&& other) noexcept;
            PreparedCudaFrame& operator=(PreparedCudaFrame&& other) noexcept;

            bool active() const noexcept;
            bool finish(void* cudaStreamOpaque, std::string& outError);
            void abort(const char* reason) noexcept;

            JuicerCuda::Resources* resources() const noexcept;
            JuicerCuda::ResourceManager::SubmissionTransaction& submission() noexcept;
            const char* failure_stage_tag() const noexcept;
            const char* failure_prefix() const noexcept;

        private:
            friend class Root;

            struct State;

            explicit PreparedCudaFrame(std::unique_ptr<State> state) noexcept;

            std::unique_ptr<State> _state;
        };

        void destroy_cuda_resources(JuicerCuda::Resources* resources) noexcept;
        PreparedCudaFrame prepare_cuda_frame(
            InstanceState& instanceState,
            const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
            const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
            std::string& outError);
        bool begin_submission(
            JuicerCuda::ResourceManager::SubmissionTransaction& transaction,
            const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
            std::string& outError);
        bool acquire_submission_plan(
            JuicerCuda::ResourceManager::SubmissionTransaction& transaction,
            std::string& outError);
        bool commit_submission(
            JuicerCuda::ResourceManager::SubmissionTransaction& transaction,
            void* cudaStreamOpaque,
            std::string& outError);
        void rollback_submission(
            JuicerCuda::ResourceManager::SubmissionTransaction& transaction,
            const char* reason) noexcept;
#endif
        JuicerAssets::Library& assets() noexcept;
        WorkingStateSharing::AcquireCoreSharedResult acquire_working_state_core(
            std::uint64_t keyHash,
            std::shared_ptr<const WorkingStateSharing::WorkingStateCorePayload> insertPayload = nullptr);

    private:
        class ShutdownToken final {
        public:
            ShutdownToken() noexcept = default;
            ~ShutdownToken();

            ShutdownToken(const ShutdownToken&) = delete;
            ShutdownToken& operator=(const ShutdownToken&) = delete;

            ShutdownToken(ShutdownToken&& other) noexcept;
            ShutdownToken& operator=(ShutdownToken&& other) noexcept;

        private:
            friend class Root;

            explicit ShutdownToken(Root* root) noexcept;
            void reset() noexcept;

            Root* _root = nullptr;
        };

        Root();

        ShutdownToken begin_shutdown() noexcept;
        bool retire_known_contexts(std::string& outError) noexcept;
        void release_cuda_host_asset_caches() noexcept;
        void release_process_host_services() noexcept;
        void release_working_state_cores() noexcept;
        void finish_shutdown() noexcept;
        void finish_frame_preparation() noexcept;
        void resume_frame_preparation() noexcept;
        void wait_for_frame_preparation() noexcept;
        void set_shutdown_retire_blocked(bool blocked) noexcept;

        std::once_flag _bootstrapOnce;
        std::string _dataDir;
        JuicerAssets::Library _assets;
        std::mutex _framePreparationMutex;
        std::condition_variable _framePreparationCv;
        std::uint32_t _activeFramePreparations = 0;
        std::uint32_t _activeShutdowns = 0;
        bool _acceptFramePreparation = true;
        bool _shutdownRetireBlocked = false;
    };

    Root& root() noexcept;

} // namespace JuicerProcess
