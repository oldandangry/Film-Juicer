#pragma once

#include <cstdint>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>

#include "ResourceAssetLibrary.h"

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
#include "Cuda/JuicerCudaResources.h"
#endif

struct InstanceState;
struct WorkingState;
namespace Print {
    struct Params;
    struct Runtime;
} // namespace Print

namespace WorkingStateSharing {
    struct AcquireCoreSharedResult;
    struct WorkingStateCorePayload;
} // namespace WorkingStateSharing

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
namespace JuicerCuda {
    struct PipelineRunParams;
    namespace ResourceManager {
        struct DeviceContextKey;
        struct ScratchRequestDescriptor;
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
        struct AutoExposureBufferRequest {
            bool enabled = false;
            int meterWidth = 0;
            int meterHeight = 0;
            std::uint64_t reusableKeyHash = 0;
        };

        class PreparedCudaFrame final {
        public:
            PreparedCudaFrame() noexcept = default;
            ~PreparedCudaFrame();

            PreparedCudaFrame(const PreparedCudaFrame&) = delete;
            PreparedCudaFrame& operator=(const PreparedCudaFrame&) = delete;

            struct WorkspaceRequest {
                bool needOptics = false;
                bool needSpatialDir = false;
                int requestedWidth = 0;
                int requestedHeight = 0;
                bool needBlurred = false;
                bool needAux = false;
                bool needGrainTriplet = false;
                bool needGrainShared = false;
                bool needGateMask = false;
            };

            class WorkspaceLeaseMarker final {
            public:
                WorkspaceLeaseMarker() noexcept = default;

                bool active() const noexcept;
                bool has_any_family() const noexcept;
                std::uint64_t lease_generation() const noexcept;

            private:
                friend class PreparedCudaFrame;

                explicit WorkspaceLeaseMarker(const WorkspaceRequest& request, std::uint64_t leaseGeneration) noexcept;

                WorkspaceRequest _request{};
                std::uint64_t _leaseGeneration = 0;
                bool _active = false;
            };

            struct GrainStaticAssets {
                const std::uint8_t* stbn = nullptr;
                int stbnWidth = 0;
                int stbnHeight = 0;
                int stbnFrames = 0;
                const std::uint8_t* wangTiles = nullptr;
                const std::uint8_t* wangLut = nullptr;
                int wangWidth = 0;
                int wangHeight = 0;
                int wangCount = 0;
                int wangColors = 0;
            };

            PreparedCudaFrame(PreparedCudaFrame&& other) noexcept;
            PreparedCudaFrame& operator=(PreparedCudaFrame&& other) noexcept;

            bool active() const noexcept;
            WorkspaceLeaseMarker bind_workspace_request(const WorkspaceRequest& request) const noexcept;
            GrainStaticAssets grain_static_assets() const noexcept;
            bool finish(void* cudaStreamOpaque, std::string& outError);
            void abort(const char* reason) noexcept;
            bool prepare_current_medium(
                const WorkingState& workingState,
                bool negativeMedium,
                const WorkspaceLeaseMarker& workspace,
                void* cudaStreamOpaque,
                std::string& outError);
            bool prepare_scan_lut(
                const WorkingState& workingState,
                bool negativeMedium,
                const WorkspaceLeaseMarker& workspace,
                void* cudaStreamOpaque,
                std::string& outError);
            bool prepare_optics_scratch(
                const WorkspaceLeaseMarker& workspace,
                void* cudaStreamOpaque,
                std::string& outError);
            bool prepare_spatial_dir_scratch(
                const WorkspaceLeaseMarker& workspace,
                void* cudaStreamOpaque,
                std::string& outError);
            bool prepare_print_illuminant_filtered(
                const WorkingState& workingState,
                const Print::Runtime& printRuntime,
                const Print::Params& printParams,
                const WorkspaceLeaseMarker& workspace,
                void* cudaStreamOpaque,
                std::string& outError);
            bool checkpoint_scratch_phase(
                const WorkspaceLeaseMarker& workspace,
                const char* stageTag,
                std::string& outError);
            bool prepare_spatial_dir_kernel(
                float sigma,
                void* cudaStreamOpaque,
                std::string& outError);
            bool prepare_gaussian_kernel(
                JuicerCuda::Resources::DeviceGaussianKernel& kernel,
                float sigma,
                void* cudaStreamOpaque,
                std::string& outError);
            bool prepare_halation_kernel(
                JuicerCuda::Resources::DeviceGaussianKernel& kernel,
                float sigma,
                void* cudaStreamOpaque,
                std::string& outError);
            bool launch_base_pipeline_graph(
                JuicerCuda::PipelineRunParams& run,
                int renderModeKey,
                void* cudaStreamOpaque,
                int& outCudaErrorCode,
                std::string& outError);

            JuicerCuda::Resources* resources() const noexcept;
            const char* failure_stage_tag() const noexcept;
            const char* failure_prefix() const noexcept;
            bool failure_marks_context_loss() const noexcept;

        private:
            friend class Root;

            struct State;

            explicit PreparedCudaFrame(std::unique_ptr<State> state) noexcept;

            static JuicerCuda::ResourceManager::ScratchRequestDescriptor make_scratch_request_descriptor(
                const WorkspaceLeaseMarker& workspace) noexcept;
            bool validate_workspace_lease_marker(
                const WorkspaceLeaseMarker& workspace,
                std::string& outError) const;

            std::unique_ptr<State> _state;
        };

        void destroy_cuda_resources(JuicerCuda::Resources* resources) noexcept;
        PreparedCudaFrame prepare_cuda_frame(
            InstanceState& instanceState,
            const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
            const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
            const WorkingState& workingState,
            const AutoExposureBufferRequest& autoExposureBufferRequest,
            void* cudaStreamOpaque,
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
