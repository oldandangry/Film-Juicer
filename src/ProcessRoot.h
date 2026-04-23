#pragma once

#include <cstdint>
#include <functional>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

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

            struct DurableBundleView {
                struct FilmRuntimeView {
                    const JuicerCuda::DeviceCurve* densB = nullptr;
                    const JuicerCuda::DeviceCurve* densG = nullptr;
                    const JuicerCuda::DeviceCurve* densR = nullptr;
                    const JuicerCuda::DeviceCurve* dirDensB = nullptr;
                    const JuicerCuda::DeviceCurve* dirDensG = nullptr;
                    const JuicerCuda::DeviceCurve* dirDensR = nullptr;
                    const JuicerCuda::DeviceCurve* sensB = nullptr;
                    const JuicerCuda::DeviceCurve* sensG = nullptr;
                    const JuicerCuda::DeviceCurve* sensR = nullptr;
                    bool hasDensityCurvesLayers = false;
                    const float* densityCurvesLayers[3][3] = { { nullptr, nullptr, nullptr }, { nullptr, nullptr, nullptr }, { nullptr, nullptr, nullptr } };
                    const float* tablesAx = nullptr;
                    const float* tablesAy = nullptr;
                    const float* tablesAz = nullptr;
                    const float* tablesIllum = nullptr;
                    int tablesK = 0;
                    float spdSInv[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                    const float* hanatosLut = nullptr;
                    int hanatosN = 0;
                    const float* hanatosLutIntegrated = nullptr;
                    int hanatosNIntegrated = 0;
                    const float* mallettBasis = nullptr;
                    int mallettBasisK = 0;
                };

                struct ScanRuntimeView {
                    const JuicerCuda::Resources::DeviceScanMedium* negativeMedium = nullptr;
                    const JuicerCuda::Resources::DeviceScanMedium* printMedium = nullptr;
                    const JuicerCuda::Resources::DeviceSpectralLut* negativeLut = nullptr;
                    const JuicerCuda::Resources::DeviceSpectralLut* printLut = nullptr;

                    const JuicerCuda::Resources::DeviceScanMedium* medium(bool negative) const noexcept {
                        return negative ? negativeMedium : printMedium;
                    }

                    const JuicerCuda::Resources::DeviceSpectralLut* lut(bool negative) const noexcept {
                        return negative ? negativeLut : printLut;
                    }
                };

                struct PrintRuntimeView {
                    const float* printIllumFiltered = nullptr;
                    int printIllumK = 0;
                    const JuicerCuda::DeviceCurve* printSensC = nullptr;
                    const JuicerCuda::DeviceCurve* printSensM = nullptr;
                    const JuicerCuda::DeviceCurve* printSensY = nullptr;
                    const JuicerCuda::DeviceCurve* printDcC = nullptr;
                    const JuicerCuda::DeviceCurve* printDcM = nullptr;
                    const JuicerCuda::DeviceCurve* printDcY = nullptr;
                    float printGammaC = 1.0f;
                    float printGammaM = 1.0f;
                    float printGammaY = 1.0f;
                    float printPreflashRaw[3] = { 0.0f, 0.0f, 0.0f };
                };

                FilmRuntimeView film{};
                ScanRuntimeView scan{};
                PrintRuntimeView print{};
            };

            PreparedCudaFrame(PreparedCudaFrame&& other) noexcept;
            PreparedCudaFrame& operator=(PreparedCudaFrame&& other) noexcept;

            bool active() const noexcept;
            WorkspaceLeaseMarker bind_workspace_request(const WorkspaceRequest& request) const noexcept;
            GrainStaticAssets grain_static_assets() const noexcept;
            DurableBundleView durable_bundle() const noexcept;
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
            bool prepare_scan_error_stage(
                int*& outScanErrorFlag,
                void* cudaStreamOpaque,
                std::string& outError);
            bool finalize_scan_error_stage(
                int* scanErrorFlag,
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

            JuicerCuda::Resources* runtime_resources() const noexcept;
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

        PreparedCudaFrame prepare_cuda_frame(
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
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        using CudaResourceOwner = std::shared_ptr<JuicerCuda::Resources>;

        struct CudaResourcesDeleter final {
            void operator()(JuicerCuda::Resources* resources) const noexcept;
        };

        struct ContextCudaResourceKey final {
            JuicerCuda::ResourceManager::DeviceContextKey deviceContextKey{};
            std::uint64_t contextEpoch = 0;

            bool operator==(const ContextCudaResourceKey& other) const noexcept {
                return deviceContextKey == other.deviceContextKey &&
                    contextEpoch == other.contextEpoch;
            }
        };

        struct ContextCudaResourceKeyHash final {
            std::size_t operator()(const ContextCudaResourceKey& key) const noexcept {
                const std::size_t hContext =
                    JuicerCuda::ResourceManager::DeviceContextKeyHash{}(key.deviceContextKey);
                const std::size_t hEpoch = std::hash<std::uint64_t>{}(key.contextEpoch);
                return hContext ^ (hEpoch + 0x9e3779b9u + (hContext << 6u) + (hContext >> 2u));
            }
        };

        using ContextCudaResourceMap = std::unordered_map<
            ContextCudaResourceKey,
            CudaResourceOwner,
            ContextCudaResourceKeyHash>;

        bool resolve_context_cuda_resources(
            const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
            std::uint64_t contextEpoch,
            ContextCudaResourceMap& contextMap,
            CudaResourceOwner& outResourceOwner,
            JuicerCuda::Resources*& outResources,
            std::string& outError);
        bool resolve_cuda_frame_resources(
            const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
            std::uint64_t contextEpoch,
            CudaResourceOwner& outResourceOwner,
            JuicerCuda::Resources*& outResources,
            std::string& outError);
        bool resolve_cuda_grain_static_resources(
            const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
            std::uint64_t contextEpoch,
            CudaResourceOwner& outResourceOwner,
            JuicerCuda::Resources*& outResources,
            std::string& outError);
#endif

        bool _acceptFramePreparation = true;
        bool _shutdownRetireBlocked = false;
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        std::mutex _cudaResourcesMutex;
        ContextCudaResourceMap _cudaResourcesByContext;
        ContextCudaResourceMap _cudaGrainStaticByContext;
#endif
    };

    Root& root() noexcept;

} // namespace JuicerProcess
