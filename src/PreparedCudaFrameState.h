#pragma once

#include "ProcessRoot.h"
#include "Cuda/JuicerCudaExecutor.h"

namespace JuicerProcess {

    using JuicerCuda::ResourceManager::WorkspaceLayout;

    struct PreparedCudaFailureStage {
        const char* tag = nullptr;
    };

    struct Root::PreparedCudaFrame::State {
        struct AutoExposureFrameWorkspace {
            JuicerCudaAutoExposureScratch scratch{};
            JuicerCudaAutoExposureDeviceState deviceState{};
            int weightsWidth = 0;
            int weightsHeight = 0;
            int weightsXCapacity = 0;
            int weightsYCapacity = 0;
            Spektrafilm::AutoExposureMethod method = Spektrafilm::AutoExposureMethod::CenterWeighted;
            std::uint64_t keyHash = 0;
            bool active = false;
        };

        struct ScanErrorFrameStage {
            JuicerCuda::ScanErrorBuffer buffer;
            bool flagUsed = false;
            bool readbackPending = false;
            JuicerCuda::ScanErrorReadbackIdentity identity;
        };

        struct HighlightBoostFrameStage {
            JuicerCuda::HighlightBoostDescriptor descriptor{};
            JuicerCuda::HighlightBoostWorkspace* workspace = nullptr;
            JuicerCuda::HighlightBoostDisposition* hostStatus = nullptr;
            cudaEvent_t event = nullptr;
            bool copyPending = false;
            bool eventRecorded = false;
            bool accepted = false;
        };

        struct FrameScratchWorkspace {
            std::optional<WorkspaceLayout> admittedLayout;
            std::optional<WorkspaceLayout> postFrameLayout;
            bool retainedLeaseActive = false;
        };

        explicit State(JuicerCuda::PendingContextLossRecovery& callRecovery)
            : recovery(callRecovery) {
        }

        // The flat call outlives its prepared frame, including abort on unwind.
        JuicerCuda::PendingContextLossRecovery& recovery;
        Root* root = nullptr;
        JuicerCuda::Resources* resources = nullptr;
        JuicerCuda::Resources* grainStaticResources = nullptr;
        void* scatterHalationFilterBlock = nullptr;
        void* scatterHalationCarrierBlock = nullptr;
        float* scatterHalationFilterTemp = nullptr;
        float* scatterHalationWeightedAccumulation = nullptr;
        int scatterHalationFrameWidth = 0;
        int scatterHalationFrameHeight = 0;
        std::array<float, 9> focusedInputRgbToXyz{};
        std::array<float, 9> focusedInputXyzAdapt{};
        std::array<float, 9> focusedXyzToLinearSrgb{};
        bool focusedApplyInputChromaticAdapt = false;
        std::optional<Scanner::ColorRuntime> focusedScannerColor;
        JuicerCuda::ScannerSpectra focusedScannerSpectra{};
        std::optional<Gamut::OutputGamutTransform> focusedOutputGamutTransform;
        void* lastCudaStreamOpaque = nullptr;
        const char* failureStageTag = "prepare_frame";
        const char* failurePrefix = "CUDA prepared frame failed";
        Root::CudaResourceOwner resourceOwner;
        Root::CudaResourceOwner grainStaticOwner;
        JuicerCuda::TrackedDeviceAllocations
            deviceAllocationRecords;
        JuicerCuda::CameraFilmLinearExposurePlanes scatterHalationCarrier{};
        Scanner::ScannerPostEffectsDescriptor scannerPostEffectsDescriptor{};
        WorkspaceDerivation requestedWorkspace;
        ScanErrorFrameStage scanErrorStage{};
        HighlightBoostFrameStage highlightBoostStage{};
        std::array<JuicerCuda::LegacyGrainPreparedGaussianView, 3>
            preparedGrainCorrelation{};
        AutoExposureFrameWorkspace autoExposureWorkspace{};
        JuicerCuda::ResourceManager::SubmissionTransaction transaction{};
        FrameScratchWorkspace scratchWorkspace{};
        JuicerCuda::LegacyGrainPreparedDensityLayersView
            preparedGrainDensityLayers{};
        JuicerCuda::PrintResourceDescriptors printDescriptors{};
        std::array<float, 3> printPreflashRawCmy{};
        float printNormalizer = 1.0F;
        Spektrafilm::DiffusionExecutionDescriptor diffusionExecutionDescriptor{};
        JuicerCuda::LegacyGrainPreparedGaussianView
            preparedGrainDyeCloud[3][3] = {};
        std::optional<Spektrafilm::LegacyGrainFrameDescriptor> legacyGrainDescriptor;
        Spektrafilm::SpatialDirDescriptor spatialDirDescriptor{};
        JuicerCuda::Diffusion::PreparedDiffusionLease diffusionLease;
        std::optional<Spektrafilm::FilmJuicerEffectsFrameDescriptor> effectsDescriptor;
        std::optional<ScatterHalationFrameDescriptor> scatterHalationDescriptor;
        JuicerCuda::ScatterHalationCarrierSource scatterHalationCarrierSource =
            JuicerCuda::ScatterHalationCarrierSource::DedicatedPreparedPlanes;
        Spektrafilm::ProfilePolarity capturePolarity =
            Spektrafilm::ProfilePolarity::Unsupported;
        Spektrafilm::ScanRoute scanRoute =
            Spektrafilm::kDefaultScanRoute;
        bool legacyGrainPrepared = false;
        const JuicerCuda::GrainV2::FilmQuad* grainV2Bank = nullptr;
        void* grainV2ProxyWorkspace = nullptr;
        bool frameUseEventSubmitted = false;
        bool frameResourcesFenced = false;
        bool grainStaticResourcesFenced = false;

        void set_failure(
            PreparedCudaFailureStage stage,
            const char* prefix) noexcept {
            failureStageTag = stage.tag;
            failurePrefix = prefix;
        }

        void remember_stream(void* cudaStreamOpaque) noexcept {
            if (cudaStreamOpaque) {
                lastCudaStreamOpaque = cudaStreamOpaque;
            }
        }

        bool allocate_device_bytes(
            void** outPtr,
            std::size_t bytes,
            const char* label,
            JuicerCuda::Failure& outError);
        bool retire_device_bytes(
            void* ptr,
            void* cudaStreamOpaque,
            const char* label,
            JuicerCuda::Failure& outError);
        bool free_device_bytes_now(void* ptr, JuicerCuda::Failure& outError) noexcept;
        bool transfer_remaining_device_allocations(
            JuicerCuda::Failure& outError) noexcept;
        bool prepare_diffusion_resources(
            const JuicerCuda::ResourceManager::DeviceContextKey& contextKey,
            const Root::PreparationIdentity& identity,
            const Spektrafilm::DiffusionFrameSetDescriptor* frameSet,
            int requestedWidth,
            int requestedHeight,
            void* cudaStreamOpaque,
            JuicerCuda::Failure& outError);
        bool prepare_scatter_halation_resources(
            const JuicerCuda::ResourceManager::DeviceContextKey& contextKey,
            const Root::PreparationIdentity& identity,
            const ScatterHalationFrameDescriptor* descriptor,
            const JuicerCuda::HighlightBoostDescriptor* boost,
            const Spektrafilm::DiffusionFrameSetDescriptor* frameSet,
            int requestedWidth,
            int requestedHeight,
            JuicerCuda::Failure& outError);
        bool prepare_highlight_boost(
            const JuicerCuda::HighlightBoostDescriptor& descriptor,
            JuicerCuda::Failure& outError);
        bool release_scatter_halation_resources(
            void* cudaStreamOpaque,
            JuicerCuda::Failure& outError);

        bool allocate_auto_exposure_workspace(
            const JuicerCuda::AutoExposurePreviewDescriptor& descriptor,
            JuicerCuda::Failure& outError);
        bool allocate_scan_error_stage(JuicerCuda::Failure& outError);
        bool release_scan_error_stage_after_use(
            void* cudaStreamOpaque,
            JuicerCuda::Failure& outError);
        bool release_auto_exposure_workspace_after_use(
            void* cudaStreamOpaque,
            JuicerCuda::Failure& outError);
        JuicerCuda::ResourceManager::ScratchRequestDescriptor post_frame_scratch_request_descriptor() const noexcept;
        bool ensure_scratch_workspace(
            const WorkspaceLayout& request,
            void* cudaStreamOpaque,
            JuicerCuda::Failure& outError);
        bool release_scratch_workspace_after_use(
            void* cudaStreamOpaque,
            JuicerCuda::Failure& outError);
        bool submit_frame_use_event(
            void* cudaStreamOpaque,
            JuicerCuda::Failure& outError);
        void free_auto_exposure_workspace_now() noexcept;
    };

    bool spatial_dir_scratch_has_required_roles(
        const JuicerCuda::Resources::DeviceSpatialDirScratch& scratch,
        Spektrafilm::DirScratchTier tier,
        const Spektrafilm::DirScratchPlaneRoles& roles) noexcept;
    bool spatial_dir_scratch_has_final_develop_roles(
        const JuicerCuda::Resources::DeviceSpatialDirScratch& scratch,
        const Spektrafilm::DirScratchPlaneRoles& targetRoles) noexcept;

} // namespace JuicerProcess
