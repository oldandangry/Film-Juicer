#pragma once

#include <cstdint>
#include <functional>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#include "RenderRecipe.h"
#include "ResourceAssetLibrary.h"
#include "FilmEffectsFrameDescriptors.h"

#include "Cuda/JuicerCudaFilmPayloads.h"
#include "Cuda/JuicerCudaResources.h"

struct InstanceState;
namespace Scanner {
    struct ScannerPostEffectsDescriptor;
} // namespace Scanner

namespace JuicerCuda {
    struct PipelineRunParams;
    namespace ResourceManager {
        struct DeviceContextKey;
        struct ScratchRequestDescriptor;
        struct SubmissionSnapshot;
        struct SubmissionTransaction;
    } // namespace ResourceManager
} // namespace JuicerCuda

namespace JuicerProcess::detail {

    struct GrainStaticInstanceState final {
        std::uint64_t latestSnapshotId = 0;
        bool active = false;
    };

    using GrainStaticInstanceMap =
        std::unordered_map<std::uint64_t, GrainStaticInstanceState>;

    struct GrainStaticMembershipChange final {
        std::uint64_t instanceToken = 0;
        std::uint64_t appliedSnapshotId = 0;
        GrainStaticInstanceState previous{};
        bool hadPrevious = false;
        bool applied = false;
    };

    enum class GrainStaticMembershipResult : std::uint8_t {
        Applied,
        Idempotent,
        StaleSnapshot,
        ConflictingEqualSnapshot,
        InvalidInput
    };

    inline bool grain_static_has_active_instance(
        const GrainStaticInstanceMap& instances) noexcept {
        for (const auto& item : instances) {
            if (item.second.active) {
                return true;
            }
        }
        return false;
    }

    inline GrainStaticMembershipResult apply_grain_static_membership(
        GrainStaticInstanceMap& instances,
        std::uint64_t instanceToken,
        std::uint64_t snapshotId,
        bool active,
        GrainStaticMembershipChange& outChange) {
        outChange = GrainStaticMembershipChange{};
        if (instanceToken == 0 || snapshotId == 0) {
            return GrainStaticMembershipResult::InvalidInput;
        }

        const auto current = instances.find(instanceToken);
        if (current != instances.end()) {
            const GrainStaticInstanceState& state = current->second;
            if (snapshotId < state.latestSnapshotId) {
                return GrainStaticMembershipResult::StaleSnapshot;
            }
            if (snapshotId == state.latestSnapshotId) {
                return active == state.active
                           ? GrainStaticMembershipResult::Idempotent
                           : GrainStaticMembershipResult::ConflictingEqualSnapshot;
            }
            outChange.previous = state;
            outChange.hadPrevious = true;
        }

        instances[instanceToken] = GrainStaticInstanceState{snapshotId, active};
        outChange.instanceToken = instanceToken;
        outChange.appliedSnapshotId = snapshotId;
        outChange.applied = true;
        return GrainStaticMembershipResult::Applied;
    }

    inline bool rollback_grain_static_membership(
        GrainStaticInstanceMap& instances,
        const GrainStaticMembershipChange& change) {
        if (!change.applied || change.instanceToken == 0 ||
            change.appliedSnapshotId == 0) {
            return false;
        }
        const auto current = instances.find(change.instanceToken);
        if (current == instances.end() ||
            current->second.latestSnapshotId != change.appliedSnapshotId) {
            return false;
        }
        if (change.hadPrevious) {
            current->second = change.previous;
        } else {
            instances.erase(current);
        }
        return true;
    }

} // namespace JuicerProcess::detail

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

        void ensure_bootstrap();
        void shutdown() noexcept;
        void retire_grain_static_instance(std::uint64_t instanceToken) noexcept;
        FramePreparationToken begin_frame_preparation() noexcept;
        bool retire_idle_context(int deviceId, void* contextOpaque, std::string& outError) noexcept;
        bool retire_reset_context(int deviceId, void* contextOpaque, std::string& outError) noexcept;
        struct AutoExposureBufferRequest {
            bool enabled = false;
            JuicerCuda::AutoExposurePreviewDescriptor descriptor{};
        };

        struct CudaFramePreparationRequest {
            const Spektrafilm::RenderRecipe* recipe = nullptr;
            const Spectral::SpectralTables* exposureTables = nullptr;
            const float* spdSInv = nullptr;
            const Spectral::FilmRawConfig* filmRawConfig = nullptr;
            const Spectral::SpectralTables* scannerTables = nullptr;
            const Scanner::ColorRuntime* scannerColor = nullptr;
            const Scanner::ScannerSpectralLutDescriptor* scannerLutDescriptor = nullptr;
            const Scanner::ScannerPostEffectsDescriptor* scannerPostEffects = nullptr;
            const Spektrafilm::SpatialDirDescriptor* spatialDirDescriptor = nullptr;
            const Spektrafilm::DiffusionFrameSetDescriptor* diffusionFrameSetDescriptor = nullptr;
            std::optional<Spektrafilm::VisualGrainFrameDescriptor> visualGrainDescriptor;
            std::optional<Spektrafilm::FilmJuicerEffectsFrameDescriptor> effectsDescriptor;
            int requestedWidth = 0;
            int requestedHeight = 0;
        };

        class PreparedCudaFrame final {
        public:
            PreparedCudaFrame() noexcept = default;
            ~PreparedCudaFrame();

            PreparedCudaFrame(const PreparedCudaFrame&) = delete;
            PreparedCudaFrame& operator=(const PreparedCudaFrame&) = delete;

            // Shared frame scratch admission covers scanner post effects, grain, halation, and spatial DIR.
            struct WorkspaceRequest {
                bool needOptics = false;
                bool needSpatialDir = false;
                std::uint64_t spatialDirDescriptorHash = 0;
                Spektrafilm::DirScratchTier spatialDirScratchTier = Spektrafilm::DirScratchTier::Tier0;
                Spektrafilm::DirScratchPlaneRoles spatialDirPlaneRoles{};
                Spektrafilm::DirScratchTier spatialDirTargetScratchTier = Spektrafilm::DirScratchTier::Tier0;
                Spektrafilm::DirScratchPlaneRoles spatialDirTargetPlaneRoles{};
                int requestedWidth = 0;
                int requestedHeight = 0;
                bool needBlurred = false;
                bool aliasScannerRgbFromSpatialDirFiltered = false;
                bool needAux = false;
                bool needSharedTmp = false;
                bool needGrainFrameUniforms = false;
                bool needGrainLayerWork = false;
                bool needGrainShared = false;
                bool needGateMask = false;

                [[nodiscard]] bool has_any_family() const noexcept {
                    return needOptics || needSpatialDir;
                }
            };

            class WorkspaceLeaseMarker final {
            public:
                WorkspaceLeaseMarker() noexcept = default;

                bool active() const noexcept;

            private:
                friend class PreparedCudaFrame;

                explicit WorkspaceLeaseMarker(const WorkspaceRequest& request, std::uint64_t leaseGeneration) noexcept;

                WorkspaceRequest _request{};
                std::uint64_t _leaseGeneration = 0;
                bool _active = false;
            };

            struct KernelView {
                const float* weights = nullptr;
                int radius = 0;
                float sigma = 0.0f;
            };

            struct CaptureFilmDensityWorkspaceView {
                float* c = nullptr;
                float* m = nullptr;
                float* y = nullptr;
                Spektrafilm::ProfilePolarity polarity =
                    Spektrafilm::ProfilePolarity::Unsupported;
                bool active = false;
            };

            struct FocusedRgbWorkspaceView {
                float* r = nullptr;
                float* g = nullptr;
                float* b = nullptr;
                bool active = false;
            };

            struct VisualGrainWorkspaceView {
                float* filterTemp = nullptr;
                float* scaleWork = nullptr;
                float* deltaAccum = nullptr;
                float* layerWork = nullptr;
                float* sharedDelta = nullptr;
                JuicerCuda::GrainFrameUniforms* frameUniforms = nullptr;
                Spektrafilm::VisualGrainScratchShape scratchShape =
                    Spektrafilm::VisualGrainScratchShape::None;
                bool active = false;
            };

            struct AutoExposureBufferView {
                JuicerCudaAutoExposureScratch scratch{};
                JuicerCudaAutoExposureDeviceState deviceState{};
                int weightsWidth = 0;
                int weightsHeight = 0;
                std::uint64_t keyHash = 0;
                bool active = false;
            };

            struct FocusedPreparedView {
                JuicerCuda::FilmPreparedView film{};
                const JuicerCuda::Resources::DeviceScanMedium* scanMedium = nullptr;
                const JuicerCuda::Resources::DeviceSpectralLut* scanLut = nullptr;
                const Scanner::ColorRuntime* scannerColor = nullptr;
                std::uint64_t densityBoundsHash = 0;
                std::uint64_t scannerDescriptorHash = 0;
                Spektrafilm::RgbToRawMethod selectedMethod = Spektrafilm::RgbToRawMethod::Hanatos2025;
                bool active = false;
            };

            using PrintPreparedView = JuicerCuda::PrintPreparedView;

            struct SpatialDirScratchView {
                float* rawCorrectionY = nullptr;
                float* rawCorrectionM = nullptr;
                float* rawCorrectionC = nullptr;
                float* filteredCorrectionY = nullptr;
                float* filteredCorrectionM = nullptr;
                float* filteredCorrectionC = nullptr;
                float* filterTemp = nullptr;
                float* filterTempM = nullptr;
                float* filterTempC = nullptr;
                float* logRawB = nullptr;
                float* logRawG = nullptr;
                float* logRawR = nullptr;
                std::uint64_t descriptorHash = 0;
                Spektrafilm::DirScratchTier scratchTier = Spektrafilm::DirScratchTier::Tier0;
                Spektrafilm::DirScratchPlaneRoles planeRoles{};
                Spektrafilm::DirScratchTier targetScratchTier = Spektrafilm::DirScratchTier::Tier0;
                Spektrafilm::DirScratchPlaneRoles targetPlaneRoles{};
                bool active = false;
            };

            struct SpatialDirPreparedView {
                KernelView gaussian{};
                KernelView exponential[3]{};
                std::uint64_t descriptorHash = 0;
                bool active = false;
            };

            struct ScannerWorkspaceView {
                float* rgbR = nullptr;
                float* rgbG = nullptr;
                float* rgbB = nullptr;
                float* tmp = nullptr;
                float* blurred = nullptr;
                float* aux = nullptr;
                float* grainTmp = nullptr;
                float* grainTmpShared = nullptr;
                float* gateMask = nullptr;
                int gateMaskWidth = 0;
                int gateMaskHeight = 0;
                std::uint64_t gateMaskHash = 0;
                bool active = false;
                bool rgbAliasedFromSpatialDirFiltered = false;
                bool hasGateMask = false;
            };

            struct ScannerPostEffectsPreparedView {
                ScannerWorkspaceView scratch{};
                KernelView lensBlur{};
                KernelView unsharp{};
                KernelView glare{};
                std::uint64_t descriptorHash = 0;
                bool active = false;
            };

            PreparedCudaFrame(PreparedCudaFrame&& other) noexcept;
            PreparedCudaFrame& operator=(PreparedCudaFrame&& other) noexcept;

            bool active() const noexcept;
            WorkspaceLeaseMarker workspace_lease() const noexcept;
            CaptureFilmDensityWorkspaceView capture_film_density_workspace(
                const WorkspaceLeaseMarker& workspace) const noexcept;
            FocusedRgbWorkspaceView focused_rgb_workspace(
                const WorkspaceLeaseMarker& workspace) const noexcept;
            JuicerCuda::PreparedVisualGrainView visual_grain_resources() const noexcept;
            const Spektrafilm::FilmJuicerEffectsFrameDescriptor*
            film_juicer_effects_descriptor() const noexcept;
            VisualGrainWorkspaceView visual_grain_workspace(
                const WorkspaceLeaseMarker& workspace) const noexcept;
            AutoExposureBufferView auto_exposure_buffers() const noexcept;
            FocusedPreparedView focused_resources() const noexcept;
            PrintPreparedView print_resources() const noexcept;
            JuicerCuda::Diffusion::DiffusionPreparedView
            diffusion_resources() const noexcept;
            void mark_diffusion_work_enqueued() noexcept;
            bool release_diffusion_resources_after_use(
                void* cudaStreamOpaque,
                std::string& outError);
            SpatialDirScratchView spatial_dir_scratch(const WorkspaceLeaseMarker& workspace) const noexcept;
            SpatialDirPreparedView spatial_dir_resources(
                const WorkspaceLeaseMarker& workspace,
                std::uint64_t descriptorHash) const noexcept;
            ScannerWorkspaceView scanner_workspace(const WorkspaceLeaseMarker& workspace) const noexcept;
            ScannerPostEffectsPreparedView scanner_post_effects_resources(
                const WorkspaceLeaseMarker& workspace,
                std::uint64_t descriptorHash) const noexcept;
            struct AutoExposureWeightsExtent {
                int width = 0;
                int height = 0;
            };

            struct AutoExposureMeteredResult {
                std::uint64_t keyHash = 0;
            };

            void mark_auto_exposure_weights_built(const AutoExposureWeightsExtent& weights) noexcept;
            void mark_auto_exposure_metered(const AutoExposureMeteredResult& result) noexcept;
            void mark_gate_mask_built(std::uint64_t gateMaskHash) noexcept;
            bool record_use(
                void* cudaStreamOpaque,
                std::string& outError);
            bool finish(void* cudaStreamOpaque, std::string& outError);
            bool stage_optical_workspace(
                const WorkspaceLeaseMarker& workspace,
                void* cudaStreamOpaque,
                std::string& outError);
            bool prepare_spatial_dir_resources(
                const Spektrafilm::SpatialDirDescriptor& descriptor,
                const WorkspaceLeaseMarker& workspace,
                void* cudaStreamOpaque,
                std::string& outError);
            bool stage_spatial_dir_cached_log_raw_for_final_develop(
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
            bool checkpoint_large_scratch_transition(
                const WorkspaceLeaseMarker& workspace,
                void* cudaStreamOpaque,
                const char* stageTag,
                std::string& outError);
            const char* failure_stage_tag() const noexcept;
            const char* failure_prefix() const noexcept;

        private:
            friend class Root;

            struct State;

            explicit PreparedCudaFrame(std::unique_ptr<State> state) noexcept;
            void abort() noexcept;

            const WorkspaceRequest& active_workspace_request(
                const WorkspaceLeaseMarker& workspace) const noexcept;
            bool workspace_marker_matches_current_frame(const WorkspaceLeaseMarker& workspace) const noexcept;
            bool validate_workspace_lease_marker(
                const WorkspaceLeaseMarker& workspace,
                std::string& outError) const;
            bool build_gaussian_kernel_slot(
                JuicerCuda::Resources::DeviceGaussianKernel& kernel,
                float sigma,
                std::string& outError);
            bool prepare_scanner_post_effects(
                const Scanner::ScannerPostEffectsDescriptor& descriptor,
                std::string& outError);
            bool prepare_visual_grain_resources(
                Root& root,
                const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
                void* cudaStreamOpaque,
                std::string& outError);

            std::unique_ptr<State> _state;
        };

        // Prepared-frame construction receives immutable render inputs captured before CUDA launch.
        PreparedCudaFrame prepare_cuda_frame(
            const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
            const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
            const CudaFramePreparationRequest& request,
            const AutoExposureBufferRequest& autoExposureBufferRequest,
            void* cudaStreamOpaque,
            std::string& outError);
        JuicerAssets::Library& assets() noexcept;

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
        void release_cuda_context_resource_owners() noexcept;
        void release_process_host_services() noexcept;
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

        struct CudaContextResourceEntry final {
            CudaResourceOwner frameOwner;
            CudaResourceOwner grainOwner;
            detail::GrainStaticInstanceMap grainInstances;
        };

        using CudaContextResourceMap = std::unordered_map<
            ContextCudaResourceKey,
            CudaContextResourceEntry,
            ContextCudaResourceKeyHash>;

        bool begin_submission(
            JuicerCuda::ResourceManager::SubmissionTransaction& transaction,
            const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
            std::shared_ptr<JuicerCuda::DeviceAllocationLedger>& outDeviceLedger,
            std::string& outError);
        bool apply_grain_static_membership_and_copy_owner(
            const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
            std::uint64_t contextEpoch,
            const std::shared_ptr<JuicerCuda::DeviceAllocationLedger>& deviceLedger,
            std::uint64_t instanceToken,
            std::uint64_t snapshotId,
            bool active,
            detail::GrainStaticMembershipChange& outChange,
            CudaResourceOwner& outOwner,
            JuicerCuda::Resources*& outResources,
            std::string& outError);
        void rollback_grain_static_membership(
            const ContextCudaResourceKey& contextKey,
            const detail::GrainStaticMembershipChange& change) noexcept;

        bool resolve_context_cuda_resources(
            const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
            std::uint64_t contextEpoch,
            const std::shared_ptr<JuicerCuda::DeviceAllocationLedger>& deviceLedger,
            CudaResourceOwner& outResourceOwner,
            JuicerCuda::Resources*& outResources,
            std::string& outError);
        bool resolve_cuda_device_ledger(
            const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
            std::shared_ptr<JuicerCuda::DeviceAllocationLedger>& outDeviceLedger,
            std::string& outError);
        bool resolve_cuda_frame_resources(
            const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
            std::uint64_t contextEpoch,
            const JuicerCuda::ResourceManager::ResolvedMemoryBudget& memoryBudget,
            const std::shared_ptr<JuicerCuda::DeviceAllocationLedger>& deviceLedger,
            CudaResourceOwner& outResourceOwner,
            JuicerCuda::Resources*& outResources,
            std::string& outError);
        bool retire_cuda_context(
            const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
            bool contextReset,
            std::string& outError) noexcept;

        bool _acceptFramePreparation = true;
        bool _shutdownRetireBlocked = false;
        std::mutex _cudaResourcesMutex;
        std::unordered_map<int, std::shared_ptr<JuicerCuda::DeviceAllocationLedger>>
            _cudaDeviceLedgers;
        CudaContextResourceMap _cudaContextResources;
    };

    Root& root() noexcept;

} // namespace JuicerProcess
