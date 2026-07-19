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
#include "FilmJuicerEffectsFrameDescriptor.h"
#include "GrainStaticActivityState.h"
#include "VisualGrainFrameDescriptor.h"

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
#include "Cuda/JuicerCudaDirectFilmPayloads.h"
#include "Cuda/JuicerCudaResources.h"
#endif

struct InstanceState;
struct WorkingState;
namespace Print {
    struct Params;
    struct Runtime;
} // namespace Print
namespace Scanner {
    struct ScannerPostEffectsDescriptor;
} // namespace Scanner

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
        void retire_grain_static_instance(std::uint64_t instanceToken) noexcept;
        FramePreparationToken begin_frame_preparation() noexcept;
        bool retire_idle_context(int deviceId, void* contextOpaque, std::string& outError) noexcept;
        bool retire_reset_context(int deviceId, void* contextOpaque, std::string& outError) noexcept;
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        struct AutoExposureBufferRequest {
            bool enabled = false;
            JuicerCuda::AutoExposurePreviewDescriptor descriptor{};
            std::uint64_t reusableKeyHash = 0;
        };

        struct DirectCudaPreparationRequest {
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
            bool needCompositeProfileWorkspace = false;
        };

        struct PrintCudaPreparationRequest {
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
            bool needCompositeProfileWorkspace = false;
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
                std::uint64_t version = 0;
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
                    const float* densityCurvesLayers[3][3] = {{nullptr, nullptr, nullptr}, {nullptr, nullptr, nullptr}, {nullptr, nullptr, nullptr}};
                    const float* tablesAx = nullptr;
                    const float* tablesAy = nullptr;
                    const float* tablesAz = nullptr;
                    const float* tablesIllum = nullptr;
                    int tablesK = 0;
                    float spdSInv[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
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
                    float printPreflashRaw[3] = {0.0f, 0.0f, 0.0f};
                };

                FilmRuntimeView film{};
                ScanRuntimeView scan{};
                PrintRuntimeView print{};
            };

            struct KernelView {
                const float* weights = nullptr;
                int radius = 0;
                float sigma = 0.0f;
            };

            struct PreparedGaussianView {
                KernelView kernel{};
                std::uint64_t descriptorHash = 0;
                bool active = false;
            };

            struct CaptureFilmDensityWorkspaceView {
                float* c = nullptr;
                float* m = nullptr;
                float* y = nullptr;
                Spektrafilm::ProfilePolarity polarity =
                    Spektrafilm::ProfilePolarity::Unsupported;
                bool overflow = false;
                bool active = false;
            };

            struct FocusedRgbWorkspaceView {
                float* r = nullptr;
                float* g = nullptr;
                float* b = nullptr;
                bool overflow = false;
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
                bool overflow = false;
                bool active = false;
            };

            struct PreparedDensityLayersView {
                JuicerCuda::DeviceCurveView baseCurvesCmy[3] = {};
                const float* curves[3][3] = {
                    {nullptr, nullptr, nullptr},
                    {nullptr, nullptr, nullptr},
                    {nullptr, nullptr, nullptr}};
                std::uint64_t hash = 0;
                bool active = false;
            };

            struct PreparedVisualGrainView {
                GrainStaticAssets staticNoise{};
                std::array<PreparedGaussianView, 3> correlation{};
                PreparedGaussianView dyeCloud[3][3] = {};
                PreparedDensityLayersView densityLayers{};
                const Spektrafilm::VisualGrainFrameDescriptor* descriptor = nullptr;
                bool active = false;
            };

            // Kernel views expose only prepared-frame-owned device memory for the current launch.
            struct OpticsKernelView {
                KernelView spatialDir{};
                KernelView scannerLensBlur{};
                KernelView scannerUnsharp{};
                KernelView scannerGlare{};
                KernelView halation[3] = {};
                KernelView halationScatter[3] = {};
            };

            struct AutoExposureBufferView {
                JuicerCudaAutoExposureScratch scratch{};
                JuicerCudaAutoExposureDeviceState deviceState{};
                int weightsWidth = 0;
                int weightsHeight = 0;
                std::uint64_t keyHash = 0;
                bool active = false;
            };

            struct UploadTraceView {
                std::uint64_t uploadedBuildCounter = 0;
                std::uint64_t printIllumBuildCounter = 0;
                std::uint64_t printIllumCoreHash = 0;
                std::uint64_t printIllumNeutralFilterHash = 0;
                float printIllumYShiftSteps = 0.0f;
                float printIllumMShiftSteps = 0.0f;
                float printIllumCShiftSteps = 0.0f;
                bool printPreflashValid = false;
                std::uint64_t printPreflashKeyHash = 0;
                std::uint64_t directUploadCounter = 0;
                std::uint64_t finalSensitivityHash = 0;
                std::uint64_t densityCurvesHash = 0;
                std::uint64_t densityBoundsHash = 0;
                std::uint64_t scannerDescriptorHash = 0;
                std::uint64_t printPreparationCounter = 0;
                std::uint64_t printPreparationDescriptorHash = 0;
                bool active = false;
            };

            struct DirectPreparedView {
                JuicerCuda::DirectFilmPreparedView film{};
                const JuicerCuda::Resources::DeviceScanMedium* scanMedium = nullptr;
                const JuicerCuda::Resources::DeviceSpectralLut* scanLut = nullptr;
                const Scanner::ColorRuntime* scannerColor = nullptr;
                std::uint64_t densityBoundsHash = 0;
                std::uint64_t scannerDescriptorHash = 0;
                Spektrafilm::RgbToRawMethod selectedMethod = Spektrafilm::RgbToRawMethod::Hanatos2025;
                bool active = false;
            };

            using PrintPreparedView = JuicerCuda::PrintPreparedView;

            struct PrintRoutePreparedView {
                JuicerCuda::DirectFilmPreparedView film{};
                const JuicerCuda::Resources::DeviceScanMedium* scanMedium = nullptr;
                const JuicerCuda::Resources::DeviceSpectralLut* scanLut = nullptr;
                const Scanner::ColorRuntime* scannerColor = nullptr;
                std::uint64_t densityBoundsHash = 0;
                std::uint64_t scannerDescriptorHash = 0;
                Spektrafilm::RgbToRawMethod selectedMethod = Spektrafilm::RgbToRawMethod::Hanatos2025;
                bool active = false;
            };

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
                bool overflow = false;
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
            WorkspaceRequest workspace_request() const noexcept;
            WorkspaceLeaseMarker bind_workspace_request(const WorkspaceRequest& request) const noexcept;
            CaptureFilmDensityWorkspaceView capture_film_density_workspace(
                const WorkspaceLeaseMarker& workspace) const noexcept;
            FocusedRgbWorkspaceView focused_rgb_workspace(
                const WorkspaceLeaseMarker& workspace) const noexcept;
            PreparedVisualGrainView visual_grain_resources() const noexcept;
            const Spektrafilm::FilmJuicerEffectsFrameDescriptor*
            film_juicer_effects_descriptor() const noexcept;
            VisualGrainWorkspaceView visual_grain_workspace(
                const WorkspaceLeaseMarker& workspace) const noexcept;
            DurableBundleView durable_bundle() const noexcept;
            OpticsKernelView optics_kernels() const noexcept;
            AutoExposureBufferView auto_exposure_buffers() const noexcept;
            UploadTraceView upload_trace_view() const noexcept;
            DirectPreparedView direct_resources() const noexcept;
            PrintPreparedView print_resources() const noexcept;
            PrintRoutePreparedView print_route_resources() const noexcept;
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
            void abort(const char* reason) noexcept;
            // Broad medium/LUT preparation remains internal to prepared-frame ownership.
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
            bool stage_optical_workspace(
                const WorkspaceLeaseMarker& workspace,
                void* cudaStreamOpaque,
                std::string& outError);
            bool try_stage_profile_optical_workspace(
                const WorkspaceLeaseMarker& workspace,
                void* cudaStreamOpaque,
                std::string& outError);
            bool prepare_spatial_dir_resources(
                const Spektrafilm::SpatialDirDescriptor& descriptor,
                const WorkspaceLeaseMarker& workspace,
                void* cudaStreamOpaque,
                std::string& outError);
            bool release_spatial_dir_build_scratch_after_build(
                const WorkspaceLeaseMarker& workspace,
                void* cudaStreamOpaque,
                std::string& outError);
            bool stage_spatial_dir_cached_log_raw_for_final_develop(
                const WorkspaceLeaseMarker& workspace,
                void* cudaStreamOpaque,
                std::string& outError);
            bool release_spatial_dir_cached_log_raw_after_final_develop(
                const WorkspaceLeaseMarker& workspace,
                void* cudaStreamOpaque,
                std::string& outError);
            bool release_spatial_dir_stage_after_scanner_output(
                const WorkspaceLeaseMarker& workspace,
                void* cudaStreamOpaque,
                std::string& outError);
            bool build_print_illuminant_filter_curve(
                const WorkingState& workingState,
                const Print::Runtime& printRt,
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
            bool checkpoint_large_scratch_transition(
                const WorkspaceLeaseMarker& workspace,
                void* cudaStreamOpaque,
                const char* stageTag,
                std::string& outError);
            bool build_lens_blur_kernel(
                float sigma,
                void* cudaStreamOpaque,
                std::string& outError);
            bool build_unsharp_kernel(
                float sigma,
                void* cudaStreamOpaque,
                std::string& outError);
            bool build_glare_kernel(
                float sigma,
                void* cudaStreamOpaque,
                std::string& outError);
            bool build_halation_kernel(
                int channel,
                float sigma,
                void* cudaStreamOpaque,
                std::string& outError);
            bool build_halation_scatter_kernel(
                int channel,
                float sigma,
                void* cudaStreamOpaque,
                std::string& outError);
            const char* failure_stage_tag() const noexcept;
            const char* failure_prefix() const noexcept;
            bool failure_marks_context_loss() const noexcept;

        private:
            friend class Root;

            struct State;

            explicit PreparedCudaFrame(std::unique_ptr<State> state) noexcept;

            static JuicerCuda::ResourceManager::ScratchRequestDescriptor make_scratch_request_descriptor(
                const WorkspaceLeaseMarker& workspace) noexcept;
            const WorkspaceRequest& active_workspace_request(
                const WorkspaceLeaseMarker& workspace) const noexcept;
            bool workspace_marker_matches_current_frame(const WorkspaceLeaseMarker& workspace) const noexcept;
            bool validate_workspace_lease_marker(
                const WorkspaceLeaseMarker& workspace,
                std::string& outError) const;
            bool build_gaussian_kernel_slot(
                JuicerCuda::Resources::DeviceGaussianKernel& kernel,
                float sigma,
                void* cudaStreamOpaque,
                std::string& outError);
            bool build_halation_kernel_slot(
                JuicerCuda::Resources::DeviceGaussianKernel& kernel,
                float sigma,
                void* cudaStreamOpaque,
                std::string& outError);
            bool prepare_scanner_post_effects(
                const Scanner::ScannerPostEffectsDescriptor& descriptor,
                void* cudaStreamOpaque,
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
            const WorkingState& workingState,
            const AutoExposureBufferRequest& autoExposureBufferRequest,
            void* cudaStreamOpaque,
            std::string& outError);
        PreparedCudaFrame prepare_cuda_frame(
            const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
            const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
            const PrintCudaPreparationRequest& request,
            const AutoExposureBufferRequest& autoExposureBufferRequest,
            void* cudaStreamOpaque,
            std::string& outError);
        PreparedCudaFrame prepare_cuda_frame(
            const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
            const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
            const DirectCudaPreparationRequest& request,
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
        bool shed_post_frame_scratch(
            JuicerCuda::ResourceManager::SubmissionTransaction& transaction,
            JuicerCuda::Resources& resources,
            const JuicerCuda::ResourceManager::ScratchRequestDescriptor& scratchRequest,
            void* cudaStreamOpaque,
            const char* commandName,
            std::string& outError);
        void rollback_submission(
            JuicerCuda::ResourceManager::SubmissionTransaction& transaction,
            const char* reason) noexcept;
#endif
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
        void release_cuda_host_asset_caches() noexcept;
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

        struct GrainStaticContextEntry final {
            CudaResourceOwner owner;
            detail::GrainStaticInstanceMap instances;
        };

        using GrainStaticContextMap = std::unordered_map<
            ContextCudaResourceKey,
            GrainStaticContextEntry,
            ContextCudaResourceKeyHash>;

        bool apply_grain_static_membership_and_copy_owner(
            const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
            std::uint64_t contextEpoch,
            const std::shared_ptr<JuicerCuda::DeviceAllocationLedger>& deviceLedger,
            std::uint64_t instanceToken,
            std::uint64_t registryGeneration,
            std::uint64_t snapshotId,
            bool active,
            detail::GrainStaticMembershipChange& outChange,
            CudaResourceOwner& outOwner,
            JuicerCuda::Resources*& outResources,
            std::string& outError);
        void rollback_grain_static_membership(
            const ContextCudaResourceKey& contextKey,
            const detail::GrainStaticMembershipChange& change) noexcept;
        void retire_grain_static_context_activity(
            const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey) noexcept;

        bool resolve_context_cuda_resources(
            const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
            std::uint64_t contextEpoch,
            const std::shared_ptr<JuicerCuda::DeviceAllocationLedger>& deviceLedger,
            ContextCudaResourceMap& contextMap,
            CudaResourceOwner& outResourceOwner,
            JuicerCuda::Resources*& outResources,
            std::string& outError);
        bool resolve_cuda_frame_resources(
            const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
            std::uint64_t contextEpoch,
            const JuicerCuda::ResourceManager::ResolvedPressurePolicy& pressurePolicy,
            CudaResourceOwner& outResourceOwner,
            JuicerCuda::Resources*& outResources,
            std::string& outError);
        bool retire_cuda_context(
            const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
            bool contextReset,
            std::string& outError) noexcept;
#endif

        bool _acceptFramePreparation = true;
        bool _shutdownRetireBlocked = false;
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        std::mutex _cudaResourcesMutex;
        std::unordered_map<int, std::shared_ptr<JuicerCuda::DeviceAllocationLedger>>
            _cudaDeviceLedgers;
        ContextCudaResourceMap _cudaResourcesByContext;
        GrainStaticContextMap _grainStaticByContext;
#endif
    };

    Root& root() noexcept;

} // namespace JuicerProcess
