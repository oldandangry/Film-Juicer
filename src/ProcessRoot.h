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
#include "Cuda/JuicerCudaDirectFilmPayloads.h"
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
        };

        struct PrintCudaPreparationRequest {
            const Spektrafilm::RenderRecipe* recipe = nullptr;
            const Spectral::SpectralTables* exposureTables = nullptr;
            const float* spdSInv = nullptr;
            const Spectral::FilmRawConfig* filmRawConfig = nullptr;
            const Spectral::SpectralTables* scannerTables = nullptr;
            const Scanner::ColorRuntime* scannerColor = nullptr;
            const Scanner::ScannerSpectralLutDescriptor* scannerLutDescriptor = nullptr;
        };

        class PreparedCudaFrame final {
        public:
            PreparedCudaFrame() noexcept = default;
            ~PreparedCudaFrame();

            PreparedCudaFrame(const PreparedCudaFrame&) = delete;
            PreparedCudaFrame& operator=(const PreparedCudaFrame&) = delete;

            // SF_TEMP_BRIDGE_Phase6BroadOpticsWorkspace owner=Phase8-scanner/Phase9-grain/legacy broad renderer;
            // allowed_call_sites=accepted spatial-DIR lease plus hard-blocked broad scanner/grain/halation source;
            // output_hash_resource_impact=no Exact policy/hash/resource ownership;
            // cleanup_symbol=SF_TEMP_BRIDGE_Phase6BroadOpticsWorkspace; disposition=split_by_owning_future_phase.
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

            // SF_TEMP_BRIDGE_Phase6BroadOpticsKernelView owner=Phase8-scanner/Phase9-grain/legacy broad renderer;
            // allowed_call_sites=hard-blocked broad launch source only;
            // output_hash_resource_impact=no Exact descriptor/prepared-view ownership;
            // cleanup_symbol=SF_TEMP_BRIDGE_Phase6BroadOpticsKernelView; disposition=split_by_owning_future_phase.
            struct OpticsKernelView {
                KernelView spatialDir{};
                KernelView scannerLensBlur{};
                KernelView scannerUnsharp{};
                KernelView scannerGlare{};
                KernelView grainBlur{};
                KernelView grainBlurMid{};
                KernelView grainBlurCoarse{};
                KernelView grainDye[3][3] = {};
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
                float* corrY = nullptr;
                float* corrM = nullptr;
                float* corrC = nullptr;
                float* mixY = nullptr;
                float* mixM = nullptr;
                float* mixC = nullptr;
                float* tmp = nullptr;
                bool active = false;
            };

            struct SpatialDirPreparedView {
                KernelView gaussian{};
                KernelView exponential[3]{};
                std::uint64_t descriptorHash = 0;
                bool active = false;
            };

            struct ScannerOpticsScratchView {
                float* rgbR = nullptr;
                float* rgbG = nullptr;
                float* rgbB = nullptr;
                float* tmp = nullptr;
                float* blurred = nullptr;
                float* aux = nullptr;
                float* grainTmp = nullptr;
                float* grainTmpShared = nullptr;
                float* grainTmpMid = nullptr;
                float* grainTmpCoarse = nullptr;
                float* gateMask = nullptr;
                int gateMaskWidth = 0;
                int gateMaskHeight = 0;
                std::uint64_t gateMaskHash = 0;
                bool active = false;
                bool hasGateMask = false;
            };

            PreparedCudaFrame(PreparedCudaFrame&& other) noexcept;
            PreparedCudaFrame& operator=(PreparedCudaFrame&& other) noexcept;

            bool active() const noexcept;
            WorkspaceLeaseMarker bind_workspace_request(const WorkspaceRequest& request) const noexcept;
            GrainStaticAssets grain_static_assets() const noexcept;
            DurableBundleView durable_bundle() const noexcept;
            OpticsKernelView optics_kernels() const noexcept;
            AutoExposureBufferView auto_exposure_buffers() const noexcept;
            UploadTraceView upload_trace_view() const noexcept;
            DirectPreparedView direct_resources() const noexcept;
            PrintPreparedView print_resources() const noexcept;
            PrintRoutePreparedView print_route_resources() const noexcept;
            SpatialDirScratchView spatial_dir_scratch(const WorkspaceLeaseMarker& workspace) const noexcept;
            SpatialDirPreparedView spatial_dir_resources(
                const WorkspaceLeaseMarker& workspace,
                std::uint64_t descriptorHash) const noexcept;
            ScannerOpticsScratchView scanner_optics_scratch(const WorkspaceLeaseMarker& workspace) const noexcept;
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
            void record_use(void* cudaStreamOpaque) noexcept;
            bool finish(void* cudaStreamOpaque, std::string& outError);
            void abort(const char* reason) noexcept;
            // SF_TEMP_BRIDGE_PreparedDirectScannerWorkingState owner=Phase4-print-route:
            // reason=legacy broad medium/LUT preparation; allowed=prepare_current_medium and
            // prepare_scan_lut calls in processImagesCUDA after the accepted direct return only;
            // output_impact=blocked print route; hash_impact=legacy scanner key;
            // resource_impact=broad WorkingState upload; removal=Phase4 print cutover.
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
            // SF_TEMP_BRIDGE_Phase6PublicOpticsPreparation owner=Phase8-scanner/Phase9-grain/legacy broad renderer;
            // allowed_call_sites=hard-blocked broad launch source only; output_hash_resource_impact=none for Exact;
            // cleanup_symbol=SF_TEMP_BRIDGE_Phase6PublicOpticsPreparation; disposition=remove_or_narrow_by_owning_phase.
            bool prepare_optics_scratch(
                const WorkspaceLeaseMarker& workspace,
                void* cudaStreamOpaque,
                std::string& outError);
            bool prepare_spatial_dir_resources(
                const Spektrafilm::SpatialDirDescriptor& descriptor,
                const WorkspaceLeaseMarker& workspace,
                void* cudaStreamOpaque,
                std::string& outError);
            // SF_TEMP_BRIDGE_PreparePrintIlluminantFiltered owner=Phase4C-print-launch:
            // reason=legacy old-filter-unit launch preparation; allowed=unreachable legacy print
            // launch block only; output_impact=none in Phase4B; hash_impact=none in Phase4B;
            // resource_impact=none in Phase4B; removal=Phase4C.
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
            bool prepare_scanner_lens_blur_kernel(
                float sigma,
                void* cudaStreamOpaque,
                std::string& outError);
            bool prepare_scanner_unsharp_kernel(
                float sigma,
                void* cudaStreamOpaque,
                std::string& outError);
            bool prepare_scanner_glare_kernel(
                float sigma,
                void* cudaStreamOpaque,
                std::string& outError);
            bool prepare_grain_blur_kernel(
                float sigma,
                void* cudaStreamOpaque,
                std::string& outError);
            bool prepare_grain_blur_mid_kernel(
                float sigma,
                void* cudaStreamOpaque,
                std::string& outError);
            bool prepare_grain_blur_coarse_kernel(
                float sigma,
                void* cudaStreamOpaque,
                std::string& outError);
            bool prepare_grain_dye_kernel(
                int layer,
                int channel,
                float sigma,
                void* cudaStreamOpaque,
                std::string& outError);
            bool prepare_halation_kernel(
                int channel,
                float sigma,
                void* cudaStreamOpaque,
                std::string& outError);
            bool prepare_halation_scatter_kernel(
                int channel,
                float sigma,
                void* cudaStreamOpaque,
                std::string& outError);
            bool launch_base_pipeline_graph(
                JuicerCuda::PipelineRunParams& run,
                int renderModeKey,
                void* cudaStreamOpaque,
                int& outCudaErrorCode,
                std::string& outError);
            bool validate_density_primitives(
                const WorkingState& workingState,
                void* cudaStreamOpaque,
                std::string& outError);
            // SF_TEMP_BRIDGE_ValidatePrintPrimitives owner=Phase4C-print-launch:
            // reason=legacy broad launch validation; allowed=unreachable legacy print launch
            // block only; output_impact=none in Phase4B; hash_impact=none in Phase4B;
            // resource_impact=none in Phase4B; removal=Phase4C.
            bool validate_print_primitives(
                const WorkingState& workingState,
                const Print::Runtime& printRuntime,
                const Print::Params& printParams,
                float midgrayFactor,
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
            bool workspace_marker_matches_current_frame(const WorkspaceLeaseMarker& workspace) const noexcept;
            bool validate_workspace_lease_marker(
                const WorkspaceLeaseMarker& workspace,
                std::string& outError) const;
            bool prepare_gaussian_kernel_slot(
                JuicerCuda::Resources::DeviceGaussianKernel& kernel,
                float sigma,
                void* cudaStreamOpaque,
                std::string& outError);
            bool prepare_halation_kernel_slot(
                JuicerCuda::Resources::DeviceGaussianKernel& kernel,
                float sigma,
                void* cudaStreamOpaque,
                std::string& outError);

            std::unique_ptr<State> _state;
        };

        // SF_TEMP_BRIDGE_PrepareCudaFrameWorkingStateInput owner=Phase4C-print-launch:
        // reason=legacy broad print launch source; allowed=unreachable processImagesCUDA legacy
        // print launch block after the accepted Phase 4B hard stop only;
        // output_impact=none in Phase4B; hash_impact=legacy print keys only;
        // resource_impact=none in Phase4B; removal=Phase4C launch cutover.
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
        void release_cuda_context_resource_owners() noexcept;
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
