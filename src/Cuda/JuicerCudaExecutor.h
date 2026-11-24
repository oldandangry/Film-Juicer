#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#if defined(JUICER_CUDA_RENDER_TEST_HOOK)
#include <vector>
#endif

#include "Cuda/JuicerCudaResources.h"
#include "Cuda/ResourceManager/JuicerCudaResourceCore.h"
#include "FilmEffectsFrameDescriptors.h"
#include "ProcessRoot.h"
#include "juicer_cuda_descriptors.h"

struct FjHostPreparation;

namespace JuicerCuda {

    struct FrameRect {
        int x1 = 0;
        int y1 = 0;
        int x2 = 0;
        int y2 = 0;
    };

    // All addresses and descriptor references are borrowed for this invocation.
    struct ExecutionFrame {
        FrameRect sourceBounds;
        FrameRect renderWindow;
        FrameRect fullFrameExtent;
        const unsigned char* sourceBase = nullptr;
        const unsigned char* source = nullptr;
        unsigned char* destination = nullptr;
        std::ptrdiff_t sourceRowBytes = 0;
        std::ptrdiff_t destinationRowBytes = 0;
        int components = 0;
        void* stream = nullptr;
        const std::optional<Spektrafilm::DiffusionFrameSetDescriptor>& diffusionFrameSet;
        const std::optional<ScatterHalationFrameDescriptor>& scatterHalation;
        const Spektrafilm::FilmJuicerEffectsGeometry& effectsGeometry;
        float pixelSizeUm = 0;
        double timeFrames = 0;
        double frameRate = 0;
        std::uint64_t sessionSeed = 1;
        std::uintptr_t clipToken = 0;
        const AutoExposurePreviewDescriptor& autoExposureDescriptor;
        bool traceInfo = false;
        bool traceVerbose = false;
    };

    struct PreparedExecutionInput {
        const JuicerProcess::Root::PreparedFrameInput& preparation;
        const PreparedDescriptors& descriptors;
        FilmPayloadInput film;
        float filmRouteCorrectionScale = 1.0f;
        float printExposure = 0.0f;
        float preflashExposure = 0.0f;
        std::uint64_t glareHash = 0;
        bool cameraAutoEnabled = false;
        const ExecutionFrame& frame;
        ResourceManager::SubmissionSnapshot& snapshot;
        const FjHostPreparation* hostPreparation = nullptr;
    };

    // Internal exception mapped by the caller after prepared-frame unwinding.
    struct ExecutionFailure {
        Failure failure;
        bool deferredDirError = false;
    };

    struct PendingContextLossRecovery {
        bool pending = false;
        Failure failure;
        const char* stage = nullptr;
    };


    bool pack_scan_color(
        ScanColorPayload& destination,
        const JuicerProcess::Root::PreparedCudaFrame::FocusedPreparedView& prepared,
        const OutputGamutRecipe& recipe);

    AutoExposurePreviewDescriptor make_auto_exposure_preview_descriptor(
        const FrameRect& sourceBounds,
        const FrameRect& meterBounds,
        Spektrafilm::AutoExposureMethod method);

    void execute_prepared(
        JuicerProcess::Root& root,
        const PreparedExecutionInput& input,
        PendingContextLossRecovery& recovery,
        FjAbortCallback abortCallback = {});

#if defined(JUICER_CUDA_RENDER_TEST_HOOK)
    namespace RenderTest {
        bool omit_film_bank() noexcept;
        bool omit_grain_plane() noexcept;
        void before_abort_query();
        void observe_film_proxy_reservation() noexcept;
        struct PrintLaunchObservation {
            struct PaperLane {
                std::uintptr_t address = 0;
                std::vector<std::uint32_t> words;
            };
            bool valid = false;
            std::uintptr_t filmDensityC = 0;
            std::uintptr_t filmDensityM = 0;
            std::uintptr_t filmDensityY = 0;
            std::uintptr_t filmBase = 0;
            std::uintptr_t mainIlluminant = 0;
            std::uintptr_t preflashIlluminant = 0;
            std::uint64_t filmDensityHash = 0;
            std::uint64_t profileTablesHash = 0;
            std::uint64_t mainIlluminantHash = 0;
            std::uint64_t preflashIlluminantHash = 0;
            int spectralSampleCount = 0;
            float printExposure = 1.0f;
            float printPreflashExposure = 0.0f;
            float printMidgrayFactor = 1.0f;
            float printCyanTap = 0.0f;
            PinnedUploadTest::UploadObservation uploads{};
            float printPreflashRaw[3] = {0.0f, 0.0f, 0.0f};
            std::array<float, 81> mainLight{};
            std::array<float, 81> preflashLight{};
            std::array<float, 81> filmBaseWords{};
            std::array<float, 243> filmDyes{};
            std::array<PaperLane, 7> paperTables;
            std::uint64_t glareSeed = 0;
            unsigned glareLaunchCount = 0;
            std::vector<float> glareField;
        };
        void set_print_launch_observer(PrintLaunchObservation* observation) noexcept;
        void observe_print_launch(const PrintPreparedView& prepared, const PrintExposePayload& expose);
        void complete_print_launch_observation(PrintLaunchObservation& out, std::uintptr_t stream);
    } // namespace RenderTest
#endif

#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
    namespace ExecutorTest {
        enum class CarrierStage : std::uint8_t {
            Exposure,
            Boost,
            CameraDiffusion,
            Spatial,
            DirLog,
            Count
        };
        struct CarrierObservation {
            CarrierStage stage = CarrierStage::Exposure;
            CameraFilmLinearExposurePlanes planes{};
            int width = 0;
            int height = 0;
            void* stream = nullptr;
            float routeCorrection = 1.0f;
        };
        void observe_carrier(CarrierObservation observation);
        enum class WorkspaceHandoff : unsigned char {
            Monolithic,
            MaterializedDensity,
            FusedDirScanner
        };
        void observe_workspace_handoff(WorkspaceHandoff handoff) noexcept;
        enum class WorkspaceRejection : unsigned char {
            Capacity,
            InvalidValue,
            ContextLoss
        };
        struct WorkspaceCandidateObservation {
            std::optional<ResourceManager::WorkspaceLayout> layout;
            unsigned backendCalls = 0;
            FjStatus result{};
            bool succeeded = false;
        };
        struct WorkspaceTransitionObservation {
            std::optional<ResourceManager::WorkspaceLayout> requested;
            std::optional<ResourceManager::WorkspaceLayout> admitted;
            std::optional<ResourceManager::WorkspaceLayout> finalDevelopment;
            std::optional<ResourceManager::WorkspaceLayout> postFrame;
            std::array<WorkspaceCandidateObservation, 6> candidates;
            unsigned candidateCount = 0;
            unsigned resourcesAcquired = 0;
            unsigned finishes = 0;
            unsigned aborts = 0;
            unsigned sourceCacheChannels = 0;
            unsigned finalCacheChannels = 0;
            bool bindingsObserved = false;
            bool scannerMapping = false;
            bool grainMapping = false;
            bool captureEnqueued = false;
            bool scannerEnqueued = false;
            FjStatus classified{};
            bool recoveryPending = false;
        };
        void reset_workspace_transitions() noexcept;
        WorkspaceTransitionObservation workspace_transitions() noexcept;
        bool arm_workspace_rejection(unsigned count, WorkspaceRejection kind) noexcept;
        void begin_workspace_candidate(unsigned index, const ResourceManager::WorkspaceLayout& layout) noexcept;
        void end_workspace_candidate(bool success, const Failure& failure) noexcept;
        bool inject_workspace_candidate_failure(Failure& failure);
        void observe_workspace_requested(const std::optional<ResourceManager::WorkspaceLayout>& layout) noexcept;
        void observe_workspace_resource_acquired() noexcept;
        void observe_workspace_admitted(const ResourceManager::WorkspaceLayout& layout) noexcept;
        void observe_workspace_final(const ResourceManager::WorkspaceLayout& layout) noexcept;
        void observe_workspace_post_frame(const std::optional<ResourceManager::WorkspaceLayout>& layout) noexcept;
        void observe_workspace_finish() noexcept;
        void observe_workspace_bindings(JuicerProcess::Root::PreparedCudaFrame& frame,
                                        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& marker) noexcept;
        struct WorkspaceCacheChannels {
            unsigned source = 0;
            unsigned final = 0;
        };
        void observe_workspace_dir_build(WorkspaceCacheChannels channels) noexcept;
        void observe_workspace_capture() noexcept;
        void observe_workspace_scanner() noexcept;
        struct EffectsFrameObservation {
            Spektrafilm::ScanRoute route = Spektrafilm::kDefaultScanRoute;
            unsigned count = 0;
            Spektrafilm::FilmJuicerEffectsFrameExtent renderExtent;
            Spektrafilm::FilmJuicerEffectsFrameExtent fullFrameExtent;
            Spektrafilm::FilmJuicerEffectsGeometry geometry;
            float filmFormatLongEdgeMm = 0.0f;
            float pixelSizeUm = 0.0f;
            double frameTime = 0.0;
            double frameRate = 0.0;
            std::uint64_t sessionSeed = 0;
            std::uint64_t clipToken = 0;
            bool success = false;
            std::optional<Spektrafilm::FilmJuicerEffectsFrameDescriptor> descriptor;
        };
        void reset_effects_frame_observation() noexcept;
        EffectsFrameObservation effects_frame_observation() noexcept;
        void observe_effects_frame(EffectsFrameObservation observation) noexcept;
        struct GrainFrameObservation {
            Spektrafilm::ScanRoute route = Spektrafilm::kDefaultScanRoute;
            unsigned count = 0;
            Spektrafilm::VisualGrainFrameExtent renderExtent;
            Spektrafilm::VisualGrainFrameExtent fullFrameExtent;
            float pixelSizeUm = 0.0f;
            double frameTime = 0.0;
            double frameRate = 0.0;
            std::uint64_t sessionSeed = 0;
            std::uint64_t clipToken = 0;
            bool success = false;
            std::optional<Spektrafilm::VisualGrainFrameDescriptor> descriptor;
        };
        void reset_grain_frame_observation() noexcept;
        GrainFrameObservation grain_frame_observation() noexcept;
        void observe_grain_frame(GrainFrameObservation observation) noexcept;
        struct DirFrameObservation {
            Spektrafilm::ScanRoute route = Spektrafilm::kDefaultScanRoute;
            unsigned count = 0;
            std::uint32_t pixelSizeBits = 0;
            Spektrafilm::DirFrameExtent renderExtent;
            Spektrafilm::DirFrameExtent fullFrameExtent;
            bool success = false;
            Spektrafilm::SpatialDirDescriptor descriptor;
        };
        void reset_dir_frame_observation() noexcept;
        DirFrameObservation dir_frame_observation() noexcept;
        void observe_dir_frame(Spektrafilm::ScanRoute route, float pixelSizeUm, Spektrafilm::DirFrameExtent renderExtent, Spektrafilm::DirFrameExtent fullFrameExtent, bool success, const Spektrafilm::SpatialDirDescriptor& descriptor) noexcept;
        bool inject_scan_error(Failure& failure, JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult& result);
        void observe_focused_resource_preparation() noexcept;
        bool inject_grain_upload_failure(const char* label, Failure& failure);
        bool inject_defect_event_create(cudaError_t& error) noexcept;
        bool inject_defect_event_record(cudaError_t& error) noexcept;
        void observe_classification(const char* stage, const Failure& failure, bool recoveryPending) noexcept;
        void observe_frame_abort() noexcept;
        void observe_recovery_start(bool pending) noexcept;
        void observe_recovery_end() noexcept;
        void observe_context_retirement(bool accepted, FjStatus invalidation, bool cleared) noexcept;
        void observe_adapter_completion(FjRenderOutcome, bool fatal, std::string_view diagnostic) noexcept;
    } // namespace ExecutorTest
#endif

} // namespace JuicerCuda
