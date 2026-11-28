#pragma once

#include "Cuda/JuicerCudaFailure.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <string>
#include <limits>
#include <optional>
#include <type_traits>
#include <variant>

#include "FilmEffectsFrameDescriptors.h"
#include "juicer_cuda_api.h"

#include <cuda_runtime.h>

#include "Cuda/Diffusion/JuicerCudaDiffusion.h"
#include "Cuda/JuicerCudaDriver.h"
#include "Cuda/JuicerCudaFilmPayloads.h"
#include "Cuda/JuicerCudaResources.h"
#include "Cuda/JuicerCudaPayloads.h"
#include "Cuda/ResourceManager/JuicerCudaResourceCore.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"

#include "Cuda/JuicerCudaGrainV2Stage.h"

#include "Cuda/JuicerCudaExecutor.h"
#include "Logging.h"
#include "Hash.h"
#include "SpectralData.h"
#include "ProcessRoot.h"
#include "Scanner.h"
#include "OutputColor.h"


namespace JuicerCuda {

    struct DiffusionStageBinding {
        Spektrafilm::PlanLayout layout{};
        Spektrafilm::DiffusionStageTileGeometry geometry{};
        Spektrafilm::DiffusionFrameDomain fullFrame{};
        JuicerCuda::Diffusion::SpectrumPackageView spectra{};
        JuicerCuda::Diffusion::ExecutionWorkspaceView execution{};
        JuicerCuda::Diffusion::StagePlaneSet stagePlanes{};
        bool active = false;
    };

    struct RouteDescriptors {
        Scanner::ScannerPostEffectsDescriptor post{};
        Spektrafilm::SpatialDirDescriptor dir{};
        std::optional<Spektrafilm::LegacyGrainFrameDescriptor> grain;
        std::optional<Spektrafilm::FilmJuicerEffectsFrameDescriptor> effects;
    };

    struct ScheduleInput {
        const JuicerCuda::ExecutionFrame& frame;
        JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot;
        JuicerProcess::Root::PreparationIdentity identity;
        const RouteDescriptors& descriptors;
        JuicerCuda::FilmPayloadInput film;
        float filmRouteCorrectionScale{};
        const JuicerCuda::LegacyGrainPayloadInput& grain;
        const OutputGamutRecipe& outputGamut;
        std::uint64_t glareHash{};
        bool cameraAutoEnabled{};
        Spektrafilm::AutoExposureMethod meteringMethod{};
        const JuicerProcess::Root::PreparedFrameInput& preparation;
        Scanner::ScannerColorCorrectionDescriptor correction;
        float printExposure = 0.0f;
        float preflashExposure = 0.0f;
        const FjHostPreparation* hostPreparation = nullptr;
    };

    struct OpticalWorkspace {
        JuicerProcess::Root::PreparedCudaFrame::CaptureFilmDensityWorkspaceView density{};
        JuicerProcess::Root::PreparedCudaFrame::FocusedRgbWorkspaceView rgb{};
        JuicerProcess::Root::PreparedCudaFrame::ScannerWorkspaceView scanner{};
    };

    struct GrainBinding {
        JuicerProcess::Root::PreparedCudaFrame::LegacyGrainWorkspaceView workspace{};
        JuicerCuda::GrainPayload payload{};
        JuicerCuda::GrainKernelPayload kernels{};
        const JuicerCuda::GrainV2::FilmQuad* grainV2Bank = nullptr;
        JuicerCuda::GrainV2::FilmTables grainV2Tables{};
        JuicerCuda::GrainV2::FilmFrame grainV2Frame{};
    };

    enum class GrainModel : std::uint8_t {
        Disabled,
        Legacy,
        GrainV2
    };
    enum class CaptureRoute : std::uint8_t {
        Direct,
        Print
    };

    enum class CaptureExecution : unsigned char {
        Monolithic,
        MaterializedDensity,
        FusedDirScanner
    };
    struct MonolithicScan {
        JuicerCuda::CameraFilmLinearExposurePlanes cameraFilmLinear{};
    };
    struct FusedDirScanner {
        JuicerCuda::CameraFilmLinearExposurePlanes cameraFilmLinear{};
    };
    struct MaterializedDensity {
        JuicerProcess::Root::PreparedCudaFrame::CaptureFilmDensityWorkspaceView planes;
    };
    // Materialized density is published after development enqueue; completion remains native.
    using ScannerInput = std::variant<MonolithicScan, FusedDirScanner, MaterializedDensity>;

    enum class DirCache : unsigned char {
        Absent,
        SourceBg,
        SourceBgr,
        StagedBgr
    };
    struct DirBuildResult {
        DirCache cache;
        bool uses_source_cache() const noexcept {
            return cache == DirCache::SourceBg || cache == DirCache::SourceBgr;
        }
    };

    struct EffectsPayloads {
        JuicerCuda::FilmDefectsPayload film{};
        JuicerCuda::GateWeavePayload weave{};
    };

    inline void copy_float9(float dst[9], const float src[9]) {
        std::memcpy(dst, src, 9u * sizeof(float));
    }

    struct ExecutionErrors {
        PendingContextLossRecovery& pendingContextLossRecovery;
        bool traceInfo = false;
        const char* restrictionPrefix = nullptr;

        [[noreturn]] void fail_policy(const char* prefix, Failure failure) const;
        [[noreturn]] void fail_stage(const char* stageTag, const char* failurePrefix, cudaError_t errorCode) const;
        [[noreturn]] void fail_grain_stage(GrainFailureSite site, cudaError_t status) const;
        [[noreturn]] void fail_submission(const char* stage, const char* prefix, const Failure& failure, bool deferredDirError = false) const;
        [[noreturn]] void fail_route(const char* diagnostic) const;
    };

    const char* nonempty_cstr_or(const char* value, const char* fallback);

    const char* detail_or_unknown(const char* detail);

    void check_abort(FjAbortCallback callback);

    void finish_frame(
        CaptureRoute route,
        JuicerProcess::Root::PreparedCudaFrame& prepared,
        const JuicerProcess::Root::PreparationIdentity& identity,
        const ScatterHalationPreparedView& halation,
        const ResourceManager::SubmissionSnapshot& snapshot,
        void* stream,
        FjAbortCallback callback,
        const ExecutionErrors& errors);

    void check_diffusion_launch(
        Diffusion::LaunchResult result, const char* prefix, const ExecutionErrors& errors);

    void clear_spatial_dir_final_develop_payload_bindings(
        JuicerCuda::FilmDevelopPayload& payload);

    GrainModel grain_model(const ScheduleInput& input);

    void meter_camera_exposure(
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        JuicerCuda::FilmPayloadPack& filmPayloads,
        const JuicerCuda::ExecutionFrame& frame,
        Spektrafilm::AutoExposureMethod cameraMeteringMethod,
        const char* missingBuffersDiagnostic,
        const ExecutionErrors& errors);

    void bind_diffusion(
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        const JuicerCuda::ExecutionFrame& frame,
        DiffusionStageBinding& cameraDiffusion,
        DiffusionStageBinding* enlargerDiffusion,
        const ExecutionErrors& errors);

    void check_highlight_reconstruction(
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        const JuicerProcess::Root::PreparationIdentity& identity,
        const ExecutionErrors& errors);

    JuicerCuda::CameraFilmLinearExposurePlanes expose_camera(
        CaptureRoute route,
        JuicerCuda::FilmScanRunParams& run,
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        const JuicerProcess::Root::PreparationIdentity& identity,
        const JuicerCuda::ExecutionFrame& frame,
        const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        bool cameraDiffusionActive,
        DiffusionStageBinding& cameraDiffusion,
        JuicerCuda::ScatterHalationPreparedView& halation,
        const ExecutionErrors& errors);

    EffectsPayloads pack_effects_payloads(
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        const std::optional<Spektrafilm::FilmJuicerEffectsFrameDescriptor>& descriptor,
        const ExecutionErrors& errors);

    OpticalWorkspace bind_optical_workspace(
        CaptureRoute route,
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& focusedWorkspace,
        void* stream,
        const std::optional<Spektrafilm::FilmJuicerEffectsFrameDescriptor>& descriptor,
        const ExecutionErrors& errors);

    void bind_grain_v2(JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
                       const FjFilmGrain& input,
                       const JuicerCuda::FrameRect& window,
                       CaptureRoute route,
                       GrainBinding& grain,
                       const ExecutionErrors& errors);

    void bind_legacy_grain(
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& focusedWorkspace,
        const JuicerCuda::LegacyGrainPayloadInput& input,
        GrainBinding& grain,
        const ExecutionErrors& errors);

    DirBuildResult build_spatial_dir(
        CaptureRoute route,
        JuicerCuda::FilmScanRunParams& run,
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& focusedWorkspace,
        const Spektrafilm::SpatialDirDescriptor& descriptor,
        const JuicerCuda::CameraFilmLinearExposurePlanes& cameraFilmLinear,
        bool cameraFilmLinearActive,
        void* stream,
        CaptureExecution execution,
        const ExecutionErrors& errors);

    void develop_capture(
        CaptureRoute route,
        JuicerCuda::FilmScanRunParams& run,
        const JuicerCuda::CameraFilmLinearExposurePlanes& cameraFilmLinear,
        bool cameraFilmLinearActive,
        const Spektrafilm::SpatialDirDescriptor& descriptor,
        const OpticalWorkspace& optical,
        GrainModel grainModel,
        const GrainBinding& grain,
        bool filmEffectsActive,
        const EffectsPayloads& effects,
        bool grainDebugActive,
        void* stream,
        const ExecutionErrors& errors);

    void build_gate_mask(CaptureRoute route, const EffectsPayloads& effects, const OpticalWorkspace& optical, void* stream, const ExecutionErrors& errors);

    void bind_run_images(FilmScanRunParams& run, const ExecutionFrame& frame);

    void bind_run_film(FilmScanRunParams& run, const FilmPayloadPack& film);

    void annotate_halation_completion(
        Failure& failure,
        const JuicerProcess::Root::PreparationIdentity& identity,
        const ScatterHalationPreparedView& halation,
        const ResourceManager::SubmissionSnapshot& snapshot,
        std::uint64_t contextEpoch,
        const char* boundary) noexcept;

    void pack_scan_stage(
        JuicerCuda::ScanStagePayload& scanStage,
        const JuicerProcess::Root::PreparedCudaFrame::FocusedPreparedView& prepared,
        const OutputGamutRecipe& outputGamut,
        const Scanner::ScannerColorCorrectionDescriptor& scannerCorrection,
        const ExecutionErrors& errors);

    void scan_direct_output(
        JuicerCuda::FilmScanRunParams& run,
        const JuicerCuda::ScannerSpectra& spectra,
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& focusedWorkspace,
        const RouteDescriptors& descriptors,
        const OpticalWorkspace& optical,
        const EffectsPayloads& effects,
        const ScannerInput& scannerInput,
        bool grainDebugActive,
        const std::optional<DirBuildResult>& dirBuild,
        bool gateOutputActive,
        const JuicerCuda::ExecutionFrame& frame,
        const ExecutionErrors& errors);

    void scan_print_output(
        JuicerCuda::PrintPipelineRunParams& run,
        const JuicerCuda::ScannerSpectra& spectra,
        JuicerProcess::Root::PreparedCudaFrame& preparedFrame,
        const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& focusedWorkspace,
        const RouteDescriptors& descriptors,
        const OpticalWorkspace& optical,
        const EffectsPayloads& effects,
        const ScannerInput& scannerInput,
        bool grainDebugActive,
        const std::optional<DirBuildResult>& dirBuild,
        bool gateOutputActive,
        const JuicerCuda::ExecutionFrame& frame,
        std::uint64_t glareHash,
        const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        const ExecutionErrors& errors);

    void bind_scan_error_readback(
        CaptureRoute route,
        JuicerProcess::Root::PreparedCudaFrame& frame,
        ScanStagePayload& scan,
        void* stream,
        const ExecutionErrors& errors);

    void finish_scan_error_readback(
        CaptureRoute route,
        JuicerProcess::Root::PreparedCudaFrame& frame,
        ScanStagePayload& scan,
        const JuicerProcess::Root::PreparationIdentity& identity,
        const ScatterHalationPreparedView& halation,
        const ResourceManager::SubmissionSnapshot& snapshot,
        void* stream,
        const ExecutionErrors& errors);

    void execute_direct_schedule(JuicerProcess::Root& root, const ScheduleInput& input, PendingContextLossRecovery& recovery, FjAbortCallback abortCallback);

    void execute_print_schedule(JuicerProcess::Root& root, const ScheduleInput& input, PendingContextLossRecovery& recovery, FjAbortCallback abortCallback);

} // namespace JuicerCuda
