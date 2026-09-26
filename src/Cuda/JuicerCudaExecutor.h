#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "Cuda/JuicerCudaResources.h"
#include "Cuda/ResourceManager/JuicerCudaResourceCore.h"
#include "RenderRecipe.h"
#include "FilmEffectsFrameDescriptors.h"
#include "ProcessRoot.h"
#include "juicer_cuda_descriptors.h"

struct FocusedRenderPayload;

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

    struct DirectExecutionInput {
        const RenderRecipe& recipe;
        const FocusedRenderPayload& payload;
        const ExecutionFrame& frame;
        ResourceManager::SubmissionSnapshot& snapshot;
    };

    struct PrintExecutionInput {
        const RenderRecipe& recipe;
        const FocusedRenderPayload& payload;
        const ExecutionFrame& frame;
        ResourceManager::SubmissionSnapshot& snapshot;
    };

    struct PreparedExecutionInput {
        const JuicerProcess::Root::PreparedFrameInput& preparation;
        const PreparedDescriptors& descriptors;
        FilmPayloadInput film;
        float filmRouteCorrectionScale = 1.0f;
        PrintExposureRecipe printExposure;
        std::uint64_t recipeHash = 0;
        bool cameraAutoEnabled = false;
        const ExecutionFrame& frame;
        ResourceManager::SubmissionSnapshot& snapshot;
    };

    // FJ_TEMP_BRIDGE: diagnostic failure/message handoff; remove S2.D.
    // Borrowed only during the route call.
    // The adapter catches delivery exceptions. Neither member may be retained.
    struct DirFailureMessage {
        void (*deliver)(void* user, const std::string& diagnostic) noexcept;
        void* user;
    };

    // Internal exception mapped by the caller after prepared-frame unwinding.
    struct ExecutionFailure {};

    // Diagnostic-based recovery is retained only until the S2.D policy cutover.
    struct PendingContextLossRecovery {
        bool pending = false;
        cudaError_t error = cudaSuccess;
        const char* stage = nullptr;
        std::string detail;
    };

    ResourceManager::DeviceContextKey inspect_frame(
        const unsigned char* srcBase,
        unsigned char* dstBase,
        bool traceInfo);

    bool is_cuda_context_loss_signal(cudaError_t error, const std::string& detail);

    AutoExposurePreviewDescriptor make_auto_exposure_preview_descriptor(
        const FrameRect& sourceBounds,
        const FrameRect& meterBounds,
        Spektrafilm::AutoExposureMethod method);

    void execute_direct(
        const DirectExecutionInput& input,
        PendingContextLossRecovery& recovery,
        const DirFailureMessage& dirFailureMessage);
    void execute_print(
        const PrintExecutionInput& input,
        PendingContextLossRecovery& recovery,
        const DirFailureMessage& dirFailureMessage);

    PreparedDescriptors describe_execution(
        const RenderRecipe& recipe,
        const FocusedRenderPayload& payload,
        const ExecutionFrame& frame);

    void execute_prepared(
        const PreparedExecutionInput& input,
        PendingContextLossRecovery& recovery,
        const DirFailureMessage& dirFailureMessage);

#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
    namespace ExecutorTest {
        bool inject_scan_error(std::string& diagnostic);
        void observe_classification(const char* stage, bool recoveryPending) noexcept;
        void observe_frame_abort() noexcept;
        void observe_recovery_start(bool pending) noexcept;
        void observe_recovery_end() noexcept;
    } // namespace ExecutorTest
#endif

} // namespace JuicerCuda
