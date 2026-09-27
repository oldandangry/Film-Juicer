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

    // Internal exception mapped by the caller after prepared-frame unwinding.
    struct ExecutionFailure {
        Failure failure;
        bool deferredScanError = false;
    };

    struct PendingContextLossRecovery {
        bool pending = false;
        Failure failure;
        const char* stage = nullptr;
    };


    AutoExposurePreviewDescriptor make_auto_exposure_preview_descriptor(
        const FrameRect& sourceBounds,
        const FrameRect& meterBounds,
        Spektrafilm::AutoExposureMethod method);

    void execute_direct(
        const DirectExecutionInput& input,
        PendingContextLossRecovery& recovery,
        FjAbortCallback abortCallback = {});
    void execute_print(
        const PrintExecutionInput& input,
        PendingContextLossRecovery& recovery,
        FjAbortCallback abortCallback = {});

    PreparedDescriptors describe_execution(
        const RenderRecipe& recipe,
        const FocusedRenderPayload& payload,
        const ExecutionFrame& frame);

    void execute_prepared(
        const PreparedExecutionInput& input,
        PendingContextLossRecovery& recovery,
        FjAbortCallback abortCallback = {});

#if defined(JUICER_CUDA_RENDER_TEST_HOOK)
    namespace RenderTest {
        void before_abort_query();
    }
#endif

#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
    namespace ExecutorTest {
        bool inject_scan_error(Failure& failure);
        bool inject_grain_upload_failure(const char* label, Failure& failure);
        bool inject_defect_event_create(cudaError_t& error) noexcept;
        bool inject_defect_event_record(cudaError_t& error) noexcept;
        void observe_classification(const char* stage, const Failure& failure, bool recoveryPending) noexcept;
        void observe_frame_abort() noexcept;
        void observe_recovery_start(bool pending) noexcept;
        void observe_recovery_end() noexcept;
    } // namespace ExecutorTest
#endif

} // namespace JuicerCuda
