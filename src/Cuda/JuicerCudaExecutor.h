#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

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
        void observe(const char* stageTag, const Failure& observed) noexcept;

        bool submissionAdmitted = false;
        const char* lossStage = nullptr;
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


} // namespace JuicerCuda
