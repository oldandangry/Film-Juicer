#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "Cuda/JuicerCudaPayloads.h"

namespace JuicerCuda {

    struct HighlightBoostDescriptor {
        std::uint64_t recipeHash = 0;
        double protection = 0.0;
        double sharpness = 0.0;
        double peakGain = 0.0;
    };

    enum class HighlightBoostDisposition : std::uint8_t {
        Bypass,
        Applied,
        InvalidCarrier,
        PeakOverflow
    };

    inline constexpr int kHighlightBoostMaxBlocks = 4096;

    struct HighlightBoostWorkspace {
        struct Partial {
            float maximum;
            std::uint32_t invalid;
        };
        Partial partials[kHighlightBoostMaxBlocks];
        HighlightBoostDisposition disposition;
        float maximum;
        float protection;
        float protectionResidual;
        float onset;
        float peak;
        float peakResidual;
        float range;
        float rangeResidual;
        float shape;
        float shapeResidual;
        float tailExpFactor;
        float tailLinearFactor;
        float delta;
        float inverseShapeNormalization;
    };

    enum class HighlightBoostLaunchStage : std::uint8_t {
        None,
        Binding,
        PartialReduction,
        FinalReduction,
        Apply
    };

    struct HighlightBoostLaunchResult {
        cudaError_t status = cudaSuccess;
        HighlightBoostLaunchStage stage = HighlightBoostLaunchStage::None;
    };

#if defined(JUICER_CONTEXT_DRAIN_TEST_HOOK)
    namespace TestSupport {
        enum class HighlightBoostFault : std::uint8_t {
            None,
            PartialReduction,
            FinalReduction,
            Apply,
            Readback,
            StatusEvent,
            RetirementEvent
        };
        void fail_next_highlight_boost(HighlightBoostFault fault) noexcept;
        cudaError_t consume_highlight_boost_fault(HighlightBoostFault fault) noexcept;
    } // namespace TestSupport
#endif

    HighlightBoostLaunchResult launch_highlight_boost(
        const HighlightBoostDescriptor& descriptor,
        CameraFilmLinearExposurePlanes carrier,
        int width,
        int height,
        HighlightBoostWorkspace* workspace,
        cudaStream_t stream,
        int blockLimit = kHighlightBoostMaxBlocks);

} // namespace JuicerCuda
