#pragma once

#include "Cuda/JuicerCudaBlur.h"

namespace JuicerCuda {
    struct FocusedScannerPostEffectOptions {
        int glareOriginX = 0;
        int glareOriginY = 0;
        std::uint64_t glareSeed = 0;
        float glarePercent = 0.0f;
        float glareRoughness = 0.0f;
        const float* glareKernel = nullptr;
        int glareRadius = 0;
    };

    cudaError_t launch_print_glare(
        const FocusedScannerPostEffectOptions& options,
        float* temporary,
        float* blurScratch,
        int width,
        int height,
        dim3 blocks,
        dim3 threads,
        cudaStream_t stream,
        const float*& field);
} // namespace JuicerCuda
