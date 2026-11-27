#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace JuicerCuda {

    struct BlurPass {
        float* input = nullptr;
        float* output = nullptr;
        int width = 0;
        int height = 0;
        const float* weights = nullptr;
        int radius = 0;
    };

    struct BlurLaunch {
        dim3 blocks;
        dim3 threads;
        cudaStream_t stream = nullptr;
    };

    enum class BlurPrecision : std::uint8_t {
        Float,
        Double
    };

    cudaError_t launch_horizontal_blur(BlurPass pass, BlurLaunch launch, BlurPrecision precision);
    cudaError_t blur_plane_in_place(BlurPass pass, BlurLaunch launch);

} // namespace JuicerCuda
