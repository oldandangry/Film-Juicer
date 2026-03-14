// Cuda/Print/JuicerCudaDevelopPrint.cu
// Stage-aligned CUDA TU for print development kernels.
#include <cuda_runtime.h>

#include <cstddef>

#include "Cuda/JuicerCudaPayloads.h"
#include "Cuda/JuicerCudaPrintPipeline.cuh"

__global__ void develop_print_density_kernel(
    JuicerCuda::PipelineRunParams params,
    float* ioC,
    float* ioM,
    float* ioY)
{
    if (!params.printExpose.active) {
        return;
    }

    if (!ioC || !ioM || !ioY) {
        return;
    }

    for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < params.height; y += blockDim.y * gridDim.y) {
        for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < params.width; x += blockDim.x * gridDim.x) {
            const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
            float D_cmy[3] = { ioC[idx], ioM[idx], ioY[idx] };
            apply_print_pipeline_device(params.printExpose, params.printDevelop, D_cmy);
            ioC[idx] = D_cmy[0];
            ioM[idx] = D_cmy[1];
            ioY[idx] = D_cmy[2];
        }
    }
}
