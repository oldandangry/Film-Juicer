#pragma once

#include <cuda_runtime_api.h>

#include "Cuda/JuicerCudaPayloads.h"

extern "C" cudaError_t juicer_cuda_direct_focused_capture_density_from_camera_film_linear(
    const JuicerCuda::DirectPipelineRunParams* params,
    JuicerCuda::CameraFilmLinearExposurePlanes planes,
    float* outC,
    float* outM,
    float* outY,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_print_focused_capture_density_from_camera_film_linear(
    const JuicerCuda::PrintPipelineRunParams* params,
    JuicerCuda::CameraFilmLinearExposurePlanes planes,
    float* outC,
    float* outM,
    float* outY,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_print_focused_continue_from_capture_density(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    float* dDensityC,
    float* dDensityM,
    float* dDensityY,
    const float* filmDustTransmittance,
    void* cudaStreamOpaque);
