#pragma once

#include <cuda_runtime_api.h>

#include "Cuda/JuicerCudaPayloads.h"

extern "C" cudaError_t juicer_cuda_capture_density(
    const JuicerCuda::FilmScanRunParams* hParams,
    float* dDensityC,
    float* dDensityM,
    float* dDensityY,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_camera_film_linear_exposure(
    const JuicerCuda::FilmScanRunParams* hParams,
    JuicerCuda::CameraFilmLinearExposurePlanes planes,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_apply_legacy_grain(
    const JuicerCuda::GrainPayload* grain,
    const JuicerCuda::GrainKernelPayload* kernels,
    int width,
    int height,
    float* densityC,
    float* densityM,
    float* densityY,
    float* filterTemp,
    float* scaleWork,
    float* deltaAccum,
    float* layerWork,
    float* sharedDelta,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_build_spatial_dir(
    const JuicerCuda::FilmScanRunParams* hParams,
    JuicerCuda::SpatialDirBuildRequest request);

extern "C" cudaError_t juicer_cuda_build_spatial_dir_cached_log_raw(
    const JuicerCuda::FilmScanRunParams* hParams,
    float* logRawB,
    float* logRawG,
    float* logRawR,
    void* cudaStreamOpaque);

extern "C" cudaError_t
juicer_cuda_build_spatial_dir_cached_log_raw_from_camera_film_linear(
    const JuicerCuda::FilmScanRunParams* hParams,
    JuicerCuda::CameraFilmLinearExposurePlanes cameraFilmLinear,
    float* logRawB,
    float* logRawG,
    float* logRawR,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_capture_density_from_camera_film_linear(
    const JuicerCuda::FilmScanRunParams* params,
    JuicerCuda::CameraFilmLinearExposurePlanes planes,
    float* outC,
    float* outM,
    float* outY,
    void* cudaStreamOpaque);
