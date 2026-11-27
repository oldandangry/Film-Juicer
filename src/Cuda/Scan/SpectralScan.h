#pragma once

#include <cuda_runtime_api.h>

#include "Cuda/JuicerCudaPayloads.h"

extern "C" cudaError_t juicer_cuda_negative_direct_pipeline(
    const JuicerCuda::FilmScanRunParams* hParams,
    const JuicerCuda::ScannerSpectra& spectra,
    void* cudaStreamOpaque,
    JuicerCuda::CameraFilmLinearExposurePlanes carrier);

extern "C" cudaError_t juicer_cuda_print_focused_pipeline(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    const JuicerCuda::ScannerSpectra& spectra,
    void* cudaStreamOpaque,
    JuicerCuda::CameraFilmLinearExposurePlanes carrier);

extern "C" cudaError_t juicer_cuda_direct_focused_scan_linear_rgb(
    const JuicerCuda::FilmScanRunParams* hParams,
    const JuicerCuda::ScannerSpectra& spectra,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    void* cudaStreamOpaque,
    JuicerCuda::CameraFilmLinearExposurePlanes carrier);

extern "C" cudaError_t juicer_cuda_print_focused_scan_linear_rgb(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    const JuicerCuda::ScannerSpectra& spectra,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    float* dScratchBlurred,
    int glareOriginX,
    int glareOriginY,
    std::uint64_t glareSeed,
    float glarePercent,
    float glareRoughness,
    const float* dGlareKernel,
    int glareRadius,
    void* cudaStreamOpaque,
    JuicerCuda::CameraFilmLinearExposurePlanes carrier);

extern "C" cudaError_t juicer_cuda_direct_focused_scan_linear_density_rgb(
    const JuicerCuda::FilmScanRunParams* hParams,
    const JuicerCuda::ScannerSpectra& spectra,
    const float* dDensityC,
    const float* dDensityM,
    const float* dDensityY,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    const float* filmDustTransmittance,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_print_focused_scan_linear_density_rgb(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    const JuicerCuda::ScannerSpectra& spectra,
    const float* dDensityC,
    const float* dDensityM,
    const float* dDensityY,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    float* dScratchBlurred,
    int glareOriginX,
    int glareOriginY,
    std::uint64_t glareSeed,
    float glarePercent,
    float glareRoughness,
    const float* dGlareKernel,
    int glareRadius,
    void* cudaStreamOpaque);
