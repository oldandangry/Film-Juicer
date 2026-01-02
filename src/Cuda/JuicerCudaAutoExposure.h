// Cuda/JuicerCudaAutoExposure.h
//
// CUDA camera auto-exposure metering.
//
// This header intentionally avoids CUDA types so it can be included from non-NVCC translation units.
//
#pragma once

#include <cstddef>

struct JuicerCudaAutoExposurePartial {
    double sumY;
    double sumW;
};

// Device-side scratch buffers for metering. All pointers are CUDA device pointers.
struct JuicerCudaAutoExposureScratch {
    JuicerCudaAutoExposurePartial* partialsA = nullptr;
    JuicerCudaAutoExposurePartial* partialsB = nullptr;
    int partialCapacity = 0;
    unsigned int* maxYBits = nullptr;
    unsigned int* histogram = nullptr; // 2048 bins
    float* weightsX = nullptr; // meterWidth floats
    float* weightsY = nullptr; // meterHeight floats
};

// Device-side outputs for metering. All pointers are CUDA device pointers.
struct JuicerCudaAutoExposureDeviceState {
    float* exposureScale = nullptr; // float scalar
    double* autoEV = nullptr;       // double scalar
    int* valid = nullptr;           // int scalar (0/1)
};

// Returns 0 on success, non-zero on failure; on failure outErrorMsg points to a stable message.
extern "C" int juicer_cuda_measure_center_weighted_Y(
    const void* srcDeviceBase,
    std::size_t srcRowBytes,
    int srcBoundsX1,
    int srcBoundsY1,
    int srcBoundsX2,
    int srcBoundsY2,
    int meterX1,
    int meterY1,
    int meterX2,
    int meterY2,
    int nComponents,
    int inputColorSpaceIndex,
    int applyCctfDecoding,
    const float* rgbToXYZ9,
    double* outY,
    void* cudaStreamOpaque,
    const char** outErrorMsg);

extern "C" int juicer_cuda_measure_median_Y(
    const void* srcDeviceBase,
    std::size_t srcRowBytes,
    int srcBoundsX1,
    int srcBoundsY1,
    int srcBoundsX2,
    int srcBoundsY2,
    int meterX1,
    int meterY1,
    int meterX2,
    int meterY2,
    int nComponents,
    int inputColorSpaceIndex,
    int applyCctfDecoding,
    const float* rgbToXYZ9,
    double* outY,
    void* cudaStreamOpaque,
    const char** outErrorMsg);

// Enqueues auto-exposure metering and writes autoEV/exposureScale to device outputs. This function
// does not synchronize; it only enqueues work on the given stream.
// Returns 0 on success, non-zero on failure; on failure outErrorMsg points to a stable message.
extern "C" int juicer_cuda_auto_exposure_meter_to_device(
    const void* srcDeviceBase,
    std::size_t srcRowBytes,
    int srcBoundsX1,
    int srcBoundsY1,
    int srcBoundsX2,
    int srcBoundsY2,
    int meterX1,
    int meterY1,
    int meterX2,
    int meterY2,
    int nComponents,
    int inputColorSpaceIndex,
    int applyCctfDecoding,
    const float* rgbToXYZ9,
    int meteringMethod,
    double sliderEV,
    JuicerCudaAutoExposureScratch scratch,
    JuicerCudaAutoExposureDeviceState outState,
    void* cudaStreamOpaque,
    const char** outErrorMsg);

// Updates exposureScale from an existing autoEV (no re-meter). This function does not synchronize.
extern "C" int juicer_cuda_auto_exposure_update_scale_to_device(
    double sliderEV,
    JuicerCudaAutoExposureDeviceState state,
    void* cudaStreamOpaque,
    const char** outErrorMsg);

// Builds separable center-weighted metering weights (wX/wY) on the GPU. This function does not
// synchronize; it only enqueues work on the given stream.
extern "C" int juicer_cuda_auto_exposure_build_center_weight_tables(
    int meterWidth,
    int meterHeight,
    float* weightsX,
    float* weightsY,
    void* cudaStreamOpaque,
    const char** outErrorMsg);
