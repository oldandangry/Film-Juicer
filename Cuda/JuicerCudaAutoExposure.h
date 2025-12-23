// Cuda/JuicerCudaAutoExposure.h
//
// CUDA camera auto-exposure metering.
//
// This header intentionally avoids CUDA types so it can be included from non-NVCC translation units.
//
#pragma once

#include <cstddef>

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
