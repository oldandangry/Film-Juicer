#pragma once

#include <cuda_runtime.h>

#include "Cuda/JuicerCudaPayloads.h"

namespace GammaValidation {

    struct CurveProbeLaunch {
        JuicerCuda::DeviceCurveView curve{};
        const float* queries = nullptr;
        const float* gammaFactors = nullptr;
        float* results = nullptr;
        int count = 0;
    };

    struct DirProbeLaunch {
        const float* logExposureBgr = nullptr;
        const float* initialDensityBgr = nullptr;
        float* correctedLogExposureBgr = nullptr;
        float* finalDensityBgr = nullptr;
        JuicerCuda::DirPayload dir{};
        JuicerCuda::DeviceCurveView densityCurvesBgr[3]{};
        float gammaFactorsBgr[3]{1.0f, 1.0f, 1.0f};
        int count = 0;
    };

    struct PrintLogExposureLaunch {
        JuicerCuda::PrintExposePayload expose{};
        const float* densityC = nullptr;
        const float* densityM = nullptr;
        const float* densityY = nullptr;
        float* logExposureR = nullptr;
        float* logExposureG = nullptr;
        float* logExposureB = nullptr;
        int count = 0;
    };

    struct PrintDevelopLogExposureLaunch {
        JuicerCuda::EnlargerPrintLinearExposurePlanes linearExposure{};
        float* logExposureR = nullptr;
        float* logExposureG = nullptr;
        float* logExposureB = nullptr;
        int width = 0;
        int height = 0;
    };

} // namespace GammaValidation

extern "C" cudaError_t gamma_validation_launch_curve_probe(
    const GammaValidation::CurveProbeLaunch* launch,
    void* cudaStreamOpaque);

extern "C" cudaError_t gamma_validation_launch_dir_probe(
    const GammaValidation::DirProbeLaunch* launch,
    void* cudaStreamOpaque);

extern "C" cudaError_t gamma_validation_launch_delay(
    unsigned long long clockCycles,
    unsigned long long* elapsedClockCycles,
    void* cudaStreamOpaque);

extern "C" cudaError_t gamma_validation_launch_print_log_exposure(
    const GammaValidation::PrintLogExposureLaunch* launch,
    void* cudaStreamOpaque);

extern "C" cudaError_t gamma_validation_launch_print_develop_log_exposure(
    const GammaValidation::PrintDevelopLogExposureLaunch* launch,
    void* cudaStreamOpaque);
