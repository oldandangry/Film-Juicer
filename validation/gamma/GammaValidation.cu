#include "GammaValidationCuda.h"

#include <limits>

#include <cuda_runtime.h>

#include "Cuda/JuicerCudaDeviceHelpers.cuh"

namespace {

    __global__ void curve_probe_kernel(GammaValidation::CurveProbeLaunch launch) {
        const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        if (index >= launch.count) {
            return;
        }
        launch.results[index] = sample_density_at_logE_device(
            launch.curve,
            launch.queries[index],
            launch.gammaFactors[index]);
    }

    __global__ void dir_probe_kernel(GammaValidation::DirProbeLaunch launch) {
        const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        if (index >= launch.count) {
            return;
        }
        const int offset = index * 3;
        float corrected[3] = {
            launch.logExposureBgr[offset],
            launch.logExposureBgr[offset + 1],
            launch.logExposureBgr[offset + 2]};
        const float density[3] = {
            launch.initialDensityBgr[offset],
            launch.initialDensityBgr[offset + 1],
            launch.initialDensityBgr[offset + 2]};
        apply_dir_runtime_logE_device(
            corrected,
            density,
            launch.dir);
        for (int channel = 0; channel < 3; ++channel) {
            launch.correctedLogExposureBgr[offset + channel] = corrected[channel];
            launch.finalDensityBgr[offset + channel] =
                sample_density_at_logE_device(
                    launch.densityCurvesBgr[channel],
                    corrected[channel],
                    launch.gammaFactorsBgr[channel]);
        }
    }

    __global__ void delay_kernel(
        unsigned long long clockCycles,
        unsigned long long* elapsedClockCycles) {
        const unsigned long long begin = clock64();
        unsigned long long elapsed = 0;
        while (elapsed < clockCycles) {
            elapsed = clock64() - begin;
        }
        *elapsedClockCycles = elapsed;
    }

    __global__ void print_log_exposure_kernel(
        GammaValidation::PrintLogExposureLaunch launch) {
        const int index =
            static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        if (index >= launch.count) {
            return;
        }
        const float filmDensityCmy[3] = {
            launch.densityC[index],
            launch.densityM[index],
            launch.densityY[index]};
        float rawPrint[3] = {0.0f, 0.0f, 0.0f};
        if (!print_spectral_integrate_device(
                launch.expose,
                filmDensityCmy,
                rawPrint)) {
            const float failure = nanf("");
            launch.logExposureR[index] = failure;
            launch.logExposureG[index] = failure;
            launch.logExposureB[index] = failure;
            return;
        }
        print_apply_exposure_scale_device(launch.expose, rawPrint);
        float logPrint[3] = {0.0f, 0.0f, 0.0f};
        print_log_encode_device(rawPrint, logPrint);
        launch.logExposureR[index] = logPrint[0];
        launch.logExposureG[index] = logPrint[1];
        launch.logExposureB[index] = logPrint[2];
    }

    __global__ void print_develop_log_exposure_kernel(
        GammaValidation::PrintDevelopLogExposureLaunch launch) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        if (x >= launch.width || y >= launch.height) {
            return;
        }
        const std::size_t planeOffset =
            static_cast<std::size_t>(y) *
                launch.linearExposure.rowStrideFloats +
            static_cast<std::size_t>(x);
        const std::size_t outputOffset =
            static_cast<std::size_t>(y) *
                static_cast<std::size_t>(launch.width) +
            static_cast<std::size_t>(x);
        const float diffusedRaw[3] = {
            launch.linearExposure.redSensitiveCForming[planeOffset],
            launch.linearExposure.greenSensitiveMForming[planeOffset],
            launch.linearExposure.blueSensitiveYForming[planeOffset]};
        float logPrint[3] = {0.0f, 0.0f, 0.0f};
        print_log_encode_diffused_device(diffusedRaw, logPrint);
        launch.logExposureR[outputOffset] = logPrint[0];
        launch.logExposureG[outputOffset] = logPrint[1];
        launch.logExposureB[outputOffset] = logPrint[2];
    }

} // namespace

extern "C" cudaError_t gamma_validation_launch_curve_probe(
    const GammaValidation::CurveProbeLaunch* launch,
    void* cudaStreamOpaque) {
    if (!launch || !launch->curve.x || !launch->curve.y ||
        !launch->queries || !launch->gammaFactors || !launch->results ||
        launch->curve.n <= 0 || launch->count <= 0) {
        return cudaErrorInvalidValue;
    }
    cudaStream_t stream = cudaStreamOpaque
                              ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque)
                              : nullptr;
    constexpr int kThreads = 128;
    const int blocks = (launch->count + kThreads - 1) / kThreads;
    curve_probe_kernel<<<blocks, kThreads, 0, stream>>>(*launch);
    return cudaGetLastError();
}

extern "C" cudaError_t gamma_validation_launch_dir_probe(
    const GammaValidation::DirProbeLaunch* launch,
    void* cudaStreamOpaque) {
    if (!launch || !launch->logExposureBgr || !launch->initialDensityBgr ||
        !launch->correctedLogExposureBgr || !launch->finalDensityBgr ||
        launch->count <= 0) {
        return cudaErrorInvalidValue;
    }
    for (const JuicerCuda::DeviceCurveView& curve : launch->densityCurvesBgr) {
        if (!curve.x || !curve.y || curve.n <= 0) {
            return cudaErrorInvalidValue;
        }
    }
    cudaStream_t stream = cudaStreamOpaque
                              ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque)
                              : nullptr;
    constexpr int kThreads = 128;
    const int blocks = (launch->count + kThreads - 1) / kThreads;
    dir_probe_kernel<<<blocks, kThreads, 0, stream>>>(*launch);
    return cudaGetLastError();
}

extern "C" cudaError_t gamma_validation_launch_delay(
    unsigned long long clockCycles,
    unsigned long long* elapsedClockCycles,
    void* cudaStreamOpaque) {
    if (clockCycles == 0 || !elapsedClockCycles) {
        return cudaErrorInvalidValue;
    }
    cudaStream_t stream = cudaStreamOpaque
                              ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque)
                              : nullptr;
    delay_kernel<<<1, 1, 0, stream>>>(clockCycles, elapsedClockCycles);
    return cudaGetLastError();
}

extern "C" cudaError_t gamma_validation_launch_print_log_exposure(
    const GammaValidation::PrintLogExposureLaunch* launch,
    void* cudaStreamOpaque) {
    if (!launch || !launch->expose.active || !launch->densityC ||
        !launch->densityM || !launch->densityY ||
        !launch->logExposureR || !launch->logExposureG ||
        !launch->logExposureB || launch->count <= 0) {
        return cudaErrorInvalidValue;
    }
    cudaStream_t stream = cudaStreamOpaque
                              ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque)
                              : nullptr;
    constexpr int kThreads = 128;
    const int blocks = (launch->count + kThreads - 1) / kThreads;
    print_log_exposure_kernel<<<blocks, kThreads, 0, stream>>>(*launch);
    return cudaGetLastError();
}

extern "C" cudaError_t gamma_validation_launch_print_develop_log_exposure(
    const GammaValidation::PrintDevelopLogExposureLaunch* launch,
    void* cudaStreamOpaque) {
    if (!launch ||
        !launch->linearExposure.redSensitiveCForming ||
        !launch->linearExposure.greenSensitiveMForming ||
        !launch->linearExposure.blueSensitiveYForming ||
        launch->linearExposure.redSensitiveCForming ==
            launch->linearExposure.greenSensitiveMForming ||
        launch->linearExposure.redSensitiveCForming ==
            launch->linearExposure.blueSensitiveYForming ||
        launch->linearExposure.greenSensitiveMForming ==
            launch->linearExposure.blueSensitiveYForming ||
        !launch->logExposureR || !launch->logExposureG ||
        !launch->logExposureB || launch->width <= 0 || launch->height <= 0 ||
        launch->linearExposure.rowStrideFloats <
            static_cast<std::size_t>(launch->width)) {
        return cudaErrorInvalidValue;
    }
    const std::size_t lastRow =
        static_cast<std::size_t>(launch->height - 1);
    const std::size_t width = static_cast<std::size_t>(launch->width);
    if (lastRow != 0 &&
        launch->linearExposure.rowStrideFloats >
            (std::numeric_limits<std::size_t>::max() - width) / lastRow) {
        return cudaErrorInvalidValue;
    }
    cudaStream_t stream = cudaStreamOpaque
                              ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque)
                              : nullptr;
    const dim3 threads(32, 8);
    const dim3 blocks(
        (static_cast<unsigned int>(launch->width) + threads.x - 1) /
            threads.x,
        (static_cast<unsigned int>(launch->height) + threads.y - 1) /
            threads.y);
    print_develop_log_exposure_kernel<<<blocks, threads, 0, stream>>>(*launch);
    return cudaGetLastError();
}
