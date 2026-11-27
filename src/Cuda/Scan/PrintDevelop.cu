#include "Cuda/Scan/PrintDevelop.h"
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "Cuda/JuicerCudaDeviceHelpers.cuh"

__global__ void develop_print_density_kernel(
    int width,
    int height,
    JuicerCuda::PrintExposePayload printExpose,
    JuicerCuda::PrintDevelopPayload printDevelop,
    float* ioC,
    float* ioM,
    float* ioY,
    const float* filmDustTransmittance) {
    if (!printExpose.active) {
        return;
    }

    if (!ioC || !ioM || !ioY || width <= 0 || height <= 0) {
        return;
    }

    const std::size_t widthCount = static_cast<std::size_t>(width);
    const std::size_t heightCount = static_cast<std::size_t>(height);
    const std::size_t yStep = static_cast<std::size_t>(blockDim.y) * static_cast<std::size_t>(gridDim.y);
    const std::size_t xStep = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    for (std::size_t y = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(blockDim.y) + static_cast<std::size_t>(threadIdx.y); y < heightCount; y += yStep) {
        for (std::size_t x = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x); x < widthCount; x += xStep) {
            const std::size_t idx = y * width + x;
            float D_cmy[3] = {ioC[idx], ioM[idx], ioY[idx]};
            float* cyan = nullptr;
            apply_print_pipeline_device(printExpose, printDevelop, D_cmy, filmDustTransmittance ? filmDustTransmittance[idx] : 1.0f, cyan);
            ioC[idx] = D_cmy[0];
            ioM[idx] = D_cmy[1];
            ioY[idx] = D_cmy[2];
        }
    }
}

// Dimensions are positional in the focused Print launches.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
__global__ void build_enlarger_print_linear_exposure_kernel(
    int width,
    int height,
    JuicerCuda::PrintExposePayload printExpose,
    const float* densityC,
    const float* densityM,
    const float* densityY,
    JuicerCuda::EnlargerPrintLinearExposurePlanes planes,
    const float* filmDustTransmittance) {
    const std::size_t widthCount = static_cast<std::size_t>(width);
    const std::size_t heightCount = static_cast<std::size_t>(height);
    const std::size_t yStep =
        static_cast<std::size_t>(blockDim.y) *
        static_cast<std::size_t>(gridDim.y);
    const std::size_t xStep =
        static_cast<std::size_t>(blockDim.x) *
        static_cast<std::size_t>(gridDim.x);
    for (std::size_t y =
             static_cast<std::size_t>(blockIdx.y) *
                 static_cast<std::size_t>(blockDim.y) +
             static_cast<std::size_t>(threadIdx.y);
         y < heightCount;
         y += yStep) {
        for (std::size_t x =
                 static_cast<std::size_t>(blockIdx.x) *
                     static_cast<std::size_t>(blockDim.x) +
                 static_cast<std::size_t>(threadIdx.x);
             x < widthCount;
             x += xStep) {
            const std::size_t frameOffset = y * width + x;
            const float filmDensityCmy[3] = {
                densityC[frameOffset],
                densityM[frameOffset],
                densityY[frameOffset]};
            float innerRaw[3] = {0.0f, 0.0f, 0.0f};
            float linearForDiffusion[3] = {0.0f, 0.0f, 0.0f};
            if (!print_inner_raw_device(
                    printExpose,
                    filmDensityCmy,
                    innerRaw,
                    filmDustTransmittance ? filmDustTransmittance[frameOffset] : 1.0f)) {
                continue;
            }
            print_linear_for_diffusion_device(
                printExpose,
                innerRaw,
                linearForDiffusion);
            const std::size_t planeOffset =
                y * planes.rowStrideFloats + x;
            planes.redSensitiveCForming[planeOffset] =
                linearForDiffusion[0];
            planes.greenSensitiveMForming[planeOffset] =
                linearForDiffusion[1];
            planes.blueSensitiveYForming[planeOffset] =
                linearForDiffusion[2];
        }
    }
}

__global__ void develop_print_density_from_enlarger_linear_kernel(
    int width,
    int height,
    JuicerCuda::PrintDevelopPayload printDevelop,
    JuicerCuda::EnlargerPrintLinearExposurePlanes planes,
    float* densityC,
    float* densityM,
    float* densityY) {
    const std::size_t widthCount = static_cast<std::size_t>(width);
    const std::size_t heightCount = static_cast<std::size_t>(height);
    const std::size_t yStep =
        static_cast<std::size_t>(blockDim.y) *
        static_cast<std::size_t>(gridDim.y);
    const std::size_t xStep =
        static_cast<std::size_t>(blockDim.x) *
        static_cast<std::size_t>(gridDim.x);
    for (std::size_t y =
             static_cast<std::size_t>(blockIdx.y) *
                 static_cast<std::size_t>(blockDim.y) +
             static_cast<std::size_t>(threadIdx.y);
         y < heightCount;
         y += yStep) {
        for (std::size_t x =
                 static_cast<std::size_t>(blockIdx.x) *
                     static_cast<std::size_t>(blockDim.x) +
                 static_cast<std::size_t>(threadIdx.x);
             x < widthCount;
             x += xStep) {
            const std::size_t frameOffset = y * width + x;
            const std::size_t planeOffset =
                y * planes.rowStrideFloats + x;
            const float diffusedRaw[3] = {
                planes.redSensitiveCForming[planeOffset],
                planes.greenSensitiveMForming[planeOffset],
                planes.blueSensitiveYForming[planeOffset]};
            float logPrint[3] = {0.0f, 0.0f, 0.0f};
            float printDensityCmy[3] = {0.0f, 0.0f, 0.0f};
            print_log_encode_diffused_device(diffusedRaw, logPrint);
            print_sample_density_curves_device(
                printDevelop,
                logPrint,
                printDensityCmy);
            densityC[frameOffset] = printDensityCmy[0];
            densityM[frameOffset] = printDensityCmy[1];
            densityY[frameOffset] = printDensityCmy[2];
        }
    }
}
// NOLINTEND(bugprone-easily-swappable-parameters)

cudaError_t launch_focused_print_continuation_from_capture_density(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    float* dDensityC,
    float* dDensityM,
    float* dDensityY,
    const float* filmDustTransmittance,
    void* cudaStreamOpaque) {
    if (!hParams || !dDensityC || !dDensityM || !dDensityY) {
        return cudaErrorInvalidValue;
    }
    if (hParams->width <= 0 || hParams->height <= 0) {
        return cudaSuccess;
    }

    cudaStream_t stream =
        cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
    dim3 threads(32, 8);
    dim3 blocks(
        (hParams->width + threads.x - 1) / threads.x,
        (hParams->height + threads.y - 1) / threads.y);
    develop_print_density_kernel<<<blocks, threads, 0, stream>>>(
        hParams->width,
        hParams->height,
        hParams->printExpose,
        hParams->printDevelop,
        dDensityC,
        dDensityM,
        dDensityY,
        filmDustTransmittance);
    return cudaGetLastError();
}

bool valid_enlarger_print_linear_planes(
    const JuicerCuda::EnlargerPrintLinearExposurePlanes& planes,
    int width,
    int height) noexcept {
    if (!planes.redSensitiveCForming ||
        !planes.greenSensitiveMForming ||
        !planes.blueSensitiveYForming ||
        planes.redSensitiveCForming == planes.greenSensitiveMForming ||
        planes.redSensitiveCForming == planes.blueSensitiveYForming ||
        planes.greenSensitiveMForming == planes.blueSensitiveYForming ||
        width <= 0 || height <= 0 ||
        planes.rowStrideFloats < static_cast<std::size_t>(width)) {
        return false;
    }
    const std::size_t lastRow = static_cast<std::size_t>(height - 1);
    const std::size_t widthCount = static_cast<std::size_t>(width);
    return lastRow == 0 ||
           planes.rowStrideFloats <=
               (std::numeric_limits<std::size_t>::max() - widthCount) /
                   lastRow;
}

cudaError_t launch_focused_enlarger_print_linear_exposure(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    const float* dDensityC,
    const float* dDensityM,
    const float* dDensityY,
    JuicerCuda::EnlargerPrintLinearExposurePlanes planes,
    const float* filmDustTransmittance,
    void* cudaStreamOpaque) {
    if (!hParams || !dDensityC || !dDensityM || !dDensityY) {
        return cudaErrorInvalidValue;
    }
    if (hParams->width <= 0 || hParams->height <= 0) {
        return cudaSuccess;
    }
    if (!hParams->printExpose.active ||
        !valid_enlarger_print_linear_planes(
            planes,
            hParams->width,
            hParams->height)) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream =
        cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
    dim3 threads(32, 8);
    dim3 blocks(
        (hParams->width + threads.x - 1) / threads.x,
        (hParams->height + threads.y - 1) / threads.y);
    build_enlarger_print_linear_exposure_kernel<<<blocks, threads, 0, stream>>>(
        hParams->width,
        hParams->height,
        hParams->printExpose,
        dDensityC,
        dDensityM,
        dDensityY,
        planes,
        filmDustTransmittance);
    return cudaGetLastError();
}

cudaError_t launch_focused_print_develop_from_enlarger_linear(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    JuicerCuda::EnlargerPrintLinearExposurePlanes planes,
    float* dDensityC,
    float* dDensityM,
    float* dDensityY,
    void* cudaStreamOpaque) {
    if (!hParams || !dDensityC || !dDensityM || !dDensityY) {
        return cudaErrorInvalidValue;
    }
    if (hParams->width <= 0 || hParams->height <= 0) {
        return cudaSuccess;
    }
    if (!hParams->printExpose.active ||
        !valid_enlarger_print_linear_planes(
            planes,
            hParams->width,
            hParams->height)) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream =
        cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
    dim3 threads(32, 8);
    dim3 blocks(
        (hParams->width + threads.x - 1) / threads.x,
        (hParams->height + threads.y - 1) / threads.y);
    develop_print_density_from_enlarger_linear_kernel<<<blocks, threads, 0, stream>>>(
        hParams->width,
        hParams->height,
        hParams->printDevelop,
        planes,
        dDensityC,
        dDensityM,
        dDensityY);
    return cudaGetLastError();
}

extern "C" cudaError_t juicer_cuda_print_focused_continue_from_capture_density(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    float* dDensityC,
    float* dDensityM,
    float* dDensityY,
    const float* filmDustTransmittance,
    void* cudaStreamOpaque) {
    return launch_focused_print_continuation_from_capture_density(
        hParams,
        dDensityC,
        dDensityM,
        dDensityY,
        filmDustTransmittance,
        cudaStreamOpaque);
}

extern "C" cudaError_t juicer_cuda_print_focused_enlarger_linear_exposure(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    const float* dDensityC,
    const float* dDensityM,
    const float* dDensityY,
    JuicerCuda::EnlargerPrintLinearExposurePlanes planes,
    const float* filmDustTransmittance,
    void* cudaStreamOpaque) {
    return launch_focused_enlarger_print_linear_exposure(
        hParams,
        dDensityC,
        dDensityM,
        dDensityY,
        planes,
        filmDustTransmittance,
        cudaStreamOpaque);
}

extern "C" cudaError_t
juicer_cuda_print_focused_develop_from_enlarger_linear(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    JuicerCuda::EnlargerPrintLinearExposurePlanes planes,
    float* dDensityC,
    float* dDensityM,
    float* dDensityY,
    void* cudaStreamOpaque) {
    return launch_focused_print_develop_from_enlarger_linear(
        hParams,
        planes,
        dDensityC,
        dDensityM,
        dDensityY,
        cudaStreamOpaque);
}
