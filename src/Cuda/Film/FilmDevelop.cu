#include "Cuda/Film/FilmLaunchers.h"
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "Cuda/JuicerCudaDeviceHelpers.cuh"
#include "Cuda/JuicerCudaPayloads.h"

#include "Cuda/Film/CameraFilmLinear.h"

#include "Cuda/Film/CaptureDensity.cuh"

namespace {

    using JuicerCuda::compute_capture_film_density_device;
    using JuicerCuda::valid_camera_film_linear_planes;

    template <typename Params>
    __global__ void camera_film_linear_exposure_kernel(
        Params params,
        JuicerCuda::CameraFilmLinearExposurePlanes planes) {
        const std::size_t width = static_cast<std::size_t>(params.width);
        const std::size_t height = static_cast<std::size_t>(params.height);
        const std::size_t yStep =
            static_cast<std::size_t>(blockDim.y) *
            static_cast<std::size_t>(gridDim.y);
        const std::size_t xStep =
            static_cast<std::size_t>(blockDim.x) *
            static_cast<std::size_t>(gridDim.x);
        const std::size_t pixelBytes =
            static_cast<std::size_t>(params.nComponents) * sizeof(float);
        for (std::size_t y =
                 static_cast<std::size_t>(blockIdx.y) *
                     static_cast<std::size_t>(blockDim.y) +
                 static_cast<std::size_t>(threadIdx.y);
             y < height;
             y += yStep) {
            const char* sourceRow =
                reinterpret_cast<const char*>(params.src) +
                y * params.srcRowBytes;
            for (std::size_t x =
                     static_cast<std::size_t>(blockIdx.x) *
                         static_cast<std::size_t>(blockDim.x) +
                     static_cast<std::size_t>(threadIdx.x);
                 x < width;
                 x += xStep) {
                const float* source = reinterpret_cast<const float*>(
                    sourceRow + x * pixelBytes);
                const float rgb[3] = {source[0], source[1], source[2]};
                // Film storage is B/G/R-sensitive; the binding names the semantic planes.
                float filmRawBgr[3] = {0.0f, 0.0f, 0.0f};
                compute_camera_film_linear_exposure_device(
                    params,
                    rgb,
                    filmRawBgr);
                const std::size_t planeOffset = y * planes.rowStrideFloats + x;
                planes.blueSensitive[planeOffset] = filmRawBgr[0];
                planes.greenSensitive[planeOffset] = filmRawBgr[1];
                planes.redSensitive[planeOffset] = filmRawBgr[2];
            }
        }
    }

    template <typename Params>
    cudaError_t launch_camera_film_linear_exposure(
        const Params* params,
        JuicerCuda::CameraFilmLinearExposurePlanes planes,
        void* cudaStreamOpaque) {
        if (!params || !params->src || params->srcRowBytes == 0 ||
            !(params->nComponents == 3 || params->nComponents == 4) ||
            !valid_camera_film_linear_planes(
                planes,
                params->width,
                params->height)) {
            return cudaErrorInvalidValue;
        }
        cudaStream_t stream = cudaStreamOpaque
                                  ? reinterpret_cast<cudaStream_t>(
                                        cudaStreamOpaque)
                                  : nullptr;
        dim3 threads(32, 8);
        dim3 blocks(
            static_cast<unsigned int>(
                (params->width + static_cast<int>(threads.x) - 1) /
                static_cast<int>(threads.x)),
            static_cast<unsigned int>(
                (params->height + static_cast<int>(threads.y) - 1) /
                static_cast<int>(threads.y)));
        camera_film_linear_exposure_kernel<<<blocks, threads, 0, stream>>>(
            *params,
            planes);
        return cudaGetLastError();
    }

    template <typename Params>
    __global__ void develop_camera_film_linear_density_kernel(
        Params params,
        JuicerCuda::CameraFilmLinearExposurePlanes planes,
        float* outC,
        float* outM,
        float* outY) {
        const JuicerCuda::FilmDevelopPayload& develop = params.filmDevelop;
        const bool useSpatialDir =
            juicer_cuda_spatial_dir_filtered_correction_active_device(develop);
        const std::size_t width = static_cast<std::size_t>(params.width);
        const std::size_t height = static_cast<std::size_t>(params.height);
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
             y < height;
             y += yStep) {
            for (std::size_t x =
                     static_cast<std::size_t>(blockIdx.x) *
                         static_cast<std::size_t>(blockDim.x) +
                     static_cast<std::size_t>(threadIdx.x);
                 x < width;
                 x += xStep) {
                const std::size_t frameOffset = y * width + x;
                const std::size_t planeOffset =
                    y * planes.rowStrideFloats + x;
                const float filmRawBgr[3] = {
                    planes.blueSensitive[planeOffset],
                    planes.greenSensitive[planeOffset],
                    planes.redSensitive[planeOffset]};

                float densityCmy[3] = {0.0f, 0.0f, 0.0f};
                if (useSpatialDir) {
                    float logRawBgr[3] = {0.0f, 0.0f, 0.0f};
                    if (juicer_cuda_spatial_dir_cached_log_raw_active_device(
                            develop)) {
                        juicer_cuda_load_spatial_dir_cached_log_raw_device(
                            develop,
                            frameOffset,
                            logRawBgr);
                    } else {
                        float logSanitizedBgr[3] = {0.0f, 0.0f, 0.0f};
                        float layerPreBgr[3] = {0.0f, 0.0f, 0.0f};
                        FilmDevelopIntermediatesDevice intermediates{};
                        intermediates.logERaw = logRawBgr;
                        intermediates.logESanitized = logSanitizedBgr;
                        intermediates.layerPre = layerPreBgr;
                        compute_logE_from_camera_film_linear_exposure_device(
                            params,
                            filmRawBgr,
                            intermediates);
                    }
                    const unsigned int failures = juicer_cuda_develop_dir_final_device(
                        develop,
                        logRawBgr,
                        frameOffset,
                        densityCmy);
                    signal_dir_failure_device(params.scanStage.scanErrorFlag, failures);
                } else {
                    float logRawBgr[3] = {0.0f, 0.0f, 0.0f};
                    float logSanitizedBgr[3] = {0.0f, 0.0f, 0.0f};
                    float layerPreBgr[3] = {0.0f, 0.0f, 0.0f};
                    FilmDevelopIntermediatesDevice intermediates{};
                    intermediates.logERaw = logRawBgr;
                    intermediates.logESanitized = logSanitizedBgr;
                    intermediates.layerPre = layerPreBgr;
                    compute_logE_from_camera_film_linear_exposure_device(
                        params,
                        filmRawBgr,
                        intermediates);

                    if (develop.dir.mode != JuicerCuda::DirMode::Inactive) {
                        float correctedLogBgr[3] = {
                            logSanitizedBgr[0],
                            logSanitizedBgr[1],
                            logSanitizedBgr[2]};
                        const unsigned int failures = apply_dir_runtime_logE_device(
                            correctedLogBgr,
                            layerPreBgr,
                            develop.dir);
                        signal_dir_failure_device(
                            params.scanStage.scanErrorFlag,
                            failures);
                        if (failures != 0u) {
                            densityCmy[0] = densityCmy[1] = densityCmy[2] = nanf("");
                            outC[frameOffset] = densityCmy[0];
                            outM[frameOffset] = densityCmy[1];
                            outY[frameOffset] = densityCmy[2];
                            continue;
                        }
                        const JuicerCuda::DeviceCurveView curveB = develop.dirDensB;
                        const JuicerCuda::DeviceCurveView curveG = develop.dirDensG;
                        const JuicerCuda::DeviceCurveView curveR = develop.dirDensR;
                        densityCmy[2] = sample_density_at_logE_device(
                            curveB,
                            correctedLogBgr[0],
                            develop.gammaFactorB);
                        densityCmy[1] = sample_density_at_logE_device(
                            curveG,
                            correctedLogBgr[1],
                            develop.gammaFactorG);
                        densityCmy[0] = sample_density_at_logE_device(
                            curveR,
                            correctedLogBgr[2],
                            develop.gammaFactorR);
                    } else {
                        densityCmy[0] = layerPreBgr[2];
                        densityCmy[1] = layerPreBgr[1];
                        densityCmy[2] = layerPreBgr[0];
                    }
                }

                outC[frameOffset] = densityCmy[0];
                outM[frameOffset] = densityCmy[1];
                outY[frameOffset] = densityCmy[2];
            }
        }
    }

    template <typename Params>
    cudaError_t launch_camera_film_linear_capture_density(
        const Params* params,
        JuicerCuda::CameraFilmLinearExposurePlanes planes,
        float* outC,
        float* outM,
        float* outY,
        void* cudaStreamOpaque) {
        if (!params || !outC || !outM || !outY ||
            !valid_camera_film_linear_planes(
                planes,
                params ? params->width : 0,
                params ? params->height : 0)) {
            return cudaErrorInvalidValue;
        }
        cudaStream_t stream = cudaStreamOpaque
                                  ? reinterpret_cast<cudaStream_t>(
                                        cudaStreamOpaque)
                                  : nullptr;
        dim3 threads(32, 8);
        dim3 blocks(
            static_cast<unsigned int>(
                (params->width + static_cast<int>(threads.x) - 1) /
                static_cast<int>(threads.x)),
            static_cast<unsigned int>(
                (params->height + static_cast<int>(threads.y) - 1) /
                static_cast<int>(threads.y)));
        develop_camera_film_linear_density_kernel<<<
            blocks,
            threads,
            0,
            stream>>>(*params, planes, outC, outM, outY);
        return cudaGetLastError();
    }


    template <typename Params>
    __global__ void focused_capture_film_density_kernel(
        Params params,
        float* outC,
        float* outM,
        float* outY) {
        if (!outC || !outM || !outY || !params.src || params.srcRowBytes == 0) {
            return;
        }

        const int nC = params.nComponents;
        if (!(nC == 3 || nC == 4)) {
            return;
        }

        if (params.width <= 0 || params.height <= 0) {
            return;
        }
        const std::size_t xIndex = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
        const std::size_t yIndex = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(blockDim.y) + static_cast<std::size_t>(threadIdx.y);
        if (xIndex >= static_cast<std::size_t>(params.width) || yIndex >= static_cast<std::size_t>(params.height)) {
            return;
        }
        const int x = static_cast<int>(xIndex);
        const int y = static_cast<int>(yIndex);

        const std::size_t idx = yIndex * static_cast<std::size_t>(params.width) + xIndex;
        const std::size_t pixelBytes = static_cast<std::size_t>(nC) * sizeof(float);
        const char* srcRow = reinterpret_cast<const char*>(params.src) +
                             static_cast<std::size_t>(y) * params.srcRowBytes;
        const float* srcPix = reinterpret_cast<const float*>(
            srcRow + static_cast<std::size_t>(x) * pixelBytes);
        const float rgbIn[3] = {srcPix[0], srcPix[1], srcPix[2]};

        float D_cmy[3] = {0.0f, 0.0f, 0.0f};
        if (!compute_capture_film_density_device(params, x, y, rgbIn, D_cmy)) {
            return;
        }
        outC[idx] = D_cmy[0];
        outM[idx] = D_cmy[1];
        outY[idx] = D_cmy[2];
    }

} // namespace

extern "C" cudaError_t juicer_cuda_camera_film_linear_exposure(
    const JuicerCuda::FilmScanRunParams* params,
    JuicerCuda::CameraFilmLinearExposurePlanes planes,
    void* cudaStreamOpaque) {
    return launch_camera_film_linear_exposure(
        params,
        planes,
        cudaStreamOpaque);
}

extern "C" cudaError_t
juicer_cuda_capture_density_from_camera_film_linear(
    const JuicerCuda::FilmScanRunParams* params,
    JuicerCuda::CameraFilmLinearExposurePlanes planes,
    float* outC,
    float* outM,
    float* outY,
    void* cudaStreamOpaque) {
    return launch_camera_film_linear_capture_density(
        params,
        planes,
        outC,
        outM,
        outY,
        cudaStreamOpaque);
}

template <typename Params>
cudaError_t launch_focused_capture_density(
    const Params* hParams,
    float* dDensityC,
    float* dDensityM,
    float* dDensityY,
    void* cudaStreamOpaque) {
    if (!hParams || !dDensityC || !dDensityM || !dDensityY) {
        return cudaErrorInvalidValue;
    }
    const Params& focusedParams = *hParams;
    if (!focusedParams.src) {
        return cudaErrorInvalidValue;
    }
    if (focusedParams.width <= 0 || focusedParams.height <= 0) {
        return cudaSuccess;
    }
    if (!(focusedParams.nComponents == 3 || focusedParams.nComponents == 4) ||
        focusedParams.srcRowBytes == 0) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream =
        cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
    dim3 threads(32, 8);
    dim3 blocks(
        (focusedParams.width + threads.x - 1) / threads.x,
        (focusedParams.height + threads.y - 1) / threads.y);

    focused_capture_film_density_kernel<<<blocks, threads, 0, stream>>>(
        *hParams,
        dDensityC,
        dDensityM,
        dDensityY);
    return cudaGetLastError();
}

extern "C" cudaError_t juicer_cuda_capture_density(
    const JuicerCuda::FilmScanRunParams* hParams,
    float* dDensityC,
    float* dDensityM,
    float* dDensityY,
    void* cudaStreamOpaque) {
    return launch_focused_capture_density(
        hParams,
        dDensityC,
        dDensityM,
        dDensityY,
        cudaStreamOpaque);
}
