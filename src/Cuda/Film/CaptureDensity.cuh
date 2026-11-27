#pragma once

#include "Cuda/JuicerCudaDeviceHelpers.cuh"

namespace JuicerCuda {
    template <bool FromCarrier = false, typename Params>
    __device__ __forceinline__ bool load_spatial_dir_log_raw_for_final_develop_device(
        const Params& params,
        std::size_t pixelIndex,
        const float* rgbIn,
        float logE_raw[3]) {
        const JuicerCuda::FilmDevelopPayload& dev = params.filmDevelop;
        int cachedLogRawMask = 0;
        if (dev.spatialDir.logRawB) {
            logE_raw[0] = ldg_f(dev.spatialDir.logRawB + pixelIndex);
            cachedLogRawMask |= kJuicerLogRawMaskB;
        }
        if (dev.spatialDir.logRawG) {
            logE_raw[1] = ldg_f(dev.spatialDir.logRawG + pixelIndex);
            cachedLogRawMask |= kJuicerLogRawMaskG;
        }
        if (dev.spatialDir.logRawR) {
            logE_raw[2] = ldg_f(dev.spatialDir.logRawR + pixelIndex);
            cachedLogRawMask |= kJuicerLogRawMaskR;
        }
        const int missingLogRawMask = kJuicerLogRawMaskBgr & ~cachedLogRawMask;
        if (missingLogRawMask == 0) {
            return true;
        }

        if (rgbIn) {
            if constexpr (FromCarrier) {
                float corrected[3]{rgbIn[0], rgbIn[1], rgbIn[2]};
                apply_film_route_correction_device(params, corrected);
                float carrierLog[3]{};
                compute_logE_raw_from_film_raw_device(params, corrected, carrierLog);
                for (int channel = 0; channel < 3; ++channel) {
                    if ((missingLogRawMask & (1 << channel)) != 0) {
                        logE_raw[channel] = carrierLog[channel];
                    }
                }
            } else {
                compute_logE_raw_selected_device(params, rgbIn, missingLogRawMask, logE_raw);
            }
            return true;
        }

        const int nC = params.nComponents;
        if (!params.src || params.srcRowBytes == 0 || params.width <= 0 ||
            !(nC == 3 || nC == 4)) {
            return false;
        }

        const int y = static_cast<int>(pixelIndex / static_cast<std::size_t>(params.width));
        const int x =
            static_cast<int>(pixelIndex - static_cast<std::size_t>(y) *
                                              static_cast<std::size_t>(params.width));
        const std::size_t pixelBytes = static_cast<std::size_t>(nC) * sizeof(float);
        const char* srcRow =
            reinterpret_cast<const char*>(params.src) +
            static_cast<std::size_t>(y) * params.srcRowBytes;
        const float* srcPix =
            reinterpret_cast<const float*>(srcRow + static_cast<std::size_t>(x) * pixelBytes);

        const float sourceRgb[3] = {srcPix[0], srcPix[1], srcPix[2]};
        compute_logE_raw_selected_device(params, sourceRgb, missingLogRawMask, logE_raw);
        return true;
    }

    template <bool FromCarrier = false, typename Params>
    __device__ __forceinline__ bool compute_capture_film_density_device(
        const Params& params,
        int x,
        int y,
        const float rgbIn[3],
        float densityCmy[3]) {
        const JuicerCuda::FilmDevelopPayload& dev = params.filmDevelop;
        const bool useSpatialDir =
            juicer_cuda_spatial_dir_filtered_correction_active_device(dev);

        if (useSpatialDir) {
            const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) +
                               static_cast<size_t>(x);
            float logE_raw[3] = {0.0f, 0.0f, 0.0f};
            if (!load_spatial_dir_log_raw_for_final_develop_device<FromCarrier>(
                    params,
                    idx,
                    rgbIn,
                    logE_raw)) {
                return false;
            }
            const unsigned int failures =
                juicer_cuda_develop_dir_final_device(dev, logE_raw, idx, densityCmy);
            signal_dir_failure_device(params.scanStage.scanErrorFlag, failures);
            return true;
        }

        float logE_raw[3] = {0.0f, 0.0f, 0.0f};
        float logE_sanitized[3] = {0.0f, 0.0f, 0.0f};
        float layerPre[3] = {0.0f, 0.0f, 0.0f};
        FilmDevelopIntermediatesDevice intermediates{};
        intermediates.logERaw = logE_raw;
        intermediates.logESanitized = logE_sanitized;
        intermediates.layerPre = layerPre;
        if constexpr (FromCarrier) {
            compute_logE_from_camera_film_linear_exposure_device(params, rgbIn, intermediates);
        } else {
            compute_logE_and_layer_pre_device(params, rgbIn, intermediates);
        }

        if (dev.dir.mode != JuicerCuda::DirMode::Inactive) {
            float logE_corr[3] = {
                logE_sanitized[0],
                logE_sanitized[1],
                logE_sanitized[2]};
            const unsigned int failures = apply_dir_runtime_logE_device(
                logE_corr,
                layerPre,
                dev.dir);
            signal_dir_failure_device(params.scanStage.scanErrorFlag, failures);
            if (failures != 0u) {
                densityCmy[0] = densityCmy[1] = densityCmy[2] = nanf("");
                return true;
            }

            const JuicerCuda::DeviceCurveView cB = dev.dirDensB;
            const JuicerCuda::DeviceCurveView cG = dev.dirDensG;
            const JuicerCuda::DeviceCurveView cR = dev.dirDensR;

            densityCmy[2] =
                sample_density_at_logE_device(cB, logE_corr[0], dev.gammaFactorB);
            densityCmy[1] =
                sample_density_at_logE_device(cG, logE_corr[1], dev.gammaFactorG);
            densityCmy[0] =
                sample_density_at_logE_device(cR, logE_corr[2], dev.gammaFactorR);
        } else {
            // Map B/G/R layer densities to C/M/Y dyes.
            densityCmy[0] = layerPre[2];
            densityCmy[1] = layerPre[1];
            densityCmy[2] = layerPre[0];
        }
        return true;
    }

} // namespace JuicerCuda
