// Cuda/Scan/JuicerCudaScanStage.cu
// Stage-aligned CUDA TU for scan stage kernels.
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "Cuda/JuicerCudaPayloads.h"
#include "Cuda/JuicerCudaKernelsUtil.cuh"
#include "Cuda/JuicerCudaPrintPipeline.cuh"
#include "Cuda/Film/JuicerCudaFilmExposure.cuh"
#include "Cuda/Film/JuicerCudaFilmDevelop.cuh"
#include "Cuda/Scan/JuicerCudaScanStage.cuh"
#include "Cuda/Scan/JuicerCudaScanOutput.cuh"

// Optics kernels are defined in Cuda/Scan/JuicerCudaScannerOptics.cu.
__global__ void optics_glare_generate_kernel(
    float* out,
    int width,
    int height,
    std::uint64_t glareSeed,
    std::uint64_t mediumId,
    int originX,
    int originY,
    float percent,
    float roughness);
__global__ void optics_blur_horizontal_kernel(
    const float* JUICER_RESTRICT in,
    float* out,
    int width,
    int height,
    const float* JUICER_RESTRICT k,
    int radius);
__global__ void optics_blur_vertical_kernel(
    const float* JUICER_RESTRICT in,
    float* out,
    int width,
    int height,
    const float* JUICER_RESTRICT k,
    int radius);
__global__ void optics_unsharp_combine_kernel(
    float* inOut,
    const float* JUICER_RESTRICT blurred,
    int n,
    float amount);

// Film/print stage kernels are defined in their respective TUs.
__global__ void expose_film_raw_kernel(
    JuicerCuda::PipelineRunParams params,
    float* outB,
    float* outG,
    float* outR);
__global__ void develop_film_density_kernel(
    JuicerCuda::PipelineRunParams params,
    float* outC,
    float* outM,
    float* outY);
__global__ void develop_film_density_from_raw_kernel(
    JuicerCuda::PipelineRunParams params,
    const float* inB,
    const float* inG,
    const float* inR,
    float* outC,
    float* outM,
    float* outY);
__global__ void grain_clear_kernel(float* out, int n);
__global__ void grain_accumulate_kernel(float* dst, const float* src, int n);
__global__ void grain_add_bias_kernel(float* inOut, int n, float bias);
__global__ void grain_multiply_kernel(float* inOut, const float* mult, int n);
__global__ void grain_apply_simple_kernel(
    JuicerCuda::PipelineRunParams params,
    float* inOut,
    int channelIndex);
__global__ void grain_layer_kernel(
    JuicerCuda::PipelineRunParams params,
    const float* inDensity,
    float* outGrain,
    int channelIndex,
    int sublayerIndex);
__global__ void develop_print_density_kernel(
    JuicerCuda::PipelineRunParams params,
    float* ioC,
    float* ioM,
    float* ioY);
__global__ void halation_apply_kernel(
    float* inOut,
    const float* blurred,
    int n,
    float strength);

namespace {

    __global__ void pipeline_direct_kernel(JuicerCuda::PipelineRunParams params) {
        const JuicerCuda::FilmDevelopPayload& dev = params.filmDevelop;
        const JuicerCuda::PrintExposePayload& printExpose = params.printExpose;
        const JuicerCuda::PrintDevelopPayload& printDevelop = params.printDevelop;
        const JuicerCuda::ScanStagePayload& scan = params.scanStage;
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= params.width || y >= params.height) {
            return;
        }

        if (!params.src || !params.dst || params.srcRowBytes == 0 || params.dstRowBytes == 0) {
            return;
        }

        const int nC = params.nComponents;
        if (!(nC == 3 || nC == 4)) {
            return;
        }

        const std::size_t pixelBytes = static_cast<std::size_t>(nC) * sizeof(float);
        const char* srcRow = reinterpret_cast<const char*>(params.src) + static_cast<std::size_t>(y) * params.srcRowBytes;
        const float* srcPix = reinterpret_cast<const float*>(srcRow + static_cast<std::size_t>(x) * pixelBytes);

        const float rgbIn[3] = { srcPix[0], srcPix[1], srcPix[2] };

        float logE_raw[3] = { 0.0f, 0.0f, 0.0f };
        float logE_sanitized[3] = { 0.0f, 0.0f, 0.0f };
        float layerPre[3] = { 0.0f, 0.0f, 0.0f };
        compute_logE_and_layer_pre_device(params, rgbIn, logE_raw, logE_sanitized, layerPre);

        float D_cmy[3] = { 0.0f, 0.0f, 0.0f };
        const bool useSpatialDir =
            dev.spatialDir.active &&
            dev.spatialDir.corrY && dev.spatialDir.corrM && dev.spatialDir.corrC;
        if (useSpatialDir) {
            const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
            const float corrY = dev.spatialDir.corrY[idx];
            const float corrM = dev.spatialDir.corrM[idx];
            const float corrC = dev.spatialDir.corrC[idx];

            float logE_corr[3] = {
                logE_raw[0] - corrY,
                logE_raw[1] - corrM,
                logE_raw[2] - corrC
            };

            const JuicerCuda::DeviceCurveView cB = dev.dirPrecorrected ? dev.dirDensB : dev.densB;
            const JuicerCuda::DeviceCurveView cG = dev.dirPrecorrected ? dev.dirDensG : dev.densG;
            const JuicerCuda::DeviceCurveView cR = dev.dirPrecorrected ? dev.dirDensR : dev.densR;

            logE_corr[0] = sanitize_inf_logE_for_curve_device(logE_corr[0], cB.x, cB.n);
            logE_corr[1] = sanitize_inf_logE_for_curve_device(logE_corr[1], cG.x, cG.n);
            logE_corr[2] = sanitize_inf_logE_for_curve_device(logE_corr[2], cR.x, cR.n);

            const float DY = sample_density_at_logE_device(cB.x, cB.y, cB.n, logE_corr[0], dev.gammaFactorB);
            const float DM = sample_density_at_logE_device(cG.x, cG.y, cG.n, logE_corr[1], dev.gammaFactorG);
            const float DC = sample_density_at_logE_device(cR.x, cR.y, cR.n, logE_corr[2], dev.gammaFactorR);

            D_cmy[0] = DC;
            D_cmy[1] = DM;
            D_cmy[2] = DY;
        }
        else if (dev.dir.active) {
            float logE_corr[3] = { logE_sanitized[0], logE_sanitized[1], logE_sanitized[2] };
            apply_dir_runtime_logE_device(logE_corr, layerPre, dev.dir, dev.densB, dev.densG, dev.densR);

            const JuicerCuda::DeviceCurveView cB = dev.dirPrecorrected ? dev.dirDensB : dev.densB;
            const JuicerCuda::DeviceCurveView cG = dev.dirPrecorrected ? dev.dirDensG : dev.densG;
            const JuicerCuda::DeviceCurveView cR = dev.dirPrecorrected ? dev.dirDensR : dev.densR;

            const float DY = sample_density_at_logE_device(cB.x, cB.y, cB.n, logE_corr[0], dev.gammaFactorB);
            const float DM = sample_density_at_logE_device(cG.x, cG.y, cG.n, logE_corr[1], dev.gammaFactorG);
            const float DC = sample_density_at_logE_device(cR.x, cR.y, cR.n, logE_corr[2], dev.gammaFactorR);

            D_cmy[0] = DC;
            D_cmy[1] = DM;
            D_cmy[2] = DY;
        }
        else {
            // Map B/G/R layer densities to C/M/Y dyes
            D_cmy[0] = layerPre[2];
            D_cmy[1] = layerPre[1];
            D_cmy[2] = layerPre[0];
        }

        apply_print_pipeline_device(printExpose, printDevelop, D_cmy);

        // Scan: normalize density -> logXYZ
        double D_norm[3];
        if (scan.scanTables.mediumIsNegative) {
            D_norm[0] = (static_cast<double>(D_cmy[0]) + static_cast<double>(scan.scanTables.min_cmy[0])) * static_cast<double>(scan.scanTables.inv_max_cmy[0]);
            D_norm[1] = (static_cast<double>(D_cmy[1]) + static_cast<double>(scan.scanTables.min_cmy[1])) * static_cast<double>(scan.scanTables.inv_max_cmy[1]);
            D_norm[2] = (static_cast<double>(D_cmy[2]) + static_cast<double>(scan.scanTables.min_cmy[2])) * static_cast<double>(scan.scanTables.inv_max_cmy[2]);
        }
        else {
            D_norm[0] = static_cast<double>(D_cmy[0]) * static_cast<double>(scan.scanTables.inv_max_cmy[0]);
            D_norm[1] = static_cast<double>(D_cmy[1]) * static_cast<double>(scan.scanTables.inv_max_cmy[1]);
            D_norm[2] = static_cast<double>(D_cmy[2]) * static_cast<double>(scan.scanTables.inv_max_cmy[2]);
        }

        double logXYZ[3] = { 0.0, 0.0, 0.0 };
        scan_log_xyz_device(scan, D_norm, logXYZ);
        const double xyz[3] = {
            pow(10.0, logXYZ[0]),
            pow(10.0, logXYZ[1]),
            pow(10.0, logXYZ[2])
        };

        double adapted[3];
        mat3_mul_vec_double_device(scan.scanColor.cat02, xyz, adapted);
        double rgbOut[3];
        mat3_mul_vec_double_device(scan.scanColor.xyzToRgb, adapted, rgbOut);

        if (!isfinite(rgbOut[0]) || !isfinite(rgbOut[1]) || !isfinite(rgbOut[2])) {
            signal_scan_error_device(scan.scanErrorFlag);
            const std::size_t pixelBytes = static_cast<std::size_t>(nC) * sizeof(float);
            char* dstRow = reinterpret_cast<char*>(params.dst) + static_cast<std::size_t>(y) * params.dstRowBytes;
            float* dstPix = reinterpret_cast<float*>(dstRow + static_cast<std::size_t>(x) * pixelBytes);
            if (dstPix) {
                dstPix[0] = 0.0f;
                dstPix[1] = 0.0f;
                dstPix[2] = 0.0f;
                if (nC == 4) {
                    const char* srcRow = reinterpret_cast<const char*>(params.src) + static_cast<std::size_t>(y) * params.srcRowBytes;
                    const float* srcPix = reinterpret_cast<const float*>(srcRow + static_cast<std::size_t>(x) * pixelBytes);
                    dstPix[3] = srcPix ? srcPix[3] : 1.0f;
                }
            }
            return;
        }

        // Output encoding + clamp
        apply_output_encoding_device(scan.scanColor.encoding, rgbOut);

        char* dstRow = reinterpret_cast<char*>(params.dst) + static_cast<std::size_t>(y) * params.dstRowBytes;
        float* dstPix = reinterpret_cast<float*>(dstRow + static_cast<std::size_t>(x) * pixelBytes);
        dstPix[0] = static_cast<float>(rgbOut[0]);
        dstPix[1] = static_cast<float>(rgbOut[1]);
        dstPix[2] = static_cast<float>(rgbOut[2]);
        if (nC == 4) {
            dstPix[3] = srcPix[3];
        }
    }

    __global__ void scan_linear_rgb_kernel(
        JuicerCuda::PipelineRunParams params,
        const float* inC,
        const float* inM,
        const float* inY,
        const float* glarePercent,
        float* outR,
        float* outG,
        float* outB)
    {
        const JuicerCuda::ScanStagePayload& scan = params.scanStage;
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= params.width || y >= params.height) {
            return;
        }

        if (!inC || !inM || !inY || !outR || !outG || !outB) {
            return;
        }

        const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);

        const float D_cmy[3] = { inC[idx], inM[idx], inY[idx] };

        // Scan: normalize density -> logXYZ
        double D_norm[3];
        if (scan.scanTables.mediumIsNegative) {
            D_norm[0] = (static_cast<double>(D_cmy[0]) + static_cast<double>(scan.scanTables.min_cmy[0])) * static_cast<double>(scan.scanTables.inv_max_cmy[0]);
            D_norm[1] = (static_cast<double>(D_cmy[1]) + static_cast<double>(scan.scanTables.min_cmy[1])) * static_cast<double>(scan.scanTables.inv_max_cmy[1]);
            D_norm[2] = (static_cast<double>(D_cmy[2]) + static_cast<double>(scan.scanTables.min_cmy[2])) * static_cast<double>(scan.scanTables.inv_max_cmy[2]);
        }
        else {
            D_norm[0] = static_cast<double>(D_cmy[0]) * static_cast<double>(scan.scanTables.inv_max_cmy[0]);
            D_norm[1] = static_cast<double>(D_cmy[1]) * static_cast<double>(scan.scanTables.inv_max_cmy[1]);
            D_norm[2] = static_cast<double>(D_cmy[2]) * static_cast<double>(scan.scanTables.inv_max_cmy[2]);
        }

        double logXYZ[3] = { 0.0, 0.0, 0.0 };
        scan_log_xyz_device(scan, D_norm, logXYZ);

        double xyz[3] = {
            pow(10.0, logXYZ[0]),
            pow(10.0, logXYZ[1]),
            pow(10.0, logXYZ[2])
        };

        if (glarePercent) {
            const double glare = static_cast<double>(glarePercent[idx]) * 0.01;
            xyz[0] += glare * static_cast<double>(scan.scanColor.illuminantXYZ[0]);
            xyz[1] += glare * static_cast<double>(scan.scanColor.illuminantXYZ[1]);
            xyz[2] += glare * static_cast<double>(scan.scanColor.illuminantXYZ[2]);
        }

        double adapted[3];
        mat3_mul_vec_double_device(scan.scanColor.cat02, xyz, adapted);
        double rgbOut[3];
        mat3_mul_vec_double_device(scan.scanColor.xyzToRgb, adapted, rgbOut);

        if (!isfinite(rgbOut[0]) || !isfinite(rgbOut[1]) || !isfinite(rgbOut[2])) {
            signal_scan_error_device(scan.scanErrorFlag);
            outR[idx] = 0.0f;
            outG[idx] = 0.0f;
            outB[idx] = 0.0f;
            return;
        }

        outR[idx] = static_cast<float>(rgbOut[0]);
        outG[idx] = static_cast<float>(rgbOut[1]);
        outB[idx] = static_cast<float>(rgbOut[2]);
    }

    __global__ void scan_output_encode_kernel(
        JuicerCuda::PipelineRunParams params,
        const float* rgbR,
        const float* rgbG,
        const float* rgbB)
    {
        const JuicerCuda::ScanStagePayload& scan = params.scanStage;
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= params.width || y >= params.height) {
            return;
        }

        if (!params.src || !params.dst || params.srcRowBytes == 0 || params.dstRowBytes == 0) {
            return;
        }
        if (!rgbR || !rgbG || !rgbB) {
            return;
        }

        const int nC = params.nComponents;
        if (!(nC == 3 || nC == 4)) {
            return;
        }

        const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
        double rgbOut[3] = {
            static_cast<double>(rgbR[idx]),
            static_cast<double>(rgbG[idx]),
            static_cast<double>(rgbB[idx])
        };
        apply_output_encoding_device(scan.scanColor.encoding, rgbOut);

        const std::size_t pixelBytes = static_cast<std::size_t>(nC) * sizeof(float);
        char* dstRow = reinterpret_cast<char*>(params.dst) + static_cast<std::size_t>(y) * params.dstRowBytes;
        float* dstPix = reinterpret_cast<float*>(dstRow + static_cast<std::size_t>(x) * pixelBytes);
        dstPix[0] = static_cast<float>(rgbOut[0]);
        dstPix[1] = static_cast<float>(rgbOut[1]);
        dstPix[2] = static_cast<float>(rgbOut[2]);

        if (nC == 4) {
            const char* srcRow = reinterpret_cast<const char*>(params.src) + static_cast<std::size_t>(y) * params.srcRowBytes;
            const float* srcPix = reinterpret_cast<const float*>(srcRow + static_cast<std::size_t>(x) * pixelBytes);
            dstPix[3] = srcPix ? srcPix[3] : 1.0f;
        }
    }

} // namespace

extern "C" cudaError_t juicer_cuda_negative_pipeline(
    const JuicerCuda::PipelineRunParams* hParams,
    void* cudaStreamOpaque)
{
    if (!hParams) {
        return cudaErrorInvalidValue;
    }

    const JuicerCuda::PipelineRunParams params = *hParams;
    if (!params.src || !params.dst) {
        return cudaErrorInvalidValue;
    }
    if (params.width <= 0 || params.height <= 0) {
        return cudaSuccess;
    }
    if (!(params.nComponents == 3 || params.nComponents == 4)) {
        return cudaErrorInvalidValue;
    }
    if (params.srcRowBytes == 0 || params.dstRowBytes == 0) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

    dim3 threads(32, 8);
    dim3 blocks(
        static_cast<unsigned int>((params.width + threads.x - 1) / threads.x),
        static_cast<unsigned int>((params.height + threads.y - 1) / threads.y));
    pipeline_direct_kernel<<<blocks, threads, 0, stream>>>(params);
    return cudaGetLastError();
}

extern "C" cudaError_t juicer_cuda_negative_pipeline_optics(
    const JuicerCuda::PipelineRunParams* hParams,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    float* dScratchBlurred,
    float* dAux,
    const float* dLensBlurKernel,
    int lensBlurRadius,
    const float* dUnsharpKernel,
    int unsharpRadius,
    float unsharpAmount,
    int glareOriginX,
    int glareOriginY,
    std::uint64_t glareSeed,
    float glarePercent,
    float glareRoughness,
    const float* dGlareKernel,
    int glareRadius,
    void* cudaStreamOpaque)
{
    if (!hParams) {
        return cudaErrorInvalidValue;
    }

    const JuicerCuda::PipelineRunParams params = *hParams;
    if (!params.src || !params.dst) {
        return cudaErrorInvalidValue;
    }
    if (params.width <= 0 || params.height <= 0) {
        return cudaSuccess;
    }
    if (!(params.nComponents == 3 || params.nComponents == 4)) {
        return cudaErrorInvalidValue;
    }
    if (params.srcRowBytes == 0 || params.dstRowBytes == 0) {
        return cudaErrorInvalidValue;
    }
    if (!dRgbR || !dRgbG || !dRgbB || !dTmp) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

    dim3 threads2D(32, 8);
    dim3 blocks2D(
        static_cast<unsigned int>((params.width + threads2D.x - 1) / threads2D.x),
        static_cast<unsigned int>((params.height + threads2D.y - 1) / threads2D.y));

    auto blur_plane_in_place = [&](float* plane, float* tmpBuf, const float* k, int radius) -> cudaError_t {
        if (!plane || !tmpBuf || !k || radius <= 0) {
            return cudaSuccess;
        }
        optics_blur_horizontal_kernel<<<blocks2D, threads2D, 0, stream>>>(plane, tmpBuf, params.width, params.height, k, radius);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            return e;
        }
        optics_blur_vertical_kernel<<<blocks2D, threads2D, 0, stream>>>(tmpBuf, plane, params.width, params.height, k, radius);
        return cudaGetLastError();
    };

    expose_film_raw_kernel<<<blocks2D, threads2D, 0, stream>>>(params, dRgbR, dRgbG, dRgbB);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }

    const JuicerCuda::HalationPayload& halation = params.halation;
    const JuicerCuda::HalationKernelPayload& halKernels = params.halationKernels;
    const bool doHalation = (halation.active != 0) &&
        (halation.strength[0] > 0.0f || halation.strength[1] > 0.0f || halation.strength[2] > 0.0f ||
         halation.scatteringStrength[0] > 0.0f || halation.scatteringStrength[1] > 0.0f || halation.scatteringStrength[2] > 0.0f);
    if (doHalation) {
        if (!dScratchBlurred) {
            return cudaErrorInvalidValue;
        }
        const int total = params.width * params.height;
        const int threads1D = 256;
        const int blocks1D = (total + threads1D - 1) / threads1D;

        auto apply_halation_pass = [&](float* plane, const float* k, int radius, float strength) -> cudaError_t {
            if (!plane || !k || radius <= 0 || !(strength > 0.0f)) {
                return cudaSuccess;
            }
            optics_blur_horizontal_kernel<<<blocks2D, threads2D, 0, stream>>>(plane, dTmp, params.width, params.height, k, radius);
            cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) {
                return e;
            }
            optics_blur_vertical_kernel<<<blocks2D, threads2D, 0, stream>>>(dTmp, dScratchBlurred, params.width, params.height, k, radius);
            e = cudaGetLastError();
            if (e != cudaSuccess) {
                return e;
            }
            halation_apply_kernel<<<blocks1D, threads1D, 0, stream>>>(plane, dScratchBlurred, total, strength);
            return cudaGetLastError();
        };

        err = apply_halation_pass(dRgbR, halKernels.halationKernel[0], halKernels.halationRadius[0], halation.strength[0]);
        if (err != cudaSuccess) return err;
        err = apply_halation_pass(dRgbG, halKernels.halationKernel[1], halKernels.halationRadius[1], halation.strength[1]);
        if (err != cudaSuccess) return err;
        err = apply_halation_pass(dRgbB, halKernels.halationKernel[2], halKernels.halationRadius[2], halation.strength[2]);
        if (err != cudaSuccess) return err;

        err = apply_halation_pass(dRgbR, halKernels.scatteringKernel[0], halKernels.scatteringRadius[0], halation.scatteringStrength[0]);
        if (err != cudaSuccess) return err;
        err = apply_halation_pass(dRgbG, halKernels.scatteringKernel[1], halKernels.scatteringRadius[1], halation.scatteringStrength[1]);
        if (err != cudaSuccess) return err;
        err = apply_halation_pass(dRgbB, halKernels.scatteringKernel[2], halKernels.scatteringRadius[2], halation.scatteringStrength[2]);
        if (err != cudaSuccess) return err;
    }

    develop_film_density_from_raw_kernel<<<blocks2D, threads2D, 0, stream>>>(params, dRgbR, dRgbG, dRgbB, dRgbR, dRgbG, dRgbB);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }

    const JuicerCuda::GrainPayload& grain = params.grain;
    const JuicerCuda::GrainKernelPayload& grainKernels = params.grainKernels;
    const bool doGrain = (grain.active != 0);
    if (doGrain) {
        const bool debugBreathing = (grain.breathingDebug != 0);
        const int total = params.width * params.height;
        const int threads1D = 256;
        const int blocks1D = (total + threads1D - 1) / threads1D;
        const bool useSublayers = (!debugBreathing && grain.sublayersActive != 0);
        const bool doFinalBlur = (!debugBreathing && grainKernels.blurKernel && grainKernels.blurRadius > 0);
        const bool needBlurScratch = (!debugBreathing) && doFinalBlur;
        if (needBlurScratch && !dScratchBlurred) {
            return cudaErrorInvalidValue;
        }

        auto process_channel_simple = [&](float* plane, int channel) -> cudaError_t {
            grain_apply_simple_kernel<<<blocks2D, threads2D, 0, stream>>>(params, plane, channel);
            cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) {
                return e;
            }
            return cudaSuccess;
        };

        auto process_channel_sublayers = [&](float* plane, int channel) -> cudaError_t {
            if (!dAux) {
                return cudaErrorInvalidValue;
            }
            const size_t bytes = static_cast<size_t>(total) * sizeof(float);
            cudaError_t e = cudaMemcpyAsync(dAux, plane, bytes, cudaMemcpyDeviceToDevice, stream);
            if (e != cudaSuccess) {
                return e;
            }

            grain_clear_kernel<<<blocks1D, threads1D, 0, stream>>>(plane, total);
            e = cudaGetLastError();
            if (e != cudaSuccess) {
                return e;
            }

            for (int sl = 0; sl < 3; ++sl) {
                grain_layer_kernel<<<blocks2D, threads2D, 0, stream>>>(params, dAux, dTmp, channel, sl);
                e = cudaGetLastError();
                if (e != cudaSuccess) {
                    return e;
                }
                const float* dyeKernel = grainKernels.dyeKernel[sl][channel];
                const int dyeRadius = grainKernels.dyeRadius[sl][channel];
                if (dyeKernel && dyeRadius > 0) {
                    if (!dScratchBlurred) {
                        return cudaErrorInvalidValue;
                    }
                    e = blur_plane_in_place(dTmp, dScratchBlurred, dyeKernel, dyeRadius);
                    if (e != cudaSuccess) {
                        return e;
                    }
                }
                grain_accumulate_kernel<<<blocks1D, threads1D, 0, stream>>>(plane, dTmp, total);
                e = cudaGetLastError();
                if (e != cudaSuccess) {
                    return e;
                }
            }

            return cudaSuccess;
        };

        if (useSublayers) {
            err = process_channel_sublayers(dRgbR, 0);
            if (err != cudaSuccess) return err;
            err = process_channel_sublayers(dRgbG, 1);
            if (err != cudaSuccess) return err;
            err = process_channel_sublayers(dRgbB, 2);
            if (err != cudaSuccess) return err;
        }
        else {
            err = process_channel_simple(dRgbR, 0);
            if (err != cudaSuccess) return err;
            err = process_channel_simple(dRgbG, 1);
            if (err != cudaSuccess) return err;
            err = process_channel_simple(dRgbB, 2);
            if (err != cudaSuccess) return err;
        }

        if (!debugBreathing) {
            auto apply_bias = [&](float* plane, int channel) -> cudaError_t {
                const float bias = -grain.densityMin[channel];
                if (std::isfinite(bias) && bias != 0.0f) {
                    grain_add_bias_kernel<<<blocks1D, threads1D, 0, stream>>>(plane, total, bias);
                    return cudaGetLastError();
                }
                return cudaSuccess;
            };

            err = apply_bias(dRgbR, 0);
            if (err != cudaSuccess) return err;
            err = apply_bias(dRgbG, 1);
            if (err != cudaSuccess) return err;
            err = apply_bias(dRgbB, 2);
            if (err != cudaSuccess) return err;

            if (doFinalBlur) {
                err = blur_plane_in_place(dRgbR, dTmp, grainKernels.blurKernel, grainKernels.blurRadius);
                if (err != cudaSuccess) return err;
                err = blur_plane_in_place(dRgbG, dTmp, grainKernels.blurKernel, grainKernels.blurRadius);
                if (err != cudaSuccess) return err;
                err = blur_plane_in_place(dRgbB, dTmp, grainKernels.blurKernel, grainKernels.blurRadius);
                if (err != cudaSuccess) return err;
            }
        }
    }

    if (params.printExpose.active) {
        develop_print_density_kernel<<<blocks2D, threads2D, 0, stream>>>(params, dRgbR, dRgbG, dRgbB);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }
    }

    const bool doGlare =
        std::isfinite(static_cast<double>(glarePercent)) && (glarePercent > 0.0f) &&
        std::isfinite(static_cast<double>(glareRoughness));
    if (doGlare) {
        if (glareRadius > 0 && dGlareKernel && !dScratchBlurred) {
            return cudaErrorInvalidValue;
        }
        const std::uint64_t mediumId = params.scanStage.scanTables.mediumIsNegative ? 0ULL : 1ULL;
        optics_glare_generate_kernel<<<blocks2D, threads2D, 0, stream>>>(
            dTmp,
            params.width,
            params.height,
            glareSeed,
            mediumId,
            glareOriginX,
            glareOriginY,
            glarePercent,
            glareRoughness);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }
        if (glareRadius > 0 && dGlareKernel) {
            err = blur_plane_in_place(dTmp, dScratchBlurred, dGlareKernel, glareRadius);
            if (err != cudaSuccess) return err;
        }
    }

    scan_linear_rgb_kernel<<<blocks2D, threads2D, 0, stream>>>(params, dRgbR, dRgbG, dRgbB, doGlare ? dTmp : nullptr, dRgbR, dRgbG, dRgbB);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }

    if (lensBlurRadius > 0 && dLensBlurKernel) {
        err = blur_plane_in_place(dRgbR, dTmp, dLensBlurKernel, lensBlurRadius);
        if (err != cudaSuccess) return err;
        err = blur_plane_in_place(dRgbG, dTmp, dLensBlurKernel, lensBlurRadius);
        if (err != cudaSuccess) return err;
        err = blur_plane_in_place(dRgbB, dTmp, dLensBlurKernel, lensBlurRadius);
        if (err != cudaSuccess) return err;
    }

    const bool doUnsharp =
        (unsharpRadius > 0) && dUnsharpKernel &&
        std::isfinite(static_cast<double>(unsharpAmount)) && (unsharpAmount != 0.0f);
    if (doUnsharp) {
        if (!dScratchBlurred) {
            return cudaErrorInvalidValue;
        }

        const int total = params.width * params.height;
        const int threads1D = 256;
        const int blocks1D = (total + threads1D - 1) / threads1D;

        auto unsharp_plane_in_place = [&](float* plane) -> cudaError_t {
            optics_blur_horizontal_kernel<<<blocks2D, threads2D, 0, stream>>>(plane, dTmp, params.width, params.height, dUnsharpKernel, unsharpRadius);
            cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) {
                return e;
            }
            optics_blur_vertical_kernel<<<blocks2D, threads2D, 0, stream>>>(dTmp, dScratchBlurred, params.width, params.height, dUnsharpKernel, unsharpRadius);
            e = cudaGetLastError();
            if (e != cudaSuccess) {
                return e;
            }
            optics_unsharp_combine_kernel<<<blocks1D, threads1D, 0, stream>>>(plane, dScratchBlurred, total, unsharpAmount);
            return cudaGetLastError();
        };

        err = unsharp_plane_in_place(dRgbR);
        if (err != cudaSuccess) return err;
        err = unsharp_plane_in_place(dRgbG);
        if (err != cudaSuccess) return err;
        err = unsharp_plane_in_place(dRgbB);
        if (err != cudaSuccess) return err;
    }

    scan_output_encode_kernel<<<blocks2D, threads2D, 0, stream>>>(params, dRgbR, dRgbG, dRgbB);
    return cudaGetLastError();
}

extern "C" cudaError_t juicer_cuda_print_pipeline(
    const JuicerCuda::PipelineRunParams* hParams,
    void* cudaStreamOpaque)
{
    if (!hParams) {
        return cudaErrorInvalidValue;
    }
    if (!hParams->printExpose.active) {
        return cudaErrorInvalidValue;
    }
    return juicer_cuda_negative_pipeline(hParams, cudaStreamOpaque);
}

extern "C" cudaError_t juicer_cuda_print_pipeline_optics(
    const JuicerCuda::PipelineRunParams* hParams,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    float* dScratchBlurred,
    float* dAux,
    const float* dLensBlurKernel,
    int lensBlurRadius,
    const float* dUnsharpKernel,
    int unsharpRadius,
    float unsharpAmount,
    int glareOriginX,
    int glareOriginY,
    std::uint64_t glareSeed,
    float glarePercent,
    float glareRoughness,
    const float* dGlareKernel,
    int glareRadius,
    void* cudaStreamOpaque)
{
    if (!hParams) {
        return cudaErrorInvalidValue;
    }
    if (!hParams->printExpose.active) {
        return cudaErrorInvalidValue;
    }
    return juicer_cuda_negative_pipeline_optics(
        hParams,
        dRgbR,
        dRgbG,
        dRgbB,
        dTmp,
        dScratchBlurred,
        dAux,
        dLensBlurKernel,
        lensBlurRadius,
        dUnsharpKernel,
        unsharpRadius,
        unsharpAmount,
        glareOriginX,
        glareOriginY,
        glareSeed,
        glarePercent,
        glareRoughness,
        dGlareKernel,
        glareRadius,
        cudaStreamOpaque);
}
