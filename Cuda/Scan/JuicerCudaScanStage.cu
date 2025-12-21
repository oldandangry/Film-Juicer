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
__global__ void develop_film_density_kernel(
    JuicerCuda::Phase3RunParams params,
    float* outC,
    float* outM,
    float* outY);
__global__ void develop_print_density_kernel(
    JuicerCuda::Phase3RunParams params,
    float* ioC,
    float* ioM,
    float* ioY);

namespace {

    __global__ void phase3_negative_only_kernel(JuicerCuda::Phase3RunParams params) {
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
            params.spatialDirActive &&
            params.spatialDirCorrY && params.spatialDirCorrM && params.spatialDirCorrC;
        if (useSpatialDir) {
            const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
            const float corrY = params.spatialDirCorrY[idx];
            const float corrM = params.spatialDirCorrM[idx];
            const float corrC = params.spatialDirCorrC[idx];

            float logE_corr[3] = {
                logE_raw[0] - corrY,
                logE_raw[1] - corrM,
                logE_raw[2] - corrC
            };

            const JuicerCuda::DeviceCurveView cB = params.dirPrecorrected ? params.dirDensB : params.densB;
            const JuicerCuda::DeviceCurveView cG = params.dirPrecorrected ? params.dirDensG : params.densG;
            const JuicerCuda::DeviceCurveView cR = params.dirPrecorrected ? params.dirDensR : params.densR;

            logE_corr[0] = sanitize_inf_logE_for_curve_device(logE_corr[0], cB.x, cB.n);
            logE_corr[1] = sanitize_inf_logE_for_curve_device(logE_corr[1], cG.x, cG.n);
            logE_corr[2] = sanitize_inf_logE_for_curve_device(logE_corr[2], cR.x, cR.n);

            const float DY = sample_density_at_logE_device(cB.x, cB.y, cB.n, logE_corr[0], params.gammaFactorB);
            const float DM = sample_density_at_logE_device(cG.x, cG.y, cG.n, logE_corr[1], params.gammaFactorG);
            const float DC = sample_density_at_logE_device(cR.x, cR.y, cR.n, logE_corr[2], params.gammaFactorR);

            D_cmy[0] = DC;
            D_cmy[1] = DM;
            D_cmy[2] = DY;
        }
        else if (params.dir.active) {
            float logE_corr[3] = { logE_sanitized[0], logE_sanitized[1], logE_sanitized[2] };
            apply_dir_runtime_logE_device(logE_corr, layerPre, params.dir, params.densB, params.densG, params.densR);

            const JuicerCuda::DeviceCurveView cB = params.dirPrecorrected ? params.dirDensB : params.densB;
            const JuicerCuda::DeviceCurveView cG = params.dirPrecorrected ? params.dirDensG : params.densG;
            const JuicerCuda::DeviceCurveView cR = params.dirPrecorrected ? params.dirDensR : params.densR;

            const float DY = sample_density_at_logE_device(cB.x, cB.y, cB.n, logE_corr[0], params.gammaFactorB);
            const float DM = sample_density_at_logE_device(cG.x, cG.y, cG.n, logE_corr[1], params.gammaFactorG);
            const float DC = sample_density_at_logE_device(cR.x, cR.y, cR.n, logE_corr[2], params.gammaFactorR);

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

        apply_print_pipeline_device(params, D_cmy);

        // Scan: normalize density -> logXYZ
        double D_norm[3];
        if (params.scan.mediumIsNegative) {
            D_norm[0] = (static_cast<double>(D_cmy[0]) + static_cast<double>(params.scan.min_cmy[0])) * static_cast<double>(params.scan.inv_max_cmy[0]);
            D_norm[1] = (static_cast<double>(D_cmy[1]) + static_cast<double>(params.scan.min_cmy[1])) * static_cast<double>(params.scan.inv_max_cmy[1]);
            D_norm[2] = (static_cast<double>(D_cmy[2]) + static_cast<double>(params.scan.min_cmy[2])) * static_cast<double>(params.scan.inv_max_cmy[2]);
        }
        else {
            D_norm[0] = static_cast<double>(D_cmy[0]) * static_cast<double>(params.scan.inv_max_cmy[0]);
            D_norm[1] = static_cast<double>(D_cmy[1]) * static_cast<double>(params.scan.inv_max_cmy[1]);
            D_norm[2] = static_cast<double>(D_cmy[2]) * static_cast<double>(params.scan.inv_max_cmy[2]);
        }

        double logXYZ[3] = { 0.0, 0.0, 0.0 };
        scan_log_xyz_device(params, D_norm, logXYZ);
        const double xyz[3] = {
            pow(10.0, logXYZ[0]),
            pow(10.0, logXYZ[1]),
            pow(10.0, logXYZ[2])
        };

        double adapted[3];
        mat3_mul_vec_double_device(params.scanColor.cat02, xyz, adapted);
        double rgbOut[3];
        mat3_mul_vec_double_device(params.scanColor.xyzToRgb, adapted, rgbOut);

        if (!isfinite(rgbOut[0]) || !isfinite(rgbOut[1]) || !isfinite(rgbOut[2])) {
            signal_scan_error_device(params.scanErrorFlag);
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
        apply_output_encoding_device(params.scanColor.encoding, rgbOut);

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
        JuicerCuda::Phase3RunParams params,
        const float* inC,
        const float* inM,
        const float* inY,
        const float* glarePercent,
        float* outR,
        float* outG,
        float* outB)
    {
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
        if (params.scan.mediumIsNegative) {
            D_norm[0] = (static_cast<double>(D_cmy[0]) + static_cast<double>(params.scan.min_cmy[0])) * static_cast<double>(params.scan.inv_max_cmy[0]);
            D_norm[1] = (static_cast<double>(D_cmy[1]) + static_cast<double>(params.scan.min_cmy[1])) * static_cast<double>(params.scan.inv_max_cmy[1]);
            D_norm[2] = (static_cast<double>(D_cmy[2]) + static_cast<double>(params.scan.min_cmy[2])) * static_cast<double>(params.scan.inv_max_cmy[2]);
        }
        else {
            D_norm[0] = static_cast<double>(D_cmy[0]) * static_cast<double>(params.scan.inv_max_cmy[0]);
            D_norm[1] = static_cast<double>(D_cmy[1]) * static_cast<double>(params.scan.inv_max_cmy[1]);
            D_norm[2] = static_cast<double>(D_cmy[2]) * static_cast<double>(params.scan.inv_max_cmy[2]);
        }

        double logXYZ[3] = { 0.0, 0.0, 0.0 };
        scan_log_xyz_device(params, D_norm, logXYZ);

        double xyz[3] = {
            pow(10.0, logXYZ[0]),
            pow(10.0, logXYZ[1]),
            pow(10.0, logXYZ[2])
        };

        if (glarePercent) {
            const double glare = static_cast<double>(glarePercent[idx]) * 0.01;
            xyz[0] += glare * static_cast<double>(params.scanColor.illuminantXYZ[0]);
            xyz[1] += glare * static_cast<double>(params.scanColor.illuminantXYZ[1]);
            xyz[2] += glare * static_cast<double>(params.scanColor.illuminantXYZ[2]);
        }

        double adapted[3];
        mat3_mul_vec_double_device(params.scanColor.cat02, xyz, adapted);
        double rgbOut[3];
        mat3_mul_vec_double_device(params.scanColor.xyzToRgb, adapted, rgbOut);

        if (!isfinite(rgbOut[0]) || !isfinite(rgbOut[1]) || !isfinite(rgbOut[2])) {
            signal_scan_error_device(params.scanErrorFlag);
            outR[idx] = 0.0f;
            outG[idx] = 0.0f;
            outB[idx] = 0.0f;
            return;
        }

        outR[idx] = static_cast<float>(rgbOut[0]);
        outG[idx] = static_cast<float>(rgbOut[1]);
        outB[idx] = static_cast<float>(rgbOut[2]);
    }

    __global__ void phase3_stageD_encode_write_kernel(
        JuicerCuda::Phase3RunParams params,
        const float* rgbR,
        const float* rgbG,
        const float* rgbB)
    {
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
        apply_output_encoding_device(params.scanColor.encoding, rgbOut);

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

extern "C" cudaError_t juicer_cuda_phase3_negative_only(
    const JuicerCuda::Phase3RunParams* hParams,
    void* cudaStreamOpaque)
{
    if (!hParams) {
        return cudaErrorInvalidValue;
    }

    const JuicerCuda::Phase3RunParams params = *hParams;
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
    phase3_negative_only_kernel<<<blocks, threads, 0, stream>>>(params);
    return cudaGetLastError();
}

extern "C" cudaError_t juicer_cuda_phase3_negative_only_optics(
    const JuicerCuda::Phase3RunParams* hParams,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    float* dScratchBlurred,
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

    const JuicerCuda::Phase3RunParams params = *hParams;
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

    const bool doGlare =
        std::isfinite(static_cast<double>(glarePercent)) && (glarePercent > 0.0f) &&
        std::isfinite(static_cast<double>(glareRoughness));
    if (doGlare) {
        const std::uint64_t mediumId = params.scan.mediumIsNegative ? 0ULL : 1ULL;
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
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }
        if (glareRadius > 0 && dGlareKernel) {
            err = blur_plane_in_place(dTmp, dRgbR, dGlareKernel, glareRadius);
            if (err != cudaSuccess) return err;
        }
    }

    develop_film_density_kernel<<<blocks2D, threads2D, 0, stream>>>(params, dRgbR, dRgbG, dRgbB);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }

    if (params.printActive) {
        develop_print_density_kernel<<<blocks2D, threads2D, 0, stream>>>(params, dRgbR, dRgbG, dRgbB);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
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

    phase3_stageD_encode_write_kernel<<<blocks2D, threads2D, 0, stream>>>(params, dRgbR, dRgbG, dRgbB);
    return cudaGetLastError();
}

extern "C" cudaError_t juicer_cuda_phase5_print_pipeline(
    const JuicerCuda::Phase3RunParams* hParams,
    void* cudaStreamOpaque)
{
    if (!hParams) {
        return cudaErrorInvalidValue;
    }
    if (!hParams->printActive) {
        return cudaErrorInvalidValue;
    }
    return juicer_cuda_phase3_negative_only(hParams, cudaStreamOpaque);
}

extern "C" cudaError_t juicer_cuda_phase5_print_pipeline_optics(
    const JuicerCuda::Phase3RunParams* hParams,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    float* dScratchBlurred,
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
    if (!hParams->printActive) {
        return cudaErrorInvalidValue;
    }
    return juicer_cuda_phase3_negative_only_optics(
        hParams,
        dRgbR,
        dRgbG,
        dRgbB,
        dTmp,
        dScratchBlurred,
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
