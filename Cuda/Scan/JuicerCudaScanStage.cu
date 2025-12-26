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

namespace {

    __device__ __forceinline__ std::uint64_t splitmix64_device(std::uint64_t x) {
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    }

    __device__ __forceinline__ float hash_to_unit_device(std::uint64_t h) {
        constexpr float kInv = 1.0f / 16777216.0f; // 2^24
        return static_cast<float>((h >> 40) & 0xFFFFFFu) * kInv;
    }

    __device__ __forceinline__ float smoothstep_device(float edge0, float edge1, float x) {
        if (edge1 <= edge0) {
            return (x < edge0) ? 0.0f : 1.0f;
        }
        float t = (x - edge0) / (edge1 - edge0);
        t = fminf(fmaxf(t, 0.0f), 1.0f);
        return t * t * (3.0f - 2.0f * t);
    }

    __device__ __forceinline__ float dust_mask_device(
        float amount,
        float x,
        float y,
        float pixelSizeUm,
        std::uint64_t seed,
        float cellPx,
        float baseProb,
        float sizeUm,
        float strength)
    {
        if (!(amount > 0.0f) || !(pixelSizeUm > 0.0f) || !(cellPx > 0.0f)) {
            return 0.0f;
        }
        const int cellX = static_cast<int>(floorf(x / cellPx));
        const int cellY = static_cast<int>(floorf(y / cellPx));
        std::uint64_t h = splitmix64_device(seed ^
            (static_cast<std::uint64_t>(cellX) * 0x8EBC6AF09C88C6E3ULL) ^
            (static_cast<std::uint64_t>(cellY) * 0x9E3779B97F4A7C15ULL));
        const float u = hash_to_unit_device(h);
        const float p = baseProb * amount;
        if (u >= p) {
            return 0.0f;
        }
        const float u1 = hash_to_unit_device(h ^ 0xBF58476D1CE4E5B9ULL);
        const float u2 = hash_to_unit_device(h ^ 0x94D049BB133111EBULL);
        const float u3 = hash_to_unit_device(h ^ 0xD6E8FEB86659FD93ULL);
        const float u4 = hash_to_unit_device(h ^ 0xA5A5A5A5A5A5A5A5ULL);
        const float cx = (static_cast<float>(cellX) + u1) * cellPx;
        const float cy = (static_cast<float>(cellY) + u2) * cellPx;
        const float baseRadius = fmaxf(0.5f, sizeUm / pixelSizeUm);
        const float radius = baseRadius * (0.5f + 1.2f * u3);
        const float dx = x - cx;
        const float dy = y - cy;
        const float dist = sqrtf(dx * dx + dy * dy);
        const float edge = fmaxf(0.5f, radius * 0.6f);
        const float mask = 1.0f - smoothstep_device(radius, radius + edge, dist);
        const float intensity = strength * amount * (0.5f + 0.5f * u4);
        return mask * intensity;
    }

    __device__ __forceinline__ float scratch_mask_device(
        float amount,
        float x,
        float y,
        float pixelSizeUm,
        std::uint64_t seed,
        float cellPxX,
        float cellPxY,
        float baseProb,
        float widthUm,
        float strength,
        float maxAngleRad)
    {
        if (!(amount > 0.0f) || !(pixelSizeUm > 0.0f) || !(cellPxX > 0.0f) || !(cellPxY > 0.0f)) {
            return 0.0f;
        }
        const int cellX = static_cast<int>(floorf(x / cellPxX));
        const int cellY = static_cast<int>(floorf(y / cellPxY));
        std::uint64_t h = splitmix64_device(seed ^
            (static_cast<std::uint64_t>(cellX) * 0xC6A4A7935BD1E995ULL) ^
            (static_cast<std::uint64_t>(cellY) * 0xD2B74407B1CE6E93ULL));
        const float u = hash_to_unit_device(h);
        const float p = baseProb * amount;
        if (u >= p) {
            return 0.0f;
        }
        const float u1 = hash_to_unit_device(h ^ 0xBF58476D1CE4E5B9ULL);
        const float u2 = hash_to_unit_device(h ^ 0x94D049BB133111EBULL);
        const float u3 = hash_to_unit_device(h ^ 0xA5A5A5A5A5A5A5A5ULL);
        const float u4 = hash_to_unit_device(h ^ 0xD6E8FEB86659FD93ULL);
        const float u5 = hash_to_unit_device(h ^ 0x9E3779B97F4A7C15ULL);
        const float cx = (static_cast<float>(cellX) + u1) * cellPxX;
        const float cy = (static_cast<float>(cellY) + u2) * cellPxY;
        const float baseWidth = fmaxf(0.5f, widthUm / pixelSizeUm);
        const float width = baseWidth * (0.6f + 1.4f * u3);
        const float halfLen = cellPxY * (0.35f + 0.4f * u4);
        const float angle = (u5 * 2.0f - 1.0f) * maxAngleRad;
        const float c = cosf(angle);
        const float s = sinf(angle);
        const float dx = x - cx;
        const float dy = y - cy;
        const float dist = fabsf(dx * c - dy * s);
        const float along = fabsf(dx * s + dy * c);
        if (along > halfLen) {
            return 0.0f;
        }
        const float edge = fmaxf(0.5f, width * 0.8f);
        const float mask = 1.0f - smoothstep_device(width, width + edge, dist);
        const float intensity = strength * amount * (0.5f + 0.5f * u2);
        return mask * intensity;
    }

} // namespace

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

    __device__ __forceinline__ int clamp_index_device(int idx, int maxIndex) {
        if (idx < 0) return 0;
        if (idx > maxIndex) return maxIndex;
        return idx;
    }

    __device__ __forceinline__ float sample_plane_mitchell_device(
        const float* JUICER_RESTRICT plane,
        int width,
        int height,
        float x,
        float y)
    {
        if (!plane || width <= 0 || height <= 0) {
            return 0.0f;
        }

        const float maxX = static_cast<float>(width - 1);
        const float maxY = static_cast<float>(height - 1);
        x = fminf(fmaxf(x, 0.0f), maxX);
        y = fminf(fmaxf(y, 0.0f), maxY);

        const int ix = static_cast<int>(floorf(x));
        const int iy = static_cast<int>(floorf(y));
        const double tx = static_cast<double>(x - static_cast<float>(ix));
        const double ty = static_cast<double>(y - static_cast<float>(iy));

        double wx[4];
        double wy[4];
        wx[0] = mitchell_weight_device(tx + 1.0);
        wx[1] = mitchell_weight_device(tx);
        wx[2] = mitchell_weight_device(tx - 1.0);
        wx[3] = mitchell_weight_device(tx - 2.0);
        wy[0] = mitchell_weight_device(ty + 1.0);
        wy[1] = mitchell_weight_device(ty);
        wy[2] = mitchell_weight_device(ty - 1.0);
        wy[3] = mitchell_weight_device(ty - 2.0);

        const int maxXi = width - 1;
        const int maxYi = height - 1;
        double acc = 0.0;
        for (int j = 0; j < 4; ++j) {
            const int sy = clamp_index_device(iy + j - 1, maxYi);
            const double wyj = wy[j];
            const std::size_t row = static_cast<std::size_t>(sy) * static_cast<std::size_t>(width);
            for (int i = 0; i < 4; ++i) {
                const int sx = clamp_index_device(ix + i - 1, maxXi);
                acc += wyj * wx[i] * static_cast<double>(plane[row + static_cast<std::size_t>(sx)]);
            }
        }
        return static_cast<float>(acc);
    }

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

    __global__ void apply_film_defects_kernel(
        JuicerCuda::PipelineRunParams params,
        float* ioC,
        float* ioM,
        float* ioY)
    {
        const JuicerCuda::GrainPayload& grain = params.grain;
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= params.width || y >= params.height) {
            return;
        }
        if (!ioC || !ioM || !ioY) {
            return;
        }
        const float dustAmount = grain.filmDustAmount;
        const float scratchAmount = grain.filmScratchAmount;
        if (!(dustAmount > 0.0f) && !(scratchAmount > 0.0f)) {
            return;
        }
        if (!(grain.pixelSizeUm > 0.0f)) {
            return;
        }

        const std::uint64_t seedBase = (grain.stbnSessionSeed != 0) ? grain.stbnSessionSeed : 1ULL;
        const std::uint64_t seedDust = splitmix64_device(seedBase ^ 0xF0D0C0B0A0908071ULL);
        const std::uint64_t seedScratch = splitmix64_device(seedBase ^ 0x8EBC6AF09C88C6E3ULL);

        const float time = static_cast<float>(grain.frameIndex) + grain.timeAlpha;
        const float rollPx = (grain.pitchPx > 0) ? static_cast<float>(grain.pitchPx) : 0.0f;

        const float absX = static_cast<float>(grain.originX + x);
        const float absY = static_cast<float>(grain.originY + y);
        const float rollY = absY + rollPx * time;

        constexpr float kDustCellPx = 64.0f;
        constexpr float kDustBaseProb = 0.02f;
        constexpr float kDustSizeUm = 25.0f;
        constexpr float kDustStrength = 0.45f;

        constexpr float kScratchCellPxX = 512.0f;
        constexpr float kScratchCellPxY = 1024.0f;
        constexpr float kScratchBaseProb = 0.01f;
        constexpr float kScratchWidthUm = 15.0f;
        constexpr float kScratchStrength = 0.35f;
        constexpr float kScratchMaxAngle = 0.08726646f; // 5 deg

        const float dustMask = dust_mask_device(
            dustAmount, absX, rollY, grain.pixelSizeUm, seedDust,
            kDustCellPx, kDustBaseProb, kDustSizeUm, kDustStrength);
        const float scratchMask = scratch_mask_device(
            scratchAmount, absX, rollY, grain.pixelSizeUm, seedScratch,
            kScratchCellPxX, kScratchCellPxY, kScratchBaseProb, kScratchWidthUm, kScratchStrength, kScratchMaxAngle);
        float delta = dustMask + scratchMask;
        if (!device_isfinite(delta)) {
            delta = 0.0f;
        }
        if (delta != 0.0f) {
            const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
            ioC[idx] = ioC[idx] + delta;
            ioM[idx] = ioM[idx] + delta;
            ioY[idx] = ioY[idx] + delta;
        }
    }

    __global__ void scan_output_encode_kernel(
        JuicerCuda::PipelineRunParams params,
        const float* rgbR,
        const float* rgbG,
        const float* rgbB)
    {
        const JuicerCuda::ScanStagePayload& scan = params.scanStage;
        const JuicerCuda::GateWeavePayload& weave = params.gateWeave;
        const JuicerCuda::GrainPayload& grain = params.grain;
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

        const int debugView = grain.debugView;
        const float time = static_cast<float>(grain.frameIndex) + grain.timeAlpha;
        const float rollPx = (grain.pitchPx > 0) ? static_cast<float>(grain.pitchPx) : 0.0f;
        const float absX = static_cast<float>(grain.originX + x);
        const float absY = static_cast<float>(grain.originY + y);
        const float rollY = absY + rollPx * time;
        const std::uint64_t seedBase = (grain.stbnSessionSeed != 0) ? grain.stbnSessionSeed : 1ULL;
        const std::uint64_t seedFilmDust = splitmix64_device(seedBase ^ 0xF0D0C0B0A0908071ULL);
        const std::uint64_t seedGateDust = splitmix64_device(seedBase ^ 0xA1B2C3D4E5F60718ULL);
        const std::uint64_t seedFilmScratch = splitmix64_device(seedBase ^ 0x8EBC6AF09C88C6E3ULL);
        const std::uint64_t seedGateScratch = splitmix64_device(seedBase ^ 0xC6A4A7935BD1E995ULL);
        double rgbOut[3];
        if (debugView == 5 || debugView == 6) {
            constexpr float kDustCellPx = 64.0f;
            constexpr float kGateDustCellPx = 96.0f;
            constexpr float kDustBaseProb = 0.02f;
            constexpr float kGateDustBaseProb = 0.01f;
            constexpr float kDustSizeUm = 25.0f;
            constexpr float kGateDustSizeUm = 28.0f;
            constexpr float kDustStrength = 0.45f;
            constexpr float kGateDustStrength = 0.35f;

            constexpr float kScratchCellPxX = 512.0f;
            constexpr float kScratchCellPxY = 1024.0f;
            constexpr float kGateScratchCellPxX = 512.0f;
            constexpr float kGateScratchCellPxY = 512.0f;
            constexpr float kScratchBaseProb = 0.01f;
            constexpr float kGateScratchBaseProb = 0.008f;
            constexpr float kScratchWidthUm = 15.0f;
            constexpr float kGateScratchWidthUm = 12.0f;
            constexpr float kScratchStrength = 0.35f;
            constexpr float kGateScratchStrength = 0.30f;
            constexpr float kScratchMaxAngle = 0.08726646f;

            float filmMask = 0.0f;
            float gateMask = 0.0f;
            if (debugView == 5) {
                filmMask = dust_mask_device(grain.filmDustAmount, absX, rollY, grain.pixelSizeUm, seedFilmDust,
                    kDustCellPx, kDustBaseProb, kDustSizeUm, kDustStrength);
                gateMask = dust_mask_device(grain.gateDustAmount, absX, absY, grain.pixelSizeUm, seedGateDust,
                    kGateDustCellPx, kGateDustBaseProb, kGateDustSizeUm, kGateDustStrength);
            }
            else {
                filmMask = scratch_mask_device(grain.filmScratchAmount, absX, rollY, grain.pixelSizeUm, seedFilmScratch,
                    kScratchCellPxX, kScratchCellPxY, kScratchBaseProb, kScratchWidthUm, kScratchStrength, kScratchMaxAngle);
                gateMask = scratch_mask_device(grain.gateScratchAmount, absX, absY, grain.pixelSizeUm, seedGateScratch,
                    kGateScratchCellPxX, kGateScratchCellPxY, kGateScratchBaseProb, kGateScratchWidthUm, kGateScratchStrength, kScratchMaxAngle);
            }
            float mask = filmMask + gateMask;
            if (!device_isfinite(mask)) {
                mask = 0.0f;
            }
            mask = fminf(fmaxf(mask, 0.0f), 1.0f);
            rgbOut[0] = static_cast<double>(mask);
            rgbOut[1] = static_cast<double>(mask);
            rgbOut[2] = static_cast<double>(mask);
        }
        else if (debugView == 3) {
            const float scale = (weave.debugScalePx > 1e-6f) ? weave.debugScalePx : 1.0f;
            float r = 0.5f + 0.5f * (weave.dxPx / scale);
            float g = 0.5f + 0.5f * (weave.dyPx / scale);
            r = fminf(fmaxf(r, 0.0f), 1.0f);
            g = fminf(fmaxf(g, 0.0f), 1.0f);
            rgbOut[0] = static_cast<double>(r);
            rgbOut[1] = static_cast<double>(g);
            rgbOut[2] = 0.5;
        }
        else if (weave.active != 0) {
            const float cx = 0.5f * static_cast<float>(params.width - 1);
            const float cy = 0.5f * static_cast<float>(params.height - 1);
            const float fx = static_cast<float>(x) - cx;
            const float fy = static_cast<float>(y) - cy;
            const float c = weave.cosRot;
            const float s = weave.sinRot;
            const float srcX = c * fx - s * fy + cx + weave.dxPx;
            const float srcY = s * fx + c * fy + cy + weave.dyPx;
            rgbOut[0] = static_cast<double>(sample_plane_mitchell_device(rgbR, params.width, params.height, srcX, srcY));
            rgbOut[1] = static_cast<double>(sample_plane_mitchell_device(rgbG, params.width, params.height, srcX, srcY));
            rgbOut[2] = static_cast<double>(sample_plane_mitchell_device(rgbB, params.width, params.height, srcX, srcY));
        }
        else {
            const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
            rgbOut[0] = static_cast<double>(rgbR[idx]);
            rgbOut[1] = static_cast<double>(rgbG[idx]);
            rgbOut[2] = static_cast<double>(rgbB[idx]);
        }
        if (debugView == 0) {
            const float gateDust = dust_mask_device(grain.gateDustAmount, absX, absY, grain.pixelSizeUm, seedGateDust,
                96.0f, 0.01f, 28.0f, 0.35f);
            const float gateScratch = scratch_mask_device(grain.gateScratchAmount, absX, absY, grain.pixelSizeUm, seedGateScratch,
                512.0f, 512.0f, 0.008f, 12.0f, 0.30f, 0.08726646f);
            float gateMask = gateDust + gateScratch;
            if (device_isfinite(gateMask) && gateMask > 0.0f) {
                gateMask = fminf(gateMask, 0.95f);
                const float trans = 1.0f - gateMask;
                rgbOut[0] *= static_cast<double>(trans);
                rgbOut[1] *= static_cast<double>(trans);
                rgbOut[2] *= static_cast<double>(trans);
            }
            apply_output_encoding_device(scan.scanColor.encoding, rgbOut);
        }

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
        const int debugView = grain.debugView;
        const bool debugAny = (grain.breathingDebug != 0) || (debugView != 0);
        const int total = params.width * params.height;
        const int threads1D = 256;
        const int blocks1D = (total + threads1D - 1) / threads1D;
        const bool useSublayers = (!debugAny && grain.sublayersActive != 0);
        const bool doFinalBlur = (!debugAny && grainKernels.blurKernel && grainKernels.blurRadius > 0);
        const bool needBlurScratch = (!debugAny) && doFinalBlur;
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

        if (!debugAny) {
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

    const bool doFilmDefects = (grain.filmDustAmount > 0.0f) || (grain.filmScratchAmount > 0.0f);
    if (grain.debugView == 0 && doFilmDefects) {
        apply_film_defects_kernel<<<blocks2D, threads2D, 0, stream>>>(params, dRgbR, dRgbG, dRgbB);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }
    }

    if (grain.debugView != 0) {
        scan_output_encode_kernel<<<blocks2D, threads2D, 0, stream>>>(params, dRgbR, dRgbG, dRgbB);
        return cudaGetLastError();
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
