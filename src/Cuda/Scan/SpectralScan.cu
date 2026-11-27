#include "Cuda/Scan/SpectralScan.h"
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "Cuda/JuicerCudaDeviceHelpers.cuh"

#include "Cuda/Scan/JuicerCudaSpectralScan.cuh"
#include "GamutCompression.h"
#include "Cuda/Film/CaptureDensity.cuh"
#include "Cuda/Scan/OutputEncoding.cuh"
#include "Cuda/Scan/PrintGlare.h"

using JuicerCuda::FocusedScannerPostEffectOptions;

namespace {


    static_assert(std::is_standard_layout_v<JuicerCuda::ScannerSpectra>);
    static_assert(std::is_trivially_copyable_v<JuicerCuda::ScannerSpectra>);
    static_assert(sizeof(JuicerCuda::ScannerWavelength) == 28);
    static_assert(sizeof(JuicerCuda::ScannerSpectra) == 2272);
    static_assert(alignof(JuicerCuda::ScannerSpectra) == 4);
    static_assert(offsetof(JuicerCuda::ScannerSpectra, inverseYNormalization) == 2268);

    template <typename... Args>
    consteval std::size_t scan_parameter_bytes() {
        std::size_t bytes = 0;
        const auto add = [&bytes](std::size_t size, std::size_t alignment) {
            bytes = (bytes + alignment - 1) / alignment * alignment + size;
        };
        (add(sizeof(Args), alignof(Args)), ...);
        return bytes;
    }

    static_assert(scan_parameter_bytes<JuicerCuda::FilmScanRunParams, JuicerCuda::ScannerSpectra>() <= 4096);
    static_assert(scan_parameter_bytes<JuicerCuda::PrintPipelineRunParams, JuicerCuda::ScannerSpectra>() <= 4096);
    static_assert(scan_parameter_bytes<JuicerCuda::FilmScanRunParams, JuicerCuda::CameraFilmLinearExposurePlanes, JuicerCuda::ScannerSpectra>() <= 4096);
    static_assert(scan_parameter_bytes<JuicerCuda::PrintPipelineRunParams, JuicerCuda::CameraFilmLinearExposurePlanes, JuicerCuda::ScannerSpectra>() <= 4096);

    using JuicerCuda::apply_output_encoding_device;
    using JuicerCuda::compute_capture_film_density_device;

    struct OklabMatrices {
        float xyzToLms[9];
        float lmsToXyz[9];
        float lmsRootToLab[9];
        float labToLmsRoot[9];
    };

    __constant__ const OklabMatrices kOklabMatrices = []() constexpr {
        OklabMatrices matrices{};
        for (std::size_t i = 0; i < 9; ++i) {
            matrices.xyzToLms[i] = Gamut::kOklabXyzToLms[i];
            matrices.lmsToXyz[i] = Gamut::kOklabLmsToXyz[i];
            matrices.lmsRootToLab[i] = Gamut::kOklabLmsRootToLab[i];
            matrices.labToLmsRoot[i] = Gamut::kOklabLabToLmsRoot[i];
        }
        return matrices;
    }();

    __device__ __forceinline__ bool apply_output_gamut_compression_device(
        const JuicerCuda::ScanColorPayload& color,
        double rgb[3]) {
        if (!color.outputGamutActive) {
            return true;
        }
        if (!color.outputGamutCmax) {
            return false;
        }
        float compressed[3] = {
            static_cast<float>(rgb[0]),
            static_cast<float>(rgb[1]),
            static_cast<float>(rgb[2])};
        const Gamut::OutputCompressionView view{
            color.outputGamutCmax,
            color.outputGamutNativeRgbToD65Xyz,
            color.outputGamutD65XyzToNativeRgb,
            kOklabMatrices.xyzToLms,
            kOklabMatrices.lmsToXyz,
            kOklabMatrices.lmsRootToLab,
            kOklabMatrices.labToLmsRoot,
            {color.outputGamutLightnessKnee[0],
             color.outputGamutLightnessKnee[1],
             color.outputGamutLightnessKnee[2]},
            {color.outputGamutChromaKnee[0],
             color.outputGamutChromaKnee[1],
             color.outputGamutChromaKnee[2]}};
        Gamut::compress_output_oklch(view, compressed);
        if (!isfinite(compressed[0]) || !isfinite(compressed[1]) ||
            !isfinite(compressed[2])) {
            return false;
        }
        rgb[0] = static_cast<double>(compressed[0]);
        rgb[1] = static_cast<double>(compressed[1]);
        rgb[2] = static_cast<double>(compressed[2]);
        return true;
    }

    __device__ __forceinline__ void signal_scan_error_device(int* flag) {
        if (flag) {
            atomicOr(reinterpret_cast<unsigned int*>(flag), 1u);
        }
    }

    __device__ __forceinline__ bool narrow_scan_rgb_device(const double rgb[3], float scratch[3]) {
        scratch[0] = __double2float_rn(rgb[0]);
        scratch[1] = __double2float_rn(rgb[1]);
        scratch[2] = __double2float_rn(rgb[2]);
        return device_isfinite(scratch[0]) && device_isfinite(scratch[1]) && device_isfinite(scratch[2]);
    }

    __device__ __forceinline__ void mat3_mul_vec_double_device(const float m9[9], const double v3[3], double out3[3]) {
        out3[0] =
            static_cast<double>(m9[0]) * v3[0] +
            static_cast<double>(m9[1]) * v3[1] +
            static_cast<double>(m9[2]) * v3[2];
        out3[1] =
            static_cast<double>(m9[3]) * v3[0] +
            static_cast<double>(m9[4]) * v3[1] +
            static_cast<double>(m9[5]) * v3[2];
        out3[2] =
            static_cast<double>(m9[6]) * v3[0] +
            static_cast<double>(m9[7]) * v3[1] +
            static_cast<double>(m9[8]) * v3[2];
    }

    __device__ bool scanner_xyz_to_rgb(
        const JuicerCuda::ScanStagePayload& scan,
        const float* glarePercent,
        std::size_t index,
        const double (&inputXyz)[3],
        double (&rgb)[3]) {
        double xyz[3] = {inputXyz[0], inputXyz[1], inputXyz[2]};
        if (scan.correctionActive) {
            const double correctedY = fmin(fmax(
                                               static_cast<double>(scan.correctionSlope) * xyz[1] +
                                                   static_cast<double>(scan.correctionOffset),
                                               0.0),
                                           1.0);
            const double scale = correctedY / (xyz[1] + 1e-10);
            xyz[0] *= scale;
            xyz[1] *= scale;
            xyz[2] *= scale;
        }
        if (glarePercent) {
            const double glare = static_cast<double>(glarePercent[index]) * 0.01;
            xyz[0] += glare * static_cast<double>(scan.scanColor.illuminantXYZ[0]);
            xyz[1] += glare * static_cast<double>(scan.scanColor.illuminantXYZ[1]);
            xyz[2] += glare * static_cast<double>(scan.scanColor.illuminantXYZ[2]);
        }
        double adapted[3];
        mat3_mul_vec_double_device(scan.scanColor.cat02, xyz, adapted);
        mat3_mul_vec_double_device(scan.scanColor.xyzToRgb, adapted, rgb);
        return isfinite(rgb[0]) && isfinite(rgb[1]) && isfinite(rgb[2]) &&
               apply_output_gamut_compression_device(scan.scanColor, rgb);
    }

    template <bool FromCarrier, typename Params>
    __device__ __forceinline__ void run_scan_pixel(
        const Params& params,
        const JuicerCuda::ScannerSpectra& spectra,
        JuicerCuda::CameraFilmLinearExposurePlanes carrier) {
        const JuicerCuda::ScanStagePayload& scan = params.scanStage;
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

        float input[3]{};
        if constexpr (FromCarrier) {
            const auto carrierIndex = yIndex * carrier.rowStrideFloats + xIndex;
            input[0] = carrier.blueSensitive[carrierIndex];
            input[1] = carrier.greenSensitive[carrierIndex];
            input[2] = carrier.redSensitive[carrierIndex];
        } else {
            input[0] = srcPix[0];
            input[1] = srcPix[1];
            input[2] = srcPix[2];
        }

        float D_cmy[3] = {0.0f, 0.0f, 0.0f};
        if (!compute_capture_film_density_device<FromCarrier>(params, x, y, input, D_cmy)) {
            return;
        }

        const std::size_t idx = yIndex * static_cast<std::size_t>(params.width) + xIndex;

        if constexpr (requires { params.printExpose; params.printDevelop; }) {
            float* cyan = nullptr;
            apply_print_pipeline_device(params.printExpose, params.printDevelop, D_cmy, 1.0f, cyan);
        }

        double xyz[3];
        double rgbOut[3];
        const bool success = JuicerCuda::spectral_xyz(spectra, D_cmy, xyz);
        const bool writeScratch = scan.linearRgbR && scan.linearRgbG && scan.linearRgbB;
        float scratch[3];
        if (!success || !scanner_xyz_to_rgb(scan, scan.glarePercent, idx, xyz, rgbOut) ||
            (writeScratch && !narrow_scan_rgb_device(rgbOut, scratch))) {
            signal_scan_error_device(scan.scanErrorFlag);
            if (writeScratch) {
                scan.linearRgbR[idx] = 0.0F;
                scan.linearRgbG[idx] = 0.0F;
                scan.linearRgbB[idx] = 0.0F;
            } else {
                char* dstRow = reinterpret_cast<char*>(params.dst) + yIndex * params.dstRowBytes;
                float* dstPix = reinterpret_cast<float*>(dstRow + xIndex * pixelBytes);
                dstPix[0] = 0.0F;
                dstPix[1] = 0.0F;
                dstPix[2] = 0.0F;
                if (nC == 4) {
                    dstPix[3] = srcPix[3];
                }
            }
            return;
        }

        if (writeScratch) {
            scan.linearRgbR[idx] = scratch[0];
            scan.linearRgbG[idx] = scratch[1];
            scan.linearRgbB[idx] = scratch[2];
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

    template <typename Params>
    __global__ void pipeline_direct_kernel(
        Params params,
        const __grid_constant__ JuicerCuda::ScannerSpectra spectra) {
        run_scan_pixel<false>(params, spectra, {});
    }

    template <typename Params>
    __global__ void pipeline_carrier_kernel(
        Params params,
        JuicerCuda::CameraFilmLinearExposurePlanes carrier,
        const __grid_constant__ JuicerCuda::ScannerSpectra spectra) {
        run_scan_pixel<true>(params, spectra, carrier);
    }

    template <typename Params>
    cudaError_t launch_scan_pipeline(
        const Params& params,
        const JuicerCuda::ScannerSpectra& spectra,
        JuicerCuda::CameraFilmLinearExposurePlanes carrier,
        dim3 blocks,
        dim3 threads,
        cudaStream_t stream) {
        if (carrier.redSensitive || carrier.greenSensitive || carrier.blueSensitive) {
            if (!carrier.redSensitive || !carrier.greenSensitive || !carrier.blueSensitive ||
                carrier.rowStrideFloats < static_cast<std::size_t>(params.width)) {
                return cudaErrorInvalidValue;
            }
            pipeline_carrier_kernel<<<blocks, threads, 0, stream>>>(params, carrier, spectra);
        } else {
            pipeline_direct_kernel<<<blocks, threads, 0, stream>>>(params, spectra);
        }
        return cudaGetLastError();
    }

    struct ScanLinearDensityPlanes {
        const float* densityC = nullptr;
        const float* densityM = nullptr;
        const float* densityY = nullptr;
        const float* glarePercent = nullptr;
        const float* filmDustTransmittance = nullptr;
        float* outputR = nullptr;
        float* outputG = nullptr;
        float* outputB = nullptr;
    };

    static_assert(scan_parameter_bytes<JuicerCuda::ScanStagePayload, int, int, ScanLinearDensityPlanes, JuicerCuda::ScannerSpectra>() <= 4096);

    __global__ void scan_linear_density_rgb_kernel(
        JuicerCuda::ScanStagePayload scan,
        int widthPixels,
        int heightPixels,
        ScanLinearDensityPlanes planes,
        const __grid_constant__ JuicerCuda::ScannerSpectra spectra) {
        const float* inC = planes.densityC;
        const float* inM = planes.densityM;
        const float* inY = planes.densityY;
        const float* glarePercent = planes.glarePercent;
        float* outR = planes.outputR;
        float* outG = planes.outputG;
        float* outB = planes.outputB;
        if (!inC || !inM || !inY || !outR || !outG || !outB || widthPixels <= 0 || heightPixels <= 0) {
            return;
        }
        const std::size_t width = static_cast<std::size_t>(widthPixels);
        const std::size_t height = static_cast<std::size_t>(heightPixels);
        const std::size_t yStep = static_cast<std::size_t>(blockDim.y) * static_cast<std::size_t>(gridDim.y);
        const std::size_t xStep = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
        for (std::size_t y = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(blockDim.y) + static_cast<std::size_t>(threadIdx.y); y < height; y += yStep) {
            for (std::size_t x = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x); x < width; x += xStep) {
                const std::size_t idx = y * width + x;
                const float D_cmy[3] = {inC[idx], inM[idx], inY[idx]};

                double xyz[3];
                const bool success = JuicerCuda::spectral_xyz(spectra, D_cmy, xyz);
                if (success && planes.filmDustTransmittance) {
                    const double transmittance = planes.filmDustTransmittance[idx];
                    xyz[0] *= transmittance;
                    xyz[1] *= transmittance;
                    xyz[2] *= transmittance;
                }
                double rgbOut[3];
                float scratch[3];
                if (!success || !scanner_xyz_to_rgb(scan, glarePercent, idx, xyz, rgbOut) ||
                    !narrow_scan_rgb_device(rgbOut, scratch)) {
                    signal_scan_error_device(scan.scanErrorFlag);
                    outR[idx] = 0.0F;
                    outG[idx] = 0.0F;
                    outB[idx] = 0.0F;
                    continue;
                }

                outR[idx] = scratch[0];
                outG[idx] = scratch[1];
                outB[idx] = scratch[2];
            }
        }
    }

    // CUDA dimensions and C/M/Y planes follow the focused ABI.

} // namespace

extern "C" cudaError_t juicer_cuda_negative_direct_pipeline(
    const JuicerCuda::FilmScanRunParams* hParams,
    const JuicerCuda::ScannerSpectra& spectra,
    void* cudaStreamOpaque,
    JuicerCuda::CameraFilmLinearExposurePlanes carrier) {
    if (!hParams) {
        return cudaErrorInvalidValue;
    }

    const JuicerCuda::FilmScanRunParams params = *hParams;
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
        (params.width + threads.x - 1) / threads.x,
        (params.height + threads.y - 1) / threads.y);
    return launch_scan_pipeline(params, spectra, carrier, blocks, threads, stream);
}

struct FocusedRgbPlanes {
    float* r = nullptr;
    float* g = nullptr;
    float* b = nullptr;
};

template <typename Params>
cudaError_t launch_focused_scan_linear_rgb(
    const Params* hParams,
    const JuicerCuda::ScannerSpectra& spectra,
    FocusedRgbPlanes rgb,
    float* dTmp,
    float* dScratchBlurred,
    const FocusedScannerPostEffectOptions& options,
    void* cudaStreamOpaque,
    JuicerCuda::CameraFilmLinearExposurePlanes carrier) {
    if (!hParams || !hParams->src || !hParams->dst || !rgb.r || !rgb.g || !rgb.b) {
        return cudaErrorInvalidValue;
    }
    Params params = *hParams;
    if (params.width <= 0 || params.height <= 0) {
        return cudaSuccess;
    }
    if (!(params.nComponents == 3 || params.nComponents == 4) ||
        params.srcRowBytes == 0 || params.dstRowBytes == 0) {
        return cudaErrorInvalidValue;
    }
    cudaStream_t stream =
        cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
    dim3 threads(32, 8);
    dim3 blocks(
        static_cast<unsigned int>((params.width + threads.x - 1) / threads.x),
        static_cast<unsigned int>((params.height + threads.y - 1) / threads.y));

    const float* glarePercent = nullptr;
    const auto glareStatus = launch_print_glare(
        options, dTmp, dScratchBlurred, params.width, params.height, blocks, threads, stream, glarePercent);
    if (glareStatus != cudaSuccess) {
        return glareStatus;
    }

    params.scanStage.glarePercent = glarePercent;
    params.scanStage.linearRgbR = rgb.r;
    params.scanStage.linearRgbG = rgb.g;
    params.scanStage.linearRgbB = rgb.b;
    const auto status = launch_scan_pipeline(params, spectra, carrier, blocks, threads, stream);
    return status;
}

template <typename Params>
cudaError_t launch_focused_scan_linear_density_rgb(
    const Params* hParams,
    const JuicerCuda::ScannerSpectra& spectra,
    const float* dDensityC,
    const float* dDensityM,
    const float* dDensityY,
    FocusedRgbPlanes rgb,
    float* dTmp,
    float* dScratchBlurred,
    const FocusedScannerPostEffectOptions& options,
    const float* filmDustTransmittance,
    void* cudaStreamOpaque) {
    if (!hParams || !dDensityC || !dDensityM || !dDensityY || !rgb.r || !rgb.g || !rgb.b) {
        return cudaErrorInvalidValue;
    }
    const Params& focusedParams = *hParams;
    if (!focusedParams.src || !focusedParams.dst) {
        return cudaErrorInvalidValue;
    }
    if (focusedParams.width <= 0 || focusedParams.height <= 0) {
        return cudaSuccess;
    }
    if (!(focusedParams.nComponents == 3 || focusedParams.nComponents == 4) ||
        focusedParams.srcRowBytes == 0 || focusedParams.dstRowBytes == 0) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream =
        cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
    dim3 threads(32, 8);
    dim3 blocks(
        (focusedParams.width + threads.x - 1) / threads.x,
        (focusedParams.height + threads.y - 1) / threads.y);

    const float* glarePercent = nullptr;
    const auto glareStatus = launch_print_glare(
        options, dTmp, dScratchBlurred, focusedParams.width, focusedParams.height, blocks, threads, stream, glarePercent);
    if (glareStatus != cudaSuccess) {
        return glareStatus;
    }

    ScanLinearDensityPlanes scanPlanes{};
    scanPlanes.densityC = dDensityC;
    scanPlanes.densityM = dDensityM;
    scanPlanes.densityY = dDensityY;
    scanPlanes.glarePercent = glarePercent;
    scanPlanes.filmDustTransmittance = filmDustTransmittance;
    scanPlanes.outputR = rgb.r;
    scanPlanes.outputG = rgb.g;
    scanPlanes.outputB = rgb.b;
    scan_linear_density_rgb_kernel<<<blocks, threads, 0, stream>>>(
        focusedParams.scanStage, focusedParams.width, focusedParams.height, scanPlanes, spectra);
    const auto status = cudaGetLastError();
    return status;
}

extern "C" cudaError_t juicer_cuda_direct_focused_scan_linear_density_rgb(
    const JuicerCuda::FilmScanRunParams* hParams,
    const JuicerCuda::ScannerSpectra& spectra,
    const float* dDensityC,
    const float* dDensityM,
    const float* dDensityY,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    const float* filmDustTransmittance,
    void* cudaStreamOpaque) {
    return launch_focused_scan_linear_density_rgb(
        hParams,
        spectra,
        dDensityC,
        dDensityM,
        dDensityY,
        FocusedRgbPlanes{dRgbR, dRgbG, dRgbB},
        nullptr,
        nullptr,
        FocusedScannerPostEffectOptions{},
        filmDustTransmittance,
        cudaStreamOpaque);
}

extern "C" cudaError_t juicer_cuda_print_focused_scan_linear_density_rgb(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    const JuicerCuda::ScannerSpectra& spectra,
    const float* dDensityC,
    const float* dDensityM,
    const float* dDensityY,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    float* dScratchBlurred,
    int glareOriginX,
    int glareOriginY,
    std::uint64_t glareSeed,
    float glarePercent,
    float glareRoughness,
    const float* dGlareKernel,
    int glareRadius,
    void* cudaStreamOpaque) {
    return launch_focused_scan_linear_density_rgb(
        hParams,
        spectra,
        dDensityC,
        dDensityM,
        dDensityY,
        FocusedRgbPlanes{dRgbR, dRgbG, dRgbB},
        dTmp,
        dScratchBlurred,
        FocusedScannerPostEffectOptions{
            glareOriginX,
            glareOriginY,
            glareSeed,
            glarePercent,
            glareRoughness,
            dGlareKernel,
            glareRadius},
        nullptr,
        cudaStreamOpaque);
}

extern "C" cudaError_t juicer_cuda_direct_focused_scan_linear_rgb(
    const JuicerCuda::FilmScanRunParams* hParams,
    const JuicerCuda::ScannerSpectra& spectra,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    void* cudaStreamOpaque,
    JuicerCuda::CameraFilmLinearExposurePlanes carrier) {
    return launch_focused_scan_linear_rgb(
        hParams,
        spectra,
        FocusedRgbPlanes{dRgbR, dRgbG, dRgbB},
        nullptr,
        nullptr,
        FocusedScannerPostEffectOptions{},
        cudaStreamOpaque,
        carrier);
}

extern "C" cudaError_t juicer_cuda_print_focused_scan_linear_rgb(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    const JuicerCuda::ScannerSpectra& spectra,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    float* dScratchBlurred,
    int glareOriginX,
    int glareOriginY,
    std::uint64_t glareSeed,
    float glarePercent,
    float glareRoughness,
    const float* dGlareKernel,
    int glareRadius,
    void* cudaStreamOpaque,
    JuicerCuda::CameraFilmLinearExposurePlanes carrier) {
    return launch_focused_scan_linear_rgb(
        hParams,
        spectra,
        FocusedRgbPlanes{dRgbR, dRgbG, dRgbB},
        dTmp,
        dScratchBlurred,
        FocusedScannerPostEffectOptions{
            glareOriginX,
            glareOriginY,
            glareSeed,
            glarePercent,
            glareRoughness,
            dGlareKernel,
            glareRadius},
        cudaStreamOpaque,
        carrier);
}

extern "C" cudaError_t juicer_cuda_print_focused_pipeline(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    const JuicerCuda::ScannerSpectra& spectra,
    void* cudaStreamOpaque,
    JuicerCuda::CameraFilmLinearExposurePlanes carrier) {
    if (!hParams || !hParams->printExpose.active || !hParams->src || !hParams->dst) {
        return cudaErrorInvalidValue;
    }
    const JuicerCuda::PrintPipelineRunParams params = *hParams;
    if (params.width <= 0 || params.height <= 0) {
        return cudaSuccess;
    }
    if (!(params.nComponents == 3 || params.nComponents == 4) ||
        params.srcRowBytes == 0 || params.dstRowBytes == 0) {
        return cudaErrorInvalidValue;
    }
    cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
    dim3 threads(32, 8);
    dim3 blocks(
        (params.width + threads.x - 1) / threads.x,
        (params.height + threads.y - 1) / threads.y);
    return launch_scan_pipeline(params, spectra, carrier, blocks, threads, stream);
}
