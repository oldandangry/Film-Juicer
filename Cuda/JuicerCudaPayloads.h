// Cuda/JuicerCudaPayloads.h
//
// Pipeline scaffolding: compact POD payloads for device kernels.
//
// Intentionally avoids CUDA headers so it can be included from both host and NVCC code.
#pragma once

#include <cstddef>
#include <cstdint>

namespace JuicerCuda {

#if !defined(JUICER_RESTRICT)
#if defined(_MSC_VER)
#define JUICER_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define JUICER_RESTRICT __restrict__
#else
#define JUICER_RESTRICT
#endif
#endif

    struct DeviceCurveView {
        const float* JUICER_RESTRICT x = nullptr;
        const float* JUICER_RESTRICT y = nullptr;
        int n = 0;
    };

    // Mirrors the subset of Spectral::FilmRawConfig needed by CUDA kernels.
    struct FilmRawPayload {
        int inputColorSpaceIndex = 0;
        int applyCctfDecoding = 0;
        int applyInputChromaticAdapt = 0;
        int spectralUpsamplingMode = 0; // 0=PreferHanatos, 1=ForceMallett (Spectral::SpectralUpsamplingMode)

        float inputRGBToXYZ[9] = {
            1,0,0,
            0,1,0,
            0,0,1
        };
        float inputXYZAdapt[9] = {
            1,0,0,
            0,1,0,
            0,0,1
        };

        float midgrayScale = 1.0f;
        float refIllumWhiteXYZ[3] = { 0.950455f, 1.0f, 1.089058f };
    };

    struct CctfPayload {
        // Matches GeneratedColorSpaces::CctfKind numeric values.
        // 0=Linear, 1=Gamma, 2=SRGB, 3=BT2020, 4=ProPhoto, 5=DaVinciIntermediate.
        int kind = 0;
        float gamma = 1.0f;
        float a = 0.0f;
        float b = 0.0f;
        float c = 0.0f;
        float d = 0.0f;
        float linearCutoff = 0.0f;
    };

    // Output encoding payload for device kernels (scanner path uses inputIsOutputSpace=1).
    struct OutputEncodingPayload {
        int outputColorSpaceIndex = 0;
        int applyCctfEncoding = 1;
        int preserveLinearRange = 0;
        int inputIsOutputSpace = 1;
        float dwgToOutput[9] = {
            1,0,0,
            0,1,0,
            0,0,1
        };
        CctfPayload cctf{};
    };

    struct DirPayload {
        int active = 0;
        float M[9] = {
            0,0,0,
            0,0,0,
            0,0,0
        };
        float highShift = 0.0f;
        float dMax[3] = { 1.0f, 1.0f, 1.0f };
    };

    // Scanner color runtime (CAT + XYZ->RGB) plus output encoding selection.
    struct ScanColorPayload {
        float cat02[9] = { 0.0f };
        float xyzToRgb[9] = { 0.0f };
        float illuminantXYZ[3] = { 0.0f, 0.0f, 0.0f };
        OutputEncodingPayload encoding{};
    };

    struct ScanTablesPayload {
        const float* JUICER_RESTRICT epsC = nullptr;
        const float* JUICER_RESTRICT epsM = nullptr;
        const float* JUICER_RESTRICT epsY = nullptr;
        const float* JUICER_RESTRICT Ax = nullptr;
        const float* JUICER_RESTRICT Ay = nullptr;
        const float* JUICER_RESTRICT Az = nullptr;
        const float* JUICER_RESTRICT baseMin = nullptr;
        int K = 0;
        int hasBaseline = 0;
        float invYn = 1.0f;
        int mediumIsNegative = 1;
        float min_cmy[3] = { 0.0f, 0.0f, 0.0f };
        float inv_max_cmy[3] = { 1.0f, 1.0f, 1.0f };
    };

    struct PipelineRunParams {
        const void* src = nullptr;
        std::size_t srcRowBytes = 0;
        void* dst = nullptr;
        std::size_t dstRowBytes = 0;
        int width = 0;
        int height = 0;
        int nComponents = 0;

        FilmRawPayload filmRaw{};
        float exposureScale = 1.0f;

        float gammaFactorB = 1.0f;
        float gammaFactorG = 1.0f;
        float gammaFactorR = 1.0f;

        int dirPrecorrected = 0;
        DirPayload dir{};
        int spatialDirActive = 0;
        const float* JUICER_RESTRICT spatialDirCorrY = nullptr;
        const float* JUICER_RESTRICT spatialDirCorrM = nullptr;
        const float* JUICER_RESTRICT spatialDirCorrC = nullptr;

        DeviceCurveView densB{};
        DeviceCurveView densG{};
        DeviceCurveView densR{};

        DeviceCurveView dirDensB{};
        DeviceCurveView dirDensG{};
        DeviceCurveView dirDensR{};

        DeviceCurveView sensB{};
        DeviceCurveView sensG{};
        DeviceCurveView sensR{};

        const float* JUICER_RESTRICT tablesAx = nullptr;
        const float* JUICER_RESTRICT tablesAy = nullptr;
        const float* JUICER_RESTRICT tablesAz = nullptr;
        int tablesK = 0;
        float spdSInv[9] = { 1,0,0, 0,1,0, 0,0,1 };

        const float* JUICER_RESTRICT hanatosLut = nullptr;
        int hanatosN = 0;
        const float* JUICER_RESTRICT hanatosLutIntegrated = nullptr;
        int hanatosNIntegrated = 0;

        int scannerUseLut = 0;
        const double* JUICER_RESTRICT scanLutLogXYZ = nullptr;
        int scanLutRes = 0;

        ScanTablesPayload scan{};
        ScanColorPayload scanColor{};
        int* scanErrorFlag = nullptr;

        // Print pipeline payloads (PrintBypass=false).
        // When printActive=1, kernels:
        //   negative density -> print raw exposure -> print density -> scan (print medium).
        // ScanTablesPayload 'scan' must be set to the selected scan medium (print), while
        // negTables must reference the negative dye extinction tables used to compute
        // negative transmitted light for the print exposure stage.
        int printActive = 0;
        ScanTablesPayload negTables{};

        // Precomputed enlarger illuminant filtered by dichroic Y/M/C at the current print params.
        const float* JUICER_RESTRICT printIllumFiltered = nullptr;
        int printIllumK = 0;

        // Print paper sensitivities (linear domain; pinned to the reference axis).
        DeviceCurveView printSensC{};
        DeviceCurveView printSensM{};
        DeviceCurveView printSensY{};

        // Print paper density curves (logE -> D) and per-channel gamma.
        DeviceCurveView printDcC{};
        DeviceCurveView printDcM{};
        DeviceCurveView printDcY{};
        float printGammaC = 1.0f;
        float printGammaM = 1.0f;
        float printGammaY = 1.0f;

        // Print exposure controls.
        float printExposure = 1.0f;
        float printPreflashExposure = 0.0f;
        float printMidgrayFactor = 1.0f;
        float printPreflashRaw[3] = { 0.0f, 0.0f, 0.0f };
    };

} // namespace JuicerCuda
