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

    struct SpatialDirPayload {
        int active = 0;
        const float* JUICER_RESTRICT corrY = nullptr;
        const float* JUICER_RESTRICT corrM = nullptr;
        const float* JUICER_RESTRICT corrC = nullptr;
    };

    struct FilmExposurePayload {
        float exposureScale = 1.0f;
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
    };

    struct FilmDevelopPayload {
        float gammaFactorB = 1.0f;
        float gammaFactorG = 1.0f;
        float gammaFactorR = 1.0f;
        int dirPrecorrected = 0;
        DirPayload dir{};
        SpatialDirPayload spatialDir{};
        DeviceCurveView densB{};
        DeviceCurveView densG{};
        DeviceCurveView densR{};
        DeviceCurveView dirDensB{};
        DeviceCurveView dirDensG{};
        DeviceCurveView dirDensR{};
    };

    struct PrintExposePayload {
        int active = 0;
        ScanTablesPayload negTables{};
        const float* JUICER_RESTRICT printIllumFiltered = nullptr;
        int printIllumK = 0;
        DeviceCurveView printSensC{};
        DeviceCurveView printSensM{};
        DeviceCurveView printSensY{};
        float printExposure = 1.0f;
        float printPreflashExposure = 0.0f;
        float printMidgrayFactor = 1.0f;
        float printPreflashRaw[3] = { 0.0f, 0.0f, 0.0f };
    };

    struct PrintDevelopPayload {
        DeviceCurveView printDcC{};
        DeviceCurveView printDcM{};
        DeviceCurveView printDcY{};
        float printGammaC = 1.0f;
        float printGammaM = 1.0f;
        float printGammaY = 1.0f;
    };

    struct ScanStagePayload {
        int scannerUseLut = 0;
        const double* JUICER_RESTRICT scanLutLogXYZ = nullptr;
        int scanLutRes = 0;
        ScanTablesPayload scanTables{};
        ScanColorPayload scanColor{};
        int* scanErrorFlag = nullptr;
    };

    struct ScannerOpticsPayload {
        const float* JUICER_RESTRICT lensBlurKernel = nullptr;
        int lensBlurRadius = 0;
        const float* JUICER_RESTRICT unsharpKernel = nullptr;
        int unsharpRadius = 0;
        float unsharpAmount = 0.0f;
        int glareOriginX = 0;
        int glareOriginY = 0;
        std::uint64_t glareSeed = 0;
        float glarePercent = 0.0f;
        float glareRoughness = 0.0f;
        const float* JUICER_RESTRICT glareKernel = nullptr;
        int glareRadius = 0;
    };

    struct PipelineRunParams {
        const void* src = nullptr;
        std::size_t srcRowBytes = 0;
        void* dst = nullptr;
        std::size_t dstRowBytes = 0;
        int width = 0;
        int height = 0;
        int nComponents = 0;

        // Stage-scoped payloads (pass 1 mapping; kernels still use legacy fields).
        FilmExposurePayload filmExpose{};
        FilmDevelopPayload filmDevelop{};
        PrintExposePayload printExpose{};
        PrintDevelopPayload printDevelop{};
        ScanStagePayload scanStage{};
        ScannerOpticsPayload scannerOptics{};

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

    inline void init_stage_payloads(PipelineRunParams& p) {
        p.filmExpose.exposureScale = p.exposureScale;
        p.filmExpose.sensB = p.sensB;
        p.filmExpose.sensG = p.sensG;
        p.filmExpose.sensR = p.sensR;
        p.filmExpose.tablesAx = p.tablesAx;
        p.filmExpose.tablesAy = p.tablesAy;
        p.filmExpose.tablesAz = p.tablesAz;
        p.filmExpose.tablesK = p.tablesK;
        for (int i = 0; i < 9; ++i) {
            p.filmExpose.spdSInv[i] = p.spdSInv[i];
        }
        p.filmExpose.hanatosLut = p.hanatosLut;
        p.filmExpose.hanatosN = p.hanatosN;
        p.filmExpose.hanatosLutIntegrated = p.hanatosLutIntegrated;
        p.filmExpose.hanatosNIntegrated = p.hanatosNIntegrated;

        p.filmDevelop.gammaFactorB = p.gammaFactorB;
        p.filmDevelop.gammaFactorG = p.gammaFactorG;
        p.filmDevelop.gammaFactorR = p.gammaFactorR;
        p.filmDevelop.dirPrecorrected = p.dirPrecorrected;
        p.filmDevelop.dir = p.dir;
        p.filmDevelop.spatialDir.active = p.spatialDirActive;
        p.filmDevelop.spatialDir.corrY = p.spatialDirCorrY;
        p.filmDevelop.spatialDir.corrM = p.spatialDirCorrM;
        p.filmDevelop.spatialDir.corrC = p.spatialDirCorrC;
        p.filmDevelop.densB = p.densB;
        p.filmDevelop.densG = p.densG;
        p.filmDevelop.densR = p.densR;
        p.filmDevelop.dirDensB = p.dirDensB;
        p.filmDevelop.dirDensG = p.dirDensG;
        p.filmDevelop.dirDensR = p.dirDensR;

        p.printExpose.active = p.printActive;
        p.printExpose.negTables = p.negTables;
        p.printExpose.printIllumFiltered = p.printIllumFiltered;
        p.printExpose.printIllumK = p.printIllumK;
        p.printExpose.printSensC = p.printSensC;
        p.printExpose.printSensM = p.printSensM;
        p.printExpose.printSensY = p.printSensY;
        p.printExpose.printExposure = p.printExposure;
        p.printExpose.printPreflashExposure = p.printPreflashExposure;
        p.printExpose.printMidgrayFactor = p.printMidgrayFactor;
        for (int i = 0; i < 3; ++i) {
            p.printExpose.printPreflashRaw[i] = p.printPreflashRaw[i];
        }

        p.printDevelop.printDcC = p.printDcC;
        p.printDevelop.printDcM = p.printDcM;
        p.printDevelop.printDcY = p.printDcY;
        p.printDevelop.printGammaC = p.printGammaC;
        p.printDevelop.printGammaM = p.printGammaM;
        p.printDevelop.printGammaY = p.printGammaY;

        p.scanStage.scannerUseLut = p.scannerUseLut;
        p.scanStage.scanLutLogXYZ = p.scanLutLogXYZ;
        p.scanStage.scanLutRes = p.scanLutRes;
        p.scanStage.scanTables = p.scan;
        p.scanStage.scanColor = p.scanColor;
        p.scanStage.scanErrorFlag = p.scanErrorFlag;
    }

} // namespace JuicerCuda
