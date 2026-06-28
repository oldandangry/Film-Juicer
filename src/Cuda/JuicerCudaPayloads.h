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
        // Inclusive finite-domain indices into x[]/y[] (precomputed on CPU to avoid per-call scans).
        int domainBegin = 0;
        int domainEnd = 0;
    };

    // Mirrors the subset of Spectral::FilmRawConfig needed by CUDA kernels.
    struct FilmRawPayload {
        int inputColorSpaceIndex = 0;
        int applyCctfDecoding = 0;
        int applyInputChromaticAdapt = 0;
        int spectralUpsamplingMode = 0; // 0=PreferHanatos, 1=ForceMallett (Spectral::SpectralUpsamplingMode)

        float inputRGBToXYZ[9] = {
            1, 0, 0, 0, 1, 0, 0, 0, 1};
        float inputXYZAdapt[9] = {
            1, 0, 0, 0, 1, 0, 0, 0, 1};

        float mallettGreenMidgrayScale = 1.0f;
        float refIllumWhiteXYZ[3] = {0.950455f, 1.0f, 1.089058f};
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
            1, 0, 0, 0, 1, 0, 0, 0, 1};
        CctfPayload cctf{};
    };

    struct DirPayload {
        int active = 0;
        int positive = 0;
        float M[9] = {
            0, 0, 0, 0, 0, 0, 0, 0, 0};
        float dMax[3] = {1.0f, 1.0f, 1.0f};
    };

    // Scanner color runtime (CAT + XYZ->RGB) plus output encoding selection.
    struct ScanColorPayload {
        float cat02[9] = {0.0f};
        float xyzToRgb[9] = {0.0f};
        float illuminantXYZ[3] = {0.0f, 0.0f, 0.0f};
        OutputEncodingPayload encoding{};
    };

    struct ScanTablesPayload {
        const float* JUICER_RESTRICT epsC = nullptr;
        const float* JUICER_RESTRICT epsM = nullptr;
        const float* JUICER_RESTRICT epsY = nullptr;
        const float* JUICER_RESTRICT Ax = nullptr;
        const float* JUICER_RESTRICT Ay = nullptr;
        const float* JUICER_RESTRICT Az = nullptr;
        const float* JUICER_RESTRICT baseDensityMin = nullptr;
        int K = 0;
        int hasBaseline = 0;
        float invYn = 1.0f;
        int mediumIsNegative = 1;
        float min_cmy[3] = {0.0f, 0.0f, 0.0f};
        float inv_max_cmy[3] = {1.0f, 1.0f, 1.0f};
    };

    struct SpatialDirPayload {
        int active = 0;
        const float* JUICER_RESTRICT corrY = nullptr;
        const float* JUICER_RESTRICT corrM = nullptr;
        const float* JUICER_RESTRICT corrC = nullptr;
        const float* JUICER_RESTRICT logRawB = nullptr;
        const float* JUICER_RESTRICT logRawG = nullptr;
        const float* JUICER_RESTRICT logRawR = nullptr;
    };

    struct HighlightBoostPayload {
        float boostEv = 0.0f;
        float boostRange = 0.3f;
        float protectEv = 4.0f;
    };

    struct FilmExposurePayload {
        float manualExposureScale = 1.0f;
        float routeCorrectionScale = 1.0f;
        const float* JUICER_RESTRICT exposureScaleDevice = nullptr;
        HighlightBoostPayload highlightBoost{};
        DeviceCurveView sensB{};
        DeviceCurveView sensG{};
        DeviceCurveView sensR{};
        const float* JUICER_RESTRICT tablesAx = nullptr;
        const float* JUICER_RESTRICT tablesAy = nullptr;
        const float* JUICER_RESTRICT tablesAz = nullptr;
        const float* JUICER_RESTRICT tablesIllum = nullptr;
        int tablesK = 0;
        float spdSInv[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        const float* JUICER_RESTRICT hanatosLut = nullptr;
        int hanatosN = 0;
        const float* JUICER_RESTRICT hanatosLutIntegrated = nullptr;
        int hanatosNIntegrated = 0;
        const float* JUICER_RESTRICT mallettBasis = nullptr;
        int mallettBasisK = 0;
    };

    struct FilmDevelopPayload {
        // Direct production packs only DirCouplersRecipe and prepared-frame views here.
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

    struct HalationPayload {
        int active = 0;
        float primaryAmount[3] = {0.0f, 0.0f, 0.0f};
        float secondaryAmount[3] = {0.0f, 0.0f, 0.0f};
    };

    struct HalationKernelPayload {
        const float* JUICER_RESTRICT halationKernel[3] = {nullptr, nullptr, nullptr};
        int halationRadius[3] = {0, 0, 0};
        const float* JUICER_RESTRICT scatteringKernel[3] = {nullptr, nullptr, nullptr};
        int scatteringRadius[3] = {0, 0, 0};
    };

    struct GrainPayload {
        int active = 0;
        int sublayersActive = 0;
        int nSubLayers = 1;
        int originX = 0;
        int originY = 0;
        std::uint64_t seedBase = 0;
        std::uint64_t seedBaseNext = 0;
        const std::uint8_t* JUICER_RESTRICT stbn = nullptr;
        int stbnWidth = 0;
        int stbnHeight = 0;
        int stbnFrames = 0;
        int stbnOffsetX = 0;
        int stbnOffsetY = 0;
        int stbnFrame = 0;
        std::int64_t frameIndex = 0;
        float timeAlpha = 0.0f;
        std::uint64_t stbnSessionSeed = 0;
        std::uint64_t clipToken = 0;
        const std::uint8_t* JUICER_RESTRICT wangTiles = nullptr;
        const std::uint8_t* JUICER_RESTRICT wangLut = nullptr;
        int wangWidth = 0;
        int wangHeight = 0;
        int wangCount = 0;
        int wangColors = 0;
        float wangCellMm = 0.0f;
        int breathingPeriodFrames = 0;
        float breathingAmplitude = 0.0f;
        float breathingCellUmSmall = 0.0f;
        float breathingCellUmLarge = 0.0f;
        float breathingMix = 0.0f;
        float breathingDriftUmPerFrame = 0.0f;
        int breathingDebug = 0;
        int debugView = 0;
        float pixelSizeUm = 0.0f; // Pixel size in micrometers.
        int pitchPx = 0;
        float blurSigmaPx = 0.0f;               // Grain blur sigma in pixels.
        float blurDyeCloudsUm = 0.0f;           // Dye-cloud blur sigma scale in pixels.
        float microStructure[2] = {0.0f, 0.0f}; // [cell_um, clump_sigma_x1e-3]
        float clumpTemporalMix = 0.0f;
        int clumpMorphPeriodFrames = 0;
        float filmDustAmount = 0.0f;
        float gateDustAmount = 0.0f;
        float filmScratchAmount = 0.0f;
        float gateScratchAmount = 0.0f;
        const float* JUICER_RESTRICT gateMask = nullptr;
        int gateMaskWidth = 0;
        int gateMaskHeight = 0;
        float densityMin[3] = {0.0f, 0.0f, 0.0f};
        float uniformity[3] = {0.0f, 0.0f, 0.0f};
        float densityMax[3] = {0.0f, 0.0f, 0.0f};
        float nParticles[3] = {0.0f, 0.0f, 0.0f};
        float odParticle[3] = {0.0f, 0.0f, 0.0f};
        float sizeMixWeight = 0.0f;
        float sizeMixWeightMid = 0.0f;
        float sizeMixScale = 1.0f;
        float sizeMixGain = 1.0f;
        float amplitude = 1.0f;
        float chromaMix = 1.0f;
        float chromaSharedWeight = 0.0f;
        float chromaIndWeight = 1.0f;
        float debugScale = 1.0f;

        float densityMaxLayers[3][3] = {{0.0f, 0.0f, 0.0f},
                                        {0.0f, 0.0f, 0.0f},
                                        {0.0f, 0.0f, 0.0f}};
        float densityMinLayers[3][3] = {{0.0f, 0.0f, 0.0f},
                                        {0.0f, 0.0f, 0.0f},
                                        {0.0f, 0.0f, 0.0f}};
        float nParticlesLayers[3][3] = {{0.0f, 0.0f, 0.0f},
                                        {0.0f, 0.0f, 0.0f},
                                        {0.0f, 0.0f, 0.0f}};
        float odParticleLayers[3][3] = {{0.0f, 0.0f, 0.0f},
                                        {0.0f, 0.0f, 0.0f},
                                        {0.0f, 0.0f, 0.0f}};
        const float* JUICER_RESTRICT densityCurvesLayers[3][3] = {
            {nullptr, nullptr, nullptr},
            {nullptr, nullptr, nullptr},
            {nullptr, nullptr, nullptr}};
    };

    struct GateWeavePayload {
        int active = 0;
        float dxPx = 0.0f;
        float dyPx = 0.0f;
        float cosRot = 1.0f;
        float sinRot = 0.0f;
        float debugScalePx = 1.0f;
    };

    struct GrainKernelPayload {
        const float* JUICER_RESTRICT blurKernel = nullptr;
        int blurRadius = 0;
        const float* JUICER_RESTRICT blurKernelMid = nullptr;
        int blurRadiusMid = 0;
        const float* JUICER_RESTRICT blurKernelCoarse = nullptr;
        int blurRadiusCoarse = 0;
        const float* JUICER_RESTRICT dyeKernel[3][3] = {
            {nullptr, nullptr, nullptr},
            {nullptr, nullptr, nullptr},
            {nullptr, nullptr, nullptr}};
        int dyeRadius[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
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
        float routeCorrectionScale = 1.0f;
        float printPreflashExposure = 0.0f;
        float printMidgrayFactor = 1.0f;
        float printPreflashRaw[3] = {0.0f, 0.0f, 0.0f};
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
        const float* JUICER_RESTRICT scanLutLog2PchipXYZ = nullptr;
        const float* JUICER_RESTRICT scanLutPchipSlopeC = nullptr;
        const float* JUICER_RESTRICT scanLutPchipSlopeM = nullptr;
        const float* JUICER_RESTRICT scanLutPchipSlopeY = nullptr;
        const float* JUICER_RESTRICT scanLutPchipCellMin = nullptr;
        const float* JUICER_RESTRICT scanLutPchipCellMax = nullptr;
        int scanLutRes = 0;
        ScanTablesPayload scanTables{};
        ScanColorPayload scanColor{};
        int correctionActive = 0;
        float correctionSlope = 1.0f;
        float correctionOffset = 0.0f;
        const float* JUICER_RESTRICT glarePercent = nullptr;
        float* JUICER_RESTRICT linearRgbR = nullptr;
        float* JUICER_RESTRICT linearRgbG = nullptr;
        float* JUICER_RESTRICT linearRgbB = nullptr;
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

    struct DirectPipelineRunParams {
        const void* src = nullptr;
        std::size_t srcRowBytes = 0;
        void* dst = nullptr;
        std::size_t dstRowBytes = 0;
        int width = 0;
        int height = 0;
        int nComponents = 0;
        FilmExposurePayload filmExpose{};
        FilmDevelopPayload filmDevelop{};
        ScanStagePayload scanStage{};
        FilmRawPayload filmRaw{};
    };

    struct PrintPipelineRunParams {
        const void* src = nullptr;
        std::size_t srcRowBytes = 0;
        void* dst = nullptr;
        std::size_t dstRowBytes = 0;
        int width = 0;
        int height = 0;
        int nComponents = 0;
        FilmExposurePayload filmExpose{};
        FilmDevelopPayload filmDevelop{};
        PrintExposePayload printExpose{};
        PrintDevelopPayload printDevelop{};
        ScanStagePayload scanStage{};
        FilmRawPayload filmRaw{};
    };

    // Broad CUDA payload used by the monolithic launch path; focused direct rendering uses
    // DirectPipelineRunParams.
    struct PipelineRunParams {
        const void* src = nullptr;
        std::size_t srcRowBytes = 0;
        void* dst = nullptr;
        std::size_t dstRowBytes = 0;
        int width = 0;
        int height = 0;
        int nComponents = 0;

        // Stage-scoped payloads.
        FilmExposurePayload filmExpose{};
        FilmDevelopPayload filmDevelop{};
        HalationPayload halation{};
        HalationKernelPayload halationKernels{};
        GrainPayload grain{};
        GrainKernelPayload grainKernels{};
        GateWeavePayload gateWeave{};
        PrintExposePayload printExpose{};
        PrintDevelopPayload printDevelop{};
        ScanStagePayload scanStage{};
        ScannerOpticsPayload scannerOptics{};

        FilmRawPayload filmRaw{};
    };

} // namespace JuicerCuda
