#pragma once

#include <cstdint>

namespace JuicerCuda {

    // Optional JUICER_DIR_PROFILE attribution for production DIR diagnosis.
    struct SpatialDirStageProfile {
        int launches = 0;
        float elapsedMs = 0.0f;
    };

    struct VisualGrainRuntimeProfile {
        int width = 0;
        int height = 0;
        int captured = 0;
        int totalLaunches = 0;
        int mixEvaluations = 0;
        int scaleEvaluations = 0;
        int frameUniformPreparationLaunches = 0;
        int clearLaunches = 0;
        int layerParticleLaunches = 0;
        int simpleParticleLaunches = 0;
        int dyeBlurPassLaunches = 0;
        int correlationBlurPassLaunches = 0;
        int formDeltaLaunches = 0;
        int subtractLaunches = 0;
        int layerAccumulateLaunches = 0;
        int weightedAccumulateLaunches = 0;
        int scaleLaunches = 0;
        int sharedMixLaunches = 0;
        int reconstructLaunches = 0;
        int debugLaunches = 0;
        int copyOperations = 0;
        std::uint64_t copyBytes = 0;
        SpatialDirStageProfile total{};
    };

    struct SpatialDirBuildProfile {
        int width = 0;
        int height = 0;
        int gaussianRadius = 0;
        float gaussianSigma = 0.0f;
        float gaussianWeight = 0.0f;
        int tailRadius[3] = {0, 0, 0};
        float tailSigma[3] = {0.0f, 0.0f, 0.0f};
        float tailWeight[3] = {0.0f, 0.0f, 0.0f};
        int correctionLaunches = 0;
        int baseFilterLaunches = 0;
        int tailFilterLaunches[3] = {0, 0, 0};
        int totalLaunches = 0;
        SpatialDirStageProfile total{};
        SpatialDirStageProfile correction{};
        SpatialDirStageProfile baseFilter{};
        SpatialDirStageProfile tailFilter[3]{};
    };

    struct SpatialDirPlanes {
        float* rawCorrectionY = nullptr;
        float* rawCorrectionM = nullptr;
        float* rawCorrectionC = nullptr;
        float* filteredCorrectionY = nullptr;
        float* filteredCorrectionM = nullptr;
        float* filteredCorrectionC = nullptr;
        float* filterTemp = nullptr;
        float* filterTempM = nullptr;
        float* filterTempC = nullptr;
        float* logRawB = nullptr;
        float* logRawG = nullptr;
        float* logRawR = nullptr;
    };

    struct SpatialDirFilterSpec {
        const float* kernel = nullptr;
        int radius = 0;
        float sigma = 0.0f;
        float weight = 0.0f;
    };

    struct SpatialDirBuildRequest {
        SpatialDirPlanes planes{};
        SpatialDirFilterSpec gaussian{};
        SpatialDirFilterSpec tails[3]{};
        void* streamOpaque = nullptr;
        SpatialDirBuildProfile* profile = nullptr;
    };

    struct PrintDevelopBreakdownProfile {
        int captured = 0;
        const char* profileKind = nullptr;
        const char* profileNote = nullptr;
        SpatialDirStageProfile total{};
        SpatialDirStageProfile spectralIntegrate{};
        SpatialDirStageProfile exposureScale{};
        SpatialDirStageProfile logEncode{};
        SpatialDirStageProfile densityCurve{};
    };

    struct CompositePipelineProfile {
        int width = 0;
        int height = 0;
        int captured = 0;
        int totalLaunches = 0;
        const char* profileKind = nullptr;
        const char* profileNote = nullptr;
        SpatialDirStageProfile total{};
        SpatialDirStageProfile filmRaw{};
        SpatialDirStageProfile filmDevelop{};
        SpatialDirStageProfile printDevelop{};
        SpatialDirStageProfile scannerLinear{};
        SpatialDirStageProfile outputEncode{};
        SpatialDirStageProfile glare{};
        SpatialDirStageProfile lensBlur{};
        SpatialDirStageProfile unsharp{};
        int aliasRouteCaptured = 0;
        SpatialDirStageProfile aliasFusedScanLinear{};
        SpatialDirStageProfile aliasScannerPostOutput{};
        SpatialDirStageProfile aliasGlare{};
        SpatialDirStageProfile aliasFinalDevelopScanLinear{};
        SpatialDirStageProfile aliasLensBlur{};
        SpatialDirStageProfile aliasUnsharp{};
        SpatialDirStageProfile aliasOutputEncode{};
        PrintDevelopBreakdownProfile printDevelopBreakdown{};
    };

} // namespace JuicerCuda
