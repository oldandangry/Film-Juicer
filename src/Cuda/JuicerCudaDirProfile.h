#pragma once

namespace JuicerCuda {

    // Optional JUICER_DIR_PROFILE attribution for production DIR diagnosis.
    struct SpatialDirStageProfile {
        int launches = 0;
        float elapsedMs = 0.0f;
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
