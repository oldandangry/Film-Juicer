#pragma once

#include <cstdint>

namespace JuicerCuda {

    // Temporary Phase 0 spatial-DIR instrumentation.
    // Enabled only when host code passes this profile pointer, currently gated by
    // JUICER_DIR_PROFILE. Remove with the Phase 0 current-path capture code.
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
        std::uint32_t correctionClampHits = 0;
        int correctionLaunches = 0;
        int baseFilterLaunches = 0;
        int tailFilterLaunches[3] = {0, 0, 0};
        int fftFilterLaunches = 0;
        int fftPadPixels = 0;
        int fftWidth = 0;
        int fftHeight = 0;
        int fftComplexWidth = 0;
        std::uint64_t fftRealBufferBytes = 0;
        std::uint64_t fftSpectrumBytes = 0;
        std::uint64_t fftTransferBytes = 0;
        std::uint64_t fftWorkAreaBytes = 0;
        std::uint64_t fftForwardWorkBytes = 0;
        std::uint64_t fftInverseWorkBytes = 0;
        double fftSetupMs = 0.0;
        int fftSetupCreated = 0;
        int fftActive = 0;
        int scaleCopyLaunches = 0;
        int addScaledLaunches = 0;
        int totalLaunches = 0;
        const char* SF_TEMP_BRIDGE_name = nullptr;
        SpatialDirStageProfile total{};
        SpatialDirStageProfile correction{};
        SpatialDirStageProfile baseFilter{};
        SpatialDirStageProfile tailFilter[3]{};
        SpatialDirStageProfile fftFilter{};
        SpatialDirStageProfile scaleCopy{};
        SpatialDirStageProfile addScaled{};
    };

} // namespace JuicerCuda
