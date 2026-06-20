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
        int scaleCopyLaunches = 0;
        int addScaledLaunches = 0;
        int totalLaunches = 0;
        SpatialDirStageProfile total{};
        SpatialDirStageProfile correction{};
        SpatialDirStageProfile baseFilter{};
        SpatialDirStageProfile tailFilter[3]{};
        SpatialDirStageProfile scaleCopy{};
        SpatialDirStageProfile addScaled{};
    };

} // namespace JuicerCuda
