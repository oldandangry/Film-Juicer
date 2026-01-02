#pragma once

#include "PipelineTypes.h"

struct WorkingState;
namespace Print {
    struct Params;
    struct Runtime;
}
namespace Couplers {
    struct Runtime;
}

namespace Pipeline {

    struct PipelineRunnerConfig {
        bool enablePrint = true;
    };

    struct DensityPixelInputs {
        RgbLinear rgb;
        float exposureScale = 1.0f;

        const Couplers::Runtime* dirRuntime = nullptr;
        bool applyDirRuntime = true;

        bool useFilmRawOverride = false;
        FilmRaw filmRawOverride;

        bool useSpatialDIR = false;
        float spatialLogECorrectionsYMC[3] = { 0.0f, 0.0f, 0.0f };

        const Print::Runtime* printRuntime = nullptr;
        const Print::Params* printParams = nullptr;
        float midgrayFactor = 1.0f;
        PrintPipelineScratch* printScratch = nullptr;
    };

    struct DensityPixelOutputs {
        FilmRaw filmRaw;
        FilmLogRaw filmLogRaw;
        NegativeDensityCMY negativeDensity;
        PrintDensityCMY printDensity;
        DensityMedium medium = DensityMedium::Negative;
    };

    class PipelineRunner {
    public:
        explicit PipelineRunner(const PipelineRunnerConfig& cfg);

        bool run_density_pixel(const WorkingState& ws, const DensityPixelInputs& in, DensityPixelOutputs& out) const;

        static float compute_midgray_factor(
            const WorkingState& ws,
            const Print::Runtime& printRuntime,
            const Print::Params& printParams,
            const Couplers::Runtime& dirRT,
            float exposureCompScale);

    private:
        PipelineRunnerConfig cfg_;
    };

} // namespace Pipeline
