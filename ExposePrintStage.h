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

    struct ExposePrintInputs {
        const Print::Runtime* printRuntime = nullptr;
        const Print::Params* printParams = nullptr;
        NegativeDensityCMY negativeDensity;
        float midgrayFactor = 1.0f;
    };

    struct ExposePrintOutputs {
        PrintRaw printRaw;
        PrintLogRaw printLogRaw;
    };

    class ExposePrintStage {
    public:
        static bool run(
            const WorkingState& ws,
            const ExposePrintInputs& in,
            ExposePrintOutputs& out,
            PrintPipelineScratch& scratch);

        static float compute_midgray_factor(
            const WorkingState& ws,
            const Print::Runtime& printRuntime,
            const Print::Params& printParams,
            const Couplers::Runtime& dirRT,
            float exposureCompScale);
    };

} // namespace Pipeline
