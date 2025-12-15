#pragma once

#include "PipelineTypes.h"

struct WorkingState;

namespace Pipeline {

    struct PipelineRunnerConfig {
        bool enablePrint = true;
    };

    class PipelineRunner {
    public:
        explicit PipelineRunner(const PipelineRunnerConfig& cfg);

        bool run_pixel(const WorkingState& ws, const RgbLinear& inRgb, float outRgb[3]) const;

    private:
        PipelineRunnerConfig cfg_;
    };

} // namespace Pipeline

