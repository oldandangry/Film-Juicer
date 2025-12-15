#pragma once

#include "PipelineTypes.h"

struct WorkingState;

namespace Pipeline {

    struct ScanInputs {
        DensityMedium medium = DensityMedium::Negative;
        NegativeDensityCMY negativeDensity;
        PrintDensityCMY printDensity;
    };

    struct ScanOutputs {
        LogXyz logXyz;
        Xyz xyz;
    };

    class ScanStage {
    public:
        static bool run(const WorkingState& ws, const ScanInputs& in, ScanOutputs& out);
    };

} // namespace Pipeline

