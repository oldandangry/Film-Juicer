#pragma once

#include "PipelineTypes.h"

struct WorkingState;
namespace Print {
    struct Runtime;
}

namespace Pipeline {

    struct DevelopPrintInputs {
        const Print::Runtime* printRuntime = nullptr;
        PrintLogRaw printLogRaw;
    };

    struct DevelopPrintOutputs {
        PrintDensityCMY printDensity;
    };

    class DevelopPrintStage {
    public:
        static bool run(const DevelopPrintInputs& in, DevelopPrintOutputs& out);
    };

} // namespace Pipeline
