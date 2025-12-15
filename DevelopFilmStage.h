#pragma once

#include "PipelineTypes.h"

struct WorkingState;

namespace Pipeline {

    struct DevelopFilmInputs {
        FilmRaw filmRaw;
    };

    struct DevelopFilmOutputs {
        FilmLogRaw filmLogRaw;
        NegativeDensityCMY negativeDensity;
    };

    class DevelopFilmStage {
    public:
        static bool run(const WorkingState& ws, const DevelopFilmInputs& in, DevelopFilmOutputs& out);
    };

} // namespace Pipeline

