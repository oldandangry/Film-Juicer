#pragma once

#include "PipelineTypes.h"

struct WorkingState;

namespace Pipeline {

    struct ExposeFilmInputs {
        RgbLinear rgb;
        float exposureScale = 1.0f;
    };

    struct ExposeFilmOutputs {
        FilmRaw filmRaw;
    };

    class ExposeFilmStage {
    public:
        static bool run(const WorkingState& ws, const ExposeFilmInputs& in, ExposeFilmOutputs& out);
    };

} // namespace Pipeline

