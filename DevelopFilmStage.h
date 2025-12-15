#pragma once

#include "PipelineTypes.h"

struct WorkingState;
namespace Couplers {
    struct Runtime;
}

namespace Pipeline {

    struct DevelopFilmInputs {
        FilmRaw filmRaw;
        const Couplers::Runtime* dirRuntime = nullptr;
        bool applyDirRuntime = true;

        // Spatial DIR path: corrections are in [Y, M, C] order and applied to
        // [B, G, R] layer log-exposures respectively (mirrors legacy implementation).
        bool useSpatialDIR = false;
        float spatialLogECorrectionsYMC[3] = { 0.0f, 0.0f, 0.0f };
    };

    struct DevelopFilmOutputs {
        FilmLogRaw filmLogRaw;
        NegativeDensityCMY negativeDensity;
    };

    class DevelopFilmStage {
    public:
        static bool run(const WorkingState& ws, const DevelopFilmInputs& in, DevelopFilmOutputs& out);

        // Canonical film-only log rule: log10(fmax(raw, 0) + 1e-10).
        static FilmLogRaw compute_log_raw(const FilmRaw& filmRaw);
    };

} // namespace Pipeline
