#include "ExposeFilmStage.h"

#include <cmath>

#include "ColorTransforms.h"
#include "WorkingState.h"

namespace Pipeline {

    bool ExposeFilmStage::run(const WorkingState& ws, const ExposeFilmInputs& in, ExposeFilmOutputs& out) {
        float rgbIn[3] = { in.rgb.v[0], in.rgb.v[1], in.rgb.v[2] };
        float E[3] = { 0.0f, 0.0f, 0.0f };

        const float exposureScaleSafe = (std::isfinite(in.exposureScale) && in.exposureScale > 0.0f)
            ? in.exposureScale
            : 1.0f;

        const Spectral::SpectralTables* tablesSPD =
            (ws.spdReady && ws.tablesRef.K > 0) ? &ws.tablesRef : nullptr;

        Spectral::rgb_input_to_film_raw(
            rgbIn, E, exposureScaleSafe,
            ws.filmRaw,
            tablesSPD,
            (ws.spdReady ? ws.spdSInv : nullptr),
            ws.spdReady,
            ws.sensB, ws.sensG, ws.sensR);

        out.filmRaw.v[0] = E[0];
        out.filmRaw.v[1] = E[1];
        out.filmRaw.v[2] = E[2];
        return true;
    }

} // namespace Pipeline
