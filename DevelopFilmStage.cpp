#include "DevelopFilmStage.h"

namespace Pipeline {

    bool DevelopFilmStage::run(const WorkingState& ws, const DevelopFilmInputs& in, DevelopFilmOutputs& out) {
        (void)ws;
        (void)in;
        out.filmLogRaw = FilmLogRaw{};
        out.negativeDensity = NegativeDensityCMY{};
        return false;
    }

} // namespace Pipeline

