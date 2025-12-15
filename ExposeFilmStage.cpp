#include "ExposeFilmStage.h"

namespace Pipeline {

    bool ExposeFilmStage::run(const WorkingState& ws, const ExposeFilmInputs& in, ExposeFilmOutputs& out) {
        (void)ws;
        (void)in;
        out.filmRaw = FilmRaw{};
        return false;
    }

} // namespace Pipeline

