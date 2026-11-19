#pragma once

#include "film_reference.h"

namespace GrainPrototype {
    struct OracleSample {
        double delta;
        double roundingBound;
        bool crossesTexel;
    };

    struct FilmProbe {
        float pitch{};
        int x{};
        int y{};
        std::uint32_t seed{};
        int record{};
        float net{};
        bool productionSubtexel = false;
    };

    // The oracle integrates piecewise constant texels from the frozen words.
    OracleSample evaluate_sample(const FilmBank& bank, FilmProbe probe);
    void calibrate_subtexel(FilmBank& bank, float pitch);
} // namespace GrainPrototype
