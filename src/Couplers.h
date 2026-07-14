#pragma once

namespace Couplers {

    struct Runtime {
        bool active = true;
        float M[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        float highShift = 0.0f;
        float dMax[3] = {1.0f, 1.0f, 1.0f};
        float spatialSigmaMicrometers = 0.0f;
        float spatialSigmaPixels = 0.0f;
    };

} // namespace Couplers
