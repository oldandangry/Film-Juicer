#pragma once

#include <limits>

#include "Cuda/JuicerCudaPayloads.h"

namespace JuicerCuda {
    inline bool valid_camera_film_linear_planes(
        const JuicerCuda::CameraFilmLinearExposurePlanes& planes,
        int width,
        int height) noexcept {
        return planes.redSensitive &&
               planes.greenSensitive &&
               planes.blueSensitive &&
               planes.redSensitive != planes.greenSensitive &&
               planes.redSensitive != planes.blueSensitive &&
               planes.greenSensitive != planes.blueSensitive &&
               width > 0 &&
               height > 0 &&
               planes.rowStrideFloats >= static_cast<std::size_t>(width) &&
               planes.rowStrideFloats <=
                   std::numeric_limits<std::size_t>::max() /
                       static_cast<std::size_t>(height);
    }

} // namespace JuicerCuda
