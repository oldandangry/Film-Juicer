// Copyright 2026 MUAStudio Inc. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "Cuda/JuicerCudaFilmGrain.h"

namespace JuicerTests::GrainMotion {
    // Exclusively active in the offline witness on its CPU submission thread.
    cudaError_t before_grain(
        JuicerCuda::FilmGrain::FilmFrame& frame,
        float samplePitch,
        JuicerCuda::FilmGrain::FilmPlanes planes,
        cudaStream_t stream,
        bool& applyGrain);
    cudaError_t after_grain(JuicerCuda::FilmGrain::FilmPlanes planes, cudaStream_t stream) noexcept;
} // namespace JuicerTests::GrainMotion
