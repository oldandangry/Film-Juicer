// Copyright 2026 MUAStudio Inc. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "Cuda/JuicerCudaFilmGrain.h"

namespace JuicerCuda {
    struct GrainFailureSite {
        const char* stage;
        const char* prefix;
    };

    namespace FilmGrain {
        struct LaunchResult {
            cudaError_t status = cudaSuccess;
            const GrainFailureSite* failure = nullptr;
        };

        // The link selects the stage owner. The prepared frame retains all device
        // resources; a bare CUDA failure is attributed to the route's Film launch.
        LaunchResult launch_developed(const FilmQuad* bank, const FilmTables& tables, const FilmFrame& frame, FilmPlanes planes, cudaStream_t stream);
    } // namespace FilmGrain
} // namespace JuicerCuda
