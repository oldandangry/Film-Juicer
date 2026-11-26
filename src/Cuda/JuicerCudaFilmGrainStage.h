// Copyright 2026 MUAStudio Inc. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "Cuda/JuicerCudaFilmGrain.h"
#include "juicer_cuda_api.h"

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

        void bind_film_grain(
            const FjFilmGrain& input, FjExtent extent, ProxyWorkspace* workspace, FilmTables& tables, FilmFrame& frame);

        // The prepared frame retains the bank, workspace and developed planes.
        LaunchResult launch_developed(const FilmQuad* bank, const FilmTables& tables, const FilmFrame& frame, FilmPlanes planes, cudaStream_t stream);
    } // namespace FilmGrain
} // namespace JuicerCuda
