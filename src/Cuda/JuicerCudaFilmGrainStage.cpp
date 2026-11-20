// Copyright 2026 MUAStudio Inc. SPDX-License-Identifier: Apache-2.0
#include "Cuda/JuicerCudaFilmGrainStage.h"

namespace JuicerCuda::FilmGrain {
    LaunchResult launch_developed(const FilmQuad* bank, const FilmTables& tables, const FilmFrame& frame, FilmPlanes planes, cudaStream_t stream) {
        return {apply(bank, tables, frame, planes, stream)};
    }
} // namespace JuicerCuda::FilmGrain
