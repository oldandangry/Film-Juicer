// Copyright 2026 MUAStudio Inc. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace JuicerCuda::FilmGrain {
    constexpr int kTileSide = 256;
    constexpr int kTileStride = kTileSide + 1;
    constexpr int kTileEntries = kTileStride * kTileStride;
    constexpr int kLevels = 17;
    constexpr int kRecords = 3;
    constexpr int kPlacementBlock = 128;
    constexpr int kPlacementMargin = 64;
    constexpr int kSeam = 16;
    constexpr int kLayings = 2;
    constexpr std::size_t kBankEntries = static_cast<std::size_t>(kRecords) * kLevels * kTileEntries;

    struct FilmLevel {
        float meanLight;
        float meanDensity;
        float correlation;
    };

    struct FilmRecord {
        float minimumGross;
        float maximumGross;
        float offset;
        FilmLevel levels[kLevels];
    };

    // Frozen single-footprint view of a bank. The planes and SAT are C/M/Y.
    struct FilmTables {
        FilmRecord records[kRecords];
        float footprint;
    };

    struct alignas(16) FilmQuad {
        float topLeft;
        float topRight;
        float bottomLeft;
        float bottomRight;
    };
    static_assert(sizeof(FilmQuad) == 16 && alignof(FilmQuad) == 16);
    constexpr std::size_t kQuadEntries = static_cast<std::size_t>(kRecords) * kLevels * kTileSide * kTileSide;

    struct FilmFrame {
        float pitch = 0.F;
        int originX = 0;
        int originY = 0;
        int width = 0;
        int height = 0;
        int stride = 0;
        std::uint32_t seed = 0;
        bool enabled = false;
        float amounts[kRecords]{1.F, 1.F, 1.F};
        std::uint32_t nextSeed = 0;
        float temporalFraction = 0.F;
        float chromaFraction = 1.F;
    };

    struct FilmPlanes {
        float* cyan;
        float* magenta;
        float* yellow;
    };

    cudaError_t apply(const FilmQuad* sums, FilmTables tables, FilmFrame frame, FilmPlanes planes, void* stream);
} // namespace JuicerCuda::FilmGrain
