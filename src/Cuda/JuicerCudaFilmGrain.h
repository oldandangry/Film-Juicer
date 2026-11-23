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
    constexpr float kMinFootprint = 0.25F;
    constexpr float kMinProxySampleFootprint = 0.244140625F;
    constexpr float kCalibratedMaxFootprint = 37.5F;
    constexpr float kSingleWrapMaxFootprint = static_cast<float>(kTileSide - 1);
    constexpr float kMaxFootprint = 240000.0F; // 120 mm at Size 0.5 µm on one pixel.
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

    struct ProxyGrid {
        double extent[2]{};
        double step[2]{};
        double origin[2]{};
    };

    constexpr std::uint64_t kProxySerialSamples = 4096;
    constexpr unsigned int kProxyPixelsPerBatch = 16;
    constexpr unsigned int kProxyBlocksPerPixel = 64;
    constexpr unsigned int kProxySamplesPerBlock = 4096;

    struct ApertureSum {
        double light[kRecords];
        double area;
    };

    struct ProxyWorkspace {
        ApertureSum partial[kProxyPixelsPerBatch][kProxyBlocksPerPixel];
        ApertureSum total[kProxyPixelsPerBatch];
    };

    // Upper bound on the number of Full cells touched by any rendered pixel.
    std::uint64_t proxy_sample_bound(const double (&extent)[2], const double (&step)[2]) noexcept;

    struct FilmFrame {
        float pitch = 0.F;
        int originX = 0;
        int originY = 0;
        int width = 0;
        int height = 0;
        int stride = 0;
        std::uint32_t seed = 0;
        float amounts[kRecords]{1.F, 1.F, 1.F};
        std::uint32_t nextSeed = 0;
        float temporalFraction = 0.F;
        float chromaFraction = 1.F;
        std::uint8_t calibratedBorder[4]{};
        ProxyGrid proxy{};
        ProxyWorkspace* proxyWorkspace = nullptr;
    };

    struct FilmPlanes {
        float* cyan;
        float* magenta;
        float* yellow;
    };

    cudaError_t apply(const FilmQuad* sums, FilmTables tables, FilmFrame frame, FilmPlanes planes, void* stream);
} // namespace JuicerCuda::FilmGrain
