// Copyright 2026 MUAStudio Inc. SPDX-License-Identifier: Apache-2.0
#include "Cuda/JuicerCudaGrainV2Stage.h"

#include <algorithm>
#include <type_traits>

namespace JuicerCuda::GrainV2 {
    void bind_grain_v2(
        const FjFilmGrain& input, FjExtent extent, ProxyWorkspace* workspace, FilmTables& tables, FilmFrame& frame) {
        static_assert(std::extent_v<decltype(FjFilmGrain::records)> == kRecords);
        static_assert(std::extent_v<decltype(FjFilmGrainRecord::mean_light)> == kLevels);
        static_assert(std::extent_v<decltype(FjFilmGrainRecord::mean_density)> == kLevels);
        static_assert(std::extent_v<decltype(FjFilmGrainRecord::correlation)> == kLevels);
        tables.footprint = input.sample_pitch;
        for (int record = 0; record < kRecords; ++record) {
            const auto& src = input.records[record];
            auto& dst = tables.records[record];
            dst.minimumGross = src.minimum_gross;
            dst.maximumGross = src.maximum_gross;
            dst.offset = src.offset;
            for (int level = 0; level < kLevels; ++level)
                dst.levels[level] = {src.mean_light[level], src.mean_density[level], src.correlation[level]};
        }
        frame.pitch = input.pitch;
        frame.originX = 0;
        frame.originY = 0;
        frame.width = extent.width;
        frame.height = extent.height;
        frame.stride = frame.width;
        frame.seed = input.seed;
        frame.nextSeed = input.next_seed;
        frame.temporalFraction = input.temporal_fraction;
        frame.chromaFraction = input.chroma_fraction;
        std::copy_n(input.calibrated_border, 4, frame.calibratedBorder);
        // Raw admission established either two zero steps or two positive steps.
        if (input.sample_step[0] > 0.0) {
            frame.proxyWorkspace = workspace;
            std::copy_n(input.sample_extent, 2, frame.proxy.extent);
            std::copy_n(input.sample_step, 2, frame.proxy.step);
            std::copy_n(input.sample_origin, 2, frame.proxy.origin);
        }
        std::fill_n(frame.amounts, kRecords, input.gain);
    }

    LaunchResult launch_developed(const FilmQuad* bank, const FilmTables& tables, const FilmFrame& frame, FilmPlanes planes, cudaStream_t stream) {
        return {apply(bank, tables, frame, planes, stream)};
    }
} // namespace JuicerCuda::GrainV2
