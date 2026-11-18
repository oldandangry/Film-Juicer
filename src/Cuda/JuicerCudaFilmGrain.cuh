// Adapted from Fotufilm's FilmTiles.h and Random.h, via the Film-Juicer G3 sampler.
// Copyright 2026 MUAStudio Inc. SPDX-License-Identifier: Apache-2.0
// See tests/grain/fotufilm-notice.md for source identity and license.
#pragma once

#include "Cuda/JuicerCudaFilmGrain.h"

namespace JuicerCuda::FilmGrain {
    namespace {
        struct FilmBox {
            float left;
            float right;
            float top;
            float bottom;
        };

        __device__ std::uint32_t pcg(std::uint32_t input) {
            const auto state = input * 747796405U + 2891336453U;
            const auto word = ((state >> ((state >> 28U) + 4U)) ^ state) * 277803737U;
            return (word >> 22U) ^ word;
        }

        // Keep the upstream block-coordinate, frame-seed and lane ordering.
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        __device__ std::uint32_t placement_hash(int bx, int by, std::uint32_t seed, int laying, int record) {
            const auto lane = static_cast<std::uint32_t>(200 + 3 * laying + record);
            return pcg(static_cast<std::uint32_t>(bx) ^
                       pcg(static_cast<std::uint32_t>(by) ^ pcg(seed ^ (lane * 0x9E3779B9U))));
        }

        __device__ FilmBox place_box(FilmBox box, std::uint32_t hash) {
            const auto turn = (hash >> 16U) & 7U;
            if ((turn & 1U) != 0U) {
                const float left = kPlacementBlock - box.right;
                box.right = kPlacementBlock - box.left;
                box.left = left;
            }
            if ((turn & 2U) != 0U) {
                const float top = kPlacementBlock - box.bottom;
                box.bottom = kPlacementBlock - box.top;
                box.top = top;
            }
            if ((turn & 4U) != 0U)
                box = {box.top, box.bottom, box.left, box.right};
            const float offsetX = static_cast<float>(hash & 255U) + kPlacementMargin;
            const float offsetY = static_cast<float>((hash >> 8U) & 255U) + kPlacementMargin;
            return {box.left + offsetX, box.right + offsetX, box.top + offsetY, box.bottom + offsetY};
        }

        __device__ float seam_weight(float coordinate, float start) {
            const float leading = __saturatef((coordinate - start + kSeam) / (2 * kSeam));
            const float trailing = __saturatef((start + kPlacementBlock + kSeam - coordinate) / (2 * kSeam));
            return leading * trailing;
        }

        __device__ void reduce_chroma(const float (&input)[kRecords], float (&grained)[kRecords], float fraction) {
            // The G2 stills use the scalar CMY transmission as linear display RGB.
            // Mix toward the input's chromaticity at the sampled luminance. Separate
            // density pivots keep each set of transmission weights well scaled.
            constexpr float weights[kRecords]{0.2126F, 0.7152F, 0.0722F};
            const float inputPivot = fminf(input[0], fminf(input[1], input[2]));
            const float grainPivot = fminf(grained[0], fminf(grained[1], grained[2]));
            float inputLight[kRecords];
            float grainLight[kRecords];
            float inputLuminance = 0.F;
            float grainLuminance = 0.F;
            for (int record = 0; record < kRecords; ++record) {
                inputLight[record] = exp10f(inputPivot - input[record]);
                grainLight[record] = exp10f(grainPivot - grained[record]);
                inputLuminance += weights[record] * inputLight[record];
                grainLuminance += weights[record] * grainLight[record];
            }
            const float sharedScale = (1.F - fraction) * grainLuminance / inputLuminance;
            for (int record = 0; record < kRecords; ++record)
                grained[record] = grainPivot - log10f(fraction * grainLight[record] + sharedScale * inputLight[record]);
        }

        __device__ float sum_inside_quad(const FilmQuad* sums, float x, float y) {
            const int ix = min(static_cast<int>(x), kTileSide - 1);
            const int iy = min(static_cast<int>(y), kTileSide - 1);
            const float ax = x - static_cast<float>(ix);
            const float ay = y - static_cast<float>(iy);
            const FilmQuad q = sums[static_cast<std::ptrdiff_t>(iy) * kTileSide + ix];
            const float cross = q.bottomRight - q.topRight - q.bottomLeft + q.topLeft;
            return q.topLeft + (q.topRight - q.topLeft) * ax + (q.bottomLeft - q.topLeft) * ay + cross * ax * ay;
        }

        // The horizontal and vertical wrap corrections retain their original order.
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        __device__ float sum_wrapped_quad(const FilmQuad* sums, float x, float y) {
            const bool wrapX = x >= kTileSide;
            const bool wrapY = y >= kTileSide;
            const float tx = wrapX ? x - kTileSide : x;
            const float ty = wrapY ? y - kTileSide : y;
            float total = sum_inside_quad(sums, tx, ty);
            if (wrapX)
                total += sum_inside_quad(sums, kTileSide, ty);
            if (wrapY)
                total += sum_inside_quad(sums, tx, kTileSide);
            if (wrapX && wrapY)
                total += sums[kTileSide * kTileSide - 1].bottomRight;
            return total;
        }

        __device__ float density_delta_quad(const FilmQuad* sums, const FilmBox& box, float area, FilmLevel level) {
            const float integrated = sum_wrapped_quad(sums, box.right, box.bottom) -
                                     sum_wrapped_quad(sums, box.left, box.bottom) -
                                     sum_wrapped_quad(sums, box.right, box.top) +
                                     sum_wrapped_quad(sums, box.left, box.top);
            const float light = level.meanLight + integrated / area;
            return -log10f(fmaxf(light, 1e-9F)) - level.meanDensity;
        }

        // One blend and footprint at this film-space center for the chosen lane.
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        __device__ float sample_laying_quad(const FilmQuad* lowSums, const FilmLevel* levels, float fraction, float footprint, float cx, float cy, std::uint32_t seed, int laying, int record) {
            const int firstX = static_cast<int>(floorf((cx - kSeam) / kPlacementBlock));
            const int lastX = static_cast<int>(floorf((cx + kSeam) / kPlacementBlock));
            const int firstY = static_cast<int>(floorf((cy - kSeam) / kPlacementBlock));
            const int lastY = static_cast<int>(floorf((cy + kSeam) / kPlacementBlock));
            const float half = footprint * 0.5F;
            float value = 0;
            float normSquared = 0;
            for (int by = firstY; by <= lastY; ++by) {
                const float top = static_cast<float>(by * kPlacementBlock);
                const float weightY = seam_weight(cy, top);
                for (int bx = firstX; bx <= lastX; ++bx) {
                    const float left = static_cast<float>(bx * kPlacementBlock);
                    const float weight = seam_weight(cx, left) * weightY;
                    if (weight == 0)
                        continue;
                    // A contributing center extends at most 16 beyond a block;
                    // half of the largest footprint is 18.75, below the 64 margin.
                    // Offsets therefore remain in [0, 512), so one wrap suffices.
                    const float x0 = cx - half;
                    const float y0 = cy - half;
                    const FilmBox local{x0 - left, (x0 + footprint) - left, y0 - top, (y0 + footprint) - top};
                    const float area = fmaxf((local.right - local.left) * (local.bottom - local.top), 1e-6F);
                    const FilmBox box = place_box(local, placement_hash(bx, by, seed, laying, record));
                    const float low = density_delta_quad(lowSums, box, area, levels[0]);
                    const float high = density_delta_quad(lowSums + static_cast<std::ptrdiff_t>(kTileSide) * kTileSide, box, area, levels[1]);
                    value += weight * (low + (high - low) * fraction);
                    normSquared += weight * weight;
                }
            }
            return value / sqrtf(fmaxf(normSquared, 1e-12F));
        }

        // Pitch, seed and pixel coordinates follow the original sampler's operands.
        template <int Record>
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        __device__ float sample_quad(const FilmQuad* sums, const FilmTables& tables, float pitch, std::uint32_t seed, int x, int y, float net) {
            const auto& table = tables.records[Record];
            const float gross = net + table.offset;
            const float level = __saturatef((gross - table.minimumGross) /
                                            fmaxf(table.maximumGross - table.minimumGross, 1e-6F)) *
                                (kLevels - 1);
            const int low = min(static_cast<int>(level), kLevels - 2);
            const float fraction = level - static_cast<float>(low);
            const auto offset = (static_cast<std::ptrdiff_t>(Record) * kLevels + low) * (static_cast<std::ptrdiff_t>(kTileSide) * kTileSide);
            const FilmQuad* lowSums = sums + offset;
            const float cx = (static_cast<float>(x) + 0.5F) * pitch;
            const float cy = (static_cast<float>(y) + 0.5F) * pitch;
            float sum = 0;
            for (int laying = 0; laying < kLayings; ++laying) {
                const float shift = static_cast<float>(kPlacementBlock * laying / kLayings);
                sum += sample_laying_quad(lowSums, table.levels + low, fraction, tables.footprint, cx + shift, cy + shift, seed, laying, Record);
            }
            const float other = 1 - fraction;
            const float kept = other * other + fraction * fraction +
                               2 * fraction * other * table.levels[low].correlation;
            return sum / sqrtf(fmaxf(kept * kLayings, 1e-12F));
        }

        // Seed pair, amount and pixel coordinates keep their frame semantics.
        template <int Record>
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        __device__ float grain_quad(const FilmQuad* sums, const FilmTables& tables, float pitch, std::uint32_t seed, std::uint32_t nextSeed, float time, float amount, int x, int y, float density, bool& changed) {
            const float delta = sample_quad<Record>(sums, tables, pitch, seed, x, y, density);
            float blended = delta;
            if (time != 0.F) {
                const float other = sample_quad<Record>(sums, tables, pitch, nextSeed, x, y, density);
                const float t = time;
                const float weight = 1.F - t;
                blended = (weight * delta + t * other) / sqrtf(weight * weight + t * t);
            }
            float grained = density;
            if (blended != 0.F && amount != 0.F) {
                grained += amount * blended;
                changed = changed || grained != density;
            }
            return grained;
        }

        __global__ void apply_quad(const FilmQuad* sums, FilmTables tables, FilmFrame frame, FilmPlanes planes) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            if (x >= frame.width || y >= frame.height)
                return;
            const auto index = static_cast<std::size_t>(y) * static_cast<std::size_t>(frame.stride) + static_cast<std::size_t>(x);
            const float cyan = planes.cyan[index];
            const float magenta = planes.magenta[index];
            const float yellow = planes.yellow[index];
            bool changed = false;
            float grained[kRecords];
            grained[0] = grain_quad<0>(sums, tables, frame.pitch, frame.seed, frame.nextSeed, frame.temporalFraction, frame.amounts[0], x + frame.originX, y + frame.originY, cyan, changed);
            grained[1] = grain_quad<1>(sums, tables, frame.pitch, frame.seed, frame.nextSeed, frame.temporalFraction, frame.amounts[1], x + frame.originX, y + frame.originY, magenta, changed);
            grained[2] = grain_quad<2>(sums, tables, frame.pitch, frame.seed, frame.nextSeed, frame.temporalFraction, frame.amounts[2], x + frame.originX, y + frame.originY, yellow, changed);
            if (changed && frame.chromaFraction < 1.F) {
                const float input[kRecords]{cyan, magenta, yellow};
                reduce_chroma(input, grained, frame.chromaFraction);
            }
            planes.cyan[index] = grained[0];
            planes.magenta[index] = grained[1];
            planes.yellow[index] = grained[2];
        }

    } // namespace
} // namespace JuicerCuda::FilmGrain
