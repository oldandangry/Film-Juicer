// Adapted to CUDA for Film-Juicer G1 from Fotufilm's FilmTiles.h and Random.h.
// Copyright 2026 MUAStudio Inc. SPDX-License-Identifier: Apache-2.0
// See fotufilm-notice.md for source identity, retained notice and license.
#include "film_reference.h"
#include "Cuda/JuicerCudaDriver.h"

#include <string>
#include <stdexcept>
#include <cstring>
#include <cmath>

#include <cuda_runtime.h>

namespace GrainPrototype {
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

        __device__ float sum_inside(const float* sums, float x, float y) {
            const int ix = min(static_cast<int>(x), kTileSide - 1);
            const int iy = min(static_cast<int>(y), kTileSide - 1);
            const float ax = x - static_cast<float>(ix);
            const float ay = y - static_cast<float>(iy);
            const float* top = sums + static_cast<std::ptrdiff_t>(iy) * kTileStride + ix;
            const float* bottom = top + kTileStride;
            const float cross = bottom[1] - top[1] - bottom[0] + top[0];
            return top[0] + (top[1] - top[0]) * ax + (bottom[0] - top[0]) * ay + cross * ax * ay;
        }

        // Ordered horizontal and vertical SAT coordinates mirror sum_inside.
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        __device__ float sum_wrapped(const float* sums, float x, float y) {
            const bool wrapX = x >= kTileSide;
            const bool wrapY = y >= kTileSide;
            const float tx = wrapX ? x - kTileSide : x;
            const float ty = wrapY ? y - kTileSide : y;
            float total = sum_inside(sums, tx, ty);
            if (wrapX)
                total += sum_inside(sums, kTileSide, ty);
            if (wrapY)
                total += sum_inside(sums, tx, kTileSide);
            if (wrapX && wrapY)
                total += sums[kTileEntries - 1];
            return total;
        }

        __device__ float density_delta(const float* sums, const FilmBox& box, float area, FilmLevel level) {
            const float integrated = sum_wrapped(sums, box.right, box.bottom) -
                                     sum_wrapped(sums, box.left, box.bottom) -
                                     sum_wrapped(sums, box.right, box.top) +
                                     sum_wrapped(sums, box.left, box.top);
            const float light = level.meanLight + integrated / area;
            return -log10f(fmaxf(light, 1e-9F)) - level.meanDensity;
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

        // One blend and footprint at this film-space center for the chosen lane.
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        __device__ float sample_laying(const float* lowSums, const FilmLevel* levels, float fraction, float footprint, float cx, float cy, std::uint32_t seed, int laying, int record) {
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
                    const float low = density_delta(lowSums, box, area, levels[0]);
                    const float high = density_delta(lowSums + kTileEntries, box, area, levels[1]);
                    value += weight * (low + (high - low) * fraction);
                    normSquared += weight * weight;
                }
            }
            return value / sqrtf(fmaxf(normSquared, 1e-12F));
        }

        // Record index, whole-frame x/y and that record's net input at the pixel.
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        __device__ float sample_record(const float* sums, const FilmTables& tables, FilmFrame frame, int record, int x, int y, float net) {
            const auto& table = tables.records[record];
            const float gross = net + table.offset;
            const float level = __saturatef((gross - table.minimumGross) /
                                            fmaxf(table.maximumGross - table.minimumGross, 1e-6F)) *
                                (kLevels - 1);
            const int low = min(static_cast<int>(level), kLevels - 2);
            const float fraction = level - static_cast<float>(low);
            const auto offset = (static_cast<std::ptrdiff_t>(record) * kLevels + low) * kTileEntries;
            const float* lowSums = sums + offset;
            const float cx = (static_cast<float>(x) + 0.5F) * frame.pitch;
            const float cy = (static_cast<float>(y) + 0.5F) * frame.pitch;
            float sum = 0;
            for (int laying = 0; laying < kLayings; ++laying) {
                const float shift = static_cast<float>(kPlacementBlock * laying / kLayings);
                sum += sample_laying(lowSums, table.levels + low, fraction, tables.footprint, cx + shift, cy + shift, frame.seed, laying, record);
            }
            const float other = 1 - fraction;
            const float kept = other * other + fraction * fraction +
                               2 * fraction * other * table.levels[low].correlation;
            return sum / sqrtf(fmaxf(kept * kLayings, 1e-12F));
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

        __global__ void apply_reference(const float* sums, FilmTables tables, FilmFrame frame, FilmPlanes planes) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            if (x >= frame.width || y >= frame.height)
                return;
            const auto index = static_cast<std::size_t>(y) * static_cast<std::size_t>(frame.stride) +
                               static_cast<std::size_t>(x);
            float* records[] = {planes.cyan, planes.magenta, planes.yellow};
            float input[kRecords];
            float grained[kRecords];
            bool changed = false;
            for (int record = 0; record < kRecords; ++record) {
                const float density = records[record][index];
                input[record] = density;
                const float delta = sample_record(sums, tables, frame, record, x + frame.originX, y + frame.originY, density);
                float blended = delta;
                if (frame.temporalFraction != 0.F) {
                    FilmFrame next = frame;
                    next.seed = frame.nextSeed;
                    const float other = sample_record(sums, tables, next, record, x + frame.originX, y + frame.originY, density);
                    const float t = frame.temporalFraction;
                    const float weight = 1.F - t;
                    blended = (weight * delta + t * other) / sqrtf(weight * weight + t * t);
                }
                grained[record] = density;
                if (blended != 0.F && frame.amounts[record] != 0.F) {
                    grained[record] += frame.amounts[record] * blended;
                    changed = changed || grained[record] != density;
                }
            }
            if (changed && frame.chromaFraction < 1.F)
                reduce_chroma(input, grained, frame.chromaFraction);
            for (int record = 0; record < kRecords; ++record)
                records[record][index] = grained[record];
        }

        // Pitch, seed and pixel coordinates follow the original sampler's operands.
        template <int Record>
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        __device__ float sample_scalar(const float* sums, const FilmTables& tables, float pitch, std::uint32_t seed, int x, int y, float net) {
            const auto& table = tables.records[Record];
            const float gross = net + table.offset;
            const float level = __saturatef((gross - table.minimumGross) /
                                            fmaxf(table.maximumGross - table.minimumGross, 1e-6F)) *
                                (kLevels - 1);
            const int low = min(static_cast<int>(level), kLevels - 2);
            const float fraction = level - static_cast<float>(low);
            const auto offset = (static_cast<std::ptrdiff_t>(Record) * kLevels + low) * kTileEntries;
            const float* lowSums = sums + offset;
            const float cx = (static_cast<float>(x) + 0.5F) * pitch;
            const float cy = (static_cast<float>(y) + 0.5F) * pitch;
            float sum = 0;
            for (int laying = 0; laying < kLayings; ++laying) {
                const float shift = static_cast<float>(kPlacementBlock * laying / kLayings);
                sum += sample_laying(lowSums, table.levels + low, fraction, tables.footprint, cx + shift, cy + shift, seed, laying, Record);
            }
            const float other = 1 - fraction;
            const float kept = other * other + fraction * fraction +
                               2 * fraction * other * table.levels[low].correlation;
            return sum / sqrtf(fmaxf(kept * kLayings, 1e-12F));
        }

        // Seed pair, amount and pixel coordinates keep their frame semantics.
        template <int Record>
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        __device__ float grain_scalar(const float* sums, const FilmTables& tables, float pitch, std::uint32_t seed, std::uint32_t nextSeed, float time, float amount, int x, int y, float density, bool& changed) {
            const float delta = sample_scalar<Record>(sums, tables, pitch, seed, x, y, density);
            float blended = delta;
            if (time != 0.F) {
                const float other = sample_scalar<Record>(sums, tables, pitch, nextSeed, x, y, density);
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

        __global__ void apply_scalar(const float* sums, FilmTables tables, FilmFrame frame, FilmPlanes planes) {
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
            grained[0] = grain_scalar<0>(sums, tables, frame.pitch, frame.seed, frame.nextSeed, frame.temporalFraction, frame.amounts[0], x + frame.originX, y + frame.originY, cyan, changed);
            grained[1] = grain_scalar<1>(sums, tables, frame.pitch, frame.seed, frame.nextSeed, frame.temporalFraction, frame.amounts[1], x + frame.originX, y + frame.originY, magenta, changed);
            grained[2] = grain_scalar<2>(sums, tables, frame.pitch, frame.seed, frame.nextSeed, frame.temporalFraction, frame.amounts[2], x + frame.originX, y + frame.originY, yellow, changed);
            if (changed && frame.chromaFraction < 1.F) {
                const float input[kRecords]{cyan, magenta, yellow};
                reduce_chroma(input, grained, frame.chromaFraction);
            }
            planes.cyan[index] = grained[0];
            planes.magenta[index] = grained[1];
            planes.yellow[index] = grained[2];
        }

        struct LayingGeometry {
            float cx;
            float cy;
            float half;
            int firstX;
            int lastX;
            int firstY;
            int lastY;
        };

        // Center coordinates follow the original x/y ordering.
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        __device__ LayingGeometry laying_geometry(float cx, float cy, float footprint) {
            return {cx, cy, footprint * 0.5F, static_cast<int>(floorf((cx - kSeam) / kPlacementBlock)), static_cast<int>(floorf((cx + kSeam) / kPlacementBlock)), static_cast<int>(floorf((cy - kSeam) / kPlacementBlock)), static_cast<int>(floorf((cy + kSeam) / kPlacementBlock))};
        }

        // The record and laying identify the original independent hash lanes.
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        __device__ float sample_laying_reuse(const float* lowSums, const FilmLevel* levels, float fraction, float footprint, const LayingGeometry& geometry, std::uint32_t seed, int laying, int record) {
            float value = 0;
            float normSquared = 0;
            for (int by = geometry.firstY; by <= geometry.lastY; ++by) {
                const float top = static_cast<float>(by * kPlacementBlock);
                const float weightY = seam_weight(geometry.cy, top);
                for (int bx = geometry.firstX; bx <= geometry.lastX; ++bx) {
                    const float left = static_cast<float>(bx * kPlacementBlock);
                    const float weight = seam_weight(geometry.cx, left) * weightY;
                    if (weight == 0)
                        continue;
                    const float x0 = geometry.cx - geometry.half;
                    const float y0 = geometry.cy - geometry.half;
                    const FilmBox local{x0 - left, (x0 + footprint) - left, y0 - top, (y0 + footprint) - top};
                    const float area = fmaxf((local.right - local.left) * (local.bottom - local.top), 1e-6F);
                    const FilmBox box = place_box(local, placement_hash(bx, by, seed, laying, record));
                    const float low = density_delta(lowSums, box, area, levels[0]);
                    const float high = density_delta(lowSums + kTileEntries, box, area, levels[1]);
                    value += weight * (low + (high - low) * fraction);
                    normSquared += weight * weight;
                }
            }
            return value / sqrtf(fmaxf(normSquared, 1e-12F));
        }

        // Both geometries identify the ordered layings; frame values retain their roles.
        template <int Record>
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        __device__ float grain_reuse(const float* sums, const FilmTables& tables, const LayingGeometry& first, const LayingGeometry& second, std::uint32_t seed, std::uint32_t nextSeed, float time, float amount, float density, bool& changed) {
            const auto& table = tables.records[Record];
            const float gross = density + table.offset;
            const float level = __saturatef((gross - table.minimumGross) /
                                            fmaxf(table.maximumGross - table.minimumGross, 1e-6F)) *
                                (kLevels - 1);
            const int low = min(static_cast<int>(level), kLevels - 2);
            const float fraction = level - static_cast<float>(low);
            const auto offset = (static_cast<std::ptrdiff_t>(Record) * kLevels + low) * kTileEntries;
            const float* lowSums = sums + offset;
            float sum = 0;
            sum += sample_laying_reuse(lowSums, table.levels + low, fraction, tables.footprint, first, seed, 0, Record);
            sum += sample_laying_reuse(lowSums, table.levels + low, fraction, tables.footprint, second, seed, 1, Record);
            const float other = 1 - fraction;
            const float kept = other * other + fraction * fraction + 2 * fraction * other * table.levels[low].correlation;
            const float normalization = sqrtf(fmaxf(kept * kLayings, 1e-12F));
            const float delta = sum / normalization;
            float blended = delta;
            if (time != 0.F) {
                float next = 0;
                next += sample_laying_reuse(lowSums, table.levels + low, fraction, tables.footprint, first, nextSeed, 0, Record);
                next += sample_laying_reuse(lowSums, table.levels + low, fraction, tables.footprint, second, nextSeed, 1, Record);
                const float nextDelta = next / normalization;
                const float t = time;
                const float weight = 1.F - t;
                blended = (weight * delta + t * nextDelta) / sqrtf(weight * weight + t * t);
            }
            float grained = density;
            if (blended != 0.F && amount != 0.F) {
                grained += amount * blended;
                changed = changed || grained != density;
            }
            return grained;
        }

        __global__ void apply_reuse(const float* sums, FilmTables tables, FilmFrame frame, FilmPlanes planes) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            if (x >= frame.width || y >= frame.height)
                return;
            const auto index = static_cast<std::size_t>(y) * static_cast<std::size_t>(frame.stride) + static_cast<std::size_t>(x);
            const float cx = (static_cast<float>(x + frame.originX) + 0.5F) * frame.pitch;
            const float cy = (static_cast<float>(y + frame.originY) + 0.5F) * frame.pitch;
            const auto first = laying_geometry(cx + 0.F, cy + 0.F, tables.footprint);
            const auto second = laying_geometry(cx + static_cast<float>(kPlacementBlock / kLayings), cy + static_cast<float>(kPlacementBlock / kLayings), tables.footprint);
            const float cyan = planes.cyan[index];
            const float magenta = planes.magenta[index];
            const float yellow = planes.yellow[index];
            bool changed = false;
            float grained[kRecords];
            grained[0] = grain_reuse<0>(sums, tables, first, second, frame.seed, frame.nextSeed, frame.temporalFraction, frame.amounts[0], cyan, changed);
            grained[1] = grain_reuse<1>(sums, tables, first, second, frame.seed, frame.nextSeed, frame.temporalFraction, frame.amounts[1], magenta, changed);
            grained[2] = grain_reuse<2>(sums, tables, first, second, frame.seed, frame.nextSeed, frame.temporalFraction, frame.amounts[2], yellow, changed);
            if (changed && frame.chromaFraction < 1.F) {
                const float input[kRecords]{cyan, magenta, yellow};
                reduce_chroma(input, grained, frame.chromaFraction);
            }
            planes.cyan[index] = grained[0];
            planes.magenta[index] = grained[1];
            planes.yellow[index] = grained[2];
        }

        struct LevelPair {
            float low;
            float high;
        };

        // Corresponding words for the two immutable levels share a coordinate.
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        __device__ LevelPair sum_inside_paired(const float* low, const float* high, float x, float y) {
            const int ix = min(static_cast<int>(x), kTileSide - 1);
            const int iy = min(static_cast<int>(y), kTileSide - 1);
            const float ax = x - static_cast<float>(ix);
            const float ay = y - static_cast<float>(iy);
            const auto offset = static_cast<std::ptrdiff_t>(iy) * kTileStride + ix;
            const float* lowTop = low + offset;
            const float* lowBottom = lowTop + kTileStride;
            const float* highTop = high + offset;
            const float* highBottom = highTop + kTileStride;
            const float l00 = lowTop[0];
            const float l10 = lowTop[1];
            const float l01 = lowBottom[0];
            const float l11 = lowBottom[1];
            const float h00 = highTop[0];
            const float h10 = highTop[1];
            const float h01 = highBottom[0];
            const float h11 = highBottom[1];
            const float lowCross = l11 - l10 - l01 + l00;
            const float lowValue = l00 + (l10 - l00) * ax + (l01 - l00) * ay + lowCross * ax * ay;
            const float highCross = h11 - h10 - h01 + h00;
            const float highValue = h00 + (h10 - h00) * ax + (h01 - h00) * ay + highCross * ax * ay;
            return {lowValue, highValue};
        }

        // Preserve each level's wrap additions and terminal word in their original order.
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        __device__ LevelPair sum_wrapped_paired(const float* low, const float* high, float x, float y) {
            const bool wrapX = x >= kTileSide;
            const bool wrapY = y >= kTileSide;
            const float tx = wrapX ? x - kTileSide : x;
            const float ty = wrapY ? y - kTileSide : y;
            auto total = sum_inside_paired(low, high, tx, ty);
            if (wrapX) {
                const auto edge = sum_inside_paired(low, high, kTileSide, ty);
                total.low += edge.low;
                total.high += edge.high;
            }
            if (wrapY) {
                const auto edge = sum_inside_paired(low, high, tx, kTileSide);
                total.low += edge.low;
                total.high += edge.high;
            }
            if (wrapX && wrapY) {
                total.low += low[kTileEntries - 1];
                total.high += high[kTileEntries - 1];
            }
            return total;
        }

        __device__ LevelPair density_delta_paired(const float* low, const FilmBox& box, float area, const FilmLevel* levels) {
            const float* high = low + kTileEntries;
            auto integrated = sum_wrapped_paired(low, high, box.right, box.bottom);
            auto corner = sum_wrapped_paired(low, high, box.left, box.bottom);
            integrated.low -= corner.low;
            integrated.high -= corner.high;
            corner = sum_wrapped_paired(low, high, box.right, box.top);
            integrated.low -= corner.low;
            integrated.high -= corner.high;
            corner = sum_wrapped_paired(low, high, box.left, box.top);
            integrated.low += corner.low;
            integrated.high += corner.high;
            const float lowLight = levels[0].meanLight + integrated.low / area;
            const float lowDelta = -log10f(fmaxf(lowLight, 1e-9F)) - levels[0].meanDensity;
            const float highLight = levels[1].meanLight + integrated.high / area;
            const float highDelta = -log10f(fmaxf(highLight, 1e-9F)) - levels[1].meanDensity;
            return {lowDelta, highDelta};
        }

        // One blend and footprint at this film-space center for the chosen lane.
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        __device__ float sample_laying_paired(const float* lowSums, const FilmLevel* levels, float fraction, float footprint, float cx, float cy, std::uint32_t seed, int laying, int record) {
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
                    const auto delta = density_delta_paired(lowSums, box, area, levels);
                    value += weight * (delta.low + (delta.high - delta.low) * fraction);
                    normSquared += weight * weight;
                }
            }
            return value / sqrtf(fmaxf(normSquared, 1e-12F));
        }

        // Pitch, seed and pixel coordinates follow the original sampler's operands.
        template <int Record>
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        __device__ float sample_paired(const float* sums, const FilmTables& tables, float pitch, std::uint32_t seed, int x, int y, float net) {
            const auto& table = tables.records[Record];
            const float gross = net + table.offset;
            const float level = __saturatef((gross - table.minimumGross) /
                                            fmaxf(table.maximumGross - table.minimumGross, 1e-6F)) *
                                (kLevels - 1);
            const int low = min(static_cast<int>(level), kLevels - 2);
            const float fraction = level - static_cast<float>(low);
            const auto offset = (static_cast<std::ptrdiff_t>(Record) * kLevels + low) * kTileEntries;
            const float* lowSums = sums + offset;
            const float cx = (static_cast<float>(x) + 0.5F) * pitch;
            const float cy = (static_cast<float>(y) + 0.5F) * pitch;
            float sum = 0;
            for (int laying = 0; laying < kLayings; ++laying) {
                const float shift = static_cast<float>(kPlacementBlock * laying / kLayings);
                sum += sample_laying_paired(lowSums, table.levels + low, fraction, tables.footprint, cx + shift, cy + shift, seed, laying, Record);
            }
            const float other = 1 - fraction;
            const float kept = other * other + fraction * fraction +
                               2 * fraction * other * table.levels[low].correlation;
            return sum / sqrtf(fmaxf(kept * kLayings, 1e-12F));
        }

        // Seed pair, amount and pixel coordinates keep their frame semantics.
        template <int Record>
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        __device__ float grain_paired(const float* sums, const FilmTables& tables, float pitch, std::uint32_t seed, std::uint32_t nextSeed, float time, float amount, int x, int y, float density, bool& changed) {
            const float delta = sample_paired<Record>(sums, tables, pitch, seed, x, y, density);
            float blended = delta;
            if (time != 0.F) {
                const float other = sample_paired<Record>(sums, tables, pitch, nextSeed, x, y, density);
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

        __global__ void apply_paired(const float* sums, FilmTables tables, FilmFrame frame, FilmPlanes planes) {
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
            grained[0] = grain_paired<0>(sums, tables, frame.pitch, frame.seed, frame.nextSeed, frame.temporalFraction, frame.amounts[0], x + frame.originX, y + frame.originY, cyan, changed);
            grained[1] = grain_paired<1>(sums, tables, frame.pitch, frame.seed, frame.nextSeed, frame.temporalFraction, frame.amounts[1], x + frame.originX, y + frame.originY, magenta, changed);
            grained[2] = grain_paired<2>(sums, tables, frame.pitch, frame.seed, frame.nextSeed, frame.temporalFraction, frame.amounts[2], x + frame.originX, y + frame.originY, yellow, changed);
            if (changed && frame.chromaFraction < 1.F) {
                const float input[kRecords]{cyan, magenta, yellow};
                reduce_chroma(input, grained, frame.chromaFraction);
            }
            planes.cyan[index] = grained[0];
            planes.magenta[index] = grained[1];
            planes.yellow[index] = grained[2];
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

        bool supported_pitch(float pitch) {
            return std::isfinite(pitch) && ((pitch >= 0.25F && pitch <= 37.5F) || pitch == 128.F);
        }

        bool supported_footprint(float width) {
            return std::isfinite(width) && width >= 0.25F && width <= 37.5F;
        }

        bool supported_axis(int origin, int size) {
            constexpr int limit = 4096;
            return origin >= -limit && origin <= limit && size > 0 && size <= limit - origin + 1;
        }
    } // namespace

    std::vector<FilmQuad> pack_film_quads(const FilmBank& bank) {
        if (bank.sums.size() != kBankEntries)
            throw std::invalid_argument("invalid Film bank for quad packing");
        std::vector<FilmQuad> quads(kQuadEntries);
        for (int record = 0; record < kRecords; ++record)
            for (int level = 0; level < kLevels; ++level) {
                const auto levelIndex = static_cast<std::size_t>(record) * kLevels + static_cast<std::size_t>(level);
                const auto* source = bank.sums.data() + levelIndex * kTileEntries;
                auto* target = quads.data() + levelIndex * kTileSide * kTileSide;
                for (int y = 0; y < kTileSide; ++y)
                    for (int x = 0; x < kTileSide; ++x) {
                        const auto at = static_cast<std::size_t>(y) * kTileStride + static_cast<std::size_t>(x);
                        auto& quad = target[y * kTileSide + x];
                        std::memcpy(&quad.topLeft, source + at, sizeof(float));
                        std::memcpy(&quad.topRight, source + at + 1, sizeof(float));
                        std::memcpy(&quad.bottomLeft, source + at + kTileStride, sizeof(float));
                        std::memcpy(&quad.bottomRight, source + at + kTileStride + 1, sizeof(float));
                    }
            }
        return quads;
    }

    DeviceFilmQuads::~DeviceFilmQuads() {
        if (quads_ != nullptr)
            static_cast<void>(cudaFree(quads_));
    }

    cudaError_t DeviceFilmQuads::upload(const std::vector<FilmQuad>& quads) {
        if (resident() || quads.size() != kQuadEntries)
            return cudaErrorInvalidValue;
        void* context = nullptr;
        std::string diagnostic;
        if (!JuicerCuda::query_current_cuda_context(context, diagnostic))
            return cudaErrorInvalidResourceHandle;
        FilmQuad* device = nullptr;
        auto status = cudaMalloc(&device, kQuadEntries * sizeof(FilmQuad));
        if (status != cudaSuccess)
            return status;
        status = cudaMemcpy(device, quads.data(), kQuadEntries * sizeof(FilmQuad), cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            static_cast<void>(cudaFree(device));
            return status;
        }
        quads_ = device;
        context_ = context;
        return cudaSuccess;
    }

    DeviceFilmBank::~DeviceFilmBank() {
        if (sums_ != nullptr)
            static_cast<void>(cudaFree(sums_));
    }

    cudaError_t DeviceFilmBank::upload(const FilmBank& bank) {
        if (resident() || bank.sums.size() != kBankEntries || !supported_footprint(bank.tables.footprint))
            return cudaErrorInvalidValue;
        void* context = nullptr;
        std::string diagnostic;
        if (!JuicerCuda::query_current_cuda_context(context, diagnostic))
            return cudaErrorInvalidResourceHandle;
        float* sums = nullptr;
        auto status = cudaMalloc(&sums, kBankEntries * sizeof(float));
        if (status != cudaSuccess)
            return status;
        status = cudaMemcpy(sums, bank.sums.data(), kBankEntries * sizeof(float), cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            static_cast<void>(cudaFree(sums));
            return status;
        }
        sums_ = sums;
        context_ = context;
        tables_ = bank.tables;
        return cudaSuccess;
    }

    cudaError_t apply_film_reference(const DeviceFilmBank& bank, FilmFrame frame, FilmPlanes planes, cudaStream_t stream) {
        return apply_film_reference(bank, bank.tables(), frame, planes, stream);
    }

    cudaError_t apply_film_reference(const DeviceFilmBank& bank, const FilmTables& tables, FilmFrame frame, FilmPlanes planes, cudaStream_t stream) {
        if (!frame.enabled)
            return cudaSuccess;
        if (!bank.resident() || !supported_footprint(tables.footprint) ||
            !std::isfinite(frame.temporalFraction) || frame.temporalFraction < 0.F || frame.temporalFraction >= 1.F ||
            !std::isfinite(frame.chromaFraction) || frame.chromaFraction < 0.F || frame.chromaFraction > 1.F ||
            !std::isfinite(frame.amounts[0]) || !std::isfinite(frame.amounts[1]) || !std::isfinite(frame.amounts[2]) ||
            frame.amounts[0] < 0.F || frame.amounts[1] < 0.F || frame.amounts[2] < 0.F || !supported_pitch(frame.pitch) || !supported_axis(frame.originX, frame.width) ||
            !supported_axis(frame.originY, frame.height) || frame.stride < frame.width || planes.cyan == nullptr ||
            planes.magenta == nullptr || planes.yellow == nullptr)
            return cudaErrorInvalidValue;
        void* context = nullptr;
        std::string diagnostic;
        if (!JuicerCuda::query_current_cuda_context(context, diagnostic) || context != bank.context())
            return cudaErrorInvalidResourceHandle;
        constexpr unsigned int blockX = 32;
        constexpr unsigned int blockY = 4;
        const dim3 blocks{(static_cast<unsigned int>(frame.width) + blockX - 1) / blockX,
                          (static_cast<unsigned int>(frame.height) + blockY - 1) / blockY};
        apply_reference<<<blocks, dim3{blockX, blockY}, 0, stream>>>(bank.sums(), tables, frame, planes);
        return cudaGetLastError();
    }
    cudaError_t apply_film_variant(const DeviceFilmBank& bank, const FilmTables& tables, FilmFrame frame, FilmPlanes planes, cudaStream_t stream, FilmVariant variant, FilmBlock block) {
        if (variant == FilmVariant::Reference && block == FilmBlock::Block32x4)
            return apply_film_reference(bank, tables, frame, planes, stream);
        if ((variant != FilmVariant::Reference && variant != FilmVariant::Scalar && variant != FilmVariant::Reuse && variant != FilmVariant::Paired) ||
            (block != FilmBlock::Block32x4 && block != FilmBlock::Block16x8 && block != FilmBlock::Block32x8))
            return cudaErrorInvalidValue;
        if (!frame.enabled)
            return cudaSuccess;
        if (!bank.resident() || !supported_footprint(tables.footprint) ||
            !std::isfinite(frame.temporalFraction) || frame.temporalFraction < 0.F || frame.temporalFraction >= 1.F ||
            !std::isfinite(frame.chromaFraction) || frame.chromaFraction < 0.F || frame.chromaFraction > 1.F ||
            !std::isfinite(frame.amounts[0]) || !std::isfinite(frame.amounts[1]) || !std::isfinite(frame.amounts[2]) ||
            frame.amounts[0] < 0.F || frame.amounts[1] < 0.F || frame.amounts[2] < 0.F || !supported_pitch(frame.pitch) || !supported_axis(frame.originX, frame.width) ||
            !supported_axis(frame.originY, frame.height) || frame.stride < frame.width || planes.cyan == nullptr || planes.magenta == nullptr || planes.yellow == nullptr)
            return cudaErrorInvalidValue;
        void* context = nullptr;
        std::string diagnostic;
        if (!JuicerCuda::query_current_cuda_context(context, diagnostic) || context != bank.context())
            return cudaErrorInvalidResourceHandle;
        const unsigned int blockX = block == FilmBlock::Block16x8 ? 16U : 32U;
        const unsigned int blockY = block == FilmBlock::Block32x4 ? 4U : 8U;
        const dim3 blocks{(static_cast<unsigned int>(frame.width) + blockX - 1) / blockX,
                          (static_cast<unsigned int>(frame.height) + blockY - 1) / blockY};
        if (variant == FilmVariant::Reference)
            apply_reference<<<blocks, dim3{blockX, blockY}, 0, stream>>>(bank.sums(), tables, frame, planes);
        else if (variant == FilmVariant::Scalar)
            apply_scalar<<<blocks, dim3{blockX, blockY}, 0, stream>>>(bank.sums(), tables, frame, planes);
        else if (variant == FilmVariant::Reuse)
            apply_reuse<<<blocks, dim3{blockX, blockY}, 0, stream>>>(bank.sums(), tables, frame, planes);
        else
            apply_paired<<<blocks, dim3{blockX, blockY}, 0, stream>>>(bank.sums(), tables, frame, planes);
        return cudaGetLastError();
    }
    cudaError_t apply_film_quads(const DeviceFilmQuads& bank, const FilmTables& tables, FilmFrame frame, FilmPlanes planes, cudaStream_t stream, FilmBlock block) {
        if (block != FilmBlock::Block32x4 && block != FilmBlock::Block16x8 && block != FilmBlock::Block32x8)
            return cudaErrorInvalidValue;
        if (!frame.enabled)
            return cudaSuccess;
        if (!bank.resident() || !supported_footprint(tables.footprint) ||
            !std::isfinite(frame.temporalFraction) || frame.temporalFraction < 0.F || frame.temporalFraction >= 1.F ||
            !std::isfinite(frame.chromaFraction) || frame.chromaFraction < 0.F || frame.chromaFraction > 1.F ||
            !std::isfinite(frame.amounts[0]) || !std::isfinite(frame.amounts[1]) || !std::isfinite(frame.amounts[2]) ||
            frame.amounts[0] < 0.F || frame.amounts[1] < 0.F || frame.amounts[2] < 0.F || !supported_pitch(frame.pitch) || !supported_axis(frame.originX, frame.width) ||
            !supported_axis(frame.originY, frame.height) || frame.stride < frame.width || planes.cyan == nullptr || planes.magenta == nullptr || planes.yellow == nullptr)
            return cudaErrorInvalidValue;
        void* context = nullptr;
        std::string diagnostic;
        if (!JuicerCuda::query_current_cuda_context(context, diagnostic) || context != bank.context())
            return cudaErrorInvalidResourceHandle;
        const unsigned int blockX = block == FilmBlock::Block16x8 ? 16U : 32U;
        const unsigned int blockY = block == FilmBlock::Block32x4 ? 4U : 8U;
        const dim3 blocks{(static_cast<unsigned int>(frame.width) + blockX - 1) / blockX,
                          (static_cast<unsigned int>(frame.height) + blockY - 1) / blockY};
        apply_quad<<<blocks, dim3{blockX, blockY}, 0, stream>>>(bank.quads(), tables, frame, planes);
        return cudaGetLastError();
    }
} // namespace GrainPrototype
