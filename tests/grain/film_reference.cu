// Adapted to CUDA for Film-Juicer G1 from Fotufilm's FilmTiles.h and Random.h.
// Copyright 2026 MUAStudio Inc. SPDX-License-Identifier: Apache-2.0
// See fotufilm-notice.md for source identity, retained notice and license.
#include "film_reference.h"
#include "Cuda/JuicerCudaDriver.h"

#include <string>

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
            const float area = footprint * footprint;
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
                    // Admitted footprints fit entirely in every contributing placement.
                    const FilmBox local{cx - half - left, cx + half - left, cy - half - top, cy + half - top};
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

        __global__ void apply_reference(const float* sums, FilmTables tables, FilmFrame frame, FilmPlanes planes) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            if (x >= frame.width || y >= frame.height)
                return;
            const auto index = static_cast<std::size_t>(y) * static_cast<std::size_t>(frame.stride) +
                               static_cast<std::size_t>(x);
            float* records[] = {planes.cyan, planes.magenta, planes.yellow};
            // With all G1 controls at 1, the pinned mix is 1 own, 0 shared.
            for (int record = 0; record < kRecords; ++record) {
                float& density = records[record][index];
                const float delta = sample_record(sums, tables, frame, record, x + frame.originX, y + frame.originY, density);
                if (delta != 0)
                    density += delta;
            }
        }

        bool supported_pitch(float pitch) {
            return pitch == 1.F || pitch == 9.375F || pitch == 32.F || pitch == 128.F;
        }

        bool supported_footprint(float width) {
            return width == 1.F || width == 9.375F || width == 32.F;
        }

        bool supported_axis(int origin, int size) {
            constexpr int limit = 4096;
            return origin >= -limit && origin <= limit && size > 0 && size <= limit - origin + 1;
        }
    } // namespace

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
        if (!frame.enabled)
            return cudaSuccess;
        if (!bank.resident() || !supported_pitch(frame.pitch) || !supported_axis(frame.originX, frame.width) ||
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
        apply_reference<<<blocks, dim3{blockX, blockY}, 0, stream>>>(bank.sums(), bank.tables(), frame, planes);
        return cudaGetLastError();
    }
} // namespace GrainPrototype
