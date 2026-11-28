#include "Cuda/Film/FilmLaunchers.h"
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "Cuda/JuicerCudaDeviceHelpers.cuh"

#include "Cuda/JuicerCudaBlur.h"
#include "Cuda/JuicerCudaBlur.cuh"
#include "openrand/philox.h"

using JuicerCuda::reflect_index_half_sample_wide_device;

namespace {

    struct LognormalSampleDevice {
        float mean = 0.0f;
        float standardDeviation = 0.0f;
        float normalSample = 0.0f;
    };

    struct LognormalParametersDevice {
        float mu = 0.0f;
        float sigma = 0.0f;
    };

    struct WangLutEdgesDevice {
        int left = 0;
        int right = 0;
        int top = 0;
        int bottom = 0;
        int colorCount = 0;
    };

    struct NoiseCoordinateDevice {
        float x = 0.0f;
        float y = 0.0f;
    };

    struct GrainRngInitializationDevice {
        std::uint64_t seed = 0;
        std::uint32_t counterX = 0;
        std::uint32_t counterY = 0;
        std::uint64_t absoluteX = 0;
        std::uint64_t absoluteY = 0;
        int useStbn = 0;
        int frameOffset = 0;
        int wangOffsetX = 0;
        int wangOffsetY = 0;
    };

    struct LayerParticleModelSampleDevice {
        float density = 0.0f;
        float densityMaximum = 0.0f;
        float particleCount = 0.0f;
        float particleOpticalDensity = 0.0f;
        float uniformity = 0.0f;
        std::uint64_t seed = 0;
        std::uint64_t absoluteX = 0;
        std::uint64_t absoluteY = 0;
        const JuicerCuda::GrainPayload* grain = nullptr;
        int useStbn = 0;
        int frameOffset = 0;
        int wangOffsetX = 0;
        int wangOffsetY = 0;
    };

    __device__ __forceinline__ float lognormal_from_mean_std_device(LognormalSampleDevice sample) {
        const float m2 = sample.mean * sample.mean;
        const float s2 = sample.standardDeviation * sample.standardDeviation;
        const float sigmaSq = logf(1.0f + (s2 / m2));
        const float sigma = sqrtf(fmaxf(0.0f, sigmaSq));
        const float mu = logf(fmaxf(1e-12f, sample.mean)) - 0.5f * sigmaSq;
        return expf(mu + sigma * sample.normalSample);
    }

    __device__ __forceinline__ LognormalParametersDevice
    lognormal_parameters_from_mean_std_device(LognormalSampleDevice sample) {
        const float m2 = sample.mean * sample.mean;
        const float s2 =
            sample.standardDeviation * sample.standardDeviation;
        const float sigmaSq = logf(1.0f + (s2 / m2));
        LognormalParametersDevice parameters{};
        parameters.sigma = sqrtf(fmaxf(0.0f, sigmaSq));
        parameters.mu =
            logf(fmaxf(1e-12f, sample.mean)) - 0.5f * sigmaSq;
        return parameters;
    }

    __device__ __forceinline__ std::uint64_t splitmix64_device(std::uint64_t v) {
        v += 0x9E3779B97F4A7C15ULL;
        v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ULL;
        v = (v ^ (v >> 27)) * 0x94D049BB133111EBULL;
        return v ^ (v >> 31);
    }

    __device__ __forceinline__ int wrap_stbn_coordinate_device(
        std::int64_t coordinate,
        int size) {
        if (size <= 0) {
            return 0;
        }
        if ((size & (size - 1)) == 0) {
            return static_cast<int>(
                static_cast<std::uint64_t>(coordinate) &
                static_cast<std::uint64_t>(size - 1));
        }
        coordinate %= static_cast<std::int64_t>(size);
        if (coordinate < 0) {
            coordinate += static_cast<std::int64_t>(size);
        }
        return static_cast<int>(coordinate);
    }

    __device__ __forceinline__ float stbn_lookup_active_device(
        const JuicerCuda::GrainPayload& grain,
        int x,
        int y,
        std::size_t frameRowBase) {
        const std::size_t idx = (frameRowBase + static_cast<std::size_t>(y)) *
                                    static_cast<std::size_t>(grain.stbnWidth) +
                                static_cast<std::size_t>(x);
        const std::uint8_t v = grain.stbn[idx];
        return (static_cast<float>(v) + 0.5f) * (1.0f / 256.0f);
    }

    __device__ __forceinline__ float smoothstep_device(float t) {
        return t * t * (3.0f - 2.0f * t);
    }

    __device__ __forceinline__ float hash01_device(int x, int y, std::uint64_t seed) {
        const std::uint64_t h = splitmix64_device(seed ^
                                                  (static_cast<std::uint64_t>(x) * 0x9E3779B97F4A7C15ULL) ^
                                                  (static_cast<std::uint64_t>(y) * 0xBF58476D1CE4E5B9ULL));
        constexpr float kInvU32 = 1.0f / 4294967296.0f;
        return static_cast<float>(static_cast<std::uint32_t>(h & 0xFFFFFFFFu)) * kInvU32;
    }

    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    __device__ __forceinline__ float grain_spatial_hash01_device(int x, int y, std::uint64_t seed) {
        const std::uint32_t seedLow = static_cast<std::uint32_t>(seed);
        const std::uint32_t seedHigh = static_cast<std::uint32_t>(seed >> 32);
        std::uint32_t h = seedLow ^ ((seedHigh << 16) | (seedHigh >> 16));
        h ^= static_cast<std::uint32_t>(x) * 0x9E3779B9u;
        h ^= static_cast<std::uint32_t>(y) * 0x85EBCA6Bu;
        h ^= h >> 16;
        h *= 0x7FEB352Du;
        h ^= h >> 15;
        h *= 0x846CA68Bu;
        h ^= h >> 16;
        constexpr float kInvU32 = 1.0f / 4294967296.0f;
        return static_cast<float>(h) * kInvU32;
    }

    __device__ __forceinline__ std::uint32_t hash_u32_device(int x, int y, std::uint64_t seed, std::uint64_t salt) {
        const std::uint64_t h = splitmix64_device(seed ^ salt ^
                                                  (static_cast<std::uint64_t>(x) * 0x9E3779B97F4A7C15ULL) ^
                                                  (static_cast<std::uint64_t>(y) * 0xBF58476D1CE4E5B9ULL));
        return static_cast<std::uint32_t>(h & 0xFFFFFFFFu);
    }

    __device__ __forceinline__ std::size_t wang_lut_index_device(WangLutEdgesDevice edges) {
        const std::size_t c = static_cast<std::size_t>(edges.colorCount);
        return (((static_cast<std::size_t>(edges.left) * c + static_cast<std::size_t>(edges.right)) * c +
                 static_cast<std::size_t>(edges.top)) *
                    c +
                static_cast<std::size_t>(edges.bottom));
    }

    __device__ __forceinline__ int wang_tile_id_device(const JuicerCuda::GrainPayload& grain, std::int64_t mx, std::int64_t my) {
        if (!grain.wangLut || grain.wangColors <= 0) {
            return -1;
        }
        const int colors = grain.wangColors;
        const std::uint64_t seed = (grain.clipToken != 0) ? grain.clipToken : grain.stbnSessionSeed;
        const std::uint64_t baseSeed = (seed != 0) ? seed : 1ULL;
        const std::uint32_t l = hash_u32_device(static_cast<int>(mx), static_cast<int>(my), baseSeed, 0xA5A5A5A5u) % colors;
        const std::uint32_t r = hash_u32_device(static_cast<int>(mx + 1), static_cast<int>(my), baseSeed, 0x5A5A5A5Au) % colors;
        const std::uint32_t t = hash_u32_device(static_cast<int>(mx), static_cast<int>(my), baseSeed, 0xC3C3C3C3u) % colors;
        const std::uint32_t b = hash_u32_device(static_cast<int>(mx), static_cast<int>(my + 1), baseSeed, 0x3C3C3C3Cu) % colors;
        WangLutEdgesDevice edges{};
        edges.left = static_cast<int>(l);
        edges.right = static_cast<int>(r);
        edges.top = static_cast<int>(t);
        edges.bottom = static_cast<int>(b);
        edges.colorCount = colors;
        const std::size_t idx = wang_lut_index_device(edges);
        return static_cast<int>(grain.wangLut[idx]);
    }

    __device__ __forceinline__ float wang_tile_sample_device(const JuicerCuda::GrainPayload& grain, int tileId, int x, int y) {
        if (!grain.wangTiles || grain.wangWidth <= 0 || grain.wangHeight <= 0 || grain.wangCount <= 0) {
            return 0.0f;
        }
        if (tileId < 0 || tileId >= grain.wangCount) {
            return 0.0f;
        }
        const int w = grain.wangWidth;
        const int h = grain.wangHeight;
        if (x < 0)
            x = 0;
        if (y < 0)
            y = 0;
        if (x >= w)
            x = w - 1;
        if (y >= h)
            y = h - 1;
        const std::size_t idx = (static_cast<std::size_t>(tileId) * static_cast<std::size_t>(h) + static_cast<std::size_t>(y)) *
                                    static_cast<std::size_t>(w) +
                                static_cast<std::size_t>(x);
        const std::uint8_t v = grain.wangTiles[idx];
        return (static_cast<float>(v) + 0.5f) * (1.0f / 256.0f);
    }

    __device__ __forceinline__ void wang_offsets_device(
        const JuicerCuda::GrainPayload& grain,
        std::uint64_t absX,
        std::uint64_t absY,
        int& outX,
        int& outY) {
        outX = 0;
        outY = 0;
        if (!grain.wangTiles || !grain.wangLut || grain.wangWidth <= 0 || grain.wangHeight <= 0 || grain.wangCount <= 0) {
            return;
        }
        if (!(grain.pixelSizeUm > 0.0f) || !(grain.wangCellMm > 0.0f)) {
            return;
        }

        const float xMm = static_cast<float>(absX) * (grain.pixelSizeUm * 0.001f);
        const float yMm = static_cast<float>(absY) * (grain.pixelSizeUm * 0.001f);
        const float invCell = 1.0f / grain.wangCellMm;
        const float cellX = xMm * invCell;
        const float cellY = yMm * invCell;
        const float cellFx = cellX - floorf(cellX);
        const float cellFy = cellY - floorf(cellY);
        const std::int64_t mx = static_cast<std::int64_t>(floorf(cellX));
        const std::int64_t my = static_cast<std::int64_t>(floorf(cellY));

        const int tileId = wang_tile_id_device(grain, mx, my);
        if (tileId < 0) {
            return;
        }

        const int w = grain.wangWidth;
        const int h = grain.wangHeight;
        int tx = static_cast<int>(floorf(cellFx * static_cast<float>(w)));
        int ty = static_cast<int>(floorf(cellFy * static_cast<float>(h)));
        if (tx >= w)
            tx = w - 1;
        if (ty >= h)
            ty = h - 1;
        const float v0 = wang_tile_sample_device(grain, tileId, tx, ty);
        const float v1 = wang_tile_sample_device(grain, tileId, (tx + (w >> 1)) % w, (ty + (h >> 1)) % h);

        const int warpX = (grain.stbnWidth > 0) ? ((grain.stbnWidth / 8) > 4 ? (grain.stbnWidth / 8) : 4) : 0;
        const int warpY = (grain.stbnHeight > 0) ? ((grain.stbnHeight / 8) > 4 ? (grain.stbnHeight / 8) : 4) : 0;
        if (warpX == 0 || warpY == 0) {
            return;
        }
        outX = static_cast<int>(floorf((v0 - 0.5f) * 2.0f * static_cast<float>(warpX)));
        outY = static_cast<int>(floorf((v1 - 0.5f) * 2.0f * static_cast<float>(warpY)));
    }

    __device__ __forceinline__ float value_noise_device(
        NoiseCoordinateDevice coordinate,
        std::uint64_t seed) {
        const int ix = static_cast<int>(floorf(coordinate.x));
        const int iy = static_cast<int>(floorf(coordinate.y));
        const float fx = coordinate.x - static_cast<float>(ix);
        const float fy = coordinate.y - static_cast<float>(iy);
        const float sx = smoothstep_device(fx);
        const float sy = smoothstep_device(fy);

        const float v00 = grain_spatial_hash01_device(ix, iy, seed);
        const float v10 = grain_spatial_hash01_device(ix + 1, iy, seed);
        const float v01 = grain_spatial_hash01_device(ix, iy + 1, seed);
        const float v11 = grain_spatial_hash01_device(ix + 1, iy + 1, seed);

        const float v0 = v00 + (v10 - v00) * sx;
        const float v1 = v01 + (v11 - v01) * sx;
        return v0 + (v1 - v0) * sy;
    }

    __device__ __forceinline__ float grain_breathing_factor_device(
        const JuicerCuda::GrainPayload& grain,
        std::uint64_t absX,
        std::uint64_t absY) {
        const JuicerCuda::GrainFrameUniforms* frame = grain.frameUniforms;
        if (!frame || frame->breathingActive == 0) {
            return 1.0f;
        }

        const float baseX =
            static_cast<float>(absX) + frame->breathingDriftOffsetX;
        const float baseY =
            static_cast<float>(absY) + frame->breathingDriftOffsetY +
            frame->breathingRollOffsetY;

        const float xSmall = baseX / frame->breathingCellSmallPx;
        const float ySmall = baseY / frame->breathingCellSmallPx;
        const float xLarge = baseX / frame->breathingCellLargePx;
        const float yLarge = baseY / frame->breathingCellLargePx;

        NoiseCoordinateDevice smallCoordinate{};
        smallCoordinate.x = xSmall;
        smallCoordinate.y = ySmall;
        NoiseCoordinateDevice largeCoordinate{};
        largeCoordinate.x = xLarge;
        largeCoordinate.y = yLarge;
        const float noiseSmallA = value_noise_device(
            smallCoordinate,
            frame->breathingSeedA ^ 0x9E3779B97F4A7C15ULL);
        const float noiseSmallB = value_noise_device(
            smallCoordinate,
            frame->breathingSeedB ^ 0x9E3779B97F4A7C15ULL);
        const float noiseLargeA = value_noise_device(
            largeCoordinate,
            frame->breathingSeedA ^ 0xBF58476D1CE4E5B9ULL);
        const float noiseLargeB = value_noise_device(
            largeCoordinate,
            frame->breathingSeedB ^ 0xBF58476D1CE4E5B9ULL);

        const float noiseSmall =
            noiseSmallA + (noiseSmallB - noiseSmallA) * frame->breathingT;
        const float noiseLarge =
            noiseLargeA + (noiseLargeB - noiseLargeA) * frame->breathingT;
        const float noise =
            frame->breathingMix * noiseSmall +
            (1.0f - frame->breathingMix) * noiseLarge;

        const float n = noise * 2.0f - 1.0f;
        const float factor = 1.0f + grain.breathingAmplitude * n;
        return (factor > 0.0f) ? factor : 0.0f;
    }

    __device__ __forceinline__ float grain_clump_factor_device(
        const JuicerCuda::GrainPayload& grain,
        std::uint64_t absX,
        std::uint64_t absY) {
        const JuicerCuda::GrainFrameUniforms* frame = grain.frameUniforms;
        if (!frame || frame->clumpActive == 0) {
            return 1.0f;
        }

        const float baseXStatic = static_cast<float>(absX);
        const float baseYStatic =
            static_cast<float>(absY) + frame->clumpRollOffsetY;
        const float xStatic = baseXStatic / frame->clumpCellPx;
        const float yStatic = baseYStatic / frame->clumpCellPx;

        NoiseCoordinateDevice primaryCoordinate{};
        primaryCoordinate.x = xStatic;
        primaryCoordinate.y = yStatic;
        NoiseCoordinateDevice secondaryCoordinate{};
        secondaryCoordinate.x = xStatic + 19.19f;
        secondaryCoordinate.y = yStatic + 7.23f;
        const float u1S = value_noise_device(
            primaryCoordinate,
            frame->clumpStaticSeed ^ 0x9E3779B97F4A7C15ULL);
        const float u2S = value_noise_device(
            secondaryCoordinate,
            frame->clumpStaticSeed ^ 0xBF58476D1CE4E5B9ULL);
        float u1 = fminf(fmaxf(u1S, 1e-6f), 1.0f - 1e-6f);
        float u2 = fminf(fmaxf(u2S, 0.0f), 1.0f);
        const float r = sqrtf(-2.0f * logf(u1));
        constexpr float kTwoPi = 6.28318530717958647692f;
        const float nStatic = r * cosf(kTwoPi * u2);

        float clumpVal =
            expf(frame->clumpMu + frame->clumpSigma * nStatic);
        if (!device_isfinite(clumpVal)) {
            clumpVal = 1.0f;
        }
        clumpVal *= frame->clumpRmsNorm;

        return clumpVal;
    }

    struct GrainRngDevice {
        openrand::Philox rng;
        const JuicerCuda::GrainPayload* grain = nullptr;
        std::uint64_t absX = 0;
        std::uint64_t absY = 0;
        std::uint64_t seed = 0;
        std::uint32_t drawIndex = 0;
        int useStbn = 0;
        int wangOffsetX = 0;
        int wangOffsetY = 0;
        std::int64_t stbnPitchFrameOffset = 0;
        std::size_t stbnFrameRowBase = 0;
        int stbnWidthMask = -1;
        int stbnHeightMask = -1;

        __device__ GrainRngDevice(
            const JuicerCuda::GrainPayload* grain_,
            GrainRngInitializationDevice initialization)
            : rng(
                  initialization.seed,
                  initialization.counterX,
                  openrand::DEFAULT_GLOBAL_SEED,
                  initialization.counterY),
              grain(grain_),
              absX(initialization.absoluteX),
              absY(initialization.absoluteY),
              seed(initialization.seed),
              useStbn(initialization.useStbn),
              wangOffsetX(initialization.wangOffsetX),
              wangOffsetY(initialization.wangOffsetY) {
            if (!useStbn || !grain) {
                useStbn = 0;
                return;
            }
            if ((grain->stbnWidth & (grain->stbnWidth - 1)) == 0 &&
                (grain->stbnHeight & (grain->stbnHeight - 1)) == 0) {
                stbnWidthMask = grain->stbnWidth - 1;
                stbnHeightMask = grain->stbnHeight - 1;
            }
            if (grain->pitchPx > 0) {
                stbnPitchFrameOffset =
                    static_cast<std::int64_t>(grain->pitchPx) *
                    (static_cast<std::int64_t>(grain->frameIndex) +
                     static_cast<std::int64_t>(initialization.frameOffset));
            }
            const int wrappedFrame = wrap_stbn_coordinate_device(
                static_cast<std::int64_t>(grain->stbnFrame) +
                    static_cast<std::int64_t>(initialization.frameOffset),
                grain->stbnFrames);
            stbnFrameRowBase =
                static_cast<std::size_t>(wrappedFrame) *
                static_cast<std::size_t>(grain->stbnHeight);
        }

        __device__ __forceinline__ float uniform() {
            if (useStbn) {
                const std::uint32_t draw = drawIndex++;
                const std::uint64_t h = splitmix64_device(seed + static_cast<std::uint64_t>(draw) * 0x9E3779B97F4A7C15ULL);
                const int offsetX = static_cast<int>(h & 0xFFFFu) + wangOffsetX;
                const int offsetY = static_cast<int>((h >> 16) & 0xFFFFu) + wangOffsetY;
                const std::int64_t xCoordinate =
                    static_cast<std::int64_t>(absX) +
                    static_cast<std::int64_t>(offsetX) +
                    static_cast<std::int64_t>(grain->stbnOffsetX);
                std::int64_t yCoordinate =
                    static_cast<std::int64_t>(absY) +
                    static_cast<std::int64_t>(offsetY) +
                    static_cast<std::int64_t>(grain->stbnOffsetY);
                yCoordinate += stbnPitchFrameOffset;
                int x = 0;
                int y = 0;
                if (stbnWidthMask >= 0) {
                    x = static_cast<int>(
                        static_cast<std::uint64_t>(xCoordinate) &
                        static_cast<std::uint64_t>(stbnWidthMask));
                    y = static_cast<int>(
                        static_cast<std::uint64_t>(yCoordinate) &
                        static_cast<std::uint64_t>(stbnHeightMask));
                } else {
                    x = wrap_stbn_coordinate_device(
                        xCoordinate,
                        grain->stbnWidth);
                    y = wrap_stbn_coordinate_device(
                        yCoordinate,
                        grain->stbnHeight);
                }
                return stbn_lookup_active_device(
                    *grain,
                    x,
                    y,
                    stbnFrameRowBase);
            }
            return rng.rand<float>();
        }

        __device__ __forceinline__ float normal() {
            float u1 = uniform();
            u1 = fminf(fmaxf(u1, 1e-7f), 1.0f);
            const float u2 = uniform();
            const float r = sqrtf(-2.0f * logf(u1));
            constexpr float kTwoPi = 6.28318530717958647692f;
            return r * cosf(kTwoPi * u2);
        }
    };

    __device__ __forceinline__ int fast_poisson_device(
        float lambda,
        GrainRngDevice& rng) {
        if (!device_isfinite(lambda) || !(lambda > 0.0f)) {
            return 0;
        }
        constexpr float kNormalApproximationThreshold = 10.0f;
        if (lambda < kNormalApproximationThreshold) {
            const float L = expf(-lambda);
            float p = 1.0f;
            int k = 0;
            while (p > L && k < 1024) {
                ++k;
                const float u = rng.uniform();
                p *= fminf(fmaxf(u, 1e-7f), 1.0f);
            }
            return k - 1;
        }

        const float z = rng.normal();
        float sample = lambda + sqrtf(lambda) * z;
        if (!device_isfinite(sample)) {
            sample = 0.0f;
        }
        int sampleInt = static_cast<int>(floorf(sample + 0.5f));
        if (sampleInt < 0)
            sampleInt = 0;
        return sampleInt;
    }

    __device__ __forceinline__ int poisson_sample_device(float lambda, GrainRngDevice& rng) {
        return fast_poisson_device(lambda, rng);
    }

    __device__ __forceinline__ float layer_particle_model_device(
        LayerParticleModelSampleDevice sample) {
        if (!sample.grain) {
            return 0.0f;
        }
        float density = sample.density;
        const float densityMax = sample.densityMaximum;
        const float nParticles = sample.particleCount;
        const float odParticle = sample.particleOpticalDensity;
        const float uniformity = sample.uniformity;
        const JuicerCuda::GrainPayload& grain = *sample.grain;
        if (!device_isfinite(density) || density < 0.0f) {
            density = 0.0f;
        }
        float probability = density / densityMax;
        probability = fminf(fmaxf(probability, 1e-6f), 1.0f - 1e-6f);

        float uniform = device_isfinite(uniformity) ? fminf(fmaxf(uniformity, 0.0f), 1.0f) : 0.0f;
        float saturation = 1.0f - probability * uniform * (1.0f - 1e-6f);
        if (!device_isfinite(saturation) || saturation <= 0.0f) {
            saturation = 1e-6f;
        }

        GrainRngInitializationDevice initialization{};
        initialization.seed = sample.seed;
        initialization.counterX = static_cast<std::uint32_t>(sample.absoluteX);
        initialization.counterY = static_cast<std::uint32_t>(sample.absoluteY);
        initialization.absoluteX = sample.absoluteX;
        initialization.absoluteY = sample.absoluteY;
        initialization.useStbn = sample.useStbn;
        initialization.frameOffset = sample.frameOffset;
        initialization.wangOffsetX = sample.wangOffsetX;
        initialization.wangOffsetY = sample.wangOffsetY;
        GrainRngDevice rng(&grain, initialization);
        const float lambda = nParticles / saturation;
        // Poisson thinning collapses the compound particle count into its
        // single-sample marginal without a second random sampler.
        const int grainCount =
            poisson_sample_device(lambda * probability, rng);
        float grainValue = static_cast<float>(grainCount) * odParticle * saturation;
        if (!device_isfinite(grainValue)) {
            grainValue = 0.0f;
        }
        return grainValue;
    }

    __device__ __forceinline__ float interp_density_layer_device(
        float density,
        const float* JUICER_RESTRICT x,
        const float* JUICER_RESTRICT y,
        int n,
        bool positiveFilm,
        bool axisFinite,
        const float* JUICER_RESTRICT axisBlockPrefixMax) {
        if (!x || !y || n <= 0) {
            return density;
        }
        if (!device_isfinite(density)) {
            return y[0];
        }
        const float query = positiveFilm ? -density : density;
        const auto axis_at = [x, positiveFilm](int index) {
            const float axisValue = ldg_f(x + index);
            return positiveFilm ? -axisValue : axisValue;
        };

        const bool useBoundedSearch =
            axisFinite && axisBlockPrefixMax && n == 256;
        int domainBegin = 0;
        int domainEnd = n - 1;
        if (!useBoundedSearch) {
            while (domainBegin < n &&
                   !device_isfinite(axis_at(domainBegin))) {
                ++domainBegin;
            }
            if (domainBegin >= n) {
                return density;
            }
            while (domainEnd > domainBegin &&
                   !device_isfinite(axis_at(domainEnd))) {
                --domainEnd;
            }
        }

        const float xmin = axis_at(domainBegin);
        const float xmax = axis_at(domainEnd);
        if (!device_isfinite(xmin) || !device_isfinite(xmax) || !(xmax >= xmin)) {
            return ldg_f(y + domainBegin);
        }

        if (query <= xmin) {
            return ldg_f(y + domainBegin);
        }
        if (query >= xmax) {
            return ldg_f(y + domainEnd);
        }

        int i1 = domainBegin + 1;
        if (useBoundedSearch) {
            int blockBegin = 0;
            int blockEnd = 15;
#pragma unroll
            for (int step = 0; step < 4; ++step) {
                const int midpoint =
                    blockBegin + (blockEnd - blockBegin) / 2;
                if (axisBlockPrefixMax[midpoint] < query) {
                    blockBegin = midpoint + 1;
                } else {
                    blockEnd = midpoint;
                }
            }
            i1 = max(1, blockBegin * 16);
            const int searchEnd =
                min(domainEnd, (blockBegin + 1) * 16 - 1);
            while (i1 <= searchEnd && axis_at(i1) < query) {
                ++i1;
            }
        } else {
            while (i1 <= domainEnd && axis_at(i1) < query) {
                ++i1;
            }
        }
        if (i1 > domainEnd) {
            return ldg_f(y + domainEnd);
        }

        const int i0 = i1 - 1;
        const float x0 = axis_at(i0);
        const float x1 = axis_at(i1);
        const float y0 = ldg_f(y + i0);
        const float y1 = ldg_f(y + i1);

        const float denom = x1 - x0;
        if (!(denom > 0.0f) || !device_isfinite(denom)) {
            return y0;
        }

        const float t = (query - x0) / denom;
        return y0 + t * (y1 - y0);
    }

} // namespace

__global__ void grain_prepare_frame_uniforms_kernel(
    JuicerCuda::GrainPayload grain) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || !grain.frameUniforms) {
        return;
    }

    JuicerCuda::GrainFrameUniforms frame{};
    const float time = static_cast<float>(grain.frameIndex) + grain.timeAlpha;

    const float breathingAmp = grain.breathingAmplitude;
    const int breathingPeriod = grain.breathingPeriodFrames;
    const float cellSmallUm = grain.breathingCellUmSmall;
    const float cellLargeUm = grain.breathingCellUmLarge;
    if (breathingAmp > 0.0f && breathingPeriod > 0 &&
        grain.pixelSizeUm > 0.0f && cellSmallUm > 0.0f &&
        cellLargeUm > 0.0f) {
        const float periodF = static_cast<float>(breathingPeriod);
        const float stepF = floorf(time / periodF);
        const std::int64_t step = static_cast<std::int64_t>(stepF);
        const float frac = (time - stepF * periodF) / periodF;
        frame.breathingT = smoothstep_device(
            fminf(fmaxf(frac, 0.0f), 1.0f));

        const std::uint64_t seed =
            grain.stbnSessionSeed != 0 ? grain.stbnSessionSeed : 1ULL;
        frame.breathingSeedA = splitmix64_device(
            seed ^ (static_cast<std::uint64_t>(step) *
                    0x8EBC6AF09C88C6E3ULL));
        frame.breathingSeedB = splitmix64_device(
            seed ^ (static_cast<std::uint64_t>(step + 1) *
                    0x8EBC6AF09C88C6E3ULL));

        const float invPixel = 1.0f / grain.pixelSizeUm;
        frame.breathingCellSmallPx =
            fmaxf(4.0f, cellSmallUm * invPixel);
        frame.breathingCellLargePx =
            fmaxf(4.0f, cellLargeUm * invPixel);

        float driftPx = 0.0f;
        if (grain.breathingDriftUmPerFrame > 0.0f) {
            driftPx = grain.breathingDriftUmPerFrame * invPixel;
        }
        float driftX = 0.0f;
        float driftY = 0.0f;
        if (driftPx > 0.0f) {
            const std::uint64_t hash = splitmix64_device(
                seed ^ 0x6A09E667F3BCC909ULL);
            constexpr float kTwoPi = 6.28318530717958647692f;
            constexpr float kInvU32 = 1.0f / 4294967296.0f;
            const float angle =
                static_cast<float>(
                    static_cast<std::uint32_t>(hash & 0xFFFFFFFFu)) *
                kInvU32 * kTwoPi;
            driftX = cosf(angle) * driftPx;
            driftY = sinf(angle) * driftPx;
        }
        const float rollPx = grain.pitchPx > 0
                                 ? static_cast<float>(grain.pitchPx)
                                 : 0.0f;
        frame.breathingDriftOffsetX = driftX * time;
        frame.breathingDriftOffsetY = driftY * time;
        frame.breathingRollOffsetY = rollPx * time;
        frame.breathingMix =
            fminf(fmaxf(grain.breathingMix, 0.0f), 1.0f);
        frame.breathingActive = 1;
    }

    const float stddevSpatial = grain.microStructure[1] * 0.001f;
    if (stddevSpatial > 0.0f && grain.pixelSizeUm > 0.0f) {
        float cellUm = grain.microStructure[0];
        if (!(cellUm > 0.0f)) {
            cellUm = 0.0f;
        }
        const float invPixel = 1.0f / grain.pixelSizeUm;
        frame.clumpCellPx = cellUm * invPixel;
        constexpr float kMinCellPx = 4.0f;
        if (!(frame.clumpCellPx > kMinCellPx)) {
            frame.clumpCellPx = kMinCellPx;
        }

        const float rollPx = grain.pitchPx > 0
                                 ? static_cast<float>(grain.pitchPx)
                                 : 0.0f;
        frame.clumpRollOffsetY = rollPx * time;
        frame.clumpStaticSeed = grain.clipToken != 0
                                    ? grain.clipToken
                                    : grain.stbnSessionSeed;
        if (frame.clumpStaticSeed == 0) {
            frame.clumpStaticSeed = 1ULL;
        }

        float stddev = stddevSpatial;
        frame.clumpRmsNorm = 1.0f;
        const float mix =
            fminf(fmaxf(grain.clumpTemporalMix, 0.0f), 1.0f);
        if (mix > 0.0f) {
            int period = grain.clumpMorphPeriodFrames;
            if (period <= 0) {
                period = 1;
            }
            const float periodF = static_cast<float>(period);
            const float stepF = floorf(time / periodF);
            const std::int64_t step = static_cast<std::int64_t>(stepF);
            const float frac = (time - stepF * periodF) / periodF;
            const float t = smoothstep_device(
                fminf(fmaxf(frac, 0.0f), 1.0f));

            const std::uint64_t temporalSeed =
                grain.stbnSessionSeed != 0
                    ? grain.stbnSessionSeed
                    : 1ULL;
            const std::uint64_t seedA = splitmix64_device(
                temporalSeed ^
                (static_cast<std::uint64_t>(step) *
                 0xD2B74407B1CE6E93ULL));
            const std::uint64_t seedB = splitmix64_device(
                temporalSeed ^
                (static_cast<std::uint64_t>(step + 1) *
                 0xD2B74407B1CE6E93ULL));

            const float u1A = hash01_device(
                0,
                0,
                seedA ^ 0x9E3779B97F4A7C15ULL);
            const float u2A = hash01_device(
                1,
                0,
                seedA ^ 0xBF58476D1CE4E5B9ULL);
            const float u1B = hash01_device(
                0,
                0,
                seedB ^ 0x9E3779B97F4A7C15ULL);
            const float u2B = hash01_device(
                1,
                0,
                seedB ^ 0xBF58476D1CE4E5B9ULL);

            float u1 = u1A + (u1B - u1A) * t;
            float u2 = u2A + (u2B - u2A) * t;
            u1 = fminf(fmaxf(u1, 1e-6f), 1.0f - 1e-6f);
            u2 = fminf(fmaxf(u2, 0.0f), 1.0f);

            constexpr float kTwoPi = 6.28318530717958647692f;
            const float rT = sqrtf(-2.0f * logf(u1));
            const float nTemporal = rT * cosf(kTwoPi * u2);
            constexpr float kClumpStrengthStdScale = 0.23597824f;
            const float strengthStd =
                mix * kClumpStrengthStdScale;
            LognormalSampleDevice strengthSample{};
            strengthSample.mean = 1.0f;
            strengthSample.standardDeviation = strengthStd;
            strengthSample.normalSample = nTemporal;
            float strength =
                lognormal_from_mean_std_device(strengthSample);
            const float strengthNorm =
                rsqrtf(1.0f + strengthStd * strengthStd);
            strength *= strengthNorm;
            stddev = stddevSpatial * strength;
            const float denom = 1.0f + stddev * stddev;
            const float numer =
                1.0f + stddevSpatial * stddevSpatial;
            frame.clumpRmsNorm = denom > 0.0f
                                     ? sqrtf(numer / denom)
                                     : 1.0f;
        }

        LognormalSampleDevice clumpParametersSample{};
        clumpParametersSample.mean = 1.0f;
        clumpParametersSample.standardDeviation = stddev;
        const LognormalParametersDevice clumpParameters =
            lognormal_parameters_from_mean_std_device(
                clumpParametersSample);
        frame.clumpMu = clumpParameters.mu;
        frame.clumpSigma = clumpParameters.sigma;
        frame.clumpActive = 1;
    }

    *grain.frameUniforms = frame;
}


__global__ void grain_clear_kernel(float* out, int n) {
    if (n <= 0) {
        return;
    }
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    if (idx >= static_cast<std::size_t>(n)) {
        return;
    }
    if (!out) {
        return;
    }
    out[idx] = 0.0f;
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters) Focused CUDA kernels use fixed launch bindings.
__global__ void grain_accumulate_kernel(
    float* dst,
    const float* src,
    int n,
    int initialize) {
    if (n <= 0) {
        return;
    }
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    if (idx >= static_cast<std::size_t>(n)) {
        return;
    }
    if (!dst || !src) {
        return;
    }
    const float previous = initialize != 0 ? 0.0f : dst[idx];
    const float v = previous + src[idx];
    dst[idx] = device_isfinite(v) ? v : 0.0f;
}

__global__ void grain_accumulate_weighted_kernel(
    float* dst,
    const float* src,
    int n,
    float weight,
    int initialize) {
    if (n <= 0) {
        return;
    }
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    if (idx >= static_cast<std::size_t>(n) || !dst || !src) {
        return;
    }
    const float w = device_isfinite(weight) ? weight : 0.0f;
    const float previous = initialize != 0 ? 0.0f : dst[idx];
    const float v = previous + w * src[idx];
    dst[idx] = device_isfinite(v) ? v : 0.0f;
}

__global__ void grain_scale_kernel(float* inOut, int n, float scale) {
    if (n <= 0) {
        return;
    }
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    if (idx >= static_cast<std::size_t>(n) || !inOut) {
        return;
    }
    const float s = device_isfinite(scale) ? scale : 0.0f;
    const float v = inOut[idx] * s;
    inOut[idx] = device_isfinite(v) ? v : 0.0f;
}

__global__ void grain_reconstruct_kernel(
    float* inOutMean,
    const float* delta,
    int n,
    float scale,
    int addMean) {
    if (n <= 0) {
        return;
    }
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    if (idx >= static_cast<std::size_t>(n) || !inOutMean || !delta) {
        return;
    }
    const float s = device_isfinite(scale) ? scale : 0.0f;
    const float mean = addMean ? inOutMean[idx] : 0.0f;
    const float v = mean + s * delta[idx];
    inOutMean[idx] = device_isfinite(v) ? v : 0.0f;
}

__global__ void grain_form_delta_kernel(
    float* inOut,
    const float* sub,
    int n,
    float bias) {
    if (n <= 0) {
        return;
    }
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    if (idx >= static_cast<std::size_t>(n) || !inOut || !sub) {
        return;
    }
    const float biasedValue = inOut[idx] + bias;
    const float biased = device_isfinite(biasedValue) ? biasedValue : 0.0f;
    const float v = (biased - sub[idx]) * 1.0f;
    inOut[idx] = device_isfinite(v) ? v : 0.0f;
}

__global__ void grain_subtract_kernel(float* inOut, const float* sub, int n, float amplitude) {
    if (n <= 0) {
        return;
    }
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    if (idx >= static_cast<std::size_t>(n)) {
        return;
    }
    if (!inOut || !sub) {
        return;
    }
    float a = device_isfinite(amplitude) ? amplitude : 1.0f;
    if (a < 0.0f) {
        a = 0.0f;
    }
    const float v = (inOut[idx] - sub[idx]) * a;
    inOut[idx] = device_isfinite(v) ? v : 0.0f;
}
// NOLINTEND(bugprone-easily-swappable-parameters)
__global__ void grain_apply_simple_kernel(
    JuicerCuda::GrainPayload grain,
    int width,
    int height,
    const float* inDensity,
    float* outGrain,
    int channelIndex) {
    if (width <= 0 || height <= 0) {
        return;
    }
    const std::size_t x = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    const std::size_t y = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(blockDim.y) + static_cast<std::size_t>(threadIdx.y);
    if (x >= static_cast<std::size_t>(width) || y >= static_cast<std::size_t>(height)) {
        return;
    }
    if (!inDensity || !outGrain) {
        return;
    }
    if (channelIndex < 0 || channelIndex > 2) {
        return;
    }

    const std::size_t idx = y * static_cast<std::size_t>(width) + x;
    float density = inDensity[idx];

    const float densityMin = grain.densityMin[channelIndex];
    const float densityMax = grain.densityMax[channelIndex];
    const float nParticles = grain.nParticles[channelIndex];
    const float odParticle = grain.odParticle[channelIndex];
    const float uniformity = grain.uniformity[channelIndex];
    const int nSubLayers = (grain.nSubLayers > 0) ? grain.nSubLayers : 1;
    float mixWeight = grain.sizeMixWeight;
    float mixScale = grain.sizeMixScale;
    const float timeAlpha = grain.timeAlpha;
    const bool useRetime = (timeAlpha > 1e-6f && timeAlpha < 0.999999f);
    const bool wantNext = useRetime;

    const std::int64_t absXSigned = static_cast<std::int64_t>(grain.originX) + static_cast<std::int64_t>(x);
    const std::int64_t absYSigned = static_cast<std::int64_t>(grain.originY) + static_cast<std::int64_t>(y);
    const std::uint64_t absX = static_cast<std::uint64_t>(absXSigned);
    const std::uint64_t absY = static_cast<std::uint64_t>(absYSigned);

    density += densityMin;

    const int useStbn = (grain.stbn && grain.stbnWidth > 0 && grain.stbnHeight > 0 && grain.stbnFrames > 0) ? 1 : 0;
    const float clumpFactor = grain_clump_factor_device(grain, absX, absY);
    const float wCoarse = fminf(fmaxf(mixWeight, 0.0f), 1.0f);
    const float wFine = 1.0f - wCoarse;
    const bool useMix = (wCoarse > 0.0f) && (mixScale > 1.0f);
    int wangOffsetX = 0;
    int wangOffsetY = 0;
    if (useStbn) {
        wang_offsets_device(grain, absX, absY, wangOffsetX, wangOffsetY);
    }

    float acc = 0.0f;
    float accNext = 0.0f;
    for (int sl = 0; sl < nSubLayers; ++sl) {
        const std::uint64_t seed = grain.seedBase ^ (static_cast<std::uint64_t>(channelIndex) + static_cast<std::uint64_t>(sl) * 10ULL);
        const std::uint64_t seedNext = grain.seedBaseNext ^ (static_cast<std::uint64_t>(channelIndex) + static_cast<std::uint64_t>(sl) * 10ULL);
        LayerParticleModelSampleDevice sample{};
        sample.density = density;
        sample.densityMaximum = densityMax;
        sample.uniformity = uniformity;
        sample.absoluteX = absX;
        sample.absoluteY = absY;
        sample.grain = &grain;
        sample.useStbn = useStbn;
        sample.wangOffsetX = wangOffsetX;
        sample.wangOffsetY = wangOffsetY;
        if (!useMix) {
            sample.particleCount = nParticles;
            sample.particleOpticalDensity = odParticle;
            sample.seed = seed;
            sample.frameOffset = 0;
            acc += layer_particle_model_device(sample);
            if (wantNext) {
                sample.seed = seedNext;
                sample.frameOffset = 1;
                accNext += layer_particle_model_device(sample);
            }
        } else {
            if (wFine > 0.0f) {
                sample.particleCount = nParticles;
                sample.particleOpticalDensity = odParticle * wFine;
                sample.seed = seed;
                sample.frameOffset = 0;
                acc += layer_particle_model_device(sample);
                if (wantNext) {
                    sample.seed = seedNext;
                    sample.frameOffset = 1;
                    accNext += layer_particle_model_device(sample);
                }
            }
            const float nParticlesCoarse = nParticles / mixScale;
            if (nParticlesCoarse > 0.0f) {
                sample.particleCount = nParticlesCoarse;
                sample.particleOpticalDensity = odParticle * wCoarse * mixScale;
                sample.seed = seed;
                sample.frameOffset = 0;
                acc += layer_particle_model_device(sample);
                if (wantNext) {
                    sample.seed = seedNext;
                    sample.frameOffset = 1;
                    accNext += layer_particle_model_device(sample);
                }
            }
        }
    }
    acc /= static_cast<float>(nSubLayers);
    if (wantNext) {
        accNext /= static_cast<float>(nSubLayers);
    }
    const float breathingFactor = grain_breathing_factor_device(grain, absX, absY);
    acc = density + (acc - density) * breathingFactor * clumpFactor;
    if (wantNext) {
        accNext = density + (accNext - density) * breathingFactor * clumpFactor;
    }
    if (useRetime) {
        const float delta0 = acc - density;
        const float delta1 = accNext - density;
        const float blend = delta0 + (delta1 - delta0) * timeAlpha;
        const float w0 = 1.0f - timeAlpha;
        const float denom = w0 * w0 + timeAlpha * timeAlpha + 1e-6f;
        const float norm = rsqrtf(denom);
        acc = density + blend * norm;
    }
    outGrain[idx] = device_isfinite(acc) ? acc : 0.0f;
}

// Hoisted per-pixel address state must retain the accepted three-block occupancy.
__global__ void __maxnreg__(79) grain_layer_kernel(
    JuicerCuda::GrainPayload grain,
    int width,
    int height,
    const float* inDensity,
    float* outGrain,
    int channelIndex,
    int sublayerIndex) {
    if (width <= 0 || height <= 0) {
        return;
    }
    const std::size_t x = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    const std::size_t y = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(blockDim.y) + static_cast<std::size_t>(threadIdx.y);
    if (x >= static_cast<std::size_t>(width) || y >= static_cast<std::size_t>(height)) {
        return;
    }
    if (!inDensity || !outGrain) {
        return;
    }
    if (channelIndex < 0 || channelIndex > 2) {
        return;
    }
    if (sublayerIndex < 0 || sublayerIndex > 2) {
        return;
    }

    const std::size_t idx = y * static_cast<std::size_t>(width) + x;
    float density = inDensity[idx];

    const float densityMin = grain.densityMinLayers[sublayerIndex][channelIndex];
    const float densityMax = grain.densityMaxLayers[sublayerIndex][channelIndex];
    const float nParticles = grain.nParticlesLayers[sublayerIndex][channelIndex];
    const float odParticle = grain.odParticleLayers[sublayerIndex][channelIndex];
    const float uniformity = grain.uniformity[channelIndex];
    float mixWeight = grain.sizeMixWeight;
    float mixScale = grain.sizeMixScale;
    const float timeAlpha = grain.timeAlpha;
    const bool useRetime = (timeAlpha > 1e-6f && timeAlpha < 0.999999f);
    const bool wantNext = useRetime;

    const std::int64_t absXSigned = static_cast<std::int64_t>(grain.originX) + static_cast<std::int64_t>(x);
    const std::int64_t absYSigned = static_cast<std::int64_t>(grain.originY) + static_cast<std::int64_t>(y);
    const std::uint64_t absX = static_cast<std::uint64_t>(absXSigned);
    const std::uint64_t absY = static_cast<std::uint64_t>(absYSigned);

    const JuicerCuda::DeviceCurveView curve = grain.densityCurveCmy[channelIndex];
    const float* layerCurve = grain.densityCurvesLayers[sublayerIndex][channelIndex];
    density = interp_density_layer_device(
        density,
        curve.y,
        layerCurve,
        curve.n,
        grain.positiveFilm != 0,
        grain.densityLayerAxisFinite[channelIndex] != 0,
        grain.densityLayerAxisBlockPrefixMax[channelIndex]);
    density += densityMin;

    const std::uint64_t seed = grain.seedBase ^ (static_cast<std::uint64_t>(channelIndex) + static_cast<std::uint64_t>(sublayerIndex) * 10ULL);
    const int useStbn = (grain.stbn && grain.stbnWidth > 0 && grain.stbnHeight > 0 && grain.stbnFrames > 0) ? 1 : 0;
    const float clumpFactor = grain_clump_factor_device(grain, absX, absY);
    const float wCoarse = fminf(fmaxf(mixWeight, 0.0f), 1.0f);
    const float wFine = 1.0f - wCoarse;
    const bool useMix = (wCoarse > 0.0f) && (mixScale > 1.0f);
    int wangOffsetX = 0;
    int wangOffsetY = 0;
    if (useStbn) {
        wang_offsets_device(grain, absX, absY, wangOffsetX, wangOffsetY);
    }

    float grainSample = 0.0f;
    float grainSampleNext = 0.0f;
    LayerParticleModelSampleDevice sample{};
    sample.density = density;
    sample.densityMaximum = densityMax;
    sample.uniformity = uniformity;
    sample.absoluteX = absX;
    sample.absoluteY = absY;
    sample.grain = &grain;
    sample.useStbn = useStbn;
    sample.wangOffsetX = wangOffsetX;
    sample.wangOffsetY = wangOffsetY;
    if (!useMix) {
        sample.particleCount = nParticles;
        sample.particleOpticalDensity = odParticle;
        sample.seed = seed;
        sample.frameOffset = 0;
        grainSample = layer_particle_model_device(sample);
        if (wantNext) {
            const std::uint64_t seedNext = grain.seedBaseNext ^ (static_cast<std::uint64_t>(channelIndex) + static_cast<std::uint64_t>(sublayerIndex) * 10ULL);
            sample.seed = seedNext;
            sample.frameOffset = 1;
            grainSampleNext = layer_particle_model_device(sample);
        }
    } else {
        if (wFine > 0.0f) {
            sample.particleCount = nParticles;
            sample.particleOpticalDensity = odParticle * wFine;
            sample.seed = seed;
            sample.frameOffset = 0;
            grainSample += layer_particle_model_device(sample);
            if (wantNext) {
                const std::uint64_t seedNext = grain.seedBaseNext ^ (static_cast<std::uint64_t>(channelIndex) + static_cast<std::uint64_t>(sublayerIndex) * 10ULL);
                sample.seed = seedNext;
                sample.frameOffset = 1;
                grainSampleNext += layer_particle_model_device(sample);
            }
        }
        const float nParticlesCoarse = nParticles / mixScale;
        if (nParticlesCoarse > 0.0f) {
            sample.particleCount = nParticlesCoarse;
            sample.particleOpticalDensity = odParticle * wCoarse * mixScale;
            sample.seed = seed;
            sample.frameOffset = 0;
            grainSample += layer_particle_model_device(sample);
            if (wantNext) {
                const std::uint64_t seedNext = grain.seedBaseNext ^ (static_cast<std::uint64_t>(channelIndex) + static_cast<std::uint64_t>(sublayerIndex) * 10ULL);
                sample.seed = seedNext;
                sample.frameOffset = 1;
                grainSampleNext += layer_particle_model_device(sample);
            }
        }
    }
    const float breathingFactor = grain_breathing_factor_device(grain, absX, absY);
    grainSample = density + (grainSample - density) * breathingFactor * clumpFactor;
    if (wantNext) {
        grainSampleNext = density + (grainSampleNext - density) * breathingFactor * clumpFactor;
    }
    if (useRetime) {
        const float delta0 = grainSample - density;
        const float delta1 = grainSampleNext - density;
        const float blend = delta0 + (delta1 - delta0) * timeAlpha;
        const float w0 = 1.0f - timeAlpha;
        const float denom = w0 * w0 + timeAlpha * timeAlpha + 1e-6f;
        const float norm = rsqrtf(denom);
        grainSample = density + blend * norm;
    }
    outGrain[idx] = device_isfinite(grainSample) ? grainSample : 0.0f;
}

// Exact-equal dye-cloud groups can sum their layers before one shared blur.
__global__ void __maxnreg__(79) grain_layer_triplet_kernel(
    JuicerCuda::GrainPayload grain,
    int width,
    int height,
    const float* inDensity,
    float* outGrain,
    int channelIndex) {
    if (width <= 0 || height <= 0) {
        return;
    }
    const std::size_t x = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    const std::size_t y = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(blockDim.y) + static_cast<std::size_t>(threadIdx.y);
    if (x >= static_cast<std::size_t>(width) || y >= static_cast<std::size_t>(height)) {
        return;
    }
    if (!inDensity || !outGrain) {
        return;
    }
    if (channelIndex < 0 || channelIndex > 2) {
        return;
    }

    const std::size_t idx = y * static_cast<std::size_t>(width) + x;
    const float inputDensity = inDensity[idx];
    const float uniformity = grain.uniformity[channelIndex];
    const float mixWeight = grain.sizeMixWeight;
    const float mixScale = grain.sizeMixScale;
    const float timeAlpha = grain.timeAlpha;
    const bool useRetime = (timeAlpha > 1e-6f && timeAlpha < 0.999999f);
    const bool wantNext = useRetime;

    const std::int64_t absXSigned = static_cast<std::int64_t>(grain.originX) + static_cast<std::int64_t>(x);
    const std::int64_t absYSigned = static_cast<std::int64_t>(grain.originY) + static_cast<std::int64_t>(y);
    const std::uint64_t absX = static_cast<std::uint64_t>(absXSigned);
    const std::uint64_t absY = static_cast<std::uint64_t>(absYSigned);

    const JuicerCuda::DeviceCurveView curve = grain.densityCurveCmy[channelIndex];
    const int useStbn = (grain.stbn && grain.stbnWidth > 0 && grain.stbnHeight > 0 && grain.stbnFrames > 0) ? 1 : 0;
    const float clumpFactor = grain_clump_factor_device(grain, absX, absY);
    const float breathingFactor = grain_breathing_factor_device(grain, absX, absY);
    const float wCoarse = fminf(fmaxf(mixWeight, 0.0f), 1.0f);
    const float wFine = 1.0f - wCoarse;
    const bool useMix = (wCoarse > 0.0f) && (mixScale > 1.0f);
    int wangOffsetX = 0;
    int wangOffsetY = 0;
    if (useStbn) {
        wang_offsets_device(grain, absX, absY, wangOffsetX, wangOffsetY);
    }

    float layerSum = 0.0f;
    for (int sublayerIndex = 0; sublayerIndex < 3; ++sublayerIndex) {
        const float densityMin = grain.densityMinLayers[sublayerIndex][channelIndex];
        const float densityMax = grain.densityMaxLayers[sublayerIndex][channelIndex];
        const float nParticles = grain.nParticlesLayers[sublayerIndex][channelIndex];
        const float odParticle = grain.odParticleLayers[sublayerIndex][channelIndex];
        const float* layerCurve = grain.densityCurvesLayers[sublayerIndex][channelIndex];
        float density = interp_density_layer_device(
            inputDensity,
            curve.y,
            layerCurve,
            curve.n,
            grain.positiveFilm != 0,
            grain.densityLayerAxisFinite[channelIndex] != 0,
            grain.densityLayerAxisBlockPrefixMax[channelIndex]);
        density += densityMin;

        const std::uint64_t seed = grain.seedBase ^ (static_cast<std::uint64_t>(channelIndex) + static_cast<std::uint64_t>(sublayerIndex) * 10ULL);
        float grainSample = 0.0f;
        float grainSampleNext = 0.0f;
        LayerParticleModelSampleDevice sample{};
        sample.density = density;
        sample.densityMaximum = densityMax;
        sample.uniformity = uniformity;
        sample.absoluteX = absX;
        sample.absoluteY = absY;
        sample.grain = &grain;
        sample.useStbn = useStbn;
        sample.wangOffsetX = wangOffsetX;
        sample.wangOffsetY = wangOffsetY;
        if (!useMix) {
            sample.particleCount = nParticles;
            sample.particleOpticalDensity = odParticle;
            sample.seed = seed;
            sample.frameOffset = 0;
            grainSample = layer_particle_model_device(sample);
            if (wantNext) {
                const std::uint64_t seedNext = grain.seedBaseNext ^ (static_cast<std::uint64_t>(channelIndex) + static_cast<std::uint64_t>(sublayerIndex) * 10ULL);
                sample.seed = seedNext;
                sample.frameOffset = 1;
                grainSampleNext = layer_particle_model_device(sample);
            }
        } else {
            if (wFine > 0.0f) {
                sample.particleCount = nParticles;
                sample.particleOpticalDensity = odParticle * wFine;
                sample.seed = seed;
                sample.frameOffset = 0;
                grainSample += layer_particle_model_device(sample);
                if (wantNext) {
                    const std::uint64_t seedNext = grain.seedBaseNext ^ (static_cast<std::uint64_t>(channelIndex) + static_cast<std::uint64_t>(sublayerIndex) * 10ULL);
                    sample.seed = seedNext;
                    sample.frameOffset = 1;
                    grainSampleNext += layer_particle_model_device(sample);
                }
            }
            const float nParticlesCoarse = nParticles / mixScale;
            if (nParticlesCoarse > 0.0f) {
                sample.particleCount = nParticlesCoarse;
                sample.particleOpticalDensity = odParticle * wCoarse * mixScale;
                sample.seed = seed;
                sample.frameOffset = 0;
                grainSample += layer_particle_model_device(sample);
                if (wantNext) {
                    const std::uint64_t seedNext = grain.seedBaseNext ^ (static_cast<std::uint64_t>(channelIndex) + static_cast<std::uint64_t>(sublayerIndex) * 10ULL);
                    sample.seed = seedNext;
                    sample.frameOffset = 1;
                    grainSampleNext += layer_particle_model_device(sample);
                }
            }
        }
        grainSample = density + (grainSample - density) * breathingFactor * clumpFactor;
        if (wantNext) {
            grainSampleNext = density + (grainSampleNext - density) * breathingFactor * clumpFactor;
        }
        if (useRetime) {
            const float delta0 = grainSample - density;
            const float delta1 = grainSampleNext - density;
            const float blend = delta0 + (delta1 - delta0) * timeAlpha;
            const float w0 = 1.0f - timeAlpha;
            const float denom = w0 * w0 + timeAlpha * timeAlpha + 1e-6f;
            const float norm = rsqrtf(denom);
            grainSample = density + blend * norm;
        }
        layerSum += device_isfinite(grainSample) ? grainSample : 0.0f;
    }
    outGrain[idx] = device_isfinite(layerSum) ? layerSum : 0.0f;
}

struct GrainBlurVerticalAccumulateInput {
    const float* horizontalBlur = nullptr;
    float* output = nullptr;
    int width = 0;
    int height = 0;
    const float* kernel = nullptr;
    int radius = 0;
    float weight = 0.0f;
    int initialize = 0;
    const float* densityToSubtract = nullptr;
    float densityBias = 0.0f;
};

template <bool Weighted>
__device__ __forceinline__ void grain_blur_vertical_accumulate_device(
    GrainBlurVerticalAccumulateInput input,
    float* verticalShared) {
    const float* JUICER_RESTRICT in = input.horizontalBlur;
    float* dst = input.output;
    const int width = input.width;
    const int height = input.height;
    const float* JUICER_RESTRICT k = input.kernel;
    const int radius = input.radius;
    const float weight = input.weight;
    const int initialize = input.initialize;
    if (!in || !dst || !k || radius <= 0 || width <= 0 || height <= 0) {
        return;
    }

    const std::size_t radiusCount = static_cast<std::size_t>(radius);
    const std::size_t kLen = 2U * radiusCount + 1U;
    const std::size_t tileW = static_cast<std::size_t>(blockDim.x);
    const std::size_t tileH = static_cast<std::size_t>(blockDim.y) + 2U * radiusCount;

    float* sWeights = verticalShared;
    float* sTile = verticalShared + kLen;

    const std::size_t tid = static_cast<std::size_t>(threadIdx.y) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    const std::size_t tcount = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockDim.y);
    const std::size_t xLocal = static_cast<std::size_t>(threadIdx.x);

    for (std::size_t i = tid; i < kLen; i += tcount) {
        sWeights[i] = k[i];
    }
    __syncthreads();

    const std::size_t widthCount = static_cast<std::size_t>(width);
    const std::size_t heightCount = static_cast<std::size_t>(height);
    const std::size_t blockYStep = static_cast<std::size_t>(blockDim.y) * static_cast<std::size_t>(gridDim.y);
    const std::size_t blockXStep = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    const float finiteWeight = device_isfinite(weight) ? weight : 0.0f;
    for (std::size_t blockY = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(blockDim.y); blockY < heightCount; blockY += blockYStep) {
        for (std::size_t blockX = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x); blockX < widthCount; blockX += blockXStep) {
            const std::size_t x = blockX + static_cast<std::size_t>(threadIdx.x);
            const std::size_t y = blockY + static_cast<std::size_t>(threadIdx.y);
            const bool inBounds = (x < widthCount && y < heightCount);
            const std::size_t xLoad = blockX + xLocal;
            if (xLoad < widthCount) {
                for (std::size_t i = static_cast<std::size_t>(threadIdx.y); i < tileH; i += static_cast<std::size_t>(blockDim.y)) {
                    const std::int64_t yLoad = static_cast<std::int64_t>(blockY) + static_cast<std::int64_t>(i) - static_cast<std::int64_t>(radius);
                    const int yy = reflect_index_half_sample_wide_device(yLoad, height);
                    sTile[i * tileW + xLocal] =
                        in[static_cast<std::size_t>(yy) * widthCount + xLoad];
                }
            }

            __syncthreads();

            if (inBounds) {
                float acc = 0.0f;
                for (std::size_t kernelOffset = 0; kernelOffset < kLen; ++kernelOffset) {
                    const std::size_t tileY = static_cast<std::size_t>(threadIdx.y) + kernelOffset;
                    const float v = sTile[tileY * tileW + xLocal];
                    const float w = sWeights[kernelOffset];
                    acc += v * w;
                }

                const float blurred =
                    (isfinite(acc) && !isnan(acc)) ? acc : 0.0f;
                const std::size_t index = y * widthCount + x;
                const float previous = initialize != 0 ? 0.0f : dst[index];
                float value = previous + blurred;
                if constexpr (Weighted) {
                    value = previous + finiteWeight * blurred;
                }
                value = device_isfinite(value) ? value : 0.0f;
                if constexpr (!Weighted) {
                    if (input.densityToSubtract) {
                        // Preserve the separate accumulation/bias/subtraction
                        // sanitation boundaries when the last layer forms delta.
                        if (input.densityBias != 0.0f) {
                            const float biased = value + input.densityBias;
                            value = device_isfinite(biased) ? biased : 0.0f;
                        }
                        const float delta = value - input.densityToSubtract[index];
                        value = device_isfinite(delta) ? delta : 0.0f;
                    }
                }
                dst[index] = value;
            }

            __syncthreads();
        }
    }
}

__global__ void grain_blur_vertical_accumulate_kernel(
    GrainBlurVerticalAccumulateInput input) {
    extern __shared__ float grainLayerVerticalShared[];
    grain_blur_vertical_accumulate_device<false>(
        input,
        grainLayerVerticalShared);
}

__global__ void grain_blur_vertical_accumulate_weighted_kernel(
    GrainBlurVerticalAccumulateInput input) {
    extern __shared__ float grainWeightedVerticalShared[];
    grain_blur_vertical_accumulate_device<true>(
        input,
        grainWeightedVerticalShared);
}

struct GrainMixSharedInput {
    float* outputDelta = nullptr;
    const float* independentDelta = nullptr;
    const float* sharedDelta = nullptr;
    int count = 0;
    float sharedWeight = 0.0f;
    float independentWeight = 0.0f;
    float amplitude = 0.0f;
};

__global__ void grain_mix_shared_kernel(GrainMixSharedInput input) {
    if (input.count <= 0) {
        return;
    }
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    if (index >= static_cast<std::size_t>(input.count) || !input.outputDelta) {
        return;
    }
    const float sharedWeight = fminf(fmaxf(input.sharedWeight, 0.0f), 1.0f);
    const float independentWeight = fminf(fmaxf(input.independentWeight, 0.0f), 1.0f);
    const float amplitude = input.amplitude;
    const float shared =
        (input.sharedDelta && sharedWeight > 0.0f) ? input.sharedDelta[index] : 0.0f;
    const float independent =
        (input.independentDelta && independentWeight > 0.0f) ? input.independentDelta[index] : 0.0f;
    const float value = amplitude * (sharedWeight * shared + independentWeight * independent);
    input.outputDelta[index] = device_isfinite(value) ? value : 0.0f;
}

struct GrainDebugAverageInput {
    float* outputR = nullptr;
    float* outputG = nullptr;
    float* outputB = nullptr;
    const float* input0 = nullptr;
    const float* input1 = nullptr;
    const float* input2 = nullptr;
    int count = 0;
    float offset = 0.0f;
    float scale = 0.0f;
};

__global__ void grain_debug_encode_avg3_kernel(GrainDebugAverageInput input) {
    if (input.count <= 0) {
        return;
    }
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    if (index >= static_cast<std::size_t>(input.count)) {
        return;
    }
    if (!input.outputR || !input.outputG || !input.outputB ||
        !input.input0 || !input.input1 || !input.input2) {
        return;
    }
    const float scale = device_isfinite(input.scale) ? input.scale : 1.0f;
    const float offset = device_isfinite(input.offset) ? input.offset : 0.0f;
    const float value0 = input.input0[index];
    const float value1 = input.input1[index];
    const float value2 = input.input2[index];
    float average = (value0 + value1 + value2) * (1.0f / 3.0f);
    if (!device_isfinite(average)) {
        average = 0.0f;
    }
    const float output = offset + average * scale;
    input.outputR[index] = output;
    input.outputG[index] = output;
    input.outputB[index] = output;
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters) The focused CUDA ABI and local schedule use fixed named bindings.
extern "C" cudaError_t juicer_cuda_apply_visual_grain(
    const JuicerCuda::GrainPayload* grain,
    const JuicerCuda::GrainKernelPayload* kernels,
    int width,
    int height,
    float* densityC,
    float* densityM,
    float* densityY,
    float* filterTemp,
    float* scaleWork,
    float* deltaAccum,
    float* layerWork,
    float* sharedDelta,
    void* cudaStreamOpaque) {
    if (!grain || !kernels) {
        return cudaErrorInvalidValue;
    }
    if (grain->active == 0) {
        return cudaSuccess;
    }
    if (width <= 0 || height <= 0 ||
        width > std::numeric_limits<int>::max() / height ||
        !densityC || !densityM || !densityY || !filterTemp ||
        !scaleWork || !deltaAccum ||
        (grain->debugView != 6 && !grain->frameUniforms)) {
        return cudaErrorInvalidValue;
    }

    const bool useSublayers = grain->sublayersActive != 0;
    if (useSublayers && !layerWork) {
        return cudaErrorInvalidValue;
    }
    for (int channel = 0; channel < 3; ++channel) {
        if (!std::isfinite(grain->densityMin[channel]) ||
            !std::isfinite(grain->densityMax[channel]) ||
            !(grain->densityMax[channel] > 0.0f) ||
            !std::isfinite(grain->nParticles[channel]) ||
            !(grain->nParticles[channel] > 0.0f) ||
            !std::isfinite(grain->odParticle[channel]) ||
            !(grain->odParticle[channel] > 0.0f)) {
            return cudaErrorInvalidValue;
        }
        if (!useSublayers) {
            continue;
        }
        const JuicerCuda::DeviceCurveView& baseCurve =
            grain->densityCurveCmy[channel];
        if (!baseCurve.x || !baseCurve.y || baseCurve.n <= 0 ||
            baseCurve.domainBegin < 0 ||
            baseCurve.domainEnd < baseCurve.domainBegin ||
            baseCurve.domainEnd >= baseCurve.n) {
            return cudaErrorInvalidValue;
        }
        for (int layer = 0; layer < 3; ++layer) {
            if (!grain->densityCurvesLayers[layer][channel] ||
                !std::isfinite(grain->densityMinLayers[layer][channel]) ||
                !std::isfinite(grain->densityMaxLayers[layer][channel]) ||
                !(grain->densityMaxLayers[layer][channel] > 0.0f) ||
                !std::isfinite(grain->nParticlesLayers[layer][channel]) ||
                !(grain->nParticlesLayers[layer][channel] > 0.0f) ||
                !std::isfinite(grain->odParticleLayers[layer][channel]) ||
                !(grain->odParticleLayers[layer][channel] > 0.0f)) {
                return cudaErrorInvalidValue;
            }
            if (kernels->dyeRadius[layer][channel] > 0 &&
                !kernels->dyeKernel[layer][channel]) {
                return cudaErrorInvalidValue;
            }
        }
    }

    constexpr std::uint64_t kSeedSaltFine = 0xA24BAED4963EE407ULL;
    constexpr std::uint64_t kSeedSaltMid = 0x9F6C1E6B2C4D7A13ULL;
    constexpr std::uint64_t kSeedSaltCoarse = 0x3C79AC492BA7B653ULL;
    constexpr std::uint64_t kSeedSaltShared = 0x7E8A1B9D3F2C65A1ULL;

    const float wFine = grain->sizeMixWeightFine;
    const float wMid = grain->sizeMixWeightMid;
    const float wCoarse = grain->sizeMixWeight;
    const float sizeMixScale = grain->sizeMixScale;
    const float sizeMixGain = grain->sizeMixGain;
    const float amplitude = grain->amplitude;
    if (!std::isfinite(wFine) || !std::isfinite(wMid) ||
        !std::isfinite(wCoarse) || wFine < 0.0f || wMid < 0.0f ||
        wCoarse < 0.0f ||
        std::fabs((wFine + wMid + wCoarse) - 1.0f) > 1e-5f ||
        !std::isfinite(sizeMixScale) || !(sizeMixScale >= 1.0f) ||
        !std::isfinite(sizeMixGain) || !(sizeMixGain > 0.0f) ||
        !std::isfinite(amplitude) || amplitude < 0.0f) {
        return cudaErrorInvalidValue;
    }
    if (wMid > 0.0f &&
        (!kernels->blurKernelMid || kernels->blurRadiusMid <= 0)) {
        return cudaErrorInvalidValue;
    }
    if (wCoarse > 0.0f &&
        (!kernels->blurKernelCoarse || kernels->blurRadiusCoarse <= 0)) {
        return cudaErrorInvalidValue;
    }

    const bool sharedChroma =
        (grain->debugView == 0 || grain->debugView == 1) &&
        grain->chromaMix < 0.999f && grain->chromaSharedWeight > 0.0f;
    if (sharedChroma && !sharedDelta) {
        return cudaErrorInvalidValue;
    }
    if (!std::isfinite(grain->chromaMix) || grain->chromaMix < 0.0f ||
        grain->chromaMix > 1.0f ||
        !std::isfinite(grain->chromaSharedWeight) ||
        grain->chromaSharedWeight < 0.0f ||
        !std::isfinite(grain->chromaIndWeight) ||
        grain->chromaIndWeight < 0.0f || grain->debugView < 0 ||
        grain->debugView > 6) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = cudaStreamOpaque
                              ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque)
                              : nullptr;
    const int total = width * height;
    const int threads1D = 256;
    const int blocks1D = (total + threads1D - 1) / threads1D;
    dim3 threads2D(32, 8);
    dim3 blocks2D(
        (width + threads2D.x - 1) / threads2D.x,
        (height + threads2D.y - 1) / threads2D.y);

    auto launch_grain_blur_horizontal = [&](float* plane,
                                            const float* kernel,
                                            int radius) -> cudaError_t {
        if (!plane || !kernel) {
            return cudaErrorInvalidValue;
        }
        return JuicerCuda::launch_horizontal_blur(
            {plane, filterTemp, width, height, kernel, radius},
            {blocks2D, threads2D, stream},
            JuicerCuda::BlurPrecision::Float);
    };

    auto generate_scale = [&](const JuicerCuda::GrainPayload& base,
                              const float* density,
                              int channel,
                              std::uint64_t salt,
                              float particleFactor,
                              const float* correlationKernel,
                              int correlationRadius,
                              bool applyCorrelation,
                              float weight,
                              bool initializeDeltaAccum) -> cudaError_t {
        JuicerCuda::GrainPayload pass = base;
        pass.seedBase ^= salt;
        pass.seedBaseNext ^= salt;
        pass.sizeMixWeightFine = 1.0f;
        pass.sizeMixWeight = 0.0f;
        pass.sizeMixWeightMid = 0.0f;
        pass.sizeMixScale = 1.0f;
        if (!std::isfinite(particleFactor) || !(particleFactor > 0.0f)) {
            return cudaErrorInvalidValue;
        }
        for (int scaleChannel = 0; scaleChannel < 3; ++scaleChannel) {
            pass.nParticles[scaleChannel] =
                base.nParticles[scaleChannel] / particleFactor;
            pass.odParticle[scaleChannel] =
                base.odParticle[scaleChannel] * particleFactor;
            for (int layer = 0; layer < 3; ++layer) {
                pass.nParticlesLayers[layer][scaleChannel] =
                    base.nParticlesLayers[layer][scaleChannel] /
                    particleFactor;
                pass.odParticleLayers[layer][scaleChannel] =
                    base.odParticleLayers[layer][scaleChannel] *
                    particleFactor;
            }
        }

        cudaError_t error = cudaSuccess;
        const float densityBias = -base.densityMin[channel];
        bool deltaFormed = false;
        if (useSublayers) {
            const int dyeRadius0 = kernels->dyeRadius[0][channel];
            const int dyeRadius1 = kernels->dyeRadius[1][channel];
            const int dyeRadius2 = kernels->dyeRadius[2][channel];
            const float* dyeKernel0 = kernels->dyeKernel[0][channel];
            const float* dyeKernel1 = kernels->dyeKernel[1][channel];
            const float* dyeKernel2 = kernels->dyeKernel[2][channel];
            const bool dyeRadiiEqual =
                dyeRadius0 == dyeRadius1 && dyeRadius0 == dyeRadius2;
            const bool dyeKernelsEqual =
                dyeKernel0 == dyeKernel1 && dyeKernel0 == dyeKernel2;
            const bool dyeKernelPrepared =
                dyeRadius0 == 0 || dyeKernel0 != nullptr;
            const bool zeroDyeKernels =
                !dyeKernel0 && !dyeKernel1 && !dyeKernel2;
            const bool batchDyeTriplet =
                dyeRadiiEqual && dyeKernelsEqual && dyeKernelPrepared &&
                (dyeRadius0 > 0 || zeroDyeKernels);
            if (batchDyeTriplet) {
                grain_layer_triplet_kernel<<<blocks2D, threads2D, 0, stream>>>(
                    pass,
                    width,
                    height,
                    density,
                    layerWork,
                    channel);
                error = cudaGetLastError();
                if (error != cudaSuccess) {
                    return error;
                }
                if (dyeRadius0 > 0) {
                    error = launch_grain_blur_horizontal(
                        layerWork,
                        dyeKernel0,
                        dyeRadius0);
                    if (error != cudaSuccess) {
                        return error;
                    }
                    const int kernelLength = 2 * dyeRadius0 + 1;
                    const size_t verticalSharedBytes =
                        (static_cast<size_t>(kernelLength) +
                         static_cast<size_t>(threads2D.x) *
                             static_cast<size_t>(threads2D.y + 2 * dyeRadius0)) *
                        sizeof(float);
                    GrainBlurVerticalAccumulateInput finishInput{};
                    finishInput.horizontalBlur = filterTemp;
                    finishInput.output = scaleWork;
                    finishInput.width = width;
                    finishInput.height = height;
                    finishInput.kernel = dyeKernel0;
                    finishInput.radius = dyeRadius0;
                    finishInput.initialize = 1;
                    finishInput.densityToSubtract = density;
                    finishInput.densityBias = densityBias;
                    grain_blur_vertical_accumulate_kernel<<<
                        blocks2D,
                        threads2D,
                        verticalSharedBytes,
                        stream>>>(finishInput);
                    error = cudaGetLastError();
                    if (error != cudaSuccess) {
                        return error;
                    }
                    deltaFormed = true;
                } else {
                    grain_accumulate_kernel<<<blocks1D, threads1D, 0, stream>>>(
                        scaleWork,
                        layerWork,
                        total,
                        1);
                    error = cudaGetLastError();
                    if (error != cudaSuccess) {
                        return error;
                    }
                }
            } else {
                for (int layer = 0; layer < 3; ++layer) {
                    grain_layer_kernel<<<blocks2D, threads2D, 0, stream>>>(
                        pass,
                        width,
                        height,
                        density,
                        layerWork,
                        channel,
                        layer);
                    error = cudaGetLastError();
                    if (error != cudaSuccess) {
                        return error;
                    }
                    const float* dyeKernel = kernels->dyeKernel[layer][channel];
                    const int dyeRadius = kernels->dyeRadius[layer][channel];
                    if (dyeRadius > 0) {
                        error = launch_grain_blur_horizontal(
                            layerWork,
                            dyeKernel,
                            dyeRadius);
                        if (error != cudaSuccess) {
                            return error;
                        }
                        const int kernelLength = 2 * dyeRadius + 1;
                        const size_t verticalSharedBytes =
                            (static_cast<size_t>(kernelLength) +
                             static_cast<size_t>(threads2D.x) *
                                 static_cast<size_t>(threads2D.y + 2 * dyeRadius)) *
                            sizeof(float);
                        GrainBlurVerticalAccumulateInput finishInput{};
                        finishInput.horizontalBlur = filterTemp;
                        finishInput.output = scaleWork;
                        finishInput.width = width;
                        finishInput.height = height;
                        finishInput.kernel = dyeKernel;
                        finishInput.radius = dyeRadius;
                        finishInput.initialize = layer == 0 ? 1 : 0;
                        if (layer == 2) {
                            finishInput.densityToSubtract = density;
                            finishInput.densityBias = densityBias;
                        }
                        grain_blur_vertical_accumulate_kernel<<<
                            blocks2D,
                            threads2D,
                            verticalSharedBytes,
                            stream>>>(finishInput);
                        error = cudaGetLastError();
                        if (error != cudaSuccess) {
                            return error;
                        }
                        deltaFormed = layer == 2;
                    } else {
                        grain_accumulate_kernel<<<blocks1D, threads1D, 0, stream>>>(
                            scaleWork,
                            layerWork,
                            total,
                            layer == 0 ? 1 : 0);
                        error = cudaGetLastError();
                        if (error != cudaSuccess) {
                            return error;
                        }
                    }
                }
            }
        } else {
            grain_apply_simple_kernel<<<blocks2D, threads2D, 0, stream>>>(
                pass,
                width,
                height,
                density,
                scaleWork,
                channel);
            error = cudaGetLastError();
            if (error != cudaSuccess) {
                return error;
            }
        }

        if (!deltaFormed && densityBias != 0.0f) {
            grain_form_delta_kernel<<<blocks1D, threads1D, 0, stream>>>(
                scaleWork,
                density,
                total,
                densityBias);
            error = cudaGetLastError();
            if (error != cudaSuccess) {
                return error;
            }
        } else if (!deltaFormed) {
            grain_subtract_kernel<<<blocks1D, threads1D, 0, stream>>>(
                scaleWork,
                density,
                total,
                1.0f);
            error = cudaGetLastError();
            if (error != cudaSuccess) {
                return error;
            }
        }
        if (applyCorrelation && correlationRadius > 0) {
            error = launch_grain_blur_horizontal(
                scaleWork,
                correlationKernel,
                correlationRadius);
            if (error != cudaSuccess) {
                return error;
            }
            const int kernelLength = 2 * correlationRadius + 1;
            const size_t verticalSharedBytes =
                (static_cast<size_t>(kernelLength) +
                 static_cast<size_t>(threads2D.x) *
                     static_cast<size_t>(threads2D.y + 2 * correlationRadius)) *
                sizeof(float);
            GrainBlurVerticalAccumulateInput finishInput{};
            finishInput.horizontalBlur = filterTemp;
            finishInput.output = deltaAccum;
            finishInput.width = width;
            finishInput.height = height;
            finishInput.kernel = correlationKernel;
            finishInput.radius = correlationRadius;
            finishInput.weight = weight;
            finishInput.initialize = initializeDeltaAccum ? 1 : 0;
            grain_blur_vertical_accumulate_weighted_kernel<<<
                blocks2D,
                threads2D,
                verticalSharedBytes,
                stream>>>(finishInput);
            return cudaGetLastError();
        }
        grain_accumulate_weighted_kernel<<<
            blocks1D,
            threads1D,
            0,
            stream>>>(
            deltaAccum,
            scaleWork,
            total,
            weight,
            initializeDeltaAccum ? 1 : 0);
        return cudaGetLastError();
    };

    const float midFactor = std::sqrt(sizeMixScale);
    auto generate_mix = [&](const JuicerCuda::GrainPayload& base,
                            const float* density,
                            int channel) -> cudaError_t {
        cudaError_t error = cudaSuccess;
        bool initializeDeltaAccum = true;
        if (wFine > 0.0f) {
            error = generate_scale(
                base,
                density,
                channel,
                kSeedSaltFine,
                1.0f,
                kernels->blurKernel,
                kernels->blurRadius,
                kernels->blurRadius > 0,
                wFine,
                initializeDeltaAccum);
            if (error != cudaSuccess) {
                return error;
            }
            initializeDeltaAccum = false;
        }
        if (wMid > 0.0f) {
            error = generate_scale(
                base,
                density,
                channel,
                kSeedSaltMid,
                midFactor,
                kernels->blurKernelMid,
                kernels->blurRadiusMid,
                true,
                wMid,
                initializeDeltaAccum);
            if (error != cudaSuccess) {
                return error;
            }
            initializeDeltaAccum = false;
        }
        if (wCoarse > 0.0f) {
            error = generate_scale(
                base,
                density,
                channel,
                kSeedSaltCoarse,
                sizeMixScale,
                kernels->blurKernelCoarse,
                kernels->blurRadiusCoarse,
                true,
                wCoarse,
                initializeDeltaAccum);
            if (error != cudaSuccess) {
                return error;
            }
            initializeDeltaAccum = false;
        }
        grain_scale_kernel<<<blocks1D, threads1D, 0, stream>>>(
            deltaAccum,
            total,
            sizeMixGain);
        return cudaGetLastError();
    };

    float* density[3] = {densityC, densityM, densityY};
    cudaError_t error = cudaSuccess;
    if (grain->debugView == 6) {
        GrainDebugAverageInput debugInput{};
        debugInput.outputR = densityC;
        debugInput.outputG = densityM;
        debugInput.outputB = densityY;
        debugInput.input0 = densityC;
        debugInput.input1 = densityM;
        debugInput.input2 = densityY;
        debugInput.count = total;
        debugInput.offset = 0.0f;
        debugInput.scale = grain->debugScale * 4.0f;
        grain_debug_encode_avg3_kernel<<<blocks1D, threads1D, 0, stream>>>(debugInput);
        return cudaGetLastError();
    }

    grain_prepare_frame_uniforms_kernel<<<1, 1, 0, stream>>>(*grain);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }

    if (grain->debugView >= 2 && grain->debugView <= 5) {
        const bool coarse =
            grain->debugView == 3 || grain->debugView == 5;
        const bool raw = grain->debugView == 4 || grain->debugView == 5;
        for (int channel = 0; channel < 3; ++channel) {
            if (!coarse || wCoarse > 0.0f) {
                error = generate_scale(
                    *grain,
                    density[channel],
                    channel,
                    coarse ? kSeedSaltCoarse : kSeedSaltFine,
                    coarse ? sizeMixScale : 1.0f,
                    coarse ? kernels->blurKernelCoarse
                           : kernels->blurKernel,
                    coarse ? kernels->blurRadiusCoarse
                           : kernels->blurRadius,
                    !raw &&
                        (coarse ? kernels->blurRadiusCoarse > 0
                                : kernels->blurRadius > 0),
                    1.0f,
                    true);
                if (error != cudaSuccess) {
                    return error;
                }
            } else {
                grain_clear_kernel<<<blocks1D, threads1D, 0, stream>>>(
                    deltaAccum,
                    total);
                error = cudaGetLastError();
                if (error != cudaSuccess) {
                    return error;
                }
            }
            grain_reconstruct_kernel<<<blocks1D, threads1D, 0, stream>>>(
                density[channel],
                deltaAccum,
                total,
                amplitude,
                0);
            error = cudaGetLastError();
            if (error != cudaSuccess) {
                return error;
            }
        }
    } else {
        if (sharedChroma) {
            JuicerCuda::GrainPayload shared = *grain;
            shared.seedBase ^= kSeedSaltShared;
            shared.seedBaseNext ^= kSeedSaltShared;
            error = generate_mix(shared, densityM, 1);
            if (error != cudaSuccess) {
                return error;
            }
            error = cudaMemcpyAsync(
                sharedDelta,
                deltaAccum,
                static_cast<size_t>(total) * sizeof(float),
                cudaMemcpyDeviceToDevice,
                stream);
            if (error != cudaSuccess) {
                return error;
            }
        }
        for (int channel = 0; channel < 3; ++channel) {
            if (grain->chromaIndWeight > 0.0f || !sharedChroma) {
                error = generate_mix(*grain, density[channel], channel);
                if (error != cudaSuccess) {
                    return error;
                }
            } else {
                grain_clear_kernel<<<blocks1D, threads1D, 0, stream>>>(
                    deltaAccum,
                    total);
                error = cudaGetLastError();
                if (error != cudaSuccess) {
                    return error;
                }
            }
            if (sharedChroma) {
                GrainMixSharedInput mixInput{};
                mixInput.outputDelta = deltaAccum;
                mixInput.independentDelta = grain->chromaIndWeight > 0.0f ? deltaAccum : nullptr;
                mixInput.sharedDelta = sharedDelta;
                mixInput.count = total;
                mixInput.sharedWeight = grain->chromaSharedWeight;
                mixInput.independentWeight = grain->chromaIndWeight;
                mixInput.amplitude = amplitude;
                grain_mix_shared_kernel<<<blocks1D, threads1D, 0, stream>>>(mixInput);
                error = cudaGetLastError();
                if (error != cudaSuccess) {
                    return error;
                }
            }
            grain_reconstruct_kernel<<<blocks1D, threads1D, 0, stream>>>(
                density[channel],
                deltaAccum,
                total,
                sharedChroma ? 1.0f : amplitude,
                grain->debugView == 0 ? 1 : 0);
            error = cudaGetLastError();
            if (error != cudaSuccess) {
                return error;
            }
        }
    }

    if (grain->debugView != 0) {
        GrainDebugAverageInput debugInput{};
        debugInput.outputR = densityC;
        debugInput.outputG = densityM;
        debugInput.outputB = densityY;
        debugInput.input0 = densityC;
        debugInput.input1 = densityM;
        debugInput.input2 = densityY;
        debugInput.count = total;
        debugInput.offset = 0.5f;
        debugInput.scale = grain->debugScale;
        grain_debug_encode_avg3_kernel<<<blocks1D, threads1D, 0, stream>>>(debugInput);
        return cudaGetLastError();
    }
    return cudaSuccess;
}

// NOLINTEND(bugprone-easily-swappable-parameters)
