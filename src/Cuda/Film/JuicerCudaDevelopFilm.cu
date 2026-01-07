// Cuda/Film/JuicerCudaDevelopFilm.cu
// Stage-aligned CUDA TU for film development kernels.
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "Cuda/JuicerCudaPayloads.h"
#include "Cuda/JuicerCudaKernelsUtil.cuh"
#include "Cuda/Film/JuicerCudaFilmExposure.cuh"
#include "Cuda/Film/JuicerCudaFilmDevelop.cuh"
#include "openrand/philox.h"

namespace {

    __device__ __forceinline__ float lognormal_from_mean_std_device(float mean, float stddev, float normalSample) {
        const float m2 = mean * mean;
        const float s2 = stddev * stddev;
        const float sigmaSq = logf(1.0f + (s2 / m2));
        const float sigma = sqrtf(fmaxf(0.0f, sigmaSq));
        const float mu = logf(fmaxf(1e-12f, mean)) - 0.5f * sigmaSq;
        return expf(mu + sigma * normalSample);
    }

    __device__ __forceinline__ std::uint64_t splitmix64_device(std::uint64_t v) {
        v += 0x9E3779B97F4A7C15ULL;
        v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ULL;
        v = (v ^ (v >> 27)) * 0x94D049BB133111EBULL;
        return v ^ (v >> 31);
    }

    __device__ __forceinline__ float stbn_lookup_device(
        const JuicerCuda::GrainPayload& grain,
        int x,
        int y,
        int t)
    {
        if (!grain.stbn || grain.stbnWidth <= 0 || grain.stbnHeight <= 0 || grain.stbnFrames <= 0) {
            return 0.0f;
        }

        if (grain.stbnWidth > 0) {
            x = (x + grain.stbnOffsetX) % grain.stbnWidth;
            if (x < 0) x += grain.stbnWidth;
        }
        if (grain.stbnHeight > 0) {
            y = (y + grain.stbnOffsetY) % grain.stbnHeight;
            if (y < 0) y += grain.stbnHeight;
        }

        if (grain.stbnFrames > 0) {
            t = t % grain.stbnFrames;
            if (t < 0) t += grain.stbnFrames;
        }

        const std::size_t idx = (static_cast<std::size_t>(t) * static_cast<std::size_t>(grain.stbnHeight) + static_cast<std::size_t>(y)) *
            static_cast<std::size_t>(grain.stbnWidth) + static_cast<std::size_t>(x);
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

    __device__ __forceinline__ std::uint32_t hash_u32_device(int x, int y, std::uint64_t seed, std::uint64_t salt) {
        const std::uint64_t h = splitmix64_device(seed ^ salt ^
            (static_cast<std::uint64_t>(x) * 0x9E3779B97F4A7C15ULL) ^
            (static_cast<std::uint64_t>(y) * 0xBF58476D1CE4E5B9ULL));
        return static_cast<std::uint32_t>(h & 0xFFFFFFFFu);
    }

    __device__ __forceinline__ std::size_t wang_lut_index_device(int l, int r, int t, int b, int colors) {
        const std::size_t c = static_cast<std::size_t>(colors);
        return (((static_cast<std::size_t>(l) * c + static_cast<std::size_t>(r)) * c +
                  static_cast<std::size_t>(t)) * c +
                static_cast<std::size_t>(b));
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
        const std::size_t idx = wang_lut_index_device(static_cast<int>(l), static_cast<int>(r), static_cast<int>(t), static_cast<int>(b), colors);
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
        if (x < 0) x = 0;
        if (y < 0) y = 0;
        if (x >= w) x = w - 1;
        if (y >= h) y = h - 1;
        const std::size_t idx = (static_cast<std::size_t>(tileId) * static_cast<std::size_t>(h) + static_cast<std::size_t>(y)) *
            static_cast<std::size_t>(w) + static_cast<std::size_t>(x);
        const std::uint8_t v = grain.wangTiles[idx];
        return (static_cast<float>(v) + 0.5f) * (1.0f / 256.0f);
    }

    __device__ __forceinline__ void wang_offsets_device(
        const JuicerCuda::GrainPayload& grain,
        std::uint64_t absX,
        std::uint64_t absY,
        int& outX,
        int& outY)
    {
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
        if (tx >= w) tx = w - 1;
        if (ty >= h) ty = h - 1;
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

    __device__ __forceinline__ float value_noise_device(float x, float y, std::uint64_t seed) {
        const int ix = static_cast<int>(floorf(x));
        const int iy = static_cast<int>(floorf(y));
        const float fx = x - static_cast<float>(ix);
        const float fy = y - static_cast<float>(iy);
        const float sx = smoothstep_device(fx);
        const float sy = smoothstep_device(fy);

        const float v00 = hash01_device(ix, iy, seed);
        const float v10 = hash01_device(ix + 1, iy, seed);
        const float v01 = hash01_device(ix, iy + 1, seed);
        const float v11 = hash01_device(ix + 1, iy + 1, seed);

        const float v0 = v00 + (v10 - v00) * sx;
        const float v1 = v01 + (v11 - v01) * sx;
        return v0 + (v1 - v0) * sy;
    }

    __device__ __forceinline__ float grain_breathing_factor_device(
        const JuicerCuda::GrainPayload& grain,
        std::uint64_t absX,
        std::uint64_t absY)
    {
        const float amp = grain.breathingAmplitude;
        const int period = grain.breathingPeriodFrames;
        if (!(amp > 0.0f) || period <= 0 || !(grain.pixelSizeUm > 0.0f)) {
            return 1.0f;
        }

        const float cellSmallUm = grain.breathingCellUmSmall;
        const float cellLargeUm = grain.breathingCellUmLarge;
        if (!(cellSmallUm > 0.0f) || !(cellLargeUm > 0.0f)) {
            return 1.0f;
        }

        const float time = static_cast<float>(grain.frameIndex) + grain.timeAlpha;
        const float periodF = static_cast<float>(period);
        const float stepF = floorf(time / periodF);
        const std::int64_t step = static_cast<std::int64_t>(stepF);
        const float frac = (time - stepF * periodF) / periodF;
        const float t = smoothstep_device(fminf(fmaxf(frac, 0.0f), 1.0f));

        const std::uint64_t seed = (grain.stbnSessionSeed != 0) ? grain.stbnSessionSeed : 1ULL;
        const std::uint64_t seedA = splitmix64_device(seed ^ (static_cast<std::uint64_t>(step) * 0x8EBC6AF09C88C6E3ULL));
        const std::uint64_t seedB = splitmix64_device(seed ^ (static_cast<std::uint64_t>(step + 1) * 0x8EBC6AF09C88C6E3ULL));

        const float invPixel = 1.0f / grain.pixelSizeUm;
        const float cellSmallPx = fmaxf(4.0f, cellSmallUm * invPixel);
        const float cellLargePx = fmaxf(4.0f, cellLargeUm * invPixel);

        float driftPx = 0.0f;
        if (grain.breathingDriftUmPerFrame > 0.0f) {
            driftPx = grain.breathingDriftUmPerFrame * invPixel;
        }
        float driftX = 0.0f;
        float driftY = 0.0f;
        if (driftPx > 0.0f) {
            const std::uint64_t h = splitmix64_device(seed ^ 0x6A09E667F3BCC909ULL);
            constexpr float kTwoPi = 6.28318530717958647692f;
            constexpr float kInvU32 = 1.0f / 4294967296.0f;
            const float angle = static_cast<float>(static_cast<std::uint32_t>(h & 0xFFFFFFFFu)) * kInvU32 * kTwoPi;
            driftX = cosf(angle) * driftPx;
            driftY = sinf(angle) * driftPx;
        }

        const float rollPx = (grain.pitchPx > 0) ? static_cast<float>(grain.pitchPx) : 0.0f;
        const float baseX = static_cast<float>(absX) + driftX * time;
        const float baseY = static_cast<float>(absY) + driftY * time + rollPx * time;

        const float xSmall = baseX / cellSmallPx;
        const float ySmall = baseY / cellSmallPx;
        const float xLarge = baseX / cellLargePx;
        const float yLarge = baseY / cellLargePx;

        const float noiseSmallA = value_noise_device(xSmall, ySmall, seedA ^ 0x9E3779B97F4A7C15ULL);
        const float noiseSmallB = value_noise_device(xSmall, ySmall, seedB ^ 0x9E3779B97F4A7C15ULL);
        const float noiseLargeA = value_noise_device(xLarge, yLarge, seedA ^ 0xBF58476D1CE4E5B9ULL);
        const float noiseLargeB = value_noise_device(xLarge, yLarge, seedB ^ 0xBF58476D1CE4E5B9ULL);

        const float noiseSmall = noiseSmallA + (noiseSmallB - noiseSmallA) * t;
        const float noiseLarge = noiseLargeA + (noiseLargeB - noiseLargeA) * t;
        const float mix = fminf(fmaxf(grain.breathingMix, 0.0f), 1.0f);
        const float noise = mix * noiseSmall + (1.0f - mix) * noiseLarge;

        const float n = noise * 2.0f - 1.0f;
        const float factor = 1.0f + amp * n;
        return (factor > 0.0f) ? factor : 0.0f;
    }

    __device__ __forceinline__ float grain_clump_factor_device(
        const JuicerCuda::GrainPayload& grain,
        std::uint64_t absX,
        std::uint64_t absY)
    {
        const float stddevSpatial = grain.microStructure[1] * 0.001f;
        if (!(stddevSpatial > 0.0f) || !(grain.pixelSizeUm > 0.0f)) {
            return 1.0f;
        }

        float cellUm = grain.microStructure[0];
        if (!(cellUm > 0.0f)) {
            cellUm = 0.0f;
        }
        const float invPixel = 1.0f / grain.pixelSizeUm;
        float cellPx = cellUm * invPixel;
        const float minCellPx = 4.0f;
        if (!(cellPx > minCellPx)) {
            cellPx = minCellPx;
        }

        const float time = static_cast<float>(grain.frameIndex) + grain.timeAlpha;

        const float rollPx = (grain.pitchPx > 0) ? static_cast<float>(grain.pitchPx) : 0.0f;
        const float baseXStatic = static_cast<float>(absX);
        const float baseYStatic = static_cast<float>(absY) + rollPx * time;
        const float xStatic = baseXStatic / cellPx;
        const float yStatic = baseYStatic / cellPx;

        std::uint64_t staticSeed = (grain.clipToken != 0) ? grain.clipToken : grain.stbnSessionSeed;
        if (staticSeed == 0) {
            staticSeed = 1ULL;
        }
        const float u1S = value_noise_device(xStatic, yStatic, staticSeed ^ 0x9E3779B97F4A7C15ULL);
        const float u2S = value_noise_device(xStatic + 19.19f, yStatic + 7.23f, staticSeed ^ 0xBF58476D1CE4E5B9ULL);
        float u1 = fminf(fmaxf(u1S, 1e-6f), 1.0f - 1e-6f);
        float u2 = fminf(fmaxf(u2S, 0.0f), 1.0f);
        const float r = sqrtf(-2.0f * logf(u1));
        constexpr float kTwoPi = 6.28318530717958647692f;
        const float nStatic = r * cosf(kTwoPi * u2);

        const float mix = fminf(fmaxf(grain.clumpTemporalMix, 0.0f), 1.0f);
        if (!(mix > 0.0f)) {
            float staticVal = lognormal_from_mean_std_device(1.0f, stddevSpatial, nStatic);
            if (!device_isfinite(staticVal)) {
                staticVal = 1.0f;
            }
            return staticVal;
        }

        int period = grain.clumpMorphPeriodFrames;
        if (period <= 0) {
            period = 1;
        }
        const float periodF = static_cast<float>(period);
        const float stepF = floorf(time / periodF);
        const std::int64_t step = static_cast<std::int64_t>(stepF);
        const float frac = (time - stepF * periodF) / periodF;
        const float t = smoothstep_device(fminf(fmaxf(frac, 0.0f), 1.0f));

        std::uint64_t temporalSeed = (grain.stbnSessionSeed != 0) ? grain.stbnSessionSeed : 1ULL;
        const std::uint64_t seedA = splitmix64_device(temporalSeed ^ (static_cast<std::uint64_t>(step) * 0xD2B74407B1CE6E93ULL));
        const std::uint64_t seedB = splitmix64_device(temporalSeed ^ (static_cast<std::uint64_t>(step + 1) * 0xD2B74407B1CE6E93ULL));

        const float u1A = hash01_device(0, 0, seedA ^ 0x9E3779B97F4A7C15ULL);
        const float u2A = hash01_device(1, 0, seedA ^ 0xBF58476D1CE4E5B9ULL);
        const float u1B = hash01_device(0, 0, seedB ^ 0x9E3779B97F4A7C15ULL);
        const float u2B = hash01_device(1, 0, seedB ^ 0xBF58476D1CE4E5B9ULL);

        u1 = u1A + (u1B - u1A) * t;
        u2 = u2A + (u2B - u2A) * t;
        u1 = fminf(fmaxf(u1, 1e-6f), 1.0f - 1e-6f);
        u2 = fminf(fmaxf(u2, 0.0f), 1.0f);

        const float rT = sqrtf(-2.0f * logf(u1));
        const float nTemporal = rT * cosf(kTwoPi * u2);

        constexpr float kClumpStrengthStdScale = 0.65f;
        const float strengthStd = mix * kClumpStrengthStdScale;
        float strength = lognormal_from_mean_std_device(1.0f, strengthStd, nTemporal);
        const float strengthNorm = rsqrtf(1.0f + strengthStd * strengthStd);
        strength *= strengthNorm;
        const float stddev = stddevSpatial * strength;
        const float denom = 1.0f + stddev * stddev;
        const float numer = 1.0f + stddevSpatial * stddevSpatial;
        const float clumpRmsNorm = (denom > 0.0f) ? sqrtf(numer / denom) : 1.0f;

        float clumpVal = lognormal_from_mean_std_device(1.0f, stddev, nStatic);
        if (!device_isfinite(clumpVal)) {
            clumpVal = 1.0f;
        }
        clumpVal *= clumpRmsNorm;

        return clumpVal;
    }

    __device__ __forceinline__ float stbn_sample_device(
        const JuicerCuda::GrainPayload& grain,
        std::uint64_t absX,
        std::uint64_t absY,
        int offsetX,
        int offsetY,
        int frameOffset)
    {
        if (!grain.stbn || grain.stbnWidth <= 0 || grain.stbnHeight <= 0 || grain.stbnFrames <= 0) {
            return 0.0f;
        }

        const int t = grain.stbnFrame + frameOffset;
        std::int64_t x64 = static_cast<std::int64_t>(absX) + static_cast<std::int64_t>(offsetX);
        std::int64_t y64 = static_cast<std::int64_t>(absY) + static_cast<std::int64_t>(offsetY);
        if (grain.pitchPx > 0) {
            y64 += static_cast<std::int64_t>(grain.pitchPx) *
                (static_cast<std::int64_t>(grain.frameIndex) + static_cast<std::int64_t>(frameOffset));
        }

        if (grain.stbnWidth > 0) {
            const std::int64_t w = static_cast<std::int64_t>(grain.stbnWidth);
            x64 %= w;
            if (x64 < 0) x64 += w;
        }
        if (grain.stbnHeight > 0) {
            const std::int64_t h = static_cast<std::int64_t>(grain.stbnHeight);
            y64 %= h;
            if (y64 < 0) y64 += h;
        }

        const int x = static_cast<int>(x64);
        const int y = static_cast<int>(y64);
        return stbn_lookup_device(grain, x, y, t);
    }

    struct GrainRngDevice {
        openrand::Philox rng;
        const JuicerCuda::GrainPayload* grain = nullptr;
        std::uint64_t absX = 0;
        std::uint64_t absY = 0;
        std::uint64_t seed = 0;
        std::uint32_t drawIndex = 0;
        int useStbn = 0;
        int frameOffset = 0;
        int wangOffsetX = 0;
        int wangOffsetY = 0;

        __device__ GrainRngDevice(
            std::uint64_t seed_,
            std::uint32_t ctr0,
            std::uint32_t ctr1,
            const JuicerCuda::GrainPayload* grain_,
            std::uint64_t absX_,
            std::uint64_t absY_,
            int useStbn_,
            int frameOffset_)
            : rng(seed_, ctr0, openrand::DEFAULT_GLOBAL_SEED, ctr1),
              grain(grain_),
              absX(absX_),
              absY(absY_),
              seed(seed_),
              useStbn(useStbn_),
              frameOffset(frameOffset_)
        {
            if (useStbn && grain) {
                wang_offsets_device(*grain, absX, absY, wangOffsetX, wangOffsetY);
            }
        }

        __device__ __forceinline__ float uniform() {
            if (useStbn && grain) {
                const std::uint32_t draw = drawIndex++;
                const std::uint64_t h = splitmix64_device(seed + static_cast<std::uint64_t>(draw) * 0x9E3779B97F4A7C15ULL);
                const int offsetX = static_cast<int>(h & 0xFFFFu) + wangOffsetX;
                const int offsetY = static_cast<int>((h >> 16) & 0xFFFFu) + wangOffsetY;
                return stbn_sample_device(*grain, absX, absY, offsetX, offsetY, frameOffset);
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
        GrainRngDevice& rng)
    {
        if (!device_isfinite(lambda) || !(lambda > 0.0f)) {
            return 0;
        }
        if (lambda < 30.0f) {
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
        if (sampleInt < 0) sampleInt = 0;
        return sampleInt;
    }

    __device__ __forceinline__ int fast_binomial_device(
        int n,
        float p,
        GrainRngDevice& rng)
    {
        if (n <= 0) {
            return 0;
        }
        if (!device_isfinite(p)) {
            return 0;
        }
        if (p <= 0.0f) {
            return 0;
        }
        if (p >= 1.0f) {
            return n;
        }

        constexpr int kThreshold = 25;
        if (n < kThreshold) {
            int count = 0;
            for (int k = 0; k < n; ++k) {
                const float u = rng.uniform();
                if (u < p) {
                    ++count;
                }
            }
            return count;
        }

        const float mean = static_cast<float>(n) * p;
        const float var = static_cast<float>(n) * p * (1.0f - p);
        if (var > 10.0f) {
            const float z = rng.normal();
            float sample = mean + sqrtf(var) * z;
            if (!device_isfinite(sample)) {
                sample = 0.0f;
            }
            int approx = static_cast<int>(floorf(sample + 0.5f));
            if (approx < 0) approx = 0;
            if (approx > n) approx = n;
            return approx;
        }

        float u = rng.uniform();
        float cdf = 0.0f;
        float prob = powf(1.0f - p, static_cast<float>(n));
        if (!device_isfinite(prob) || prob <= 0.0f) {
            const float z = rng.normal();
            float sample = mean + sqrtf(fmaxf(0.0f, var)) * z;
            if (!device_isfinite(sample)) {
                sample = 0.0f;
            }
            int approx = static_cast<int>(floorf(sample + 0.5f));
            if (approx < 0) approx = 0;
            if (approx > n) approx = n;
            return approx;
        }

        int k = 0;
        while (cdf < u && k <= n) {
            cdf += prob;
            if (k < n) {
                const float denom = 1.0f - p;
                if (!(denom > 0.0f)) {
                    return n;
                }
                prob = prob * ((static_cast<float>(n - k) / static_cast<float>(k + 1)) * (p / denom));
            }
            ++k;
        }
        return k - 1;
    }

    __device__ __forceinline__ int poisson_sample_device(float lambda, GrainRngDevice& rng) {
        return fast_poisson_device(lambda, rng);
    }

    __device__ __forceinline__ int binomial_sample_device(int n, float p, GrainRngDevice& rng) {
        return fast_binomial_device(n, p, rng);
    }

    __device__ __forceinline__ float layer_particle_model_device(
        float density,
        float densityMax,
        float nParticles,
        float odParticle,
        float uniformity,
        std::uint64_t seed,
        std::uint64_t absX,
        std::uint64_t absY,
        const JuicerCuda::GrainPayload& grain,
        int useStbn,
        int frameOffset)
    {
        if (!device_isfinite(density) || density < 0.0f) {
            density = 0.0f;
        }
        if (!device_isfinite(densityMax) || !(densityMax > 0.0f)) {
            return 0.0f;
        }
        if (!device_isfinite(nParticles) || !(nParticles > 0.0f)) {
            return 0.0f;
        }
        if (!device_isfinite(odParticle) || !(odParticle > 0.0f)) {
            return 0.0f;
        }

        float probability = density / densityMax;
        probability = fminf(fmaxf(probability, 1e-6f), 1.0f - 1e-6f);

        float uniform = device_isfinite(uniformity) ? fminf(fmaxf(uniformity, 0.0f), 1.0f) : 0.0f;
        float saturation = 1.0f - probability * uniform * (1.0f - 1e-6f);
        if (!device_isfinite(saturation) || saturation <= 0.0f) {
            saturation = 1e-6f;
        }

        GrainRngDevice rng(seed,
                           static_cast<std::uint32_t>(absX),
                           static_cast<std::uint32_t>(absY),
                           &grain,
                           absX,
                           absY,
                           useStbn,
                           frameOffset);
        const float lambda = nParticles / saturation;
        const int seeds = poisson_sample_device(lambda, rng);
        const int grainCount = binomial_sample_device(seeds, probability, rng);
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
        int n)
    {
        if (!x || !y || n <= 0) {
            return density;
        }
        if (!device_isfinite(density)) {
            return y[0];
        }

        int domainBegin = 0;
        while (domainBegin < n && !device_isfinite(ldg_f(x + domainBegin))) {
            ++domainBegin;
        }
        if (domainBegin >= n) {
            return density;
        }
        int domainEnd = n - 1;
        while (domainEnd > domainBegin && !device_isfinite(ldg_f(x + domainEnd))) {
            --domainEnd;
        }

        const float xmin = ldg_f(x + domainBegin);
        const float xmax = ldg_f(x + domainEnd);
        if (!device_isfinite(xmin) || !device_isfinite(xmax) || !(xmax >= xmin)) {
            return ldg_f(y + domainBegin);
        }

        if (density <= xmin) {
            return ldg_f(y + domainBegin);
        }
        if (density >= xmax) {
            return ldg_f(y + domainEnd);
        }

        int i1 = domainBegin + 1;
        while (i1 <= domainEnd && ldg_f(x + i1) < density) {
            ++i1;
        }
        if (i1 > domainEnd) {
            return ldg_f(y + domainEnd);
        }

        const int i0 = i1 - 1;
        const float x0 = ldg_f(x + i0);
        const float x1 = ldg_f(x + i1);
        const float y0 = ldg_f(y + i0);
        const float y1 = ldg_f(y + i1);

        const float denom = x1 - x0;
        if (!(denom > 0.0f) || !device_isfinite(denom)) {
            return y0;
        }

        const float t = (density - x0) / denom;
        return y0 + t * (y1 - y0);
    }

    __device__ __forceinline__ JuicerCuda::DeviceCurveView curve_for_channel_device(
        const JuicerCuda::FilmDevelopPayload& dev,
        int channel)
    {
        if (channel == 0) {
            return dev.densR;
        }
        if (channel == 1) {
            return dev.densG;
        }
        return dev.densB;
    }

} // namespace

__global__ void develop_film_density_kernel(
    JuicerCuda::PipelineRunParams params,
    float* outC,
    float* outM,
    float* outY)
{
    const JuicerCuda::FilmDevelopPayload& dev = params.filmDevelop;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    if (!params.src || params.srcRowBytes == 0) {
        return;
    }
    if (!outC || !outM || !outY) {
        return;
    }

    const int nC = params.nComponents;
    if (!(nC == 3 || nC == 4)) {
        return;
    }

    const std::size_t pixelBytes = static_cast<std::size_t>(nC) * sizeof(float);
    const char* srcRow = reinterpret_cast<const char*>(params.src) + static_cast<std::size_t>(y) * params.srcRowBytes;
    const float* srcPix = reinterpret_cast<const float*>(srcRow + static_cast<std::size_t>(x) * pixelBytes);
    if (!srcPix) {
        return;
    }

    const float rgbIn[3] = { srcPix[0], srcPix[1], srcPix[2] };

    const bool useSpatialDir =
        dev.spatialDir.active &&
        dev.spatialDir.corrY && dev.spatialDir.corrM && dev.spatialDir.corrC;

    float D_cmy[3] = { 0.0f, 0.0f, 0.0f };
    if (useSpatialDir) {
        float logE_raw[3] = { 0.0f, 0.0f, 0.0f };
        compute_logE_raw_device(params, rgbIn, logE_raw);

        const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
        const float corrY = dev.spatialDir.corrY[idx];
        const float corrM = dev.spatialDir.corrM[idx];
        const float corrC = dev.spatialDir.corrC[idx];

        float logE_corr[3] = {
            logE_raw[0] - corrY,
            logE_raw[1] - corrM,
            logE_raw[2] - corrC
        };

        const JuicerCuda::DeviceCurveView cB = dev.dirPrecorrected ? dev.dirDensB : dev.densB;
        const JuicerCuda::DeviceCurveView cG = dev.dirPrecorrected ? dev.dirDensG : dev.densG;
        const JuicerCuda::DeviceCurveView cR = dev.dirPrecorrected ? dev.dirDensR : dev.densR;

        logE_corr[0] = sanitize_inf_logE_for_curve_device(logE_corr[0], cB);
        logE_corr[1] = sanitize_inf_logE_for_curve_device(logE_corr[1], cG);
        logE_corr[2] = sanitize_inf_logE_for_curve_device(logE_corr[2], cR);

        const float DY = sample_density_at_logE_device(cB, logE_corr[0], dev.gammaFactorB);
        const float DM = sample_density_at_logE_device(cG, logE_corr[1], dev.gammaFactorG);
        const float DC = sample_density_at_logE_device(cR, logE_corr[2], dev.gammaFactorR);

        D_cmy[0] = DC;
        D_cmy[1] = DM;
        D_cmy[2] = DY;
    }
    else {
        float logE_raw[3] = { 0.0f, 0.0f, 0.0f };
        float logE_sanitized[3] = { 0.0f, 0.0f, 0.0f };
        float layerPre[3] = { 0.0f, 0.0f, 0.0f };
        compute_logE_and_layer_pre_device(params, rgbIn, logE_raw, logE_sanitized, layerPre);

        if (dev.dir.active) {
            float logE_corr[3] = { logE_sanitized[0], logE_sanitized[1], logE_sanitized[2] };
            apply_dir_runtime_logE_device(logE_corr, layerPre, dev.dir, dev.densB, dev.densG, dev.densR);

            const JuicerCuda::DeviceCurveView cB = dev.dirPrecorrected ? dev.dirDensB : dev.densB;
            const JuicerCuda::DeviceCurveView cG = dev.dirPrecorrected ? dev.dirDensG : dev.densG;
            const JuicerCuda::DeviceCurveView cR = dev.dirPrecorrected ? dev.dirDensR : dev.densR;

            const float DY = sample_density_at_logE_device(cB, logE_corr[0], dev.gammaFactorB);
            const float DM = sample_density_at_logE_device(cG, logE_corr[1], dev.gammaFactorG);
            const float DC = sample_density_at_logE_device(cR, logE_corr[2], dev.gammaFactorR);

            D_cmy[0] = DC;
            D_cmy[1] = DM;
            D_cmy[2] = DY;
        }
        else {
            // Map B/G/R layer densities to C/M/Y dyes
            D_cmy[0] = layerPre[2];
            D_cmy[1] = layerPre[1];
            D_cmy[2] = layerPre[0];
        }
    }

    const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
    outC[idx] = D_cmy[0];
    outM[idx] = D_cmy[1];
    outY[idx] = D_cmy[2];
}

__global__ void grain_clear_kernel(float* out, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    if (!out) {
        return;
    }
    out[idx] = 0.0f;
}

__global__ void grain_accumulate_kernel(float* dst, const float* src, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    if (!dst || !src) {
        return;
    }
    const float v = dst[idx] + src[idx];
    dst[idx] = device_isfinite(v) ? v : 0.0f;
}

__global__ void grain_add_bias_kernel(float* inOut, int n, float bias) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    if (!inOut) {
        return;
    }
    const float v = inOut[idx] + bias;
    inOut[idx] = device_isfinite(v) ? v : 0.0f;
}

__global__ void grain_multiply_kernel(float* inOut, const float* mult, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    if (!inOut || !mult) {
        return;
    }
    const float v = inOut[idx] * mult[idx];
    inOut[idx] = device_isfinite(v) ? v : 0.0f;
}

__global__ void grain_subtract_kernel(float* inOut, const float* sub, int n, float amplitude) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
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

__global__ void grain_mix_delta_kernel(
    float* outDelta,
    const float* fineDelta,
    const float* coarseDelta,
    int n,
    float wCoarse,
    float gain)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    if (!outDelta || !fineDelta || !coarseDelta) {
        return;
    }
    const float w = fminf(fmaxf(wCoarse, 0.0f), 1.0f);
    const float g = device_isfinite(gain) ? gain : 1.0f;
    const float fine = fineDelta[idx];
    const float coarse = coarseDelta[idx];
    const float v = g * ((1.0f - w) * fine + w * coarse);
    outDelta[idx] = device_isfinite(v) ? v : 0.0f;
}

__global__ void grain_mix_delta3_kernel(
    float* outDelta,
    const float* fineDelta,
    const float* midDelta,
    const float* coarseDelta,
    int n,
    float wMid,
    float wCoarse,
    float gain,
    float amplitude)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    if (!outDelta || !fineDelta || !midDelta || !coarseDelta) {
        return;
    }
    const float wM = fminf(fmaxf(wMid, 0.0f), 1.0f);
    const float wC = fminf(fmaxf(wCoarse, 0.0f), 1.0f);
    float wF = 1.0f - wM - wC;
    if (!device_isfinite(wF) || wF < 0.0f) {
        wF = 0.0f;
    }
    const float g = device_isfinite(gain) ? gain : 1.0f;
    float a = device_isfinite(amplitude) ? amplitude : 1.0f;
    if (a < 0.0f) {
        a = 0.0f;
    }
    const float fine = fineDelta[idx];
    const float mid = midDelta[idx];
    const float coarse = coarseDelta[idx];
    const float v = a * g * (wF * fine + wM * mid + wC * coarse);
    outDelta[idx] = device_isfinite(v) ? v : 0.0f;
}

__global__ void grain_mix_shared_kernel(
    float* outDelta,
    const float* indDelta,
    const float* sharedDelta,
    int n,
    float wShared,
    float wInd,
    float amplitude)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    if (!outDelta) {
        return;
    }
    const float ws = fminf(fmaxf(wShared, 0.0f), 1.0f);
    const float wi = fminf(fmaxf(wInd, 0.0f), 1.0f);
    float a = device_isfinite(amplitude) ? amplitude : 1.0f;
    if (a < 0.0f) {
        a = 0.0f;
    }
    const float shared = (sharedDelta && ws > 0.0f) ? sharedDelta[idx] : 0.0f;
    const float ind = (indDelta && wi > 0.0f) ? indDelta[idx] : 0.0f;
    const float v = a * (ws * shared + wi * ind);
    outDelta[idx] = device_isfinite(v) ? v : 0.0f;
}

__global__ void grain_debug_encode_avg3_kernel(
    float* outR,
    float* outG,
    float* outB,
    const float* in0,
    const float* in1,
    const float* in2,
    int n,
    float offset,
    float scale)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    if (!outR || !outG || !outB || !in0 || !in1 || !in2) {
        return;
    }
    const float v0 = in0[idx];
    const float v1 = in1[idx];
    const float v2 = in2[idx];
    float avg = (v0 + v1 + v2) * (1.0f / 3.0f);
    if (!device_isfinite(avg)) {
        avg = 0.0f;
    }
    const float s = device_isfinite(scale) ? scale : 1.0f;
    const float o = device_isfinite(offset) ? offset : 0.0f;
    const float out = o + avg * s;
    outR[idx] = out;
    outG[idx] = out;
    outB[idx] = out;
}

__global__ void grain_apply_simple_kernel(
    JuicerCuda::PipelineRunParams params,
    float* inOut,
    int channelIndex)
{
    const JuicerCuda::GrainPayload& grain = params.grain;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) {
        return;
    }
    if (!inOut) {
        return;
    }
    if (channelIndex < 0 || channelIndex > 2) {
        return;
    }

    const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
    float density = inOut[idx];

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

    if (!device_isfinite(densityMax) || !(densityMax > 0.0f) ||
        !device_isfinite(nParticles) || !(nParticles > 0.0f) ||
        !device_isfinite(odParticle) || !(odParticle > 0.0f))
    {
        return;
    }

    const std::uint64_t absX = static_cast<std::uint64_t>(grain.originX + x);
    const std::uint64_t absY = static_cast<std::uint64_t>(grain.originY + y);

    density += densityMin;

    const int useStbn = (grain.stbn && grain.stbnWidth > 0 && grain.stbnHeight > 0 && grain.stbnFrames > 0) ? 1 : 0;

    const float clumpFactor = grain_clump_factor_device(grain, absX, absY);
    const float wCoarse = fminf(fmaxf(mixWeight, 0.0f), 1.0f);
    const float wFine = 1.0f - wCoarse;
    const bool useMix = (wCoarse > 0.0f) && (mixScale > 1.0f);

    float acc = 0.0f;
    float accNext = 0.0f;
    for (int sl = 0; sl < nSubLayers; ++sl) {
        const std::uint64_t seed = grain.seedBase ^ (static_cast<std::uint64_t>(channelIndex) + static_cast<std::uint64_t>(sl) * 10ULL);
        const std::uint64_t seedNext = grain.seedBaseNext ^ (static_cast<std::uint64_t>(channelIndex) + static_cast<std::uint64_t>(sl) * 10ULL);
        if (!useMix) {
            acc += layer_particle_model_device(density, densityMax, nParticles, odParticle, uniformity, seed, absX, absY, grain, useStbn, 0);
            if (wantNext) {
                accNext += layer_particle_model_device(density, densityMax, nParticles, odParticle, uniformity, seedNext, absX, absY, grain, useStbn, 1);
            }
        }
        else {
            if (wFine > 0.0f) {
                acc += layer_particle_model_device(density, densityMax, nParticles, odParticle * wFine, uniformity, seed, absX, absY, grain, useStbn, 0);
                if (wantNext) {
                    accNext += layer_particle_model_device(density, densityMax, nParticles, odParticle * wFine, uniformity, seedNext, absX, absY, grain, useStbn, 1);
                }
            }
            const float nParticlesCoarse = nParticles / mixScale;
            if (nParticlesCoarse > 0.0f) {
                acc += layer_particle_model_device(density, densityMax, nParticlesCoarse, odParticle * wCoarse * mixScale, uniformity, seed, absX, absY, grain, useStbn, 0);
                if (wantNext) {
                    accNext += layer_particle_model_device(density, densityMax, nParticlesCoarse, odParticle * wCoarse * mixScale, uniformity, seedNext, absX, absY, grain, useStbn, 1);
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
    inOut[idx] = device_isfinite(acc) ? acc : 0.0f;
}

__global__ void grain_layer_kernel(
    JuicerCuda::PipelineRunParams params,
    const float* inDensity,
    float* outGrain,
    int channelIndex,
    int sublayerIndex)
{
    const JuicerCuda::GrainPayload& grain = params.grain;
    const JuicerCuda::FilmDevelopPayload& dev = params.filmDevelop;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) {
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

    const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
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

    if (!device_isfinite(densityMax) || !(densityMax > 0.0f) ||
        !device_isfinite(nParticles) || !(nParticles > 0.0f) ||
        !device_isfinite(odParticle) || !(odParticle > 0.0f))
    {
        outGrain[idx] = 0.0f;
        return;
    }

    const std::uint64_t absX = static_cast<std::uint64_t>(grain.originX + x);
    const std::uint64_t absY = static_cast<std::uint64_t>(grain.originY + y);

    const JuicerCuda::DeviceCurveView curve = curve_for_channel_device(dev, channelIndex);
    const float* layerCurve = grain.densityCurvesLayers[sublayerIndex][channelIndex];
    density = interp_density_layer_device(density, curve.y, layerCurve, curve.n);
    density += densityMin;

    const std::uint64_t seed = grain.seedBase ^ (static_cast<std::uint64_t>(channelIndex) + static_cast<std::uint64_t>(sublayerIndex) * 10ULL);
    const int useStbn = (grain.stbn && grain.stbnWidth > 0 && grain.stbnHeight > 0 && grain.stbnFrames > 0) ? 1 : 0;

    const float clumpFactor = grain_clump_factor_device(grain, absX, absY);
    const float wCoarse = fminf(fmaxf(mixWeight, 0.0f), 1.0f);
    const float wFine = 1.0f - wCoarse;
    const bool useMix = (wCoarse > 0.0f) && (mixScale > 1.0f);

    float grainSample = 0.0f;
    float grainSampleNext = 0.0f;
    if (!useMix) {
        grainSample = layer_particle_model_device(density, densityMax, nParticles, odParticle, uniformity, seed, absX, absY, grain, useStbn, 0);
        if (wantNext) {
            const std::uint64_t seedNext = grain.seedBaseNext ^ (static_cast<std::uint64_t>(channelIndex) + static_cast<std::uint64_t>(sublayerIndex) * 10ULL);
            grainSampleNext = layer_particle_model_device(density, densityMax, nParticles, odParticle, uniformity, seedNext, absX, absY, grain, useStbn, 1);
        }
    }
    else {
        if (wFine > 0.0f) {
            grainSample += layer_particle_model_device(density, densityMax, nParticles, odParticle * wFine, uniformity, seed, absX, absY, grain, useStbn, 0);
            if (wantNext) {
                const std::uint64_t seedNext = grain.seedBaseNext ^ (static_cast<std::uint64_t>(channelIndex) + static_cast<std::uint64_t>(sublayerIndex) * 10ULL);
                grainSampleNext += layer_particle_model_device(density, densityMax, nParticles, odParticle * wFine, uniformity, seedNext, absX, absY, grain, useStbn, 1);
            }
        }
        const float nParticlesCoarse = nParticles / mixScale;
        if (nParticlesCoarse > 0.0f) {
            grainSample += layer_particle_model_device(density, densityMax, nParticlesCoarse, odParticle * wCoarse * mixScale, uniformity, seed, absX, absY, grain, useStbn, 0);
            if (wantNext) {
                const std::uint64_t seedNext = grain.seedBaseNext ^ (static_cast<std::uint64_t>(channelIndex) + static_cast<std::uint64_t>(sublayerIndex) * 10ULL);
                grainSampleNext += layer_particle_model_device(density, densityMax, nParticlesCoarse, odParticle * wCoarse * mixScale, uniformity, seedNext, absX, absY, grain, useStbn, 1);
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

__global__ void develop_film_density_from_raw_kernel(
    JuicerCuda::PipelineRunParams params,
    const float* inB,
    const float* inG,
    const float* inR,
    float* outC,
    float* outM,
    float* outY)
{
    const JuicerCuda::FilmDevelopPayload& dev = params.filmDevelop;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    if (!inB || !inG || !inR || !outC || !outM || !outY) {
        return;
    }

    const int nC = params.nComponents;
    if (!(nC == 3 || nC == 4)) {
        return;
    }

    const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
    const float filmRaw[3] = { inB[idx], inG[idx], inR[idx] };

    const bool useSpatialDir =
        dev.spatialDir.active &&
        dev.spatialDir.corrY && dev.spatialDir.corrM && dev.spatialDir.corrC;

    float D_cmy[3] = { 0.0f, 0.0f, 0.0f };
    if (useSpatialDir) {
        float logE_raw[3] = { 0.0f, 0.0f, 0.0f };
        compute_logE_raw_from_film_raw_device(params, filmRaw, logE_raw);

        const float corrY = dev.spatialDir.corrY[idx];
        const float corrM = dev.spatialDir.corrM[idx];
        const float corrC = dev.spatialDir.corrC[idx];

        float logE_corr[3] = {
            logE_raw[0] - corrY,
            logE_raw[1] - corrM,
            logE_raw[2] - corrC
        };

        const JuicerCuda::DeviceCurveView cB = dev.dirPrecorrected ? dev.dirDensB : dev.densB;
        const JuicerCuda::DeviceCurveView cG = dev.dirPrecorrected ? dev.dirDensG : dev.densG;
        const JuicerCuda::DeviceCurveView cR = dev.dirPrecorrected ? dev.dirDensR : dev.densR;

        logE_corr[0] = sanitize_inf_logE_for_curve_device(logE_corr[0], cB);
        logE_corr[1] = sanitize_inf_logE_for_curve_device(logE_corr[1], cG);
        logE_corr[2] = sanitize_inf_logE_for_curve_device(logE_corr[2], cR);

        const float DY = sample_density_at_logE_device(cB, logE_corr[0], dev.gammaFactorB);
        const float DM = sample_density_at_logE_device(cG, logE_corr[1], dev.gammaFactorG);
        const float DC = sample_density_at_logE_device(cR, logE_corr[2], dev.gammaFactorR);

        D_cmy[0] = DC;
        D_cmy[1] = DM;
        D_cmy[2] = DY;
    }
    else {
        float logE_raw[3] = { 0.0f, 0.0f, 0.0f };
        float logE_sanitized[3] = { 0.0f, 0.0f, 0.0f };
        float layerPre[3] = { 0.0f, 0.0f, 0.0f };
        compute_logE_from_film_raw_device(params, filmRaw, logE_raw, logE_sanitized, layerPre);

        if (dev.dir.active) {
            float logE_corr[3] = { logE_sanitized[0], logE_sanitized[1], logE_sanitized[2] };
            apply_dir_runtime_logE_device(logE_corr, layerPre, dev.dir, dev.densB, dev.densG, dev.densR);

            const JuicerCuda::DeviceCurveView cB = dev.dirPrecorrected ? dev.dirDensB : dev.densB;
            const JuicerCuda::DeviceCurveView cG = dev.dirPrecorrected ? dev.dirDensG : dev.densG;
            const JuicerCuda::DeviceCurveView cR = dev.dirPrecorrected ? dev.dirDensR : dev.densR;

            const float DY = sample_density_at_logE_device(cB, logE_corr[0], dev.gammaFactorB);
            const float DM = sample_density_at_logE_device(cG, logE_corr[1], dev.gammaFactorG);
            const float DC = sample_density_at_logE_device(cR, logE_corr[2], dev.gammaFactorR);

            D_cmy[0] = DC;
            D_cmy[1] = DM;
            D_cmy[2] = DY;
        }
        else {
            D_cmy[0] = layerPre[2];
            D_cmy[1] = layerPre[1];
            D_cmy[2] = layerPre[0];
        }
    }

    outC[idx] = D_cmy[0];
    outM[idx] = D_cmy[1];
    outY[idx] = D_cmy[2];
}
