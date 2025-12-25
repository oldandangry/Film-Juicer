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

    __device__ __forceinline__ void stbn_macro_offset_device(
        const JuicerCuda::GrainPayload& grain,
        int tileX,
        int tileY,
        int& outX,
        int& outY)
    {
        if (grain.stbnWidth <= 0 || grain.stbnHeight <= 0) {
            outX = 0;
            outY = 0;
            return;
        }
        const std::uint64_t seed = (grain.stbnSessionSeed != 0) ? grain.stbnSessionSeed : 1ULL;
        const std::uint64_t h = splitmix64_device(seed ^ (static_cast<std::uint64_t>(tileX) * 0x9E3779B97F4A7C15ULL) ^
            (static_cast<std::uint64_t>(tileY) * 0xBF58476D1CE4E5B9ULL));
        outX = static_cast<int>(h % static_cast<std::uint64_t>(grain.stbnWidth));
        outY = static_cast<int>((h >> 32) % static_cast<std::uint64_t>(grain.stbnHeight));
    }

    __device__ __forceinline__ void stbn_macro_jitter_device(
        const JuicerCuda::GrainPayload& grain,
        int tileX,
        int tileY,
        float& outX,
        float& outY)
    {
        if (grain.stbnWidth <= 0 || grain.stbnHeight <= 0) {
            outX = 0.0f;
            outY = 0.0f;
            return;
        }
        const std::uint64_t seed = (grain.stbnSessionSeed != 0) ? grain.stbnSessionSeed : 1ULL;
        const std::uint64_t h = splitmix64_device(seed ^ (static_cast<std::uint64_t>(tileX) * 0xD2B74407B1CE6E93ULL) ^
            (static_cast<std::uint64_t>(tileY) * 0xCA5A826395121157ULL));
        constexpr float kInvU32 = 1.0f / 4294967296.0f;
        const std::uint32_t h0 = static_cast<std::uint32_t>(h & 0xFFFFFFFFu);
        const std::uint32_t h1 = static_cast<std::uint32_t>(h >> 32);
        outX = static_cast<float>(h0) * kInvU32 - 0.5f;
        outY = static_cast<float>(h1) * kInvU32 - 0.5f;
    }

    __device__ __forceinline__ float stbn_sample_device(
        const JuicerCuda::GrainPayload& grain,
        std::uint64_t absX,
        std::uint64_t absY,
        int offsetX,
        int offsetY,
        int offsetT,
        int useClumpParams)
    {
        if (!grain.stbn || grain.stbnWidth <= 0 || grain.stbnHeight <= 0 || grain.stbnFrames <= 0) {
            return 0.0f;
        }

        const int macroTileSize = (useClumpParams != 0) ? grain.clumpMacroTileSize : grain.macroTileSize;
        const float weaveAmplitude = (useClumpParams != 0) ? grain.clumpWeaveAmplitudePx : grain.weaveAmplitudePx;
        const int weavePeriod = grain.weavePeriodFrames;

        float weaveX = 0.0f;
        float weaveY = 0.0f;
        if (weaveAmplitude > 0.0f && weavePeriod > 0) {
            const float t = static_cast<float>(grain.frameIndex);
            constexpr float kTwoPi = 6.28318530717958647692f;
            const float phase = kTwoPi * (t / static_cast<float>(weavePeriod));
            weaveX = weaveAmplitude * sinf(phase);
            weaveY = weaveAmplitude * cosf(phase);
        }

        const float baseX = static_cast<float>(absX) + weaveX;
        const float baseY = static_cast<float>(absY) + weaveY;
        const int t = grain.stbnFrame + offsetT;

        if (macroTileSize > 0) {
            const float tileSize = static_cast<float>(macroTileSize);
            const float invTile = 1.0f / tileSize;
            const int tileX = static_cast<int>(floorf(baseX * invTile));
            const int tileY = static_cast<int>(floorf(baseY * invTile));

            constexpr float kJitterScale = 0.45f;
            float bestDist2 = 1e30f;
            int bestTileX = tileX;
            int bestTileY = tileY;

            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int cellX = tileX + dx;
                    const int cellY = tileY + dy;
                    float jitterX = 0.0f;
                    float jitterY = 0.0f;
                    stbn_macro_jitter_device(grain, cellX, cellY, jitterX, jitterY);
                    const float centerX = (static_cast<float>(cellX) + 0.5f + jitterX * kJitterScale) * tileSize;
                    const float centerY = (static_cast<float>(cellY) + 0.5f + jitterY * kJitterScale) * tileSize;
                    const float dxp = baseX - centerX;
                    const float dyp = baseY - centerY;
                    const float dist2 = dxp * dxp + dyp * dyp;
                    if (dist2 < bestDist2) {
                        bestDist2 = dist2;
                        bestTileX = cellX;
                        bestTileY = cellY;
                    }
                }
            }

            int offX = 0, offY = 0;
            stbn_macro_offset_device(grain, bestTileX, bestTileY, offX, offY);
            const int baseXi = static_cast<int>(floorf(baseX)) + offsetX;
            const int baseYi = static_cast<int>(floorf(baseY)) + offsetY;
            return stbn_lookup_device(grain, baseXi + offX, baseYi + offY, t);
        }

        const int x = static_cast<int>(absX) + offsetX;
        const int y = static_cast<int>(absY) + offsetY;
        return stbn_lookup_device(grain, x, y, t);
    }

    __device__ __forceinline__ float stbn_sample_device(
        const JuicerCuda::GrainPayload& grain,
        std::uint64_t absX,
        std::uint64_t absY)
    {
        return stbn_sample_device(grain, absX, absY, 0, 0, 0, 0);
    }

    struct GrainRngDevice {
        openrand::Philox rng;
        const JuicerCuda::GrainPayload* grain = nullptr;
        std::uint64_t absX = 0;
        std::uint64_t absY = 0;
        std::uint64_t seed = 0;
        std::uint32_t drawIndex = 0;
        int useStbn = 0;
        int useClumpParams = 0;

        __device__ GrainRngDevice(
            std::uint64_t seed_,
            std::uint32_t ctr0,
            std::uint32_t ctr1,
            const JuicerCuda::GrainPayload* grain_,
            std::uint64_t absX_,
            std::uint64_t absY_,
            int useStbn_,
            int useClumpParams_)
            : rng(seed_, ctr0, openrand::DEFAULT_GLOBAL_SEED, ctr1),
              grain(grain_),
              absX(absX_),
              absY(absY_),
              seed(seed_),
              useStbn(useStbn_),
              useClumpParams(useClumpParams_)
        {}

        __device__ __forceinline__ float uniform() {
            if (useStbn && grain) {
                const std::uint32_t draw = drawIndex++;
                const std::uint64_t h = splitmix64_device(seed + static_cast<std::uint64_t>(draw) * 0x9E3779B97F4A7C15ULL);
                const int offsetX = static_cast<int>(h & 0xFFFFu);
                const int offsetY = static_cast<int>((h >> 16) & 0xFFFFu);
                return stbn_sample_device(*grain, absX, absY, offsetX, offsetY, 0, useClumpParams);
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

    __device__ __forceinline__ int poisson_sample_device(float lambda, GrainRngDevice& rng, bool useFastStats) {
        (void)useFastStats;
        return fast_poisson_device(lambda, rng);
    }

    __device__ __forceinline__ int binomial_sample_device(int n, float p, GrainRngDevice& rng, bool useFastStats) {
        (void)useFastStats;
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
        bool useFastStats,
        int useStbn)
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

        GrainRngDevice rng(seed, static_cast<std::uint32_t>(absX), static_cast<std::uint32_t>(absY), &grain, absX, absY, useStbn, 0);
        const float lambda = nParticles / saturation;
        const int seeds = poisson_sample_device(lambda, rng, useFastStats);
        const int grainCount = binomial_sample_device(seeds, probability, rng, useFastStats);
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

    float logE_raw[3] = { 0.0f, 0.0f, 0.0f };
    float logE_sanitized[3] = { 0.0f, 0.0f, 0.0f };
    float layerPre[3] = { 0.0f, 0.0f, 0.0f };
    compute_logE_and_layer_pre_device(params, rgbIn, logE_raw, logE_sanitized, layerPre);

    float D_cmy[3] = { 0.0f, 0.0f, 0.0f };
    const bool useSpatialDir =
        dev.spatialDir.active &&
        dev.spatialDir.corrY && dev.spatialDir.corrM && dev.spatialDir.corrC;
    if (useSpatialDir) {
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

        logE_corr[0] = sanitize_inf_logE_for_curve_device(logE_corr[0], cB.x, cB.n);
        logE_corr[1] = sanitize_inf_logE_for_curve_device(logE_corr[1], cG.x, cG.n);
        logE_corr[2] = sanitize_inf_logE_for_curve_device(logE_corr[2], cR.x, cR.n);

        const float DY = sample_density_at_logE_device(cB.x, cB.y, cB.n, logE_corr[0], dev.gammaFactorB);
        const float DM = sample_density_at_logE_device(cG.x, cG.y, cG.n, logE_corr[1], dev.gammaFactorG);
        const float DC = sample_density_at_logE_device(cR.x, cR.y, cR.n, logE_corr[2], dev.gammaFactorR);

        D_cmy[0] = DC;
        D_cmy[1] = DM;
        D_cmy[2] = DY;
    }
    else if (dev.dir.active) {
        float logE_corr[3] = { logE_sanitized[0], logE_sanitized[1], logE_sanitized[2] };
        apply_dir_runtime_logE_device(logE_corr, layerPre, dev.dir, dev.densB, dev.densG, dev.densR);

        const JuicerCuda::DeviceCurveView cB = dev.dirPrecorrected ? dev.dirDensB : dev.densB;
        const JuicerCuda::DeviceCurveView cG = dev.dirPrecorrected ? dev.dirDensG : dev.densG;
        const JuicerCuda::DeviceCurveView cR = dev.dirPrecorrected ? dev.dirDensR : dev.densR;

        const float DY = sample_density_at_logE_device(cB.x, cB.y, cB.n, logE_corr[0], dev.gammaFactorB);
        const float DM = sample_density_at_logE_device(cG.x, cG.y, cG.n, logE_corr[1], dev.gammaFactorG);
        const float DC = sample_density_at_logE_device(cR.x, cR.y, cR.n, logE_corr[2], dev.gammaFactorR);

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

    if (!device_isfinite(densityMax) || !(densityMax > 0.0f) ||
        !device_isfinite(nParticles) || !(nParticles > 0.0f) ||
        !device_isfinite(odParticle) || !(odParticle > 0.0f))
    {
        return;
    }

    density += densityMin;

    const std::uint64_t absX = static_cast<std::uint64_t>(grain.originX + x);
    const std::uint64_t absY = static_cast<std::uint64_t>(grain.originY + y);
    const int useStbn = (grain.stbn && grain.stbnWidth > 0 && grain.stbnHeight > 0 && grain.stbnFrames > 0) ? 1 : 0;

    float acc = 0.0f;
    for (int sl = 0; sl < nSubLayers; ++sl) {
        const std::uint64_t seed = grain.seedBase ^ (static_cast<std::uint64_t>(channelIndex) + static_cast<std::uint64_t>(sl) * 10ULL);
        acc += layer_particle_model_device(density, densityMax, nParticles, odParticle, uniformity, seed, absX, absY, grain, false, useStbn);
    }
    acc /= static_cast<float>(nSubLayers);
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

    if (!device_isfinite(densityMax) || !(densityMax > 0.0f) ||
        !device_isfinite(nParticles) || !(nParticles > 0.0f) ||
        !device_isfinite(odParticle) || !(odParticle > 0.0f))
    {
        outGrain[idx] = 0.0f;
        return;
    }

    const JuicerCuda::DeviceCurveView curve = curve_for_channel_device(dev, channelIndex);
    const float* layerCurve = grain.densityCurvesLayers[sublayerIndex][channelIndex];
    density = interp_density_layer_device(density, curve.y, layerCurve, curve.n);
    density += densityMin;

    const std::uint64_t absX = static_cast<std::uint64_t>(grain.originX + x);
    const std::uint64_t absY = static_cast<std::uint64_t>(grain.originY + y);
    const std::uint64_t seed = grain.seedBase ^ (static_cast<std::uint64_t>(channelIndex) + static_cast<std::uint64_t>(sublayerIndex) * 10ULL);
    const int useStbn = (grain.stbn && grain.stbnWidth > 0 && grain.stbnHeight > 0 && grain.stbnFrames > 0) ? 1 : 0;

    const bool useFastStats = (grain.useFastStats != 0);
    float grainSample = layer_particle_model_device(density, densityMax, nParticles, odParticle, uniformity, seed, absX, absY, grain, useFastStats, useStbn);
    outGrain[idx] = device_isfinite(grainSample) ? grainSample : 0.0f;
}

__global__ void grain_build_clumping_kernel(
    float* out,
    int width,
    int height,
    JuicerCuda::GrainPayload grain,
    std::uint64_t seedBase,
    float mean,
    float stddev)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    if (!out) {
        return;
    }

    const std::uint64_t absX = static_cast<std::uint64_t>(grain.originX + x);
    const std::uint64_t absY = static_cast<std::uint64_t>(grain.originY + y);
    const int useStbn = (grain.stbn && grain.stbnWidth > 0 && grain.stbnHeight > 0 && grain.stbnFrames > 0) ? 1 : 0;
    GrainRngDevice rng(seedBase, static_cast<std::uint32_t>(absX), static_cast<std::uint32_t>(absY), &grain, absX, absY, useStbn, 1);
    const float n = rng.normal();
    float v = lognormal_from_mean_std_device(mean, stddev, n);
    if (!device_isfinite(v)) {
        v = 1.0f;
    }
    out[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] = v;
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

    float logE_raw[3] = { 0.0f, 0.0f, 0.0f };
    float logE_sanitized[3] = { 0.0f, 0.0f, 0.0f };
    float layerPre[3] = { 0.0f, 0.0f, 0.0f };
    compute_logE_from_film_raw_device(params, filmRaw, logE_raw, logE_sanitized, layerPre);

    float D_cmy[3] = { 0.0f, 0.0f, 0.0f };
    const bool useSpatialDir =
        dev.spatialDir.active &&
        dev.spatialDir.corrY && dev.spatialDir.corrM && dev.spatialDir.corrC;
    if (useSpatialDir) {
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

        logE_corr[0] = sanitize_inf_logE_for_curve_device(logE_corr[0], cB.x, cB.n);
        logE_corr[1] = sanitize_inf_logE_for_curve_device(logE_corr[1], cG.x, cG.n);
        logE_corr[2] = sanitize_inf_logE_for_curve_device(logE_corr[2], cR.x, cR.n);

        const float DY = sample_density_at_logE_device(cB.x, cB.y, cB.n, logE_corr[0], dev.gammaFactorB);
        const float DM = sample_density_at_logE_device(cG.x, cG.y, cG.n, logE_corr[1], dev.gammaFactorG);
        const float DC = sample_density_at_logE_device(cR.x, cR.y, cR.n, logE_corr[2], dev.gammaFactorR);

        D_cmy[0] = DC;
        D_cmy[1] = DM;
        D_cmy[2] = DY;
    }
    else if (dev.dir.active) {
        float logE_corr[3] = { logE_sanitized[0], logE_sanitized[1], logE_sanitized[2] };
        apply_dir_runtime_logE_device(logE_corr, layerPre, dev.dir, dev.densB, dev.densG, dev.densR);

        const JuicerCuda::DeviceCurveView cB = dev.dirPrecorrected ? dev.dirDensB : dev.densB;
        const JuicerCuda::DeviceCurveView cG = dev.dirPrecorrected ? dev.dirDensG : dev.densG;
        const JuicerCuda::DeviceCurveView cR = dev.dirPrecorrected ? dev.dirDensR : dev.densR;

        const float DY = sample_density_at_logE_device(cB.x, cB.y, cB.n, logE_corr[0], dev.gammaFactorB);
        const float DM = sample_density_at_logE_device(cG.x, cG.y, cG.n, logE_corr[1], dev.gammaFactorG);
        const float DC = sample_density_at_logE_device(cR.x, cR.y, cR.n, logE_corr[2], dev.gammaFactorR);

        D_cmy[0] = DC;
        D_cmy[1] = DM;
        D_cmy[2] = DY;
    }
    else {
        D_cmy[0] = layerPre[2];
        D_cmy[1] = layerPre[1];
        D_cmy[2] = layerPre[0];
    }

    outC[idx] = D_cmy[0];
    outM[idx] = D_cmy[1];
    outY[idx] = D_cmy[2];
}
