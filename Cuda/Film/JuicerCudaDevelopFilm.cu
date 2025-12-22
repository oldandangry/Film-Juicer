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

namespace {

    __device__ __forceinline__ std::uint64_t fnv1a_update_u64_device(std::uint64_t h, std::uint64_t v) {
        constexpr std::uint64_t kFnvPrime = 0x100000001b3ULL;
        for (int i = 0; i < 8; ++i) {
            h ^= static_cast<std::uint64_t>((v >> (8 * i)) & 0xffULL);
            h *= kFnvPrime;
        }
        return h;
    }

    __device__ __forceinline__ std::uint64_t fnv1a_hash_u64_5_device(
        std::uint64_t a,
        std::uint64_t b,
        std::uint64_t c,
        std::uint64_t d,
        std::uint64_t e)
    {
        std::uint64_t h = 0xcbf29ce484222325ULL;
        h = fnv1a_update_u64_device(h, a);
        h = fnv1a_update_u64_device(h, b);
        h = fnv1a_update_u64_device(h, c);
        h = fnv1a_update_u64_device(h, d);
        h = fnv1a_update_u64_device(h, e);
        return h;
    }

    __device__ __forceinline__ float hash_to_uniform_device(std::uint64_t h) {
        constexpr double kInvU64Max = 1.0 / 18446744073709551615.0;
        return static_cast<float>((static_cast<double>(h) + 0.5) * kInvU64Max);
    }

    __device__ __forceinline__ float box_muller_device(std::uint64_t h1, std::uint64_t h2) {
        float u1 = hash_to_uniform_device(h1);
        u1 = fminf(fmaxf(u1, 1e-7f), 1.0f);
        const float u2 = hash_to_uniform_device(h2);
        const float r = sqrtf(-2.0f * logf(u1));
        constexpr float kTwoPi = 6.28318530717958647692f;
        const float theta = kTwoPi * u2;
        return r * cosf(theta);
    }

    __device__ __forceinline__ float lognormal_from_mean_std_device(float mean, float stddev, float normalSample) {
        const float m2 = mean * mean;
        const float s2 = stddev * stddev;
        const float sigmaSq = logf(1.0f + (s2 / m2));
        const float sigma = sqrtf(fmaxf(0.0f, sigmaSq));
        const float mu = logf(fmaxf(1e-12f, mean)) - 0.5f * sigmaSq;
        return expf(mu + sigma * normalSample);
    }

    __device__ __forceinline__ float next_uniform_device(
        std::uint64_t seed,
        std::uint64_t x,
        std::uint64_t y,
        int counter)
    {
        const std::uint64_t h = fnv1a_hash_u64_5_device(seed, x, y, static_cast<std::uint64_t>(counter), 0ULL);
        return hash_to_uniform_device(h);
    }

    __device__ __forceinline__ float next_normal_device(
        std::uint64_t seed,
        std::uint64_t x,
        std::uint64_t y,
        int& counter)
    {
        const std::uint64_t h1 = fnv1a_hash_u64_5_device(seed, x, y, static_cast<std::uint64_t>(counter), 0ULL);
        ++counter;
        const std::uint64_t h2 = fnv1a_hash_u64_5_device(seed, x, y, static_cast<std::uint64_t>(counter), 1ULL);
        ++counter;
        return box_muller_device(h1, h2);
    }

    __device__ __forceinline__ int fast_poisson_device(
        float lambda,
        std::uint64_t seed,
        std::uint64_t x,
        std::uint64_t y,
        int& counter)
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
                const float u = next_uniform_device(seed, x, y, counter++);
                p *= fminf(fmaxf(u, 1e-7f), 1.0f);
            }
            return k - 1;
        }

        const float z = next_normal_device(seed, x, y, counter);
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
        std::uint64_t seed,
        std::uint64_t x,
        std::uint64_t y,
        int& counter)
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
                const float u = next_uniform_device(seed, x, y, counter++);
                if (u < p) {
                    ++count;
                }
            }
            return count;
        }

        const float mean = static_cast<float>(n) * p;
        const float var = static_cast<float>(n) * p * (1.0f - p);
        if (var > 10.0f) {
            const float z = next_normal_device(seed, x, y, counter);
            float sample = mean + sqrtf(var) * z;
            if (!device_isfinite(sample)) {
                sample = 0.0f;
            }
            int approx = static_cast<int>(floorf(sample + 0.5f));
            if (approx < 0) approx = 0;
            if (approx > n) approx = n;
            return approx;
        }

        float u = next_uniform_device(seed, x, y, counter++);
        float cdf = 0.0f;
        float prob = powf(1.0f - p, static_cast<float>(n));
        if (!device_isfinite(prob) || prob <= 0.0f) {
            const float z = next_normal_device(seed, x, y, counter);
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

    __device__ __forceinline__ float layer_particle_model_device(
        float density,
        float densityMax,
        float nParticles,
        float odParticle,
        float uniformity,
        std::uint64_t seed,
        std::uint64_t absX,
        std::uint64_t absY)
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

        int counter = 0;
        const float lambda = nParticles / saturation;
        const int seeds = fast_poisson_device(lambda, seed, absX, absY, counter);
        const int grainCount = fast_binomial_device(seeds, probability, seed, absX, absY, counter);
        float grain = static_cast<float>(grainCount) * odParticle * saturation;
        if (!device_isfinite(grain)) {
            grain = 0.0f;
        }
        return grain;
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

    float acc = 0.0f;
    for (int sl = 0; sl < nSubLayers; ++sl) {
        const std::uint64_t seed = static_cast<std::uint64_t>(channelIndex) + static_cast<std::uint64_t>(sl) * 10ULL;
        acc += layer_particle_model_device(density, densityMax, nParticles, odParticle, uniformity, seed, absX, absY);
    }
    acc /= static_cast<float>(nSubLayers);
    acc -= densityMin;
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
    const std::uint64_t seed = static_cast<std::uint64_t>(channelIndex) + static_cast<std::uint64_t>(sublayerIndex) * 10ULL;

    float grainSample = layer_particle_model_device(density, densityMax, nParticles, odParticle, uniformity, seed, absX, absY);
    outGrain[idx] = device_isfinite(grainSample) ? grainSample : 0.0f;
}

__global__ void grain_build_clumping_kernel(
    float* out,
    int width,
    int height,
    int originX,
    int originY,
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

    const std::uint64_t absX = static_cast<std::uint64_t>(originX + x);
    const std::uint64_t absY = static_cast<std::uint64_t>(originY + y);

    int counter = 0;
    const float n = next_normal_device(seedBase, absX, absY, counter);
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
