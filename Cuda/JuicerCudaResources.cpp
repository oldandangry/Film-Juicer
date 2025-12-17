// Cuda/JuicerCudaResources.cpp
//
// Phase 2: WorkingState uploads + primitive validation hooks.
//
#include "Cuda/JuicerCudaResources.h"

#include "FilmProcessing.h"
#include "WorkingState.h"

#include "Logging.h"

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <limits>

// Implemented in Cuda/JuicerCudaPrimitives.cu
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
extern "C" cudaError_t juicer_cuda_probe_density_curve(
    const float* dX,
    const float* dY,
    int n,
    float gammaFactor,
    const float* hLogE,
    int m,
    float* hOut,
    void* cudaStreamOpaque);
#endif

namespace JuicerCuda {

    static void free_curve(DeviceCurve& c) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (c.x) {
            cudaFree(c.x);
            c.x = nullptr;
        }
        if (c.y) {
            cudaFree(c.y);
            c.y = nullptr;
        }
#endif
        c.n = 0;
    }

    static bool alloc_and_upload_curve(DeviceCurve& dst, const Spectral::Curve& src, void* cudaStreamOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)dst;
        (void)src;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        if (src.lambda_nm.empty() || src.linear.empty() || src.lambda_nm.size() != src.linear.size()) {
            outError = "curve has no samples or mismatched arrays";
            return false;
        }

        const int n = static_cast<int>(src.lambda_nm.size());
        if (n <= 0) {
            outError = "curve sample count invalid";
            return false;
        }

        const size_t bytes = static_cast<size_t>(n) * sizeof(float);
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&dst.x), bytes);
        if (err != cudaSuccess) {
            outError = std::string("cudaMalloc(curve.x) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            free_curve(dst);
            return false;
        }
        err = cudaMalloc(reinterpret_cast<void**>(&dst.y), bytes);
        if (err != cudaSuccess) {
            outError = std::string("cudaMalloc(curve.y) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            free_curve(dst);
            return false;
        }

        const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
        err = cudaMemcpyAsync(dst.x, src.lambda_nm.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            outError = std::string("cudaMemcpyAsync(curve.x) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            free_curve(dst);
            return false;
        }
        err = cudaMemcpyAsync(dst.y, src.linear.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            outError = std::string("cudaMemcpyAsync(curve.y) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            free_curve(dst);
            return false;
        }

        dst.n = n;
        return true;
#endif
    }

    Resources::~Resources() {
        free_curve(densB);
        free_curve(densG);
        free_curve(densR);
        free_curve(sensB);
        free_curve(sensG);
        free_curve(sensR);
    }

    Resources* create() noexcept {
        try {
            return new Resources();
        }
        catch (...) {
            return nullptr;
        }
    }

    void destroy(Resources* resources) noexcept {
        delete resources;
    }

    bool ensure_uploaded(Resources& resources, const WorkingState& ws, void* cudaStreamOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)ws;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        std::lock_guard<std::mutex> lock(resources.m);
        if (resources.uploadedBuildCounter == ws.buildCounter && ws.buildCounter != 0) {
            return true;
        }

        if (ws.buildCounter == 0) {
            outError = "WorkingState buildCounter is 0";
            return false;
        }

        // Important: OFX CUDA renders are async (we enqueue work on the host stream). When rebuilding
        // GPU resources, ensure no in-flight work can still reference the previous device pointers.
        // For now we take the simple/robust approach and synchronize on the host-provided stream.
        // This is expected to be rare (only on WorkingState rebuilds), and can later be replaced by
        // cudaFreeAsync or stream callbacks when we begin heavy per-frame kernel work.
        if (resources.uploadedBuildCounter != ws.buildCounter) {
            if (resources.uploadedBuildCounter != 0) {
                const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
                const cudaError_t syncErr = cudaStreamSynchronize(stream);
                if (syncErr != cudaSuccess) {
                    outError = std::string("cudaStreamSynchronize before rebuild failed: ") + (cudaGetErrorString(syncErr) ? cudaGetErrorString(syncErr) : "(unknown)");
                    return false;
                }
            }

            // Clear any previously uploaded (or partially uploaded) curves before re-uploading.
            free_curve(resources.densB);
            free_curve(resources.densG);
            free_curve(resources.densR);
            free_curve(resources.sensB);
            free_curve(resources.sensG);
            free_curve(resources.sensR);

            resources.validatedBuildCounter = 0;
        }

        if (!alloc_and_upload_curve(resources.densB, ws.densB, cudaStreamOpaque, outError)) return false;
        if (!alloc_and_upload_curve(resources.densG, ws.densG, cudaStreamOpaque, outError)) return false;
        if (!alloc_and_upload_curve(resources.densR, ws.densR, cudaStreamOpaque, outError)) return false;

        if (!alloc_and_upload_curve(resources.sensB, ws.sensB, cudaStreamOpaque, outError)) return false;
        if (!alloc_and_upload_curve(resources.sensG, ws.sensG, cudaStreamOpaque, outError)) return false;
        if (!alloc_and_upload_curve(resources.sensR, ws.sensR, cudaStreamOpaque, outError)) return false;

        resources.uploadedBuildCounter = ws.buildCounter;
        return true;
#endif
    }

    bool validate_density_primitives(Resources& resources, const WorkingState& ws, void* cudaStreamOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)ws;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        std::lock_guard<std::mutex> lock(resources.m);
        if (resources.validatedBuildCounter == ws.buildCounter && ws.buildCounter != 0) {
            return true;
        }

        // Lightweight parity probe: sample density curves at a few canonical logE values and compare
        // against the CPU implementation (FilmProcessing.h). This is a development-only check and is
        // expected to be compiled out in shipping builds.
        static constexpr float kLogE[5] = { -3.0f, -1.0f, 0.0f, 2.0f, 4.0f };
        float outB[5] = {};
        float outG[5] = {};
        float outR[5] = {};

        cudaError_t err = ::juicer_cuda_probe_density_curve(resources.densB.x, resources.densB.y, resources.densB.n,
            ws.gammaFactorB, kLogE, 5, outB, cudaStreamOpaque);
        if (err != cudaSuccess) {
            outError = std::string("probe densB failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            return false;
        }
        err = ::juicer_cuda_probe_density_curve(resources.densG.x, resources.densG.y, resources.densG.n,
            ws.gammaFactorG, kLogE, 5, outG, cudaStreamOpaque);
        if (err != cudaSuccess) {
            outError = std::string("probe densG failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            return false;
        }
        err = ::juicer_cuda_probe_density_curve(resources.densR.x, resources.densR.y, resources.densR.n,
            ws.gammaFactorR, kLogE, 5, outR, cudaStreamOpaque);
        if (err != cudaSuccess) {
            outError = std::string("probe densR failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            return false;
        }

        float maxAbs = 0.0f;
        for (int i = 0; i < 5; ++i) {
            const float cpuB = Spectral::sample_density_at_logE(ws.densB, kLogE[i], ws.gammaFactorB);
            const float cpuG = Spectral::sample_density_at_logE(ws.densG, kLogE[i], ws.gammaFactorG);
            const float cpuR = Spectral::sample_density_at_logE(ws.densR, kLogE[i], ws.gammaFactorR);

            if (!std::isfinite(cpuB) || !std::isfinite(cpuG) || !std::isfinite(cpuR) ||
                !std::isfinite(outB[i]) || !std::isfinite(outG[i]) || !std::isfinite(outR[i])) {
                outError = "density probe produced non-finite values";
                return false;
            }

            maxAbs = std::max(maxAbs, std::fabs(outB[i] - cpuB));
            maxAbs = std::max(maxAbs, std::fabs(outG[i] - cpuG));
            maxAbs = std::max(maxAbs, std::fabs(outR[i] - cpuR));
        }

        // This primitive is expected to match exactly (same math, float-only) within a tiny epsilon.
        if (!(maxAbs <= 1e-6f)) {
            outError = "density probe mismatch: maxAbs=" + std::to_string(maxAbs);
            return false;
        }

        resources.validatedBuildCounter = ws.buildCounter;
        return true;
#endif
    }

} // namespace JuicerCuda
