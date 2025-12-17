// Cuda/JuicerCudaResources.cpp
//
// Phase 2: WorkingState uploads + primitive validation hooks.
//
#include "Cuda/JuicerCudaResources.h"

#include "FilmProcessing.h"
#include "ColorTransforms.h"
#include "SpectralProcessing.h"
#include "WorkingState.h"

#include "Logging.h"
#include "SpectralContext.h"

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
extern "C" cudaError_t juicer_cuda_probe_density_curve_sanitize_inf(
    const float* dX,
    const float* dY,
    int n,
    float gammaFactor,
    const float* hLogE,
    int m,
    float* hOut,
    void* cudaStreamOpaque);

// Implemented in Cuda/JuicerCudaPrimitives.cu
extern "C" cudaError_t juicer_cuda_probe_hanatos_layer_exposures(
    const float rgbDWG[3],
    const float* dHanatosLut,
    int hanatosN,
    const float refIllumWhiteXYZ[3],
    const float* dSensB,
    const float* dSensG,
    const float* dSensR,
    int K,
    float exposureScale,
    int applyDeltaLambda,
    float* outE3,
    void* cudaStreamOpaque);

// Implemented in Cuda/JuicerCudaPrimitives.cu
extern "C" cudaError_t juicer_cuda_probe_convert_input_to_DWG(
    const float rgbIn[3],
    int inputColorSpaceIndex,
    int applyCctfDecoding,
    int applyInputChromaticAdapt,
    const float inputRGBToXYZ9[9],
    const float inputXYZAdapt9[9],
    float outRgbDWG[3],
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

    static void free_hanatos(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (resources.hanatosLut) {
            cudaFree(resources.hanatosLut);
            resources.hanatosLut = nullptr;
        }
#endif
        resources.hanatosN = 0;
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
        free_hanatos(*this);
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

        // Upload Hanatos LUT if available (uploaded once and reused across WorkingState rebuilds).
        {
            Spectral::SpectralContext& ctx = Spectral::context();
            const bool hanatosAvailable = ctx.hanatosAvailable.load(std::memory_order_acquire);
            const int N = ctx.hanSpectra.size;
            const int K = ctx.hanSpectra.numSamples;
            const bool want = hanatosAvailable && N > 0 && K == Spectral::kNumSamples && !ctx.hanSpectra.data.empty();
            if (!want) {
                // If Hanatos becomes unavailable (asset missing/mismatch), drop the device copy.
                if (resources.hanatosLut) {
                    const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
                    const cudaError_t syncErr = cudaStreamSynchronize(stream);
                    if (syncErr != cudaSuccess) {
                        outError = std::string("cudaStreamSynchronize before dropping Hanatos failed: ") + (cudaGetErrorString(syncErr) ? cudaGetErrorString(syncErr) : "(unknown)");
                        return false;
                    }
                    free_hanatos(resources);
                }
            } else {
                if (!resources.hanatosLut || resources.hanatosN != N) {
                    const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
                    if (resources.hanatosLut) {
                        const cudaError_t syncErr = cudaStreamSynchronize(stream);
                        if (syncErr != cudaSuccess) {
                            outError = std::string("cudaStreamSynchronize before Hanatos reupload failed: ") + (cudaGetErrorString(syncErr) ? cudaGetErrorString(syncErr) : "(unknown)");
                            return false;
                        }
                        free_hanatos(resources);
                    }

                    const size_t count = static_cast<size_t>(N) * static_cast<size_t>(N) * static_cast<size_t>(K);
                    const size_t bytes = count * sizeof(float);
                    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.hanatosLut), bytes);
                    if (err != cudaSuccess) {
                        outError = std::string("cudaMalloc(Hanatos LUT) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                        free_hanatos(resources);
                        return false;
                    }
                    err = cudaMemcpyAsync(resources.hanatosLut, ctx.hanSpectra.data.data(), bytes, cudaMemcpyHostToDevice, stream);
                    if (err != cudaSuccess) {
                        outError = std::string("cudaMemcpyAsync(Hanatos LUT) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                        free_hanatos(resources);
                        return false;
                    }
                    resources.hanatosN = N;
                }
            }
        }

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

        // Validate +/-inf sanitization before sampling (DevelopFilmStage parity).
        {
            const float kLogEInf[3] = {
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                0.0f
            };
            float outBInf[3] = {};
            float outGInf[3] = {};
            float outRInf[3] = {};

            auto cpu_sanitize_inf = [](float logE, const Spectral::Curve& c) -> float {
                if (std::isfinite(logE) || std::isnan(logE)) {
                    return logE;
                }
                float xmin = 0.0f, xmax = 0.0f;
                const size_t n = c.lambda_nm.size();
                if (n == 0 || c.linear.size() != n) {
                    return logE;
                }
                size_t begin = 0;
                while (begin < n && !std::isfinite(c.lambda_nm[begin])) {
                    ++begin;
                }
                if (begin == n) {
                    return logE;
                }
                size_t end = n - 1;
                while (end > begin && !std::isfinite(c.lambda_nm[end])) {
                    --end;
                }
                xmin = c.lambda_nm[begin];
                xmax = c.lambda_nm[end];
                if (!std::isfinite(xmin) || !std::isfinite(xmax) || !(xmax >= xmin)) {
                    return logE;
                }
                return (logE > 0.0f) ? xmax : xmin;
            };

            cudaError_t err = ::juicer_cuda_probe_density_curve_sanitize_inf(resources.densB.x, resources.densB.y, resources.densB.n,
                ws.gammaFactorB, kLogEInf, 3, outBInf, cudaStreamOpaque);
            if (err != cudaSuccess) {
                outError = std::string("probe densB sanitize_inf failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                return false;
            }
            err = ::juicer_cuda_probe_density_curve_sanitize_inf(resources.densG.x, resources.densG.y, resources.densG.n,
                ws.gammaFactorG, kLogEInf, 3, outGInf, cudaStreamOpaque);
            if (err != cudaSuccess) {
                outError = std::string("probe densG sanitize_inf failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                return false;
            }
            err = ::juicer_cuda_probe_density_curve_sanitize_inf(resources.densR.x, resources.densR.y, resources.densR.n,
                ws.gammaFactorR, kLogEInf, 3, outRInf, cudaStreamOpaque);
            if (err != cudaSuccess) {
                outError = std::string("probe densR sanitize_inf failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                return false;
            }

            float maxDiff = 0.0f;
            for (int i = 0; i < 3; ++i) {
                const float cpuB = Spectral::sample_density_at_logE(ws.densB, cpu_sanitize_inf(kLogEInf[i], ws.densB), ws.gammaFactorB);
                const float cpuG = Spectral::sample_density_at_logE(ws.densG, cpu_sanitize_inf(kLogEInf[i], ws.densG), ws.gammaFactorG);
                const float cpuR = Spectral::sample_density_at_logE(ws.densR, cpu_sanitize_inf(kLogEInf[i], ws.densR), ws.gammaFactorR);
                maxDiff = std::max(maxDiff, std::fabs(outBInf[i] - cpuB));
                maxDiff = std::max(maxDiff, std::fabs(outGInf[i] - cpuG));
                maxDiff = std::max(maxDiff, std::fabs(outRInf[i] - cpuR));
            }
            if (!(maxDiff <= 1e-6f)) {
                outError = "sanitize_inf_logE mismatch: maxAbs=" + std::to_string(maxDiff);
                return false;
            }
        }

        // Validate input RGB -> DWG conversion against the CPU implementation.
        {
            const int inputColorSpaceIndex = Spectral::inputColorSpaceToIndex(ws.filmRaw.inputColorSpace);
            const int applyDecode = ws.filmRaw.applyCctfDecoding ? 1 : 0;
            const int applyAdapt = ws.filmRaw.applyInputChromaticAdapt ? 1 : 0;

            const float samplesIn[][3] = {
                { 0.184f, 0.184f, 0.184f },
                { 1.2f, -0.1f, 0.5f },   // includes negative input channel (must match CPU sanitize-only behavior)
                { 0.0f, 0.5f, 2.0f }
            };

            for (const auto& rgbIn : samplesIn) {
                float gpuDWG[3] = { 0.0f, 0.0f, 0.0f };
                const cudaError_t err = ::juicer_cuda_probe_convert_input_to_DWG(
                    rgbIn,
                    inputColorSpaceIndex,
                    applyDecode,
                    applyAdapt,
                    ws.filmRaw.inputRGBToXYZ.m,
                    ws.filmRaw.inputXYZAdapt.m,
                    gpuDWG,
                    cudaStreamOpaque);
                if (err != cudaSuccess) {
                    outError = std::string("convert_input_rgb_to_DWG probe failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                    return false;
                }

                float cpuDWG[3] = { 0.0f, 0.0f, 0.0f };
                Spectral::convert_input_rgb_to_DWG(ws.filmRaw, rgbIn, cpuDWG);

                float maxDiff = 0.0f;
                for (int c = 0; c < 3; ++c) {
                    maxDiff = std::max(maxDiff, std::fabs(gpuDWG[c] - cpuDWG[c]));
                }
                if (!(maxDiff <= 2e-5f)) {
                    outError = "convert_input_rgb_to_DWG mismatch: maxAbs=" + std::to_string(maxDiff);
                    return false;
                }
            }
        }

        // If Hanatos is available and uploaded, validate the SPD→exposure integration primitive on a
        // few representative DWG RGB samples.
        if (resources.hanatosLut && resources.hanatosN > 0) {
            const float refWhite[3] = {
                ws.tablesRef.refIllumWhiteXYZ[0],
                ws.tablesRef.refIllumWhiteXYZ[1],
                ws.tablesRef.refIllumWhiteXYZ[2]
            };
            const float samples[][3] = {
                { 0.184f, 0.184f, 0.184f }, // mid-gray
                { 0.9f, 0.1f, 0.1f },       // red-ish
                { 0.05f, 0.2f, 0.9f }       // blue-ish
            };

            for (const auto& rgbDWG : samples) {
                float gpuE[3] = { 0.0f, 0.0f, 0.0f };
                const cudaError_t err = ::juicer_cuda_probe_hanatos_layer_exposures(
                    rgbDWG,
                    resources.hanatosLut,
                    resources.hanatosN,
                    refWhite,
                    resources.sensB.y,
                    resources.sensG.y,
                    resources.sensR.y,
                    Spectral::kNumSamples,
                    1.0f,
                    0,
                    gpuE,
                    cudaStreamOpaque);
                if (err != cudaSuccess) {
                    outError = std::string("Hanatos exposure probe failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                    return false;
                }

                // CPU reference: match Spectral::rgbDWG_to_layerExposures_from_tables_with_curves
                // semantics for Hanatos path (applyDeltaLambda = false).
                std::vector<float> Ee;
                Spectral::reconstruct_Ee_from_DWG_RGB_hanatos(rgbDWG, Ee, refWhite);
                float cpuE[3] = { 0.0f, 0.0f, 0.0f };
                Spectral::layerExposures_from_sceneSPD_with_curves(Ee, ws.sensB, ws.sensG, ws.sensR, cpuE, 1.0f, false);

                float maxDiff = 0.0f;
                for (int c = 0; c < 3; ++c) {
                    maxDiff = std::max(maxDiff, std::fabs(gpuE[c] - cpuE[c]));
                }
                if (!(maxDiff <= 2e-4f)) {
                    outError = "Hanatos exposure mismatch: maxAbs=" + std::to_string(maxDiff);
                    return false;
                }
            }
        }

        resources.validatedBuildCounter = ws.buildCounter;
        return true;
#endif
    }

} // namespace JuicerCuda
