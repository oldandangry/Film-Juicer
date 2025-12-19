// Cuda/JuicerCudaResources.h
//
// Phase 2: per-instance CUDA resource cache keyed by WorkingState.buildCounter.
//
// This module intentionally owns only GPU-side mirrors of CPU WorkingState data (curves/tables/etc).
// The render path remains responsible for gating unsupported features (e.g. auto-exposure) until
// they are ported to CUDA.
//
#pragma once

#include <cstdint>
#include <mutex>
#include <string>

struct WorkingState;

namespace JuicerCuda {

    struct DeviceCurve {
        float* x = nullptr;
        float* y = nullptr;
        int n = 0;
    };

    struct Resources {
        std::mutex m;
        int deviceId = -1;
        std::uint64_t uploadedBuildCounter = 0;
        std::uint64_t validatedBuildCounter = 0;

        // Opaque CUDA event (cudaEvent_t) recorded on the stream after enqueuing work that
        // uses this resource set. Used to safely retire/rebuild buffers across streams.
        void* lastUseEventOpaque = nullptr;

        DeviceCurve densB;
        DeviceCurve densG;
        DeviceCurve densR;

        DeviceCurve dirDensB;
        DeviceCurve dirDensG;
        DeviceCurve dirDensR;

        DeviceCurve sensB;
        DeviceCurve sensG;
        DeviceCurve sensR;

        // SPD reconstruction (Mallett 2019 basis via per-instance tables + S_inv).
        float* tablesAx = nullptr;
        float* tablesAy = nullptr;
        float* tablesAz = nullptr;
        int tablesK = 0;
        float spdSInv[9] = { 1,0,0, 0,1,0, 0,0,1 };
        float refIllumWhiteXYZ[3] = { 0.950455f, 1.0f, 1.089058f };

        struct DeviceSpectralTables {
            float* epsC = nullptr;
            float* epsM = nullptr;
            float* epsY = nullptr;
            float* Ax = nullptr;
            float* Ay = nullptr;
            float* Az = nullptr;
            float* baseMin = nullptr;
            int K = 0;
            int hasBaseline = 0;
            float invYn = 1.0f;
        };

        struct DeviceScanMedium {
            DeviceSpectralTables tables;
            int mediumIsNegative = 1;
            float min_cmy[3] = { 0.0f, 0.0f, 0.0f };
            float inv_max_cmy[3] = { 1.0f, 1.0f, 1.0f };
        };

        DeviceScanMedium scanNegative;
        DeviceScanMedium scanPrint;

        struct DeviceSpectralLut {
            double* logXYZ = nullptr; // layout: ((z*res + y)*res + x) * 3 + c
            std::uint32_t res = 0;
            std::uint64_t hash = 0;
        };

        DeviceSpectralLut scanNegativeLut;
        DeviceSpectralLut scanPrintLut;

        struct DeviceGaussianKernel {
            float* weights = nullptr;
            int radius = 0;
            float sigma = 0.0f;
        };

        struct DeviceOpticsScratch {
            float* rgbR = nullptr;
            float* rgbG = nullptr;
            float* rgbB = nullptr;
            float* tmp = nullptr;
            float* blurred = nullptr;
            int width = 0;
            int height = 0;
        };

        DeviceGaussianKernel scannerLensBlurKernel;
        DeviceGaussianKernel scannerUnsharpKernel;
        DeviceGaussianKernel scannerGlareKernel;
        DeviceOpticsScratch scannerScratch;

        // Hanatos LUT (process-global on CPU, uploaded on demand).
        // Layout matches NpySpectraLUT: ((x*N + y) * K + k), K=81.
        float* hanatosLut = nullptr;
        int hanatosN = 0;

        Resources() = default;
        Resources(const Resources&) = delete;
        Resources& operator=(const Resources&) = delete;

        ~Resources();
    };

    Resources* create() noexcept;
    void destroy(Resources* resources) noexcept;

    // Uploads WorkingState curves when buildCounter changes; returns false on failure with outError filled.
    bool ensure_uploaded(Resources& resources, const WorkingState& ws, void* cudaStreamOpaque, std::string& outError);

    // Builds + uploads scan-stage logXYZ LUT on demand (Mitchell cubic sampler parity with CPU).
    bool ensure_scan_lut(Resources& resources, const WorkingState& ws, bool negativeMedium, void* cudaStreamOpaque, std::string& outError);

    // Allocates scratch buffers used by scanner optics (blur/unsharp/glare) when active.
    bool ensure_optics_scratch(Resources& resources, int width, int height, bool needUnsharpScratch, void* cudaStreamOpaque, std::string& outError);

    // Builds and uploads a SciPy-compatible Gaussian kernel (truncate=4.0, radius clamp=75) for optics.
    bool ensure_gaussian_kernel(Resources& resources, Resources::DeviceGaussianKernel& kernel, float sigma, void* cudaStreamOpaque, std::string& outError);

    // Optional debug validation of primitives (kept here to avoid a separate JUICER_TESTS harness).
    bool validate_density_primitives(Resources& resources, const WorkingState& ws, void* cudaStreamOpaque, std::string& outError);

    // Records a "last use" event on the given stream to allow safe rebuilds without global sync.
    void record_use(Resources& resources, void* cudaStreamOpaque) noexcept;

} // namespace JuicerCuda
