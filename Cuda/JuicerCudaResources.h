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
        std::uint64_t uploadedBuildCounter = 0;
        std::uint64_t validatedBuildCounter = 0;

        DeviceCurve densB;
        DeviceCurve densG;
        DeviceCurve densR;

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

    // Optional debug validation of primitives (kept here to avoid a separate JUICER_TESTS harness).
    bool validate_density_primitives(Resources& resources, const WorkingState& ws, void* cudaStreamOpaque, std::string& outError);

} // namespace JuicerCuda
