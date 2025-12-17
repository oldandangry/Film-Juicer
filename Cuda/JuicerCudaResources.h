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
